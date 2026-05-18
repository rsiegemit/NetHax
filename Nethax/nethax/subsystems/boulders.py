"""Boulder pushing + Sokoban pit detection + container-trap stub.

Canonical sources:
  vendor/nethack/src/hack.c::moverock (boulder push + pit fill)
  vendor/nethack/src/sokoban.c        (prize detection)
  vendor/nethack/src/pickup.c::container_trap (container trap fire)

Design
------
Boulders are represented as ground items with::

    category == ItemCategory.ROCK  (14)
    type_id  == BOULDER_TYPE_ID    (defined below)

``try_push_boulder`` is JIT-pure: it takes explicit branch/level
indices and the current ground_items pytree so it can be called from
inside jax.lax.cond without tracing the full state.

Sokoban detection
-----------------
Cite: vendor/nethack/src/sokoban.c::sokoban_in_play / sokoban_prize.
When a boulder is pushed into a pit (trap type PIT or HOLE) the pit is
"filled" — the trap is disarmed and the boulder removed.  After enough
fills the prize (bag of holding or amulet of reflection) spawns at the
level exit tile.  We track this with ``state.sokoban_boulders_pitted``
(int8 field added to EnvState).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Boulder ground-item identity.
# category = ItemCategory.ROCK (14 in inventory.py)
BOULDER_CATEGORY: int = 14   # ItemCategory.ROCK
BOULDER_TYPE_ID:  int = 0    # sub-type 0 = generic boulder

# Sokoban: number of pits that must be filled to earn the prize.
# Cite: sokoban.c — 4 pits per level on the standard Sokoban maps.
SOKOBAN_PITS_TO_FILL: int = 4

# Prize item constants (spawned at exit after all pits filled).
# category = TOOL (6), type_id = 0 = bag-of-holding stub.
# Cite: sokoban.c::sokoban_prize — random choice boh / amulet-of-reflection.
_PRIZE_CATEGORY: int = 6   # ItemCategory.TOOL
_PRIZE_TYPE_ID:  int = 0   # bag-of-holding placeholder

# Sokoban branch index (mirrors dungeon/branches.py Branch.SOKOBAN = 2).
SOKOBAN_BRANCH_IDX: int = 2

# Trap type ids for pit / hole (mirrors subsystems/traps.py TrapType).
_TRAP_PIT:  int = 11   # TrapType.PIT
_TRAP_HOLE: int = 13   # TrapType.HOLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tile_has_boulder(ground_items, b, lv, row, col) -> jnp.ndarray:
    """Return True iff ground_items[b, lv, row, col, 0] is a boulder."""
    cat = ground_items.category[b, lv, row, col, 0].astype(jnp.int32)
    tid = ground_items.type_id[b, lv, row, col, 0].astype(jnp.int32)
    return (cat == jnp.int32(BOULDER_CATEGORY)) & (tid == jnp.int32(BOULDER_TYPE_ID))


def _tile_has_pit(trap_state, flat_lv, row, col) -> jnp.ndarray:
    """Return True iff the tile has an active pit/hole trap.

    Cite: sokoban.c — boulders fill pits (dotrap PIT / HOLE branches).
    Uses traps.trap_type[flat_lv, row, col] which is 0 when no trap.
    """
    tt = trap_state.trap_type[flat_lv, row, col].astype(jnp.int32)
    return (tt == jnp.int32(_TRAP_PIT)) | (tt == jnp.int32(_TRAP_HOLE))


def _place_boulder(ground_items, b, lv, row, col):
    """Write a boulder into ground_items[b, lv, row, col, 0]."""
    gi = ground_items
    return gi.replace(
        category=gi.category.at[b, lv, row, col, 0].set(
            jnp.int8(BOULDER_CATEGORY)
        ),
        type_id=gi.type_id.at[b, lv, row, col, 0].set(
            jnp.int16(BOULDER_TYPE_ID)
        ),
        quantity=gi.quantity.at[b, lv, row, col, 0].set(jnp.int16(1)),
        weight=gi.weight.at[b, lv, row, col, 0].set(jnp.int32(1000)),
        identified=gi.identified.at[b, lv, row, col, 0].set(jnp.bool_(True)),
    )


def _remove_boulder(ground_items, b, lv, row, col):
    """Zero-out the boulder slot at ground_items[b, lv, row, col, 0]."""
    gi = ground_items
    return gi.replace(
        category=gi.category.at[b, lv, row, col, 0].set(jnp.int8(0)),
        type_id=gi.type_id.at[b, lv, row, col, 0].set(jnp.int16(0)),
        quantity=gi.quantity.at[b, lv, row, col, 0].set(jnp.int16(0)),
        weight=gi.weight.at[b, lv, row, col, 0].set(jnp.int32(0)),
        identified=gi.identified.at[b, lv, row, col, 0].set(jnp.bool_(False)),
    )


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def try_push_boulder(state, from_pos, to_pos, dy, dx):
    """Attempt to push a boulder from ``to_pos`` into the tile beyond it.

    Cite: vendor/nethack/src/hack.c::moverock (~line 200).

    ``moverock`` checks (in order):
      1. The "beyond" tile must be in bounds.
      2. The beyond tile must not be solid (wall) or contain another boulder.
      3. The beyond tile must not contain a live monster.
      If any check fails, push is blocked (player does not move either).

    Parameters
    ----------
    state    : EnvState
    from_pos : jnp.ndarray int32[2]  — player's current position (row, col)
    to_pos   : jnp.ndarray int32[2]  — boulder's current tile (row, col)
    dy, dx   : int or jnp.ndarray int32 — direction of push

    Returns
    -------
    (new_state, pushed : bool)
    """
    dy = jnp.asarray(dy, dtype=jnp.int32)
    dx = jnp.asarray(dx, dtype=jnp.int32)
    terrain_2d = state.terrain[
        state.dungeon.current_branch,
        state.dungeon.current_level - 1,
    ]
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = (state.dungeon.current_level.astype(jnp.int32) - 1)

    map_h, map_w = terrain_2d.shape

    # Beyond tile = one step further in the same direction.
    beyond = to_pos + jnp.stack([jnp.int32(dy), jnp.int32(dx)])

    # ---- hack.c moverock checks ----
    beyond_in_bounds = (
        (beyond[0] >= 0) & (beyond[0] < map_h)
        & (beyond[1] >= 0) & (beyond[1] < map_w)
    )
    safe_br = jnp.clip(beyond[0], 0, map_h - 1)
    safe_bc = jnp.clip(beyond[1], 0, map_w - 1)

    beyond_tile = terrain_2d[safe_br, safe_bc].astype(jnp.int32)

    # Solid check: VOID (0) or WALL (3) block.
    beyond_is_solid = (beyond_tile == jnp.int32(0)) | (beyond_tile == jnp.int32(3))

    # Another boulder in beyond tile blocks.
    beyond_has_boulder = _tile_has_boulder(
        state.ground_items, b, lv, safe_br, safe_bc
    )

    # Live monster in beyond tile blocks (hack.c moverock monster_at check).
    mai = state.monster_ai
    mon_r = mai.pos[:, 0].astype(jnp.int32)
    mon_c = mai.pos[:, 1].astype(jnp.int32)
    beyond_has_monster = jnp.any(
        mai.alive & (mon_r == beyond[0]) & (mon_c == beyond[1])
    )

    can_push = (
        beyond_in_bounds
        & ~beyond_is_solid
        & ~beyond_has_boulder
        & ~beyond_has_monster
    )

    # ---- Sokoban: detect pit fill ----
    # flat_lv for TrapState (shaped [N_BRANCHES * MAX_LEVELS, MAP_H, MAP_W]).
    max_levels = state.terrain.shape[1]
    flat_lv = b * jnp.int32(max_levels) + lv

    beyond_is_pit = _tile_has_pit(state.traps, flat_lv, safe_br, safe_bc)
    in_sokoban = state.dungeon.current_branch.astype(jnp.int32) == jnp.int32(
        SOKOBAN_BRANCH_IDX
    )
    fills_pit = can_push & in_sokoban & beyond_is_pit

    # ---- Apply push ----
    # Move boulder: remove from to_pos, place at beyond (unless pit).
    boulder_r = to_pos[0].astype(jnp.int32)
    boulder_c = to_pos[1].astype(jnp.int32)

    # Remove boulder from original tile.
    gi_after_remove = jax.lax.cond(
        can_push,
        lambda gi: _remove_boulder(gi, b, lv, boulder_r, boulder_c),
        lambda gi: gi,
        state.ground_items,
    )

    # Place boulder at beyond tile only when NOT filling a pit.
    gi_after_place = jax.lax.cond(
        can_push & ~fills_pit,
        lambda gi: _place_boulder(gi, b, lv, safe_br, safe_bc),
        lambda gi: gi,
        gi_after_remove,
    )

    # Disarm pit trap when filling it (sokoban.c — pit becomes floor).
    new_trap_type = jax.lax.cond(
        fills_pit,
        lambda tt: tt.at[flat_lv, safe_br, safe_bc].set(jnp.int8(0)),
        lambda tt: tt,
        state.traps.trap_type,
    )
    new_traps = state.traps.replace(trap_type=new_trap_type)

    # Increment sokoban_boulders_pitted counter.
    new_pitted = jnp.where(
        fills_pit,
        (state.sokoban_boulders_pitted.astype(jnp.int32) + 1).astype(jnp.int8),
        state.sokoban_boulders_pitted,
    )

    # Spawn prize when pitted count reaches threshold.
    # Place prize at (1, 1) as a stand-in for the level-exit tile.
    # Cite: sokoban.c::sokoban_prize — spawn at level end.
    prize_due = new_pitted >= jnp.int8(SOKOBAN_PITS_TO_FILL)
    gi_with_prize = jax.lax.cond(
        prize_due & fills_pit,
        lambda gi: gi.replace(
            category=gi.category.at[b, lv, 1, 1, 0].set(jnp.int8(_PRIZE_CATEGORY)),
            type_id=gi.type_id.at[b, lv, 1, 1, 0].set(jnp.int16(_PRIZE_TYPE_ID)),
            quantity=gi.quantity.at[b, lv, 1, 1, 0].set(jnp.int16(1)),
            identified=gi.identified.at[b, lv, 1, 1, 0].set(jnp.bool_(True)),
        ),
        lambda gi: gi,
        gi_after_place,
    )

    new_state = state.replace(
        ground_items=gi_with_prize,
        traps=new_traps,
        sokoban_boulders_pitted=new_pitted,
    )
    return new_state, can_push
