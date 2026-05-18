"""Digging subsystem — multi-turn pickaxe/mattock dig mechanic.

Canonical source: vendor/nethack/src/dig.c

Design
------
``DigState`` tracks an in-progress dig.  Each turn ``dig_tick`` is called from
``_step_impl``; when accumulated effort reaches the threshold the target tile is
converted to CORRIDOR (horizontal dig) or a HOLE is created and the player
descends one level (downward dig).

Direction encoding (matches ``_DIR_TABLE`` in action_dispatch.py):
    0=N  1=E  2=S  3=W  4=NE  5=SE  6=SW  7=NW  8=DOWN
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from Nethax.nethax.constants.tiles import TileType

# ---------------------------------------------------------------------------
# Pickaxe / mattock type IDs (vendor/nethack/include/objects.h).
# pick-axe  : type_id 234 (TOOL_CLASS)   — objects.py line 4864
# dwarvish mattock : type_id 50 (WEAPON_CLASS) — objects.py line 1184
# ---------------------------------------------------------------------------
PICKAXE_TYPE_ID: int = 234
MATTOCK_TYPE_ID: int = 50

# Direction constant for digging down.
DIG_DOWN: int = 8

# Effort table: maps TileType int-value to effort-needed (int32).
# Vendor dig.c: WALL/STONE/ROCK all require the same effort (~200 swing-points),
# BOULDER is lighter (100).  Floor and other passable tiles are not diggable (0).
# Cite: vendor/nethack/src/dig.c::dig_typ and dodig (line ~445).
_HARDNESS_TABLE: jnp.ndarray = jnp.array(
    [
        0,    # VOID        = 0
        0,    # FLOOR       = 1
        0,    # CORRIDOR    = 2
        200,  # WALL        = 3
        0,    # CLOSED_DOOR = 4
        0,    # OPEN_DOOR   = 5
        0,    # STAIRCASE_UP= 6
        0,    # STAIRCASE_DOWN=7
        0,    # WATER       = 8
        0,    # LAVA        = 9
        0,    # ALTAR       = 10
        0,    # FOUNTAIN    = 11
        0,    # TRAP        = 12
        0,    # HIDDEN_TRAP = 13
        0,    # THRONE      = 14
        0,    # GRAVE       = 15
        0,    # SHOP_FLOOR  = 16
        0,    # DRAWBRIDGE  = 17
        0,    # ICE_FLOOR   = 18
        0,    # POOL        = 19
        0,    # TREE        = 20
    ],
    dtype=jnp.int32,
)

# Direction delta table: dir_idx (0-7) -> (dy, dx).  8=DOWN has no delta.
_DIR_DY = jnp.array([-1,  0,  1,  0, -1,  1,  1, -1], dtype=jnp.int32)
_DIR_DX = jnp.array([ 0,  1,  0, -1,  1,  1, -1, -1], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# State struct
# ---------------------------------------------------------------------------

@struct.dataclass
class DigState:
    """Per-turn pickaxe dig state.

    Fields
    ------
    target_pos : int16[2] — (row, col) of tile being dug; (-1, -1) if inactive.
    effort     : int32    — accumulated effort points this dig session.
    needed     : int32    — total effort to complete (terrain hardness).
    direction  : int8     — direction index (0-7 cardinal/intercardinal, 8=DOWN).
    active     : bool     — whether a dig is currently in progress.
    """

    target_pos: jnp.ndarray  # int16[2]
    effort: jnp.ndarray      # int32
    needed: jnp.ndarray      # int32
    direction: jnp.ndarray   # int8
    active: jnp.ndarray      # bool

    @classmethod
    def default(cls) -> "DigState":
        return cls(
            target_pos=jnp.full((2,), -1, dtype=jnp.int16),
            effort=jnp.int32(0),
            needed=jnp.int32(0),
            direction=jnp.int8(0),
            active=jnp.bool_(False),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _terrain_hardness(tile_type: jnp.ndarray) -> jnp.ndarray:
    """Return effort needed to dig through ``tile_type``.

    Returns 0 for non-diggable tiles.
    Cite: vendor/nethack/src/dig.c::dig_typ (line ~445).
    """
    idx = jnp.clip(tile_type.astype(jnp.int32), 0, len(_HARDNESS_TABLE) - 1)
    return _HARDNESS_TABLE[idx]


def _current_level_terrain(state):
    """Extract the 2-D terrain slice for the player's current branch/level."""
    b = state.dungeon.current_branch
    lv = state.dungeon.current_level - jnp.int8(1)  # 1-based → 0-based
    return state.terrain[b, lv]


def _wielded_type_id(state) -> jnp.ndarray:
    """Return the type_id of the wielded item (-1 if bare hands)."""
    wslot = state.inventory.wielded.astype(jnp.int32)
    # If wslot == -1 (bare hands) clamp to slot 0 and return 0 (no item).
    safe_slot = jnp.clip(wslot, 0, None)
    raw_type_id = state.inventory.items.type_id[safe_slot]
    return jnp.where(wslot >= 0, raw_type_id.astype(jnp.int32), jnp.int32(-1))


def _has_digging_tool(state) -> jnp.ndarray:
    """Return True if the wielded item is a pickaxe or mattock.

    Cite: vendor/nethack/src/dig.c line ~445 (wield-check before digging).
    """
    tid = _wielded_type_id(state)
    return (tid == jnp.int32(PICKAXE_TYPE_ID)) | (tid == jnp.int32(MATTOCK_TYPE_ID))


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def start_dig(state, direction: int):
    """Initiate a dig in ``direction``.

    Sets DigState.target_pos, direction, needed (hardness), resets effort=0,
    active=True.  Is a no-op when the player does not wield a pickaxe/mattock
    or when the target tile is not diggable.

    Cite: vendor/nethack/src/dig.c::dodig (line 445).

    Parameters
    ----------
    state     : EnvState
    direction : int — 0-7 compass direction or 8 for down.
    """
    dir_i8 = jnp.int8(direction)

    # --- pickaxe check ---
    has_tool = _has_digging_tool(state)

    # --- compute target tile ---
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)

    dir_idx = jnp.int32(direction)
    # For down-dig, target == player position (dig through the floor).
    is_down = dir_idx == jnp.int32(DIG_DOWN)

    dy = jnp.where(is_down, jnp.int32(0), _DIR_DY[jnp.clip(dir_idx, 0, 7)])
    dx = jnp.where(is_down, jnp.int32(0), _DIR_DX[jnp.clip(dir_idx, 0, 7)])

    trow = jnp.clip(row + dy, 0, state.terrain.shape[2] - 1)
    tcol = jnp.clip(col + dx, 0, state.terrain.shape[3] - 1)

    terrain_2d = _current_level_terrain(state)
    tile = terrain_2d[trow, tcol].astype(jnp.int32)

    # For horizontal: must be WALL/STONE/ROCK (hardness > 0).
    # For down: floor is always diggable (override hardness to 100).
    hardness_from_table = _terrain_hardness(terrain_2d[trow, tcol])
    hardness = jnp.where(is_down, jnp.int32(100), hardness_from_table)

    is_diggable = jnp.where(is_down, jnp.bool_(True), hardness > jnp.int32(0))

    can_start = has_tool & is_diggable

    new_dig = DigState(
        target_pos=jnp.where(
            can_start,
            jnp.array([trow, tcol], dtype=jnp.int16),
            state.dig.target_pos,
        ),
        effort=jnp.where(can_start, jnp.int32(0), state.dig.effort),
        needed=jnp.where(can_start, hardness, state.dig.needed),
        direction=jnp.where(can_start, dir_i8, state.dig.direction),
        active=jnp.where(can_start, jnp.bool_(True), state.dig.active),
    )
    return state.replace(dig=new_dig)


def _complete_dig(state):
    """Finalise a completed dig: modify terrain or descend.

    Horizontal: terrain[target] = CORRIDOR.
    Cite: vendor/nethack/src/dig.c line 830 (set levl[x][y].typ = CORR).

    Down: create HOLE at player pos and advance player one level.
    Cite: vendor/nethack/src/dig.c::digactualhole.
    """
    b = state.dungeon.current_branch
    lv = (state.dungeon.current_level - jnp.int8(1)).astype(jnp.int32)

    trow = state.dig.target_pos[0].astype(jnp.int32)
    tcol = state.dig.target_pos[1].astype(jnp.int32)

    is_down = state.dig.direction.astype(jnp.int32) == jnp.int32(DIG_DOWN)

    # Horizontal: carve the target tile into CORRIDOR.
    new_terrain_horiz = state.terrain.at[b, lv, trow, tcol].set(
        jnp.int8(TileType.CORRIDOR)
    )

    # Down: carve player tile into HOLE (use STAIRCASE_DOWN as proxy; no HOLE
    # tile type exists in this enum) and advance level by 1.
    prow = state.player_pos[0].astype(jnp.int32)
    pcol = state.player_pos[1].astype(jnp.int32)
    new_terrain_down = state.terrain.at[b, lv, prow, pcol].set(
        jnp.int8(TileType.STAIRCASE_DOWN)
    )

    new_terrain = jnp.where(is_down, new_terrain_down, new_terrain_horiz)

    max_level = jnp.int8(state.terrain.shape[1])
    new_level = jnp.where(
        is_down,
        jnp.minimum(max_level, state.dungeon.current_level + jnp.int8(1)),
        state.dungeon.current_level,
    )
    new_dungeon = state.dungeon.replace(current_level=new_level)

    return state.replace(terrain=new_terrain, dungeon=new_dungeon)


def dig_tick(state, rng):
    """Advance one turn of an in-progress dig.

    Called every turn from ``_step_impl``.  If ``state.dig.active``:
      - Increment effort by player_str // 2 (+ skill bonus if SkillState present).
      - If effort >= needed: call ``_complete_dig`` and reset DigState.
      - If player has moved away from target_pos (horizontal), cancel.

    Cite: vendor/nethack/src/dig.c::dig_check_down/digactualhole.
    """
    def _tick(s):
        # Effort gain: STR/2.  No SkillState in current wave; extend when wired.
        str_bonus = (s.player_str.astype(jnp.int32) // jnp.int32(2))
        effort_gain = jnp.maximum(str_bonus, jnp.int32(1))
        new_effort = s.dig.effort + effort_gain

        is_down = s.dig.direction.astype(jnp.int32) == jnp.int32(DIG_DOWN)

        # Cancel check: player moved away from target (horizontal digs only).
        prow = s.player_pos[0].astype(jnp.int32)
        pcol = s.player_pos[1].astype(jnp.int32)
        trow = s.dig.target_pos[0].astype(jnp.int32)
        tcol = s.dig.target_pos[1].astype(jnp.int32)

        # For horizontal dig, target is adjacent to starting player pos.
        # Cancel if player pos differs from where we expect them.
        # We store the expected player position as (target - delta), i.e.
        # player should still be adjacent.  For simplicity: cancel when the
        # L-inf distance from target > 1 (player walked away).
        dist = jnp.maximum(jnp.abs(prow - trow), jnp.abs(pcol - tcol))
        player_moved_away = (~is_down) & (dist > jnp.int32(1))

        # Complete?
        done = new_effort >= s.dig.needed

        # Build updated DigState for the "still digging" case.
        still_digging = DigState(
            target_pos=s.dig.target_pos,
            effort=new_effort,
            needed=s.dig.needed,
            direction=s.dig.direction,
            active=jnp.bool_(True),
        )
        reset_dig = DigState.default()

        # Apply completion.
        completed_state = _complete_dig(s.replace(dig=still_digging))
        completed_state = completed_state.replace(dig=reset_dig)

        # If cancelled, just reset dig without terrain change.
        cancelled_state = s.replace(dig=reset_dig)

        # In-progress: update effort.
        in_progress_state = s.replace(dig=still_digging)

        # Priority: cancel > complete > in-progress.
        s1 = jax.lax.cond(done, lambda _: completed_state, lambda _: in_progress_state, operand=None)
        s2 = jax.lax.cond(player_moved_away, lambda _: cancelled_state, lambda _: s1, operand=None)
        return s2

    return jax.lax.cond(state.dig.active, _tick, lambda s: s, operand=state)
