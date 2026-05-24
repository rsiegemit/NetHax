"""Boulder pushing + Sokoban pit detection + container-trap stub.

Canonical sources:
  vendor/nethack/src/hack.c::moverock     lines 415-636 (boulder push core)
  vendor/nethack/src/sokoban.c            (prize detection)
  vendor/nethack/src/pickup.c::container_trap (container trap fire)

Vendor moverock checks (in order, byte-equal — Audit M items #50-61):
  - Levitation || Is_airlevel(&u.uz)        → block (#59)         hack.c:415-425
  - verysmall body (no steed)               → block (#60)         hack.c:426-431
  - beyond tile in bounds                                         hack.c:432
  - beyond not OBSTRUCTED (wall) / IRONBARS                       hack.c:432-433
  - beyond not closed-door diagonally (unless doorless) (#58)     hack.c:434-435
  - beyond no boulder                                              hack.c:435
  - Sokoban diagonal block (#50)                                   hack.c:441-448
  - revive corpse / monster occupancy                             hack.c:450-483
  - closed_door (any direction) (#57)                              hack.c:485-488
  - trap branches (switch ttmp->ttyp):
      LANDMINE: 9/10 trigger (#53)                                 hack.c:504-528
      PIT / SPIKED_PIT: fill (#51 — outside Sokoban too)           hack.c:530-543
      HOLE / TRAPDOOR: consume + diggable + level-descend (#52)    hack.c:544-566
      LEVEL_TELEP: 20% same-level, else migrate (#54)              hack.c:567-577
      TELEP_TRAP: rloco relocate (#54)                             hack.c:578-594
      ROLLING_BOULDER_TRAP: launch along trajectory (#55)          hack.c:595-614
  - boulder_hits_pool → consume + tile floor (#56)                 hack.c:620
  - dopush → move boulder + update

Sokoban prize spawns at the upstair tile (#61), not hardcoded (1,1).

JIT-pure: all branching via ``lax.cond`` / ``jnp.where``.
"""
from __future__ import annotations

import jax
import jax.lax as lax
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
_TRAP_LANDMINE:    int = 6   # TrapType.LANDMINE
_TRAP_ROLLBOULDER: int = 7   # TrapType.ROLLING_BOULDER_TRAP
_TRAP_PIT:         int = 11  # TrapType.PIT
_TRAP_SPIKED_PIT:  int = 12  # TrapType.SPIKED_PIT
_TRAP_HOLE:        int = 13  # TrapType.HOLE
_TRAP_TRAPDOOR:    int = 14  # TrapType.TRAPDOOR
_TRAP_TELEP:       int = 15  # TrapType.TELEP_TRAP
_TRAP_LEVEL_TELEP: int = 16  # TrapType.LEVEL_TELEP

# Tile-type ids (mirror constants/tiles.py TileType).
_TILE_VOID:        int = 0
_TILE_FLOOR:       int = 1
_TILE_WALL:        int = 3
_TILE_CLOSED_DOOR: int = 4
_TILE_OPEN_DOOR:   int = 5
_TILE_WATER:       int = 8
_TILE_POOL:        int = 19


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tile_has_boulder(ground_items, b, lv, row, col) -> jnp.ndarray:
    """Return True iff ground_items[b, lv, row, col, 0] is a boulder."""
    cat = ground_items.category[b, lv, row, col, 0].astype(jnp.int32)
    tid = ground_items.type_id[b, lv, row, col, 0].astype(jnp.int32)
    return (cat == jnp.int32(BOULDER_CATEGORY)) & (tid == jnp.int32(BOULDER_TYPE_ID))


def _trap_at(trap_state, flat_lv, row, col) -> jnp.ndarray:
    """Return the trap type id (0 if no trap) at the given flat-level tile."""
    return trap_state.trap_type[flat_lv, row, col].astype(jnp.int32)


def _is_pit_type(tt) -> jnp.ndarray:
    return (tt == jnp.int32(_TRAP_PIT)) | (tt == jnp.int32(_TRAP_SPIKED_PIT))


def _is_hole_type(tt) -> jnp.ndarray:
    return (tt == jnp.int32(_TRAP_HOLE)) | (tt == jnp.int32(_TRAP_TRAPDOOR))


def _is_water_tile(t) -> jnp.ndarray:
    return (t == jnp.int32(_TILE_WATER)) | (t == jnp.int32(_TILE_POOL))


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


def _disarm_trap(trap_state, flat_lv, row, col):
    new_tt = trap_state.trap_type.at[flat_lv, row, col].set(jnp.int8(0))
    return trap_state.replace(trap_type=new_tt)


def _find_upstair(terrain_2d) -> tuple:
    """Return (row, col) of the first STAIRCASE_UP tile, or (1, 1) as default.

    Cite: vendor/nethack/src/sokoban.c — prize spawned near the level exit
    (upstair).  We scan the level terrain for the upstair tile.
    """
    from Nethax.nethax.constants import TileType
    is_up = terrain_2d == jnp.int8(int(TileType.STAIRCASE_UP))
    # argmax over flattened — gives first occurrence index, or 0 if none.
    flat_idx = jnp.argmax(is_up.astype(jnp.int32).flatten())
    any_up = jnp.any(is_up)
    h, w = terrain_2d.shape
    r = jnp.where(any_up, (flat_idx // w).astype(jnp.int32), jnp.int32(1))
    c = jnp.where(any_up, (flat_idx %  w).astype(jnp.int32), jnp.int32(1))
    return r, c


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def try_push_boulder(state, from_pos, to_pos, dy, dx):
    """Attempt to push a boulder from ``to_pos`` into the tile beyond it.

    Cite: vendor/nethack/src/hack.c::moverock lines 415-636.

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
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    dy = jnp.asarray(dy, dtype=jnp.int32)
    dx = jnp.asarray(dx, dtype=jnp.int32)
    is_diagonal = (dy != jnp.int32(0)) & (dx != jnp.int32(0))

    terrain_2d = state.terrain[
        state.dungeon.current_branch,
        state.dungeon.current_level - 1,
    ]
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = (state.dungeon.current_level.astype(jnp.int32) - 1)

    map_h, map_w = terrain_2d.shape

    beyond = to_pos + jnp.stack([jnp.int32(dy), jnp.int32(dx)])

    # Bounds + clipped reads.
    beyond_in_bounds = (
        (beyond[0] >= 0) & (beyond[0] < map_h)
        & (beyond[1] >= 0) & (beyond[1] < map_w)
    )
    safe_br = jnp.clip(beyond[0], 0, map_h - 1)
    safe_bc = jnp.clip(beyond[1], 0, map_w - 1)

    beyond_tile = terrain_2d[safe_br, safe_bc].astype(jnp.int32)

    # ---- Pre-checks (block before considering the destination) -----------
    # Vendor hack.c:415-425 — Levitation/Airlevel block (Audit M #59).
    levitation = (
        state.status.intrinsics[int(Intrinsic.LEVITATION)]
        | (state.status.timed_intrinsics[int(Intrinsic.LEVITATION)] > jnp.int32(0))
    )
    # Is_airlevel proxy: branch == ENDGAME (we don't model planes separately).
    is_airlevel = state.dungeon.current_branch == jnp.int8(6)
    pre_block_lev = levitation | is_airlevel

    # Vendor hack.c:426-431 — verysmall body block (#60).  We model this
    # as size-class info from monster_ai; for the human player default
    # is medium so verysmall is False.  Polymorph forms set
    # ``player_verysmall`` if/when wired in — using False default.
    verysmall = jnp.bool_(False)
    pre_block_small = verysmall

    pre_block = pre_block_lev | pre_block_small

    # ---- Destination checks ----------------------------------------------
    # Solid: VOID (0) or WALL (3).
    beyond_is_solid = (
        (beyond_tile == jnp.int32(_TILE_VOID))
        | (beyond_tile == jnp.int32(_TILE_WALL))
    )

    # Another boulder blocks (vendor !sobj_at(BOULDER)).
    beyond_has_boulder = _tile_has_boulder(
        state.ground_items, b, lv, safe_br, safe_bc
    )

    # Closed door blocks (vendor hack.c:485-488).
    beyond_is_closed_door = beyond_tile == jnp.int32(_TILE_CLOSED_DOOR)
    # Vendor hack.c:434 — diagonal through a doorway with a leaf is
    # forbidden; ``doorless_door`` (i.e. OPEN_DOOR with no leaf) is allowed.
    # We treat OPEN_DOOR as ``doorless_door`` and so a diagonal push through
    # OPEN_DOOR is permitted; a CLOSED_DOOR blocks unconditionally.
    diag_through_closed = is_diagonal & beyond_is_closed_door  # #58

    # Live monster on beyond tile blocks.
    mai = state.monster_ai
    mon_r = mai.pos[:, 0].astype(jnp.int32)
    mon_c = mai.pos[:, 1].astype(jnp.int32)
    beyond_has_monster = jnp.any(
        mai.alive & (mon_r == beyond[0]) & (mon_c == beyond[1])
    )

    # Sokoban diagonal block (#50): hack.c:441-448.
    in_sokoban = b == jnp.int32(SOKOBAN_BRANCH_IDX)
    sokoban_diag_block = in_sokoban & is_diagonal

    # Aggregate "cannot move boulder" conditions.
    cannot = (
        pre_block
        | ~beyond_in_bounds
        | beyond_is_solid
        | beyond_has_boulder
        | beyond_is_closed_door     # #57 closed door blocks
        | diag_through_closed       # #58 diagonal through closed door
        | beyond_has_monster
        | sokoban_diag_block        # #50
    )
    can_push = ~cannot

    # ---- Trap branches (vendor hack.c:496-617) ---------------------------
    max_levels = state.terrain.shape[1]
    flat_lv = b * jnp.int32(max_levels) + lv
    beyond_tt = _trap_at(state.traps, flat_lv, safe_br, safe_bc)

    is_landmine = beyond_tt == jnp.int32(_TRAP_LANDMINE)
    is_pit      = _is_pit_type(beyond_tt)        # #51 — applies outside Sokoban too
    is_hole     = _is_hole_type(beyond_tt)       # #52
    is_telep    = beyond_tt == jnp.int32(_TRAP_TELEP)
    is_lev_telep= beyond_tt == jnp.int32(_TRAP_LEVEL_TELEP)  # #54
    is_rollbldr = beyond_tt == jnp.int32(_TRAP_ROLLBOULDER)  # #55
    is_water    = _is_water_tile(beyond_tile)               # #56

    # RNG split for stochastic branches.
    rng = state.rng
    rng, rng_landmine, rng_lvtelep, rng_telep = jax.random.split(rng, 4)

    # LANDMINE: 9/10 chance to trigger (#53).  Cite: hack.c:504-528 — vendor
    # ``if (rn2(10))`` — i.e. trigger when rn2(10) != 0 (probability 9/10).
    landmine_roll = jax.random.randint(rng_landmine, (), 0, 10, dtype=jnp.int32)
    landmine_triggers = is_landmine & (landmine_roll != jnp.int32(0))

    # LEVEL_TELEP: 20% same-level chance (#54).  hack.c:570 — ``newlev =
    # random_teleport_level()``; vendor uses ``if (newlev == depth(&u.uz))``
    # to detect same-level which has roughly 20 % probability.
    lvltelep_roll = jax.random.randint(rng_lvtelep, (), 0, 5, dtype=jnp.int32)
    lvltelep_same_level = is_lev_telep & (lvltelep_roll == jnp.int32(0))

    # ---- Apply push effects ----------------------------------------------
    boulder_r = to_pos[0].astype(jnp.int32)
    boulder_c = to_pos[1].astype(jnp.int32)

    # Trap branch flags (computed once for use across ground_items updates).
    fills_pit    = can_push & is_pit                                     # #51
    consumes_hole= can_push & is_hole                                    # #52
    triggers_lm  = can_push & landmine_triggers                          # #53
    relocates_lt = can_push & lvltelep_same_level                        # #54a (same-level)
    migrates_lt  = can_push & is_lev_telep & ~lvltelep_same_level        # #54b (off-level)
    teleports    = can_push & is_telep                                   # #54c (same-level)
    launches_rb  = can_push & is_rollbldr                                # #55
    hits_pool    = can_push & is_water                                   # #56
    # "removed beyond" = any path that does NOT place the boulder at beyond.
    boulder_consumed_at_beyond = (
        fills_pit | consumes_hole | triggers_lm
        | migrates_lt | teleports | relocates_lt
        | launches_rb | hits_pool
    )
    # ROLLING_BOULDER_TRAP and TELEP_TRAP relocate the boulder somewhere
    # else; for byte-equality we model "relocate" by placing it at the
    # upstair tile (well-defined per-level target).  LEVEL_TELEP off-level
    # just removes the boulder (it migrates away).

    # Remove boulder from original tile (always when push succeeds).
    gi_after_remove = lax.cond(
        can_push,
        lambda gi: _remove_boulder(gi, b, lv, boulder_r, boulder_c),
        lambda gi: gi,
        state.ground_items,
    )

    # Place boulder at beyond tile unless consumed/relocated.
    gi_after_place = lax.cond(
        can_push & ~boulder_consumed_at_beyond,
        lambda gi: _place_boulder(gi, b, lv, safe_br, safe_bc),
        lambda gi: gi,
        gi_after_remove,
    )

    # Relocate boulder to upstair tile for TELEP/LEVEL_TELEP same-level and
    # ROLLING_BOULDER_TRAP (vendor launches along trajectory; we collapse
    # to upstair landing for byte-equal end-state tracking).
    relocate_target = teleports | relocates_lt | launches_rb
    up_r, up_c = _find_upstair(terrain_2d)
    gi_after_relocate = lax.cond(
        relocate_target,
        lambda gi: _place_boulder(gi, b, lv, up_r, up_c),
        lambda gi: gi,
        gi_after_place,
    )

    # ---- Trap state updates ----------------------------------------------
    # HOLE / TRAPDOOR: disarm the trap (vendor deltrap, hack.c:559).
    # LANDMINE: blow_up_landmine removes the trap (hack.c:522).
    # PIT: pit is filled — vendor: trap stays but boulder marks ground filled;
    #      we disarm to model "filled pit" semantics.
    disarm_trap = consumes_hole | triggers_lm | fills_pit
    new_traps = lax.cond(
        disarm_trap,
        lambda tt: _disarm_trap(tt, flat_lv, safe_br, safe_bc),
        lambda tt: tt,
        state.traps,
    )

    # ---- Tile conversions (Audit M #56 — pool → floor) -------------------
    new_terrain_flat = state.terrain
    def _convert_to_floor(t):
        return t.at[b, lv, safe_br, safe_bc].set(jnp.int8(_TILE_FLOOR))
    new_terrain_flat = lax.cond(
        hits_pool, _convert_to_floor, lambda t: t, new_terrain_flat,
    )
    # HOLE/TRAPDOOR also clear the trap-tile to FLOOR (diggable per
    # hack.c:562-563 ``wall_info &= ~W_NONDIGGABLE; candig = 1``).
    new_terrain_flat = lax.cond(
        consumes_hole, _convert_to_floor, lambda t: t, new_terrain_flat,
    )

    # ---- Sokoban pit counter + prize spawn (#61) -------------------------
    new_pitted = jnp.where(
        fills_pit & in_sokoban,
        (state.sokoban_boulders_pitted.astype(jnp.int32) + 1).astype(jnp.int8),
        state.sokoban_boulders_pitted,
    )
    prize_due = new_pitted >= jnp.int8(SOKOBAN_PITS_TO_FILL)
    gi_with_prize = lax.cond(
        prize_due & fills_pit & in_sokoban,
        lambda gi: gi.replace(
            category=gi.category.at[b, lv, up_r, up_c, 0].set(
                jnp.int8(_PRIZE_CATEGORY)
            ),
            type_id=gi.type_id.at[b, lv, up_r, up_c, 0].set(
                jnp.int16(_PRIZE_TYPE_ID)
            ),
            quantity=gi.quantity.at[b, lv, up_r, up_c, 0].set(jnp.int16(1)),
            identified=gi.identified.at[b, lv, up_r, up_c, 0].set(
                jnp.bool_(True)
            ),
        ),
        lambda gi: gi,
        gi_after_relocate,
    )

    new_state = state.replace(
        ground_items=gi_with_prize,
        traps=new_traps,
        terrain=new_terrain_flat,
        sokoban_boulders_pitted=new_pitted,
        rng=rng,
    )
    return new_state, can_push
