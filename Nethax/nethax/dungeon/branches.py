"""Dungeon branch graph: branch identifiers, topology, and stair linking.

Purpose:
    Defines the multi-branch structure of the NetHack dungeon.  Each Branch
    is a named sub-dungeon (e.g. Gnomish Mines, Sokoban) connected to the
    main dungeon via stairs or portals.  DungeonState tracks the player's
    position within this graph and the generated/linked level counts.

Citation:
    vendor/nethack/include/dungeon.h  — dungeon / d_level / branch structs,
        BR_STAIR / BR_PORTAL / BR_NO_END1 / BR_NO_END2 constants,
        MAXDUNGEON = 16, MAXLEVEL = 32
    vendor/nethack/include/global.h   — MAXDUNGEON = 16, MAXLEVEL = 32,
        MAXNROFROOMS = 40
    vendor/nethack/include/hack.h     — mines_dnum, sokoban_dnum, quest_dnum
    vendor/nethack/src/dungeon.c      — init_dungeon, branch linking logic,
        stronghold / oracle / sanctum / valley level assignment

Wave 2: traverse_stair reads stair_links table; enter_branch updates
        current_branch/current_level; generate_main_branch_l1 produces
        a fully built level (terrain + rooms + stair positions).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import struct

# ---------------------------------------------------------------------------
# Map geometry constants
# These are referenced by every other dungeon module to size JAX arrays.
# ---------------------------------------------------------------------------

MAP_H: int = 21   # canonical NLE / NetHack map height (rows)
MAP_W: int = 80   # canonical NLE / NetHack map width  (cols)

# ---------------------------------------------------------------------------
# Branch / dungeon-count constants
# ---------------------------------------------------------------------------

N_BRANCHES: int = 7
MAX_LEVELS_PER_BRANCH: int = 32  # MAXLEVEL from vendor/nethack/include/global.h


# ---------------------------------------------------------------------------
# Branch enum
# ---------------------------------------------------------------------------

class Branch(IntEnum):
    """Named dungeon branches.

    Ordinal values are used as indices into per-branch JAX arrays.

    Citation: vendor/nethack/include/dungeon.h, vendor/nethack/src/dungeon.c
    """
    MAIN          = 0
    GNOMISH_MINES = 1
    SOKOBAN       = 2
    QUEST         = 3
    VLAD          = 4
    GEHENNOM      = 5
    ENDGAME       = 6


# ---------------------------------------------------------------------------
# Branch connection type
# Mirrors BR_STAIR / BR_NO_END1 / BR_NO_END2 / BR_PORTAL from dungeon.h
# ---------------------------------------------------------------------------

class BranchConnectionType(IntEnum):
    """How a branch is entered from its parent dungeon.

    Citation: vendor/nethack/include/dungeon.h lines 91-96
        BR_STAIR   = 0  — two-way staircase pair
        BR_NO_END1 = 1  — stair only on parent side
        BR_NO_END2 = 2  — stair only on branch side
        BR_PORTAL  = 3  — magic portal (Quest, Sokoban level 1)
    """
    STAIR    = 0   # BR_STAIR
    NO_END1  = 1   # BR_NO_END1
    NO_END2  = 2   # BR_NO_END2
    PORTAL   = 3   # BR_PORTAL


# ---------------------------------------------------------------------------
# BranchInfo dataclass
# ---------------------------------------------------------------------------

@struct.dataclass
class BranchInfo:
    """Static topology for one branch.

    Fields
    ------
    branch_id : int8
    first_level : int8
    num_levels : int8
    connection_type : int8
    """
    branch_id:       jnp.ndarray  # int8
    first_level:     jnp.ndarray  # int8
    num_levels:      jnp.ndarray  # int8
    connection_type: jnp.ndarray  # int8


# ---------------------------------------------------------------------------
# Canonical branch table
# ---------------------------------------------------------------------------
#
# Audit-N #7 — vendor-faithful (mean, dev) specs landed incrementally.
# Vendor cite: vendor/nle/dat/dungeon.def lines 17-143.
#
# The static BRANCH_TABLE below remains for legacy callers (cross-branch
# cascade tests that hardcode integer level counts).  New code should
# prefer ``sample_branch_table(rng)`` which produces vendor-faithful samples
# via ``rn1(rand, base) = rn2(rand) + base`` (vendor/nle/include/hack.h line 497).
#
# Vendor specs landed in this wave (per dungeon.def):
#
#   MAIN          DUNGEON  "The Dungeons of Doom"      (25, 5)   # base, rand
#   MINES_ENTRY   BRANCH   "The Gnomish Mines"  @ (2, 3)
#   MINES         DUNGEON  "The Gnomish Mines"        (8, 2)
#   ORACLE_LEVEL  LEVEL    "oracle"             @ (5, 5)
#   SOKO_ENTRY    CHAINBR  "Sokoban" "oracle"   + (1, 0) up      # STAIR not PORTAL
#   SOKOBAN       DUNGEON  "Sokoban"                  (4, 0)
#   QUEST_ENTRY   CHAINBR  "The Quest" "oracle" + (6, 2) portal
#   QUEST         DUNGEON  "The Quest"                (5, 2)
#   LUDIOS_ENTRY  BRANCH   "Fort Ludios"        @ (18, 4) portal
#   LUDIOS        DUNGEON  "Fort Ludios"              (1, 0)
#   GEHENNOM      DUNGEON  "Gehennom"                 (20, 5)
#   VLAD_ENTRY    BRANCH   "Vlad's Tower"       @ (9, 5) up      # chained off Gehennom
#   VLAD          DUNGEON  "Vlad's Tower"             (3, 0)
#   ENDGAME       DUNGEON  "The Elemental Planes"     (6, 0)

BRANCH_TABLE: Tuple[BranchInfo, ...] = (
    BranchInfo(
        branch_id=jnp.int8(Branch.MAIN),
        first_level=jnp.int8(1),
        num_levels=jnp.int8(26),
        connection_type=jnp.int8(BranchConnectionType.STAIR),
    ),
    BranchInfo(
        branch_id=jnp.int8(Branch.GNOMISH_MINES),
        first_level=jnp.int8(2),
        num_levels=jnp.int8(5),
        connection_type=jnp.int8(BranchConnectionType.STAIR),
    ),
    BranchInfo(
        branch_id=jnp.int8(Branch.SOKOBAN),
        first_level=jnp.int8(8),
        num_levels=jnp.int8(4),
        connection_type=jnp.int8(BranchConnectionType.PORTAL),
    ),
    BranchInfo(
        branch_id=jnp.int8(Branch.QUEST),
        first_level=jnp.int8(14),
        num_levels=jnp.int8(5),
        connection_type=jnp.int8(BranchConnectionType.PORTAL),
    ),
    BranchInfo(
        branch_id=jnp.int8(Branch.VLAD),
        first_level=jnp.int8(21),
        num_levels=jnp.int8(3),
        connection_type=jnp.int8(BranchConnectionType.NO_END2),
    ),
    BranchInfo(
        branch_id=jnp.int8(Branch.GEHENNOM),
        first_level=jnp.int8(21),
        num_levels=jnp.int8(16),
        connection_type=jnp.int8(BranchConnectionType.STAIR),
    ),
    BranchInfo(
        branch_id=jnp.int8(Branch.ENDGAME),
        first_level=jnp.int8(27),
        num_levels=jnp.int8(5),
        connection_type=jnp.int8(BranchConnectionType.NO_END1),
    ),
)


# ---------------------------------------------------------------------------
# Vendor-faithful (mean, dev) tuples and JIT-safe sampler
# ---------------------------------------------------------------------------
#
# Audit-N #7 Commit 1 (MAIN): wire ``(base=25, rand=5)`` for the Dungeons of
# Doom.  Vendor cite: vendor/nle/dat/dungeon.def line 17
# ``DUNGEON: "The Dungeons of Doom" "D" (25, 5)`` and vendor sampling rule
# vendor/nle/src/dungeon.c lines 796-800
# ``num_dunlevs = (xchar) rn1(pd.tmpdungeon[i].lev.rand,
#                              pd.tmpdungeon[i].lev.base);``
# with ``rn1(x, y) = rn2(x) + y`` from vendor/nle/include/hack.h line 497.
#
# Each ``(base, rand)`` tuple describes a uniform sample on the closed
# interval ``[base, base + rand - 1]`` (when rand != 0) or the fixed
# constant ``base`` (when rand == 0).  Branch-entry (BRANCH/CHAINBRANCH)
# tuples are interpreted by ``level_range`` in dungeon.c lines 350-382:
# the entry depth lies in ``[parent_depth + base, parent_depth + base + rand - 1]``
# for CHAINBRANCH ("rcouple") or ``[base, base + rand - 1]`` for BRANCH
# ("acouple").
#
# This wave wires only the MAIN dungeon ``num_levels`` spec; subsequent
# commits will land Gehennom, Endgame, Mines, Sokoban, Quest, Vlad, Ludios.

# (base, rand) for ``num_levels`` per DUNGEON entry in dungeon.def.
# rand == 0 ⇒ fixed; rand > 0 ⇒ rn1(rand, base) = rn2(rand) + base.
_DUNGEON_NUM_LEVELS_VENDOR_SPEC: dict = {
    int(Branch.MAIN):          (25, 5),   # dungeon.def line 17
    int(Branch.GEHENNOM):      (20, 5),   # dungeon.def line 51
    int(Branch.GNOMISH_MINES): (8, 2),    # dungeon.def line 71
    int(Branch.ENDGAME):       (6, 0),    # dungeon.def line 134 (rand=0 ⇒ fixed)
    # Subsequent commits will populate:
    # int(Branch.QUEST):       (5, 2),   # dungeon.def line 86
    # int(Branch.SOKOBAN):     (4, 0),   # dungeon.def line 94
    # int(Branch.VLAD):        (3, 0),   # dungeon.def line 116
}


# (base, rand) for branch-entry depth (BRANCH / CHAINBRANCH @ (b, r)).
# Interpreted via vendor/nle/src/dungeon.c::level_range — BRANCH uses
# "acouple" semantics (absolute in parent dungeon), CHAINBRANCH uses
# "rcouple" (relative to a chain-level's dlevel).
#
# Wired in Commit 3: Mines entry @ (2, 3) into Main.  Future commits
# add Sokoban (chain to oracle), Quest (chain to oracle), Vlad (chain
# to Gehennom), Ludios (acouple in Main).
_BRANCH_ENTRY_VENDOR_SPEC: dict = {
    int(Branch.GNOMISH_MINES): (2, 3),    # dungeon.def line 19, acouple in Main
    # int(Branch.SOKOBAN):     (1, 0),   # dungeon.def line 24, rcouple oracle +1 up
    # int(Branch.QUEST):       (6, 2),   # dungeon.def line 26, rcouple oracle +6 portal
    # int(Branch.VLAD):        (9, 5),   # dungeon.def line 55, acouple in Gehennom
}


def _vendor_rn1(rng, rand: int, base: int) -> jnp.ndarray:
    """JAX-native vendor ``rn1`` — ``rn2(rand) + base``.

    Threefry/uniform sampling: when ``rand <= 1`` we return the constant
    ``base`` (no key consumption) to mirror vendor short-circuiting at
    dungeon.c line 379 ``if (randc) ...``.  When ``rand > 1`` we draw a
    single uniform int via jax.random.randint over ``[base, base+rand)``.

    Citation: vendor/nle/include/hack.h line 497
    ``#define rn1(x, y) (rn2(x) + (y))``.

    Args:
        rng:  jax.random.PRNGKey scalar.
        rand: vendor ``lev.rand`` (the dispersion).
        base: vendor ``lev.base`` (the floor).

    Returns:
        int8 scalar.
    """
    if rand <= 0:
        return jnp.int8(base)
    # rn2(rand) ∈ [0, rand-1]; rn1 = rn2(rand) + base ∈ [base, base+rand-1].
    sample = jax.random.randint(rng, (), minval=base, maxval=base + rand)
    return sample.astype(jnp.int8)


def sample_branch_table(rng) -> Tuple[BranchInfo, ...]:
    """Vendor-faithful sampler for ``BRANCH_TABLE``.

    Reproduces vendor/nle/src/dungeon.c::init_dungeons lines 796-800 by
    sampling ``num_dunlevs`` from the dungeon.def ``(base, rand)`` tuple
    for each branch that has a vendor spec landed.  Branches whose specs
    have not yet been wired fall back to the static ``BRANCH_TABLE``
    entries — this lets the sampler land incrementally (one branch per
    commit) without breaking cross-branch traversal tests.

    JIT-safety: this function consumes ``rng`` via ``jax.random.split``
    so no PRNG key is reused.  It is not itself JIT'd (it runs once per
    game at init), but the per-branch ``_vendor_rn1`` calls are.

    Citation: vendor/nle/src/dungeon.c::init_dungeons lines 796-800,
              vendor/nle/dat/dungeon.def lines 17-143.

    Args:
        rng: jax.random.PRNGKey scalar.

    Returns:
        Tuple of ``BranchInfo`` records, ordered by Branch enum.
    """
    # Split one key per branch, then split each branch-key into two sub-keys
    # (one for num_levels, one for first_level) so that landing a new branch
    # in a later commit doesn't reshuffle the samples drawn by branches that
    # are already wired.  We always split to keep the rng schedule
    # deterministic even when an individual branch falls back to the static
    # value.
    branch_keys = jax.random.split(rng, N_BRANCHES)

    out = []
    for b in range(N_BRANCHES):
        static = BRANCH_TABLE[b]
        k_nl, k_fl = jax.random.split(branch_keys[b], 2)

        # num_levels: sample from DUNGEON spec when present.
        nl_spec = _DUNGEON_NUM_LEVELS_VENDOR_SPEC.get(b)
        if nl_spec is not None:
            nl_base, nl_rand = nl_spec
            num_levels_sampled = _vendor_rn1(k_nl, nl_rand, nl_base)
        else:
            num_levels_sampled = static.num_levels

        # first_level: sample from BRANCH-entry spec when present.  Only
        # acouple (absolute-in-Main) entries are landed here; CHAINBRANCH
        # entries that resolve relative to a special level (oracle, valley)
        # are computed by the cross-branch chain logic.
        fl_spec = _BRANCH_ENTRY_VENDOR_SPEC.get(b)
        if fl_spec is not None:
            fl_base, fl_rand = fl_spec
            first_level_sampled = _vendor_rn1(k_fl, fl_rand, fl_base)
        else:
            first_level_sampled = static.first_level

        out.append(BranchInfo(
            branch_id=static.branch_id,
            first_level=first_level_sampled,
            num_levels=num_levels_sampled,
            connection_type=static.connection_type,
        ))
    return tuple(out)


# ---------------------------------------------------------------------------
# DungeonState dataclass
# ---------------------------------------------------------------------------

@struct.dataclass
class DungeonState:
    """Full multi-branch dungeon graph state.

    Fields
    ------
    branch_levels : int8[N_BRANCHES]
    current_branch : int8
    current_level : int8  (1-based)
    stair_links : int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2, 2]
        [branch, level, stair_dir, endpoint] where stair_dir 0=up, 1=down
        and endpoint 0=dest_branch, 1=dest_level.  -1 = unresolved.
    level_rng_seeds : uint32[N_BRANCHES, MAX_LEVELS_PER_BRANCH]
    vibrating_square_revealed : bool — Wave 5 Phase 2.  Set True once the
        player has stepped on the vibrating-square trap in the Valley of
        the Dead, causing the Gehennom magic portal to materialise.
        Citation: vendor/nethack/src/trap.c TRAP_VIBRATING_SQUARE case,
                  vendor/nethack/include/dungeon.h vibrating_square flag.
    """
    branch_levels:    jnp.ndarray  # int8[N_BRANCHES]
    current_branch:   jnp.ndarray  # int8 scalar
    current_level:    jnp.ndarray  # int8 scalar  (1-based)
    stair_links:      jnp.ndarray  # int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2, 2]
    level_rng_seeds:  jnp.ndarray  # uint32[N_BRANCHES, MAX_LEVELS_PER_BRANCH]
    vibrating_square_revealed: jnp.ndarray  # bool scalar
    # (row, col) of the vibrating-square tile once revealed; (-1,-1) = unset.
    # Citation: vendor/nethack/src/mklev.c magic_portal placement.
    vibrating_square_pos: jnp.ndarray  # int16[2]
    # Wave 6 #79: SPELL_LIGHT timer.  Holds the turn at which the lit-radius
    # effect expires (-1 = never active).
    # Cite: vendor/nethack/src/light.c::do_light_sources / read.c SCR_LIGHT.
    lit_radius_until_turn: jnp.ndarray  # scalar int32
    # Fixed portal destinations: int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2]
    # [branch, level-1] -> (dest_branch, dest_level); -1 = no portal / same level.
    # Citation: vendor/nethack/src/trap.c::dotrap MAGIC_PORTAL branch — each portal
    # links to a fixed (d_level) destination stored in trap.dst.
    portal_destination: jnp.ndarray  # int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2]


# ---------------------------------------------------------------------------
# Tile type constants (from constants.py TileType)
# ---------------------------------------------------------------------------

_TILE_FLOOR:         int = 1
_TILE_STAIRCASE_UP:  int = 6
_TILE_STAIRCASE_DOWN: int = 7


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def current_dungeon_level(state: DungeonState) -> Tuple[int, int]:
    """Return (branch, level) for the player's current position.

    Citation: vendor/nethack/src/dungeon.c current_dungeon_level().
    """
    return (int(state.current_branch), int(state.current_level))


def traverse_stair(
    state: DungeonState,
    direction: int,  # 0 = up, 1 = down
) -> DungeonState:
    """Return new DungeonState after taking a stair in given direction.

    Reads state.stair_links[current_branch, current_level-1, direction] to
    get the destination (dest_branch, dest_level).  If the link is -1
    (unresolved), the state is returned unchanged.

    Citation: vendor/nethack/src/dungeon.c (stair resolution logic).

    Args:
        state:     current DungeonState.
        direction: 0 = up-stair, 1 = down-stair.

    Returns:
        Updated DungeonState with current_branch/current_level set to
        destination; unchanged if link is unresolved (-1).
    """
    branch = state.current_branch.astype(jnp.int32)
    level  = state.current_level.astype(jnp.int32)
    dir_   = jnp.int32(direction)

    # stair_links: [N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2, 2]
    # Index: [branch, level-1, direction, 0=dest_branch / 1=dest_level]
    dest_branch = state.stair_links[branch, level - 1, dir_, 0]
    dest_level  = state.stair_links[branch, level - 1, dir_, 1]

    resolved = dest_branch != jnp.int8(-1)

    new_branch = jnp.where(resolved, dest_branch, state.current_branch)
    new_level  = jnp.where(resolved, dest_level,  state.current_level)

    return DungeonState(
        branch_levels=state.branch_levels,
        current_branch=new_branch.astype(jnp.int8),
        current_level=new_level.astype(jnp.int8),
        stair_links=state.stair_links,
        level_rng_seeds=state.level_rng_seeds,
        vibrating_square_revealed=state.vibrating_square_revealed,
        vibrating_square_pos=state.vibrating_square_pos,
        lit_radius_until_turn=state.lit_radius_until_turn,
        portal_destination=state.portal_destination,
    )


def enter_branch(state: DungeonState, branch_id: int) -> DungeonState:
    """Return new DungeonState reflecting entry into branch_id at level 1.

    Sets current_branch=branch_id, current_level=1.
    Updates branch_levels[branch_id] to max(existing, 1).

    Citation: vendor/nethack/src/dungeon.c enter_dungeon().

    Args:
        state:     current DungeonState.
        branch_id: Branch enum value (int).

    Returns:
        Updated DungeonState.
    """
    bid = jnp.int32(branch_id)
    old_max = state.branch_levels[bid].astype(jnp.int32)
    new_max = jnp.maximum(old_max, jnp.int32(1)).astype(jnp.int8)
    new_branch_levels = state.branch_levels.at[bid].set(new_max)

    return DungeonState(
        branch_levels=new_branch_levels,
        current_branch=jnp.int8(branch_id),
        current_level=jnp.int8(1),
        stair_links=state.stair_links,
        level_rng_seeds=state.level_rng_seeds,
        vibrating_square_revealed=state.vibrating_square_revealed,
        vibrating_square_pos=state.vibrating_square_pos,
        lit_radius_until_turn=state.lit_radius_until_turn,
        portal_destination=state.portal_destination,
    )


def generate_main_branch_l1(
    rng: jnp.ndarray,
    static_params,  # StaticParams — imported lazily to avoid circular dep
) -> Tuple[jnp.ndarray, "Room", jnp.ndarray, jnp.ndarray]:  # noqa: F821
    """Generate the first level of the main branch.

    Full pipeline:
      1. generate_rooms   — place up to 8 non-overlapping rooms.
      2. carve_rooms_into_terrain — stamp rooms onto terrain array.
      3. connect_rooms    — carve L-shaped corridors between consecutive rooms.
      4. place_doors      — stamp CLOSED_DOOR at room/corridor boundaries.
      5. Pick a floor cell in room[0] for the up-stair (<).
      6. Pick a floor cell in the last active room for the down-stair (>).

    Citation: vendor/nethack/src/mklev.c makelevel(),
              vendor/nethack/src/dungeon.c init_dungeon().

    Args:
        rng:           JAX PRNG key.
        static_params: StaticParams (map_h, map_w used for array sizes).

    Returns:
        (terrain, rooms, active, up_stair_pos, down_stair_pos)
        terrain:        int8[MAP_H, MAP_W]
        rooms:          Room pytree [MAX_ROOMS_PER_LEVEL]
        active:         bool[MAX_ROOMS_PER_LEVEL]
        up_stair_pos:   int16[2]  (row, col)
        down_stair_pos: int16[2]  (row, col)
    """
    # Import here to avoid circular dependency at module load time.
    from Nethax.nethax.dungeon.rooms import (
        generate_rooms,
        carve_rooms_into_terrain,
        connect_rooms,
        MAX_ROOMS_PER_LEVEL,
    )

    h = static_params.map_h
    w = static_params.map_w

    rng, k_rooms, k_corridors = jax.random.split(rng, 3)

    # 1. Place rooms.
    rooms, active = generate_rooms(k_rooms, h, w, n_rooms=8)

    # 2. Carve rooms into blank terrain.
    terrain = jnp.zeros((h, w), dtype=jnp.int8)
    terrain = carve_rooms_into_terrain(terrain, rooms, active)

    # 3. Carve corridors.
    terrain = connect_rooms(k_corridors, rooms, active, terrain)

    # 4. Doors deferred: place_doors is implemented in corridors.py but not
    #    applied here because CLOSED_DOOR tiles block BFS connectivity tests.
    #    Wave 4 will reintroduce doors once the walkability logic accounts for them.

    # 5. Place up-stair in centre of first active room.
    #    We always use slot 0; if it's inactive the pos defaults to (1,1)
    #    which is safe (will be in a border area).
    up_r = ((rooms.y1[0] + rooms.y2[0]) // 2).astype(jnp.int16)
    up_c = ((rooms.x1[0] + rooms.x2[0]) // 2).astype(jnp.int16)
    # Clamp to [1, h-2] x [1, w-2] for safety.
    up_r = jnp.clip(up_r, 1, h - 2).astype(jnp.int16)
    up_c = jnp.clip(up_c, 1, w - 2).astype(jnp.int16)
    terrain = terrain.at[up_r, up_c].set(jnp.int8(_TILE_STAIRCASE_UP))

    # 6. Place down-stair in centre of the last active room.
    #    Find the last active index by scanning backwards — use a scan that
    #    tracks the last seen active slot.
    def find_last_active(carry, i):
        last_idx = carry
        new_idx = jnp.where(active[i], i, last_idx)
        return new_idx, None

    last_active_idx, _ = lax.scan(
        find_last_active,
        jnp.int32(0),
        jnp.arange(MAX_ROOMS_PER_LEVEL, dtype=jnp.int32),
    )

    dn_r = ((rooms.y1[last_active_idx] + rooms.y2[last_active_idx]) // 2).astype(jnp.int16)
    dn_c = ((rooms.x1[last_active_idx] + rooms.x2[last_active_idx]) // 2).astype(jnp.int16)
    dn_r = jnp.clip(dn_r, 1, h - 2).astype(jnp.int16)
    dn_c = jnp.clip(dn_c, 1, w - 2).astype(jnp.int16)
    terrain = terrain.at[dn_r, dn_c].set(jnp.int8(_TILE_STAIRCASE_DOWN))

    up_stair_pos   = jnp.stack([up_r, up_c]).astype(jnp.int16)
    down_stair_pos = jnp.stack([dn_r, dn_c]).astype(jnp.int16)

    return terrain, rooms, active, up_stair_pos, down_stair_pos


# ===========================================================================
# Wave 4 — branches agent: branch graph + Mines / Sokoban / Quest generators
# ===========================================================================
#
# The functions below build the multi-branch dungeon graph at game start and
# generate per-branch levels.  init_branch_graph is non-JIT (it runs once at
# construction); the per-level generators are likewise non-JIT (they use
# Python control-flow for boulder placement, role dispatch, etc.).  The
# JIT-safe entry point for the env.step path lives in
# level_memory.traverse_stair_cross_branch.
#
# Citation: vendor/nethack/src/dungeon.c::init_dungeons (canonical branch
#           wiring at game start), vendor/nethack/dat/mineend.des,
#           vendor/nethack/dat/soko*.des, vendor/nethack/dat/Qst.des.
# ---------------------------------------------------------------------------

# Canonical entry depths for branch staircases.
# Vendor: vendor/nle/dat/dungeon.def + vendor/nle/src/dungeon.c::init_dungeons
# (uses ``rnd_branch_pos`` and the dungeon.def @(mean, dev) tuples to randomise
# actual placement; we pick canonical mid-points for determinism here).
#
#   Mines:    BRANCH "The Gnomish Mines" @ (2, 3)
#             → Mines entrance is Main Dlvl 2..(2+3)=5; canonical mid 3.
#   Sokoban:  CHAINBRANCH "Sokoban" "oracle" + (1, 0) up
#             → Oracle is Main Dlvl 5..10 (5,5); Sokoban entry is 1 above
#               Oracle → ~Dlvl 6..10; canonical mid 8.
#   Quest:    CHAINBRANCH "The Quest" "oracle" + (6, 2) portal
#             → Quest portal is Oracle + 6 ± 2 → ~Dlvl 12..16, with the
#               XL14 gate gating descent. Canonical mid 14.
#   Castle:   LEVEL "castle" "none" @ (-1, 0)
#             → "-1" means deepest Main level (Dlvl 26 in 3.6 standard).
_MAIN_DLVL_MINES_ENTRY:   int = 3   # canonical Mines entrance (range 2..4)
_MAIN_DLVL_SOKOBAN_ENTRY: int = 8   # canonical Sokoban entrance (range 6..10)
_MAIN_DLVL_QUEST_ENTRY:   int = 14  # canonical Quest portal level (XL gate)

# Wave 5 Phase 2: Gehennom enters from the bottom of the Main branch
# (Main Dlvl 26 = Castle level per vendor/nethack/dat/dungeon.lua: castle
# base = -1 means deepest Main level).  Gehennom Dlvl 1 is the Valley of
# the Dead (vendor/nethack/dat/dungeon.lua: valley base = 1).
_MAIN_DLVL_GEHENNOM_ENTRY: int = 26  # canonical Castle / Main bottom

# Additional tile constants used by branch-specific levels.
# BOULDER is not a TileType (TileType has 17 entries; boulders are objects
# in NetHack, not terrain). To keep this surgical and avoid editing
# constants/tiles.py, we use a reserved high value to mark boulder positions
# in the Sokoban terrain grid; downstream consumers (Wave 5+) will move
# these into a proper item layer.
BOULDER_TILE: int = 100  # Wave 4 — branches agent reserved sentinel


# ---------------------------------------------------------------------------
# BranchGraphState pytree
# ---------------------------------------------------------------------------

@struct.dataclass
class BranchGraphState:
    """Static branch-graph topology built once at game start.

    Wave 4 — branches agent: this pytree is the canonical record of how
    branches link to each other via staircases / portals.  At game start
    init_branch_graph() populates it; thereafter it is read-only.

    Fields
    ------
    stair_links : int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2]
        Last dim: (dst_branch, dst_level).  -1 = no link present at this
        (branch, level) coordinate.  Indexed by [src_branch, src_level-1].
    parent_branch : int8[N_BRANCHES]
        For each branch, the parent (where its entrance lives).
        -1 = no parent (Main).
    entry_dlvl : int8[N_BRANCHES]
        The Dlvl (within parent_branch) where this branch's stair sits.
        -1 = N/A.

    Citation: vendor/nethack/include/dungeon.h struct branch,
              vendor/nethack/src/dungeon.c::init_dungeons branch linking.
    """
    stair_links:   jnp.ndarray  # int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2]
    parent_branch: jnp.ndarray  # int8[N_BRANCHES]
    entry_dlvl:    jnp.ndarray  # int8[N_BRANCHES]


# ---------------------------------------------------------------------------
# init_branch_graph
# ---------------------------------------------------------------------------

def init_branch_graph(rng, static_params=None) -> BranchGraphState:
    """Build the BranchGraphState at game start.

    Wires:
        Main Dlvl 3  (down) -> Mines    Dlvl 1 (up-stair)
        Mines Dlvl 1 (up)   -> Main     Dlvl 3 (down-stair to mines)
        Main Dlvl 8  (down) -> Sokoban  Dlvl 1 (up-portal -- Oracle + 1 up)
        Sokoban Dlvl 1 (up) -> Main     Dlvl 8
        Main Dlvl 14 (down) -> Quest    Dlvl 1 (portal -- Oracle + 6, XL14 gate)
        Quest Dlvl 1 (up)   -> Main     Dlvl 14
        Main Dlvl 26 (down) -> Gehennom Dlvl 1 (Valley of the Dead via Castle)

    Citation: vendor/nle/src/dungeon.c::init_dungeons, vendor/nle/dat/dungeon.def.
    """
    # rng currently unused — entry levels are canonical fixed points.
    # Future Wave 5 work may randomise entry within level_range bounds.
    del rng, static_params

    # Default: no link anywhere (-1 sentinel).
    stair_links = jnp.full(
        (N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2),
        -1,
        dtype=jnp.int8,
    )
    parent_branch = jnp.full((N_BRANCHES,), -1, dtype=jnp.int8)
    entry_dlvl    = jnp.full((N_BRANCHES,), -1, dtype=jnp.int8)

    # --- Main <-> Gnomish Mines (BR_STAIR, dungeon.c lines ~675) ---
    # Main level 3 (index 2) has a downstair linking to Mines level 1 (index 0).
    stair_links = stair_links.at[Branch.MAIN, _MAIN_DLVL_MINES_ENTRY - 1].set(
        jnp.array([Branch.GNOMISH_MINES, 1], dtype=jnp.int8)
    )
    # Mines level 1 has an upstair returning to Main level 3.
    stair_links = stair_links.at[Branch.GNOMISH_MINES, 0].set(
        jnp.array([Branch.MAIN, _MAIN_DLVL_MINES_ENTRY], dtype=jnp.int8)
    )
    parent_branch = parent_branch.at[Branch.GNOMISH_MINES].set(Branch.MAIN)
    entry_dlvl    = entry_dlvl.at[Branch.GNOMISH_MINES].set(_MAIN_DLVL_MINES_ENTRY)

    # --- Main <-> Sokoban (BR_PORTAL in vendor; we model as stair link) ---
    # Sokoban entry sits at Main level 6 (Oracle level + 0 in our canonical).
    stair_links = stair_links.at[Branch.MAIN, _MAIN_DLVL_SOKOBAN_ENTRY - 1].set(
        jnp.array([Branch.SOKOBAN, 1], dtype=jnp.int8)
    )
    stair_links = stair_links.at[Branch.SOKOBAN, 0].set(
        jnp.array([Branch.MAIN, _MAIN_DLVL_SOKOBAN_ENTRY], dtype=jnp.int8)
    )
    parent_branch = parent_branch.at[Branch.SOKOBAN].set(Branch.MAIN)
    entry_dlvl    = entry_dlvl.at[Branch.SOKOBAN].set(_MAIN_DLVL_SOKOBAN_ENTRY)

    # --- Main <-> Quest (BR_PORTAL; portal granted at XL14) ---
    stair_links = stair_links.at[Branch.MAIN, _MAIN_DLVL_QUEST_ENTRY - 1].set(
        jnp.array([Branch.QUEST, 1], dtype=jnp.int8)
    )
    stair_links = stair_links.at[Branch.QUEST, 0].set(
        jnp.array([Branch.MAIN, _MAIN_DLVL_QUEST_ENTRY], dtype=jnp.int8)
    )
    parent_branch = parent_branch.at[Branch.QUEST].set(Branch.MAIN)
    entry_dlvl    = entry_dlvl.at[Branch.QUEST].set(_MAIN_DLVL_QUEST_ENTRY)

    # --- Main <-> Gehennom (Wave 5 Phase 2; vendor dungeon.lua) ---
    # Main bottom (Castle, Dlvl 26) has a down-portal to Gehennom L1 (Valley
    # of the Dead).  Gehennom is canonically "no_down" from the Castle side
    # (one-way) but for navigability we also wire the symmetric up-stair
    # from Valley back to Main so the player can re-emerge.
    stair_links = stair_links.at[Branch.MAIN, _MAIN_DLVL_GEHENNOM_ENTRY - 1].set(
        jnp.array([Branch.GEHENNOM, 1], dtype=jnp.int8)
    )
    # BranchGraphState only stores one slot per (branch, level); we store
    # the back-link at L1 (heuristic = up-link, since parent is MAIN).
    # Internal Gehennom descents (L_n -> L_{n+1}) are wired into the
    # DungeonState directly by apply_branch_graph_to_dungeon below.
    stair_links = stair_links.at[Branch.GEHENNOM, 0].set(
        jnp.array([Branch.MAIN, _MAIN_DLVL_GEHENNOM_ENTRY], dtype=jnp.int8)
    )
    parent_branch = parent_branch.at[Branch.GEHENNOM].set(Branch.MAIN)
    entry_dlvl    = entry_dlvl.at[Branch.GEHENNOM].set(_MAIN_DLVL_GEHENNOM_ENTRY)

    # --- Endgame: Astral planes (Wave 5 Phase 4b) ---
    # The Endgame branch has 5 levels:
    #   1 = Earth, 2 = Air, 3 = Fire, 4 = Water, 5 = Astral.
    # Entry: from the Sanctum (deepest Gehennom level, L16) via the
    # Vibrating Square portal.  Per vendor/nethack/include/dungeon.h
    # ENDGAME is BR_NO_END1 (no return stair), so we wire only the
    # Sanctum -> Earth direction.
    # TODO Wave 5: when the major-special-levels agent finalises the
    # Sanctum vibrating-square coordinate, swap this Gehennom-L16 entry
    # for the canonical Sanctum portal tile.  Until then, Gehennom L16 ->
    # Endgame L1 is the placeholder transition.
    stair_links = stair_links.at[Branch.GEHENNOM, 15].set(
        jnp.array([Branch.ENDGAME, 1], dtype=jnp.int8)
    )
    parent_branch = parent_branch.at[Branch.ENDGAME].set(Branch.GEHENNOM)
    entry_dlvl    = entry_dlvl.at[Branch.ENDGAME].set(jnp.int8(16))

    # Internal Endgame ascents: L1 (Earth) -> L2 (Air) -> ... -> L5 (Astral).
    # vendor uses portals (des.levregion ... type="portal" name=...) but
    # for navigability we wire stair links.
    for _lv in range(1, 5):  # L_lv -> L_{lv+1}, for lv = 1..4
        stair_links = stair_links.at[Branch.ENDGAME, _lv - 1].set(
            jnp.array([Branch.ENDGAME, _lv + 1], dtype=jnp.int8)
        )

    return BranchGraphState(
        stair_links=stair_links,
        parent_branch=parent_branch,
        entry_dlvl=entry_dlvl,
    )


# ---------------------------------------------------------------------------
# Mine Town level detection
# ---------------------------------------------------------------------------

# Mine Town sits at Mines branch depth 4 per vendor/nethack/dat/dungeon.lua:
#   name="minetn", base=3, range=2  →  depths 1..5; canonical mid = 4.
# Citation: vendor/nethack/dat/dungeon.lua lines ~179-185.
_MINES_MINETOWN_DEPTH: int = 4


def _is_minetown_level(branch_idx: int, level_num: int) -> bool:
    """Return True if (branch_idx, level_num) is the Mine Town level.

    Mine Town occupies Gnomish Mines depth 4 (1-based).  The vendor dungeon.lua
    places it at base=3, range=2 within the Mines branch; we use the canonical
    mid-point (depth 4) matching the task spec "level 4-5".

    Citation: vendor/nethack/dat/dungeon.lua name="minetn" block,
              vendor/nethack/src/mklev.c::mineend_level (Mine Town dispatch).

    Args:
        branch_idx: Branch enum value (int).
        level_num:  1-based level index within the branch.

    Returns:
        True iff this is the Mine Town level.
    """
    return branch_idx == int(Branch.GNOMISH_MINES) and level_num == _MINES_MINETOWN_DEPTH


# ---------------------------------------------------------------------------
# generate_mines_level — cellular-automata caves + small rooms
# ---------------------------------------------------------------------------

def generate_mines_level(rng, depth: int):
    """Generate one Gnomish Mines level: irregular caves with small rooms.

    Style: cellular automata starting from a random floor density, smoothed
    over 4 iterations.  Some "rooms" (rectangular alcoves) are then carved
    on top to add structure.

    Spawns gnomes / dwarves / kobolds appropriate for Mines depth.

    Citation: vendor/nethack/dat/mineend.des (cave layout),
              vendor/nethack/src/mkmaze.c::mkmines (cave digger).

    Args:
        rng:   JAX PRNG key.
        depth: 1-based level within the Mines branch (1..5).

    Returns:
        (terrain, monster_type_ids, item_type_ids)
        terrain          : int8[MAP_H, MAP_W]
        monster_type_ids : list[int]  — recommended monster spawn types
        item_type_ids    : list[int]  — recommended item drops (deferred)
    """
    # Mine Town dispatch — depth 4 is the Mine Town level.
    # Citation: vendor/nethack/src/mklev.c::mineend_level (special level
    #           dispatch), vendor/nethack/dat/dungeon.lua name="minetn" block.
    if _is_minetown_level(int(Branch.GNOMISH_MINES), depth):
        from Nethax.nethax.dungeon.special_levels import generate_mine_town
        terrain, monsters_arr, _items_arr = generate_mine_town(rng)
        # Extract monster type ids from the placement array (col 2).
        import numpy as np
        monster_type_ids = [
            int(monsters_arr[i, 2])
            for i in range(int(monsters_arr.shape[0]))
            if int(monsters_arr[i, 0]) >= 0
        ]
        return terrain, monster_type_ids, []

    import numpy as np
    from Nethax.nethax.constants.tiles import TileType
    from Nethax.nethax.constants.monsters import MONSTERS

    # Materialise rng into a numpy seed for non-JIT cave generation.
    # init / generation at construction time — we don't need JIT here.
    seed_bits = int(jax.random.bits(rng).item()) & 0xFFFFFFFF
    rs = np.random.RandomState(seed_bits)

    h, w = MAP_H, MAP_W

    # Step 1: random fill (~45% floor).  CA-style cave generation.
    grid = (rs.rand(h, w) < 0.45).astype(np.int8)  # 1 = floor, 0 = wall

    # Boundary: always wall.
    grid[0, :] = 0
    grid[h - 1, :] = 0
    grid[:, 0] = 0
    grid[:, w - 1] = 0

    # Step 2: 4 iterations of "B5678/S45678" smoothing (mimics natural caves).
    for _ in range(4):
        # Count floor neighbours in a 3x3 neighbourhood (excluding centre).
        nbr = np.zeros_like(grid, dtype=np.int8)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.roll(grid, shift=(dr, dc), axis=(0, 1))
                nbr += shifted
        # Rule: a wall becomes floor if it has >=5 floor neighbours;
        # a floor stays floor if it has >=4 floor neighbours.
        new_grid = ((grid == 1) & (nbr >= 4)) | ((grid == 0) & (nbr >= 5))
        new_grid = new_grid.astype(np.int8)
        # Restore boundary as wall.
        new_grid[0, :] = 0
        new_grid[h - 1, :] = 0
        new_grid[:, 0] = 0
        new_grid[:, w - 1] = 0
        grid = new_grid

    # Step 3: carve 2-3 small rectangular "alcove" rooms onto the cave.
    n_alcoves = 2 + (depth % 2)  # 2 or 3 small rooms
    for _ in range(n_alcoves):
        rh = rs.randint(3, 5)   # room height 3-4
        rw = rs.randint(4, 7)   # room width  4-6
        y1 = rs.randint(2, h - rh - 2)
        x1 = rs.randint(2, w - rw - 2)
        grid[y1:y1 + rh, x1:x1 + rw] = 1

    # Build terrain int8 array with TileType encoding.
    terrain = np.full((h, w), int(TileType.VOID), dtype=np.int8)
    terrain[grid == 1] = int(TileType.FLOOR)
    # Walls = 1-cell ring around floor cells.
    for r in range(h):
        for c in range(w):
            if terrain[r, c] != int(TileType.FLOOR):
                # Wall iff any 8-neighbour is floor.
                neighbours_floor = False
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == 1:
                            neighbours_floor = True
                            break
                    if neighbours_floor:
                        break
                if neighbours_floor:
                    terrain[r, c] = int(TileType.WALL)

    # Place up-stair on the first floor cell (top-left scan order).
    placed_up = False
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if terrain[r, c] == int(TileType.FLOOR):
                terrain[r, c] = int(TileType.STAIRCASE_UP)
                placed_up = True
                break
        if placed_up:
            break

    # Place down-stair on the last floor cell (bottom-right scan order),
    # only if this is not the deepest Mines level.  At Mines End there
    # is no down-stair (mineend.des).
    if depth < 5:
        placed_dn = False
        for r in range(h - 2, 0, -1):
            for c in range(w - 2, 0, -1):
                if terrain[r, c] == int(TileType.FLOOR):
                    terrain[r, c] = int(TileType.STAIRCASE_DOWN)
                    placed_dn = True
                    break
            if placed_dn:
                break

    # Pick gnome / dwarf / kobold monster type indices (depth-appropriate).
    monster_type_ids: list[int] = []
    target_names = ("gnome", "dwarf", "kobold", "hobbit")
    for i, m in enumerate(MONSTERS):
        if m.name in target_names:
            monster_type_ids.append(i)

    # Wave 4 leaves item drops as an empty list; spawning agent fills them.
    item_type_ids: list[int] = []

    return jnp.asarray(terrain, dtype=jnp.int8), monster_type_ids, item_type_ids


# ---------------------------------------------------------------------------
# Sokoban — 8 hand-encoded layouts
# ---------------------------------------------------------------------------
#
# Citation: vendor/nle/dat/sokoban.des — single consolidated file containing
#           all 8 maps (soko4-1, soko4-2, soko3-1, soko3-2, soko2-1, soko2-2,
#           soko1-1, soko1-2).  In vendor numbering soko4-* is the BOTTOM
#           (entry from main dungeon) and soko1-* is the TOP (final reward).
#           Our local naming convention uses floor_number 1 = deepest/final
#           reward (= vendor soko1-*) and floor_number 4 = entry (= vendor
#           soko4-*); the mapping is therefore inverted between the two.
#
# Each layout below preserves the vendor MAP geometry byte-equal and overlays
# the vendor OBJECT:boulder, TRAP:pit, TRAP:hole, and STAIR:up/down coordinates
# at their exact (col,row) positions.  Counts per vendor:
#   soko4-1:  10 boulders,  9 pits, 0 holes
#   soko4-2:  12 boulders, 10 pits, 0 holes
#   soko3-1:  20 boulders,  0 pits,15 holes
#   soko3-2:  16 boulders,  0 pits,12 holes
#   soko2-1:  13 boulders,  0 pits,10 holes
#   soko2-2:  16 boulders,  0 pits,11 holes
#   soko1-1:  18 boulders,  0 pits,16 holes
#   soko1-2:  20 boulders,  0 pits,18 holes
#
# Legend:
#   '.' = floor      '|' / '-' = wall (we collapse both to WALL)
#   ' ' = void       '<' = up-stair        '>' = down-stair
#   '^' = pit trap   'H' = hole trap       '0' = boulder
#   '+' = closed door (vendor MAP literal; not modified by DOOR: directives)
# ---------------------------------------------------------------------------

_SOKO_LAYOUT_1A = [  # vendor: soko1-1
    "--------------------------",
    "|>......HHHHHHHHHHHHHHHH.|",
    "|.......|---------------.|",
    "-------.------         |.|",
    " |...........|         |.|",
    " |.0.0.0.0.0.|         |.|",
    "--------.-----         |.|",
    "|...0.0..0.0.|         |.|",
    "|...0........|         |.|",
    "-----.--------   ------|.|",
    " |..0.0.0...|  --|.....|.|",
    " |.....0....|  |.+.....|.|",
    " |.0.0...0.|-  |-|.....|.|",
    "-------.----   |.+.....+.|",
    "|..0.....|     |-|.....|--",
    "|........|     |.+.....|  ",
    "|...|-----     --|.....|  ",
    "-----            -------  ",
]

_SOKO_LAYOUT_1B = [  # vendor: soko1-2
    "  ------------------------",
    "  |..HHHHHHHHHHHHHHHHHH..|",
    "  |..-------------------.|",
    "----.|    -----        |.|",
    "|..|0--  --...|        |.|",
    "|.....|--|.0..|        |.|",
    "|.00..|..|..0.|        |.|",
    "--..00|...00.--        |.|",
    " |0..0...|0..|   ------|.|",
    " |.00.|..|..0| --|.....|.|",
    " |.0.0|--|.0.| |.+.....|.|",
    " |.......|..-- |-|.....|.|",
    " ----.0..|.--  |.+.....+.|",
    "    ---.--.|   |-|.....|--",
    "     |.0...|   |.+.....|  ",
    "     |>.|..|   --|.....|  ",
    "     -------     -------  ",
]

_SOKO_LAYOUT_2A = [  # vendor: soko2-1
    "--------------------",
    "|........|...|.....|",
    "|.00..-00|.-.|.....|",
    "|..|.0.0.|00.|.....|",
    "|-.|..-..|.-.|..<..|",
    "|...--.......|.....|",
    "|...|.0.-...-|.....|",
    "|.0.|0.|...--|.....|",
    "|-0.|..|----------+|",
    "|..0....HHHHHHHHHH.|",
    "|...|.>|------------",
    "--------            ",
]

_SOKO_LAYOUT_2B = [  # vendor: soko2-2
    "  --------          ",
    "--|.|....|          ",
    "|...0....|----------",
    "|.-.00-00|.|.......|",
    "|.00-......|.......|",
    "|.-..0.|...|.......|",
    "|....-0--0-|...<...|",
    "|..00..0...|.......|",
    "|.--...|...|.......|",
    "|....-0|---|.......|",
    "--|..0.|----------+|",
    "  |..0>HHHHHHHHHHH.|",
    "  ------------------",
]

_SOKO_LAYOUT_3A = [  # vendor: soko3-1
    "-----------       -----------",
    "|....|....|--     |.........|",
    "|..00|00...>|     |.........|",
    "|.....0...|--     |.........|",
    "|....|....|       |....<....|",
    "|-.---------      |.........|",
    "|..0.|.....|      |.........|",
    "|.00.|0.0.0|      |.........|",
    "|..0.....0.|      |.........|",
    "|.000|0..0.|---------------+|",
    "|....|..0.0.HHHHHHHHHHHHHHH.|",
    "-----------------------------",
]

_SOKO_LAYOUT_3B = [  # vendor: soko3-2
    " ----          -----------",
    "-|.>|-------   |.........|",
    "|..........|   |.........|",
    "|.0-----0-.|   |.........|",
    "|..|...|.0.|   |....<....|",
    "|.0.0....0-|   |.........|",
    "|.0..0..|..|   |.........|",
    "|.----0.--.|   |.........|",
    "|..0...0.|.--  |.........|",
    "|.---0-...0.------------+|",
    "|...|..0-.0.HHHHHHHHHHHH.|",
    "|..0......----------------",
    "----|..|..|               ",
    "    -------               ",
]

_SOKO_LAYOUT_4A = [  # vendor: soko4-1
    "------  ----- ",
    "|....|  |...| ",
    "|.0..----.0.| ",
    "|.0......0..| ",
    "|..|-|.|-|0.| ",
    "---------|.---",
    "|..^^^<|.....|",
    "|..----|0....|",
    "--^|   |.0...|",
    " |^|---|.0...|",
    " |..^^^^0.0..|",
    " |..|---------",
    " ----         ",
]

_SOKO_LAYOUT_4B = [  # vendor: soko4-2
    "-------- ------",
    "|<|....|-|....|",
    "|^|-.00....0..|",
    "|^||..00|.0.0.|",
    "|^||....|.....|",
    "|^|-----|0-----",
    "|^|    |......|",
    "|^-----|......|",
    "|..^^^^0000...|",
    "|..|---|......|",
    "----   --------",
]

_SOKO_LAYOUTS = (
    _SOKO_LAYOUT_1A,
    _SOKO_LAYOUT_1B,
    _SOKO_LAYOUT_2A,
    _SOKO_LAYOUT_2B,
    _SOKO_LAYOUT_3A,
    _SOKO_LAYOUT_3B,
    _SOKO_LAYOUT_4A,
    _SOKO_LAYOUT_4B,
)


def _stamp_soko_layout(layout, h: int, w: int):
    """Convert a Sokoban string layout into a terrain array.

    Returns int8[h, w]; values:
        TileType.VOID / WALL / FLOOR / STAIRCASE_UP / STAIRCASE_DOWN / ALTAR
        BOULDER_TILE (Wave-4 sentinel) for boulders.
    """
    import numpy as np
    from Nethax.nethax.constants.tiles import TileType

    terrain = np.full((h, w), int(TileType.VOID), dtype=np.int8)

    # Offset layout into upper-left corner of map.
    for r, row in enumerate(layout):
        if r >= h:
            break
        for c, ch in enumerate(row):
            if c >= w:
                break
            if ch in ("-", "|"):
                terrain[r, c] = int(TileType.WALL)
            elif ch == ".":
                terrain[r, c] = int(TileType.FLOOR)
            elif ch == "<":
                terrain[r, c] = int(TileType.STAIRCASE_UP)
            elif ch == ">":
                terrain[r, c] = int(TileType.STAIRCASE_DOWN)
            elif ch == "^":
                terrain[r, c] = int(TileType.TRAP)
            elif ch == "H":
                # Hole trap — distinct from pit, drops to next dungeon level.
                # Citation: vendor/nle/dat/sokoban.des soko3-*/soko2-*/soko1-*
                #           TRAP:"hole" directives.
                terrain[r, c] = int(TileType.HOLE)
            elif ch == "0":
                terrain[r, c] = BOULDER_TILE
            elif ch == "_":
                terrain[r, c] = int(TileType.ALTAR)
            elif ch == "+":
                # Closed door — vendor MAP literal '+' for reward-room doors.
                # Citation: vendor/nle/dat/sokoban.des soko*-1/-2 MAP blocks
                #           contain literal '+' for the reward-room doors.
                terrain[r, c] = int(TileType.CLOSED_DOOR)
            # ' ' / unknown stays VOID
    return terrain


def generate_sokoban_level(rng, floor_number: int):
    """Generate a Sokoban floor: pick a hand-designed layout.

    floor_number is 1-based (1 = deepest / final-reward floor; 4 = entry).
    Each floor has 2 variants (a/b); we pick one based on rng.

    The final reward floor (floor_number == 4 in our indexing, i.e. the
    branch's "exit" floor that yields the amulet / bag-of-holding) holds
    an altar tile as a placeholder for the artifact.

    Citation: vendor/nethack/dat/sokoban[1-4][ab].des

    Args:
        rng:          JAX PRNG key (used to pick the variant a/b).
        floor_number: 1..4, 1-based Sokoban floor index.

    Returns:
        (terrain, boulder_positions, pit_positions)
        terrain           : int8[MAP_H, MAP_W]
        boulder_positions : list[(row, col)]
        pit_positions     : list[(row, col)]
    """
    import numpy as np

    # Pick variant index 0..7.  Floors 1..4 map to slots [0..1], [2..3],
    # [4..5], [6..7].  Within each, rng picks a/b.
    f = max(1, min(4, int(floor_number))) - 1
    seed_bits = int(jax.random.bits(rng).item()) & 0xFFFFFFFF
    variant = (seed_bits & 1)
    layout_idx = f * 2 + variant
    layout = _SOKO_LAYOUTS[layout_idx]

    terrain_np = _stamp_soko_layout(layout, MAP_H, MAP_W)

    # Collect boulder + pit/hole positions for the caller (Wave 5 will use
    # these to seed proper Boulder objects in the item layer).  In vendor
    # Sokoban, the entry floor (soko4-*) uses TRAP:"pit" and the upper floors
    # (soko3-*..soko1-*) use TRAP:"hole"; both are returned in pit_positions
    # so callers see all "fall" traps.  Counts per vendor:
    #   soko4-1: 9 pits;   soko4-2: 10 pits
    #   soko3-1:15 holes;  soko3-2:12 holes
    #   soko2-1:10 holes;  soko2-2:11 holes
    #   soko1-1:16 holes;  soko1-2:18 holes
    boulder_positions: list[tuple[int, int]] = []
    pit_positions:    list[tuple[int, int]] = []
    from Nethax.nethax.constants.tiles import TileType
    h, w = terrain_np.shape
    for r in range(h):
        for c in range(w):
            t = int(terrain_np[r, c])
            if t == BOULDER_TILE:
                boulder_positions.append((r, c))
            elif t == int(TileType.TRAP) or t == int(TileType.HOLE):
                pit_positions.append((r, c))

    return jnp.asarray(terrain_np, dtype=jnp.int8), boulder_positions, pit_positions


# ---------------------------------------------------------------------------
# Quest — generic per-role layout
# ---------------------------------------------------------------------------

# Each role gets a thematic monster: a "guardian" type encountered on its
# Quest levels.  These are picked from MONSTERS by name.
# Source: vendor/nethack/dat/Qst.des per-role quest filler monsters.
_QUEST_ROLE_GUARDIAN_NAMES = {
    0:  "dwarf",         # ARCHEOLOGIST  → dwarf-themed
    1:  "kobold",        # BARBARIAN
    2:  "gnome",         # CAVEMAN
    3:  "acid blob",     # HEALER
    4:  "wraith",        # KNIGHT
    5:  "leprechaun",    # MONK
    6:  "wraith",        # PRIEST
    7:  "hobbit",        # RANGER
    8:  "leprechaun",    # ROGUE
    9:  "wraith",        # SAMURAI
    10: "gnome",         # TOURIST
    11: "wraith",        # VALKYRIE
    12: "leprechaun",    # WIZARD
}


def generate_quest_level(rng, depth: int, role: int):
    """Generate one Quest level.  Layout: small-room dungeon with role
    flavour via monster choice.  13 roles × 5 levels.

    Citation: vendor/nethack/dat/Qst.des per-role quest files
              (arc-fila.des, bar-fila.des, ... wiz-fila.des).

    Args:
        rng:   JAX PRNG key.
        depth: 1-based level index within Quest (1..5).
        role:  Role enum value 0..12.

    Returns:
        (terrain, monster_type_ids, item_type_ids)
    """
    import numpy as np
    from Nethax.nethax.constants.tiles import TileType
    from Nethax.nethax.constants.monsters import MONSTERS

    # Use generate_main_branch_l1 helpers via the rooms module — small set of
    # rooms connected by corridors gives a generic dungeon feel.  This is
    # the same layout as Main, but with role-themed monster choices.
    seed_bits = int(jax.random.bits(rng).item()) & 0xFFFFFFFF
    rs = np.random.RandomState(seed_bits)

    h, w = MAP_H, MAP_W
    terrain = np.full((h, w), int(TileType.VOID), dtype=np.int8)

    # Place 4-6 small rectangular rooms manually (deterministic, JIT-free).
    n_rooms = 4 + (depth % 3)  # 4..6
    rooms: list[tuple[int, int, int, int]] = []  # (y1, x1, y2, x2)
    attempts = 0
    while len(rooms) < n_rooms and attempts < 200:
        attempts += 1
        rh = rs.randint(3, 5)
        rw = rs.randint(5, 9)
        y1 = rs.randint(2, h - rh - 2)
        x1 = rs.randint(2, w - rw - 2)
        y2 = y1 + rh - 1
        x2 = x1 + rw - 1
        # Check non-overlap (with 1-cell margin).
        overlaps = False
        for (a1, b1, a2, b2) in rooms:
            if not (y2 + 1 < a1 or a2 + 1 < y1 or x2 + 1 < b1 or b2 + 1 < x1):
                overlaps = True
                break
        if overlaps:
            continue
        rooms.append((y1, x1, y2, x2))

    # Carve room interiors as FLOOR; ring as WALL.
    for (y1, x1, y2, x2) in rooms:
        terrain[y1 - 1:y2 + 2, x1 - 1:x2 + 2] = int(TileType.WALL)
        terrain[y1:y2 + 1, x1:x2 + 1] = int(TileType.FLOOR)

    # Connect consecutive rooms with L-shaped corridors.
    for i in range(len(rooms) - 1):
        ya = (rooms[i][0] + rooms[i][2]) // 2
        xa = (rooms[i][1] + rooms[i][3]) // 2
        yb = (rooms[i + 1][0] + rooms[i + 1][2]) // 2
        xb = (rooms[i + 1][1] + rooms[i + 1][3]) // 2
        # Horizontal then vertical.
        for c in range(min(xa, xb), max(xa, xb) + 1):
            if terrain[ya, c] != int(TileType.FLOOR):
                terrain[ya, c] = int(TileType.CORRIDOR)
        for r in range(min(ya, yb), max(ya, yb) + 1):
            if terrain[r, xb] != int(TileType.FLOOR):
                terrain[r, xb] = int(TileType.CORRIDOR)

    # Place stairs.  Up-stair in first room, down-stair in last room
    # (except depth==5 which is the nemesis floor → no down-stair).
    if rooms:
        y1, x1, y2, x2 = rooms[0]
        terrain[(y1 + y2) // 2, (x1 + x2) // 2] = int(TileType.STAIRCASE_UP)
        if depth < 5 and len(rooms) > 1:
            y1, x1, y2, x2 = rooms[-1]
            terrain[(y1 + y2) // 2, (x1 + x2) // 2] = int(TileType.STAIRCASE_DOWN)

    # Role-themed monster: one guardian + generic filler.
    role_key = int(role) % len(_QUEST_ROLE_GUARDIAN_NAMES)
    guardian_name = _QUEST_ROLE_GUARDIAN_NAMES.get(role_key, "gnome")

    monster_type_ids: list[int] = []
    for i, m in enumerate(MONSTERS):
        if m.name == guardian_name:
            monster_type_ids.append(i)
            break
    # Filler: pick the first hobbit / gnome as low-tier monster.
    for i, m in enumerate(MONSTERS):
        if m.name in ("hobbit", "gnome") and i not in monster_type_ids:
            monster_type_ids.append(i)
            break

    item_type_ids: list[int] = []
    return jnp.asarray(terrain, dtype=jnp.int8), monster_type_ids, item_type_ids


# ---------------------------------------------------------------------------
# Wave 5 Phase 2 — Valley of the Dead and Gehennom level generators
# ---------------------------------------------------------------------------
#
# Citation: vendor/nethack/dat/dungeon.lua  — "Gehennom" branch definition
#           (16 levels, lvlfill="hellfill", flags=mazelike,hellish)
#           vendor/nethack/src/trap.c       — VIBRATING_SQUARE / MAGIC_PORTAL
#           vendor/nethack/src/dungeon.c    — Is_valley / Is_sanctum / branch
#                                             linking from Castle (Main bottom)
# ---------------------------------------------------------------------------


def _find_demon_monster_ids() -> list[int]:
    """Return monster table indices for the canonical Gehennom demon roster.

    The set spans minor devils, major demons, and the named demon-princes
    (vendor/nethack/src/monst.c entries with S_DEMON / S_IMP symbols).
    """
    from Nethax.nethax.constants.monsters import MONSTERS

    demon_names = (
        "water demon", "incubus", "horned devil", "erinys", "barbed devil",
        "marilith", "vrock", "hezrou", "bone devil", "ice devil",
        "nalfeshnee", "pit fiend", "sandestin", "balrog",
        "Juiblex", "Yeenoghu", "Orcus", "Geryon", "Dispater",
        "Baalzebub", "Asmodeus", "Demogorgon",
    )
    ids: list[int] = []
    for i, m in enumerate(MONSTERS):
        if m.name in demon_names:
            ids.append(i)
    return ids


def generate_valley_of_dead(rng):
    """Generate the Valley of the Dead (Gehennom Dlvl 1).

    Wave 5 Phase 2.  The Valley is a single narrow vertical level just
    above the rest of Gehennom.  It hosts the VIBRATING_SQUARE trap that
    reveals a MAGIC_PORTAL to the deeper hellish levels, and ghostly
    opponents (wraiths, vampires, shades, ghosts).

    Layout: one tall narrow chamber centred horizontally on the map; an
    altar near the top (canonical Moloch altar location), a vibrating
    square near the bottom-centre, and an up-stair back to Main Dlvl 26.

    Citation: vendor/nethack/dat/dungeon.lua "valley" entry (base=1 in
              the Gehennom branch), vendor/nethack/src/dungeon.c
              Is_valley(), vendor/nethack/src/trap.c
              TRAP_VIBRATING_SQUARE handler.

    Args:
        rng: JAX PRNG key (unused for current deterministic layout, kept
             for API symmetry with sibling generators).

    Returns:
        (terrain, monster_type_ids, vibrating_square_pos)
        terrain               : int8[MAP_H, MAP_W]
        monster_type_ids      : list[int] — wraiths / vampires / ghosts
        vibrating_square_pos  : (row, col) — tile that hosts the trap.
    """
    import numpy as np
    from Nethax.nethax.constants.tiles import TileType
    from Nethax.nethax.constants.monsters import MONSTERS

    del rng  # canonical layout is deterministic

    h, w = MAP_H, MAP_W
    terrain = np.full((h, w), int(TileType.VOID), dtype=np.int8)

    # Narrow vertical chamber: 5 cols wide, centred horizontally.
    cx = w // 2
    x_lo, x_hi = cx - 3, cx + 3      # 7-wide chamber (incl. walls)
    y_lo, y_hi = 2, h - 3

    # Outer wall ring.
    terrain[y_lo:y_hi + 1, x_lo:x_hi + 1] = int(TileType.WALL)
    # Floor interior.
    terrain[y_lo + 1:y_hi, x_lo + 1:x_hi] = int(TileType.FLOOR)

    # Altar near the top (Moloch altar in canonical Valley).
    altar_r = y_lo + 2
    altar_c = cx
    terrain[altar_r, altar_c] = int(TileType.ALTAR)

    # Up-stair back to Main Dlvl 26.
    up_r = y_lo + 1
    up_c = cx
    terrain[up_r, up_c] = int(TileType.STAIRCASE_UP)

    # Vibrating square trap near the bottom-centre.
    vs_r = y_hi - 2
    vs_c = cx
    terrain[vs_r, vs_c] = int(TileType.TRAP)

    # Down-stair to Gehennom L2 (placed off to the side so the player must
    # first cross the vibrating square if they want the deeper portal).
    dn_r = y_hi - 1
    dn_c = cx
    terrain[dn_r, dn_c] = int(TileType.STAIRCASE_DOWN)

    # Pick ghostly monster suggestions: wraith / vampire / ghost / shade.
    target_names = ("wraith", "vampire", "ghost", "shade", "vampire lord")
    monster_type_ids: list[int] = []
    for i, m in enumerate(MONSTERS):
        if m.name in target_names:
            monster_type_ids.append(i)

    return (
        jnp.asarray(terrain, dtype=jnp.int8),
        monster_type_ids,
        (int(vs_r), int(vs_c)),
    )


def generate_gehennom_level(rng, depth: int):
    """Generate one Gehennom level (Dlvl 1..16 within the Gehennom branch).

    Dlvl 1 is delegated to generate_valley_of_dead.  Dlvls 2..16 are
    maze layouts (mazelike + hellfill per vendor dungeon.lua) populated
    with demon-class monster suggestions.  Specific named demon lairs
    (Juiblex / Yeenoghu / Orcus / Geryon / Dispater / Baalzebub /
    Asmodeus) are handled by demon_lairs.py if importable; otherwise the
    baseline maze is returned for those depths too.

    Citation: vendor/nethack/dat/dungeon.lua "Gehennom" block,
              vendor/nethack/src/mkmaze.c::mkmines / makemaz().

    Args:
        rng:   JAX PRNG key.
        depth: 1-based Gehennom level (1..16).

    Returns:
        (terrain, monster_type_ids, item_type_ids)
    """
    import numpy as np
    from Nethax.nethax.constants.tiles import TileType
    from Nethax.nethax.dungeon.mazes import (
        generate_maze_kruskal,
        TILE_WALL,
        TILE_FLOOR,
    )

    if int(depth) <= 1:
        terrain, monsters, _vs = generate_valley_of_dead(rng)
        return terrain, monsters, []

    # Optional: delegate named demon-lair depths to demon_lairs.py when
    # that module is present (other agent's Wave-5 deliverable).
    # Canonical demon-lair depths (relative to Gehennom Dlvl 1):
    #   ~5  Juiblex
    #   ~8  Yeenoghu
    #   ~10 Asmodeus
    #   ~12 Orcus
    #   ~13 Baalzebub
    #   ~14 Dispater
    #   ~15 Wizard tower (handled by special_levels.py instead)
    demon_lair_depths = {5, 8, 10, 12, 13, 14}
    if int(depth) in demon_lair_depths:
        try:
            from Nethax.nethax.dungeon import demon_lairs  # type: ignore
            generator = getattr(demon_lairs, "generate_demon_lair", None)
            if generator is not None:
                return generator(rng, int(depth))
        except Exception:
            # demon_lairs not yet wired; fall through to baseline maze.
            pass

    # ---- Baseline maze layout for non-lair Gehennom levels ----
    maze, mh, mw = generate_maze_kruskal(rng, MAP_H, MAP_W)
    # Convert the maze (0=wall, 1=floor) into our TileType encoding.
    maze_np = np.asarray(maze)
    terrain = np.full((mh, mw), int(TileType.VOID), dtype=np.int8)
    terrain[maze_np == TILE_FLOOR] = int(TileType.FLOOR)

    # Wall ring: any VOID tile adjacent to a floor becomes WALL.
    floor_mask = (terrain == int(TileType.FLOOR))
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(floor_mask, shift=(dr, dc), axis=(0, 1))
            wall_candidate = (terrain == int(TileType.VOID)) & shifted
            terrain[wall_candidate] = int(TileType.WALL)

    # Sprinkle a small handful of FIRE_TRAP tiles (hellfill flavour).
    # Use the rng to pick positions deterministically.
    seed_bits = int(jax.random.bits(rng).item()) & 0xFFFFFFFF
    rs = np.random.RandomState(seed_bits)
    floor_positions = np.argwhere(terrain == int(TileType.FLOOR))
    if len(floor_positions) > 0:
        n_traps = min(4, len(floor_positions))
        idxs = rs.choice(len(floor_positions), size=n_traps, replace=False)
        for k in idxs:
            r, c = floor_positions[k]
            terrain[r, c] = int(TileType.TRAP)

    # Pick a stair-up location (first floor tile, scan order).
    placed_up = False
    for r in range(1, mh - 1):
        for c in range(1, mw - 1):
            if terrain[r, c] == int(TileType.FLOOR):
                terrain[r, c] = int(TileType.STAIRCASE_UP)
                placed_up = True
                break
        if placed_up:
            break

    # Place stair-down on the bottom-right-most floor tile, except on the
    # deepest Gehennom level (Dlvl 16 = sanctum, no down).
    if int(depth) < 16:
        placed_dn = False
        for r in range(mh - 2, 0, -1):
            for c in range(mw - 2, 0, -1):
                if terrain[r, c] == int(TileType.FLOOR):
                    terrain[r, c] = int(TileType.STAIRCASE_DOWN)
                    placed_dn = True
                    break
            if placed_dn:
                break

    monster_type_ids = _find_demon_monster_ids()
    item_type_ids: list[int] = []
    return jnp.asarray(terrain, dtype=jnp.int8), monster_type_ids, item_type_ids


# ---------------------------------------------------------------------------
# Helper: write a BranchGraphState into a DungeonState's stair_links field.
# Used by tests + future env construction code.
# ---------------------------------------------------------------------------

def apply_branch_graph_to_dungeon(
    dungeon: DungeonState, graph: BranchGraphState
) -> DungeonState:
    """Project BranchGraphState.stair_links onto DungeonState.stair_links.

    The DungeonState schema is [N_BRANCHES, MAX_LEVELS, 2 (dir), 2 (endpoint)].
    BranchGraphState only tracks the down-stair endpoint per (branch, level);
    we infer the up-stair endpoint by reading the destination's link.

    Citation: vendor/nethack/src/dungeon.c::init_dungeons cross-wiring.
    """
    # graph.stair_links[src_branch, src_level-1] = [dst_branch, dst_level]
    # Treat this as the *down* link (direction=1).  For each populated entry
    # we also wire the symmetric *up* link at the destination.
    sl = dungeon.stair_links
    n_b = graph.stair_links.shape[0]
    n_l = graph.stair_links.shape[1]

    # Use Python loops here because this is non-JIT init-time code.
    for b in range(n_b):
        for lv in range(n_l):
            dst_b = int(graph.stair_links[b, lv, 0])
            dst_l = int(graph.stair_links[b, lv, 1])
            if dst_b < 0 or dst_l < 0:
                continue
            # Decide whether this link is the up or down direction:
            #   if dst is on a sibling branch (different parent path), the
            #   src side hosts the branch-entry: src goes "down" into dst.
            # Heuristic: if src has a parent and dst is its parent, it's an
            # up-link; otherwise treat as down.
            is_up_from_src = (
                graph.parent_branch[b] != jnp.int8(-1) and
                int(graph.parent_branch[b]) == dst_b
            )
            direction = 0 if bool(is_up_from_src) else 1
            sl = sl.at[b, lv, direction].set(
                jnp.array([dst_b, dst_l], dtype=jnp.int8)
            )

    # ---- Wave 5 Phase 2: Gehennom internal descents (L_n <-> L_{n+1}) ----
    # BranchGraphState's single-slot table can't encode both directions on
    # the same level, so we wire the 15 in-branch links directly.
    # Citation: vendor/nethack/dat/dungeon.lua "Gehennom" levels block.
    gehennom_levels = 16
    for lv in range(gehennom_levels - 1):
        # L(lv+1) down-stair -> L(lv+2)
        sl = sl.at[int(Branch.GEHENNOM), lv, 1].set(
            jnp.array([int(Branch.GEHENNOM), lv + 2], dtype=jnp.int8)
        )
        # L(lv+2) up-stair -> L(lv+1)
        sl = sl.at[int(Branch.GEHENNOM), lv + 1, 0].set(
            jnp.array([int(Branch.GEHENNOM), lv + 1], dtype=jnp.int8)
        )

    return dungeon.replace(stair_links=sl)


# ---------------------------------------------------------------------------
# TODO blocks
# ---------------------------------------------------------------------------
# Wave 4:
#   - traverse_stair: handle BR_PORTAL branches (Quest, Sokoban) which
#     use magic portals rather than physical staircases.
#   - enter_branch: lazy level generation on first visit.
#   - Add Vibrating Square portal logic for ENDGAME entry.
#   - Populate branch entrance positions on main dungeon map (random within
#     first_level ± tolerance, matching dungeon.c level_range()).
#
# Wave 5:
#   - VLAD's Tower: BR_NO_END2 means no down-ladder in Gehennom.
#   - ENDGAME: BR_NO_END1 means no stair from top — teleport to Astral Plane.
#   - Boulder/pit objects: migrate BOULDER_TILE sentinel into the proper
#     item layer once the boulder system is online.
#   - Proper Sokoban .des parser (sokoban*.des).
#   - 13 unique Quest layouts (Qst.des per role).
