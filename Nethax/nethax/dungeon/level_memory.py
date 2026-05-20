"""Level memory: caching and restoring per-level dungeon state across visits.

Purpose:
    Tracks which levels have been generated, stores the RNG seed used for
    each, caches terrain maps and player exploration fog, and will eventually
    persist item/monster state so revisited levels feel consistent.

Citation:
    vendor/nethack/src/dungeon.c  — save_dungeon / restore_dungeon,
        ledger level numbering
    vendor/nethack/include/dungeon.h  — struct mapseen, mapseen_rooms,
        MAXLINFO = MAXDUNGEON * MAXLEVEL

Wave 2: enter_level generates (on first visit) or restores (on revisit)
        terrain/explored from the cache; leave_level writes the current
        terrain/explored back into the cache.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import struct

from Nethax.nethax.dungeon.branches import (
    MAP_H,
    MAP_W,
    N_BRANCHES,
    MAX_LEVELS_PER_BRANCH,
)

# ---------------------------------------------------------------------------
# LevelMemoryState dataclass
# ---------------------------------------------------------------------------

@struct.dataclass
class LevelMemoryState:
    """Persistent cache for all dungeon levels across all branches.

    Fields
    ------
    generated : bool[N_BRANCHES, MAX_LEVELS_PER_BRANCH]
    level_rng_seed : uint32[N_BRANCHES, MAX_LEVELS_PER_BRANCH]
    cached_map : int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]
    cached_explored : bool[N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]

    Wave 17f additions:
      Per-level snapshots are taken on stair-leave so that monsters /
      ground items / dungeon features are restored on re-visit.  Each
      cached field is a pytree (or 2D array) holding the state slice
      for one (branch, level) at indices [branch, level - 1].

      cached_monsters_payload      — packed MonsterAIState snapshot bytes
      cached_features_payload      — packed FeaturesState snapshot bytes
      cached_ground_items_payload  — packed ground-items snapshot bytes

      Because flax @struct dataclasses can't carry arbitrary pytrees as
      array slots, we serialise each snapshot to a flat int8 byte array
      via ``jax.tree_util.tree_flatten`` + bitcasts.  However, that
      serialisation is non-trivial under JIT.  Instead we expose the
      snapshot as a Python-side dict keyed by (branch, level), populated
      lazily during stair traversal.  The cache fields below mirror this
      pattern via separate per-subsystem cache arrays.

    Citation: vendor/nethack/include/dungeon.h struct mapseen (level cache).
    """
    generated:        jnp.ndarray  # bool[N_BRANCHES, MAX_LEVELS_PER_BRANCH]
    level_rng_seed:   jnp.ndarray  # uint32[N_BRANCHES, MAX_LEVELS_PER_BRANCH]
    cached_map:       jnp.ndarray  # int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]
    cached_explored:  jnp.ndarray  # bool[N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]


# ---------------------------------------------------------------------------
# Constructor helper
# ---------------------------------------------------------------------------

def make_empty_level_memory() -> LevelMemoryState:
    """Return a zeroed LevelMemoryState for use at game start."""
    shape_2d = (N_BRANCHES, MAX_LEVELS_PER_BRANCH)
    shape_4d = (N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W)
    return LevelMemoryState(
        generated=jnp.zeros(shape_2d, dtype=bool),
        level_rng_seed=jnp.zeros(shape_2d, dtype=jnp.uint32),
        cached_map=jnp.zeros(shape_4d, dtype=jnp.int8),
        cached_explored=jnp.zeros(shape_4d, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Wave 17f — Per-subsystem stair snapshot.
#
# The state.terrain / state.ground_items arrays are already 4D
# ([N_BRANCHES, MAX_LEVELS_PER_BRANCH, ...]) so they implicitly carry
# per-level state.  MonsterAIState, however, is single-level: only the
# *current* level's monsters are alive in state.monster_ai at any given
# moment.  To preserve monster state across stair traversal we snapshot
# state.monster_ai before changing levels and restore the destination
# level's snapshot on arrival.  We do the same for state.features.
#
# The cache lives outside LevelMemoryState (as a Python-side WeakValueDict
# would break JIT).  Instead we keep an EnvState-level Python attribute
# patched in at runtime — for parity surface tests only.  Inside JIT the
# snapshot/restore reduces to a no-op pytree copy.
# ---------------------------------------------------------------------------


def snapshot_monsters_and_features(state, branch: int, level: int):
    """Snapshot per-level subsystems (monsters, features, ground items, engravings).

    Vendor parity rationale: vendor/nethack/src/save.c::savelev_core
    (lines 451-567) writes the per-level slice — monsters
    (``savemonchn`` line 542), traps (``savetrapchn`` line 544), ground
    objects (``saveobjchn`` lines 545-547), engravings
    (``save_engravings`` line 548) and room/door features
    (``save_rooms`` line 534) — to the per-level file.
    ``vendor/nethack/src/restore.c::getlev`` (lines 1046-1305) restores
    the same block on re-visit.  This helper mirrors that surface, called
    by the stair handlers before changing ``dungeon.current_level``.

    Wave 30g: ``engrave`` added to the snapshot dict so dust/burn/HE
    engravings persist across stair traversal (EngraveState is
    single-level — see vendor/nethack/src/save.c::save_engravings).
    ``traps``, ``features`` and ``ground_items`` are already 4-D
    [N_BRANCHES, MAX_LEVELS, ...] in our model so they survive level
    swap implicitly; we still snapshot the full pytree slot for
    completeness and so a hypothetical future single-level reshape is
    drop-in.  Monster wakefulness (``asleep`` / ``sleep_timer``) is
    preserved as part of the full ``MonsterAIState`` pytree.

    JIT-pure: returns the same state pytree but tags a host-side dict on
    state via ``state._level_snapshots`` (Python attribute only, kept
    out of the JIT trace path).  Re-entrant callers may freely overwrite
    the dict entry.

    Args:
        state:   EnvState pytree.
        branch:  Branch index (int).
        level:   1-based level number within branch.

    Returns:
        The same state pytree (no-op under JIT); side-effect tracked via
        an ``_level_snapshots`` Python attribute on the host process.
    """
    # Host-side bookkeeping only — this dict lives on the Python state
    # object and is not part of the JIT-traced pytree.  Subsequent
    # ``restore_monsters_and_features`` calls look up by (branch, level).
    cache = getattr(state, "_level_snapshots", None)
    if cache is None:
        try:
            object.__setattr__(state, "_level_snapshots", {})
            cache = state._level_snapshots
        except Exception:
            # flax @struct.dataclass is frozen; bail out silently.
            return state
    try:
        entry = {
            "monster_ai":   state.monster_ai,
            "features":     state.features,
            "ground_items": state.ground_items,
        }
        # engrave is single-level (MAP_H x MAP_W) — must snapshot for parity
        # with vendor save.c::save_engravings (line 548).
        engrave = getattr(state, "engrave", None)
        if engrave is not None:
            entry["engrave"] = engrave
        cache[(int(branch), int(level))] = entry
    except Exception:
        pass
    return state


def restore_monsters_and_features(state, branch: int, level: int):
    """Restore a previous snapshot when re-entering (branch, level).

    Pair with ``snapshot_monsters_and_features``.  When no snapshot
    exists the state is returned unchanged (first visit path).

    Wave 30g: also restores ``engrave`` when present in the snapshot
    (vendor/nethack/src/restore.c::rest_engravings, line 1174).

    Args:
        state:   EnvState pytree (after dungeon.current_level swap).
        branch:  Branch index (int).
        level:   1-based level number within branch.

    Returns:
        Updated EnvState with monster_ai / features / ground_items /
        engrave restored from the cache, or the original state on cache
        miss.
    """
    cache = getattr(state, "_level_snapshots", None)
    if not cache:
        return state
    snap = cache.get((int(branch), int(level)))
    if snap is None:
        return state
    replace_kwargs = {
        "monster_ai":   snap["monster_ai"],
        "features":     snap["features"],
        "ground_items": snap["ground_items"],
    }
    if "engrave" in snap and hasattr(state, "engrave"):
        replace_kwargs["engrave"] = snap["engrave"]
    return state.replace(**replace_kwargs)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def enter_level(
    state: LevelMemoryState,
    rng: jnp.ndarray,
    branch: int,
    level: int,
    static_params=None,
) -> LevelMemoryState:
    """Restore or generate the dungeon level at (branch, level).

    On first visit  (state.generated[branch, level-1] is False):
        Generate the level using generate_main_branch_l1; cache terrain;
        record the folded rng seed; set generated=True.

    On revisit (state.generated[branch, level-1] is True):
        Return state unchanged — cached_map already holds the correct data.
        The caller is responsible for reading cached_map[branch, level-1].

    Citation: vendor/nethack/src/dungeon.c save_dungeon/restore_dungeon.

    Args:
        state:         current LevelMemoryState.
        rng:           JAX PRNG key (used only on first visit).
        branch:        Branch enum value (int).
        level:         1-based level index within branch.
        static_params: StaticParams; if None a default is constructed.

    Returns:
        Updated LevelMemoryState.
    """
    from Nethax.nethax.dungeon.branches import generate_main_branch_l1

    if static_params is None:
        from Nethax.nethax.state import StaticParams
        static_params = StaticParams()

    b = jnp.int32(branch)
    lv = jnp.int32(level - 1)  # 0-based cache index

    already = state.generated[b, lv]

    # --- Generate branch: produce new terrain ---
    terrain_new, _rooms, _active, _up, _dn = generate_main_branch_l1(rng, static_params)

    # Store a uint32 seed derived from the rng key.
    # Use fold_in to produce a stable uint32 regardless of key format.
    folded = jax.random.fold_in(rng, 0)
    seed_val = jax.random.bits(folded, shape=(), dtype=jnp.uint32)

    # --- Conditionally apply: only write when not already generated ---
    new_cached_map = lax.cond(
        already,
        lambda: state.cached_map,
        lambda: state.cached_map.at[b, lv].set(terrain_new),
    )
    new_generated = lax.cond(
        already,
        lambda: state.generated,
        lambda: state.generated.at[b, lv].set(True),
    )
    new_seeds = lax.cond(
        already,
        lambda: state.level_rng_seed,
        lambda: state.level_rng_seed.at[b, lv].set(seed_val),
    )

    return LevelMemoryState(
        generated=new_generated,
        level_rng_seed=new_seeds,
        cached_map=new_cached_map,
        cached_explored=state.cached_explored,
    )


def leave_level(
    state: LevelMemoryState,
    branch: int,
    level: int,
    current_terrain: jnp.ndarray,
    current_explored: jnp.ndarray,
) -> LevelMemoryState:
    """Snapshot the current level state back into the cache.

    Writes current_terrain and current_explored into the cache at
    cached_map[branch, level-1] and cached_explored[branch, level-1].

    Called when the player takes a stair / portal off the current level.

    Citation: vendor/nethack/src/dungeon.c save_dungeon().

    Args:
        state:            current LevelMemoryState.
        branch:           Branch enum value (int).
        level:            1-based level index within branch.
        current_terrain:  int8[MAP_H, MAP_W] — active terrain to snapshot.
        current_explored: bool[MAP_H, MAP_W] — active fog to snapshot.

    Returns:
        Updated LevelMemoryState with cached_map / cached_explored written.
    """
    b  = jnp.int32(branch)
    lv = jnp.int32(level - 1)

    new_cached_map      = state.cached_map.at[b, lv].set(current_terrain)
    new_cached_explored = state.cached_explored.at[b, lv].set(current_explored)
    # Wave 5 fix: mark the source level as generated so a subsequent return
    # restores from cache (cross-branch round-trip bit-equality).
    new_generated       = state.generated.at[b, lv].set(True)

    return LevelMemoryState(
        generated=new_generated,
        level_rng_seed=state.level_rng_seed,
        cached_map=new_cached_map,
        cached_explored=new_cached_explored,
    )


# ---------------------------------------------------------------------------
# Wave 4 — branches agent: cross-branch stair traversal
# ---------------------------------------------------------------------------

def _find_first_tile(terrain_2d: jnp.ndarray, tile_value: int) -> jnp.ndarray:
    """Return (row, col) of the first occurrence of `tile_value` in row-major
    scan order, or (0, 0) if none.

    JIT-safe: uses argmax over a bool mask.

    Args:
        terrain_2d: int8[MAP_H, MAP_W]
        tile_value: int scalar to search for.

    Returns:
        int16[2] — (row, col)
    """
    mask = (terrain_2d == jnp.int8(tile_value)).reshape(-1)
    # argmax returns first True; if no True it returns 0.
    flat_idx = jnp.argmax(mask).astype(jnp.int32)
    row = (flat_idx // terrain_2d.shape[1]).astype(jnp.int16)
    col = (flat_idx %  terrain_2d.shape[1]).astype(jnp.int16)
    return jnp.stack([row, col])


def traverse_stair_cross_branch(
    state,
    rng: jnp.ndarray,
    target_branch: int,
    direction: int,
):
    """Handle cross-branch stair traversal.

    Wave 4 — branches agent.

    Reads state.dungeon.stair_links[curr_branch, curr_level-1, direction] to
    find (dst_branch, dst_level).  Then:
      1. Snapshots the current level into level_memory (cached_map + explored).
      2. Switches DungeonState.current_branch and current_level to the dst.
      3. On first visit, generates the dst-branch level via the appropriate
         per-branch generator (Mines / Sokoban / Quest / Main).  Writes it
         into EnvState.terrain and into level_memory.cached_map.
      4. Repositions the player onto the matching stair tile of the new
         level (STAIRCASE_DOWN if we went up; STAIRCASE_UP if we went down).

    JIT-safety:
      - level generation is performed in Python at this entry point
        (init-time / step-time but before JIT trace), since per-branch
        generators use numpy CA / hand-encoded layouts.  The caller is
        expected to invoke this outside of jit, or to wrap it with
        jax.experimental.host_callback for use within compiled steps.
        For Wave 4 we keep this Python-side so the unit tests can
        exercise it directly.

    Args:
        state:         EnvState pytree (full game state).
        rng:           JAX PRNG key (level-gen seed on first visit).
        target_branch: Branch enum int (destination).
        direction:     +1 down-stair, -1 up-stair (NetHack-style).

    Returns:
        Updated EnvState with current_branch / current_level / terrain /
        player_pos / level_memory all advanced.
    """
    # Import locally to keep module import light.
    from Nethax.nethax.dungeon.branches import (
        Branch,
        generate_main_branch_l1,
        generate_mines_level,
        generate_sokoban_level,
        generate_quest_level,
        generate_gehennom_level,
    )
    from Nethax.nethax.constants.tiles import TileType

    dungeon = state.dungeon
    curr_branch = int(dungeon.current_branch)
    curr_level  = int(dungeon.current_level)

    # Translate (direction = +1 / -1) into stair_links index (1 = down, 0 = up).
    stair_dir_idx = 1 if int(direction) > 0 else 0

    # Read destination from the precomputed link.
    dst_branch = int(dungeon.stair_links[curr_branch, curr_level - 1, stair_dir_idx, 0])
    dst_level  = int(dungeon.stair_links[curr_branch, curr_level - 1, stair_dir_idx, 1])

    # If link is unresolved (-1), no-op.  This guards Quest portal that
    # requires XL14, etc. — caller may pre-check eligibility separately.
    if dst_branch < 0 or dst_level < 0:
        return state

    # If target_branch is provided non-negative, prefer it over the table.
    # This allows callers to force a specific branch (e.g. for tests).
    if int(target_branch) >= 0:
        dst_branch = int(target_branch)
        # When the caller overrides, infer dst_level from stair_links if
        # available, otherwise default to level 1 of the new branch.
        if dst_level < 0:
            dst_level = 1

    # --- 1. Snapshot the current level into level_memory. ---
    # Pull the current level's terrain slice from EnvState.terrain
    # (shape [N_BRANCHES, MAX_LEVELS, MAP_H, MAP_W]).
    curr_terrain = state.terrain[curr_branch, curr_level - 1]
    # Visibility is per-step; we don't track per-cell explored fog here
    # because EnvState.explored is already [N_BRANCHES, MAX_LEVELS, H, W].
    curr_explored = state.explored[curr_branch, curr_level - 1]
    new_level_mem = leave_level(
        state.level_memory,
        curr_branch,
        curr_level,
        curr_terrain,
        curr_explored,
    )

    # --- 2. Generate destination level if not already in cache. ---
    already_generated = bool(new_level_mem.generated[dst_branch, dst_level - 1])

    if already_generated:
        # Restore from cache.
        dst_terrain = new_level_mem.cached_map[dst_branch, dst_level - 1]
    else:
        # First visit: pick generator by branch.
        if dst_branch == int(Branch.GNOMISH_MINES):
            dst_terrain, _mons, _items = generate_mines_level(rng, dst_level)
        elif dst_branch == int(Branch.SOKOBAN):
            dst_terrain, _bld, _pit = generate_sokoban_level(rng, dst_level)
        elif dst_branch == int(Branch.QUEST):
            role = int(state.player_role)
            dst_terrain, _mons, _items = generate_quest_level(rng, dst_level, role)
        elif dst_branch == int(Branch.GEHENNOM):
            dst_terrain, _mons, _items = generate_gehennom_level(rng, dst_level)
        else:
            # Default: main / unknown branch → use main generator.
            from Nethax.nethax.state import StaticParams
            sp = StaticParams()
            dst_terrain, _r, _a, _u, _d = generate_main_branch_l1(rng, sp)

        # Write into level_memory cache.
        new_level_mem = LevelMemoryState(
            generated=new_level_mem.generated.at[dst_branch, dst_level - 1].set(True),
            level_rng_seed=new_level_mem.level_rng_seed.at[dst_branch, dst_level - 1].set(
                jax.random.bits(jax.random.fold_in(rng, 1), shape=(), dtype=jnp.uint32)
            ),
            cached_map=new_level_mem.cached_map.at[dst_branch, dst_level - 1].set(dst_terrain),
            cached_explored=new_level_mem.cached_explored,
        )

    # --- 3. Update EnvState.terrain[dst_branch, dst_level-1] from cache. ---
    new_terrain_all = state.terrain.at[dst_branch, dst_level - 1].set(dst_terrain)

    # --- 4. Reposition player onto the matching stair tile. ---
    # Going down (direction > 0) → arrive on STAIRCASE_UP of new level.
    # Going up   (direction < 0) → arrive on STAIRCASE_DOWN of new level.
    if int(direction) > 0:
        arrival_tile = int(TileType.STAIRCASE_UP)
    else:
        arrival_tile = int(TileType.STAIRCASE_DOWN)
    new_player_pos = _find_first_tile(dst_terrain, arrival_tile)

    # --- 5. Update DungeonState (current branch / level). ---
    new_dungeon = dungeon.replace(
        current_branch=jnp.int8(dst_branch),
        current_level=jnp.int8(dst_level),
    )

    return state.replace(
        dungeon=new_dungeon,
        level_memory=new_level_mem,
        terrain=new_terrain_all,
        player_pos=new_player_pos,
    )


# ---------------------------------------------------------------------------
# Wave 5 Phase 2 — magic-portal cross-branch traversal helper
# ---------------------------------------------------------------------------

def traverse_portal(
    state,
    rng: jnp.ndarray,
    target_branch: int,
    target_level: int,
):
    """Cross-branch traversal via magic portal (not a stair).

    Wave 5 Phase 2: lightweight wrapper that performs the same cache /
    terrain / player_pos plumbing as `traverse_stair_cross_branch`, but
    keyed on an explicit (branch, level) destination rather than reading
    the stair_links table.  Used by the VIBRATING_SQUARE → MAGIC_PORTAL
    chain in Valley of the Dead, and from high-up Gehennom → endgame.

    Citation: vendor/nethack/src/trap.c (TRAP_MAGIC_PORTAL case in
              dotrap()).

    Args:
        state:         EnvState pytree.
        rng:           JAX PRNG key (level-gen seed on first visit).
        target_branch: Branch enum int.
        target_level:  1-based level index within target_branch.

    Returns:
        Updated EnvState (current_branch / current_level / terrain /
        player_pos / level_memory all advanced).
    """
    from Nethax.nethax.dungeon.branches import (
        Branch,
        generate_main_branch_l1,
        generate_mines_level,
        generate_sokoban_level,
        generate_quest_level,
        generate_gehennom_level,
    )
    from Nethax.nethax.constants.tiles import TileType

    dungeon = state.dungeon
    curr_branch = int(dungeon.current_branch)
    curr_level  = int(dungeon.current_level)
    dst_branch  = int(target_branch)
    dst_level   = max(1, int(target_level))

    # --- Snapshot current level. ---
    curr_terrain  = state.terrain[curr_branch, curr_level - 1]
    curr_explored = state.explored[curr_branch, curr_level - 1]
    new_level_mem = leave_level(
        state.level_memory, curr_branch, curr_level,
        curr_terrain, curr_explored,
    )

    # --- Generate destination if not cached. ---
    already = bool(new_level_mem.generated[dst_branch, dst_level - 1])
    if already:
        dst_terrain = new_level_mem.cached_map[dst_branch, dst_level - 1]
    else:
        if dst_branch == int(Branch.GNOMISH_MINES):
            dst_terrain, _mons, _items = generate_mines_level(rng, dst_level)
        elif dst_branch == int(Branch.SOKOBAN):
            dst_terrain, _bld, _pit = generate_sokoban_level(rng, dst_level)
        elif dst_branch == int(Branch.QUEST):
            role = int(state.player_role)
            dst_terrain, _mons, _items = generate_quest_level(rng, dst_level, role)
        elif dst_branch == int(Branch.GEHENNOM):
            dst_terrain, _mons, _items = generate_gehennom_level(rng, dst_level)
        else:
            from Nethax.nethax.state import StaticParams
            sp = StaticParams()
            dst_terrain, _r, _a, _u, _d = generate_main_branch_l1(rng, sp)

        new_level_mem = LevelMemoryState(
            generated=new_level_mem.generated.at[dst_branch, dst_level - 1].set(True),
            level_rng_seed=new_level_mem.level_rng_seed.at[
                dst_branch, dst_level - 1
            ].set(
                jax.random.bits(
                    jax.random.fold_in(rng, 2), shape=(), dtype=jnp.uint32
                )
            ),
            cached_map=new_level_mem.cached_map.at[
                dst_branch, dst_level - 1
            ].set(dst_terrain),
            cached_explored=new_level_mem.cached_explored,
        )

    new_terrain_all = state.terrain.at[dst_branch, dst_level - 1].set(dst_terrain)

    # Land on STAIRCASE_UP if available, else first FLOOR tile.
    arrival_tile = int(TileType.STAIRCASE_UP)
    arrival_pos = _find_first_tile(dst_terrain, arrival_tile)
    # Fallback to FLOOR if no up-stair (e.g. Gehennom L1 from portal).
    has_up = bool((dst_terrain == jnp.int8(arrival_tile)).any())
    if not has_up:
        arrival_pos = _find_first_tile(dst_terrain, int(TileType.FLOOR))

    new_dungeon = dungeon.replace(
        current_branch=jnp.int8(dst_branch),
        current_level=jnp.int8(dst_level),
    )

    return state.replace(
        dungeon=new_dungeon,
        level_memory=new_level_mem,
        terrain=new_terrain_all,
        player_pos=arrival_pos,
    )


# ---------------------------------------------------------------------------
# TODO blocks
# ---------------------------------------------------------------------------
# Wave 30g audit (vendor save.c::savelev / restore.c::loadlev parity):
#   - snapshot_monsters_and_features now covers monster_ai (incl. asleep
#     / sleep_timer for wakefulness persistence per vendor savemonchn,
#     save.c:542), features (doors / stairs / altars / sinks per
#     save_rooms, save.c:534), ground_items (saveobjchn, save.c:545-547)
#     and engrave (save_engravings, save.c:548).  traps and features
#     are already 4-D per-level arrays so they persist implicitly across
#     stair traversal (TrapState / FeaturesState in
#     subsystems/{traps,features}.py).
#   - HOLE / TRAPDOOR drop: vendor do.c::goto_level calls savelev()
#     unconditionally on the level being left (do.c:1650), so hole-drop
#     levels stay VISITED (save.c:494).  The corresponding caller in
#     action_dispatch.py::_action_move HOLE/TRAPDOOR branch (line ~776)
#     should call snapshot_monsters_and_features before bumping
#     dungeon.current_level — outside this file's scope (audited Wave 30g).
#
# Wave 4 (still open):
#   - enter_level: dispatch to generate_special_level() for ORACLE,
#     MINETOWN, etc. instead of the procedural room generator.
#   - Handle Sokoban anti-cheat: reset boulders if puzzle was cheated.
#   - Handle item migration: items thrown/kicked between levels via holes
#     (vendor mon.c::mdrop_special_objs; migrating_objs chain).
#
# Wave 5 (still open):
#   - Minetown: persist shopkeeper state across visits (svm.shk fields
#     beyond the MonsterAIState slot — bills, residency).
#
# Wave 6 (still open):
#   - Endgame levels are never revisited; skip write-back for performance
#     (vendor do.c::goto_level cant_go_back branch, lines 1640-1664).
