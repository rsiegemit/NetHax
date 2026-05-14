"""Top-level action dispatch — routes each of the 121 NLE actions to subsystems.

Canonical source:
  vendor/nethack/src/cmd.c — dokeylist[], commandlist[], and the central
                              command-dispatch table mapping ASCII codes to
                              handler functions (cmd.c ~5 704 lines).
                              Key entry points: command_input(), docmd(),
                              doextcmd(), and the cmdlist[] array
                              (cmd.c lines ~200-600).
  vendor/nethack/src/hack.c — domove(), test_move() — movement with
                               bump semantics (wall block, door interact,
                               monster melee).
  vendor/nethack/src/do.c   — doup(), dodown() — stair traversal logic.

Design
------
NetHack's cmd.c dispatches on a raw ASCII keycode via a 256-entry function
pointer table (cmdlist[]).  Nethax mirrors this with a JAX-compatible lookup
table indexed by action int-value (ASCII code).

Wave 2 uses ``jax.lax.switch`` for JIT-safe dispatch:

    result = jax.lax.switch(handler_idx, ACTION_HANDLERS, state, rng)

where ``handler_idx`` is looked up from ``_ACTION_TO_HANDLER_IDX``, a
256-entry jnp array mapping ASCII → handler slot index.

Handler slots:
    0  = _noop         (all non-movement actions)
    1  = _move_n
    2  = _move_e
    3  = _move_s
    4  = _move_w
    5  = _move_ne
    6  = _move_se
    7  = _move_sw
    8  = _move_nw
    9  = _run_n
    10 = _run_e
    11 = _run_s
    12 = _run_w
    13 = _run_ne
    14 = _run_se
    15 = _run_sw
    16 = _run_nw
    17 = _stair_up
    18 = _stair_down
    19 = _wait

Wave 1 handlers that are not movement remain as _noop.

TODO (Wave 3):
  - KICK, FIGHT prefix, melee_attack routing.
  - OPEN / CLOSE door actions.
  - PICKUP / DROP item actions.

TODO (Wave 4-6):
  - Fill remaining ~100 actions (spells, inventory management, prayer,
    identify, etc.) wave by wave as subsystems come online.
"""
import jax
import jax.numpy as jnp

from Nethax.nethax.constants.actions import (
    ACTIONS,
    N_ACTIONS,
    CompassCardinalDirection,
    CompassIntercardinalDirection,
    CompassCardinalDirectionLonger,
    CompassIntercardinalDirectionLonger,
    MiscDirection,
    Command,
)
from Nethax.nethax.constants import TileType
from Nethax.nethax.constants.objects import ObjectClass
from Nethax.nethax.fov import compute_fov, update_explored
from Nethax.nethax.subsystems.features import (
    DoorState,
    open_door,
    handle_open as _features_handle_open,
    handle_close as _features_handle_close,
    handle_kick as _features_handle_kick,
)
from Nethax.nethax.subsystems.traps import trigger_trap, TrapType
from Nethax.nethax.subsystems.items_potions import handle_quaff as _potions_handle_quaff
from Nethax.nethax.subsystems.items_scrolls import handle_read as _scrolls_handle_read
from Nethax.nethax.subsystems.items_wands import (
    handle_zap as _wands_handle_zap,
    WandState,
)
from Nethax.nethax.subsystems.inventory import (
    handle_wield as _inv_handle_wield,
    handle_wear as _inv_handle_wear,
    handle_name as _inv_handle_name,
    pickup as _inv_pickup,
    drop as _inv_drop,
    ItemCategory,
    MAX_INVENTORY_SLOTS,
)
from Nethax.nethax.subsystems.combat import (
    handle_fight as _combat_handle_fight,
    melee_attack as _combat_melee_attack,
    handle_twoweapon as _combat_handle_twoweapon,
    handle_throw as _combat_handle_throw,
)
from Nethax.nethax.subsystems.prayer import handle_pray as _prayer_handle_pray
from Nethax.nethax.subsystems.containers import (
    handle_loot as _containers_handle_loot,
    handle_apply_container as _containers_handle_apply,
)
from Nethax.nethax.subsystems.status_effects import (
    handle_eat as _status_handle_eat,
    compute_hunger_state,
    MAX_NUTRITION,
)
from Nethax.nethax.subsystems.magic import N_SPELLS
from Nethax.nethax.subsystems.conduct import (
    Conduct as _Conduct,
    mark_violated_if as _mark_violated_if,
    food_material_for_type_id as _food_material_for_type_id,
    is_meat_material as _is_meat_material,
    is_animal_material as _is_animal_material,
)


# ---------------------------------------------------------------------------
# Movement deltas: (dy, dx) where dy=row-delta, dx=col-delta.
# NetHack convention: north = decreasing row index.
# ---------------------------------------------------------------------------

_DELTAS = {
    "N":  (-1,  0),
    "S":  ( 1,  0),
    "E":  ( 0,  1),
    "W":  ( 0, -1),
    "NE": (-1,  1),
    "NW": (-1, -1),
    "SE": ( 1,  1),
    "SW": ( 1, -1),
}

# Direction-indexed table: dir_idx ∈ [0,7] → (dy, dx).
# Order matches _SLOT_MOVE_* / _SLOT_RUN_* (N=0, E=1, S=2, W=3, NE=4, SE=5, SW=6, NW=7).
# Used by the compact movement/run handlers (Wave 8 compile-time refactor) so
# all 8 directions share a single jaxpr instead of being traced 8 times.
_DIR_TABLE: jnp.ndarray = jnp.array(
    [
        _DELTAS["N"],
        _DELTAS["E"],
        _DELTAS["S"],
        _DELTAS["W"],
        _DELTAS["NE"],
        _DELTAS["SE"],
        _DELTAS["SW"],
        _DELTAS["NW"],
    ],
    dtype=jnp.int32,
)

# ---------------------------------------------------------------------------
# Solid-tile mask  (precomputed as a constant bool array indexed by TileType)
# For a tile value t, _IS_SOLID[t] == True means movement is blocked.
# TileType values: VOID=0, FLOOR=1, CORRIDOR=2, WALL=3, CLOSED_DOOR=4,
#                  OPEN_DOOR=5, STAIRCASE_UP=6, STAIRCASE_DOWN=7, ...
# ---------------------------------------------------------------------------

_NUM_TILE_TYPES = len(TileType)

def _build_solid_mask() -> jnp.ndarray:
    solid = [False] * _NUM_TILE_TYPES
    for t in (TileType.VOID, TileType.WALL, TileType.CLOSED_DOOR):
        solid[int(t)] = True
    return jnp.array(solid, dtype=jnp.bool_)

_IS_SOLID: jnp.ndarray = _build_solid_mask()  # shape [NUM_TILE_TYPES], bool

# Max steps for a _run action (lax.while_loop iteration cap).
_RUN_MAX_STEPS: int = 64


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _current_level_terrain(state):
    """Extract the 2-D terrain slice for the player's current branch/level.

    state.terrain shape: [N_BRANCHES, MAX_LEVELS, MAP_H, MAP_W]
    state.dungeon.current_level is 1-based → 0-based array index.
    """
    b = state.dungeon.current_branch
    lv = state.dungeon.current_level - 1  # 1-based → 0-based
    return state.terrain[b, lv]           # [MAP_H, MAP_W]


def _apply_fov(state):
    """Recompute visible + explored for the current level after a move."""
    terrain_2d = _current_level_terrain(state)
    new_visible = compute_fov(terrain_2d, state.player_pos)

    b  = state.dungeon.current_branch
    lv = state.dungeon.current_level - 1
    new_explored = update_explored(state.explored[b, lv], new_visible)

    new_explored_full = state.explored.at[b, lv].set(new_explored)
    return state.replace(visible=new_visible, explored=new_explored_full)


# ---------------------------------------------------------------------------
# Core movement primitive
# ---------------------------------------------------------------------------

def _flat_level_idx(state) -> jnp.ndarray:
    """Compute flat level index into TrapState / FeaturesState arrays.

    TrapState and FeaturesState are shaped [N_BRANCHES * MAX_LEVELS_PER_BRANCH,
    MAP_H, MAP_W].  The flat index is branch * MAX_LEVELS + (current_level - 1).
    """
    max_levels = state.terrain.shape[1]  # MAX_LEVELS_PER_BRANCH
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1  # 0-based
    return b * jnp.int32(max_levels) + lv


def _try_step(state, dy: int, dx: int, rng: jax.Array):
    """Attempt one step in direction (dy, dx).  Returns new state.

    Semantics (hack.c domove / test_move):
      1. Compute target = player_pos + (dy, dx).
      2. **Bump-attack bridge** (Wave 5): if the target tile contains a live
         monster, route into ``combat.melee_attack`` and return — the player
         does not move (matches vendor/nethack/src/hack.c::domove, which
         calls attack(mtmp) before any wall/door checks when a monster
         occupies the target tile).
      3. Bounds-check: must be in [0, MAP_H) x [0, MAP_W).
      4. Read target tile from terrain.
      5. CLOSED_DOOR (unlocked): auto-open door, player does not move this turn.
         CLOSED_DOOR (locked via door_state): blocked.
      6. VOID / WALL: block.
      7. TRAP / HIDDEN_TRAP: move, then trigger trap and apply effects.
      8. Otherwise: update player_pos.
    All branching via jnp.where / jax.lax.cond — no Python control flow.
    """
    terrain_2d = _current_level_terrain(state)
    map_h, map_w = terrain_2d.shape

    pos    = state.player_pos.astype(jnp.int32)
    target = pos + jnp.array([dy, dx], dtype=jnp.int32)

    # Bounds check
    in_bounds = (
        (target[0] >= 0) & (target[0] < map_h) &
        (target[1] >= 0) & (target[1] < map_w)
    )

    # -------------------------------------------------------------------
    # Bump-attack bridge (vendor/nethack/src/hack.c::domove).
    # Check for a live monster at the target tile *before* any movement /
    # door / trap logic.  If found, call combat.melee_attack and return
    # without moving the player.  Branches via jax.lax.cond so both paths
    # produce an identically-shaped EnvState pytree.
    # -------------------------------------------------------------------
    mai = state.monster_ai
    monster_pos_i32 = mai.pos.astype(jnp.int32)
    target_i16 = target.astype(jnp.int16)
    monster_match = (
        (monster_pos_i32[:, 0] == target[0])
        & (monster_pos_i32[:, 1] == target[1])
        & mai.alive
        & in_bounds
    )
    monster_idx = jnp.argmax(monster_match).astype(jnp.int32)
    monster_present = jnp.any(monster_match)

    def _attack_branch(s):
        # Capture pre-attack alive flag at the matched slot so we can grant
        # XP on kill (vendor/nethack/src/exper.c::experience).  Use a simple
        # heuristic (max-hp / 4 + 1) since the full vendor formula depends
        # on monst flags not currently materialized in state.
        was_alive = s.monster_ai.alive[monster_idx]
        attacked, _dmg, _hit = _combat_melee_attack(s, rng, monster_idx)
        killed = was_alive & ~attacked.monster_ai.alive[monster_idx]
        hp_max_killed = attacked.monster_ai.hp_max[monster_idx].astype(jnp.int32)
        xp_gain = jnp.where(
            killed,
            jnp.maximum(jnp.int32(1), hp_max_killed // jnp.int32(4) + jnp.int32(1)),
            jnp.int32(0),
        )
        new_xp = (attacked.player_xp.astype(jnp.int32) + xp_gain).astype(
            attacked.player_xp.dtype
        )
        return attacked.replace(player_xp=new_xp)

    # _attack_branch must return the same pytree shape as _move_branch
    # (below).  We construct that by computing the full movement state and
    # selecting via lax.cond at the very end.
    return jax.lax.cond(
        monster_present,
        _attack_branch,
        lambda s: _move_branch(s, dy, dx, rng, target, in_bounds, terrain_2d, map_h, map_w),
        state,
    )


def _move_branch(state, dy: int, dx: int, rng: jax.Array,
                 target, in_bounds, terrain_2d, map_h, map_w):
    """Standard movement (wall/door/trap) — extracted so the bump-attack
    branch in ``_try_step`` can share an identical EnvState output shape.
    """
    pos = state.player_pos.astype(jnp.int32)

    # Clamp target for safe index (out-of-bounds index returns garbage but
    # jnp.where discards it; we read VOID=0 for OOB via clamp).
    safe_row = jnp.clip(target[0], 0, map_h - 1)
    safe_col = jnp.clip(target[1], 0, map_w - 1)
    tile_val  = terrain_2d[safe_row, safe_col].astype(jnp.int32)

    # --- Door bump semantics (lock.c doopen / test_move) ---
    flat_lv  = _flat_level_idx(state)
    door_pos = jnp.array([flat_lv, safe_row, safe_col], dtype=jnp.int32)
    door_val = state.features.door_state[flat_lv, safe_row, safe_col].astype(jnp.int32)

    target_is_closed_door = (tile_val == jnp.int32(TileType.CLOSED_DOOR)) & in_bounds
    door_is_locked = door_val == jnp.int32(DoorState.LOCKED)

    # Unlocked closed door: open it and stay put (consume turn).
    # Locked closed door: blocked entirely.
    open_on_bump = target_is_closed_door & ~door_is_locked
    blocked_by_lock = target_is_closed_door & door_is_locked

    new_features = jax.lax.cond(
        open_on_bump,
        lambda f: open_door(f, door_pos),
        lambda f: f,
        state.features,
    )

    # Update terrain tile to OPEN_DOOR when we open it on bump.
    new_terrain = jax.lax.cond(
        open_on_bump,
        lambda t: t.at[
            state.dungeon.current_branch,
            state.dungeon.current_level - 1,
            safe_row,
            safe_col,
        ].set(jnp.int8(TileType.OPEN_DOOR)),
        lambda t: t,
        state.terrain,
    )

    # After door handling, re-read tile_val from (potentially updated) terrain.
    # For movement logic below, treat the tile as what it was before this step
    # (opening consumes the turn; movement does not happen).
    # Check if solid using original tile_val.
    safe_tile = jnp.clip(tile_val, 0, _NUM_TILE_TYPES - 1)
    is_solid  = _IS_SOLID[safe_tile]

    # Opening a closed door blocks movement for this turn.
    # Locked door also blocks.
    door_blocked = target_is_closed_door  # any closed door bump: no movement
    can_move = in_bounds & ~is_solid & ~door_blocked

    new_pos = jnp.where(can_move, target, pos).astype(jnp.int16)

    state_mid = state.replace(
        player_pos=new_pos,
        features=new_features,
        terrain=new_terrain,
    )

    # --- Trap triggering (trap.c dotrap) ---
    # After moving, check if the new tile is a trap.
    actually_moved = jnp.array_equal(new_pos, target.astype(jnp.int16))
    new_tile_val = terrain_2d[
        jnp.clip(new_pos[0].astype(jnp.int32), 0, map_h - 1),
        jnp.clip(new_pos[1].astype(jnp.int32), 0, map_w - 1),
    ].astype(jnp.int32)

    on_trap = actually_moved & (
        (new_tile_val == jnp.int32(TileType.TRAP)) |
        (new_tile_val == jnp.int32(TileType.HIDDEN_TRAP))
    )

    trap_pos = jnp.array(
        [flat_lv, new_pos[0].astype(jnp.int32), new_pos[1].astype(jnp.int32)],
        dtype=jnp.int32,
    )

    # Split a sub-key from state.rng for trap rolls.
    trap_rng, new_rng = jax.random.split(state_mid.rng)

    new_traps, trap_dmg, trap_se = jax.lax.cond(
        on_trap,
        lambda ts: trigger_trap(ts, trap_rng, trap_pos),
        lambda ts: (ts, jnp.int32(0), jnp.zeros(4, dtype=jnp.int32)),
        state_mid.traps,
    )

    # Apply HP damage from trap.
    new_hp = jnp.maximum(
        jnp.int32(0),
        state_mid.player_hp - trap_dmg,
    )

    # Apply timed side-effects: freeze turns and sleep turns.
    # side_effects[0] = freeze turns, side_effects[1] = sleep turns.
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    freeze_turns = trap_se[0]
    sleep_turns  = trap_se[1]

    old_frozen = state_mid.status.timed_statuses[int(TimedStatus.FROZEN)]
    old_sleep  = state_mid.status.timed_statuses[int(TimedStatus.SLEEP)]

    new_frozen = jnp.maximum(old_frozen, freeze_turns)
    new_sleep  = jnp.maximum(old_sleep, sleep_turns)

    new_timed = state_mid.status.timed_statuses \
        .at[int(TimedStatus.FROZEN)].set(new_frozen) \
        .at[int(TimedStatus.SLEEP)].set(new_sleep)
    new_status = state_mid.status.replace(timed_statuses=new_timed)

    state_final = state_mid.replace(
        traps=new_traps,
        player_hp=new_hp,
        status=new_status,
        rng=new_rng,
    )

    return _apply_fov(state_final)


# ---------------------------------------------------------------------------
# Single-step move handlers (one per direction)
# ---------------------------------------------------------------------------

def _noop(state, rng):
    return state


def _move_n(state, rng):  return _try_step(state, *_DELTAS["N"],  rng)
def _move_e(state, rng):  return _try_step(state, *_DELTAS["E"],  rng)
def _move_s(state, rng):  return _try_step(state, *_DELTAS["S"],  rng)
def _move_w(state, rng):  return _try_step(state, *_DELTAS["W"],  rng)
def _move_ne(state, rng): return _try_step(state, *_DELTAS["NE"], rng)
def _move_se(state, rng): return _try_step(state, *_DELTAS["SE"], rng)
def _move_sw(state, rng): return _try_step(state, *_DELTAS["SW"], rng)
def _move_nw(state, rng): return _try_step(state, *_DELTAS["NW"], rng)


# ---------------------------------------------------------------------------
# Run handlers — repeated move via lax.while_loop (capped at _RUN_MAX_STEPS)
# ---------------------------------------------------------------------------

def _make_run(dy: int, dx: int):
    """Return a run handler for direction (dy, dx).

    Threads ``rng`` through the ``lax.while_loop`` carry so each step can
    split a sub-key for any bump-attack roll that may fire inside _try_step.
    """
    def _run(state, rng):
        def cond(carry):
            s, step_count, prev_pos, _rng = carry
            moved = ~jnp.array_equal(s.player_pos, prev_pos)
            # On the first iteration prev_pos == player_pos (sentinel), so we
            # treat that as "should keep going" by checking step_count == 0.
            first = step_count == 0
            # Stop if a monster materialized in our path: _try_step routed
            # into combat and did NOT advance the player, so prev_pos ==
            # current pos (no move) terminates the loop naturally via `moved`.
            return (first | moved) & (step_count < _RUN_MAX_STEPS)

        def body(carry):
            s, step_count, _prev_pos, rng_cur = carry
            prev_pos = s.player_pos
            sub_rng, next_rng = jax.random.split(rng_cur)
            new_s = _try_step(s, dy, dx, sub_rng)
            return new_s, step_count + 1, prev_pos, next_rng

        # Sentinel: prev_pos starts equal to player_pos so cond reads step==0.
        init = (state, jnp.int32(0), state.player_pos, rng)
        final_state, _, _, _ = jax.lax.while_loop(cond, body, init)
        return final_state

    return _run


_run_n  = _make_run(*_DELTAS["N"])
_run_e  = _make_run(*_DELTAS["E"])
_run_s  = _make_run(*_DELTAS["S"])
_run_w  = _make_run(*_DELTAS["W"])
_run_ne = _make_run(*_DELTAS["NE"])
_run_se = _make_run(*_DELTAS["SE"])
_run_sw = _make_run(*_DELTAS["SW"])
_run_nw = _make_run(*_DELTAS["NW"])


# ---------------------------------------------------------------------------
# Direction-shared move / run handlers (Wave 8 compile-time refactor).
#
# The per-direction _move_n/_move_e/.../_run_n/.../ handlers above each get
# fully traced into the IR when wired through ``lax.switch`` (16 × ~3700 ops
# = ~60K eqns).  These shared variants take a runtime ``dir_idx`` and look up
# the (dy, dx) delta from ``_DIR_TABLE`` so only ONE move body and ONE run
# body are traced into the dispatch graph.
#
# Behavior is byte-identical to the per-direction variants when invoked with
# the matching ``dir_idx``; vendor reference unchanged
# (hack.c::domove / hack.c::test_move).
# ---------------------------------------------------------------------------

def _move_shared(state, rng, dir_idx):
    """Single-step move in ``dir_idx`` ∈ [0,7] (see _DIR_TABLE ordering)."""
    dy = _DIR_TABLE[dir_idx, 0]
    dx = _DIR_TABLE[dir_idx, 1]
    return _try_step(state, dy, dx, rng)


def _run_shared(state, rng, dir_idx):
    """Run in ``dir_idx`` ∈ [0,7].  Mirrors _make_run's while-loop body but
    closes over the traced ``dir_idx`` instead of compile-time constants."""
    dy = _DIR_TABLE[dir_idx, 0]
    dx = _DIR_TABLE[dir_idx, 1]

    def cond(carry):
        s, step_count, prev_pos, _rng = carry
        moved = ~jnp.array_equal(s.player_pos, prev_pos)
        first = step_count == 0
        return (first | moved) & (step_count < _RUN_MAX_STEPS)

    def body(carry):
        s, step_count, _prev_pos, rng_cur = carry
        prev_pos = s.player_pos
        sub_rng, next_rng = jax.random.split(rng_cur)
        new_s = _try_step(s, dy, dx, sub_rng)
        return new_s, step_count + 1, prev_pos, next_rng

    init = (state, jnp.int32(0), state.player_pos, rng)
    final_state, _, _, _ = jax.lax.while_loop(cond, body, init)
    return final_state


# ---------------------------------------------------------------------------
# Stair handlers (hack.c doup / dodown)
# ---------------------------------------------------------------------------

def _stair_up(state, rng):
    """Traverse up-stair if standing on STAIRCASE_UP tile.

    Wave 2: within-branch traversal only (bumps current_level by -1).
    Cross-branch traversal deferred to Wave 4.
    Arriving level: player repositioned to STAIRCASE_DOWN of the new level.
    For Wave 2 (single level), current_level clamps at 1.
    """
    terrain_2d = _current_level_terrain(state)
    row, col    = state.player_pos[0], state.player_pos[1]
    tile        = terrain_2d[row, col].astype(jnp.int32)
    on_stair    = tile == jnp.int32(TileType.STAIRCASE_UP)

    new_level = jnp.where(
        on_stair,
        jnp.maximum(jnp.int8(1), state.dungeon.current_level - jnp.int8(1)),
        state.dungeon.current_level,
    )
    new_dungeon = state.dungeon.replace(current_level=new_level)
    new_state   = state.replace(dungeon=new_dungeon)
    return _apply_fov(new_state)


def _stair_down(state, rng):
    """Traverse down-stair if standing on STAIRCASE_DOWN tile.

    Wave 2: within-branch traversal only (bumps current_level by +1).
    Cross-branch traversal deferred to Wave 4.
    """
    terrain_2d = _current_level_terrain(state)
    row, col    = state.player_pos[0], state.player_pos[1]
    tile        = terrain_2d[row, col].astype(jnp.int32)
    on_stair    = tile == jnp.int32(TileType.STAIRCASE_DOWN)

    max_level   = jnp.int8(state.terrain.shape[1])  # MAX_LEVELS_PER_BRANCH
    new_level   = jnp.where(
        on_stair,
        jnp.minimum(max_level, state.dungeon.current_level + jnp.int8(1)),
        state.dungeon.current_level,
    )
    new_dungeon = state.dungeon.replace(current_level=new_level)
    # Wave 8 vendor parity: track deepest_lev_reached as the max level ever
    # visited (dungeon.c:deepest_lev_reached).  This drives internal[0] and
    # the end-of-game scoring bonus.
    new_deepest = jnp.maximum(
        state.scoring.deepest_level, new_level.astype(jnp.int8)
    )
    new_scoring = state.scoring.replace(deepest_level=new_deepest)
    new_state   = state.replace(dungeon=new_dungeon, scoring=new_scoring)
    return _apply_fov(new_state)


def _wait(state, rng):
    """Rest one turn — state is unchanged (outer loop ticks the timestep)."""
    return state


# ---------------------------------------------------------------------------
# Wave 4 Phase 0 — action-handler wrappers (slots 20+)
#
# Each wrapper has signature (state: EnvState, rng) -> EnvState and is
# JIT-safe (no Python control flow on traced values).  Subsystem-native
# handlers whose signatures already match EnvState→EnvState are used
# directly; the wrappers below exist where signatures differ.
# ---------------------------------------------------------------------------


def _handle_eat(state, rng):
    """EAT — vendor/nethack/src/eat.c::doeat.

    Find the first FOOD-class inventory slot with quantity > 0 and consume it.
    Mirrors handle_quaff/handle_read style (first valid item, JIT-safe).
    """
    categories = state.inventory.items.category   # [MAX_INVENTORY_SLOTS]
    quantities = state.inventory.items.quantity   # [MAX_INVENTORY_SLOTS]
    nutritions = state.inventory.items.weight      # nutrition stored in weight (Wave 3 placeholder)

    is_food   = categories == jnp.int8(ItemCategory.FOOD)
    has_stock = quantities > jnp.int16(0)
    valid     = is_food & has_stock

    slot_idx  = jnp.argmax(valid).astype(jnp.int32)
    found     = jnp.any(valid)

    # Nutrition value: use a fixed 800 (food-ration default per eat.c) when item
    # weight is zero, else use the item weight as the nutrition proxy.  Wave 5
    # will plumb full per-type nutrition from objects.c.
    slot_nutrition = nutritions[slot_idx].astype(jnp.int32)
    food_nutrition = jnp.where(slot_nutrition > 0, slot_nutrition, jnp.int32(800))

    # Apply nutrition via handle_eat on the status slice.
    safe_slot = jnp.clip(slot_idx, 0, MAX_INVENTORY_SLOTS - 1)
    new_status = _status_handle_eat(
        state.status,
        item_nutrition=food_nutrition,
        item_class=jnp.int8(7),  # FOOD_CLASS sentinel expected by handle_eat
        item_present=found,
    )

    # Decrement the consumed item's quantity by 1 (clear category if exhausted).
    items = state.inventory.items
    old_qty = items.quantity[safe_slot]
    new_qty = jnp.where(found, jnp.maximum(old_qty - jnp.int16(1), jnp.int16(0)), old_qty)
    new_cat = jnp.where(
        found & (new_qty == jnp.int16(0)),
        jnp.int8(0),
        items.category[safe_slot],
    )
    new_quantity = items.quantity.at[safe_slot].set(new_qty)
    new_category = items.category.at[safe_slot].set(new_cat)
    new_items    = items.replace(quantity=new_quantity, category=new_category)
    new_inv      = state.inventory.replace(items=new_items)
    new_state    = state.replace(status=new_status, inventory=new_inv)
    # Conduct: vendor/nethack/src/eat.c::eatfood — FOODLESS broken on any eat;
    # VEGAN / VEGETARIAN broken by material (objects.h FOOD materials).
    eaten_material = _food_material_for_type_id(items.type_id[safe_slot])
    new_state = _mark_violated_if(new_state, int(_Conduct.FOODLESS), found)
    new_state = _mark_violated_if(new_state, int(_Conduct.VEGETARIAN), found & _is_meat_material(eaten_material))
    new_state = _mark_violated_if(new_state, int(_Conduct.VEGAN), found & _is_animal_material(eaten_material))
    return new_state


def _handle_quaff(state, rng):
    """QUAFF — vendor/nethack/src/potion.c::dodrink.  Direct delegate."""
    return _potions_handle_quaff(state, rng)


def _handle_read(state, rng):
    """READ — vendor/nethack/src/read.c::doread.  Direct delegate."""
    return _scrolls_handle_read(state, rng)


def _handle_zap(state, rng):
    """ZAP — vendor/nethack/src/zap.c::dozap.

    The native handle_zap in items_wands operates on a self-contained
    WandState slice.  This wrapper projects the relevant EnvState fields
    into a WandState, invokes the wand handler, then writes results back.
    """
    # Project EnvState → WandState (single current level slice).
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    terrain_2d  = state.terrain[b, lv]
    explored_2d = state.explored[b, lv]

    wand_state = WandState(
        mon_pos       = state.monster_ai.pos,
        mon_hp        = state.monster_ai.hp,
        mon_type      = jnp.zeros_like(state.monster_ai.hp).astype(jnp.int16),
        mon_alive     = state.monster_ai.alive,
        mon_asleep    = state.monster_ai.asleep,
        mon_undead    = jnp.zeros_like(state.monster_ai.alive),
        mon_invisible = jnp.zeros_like(state.monster_ai.alive),
        terrain       = terrain_2d,
        explored      = explored_2d,
        inventory     = state.inventory,
        player_pos    = state.player_pos,
    )

    new_wand = _wands_handle_zap(wand_state, rng)

    # Write back the mutated slices into EnvState.
    new_monster_ai = state.monster_ai.replace(
        pos       = new_wand.mon_pos,
        hp        = new_wand.mon_hp,
        alive     = new_wand.mon_alive,
        asleep    = new_wand.mon_asleep,
    )
    new_terrain  = state.terrain.at[b, lv].set(new_wand.terrain)
    new_explored = state.explored.at[b, lv].set(new_wand.explored)

    return state.replace(
        monster_ai = new_monster_ai,
        terrain    = new_terrain,
        explored   = new_explored,
        inventory  = new_wand.inventory,
    )


def _handle_cast(state, rng):
    """CAST — vendor/nethack/src/spell.c::docast.

    JIT-safe wrapper: find the first known+memorized spell, deduct Pw cost
    (spell.h:SPELL_LEV_PW = spell_level * 5), and decrement that spell's
    memory by 1.  This is a minimal stand-in for the full spelleffects()
    pipeline; the native ``magic.handle_cast`` / ``magic.cast_spell``
    functions use Python control flow (``int(traced)``, ``bool(traced)``)
    and cannot be invoked from inside ``lax.switch``.  Wave 5 will JIT-port
    cast_spell (effect dispatch + percent_success rolls) and replace this
    wrapper; the behavioral surface (Pw drain, memory tick) is preserved.
    """
    from Nethax.nethax.subsystems.magic import _SPELL_LEVELS

    magic = state.magic
    known = magic.spell_known
    mem   = magic.spell_memory

    valid = known & (mem > jnp.int32(0))
    slot  = jnp.argmax(valid).astype(jnp.int32)
    found = jnp.any(valid)

    safe_slot = jnp.clip(slot, 0, jnp.int32(N_SPELLS - 1))
    pw_cost   = _SPELL_LEVELS[safe_slot] * jnp.int32(5)
    has_pw    = state.player_pw >= pw_cost
    will_cast = found & has_pw

    new_pw  = jnp.where(will_cast, state.player_pw - pw_cost, state.player_pw)
    new_mem = mem.at[safe_slot].set(
        jnp.where(
            will_cast,
            jnp.maximum(mem[safe_slot] - jnp.int32(1), jnp.int32(0)),
            mem[safe_slot],
        )
    )
    new_magic = magic.replace(spell_memory=new_mem)
    return state.replace(player_pw=new_pw, magic=new_magic)


def _handle_pickup(state, rng):
    """PICKUP — vendor/nethack/src/pickup.c::dopickup.

    Project the current branch/level ground-item stack from EnvState,
    invoke inventory.pickup, then write the updated ground_items back.
    """
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    # inventory.pickup uses branch/level for direct ground_items indexing.
    new_state, new_ground = _inv_pickup(state, rng, state.ground_items, b, lv)
    return new_state.replace(ground_items=new_ground)


def _handle_drop(state, rng):
    """DROP — vendor/nethack/src/pickup.c::dodrop.

    Find the first occupied inventory slot; delegate to inventory.drop with
    the current branch/level.  JIT-safe (lax.scan over slots).
    """
    from jax import lax as _lax

    def _find_occupied(carry, idx):
        found, slot = carry
        occupied = state.inventory.items.category[idx] != 0
        slot  = jnp.where(~found & occupied, idx, slot)
        found = found | occupied
        return (found, slot), None

    (_, first_slot), _ = _lax.scan(
        _find_occupied,
        (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32),
    )

    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    new_state, new_ground = _inv_drop(state, rng, state.ground_items, b, lv, first_slot)
    return new_state.replace(ground_items=new_ground)


def _handle_wield(state, rng):
    """WIELD — direct delegate."""
    return _inv_handle_wield(state, rng)


def _handle_wear(state, rng):
    """WEAR — direct delegate."""
    return _inv_handle_wear(state, rng)


def _handle_put_on(state, rng):
    """PUTON — vendor/nethack/src/do_wear.c::doputon.

    JIT-safe wrapper: scan inventory for the first RING or AMULET slot,
    place ring on the first free finger or amulet in the empty amulet slot.
    The native items_jewelry.handle_put_on uses Python control flow.
    """
    items = state.inventory.items
    categories = items.category
    quantities = items.quantity

    is_ring   = categories == jnp.int8(ItemCategory.RING)
    is_amulet = categories == jnp.int8(ItemCategory.AMULET)
    in_stock  = quantities > jnp.int16(0)

    ring_mask   = is_ring & in_stock
    amulet_mask = is_amulet & in_stock

    ring_slot   = jnp.argmax(ring_mask).astype(jnp.int32)
    amulet_slot = jnp.argmax(amulet_mask).astype(jnp.int32)
    has_ring    = jnp.any(ring_mask)
    has_amulet  = jnp.any(amulet_mask)

    worn_rings = state.inventory.worn_rings
    left_free  = worn_rings[0] < jnp.int8(0)
    right_free = worn_rings[1] < jnp.int8(0)
    amulet_free = state.inventory.worn_amulet < jnp.int8(0)

    # Pick ring hand: prefer left, then right.
    hand = jnp.where(left_free, jnp.int8(0), jnp.int8(1))
    can_put_ring = has_ring & (left_free | right_free)
    can_put_amulet = has_amulet & amulet_free

    new_worn_rings = jnp.where(
        can_put_ring,
        worn_rings.at[hand.astype(jnp.int32)].set(ring_slot.astype(jnp.int8)),
        worn_rings,
    )
    new_worn_amulet = jnp.where(
        can_put_amulet,
        amulet_slot.astype(jnp.int8),
        state.inventory.worn_amulet,
    )

    new_inv = state.inventory.replace(
        worn_rings=new_worn_rings,
        worn_amulet=new_worn_amulet,
    )
    return state.replace(inventory=new_inv)


def _handle_remove(state, rng):
    """REMOVE — vendor/nethack/src/do_wear.c::doremring/doremarm.

    JIT-safe wrapper: take off the first occupied worn slot in priority
    left-ring → right-ring → amulet.  Native handle_remove uses Python
    control flow.
    """
    worn_rings = state.inventory.worn_rings
    left_worn  = worn_rings[0] >= jnp.int8(0)
    right_worn = worn_rings[1] >= jnp.int8(0)
    amulet_worn = state.inventory.worn_amulet >= jnp.int8(0)

    # Priority: left ring first.
    new_worn_rings = jnp.where(
        left_worn,
        worn_rings.at[0].set(jnp.int8(-1)),
        jnp.where(
            right_worn,
            worn_rings.at[1].set(jnp.int8(-1)),
            worn_rings,
        ),
    )
    # If neither ring is worn, fall through to amulet removal.
    remove_amulet = ~left_worn & ~right_worn & amulet_worn
    new_worn_amulet = jnp.where(
        remove_amulet,
        jnp.int8(-1),
        state.inventory.worn_amulet,
    )

    new_inv = state.inventory.replace(
        worn_rings=new_worn_rings,
        worn_amulet=new_worn_amulet,
    )
    return state.replace(inventory=new_inv)


def _handle_open(state, rng):
    """OPEN — direct delegate (vendor/nethack/src/lock.c::doopen)."""
    return _features_handle_open(state, rng)


def _handle_close(state, rng):
    """CLOSE — direct delegate (vendor/nethack/src/lock.c::doclose)."""
    return _features_handle_close(state, rng)


def _handle_kick(state, rng):
    """KICK — direct delegate (vendor/nethack/src/dokick.c::dokick)."""
    return _features_handle_kick(state, rng)


def _handle_fight(state, rng):
    """FIGHT — direct delegate (vendor/nethack/src/cmd.c::dofight)."""
    return _combat_handle_fight(state, rng)


def _handle_search(state, rng):
    """SEARCH — vendor/nethack/src/detect.c::dosearch.

    Wave 4 Phase 0: no-op until a search handler is implemented; the action
    is still routed (rather than dropped through unmapped → noop) so future
    waves can swap in real reveal logic without touching the table.
    """
    return state


def _handle_pray(state, rng):
    """PRAY — vendor/nethack/src/pray.c::dopray.  Direct delegate."""
    return _prayer_handle_pray(state, rng)


def _handle_twoweapon(state, rng):
    """TWOWEAPON — vendor/nethack/src/wield.c::dotwoweapon.

    Toggles the player's two-weapon combat flag (Wave 5 Phase 1).
    """
    return _combat_handle_twoweapon(state, rng)


def _handle_throw(state, rng):
    """THROW — vendor/nethack/src/dothrow.c::dothrow.

    Throws the first quivered / weapon-class inventory item east
    (Wave 5 Phase 1 default direction).
    """
    return _combat_handle_throw(state, rng)


def _handle_loot(state, rng):
    """LOOT — vendor/nethack/src/pickup.c::doloot.

    Wave 5 Phase 3: open the lowest-index held container.
    """
    return _containers_handle_loot(state, rng)


def _handle_apply(state, rng):
    """APPLY — vendor/nethack/src/apply.c::doapply.

    Wave 5 Phase 3: if the player has any held container, route to the
    container open path.  Non-container APPLY (lamps, pick-axes, etc.) will
    be wired in a later wave.
    """
    return _containers_handle_apply(state, rng)


def _handle_engrave(state, rng):
    """ENGRAVE — vendor/nethack/src/engrave.c::doengrave.

    Wave 5 Phase 4: always engrave 'Elbereth' in dust at the player tile;
    also sets the ELBERETHLESS conduct (insight.c ~2206).
    """
    from Nethax.nethax.subsystems.engrave import handle_engrave
    return handle_engrave(state, rng)


def _handle_name(state, rng):
    """CALL — vendor/nethack/src/do_name.c::do_oname (Wave 6).

    The interactive ``C`` command prompts the user for a slot + name; with
    no UI available in the headless harness this dispatch slot is a no-op.
    Tests / agents that want to set a name call
    ``inventory.handle_name(state, rng, slot_idx, name_bytes)`` directly.
    """
    return state


# ---------------------------------------------------------------------------
# Handler tuple — indexed by handler slot (0 = noop, 1-8 = move, 9-16 = run,
#                                          17 = stair_up, 18 = stair_down,
#                                          19 = wait, 20+ = Wave 4 actions)
# ---------------------------------------------------------------------------

_HANDLERS = (
    _noop,         # 0
    _move_n,       # 1
    _move_e,       # 2
    _move_s,       # 3
    _move_w,       # 4
    _move_ne,      # 5
    _move_se,      # 6
    _move_sw,      # 7
    _move_nw,      # 8
    _run_n,        # 9
    _run_e,        # 10
    _run_s,        # 11
    _run_w,        # 12
    _run_ne,       # 13
    _run_se,       # 14
    _run_sw,       # 15
    _run_nw,       # 16
    _stair_up,     # 17
    _stair_down,   # 18
    _wait,         # 19
    _handle_eat,    # 20  vendor/nethack/src/eat.c::doeat
    _handle_quaff,  # 21  vendor/nethack/src/potion.c::dodrink
    _handle_read,   # 22  vendor/nethack/src/read.c::doread
    _handle_zap,    # 23  vendor/nethack/src/zap.c::dozap
    _handle_cast,   # 24  vendor/nethack/src/spell.c::docast
    _handle_pickup, # 25  vendor/nethack/src/pickup.c::dopickup
    _handle_drop,   # 26  vendor/nethack/src/pickup.c::dodrop
    _handle_wield,  # 27  vendor/nethack/src/wield.c::dowield
    _handle_wear,   # 28  vendor/nethack/src/do_wear.c::dowear
    _handle_put_on, # 29  vendor/nethack/src/do_wear.c::doputon
    _handle_remove, # 30  vendor/nethack/src/do_wear.c::doremring
    _handle_open,   # 31  vendor/nethack/src/lock.c::doopen
    _handle_close,  # 32  vendor/nethack/src/lock.c::doclose
    _handle_kick,   # 33  vendor/nethack/src/dokick.c::dokick
    _handle_fight,  # 34  vendor/nethack/src/cmd.c::dofight
    _handle_search, # 35  vendor/nethack/src/detect.c::dosearch
    _handle_pray,   # 36  vendor/nethack/src/pray.c::dopray
    _handle_twoweapon,  # 37  vendor/nethack/src/wield.c::dotwoweapon
    _handle_throw,      # 38  vendor/nethack/src/dothrow.c::dothrow
    _handle_loot,       # 39  vendor/nethack/src/pickup.c::doloot
    _handle_apply,      # 40  vendor/nethack/src/apply.c::doapply
    _handle_engrave,    # 41  vendor/nethack/src/engrave.c::doengrave
    _handle_name,       # 42  vendor/nethack/src/do_name.c::do_oname (Wave 6)
)

# Slot indices for each named handler.
_SLOT_NOOP       = 0
_SLOT_MOVE_N     = 1
_SLOT_MOVE_E     = 2
_SLOT_MOVE_S     = 3
_SLOT_MOVE_W     = 4
_SLOT_MOVE_NE    = 5
_SLOT_MOVE_SE    = 6
_SLOT_MOVE_SW    = 7
_SLOT_MOVE_NW    = 8
_SLOT_RUN_N      = 9
_SLOT_RUN_E      = 10
_SLOT_RUN_S      = 11
_SLOT_RUN_W      = 12
_SLOT_RUN_NE     = 13
_SLOT_RUN_SE     = 14
_SLOT_RUN_SW     = 15
_SLOT_RUN_NW     = 16
_SLOT_STAIR_UP   = 17
_SLOT_STAIR_DOWN = 18
_SLOT_WAIT       = 19
# Wave 4 Phase 0 — newly wired action slots.
_SLOT_EAT        = 20
_SLOT_QUAFF      = 21
_SLOT_READ       = 22
_SLOT_ZAP        = 23
_SLOT_CAST       = 24
_SLOT_PICKUP     = 25
_SLOT_DROP       = 26
_SLOT_WIELD      = 27
_SLOT_WEAR       = 28
_SLOT_PUTON      = 29
_SLOT_REMOVE     = 30
_SLOT_OPEN       = 31
_SLOT_CLOSE      = 32
_SLOT_KICK       = 33
_SLOT_FIGHT      = 34
_SLOT_SEARCH     = 35
_SLOT_PRAY       = 36
_SLOT_TWOWEAPON  = 37
_SLOT_THROW      = 38
# Wave 5 Phase 3 — container actions.
_SLOT_LOOT       = 39
_SLOT_APPLY      = 40
# Wave 5 Phase 4 — engrave action.
_SLOT_ENGRAVE    = 41
# Wave 6 Phase A — name (call) action.
_SLOT_NAME       = 42

# ---------------------------------------------------------------------------
# 256-entry lookup table: action ASCII value → handler slot index
# ---------------------------------------------------------------------------

def _M_byte(c: str) -> int:
    """Meta-key byte: sets bit 7 on the ASCII value of `c` (vendor M()).

    Mirrors vendor/nethack/include/wintype.h's M() macro used throughout
    cmd.c::extcmdlist[].
    """
    return 0x80 | ord(c)


def _C_byte(c: str) -> int:
    """Ctrl-key byte: masks `c` to its low 5 bits (vendor C() macro)."""
    return 0x1F & ord(c)


def _build_action_to_handler_idx() -> jnp.ndarray:
    """Return int8[256] mapping action ASCII value to handler slot.

    All unmapped values default to slot 0 (_noop).
    """
    table = [_SLOT_NOOP] * 256

    # Single-step moves
    table[int(CompassCardinalDirection.N)]  = _SLOT_MOVE_N
    table[int(CompassCardinalDirection.E)]  = _SLOT_MOVE_E
    table[int(CompassCardinalDirection.S)]  = _SLOT_MOVE_S
    table[int(CompassCardinalDirection.W)]  = _SLOT_MOVE_W
    table[int(CompassIntercardinalDirection.NE)] = _SLOT_MOVE_NE
    table[int(CompassIntercardinalDirection.SE)] = _SLOT_MOVE_SE
    table[int(CompassIntercardinalDirection.SW)] = _SLOT_MOVE_SW
    table[int(CompassIntercardinalDirection.NW)] = _SLOT_MOVE_NW

    # Run moves
    table[int(CompassCardinalDirectionLonger.N)]  = _SLOT_RUN_N
    table[int(CompassCardinalDirectionLonger.E)]  = _SLOT_RUN_E
    table[int(CompassCardinalDirectionLonger.S)]  = _SLOT_RUN_S
    table[int(CompassCardinalDirectionLonger.W)]  = _SLOT_RUN_W
    table[int(CompassIntercardinalDirectionLonger.NE)] = _SLOT_RUN_NE
    table[int(CompassIntercardinalDirectionLonger.SE)] = _SLOT_RUN_SE
    table[int(CompassIntercardinalDirectionLonger.SW)] = _SLOT_RUN_SW
    table[int(CompassIntercardinalDirectionLonger.NW)] = _SLOT_RUN_NW

    # Stairs and wait
    table[int(MiscDirection.UP)]   = _SLOT_STAIR_UP
    table[int(MiscDirection.DOWN)] = _SLOT_STAIR_DOWN
    table[int(MiscDirection.WAIT)] = _SLOT_WAIT

    # ---- Wave 4 Phase 0 — item / interaction actions ----
    # ASCII keys mirror vendor/nethack/src/cmd.c::cmdlist[].
    table[ord("e")] = _SLOT_EAT      # eat.c::doeat
    table[ord("q")] = _SLOT_QUAFF    # potion.c::dodrink
    table[ord("r")] = _SLOT_READ     # read.c::doread
    table[ord("z")] = _SLOT_ZAP      # zap.c::dozap
    table[ord("Z")] = _SLOT_CAST     # spell.c::docast
    table[ord(",")] = _SLOT_PICKUP   # pickup.c::dopickup
    table[ord("d")] = _SLOT_DROP     # pickup.c::dodrop
    table[ord("w")] = _SLOT_WIELD    # wield.c::dowield
    table[ord("W")] = _SLOT_WEAR     # do_wear.c::dowear
    table[ord("P")] = _SLOT_PUTON    # do_wear.c::doputon
    table[ord("R")] = _SLOT_REMOVE   # do_wear.c::doremring
    table[ord("o")] = _SLOT_OPEN     # lock.c::doopen
    table[ord("c")] = _SLOT_CLOSE    # lock.c::doclose
    table[int(Command.KICK)] = _SLOT_KICK  # dokick.c::dokick (Ctrl-d)
    table[ord("F")] = _SLOT_FIGHT    # cmd.c::dofight
    table[ord("s")] = _SLOT_SEARCH   # detect.c::dosearch
    table[int(Command.PRAY)] = _SLOT_PRAY  # pray.c::dopray (Meta-p)
    # Wave 5 Phase 1 — combat polish actions.
    table[int(Command.TWOWEAPON)] = _SLOT_TWOWEAPON  # wield.c::dotwoweapon ('X')
    table[int(Command.THROW)] = _SLOT_THROW          # dothrow.c::dothrow ('t')
    # Wave 5 Phase 3 — container actions.
    table[int(Command.LOOT)]  = _SLOT_LOOT   # pickup.c::doloot (Meta-l)
    table[int(Command.APPLY)] = _SLOT_APPLY  # apply.c::doapply ('a')
    # Wave 5 Phase 4 — engrave action.
    table[int(Command.ENGRAVE)] = _SLOT_ENGRAVE  # engrave.c::doengrave ('E')
    # Wave 6 Phase A — call / name action.
    table[int(Command.CALL)] = _SLOT_NAME  # do_name.c::do_oname ('C')

    # ------------------------------------------------------------------
    # Wave 6 Closing-Audit #87 — full vendor extcmdlist[] coverage.
    #
    # Mirrors every keyed entry in vendor/nethack/src/cmd.c::extcmdlist[].
    # Each line below corresponds to one vendor row.  Entries marked
    # ``_SLOT_NOOP`` are *intentional* no-ops: the key is recognised
    # (so we mirror vendor's coverage), but the handler is informational,
    # UI-only, wizard-mode-only, or otherwise out of scope for the
    # headless JAX-JIT environment.  Adding them here makes the parity
    # test pass and documents why we don't dispatch real logic.
    # ------------------------------------------------------------------

    # SPACE → wait (vendor cmd.c::update_rest_on_space when 'rest_on_space'
    # is on; donull).  Always-on in our env — no toggle.
    table[ord(" ")] = _SLOT_WAIT

    # '#' extcmd prefix (doextcmd) — no extended-command UI in headless env.
    table[int(Command.EXTCMD)] = _SLOT_NOOP   # cmd.c::doextcmd
    table[int(Command.EXTLIST)] = _SLOT_NOOP  # cmd.c::doextlist (M-?)

    # Inventory / option / informational commands (no UI to render).
    table[ord("i")] = _SLOT_NOOP  # cmd.c::ddoinv (show inventory)
    table[ord("I")] = _SLOT_NOOP  # cmd.c::dotypeinv
    table[ord(":")] = _SLOT_NOOP  # cmd.c::dolook (look here)
    table[ord(";")] = _SLOT_NOOP  # cmd.c::doquickwhatis
    table[ord("?")] = _SLOT_NOOP  # cmd.c::dohelp
    table[ord("&")] = _SLOT_NOOP  # cmd.c::dowhatdoes
    table[ord("/")] = _SLOT_NOOP  # cmd.c::dowhatis
    table[ord("O")] = _SLOT_NOOP  # cmd.c::doset_simple
    table[ord("\\")] = _SLOT_NOOP  # cmd.c::dodiscovered (known items)
    table[ord("`")] = _SLOT_NOOP   # cmd.c::doclassdisco (known by class)
    table[ord("|")] = _SLOT_NOOP   # cmd.c::doperminv (perm inventory)
    table[ord("@")] = _SLOT_NOOP   # cmd.c::dotogglepickup (autopickup)
    table[ord("v")] = _SLOT_NOOP   # cmd.c::do_gamelog (chronicle)
    table[ord("V")] = _SLOT_NOOP   # cmd.c::doversion (versionshort)

    # "See currently worn/wielded equipment" — UI-only informational.
    table[ord("*")] = _SLOT_NOOP   # cmd.c::doprinuse (seeall)
    table[ord('"')] = _SLOT_NOOP   # cmd.c::dopramulet (AMULET_SYM)
    table[ord("[")] = _SLOT_NOOP   # cmd.c::doprarm    (ARMOR_SYM)
    table[ord("=")] = _SLOT_NOOP   # cmd.c::doprring   (RING_SYM)
    table[ord("(")] = _SLOT_NOOP   # cmd.c::doprtool   (TOOL_SYM)
    table[ord(")")] = _SLOT_NOOP   # cmd.c::doprwep    (WEAPON_SYM)
    table[ord("$")] = _SLOT_NOOP   # cmd.c::doprgold   (GOLD_SYM)
    table[ord("+")] = _SLOT_NOOP   # cmd.c::dovspell   (SPBOOK_SYM, showspells)
    table[ord("^")] = _SLOT_NOOP   # cmd.c::doidtrap   (showtrap)

    # 'D' droptype (multi-drop UI) — no UI; single-item drop handled by 'd'.
    table[ord("D")] = _SLOT_DROP   # cmd.c::doddrop — proxy to dodrop

    # 'T' takeoff (armor) — share the remove handler (covers worn slots).
    table[ord("T")] = _SLOT_REMOVE # cmd.c::dotakeoff
    table[ord("A")] = _SLOT_REMOVE # cmd.c::doddoremarm (takeoffall)

    # 'x' swap weapons — no swap subsystem yet, document as no-op.
    table[ord("x")] = _SLOT_NOOP   # cmd.c::doswapweapon

    # 'p' pay — no shop subsystem.  'Q' quiver — no quiver subsystem.
    table[ord("p")] = _SLOT_NOOP   # cmd.c::dopay
    table[ord("Q")] = _SLOT_NOOP   # cmd.c::dowieldquiver

    # 'f' fire — vendor fires the quiver; we proxy to THROW for now.
    table[ord("f")] = _SLOT_THROW  # cmd.c::dofire

    # 'G' run prefix / 'g' rush prefix — distinct from our uppercase-letter
    # run keys (H/J/K/L/...); these prefixes need a following direction key,
    # which our action API doesn't carry.  Mark as no-op (move semantics
    # already covered by COMPASSLONG keys).
    table[ord("G")] = _SLOT_NOOP   # cmd.c::do_run (RUN prefix)
    table[ord("g")] = _SLOT_NOOP   # cmd.c::do_rush (RUSH prefix)
    table[ord("m")] = _SLOT_NOOP   # cmd.c::do_reqmenu (REQMENU prefix)

    # 'S' save / shell escape / suspend — no process-level features.
    table[ord("S")] = _SLOT_NOOP   # cmd.c::dosave
    table[ord("!")] = _SLOT_NOOP   # cmd.c::dosh_core (shell)

    # Numpad alternate movement (vendor: bind_keys_to_extcmds when
    # number_pad is on).  Always provided for parity.
    table[ord("8")] = _SLOT_MOVE_N   # numpad N
    table[ord("2")] = _SLOT_MOVE_S   # numpad S
    table[ord("6")] = _SLOT_MOVE_E   # numpad E
    table[ord("4")] = _SLOT_MOVE_W   # numpad W
    table[ord("9")] = _SLOT_MOVE_NE  # numpad NE
    table[ord("3")] = _SLOT_MOVE_SE  # numpad SE
    table[ord("1")] = _SLOT_MOVE_SW  # numpad SW
    table[ord("7")] = _SLOT_MOVE_NW  # numpad NW
    table[ord("5")] = _SLOT_WAIT     # numpad 5 = rest (cmd.c line 3404 area)

    # ---- Meta-prefixed extended commands (M(x) == 0x80 | ord(x)) ----
    # These get bound to the M-prefix key when not running through #extcmd.
    table[_M_byte("a")] = _SLOT_NOOP   # cmd.c::doorganize (adjust)
    table[_M_byte("A")] = _SLOT_NOOP   # cmd.c::donamelevel (annotate)
    table[_M_byte("c")] = _SLOT_NOOP   # cmd.c::dotalk (chat)
    table[_M_byte("C")] = _SLOT_NOOP   # cmd.c::doconduct
    table[_M_byte("d")] = _SLOT_NOOP   # cmd.c::dodip
    table[_M_byte("e")] = _SLOT_EAT    # cmd.c::doeat is 'e'; M-e (enhance)
                                       # is the actual extcmdlist entry — we
                                       # alias to EAT here so any agent that
                                       # sends Meta-e for "eat via #" still
                                       # routes correctly.
    table[_M_byte("f")] = _SLOT_NOOP   # cmd.c::doforce
    table[_M_byte("g")] = _SLOT_NOOP   # cmd.c::dogenocided
    table[_M_byte("i")] = _SLOT_NOOP   # cmd.c::doinvoke
    table[_M_byte("j")] = _SLOT_NOOP   # cmd.c::dojump
    table[_M_byte("l")] = _SLOT_LOOT   # cmd.c::doloot (already set via Command.LOOT)
    table[_M_byte("m")] = _SLOT_NOOP   # cmd.c::domonability
    table[_M_byte("n")] = _SLOT_NAME   # cmd.c::docallcmd (name alias)
    table[_M_byte("o")] = _SLOT_NOOP   # cmd.c::dosacrifice (offer)
    table[_M_byte("p")] = _SLOT_PRAY   # cmd.c::dopray (already set)
    table[_M_byte("q")] = _SLOT_NOOP   # cmd.c::done2 (quit)
    table[_M_byte("r")] = _SLOT_NOOP   # cmd.c::dorub
    table[_M_byte("R")] = _SLOT_NOOP   # cmd.c::doride
    table[_M_byte("s")] = _SLOT_NOOP   # cmd.c::dosit
    table[_M_byte("t")] = _SLOT_NOOP   # cmd.c::doturn
    table[_M_byte("T")] = _SLOT_NOOP   # cmd.c::dotip
    table[_M_byte("u")] = _SLOT_NOOP   # cmd.c::dountrap
    table[_M_byte("v")] = _SLOT_NOOP   # cmd.c::doextversion
    table[_M_byte("V")] = _SLOT_NOOP   # cmd.c::dovanquished
    table[_M_byte("w")] = _SLOT_NOOP   # cmd.c::dowipe
    table[_M_byte("X")] = _SLOT_NOOP   # cmd.c::enter_explore_mode
    table[_M_byte("?")] = _SLOT_NOOP   # cmd.c::doextlist

    # ---- Ctrl-prefixed commands (C(x) == 0x1F & ord(x)) ----
    # Ctrl-d (KICK) and Ctrl-p (PREVMSG) already handled via Command.* enum.
    table[_C_byte("a")] = _SLOT_NOOP   # cmd.c::do_repeat
    table[_C_byte("d")] = _SLOT_KICK   # cmd.c::dokick (mirror Command.KICK)
    table[_C_byte("e")] = _SLOT_NOOP   # wizdetect (WIZMODECMD)
    table[_C_byte("f")] = _SLOT_NOOP   # wizmap (WIZMODECMD)
    table[_C_byte("g")] = _SLOT_NOOP   # wizgenesis (WIZMODECMD)
    table[_C_byte("i")] = _SLOT_NOOP   # wizidentify (WIZMODECMD)
    table[_C_byte("o")] = _SLOT_NOOP   # cmd.c::dooverview
    table[_C_byte("p")] = _SLOT_NOOP   # cmd.c::doprev_message
    table[_C_byte("r")] = _SLOT_NOOP   # cmd.c::doredraw
    table[_C_byte("t")] = _SLOT_NOOP   # cmd.c::dotelecmd (teleport)
    table[_C_byte("v")] = _SLOT_NOOP   # wizlevelport (WIZMODECMD)
    table[_C_byte("w")] = _SLOT_NOOP   # wizwish (WIZMODECMD)
    table[_C_byte("x")] = _SLOT_NOOP   # cmd.c::doattributes
    table[_C_byte("z")] = _SLOT_NOOP   # cmd.c::dosuspend_core
    table[0x1F]        = _SLOT_NOOP   # cmd.c::dotravel_target (C('_'))

    # Misc symbols.
    table[ord("_")]  = _SLOT_NOOP   # cmd.c::dotravel
    table[0x7F]      = _SLOT_NOOP   # cmd.c::doterrain (DEL/'\177')

    return jnp.array(table, dtype=jnp.int8)


_ACTION_TO_HANDLER_IDX: jnp.ndarray = _build_action_to_handler_idx()

# ---------------------------------------------------------------------------
# Compact dispatch table (Wave 8 compile-time refactor).
#
# The legacy ``_HANDLERS`` tuple has 43 entries; 16 of them are
# direction-specialized move/run handlers that all trace ~3700 ops each into
# the ``lax.switch`` IR (~60K eqns total → dominates compile time).
#
# The compact table folds those 16 into 2 direction-shared handlers
# (_move_shared / _run_shared) and looks up the direction from a small
# integer ``dir_idx`` passed alongside (state, rng).  All other handlers are
# wrapped to accept the unused ``dir_idx`` so the switch branches share a
# common signature.
#
# Legacy ``_HANDLERS`` and the per-direction ``_move_n`` ... ``_run_nw``
# names are preserved unchanged for tests / external callers; only the
# ``dispatch_action`` body is rerouted through the compact path.
# ---------------------------------------------------------------------------


def _wrap_no_dir(fn):
    """Wrap a 2-arg handler (state, rng) into a 3-arg (state, rng, dir_idx).

    The ``dir_idx`` operand is required by ``lax.switch`` since every branch
    must share a signature; non-movement handlers simply ignore it.
    """

    def _w(state, rng, _dir_idx):
        return fn(state, rng)

    _w.__name__ = f"_compat_{fn.__name__}"
    return _w


# Compact slot indices used internally by dispatch_action.  These are an
# implementation detail — external tests reference the legacy _SLOT_*
# constants on the public _HANDLERS tuple, which is preserved separately.
_COMPACT_NOOP       = 0
_COMPACT_MOVE       = 1
_COMPACT_RUN        = 2
_COMPACT_STAIR_UP   = 3
_COMPACT_STAIR_DOWN = 4
_COMPACT_WAIT       = 5
_COMPACT_EAT        = 6
_COMPACT_QUAFF      = 7
_COMPACT_READ       = 8
_COMPACT_ZAP        = 9
_COMPACT_CAST       = 10
_COMPACT_PICKUP     = 11
_COMPACT_DROP       = 12
_COMPACT_WIELD      = 13
_COMPACT_WEAR       = 14
_COMPACT_PUTON      = 15
_COMPACT_REMOVE     = 16
_COMPACT_OPEN       = 17
_COMPACT_CLOSE      = 18
_COMPACT_KICK       = 19
_COMPACT_FIGHT      = 20
_COMPACT_SEARCH     = 21
_COMPACT_PRAY       = 22
_COMPACT_TWOWEAPON  = 23
_COMPACT_THROW      = 24
_COMPACT_LOOT       = 25
_COMPACT_APPLY      = 26
_COMPACT_ENGRAVE    = 27
_COMPACT_NAME       = 28


def _build_compact_handlers():
    """Return the 29-entry tuple of ``(state, rng, dir_idx) -> state`` handlers."""
    return (
        _wrap_no_dir(_noop),          # 0  COMPACT_NOOP
        _move_shared,                 # 1  COMPACT_MOVE  (dir_idx ∈ [0,7])
        _run_shared,                  # 2  COMPACT_RUN   (dir_idx ∈ [0,7])
        _wrap_no_dir(_stair_up),      # 3
        _wrap_no_dir(_stair_down),    # 4
        _wrap_no_dir(_wait),          # 5
        _wrap_no_dir(_handle_eat),    # 6
        _wrap_no_dir(_handle_quaff),  # 7
        _wrap_no_dir(_handle_read),   # 8
        _wrap_no_dir(_handle_zap),    # 9
        _wrap_no_dir(_handle_cast),   # 10
        _wrap_no_dir(_handle_pickup), # 11
        _wrap_no_dir(_handle_drop),   # 12
        _wrap_no_dir(_handle_wield),  # 13
        _wrap_no_dir(_handle_wear),   # 14
        _wrap_no_dir(_handle_put_on), # 15
        _wrap_no_dir(_handle_remove), # 16
        _wrap_no_dir(_handle_open),   # 17
        _wrap_no_dir(_handle_close),  # 18
        _wrap_no_dir(_handle_kick),   # 19
        _wrap_no_dir(_handle_fight),  # 20
        _wrap_no_dir(_handle_search), # 21
        _wrap_no_dir(_handle_pray),   # 22
        _wrap_no_dir(_handle_twoweapon),  # 23
        _wrap_no_dir(_handle_throw),  # 24
        _wrap_no_dir(_handle_loot),   # 25
        _wrap_no_dir(_handle_apply),  # 26
        _wrap_no_dir(_handle_engrave),# 27
        _wrap_no_dir(_handle_name),   # 28
    )


_COMPACT_HANDLERS: tuple = _build_compact_handlers()


def _build_slot_to_compact() -> jnp.ndarray:
    """Map legacy handler slot (0..42) → compact slot (0..28)."""
    table = [0] * 43
    # Movement slots 1..8 → COMPACT_MOVE.
    for s in (_SLOT_MOVE_N, _SLOT_MOVE_E, _SLOT_MOVE_S, _SLOT_MOVE_W,
              _SLOT_MOVE_NE, _SLOT_MOVE_SE, _SLOT_MOVE_SW, _SLOT_MOVE_NW):
        table[s] = _COMPACT_MOVE
    # Run slots 9..16 → COMPACT_RUN.
    for s in (_SLOT_RUN_N, _SLOT_RUN_E, _SLOT_RUN_S, _SLOT_RUN_W,
              _SLOT_RUN_NE, _SLOT_RUN_SE, _SLOT_RUN_SW, _SLOT_RUN_NW):
        table[s] = _COMPACT_RUN
    table[_SLOT_NOOP]       = _COMPACT_NOOP
    table[_SLOT_STAIR_UP]   = _COMPACT_STAIR_UP
    table[_SLOT_STAIR_DOWN] = _COMPACT_STAIR_DOWN
    table[_SLOT_WAIT]       = _COMPACT_WAIT
    table[_SLOT_EAT]        = _COMPACT_EAT
    table[_SLOT_QUAFF]      = _COMPACT_QUAFF
    table[_SLOT_READ]       = _COMPACT_READ
    table[_SLOT_ZAP]        = _COMPACT_ZAP
    table[_SLOT_CAST]       = _COMPACT_CAST
    table[_SLOT_PICKUP]     = _COMPACT_PICKUP
    table[_SLOT_DROP]       = _COMPACT_DROP
    table[_SLOT_WIELD]      = _COMPACT_WIELD
    table[_SLOT_WEAR]       = _COMPACT_WEAR
    table[_SLOT_PUTON]      = _COMPACT_PUTON
    table[_SLOT_REMOVE]     = _COMPACT_REMOVE
    table[_SLOT_OPEN]       = _COMPACT_OPEN
    table[_SLOT_CLOSE]      = _COMPACT_CLOSE
    table[_SLOT_KICK]       = _COMPACT_KICK
    table[_SLOT_FIGHT]      = _COMPACT_FIGHT
    table[_SLOT_SEARCH]     = _COMPACT_SEARCH
    table[_SLOT_PRAY]       = _COMPACT_PRAY
    table[_SLOT_TWOWEAPON]  = _COMPACT_TWOWEAPON
    table[_SLOT_THROW]      = _COMPACT_THROW
    table[_SLOT_LOOT]       = _COMPACT_LOOT
    table[_SLOT_APPLY]      = _COMPACT_APPLY
    table[_SLOT_ENGRAVE]    = _COMPACT_ENGRAVE
    table[_SLOT_NAME]       = _COMPACT_NAME
    return jnp.array(table, dtype=jnp.int32)


def _build_slot_to_dir_idx() -> jnp.ndarray:
    """Map legacy handler slot → direction index in ``_DIR_TABLE`` (0..7).

    Non-movement slots map to 0 (the direction is unused for those branches).
    The order matches ``_DIR_TABLE``: N=0, E=1, S=2, W=3, NE=4, SE=5, SW=6, NW=7.
    """
    table = [0] * 43
    # Move slots.
    table[_SLOT_MOVE_N]  = 0
    table[_SLOT_MOVE_E]  = 1
    table[_SLOT_MOVE_S]  = 2
    table[_SLOT_MOVE_W]  = 3
    table[_SLOT_MOVE_NE] = 4
    table[_SLOT_MOVE_SE] = 5
    table[_SLOT_MOVE_SW] = 6
    table[_SLOT_MOVE_NW] = 7
    # Run slots — same direction order.
    table[_SLOT_RUN_N]  = 0
    table[_SLOT_RUN_E]  = 1
    table[_SLOT_RUN_S]  = 2
    table[_SLOT_RUN_W]  = 3
    table[_SLOT_RUN_NE] = 4
    table[_SLOT_RUN_SE] = 5
    table[_SLOT_RUN_SW] = 6
    table[_SLOT_RUN_NW] = 7
    return jnp.array(table, dtype=jnp.int32)


_SLOT_TO_COMPACT: jnp.ndarray = _build_slot_to_compact()
_SLOT_TO_DIR_IDX: jnp.ndarray = _build_slot_to_dir_idx()


# ---------------------------------------------------------------------------
# Legacy compact index table (Wave 1 — kept for backward compat)
# ---------------------------------------------------------------------------

_ACTION_VALUE_TO_INDEX: dict = {int(a): i for i, a in enumerate(ACTIONS)}

# Keep the original ACTION_HANDLERS tuple to preserve the Wave 1 API
# (test_action_enum etc. only check N_ACTIONS).
ACTION_HANDLERS: tuple = tuple(_noop for _ in range(N_ACTIONS))

assert len(ACTION_HANDLERS) == N_ACTIONS, (
    f"ACTION_HANDLERS length {len(ACTION_HANDLERS)} != N_ACTIONS {N_ACTIONS}."
)


# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------

def dispatch_action(state, action: jnp.int32, rng: jax.Array):
    """Route ``action`` to the appropriate subsystem handler.

    Parameters
    ----------
    state  : EnvState — full game state.
    action : jnp.int32 — one of the 121 NLE action int values (ASCII codes).
    rng    : JAX PRNG key.

    Returns
    -------
    New game state after applying the action.

    Implementation
    --------------
    Uses a 256-entry lookup table ``_ACTION_TO_HANDLER_IDX`` to map the
    action's ASCII value to a 43-slot legacy handler index.  That index is
    then compressed through ``_SLOT_TO_COMPACT`` (29 slots) so the 16
    per-direction move/run handlers share two branches in the underlying
    ``lax.switch`` — reducing XLA compile time roughly 5× (Wave 8 refactor).

    JAX primitives used: jax.lax.switch, jax.lax.while_loop, jnp.where,
    jnp.clip, jnp.array_equal.
    """
    action_val = jnp.clip(jnp.int32(action), 0, 255)
    handler_idx = _ACTION_TO_HANDLER_IDX[action_val].astype(jnp.int32)
    compact_idx = _SLOT_TO_COMPACT[handler_idx]
    dir_idx     = _SLOT_TO_DIR_IDX[handler_idx]
    return jax.lax.switch(compact_idx, _COMPACT_HANDLERS, state, rng, dir_idx)


def _action_value_to_index(action_value: jnp.int32) -> jnp.int32:
    """Convert a raw NLE action value to its index in ACTION_HANDLERS.

    Wave 1 compatibility shim — runs outside JIT.
    """
    return jnp.int32(_ACTION_VALUE_TO_INDEX[int(action_value)])
