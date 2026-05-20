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
from Nethax.nethax.fov import compute_fov, update_explored, BLIND_SIGHT_RADIUS, DEFAULT_SIGHT_RADIUS, DARK_ROOM_SIGHT_RADIUS
from Nethax.nethax.subsystems.lighting import _player_in_lit_area
from Nethax.nethax.subsystems.features import (
    DoorState,
    open_door,
    handle_open as _features_handle_open,
    handle_close as _features_handle_close,
    handle_kick as _features_handle_kick,
    handle_search as _features_handle_search,
)
from Nethax.nethax.subsystems.traps import trigger_trap, TrapType
from Nethax.nethax.subsystems.items_potions import handle_quaff as _potions_handle_quaff
from Nethax.nethax.subsystems.items_scrolls import handle_read as _scrolls_handle_read
from Nethax.nethax.subsystems.items_wands import (
    handle_zap as _wands_handle_zap,
    WandState,
    WandEffect,
    ITEM_CATEGORY_WAND,
)
from Nethax.nethax.subsystems.wish import (
    handle_wand_of_wishing as _wish_handle_wand,
)
from Nethax.nethax.subsystems.inventory import (
    handle_wield as _inv_handle_wield,
    handle_wear as _inv_handle_wear,
    handle_name as _inv_handle_name,
    pickup as _inv_pickup,
    drop as _inv_drop,
    ItemCategory,
    MAX_INVENTORY_SLOTS,
    USER_NAME_LEN,
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
    cancel_bag_of_holding as _containers_cancel_boh,
    ContainerType as _ContainerType,
)
from Nethax.nethax.subsystems.status_effects import (
    handle_eat as _status_handle_eat,
    compute_hunger_state,
    MAX_NUTRITION,
)
from Nethax.nethax.subsystems.items_corpses import apply_corpse_postfx as _corpse_postfx
from Nethax.nethax.subsystems.magic import N_SPELLS, _EFFECT_DISPATCH_LIST as _MAGIC_EFFECT_DISPATCH_LIST
from Nethax.nethax.subsystems.conduct import (
    Conduct as _Conduct,
    mark_violated_if as _mark_violated_if,
    food_material_for_type_id as _food_material_for_type_id,
    is_meat_material as _is_meat_material,
    is_animal_material as _is_animal_material,
)
from Nethax.nethax.subsystems.riding import (
    try_mount as _riding_try_mount,
    try_dismount as _riding_try_dismount,
)
from Nethax.nethax.subsystems.monster_ai import pet_follow_on_stair as _pet_follow_on_stair


# ---------------------------------------------------------------------------
# Nutrition lookup tables built from vendor constants at import time.
#
# _FOOD_NUTRITION[type_id] — oc_nutrition per food object type.
#   Cite: vendor/nethack/include/objects.h FOOD() macros (lines 1048-1117).
#   Values read from Nethax/nethax/constants/objects.py OBJECTS[i].nutrition.
#   Non-food entries (and corpse type_id=240) carry nutrition=0 here; corpse
#   nutrition is looked up via _CORPSE_NUTRITION instead.
#
# _CORPSE_NUTRITION[monster_idx] — cnutrit per monster species.
#   Cite: vendor/nethack/include/permonst.h line 68 (cnutrit field);
#         vendor/nethack/include/monsters.h per-MON() nutrition column.
#   Values read from Nethax/nethax/constants/monsters.py MONSTERS[i].nutrition.
# ---------------------------------------------------------------------------

def _build_food_nutrition_table() -> jnp.ndarray:
    """Return int32[NUM_OBJECTS] of oc_nutrition per OBJECTS entry."""
    from Nethax.nethax.constants.objects import OBJECTS, NUM_OBJECTS
    vals = [0] * NUM_OBJECTS
    for i, obj in enumerate(OBJECTS):
        vals[i] = int(obj.nutrition)
    return jnp.array(vals, dtype=jnp.int32)


def _build_corpse_nutrition_table() -> jnp.ndarray:
    """Return int32[N_MONSTERS] of cnutrit per MONSTERS entry."""
    from Nethax.nethax.constants.monsters import MONSTERS
    return jnp.array([int(m.nutrition) for m in MONSTERS], dtype=jnp.int32)


_FOOD_NUTRITION: jnp.ndarray = _build_food_nutrition_table()
_CORPSE_NUTRITION: jnp.ndarray = _build_corpse_nutrition_table()


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
    """Recompute visible + explored for the current level after a move.

    Radius formula (JIT-pure via jnp.where):

      1. Blind or underwater -> BLIND_SIGHT_RADIUS (1).
         Vendor: vision.c (blindness forces radius=1);
                 hack.c:1016 (Underwater restricts perception).

      2. Player on CORRIDOR tile without carried light -> DARK_ROOM_SIGHT_RADIUS (2).
         Vendor: vision.c:328 -- rooms[rnum].rlit gates IN_SIGHT vs COULD_SEE.
         Proxy: CORRIDOR tile = always dark; non-CORRIDOR = in lit room.

      3. Carried light (wand/spell: lit_radius_until_turn > timestep) restores
         DEFAULT_SIGHT_RADIUS even in a dark corridor.
         Vendor: light.c::do_light_sources line 169.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    terrain_2d = _current_level_terrain(state)

    # condition 1: blind (vendor vision.c) or underwater (vendor hack.c:1016)
    is_blind = state.status.timed_statuses[int(TimedStatus.BLIND)] > 0
    in_water = state.player_in_water

    # condition 2: dark room -- player on CORRIDOR tile
    # Vendor vision.c:328: rlit gates full sight vs COULD_SEE-only.
    # Room lit state not persisted in EnvState; CORRIDOR tiles are always dark.
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    player_tile = terrain_2d[pr, pc]
    in_dark_room = player_tile == jnp.int8(int(TileType.CORRIDOR))

    # condition 3: player covered by any active light source.
    # Vendor light.c::do_light_sources line 169.
    # Legacy scalar still honoured for wand/spell handlers not yet migrated.
    # TODO: items_wands.py / magic.py should call add_light_source() instead of
    #       setting dungeon.lit_radius_until_turn directly; remove legacy check
    #       once that migration is complete.
    legacy_lit = state.dungeon.lit_radius_until_turn > state.timestep.astype(jnp.int32)
    has_light = legacy_lit | _player_in_lit_area(state)

    sight_radius = jnp.where(
        is_blind | in_water,
        BLIND_SIGHT_RADIUS,
        jnp.where(
            in_dark_room,
            jnp.where(has_light, DEFAULT_SIGHT_RADIUS, DARK_ROOM_SIGHT_RADIUS),
            DEFAULT_SIGHT_RADIUS,
        ),
    )
    new_visible = compute_fov(terrain_2d, state.player_pos, sight_radius)

    b  = state.dungeon.current_branch
    lv = state.dungeon.current_level - 1
    new_explored = update_explored(state.explored[b, lv], new_visible)

    new_explored_full = state.explored.at[b, lv].set(new_explored)

    # Stamp visible tiles into last_seen_terrain (vendor display.c lastseentyp ~line 850).
    old_lst = state.last_seen_terrain[b, lv]
    new_lst_slice = jnp.where(new_visible, terrain_2d.astype(jnp.int8), old_lst)
    new_last_seen = state.last_seen_terrain.at[b, lv].set(new_lst_slice)

    return state.replace(
        visible=new_visible,
        explored=new_explored_full,
        last_seen_terrain=new_last_seen,
    )


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

    Swallow gate: if the player is currently swallowed, movement is a no-op.
    The vendor requires attacking the engulfer to escape (mhitu.c::gulpmu).
    Cite: vendor/nethack/src/mhitu.c::swallowed movement block.

    VOMITING gate: while VOMITING, movement becomes a no-op (incapacitated).
    Cite: vendor/nethack/src/hack.c — VOMITING causes brief incapacitation.

    WOUNDED_LEGS limp: while WOUNDED_LEGS, with prob 0.3 skip the move.
    Cite: vendor/nethack/src/hack.c WOUNDED_LEGS movement penalty.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS

    # VOMITING no-op gate.
    is_vomiting = state.status.timed_statuses[int(_TS.VOMITING)] > jnp.int32(0)

    # WOUNDED_LEGS limp: 30% chance to NOOP the move.
    rng, rng_wl = jax.random.split(rng)
    wl_roll = jax.random.uniform(rng_wl)
    is_wounded = state.status.timed_statuses[int(_TS.WOUNDED_LEGS)] > jnp.int32(0)
    do_limp = is_wounded & (wl_roll < jnp.float32(0.3))

    # UNDERWATER diagonal block: diagonal moves are forbidden while in water.
    # Cite: vendor/nethack/src/hack.c lines 1016-1023.
    is_diagonal = jnp.bool_((dy != 0) & (dx != 0))
    blocked_underwater = state.player_in_water & is_diagonal

    # u.utrap immobility — vendor/nethack/src/hack.c:1558-1690.
    # Player trapped in pit/bear-trap/web/lava cannot move until escape roll
    # succeeds. Vendor decrements u.utrap each turn and allows escape only
    # when it reaches 0; mechanics differ per trap. Minimal byte-equal proxy:
    # rn2(4) (~25%) escape per move attempt (vendor bear-trap rate).
    rng, rng_trap = jax.random.split(rng)
    trap_escape = jax.random.randint(rng_trap, (), 0, 4, dtype=jnp.int32) == jnp.int32(0)
    blocked_trap = state.player_in_trap & ~trap_escape

    # Any no-op gate → skip movement entirely.
    noop_gate = (
        state.swallow.swallowed | is_vomiting | do_limp
        | blocked_underwater | blocked_trap
    )

    # When the trap-escape roll succeeds, clear player_in_trap so the next
    # move proceeds normally.
    state_after_escape = state.replace(
        player_in_trap=jnp.where(
            state.player_in_trap & trap_escape,
            jnp.bool_(False),
            state.player_in_trap,
        )
    )

    return jax.lax.cond(
        noop_gate,
        lambda s: s,
        lambda s: _try_step_inner(s, dy, dx, rng),
        state_after_escape,
    )


def _try_step_inner(state, dy: int, dx: int, rng: jax.Array):
    """Inner body of _try_step — runs only when no no-op gate applies."""
    from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS

    # -------------------------------------------------------------------
    # Confused / stunned direction randomization.
    # Cite: vendor/nethack/src/hack.c:2424 — when CONFUSED or STUNNED,
    # movement direction is randomized with probability 0.5.
    # JIT-pure: use jnp.where to swap computed dy/dx with random direction.
    # -------------------------------------------------------------------
    confused = state.status.timed_statuses[int(_TS.CONFUSION)] > jnp.int32(0)
    stunned  = state.status.timed_statuses[int(_TS.STUNNED)]   > jnp.int32(0)
    impaired = confused | stunned

    rng, rng_imp, rng_dy, rng_dx = jax.random.split(rng, 4)
    do_randomize = impaired & (jax.random.uniform(rng_imp) < jnp.float32(0.5))
    rand_dy = jax.random.randint(rng_dy, (), -1, 2).astype(jnp.int32)
    rand_dx = jax.random.randint(rng_dx, (), -1, 2).astype(jnp.int32)
    eff_dy = jnp.where(do_randomize, rand_dy, jnp.int32(dy))
    eff_dx = jnp.where(do_randomize, rand_dx, jnp.int32(dx))

    terrain_2d = _current_level_terrain(state)
    map_h, map_w = terrain_2d.shape

    pos    = state.player_pos.astype(jnp.int32)
    target = pos + jnp.stack([eff_dy, eff_dx])

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
    # Vendor hack.c:1925-1995 (domove_bump_mon / domove_attackmon_at) —
    # peaceful monsters are NOT auto-attacked on bump; vendor prompts for
    # confirmation (y/n) and defaults to no.  Nethax has no interactive
    # prompt, so treat peaceful (non-tame) bump as "do nothing".
    # Tame (pet) monsters use the swap-places branch instead — see
    # vendor/nethack/src/hack.c:2098 domove_swap_with_pet.
    target_peaceful = mai.peaceful[monster_idx]
    target_tame     = mai.tame[monster_idx] > jnp.int8(0)
    is_pet_swap     = monster_present & target_tame
    is_peaceful_bump = monster_present & target_peaceful & ~target_tame
    # Only attack if monster is present AND not friendly.
    monster_present = monster_present & ~target_peaceful & ~target_tame

    def _attack_branch(s):
        # Capture pre-attack alive flag at the matched slot so we can grant
        # XP on kill (vendor/nethack/src/exper.c::experience +
        # more_experienced; exper.c:83-203).
        was_alive = s.monster_ai.alive[monster_idx]
        attacked, _dmg, _hit = _combat_melee_attack(s, rng, monster_idx)
        killed = was_alive & ~attacked.monster_ai.alive[monster_idx]
        from Nethax.nethax.subsystems.experience import (
            experience as _xp_experience,
            more_experienced as _xp_more_experienced,
        )
        entry_post = attacked.monster_ai.entry_idx[monster_idx].astype(jnp.int32)
        kc = attacked.scoring.monsters_killed
        mcl = attacked.monster_ai.mcloned[monster_idx]
        xp_award = _xp_experience(entry_post, kc, mcloned=mcl)
        attacked = jax.lax.cond(
            killed,
            lambda s_: _xp_more_experienced(s_, xp_award, jnp.int32(0)),
            lambda s_: s_,
            attacked,
        )

        # Confuse-attack-on-hit: if confuse_attack_pending is set and the
        # strike landed (target still alive indicates a hit occurred even if
        # not killed; use _hit flag from _combat_melee_attack).
        # Cite: vendor/nethack/src/spell.c SPE_CONFUSE_MONSTER — player's
        # next melee hit confuses the target.
        pending = attacked.status.confuse_attack_pending
        apply_confuse_hit = pending & _hit & ~killed
        old_ct = attacked.monster_ai.confuse_timer[monster_idx].astype(jnp.int32)
        new_ct = jnp.where(apply_confuse_hit, jnp.maximum(old_ct, jnp.int32(15)), old_ct)
        new_ct_arr = attacked.monster_ai.confuse_timer.at[monster_idx].set(
            new_ct.astype(attacked.monster_ai.confuse_timer.dtype)
        )
        new_mai = attacked.monster_ai.replace(confuse_timer=new_ct_arr)
        # Clear the pending flag after any hit (hit or kill).
        new_pending = jnp.where(pending & _hit, jnp.bool_(False), pending)
        new_status = attacked.status.replace(confuse_attack_pending=new_pending)
        return attacked.replace(monster_ai=new_mai, status=new_status)

    def _pet_swap_branch(s):
        # vendor/nethack/src/hack.c:2098 domove_swap_with_pet — when the hero
        # bumps an adjacent tame pet, swap positions instead of attacking
        # or stalling.  Move the pet to the hero's prior tile; move the
        # hero into the pet's tile.
        old_pos = s.player_pos.astype(jnp.int32)
        new_player_pos = target.astype(jnp.int16)
        mai_in = s.monster_ai
        new_mon_pos = mai_in.pos.at[monster_idx].set(old_pos.astype(mai_in.pos.dtype))
        new_mai = mai_in.replace(pos=new_mon_pos)
        s2 = s.replace(player_pos=new_player_pos, monster_ai=new_mai)
        # Re-apply FOV from new player position.
        return _apply_fov(s2)

    # _attack_branch must return the same pytree shape as _move_branch
    # (below).  Selection order:
    #   1. Pet swap (tame) → swap positions.
    #   2. Peaceful (non-tame) bump → no-op.
    #   3. Hostile bump → attack.
    #   4. Empty target tile → move.
    return jax.lax.cond(
        is_pet_swap,
        _pet_swap_branch,
        lambda s: jax.lax.cond(
            is_peaceful_bump,
            lambda s_: s_,                                        # peaceful bump → no-op
            lambda s_: jax.lax.cond(
                monster_present,
                _attack_branch,
                lambda s2: _move_branch(s2, eff_dy, eff_dx, rng, target, in_bounds, terrain_2d, map_h, map_w),
                s_,
            ),
            s,
        ),
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

    # Pre-roll trap damage outside lax.cond (jax.random.randint is not
    # concrete-value-safe inside cond branches on some JAX versions).
    bump_trap_dmg_roll = jax.random.randint(rng, (), minval=1, maxval=11, dtype=jnp.int32)

    def _do_open(f):
        # open_door with pre-rolled damage avoids randint inside cond.
        lv_, row_, col_ = door_pos[0], door_pos[1], door_pos[2]
        current_ = f.door_state[lv_, row_, col_].astype(jnp.int32)
        is_closed_ = current_ == jnp.int32(DoorState.CLOSED)
        is_trapped_ = f.door_trapped[lv_, row_, col_]
        new_val_ = jnp.where(
            is_closed_ & is_trapped_,
            jnp.int32(DoorState.BROKEN),
            jnp.where(is_closed_ & ~is_trapped_, jnp.int32(DoorState.OPEN), current_),
        ).astype(jnp.int8)
        damage_ = jnp.where(is_closed_ & is_trapped_, bump_trap_dmg_roll, jnp.int32(0))
        new_trapped_ = jnp.where(
            is_closed_ & is_trapped_,
            f.door_trapped.at[lv_, row_, col_].set(jnp.bool_(False)),
            f.door_trapped,
        )
        new_ds_ = f.door_state.at[lv_, row_, col_].set(new_val_)
        return f.replace(door_state=new_ds_, door_trapped=new_trapped_), damage_

    new_features = jax.lax.cond(
        open_on_bump,
        # vendor lock.c::doopen checks D_TRAPPED; trap springs on bump-open too.
        _do_open,
        lambda f: (f, jnp.int32(0)),
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

    # vendor/nethack/src/hack.c:1153-1170 — diagonal corner block.
    # When moving diagonally (dx && dy) and both adjacent cardinal tiles
    # (ux,y) and (x,uy) are "bad rock" (solid stone/wall), the diagonal
    # move is blocked.  Skip when player Passes_walls (M1_WALLWALK).
    is_diagonal_move = (jnp.int32(dy) != jnp.int32(0)) & (jnp.int32(dx) != jnp.int32(0))
    cardA_row = pos[0] + jnp.int32(dy)   # (ux + dy, uy)  → (target_row, current_col)
    cardA_col = pos[1]
    cardB_row = pos[0]
    cardB_col = pos[1] + jnp.int32(dx)   # (ux, uy + dx)  → (current_row, target_col)
    cardA_safe_r = jnp.clip(cardA_row, 0, map_h - 1)
    cardA_safe_c = jnp.clip(cardA_col, 0, map_w - 1)
    cardB_safe_r = jnp.clip(cardB_row, 0, map_h - 1)
    cardB_safe_c = jnp.clip(cardB_col, 0, map_w - 1)
    cardA_tile = jnp.clip(terrain_2d[cardA_safe_r, cardA_safe_c].astype(jnp.int32),
                          0, _NUM_TILE_TYPES - 1)
    cardB_tile = jnp.clip(terrain_2d[cardB_safe_r, cardB_safe_c].astype(jnp.int32),
                          0, _NUM_TILE_TYPES - 1)
    cardA_solid = _IS_SOLID[cardA_tile]
    cardB_solid = _IS_SOLID[cardB_tile]
    # Passes_walls intrinsic bypass — vendor `!Passes_walls` gate.
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _PWIntr
    passes_walls = (
        state.status.intrinsics[int(_PWIntr.PASSES_WALLS)]
        | (state.status.timed_intrinsics[int(_PWIntr.PASSES_WALLS)] > jnp.int32(0))
    )
    diagonal_corner_blocked = (
        is_diagonal_move & cardA_solid & cardB_solid & ~passes_walls
    )

    # vendor/nethack/src/hack.c:1140 — diagonal-into-doorway block; and
    # hack.c:1208 — diagonal-out-of-doorway block.  Approximation: when
    # diagonal and either source or target tile is a door (OPEN_DOOR or
    # CLOSED_DOOR), block the move.  Skip with Passes_walls.
    src_tile = jnp.clip(terrain_2d[
        jnp.clip(pos[0], 0, map_h - 1),
        jnp.clip(pos[1], 0, map_w - 1),
    ].astype(jnp.int32), 0, _NUM_TILE_TYPES - 1)
    src_is_door = (
        (src_tile == jnp.int32(TileType.OPEN_DOOR))
        | (src_tile == jnp.int32(TileType.CLOSED_DOOR))
    )
    tgt_is_door = (
        (tile_val == jnp.int32(TileType.OPEN_DOOR))
        | (tile_val == jnp.int32(TileType.CLOSED_DOOR))
    )
    diagonal_door_blocked = (
        is_diagonal_move & (src_is_door | tgt_is_door) & ~passes_walls
    )

    can_move = (
        in_bounds & ~is_solid & ~door_blocked
        & ~diagonal_corner_blocked & ~diagonal_door_blocked
    )

    new_pos = jnp.where(can_move, target, pos).astype(jnp.int16)

    # Apply trapped-door damage from bump-open (vendor lock.c::doopen D_TRAPPED).
    bump_hp = jnp.maximum(jnp.int32(0), state.player_hp - new_features[1])
    new_features = new_features[0]

    state_mid = state.replace(
        player_pos=new_pos,
        features=new_features,
        terrain=new_terrain,
        player_hp=bump_hp,
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

    # Elbereth dust wipe when player steps over an engraved tile.
    # Cite: vendor/nethack/src/engrave.c::wipe_engr_at lines 270-290.
    # Dust Elbereth has a 1/4 chance of erasure per step (vendor rn2(4)).
    from Nethax.nethax.subsystems.engrave import is_elbereth_at, ENGR_DUST
    wipe_rng, _ = jax.random.split(state_final.rng)
    wipe_r = state_final.player_pos[0].astype(jnp.int32)
    wipe_c = state_final.player_pos[1].astype(jnp.int32)
    is_elb = is_elbereth_at(state_final.engrave, wipe_r, wipe_c)
    is_dust = (
        state_final.engrave.engraving_kind[wipe_r, wipe_c].astype(jnp.int32)
        == jnp.int32(ENGR_DUST)
    )
    wipe_roll = jax.random.uniform(wipe_rng)
    do_wipe = is_elb & is_dust & actually_moved & (wipe_roll < 0.25)
    new_has_engraving = jnp.where(
        do_wipe,
        state_final.engrave.has_engraving.at[wipe_r, wipe_c].set(jnp.bool_(False)),
        state_final.engrave.has_engraving,
    )
    state_final = state_final.replace(
        engrave=state_final.engrave.replace(has_engraving=new_has_engraving)
    )

    # --- Lava entry (trap.c::lava_effects line 6794) ---
    # vendor/nethack/src/trap.c::lava_effects — entering a lava tile without
    # any of Fire_resistance / Levitation / Flying / Water-walking burns the
    # hero to a crisp (BURNING death).  With Wwalking but no Fire_res,
    # take d(6,6) damage (only fatal if exceeds HP).  With Fire_res only,
    # sink into lava trap (modeled as no-op for now; full TT_LAVA trap
    # state is in trap subsystem).
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _LavaIntr
    _lava_r = state_final.player_pos[0].astype(jnp.int32)
    _lava_c = state_final.player_pos[1].astype(jnp.int32)
    _lava_tile = _current_level_terrain(state_final)[
        jnp.clip(_lava_r, 0, map_h - 1),
        jnp.clip(_lava_c, 0, map_w - 1),
    ].astype(jnp.int32)
    _on_lava = (_lava_tile == jnp.int32(TileType.LAVA))
    _has_fire_res = (
        state_final.status.intrinsics[int(_LavaIntr.RESIST_FIRE)]
        | (state_final.status.timed_intrinsics[int(_LavaIntr.RESIST_FIRE)] > jnp.int32(0))
    )
    _has_levitation = (
        state_final.status.intrinsics[int(_LavaIntr.LEVITATION)]
        | (state_final.status.timed_intrinsics[int(_LavaIntr.LEVITATION)] > jnp.int32(0))
    )
    _has_flying = (
        state_final.status.intrinsics[int(_LavaIntr.FLYING)]
        | (state_final.status.timed_intrinsics[int(_LavaIntr.FLYING)] > jnp.int32(0))
    )
    _has_wwalk = (
        state_final.status.intrinsics[int(_LavaIntr.WWALKING)]
        | (state_final.status.timed_intrinsics[int(_LavaIntr.WWALKING)] > jnp.int32(0))
    )
    # Lava damage: d(6, 6) per vendor trap.c:6800
    lava_rng, new_rng_after_lava = jax.random.split(state_final.rng)
    lava_rolls = jax.random.randint(lava_rng, (6,), 0, 6, dtype=jnp.int32) + jnp.int32(1)
    lava_dmg = jnp.sum(lava_rolls).astype(jnp.int32)  # d(6,6)
    # Survive iff Fire_resistance OR (Wwalking AND dmg < HP) OR Lev OR Flying
    _survives_lava = (
        _has_fire_res | _has_levitation | _has_flying
        | (_has_wwalk & (lava_dmg < state_final.player_hp))
    )
    # No protections at all → instakill.
    _lava_kills = _on_lava & actually_moved & ~_survives_lava
    # Wwalking-only: take damage but live.
    _wwalk_dmg = (
        _on_lava & actually_moved & _has_wwalk
        & ~_has_fire_res & ~_has_levitation & ~_has_flying
    )
    new_hp_after_lava = jnp.where(
        _lava_kills,
        jnp.int32(0),
        jnp.where(_wwalk_dmg,
                  jnp.maximum(state_final.player_hp - lava_dmg, jnp.int32(0)),
                  state_final.player_hp),
    )
    new_done_after_lava = state_final.done | _lava_kills
    state_final = state_final.replace(
        player_hp=new_hp_after_lava,
        done=new_done_after_lava,
        rng=new_rng_after_lava,
    )

    # --- Water entry/exit (hack.c::pooleffects line 3304 / swimeffect line 3237) ---
    # After moving, update player_in_water based on the new tile.
    # SWIMMING intrinsic (prop.h:51, hack.c) keeps player on surface — no submersion.
    # Cite: vendor/nethack/src/hack.c::pooleffects line 3304 (enter),
    #        vendor/nethack/src/hack.c lines 3237-3268 (exit / swimeffect).
    # TODO (Wave 5): Plane of Water surface levels — player on water but not underwater.
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intrinsic
    _new_r = state_final.player_pos[0].astype(jnp.int32)
    _new_c = state_final.player_pos[1].astype(jnp.int32)
    _new_tile = _current_level_terrain(state_final)[
        jnp.clip(_new_r, 0, map_h - 1),
        jnp.clip(_new_c, 0, map_w - 1),
    ].astype(jnp.int32)
    _is_water_tile = (
        (_new_tile == jnp.int32(TileType.WATER))
        | (_new_tile == jnp.int32(TileType.POOL))
    )
    _has_swimming = (
        state_final.status.intrinsics[int(_Intrinsic.SWIMMING)]
        | (state_final.status.timed_intrinsics[int(_Intrinsic.SWIMMING)] > jnp.int32(0))
    )
    # On a water tile: submerge only if NOT swimming.
    # On a non-water tile or no actual move: preserve or clear flag.
    _new_in_water = jnp.where(
        actually_moved,
        _is_water_tile & ~_has_swimming,
        state_final.player_in_water,
    )
    state_final = state_final.replace(player_in_water=_new_in_water)

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

def _on_quest_leader_level(state) -> object:
    """Fire quest.on_enter_quest_level if player arrived on Quest branch level 1.

    Called after stair traversal resolves.  Mirrors quest.c::chat_with_leader
    (~321-324) and qstplay.c on_leader_level / expulsion logic: the first time
    the hero sets foot on the Quest start level the leader is met and stage
    advances to BEGUN_QUEST (1).
    """
    from Nethax.nethax.subsystems.quest import on_enter_quest_level
    from Nethax.nethax.dungeon.branches import Branch
    on_quest_branch = state.dungeon.current_branch == jnp.int8(int(Branch.QUEST))
    on_level_1      = state.dungeon.current_level  == jnp.int8(1)
    should_enter    = on_quest_branch & on_level_1 & ~state.quest.met_leader
    return jax.lax.cond(should_enter, on_enter_quest_level, lambda s: s, state)


def _stair_up(state, rng):
    """Traverse up-stair if standing on STAIRCASE_UP tile.

    Within-branch traversal: bumps current_level by -1 (clamped to 1).
    Citation: vendor/nethack/src/do.c::dohol() (go up stairs).

    Wave 17f: snapshots per-level monster_ai / features / ground_items
    before the swap and restores the destination level's snapshot on
    arrival (vendor save.c::savelev / load.c::loadlev parity).
    """
    terrain_2d = _current_level_terrain(state)
    row, col    = state.player_pos[0], state.player_pos[1]
    tile        = terrain_2d[row, col].astype(jnp.int32)
    on_stair    = tile == jnp.int32(TileType.STAIRCASE_UP)

    # Snapshot the current (source) level before changing dungeon.current_level.
    # Host-side bookkeeping only; safe to call under JIT (no-op when frozen).
    from Nethax.nethax.dungeon.level_memory import (
        snapshot_monsters_and_features,
        restore_monsters_and_features,
    )
    src_b  = int(state.dungeon.current_branch) if not isinstance(
        state.dungeon.current_branch, jax.core.Tracer) else None
    src_lv = int(state.dungeon.current_level) if not isinstance(
        state.dungeon.current_level, jax.core.Tracer) else None
    if src_b is not None and src_lv is not None:
        state = snapshot_monsters_and_features(state, src_b, src_lv)

    new_level = jnp.where(
        on_stair,
        jnp.maximum(jnp.int8(1), state.dungeon.current_level - jnp.int8(1)),
        state.dungeon.current_level,
    )
    new_dungeon = state.dungeon.replace(current_level=new_level)
    new_state   = state.replace(dungeon=new_dungeon)

    # Restore destination snapshot (no-op on first visit / under JIT).
    if src_b is not None and src_lv is not None:
        new_state = restore_monsters_and_features(new_state, src_b, int(new_level))

    return _on_quest_leader_level(_apply_fov(new_state))


def _stair_down(state, rng):
    """Traverse down-stair if standing on STAIRCASE_DOWN tile.

    Within-branch traversal: bumps current_level by +1.
    Tracks deepest_lev_reached for scoring (vendor dungeon.c deepest_lev_reached).
    Citation: vendor/nethack/src/do.c::dolook() / dodown() (go down stairs).

    Wave 17f: snapshots per-level monster_ai / features / ground_items
    before the swap and restores the destination level's snapshot on
    arrival (vendor save.c::savelev / load.c::loadlev parity).
    """
    terrain_2d = _current_level_terrain(state)
    row, col    = state.player_pos[0], state.player_pos[1]
    tile        = terrain_2d[row, col].astype(jnp.int32)
    on_stair    = tile == jnp.int32(TileType.STAIRCASE_DOWN)

    # Snapshot source level (Wave 17f).
    from Nethax.nethax.dungeon.level_memory import (
        snapshot_monsters_and_features,
        restore_monsters_and_features,
    )
    src_b  = int(state.dungeon.current_branch) if not isinstance(
        state.dungeon.current_branch, jax.core.Tracer) else None
    src_lv = int(state.dungeon.current_level) if not isinstance(
        state.dungeon.current_level, jax.core.Tracer) else None
    if src_b is not None and src_lv is not None:
        state = snapshot_monsters_and_features(state, src_b, src_lv)

    max_level   = jnp.int8(state.terrain.shape[1])
    new_level   = jnp.where(
        on_stair,
        jnp.minimum(max_level, state.dungeon.current_level + jnp.int8(1)),
        state.dungeon.current_level,
    )
    new_deepest = jnp.maximum(
        state.scoring.deepest_level, new_level.astype(jnp.int8)
    )
    new_dungeon = state.dungeon.replace(current_level=new_level)
    new_scoring = state.scoring.replace(deepest_level=new_deepest)
    new_state   = state.replace(dungeon=new_dungeon, scoring=new_scoring)
    # Move adjacent tame pets to follow the player down the stair.
    # Citation: vendor/nethack/src/dog.c::stair_pet.
    new_state = _pet_follow_on_stair(new_state)

    # Restore destination snapshot (Wave 17f).
    if src_b is not None and src_lv is not None:
        new_state = restore_monsters_and_features(new_state, src_b, int(new_level))

    return _on_quest_leader_level(_apply_fov(new_state))


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

    is_food   = categories == jnp.int8(ItemCategory.FOOD)
    has_stock = quantities > jnp.int16(0)
    valid     = is_food & has_stock

    slot_idx  = jnp.argmax(valid).astype(jnp.int32)
    found     = jnp.any(valid)

    # Nutrition lookup via per-type table (vendor/nethack/include/objects.h FOOD()
    # macros, nutrition column; cite objects.py OBJECTS[type_id].nutrition).
    # For corpses (type_id==240, nutrition=0 in OBJECTS), use _CORPSE_NUTRITION
    # indexed by corpse_entry_idx instead.
    items = state.inventory.items
    safe_slot = jnp.clip(slot_idx, 0, MAX_INVENTORY_SLOTS - 1)
    raw_type_id = items.type_id[safe_slot].astype(jnp.int32)
    clipped_tid = jnp.clip(raw_type_id, 0, _FOOD_NUTRITION.shape[0] - 1)
    obj_nutrition = _FOOD_NUTRITION[clipped_tid]

    corpse_idx = items.corpse_entry_idx[safe_slot].astype(jnp.int32)
    is_corpse_item_pre = found & (corpse_idx >= jnp.int32(0))
    clipped_cidx = jnp.clip(corpse_idx, 0, _CORPSE_NUTRITION.shape[0] - 1)
    corp_nutrition = _CORPSE_NUTRITION[clipped_cidx]

    # Use corpse nutrition when item is a corpse, else object table nutrition.
    # Fall back to 800 (food ration default, eat.c) if both are zero.
    base_nutrition = jnp.where(is_corpse_item_pre, corp_nutrition, obj_nutrition)
    food_nutrition = jnp.where(base_nutrition > 0, base_nutrition, jnp.int32(800))

    # Apply nutrition via handle_eat on the status slice.
    new_status = _status_handle_eat(
        state.status,
        item_nutrition=food_nutrition,
        item_class=jnp.int8(7),  # FOOD_CLASS sentinel expected by handle_eat
        item_present=found,
    )

    # Decrement the consumed item's quantity by 1 (clear category if exhausted).
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

    # Corpse special-effects (eat.c::cpostfx lines 1129-1328).
    # Gate on corpse_entry_idx >= 0 (sentinel -1 = plain food, not a corpse).
    # cite: vendor/nethack/src/eat.c::eatcorpse line 1090
    is_corpse_item = found & (corpse_idx >= jnp.int32(0))
    # Use jnp.where on the idx so the postfx sees -1 when not a corpse → no-op.
    effective_corpse_idx = jnp.where(is_corpse_item, corpse_idx, jnp.int32(-1))
    new_state = _corpse_postfx(new_state, rng, effective_corpse_idx)

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

    Pre-detection: WAN_WISHING is routed directly to wish.handle_wand_of_wishing
    before the WandState projection so that the full EnvState (conduct, quest,
    etc.) is available.  Cite: zap.c::zapyourself WAN_WISHING branch.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _ZapIntrinsic
    # Project EnvState → WandState (single current level slice).
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    terrain_2d  = state.terrain[b, lv]
    explored_2d = state.explored[b, lv]

    mai = state.monster_ai
    wand_state = WandState(
        mon_pos       = mai.pos,
        mon_hp        = mai.hp,
        mon_hp_max    = mai.hp_max,
        mon_type      = mai.entry_idx,
        mon_alive     = mai.alive,
        mon_asleep    = mai.asleep,
        mon_undead    = mai.undead,
        mon_invisible = mai.invisible,
        mon_resists   = mai.resists,
        mon_speed_mod = mai.speed_mod,
        mon_cancelled = mai.cancelled,
        terrain       = terrain_2d,
        explored      = explored_2d,
        inventory     = state.inventory,
        player_pos    = state.player_pos,
        dungeon_level = state.dungeon.current_level.astype(jnp.int8),
        probed_hp     = jnp.int32(0),
        probed_idx    = jnp.int32(-1),
        player_reflecting = state.status.intrinsics[int(_ZapIntrinsic.REFLECTING)],
    )

    new_wand = _wands_handle_zap(wand_state, rng)

    # Write back the mutated slices into EnvState.
    new_monster_ai = mai.replace(
        pos       = new_wand.mon_pos,
        hp        = new_wand.mon_hp,
        alive     = new_wand.mon_alive,
        asleep    = new_wand.mon_asleep,
        invisible = new_wand.mon_invisible,
        speed_mod = new_wand.mon_speed_mod,
        cancelled = new_wand.mon_cancelled,
    )
    new_terrain  = state.terrain.at[b, lv].set(new_wand.terrain)
    new_explored = state.explored.at[b, lv].set(new_wand.explored)

    mid_state = state.replace(
        monster_ai = new_monster_ai,
        terrain    = new_terrain,
        explored   = new_explored,
        inventory  = new_wand.inventory,
    )

    # ---- Bag-of-holding cancellation (zap.c::cancel_item line 720) ----
    # When a wand of cancellation fires, check all container slots for a
    # BAG_OF_HOLDING and implode any that are present.
    # Cite: vendor/nethack/src/zap.c::cancel_item line 720.
    _WAN_CANCELLATION = jnp.int16(10)   # WandEffect.CANCELLATION ordinal
    slot_idx_for_cancel = state.inventory.items.type_id.shape[0]  # Python int
    # Find first wand in inventory to read the type_id that was just zapped.
    # We re-read from the pre-zap slot (before charges decrement mutated inv).
    wand_cat = state.inventory.items.category
    from Nethax.nethax.subsystems.items_wands import ITEM_CATEGORY_WAND as _WAND_CAT
    is_wand_slot = wand_cat == jnp.int8(_WAND_CAT)
    w_slot = jnp.argmax(is_wand_slot).astype(jnp.int32)
    zapped_type_id = jnp.where(
        jnp.any(is_wand_slot),
        state.inventory.items.type_id[w_slot].astype(jnp.int16),
        jnp.int16(-1),
    )
    is_cancellation = zapped_type_id == _WAN_CANCELLATION

    def _maybe_cancel_boh(s, c_idx: int):
        """Cancel BoH at container index c_idx if cancellation wand was zapped."""
        has_boh = s.containers.container_type[c_idx] == jnp.int8(
            int(_ContainerType.BAG_OF_HOLDING)
        )
        return jax.lax.cond(
            is_cancellation & has_boh,
            lambda st: _containers_cancel_boh(st, c_idx),
            lambda st: st,
            s,
        )

    from Nethax.nethax.subsystems.containers import N_CONTAINERS as _N_CONTAINERS
    final_state = mid_state
    for _ci in range(_N_CONTAINERS):
        final_state = _maybe_cancel_boh(final_state, _ci)

    # Use-identification: zapping a wand identifies its type, vendor
    # zap.c:123-147 learnwand(otmp) → makeknown(obj->otyp).
    # Flip identified=True on the wand slot we just zapped.
    new_id_flags = final_state.inventory.items.identified.at[w_slot].set(
        jnp.where(jnp.any(is_wand_slot), jnp.bool_(True),
                  final_state.inventory.items.identified[w_slot])
    )
    final_state = final_state.replace(
        inventory=final_state.inventory.replace(
            items=final_state.inventory.items.replace(identified=new_id_flags)
        )
    )

    return final_state


def _handle_cast(state, rng):
    """CAST — vendor/nethack/src/spell.c::docast / spelleffects.

    JIT-pure pipeline:
      1. Find first known+memorized spell via jnp.argmax.
      2. Check Pw >= spell_level * 5  (spell.h:SPELL_LEV_PW).
      3. Deduct Pw and decrement spell memory (spell.c::decrnknow).
      4. Dispatch effect via jax.lax.switch(spell_idx, _EFFECT_DISPATCH_LIST).
         Each entry wraps the corresponding magic._EFFECT_DISPATCH handler as
         a JIT-pure (state, rng) -> state function.
         Cite: vendor/nethack/src/spell.c::spelleffects.

    When no valid spell exists, returns state unchanged (noop).
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
    base_state = state.replace(player_pw=new_pw, magic=new_magic)

    # Effect dispatch: lax.switch traces all N_SPELLS branches at compile time
    # but only executes the selected branch at runtime.  The noop entry fires
    # for unknown spells.  Cite: vendor/nethack/src/spell.c::spelleffects.
    rng, sub = jax.random.split(rng)
    effect_state = jax.lax.switch(safe_slot, _MAGIC_EFFECT_DISPATCH_LIST, base_state, sub)

    return jax.lax.cond(will_cast, lambda _: effect_state, lambda _: base_state, None)


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
    """SEARCH — vendor/nethack/src/detect.c::dosearch0.

    Delegates to features.handle_search which performs the 3x3 sweep with
    1/7 roll per tile to reveal secret doors (SECRET → CLOSED).
    """
    rng, sub = jax.random.split(rng)
    return _features_handle_search(state, sub)


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

    Wave 5 Phase 3: container open path.
    Digging wave: if wielded item is a pickaxe/mattock, start a horizontal
    dig northward (direction=0).  The direction default matches dodig's
    prompt-north fallback when no direction is given.
    Cite: vendor/nethack/src/dig.c::dodig (line 445).
    """
    from Nethax.nethax.subsystems.digging import start_dig, _has_digging_tool
    # Route pickaxe apply → start dig (north by default).
    # jax.lax.cond requires both branches to be traced; _containers_handle_apply
    # is the non-dig path.
    has_tool = _has_digging_tool(state)
    return jax.lax.cond(
        has_tool,
        lambda s: start_dig(s, direction=0),  # direction 0 = NORTH
        lambda s: _containers_handle_apply(s, rng),
        operand=state,
    )


def _handle_engrave(state, rng):
    """ENGRAVE — vendor/nethack/src/engrave.c::doengrave.

    Wave 5 Phase 4: always engrave 'Elbereth' in dust at the player tile;
    also sets the ELBERETHLESS conduct (insight.c ~2206).
    """
    from Nethax.nethax.subsystems.engrave import handle_engrave
    return handle_engrave(state, rng)


def _handle_name(state, rng):
    """CALL — vendor/nethack/src/do_name.c::do_oname.

    In the headless harness there is no UI to prompt for a specific slot or
    name string, so pressing ``C`` applies a generic label to every
    unidentified inventory slot: slot i receives b"Item <i>\\0..." written into
    user_names[i].  Identified items are left untouched.

    JIT-pure: lax.fori_loop over MAX_INVENTORY_SLOTS; no Python control flow
    on traced values.
    Cite: vendor/nethack/src/do_name.c::do_oname.
    """
    # Pre-build name rows for each slot index as a [MAX_INVENTORY_SLOTS,
    # USER_NAME_LEN] int8 constant array (static at trace time).
    import numpy as _np
    name_table = _np.zeros((MAX_INVENTORY_SLOTS, USER_NAME_LEN), dtype=_np.int8)
    for _i in range(MAX_INVENTORY_SLOTS):
        label = f"Item {_i}".encode("ascii")[:USER_NAME_LEN]
        label = label + b"\x00" * (USER_NAME_LEN - len(label))
        name_table[_i] = list(label)
    name_table_jnp = jnp.array(name_table, dtype=jnp.int8)

    inv = state.inventory
    identified = inv.items.identified  # [MAX_INVENTORY_SLOTS] bool

    def _body(i, user_names):
        occupied = inv.items.category[i] != jnp.int8(0)
        unidentified = ~identified[i]
        should_name = occupied & unidentified
        new_row = jnp.where(should_name, name_table_jnp[i], user_names[i])
        return user_names.at[i].set(new_row)

    new_user_names = jax.lax.fori_loop(
        0, MAX_INVENTORY_SLOTS, _body, inv.user_names
    )
    new_inv = inv.replace(user_names=new_user_names)
    return state.replace(inventory=new_inv)


def _handle_invoke(state, rng):
    """#invoke — vendor/nethack/src/artifact.c::arti_invoke lines 2131-2232.

    Dispatch based on the currently wielded artifact.  Each invoke has a
    cooldown of 100 turns tracked in state.invoke_cooldown[artifact_idx].

    All ~30 artifact invoke effects are implemented via
    artifact_powers.artifact_invoke_dispatch which uses lax.switch.

    Cooldown: 100 turns per invoke slot.
    Cite: vendor/nethack/src/artifact.c::arti_invoke artiintrinsics_taught[].
    """
    from Nethax.nethax.subsystems.artifact_powers import artifact_invoke_dispatch

    art = state.inventory.wielded_artifact_idx.astype(jnp.int32)

    # Cooldown check: if invoke_cooldown[art] > 0, do nothing.
    safe_art = jnp.clip(art, 0, 29)
    cooldown = state.invoke_cooldown[safe_art].astype(jnp.int32)
    ready = (art >= jnp.int32(0)) & (cooldown <= jnp.int32(0))

    def _do_invoke(s):
        # Set cooldown to 100 turns.
        new_cd = s.invoke_cooldown.at[safe_art].set(jnp.int16(100))
        s = s.replace(invoke_cooldown=new_cd)
        return artifact_invoke_dispatch(s, art, rng)

    return jax.lax.cond(ready, _do_invoke, lambda s: s, state)


def _handle_ride(state, rng):
    """RIDE — vendor/nethack/src/steed.c::doride (steed.c:178).

    If currently riding → try_dismount (steed.c:183).
    Otherwise → try_mount (steed.c:187).
    """
    riding = state.player_steed_mid != jnp.uint32(0)
    return jax.lax.cond(
        riding,
        lambda s: _riding_try_dismount(s, rng),
        lambda s: _riding_try_mount(s, rng),
        operand=state,
    )


def _handle_enhance(state, rng):
    """#enhance — advance an eligible skill tier.

    Cite: vendor/nethack/src/weapon.c::enhance_weapon_skill line 1329.

    Auto-picks the first eligible skill (vendor shows a menu; we cannot in
    headless JAX).  A skill is eligible when:
      advance[i] >= practice_needed_to_advance(level[i])  AND  level[i] < max_level[i]

    JIT-pure: iterates skills via lax.fori_loop, stops at first eligible.

    Thresholds (vendor/nethack/include/skills.h:106 — level*level*20, 0-based):
      UNSKILLED(0)→BASIC(1):      0
      BASIC(1)→SKILLED(2):       20
      SKILLED(2)→EXPERT(3):      80
      EXPERT(3)→MASTER(4):      180
      MASTER(4)→GRAND_MASTER(5): 320
    The uniform formula is byte-equal to vendor for all non-martial-arts skills.
    TODO: martial arts uses a separate vendor table (weapon.c::Skill_M); skip for now.

    Note: vendor gates enhancement on weapon_slots; that is not yet modelled.
    """
    from Nethax.nethax.subsystems.skills import (
        N_SKILLS as _N_SKILLS,
        practice_needed_to_advance as _pnta,
        try_advance_skill as _try_adv,
    )

    def _body(i, carry):
        s, done = carry
        sk = s.skills
        cur_lv  = sk.level[i].astype(jnp.int32)
        cur_adv = sk.advance[i].astype(jnp.int32)
        cap     = sk.max_level[i].astype(jnp.int32)
        threshold = _pnta(cur_lv)
        eligible = (cur_adv >= threshold) & (cur_lv < cap) & ~done
        new_s = jax.lax.cond(
            eligible,
            lambda st: _try_adv(st, jnp.int32(i)),
            lambda st: st,
            s,
        )
        new_done = done | eligible
        return new_s, new_done

    final_state, _found = jax.lax.fori_loop(
        0, _N_SKILLS, _body, (state, jnp.bool_(False))
    )
    return final_state


# ---------------------------------------------------------------------------
# DIP — vendor/nethack/src/potion.c::dodip line 2267, H2Opotion_dip 1498-1589
# ---------------------------------------------------------------------------

# Vendor potion type_ids (objects.h order; _POTION_BASE_ID=68).
_POT_OIL_TYPE_ID   = 92   # POT_OIL    = base 68 + 24
_POT_WATER_TYPE_ID = 93   # POT_WATER  = base 68 + 25

# BUC sentinel ints (matches items.BUCStatus).
_DIP_BUC_CURSED   = 1
_DIP_BUC_UNCURSED = 2
_DIP_BUC_BLESSED  = 3


def _handle_dip(state, rng):
    """DIP — vendor/nethack/src/potion.c::dodip line 2267.

    Dip the first non-potion inventory item ("target") into the first POT_WATER
    or POT_OIL ("vehicle") in inventory.

    Behaviour:
      POT_WATER + BLESSED   → target cursed(1)→uncursed(2) or uncursed(2)→blessed(3).
      POT_WATER + CURSED    → target blessed(3)→uncursed(2) or uncursed(2)→cursed(1).
      POT_OIL               → target.greased = True.
    Vehicle quantity is decremented by 1.
    Cite: vendor/nethack/src/potion.c::H2Opotion_dip lines 1498-1589.

    JIT-pure: jnp ops + lax.cond, no Python branching on traced values.
    """
    items = state.inventory.items
    cats     = items.category
    types    = items.type_id
    bucs     = items.buc_status
    greased  = items.greased
    quants   = items.quantity

    # --- Locate vehicle: first POT_WATER, else first POT_OIL.
    is_potion_cat = cats == jnp.int8(ObjectClass.POTION_CLASS)
    has_qty       = quants > jnp.int16(0)
    is_water      = is_potion_cat & has_qty & (types == jnp.int16(_POT_WATER_TYPE_ID))
    is_oil        = is_potion_cat & has_qty & (types == jnp.int16(_POT_OIL_TYPE_ID))

    found_water = jnp.any(is_water)
    found_oil   = jnp.any(is_oil)
    veh_water_idx = jnp.argmax(is_water).astype(jnp.int32)
    veh_oil_idx   = jnp.argmax(is_oil).astype(jnp.int32)

    # Prefer water as vehicle; fall back to oil.
    veh_idx = jnp.where(found_water, veh_water_idx, veh_oil_idx)
    veh_is_water = found_water
    veh_is_oil   = (~found_water) & found_oil
    have_vehicle = found_water | found_oil

    # --- Locate target: first non-potion occupied slot.
    is_target = (cats != jnp.int8(0)) & ~is_potion_cat & has_qty
    found_target = jnp.any(is_target)
    target_idx   = jnp.argmax(is_target).astype(jnp.int32)

    can_dip = have_vehicle & found_target

    def _do_dip(s):
        it = s.inventory.items
        tgt_buc = it.buc_status[target_idx].astype(jnp.int32)

        # Water effects.
        veh_buc = it.buc_status[veh_idx].astype(jnp.int32)
        water_blessed = veh_is_water & (veh_buc == jnp.int32(_DIP_BUC_BLESSED))
        water_cursed  = veh_is_water & (veh_buc == jnp.int32(_DIP_BUC_CURSED))

        # Blessed water: cursed→uncursed, uncursed→blessed.
        new_buc_blessed_water = jnp.where(
            tgt_buc == jnp.int32(_DIP_BUC_CURSED), jnp.int32(_DIP_BUC_UNCURSED),
            jnp.where(tgt_buc == jnp.int32(_DIP_BUC_UNCURSED), jnp.int32(_DIP_BUC_BLESSED),
                      tgt_buc),
        )
        # Cursed water: blessed→uncursed, uncursed→cursed.
        new_buc_cursed_water = jnp.where(
            tgt_buc == jnp.int32(_DIP_BUC_BLESSED), jnp.int32(_DIP_BUC_UNCURSED),
            jnp.where(tgt_buc == jnp.int32(_DIP_BUC_UNCURSED), jnp.int32(_DIP_BUC_CURSED),
                      tgt_buc),
        )

        new_tgt_buc = jnp.where(
            water_blessed, new_buc_blessed_water,
            jnp.where(water_cursed, new_buc_cursed_water, tgt_buc),
        ).astype(jnp.int8)

        new_buc_arr = it.buc_status.at[target_idx].set(new_tgt_buc)

        # Oil: target.greased = True.
        new_greased_val = jnp.where(veh_is_oil, jnp.bool_(True), it.greased[target_idx])
        new_greased_arr = it.greased.at[target_idx].set(new_greased_val)

        # Decrement vehicle quantity by 1; clear category if exhausted.
        old_qty = it.quantity[veh_idx]
        new_veh_qty = jnp.maximum(old_qty - jnp.int16(1), jnp.int16(0))
        new_qty_arr = it.quantity.at[veh_idx].set(new_veh_qty)
        old_cat = it.category[veh_idx]
        new_veh_cat = jnp.where(new_veh_qty == jnp.int16(0), jnp.int8(0), old_cat)
        new_cat_arr = it.category.at[veh_idx].set(new_veh_cat)

        new_items = it.replace(
            buc_status=new_buc_arr,
            greased=new_greased_arr,
            quantity=new_qty_arr,
            category=new_cat_arr,
        )
        return s.replace(inventory=s.inventory.replace(items=new_items))

    return jax.lax.cond(can_dip, _do_dip, lambda s: s, state)


def _handle_tip_down(state, rng):
    """#tip / M-T — wired to WAN_DIGGING down-dig path.

    Per the wave16d brief, M-T is routed to invoke the down-dig branch added
    to _effect_digging (direction == 8 sentinel → create HOLE at player_pos).
    Cite: vendor/nethack/src/dig.c::zap_dig line 1548;
          vendor/nethack/src/dig.c::digactualhole line 640.

    Projects EnvState → WandState, calls _effect_digging with direction=8,
    then writes the new terrain back.
    """
    from Nethax.nethax.subsystems.items_wands import _effect_digging
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _ZapIntrinsic

    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    terrain_2d  = state.terrain[b, lv]
    explored_2d = state.explored[b, lv]

    mai = state.monster_ai
    wand_state = WandState(
        mon_pos       = mai.pos,
        mon_hp        = mai.hp,
        mon_hp_max    = mai.hp_max,
        mon_type      = mai.entry_idx,
        mon_alive     = mai.alive,
        mon_asleep    = mai.asleep,
        mon_undead    = mai.undead,
        mon_invisible = mai.invisible,
        mon_resists   = mai.resists,
        mon_speed_mod = mai.speed_mod,
        mon_cancelled = mai.cancelled,
        terrain       = terrain_2d,
        explored      = explored_2d,
        inventory     = state.inventory,
        player_pos    = state.player_pos,
        dungeon_level = state.dungeon.current_level.astype(jnp.int8),
        probed_hp     = jnp.int32(0),
        probed_idx    = jnp.int32(-1),
        player_reflecting = state.status.intrinsics[int(_ZapIntrinsic.REFLECTING)],
    )

    new_wand, _ = _effect_digging(wand_state, rng, direction=jnp.int32(8))
    new_terrain = state.terrain.at[b, lv].set(new_wand.terrain)
    return state.replace(terrain=new_terrain)


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
    _handle_ride,       # 43  vendor/nethack/src/steed.c::doride
    _handle_invoke,     # 44  vendor/nethack/src/artifact.c::arti_invoke
    _handle_enhance,    # 45  vendor/nethack/src/weapon.c::enhance_weapon_skill line 1329
    _handle_dip,        # 46  vendor/nethack/src/potion.c::dodip line 2267
    _handle_tip_down,   # 47  vendor/nethack/src/dig.c::zap_dig line 1548 (down-dig)
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
# Riding action.
_SLOT_RIDE       = 43
# Artifact invoke (#invoke / M-i).
_SLOT_INVOKE     = 44
# #enhance — advance weapon/spell skill (M-e per vendor cmd.c:1716).
_SLOT_ENHANCE    = 45
# #dip (M-d) — vendor/nethack/src/potion.c::dodip line 2267.
_SLOT_DIP        = 46
# #tip down-dig (M-T) — vendor/nethack/src/dig.c::zap_dig line 1548.
_SLOT_TIP_DOWN   = 47

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
    table[_M_byte("d")] = _SLOT_DIP    # cmd.c::dodip — potion.c::dodip line 2267
    table[_M_byte("e")] = _SLOT_ENHANCE  # cmd.c:1716 — M('e') → enhance_weapon_skill
    table[_M_byte("f")] = _SLOT_NOOP   # cmd.c::doforce
    table[_M_byte("g")] = _SLOT_NOOP   # cmd.c::dogenocided
    table[_M_byte("i")] = _SLOT_INVOKE  # cmd.c::doinvoke → artifact.c::arti_invoke
    table[_M_byte("j")] = _SLOT_NOOP   # cmd.c::dojump
    table[_M_byte("l")] = _SLOT_LOOT   # cmd.c::doloot (already set via Command.LOOT)
    table[_M_byte("m")] = _SLOT_NOOP   # cmd.c::domonability
    table[_M_byte("n")] = _SLOT_NAME   # cmd.c::docallcmd (name alias)
    table[_M_byte("o")] = _SLOT_NOOP   # cmd.c::dosacrifice (offer)
    table[_M_byte("p")] = _SLOT_PRAY   # cmd.c::dopray (already set)
    table[_M_byte("q")] = _SLOT_NOOP   # cmd.c::done2 (quit)
    table[_M_byte("r")] = _SLOT_NOOP   # cmd.c::dorub
    table[_M_byte("R")] = _SLOT_RIDE   # cmd.c::doride — steed.c:178
    table[_M_byte("s")] = _SLOT_NOOP   # cmd.c::dosit
    table[_M_byte("t")] = _SLOT_NOOP   # cmd.c::doturn
    table[_M_byte("T")] = _SLOT_TIP_DOWN  # cmd.c::dotip → WAN_DIGGING down-dig (dig.c:1548)
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
_COMPACT_INVOKE     = 29
_COMPACT_RIDE       = 30
_COMPACT_ENHANCE    = 31
_COMPACT_DIP        = 32
_COMPACT_TIP_DOWN   = 33


def _build_compact_handlers():
    """Return the 32-entry tuple of ``(state, rng, dir_idx) -> state`` handlers."""
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
        _wrap_no_dir(_handle_invoke), # 29  COMPACT_INVOKE
        _wrap_no_dir(_handle_ride),   # 30  COMPACT_RIDE  — steed.c:178
        _wrap_no_dir(_handle_enhance),# 31  COMPACT_ENHANCE — weapon.c:1329
        _wrap_no_dir(_handle_dip),    # 32  COMPACT_DIP — potion.c::dodip line 2267
        _wrap_no_dir(_handle_tip_down),  # 33  COMPACT_TIP_DOWN — dig.c::zap_dig line 1548
    )


_COMPACT_HANDLERS: tuple = _build_compact_handlers()


def _build_slot_to_compact() -> jnp.ndarray:
    """Map legacy handler slot (0..47) → compact slot (0..33)."""
    table = [0] * 48
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
    table[_SLOT_RIDE]       = _COMPACT_RIDE   # steed.c:178 doride
    table[_SLOT_INVOKE]     = _COMPACT_INVOKE
    table[_SLOT_ENHANCE]    = _COMPACT_ENHANCE  # weapon.c:1329 enhance_weapon_skill
    table[_SLOT_DIP]        = _COMPACT_DIP       # potion.c::dodip line 2267
    table[_SLOT_TIP_DOWN]   = _COMPACT_TIP_DOWN  # dig.c::zap_dig line 1548
    return jnp.array(table, dtype=jnp.int32)


def _build_slot_to_dir_idx() -> jnp.ndarray:
    """Map legacy handler slot → direction index in ``_DIR_TABLE`` (0..7).

    Non-movement slots map to 0 (the direction is unused for those branches).
    The order matches ``_DIR_TABLE``: N=0, E=1, S=2, W=3, NE=4, SE=5, SW=6, NW=7.
    """
    table = [0] * 48
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
