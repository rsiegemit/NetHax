"""Game logic for Tier 3 Combat environments.

Supports: movement with bump-attack (do_melee_attack), trap checking,
multi-step item actions (ZAP 3-step, APPLY 2-step, EAT 2-step),
door operations (open, kick, unlock), terrain damage (lava), levitation tick,
full monster AI, monster attacks, goal check.

Item phase state machine:
  0 = normal play
  1 = awaiting slot for ZAP
  2 = awaiting direction for ZAP
  3 = awaiting slot for APPLY
  4 = awaiting slot for EAT
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    Action, TileType, DIRECTION_VECTORS, ItemType,
    MONSTER_STATS, MONSTER_XP_SCORE,
)
from Nethax.minihax.primitives.combat import do_melee_attack
from Nethax.minihax.primitives.monster_ai import full_monster_ai, do_monster_attacks
from Nethax.minihax.primitives.visibility import compute_visible, update_seen_map
from Nethax.minihax.primitives.movement import move_player, check_stair_goal
from Nethax.minihax.primitives.terrain import apply_terrain_damage
from Nethax.minihax.primitives.items import (
    pickup_item,
    has_any_zappable, check_zap_slot, consume_zap_item,
    apply_death_ray, apply_cold_ray,
    has_any_applicable, has_any_food,
    check_apply_slot, check_food_slot,
    apply_item_at_slot, eat_item_at_slot,
)
from Nethax.minihax.primitives.doors import kick_door, open_door_adjacent, unlock_door_adjacent
from Nethax.minihax.primitives.traps import check_traps, search_traps
from Nethax.minihax.primitives.leveling import check_multi_levelup
from Nethax.minihax.util.game_logic_utils import in_bounds


def combat_step(rng, state, action, params, static_params):
    """Tier 3 combat step function.

    Phases:
    0. Item phase state machine (ZAP/APPLY/EAT multi-step handling)
    1. Player action: move (with bump attack), pickup, kick, open, unlock
    2. Auto-open closed doors on move
    3. Trap check
    4. Terrain damage (lava)
    5. Levitation tick
    6. Full monster AI
    7. Monster attacks
    8. Goal check + reward
    9. Select skip vs normal tick

    Args:
        rng: JAX PRNG key
        state: CombatState
        action: int (0-19)
        params: EnvParams
        static_params: CombatStaticParams

    Returns:
        new_state: CombatState
        reward: float
    """
    map_h = static_params.map_height
    map_w = static_params.map_width

    rng_action, rng_bump, rng_monsters, rng_attacks, rng_terrain, rng_kick, \
        rng_traps, rng_search, rng_zap, rng_next = jax.random.split(rng, 10)

    orig_state = state

    # ================================================================
    # Phase 0: Item phase state machine
    # ================================================================
    phase = state.item_phase
    is_direction = action < 8
    is_zap = action == Action.ZAP
    is_apply = action == Action.APPLY
    is_eat = action == Action.EAT
    is_slot = (action == Action.SLOT_0) | (action == Action.SLOT_1) | (action == Action.SLOT_2)
    slot_idx = action - Action.SLOT_0
    safe_slot_idx = jnp.where(is_slot, slot_idx, 0)

    # --- Phase 0 → 1: ZAP starts slot selection ---
    start_zap = (phase == 0) & is_zap & has_any_zappable(state.inventory)
    # --- Phase 0 → 3: APPLY starts slot selection ---
    start_apply = (phase == 0) & is_apply & has_any_applicable(state.inventory)
    # --- Phase 0 → 4: EAT starts slot selection ---
    start_eat = (phase == 0) & is_eat & has_any_food(state.inventory)

    # --- Phase 1 → 2: SLOT_N selects zap slot ---
    zap_slot_valid, zap_slot_item = check_zap_slot(state.inventory, safe_slot_idx)
    select_zap_slot = (phase == 1) & is_slot & zap_slot_valid
    cancel_phase1 = (phase == 1) & ~(is_slot & zap_slot_valid)

    # --- Phase 2 → 0: Direction fires the wand ---
    resolve_zap = (phase == 2) & is_direction
    cancel_phase2 = (phase == 2) & ~is_direction

    # --- Phase 3 → 0: SLOT_N applies self-targeted item ---
    apply_slot_valid, apply_slot_item = check_apply_slot(state.inventory, safe_slot_idx)
    resolve_apply = (phase == 3) & is_slot & apply_slot_valid
    cancel_phase3 = (phase == 3) & ~(is_slot & apply_slot_valid)

    # --- Phase 4 → 0: SLOT_N eats food ---
    food_slot_valid, food_slot_item = check_food_slot(state.inventory, safe_slot_idx)
    resolve_eat = (phase == 4) & is_slot & food_slot_valid
    cancel_phase4 = (phase == 4) & ~(is_slot & food_slot_valid)

    # Compute new item_phase
    new_item_phase = jnp.where(start_zap, 1,
                     jnp.where(select_zap_slot, 2,
                     jnp.where(start_apply, 3,
                     jnp.where(start_eat, 4,
                     jnp.where(resolve_zap | resolve_apply | resolve_eat
                               | cancel_phase1 | cancel_phase2 | cancel_phase3 | cancel_phase4, 0,
                               phase)))))

    new_pending_slot = jnp.where(select_zap_slot, safe_slot_idx, state.pending_item_slot)

    # Skip game tick: initiating any multi-step action, selecting zap slot, or cancelling
    skip_game_tick = (start_zap | start_apply | start_eat
                      | select_zap_slot
                      | cancel_phase1 | cancel_phase2 | cancel_phase3 | cancel_phase4)

    # --- Resolve ZAP: fire wand from pending slot ---
    zap_dir = DIRECTION_VECTORS[jnp.where(is_direction, action, 0)]
    pending_slot = state.pending_item_slot
    pending_item_type = state.inventory.item_ids[pending_slot]
    is_death_zap = resolve_zap & (pending_item_type == ItemType.WAND_DEATH)
    is_cold_zap = resolve_zap & (pending_item_type != ItemType.WAND_DEATH)

    zapped_monsters, death_killed, death_killed_idx, death_killed_type = \
        apply_death_ray(state.player_position, zap_dir, state.monsters, state.map, map_h, map_w)
    zapped_map = apply_cold_ray(state.player_position, zap_dir, state.map, map_h, map_w)
    zap_consumed_inv = consume_zap_item(state.inventory, pending_slot, resolve_zap)

    xp_gain = jnp.where(is_death_zap & death_killed, MONSTER_STATS[death_killed_type, 13], 0)
    score_gain = jnp.where(is_death_zap & death_killed, MONSTER_XP_SCORE[death_killed_type], 0)
    kill_count = jnp.where(is_death_zap & death_killed, 1, 0)
    post_zap_stats = state.player_stats.replace(
        xp=state.player_stats.xp + xp_gain,
        score=state.player_stats.score + score_gain,
        monsters_killed=state.player_stats.monsters_killed + kill_count,
    )
    post_zap_stats = check_multi_levelup(rng_zap, post_zap_stats)

    # Conditionally apply zap effects
    post_zap_mons = jax.tree.map(
        lambda z, o: jnp.where(is_death_zap, z, o), zapped_monsters, state.monsters)
    post_zap_map = jnp.where(is_cold_zap, zapped_map, state.map)
    post_zap_inv = jax.tree.map(
        lambda z, o: jnp.where(resolve_zap, z, o), zap_consumed_inv, state.inventory)
    post_zap_player_stats = jax.tree.map(
        lambda z, o: jnp.where(resolve_zap, z, o), post_zap_stats, state.player_stats)

    # --- Resolve APPLY: self-targeted item effect ---
    applied_inv, got_levitation, got_key = apply_item_at_slot(
        state.inventory, safe_slot_idx, resolve_apply)
    post_apply_inv = jax.tree.map(
        lambda a, o: jnp.where(resolve_apply, a, o), applied_inv, post_zap_inv)

    apply_levitating = state.player_levitating | got_levitation
    apply_lev_turns = jnp.where(got_levitation, state.levitation_turns + 100, state.levitation_turns)
    apply_has_key = state.player_has_key | got_key

    # --- Resolve EAT: consume food ---
    eaten_inv, hp_after_eat = eat_item_at_slot(
        state.inventory, safe_slot_idx, state.player_stats.hp, resolve_eat)
    post_eat_inv = jax.tree.map(
        lambda e, o: jnp.where(resolve_eat, e, o), eaten_inv, post_apply_inv)
    post_eat_hp = jnp.where(resolve_eat, hp_after_eat, state.player_stats.hp)

    # Build post-resolution state
    resolved_any = resolve_zap | resolve_apply | resolve_eat
    final_inv = post_eat_inv
    final_stats = post_zap_player_stats.replace(hp=jnp.where(resolve_eat, post_eat_hp, post_zap_player_stats.hp))
    final_levitating = jnp.where(resolve_apply, apply_levitating, state.player_levitating)
    final_lev_turns = jnp.where(resolve_apply, apply_lev_turns, state.levitation_turns)
    final_has_key = jnp.where(resolve_apply, apply_has_key, state.player_has_key)

    state = state.replace(
        monsters=post_zap_mons,
        map=post_zap_map,
        inventory=final_inv,
        player_stats=final_stats,
        player_levitating=final_levitating,
        levitation_turns=final_lev_turns,
        player_has_key=final_has_key,
    )

    # Override action: resolved phases should not trigger movement/pickup/etc.
    # Use SLOT_2 (19) as no-op sentinel — it doesn't match any Phase 1 handler
    action = jnp.where(resolved_any, jnp.int32(Action.SLOT_2), action)

    # ================================================================
    # Phase 1: Player Action
    # ================================================================
    new_timestep = state.timestep + 1

    is_move = action < 8
    is_pickup = action == Action.PICKUP
    is_kick = action == Action.KICK
    is_open = action == Action.OPEN_DOOR
    is_unlock = action == Action.UNLOCK_DOOR
    is_search = action == Action.SEARCH

    # --- Movement + bump attack ---
    delta = DIRECTION_VECTORS[action]
    target_pos = state.player_position + delta

    mon_pos_r = state.monsters.position[:, 0] == target_pos[0]
    mon_pos_c = state.monsters.position[:, 1] == target_pos[1]
    mon_at_target = mon_pos_r & mon_pos_c & state.monsters.mask
    has_monster_at_target = is_move & jnp.any(mon_at_target)

    attack_state = do_melee_attack(rng_bump, state, target_pos, static_params)

    moved_pos, can_move = move_player(
        state.player_position, action, state.map, map_h, map_w
    )

    move_pos = jnp.where(is_move & jnp.logical_not(has_monster_at_target),
                         moved_pos, state.player_position)
    move_state = state.replace(player_position=move_pos)

    phase1_state = jax.tree.map(
        lambda a, m: jnp.where(has_monster_at_target, a, m),
        attack_state, move_state,
    )

    # --- Auto-open closed doors on move ---
    target_r = jnp.clip(state.player_position[0] + delta[0], 0, map_h - 1)
    target_c = jnp.clip(state.player_position[1] + delta[1], 0, map_w - 1)
    target_in_bounds = in_bounds(state.player_position + delta, map_h, map_w)
    tile_at_target = state.map[target_r, target_c]
    is_diagonal = (delta[0] != 0) & (delta[1] != 0)
    is_closed_door = (tile_at_target == TileType.DOOR_CLOSED) & is_move & target_in_bounds & jnp.logical_not(is_diagonal)
    map_after_autodoor = state.map.at[target_r, target_c].set(
        jnp.where(is_closed_door, TileType.DOOR_OPEN, state.map[target_r, target_c])
    )

    auto_door_pos = jnp.where(
        is_move & is_closed_door & jnp.logical_not(has_monster_at_target),
        jnp.array([target_r, target_c]),
        phase1_state.player_position,
    )
    auto_valid = in_bounds(auto_door_pos, map_h, map_w)
    final_move_pos = jnp.where(auto_valid, auto_door_pos, phase1_state.player_position)

    # --- Pickup ---
    new_ground_pickup, new_inv_pickup, picked_up = pickup_item(
        state.ground_items, final_move_pos, state.inventory
    )

    # --- Kick ---
    map_after_kick, kick_success, rng_kick = kick_door(
        rng_kick, map_after_autodoor, final_move_pos, map_h, map_w, player_strength=state.player_stats.strength
    )

    # --- Open door ---
    map_after_open, opened = open_door_adjacent(
        map_after_autodoor, final_move_pos, map_h, map_w
    )

    # --- Unlock door ---
    map_after_unlock, unlocked = unlock_door_adjacent(
        map_after_autodoor, final_move_pos, state.player_has_key, map_h, map_w
    )

    # ================================================================
    # Combine action results
    # ================================================================
    new_pos = final_move_pos

    new_map = jnp.where(is_kick, map_after_kick,
              jnp.where(is_open, map_after_open,
              jnp.where(is_unlock, map_after_unlock,
                        map_after_autodoor)))

    new_monsters = phase1_state.monsters

    new_inv = jax.tree.map(
        lambda p, o: jnp.where(is_pickup, p, o),
        new_inv_pickup, state.inventory,
    )

    new_ground = jax.tree.map(
        lambda p, o: jnp.where(is_pickup, p, o),
        new_ground_pickup, state.ground_items,
    )

    new_stats = jax.tree.map(
        lambda a, o: jnp.where(has_monster_at_target, a, o),
        phase1_state.player_stats, state.player_stats,
    )

    new_hp = jnp.where(has_monster_at_target, new_stats.hp, state.player_stats.hp)

    new_levitating = state.player_levitating
    new_lev_turns = state.levitation_turns
    new_has_key = state.player_has_key

    # ================================================================
    # Phase 2: Trap check
    # ================================================================
    trap_hp_delta, new_traps, noise = check_traps(new_pos, state.traps, rng_traps)
    hp_after_traps = new_hp + trap_hp_delta

    mon_dist_r = jnp.abs(state.monsters.position[:, 0] - new_pos[0])
    mon_dist_c = jnp.abs(state.monsters.position[:, 1] - new_pos[1])
    mon_dist = jnp.maximum(mon_dist_r, mon_dist_c)
    wake_mask = noise & state.monsters.mask & (mon_dist <= 5)
    new_sleeping = jnp.where(wake_mask, False, state.monsters.is_sleeping)
    new_monsters = new_monsters.replace(is_sleeping=new_sleeping)

    searched_traps = search_traps(rng_search, new_pos, new_traps)
    new_traps = jax.tree.map(
        lambda s, o: jnp.where(is_search, s, o),
        searched_traps, new_traps,
    )

    # ================================================================
    # Phase 3: Terrain damage
    # ================================================================
    hp_after_terrain, rng_terrain = apply_terrain_damage(
        new_pos, hp_after_traps, new_map, new_levitating, state.player_stats.intrinsics, rng_terrain
    )

    # ================================================================
    # Phase 4: Levitation tick
    # ================================================================
    new_lev_turns_tick = jnp.where(new_levitating, new_lev_turns - 1, new_lev_turns)
    lev_expired = new_levitating & (new_lev_turns_tick <= 0)
    final_levitating = new_levitating & jnp.logical_not(lev_expired)
    final_lev_turns = jnp.where(lev_expired, 0, new_lev_turns_tick)

    mid_stats = new_stats.replace(hp=hp_after_terrain)

    mid_state = state.replace(
        map=new_map,
        player_position=new_pos,
        player_stats=mid_stats,
        player_levitating=final_levitating,
        levitation_turns=final_lev_turns,
        player_has_key=new_has_key,
        item_phase=jnp.int32(0),
        pending_item_slot=state.pending_item_slot,
        inventory=new_inv,
        monsters=new_monsters,
        traps=new_traps,
        ground_items=new_ground,
        timestep=new_timestep,
        terminal=state.terminal,
        state_rng=rng_next,
    )

    # ================================================================
    # Phase 5: Full monster AI
    # ================================================================
    _zero2 = jnp.zeros(2, dtype=jnp.int32)
    mid_state = full_monster_ai(
        rng_monsters, mid_state, static_params,
        _zero2, _zero2, _zero2, static_params.has_temple,
    )

    # ================================================================
    # Phase 6: Monster attacks
    # ================================================================
    mid_state = do_monster_attacks(rng_attacks, mid_state, static_params)

    # ================================================================
    # Phase 7: Goal check + reward
    # ================================================================
    on_stair = check_stair_goal(new_pos, state.downstair_position)
    go_down = (action == 10)
    stair_won = jnp.where(params.auto_descend, on_stair, on_stair & go_down)
    target_dead = ~mid_state.monsters.mask[static_params.goal_monster_idx]
    won = jnp.where(static_params.goal_type == 0, stair_won, target_dead)
    dead = mid_state.player_stats.hp <= 0
    timeout = new_timestep >= params.max_timesteps
    terminal = won | dead | timeout

    trap_triggered = trap_hp_delta < 0
    memento_trap = (static_params.goal_type == 1) & trap_triggered

    goal_reward = jnp.where(won, 1.0, 0.0)
    trap_penalty = jnp.where(memento_trap, -1.0, 0.0)
    reward = goal_reward + trap_penalty

    non_trap_death = dead & ~memento_trap
    reward = jnp.where(non_trap_death | timeout, 0.0, reward)

    visible_map = compute_visible(new_pos, new_map, map_h, map_w, state.lit_map)
    new_seen_map = update_seen_map(state.seen_map, visible_map)

    normal_state = mid_state.replace(
        terminal=terminal,
        seen_map=new_seen_map,
        visible_map=visible_map,
    )

    # ================================================================
    # Phase 8: Select skip vs normal tick
    # ================================================================
    skip_state = orig_state.replace(
        item_phase=new_item_phase,
        pending_item_slot=new_pending_slot,
    )

    output_state = jax.tree.map(
        lambda s, n: jnp.where(skip_game_tick, s, n),
        skip_state, normal_state,
    )
    output_reward = jnp.where(skip_game_tick, 0.0, reward)

    return output_state, output_reward
