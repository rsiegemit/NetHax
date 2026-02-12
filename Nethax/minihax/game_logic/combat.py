"""Game logic for Tier 3 Combat environments.

Supports: movement with bump-attack (do_melee_attack), trap checking,
item pickup/use (levitation, freeze lava, key, apple), door operations
(open, kick, unlock), terrain damage (lava), levitation tick,
full monster AI (movement points, sleeping, pursuit, temple),
monster attacks with AC system (do_monster_attacks), goal check.
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    Action, TileType, DIRECTION_VECTORS,
)
from Nethax.minihax.primitives.combat import do_melee_attack
from Nethax.minihax.primitives.monster_ai import full_monster_ai, do_monster_attacks
from Nethax.minihax.primitives.visibility import compute_visible, update_seen_map
from Nethax.minihax.primitives.movement import move_player, check_stair_goal
from Nethax.minihax.primitives.terrain import apply_terrain_damage
from Nethax.minihax.primitives.items import pickup_item, use_first_item
from Nethax.minihax.primitives.doors import kick_door, open_door_adjacent, unlock_door_adjacent
from Nethax.minihax.primitives.traps import check_traps
from Nethax.minihax.util.game_logic_utils import in_bounds


def combat_step(rng, state, action, params, static_params):
    """Tier 3 combat step function.

    Phases:
    1. Player action: move (with bump attack via do_melee_attack), pickup,
       use_item, kick, open_door, unlock_door
    2. Auto-open closed doors on move
    3. Trap check at new position
    4. Terrain damage (lava)
    5. Levitation tick
    6. Full monster AI (via full_monster_ai from primitives)
    7. Monster attacks (via do_monster_attacks from primitives)
    8. Goal check + reward

    Args:
        rng: JAX PRNG key
        state: CombatState
        action: int (0-15)
        params: EnvParams
        static_params: CombatStaticParams

    Returns:
        new_state: CombatState
        reward: float
    """
    map_h = static_params.map_height
    map_w = static_params.map_width

    rng_action, rng_bump, rng_monsters, rng_attacks, rng_terrain, rng_kick, rng_next = \
        jax.random.split(rng, 7)

    new_timestep = state.timestep + 1
    old_score = state.score

    is_move = action < 8
    is_downstair = action == Action.GO_DOWN_STAIRS
    is_pickup = action == Action.PICKUP
    is_use = action == Action.USE_ITEM
    is_kick = action == Action.KICK
    is_open = action == Action.OPEN_DOOR
    is_unlock = action == Action.UNLOCK_DOOR

    # ================================================================
    # Phase 1: Player Action
    # ================================================================

    # --- Movement + bump attack ---
    delta = DIRECTION_VECTORS[action]
    target_pos = state.player_position + delta

    # Check for monster at target position (for bump attack)
    mon_pos_r = state.monsters.position[:, 0] == target_pos[0]
    mon_pos_c = state.monsters.position[:, 1] == target_pos[1]
    mon_at_target = mon_pos_r & mon_pos_c & state.monsters.mask
    has_monster_at_target = is_move & jnp.any(mon_at_target)

    # Do bump attack if monster at target (uses full do_melee_attack with XP/score)
    attack_state = do_melee_attack(rng_bump, state, target_pos, static_params)

    # Movement (only if no monster at target)
    moved_pos, can_move = move_player(
        state.player_position, action, state.map, map_h, map_w
    )

    # If bump attack: don't move, just attack (take attack_state)
    # If no monster: move normally (keep original state + new position)
    move_pos = jnp.where(is_move & jnp.logical_not(has_monster_at_target),
                         moved_pos, state.player_position)
    move_state = state.replace(player_position=move_pos)

    # Select: attack state if monster at target, else move state
    phase1_state = jax.tree.map(
        lambda a, m: jnp.where(has_monster_at_target, a, m),
        attack_state, move_state,
    )

    # --- Auto-open closed doors on move ---
    safe_mr = jnp.clip(phase1_state.player_position[0], 0, map_h - 1)
    safe_mc = jnp.clip(phase1_state.player_position[1], 0, map_w - 1)
    tile_at_dest = state.map[safe_mr, safe_mc]
    is_closed_door = (tile_at_dest == TileType.DOOR_CLOSED) & is_move
    map_after_autodoor = state.map.at[safe_mr, safe_mc].set(
        jnp.where(is_closed_door, TileType.DOOR_OPEN, state.map[safe_mr, safe_mc])
    )

    # If door was closed, move_player would have rejected (DOOR_CLOSED is solid).
    # Override position if we auto-opened the door.
    auto_door_pos = jnp.where(
        is_move & is_closed_door & jnp.logical_not(has_monster_at_target),
        state.player_position + delta,
        phase1_state.player_position,
    )
    auto_valid = in_bounds(auto_door_pos, map_h, map_w)
    final_move_pos = jnp.where(auto_valid, auto_door_pos, phase1_state.player_position)

    # --- Pickup ---
    new_ground_pickup, new_inv_pickup, picked_up = pickup_item(
        state.ground_items, final_move_pos, state.inventory
    )

    # --- Use item ---
    new_inv_use, map_after_use, hp_after_use, got_levitation, got_key = use_first_item(
        state.inventory, map_after_autodoor, final_move_pos,
        state.player_hp, map_h, map_w,
    )

    # --- Kick ---
    map_after_kick, kick_success, rng_kick = kick_door(
        rng_kick, map_after_autodoor, final_move_pos, map_h, map_w
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

    # Position
    new_pos = final_move_pos

    # Map: depends on action taken
    new_map = jnp.where(is_use, map_after_use,
              jnp.where(is_kick, map_after_kick,
              jnp.where(is_open, map_after_open,
              jnp.where(is_unlock, map_after_unlock,
                        map_after_autodoor))))

    # Monsters: from phase1_state (may have been attacked)
    new_monsters = phase1_state.monsters

    # Inventory: depends on pickup vs use_item
    new_inv = jax.tree.map(
        lambda p, u, o: jnp.where(is_pickup, p,
                        jnp.where(is_use, u, o)),
        new_inv_pickup, new_inv_use, state.inventory,
    )

    # Ground items: only change on pickup
    new_ground = jax.tree.map(
        lambda p, o: jnp.where(is_pickup, p, o),
        new_ground_pickup, state.ground_items,
    )

    # HP: from attack_state if bump, or use_item if USE, else original
    new_hp = jnp.where(has_monster_at_target, phase1_state.player_hp,
             jnp.where(is_use, hp_after_use, state.player_hp))

    # XP, level, score, kills: from attack state if bump, else original
    new_xp = jnp.where(has_monster_at_target, phase1_state.player_xp, state.player_xp)
    new_xp_level = jnp.where(has_monster_at_target, phase1_state.player_xp_level, state.player_xp_level)
    new_score = jnp.where(has_monster_at_target, phase1_state.score, state.score)
    new_max_hp = jnp.where(has_monster_at_target, phase1_state.player_max_hp, state.player_max_hp)
    new_kills = jnp.where(has_monster_at_target, phase1_state.monsters_killed, state.monsters_killed)

    # Levitation: from use_item
    new_levitating = state.player_levitating | (is_use & got_levitation)
    new_lev_turns = jnp.where(
        is_use & got_levitation,
        state.levitation_turns + 100,  # 100 turns of levitation
        state.levitation_turns,
    )

    # Key: from use_item
    new_has_key = state.player_has_key | (is_use & got_key)

    # ================================================================
    # Phase 2: Trap check
    # ================================================================
    trap_hp_delta, new_traps = check_traps(new_pos, state.traps)
    hp_after_traps = new_hp + trap_hp_delta

    # ================================================================
    # Phase 3: Terrain damage
    # ================================================================
    hp_after_terrain, rng_terrain = apply_terrain_damage(
        new_pos, hp_after_traps, new_map, new_levitating, rng_terrain
    )

    # ================================================================
    # Phase 4: Levitation tick
    # ================================================================
    new_lev_turns_tick = jnp.where(new_levitating, new_lev_turns - 1, new_lev_turns)
    lev_expired = new_levitating & (new_lev_turns_tick <= 0)
    final_levitating = new_levitating & jnp.logical_not(lev_expired)
    final_lev_turns = jnp.where(lev_expired, 0, new_lev_turns_tick)

    # Build intermediate state for monster AI
    mid_state = state.replace(
        map=new_map,
        player_position=new_pos,
        player_hp=hp_after_terrain,
        player_max_hp=new_max_hp,
        player_xp=new_xp,
        player_xp_level=new_xp_level,
        player_ac=state.player_ac,
        player_strength=state.player_strength,
        player_levitating=final_levitating,
        levitation_turns=final_lev_turns,
        player_has_key=new_has_key,
        inventory=new_inv,
        monsters=new_monsters,
        traps=new_traps,
        ground_items=new_ground,
        score=new_score,
        monsters_killed=new_kills,
        timestep=new_timestep,
        terminal=state.terminal,
        state_rng=rng_next,
    )

    # ================================================================
    # Phase 5: Full monster AI
    # ================================================================
    # Temple params: zeros for non-temple envs
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
    # Goal type 0: reach downstair
    on_stair = check_stair_goal(new_pos, state.downstair_position)
    stair_won = on_stair & is_downstair
    # Goal type 1: kill target monster (e.g., grid bug in Memento)
    target_dead = ~mid_state.monsters.mask[static_params.goal_monster_idx]
    won = jnp.where(static_params.goal_type == 0, stair_won, target_dead)
    dead = mid_state.player_hp <= 0
    timeout = new_timestep >= params.max_timesteps
    terminal = won | dead | timeout

    # Detect board trap trigger for Memento envs (goal_type==1)
    trap_triggered = trap_hp_delta < 0
    memento_trap = (static_params.goal_type == 1) & trap_triggered

    # Reward: +1 on goal, -1 on trap (Memento only), 0 otherwise
    goal_reward = jnp.where(won, 1.0, 0.0)
    trap_penalty = jnp.where(memento_trap, -1.0, 0.0)
    reward = goal_reward + trap_penalty

    # Zero reward on non-trap death or timeout
    non_trap_death = dead & ~memento_trap
    reward = jnp.where(non_trap_death | timeout, 0.0, reward)

    # Visibility update
    visible_map = compute_visible(new_pos, new_map, map_h, map_w)
    new_seen_map = update_seen_map(state.seen_map, visible_map)

    final_state = mid_state.replace(
        terminal=terminal,
        seen_map=new_seen_map,
        visible_map=visible_map,
    )

    return final_state, reward
