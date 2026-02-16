"""Game logic for Tier 2 Hazard environments.

Supports: movement, bump attacks, lava damage, items (pickup/use),
doors (kick), simple monsters (pursue/wander each turn), goal check.
"""
import jax
import jax.numpy as jnp
from jax import lax

from Nethax.minihax.constants import (
    Action, TileType, DIRECTION_VECTORS, MONSTER_STATS, MONSTER_FLAGS,
    MF_HOSTILE, ABON_STR, ABON_DEX, DBON_STR, MONK_MARTIAL_SIDES,
    MONK_MARTIAL_BONUS, MONSTER_XP_SCORE, RoleType,
)
from Nethax.minihax.primitives.movement import move_player, check_stair_goal
from Nethax.minihax.primitives.terrain import apply_terrain_damage
from Nethax.minihax.primitives.visibility import compute_visible, update_seen_map
from Nethax.minihax.primitives.items import pickup_item, use_first_item
from Nethax.minihax.primitives.doors import kick_door
from Nethax.minihax.primitives.leveling import check_multi_levelup
from Nethax.minihax.util.game_logic_utils import (
    is_solid, in_bounds, dist2, compute_monster_damage,
)

# All 8 direction deltas
_ALL_DELTAS = jnp.array([
    [-1, 0], [1, 0], [0, 1], [0, -1],
    [-1, 1], [-1, -1], [1, 1], [1, -1],
])


def _simple_bump_attack(rng, state, target_pos, max_m):
    """Melee attack against a monster at target_pos with full stat system.

    Uses player_stats for to-hit, damage, XP, and leveling.
    Works with SimpleMonsters (Tier 2 — no movement points).

    Args:
        rng: JAX PRNG key
        state: HazardState (reads .monsters and .player_stats)
        target_pos: jnp.ndarray [2] — position to attack
        max_m: int — max monsters

    Returns:
        new_monsters: SimpleMonsters with damaged/killed monster
        new_stats: PlayerStats (updated XP, level, score, kills on kill)
        hit_monster: bool — whether a monster was present and attacked
        new_rng: JAX PRNG key
    """
    monsters = state.monsters
    stats = state.player_stats

    pos_r = monsters.position[:, 0] == target_pos[0]
    pos_c = monsters.position[:, 1] == target_pos[1]
    at_dest = pos_r & pos_c & monsters.mask
    any_mon = jnp.any(at_dest)
    idx = jnp.argmax(at_dest)
    safe_idx = jnp.where(any_mon, idx, 0)

    rng, rng_dmg, rng_dmg2, rng_hit, rng_hit_ac, rng_lvl = jax.random.split(rng, 6)

    # To-hit: NetHack formula
    mon_type = monsters.type_id[safe_idx]
    mon_ac = MONSTER_STATS[mon_type, 2]
    ac_value = jnp.where(
        mon_ac >= 0, mon_ac,
        -jax.random.randint(rng_hit_ac, (), 1, jnp.maximum(-mon_ac, 1) + 1),
    )
    str_bonus = ABON_STR[jnp.clip(stats.strength, 0, 25)]
    dex_bonus = ABON_DEX[jnp.clip(stats.dexterity, 0, 25)]
    low_level_bonus = jnp.where(stats.xp_level < 3, 1, 0)
    to_hit = 1 + str_bonus + dex_bonus + low_level_bonus + ac_value + stats.xp_level
    hit_roll = jax.random.randint(rng_hit, (), 1, 21)
    hits = to_hit > hit_roll

    # Damage: role-dependent
    is_monk = (stats.role_id == RoleType.MONK)
    bracket = jnp.minimum((stats.xp_level - 1) // 4, 4)
    monk_sides = MONK_MARTIAL_SIDES[bracket]
    monk_damage = jax.random.randint(rng_dmg, (), 1, monk_sides + 1) + MONK_MARTIAL_BONUS
    non_monk_damage = jax.random.randint(rng_dmg2, (), 1, 5)
    base_damage = jnp.where(is_monk, monk_damage, non_monk_damage)
    str_dmg = DBON_STR[jnp.clip(stats.strength, 0, 25)]
    damage = base_damage + str_dmg
    damage = jnp.maximum(damage, 1)
    damage = jnp.where(any_mon & hits, damage, 0)

    # Apply damage
    new_hp = monsters.health[safe_idx] - damage
    killed = any_mon & (new_hp <= 0)

    new_health = monsters.health.at[safe_idx].set(
        jnp.where(any_mon, new_hp, monsters.health[safe_idx])
    )
    new_mask = monsters.mask.at[safe_idx].set(
        jnp.where(killed, False, monsters.mask[safe_idx])
    )
    new_monsters = monsters.replace(health=new_health, mask=new_mask)

    # XP on kill + multi-level-up
    xp_gain = jnp.where(killed, MONSTER_STATS[mon_type, 13], 0)
    score_gain = jnp.where(killed, MONSTER_XP_SCORE[mon_type], 0)
    new_stats = stats.replace(
        xp=stats.xp + xp_gain,
        score=stats.score + score_gain,
        monsters_killed=stats.monsters_killed + jnp.where(killed, 1, 0),
    )
    new_stats = check_multi_levelup(rng_lvl, new_stats)

    return new_monsters, new_stats, any_mon, rng


def _simple_monster_ai(rng, monsters, player_pos, game_map, max_m, map_h, map_w):
    """Simplified monster AI for Tier 2: every alive monster acts each turn.

    No movement points, no sleeping. Hostile monsters pursue if within
    squared distance 100, otherwise wander randomly. Monsters adjacent
    to the player do not move (they attack in the attack phase).

    Args:
        rng: JAX PRNG key
        monsters: SimpleMonsters struct
        player_pos: jnp.ndarray [2]
        game_map: jnp.ndarray [map_h, map_w]
        max_m: int
        map_h: int
        map_w: int

    Returns:
        new_monsters: SimpleMonsters with updated positions
    """
    rngs = jax.random.split(rng, max_m + 1)

    def monster_step(carry, i):
        mon_positions = carry
        mon_rng = rngs[i + 1]
        mon_rng_wander = mon_rng

        alive = monsters.mask[i]
        pos = mon_positions[i]
        mon_type = monsters.type_id[i]

        flags = MONSTER_FLAGS[mon_type]
        is_hostile = (flags & MF_HOSTILE) != 0

        # Distance to player
        d2 = dist2(pos, player_pos)
        aware = d2 <= 100

        # Adjacent: Chebyshev distance <= 1 and not same tile
        adj_dx = jnp.abs(pos[0] - player_pos[0])
        adj_dy = jnp.abs(pos[1] - player_pos[1])
        adjacent = (adj_dx <= 1) & (adj_dy <= 1) & (d2 > 0)

        # Evaluate all 8 candidate positions
        candidates = pos[None, :] + _ALL_DELTAS  # [8, 2]

        def valid_candidate(cand_pos):
            vb = in_bounds(cand_pos, map_h, map_w)
            sr = jnp.clip(cand_pos[0], 0, map_h - 1)
            sc = jnp.clip(cand_pos[1], 0, map_w - 1)
            ns = jnp.logical_not(is_solid(game_map[sr, sc]))
            # Don't move onto player
            np_ = jnp.logical_not(
                (cand_pos[0] == player_pos[0]) & (cand_pos[1] == player_pos[1])
            )
            # Monster collision
            pm = (mon_positions[:, 0] == cand_pos[0]) & (mon_positions[:, 1] == cand_pos[1])
            not_self = jnp.arange(max_m) != i
            no_mon = jnp.logical_not(jnp.any(pm & monsters.mask & not_self))
            return vb & ns & np_ & no_mon

        valid_mask = jax.vmap(valid_candidate)(candidates)  # [8]

        # Pursuit: pick valid candidate closest to player
        cand_dists = jax.vmap(lambda c: dist2(c, player_pos))(candidates)
        cand_dists_pursuit = jnp.where(valid_mask, cand_dists, jnp.int32(999999))
        best_pursuit_idx = jnp.argmin(cand_dists_pursuit)
        pursuit_delta = _ALL_DELTAS[best_pursuit_idx]

        # Wander: random direction
        wander_idx = jax.random.randint(mon_rng_wander, (), 0, 8)
        wander_delta = _ALL_DELTAS[wander_idx]

        # Decision
        can_act = alive & is_hostile
        should_pursue = can_act & aware & jnp.logical_not(adjacent)
        should_wander = can_act & jnp.logical_not(aware) & jnp.logical_not(adjacent)

        delta = jnp.where(should_pursue, pursuit_delta,
                jnp.where(should_wander, wander_delta,
                          jnp.array([0, 0])))

        new_pos = pos + delta

        # Final validation
        valid_bounds = in_bounds(new_pos, map_h, map_w)
        safe_r = jnp.clip(new_pos[0], 0, map_h - 1)
        safe_c = jnp.clip(new_pos[1], 0, map_w - 1)
        not_solid = jnp.logical_not(is_solid(game_map[safe_r, safe_c]))
        not_player = jnp.logical_not(
            (new_pos[0] == player_pos[0]) & (new_pos[1] == player_pos[1])
        )
        pos_match = (mon_positions[:, 0] == new_pos[0]) & (mon_positions[:, 1] == new_pos[1])
        not_self = jnp.arange(max_m) != i
        not_blocked = jnp.logical_not(jnp.any(pos_match & monsters.mask & not_self))

        valid_move = can_act & valid_bounds & not_solid & not_player & not_blocked
        final_pos = jnp.where(valid_move, new_pos, pos)
        new_positions = mon_positions.at[i].set(final_pos)

        return new_positions, None

    new_positions, _ = lax.scan(
        monster_step, monsters.position, jnp.arange(max_m)
    )

    return monsters.replace(position=new_positions)


def _simple_monster_attacks(rng, monsters, player_pos, player_hp, max_m):
    """Simplified monster attacks for Tier 2.

    All adjacent alive hostile monsters attack. Simplified to-hit:
    always hits. Damage: d(atk_dice, atk_sides), minimum 1.

    Args:
        rng: JAX PRNG key
        monsters: SimpleMonsters struct
        player_pos: jnp.ndarray [2]
        player_hp: int
        max_m: int

    Returns:
        new_hp: int — player HP after attacks
    """
    rngs = jax.random.split(rng, max_m + 1)

    def attack_step(hp, i):
        mon_rng = rngs[i + 1]
        alive = monsters.mask[i]
        mon_pos = monsters.position[i]

        # Adjacency
        dx = jnp.abs(mon_pos[0] - player_pos[0])
        dy = jnp.abs(mon_pos[1] - player_pos[1])
        adjacent = (dx <= 1) & (dy <= 1) & ((dx > 0) | (dy > 0))

        mon_type = monsters.type_id[i]
        flags = MONSTER_FLAGS[mon_type]
        is_hostile = (flags & MF_HOSTILE) != 0

        can_attack = alive & adjacent & is_hostile

        # Damage: use monster stats
        damage = compute_monster_damage(mon_rng, mon_type)
        damage = jnp.maximum(damage, 1)
        actual_damage = jnp.where(can_attack, damage, 0)
        new_hp = hp - actual_damage

        return new_hp, None

    new_hp, _ = lax.scan(attack_step, player_hp, jnp.arange(max_m))
    return new_hp


def hazard_step(rng, state, action, params, static_params):
    """Tier 2 step function: movement + lava + items + simple monsters + doors.

    Phases:
    1. Player action: move (with bump attack on monsters), pickup, use_item, kick
    2. Auto-open closed doors on move
    3. Terrain damage (lava)
    4. Simple monster AI
    5. Monster attacks
    6. Goal check

    Args:
        rng: JAX PRNG key
        state: HazardState
        action: int (0-13)
        params: EnvParams
        static_params: HazardStaticParams

    Returns:
        new_state: HazardState
        reward: float
    """
    map_h = static_params.map_height
    map_w = static_params.map_width
    max_m = static_params.max_monsters

    rng_action, rng_bump, rng_monsters, rng_attacks, rng_terrain, rng_kick, rng_next = \
        jax.random.split(rng, 7)

    new_timestep = state.timestep + 1

    is_move = action < 8
    is_downstair = action == Action.GO_DOWN_STAIRS
    is_pickup = action == Action.PICKUP
    is_use = action == Action.USE_ITEM
    is_kick = action == Action.KICK

    # ================================================================
    # Phase 1: Player Action
    # ================================================================

    # --- Movement + bump attack ---
    # First check where we'd move to
    delta = DIRECTION_VECTORS[action]
    target_pos = state.player_position + delta

    # Check for monster at target position (for bump attack)
    mon_pos_r = state.monsters.position[:, 0] == target_pos[0]
    mon_pos_c = state.monsters.position[:, 1] == target_pos[1]
    mon_at_target = mon_pos_r & mon_pos_c & state.monsters.mask
    has_monster_at_target = is_move & jnp.any(mon_at_target)

    # Do bump attack if monster at target (now returns updated stats too)
    bumped_monsters, bumped_stats, did_bump, rng_bump = _simple_bump_attack(
        rng_bump, state, target_pos, max_m
    )

    # Movement (only if no monster at target)
    moved_pos, can_move = move_player(
        state.player_position, action, state.map, map_h, map_w
    )

    # If bump attack: don't move, just attack
    # If no monster: move normally
    move_pos = jnp.where(is_move & jnp.logical_not(has_monster_at_target),
                         moved_pos, state.player_position)
    move_monsters = jax.tree.map(
        lambda b, o: jnp.where(has_monster_at_target, b, o),
        bumped_monsters, state.monsters,
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
        move_pos,
    )
    auto_valid = in_bounds(auto_door_pos, map_h, map_w)
    final_move_pos = jnp.where(auto_valid, auto_door_pos, move_pos)

    # --- Pickup ---
    new_ground_pickup, new_inv_pickup, picked_up = pickup_item(
        state.ground_items, final_move_pos, state.inventory
    )

    # --- Use item ---
    new_inv_use, map_after_use, hp_after_use, got_levitation, got_key = use_first_item(
        state.inventory, map_after_autodoor, final_move_pos,
        state.player_stats.hp, map_h, map_w,
    )

    # --- Kick ---
    map_after_kick, kick_success, rng_kick = kick_door(
        rng_kick, map_after_autodoor, final_move_pos, map_h, map_w, player_strength=state.player_stats.strength
    )

    # ================================================================
    # Combine action results
    # ================================================================

    # Position: always use final_move_pos (only changes on movement action)
    new_pos = final_move_pos

    # Map: depends on action taken
    new_map = jnp.where(is_use, map_after_use,
              jnp.where(is_kick, map_after_kick,
                        map_after_autodoor))

    # Monsters: update only on move (bump attack)
    new_monsters = jax.tree.map(
        lambda m, o: jnp.where(is_move, m, o),
        move_monsters, state.monsters,
    )

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

    # HP: use_item can heal (apple), bump attack can level-up (changing HP)
    # After bump: bumped_stats.hp may differ from state.player_stats.hp due to level-up
    base_hp = jnp.where(has_monster_at_target, bumped_stats.hp, state.player_stats.hp)
    new_hp = jnp.where(is_use, hp_after_use, base_hp)

    # Levitation: from use_item
    new_levitating = state.player_levitating | (is_use & got_levitation)
    new_lev_turns = jnp.where(
        is_use & got_levitation,
        state.levitation_turns + 100,  # 100 turns of levitation
        state.levitation_turns,
    )

    # ================================================================
    # Phase 2: Terrain damage
    # ================================================================
    hp_after_terrain, rng_terrain = apply_terrain_damage(
        new_pos, new_hp, new_map, new_levitating, state.player_stats.intrinsics, rng_terrain
    )

    # ================================================================
    # Phase 3: Levitation tick
    # ================================================================
    new_lev_turns_tick = jnp.where(new_levitating, new_lev_turns - 1, new_lev_turns)
    lev_expired = new_levitating & (new_lev_turns_tick <= 0)
    final_levitating = new_levitating & jnp.logical_not(lev_expired)
    final_lev_turns = jnp.where(lev_expired, 0, new_lev_turns_tick)

    # ================================================================
    # Phase 4: Simple monster AI
    # ================================================================
    new_monsters_ai = _simple_monster_ai(
        rng_monsters, new_monsters, new_pos, new_map,
        max_m, map_h, map_w,
    )

    # ================================================================
    # Phase 5: Monster attacks
    # ================================================================
    hp_after_attacks = _simple_monster_attacks(
        rng_attacks, new_monsters_ai, new_pos, hp_after_terrain, max_m
    )

    # ================================================================
    # Phase 6: Goal check
    # ================================================================
    on_stair = check_stair_goal(new_pos, state.downstair_position)
    go_down = (action == 10)  # Action.GO_DOWN_STAIRS
    won = jnp.where(params.auto_descend, on_stair, on_stair & go_down)
    dead = hp_after_attacks <= 0
    timeout = new_timestep >= params.max_timesteps
    terminal = won | dead | timeout

    reward = jnp.where(won, 1.0, 0.0)

    # Visibility update
    visible_map = compute_visible(new_pos, new_map, map_h, map_w, state.lit_map)
    new_seen_map = update_seen_map(state.seen_map, visible_map)

    # Build final player_stats: conditionally select bumped_stats if bump happened
    final_stats = jax.tree.map(
        lambda b, o: jnp.where(has_monster_at_target, b, o),
        bumped_stats, state.player_stats,
    )
    # Set final HP (after terrain damage, monster attacks, item use)
    final_stats = final_stats.replace(hp=hp_after_attacks)

    new_state = state.replace(
        map=new_map,
        player_position=new_pos,
        player_stats=final_stats,
        player_levitating=final_levitating,
        levitation_turns=final_lev_turns,
        inventory=new_inv,
        monsters=new_monsters_ai,
        ground_items=new_ground,
        seen_map=new_seen_map,
        visible_map=visible_map,
        timestep=new_timestep,
        terminal=terminal,
        state_rng=rng_next,
    )

    return new_state, reward
