"""Monster AI primitives shared across Tier 2 and 3."""
import jax
import jax.numpy as jnp
from jax import lax

from Nethax.minihax.constants import (
    MONSTER_STATS, MONSTER_FLAGS, MF_HOSTILE, MF_PEACEFUL,
    NORMAL_SPEED,
)
from Nethax.minihax.util.game_logic_utils import (
    is_solid, in_bounds, dist2, compute_monster_damage,
)

# All 8 direction deltas (matching NetHack)
_ALL_DELTAS = jnp.array([
    [-1, 0], [1, 0], [0, 1], [0, -1],    # N, S, E, W
    [-1, 1], [-1, -1], [1, 1], [1, -1],   # NE, NW, SE, SW
])


def full_monster_ai(rng, state, static_params,
                    altar_position, temple_min, temple_max, has_temple):
    """Full Tier 3 monster AI with movement points, sleeping, pursuit, temple sanctuary.

    Generalized from zombie_horde.py::update_monsters.
    Temple logic is conditional on the has_temple flag.
    When has_temple is True, uses altar_position, temple_min, temple_max
    for priest milling and hostile temple avoidance.

    Args:
        rng: JAX PRNG key
        state: game state with monsters, player_position, map
        static_params: static env params (max_monsters, map_height, map_width)
        altar_position: jnp array [2] — priest mills around this
        temple_min: jnp array [2] — temple region min (row, col) inclusive
        temple_max: jnp array [2] — temple region max (row, col) inclusive
        has_temple: bool — whether temple sanctuary logic applies
    """
    max_m = static_params.max_monsters
    player_pos = state.player_position

    # Check if any peaceful monster (priest) is alive — for temple sanctuary
    peaceful_flags = MONSTER_FLAGS[state.monsters.type_id] & MF_PEACEFUL
    priest_alive = jnp.where(has_temple,
        jnp.any(state.monsters.mask & (peaceful_flags != 0)),
        False)

    # --- Phase 1: mcalcmove — ALL monsters accumulate movement points ---
    # In NetHack, sleeping monsters also accumulate (they spend and waste in dochug)
    rng, rng_speed = jax.random.split(rng)
    speed_rngs = jax.random.split(rng_speed, max_m)

    mmoves = MONSTER_STATS[state.monsters.type_id, 1]  # [max_m]
    mmove_adj = mmoves % NORMAL_SPEED
    base_move = mmoves - mmove_adj
    speed_rolls = jax.vmap(
        lambda k: jax.random.randint(k, (), 0, NORMAL_SPEED)
    )(speed_rngs)
    gets_bonus = speed_rolls < mmove_adj
    points_gained = base_move + jnp.where(gets_bonus, NORMAL_SPEED, 0)
    # ALL monsters gain points (no sleeping filter — matches NetHack)
    new_movement = state.monsters.movement_points + points_gained

    # --- Phase 2: disturb — wake sleeping monsters ---
    # dist2 <= 100 (squared Euclidean), 1/7 chance per turn
    rng, rng_wake = jax.random.split(rng)
    wake_rngs = jax.random.split(rng_wake, max_m)
    wake_rolls = jax.vmap(
        lambda k: jax.random.randint(k, (), 0, 7)
    )(wake_rngs)

    mon_positions_init = state.monsters.position
    dists = jax.vmap(
        lambda pos: dist2(pos, player_pos)
    )(mon_positions_init)
    within_range = dists <= 100  # dist2 <= 100 = Euclidean distance <= 10
    wake_check = (wake_rolls == 0) & within_range & state.monsters.is_sleeping & state.monsters.mask
    new_sleeping = state.monsters.is_sleeping & jnp.logical_not(wake_check)

    # --- Phase 3: Movement (via lax.scan) ---
    rngs = jax.random.split(rng, max_m + 1)

    def monster_step(carry, i):
        mon_positions, map_data, mov_points, sleeping = carry
        mon_rng = rngs[i + 1]
        mon_rng_wander, mon_rng_mill_r, mon_rng_mill_c = jax.random.split(mon_rng, 3)

        alive = state.monsters.mask[i]
        pos = mon_positions[i]
        mon_type = state.monsters.type_id[i]
        cur_sleeping = sleeping[i]
        cur_move_pts = mov_points[i]

        # Has enough movement points to act?
        has_points = alive & (cur_move_pts >= NORMAL_SPEED)
        # Can actually move? Need points AND awake
        can_act = has_points & jnp.logical_not(cur_sleeping)

        # Distance to player (squared Euclidean, NetHack dist2)
        d2 = dist2(pos, player_pos)

        flags = MONSTER_FLAGS[mon_type]
        is_hostile = (flags & MF_HOSTILE) != 0
        is_peaceful = (flags & MF_PEACEFUL) != 0
        aware = d2 <= 100  # within ~10 squares

        # Adjacent: Chebyshev distance <= 1 and not same tile
        adj_dx = jnp.abs(pos[0] - player_pos[0])
        adj_dy = jnp.abs(pos[1] - player_pos[1])
        adjacent = (adj_dx <= 1) & (adj_dy <= 1) & (d2 > 0)

        # --- Evaluate all 8 candidate positions ---
        candidates = pos[None, :] + _ALL_DELTAS  # [8, 2]

        def valid_candidate(cand_pos):
            vb = in_bounds(cand_pos, static_params.map_height, static_params.map_width)
            sr = jnp.clip(cand_pos[0], 0, static_params.map_height - 1)
            sc = jnp.clip(cand_pos[1], 0, static_params.map_width - 1)
            ns = jnp.logical_not(is_solid(map_data[sr, sc]))
            np_ = jnp.logical_not((cand_pos[0] == player_pos[0]) & (cand_pos[1] == player_pos[1]))
            # Monster collision
            pm = (mon_positions[:, 0] == cand_pos[0]) & (mon_positions[:, 1] == cand_pos[1])
            not_self = jnp.arange(max_m) != i
            no_mon = jnp.logical_not(jnp.any(pm & state.monsters.mask & not_self))
            # Temple avoidance for hostile
            in_t = ((cand_pos[0] >= temple_min[0]) & (cand_pos[0] <= temple_max[0]) &
                     (cand_pos[1] >= temple_min[1]) & (cand_pos[1] <= temple_max[1]))
            no_temple = jnp.logical_not(is_hostile & in_t & priest_alive)
            return vb & ns & np_ & no_mon & no_temple

        valid_mask = jax.vmap(valid_candidate)(candidates)  # [8]

        # Pursuit: pick valid candidate with minimum dist2 to player (greedy)
        cand_dists = jax.vmap(lambda c: dist2(c, player_pos))(candidates)
        cand_dists_pursuit = jnp.where(valid_mask, cand_dists, jnp.int32(999999))
        best_pursuit_idx = jnp.argmin(cand_dists_pursuit)
        pursuit_delta = _ALL_DELTAS[best_pursuit_idx]

        # Wander: random from all 8 directions
        wander_idx = jax.random.randint(mon_rng_wander, (), 0, 8)
        wander_delta = _ALL_DELTAS[wander_idx]

        # Priest: mill around altar (pri_move from priest.c)
        mill_offset_r = jax.random.randint(mon_rng_mill_r, (), -1, 2)
        mill_offset_c = jax.random.randint(mon_rng_mill_c, (), -1, 2)
        mill_target = altar_position + jnp.array([mill_offset_r, mill_offset_c])
        cand_dists_mill = jax.vmap(lambda c: dist2(c, mill_target))(candidates)
        cand_dists_mill = jnp.where(valid_mask, cand_dists_mill, jnp.int32(999999))
        best_mill_idx = jnp.argmin(cand_dists_mill)
        mill_delta = _ALL_DELTAS[best_mill_idx]

        # Decision: which behavior?
        should_pursue = can_act & is_hostile & aware & jnp.logical_not(adjacent)
        should_wander = can_act & is_hostile & jnp.logical_not(aware) & jnp.logical_not(adjacent)
        should_mill = can_act & is_peaceful

        delta = jnp.where(should_pursue, pursuit_delta,
                jnp.where(should_mill, mill_delta,
                jnp.where(should_wander, wander_delta,
                          jnp.array([0, 0]))))

        new_pos = pos + delta

        # Final validation (catches invalid wander directions)
        valid_bounds = in_bounds(new_pos, static_params.map_height, static_params.map_width)
        safe_r = jnp.clip(new_pos[0], 0, static_params.map_height - 1)
        safe_c = jnp.clip(new_pos[1], 0, static_params.map_width - 1)
        tile_at_new = map_data[safe_r, safe_c]
        not_solid = jnp.logical_not(is_solid(tile_at_new))
        not_player = jnp.logical_not(
            (new_pos[0] == player_pos[0]) & (new_pos[1] == player_pos[1])
        )

        # Collision avoidance
        pos_match = (mon_positions[:, 0] == new_pos[0]) & (mon_positions[:, 1] == new_pos[1])
        not_self = jnp.arange(max_m) != i
        blocked_by_mon = jnp.any(pos_match & state.monsters.mask & not_self)
        not_blocked = jnp.logical_not(blocked_by_mon)

        # Temple avoidance
        in_temple = ((new_pos[0] >= temple_min[0]) & (new_pos[0] <= temple_max[0]) &
                     (new_pos[1] >= temple_min[1]) & (new_pos[1] <= temple_max[1]))
        avoids_temple = jnp.logical_not(is_hostile & in_temple & priest_alive)

        valid_move = can_act & valid_bounds & not_solid & not_player & not_blocked & avoids_temple

        final_pos = jnp.where(valid_move, new_pos, pos)
        new_positions = mon_positions.at[i].set(final_pos)

        # Spend movement points: ALL monsters with enough points spend them
        # (sleeping monsters spend but waste, matching NetHack dochug behavior)
        new_pts = jnp.where(has_points, cur_move_pts - NORMAL_SPEED, cur_move_pts)
        new_mov_points = mov_points.at[i].set(new_pts)

        return (new_positions, map_data, new_mov_points, sleeping), None

    (new_positions, _, spent_movement, _), _ = lax.scan(
        monster_step,
        (state.monsters.position, state.map, new_movement, new_sleeping),
        jnp.arange(max_m),
    )

    monsters = state.monsters.replace(
        position=new_positions,
        movement_points=spent_movement,
        is_sleeping=new_sleeping,
    )
    state = state.replace(monsters=monsters)
    return state


def do_monster_attacks(rng, state, static_params):
    """All adjacent hostile monsters attack the player (NetHack mhitu.c).

    To-hit: AC_VALUE(AC) + 10 + monster_level > rnd(20)
      where AC_VALUE(AC>=0) = AC, AC_VALUE(AC<0) = -rnd(-AC)
    Damage: d(atk_dice, atk_sides) - rnd(-AC) if AC < 0, minimum 1
    Only awake, alive, hostile, adjacent monsters can attack.
    """
    max_m = static_params.max_monsters
    player_ac = state.player_stats.ac

    rngs = jax.random.split(rng, max_m + 1)

    def attack_step(carry, i):
        hp = carry
        mon_rng = rngs[i + 1]
        mon_rng_hit, mon_rng_dmg, mon_rng_ac, mon_rng_acval = jax.random.split(mon_rng, 4)

        alive = state.monsters.mask[i]
        not_sleeping = jnp.logical_not(state.monsters.is_sleeping[i])
        mon_pos = state.monsters.position[i]

        # Adjacency: Chebyshev distance <= 1 and not same position
        dx = jnp.abs(mon_pos[0] - state.player_position[0])
        dy = jnp.abs(mon_pos[1] - state.player_position[1])
        adjacent = (dx <= 1) & (dy <= 1) & ((dx > 0) | (dy > 0))

        mon_type = state.monsters.type_id[i]
        flags = MONSTER_FLAGS[mon_type]
        is_hostile = (flags & MF_HOSTILE) != 0

        can_attack = alive & adjacent & is_hostile & not_sleeping

        # To-hit: AC_VALUE(AC) + 10 + mlev > rnd(20)  (NetHack mhitu.c:707-808)
        # AC_VALUE: AC if AC>=0, else -rnd(-AC)  (hack.h:1543)
        mon_level = MONSTER_STATS[mon_type, 0]
        ac_value = jnp.where(
            player_ac >= 0,
            player_ac,
            -jax.random.randint(mon_rng_acval, (), 1, jnp.maximum(-player_ac, 1) + 1),
        )
        tmp = ac_value + 10 + mon_level
        roll = jax.random.randint(mon_rng_hit, (), 1, 21)  # rnd(20) = 1..20
        hits = tmp > roll

        # Base damage
        base_damage = compute_monster_damage(mon_rng_dmg, mon_type)

        # Negative AC damage reduction (NetHack)
        ac_reduction = jnp.where(
            player_ac < 0,
            jax.random.randint(mon_rng_ac, (), 1, jnp.maximum(-player_ac, 1) + 1),
            0
        )
        damage = jnp.maximum(base_damage - ac_reduction, 1)

        actual_damage = jnp.where(can_attack & hits, damage, 0)

        new_hp = hp - actual_damage
        return new_hp, None

    new_hp, _ = lax.scan(attack_step, state.player_stats.hp, jnp.arange(max_m))
    state = state.replace(player_stats=state.player_stats.replace(hp=new_hp))
    return state
