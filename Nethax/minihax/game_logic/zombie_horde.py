"""Game logic for minihax ZombieHorde — matching NetHack 3.7 mechanics."""
import jax
import jax.numpy as jnp
from jax import lax

from Nethax.minihax.constants import (
    Action, DIRECTION_VECTORS,
    ALTAR_POSITION, TEMPLE_MIN, TEMPLE_MAX,
)
from Nethax.minihax.primitives.combat import do_melee_attack
from Nethax.minihax.primitives.monster_ai import full_monster_ai, do_monster_attacks
from Nethax.minihax.primitives.visibility import compute_visible, update_seen_map
from Nethax.minihax.util.game_logic_utils import is_solid, in_bounds


def is_game_over(state, params, static_params):
    """Terminal: player death or max timesteps."""
    done_steps = state.timestep >= params.max_timesteps
    is_dead = state.player_hp <= 0
    return done_steps | is_dead


def do_move(rng, state, action, static_params):
    """Handle player movement. Attack on bump."""
    delta = DIRECTION_VECTORS[action]
    new_pos = state.player_position + delta

    valid = in_bounds(new_pos, static_params.map_height, static_params.map_width)

    # Check tile at destination (safe index)
    safe_r = jnp.clip(new_pos[0], 0, static_params.map_height - 1)
    safe_c = jnp.clip(new_pos[1], 0, static_params.map_width - 1)
    target_tile = state.map[safe_r, safe_c]
    walkable = jnp.logical_not(is_solid(target_tile))

    # Check for monster at destination
    max_m = static_params.max_monsters
    mon_positions = state.monsters.position
    mon_mask = state.monsters.mask

    pos_match_r = mon_positions[:, 0] == new_pos[0]
    pos_match_c = mon_positions[:, 1] == new_pos[1]
    at_dest = pos_match_r & pos_match_c & mon_mask

    any_monster_there = jnp.any(at_dest)
    monster_at_dest = jnp.logical_and(valid, any_monster_there)

    blocked_by_monster = monster_at_dest

    can_move = jnp.logical_and(valid, walkable) & jnp.logical_not(blocked_by_monster)

    # Compute both branches
    rng, rng_attack = jax.random.split(rng)
    attack_state = do_melee_attack(rng_attack, state, new_pos, static_params)

    new_position = jnp.where(can_move, new_pos, state.player_position)
    move_state = state.replace(player_position=new_position)

    # Select: attack if monster, else move
    def _pick(attack_val, move_val):
        return jnp.where(monster_at_dest, attack_val, move_val)

    result_state = jax.tree.map(_pick, attack_state, move_state)
    return result_state


def update_monsters(rng, state, static_params):
    """Wrapper calling full_monster_ai with temple parameters."""
    return full_monster_ai(
        rng, state, static_params,
        ALTAR_POSITION, TEMPLE_MIN, TEMPLE_MAX, True
    )


def minihax_step(rng, state, action, params, static_params):
    """Main step function for ZombieHorde.

    1. Process player action (move/attack, search, or eat)
    2. Monster AI movement
    3. Monster attacks
    4. Score tracking (reward = score delta)
    """
    rng, _rng_action, _rng_monsters, _rng_attacks = jax.random.split(rng, 4)

    state = state.replace(timestep=state.timestep + 1)
    old_score = state.score

    # 1. Player action
    is_move = action <= Action.MOVE_SW  # Actions 0-7 are movement
    state = lax.cond(
        is_move,
        lambda s: do_move(_rng_action, s, action, static_params),
        lambda s: s,
        state,
    )

    # SEARCH (action 8): do nothing (wait in place)
    # EAT (action 9): no-op for now (no corpse/inventory system)

    # 2. Monster AI
    state = update_monsters(_rng_monsters, state, static_params)

    # 3. Monster attacks
    state = do_monster_attacks(_rng_attacks, state, static_params)

    # 4. Visibility update
    visible_map = compute_visible(
        state.player_position, state.map,
        static_params.map_height, static_params.map_width,
    )
    new_seen_map = update_seen_map(state.seen_map, visible_map)
    state = state.replace(seen_map=new_seen_map, visible_map=visible_map)

    # 5. Reward = score delta (matches ScoreScore from sol-main)
    reward = (state.score - old_score).astype(jnp.float32)

    # Zero reward on terminal (matches ScoreScore behavior)
    done = is_game_over(state, params, static_params)
    reward = jnp.where(done, 0.0, reward)

    return state, reward
