"""Game logic for Tier 1 Navigation environments.

The simplest tier: movement + door interaction + stair goal check.
No combat, no items, no monsters.
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.primitives.movement import move_player, check_stair_goal
from Nethax.minihax.primitives.doors import open_door_adjacent, kick_door
from Nethax.minihax.primitives.visibility import compute_visible, update_seen_map


def navigation_step(rng, state, action, params, static_params):
    """Navigation step function: movement + doors + stair check.

    Action space (matches MiniHack NAVIGATE_ACTIONS):
        0-7: compass directions
        8:   OPEN (open adjacent closed door)
        9:   KICK (kick adjacent locked door)
        10:  SEARCH/WAIT (no-op)

    Args:
        rng: JAX PRNG key
        state: NavigationState
        action: int (0-10)
        params: EnvParams
        static_params: NavigationStaticParams

    Returns:
        new_state: NavigationState
    """
    map_h = static_params.map_height
    map_w = static_params.map_width

    rng, rng_kick, rng_next = jax.random.split(rng, 3)

    # Movement actions (0-7)
    is_move = action < 8
    is_open = (action == 8)
    is_kick = (action == 9)

    # Try movement
    new_pos, moved = move_player(
        state.player_position, action, state.map, map_h, map_w
    )

    final_pos = jnp.where(is_move, new_pos, state.player_position)

    # --- OPEN: open adjacent closed door ---
    map_after_open, opened = open_door_adjacent(
        state.map, final_pos, map_h, map_w
    )

    # --- KICK: kick adjacent locked door ---
    map_after_kick, kick_success, rng_kick = kick_door(
        rng_kick, state.map, final_pos, map_h, map_w
    )

    # Select map based on action
    new_map = jnp.where(is_open, map_after_open,
              jnp.where(is_kick, map_after_kick,
                        state.map))

    # Check win condition
    on_stair = check_stair_goal(final_pos, state.downstair_position)
    go_down = (action == 10)  # SEARCH/WAIT doubles as descend check when auto_descend=False
    won = jnp.where(params.auto_descend, on_stair, on_stair & go_down)

    # Terminal: win or max timesteps
    new_timestep = state.timestep + 1
    timeout = new_timestep >= params.max_timesteps
    terminal = won | timeout

    # Visibility update
    visible_map = compute_visible(final_pos, new_map, map_h, map_w, state.lit_map)
    new_seen_map = update_seen_map(state.seen_map, visible_map)

    new_state = state.replace(
        player_position=final_pos,
        map=new_map,
        seen_map=new_seen_map,
        visible_map=visible_map,
        timestep=new_timestep,
        terminal=terminal,
        state_rng=rng_next,
    )

    return new_state


def is_navigation_done(state, params):
    """Check if navigation episode is done."""
    # Terminal is already set in state
    return state.terminal
