"""Game logic for Tier 1 Navigation environments.

The simplest tier: movement + stair goal check.
No combat, no items, no monsters.
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.primitives.movement import move_player, check_stair_goal
from Nethax.minihax.primitives.visibility import compute_visible, update_seen_map


def navigation_step(rng, state, action, params, static_params):
    """Navigation step function: movement + stair check.

    Action space: 8 directional moves + SEARCH/wait + GO_DOWN_STAIRS = 10 actions.

    Win condition: Player on downstair AND GO_DOWN_STAIRS action taken.

    Args:
        rng: JAX PRNG key
        state: NavigationState
        action: int (0-9)
        params: EnvParams
        static_params: NavigationStaticParams

    Returns:
        new_state: NavigationState
    """
    # Split RNG for next step
    rng, rng_next = jax.random.split(rng)

    # Movement actions (0-7)
    is_move = action < 8

    # Try movement
    new_pos, moved = move_player(
        state.player_position, action, state.map,
        static_params.map_height, static_params.map_width
    )

    # SEARCH/wait (8) does nothing
    # GO_DOWN_STAIRS (10) requires being on stair
    final_pos = jnp.where(is_move, new_pos, state.player_position)

    # Check win condition: GO_DOWN_STAIRS action while on stair
    on_stair = check_stair_goal(final_pos, state.downstair_position)
    go_down_action = action == 10  # GO_DOWN_STAIRS
    won = on_stair & go_down_action

    # Terminal: win or max timesteps
    new_timestep = state.timestep + 1
    timeout = new_timestep >= params.max_timesteps
    terminal = won | timeout

    # Visibility update
    visible_map = compute_visible(final_pos, state.map, static_params.map_height, static_params.map_width)
    new_seen_map = update_seen_map(state.seen_map, visible_map)

    new_state = state.replace(
        player_position=final_pos,
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
