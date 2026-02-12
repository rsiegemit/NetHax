"""Game logic for Tier 4 Sokoban environments."""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import Action, NUM_ACTIONS_TIER4
from Nethax.minihax.primitives.movement import move_player, check_stair_goal, push_boulder
from Nethax.minihax.primitives.visibility import compute_visible, update_seen_map


def sokoban_step(rng, state, action, params, static_params):
    """Sokoban step function: movement + boulder pushing + pit filling.

    Action space: 8 directional moves + SEARCH/wait (9 actions total).

    Boulder push logic:
    - If moving into a boulder AND the tile beyond is pushable (FLOOR, PIT, etc.):
      - Move boulder to beyond tile (or fill pit)
      - Player moves to boulder's old position
    - Otherwise, try normal movement (no boulder or blocked push)

    Win condition: All pits filled (pits_remaining == 0) AND player on downstair.

    Args:
        rng: JAX PRNG key
        state: SokobanState
        action: int (0-8)
        params: EnvParams
        static_params: SokobanStaticParams

    Returns:
        new_state: SokobanState
    """
    # Split RNG for next step
    rng, rng_next = jax.random.split(rng)

    # Movement actions (0-7)
    is_move = action < 8

    # Try boulder push first (only on movement actions)
    new_map_push, new_pos_push, new_pits_push, pushed = push_boulder(
        state.map, state.player_position, action,
        static_params.map_height, static_params.map_width,
        state.pits_remaining
    )

    # Try normal movement (no boulder push)
    new_pos_move, moved = move_player(
        state.player_position, action, state.map,
        static_params.map_height, static_params.map_width
    )

    # Select: if pushed, use push result; else use normal move result
    final_pos = jnp.where(pushed, new_pos_push, new_pos_move)
    final_map = jnp.where(pushed, new_map_push, state.map)
    final_pits = jnp.where(pushed, new_pits_push, state.pits_remaining)

    # SEARCH/wait action (8) does nothing
    final_pos = jnp.where(is_move, final_pos, state.player_position)
    final_map = jnp.where(is_move, final_map, state.map)
    final_pits = jnp.where(is_move, final_pits, state.pits_remaining)

    # Check win condition: all pits filled AND on downstair
    all_pits_filled = final_pits == 0
    on_stair = check_stair_goal(final_pos, state.downstair_position)
    won = all_pits_filled & on_stair

    # Terminal: win or max timesteps
    new_timestep = state.timestep + 1
    timeout = new_timestep >= params.max_timesteps
    terminal = won | timeout

    # Visibility update (use final_map since boulders may have moved)
    visible_map = compute_visible(final_pos, final_map, static_params.map_height, static_params.map_width)
    new_seen_map = update_seen_map(state.seen_map, visible_map)

    new_state = state.replace(
        map=final_map,
        player_position=final_pos,
        pits_remaining=final_pits,
        seen_map=new_seen_map,
        visible_map=visible_map,
        timestep=new_timestep,
        terminal=terminal,
        state_rng=rng_next,
    )

    return new_state


def is_sokoban_done(state, params):
    """Check if Sokoban episode is done."""
    won = (state.pits_remaining == 0) & check_stair_goal(
        state.player_position, state.downstair_position
    )
    timeout = state.timestep >= params.max_timesteps
    return won | timeout
