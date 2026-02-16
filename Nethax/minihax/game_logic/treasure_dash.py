"""Game logic for TreasureDash environment.

Simplified step: movement + autopickup gold + auto-descend on stair.
No monsters, no HP/terrain damage, no items to use.
Reward = gold_picked_this_step + 20 * reached_stair.
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.primitives.movement import move_player
from Nethax.minihax.primitives.visibility import compute_visible, update_seen_map


def treasure_dash_step(rng, state, action, params, static_params):
    """TreasureDash step function.

    Phases:
    1. Player movement (8 compass + search/wait)
    2. Autopickup gold at new position
    3. Goal check (on downstair = terminal)
    4. Timeout check

    Args:
        rng: JAX PRNG key
        state: HazardState
        action: int (0-8: 8 compass directions + search)
        params: EnvParams
        static_params: HazardStaticParams

    Returns:
        new_state: HazardState
        reward: float
    """
    map_h = static_params.map_height
    map_w = static_params.map_width

    rng, rng_next = jax.random.split(rng)
    new_timestep = state.timestep + 1

    # ================================================================
    # Phase 1: Movement
    # ================================================================
    is_move = action < 8
    moved_pos, can_move = move_player(
        state.player_position, action, state.map, map_h, map_w
    )
    new_pos = jnp.where(is_move, moved_pos, state.player_position)

    # ================================================================
    # Phase 2: Autopickup gold at new position
    # ================================================================
    pos_match_r = state.ground_items.position[:, 0] == new_pos[0]
    pos_match_c = state.ground_items.position[:, 1] == new_pos[1]
    at_player = pos_match_r & pos_match_c & state.ground_items.mask

    # Count gold picked up this step
    gold_picked = jnp.sum(at_player.astype(jnp.int32))

    # Remove picked items (set mask to False)
    new_gi_mask = jnp.where(at_player, False, state.ground_items.mask)
    new_ground_items = state.ground_items.replace(mask=new_gi_mask)

    # ================================================================
    # Phase 3: Goal check — stepping on downstair = win (auto-descend)
    # ================================================================
    on_stair = (
        (new_pos[0] == state.downstair_position[0]) &
        (new_pos[1] == state.downstair_position[1])
    )

    # ================================================================
    # Phase 4: Terminal and reward
    # ================================================================
    timeout = new_timestep >= params.max_timesteps
    terminal = on_stair | timeout

    reward = gold_picked.astype(jnp.float32) + 20.0 * on_stair.astype(jnp.float32)

    # ================================================================
    # Visibility update
    # ================================================================
    visible_map = compute_visible(new_pos, state.map, map_h, map_w, state.lit_map)
    new_seen_map = update_seen_map(state.seen_map, visible_map)

    new_state = state.replace(
        player_position=new_pos,
        ground_items=new_ground_items,
        seen_map=new_seen_map,
        visible_map=visible_map,
        timestep=new_timestep,
        terminal=terminal,
        state_rng=rng_next,
    )

    return new_state, reward
