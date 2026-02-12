"""World generator for Minihax-Chest-v0.

chest.des: 5x5 floor area. Skeleton key + chest (apple inside) on floor.
We add a downstair at a random position for the goal condition.
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import TileType, ItemType
from Nethax.minihax.primitives.visibility import compute_visible
from Nethax.minihax.world_gen.combat_common import (
    pad_map, empty_combat_state,
)


def generate_chest(rng, params, static_params):
    """Generate a Chest environment state.

    7x7 map (5x5 interior floor + walls). Skeleton key and apple (from chest)
    placed as ground items at random floor positions. Downstair at random pos.
    """
    rng, rng_pr, rng_pc, rng_kr, rng_kc, rng_ar, rng_ac = jax.random.split(rng, 7)
    rng, rng_sr, rng_sc, rng_state = jax.random.split(rng, 4)

    map_h, map_w = 7, 7
    game_map = jnp.full((map_h, map_w), TileType.VOID, dtype=jnp.int32)

    rows = jnp.arange(map_h)[:, None]
    cols = jnp.arange(map_w)[None, :]

    # Walls
    is_top = (rows == 0)
    is_bot = (rows == map_h - 1)
    is_left = (cols == 0) & (rows > 0) & (rows < map_h - 1)
    is_right = (cols == map_w - 1) & (rows > 0) & (rows < map_h - 1)
    is_interior = (rows > 0) & (rows < map_h - 1) & (cols > 0) & (cols < map_w - 1)

    game_map = jnp.where(is_top | is_bot, TileType.HWALL, game_map)
    game_map = jnp.where(is_left | is_right, TileType.VWALL, game_map)
    game_map = jnp.where(is_interior, TileType.FLOOR, game_map)

    # Player position: random in interior (rows 1-5, cols 1-5)
    player_r = jax.random.randint(rng_pr, (), 1, 6)
    player_c = jax.random.randint(rng_pc, (), 1, 6)
    player_pos = jnp.array([player_r, player_c], dtype=jnp.int32)

    # Downstair: random in interior
    stair_r = jax.random.randint(rng_sr, (), 1, 6)
    stair_c = jax.random.randint(rng_sc, (), 1, 6)
    stair_pos = jnp.array([stair_r, stair_c], dtype=jnp.int32)
    game_map = game_map.at[stair_r, stair_c].set(TileType.DOWNSTAIR)

    padded_map = pad_map(game_map, static_params)

    # Ground items: skeleton key + apple (from chest)
    key_r = jax.random.randint(rng_kr, (), 1, 6)
    key_c = jax.random.randint(rng_kc, (), 1, 6)
    apple_r = jax.random.randint(rng_ar, (), 1, 6)
    apple_c = jax.random.randint(rng_ac, (), 1, 6)

    max_gi = static_params.max_ground_items
    gi_positions = jnp.zeros((max_gi, 2), dtype=jnp.int32)
    gi_positions = gi_positions.at[0].set(jnp.array([key_r, key_c]))
    gi_positions = gi_positions.at[1].set(jnp.array([apple_r, apple_c]))

    gi_types = jnp.zeros(max_gi, dtype=jnp.int32)
    gi_types = gi_types.at[0].set(ItemType.SKELETON_KEY)
    gi_types = gi_types.at[1].set(ItemType.APPLE)

    gi_mask = jnp.zeros(max_gi, dtype=jnp.bool_)
    gi_mask = gi_mask.at[0].set(True)
    gi_mask = gi_mask.at[1].set(True)

    visible_map = compute_visible(player_pos, padded_map, static_params.map_height, static_params.map_width)
    state = empty_combat_state(static_params, rng_state)
    state = state.replace(
        map=padded_map,
        player_position=player_pos,
        downstair_position=stair_pos,
        seen_map=visible_map,
        visible_map=visible_map,
        ground_items=state.ground_items.replace(
            position=gi_positions,
            type_id=gi_types,
            mask=gi_mask,
        ),
    )
    return state
