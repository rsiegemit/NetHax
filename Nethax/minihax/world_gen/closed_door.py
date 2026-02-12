"""World generator for Minihax-ClosedDoor-v0.

closed_door.des: 8x8 room with 4x4 subroom (random position within room).
Closed door on random wall of subroom. Downstair random in subroom.
Player starts random in outer room.
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import TileType
from Nethax.minihax.primitives.visibility import compute_visible
from Nethax.minihax.world_gen.combat_common import (
    pad_map, empty_combat_state,
)


def generate_closed_door(rng, params, static_params):
    """Generate a ClosedDoor environment state.

    Fixed outer room: 10x10 (rows 0-9, cols 0-9), floor inside.
    Subroom: 4x4 placed at a random valid position inside the outer room.
    Closed door on the left wall of the subroom.
    """
    rng, rng_sub_r, rng_sub_c = jax.random.split(rng, 3)
    rng, rng_stair_r, rng_stair_c = jax.random.split(rng, 3)
    rng, rng_player_r, rng_player_c, rng_state = jax.random.split(rng, 4)

    outer_h, outer_w = 10, 10
    sub_h, sub_w = 4, 4

    game_map = jnp.full((outer_h, outer_w), TileType.VOID, dtype=jnp.int32)
    rows = jnp.arange(outer_h)[:, None]
    cols = jnp.arange(outer_w)[None, :]

    # Outer walls
    is_top = (rows == 0) & (cols < outer_w)
    is_bot = (rows == outer_h - 1) & (cols < outer_w)
    is_left = (cols == 0) & (rows > 0) & (rows < outer_h - 1)
    is_right = (cols == outer_w - 1) & (rows > 0) & (rows < outer_h - 1)
    is_interior = (rows > 0) & (rows < outer_h - 1) & (cols > 0) & (cols < outer_w - 1)

    game_map = jnp.where(is_top | is_bot, TileType.HWALL, game_map)
    game_map = jnp.where(is_left | is_right, TileType.VWALL, game_map)
    game_map = jnp.where(is_interior, TileType.FLOOR, game_map)

    # Random subroom position (must fit inside outer room)
    # sub_r0 in [1, outer_h - sub_h - 1], sub_c0 in [2, outer_w - sub_w - 1]
    # so subroom doesn't overlap outer walls and leaves room to the left
    sub_r0 = jax.random.randint(rng_sub_r, (), 1, outer_h - sub_h)
    sub_c0 = jax.random.randint(rng_sub_c, (), 3, outer_w - sub_w)

    # Subroom walls
    is_sub_top = (rows == sub_r0) & (cols >= sub_c0) & (cols < sub_c0 + sub_w)
    is_sub_bot = (rows == sub_r0 + sub_h - 1) & (cols >= sub_c0) & (cols < sub_c0 + sub_w)
    is_sub_left = (cols == sub_c0) & (rows > sub_r0) & (rows < sub_r0 + sub_h - 1)
    is_sub_right = (cols == sub_c0 + sub_w - 1) & (rows > sub_r0) & (rows < sub_r0 + sub_h - 1)
    is_sub_interior = (rows > sub_r0) & (rows < sub_r0 + sub_h - 1) & \
                      (cols > sub_c0) & (cols < sub_c0 + sub_w - 1)

    game_map = jnp.where(is_sub_top | is_sub_bot, TileType.HWALL, game_map)
    game_map = jnp.where(is_sub_left | is_sub_right, TileType.VWALL, game_map)
    game_map = jnp.where(is_sub_interior, TileType.FLOOR, game_map)

    # Closed door on left wall of subroom (middle row)
    door_r = sub_r0 + 1
    game_map = game_map.at[door_r, sub_c0].set(TileType.DOOR_CLOSED)

    # Player: random in outer room floor, left of subroom
    player_r = jax.random.randint(rng_player_r, (), 1, outer_h - 1)
    player_c = jax.random.randint(rng_player_c, (), 1, jnp.maximum(sub_c0, 2))
    player_pos = jnp.array([player_r, player_c], dtype=jnp.int32)

    # Stair: in subroom interior
    stair_r = jax.random.randint(rng_stair_r, (), sub_r0 + 1,
                                  jnp.maximum(sub_r0 + sub_h - 1, sub_r0 + 2))
    stair_c = jax.random.randint(rng_stair_c, (), sub_c0 + 1,
                                  jnp.maximum(sub_c0 + sub_w - 1, sub_c0 + 2))
    stair_pos = jnp.array([stair_r, stair_c], dtype=jnp.int32)
    game_map = game_map.at[stair_r, stair_c].set(TileType.DOWNSTAIR)

    padded_map = pad_map(game_map, static_params)

    visible_map = compute_visible(player_pos, padded_map, static_params.map_height, static_params.map_width)
    state = empty_combat_state(static_params, rng_state)
    state = state.replace(
        map=padded_map,
        player_position=player_pos,
        downstair_position=stair_pos,
        seen_map=visible_map,
        visible_map=visible_map,
    )
    return state
