"""World generators for Minihax-KeyAndDoor-v0 and Minihax-KeyAndDoorTmp-v0.

key_and_door.des: 5x5 room with 2x2 subroom. Locked door at relative (col=2, row=1)
of the room. Skeleton key random in room. Downstair random in subroom.

key_and_door_tmp.des: similar but randomized room/subroom sizes.
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    TileType, ItemType,
)
from Nethax.minihax.primitives.visibility import compute_visible, compute_lit_map
from Nethax.minihax.world_gen.combat_common import (
    pad_map, empty_combat_state,
)


def _build_room(game_map, r0, c0, h, w):
    """Build a walled room at position (r0, c0) with size (h, w) including walls.

    Returns updated game_map.
    """
    # Top and bottom walls
    game_map = game_map.at[r0, c0:c0+w].set(TileType.HWALL)
    game_map = game_map.at[r0+h-1, c0:c0+w].set(TileType.HWALL)
    # Left and right walls
    game_map = game_map.at[r0+1:r0+h-1, c0].set(TileType.VWALL)
    game_map = game_map.at[r0+1:r0+h-1, c0+w-1].set(TileType.VWALL)
    # Floor
    rows = jnp.arange(game_map.shape[0])[:, None]
    cols = jnp.arange(game_map.shape[1])[None, :]
    interior = (rows >= r0+1) & (rows <= r0+h-2) & (cols >= c0+1) & (cols <= c0+w-2)
    game_map = jnp.where(interior, TileType.FLOOR, game_map)
    return game_map


def generate_key_and_door(rng, params, static_params):
    """Generate a KeyAndDoor environment state.

    Fixed layout: 7x7 room (rows 0-6, cols 0-6) with 4x4 subroom (rows 1-4, cols 3-6).
    Locked door on left wall of subroom at (row=2, col=3).
    Skeleton key placed randomly in the outer room area.
    Downstair placed randomly in subroom interior.
    """
    rng, rng_key_r, rng_key_c, rng_stair_r, rng_stair_c, rng_player_r, rng_player_c, rng_state = \
        jax.random.split(rng, 8)

    # Active map: 7x7
    map_h, map_w = 7, 7
    game_map = jnp.full((map_h, map_w), TileType.VOID, dtype=jnp.int32)

    # Outer room: rows 0-6, cols 0-6
    game_map = _build_room(game_map, 0, 0, 7, 7)

    # Subroom: rows 1-4, cols 3-6 (4 tall x 4 wide including walls)
    # Subroom walls overwrite outer room floor
    game_map = game_map.at[1, 3:7].set(TileType.HWALL)
    game_map = game_map.at[4, 3:7].set(TileType.HWALL)
    game_map = game_map.at[2:4, 3].set(TileType.VWALL)
    # Right wall is outer room wall already

    # Subroom interior: rows 2-3, cols 4-5
    rows = jnp.arange(map_h)[:, None]
    cols = jnp.arange(map_w)[None, :]
    subroom_int = (rows >= 2) & (rows <= 3) & (cols >= 4) & (cols <= 5)
    game_map = jnp.where(subroom_int, TileType.FLOOR, game_map)

    # Locked door at (row=2, col=3) -- left wall of subroom
    game_map = game_map.at[2, 3].set(TileType.DOOR_LOCKED)

    # Player position: random in outer room (not subroom)
    # Outer room floor: rows 1-5, cols 1-2
    player_r = jax.random.randint(rng_player_r, (), 1, 6)
    player_c = jax.random.randint(rng_player_c, (), 1, 3)
    player_pos = jnp.array([player_r, player_c], dtype=jnp.int32)

    # Skeleton key: random in outer room area (rows 1-5, cols 1-2)
    key_r = jax.random.randint(rng_key_r, (), 1, 6)
    key_c = jax.random.randint(rng_key_c, (), 1, 3)

    # Downstair: random in subroom interior (rows 2-3, cols 4-5)
    stair_r = jax.random.randint(rng_stair_r, (), 2, 4)
    stair_c = jax.random.randint(rng_stair_c, (), 4, 6)
    stair_pos = jnp.array([stair_r, stair_c], dtype=jnp.int32)
    game_map = game_map.at[stair_r, stair_c].set(TileType.DOWNSTAIR)

    padded_map = pad_map(game_map, static_params)

    # Ground items: skeleton key
    max_gi = static_params.max_ground_items
    gi_positions = jnp.zeros((max_gi, 2), dtype=jnp.int32)
    gi_positions = gi_positions.at[0].set(jnp.array([key_r, key_c]))
    gi_types = jnp.zeros(max_gi, dtype=jnp.int32)
    gi_types = gi_types.at[0].set(ItemType.SKELETON_KEY)
    gi_mask = jnp.zeros(max_gi, dtype=jnp.bool_)
    gi_mask = gi_mask.at[0].set(True)

    lit_map = compute_lit_map(padded_map)
    visible_map = compute_visible(player_pos, padded_map, static_params.map_height, static_params.map_width, lit_map)
    state = empty_combat_state(static_params, rng_state)
    state = state.replace(
        map=padded_map,
        player_position=player_pos,
        downstair_position=stair_pos,
        seen_map=visible_map,
        visible_map=visible_map,
        lit_map=lit_map,
        ground_items=state.ground_items.replace(
            position=gi_positions,
            type_id=gi_types,
            mask=gi_mask,
        ),
    )
    return state


def generate_key_and_door_tmp(rng, params, static_params):
    """Generate a KeyAndDoorTmp environment state.

    Randomized room sizes. Outer room: 7-9 wide x 7-9 tall.
    Subroom: 3-4 wide x 3-4 tall, placed inside outer room.
    """
    rng, rng_ow, rng_oh, rng_sw, rng_sh = jax.random.split(rng, 5)
    rng, rng_key_r, rng_key_c, rng_stair_r, rng_stair_c = jax.random.split(rng, 5)
    rng, rng_player_r, rng_player_c, rng_door_side, rng_state = jax.random.split(rng, 5)

    # Outer room dimensions (including walls)
    outer_w = jax.random.randint(rng_ow, (), 7, 10)  # 7-9
    outer_h = jax.random.randint(rng_oh, (), 7, 10)  # 7-9

    # Use max possible size for the fixed-shape map, fill with VOID
    max_dim = 10
    game_map = jnp.full((max_dim, max_dim), TileType.VOID, dtype=jnp.int32)

    # Build outer room programmatically using vectorized ops
    rows = jnp.arange(max_dim)[:, None]
    cols = jnp.arange(max_dim)[None, :]

    # Outer walls
    is_top = (rows == 0) & (cols < outer_w)
    is_bot = (rows == outer_h - 1) & (cols < outer_w)
    is_left = (cols == 0) & (rows > 0) & (rows < outer_h - 1)
    is_right = (cols == outer_w - 1) & (rows > 0) & (rows < outer_h - 1)
    is_interior = (rows > 0) & (rows < outer_h - 1) & (cols > 0) & (cols < outer_w - 1)

    game_map = jnp.where(is_top | is_bot, TileType.HWALL, game_map)
    game_map = jnp.where(is_left | is_right, TileType.VWALL, game_map)
    game_map = jnp.where(is_interior, TileType.FLOOR, game_map)

    # Subroom: placed at right side of outer room
    sub_w = jax.random.randint(rng_sw, (), 3, 5)  # 3-4
    sub_h = jax.random.randint(rng_sh, (), 3, 5)  # 3-4

    # Subroom position: top-right inside outer room
    sub_r0 = jnp.int32(1)
    sub_c0 = outer_w - 1 - sub_w  # Align to right wall

    # Subroom walls
    is_sub_top = (rows == sub_r0) & (cols >= sub_c0) & (cols < sub_c0 + sub_w)
    is_sub_bot = (rows == sub_r0 + sub_h - 1) & (cols >= sub_c0) & (cols < sub_c0 + sub_w)
    is_sub_left = (cols == sub_c0) & (rows > sub_r0) & (rows < sub_r0 + sub_h - 1)
    is_sub_interior = (rows > sub_r0) & (rows < sub_r0 + sub_h - 1) & \
                      (cols > sub_c0) & (cols < sub_c0 + sub_w - 1)

    game_map = jnp.where(is_sub_top | is_sub_bot, TileType.HWALL, game_map)
    game_map = jnp.where(is_sub_left, TileType.VWALL, game_map)
    game_map = jnp.where(is_sub_interior, TileType.FLOOR, game_map)

    # Locked door on left wall of subroom (middle row)
    door_r = sub_r0 + 1
    game_map = game_map.at[door_r, sub_c0].set(TileType.DOOR_LOCKED)

    # Player: random in outer room floor (left of subroom)
    player_r = jax.random.randint(rng_player_r, (), 1, jnp.maximum(outer_h - 1, 2))
    player_c = jax.random.randint(rng_player_c, (), 1, jnp.maximum(sub_c0, 2))
    player_pos = jnp.array([player_r, player_c], dtype=jnp.int32)

    # Key: random in outer room
    key_r = jax.random.randint(rng_key_r, (), 1, jnp.maximum(outer_h - 1, 2))
    key_c = jax.random.randint(rng_key_c, (), 1, jnp.maximum(sub_c0, 2))

    # Stair: in subroom interior
    stair_r = jax.random.randint(rng_stair_r, (), sub_r0 + 1,
                                  jnp.maximum(sub_r0 + sub_h - 1, sub_r0 + 2))
    stair_c = jax.random.randint(rng_stair_c, (), sub_c0 + 1,
                                  jnp.maximum(sub_c0 + sub_w - 1, sub_c0 + 2))
    stair_pos = jnp.array([stair_r, stair_c], dtype=jnp.int32)
    game_map = game_map.at[stair_r, stair_c].set(TileType.DOWNSTAIR)

    padded_map = pad_map(game_map, static_params)

    # Ground items: skeleton key
    max_gi = static_params.max_ground_items
    gi_positions = jnp.zeros((max_gi, 2), dtype=jnp.int32)
    gi_positions = gi_positions.at[0].set(jnp.array([key_r, key_c]))
    gi_types = jnp.zeros(max_gi, dtype=jnp.int32)
    gi_types = gi_types.at[0].set(ItemType.SKELETON_KEY)
    gi_mask = jnp.zeros(max_gi, dtype=jnp.bool_)
    gi_mask = gi_mask.at[0].set(True)

    lit_map = compute_lit_map(padded_map)
    visible_map = compute_visible(player_pos, padded_map, static_params.map_height, static_params.map_width, lit_map)
    state = empty_combat_state(static_params, rng_state)
    state = state.replace(
        map=padded_map,
        player_position=player_pos,
        downstair_position=stair_pos,
        seen_map=visible_map,
        visible_map=visible_map,
        lit_map=lit_map,
        ground_items=state.ground_items.replace(
            position=gi_positions,
            type_id=gi_types,
            mask=gi_mask,
        ),
    )
    return state
