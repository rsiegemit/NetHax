"""Movement primitives shared across all tiers."""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    Action, TileType, DIRECTION_VECTORS,
)
from Nethax.minihax.util.game_logic_utils import is_solid, in_bounds


def move_player(player_pos, action, game_map, map_h, map_w):
    """Pure player movement without combat or monster checking.

    Checks bounds and tile solidity only. Does NOT check for monsters
    at the destination (that's handled by tier-specific step functions).

    Args:
        player_pos: jnp.ndarray [2] — current (row, col)
        action: int — action index (0-7 for movement, others return no movement)
        game_map: jnp.ndarray [map_h, map_w] — tile type IDs
        map_h: int — map height
        map_w: int — map width

    Returns:
        new_pos: jnp.ndarray [2] — new position (same as old if blocked)
        moved: bool — whether the player actually moved
    """
    delta = DIRECTION_VECTORS[action]
    new_pos = player_pos + delta

    valid = in_bounds(new_pos, map_h, map_w)

    # Check tile at destination (safe index to prevent OOB)
    safe_r = jnp.clip(new_pos[0], 0, map_h - 1)
    safe_c = jnp.clip(new_pos[1], 0, map_w - 1)
    target_tile = game_map[safe_r, safe_c]
    walkable = jnp.logical_not(is_solid(target_tile))

    can_move = valid & walkable
    final_pos = jnp.where(can_move, new_pos, player_pos)

    return final_pos, can_move


def check_stair_goal(player_pos, stair_pos):
    """Check if player is standing on the downstair tile.

    Args:
        player_pos: jnp.ndarray [2] — (row, col)
        stair_pos: jnp.ndarray [2] — (row, col) of downstair

    Returns:
        bool — True if player is on the downstair
    """
    return (player_pos[0] == stair_pos[0]) & (player_pos[1] == stair_pos[1])


def push_boulder(game_map, player_pos, action, map_h, map_w, pits_remaining, restrict_diagonal=False):
    """Boulder push logic for Sokoban. Pure map operations, no lax.scan.

    If player moves into a BOULDER tile and the tile beyond is FLOOR or PIT:
    - If beyond is PIT: set beyond to PIT_FILLED, remove boulder, decrement pits_remaining
    - If beyond is FLOOR: move boulder to beyond tile
    - Move player to boulder's old position

    If the tile beyond is NOT pushable (wall, another boulder, OOB), nothing happens.

    Args:
        game_map: jnp.ndarray [map_h, map_w]
        player_pos: jnp.ndarray [2]
        action: int
        map_h, map_w: int
        pits_remaining: int
        restrict_diagonal: bool — if True, diagonal boulder pushes are forbidden (Sokoban rule)

    Returns:
        new_map: jnp.ndarray [map_h, map_w]
        new_player_pos: jnp.ndarray [2]
        new_pits_remaining: int
        pushed: bool
    """
    delta = DIRECTION_VECTORS[action]
    target_pos = player_pos + delta
    beyond_pos = target_pos + delta

    # Check if diagonal push (both row and column change)
    is_diagonal = (delta[0] != 0) & (delta[1] != 0)

    # Safe indexing for target
    t_r = jnp.clip(target_pos[0], 0, map_h - 1)
    t_c = jnp.clip(target_pos[1], 0, map_w - 1)
    target_tile = game_map[t_r, t_c]

    # Safe indexing for beyond
    b_r = jnp.clip(beyond_pos[0], 0, map_h - 1)
    b_c = jnp.clip(beyond_pos[1], 0, map_w - 1)
    beyond_tile = game_map[b_r, b_c]

    # Check conditions
    target_valid = in_bounds(target_pos, map_h, map_w)
    beyond_valid = in_bounds(beyond_pos, map_h, map_w)
    is_boulder = target_tile == TileType.BOULDER
    beyond_floor = beyond_tile == TileType.FLOOR
    beyond_pit = beyond_tile == TileType.PIT
    beyond_open_door = beyond_tile == TileType.DOOR_OPEN
    beyond_pit_filled = beyond_tile == TileType.PIT_FILLED
    beyond_downstair = beyond_tile == TileType.DOWNSTAIR
    beyond_pushable = beyond_floor | beyond_pit | beyond_open_door | beyond_pit_filled | beyond_downstair

    # Apply diagonal restriction if enabled
    diagonal_blocked = restrict_diagonal & is_diagonal
    can_push = target_valid & beyond_valid & is_boulder & beyond_pushable & jnp.logical_not(diagonal_blocked)

    # Compute new map for push case
    # Remove boulder from target position -> becomes FLOOR
    new_map = game_map.at[t_r, t_c].set(
        jnp.where(can_push, TileType.FLOOR, target_tile)
    )
    # Place boulder at beyond position OR fill pit
    beyond_new_tile = jnp.where(beyond_pit, TileType.PIT_FILLED, TileType.BOULDER)
    new_map = new_map.at[b_r, b_c].set(
        jnp.where(can_push, beyond_new_tile, new_map[b_r, b_c])
    )

    # Pits remaining: decrement if pushed into pit
    pit_filled = can_push & beyond_pit
    new_pits = pits_remaining - jnp.where(pit_filled, 1, 0)

    # Player moves to boulder's old position
    new_player = jnp.where(can_push, target_pos, player_pos)

    return new_map, new_player, new_pits, can_push
