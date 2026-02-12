"""Door primitives for Tier 2 and 3."""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import TileType
from Nethax.minihax.util.game_logic_utils import in_bounds

# 8 cardinal + diagonal directions
_DOOR_DELTAS = jnp.array([
    [-1, 0], [1, 0], [0, 1], [0, -1],
    [-1, 1], [-1, -1], [1, 1], [1, -1],
], dtype=jnp.int32)


def kick_door(rng, game_map, player_pos, map_h, map_w):
    """Kick the nearest adjacent locked door. 1/4 chance to break it open.

    Scans all 8 adjacent tiles for a DOOR_LOCKED and attempts to kick
    the first one found.

    Args:
        rng: JAX PRNG key
        game_map: jnp.ndarray [map_h, map_w]
        player_pos: jnp.ndarray [2] — (row, col)
        map_h: int
        map_w: int

    Returns:
        new_map: jnp.ndarray [map_h, map_w]
        success: bool — whether the door was broken open
        new_rng: JAX PRNG key
    """
    # Check all 8 adjacent tiles for locked doors
    candidates = player_pos[None, :] + _DOOR_DELTAS  # [8, 2]

    def check_locked(cand_pos):
        valid = in_bounds(cand_pos, map_h, map_w)
        sr = jnp.clip(cand_pos[0], 0, map_h - 1)
        sc = jnp.clip(cand_pos[1], 0, map_w - 1)
        tile = game_map[sr, sc]
        return valid & (tile == TileType.DOOR_LOCKED)

    is_locked = jax.vmap(check_locked)(candidates)  # [8]
    any_locked = jnp.any(is_locked)
    door_idx = jnp.argmax(is_locked)
    safe_idx = jnp.where(any_locked, door_idx, 0)
    door_pos = candidates[safe_idx]

    sr = jnp.clip(door_pos[0], 0, map_h - 1)
    sc = jnp.clip(door_pos[1], 0, map_w - 1)

    rng, rng_kick = jax.random.split(rng)
    roll = jax.random.randint(rng_kick, (), 0, 4)
    success = any_locked & (roll == 0)  # 1/4 chance

    tile = game_map[sr, sc]
    new_tile = jnp.where(success, TileType.DOOR_OPEN, tile)
    new_map = game_map.at[sr, sc].set(new_tile)

    return new_map, success, rng


def open_door_adjacent(game_map, player_pos, map_h, map_w):
    """Open the nearest adjacent closed door.

    Scans all 8 adjacent tiles for a DOOR_CLOSED and opens the first one found.

    Args:
        game_map: jnp.ndarray [map_h, map_w]
        player_pos: jnp.ndarray [2] — (row, col)
        map_h: int
        map_w: int

    Returns:
        new_map: jnp.ndarray [map_h, map_w]
        opened: bool — whether a door was opened
    """
    candidates = player_pos[None, :] + _DOOR_DELTAS  # [8, 2]

    def check_closed(cand_pos):
        valid = in_bounds(cand_pos, map_h, map_w)
        sr = jnp.clip(cand_pos[0], 0, map_h - 1)
        sc = jnp.clip(cand_pos[1], 0, map_w - 1)
        tile = game_map[sr, sc]
        return valid & (tile == TileType.DOOR_CLOSED)

    is_closed = jax.vmap(check_closed)(candidates)  # [8]
    any_closed = jnp.any(is_closed)
    door_idx = jnp.argmax(is_closed)
    safe_idx = jnp.where(any_closed, door_idx, 0)
    door_pos = candidates[safe_idx]

    sr = jnp.clip(door_pos[0], 0, map_h - 1)
    sc = jnp.clip(door_pos[1], 0, map_w - 1)

    tile = game_map[sr, sc]
    new_tile = jnp.where(any_closed, TileType.DOOR_OPEN, tile)
    new_map = game_map.at[sr, sc].set(new_tile)

    return new_map, any_closed


def unlock_door_adjacent(game_map, player_pos, has_key, map_h, map_w):
    """Unlock the nearest adjacent locked door if player has a key.

    Args:
        game_map: jnp.ndarray [map_h, map_w]
        player_pos: jnp.ndarray [2] — (row, col)
        has_key: bool — whether player has a skeleton key
        map_h: int
        map_w: int

    Returns:
        new_map: jnp.ndarray [map_h, map_w]
        unlocked: bool — whether a door was unlocked
    """
    candidates = player_pos[None, :] + _DOOR_DELTAS  # [8, 2]

    def check_locked(cand_pos):
        valid = in_bounds(cand_pos, map_h, map_w)
        sr = jnp.clip(cand_pos[0], 0, map_h - 1)
        sc = jnp.clip(cand_pos[1], 0, map_w - 1)
        tile = game_map[sr, sc]
        return valid & (tile == TileType.DOOR_LOCKED)

    is_locked = jax.vmap(check_locked)(candidates)  # [8]
    any_locked = jnp.any(is_locked)
    door_idx = jnp.argmax(is_locked)
    safe_idx = jnp.where(any_locked, door_idx, 0)
    door_pos = candidates[safe_idx]

    sr = jnp.clip(door_pos[0], 0, map_h - 1)
    sc = jnp.clip(door_pos[1], 0, map_w - 1)

    can_unlock = any_locked & has_key
    tile = game_map[sr, sc]
    new_tile = jnp.where(can_unlock, TileType.DOOR_CLOSED, tile)  # Unlock -> closed (openable)
    new_map = game_map.at[sr, sc].set(new_tile)

    return new_map, can_unlock
