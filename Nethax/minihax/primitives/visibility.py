"""Visibility/fog-of-war computation for minihax environments."""
import jax
import jax.numpy as jnp
from jax import lax

from Nethax.minihax.constants import TileType, NUM_TILE_TYPES


# Lookup table: True if tile type blocks vision
_blocks_list = [False] * NUM_TILE_TYPES
for _t in [TileType.VWALL, TileType.HWALL, TileType.TLCORN, TileType.TRCORN,
           TileType.BLCORN, TileType.BRCORN, TileType.TREE, TileType.CLOUD,
           TileType.DOOR_CLOSED, TileType.DOOR_LOCKED, TileType.VOID]:
    _blocks_list[int(_t)] = True
BLOCKS_VISION_TABLE = jnp.array(_blocks_list, dtype=jnp.bool_)


def _blocks_vision(tile_type):
    """Check if a tile blocks vision (scalar)."""
    return BLOCKS_VISION_TABLE[tile_type]


def compute_visible(player_position, game_map, map_height, map_width):
    """Compute which tiles are currently visible from the player position.

    Uses ray-casting (DDA) from the player to every tile on the map.
    A tile is visible if no vision-blocking tile exists on the ray
    between the player and that tile. Walls are visible if the ray
    reaches them (they block further propagation but are themselves seen).

    Args:
        player_position: jnp.ndarray [2] -- row, col
        game_map: jnp.ndarray [map_h, map_w] -- tile type IDs
        map_height: int
        map_width: int

    Returns:
        visible_map: jnp.ndarray [map_h, map_w] bool -- currently visible tiles
    """
    pr = player_position[0]
    pc = player_position[1]

    max_dist = 80  # covers the largest map dimension (80 cols)

    # Target positions for all cells
    rows = jnp.arange(map_height)[:, None]  # [H, 1]
    cols = jnp.arange(map_width)[None, :]   # [1, W]

    dr = (rows - pr).astype(jnp.float32)  # [H, W]
    dc = (cols - pc).astype(jnp.float32)  # [H, W]

    # Chebyshev distance = number of DDA steps to reach target
    dist = jnp.maximum(jnp.abs(dr), jnp.abs(dc))  # [H, W]
    safe_dist = jnp.maximum(dist, 1.0)

    # DDA step sizes
    step_r = dr / safe_dist  # [H, W]
    step_c = dc / safe_dist  # [H, W]

    # Ray steps 1..max_dist
    steps = jnp.arange(1, max_dist + 1).astype(jnp.float32)  # [D]

    # Compute intermediate positions along each ray: [H, W, D]
    ri = jnp.clip(
        jnp.round(pr + step_r[:, :, None] * steps[None, None, :]).astype(jnp.int32),
        0, map_height - 1
    )
    ci = jnp.clip(
        jnp.round(pc + step_c[:, :, None] * steps[None, None, :]).astype(jnp.int32),
        0, map_width - 1
    )

    # Only intermediate steps matter (before reaching the target)
    is_intermediate = steps[None, None, :] < dist[:, :, None]  # [H, W, D]

    # Check which cells along each ray block vision
    tiles_along_ray = game_map[ri, ci]  # [H, W, D]
    blocks = BLOCKS_VISION_TABLE[tiles_along_ray]  # [H, W, D]

    # Target is NOT visible if any intermediate cell blocks the ray
    any_blocked = jnp.any(blocks & is_intermediate, axis=-1)  # [H, W]
    visible_map = ~any_blocked

    # Player tile always visible
    visible_map = visible_map.at[pr, pc].set(True)

    # VOID tiles are never considered visible (always render as black)
    visible_map = visible_map & (game_map != TileType.VOID)

    return visible_map


def update_seen_map(seen_map, visible_map):
    """Update the seen map with newly visible tiles.

    Args:
        seen_map: jnp.ndarray [map_h, map_w] bool -- tiles ever seen
        visible_map: jnp.ndarray [map_h, map_w] bool -- tiles currently visible

    Returns:
        new_seen_map: jnp.ndarray [map_h, map_w] bool
    """
    return seen_map | visible_map
