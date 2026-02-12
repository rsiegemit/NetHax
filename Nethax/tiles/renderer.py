"""JAX-compatible pixel renderer using NetHack tile sprites."""
import jax.numpy as jnp
import numpy as np
import os


TILE_SIZE = 16


def load_tiles():
    """Load tile sprites from tiles.npy.

    Returns:
        jnp.ndarray of shape [num_tiles, 16, 16, 3], uint8
    """
    tiles_path = os.path.join(os.path.dirname(__file__), "tiles.npy")
    tiles_np = np.load(tiles_path)
    return jnp.array(tiles_np, dtype=jnp.uint8)


def render_pixels(tile_map, glyph2tile, tiles_array):
    """Render a 2D glyph map to pixel RGB image.

    Args:
        tile_map: jnp.ndarray of shape [map_h, map_w] with tile type IDs.
                  These are glyph IDs that index into glyph2tile.
        glyph2tile: jnp.ndarray mapping glyph IDs to tile sprite indices.
        tiles_array: jnp.ndarray of shape [num_tiles, 16, 16, 3].

    Returns:
        jnp.ndarray of shape [map_h * 16, map_w * 16, 3], uint8
    """
    map_h, map_w = tile_map.shape

    # Map glyph IDs to tile indices
    tile_indices = glyph2tile[tile_map]  # [map_h, map_w]

    # Look up tile sprites
    sprites = tiles_array[tile_indices]  # [map_h, map_w, 16, 16, 3]

    # Reshape to pixel image: transpose and reshape
    # sprites shape: [map_h, map_w, tile_h, tile_w, 3]
    # Want: [map_h * tile_h, map_w * tile_w, 3]
    pixels = sprites.transpose(0, 2, 1, 3, 4)  # [map_h, tile_h, map_w, tile_w, 3]
    pixels = pixels.reshape(map_h * TILE_SIZE, map_w * TILE_SIZE, 3)

    return pixels
