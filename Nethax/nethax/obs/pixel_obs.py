"""Pixel (sprite-rendered) observation builder for nethax.

Renders the dungeon map as an RGB image by looking up each glyph in a tile
atlas. Matches the visual fidelity expected by CNN-based vision policies.

Canonical reference:
  - vendor/nle/nle/nethack/nethack.py  (DUNGEON_SHAPE = 21 x 79)
  - Nethax/tiles/                      tile atlas
"""

import pathlib

import numpy as np
import jax.numpy as jnp

from Nethax.tiles.tile_data import GLYPH2TILE

# Map dimensions in cells (matches NLE DUNGEON_SHAPE)
MAP_H: int = 21
MAP_W: int = 79  # NLE convention: COLNO-1 = 79

# Sprite size in pixels per tile
TILE_PX: int = 16

# Output image shape: (height_px, width_px, RGB)
PIXEL_OBS_SHAPE: tuple[int, int, int] = (MAP_H * TILE_PX, MAP_W * TILE_PX, 3)

# Module-level cache for the tile atlas (loaded once, outside JIT boundary)
_TILES_CACHE = None


def _get_tiles() -> jnp.ndarray:
    """Load tiles atlas from disk on first call; return cached copy thereafter.

    Returns:
        jnp.ndarray of shape [N_TILES, 16, 16, 3], dtype uint8.
    """
    global _TILES_CACHE
    if _TILES_CACHE is None:
        path = pathlib.Path(__file__).parent.parent.parent / "tiles" / "tiles.npy"
        _TILES_CACHE = jnp.asarray(np.load(str(path)), dtype=jnp.uint8)
    return _TILES_CACHE


def build_pixel_observation(env_state) -> jnp.ndarray:
    """Render the current level's glyph map to a HxWx3 uint8 image.

    Pipeline:
        env_state -> glyphs (21x79 int16) -> tile_indices (21x79 int32)
                  -> sprites (21, 79, 16, 16, 3) -> image (336, 1264, 3).

    The tile atlas is loaded once at module level (outside JIT).  The
    rendering itself is pure JAX and fully jit-compatible.

    Args:
        env_state: nethax EnvState.

    Returns:
        jnp.ndarray of shape (MAP_H*TILE_PX, MAP_W*TILE_PX, 3) uint8.
    """
    from Nethax.nethax.obs.nle_obs import build_glyphs

    glyphs = build_glyphs(env_state)  # (21, 79) int16

    tiles = _get_tiles()              # (N_TILES, 16, 16, 3) uint8

    glyph2tile = jnp.asarray(GLYPH2TILE, dtype=jnp.int32)

    # Clamp glyph IDs to valid range; out-of-range glyphs fall back to tile 0
    safe_glyphs = jnp.clip(glyphs, 0, len(GLYPH2TILE) - 1).astype(jnp.int32)
    tile_indices = glyph2tile[safe_glyphs]  # (21, 79) int32

    # Gather sprites: (21, 79, 16, 16, 3)
    sprites = tiles[tile_indices]

    # Rearrange (21, 79, 16, 16, 3) -> (21*16, 79*16, 3)
    img = sprites.transpose(0, 2, 1, 3, 4).reshape(MAP_H * TILE_PX, MAP_W * TILE_PX, 3)
    return img
