"""Pixel observation parity / structural tests.

Vendor NLE renders pixel observations via ``vendor/nle/nle/env/rendering.py``
(see ``RenderTiles``) using a 16x16 RGB tile atlas indexed by glyph.  Our
implementation in ``Nethax.nethax.obs.pixel_obs`` follows the same pipeline:

    env_state -> glyphs (21x79 int16)
              -> GLYPH2TILE[glyphs] (21x79 int32)
              -> tiles[tile_idx]    (21x79x16x16x3 uint8)
              -> reshape            (336x1264x3  uint8)

These tests verify the structural contract: shape, dtype, JIT-safety, and
the tile-atlas mapping (each cell renders to a contiguous 16x16 block).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.obs.pixel_obs import (
    MAP_H,
    MAP_W,
    TILE_PX,
    PIXEL_OBS_SHAPE,
    build_pixel_observation,
)


_RNG = jax.random.PRNGKey(0)


def _make_state():
    from Nethax.nethax.state import EnvState
    return EnvState.default(_RNG)


# ---------------------------------------------------------------------------
# Structural parity
# ---------------------------------------------------------------------------


def test_pixel_obs_shape_matches_nle_tile_grid():
    """Image is (MAP_H*TILE_PX, MAP_W*TILE_PX, 3) — matches NLE DUNGEON_SHAPE."""
    # NLE DUNGEON_SHAPE is (21, 79) — confirm our constants pin to that.
    assert MAP_H == 21, "MAP_H must equal NLE ROWNO"
    assert MAP_W == 79, "MAP_W must equal NLE COLNO-1"
    assert TILE_PX == 16, "TILE_PX must equal vendor tile-atlas sprite size"
    assert PIXEL_OBS_SHAPE == (21 * 16, 79 * 16, 3)
    assert PIXEL_OBS_SHAPE == (336, 1264, 3)


def test_pixel_obs_dtype_is_uint8():
    """Pixel observation is uint8 RGB."""
    state = _make_state()
    img = build_pixel_observation(state)
    assert img.dtype == jnp.uint8, f"Expected uint8, got {img.dtype}"
    assert img.ndim == 3
    assert img.shape[2] == 3  # RGB channels


def test_pixel_obs_is_jit_safe():
    """build_pixel_observation works under jax.jit without errors."""
    state = _make_state()
    jitted = jax.jit(build_pixel_observation)
    img = jitted(state)
    assert tuple(img.shape) == PIXEL_OBS_SHAPE
    assert img.dtype == jnp.uint8
    # JIT-traced output must be a concrete jnp array.
    arr = np.asarray(img)
    assert arr.shape == PIXEL_OBS_SHAPE


def test_pixel_obs_renders_non_black_pixels():
    """With FLOOR terrain everywhere, the rendered image is not all black.

    Vendor NLE tile 0 (or whichever GLYPH2TILE[FLOOR_GLYPH] resolves to)
    is a non-zero sprite.  Verifies the tile atlas loaded and indexing works.
    """
    from Nethax.nethax.state import EnvState
    from Nethax.nethax.constants import TileType

    state = EnvState.default(_RNG)
    floor_terrain = jnp.full_like(state.terrain, TileType.FLOOR)
    state = state.replace(terrain=floor_terrain)
    img = build_pixel_observation(state)
    assert jnp.any(img > 0), "All pixels are zero — tile atlas not loaded?"


def test_pixel_obs_tile_block_structure():
    """Each (cell_row, cell_col) maps to a contiguous 16x16 block in the image.

    Two cells with the same glyph must produce identical 16x16 sprite blocks.
    """
    state = _make_state()
    img = build_pixel_observation(state)
    arr = np.asarray(img)

    # Block at (cell 0, 0) and block at (cell 0, 1).
    block_00 = arr[0:TILE_PX, 0:TILE_PX, :]
    block_01 = arr[0:TILE_PX, TILE_PX:2 * TILE_PX, :]

    # Either both blocks are identical (same glyph rendered to both cells, which
    # is the default zero-state behavior) or they differ but each is exactly
    # 16x16x3.
    assert block_00.shape == (TILE_PX, TILE_PX, 3)
    assert block_01.shape == (TILE_PX, TILE_PX, 3)

    # If glyphs at (0,0) and (0,1) come from the same source, blocks should match.
    from Nethax.nethax.obs.nle_obs import build_glyphs
    glyphs = build_glyphs(state)
    if int(glyphs[0, 0]) == int(glyphs[0, 1]):
        assert np.array_equal(block_00, block_01), (
            "Equal glyphs produced different 16x16 sprite blocks — broken tile mapping"
        )


def test_pixel_obs_uses_glyph_to_tile_mapping():
    """Sprites at every cell come from the GLYPH2TILE-indexed tile atlas.

    Manual reconstruction: gather sprites directly via the same lookup chain
    and confirm they match the rendered output bit-for-bit.
    """
    import pathlib
    from Nethax.tiles.tile_data import GLYPH2TILE
    from Nethax.nethax.obs.nle_obs import build_glyphs

    state = _make_state()
    img = np.asarray(build_pixel_observation(state))

    tiles_path = pathlib.Path(
        "/Users/rsiegelmann/Downloads/Projects/nethax/Nethax/tiles/tiles.npy"
    )
    tiles = np.load(str(tiles_path))  # (N_TILES, 16, 16, 3) uint8

    glyphs = np.asarray(build_glyphs(state)).astype(np.int32)
    glyph2tile = np.asarray(GLYPH2TILE, dtype=np.int32)
    safe = np.clip(glyphs, 0, len(glyph2tile) - 1)
    tile_idx = glyph2tile[safe]
    expected_sprites = tiles[tile_idx]  # (21, 79, 16, 16, 3)
    expected_img = expected_sprites.transpose(0, 2, 1, 3, 4).reshape(
        MAP_H * TILE_PX, MAP_W * TILE_PX, 3
    )

    assert img.shape == expected_img.shape
    assert np.array_equal(img, expected_img), (
        "Rendered image diverges from direct GLYPH2TILE lookup — mapping broken"
    )
