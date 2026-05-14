"""Tests for Wave 2 pixel observation rendering."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

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

_RNG = jax.random.PRNGKey(42)


def _make_state():
    from Nethax.nethax.state import EnvState
    return EnvState.default(_RNG)


def test_pixel_shape():
    """build_pixel_observation returns shape (336, 1264, 3) uint8."""
    state = _make_state()
    img = build_pixel_observation(state)
    assert tuple(img.shape) == PIXEL_OBS_SHAPE, (
        f"Expected shape {PIXEL_OBS_SHAPE}, got {tuple(img.shape)}"
    )
    assert img.dtype == jnp.uint8, f"Expected uint8, got {img.dtype}"


def test_pixel_shape_values():
    """Verify the constants themselves match the 336x1264x3 spec."""
    assert MAP_H == 21
    assert MAP_W == 79
    assert TILE_PX == 16
    assert PIXEL_OBS_SHAPE == (21 * 16, 79 * 16, 3)
    assert PIXEL_OBS_SHAPE == (336, 1264, 3)


def test_pixel_jits():
    """jax.jit(build_pixel_observation)(state) executes without error."""
    state = _make_state()
    jitted = jax.jit(build_pixel_observation)
    img = jitted(state)
    assert tuple(img.shape) == PIXEL_OBS_SHAPE
    assert img.dtype == jnp.uint8


def test_pixel_non_zero_when_terrain_present():
    """Tile atlas sprites are not all black — rendered image has non-zero pixels.

    build_glyphs is a Wave 1 stub that returns zeros regardless of state,
    so all cells map to glyph 0 -> GLYPH2TILE[0] = tile 0.  The NetHack
    tile atlas tile 0 is a real sprite (not pure black), so the image must
    contain at least one non-zero pixel value.
    """
    from Nethax.nethax.state import EnvState
    from Nethax.nethax.constants import TileType

    state = EnvState.default(_RNG)
    # Set terrain to all FLOOR on the default branch/level (index [0, 0, :, :])
    floor_terrain = jnp.full_like(state.terrain, TileType.FLOOR)
    state = state.replace(terrain=floor_terrain)

    img = build_pixel_observation(state)
    assert jnp.any(img > 0), "Rendered image is entirely black — tile sprites did not load"
