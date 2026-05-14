"""Wave 2 tests for Nethax/nethax/fov.py — Bresenham raycast FOV."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.fov import BLIND_SIGHT_RADIUS, DEFAULT_SIGHT_RADIUS, compute_fov
from Nethax.nethax.constants import TileType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _floor_map(h: int, w: int) -> jnp.ndarray:
    return jnp.full((h, w), TileType.FLOOR, dtype=jnp.int32)


def _pos(r: int, c: int) -> jnp.ndarray:
    return jnp.array([r, c], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_player_sees_self():
    terrain = _floor_map(10, 10)
    pos = _pos(5, 5)
    fov = compute_fov(terrain, pos)
    assert bool(fov[5, 5]), "Player must always see their own tile."


def test_open_room_sees_neighbors():
    """5x5 all-floor grid, player at center: all 24 other tiles visible."""
    terrain = _floor_map(5, 5)
    pos = _pos(2, 2)
    fov = compute_fov(terrain, pos, sight_radius=4)
    assert jnp.all(fov), "In a fully open room the entire grid should be visible."


def test_wall_blocks_los():
    """Wall directly north of player: tiles 2+ steps north are not visible."""
    h, w = 10, 10
    terrain = _floor_map(h, w)
    # Player at row 5, col 5. Wall at row 4, col 5.
    terrain = terrain.at[4, 5].set(TileType.WALL)
    pos = _pos(5, 5)
    fov = compute_fov(terrain, pos)

    # The wall itself should be visible.
    assert bool(fov[4, 5]), "Wall tile directly north should be visible."
    # Tiles further north (rows 0-3 in col 5) should be blocked.
    for r in range(0, 4):
        assert not bool(fov[r, 5]), f"Tile ({r}, 5) behind wall should be hidden."


def test_fov_jits():
    """compute_fov must compile cleanly under jax.jit."""
    terrain = _floor_map(10, 10)
    pos = _pos(5, 5)
    jit_fov = jax.jit(compute_fov)
    result = jit_fov(terrain, pos)
    assert result.shape == (10, 10)
    assert result.dtype == jnp.bool_


def test_blind_sees_only_adjacent():
    """With BLIND_SIGHT_RADIUS=1, only immediately adjacent tiles are visible."""
    terrain = _floor_map(9, 9)
    pos = _pos(4, 4)
    fov = compute_fov(terrain, pos, sight_radius=BLIND_SIGHT_RADIUS)

    # All 9 tiles in the 3x3 neighbourhood (including self) should be visible.
    neighbourhood = fov[3:6, 3:6]
    assert jnp.all(neighbourhood), "3x3 neighbourhood must be visible at radius 1."

    # Tiles at distance 2 should not be visible (e.g. row 2 or row 6).
    far_tiles = jnp.concatenate([fov[2, :], fov[6, :], fov[:, 2], fov[:, 6]])
    assert not jnp.any(far_tiles), "Tiles at distance 2 must be hidden at radius 1."
