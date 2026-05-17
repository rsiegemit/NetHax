"""Parity test: blindness intrinsic reduces FOV to radius 1.

Vendor reference: vendor/nethack/src/vision.c — blindness forces vision radius=1.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.status_effects import TimedStatus
from Nethax.nethax.subsystems.action_dispatch import _apply_fov
from Nethax.nethax.constants import TileType


_RNG = jax.random.PRNGKey(0)


def _make_open_state(blind: bool = False):
    """State on an all-FLOOR map, player at (10, 10), optionally blind."""
    state = EnvState.default(_RNG)
    # Fill level 0 terrain with FLOOR so FOV is unobstructed.
    floor_terrain = jnp.full(
        state.terrain[0, 0].shape, TileType.FLOOR, dtype=state.terrain.dtype
    )
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor_terrain),
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
    )
    if blind:
        ts = state.status.timed_statuses.at[int(TimedStatus.BLIND)].set(jnp.int32(50))
        state = state.replace(status=state.status.replace(timed_statuses=ts))
    return state


def test_normal_sight_radius():
    """Without blindness _apply_fov should expose significantly more than 9 tiles."""
    state = _make_open_state(blind=False)
    new_state = _apply_fov(state)
    visible_count = int(jnp.sum(new_state.visible))
    # Radius-7 open map: all tiles within L∞=7 are visible — (2*7+1)^2 = 225.
    assert visible_count > 9, (
        f"Normal sight expected >9 visible tiles, got {visible_count}."
    )


def test_blind_sight_radius_1():
    """With BLIND timer active _apply_fov must expose at most 3x3 = 9 tiles.

    Vendor: vision.c — blindness forces radius=1 (adjacent tiles only).
    """
    state = _make_open_state(blind=True)
    new_state = _apply_fov(state)
    visible_count = int(jnp.sum(new_state.visible))
    assert visible_count <= 9, (
        f"Blind sight expected <=9 visible tiles (3x3), got {visible_count}."
    )
