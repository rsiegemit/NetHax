"""Wave 2 movement dispatch tests.

Each test constructs an EnvState via EnvState.default(), manually sets terrain
and player_pos, then calls dispatch_action and asserts the result.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants import TileType
from Nethax.nethax.constants.actions import (
    CompassCardinalDirection,
    MiscDirection,
)
from Nethax.nethax.subsystems.action_dispatch import dispatch_action

_RNG = jax.random.PRNGKey(0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(player_row: int, player_col: int) -> EnvState:
    """Return a default EnvState with player at (player_row, player_col)."""
    state = EnvState.default(_RNG)
    return state.replace(
        player_pos=jnp.array([player_row, player_col], dtype=jnp.int16)
    )


def _set_tile(state: EnvState, row: int, col: int, tile: TileType) -> EnvState:
    """Return state with terrain[current_branch, current_level-1, row, col] = tile."""
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1  # 1-based → 0-based
    new_terrain = state.terrain.at[b, lv, row, col].set(jnp.int8(tile))
    return state.replace(terrain=new_terrain)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_move_into_wall_blocks():
    """Player facing north hits a wall — player_pos must be unchanged."""
    state = _make_state(player_row=5, player_col=5)
    # Place a wall one step north (row 4, same col).
    state = _set_tile(state, row=4, col=5, tile=TileType.WALL)

    result = dispatch_action(state, jnp.int32(CompassCardinalDirection.N), _RNG)

    assert jnp.array_equal(result.player_pos, state.player_pos), (
        f"Expected pos {state.player_pos}, got {result.player_pos}"
    )


def test_move_onto_floor_succeeds():
    """Player facing north onto floor — player_pos must update to target."""
    state = _make_state(player_row=5, player_col=5)
    # Place a floor one step north.
    state = _set_tile(state, row=4, col=5, tile=TileType.FLOOR)

    result = dispatch_action(state, jnp.int32(CompassCardinalDirection.N), _RNG)

    expected = jnp.array([4, 5], dtype=jnp.int16)
    assert jnp.array_equal(result.player_pos, expected), (
        f"Expected pos {expected}, got {result.player_pos}"
    )


def test_move_out_of_bounds_blocks():
    """Player at (0, 0) moving north would go to row -1 — must be blocked."""
    state = _make_state(player_row=0, player_col=0)
    # No terrain modifications needed; row -1 is out of bounds.

    result = dispatch_action(state, jnp.int32(CompassCardinalDirection.N), _RNG)

    assert jnp.array_equal(result.player_pos, state.player_pos), (
        f"Expected pos {state.player_pos}, got {result.player_pos}"
    )


def test_wait_is_noop_except_timestep():
    """WAIT action must not change player_pos."""
    state = _make_state(player_row=5, player_col=5)

    result = dispatch_action(state, jnp.int32(MiscDirection.WAIT), _RNG)

    assert jnp.array_equal(result.player_pos, state.player_pos), (
        f"Expected pos {state.player_pos}, got {result.player_pos}"
    )
