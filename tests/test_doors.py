"""Wave 3 door tests.

Tests cover open_door / close_door / kick_door / unlock_door /
door_blocks_movement and the bump-to-open integration in action_dispatch.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants import TileType
from Nethax.nethax.constants.actions import CompassCardinalDirection
from Nethax.nethax.subsystems.action_dispatch import dispatch_action
from Nethax.nethax.subsystems.features import (
    DoorState,
    open_door,
    close_door,
    kick_door,
    unlock_door,
    door_blocks_movement,
)

_RNG = jax.random.PRNGKey(7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(player_row: int = 5, player_col: int = 5) -> EnvState:
    state = EnvState.default(_RNG)
    return state.replace(
        player_pos=jnp.array([player_row, player_col], dtype=jnp.int16),
    )


def _set_tile(state: EnvState, row: int, col: int, tile: TileType) -> EnvState:
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain.at[b, lv, row, col].set(jnp.int8(tile))
    return state.replace(terrain=new_terrain)


def _flat_lv(state: EnvState) -> int:
    b      = int(state.dungeon.current_branch)
    lv     = int(state.dungeon.current_level) - 1
    max_lv = int(state.terrain.shape[1])
    return b * max_lv + lv


def _set_door_state(state: EnvState, row: int, col: int, ds: DoorState) -> EnvState:
    flat = _flat_lv(state)
    new_door = state.features.door_state.at[flat, row, col].set(jnp.int8(ds))
    new_features = state.features.replace(door_state=new_door)
    return state.replace(features=new_features)


def _get_door_state(state: EnvState, row: int, col: int) -> int:
    flat = _flat_lv(state)
    return int(state.features.door_state[flat, row, col])


def _door_pos(state: EnvState, row: int, col: int) -> jnp.ndarray:
    flat = _flat_lv(state)
    return jnp.array([flat, row, col], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Unit tests for open_door
# ---------------------------------------------------------------------------

class TestOpenDoor:
    def test_closed_becomes_open(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.CLOSED)
        pos = _door_pos(state, 3, 3)
        new_features, _dmg = open_door(state.features, pos)
        flat = _flat_lv(state)
        assert int(new_features.door_state[flat, 3, 3]) == DoorState.OPEN

    def test_locked_unchanged(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.LOCKED)
        pos = _door_pos(state, 3, 3)
        new_features, _dmg = open_door(state.features, pos)
        flat = _flat_lv(state)
        assert int(new_features.door_state[flat, 3, 3]) == DoorState.LOCKED

    def test_already_open_unchanged(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.OPEN)
        pos = _door_pos(state, 3, 3)
        new_features, _dmg = open_door(state.features, pos)
        flat = _flat_lv(state)
        assert int(new_features.door_state[flat, 3, 3]) == DoorState.OPEN

    def test_broken_unchanged(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.BROKEN)
        pos = _door_pos(state, 3, 3)
        new_features, _dmg = open_door(state.features, pos)
        flat = _flat_lv(state)
        assert int(new_features.door_state[flat, 3, 3]) == DoorState.BROKEN


# ---------------------------------------------------------------------------
# Unit tests for close_door
# ---------------------------------------------------------------------------

class TestCloseDoor:
    def test_open_becomes_closed(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.OPEN)
        pos = _door_pos(state, 3, 3)
        new_features = close_door(state.features, pos)
        flat = _flat_lv(state)
        assert int(new_features.door_state[flat, 3, 3]) == DoorState.CLOSED

    def test_closed_unchanged(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.CLOSED)
        pos = _door_pos(state, 3, 3)
        new_features = close_door(state.features, pos)
        flat = _flat_lv(state)
        assert int(new_features.door_state[flat, 3, 3]) == DoorState.CLOSED

    def test_locked_unchanged(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.LOCKED)
        pos = _door_pos(state, 3, 3)
        new_features = close_door(state.features, pos)
        flat = _flat_lv(state)
        assert int(new_features.door_state[flat, 3, 3]) == DoorState.LOCKED


# ---------------------------------------------------------------------------
# Unit tests for kick_door
# ---------------------------------------------------------------------------

class TestKickDoor:
    def test_returns_two_values(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.CLOSED)
        pos = _door_pos(state, 3, 3)
        result = kick_door(state.features, _RNG, pos)
        assert len(result) == 2

    def test_result_is_broken_or_unchanged(self):
        """After kicking, door is BROKEN or still CLOSED — never another state."""
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.CLOSED)
        pos = _door_pos(state, 3, 3)
        new_features, dmg = kick_door(state.features, _RNG, pos)
        flat = _flat_lv(state)
        result_state = int(new_features.door_state[flat, 3, 3])
        assert result_state in (DoorState.BROKEN, DoorState.CLOSED), (
            f"Unexpected door state after kick: {result_state}"
        )

    def test_self_damage_zero(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.CLOSED)
        pos = _door_pos(state, 3, 3)
        _, dmg = kick_door(state.features, _RNG, pos)
        assert int(dmg) == 0


# ---------------------------------------------------------------------------
# Unit tests for unlock_door
# ---------------------------------------------------------------------------

class TestUnlockDoor:
    def test_locked_becomes_closed(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.LOCKED)
        pos = _door_pos(state, 3, 3)
        new_features, success = unlock_door(state.features, _RNG, pos, jnp.int32(0))
        flat = _flat_lv(state)
        assert int(new_features.door_state[flat, 3, 3]) == DoorState.CLOSED
        assert bool(success)

    def test_closed_unchanged_on_unlock(self):
        """Attempting to unlock an already-closed door returns unchanged state."""
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.CLOSED)
        pos = _door_pos(state, 3, 3)
        new_features, success = unlock_door(state.features, _RNG, pos, jnp.int32(0))
        flat = _flat_lv(state)
        assert int(new_features.door_state[flat, 3, 3]) == DoorState.CLOSED
        assert not bool(success)


# ---------------------------------------------------------------------------
# Unit tests for door_blocks_movement
# ---------------------------------------------------------------------------

class TestDoorBlocksMovement:
    def test_closed_blocks(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.CLOSED)
        pos = _door_pos(state, 3, 3)
        assert bool(door_blocks_movement(state.features, pos))

    def test_locked_blocks(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.LOCKED)
        pos = _door_pos(state, 3, 3)
        assert bool(door_blocks_movement(state.features, pos))

    def test_secret_blocks(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.SECRET)
        pos = _door_pos(state, 3, 3)
        assert bool(door_blocks_movement(state.features, pos))

    def test_open_allows(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.OPEN)
        pos = _door_pos(state, 3, 3)
        assert not bool(door_blocks_movement(state.features, pos))

    def test_broken_allows(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.BROKEN)
        pos = _door_pos(state, 3, 3)
        assert not bool(door_blocks_movement(state.features, pos))

    def test_gone_allows(self):
        state = _make_state()
        state = _set_door_state(state, 3, 3, DoorState.GONE)
        pos = _door_pos(state, 3, 3)
        assert not bool(door_blocks_movement(state.features, pos))


# ---------------------------------------------------------------------------
# Integration tests: bump-to-open via dispatch_action
# ---------------------------------------------------------------------------

class TestBumpDoor:
    def test_bump_closed_door_opens_it(self):
        """Bumping an unlocked CLOSED_DOOR: door becomes OPEN."""
        state = _make_state(player_row=5, player_col=5)
        state = _set_tile(state, 4, 5, TileType.CLOSED_DOOR)
        # door_state defaults to 0 (GONE), set it to CLOSED for the features layer.
        state = _set_door_state(state, 4, 5, DoorState.CLOSED)

        result = dispatch_action(state, jnp.int32(CompassCardinalDirection.N), _RNG)

        assert _get_door_state(result, 4, 5) == DoorState.OPEN, (
            f"Expected OPEN after bump, got {_get_door_state(result, 4, 5)}"
        )

    def test_bump_closed_door_player_pos_unchanged(self):
        """Bumping an unlocked CLOSED_DOOR: player does NOT move this turn."""
        state = _make_state(player_row=5, player_col=5)
        state = _set_tile(state, 4, 5, TileType.CLOSED_DOOR)
        state = _set_door_state(state, 4, 5, DoorState.CLOSED)

        result = dispatch_action(state, jnp.int32(CompassCardinalDirection.N), _RNG)

        assert jnp.array_equal(result.player_pos, jnp.array([5, 5], dtype=jnp.int16)), (
            f"Player should stay at (5,5), got {result.player_pos}"
        )

    def test_bump_locked_door_player_pos_unchanged(self):
        """Bumping a LOCKED door: player blocked, door state unchanged."""
        state = _make_state(player_row=5, player_col=5)
        state = _set_tile(state, 4, 5, TileType.CLOSED_DOOR)
        state = _set_door_state(state, 4, 5, DoorState.LOCKED)

        result = dispatch_action(state, jnp.int32(CompassCardinalDirection.N), _RNG)

        assert jnp.array_equal(result.player_pos, jnp.array([5, 5], dtype=jnp.int16)), (
            f"Player should stay at (5,5), got {result.player_pos}"
        )
        assert _get_door_state(result, 4, 5) == DoorState.LOCKED, (
            "Locked door must remain LOCKED after bump"
        )

    def test_walk_through_open_door(self):
        """Walking toward an OPEN_DOOR tile: player advances through."""
        state = _make_state(player_row=5, player_col=5)
        state = _set_tile(state, 4, 5, TileType.OPEN_DOOR)
        state = _set_door_state(state, 4, 5, DoorState.OPEN)

        result = dispatch_action(state, jnp.int32(CompassCardinalDirection.N), _RNG)

        assert jnp.array_equal(result.player_pos, jnp.array([4, 5], dtype=jnp.int16)), (
            f"Player should move to (4,5) through open door, got {result.player_pos}"
        )

    def test_bump_closed_door_terrain_updated_to_open(self):
        """After bump-open, the terrain tile must be updated to OPEN_DOOR."""
        state = _make_state(player_row=5, player_col=5)
        state = _set_tile(state, 4, 5, TileType.CLOSED_DOOR)
        state = _set_door_state(state, 4, 5, DoorState.CLOSED)

        result = dispatch_action(state, jnp.int32(CompassCardinalDirection.N), _RNG)

        b  = int(result.dungeon.current_branch)
        lv = int(result.dungeon.current_level) - 1
        tile = int(result.terrain[b, lv, 4, 5])
        assert tile == int(TileType.OPEN_DOOR), (
            f"Terrain tile must be OPEN_DOOR after bump-open, got {tile}"
        )
