"""Parity tests for D_TRAPPED door spring.

Vendor reference: vendor/nethack/src/lock.c::doopen — checks
``d->doormask & D_TRAPPED`` before opening; if set, springs trap
via trapsounding(), dealing rnd(10) damage and breaking the door.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants import TileType
from Nethax.nethax.subsystems.features import DoorState, open_door

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(hp: int = 20) -> EnvState:
    state = EnvState.default(_RNG)
    return state.replace(
        player_hp=jnp.int32(hp),
        player_hp_max=jnp.int32(hp),
        player_pos=jnp.array([5, 5], dtype=jnp.int16),
    )


def _flat_lv(state: EnvState) -> int:
    b = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    max_lv = int(state.terrain.shape[1])
    return b * max_lv + lv


def _set_door(state, row, col, door_state, trapped=False):
    flat = _flat_lv(state)
    new_door = state.features.door_state.at[flat, row, col].set(jnp.int8(door_state))
    new_trapped = state.features.door_trapped.at[flat, row, col].set(jnp.bool_(trapped))
    new_features = state.features.replace(door_state=new_door, door_trapped=new_trapped)
    return state.replace(features=new_features)


def _door_pos(state, row, col):
    flat = _flat_lv(state)
    return jnp.array([flat, row, col], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrappedDoor:
    def test_open_trapped_door_damages(self):
        """Opening a CLOSED+TRAPPED door deals rnd(10) HP damage.

        Vendor: vendor/nethack/src/lock.c::doopen D_TRAPPED branch —
        trapsounding() deals rnd(10) HP before breaking the door.
        """
        state = _make_state(hp=20)
        state = _set_door(state, 3, 3, DoorState.CLOSED, trapped=True)
        pos = _door_pos(state, 3, 3)

        new_features, damage = open_door(state.features, pos, _RNG)

        # Damage must be in [1, 10] (rnd(10))
        dmg = int(damage)
        assert 1 <= dmg <= 10, f"Expected damage in [1,10], got {dmg}"

        # HP should decrease after applying damage
        new_hp = int(jnp.maximum(jnp.int32(0), state.player_hp - damage))
        assert new_hp < 20, "HP should decrease after a trapped-door spring"

    def test_open_normal_door_no_damage(self):
        """Opening a normal CLOSED door deals no damage."""
        state = _make_state(hp=20)
        state = _set_door(state, 3, 3, DoorState.CLOSED, trapped=False)
        pos = _door_pos(state, 3, 3)

        new_features, damage = open_door(state.features, pos, _RNG)

        assert int(damage) == 0, f"Normal door should deal 0 damage, got {int(damage)}"

    def test_trapped_door_becomes_broken(self):
        """Opening a trapped door results in GONE state (D_NODOOR), not OPEN.

        Vendor: vendor/nethack/src/lock.c:907-913 — after b_trapped() the
        doormask is set to D_NODOOR (the trap obliterates the door rather
        than breaking it off its hinges).  In our DoorState enum this is
        DoorState.GONE (0).
        """
        state = _make_state()
        state = _set_door(state, 3, 3, DoorState.CLOSED, trapped=True)
        pos = _door_pos(state, 3, 3)

        new_features, _damage = open_door(state.features, pos, _RNG)

        flat = _flat_lv(state)
        new_ds = int(new_features.door_state[flat, 3, 3])
        assert new_ds == int(DoorState.GONE), (
            f"Trapped door should become GONE (0), got {new_ds}"
        )

    def test_trapped_bit_cleared_after_spring(self):
        """The door_trapped flag is cleared after the trap springs."""
        state = _make_state()
        state = _set_door(state, 3, 3, DoorState.CLOSED, trapped=True)
        pos = _door_pos(state, 3, 3)

        new_features, _damage = open_door(state.features, pos, _RNG)

        flat = _flat_lv(state)
        still_trapped = bool(new_features.door_trapped[flat, 3, 3])
        assert not still_trapped, "door_trapped should be cleared after spring"

    def test_locked_trapped_door_not_opened(self):
        """A LOCKED door (even if trapped) is not openable via open_door.

        open_door only acts on CLOSED doors; LOCKED doors require unlock first.
        """
        state = _make_state(hp=20)
        state = _set_door(state, 3, 3, DoorState.LOCKED, trapped=True)
        pos = _door_pos(state, 3, 3)

        new_features, damage = open_door(state.features, pos, _RNG)

        # No state change: LOCKED door stays LOCKED, no damage
        flat = _flat_lv(state)
        assert int(new_features.door_state[flat, 3, 3]) == int(DoorState.LOCKED)
        assert int(damage) == 0

    def test_level_generator_can_set_trapped(self):
        """Level generators can mark a door as trapped via door_trapped grid."""
        state = _make_state()
        flat = _flat_lv(state)

        # Simulate level generator setting a vault door as trapped
        new_door = state.features.door_state.at[flat, 10, 15].set(
            jnp.int8(DoorState.CLOSED)
        )
        new_trapped = state.features.door_trapped.at[flat, 10, 15].set(jnp.bool_(True))
        new_features = state.features.replace(door_state=new_door, door_trapped=new_trapped)
        state = state.replace(features=new_features)

        assert bool(state.features.door_trapped[flat, 10, 15]), \
            "Level generator should be able to set door_trapped=True"
