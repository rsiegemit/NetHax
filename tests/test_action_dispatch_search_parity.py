"""Parity test: SEARCH action wires through to features.handle_search.

Vendor reference: vendor/nethack/src/detect.c::dosearch0.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.features import DoorState
from Nethax.nethax.subsystems.action_dispatch import dispatch_action
from Nethax.nethax.constants.actions import Command


_RNG = jax.random.PRNGKey(42)
_SEARCH_ACTION = jnp.int32(ord("s"))


def _make_state_with_secret_door(player_row: int = 5, player_col: int = 5):
    """Build a state with a SECRET door immediately east of the player."""
    state = EnvState.default(_RNG)
    state = state.replace(
        player_pos=jnp.array([player_row, player_col], dtype=jnp.int16),
    )
    # Flat level index for branch=0, level=0 is 0.
    new_door_state = state.features.door_state.at[0, player_row, player_col + 1].set(
        jnp.int8(DoorState.SECRET)
    )
    new_features = state.features.replace(door_state=new_door_state)
    return state.replace(features=new_features)


def test_search_calls_features_handle_search():
    """Search action must forward to features.handle_search.

    With a SECRET door adjacent and a fixed rng, the door_state changes
    deterministically — confirming the stub is gone.
    """
    state = _make_state_with_secret_door(player_row=5, player_col=5)

    # Confirm precondition: tile (5, 6) starts as SECRET.
    assert int(state.features.door_state[0, 5, 6]) == int(DoorState.SECRET)

    # Run many seeds; at least one must reveal the door (1/7 chance per attempt).
    revealed = False
    for seed in range(50):
        rng = jax.random.PRNGKey(seed)
        new_state = dispatch_action(state, _SEARCH_ACTION, rng)
        if int(new_state.features.door_state[0, 5, 6]) != int(DoorState.SECRET):
            revealed = True
            break

    assert revealed, (
        "No seed in 0-49 revealed the secret door — "
        "_handle_search may still be a no-op stub."
    )


def test_search_does_not_reveal_non_secret_tiles():
    """Search on a plain floor state must not alter door_state."""
    state = EnvState.default(_RNG).replace(
        player_pos=jnp.array([5, 5], dtype=jnp.int16),
    )
    before = state.features.door_state
    for seed in range(10):
        rng = jax.random.PRNGKey(seed)
        new_state = dispatch_action(state, _SEARCH_ACTION, rng)

    assert jnp.array_equal(before, new_state.features.door_state), (
        "Search modified door_state when no secret doors were adjacent."
    )
