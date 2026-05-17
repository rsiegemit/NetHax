"""Parity test: starting pet is spawned on env.reset().

Vendor reference: vendor/nethack/src/u_init.c::makedog (called from u_init()).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.subsystems.character import get_starting_pet


_RNG = jax.random.PRNGKey(0)


def _reset(role: Role = Role.VALKYRIE):
    env = NethaxEnv()
    state, _ = env.reset(_RNG, role=role)
    return state


def test_pet_exists_after_reset():
    """At least one monster slot must have tame=True after reset."""
    state = _reset()
    assert jnp.any(state.monster_ai.tame), (
        "No tame monster found after reset — starting pet was not spawned."
    )


def test_pet_adjacent_to_player():
    """The tame monster's position must be within Chebyshev distance 1 of player_pos."""
    state = _reset()
    tame_mask = state.monster_ai.tame  # bool[MAX_MONSTERS]
    # Find the first tame slot.
    tame_idx = int(jnp.argmax(tame_mask))
    pet_pos = state.monster_ai.pos[tame_idx]  # int16[2]
    player_pos = state.player_pos             # int16[2]
    delta = jnp.abs(pet_pos.astype(jnp.int32) - player_pos.astype(jnp.int32))
    chebyshev = int(jnp.maximum(delta[0], delta[1]))
    assert chebyshev <= 1, (
        f"Pet pos {tuple(int(x) for x in pet_pos)} is Chebyshev distance "
        f"{chebyshev} from player {tuple(int(x) for x in player_pos)} — expected <=1."
    )


@pytest.mark.parametrize("role,expected_name", [
    (Role.VALKYRIE, "kitten"),
    (Role.WIZARD,   "kitten"),
    (Role.CAVEMAN,  "little dog"),
    (Role.KNIGHT,   "pony"),
    (Role.SAMURAI,  "little dog"),
    (Role.RANGER,   "little dog"),
])
def test_pet_species_matches_role(role, expected_name):
    """The tame monster's entry_idx must match get_starting_pet(role).

    Vendor: role.c petnum field selects kitten, little dog, or pony per role.
    """
    state = _reset(role=role)
    # Resolve expected name → MONSTERS index.
    expected_pm = next(
        (i for i, m in enumerate(MONSTERS) if m.name == expected_name), None
    )
    assert expected_pm is not None, f"Monster '{expected_name}' not found in MONSTERS table."

    tame_mask = state.monster_ai.tame
    tame_idx = int(jnp.argmax(tame_mask))
    actual_pm = int(state.monster_ai.entry_idx[tame_idx])

    assert actual_pm == expected_pm, (
        f"Role {role.name}: expected entry_idx={expected_pm} ({expected_name}), "
        f"got {actual_pm} ({MONSTERS[actual_pm].name})."
    )
