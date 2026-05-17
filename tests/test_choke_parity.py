"""Choke-on-overeating parity tests.

Vendor reference: vendor/nethack/src/eat.c lines 1980-2020.
When the player is SATIATED and eats, there is a 1-in-3 chance of choking
to death (vendor line 1990: ``!rn2(3)``).  Ring of slow digestion suppresses
it (vendor line 1985).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.action_dispatch import _handle_eat
from Nethax.nethax.subsystems.inventory import ItemCategory
from Nethax.nethax.subsystems.scoring import DeathCause
from Nethax.nethax.subsystems.status_effects import HungerState, Intrinsic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_with_food(hunger_state: HungerState, slow_digestion: bool = False) -> EnvState:
    """Return an EnvState with one food-ration in slot 0 and the given hunger."""
    base = EnvState.default(jax.random.PRNGKey(0))

    # Plant a FOOD item (weight=800 = food-ration nutrition) in slot 0.
    inv = base.inventory
    new_items = inv.items.replace(
        category=inv.items.category.at[0].set(jnp.int8(int(ItemCategory.FOOD))),
        quantity=inv.items.quantity.at[0].set(jnp.int16(5)),
        weight=inv.items.weight.at[0].set(jnp.int32(800)),
    )
    base = base.replace(inventory=inv.replace(items=new_items))

    # Force the desired hunger state.
    new_status = base.status.replace(hunger_state=jnp.int8(int(hunger_state)))

    # Optionally grant SLOW_DIGESTION intrinsic.
    if slow_digestion:
        new_intrinsics = new_status.intrinsics.at[Intrinsic.SLOW_DIGESTION].set(True)
        new_status = new_status.replace(intrinsics=new_intrinsics)

    return base.replace(status=new_status)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_choke_at_satiated():
    """~1/3 of satiated eat-attempts choke (done=True, hp=0).

    Vendor eat.c line 1990: ``if (!rn2(3)) { ... choke ... }``.
    """
    state = _state_with_food(HungerState.SATIATED)
    n_trials = 100
    choke_count = 0
    for i in range(n_trials):
        rng = jax.random.PRNGKey(i)
        result = _handle_eat(state, rng)
        if bool(result.done):
            choke_count += 1
            assert int(result.player_hp) == 0, "choked player should have 0 hp"

    # Vendor probability is exactly 1/3.  Allow generous window for 100 trials.
    rate = choke_count / n_trials
    assert 0.15 <= rate <= 0.55, (
        f"Choke rate {rate:.2f} far from expected ~0.33 over {n_trials} trials"
    )


def test_no_choke_when_hungry():
    """A hungry player never chokes regardless of RNG.

    Vendor eat.c line 1980: choke check is gated on ``u.uhs == SATIATED``.
    """
    state = _state_with_food(HungerState.HUNGRY)
    for i in range(100):
        rng = jax.random.PRNGKey(i)
        result = _handle_eat(state, rng)
        assert not bool(result.done), f"seed {i}: hungry player should never choke"
        assert int(result.player_hp) > 0


def test_slow_digestion_no_choke():
    """SLOW_DIGESTION intrinsic completely suppresses choking.

    Vendor eat.c line 1985: ``if (Slow_digestion) ... (no choke)``.
    """
    state = _state_with_food(HungerState.SATIATED, slow_digestion=True)
    for i in range(100):
        rng = jax.random.PRNGKey(i)
        result = _handle_eat(state, rng)
        assert not bool(result.done), (
            f"seed {i}: SLOW_DIGESTION holder should never choke"
        )
        assert int(result.player_hp) > 0


def test_choke_records_choking_cause():
    """After a choke, scoring.death_cause must be DeathCause.CHOKING.

    Vendor eat.c ~line 2005: ``killer.name = \"choked\"`` / CHOKING cause.
    """
    state = _state_with_food(HungerState.SATIATED)
    # Find a seed that causes a choke.
    choke_result = None
    for i in range(200):
        rng = jax.random.PRNGKey(i)
        result = _handle_eat(state, rng)
        if bool(result.done):
            choke_result = result
            break

    assert choke_result is not None, "No choke found in 200 trials — unexpected"
    assert int(choke_result.scoring.death_cause) == int(DeathCause.CHOKING), (
        f"Expected death_cause={int(DeathCause.CHOKING)} (CHOKING), "
        f"got {int(choke_result.scoring.death_cause)}"
    )
