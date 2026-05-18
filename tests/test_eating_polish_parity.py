"""Eating polish parity tests — Wave 7 additions.

Covers:
  1. Corpse age tracking via corpse_creation_turn field (eat.c:1885)
  2. Fresh corpse does not trigger vomiting
  3. Tin of spinach grants +1 STR (eat.c:1684)
  4. Tin of newt grants pw_max bump (eat.c:1311 eye_of_newt_buzz)
  5. Poisoned tin deals HP damage (eat.c:1537)
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.inventory import make_item, ItemCategory
from Nethax.nethax.subsystems.items_corpses import (
    apply_old_corpse_effects,
    apply_eattin,
    _NEWT_IDX_NP,
    _CORPSE_AGE_THRESHOLD,
)
from Nethax.nethax.subsystems.status_effects import TimedStatus


def _base_state(hp: int = 50, str_val: int = 10, pw_max: int = 10) -> EnvState:
    state = EnvState.default(jax.random.PRNGKey(0))
    return state.replace(
        player_hp=jnp.int32(hp),
        player_hp_max=jnp.int32(hp),
        player_str=jnp.int16(str_val),
        player_pw_max=jnp.int32(pw_max),
    )


# ---------------------------------------------------------------------------
# 1. test_old_corpse_uses_age_tracking
#    Corpse created at turn 0, eaten at turn 100 → age = 100 > 50 → vomits.
#    cite: vendor/nethack/src/eat.c::eatcorpse line 1885-1916
# ---------------------------------------------------------------------------

def test_old_corpse_uses_age_tracking():
    """Age = timestep - corpse_creation_turn > 50 → VOMITING + SICK timers set."""
    state = _base_state()
    # Set timestep = 100, corpse_creation_turn = 0 → age = 100 > 50
    state = state.replace(timestep=jnp.int32(100))

    corpse_creation_turn = jnp.int32(0)
    age = state.timestep - corpse_creation_turn
    is_old = age > jnp.int32(_CORPSE_AGE_THRESHOLD)

    result = apply_old_corpse_effects(state, jax.random.PRNGKey(0), is_old)

    assert bool(is_old), "age should be > threshold for this test"
    vomit_timer = int(result.status.timed_statuses[int(TimedStatus.VOMITING)])
    sick_timer = int(result.status.timed_statuses[int(TimedStatus.SICK)])
    assert vomit_timer > 0, f"Expected VOMITING timer > 0, got {vomit_timer}"
    assert sick_timer > 0, f"Expected SICK timer > 0, got {sick_timer}"
    assert int(result.status.sick_kind) == 1, "sick_kind should be 1 (food poisoning)"


# ---------------------------------------------------------------------------
# 2. test_fresh_corpse_no_vomit
#    Corpse created at turn 95, eaten at turn 100 → age = 5 ≤ 50 → no vomit.
#    cite: vendor/nethack/src/eat.c::eatcorpse line 1895 rotted > 5L check
# ---------------------------------------------------------------------------

def test_fresh_corpse_no_vomit():
    """Age = 5 ≤ 50 → VOMITING and SICK timers stay at 0."""
    state = _base_state()
    state = state.replace(timestep=jnp.int32(100))

    corpse_creation_turn = jnp.int32(95)
    age = state.timestep - corpse_creation_turn
    is_old = age > jnp.int32(_CORPSE_AGE_THRESHOLD)

    result = apply_old_corpse_effects(state, jax.random.PRNGKey(0), is_old)

    assert not bool(is_old), "age should be ≤ threshold for this test"
    vomit_timer = int(result.status.timed_statuses[int(TimedStatus.VOMITING)])
    sick_timer = int(result.status.timed_statuses[int(TimedStatus.SICK)])
    assert vomit_timer == 0, f"Expected no VOMITING, got {vomit_timer}"
    assert sick_timer == 0, f"Expected no SICK, got {sick_timer}"


# ---------------------------------------------------------------------------
# 3. test_tin_of_spinach_str_bonus
#    Opening a spinach tin (enchantment==1) gives +1 STR.
#    cite: vendor/nethack/src/eat.c::consume_tin line 1684 gainstr()
# ---------------------------------------------------------------------------

def test_tin_of_spinach_str_bonus():
    """Spinach tin (enchantment=1) raises player_str by 1."""
    state = _base_state(str_val=10)
    # enchantment=1 is the spinach marker (vendor spe==1, eat.c:1470)
    item = make_item(
        category=int(ItemCategory.FOOD),
        type_id=42,
        enchantment=1,   # spinach marker
        corpse_entry_idx=-1,
    )
    result = apply_eattin(state, jax.random.PRNGKey(0), item)
    assert int(result.player_str) == 11, (
        f"Expected str 11 after spinach tin, got {int(result.player_str)}"
    )


def test_tin_of_spinach_str_clamp():
    """Spinach tin does not raise STR above 18."""
    state = _base_state(str_val=18)
    item = make_item(
        category=int(ItemCategory.FOOD),
        type_id=42,
        enchantment=1,
        corpse_entry_idx=-1,
    )
    result = apply_eattin(state, jax.random.PRNGKey(0), item)
    assert int(result.player_str) == 18, (
        f"Expected str clamped at 18, got {int(result.player_str)}"
    )


# ---------------------------------------------------------------------------
# 4. test_tin_of_newt_grants_pw
#    Opening a tin with corpse_entry_idx=newt may bump pw_max.
#    cite: vendor/nethack/src/eat.c:1311 eye_of_newt_buzz — 1/3 chance pw_max+1
# ---------------------------------------------------------------------------

def test_tin_of_newt_grants_pw():
    """Opening a newt tin sometimes bumps pw_max (1/3 chance)."""
    state = _base_state(pw_max=10)
    item = make_item(
        category=int(ItemCategory.FOOD),
        type_id=42,
        enchantment=0,
        corpse_entry_idx=_NEWT_IDX_NP,
    )
    # Run multiple seeds; at least one should hit the 1/3 pw_max bump.
    bumped = any(
        int(apply_eattin(state, jax.random.PRNGKey(i), item).player_pw_max) > 10
        for i in range(30)
    )
    assert bumped, "Expected at least one pw_max bump across 30 newt-tin trials"


# ---------------------------------------------------------------------------
# 5. test_poisoned_tin_damages
#    A poisoned tin deals 1-15 HP damage and sets SICK timer.
#    cite: vendor/nethack/src/eat.c::consume_tin line 1537 tin->otrapped check
# ---------------------------------------------------------------------------

def test_poisoned_tin_damages():
    """tin_poisoned=True reduces HP by 1-15 and sets SICK timer."""
    state = _base_state(hp=50)
    item = make_item(
        category=int(ItemCategory.FOOD),
        type_id=42,
        enchantment=0,
        corpse_entry_idx=-1,
        tin_poisoned=True,
    )
    results = [
        apply_eattin(state, jax.random.PRNGKey(i), item)
        for i in range(20)
    ]
    hp_losses = [50 - int(r.player_hp) for r in results]
    assert all(1 <= d <= 15 for d in hp_losses), (
        f"HP losses should be in [1,15], got {hp_losses}"
    )
    assert all(
        int(r.status.timed_statuses[int(TimedStatus.SICK)]) > 0
        for r in results
    ), "SICK timer should be set for poisoned tin"
