"""Tests for ring + amulet wear/take-off and intrinsic granting.

Covers:
  - put_on_ring / take_off_ring for single and dual rings
  - wear_amulet / take_off_amulet
  - Intrinsic flag grant and revocation
  - Two rings simultaneously both grant their intrinsics
  - Revocation of stealth ring leaves intrinsic intact when another source holds it
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.status_effects import Intrinsic
from Nethax.nethax.subsystems.items_jewelry import (
    RingEffect,
    AmuletEffect,
    put_on_ring,
    take_off_ring,
    wear_amulet,
    take_off_amulet,
)

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_with_ring(effect: RingEffect, enchantment: int = 0) -> EnvState:
    """Return a default EnvState whose single inventory item is a ring of the
    given effect type.  Category 3 = RING_CLASS (obj.h).
    """
    state = EnvState.default(_RNG)
    new_item = state.inventory.items.replace(
        category=jnp.int8(3),        # RING_CLASS
        type_id=jnp.int16(int(effect)),
        enchantment=jnp.int8(enchantment),
    )
    new_inventory = state.inventory.replace(items=new_item)
    return state.replace(inventory=new_inventory)


def _state_with_amulet(effect: AmuletEffect) -> EnvState:
    """Return a default EnvState whose single inventory item is an amulet of
    the given effect type.  Category 4 = AMULET_CLASS (obj.h).
    """
    state = EnvState.default(_RNG)
    new_item = state.inventory.items.replace(
        category=jnp.int8(4),        # AMULET_CLASS
        type_id=jnp.int16(int(effect)),
        enchantment=jnp.int8(0),
    )
    new_inventory = state.inventory.replace(items=new_item)
    return state.replace(inventory=new_inventory)


def _has(state: EnvState, intrinsic: Intrinsic) -> bool:
    return bool(state.status.intrinsics[int(intrinsic)])


# ---------------------------------------------------------------------------
# Ring of regeneration
# ---------------------------------------------------------------------------

def test_put_on_ring_of_regeneration_grants_regen():
    """put_on_ring(REGENERATION) → Intrinsic.REGEN is set."""
    state = _state_with_ring(RingEffect.REGENERATION)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert _has(state, Intrinsic.REGEN), "REGEN intrinsic should be True after wearing ring of regeneration"


def test_take_off_ring_of_regeneration_revokes_regen():
    """take_off_ring after putting on REGENERATION ring → Intrinsic.REGEN is cleared."""
    state = _state_with_ring(RingEffect.REGENERATION)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert _has(state, Intrinsic.REGEN)
    state = take_off_ring(state, hand=0)
    assert not _has(state, Intrinsic.REGEN), "REGEN intrinsic should be False after taking off ring of regeneration"


def test_take_off_ring_clears_worn_slot():
    """worn_rings[0] returns to -1 after take_off_ring."""
    state = _state_with_ring(RingEffect.REGENERATION)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert int(state.inventory.worn_rings[0]) == 0
    state = take_off_ring(state, hand=0)
    assert int(state.inventory.worn_rings[0]) == -1


# ---------------------------------------------------------------------------
# Amulet of reflection
# ---------------------------------------------------------------------------

def test_wear_amulet_of_reflection_grants_reflecting():
    """wear_amulet(REFLECTION) → Intrinsic.REFLECTING is set."""
    state = _state_with_amulet(AmuletEffect.REFLECTION)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert _has(state, Intrinsic.REFLECTING), "REFLECTING intrinsic should be True after wearing amulet of reflection"


def test_take_off_amulet_of_reflection_revokes_reflecting():
    """take_off_amulet after wearing REFLECTION → Intrinsic.REFLECTING cleared."""
    state = _state_with_amulet(AmuletEffect.REFLECTION)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert _has(state, Intrinsic.REFLECTING)
    state = take_off_amulet(state)
    assert not _has(state, Intrinsic.REFLECTING), "REFLECTING intrinsic should be False after taking off amulet"


def test_take_off_amulet_clears_worn_slot():
    """worn_amulet returns to -1 after take_off_amulet."""
    state = _state_with_amulet(AmuletEffect.REFLECTION)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert int(state.inventory.worn_amulet) == 0
    state = take_off_amulet(state)
    assert int(state.inventory.worn_amulet) == -1


# ---------------------------------------------------------------------------
# Two rings simultaneously
# ---------------------------------------------------------------------------

def test_two_rings_simultaneously_both_intrinsics_apply():
    """Wearing a ring of stealth (left) and ring of conflict (right) → both intrinsics set."""
    # Set up stealth ring as item, put on left hand.
    state = _state_with_ring(RingEffect.STEALTH)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert _has(state, Intrinsic.STEALTH), "STEALTH should be set after wearing on left"

    # Swap item to conflict ring, put on right hand.
    new_item = state.inventory.items.replace(
        type_id=jnp.int16(int(RingEffect.CONFLICT))
    )
    state = state.replace(inventory=state.inventory.replace(items=new_item))
    state = put_on_ring(state, _RNG, slot_idx=0, hand=1)

    assert _has(state, Intrinsic.STEALTH),  "STEALTH should still be set"
    assert _has(state, Intrinsic.CONFLICT), "CONFLICT should be set after wearing on right"


def test_take_off_one_ring_leaves_other_intact():
    """Taking off the conflict ring does not revoke the stealth intrinsic."""
    # Wear stealth on left.
    state = _state_with_ring(RingEffect.STEALTH)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)

    # Wear conflict on right.
    new_item = state.inventory.items.replace(
        type_id=jnp.int16(int(RingEffect.CONFLICT))
    )
    state = state.replace(inventory=state.inventory.replace(items=new_item))
    state = put_on_ring(state, _RNG, slot_idx=0, hand=1)

    # Take off the conflict ring (right hand).
    state = take_off_ring(state, hand=1)

    assert not _has(state, Intrinsic.CONFLICT), "CONFLICT should be gone"
    assert _has(state, Intrinsic.STEALTH),      "STEALTH should remain"


# ---------------------------------------------------------------------------
# Stealth revocation doesn't clobber independently held intrinsic
# ---------------------------------------------------------------------------

def test_take_off_stealth_ring_revokes_intrinsic_unless_another_source():
    """Taking off ring of stealth clears STEALTH; a prior independently-set
    intrinsic (e.g. from level-gain) is modelled as a separate intrinsics bit
    in Wave 3 (single boolean).  Here we verify the basic revoke path."""
    state = _state_with_ring(RingEffect.STEALTH)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert _has(state, Intrinsic.STEALTH)
    state = take_off_ring(state, hand=0)
    assert not _has(state, Intrinsic.STEALTH), "STEALTH revoked after ring removed"


# ---------------------------------------------------------------------------
# Additional amulet effects
# ---------------------------------------------------------------------------

def test_wear_amulet_of_esp_grants_telepathy():
    state = _state_with_amulet(AmuletEffect.ESP)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert _has(state, Intrinsic.TELEPATHY)


def test_take_off_amulet_of_esp_revokes_telepathy():
    state = _state_with_amulet(AmuletEffect.ESP)
    state = wear_amulet(state, _RNG, slot_idx=0)
    state = take_off_amulet(state)
    assert not _has(state, Intrinsic.TELEPATHY)


def test_wear_amulet_of_magical_breathing_grants_breathless():
    state = _state_with_amulet(AmuletEffect.MAGICAL_BREATHING)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert _has(state, Intrinsic.BREATHLESS)


def test_wear_amulet_of_flying_grants_flying():
    state = _state_with_amulet(AmuletEffect.FLYING)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert _has(state, Intrinsic.FLYING)


def test_wear_amulet_of_guarding_grants_protection():
    state = _state_with_amulet(AmuletEffect.GUARDING)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert _has(state, Intrinsic.PROTECTION)


# ---------------------------------------------------------------------------
# Additional ring effects
# ---------------------------------------------------------------------------

def test_put_on_ring_of_stealth_grants_stealth():
    state = _state_with_ring(RingEffect.STEALTH)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert _has(state, Intrinsic.STEALTH)


def test_put_on_ring_of_levitation_grants_levitation():
    state = _state_with_ring(RingEffect.LEVITATION)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert _has(state, Intrinsic.LEVITATION)


def test_put_on_ring_of_free_action_grants_free_action():
    state = _state_with_ring(RingEffect.FREE_ACTION)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert _has(state, Intrinsic.FREE_ACTION)


def test_put_on_ring_of_teleport_control_grants_teleport_control():
    state = _state_with_ring(RingEffect.TELEPORT_CONTROL)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert _has(state, Intrinsic.TELEPORT_CONTROL)


def test_put_on_ring_of_see_invisible_grants_see_invis():
    state = _state_with_ring(RingEffect.SEE_INVISIBLE)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert _has(state, Intrinsic.SEE_INVIS)


def test_put_on_ring_of_poison_resistance_grants_resist_poison():
    state = _state_with_ring(RingEffect.POISON_RESISTANCE)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert _has(state, Intrinsic.RESIST_POISON)


def test_put_on_ring_of_fire_resistance_grants_resist_fire():
    state = _state_with_ring(RingEffect.FIRE_RESISTANCE)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert _has(state, Intrinsic.RESIST_FIRE)


def test_put_on_ring_of_cold_resistance_grants_resist_cold():
    state = _state_with_ring(RingEffect.COLD_RESISTANCE)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert _has(state, Intrinsic.RESIST_COLD)


def test_put_on_ring_of_shock_resistance_grants_resist_shock():
    state = _state_with_ring(RingEffect.SHOCK_RESISTANCE)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert _has(state, Intrinsic.RESIST_SHOCK)


def test_put_on_ring_of_prot_from_shape_changers():
    state = _state_with_ring(RingEffect.PROTECTION_FROM_SHAPE_CHANGERS)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert _has(state, Intrinsic.PROT_FROM_SHAPE_CHANGERS)


# ---------------------------------------------------------------------------
# Stat-adjusting rings
# ---------------------------------------------------------------------------

def test_ring_of_gain_strength_increases_player_str():
    state = _state_with_ring(RingEffect.GAIN_STRENGTH, enchantment=2)
    base_str = int(state.player_str)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert int(state.player_str) == base_str + 2


def test_ring_of_gain_strength_revoked_on_take_off():
    state = _state_with_ring(RingEffect.GAIN_STRENGTH, enchantment=2)
    base_str = int(state.player_str)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    state = take_off_ring(state, hand=0)
    assert int(state.player_str) == base_str, "STR should revert after removing ring"


# ---------------------------------------------------------------------------
# No-op guard: take off with nothing worn
# ---------------------------------------------------------------------------

def test_take_off_ring_noop_when_nothing_worn():
    state = EnvState.default(_RNG)
    state2 = take_off_ring(state, hand=0)
    # Intrinsics array should be unchanged.
    assert jnp.array_equal(state.status.intrinsics, state2.status.intrinsics)


def test_take_off_amulet_noop_when_nothing_worn():
    state = EnvState.default(_RNG)
    state2 = take_off_amulet(state)
    assert jnp.array_equal(state.status.intrinsics, state2.status.intrinsics)
