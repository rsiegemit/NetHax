"""Parity tests for the four previously-stubbed potion effects.

Covers:
  - _effect_restore_ability   (potion.c::peffect_restore_ability)
  - _effect_object_detection  (potion.c::peffect_object_detection)
  - _effect_oil               (potion.c::peffect_oil)
  - _effect_gain_level        (potion.c::peffect_gain_level, cursed branch)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.inventory import (
    InventoryState,
    make_item,
    ItemCategory,
)
from Nethax.nethax.subsystems.items_potions import (
    PotionEffect,
    _POTION_BASE_ID,
    quaff_potion,
)

_RNG = jax.random.PRNGKey(0)

_BUC_UNCURSED = 2
_BUC_CURSED   = 1
_BUC_BLESSED  = 3


def _base_state(**overrides) -> EnvState:
    """EnvState with sensible defaults; keyword args forwarded to replace()."""
    s = EnvState.default(_RNG)
    if overrides:
        s = s.replace(**overrides)
    return s


def _state_with_potion(effect: PotionEffect, buc: int = _BUC_UNCURSED,
                       **state_kwargs) -> EnvState:
    type_id = _POTION_BASE_ID + int(effect)
    item = make_item(
        category=int(ItemCategory.POTION),
        type_id=type_id,
        quantity=1,
        buc_status=buc,
    )
    inv = InventoryState.from_items([item])
    s = _base_state(**state_kwargs)
    return s.replace(inventory=inv)


# ---------------------------------------------------------------------------
# test_restore_ability_caps_all_to_max
# ---------------------------------------------------------------------------

def test_restore_ability_caps_all_to_max():
    """All 6 stats below 18 are raised to 18; stats already ≥18 are unchanged."""
    # Set all stats to low values.
    s = _state_with_potion(PotionEffect.RESTORE_ABILITY)
    s = s.replace(
        player_str=jnp.int16(5),
        player_dex=jnp.int8(6),
        player_con=jnp.int8(7),
        player_int=jnp.int8(8),
        player_wis=jnp.int8(9),
        player_cha=jnp.int8(10),
    )
    out = quaff_potion(s, _RNG, 0)
    assert int(out.player_str) == 18
    assert int(out.player_dex) == 18
    assert int(out.player_con) == 18
    assert int(out.player_int) == 18
    assert int(out.player_wis) == 18
    assert int(out.player_cha) == 18


def test_restore_ability_leaves_high_stats_unchanged():
    """Stats already above 18 are not reduced."""
    s = _state_with_potion(PotionEffect.RESTORE_ABILITY)
    s = s.replace(
        player_str=jnp.int16(20),
        player_dex=jnp.int8(20),
        player_con=jnp.int8(20),
        player_int=jnp.int8(20),
        player_wis=jnp.int8(20),
        player_cha=jnp.int8(20),
    )
    out = quaff_potion(s, _RNG, 0)
    assert int(out.player_str) == 20
    assert int(out.player_dex) == 20


# ---------------------------------------------------------------------------
# test_object_detection_sets_timer
# ---------------------------------------------------------------------------

def test_object_detection_sets_timer():
    """Quaffing object detection sets detect_objects_until_turn = timestep+100."""
    s = _state_with_potion(PotionEffect.OBJECT_DETECTION)
    s = s.replace(timestep=jnp.int32(50))
    out = quaff_potion(s, _RNG, 0)
    assert int(out.identification.detect_objects_until_turn) == 150


def test_object_detection_timer_at_zero_timestep():
    """Timer is 100 when timestep is 0."""
    s = _state_with_potion(PotionEffect.OBJECT_DETECTION)
    out = quaff_potion(s, _RNG, 0)
    assert int(out.identification.detect_objects_until_turn) == 100


# ---------------------------------------------------------------------------
# test_oil_greases_weapon
# ---------------------------------------------------------------------------

def test_oil_greases_weapon():
    """Quaffing oil with a wielded (non-greased) weapon sets greased=True."""
    from Nethax.nethax.subsystems.inventory import make_item, ItemCategory
    weapon = make_item(
        category=int(ItemCategory.WEAPON),
        type_id=1,
        quantity=1,
        buc_status=_BUC_UNCURSED,
    )
    potion = make_item(
        category=int(ItemCategory.POTION),
        type_id=_POTION_BASE_ID + int(PotionEffect.OIL),
        quantity=1,
        buc_status=_BUC_UNCURSED,
    )
    inv = InventoryState.from_items([weapon, potion])
    # Wield slot 0 (weapon).
    inv = inv.replace(wielded=jnp.int8(0))
    s = _base_state().replace(inventory=inv)
    out = quaff_potion(s, _RNG, 1)   # slot 1 = potion
    assert bool(out.inventory.items.greased[0]), "wielded weapon should be greased"


def test_oil_no_weapon_noop():
    """Quaffing oil with no wielded weapon leaves state (hp) unchanged."""
    s = _state_with_potion(PotionEffect.OIL, player_hp=jnp.int32(10),
                           player_hp_max=jnp.int32(10))
    # wielded is -1 by default in a fresh InventoryState.
    out = quaff_potion(s, _RNG, 0)
    assert int(out.player_hp) == 10


# ---------------------------------------------------------------------------
# test_cursed_gain_level_ascends
# ---------------------------------------------------------------------------

def test_cursed_gain_level_ascends():
    """Cursed gain_level decrements current_level (ascend one floor)."""
    s = _state_with_potion(PotionEffect.GAIN_LEVEL, buc=_BUC_CURSED)
    s = s.replace(dungeon=s.dungeon.replace(current_level=jnp.int8(5)))
    out = quaff_potion(s, _RNG, 0)
    assert int(out.dungeon.current_level) == 4


def test_cursed_gain_level_floor_at_1():
    """Cursed gain_level at level 1 stays at 1 (no underflow)."""
    s = _state_with_potion(PotionEffect.GAIN_LEVEL, buc=_BUC_CURSED)
    s = s.replace(dungeon=s.dungeon.replace(current_level=jnp.int8(1)))
    out = quaff_potion(s, _RNG, 0)
    assert int(out.dungeon.current_level) == 1


def test_uncursed_gain_level_increments_xl():
    """Uncursed gain_level increments XL and does not change dungeon level."""
    s = _state_with_potion(PotionEffect.GAIN_LEVEL, buc=_BUC_UNCURSED)
    s = s.replace(player_xl=jnp.int32(3),
                  dungeon=s.dungeon.replace(current_level=jnp.int8(5)))
    out = quaff_potion(s, _RNG, 0)
    assert int(out.player_xl) == 4
    assert int(out.dungeon.current_level) == 5
