"""Parity tests for #dip — vendor/nethack/src/potion.c::dodip line 2267.

Covers the action_dispatch ``_handle_dip`` (M-d) handler that dips the first
non-potion inventory slot into the first POT_WATER (type_id=93) or
POT_OIL (type_id=92) potion.

  POT_WATER + BLESSED  → target cursed(1)→uncursed(2), uncursed(2)→blessed(3).
  POT_WATER + CURSED   → target blessed(3)→uncursed(2), uncursed(2)→cursed(1).
  POT_OIL              → target.greased = True.

Cite: vendor/nethack/src/potion.c::H2Opotion_dip lines 1498-1589.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.inventory import (
    InventoryState, make_item, ItemCategory,
)
from Nethax.nethax.subsystems.action_dispatch import (
    dispatch_action, _SLOT_DIP, _M_byte, _ACTION_TO_HANDLER_IDX,
)


_RNG = jax.random.PRNGKey(0)
_BUC_CURSED   = 1
_BUC_UNCURSED = 2
_BUC_BLESSED  = 3
_POT_OIL_ID   = 92
_POT_WATER_ID = 93


def _state_with_inv(items):
    state = EnvState.default(_RNG)
    return state.replace(inventory=InventoryState.from_items(items))


def test_md_routes_to_dip_slot():
    """M-d action byte maps to _SLOT_DIP."""
    md = _M_byte("d")
    assert int(_ACTION_TO_HANDLER_IDX[md]) == _SLOT_DIP


def test_dip_blessed_water_uncursed_target_becomes_blessed():
    """Blessed water + uncursed target → target becomes blessed."""
    target = make_item(category=int(ItemCategory.WEAPON), type_id=1,
                       quantity=1, buc_status=_BUC_UNCURSED)
    water  = make_item(category=int(ItemCategory.POTION), type_id=_POT_WATER_ID,
                       quantity=1, buc_status=_BUC_BLESSED)
    state = _state_with_inv([target, water])
    out = dispatch_action(state, jnp.int32(_M_byte("d")), _RNG)
    assert int(out.inventory.items.buc_status[0]) == _BUC_BLESSED
    # Water vehicle decremented.
    assert int(out.inventory.items.quantity[1]) == 0


def test_dip_blessed_water_cursed_target_becomes_uncursed():
    """Blessed water + cursed target → target becomes uncursed."""
    target = make_item(category=int(ItemCategory.WEAPON), type_id=1,
                       quantity=1, buc_status=_BUC_CURSED)
    water  = make_item(category=int(ItemCategory.POTION), type_id=_POT_WATER_ID,
                       quantity=1, buc_status=_BUC_BLESSED)
    state = _state_with_inv([target, water])
    out = dispatch_action(state, jnp.int32(_M_byte("d")), _RNG)
    assert int(out.inventory.items.buc_status[0]) == _BUC_UNCURSED


def test_dip_cursed_water_blessed_target_becomes_uncursed():
    """Cursed water + blessed target → target becomes uncursed."""
    target = make_item(category=int(ItemCategory.WEAPON), type_id=1,
                       quantity=1, buc_status=_BUC_BLESSED)
    water  = make_item(category=int(ItemCategory.POTION), type_id=_POT_WATER_ID,
                       quantity=1, buc_status=_BUC_CURSED)
    state = _state_with_inv([target, water])
    out = dispatch_action(state, jnp.int32(_M_byte("d")), _RNG)
    assert int(out.inventory.items.buc_status[0]) == _BUC_UNCURSED


def test_dip_cursed_water_uncursed_target_becomes_cursed():
    """Cursed water + uncursed target → target becomes cursed."""
    target = make_item(category=int(ItemCategory.WEAPON), type_id=1,
                       quantity=1, buc_status=_BUC_UNCURSED)
    water  = make_item(category=int(ItemCategory.POTION), type_id=_POT_WATER_ID,
                       quantity=1, buc_status=_BUC_CURSED)
    state = _state_with_inv([target, water])
    out = dispatch_action(state, jnp.int32(_M_byte("d")), _RNG)
    assert int(out.inventory.items.buc_status[0]) == _BUC_CURSED


def test_dip_oil_marks_target_greased():
    """POT_OIL → target.greased = True; oil quantity decremented."""
    target = make_item(category=int(ItemCategory.WEAPON), type_id=1,
                       quantity=1, buc_status=_BUC_UNCURSED)
    oil    = make_item(category=int(ItemCategory.POTION), type_id=_POT_OIL_ID,
                       quantity=1, buc_status=_BUC_UNCURSED)
    state = _state_with_inv([target, oil])
    out = dispatch_action(state, jnp.int32(_M_byte("d")), _RNG)
    assert bool(out.inventory.items.greased[0]) is True
    # BUC unchanged for oil dip.
    assert int(out.inventory.items.buc_status[0]) == _BUC_UNCURSED
    # Oil vehicle decremented.
    assert int(out.inventory.items.quantity[1]) == 0


def test_dip_no_vehicle_noop():
    """No water/oil in inventory → state unchanged."""
    target = make_item(category=int(ItemCategory.WEAPON), type_id=1,
                       quantity=1, buc_status=_BUC_UNCURSED)
    state = _state_with_inv([target])
    out = dispatch_action(state, jnp.int32(_M_byte("d")), _RNG)
    assert int(out.inventory.items.buc_status[0]) == _BUC_UNCURSED
    assert bool(out.inventory.items.greased[0]) is False


def test_dip_no_target_noop():
    """Only potions in inventory → no eligible target → state unchanged."""
    water = make_item(category=int(ItemCategory.POTION), type_id=_POT_WATER_ID,
                      quantity=1, buc_status=_BUC_BLESSED)
    state = _state_with_inv([water])
    out = dispatch_action(state, jnp.int32(_M_byte("d")), _RNG)
    # No target → vehicle quantity is unchanged.
    assert int(out.inventory.items.quantity[0]) == 1


def test_dip_water_preferred_over_oil():
    """When both water and oil exist, water is used as vehicle."""
    target = make_item(category=int(ItemCategory.WEAPON), type_id=1,
                       quantity=1, buc_status=_BUC_UNCURSED)
    oil    = make_item(category=int(ItemCategory.POTION), type_id=_POT_OIL_ID,
                       quantity=1, buc_status=_BUC_UNCURSED)
    water  = make_item(category=int(ItemCategory.POTION), type_id=_POT_WATER_ID,
                       quantity=1, buc_status=_BUC_BLESSED)
    state = _state_with_inv([target, oil, water])
    out = dispatch_action(state, jnp.int32(_M_byte("d")), _RNG)
    # Water (slot 2) consumed, not oil (slot 1).
    assert int(out.inventory.items.quantity[2]) == 0  # water gone
    assert int(out.inventory.items.quantity[1]) == 1  # oil intact
    # Target should be blessed (water effect), not greased (oil effect).
    assert int(out.inventory.items.buc_status[0]) == _BUC_BLESSED
    assert bool(out.inventory.items.greased[0]) is False
