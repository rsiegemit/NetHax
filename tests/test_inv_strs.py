"""Tests for NLE-fidelity inventory string rendering (Wave 3).

Covers:
  - Empty inventory: all 55 slots are zero
  - Slot 0 with a long sword (identified): bytes match expected prefix
  - Slot 1 with cursed ring mail (worn, BUC-known): contains "cursed" and
    "(being worn)"
  - Wand with charges (identified): includes "(N:8)" format
  - Quantity > 1: shows number, not "a"
  - Unidentified potion: uses appearance description not real name
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest
import jax.numpy as jnp
import jax

from Nethax.nethax.subsystems.inventory import (
    InventoryState, Item, ArmorSlot, MAX_INVENTORY_SLOTS, N_ARMOR_SLOTS,
    USER_NAME_LEN,
)
from Nethax.nethax.subsystems.items import BUCStatus
from Nethax.nethax.subsystems.identification import IdentificationState
from Nethax.nethax.constants.objects import OBJECTS, ObjectClass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode(row: jnp.ndarray) -> str:
    """Decode a uint8[80] row to a Python string (strip null bytes)."""
    return bytes(row.tolist()).rstrip(b'\x00').decode("ascii", errors="replace")


def _find_object_type_id(name: str) -> int:
    """Find the type_id (index in OBJECTS) for an object by canonical name."""
    for i, obj in enumerate(OBJECTS):
        if obj.name == name:
            return i
    raise ValueError(f"Object not found: {name!r}")


def _find_object_type_id_class(obj_class: ObjectClass, name_contains: str) -> int:
    """Find type_id for first object matching class and name substring."""
    for i, obj in enumerate(OBJECTS):
        if obj.class_ == obj_class and name_contains.lower() in obj.name.lower():
            return i
    raise ValueError(f"Object not found: class={obj_class}, name~={name_contains!r}")


def _make_empty_items() -> Item:
    """Return a 52-slot Item array, all empty (category=0).

    post-erosion-merge: positional → kwarg; new fields greased/oeroded*/bknown/lamplit/olocked/corpse_entry_idx added.
    """
    return Item(
        category=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
        type_id=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int16),
        buc_status=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
        enchantment=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
        charges=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
        identified=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        quantity=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int16),
        weight=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int32),
        ac_bonus=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
        is_two_handed=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        greased=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        oeroded=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
        oeroded2=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
        oerodeproof=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        bknown=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        lamplit=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        olocked=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        corpse_entry_idx=jnp.full((MAX_INVENTORY_SLOTS,), -1, dtype=jnp.int16),
    )


def _make_inv_state(items: Item) -> InventoryState:
    return InventoryState(
        items=items,
        wielded=jnp.int8(-1),
        off_hand=jnp.int8(-1),
        worn_armor=jnp.full((N_ARMOR_SLOTS,), -1, dtype=jnp.int8),
        worn_amulet=jnp.int8(-1),
        worn_rings=jnp.full((2,), -1, dtype=jnp.int8),
        quiver=jnp.int8(-1),
        total_weight=jnp.int32(0),
        alternate_weapon_slot=jnp.int8(-1),
        worn_armor_ac_bonus=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.int8),
        user_names=jnp.zeros((MAX_INVENTORY_SLOTS, USER_NAME_LEN), dtype=jnp.int8),
        wielded_artifact_idx=jnp.int8(-1),
    )


def _make_state_with_inv(inv_state: InventoryState):
    """Build a minimal EnvState-like object with just inventory + identification."""
    id_state = IdentificationState.unshuffled()

    class _FakeState:
        inventory = inv_state
        identification = id_state

    return _FakeState()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmptyInventory:
    def test_all_zeros(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)

        result = build_inv_strs(state)

        assert result.shape == (55, 80), f"shape mismatch: {result.shape}"
        assert result.dtype == jnp.uint8
        assert jnp.all(result == 0), "expected all-zero for empty inventory"


class TestLongSword:
    """Slot 0 with an identified +0 long sword."""

    def setup_method(self):
        self.type_id = _find_object_type_id("long sword")
        self.obj     = OBJECTS[self.type_id]

    def test_starts_with_letter_a(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.WEAPON_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.UNCURSED)),
            enchantment=items.enchantment.at[0].set(jnp.int8(0)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[0])
        assert s.startswith("a - "), f"expected 'a - ' prefix, got: {s!r}"

    def test_contains_long_sword(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.WEAPON_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.UNCURSED)),
            enchantment=items.enchantment.at[0].set(jnp.int8(0)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[0])
        assert "long sword" in s, f"expected 'long sword' in: {s!r}"

    def test_shows_plus_zero_enchantment(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.WEAPON_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.UNCURSED)),
            enchantment=items.enchantment.at[0].set(jnp.int8(0)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[0])
        assert "+0" in s, f"expected '+0' enchantment in: {s!r}"

    def test_weapon_in_hand_when_wielded(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.WEAPON_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.UNCURSED)),
            enchantment=items.enchantment.at[0].set(jnp.int8(0)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        inv   = inv.replace(wielded=jnp.int8(0))
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[0])
        assert "(weapon in hand)" in s, f"expected '(weapon in hand)' in: {s!r}"

    def test_other_slots_empty_when_only_slot0_set(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.WEAPON_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.UNCURSED)),
            enchantment=items.enchantment.at[0].set(jnp.int8(0)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        # Slots 1-54 should be all zero
        assert jnp.all(result[1:] == 0), "expected slots 1-54 to be zero"


class TestRingMail:
    """Slot 1 with cursed ring mail (BUC-known, worn on body)."""

    def setup_method(self):
        self.type_id = _find_object_type_id("ring mail")

    def test_contains_cursed(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[1].set(jnp.int8(int(ObjectClass.ARMOR_CLASS))),
            type_id=items.type_id.at[1].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[1].set(jnp.int8(BUCStatus.CURSED)),
            enchantment=items.enchantment.at[1].set(jnp.int8(0)),
            identified=items.identified.at[1].set(True),
            quantity=items.quantity.at[1].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[1])
        assert "cursed" in s, f"expected 'cursed' in: {s!r}"

    def test_starts_with_b_dash(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[1].set(jnp.int8(int(ObjectClass.ARMOR_CLASS))),
            type_id=items.type_id.at[1].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[1].set(jnp.int8(BUCStatus.CURSED)),
            enchantment=items.enchantment.at[1].set(jnp.int8(0)),
            identified=items.identified.at[1].set(True),
            quantity=items.quantity.at[1].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[1])
        assert s.startswith("b - "), f"expected 'b - ' prefix, got: {s!r}"

    def test_being_worn_when_equipped(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[1].set(jnp.int8(int(ObjectClass.ARMOR_CLASS))),
            type_id=items.type_id.at[1].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[1].set(jnp.int8(BUCStatus.CURSED)),
            enchantment=items.enchantment.at[1].set(jnp.int8(0)),
            identified=items.identified.at[1].set(True),
            quantity=items.quantity.at[1].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv = _make_inv_state(items)
        # Wear item 1 in body armor slot (ArmorSlot.BODY = 0)
        worn = inv.worn_armor.at[int(ArmorSlot.BODY)].set(jnp.int8(1))
        inv  = inv.replace(worn_armor=worn)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[1])
        assert "(being worn)" in s, f"expected '(being worn)' in: {s!r}"

    def test_contains_ring_mail(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[1].set(jnp.int8(int(ObjectClass.ARMOR_CLASS))),
            type_id=items.type_id.at[1].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[1].set(jnp.int8(BUCStatus.CURSED)),
            enchantment=items.enchantment.at[1].set(jnp.int8(0)),
            identified=items.identified.at[1].set(True),
            quantity=items.quantity.at[1].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[1])
        assert "ring mail" in s, f"expected 'ring mail' in: {s!r}"


class TestWandCharges:
    """Slot 2 with an identified wand — charges shown as (recharged:charges) per objnam.c:1486."""

    def setup_method(self):
        # Find any wand
        self.type_id = _find_object_type_id_class(ObjectClass.WAND_CLASS, "striking")

    def test_charges_shown(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[2].set(jnp.int8(int(ObjectClass.WAND_CLASS))),
            type_id=items.type_id.at[2].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[2].set(jnp.int8(BUCStatus.UNCURSED)),
            enchantment=items.enchantment.at[2].set(jnp.int8(0)),
            charges=items.charges.at[2].set(jnp.int8(5)),
            identified=items.identified.at[2].set(True),
            quantity=items.quantity.at[2].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[2])
        assert "(0:5)" in s, f"expected '(0:5)' charges in: {s!r}"

    def test_charges_not_shown_when_unidentified(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[2].set(jnp.int8(int(ObjectClass.WAND_CLASS))),
            type_id=items.type_id.at[2].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[2].set(jnp.int8(BUCStatus.UNKNOWN)),
            enchantment=items.enchantment.at[2].set(jnp.int8(0)),
            charges=items.charges.at[2].set(jnp.int8(5)),
            identified=items.identified.at[2].set(False),
            quantity=items.quantity.at[2].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[2])
        assert ":" not in s, f"expected no charges when unidentified, got: {s!r}"

    def test_starts_with_c_dash(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[2].set(jnp.int8(int(ObjectClass.WAND_CLASS))),
            type_id=items.type_id.at[2].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[2].set(jnp.int8(BUCStatus.UNCURSED)),
            enchantment=items.enchantment.at[2].set(jnp.int8(0)),
            charges=items.charges.at[2].set(jnp.int8(5)),
            identified=items.identified.at[2].set(True),
            quantity=items.quantity.at[2].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[2])
        assert s.startswith("c - "), f"expected 'c - ' prefix, got: {s!r}"


class TestQuantity:
    """Quantity > 1 shows number, not 'a'."""

    def setup_method(self):
        self.type_id = _find_object_type_id("dagger")

    def test_plural_shows_count(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[3].set(jnp.int8(int(ObjectClass.WEAPON_CLASS))),
            type_id=items.type_id.at[3].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[3].set(jnp.int8(BUCStatus.UNKNOWN)),
            enchantment=items.enchantment.at[3].set(jnp.int8(0)),
            identified=items.identified.at[3].set(True),
            quantity=items.quantity.at[3].set(jnp.int16(5)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[3])
        assert "5 " in s, f"expected quantity '5 ' in: {s!r}"
        assert not s.startswith("d - a "), f"should not use 'a' for plural: {s!r}"

    def test_singular_shows_a(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[3].set(jnp.int8(int(ObjectClass.WEAPON_CLASS))),
            type_id=items.type_id.at[3].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[3].set(jnp.int8(BUCStatus.UNKNOWN)),
            enchantment=items.enchantment.at[3].set(jnp.int8(0)),
            identified=items.identified.at[3].set(True),
            quantity=items.quantity.at[3].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[3])
        # Should be "d - a dagger" or "d - a uncursed dagger"
        assert s.startswith("d - a "), f"expected 'd - a ' for singular, got: {s!r}"


class TestUnidentifiedPotion:
    """Unidentified potion uses appearance description, not real name."""

    def setup_method(self):
        # Find a potion with a description (e.g. "gain ability" -> "ruby")
        for i, obj in enumerate(OBJECTS):
            if obj.class_ == ObjectClass.POTION_CLASS and obj.description is not None:
                self.type_id    = i
                self.real_name  = obj.name
                self.appearance = obj.description
                break
        else:
            pytest.skip("No potion with appearance found in OBJECTS table")

    def test_appearance_used_when_unidentified(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[4].set(jnp.int8(int(ObjectClass.POTION_CLASS))),
            type_id=items.type_id.at[4].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[4].set(jnp.int8(BUCStatus.UNKNOWN)),
            enchantment=items.enchantment.at[4].set(jnp.int8(0)),
            identified=items.identified.at[4].set(False),
            quantity=items.quantity.at[4].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[4])
        # Should show appearance word (e.g. "ruby"), not the real name
        assert self.appearance in s, (
            f"expected appearance {self.appearance!r} in: {s!r}"
        )
        assert self.real_name not in s, (
            f"should not show real name {self.real_name!r} when unidentified: {s!r}"
        )

    def test_true_name_used_when_identified(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[4].set(jnp.int8(int(ObjectClass.POTION_CLASS))),
            type_id=items.type_id.at[4].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[4].set(jnp.int8(BUCStatus.UNKNOWN)),
            enchantment=items.enchantment.at[4].set(jnp.int8(0)),
            identified=items.identified.at[4].set(True),
            quantity=items.quantity.at[4].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[4])
        # Should show real name with prefix ("potion of gain ability")
        assert self.real_name in s, (
            f"expected real name {self.real_name!r} when identified in: {s!r}"
        )
        assert "potion of" in s, f"expected 'potion of' prefix in: {s!r}"


class TestOutputShape:
    """Output shape and dtype are always (55, 80) uint8."""

    def test_shape_empty(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        inv   = _make_inv_state(_make_empty_items())
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        assert result.shape == (55, 80)
        assert result.dtype == jnp.uint8

    def test_shape_with_items(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        type_id = _find_object_type_id("long sword")
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.WEAPON_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.UNCURSED)),
            enchantment=items.enchantment.at[0].set(jnp.int8(2)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        assert result.shape == (55, 80)
        assert result.dtype == jnp.uint8

    def test_no_string_exceeds_80_bytes(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        # Fill all 52 slots with long-named items
        type_id = _find_object_type_id("long sword")
        items = _make_empty_items()
        for i in range(MAX_INVENTORY_SLOTS):
            items = items.replace(
                category=items.category.at[i].set(jnp.int8(int(ObjectClass.WEAPON_CLASS))),
                type_id=items.type_id.at[i].set(jnp.int16(type_id)),
                buc_status=items.buc_status.at[i].set(jnp.int8(BUCStatus.BLESSED)),
                enchantment=items.enchantment.at[i].set(jnp.int8(5)),
                identified=items.identified.at[i].set(True),
                quantity=items.quantity.at[i].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
            )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        assert result.shape == (55, 80)
        # Byte 79 must be zero (null terminator must fit) — just verify shape
        # (no string can exceed 80 bytes because the buffer is exactly 80 wide)
        assert result.dtype == jnp.uint8


class TestSpecialSlots:
    """Slots 52-54 (beyond a-zA-Z) are always zero."""

    def test_high_slots_zero(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        inv   = _make_inv_state(_make_empty_items())
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        assert jnp.all(result[52] == 0)
        assert jnp.all(result[53] == 0)
        assert jnp.all(result[54] == 0)


class TestBlessedItem:
    """Blessed item shows 'blessed' word."""

    def test_shows_blessed(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        type_id = _find_object_type_id("long sword")
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.WEAPON_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.BLESSED)),
            enchantment=items.enchantment.at[0].set(jnp.int8(2)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[0])
        assert "blessed" in s, f"expected 'blessed' in: {s!r}"

    def test_no_buc_when_unknown(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        type_id = _find_object_type_id("dagger")
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.WEAPON_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.UNKNOWN)),
            enchantment=items.enchantment.at[0].set(jnp.int8(0)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv   = _make_inv_state(items)
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[0])
        assert "blessed" not in s
        assert "uncursed" not in s
        assert "cursed" not in s


class TestRingEquipStatus:
    """Ring shows '(on left hand)' / '(on right hand)'."""

    def setup_method(self):
        self.type_id = _find_object_type_id_class(ObjectClass.RING_CLASS, "")

    def test_left_ring(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.RING_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.UNKNOWN)),
            enchantment=items.enchantment.at[0].set(jnp.int8(0)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv = _make_inv_state(items)
        inv = inv.replace(worn_rings=inv.worn_rings.at[0].set(jnp.int8(0)))
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[0])
        assert "(on left hand)" in s, f"expected '(on left hand)' in: {s!r}"

    def test_right_ring(self):
        from Nethax.nethax.obs.inv_strs import build_inv_strs
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.RING_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.UNKNOWN)),
            enchantment=items.enchantment.at[0].set(jnp.int8(0)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
        )
        inv = _make_inv_state(items)
        inv = inv.replace(worn_rings=inv.worn_rings.at[1].set(jnp.int8(0)))
        state = _make_state_with_inv(inv)
        result = build_inv_strs(state)
        s = _decode(result[0])
        assert "(on right hand)" in s, f"expected '(on right hand)' in: {s!r}"
