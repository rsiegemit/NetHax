"""objnam.c xname/doname gap-fill parity tests.

Covers:
  1. Holy water (vendor objnam.c:841-843)
  2. Unholy water
  3. Ring +N enchantment (vendor objnam.c:1500)
  4. Wand (recharged:charges) counter (vendor objnam.c:1486)
  5. Corpse name with monster (vendor objnam.c:1824)
  6. Tin of spinach (vendor eat.c:tin_details)
  7. Quiver pouch for ring (vendor objnam.c:1636)
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np
import pytest

from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.constants.objects import OBJECTS, ObjectClass
from Nethax.nethax.subsystems.identification import IdentificationState
from Nethax.nethax.subsystems.inventory import (
    MAX_INVENTORY_SLOTS,
    N_ARMOR_SLOTS,
    USER_NAME_LEN,
    InventoryState,
    Item,
)
from Nethax.nethax.subsystems.items import BUCStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode(row) -> str:
    arr = np.asarray(row).astype(np.uint8)
    return bytes(arr.tolist()).rstrip(b"\x00").decode("ascii", errors="replace")


def _find_type_id(name: str) -> int:
    for i, obj in enumerate(OBJECTS):
        if obj.name == name:
            return i
    raise ValueError(f"Object not found: {name!r}")


def _find_type_id_class(cls: ObjectClass, substr: str) -> int:
    for i, obj in enumerate(OBJECTS):
        if obj.class_ == cls and substr.lower() in (obj.name or "").lower():
            return i
    raise ValueError(f"Object not found: class={cls}, name~={substr!r}")


def _find_monster_idx(name: str) -> int:
    for i, m in enumerate(MONSTERS):
        if m is not None and m.name and m.name.lower() == name.lower():
            return i
    raise ValueError(f"Monster not found: {name!r}")


def _make_empty_items() -> Item:
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
        recharged=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
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
        welded=jnp.bool_(False),
        worn_armor_welded=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.bool_),
        worn_amulet_welded=jnp.bool_(False),
        worn_rings_welded=jnp.zeros((2,), dtype=jnp.bool_),
    )


def _make_state(inv: InventoryState):
    id_state = IdentificationState.unshuffled()

    class _FakeState:
        inventory = inv
        identification = id_state

    return _FakeState()


def _render(items: Item, slot: int = 0) -> str:
    from Nethax.nethax.obs.inv_strs import build_inv_strs

    inv   = _make_inv_state(items)
    state = _make_state(inv)
    result = build_inv_strs(state)
    return _decode(result[slot])


# ---------------------------------------------------------------------------
# Gap 1 & 2: Holy / unholy water  (vendor objnam.c:841-843)
# ---------------------------------------------------------------------------

class TestHolyUnholyWater:
    def setup_method(self):
        self.type_id = _find_type_id("water")   # POT_WATER

    def test_blessed_water_is_holy_water(self):
        """Identified blessed POT_WATER renders 'holy water' (objnam.c:841-843)."""
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.POTION_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.BLESSED)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        )
        s = _render(items)
        assert "holy water" in s, f"expected 'holy water' in: {s!r}"
        assert "blessed" not in s, f"'blessed' prefix should be absent, got: {s!r}"

    def test_cursed_water_is_unholy_water(self):
        """Identified cursed POT_WATER renders 'unholy water' (objnam.c:843)."""
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.POTION_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.CURSED)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        )
        s = _render(items)
        assert "unholy water" in s, f"expected 'unholy water' in: {s!r}"
        assert "cursed" not in s, f"'cursed' prefix should be absent, got: {s!r}"

    def test_uncursed_water_shows_potion_of_water(self):
        """Identified uncursed POT_WATER renders 'potion of water' (normal path)."""
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.POTION_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(self.type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.UNCURSED)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        )
        s = _render(items)
        assert "potion of water" in s, f"expected 'potion of water' in: {s!r}"


# ---------------------------------------------------------------------------
# Gap 3: Ring +N enchantment  (vendor objnam.c:1500)
# ---------------------------------------------------------------------------

class TestRingEnchantment:
    def test_ring_of_protection_shows_enchant(self):
        """Identified +2 ring of protection renders '+2 ring of protection' (objnam.c:1500)."""
        type_id = _find_type_id("protection")
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.RING_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.BLESSED)),
            enchantment=items.enchantment.at[0].set(jnp.int8(2)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        )
        s = _render(items)
        assert "+2" in s, f"expected '+2' enchantment in: {s!r}"
        assert "protection" in s, f"expected 'protection' in: {s!r}"

    def test_ring_of_searching_no_enchant(self):
        """Ring of searching (not oc_charged) should NOT show enchantment."""
        type_id = _find_type_id("searching")
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.RING_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.UNCURSED)),
            enchantment=items.enchantment.at[0].set(jnp.int8(3)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        )
        s = _render(items)
        assert "+3" not in s, f"non-charged ring should not show '+3', got: {s!r}"


# ---------------------------------------------------------------------------
# Gap 4: Wand recharged counter  (vendor objnam.c:1486)
# ---------------------------------------------------------------------------

class TestWandRechargedCounter:
    def test_wand_shows_recharged_counter(self):
        """Identified wand with recharged=1, charges=5 renders '(1:5)' (objnam.c:1486)."""
        type_id = _find_type_id_class(ObjectClass.WAND_CLASS, "striking")
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.WAND_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.UNCURSED)),
            charges=items.charges.at[0].set(jnp.int8(5)),
            recharged=items.recharged.at[0].set(jnp.int8(1)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        )
        s = _render(items)
        assert "(1:5)" in s, f"expected '(1:5)' charges in: {s!r}"

    def test_wand_zero_recharged(self):
        """Wand with recharged=0, charges=3 renders '(0:3)'."""
        type_id = _find_type_id_class(ObjectClass.WAND_CLASS, "striking")
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.WAND_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(type_id)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.UNCURSED)),
            charges=items.charges.at[0].set(jnp.int8(3)),
            recharged=items.recharged.at[0].set(jnp.int8(0)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        )
        s = _render(items)
        assert "(0:3)" in s, f"expected '(0:3)' charges in: {s!r}"


# ---------------------------------------------------------------------------
# Gap 5: Corpse name with monster  (vendor objnam.c:1824)
# ---------------------------------------------------------------------------

class TestCorpseName:
    def test_corpse_named_with_monster(self):
        """Corpse with newt corpse_entry_idx renders 'newt corpse' (objnam.c:1824)."""
        type_id = _find_type_id("corpse")
        monster_idx = _find_monster_idx("newt")
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.FOOD_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(type_id)),
            corpse_entry_idx=items.corpse_entry_idx.at[0].set(jnp.int16(monster_idx)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        )
        s = _render(items)
        assert "newt corpse" in s, f"expected 'newt corpse' in: {s!r}"

    def test_corpse_no_monster_shows_generic(self):
        """Corpse without monster idx shows 'corpse' (generic fallback)."""
        type_id = _find_type_id("corpse")
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.FOOD_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(type_id)),
            # corpse_entry_idx stays -1 (no monster)
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        )
        s = _render(items)
        assert "corpse" in s, f"expected 'corpse' in generic path: {s!r}"


# ---------------------------------------------------------------------------
# Gap 6: Tin contents  (vendor eat.c:tin_details)
# ---------------------------------------------------------------------------

class TestTinContents:
    def test_tin_of_spinach(self):
        """Tin with enchantment==1 (spe==1) renders 'tin of spinach'."""
        type_id = _find_type_id("tin")
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.FOOD_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(type_id)),
            enchantment=items.enchantment.at[0].set(jnp.int8(1)),  # spe==1 -> spinach
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        )
        s = _render(items)
        assert "tin of spinach" in s, f"expected 'tin of spinach' in: {s!r}"

    def test_tin_of_monster_meat(self):
        """Tin with corpse_entry_idx for newt renders 'tin of newt meat'."""
        type_id = _find_type_id("tin")
        monster_idx = _find_monster_idx("newt")
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.FOOD_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(type_id)),
            corpse_entry_idx=items.corpse_entry_idx.at[0].set(jnp.int16(monster_idx)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        )
        s = _render(items)
        assert "tin of" in s, f"expected 'tin of' in: {s!r}"
        assert "newt" in s, f"expected 'newt' in: {s!r}"
        assert "meat" in s, f"expected 'meat' in: {s!r}"


# ---------------------------------------------------------------------------
# Gap 7: Quiver suffix tags  (vendor objnam.c:1615-1646)
# ---------------------------------------------------------------------------

class TestQuiverSuffixTags:
    def test_quiver_pouch_for_ring(self):
        """Ring in quiver renders '(in quiver pouch)' not '(in quiver)' (objnam.c:1636)."""
        from Nethax.nethax.obs.inv_strs import build_inv_strs

        type_id = _find_type_id("protection")
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.RING_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(type_id)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        )
        inv = _make_inv_state(items)
        # Put ring in quiver slot
        inv = inv.replace(quiver=jnp.int8(0))
        state = _make_state(inv)
        result = build_inv_strs(state)
        s = _decode(result[0])
        assert "(in quiver pouch)" in s, f"expected '(in quiver pouch)' in: {s!r}"
        assert "(in quiver)" not in s or s.count("(in quiver") == 1 and "pouch" in s, \
            f"should use pouch variant, got: {s!r}"

    def test_at_the_ready_for_food_in_quiver(self):
        """Food in quiver renders '(at the ready)' (objnam.c:1639 Qtyp==3)."""
        from Nethax.nethax.obs.inv_strs import build_inv_strs

        type_id = _find_type_id("tin")
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(int(ObjectClass.FOOD_CLASS))),
            type_id=items.type_id.at[0].set(jnp.int16(type_id)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        )
        inv = _make_inv_state(items)
        inv = inv.replace(quiver=jnp.int8(0))
        state = _make_state(inv)
        result = build_inv_strs(state)
        s = _decode(result[0])
        assert "(at the ready)" in s, f"expected '(at the ready)' in: {s!r}"
