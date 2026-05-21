"""objnam.c erosion-ordering and bknown parity tests (Wave 3 polish, batch 2).

Covers:
  1. Erosion order: oeroded (rusty) before oeroded2 (corroded) for iron items
     (vendor/nethack/src/objnam.c::add_erosion_words() lines 1142-1191)
  2. Thoroughly qualifier (oeroded=3 -> "thoroughly rusty")
  3. rustproof only shown when identified (vendor objnam.c:1183: rknown && oerodeproof)
  4. bknown=False suppresses BUC prefix (vendor objnam.c:1318)
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import pytest

from Nethax.nethax.subsystems.inventory import (
    MAX_INVENTORY_SLOTS,
    N_ARMOR_SLOTS,
    USER_NAME_LEN,
    InventoryState,
    Item,
)
from Nethax.nethax.subsystems.identification import IdentificationState
from Nethax.nethax.subsystems.items import BUCStatus
from Nethax.nethax.constants.objects import ObjectClass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WEAPON_CAT = int(ObjectClass.WEAPON_CLASS)
_IRON_WEAPON_TYPE = 37   # long sword (IRON -> rustprone + corrodeable)


def _decode(row) -> str:
    import numpy as np
    arr = np.asarray(row).astype(np.uint8)
    return bytes(arr.tolist()).rstrip(b"\x00").decode("ascii", errors="replace")


def _make_empty_items() -> Item:
    n = MAX_INVENTORY_SLOTS
    return Item(
        category=jnp.zeros((n,), dtype=jnp.int8),
        type_id=jnp.zeros((n,), dtype=jnp.int16),
        buc_status=jnp.zeros((n,), dtype=jnp.int8),
        enchantment=jnp.zeros((n,), dtype=jnp.int8),
        charges=jnp.zeros((n,), dtype=jnp.int8),
        identified=jnp.zeros((n,), dtype=jnp.bool_),
        quantity=jnp.zeros((n,), dtype=jnp.int16),
        weight=jnp.zeros((n,), dtype=jnp.int32),
        ac_bonus=jnp.zeros((n,), dtype=jnp.int8),
        is_two_handed=jnp.zeros((n,), dtype=jnp.bool_),
        greased=jnp.zeros((n,), dtype=jnp.bool_),
        oeroded=jnp.zeros((n,), dtype=jnp.int8),
        oeroded2=jnp.zeros((n,), dtype=jnp.int8),
        oerodeproof=jnp.zeros((n,), dtype=jnp.bool_),
        bknown=jnp.zeros((n,), dtype=jnp.bool_),
        lamplit=jnp.zeros((n,), dtype=jnp.bool_),
        olocked=jnp.zeros((n,), dtype=jnp.bool_),
        corpse_entry_idx=jnp.full((n,), -1, dtype=jnp.int16),
        recharged=jnp.zeros((n,), dtype=jnp.int8),
    )


def _make_inv_state(items: Item) -> InventoryState:
    return InventoryState(
        items=items,
        wielded=jnp.int8(-1),
        off_hand=jnp.int8(-1),
        alternate_weapon_slot=jnp.int8(-1),
        worn_armor=jnp.full((N_ARMOR_SLOTS,), -1, dtype=jnp.int8),
        worn_armor_ac_bonus=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.int8),
        armor_stat_bonus=jnp.zeros((6,), dtype=jnp.int8),
        worn_amulet=jnp.int8(-1),
        worn_rings=jnp.full((2,), -1, dtype=jnp.int8),
        quiver=jnp.int8(-1),
        total_weight=jnp.int32(0),
        user_names=jnp.zeros((MAX_INVENTORY_SLOTS, USER_NAME_LEN), dtype=jnp.int8),
        wielded_artifact_idx=jnp.int8(-1),
        welded=jnp.bool_(False),
        worn_armor_welded=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.bool_),
        worn_amulet_welded=jnp.bool_(False),
        worn_rings_welded=jnp.zeros((2,), dtype=jnp.bool_),
    )


def _make_state(items: Item):
    id_state = IdentificationState.unshuffled()

    class _FakeState:
        inventory = _make_inv_state(items)
        identification = id_state

    return _FakeState()


def _render(items: Item, slot: int = 0) -> str:
    from Nethax.nethax.obs.inv_strs import build_inv_strs
    result = build_inv_strs(_make_state(items))
    return _decode(result[slot])


# ---------------------------------------------------------------------------
# 1. Erosion order: rusty before corroded (vendor objnam.c:1142)
# ---------------------------------------------------------------------------

class TestErosionOrder:
    """oeroded and oeroded2 words appear in the vendor-specified order:
    oeroded (rusty/burnt) first, then oeroded2 (corroded/rotted).
    Canonical: vendor/nethack/src/objnam.c::add_erosion_words() lines 1156-1178.
    """

    def test_erosion_order_rust_then_corrode(self):
        """oeroded=1, oeroded2=1 on iron weapon -> 'rusty corroded long sword'."""
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(_WEAPON_CAT)),
            type_id=items.type_id.at[0].set(jnp.int16(_IRON_WEAPON_TYPE)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
            oeroded=items.oeroded.at[0].set(jnp.int8(1)),
            oeroded2=items.oeroded2.at[0].set(jnp.int8(1)),
        )
        s = _render(items)
        assert "rusty " in s, f"expected 'rusty ' in {s!r}"
        assert "corroded " in s, f"expected 'corroded ' in {s!r}"
        # rusty must come before corroded
        assert s.index("rusty ") < s.index("corroded "), (
            f"'rusty' should precede 'corroded' in {s!r}"
        )

    def test_thoroughly_qualifier(self):
        """oeroded=3 on iron weapon -> 'thoroughly rusty' prefix.
        Canonical: vendor objnam.c:1162-1163 (case 3: 'thoroughly ').
        """
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(_WEAPON_CAT)),
            type_id=items.type_id.at[0].set(jnp.int16(_IRON_WEAPON_TYPE)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
            oeroded=items.oeroded.at[0].set(jnp.int8(3)),
        )
        s = _render(items)
        assert "thoroughly rusty" in s, f"expected 'thoroughly rusty' in {s!r}"


# ---------------------------------------------------------------------------
# 2. rustproof only when identified (vendor objnam.c:1183)
# ---------------------------------------------------------------------------

class TestRustproofOnlyWhenKnown:
    """oerodeproof shows 'rustproof' only when identified.
    Canonical: vendor/nethack/src/objnam.c:1183: rknown && oerodeproof.
    """

    def test_rustproof_shown_when_identified(self):
        """oerodeproof=True + identified=True -> 'rustproof' in string."""
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(_WEAPON_CAT)),
            type_id=items.type_id.at[0].set(jnp.int16(_IRON_WEAPON_TYPE)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
            oerodeproof=items.oerodeproof.at[0].set(True),
        )
        s = _render(items)
        assert "rustproof" in s, f"expected 'rustproof' in {s!r}"

    def test_rustproof_only_when_known(self):
        """oerodeproof=True + identified=False -> no 'rustproof' in string."""
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(_WEAPON_CAT)),
            type_id=items.type_id.at[0].set(jnp.int16(_IRON_WEAPON_TYPE)),
            identified=items.identified.at[0].set(False),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
            oerodeproof=items.oerodeproof.at[0].set(True),
        )
        s = _render(items)
        assert "rustproof" not in s, f"'rustproof' should be absent when unidentified: {s!r}"


# ---------------------------------------------------------------------------
# 3. bknown suppresses BUC prefix (vendor objnam.c:1318)
# ---------------------------------------------------------------------------

class TestBknownSuppressesBuc:
    """bknown=False suppresses 'cursed'/'blessed'/'uncursed' prefix.
    Canonical: vendor/nethack/src/objnam.c:1318: if (bknown && ...).
    """

    def test_bknown_false_suppresses_buc(self):
        """bknown=False: no 'cursed' prefix even when buc_status=CURSED."""
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(_WEAPON_CAT)),
            type_id=items.type_id.at[0].set(jnp.int16(_IRON_WEAPON_TYPE)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.CURSED)),
            bknown=items.bknown.at[0].set(False),
        )
        s = _render(items)
        assert "cursed" not in s, f"'cursed' should be absent when bknown=False: {s!r}"

    def test_bknown_true_shows_buc(self):
        """bknown=True: 'cursed' prefix shown when buc_status=CURSED."""
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(_WEAPON_CAT)),
            type_id=items.type_id.at[0].set(jnp.int16(_IRON_WEAPON_TYPE)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.CURSED)),
            bknown=items.bknown.at[0].set(True),
        )
        s = _render(items)
        assert "cursed" in s, f"expected 'cursed' in {s!r}"

    def test_bknown_false_suppresses_blessed(self):
        """bknown=False: no 'blessed' prefix when buc_status=BLESSED."""
        items = _make_empty_items()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(_WEAPON_CAT)),
            type_id=items.type_id.at[0].set(jnp.int16(_IRON_WEAPON_TYPE)),
            identified=items.identified.at[0].set(True),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
            buc_status=items.buc_status.at[0].set(jnp.int8(BUCStatus.BLESSED)),
            bknown=items.bknown.at[0].set(False),
        )
        s = _render(items)
        assert "blessed" not in s, f"'blessed' should be absent when bknown=False: {s!r}"
