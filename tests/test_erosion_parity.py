"""Erosion prefix rendering parity tests.

Verifies that item.oeroded / oeroded2 / oerodeproof produce the correct
prefix words in inventory strings, matching vendor/nethack/src/objnam.c::
add_erosion_words() lines 1142-1191.

Type-id reference (Nethax/nethax/constants/objects.py):
  37  = long sword  (WEAPON, IRON  -> rustprone)
  35  = broadsword  (WEAPON, IRON  -> rustprone)
  34  = silver saber (WEAPON, SILVER -> corrodeable-only)
  113 = leather armor (ARMOR, LEATHER -> flammable)
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
    make_item,
)
from Nethax.nethax.subsystems.identification import IdentificationState
from Nethax.nethax.constants.objects import ObjectClass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WEAPON_CAT = int(ObjectClass.WEAPON_CLASS)
_ARMOR_CAT  = int(ObjectClass.ARMOR_CLASS)

# Vendor type-ids (see module docstring)
_IRON_WEAPON_TYPE   = 37   # long sword (IRON)
_LEATHER_ARMOR_TYPE = 113  # leather armor (LEATHER)
_SILVER_WEAPON_TYPE = 34   # silver saber (SILVER, corrodeable-only)


def _decode(row: jnp.ndarray) -> str:
    return bytes(row.tolist()).rstrip(b"\x00").decode("ascii", errors="replace")


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
        corpse_creation_turn=jnp.full((n,), -1, dtype=jnp.int32),
        tin_poisoned=jnp.zeros((n,), dtype=jnp.bool_),
    )


def _make_inv_state(items: Item) -> InventoryState:
    return InventoryState(
        items=items,
        wielded=jnp.int8(-1),
        off_hand=jnp.int8(-1),
        alternate_weapon_slot=jnp.int8(-1),
        worn_armor=jnp.full((N_ARMOR_SLOTS,), -1, dtype=jnp.int8),
        worn_armor_ac_bonus=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.int8),
        worn_amulet=jnp.int8(-1),
        worn_rings=jnp.full((2,), -1, dtype=jnp.int8),
        quiver=jnp.int8(-1),
        total_weight=jnp.int32(0),
        user_names=jnp.zeros((MAX_INVENTORY_SLOTS, USER_NAME_LEN), dtype=jnp.int8),
        wielded_artifact_idx=jnp.int8(-1),
    )


def _make_state(items: Item):
    id_state = IdentificationState.unshuffled()

    class _FakeState:
        inventory = _make_inv_state(items)
        identification = id_state

    return _FakeState()


def _render_slot0(
    cat: int,
    type_id: int,
    oeroded: int = 0,
    oeroded2: int = 0,
    oerodeproof: bool = False,
) -> str:
    """Render inventory slot 0 with the given item fields and return string."""
    from Nethax.nethax.obs.inv_strs import build_inv_strs

    items = _make_empty_items()
    items = items.replace(
        category=items.category.at[0].set(jnp.int8(cat)),
        type_id=items.type_id.at[0].set(jnp.int16(type_id)),
        identified=items.identified.at[0].set(jnp.bool_(True)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
        oeroded=items.oeroded.at[0].set(jnp.int8(oeroded)),
        oeroded2=items.oeroded2.at[0].set(jnp.int8(oeroded2)),
        oerodeproof=items.oerodeproof.at[0].set(jnp.bool_(oerodeproof)),
    )
    result = build_inv_strs(_make_state(items))
    return _decode(result[0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRustyIronWeapon:
    """oeroded on IRON weapon -> rusty/very rusty/thoroughly rusty prefix.

    Canonical: vendor/nethack/src/objnam.c lines 1156-1167.
    """

    def test_rusty_prefix(self):
        """oeroded=1 on iron weapon -> 'rusty ' in inv text."""
        s = _render_slot0(_WEAPON_CAT, _IRON_WEAPON_TYPE, oeroded=1)
        assert "rusty " in s, f"expected 'rusty ' in {s!r}"
        assert "very rusty" not in s

    def test_very_rusty(self):
        """oeroded=2 -> 'very rusty '."""
        s = _render_slot0(_WEAPON_CAT, _IRON_WEAPON_TYPE, oeroded=2)
        assert "very rusty " in s, f"expected 'very rusty ' in {s!r}"

    def test_thoroughly_rusty(self):
        """oeroded=3 -> 'thoroughly rusty '."""
        s = _render_slot0(_WEAPON_CAT, _IRON_WEAPON_TYPE, oeroded=3)
        assert "thoroughly rusty " in s, f"expected 'thoroughly rusty ' in {s!r}"

    def test_no_erosion(self):
        """oeroded=0 -> no rust prefix."""
        s = _render_slot0(_WEAPON_CAT, _IRON_WEAPON_TYPE, oeroded=0)
        assert "rusty" not in s, f"unexpected rust prefix in {s!r}"
        assert "burnt" not in s
        assert "corroded" not in s


class TestRustproofKnown:
    """oerodeproof=True on identified iron item -> 'rustproof ' prefix.

    Canonical: vendor/nethack/src/objnam.c line 1183-1190.
    """

    def test_rustproof_known(self):
        """oerodeproof=True, identified iron weapon -> 'rustproof '."""
        s = _render_slot0(_WEAPON_CAT, _IRON_WEAPON_TYPE, oerodeproof=True)
        assert "rustproof " in s, f"expected 'rustproof ' in {s!r}"

    def test_rustproof_with_rust(self):
        """Can be both rusty AND rustproof (cursed scroll edge-case per vendor comment line 1180)."""
        s = _render_slot0(_WEAPON_CAT, _IRON_WEAPON_TYPE, oeroded=1, oerodeproof=True)
        assert "rusty " in s, f"expected 'rusty ' in {s!r}"
        assert "rustproof " in s, f"expected 'rustproof ' in {s!r}"


class TestBurntLeatherArmor:
    """oeroded on LEATHER armor -> burnt/very burnt/thoroughly burnt.

    Canonical: vendor/nethack/src/objnam.c line 1166 (is_flammable -> 'burnt').
    """

    def test_burnt_leather(self):
        """oeroded=1, leather armor -> 'burnt '."""
        s = _render_slot0(_ARMOR_CAT, _LEATHER_ARMOR_TYPE, oeroded=1)
        assert "burnt " in s, f"expected 'burnt ' in {s!r}"
        assert "rusty" not in s

    def test_very_burnt(self):
        """oeroded=2 -> 'very burnt '."""
        s = _render_slot0(_ARMOR_CAT, _LEATHER_ARMOR_TYPE, oeroded=2)
        assert "very burnt " in s, f"expected 'very burnt ' in {s!r}"

    def test_thoroughly_burnt(self):
        """oeroded=3 -> 'thoroughly burnt '."""
        s = _render_slot0(_ARMOR_CAT, _LEATHER_ARMOR_TYPE, oeroded=3)
        assert "thoroughly burnt " in s, f"expected 'thoroughly burnt ' in {s!r}"

    def test_rotted_leather(self):
        """oeroded2=1 on leather -> 'rotted '."""
        s = _render_slot0(_ARMOR_CAT, _LEATHER_ARMOR_TYPE, oeroded2=1)
        assert "rotted " in s, f"expected 'rotted ' in {s!r}"

    def test_fireproof_leather(self):
        """oerodeproof=True, leather -> 'fireproof '."""
        s = _render_slot0(_ARMOR_CAT, _LEATHER_ARMOR_TYPE, oerodeproof=True)
        assert "fireproof " in s, f"expected 'fireproof ' in {s!r}"

    def test_rotproof_leather(self):
        """oerodeproof=True + oeroded2>0 on leather -> 'rotproof '."""
        s = _render_slot0(_ARMOR_CAT, _LEATHER_ARMOR_TYPE, oeroded2=1, oerodeproof=True)
        assert "rotproof " in s, f"expected 'rotproof ' in {s!r}"
        assert "fireproof" not in s


class TestCorrodeableSilver:
    """SILVER weapon: oeroded2 -> corroded series; oeroded unused.

    Canonical: vendor/nethack/src/objnam.c lines 1169-1178 (is_corrodeable).
    """

    def test_corroded_silver(self):
        """oeroded2=1 on silver weapon -> 'corroded '."""
        s = _render_slot0(_WEAPON_CAT, _SILVER_WEAPON_TYPE, oeroded2=1)
        assert "corroded " in s, f"expected 'corroded ' in {s!r}"

    def test_very_corroded_silver(self):
        """oeroded2=2 -> 'very corroded '."""
        s = _render_slot0(_WEAPON_CAT, _SILVER_WEAPON_TYPE, oeroded2=2)
        assert "very corroded " in s, f"expected 'very corroded ' in {s!r}"

    def test_corrodeproof_silver(self):
        """oerodeproof=True on silver weapon -> 'corrodeproof '."""
        s = _render_slot0(_WEAPON_CAT, _SILVER_WEAPON_TYPE, oerodeproof=True)
        assert "corrodeproof " in s, f"expected 'corrodeproof ' in {s!r}"


class TestMakeItemErosionFields:
    """make_item() now accepts oeroded/oeroded2/oerodeproof kwargs."""

    def test_make_item_defaults(self):
        item = make_item(category=_WEAPON_CAT, type_id=_IRON_WEAPON_TYPE)
        assert int(item.oeroded) == 0
        assert int(item.oeroded2) == 0
        assert bool(item.oerodeproof) is False

    def test_make_item_erosion_kwargs(self):
        item = make_item(
            category=_WEAPON_CAT,
            type_id=_IRON_WEAPON_TYPE,
            oeroded=2,
            oeroded2=1,
            oerodeproof=True,
        )
        assert int(item.oeroded) == 2
        assert int(item.oeroded2) == 1
        assert bool(item.oerodeproof) is True
