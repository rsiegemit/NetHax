"""Wave 6 Phase A polish tests for inventory string rendering.

Covers the four Wave 3 simplifications closed in Wave 6:
  1. Vowel-aware a/an article for singular items.
  2. User-given object names (' named X' suffix).
  3. Irregular plurals (knife→knives, staff→staves) plus regular plurals.
  4. Two-weapon "(alternate weapon)" status marker.

Canonical references:
  vendor/nethack/src/objnam.c::an, doname, makeplural
  vendor/nethack/src/wield.c (two-weapon status)
  vendor/nethack/src/do_name.c::do_oname
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest
import jax.numpy as jnp

from Nethax.nethax.subsystems.inventory import (
    InventoryState, Item, MAX_INVENTORY_SLOTS, N_ARMOR_SLOTS, USER_NAME_LEN,
    handle_name,
)
from Nethax.nethax.subsystems.items import BUCStatus
from Nethax.nethax.subsystems.identification import IdentificationState
from Nethax.nethax.constants.objects import OBJECTS, ObjectClass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode(row: jnp.ndarray) -> str:
    return bytes(row.tolist()).rstrip(b"\x00").decode("ascii", errors="replace")


def _find_type_id_by_name(name: str) -> int:
    for i, obj in enumerate(OBJECTS):
        if obj.name == name:
            return i
    raise ValueError(f"Object not found by name: {name!r}")


def _find_type_id_by_appearance(appearance: str, obj_class: ObjectClass | None = None) -> int:
    for i, obj in enumerate(OBJECTS):
        if obj.description == appearance and (obj_class is None or obj.class_ == obj_class):
            return i
    raise ValueError(f"Object not found by appearance: {appearance!r}")


def _empty_items() -> Item:
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
    )


def _set_slot(items: Item, slot: int, *, type_id: int, category: int,
              quantity: int = 1, identified: bool = True,
              buc_status: int = BUCStatus.UNKNOWN, enchantment: int = 0) -> Item:
    return items.replace(
        category   = items.category.at[slot].set(jnp.int8(category)),
        type_id    = items.type_id.at[slot].set(jnp.int16(type_id)),
        buc_status = items.buc_status.at[slot].set(jnp.int8(buc_status)),
        enchantment= items.enchantment.at[slot].set(jnp.int8(enchantment)),
        identified = items.identified.at[slot].set(jnp.bool_(identified)),
        quantity   = items.quantity.at[slot].set(jnp.int16(quantity)),
    )


def _make_inv(items: Item, *, alt_slot: int = -1) -> InventoryState:
    return InventoryState(
        items=items,
        wielded=jnp.int8(-1),
        off_hand=jnp.int8(-1),
        alternate_weapon_slot=jnp.int8(alt_slot),
        worn_armor=jnp.full((N_ARMOR_SLOTS,), -1, dtype=jnp.int8),
        worn_armor_ac_bonus=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.int8),
        worn_amulet=jnp.int8(-1),
        worn_rings=jnp.full((2,), -1, dtype=jnp.int8),
        quiver=jnp.int8(-1),
        total_weight=jnp.int32(0),
        user_names=jnp.zeros((MAX_INVENTORY_SLOTS, USER_NAME_LEN), dtype=jnp.int8),
    )


class _FakeCombat:
    def __init__(self, two_weapon: bool = False):
        self.two_weapon = jnp.bool_(two_weapon)


class _FakeState:
    def __init__(self, inv: InventoryState, two_weapon: bool = False):
        self.inventory = inv
        self.identification = IdentificationState.unshuffled()
        self.combat = _FakeCombat(two_weapon)


# ---------------------------------------------------------------------------
# 1. Vowel-aware article
# ---------------------------------------------------------------------------

def test_an_article_for_vowel_appearance():
    """Unidentified amulet with vowel-start appearance (e.g. 'oval') yields 'an'."""
    from Nethax.nethax.obs.inv_strs import build_inv_strs

    # 'oval' is an amulet appearance.
    type_id = _find_type_id_by_appearance("oval", ObjectClass.AMULET_CLASS)
    items = _set_slot(
        _empty_items(), 0,
        type_id=type_id, category=int(ObjectClass.AMULET_CLASS),
        identified=False, buc_status=BUCStatus.UNKNOWN,
    )
    state = _FakeState(_make_inv(items))
    result = build_inv_strs(state)
    s = _decode(result[0])
    assert s.startswith("a - an oval"), f"expected 'an oval' article, got: {s!r}"


def test_a_article_for_consonant_appearance():
    """Unidentified ring with consonant appearance ('wooden') yields 'a'."""
    from Nethax.nethax.obs.inv_strs import build_inv_strs

    type_id = _find_type_id_by_appearance("wooden", ObjectClass.RING_CLASS)
    items = _set_slot(
        _empty_items(), 0,
        type_id=type_id, category=int(ObjectClass.RING_CLASS),
        identified=False, buc_status=BUCStatus.UNKNOWN,
    )
    state = _FakeState(_make_inv(items))
    result = build_inv_strs(state)
    s = _decode(result[0])
    assert s.startswith("a - a wooden"), f"expected 'a wooden' article, got: {s!r}"
    # Make sure we didn't emit 'an' for a consonant start.
    assert not s.startswith("a - an "), f"unexpected 'an' article: {s!r}"


# ---------------------------------------------------------------------------
# 2. User-given names
# ---------------------------------------------------------------------------

def test_user_named_item_appears_in_string():
    """A long sword with user_name 'Sting' renders 'long sword named Sting'."""
    from Nethax.nethax.obs.inv_strs import build_inv_strs

    type_id = _find_type_id_by_name("long sword")
    items = _set_slot(
        _empty_items(), 0,
        type_id=type_id, category=int(ObjectClass.WEAPON_CLASS),
        identified=True, buc_status=BUCStatus.UNCURSED,
    )
    inv = _make_inv(items)
    name = b"Sting" + b"\x00" * (USER_NAME_LEN - len(b"Sting"))
    new_user_names = inv.user_names.at[0].set(jnp.array(list(name), dtype=jnp.int8))
    inv = inv.replace(user_names=new_user_names)
    state = _FakeState(inv)
    result = build_inv_strs(state)
    s = _decode(result[0])
    assert "long sword named Sting" in s, (
        f"expected 'long sword named Sting' in: {s!r}"
    )


def test_handle_name_action_sets_user_name():
    """inventory.handle_name writes name into user_names[slot]."""
    from Nethax.nethax.state import EnvState
    import jax

    state = EnvState.default(jax.random.PRNGKey(0))
    # Verify initial state is zero.
    assert int(state.inventory.user_names[0, 0]) == 0
    new_state = handle_name(state, jax.random.PRNGKey(1), 0, "Excalibur")
    # First byte should now be 'E' = 0x45 = 69.
    assert int(new_state.inventory.user_names[0, 0]) == ord("E")
    # Remaining bytes should spell out the rest.
    encoded = bytes(new_state.inventory.user_names[0].tolist()).rstrip(b"\x00")
    assert encoded == b"Excalibur"


# ---------------------------------------------------------------------------
# 3. Plural irregulars + regulars
# ---------------------------------------------------------------------------

def test_irregular_plural_knife_knives():
    """3 identified knives renders as 'knives', not 'knifes'."""
    from Nethax.nethax.obs.inv_strs import build_inv_strs

    type_id = _find_type_id_by_name("knife")
    items = _set_slot(
        _empty_items(), 0,
        type_id=type_id, category=int(ObjectClass.WEAPON_CLASS),
        identified=True, quantity=3,
    )
    state = _FakeState(_make_inv(items))
    result = build_inv_strs(state)
    s = _decode(result[0])
    assert "knives" in s, f"expected 'knives' in: {s!r}"
    assert "knifes" not in s, f"should not contain 'knifes' in: {s!r}"


def test_irregular_plural_helper_staves():
    """pluralize('staff') -> 'staves' (helper-level test; no staff in OBJECTS)."""
    from Nethax.nethax.obs.inv_strs import pluralize, _pluralize_phrase
    assert pluralize("staff") == "staves"
    assert pluralize("knife") == "knives"
    assert pluralize("man") == "men"
    assert pluralize("ox") == "oxen"
    assert pluralize("mouse") == "mice"
    # Phrase-level helper preserves the "of" infix.
    assert _pluralize_phrase("ring of protection") == "rings of protection"
    assert _pluralize_phrase("long sword") == "long swords"


def test_regular_plural_appends_s():
    """5 identified long swords renders as 'long swords'."""
    from Nethax.nethax.obs.inv_strs import build_inv_strs

    type_id = _find_type_id_by_name("long sword")
    items = _set_slot(
        _empty_items(), 0,
        type_id=type_id, category=int(ObjectClass.WEAPON_CLASS),
        identified=True, quantity=5,
    )
    state = _FakeState(_make_inv(items))
    result = build_inv_strs(state)
    s = _decode(result[0])
    assert "5 " in s, f"expected quantity '5 ' in: {s!r}"
    assert "long swords" in s, f"expected 'long swords' in: {s!r}"


def test_regular_plural_es_for_sh_ending():
    """A name ending in 'sh' takes 'es' (leash -> leashes)."""
    from Nethax.nethax.obs.inv_strs import pluralize
    assert pluralize("leash") == "leashes"
    assert pluralize("fox") == "foxes"
    assert pluralize("bus") == "buses"
    assert pluralize("church") == "churches"


# ---------------------------------------------------------------------------
# 4. Two-weapon "(alternate weapon)" marker
# ---------------------------------------------------------------------------

def test_two_weapon_alternate_marker_when_toggled():
    """When state.combat.two_weapon=True and alt_slot=X, slot X gets the marker."""
    from Nethax.nethax.obs.inv_strs import build_inv_strs

    type_id = _find_type_id_by_name("dagger")
    items = _set_slot(
        _empty_items(), 2,
        type_id=type_id, category=int(ObjectClass.WEAPON_CLASS),
        identified=True, buc_status=BUCStatus.UNCURSED,
    )
    inv = _make_inv(items, alt_slot=2)
    state = _FakeState(inv, two_weapon=True)
    result = build_inv_strs(state)
    s = _decode(result[2])
    assert "(alternate weapon)" in s, f"expected '(alternate weapon)' in: {s!r}"


def test_no_alternate_marker_when_two_weapon_off():
    """With two_weapon=False, the marker is suppressed even if alt_slot is set."""
    from Nethax.nethax.obs.inv_strs import build_inv_strs

    type_id = _find_type_id_by_name("dagger")
    items = _set_slot(
        _empty_items(), 2,
        type_id=type_id, category=int(ObjectClass.WEAPON_CLASS),
        identified=True, buc_status=BUCStatus.UNCURSED,
    )
    inv = _make_inv(items, alt_slot=2)
    state = _FakeState(inv, two_weapon=False)
    result = build_inv_strs(state)
    s = _decode(result[2])
    assert "(alternate weapon)" not in s, (
        f"should not show '(alternate weapon)' when two_weapon=False: {s!r}"
    )


def test_alt_marker_only_on_alt_slot():
    """The marker appears on the alt slot only, not on other occupied slots."""
    from Nethax.nethax.obs.inv_strs import build_inv_strs

    type_id = _find_type_id_by_name("dagger")
    items = _empty_items()
    items = _set_slot(items, 0, type_id=type_id, category=int(ObjectClass.WEAPON_CLASS),
                      identified=True, buc_status=BUCStatus.UNCURSED)
    items = _set_slot(items, 1, type_id=type_id, category=int(ObjectClass.WEAPON_CLASS),
                      identified=True, buc_status=BUCStatus.UNCURSED)
    inv = _make_inv(items, alt_slot=1)
    state = _FakeState(inv, two_weapon=True)
    result = build_inv_strs(state)
    s0 = _decode(result[0])
    s1 = _decode(result[1])
    assert "(alternate weapon)" not in s0, f"slot 0 should not be marked: {s0!r}"
    assert "(alternate weapon)" in s1, f"slot 1 should be marked: {s1!r}"
