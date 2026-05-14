"""Wave 6 Phase B wish subsystem tests.

Covers:
  - parse_wish_string: BUC prefix, enchantment prefix, artifact lookup,
    unknown name handling.
  - grant_wish: inventory placement, ground-fallback on full inventory,
    WISHLESS / ARTIWISHLESS conduct flips.
  - handle_wand_of_wishing: canonical wand-of-wishing default wish.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants.objects import OBJECTS, OBJECT_NAME_ALIASES, ObjectClass
from Nethax.nethax.subsystems.conduct import Conduct
from Nethax.nethax.subsystems.inventory import ItemCategory, MAX_INVENTORY_SLOTS
from Nethax.nethax.subsystems.items import BUCStatus
from Nethax.nethax.subsystems.wish import (
    grant_wish,
    handle_wand_of_wishing,
    parse_wish_string,
)


_RNG = jax.random.PRNGKey(0)


def _fresh_state() -> EnvState:
    return EnvState.default(_RNG)


def _type_id(name: str, cls: ObjectClass | None = None) -> int:
    """Resolve an object name to its OBJECTS index.

    When ``cls`` is provided, prefer the entry whose ``class_`` matches —
    necessary because some bare names (e.g. ``identify``) exist in both
    the SCROLL_CLASS and SPBOOK_CLASS tables.  Mirrors vendor wishymatch's
    class-aware resolution path (objnam.c::wishymatch).
    """
    if cls is not None:
        target = int(cls)
        for i, entry in enumerate(OBJECTS):
            if entry.name == name and int(entry.class_) == target:
                return i
    for i, entry in enumerate(OBJECTS):
        if entry.name == name:
            return i
    raise AssertionError(f"object {name!r} not in OBJECTS table")


# ---------------------------------------------------------------------------
# parse_wish_string
# ---------------------------------------------------------------------------

def test_parse_wish_string_unknown_returns_negative():
    """Unknown name returns type_id == -1."""
    cat, tid, buc, ench, art = parse_wish_string(b"flux capacitor")
    assert tid == -1
    assert art == -1


def test_parse_wish_string_blessed_prefix():
    """'blessed scroll of identify' parses BUC=BLESSED."""
    # Wave 6 parity-fix: updated to match vendor/nethack/src/objnam.c:3243
    # (wishymatch resolves the class-prefixed alias 'scroll of identify' to
    # the bare-name canonical object 'identify' in the SCROLL_CLASS table;
    # the same bare name exists in SPBOOK_CLASS for 'spellbook of identify'
    # so the lookup must be class-aware).
    cat, tid, buc, ench, art = parse_wish_string(b"blessed scroll of identify")
    assert tid == _type_id("identify", ObjectClass.SCROLL_CLASS)
    assert buc == int(BUCStatus.BLESSED)
    assert art == -1


def test_parse_wish_string_enchantment_prefix():
    """'+3 long sword' parses enchantment=3."""
    cat, tid, buc, ench, art = parse_wish_string(b"+3 long sword")
    assert tid == _type_id("long sword")
    assert ench == 3


def test_parse_wish_string_artifact():
    """'Excalibur' resolves to artifact_idx=0 with long-sword base."""
    cat, tid, buc, ench, art = parse_wish_string(b"Excalibur")
    assert art >= 0
    assert tid == _type_id("long sword")
    assert cat == int(ObjectClass.WEAPON_CLASS)


# ---------------------------------------------------------------------------
# grant_wish
# ---------------------------------------------------------------------------

def test_grant_wish_creates_item_in_inventory():
    """grant_wish 'long sword' populates inventory slot 0 with that item."""
    state = _fresh_state()
    new_state = grant_wish(state, _RNG, b"long sword")

    assert int(new_state.inventory.items.type_id[0]) == _type_id("long sword")
    assert int(new_state.inventory.items.category[0]) == int(ItemCategory.WEAPON)
    assert int(new_state.inventory.items.quantity[0]) == 1


def test_grant_wish_blessed_status_parsed():
    """grant_wish 'blessed scroll of identify' sets BUC=BLESSED."""
    # Wave 6 parity-fix: updated to match vendor/nethack/src/objnam.c:3243
    # ('scroll of identify' resolves to the SCROLL_CLASS 'identify' entry,
    # not the SPBOOK_CLASS one that shares the bare name).
    state = _fresh_state()
    new_state = grant_wish(state, _RNG, b"blessed scroll of identify")

    assert int(new_state.inventory.items.buc_status[0]) == int(BUCStatus.BLESSED)
    assert int(new_state.inventory.items.type_id[0]) == _type_id(
        "identify", ObjectClass.SCROLL_CLASS,
    )


def test_grant_wish_enchantment_parsed():
    """grant_wish '+3 long sword' stores enchantment=3."""
    state = _fresh_state()
    new_state = grant_wish(state, _RNG, b"+3 long sword")

    assert int(new_state.inventory.items.enchantment[0]) == 3
    assert int(new_state.inventory.items.type_id[0]) == _type_id("long sword")


def test_grant_wish_violates_wishless_conduct():
    """Granting any wish flips WISHLESS."""
    state = _fresh_state()
    assert bool(state.conduct.violations[int(Conduct.WISHLESS)]) is False

    new_state = grant_wish(state, _RNG, b"long sword")
    assert bool(new_state.conduct.violations[int(Conduct.WISHLESS)]) is True
    # Non-artifact wish must NOT flip ARTIWISHLESS.
    assert bool(new_state.conduct.violations[int(Conduct.ARTIWISHLESS)]) is False


def test_grant_wish_artifact_violates_both_conducts():
    """Wishing for Excalibur flips both WISHLESS and ARTIWISHLESS."""
    state = _fresh_state()
    new_state = grant_wish(state, _RNG, b"Excalibur")

    assert bool(new_state.conduct.violations[int(Conduct.WISHLESS)]) is True
    assert bool(new_state.conduct.violations[int(Conduct.ARTIWISHLESS)]) is True
    # The base object should be a long sword.
    assert int(new_state.inventory.items.type_id[0]) == _type_id("long sword")


def test_grant_wish_unknown_name_returns_unchanged():
    """Unknown wish leaves state untouched (no item, no conduct flip)."""
    state = _fresh_state()
    new_state = grant_wish(state, _RNG, b"flux capacitor")

    assert int(new_state.inventory.items.category[0]) == 0
    assert bool(new_state.conduct.violations[int(Conduct.WISHLESS)]) is False
    assert bool(new_state.conduct.violations[int(Conduct.ARTIWISHLESS)]) is False


def test_wish_full_inventory_drops_on_floor():
    """When the inventory is full, the wished item lands on the ground."""
    state = _fresh_state()
    state = state.replace(player_pos=jnp.array([4, 7], dtype=jnp.int16))

    # Fill every inventory slot with a placeholder (non-zero category).
    items = state.inventory.items
    full_categories = jnp.full((MAX_INVENTORY_SLOTS,), int(ItemCategory.WEAPON),
                               dtype=jnp.int8)
    items = items.replace(
        category=full_categories,
        quantity=jnp.ones_like(items.quantity),
    )
    state = state.replace(inventory=state.inventory.replace(items=items))

    new_state = grant_wish(state, _RNG, b"long sword")

    b = int(new_state.dungeon.current_branch)
    lv = int(new_state.dungeon.current_level) - 1
    r, c = 4, 7
    assert int(new_state.ground_items.type_id[b, lv, r, c, 0]) == _type_id("long sword")
    # Conduct still flipped.
    assert bool(new_state.conduct.violations[int(Conduct.WISHLESS)]) is True


# ---------------------------------------------------------------------------
# Wand of wishing
# ---------------------------------------------------------------------------

def test_wand_of_wishing_zap_grants_wish():
    """handle_wand_of_wishing grants the canonical canned wish and sets conduct."""
    state = _fresh_state()
    new_state = handle_wand_of_wishing(state, _RNG)

    # Default wish = gray dragon scale mail; check inventory slot 0 was filled.
    assert int(new_state.inventory.items.type_id[0]) == _type_id("gray dragon scale mail")
    assert int(new_state.inventory.items.category[0]) == int(ItemCategory.ARMOR)
    assert int(new_state.inventory.items.enchantment[0]) == 3
    assert int(new_state.inventory.items.buc_status[0]) == int(BUCStatus.BLESSED)
    assert bool(new_state.conduct.violations[int(Conduct.WISHLESS)]) is True


def test_wand_of_wishing_override_wish():
    """handle_wand_of_wishing accepts an override wish_string."""
    state = _fresh_state()
    new_state = handle_wand_of_wishing(state, _RNG, b"long sword")
    assert int(new_state.inventory.items.type_id[0]) == _type_id("long sword")
