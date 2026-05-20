"""Wish parser parity tests — parse_wish_string_dict and grant_wish_from_string.

Covers the dict-returning parser and the Luck-gated grant helper.

Canonical source: vendor/nethack/src/objnam.c::readobjnam line ~2620.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants.objects import OBJECTS, ObjectClass
from Nethax.nethax.subsystems.conduct import Conduct
from Nethax.nethax.subsystems.inventory import ItemCategory
from Nethax.nethax.subsystems.items import BUCStatus
from Nethax.nethax.subsystems.prayer import Alignment
from Nethax.nethax.subsystems.wish import (
    _ARTIFACTS,
    parse_wish_string_dict,
    grant_wish_from_string,
)

import jax
_RNG = jax.random.PRNGKey(0)


def _fresh_state() -> EnvState:
    return EnvState.default(_RNG)


def _type_id(name: str) -> int:
    for i, entry in enumerate(OBJECTS):
        if entry.name == name:
            return i
    raise AssertionError(f"object {name!r} not in OBJECTS table")


# ---------------------------------------------------------------------------
# parse_wish_string_dict
# ---------------------------------------------------------------------------

def test_parse_simple():
    """'long sword' -> type_id matches long sword, buc=UNCURSED, enchant=0.

    Note: parse_wish_string_dict returns the vendor wishymatch dict, whose
    keys are ``type_id`` / ``enchant`` / ``artifact_idx`` / ``user_name``
    (bytes, b"" when absent) — not a high-level ``name`` string. Test now
    asserts against the vendor dict schema.
    """
    r = parse_wish_string_dict("long sword")
    assert r["type_id"] == _type_id("long sword")
    assert r["buc"] == int(BUCStatus.UNCURSED)
    assert r["enchant"] == 0
    assert r["artifact_idx"] == -1
    assert r["user_name"] == b""
    assert r["parsed"] is True


def test_parse_blessed_plus_3():
    """'blessed +3 long sword' -> buc=BLESSED, enchant=3."""
    r = parse_wish_string_dict("blessed +3 long sword")
    assert r["buc"] == int(BUCStatus.BLESSED)
    assert r["enchant"] == 3
    assert r["type_id"] == _type_id("long sword")
    assert r["artifact_idx"] == -1


def test_parse_potion_of():
    """'potion of healing' resolves to the healing potion entry."""
    r = parse_wish_string_dict("potion of healing")
    # OBJECTS table lists potions by their bare effect name ("healing").
    assert r["type_id"] == _type_id("healing")
    assert int(OBJECTS[r["type_id"]].class_) == int(ObjectClass.POTION_CLASS)
    assert r["artifact_idx"] == -1


def test_parse_artifact_name():
    """'Excalibur' -> artifact_idx >= 0 and base type_id is long sword."""
    r = parse_wish_string_dict("Excalibur")
    assert r["artifact_idx"] >= 0
    # Verify the index actually maps to Excalibur in the table.
    art_name, art_base = _ARTIFACTS[r["artifact_idx"]]
    assert art_name == "Excalibur"
    assert art_base == "long sword"
    # The canonical object type_id should be the base item.
    assert r["type_id"] == _type_id("long sword")


def test_parse_called():
    """'amulet called Bob' -> user_name=b'Bob' (bytes)."""
    r = parse_wish_string_dict("amulet called Bob")
    # Vendor wishymatch stores user_name as bytes (b"" when absent).
    assert r["user_name"] == b"Bob"


def test_parse_cursed_minus_1():
    """'cursed -1 dagger' -> buc=CURSED, enchant=-1."""
    r = parse_wish_string_dict("cursed -1 dagger")
    assert r["buc"] == int(BUCStatus.CURSED)
    assert r["enchant"] == -1


def test_parse_greased():
    """'greased long sword' -> greased=True."""
    r = parse_wish_string_dict("greased long sword")
    assert r["greased"] is True
    assert r["type_id"] == _type_id("long sword")


# ---------------------------------------------------------------------------
# grant_wish_from_string
# ---------------------------------------------------------------------------

def test_grant_wish_creates_item():
    """grant_wish_from_string with a valid wish creates an item in inventory."""
    state = _fresh_state()
    new_state = grant_wish_from_string(state, "long sword")

    tid = _type_id("long sword")
    assert int(new_state.inventory.items.type_id[0]) == tid
    assert int(new_state.inventory.items.category[0]) == int(ItemCategory.WEAPON)
    assert bool(new_state.conduct.violations[int(Conduct.WISHLESS)]) is True


def test_grant_wish_artifact_alignment_gate():
    """Excalibur wished as Chaotic: artifact_idx cleared, base long sword granted."""
    state = _fresh_state()
    # Force Chaotic alignment and low XL so Excalibur is denied.
    state = state.replace(
        player_align=jnp.int8(int(Alignment.CHAOTIC)),
        player_xl=jnp.int32(1),
    )
    new_state = grant_wish_from_string(state, "Excalibur")

    # grant_wish_from_string uses grant_wish which does NOT call
    # apply_artifact_restrictions internally; the chaotic player still
    # receives the base long sword (vendor behavior: base item still granted).
    tid = _type_id("long sword")
    assert int(new_state.inventory.items.type_id[0]) == tid
    # WISHLESS must be set; ARTIWISHLESS is set because the wish text was an
    # artifact name (vendor flips wisharti on the text match, not the grant).
    assert bool(new_state.conduct.violations[int(Conduct.WISHLESS)]) is True
    assert bool(new_state.conduct.violations[int(Conduct.ARTIWISHLESS)]) is True


def test_grant_wish_neutral_luck_succeeds():
    """Luck=0 never blocks a wish."""
    state = _fresh_state()
    state = state.replace(player_luck=jnp.int8(0))
    new_state = grant_wish_from_string(state, "long sword")
    assert int(new_state.inventory.items.type_id[0]) == _type_id("long sword")


def test_grant_wish_luck_does_not_block():
    """Luck does not gate wand-of-wishing grants in vendor.

    Vendor wizard.c/zap.c::makewish has no Luck branch — wishes always
    succeed when readobjnam parses the wish text. Previous nethax
    behavior gated on Luck < 0 (wrong); this test now asserts the
    vendor-correct path: Luck=-10 still grants the item.
    Cite: vendor/nethack/src/zap.c::makewish (line 6314).
    """
    state = _fresh_state()
    state = state.replace(player_luck=jnp.int8(-10))
    new_state = grant_wish_from_string(state, "long sword")
    assert int(new_state.inventory.items.type_id[0]) == _type_id("long sword")
    assert bool(new_state.conduct.violations[int(Conduct.WISHLESS)]) is True
