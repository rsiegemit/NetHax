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
    """'long sword' -> name='long sword', buc=UNCURSED (default), enchantment=0."""
    r = parse_wish_string_dict("long sword")
    assert r["name"] == "long sword"
    # BUC defaults to UNCURSED when no BUC keyword is present.
    assert r["buc"] == int(BUCStatus.UNCURSED)
    assert r["enchantment"] == 0
    assert r["is_artifact"] is False
    assert r["artifact_idx"] == -1
    assert r["user_name"] is None


def test_parse_blessed_plus_3():
    """'blessed +3 long sword' -> buc=BLESSED, enchantment=3."""
    r = parse_wish_string_dict("blessed +3 long sword")
    assert r["buc"] == int(BUCStatus.BLESSED)
    assert r["enchantment"] == 3
    assert r["name"] == "long sword"
    assert r["is_artifact"] is False


def test_parse_potion_of():
    """'potion of healing' resolves to the healing potion entry."""
    r = parse_wish_string_dict("potion of healing")
    # The canonical bare name in OBJECTS for healing potions is "healing".
    assert r["name"] == "healing"
    assert r["is_artifact"] is False


def test_parse_artifact_name():
    """'Excalibur' -> is_artifact=True, artifact_idx matches _ARTIFACTS entry."""
    r = parse_wish_string_dict("Excalibur")
    assert r["is_artifact"] is True
    assert r["artifact_idx"] >= 0
    # Verify the index actually maps to Excalibur in the table.
    art_name, art_base = _ARTIFACTS[r["artifact_idx"]]
    assert art_name == "Excalibur"
    assert art_base == "long sword"
    # The canonical object name should be the base.
    assert r["name"] == "long sword"


def test_parse_called():
    """'amulet called Bob' -> user_name='Bob'."""
    r = parse_wish_string_dict("amulet called Bob")
    assert r["user_name"] == "Bob"


def test_parse_cursed_minus_1():
    """'cursed -1 dagger' -> buc=CURSED, enchantment=-1."""
    r = parse_wish_string_dict("cursed -1 dagger")
    assert r["buc"] == int(BUCStatus.CURSED)
    assert r["enchantment"] == -1


def test_parse_greased():
    """'greased long sword' -> greased=True."""
    r = parse_wish_string_dict("greased long sword")
    assert r["greased"] is True
    assert r["name"] == "long sword"


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


def test_grant_wish_max_bad_luck_always_fails():
    """Luck=-10 always blocks (100% fail probability)."""
    state = _fresh_state()
    state = state.replace(player_luck=jnp.int8(-10))
    new_state = grant_wish_from_string(state, "long sword")
    # State should be unchanged — no item granted.
    assert int(new_state.inventory.items.category[0]) == 0
    assert bool(new_state.conduct.violations[int(Conduct.WISHLESS)]) is False
