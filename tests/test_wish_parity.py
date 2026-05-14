"""Wave 6 Phase B+ wish parity tests.

Covers the full vendor wishymatch behavior:
  - Fuzzy abbreviation match (longsword, gdsm).
  - Plural normalization (scrolls -> scroll).
  - "the " artifact prefix.
  - Multi-modifier combos (blessed greased rustproof +N name).
  - "named X" suffix wiring user_names.
  - Artifact SPFX_RESTR alignment / XL gating.
  - Excalibur->Stormbringer chaotic substitution.

Cite: vendor/nethack/src/objnam.c::wishymatch + readobjnam, artifact.c::spec_applies.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants.objects import OBJECTS, ObjectClass
from Nethax.nethax.subsystems.conduct import Conduct
from Nethax.nethax.subsystems.inventory import ItemCategory
from Nethax.nethax.subsystems.items import BUCStatus
from Nethax.nethax.subsystems.prayer import Alignment
from Nethax.nethax.subsystems.wish import (
    apply_artifact_restrictions,
    grant_wish,
    wishymatch,
)


_RNG = jax.random.PRNGKey(0)


def _fresh_state() -> EnvState:
    return EnvState.default(_RNG)


def _state_with(align: int, xl: int) -> EnvState:
    """Fresh state with a chosen alignment + experience level."""
    s = _fresh_state()
    return s.replace(
        player_align=jnp.int8(align),
        player_xl=jnp.int32(xl),
    )


def _object_index(name: str, cls: ObjectClass | None = None) -> int:
    """Find an OBJECTS index by canonical bare name (optionally class-scoped).

    Mirrors vendor wishymatch's class-aware resolution (objnam.c::wishymatch):
    bare names can collide across classes (e.g. 'identify' exists in both
    SCROLL_CLASS and SPBOOK_CLASS) and the wish parser disambiguates by the
    "scroll of " / "spellbook of " prefix.
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
# Fuzzy abbreviation
# ---------------------------------------------------------------------------

def test_fuzzy_longsword_resolves_long_sword():
    """'longsword' (no space) resolves to the 'long sword' OBJECT entry."""
    out = wishymatch(b"longsword")
    assert out["parsed"] is True
    assert out["type_id"] == _object_index("long sword")
    assert out["category"] == int(ObjectClass.WEAPON_CLASS)
    assert out["artifact_idx"] == -1


def test_fuzzy_gdsm_resolves_gray_dragon_scale_mail():
    """'gdsm' (first-letter abbreviation) resolves to gray dragon scale mail."""
    out = wishymatch(b"gdsm")
    assert out["parsed"] is True
    assert out["type_id"] == _object_index("gray dragon scale mail")
    assert out["category"] == int(ObjectClass.ARMOR_CLASS)


# ---------------------------------------------------------------------------
# Plural normalization
# ---------------------------------------------------------------------------

def test_plural_scrolls_resolves_singular():
    """'scrolls of identify' is singularized to 'scroll of identify'."""
    out = wishymatch(b"scrolls of identify")
    assert out["parsed"] is True
    # OBJECT_NAME_ALIASES maps 'scroll of identify' -> the SCROLL_CLASS
    # 'identify' index (the SPBOOK_CLASS 'identify' is a different entry).
    expected = _object_index("identify", ObjectClass.SCROLL_CLASS)
    assert out["type_id"] == expected
    assert out["category"] == int(ObjectClass.SCROLL_CLASS)


# ---------------------------------------------------------------------------
# "the" prefix
# ---------------------------------------------------------------------------

def test_the_prefix_resolves_artifact():
    """'the Eye of the Aethiopica' resolves to the Aethiopica artifact."""
    out = wishymatch(b"the Eye of the Aethiopica")
    assert out["parsed"] is True
    assert out["artifact_idx"] >= 0
    # Artifact 21 in our table is Eye of the Aethiopica (amulet of ESP base).
    assert out["artifact_idx"] == 21


# ---------------------------------------------------------------------------
# Multi-modifier combos
# ---------------------------------------------------------------------------

def test_multi_modifier_combo():
    """'blessed greased rustproof +3 long sword' parses every field."""
    out = wishymatch(b"blessed greased rustproof +3 long sword")
    assert out["parsed"] is True
    assert out["buc"] == int(BUCStatus.BLESSED)
    assert out["greased"] is True
    assert out["erodeproof"] is True
    assert out["enchant"] == 3
    assert out["type_id"] == _object_index("long sword")


def test_erodeproof_modifier_sets_flag():
    """'rustproof long sword' sets erodeproof=True with default BUC."""
    out = wishymatch(b"rustproof long sword")
    assert out["parsed"] is True
    assert out["erodeproof"] is True
    assert out["greased"] is False
    assert out["type_id"] == _object_index("long sword")


# ---------------------------------------------------------------------------
# Named suffix
# ---------------------------------------------------------------------------

def test_named_modifier_sets_user_name():
    """'+1 elven dagger named Sting' carries the user-name through."""
    out = wishymatch(b"+1 elven dagger named Sting")
    assert out["parsed"] is True
    assert out["enchant"] == 1
    assert out["type_id"] == _object_index("elven dagger")
    assert out["user_name"] == b"Sting"

    # And grant_wish should write the name into inventory.user_names[0].
    state = _fresh_state()
    new_state = grant_wish(state, _RNG, b"+1 elven dagger named Sting")
    name_row = bytes(int(b) & 0xFF for b in new_state.inventory.user_names[0])
    name = name_row.split(b"\x00", 1)[0]
    assert name == b"Sting"


# ---------------------------------------------------------------------------
# Unknown name handling
# ---------------------------------------------------------------------------

def test_unknown_name_returns_unparsed():
    """Garbage wish text returns parsed=False with sentinel fields."""
    out = wishymatch(b"flux capacitor of mystery")
    assert out["parsed"] is False
    assert out["type_id"] == -1
    assert out["artifact_idx"] == -1


# ---------------------------------------------------------------------------
# Artifact SPFX alignment / XL restrictions
# ---------------------------------------------------------------------------

def test_excalibur_for_lawful_xl5_grants():
    """Lawful XL5 player wishing Excalibur passes spec_applies."""
    state = _state_with(int(Alignment.LAWFUL), 5)
    parsed = wishymatch(b"Excalibur")
    gated = apply_artifact_restrictions(parsed, int(state.player_align),
                                        int(state.player_xl))
    assert gated["artifact_idx"] == 0
    assert gated["type_id"] == _object_index("long sword")


def test_excalibur_for_chaotic_denied():
    """Chaotic player wishing Excalibur is rerouted to Stormbringer."""
    state = _state_with(int(Alignment.CHAOTIC), 5)
    parsed = wishymatch(b"Excalibur")
    gated = apply_artifact_restrictions(parsed, int(state.player_align),
                                        int(state.player_xl))
    # Stormbringer (artifact idx 2) granted instead.
    assert gated["artifact_idx"] == 2
    assert gated["type_id"] == _object_index("runesword")


def test_artifact_spfx_alignment_restriction():
    """Neutral player wishing Excalibur loses the artifact, keeps long sword."""
    state = _state_with(int(Alignment.NEUTRAL), 10)
    parsed = wishymatch(b"Excalibur")
    gated = apply_artifact_restrictions(parsed, int(state.player_align),
                                        int(state.player_xl))
    assert gated["artifact_idx"] == -1
    assert gated["type_id"] == _object_index("long sword")
