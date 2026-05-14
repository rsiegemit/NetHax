"""Full object table vendor parity tests (Wave 6 Phase B+, agent #63).

Ground truth: vendor/nle/src/objects.c::objects[] (430 named entries +
23 None-named appearance slots = 453 total, matching live NLE NUM_OBJECTS).

Indices in this file are vendor canonical positions (`OBJ_*` enum values).
"""
from __future__ import annotations

import pytest

from Nethax.nethax.constants.objects import (
    OBJECT_NAME_ALIASES,
    OBJECTS,
    NUM_OBJECTS,
    ObjectClass,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_by_name(name: str):
    """Return (index, ObjectEntry) for the given canonical name, or (None, None)."""
    for i, o in enumerate(OBJECTS):
        if o.name == name:
            return i, o
    return None, None


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def test_num_objects_is_453():
    """Total object count must match live NLE binary == vendor objects.c length.

    Wave 6 parity-fix: updated to match vendor/nle/src/objects.c (430 named
    entries + 23 None-named shuffled-appearance slots = 453 total).
    """
    assert NUM_OBJECTS == 453, f"expected 453, got {NUM_OBJECTS}"
    assert len(OBJECTS) == 453, f"OBJECTS tuple length {len(OBJECTS)} != 453"


def test_strange_object_is_index_zero():
    """vendor/nle/src/objects.c:82 — OBJECT(OBJ("strange object", None), ...)."""
    o = OBJECTS[0]
    assert o.name == "strange object"
    assert o.class_ == ObjectClass.ILLOBJ_CLASS


# ---------------------------------------------------------------------------
# Removed-extras: assert formerly-bogus entries are gone
# ---------------------------------------------------------------------------


def test_no_generic_filler_entries():
    """The 17 'generic *' entries from nethack/include/objects.h:86-102 are NOT
    in vendor/nle/src/objects.c and must NOT be in our table."""
    bogus = {
        "generic strange", "generic weapon", "generic armor", "generic ring",
        "generic amulet", "generic tool", "generic food", "generic potion",
        "generic scroll", "generic spellbook", "generic wand", "generic coin",
        "generic gem", "generic large rock", "generic iron ball",
        "generic iron chain", "generic venom",
    }
    present = {o.name for o in OBJECTS if o.name in bogus}
    assert not present, f"unexpected 'generic *' entries present: {present}"


def test_no_deferred_dragon_variants():
    """vendor/nle/src/objects.c:380-396 — shimmering / gold dragon scale variants
    are wrapped in #if 0 DEFERRED and must NOT appear in our table."""
    deferred = {
        "shimmering dragon scale mail", "shimmering dragon scales",
        "gold dragon scale mail", "gold dragon scales",
    }
    present = {o.name for o in OBJECTS if o.name in deferred}
    assert not present, f"unexpected deferred dragon variants: {present}"


# ---------------------------------------------------------------------------
# Added-missing: previously-absent entries that vendor expects
# ---------------------------------------------------------------------------


def test_acid_venom_added():
    """vendor/nle/src/objects.c:1150 — OBJECT(OBJ("acid venom", "splash of venom"), ...)."""
    i, o = _find_by_name("acid venom")
    assert o is not None, "'acid venom' missing from OBJECTS"
    assert o.class_ == ObjectClass.VENOM_CLASS
    assert o.description == "splash of venom"


def test_blinding_venom_added():
    """vendor/nle/src/objects.c:1147 — OBJECT(OBJ("blinding venom", "splash of venom"), ...)."""
    i, o = _find_by_name("blinding venom")
    assert o is not None, "'blinding venom' missing from OBJECTS"
    assert o.class_ == ObjectClass.VENOM_CLASS
    assert o.description == "splash of venom"


def test_novel_added():
    """vendor/nle/src/objects.c:1003 — OBJECT(OBJ("novel", "paperback"), ..., SPBOOK_CLASS, ...)."""
    i, o = _find_by_name("novel")
    assert o is not None, "'novel' missing from OBJECTS"
    assert o.class_ == ObjectClass.SPBOOK_CLASS
    assert o.description == "paperback"


def test_strange_object_added():
    """vendor/nle/src/objects.c:82 — OBJECT(OBJ("strange object", None), ..., ILLOBJ_CLASS, ...)."""
    i, o = _find_by_name("strange object")
    assert o is not None, "'strange object' missing from OBJECTS"
    assert i == 0, f"strange object must be index 0 (vendor canonical), got {i}"
    assert o.class_ == ObjectClass.ILLOBJ_CLASS


def test_gold_piece_added():
    """vendor/nle/src/objects.c:1054 — COIN("gold piece", 1000, GOLD, 1) → COIN_CLASS."""
    i, o = _find_by_name("gold piece")
    assert o is not None, "'gold piece' missing from OBJECTS"
    assert o.class_ == ObjectClass.COIN_CLASS


def test_huge_chunk_of_meat_added():
    """vendor/nle/src/objects.c:744 — FOOD("huge chunk of meat", ...) → FOOD_CLASS."""
    i, o = _find_by_name("huge chunk of meat")
    assert o is not None, "'huge chunk of meat' missing from OBJECTS"
    assert o.class_ == ObjectClass.FOOD_CLASS


def test_meat_ring_added():
    """vendor/nle/src/objects.c:746 — OBJECT(OBJ("meat ring", None), ..., FOOD_CLASS, ...)."""
    i, o = _find_by_name("meat ring")
    assert o is not None, "'meat ring' missing from OBJECTS"
    # meat ring is FOOD_CLASS in vendor, despite the name
    assert o.class_ == ObjectClass.FOOD_CLASS


# ---------------------------------------------------------------------------
# No dual-named potions / scrolls — canonical bare names only
# ---------------------------------------------------------------------------


def test_no_dual_named_potions():
    """vendor stores bare potion names ('healing'); the 'potion of ' prefix is
    prepended at render time by objnam.c::xname. The bare name must be present
    and the prefixed form must be absent as a real OBJECTS entry."""
    bare_idx, bare = _find_by_name("healing")
    assert bare is not None, "'healing' missing"
    assert bare.class_ == ObjectClass.POTION_CLASS
    dup_idx, dup = _find_by_name("potion of healing")
    assert dup is None, "'potion of healing' duplicate entry exists"
    # Alias map must still resolve "potion of healing" → bare index.
    assert OBJECT_NAME_ALIASES.get("potion of healing") == bare_idx


def test_no_dual_named_scrolls():
    """Same as potions but for scrolls."""
    bare_idx, bare = _find_by_name("identify")
    assert bare is not None, "'identify' missing"
    assert bare.class_ == ObjectClass.SCROLL_CLASS
    dup_idx, dup = _find_by_name("scroll of identify")
    assert dup is None, "'scroll of identify' duplicate entry exists"
    assert OBJECT_NAME_ALIASES.get("scroll of identify") == bare_idx


# ---------------------------------------------------------------------------
# Per-field vendor parity
# ---------------------------------------------------------------------------


def test_long_sword_weight_40():
    """vendor/nle/src/objects.c:206-207 — WEAPON("long sword", ..., 50, 40, 15, 8, 12, ...).

    Wave 6 parity-fix: brief said weight=30, but vendor canonical value is 40.
    """
    i, o = _find_by_name("long sword")
    assert o is not None, "'long sword' missing"
    assert o.weight == 40, f"expected weight 40 (vendor), got {o.weight}"
    assert o.cost == 15, f"expected cost 15 (vendor), got {o.cost}"


def test_long_sword_damage_d8_vs_small_d12_vs_large():
    """vendor/nle/src/objects.c:207 — long sword sdam=8, ldam=12 (each is 1dN)."""
    i, o = _find_by_name("long sword")
    assert o is not None
    assert o.sdam == (1, 8), f"expected sdam (1,8), got {o.sdam}"
    assert o.ldam == (1, 12), f"expected ldam (1,12), got {o.ldam}"


def test_arrow_at_vendor_index_1():
    """vendor/nle/src/objects.c:113 — PROJECTILE("arrow", None, 1, 55, 1, 2, 6, 6, 0, ...).

    Wave 6 parity-fix: vendor canonical index 1 is "arrow" (after strange object).
    """
    o = OBJECTS[1]
    assert o.name == "arrow"
    assert o.class_ == ObjectClass.WEAPON_CLASS
    assert o.weight == 1
    assert o.cost == 2
    assert o.sdam == (1, 6)
    assert o.ldam == (1, 6)


def test_orcish_arrow_at_vendor_index_3():
    """vendor/nle/src/objects.c:117-118 — PROJECTILE("orcish arrow", "crude arrow", ...).

    Wave 6 parity-fix: vendor index 3 is "orcish arrow" (was wrongly "silver
    arrow" in our pre-Wave-6 table that omitted "strange object" at index 0).
    """
    o = OBJECTS[3]
    assert o.name == "orcish arrow"
    assert o.description == "crude arrow"
    assert o.class_ == ObjectClass.WEAPON_CLASS


# ---------------------------------------------------------------------------
# Unique items
# ---------------------------------------------------------------------------


def test_amulet_of_yendor_class_amulet_and_unique():
    """vendor/nle/src/objects.c:626-627 — OBJECT(OBJ("Amulet of Yendor", ...), ...,
    AMULET_CLASS, ...).  The Amulet of Yendor is the only entry with that name."""
    matches = [(i, o) for i, o in enumerate(OBJECTS) if o.name == "Amulet of Yendor"]
    assert len(matches) == 1, f"expected exactly 1 'Amulet of Yendor', got {len(matches)}"
    i, o = matches[0]
    assert o.class_ == ObjectClass.AMULET_CLASS


def test_holy_water_class_potion():
    """vendor/nle/src/objects.c:820 — POTION("water", "clear", ...).  Holy/unholy
    water is the same object as plain water (with cursed/blessed flag).  The
    canonical entry is class POTION_CLASS."""
    i, o = _find_by_name("water")
    assert o is not None, "'water' missing"
    assert o.class_ == ObjectClass.POTION_CLASS


# ---------------------------------------------------------------------------
# Dragon armor (chromatic variants)
# ---------------------------------------------------------------------------


def test_dragon_scale_mail_chromatic_variants():
    """vendor/nle/src/objects.c:378-389 — DRGN_ARMR(...) declares scale mail in
    9 chromatic variants (gray, silver, red, white, orange, black, blue, green,
    yellow).  The 'shimmering' (line 381) is #if 0 deferred, NOT included.

    Wave 6 parity-fix: brief mentioned 5 chromatic variants, but vendor reality
    is 9 once shimmering/gold (deferred) are excluded.
    """
    expected = {
        "gray dragon scale mail", "silver dragon scale mail",
        "red dragon scale mail", "white dragon scale mail",
        "orange dragon scale mail", "black dragon scale mail",
        "blue dragon scale mail", "green dragon scale mail",
        "yellow dragon scale mail",
    }
    actual = {o.name for o in OBJECTS if o.name and o.name.endswith("dragon scale mail")}
    assert actual == expected, (
        f"chromatic dragon scale mail mismatch.\n"
        f"  missing: {expected - actual}\n"
        f"  extra:   {actual - expected}"
    )


def test_dragon_scales_chromatic_variants():
    """vendor/nle/src/objects.c:393-404 — DRGN_ARMR scales (non-mail) in the
    same 9 chromatic variants.  Wave 6 parity-fix: ensures gold/shimmering
    are excluded (they are #if 0 DEFERRED in vendor)."""
    expected = {
        "gray dragon scales", "silver dragon scales",
        "red dragon scales", "white dragon scales",
        "orange dragon scales", "black dragon scales",
        "blue dragon scales", "green dragon scales",
        "yellow dragon scales",
    }
    actual = {o.name for o in OBJECTS if o.name and o.name.endswith("dragon scales")}
    assert actual == expected, (
        f"chromatic dragon scales mismatch.\n"
        f"  missing: {expected - actual}\n"
        f"  extra:   {actual - expected}"
    )


# ---------------------------------------------------------------------------
# Sanity: appearance-slot None entries
# ---------------------------------------------------------------------------


def test_appearance_slot_entries_have_descriptions():
    """vendor/nle/src/objects.c:855-875 — extra SCROLL appearance slots with
    name=None and a fixed description ('FOOBIE BLETCH', 'TEMOV', etc.).  These
    are real objects[] entries in NLE; OBJECTS must include them so that
    inv_strs.py / glyph projection stays index-aligned with live NLE."""
    none_named = [
        (i, o) for i, o in enumerate(OBJECTS) if o.name is None and o.description
    ]
    # Vendor.c declares 23 None-named entries total (20 SCROLL + 3 WAND).
    assert len(none_named) == 23, (
        f"expected 23 None-named appearance slots (20 scroll + 3 wand), "
        f"got {len(none_named)}"
    )
