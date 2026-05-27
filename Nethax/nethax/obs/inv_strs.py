"""NLE-fidelity inventory string rendering in JAX.

Produces ``inv_strs: uint8[55, 80]`` — 55 inventory slots, each an 80-byte
null-terminated ASCII string matching the format produced by NetHack's
``doname``/``xname`` (vendor/nethack/src/objnam.c).

String structure per slot::

    <letter> - <qty_word> <buc_word> <enchant> <name> <equip_status> <charges>

Wave 3 simplifications (documented per feature):
- User-given names (oname/oextra) skipped entirely
- Two-weapon "alternate weapon" status skipped
- Charges shown as "(recharged:charges)" matching vendor objnam.c:1486
- Slots 52-54 (NLE extras beyond a-zA-Z) always rendered empty

The "a"/"an" article selection is now byte-equal to vendor objnam.c::an
via the pre-computed _OBJECT_USE_AN / _APP_USE_AN tables (Wave 6); the
old "no vowel check" simplification has been retired.

Canonical sources:
  vendor/nethack/src/objnam.c  — doname / xname / an
  vendor/nethack/src/invent.c  — display_inventory
  vendor/nle/include/nleobs.h  — NLE_INVENTORY_SIZE=55, NLE_INVENTORY_STR_LENGTH=80
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from Nethax.nethax.constants.objects import OBJECTS, ObjectClass, Material
from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.subsystems.items import BUCStatus
from Nethax.nethax.subsystems.inventory import (
    MAX_INVENTORY_SLOTS,
    ArmorSlot,
    USER_NAME_LEN,
)

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

NLE_INV_SLOTS   = 55   # NLE_INVENTORY_SIZE
NLE_STR_LEN     = 80   # NLE_INVENTORY_STR_LENGTH
_MAX_OBJ        = len(OBJECTS)

# Max byte-lengths for each padded table column
_MAX_NAME_LEN       = 40   # longest canonical object name
_MAX_APP_LEN        = 30   # longest appearance description
_MAX_BUC_LEN        = 10   # "uncursed\0" = 9 bytes
_MAX_EQUIP_LEN      = 24   # "(in quiver pouch)\0" = 18 bytes
_MAX_ENCHANT_LEN    =  5   # "+127\0"
_MAX_QTY_LEN        =  8   # "999999\0"

# ---------------------------------------------------------------------------
# Static byte tables  (built once at module-load; never re-entered by JIT)
# ---------------------------------------------------------------------------

def _pad_bytes(s: str | None, width: int) -> list[int]:
    """Return bytes of *s* zero-padded to exactly *width* bytes."""
    b = s.encode("ascii") if s else b""
    b = b[:width]
    return list(b) + [0] * (width - len(b))


# Object canonical names  — shape (NUM_OBJECTS, _MAX_NAME_LEN)
_OBJECT_NAMES_BYTES: jnp.ndarray = jnp.array(
    [_pad_bytes(obj.name, _MAX_NAME_LEN) for obj in OBJECTS],
    dtype=jnp.uint8,
)  # uint8[NUM_OBJECTS, _MAX_NAME_LEN]

# Appearance descriptions — shape (NUM_OBJECTS, _MAX_APP_LEN)
# For objects with no appearance (description=None), the table holds zeros,
# and we fall back to the canonical name at render time.
_APPEARANCE_BYTES: jnp.ndarray = jnp.array(
    [_pad_bytes(obj.description, _MAX_APP_LEN) for obj in OBJECTS],
    dtype=jnp.uint8,
)  # uint8[NUM_OBJECTS, _MAX_APP_LEN]

# Whether each object has an appearance description (description != None)
_HAS_APPEARANCE: jnp.ndarray = jnp.array(
    [obj.description is not None for obj in OBJECTS],
    dtype=jnp.bool_,
)  # bool[NUM_OBJECTS]

# Object class for each entry — used to decide whether to show enchantment,
# charges, etc.
_OBJECT_CLASS: jnp.ndarray = jnp.array(
    [int(obj.class_) for obj in OBJECTS],
    dtype=jnp.uint8,
)  # uint8[NUM_OBJECTS]

# ---------------------------------------------------------------------------
# Erosion rendering tables
# Canonical: vendor/nethack/src/objnam.c::add_erosion_words() lines 1142-1191
#
# The per-object erosion-class lookup (``_OBJECT_EROSION_CLASS``) now lives
# in ``Nethax.nethax.subsystems.items`` so it is constructed at top-level
# import — before any JIT trace runs.  Previously it was defined here and
# pulled in via a deferred import inside ``items.erode_obj_slot``, which
# triggered ``UnexpectedTracerError`` when ``_do_corrode`` (or any other
# JIT-traced caller) was the first thing to load this module.
# ---------------------------------------------------------------------------
from Nethax.nethax.subsystems.items import _OBJECT_EROSION_CLASS  # noqa: E402

# Erosion word tables indexed [mat_class 0..3][level 0..3].
# vendor/nethack/src/objnam.c lines 1156-1168 (oeroded), 1169-1178 (oeroded2).
_MAX_EROSION_WORD_LEN = 22  # "thoroughly corroded \0" = 21 bytes

_EROSION_OERODED_STRS = [
    ["", "", "", ""],  # mat 0: none
    ["", "rusty ", "very rusty ", "thoroughly rusty "],  # mat 1: rustprone
    ["", "burnt ", "very burnt ", "thoroughly burnt "],  # mat 2: flammable
    ["", "", "", ""],  # mat 3: corrode-only (oeroded unused)
]
_EROSION_OERODED2_STRS = [
    ["", "", "", ""],  # mat 0: none
    ["", "corroded ", "very corroded ", "thoroughly corroded "],  # mat 1: rustprone->corrode
    ["", "rotted ", "very rotted ", "thoroughly rotted "],  # mat 2: flammable->rot
    ["", "corroded ", "very corroded ", "thoroughly corroded "],  # mat 3: corrode-only
]
_EROSION_PROOF_STRS = ["", "rustproof ", "fireproof ", "corrodeproof "]
_ROTPROOF_STR = "rotproof "  # mat_class 2 + oerodeproof + oeroded2 nonzero

_EROSION_OERODED_BYTES: jnp.ndarray = jnp.array(
    [[_pad_bytes(w, _MAX_EROSION_WORD_LEN) for w in row] for row in _EROSION_OERODED_STRS],
    dtype=jnp.uint8,
)  # uint8[4, 4, _MAX_EROSION_WORD_LEN]

_EROSION_OERODED2_BYTES: jnp.ndarray = jnp.array(
    [[_pad_bytes(w, _MAX_EROSION_WORD_LEN) for w in row] for row in _EROSION_OERODED2_STRS],
    dtype=jnp.uint8,
)  # uint8[4, 4, _MAX_EROSION_WORD_LEN]

_EROSION_PROOF_BYTES: jnp.ndarray = jnp.array(
    [_pad_bytes(w, _MAX_EROSION_WORD_LEN) for w in _EROSION_PROOF_STRS],
    dtype=jnp.uint8,
)  # uint8[4, _MAX_EROSION_WORD_LEN]

_ROTPROOF_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes(_ROTPROOF_STR, _MAX_EROSION_WORD_LEN), dtype=jnp.uint8
)  # uint8[_MAX_EROSION_WORD_LEN]

# BUC words — index 0=empty, 1=cursed, 2=uncursed, 3=blessed
# Matches BUCStatus enum: UNKNOWN=0, CURSED=1, UNCURSED=2, BLESSED=3
_BUC_STRS = ["", "cursed", "uncursed", "blessed"]
_BUC_BYTES: jnp.ndarray = jnp.array(
    [_pad_bytes(s, _MAX_BUC_LEN) for s in _BUC_STRS],
    dtype=jnp.uint8,
)  # uint8[4, _MAX_BUC_LEN]

# Inventory slot letters — a-z (0-25), A-Z (26-51), then slots 52-54 get '\0'
def _make_letters() -> jnp.ndarray:
    letters = []
    for i in range(26):
        letters.append(ord('a') + i)
    for i in range(26):
        letters.append(ord('A') + i)
    for _ in range(3):
        letters.append(0)
    return jnp.array(letters, dtype=jnp.uint8)

_LETTERS: jnp.ndarray = _make_letters()  # uint8[55]

# Equip status strings — indexed by a small enum defined below
# 0: empty
# 1: (weapon in hand)
# 2: (being worn)
# 3: (on left hand)
# 4: (on right hand)
# 5: (in quiver)           — ammo for a bow  (vendor objnam.c:1629 Qtyp==1)
# 6: (being worn)          — amulet
# 7: (being worn)          — shirt / cloak (all non-ring/non-weapon armor)
# 8: (in quiver pouch)     — small non-ammo: ring/amulet/wand/gem/coin
#                            vendor objnam.c:1636 Qtyp==2
# 9: (at the ready)        — non-ammo weapon-class or other odd items in quiver
#                            vendor objnam.c:1639 Qtyp==3
_EQUIP_STRS = [
    "",
    "(weapon in hand)",
    "(being worn)",
    "(on left hand)",
    "(on right hand)",
    "(in quiver)",
    "(being worn)",        # amulet
    "(being worn)",        # shirt / cloak (all non-ring/non-weapon armor)
    "(in quiver pouch)",   # vendor objnam.c:1644 Qtyp==2
    "(at the ready)",      # vendor objnam.c:1645 Qtyp==3
]
_EQUIP_BYTES: jnp.ndarray = jnp.array(
    [_pad_bytes(s, _MAX_EQUIP_LEN) for s in _EQUIP_STRS],
    dtype=jnp.uint8,
)  # uint8[10, _MAX_EQUIP_LEN]

# ObjectClass values for classes that get enchantment shown
_WEAPON_CLASS_VAL = int(ObjectClass.WEAPON_CLASS)
_ARMOR_CLASS_VAL  = int(ObjectClass.ARMOR_CLASS)
# ObjectClass values for classes that get charges shown
_WAND_CLASS_VAL   = int(ObjectClass.WAND_CLASS)
_TOOL_CLASS_VAL   = int(ObjectClass.TOOL_CLASS)
# ObjectClass values for classes whose names carry "potion of" / "scroll of" prefix
_POTION_CLASS_VAL  = int(ObjectClass.POTION_CLASS)
_SCROLL_CLASS_VAL  = int(ObjectClass.SCROLL_CLASS)
_SPBOOK_CLASS_VAL  = int(ObjectClass.SPBOOK_CLASS)
_RING_CLASS_VAL    = int(ObjectClass.RING_CLASS)
_AMULET_CLASS_VAL  = int(ObjectClass.AMULET_CLASS)
_FOOD_CLASS_VAL    = int(ObjectClass.FOOD_CLASS)
_COIN_CLASS_VAL    = int(ObjectClass.COIN_CLASS)

# Special type-ID sentinels — looked up once at module load.
# vendor/nethack/src/objnam.c:841 (holy/unholy water),
# vendor/nethack/src/eat.c:tin_details (tin contents),
# vendor/nethack/src/objnam.c:1507 (corpse naming).
def _find_type_id(name: str, cls: ObjectClass) -> int:
    for i, obj in enumerate(OBJECTS):
        if obj.name == name and obj.class_ == cls:
            return i
    return -1

_POT_WATER_TYPE_ID: int = _find_type_id("water", ObjectClass.POTION_CLASS)   # 297
_TIN_TYPE_ID:       int = _find_type_id("tin",   ObjectClass.FOOD_CLASS)      # 271
_CORPSE_TYPE_ID:    int = _find_type_id("corpse", ObjectClass.FOOD_CLASS)     # 240

# Rings with oc_charged==1 (vendor/nle/src/objects.c RING macro, spec arg):
# adornment, gain strength, gain constitution, increase accuracy,
# increase damage, protection all have spec=1.
# vendor/nethack/src/objnam.c:1500: if (known && objects[obj->otyp].oc_charged)
_CHARGED_RING_NAMES = frozenset({
    "adornment", "gain strength", "gain constitution",
    "increase accuracy", "increase damage", "protection",
})
_OBJECT_IS_CHARGED: jnp.ndarray = jnp.array(
    [
        obj.class_ == ObjectClass.RING_CLASS and obj.name in _CHARGED_RING_NAMES
        for obj in OBJECTS
    ],
    dtype=jnp.bool_,
)  # bool[NUM_OBJECTS] — True iff ring has oc_charged (vendor objnam.c:1500)

# Monster name byte table — for corpse/tin rendering.
# vendor/nethack/src/objnam.c:1824 (corpse_xname), eat.c:1456 (tin monster meat).
_MAX_MONSTER_NAME_LEN = 32
_NUM_MONSTERS = len(MONSTERS)
_MONSTER_NAME_BYTES: jnp.ndarray = jnp.array(
    [
        _pad_bytes(m.name if m is not None else "", _MAX_MONSTER_NAME_LEN)
        for m in MONSTERS
    ],
    dtype=jnp.uint8,
)  # uint8[NUM_MONSTERS, _MAX_MONSTER_NAME_LEN]

# Suffix/prefix byte constants for corpse and tin name rendering.
_CORPSE_SUFFIX_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes(" corpse", 8), dtype=jnp.uint8,
)  # uint8[8]
_TIN_OF_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes("tin of ", 8), dtype=jnp.uint8,
)  # uint8[8]
_SPINACH_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes("spinach", 8), dtype=jnp.uint8,
)  # uint8[8]
_MEAT_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes(" meat", 6), dtype=jnp.uint8,
)  # uint8[6]

# Holy / unholy water name bytes (vendor objnam.c:841-843).
_HOLY_WATER_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes("holy water", 11), dtype=jnp.uint8,
)  # uint8[11]
_UNHOLY_WATER_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes("unholy water", 13), dtype=jnp.uint8,
)  # uint8[13]

# Class prefix strings — prepended before the name or appearance for certain classes
# Index matches ObjectClass integer values (0-17)
_CLASS_PREFIX_STRS = [""] * 18
_CLASS_PREFIX_STRS[_POTION_CLASS_VAL] = "potion of "
_CLASS_PREFIX_STRS[_SCROLL_CLASS_VAL] = "scroll of "
_CLASS_PREFIX_STRS[_SPBOOK_CLASS_VAL] = "spellbook of "
_CLASS_PREFIX_STRS[_RING_CLASS_VAL]   = "ring of "
_CLASS_PREFIX_STRS[_AMULET_CLASS_VAL] = "amulet of "

_MAX_PREFIX_LEN = 14  # "spellbook of \0"
_CLASS_PREFIX_BYTES: jnp.ndarray = jnp.array(
    [_pad_bytes(s, _MAX_PREFIX_LEN) for s in _CLASS_PREFIX_STRS],
    dtype=jnp.uint8,
)  # uint8[18, _MAX_PREFIX_LEN]

# Prefix applies only when the item is IDENTIFIED (showing true name).
# For unidentified items with appearances, no prefix (just "orange potion", etc.)
# For identified items with class prefix: show "potion of healing" etc.
# For identified items without prefix (weapons, armor): just the name.

# ---------------------------------------------------------------------------
# "pair of " / "set of " noun-cluster prefix tables.
# Vendor citations:
#   objnam.c:724-726  — ARMOR_CLASS, is_boots/is_gloves -> "pair of "
#   objnam.c:721-723  — ARMOR_CLASS, GRAY..YELLOW DRAGON_SCALES -> "set of "
#   objnam.c:694-695  — TOOL_CLASS, LENSES -> "pair of "
#   obj.h:427         — #define pair_of(o) ((o)->otyp == LENSES
#                       || is_gloves(o) || is_boots(o))
# ARM_GLOVES = 3, ARM_BOOTS = 4 (objclass.h lines 41-42).
# When quantity > 1, vendor renders "pairs of " (objnam.c:2879).
# ---------------------------------------------------------------------------

_LENSES_TYPE_ID = _find_type_id("lenses", ObjectClass.TOOL_CLASS)

def _is_pair_of(obj, otyp: int) -> bool:
    if obj is None or obj.name is None:
        return False
    if obj.class_ == ObjectClass.ARMOR_CLASS:
        return obj.oc_armor_class in (3, 4)  # ARM_GLOVES, ARM_BOOTS
    if obj.class_ == ObjectClass.TOOL_CLASS and otyp == _LENSES_TYPE_ID:
        return True
    return False


def _is_set_of(obj) -> bool:
    """True for gray..yellow dragon scales (NOT scale mail).

    Vendor: objnam.c:721 — range GRAY_DRAGON_SCALES..YELLOW_DRAGON_SCALES.
    Match by name: '<color> dragon scales' (with trailing 's', not "scale mail").
    """
    if obj is None or obj.name is None:
        return False
    if obj.class_ != ObjectClass.ARMOR_CLASS:
        return False
    return obj.name.endswith(" dragon scales")


_PAIR_OF: jnp.ndarray = jnp.array(
    [_is_pair_of(obj, i) for i, obj in enumerate(OBJECTS)],
    dtype=jnp.bool_,
)  # bool[NUM_OBJECTS]

_SET_OF: jnp.ndarray = jnp.array(
    [_is_set_of(obj) for obj in OBJECTS],
    dtype=jnp.bool_,
)  # bool[NUM_OBJECTS]

_PAIR_OF_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes("pair of ", 9), dtype=jnp.uint8,
)  # uint8[9]
_PAIRS_OF_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes("pairs of ", 10), dtype=jnp.uint8,
)  # uint8[10]
_SET_OF_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes("set of ", 8), dtype=jnp.uint8,
)  # uint8[8]

# Digit table for integer rendering  (0..9 -> b'0'..b'9')
_DIGITS: jnp.ndarray = jnp.arange(10, dtype=jnp.uint8) + jnp.uint8(ord('0'))

# ---------------------------------------------------------------------------
# Class noun table for unidentified items  (appended after appearance word)
# Must be defined before the rendering helpers that reference it.
# ---------------------------------------------------------------------------

_CLASS_NOUN_STRS = [""] * 18
_CLASS_NOUN_STRS[int(ObjectClass.POTION_CLASS)] = " potion"
_CLASS_NOUN_STRS[int(ObjectClass.SCROLL_CLASS)] = " scroll"
_CLASS_NOUN_STRS[int(ObjectClass.SPBOOK_CLASS)] = " spellbook"
_CLASS_NOUN_STRS[int(ObjectClass.WAND_CLASS)]   = " wand"
_CLASS_NOUN_STRS[int(ObjectClass.RING_CLASS)]   = " ring"
_CLASS_NOUN_STRS[int(ObjectClass.AMULET_CLASS)] = " amulet"
_CLASS_NOUN_STRS[int(ObjectClass.GEM_CLASS)]    = " gem"

_MAX_CLASS_NOUN_LEN = 12  # " spellbook\0" = 11 bytes
_CLASS_NOUN_BYTES: jnp.ndarray = jnp.array(
    [_pad_bytes(s, _MAX_CLASS_NOUN_LEN) for s in _CLASS_NOUN_STRS],
    dtype=jnp.uint8,
)  # uint8[18, _MAX_CLASS_NOUN_LEN]


# ---------------------------------------------------------------------------
# Wave 6 Phase A — vowel-article mask, pluralization tables, name suffix,
# two-weapon "(alternate weapon)" marker.
# Canonical: vendor/nethack/src/objnam.c::an, doname, makeplural,
#            vendor/nethack/src/wield.c (u.twoweap status display).
# ---------------------------------------------------------------------------

# Per-byte vowel mask: True iff the byte is one of 'a','e','i','o','u'
# (lowercase only; appearance strings in OBJECTS are lowercase).  Used to
# decide between "a " and "an " when prefixing an item with quantity 1.
def _build_vowel_mask() -> jnp.ndarray:
    mask = [False] * 256
    for ch in b"aeiou":
        mask[ch] = True
    return jnp.array(mask, dtype=jnp.bool_)

_VOWEL_MASK: jnp.ndarray = _build_vowel_mask()  # bool[256]


# Per-object "use 'an'" table — True iff the item's effective display name
# starts with a vowel-sound that warrants "an" (rather than "a").
# Implements the same exception logic as vendor/nethack/src/objnam.c::just_an()
# lines 2108-2142 but evaluated at module load time on the known object names
# and appearance strings (not inside JIT).
#
# Long-u / 'one' exceptions (vendor just_an lines 2129-2135):
#   starts with "eu", "uke", "ukulele", "unicorn", "uranium", "useful" -> "a"
# Single-letter rule (vendor just_an line 2117):
#   only letters in "aefhilmnosx" take "an"
# 'x' + consonant (vendor just_an line 2136): -> "an"
_LONG_U_PREFIXES_INV = ("eu", "uke", "ukulele", "unicorn", "uranium", "useful")
_AN_SINGLE_LETTERS_INV = set("aefhilmnosx")
_NO_ARTICLE_NAMES_INV = {"molten lava", "iron bars", "ice"}


def _compute_use_an(name: str | None) -> bool:
    """Return True iff just_an() would choose 'an' for this name.

    Port of vendor/nethack/src/objnam.c::just_an() lines 2108-2142.
    Evaluated at module load time to populate static per-object tables.
    """
    if not name:
        return False
    lower = name.lower()
    if lower in _NO_ARTICLE_NAMES_INV:
        return False
    if lower.startswith("the "):
        return False
    c0 = lower[0]
    # Single-letter word (just_an line 2115-2117)
    if len(name) == 1 or name[1] == ' ':
        return c0 in _AN_SINGLE_LETTERS_INV
    if c0 in "aeiou":
        # 'one' + separator -> "a" (just_an line 2130)
        if lower.startswith("one") and (len(name) == 3 or name[3] in "-_ "):
            return False
        # long-u prefixes -> "a" (just_an lines 2132-2135)
        for pfx in _LONG_U_PREFIXES_INV:
            if lower.startswith(pfx):
                return False
        return True
    # 'x' + consonant -> "an" (just_an line 2136)
    if c0 == 'x' and len(name) > 1 and name[1].lower() not in "aeiou":
        return True
    return False


# shape (NUM_OBJECTS,) — True means write "an ", False means write "a ".
# Based on canonical name (identified path).
_OBJECT_USE_AN: jnp.ndarray = jnp.array(
    [_compute_use_an(obj.name) for obj in OBJECTS],
    dtype=jnp.bool_,
)  # bool[NUM_OBJECTS]

# Same for appearance descriptions (unidentified path).
_APP_USE_AN: jnp.ndarray = jnp.array(
    [_compute_use_an(obj.description) for obj in OBJECTS],
    dtype=jnp.bool_,
)  # bool[NUM_OBJECTS]

# Per-BUC-status "use 'an'" flag.  Vendor doname_base (objnam.c:1686-1692)
# calls just_an() on the first word in prefix after "a "; when BUC is shown
# that first word is the BUC word itself, not the object name.
# "cursed"   -> 'c' consonant -> False
# "uncursed" -> 'u' vowel     -> True   (this is the divergence vs. the old
#                                         hardcoded False for all BUC states)
# "blessed"  -> 'b' consonant -> False
# index 0 (UNKNOWN / not shown) is unused in the buc_known branch.
_BUC_USE_AN: jnp.ndarray = jnp.array(
    [_compute_use_an(s) for s in _BUC_STRS],
    dtype=jnp.bool_,
)  # bool[4]  — vendor objnam.c:1686-1692


# Suffix-level irregular plurals (apply to compound words too:
# crysknife -> crysknives, midwife -> midwives).
# Source: vendor/nethack/src/objnam.c::makeplural ~lines 340-460.
_SUFFIX_IRREGULARS: tuple = (
    ("knife", "knives"),
    ("wife",  "wives"),
    ("life",  "lives"),
    ("loaf",  "loaves"),
    ("leaf",  "leaves"),
)

# Whole-word (or hyphenated/space-separated) irregular plurals.  An entry
# "ox" matches "ox" or "musk-ox" but never "fox".
_WORD_IRREGULARS: tuple = (
    ("staff",   "staves"),
    ("tooth",   "teeth"),
    ("foot",    "feet"),
    ("mouse",   "mice"),
    ("louse",   "lice"),
    ("goose",   "geese"),
    ("man",     "men"),
    ("woman",   "women"),
    ("child",   "children"),
    ("ox",      "oxen"),
    ("octopus", "octopi"),
    ("elf",     "elves"),
    ("dwarf",   "dwarves"),
    ("wolf",    "wolves"),
)


def pluralize(word: str) -> str:
    """Return the English plural of *word*.

    Implements the subset of NetHack's ``makeplural`` (objnam.c) needed for
    object names.  Two irregular tables are consulted before the regular
    rule:

      * Suffix irregulars (knife, wife, ...) match any word ending in the
        suffix, so "crysknife" pluralizes to "crysknives".
      * Word irregulars (man, ox, ...) match only as a whole word or after
        a separator, so "fox" pluralizes to "foxes" (not "foxen").

    Falls back to "es" for sibilant endings ('s','x','z','sh','ch') and "s"
    otherwise.  Pure Python -- invoked only at module import time to populate
    static byte tables.
    """
    if not word:
        return word
    lower = word.lower()
    for sing, plur in _SUFFIX_IRREGULARS:
        if lower.endswith(sing):
            return word[: len(word) - len(sing)] + plur
    for sing, plur in _WORD_IRREGULARS:
        if lower == sing or lower.endswith(" " + sing) or lower.endswith("-" + sing):
            return word[: len(word) - len(sing)] + plur
    # Sibilant endings require 'es'.
    if (
        lower.endswith("s")
        or lower.endswith("x")
        or lower.endswith("z")
        or lower.endswith("sh")
        or lower.endswith("ch")
    ):
        return word + "es"
    return word + "s"


def _pluralize_phrase(name) -> str:
    """Pluralize the head noun of a NetHack object phrase.

    Wave 6 parity-fix: accepts None for shuffled-appearance OBJECTS slots
    (vendor/nle/src/objects.c lines 855-875 and 1044-1046 -- 23 None-named
    entries with only a description).

    Handles the "X of Y" pattern (e.g. "ring of protection" -> "rings of
    protection") by pluralizing the word before ' of '.  Otherwise pluralizes
    the last word.
    """
    if not name:
        return ""
    of_idx = name.find(" of ")
    if of_idx >= 0:
        head = name[:of_idx]
        tail = name[of_idx:]
        # Pluralize last word of head.
        sp = head.rfind(" ")
        if sp >= 0:
            return head[: sp + 1] + pluralize(head[sp + 1 :]) + tail
        return pluralize(head) + tail
    # No " of " -- pluralize the trailing word.
    sp = name.rfind(" ")
    if sp >= 0:
        return name[: sp + 1] + pluralize(name[sp + 1 :])
    return pluralize(name)


# Plural-name table -- one row per object, padded to a width wide enough to
# hold the longest plural form.
_MAX_PLURAL_NAME_LEN = max(_MAX_NAME_LEN, max(len(_pluralize_phrase(obj.name)) for obj in OBJECTS) + 1)
_NAME_PLURAL_BYTES: jnp.ndarray = jnp.array(
    [_pad_bytes(_pluralize_phrase(obj.name), _MAX_PLURAL_NAME_LEN) for obj in OBJECTS],
    dtype=jnp.uint8,
)  # uint8[NUM_OBJECTS, _MAX_PLURAL_NAME_LEN]

# Singular canonical-name table re-padded to the plural width so the two can
# be selected by jnp.where with a common shape.
_OBJECT_NAMES_BYTES_PADDED: jnp.ndarray = jnp.array(
    [_pad_bytes(obj.name, _MAX_PLURAL_NAME_LEN) for obj in OBJECTS],
    dtype=jnp.uint8,
)  # uint8[NUM_OBJECTS, _MAX_PLURAL_NAME_LEN]

# Plural-appearance table -- pluralize the last word of the description so
# "wooden" -> "wooden" (no last word change needed; an "s" comes via the class
# noun route) but "elven dagger" -> "elven daggers".  When the description is
# None/empty the row is all zeros (and is unused -- has_appearance handles
# fallback to the canonical-name path).
def _safe_pluralize_app(desc: str | None) -> str:
    if not desc:
        return ""
    return _pluralize_phrase(desc)

_MAX_PLURAL_APP_LEN = max(_MAX_APP_LEN, max((len(_safe_pluralize_app(obj.description)) for obj in OBJECTS), default=0) + 1)
_APPEARANCE_PLURAL_BYTES: jnp.ndarray = jnp.array(
    [_pad_bytes(_safe_pluralize_app(obj.description), _MAX_PLURAL_APP_LEN) for obj in OBJECTS],
    dtype=jnp.uint8,
)  # uint8[NUM_OBJECTS, _MAX_PLURAL_APP_LEN]

# Singular appearance table re-padded to the plural width so jnp.where can
# select between them on a per-slot basis.
_APPEARANCE_BYTES_PADDED: jnp.ndarray = jnp.array(
    [_pad_bytes(obj.description or "", _MAX_PLURAL_APP_LEN) for obj in OBJECTS],
    dtype=jnp.uint8,
)  # uint8[NUM_OBJECTS, _MAX_PLURAL_APP_LEN]


# " named " separator + " (alternate weapon)" suffix, padded for fixed-width
# copy via _write_fixed.
_NAMED_PREFIX_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes(" named ", 8), dtype=jnp.uint8,
)  # uint8[8]
# Vendor objnam.c line 1619 emits " (alternate weapon; not wielded)".
_ALT_WEAPON_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes(" (alternate weapon; not wielded)", 33), dtype=jnp.uint8,
)  # uint8[33]

# Vendor objnam.c prefix tokens emitted into the ``prefix`` buffer before
# erosion / enchant: ``greased`` (line 1371), WEAPON_CLASS ``poisoned``
# (line 1420), FOOD_CLASS ``partly eaten`` (line 1506).
_GREASED_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes("greased ", 9), dtype=jnp.uint8,
)  # uint8[9]
_POISONED_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes("poisoned ", 10), dtype=jnp.uint8,
)  # uint8[10]
_PARTLY_EATEN_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes("partly eaten ", 14), dtype=jnp.uint8,
)  # uint8[14]


# ---------------------------------------------------------------------------
# JIT-compatible byte-buffer helpers
# ---------------------------------------------------------------------------

def _write_fixed(buf: jax.Array, cursor: jax.Array,
                  src: jax.Array, src_len: int) -> tuple[jax.Array, jax.Array]:
    """Append up to *src_len* bytes from *src* (zero-terminated) into *buf*.

    Copies bytes from src[0..src_len) until it hits a zero byte or exhausts
    src_len, advancing cursor accordingly.  Both buf and cursor are immutable
    (returns new values).

    JIT-compatible: uses lax.fori_loop with fixed trip count *src_len*.

    Args:
        buf:     uint8[NLE_STR_LEN] output buffer.
        cursor:  int32 write position.
        src:     uint8[src_len] source bytes (zero-terminated string).
        src_len: static trip count (must be a Python int).

    Returns:
        (new_buf, new_cursor)
    """
    def body(i, carry):
        b, c = carry
        ch = src[i]
        # Only write if: cursor+i is in range, ch != 0, and we haven't hit a
        # zero yet.  We track "hit zero" implicitly by checking ch != 0.
        in_range = (c < NLE_STR_LEN)
        ch_ok    = ch != jnp.uint8(0)
        should_write = in_range & ch_ok
        new_b = lax.cond(
            should_write,
            lambda _b: _b.at[c].set(ch),
            lambda _b: _b,
            b,
        )
        new_c = jnp.where(should_write, c + jnp.int32(1), c)
        return new_b, new_c

    return lax.fori_loop(0, src_len, body, (buf, cursor))


def _write_byte(buf: jax.Array, cursor: jax.Array,
                ch: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Write a single byte *ch* at *cursor*, returning (new_buf, new_cursor)."""
    in_range = cursor < NLE_STR_LEN
    new_buf = lax.cond(
        in_range,
        lambda b: b.at[cursor].set(ch.astype(jnp.uint8)),
        lambda b: b,
        buf,
    )
    new_cursor = jnp.where(in_range, cursor + jnp.int32(1), cursor)
    return new_buf, new_cursor


def _write_space(buf: jax.Array, cursor: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _write_byte(buf, cursor, jnp.uint8(ord(' ')))


def _write_uint(buf: jax.Array, cursor: jax.Array,
                value: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Write a non-negative integer (up to 6 digits) as ASCII bytes.

    Simplified: renders value modulo 1_000_000. Works for quantities and charges.
    Uses a fixed-width reverse-fill approach compatible with JAX's static shapes.

    Returns (new_buf, new_cursor).
    """
    # We render up to 6 decimal digits.  Allocate a 6-byte scratch, fill
    # right-to-left, then emit only the significant portion.
    _MAX_DIGITS = 6

    # Build digit array from least-significant to most-significant
    def make_digits(v):
        # Returns digits[0..5] where digits[0] is least-significant
        digits = jnp.zeros((_MAX_DIGITS,), dtype=jnp.uint8)
        def body(i, carry):
            digs, rem = carry
            d = (rem % jnp.int32(10)).astype(jnp.uint8) + jnp.uint8(ord('0'))
            digs = digs.at[i].set(d)
            rem = rem // jnp.int32(10)
            return digs, rem
        digits, _ = lax.fori_loop(0, _MAX_DIGITS, body,
                                  (digits, value.astype(jnp.int32)))
        return digits

    digits = make_digits(value)  # digits[0] = LSD
    # Count significant digits (everything above 0 is significant; at least 1)
    n_digits = jnp.int32(1)
    for k in range(1, _MAX_DIGITS):
        # digits[k] != ord('0') means this position is non-zero
        n_digits = jnp.where(digits[k] != jnp.uint8(ord('0')),
                             jnp.int32(k + 1), n_digits)

    # Emit most-significant to least-significant
    def emit_body(i, carry):
        b, c, nd = carry
        # i goes 0.._MAX_DIGITS-1; we want digit at position (nd-1-i)
        # but only emit if i < nd
        pos = nd - jnp.int32(1) - i
        ch  = digits[pos]
        in_range  = (i < nd) & (c < NLE_STR_LEN)
        new_b = lax.cond(in_range, lambda _b: _b.at[c].set(ch), lambda _b: _b, b)
        new_c = jnp.where(in_range, c + jnp.int32(1), c)
        return new_b, new_c, nd

    buf, cursor, _ = lax.fori_loop(0, _MAX_DIGITS, emit_body,
                                   (buf, cursor, n_digits))
    return buf, cursor


# ---------------------------------------------------------------------------
# Equip-status resolver
# ---------------------------------------------------------------------------

def _equip_status_idx(inv_state, slot_idx: jax.Array,
                      item_class: jax.Array) -> jax.Array:
    """Return the equip-status table index (int32) for the given inventory slot.

    0 = no status, 1 = weapon in hand, 2 = being worn (body/helm/gloves/boots/
    cloak/shirt), 3 = on left hand (ring), 4 = on right hand (ring),
    5 = in quiver (bow ammo), 6 = being worn (amulet),
    8 = in quiver pouch (small non-ammo: ring/amulet/wand/gem/coin),
    9 = at the ready (non-ammo weapon-class or odd quivered items).

    Vendor: objnam.c:1622-1646 -- quiver Qtyp switch.
    JIT-compatible: only jnp.where / == comparisons.
    """
    i8 = slot_idx.astype(jnp.int8)

    # Wielded weapon
    is_wielded = inv_state.wielded == i8

    # Worn armor -- any of the 7 slots
    # worn_armor[j] == slot_idx  means slot is worn in armor position j
    is_armor_worn = jnp.any(inv_state.worn_armor == i8)

    # Rings -- left = worn_rings[0], right = worn_rings[1]
    is_ring_l = inv_state.worn_rings[0] == i8
    is_ring_r = inv_state.worn_rings[1] == i8

    # Amulet
    is_amulet = inv_state.worn_amulet == i8

    # Quiver -- differentiate by item class per vendor objnam.c:1622-1646.
    # RING/AMULET/WAND/COIN/GEM class -> Qtyp 2 "in quiver pouch" (idx 8).
    # WEAPON class -> idx 5 "(in quiver)" (approximation: all weapon-class).
    # Other classes -> idx 9 "at the ready".
    is_quiver = inv_state.quiver == i8
    cls = item_class.astype(jnp.int32)
    is_pouch_class = (
        (cls == jnp.int32(_RING_CLASS_VAL))   |
        (cls == jnp.int32(_AMULET_CLASS_VAL)) |
        (cls == jnp.int32(_WAND_CLASS_VAL))   |
        (cls == jnp.int32(int(ObjectClass.COIN_CLASS))) |
        (cls == jnp.int32(int(ObjectClass.GEM_CLASS)))
    )
    is_weapon_class = cls == jnp.int32(_WEAPON_CLASS_VAL)
    quiver_idx = jnp.where(is_pouch_class, jnp.int32(8),
                           jnp.where(is_weapon_class, jnp.int32(5), jnp.int32(9)))

    result = jnp.int32(0)
    result = jnp.where(is_quiver,     quiver_idx, result)
    result = jnp.where(is_amulet,     jnp.int32(6), result)
    result = jnp.where(is_ring_r,     jnp.int32(4), result)
    result = jnp.where(is_ring_l,     jnp.int32(3), result)
    result = jnp.where(is_armor_worn, jnp.int32(2), result)
    result = jnp.where(is_wielded,    jnp.int32(1), result)
    return result


# ---------------------------------------------------------------------------
# Single-slot renderer
# ---------------------------------------------------------------------------

def _render_slot(inv_state, id_state, slot_idx: jax.Array,
                  two_weapon: jax.Array, alt_slot: jax.Array) -> jax.Array:
    """Render one inventory slot as an 80-byte uint8 string.

    Empty slots (category == 0) return all-zero buffers.

    Args:
        inv_state:  InventoryState pytree (items fields are [52]-arrays)
        id_state:   IdentificationState pytree
        slot_idx:   int32 scalar, 0..54
        two_weapon: bool scalar -- state.combat.two_weapon flag (Wave 6).
        alt_slot:   int32 scalar -- state.inventory.alternate_weapon_slot.

    Returns:
        uint8[80]
    """
    buf    = jnp.zeros((NLE_STR_LEN,), dtype=jnp.uint8)
    cursor = jnp.int32(0)

    # Clamp to valid item-array range (slots 52-54 don't exist in items[52])
    safe_idx = jnp.clip(slot_idx, 0, MAX_INVENTORY_SLOTS - 1).astype(jnp.int32)

    # Read item fields
    category    = inv_state.items.category[safe_idx].astype(jnp.int32)
    type_id     = inv_state.items.type_id[safe_idx].astype(jnp.int32)
    buc_status  = inv_state.items.buc_status[safe_idx].astype(jnp.int32)
    enchantment = inv_state.items.enchantment[safe_idx].astype(jnp.int32)
    charges     = inv_state.items.charges[safe_idx].astype(jnp.int32)
    # Identification gate: vendor xname (objnam.c:208) uses the per-type
    # ``ocl->oc_name_known`` as the primary gate; per-item ``dknown``/``known``
    # only modulate enchantment/BUC display.  We OR the per-item identified
    # flag with the per-type ``state.identification.identified[type_id]`` so
    # items of a type discovered via learnwand/learnring/learnscroll/learnpotion
    # render with their canonical name.
    # Cite: vendor/nethack/src/objnam.c::xname line 208 ``nn = ocl->oc_name_known``.
    _per_item_id   = inv_state.items.identified[safe_idx]
    _type_mask     = id_state.identified
    _safe_otyp     = jnp.clip(type_id, jnp.int32(0),
                              jnp.int32(_type_mask.shape[0] - 1))
    _per_type_id   = _type_mask[_safe_otyp]
    identified  = _per_item_id | _per_type_id
    quantity    = inv_state.items.quantity[safe_idx].astype(jnp.int32)
    # recharged: vendor obj.h recharged field -- recharge counter for wands.
    # vendor/nethack/src/objnam.c:1486: ConcatF2(bp,0," (%d:%d)",(int)obj->recharged,obj->spe)
    # Guard: legacy Item constructions may have recharged as a scalar default;
    # broadcast to a length-52 array so safe_idx indexing always works.
    _recharged_raw = getattr(inv_state.items, "recharged",
                             jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8))
    _recharged_arr = jnp.broadcast_to(
        jnp.asarray(_recharged_raw, dtype=jnp.int8),
        (MAX_INVENTORY_SLOTS,),
    )
    recharged   = _recharged_arr[safe_idx].astype(jnp.int32)
    # corpse_entry_idx: index into MONSTERS (-1 = not a corpse/tin-with-monster).
    corpse_idx  = inv_state.items.corpse_entry_idx[safe_idx].astype(jnp.int32)

    # Slot is empty when category == 0 OR slot_idx >= MAX_INVENTORY_SLOTS
    is_empty = (category == jnp.int32(0)) | (slot_idx >= jnp.int32(MAX_INVENTORY_SLOTS))

    # Precompute the a/an article choice.  The article precedes BUC/enchant/
    # name, so we pick whichever segment will actually be written first:
    #   - BUC word (if buc known): "blessed"/"cursed"/"uncursed" all start
    #     with a consonant, so always "a ".
    #   - appearance description (if unidentified + has appearance)
    #   - canonical name (otherwise)
    # We use pre-computed _OBJECT_USE_AN / _APP_USE_AN tables (built at
    # module load via just_an() logic) to encode all vendor exceptions
    # (long-u, 'one', single-letter, x-consonant).
    # Vendor: objnam.c::just_an() lines 2108-2142.
    safe_type = jnp.clip(type_id, 0, _MAX_OBJ - 1).astype(jnp.int32)
    has_app   = _HAS_APPEARANCE[safe_type]
    # bknown: player knows the BUC status of this item (vendor objnam.c:1318).
    # When False, suppress the "blessed"/"cursed"/"uncursed" prefix.
    bknown    = inv_state.items.bknown[safe_idx]
    buc_known = (buc_status != jnp.int32(BUCStatus.UNKNOWN)) & bknown
    buc_row   = jnp.clip(buc_status, 0, 3).astype(jnp.int32)
    show_app  = (~identified) & has_app
    # Article is chosen based on the first word that follows "a/an":
    #   - If BUC is shown, the BUC word is first ("uncursed" -> "an", etc.)
    #   - Otherwise, the noun/appearance is first.
    # Vendor: doname_base objnam.c:1686-1692 calls just_an() on the first
    # non-article word in the prefix buffer (BUC word when present, else name).
    noun_use_an = jnp.where(show_app, _APP_USE_AN[safe_type], _OBJECT_USE_AN[safe_type])
    buc_use_an  = _BUC_USE_AN[buc_row]
    article_use_an = jnp.where(buc_known, buc_use_an, noun_use_an)

    def render_nonempty(args):
        b, c = args

        # 1. NLE byte-parity: NLE inv_strs does NOT include a letter prefix.
        # Vendor: vendor/nle/win/rl/winrl.cc:459 writes `item.str` which is
        # the result of vendor `doname(otmp)` — that does NOT include a
        # leading "<letter> - " (the letter is conveyed via inv_letters[i]).
        # NetHack/Nethax mode keeps the legacy prefix for human-readable
        # tty output; NLE mode omits it.  Cite: docs/INV_STRS_FORMAT_DIFF.md.
        from Nethax.nethax.parity_mode import is_nle_mode as _is_nle
        if not _is_nle():
            letter = _LETTERS[slot_idx]
            b, c = _write_byte(b, c, letter)
            b, c = _write_byte(b, c, jnp.uint8(ord(' ')))
            b, c = _write_byte(b, c, jnp.uint8(ord('-')))
            b, c = _write_byte(b, c, jnp.uint8(ord(' ')))

        # 2. Quantity word: "<N> " for stacks, "a "/"an " for singletons.
        # Full just_an() exceptions via _OBJECT_USE_AN/_APP_USE_AN tables.
        # Vendor: objnam.c::just_an() lines 2108-2142.
        is_plural = quantity > jnp.int32(1)
        b, c = lax.cond(
            is_plural,
            lambda bc: _write_uint_space(bc[0], bc[1], quantity),
            lambda bc: _write_article_space(bc[0], bc[1], article_use_an),
            (b, c),
        )

        # 3. BUC word (only if buc_status != UNKNOWN == 0).
        # Exception 1: holy/unholy water -- BUC is encoded in the name itself
        # (vendor objnam.c:841-843), so suppress the BUC prefix for that case.
        # Exception 2: COIN_CLASS -- vendor objnam.c:1318 explicitly excludes
        # coins from the bknown BUC branch (``obj->oclass != COIN_CLASS``).
        is_water_special = (
            (category == jnp.int32(_POTION_CLASS_VAL)) &
            (type_id  == jnp.int32(_POT_WATER_TYPE_ID)) &
            identified &
            ((buc_status == jnp.int32(BUCStatus.BLESSED)) |
             (buc_status == jnp.int32(BUCStatus.CURSED)))
        )
        is_coin = category == jnp.int32(_COIN_CLASS_VAL)
        show_buc = buc_known & ~is_water_special & ~is_coin
        b, c = lax.cond(
            show_buc,
            lambda bc: _write_buc(bc[0], bc[1], buc_row),
            lambda bc: bc,
            (b, c),
        )

        # 3a. Vendor objnam.c prefix tokens that go into the ``prefix`` buffer
        # BEFORE add_erosion_words / enchant.  Order mirrors vendor:
        #   greased  (objnam.c:1370-1371) — any item, ungated.
        #   poisoned (objnam.c:1419-1420) — WEAPON_CLASS only, ispoisoned flag.
        #   partly eaten (objnam.c:1505-1506) — FOOD_CLASS only, oeaten!=0.
        # All three are unconditional in the sense that vendor doesn't gate
        # them on identification (the player sees them whenever the bit is
        # set on the object).
        # ``getattr`` guards against legacy Item arrays that pre-date these
        # bitfields, broadcasting a scalar default across all 52 slots so
        # safe_idx indexing always works.
        _greased_raw = getattr(inv_state.items, "greased",
                               jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_))
        _greased_arr = jnp.broadcast_to(
            jnp.asarray(_greased_raw, dtype=jnp.bool_),
            (MAX_INVENTORY_SLOTS,),
        )
        is_greased = _greased_arr[safe_idx]
        b, c = lax.cond(
            is_greased,
            lambda bc: _write_fixed(bc[0], bc[1], _GREASED_BYTES, 9),
            lambda bc: bc,
            (b, c),
        )

        _opoisoned_raw = getattr(inv_state.items, "opoisoned",
                                 jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_))
        _opoisoned_arr = jnp.broadcast_to(
            jnp.asarray(_opoisoned_raw, dtype=jnp.bool_),
            (MAX_INVENTORY_SLOTS,),
        )
        is_weapon_cat = category == jnp.int32(_WEAPON_CLASS_VAL)
        show_poisoned = is_weapon_cat & _opoisoned_arr[safe_idx]
        b, c = lax.cond(
            show_poisoned,
            lambda bc: _write_fixed(bc[0], bc[1], _POISONED_BYTES, 10),
            lambda bc: bc,
            (b, c),
        )

        _oeaten_raw = getattr(inv_state.items, "oeaten",
                              jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8))
        _oeaten_arr = jnp.broadcast_to(
            jnp.asarray(_oeaten_raw, dtype=jnp.int8),
            (MAX_INVENTORY_SLOTS,),
        )
        is_food_cat = category == jnp.int32(_FOOD_CLASS_VAL)
        show_partly_eaten = is_food_cat & (
            _oeaten_arr[safe_idx].astype(jnp.int32) > jnp.int32(0)
        )
        b, c = lax.cond(
            show_partly_eaten,
            lambda bc: _write_fixed(bc[0], bc[1], _PARTLY_EATEN_BYTES, 14),
            lambda bc: bc,
            (b, c),
        )

        # 3b. Erosion prefix (rusty/burnt/corroded/rotted/rustproof/fireproof/...)
        # Canonical: vendor/nethack/src/objnam.c::add_erosion_words() lines 1142-1191.
        # Read erosion fields from the item; clamp levels to 0-3.
        oeroded      = inv_state.items.oeroded[safe_idx].astype(jnp.int32)
        oeroded2     = inv_state.items.oeroded2[safe_idx].astype(jnp.int32)
        oerodeproof  = inv_state.items.oerodeproof[safe_idx]
        erod_lvl1 = jnp.clip(oeroded,  0, 3)
        erod_lvl2 = jnp.clip(oeroded2, 0, 3)
        emat_class = _OBJECT_EROSION_CLASS[safe_type].astype(jnp.int32)
        has_erosion = (emat_class > jnp.int32(0)) & (
            (erod_lvl1 > jnp.int32(0)) | (erod_lvl2 > jnp.int32(0)) | oerodeproof
        )
        b, c = lax.cond(
            has_erosion,
            lambda bc: _write_erosion(bc[0], bc[1], emat_class, erod_lvl1, erod_lvl2, oerodeproof, identified),
            lambda bc: bc,
            (b, c),
        )

        # 4. Enchantment -- weapons, armor, and oc_charged rings, only if identified.
        # vendor/nethack/src/objnam.c:1500: rings with oc_charged show +N prefix.
        obj_class   = _OBJECT_CLASS[jnp.clip(type_id, 0, _MAX_OBJ - 1)]
        is_charged_ring = _OBJECT_IS_CHARGED[safe_type]
        show_enchant = identified & (
            (obj_class == jnp.uint8(_WEAPON_CLASS_VAL)) |
            (obj_class == jnp.uint8(_ARMOR_CLASS_VAL))  |
            ((obj_class == jnp.uint8(_RING_CLASS_VAL)) & is_charged_ring)
        )
        b, c = lax.cond(
            show_enchant,
            lambda bc: _write_enchant(bc[0], bc[1], enchantment),
            lambda bc: bc,
            (b, c),
        )

        # 5. Class prefix (e.g. "potion of ") + name or appearance.
        # Special cases handled inside _write_true_name:
        #   - holy/unholy water (vendor objnam.c:841-843)
        #   - corpse name (vendor objnam.c:1824)
        #   - tin contents (vendor eat.c:tin_details)
        # Unidentified + has appearance  -> appearance description (no prefix)
        # Identified OR no appearance    -> class prefix (if any) + canonical name
        b, c = lax.cond(
            show_app,
            lambda bc: _write_appearance(bc[0], bc[1], safe_type, quantity),
            lambda bc: _write_true_name(bc[0], bc[1], safe_type, obj_class,
                                        quantity, category, type_id,
                                        buc_status, identified, enchantment,
                                        corpse_idx),
            (b, c),
        )

        # 5b. User-given name suffix " named <name>" (Wave 6) -- emitted
        #     when inventory.user_names[slot, 0] != 0.
        name_row = inv_state.user_names[safe_idx]
        has_user_name = name_row[0] != jnp.int8(0)
        b, c = lax.cond(
            has_user_name,
            lambda bc: _write_user_name(bc[0], bc[1], name_row),
            lambda bc: bc,
            (b, c),
        )

        # 6. Equip status -- pass item_class so quiver can pick pouch vs bow-ammo.
        eq_idx = _equip_status_idx(inv_state, slot_idx, obj_class)
        b, c = lax.cond(
            eq_idx > jnp.int32(0),
            lambda bc: _write_equip(bc[0], bc[1], eq_idx),
            lambda bc: bc,
            (b, c),
        )

        # 7. Charges "(recharged:charges)" for wands/tools, only if identified.
        # vendor/nethack/src/objnam.c:1486
        show_charges = identified & (
            (obj_class == jnp.uint8(_WAND_CLASS_VAL)) |
            (obj_class == jnp.uint8(_TOOL_CLASS_VAL))
        )
        b, c = lax.cond(
            show_charges,
            lambda bc: _write_charges(bc[0], bc[1], recharged, charges),
            lambda bc: bc,
            (b, c),
        )

        # 8. Two-weapon "(alternate weapon)" marker (Wave 6).  Emitted only
        #    when state.combat.two_weapon is True AND this slot matches
        #    state.inventory.alternate_weapon_slot.
        is_alt = two_weapon & (slot_idx.astype(jnp.int32) == alt_slot.astype(jnp.int32))
        b, c = lax.cond(
            is_alt,
            lambda bc: _write_alt_weapon(bc[0], bc[1]),
            lambda bc: bc,
            (b, c),
        )

        return b, c

    buf, cursor = lax.cond(
        is_empty,
        lambda args: args,
        render_nonempty,
        (buf, cursor),
    )

    return buf


# ---------------------------------------------------------------------------
# Rendering sub-helpers (called inside lax.cond; must be JIT-compatible)
# ---------------------------------------------------------------------------

def _write_uint_space(buf, cursor, qty):
    """Write integer quantity followed by a space."""
    buf, cursor = _write_uint(buf, cursor, qty)
    buf, cursor = _write_space(buf, cursor)
    return buf, cursor


def _write_a_space(buf, cursor):
    """Write 'a ' (singular article, simplified -- no vowel check)."""
    buf, cursor = _write_byte(buf, cursor, jnp.uint8(ord('a')))
    buf, cursor = _write_space(buf, cursor)
    return buf, cursor


def _write_article_space(buf, cursor, use_an):
    """Write 'a ' or 'an ' based on *use_an* boolean.

    Full port of vendor/nethack/src/objnam.c::just_an() lines 2108-2142.
    *use_an* is pre-computed at module load time via _OBJECT_USE_AN /
    _APP_USE_AN tables (which encode the long-u / 'one' / single-letter /
    x-consonant exceptions), so this function stays JIT-compatible while
    applying all vendor exception rules.
    """
    # Write 'a' always, conditionally append 'n', then a space.
    buf, cursor = _write_byte(buf, cursor, jnp.uint8(ord('a')))
    buf, cursor = lax.cond(
        use_an,
        lambda bc: _write_byte(bc[0], bc[1], jnp.uint8(ord('n'))),
        lambda bc: bc,
        (buf, cursor),
    )
    buf, cursor = _write_space(buf, cursor)
    return buf, cursor


def _write_buc(buf, cursor, buc_row):
    """Write BUC word followed by a space."""
    buc_src = _BUC_BYTES[buc_row]
    buf, cursor = _write_fixed(buf, cursor, buc_src, _MAX_BUC_LEN)
    buf, cursor = _write_space(buf, cursor)
    return buf, cursor


def _write_erosion(buf, cursor, emat_class, erod_lvl1, erod_lvl2, oerodeproof, identified):
    """Write erosion prefix words for an item.

    Mirrors vendor/nethack/src/objnam.c::add_erosion_words() lines 1142-1191.

    oeroded  (erod_lvl1 1-3) -> rusty/burnt series based on emat_class.
    oeroded2 (erod_lvl2 1-3) -> corroded/rotted series.
    oerodeproof (rknown)     -> rustproof/fireproof/corrodeproof/rotproof, only
                                 when identified (vendor objnam.c:1183: rknown &&
                                 oerodeproof; rknown mirrors identified here).
    All words are table-looked-up (JIT-compatible; no Python branching at trace).
    """
    # oeroded word (rust/burn)
    oeroded_src = _EROSION_OERODED_BYTES[emat_class, erod_lvl1]
    buf, cursor = _write_fixed(buf, cursor, oeroded_src, _MAX_EROSION_WORD_LEN)

    # oeroded2 word (corrode/rot)
    oeroded2_src = _EROSION_OERODED2_BYTES[emat_class, erod_lvl2]
    buf, cursor = _write_fixed(buf, cursor, oeroded2_src, _MAX_EROSION_WORD_LEN)

    # proof prefix: rustproof/fireproof/corrodeproof (indexed by emat_class).
    # For mat_class 2 (flammable), oeroded2 controls "rotproof" separately.
    # vendor objnam.c:1183: rknown && oerodeproof — only show when identified.
    show_proof = oerodeproof & identified
    proof_src = _EROSION_PROOF_BYTES[emat_class]
    is_rotproof = show_proof & (emat_class == jnp.int32(2)) & (erod_lvl2 > jnp.int32(0))
    buf, cursor = lax.cond(
        show_proof & ~is_rotproof,
        lambda bc: _write_fixed(bc[0], bc[1], proof_src, _MAX_EROSION_WORD_LEN),
        lambda bc: bc,
        (buf, cursor),
    )
    buf, cursor = lax.cond(
        is_rotproof,
        lambda bc: _write_fixed(bc[0], bc[1], _ROTPROOF_BYTES, _MAX_EROSION_WORD_LEN),
        lambda bc: bc,
        (buf, cursor),
    )
    return buf, cursor


def _write_enchant(buf, cursor, enchantment):
    """Write '+N' or '-N' enchantment followed by a space."""
    is_neg = enchantment < jnp.int32(0)
    sign   = jnp.where(is_neg, jnp.uint8(ord('-')), jnp.uint8(ord('+')))
    abs_enc = jnp.abs(enchantment).astype(jnp.int32)
    buf, cursor = _write_byte(buf, cursor, sign)
    buf, cursor = _write_uint(buf, cursor, abs_enc)
    buf, cursor = _write_space(buf, cursor)
    return buf, cursor


def _write_appearance(buf, cursor, safe_type, quantity):
    """Write appearance description (for unidentified items).

    Uses the object's ``description`` field, e.g. "orange" for an unidentified
    potion.  The appearance is augmented with the class noun ("potion",
    "scroll", etc.) from the name.  When ``quantity > 1`` the appearance
    table is sourced from the pluralized form (e.g. "elven dagger" ->
    "elven daggers") and the class noun gets an "s" appended for regular-
    plural class nouns.
    """
    is_plural = quantity > jnp.int32(1)

    # Choose between singular and pluralized appearance row.  Both source
    # tables are zero-padded to the same width (_MAX_PLURAL_APP_LEN) so
    # they have a common shape for jnp.where.
    sing_row = _APPEARANCE_BYTES_PADDED[safe_type]
    plur_row = _APPEARANCE_PLURAL_BYTES[safe_type]
    app_src = jnp.where(is_plural, plur_row, sing_row)
    buf, cursor = _write_fixed(buf, cursor, app_src, _MAX_PLURAL_APP_LEN)

    # Append class noun (" potion", " scroll", etc.) so callers reading
    # appearance-only descriptions still see the class.
    obj_class = _OBJECT_CLASS[safe_type].astype(jnp.int32)
    buf, cursor = _write_fixed(
        buf, cursor,
        _CLASS_NOUN_BYTES[obj_class],
        _MAX_CLASS_NOUN_LEN,
    )

    # Pluralise the class noun on the appearance path by appending 's' when
    # quantity > 1 (all class nouns are regular: "potions", "scrolls", ...).
    buf, cursor = lax.cond(
        is_plural,
        lambda bc: _write_byte(bc[0], bc[1], jnp.uint8(ord('s'))),
        lambda bc: bc,
        (buf, cursor),
    )

    return buf, cursor


def _write_true_name(buf, cursor, safe_type, obj_class, quantity,
                     category, type_id, buc_status, identified,
                     enchantment, corpse_idx):
    """Write class prefix (if any) + canonical object name.

    Handles special cases (host-side Python control flow OK per spec):
    - Holy/unholy water: vendor objnam.c:841-843 -- blessed POT_WATER -> "holy water",
      cursed -> "unholy water".  No BUC prefix (suppressed in _render_slot step 3).
    - Corpse name: vendor objnam.c:1824 -- "<monster> corpse".
    - Tin contents: vendor eat.c:tin_details -- "tin of spinach" / "tin of X meat".

    Wave 6: when ``quantity > 1`` the precomputed pluralized form is written
    instead of the singular + 's' fallback.  Handles irregular plurals like
    knife->knives, staff->staves, man->men.  See ``pluralize`` for the rules.
    """
    cls_int = obj_class.astype(jnp.int32)
    cls_safe = jnp.clip(cls_int, 0, 17).astype(jnp.int32)
    is_plural = quantity > jnp.int32(1)

    # --- Special case: holy / unholy water (vendor objnam.c:841-843) ---
    # Identified blessed POT_WATER -> "holy water" (no "blessed " prefix).
    # Identified cursed  POT_WATER -> "unholy water" (no "cursed " prefix).
    is_water_special = (
        (category == jnp.int32(_POTION_CLASS_VAL)) &
        (type_id  == jnp.int32(_POT_WATER_TYPE_ID)) &
        identified &
        ((buc_status == jnp.int32(BUCStatus.BLESSED)) |
         (buc_status == jnp.int32(BUCStatus.CURSED)))
    )
    is_holy   = is_water_special & (buc_status == jnp.int32(BUCStatus.BLESSED))
    is_unholy = is_water_special & (buc_status == jnp.int32(BUCStatus.CURSED))

    def write_holy(bc):
        return _write_fixed(bc[0], bc[1], _HOLY_WATER_BYTES, 11)

    def write_unholy(bc):
        return _write_fixed(bc[0], bc[1], _UNHOLY_WATER_BYTES, 13)

    # --- Special case: corpse name (vendor objnam.c:1824) ---
    # FOOD_CLASS + type_id==CORPSE + corpse_entry_idx >= 0 -> "<monster> corpse".
    is_corpse = (
        (category == jnp.int32(_FOOD_CLASS_VAL)) &
        (type_id  == jnp.int32(_CORPSE_TYPE_ID)) &
        (corpse_idx >= jnp.int32(0))
    )
    safe_monster_idx = jnp.clip(corpse_idx, 0, _NUM_MONSTERS - 1).astype(jnp.int32)

    def write_corpse(bc):
        b, c = bc
        mon_src = _MONSTER_NAME_BYTES[safe_monster_idx]
        b, c = _write_fixed(b, c, mon_src, _MAX_MONSTER_NAME_LEN)
        b, c = _write_fixed(b, c, _CORPSE_SUFFIX_BYTES, 8)
        return b, c

    # --- Special case: tin contents (vendor eat.c:tin_details) ---
    # FOOD_CLASS + type_id==TIN + identified:
    #   enchantment==1 (spe==1 in vendor) -> "tin of spinach"
    #   corpse_idx >= 0                   -> "tin of <monster> meat"
    #   otherwise                         -> "tin" (empty / unknown)
    is_tin = (
        (category == jnp.int32(_FOOD_CLASS_VAL)) &
        (type_id  == jnp.int32(_TIN_TYPE_ID)) &
        identified
    )
    is_spinach_tin = is_tin & (enchantment == jnp.int32(1))
    is_monster_tin = is_tin & (corpse_idx >= jnp.int32(0)) & ~is_spinach_tin

    def write_spinach_tin(bc):
        b, c = bc
        b, c = _write_fixed(b, c, _TIN_OF_BYTES, 8)
        b, c = _write_fixed(b, c, _SPINACH_BYTES, 8)
        return b, c

    def write_monster_tin(bc):
        b, c = bc
        b, c = _write_fixed(b, c, _TIN_OF_BYTES, 8)
        mon_src = _MONSTER_NAME_BYTES[safe_monster_idx]
        b, c = _write_fixed(b, c, mon_src, _MAX_MONSTER_NAME_LEN)
        b, c = _write_fixed(b, c, _MEAT_BYTES, 6)
        return b, c

    # --- Normal path: class prefix + canonical name ---
    # Vendor objnam.c:721-726 + 694-695 — "pair of "/"pairs of "/"set of "
    # prefix for boots/gloves/lenses/dragon scales.  Looked up via per-otyp
    # _PAIR_OF / _SET_OF tables.  Set-of dragon scales always uses the
    # singular noun (vendor sprintf"set of %s",actualn; no plural path).
    is_pair = _PAIR_OF[safe_type]
    is_set  = _SET_OF[safe_type]

    def write_normal(bc):
        b, c = bc
        # Step 1: "pair of "/"pairs of "/"set of " noun-cluster prefix
        # (precedes any class-of prefix, and applies even when plural).
        def write_pair(_bc):
            return lax.cond(
                is_plural,
                lambda x: _write_fixed(x[0], x[1], _PAIRS_OF_BYTES, 10),
                lambda x: _write_fixed(x[0], x[1], _PAIR_OF_BYTES, 9),
                _bc,
            )

        def write_set(_bc):
            return _write_fixed(_bc[0], _bc[1], _SET_OF_BYTES, 8)

        b, c = lax.cond(
            is_pair,
            write_pair,
            lambda _bc: lax.cond(is_set, write_set, lambda x: x, _bc),
            (b, c),
        )

        # Step 2: class-of prefix ("ring of ", "potion of ", ...).  Only when
        # singular (vendor: makeplural pluralizes the head noun, not the prefix).
        pfx_src = _CLASS_PREFIX_BYTES[cls_safe]
        b, c = lax.cond(
            is_plural,
            lambda _bc: _bc,
            lambda _bc: _write_fixed(_bc[0], _bc[1], pfx_src, _MAX_PREFIX_LEN),
            (b, c),
        )

        # Step 3: canonical name (singular or pluralised).  For set-of dragon
        # scales the singular row is used unconditionally (vendor objnam.c:722
        # writes actualn directly).
        sing_row = _OBJECT_NAMES_BYTES_PADDED[safe_type]
        plur_row = _NAME_PLURAL_BYTES[safe_type]
        name_src = jnp.where(is_plural & ~is_set, plur_row, sing_row)
        b, c = _write_fixed(b, c, name_src, _MAX_PLURAL_NAME_LEN)
        return b, c

    # Dispatch: water_special > corpse > spinach_tin > monster_tin > normal.
    buf, cursor = lax.cond(
        is_holy,
        write_holy,
        lambda bc: lax.cond(
            is_unholy,
            write_unholy,
            lambda bc2: lax.cond(
                is_corpse,
                write_corpse,
                lambda bc3: lax.cond(
                    is_spinach_tin,
                    write_spinach_tin,
                    lambda bc4: lax.cond(
                        is_monster_tin,
                        write_monster_tin,
                        write_normal,
                        bc4,
                    ),
                    bc3,
                ),
                bc2,
            ),
            bc,
        ),
        (buf, cursor),
    )

    return buf, cursor


def _write_equip(buf, cursor, eq_idx):
    """Write equip status string preceded by a space."""
    buf, cursor = _write_space(buf, cursor)
    eq_src = _EQUIP_BYTES[eq_idx]
    buf, cursor = _write_fixed(buf, cursor, eq_src, _MAX_EQUIP_LEN)
    return buf, cursor


def _write_user_name(buf, cursor, name_row):
    """Append ' named <user_name>' to the buffer.

    *name_row* is a uint8[USER_NAME_LEN] zero-terminated byte string from
    ``state.inventory.user_names[slot]``.  Mirrors the ' named X' suffix
    emitted by vendor/nethack/src/objnam.c::doname when ONAME is set.
    """
    buf, cursor = _write_fixed(buf, cursor, _NAMED_PREFIX_BYTES, 8)
    buf, cursor = _write_fixed(buf, cursor, name_row.astype(jnp.uint8), USER_NAME_LEN)
    return buf, cursor


def _write_alt_weapon(buf, cursor):
    """Append ' (alternate weapon; not wielded)' to the buffer.

    Vendor objnam.c line 1619 emits ' (alternate weapon; not wielded)'.
    Emitted when ``state.combat.two_weapon`` is True and the slot matches
    ``state.inventory.alternate_weapon_slot``.
    """
    buf, cursor = _write_fixed(buf, cursor, _ALT_WEAPON_BYTES, 33)
    return buf, cursor


def _write_charges(buf, cursor, recharged, charges):
    """Write charge counter as ' (recharged:charges)'.

    vendor/nethack/src/objnam.c:1486:
        ConcatF2(bp, 0, " (%d:%d)", (int) obj->recharged, obj->spe)
    where obj->recharged is the recharge count and obj->spe is remaining charges.
    """
    buf, cursor = _write_space(buf, cursor)
    buf, cursor = _write_byte(buf, cursor, jnp.uint8(ord('(')))
    buf, cursor = _write_uint(buf, cursor, jnp.clip(recharged, 0, 99).astype(jnp.int32))
    buf, cursor = _write_byte(buf, cursor, jnp.uint8(ord(':')))
    buf, cursor = _write_uint(buf, cursor, jnp.clip(charges, 0, 99).astype(jnp.int32))
    buf, cursor = _write_byte(buf, cursor, jnp.uint8(ord(')')))
    return buf, cursor


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def build_inv_strs(state) -> jnp.ndarray:
    """Build the NLE ``inv_strs`` observation field.

    Renders all 55 inventory slots as ASCII strings packed into a
    ``uint8[55, 80]`` array.  Empty slots produce all-zero rows.

    JIT-compatible: all branching via ``lax.cond``; loop via ``lax.map``
    (implemented as ``vmap`` over slot indices).

    Args:
        state: nethax ``EnvState`` (must have ``inventory`` and
               ``identification`` subsystem fields).

    Returns:
        uint8[55, 80]
    """
    inv_state = state.inventory
    id_state  = state.identification

    # Wave 6: read state.combat.two_weapon and inventory.alternate_weapon_slot
    # so each slot can emit the " (alternate weapon)" marker when appropriate.
    # Graceful default for legacy callers that pass a minimal state object
    # without a combat field (used by some Wave 3 tests).
    combat_state = getattr(state, "combat", None)
    if combat_state is None or not hasattr(combat_state, "two_weapon"):
        two_weapon = jnp.bool_(False)
    else:
        two_weapon = combat_state.two_weapon
    alt_slot = getattr(inv_state, "alternate_weapon_slot", jnp.int8(-1))
    alt_slot_i32 = jnp.asarray(alt_slot).astype(jnp.int32)

    # vmap over slot indices [0..54]
    slot_indices = jnp.arange(NLE_INV_SLOTS, dtype=jnp.int32)

    # lax.map applies f sequentially (lower memory than vmap for large arrays)
    def render_one(slot_idx):
        return _render_slot(inv_state, id_state, slot_idx, two_weapon, alt_slot_i32)

    result = lax.map(render_one, slot_indices)  # uint8[55, 80]
    return result


# ---------------------------------------------------------------------------
# Wave 8c: Grouped inventory rendering (vendor `i` menu parity)
# ---------------------------------------------------------------------------
#
# Vendor's display_inventory groups inventory by ObjectClass and prints a class
# header before each group, in the order given by `flags.inv_order`
# (vendor/nethack/src/options.c::def_inv_order).  Each item line is the
# 'doname'-formatted string (the same one already produced by
# ``build_inv_strs``).
#
# Header strings come from invent.c::names[] (lines 4789-4793):
#   {0, "Illegal objects", "Weapons", "Armor", "Rings", "Amulets", "Tools",
#    "Comestibles", "Potions", "Scrolls", "Spellbooks", "Wands", "Coins",
#    "Gems/Stones", "Boulders/Statues", "Iron balls", "Chains", "Venoms"}
#
# Default packing order (options.c::def_inv_order, line 118):
#   COIN_CLASS, AMULET_CLASS, WEAPON_CLASS, ARMOR_CLASS, FOOD_CLASS,
#   SCROLL_CLASS, SPBOOK_CLASS, POTION_CLASS, RING_CLASS, WAND_CLASS,
#   TOOL_CLASS, GEM_CLASS, ROCK_CLASS, BALL_CLASS, CHAIN_CLASS
# ---------------------------------------------------------------------------

# Class -> header string  (vendor invent.c::names[])
_CLASS_HEADERS: dict[int, str] = {
    int(ObjectClass.ILLOBJ_CLASS):  "Illegal objects",
    int(ObjectClass.WEAPON_CLASS):  "Weapons",
    int(ObjectClass.ARMOR_CLASS):   "Armor",
    int(ObjectClass.RING_CLASS):    "Rings",
    int(ObjectClass.AMULET_CLASS):  "Amulets",
    int(ObjectClass.TOOL_CLASS):    "Tools",
    int(ObjectClass.FOOD_CLASS):    "Comestibles",
    int(ObjectClass.POTION_CLASS):  "Potions",
    int(ObjectClass.SCROLL_CLASS):  "Scrolls",
    int(ObjectClass.SPBOOK_CLASS):  "Spellbooks",
    int(ObjectClass.WAND_CLASS):    "Wands",
    int(ObjectClass.COIN_CLASS):    "Coins",
    int(ObjectClass.GEM_CLASS):     "Gems/Stones",
    int(ObjectClass.ROCK_CLASS):    "Boulders/Statues",
    int(ObjectClass.BALL_CLASS):    "Iron balls",
    int(ObjectClass.CHAIN_CLASS):   "Chains",
    int(ObjectClass.VENOM_CLASS):   "Venoms",
}

# Default inv-order from vendor options.c::def_inv_order (line 118).
_DEFAULT_INV_ORDER: tuple = (
    int(ObjectClass.COIN_CLASS),
    int(ObjectClass.AMULET_CLASS),
    int(ObjectClass.WEAPON_CLASS),
    int(ObjectClass.ARMOR_CLASS),
    int(ObjectClass.FOOD_CLASS),
    int(ObjectClass.SCROLL_CLASS),
    int(ObjectClass.SPBOOK_CLASS),
    int(ObjectClass.POTION_CLASS),
    int(ObjectClass.RING_CLASS),
    int(ObjectClass.WAND_CLASS),
    int(ObjectClass.TOOL_CLASS),
    int(ObjectClass.GEM_CLASS),
    int(ObjectClass.ROCK_CLASS),
    int(ObjectClass.BALL_CLASS),
    int(ObjectClass.CHAIN_CLASS),
)


def _decode_row(row) -> str:
    """Decode a uint8[80] row (or numpy array) into a Python string, trimming nulls."""
    import numpy as np
    arr = np.asarray(row).astype(np.uint8)
    return bytes(arr.tolist()).rstrip(b"\x00").decode("ascii", errors="replace")


def build_grouped_inv_text(state) -> list[str]:
    """Vendor-parity ``i`` menu output: class-grouped inventory listing.

    Returns a list of text lines: each class with at least one non-empty
    inventory slot produces a header line (e.g. "Weapons") followed by one
    item line per slot in that class, in slot-letter order.

    Mirrors vendor/nethack/src/invent.c::display_pickinv (~line 3266+):
        for each class in flags.inv_order:
            print class header (let_to_name)
            for each item whose oclass matches:
                print doname(item)

    Host-side helper: NOT jitted; intended for human-readable debug dumps
    and parity tests against the vendor menu.
    """
    # Build the flat doname-strings via the existing JITed builder.
    inv_rows = build_inv_strs(state)  # uint8[55, 80]
    inv = state.inventory
    items = inv.items

    # Extract per-slot class id (only first MAX_INVENTORY_SLOTS are real).
    import numpy as np
    categories = np.asarray(items.category).astype(np.int32)

    lines: list[str] = []
    for class_id in _DEFAULT_INV_ORDER:
        # Collect slots in this class, in invlet (slot index) order.
        slot_idxs = [
            i for i in range(MAX_INVENTORY_SLOTS) if int(categories[i]) == class_id
        ]
        if not slot_idxs:
            continue
        header = _CLASS_HEADERS.get(class_id, "Items")
        lines.append(header)
        for i in slot_idxs:
            row_str = _decode_row(inv_rows[i])
            if row_str:
                lines.append(row_str)
    return lines
