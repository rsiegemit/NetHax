"""NLE-fidelity inventory string rendering in JAX.

Produces ``inv_strs: uint8[55, 80]`` — 55 inventory slots, each an 80-byte
null-terminated ASCII string matching the format produced by NetHack's
``doname``/``xname`` (vendor/nethack/src/objnam.c).

String structure per slot::

    <letter> - <qty_word> <buc_word> <enchant> <name> <equip_status> <charges>

Wave 3 simplifications (documented per feature):
- article "a" always used for singular items (no a/an vowel check — simplified)
- User-given names (oname/oextra) skipped entirely
- Two-weapon "alternate weapon" status skipped
- Charges shown as "(N:M)" where N=charges, M=max_charges (max pinned at 8)
- Slots 52-54 (NLE extras beyond a-zA-Z) always rendered empty

Canonical sources:
  vendor/nethack/src/objnam.c  — doname / xname / an
  vendor/nethack/src/invent.c  — display_inventory
  vendor/nle/include/nleobs.h  — NLE_INVENTORY_SIZE=55, NLE_INVENTORY_STR_LENGTH=80
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from Nethax.nethax.constants.objects import OBJECTS, ObjectClass
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
_MAX_EQUIP_LEN      = 24   # "(weapon in hand)\0" = 17 bytes
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
# 5: (in quiver)
# 6: (on left hand) — same as 3, used for worn amulet display
# 7: (being worn)   — alias for body armor
_EQUIP_STRS = [
    "",
    "(weapon in hand)",
    "(being worn)",
    "(on left hand)",
    "(on right hand)",
    "(in quiver)",
    "(being worn)",   # amulet
    "(being worn)",   # shirt / cloak (all non-ring/non-weapon armor)
]
_EQUIP_BYTES: jnp.ndarray = jnp.array(
    [_pad_bytes(s, _MAX_EQUIP_LEN) for s in _EQUIP_STRS],
    dtype=jnp.uint8,
)  # uint8[8, _MAX_EQUIP_LEN]

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


# Suffix-level irregular plurals (apply to compound words too:
# crysknife → crysknives, midwife → midwives).
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
    otherwise.  Pure Python — invoked only at module import time to populate
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
    (vendor/nle/src/objects.c lines 855-875 and 1044-1046 — 23 None-named
    entries with only a description).

    Handles the "X of Y" pattern (e.g. "ring of protection" → "rings of
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
    # No " of " — pluralize the trailing word.
    sp = name.rfind(" ")
    if sp >= 0:
        return name[: sp + 1] + pluralize(name[sp + 1 :])
    return pluralize(name)


# Plural-name table — one row per object, padded to a width wide enough to
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

# Plural-appearance table — pluralize the last word of the description so
# "wooden" → "wooden" (no last word change needed; an "s" comes via the class
# noun route) but "elven dagger" → "elven daggers".  When the description is
# None/empty the row is all zeros (and is unused — has_appearance handles
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
_ALT_WEAPON_BYTES: jnp.ndarray = jnp.array(
    _pad_bytes(" (alternate weapon)", 20), dtype=jnp.uint8,
)  # uint8[20]


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

def _equip_status_idx(inv_state, slot_idx: jax.Array) -> jax.Array:
    """Return the equip-status table index (int32) for the given inventory slot.

    0 = no status, 1 = weapon in hand, 2 = being worn (body/helm/gloves/boots/
    cloak/shirt), 3 = on left hand (ring), 4 = on right hand (ring),
    5 = in quiver, 6 = being worn (amulet).

    JIT-compatible: only jnp.where / == comparisons.
    """
    i8 = slot_idx.astype(jnp.int8)

    # Wielded weapon
    is_wielded = inv_state.wielded == i8

    # Worn armor — any of the 7 slots
    # worn_armor[j] == slot_idx  means slot is worn in armor position j
    is_armor_worn = jnp.any(inv_state.worn_armor == i8)

    # Rings — left = worn_rings[0], right = worn_rings[1]
    is_ring_l = inv_state.worn_rings[0] == i8
    is_ring_r = inv_state.worn_rings[1] == i8

    # Amulet
    is_amulet = inv_state.worn_amulet == i8

    # Quiver
    is_quiver = inv_state.quiver == i8

    result = jnp.int32(0)
    result = jnp.where(is_quiver,    jnp.int32(5), result)
    result = jnp.where(is_amulet,    jnp.int32(6), result)
    result = jnp.where(is_ring_r,    jnp.int32(4), result)
    result = jnp.where(is_ring_l,    jnp.int32(3), result)
    result = jnp.where(is_armor_worn, jnp.int32(2), result)
    result = jnp.where(is_wielded,   jnp.int32(1), result)
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
        two_weapon: bool scalar — state.combat.two_weapon flag (Wave 6).
        alt_slot:   int32 scalar — state.inventory.alternate_weapon_slot.

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
    identified  = inv_state.items.identified[safe_idx]
    quantity    = inv_state.items.quantity[safe_idx].astype(jnp.int32)

    # Slot is empty when category == 0 OR slot_idx >= MAX_INVENTORY_SLOTS
    is_empty = (category == jnp.int32(0)) | (slot_idx >= jnp.int32(MAX_INVENTORY_SLOTS))

    # Precompute the first byte that determines the a/an article.  The
    # article precedes BUC/enchant/name, so we pick the first byte of
    # whichever segment will actually be written first:
    #   - BUC word (if buc known)
    #   - appearance description (if unidentified + has appearance)
    #   - canonical name (otherwise)
    safe_type = jnp.clip(type_id, 0, _MAX_OBJ - 1).astype(jnp.int32)
    has_app   = _HAS_APPEARANCE[safe_type]
    buc_known = buc_status != jnp.int32(BUCStatus.UNKNOWN)
    buc_row   = jnp.clip(buc_status, 0, 3).astype(jnp.int32)
    buc_first = _BUC_BYTES[buc_row, 0]
    app_first = _APPEARANCE_BYTES[safe_type, 0]
    name_first = _OBJECT_NAMES_BYTES[safe_type, 0]
    show_app  = (~identified) & has_app
    noun_first = jnp.where(show_app, app_first, name_first)
    article_first = jnp.where(buc_known, buc_first, noun_first)

    def render_nonempty(args):
        b, c = args

        # 1. Letter prefix: "a - "
        letter = _LETTERS[slot_idx]
        b, c = _write_byte(b, c, letter)
        b, c = _write_byte(b, c, jnp.uint8(ord(' ')))
        b, c = _write_byte(b, c, jnp.uint8(ord('-')))
        b, c = _write_byte(b, c, jnp.uint8(ord(' ')))

        # 2. Quantity word: "<N> " for stacks, "a " or "an " for singletons.
        # Wave 6: vowel check via _VOWEL_MASK selects between 'a' and 'an'.
        is_plural = quantity > jnp.int32(1)
        b, c = lax.cond(
            is_plural,
            lambda bc: _write_uint_space(bc[0], bc[1], quantity),
            lambda bc: _write_article_space(bc[0], bc[1], article_first),
            (b, c),
        )

        # 3. BUC word (only if buc_status != UNKNOWN == 0)
        b, c = lax.cond(
            buc_known,
            lambda bc: _write_buc(bc[0], bc[1], buc_row),
            lambda bc: bc,
            (b, c),
        )

        # 4. Enchantment (only for weapon/armor, only if identified)
        obj_class   = _OBJECT_CLASS[jnp.clip(type_id, 0, _MAX_OBJ - 1)]
        show_enchant = identified & (
            (obj_class == jnp.uint8(_WEAPON_CLASS_VAL)) |
            (obj_class == jnp.uint8(_ARMOR_CLASS_VAL))
        )
        b, c = lax.cond(
            show_enchant,
            lambda bc: _write_enchant(bc[0], bc[1], enchantment),
            lambda bc: bc,
            (b, c),
        )

        # 5. Class prefix (e.g. "potion of ") + name or appearance
        # Unidentified + has appearance  -> appearance description (no prefix)
        # Identified OR no appearance    -> class prefix (if any) + canonical name
        b, c = lax.cond(
            show_app,
            lambda bc: _write_appearance(bc[0], bc[1], safe_type, quantity),
            lambda bc: _write_true_name(bc[0], bc[1], safe_type, obj_class, quantity),
            (b, c),
        )

        # 5b. User-given name suffix " named <name>" (Wave 6) — emitted
        #     when inventory.user_names[slot, 0] != 0.
        name_row = inv_state.user_names[safe_idx]
        has_user_name = name_row[0] != jnp.int8(0)
        b, c = lax.cond(
            has_user_name,
            lambda bc: _write_user_name(bc[0], bc[1], name_row),
            lambda bc: bc,
            (b, c),
        )

        # 6. Equip status
        eq_idx = _equip_status_idx(inv_state, slot_idx)
        b, c = lax.cond(
            eq_idx > jnp.int32(0),
            lambda bc: _write_equip(bc[0], bc[1], eq_idx),
            lambda bc: bc,
            (b, c),
        )

        # 7. Charges "(N:M)" for wands/tools, only if identified
        show_charges = identified & (
            (obj_class == jnp.uint8(_WAND_CLASS_VAL)) |
            (obj_class == jnp.uint8(_TOOL_CLASS_VAL))
        )
        b, c = lax.cond(
            show_charges,
            lambda bc: _write_charges(bc[0], bc[1], charges),
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
    """Write 'a ' (singular article, simplified — no vowel check)."""
    buf, cursor = _write_byte(buf, cursor, jnp.uint8(ord('a')))
    buf, cursor = _write_space(buf, cursor)
    return buf, cursor


def _write_article_space(buf, cursor, first_byte):
    """Write 'a ' or 'an ' based on whether *first_byte* starts a vowel.

    Mirrors vendor/nethack/src/objnam.c::an, which picks 'an' when the
    following word begins with a vowel ('a','e','i','o','u').
    """
    safe = first_byte.astype(jnp.int32)
    safe = jnp.clip(safe, 0, 255)
    is_vowel = _VOWEL_MASK[safe]
    # Write 'a' always, conditionally append 'n', then a space.
    buf, cursor = _write_byte(buf, cursor, jnp.uint8(ord('a')))
    buf, cursor = lax.cond(
        is_vowel,
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
    table is sourced from the pluralized form (e.g. "elven dagger" →
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


def _write_true_name(buf, cursor, safe_type, obj_class, quantity):
    """Write class prefix (if any) + canonical object name.

    Wave 6: when ``quantity > 1`` the precomputed pluralized form is written
    instead of the singular + 's' fallback.  Handles irregular plurals like
    knife→knives, staff→staves, man→men.  See ``pluralize`` for the rules.
    """
    cls_int = obj_class.astype(jnp.int32)
    cls_safe = jnp.clip(cls_int, 0, 17).astype(jnp.int32)

    # Class prefix ("potion of ", "ring of ", ...).  Class prefix is only
    # emitted for the singular form; pluralized names already include the
    # head noun (e.g. "rings of protection"), so we suppress the prefix
    # in the plural branch.
    is_plural = quantity > jnp.int32(1)
    pfx_src = _CLASS_PREFIX_BYTES[cls_safe]
    buf, cursor = lax.cond(
        is_plural,
        lambda bc: bc,
        lambda bc: _write_fixed(bc[0], bc[1], pfx_src, _MAX_PREFIX_LEN),
        (buf, cursor),
    )

    # Canonical name — choose between singular and pluralized rows.
    sing_row = _OBJECT_NAMES_BYTES_PADDED[safe_type]
    plur_row = _NAME_PLURAL_BYTES[safe_type]
    name_src = jnp.where(is_plural, plur_row, sing_row)
    buf, cursor = _write_fixed(buf, cursor, name_src, _MAX_PLURAL_NAME_LEN)

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
    """Append ' (alternate weapon)' to the buffer.

    Emitted when ``state.combat.two_weapon`` is True and the slot matches
    ``state.inventory.alternate_weapon_slot``.  Mirrors the two-weapon
    status marker shown by vendor/nethack/src/wield.c.
    """
    buf, cursor = _write_fixed(buf, cursor, _ALT_WEAPON_BYTES, 20)
    return buf, cursor


def _write_charges(buf, cursor, charges):
    """Write charge count as ' (N:8)' — max_charges pinned to 8 (simplified)."""
    # Simplified: max charges shown as 8 always; Wave 4 can track max_charges.
    buf, cursor = _write_space(buf, cursor)
    buf, cursor = _write_byte(buf, cursor, jnp.uint8(ord('(')))
    buf, cursor = _write_uint(buf, cursor, jnp.clip(charges, 0, 99).astype(jnp.int32))
    buf, cursor = _write_byte(buf, cursor, jnp.uint8(ord(':')))
    buf, cursor = _write_uint(buf, cursor, jnp.int32(8))
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
