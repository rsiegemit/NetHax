"""Wish subsystem — grant a wish for an object or artifact.

Canonical sources:
  vendor/nethack/src/zap.c::makewish        — interactive wish handler
  vendor/nethack/src/zap.c::WAN_WISHING     — wand-of-wishing dispatch (line 2575)
  vendor/nethack/src/objnam.c::readobjnam   — free-form wish-text parser
  vendor/nethack/src/objnam.c::wishymatch   — name + artifact resolution
  vendor/nethack/src/read.c::do_genocide    — wish-like genocide
  vendor/nethack/src/spell.c::wishcmdassist — wish command help

Vendor parity:
  ``wishymatch()`` ports the full vendor readobjnam grammar — BUC, holy/unholy,
  erodeproof, greased, enchantment ("+N"/"-N"), quantity prefix, "named X" /
  "called X" suffix, gold-piece short-forms, "the " article, plural -> singular
  normalization, fuzzy abbreviation (longsword, gdsm), nowish substitution,
  and artifact lookup with SPFX_RESTR alignment/XL gating.

  Grant creates the wished-for item in the first empty inventory slot (or
  drops it on the ground at the player position when the inventory is full),
  and sets WISHLESS / ARTIWISHLESS conducts.

JAX-required: the parser is Python-side (not JIT) because free-form string
matching is not expressible in jax.lax primitives.  Wish parsing happens at
action-handler dispatch time, never in the hot per-step loop, so the Python
fallback is sound.

Wired conducts:
    WISHLESS      — set on every successful grant (wishymatch parsed=True).
    ARTIWISHLESS  — set when the wish text matched an artifact name (vendor
                    readobjnam flips wisharti on the text match, not the
                    SPFX-gated grant — objnam.c:5362).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.constants.objects import OBJECTS, ObjectClass, OBJECT_NAME_ALIASES
from Nethax.nethax.subsystems.conduct import Conduct
from Nethax.nethax.subsystems.inventory import (
    MAX_GROUND_STACK,
    MAX_INVENTORY_SLOTS,
    USER_NAME_LEN,
)
from Nethax.nethax.subsystems.items import BUCStatus
from Nethax.nethax.subsystems.prayer import Alignment


# ---------------------------------------------------------------------------
# Artifact table — minimal subset for Wave 6.
#
# Mirrors vendor/nethack/include/artilist.h ordering (1-based in vendor;
# we use 0-based indices).  Each entry pairs the artifact name with the
# base object name used by the underlying Item.type_id.
# ---------------------------------------------------------------------------
_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("Excalibur",                "long sword"),                # 0
    ("Snickersnee",              "katana"),                    # 1
    ("Stormbringer",             "runesword"),                 # 2
    ("Mjollnir",                 "war hammer"),                # 3
    ("Cleaver",                  "battle-axe"),                # 4
    ("Sting",                    "elven dagger"),              # 5
    ("Orcrist",                  "elven broadsword"),          # 6
    ("Grayswandir",              "silver saber"),              # 7
    ("Vorpal Blade",             "long sword"),                # 8
    ("Sceptre of Might",         "mace"),                      # 9
    ("Tsurugi of Muramasa",      "tsurugi"),                   # 10
    ("Magic Mirror of Merlin",   "mirror"),                    # 11
    ("Orb of Detection",         "crystal ball"),              # 12
    ("Heart of Ahriman",         "luckstone"),                 # 13
    ("Staff of Aesculapius",     "quarterstaff"),              # 14
    ("Eyes of the Overworld",    "pair of lenses"),            # 15
    ("Mitre of Holiness",        "helmet"),                    # 16
    ("Longbow of Diana",         "bow"),                       # 17
    ("Master Key of Thievery",   "skeleton key"),              # 18
    ("Yendorian Express Card",   "credit card"),               # 19
    ("Orb of Fate",              "crystal ball"),              # 20
    ("Eye of the Aethiopica",    "amulet of ESP"),             # 21
    # ---------------------------------------------------------------
    # wave17a: P0 #1 — 11 missing artifacts appended (byte-equal vendor
    # artilist.h order).  Indices 22-32 align with artifact_powers.py
    # _ARTIFACT_BONUS_TABLE entries and tests in test_artifact_powers_parity.
    # Cite: vendor/nethack/include/artilist.h lines 149-212 + 123 + 145.
    # ---------------------------------------------------------------
    ("Frost Brand",              "long sword"),                # 22  artilist.h:149
    ("Fire Brand",               "long sword"),                # 23  artilist.h:153
    ("Dragonbane",               "broadsword"),                # 24  artilist.h:157
    ("Demonbane",                "silver mace"),               # 25  artilist.h:162
    ("Werebane",                 "silver saber"),              # 26  artilist.h:166
    ("Trollsbane",               "morning star"),              # 27  artilist.h:182
    ("Grimtooth",                "orcish dagger"),             # 28  artilist.h:123
    ("Magicbane",                "athame"),                    # 29  artilist.h:145
    ("Giantslayer",              "long sword"),                # 30  artilist.h:174
    ("Ogresmasher",              "war hammer"),                # 31  artilist.h:178
    ("Sunsword",                 "long sword"),                # 32  artilist.h:209
)


# ---------------------------------------------------------------------------
# Static lookup tables (built at import time).
# ---------------------------------------------------------------------------
def _build_object_index() -> dict[str, int]:
    """Map object name -> type_id (position in OBJECTS).

    Wave 6 Phase B: object names are stored in their bare canonical form
    ("healing", "identify", "magic missile") and the class prefix is added
    at render time.  For backwards compatibility, prefixed aliases like
    "potion of healing" → bare-name index are merged in via
    ``OBJECT_NAME_ALIASES``.

    Wave 6 parity-fix (CA #63): OBJECTS regenerated from vendor objects.c
    contains 23 anonymous separator entries (``name is None``).  Skip them
    here so the lookup map is well-formed (None has no ``.lower``).
    Cite: vendor/nethack/src/objects.c — sentinel zero rows separate classes.
    """
    index: dict[str, int] = {}
    for idx, entry in enumerate(OBJECTS):
        if entry.name is None:
            continue
        # Prefer the FIRST occurrence of a bare name so cross-class collisions
        # (e.g. "identify" exists in SCROLL_CLASS at 311 and SPBOOK_CLASS at
        # 371) resolve to the earlier class — vendor wishymatch walks the
        # bare-name table in declaration order.
        index.setdefault(entry.name, idx)
    # Merge backwards-compat aliases for verbose "<prefix> <name>" lookups.
    for alias, idx in OBJECT_NAME_ALIASES.items():
        index.setdefault(alias, idx)
    return index


def _build_artifact_index() -> dict[str, int]:
    """Map artifact name -> artifact_idx (position in _ARTIFACTS)."""
    return {name: idx for idx, (name, _base) in enumerate(_ARTIFACTS)}


_OBJECT_BY_NAME: dict[str, int] = _build_object_index()
_ARTIFACT_BY_NAME: dict[str, int] = _build_artifact_index()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def _decode(wish_bytes) -> str:
    """Decode wish input to a normalised lowercase string."""
    if isinstance(wish_bytes, (bytes, bytearray)):
        s = bytes(wish_bytes).decode("ascii", errors="ignore")
    elif isinstance(wish_bytes, str):
        s = wish_bytes
    else:
        try:
            s = bytes(list(wish_bytes)).decode("ascii", errors="ignore")
        except Exception:
            s = str(wish_bytes)
    return s.strip()


def _strip_buc_prefix(text: str) -> tuple[str, int]:
    """Strip leading BUC keyword.

    Returns (remaining_text, buc_status).  buc_status follows the
    Nethax BUCStatus enum: UNKNOWN=0, CURSED=1, UNCURSED=2, BLESSED=3.
    """
    lower = text.lower()
    for prefix, buc in (
        ("blessed ",  int(BUCStatus.BLESSED)),
        ("uncursed ", int(BUCStatus.UNCURSED)),
        ("cursed ",   int(BUCStatus.CURSED)),
    ):
        if lower.startswith(prefix):
            return text[len(prefix):].strip(), buc
    return text, int(BUCStatus.UNCURSED)  # vendor default: wished items uncursed


def _strip_misc_prefixes(text: str) -> str:
    """Strip cosmetic prefixes ("greased", "fixed", "fireproof", ...) that
    NetHack accepts in wishymatch but which we don't model individually.
    """
    keywords = (
        "greased ", "fixed ", "fireproof ", "rustproof ", "corrodeproof ",
        "rotproof ", "thoroughly rusty ", "thoroughly burnt ",
        "rusty ", "corroded ", "burnt ", "rotted ",
    )
    lower = text.lower()
    changed = True
    while changed:
        changed = False
        for kw in keywords:
            if lower.startswith(kw):
                text = text[len(kw):]
                lower = text.lower()
                changed = True
                break
    return text.strip()


def _strip_enchant_prefix(text: str) -> tuple[str, int]:
    """Strip leading enchantment prefix like "+3 " or "-1 ".

    Returns (remaining_text, enchantment).  Defaults to 0 when no prefix.
    """
    text = text.strip()
    if not text:
        return text, 0
    if text[0] not in "+-":
        return text, 0
    # Read sign + digits
    i = 1
    while i < len(text) and text[i].isdigit():
        i += 1
    if i == 1:  # only the sign, no digits
        return text, 0
    try:
        value = int(text[:i])
    except ValueError:
        return text, 0
    return text[i:].strip(), value


# ---------------------------------------------------------------------------
# Full vendor wishymatch parser (Wave 6 Phase B+).
#
# Mirrors vendor/nethack/src/objnam.c::wishymatch + readobjnam:
#   - "the" prefix strip for artifacts
#   - Plural normalization (scrolls→scroll, knives→knife, men→man)
#   - Fuzzy abbreviation match (gdsm→gray dragon scale mail, longsword→long sword)
#   - Multi-modifier combos (blessed rustproof greased fixed +N ... named X)
#   - Artifact SPFX alignment restriction (Excalibur lawful XL>=5, etc.)
# ---------------------------------------------------------------------------

# Modifier keywords parsed left-to-right (lower-cased input).  Each entry is
# (keyword, slot_name, slot_value).  slot_name is one of:
#   "buc"        -> int BUC status
#   "erodeproof" -> bool True
#   "greased"    -> bool True
# Pure cosmetic erosion adjectives (rusty, corroded, ...) are silently dropped
# (vendor wishymatch accepts them but they have no Wave 6 model fidelity).
_MOD_KEYWORDS: tuple[tuple[str, str, object], ...] = (
    ("blessed",        "buc", int(BUCStatus.BLESSED)),
    ("uncursed",       "buc", int(BUCStatus.UNCURSED)),
    ("cursed",         "buc", int(BUCStatus.CURSED)),
    ("holy",           "buc", int(BUCStatus.BLESSED)),
    ("unholy",         "buc", int(BUCStatus.CURSED)),
    ("rustproof",      "erodeproof", True),
    ("fireproof",      "erodeproof", True),
    ("corrodeproof",   "erodeproof", True),
    ("rotproof",       "erodeproof", True),
    ("fixed",          "erodeproof", True),
    ("erodeproof",     "erodeproof", True),
    ("greased",        "greased", True),
)

# Cosmetic erosion adjectives consumed and discarded.
_COSMETIC_DROPS: tuple[str, ...] = (
    "thoroughly rusty", "thoroughly burnt", "thoroughly corroded",
    "thoroughly rotted",
    "very rusty", "very burnt", "very corroded", "very rotted",
    "rusty", "burnt", "corroded", "rotted",
    "diluted",
)

# Plural -> singular irregulars used by vendor's makeplural inverse.
# Source: vendor/nethack/src/objnam.c::makesingular.
_IRREGULAR_PLURALS: dict = {
    "knives":     "knife",
    "wolves":     "wolf",
    "leaves":     "leaf",
    "loaves":     "loaf",
    "men":        "man",
    "women":      "woman",
    "children":   "child",
    "teeth":      "tooth",
    "feet":       "foot",
    "geese":      "goose",
    "mice":       "mouse",
    "lice":       "louse",
    "dice":       "die",
    "oxen":       "ox",
    "fungi":      "fungus",
    "octopi":     "octopus",
    "cacti":      "cactus",
    "matzot":     "matzo",
    "shuriken":   "shuriken",  # already singular
    "ya":         "ya",        # already singular
}


def _strip_the_prefix(text: str) -> str:
    """Strip a leading 'the ' (case-insensitive).  Used for artifact lookup."""
    if text[:4].lower() == "the ":
        return text[4:].lstrip()
    return text


def _singularize_word(word: str) -> str:
    """Inverse of vendor makeplural for a single word.

    Handles irregular forms first, then suffix rules:
      -ies -> -y    (berries -> berry)
      -ves -> -f    (handled by irregulars map for common cases)
      -ches -> -ch  (torches -> torch)
      -shes -> -sh  (wishes -> wish)
      -xes  -> -x   (boxes -> box)
      -ses  -> -s   (glasses -> glass)
      -s    -> ''   (scrolls -> scroll)
    """
    lower = word.lower()
    if lower in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[lower]
    if len(lower) > 3 and lower.endswith("ies"):
        return word[:-3] + "y"
    if len(lower) > 4 and lower.endswith(("ches", "shes")):
        return word[:-2]
    if len(lower) > 3 and lower.endswith(("xes", "ses", "zes")):
        return word[:-2]
    if len(lower) > 1 and lower.endswith("s") and not lower.endswith("ss"):
        # Avoid stripping the trailing 's' of intrinsically singular words
        # like "scales", "lenses" (handled above), "pants".
        return word[:-1]
    return word


def _singularize_phrase(text: str) -> str:
    """Singularize the first noun token of a multi-word phrase.

    Vendor pluralization in inv_strs polish is single-noun based:
      "scrolls of identify" -> "scroll of identify"
      "potions of healing"  -> "potion of healing"
      "pairs of lenses"     -> "pair of lenses"
    We singularize only the leading word (before " of ") plus a trailing-word
    fallback for "long swords" -> "long sword".
    """
    if " of " in text:
        head, tail = text.split(" of ", 1)
        head_words = head.split()
        if head_words:
            head_words[-1] = _singularize_word(head_words[-1])
            head = " ".join(head_words)
        return f"{head} of {tail}"
    # No " of " marker — singularize the last word (handles "long swords").
    words = text.split()
    if not words:
        return text
    singular_last = _singularize_word(words[-1])
    if singular_last != words[-1]:
        return " ".join(words[:-1] + [singular_last])
    return text


def _fuzzy_object_lookup(text: str) -> int:
    """Vendor wishymatch-style fuzzy lookup.

    Strategy (mirrors objnam.c::wishymatch lines 3243+):
      1. Exact match in _OBJECT_BY_NAME (case-insensitive).
      2. Strip-spaces match: "longsword" matches the no-space form of
         "long sword"; "graydragonscalemail" -> "gray dragon scale mail".
      3. Abbreviation match: leading-initials per word, e.g. "gdsm" ->
         "gray dragon scale mail".
    Returns the OBJECTS index, or -1 on miss.
    """
    if not text:
        return -1
    lower = text.lower()

    # Pass 1 — exact (case-insensitive).
    for name, idx in _OBJECT_BY_NAME.items():
        if name.lower() == lower:
            return idx

    # Pass 2 — strip-spaces.  Compare wish-no-spaces to object-no-spaces.
    needle = lower.replace(" ", "").replace("-", "")
    if needle:
        for name, idx in _OBJECT_BY_NAME.items():
            haystack = name.lower().replace(" ", "").replace("-", "")
            if haystack == needle:
                return idx

    # Pass 3 — abbreviation (first letter of each word).  Only consider
    # multi-word object names so single-letter "p" doesn't match "potion".
    if needle and needle.isalpha() and len(needle) >= 2:
        for name, idx in _OBJECT_BY_NAME.items():
            words = name.lower().split()
            if len(words) < 2:
                continue
            initials = "".join(w[0] for w in words if w)
            if initials == needle:
                return idx

    return -1


def _fuzzy_artifact_lookup(text: str) -> int:
    """Vendor wishymatch artifact lookup with case-insensitive fallback.

    Vendor matches the artifact's proper noun (case-sensitive in artilist.h),
    but readobjnam tolerates case differences.  Returns artifact index or -1.
    """
    if not text:
        return -1
    # Direct hit on proper noun.
    idx = _ARTIFACT_BY_NAME.get(text, -1)
    if idx >= 0:
        return idx
    lower = text.lower()
    for art_name, art_idx in _ARTIFACT_BY_NAME.items():
        if art_name.lower() == lower:
            return art_idx
    return -1


def _strip_named_suffix(text: str) -> tuple[str, bytes]:
    """Strip a trailing ' named <X>' or ' called <X>' clause.

    Returns (remaining_text, name_bytes).  name_bytes is b'' when no suffix.
    The vendor accepts both "named" and "called" (do_name.c::do_oname).
    """
    lower = text.lower()
    for sep in (" named ", " called "):
        pos = lower.rfind(sep)
        if pos >= 0:
            head = text[:pos].rstrip()
            tail = text[pos + len(sep):].strip()
            # Truncate to USER_NAME_LEN (vendor caps at ONAME_LEN).
            tail_bytes = tail.encode("ascii", errors="ignore")[:USER_NAME_LEN]
            return head, tail_bytes
    return text, b""


def _consume_modifiers(text: str) -> dict:
    """Walk modifier keywords left-to-right.

    Returns a dict with parsed modifier fields and the remaining text:
      {'text': str, 'buc': int|None, 'enchant': int, 'erodeproof': bool,
       'greased': bool}
    """
    result = {
        "buc": None,
        "enchant": 0,
        "erodeproof": False,
        "greased": False,
        "text": text,
    }
    progress = True
    while progress and result["text"]:
        progress = False
        lower = result["text"].lower()

        # Try keyword modifiers.
        for kw, slot, val in _MOD_KEYWORDS:
            prefix = kw + " "
            if lower.startswith(prefix):
                if slot == "buc" and result["buc"] is None:
                    result["buc"] = val
                elif slot == "erodeproof":
                    result["erodeproof"] = True
                elif slot == "greased":
                    result["greased"] = True
                result["text"] = result["text"][len(prefix):].lstrip()
                progress = True
                break
        if progress:
            continue

        # Cosmetic erosion adjectives — consume and ignore.
        for kw in _COSMETIC_DROPS:
            prefix = kw + " "
            if lower.startswith(prefix):
                result["text"] = result["text"][len(prefix):].lstrip()
                progress = True
                break
        if progress:
            continue

        # Enchantment prefix: +N or -N.
        new_text, ench = _strip_enchant_prefix(result["text"])
        if new_text != result["text"]:
            result["enchant"] = ench
            result["text"] = new_text
            progress = True
            continue

    return result


def _artifact_alignment(artifact_idx: int) -> int:
    """Return the canonical alignment requirement for an artifact.

    Hardcoded table (mirrors artilist.h A_LAWFUL / A_NEUTRAL / A_CHAOTIC /
    A_NONE).  Used for the SPFX_RESTR check.  Values use the Nethax
    Alignment enum (CHAOTIC=0, NEUTRAL=1, LAWFUL=2, UNALIGNED=3).
    """
    # Per vendor/nethack/include/artilist.h.
    table = {
        0:  int(Alignment.LAWFUL),     # Excalibur
        1:  int(Alignment.LAWFUL),     # Snickersnee
        2:  int(Alignment.CHAOTIC),    # Stormbringer
        3:  int(Alignment.NEUTRAL),    # Mjollnir
        4:  int(Alignment.NEUTRAL),    # Cleaver
        5:  int(Alignment.CHAOTIC),    # Sting
        6:  int(Alignment.CHAOTIC),    # Orcrist
        7:  int(Alignment.LAWFUL),     # Grayswandir
        8:  int(Alignment.NEUTRAL),    # Vorpal Blade
        9:  int(Alignment.LAWFUL),     # Sceptre of Might
        10: int(Alignment.LAWFUL),     # Tsurugi of Muramasa
        11: int(Alignment.NEUTRAL),    # Magic Mirror of Merlin
        12: int(Alignment.NEUTRAL),    # Orb of Detection
        13: int(Alignment.NEUTRAL),    # Heart of Ahriman
        14: int(Alignment.NEUTRAL),    # Staff of Aesculapius
        15: int(Alignment.NEUTRAL),    # Eyes of the Overworld
        16: int(Alignment.LAWFUL),     # Mitre of Holiness
        17: int(Alignment.NEUTRAL),    # Longbow of Diana
        18: int(Alignment.CHAOTIC),    # Master Key of Thievery
        19: int(Alignment.NEUTRAL),    # Yendorian Express Card
        20: int(Alignment.NEUTRAL),    # Orb of Fate
        21: int(Alignment.NEUTRAL),    # Eye of the Aethiopica
        # wave17a: P0 #1 — 11 newly added artifacts (cite artilist.h al col).
        22: int(Alignment.UNALIGNED),  # Frost Brand   A_NONE   (artilist.h:150)
        23: int(Alignment.UNALIGNED),  # Fire Brand    A_NONE   (artilist.h:154)
        24: int(Alignment.UNALIGNED),  # Dragonbane    A_NONE   (artilist.h:159)
        25: int(Alignment.LAWFUL),     # Demonbane     A_LAWFUL (artilist.h:163)
        26: int(Alignment.UNALIGNED),  # Werebane      A_NONE   (artilist.h:167)
        27: int(Alignment.UNALIGNED),  # Trollsbane    A_NONE   (artilist.h:183)
        28: int(Alignment.CHAOTIC),    # Grimtooth     A_CHAOTIC (artilist.h:125)
        29: int(Alignment.NEUTRAL),    # Magicbane     A_NEUTRAL (artilist.h:146)
        30: int(Alignment.NEUTRAL),    # Giantslayer   A_NEUTRAL (artilist.h:175)
        31: int(Alignment.UNALIGNED),  # Ogresmasher   A_NONE   (artilist.h:179)
        32: int(Alignment.LAWFUL),     # Sunsword      A_LAWFUL (artilist.h:210)
    }
    return table.get(artifact_idx, int(Alignment.UNALIGNED))


_EXCALIBUR_MIN_XL = 5  # vendor: u.ulevel >= 5 required for Excalibur.


def _spec_applies(artifact_idx: int, player_align: int, player_xl: int) -> bool:
    """Check SPFX_RESTR alignment/XL restrictions for granting an artifact.

    Mirrors artifact.c::spec_applies (the wish-side check).  Returns True
    when the player satisfies the artifact's requirements.
    """
    if artifact_idx < 0:
        return False
    required_align = _artifact_alignment(artifact_idx)
    if required_align != int(Alignment.UNALIGNED):
        if player_align != required_align:
            return False
    # Excalibur additionally requires XL >= 5.
    if artifact_idx == 0 and player_xl < _EXCALIBUR_MIN_XL:
        return False
    return True


def apply_artifact_restrictions(parsed: dict, player_align: int,
                                player_xl: int) -> dict:
    """Apply SPFX_RESTR alignment/XL gating to a parsed wishymatch dict.

    Returns a new dict (the input is not mutated).  Mirrors vendor
    readobjnam's spec_applies branch and the Excalibur->Stormbringer
    chaotic substitution shorthand.

    Behavior:
      - Non-artifact input: unchanged.
      - Artifact passes spec_applies: unchanged.
      - Chaotic player wishing Excalibur and Stormbringer's gates open:
        the dict is rewritten to grant Stormbringer (runesword base).
      - Any other denial: artifact_idx becomes -1 but the base object
        remains (vendor: still grants the underlying long sword / bow / ...).
    """
    if not parsed.get("parsed", False):
        return dict(parsed)
    art = parsed.get("artifact_idx", -1)
    if art < 0:
        return dict(parsed)
    if _spec_applies(art, player_align, player_xl):
        return dict(parsed)
    out = dict(parsed)
    if (art == 0
        and player_align == int(Alignment.CHAOTIC)
        and _spec_applies(2, player_align, player_xl)):
        sub_idx = 2  # Stormbringer
        base = _ARTIFACTS[sub_idx][1]
        sub_type = _OBJECT_BY_NAME.get(base, -1)
        if sub_type < 0:
            sub_type = _fuzzy_object_lookup(base)
        if sub_type >= 0:
            out["artifact_idx"] = sub_idx
            out["type_id"]      = sub_type
            out["category"]     = int(OBJECTS[sub_type].class_)
            return out
    # Denial: strip artifact_idx, keep the base object.
    out["artifact_idx"] = -1
    return out


# wave17h P0 #1: quantity prefix parser (objnam.c:3982-3987).
def _strip_quantity_prefix(text: str) -> tuple[str, int]:
    """Strip leading integer quantity prefix.

    Cite: vendor/nethack/src/objnam.c::readobjnam lines 3982-3987:
        else if (!d->cnt && digit(*d->bp) && strcmp(d->bp, "0")) {
            d->cnt = atoi(d->bp);
            while (digit(*d->bp)) d->bp++;
            while (*d->bp == ' ') d->bp++;
        }
    Returns (remaining_text, cnt). cnt=0 when no prefix present.
    """
    text = text.lstrip()
    if not text or not text[0].isdigit():
        return text, 0
    i = 0
    while i < len(text) and text[i].isdigit():
        i += 1
    digits = text[:i]
    # vendor: strcmp(d->bp, "0") guard — a bare "0" is rejected.
    if digits == "0":
        return text[i:].lstrip(), 0
    try:
        n = int(digits)
    except ValueError:
        return text, 0
    return text[i:].lstrip(), n


# wave17h P0 #2: gold-wish constants (objnam.c:4533-4546).
_GOLD_PIECE_TYPE_ID: int = 410   # objects.py: gold piece index 410


def _try_gold_wish(text: str) -> bool:
    """Detect a gold-piece wish keyword.

    Cite: vendor/nethack/src/objnam.c::readobjnam lines 4533-4536:
        if (!BSTRCMPI(d->bp, d->p - 10, "gold piece")
            || !BSTRCMPI(d->bp, d->p - 7, "zorkmid")
            || !strcmpi(d->bp, "gold") || !strcmpi(d->bp, "money")
            || !strcmpi(d->bp, "coin") || *d->bp == GOLD_SYM)
    """
    if not text:
        return False
    lower = text.lower()
    if lower in ("gold", "money", "coin", "coins", "zorkmid", "zorkmids",
                 "gold piece", "gold pieces"):
        return True
    if text.startswith("$") or text == "$":
        return True
    if lower.endswith("gold piece") or lower.endswith("gold pieces"):
        return True
    if lower.endswith("zorkmid") or lower.endswith("zorkmids"):
        return True
    return False


# wave17h P0 #3: nowish/wizard-only substitutions (objnam.c:5001-5025).
# Each entry maps a "wizard-only" object name to its safe substitute.
_NOWISH_SUBSTITUTIONS: dict = {
    "Amulet of Yendor":           "cheap plastic imitation of the Amulet of Yendor",
    "Candelabrum of Invocation":  "tallow candle",   # rnd_class(TALLOW_CANDLE,WAX_CANDLE)
    "Bell of Opening":            "bell",
    "Book of the Dead":           "blank paper",     # SPE_BOOK_OF_THE_DEAD -> SPE_BLANK_PAPER
    "magic lamp":                 "oil lamp",
}


def _maybe_substitute_nowish(type_id: int) -> int:
    """Apply vendor nowish substitution table when not in wizard mode.

    Cite: vendor/nethack/src/objnam.c::readobjnam lines 5001-5025.
    Since this env has no wizard-mode flag, substitution is always active
    (matches vendor default non-wizard behavior).
    """
    if type_id < 0 or type_id >= len(OBJECTS):
        return type_id
    entry = OBJECTS[type_id]
    sub_name = _NOWISH_SUBSTITUTIONS.get(entry.name)
    if sub_name is None:
        return type_id
    sub_id = _OBJECT_BY_NAME.get(sub_name, -1)
    if sub_id < 0:
        sub_id = _fuzzy_object_lookup(sub_name)
    return sub_id if sub_id >= 0 else type_id


def wishymatch(wish_bytes) -> dict:
    """Full vendor wishymatch parser (Wave 6 Phase B+).

    Returns a dict shaped:
        {
          'category': int,        # ObjectClass; -1 on parse miss
          'type_id': int,         # OBJECTS index; -1 on parse miss
          'buc': int,             # BUCStatus int (UNCURSED default)
          'enchant': int,         # signed enchantment
          'artifact_idx': int,    # -1 when not an artifact wish
          'user_name': bytes,     # b'' when no "named X" clause
          'erodeproof': bool,
          'greased': bool,
          'quantity': int,        # wave17h P0: parsed cnt (gated by oc_merge)
          'is_gold': bool,        # wave17h P0: gold-piece wish flag
          'parsed': bool,         # False when name lookup fails
        }

    Cite: vendor/nethack/src/objnam.c::wishymatch + readobjnam.
    """
    out = {
        "category": -1,
        "type_id": -1,
        "buc": int(BUCStatus.UNCURSED),
        "enchant": 0,
        "artifact_idx": -1,
        "user_name": b"",
        "erodeproof": False,
        "greased": False,
        "quantity": 1,
        "is_gold": False,
        "parsed": False,
    }
    text = _decode(wish_bytes)
    if not text:
        return out

    # 1. Strip trailing "named X" / "called X".
    text, user_name = _strip_named_suffix(text)
    out["user_name"] = user_name

    # wave17h P0 #1: parse leading integer quantity (objnam.c:3982-3987).
    text, cnt = _strip_quantity_prefix(text)

    # 2. Walk left-to-right modifier keywords + enchantment.
    mods = _consume_modifiers(text)
    text = mods["text"]
    out["enchant"]    = mods["enchant"]
    out["erodeproof"] = mods["erodeproof"]
    out["greased"]    = mods["greased"]
    if mods["buc"] is not None:
        out["buc"] = mods["buc"]

    text = text.strip()
    if not text:
        return out

    # wave17h P0 #2: gold-piece wish detection (objnam.c:4533-4546).
    if _try_gold_wish(text):
        gcnt = cnt
        # vendor: "$5000" form — digits after the $ sign provide cnt.
        if gcnt == 0 and text.startswith("$") and len(text) > 1 and text[1].isdigit():
            j = 1
            while j < len(text) and text[j].isdigit():
                j += 1
            try:
                gcnt = int(text[1:j])
            except ValueError:
                gcnt = 0
        if gcnt > 5000:
            gcnt = 5000
        if gcnt < 1:
            gcnt = 1
        out["category"]     = int(OBJECTS[_GOLD_PIECE_TYPE_ID].class_)
        out["type_id"]      = _GOLD_PIECE_TYPE_ID
        out["quantity"]     = gcnt
        out["is_gold"]      = True
        out["parsed"]       = True
        return out

    # 3. Plural normalization first (so "the scrolls of identify" works
    #    after the "the" strip below).
    text = _singularize_phrase(text)

    # 4. "the" prefix — only meaningful for artifact lookup, but harmless
    #    to also drop before the object lookup fallback.
    text_no_the = _strip_the_prefix(text)

    # 5. Artifact lookup (case-sensitive on proper noun, then fuzzy).
    artifact_idx = _fuzzy_artifact_lookup(text_no_the)
    if artifact_idx < 0:
        # Some artifact names embed "the" internally (Eye of the Aethiopica)
        # so also try the un-stripped form.
        artifact_idx = _fuzzy_artifact_lookup(text)

    if artifact_idx >= 0:
        base_name = _ARTIFACTS[artifact_idx][1]
        type_id = _OBJECT_BY_NAME.get(base_name, -1)
        if type_id < 0:
            type_id = _fuzzy_object_lookup(base_name)
        if type_id >= 0:
            out["category"]     = int(OBJECTS[type_id].class_)
            out["type_id"]      = type_id
            out["artifact_idx"] = artifact_idx
            out["parsed"]       = True
        return out

    # 6. Object name fuzzy lookup.
    type_id = _fuzzy_object_lookup(text_no_the)
    if type_id < 0:
        type_id = _fuzzy_object_lookup(text)
    if type_id < 0:
        return out

    # wave17h P0 #3: nowish substitutions (objnam.c:5001-5025).
    type_id = _maybe_substitute_nowish(type_id)

    out["category"] = int(OBJECTS[type_id].class_)
    out["type_id"]  = type_id
    # wave17h P0 #1: apply oc_merge gate. Vendor oc_merge is true for stackable
    # classes (COIN, SCROLL, POTION, FOOD, GEM, ROCK, WEAPON arrows/darts, ...).
    # We model the gate as: only allow cnt > 1 for the inherently stackable
    # classes; otherwise clamp to 1. Cite: vendor/nethack/include/objclass.h
    # oc_merge bit + objnam.c:5040-5072 quantity-honoring branch.
    cls = int(OBJECTS[type_id].class_)
    _MERGEABLE_CLASSES = {
        9,    # SCROLL_CLASS
        8,    # POTION_CLASS
        7,    # FOOD_CLASS
        12,   # COIN_CLASS
        13,   # GEM_CLASS
        14,   # ROCK_CLASS
        17,   # VENOM_CLASS
    }
    if cnt > 1 and cls in _MERGEABLE_CLASSES:
        # vendor: !wizard caps at 5000 (objnam.c:4537), but for non-gold
        # objects we mirror the spirit of the cap.
        cnt_q = min(max(cnt, 1), 5000)
        out["quantity"] = cnt_q
    else:
        out["quantity"] = 1
    out["parsed"]   = True
    return out


def parse_wish_string(wish_bytes) -> tuple[int, int, int, int, int]:
    """Parse a wish input into structured fields (5-tuple compatibility view).

    Thin compatibility wrapper over ``wishymatch`` that projects the full
    vendor parser dict down to the 5-tuple shape used by early callers.
    The underlying parser implements the full vendor readobjnam grammar
    (BUC, holy/unholy, erodeproof, greased, +N enchantment, quantity prefix,
    "named X" suffix, gold short-forms, "the " article, plural normalization,
    fuzzy abbreviation, nowish substitution, artifact lookup).

    Returns
    -------
    (category, type_id, buc_status, enchantment, artifact_idx)

    category     : ObjectClass int (-1 on miss)
    type_id      : index into OBJECTS (-1 on miss)
    buc_status   : BUCStatus int (UNCURSED default)
    enchantment  : int8-range
    artifact_idx : 0-based index into _ARTIFACTS, or -1 if not an artifact

    JAX-required: Python-side (not JIT).  Free-form string parsing is not
    expressible in jax.lax primitives, and wish parsing only runs at action-
    handler dispatch time — never inside the hot per-step loop.

    Cite: vendor/nethack/src/objnam.c::readobjnam + wishymatch.
    """
    parsed = wishymatch(wish_bytes)
    if not parsed["parsed"]:
        return (-1, -1, parsed["buc"], parsed["enchant"], -1)
    return (
        parsed["category"],
        parsed["type_id"],
        parsed["buc"],
        parsed["enchant"],
        parsed["artifact_idx"],
    )


def grant_wish_from_string(state, wish_input, rng=None):
    """Grant a wish from a plain string (no rng required for parse path).

    Thin wrapper over ``grant_wish`` that accepts str or bytes and uses a
    deterministic RNG when none is provided.  Added as a public alias for the
    wave-13 wish-parser agent.

    Cite: vendor/nethack/src/wizard.c::makewish.
    """
    import jax
    if rng is None:
        rng = jax.random.PRNGKey(0)
    if isinstance(wish_input, str):
        wish_input = wish_input.encode()
    return grant_wish(state, rng, wish_input)


def parse_wish_string_dict(wish_input) -> dict:
    """Parse a wish string and return the full vendor wishymatch dict.

    Thin wrapper over ``wishymatch`` that accepts either str or bytes and
    always returns the raw parsed dict (same shape as wishymatch's return
    value).  Introduced as a public alias for the wave-13 wish-parser agent.

    Cite: vendor/nethack/src/objnam.c::wishymatch.
    """
    if isinstance(wish_input, str):
        wish_input = wish_input.encode()
    return wishymatch(wish_input)


# ---------------------------------------------------------------------------
# Grant
# ---------------------------------------------------------------------------
def _find_first_empty_slot(items) -> int:
    """Python-side: return first empty inventory slot index, or -1 if full."""
    cats = items.category
    for i in range(MAX_INVENTORY_SLOTS):
        if int(cats[i]) == 0:
            return i
    return -1


def _find_first_empty_ground_slot(ground_items, b, lv, r, c) -> int:
    """Return first empty ground-stack slot index at (b,lv,r,c), or -1 if full."""
    cats = ground_items.category
    for i in range(MAX_GROUND_STACK):
        if int(cats[b, lv, r, c, i]) == 0:
            return i
    return -1


def _write_inventory_slot(state, slot_idx: int, category: int, type_id: int,
                          buc: int, enchant: int, weight: int, quantity: int = 1,
                          artifact_idx: int = -1):
    """Return new state with the given inventory slot populated with the wished item.

    ``artifact_idx`` populates ``Item.artifact_idx`` (vendor obj->oartifact)
    so ``apply_carried_artifact_extrinsics`` (artifact_powers.py) walks the
    carried artifact and ORs its cspfx bits into the player's extrinsics.
    Defaults to -1 (non-artifact).  cite: vendor/nethack/include/obj.h
    obj->oartifact; artifact.c::set_artifact_intrinsic.
    """
    items = state.inventory.items
    new_items = items.replace(
        category    = items.category.at[slot_idx].set(jnp.int8(category)),
        type_id     = items.type_id.at[slot_idx].set(jnp.int16(type_id)),
        buc_status  = items.buc_status.at[slot_idx].set(jnp.int8(buc)),
        enchantment = items.enchantment.at[slot_idx].set(jnp.int8(enchant)),
        charges     = items.charges.at[slot_idx].set(jnp.int8(0)),
        identified  = items.identified.at[slot_idx].set(jnp.bool_(True)),
        quantity    = items.quantity.at[slot_idx].set(jnp.int16(quantity)),
        weight      = items.weight.at[slot_idx].set(jnp.int32(weight * quantity)),
        ac_bonus    = items.ac_bonus.at[slot_idx].set(jnp.int8(0)),
        is_two_handed = items.is_two_handed.at[slot_idx].set(jnp.bool_(False)),
        artifact_idx = items.artifact_idx.at[slot_idx].set(jnp.int8(artifact_idx)),
    )
    new_inv = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


def _write_ground_slot(state, b: int, lv: int, r: int, c: int, gslot: int,
                       category: int, type_id: int, buc: int, enchant: int,
                       weight: int, quantity: int = 1, artifact_idx: int = -1):
    """Return new state with the wished item placed on the ground stack.

    See ``_write_inventory_slot`` for ``artifact_idx`` semantics.  The
    ground-stack copy of the artifact_idx field becomes carried over the
    next ``pickup`` thanks to the field-by-field copy in
    ``inventory.pickup`` (added in commit 45827a5 / Wave 42d stack-merge).
    """
    g = state.ground_items
    new_g = g.replace(
        category    = g.category.at[b, lv, r, c, gslot].set(jnp.int8(category)),
        type_id     = g.type_id.at[b, lv, r, c, gslot].set(jnp.int16(type_id)),
        buc_status  = g.buc_status.at[b, lv, r, c, gslot].set(jnp.int8(buc)),
        enchantment = g.enchantment.at[b, lv, r, c, gslot].set(jnp.int8(enchant)),
        charges     = g.charges.at[b, lv, r, c, gslot].set(jnp.int8(0)),
        identified  = g.identified.at[b, lv, r, c, gslot].set(jnp.bool_(True)),
        quantity    = g.quantity.at[b, lv, r, c, gslot].set(jnp.int16(quantity)),
        weight      = g.weight.at[b, lv, r, c, gslot].set(jnp.int32(weight * quantity)),
        ac_bonus    = g.ac_bonus.at[b, lv, r, c, gslot].set(jnp.int8(0)),
        is_two_handed = g.is_two_handed.at[b, lv, r, c, gslot].set(jnp.bool_(False)),
        artifact_idx = g.artifact_idx.at[b, lv, r, c, gslot].set(jnp.int8(artifact_idx)),
    )
    return state.replace(ground_items=new_g)


def _mark_wish_conducts(state, artifact: bool):
    """Bump WISHLESS (and ARTIWISHLESS if artifact) counters on EnvState.conduct.

    Vendor: ``u.uconduct.wishes++`` on every wish, plus ``u.uconduct.wisharti++``
    when the wish granted an artifact (insight.c lines ~2183-2202 consume both
    counter values for display: ``"used %ld wish%s"`` / ``"%ld for artifacts"``;
    topten.c:385-386 emits ``wish_cnt`` / ``arti_wish_cnt`` to xlog).
    """
    from Nethax.nethax.subsystems.conduct import increment_counter
    state = increment_counter(state, int(Conduct.WISHLESS))
    if artifact:
        state = increment_counter(state, int(Conduct.ARTIWISHLESS))
    return state


def _set_user_name_at(state, slot_idx: int, name_bytes: bytes):
    """Write ``name_bytes`` (zero-padded) into ``state.inventory.user_names[slot]``.

    Mirrors handle_name() but Python-side without the JIT path.  Used when
    the wish string contains a trailing ``named X`` clause.
    """
    padded = bytes(name_bytes)[:USER_NAME_LEN]
    padded = padded + b"\x00" * (USER_NAME_LEN - len(padded))
    name_row = jnp.array(list(padded), dtype=jnp.int8)
    new_user_names = state.inventory.user_names.at[slot_idx].set(name_row)
    new_inv = state.inventory.replace(user_names=new_user_names)
    return state.replace(inventory=new_inv)


def grant_wish(state, rng, wish_string):
    """Grant a wish: parse the wish string, create the item, set conducts.

    Steps (mirrors vendor/nethack/src/wizard.c::makewish):
      1. Parse wish_string Python-side via ``wishymatch``.
      2. On parse miss, return state unchanged (no conduct flip).
      3. Enforce artifact SPFX_RESTR alignment/XL check: if the player can't
         have the wished artifact, fall back to the base item with no
         artifact mark (vendor: alignment-mismatched wishes still grant the
         base object).  Special case: chaotic wishing Excalibur receives
         Stormbringer instead, mirroring vendor's flavor-aware behavior.
      4. Place the item in the first empty inventory slot.  If the inventory
         is full, drop it on the ground at the player position.
      5. Apply user-given name (``named X`` clause) at the chosen slot.
      6. Mark WISHLESS conduct (insight.c ~2163 u.uconduct.wishes).
      7. Mark ARTIWISHLESS conduct when an artifact was actually granted
         (insight.c ~2166 u.uconduct.wisharti).
    """
    parsed = wishymatch(wish_string)
    if not parsed["parsed"]:
        return state  # unknown wish: state unchanged, no conduct flip

    category     = parsed["category"]
    type_id      = parsed["type_id"]
    buc          = parsed["buc"]
    enchant      = parsed["enchant"]
    artifact_idx = parsed["artifact_idx"]
    user_name    = parsed["user_name"]
    quantity     = int(parsed.get("quantity", 1))

    # Artifact SPFX restriction.  Vendor objnam.c::readobjnam (line 5362)
    # flips u.uconduct.wisharti when the wish *text* was an artifact name,
    # even if SPFX_RESTR ultimately denies the grant -- so we preserve the
    # original artifact_idx for the conduct flip and grant the base object.
    # The alignment-aware Excalibur->Stormbringer substitution is exposed
    # via ``apply_artifact_restrictions`` for callers that want the full
    # vendor reroute behavior.

    # Resolve weight from OBJECTS table for fidelity to vendor item weights.
    weight = int(OBJECTS[type_id].weight)

    # Place item: prefer inventory; fall back to ground stack.
    slot = _find_first_empty_slot(state.inventory.items)
    if slot >= 0:
        # Thread artifact_idx through so apply_carried_artifact_extrinsics
        # picks up the wished artifact (Wave 41a / Audit K wire-up).
        state = _write_inventory_slot(state, slot, category, type_id, buc,
                                      enchant, weight, quantity=quantity,
                                      artifact_idx=int(artifact_idx))
        if user_name:
            state = _set_user_name_at(state, slot, user_name)
    else:
        b  = int(state.dungeon.current_branch)
        lv = int(state.dungeon.current_level) - 1
        r  = int(state.player_pos[0])
        c  = int(state.player_pos[1])
        gslot = _find_first_empty_ground_slot(state.ground_items, b, lv, r, c)
        if gslot >= 0:
            state = _write_ground_slot(state, b, lv, r, c, gslot,
                                       category, type_id, buc, enchant, weight,
                                       quantity=quantity,
                                       artifact_idx=int(artifact_idx))
        # else: nowhere to put it; vendor would print "nothing happens".

    return _mark_wish_conducts(state, artifact=(artifact_idx >= 0))


# ---------------------------------------------------------------------------
# Wand of wishing — EnvState-level handler.
#
# Vendor flow (zap.c::dozap → WAN_WISHING at line 2575 → makewish() at line
# 6314 → readobjnam()): the engine prompts the player via getlin(), then
# pipes the response through readobjnam to materialize the wished object.
#
# Headless adaptation: the JAX env has no interactive terminal, so the wish
# text is supplied as the ``wish_string`` parameter and routed through the
# same readobjnam port (``wishymatch`` → ``grant_wish``) the action-dispatch
# layer uses for player-issued wishes.  A canonical default text is provided
# for callers that do not pre-supply one (matching the historical "blessed
# greased +3 gray dragon scale mail" reference wish used in vendor tutorials
# and the NetHack wiki strategy pages).
#
# The items_wands subsystem operates on its own WandState slice which lacks
# the ConductState/QuestState/etc. needed to mark WISHLESS / ARTIWISHLESS.
# This helper is the canonical EnvState entry point: callers wrap the wand
# zap at the action-dispatch layer (or call this directly in tests).
# Cite: vendor/nethack/src/zap.c::dozap WAN_WISHING branch + makewish().
# ---------------------------------------------------------------------------
_DEFAULT_WAND_WISH = b"blessed greased +3 gray dragon scale mail"


def handle_wand_of_wishing(state, rng, wish_string=None):
    """Grant a wand-of-wishing wish on EnvState.

    Vendor parity: vendor zap.c::dozap routes WAN_WISHING to makewish(), which
    reads a free-form wish string from the player via getlin() and passes it to
    readobjnam() for parsing.  In our headless env the wish text is provided as
    the ``wish_string`` parameter (mirroring the prompt buffer vendor would
    populate) and routed through the same readobjnam port (``wishymatch``) via
    ``grant_wish``.  When no text is supplied, the canonical default is used.

    Cite: vendor/nethack/src/zap.c::dozap WAN_WISHING (line 2575) +
          vendor/nethack/src/zap.c::makewish (line 6314) +
          vendor/nethack/src/objnam.c::readobjnam.
    """
    if wish_string is None:
        wish_string = _DEFAULT_WAND_WISH
    return grant_wish(state, rng, wish_string)
