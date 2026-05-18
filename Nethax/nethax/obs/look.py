"""Look/`;` command output — port of vendor pager.c::lookat + invent.c::look_here.

Sources:
  vendor/nethack/src/pager.c::lookat       (lines 656-810) — name a glyph at (x,y)
  vendor/nethack/src/invent.c::look_here   (lines 4101-4326) — describe ground items
  vendor/nethack/src/invent.c::dolook      (lines 4329+) — entry point for ';'

The vendor flow:
  - If looking at the player's cell ('u_at(x,y)'): show "yourself" + (if invis)
    bracketed sense-list.
  - If a monster is at (x,y): show the monster's full name (vendor look_at_monster).
  - If only an object on floor: "You see here <obj>." (verb is "feel" when Blind).
  - If multiple objects: "Things that are here:" + list.
  - If nothing: "You see no objects here." (or "feel" when Blind).
  - For a remote cell with no obj/monster: terrain noun (doorway, altar, ...).

Host-side helper (not jit'd) — used for UI prompts + parity tests.

NOTE on vendor parity: nethax does not yet model `Blind` consistently for the
look path (the player can always "see" what they look at), so we emit "see"
unconditionally and document the divergence here.  We also do NOT model
underwater / swallowed / hallucination — those would require additional state
queries that are not yet plumbed through.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.constants.objects import OBJECTS


# ---------------------------------------------------------------------------
# Vendor "an / a" article helper.
# Full port of vendor/nethack/src/objnam.c::just_an() lines 2108-2142.
# ---------------------------------------------------------------------------

_VOWELS = set("aeiouAEIOU")

# Single-letter words that take "an" — vendor objnam.c just_an() line 2117.
_AN_SINGLE_LETTERS = set("aefhilmnosx")

# Long-'u' prefixes that take "a" not "an" — objnam.c just_an() lines 2132-2135.
_LONG_U_PREFIXES = ("eu", "uke", "ukulele", "unicorn", "uranium", "useful")


def _an(name: str) -> str:
    """Port of vendor/nethack/src/objnam.c::just_an() lines 2108-2142.

    Returns name prepended with 'a ', 'an ', or no article (bare name).
    Pass-through cases (already has article / starts with digit):
      - already starts with 'the ', 'some ', 'a ', 'an ': return as-is.
      - starts with a digit: return as-is (vendor an() caller handles).
    No-article cases (vendor just_an lines 2121-2125):
      - "molten lava", "iron bars", "ice".
    Single-letter rule (vendor just_an line 2117):
      - only letters in "aefhilmnosx" take "an".
    Long-u / 'one' exceptions (vendor just_an lines 2129-2135):
      - starts with "one" (followed by separator or end): "a".
      - starts with eu/uke/ukulele/unicorn/uranium/useful: "a".
    'x' followed by consonant (vendor just_an line 2136):
      - e.g. "xorn" → "a", "xerox" → "an".
    """
    if not name:
        return name

    # Pass-through: already has a leading article.
    if name.startswith(("the ", "The ", "some ", "Some ", "a ", "an ", "An ", "A ")):
        return name

    # Pass-through: starts with digit (caller formats quantity itself).
    if name[0].isdigit():
        return name

    c0 = name[0].lower()

    # Single-letter word (vendor just_an line 2115-2117).
    if len(name) == 1 or name[1] == ' ':
        if c0 in _AN_SINGLE_LETTERS:
            return "an " + name
        return "a " + name

    # No-article items (vendor just_an lines 2121-2125).
    lower = name.lower()
    if lower in ("molten lava", "iron bars", "ice"):
        return name

    # Starts with "the " (case-insensitive) — no extra article.
    if lower.startswith("the "):
        return name

    # Decide "an" vs "a" (vendor just_an lines 2128-2139).
    if c0 in "aeiou":
        # 'wun' initial sound: "one" followed by separator or end (line 2130).
        if lower.startswith("one") and (len(name) == 3 or name[3] in "-_ "):
            return "a " + name
        # Long 'u' initial sound (lines 2132-2135).
        lower_name = name.lower()
        for prefix in _LONG_U_PREFIXES:
            if lower_name.startswith(prefix):
                return "a " + name
        return "an " + name

    # 'x' followed by consonant → "an" (vendor just_an line 2136).
    if c0 == 'x' and len(name) > 1 and name[1].lower() not in "aeiou":
        return "an " + name

    return "a " + name


# ---------------------------------------------------------------------------
# Vendor "the" article helper.
# Simplified port of vendor/nethack/src/objnam.c::the() lines 2170-2231.
#
# For the host-side display path we apply a pragmatic subset:
#   - Pass-through if already has a leading article.
#   - Proper nouns (starts uppercase, no lowercase word after last separator)
#     get no "the" added.
#   - Everything else gets "the ".
# ---------------------------------------------------------------------------

def _the(name: str) -> str:
    """Simplified port of vendor/nethack/src/objnam.c::the() lines 2170-2231.

    Prepend 'the ' to *name* unless:
      - name already starts with 'the ', 'The ', 'a ', 'an ', 'some ', 'Some '
        → return as-is (vendor: idempotent, line 2181).
      - name starts with an uppercase letter AND the last word (after space/-)
        is also capitalised (i.e. a proper noun like "Wizard of Yendor")
        → return as-is (no article for proper names, vendor lines 2185-2196).
    Otherwise prepend 'the '.
    """
    if not name:
        return name
    if name.startswith(("the ", "The ", "a ", "an ", "An ", "A ", "some ", "Some ")):
        return name
    # Proper-noun check: starts uppercase and every space/hyphen-separated
    # word also starts uppercase → treat as proper name, no article.
    if name[0].isupper():
        words = name.replace('-', ' ').split()
        if all(w[0].isupper() for w in words if w):
            return name
    return "the " + name


# ---------------------------------------------------------------------------
# Cmap terrain descriptions (subset of vendor pager.c::lookat switch on cmap).
# ---------------------------------------------------------------------------

_CMAP_DESCRIPTION = {
    # tile-type -> noun  (vendor defsyms[] + the symidx switch in lookat).
    1: "wall",          # WALL (HWALL/VWALL collapsed)
    2: "wall",
    3: "wall",
    4: "floor of a room",
    5: "doorway",
    6: "floor of a room",
    7: "dark part of a room",  # corridor / dark room
    8: "doorway",       # ndoor (no door)
    9: "open door",
    10: "closed door",
    11: "staircase up",
    12: "staircase down",
    13: "ladder up",
    14: "ladder down",
    15: "altar",
    16: "fountain",
    17: "sink",
    18: "throne",
    19: "grave",
}


def _player_at(state, r: int, c: int) -> bool:
    return int(state.player_pos[0]) == r and int(state.player_pos[1]) == c


def _is_hallucinating(state) -> bool:
    """Return True when the HALLUCINATION timed status is active.

    Cite: vendor/nethack/src/do_name.c::rndmonnam line 1199 — replaces the
    canonical monster name with a random one when Hallu.
    """
    try:
        return int(state.status.timed_statuses[10]) > 0
    except (AttributeError, IndexError):
        return False


def _random_monster_name() -> str:
    """Return a random monster name (vendor do_name.c::rndmonnam line 1199)."""
    import random
    # Pick a random named monster from MONSTERS.
    named = [m.name for m in MONSTERS if m is not None and m.name]
    if not named:
        return "creature"
    return random.choice(named)


def _monster_at(state, r: int, c: int) -> Optional[str]:
    mons = state.monster_ai
    pos = np.asarray(mons.pos)
    alive = np.asarray(mons.alive).astype(bool)
    entry = np.asarray(mons.entry_idx).astype(np.int32)
    for i in range(pos.shape[0]):
        if alive[i] and int(pos[i, 0]) == r and int(pos[i, 1]) == c:
            # Vendor do_name.c:1199 — hallucinating: random monster name.
            if _is_hallucinating(state):
                return _random_monster_name()
            idx = int(entry[i])
            if 0 <= idx < len(MONSTERS) and MONSTERS[idx] is not None:
                return MONSTERS[idx].name or "creature"
    return None


def _objects_at(state, r: int, c: int) -> List[str]:
    """Return the canonical names of items on the floor at (r,c)."""
    try:
        branch = int(state.dungeon.current_branch)
        level = int(state.dungeon.current_level) - 1
        gi = state.dungeon.ground_items
        cats = np.asarray(gi.category)[branch, level, r, c]
        tids = np.asarray(gi.type_id)[branch, level, r, c]
        out: List[str] = []
        for cat, tid in zip(cats, tids):
            if int(cat) == 0:
                continue
            obj = OBJECTS[int(tid)] if 0 <= int(tid) < len(OBJECTS) else None
            if obj is None or obj.name is None:
                out.append("object")
            else:
                out.append(obj.name)
        return out
    except (AttributeError, IndexError):
        return []


def _terrain_noun(state, r: int, c: int) -> str:
    """Resolve the cmap noun at (r,c).  Vendor pager.c::lookat case S_*."""
    try:
        branch = int(state.dungeon.current_branch)
        level = int(state.dungeon.current_level) - 1
        tile = int(state.terrain[branch, level, r, c])
    except (AttributeError, IndexError):
        return "unexplored area"
    return _CMAP_DESCRIPTION.get(tile, "unexplored area")


# Engraving kind constants (mirrors engrave.py ENGR_* lines 43-47).
_ENGR_DUST    = 1
_ENGR_ENGRAVE = 2
_ENGR_BURN    = 3


def _engrave_descriptor(state, r: int, c: int) -> List[str]:
    """Return lines describing the engraving at (r, c), or [] if none.

    Vendor citation:
      engrave.c::read_engr_at lines 328-397 — per-kind pline + "You read: …"
      pager.c::lookat line 777 — "engraving" terrain noun (S_engroom/S_engrcorr)
      pager.c::add_quoted_engraving lines 1629-1667 — appends engraving text

    Host-side only (no jit).
    """
    try:
        has = bool(state.engrave.has_engraving[r, c])
    except (AttributeError, IndexError):
        return []
    if not has:
        return []

    # Decode text: uint8/int8 array, stop at first zero byte.
    # engrave.py stores int8 but values are ASCII (non-negative range 0-127).
    raw = np.asarray(state.engrave.text[r, c, :])
    chars = []
    for b in raw:
        b = int(b)
        if b == 0:
            break
        chars.append(chr(b & 0xFF))
    text = "".join(chars)
    if not text:
        return []

    kind = int(state.engrave.engraving_kind[r, c])

    # Vendor engrave.c::read_engr_at lines 329-348: per-kind announcement.
    if kind == _ENGR_DUST:
        # engrave.c:332 — "Something is written here in the dust."
        intro = "Something is written here in the dust."
    elif kind == _ENGR_BURN:
        # engrave.c:346-347 — "Some text has been burned into the floor here."
        intro = "Some text has been burned into the floor here."
    else:
        # ENGRAVE (kind==3) or unknown — engrave.c:340
        # "Something is engraved here on the floor."
        intro = "Something is engraved here on the floor."

    # Vendor engrave.c:396 — 'You("read: \"%s\"…", et, endpunct)'
    read_line = f"You read: \"{text}\"."
    return [intro, read_line]


# ---------------------------------------------------------------------------
# Public API — two entry points:
#   build_look_text(state, r, c)   — name of glyph at (r,c)    (vendor lookat)
#   build_look_here_text(state)    — full "what's at my feet"  (vendor look_here)
# ---------------------------------------------------------------------------

def build_look_text(state, target_row: int, target_col: int) -> str:
    """Vendor parity: pager.c::lookat (lines 656-810).

    Returns a noun-phrase naming the glyph at (target_row, target_col):
      - Player: "yourself"
      - Monster: "the <name>" (e.g. "the giant ant"); G_UNIQ monsters get
        proper-name treatment (e.g. "the Wizard of Yendor").
        Vendor: look_at_monster calls the() / The() on the monster name
        (pager.c ~line 700).
      - Object: the object's name (e.g. "long sword") — article added by
        caller (vendor doname handles articles internally).
      - Cmap (terrain): "the <noun>" (e.g. "the doorway", "the altar").
        Vendor: lookat emits terrain noun via the() (pager.c ~line 780).
      - Engraving: appends engrave description lines (pager.c line 1612,
        engrave.c::read_engr_at lines 328-397).
    """
    if _player_at(state, target_row, target_col):
        return "yourself"

    mon = _monster_at(state, target_row, target_col)
    if mon is not None:
        return _the(mon)

    objs = _objects_at(state, target_row, target_col)
    if objs:
        primary = objs[0]
    else:
        terrain = _terrain_noun(state, target_row, target_col)
        primary = _the(terrain)

    engrave_lines = _engrave_descriptor(state, target_row, target_col)
    if engrave_lines:
        return "\n".join([primary] + engrave_lines)
    return primary


def build_look_here_text(state) -> str:
    """Vendor parity: invent.c::look_here (lines 4101-4326).

    Returns the prose for the player's own cell, e.g.::

        "You see here a long sword."
        "You see no objects here."
        "Things that are here:\\n  a long sword\\n  three darts"

    Engraving lines appended after items/no-objects line per
    engrave.c::read_engr_at lines 328-397.

    Citation: vendor/nethack/src/invent.c::look_here lines 4180-4310.
    """
    pr = int(state.player_pos[0])
    pc = int(state.player_pos[1])
    objs = _objects_at(state, pr, pc)

    if not objs:
        # Vendor: 'You("%s no objects here.", verb);'  (invent.c:4247)
        lines = ["You see no objects here."]
    elif len(objs) == 1:
        # Vendor: 'You("%s here %s.", verb, doname_with_price(otmp));'
        lines = [f"You see here {_an(objs[0])}."]
    else:
        # Multiple objects.
        lines = ["Things that are here:"]
        for o in objs:
            lines.append(f"  {_an(o)}")

    engrave_lines = _engrave_descriptor(state, pr, pc)
    return "\n".join(lines + engrave_lines)
