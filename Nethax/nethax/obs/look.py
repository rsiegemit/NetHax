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
# Vendor "an / a" article helper (vendor objnam.c::an).  Strips definite
# article from words starting with vowels; respects "the" / "some" prefixes
# already in the noun.
# ---------------------------------------------------------------------------

_VOWELS = set("aeiouAEIOU")


def _an(name: str) -> str:
    """Vendor objnam.c::an(name) — prepend 'a' or 'an'."""
    if not name:
        return name
    # Vendor edge cases: names starting with "the ", "some ", a digit, or
    # proper nouns get no article.  Simplified rule below.
    if name.startswith(("the ", "some ", "a ", "an ")):
        return name
    if name[0].isdigit():
        return name
    if name[0] in _VOWELS:
        return "an " + name
    return "a " + name


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


def _monster_at(state, r: int, c: int) -> Optional[str]:
    mons = state.monster_ai
    pos = np.asarray(mons.pos)
    alive = np.asarray(mons.alive).astype(bool)
    entry = np.asarray(mons.entry_idx).astype(np.int32)
    for i in range(pos.shape[0]):
        if alive[i] and int(pos[i, 0]) == r and int(pos[i, 1]) == c:
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


# ---------------------------------------------------------------------------
# Public API — two entry points:
#   build_look_text(state, r, c)   — name of glyph at (r,c)    (vendor lookat)
#   build_look_here_text(state)    — full "what's at my feet"  (vendor look_here)
# ---------------------------------------------------------------------------

def build_look_text(state, target_row: int, target_col: int) -> str:
    """Vendor parity: pager.c::lookat (lines 656-810).

    Returns a noun-phrase naming the glyph at (target_row, target_col):
      - Player: "yourself"
      - Monster: the monster's name (e.g. "giant ant")
      - Object: the object's name (e.g. "long sword")
      - Cmap (terrain): the terrain noun (e.g. "doorway", "altar", "wall")
    """
    if _player_at(state, target_row, target_col):
        return "yourself"

    mon = _monster_at(state, target_row, target_col)
    if mon is not None:
        return mon

    objs = _objects_at(state, target_row, target_col)
    if objs:
        return objs[0]

    return _terrain_noun(state, target_row, target_col)


def build_look_here_text(state) -> str:
    """Vendor parity: invent.c::look_here (lines 4101-4326).

    Returns the prose for the player's own cell, e.g.::

        "You see here a long sword."
        "You see no objects here."
        "Things that are here:\\n  a long sword\\n  three darts"

    Citation: vendor/nethack/src/invent.c::look_here lines 4180-4310.
    """
    pr = int(state.player_pos[0])
    pc = int(state.player_pos[1])
    objs = _objects_at(state, pr, pc)

    if not objs:
        # Vendor: 'You("%s no objects here.", verb);'  (invent.c:4247)
        return "You see no objects here."

    if len(objs) == 1:
        # Vendor: 'You("%s here %s.", verb, doname_with_price(otmp));'
        return f"You see here {_an(objs[0])}."

    # Multiple objects.
    out = ["Things that are here:"]
    for o in objs:
        out.append(f"  {_an(o)}")
    return "\n".join(out)
