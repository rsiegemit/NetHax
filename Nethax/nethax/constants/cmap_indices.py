"""Single source of truth for the NetHack cmap (screen-symbol) index layout.

Two vendor trees disagree on the cmap enum ordering:

* **NLE 3.x** (``vendor/nle/include/rm.h``, lines 116-227): the layout NLE is
  actually built from.  Active by default and what
  ``tests/test_nle_byte_parity.py`` validates against.

* **NetHack 5.x / 3.7** (``vendor/nethack/include/defsym.h``, lines 90-183):
  inserts ``S_litcorr`` (23), ``S_engroom`` (21), and ``S_engrcorr`` (24) into
  the dungeon-character block, pushing the staircase / altar / fountain / trap
  blocks down by 2 indices.  Used by Nethax only when an external caller
  explicitly requests it (e.g. replaying a real NetHack 5.x save).

The active layout is chosen at import time via the ``NETHAX_VENDOR_TREE``
environment variable:

  - ``nle_3x``     (default) — vendor/nle/include/rm.h
  - ``nethack_5x``           — vendor/nethack/include/defsym.h

Callers should ``from Nethax.nethax.constants.cmap_indices import S_upstair``
(or use ``ACTIVE["S_upstair"]``) and never hard-code numeric cmap indices.

Tables exposed at import time (all numpy / pure Python — JAX-friendly to
wrap into ``jnp.array`` constants at module load):

  - ``ACTIVE``             : dict[str, int]   — symbol name -> cmap index
  - ``CMAP_TO_CHAR``       : np.ndarray[64]   — cmap index -> ASCII byte
  - ``CMAP_TO_COLOR``      : np.ndarray[64]   — cmap index -> ANSI colour (0-15)
  - ``CMAP_DESC``          : dict[int, str]   — cmap index -> pager description
  - All named constants (``S_stone``, ``S_vwall``, ..., ``S_arrow_trap``)
    resolved against the active layout.

The same symbol always carries the same char/colour/description; only its
numeric index shifts between layouts.  Build the per-index tables by
indexing the per-symbol dictionaries through ``ACTIVE``.
"""

from __future__ import annotations

import os

import numpy as np


# ---------------------------------------------------------------------------
# Layout 1 — NLE 3.x (vendor/nle/include/rm.h:116-227).
# ---------------------------------------------------------------------------

CMAP_3X: dict[str, int] = {
    # Dungeon characters (0..41)
    "S_stone":      0,
    "S_vwall":      1,
    "S_hwall":      2,
    "S_tlcorn":     3,
    "S_trcorn":     4,
    "S_blcorn":     5,
    "S_brcorn":     6,
    "S_crwall":     7,
    "S_tuwall":     8,
    "S_tdwall":     9,
    "S_tlwall":    10,
    "S_trwall":    11,
    "S_ndoor":     12,
    "S_vodoor":    13,
    "S_hodoor":    14,
    "S_vcdoor":    15,
    "S_hcdoor":    16,
    "S_bars":      17,
    "S_tree":      18,
    "S_room":      19,
    "S_darkroom":  20,
    "S_corr":      21,
    "S_litcorr":   22,
    "S_upstair":   23,
    "S_dnstair":   24,
    "S_upladder":  25,
    "S_dnladder":  26,
    "S_altar":     27,
    "S_grave":     28,
    "S_throne":    29,
    "S_sink":      30,
    "S_fountain":  31,
    "S_pool":      32,
    "S_ice":       33,
    "S_lava":      34,
    "S_vodbridge": 35,
    "S_hodbridge": 36,
    "S_vcdbridge": 37,
    "S_hcdbridge": 38,
    "S_air":       39,
    "S_cloud":     40,
    "S_water":     41,
    # Traps (42..63)
    "S_arrow_trap":           42,
    "S_dart_trap":            43,
    "S_falling_rock_trap":    44,
    "S_squeaky_board":        45,
    "S_bear_trap":            46,
    "S_land_mine":            47,
    "S_rolling_boulder_trap": 48,
    "S_sleeping_gas_trap":    49,
    "S_rust_trap":            50,
    "S_fire_trap":            51,
    "S_pit":                  52,
    "S_spiked_pit":           53,
    "S_hole":                 54,
    "S_trap_door":            55,
    "S_teleportation_trap":   56,
    "S_level_teleporter":     57,
    "S_magic_portal":         58,
    "S_web":                  59,
    "S_statue_trap":          60,
    "S_magic_trap":           61,
    "S_anti_magic_trap":      62,
    "S_polymorph_trap":       63,
}

# 3.x has no engraving symbols (5.x-only); branch ladders are absent too.
# Anything referenced by code but missing here is mapped to a sane fallback
# in _CHAR_FOR / _COLOR_FOR (e.g. S_engroom -> reuse S_room char).

# ---------------------------------------------------------------------------
# Layout 2 — NetHack 5.x / 3.7 (vendor/nethack/include/defsym.h:90-183).
# ---------------------------------------------------------------------------

CMAP_5X: dict[str, int] = {
    "S_stone":     0,
    "S_vwall":     1,
    "S_hwall":     2,
    "S_tlcorn":    3,
    "S_trcorn":    4,
    "S_blcorn":    5,
    "S_brcorn":    6,
    "S_crwall":    7,
    "S_tuwall":    8,
    "S_tdwall":    9,
    "S_tlwall":   10,
    "S_trwall":   11,
    "S_ndoor":    12,
    "S_vodoor":   13,
    "S_hodoor":   14,
    "S_vcdoor":   15,
    "S_hcdoor":   16,
    "S_bars":     17,
    "S_tree":     18,
    "S_room":     19,
    "S_darkroom": 20,
    "S_engroom":  21,   # NEW in 5.x: engraving in a room
    "S_corr":     22,   # shifted +1 vs 3.x (3.x: 21)
    "S_litcorr":  23,
    "S_engrcorr": 24,   # NEW in 5.x: engraving in a corridor
    "S_upstair":  25,   # shifted +2 vs 3.x (3.x: 23)
    "S_dnstair":  26,
    "S_upladder": 27,
    "S_dnladder": 28,
    "S_brupstair":  29, # NEW in 5.x: branch staircase up
    "S_brdnstair":  30,
    "S_brupladder": 31,
    "S_brdnladder": 32,
    "S_altar":    33,   # shifted +6 vs 3.x (3.x: 27)
    "S_grave":    34,
    "S_throne":   35,
    "S_sink":     36,
    "S_fountain": 37,
    "S_pool":     38,
    "S_ice":      39,
    "S_lava":     40,
    "S_lavawall": 41,   # NEW in 5.x
    "S_vodbridge": 42,
    "S_hodbridge": 43,
    "S_vcdbridge": 44,
    "S_hcdbridge": 45,
    "S_air":      46,
    "S_cloud":    47,
    "S_water":    48,
    # Traps (49..70)
    "S_arrow_trap":           49,
    "S_dart_trap":            50,
    "S_falling_rock_trap":    51,
    "S_squeaky_board":        52,
    "S_bear_trap":            53,
    "S_land_mine":            54,
    "S_rolling_boulder_trap": 55,
    "S_sleeping_gas_trap":    56,
    "S_rust_trap":            57,
    "S_fire_trap":            58,
    "S_pit":                  59,
    "S_spiked_pit":           60,
    "S_hole":                 61,
    "S_trap_door":            62,
    "S_teleportation_trap":   63,
    "S_level_teleporter":     64,
    "S_magic_portal":         65,
    "S_web":                  66,
    "S_statue_trap":          67,
    "S_magic_trap":           68,
    "S_anti_magic_trap":      69,
    "S_polymorph_trap":       70,
}


# ---------------------------------------------------------------------------
# Per-symbol char / colour / description (layout-independent).
#
# Same symbol → same character in both layouts; only the numeric index moves.
# Char/colour sources: vendor/nethack/include/defsym.h PCHAR / PCHAR2 entries
# (the canonical declarations both forks reuse).
# ---------------------------------------------------------------------------

# CLR_* constants from vendor/nethack/include/color.h.
_CLR_BLACK         = 0
_CLR_RED           = 1
_CLR_GREEN         = 2
_CLR_BROWN         = 3
_CLR_BLUE          = 4
_CLR_MAGENTA       = 5
_CLR_CYAN          = 6
_CLR_GRAY          = 7
_CLR_ORANGE        = 9
_CLR_BRIGHT_GREEN  = 10
_CLR_YELLOW        = 11
_CLR_BRIGHT_BLUE   = 12
_CLR_BRIGHT_CYAN   = 14
_CLR_WHITE         = 15

_CHAR_FOR: dict[str, str] = {
    "S_stone":      " ",
    "S_vwall":      "|",
    "S_hwall":      "-",
    "S_tlcorn":     "-",
    "S_trcorn":     "-",
    "S_blcorn":     "-",
    "S_brcorn":     "-",
    "S_crwall":     "-",
    "S_tuwall":     "-",
    "S_tdwall":     "-",
    "S_tlwall":     "|",
    "S_trwall":     "|",
    "S_ndoor":      ".",
    "S_vodoor":     "-",
    "S_hodoor":     "|",
    "S_vcdoor":     "+",
    "S_hcdoor":     "+",
    "S_bars":       "#",
    "S_tree":       "#",
    "S_room":       ".",
    "S_darkroom":   ".",
    "S_engroom":    "`",
    "S_corr":       "#",
    "S_litcorr":    "#",
    "S_engrcorr":   "`",
    "S_upstair":    "<",
    "S_dnstair":    ">",
    "S_upladder":   "<",
    "S_dnladder":   ">",
    "S_brupstair":  "<",
    "S_brdnstair":  ">",
    "S_brupladder": "<",
    "S_brdnladder": ">",
    "S_altar":      "_",
    "S_grave":      "|",
    "S_throne":     "\\",
    "S_sink":       "{",
    "S_fountain":   "{",
    "S_pool":       "}",
    "S_ice":        ".",
    "S_lava":       "}",
    "S_lavawall":   "}",
    "S_vodbridge":  ".",
    "S_hodbridge":  ".",
    "S_vcdbridge":  "#",
    "S_hcdbridge":  "#",
    "S_air":        " ",
    "S_cloud":      "#",
    "S_water":      "}",
    # Traps all render as '^' (vendor defsym.h:157-180).
    "S_arrow_trap":           "^",
    "S_dart_trap":            "^",
    "S_falling_rock_trap":    "^",
    "S_squeaky_board":        "^",
    "S_bear_trap":            "^",
    "S_land_mine":            "^",
    "S_rolling_boulder_trap": "^",
    "S_sleeping_gas_trap":    "^",
    "S_rust_trap":            "^",
    "S_fire_trap":            "^",
    "S_pit":                  "^",
    "S_spiked_pit":           "^",
    "S_hole":                 "^",
    "S_trap_door":            "^",
    "S_teleportation_trap":   "^",
    "S_level_teleporter":     "^",
    "S_magic_portal":         "^",
    "S_web":                  '"',
    "S_statue_trap":          "^",
    "S_magic_trap":           "^",
    "S_anti_magic_trap":      "^",
    "S_polymorph_trap":       "^",
}

_COLOR_FOR: dict[str, int] = {
    "S_stone":      _CLR_BLACK,
    "S_vwall":      _CLR_GRAY,
    "S_hwall":      _CLR_GRAY,
    "S_tlcorn":     _CLR_GRAY,
    "S_trcorn":     _CLR_GRAY,
    "S_blcorn":     _CLR_GRAY,
    "S_brcorn":     _CLR_GRAY,
    "S_crwall":     _CLR_GRAY,
    "S_tuwall":     _CLR_GRAY,
    "S_tdwall":     _CLR_GRAY,
    "S_tlwall":     _CLR_GRAY,
    "S_trwall":     _CLR_GRAY,
    "S_ndoor":      _CLR_GRAY,
    "S_vodoor":     _CLR_BROWN,
    "S_hodoor":     _CLR_BROWN,
    "S_vcdoor":     _CLR_BROWN,
    "S_hcdoor":     _CLR_BROWN,
    "S_bars":       _CLR_GRAY,
    "S_tree":       _CLR_GREEN,
    "S_room":       _CLR_GRAY,
    "S_darkroom":   _CLR_BLACK,
    "S_engroom":    _CLR_GRAY,
    "S_corr":       _CLR_GRAY,
    "S_litcorr":    _CLR_GRAY,
    "S_engrcorr":   _CLR_GRAY,
    "S_upstair":    _CLR_GRAY,
    "S_dnstair":    _CLR_GRAY,
    "S_upladder":   _CLR_BROWN,
    "S_dnladder":   _CLR_BROWN,
    "S_brupstair":  _CLR_YELLOW,
    "S_brdnstair":  _CLR_YELLOW,
    "S_brupladder": _CLR_YELLOW,
    "S_brdnladder": _CLR_YELLOW,
    "S_altar":      _CLR_GRAY,
    "S_grave":      _CLR_WHITE,
    "S_throne":     _CLR_GRAY,
    "S_sink":       _CLR_WHITE,
    "S_fountain":   _CLR_BRIGHT_BLUE,
    "S_pool":       _CLR_BLUE,
    "S_ice":        _CLR_CYAN,
    "S_lava":       _CLR_RED,
    "S_lavawall":   _CLR_ORANGE,
    "S_vodbridge":  _CLR_BROWN,
    "S_hodbridge":  _CLR_BROWN,
    "S_vcdbridge":  _CLR_BROWN,
    "S_hcdbridge":  _CLR_BROWN,
    "S_air":        _CLR_CYAN,
    "S_cloud":      _CLR_GRAY,
    "S_water":      _CLR_BRIGHT_BLUE,
    "S_arrow_trap":           _CLR_GRAY,
    "S_dart_trap":            _CLR_GRAY,
    "S_falling_rock_trap":    _CLR_GRAY,
    "S_squeaky_board":        _CLR_BROWN,
    "S_bear_trap":            _CLR_RED,
    "S_land_mine":            _CLR_RED,
    "S_rolling_boulder_trap": _CLR_GRAY,
    "S_sleeping_gas_trap":    _CLR_GRAY,
    "S_rust_trap":            _CLR_BLUE,
    "S_fire_trap":            _CLR_ORANGE,
    "S_pit":                  _CLR_BLACK,
    "S_spiked_pit":           _CLR_BLACK,
    "S_hole":                 _CLR_BROWN,
    "S_trap_door":            _CLR_BROWN,
    "S_teleportation_trap":   _CLR_MAGENTA,
    "S_level_teleporter":     _CLR_MAGENTA,
    "S_magic_portal":         _CLR_MAGENTA,
    "S_web":                  _CLR_GRAY,
    "S_statue_trap":          _CLR_GRAY,
    "S_magic_trap":           _CLR_GRAY,
    "S_anti_magic_trap":      _CLR_GRAY,
    "S_polymorph_trap":       _CLR_BRIGHT_GREEN,
}

_DESC_FOR: dict[str, str] = {
    "S_stone":      "dark part of a room",
    "S_vwall":      "wall",
    "S_hwall":      "wall",
    "S_tlcorn":     "wall",
    "S_trcorn":     "wall",
    "S_blcorn":     "wall",
    "S_brcorn":     "wall",
    "S_crwall":     "wall",
    "S_tuwall":     "wall",
    "S_tdwall":     "wall",
    "S_tlwall":     "wall",
    "S_trwall":     "wall",
    "S_ndoor":      "doorway",
    "S_vodoor":     "open door",
    "S_hodoor":     "open door",
    "S_vcdoor":     "closed door",
    "S_hcdoor":     "closed door",
    "S_bars":       "iron bars",
    "S_tree":       "tree",
    "S_room":       "floor of a room",
    "S_darkroom":   "dark part of a room",
    "S_engroom":    "engraving",
    "S_corr":       "corridor",
    "S_litcorr":    "lit corridor",
    "S_engrcorr":   "engraving",
    "S_upstair":    "staircase up",
    "S_dnstair":    "staircase down",
    "S_upladder":   "ladder up",
    "S_dnladder":   "ladder down",
    "S_brupstair":  "branch staircase up",
    "S_brdnstair":  "branch staircase down",
    "S_brupladder": "branch ladder up",
    "S_brdnladder": "branch ladder down",
    "S_altar":      "altar",
    "S_grave":      "grave",
    "S_throne":     "throne",
    "S_sink":       "sink",
    "S_fountain":   "fountain",
    "S_pool":       "water",
    "S_ice":        "ice",
    "S_lava":       "molten lava",
    "S_lavawall":   "wall of lava",
    "S_vodbridge":  "drawbridge",
    "S_hodbridge":  "drawbridge",
    "S_vcdbridge":  "drawbridge",
    "S_hcdbridge":  "drawbridge",
    "S_air":        "air",
    "S_cloud":      "cloud",
    "S_water":      "water",
    "S_arrow_trap":           "arrow trap",
    "S_dart_trap":            "dart trap",
    "S_falling_rock_trap":    "falling rock trap",
    "S_squeaky_board":        "squeaky board",
    "S_bear_trap":            "bear trap",
    "S_land_mine":            "land mine",
    "S_rolling_boulder_trap": "rolling boulder trap",
    "S_sleeping_gas_trap":    "sleeping gas trap",
    "S_rust_trap":            "rust trap",
    "S_fire_trap":            "fire trap",
    "S_pit":                  "pit",
    "S_spiked_pit":           "spiked pit",
    "S_hole":                 "hole",
    "S_trap_door":            "trap door",
    "S_teleportation_trap":   "teleportation trap",
    "S_level_teleporter":     "level teleporter",
    "S_magic_portal":         "magic portal",
    "S_web":                  "web",
    "S_statue_trap":          "statue trap",
    "S_magic_trap":           "magic trap",
    "S_anti_magic_trap":      "anti magic trap",
    "S_polymorph_trap":       "polymorph trap",
}


# ---------------------------------------------------------------------------
# Runtime layout selection.
# ---------------------------------------------------------------------------

_VENDOR_TREE_3X: str = "nle_3x"
_VENDOR_TREE_5X: str = "nethack_5x"


def _select_layout() -> tuple[str, dict[str, int]]:
    """Resolve the active layout once, at import time.

    Reads the ``NETHAX_VENDOR_TREE`` env var; defaults to ``nle_3x``.
    Unrecognised values fall back to ``nle_3x`` (no surprise alt-tree).
    """
    env = os.environ.get("NETHAX_VENDOR_TREE", _VENDOR_TREE_3X).strip().lower()
    if env == _VENDOR_TREE_5X:
        return _VENDOR_TREE_5X, CMAP_5X
    return _VENDOR_TREE_3X, CMAP_3X


VENDOR_TREE, ACTIVE = _select_layout()


def is_nle_3x() -> bool:
    """True iff the active layout is NLE 3.x (vendor/nle/include/rm.h)."""
    return VENDOR_TREE == _VENDOR_TREE_3X


def is_nethack_5x() -> bool:
    """True iff the active layout is NetHack 5.x (vendor/nethack/include/rm.h)."""
    return VENDOR_TREE == _VENDOR_TREE_5X


# ---------------------------------------------------------------------------
# Named constants exposed from the selected layout.
#
# Anything that exists in BOTH layouts gets a top-level name; symbols that
# only exist in one tree (e.g. S_engroom is 5.x-only; S_brupstair is 5.x-only)
# are exposed via ``ACTIVE.get(...)`` returning None when unavailable.
# ---------------------------------------------------------------------------

S_stone     = ACTIVE["S_stone"]
S_vwall     = ACTIVE["S_vwall"]
S_hwall     = ACTIVE["S_hwall"]
S_tlcorn    = ACTIVE["S_tlcorn"]
S_trcorn    = ACTIVE["S_trcorn"]
S_blcorn    = ACTIVE["S_blcorn"]
S_brcorn    = ACTIVE["S_brcorn"]
S_crwall    = ACTIVE["S_crwall"]
S_tuwall    = ACTIVE["S_tuwall"]
S_tdwall    = ACTIVE["S_tdwall"]
S_tlwall    = ACTIVE["S_tlwall"]
S_trwall    = ACTIVE["S_trwall"]
S_ndoor     = ACTIVE["S_ndoor"]
S_vodoor    = ACTIVE["S_vodoor"]
S_hodoor    = ACTIVE["S_hodoor"]
S_vcdoor    = ACTIVE["S_vcdoor"]
S_hcdoor    = ACTIVE["S_hcdoor"]
S_bars      = ACTIVE["S_bars"]
S_tree      = ACTIVE["S_tree"]
S_room      = ACTIVE["S_room"]
S_darkroom  = ACTIVE["S_darkroom"]
S_corr      = ACTIVE["S_corr"]
S_litcorr   = ACTIVE["S_litcorr"]
S_upstair   = ACTIVE["S_upstair"]
S_dnstair   = ACTIVE["S_dnstair"]
S_upladder  = ACTIVE["S_upladder"]
S_dnladder  = ACTIVE["S_dnladder"]
S_altar     = ACTIVE["S_altar"]
S_grave     = ACTIVE["S_grave"]
S_throne    = ACTIVE["S_throne"]
S_sink      = ACTIVE["S_sink"]
S_fountain  = ACTIVE["S_fountain"]
S_pool      = ACTIVE["S_pool"]
S_ice       = ACTIVE["S_ice"]
S_lava      = ACTIVE["S_lava"]
S_vodbridge = ACTIVE["S_vodbridge"]
S_hodbridge = ACTIVE["S_hodbridge"]
S_vcdbridge = ACTIVE["S_vcdbridge"]
S_hcdbridge = ACTIVE["S_hcdbridge"]
S_air       = ACTIVE["S_air"]
S_cloud     = ACTIVE["S_cloud"]
S_water     = ACTIVE["S_water"]
S_arrow_trap = ACTIVE["S_arrow_trap"]


# 5.x-only symbols (None under 3.x).
S_engroom    = ACTIVE.get("S_engroom")
S_engrcorr   = ACTIVE.get("S_engrcorr")
S_lavawall   = ACTIVE.get("S_lavawall")
S_brupstair  = ACTIVE.get("S_brupstair")
S_brdnstair  = ACTIVE.get("S_brdnstair")
S_brupladder = ACTIVE.get("S_brupladder")
S_brdnladder = ACTIVE.get("S_brdnladder")


# ---------------------------------------------------------------------------
# Per-index lookup tables — built once at import.
#
# We size all tables at 64 entries (max cmap index in either layout is 63 for
# 3.x and 70 for 5.x; we use 80 to be safe for 5.x).  Callers that need a
# fixed-size JAX constant should wrap with ``jnp.asarray(..., dtype=...)``.
# ---------------------------------------------------------------------------

_TABLE_WIDTH: int = 80  # >= max index across both layouts (5.x: 70).


def _build_char_table() -> np.ndarray:
    arr = np.full((_TABLE_WIDTH,), ord(" "), dtype=np.uint8)
    for sym, idx in ACTIVE.items():
        ch = _CHAR_FOR.get(sym, " ")
        arr[idx] = ord(ch)
    return arr


def _build_color_table() -> np.ndarray:
    arr = np.full((_TABLE_WIDTH,), _CLR_GRAY, dtype=np.uint8)
    arr[0] = _CLR_BLACK  # S_stone -> black
    for sym, idx in ACTIVE.items():
        arr[idx] = _COLOR_FOR.get(sym, _CLR_GRAY)
    return arr


def _build_desc_dict() -> dict[int, str]:
    out: dict[int, str] = {}
    for sym, idx in ACTIVE.items():
        out[idx] = _DESC_FOR.get(sym, "")
    return out


CMAP_TO_CHAR: np.ndarray = _build_char_table()
CMAP_TO_COLOR: np.ndarray = _build_color_table()
CMAP_DESC: dict[int, str] = _build_desc_dict()


__all__ = [
    "ACTIVE",
    "CMAP_3X",
    "CMAP_5X",
    "CMAP_TO_CHAR",
    "CMAP_TO_COLOR",
    "CMAP_DESC",
    "VENDOR_TREE",
    "is_nle_3x",
    "is_nethack_5x",
    # Named constants
    "S_stone", "S_vwall", "S_hwall", "S_tlcorn", "S_trcorn", "S_blcorn",
    "S_brcorn", "S_crwall", "S_tuwall", "S_tdwall", "S_tlwall", "S_trwall",
    "S_ndoor", "S_vodoor", "S_hodoor", "S_vcdoor", "S_hcdoor", "S_bars",
    "S_tree", "S_room", "S_darkroom", "S_corr", "S_litcorr", "S_upstair",
    "S_dnstair", "S_upladder", "S_dnladder", "S_altar", "S_grave",
    "S_throne", "S_sink", "S_fountain", "S_pool", "S_ice", "S_lava",
    "S_vodbridge", "S_hodbridge", "S_vcdbridge", "S_hcdbridge", "S_air",
    "S_cloud", "S_water", "S_arrow_trap",
    # 5.x-only (None under 3.x)
    "S_engroom", "S_engrcorr", "S_lavawall",
    "S_brupstair", "S_brdnstair", "S_brupladder", "S_brdnladder",
]
