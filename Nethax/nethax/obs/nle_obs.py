# Wave 2: implemented build_blstats, build_glyphs, build_message, build_tty,
#          build_nle_observation.
# Wave 3: implemented build_colors, build_specials, build_tty_colors,
#          build_inv_glyphs, build_inv_letters, build_inv_oclasses.
# Wave 3: implemented build_inventory_strings (objnam-style; delegates to inv_strs.py)
# Wave 4: wired build_specials (trap/pile/corpse/object/secret-door overlays),
#          build_internal (NLE 9-int internal), build_screen_descriptions
#          (per-glyph descriptions), upgraded build_colors with monster/object
#          glyph overlays via _GLYPH_TO_COLOR static table.
"""NLE-parity observation builder for nethax.

Produces observation dicts whose keys, shapes, and dtypes match NLE exactly,
so RL agents trained on NLE can run on nethax without rewiring.

Canonical sources:
  - vendor/nle/nle/nethack/nethack.py  OBSERVATION_DESC (lines 32-50)
  - vendor/nle/nle/env/base.py         observation_space construction (36-134)
  - vendor/nle/include/nleobs.h        C-side observation struct (48-72)
  - vendor/nle/win/rl/pynethack.cc     glyph offset constants
  - vendor/nethack/include/defsym.h    cmap symbol indices and ASCII chars
  - vendor/nethack/include/sym.h       MAXPCHARS / cmap enum

Wave 2 status: build_blstats, build_glyphs, build_message, build_tty implemented.
"""

import jax
import jax.numpy as jnp

from Nethax.nethax.constants.blstats import (
    BL_X, BL_Y, BL_STR25, BL_STR125, BL_DEX, BL_CON, BL_INT, BL_WIS, BL_CHA,
    BL_SCORE, BL_HP, BL_HPMAX, BL_DEPTH, BL_GOLD, BL_ENE, BL_ENEMAX,
    BL_AC, BL_HD, BL_XP, BL_EXP, BL_TIME, BL_HUNGER, BL_CAP,
    BL_DNUM, BL_DLEVEL, BL_CONDITION, BL_ALIGN,
)
from Nethax.nethax.constants.glyphs import (
    GLYPH_MON_OFF, GLYPH_PET_OFF, GLYPH_INVIS_OFF, GLYPH_DETECT_OFF,
    GLYPH_BODY_OFF, GLYPH_RIDDEN_OFF, GLYPH_OBJ_OFF, GLYPH_CMAP_OFF,
    GLYPH_EXPLODE_OFF, GLYPH_ZAP_OFF, GLYPH_SWALLOW_OFF,
    GLYPH_WARNING_OFF, GLYPH_STATUE_OFF, MAX_GLYPH, NO_GLYPH,
)
from Nethax.nethax.constants import NUM_TILE_TYPES


# ---------------------------------------------------------------------------
# Monster-symbol → char  (vendor/nethack/include/monsym.h ::: S_*)
# Index = MonsterSymbol value (0..60), entry = ASCII char.
# ---------------------------------------------------------------------------

def _build_mon_sym_to_char() -> jnp.ndarray:
    table = [ord(' ')] + [0] * 60
    # Lowercase a-z (1..26)
    for i in range(26):
        table[1 + i] = ord('a') + i
    # Uppercase A-Z (27..52)
    for i in range(26):
        table[27 + i] = ord('A') + i
    table[53] = ord('@')   # S_HUMAN
    table[54] = ord(' ')   # S_GHOST
    table[55] = ord("'")   # S_GOLEM
    table[56] = ord('&')   # S_DEMON
    table[57] = ord(';')   # S_EEL
    table[58] = ord(':')   # S_LIZARD
    table[59] = ord('~')   # S_WORM_TAIL
    table[60] = ord(']')   # S_MIMIC_DEF
    return jnp.array(table, dtype=jnp.uint8)


_MON_SYM_TO_CHAR = _build_mon_sym_to_char()


# Monster-index → char  (resolves MONSTERS[idx].symbol → MON_SYM_TO_CHAR).
# Built once at module-load; 381-entry uint8 table.
def _build_mon_idx_to_char() -> jnp.ndarray:
    from Nethax.nethax.constants.monsters import MONSTERS
    sym_to_char_py = bytes(int(_MON_SYM_TO_CHAR[i]) for i in range(61))
    arr = []
    for m in MONSTERS:
        if m is None:
            arr.append(ord(' '))
        else:
            s = int(m.symbol)
            arr.append(sym_to_char_py[s] if 0 <= s < 61 else ord('?'))
    return jnp.array(arr, dtype=jnp.uint8)


_MON_IDX_TO_CHAR = _build_mon_idx_to_char()


# Object-index → char  (resolves OBJECTS[idx].class_ → def_oc_syms char).
# vendor/nethack/src/objects.c::def_oc_syms.
def _build_obj_idx_to_char() -> jnp.ndarray:
    from Nethax.nethax.constants.objects import OBJECTS
    _CLASS_CHAR = {
        0: '*', 1: ']', 2: ')', 3: '[', 4: '=', 5: '"', 6: '(', 7: '%',
        8: '!', 9: '?', 10: '+', 11: '/', 12: '$', 13: '*', 14: '`',
        15: '0', 16: '_', 17: '.',
    }
    arr = []
    for o in OBJECTS:
        c = ord(_CLASS_CHAR.get(int(o.class_), '?')) if o is not None else ord(' ')
        arr.append(c)
    return jnp.array(arr, dtype=jnp.uint8)


_OBJ_IDX_TO_CHAR = _build_obj_idx_to_char()

# ---------------------------------------------------------------------------
# Key registry — 17 keys, matches vendor/nle/nle/nethack/nethack.py exactly
# ---------------------------------------------------------------------------

NLE_OBSERVATION_KEYS = (
    "glyphs",
    "chars",
    "colors",
    "specials",
    "blstats",
    "message",
    "program_state",
    "internal",
    "inv_glyphs",
    "inv_letters",
    "inv_oclasses",
    "inv_strs",
    "screen_descriptions",
    "tty_chars",
    "tty_colors",
    "tty_cursor",
    "misc",
)

# ---------------------------------------------------------------------------
# Shape and dtype tables — values verified against nleobs.h and nethack.py
# ---------------------------------------------------------------------------

# DUNGEON_SHAPE = (ROWNO=21, COLNO-1=79)
# TERMINAL_SHAPE = (NLE_TERM_LI=24, NLE_TERM_CO=80)
# INV_SIZE = (NLE_INVENTORY_SIZE=55,)

NLE_OBSERVATION_SHAPES: dict[str, tuple[int, ...]] = {
    "glyphs":               (21, 79),
    "chars":                (21, 79),
    "colors":               (21, 79),
    "specials":             (21, 79),
    "blstats":              (27,),
    "message":              (256,),
    "program_state":        (6,),
    "internal":             (9,),
    "inv_glyphs":           (55,),
    "inv_letters":          (55,),
    "inv_oclasses":         (55,),
    "inv_strs":             (55, 80),
    "screen_descriptions":  (21, 79, 80),
    "tty_chars":            (24, 80),
    "tty_colors":           (24, 80),
    "tty_cursor":           (2,),
    "misc":                 (3,),
}

NLE_OBSERVATION_DTYPES: dict[str, jnp.dtype] = {
    "glyphs":               jnp.int16,
    "chars":                jnp.uint8,
    "colors":               jnp.uint8,
    "specials":             jnp.uint8,
    "blstats":              jnp.int64,
    "message":              jnp.uint8,
    "program_state":        jnp.int32,
    "internal":             jnp.int32,
    "inv_glyphs":           jnp.int16,
    "inv_letters":          jnp.uint8,
    "inv_oclasses":         jnp.uint8,
    "inv_strs":             jnp.uint8,
    "screen_descriptions":  jnp.uint8,
    "tty_chars":            jnp.uint8,
    "tty_colors":           jnp.int8,
    "tty_cursor":           jnp.uint8,
    "misc":                 jnp.int32,
}

# ---------------------------------------------------------------------------
# Cmap lookup table: TileType -> NLE cmap index (S_* values from defsym.h)
#
# TileType enum (from constants.py):
#   VOID=0, FLOOR=1, CORRIDOR=2, WALL=3, CLOSED_DOOR=4, OPEN_DOOR=5,
#   STAIRCASE_UP=6, STAIRCASE_DOWN=7, WATER=8, LAVA=9, ALTAR=10,
#   FOUNTAIN=11, TRAP=12, HIDDEN_TRAP=13, THRONE=14, GRAVE=15, SHOP_FLOOR=16
#
# defsym.h cmap indices (idx, char, S_* name):
#   0  ' '  S_stone       19 '.'  S_room        22 '#'  S_corr
#   1  '|'  S_vwall       25 '<'  S_upstair     26 '>'  S_dnstair
#   15 '+'  S_vcdoor      13 '-'  S_vodoor      33 '_'  S_altar
#   34 '|'  S_grave       35 '\\' S_throne      37 '{'  S_fountain
#   38 '}'  S_pool        40 '}'  S_lava        49 '^'  S_arrow_trap
# ---------------------------------------------------------------------------

_S_stone    = 0
_S_vwall    = 1
_S_room     = 19
_S_darkroom = 20
_S_corr     = 22
_S_litcorr  = 23
_S_upstair  = 25
_S_dnstair  = 26
_S_altar    = 33
_S_grave    = 34
_S_throne   = 35
_S_fountain = 37
_S_pool     = 38
_S_lava     = 40
_S_trap     = 49   # S_arrow_trap — generic visible trap
_S_vcdoor   = 15   # closed door (vertical)
_S_vodoor   = 13   # open door (vertical)

# Indexed by TileType integer value; must cover indices 0..NUM_TILE_TYPES-1
_TILE_TO_CMAP: jnp.ndarray = jnp.array([
    _S_stone,    # 0  VOID
    _S_room,     # 1  FLOOR
    _S_corr,     # 2  CORRIDOR
    _S_vwall,    # 3  WALL
    _S_vcdoor,   # 4  CLOSED_DOOR
    _S_vodoor,   # 5  OPEN_DOOR
    _S_upstair,  # 6  STAIRCASE_UP
    _S_dnstair,  # 7  STAIRCASE_DOWN
    _S_pool,     # 8  WATER
    _S_lava,     # 9  LAVA
    _S_altar,    # 10 ALTAR
    _S_fountain, # 11 FOUNTAIN
    _S_trap,     # 12 TRAP
    _S_room,     # 13 HIDDEN_TRAP (looks like floor to the player)
    _S_throne,   # 14 THRONE
    _S_grave,    # 15 GRAVE
    _S_room,     # 16 SHOP_FLOOR (same floor tile)
], dtype=jnp.int16)

# ---------------------------------------------------------------------------
# TTY char lookup table: cmap index -> ASCII character
#
# Derived from defsym.h PCHAR entries (idx, char, ...).  Only the indices
# used by _TILE_TO_CMAP need to be correct; others default to ' '.
# The full 49+ entry table is kept for completeness.
# ---------------------------------------------------------------------------

# Build a 64-entry table covering all cmap indices we use (max index is 49).
_CMAP_TO_CHAR: jnp.ndarray = jnp.array([
    ord(' '),  # 0  S_stone
    ord('|'),  # 1  S_vwall
    ord('-'),  # 2  S_hwall
    ord('-'),  # 3  S_tlcorn
    ord('-'),  # 4  S_trcorn
    ord('-'),  # 5  S_blcorn
    ord('-'),  # 6  S_brcorn
    ord('-'),  # 7  S_crwall
    ord('-'),  # 8  S_tuwall
    ord('-'),  # 9  S_tdwall
    ord('|'),  # 10 S_tlwall
    ord('|'),  # 11 S_trwall
    ord('.'),  # 12 S_ndoor
    ord('-'),  # 13 S_vodoor
    ord('|'),  # 14 S_hodoor
    ord('+'),  # 15 S_vcdoor
    ord('+'),  # 16 S_hcdoor
    ord('#'),  # 17 S_bars
    ord('#'),  # 18 S_tree
    ord('.'),  # 19 S_room
    ord('.'),  # 20 S_darkroom
    ord('`'),  # 21 S_engroom
    ord('#'),  # 22 S_corr
    ord('#'),  # 23 S_litcorr
    ord('#'),  # 24 S_engrcorr
    ord('<'),  # 25 S_upstair
    ord('>'),  # 26 S_dnstair
    ord('<'),  # 27 S_upladder
    ord('>'),  # 28 S_dnladder
    ord('<'),  # 29 S_brupstair
    ord('>'),  # 30 S_brdnstair
    ord('<'),  # 31 S_brupladder
    ord('>'),  # 32 S_brdnladder
    ord('_'),  # 33 S_altar
    ord('|'),  # 34 S_grave
    ord('\\'), # 35 S_throne
    ord('{'),  # 36 S_sink
    ord('{'),  # 37 S_fountain
    ord('}'),  # 38 S_pool
    ord('.'),  # 39 S_ice
    ord('}'),  # 40 S_lava
    ord('}'),  # 41 S_lavawall
    ord('.'),  # 42 S_vodbridge
    ord('.'),  # 43 S_hodbridge
    ord('#'),  # 44 S_vcdbridge
    ord('#'),  # 45 S_hcdbridge
    ord(' '),  # 46 S_air
    ord('#'),  # 47 S_cloud
    ord('}'),  # 48 S_water
    ord('^'),  # 49 S_arrow_trap (generic trap char)
    ord('^'),  # 50 S_dart_trap
    ord('^'),  # 51 S_falling_rock_trap
    ord('^'),  # 52 S_squeaky_board
    ord('^'),  # 53 S_bear_trap
    ord('^'),  # 54 S_land_mine
    ord('^'),  # 55 S_rolling_boulder_trap
    ord('^'),  # 56 S_sleeping_gas_trap
    ord('^'),  # 57 S_rust_trap
    ord('^'),  # 58 S_fire_trap
    ord('^'),  # 59 S_pit
    ord('^'),  # 60 S_spiked_pit
    ord('^'),  # 61 S_hole
    ord(' '),  # 62 padding
    ord(' '),  # 63 padding
], dtype=jnp.uint8)

# ---------------------------------------------------------------------------
# ANSI color lookup table: cmap index -> ANSI color (0-15)
#
# Derived from vendor/nethack/include/defsym.h PCHAR2() entries.
# CLR_* values from vendor/nethack/include/color.h:
#   CLR_BLACK=0 CLR_RED=1 CLR_GREEN=2 CLR_BROWN=3 CLR_BLUE=4 CLR_MAGENTA=5
#   CLR_CYAN=6  CLR_GRAY=7 NO_COLOR=8→7  CLR_ORANGE=9 CLR_BRIGHT_GREEN=10
#   CLR_YELLOW=11 CLR_BRIGHT_BLUE=12 CLR_BRIGHT_MAGENTA=13 CLR_BRIGHT_CYAN=14
#   CLR_WHITE=15
# NO_COLOR is treated as CLR_GRAY (7) — same as NLE's fallback rendering.
# ---------------------------------------------------------------------------

_CMAP_TO_COLOR: jnp.ndarray = jnp.array([
    0,   # 0  S_stone       NO_COLOR → black (invisible)
    7,   # 1  S_vwall       CLR_GRAY
    7,   # 2  S_hwall       CLR_GRAY
    7,   # 3  S_tlcorn      CLR_GRAY
    7,   # 4  S_trcorn      CLR_GRAY
    7,   # 5  S_blcorn      CLR_GRAY
    7,   # 6  S_brcorn      CLR_GRAY
    7,   # 7  S_crwall      CLR_GRAY
    7,   # 8  S_tuwall      CLR_GRAY
    7,   # 9  S_tdwall      CLR_GRAY
    7,   # 10 S_tlwall      CLR_GRAY
    7,   # 11 S_trwall      CLR_GRAY
    7,   # 12 S_ndoor       CLR_GRAY
    3,   # 13 S_vodoor      CLR_BROWN
    3,   # 14 S_hodoor      CLR_BROWN
    3,   # 15 S_vcdoor      CLR_BROWN
    3,   # 16 S_hcdoor      CLR_BROWN
    7,   # 17 S_bars        CLR_GRAY (no explicit color in defsym.h; default)
    2,   # 18 S_tree        CLR_GREEN
    7,   # 19 S_room        CLR_GRAY
    0,   # 20 S_darkroom    CLR_BLACK
    7,   # 21 S_engroom     CLR_GRAY (no explicit; default)
    7,   # 22 S_corr        CLR_GRAY
    7,   # 23 S_litcorr     CLR_GRAY
    7,   # 24 S_engrcorr    CLR_GRAY (no explicit; default)
    7,   # 25 S_upstair     CLR_GRAY
    7,   # 26 S_dnstair     CLR_GRAY
    3,   # 27 S_upladder    CLR_BROWN
    3,   # 28 S_dnladder    CLR_BROWN
    11,  # 29 S_brupstair   CLR_YELLOW
    11,  # 30 S_brdnstair   CLR_YELLOW
    11,  # 31 S_brupladder  CLR_YELLOW
    11,  # 32 S_brdnladder  CLR_YELLOW
    7,   # 33 S_altar       CLR_GRAY
    15,  # 34 S_grave       CLR_WHITE
    7,   # 35 S_throne      CLR_GRAY (no explicit in snippet; default gray)
    15,  # 36 S_sink        CLR_WHITE
    12,  # 37 S_fountain    CLR_BRIGHT_BLUE
    4,   # 38 S_pool        CLR_BLUE
    6,   # 39 S_ice         CLR_CYAN
    1,   # 40 S_lava        CLR_RED
    9,   # 41 S_lavawall    CLR_ORANGE
    3,   # 42 S_vodbridge   CLR_BROWN
    3,   # 43 S_hodbridge   CLR_BROWN
    3,   # 44 S_vcdbridge   CLR_BROWN
    3,   # 45 S_hcdbridge   CLR_BROWN
    6,   # 46 S_air         CLR_CYAN
    7,   # 47 S_cloud       CLR_GRAY
    12,  # 48 S_water       CLR_BRIGHT_BLUE
    7,   # 49 S_arrow_trap  CLR_GRAY
    7,   # 50 S_dart_trap   CLR_GRAY
    7,   # 51 S_falling_rock_trap  CLR_GRAY
    3,   # 52 S_squeaky_board      CLR_BROWN
    1,   # 53 S_bear_trap   CLR_RED (no explicit; use red for danger)
    1,   # 54 S_land_mine   CLR_RED
    7,   # 55 S_rolling_boulder_trap  CLR_GRAY
    7,   # 56 S_sleeping_gas_trap  CLR_GRAY (no explicit)
    4,   # 57 S_rust_trap   CLR_BLUE
    9,   # 58 S_fire_trap   CLR_ORANGE
    0,   # 59 S_pit         CLR_BLACK
    0,   # 60 S_spiked_pit  CLR_BLACK
    3,   # 61 S_hole        CLR_BROWN
    3,   # 62 S_trap_door   CLR_BROWN
    5,   # 63 S_teleportation_trap  CLR_MAGENTA
], dtype=jnp.uint8)

# ---------------------------------------------------------------------------
# Inventory letter table: slot index (0-54) -> ASCII byte
#
# NetHack uses a-z (slots 0-25) then A-Z (slots 26-51).
# NLE_INVENTORY_SIZE = 55: the extra 3 slots (52-54) use '@', '[', '$' in NLE
# for quiver/wielded/worn shortcuts; we emit 0 for those (unassigned).
# ---------------------------------------------------------------------------

_INV_LETTERS: jnp.ndarray = jnp.array(
    [ord('a') + i for i in range(26)]          # slots 0-25: a-z
    + [ord('A') + i for i in range(26)]        # slots 26-51: A-Z
    + [0, 0, 0],                               # slots 52-54: unassigned
    dtype=jnp.uint8,
)


# ---------------------------------------------------------------------------
# Static glyph -> color / description lookup tables (Wave 4)
#
# Size: MAX_GLYPH = 5976.  Built once at module load time using Python +
# numpy (no JAX), then exposed as JAX arrays.  Lookup at JIT-time is a single
# fancy index: arr[glyphs] -> per-tile color or per-tile description bytes.
#
# Color sources (vendor/nethack/include/color.h):
#   - Monsters     [GLYPH_MON_OFF .. +NUMMONS)         -> MONSTERS[i].color
#   - Pets         [GLYPH_PET_OFF .. +NUMMONS)         -> MONSTERS[i].color
#   - Invisible    [GLYPH_INVIS_OFF]                   -> CLR_BLUE
#   - Detected     [GLYPH_DETECT_OFF .. +NUMMONS)      -> MONSTERS[i].color
#   - Bodies       [GLYPH_BODY_OFF .. +NUMMONS)        -> CLR_BROWN (corpses)
#   - Ridden       [GLYPH_RIDDEN_OFF .. +NUMMONS)      -> MONSTERS[i].color
#   - Objects      [GLYPH_OBJ_OFF .. +NUM_OBJECTS)     -> OBJECTS[i].color
#   - Cmap         [GLYPH_CMAP_OFF .. +MAXPCHARS)      -> _CMAP_TO_COLOR[idx]
#   - Others (explode/zap/swallow/warning/statue)      -> CLR_GRAY default
#
# Description sources (vendor/nethack/src/pager.c::do_screen_description):
#   - Monsters/pets/detected/ridden -> MONSTERS[i].name
#   - Bodies                        -> "<name> corpse"
#   - Objects                       -> OBJECTS[i].name
#   - Cmap                          -> static terrain name (floor/wall/door/...)
#   - NO_GLYPH / others             -> empty string (zero bytes)
# ---------------------------------------------------------------------------

def _build_glyph_lookups():  # pragma: no cover — runs once at import
    """Build (color, description) lookup tables indexed by glyph id.

    Returns:
        (colors_arr, desc_arr) where
          colors_arr : uint8[MAX_GLYPH]
          desc_arr   : uint8[MAX_GLYPH, 80]
    """
    import numpy as _np
    from Nethax.nethax.constants.monsters import MONSTERS
    from Nethax.nethax.constants.objects import OBJECTS
    from Nethax.nethax.constants.glyphs import (
        GLYPH_MON_OFF as _MON, GLYPH_PET_OFF as _PET, GLYPH_INVIS_OFF as _INV,
        GLYPH_DETECT_OFF as _DET, GLYPH_BODY_OFF as _BOD, GLYPH_RIDDEN_OFF as _RID,
        GLYPH_OBJ_OFF as _OBJ, GLYPH_CMAP_OFF as _CMAP,
        MAX_GLYPH as _MAX,
    )

    colors = _np.zeros((_MAX,), dtype=_np.uint8)
    desc = _np.zeros((_MAX, 80), dtype=_np.uint8)
    # CMAP description table (TileType-ish names, keyed by cmap index)
    # Indices and chosen description strings follow vendor/nethack/src/drawing.c
    # default_showsyms[] / def_oc_syms[] and pager.c::lookat() outputs.
    cmap_desc = {
        0: "dark part of a room",
        1: "wall", 2: "wall", 3: "wall", 4: "wall", 5: "wall",
        6: "wall", 7: "wall", 8: "wall", 9: "wall", 10: "wall", 11: "wall",
        12: "doorway", 13: "open door", 14: "open door",
        15: "closed door", 16: "closed door",
        17: "iron bars", 18: "tree",
        19: "floor", 20: "dark part of a room",
        21: "engraving", 22: "corridor", 23: "lit corridor", 24: "engraving",
        25: "staircase up", 26: "staircase down",
        27: "ladder up", 28: "ladder down",
        29: "branch staircase up", 30: "branch staircase down",
        31: "branch ladder up", 32: "branch ladder down",
        33: "altar", 34: "grave", 35: "throne", 36: "sink",
        37: "fountain", 38: "water", 39: "ice", 40: "molten lava",
        41: "wall of lava",
        42: "drawbridge", 43: "drawbridge", 44: "drawbridge", 45: "drawbridge",
        46: "air", 47: "cloud", 48: "water",
        49: "arrow trap", 50: "dart trap", 51: "falling rock trap",
        52: "squeaky board", 53: "bear trap", 54: "land mine",
        55: "rolling boulder trap", 56: "sleeping gas trap",
        57: "rust trap", 58: "fire trap", 59: "pit", 60: "spiked pit",
        61: "hole", 62: "trap door", 63: "teleportation trap",
    }

    def _bytes_for(s) -> _np.ndarray:
        # Wave 6 parity-fix: allow None (shuffled appearance slots in OBJECTS).
        b = _np.zeros((80,), dtype=_np.uint8)
        if not s:
            return b
        enc = s.encode("ascii", errors="ignore")[:79]  # leave null terminator
        b[:len(enc)] = _np.frombuffer(enc, dtype=_np.uint8)
        return b

    # CLR_* constants from vendor/nethack/include/color.h
    CLR_GRAY = 7
    CLR_BROWN = 3
    CLR_BLUE = 4
    CLR_WHITE = 15

    n_mon = len(MONSTERS)
    n_obj = len(OBJECTS)

    # Monsters & pets & detected & ridden — share name + color
    for i, m in enumerate(MONSTERS):
        c = int(m.color) & 0xFF
        nm = _bytes_for(m.name)
        for base in (_MON, _PET, _DET, _RID):
            g = base + i
            if 0 <= g < _MAX:
                colors[g] = c
                desc[g] = nm
        # Bodies (corpses)
        g = _BOD + i
        if 0 <= g < _MAX:
            colors[g] = CLR_BROWN
            desc[g] = _bytes_for(f"{m.name} corpse")

    # Invisible monster glyph (single id at GLYPH_INVIS_OFF)
    if 0 <= _INV < _MAX:
        colors[_INV] = CLR_BLUE
        desc[_INV] = _bytes_for("invisible creature")

    # Objects
    for i, o in enumerate(OBJECTS):
        g = _OBJ + i
        if 0 <= g < _MAX:
            colors[g] = int(o.color) & 0xFF
            desc[g] = _bytes_for(o.name)

    # Cmap (terrain)
    for cmap_i, txt in cmap_desc.items():
        g = _CMAP + cmap_i
        if 0 <= g < _MAX:
            desc[g] = _bytes_for(txt)
            # color from existing _CMAP_TO_COLOR table (mirror exactly)
            # _CMAP_TO_COLOR is a JAX array of length 64; safe to read at import.
            colors[g] = int(_CMAP_TO_COLOR[cmap_i]) if cmap_i < len(_CMAP_TO_COLOR) else CLR_GRAY

    # Default fill for any remaining slot: CLR_GRAY / empty desc (already zeros).
    # Replace any literal black 0 outside cmap with gray? No — black is valid for
    # unexplored / dark.  Leave zero defaults.

    return (
        jnp.asarray(colors, dtype=jnp.uint8),
        jnp.asarray(desc, dtype=jnp.uint8),
    )


_GLYPH_TO_COLOR, _GLYPH_TO_DESCRIPTION_BYTES = _build_glyph_lookups()

# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


def empty_nle_observation() -> dict[str, jnp.ndarray]:
    """Return a zero-filled NLE-compatible observation dict.

    All arrays have the canonical NLE shapes and dtypes. Useful as a default
    value and for unit tests that need a well-typed observation without a live
    game state.
    """
    return {
        key: jnp.zeros(NLE_OBSERVATION_SHAPES[key], dtype=NLE_OBSERVATION_DTYPES[key])
        for key in NLE_OBSERVATION_KEYS
    }


def build_nle_observation(env_state) -> dict[str, jnp.ndarray]:
    """Build an NLE-compatible observation dict from nethax EnvState.

    Wave 4 status: all 17 NLE observation keys are wired.
      Real (state-projecting): glyphs, chars, colors, specials, blstats,
        message, internal, inv_glyphs, inv_letters, inv_oclasses, inv_strs,
        screen_descriptions, tty_chars, tty_colors, tty_cursor.
      Stub (always zero, by design — nethax has no menus/dialogs):
        program_state[0..2,5], misc.

    Args:
        env_state: nethax EnvState.

    Returns:
        dict mapping each of the 17 NLE_OBSERVATION_KEYS to a jnp.ndarray.
    """
    glyphs = build_glyphs(env_state)
    blstats = build_blstats(env_state)
    message = build_message(env_state)
    tty = build_tty(env_state)

    return {
        "glyphs":              glyphs,
        "chars":               _build_chars(glyphs),
        "colors":              build_colors(env_state),
        "specials":            build_specials(env_state),
        "blstats":             blstats,
        "message":             message,
        "program_state":       build_program_state(env_state),
        "internal":            build_internal(env_state),
        "inv_glyphs":          build_inv_glyphs(env_state),
        "inv_letters":         build_inv_letters(env_state),
        "inv_oclasses":        build_inv_oclasses(env_state),
        "inv_strs":            build_inventory_strings(env_state),
        "screen_descriptions": build_screen_descriptions(env_state),
        "tty_chars":           tty["tty_chars"],
        "tty_colors":          build_tty_colors(env_state),
        "tty_cursor":          tty["tty_cursor"],
        "misc":                build_misc(env_state),
    }


# ---------------------------------------------------------------------------
# Per-field builders (Wave 3)
# ---------------------------------------------------------------------------


def build_colors(env_state) -> jnp.ndarray:
    """Per-tile ANSI color (0-15). Shape (21, 79) uint8.

    Algorithm:
      1. Map terrain tile type through _TILE_TO_CMAP to get cmap index.
      2. Map cmap index through _CMAP_TO_COLOR to get ANSI color.
      3. Unexplored tiles get color 0 (black — invisible).
      4. Player position gets color 15 (CLR_WHITE / CLR_BRIGHT_YELLOW).

    Returns:
        uint8[21, 79]
    """
    branch = jnp.int32(env_state.dungeon.current_branch)
    level_idx = jnp.int32(env_state.dungeon.current_level) - 1

    tile = env_state.terrain[branch, level_idx, :21, :79]
    tile_idx = jnp.clip(tile.astype(jnp.int16), 0, NUM_TILE_TYPES - 1)
    cmap_idx = _TILE_TO_CMAP[tile_idx]  # int16[21,79]
    cmap_clamped = jnp.clip(cmap_idx, 0, len(_CMAP_TO_COLOR) - 1)
    colors = _CMAP_TO_COLOR[cmap_clamped]  # uint8[21,79]

    # Unexplored tiles -> color 0 (black)
    explored = env_state.explored[branch, level_idx, :21, :79]
    colors = jnp.where(explored, colors, jnp.uint8(0))

    # Player tile -> bright yellow (15)
    pr = jnp.int32(env_state.player_pos[0])
    pc = jnp.clip(jnp.int32(env_state.player_pos[1]), 0, 78)
    colors = colors.at[pr, pc].set(jnp.uint8(15))

    return colors


def build_specials(env_state) -> jnp.ndarray:
    """Per-tile special flags. Shape (21, 79) uint8.

    Bit definitions (vendor/nethack/include/display.h MG_* macros — we use a
    compact 6-bit encoding suitable for RL agents):
      bit 0 (0x01): corpse on floor (body glyph or food/"corpse" object)
      bit 1 (0x02): pile (2+ stacks of items on tile, MG_OBJPILE)
      bit 2 (0x04): trap visible (revealed trap on this tile)
      bit 3 (0x08): secret door (currently unset — engine has no secret-door
                                  discovery state yet)
      bit 4 (0x10): invisible monster sensed (currently unset — no invis flag)
      bit 5 (0x20): object present (any ground item)

    Reads:
      - state.traps.trap_type / .revealed   for trap bit
      - state.ground_items.category         for object / pile / corpse bits
      - state.ground_items.type_id          for corpse type identification

    Returns:
        uint8[21, 79]
    """
    branch = jnp.int32(env_state.dungeon.current_branch)
    level_idx = jnp.int32(env_state.dungeon.current_level) - 1

    # Trap layer: traps.trap_type is [n_branches*n_levels, H, W].
    # Linear index follows make_traps_state convention: branch*max_levels + level.
    # However the TrapState was built with num_levels=b*l (single flat axis),
    # so use that linear index directly.
    n_levels = env_state.terrain.shape[1]
    flat_lv = branch * jnp.int32(n_levels) + level_idx
    trap_revealed = env_state.traps.revealed[flat_lv, :21, :79]  # bool[21,79]
    trap_type = env_state.traps.trap_type[flat_lv, :21, :79]     # int8[21,79]
    has_trap = (trap_type != 0) & trap_revealed                  # bool[21,79]

    # Ground items: category[branch, level, row, col, stack] (int8)
    # stack dim is MAX_GROUND_STACK = 8; non-zero means item present.
    gi_cat = env_state.ground_items.category[branch, level_idx, :21, :79, :]  # int8[21,79,8]
    gi_typ = env_state.ground_items.type_id[branch, level_idx, :21, :79, :]   # int16[21,79,8]

    occupied = gi_cat != 0                                       # bool[21,79,8]
    has_object = jnp.any(occupied, axis=-1)                      # bool[21,79]
    stack_count = jnp.sum(occupied.astype(jnp.int32), axis=-1)   # int32[21,79]
    has_pile = stack_count >= 2                                  # bool[21,79]

    # Corpse: category == FOOD_CLASS (7) and type_id == CORPSE_OBJ_TYPE_ID (260).
    # Per vendor/nethack/include/objects.h FOOD("corpse", ...), corpse is the
    # canonical food entry; in our OBJECTS table that lands at index 260.
    from Nethax.nethax.subsystems.inventory import ItemCategory as _IC
    FOOD_CLASS = jnp.int8(int(_IC.FOOD))
    CORPSE_TYPE_ID = jnp.int16(260)
    is_corpse_stack = (gi_cat == FOOD_CLASS) & (gi_typ == CORPSE_TYPE_ID)
    has_corpse = jnp.any(is_corpse_stack, axis=-1)               # bool[21,79]

    # Build the 6-bit specials byte.
    specials = (
        has_corpse.astype(jnp.uint8) * jnp.uint8(0x01)
        | has_pile.astype(jnp.uint8) * jnp.uint8(0x02)
        | has_trap.astype(jnp.uint8) * jnp.uint8(0x04)
        # bit 3 (secret door): always 0 in Wave 4 — no secret-door state yet
        # bit 4 (invisible mon): always 0 in Wave 4 — no invis monster state yet
        | has_object.astype(jnp.uint8) * jnp.uint8(0x20)
    )
    return specials.astype(jnp.uint8)


def build_internal(env_state) -> jnp.ndarray:
    """Internal NLE state vector. Shape (9,) int32.

    Field layout per vendor/nle/win/rl/winrl.cc:278-287
    (called from NetHackRL::update_observation):
        [0] deepest_lev_reached   — max level depth ever visited
        [1] in_yn_function        — always 0 in nethax (no y/n prompts)
        [2] in_getlin             — always 0 in nethax (no text-input prompts)
        [3] xwaitingforspace      — always 0 in nethax (no --More-- pauses)
        [4] stairs_down           — 1 if player is standing on a down-stair
        [5] 0 (legacy core RNG seed slot)
        [6] 0 (legacy display RNG seed slot)
        [7] uhunger               — raw hunger counter (0..2000 typical)
        [8] urexp                 — total experience score

    Returns:
        int32[9]
    """
    from Nethax.nethax.constants import TileType
    branch = jnp.int32(env_state.dungeon.current_branch)
    level_idx = jnp.int32(env_state.dungeon.current_level) - 1
    pr = jnp.clip(jnp.int32(env_state.player_pos[0]), 0, 20)
    pc = jnp.clip(jnp.int32(env_state.player_pos[1]), 0, 78)
    cur_tile = env_state.terrain[branch, level_idx, pr, pc]
    stairs_down = (cur_tile == jnp.int8(TileType.STAIRCASE_DOWN)).astype(jnp.int32)

    cur_level = jnp.int32(env_state.dungeon.current_level)

    out = jnp.zeros((9,), dtype=jnp.int32)
    out = out.at[0].set(cur_level)                                 # deepest known
    out = out.at[1].set(jnp.int32(0))                              # in_yn_function
    out = out.at[2].set(jnp.int32(0))                              # in_getlin
    out = out.at[3].set(jnp.int32(0))                              # xwaitingforspace
    out = out.at[4].set(stairs_down)                               # stairs_down
    out = out.at[5].set(jnp.int32(0))                              # legacy core seed
    out = out.at[6].set(jnp.int32(0))                              # legacy disp seed
    # NLE puts raw hunger counter (uhunger) here; nethax stores hunger as an
    # enum state in status.hunger_state — pass that through (close-enough for
    # agent parity, since uhunger is mainly a derived signal).
    out = out.at[7].set(jnp.int32(env_state.status.hunger_state))
    out = out.at[8].set(jnp.int32(env_state.scoring.score))
    return out


def build_screen_descriptions(env_state) -> jnp.ndarray:
    """Per-tile description bytes. Shape (21, 79, 80) uint8.

    For each glyph at (r, c), looks up an ASCII description string and packs
    the first 80 bytes (null-padded) into screen_descriptions[r, c, :].

    Implementation: single fancy-index into the static _GLYPH_TO_DESCRIPTION_BYTES
    lookup table built at module load.  Unexplored tiles (NO_GLYPH) yield zero
    bytes naturally because the lookup table is zero-filled at NO_GLYPH.

    Reference: vendor/nethack/src/pager.c::do_screen_description and
               vendor/nle/win/rl/winrl.cc::store_screen_description.

    Returns:
        uint8[21, 79, 80]
    """
    glyphs = build_glyphs(env_state)  # int16[21,79]
    # Clamp index into [0, MAX_GLYPH-1] before fancy indexing.
    g_idx = jnp.clip(glyphs.astype(jnp.int32), 0, _GLYPH_TO_DESCRIPTION_BYTES.shape[0] - 1)
    return _GLYPH_TO_DESCRIPTION_BYTES[g_idx]


def build_program_state(env_state) -> jnp.ndarray:
    """NLE program_state vector. Shape (6,) int32.

    Fields per vendor/nle/win/rl/winrl.cc::update_program_state (around the
    program_state writes that mirror NetHack's program_state struct):
        [0] gameover           — 1 if really_done (post-death menu)
        [1] panicking          — 1 if NetHack panicked
        [2] exiting            — 1 if exiting normally
        [3] in_moveloop        — 1 during normal turn loop
        [4] something_worth_saving — 1 once the game has begun
        [5] 0 reserved

    nethax has no menus or panic states; we report in_moveloop=1 once
    state.timestep > 0 (game started), something_worth_saving=1 likewise.

    Returns:
        int32[6]
    """
    started = (env_state.timestep > 0).astype(jnp.int32)
    out = jnp.zeros((6,), dtype=jnp.int32)
    out = out.at[3].set(started)
    out = out.at[4].set(started)
    return out


def build_misc(env_state) -> jnp.ndarray:
    """NLE misc vector. Shape (3,) int32.

    Fields per vendor/nle/win/rl/winrl.cc:289-293:
        [0] in_yn_function     — always 0 in nethax
        [1] in_getlin          — always 0 in nethax
        [2] xwaitingforspace   — always 0 in nethax
    """
    return jnp.zeros((3,), dtype=jnp.int32)


def build_tty_colors(env_state) -> jnp.ndarray:
    """Terminal color grid. Shape (24, 80) int8.

    Layout mirrors build_tty():
      Row 0:     message line — white (7)
      Rows 1-21: map area — colors from build_colors(), padded to 80 cols
      Row 22-23: status lines — white (7)

    Returns:
        int8[24, 80]
    """
    tty_colors = jnp.zeros((24, 80), dtype=jnp.int8)

    # Row 0: message line -> white (7)
    tty_colors = tty_colors.at[0, :].set(jnp.int8(7))

    # Rows 1-21: map colors, padded to 80 cols
    map_colors = build_colors(env_state)  # uint8[21,79]
    pad_col = jnp.zeros((21, 1), dtype=jnp.int8)
    map_colors_80 = jnp.concatenate(
        [map_colors.astype(jnp.int8), pad_col], axis=1
    )  # int8[21,80]
    tty_colors = tty_colors.at[1:22, :].set(map_colors_80)

    # Rows 22-23: status lines -> white (7)
    tty_colors = tty_colors.at[22, :].set(jnp.int8(7))
    tty_colors = tty_colors.at[23, :].set(jnp.int8(7))

    return tty_colors


def build_inv_glyphs(env_state) -> jnp.ndarray:
    """Glyph for each inventory slot. Shape (55,) int16.

    Formula: glyph = GLYPH_OBJ_OFF + type_id for occupied slots.
    Empty slots (category == 0) yield glyph 0.

    Wave 3 note: InventoryState.items is a scalar stub (single Item, not a
    52-element array). Slot 0 reflects the stub item when category != 0;
    all other slots are 0.  Full per-slot wiring requires the Wave 3 inventory
    array migration.

    Returns:
        int16[55]
    """
    inv = jnp.zeros((55,), dtype=jnp.int16)
    items = env_state.inventory.items  # batched Item: each field shape (52,)
    cat = items.category.astype(jnp.int16)
    typ = items.type_id.astype(jnp.int16)
    glyphs_52 = jnp.where(cat != 0, jnp.int16(GLYPH_OBJ_OFF) + typ, jnp.int16(0))
    inv = inv.at[:52].set(glyphs_52)
    return inv


def build_inv_letters(env_state) -> jnp.ndarray:
    """ASCII letter for each inventory slot. Shape (55,) uint8.

    Mapping: slot 0 -> ord('a'), ..., slot 25 -> ord('z'),
             slot 26 -> ord('A'), ..., slot 51 -> ord('Z'),
             slots 52-54 -> 0.

    Returns:
        uint8[55]
    """
    return _INV_LETTERS


def build_inv_oclasses(env_state) -> jnp.ndarray:
    """ObjectClass enum value for each inventory slot. Shape (55,) uint8.

    Wave 3 note: InventoryState.items is a scalar stub. Slot 0 reflects
    the stub item's category; all other slots are 0.

    Returns:
        uint8[55]
    """
    inv = jnp.zeros((55,), dtype=jnp.uint8)
    cat_52 = env_state.inventory.items.category.astype(jnp.uint8)
    inv = inv.at[:52].set(cat_52)
    return inv


# ---------------------------------------------------------------------------
# Per-field builders (Wave 2)
# ---------------------------------------------------------------------------


def build_blstats(env_state) -> jnp.ndarray:
    """Project EnvState player/dungeon fields into the NLE-canonical 27-vector.

    All indices sourced from Nethax.nethax.constants.blstats BL_* constants,
    which mirror vendor/nle/include/nleobs.h:17-43.

    JIT-compatible: only jnp.zeros + .at[].set() calls.

    Returns:
        int64[27]
    """
    result = jnp.zeros((27,), dtype=jnp.int64)

    # Position (col, row)
    result = result.at[BL_X].set(jnp.int64(env_state.player_pos[1]))
    result = result.at[BL_Y].set(jnp.int64(env_state.player_pos[0]))

    # Strength: NLE stores the clamped display value [3..25] at BL_STR25
    # and the raw internal value [3..125] at BL_STR125.
    # NetHack botl.c: display shows str//5 when str > 25 (i.e. exceptional str).
    result = result.at[BL_STR25].set(
        jnp.int64(jnp.minimum(env_state.player_str, jnp.int16(25)))
    )
    result = result.at[BL_STR125].set(jnp.int64(env_state.player_str))

    result = result.at[BL_DEX].set(jnp.int64(env_state.player_dex))
    result = result.at[BL_CON].set(jnp.int64(env_state.player_con))
    result = result.at[BL_INT].set(jnp.int64(env_state.player_int))
    result = result.at[BL_WIS].set(jnp.int64(env_state.player_wis))
    result = result.at[BL_CHA].set(jnp.int64(env_state.player_cha))

    # Score
    result = result.at[BL_SCORE].set(jnp.int64(env_state.scoring.score))

    # HP
    result = result.at[BL_HP].set(jnp.int64(env_state.player_hp))
    result = result.at[BL_HPMAX].set(jnp.int64(env_state.player_hp_max))

    # Dungeon depth (current_level is 1-based, NLE BL_DEPTH is also 1-based)
    result = result.at[BL_DEPTH].set(jnp.int64(env_state.dungeon.current_level))

    # Gold
    result = result.at[BL_GOLD].set(jnp.int64(env_state.player_gold))

    # Power (spell energy)
    result = result.at[BL_ENE].set(jnp.int64(env_state.player_pw))
    result = result.at[BL_ENEMAX].set(jnp.int64(env_state.player_pw_max))

    # Armor class: from state.player_ac (Wave 3+)
    result = result.at[BL_AC].set(jnp.int64(env_state.player_ac))

    # Monster level / hit dice: 0 for player (BL_HD only nonzero when polymorphed)
    result = result.at[BL_HD].set(jnp.int64(0))

    # Experience level and points
    result = result.at[BL_XP].set(jnp.int64(env_state.player_xl))
    result = result.at[BL_EXP].set(jnp.int64(env_state.player_xp))

    # Game time (turn counter)
    result = result.at[BL_TIME].set(jnp.int64(env_state.timestep))

    # Hunger state (int8 enum from status_effects.HungerState)
    result = result.at[BL_HUNGER].set(jnp.int64(env_state.status.hunger_state))

    # Encumbrance (int8 enum from status_effects.Encumbrance)
    result = result.at[BL_CAP].set(jnp.int64(env_state.status.encumbrance))

    # Dungeon branch number and level
    result = result.at[BL_DNUM].set(jnp.int64(env_state.dungeon.current_branch))
    result = result.at[BL_DLEVEL].set(jnp.int64(env_state.dungeon.current_level))

    # Condition bitmask: placeholder 0 (Wave 3 wires status flags)
    result = result.at[BL_CONDITION].set(jnp.int64(0))

    # Alignment (-1=chaotic, 0=neutral, 1=lawful — matches botl.c)
    result = result.at[BL_ALIGN].set(jnp.int64(env_state.player_align))

    return result


def build_glyphs(env_state) -> jnp.ndarray:
    """Map the current level's terrain to NLE glyph IDs.

    Algorithm (all JAX, JIT-compatible):
    1. Slice current level terrain from env_state.terrain[branch, level-1, :21, :79].
    2. Map each tile type through _TILE_TO_CMAP to get a cmap symbol index.
    3. Add GLYPH_CMAP_OFF to get NLE glyph ID.
    4. Where explored==False, substitute NO_GLYPH (cast to int16).
    5. Overlay the player glyph (GLYPH_MON_OFF + 0) at player_pos.

    Returns:
        int16[21, 79]
    """
    branch = jnp.int32(env_state.dungeon.current_branch)
    level_idx = jnp.int32(env_state.dungeon.current_level) - 1  # 0-based

    # terrain shape: [N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H=21, MAP_W=80]
    # NLE glyphs shape: [21, 79] — trim last column
    level_terrain = env_state.terrain[branch, level_idx, :21, :79]   # int8[21,79]

    # Clamp tile index to valid _TILE_TO_CMAP range (in case of corrupt data)
    tile_idx = jnp.clip(level_terrain.astype(jnp.int16), 0, NUM_TILE_TYPES - 1)
    cmap_idx = _TILE_TO_CMAP[tile_idx]                               # int16[21,79]

    # Terrain glyph IDs
    terrain_glyphs = (cmap_idx + jnp.int16(GLYPH_CMAP_OFF)).astype(jnp.int16)

    # Explored mask: unexplored tiles -> NO_GLYPH
    explored = env_state.explored[branch, level_idx, :21, :79]       # bool[21,79]
    no_glyph_val = jnp.int16(NO_GLYPH & 0xFFFF)                      # NO_GLYPH as int16
    glyphs = jnp.where(explored, terrain_glyphs, no_glyph_val)

    # Overlay player at player_pos (row, col).
    # The player's display glyph is their race's monster type (human=256,
    # elf=260, dwarf=43, gnome=162, orc=71 in our MONSTERS table), unless
    # they are polymorphed (then it's the polymorph form).
    #
    # Citation: vendor/nethack/src/display.c::display_self / show_glyph
    #   uses `u.umonnum` (the player's current monster type).
    player_row = jnp.int32(env_state.player_pos[0])
    player_col = jnp.int32(env_state.player_pos[1])
    # Clamp col to [0,78] since glyphs is 79 wide, terrain is 80 wide.
    player_col_clamped = jnp.clip(player_col, 0, 78)

    # Race -> base monster index in MONSTERS.  Order matches Race enum:
    #   HUMAN=0, ELF=1, DWARF=2, GNOME=3, ORC=4
    _RACE_TO_MON_IDX = jnp.array([256, 260, 43, 162, 71], dtype=jnp.int32)
    race_idx = jnp.clip(jnp.int32(env_state.player_race), 0, 4)
    base_mon = _RACE_TO_MON_IDX[race_idx]

    # If polymorphed, use the polymorph form instead.
    is_poly = env_state.polymorph.is_polymorphed
    poly_form = jnp.int32(env_state.polymorph.current_form_idx)
    mon_idx = jnp.where(is_poly, poly_form, base_mon)

    player_glyph = (jnp.int32(GLYPH_MON_OFF) + mon_idx).astype(jnp.int16)
    glyphs = glyphs.at[player_row, player_col_clamped].set(player_glyph)

    return glyphs


def build_message(env_state) -> jnp.ndarray:
    """Return the current message buffer as a 256-byte uint8 array.

    Wave 2: directly returns env_state.messages.message_buffer (already 256
    uint8). Pads with zeros or truncates if somehow mismatched.

    Returns:
        uint8[256]
    """
    buf = env_state.messages.message_buffer  # uint8[256]
    # Ensure shape is exactly (256,) — pad with zeros if shorter, trim if longer
    buf = buf[:256]
    pad_len = 256 - buf.shape[0]
    buf = jnp.concatenate([buf, jnp.zeros((pad_len,), dtype=jnp.uint8)]) if pad_len > 0 else buf
    return buf.astype(jnp.uint8)


def build_tty(env_state) -> dict[str, jnp.ndarray]:
    """Render the 24x80 TTY terminal grid.

    Layout:
      Row 0:     message line (env_state.messages.message_buffer[:80])
      Rows 1-21: map area (glyphs converted to ASCII chars)
      Row 22:    status line 1 (St/Dx/Co/In/Wi/Ch  Dlvl:n  HP  Pw  AC  XP)
      Row 23:    status line 2 (Dlvl: n  T: n)

    Colors are zeros in Wave 2 (Wave 3 implements color tables).
    Cursor is at player position offset by +1 row (message row 0 occupies row 0).

    Returns:
        dict with keys:
          tty_chars  : uint8[24, 80]
          tty_colors : int8[24, 80]   (zeros in Wave 2)
          tty_cursor : uint8[2]       (row, col)

    JIT-compatible: all operations use jnp.where / at[].set().
    """
    tty = jnp.zeros((24, 80), dtype=jnp.uint8)

    # --- Row 0: message line ---
    msg = env_state.messages.message_buffer[:80].astype(jnp.uint8)
    tty = tty.at[0, :].set(msg)

    # --- Rows 1-21: map area ---
    glyphs = build_glyphs(env_state)   # int16[21,79]

    # Convert glyph -> cmap index.  For terrain glyphs:
    #   glyph = GLYPH_CMAP_OFF + cmap_idx  =>  cmap_idx = glyph - GLYPH_CMAP_OFF
    # For NO_GLYPH (unexplored) we show ' ' (space = 32).
    # For the player glyph (GLYPH_MON_OFF = 0) we show '@'.
    no_glyph_val = jnp.int16(NO_GLYPH & 0xFFFF)
    player_glyph_val = jnp.int16(GLYPH_MON_OFF)

    # Compute cmap_idx for terrain glyphs; clamp to valid table range.
    cmap_idx = (glyphs.astype(jnp.int32) - GLYPH_CMAP_OFF).astype(jnp.int32)
    cmap_idx_clamped = jnp.clip(cmap_idx, 0, len(_CMAP_TO_CHAR) - 1)
    terrain_chars = _CMAP_TO_CHAR[cmap_idx_clamped]                 # uint8[21,79]

    # Player '@' overlaid where glyph == GLYPH_MON_OFF
    is_player = glyphs == player_glyph_val
    # Unexplored ' '
    is_unexplored = glyphs == no_glyph_val

    map_chars = jnp.where(is_player, jnp.uint8(ord('@')), terrain_chars)
    map_chars = jnp.where(is_unexplored, jnp.uint8(ord(' ')), map_chars)

    # Pad 79-wide map to 80 columns with spaces
    pad_col = jnp.full((21, 1), ord(' '), dtype=jnp.uint8)
    map_chars_80 = jnp.concatenate([map_chars, pad_col], axis=1)    # uint8[21,80]
    tty = tty.at[1:22, :].set(map_chars_80)

    # --- Rows 22-23: status lines ---
    # Row 22: attribute abbreviations — written as static bytes built from blstats.
    # We fill these rows with spaces (already zero = '\x00' which NLE treats as
    # null; use ord(' ')=32 for legibility).
    blstats = build_blstats(env_state)

    # Build a 40-char status line 1 in Python at trace time would break JIT;
    # instead we leave rows 22-23 as zeros (null bytes).  NLE agents typically
    # read blstats directly; tty status rows are cosmetic.
    # TODO Wave 3: encode status line bytes via lax.dynamic_slice / digit tables.

    # --- Cursor: row = player_row + 1 (offset for message line), col = player_col ---
    player_row = jnp.uint8(jnp.clip(env_state.player_pos[0], 0, 20) + 1)
    player_col = jnp.uint8(jnp.clip(env_state.player_pos[1], 0, 78))

    return {
        "tty_chars":  tty,
        "tty_colors": build_tty_colors(env_state),
        "tty_cursor": jnp.array([player_row, player_col], dtype=jnp.uint8),
    }


# ---------------------------------------------------------------------------
# Derived map (chars) from glyphs
# ---------------------------------------------------------------------------


def _build_chars(glyphs: jnp.ndarray) -> jnp.ndarray:
    """Derive the chars (21x79 uint8) map from the glyph grid.

    chars[r,c] is the ASCII character corresponding to glyphs[r,c]:
      - Monster / pet / ridden / detected / statue → MonsterSymbol char
      - Object                                     → ObjectClass char
      - Cmap (terrain)                             → defsym char
      - Body / corpse                              → '%'
      - Invis monster                              → 'I'
      - Warning                                    → '0'-'5'
      - Unexplored (NO_GLYPH)                      → ' '

    Citation: vendor/nethack/include/monsym.h + defsym.h + vendor/nethack/
              src/objects.c::def_oc_syms.

    Returns:
        uint8[21, 79]
    """
    g = glyphs.astype(jnp.int32)
    no_glyph_val = jnp.int32(NO_GLYPH)

    # ---- per-glyph category resolution (jit-safe via lookup tables) ----
    cmap_idx = jnp.clip(g - GLYPH_CMAP_OFF, 0, len(_CMAP_TO_CHAR) - 1)
    terrain_chars = _CMAP_TO_CHAR[cmap_idx]

    # Monster / pet / ridden / detected → use MONSTERS[idx].symbol → char
    mon_idx_raw = jnp.where(g < GLYPH_PET_OFF, g - GLYPH_MON_OFF,
                  jnp.where(g < GLYPH_INVIS_OFF, g - GLYPH_PET_OFF,
                  jnp.where(g < GLYPH_BODY_OFF, g - GLYPH_DETECT_OFF,
                  jnp.where(g < GLYPH_OBJ_OFF, g - GLYPH_RIDDEN_OFF, 0))))
    mon_idx = jnp.clip(mon_idx_raw, 0, _MON_IDX_TO_CHAR.shape[0] - 1)
    mon_chars = _MON_IDX_TO_CHAR[mon_idx]

    # Object → OBJECTS[idx].class_ → def_oc_syms char
    obj_idx = jnp.clip(g - GLYPH_OBJ_OFF, 0, _OBJ_IDX_TO_CHAR.shape[0] - 1)
    obj_chars = _OBJ_IDX_TO_CHAR[obj_idx]

    # Statue glyph → '`'  (vendor S_grave-ish; just shows as backtick)
    statue_char = jnp.uint8(ord('`'))

    # Body / corpse → '%'
    body_char = jnp.uint8(ord('%'))

    # Invisible monster → 'I'
    invis_char = jnp.uint8(ord('I'))

    # Warning → '0'-'5' (6 levels)
    warn_idx = jnp.clip(g - GLYPH_WARNING_OFF, 0, 5)
    warn_chars = (jnp.uint8(ord('0')) + warn_idx).astype(jnp.uint8)

    # ---- category masks ----
    is_mon       = (g >= GLYPH_MON_OFF)     & (g < GLYPH_INVIS_OFF)
    is_invis     = (g >= GLYPH_INVIS_OFF)   & (g < GLYPH_DETECT_OFF)
    is_detect    = (g >= GLYPH_DETECT_OFF)  & (g < GLYPH_BODY_OFF)
    is_body      = (g >= GLYPH_BODY_OFF)    & (g < GLYPH_RIDDEN_OFF)
    is_ridden    = (g >= GLYPH_RIDDEN_OFF)  & (g < GLYPH_OBJ_OFF)
    is_obj       = (g >= GLYPH_OBJ_OFF)     & (g < GLYPH_CMAP_OFF)
    is_cmap      = (g >= GLYPH_CMAP_OFF)    & (g < GLYPH_EXPLODE_OFF)
    is_warn      = (g >= GLYPH_WARNING_OFF) & (g < GLYPH_STATUE_OFF)
    is_statue    = (g >= GLYPH_STATUE_OFF)  & (g < MAX_GLYPH)
    is_unexplored = (g == no_glyph_val)

    # Default to space (covers EXPLODE / ZAP / SWALLOW / unmapped).
    chars = jnp.full(g.shape, jnp.uint8(ord(' ')))
    chars = jnp.where(is_cmap,    terrain_chars, chars)
    chars = jnp.where(is_obj,     obj_chars,     chars)
    chars = jnp.where(is_mon | is_detect | is_ridden, mon_chars, chars)
    chars = jnp.where(is_body,    body_char,     chars)
    chars = jnp.where(is_invis,   invis_char,    chars)
    chars = jnp.where(is_warn,    warn_chars,    chars)
    chars = jnp.where(is_statue,  statue_char,   chars)
    chars = jnp.where(is_unexplored, jnp.uint8(ord(' ')), chars)
    return chars


# ---------------------------------------------------------------------------
# Inventory stubs (Wave 3)
# ---------------------------------------------------------------------------


def build_inventory_strings(env_state) -> jnp.ndarray:
    """Wave 3: render all 55 inventory slots as objnam-style ASCII strings.

    Delegates to Nethax.nethax.obs.inv_strs.build_inv_strs, which produces
    a uint8[55, 80] array with NLE-canonical per-slot strings.

    Returns:
        uint8[55, 80]
    """
    from Nethax.nethax.obs.inv_strs import build_inv_strs
    return build_inv_strs(env_state)
