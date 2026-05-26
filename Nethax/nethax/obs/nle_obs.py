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
# Hallucination hcolor pool — vendor/nethack/src/do_name.c:1461
#
# When hallucinating, object glyphs are replaced with random "junk" objects
# that differ in color from the true object.  We build a static pool of
# NUM_OBJECTS object indices (one canonical representative per CLR_* value,
# plus enough filler to reach 32 entries) so the per-tile hash can pick a
# replacement index without requiring a dynamic table at JIT trace time.
#
# The pool only needs to be large enough that modular selection produces a
# visually varied result.  We pick the first object index for each distinct
# color (up to 16 colors), then repeat the full sweep to pad to 32 entries.
# ---------------------------------------------------------------------------

def _build_hcolor_pool() -> jnp.ndarray:
    """Build a uint16 array of object indices for hallucination color scramble."""
    import numpy as _np
    from Nethax.nethax.constants.objects import OBJECTS
    seen: dict[int, int] = {}
    for i, o in enumerate(OBJECTS):
        if o is None:
            continue
        c = int(o.color) & 0xFF
        if c not in seen:
            seen[c] = i
    # Collect in color order, then pad to 32 by cycling.
    pool = list(seen.values())
    while len(pool) < 32:
        pool.extend(pool)
    pool = pool[:32]
    return jnp.array(pool, dtype=jnp.int32)


_HCOLOR_POOL: jnp.ndarray = _build_hcolor_pool()   # int32[32]
_HCOLOR_POOL_SIZE: int = 32

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

    Bit definitions match NLE's bundled NetHack header
    (vendor/nle/include/hack.h:77-84) which is what NLE-trained agents
    observe — NOT vendor/nethack/include/display.h (whose layout shifts
    every bit because it adds a MG_HERO=0x01 entry at the bottom).

      0x01  MG_CORPSE   — corpse on floor
      0x02  MG_INVIS    — invisible monster sensed
      0x04  MG_DETECT   — detected monster
      0x08  MG_PET      — pet
      0x10  MG_RIDDEN   — ridden monster
      0x20  MG_STATUE   — statue
      0x40  MG_OBJPILE  — 2+ object stacks on this tile
      0x80  MG_BW_LAVA  — black&white lava highlight (or BW_ICE/BW_SINK/BW_ENGR alias)

    Note: NLE has NO "MG_HERO" bit (the hero is conveyed separately via
    blstats[0,1] = (x, y)).  Nethax previously emitted MG_HERO=0x01 which
    silently re-tagged every tile as containing a corpse from NLE's
    perspective.

    nethax-current support:
      MG_CORPSE   : food category + corpse type_id on ground stack
      MG_OBJPILE  : 2+ ground stacks
      MG_PET      : 1 if at least one pet exists at this tile (state.pets)
      MG_INVIS/MG_DETECT/MG_RIDDEN/MG_STATUE/MG_BW_LAVA: unset for now.

    Returns:
        uint8[21, 79]
    """
    branch = jnp.int32(env_state.dungeon.current_branch)
    level_idx = jnp.int32(env_state.dungeon.current_level) - 1

    # Ground items: category[branch, level, row, col, stack] (int8)
    # stack dim is MAX_GROUND_STACK = 8; non-zero means item present.
    gi_cat = env_state.ground_items.category[branch, level_idx, :21, :79, :]
    gi_typ = env_state.ground_items.type_id[branch, level_idx, :21, :79, :]

    occupied = gi_cat != 0
    stack_count = jnp.sum(occupied.astype(jnp.int32), axis=-1)
    has_objpile = stack_count >= 2

    # Corpse: category == FOOD_CLASS (7) and type_id == CORPSE_OBJ_TYPE_ID (260).
    # Per vendor/nethack/include/objects.h FOOD("corpse", ...), corpse is the
    # canonical food entry; in our OBJECTS table that lands at index 260.
    from Nethax.nethax.subsystems.inventory import ItemCategory as _IC
    FOOD_CLASS = jnp.int8(int(_IC.FOOD))
    CORPSE_TYPE_ID = jnp.int16(260)
    is_corpse_stack = (gi_cat == FOOD_CLASS) & (gi_typ == CORPSE_TYPE_ID)
    has_corpse = jnp.any(is_corpse_stack, axis=-1)

    # MG_PET: any tile occupied by a tame monster (pets state).
    # The pets subsystem layout exposes positions per pet; fall back to all
    # zeros if not available.  We resolve dynamically to avoid import cycles.
    has_pet = _pet_mask(env_state, branch, level_idx)

    # Parity-mode-aware MG_* layout (Nethax/nethax/parity_mode.py).
    # NLE mode (default):     vendor/nle/include/hack.h:77-84
    # NetHack 3.7 mode:       vendor/nethack/include/display.h:995-1009
    from Nethax.nethax.parity_mode import mg_bits, is_nethack_mode
    bits = mg_bits()

    specials = (
          has_corpse.astype(jnp.uint8) * jnp.uint8(bits.MG_CORPSE)
        | has_pet.astype(jnp.uint8)    * jnp.uint8(bits.MG_PET)
        | has_objpile.astype(jnp.uint8) * jnp.uint8(bits.MG_OBJPILE)
    )
    # NetHack-mode only: set MG_HERO bit at player position.
    # NLE conveys hero position via blstats[0,1] instead.
    if is_nethack_mode():
        pr = jnp.clip(jnp.int32(env_state.player_pos[0]), 0, 20)
        pc = jnp.clip(jnp.int32(env_state.player_pos[1]), 0, 78)
        hero_mask = jnp.zeros((21, 79), dtype=jnp.uint8).at[pr, pc].set(
            jnp.uint8(bits.MG_HERO)
        )
        specials = specials | hero_mask
    return specials.astype(jnp.uint8)


def _pet_mask(env_state, branch, level_idx) -> jnp.ndarray:
    """Return a bool[21,79] mask of tiles occupied by a pet (tame monster).

    Resilient to missing pet state — returns all False if pets aren't tracked.
    """
    pets = getattr(env_state, "pets", None)
    if pets is None:
        return jnp.zeros((21, 79), dtype=jnp.bool_)
    # Common layout: pets.active is bool[N_PETS]; pets.pos is int16[N_PETS,2];
    # pets.branch/pets.level identify which level each pet is on.
    active = getattr(pets, "active", None)
    pos = getattr(pets, "pos", None)
    pet_branch = getattr(pets, "branch", None)
    pet_level = getattr(pets, "level", None)
    if active is None or pos is None:
        return jnp.zeros((21, 79), dtype=jnp.bool_)
    mask = jnp.zeros((21, 79), dtype=jnp.bool_)
    n_pets = pos.shape[0]
    for i in range(n_pets):
        r = jnp.clip(jnp.int32(pos[i, 0]), 0, 20)
        c = jnp.clip(jnp.int32(pos[i, 1]), 0, 78)
        on_level = jnp.bool_(True)
        if pet_branch is not None:
            on_level = on_level & (jnp.int32(pet_branch[i]) == branch)
        if pet_level is not None:
            on_level = on_level & (jnp.int32(pet_level[i]) == (level_idx + 1))
        is_here = jnp.bool_(active[i]) & on_level
        mask = mask.at[r, c].set(mask[r, c] | is_here)
    return mask


def build_internal(env_state) -> jnp.ndarray:
    """Internal NLE state vector. Shape (9,) int32.

    Field layout per vendor/nle/win/rl/winrl.cc:278-287
    (called from NetHackRL::update_observation):
        [0] deepest_lev_reached   — max level depth ever visited (Wave 8: vendor parity)
        [1] in_yn_function        — always 0 in nethax (no y/n prompts)
        [2] in_getlin             — always 0 in nethax (no text-input prompts)
        [3] xwaitingforspace      — always 0 in nethax (no --More-- pauses)
        [4] stairs_down           — 1 if player is standing on a down-stair
        [5] 0 (legacy core RNG seed slot)
        [6] 0 (legacy display RNG seed slot)
        [7] uhunger               — raw nutrition counter (Wave 8: vendor parity)
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

    # Vendor parity (Wave 8): deepest_lev_reached comes from
    # scoring.deepest_level (record_deepest_level is called from action_dispatch
    # whenever the player advances to a previously-unvisited level).  Falls
    # back to current_level when the tracker hasn't been initialized.
    deepest = jnp.maximum(
        jnp.int32(env_state.scoring.deepest_level),
        jnp.int32(env_state.dungeon.current_level),
    )

    out = jnp.zeros((9,), dtype=jnp.int32)
    out = out.at[0].set(deepest)                                   # deepest_lev_reached
    out = out.at[1].set(jnp.int32(0))                              # in_yn_function
    out = out.at[2].set(jnp.int32(0))                              # in_getlin
    out = out.at[3].set(jnp.int32(0))                              # xwaitingforspace
    out = out.at[4].set(stairs_down)                               # stairs_down
    out = out.at[5].set(jnp.int32(0))                              # legacy core seed
    out = out.at[6].set(jnp.int32(0))                              # legacy disp seed
    # Vendor parity (Wave 8): raw u.uhunger nutrition counter (0..2000+).
    # winrl.cc:285 — obs->internal[7] = u.uhunger.
    out = out.at[7].set(jnp.int32(env_state.status.nutrition))
    # Vendor parity (wave17i): winrl.cc:286-287 — obs->internal[8] = u.urexp,
    # the running 64-bit "real experience" accumulator (you.h:399).  This is
    # distinct from blstats[BL_EXP] (= u.uexp) and from botl_score() which
    # populates blstats[BL_SCORE].  Previously this slot collapsed onto
    # scoring.score; now it reads the dedicated player_urexp field added in
    # wave16a.
    out = out.at[8].set(jnp.int32(env_state.player_urexp))
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

    Fields per vendor/nle/win/rl/winrl.cc::fill_obs (lines 262-268):
        [0] gameover               — 1 when game is over (state.done)
        [1] panicking              — always 0 in nethax
        [2] exiting                — 1 when game is over (state.done)
        [3] in_moveloop            — always 1 after reset
        [4] in_impossible          — always 0 in nethax
        [5] something_worth_saving — always 1 after reset

    nethax has no panic states; in_moveloop=1 and something_worth_saving=1
    from the first turn.  gameover and exiting both mirror state.done.

    Returns:
        int32[6]
    """
    done = jnp.int32(env_state.done)
    out = jnp.zeros((6,), dtype=jnp.int32)
    out = out.at[0].set(done)            # gameover
    out = out.at[2].set(done)            # exiting
    out = out.at[3].set(jnp.int32(1))   # in_moveloop
    out = out.at[5].set(jnp.int32(1))   # something_worth_saving
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

    Vendor parity (vendor/nle/win/rl/winrl.cc::observation_glyphs ~line 379):
      Occupied slots  : glyph = GLYPH_OBJ_OFF + otyp
      Empty slots     : NO_GLYPH (MAX_GLYPH = 5976)

    Returns:
        int16[55]
    """
    inv = jnp.full((55,), jnp.int16(NO_GLYPH & 0xFFFF), dtype=jnp.int16)
    items = env_state.inventory.items
    cat = items.category.astype(jnp.int16)
    typ = items.type_id.astype(jnp.int16)
    glyphs_52 = jnp.where(
        cat != 0,
        jnp.int16(GLYPH_OBJ_OFF) + typ,
        jnp.int16(NO_GLYPH & 0xFFFF),
    )
    inv = inv.at[:52].set(glyphs_52)
    return inv


def build_inv_letters(env_state) -> jnp.ndarray:
    """ASCII letter for each inventory slot. Shape (55,) uint8.

    Vendor parity (vendor/nle/win/rl/winrl.cc::observation_letters ~line 396):
      Occupied slots  : obj->invlet  (i.e. 'a'..'z', 'A'..'Z')
      Empty slots     : 0

    Returns:
        uint8[55]
    """
    cat = env_state.inventory.items.category
    occupied = (cat != 0)
    letters_52 = jnp.where(occupied, _INV_LETTERS[:52], jnp.uint8(0))
    inv = jnp.zeros((55,), dtype=jnp.uint8)
    inv = inv.at[:52].set(letters_52)
    return inv


# ObjectClass enum max value + 1; matches vendor MAXOCLASSES in objclass.h.
_MAXOCLASSES: int = 18


def build_inv_oclasses(env_state) -> jnp.ndarray:
    """ObjectClass enum value for each inventory slot. Shape (55,) uint8.

    Vendor parity (vendor/nle/win/rl/winrl.cc::observation_oclasses ~line 413):
      Occupied slots  : obj->oclass  (ObjectClass enum)
      Empty slots     : MAXOCLASSES (=18, the past-end sentinel)

    Returns:
        uint8[55]
    """
    inv = jnp.full((55,), jnp.uint8(_MAXOCLASSES), dtype=jnp.uint8)
    cat = env_state.inventory.items.category
    occupied = (cat != 0)
    oclass_52 = jnp.where(occupied,
                          cat.astype(jnp.uint8),
                          jnp.uint8(_MAXOCLASSES))
    inv = inv.at[:52].set(oclass_52)
    return inv


# ---------------------------------------------------------------------------
# Status line rendering — vendor parity for tty_chars rows 22-23.
#
# vendor/nethack/src/botl.c::do_statusline1() (lines 48-98) format:
#   "<Name> the <Title>      St:%s Dx:%-1d Co:%-1d In:%-1d Wi:%-1d Ch:%-1d  <Align>"
# vendor/nethack/src/botl.c::do_statusline2() (lines 100-249) format:
#   "Dlvl:%d $:%-2ld HP:%d(%d) Pw:%d(%d) AC:%-2d Xp:%d T:%ld <conds>"
#
# These helpers fill 80-byte rows with the same field order using a JIT-friendly
# digit table (no Python string formatting at trace time).
# ---------------------------------------------------------------------------

# Static role rank titles — port of vendor/nethack/src/role.c::roles[].rank[].
# Index is rank 0..8.  rank_of() uses xlev_to_rank() to derive rank from xlevel.
_ROLE_RANK_TITLES: tuple[tuple[str, ...], ...] = (
    # Archeologist
    ("Digger", "Field Worker", "Investigator", "Exhumer", "Excavator",
     "Spelunker", "Speleologist", "Collector", "Curator"),
    # Barbarian
    ("Plunderer", "Pillager", "Bandit", "Brigand", "Raider",
     "Reaver", "Slayer", "Chieftain", "Conqueror"),
    # Caveman
    ("Troglodyte", "Aborigine", "Wanderer", "Vagrant", "Wayfarer",
     "Roamer", "Nomad", "Rover", "Pioneer"),
    # Healer
    ("Rhizotomist", "Empiric", "Embalmer", "Dresser", "Medicus ossium",
     "Herbalist", "Magister", "Physician", "Chirurgeon"),
    # Knight
    ("Gallant", "Esquire", "Bachelor", "Sergeant", "Knight",
     "Banneret", "Chevalier", "Seignieur", "Paladin"),
    # Monk
    ("Candidate", "Novice", "Initiate", "Student of Stones", "Student of Waters",
     "Student of Metals", "Student of Winds", "Student of Fire", "Master"),
    # Priest
    ("Aspirant", "Acolyte", "Adept", "Priest", "Curate",
     "Canon", "Lama", "Patriarch", "High Priest"),
    # Ranger
    ("Tenderfoot", "Lookout", "Trailblazer", "Reconnoiterer", "Scout",
     "Arbalester", "Archer", "Sharpshooter", "Marksman"),
    # Rogue
    ("Footpad", "Cutpurse", "Rogue", "Pilferer", "Robber",
     "Burglar", "Filcher", "Magsman", "Thief"),
    # Samurai
    ("Hatamoto", "Ronin", "Ninja", "Joshu", "Ryoshu",
     "Kokushu", "Daimyo", "Kuge", "Shogun"),
    # Tourist
    ("Rambler", "Sightseer", "Excursionist", "Peregrinator", "Traveler",
     "Journeyer", "Voyager", "Explorer", "Adventurer"),
    # Valkyrie
    ("Stripling", "Skirmisher", "Fighter", "Man-at-arms", "Warrior",
     "Swashbuckler", "Hero", "Champion", "Lord"),
    # Wizard
    ("Evoker", "Conjurer", "Thaumaturge", "Magician", "Enchanter",
     "Sorcerer", "Necromancer", "Wizard", "Mage"),
)


def _xlev_to_rank(xlev: int) -> int:
    """Vendor botl.c::xlev_to_rank — convert xlevel to rank index (0..8)."""
    if xlev <= 2:
        return 0
    if xlev <= 30:
        return (xlev + 2) // 4
    return 8


def role_rank_title(role_idx: int, xlevel: int) -> str:
    """Return the rank title string for a (role, xlevel) pair.

    Mirrors vendor/nethack/src/botl.c::rank_of (lines 331-358).
    Female-variant titles are not modeled (always returns the male form).
    """
    if role_idx < 0 or role_idx >= len(_ROLE_RANK_TITLES):
        return "Player"
    return _ROLE_RANK_TITLES[role_idx][_xlev_to_rank(int(xlevel))]


# ---------------------------------------------------------------------------
# JIT-friendly digit emission for status rows.
#
# We avoid long chains of `.at[].set()` (which XLA's algebraic simplifier
# can spend many passes folding).  Instead, build a single uint8[80] array
# via concatenation of small pieces.  All string literals are precomputed
# at module load.
# ---------------------------------------------------------------------------

_DIGITS = jnp.array([ord('0') + i for i in range(10)], dtype=jnp.uint8)


def _str_to_bytes(s: str) -> jnp.ndarray:
    """Precompute a uint8 byte-array for a literal string (module load only)."""
    return jnp.array([ord(c) for c in s], dtype=jnp.uint8)


def _uint_to_bytes(value, width):
    """Return a uint8[width] array of right-aligned digit bytes for `value`.

    Leading zeros become spaces (except the units digit, which is always shown).
    """
    v = jnp.maximum(jnp.int32(value), jnp.int32(0))
    digit_arrs = []
    for _ in range(width):
        digit_arrs.append((v % 10).astype(jnp.int32))
        v = v // 10
    # digit_arrs is LSB-first; reverse to MSB-first.
    digit_arrs = digit_arrs[::-1]

    # Build a boolean "seen_nonzero" prefix for each position.
    nonzero_flags = [d != 0 for d in digit_arrs]
    # cumulative OR
    cum = []
    acc = jnp.bool_(False)
    for f in nonzero_flags:
        acc = acc | f
        cum.append(acc)

    chars = []
    for i, (d, seen) in enumerate(zip(digit_arrs, cum)):
        is_units = (i == width - 1)
        ch = jnp.where(seen | is_units,
                       _DIGITS[d],
                       jnp.uint8(ord(' ')))
        chars.append(ch.reshape(1))
    return jnp.concatenate(chars).astype(jnp.uint8)


# Precomputed static byte sequences for status rows (module-load).
# Legacy "Player the Adventurer" header — retained as a fallback when the
# (role, xlevel) -> rank lookup yields no usable title.  Cite: vendor
# botl.c::do_statusline1 line 51 ("%s the %s", plname, rank_of(...)).
_S_NAME_PREFIX  = _str_to_bytes("Player the Adventurer")


# ---------------------------------------------------------------------------
# Wave17i parity: precomputed (role, rank) -> "Player the <Title>" byte rows.
# Each row is padded to 27 bytes (the header column the status line allocates
# before the stats group "St:NN Dx:NN ...").
# Cite: vendor/nethack/src/botl.c::do_statusline1 — header is
#   sprintf(newbot, "%s the %s", plname, rank_of(u.ulevel, ..., u.ufemale)).
# ---------------------------------------------------------------------------

_HEADER_PAD_W = 27   # status row 1 reserves cols 0..26 for "<Name> the <Title>".

def _build_role_header_table() -> jnp.ndarray:
    """Return uint8[N_ROLES, N_RANKS, _HEADER_PAD_W] of header bytes.

    For role r, xlevel-rank k the row is "Player the <title>" left-justified,
    right-padded with spaces to _HEADER_PAD_W bytes.
    """
    n_roles = len(_ROLE_RANK_TITLES)
    n_ranks = 9  # vendor rank_of returns 0..8
    rows = []
    for r in range(n_roles):
        for k in range(n_ranks):
            title = _ROLE_RANK_TITLES[r][k]
            s = f"Player the {title}"[:_HEADER_PAD_W]
            s = s.ljust(_HEADER_PAD_W)
            rows.append([ord(c) & 0xFF for c in s])
    arr = jnp.array(rows, dtype=jnp.uint8)
    return arr.reshape(n_roles, n_ranks, _HEADER_PAD_W)

# Default fallback row (matches the legacy hardcoded header).
_DEFAULT_HEADER_ROW = jnp.array(
    [ord(c) & 0xFF for c in "Player the Adventurer".ljust(_HEADER_PAD_W)],
    dtype=jnp.uint8,
)

_ROLE_HEADER_TABLE = _build_role_header_table()
_N_HEADER_ROLES, _N_HEADER_RANKS, _ = _ROLE_HEADER_TABLE.shape


def _xlev_to_rank_jax(xlev) -> jnp.ndarray:
    """JIT-friendly vendor rank_of(u.ulevel) — botl.c::xlev_to_rank.

    Maps experience level → rank index in [0..8]:
        xlev <= 2     → 0
        2 < xlev <=30 → (xlev + 2) // 4
        xlev > 30     → 8
    """
    xl = jnp.int32(xlev)
    base = (xl + jnp.int32(2)) // jnp.int32(4)
    rank = jnp.where(xl <= 2, jnp.int32(0),
                     jnp.where(xl <= 30, base, jnp.int32(8)))
    return jnp.clip(rank, 0, _N_HEADER_RANKS - 1)


def _role_header_bytes(role_idx, xlevel) -> jnp.ndarray:
    """Return uint8[_HEADER_PAD_W] for the ``(role, xlevel)`` header row.

    Falls back to the legacy "Player the Adventurer" row when role_idx is
    outside the table range (e.g. uninitialised state).
    """
    r = jnp.int32(role_idx)
    in_range = (r >= 0) & (r < _N_HEADER_ROLES)
    safe_r = jnp.where(in_range, r, jnp.int32(0))
    rank = _xlev_to_rank_jax(xlevel)
    candidate = _ROLE_HEADER_TABLE[safe_r, rank]
    return jnp.where(in_range, candidate, _DEFAULT_HEADER_ROW)
_S_ST           = _str_to_bytes("St:")
_S_SP_DX        = _str_to_bytes(" Dx:")
_S_SP_CO        = _str_to_bytes(" Co:")
_S_SP_IN        = _str_to_bytes(" In:")
_S_SP_WI        = _str_to_bytes(" Wi:")
_S_SP_CH        = _str_to_bytes(" Ch:")
_S_ALIGN_LAW    = _str_to_bytes("  Lawful ")
_S_ALIGN_NEU    = _str_to_bytes("  Neutral")
_S_ALIGN_CHA    = _str_to_bytes("  Chaotic")

_S_DLVL         = _str_to_bytes("Dlvl:")
_S_SP_DOLLAR    = _str_to_bytes(" $:")
_S_SP_HP        = _str_to_bytes(" HP:")
_S_OPEN         = _str_to_bytes("(")
_S_CLOSE_SP_PW  = _str_to_bytes(") Pw:")
_S_CLOSE_SP_AC  = _str_to_bytes(") AC:")
_S_SP_XP        = _str_to_bytes(" Xp:")
_S_SP_T         = _str_to_bytes(" T:")
_S_PAD80        = jnp.full((80,), ord(' '), dtype=jnp.uint8)


def _pad_to(arr: jnp.ndarray, n: int) -> jnp.ndarray:
    """Right-pad `arr` with spaces up to `n` bytes (truncate if too long)."""
    if arr.shape[0] >= n:
        return arr[:n]
    return jnp.concatenate([arr, jnp.full((n - arr.shape[0],), ord(' '), dtype=jnp.uint8)])


def _build_status_row1(env_state, blstats) -> jnp.ndarray:
    """Render row 22 of tty_chars — vendor do_statusline1 format.

    Format: "Player the Adventurer    St:NN Dx:NN Co:NN In:NN Wi:NN Ch:NN  <Align>"

    Citation: vendor/nethack/src/botl.c::do_statusline1 (lines 48-98).
    """
    al = blstats[BL_ALIGN]
    # Select alignment bytes (length 9).
    is_chaotic = (al == jnp.int64(-1))
    is_neutral = (al == jnp.int64(0))
    align_bytes = jnp.where(
        is_chaotic, _S_ALIGN_CHA,
        jnp.where(is_neutral, _S_ALIGN_NEU, _S_ALIGN_LAW),
    )

    # Header: "Player the <RankTitle>" left-padded to col 27.
    # Wave17i parity: use the role-aware rank title (botl.c::rank_of) instead
    # of the legacy hardcoded "Adventurer" suffix.  ``player_role`` is the
    # Role enum int8; ``player_xl`` (u.ulevel) drives xlev_to_rank.
    header = _role_header_bytes(env_state.player_role, env_state.player_xl)

    # Stats fragment.  Strength uses vendor botl.c::get_strength_str format
    # (see Nethax.nethax.obs.strength_format) — fixed 5-byte field so
    # downstream column offsets stay deterministic across the [3..125] range.
    from Nethax.nethax.obs.strength_format import render_strength_bytes
    parts = [
        header,
        _S_ST,                                       # 3
        render_strength_bytes(blstats[BL_STR125]),
        _S_SP_DX,
        _uint_to_bytes(blstats[BL_DEX], 2),
        _S_SP_CO,
        _uint_to_bytes(blstats[BL_CON], 2),
        _S_SP_IN,
        _uint_to_bytes(blstats[BL_INT], 2),
        _S_SP_WI,
        _uint_to_bytes(blstats[BL_WIS], 2),
        _S_SP_CH,
        _uint_to_bytes(blstats[BL_CHA], 2),
        align_bytes,
    ]
    row = jnp.concatenate(parts)
    return _pad_to(row, 80)


# ---------------------------------------------------------------------------
# Status-line condition keywords  (vendor botl.c::do_statusline2 ~line 220-249)
#
# Vendor appends a space-separated tail of keywords for every active player
# status: " Conf", " Stun", " Hallu", " Blind", " FoodPois", " Ill", " Slime",
# " Strngl", " Burdened", " Stressed", " Strained", " Overtaxed", " Overloaded".
#
# We pack each keyword as a fixed-width uint8 chunk, then mask each chunk to
# spaces if the corresponding flag is inactive, then concatenate.  Pad to 80.
# ---------------------------------------------------------------------------

def _kw_bytes(s: str, width: int) -> jnp.ndarray:
    arr = list(s.encode("ascii")) + [ord(' ')] * (width - len(s))
    return jnp.array(arr[:width], dtype=jnp.uint8)


# Keywords with leading space; widths chosen to be just long enough.
_KW_CONF      = _kw_bytes(" Conf",       5)
_KW_STUN      = _kw_bytes(" Stun",       5)
_KW_HALLU     = _kw_bytes(" Hallu",      6)
_KW_BLIND     = _kw_bytes(" Blind",      6)
_KW_FOODPOIS  = _kw_bytes(" FoodPois",   9)
_KW_ILL       = _kw_bytes(" Ill",        4)
_KW_SLIME     = _kw_bytes(" Slime",      6)
_KW_STRNGL    = _kw_bytes(" Strngl",     7)
_KW_BURDENED  = _kw_bytes(" Burdened",   9)
_KW_STRESSED  = _kw_bytes(" Stressed",   9)
_KW_STRAINED  = _kw_bytes(" Strained",   9)
_KW_OVERTAX   = _kw_bytes(" Overtaxed", 10)
_KW_OVERLOAD  = _kw_bytes(" Overloaded",11)


def build_status_conditions(env_state) -> jnp.ndarray:
    """Vendor-format status-condition keyword tail as a 32-byte uint8 vector.

    Reads ``env_state.status.timed_statuses`` (TimedStatus enum indices) and
    ``env_state.status.encumbrance`` (Encumbrance enum value), masks each
    keyword's bytes to spaces when inactive, concatenates and pads to 32
    bytes.  Designed to be appended to the row-23 tail in build_tty.

    Citation: vendor/nethack/src/botl.c::do_statusline2 (lines ~220-249)
              where each ``Strcpy(nb = eos(nb), " <KW>")`` is gated by the
              corresponding ``HConfusion``, ``Blind``, ``Stunned`` ... flag.
    """
    # TimedStatus indices (must match status_effects.TimedStatus enum order).
    ts = env_state.status.timed_statuses                         # int32[N]
    is_stun       = ts[0]  > 0   # STUNNED
    is_conf       = ts[1]  > 0   # CONFUSION
    is_blind      = ts[2]  > 0   # BLIND
    is_sick       = ts[4]  > 0   # SICK
    is_strngl     = ts[6]  > 0   # STRANGLED
    is_slime      = ts[9]  > 0   # SLIMED
    is_hallu      = ts[10] > 0   # HALLUCINATION

    # SICK splits into FoodPois vs Ill based on status.sick_kind.
    sick_kind = jnp.int32(env_state.status.sick_kind)
    is_foodpois = is_sick & (sick_kind == 1)
    is_ill      = is_sick & (sick_kind == 2)

    # Encumbrance (Encumbrance enum: 0=UN, 1=BURDENED, 2=STRESSED, 3=STRAINED,
    # 4=OVERTAXED, 5=OVERLOADED).  See status_effects.Encumbrance.
    enc = jnp.int32(env_state.status.encumbrance)
    is_burdened  = enc == 1
    is_stressed  = enc == 2
    is_strained  = enc == 3
    is_overtaxed = enc == 4
    is_overload  = enc == 5

    def _mask(kw: jnp.ndarray, active: jnp.ndarray) -> jnp.ndarray:
        """Return kw bytes if active else all spaces, same length as kw."""
        spaces = jnp.full(kw.shape, jnp.uint8(ord(' ')), dtype=jnp.uint8)
        return jnp.where(active, kw, spaces)

    chunks = [
        _mask(_KW_CONF,     is_conf),
        _mask(_KW_STUN,     is_stun),
        _mask(_KW_HALLU,    is_hallu),
        _mask(_KW_BLIND,    is_blind),
        _mask(_KW_FOODPOIS, is_foodpois),
        _mask(_KW_ILL,      is_ill),
        _mask(_KW_SLIME,    is_slime),
        _mask(_KW_STRNGL,   is_strngl),
        _mask(_KW_BURDENED, is_burdened),
        _mask(_KW_STRESSED, is_stressed),
        _mask(_KW_STRAINED, is_strained),
        _mask(_KW_OVERTAX,  is_overtaxed),
        _mask(_KW_OVERLOAD, is_overload),
    ]
    return jnp.concatenate(chunks)


def _build_status_row2(env_state, blstats) -> jnp.ndarray:
    """Render row 23 of tty_chars — vendor do_statusline2 format.

    Format: "Dlvl:N $:M HP:H(Hmax) Pw:P(Pmax) AC:A Xp:X T:T <conditions>"

    The ``<conditions>`` suffix is the keyword tail produced by
    ``build_status_conditions`` (e.g., " Conf Blind Burdened").

    Citation: vendor/nethack/src/botl.c::do_statusline2 (lines 100-249)
    where ``Strcpy(nb = eos(nb), " Conf")`` etc. append each active
    condition keyword to the status line tail.
    """
    ac = blstats[BL_AC]
    neg = ac < 0
    abs_ac = jnp.abs(ac)
    sign_byte = jnp.where(neg, jnp.uint8(ord('-')), jnp.uint8(ord(' '))).reshape(1)

    parts = [
        _S_DLVL,
        _uint_to_bytes(blstats[BL_DEPTH], 2),
        _S_SP_DOLLAR,
        _uint_to_bytes(blstats[BL_GOLD], 4),
        _S_SP_HP,
        _uint_to_bytes(blstats[BL_HP], 4),
        _S_OPEN,
        _uint_to_bytes(blstats[BL_HPMAX], 4),
        _S_CLOSE_SP_PW,
        _uint_to_bytes(blstats[BL_ENE], 3),
        _S_OPEN,
        _uint_to_bytes(blstats[BL_ENEMAX], 3),
        _S_CLOSE_SP_AC,
        sign_byte,
        _uint_to_bytes(abs_ac, 2),
        _S_SP_XP,
        _uint_to_bytes(blstats[BL_XP], 2),
        _S_SP_T,
        _uint_to_bytes(blstats[BL_TIME], 5),
        build_status_conditions(env_state),
    ]
    row = jnp.concatenate(parts)
    return _pad_to(row, 80)


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
    # Effective STR honors the Gauntlets-of-Power cap (acurr branch at
    # vendor/nethack/src/attrib.c:1213-1215 forces 125 when uarmg==GoP).
    from Nethax.nethax.subsystems.armor_effects import compute_effective_str
    eff_str = compute_effective_str(env_state)
    result = result.at[BL_STR25].set(jnp.int64(jnp.minimum(eff_str, jnp.int32(25))))
    result = result.at[BL_STR125].set(jnp.int64(eff_str))

    result = result.at[BL_DEX].set(jnp.int64(env_state.player_dex))
    result = result.at[BL_CON].set(jnp.int64(env_state.player_con))
    result = result.at[BL_INT].set(jnp.int64(env_state.player_int))
    result = result.at[BL_WIS].set(jnp.int64(env_state.player_wis))
    result = result.at[BL_CHA].set(jnp.int64(env_state.player_cha))

    # Score — vendor BL_SCORE = botl_score() (winrl.cc:544).  In nethax we
    # track the *displayed* running score in ``scoring.score`` (which is
    # what botl.c::botl_score() returns mid-game) until the game ends; at
    # end-of-game ``scoring.final_score`` is populated by
    # compute_final_score (end.c:1325-1352).  Use the cached final value
    # when nonzero so post-death observations report the canonical score
    # the vendor would surface in ``end.c``; otherwise fall back to the
    # running counter.  This matches winrl.cc:544's behaviour (which reads
    # botl_score() — a function that switches to the final tally once
    # really_done() has fired).
    _final = jnp.int64(env_state.scoring.final_score)
    _running = jnp.int64(env_state.scoring.score)
    result = result.at[BL_SCORE].set(jnp.where(_final > 0, _final, _running))

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

    # Condition bitmask — vendor botl.c::do_statusline2 + botl.h:107-134.
    # Wave 8 parity: derive BL_MASK_* bits from StatusState.timed_statuses
    # countdowns.  A status is "active" when its remaining-turns counter > 0.
    # Cite: vendor/nethack/include/botl.h:107-134 (BL_MASK_* constants),
    #       vendor/nethack/src/botl.c::do_statusline2 (condition rendering).
    from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS
    from Nethax.nethax.constants.blstats import (
        BL_MASK_BLIND, BL_MASK_CONF, BL_MASK_DEAF, BL_MASK_FOODPOIS,
        BL_MASK_HALLU, BL_MASK_SLIME, BL_MASK_STONE, BL_MASK_STRNGL,
        BL_MASK_STUN, BL_MASK_TERMILL, BL_MASK_PARLYZ, BL_MASK_LEV,
        BL_MASK_FLY,
    )
    ts = env_state.status.timed_statuses
    cond = jnp.int64(0)
    cond = cond | jnp.where(ts[int(_TS.BLIND)]      > 0, jnp.int64(BL_MASK_BLIND),    jnp.int64(0))
    cond = cond | jnp.where(ts[int(_TS.CONFUSION)]  > 0, jnp.int64(BL_MASK_CONF),     jnp.int64(0))
    cond = cond | jnp.where(ts[int(_TS.STUNNED)]    > 0, jnp.int64(BL_MASK_STUN),     jnp.int64(0))
    cond = cond | jnp.where(ts[int(_TS.HALLUCINATION)] > 0, jnp.int64(BL_MASK_HALLU), jnp.int64(0))
    cond = cond | jnp.where(ts[int(_TS.DEAF)]       > 0, jnp.int64(BL_MASK_DEAF),     jnp.int64(0))
    cond = cond | jnp.where(ts[int(_TS.STONED)]     > 0, jnp.int64(BL_MASK_STONE),    jnp.int64(0))
    cond = cond | jnp.where(ts[int(_TS.SLIMED)]     > 0, jnp.int64(BL_MASK_SLIME),    jnp.int64(0))
    cond = cond | jnp.where(ts[int(_TS.STRANGLED)]  > 0, jnp.int64(BL_MASK_STRNGL),   jnp.int64(0))
    cond = cond | jnp.where(ts[int(_TS.FROZEN)]     > 0, jnp.int64(BL_MASK_PARLYZ),   jnp.int64(0))
    cond = cond | jnp.where(ts[int(_TS.LEVITATION_TMP)] > 0, jnp.int64(BL_MASK_LEV),  jnp.int64(0))
    cond = cond | jnp.where(ts[int(_TS.FLYING_TMP)] > 0, jnp.int64(BL_MASK_FLY),      jnp.int64(0))
    # SICK with food-poisoning kind -> FOODPOIS; chronic illness -> TERMILL.
    sick_active = ts[int(_TS.SICK)] > 0
    is_foodpois = env_state.status.sick_kind == jnp.int8(1)
    cond = cond | jnp.where(sick_active & is_foodpois, jnp.int64(BL_MASK_FOODPOIS), jnp.int64(0))
    cond = cond | jnp.where(sick_active & ~is_foodpois, jnp.int64(BL_MASK_TERMILL), jnp.int64(0))
    result = result.at[BL_CONDITION].set(cond)

    # Alignment.  Vendor (botl.c::status_bl_init) uses u.ualign.type:
    #     A_LAWFUL  =  1
    #     A_NEUTRAL =  0
    #     A_CHAOTIC = -1
    # Our state stores 0=lawful, 1=neutral, 2=chaotic.  Map via 1 - x.
    result = result.at[BL_ALIGN].set(
        jnp.int64(1) - jnp.int64(env_state.player_align)
    )

    return result


# ---------------------------------------------------------------------------
# Wall-angle pass for build_glyphs.
#
# Vendor citation: vendor/nethack/src/display.c::wall_angle (decl line 151,
# body 3512+) and set_wall_state / xy_set_wall_state (3275-3354) — those
# compute per-tile wall mode from neighbours and the cell type so the
# renderer can pick S_vwall / S_hwall / S_tlcorn / S_trcorn / S_blcorn /
# S_brcorn / S_crwall / S_tuwall / S_tdwall / S_tlwall / S_trwall.
#
# Cmap-index mapping (vendor/nethack/include/defsym.h):
#    1 S_vwall   '|'      vertical
#    2 S_hwall   '-'      horizontal
#    3 S_tlcorn  '-'      top-left corner of room   (opens S and E)
#    4 S_trcorn  '-'      top-right corner of room  (opens S and W)
#    5 S_blcorn  '-'      bottom-left corner        (opens N and E)
#    6 S_brcorn  '-'      bottom-right corner       (opens N and W)
#    7 S_crwall  '-'      crossing (+)              (all four neighbours)
#    8 S_tuwall  '-'      T pointing up   (open up, walls on S/E/W)
#    9 S_tdwall  '-'      T pointing down (open down, walls on N/E/W)
#   10 S_tlwall  '|'      T pointing left (open left, walls on N/S/E)
#   11 S_trwall  '|'      T pointing right(open right, walls on N/S/W)
#
# Our JAX terrain map only stores a single TileType.WALL (=3) and lacks
# vendor's pre-baked typ split (VWALL/HWALL/T*WALL/CROSSWALL).  We instead
# derive the variant at render time by counting wall neighbours in each of
# the four cardinal directions.  Doors are treated as continuations of the
# wall they are embedded in (a wall flowing through a door is the vendor
# default: walls and doors share an `IS_ROCK` / `IS_DOOR` continuation in
# check_pos()).
# ---------------------------------------------------------------------------

# Pattern bits: N=1, S=2, E=4, W=8 -> 16-entry table of cmap indices.
def _build_wall_angle_table():
    import numpy as _np
    S_vwall, S_hwall = 1, 2
    S_tlcorn, S_trcorn, S_blcorn, S_brcorn = 3, 4, 5, 6
    S_crwall = 7
    S_tuwall, S_tdwall, S_tlwall, S_trwall = 8, 9, 10, 11
    N, S, E, W = 1, 2, 4, 8

    tbl = _np.zeros((16,), dtype=_np.int16)
    # No neighbours / single neighbour -> default vwall or hwall.
    tbl[0]              = S_vwall                    # isolated stub
    tbl[N]              = S_vwall                    # N only
    tbl[S]              = S_vwall                    # S only
    tbl[E]              = S_hwall                    # E only
    tbl[W]              = S_hwall                    # W only
    tbl[N | S]          = S_vwall                    # vertical run
    tbl[E | W]          = S_hwall                    # horizontal run
    # Two-neighbour L-corners: name reflects where the corner SITS, the
    # opening directions are the two wall-neighbours.
    tbl[S | E]          = S_tlcorn                   # top-left of room: opens down + right
    tbl[S | W]          = S_trcorn                   # top-right of room
    tbl[N | E]          = S_blcorn                   # bottom-left of room
    tbl[N | W]          = S_brcorn                   # bottom-right of room
    # Three-neighbour T-junctions: name reflects the OPEN direction.
    tbl[N | S | E]      = S_tlwall                   # only W is open
    tbl[N | S | W]      = S_trwall                   # only E is open
    tbl[N | E | W]      = S_tdwall                   # only S is open (T points down)
    tbl[S | E | W]      = S_tuwall                   # only N is open (T points up)
    # Four-neighbour cross
    tbl[N | S | E | W]  = S_crwall
    return jnp.array(tbl, dtype=jnp.int16)


_WALL_ANGLE_TABLE: jnp.ndarray = _build_wall_angle_table()


def _apply_wall_angle(display_terrain: jnp.ndarray,
                      cmap_idx: jnp.ndarray) -> jnp.ndarray:
    """Replace generic S_vwall on WALL tiles with the correct corner variant.

    Args:
        display_terrain: int8/int16[21, 79] terrain TileType per cell.
        cmap_idx:        int16[21, 79] current cmap indices from _TILE_TO_CMAP.

    Returns:
        int16[21, 79] cmap indices with wall variants resolved.
    """
    from Nethax.nethax.constants.tiles import TileType

    t = display_terrain.astype(jnp.int16)
    WALL = jnp.int16(int(TileType.WALL))
    CLOSED = jnp.int16(int(TileType.CLOSED_DOOR))
    OPEN = jnp.int16(int(TileType.OPEN_DOOR))

    # A neighbour counts as a wall-continuation when it is WALL or a door.
    # Vendor check_pos() treats walls + doors as connected segments.
    is_wallish = (t == WALL) | (t == CLOSED) | (t == OPEN)

    H, W = is_wallish.shape

    # Pad with False (out-of-bounds = open) so edge cells behave like rooms.
    zero_row = jnp.zeros((1, W), dtype=jnp.bool_)
    zero_col = jnp.zeros((H, 1), dtype=jnp.bool_)

    n = jnp.concatenate([zero_row, is_wallish[:-1, :]], axis=0)   # north neighbour
    s = jnp.concatenate([is_wallish[1:, :], zero_row], axis=0)    # south neighbour
    w = jnp.concatenate([zero_col, is_wallish[:, :-1]], axis=1)   # west neighbour
    e = jnp.concatenate([is_wallish[:, 1:], zero_col], axis=1)    # east neighbour

    pattern = (n.astype(jnp.int16)
               | (s.astype(jnp.int16) << jnp.int16(1))
               | (e.astype(jnp.int16) << jnp.int16(2))
               | (w.astype(jnp.int16) << jnp.int16(3)))           # int16[21,79], 0..15

    wall_variant = _WALL_ANGLE_TABLE[pattern]                     # int16[21,79]

    # Only rewrite cells that are themselves WALL.  Door tiles keep their
    # CLOSED_DOOR / OPEN_DOOR cmap (which is _NOT_ a wall variant).
    is_wall_cell = (t == WALL)
    return jnp.where(is_wall_cell, wall_variant, cmap_idx)


def build_glyphs(env_state) -> jnp.ndarray:
    """Map the current level's terrain to NLE glyph IDs.

    Algorithm (all JAX, JIT-compatible):
    1. Slice current level terrain from env_state.terrain[branch, level-1, :21, :79].
    2. Map each tile type through _TILE_TO_CMAP to get a cmap symbol index.
    3. Add GLYPH_CMAP_OFF to get NLE glyph ID.
    4. Three-way visibility split (mirrors vendor/nethack/src/display.c::lastseentyp
       ~line 850):
         - visible tile      → render from terrain (live truth)
         - explored+not-visible → render from last_seen_terrain (stale memory)
         - unexplored        → NO_GLYPH
    5. Overlay live monster glyphs; scramble via per-(timestep,tile) hash when
       hallucinating (vendor/nethack/src/display.c::display_monster ~line 599).
    6. Overlay the player glyph at player_pos.
    7. Scramble object glyphs (GLYPH_OBJ_OFF range) when hallucinating via the
       _HCOLOR_POOL table (vendor/nethack/src/do_name.c:1461 hcolor scramble).
       Terrain glyphs (GLYPH_CMAP_OFF range) are never scrambled.

    Returns:
        int16[21, 79]
    """
    branch = jnp.int32(env_state.dungeon.current_branch)
    level_idx = jnp.int32(env_state.dungeon.current_level) - 1  # 0-based

    # terrain shape: [N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H=21, MAP_W=80]
    # NLE glyphs shape: [21, 79] — trim last column
    level_terrain = env_state.terrain[branch, level_idx, :21, :79]         # int8[21,79]
    last_seen = env_state.last_seen_terrain[branch, level_idx, :21, :79]   # int8[21,79]

    # For explored-but-not-visible tiles use last_seen_terrain; sentinel -1
    # (never seen) falls back to 0 (VOID/stone) which renders as NO_GLYPH anyway.
    visible = env_state.visible[:21, :79]                                   # bool[21,79]
    explored = env_state.explored[branch, level_idx, :21, :79]             # bool[21,79]

    # Pick the terrain source per cell.
    display_terrain = jnp.where(
        visible,
        level_terrain,
        jnp.where(last_seen >= jnp.int8(0), last_seen, jnp.int8(0)),
    )

    # Clamp tile index to valid _TILE_TO_CMAP range (in case of corrupt data)
    tile_idx = jnp.clip(display_terrain.astype(jnp.int16), 0, NUM_TILE_TYPES - 1)
    cmap_idx = _TILE_TO_CMAP[tile_idx]                                     # int16[21,79]

    # Wall-angle pass — replace generic S_vwall (cmap 1) for WALL tiles with
    # the correct corner / T-junction / cross / horizontal variant, derived
    # from the 4 cardinal neighbours' wall pattern.  Vendor reference:
    # vendor/nethack/src/display.c::wall_angle (lines 143-151 forward decl,
    # body at lines 3512-3700) and set_wall_state / xy_set_wall_state.
    cmap_idx = _apply_wall_angle(display_terrain, cmap_idx)

    # Terrain glyph IDs
    terrain_glyphs = (cmap_idx + jnp.int16(GLYPH_CMAP_OFF)).astype(jnp.int16)

    # Unexplored tiles -> NO_GLYPH
    no_glyph_val = jnp.int16(NO_GLYPH & 0xFFFF)                           # NO_GLYPH as int16
    glyphs = jnp.where(explored, terrain_glyphs, no_glyph_val)

    # Overlay live monsters at their tile positions.  Each visible, alive
    # monster slot writes GLYPH_MON_OFF + entry_idx at its (row, col).
    # Vendor reference: display.c::show_glyph; mhitu.c writes monster glyph
    # via map_location each turn.
    mai = env_state.monster_ai
    mon_pos = mai.pos                          # int16[N, 2]
    mon_alive = mai.alive                      # bool[N]
    mon_entry = mai.entry_idx.astype(jnp.int32)  # int32[N]

    rows = jnp.clip(mon_pos[:, 0].astype(jnp.int32), 0, 20)
    cols = jnp.clip(mon_pos[:, 1].astype(jnp.int32), 0, 78)
    # Only overlay monsters that are alive AND on visible tiles.
    tile_visible = visible[rows, cols]
    write_mask = mon_alive & tile_visible & (mon_entry >= jnp.int32(0))
    mon_glyphs = (jnp.int32(GLYPH_MON_OFF) + mon_entry).astype(jnp.int16)

    # Hallucination scramble — vendor/nethack/src/display.c:599 randomizes
    # the monster glyph each render when Hallu.  We use a per-(timestep,row,col)
    # deterministic scramble so the same frame is consistent.
    # Cite: vendor/nethack/src/display.c::display_monster (line ~599).
    is_hallu = env_state.status.timed_statuses[10] > 0  # TimedStatus.HALLUCINATION
    NUMMONS = 381
    ts_u = env_state.timestep.astype(jnp.uint32)
    rows_u = rows.astype(jnp.uint32)
    cols_u = cols.astype(jnp.uint32)
    # Stable hash over (timestep, row, col) → int in [0, NUMMONS).  uint32 so
    # the Knuth/multiplier constants don't overflow int32.
    hash_seed = (ts_u * jnp.uint32(2654435761)
                 + rows_u * jnp.uint32(1597334677)
                 + cols_u * jnp.uint32(1431655781))
    hallu_entry = jnp.mod(hash_seed, jnp.uint32(NUMMONS)).astype(jnp.int32)
    hallu_glyphs = (jnp.int32(GLYPH_MON_OFF) + hallu_entry).astype(jnp.int16)
    final_mon_glyphs = jnp.where(is_hallu, hallu_glyphs, mon_glyphs)

    # JIT-safe scatter: for each slot, replace glyph at (row, col) only when
    # write_mask is True.  Using a vectorised .at[...] update is safe since
    # later writes naturally overwrite earlier ones for duplicate positions
    # (rare for live monsters).
    glyphs = glyphs.at[rows, cols].set(
        jnp.where(write_mask, final_mon_glyphs, glyphs[rows, cols])
    )

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

    # Hallucination hcolor scramble for object glyphs — vendor do_name.c:1461.
    # Any tile whose glyph is in [GLYPH_OBJ_OFF, GLYPH_CMAP_OFF) gets replaced
    # with a random entry from _HCOLOR_POOL when hallucinating.
    # The per-tile seed uses the same (timestep, row, col) hash as the monster
    # scramble above so the result is deterministic within a frame and changes
    # across frames.  Terrain glyphs (GLYPH_CMAP_OFF+) are left untouched.
    # TODO: status message scramble (status surface not yet in obs).
    # TODO: "Everything looks boring now" message on HALLUCINATION→0 transition.
    rows_all = jnp.arange(21, dtype=jnp.uint32)
    cols_all = jnp.arange(79, dtype=jnp.uint32)
    rc_rows, rc_cols = jnp.meshgrid(rows_all, cols_all, indexing='ij')  # [21,79]
    ts_grid = env_state.timestep.astype(jnp.uint32)
    hash_grid = (ts_grid * jnp.uint32(2654435761)
                 + rc_rows * jnp.uint32(1597334677)
                 + rc_cols * jnp.uint32(1431655781))
    pool_idx = jnp.mod(hash_grid, jnp.uint32(_HCOLOR_POOL_SIZE)).astype(jnp.int32)
    scrambled_obj_idx = _HCOLOR_POOL[pool_idx]                      # int32[21,79]
    scrambled_obj_glyphs = (jnp.int32(GLYPH_OBJ_OFF) + scrambled_obj_idx).astype(jnp.int16)
    is_obj_glyph = (glyphs.astype(jnp.int32) >= jnp.int32(GLYPH_OBJ_OFF)) & \
                   (glyphs.astype(jnp.int32) < jnp.int32(GLYPH_CMAP_OFF))
    scramble_mask = is_obj_glyph & is_hallu
    glyphs = jnp.where(scramble_mask, scrambled_obj_glyphs, glyphs)

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
    # Delegate to the canonical _build_chars so tty_chars is byte-equal to
    # obs["chars"]: full category dispatch (monster/object/cmap/body/invis/
    # warning/statue + NO_GLYPH=space).  See _build_chars for the table.
    glyphs = build_glyphs(env_state)              # int16[21,79]
    map_chars = _build_chars(glyphs)              # uint8[21,79]

    # Pad 79-wide map to 80 columns with spaces (NLE's tty grid is 24x80).
    pad_col = jnp.full((21, 1), ord(' '), dtype=jnp.uint8)
    map_chars_80 = jnp.concatenate([map_chars, pad_col], axis=1)    # uint8[21,80]
    tty = tty.at[1:22, :].set(map_chars_80)

    # --- Rows 22-23: status lines ---
    # Vendor (botl.c::do_statusline1/2) renders strings like:
    #   row 22: "Player the Adventurer  St:18 Dx:17 Co:17 In:7 Wi:8 Ch:8  Lawful"
    #   row 23: "Dlvl:1 $:0 HP:15(15) Pw:2(2) AC:8 Xp:1 T:1"
    # We render the same field order using a JIT-friendly digit table; see
    # _build_status_row1 / _build_status_row2 for vendor citations.
    blstats = build_blstats(env_state)
    row22 = _build_status_row1(env_state, blstats)
    row23 = _build_status_row2(env_state, blstats)
    tty = tty.at[22, :].set(row22)
    tty = tty.at[23, :].set(row23)

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
