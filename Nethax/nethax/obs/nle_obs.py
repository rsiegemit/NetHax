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
from Nethax.nethax.constants import cmap_indices as _cmap


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
# Cmap lookup table: TileType -> NLE cmap index.
#
# TileType enum (from constants.py):
#   VOID=0, FLOOR=1, CORRIDOR=2, WALL=3, CLOSED_DOOR=4, OPEN_DOOR=5,
#   STAIRCASE_UP=6, STAIRCASE_DOWN=7, WATER=8, LAVA=9, ALTAR=10,
#   FOUNTAIN=11, TRAP=12, HIDDEN_TRAP=13, THRONE=14, GRAVE=15, SHOP_FLOOR=16
#
# All cmap (S_*) indices come from Nethax.nethax.constants.cmap_indices, which
# resolves the active layout (NLE 3.x by default, NetHack 5.x when
# NETHAX_VENDOR_TREE=nethack_5x).  Cite:
#   - vendor/nle/include/rm.h:116-227         (3.x default)
#   - vendor/nethack/include/defsym.h:90-183  (5.x alternate)
# ---------------------------------------------------------------------------

_S_stone    = _cmap.S_stone
_S_vwall    = _cmap.S_vwall
_S_room     = _cmap.S_room
_S_darkroom = _cmap.S_darkroom
_S_corr     = _cmap.S_corr
_S_litcorr  = _cmap.S_litcorr
_S_upstair  = _cmap.S_upstair
_S_dnstair  = _cmap.S_dnstair
_S_altar    = _cmap.S_altar
_S_grave    = _cmap.S_grave
_S_throne   = _cmap.S_throne
_S_fountain = _cmap.S_fountain
_S_pool     = _cmap.S_pool
_S_lava     = _cmap.S_lava
_S_trap     = _cmap.S_arrow_trap   # generic trap base
_S_vcdoor   = _cmap.S_vcdoor
_S_vodoor   = _cmap.S_vodoor
_S_hcdoor   = _cmap.S_hcdoor
_S_hodoor   = _cmap.S_hodoor
_S_ndoor    = _cmap.S_ndoor

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
    # Indices 17..22 (DRAWBRIDGE_UP, ICE_FLOOR, POOL, TREE, HOLE, SINK) were
    # never enumerated before this row was added; the lookup clamps the tile
    # index to NUM_TILE_TYPES-1 (now 23) and JAX out-of-bounds gather used to
    # pin them to the final entry (_S_room).  To keep the table total now that
    # DOORWAY=23 must be reachable we fill 17..22 with _S_room, exactly
    # preserving that prior implicit behaviour (no render change for those
    # tiles), and add the real DOORWAY entry at index 23.
    _S_room,     # 17 DRAWBRIDGE_UP  (prior implicit clamp value)
    _S_room,     # 18 ICE_FLOOR      (prior implicit clamp value)
    _S_room,     # 19 POOL           (prior implicit clamp value)
    _S_room,     # 20 TREE           (prior implicit clamp value)
    _S_room,     # 21 HOLE           (prior implicit clamp value)
    _S_room,     # 22 SINK           (prior implicit clamp value)
    _S_ndoor,    # 23 DOORWAY        S_ndoor (doorless gap, char '.')
], dtype=jnp.int16)

# ---------------------------------------------------------------------------
# TTY char lookup table: cmap index -> ASCII character.
#
# Built from Nethax.nethax.constants.cmap_indices.CMAP_TO_CHAR (which derives
# from the active layout's symbolic dict, so the same symbol always renders
# the same char regardless of whether NLE 3.x or NetHack 5.x is selected).
# We slice to 64 entries (the table size this module historically promised);
# all indices used by _TILE_TO_CMAP fall inside that window in both layouts.
# ---------------------------------------------------------------------------

_CMAP_TO_CHAR: jnp.ndarray = jnp.asarray(_cmap.CMAP_TO_CHAR[:64], dtype=jnp.uint8)

# ---------------------------------------------------------------------------
# ANSI color lookup table: cmap index -> ANSI color (0-15).
#
# Built from Nethax.nethax.constants.cmap_indices.CMAP_TO_COLOR (which derives
# from the active layout's symbolic dict).  CLR_* values from
# vendor/nethack/include/color.h:
#   CLR_BLACK=0 CLR_RED=1 CLR_GREEN=2 CLR_BROWN=3 CLR_BLUE=4 CLR_MAGENTA=5
#   CLR_CYAN=6  CLR_GRAY=7 NO_COLOR=8→7  CLR_ORANGE=9 CLR_BRIGHT_GREEN=10
#   CLR_YELLOW=11 CLR_BRIGHT_BLUE=12 CLR_BRIGHT_MAGENTA=13 CLR_BRIGHT_CYAN=14
#   CLR_WHITE=15
# Sliced to 64 entries to match the historical _CMAP_TO_COLOR table size; all
# active indices fall inside that window in both layouts.
# ---------------------------------------------------------------------------

_CMAP_TO_COLOR: jnp.ndarray = jnp.asarray(_cmap.CMAP_TO_COLOR[:64], dtype=jnp.uint8)

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
    # CMAP description table (TileType-ish names, keyed by cmap index).
    # Built from Nethax.nethax.constants.cmap_indices.CMAP_DESC, which
    # indexes per-symbol descriptions through the active layout — so
    # "staircase up" lands at 23 under NLE 3.x and 25 under NetHack 5.x.
    # Cite: vendor/nethack/src/drawing.c default_showsyms[],
    #       vendor/nethack/src/pager.c::lookat() outputs.
    cmap_desc = dict(_cmap.CMAP_DESC)

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

    # Monsters & pets & detected & ridden — share name + color.  Pets get a
    # "tame " prefix: a GLYPH_PET_OFF glyph is, by construction, always a tame
    # monster, so the prefix is glyph-deterministic.  Vendor pager.c::lookat ->
    # look_at_monster (line 438) prepends "tame " (else "peaceful " for
    # mpeaceful) to distant_monnam() before storing firstmatch.
    for i, m in enumerate(MONSTERS):
        c = int(m.color) & 0xFF
        nm = _bytes_for(m.name)
        pet_nm = _bytes_for(f"tame {m.name}")
        for base in (_MON, _DET, _RID):
            g = base + i
            if 0 <= g < _MAX:
                colors[g] = c
                desc[g] = nm
        g = _PET + i
        if 0 <= g < _MAX:
            colors[g] = c
            desc[g] = pet_nm
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
    #
    # Gold-piece (COIN_CLASS) farlook special-case: vendor pager.c renders
    # ground gold as "some gold pieces", not the bare object name "gold piece".
    # The flow is:
    #   pager.c::look_at_object -> distant_name(otmp, doname_vague_quan)
    #     -> doname_base with DONAME_VAGUE_QUAN
    # Inside object_from_map (pager.c:189-190), if no real obj is at the spot
    # a fakeobj is created with otmp->quan = 2L "to force pluralization"; and
    # doname_base (objnam.c:966-970) emits "some " when quan != 1 and
    # !dknown (which doname_vague_quan implies for farlook).  Real ground
    # gold piles likewise carry quan>1 and dknown=0, yielding the same
    # "some gold pieces" string.  We bake that string into the static
    # glyph->description table for the single COIN_CLASS object slot.
    #
    # All other classes: vendor pager.c::look_at_object calls
    #   distant_name(otmp, dknown ? doname_with_price : doname_vague_quan).
    # For an unidentified item viewed at distance (the common screen_descriptions
    # case at level 1), this resolves to doname_base -> xname with nn=0
    # (oc_name_known false at game start; see o_init.c:341-371) and dknown
    # initially false.  xname (objnam.c:471-722) then emits the random
    # appearance string (OBJ_DESCR, our `description` field) for classes that
    # have one, with a class-specific suffix (e.g. "ruby potion", "wooden ring",
    # "etched helmet").  doname_base finally prepends an "a "/"an " article via
    # just_an() (objnam.c:1269-1276, 1648-1674).
    #
    # We bake the unidentified-appearance string + article into the static
    # table because (a) at game start no objects are name_known, (b) the
    # screen_descriptions buffer is indexed by *shuffled* glyph (the appearance
    # glyph that build_glyphs emits via shuffled_glyph), so OBJECTS[N].description
    # at appearance-slot N is exactly the obj_descr the player would see.
    # Identified-item paths (uncommon for floor obs) are deferred — they would
    # require a runtime table keyed on env_state.identification.
    from Nethax.nethax.constants.objects import ObjectClass as _OC, Material as _MAT

    # Vendor just_an() (objnam.c:1648-1674): pick "a "/"an " by first char of
    # the description.  Vowels and 'x' (when followed by a consonant) take
    # "an "; exceptions ("one-", "eucalyptus", "unicorn", "uranium", "useful")
    # take "a ".  Empty/None strings get no article.
    _VOWELS = "aeiouAEIOU"
    _AN_EXCEPTIONS = ("one-", "eucalyptus", "unicorn", "uranium", "useful")

    def _article(s: str) -> str:
        if not s:
            return ""
        c0 = s[0]
        sl = s.lower()
        if any(sl.startswith(e) for e in _AN_EXCEPTIONS):
            return "a "
        if c0 in _VOWELS:
            return "an "
        if c0 in ("x", "X") and len(s) > 1 and s[1].lower() not in "aeiou":
            return "an "
        return "a "

    # Vendor xname() (objnam.c:471-722) class-specific unidentified format.
    # Each branch returns the bare description string (no article); article is
    # prepended below.  When a description is absent (None) for an
    # appearance-bearing class, vendor falls back to the canonical name
    # (objnam.c:440-441 "if (!dn) dn = actualn;").
    def _unidentified_str(idx: int, o) -> str:
        if o is None or o.name is None:
            return ""
        cls = o.class_
        # Vendor: if (!dn) dn = actualn — fall back to canonical name.
        dn = o.description if o.description else o.name
        actualn = o.name

        if cls == _OC.COIN_CLASS:
            # Handled separately above via "some gold pieces".
            return actualn
        if cls == _OC.AMULET_CLASS:
            # xname: "%s amulet" using dn (when nn=0, un=0).
            return f"{dn} amulet"
        if cls in (_OC.WEAPON_CLASS, _OC.VENOM_CLASS, _OC.TOOL_CLASS):
            # xname WEAPON/VENOM/TOOL: bare dn (no suffix) when !dknown.
            # Special prefixes: LENSES -> "pair of "; FIGURINE/wet-towel
            # require runtime obj state and don't apply at distance.
            if actualn == "lenses":
                return f"pair of {dn}"
            return dn
        if cls == _OC.ARMOR_CLASS:
            # xname ARMOR: ELVEN_SHIELD..ORCISH_SHIELD -> "shield"; SHIELD_OF_REFLECTION
            # -> "smooth shield"; boots/gloves prefix "pair of "; else bare dn.
            if actualn in ("elven shield", "Uruk-hai shield", "orcish shield"):
                return "shield"
            if actualn == "shield of reflection":
                return "smooth shield"
            armcat = getattr(o, "oc_armor_class", -1)
            if armcat == 3:   # ARM_GLOVES
                return f"pair of {dn}"
            if armcat == 4:   # ARM_BOOTS
                return f"pair of {dn}"
            return dn
        if cls == _OC.FOOD_CLASS:
            # xname FOOD: always actualn (no dn).
            return actualn
        if cls == _OC.ROCK_CLASS:
            # Includes STATUE (corpsenm-specific) and ROCK; at distance
            # we have no corpsenm, so fall back to actualn.
            return actualn
        if cls == _OC.BALL_CLASS:
            # xname BALL: hardcoded "heavy iron ball".
            return "heavy iron ball"
        if cls == _OC.CHAIN_CLASS:
            return actualn
        if cls == _OC.POTION_CLASS:
            # xname POTION (!dknown branch): "%s potion".
            return f"{dn} potion"
        if cls == _OC.SCROLL_CLASS:
            # xname SCROLL (!dknown): "scroll" (just the class name).
            return "scroll"
        if cls == _OC.WAND_CLASS:
            # xname WAND (!dknown): "wand".
            return "wand"
        if cls == _OC.SPBOOK_CLASS:
            # xname SPBOOK (!dknown, non-NOVEL): "spellbook".
            return "spellbook"
        if cls == _OC.RING_CLASS:
            # xname RING (!dknown): "ring".
            return "ring"
        if cls == _OC.GEM_CLASS:
            # xname GEM (!dknown): "gem" or "stone" (MINERAL material).
            return "stone" if o.material == _MAT.MINERAL else "gem"
        # ILLOBJ / RANDOM / unknown — fall through to canonical name.
        return actualn

    for i, o in enumerate(OBJECTS):
        g = _OBJ + i
        if 0 <= g < _MAX:
            colors[g] = int(o.color) & 0xFF
            if o is not None and getattr(o, "class_", None) == _OC.COIN_CLASS:
                desc[g] = _bytes_for("some gold pieces")
            else:
                body = _unidentified_str(i, o)
                if body:
                    desc[g] = _bytes_for(_article(body) + body)
                else:
                    desc[g] = _bytes_for("")

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


def consume_disp_for_obs(env_state):
    """Drain the DISP ISAAC64 stream for one observation-build pass.

    Vendor NLE consumes ``rn2_on_display_rng`` (rnglist[DISP]) at two
    obs-emit sites every time an observation is constructed:

      1. ``glyphs`` channel — per visible alive monster, vendor's
         ``map_object_or_mimic`` / ``display_warning`` chain at
         vendor/nle/src/display.c:486-498 calls one of
         ``mon_to_glyph`` / ``detected_mon_to_glyph`` / ``pet_to_glyph``,
         each of which threads ``rn2_on_display_rng`` through
         ``random_monster``.  One DISP draw fires per visible monster.

      2. ``inv_glyphs`` channel — vendor's RL inventory callback at
         vendor/nle/win/rl/winrl.cc:458 calls
         ``obj_to_glyph(otmp, rn2_on_display_rng)`` once per occupied
         inventory slot.

    To preserve byte parity on these channels without polluting CORE,
    Nethax draws the matching number of DISP words here and returns the
    updated ISAAC64 state.  Each draw is the same scalar ``rn2_on_display_rng(2)``
    call that vendor ``random_monster`` / ``obj_to_glyph`` ultimately bottom
    out in (the modulus is irrelevant for stream advancement — what matters
    is that exactly one ``isaac64_next_uint64()`` fires).

    JIT-pure: uses ``jax.lax.scan`` over the fixed-size monster + inventory
    arrays, conditionally advancing DISP via ``lax.cond`` on the per-slot
    mask bit so only slots that vendor would actually emit consume a draw.
    """
    from Nethax.nethax import vendor_rng as _vendor_rng

    # 1. Per-visible-alive-monster DISP draws (glyphs channel).
    mai = env_state.monster_ai
    branch = jnp.int32(env_state.dungeon.current_branch)
    level_idx = jnp.int32(env_state.dungeon.current_level) - 1
    visible = env_state.visible[:21, :79]
    rows_m = jnp.clip(mai.pos[:, 0].astype(jnp.int32), 0, 20)
    cols_m = jnp.clip(mai.pos[:, 1].astype(jnp.int32), 0, 78)
    tile_visible = visible[rows_m, cols_m]
    mon_mask = (
        mai.alive
        & tile_visible
        & (mai.entry_idx.astype(jnp.int32) >= jnp.int32(0))
    )

    def _maybe_draw(rng, mask_bit):
        # When mask_bit is True, advance DISP by one uint64 draw;
        # otherwise leave the stream untouched.  Mirrors vendor's
        # per-slot ``if (visible) rn2_on_display_rng(...)`` gate.
        drew_rng, _ = _vendor_rng.next_uint64_jax(rng)
        return jax.lax.cond(mask_bit, lambda _: drew_rng, lambda _: rng, operand=None), None

    rng_after_mon, _ = jax.lax.scan(_maybe_draw, env_state.vendor_rng_disp, mon_mask)

    # 2. Per-occupied-inventory-slot DISP draws (inv_glyphs channel).
    cat = env_state.inventory.items.category
    inv_mask = (cat != jnp.int8(0))

    rng_after_inv, _ = jax.lax.scan(_maybe_draw, rng_after_mon, inv_mask)

    return rng_after_inv


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

    # Drop internal column 0 (obs col c = internal col c+1), matching NLE.
    tile = env_state.terrain[branch, level_idx, :21, 1:80]
    tile_idx = jnp.clip(tile.astype(jnp.int16), 0, NUM_TILE_TYPES - 1)
    cmap_idx = _TILE_TO_CMAP[tile_idx]  # int16[21,79]
    cmap_clamped = jnp.clip(cmap_idx, 0, len(_CMAP_TO_COLOR) - 1)
    colors = _CMAP_TO_COLOR[cmap_clamped]  # uint8[21,79]

    # Unexplored tiles -> color 0 (black)
    explored = env_state.explored[branch, level_idx, :21, 1:80]
    colors = jnp.where(explored, colors, jnp.uint8(0))

    # Object-color overlay (mirrors object-glyph overlay in build_glyphs).  Each
    # visible cell whose ground stack has a non-zero slot 0 picks up the
    # object's display color from the per-glyph color table.  Vendor cite:
    # vendor/nle/src/mapglyph.c GLYPH_OBJ_OFF branch sets
    # ``color = objects[obj_descr_idx].oc_color``.
    visible_all = env_state.visible[:21, 1:80]                     # bool[21,79]
    gi_cat0 = env_state.ground_items.category[branch, level_idx, :21, 1:80, 0]
    gi_typ0 = env_state.ground_items.type_id[branch, level_idx, :21, 1:80, 0]
    has_obj = (gi_cat0 != jnp.int8(0)) & visible_all
    obj_glyph_idx = jnp.clip(
        gi_typ0.astype(jnp.int32) + jnp.int32(GLYPH_OBJ_OFF),
        0, _GLYPH_TO_COLOR.shape[0] - 1,
    )
    obj_colors = _GLYPH_TO_COLOR[obj_glyph_idx]                    # uint8[21,79]
    colors = jnp.where(has_obj, obj_colors, colors)

    # Monster-color overlay.  Mirror the monster-glyph overlay in build_glyphs:
    # every visible, alive monster cell takes the monster's own display color
    # (vendor permonst.mcolor), NOT the terrain color underneath.  Pets are NOT
    # specially recolored here — pet-ness is conveyed via the specials MG_PET
    # bit; the cell simply shows the monster's own color (e.g. kitten = 15).
    # Cite: vendor/nle/src/mapglyph.c — the GLYPH_MON_OFF / GLYPH_PET_OFF
    # branches set `color = mons[mnum].mcolor`.  We read the per-entry color
    # from the same MONSTERS-derived table build_glyphs uses for glyphs
    # (_GLYPH_TO_COLOR[GLYPH_MON_OFF + entry] == MONSTERS[entry].color).
    visible = env_state.visible[:21, 1:80]                         # bool[21,79]
    mai = env_state.monster_ai
    mon_entry = mai.entry_idx.astype(jnp.int32)                    # int32[N]
    rows = jnp.clip(mai.pos[:, 0].astype(jnp.int32), 0, 20)
    state_cols = mai.pos[:, 1].astype(jnp.int32)
    mon_oncol0 = state_cols <= jnp.int32(0)
    cols = jnp.clip(state_cols - jnp.int32(1), 0, 78)
    tile_visible = visible[rows, cols]
    write_mask = (
        mai.alive & tile_visible & (mon_entry >= jnp.int32(0)) & (~mon_oncol0)
    )
    # Per-monster display color from the MONSTERS table (clamp index to table).
    glyph_idx = jnp.clip(
        jnp.int32(GLYPH_MON_OFF) + mon_entry,
        0, _GLYPH_TO_COLOR.shape[0] - 1,
    )
    mon_colors = _GLYPH_TO_COLOR[glyph_idx]                        # uint8[N]
    # Vectorized scatter: only overwrite where write_mask is True (later writes
    # naturally win for duplicate cells, matching build_glyphs).
    colors = colors.at[rows, cols].set(
        jnp.where(write_mask, mon_colors, colors[rows, cols])
    )

    # Player tile -> bright yellow (15).  player_pos[1] is a STATE column; the
    # obs map drops internal column 0, so the obs column is player_pos[1] - 1.
    # Applied AFTER the monster overlay so the hero cell always wins.
    pr = jnp.int32(env_state.player_pos[0])
    pc = jnp.clip(jnp.int32(env_state.player_pos[1]) - jnp.int32(1), 0, 78)
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
      MG_PET      : tame alive monster on tile (monster_ai.tame & monster_ai.alive)
      MG_INVIS/MG_DETECT/MG_RIDDEN/MG_STATUE/MG_BW_LAVA: unset for now.

    Returns:
        uint8[21, 79]
    """
    branch = jnp.int32(env_state.dungeon.current_branch)
    level_idx = jnp.int32(env_state.dungeon.current_level) - 1

    # Ground items: category[branch, level, row, col, stack] (int8)
    # stack dim is MAX_GROUND_STACK = 8; non-zero means item present.
    # Drop internal column 0 (obs col c = internal col c+1), matching NLE.
    gi_cat = env_state.ground_items.category[branch, level_idx, :21, 1:80, :]
    gi_typ = env_state.ground_items.type_id[branch, level_idx, :21, 1:80, :]

    occupied = gi_cat != 0
    stack_count = jnp.sum(occupied.astype(jnp.int32), axis=-1)
    # Vendor cite: vendor/nle/src/mapglyph.c:91-92, 164-165, 183-184 — the
    # ``MG_OBJPILE`` flag is set ONLY inside the ``GLYPH_STATUE``,
    # ``GLYPH_OBJ`` (non-BOULDER), and ``GLYPH_BODY`` branches.  Those
    # branches fire only when the displayed glyph at (x,y) is the
    # corresponding object/statue/corpse glyph — i.e. when the player can
    # currently see the cell (or remembers an object there).  Nethax's
    # object overlay in ``build_glyphs`` already gates ``has_obj`` on
    # ``visible`` (see :func:`build_glyphs` ~line 2088), so the displayed
    # glyph at an unseen cell stays as the terrain glyph and vendor's
    # mapglyph never enters the object branch.  We mirror that here by
    # gating ``has_objpile`` on visibility.  Without this gate, a hidden
    # 2-stack pile (e.g. seed=2 (15, 42)) sets MG_OBJPILE in Nethax while
    # vendor leaves the byte clean.
    visible = env_state.visible[:21, 1:80]                             # bool[21,79]
    has_objpile = (stack_count >= 2) & visible

    # Corpse: category == FOOD_CLASS (7) and type_id == CORPSE_OBJ_TYPE_ID (260).
    # Per vendor/nethack/include/objects.h FOOD("corpse", ...), corpse is the
    # canonical food entry; in our OBJECTS table that lands at index 260.
    from Nethax.nethax.subsystems.inventory import ItemCategory as _IC
    FOOD_CLASS = jnp.int8(int(_IC.FOOD))
    CORPSE_TYPE_ID = jnp.int16(260)
    is_corpse_stack = (gi_cat == FOOD_CLASS) & (gi_typ == CORPSE_TYPE_ID)
    # Gate on visibility — vendor mapglyph.c only sets MG_CORPSE in the
    # GLYPH_BODY branch when the displayed glyph is the corpse glyph
    # (i.e. when the cell is currently seen or remembered).  Without
    # this gate Nethax sets MG_CORPSE at off-screen corpses where
    # vendor leaves the byte clean (seed=9 @503 row=6 col=30).
    has_corpse = jnp.any(is_corpse_stack, axis=-1) & visible

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
        # STATE column -> obs column (drop internal col 0).
        pc = jnp.clip(jnp.int32(env_state.player_pos[1]) - jnp.int32(1), 0, 78)
        hero_mask = jnp.zeros((21, 79), dtype=jnp.uint8).at[pr, pc].set(
            jnp.uint8(bits.MG_HERO)
        )
        specials = specials | hero_mask
    return specials.astype(jnp.uint8)


def _pet_mask(env_state, branch, level_idx) -> jnp.ndarray:
    """Return a bool[21,79] mask of tiles occupied by a pet (tame monster).

    Vendor cite: vendor/nle/src/mapglyph.c line 201-207 — the GLYPH_PET_OFF
    branch sets ``special |= MG_PET`` for every tame monster glyph.  We mirror
    that by marking any tile where an alive & tame monster (mai.tame[i] == True)
    currently resides.

    monster_ai holds all monsters for the *current* level only, so no
    branch/level filtering is needed — the state is already level-local.
    """
    mai = env_state.monster_ai
    rows = jnp.clip(mai.pos[:, 0].astype(jnp.int32), 0, 20)
    # Convert STATE column (0..79) to obs column (state_col - 1, 0..78); pets on
    # internal column 0 fall off the left edge of the obs map and are dropped.
    state_cols = mai.pos[:, 1].astype(jnp.int32)
    oncol0 = state_cols <= jnp.int32(0)
    cols = jnp.clip(state_cols - jnp.int32(1), 0, 78)
    is_pet = mai.alive & mai.tame & (~oncol0)                  # bool[N]
    # Scatter: for each pet slot, set mask[r, c] = True.
    flat_idx = rows * jnp.int32(79) + cols                    # int32[N]
    flat_mask = jnp.zeros(21 * 79, dtype=jnp.bool_)
    flat_mask = flat_mask.at[flat_idx].max(is_pet)
    return flat_mask.reshape(21, 79)


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
    lookup table built at module load.

    Vendor parity (vendor/nle/win/rl/winrl.cc::store_screen_description, line
    940-941): store_screen_description is only called inside show_glyph(), which
    only fires for explored/visible cells.  The screen_descriptions buffer is
    memset to 0 before each turn (winrl.cc:316-317, 651), so unexplored cells
    (those that never had show_glyph called) remain all-zero.  We replicate
    this by zeroing the description for any cell where explored==False.

    Reference: vendor/nethack/src/pager.c::do_screen_description and
               vendor/nle/win/rl/winrl.cc::store_screen_description.

    Returns:
        uint8[21, 79, 80]
    """
    glyphs = build_glyphs(env_state)  # int16[21,79]
    # Clamp index into [0, MAX_GLYPH-1] before fancy indexing.
    g_idx = jnp.clip(glyphs.astype(jnp.int32), 0, _GLYPH_TO_DESCRIPTION_BYTES.shape[0] - 1)
    desc = _GLYPH_TO_DESCRIPTION_BYTES[g_idx]  # uint8[21,79,80]

    # Zero out descriptions for unexplored tiles — vendor only calls
    # store_screen_description for cells that went through show_glyph().
    branch = jnp.int32(env_state.dungeon.current_branch)
    level_idx = jnp.int32(env_state.dungeon.current_level) - 1  # 0-based
    # Drop internal column 0 to match build_glyphs / NLE obs layout.
    explored = env_state.explored[branch, level_idx, :21, 1:80]  # bool[21,79]
    # Broadcast explored[21,79] -> [21,79,1] to mask [21,79,80]
    desc = jnp.where(explored[:, :, None], desc, jnp.zeros_like(desc))

    # Hero-tile override: self_lookat() yields "<race-adj> <role-mon> called
    # <name>" (vendor pager.c:116), richer than the bare role-monster name the
    # glyph table produces.  Skip when polymorphed — the poly-form name from the
    # glyph table is closer there (self_lookat drops the race prefix).  The hero
    # tile is always explored, so this survives the mask above.
    is_poly = env_state.polymorph.is_polymorphed
    role_idx = jnp.clip(jnp.int32(env_state.player_role), 0, _HERO_DESC_BYTES.shape[0] - 1)
    race_idx = jnp.clip(jnp.int32(env_state.player_race), 0, _HERO_DESC_BYTES.shape[1] - 1)
    hero_bytes = _HERO_DESC_BYTES[role_idx, race_idx]            # uint8[80]
    pr = jnp.int32(env_state.player_pos[0])
    pc = jnp.clip(jnp.int32(env_state.player_pos[1]) - jnp.int32(1), 0, 78)
    new_row = jnp.where(is_poly, desc[pr, pc, :], hero_bytes)    # uint8[80]
    desc = desc.at[pr, pc, :].set(new_row)
    return desc


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
      Row 0:     message line — CLR_GRAY (7) on text, 0 on blanks
      Rows 1-21: map area — colors from build_colors(), padded to 80 cols
      Row 22-23: status lines — CLR_GRAY (7) on text, 0 on blanks

    NLE's tty_colors come from libtmt's virtual terminal: only cells the
    windowport actually *wrote a glyph to* carry a color attribute; every
    untouched (ASCII-space) cell stays color 0.  This holds uniformly across
    the whole 24x80 grid — verified against vendor NLE (rog-hum-cha-mal seed 0,
    reset + North steps): for every cell, ``tty_colors == 0`` iff
    ``tty_chars == ' '`` (0x20).  We therefore build the base color grid and
    then zero out any cell whose tty_chars byte is a space.

    Cite: vendor/nle/src/nle.c::nle_vt_callback (libtmt) — color attributes
          only set on written cells; vendor/nethack/win/tty draws toplines /
          status lines via tty_putsym which emits color only for glyph cells.

    Returns:
        int8[24, 80]
    """
    tty_colors = jnp.zeros((24, 80), dtype=jnp.int8)

    # Row 0: message line -> CLR_GRAY (7) base (masked to 0 on blanks below).
    tty_colors = tty_colors.at[0, :].set(jnp.int8(7))

    # Rows 1-21: map colors, padded to 80 cols
    map_colors = build_colors(env_state)  # uint8[21,79]
    pad_col = jnp.zeros((21, 1), dtype=jnp.int8)
    map_colors_80 = jnp.concatenate(
        [map_colors.astype(jnp.int8), pad_col], axis=1
    )  # int8[21,80]
    tty_colors = tty_colors.at[1:22, :].set(map_colors_80)

    # Rows 22-23: status lines -> CLR_GRAY (7) base (masked to 0 on blanks).
    tty_colors = tty_colors.at[22, :].set(jnp.int8(7))
    tty_colors = tty_colors.at[23, :].set(jnp.int8(7))

    # Universal blank mask: any tty cell rendered as ASCII space gets color 0,
    # matching NLE's libtmt (unwritten cells carry no color attribute).  The
    # map region already zeroes unexplored tiles via build_colors, but its
    # padding column and the message / status blanks need the same treatment.
    tty_chars = _build_tty_chars(env_state)                     # uint8[24,80]
    is_blank = (tty_chars == jnp.uint8(ord(' ')))
    tty_colors = jnp.where(is_blank, jnp.int8(0), tty_colors)

    return tty_colors


def build_inv_glyphs(env_state) -> jnp.ndarray:
    """Glyph for each inventory slot. Shape (55,) int16.

    Vendor parity (vendor/nle/win/rl/winrl.cc::observation_glyphs ~line 379
    and update_inventory_method ~line 444): each occupied slot emits
    ``shuffled_glyph(GLYPH_OBJ_OFF + otyp)`` so the player sees the
    per-run appearance, not the canonical type.  The shuffle table
    ``env_state.descr_idx`` is the identity permutation in default
    ``ParityMode.NLE``, so this is a no-op there; under
    ``NLE_BYTEPARITY`` it remaps every object glyph through the
    vendor-replay shuffle (winrl.cc lines 80-87).

    Returns:
        int16[55]
    """
    from Nethax.nethax.obs.glyph_shuffle import shuffled_glyph as _shuffled_glyph
    inv = jnp.full((55,), jnp.int16(NO_GLYPH & 0xFFFF), dtype=jnp.int16)
    items = env_state.inventory.items
    cat = items.category.astype(jnp.int16)
    typ = items.type_id.astype(jnp.int16)
    canonical = jnp.int16(GLYPH_OBJ_OFF) + typ
    shuffled = _shuffled_glyph(canonical, env_state.descr_idx)
    glyphs_52 = jnp.where(
        cat != 0,
        shuffled,
        jnp.int16(NO_GLYPH & 0xFFFF),
    )
    inv = inv.at[:52].set(glyphs_52)
    return inv


def build_inv_letters(env_state) -> jnp.ndarray:
    """ASCII letter for each inventory slot. Shape (55,) uint8.

    Vendor parity (vendor/nle/win/rl/winrl.cc::observation_letters ~line 396):
      Occupied slots  : obj->invlet  (i.e. 'a'..'z', 'A'..'Z')
      Empty slots     : 0

    Source-of-truth: ``InventoryState.letters`` — populated at character
    creation (subsystems/inventory.py::InventoryState.from_items) and on
    pickup (subsystems/inventory.py::pickup) per vendor
    invent.c::assigninvlet (lines 693-732).  Letters stick with the item
    through wield/wear and are freed on drop.

    Returns:
        uint8[55]
    """
    cat = env_state.inventory.items.category
    occupied = (cat != 0)
    # InventoryState.letters is int8 (signed); cast to uint8 for the obs.
    letters_raw = env_state.inventory.letters.astype(jnp.uint8)
    letters_52 = jnp.where(occupied, letters_raw, jnp.uint8(0))
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

# Sentinel byte used to mark *real* (literal) separator spaces while building a
# status row.  Numeric fields are emitted at a fixed max width and right-padded
# with ASCII space (0x20); literal separators (" ", labels, "(", ")") never
# contain 0x20 — they use this sentinel for any space they want to keep.  After
# the field stream is assembled, ``_compact_pad_spaces`` deletes every 0x20
# (the numeric trailing pad) and converts the sentinel back to a real space,
# yielding the natural-width, single-separator layout NLE's tty status engine
# produces (e.g. "St:14 Dx:18 ... Ch:9", "HP:12(12)").  0x1F is the ASCII unit
# separator — it never appears in a status line.
_SEP = 0x1F


def _str_to_bytes(s: str) -> jnp.ndarray:
    """Precompute a uint8 byte-array for a literal string (module load only)."""
    return jnp.array([ord(c) for c in s], dtype=jnp.uint8)


def _sep_bytes(s: str) -> jnp.ndarray:
    """Like ``_str_to_bytes`` but every ASCII space becomes the ``_SEP`` sentinel.

    Use for status-row *separator/label* literals so their spaces survive the
    ``_compact_pad_spaces`` pass (which strips numeric trailing-pad 0x20s).
    """
    return jnp.array([(_SEP if c == ' ' else ord(c)) for c in s], dtype=jnp.uint8)


def _compact_pad_spaces(buf: jnp.ndarray, out_w: int) -> jnp.ndarray:
    """Delete numeric trailing-pad spaces and restore separator sentinels.

    Stable left-compaction that removes every ASCII space (0x20) from ``buf``
    (these are the right-pad bytes emitted by fixed-width numeric renderers),
    keeps all other bytes in order, then rewrites the ``_SEP`` sentinel back to
    a real space.  The result is right-padded with spaces to ``out_w``.

    JIT-safe: uses a stable argsort to gather kept bytes to the front (no data-
    dependent control flow, static shapes throughout).
    """
    n = buf.shape[0]
    keep = buf != jnp.uint8(0x20)                       # bool[n]
    # Stable partition: kept bytes first (in original order), dropped after.
    # rank = position among kept bytes; dropped bytes get pushed past the end.
    kept_rank = jnp.cumsum(keep.astype(jnp.int32)) - 1  # 0-based index per kept
    n_kept = jnp.sum(keep.astype(jnp.int32))
    dest = jnp.where(keep, kept_rank, jnp.int32(n))     # dropped -> sentinel slot
    # Scatter into an oversized buffer (n+1 slots; slot n collects all drops).
    compacted = jnp.full((n + 1,), jnp.uint8(0x20), dtype=jnp.uint8)
    compacted = compacted.at[dest].set(buf, mode="drop")
    compacted = compacted[:n]
    # Restore separator sentinels to real spaces; blank everything past n_kept.
    idx = jnp.arange(n, dtype=jnp.int32)
    compacted = jnp.where(compacted == jnp.uint8(_SEP), jnp.uint8(0x20), compacted)
    compacted = jnp.where(idx < n_kept, compacted, jnp.uint8(0x20))
    return _pad_to(compacted, out_w)


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


def _uint_to_bytes_left(value, max_w: int):
    """Return a uint8[max_w] array of LEFT-aligned digit bytes for ``value``.

    Mirrors vendor printf ``%d`` semantics: writes the integer's decimal digits
    starting at position 0 (MSB-first), then pads the remainder with ASCII
    spaces (0x20).  Caller chooses ``max_w`` large enough for the worst-case
    digit count; trailing spaces emulate vendor ``%-Nd`` width padding.

    Implementation is JIT-safe: digits are extracted MSB-first via a static
    Python loop bound by ``max_w``; the "right-shift to leading position"
    step uses a static permutation table built at trace time.

    For ``value == 0`` we emit a single ``'0'`` at position 0 followed by
    spaces.

    Citation: vendor/nethack/src/botl.c::do_statusline2 uses ``%d`` / ``%-2d``
    for every numeric field on row 23 — see lines 130, 143, 152, 159.
    """
    v = jnp.maximum(jnp.int32(value), jnp.int32(0))
    # Extract digits LSB-first, then reverse to MSB-first.
    digit_arrs = []
    for _ in range(max_w):
        digit_arrs.append((v % 10).astype(jnp.int32))
        v = v // 10
    digit_arrs = digit_arrs[::-1]                          # MSB-first, len=max_w
    digits = jnp.stack(digit_arrs)                         # int32[max_w]

    # Count leading zero positions (each one becomes a space in MSB-first
    # right-aligned form).  ``digit_count`` = number of significant digits;
    # always >= 1 (we always show the units digit for value=0).
    is_zero = digits == 0
    # cumulative AND from the left: True while we've only seen zeros so far.
    leading_zero_mask = jnp.cumprod(is_zero.astype(jnp.int32)) > 0
    leading_zero_count = jnp.sum(leading_zero_mask.astype(jnp.int32))
    # If every digit is zero, value==0 → show one digit.
    digit_count = jnp.maximum(jnp.int32(max_w) - leading_zero_count, jnp.int32(1))
    # Position of the first significant digit in MSB-first form:
    first_digit_pos = jnp.int32(max_w) - digit_count       # in [0, max_w-1]

    # Left-shift: for output index i, take digit at index (first_digit_pos + i)
    # when i < digit_count, else emit space.
    positions = jnp.arange(max_w, dtype=jnp.int32)
    src_idx = jnp.clip(first_digit_pos + positions, 0, max_w - 1)
    shifted_digits = digits[src_idx]
    is_digit = positions < digit_count
    chars = jnp.where(is_digit,
                      _DIGITS[shifted_digits],
                      jnp.uint8(ord(' ')))
    return chars.astype(jnp.uint8)


# Precomputed static byte sequences for status rows (module-load).
# Player name used on the status line.  NLE forces playername = "Agent" for
# every character spec (vendor/nle/nle/env/base.py:306 sets plname = "Agent-"
# + role/race/...), so the botl header reads "Agent the <Title>".  Cite:
# vendor/nethack/src/botl.c::do_statusline1 lines 57-59 (Strcpy(newbot1,
# svp.plname) then capitalise first letter -> "Agent").
_NLE_PLAYER_NAME = "Agent"
# Legacy fallback header — retained when the (role, xlevel) -> rank lookup
# yields no usable title.
_S_NAME_PREFIX  = _str_to_bytes(f"{_NLE_PLAYER_NAME} the Adventurer")


def _build_hero_desc_bytes():  # pragma: no cover — runs once at import
    """self_lookat() hero description bytes per (role, race), shape [13,5,80].

    NLE stores the hero tile's screen_description via lookat()->self_lookat()
    (vendor/nethack/src/pager.c:116):
        Sprintf(outbuf, "%s%s%s called %s",
                invis?, urace.adj+" ", pmname(mons[u.umonnum]), plname);
    When not invisible / not polymorphed this is
    "<race-adj> <role-monster-name> called <plname>", e.g. a Human Rogue named
    "Agent" -> "human rogue called Agent".  This is richer than the bare
    role-monster name ("rogue") the glyph->description table yields, so the
    hero cell needs a per-tile override.

    role-monster index order matches the Role enum (ARCHEOLOGIST=0..WIZARD=12),
    same urole.malenum values build_glyphs uses.  race-adj order matches the
    Race enum (HUMAN=0..ORC=4); strings are vendor races[].adj (role.c).
    """
    import numpy as _np
    from Nethax.nethax.constants.monsters import MONSTERS
    role_mon_idx = [327, 328, 329, 331, 332, 333, 334, 336, 337, 338, 339, 340, 341]
    race_adj = ["human", "elven", "dwarvish", "gnomish", "orcish"]
    out = _np.zeros((len(role_mon_idx), len(race_adj), 80), dtype=_np.uint8)
    for r, midx in enumerate(role_mon_idx):
        role_name = MONSTERS[midx].name
        for rc, adj in enumerate(race_adj):
            enc = f"{adj} {role_name} called {_NLE_PLAYER_NAME}".encode(
                "ascii", errors="ignore"
            )[:79]
            out[r, rc, : len(enc)] = _np.frombuffer(enc, dtype=_np.uint8)
    return jnp.asarray(out, dtype=jnp.uint8)


_HERO_DESC_BYTES = _build_hero_desc_bytes()


# ---------------------------------------------------------------------------
# Precomputed (role, rank) -> "Agent the <Title>" byte rows.
# Each row is padded to _HEADER_PAD_W bytes — the fixed column at which NLE's
# tty status engine starts the stats group ("St:NN Dx:NN ...").  Instrumenting
# NLE (reset rog-hum-cha-mal, seed 0) shows "St:" begins at tty col 31, so the
# header field is exactly 31 bytes wide.  (Vendor do_statusline1's
# ``i = gm.mrank_sz + 15`` pad formula does not reconcile with the observed
# column under NLE's tty window port; we match NLE's measured layout directly.)
# Cite: vendor/nethack/src/botl.c::do_statusline1 lines 57-83 (name + " the " +
#   rank() + space-pad before "St:").
# ---------------------------------------------------------------------------

_HEADER_PAD_W = 31   # status row 1 reserves cols 0..30 for "<Name> the <Title>".

def _build_role_header_table() -> jnp.ndarray:
    """Return uint8[N_ROLES, N_RANKS, _HEADER_PAD_W] of header bytes.

    For role r, xlevel-rank k the row is "Agent the <title>" left-justified,
    right-padded with spaces to _HEADER_PAD_W bytes.
    """
    n_roles = len(_ROLE_RANK_TITLES)
    n_ranks = 9  # vendor rank_of returns 0..8
    rows = []
    for r in range(n_roles):
        for k in range(n_ranks):
            title = _ROLE_RANK_TITLES[r][k]
            s = f"{_NLE_PLAYER_NAME} the {title}"[:_HEADER_PAD_W]
            s = s.ljust(_HEADER_PAD_W)
            rows.append([ord(c) & 0xFF for c in s])
    arr = jnp.array(rows, dtype=jnp.uint8)
    return arr.reshape(n_roles, n_ranks, _HEADER_PAD_W)

# Default fallback row (matches the legacy hardcoded header).
_DEFAULT_HEADER_ROW = jnp.array(
    [ord(c) & 0xFF for c in f"{_NLE_PLAYER_NAME} the Adventurer".ljust(_HEADER_PAD_W)],
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
# Status-row separator/label literals.  Spaces here are *real* separators that
# must survive the trailing-pad compaction (``_compact_pad_spaces``), so they
# use the ``_SEP`` sentinel via ``_sep_bytes``.  Every value field placed
# between these is emitted at a fixed max width and right-padded with ASCII
# space (0x20); the compaction step deletes that pad, leaving single-separator,
# natural-width fields (e.g. "St:14 Dx:18 ... Ch:9").  Alignment uses a single
# leading separator space to match NLE's observed layout ("Ch:9 Chaotic").
_S_ST           = _sep_bytes("St:")
_S_SP_DX        = _sep_bytes(" Dx:")
_S_SP_CO        = _sep_bytes(" Co:")
_S_SP_IN        = _sep_bytes(" In:")
_S_SP_WI        = _sep_bytes(" Wi:")
_S_SP_CH        = _sep_bytes(" Ch:")
_S_ALIGN_LAW    = _sep_bytes(" Lawful")
_S_ALIGN_NEU    = _sep_bytes(" Neutral")
_S_ALIGN_CHA    = _sep_bytes(" Chaotic")

_S_DLVL         = _sep_bytes("Dlvl:")
_S_SP_DOLLAR    = _sep_bytes(" $:")
_S_SP_HP        = _sep_bytes(" HP:")
_S_OPEN         = _sep_bytes("(")
_S_CLOSE_SP_PW  = _sep_bytes(") Pw:")
_S_CLOSE_SP_AC  = _sep_bytes(") AC:")
_S_SP_XP        = _sep_bytes(" Xp:")
_S_SLASH        = _sep_bytes("/")
_S_SP_T         = _sep_bytes(" T:")
_S_PAD80        = jnp.full((80,), ord(' '), dtype=jnp.uint8)


def _pad_to(arr: jnp.ndarray, n: int) -> jnp.ndarray:
    """Right-pad `arr` with spaces up to `n` bytes (truncate if too long)."""
    if arr.shape[0] >= n:
        return arr[:n]
    return jnp.concatenate([arr, jnp.full((n - arr.shape[0],), ord(' '), dtype=jnp.uint8)])


_S_SP_S = _sep_bytes(" S:")


def _build_status_row1(env_state, blstats) -> jnp.ndarray:
    """Render row 22 of tty_chars — NLE tty status line 1.

    Observed NLE layout (reset rog-hum-cha-mal, seed 0):
        "Agent the Footpad              St:14 Dx:18 Co:14 In:11 Wi:10 Ch:9 Chaotic S:0"
         ^col 0                         ^col 31

    Layout rules matched against instrumented NLE:
      * Header "Agent the <RankTitle>" left-justified, padded to col 31
        (where the "St:" stats group begins).
      * Stats are single-space joined with *natural width* numbers
        (vendor "%-1d"): "St:14 Dx:18 Co:14 In:11 Wi:10 Ch:9" — note "Ch:9"
        is one digit, NOT zero-padded to two.
      * Alignment follows with a single leading space: " Chaotic".
      * Score tail " S:<n>" is always present (flags.showscore is on), shown
        even when the score is 0.

    Implementation: every numeric field is rendered at a fixed max width
    (right-padded with 0x20) and joined with sentinel-space separators; the
    header stays verbatim while the stats stream is fed through
    ``_compact_pad_spaces`` to collapse the numeric trailing-pad into the
    natural-width, single-separator NLE form.

    Citation: vendor/nethack/src/botl.c::do_statusline1 (lines 48-98) supplies
    the field order; NLE's tty window port supplies the exact column layout.
    """
    al = blstats[BL_ALIGN]
    # Select alignment bytes (all length 8: " Lawful"=7, " Neutral"=8,
    # " Chaotic"=8 -> pad shorter to 8 so jnp.where shapes match; trailing 0x20
    # pad is stripped by compaction).
    is_chaotic = (al == jnp.int64(-1))
    is_neutral = (al == jnp.int64(0))
    align_bytes = jnp.where(
        is_chaotic, _pad_to(_S_ALIGN_CHA, 8),
        jnp.where(is_neutral, _pad_to(_S_ALIGN_NEU, 8), _pad_to(_S_ALIGN_LAW, 8)),
    )

    # Header: "Agent the <RankTitle>" padded to col 31 (verbatim — its layout
    # spaces are kept, NOT compacted).  ``player_role`` is the Role enum int8;
    # ``player_xl`` (u.ulevel) drives xlev_to_rank.
    header = _role_header_bytes(env_state.player_role, env_state.player_xl)

    # Strength uses vendor get_strength_str format (fixed 5-byte field,
    # right-padded with 0x20 — the pad is stripped by compaction).
    from Nethax.nethax.obs.strength_format import render_strength_bytes
    # Score-on-botl tail: " S:%ld", always shown (flags.showscore on in NLE).
    score = blstats[BL_SCORE]

    # Stats stream (everything after the header).  Numeric fields are
    # max-width, right-padded; separators carry the _SEP sentinel.
    stats_parts = [
        _S_ST,                                       # "St:"
        render_strength_bytes(blstats[BL_STR125]),   # width 5, 0x20 padded
        _S_SP_DX,
        _uint_to_bytes_left(blstats[BL_DEX], 2),
        _S_SP_CO,
        _uint_to_bytes_left(blstats[BL_CON], 2),
        _S_SP_IN,
        _uint_to_bytes_left(blstats[BL_INT], 2),
        _S_SP_WI,
        _uint_to_bytes_left(blstats[BL_WIS], 2),
        _S_SP_CH,
        _uint_to_bytes_left(blstats[BL_CHA], 2),
        align_bytes,
        _S_SP_S,                                     # " S:"
        _uint_to_bytes_left(score, 7),
    ]
    stats = _compact_pad_spaces(jnp.concatenate(stats_parts), 80 - _HEADER_PAD_W)
    row = jnp.concatenate([header, stats])
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
    """Build a fixed-width keyword chunk for the row-23 conditions tail.

    The keyword's *leading separator space* uses the ``_SEP`` sentinel so it
    survives ``_compact_pad_spaces``; trailing pad (0x20) is stripped.  Keyword
    names themselves contain no interior spaces.
    """
    raw = list(s.encode("ascii"))
    arr = [(_SEP if b == ord(' ') else b) for b in raw] + [ord(' ')] * (width - len(s))
    return jnp.array(arr[:width], dtype=jnp.uint8)


# Keywords with leading space; widths chosen to be just long enough.
_KW_STONE     = _kw_bytes(" Stone",      6)
_KW_SLIME     = _kw_bytes(" Slime",      6)
_KW_STRNGL    = _kw_bytes(" Strngl",     7)
_KW_FOODPOIS  = _kw_bytes(" FoodPois",   9)
_KW_TERMILL   = _kw_bytes(" TermIll",    8)
# Hunger keywords — vendor eat.c::hu_stat lines 70-73 (8-char padded names,
# but printed via ``" %s"`` then ``mungspaces`` collapses trailing pad).  We
# emit the trimmed names with a leading space to match the post-mungspaces
# vendor output.
_KW_HUNGRY    = _kw_bytes(" Hungry",     7)
_KW_WEAK      = _kw_bytes(" Weak",       5)
_KW_FAINTING  = _kw_bytes(" Fainting",   9)
_KW_FAINTED   = _kw_bytes(" Fainted",    8)
_KW_STARVED   = _kw_bytes(" Starved",    8)
_KW_BURDENED  = _kw_bytes(" Burdened",   9)
_KW_STRESSED  = _kw_bytes(" Stressed",   9)
_KW_STRAINED  = _kw_bytes(" Strained",   9)
_KW_OVERTAX   = _kw_bytes(" Overtaxed", 10)
_KW_OVERLOAD  = _kw_bytes(" Overloaded",11)
_KW_BLIND     = _kw_bytes(" Blind",      6)
_KW_DEAF      = _kw_bytes(" Deaf",       5)
_KW_STUN      = _kw_bytes(" Stun",       5)
_KW_CONF      = _kw_bytes(" Conf",       5)
_KW_HALLU     = _kw_bytes(" Hallu",      6)
_KW_LEV       = _kw_bytes(" Lev",        4)
_KW_FLY       = _kw_bytes(" Fly",        4)
_KW_RIDE      = _kw_bytes(" Ride",       5)


def build_status_conditions(env_state) -> jnp.ndarray:
    """Vendor-format status-condition keyword tail as a uint8 vector.

    Reads ``env_state.status`` (TimedStatus enum, hunger_state, encumbrance,
    sick_kind) plus ``env_state.player_steed_mid`` for the riding flag, then
    masks each keyword's bytes to spaces when inactive, concatenates in
    vendor emit order (botl.c::do_statusline2 lines 173-205):

        Stone, Slime, Strngl, FoodPois, TermIll, <Hunger>, <Encumbrance>,
        Blind, Deaf, Stun, Conf, Hallu, Lev, Fly, Ride.

    Citation: vendor/nethack/src/botl.c::do_statusline2 (lines 173-205)
              where each ``Strcpy(nb = eos(nb), " <KW>")`` is gated by the
              corresponding ``Stoned``, ``Slimed``, ``Blind`` ... flag.
    """
    # TimedStatus indices (must match status_effects.TimedStatus enum order).
    ts = env_state.status.timed_statuses                         # int32[N]
    is_stun       = ts[0]  > 0   # STUNNED
    is_conf       = ts[1]  > 0   # CONFUSION
    is_blind      = ts[2]  > 0   # BLIND
    is_deaf       = ts[3]  > 0   # DEAF
    is_sick       = ts[4]  > 0   # SICK
    is_stone      = ts[5]  > 0   # STONED
    is_strngl     = ts[6]  > 0   # STRANGLED
    is_slime      = ts[9]  > 0   # SLIMED
    is_hallu      = ts[10] > 0   # HALLUCINATION
    # Levitation / Flying — TimedStatus.LEVITATION_TMP=18, FLYING_TMP=19.
    is_lev        = ts[18] > 0   # LEVITATION_TMP
    is_fly        = ts[19] > 0   # FLYING_TMP

    # SICK splits into FoodPois vs TermIll based on status.sick_kind.  Vendor
    # botl.c lines 180-183 uses ``u.usick_type & SICK_VOMITABLE`` /
    # ``SICK_NONVOMITABLE``; we model the same split via sick_kind == 1 / 2.
    sick_kind = jnp.int32(env_state.status.sick_kind)
    is_foodpois = is_sick & (sick_kind == 1)
    is_termill  = is_sick & (sick_kind == 2)

    # Hunger — vendor eat.c::hu_stat (lines 70-73) keyed by u.uhs;
    # NOT_HUNGRY (1) is the "no condition" sentinel and is skipped per
    # ``if (u.uhs != NOT_HUNGRY)`` (botl.c:185).
    hs = jnp.int32(env_state.status.hunger_state)
    is_satiated = hs == 0
    is_hungry   = hs == 2
    is_weak     = hs == 3
    is_fainting = hs == 4
    is_fainted  = hs == 5
    is_starved  = hs == 6

    # Encumbrance (Encumbrance enum: 0=UN, 1=BURDENED, 2=STRESSED, 3=STRAINED,
    # 4=OVERTAXED, 5=OVERLOADED).  See status_effects.Encumbrance.
    enc = jnp.int32(env_state.status.encumbrance)
    is_burdened  = enc == 1
    is_stressed  = enc == 2
    is_strained  = enc == 3
    is_overtaxed = enc == 4
    is_overload  = enc == 5

    # Riding — vendor botl.c:204 ``if (u.usteed)``.  Nethax stores the
    # mounted-steed monster id in ``player_steed_mid`` (0 when not riding).
    is_ride = env_state.player_steed_mid > jnp.uint32(0)

    def _mask(kw: jnp.ndarray, active: jnp.ndarray) -> jnp.ndarray:
        """Return kw bytes if active else all spaces, same length as kw."""
        spaces = jnp.full(kw.shape, jnp.uint8(ord(' ')), dtype=jnp.uint8)
        return jnp.where(active, kw, spaces)

    # Vendor order (botl.c::do_statusline2 lines 173-205):
    #   Stone, Slime, Strngl, FoodPois, TermIll, <Hunger>, <Encumbrance>,
    #   Blind, Deaf, Stun, Conf, Hallu, Lev, Fly, Ride.
    chunks = [
        _mask(_KW_STONE,    is_stone),
        _mask(_KW_SLIME,    is_slime),
        _mask(_KW_STRNGL,   is_strngl),
        _mask(_KW_FOODPOIS, is_foodpois),
        _mask(_KW_TERMILL,  is_termill),
        # Hunger group — only one active at a time.
        _mask(_KW_HUNGRY,   is_hungry),
        _mask(_KW_WEAK,     is_weak),
        _mask(_KW_FAINTING, is_fainting),
        _mask(_KW_FAINTED,  is_fainted),
        _mask(_KW_STARVED,  is_starved),
        # SATIATED is also emitted by vendor via hu_stat[0]="Satiated".
        _mask(_kw_bytes(" Satiated", 9), is_satiated),
        # Encumbrance group — only one active at a time.
        _mask(_KW_BURDENED, is_burdened),
        _mask(_KW_STRESSED, is_stressed),
        _mask(_KW_STRAINED, is_strained),
        _mask(_KW_OVERTAX,  is_overtaxed),
        _mask(_KW_OVERLOAD, is_overload),
        _mask(_KW_BLIND,    is_blind),
        _mask(_KW_DEAF,     is_deaf),
        _mask(_KW_STUN,     is_stun),
        _mask(_KW_CONF,     is_conf),
        _mask(_KW_HALLU,    is_hallu),
        _mask(_KW_LEV,      is_lev),
        _mask(_KW_FLY,      is_fly),
        _mask(_KW_RIDE,     is_ride),
    ]
    return jnp.concatenate(chunks)


def _build_status_row2(env_state, blstats) -> jnp.ndarray:
    """Render row 23 of tty_chars — NLE tty status line 2.

    Observed NLE layout (reset rog-hum-cha-mal, seed 0):
        "Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0 T:1"

    Format (vendor botl.c::do_statusline2 lines 130, 143, 152, 159):
        "Dlvl:%-2d $:%-2ld HP:%d(%d) Pw:%d(%d) AC:%-2d Xp:%d/%-1ld T:%ld <conds>"

    Layout rules matched against instrumented NLE:
      * Single-space separators, natural-width numbers (no field padding).
      * "HP:12(12)" / "Pw:2(2)" — no space between value and parens.
      * "AC:7" — positive AC carries no sign and no leading space; negative
        AC keeps its '-'.
      * "Xp:1/0" — experience LEVEL "/" experience POINTS (BL_XP / BL_EXP).

    Implementation: numeric fields are emitted at fixed max width (0x20
    right-padded), separators/parens carry the _SEP sentinel, and the whole
    stream (including the conditions tail) is fed through
    ``_compact_pad_spaces`` to collapse trailing pad into the natural-width,
    single-separator NLE form.

    The ``<conditions>`` suffix is the keyword tail produced by
    ``build_status_conditions`` (sentinel-separated; inactive keywords blank
    to 0x20 and are stripped by compaction).

    Citation: vendor/nethack/src/botl.c::do_statusline2 (lines 100-249).
    """
    ac = blstats[BL_AC]
    neg = ac < 0
    abs_ac = jnp.abs(ac)
    # Sign byte: '-' when negative, else 0x20 (stripped by compaction so
    # positive AC renders "AC:7" with no leading space).
    sign_byte = jnp.where(neg, jnp.uint8(ord('-')), jnp.uint8(ord(' '))).reshape(1)

    # Per-field max widths bound the worst-case digit count; trailing 0x20 pad
    # is stripped by compaction:
    #   Dlvl: 2  $: 6  HP/Pw: 4  AC: 2(+sign)  Xp(level): 2  Xp(points): 8
    #   T: 7
    parts = [
        _S_DLVL,                                                # "Dlvl:"
        _uint_to_bytes_left(blstats[BL_DEPTH], 2),
        _S_SP_DOLLAR,                                           # " $:"
        _uint_to_bytes_left(blstats[BL_GOLD], 6),
        _S_SP_HP,                                               # " HP:"
        _uint_to_bytes_left(blstats[BL_HP], 4),
        _S_OPEN,                                                # "("
        _uint_to_bytes_left(blstats[BL_HPMAX], 4),
        _S_CLOSE_SP_PW,                                         # ") Pw:"
        _uint_to_bytes_left(blstats[BL_ENE], 4),
        _S_OPEN,                                                # "("
        _uint_to_bytes_left(blstats[BL_ENEMAX], 4),
        _S_CLOSE_SP_AC,                                         # ") AC:"
        sign_byte,
        _uint_to_bytes_left(abs_ac, 2),
        _S_SP_XP,                                               # " Xp:"
        _uint_to_bytes_left(blstats[BL_XP], 2),                 # experience level
        _S_SLASH,                                               # "/"
        _uint_to_bytes_left(blstats[BL_EXP], 8),                # experience points
        _S_SP_T,                                                # " T:"
        _uint_to_bytes_left(blstats[BL_TIME], 7),
        build_status_conditions(env_state),
    ]
    row = _compact_pad_spaces(jnp.concatenate(parts), 80)
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

    # Position (col, row).  NLE reports blstats[BL_X] = u.ux - 1 (the hero's
    # INTERNAL column minus 1), because NLE drops unused map column 0 from its
    # observations.  player_pos[1] is the internal column (1..79); subtract 1
    # and clamp so it can't go negative.  BL_Y = row is left unchanged (rows are
    # not dropped; ROWNO=21 maps 1:1).  Cite: vendor/nle/win/rl/winrl.cc.
    result = result.at[BL_X].set(
        jnp.maximum(jnp.int64(env_state.player_pos[1]) - jnp.int64(1), jnp.int64(0))
    )
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

    # Game time (turn counter) — vendor ``moves`` (BL_TIME).  Sourced from
    # game_moves, which advances only on time-consuming actions; a blocked
    # wall-bump takes zero game time and does not tick it.  (Distinct from
    # env_state.timestep, the monotonic per-env-step clock.)
    result = result.at[BL_TIME].set(jnp.int64(env_state.game_moves))

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
    DOORWAY = jnp.int16(int(TileType.DOORWAY))

    # A neighbour counts as a wall-continuation when it is WALL or any door
    # (closed, open, or a doorless DOORWAY).  Vendor check_pos() treats walls +
    # doors as connected segments, so a doorless doorway carved into a wall run
    # still lets the adjacent room corner resolve to the correct corner variant.
    is_wallish = (t == WALL) | (t == CLOSED) | (t == OPEN) | (t == DOORWAY)

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

    # Only rewrite WALL cells with the corner/T-junction variant.
    is_wall_cell = (t == WALL)
    cmap_idx = jnp.where(is_wall_cell, wall_variant, cmap_idx)

    # ---- Door orientation: vendor display.c:1737-1739 picks h*door vs v*door
    # from ``ptr->horizontal`` which is set when the door is embedded in a
    # horizontal wall (mklev.c:163, top/bottom of room).  We derive horizontal
    # at render time from the wall-continuation neighbours computed above:
    # a door is horizontal when its east/west neighbours are wall-ish (i.e.
    # part of an HWALL run) and vertical otherwise.  Mirrors the same vendor
    # rule that ``ptr->horizontal=1`` is set on HWALL cells.
    is_horiz_door = e & w  # E and W neighbours both wall-ish -> horizontal door
    is_closed = (t == CLOSED)
    is_open = (t == OPEN)
    cmap_idx = jnp.where(
        is_closed & is_horiz_door, jnp.int16(_S_hcdoor), cmap_idx,
    )
    cmap_idx = jnp.where(
        is_open & is_horiz_door, jnp.int16(_S_hodoor), cmap_idx,
    )
    return cmap_idx


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
    # NLE glyphs shape: [21, 79].  NLE DROPS NetHack internal column 0 (always
    # blank) — obs column c (0..78) is NetHack internal column c+1 (1..79).  We
    # mirror that by slicing state cols 1..79 (``[:21, 1:80]``) rather than the
    # old cols 0..78 (``[:21, :79]``), which had the whole map shifted +1 right
    # of NLE.  Cite: vendor/nle/win/rl/winrl.cc store_glyph drops levl col 0.
    level_terrain = env_state.terrain[branch, level_idx, :21, 1:80]        # int8[21,79]
    last_seen = env_state.last_seen_terrain[branch, level_idx, :21, 1:80]  # int8[21,79]

    # For explored-but-not-visible tiles use last_seen_terrain; sentinel -1
    # (never seen) falls back to 0 (VOID/stone) which renders as NO_GLYPH anyway.
    visible = env_state.visible[:21, 1:80]                                  # bool[21,79]
    explored = env_state.explored[branch, level_idx, :21, 1:80]            # bool[21,79]

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

    # Unexplored tiles -> cmap_to_glyph(S_stone) (= GLYPH_CMAP_OFF + 0 = 2359),
    # NOT NO_GLYPH.  NLE's RL window seeds the entire `glyphs` obs array with
    # `nul_glyph = cmap_to_glyph(S_stone)` and re-fills any never-seen/background
    # cell with the same value; NO_GLYPH (== MAX_GLYPH == 5976) is reserved for
    # inventory slots and internal sentinels and is never written into the map
    # observation.  Vendor cite: vendor/nle/win/rl/winrl.cc:61
    # (`const int nul_glyph = cmap_to_glyph(S_stone);`) + winrl.cc:250,304,646
    # (`glyphs_.fill(nul_glyph)` / `std::fill_n(obs->glyphs, ..., nul_glyph)`)
    # and vendor/nethack/src/display.c:436
    # (`levl[x][y].glyph = cmap_to_glyph(S_stone); /* default val */`).
    stone_glyph_val = jnp.int16(GLYPH_CMAP_OFF + _S_stone)               # 2359
    glyphs = jnp.where(explored, terrain_glyphs, stone_glyph_val)

    # Overlay ground objects on visible cells (vendor order: terrain → trap →
    # object → monster → player; vendor/nethack/src/display.c::map_location
    # lines 350-430).  Nethax's ground_items stack has slot 0 = TOP of pile
    # (head of vendor's level.objects[x][y] linked list); see
    # Nethax/nethax/subsystems/inventory.py:754-761 (pickup reads slot 0) and
    # the drop scan at lines 1067-1095 (drop fills first-empty slot).  We
    # therefore render slot 0's type_id as the visible object glyph.
    gi_cat0 = env_state.ground_items.category[branch, level_idx, :21, 1:80, 0]   # int8[21,79]
    gi_typ0 = env_state.ground_items.type_id[branch, level_idx, :21, 1:80, 0]    # int16[21,79]
    has_obj = (gi_cat0 != jnp.int8(0)) & visible                                 # bool[21,79]
    obj_glyphs = (gi_typ0.astype(jnp.int32) + jnp.int32(GLYPH_OBJ_OFF)).astype(jnp.int16)
    glyphs = jnp.where(has_obj, obj_glyphs, glyphs)

    # Overlay live monsters at their tile positions.  Each visible, alive
    # monster slot writes GLYPH_MON_OFF + entry_idx at its (row, col).
    # Vendor reference: display.c::show_glyph; mhitu.c writes monster glyph
    # via map_location each turn.
    mai = env_state.monster_ai
    mon_pos = mai.pos                          # int16[N, 2]
    mon_alive = mai.alive                      # bool[N]
    mon_entry = mai.entry_idx.astype(jnp.int32)  # int32[N]

    rows = jnp.clip(mon_pos[:, 0].astype(jnp.int32), 0, 20)
    # Monster positions are in STATE column space (0..79).  The obs map drops
    # internal column 0, so obs_col = state_col - 1 (0..78).  Monsters sitting
    # on internal column 0 fall off the left edge of the obs map and are not
    # drawn (mon_oncol0 mask below).
    state_cols = mon_pos[:, 1].astype(jnp.int32)
    mon_oncol0 = state_cols <= jnp.int32(0)
    cols = jnp.clip(state_cols - jnp.int32(1), 0, 78)
    # Only overlay monsters that are alive AND on visible tiles.
    tile_visible = visible[rows, cols]
    write_mask = (
        mon_alive & tile_visible & (mon_entry >= jnp.int32(0)) & (~mon_oncol0)
    )
    # Tame monsters render at GLYPH_PET_OFF + entry, NOT GLYPH_MON_OFF.
    # Vendor display.c:599-603: `if (mon->mtame && !Hallucination) num =
    # pet_to_glyph(mon, ...)`, and pet_to_glyph adds GLYPH_PET_OFF.  Hostile /
    # peaceful (non-tame) monsters use the plain GLYPH_MON_OFF base.  The
    # Hallucination branch below overrides both with random GLYPH_MON_OFF
    # glyphs (vendor skips pet_to_glyph when Hallu), so only the non-hallu
    # base distinguishes pets here.
    mon_tame = mai.tame                                            # bool[N]
    mon_base = jnp.where(mon_tame, jnp.int32(GLYPH_PET_OFF),
                         jnp.int32(GLYPH_MON_OFF))                 # int32[N]
    mon_glyphs = (mon_base + mon_entry).astype(jnp.int16)

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
    # The player's display glyph is `u.umonnum`, which (when not polymorphed)
    # vendor sets to the ROLE's player-monster `urole.malenum`, NOT the race.
    # Cite: vendor/nle/src/u_init.c:628
    #   `u.umonnum = u.umonster = ... urole.malenum;`
    # and vendor/nethack/src/display.c::display_self / show_glyph which renders
    # GLYPH_MON_OFF + u.umonnum.  Each role's malenum is a distinct entry in the
    # mons[] table (e.g. PM_ROGUE -> mons index 337 "rogue", PM_VALKYRIE -> 340),
    # so a Human Rogue renders glyph 337 — the player-monster, not "human" (256).
    player_row = jnp.int32(env_state.player_pos[0])
    player_col = jnp.int32(env_state.player_pos[1])
    # player_col is a STATE column (1..79); the obs map drops internal column 0,
    # so the obs column is player_col - 1.  Clamp to [0,78] (glyphs is 79 wide).
    player_col_clamped = jnp.clip(player_col - jnp.int32(1), 0, 78)

    # Role -> player-monster index in the NLE mons[] table.  Order matches the
    # Role enum (ARCHEOLOGIST=0 .. WIZARD=12).  Values are the `urole.malenum`
    # monster indices verified against NLE permonst() by name.
    #   ARCHEOLOGIST=327 BARBARIAN=328 CAVEMAN=329 HEALER=331 KNIGHT=332
    #   MONK=333 PRIEST=334 RANGER=336 ROGUE=337 SAMURAI=338 TOURIST=339
    #   VALKYRIE=340 WIZARD=341
    _ROLE_TO_MON_IDX = jnp.array(
        [327, 328, 329, 331, 332, 333, 334, 336, 337, 338, 339, 340, 341],
        dtype=jnp.int32,
    )
    role_idx = jnp.clip(jnp.int32(env_state.player_role), 0, 12)
    base_mon = _ROLE_TO_MON_IDX[role_idx]

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

    # Final pass: apply the vendor description shuffle so any object
    # glyph in the map is mapped through ``descr_idx`` to its shuffled
    # appearance.  Identity-permutation in default ParityMode.NLE,
    # non-trivial under NLE_BYTEPARITY.  Cite:
    # vendor/nle/win/rl/winrl.cc::shuffled_glyph (lines 80-87) +
    # store_glyph (line 473) — every map glyph emit passes through
    # shuffled_glyph before the obs array is written.
    from Nethax.nethax.obs.glyph_shuffle import shuffled_glyph as _shuffled_glyph
    glyphs = _shuffled_glyph(glyphs, env_state.descr_idx)

    return glyphs


def build_message(env_state) -> jnp.ndarray:
    """Return the current message buffer as a 256-byte uint8 array.

    The internal message_buffer layout (messages.py::_bake_templates) stores
    the msg_id integer at byte 0 and the ASCII text at bytes 1..255.  NLE's
    ``message`` observation (vendor/nle/include/nleobs.h) is pure ASCII from
    byte 0 — no leading msg_id byte.  We therefore strip byte 0 and return
    bytes 1..256 as the 256-byte obs, matching NLE exactly.

    Cite: vendor/nle/include/nleobs.h — message field is char[256] raw text;
          vendor/nle/win/rl/pynethack.cc — copies WIN_MESSAGE text verbatim.

    Returns:
        uint8[256]
    """
    buf = env_state.messages.message_buffer  # uint8[256], byte 0 = msg_id
    # Skip the msg_id prefix byte; text occupies bytes 1..255 (255 usable chars).
    # Append one zero byte so the result is exactly 256 bytes wide.
    text = buf[1:256]                                          # uint8[255]
    return jnp.concatenate(
        [text, jnp.zeros((1,), dtype=jnp.uint8)]
    ).astype(jnp.uint8)                                       # uint8[256]


def _build_tty_chars(env_state) -> jnp.ndarray:
    """Render the 24x80 TTY char grid (no colors / cursor).

    Shared by build_tty (which adds cursor + colors) and build_tty_colors
    (which masks colors to 0 on blank cells).  Kept separate to avoid a
    build_tty <-> build_tty_colors recursion.

    Layout:
      Row 0:     message line (message_buffer[1:81], msg_id byte stripped)
      Rows 1-21: map area (glyphs converted to ASCII chars)
      Row 22-23: status lines (botl do_statusline1 / do_statusline2)

    Returns:
        uint8[24, 80]
    """
    # NLE's libtmt virtual terminal initialises every cell to ASCII space
    # (0x20) — see vendor/nle/src/nle.c::nle_vt_callback (TMT_MSG_UPDATE) and
    # the TMT default cell init.  Using 0x00 here would make every unwritten
    # cell byte-mismatch vs NLE.  Cite: TTY_LAYOUT_DIFF.md D1.
    tty = jnp.full((24, 80), jnp.uint8(ord(' ')), dtype=jnp.uint8)

    # --- Row 0: message line ---
    # message_buffer byte 0 holds the msg_id sentinel (messages.py: "Byte 0 is
    # msg_id; bytes 1.. hold the rendered ASCII line").  NLE's tty row 0 is the
    # raw toplines text starting at column 0, with NO leading sentinel — so we
    # skip byte 0 and copy bytes 1..80 (same stripping build_message does).
    # Previously we copied [:80] including the msg_id byte, which shifted the
    # whole message one column right and left a stray glyph at tty col 0.
    # message_buffer is zero-padded after the message text; vendor outputs
    # ASCII space in the tail (terminal default), so rewrite NULs to space.
    # Cite: TTY_LAYOUT_DIFF.md D2; subsystems/messages.py:339-340.
    msg = env_state.messages.message_buffer[1:81].astype(jnp.uint8)
    msg = jnp.where(msg == jnp.uint8(0), jnp.uint8(ord(' ')), msg)
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
    return tty


def build_tty(env_state) -> dict[str, jnp.ndarray]:
    """Render the 24x80 TTY terminal grid.

    Layout:
      Row 0:     message line (message_buffer[1:81], msg_id byte stripped)
      Rows 1-21: map area (glyphs converted to ASCII chars)
      Row 22:    status line 1 (St/Dx/Co/In/Wi/Ch  Dlvl:n  HP  Pw  AC  XP)
      Row 23:    status line 2 (Dlvl: n  T: n)

    Cursor is at player position offset by +1 row (message row 0 occupies row 0).

    Returns:
        dict with keys:
          tty_chars  : uint8[24, 80]
          tty_colors : int8[24, 80]
          tty_cursor : uint8[2]       (row, col)

    JIT-compatible: all operations use jnp.where / at[].set().
    """
    tty = _build_tty_chars(env_state)

    # --- Cursor: row = player_row + 1 (offset for the message line at row 0),
    # col = player_x - 1.  Vendor places the tty cursor at (player_y + 1,
    # player_x - 1) because the tty windowport decrements x by one (tty_curs
    # --x) so internal column 1 lands at tty column 0.  player_pos[1] is the
    # internal column (1..79).  Cite: TTY_LAYOUT_DIFF.md Cursor + D8.
    player_row = jnp.uint8(jnp.clip(env_state.player_pos[0], 0, 20) + 1)
    player_col = jnp.uint8(
        jnp.clip(jnp.int32(env_state.player_pos[1]) - jnp.int32(1), 0, 78)
    )

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
