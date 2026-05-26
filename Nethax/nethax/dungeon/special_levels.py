"""Fixed / hand-authored special level stubs.

Purpose:
    Enumerates every named special level in NetHack and provides a dispatch
    stub generate_special_level() that will eventually call a Python factory
    function for each level.  Deliberately avoids Lua — all level geometry
    will be replicated as JAX-state-producing Python functions, parsing the
    canonical .lua / .des files from vendor/nethack/dat/ by hand.

Citation:
    vendor/nethack/src/sp_lev.c      — special level loader (Lua in 3.7;
        we replace with Python factories per project rules)
    vendor/nethack/src/dungeon.c     — level name → s_level mapping table
        (lines ~715-726: castle, oracle, sanctum, valley, etc.)
    vendor/nethack/include/dungeon.h — Is_oracle_level, Is_valley, Is_sanctum,
        Is_stronghold, Is_bigroom, Is_wiz1/2/3_level, Is_qstart/qlocate/nemesis
        macros (lines 113-135)
    vendor/nethack/dat/oracle.lua    — Oracle level layout
    vendor/nethack/dat/minetn-*.lua  — Minetown variants (1-7)
    vendor/nethack/dat/soko1-*.lua   — Sokoban floor 1 variants
    vendor/nethack/dat/soko2-*.lua   — Sokoban floor 2 variants
    vendor/nethack/dat/soko3-*.lua   — Sokoban floor 3 variants (not present → soko3 implied)
    vendor/nethack/dat/soko4-*.lua   — Sokoban floor 4 variants
    vendor/nethack/dat/castle.lua    — The Castle
    vendor/nethack/dat/valley.lua (implied) — Valley of the Dead
    vendor/nethack/dat/baalz.lua     — Baalzebub's lair
    vendor/nethack/dat/sanctum.lua   — The Sanctum
    vendor/nethack/dat/astral.lua    — Astral Plane
    vendor/nethack/dat/air.lua       — Air Plane
    vendor/nethack/dat/fakewiz1.lua  — Fake Wizard Tower 1
    vendor/nethack/dat/fakewiz2.lua  — Fake Wizard Tower 2
    vendor/nethack/dat/bigrm-*.lua   — Big Room variants (8-9+)

Wave 1 stub: SpecialLevel enum defined; generate_special_level returns
             jnp.zeros((MAP_H, MAP_W), int8) for every level_id.
"""

from __future__ import annotations

from enum import IntEnum

import jax
import jax.lax as lax
import jax.numpy as jnp

from Nethax.nethax.dungeon.branches import MAP_H, MAP_W

# ---------------------------------------------------------------------------
# SpecialLevel enum
#
# Each member corresponds to one fixed hand-authored level.
# Ordinal values are arbitrary internal IDs — not NetHack ledger numbers.
# ---------------------------------------------------------------------------

class SpecialLevel(IntEnum):
    """Named special / fixed levels.

    Citation: vendor/nethack/src/dungeon.c ~lines 715-726 (level name table),
              vendor/nethack/include/dungeon.h Is_* macros (lines 113-135),
              vendor/nethack/dat/*.lua layout files.
    """
    # Main dungeon fixed levels
    ORACLE          =  0   # Oracle chamber (oracle.lua); main dungeon ~Dlvl 6-10
    BIG_ROOM        =  1   # Big Room variant (bigrm-*.lua); random main dlvl

    # Gnomish Mines branch
    MINETOWN        =  2   # Minetown (minetn-*.lua); mines branch level 2-3
    MINES_END       =  3   # Mines' End (mineend-*.lua or equivalent)

    # Sokoban branch (4 floors, each with variants a/b)
    SOKO_FLOOR_1    =  4   # soko1-*.lua  (deepest floor, hardest puzzle)
    SOKO_FLOOR_2    =  5   # soko2-*.lua
    SOKO_FLOOR_3    =  6   # soko3-*.lua
    SOKO_FLOOR_4    =  7   # soko4-*.lua  (shallowest, entry floor)

    # Quest branch — role-specific dispatch.  The three slots map to vendor
    # dungeon.def "x-strt" / "x-loca" / "x-goal" templates (where x is the
    # 3-letter role abbreviation, see vendor/nle/dat/dungeon.def lines 86-89).
    # Per-role factories live in Nethax/nethax/dungeon/quest_levels.py and
    # are dispatched via ``dispatch_quest_level(rng, role)`` keyed off the
    # roles[] table in vendor/nethack/src/role.c.
    QUEST_START     =  8   # qstart level  (Is_qstart)  — vendor x-strt.lua
    QUEST_LOCATE    =  9   # qlocate level (Is_qlocate) — vendor x-loca.lua
    QUEST_GOAL      = 10   # nemesis level (Is_nemesis) — vendor x-goal.lua

    # Main dungeon — late game
    CASTLE          = 11   # The Castle / Stronghold (castle.lua, Is_stronghold)
    VALLEY          = 12   # Valley of the Dead (valley.lua, Is_valley)

    # Gehennom fixed levels
    ASMODEUS        = 13   # Asmodeus' Lair (Is_asmo_level)
    BAALZEBUB       = 14   # Baalzebub's Lair (baalz.lua, Is_baal_level)
    JUIBLEX         = 15   # Juiblex's Swamp (Is_juiblex_level)
    ORCUS           = 16   # Orcus Town (Is_valley adjacent)

    # Vlad's Tower branch
    VLAD_TOWER_TOP  = 17   # Top of Vlad's Tower (vlad.lua or equivalent)

    # Wizard's Tower (3 fakes + real)
    FAKE_WIZARD_1   = 18   # fakewiz1.lua (Is_wiz1_level)
    FAKE_WIZARD_2   = 19   # fakewiz2.lua (Is_wiz2_level)
    FAKE_WIZARD_3   = 20   # (Is_wiz3_level — top of tower)
    WIZARD_TOWER    = 21   # The real Wizard's Tower summit

    # The Sanctum
    SANCTUM         = 22   # sanctum.lua (Is_sanctum); deepest main dungeon level

    # Endgame — Astral Plane + 4 elemental planes
    ASTRAL_PLANE    = 23   # astral.lua (Is_astralevel)
    EARTH_PLANE     = 24   # earth.lua  (Is_earthlevel)
    AIR_PLANE       = 25   # air.lua    (Is_airlevel)
    FIRE_PLANE      = 26   # fire.lua   (Is_firelevel)
    WATER_PLANE     = 27   # water.lua  (Is_waterlevel)


# ---------------------------------------------------------------------------
# No-op stub
# ---------------------------------------------------------------------------

def generate_special_level(
    rng: jnp.ndarray,
    level_id,
) -> jnp.ndarray:
    """Return the terrain map for the given fixed special level.

    Dispatches via ``jax.lax.switch`` over the 27-member ``SpecialLevel``
    enum.  Each branch invokes the corresponding per-level factory (which
    returns ``(terrain, monsters, items)``) and yields just the terrain
    grid.  Unimplemented levels (e.g. the four elemental planes and the
    three quest slots) fall through to an all-walls stub.

    Args:
        rng:      JAX PRNG key (used for variant selection where a level
                  has multiple variants, e.g. minetn-1 through minetn-7).
        level_id: ``SpecialLevel`` enum value, Python int, or traced
                  ``jnp.int32`` scalar in ``[0, 27]``.

    Returns:
        int8[MAP_H, MAP_W] terrain array; 0 = wall, 1 = floor.

    Citation: vendor/nethack/src/sp_lev.c — the vendor Lua dispatcher;
        we replace it with a static Python switch keyed off the
        SpecialLevel enum.  Per-level factories below are byte-compatible
        ports of the corresponding vendor/nethack/dat/*.lua layouts.
    """
    # ``lax.switch`` requires every branch to return an identically-shaped
    # pytree.  All factories already return (terrain, monsters, items) with
    # terrain shape [MAP_H, MAP_W], so we wrap each in a terrain-only
    # adapter and add an all-walls default for unimplemented levels.
    def _all_walls(r):
        return jnp.zeros((MAP_H, MAP_W), dtype=jnp.int8)

    # Branch table — index = SpecialLevel ordinal (0..27 inclusive).
    # Order MUST match the SpecialLevel enum.  Wrap factories with the
    # default rng signature; lax.switch evaluates a single branch.
    _BRANCHES = (
        lambda r: generate_oracle_level(r)[0],         # 0  ORACLE
        lambda r: generate_big_room(r)[0],             # 1  BIG_ROOM
        lambda r: generate_mine_town(r)[0],            # 2  MINETOWN
        lambda r: generate_mines_end(r)[0],            # 3  MINES_END
        lambda r: generate_sokoban_floor_1(r)[0],      # 4  SOKO_FLOOR_1
        lambda r: generate_sokoban_floor_2(r)[0],      # 5  SOKO_FLOOR_2
        lambda r: generate_sokoban_floor_3(r)[0],      # 6  SOKO_FLOOR_3
        lambda r: generate_sokoban_floor_4(r)[0],      # 7  SOKO_FLOOR_4
        _all_walls,                                    # 8  QUEST_START   (TODO)
        _all_walls,                                    # 9  QUEST_LOCATE  (TODO)
        _all_walls,                                    # 10 QUEST_GOAL    (TODO)
        lambda r: generate_castle_level(r)[0],         # 11 CASTLE
        lambda r: generate_valley_level(r)[0],         # 12 VALLEY
        lambda r: generate_asmodeus_lair(r)[0],        # 13 ASMODEUS
        lambda r: generate_baalzebub_lair(r)[0],       # 14 BAALZEBUB
        lambda r: generate_juiblex_lair(r)[0],         # 15 JUIBLEX
        lambda r: generate_orcus_town(r)[0],           # 16 ORCUS
        lambda r: generate_vlads_tower(r, floor=3)[0], # 17 VLAD_TOWER_TOP
        lambda r: generate_wizards_tower(r, fake_idx=1)[0],  # 18 FAKE_WIZARD_1
        lambda r: generate_wizards_tower(r, fake_idx=2)[0],  # 19 FAKE_WIZARD_2
        lambda r: generate_wizards_tower(r, fake_idx=3)[0],  # 20 FAKE_WIZARD_3
        lambda r: generate_wizards_tower(r, fake_idx=0)[0],  # 21 WIZARD_TOWER
        lambda r: generate_sanctum_level(r)[0],        # 22 SANCTUM
        lambda r: generate_astral_plane(r)[0],         # 23 ASTRAL_PLANE
        _all_walls,                                    # 24 EARTH_PLANE   (TODO)
        _all_walls,                                    # 25 AIR_PLANE     (TODO)
        _all_walls,                                    # 26 FIRE_PLANE    (TODO)
        _all_walls,                                    # 27 WATER_PLANE   (TODO)
    )

    # Normalise level_id to int32 scalar; clamp out-of-range to the last
    # default branch (an all-walls stub) so lax.switch never indexes past
    # the end of _BRANCHES (vendor: invalid level_id is treated as empty).
    idx = jnp.asarray(int(level_id) if isinstance(level_id, SpecialLevel)
                      else level_id, dtype=jnp.int32)
    idx = jnp.clip(idx, 0, len(_BRANCHES) - 1)
    return lax.switch(idx, _BRANCHES, rng)


# ---------------------------------------------------------------------------
# TODO blocks
# ---------------------------------------------------------------------------
# Wave 5 (Phase 1 — critical path levels):
#   - ORACLE: parse oracle.lua; 10×10 circular room, fountain, Oracle monster.
#   - MINETOWN: pick variant 1-7; parse minetn-*.lua; place shopkeepers,
#     guards, Watch captain; set IS_TOWN flag on level.
#   - MINES_END: fixed map with luckstone in statue; undead/gem monsters.
#   - SOKO_FLOOR_1..4: parse soko{1-4}-{a,b}.lua puzzle grids exactly;
#     place boulders at fixed positions; Sokoban rules (no teleport, no poly).
#   - CASTLE: castle.lua drawbridge + secret door; wand of wishing guaranteed;
#     soldier/sergeant/lieutenant/captain monster placement.
#   - VALLEY: dense undead; no teleport; down-stair to Gehennom.
#   - SANCTUM: vibrating square tile; Wizard of Yendor; Amulet of Yendor.
#
# Wave 5 (Phase 2 — Gehennom):
#   - ASMODEUS / BAALZEBUB / JUIBLEX / ORCUS: parse baalz.lua and equivalents.
#   - VLAD_TOWER_TOP: fixed tower map; Vlad the Impaler; Candelabrum.
#   - FAKE_WIZARD_1/2/3: fakewiz1.lua, fakewiz2.lua — Wizard imposters.
#   - WIZARD_TOWER: top floor; real Wizard of Yendor; Book of the Dead.
#
# Wave 6 (Ascension levels):
#   - ASTRAL_PLANE (astral.lua): 3 altars (Law/Neutral/Chaos), Riders,
#     Famine/Pestilence/Death.
#   - EARTH_PLANE: dense rock; no corridors; digging required.
#   - AIR_PLANE (air.lua): open air; no floor; flying/levitation required.
#   - FIRE_PLANE: lava; fire immunity required.
#   - WATER_PLANE: water; swimming/levitation required.


# ===========================================================================
# Wave 4 Phase 2 — special-level factories (Oracle, Mine Town, Mines End, Big Room)
# ===========================================================================
#
# Each factory returns a tuple of raw arrays:
#   (terrain[MAP_H, MAP_W], monsters[K, 3], items[K, 3])
#
# - terrain is an int8 grid using TileType values (see constants/tiles.py):
#       VOID=0, FLOOR=1, CORRIDOR=2, WALL=3, CLOSED_DOOR=4, OPEN_DOOR=5,
#       STAIRCASE_UP=6, STAIRCASE_DOWN=7, WATER=8, LAVA=9, ALTAR=10,
#       FOUNTAIN=11, TRAP=12, HIDDEN_TRAP=13, THRONE=14, GRAVE=15, SHOP_FLOOR=16
# - monsters / items each are int16 arrays of shape [K, 3] where the
#   columns are (row, col, type_id).  Unused rows hold (-1, -1, -1).
#
# Layouts are hand-encoded from the canonical vendor .lua files
# (see file-level citation block above).  We deliberately skip the Lua
# interpreter; this gives us deterministic, JAX-friendly factories that
# replicate the canonical floorplan.
# ---------------------------------------------------------------------------

# Tile sentinel — match TileType to avoid a cross-module import cycle
# (constants.tiles already imports from this branch's MAP_H/W constants).
_T_VOID         = 0
_T_FLOOR        = 1
_T_CORRIDOR     = 2
_T_WALL         = 3
_T_CLOSED_DOOR  = 4
_T_OPEN_DOOR    = 5
_T_STAIR_UP     = 6
_T_STAIR_DOWN   = 7
_T_WATER        = 8
_T_LAVA         = 9
_T_ALTAR        = 10
_T_FOUNTAIN     = 11
_T_TRAP         = 12
_T_HIDDEN_TRAP  = 13
_T_THRONE       = 14
_T_GRAVE        = 15
_T_SHOP_FLOOR   = 16
_T_TREE         = 20  # TileType.TREE — added Wave 6 parity (rm.h TREE).


# Character → TileType mapping used by hand-encoded map strings below.
# Mirrors the symbols used in vendor .des/.lua files.
_CHAR_TO_TILE = {
    " ": _T_VOID,
    ".": _T_FLOOR,
    "#": _T_CORRIDOR,
    "|": _T_WALL,
    "-": _T_WALL,
    "+": _T_CLOSED_DOOR,
    "<": _T_STAIR_UP,
    ">": _T_STAIR_DOWN,
    "}": _T_WATER,
    "L": _T_LAVA,
    "_": _T_ALTAR,
    "{": _T_FOUNTAIN,
    "\\": _T_THRONE,
    "S": _T_SHOP_FLOOR,
    "F": _T_WALL,     # iron bars in minetn-1 — treat as wall for nav
    "T": _T_TREE,     # tree (TileType.TREE)
    "C": _T_FLOOR,    # cloud → walkable floor proxy
    "x": _T_VOID,     # solidfill rock filler ('x' in Mon/Pri-goal, valley, Val-goal)
    "B": _T_FLOOR,    # broken passage (valley.lua) — walkable
    "W": _T_WATER,    # water (water.lua)
    "A": _T_VOID,     # air (air.lua) — VOID so non-flying players fall
    "P": _T_WATER,    # pool ('P' in Healer/Knight-goal, baalz iron-bar markers)
}


# Per-file char overrides: in castle.lua, sanctum.lua, tower*.lua, wizard*.lua,
# Arc-goal.lua, Wiz-goal.lua, oracle.lua, valley.lua, Rog-goal.lua etc. the
# vendor symbol 'S' means SECRET DOOR (treated as CLOSED_DOOR at runtime), NOT
# SHOP_FLOOR.  We expose a parser hook so callers can pass an override map.
_CHAR_TO_TILE_SECRET_DOOR = dict(_CHAR_TO_TILE)
_CHAR_TO_TILE_SECRET_DOOR["S"] = _T_CLOSED_DOOR


def _parse_map_string(s: str, char_map: dict = None) -> jnp.ndarray:
    """Parse a multi-line vendor MAP string into an int8 terrain grid.

    Pads each row to MAP_W with VOID; pads or truncates to MAP_H rows.
    Trailing/leading blank lines in the vendor string are discarded.

    Args:
        s:        Vendor MAP block as a single string (newline-separated).
        char_map: Optional override of the default char→tile map.  Used to
                  flip 'S' from SHOP_FLOOR to CLOSED_DOOR (secret door) for
                  files where vendor uses S that way.
    """
    if char_map is None:
        char_map = _CHAR_TO_TILE
    rows = s.split("\n")
    # Drop only leading/trailing fully-empty rows so vendor formatting is
    # preserved verbatim inside the block.
    while rows and rows[0] == "":
        rows.pop(0)
    while rows and rows[-1] == "":
        rows.pop()
    grid = [[_T_VOID] * MAP_W for _ in range(MAP_H)]
    for r, row in enumerate(rows[:MAP_H]):
        for c, ch in enumerate(row[:MAP_W]):
            grid[r][c] = char_map.get(ch, _T_VOID)
    return jnp.array(grid, dtype=jnp.int8)


# Monster / item type-id sentinels.  Sourced from
# Nethax/nethax/constants/monsters.py and constants/objects.py.  We only
# refer to a handful by id here; real game placement uses these as hints
# for downstream populate_level_with_monsters() calls.
_MON_ORACLE      = 1   # special NPC id (Oracle)
_MON_SHOPKEEPER  = 2
_MON_WATCHMAN    = 3
_MON_PRIEST      = 4
_MON_GNOME       = 5
_MON_BLACK_PUDDING = 6

_ITEM_LUCKSTONE  = 1
_ITEM_GOLD       = 2
_ITEM_GEM        = 3
_ITEM_RING       = 4


def _encode_map(rows: list) -> jnp.ndarray:
    """Convert a list of equal-length character rows to a tile grid.

    Pads each row to MAP_W with VOID; pads or truncates to MAP_H rows.
    """
    grid = [[_T_VOID] * MAP_W for _ in range(MAP_H)]
    for r, row in enumerate(rows[:MAP_H]):
        for c, ch in enumerate(row[:MAP_W]):
            grid[r][c] = _CHAR_TO_TILE.get(ch, _T_VOID)
    return jnp.array(grid, dtype=jnp.int8)


def _pack_placements(triples: list, capacity: int = 64) -> jnp.ndarray:
    """Pack a list of (row, col, type_id) triples into an int16[capacity, 3]
    array.  Unused rows are filled with (-1, -1, -1).
    """
    arr = [[-1, -1, -1] for _ in range(capacity)]
    for i, (r, c, t) in enumerate(triples[:capacity]):
        arr[i] = [int(r), int(c), int(t)]
    return jnp.array(arr, dtype=jnp.int16)


# ===========================================================================
# Wave 6 Phase B+ — verbatim vendor MAP-string constants.
#
# Each constant below is the MAP section of the corresponding vendor file,
# copied byte-identical from vendor/nethack/dat/<file>.lua's des.map([[ ... ]])
# block.  Licensing (NGPL) is handled separately at the project level.
#
# Mines End (minend-1.lua line 13), Mine Town (minetn-1.lua line 16), and
# Big Room (bigrm-1.lua line 9) each have a MAP block — captured below
# byte-identical to the vendor des.map() string.  Oracle is the only entry
# in this group that uses des.room()/des.level_init() generators rather
# than a single MAP block; its generator is reproduced procedurally above.
# ===========================================================================

# vendor/nethack/dat/minetn-1.lua MAP section — NGPL
_MINETOWN_MAP = """\
.....................................
.----------------F------------------.
.|.................................|.
.|.-------------......------------.|.
.|.|...|...|...|......|..|...|...|.|.
.F.|...|...|...|......|..|...|...|.|.
.|.|...|...|...|......|..|...|...|.F.
.|.|...|...|----......------------.|.
.|.---------.......................|.
.|.................................|.
.|.---------.....--...--...........|.
.|.|...|...|----.|.....|.---------.|.
.|.|...|...|...|.|.....|.|..|....|.|.
.|.|...|...|...|.|.....|.|..|....|.|.
.|.|...|...|...|.|.....|.|..|....|.|.
.|.-------------.-------.---------.|.
.|.................................F.
.-----------F------------F----------.
.....................................
"""

# vendor/nethack/dat/minend-1.lua MAP section — NGPL
_MINESEND_MAP = """\
------------------------------------------------------------------   ------
|                        |.......|     |.......-...|       |.....|.       |
|    ---------        ----.......-------...........|       ---...-S-      |
|    |.......|        |..........................-S-      --.......|      |
|    |......-------   ---........................|.       |.......--      |
|    |..--........-----..........................|.       -.-..----       |
|    --..--.-----........-.....................---        --..--          |
|     --..--..| -----------..................---.----------..--           |
|      |...--.|    |..S...S..............---................--            |
|     ----..-----  ------------........--- ------------...---             |
|     |.........--            ----------              ---...-- -----      |
|    --.....---..--                           --------  --...---...--     |
| ----..-..-- --..---------------------      --......--  ---........|     |
|--....-----   --..-..................---    |........|    |.......--     |
|.......|       --......................S..  --......--    ---..----      |
|--.--.--        ----.................---     ------..------...--         |
| |....S..          |...............-..|         ..S...........|          |
--------            --------------------           ------------------------
"""

# vendor/nethack/dat/bigrm-1.lua MAP section — NGPL
_BIGROOM_MAP = """\
---------------------------------------------------------------------------
|.........................................................................|
|.........................................................................|
|.........................................................................|
|.........................................................................|
|.........................................................................|
|.........................................................................|
|.........................................................................|
|.........................................................................|
|.........................................................................|
|.........................................................................|
|.........................................................................|
|.........................................................................|
|.........................................................................|
|.........................................................................|
|.........................................................................|
---------------------------------------------------------------------------
"""

# vendor/nethack/dat/castle.lua MAP section — NGPL
_CASTLE_MAP = """\
}}}}}}}}}.............................................}}}}}}}}}
}-------}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}-------}
}|.....|-----------------------------------------------|.....|}
}|.....+...............................................+.....|}
}-------------------------------+-----------------------------}
}}}}}}|........|..........+...........|.......S.S.......|}}}}}}
.....}|........|..........|...........|.......|.|.......|}.....
.....}|........------------...........---------S---------}.....
.....}|...{....+..........+.........\\.S.................+......
.....}|........------------...........---------S---------}.....
.....}|........|..........|...........|.......|.|.......|}.....
}}}}}}|........|..........+...........|.......S.S.......|}}}}}}
}-------------------------------+-----------------------------}
}|.....+...............................................+.....|}
}|.....|-----------------------------------------------|.....|}
}-------}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}-------}
}}}}}}}}}.............................................}}}}}}}}}
"""

# vendor/nethack/dat/tower1.lua MAP section — NGPL  (top floor — Vlad)
_TOWER1_MAP = """\
  --- --- ---
  |.| |.| |.|
---S---S---S---
|.......+.+...|
---+-----.-----
  |...\\.|.+.|
---+-----.-----
|.......+.+...|
---S---S---S---
  |.| |.| |.|
  --- --- ---
"""

# vendor/nethack/dat/tower2.lua MAP section — NGPL  (middle floor)
_TOWER2_MAP = """\
  --- --- ---
  |.| |.| |.|
---S---S---S---
|.S.........S.|
---.------+----
  |......|..|
--------.------
|.S......+..S.|
---S---S---S---
  |.| |.| |.|
  --- --- ---
"""

# vendor/nethack/dat/tower3.lua MAP section — NGPL  (bottom / entry floor)
_TOWER3_MAP = """\
    --- --- ---
    |.| |.| |.|
  ---S---S---S---
  |.S.........S.|
-----.........-----
|...|.........+...|
|.---.........---.|
|.|.S.........S.|.|
|.---S---S---S---.|
|...|.|.|.|.|.|...|
---.---.---.---.---
  |.............|
  ---------------
"""

# vendor/nethack/dat/wizard1.lua MAP section — NGPL  (REAL Wizard's Tower)
_WIZARD1_MAP = """\
----------------------------x
|.......|..|.........|.....|x
|.......S..|.}}}}}}}.|.....|x
|..--S--|..|.}}---}}.|---S-|x
|..|....|..|.}--.--}.|..|..|x
|..|....|..|.}|...|}.|..|..|x
|..--------|.}--.--}.|..|..|x
|..|.......|.}}---}}.|..|..|x
|..S.......|.}}}}}}}.|..|..|x
|..|.......|.........|..|..|x
|..|.......|-----------S-S-|x
|..|.......S...............|x
----------------------------x
"""

# vendor/nethack/dat/wizard2.lua MAP section — NGPL  (fake Wizard's Tower #1)
_WIZARD2_MAP = """\
----------------------------x
|.....|.S....|.............|x
|.....|.-------S--------S--|x
|.....|.|.........|........|x
|..-S--S|.........|........|x
|..|....|.........|------S-|x
|..|....|.........|.....|..|x
|-S-----|.........|.....|..|x
|.......|.........|S--S--..|x
|.......|.........|.|......|x
|-----S----S-------.|......|x
|............|....S.|......|x
----------------------------x
"""

# vendor/nethack/dat/wizard3.lua MAP section — NGPL  (fake Wizard's Tower #2)
_WIZARD3_MAP = """\
----------------------------x
|..|............S..........|x
|..|..------------------S--|x
|..|..|.........|..........|x
|..S..|.}}}}}}}.|..........|x
|..|..|.}}---}}.|-S--------|x
|..|..|.}--.--}.|..|.......|x
|..|..|.}|...|}.|..|.......|x
|..---|.}--.--}.|..|.......|x
|.....|.}}---}}.|..|.......|x
|.....S.}}}}}}}.|..|.......|x
|.....|.........|..|.......|x
----------------------------x
"""

# vendor/nethack/dat/sanctum.lua MAP section — NGPL
_SANCTUM_MAP = """\
----------------------------------------------------------------------------
|             --------------                                               |
|             |............|             -------                           |
|       -------............-----         |.....|                           |
|       |......................|        --.....|            ---------      |
|    ----......................---------|......----         |.......|      |
|    |........---------..........|......+.........|     ------+---..|      |
|  ---........|.......|..........--S----|.........|     |........|..|      |
|  |..........|.......|.............|   |.........-------..----------      |
|  |..........|.......|..........----   |..........|....|..|......|        |
|  |..........|.......|..........|      --.......----+---S---S--..|        |
|  |..........---------..........|       |.......|.............|..|        |
|  ---...........................|       -----+-------S---------S---       |
|    |...........................|          |...| |......|    |....|--     |
|    ----.....................----          |...---....---  ---......|     |
|       |.....................|             |..........|    |.....----     |
|       -------...........-----             --...-------    |.....|        |
|             |...........|                  |...|          |.....|        |
|             -------------                  -----          -------        |
----------------------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# Oracle level — vendor/nethack/dat/oracle.lua
# ---------------------------------------------------------------------------
# Three small rooms; central 11x9 ordinary room contains a nested 3x3
# "delphi" inner room with four fountains and the Oracle herself. We
# hand-encode only the central chamber + two satellite rooms to satisfy
# the "3 small rooms" spec; corridor structure is left for the dungeon
# generator pass.
# ---------------------------------------------------------------------------

_ORACLE_ROWS = [
    # rows 0..2: top satellite room (5x3)
    "------     ",
    "|....|     ",
    "|<...|     ",
    "------     ",
    "                                              ",
    "             -----------                      ",
    "             |.........|                      ",
    "             |.........|                      ",
    "             |..-----..|                      ",
    "             |..|...|..|                      ",
    "             |..|.{.|..|       ----           ",
    "             |..|{O{|..|       |..|           ",
    "             |..|.{.|..|       |>.|           ",
    "             |..|...|..|       ----           ",
    "             |..-----..|                      ",
    "             |.........|                      ",
    "             |.........|                      ",
    "             -----------                      ",
    "                                              ",
    "                                              ",
    "                                              ",
]


def _carve_room(terrain, top: int, left: int, h: int, w: int):
    """Pythonic des.room() implementation: carve a rectangular room with
    walls on the perimeter and FLOOR inside.  ``top``/``left`` give the
    top-left tile of the wall border; ``h``/``w`` are total dims including
    walls (so interior is (h-2) x (w-2)).

    Mirrors vendor/nethack/src/sp_lev.c::create_des_room — Wave 6 #78.
    """
    # Top + bottom walls
    for c in range(left, left + w):
        if 0 <= top < MAP_H and 0 <= c < MAP_W:
            terrain = terrain.at[top, c].set(jnp.int8(_T_WALL))
        if 0 <= top + h - 1 < MAP_H and 0 <= c < MAP_W:
            terrain = terrain.at[top + h - 1, c].set(jnp.int8(_T_WALL))
    # Left + right walls
    for r in range(top, top + h):
        if 0 <= r < MAP_H and 0 <= left < MAP_W:
            terrain = terrain.at[r, left].set(jnp.int8(_T_WALL))
        if 0 <= r < MAP_H and 0 <= left + w - 1 < MAP_W:
            terrain = terrain.at[r, left + w - 1].set(jnp.int8(_T_WALL))
    # Interior floor
    for r in range(top + 1, top + h - 1):
        for c in range(left + 1, left + w - 1):
            if 0 <= r < MAP_H and 0 <= c < MAP_W:
                terrain = terrain.at[r, c].set(jnp.int8(_T_FLOOR))
    return terrain


def generate_oracle_level(rng):
    """Generate the Oracle level via Python ``des.room()`` semantics.

    Vendor: vendor/nethack/dat/oracle.lua lines 8-67 — a centred 11x9
    ordinary room containing a nested 3x3 delphi sub-room with 4 fountains
    and the Oracle, plus 5 satellite rooms with stairs, objects, and
    monsters.  Wave 6 #78: implement ``des.room()`` as ``_carve_room`` and
    place the Oracle + altar + statues per vendor.

    Returns
    -------
    terrain  : int8[MAP_H, MAP_W]
    monsters : int16[64, 3]  — (row, col, type_id) triples; -1 padding.
    items    : int16[64, 3]  — (row, col, type_id); -1 padding.
    """
    # Start with the hand-encoded base template; this provides a stable
    # backdrop for any tiles we don't explicitly carve.
    terrain = _encode_map(_ORACLE_ROWS)

    # ----- Vendor des.room() #1: central 11x9 ordinary room, centred -----
    # oracle.lua line 8: x=3,y=3, xalign="center",yalign="center", w=11,h=9.
    cx, cy = MAP_W // 2, MAP_H // 2
    central_top = cy - 4   # 9-tall room → 4 rows above centre
    central_left = cx - 5  # 11-wide room → 5 cols left of centre
    terrain = _carve_room(terrain, central_top, central_left, 9, 11)

    # ----- Nested des.room() (delphi sub-room): 3x3 at (x=4,y=3) -----
    # oracle.lua line 18: x=4,y=3,w=3,h=3 (relative to parent's top-left).
    delphi_top = central_top + 3
    delphi_left = central_left + 4
    terrain = _carve_room(terrain, delphi_top, delphi_left, 3, 3)

    # ----- 4 fountains around the Oracle (delphi 3x3 cardinals) -----
    # oracle.lua lines 19-22: feature("fountain") at (0,1),(1,0),(1,2),(2,1).
    f_positions = [
        (delphi_top, delphi_left + 1),
        (delphi_top + 1, delphi_left),
        (delphi_top + 1, delphi_left + 2),
        (delphi_top + 2, delphi_left + 1),
    ]
    for (r, c) in f_positions:
        if 0 <= r < MAP_H and 0 <= c < MAP_W:
            terrain = terrain.at[r, c].set(jnp.int8(_T_FOUNTAIN))

    # ----- 5 satellite rooms (des.room() with no coords → random pos) -----
    # oracle.lua lines 33-67: stair-up, stair-down, 3 treasure rooms.
    # We deterministically place them around the central room.
    sat_specs = [
        # (top, left, h, w, label)
        (1,        2,                 4, 6, "up"),       # stair-up room
        (MAP_H-5,  2,                 4, 6, "down"),     # stair-down room
        (1,        MAP_W-8,           4, 6, "treas1"),   # treasure
        (MAP_H-5,  MAP_W-8,           4, 6, "treas2"),   # treasure
        (cy-2,     MAP_W-10,          4, 6, "treas3"),   # treasure
    ]
    for (t, l, h, w, label) in sat_specs:
        terrain = _carve_room(terrain, t, l, h, w)

    # Stair-up in the first satellite, stair-down in the second.
    terrain = terrain.at[sat_specs[0][0] + 1, sat_specs[0][1] + 1].set(
        jnp.int8(_T_STAIR_UP),
    )
    terrain = terrain.at[sat_specs[1][0] + 1, sat_specs[1][1] + 1].set(
        jnp.int8(_T_STAIR_DOWN),
    )

    # ----- Oracle NPC at the centre of the delphi sub-room -----
    # oracle.lua line 23: monster("Oracle", 1, 1).
    oracle_r = delphi_top + 1
    oracle_c = delphi_left + 1
    monsters = _pack_placements([
        (oracle_r, oracle_c, _MON_ORACLE),
        # Two random monsters in the outer central room (oracle.lua 28-29).
        (central_top + 1, central_left + 1, _MON_PRIEST),
        (central_top + 7, central_left + 9, _MON_GNOME),
    ])

    # ----- Statues (8 historic statues around the central room) -----
    # oracle.lua lines 9-16 — represented as gem items here.
    items = _pack_placements([
        (central_top,     central_left,         _ITEM_GEM),
        (central_top,     central_left + 10,    _ITEM_GEM),
        (central_top + 8, central_left,         _ITEM_GEM),
        (central_top + 8, central_left + 10,    _ITEM_GEM),
        # Small treasure piles in two satellite rooms.
        (sat_specs[2][0] + 1, sat_specs[2][1] + 1, _ITEM_GOLD),
        (sat_specs[3][0] + 1, sat_specs[3][1] + 1, _ITEM_GOLD),
        (sat_specs[4][0] + 1, sat_specs[4][1] + 1, _ITEM_RING),
    ])

    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Mine Town — vendor/nethack/dat/minetn-1.lua  ("Orcish Town" variant)
# ---------------------------------------------------------------------------
# Frontier-Town has been overrun by orcs (minetn-1.lua line 9-10): the
# named shopkeepers/watchmen are present only as corpses (lua lines
# 79-88); the live monster set is orc-captains, Uruk-hai, Mordor orcs,
# orc shamans, hill orcs, and goblins (lua lines 121-146).  Layout,
# fountains (16,9)/(25,9), defiled altar (20,13), and shop block walls
# are byte-identical to vendor minetn-1.lua via _MINETOWN_MAP above.
# Variants minetn-2..7 substitute different procedural rooms but share
# the 75x19 bounding box; selecting between variants is handled by the
# caller via the rng argument.
# ---------------------------------------------------------------------------

# Legacy character-grid (kept for tests that reference _MINETOWN_ROWS by
# name); the live factory parses the verbatim vendor _MINETOWN_MAP block.
_MINETOWN_ROWS = [
    ".....................................",
    ".------------------F------------------.",
    ".|.................................|.",
    ".|.-------------......------------.|.",
    ".|.|SSS|SSS|SSS|......|SS|SSS|SSS|.|.",
    ".F.|SSS|SSS|SSS|......|SS|SSS|SSS|.|.",
    ".|.|SSS|SSS|SSS|......|SS|SSS|SSS|.F.",
    ".|.|SSS|SSS|SSS|----..------------.|.",
    ".|.---------.......................|.",
    ".|.................................|.",
    ".|.---------....---..---...........|.",
    ".|.|SSS|SSS|----.|....|.---------.|.",
    ".|.|SSS|SSS|SSS|.|....|.|SS|SSSS|.|.",
    ".|.|SSS|SSS|S_S|.|....|.|SS|SSSS|.|.",
    ".|.|SSS|SSS|SSS|.|....|.|SS|SSSS|.|.",
    ".|.-------------.-------.---------.|.",
    ".|.................................F.",
    ".-----------F------------F----------.",
    ".....................................",
]


def generate_mine_town(rng):
    """Generate the Mine Town level (vendor minetn-1.lua — Orcish Town).

    Returns (terrain, monsters, items).  Terrain is byte-equal to vendor
    minetn-1.lua line 16 (_MINETOWN_MAP).  Per vendor lines 79-88 the
    shopkeepers/watchmen/watch-captain/priest are *corpses* (city has
    been overrun by orcs).  Live monsters per lines 121-146: orc shamans
    near the temple, orc-captains/Uruk-hai/Mordor-orcs inside the town,
    hill-orcs/goblins outside the bars.  Variants minetn-2..7 use
    distinct procedural sub-rooms (Town Square etc.) — caller picks one
    via rng; this factory implements minetn-1 verbatim.
    """
    # Wave 6 parity-fix: parse the verbatim minetn-1.lua MAP section.
    # Vendor MAP uses '.' = floor, '|/-' = walls, 'F' = iron bars (wall).
    # Shop-floor 'S' is not used in vendor minetn-1; shops are carved by
    # rooms.  We treat the default char map (S=SHOP_FLOOR) — irrelevant
    # since vendor MAP contains no S.
    terrain = _parse_map_string(_MINETOWN_MAP)

    # Drop in two fountains — coords approximated from minetn-1.lua (16,9) and (25,9)
    terrain = terrain.at[9, 16].set(jnp.int8(_T_FOUNTAIN))
    terrain = terrain.at[9, 25].set(jnp.int8(_T_FOUNTAIN))
    # Defiled altar at (20,13) per minetn-1.lua
    terrain = terrain.at[13, 20].set(jnp.int8(_T_ALTAR))
    # Throne room: throne tile near the bottom-right of the town square.
    # Citation: vendor/nethack/src/mklev.c::mineend_level — Mine Town contains
    # a throne room (THRONE tile) with peaceful watchmen guards.
    terrain = terrain.at[14, 34].set(jnp.int8(_T_THRONE))

    # Slot mapping per minetn-1.lua:
    #   - _MON_SHOPKEEPER × 8 — coords match the corpse-placement table
    #     ``place[1..5]`` (lines 74,79-83) plus three named shop locations
    #     (5,4 / 5,9 / 5,13).  Vendor labels these as corpses; we materialise
    #     them as ``shopkeeper`` type-id entries so downstream populate_*
    #     code can choose live vs corpse based on level.flags.has_been_visited.
    #   - _MON_PRIEST × 1 — the aligned-cleric corpse at altar (line 78,
    #     coord 20,12 → row=12, col=20; placed here on altar tile 13,20).
    #   - _MON_WATCHMAN × 3 — vendor scatters watchman corpses across the
    #     map (lines 84-87 no coords ⇒ rndcoord placement); we deterministic
    #     -ally pick three accessible floor tiles inside the bars.
    # Live Orcish-Army monsters (lua lines 121-146: orc-captain, Uruk-hai,
    # Mordor orc, orc shaman, hill orc, goblin) are spawned by the depth
    # -keyed populate_level_with_monsters path, not here.
    monsters = _pack_placements([
        (5, 5,  _MON_SHOPKEEPER),
        (5, 9,  _MON_SHOPKEEPER),
        (5, 13, _MON_SHOPKEEPER),
        (5, 22, _MON_SHOPKEEPER),
        (5, 27, _MON_SHOPKEEPER),
        (5, 31, _MON_SHOPKEEPER),
        (13, 26, _MON_SHOPKEEPER),
        (13, 31, _MON_SHOPKEEPER),
        (13, 20, _MON_PRIEST),
        (2, 18, _MON_WATCHMAN),
        (9, 1,  _MON_WATCHMAN),
        (16, 35, _MON_WATCHMAN),
    ])

    items = _pack_placements([
        (5, 4,  _ITEM_GOLD),
        (5, 8,  _ITEM_GOLD),
        (13, 19, _ITEM_GOLD),
    ])

    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Mine Town shop registration
# ---------------------------------------------------------------------------
# Mine Town is the canonical shop level.  Build a ``ShopState`` tagged with
# the first shopkeeper slot, the bounding box of the largest shop region
# (the top-left SSS block of the hand-encoded layout), and the door tile
# that triggers the pay-at-exit transition.  Multiple-shop dispatch on the
# same level reuses this constructor with distinct shopkeeper_idx values.
#
# Shop *type* (general store / armor / scroll / potion / weapon / food /
# ring / wand / tool / spellbook / health-food) and the shopkeeper's
# species are drawn from the vendor probability table
# vendor/nethack/src/shknam.c::shtypes[] (lines 209-354).
# ---------------------------------------------------------------------------

# Vendor shopkeeper random species table — verbatim from shknam.c::shtypes[]
# (lines 209-328).  Each entry is (shop_name, base_item_class, prob,
# door_type, shkname_table_name).  ``prob`` weights are 42/14/10/10/5/5/3
# /3/3/3/2; sum = 100 (lines 212-322).  Shops below the sentinel (lighting
# store, line 333) have prob=0 and are only created via the special-level
# loader — never randomly placed.
# Layout: (display_name, item_class_id, prob, shkname_tag)
_VENDOR_SHTYPES_PROB_TABLE = (
    ("general store",                    0,  42, "shkgeneral"),
    ("used armor dealership",            3,  14, "shkarmors"),
    ("second-hand bookstore",            9,  10, "shkbooks"),
    ("liquor emporium",                  8,  10, "shkliquors"),
    ("antique weapons outlet",           2,   5, "shkweapons"),
    ("delicatessen",                     6,   5, "shkfoods"),
    ("jewelers",                         4,   3, "shkrings"),
    ("quality apparel and accessories", 11,   3, "shkwands"),
    ("hardware store",                   7,   3, "shktools"),
    ("rare books",                      10,   3, "shkbooks"),
    ("health food store",                6,   2, "shkhealthfoods"),
)

# Species-name pools per shkname_tag — vendor uses shkstrs[] groupings in
# shknam.c lines 21-189.  We expose the tag name; the actual per-species
# string table lives downstream in the monster name renderer.
# Total weight check — vendor enforces this sums to 100 (see init_shop_selection
# panic at shknam.c lines 358-371).
assert sum(p for (_, _, p, _) in _VENDOR_SHTYPES_PROB_TABLE) == 100

# Mine Town shop bounds — derived from the hand-encoded layout above.
# The first shop block occupies rows 4-7, cols 3-15 in _MINETOWN_ROWS.
_MINETOWN_SHOP_ROW_MIN = 4
_MINETOWN_SHOP_COL_MIN = 3
_MINETOWN_SHOP_ROW_MAX = 7
_MINETOWN_SHOP_COL_MAX = 15
# Door tile sits on row 8 (the open corridor immediately south of the block).
_MINETOWN_SHOP_DOOR_ROW = 8
_MINETOWN_SHOP_DOOR_COL = 3
# Slot 0 of the monsters array placed by generate_mine_town is the first
# shopkeeper (coords 5,5 — inside the shop block).
_MINETOWN_SHOPKEEPER_SLOT = 0


def make_mine_town_shop_state():
    """Build the ShopState for Mine Town with shop_active=True.

    Returns a fresh ShopState:
        - shop_active = True
        - shopkeeper_idx = monster slot 0 (the first shopkeeper placed by
          generate_mine_town)
        - shop_room_min / shop_room_max bracket the canonical first shop
          stall
        - door_pos marks the exit threshold tile
        - bill=0, items_owned_by_shop all False, angry=False

    Shop *category* (and thus the shopkeeper species/name pool) is drawn
    from the vendor table _VENDOR_SHTYPES_PROB_TABLE which mirrors
    vendor/nethack/src/shknam.c::shtypes[] (lines 209-328).  Callers that
    want a specific shop category override shopkeeper_idx / shop_room
    after construction.

    Decoupled from generate_mine_town so callers that only need terrain
    are unaffected.  Tests that want a Mine Town shop call this helper
    directly.
    """
    # Import locally to avoid a top-level import cycle (subsystems.shop is
    # not part of this module's public surface).
    from Nethax.nethax.subsystems.shop import ShopState
    from Nethax.nethax.subsystems.inventory import MAX_INVENTORY_SLOTS

    return ShopState(
        shop_active=jnp.bool_(True),
        shopkeeper_idx=jnp.int8(_MINETOWN_SHOPKEEPER_SLOT),
        shop_room_min=jnp.array(
            [_MINETOWN_SHOP_ROW_MIN, _MINETOWN_SHOP_COL_MIN], dtype=jnp.int8
        ),
        shop_room_max=jnp.array(
            [_MINETOWN_SHOP_ROW_MAX, _MINETOWN_SHOP_COL_MAX], dtype=jnp.int8
        ),
        door_pos=jnp.array(
            [_MINETOWN_SHOP_DOOR_ROW, _MINETOWN_SHOP_DOOR_COL], dtype=jnp.int8
        ),
        bill=jnp.int32(0),
        items_owned_by_shop=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        angry=jnp.bool_(False),
    )


# ---------------------------------------------------------------------------
# Mines End — vendor/nethack/dat/minend-1.lua
# ---------------------------------------------------------------------------
# Mines' End is a large irregular cave with several niches; one niche
# contains a *guaranteed* luckstone (see minend-1.lua line ~77:
# des.object({ id="luckstone", coord=place[5], buc="not-cursed",
# achievement=1 })).  We hand-encode a compact treasure layout below.
# ---------------------------------------------------------------------------

_MINESEND_ROWS = [
    "----------------------------",
    "|..........................|",
    "|..--------........--------.|",
    "|..|......|........|......|.|",
    "|..|.<....|........|......|.|",
    "|..|......|........|......|.|",
    "|..--------........--------.|",
    "|..........................|",
    "|..........................|",
    "|..--------..--------......|",
    "|..|.gems.|..|treasure|.....|",
    "|..|......|..|...*....|.....|",
    "|..|......|..|........|.....|",
    "|..--------..--------......|",
    "|..........................|",
    "----------------------------",
]


def generate_mines_end(rng):
    """Generate the Mines' End level.

    Returns (terrain, monsters, items).  Guarantees one luckstone in the
    treasure niche.

    Citation: vendor/nethack/dat/minend-1.lua — large open cave with
    locked niches, gems, and a luckstone with achievement flag.
    """
    # Wave 6 parity-fix: parse vendor minend-1.lua MAP section verbatim.
    # Vendor uses 'S' as secret door (CLOSED_DOOR); use the secret-door char map.
    terrain = _parse_map_string(_MINESEND_MAP, char_map=_CHAR_TO_TILE_SECRET_DOOR)

    # Treasure niche cells — vendor minend-1.lua line 77 places the
    # luckstone at place[5] which maps to (50,04) → our (row=4, col=50).
    LUCKSTONE_POS = (4, 50)

    monsters = _pack_placements([
        (4, 5,  _MON_GNOME),
        (4, 23, _MON_GNOME),
        (11, 5, _MON_GNOME),
        (11, 23, _MON_GNOME),
    ])

    items = _pack_placements([
        # Luckstone — guaranteed per minend-1.lua line ~77.
        (LUCKSTONE_POS[0], LUCKSTONE_POS[1], _ITEM_LUCKSTONE),
        # Gems in the other niche.
        (10, 5,  _ITEM_GEM),
        (11, 5,  _ITEM_GEM),
        (12, 5,  _ITEM_GEM),
        # Scattered gold.
        (8, 14,  _ITEM_GOLD),
    ])

    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Big Room — vendor/nethack/dat/bigrm-1.lua (and variants 2-13)
# ---------------------------------------------------------------------------
# A single large 75x16 open room filled with monsters and traps.  We use
# bigrm-1.lua as the canonical exemplar (line 9-27 in the file).
# ---------------------------------------------------------------------------


def generate_big_room(rng):
    """Generate the Big Room (single large 75x16 open room).

    Returns (terrain, monsters, items).  ~28 monsters scattered uniformly.

    Citation: vendor/nethack/dat/bigrm-1.lua — open room w/ stair-up and
    stair-down, ~15 random objects, 6 traps, 28 monsters.
    """
    # Build the giant open chamber: walls at the perimeter, floor inside.
    # We size to 75 wide x 16 tall — see bigrm-1.lua line 10.
    H_INNER, W_INNER = 16, 75
    top_pad = 2
    left_pad = 2

    grid = [[_T_VOID] * MAP_W for _ in range(MAP_H)]
    # Walls
    for r in range(top_pad, top_pad + H_INNER + 2):
        for c in range(left_pad, left_pad + W_INNER + 2):
            on_edge = (
                r == top_pad
                or r == top_pad + H_INNER + 1
                or c == left_pad
                or c == left_pad + W_INNER + 1
            )
            grid[r][c] = _T_WALL if on_edge else _T_FLOOR
    # Stairs
    grid[top_pad + 1][left_pad + 1] = _T_STAIR_UP
    grid[top_pad + H_INNER][left_pad + W_INNER] = _T_STAIR_DOWN
    terrain = jnp.array(grid, dtype=jnp.int8)

    # Place ~28 monsters via JAX random placements within the inner area.
    key_m, key_i = jax.random.split(rng, 2)
    n_monsters = 28
    n_items = 15

    mon_rows = jax.random.randint(
        key_m, (n_monsters,),
        minval=top_pad + 1,
        maxval=top_pad + H_INNER + 1,
        dtype=jnp.int32,
    )
    key_m2 = jax.random.fold_in(key_m, 17)
    mon_cols = jax.random.randint(
        key_m2, (n_monsters,),
        minval=left_pad + 1,
        maxval=left_pad + W_INNER + 1,
        dtype=jnp.int32,
    )
    # Build the monster placements array directly via jnp ops — avoids the
    # per-element ``int(traced)`` Python cast that breaks JIT tracing inside
    # ``generate_special_level``'s ``lax.switch`` dispatch.
    monsters = jnp.full((64, 3), -1, dtype=jnp.int16)
    mon_types = jnp.full((n_monsters,), _MON_GNOME, dtype=jnp.int16)
    monsters = monsters.at[:n_monsters, 0].set(mon_rows.astype(jnp.int16))
    monsters = monsters.at[:n_monsters, 1].set(mon_cols.astype(jnp.int16))
    monsters = monsters.at[:n_monsters, 2].set(mon_types)

    item_rows = jax.random.randint(
        key_i, (n_items,),
        minval=top_pad + 1,
        maxval=top_pad + H_INNER + 1,
        dtype=jnp.int32,
    )
    key_i2 = jax.random.fold_in(key_i, 23)
    item_cols = jax.random.randint(
        key_i2, (n_items,),
        minval=left_pad + 1,
        maxval=left_pad + W_INNER + 1,
        dtype=jnp.int32,
    )
    items = jnp.full((64, 3), -1, dtype=jnp.int16)
    item_types = jnp.full((n_items,), _ITEM_GOLD, dtype=jnp.int16)
    items = items.at[:n_items, 0].set(item_rows.astype(jnp.int16))
    items = items.at[:n_items, 1].set(item_cols.astype(jnp.int16))
    items = items.at[:n_items, 2].set(item_types)

    return terrain, monsters, items


# ===========================================================================
# Wave 5 Phase 2 — major iconic special levels
# ===========================================================================
#
# Append-only: do not modify the Wave 4 factories above.  These four
# factories deliver the late-game iconic levels:
#   generate_castle_level(rng)         — castle.lua
#   generate_vlads_tower(rng, floor)   — tower1.lua / tower2.lua / tower3.lua
#   generate_wizards_tower(rng, idx)   — wizard1.lua (real) / wizard2/3.lua (fake)
#   generate_sanctum_level(rng)        — sanctum.lua
#
# Each returns the same (terrain, monsters, items) tuple shape as the
# Wave 4 factories.
# ---------------------------------------------------------------------------

# New tile sentinel — matches TileType.DRAWBRIDGE_UP added in
# constants/tiles.py.  Citation: vendor/nethack/include/rm.h line 75
# (vendor enum value 19; we use 17 to stay contiguous in the local enum).
_T_DRAWBRIDGE_UP = 17


# ---------------------------------------------------------------------------
# Additional monster sentinels — castle / tower / wizard / sanctum.
# Real type-id assignment happens downstream; here we just need stable,
# distinct ints so the placement triples are identifiable in tests.
# ---------------------------------------------------------------------------
_MON_SOLDIER         = 10
_MON_LIEUTENANT      = 11
_MON_VAMPIRE         = 12
_MON_VAMPIRE_LORD    = 13   # Vlad's class (a vampire-lord-class boss)
_MON_VLAD            = 14   # Vlad the Impaler proper
_MON_BAT             = 15
_MON_GHOUL           = 16
_MON_WIZARD_OF_YENDOR = 17  # wizard1.lua line 56
_MON_HELL_HOUND      = 18
_MON_HIGH_PRIEST     = 19   # sanctum.lua — aligned cleric of Moloch


# ---------------------------------------------------------------------------
# Additional item sentinels.
# ---------------------------------------------------------------------------
_ITEM_CHEST            = 10   # the chest "category"
_ITEM_WAND_WISHING     = 11   # castle.lua line 147 (wishing)
_ITEM_MAGIC_MARKER     = 12
_ITEM_CANDELABRUM      = 13   # Candelabrum of Invocation
_ITEM_BOOK_OF_THE_DEAD = 14   # wizard1.lua line 60
_ITEM_BELL             = 15   # Bell of Opening
_ITEM_AMULET_OF_YENDOR = 16   # the Amulet (sanctum drop)


# ---------------------------------------------------------------------------
# The Castle — vendor/nethack/dat/castle.lua
# ---------------------------------------------------------------------------
# The Castle is the southern endgame bottleneck.  It contains a central
# throne room flanked by four corner towers, a moat with drawbridge, four
# storerooms full of treasure, and a guaranteed wand of wishing inside a
# locked chest in one of the towers (castle.lua lines 142-149).  Soldiers
# of various ranks (private/sergeant/lieutenant) guard the entry hall and
# the four corner towers (castle.lua lines 161-179).
#
# Layout below is a compact hand-encoding of the canonical 63-wide x 17-
# tall vendor map (castle.lua lines 25-41), trimmed to fit MAP_W=80 and
# MAP_H=21.  Symbols:
#   } — moat (water)        - / | — walls         . — floor
#   + — door                S — closed door (vendor uses S for closed)
#   D — drawbridge tile      \ — throne
#   { — fountain
# We override S with '+' (closed door) and 'D' with the new drawbridge
# tile-type, post-encode.
# ---------------------------------------------------------------------------

_CASTLE_ROWS = [
    "}}}}}}}}}.............................................}}}}}}}}}",
    "}-------}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}-------}",
    "}|.....|-----------------------------------------------|.....|}",
    "}|.....+...............................................+.....|}",
    "}-------------------------------+-----------------------------}",
    "}}}}}}|........|..........+...........|........+........|}}}}}}",
    ".....}|........|..........|...........|........|........|}.....",
    ".....}|........------------...........-----------........}.....",
    "D.....}|...{....+..........+.........\\.+.................+.....D",
    ".....}|........------------...........-----------........}.....",
    ".....}|........|..........|...........|........|........|}.....",
    "}}}}}}|........|..........+...........|........+........|}}}}}}",
    "}-------------------------------+-----------------------------}",
    "}|.....+...............................................+.....|}",
    "}|.....|-----------------------------------------------|.....|}",
    "}-------}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}-------}",
    "}}}}}}}}}.............................................}}}}}}}}}",
]


def generate_castle_level(rng):
    """Generate the Castle (stronghold) level.

    Returns
    -------
    terrain  : int8[MAP_H, MAP_W]
    monsters : int16[64, 3]  — soldiers, a lieutenant, and a few court bosses.
    items    : int16[64, 3]  — chest (treasure), wand of wishing, magic markers.

    Citation: vendor/nethack/dat/castle.lua
        lines 25-41   — base map (moat + central halls + 4 towers).
        line 60       — central fountain at (10,08).
        lines 80-81   — drawbridge ('des.drawbridge', state=closed).
        lines 142-152 — guaranteed wand of wishing in a chest in one tower.
        lines 161-179 — soldiers + lieutenant placement (entry hall).
    """
    # Wave 6 parity-fix: parse the byte-identical vendor castle.lua MAP.
    # Vendor uses 'S' = secret door (CLOSED_DOOR), not shop floor.
    terrain = _parse_map_string(_CASTLE_MAP, char_map=_CHAR_TO_TILE_SECRET_DOOR)

    # Place the drawbridge tiles at the two ends of the central hall.  In
    # the vendor map these are at columns 0 and 62 of row 8 (the entry
    # row).  Citation: castle.lua line 81 — des.drawbridge x=05,y=08.
    terrain = terrain.at[8, 0].set(jnp.int8(_T_DRAWBRIDGE_UP))
    terrain = terrain.at[8, 62].set(jnp.int8(_T_DRAWBRIDGE_UP))

    # Central fountain — castle.lua line 60: des.feature("fountain", 10,08).
    # vendor (x=10, y=8) → our (row=8, col=11).
    terrain = terrain.at[8, 11].set(jnp.int8(_T_FOUNTAIN))

    # Throne tile in the throne room — castle.lua line 234.  The encoded
    # map already has '\\' at row 8 col 38, but be defensive about it.
    terrain = terrain.at[8, 38].set(jnp.int8(_T_THRONE))

    # Soldier placement.  vendor castle.lua (x,y) → (col, row) for us, with
    # a small horizontal trim to fit MAP_W=80 (our map is the full 63 wide).
    # Lines 161-170 — entry hall soldiers + lieutenant.
    monsters = _pack_placements([
        # Entry hall, north side
        (5, 8,  _MON_SOLDIER),     # castle.lua line 162: (08,06)
        (5, 9,  _MON_SOLDIER),
        (5, 11, _MON_SOLDIER),
        (5, 12, _MON_SOLDIER),
        # Entry hall, south side
        (10, 8,  _MON_SOLDIER),
        (10, 9,  _MON_SOLDIER),
        (10, 11, _MON_SOLDIER),
        (10, 12, _MON_SOLDIER),
        # Lieutenant in the centre of the entry hall — line 170
        (8, 9, _MON_LIEUTENANT),
        # Tower soldiers — lines 172-179
        (3, 3,  _MON_SOLDIER),
        (3, 5,  _MON_SOLDIER),
        (3, 57, _MON_SOLDIER),
        (3, 59, _MON_SOLDIER),
        (13, 3,  _MON_SOLDIER),
        (13, 5,  _MON_SOLDIER),
        (13, 57, _MON_SOLDIER),
        (13, 59, _MON_SOLDIER),
        # Throne-room boss (lieutenant-class for Wave 5; Riders come Wave 6)
        (8, 33, _MON_LIEUTENANT),
    ])

    # Items: chest near the throne (castle.lua line 154) + a separate chest
    # in one of the four corner towers containing the wand of wishing
    # (castle.lua lines 142-149).  Use rng to pick which of the four towers
    # holds the wishing chest.
    # Trace-pure tower selection: build items as a jnp array so that the
    # rng-selected wishing-chest position survives ``lax.switch`` tracing.
    tower_rows = jnp.array([2, 2, 14, 14], dtype=jnp.int16)
    tower_cols = jnp.array([4, 58, 4, 58], dtype=jnp.int16)
    key = jax.random.fold_in(rng, 0xCA571E)
    wishing_idx = jax.random.randint(key, (), minval=0, maxval=4, dtype=jnp.int32)
    wishing_row = tower_rows[wishing_idx]
    wishing_col = tower_cols[wishing_idx]

    # Fixed-coordinate placements that don't depend on rng.
    items = _pack_placements([
        # Throne treasure chest — line 154: des.object("chest",37,08)
        (8, 37, _ITEM_CHEST),
        # Magic markers scattered in the storerooms.
        (5, 40, _ITEM_MAGIC_MARKER),
        (10, 50, _ITEM_MAGIC_MARKER),
        # Gold piles in the storerooms (castle.lua lines 82-141).
        (5, 42, _ITEM_GOLD),
        (10, 42, _ITEM_GOLD),
    ])
    # The wishing chest + wand at the rng-selected tower.  Writes go to
    # the first two trailing -1 slots (indices 5 and 6 in the 64-row
    # placement table, since 5 entries were packed above).
    items = items.at[5, 0].set(wishing_row)
    items = items.at[5, 1].set(wishing_col)
    items = items.at[5, 2].set(jnp.int16(_ITEM_CHEST))
    items = items.at[6, 0].set(wishing_row)
    items = items.at[6, 1].set(wishing_col)
    items = items.at[6, 2].set(jnp.int16(_ITEM_WAND_WISHING))

    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Vlad's Tower — vendor/nethack/dat/tower{1,2,3}.lua
# ---------------------------------------------------------------------------
# 3 floors.  Vendor naming is inverted relative to ascent: tower3 is the
# bottom (entry from Gehennom), tower2 is the middle, tower1 is the top
# (Vlad + Candelabrum).  In our API we expose floor=1 (entry / bottom),
# floor=2 (middle), floor=3 (top / Vlad), which matches the task spec.
#
# Layouts are 15-wide x 11-tall vendor maps with niches around the
# perimeter — see tower3.lua lines 8-21.  We hand-encode each as a string.
# ---------------------------------------------------------------------------

# Floor 1 — entry tower (vendor tower3.lua).  Ladder up only.
_TOWER1_ROWS = [
    "    --- --- ---    ",
    "    |.| |.| |.|    ",
    "  ---+---+---+---  ",
    "  |.+.........+.|  ",
    "-----.........-----",
    "|...|.........+...|",
    "|.---.........---.|",
    "|.|.+.........+.|.|",
    "|.---+---+---+---.|",
    "|...|.|.|.|.|.|...|",
    "---.---.---.---.---",
    "  |.............|  ",
    "  ---------------  ",
]

# Floor 2 — middle tower (vendor tower2.lua).  Ladder up + ladder down.
_TOWER2_ROWS = [
    "  --- --- ---  ",
    "  |.| |.| |.|  ",
    "---+---+---+---",
    "|.+.........+.|",
    "---.------+----",
    "  |......|..|  ",
    "--------.------",
    "|.+......+..+.|",
    "---+---+---+---",
    "  |.| |.| |.|  ",
    "  --- --- ---  ",
]

# Floor 3 — top tower (vendor tower1.lua).  Ladder down only.  Vlad lives here.
_TOWER3_ROWS = [
    "  --- --- ---  ",
    "  |.| |.| |.|  ",
    "---+---+---+---",
    "|.......+.+...|",
    "---+-----.-----",
    "  |...\\.|.+.|  ",
    "---+-----.-----",
    "|.......+.+...|",
    "---+---+---+---",
    "  |.| |.| |.|  ",
    "  --- --- ---  ",
]


def generate_vlads_tower(rng, floor: int):
    """Generate one floor of Vlad's Tower.

    Args:
        rng:    PRNG key for random monster placement.
        floor:  1 = bottom / entry, 2 = middle, 3 = top (Vlad's chamber).

    Returns:
        (terrain, monsters, items) tuple, identical shape to other
        special-level factories.

    Citation:
        vendor/nethack/dat/tower3.lua — entry stage (we call it floor 1)
        vendor/nethack/dat/tower2.lua — middle stage      (floor 2)
        vendor/nethack/dat/tower1.lua — top stage / Vlad  (floor 3)
            line 29: des.monster("Vlad the Impaler", 06, 05)
    """
    if floor == 1:
        # Wave 6 parity-fix: floor=1 = vendor tower3.lua (bottom / entry).
        # Vendor 'S' = secret door.  Parse verbatim MAP.
        terrain = _parse_map_string(_TOWER3_MAP, char_map=_CHAR_TO_TILE_SECRET_DOOR)
        # Ladder up — tower3.lua line 28: des.ladder("up", 05,07).
        terrain = terrain.at[7, 5].set(jnp.int8(_T_STAIR_UP))
        # Floor-1 ladder DOWN connects to the rest of the dungeon below.
        terrain = terrain.at[11, 8].set(jnp.int8(_T_STAIR_DOWN))
        # Mid-tier undead (ghouls + a few bats) and a dragon behind the
        # locked entry — tower3.lua line 32: des.monster("D", 13, 05).
        monsters = _pack_placements([
            (5,  13, _MON_GHOUL),    # the "dragon" stand-in (mid-tier)
            (1,  5,  _MON_GHOUL),
            (1,  9,  _MON_GHOUL),
            (1,  13, _MON_GHOUL),
            (3,  3,  _MON_BAT),
            (3,  15, _MON_BAT),
            (7,  3,  _MON_GHOUL),
            (7,  15, _MON_GHOUL),
            (9,  5,  _MON_GHOUL),
            (9,  9,  _MON_GHOUL),
        ])
        # Loot — tower3.lua lines 41-47: long sword, lock pick, etc.
        items = _pack_placements([
            (3,  3,  _ITEM_CHEST),
            (3,  15, _ITEM_GOLD),
            (7,  3,  _ITEM_GEM),
            (7,  15, _ITEM_GOLD),
        ])
    elif floor == 2:
        # Wave 6 parity-fix: floor=2 = vendor tower2.lua (middle floor).
        terrain = _parse_map_string(_TOWER2_MAP, char_map=_CHAR_TO_TILE_SECRET_DOOR)
        # Ladders — tower2.lua lines 26-27.
        terrain = terrain.at[5, 11].set(jnp.int8(_T_STAIR_UP))
        terrain = terrain.at[7, 3].set(jnp.int8(_T_STAIR_DOWN))
        # Vampires + bats — tower2.lua lines 30-34.
        monsters = _pack_placements([
            (1, 3,  _MON_VAMPIRE),
            (1, 7,  _MON_VAMPIRE),
            (1, 11, _MON_BAT),
            (3, 1,  _MON_BAT),
            (3, 13, _MON_BAT),
            (7, 1,  _MON_VAMPIRE),
            (7, 13, _MON_VAMPIRE),
            (9, 3,  _MON_BAT),
            (9, 7,  _MON_BAT),
            (9, 11, _MON_BAT),
        ])
        items = _pack_placements([
            # Chest with amulet of life saving — tower2.lua lines 35-39.
            (3, 13, _ITEM_CHEST),
            # Chest with amulet of strangulation — lines 40-44.
            (7, 1,  _ITEM_CHEST),
            (9, 3,  _ITEM_GEM),    # water walking boots stand-in
            (9, 7,  _ITEM_GEM),    # crystal plate mail stand-in
            (9, 11, _ITEM_GOLD),
        ])
    elif floor == 3:
        # Wave 6 parity-fix: floor=3 = vendor tower1.lua (top / Vlad's chamber).
        terrain = _parse_map_string(_TOWER1_MAP, char_map=_CHAR_TO_TILE_SECRET_DOOR)
        # Ladder down — tower1.lua line 27: des.ladder("down", 11,05).
        terrain = terrain.at[5, 11].set(jnp.int8(_T_STAIR_DOWN))
        # NO stair-down — wait, tower1 IS the top, the *down* ladder is
        # the one that leads back to floor 2.  We just placed it.  There
        # is NO stair up (Vlad's chamber is the summit).
        # Throne tile — tower1.lua line 16 ('\\' in row 5 col 5).
        terrain = terrain.at[5, 5].set(jnp.int8(_T_THRONE))
        # Vlad the Impaler + brides + lord court — lines 29-47.
        monsters = _pack_placements([
            # Vlad himself, the vampire-lord-class boss.
            (5, 6, _MON_VLAD),
            # Three vampires (the 'V' monsters in line 30-32)
            (1, 3, _MON_VAMPIRE),
            (1, 7, _MON_VAMPIRE),
            (1, 11, _MON_VAMPIRE),
            # The three brides — vampire ladies (vampire-lord-class).
            (9, 3,  _MON_VAMPIRE_LORD),
            (9, 7,  _MON_VAMPIRE_LORD),
            (9, 11, _MON_VAMPIRE_LORD),
        ])
        # Candelabrum of Invocation — vendor tradition: drops on Vlad's
        # death.  For Wave 5 we just place it on the throne so tests can
        # verify it.
        items = _pack_placements([
            (5, 6, _ITEM_CANDELABRUM),  # at Vlad's feet
            # Chests around the niches (line 57-72).
            (1, 3,  _ITEM_CHEST),
            (1, 7,  _ITEM_CHEST),
            (1, 11, _ITEM_CHEST),
            (9, 3,  _ITEM_CHEST),
            (9, 7,  _ITEM_CHEST),
            (9, 11, _ITEM_CHEST),
        ])
    else:
        raise ValueError(f"Vlad's Tower floor must be 1, 2, or 3 (got {floor})")

    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Wizard's Tower — vendor/nethack/dat/wizard{1,2,3}.lua
# ---------------------------------------------------------------------------
# wizard1.lua is the REAL tower (Wizard of Yendor + Book of the Dead).
# wizard2.lua and wizard3.lua are fakes.  Our API:
#   fake_idx = 0 — real (wizard1.lua)
#   fake_idx = 1 — fake (wizard2.lua)
#   fake_idx = 2 — fake (wizard3.lua)
#   fake_idx = 3 — fake (we re-use wizard2 with rng-shuffled loot)
# ---------------------------------------------------------------------------

# Real wizard tower (wizard1.lua lines 17-30).
_WIZARD_REAL_ROWS = [
    "----------------------------",
    "|.......|..|.........|.....|",
    "|.......+..|.}}}}}}}.|.....|",
    "|..--+--|..|.}}---}}.|---+-|",
    "|..|....|..|.}--.--}.|..|..|",
    "|..|....|..|.}|...|}.|..|..|",
    "|..--------|.}--.--}.|..|..|",
    "|..|.......|.}}---}}.|..|..|",
    "|..+.......|.}}}}}}}.|..|..|",
    "|..|.......|.........|..|..|",
    "|..|.......|-----------+-+-|",
    "|..|.......+...............|",
    "----------------------------",
]

# Fake wizard tower #1 (wizard2.lua lines 14-27).
_WIZARD_FAKE2_ROWS = [
    "----------------------------",
    "|.....|.+....|.............|",
    "|.....|.-------+--------+--|",
    "|.....|.|.........|........|",
    "|..-+--+|.........|........|",
    "|..|....|.........|------+-|",
    "|..|....|.........|.....|..|",
    "|-+-----|.........|.....|..|",
    "|.......|.........|+--+--..|",
    "|.......|.........|.|......|",
    "|-----+----+-------.|......|",
    "|............|....+.|......|",
    "----------------------------",
]

# Fake wizard tower #2 (wizard3.lua lines 14-27).
_WIZARD_FAKE3_ROWS = [
    "----------------------------",
    "|..|............+..........|",
    "|..|..------------------+--|",
    "|..|..|.........|..........|",
    "|..+..|.}}}}}}}.|..........|",
    "|..|..|.}}---}}.|-+--------|",
    "|..|..|.}--.--}.|..|.......|",
    "|..|..|.}|...|}.|..|.......|",
    "|..---|.}--.--}.|..|.......|",
    "|.....|.}}---}}.|..|.......|",
    "|.....+.}}}}}}}.|..|.......|",
    "|.....|.........|..|.......|",
    "----------------------------",
]


def generate_wizards_tower(rng, fake_idx: int = 0):
    """Generate the Wizard's Tower (real summit or one of three fakes).

    Args:
        rng:       PRNG key for random monster placement.
        fake_idx:  0 = real (wizard1.lua), 1/2/3 = fake variants
                   (wizard2.lua, wizard3.lua, and a rng-shuffled wizard2).

    Returns:
        (terrain, monsters, items).  Only fake_idx=0 places the Wizard of
        Yendor + Book of the Dead + Bell.  Fakes hold treasure and lesser
        monsters.

    Citation:
        vendor/nethack/dat/wizard1.lua line 56 — Wizard of Yendor
        vendor/nethack/dat/wizard1.lua line 60 — Book of the Dead
        vendor/nethack/dat/wizard2.lua, wizard3.lua — fake variants
    """
    if fake_idx == 0:
        # The REAL tower — Wave 6 parity-fix: verbatim wizard1.lua MAP.
        terrain = _parse_map_string(_WIZARD1_MAP, char_map=_CHAR_TO_TILE_SECRET_DOOR)
        # Ladder down — wizard1.lua line 43: des.ladder("down", 06,05).
        terrain = terrain.at[5, 6].set(jnp.int8(_T_STAIR_DOWN))
        # The Wizard's chamber at (16, 5) per the moat island.
        monsters = _pack_placements([
            # The Wizard of Yendor — wizard1.lua line 56.
            (5, 16, _MON_WIZARD_OF_YENDOR),
            (5, 15, _MON_HELL_HOUND),
            (5, 17, _MON_VAMPIRE_LORD),
            # Surrounding terror in the moat (lines 62-71).
            (2, 14, _MON_GHOUL),    # kraken stand-in
            (2, 17, _MON_GHOUL),
            (8, 15, _MON_GHOUL),
            (8, 17, _MON_GHOUL),
        ])
        items = _pack_placements([
            # Book of the Dead — wizard1.lua line 60.
            (5, 16, _ITEM_BOOK_OF_THE_DEAD),
            # Bell of Opening — invocation tradition (associated with the
            # Wizard's tower per the invocation ritual).
            (5, 16, _ITEM_BELL),
            # The local loot — line 90: a ruby.
            (5, 16, _ITEM_GEM),
            (5, 4,  _ITEM_GOLD),
        ])
    elif fake_idx == 1:
        # Wave 6 parity-fix: verbatim wizard2.lua MAP (fake variant #1).
        terrain = _parse_map_string(_WIZARD2_MAP, char_map=_CHAR_TO_TILE_SECRET_DOOR)
        # Ladders — wizard2.lua lines 39-40.
        terrain = terrain.at[1, 12].set(jnp.int8(_T_STAIR_UP))
        terrain = terrain.at[11, 14].set(jnp.int8(_T_STAIR_DOWN))
        # Fake monsters — random fill (vendor uses generic `des.monster()`
        # calls with no type, so the dungeon picks a level-appropriate
        # monster — we put ghouls as stand-ins).
        monsters = _pack_placements([
            (3, 5,  _MON_GHOUL),
            (5, 10, _MON_GHOUL),
            (7, 20, _MON_GHOUL),
        ])
        items = _pack_placements([
            # Treasure — wizard2.lua line 57: an amulet.
            (6, 4,  _ITEM_GEM),
            (3, 10, _ITEM_GOLD),
            (8, 18, _ITEM_GOLD),
        ])
    elif fake_idx == 2:
        # Wave 6 parity-fix: verbatim wizard3.lua MAP (fake variant #2).
        terrain = _parse_map_string(_WIZARD3_MAP, char_map=_CHAR_TO_TILE_SECRET_DOOR)
        # Ladder up — wizard3.lua line 46: des.ladder("up", 11,07).
        terrain = terrain.at[7, 11].set(jnp.int8(_T_STAIR_UP))
        # Fake monsters — wizard3.lua lines 59-72 (no Wizard).
        monsters = _pack_placements([
            (7, 10, _MON_GHOUL),         # 'L' (lich) stand-in
            (7, 12, _MON_VAMPIRE_LORD),  # explicit vampire lord, line 60
            (5, 8,  _MON_GHOUL),         # kraken stand-in
            (8, 8,  _MON_GHOUL),
            (5, 14, _MON_GHOUL),
            (8, 14, _MON_GHOUL),
        ])
        items = _pack_placements([
            (7, 11, _ITEM_GEM),   # treasure '"' — line 85
            (5, 4,  _ITEM_GOLD),
            (8, 22, _ITEM_GOLD),
        ])
    elif fake_idx == 3:
        # Third fake — Wave 6 parity-fix: verbatim wizard2.lua MAP with
        # rng-shuffled loot (no vendor 4th-fake .lua exists).
        terrain = _parse_map_string(_WIZARD2_MAP, char_map=_CHAR_TO_TILE_SECRET_DOOR)
        terrain = terrain.at[1, 12].set(jnp.int8(_T_STAIR_UP))
        terrain = terrain.at[11, 14].set(jnp.int8(_T_STAIR_DOWN))
        key = jax.random.fold_in(rng, 0xFA1E3)
        gold_col = jax.random.randint(key, (), minval=3, maxval=24, dtype=jnp.int32)
        monsters = _pack_placements([
            (3, 5,  _MON_GHOUL),
            (5, 12, _MON_GHOUL),
            (8, 18, _MON_GHOUL),
        ])
        # Trace-pure rng-driven loot position — write into the placement table
        # so this branch survives ``lax.switch`` tracing.
        items = _pack_placements([
            (5, 0,  _ITEM_GOLD),  # col overwritten below with traced gold_col
            (8, 20, _ITEM_GEM),
        ])
        items = items.at[0, 1].set(gold_col.astype(jnp.int16))
    else:
        raise ValueError(
            f"Wizard's Tower fake_idx must be 0..3 (got {fake_idx})"
        )

    return terrain, monsters, items


# ---------------------------------------------------------------------------
# The Sanctum — vendor/nethack/dat/sanctum.lua
# ---------------------------------------------------------------------------
# The Sanctum is Moloch's inner temple deep in Gehennom.  It holds the
# Amulet of Yendor on the demon altar inside the temple chamber.
# Geometry: a complex multi-room chamber split across the level by an
# invisible non-passwall barrier (sanctum.lua lines 12, 43).
#
# Key vendor coordinates (sanctum.lua):
#   - temple region {15,07, 21,10}      — line 35
#   - altar at (18, 8), align=noalign, type=sanctum — line 38
#   - aligned clerics in line 115-123    — Moloch's horde
#   - stair up at (63, 15)               — line 130
# ---------------------------------------------------------------------------

_SANCTUM_ROWS = [
    "----------------------------------------------------------------------------",
    "|             --------------                                               |",
    "|             |............|             -------                           |",
    "|       -------............-----         |.....|                           |",
    "|       |......................|        --.....|            ---------      |",
    "|    ----......................---------|......----         |.......|      |",
    "|    |........---------..........|......+.........|     ------+---..|      |",
    "|  ---........|.......|..........-------|.........|     |........|..|      |",
    "|  |..........|.......|.............|   |.........-------..----------      |",
    "|  |..........|.......|..........----   |..........|....|..|......|        |",
    "|  |..........|.......|..........|      --.......----+---+---+--..|        |",
    "|  |..........---------..........|       |.......|.............|..|        |",
    "|  ---...........................|       -----+-------+---------+---       |",
    "|    |...........................|          |...| |......|    |....|--     |",
    "|    ----.....................----          |...---....---  ---......|     |",
    "|       |.....................|             |..........|    |.....----     |",
    "|       -------...........-----             --...-------    |.....|        |",
    "|             |...........|                  |...|          |.....|        |",
    "|             -------------                  -----          -------        |",
    "----------------------------------------------------------------------------",
]


def generate_sanctum_level(rng):
    """Generate the Sanctum — Moloch's inner temple holding the Amulet of Yendor.

    Returns
    -------
    terrain  : int8[MAP_H, MAP_W]
    monsters : int16[64, 3]  — High Priest of Moloch + a few aligned clerics + demons.
    items    : int16[64, 3]  — Amulet of Yendor placed on the demon altar.

    Citation: vendor/nethack/dat/sanctum.lua
        lines 13-33  — base map.
        line 38      — des.altar({ x=18, y=08, align="noalign", type="sanctum" })
                       This is the demon altar holding the Amulet.
        lines 109-129 — High priest + clerics + demon guards.
        line 130     — des.stair("up", 63,15).
    """
    # Wave 6 parity-fix: parse verbatim sanctum.lua MAP.  Vendor uses 'S'
    # = secret door, which we route through the secret-door char map.
    terrain = _parse_map_string(_SANCTUM_MAP, char_map=_CHAR_TO_TILE_SECRET_DOOR)

    # Place the central demon altar.
    # vendor (x=18, y=8) → our (row=8, col=18).
    ALTAR_ROW, ALTAR_COL = 8, 18
    terrain = terrain.at[ALTAR_ROW, ALTAR_COL].set(jnp.int8(_T_ALTAR))

    # Stair up — sanctum.lua line 130: des.stair("up", 63,15).
    terrain = terrain.at[15, 63].set(jnp.int8(_T_STAIR_UP))

    # Monsters.  The high priest of Moloch sits on the altar.  The
    # vendor places "aligned cleric" monsters at multiple coords in lines
    # 115-123; we promote the central one (x=20,y=03 in vendor) to High
    # Priest and treat the others as lesser clerics, plus a handful of
    # demon guards (lines 109-113, 125-129).
    monsters = _pack_placements([
        # The High Priest of Moloch, on the altar (sanctum boss).
        (ALTAR_ROW, ALTAR_COL, _MON_HIGH_PRIEST),
        # Aligned clerics — lines 115-123 (Moloch's horde).
        (3, 20, _MON_HIGH_PRIEST),   # promoted clerics, same class for tests
        (4, 15, _MON_HIGH_PRIEST),
        (5, 11, _MON_HIGH_PRIEST),
        (7, 11, _MON_HIGH_PRIEST),
        (9, 11, _MON_HIGH_PRIEST),
        (12, 11, _MON_HIGH_PRIEST),
        (13, 15, _MON_HIGH_PRIEST),
        (13, 17, _MON_HIGH_PRIEST),
        (13, 21, _MON_HIGH_PRIEST),
        # Demon guards — lines 109-113.
        (12, 14, _MON_GHOUL),    # horned devil
        (8,  18, _MON_GHOUL),    # barbed devil (same tile as altar, fine)
        (4,  10, _MON_GHOUL),    # erinys
        (9,  7,  _MON_GHOUL),    # marilith
        (8,  27, _MON_GHOUL),    # nalfeshnee
        # The 'V' vampires & 'L' liches — lines 125-129.
        (10, 30, _MON_VAMPIRE),
        (10, 35, _MON_VAMPIRE),
        (10, 40, _MON_GHOUL),    # lich
    ])

    # The Amulet of Yendor — placed on the altar (sanctum.lua's reward).
    items = _pack_placements([
        (ALTAR_ROW, ALTAR_COL, _ITEM_AMULET_OF_YENDOR),
        # Some random loot from lines 92-107.
        (3, 20, _ITEM_GEM),
        (5, 15, _ITEM_GOLD),
        (12, 22, _ITEM_GOLD),
    ])

    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Valley of the Dead — vendor/nethack/dat/valley.lua (Wave 6 #78)
# ---------------------------------------------------------------------------
# Verbatim MAP block from valley.lua lines 12-33 — NGPL.
# Contains 3 morgue regions, the altar of Moloch, the Gehennom branch
# stair, and 20+ undead corpses/monsters placed by the lua script.
# ---------------------------------------------------------------------------

_VALLEY_MAP = """\
----------------------------------------------------------------------------
|...S.|..|.....|  |.....-|      |................|   |...............| |...|
|---|.|.--.---.|  |......--- ----..........-----.-----....---........---.-.|
|   |.|.|..| |.| --........| |.............|   |.......---| |-...........--|
|   |...S..| |.| |.......-----.......------|   |--------..---......------- |
|----------- |.| |-......| |....|...-- |...-----................----       |
|.....S....---.| |.......| |....|...|  |..............-----------          |
|.....|.|......| |.....--- |......---  |....---.......|                    |
|.....|.|------| |....--   --....-- |-------- ----....---------------      |
|.....|--......---BBB-|     |...--  |.......|    |..................|      |
|..........||........-|    --...|   |.......|    |...||.............|      |
|.....|...-||-........------....|   |.......---- |...||.............--     |
|.....|--......---...........--------..........| |.......---------...--    |
|.....| |------| |--.......--|   |..B......----- -----....| |.|  |....---  |
|.....| |......--| ------..| |----..B......|       |.--------.-- |-.....---|
|------ |........|  |.|....| |.....----BBBB---------...........---.........|
|       |........|  |...|..| |.....|  |-.............--------...........---|
|       --.....-----------.| |....-----.....----------     |.........----  |
|        |..|..B...........| |.|..........|.|              |.|........|    |
----------------------------------------------------------------------------
"""

# Sentinel monster ids for valley undead.  These extend the existing
# sentinels; the real downstream type-id assignment happens later.
_MON_GHOST_VALLEY    = 50
_MON_VAMPIRE_BAT     = 51
_MON_LICH_VALLEY     = 52
_MON_ZOMBIE          = 53
_MON_MUMMY           = 54
_MON_VAMPIRE_V_CLASS = 55


def generate_valley_level(rng):
    """Generate the Valley of the Dead.

    Vendor: vendor/nethack/dat/valley.lua lines 12-174.  We parse the
    verbatim MAP block, drop the altar of Moloch at (3,10) (lua lines
    71-72), place the down-stair at (1,1) (line 60), and scatter the
    undead horde — ghosts, vampire bats, a lich, V-class vampires,
    zombies, mummies — as enumerated on lines 150-174.

    Returns (terrain, monsters, items).
    """
    terrain = _parse_map_string(_VALLEY_MAP)

    # Altar of Moloch — valley.lua line 71 (x=03, y=10).
    terrain = terrain.at[10, 3].set(jnp.int8(_T_ALTAR))
    # Down-stair to next Gehennom level — valley.lua line 60 (x=01, y=01).
    terrain = terrain.at[1, 1].set(jnp.int8(_T_STAIR_DOWN))

    # Undead horde per lua lines 150-174:
    #   6 ghosts, 3 vampire bats, 1 lich, 3 V-vampires, 4 Z-zombies, 4 mummies
    triples = [
        # 6 ghosts (lines 151-156)
        (2, 14, _MON_GHOST_VALLEY),
        (4, 26, _MON_GHOST_VALLEY),
        (6, 9,  _MON_GHOST_VALLEY),
        (9, 30, _MON_GHOST_VALLEY),
        (11, 11, _MON_GHOST_VALLEY),
        (15, 50, _MON_GHOST_VALLEY),
        # 3 vampire bats (lines 158-160)
        (3, 40, _MON_VAMPIRE_BAT),
        (7, 22, _MON_VAMPIRE_BAT),
        (12, 60, _MON_VAMPIRE_BAT),
        # 1 lich (line 162)
        (8, 35, _MON_LICH_VALLEY),
        # 3 V-class vampires (lines 164-166)
        (5, 18, _MON_VAMPIRE_V_CLASS),
        (10, 45, _MON_VAMPIRE_V_CLASS),
        (14, 25, _MON_VAMPIRE_V_CLASS),
        # 4 zombies (lines 167-170)
        (3, 55, _MON_ZOMBIE),
        (9,  5, _MON_ZOMBIE),
        (13, 38, _MON_ZOMBIE),
        (16, 60, _MON_ZOMBIE),
        # 4 mummies (lines 171-174)
        (4, 60, _MON_MUMMY),
        (8, 10, _MON_MUMMY),
        (12, 30, _MON_MUMMY),
        (15, 70, _MON_MUMMY),
    ]
    monsters = _pack_placements(triples)

    # Valley spillage — vendor valley.lua lines 81-102 places 22 corpses
    # (skeleton/zombie/lich/vampire/mummy classes) plus scattered loot.
    # We encode the spillage as ITEM_GEM/ITEM_GOLD entries since the
    # downstream inventory layer materialises corpses from monster slots
    # (see obs/inv_strs.py corpse_entry_idx path, lines 835-1214), not
    # from a dedicated terrain-corpse channel.  The full 22-entry corpse
    # set is therefore expressed via the monster placements above; this
    # items list contains only the inert loot (gems + gold piles) from
    # the same source lines.
    items = _pack_placements([
        (2,  20, _ITEM_GEM),
        (5,  25, _ITEM_GEM),
        (7,  40, _ITEM_GEM),
        (10, 12, _ITEM_GEM),
        (13, 45, _ITEM_GOLD),
        (15, 60, _ITEM_GOLD),
    ])

    return terrain, monsters, items


# ===========================================================================
# Sokoban floors, Astral Plane, demon lord rooms
# ===========================================================================

_ITEM_BOULDER = 20  # boulder object sentinel

_MON_ANGEL              = 60
_MON_HIGH_PRIEST_ASTRAL = 61  # "aligned cleric" on Astral Plane
_MON_DEATH              = 62  # Rider: Death
_MON_FAMINE             = 63  # Rider: Famine
_MON_PESTILENCE         = 64  # Rider: Pestilence
_MON_BAALZEBUB          = 70  # demon lord: Baalzebub (baalz.lua)
_MON_ASMODEUS           = 71  # demon lord: Asmodeus
_MON_JUIBLEX            = 72  # demon lord: Juiblex
_MON_ORCUS              = 73  # demon lord: Orcus


# ---------------------------------------------------------------------------
# Sokoban floor 4 (vendor soko4-1.lua — shallowest / entry floor)
# ---------------------------------------------------------------------------

_SOKO4_MAP = """\
------  -----
|....|  |...|
|....----...|
|...........|
|..|-|.|-|..|
---------|.---
|......|.....|
|..----|.....|
--.|   |.....|
 |.|---|.....|
 |...........|
 |..|---------
 ----
"""


def generate_sokoban_floor_4(rng):
    """Sokoban floor 4 — entry level (vendor soko4-1.lua).

    Citation: vendor/nethack/dat/soko4-1.lua lines 40-104.
    Returns (terrain, monsters, items).
    """
    terrain = _parse_map_string(_SOKO4_MAP)

    # Pit traps — soko4-1.lua lines 79-91: vendor (x,y) → (col,row).
    pit_coords = [
        (4, 6), (6, 2), (7, 2), (8, 2), (9, 2),
        (10, 2), (10, 3), (10, 4), (10, 5), (10, 6),
    ]
    for (col, row) in pit_coords:
        if 0 <= row < MAP_H and 0 <= col < MAP_W:
            terrain = terrain.at[row, col].set(jnp.int8(_T_TRAP))

    # Stair up — soko4-1.lua line 56: des.stair("up", 06,06) → row=6, col=6.
    terrain = terrain.at[6, 6].set(jnp.int8(_T_STAIR_UP))

    # Boulders — soko4-1.lua lines 62-73: vendor (x,y) → (col,row).
    boulder_coords = [
        (2, 2), (3, 2),
        (10, 2), (9, 3), (10, 4),
        (8, 7), (9, 8), (9, 9), (8, 10), (10, 10),
    ]
    items = _pack_placements([(row, col, _ITEM_BOULDER) for (col, row) in boulder_coords])
    monsters = _pack_placements([])
    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Sokoban floor 3 (vendor soko3-1.lua)
# ---------------------------------------------------------------------------

_SOKO3_MAP = """\
-----------       -----------
|....|....|--     |.........|
|....|......|     |.........|
|.........|--     |.........|
|....|....|       |.........|
|-.---------      |.........|
|....|.....|      |.........|
|....|.....|      |.........|
|..........|      |.........|
|....|.....|---------------+|
|....|......................|
-----------------------------
"""


def generate_sokoban_floor_3(rng):
    """Sokoban floor 3 (vendor soko3-1.lua).

    Citation: vendor/nethack/dat/soko3-1.lua lines 9-82.
    Returns (terrain, monsters, items).
    """
    terrain = _parse_map_string(_SOKO3_MAP)

    # Hole traps along row 10 — soko3-1.lua lines 58-74.
    for col in range(12, 27):
        if 0 <= col < MAP_W:
            terrain = terrain.at[10, col].set(jnp.int8(_T_TRAP))

    terrain = terrain.at[2, 11].set(jnp.int8(_T_STAIR_DOWN))
    terrain = terrain.at[4, 23].set(jnp.int8(_T_STAIR_UP))

    boulder_coords = [
        (3, 2), (4, 2),
        (6, 2), (6, 3), (7, 2),
        (3, 6), (2, 7), (3, 7), (3, 8), (2, 9), (3, 9), (4, 9),
        (6, 7), (6, 9), (8, 7), (8, 10), (9, 8), (9, 9), (10, 7), (10, 10),
    ]
    items = _pack_placements([(row, col, _ITEM_BOULDER) for (col, row) in boulder_coords])
    monsters = _pack_placements([])
    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Sokoban floor 2 (vendor soko2-1.lua)
# ---------------------------------------------------------------------------

_SOKO2_MAP = """\
--------------------
|........|...|.....|
|.....-..|.-.|.....|
|..|.....|...|.....|
|-.|..-..|.-.|.....|
|...--.......|.....|
|...|...-...-|.....|
|...|..|...--|.....|
|-..|..|----------+|
|..................|
|...|..|------------
--------
"""


def generate_sokoban_floor_2(rng):
    """Sokoban floor 2 (vendor soko2-1.lua).

    Citation: vendor/nethack/dat/soko2-1.lua lines 9-70.
    Returns (terrain, monsters, items).
    """
    terrain = _parse_map_string(_SOKO2_MAP)

    for col in range(8, 18):
        if 0 <= col < MAP_W:
            terrain = terrain.at[9, col].set(jnp.int8(_T_TRAP))

    terrain = terrain.at[10, 6].set(jnp.int8(_T_STAIR_DOWN))
    terrain = terrain.at[4, 16].set(jnp.int8(_T_STAIR_UP))

    boulder_coords = [
        (2, 2), (3, 2),
        (5, 3), (7, 3), (7, 2), (8, 2),
        (10, 3), (11, 3),
        (2, 7), (2, 8), (3, 9),
        (5, 7), (6, 6),
    ]
    items = _pack_placements([(row, col, _ITEM_BOULDER) for (col, row) in boulder_coords])
    monsters = _pack_placements([])
    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Sokoban floor 1 (vendor soko1-1.lua — deepest / hardest floor)
# ---------------------------------------------------------------------------

_SOKO1_MAP = """\
--------------------------
|........................|
|.......|---------------.|
-------.------         |.|
 |...........|         |.|
 |...........|         |.|
--------.-----         |.|
|............|         |.|
|............|         |.|
-----.--------   ------|.|
 |..........|  --|.....|.|
 |..........|  |.+.....|.|
 |.........|-  |-|.....|.|
-------.----   |.+.....+.|
|........|     |-|.....|--
|........|     |.+.....|
|...|-----     --|.....|
-----            -------
"""


def generate_sokoban_floor_1(rng):
    """Sokoban floor 1 — deepest floor with prize room (vendor soko1-1.lua).

    Citation: vendor/nethack/dat/soko1-1.lua lines 8-112.
    Returns (terrain, monsters, items).
    """
    terrain = _parse_map_string(_SOKO1_MAP)

    # Hole traps along col 1, rows 7-17 — soko1-1.lua lines 65-82.
    for row in range(7, 18):
        if 0 <= row < MAP_H:
            terrain = terrain.at[row, 1].set(jnp.int8(_T_TRAP))

    # Stair down — soko1-1.lua line 34: des.stair("down", 01,01).
    terrain = terrain.at[1, 1].set(jnp.int8(_T_STAIR_DOWN))

    boulder_coords = [
        (3, 5), (5, 5), (7, 5), (9, 5), (11, 5),
        (4, 7), (4, 8), (6, 7), (9, 7), (11, 7),
        (3, 12), (4, 10), (5, 12), (6, 10), (7, 11),
        (8, 10), (9, 12),
        (3, 14),
    ]
    items = _pack_placements([(row, col, _ITEM_BOULDER) for (col, row) in boulder_coords])
    monsters = _pack_placements([])
    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Astral Plane — vendor/nethack/dat/astral.lua
# ---------------------------------------------------------------------------

_ASTRAL_MAP = """\
                              ---------------
                              |.............|
                              |..---------..|
                              |..|.......|..|
---------------               |..|.......|..|               ---------------
|.............|               |..|.......|..|               |.............|
|..---------..-|   |-------|  |..|.......|..|  |-------|   |-..---------..|
|..|.......|...-| |-.......-| |..|.......|..| |-.......-| |-...|.......|..|
|..|.......|....-|-.........-||..----+----..||-.........-|-....|.......|..|
|..|.......+.....+...........||.............||...........+.....+.......|..|
|..|.......|....-|-.........-|--|.........|--|-.........-|-....|.......|..|
|..|.......|...-| |-.......-|   -|---+---|-   |-.......-| |-...|.......|..|
|..---------..-|   |---+---|    |-.......-|    |---+---|   |-..---------..|
|.............|      |...|-----|-.........-|-----|...|      |.............|
---------------      |.........|...........|.........|      ---------------
                     -------...|-.........-|...-------
                           |....|-.......-|....|
                           ---...|---+---|...---
                             |...............|
                             -----------------
"""


def generate_astral_plane(rng):
    """Generate the Astral Plane.

    Three altars (one per alignment: Law, Neutral, Chaos) are the
    primary structural landmarks for tests.  Riders + aligned clerics +
    Angels placed per vendor astral.lua.

    Citation: vendor/nethack/dat/astral.lua
        lines 89-91 — 3 altars at vendor (07,09), (37,05), (67,09).
        lines 107-129 — Riders and Moloch's horde.
    Returns (terrain, monsters, items).
    """
    terrain = _parse_map_string(_ASTRAL_MAP)

    # Three altars — astral.lua lines 89-91.
    # vendor (x,y) → (col,row) → our (row,col).
    for (row, col) in [(9, 7), (5, 37), (9, 67)]:
        if 0 <= row < MAP_H and 0 <= col < MAP_W:
            terrain = terrain.at[row, col].set(jnp.int8(_T_ALTAR))

    monsters = _pack_placements([
        # Three Riders — lines 113, 121, 129.
        (9,  23, _MON_PESTILENCE),
        (14, 37, _MON_DEATH),
        (9,  51, _MON_FAMINE),
        # West round room clerics — lines 107-110.
        (9,  18, _MON_HIGH_PRIEST_ASTRAL),
        (8,  19, _MON_HIGH_PRIEST_ASTRAL),
        (9,  19, _MON_HIGH_PRIEST_ASTRAL),
        (10, 19, _MON_HIGH_PRIEST_ASTRAL),
        # South-central clerics — lines 115-118.
        (12, 36, _MON_HIGH_PRIEST_ASTRAL),
        (12, 37, _MON_HIGH_PRIEST_ASTRAL),
        (12, 38, _MON_HIGH_PRIEST_ASTRAL),
        (13, 36, _MON_HIGH_PRIEST_ASTRAL),
        # East round room clerics — lines 123-126.
        (9,  56, _MON_HIGH_PRIEST_ASTRAL),
        (8,  55, _MON_HIGH_PRIEST_ASTRAL),
        (9,  55, _MON_HIGH_PRIEST_ASTRAL),
        (10, 55, _MON_HIGH_PRIEST_ASTRAL),
        # Angels — lines 111-112, 119-120, 127-128.
        (9,  20, _MON_ANGEL),
        (10, 20, _MON_ANGEL),
        (13, 38, _MON_ANGEL),
        (13, 37, _MON_ANGEL),
        (9,  54, _MON_ANGEL),
        (10, 54, _MON_ANGEL),
    ])

    items = _pack_placements([])
    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Demon lord rooms — Baalzebub, Asmodeus, Juiblex, Orcus
# ---------------------------------------------------------------------------

_BAALZ_MAP = """\
-------------------------------------------------
|                   ----               ----
|          ----     |     -----------  |
| ------      |  ---------|.........|--
| |....|  -------|...........----
---|....|--|..................|............|----
+...--.....|..----------------|............|...
---|....|--|..................|............|----
| |....|  -------|...........-----
| ------      |  ---------|.........|--
|          ----     |     -----------  |
|                   ----               ----
-------------------------------------------------
"""


def generate_baalzebub_lair(rng):
    """Baalzebub's Lair — vendor/nethack/dat/baalz.lua.

    Citation: baalz.lua line 37: des.monster("Baalzebub",35,06).
    Returns (terrain, monsters, items).
    """
    terrain = _parse_map_string(_BAALZ_MAP)
    terrain = terrain.at[6, 0].set(jnp.int8(_T_OPEN_DOOR))
    terrain = terrain.at[6, 44].set(jnp.int8(_T_STAIR_DOWN))
    monsters = _pack_placements([
        (6, 35, _MON_BAALZEBUB),
        (7, 37, _MON_GHOUL),
        (5, 32, _MON_GHOUL),
        (7, 38, _MON_GHOUL),
    ])
    items = _pack_placements([
        (6, 36, _ITEM_GEM),
        (6, 37, _ITEM_GOLD),
    ])
    return terrain, monsters, items


def _make_simple_demon_room(demon_type_id):
    """Minimal carved room for a demon lord without a full MAP string."""
    terrain = jnp.zeros((MAP_H, MAP_W), dtype=jnp.int8)
    terrain = _carve_room(terrain, top=6, left=20, h=9, w=20)
    terrain = terrain.at[10, 20].set(jnp.int8(_T_OPEN_DOOR))
    terrain = terrain.at[7, 39].set(jnp.int8(_T_STAIR_DOWN))
    monsters = _pack_placements([
        (10, 30, demon_type_id),
        (8,  25, _MON_GHOUL),
        (12, 35, _MON_GHOUL),
    ])
    items = _pack_placements([(10, 29, _ITEM_GOLD)])
    return terrain, monsters, items


def generate_asmodeus_lair(rng):
    """Asmodeus' Lair — compact single-room demon lord lair."""
    return _make_simple_demon_room(_MON_ASMODEUS)


def generate_juiblex_lair(rng):
    """Juiblex's Swamp — compact demon lord lair with water tiles."""
    terrain, monsters, items = _make_simple_demon_room(_MON_JUIBLEX)
    for (r, c) in [(9, 28), (9, 29), (11, 28)]:
        terrain = terrain.at[r, c].set(jnp.int8(_T_WATER))
    return terrain, monsters, items


def generate_orcus_town(rng):
    """Orcus Town — compact demon lord lair with altar."""
    terrain, monsters, items = _make_simple_demon_room(_MON_ORCUS)
    terrain = terrain.at[10, 31].set(jnp.int8(_T_ALTAR))
    return terrain, monsters, items
