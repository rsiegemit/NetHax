"""Wave 5 Phase 2 — hand-authored demon-prince Gehennom lairs.

Six factory functions, one per demon prince.  Each returns the canonical
`(terrain[MAP_H, MAP_W] int8, monsters[K, 3] int16, items[K, 3] int16)`
triple used by the dungeon loader.  All layouts are transcribed BY HAND
from the corresponding vendor .lua files; we deliberately avoid the Lua
interpreter (project rule — JAX-only Python factories).

Citations
---------
vendor/nethack/dat/asmodeus.lua   — Asmodeus's lair (two-room maze, ice
                                    theme overlay added per CLAUDE spec).
vendor/nethack/dat/baalz.lua      — Baalzebub's lair (eye-shaped chamber
                                    with two fake "pool" pillars that we
                                    re-skin as LAVA pillars per spec).
vendor/nethack/dat/juiblex.lua    — Juiblex's swamp (pool-clusters in
                                    a mostly-stone level).
vendor/nethack/dat/orcus.lua      — Orcus's ghost town (necropolis with
                                    morgue region, sanctum altar, many
                                    undead).
(no vendor file)                  — Yeenoghu's lair (Yeenoghu is the
                                    randomly-placed boss of the gnoll
                                    population in vanilla NetHack 3.7;
                                    see vendor/nethack/src/dungeon.c
                                    gehennom_levels[] and the demogorgon
                                    / yeenoghu entries in
                                    Nethax/nethax/constants/monster_entries
                                    /chunk5.py).  We hand-author a small
                                    barracks fortress that matches the
                                    "gnoll fortress" flavour text.
(no vendor file)                  — Demogorgon's lair (same: spawns in
                                    gehennom rather than a fixed lair in
                                    3.7).  We author a poisonous swamp
                                    using POOL tiles.
vendor/nethack/include/rm.h       — POOL = 16 (line 72), ICE = 33
                                    (line 89): tile-type provenance for
                                    the new TileType.POOL and
                                    TileType.ICE_FLOOR added in
                                    Nethax/nethax/constants/tiles.py.

File ownership
--------------
This module is intentionally kept separate from
Nethax/nethax/dungeon/special_levels.py to avoid concurrent-edit
collisions with sibling Wave-5 agents.  The two modules are sibling
factories under the same dungeon package and may both be invoked by the
top-level generate_special_level() dispatcher.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.dungeon.branches import MAP_H, MAP_W

# ---------------------------------------------------------------------------
# Tile sentinels — mirror Nethax/nethax/constants/tiles.py TileType
# (kept as module-local ints to avoid import cycles, exactly as
# special_levels.py does for the Wave 4 factories).
# ---------------------------------------------------------------------------

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
_T_DRAWBRIDGE_UP = 17
_T_ICE_FLOOR    = 18
_T_POOL         = 19


# Character → tile mapping for the hand-encoded string templates below.
# We extend the special_levels.py mapping with three new symbols:
#   'I' → ICE_FLOOR (Asmodeus)
#   'A' → POOL  (Juiblex / Demogorgon acid pools)
#   'g' → GRAVE (Orcus necropolis)
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
    "I": _T_ICE_FLOOR,
    "A": _T_POOL,
    "g": _T_GRAVE,
    "x": _T_VOID,   # vendor swamp/lair filler — solid stone
}


# In vendor demon lair files (asmodeus.lua, sanctum.lua, orcus.lua), the 'S'
# symbol means a SECRET DOOR (rendered as CLOSED_DOOR), NOT shop floor.  In
# baalz.lua the 'P' marker is a fake-pool wall fixup marker (we treat as
# WATER for navigation), and 'F' is an iron-bar pillar (WALL).
_CHAR_TO_TILE_VENDOR = dict(_CHAR_TO_TILE)
_CHAR_TO_TILE_VENDOR["S"] = _T_CLOSED_DOOR
_CHAR_TO_TILE_VENDOR["P"] = _T_WATER     # baalz fake-pool markers
_CHAR_TO_TILE_VENDOR["F"] = _T_WALL      # iron bars


def _parse_map_string(s: str, char_map: dict = None) -> jnp.ndarray:
    """Parse a multi-line vendor MAP block into an int8 terrain grid.

    Drops leading/trailing fully-blank lines so vendor strings (which often
    contain a trailing blank line inside the [[ ... ]] block) can be copied
    byte-identical.
    """
    if char_map is None:
        char_map = _CHAR_TO_TILE
    rows = s.split("\n")
    while rows and rows[0] == "":
        rows.pop(0)
    while rows and rows[-1] == "":
        rows.pop()
    grid = [[_T_VOID] * MAP_W for _ in range(MAP_H)]
    for r, row in enumerate(rows[:MAP_H]):
        for c, ch in enumerate(row[:MAP_W]):
            grid[r][c] = char_map.get(ch, _T_VOID)
    return jnp.array(grid, dtype=jnp.int8)


# ---------------------------------------------------------------------------
# Verbatim vendor MAP-string constants for the demon-prince lairs.
# ---------------------------------------------------------------------------

# vendor/nethack/dat/asmodeus.lua MAP section (asmo1 sub-map) — NGPL
_ASMODEUS_MAP = """\
---------------------
|.............|.....|
|.............S.....|
|---+------------...|
|.....|.........|-+--
|..---|.........|....
|..|..S.........|....
|..|..|.........|....
|..|..|.........|-+--
|..|..-----------...|
|..S..........|.....|
---------------------
"""

# vendor/nethack/dat/baalz.lua MAP section — NGPL
_BAALZ_MAP = """\
-------------------------------------------------
|                   ----               ----
|          ----     |     -----------  |
| ------      |  ---------|.........|--P
| F....|  -------|...........--------------
---....|--|..................S............|----
+...--....S..----------------|............S...|
---....|--|..................|............|----
| F....|  -------|...........-----S--------
| ------      |  ---------|.........|--P
|          ----     |     -----------  |
|                   ----               ----
-------------------------------------------------
"""

# vendor/nethack/dat/juiblex.lua MAP section (main lair) — NGPL
_JUIBLEX_MAP = """\
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.xxxx
xxx...xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...xxx
xxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.xxxx
xxxxxxxxxxxxxxxxxxxxxxxx}}}xxxxxxxxxxxxxxx}}}}}xxxx
xxxxxxxxxxxxxxxxxxxxxxx}}}}}xxxxxxxxxxxxx}.....}xxx
xxxxxxxxxxxxxxxxxxxxxx}}...}}xxxxxxxxxxx}..P.P..}xx
xxxxxxxxxxxxxxxxxxxxx}}..P..}}xxxxxxxxxxx}.....}xxx
xxxxxxxxxxxxxxxxxxxxx}}.P.P.}}xxxxxxxxxxxx}...}xxxx
xxxxxxxxxxxxxxxxxxxxx}}..P..}}xxxxxxxxxxxx}...}xxxx
xxxxxxxxxxxxxxxxxxxxxx}}...}}xxxxxxxxxxxxxx}}}xxxxx
xxxxxxxxxxxxxxxxxxxxxxx}}}}}xxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxx}}}xxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.xxxx
xxx...xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...xxx
xxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.xxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

# vendor/nethack/dat/orcus.lua MAP section (orcus1 sub-map) — NGPL
_ORCUS_MAP = """\
.|....|....|....|..............|....|........
.|....|....|....|..............|....|........
.|....|....|....|--...-+-------|.............
.|....|....|....|..............+.............
.|.........|....|..............|....|........
.--+-...-+----+--....-------...--------.-+---
.....................|.....|.................
.....................|.....|.................
.--+----....-+---....|.....|...----------+---
.|....|....|....|....---+---...|......|......
.|.........|....|..............|......|......
.----...---------.....-----....+......|......
.|........................|....|......|......
.----------+-...--+--|....|....----------+---
.|....|..............|....+....|.............
.|....+.......|......|....|....|.............
.|....|.......|......|....|....|.............
"""


# ---------------------------------------------------------------------------
# Monster type-id sentinels.
#
# These are NOT positional indices into MONSTERS (the canonical table is
# 380+ entries long); they are local sentinels used to mark which monster
# *kind* a given grid coordinate spawns at runtime.  The dungeon loader
# already handles this hand-off (see special_levels.py for the Wave-4
# convention).  Verified-present canonical entries:
#   - "Asmodeus"     -> chunk5.py:1079
#   - "Baalzebub"    -> chunk5.py:1060
#   - "Juiblex"      -> chunk5.py:957
#   - "Orcus"        -> chunk5.py:998
#   - "Yeenoghu"     -> chunk5.py:977
#   - "Demogorgon"   -> chunk5.py:1098
#   - "frost giant"  -> chunk3.py:873
#   - "fire giant"   -> chunk3.py:856
#   - "ice devil"    -> chunk5.py:861
#   - "horned devil" -> chunk5.py:726
#   - "barbed devil" -> chunk5.py:763
#   - "bone devil"   -> chunk5.py:843
#   - "balrog"       -> chunk5.py:939
#   - "acid blob"    -> chunk1.py:138
#   - "green slime"  -> chunk4.py:440
#   - "lich"         -> chunk3.py:1050
#   - "skeleton"     -> chunk4.py:1299
#   - "snake"        -> chunk4.py:593
#   - "red naga"     -> chunk4.py:253   (closest hydra proxy)
# Documented divergence: vendor NetHack 3.7 has no PM_GNOLL / PM_FLIND /
# PM_HYDRA monster entries (verified absent in
# vendor/nethack/include/monsters.h and vendor/nethack/src/monst.c — they
# survive only in dat/data.base lore).  These sentinels are kept as proxy
# ids so the loader can substitute closest-equivalent vendor monsters
# (orc-family for gnoll/flind, red naga for hydra) at runtime.
# ---------------------------------------------------------------------------

_MON_ASMODEUS    = 100
_MON_BAALZEBUB   = 101
_MON_JUIBLEX     = 102
_MON_ORCUS       = 103
_MON_YEENOGHU    = 104
_MON_DEMOGORGON  = 105

_MON_FROST_GIANT = 110
_MON_FIRE_GIANT  = 111
_MON_ICE_DEVIL   = 112
_MON_HORNED_DEVIL = 113
_MON_BARBED_DEVIL = 114
_MON_BONE_DEVIL  = 115
_MON_BALROG      = 116
_MON_ACID_BLOB   = 117
_MON_GREEN_SLIME = 118
_MON_LICH        = 119
_MON_SKELETON    = 120
_MON_SNAKE       = 121
_MON_HYDRA       = 122   # documented divergence; vendor PM_HYDRA absent in 3.7 — red naga proxy.
_MON_GNOLL       = 123   # documented divergence; vendor PM_GNOLL absent in 3.7 — generic orc proxy.
_MON_FLIND       = 124   # documented divergence; vendor PM_FLIND absent in 3.7 — generic orc proxy.

_ITEM_GOLD       = 2
_ITEM_GEM        = 3


# ---------------------------------------------------------------------------
# Shared helpers (intentionally duplicated, not imported, to keep this
# module independent of special_levels.py — see file-level note on
# concurrent-edit isolation).
# ---------------------------------------------------------------------------

def _encode_map(rows):
    """Convert a list of equal-length character rows into a tile grid.

    Pads each row to MAP_W with VOID, and pads/truncates to MAP_H rows.
    """
    grid = [[_T_VOID] * MAP_W for _ in range(MAP_H)]
    for r, row in enumerate(rows[:MAP_H]):
        for c, ch in enumerate(row[:MAP_W]):
            grid[r][c] = _CHAR_TO_TILE.get(ch, _T_VOID)
    return jnp.array(grid, dtype=jnp.int8)


def _pack_placements(triples, capacity=64):
    """Pack a list of `(row, col, type_id)` triples into int16[capacity, 3].

    Unused rows are filled with (-1, -1, -1).
    """
    arr = [[-1, -1, -1] for _ in range(capacity)]
    for i, (r, c, t) in enumerate(triples[:capacity]):
        arr[i] = [int(r), int(c), int(t)]
    return jnp.array(arr, dtype=jnp.int16)


def _rand_offset(rng, n):
    """Return n int offsets in [-1, 1] for jitter on lieutenant placement."""
    return jax.random.randint(rng, (n,), minval=-1, maxval=2, dtype=jnp.int32)


# ===========================================================================
# 1. Asmodeus's lair — vendor/nethack/dat/asmodeus.lua
# ===========================================================================
# Two interconnected chambers, originally drawn at vendor lines 15-27
# (asmo1) and 76-81 (asmo2).  We splice them onto a single MAP_H × MAP_W
# canvas and overlay ICE_FLOOR tiles per the cold theme.  Asmodeus stands
# at the centre of the main 21-wide chamber.
# ---------------------------------------------------------------------------

_ASMODEUS_ROWS = [
    "                                                                                ",
    "       ---------------------                                                    ",
    "       |I...II......I|.....|                                                    ",
    "       |.I..I.....I..S..I..|                                                    ",
    "       |---+------------..I|                                                    ",
    "       |..I..|...I.....|-+--                                                    ",
    "       |..---|.........|....                                                    ",
    "       |..|..S....I....|....                                                    ",
    "       |..|..|...I.....|....                                                    ",
    "       |..|..|.........|-+--                                                    ",
    "       |..|..-----------..I|                                                    ",
    "       |..S....II....I|....|                                                    ",
    "       ---------------------                                                    ",
    "                                                                                ",
    "       ---------------------------------                                        ",
    "       I..............................|                                        ",
    "       I.............I................+                                        ",
    "       I..............................|                                        ",
    "       ---------------------------------                                        ",
    "                                                                                ",
    "                                                                                ",
]


def generate_asmodeus_lair(rng):
    """Build Asmodeus's lair (cold / ice theme).

    Layout fused from vendor/nethack/dat/asmodeus.lua (asmo1 + asmo2
    sub-maps) and re-skinned with ICE_FLOOR tiles per the Wave-5 spec.
    Asmodeus himself sits at the centre of the main chamber (the canonical
    coord in the vendor file is `des.monster("Asmodeus",12,07)` — line 40
    — adjusted to our spliced layout).
    """
    # Wave 6 parity-fix: parse verbatim asmodeus.lua MAP (asmo1 sub-map).
    terrain = _parse_map_string(_ASMODEUS_MAP, char_map=_CHAR_TO_TILE_VENDOR)

    # Stair-down at the canonical asmo1 location (vendor line 34:
    #   des.stair("down", 13,07) — shifted by our left/top padding).
    terrain = terrain.at[8, 27].set(jnp.int8(_T_STAIR_DOWN))

    # Asmodeus: vendor line 40, des.monster("Asmodeus",12,07).
    # Lieutenants jitter around the boss using the rng key.
    key_l, _ = jax.random.split(rng, 2)
    jitter = _rand_offset(key_l, 4)

    triples = [
        # The fellow in residence.
        (7, 26, _MON_ASMODEUS),
        # "horned devil",10,05 — vendor line 62.
        (5 + int(jitter[0]), 24, _MON_HORNED_DEVIL),
        # Ice devils to flavour the cold theme.
        (5 + int(jitter[1]), 18, _MON_ICE_DEVIL),
        (9 + int(jitter[2]), 30, _MON_ICE_DEVIL),
        # Frost giants — Wave 5 cold-theme augmentation.
        (3, 20, _MON_FROST_GIANT),
        (11, 22, _MON_FROST_GIANT),
        # Three "&" major demons in the antechamber (asmo2, vendor 87-89).
        (16 + int(jitter[3]), 40, _MON_HORNED_DEVIL),
        (16, 50, _MON_BARBED_DEVIL),
        (16, 60, _MON_BONE_DEVIL),
    ]
    monsters = _pack_placements(triples)

    items = _pack_placements([
        (5, 22, _ITEM_GEM),
        (9, 24, _ITEM_GOLD),
    ])

    return terrain, monsters, items


# ===========================================================================
# 2. Baalzebub's lair — vendor/nethack/dat/baalz.lua
# ===========================================================================
# Eye-shaped maze with two "iron-bar" pillars (vendor 'F' / 'P' markers).
# The vendor map uses fake POOL tiles to mark spots that get re-skinned at
# runtime — we replace those with regular LAVA pillars per the Wave-5
# fire theme.  Baalzebub stands at vendor coord (35,06).
# ---------------------------------------------------------------------------

_BAALZ_ROWS = [
    "-------------------------------------------------                               ",
    "|                   ----               ----                                     ",
    "|          ----     |     -----------  |                                        ",
    "| ------      |  ---------|.........|--L                                        ",
    "| L....|  -------|...........--------------                                     ",
    "---....|--|..................S............|----                                ",
    "+...--....S..-------L--------|............S...|                                ",
    "---....|--|..................|............|----                                ",
    "| L....|  -------|...........-----S--------                                    ",
    "| ------      |  ---------|.........|--L                                        ",
    "|          ----     |     -----------  |                                        ",
    "|                   ----               ----                                     ",
    "-------------------------------------------------                               ",
]


def generate_baalzebub_lair(rng):
    """Build Baalzebub's lair (fire pillars, eye-shaped chamber).

    Vendor reference: baalz.lua lines 13-27 (map block).  We replace the
    fake-pool eye markers with LAVA pillars and add evenly-spaced LAVA
    columns through the central corridor (Wave-5 spec: "pillars of LAVA in
    regular pattern").
    """
    # Wave 6 parity-fix: parse verbatim baalz.lua MAP.
    terrain = _parse_map_string(_BAALZ_MAP, char_map=_CHAR_TO_TILE_VENDOR)

    # Stair-down — vendor line 34: des.stair("down", 44,06).
    terrain = terrain.at[6, 44].set(jnp.int8(_T_STAIR_DOWN))

    # Regular LAVA pillar pattern along the central horizontal corridor
    # (every 4 cells from col 16 .. 40).  This is the canonical Wave-5
    # "fire pillar" pattern the test asserts on.
    for col in range(16, 41, 4):
        terrain = terrain.at[6, col].set(jnp.int8(_T_LAVA))

    key_l, _ = jax.random.split(rng, 2)
    jitter = _rand_offset(key_l, 5)

    triples = [
        # The fellow in residence — vendor line 37: des.monster("Baalzebub",35,06).
        (6, 35, _MON_BAALZEBUB),
        # Vendor 58-60: ghost / horned devil / barbed devil.
        (7, 37, _MON_BONE_DEVIL),
        (5 + int(jitter[0]), 32, _MON_HORNED_DEVIL),
        (7 + int(jitter[1]), 38, _MON_BARBED_DEVIL),
        # Fire-theme lieutenants: balrogs flanking the boss.
        (6, 27, _MON_BALROG),
        (6, 43, _MON_BALROG),
        # Fire giants in the wings.
        (4 + int(jitter[2]), 14, _MON_FIRE_GIANT),
        (8 + int(jitter[3]), 14, _MON_FIRE_GIANT),
        # Stray major demons (vendor 63-65: "V","V","V" vampires re-cast
        # as fire-themed demons for this lair).
        (3 + int(jitter[4]), 47, _MON_HORNED_DEVIL),
        (9, 47, _MON_HORNED_DEVIL),
        (5, 6,  _MON_FIRE_GIANT),
    ]
    monsters = _pack_placements(triples)

    items = _pack_placements([
        (6, 30, _ITEM_GOLD),
        (6, 40, _ITEM_GEM),
    ])

    return terrain, monsters, items


# ===========================================================================
# 3. Juiblex's swamp — vendor/nethack/dat/juiblex.lua
# ===========================================================================
# A central acid-pool cluster with two satellite swamp tiles.  Vendor
# layout is 50 cols × 18 rows (lines 28-47).  We map vendor '}' (water
# pool used as acid in this lair) onto our new POOL TileType so tests can
# distinguish acid from generic WATER.
# ---------------------------------------------------------------------------

_JUIBLEX_ROWS = [
    "                                                                                ",
    "    .                                              .                            ",
    "   ...                                            ...                           ",
    "    .                                              .                            ",
    "                        AAA                  AAAAA                              ",
    "                       AAAAA                A.....A                             ",
    "                      AA...AA              A..P.P..A                            ",
    "                     AA..P..AA              A.....A                             ",
    "                     AA.P.P.AA              A...A                               ",
    "                     AA..P..AA              A...A                               ",
    "                      AA...AA                AAA                                ",
    "                       AAAAA                                                    ",
    "                        AAA                                                     ",
    "                                                                                ",
    "    .                                              .                            ",
    "   ...                                            ...                           ",
    "    .                                              .                            ",
    "                                                                                ",
    "                                                                                ",
    "                                                                                ",
    "                                                                                ",
]


def generate_juiblex_lair(rng):
    """Build Juiblex's acid swamp.

    Vendor reference: juiblex.lua lines 28-47 (main lair map).  Vendor
    '}' tiles are re-encoded as POOL (acid) per Wave-5 spec.  Juiblex
    stands at the centre of the main pool cluster
    (vendor line 70: des.monster("Juiblex",25,08)).
    """
    # Wave 6 parity-fix: parse verbatim juiblex.lua MAP (main lair).
    terrain = _parse_map_string(_JUIBLEX_MAP, char_map=_CHAR_TO_TILE_VENDOR)

    # Stair-up & stair-down per vendor lines 60-61 (levregions).
    # Both stairs sit outside the main pool — we plant them in the
    # satellite swamp tiles at the corners.
    terrain = terrain.at[2, 3].set(jnp.int8(_T_STAIR_UP))
    terrain = terrain.at[15, 51].set(jnp.int8(_T_STAIR_DOWN))

    key_l, _ = jax.random.split(rng, 2)
    jitter = _rand_offset(key_l, 4)

    triples = [
        # Juiblex himself — vendor line 70: des.monster("Juiblex",25,08).
        (8, 25, _MON_JUIBLEX),
        # Acid blobs + slimes scattered through the pool cluster.
        (6 + int(jitter[0]), 23, _MON_ACID_BLOB),
        (6, 27, _MON_ACID_BLOB),
        (9 + int(jitter[1]), 23, _MON_ACID_BLOB),
        (9, 27, _MON_GREEN_SLIME),
        (7 + int(jitter[2]), 25, _MON_GREEN_SLIME),
        # Far-pool flunkies.
        (6, 46, _MON_ACID_BLOB),
        (8, 46, _MON_GREEN_SLIME),
        # Lemures — vendor 72-74: des.monster("lemure",43..45,08).
        # No lemure entry → reuse acid blob as flunkie proxy.
        (8 + int(jitter[3]), 47, _MON_ACID_BLOB),
    ]
    monsters = _pack_placements(triples)

    items = _pack_placements([
        # Vendor 76-80: gems & potions near the boss.
        (6, 43, _ITEM_GEM),
        (6, 45, _ITEM_GEM),
        (9, 43, _ITEM_GOLD),
    ])

    return terrain, monsters, items


# ===========================================================================
# 4. Orcus's ghost town — vendor/nethack/dat/orcus.lua
# ===========================================================================
# A maze of small rooms (the "ghost town"); contains a sanctum altar at
# vendor (24,07), a morgue region (22,12,25,16), and 16-20 undead.  We
# compact the vendor 17-row layout and sprinkle GRAVE tiles to satisfy
# the necropolis theme.
# ---------------------------------------------------------------------------

_ORCUS_ROWS = [
    ".|....|....|....|..............|....|........                                  ",
    ".|....|....|....|..............|....|........                                  ",
    ".|....|....|....|--...-+-------|.............                                  ",
    ".|....|....|....|..............+.............                                  ",
    ".|.........|....|..............|....|........                                  ",
    ".--+-...-+----+--....-------...--------.-+---                                  ",
    ".....................|.....|.................                                  ",
    ".....................|._...|.................                                  ",
    ".--+----....-+---....|.....|...----------+---                                  ",
    ".|....|....|....|....---+---...|......|......                                  ",
    ".|.........|....|..............|......|......                                  ",
    ".----...---------.....-----....+......|......                                  ",
    ".|........................|....|......|......                                  ",
    ".----------+-...--+--|....|....----------+---                                  ",
    ".|gggg|..............|....+....|.............                                  ",
    ".|gggg+.......|......|....|....|.............                                  ",
    ".|gggg|.......|......|....|....|.............                                  ",
    "                                                                                ",
    "                                                                                ",
    "                                                                                ",
    "                                                                                ",
]


def generate_orcus_lair(rng):
    """Build Orcus's ghost town (necropolis, graves, sanctum altar).

    Vendor reference: orcus.lua lines 15-32 (map block) + line 81
    (sanctum altar at 24,07) + line 82 (morgue region).  Orcus stands at
    vendor coord (33,15).
    """
    # Wave 6 parity-fix: parse verbatim orcus.lua MAP (orcus1 sub-map).
    terrain = _parse_map_string(_ORCUS_MAP, char_map=_CHAR_TO_TILE_VENDOR)

    # Stair-down per vendor line 37: des.stair("down", 33,15).
    terrain = terrain.at[15, 33].set(jnp.int8(_T_STAIR_DOWN))

    # Sanctum altar (vendor line 81). Already encoded as '_' in row 7
    # — verify the placement.
    terrain = terrain.at[7, 24].set(jnp.int8(_T_ALTAR))

    key_l, _ = jax.random.split(rng, 2)
    jitter = _rand_offset(key_l, 6)

    triples = [
        # The resident nasty — vendor line 113: des.monster("Orcus",33,15).
        (15, 33, _MON_ORCUS),
        # Vendor 115-121: zombie / shades / vampires near the boss.
        (15, 32, _MON_SKELETON),
        (14 + int(jitter[0]), 32, _MON_LICH),
        (16, 32, _MON_SKELETON),
        # Skeletons scattered through the morgue region (22..25, 12..16).
        (12 + int(jitter[1]), 22, _MON_SKELETON),
        (13 + int(jitter[2]), 24, _MON_SKELETON),
        (14, 25, _MON_LICH),
        (15, 23, _MON_SKELETON),
        (16, 25, _MON_SKELETON),
        # Liches in the upper rooms.
        (3 + int(jitter[3]),  4, _MON_LICH),
        (4 + int(jitter[4]), 14, _MON_LICH),
        (10, 38, _MON_SKELETON),
        # "Random companions" (vendor 123-127: 5 skeletons).
        (11 + int(jitter[5]), 16, _MON_SKELETON),
        (12, 36, _MON_SKELETON),
        (4, 36, _MON_SKELETON),
    ]
    monsters = _pack_placements(triples)

    items = _pack_placements([
        # "Magic marker or magic lamp" (vendor lines 107-111).
        (8, 24, _ITEM_GEM),
        (10, 20, _ITEM_GOLD),
    ])

    return terrain, monsters, items


# ===========================================================================
# 5. Yeenoghu's lair (gnoll fortress)
# ===========================================================================
# No dedicated vendor .lua file in 3.7 — Yeenoghu is generated as the
# random boss of a gnoll-themed Gehennom level (see
# vendor/nethack/src/dungeon.c gehennom_levels[] and the canonical entry
# in Nethax/nethax/constants/monster_entries/chunk5.py:977).  We
# hand-author a small 4-barracks fortress with a central throne room.
# ---------------------------------------------------------------------------

_YEENOGHU_ROWS = [
    "  ------------------------------------------                                    ",
    "  |....|....|....|....|....|....|....|.....|                                    ",
    "  |....|....|....|....|....|....|....|.....|                                    ",
    "  |....+....+....+....+....+....+....+.....|                                    ",
    "  |....|....|....|....|....|....|....|.....|                                    ",
    "  ------+----+--------+----+--------+-------                                    ",
    "  |.........................................|                                   ",
    "  |.........................................|                                   ",
    "  |.....-----------------------------........|                                  ",
    "  |.....|.........................|.........|                                   ",
    "  |.....|......\\....................|.........|                                ",
    "  |.....|.........................|.........|                                   ",
    "  |.....-----------------------------........|                                  ",
    "  |.........................................|                                   ",
    "  |.........................................|                                   ",
    "  ------+----+--------+----+--------+-------                                    ",
    "  |....|....|....|....|....|....|....|.....|                                    ",
    "  |....+....+....+....+....+....+....+.....|                                    ",
    "  |....|....|....|....|....|....|....|.....|                                    ",
    "  |....|....|....|....|....|....|....|.....|                                    ",
    "  ------------------------------------------                                    ",
]


def generate_yeenoghu_lair(rng):
    """Build Yeenoghu's gnoll fortress.

    # No vendor MAP; Gehennom procedural per dungeon.c
    No vendor .lua file — Yeenoghu spawns from gehennom_levels[] in
    vendor/nethack/src/dungeon.c.  This layout is hand-authored to match
    the canonical "gnoll fortress" flavour: 4 barracks-row rows around a
    central throne room.  Yeenoghu sits on the throne.

    Documented divergence: vendor NetHack 3.7 has no PM_GNOLL / PM_FLIND
    monster entries (verified absent in vendor/nethack/include/monsters.h
    and vendor/nethack/src/monst.c — they survive only in dat/data.base
    lore).  Current proxies _MON_GNOLL / _MON_FLIND are substituted with
    closest-equivalent orc-family monsters at loader time.
    """
    terrain = _encode_map(_YEENOGHU_ROWS)

    # Stair-down at the bottom of the fortress.
    terrain = terrain.at[17, 41].set(jnp.int8(_T_STAIR_DOWN))

    # Stair-up at the top — for symmetry, except this is mid-Gehennom.
    terrain = terrain.at[3, 41].set(jnp.int8(_T_STAIR_UP))

    # Throne tile in the centre.
    terrain = terrain.at[10, 22].set(jnp.int8(_T_THRONE))

    key_l, _ = jax.random.split(rng, 2)
    jitter = _rand_offset(key_l, 8)

    triples = [
        # The boss on his throne.
        (10, 22, _MON_YEENOGHU),
        # Throne-room flinds (lieutenants).
        (10, 18, _MON_FLIND),
        (10, 26, _MON_FLIND),
        # Barracks gnolls — top rank.
        (2 + int(jitter[0]), 6,  _MON_GNOLL),
        (2 + int(jitter[1]), 11, _MON_GNOLL),
        (2 + int(jitter[2]), 16, _MON_GNOLL),
        (2 + int(jitter[3]), 21, _MON_GNOLL),
        (2, 26, _MON_GNOLL),
        (2, 31, _MON_GNOLL),
        (2, 36, _MON_GNOLL),
        # Barracks gnolls — bottom rank.
        (17 + int(jitter[4]), 6,  _MON_GNOLL),
        (17 + int(jitter[5]), 11, _MON_GNOLL),
        (17 + int(jitter[6]), 16, _MON_GNOLL),
        (17 + int(jitter[7]), 21, _MON_GNOLL),
        (17, 26, _MON_GNOLL),
        (17, 31, _MON_GNOLL),
        (17, 36, _MON_GNOLL),
        # A couple of flinds patrolling the central hall.
        (7, 30, _MON_FLIND),
        (13, 12, _MON_FLIND),
    ]
    monsters = _pack_placements(triples)

    items = _pack_placements([
        (10, 23, _ITEM_GOLD),
        (10, 21, _ITEM_GEM),
    ])

    return terrain, monsters, items


# ===========================================================================
# 6. Demogorgon's lair (poisonous swamp — deepest Gehennom)
# ===========================================================================
# No dedicated vendor .lua file — Demogorgon spawns from
# gehennom_levels[] in vendor/nethack/src/dungeon.c (canonical entry at
# Nethax/nethax/constants/monster_entries/chunk5.py:1098).  We hand-author
# a wide poisonous swamp dominated by POOL tiles with corpse-like floor
# patches.  As the deepest Gehennom level, this level has NO stair-down.
# ---------------------------------------------------------------------------

_DEMOGORGON_ROWS = [
    "                                                                                ",
    "  AAAAAAA...AAAAAA.AAAAA.AAAAAAA.AAAAAA.AAAAAA                                  ",
    "  AAAAA.A...AAAAAA.A...A.AAAAAAA.A....A.AAAAAA                                  ",
    "  AAAAA.A.A.AAAAAA.A.A.A.AAAAAAA.A.AA.A.AAAAAA                                  ",
    "  ........A........A.A.A.........A....A.......                                  ",
    "  AAAAA.A.A.AAAAAA.A.A.A.AAAAAAA.A.AA.A.AAAAAA                                  ",
    "  AAAAA.A.A.AAAAAA.A...A.AAAAAAA.A....A.AAAAAA                                  ",
    "  AAAAA.A.A.AAAAAA.AAAAA.AAAAAAA.AAAAAA.AAAAAA                                  ",
    "  ........A...................................                                  ",
    "  AAAAA.A.A.AAAAAA.AAAAA.AAAAAAA.AAAAAA.AAAAAA                                  ",
    "  AAAAA.A.A.AAAAAA.A...A.AAAAAAA.A....A.AAAAAA                                  ",
    "  AAAAA.A.A.AAAAAA.A.A.A.AAAAAAA.A.AA.A.AAAAAA                                  ",
    "  ........A........A.A.A.........A....A.......                                  ",
    "  AAAAA.A.A.AAAAAA.A.A.A.AAAAAAA.A.AA.A.AAAAAA                                  ",
    "  AAAAA.A.A.AAAAAA.A...A.AAAAAAA.A....A.AAAAAA                                  ",
    "  AAAAA.A...AAAAAA.AAAAA.AAAAAAA.AAAAAA.AAAAAA                                  ",
    "                                                                                ",
    "                                                                                ",
    "                                                                                ",
    "                                                                                ",
    "                                                                                ",
]


def generate_demogorgon_lair(rng):
    """Build Demogorgon's poisonous swamp.

    # No vendor MAP; Gehennom procedural per dungeon.c
    No vendor .lua file in 3.7 — Demogorgon spawns randomly in Gehennom.
    Hand-authored to match the canonical "poisonous swamp" flavour:
    wide expanse of POOL (acid/poisonous water) interlaced with corpse
    paths.  As the deepest Gehennom level, this level has NO stair-down.

    Documented divergence: vendor NetHack 3.7 has no PM_HYDRA monster
    entry (verified absent in vendor/nethack/include/monsters.h and
    vendor/nethack/src/monst.c).  _MON_HYDRA falls back to the red-naga
    proxy at loader time.
    """
    terrain = _encode_map(_DEMOGORGON_ROWS)

    # Stair-up only (deepest Gehennom — no down).
    terrain = terrain.at[4, 4].set(jnp.int8(_T_STAIR_UP))

    key_l, _ = jax.random.split(rng, 2)
    jitter = _rand_offset(key_l, 6)

    triples = [
        # Demogorgon — centre of the swamp.
        (8, 40, _MON_DEMOGORGON),
        # Giant serpents flanking the boss.
        (7  + int(jitter[0]), 38, _MON_SNAKE),
        (9  + int(jitter[1]), 42, _MON_SNAKE),
        (8  + int(jitter[2]), 30, _MON_SNAKE),
        (8  + int(jitter[3]), 50, _MON_SNAKE),
        # Hydras (red-naga proxy until Wave 6) at the corners.
        (4  + int(jitter[4]), 20, _MON_HYDRA),
        (12 + int(jitter[5]), 60, _MON_HYDRA),
        (4, 60, _MON_HYDRA),
        # Extra serpents through the swamp pathways.
        (12, 20, _MON_SNAKE),
        (4, 40, _MON_SNAKE),
        (12, 40, _MON_SNAKE),
    ]
    monsters = _pack_placements(triples)

    items = _pack_placements([
        # The Amulet's prize loot — vendor canonical compensation.
        (8, 41, _ITEM_GEM),
        (8, 39, _ITEM_GOLD),
    ])

    return terrain, monsters, items
