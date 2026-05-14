"""Wave 5 Phase 4b — Endgame: the five Astral planes.

Hand-encoded factories for the five endgame levels.  Each returns the same
(terrain, monsters, items) triple used by the Wave 4/5 special-level
factories in special_levels.py:

    generate_earth_plane(rng)  — earth.lua : caverns in solid rock.
    generate_air_plane(rng)    — air.lua   : almost-no-floor; flight needed.
    generate_fire_plane(rng)   — fire.lua  : lava lake with floor islands.
    generate_water_plane(rng)  — water.lua : pool everywhere, floor bubbles.
    generate_astral_plane(rng) — astral.lua: open field with 3 altars.

Citations:
    vendor/nethack/dat/earth.lua, air.lua, fire.lua, water.lua, astral.lua
    vendor/nethack/src/end.c::done_ascend
    vendor/nethack/src/wizard.c::amulet

Status: Wave 5 — high-fidelity hand-encoded layouts.  Layouts are
direct (subset) transcriptions of the canonical vendor .lua files; we
sample fewer monsters than the vendor to fit within the [K, 3] placement
capacity used elsewhere in the dungeon module.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from Nethax.nethax.dungeon.branches import MAP_H, MAP_W


# ---------------------------------------------------------------------------
# Tile sentinels — match TileType values from constants/tiles.py without
# pulling the IntEnum into our JIT scope.
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
_T_POOL         = 19   # TileType.POOL — used for water plane


# ---------------------------------------------------------------------------
# Monster id sentinels — stable distinct ints used by tests / spawning.
# These do not have to align with the global MONSTERS table; downstream
# spawners use these as hints + name lookups (see branches.py /
# spawning.py for the canonical lookup).
# ---------------------------------------------------------------------------
_MON_EARTH_ELEMENTAL = 30
_MON_AIR_ELEMENTAL   = 31
_MON_FIRE_ELEMENTAL  = 32
_MON_WATER_ELEMENTAL = 33
_MON_SALAMANDER      = 34
_MON_KRAKEN          = 35
_MON_ALEAX           = 36   # Aleax angel (astral.lua line 62)
_MON_HIGH_PRIEST     = 19   # matches special_levels.py for cross-module consistency
_MON_ANGEL           = 37


# ---------------------------------------------------------------------------
# Altar alignment sentinels — written into the items array per altar so the
# ascension condition can look them up (row, col, align) without needing a
# separate per-tile alignment overlay.
# 0 = Lawful, 1 = Neutral, 2 = Chaotic.
# ---------------------------------------------------------------------------
ASTRAL_ALIGN_LAWFUL  = 0
ASTRAL_ALIGN_NEUTRAL = 1
ASTRAL_ALIGN_CHAOTIC = 2


def _empty_placements(capacity: int = 64) -> jnp.ndarray:
    """Return an int16[capacity, 3] placement array filled with (-1,-1,-1)."""
    return jnp.full((capacity, 3), -1, dtype=jnp.int16)


def _pack_placements(triples, capacity: int = 64) -> jnp.ndarray:
    """Pack (row, col, type_id) triples into an int16[capacity, 3] array.

    Unused rows are filled with (-1, -1, -1).
    """
    arr = [[-1, -1, -1] for _ in range(capacity)]
    for i, (r, c, t) in enumerate(triples[:capacity]):
        arr[i] = [int(r), int(c), int(t)]
    return jnp.asarray(arr, dtype=jnp.int16)


# ---------------------------------------------------------------------------
# Plane of Earth — earth.lua
# ---------------------------------------------------------------------------
# Mostly VOID ("rock"); scattered floor caverns.  Player arrives in the
# lower-right cavern.  The vendor map is 76 wide; we transcribe directly,
# truncating to MAP_W=80 by padding with VOID.
#
# Citation: vendor/nethack/dat/earth.lua lines 26-47.
# ---------------------------------------------------------------------------

_EARTH_ROWS = [
    "                                                                            ",
    "  ...                                                                       ",
    " ....                ..                                                     ",
    " .....             ...                                      ..              ",
    "  ....              ....                                     ...            ",
    "   ....              ...                ....                 ...      .     ",
    "    ..                ..              .......                 .      ..     ",
    "                                      ..  ...                        .      ",
    "              .                      ..    .                         ...    ",
    "             ..  ..                  .     ..                         .     ",
    "            ..   ...                        .                               ",
    "            ...   ...                                                       ",
    "              .. ...                                 ..                     ",
    "               ....                                 ..                      ",
    "                          ..                                       ...      ",
    "                         ..                                       .....     ",
    "  ...                                                              ...      ",
    " ....                                                                       ",
    "   ..                                                                       ",
    "                                                                            ",
]

# vendor/nethack/dat/earth.lua MAP section — NGPL  (byte-identical, 76 wide)
_EARTH_MAP = """\

  ...
 ....                ..
 .....             ...                                      ..
  ....              ....                                     ...
   ....              ...                ....                 ...      .
    ..                ..              .......                 .      ..
                                      ..  ...                        .
              .                      ..    .                         ...
             ..  ..                  .     ..                         .
            ..   ...                        .
            ...   ...
              .. ...                                 ..
               ....                                 ..
                          ..                                       ...
                         ..                                       .....
  ...                                                              ...
 ....
   ..

"""

# vendor/nethack/dat/air.lua MAP section — NGPL  (byte-identical, 76 wide, 20 rows of 'A')
_AIR_MAP = """\
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
"""

# vendor/nethack/dat/water.lua MAP section — NGPL  (byte-identical, 76 wide, 20 rows of 'W')
_WATER_MAP = """\
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
"""

# vendor/nethack/dat/astral.lua MAP section — NGPL  (byte-identical)
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

# vendor/nethack/dat/fire.lua MAP section — NGPL  (byte-identical)
_FIRE_MAP = """\
LL.............LL..............L...LL.........LL.................LL...........L
LL....LLLLLLLL............L...L.............LL....LLL.......................LL.
L....LL...................L......................LLLL................LL........
.....L.............LLLL...LL....LL...............LLLLL.............LLL.........
.L.LLLL..............LL....L.....LLL..............LLLL..............LLLL......L
LL..........LLLL...LLLL...LLL....LLL......L........LLLL....LL........LLL......L
LL........LLLLLLL...LL.....L......L......LL.........LL......LL........LL...L...
L.........LL..LLL..LL......LL......LLLL..L.........LL......LLL............LL...
......L..LL....LLLLL.................LLLLLLL.......L......LL............LLLLLL.
......L..L.....LL.LLLL.......L............L........LLLLL.LL......LL.........LL.
......LL........L...LL......LL.............LLL.....L...LLL.......LLL.........L.
.L.....LLLLLL........L.......LLL.............L....LL...L.LLL......LLLLLLL......
LL..........LLLL............LL.L.............L....L...LL.........LLL..LLL......
.L...........................LLLLL...........LL...L...L........LLLL..LLLLLL...L
.L.....LLLL.............LL....LL.......LLL...LL.......L..LLL....LLLLLLL.......L
.........LLL.........LLLLLLLLLLL......LLLLL...L...........LL...LL...LL.........
...........LL.......LL.........LL.......LLL....L..LLL....LL.........LL.........
............LLLLLLLLL...........LL....LLL.......LLLLL.....LL........LL.........
.LL...............L.............LLLLLL............LL...LLLL.........LL.......L.
LL.....L..........................LL....................LL..................LLL
L.....LLL......................LLLLL.........L.........LLLLLLLL..............LL
"""


_PLANE_CHAR_TO_TILE = {
    " ": _T_VOID,
    ".": _T_FLOOR,
    "L": _T_LAVA,
    "W": _T_WATER,
    "A": _T_VOID,        # Air → VOID per Wave-5 convention
    "|": _T_WALL,
    "-": _T_WALL,
    "+": _T_CLOSED_DOOR,
    "<": _T_STAIR_UP,
    ">": _T_STAIR_DOWN,
}


def _parse_plane_map(s: str) -> jnp.ndarray:
    """Parse a vendor plane MAP block into an int8 terrain grid."""
    rows = s.split("\n")
    while rows and rows[0] == "":
        rows.pop(0)
    while rows and rows[-1] == "":
        rows.pop()
    grid = [[_T_VOID] * MAP_W for _ in range(MAP_H)]
    for r, row in enumerate(rows[:MAP_H]):
        for c, ch in enumerate(row[:MAP_W]):
            grid[r][c] = _PLANE_CHAR_TO_TILE.get(ch, _T_VOID)
    return jnp.asarray(grid, dtype=jnp.int8)


def _encode_earth(rows) -> jnp.ndarray:
    """Convert earth.lua rows to terrain (VOID='rock', '.'=FLOOR)."""
    grid = [[_T_VOID] * MAP_W for _ in range(MAP_H)]
    for r, row in enumerate(rows[:MAP_H]):
        for c, ch in enumerate(row[:MAP_W]):
            if ch == ".":
                grid[r][c] = _T_FLOOR
            # everything else (space) stays VOID
    return jnp.asarray(grid, dtype=jnp.int8)


def generate_earth_plane(rng) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Plane of Earth — solid rock with corridors / caverns.

    Earth elementals + a Sokoban-style boulder; the vendor places one boulder
    via ``des.object("boulder")`` (line 129) but we omit per-object boulder
    placement here.

    Reference: vendor/nethack/dat/earth.lua

    Returns
    -------
    (terrain, monsters, items)
        terrain  : int8[MAP_H, MAP_W] using TileType values.
        monsters : int16[64, 3] — (row, col, monster_type_id) triples.
        items    : int16[64, 3] — empty (-1 padded) by default.
    """
    del rng  # deterministic; layout is fully fixed.
    # Wave 6 parity-fix: parse byte-identical earth.lua MAP block.
    terrain = _parse_plane_map(_EARTH_MAP)

    # Earth elementals + role-flavour monsters from earth.lua lines 57-127.
    # vendor coords are (x, y) i.e. (col, row); flip to (row, col).
    monsters = _pack_placements([
        # lower-right arrival cavern (lines 57-64).
        (16, 67, _MON_EARTH_ELEMENTAL),
        (14, 67, _MON_EARTH_ELEMENTAL),
        (13, 52, _MON_EARTH_ELEMENTAL),
        (13, 53, _MON_EARTH_ELEMENTAL),
        # upper-right cavern (lines 66-78).
        (5,  70, _MON_EARTH_ELEMENTAL),
        (6,  69, _MON_EARTH_ELEMENTAL),
        (8,  70, _MON_EARTH_ELEMENTAL),
        (3,  60, _MON_EARTH_ELEMENTAL),
        (4,  61, _MON_EARTH_ELEMENTAL),
        (4,  62, _MON_EARTH_ELEMENTAL),
        (5,  61, _MON_EARTH_ELEMENTAL),
        # middle cavern (lines 80-91).
        (5,  40, _MON_EARTH_ELEMENTAL),
        (6,  39, _MON_EARTH_ELEMENTAL),
        (6,  41, _MON_EARTH_ELEMENTAL),
        (7,  38, _MON_EARTH_ELEMENTAL),
        (7,  43, _MON_EARTH_ELEMENTAL),
        # left-upper cavern (lines 93-101).
        (1,  2,  _MON_EARTH_ELEMENTAL),
        (1,  3,  _MON_EARTH_ELEMENTAL),
        (2,  2,  _MON_EARTH_ELEMENTAL),
        (5,  4,  _MON_EARTH_ELEMENTAL),
        # mid-upper-mid cavern (lines 103-109).
        (2,  21, _MON_EARTH_ELEMENTAL),
        (3,  21, _MON_EARTH_ELEMENTAL),
        (5,  21, _MON_EARTH_ELEMENTAL),
        (6,  22, _MON_EARTH_ELEMENTAL),
        (6,  23, _MON_EARTH_ELEMENTAL),
        # middle-left cavern (lines 111-121).
        (10, 13, _MON_EARTH_ELEMENTAL),
        (12, 14, _MON_EARTH_ELEMENTAL),
        (13, 15, _MON_EARTH_ELEMENTAL),
        (10, 18, _MON_EARTH_ELEMENTAL),
        (11, 18, _MON_EARTH_ELEMENTAL),
        # lower-left cavern (lines 123-127).
        (16, 3,  _MON_EARTH_ELEMENTAL),
        (17, 4,  _MON_EARTH_ELEMENTAL),
        (18, 4,  _MON_EARTH_ELEMENTAL),
    ])

    # Earth plane carries no special floor items; we leave items empty.
    items = _empty_placements()
    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Plane of Air — air.lua
# ---------------------------------------------------------------------------
# All air, no floor.  Vendor uses the 'A' character for "air" which is a
# special CLOUD/air tile.  We model air as VOID — without flight/levitation
# the player simply has no walkable tile, which is the intended hazard.
#
# We sprinkle a handful of cloud "platforms" (POOL on water has its own
# semantics; we treat clouds as VOID with a small cluster of FLOOR landing
# pads so the air-elemental coordinates can be valid spawn cells.)
#
# Citation: vendor/nethack/dat/air.lua lines 20-40.
# ---------------------------------------------------------------------------


def generate_air_plane(rng) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Plane of Air — almost-no-floor, flight/levitation required.

    The vendor map is uniformly 'A' (air); we encode this as VOID so the
    player falls without levitation.  A small landing patch in the lower-
    left provides the teleport_region arrival square (air.lua line 45).

    Reference: vendor/nethack/dat/air.lua

    Returns
    -------
    (terrain, monsters, items)
    """
    del rng
    # Wave 6 parity-fix: parse byte-identical air.lua MAP block.  All 'A' →
    # VOID (Air is non-walkable without flight/levitation per vendor).
    terrain = _parse_plane_map(_AIR_MAP)
    # Add tiny landing patches inside the vendor teleport regions so the
    # player can actually be placed somewhere when no flight is granted.
    # air.lua lines 45-46.
    grid = [list(row) for row in terrain.tolist()]
    for r in range(10, 13):
        for c in range(10, 14):
            grid[r][c] = _T_FLOOR
    for r in range(10, 13):
        for c in range(65, 70):
            grid[r][c] = _T_FLOOR
    terrain = jnp.asarray(grid, dtype=jnp.int8)

    # Air elementals — vendor places 11 of them (lines 51-61), plus
    # djinn, energy/steam vortices, etc.  Spread across the level.
    monsters_list: list[tuple[int, int, int]] = []
    for i in range(11):
        # Distribute across the level on the floor patches when possible.
        r = 10 + (i % 3)
        c = (10 if i % 2 == 0 else 65) + (i // 2)
        monsters_list.append((r, c, _MON_AIR_ELEMENTAL))
    # A handful of djinn / vortices (lines 86-107) — same elemental id
    # since downstream spawn just needs a distinct sentinel.
    for i in range(5):
        monsters_list.append((10 + (i % 3), 30 + i * 3, _MON_AIR_ELEMENTAL))
    monsters = _pack_placements(monsters_list)
    items = _empty_placements()
    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Plane of Fire — fire.lua
# ---------------------------------------------------------------------------
# Mostly LAVA with floor "islands".  Reproduce the vendor map directly.
#
# Citation: vendor/nethack/dat/fire.lua lines 16-38.
# ---------------------------------------------------------------------------

_FIRE_ROWS = [
    "LL.............LL..............L...LL.........LL.................LL...........L",
    "LL....LLLLLLLL............L...L.............LL....LLL.......................LL.",
    "L....LL...................L......................LLLL................LL........",
    ".....L.............LLLL...LL....LL...............LLLLL.............LLL.........",
    ".L.LLLL..............LL....L.....LLL..............LLLL..............LLLL......L",
    "LL..........LLLL...LLLL...LLL....LLL......L........LLLL....LL........LLL......L",
    "LL........LLLLLLL...LL.....L......L......LL.........LL......LL........LL...L...",
    "L.........LL..LLL..LL......LL......LLLL..L.........LL......LLL............LL...",
    "......L..LL....LLLLL.................LLLLLLL.......L......LL............LLLLLL.",
    "......L..L.....LL.LLLL.......L............L........LLLLL.LL......LL.........LL.",
    "......LL........L...LL......LL.............LLL.....L...LLL.......LLL.........L.",
    ".L.....LLLLLL........L.......LLL.............L....LL...L.LLL......LLLLLLL......",
    "LL..........LLLL............LL.L.............L....L...LL.........LLL..LLL......",
    ".L...........................LLLLL...........LL...L...L........LLLL..LLLLLL...L",
    ".L.....LLLL.............LL....LL.......LLL...LL.......L..LLL....LLLLLLL.......L",
    ".........LLL.........LLLLLLLLLLL......LLLLL...L...........LL...LL...LL.........",
    "...........LL.......LL.........LL.......LLL....L..LLL....LL.........LL.........",
    "............LLLLLLLLL...........LL....LLL.......LLLLL.....LL........LL.........",
    ".LL...............L.............LLLLLL............LL...LLLL.........LL.......L.",
    "LL.....L..........................LL....................LL..................LLL",
    "L.....LLL......................LLLLL.........L.........LLLLLLLL..............LL",
]


def _encode_fire(rows) -> jnp.ndarray:
    """Convert fire.lua rows to terrain (L=LAVA, .=FLOOR)."""
    grid = [[_T_VOID] * MAP_W for _ in range(MAP_H)]
    for r, row in enumerate(rows[:MAP_H]):
        for c, ch in enumerate(row[:MAP_W]):
            if ch == "L":
                grid[r][c] = _T_LAVA
            elif ch == ".":
                grid[r][c] = _T_FLOOR
    return jnp.asarray(grid, dtype=jnp.int8)


def generate_fire_plane(rng) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Plane of Fire — LAVA everywhere with FLOOR islands.

    Fire elementals + salamanders.  Vendor places 19 fire traps; we omit
    trap placement here (the LAVA tiles themselves are lethal).

    Reference: vendor/nethack/dat/fire.lua

    Returns
    -------
    (terrain, monsters, items)
    """
    del rng
    # Wave 6 parity-fix: parse byte-identical fire.lua MAP block.
    terrain = _parse_plane_map(_FIRE_MAP)

    # Place fire elementals & salamanders on the floor islands.
    # The vendor uses des.monster() without coords so they pick random
    # floor cells; we hand-pick a spread of valid floor coords from the map.
    monsters = _pack_placements([
        # Fire elementals (~10 of them).
        (0,  16, _MON_FIRE_ELEMENTAL),
        (1,  44, _MON_FIRE_ELEMENTAL),
        (3,  3,  _MON_FIRE_ELEMENTAL),
        (5,  10, _MON_FIRE_ELEMENTAL),
        (7,  0,  _MON_FIRE_ELEMENTAL),
        (9,  44, _MON_FIRE_ELEMENTAL),
        (11, 25, _MON_FIRE_ELEMENTAL),
        (13, 1,  _MON_FIRE_ELEMENTAL),
        (15, 0,  _MON_FIRE_ELEMENTAL),
        (18, 1,  _MON_FIRE_ELEMENTAL),
        # Salamanders.
        (2,  4,  _MON_SALAMANDER),
        (4,  0,  _MON_SALAMANDER),
        (6,  30, _MON_SALAMANDER),
        (8,  4,  _MON_SALAMANDER),
        (10, 6,  _MON_SALAMANDER),
        (12, 0,  _MON_SALAMANDER),
        (14, 0,  _MON_SALAMANDER),
        (16, 0,  _MON_SALAMANDER),
        (17, 0,  _MON_SALAMANDER),
        (19, 6,  _MON_SALAMANDER),
    ])
    items = _empty_placements()
    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Plane of Water — water.lua
# ---------------------------------------------------------------------------
# Vendor map is uniformly 'W' (deep water).  Bubbles (FLOOR islands) are
# generated procedurally by vendor mkmaze.c; we place a handful of bubbles
# directly so the player has somewhere to land.
#
# Citation: vendor/nethack/dat/water.lua lines 17-40.
# ---------------------------------------------------------------------------


def generate_water_plane(rng) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Plane of Water — POOL tiles everywhere with FLOOR bubbles.

    Krakens + electric/giant eels.  Vendor uses class-letter spawns
    (e.g. ``des.monster("kraken")``) without coords; we place at
    hand-picked pool coords.

    Reference: vendor/nethack/dat/water.lua

    Returns
    -------
    (terrain, monsters, items)
    """
    del rng
    # Wave 6 parity-fix: parse byte-identical water.lua MAP block.  Vendor
    # uses 'W' (WATER) everywhere; we stamp bubble (FLOOR) clusters on top.
    terrain = _parse_plane_map(_WATER_MAP)
    grid = [list(row) for row in terrain.tolist()]
    # Vendor uses WATER but historically we modeled with POOL for the wave-5
    # acid/poison semantics.  Re-skin from WATER to POOL to preserve
    # downstream subsystem behaviour while keeping the verbatim MAP-shape.
    for r in range(MAP_H):
        for c in range(MAP_W):
            if grid[r][c] == _T_WATER:
                grid[r][c] = _T_POOL

    # Three bubble clusters: arrival (left), middle, exit (right).
    # water.lua line 38: teleport_region {0,0,25,19} arrival.
    # water.lua line 39: levregion {51,0,75,19} exit portal.
    def stamp_bubble(cr: int, cc: int, r: int = 1) -> None:
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                rr, cc2 = cr + dr, cc + dc
                if 0 <= rr < MAP_H and 0 <= cc2 < MAP_W:
                    grid[rr][cc2] = _T_FLOOR

    stamp_bubble(10, 5,  1)   # arrival bubble (left third)
    stamp_bubble(8,  15, 1)   # mid-left bubble
    stamp_bubble(12, 38, 1)   # central bubble
    stamp_bubble(8,  55, 1)   # mid-right bubble
    stamp_bubble(10, 65, 1)   # exit bubble (right third)

    terrain = jnp.asarray(grid, dtype=jnp.int8)

    # Krakens + electric eels + giant eels (water.lua lines 41-60).
    monsters = _pack_placements([
        # Krakens (4) — main threat.
        (5,  20, _MON_KRAKEN),
        (15, 35, _MON_KRAKEN),
        (3,  50, _MON_KRAKEN),
        (17, 60, _MON_KRAKEN),
        # Water elementals — vendor places water elementals as class "E".
        (10, 30, _MON_WATER_ELEMENTAL),
        (8,  45, _MON_WATER_ELEMENTAL),
        (12, 25, _MON_WATER_ELEMENTAL),
        (4,  40, _MON_WATER_ELEMENTAL),
        # Giant eels (8).
        (2,  10, _MON_WATER_ELEMENTAL),
        (6,  18, _MON_WATER_ELEMENTAL),
        (9,  22, _MON_WATER_ELEMENTAL),
        (14, 28, _MON_WATER_ELEMENTAL),
        (16, 33, _MON_WATER_ELEMENTAL),
        (18, 42, _MON_WATER_ELEMENTAL),
        (11, 48, _MON_WATER_ELEMENTAL),
        (13, 52, _MON_WATER_ELEMENTAL),
    ])
    items = _empty_placements()
    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Plane of Astral — astral.lua
# ---------------------------------------------------------------------------
# Three temples each with a SANCTUM altar (one Lawful, one Neutral, one
# Chaotic).  This is the ascension level: stepping on the altar that
# matches your alignment while carrying the Amulet completes the game.
#
# Vendor altar coords (astral.lua lines 89-91):
#     des.altar({ x=07, y=09, align=align[1], type="sanctum" })   -- left
#     des.altar({ x=37, y=05, align=align[2], type="sanctum" })   -- center
#     des.altar({ x=67, y=09, align=align[3], type="sanctum" })   -- right
# vendor (x,y) → our (row, col) = (y, x).
# ---------------------------------------------------------------------------

# Astral altar positions (row, col) — fixed canonical layout.
ASTRAL_ALTAR_LAWFUL  = (9,  7)
ASTRAL_ALTAR_NEUTRAL = (5,  37)
ASTRAL_ALTAR_CHAOTIC = (9,  67)


def generate_astral_plane(rng) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Plane of Astral — open field with THREE altars (Lawful, Neutral, Chaotic).

    Aleax angels + High Priests guard each altar.  This is the final level
    of the game; ascension occurs when the player stands on the altar
    matching their alignment while carrying the Amulet of Yendor.

    The 3 altar tiles are marked with TileType.ALTAR.  The 3 altars' alignment
    is encoded in the *items* return array as triples
    (row, col, alignment_code), allowing the ascension subsystem to look up
    each altar's alignment without a separate per-tile overlay.

    Reference: vendor/nethack/dat/astral.lua

    Returns
    -------
    (terrain, monsters, items)
        terrain  : int8[MAP_H, MAP_W]; mostly FLOOR with WALL temple borders.
        monsters : int16[64, 3] — Aleax + high priests + angel guards.
        items    : int16[64, 3] — first 3 entries are
                   (altar_row, altar_col, align_code) for the three altars.
    """
    del rng
    # Wave 6 parity-fix: parse byte-identical astral.lua MAP block.  This
    # includes the three temple courts (West / Central / East) and the
    # connecting halls — verbatim from vendor/nethack/dat/astral.lua:13-34.
    terrain = _parse_plane_map(_ASTRAL_MAP)
    # Stamp the three altars (vendor astral.lua lines 89-91:
    #   des.altar({x=7,y=9}); des.altar({x=37,y=5}); des.altar({x=67,y=9}))
    terrain = terrain.at[ASTRAL_ALTAR_LAWFUL [0], ASTRAL_ALTAR_LAWFUL [1]].set(jnp.int8(_T_ALTAR))
    terrain = terrain.at[ASTRAL_ALTAR_NEUTRAL[0], ASTRAL_ALTAR_NEUTRAL[1]].set(jnp.int8(_T_ALTAR))
    terrain = terrain.at[ASTRAL_ALTAR_CHAOTIC[0], ASTRAL_ALTAR_CHAOTIC[1]].set(jnp.int8(_T_ALTAR))

    # Monsters — astral.lua lines 107-176.  High priests by each altar,
    # Aleax angels guarding, plus Riders (Death/Famine/Pestilence) at the
    # three "place" coords (lines 73-75).  We keep the count small but
    # cover one priest + angels per altar.
    lr, lc = ASTRAL_ALTAR_LAWFUL
    nr, nc = ASTRAL_ALTAR_NEUTRAL
    cr, cc = ASTRAL_ALTAR_CHAOTIC
    monsters = _pack_placements([
        # High priests on each altar.
        (lr, lc, _MON_HIGH_PRIEST),
        (nr, nc, _MON_HIGH_PRIEST),
        (cr, cc, _MON_HIGH_PRIEST),
        # Aleax angels — at least 3 per altar.
        (lr - 1, lc, _MON_ALEAX),
        (lr + 1, lc, _MON_ALEAX),
        (lr,     lc - 1, _MON_ALEAX),
        (nr - 1, nc, _MON_ALEAX),
        (nr + 1, nc, _MON_ALEAX),
        (nr,     nc - 1, _MON_ALEAX),
        (cr - 1, cc, _MON_ALEAX),
        (cr + 1, cc, _MON_ALEAX),
        (cr,     cc + 1, _MON_ALEAX),
        # Angels (vendor 'Angel' id).
        (lr,     lc + 1, _MON_ANGEL),
        (nr,     nc + 1, _MON_ANGEL),
        (cr,     cc - 1, _MON_ANGEL),
    ])

    # Items: store altar (row, col, align_code) triples in the first 3 slots.
    # This is used by check_ascension() to determine the alignment of the
    # altar the player is standing on.
    items = _pack_placements([
        (lr, lc, ASTRAL_ALIGN_LAWFUL),
        (nr, nc, ASTRAL_ALIGN_NEUTRAL),
        (cr, cc, ASTRAL_ALIGN_CHAOTIC),
    ])

    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Endgame branch level dispatch — the env's per-level generator will look
# up which plane to build by 1-based level index within ENDGAME.
#
#   Endgame Dlvl 1 = Earth Plane
#   Endgame Dlvl 2 = Air Plane
#   Endgame Dlvl 3 = Fire Plane
#   Endgame Dlvl 4 = Water Plane
#   Endgame Dlvl 5 = Astral Plane (ascension level)
# ---------------------------------------------------------------------------

def generate_endgame_level(rng, depth: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Dispatch to the correct plane factory by 1-based depth.

    Args:
        rng:   JAX PRNG key.
        depth: 1..5 — endgame level number.

    Returns:
        (terrain, monsters, items)
    """
    if depth == 1:
        return generate_earth_plane(rng)
    if depth == 2:
        return generate_air_plane(rng)
    if depth == 3:
        return generate_fire_plane(rng)
    if depth == 4:
        return generate_water_plane(rng)
    if depth == 5:
        return generate_astral_plane(rng)
    raise ValueError(f"Endgame depth must be 1..5, got {depth}")


# ---------------------------------------------------------------------------
# TODO (Wave 6):
#   - Wire boulders + fire traps as object/trap entities once the endgame
#     branch generator integrates with the item & trap layers.
#   - Replace VOID-as-air with a dedicated CLOUD/AIR TileType so the FOV
#     and movement subsystems can distinguish "fly-only" from "rock".
#   - Vendor astral.lua randomises altar alignment ordering per game
#     (align[1..3]); we currently fix the order as L/N/C.
# ---------------------------------------------------------------------------
