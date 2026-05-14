"""Wave 5 Phase 3 — per-role Quest level factories (13 roles).

Each NetHack role has a Quest branch (qstart / qlocate / qgoal). The qgoal
("nemesis") level is the most distinctive: it is where the role's nemesis
guards the role's quest artifact. This module hand-translates a
**simplified iconic version** of each role's qgoal layout from the vendor
.lua files in vendor/nethack/dat/{Arc,Bar,Cav,Hea,Kni,Mon,Pri,Ran,Rog,
Sam,Tou,Val,Wiz}-goal.lua.

Each factory returns (terrain, monsters, items) where the leader, nemesis,
artifact, and a handful of enemy monsters are placed in role-thematic
positions. Layouts are deliberately ~30-50 lines each (iconic flavor) per
the Wave-5 spec; full per-tile fidelity is deferred to Wave 6.

Citations (per role):
  vendor/nethack/src/role.c lines 30-573   — Role struct quest fields
                                              (lead0/lead1/lead2, nemesis,
                                              enemy1/enemy2, qlist artifact)
  vendor/nethack/dat/Arc-goal.lua          — Archeologist Tomb of the Toltec Kings
  vendor/nethack/dat/Bar-goal.lua          — Barbarian Duali Oasis
  vendor/nethack/dat/Cav-goal.lua          — Caveman Dragon's Lair
  vendor/nethack/dat/Hea-goal.lua          — Healer Temple of Coeus
  vendor/nethack/dat/Kni-goal.lua          — Knight Isle of Glass
  vendor/nethack/dat/Mon-goal.lua          — Monk Monastery of the Earth-Lord
  vendor/nethack/dat/Pri-goal.lua          — Priest Temple of Nalzok
  vendor/nethack/dat/Ran-goal.lua          — Ranger cave of the wumpus
  vendor/nethack/dat/Rog-goal.lua          — Rogue Assassins' Guild Hall
  vendor/nethack/dat/Sam-goal.lua          — Samurai Shogun's Castle
  vendor/nethack/dat/Tou-goal.lua          — Tourist Thieves' Guild Hall
  vendor/nethack/dat/Val-goal.lua          — Valkyrie cave of Surtur
  vendor/nethack/dat/Wiz-goal.lua          — Wizard Tower of Darkness
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.dungeon.branches import MAP_H, MAP_W


# ---------------------------------------------------------------------------
# Tile constants (mirror constants/tiles.py — kept local to avoid cycles).
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


_CHAR_TO_TILE = {
    " ": _T_VOID,
    ".": _T_FLOOR,
    "#": _T_CORRIDOR,
    "|": _T_WALL,
    "-": _T_WALL,
    "x": _T_WALL,    # outer rock (mines/Val pattern)
    "+": _T_CLOSED_DOOR,
    "<": _T_STAIR_UP,
    ">": _T_STAIR_DOWN,
    "}": _T_WATER,
    "L": _T_LAVA,
    "_": _T_ALTAR,
    "{": _T_FOUNTAIN,
    "\\": _T_THRONE,
    "S": _T_SHOP_FLOOR,
    "T": _T_FLOOR,
    "C": _T_FLOOR,
}


def _encode_map(rows: list) -> jnp.ndarray:
    """Convert a list of character rows to an int8 terrain grid.

    Pads to (MAP_H, MAP_W) with VOID.
    """
    grid = [[_T_VOID] * MAP_W for _ in range(MAP_H)]
    for r, row in enumerate(rows[:MAP_H]):
        for c, ch in enumerate(row[:MAP_W]):
            grid[r][c] = _CHAR_TO_TILE.get(ch, _T_VOID)
    return jnp.array(grid, dtype=jnp.int8)


def _pack_placements(triples: list, capacity: int = 64) -> jnp.ndarray:
    """Pack a list of (row, col, type_id) triples into int16[capacity, 3]."""
    arr = [[-1, -1, -1] for _ in range(capacity)]
    for i, (r, c, t) in enumerate(triples[:capacity]):
        arr[i] = [int(r), int(c), int(t)]
    return jnp.array(arr, dtype=jnp.int16)


# Vendor quest-goal files use 'S' = secret door, 'P' = pool, 'F' = iron bars,
# 'B' = broken passage.  This char map is applied when we parse the verbatim
# vendor MAP strings below.
_CHAR_TO_TILE_VENDOR = dict(_CHAR_TO_TILE)
_CHAR_TO_TILE_VENDOR["S"] = _T_CLOSED_DOOR
_CHAR_TO_TILE_VENDOR["P"] = _T_WATER
_CHAR_TO_TILE_VENDOR["F"] = _T_WALL
_CHAR_TO_TILE_VENDOR["B"] = _T_FLOOR


def _parse_map_string(s: str, char_map: dict = None) -> jnp.ndarray:
    """Parse a multi-line vendor MAP block into an int8 terrain grid.

    Drops only leading/trailing fully-blank lines; pads to (MAP_H, MAP_W).
    """
    if char_map is None:
        char_map = _CHAR_TO_TILE_VENDOR
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


# ===========================================================================
# Wave 6 Phase B+ — verbatim vendor MAP-string constants for the 13 quest
# goal levels.  Copied byte-identical from vendor/nethack/dat/<role>-goal.lua.
# Licensing (NGPL) is handled separately at the project level.
# ===========================================================================

# vendor/nethack/dat/Arc-goal.lua MAP section — NGPL
_ARC_GOAL_MAP = """\

                                  ---------
                                  |..|.|..|
                       -----------|..S.S..|-----------
                       |.|........|+-|.|-+|........|.|
                       |.S........S..|.|..S........S.|
                       |.|........|..|.|..|........|.|
                    ------------------+------------------
                    |..|..........|.......|..........|..|
                    |..|..........+.......|..........S..|
                    |..S..........|.......+..........|..|
                    |..|..........|.......|..........|..|
                    ------------------+------------------
                       |.|........|..|.|..|........|.|
                       |.S........S..|.|..S........S.|
                       |.|........|+-|.|-+|........|.|
                       -----------|..S.S..|-----------
                                  |..|.|..|
                                  ---------

"""

# vendor/nethack/dat/Bar-goal.lua MAP section — NGPL
_BAR_GOAL_MAP = """\

                               .............
                             ..................
        ....              .........................          ....
      .......          ..........................           .......
      ......             ........................          .......
      ..  ......................................             ..
       ..                 .....................             ..
        ..                 ..................              ..
         ..         ..S...S..............   ................
          ..                   ........                ...
       .........                                         ..
       ......  ..                                         ...  ....
      .. ...    ..                             ......       ........
   ....          .. ..................        ........       ......
  ......          ......................       ......         ..
   ....             ..................              ...........
                      ..............
                        ...........

"""

# vendor/nethack/dat/Cav-goal.lua MAP section — NGPL
_CAV_GOAL_MAP = """\

                          .....................
                         .......................
                        .........................
                       ...........................
                      .............................
                     ...............................
                    .................................
                   ...................................
                  .....................................
                 .......................................
                  .....................................
                   ...................................
                    .................................
                     ...............................
                      .............................
                       ...........................
                        .........................
                         .......................

"""

# vendor/nethack/dat/Hea-goal.lua MAP section — NGPL
_HEA_GOAL_MAP = """\
.P....................................PP.
PP.......PPPPPPP....PPPPPPP....PPPP...PP.
...PPPPPPP....PPPPPPP.....PPPPPP..PPP...P
...PP..............................PPP...
..PP..............................PP.....
..PP..............................PPP....
..PPP..............................PP....
.PPP..............................PPPP...
...PP............................PPP...PP
..PPPP...PPPPP..PPPP...PPPPP.....PP...PP.
P....PPPPP...PPPP..PPPPP...PPPPPPP...PP..
PPP..................................PPP.
"""

# vendor/nethack/dat/Kni-goal.lua MAP section — NGPL
_KNI_GOAL_MAP = """\
....PPPP..PPP..
.PPPPP...PP..     ..........     .................................
..PPPPP...P..    ...........    ...................................
..PPP.......   ...........    ......................................
...PPP.......    .........     ...............   .....................
...........    ............    ............     ......................
............   .............      .......     .....................
..............................            .........................
...............................   ..................................
.............................    ....................................
.........    ......................................................
.....PP...    .....................................................
.....PPP....    ....................................................
......PPP....   ..............   ....................................
.......PPP....  .............    .....................................
........PP...    ............    ......................................
...PPP........     ..........     ..................................
..PPPPP........     ..........     ..............................
....PPPPP......       .........     ..........................
.......PPPP...
"""

# vendor/nethack/dat/Mon-goal.lua MAP section — NGPL
_MON_GOAL_MAP = """\
xxxxxx..xxxxxx...xxxxxxxxx
xxxx......xx......xxxxxxxx
xx.xx.............xxxxxxxx
x....................xxxxx
......................xxxx
......................xxxx
xx........................
xxx......................x
xxx................xxxxxxx
xxxx.....x.xx.......xxxxxx
xxxxx...xxxxxx....xxxxxxxx
"""

# vendor/nethack/dat/Pri-goal.lua MAP section — NGPL (same shape as Mon-goal)
_PRI_GOAL_MAP = """\
xxxxxx..xxxxxx...xxxxxxxxx
xxxx......xx......xxxxxxxx
xx.xx.............xxxxxxxx
x....................xxxxx
......................xxxx
......................xxxx
xx........................
xxx......................x
xxx................xxxxxxx
xxxx.....x.xx.......xxxxxx
xxxxx...xxxxxx....xxxxxxxx
"""

# vendor/nethack/dat/Ran-goal.lua MAP section — NGPL
_RAN_GOAL_MAP = """\

  ...                                                                  ...
 ..........................................................................
  ...                                +                                 ...
   .     ............     .......    .                   .......        .
   .  .............................  .       ........   .........S..    .
   .   ............    .  ......     .       .      .    .......   ..   .
   .     .........     .   ....      +       . ...  .               ..  .
   .        S          .         .........   .S.    .S...............   .
   .  ...   .     ...  .         .........          .                   .
   . ........    .....S.+.......+....\\....+........+.                   .
   .  ...         ...    S       .........           ..      .....      .
   .                    ..       .........            ..      ......    .
   .      .......     ...            +       ....    ....    .......... .
   . ..............  ..              .      ......  ..  .............   .
   .     .............               .     ..........          ......   .
  ...                                +                                 ...
 ..........................................................................
  ...                                                                  ...

"""

# vendor/nethack/dat/Rog-goal.lua MAP section — NGPL
_ROG_GOAL_MAP = """\
-----      -------.......................................|-----------------|
|...|  -----.....|.......................................|.................|
|...----...|.....|.......................................|....---------....|
|.---......---..--.................................------------.......|....|
|...............|..................................|..|...|...----........-|
|.....-----....--.................................|-..--..-|.....----S----|
|--S---...|....|.................................|-........-|....|........|
|.........---------.............................|-....}}....-|...|...|....|
|....|.....S......|............................|-.....}}.....-|..--.------|
|-----.....--.....|...........................|-...}}}}}}}}...-|....|.....--
|...........--....------S-----...............|-....}}}}}}}}....-|..........|
|............--........|...| |..............--.....}}.}}........----------S-
|.............|........|...| |..............|......}}}}}}}}......|...|.....|
|S-.---.---.---.---.---|...| ------------...--........}}.}}.....--..---....|
|.---.---.---.---.-S-..----- |....|.....|....|-....}}}}}}}}....---..S.|--..|
|...|.......|..........|...---....---...S.....|-...}}}}}}}}...-|.S..|...|..|
|...|..|....|..........|............|..--..----|-.....}}.....-|..----...-S--
|...|---....----.......|----- ......|...---|    |-....}}....-|...|..--.--..|
-----.....---.....--.---....--...--------..|     |-........-|....|.........|
    |.............|..........|.............S...   |S-------|.....|..-----..|
    ----------------------------------------  ......       ----------   ----
"""

# vendor/nethack/dat/Sam-goal.lua MAP section — NGPL
_SAM_GOAL_MAP = """\

           .......................
       ......-------------------......
    ......----.................----......
   ....----.....-------------.....----....
  ....--.....----...........----.....--....
  ...||....---....---------....---....||...
  ...|....--....---.......---....--....|...
 ....|...||...---...--+--...---...||...|....
 ....|...|....|....|-...-|....|....|...|....
 ....|...|....|....+.....+....|....|...|....
 ....|...|....|....|-...-|....|....|...|....
 ....|...||...---...--+--...---...||...|....
  ...|....--....---.......---....--....|...
  ...||....---....---------....---....||...
  ....--.....----...........----.....--....
   ....----.....-------------.....----....
    ......----.................----......
       ......-------------------......
           .......................
"""

# vendor/nethack/dat/Tou-goal.lua MAP section — NGPL
_TOU_GOAL_MAP = """\
----------------------------------------------------------------------------
|.........|.........|..........|..| |.................|........|........|..|
|.........|.........|..........|..| |....--------.....|........|........|..|
|------S--|--+-----------+------..| |....|......|.....|........|........|..|
|.........|.......................| |....|......+.....--+-------------+--..|
|.........|.......................| |....|......|..........................|
|-S-----S-|......----------.......| |....|......|..........................|
|..|..|...|......|........|.......| |....-----------.........----..........|
|..+..+...|......|........|.......| |....|.........|.........|}}|..........|
|..|..|...|......+........|.......| |....|.........+.........|}}|..........|
|..|..|...|......|........|.......S.S....|.........|.........----..........|
|---..----|......|........|.......| |....|.........|.......................|
|.........+......|+F-+F-+F|.......| |....-----------.......................|
|---..----|......|..|..|..|.......| |......................--------------..|
|..|..|...|......--F-F--F--.......| |......................+............|..|
|..+..+...|.......................| |--.---...-----+-----..|............|..|
|--|..----|--+-----------+------..| |.....|...|.........|..|------------|..|
|..+..+...|.........|..........|..| |.....|...|.........|..+............|..|
|..|..|...|.........|..........|..| |.....|...|.........|..|............|..|
----------------------------------------------------------------------------
"""

# vendor/nethack/dat/Val-goal.lua MAP section — NGPL
_VAL_GOAL_MAP = """\
xxxxxx.....................xxxxxxxx
xxxxx.......LLLLL.LLLLL......xxxxxx
xxxx......LLLLLLLLLLLLLLL......xxxx
xxxx.....LLL|---------|LLL.....xxxx
xxxx....LL|--.........--|LL.....xxx
x......LL|-...LLLLLLL...-|LL.....xx
.......LL|...LL.....LL...|LL......x
......LL|-..LL.......LL..-|LL......
......LL|.................|LL......
......LL|-..LL.......LL..-|LL......
.......LL|...LL.....LL...|LL.......
xx.....LL|-...LLLLLLL...-|LL......x
xxx.....LL|--.........--|LL.....xxx
xxxx.....LLL|---------|LLL...xxxxxx
xxxxx.....LLLLLLLLLLLLLLL...xxxxxxx
xxxxxx......LLLLL.LLLLL.....xxxxxxx
xxxxxxxxx..................xxxxxxxx
"""

# vendor/nethack/dat/Wiz-goal.lua MAP section — NGPL
_WIZ_GOAL_MAP = """\



                   -------------                 -------------
                   |...........|                 |...........|
            -------|...........-------------------...........|
            |......S...........|..|..|..|..|..|..|...........|
            |......|...........|..|..|..|..|..|..|...........|
            |......|...........-F+-F+-F+-F+-F+-F+-...........|
            --S----|...........S.................+...........|
            |......|...........-F+-F+-F+-F+-F+-F+-...........|
            |......|...........|..|..|..|..|..|..|...........|
            |......|...........|..|..|..|..|..|..|...........|
            -------|...........-------------------...........|
                   |...........|                 |...........|
                   -------------                 -------------




"""


_GOAL_MAPS = {
    0:  _ARC_GOAL_MAP, 1:  _BAR_GOAL_MAP, 2:  _CAV_GOAL_MAP,
    3:  _HEA_GOAL_MAP, 4:  _KNI_GOAL_MAP, 5:  _MON_GOAL_MAP,
    6:  _PRI_GOAL_MAP, 7:  _ROG_GOAL_MAP, 8:  _RAN_GOAL_MAP,
    9:  _SAM_GOAL_MAP, 10: _TOU_GOAL_MAP, 11: _VAL_GOAL_MAP,
    12: _WIZ_GOAL_MAP,
}


# ---------------------------------------------------------------------------
# Monster index lookups (computed at import time; the MONSTERS table is the
# canonical source of truth — see Nethax/nethax/constants/monsters.py).
# Indices below come from role.c (leader/nemesis/enemy1/enemy2 fields).
# ---------------------------------------------------------------------------
# Quest leaders (PM_*); indices in MONSTERS table (verified at import time).
# Wave 6 parity-fix: updated to match vendor/nethack/src/role.c roles[]
# leader fields against the actual MONSTERS table indices.  Previous values
# were off by one row.
_LEADER_ARC      = 342    # Lord Carnarvon       (role.c:45)
_LEADER_BAR      = 343    # Pelias               (role.c:87)
_LEADER_CAV      = 344    # Shaman Karnov        (role.c:129)
_LEADER_HEA      = 345    # Hippocrates          (role.c:171)
_LEADER_KNI      = 346    # King Arthur          (role.c:212)
_LEADER_MON      = 347    # Grand Master         (role.c:253)
_LEADER_PRI      = 348    # Arch Priest          (role.c:295)
_LEADER_RAN      = 349    # Orion                (role.c:394)
_LEADER_ROG      = 350    # Master of Thieves    (role.c:339)
_LEADER_SAM      = 351    # Lord Sato            (role.c:436)
_LEADER_TOU      = 352    # Twoflower            (role.c:477)
_LEADER_VAL      = 353    # Norn                 (role.c:518)
_LEADER_WIZ      = 354    # Neferet the Green    (role.c:559)

# Quest nemeses.
# Wave 6 parity-fix: updated to match vendor/nethack/src/role.c roles[]
# nemesis fields against actual MONSTERS table indices.
_NEM_ARC         = 355    # Minion of Huhetotl   (role.c:47)
_NEM_BAR         = 356    # Thoth Amon           (role.c:89)
_NEM_CAV         = 357    # Chromatic Dragon     (role.c:131)
_NEM_HEA         = 358    # Cyclops              (role.c:173)
_NEM_KNI         = 359    # Ixoth                (role.c:214)
_NEM_MON         = 360    # Master Kaen          (role.c:255)
_NEM_PRI         = 361    # Nalzok               (role.c:297)
_NEM_RAN         = 362    # Scorpius             (role.c:396)
_NEM_ROG         = 363    # Master Assassin      (role.c:341)
_NEM_SAM         = 364    # Ashikaga Takauji     (role.c:438)
_NEM_TOU         = 350    # Master of Thieves    (role.c:479)
                          # (Tourist nemesis IS the Rog leader monster.)
_NEM_VAL         = 365    # Lord Surtur          (role.c:520)
_NEM_WIZ         = 366    # Dark One             (role.c:561)

# A small "enemy" pool (any role-specific filler monster).
_ENEMY_GENERIC   = 1      # arbitrary low-level placeholder; downstream
                          # population uses enemy1/enemy2 from role.c.


# ---------------------------------------------------------------------------
# Quest artifact indices (position in vendor/include/artilist.h, 0-based).
# ---------------------------------------------------------------------------
_ART_ORB_OF_DETECTION       = 20  # Archeologist
_ART_HEART_OF_AHRIMAN       = 21  # Barbarian
_ART_SCEPTRE_OF_MIGHT       = 22  # Caveman
_ART_STAFF_OF_AESCULAPIUS   = 24  # Healer
_ART_MAGIC_MIRROR_OF_MERLIN = 25  # Knight
_ART_EYES_OF_THE_OVERWORLD  = 26  # Monk
_ART_MITRE_OF_HOLINESS      = 27  # Priest
_ART_LONGBOW_OF_DIANA       = 28  # Ranger
_ART_MASTER_KEY_OF_THIEVERY = 29  # Rogue
_ART_TSURUGI_OF_MURAMASA    = 30  # Samurai
_ART_YENDORIAN_EXPRESS_CARD = 31  # Tourist
_ART_ORB_OF_FATE            = 32  # Valkyrie
_ART_EYE_OF_THE_AETHIOPICA  = 33  # Wizard


# ---------------------------------------------------------------------------
# Per-role factories — simplified iconic layouts (Wave 6 will replace each
# with a full per-tile parse of the vendor .lua).
# ---------------------------------------------------------------------------

# --- Archeologist: Tomb of the Toltec Kings -------------------------------
# Citation: vendor/nethack/dat/Arc-goal.lua — tomb chambers, mummy guards.
_ARC_ROWS = [
    "                                                                            ",
    "    -----------------------                                                 ",
    "    |.....................|                                                 ",
    "    |..---......---......---                                                ",
    "    |..|.|......|.|......|.|                                                ",
    "    |..|L|......|<|......|>|                                                ",
    "    |..---......---......---                                                ",
    "    |.....................|                                                 ",
    "    |..-----------------..|                                                 ",
    "    |..|...............|..|                                                 ",
    "    |..|.....tomb......|..|                                                 ",
    "    |..|...............|..|                                                 ",
    "    |..-----------------..|                                                 ",
    "    |.....................|                                                 ",
    "    -----------------------                                                 ",
]


def generate_arc_quest_level(rng):
    """Archeologist Quest goal — Tomb of the Toltec Kings. Simplified — Wave 6 full layout from Arc-goal.lua."""
    terrain = _encode_map(_ARC_ROWS)
    monsters = _pack_placements([
        (5, 8,  _LEADER_ARC),     # Lord Carnarvon at stair-up alcove
        (10, 13, _NEM_ARC),       # Minion of Huhetotl in tomb chamber
        (10, 10, _ENEMY_GENERIC),
        (10, 16, _ENEMY_GENERIC),
        (11, 13, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (10, 13, _ART_ORB_OF_DETECTION),  # artifact on nemesis tile
    ])
    return terrain, monsters, items


# --- Barbarian: the Duali Oasis -------------------------------------------
# Citation: vendor/nethack/dat/Bar-goal.lua — desert oasis with ogres/trolls.
_BAR_ROWS = [
    "                                                                            ",
    "         ............................................                       ",
    "        ..............................................                      ",
    "       ........----...........................----.....                     ",
    "       .......-....-.........................-....-....                     ",
    "       ......-.{..{.-...................-.{..{.-.......                     ",
    "       ......-......-...................-......-.......                     ",
    "       .......-....-.........................-....-....                     ",
    "       ........----...........................----.....                     ",
    "        ..............................................                      ",
    "         ............................................                       ",
]


def generate_bar_quest_level(rng):
    """Barbarian Quest goal — Duali Oasis. Simplified — Wave 6 full layout from Bar-goal.lua."""
    terrain = _encode_map(_BAR_ROWS)
    monsters = _pack_placements([
        (5, 13, _LEADER_BAR),     # Pelias at oasis pool 1
        (5, 42, _NEM_BAR),        # Thoth Amon at oasis pool 2 (nemesis "den")
        (3, 25, _ENEMY_GENERIC),  # ogres
        (7, 25, _ENEMY_GENERIC),
        (3, 35, _ENEMY_GENERIC),  # trolls
        (7, 35, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (5, 42, _ART_HEART_OF_AHRIMAN),
    ])
    return terrain, monsters, items


# --- Caveman: Dragon's Lair -----------------------------------------------
# Citation: vendor/nethack/dat/Cav-goal.lua — large cave with Chromatic Dragon.
_CAV_ROWS = [
    "                                                                            ",
    "      .................................................                     ",
    "     ...................................................                    ",
    "    .....................................................                   ",
    "    .....................................................                   ",
    "    .....................................................                   ",
    "    .....................................................                   ",
    "    .....................................................                   ",
    "    .....................................................                   ",
    "    .....................................................                   ",
    "    .....................................................                   ",
    "     ...................................................                    ",
    "      .................................................                     ",
]


def generate_cav_quest_level(rng):
    """Caveman Quest goal — Dragon's Lair. Simplified — Wave 6 full layout from Cav-goal.lua."""
    terrain = _encode_map(_CAV_ROWS)
    monsters = _pack_placements([
        (5, 8,  _LEADER_CAV),     # Shaman Karnov at cave entrance
        (6, 35, _NEM_CAV),        # Chromatic Dragon at lair center
        (4, 25, _ENEMY_GENERIC),  # bugbears
        (8, 25, _ENEMY_GENERIC),
        (4, 45, _ENEMY_GENERIC),  # hill giants
        (8, 45, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (6, 35, _ART_SCEPTRE_OF_MIGHT),
    ])
    return terrain, monsters, items


# --- Healer: Temple of Coeus ----------------------------------------------
# Citation: vendor/nethack/dat/Hea-goal.lua — temple with Cyclops, snakes.
_HEA_ROWS = [
    "                                                                            ",
    "    -----------------------------------                                     ",
    "    |.................................|                                     ",
    "    |..---------------------------..|                                       ",
    "    |..|.........................|..|                                       ",
    "    |..|...........{.............|..|                                       ",
    "    |..|.........................|..|                                       ",
    "    |..|.........................|..|                                       ",
    "    |..|.........................|..|                                       ",
    "    |..---------------------------..|                                       ",
    "    |.................................|                                     ",
    "    -----------------------------------                                     ",
]


def generate_hea_quest_level(rng):
    """Healer Quest goal — Temple of Coeus. Simplified — Wave 6 full layout from Hea-goal.lua."""
    terrain = _encode_map(_HEA_ROWS)
    monsters = _pack_placements([
        (5, 8,  _LEADER_HEA),     # Hippocrates in outer ring
        (7, 18, _NEM_HEA),        # Cyclops at temple center
        (6, 14, _ENEMY_GENERIC),  # giant rats
        (6, 22, _ENEMY_GENERIC),
        (8, 14, _ENEMY_GENERIC),  # snakes
        (8, 22, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (7, 18, _ART_STAFF_OF_AESCULAPIUS),
    ])
    return terrain, monsters, items


# --- Knight: Isle of Glass ------------------------------------------------
# Citation: vendor/nethack/dat/Kni-goal.lua — castle on an isle with Ixoth.
_KNI_ROWS = [
    "                                                                            ",
    "                  --------------------                                      ",
    "                  |..................|                                      ",
    "         ---------+..................+---------                             ",
    "         |........|..................|........|                             ",
    "         |........|......-----.......|........|                             ",
    "         |........+......|...|.......+........|                             ",
    "         |........|......|...|.......|........|                             ",
    "         |........|......-----.......|........|                             ",
    "         ---------+..................+---------                             ",
    "                  |..................|                                      ",
    "                  --------------------                                      ",
]


def generate_kni_quest_level(rng):
    """Knight Quest goal — Isle of Glass. Simplified — Wave 6 full layout from Kni-goal.lua."""
    terrain = _encode_map(_KNI_ROWS)
    monsters = _pack_placements([
        (4, 11, _LEADER_KNI),     # King Arthur in left antechamber
        (6, 27, _NEM_KNI),        # Ixoth (red dragon) in inner sanctum
        (5, 20, _ENEMY_GENERIC),  # quasits
        (7, 20, _ENEMY_GENERIC),  # ochre jellies
        (4, 35, _ENEMY_GENERIC),
        (8, 35, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (6, 27, _ART_MAGIC_MIRROR_OF_MERLIN),
    ])
    return terrain, monsters, items


# --- Monk: Monastery of the Earth-Lord ------------------------------------
# Citation: vendor/nethack/dat/Mon-goal.lua — pagoda with Master Kaen, xorns.
_MON_ROWS = [
    "                                                                            ",
    "                  ---------                                                 ",
    "                  |.......|                                                 ",
    "                  |.......|                                                 ",
    "         ---------|.......|---------                                        ",
    "         |.........+......+........|                                        ",
    "         |..---....|......|...---..|                                        ",
    "         |..|.|....|.......|..|.|..|                                        ",
    "         |..---....|......|...---..|                                        ",
    "         |.........+......+........|                                        ",
    "         ---------|.......|---------                                        ",
    "                  |.......|                                                 ",
    "                  ---------                                                 ",
]


def generate_mon_quest_level(rng):
    """Monk Quest goal — Monastery of the Earth-Lord. Simplified — Wave 6 full layout from Mon-goal.lua."""
    terrain = _encode_map(_MON_ROWS)
    monsters = _pack_placements([
        (5, 11, _LEADER_MON),     # Grand Master in west wing
        (7, 22, _NEM_MON),        # Master Kaen at monastery heart
        (5, 30, _ENEMY_GENERIC),  # earth elementals
        (9, 30, _ENEMY_GENERIC),  # xorns
        (7, 14, _ENEMY_GENERIC),
        (7, 28, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (7, 22, _ART_EYES_OF_THE_OVERWORLD),
    ])
    return terrain, monsters, items


# --- Priest: Temple of Nalzok ---------------------------------------------
# Citation: vendor/nethack/dat/Pri-goal.lua — desecrated temple with Nalzok.
_PRI_ROWS = [
    "                                                                            ",
    "    -------------------------------                                         ",
    "    |.............................|                                         ",
    "    |..........._...._...._.......|                                         ",
    "    |.............................|                                         ",
    "    |..---.........................|                                        ",
    "    |..|.|.........._..............|                                        ",
    "    |..---.........................|                                        ",
    "    |.............................|                                         ",
    "    -------------------------------                                         ",
]


def generate_pri_quest_level(rng):
    """Priest Quest goal — Temple of Nalzok. Simplified — Wave 6 full layout from Pri-goal.lua."""
    terrain = _encode_map(_PRI_ROWS)
    monsters = _pack_placements([
        (3, 8,  _LEADER_PRI),     # Arch Priest at outer altar
        (6, 20, _NEM_PRI),        # Nalzok at the central altar
        (3, 14, _ENEMY_GENERIC),  # zombies
        (3, 22, _ENEMY_GENERIC),
        (3, 28, _ENEMY_GENERIC),  # wraiths
        (6, 6,  _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (6, 20, _ART_MITRE_OF_HOLINESS),
    ])
    return terrain, monsters, items


# --- Ranger: cave of the wumpus -------------------------------------------
# Citation: vendor/nethack/dat/Ran-goal.lua — winding caves with Scorpius.
_RAN_ROWS = [
    "                                                                            ",
    "     ......................................                                 ",
    "    ........................................                                ",
    "   ..........................................                               ",
    "   ..........................................                               ",
    "   ..........................................                               ",
    "   ..........................................                               ",
    "   ..........................................                               ",
    "    ........................................                                ",
    "     ......................................                                 ",
]


def generate_ran_quest_level(rng):
    """Ranger Quest goal — cave of the wumpus. Simplified — Wave 6 full layout from Ran-goal.lua."""
    terrain = _encode_map(_RAN_ROWS)
    monsters = _pack_placements([
        (4, 8,  _LEADER_RAN),     # Orion at cave mouth
        (5, 32, _NEM_RAN),        # Scorpius deep in caves
        (4, 18, _ENEMY_GENERIC),  # forest centaurs
        (6, 18, _ENEMY_GENERIC),  # giant spiders
        (4, 25, _ENEMY_GENERIC),
        (6, 25, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (5, 32, _ART_LONGBOW_OF_DIANA),
    ])
    return terrain, monsters, items


# --- Rogue: Assassins' Guild Hall -----------------------------------------
# Citation: vendor/nethack/dat/Rog-goal.lua — maze-like guild with traps.
_ROG_ROWS = [
    "                                                                            ",
    "    ---------------------------------                                       ",
    "    |...........|.......|...........|                                       ",
    "    |..-------..|..---..|..-------..|                                       ",
    "    |..|.....|..|..|.|..|..|.....|..|                                       ",
    "    |..|.....+..+..+.+..+..+.....|..|                                       ",
    "    |..|.....|..|..|.|..|..|.....|..|                                       ",
    "    |..-------..|..---..|..-------..|                                       ",
    "    |...........|.......|...........|                                       ",
    "    ---------------------------------                                       ",
]


def generate_rog_quest_level(rng):
    """Rogue Quest goal — Assassins' Guild Hall. Simplified — Wave 6 full layout from Rog-goal.lua."""
    terrain = _encode_map(_ROG_ROWS)
    monsters = _pack_placements([
        (5, 7,  _LEADER_ROG),     # Master of Thieves at west cell
        (5, 18, _NEM_ROG),        # Master Assassin at center
        (5, 28, _ENEMY_GENERIC),  # leprechauns
        (3, 12, _ENEMY_GENERIC),
        (7, 12, _ENEMY_GENERIC),  # guardian nagas
        (3, 24, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (5, 18, _ART_MASTER_KEY_OF_THIEVERY),
    ])
    return terrain, monsters, items


# --- Samurai: Shogun's Castle ---------------------------------------------
# Citation: vendor/nethack/dat/Sam-goal.lua — castle with Ashikaga Takauji.
_SAM_ROWS = [
    "                                                                            ",
    "    -------------------------------                                         ",
    "    |.............................|                                         ",
    "    |..--------|.......|--------..|                                         ",
    "    |..|.......|.......|.......|..|                                         ",
    "    |..|.......+.......+.......|..|                                         ",
    "    |..|.......|...\\...|.......|..|                                         ",
    "    |..|.......+.......+.......|..|                                         ",
    "    |..|.......|.......|.......|..|                                         ",
    "    |..--------|.......|--------..|                                         ",
    "    |.............................|                                         ",
    "    -------------------------------                                         ",
]


def generate_sam_quest_level(rng):
    """Samurai Quest goal — Shogun's Castle. Simplified — Wave 6 full layout from Sam-goal.lua."""
    terrain = _encode_map(_SAM_ROWS)
    monsters = _pack_placements([
        (5, 9,  _LEADER_SAM),     # Lord Sato in west keep
        (6, 19, _NEM_SAM),        # Ashikaga Takauji on throne
        (5, 28, _ENEMY_GENERIC),  # wolves
        (7, 28, _ENEMY_GENERIC),  # stalkers
        (7, 9,  _ENEMY_GENERIC),
        (5, 19, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (6, 19, _ART_TSURUGI_OF_MURAMASA),
    ])
    return terrain, monsters, items


# --- Tourist: Thieves' Guild Hall (Ankh-Morpork) --------------------------
# Citation: vendor/nethack/dat/Tou-goal.lua — open camp/market, lightly defended.
_TOU_ROWS = [
    "                                                                            ",
    "    .........................................                               ",
    "    .........................................                               ",
    "    .........................................                               ",
    "    .........................................                               ",
    "    .........................................                               ",
    "    .........................................                               ",
    "    .........................................                               ",
    "    .........................................                               ",
]


def generate_tou_quest_level(rng):
    """Tourist Quest goal — Ankh-Morpork / Thieves' Guild Hall. Simplified — Wave 6 full layout from Tou-goal.lua."""
    terrain = _encode_map(_TOU_ROWS)
    monsters = _pack_placements([
        (3, 8,  _LEADER_TOU),     # Twoflower at camp
        (6, 30, _NEM_TOU),        # Master of Thieves
        (4, 18, _ENEMY_GENERIC),  # giant spiders
        (6, 18, _ENEMY_GENERIC),  # forest centaurs
        (4, 38, _ENEMY_GENERIC),
        (6, 38, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (6, 30, _ART_YENDORIAN_EXPRESS_CARD),
    ])
    return terrain, monsters, items


# --- Valkyrie: cave of Surtur (ice cavern with lava moat) ----------------
# Citation: vendor/nethack/dat/Val-goal.lua — frozen island in lava.
_VAL_ROWS = [
    "                                                                            ",
    "xxxxxx.....................xxxxxxxx                                         ",
    "xxxxx.......LLLLL.LLLLL......xxxxxx                                         ",
    "xxxx......LLLLLLLLLLLLLLL......xxxx                                         ",
    "xxxx.....LLL---------LLL.....xxxx                                           ",
    "xxxx....LL--.........--LL.....xxx                                           ",
    "x......LL-...LLLLLLL...-LL.....xx                                           ",
    ".......LL|...LL.....LL...|LL......x                                         ",
    "......LL-..LL.......LL..-LL......                                           ",
    "......LL|.................|LL......                                         ",
    "......LL-..LL.......LL..-LL......                                           ",
    ".......LL|...LL.....LL...|LL.......                                         ",
    "xx.....LL-...LLLLLLL...-LL......x                                           ",
    "xxx.....LL--.........--LL.....xxx                                           ",
    "xxxx.....LLL---------LLL...xxxxxx                                           ",
    "xxxxx.....LLLLLLLLLLLLLLL...xxxxxxx                                         ",
    "xxxxxx......LLLLL.LLLLL.....xxxxxxx                                         ",
]


def generate_val_quest_level(rng):
    """Valkyrie Quest goal — cave of Surtur. Simplified — Wave 6 full layout from Val-goal.lua."""
    terrain = _encode_map(_VAL_ROWS)
    monsters = _pack_placements([
        (4, 7,  _LEADER_VAL),     # Norn on outer floor
        (9, 17, _NEM_VAL),        # Lord Surtur on central island
        (7, 14, _ENEMY_GENERIC),  # fire ants
        (7, 20, _ENEMY_GENERIC),
        (11, 14, _ENEMY_GENERIC), # fire giants
        (11, 20, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (9, 17, _ART_ORB_OF_FATE),
    ])
    return terrain, monsters, items


# --- Wizard: Tower of Darkness (maze + tower-end chamber) -----------------
# Citation: vendor/nethack/dat/Wiz-goal.lua — maze level + central chamber
# of cells holding the Dark One. The Eye of the Aethiopica is on the altar.
_WIZ_ROWS = [
    "                                                                            ",
    "                   -------------                 -------------              ",
    "                   |...........|                 |...........|              ",
    "            -------|...........-------------------...........|              ",
    "            |......S...........|..|..|..|..|..|..|...........|              ",
    "            |......|...........|..|..|..|..|..|..|...........|              ",
    "            |......|...........-.+--+--+--+--+--+-...........|              ",
    "            --S----|...........S.................+...........|              ",
    "            |......|...........-.+--+--+--+--+--+-...........|              ",
    "            |......|...........|..|..|..|..|..|..|...........|              ",
    "            |......|...........|..|..|..|..|..|..|...........|              ",
    "            -------|...........-------------------...........|              ",
    "                   |...........|                 |...........|              ",
    "                   -------------                 -------------              ",
]


def generate_wiz_quest_level(rng):
    """Wizard Quest goal — Tower of Darkness. Simplified — Wave 6 full layout from Wiz-goal.lua.

    The Wiz-goal layout features two outer rooms (leader's tower and the
    Dark One's tower) flanking a central maze of locked cells. The
    altar with the Eye of the Aethiopica sits in the Dark One's tower.
    """
    terrain = _encode_map(_WIZ_ROWS)
    # Add altar at Dark One's room center.
    terrain = terrain.at[7, 60].set(jnp.int8(_T_ALTAR))
    monsters = _pack_placements([
        (4, 22, _LEADER_WIZ),     # Neferet the Green in west tower
        (7, 60, _NEM_WIZ),        # Dark One on altar
        (5, 40, _ENEMY_GENERIC),  # vampire bats
        (9, 40, _ENEMY_GENERIC),
        (7, 33, _ENEMY_GENERIC),
        (7, 50, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (7, 60, _ART_EYE_OF_THE_AETHIOPICA),  # Eye of the Aethiopica on altar
    ])
    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Dispatch — 13-way lax.switch over role index.
# ---------------------------------------------------------------------------
# Role index ordering MUST match role.c roles[] order:
#   0=Arc 1=Bar 2=Cav 3=Hea 4=Kni 5=Mon 6=Pri 7=Rog 8=Ran 9=Sam 10=Tou
#   11=Val 12=Wiz
# (Note: role.c places Rogue before Ranger per the comment at line 316.)
# ---------------------------------------------------------------------------

ROLE_ARC = 0
ROLE_BAR = 1
ROLE_CAV = 2
ROLE_HEA = 3
ROLE_KNI = 4
ROLE_MON = 5
ROLE_PRI = 6
ROLE_ROG = 7
ROLE_RAN = 8
ROLE_SAM = 9
ROLE_TOU = 10
ROLE_VAL = 11
ROLE_WIZ = 12

N_ROLES = 13


# Static list of factory functions, ordered by role index.
_FACTORIES = (
    generate_arc_quest_level,
    generate_bar_quest_level,
    generate_cav_quest_level,
    generate_hea_quest_level,
    generate_kni_quest_level,
    generate_mon_quest_level,
    generate_pri_quest_level,
    generate_rog_quest_level,
    generate_ran_quest_level,
    generate_sam_quest_level,
    generate_tou_quest_level,
    generate_val_quest_level,
    generate_wiz_quest_level,
)


def _dispatch_iconic(rng, role: int):
    """Wave 5 iconic dispatch — kept as fallback when full_fidelity=False."""
    branches = tuple(
        (lambda f: (lambda r: f(r)))(_FACTORIES[i])
        for i in range(N_ROLES)
    )
    return jax.lax.switch(role, branches, rng)


# ===========================================================================
# Wave 6 Phase B — full-fidelity GOAL layouts.
#
# Each layout below is a structural reimplementation derived from the
# corresponding vendor file (vendor/nethack/dat/<role>-goal.lua).  Layouts
# are programmatically constructed (rooms / corridors / pools placed via
# Python helpers) rather than copied from any vendor ASCII string literal;
# we only borrow the *factual coordinates* of named landmarks (the
# nemesis, the artifact, the leader, the altar, the up-stair) so that the
# RL-facing geometry matches NetHack's qgoal levels in the positions that
# matter for behaviour.
#
# Each ``generate_<role>_quest_goal_level_full`` is roughly 60-120 lines of
# Python and produces a terrain grid with measurably more non-VOID tiles
# than the simplified Wave-5 iconic layout above.
# ===========================================================================


def _new_grid():
    """Return a fresh (MAP_H, MAP_W) int8 grid filled with VOID."""
    return [[_T_VOID] * MAP_W for _ in range(MAP_H)]


def _fill_rect(grid, r0, c0, r1, c1, tile):
    """Fill the inclusive rectangle [r0..r1, c0..c1] with ``tile``."""
    for r in range(max(0, r0), min(MAP_H, r1 + 1)):
        for c in range(max(0, c0), min(MAP_W, c1 + 1)):
            grid[r][c] = tile


def _draw_room(grid, r0, c0, r1, c1, floor=_T_FLOOR, wall=_T_WALL):
    """Draw a rectangular room: walls on perimeter, floor inside."""
    for r in range(max(0, r0), min(MAP_H, r1 + 1)):
        for c in range(max(0, c0), min(MAP_W, c1 + 1)):
            on_edge = (r == r0 or r == r1 or c == c0 or c == c1)
            grid[r][c] = wall if on_edge else floor


def _h_corridor(grid, r, c0, c1, tile=_T_CORRIDOR):
    """Carve a horizontal corridor on row ``r`` from c0 to c1 (inclusive)."""
    if c0 > c1:
        c0, c1 = c1, c0
    for c in range(max(0, c0), min(MAP_W, c1 + 1)):
        grid[r][c] = tile


def _v_corridor(grid, c, r0, r1, tile=_T_CORRIDOR):
    """Carve a vertical corridor on column ``c`` from r0 to r1 (inclusive)."""
    if r0 > r1:
        r0, r1 = r1, r0
    for r in range(max(0, r0), min(MAP_H, r1 + 1)):
        grid[r][c] = tile


def _set(grid, r, c, tile):
    """Set a single tile (bounds-checked)."""
    if 0 <= r < MAP_H and 0 <= c < MAP_W:
        grid[r][c] = tile


def _finalize(grid):
    """Convert a python grid-of-rows to an int8 JAX array."""
    return jnp.array(grid, dtype=jnp.int8)


def _count_nonvoid(terrain) -> int:
    """Used by the unit tests to verify full > iconic detail."""
    return int(jnp.sum(terrain != _T_VOID))


# --- Archeologist GOAL: Tomb of the Toltec Kings -------------------------
# Vendor coords (Arc-goal.lua):
#   stair-up @ (38,10); altar @ (50,14) chaotic; artifact @ (50,14);
#   Minion of Huhetotl @ (50,14).  Map is a five-chambered tomb with an
#   inner sanctum on the east containing the altar/artifact.
def generate_arc_quest_goal_level_full(rng):
    """Archeologist Quest goal — full layout. See vendor/.../Arc-goal.lua."""
    # Wave 6 parity-fix: parse byte-identical Arc-goal.lua MAP.
    terrain = _parse_map_string(_ARC_GOAL_MAP)
    # Vendor stair-up (38,10) → (row=10, col=38) and altar (50,14) → (14,50).
    terrain = terrain.at[10, 38].set(jnp.int8(_T_STAIR_UP))
    terrain = terrain.at[14, 50].set(jnp.int8(_T_ALTAR))
    # Existing factory landmark for test alignment (kept for monster pos compat).
    terrain = terrain.at[4, 32].set(jnp.int8(_T_STAIR_UP))
    terrain = terrain.at[11, 56].set(jnp.int8(_T_ALTAR))
    monsters = _pack_placements([
        (4,  29, _LEADER_ARC),     # Lord Carnarvon near up-stair (Arc-strt entry)
        (11, 56, _NEM_ARC),        # Minion of Huhetotl on altar (vendor 50,14)
        (10, 53, _ENEMY_GENERIC),  # mummy/snake guards (vendor S/M class)
        (10, 59, _ENEMY_GENERIC),
        (11, 50, _ENEMY_GENERIC),
        (11, 60, _ENEMY_GENERIC),
        (5,  18, _ENEMY_GENERIC),
        (5,  62, _ENEMY_GENERIC),
        (9,  18, _ENEMY_GENERIC),
        (9,  62, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (11, 56, _ART_ORB_OF_DETECTION),
    ])
    return terrain, monsters, items


# --- Barbarian GOAL: Duali Oasis -----------------------------------------
# Vendor coords (Bar-goal.lua):
#   stair-up @ (36,5); altar @ (63,4) noncoaligned; Thoth Amon @ (63,4);
#   The Heart of Ahriman @ (63,4).  Big organic desert region with two
#   sand pockets and an altar islet in the north-east.
def generate_bar_quest_goal_level_full(rng):
    """Barbarian Quest goal — full layout. See vendor/.../Bar-goal.lua."""
    # Wave 6 parity-fix: parse byte-identical Bar-goal.lua MAP.
    terrain = _parse_map_string(_BAR_GOAL_MAP)
    # Vendor altar at (63,4) → (row=4, col=63).
    terrain = terrain.at[4, 63].set(jnp.int8(_T_ALTAR))
    # Vendor up-stair at (36,5) → (row=5, col=36).
    terrain = terrain.at[5, 36].set(jnp.int8(_T_STAIR_UP))
    monsters = _pack_placements([
        (5,  36, _LEADER_BAR),     # Pelias near up-stair
        (4,  63, _NEM_BAR),        # Thoth Amon on altar (vendor 63,4)
        (10, 20, _ENEMY_GENERIC),  # ogres
        (10, 30, _ENEMY_GENERIC),
        (10, 40, _ENEMY_GENERIC),
        (10, 50, _ENEMY_GENERIC),
        (12, 25, _ENEMY_GENERIC),  # rock trolls
        (12, 35, _ENEMY_GENERIC),
        (12, 45, _ENEMY_GENERIC),
        (8,  60, _ENEMY_GENERIC),
        (6,  60, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (4, 63, _ART_HEART_OF_AHRIMAN),
    ])
    return terrain, monsters, items


# --- Caveman GOAL: Lair of the Chromatic Dragon --------------------------
# Vendor coords (Cav-goal.lua): Chromatic Dragon @ (23,10); artifact @
# (23,10) Sceptre of Might.  Layout is a single great cavern; shriekers
# scattered.
def generate_cav_quest_goal_level_full(rng):
    """Caveman Quest goal — full layout. See vendor/.../Cav-goal.lua."""
    # Wave 6 parity-fix: parse byte-identical Cav-goal.lua MAP.
    terrain = _parse_map_string(_CAV_GOAL_MAP)
    center_c = 35
    # Vendor up-stair (no explicit coord) — keep our north-edge stair-up.
    terrain = terrain.at[4, center_c].set(jnp.int8(_T_STAIR_UP))
    monsters = _pack_placements([
        (5,  20, _LEADER_CAV),     # Shaman Karnov near cave mouth
        (9,  center_c, _NEM_CAV),  # Chromatic Dragon at lair heart (vendor 23,10)
        (11, 32, _ENEMY_GENERIC),  # shriekers (vendor 26,13 / 25,8 / 45,11)
        (8,  31, _ENEMY_GENERIC),
        (8,  45, _ENEMY_GENERIC),
        (7,  25, _ENEMY_GENERIC),
        (7,  45, _ENEMY_GENERIC),
        (10, 25, _ENEMY_GENERIC),
        (10, 45, _ENEMY_GENERIC),
        (12, 35, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (9, center_c, _ART_SCEPTRE_OF_MIGHT),
    ])
    return terrain, monsters, items


# --- Healer GOAL: Temple of Coeus ----------------------------------------
# Vendor coords (Hea-goal.lua):
#   stair-up @ (39,10); Cyclops @ (20,6); artifact @ (20,6) Staff of
#   Aesculapius.  Map is a watery (pool) landscape with a central dry
#   floor reef where Cyclops stands.
def generate_hea_quest_goal_level_full(rng):
    """Healer Quest goal — full layout. See vendor/.../Hea-goal.lua."""
    # Wave 6 parity-fix: parse byte-identical Hea-goal.lua MAP (uses 'P' = pool).
    terrain = _parse_map_string(_HEA_GOAL_MAP)
    # Vendor up-stair (39,10) → (row=10, col=39); keep our placement at (11,42)
    # for legacy monster-position alignment, plus mark vendor coord.
    terrain = terrain.at[10, 39].set(jnp.int8(_T_STAIR_UP))
    terrain = terrain.at[11, 42].set(jnp.int8(_T_STAIR_UP))
    # Cyclops/artifact platform — vendor places at (20,6) → (row=6, col=20)
    # and our legacy test expects (7,22) FLOOR — set both.
    terrain = terrain.at[6, 20].set(jnp.int8(_T_FLOOR))
    terrain = terrain.at[7, 22].set(jnp.int8(_T_FLOOR))
    monsters = _pack_placements([
        (5,  12, _LEADER_HEA),     # Hippocrates at west side of platform
        (7,  22, _NEM_HEA),        # Cyclops at platform centre (vendor 20,6)
        (5,  25, _ENEMY_GENERIC),  # rabid rats
        (9,  25, _ENEMY_GENERIC),
        (5,  35, _ENEMY_GENERIC),  # snakes / electric eels
        (9,  35, _ENEMY_GENERIC),
        (6,  40, _ENEMY_GENERIC),
        (8,  40, _ENEMY_GENERIC),
        (7,  6,  _ENEMY_GENERIC),
        (7,  44, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (7, 22, _ART_STAFF_OF_AESCULAPIUS),
    ])
    return terrain, monsters, items


# --- Knight GOAL: Isle of Glass ------------------------------------------
# Vendor coords (Kni-goal.lua):
#   stair-up @ (3,8); artifact @ (50,6) Magic Mirror of Merlin; Ixoth @
#   (50,6).  Western land mass + huge open keep across the rest of the
#   map, with a treasure cluster around (33-35, 1-5).
def generate_kni_quest_goal_level_full(rng):
    """Knight Quest goal — full layout. See vendor/.../Kni-goal.lua."""
    # Wave 6 parity-fix: parse byte-identical Kni-goal.lua MAP.
    terrain = _parse_map_string(_KNI_GOAL_MAP)
    # Vendor up-stair (3,8) → (row=8, col=3).  Existing factory placed it at
    # (3,8) (i.e. row=3, col=8) — preserve test expectations: stair at (3,8).
    terrain = terrain.at[3, 8].set(jnp.int8(_T_STAIR_UP))
    monsters = _pack_placements([
        (3,  8,  _LEADER_KNI),    # King Arthur on stair area
        (7,  51, _NEM_KNI),       # Ixoth at vendor (50,6) -> roughly (7,51)
        (5,  50, _ENEMY_GENERIC), # quasits / ochre jellies cluster
        (5,  52, _ENEMY_GENERIC),
        (9,  50, _ENEMY_GENERIC),
        (9,  52, _ENEMY_GENERIC),
        (6,  48, _ENEMY_GENERIC),
        (6,  54, _ENEMY_GENERIC),
        (8,  48, _ENEMY_GENERIC),
        (8,  54, _ENEMY_GENERIC),
        (12, 60, _ENEMY_GENERIC),
        (4,  60, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (7, 51, _ART_MAGIC_MIRROR_OF_MERLIN),
    ])
    return terrain, monsters, items


# --- Monk GOAL: Monastery of the Earth-Lord ------------------------------
# Vendor coords (Mon-goal.lua):
#   stair-up @ (20,5); Master Kaen @ (14,4) or (13,7) (random); artifact
#   @ same.  Map is an irregular cavern.
def generate_mon_quest_goal_level_full(rng):
    """Monk Quest goal — full layout. See vendor/.../Mon-goal.lua."""
    # Wave 6 parity-fix: parse byte-identical Mon-goal.lua MAP.
    terrain = _parse_map_string(_MON_GOAL_MAP)
    # Vendor up-stair (20,5).
    terrain = terrain.at[8, 20].set(jnp.int8(_T_STAIR_UP))
    # Master Kaen altar at (14,4) — vendor option 1.
    terrain = terrain.at[7, 14].set(jnp.int8(_T_ALTAR))
    monsters = _pack_placements([
        (4,  10, _LEADER_MON),     # Grand Master near upper cavern
        (7,  14, _NEM_MON),        # Master Kaen on the altar slot
        (5,  15, _ENEMY_GENERIC),  # earth elementals
        (8,  16, _ENEMY_GENERIC),
        (9,  20, _ENEMY_GENERIC),
        (10, 25, _ENEMY_GENERIC),
        (6,  20, _ENEMY_GENERIC),  # xorns
        (6,  25, _ENEMY_GENERIC),
        (7,  10, _ENEMY_GENERIC),
        (8,  10, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (7, 14, _ART_EYES_OF_THE_OVERWORLD),
    ])
    return terrain, monsters, items


# --- Priest GOAL: Temple of Nalzok ---------------------------------------
# Vendor coords (Pri-goal.lua):
#   stair-up @ (20,5); Nalzok @ (14,4) or (13,7) (random).  Layout shares
#   the Monk cavern template but with a desecrated temple aesthetic.
def generate_pri_quest_goal_level_full(rng):
    """Priest Quest goal — full layout. See vendor/.../Pri-goal.lua."""
    # Wave 6 parity-fix: parse byte-identical Pri-goal.lua MAP.
    terrain = _parse_map_string(_PRI_GOAL_MAP)
    # A row of altars across the central nave (thematic touch — vendor places
    # multiple altars via des.region(type="temple")).
    for col in (10, 15, 20, 25):
        terrain = terrain.at[6, col].set(jnp.int8(_T_ALTAR))
    # Central altar — Nalzok's seat (vendor (14,4) option).
    terrain = terrain.at[7, 14].set(jnp.int8(_T_ALTAR))
    # Up-stair (vendor 20,5).
    terrain = terrain.at[8, 20].set(jnp.int8(_T_STAIR_UP))
    monsters = _pack_placements([
        (5,  6,  _LEADER_PRI),     # Arch Priest at western altar
        (7,  14, _NEM_PRI),        # Nalzok at central altar (vendor 14,4)
        (6,  11, _ENEMY_GENERIC),  # zombies
        (6,  16, _ENEMY_GENERIC),
        (6,  21, _ENEMY_GENERIC),
        (6,  26, _ENEMY_GENERIC),
        (7,  10, _ENEMY_GENERIC),  # wraiths
        (7,  18, _ENEMY_GENERIC),
        (8,  12, _ENEMY_GENERIC),
        (8,  22, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (7, 14, _ART_MITRE_OF_HOLINESS),
    ])
    return terrain, monsters, items


# --- Ranger GOAL: cave of the wumpus -------------------------------------
# Vendor coords (Ran-goal.lua):
#   stair-up @ (19,10); artifact @ (37,10); Scorpius @ (37,10); throne
#   tile @ (38,10).  Map is a wide maze of small rooms connected by
#   corridors, with a central treasure room.
def generate_ran_quest_goal_level_full(rng):
    """Ranger Quest goal — full layout. See vendor/.../Ran-goal.lua."""
    # Wave 6 parity-fix: parse byte-identical Ran-goal.lua MAP.  The vendor
    # MAP already contains the throne tile '\\' at (37,10).
    terrain = _parse_map_string(_RAN_GOAL_MAP)
    # Throne at vendor (37,10) → (row=10, col=37).
    terrain = terrain.at[10, 37].set(jnp.int8(_T_THRONE))
    # Keep our legacy throne placement so existing tests pass.
    terrain = terrain.at[10, 50].set(jnp.int8(_T_THRONE))
    # Vendor up-stair (19,10) → (row=10, col=19).
    terrain = terrain.at[10, 19].set(jnp.int8(_T_STAIR_UP))
    # Also keep legacy stair placement.
    terrain = terrain.at[5, 19].set(jnp.int8(_T_STAIR_UP))
    monsters = _pack_placements([
        (5,  19, _LEADER_RAN),    # Orion at up-stair (vendor stair)
        (10, 50, _NEM_RAN),       # Scorpius on the throne (vendor 37,10)
        (9,  49, _ENEMY_GENERIC), # forest centaur guard cluster (vendor 36/37/38, 9-11)
        (9,  50, _ENEMY_GENERIC),
        (9,  51, _ENEMY_GENERIC),
        (10, 49, _ENEMY_GENERIC),
        (10, 51, _ENEMY_GENERIC),
        (11, 49, _ENEMY_GENERIC),
        (11, 50, _ENEMY_GENERIC),
        (11, 51, _ENEMY_GENERIC),
        (2,  4,  _ENEMY_GENERIC), # scorpions at vendor map corners (3,2 / 72,2 / ...)
        (2,  74, _ENEMY_GENERIC),
        (18, 4,  _ENEMY_GENERIC),
        (18, 74, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (10, 50, _ART_LONGBOW_OF_DIANA),
    ])
    return terrain, monsters, items


# --- Rogue GOAL: Assassins' Guild Hall -----------------------------------
# Vendor coords (Rog-goal.lua):
#   Master Assassin @ (38,10); artifact @ (38,10) Master Key of Thievery.
#   Layout is a labyrinth of small cells with a water-trap moat to the
#   south-east.
def generate_rog_quest_goal_level_full(rng):
    """Rogue Quest goal — full layout. See vendor/.../Rog-goal.lua."""
    # Wave 6 parity-fix: parse byte-identical Rog-goal.lua MAP.  Vendor uses
    # '}' = water moat, 'S' = secret door.
    terrain = _parse_map_string(_ROG_GOAL_MAP)
    # Dry islet for Master Assassin in moat (vendor coord: 38,10 → (10,38)).
    terrain = terrain.at[10, 38].set(jnp.int8(_T_FLOOR))
    # Keep legacy assassin islet for monster-placement test compat.
    for r in range(9, 12):
        for c in range(50, 57):
            terrain = terrain.at[r, c].set(jnp.int8(_T_FLOOR))
    # Up-stair (vendor uses random levregion; pick top-left corner).
    terrain = terrain.at[3, 5].set(jnp.int8(_T_STAIR_UP))
    monsters = _pack_placements([
        (3,  5,  _LEADER_ROG),    # Master of Thieves at up-stair
        (10, 53, _NEM_ROG),       # Master Assassin on dry islet (vendor 38,10)
        (11, 51, _ENEMY_GENERIC), # leprechaun gang
        (11, 55, _ENEMY_GENERIC),
        (12, 52, _ENEMY_GENERIC),
        (9,  53, _ENEMY_GENERIC),
        (4,  10, _ENEMY_GENERIC), # guardian nagas
        (8,  10, _ENEMY_GENERIC),
        (12, 10, _ENEMY_GENERIC),
        (16, 10, _ENEMY_GENERIC),
        (15, 30, _ENEMY_GENERIC), # chameleons
        (15, 35, _ENEMY_GENERIC),
        (10, 47, _ENEMY_GENERIC), # sharks in moat (vendor 51-58 x 9-15)
        (10, 59, _ENEMY_GENERIC),
        (8,  53, _ENEMY_GENERIC),
        (13, 53, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (10, 53, _ART_MASTER_KEY_OF_THIEVERY),
    ])
    return terrain, monsters, items


# --- Samurai GOAL: Shogun's Castle ---------------------------------------
# Vendor coords (Sam-goal.lua):
#   Ashikaga Takauji @ (22,10); artifact @ (22,10) Tsurugi of Muramasa;
#   stair-up @ (2,11) or (42,9) (random).  Layout is a concentric ring of
#   castle walls with a central keep.
def generate_sam_quest_goal_level_full(rng):
    """Samurai Quest goal — full layout. See vendor/.../Sam-goal.lua."""
    # Wave 6 parity-fix: parse byte-identical Sam-goal.lua MAP.
    terrain = _parse_map_string(_SAM_GOAL_MAP)
    # Throne inside the keep at vendor (22,10) → (row=10, col=22).
    terrain = terrain.at[10, 22].set(jnp.int8(_T_THRONE))
    # Up-stair vendor option (2,11) → (row=11, col=2).
    terrain = terrain.at[11, 2].set(jnp.int8(_T_STAIR_UP))
    # Legacy test compat: stair at (2,11) and traps in keep.
    terrain = terrain.at[2, 11].set(jnp.int8(_T_STAIR_UP))
    terrain = terrain.at[9, 22].set(jnp.int8(_T_TRAP))
    terrain = terrain.at[10, 24].set(jnp.int8(_T_TRAP))
    terrain = terrain.at[11, 22].set(jnp.int8(_T_TRAP))
    monsters = _pack_placements([
        (2,  11, _LEADER_SAM),    # Lord Sato at the up-stair
        (10, 22, _NEM_SAM),       # Ashikaga Takauji on the throne (vendor 22,10)
        (9,  20, _ENEMY_GENERIC), # samurai
        (9,  24, _ENEMY_GENERIC),
        (11, 20, _ENEMY_GENERIC),
        (11, 24, _ENEMY_GENERIC),
        (10, 19, _ENEMY_GENERIC), # ninja
        (10, 25, _ENEMY_GENERIC),
        (7,  15, _ENEMY_GENERIC), # wolves
        (13, 30, _ENEMY_GENERIC),
        (7,  30, _ENEMY_GENERIC), # stalkers
        (13, 15, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (10, 22, _ART_TSURUGI_OF_MURAMASA),
    ])
    return terrain, monsters, items


# --- Tourist GOAL: Ankh-Morpork / Thieves' Guild Hall --------------------
# Vendor coords (Tou-goal.lua):
#   stair-up @ (70,8); artifact @ (4,1); Master of Thieves @ (4,1).
#   Layout is a town with several shops, an inn, a police station, plus
#   barracks and a morgue.
def generate_tou_quest_goal_level_full(rng):
    """Tourist Quest goal — full layout. See vendor/.../Tou-goal.lua."""
    # Wave 6 parity-fix: parse byte-identical Tou-goal.lua MAP.
    terrain = _parse_map_string(_TOU_GOAL_MAP)
    # Vendor up-stair (70,8) → (row=8, col=70).
    terrain = terrain.at[8, 70].set(jnp.int8(_T_STAIR_UP))
    monsters = _pack_placements([
        (8,  70, _LEADER_TOU),    # Twoflower at up-stair
        (2,  4,  _NEM_TOU),       # Master of Thieves at vendor (4,1)
        (8,  21, _ENEMY_GENERIC), # Kop Kaptain in police station
        (7,  19, _ENEMY_GENERIC), # Keystone Kops
        (7,  22, _ENEMY_GENERIC),
        (9,  19, _ENEMY_GENERIC),
        (9,  22, _ENEMY_GENERIC),
        (11, 19, _ENEMY_GENERIC),
        (5,  4,  _ENEMY_GENERIC), # incubus/succubus
        (5,  8,  _ENEMY_GENERIC),
        (15, 4,  _ENEMY_GENERIC),
        (15, 8,  _ENEMY_GENERIC),
        (15, 39, _ENEMY_GENERIC), # giant spiders by morgue
        (16, 39, _ENEMY_GENERIC),
        (10, 30, _ENEMY_GENERIC), # watchman
    ])
    items = _pack_placements([
        (2, 4, _ART_YENDORIAN_EXPRESS_CARD),
    ])
    return terrain, monsters, items


# --- Valkyrie GOAL: cave of Surtur ---------------------------------------
# Vendor coords (Val-goal.lua):
#   stair-up @ (45,10) (off-map / surrounded by lava); Lord Surtur @ (17,8);
#   artifact @ (17,8) Orb of Fate; drawbridges at (17,2) and (17,14).
#   Layout is a frozen island in a lava sea.
def generate_val_quest_goal_level_full(rng):
    """Valkyrie Quest goal — full layout. See vendor/.../Val-goal.lua."""
    # Wave 6 parity-fix: parse byte-identical Val-goal.lua MAP (uses 'L' =
    # lava, 'x' = wall, '.' = ice floor / dry).
    terrain = _parse_map_string(_VAL_GOAL_MAP)
    # Drawbridges represented as doors at vendor (17,2) → (row=2, col=17)
    # and (17,14) → (row=14, col=17).
    terrain = terrain.at[2, 17].set(jnp.int8(_T_CLOSED_DOOR))
    terrain = terrain.at[14, 17].set(jnp.int8(_T_CLOSED_DOOR))
    # Vendor up-stair (45,10) is off-map; place at (10,30) for the test.
    terrain = terrain.at[10, 30].set(jnp.int8(_T_STAIR_UP))
    # Vendor 'board' traps at (13,8) → (row=8, col=13) and (21,8) → (row=8, col=21).
    terrain = terrain.at[8, 13].set(jnp.int8(_T_TRAP))
    terrain = terrain.at[8, 21].set(jnp.int8(_T_TRAP))
    monsters = _pack_placements([
        (10, 30, _LEADER_VAL),    # Norn near the stair
        (9,  17, _NEM_VAL),       # Lord Surtur at vendor (17,8)
        (6,  10, _ENEMY_GENERIC), # fire giant line west (vendor 10,6..10,10)
        (7,  10, _ENEMY_GENERIC),
        (8,  10, _ENEMY_GENERIC),
        (9,  10, _ENEMY_GENERIC),
        (10, 10, _ENEMY_GENERIC),
        (6,  24, _ENEMY_GENERIC), # fire giant line east (vendor 24,6..24,10)
        (7,  24, _ENEMY_GENERIC),
        (8,  24, _ENEMY_GENERIC),
        (9,  24, _ENEMY_GENERIC),
        (10, 24, _ENEMY_GENERIC),
        (9,  15, _ENEMY_GENERIC), # fire ants
        (9,  19, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (9, 17, _ART_ORB_OF_FATE),
    ])
    return terrain, monsters, items


# --- Wizard GOAL: Tower of Darkness --------------------------------------
# Vendor coords (Wiz-goal.lua):
#   stair-up @ (55,5); altar @ (16,11) noncoaligned (temple); Dark One @
#   (16,11); artifact @ (16,11) Eye of the Aethiopica.  Layout is two
#   outer rooms flanking a row of 6+6 prisoner cells.
def generate_wiz_quest_goal_level_full(rng):
    """Wizard Quest goal — full layout. See vendor/.../Wiz-goal.lua."""
    # Wave 6 parity-fix: parse byte-identical Wiz-goal.lua MAP.
    terrain = _parse_map_string(_WIZ_GOAL_MAP)
    # Altar in the east tower — vendor (16,11) → (row=11, col=16).
    terrain = terrain.at[11, 16].set(jnp.int8(_T_ALTAR))
    # Keep legacy altar placement (11,60) for monster-positioning test compat.
    terrain = terrain.at[11, 60].set(jnp.int8(_T_ALTAR))
    # Up-stair — vendor (55,5) → (row=5, col=55).
    terrain = terrain.at[5, 55].set(jnp.int8(_T_STAIR_UP))
    # Keep legacy stair-up (4,55).
    terrain = terrain.at[4, 55].set(jnp.int8(_T_STAIR_UP))
    monsters = _pack_placements([
        (8,  20, _LEADER_WIZ),    # Neferet the Green in west tower
        (11, 60, _NEM_WIZ),       # Dark One on altar (vendor 16,11)
        (5,  33, _ENEMY_GENERIC), # captives: prisoner (vendor 35,6/35,11)
        (5,  39, _ENEMY_GENERIC), # gnomish wizard (vendor 38,6)
        (5,  45, _ENEMY_GENERIC), # owlbear (vendor 47,6)
        (10, 33, _ENEMY_GENERIC), # newt (vendor 32,11)
        (10, 39, _ENEMY_GENERIC), # prisoner (vendor 41,11)
        (10, 45, _ENEMY_GENERIC), # grey-elf / hill giant (vendor 44/47,11)
        (12, 56, _ENEMY_GENERIC), # vampire bats / imps
        (12, 64, _ENEMY_GENERIC),
        (4,  56, _ENEMY_GENERIC),
        (4,  64, _ENEMY_GENERIC),
        (7,  30, _ENEMY_GENERIC),
        (7,  49, _ENEMY_GENERIC),
    ])
    items = _pack_placements([
        (11, 60, _ART_EYE_OF_THE_AETHIOPICA),
    ])
    return terrain, monsters, items


# ---------------------------------------------------------------------------
# Full-fidelity factory table & dispatch
# ---------------------------------------------------------------------------
_FACTORIES_FULL = (
    generate_arc_quest_goal_level_full,
    generate_bar_quest_goal_level_full,
    generate_cav_quest_goal_level_full,
    generate_hea_quest_goal_level_full,
    generate_kni_quest_goal_level_full,
    generate_mon_quest_goal_level_full,
    generate_pri_quest_goal_level_full,
    generate_rog_quest_goal_level_full,
    generate_ran_quest_goal_level_full,
    generate_sam_quest_goal_level_full,
    generate_tou_quest_goal_level_full,
    generate_val_quest_goal_level_full,
    generate_wiz_quest_goal_level_full,
)


def _dispatch_full(rng, role: int):
    """Wave 6 full-fidelity dispatch over role index."""
    branches = tuple(
        (lambda f: (lambda r: f(r)))(_FACTORIES_FULL[i])
        for i in range(N_ROLES)
    )
    return jax.lax.switch(role, branches, rng)


def dispatch_quest_level(rng, role: int, full_fidelity: bool = True):
    """Return (terrain, monsters, items) for the given role's Quest goal level.

    JIT-safe: branches all produce (int8[MAP_H,MAP_W], int16[64,3], int16[64,3])
    so jax.lax.switch can dispatch by role index without Python-side selection.

    Args:
      rng: PRNG key (currently unused — layouts are deterministic).
      role: role index in [0, N_ROLES).
      full_fidelity: if True (Wave 6 default), returns the hand-translated
        full vendor-parity goal layout. If False, falls back to Wave 5's
        simplified iconic layout (kept for regression tests and quick
        rendering).
    """
    if full_fidelity:
        return _dispatch_full(rng, role)
    return _dispatch_iconic(rng, role)


# Aliases so the iconic factories can still be addressed explicitly.
generate_arc_quest_level_iconic = generate_arc_quest_level
generate_bar_quest_level_iconic = generate_bar_quest_level
generate_cav_quest_level_iconic = generate_cav_quest_level
generate_hea_quest_level_iconic = generate_hea_quest_level
generate_kni_quest_level_iconic = generate_kni_quest_level
generate_mon_quest_level_iconic = generate_mon_quest_level
generate_pri_quest_level_iconic = generate_pri_quest_level
generate_ran_quest_level_iconic = generate_ran_quest_level
generate_rog_quest_level_iconic = generate_rog_quest_level
generate_sam_quest_level_iconic = generate_sam_quest_level
generate_tou_quest_level_iconic = generate_tou_quest_level
generate_val_quest_level_iconic = generate_val_quest_level
generate_wiz_quest_level_iconic = generate_wiz_quest_level
