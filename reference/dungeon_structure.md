# NetHack 3.7 Dungeon Structure Reference

## Core Dimensions

- **COLNO**: 80 (map width in columns)
- **ROWNO**: 21 (map height in rows)
- **MAXDUNGEON**: 16 (maximum number of dungeon branches)
- **MAXLEVEL**: 32 (maximum total levels across all branches)
- **MAXNROFROOMS**: 40 (maximum rooms per level)
- **MAXULEV**: 30 (maximum player level)

## Tile Types (rm.h)

Complete enumeration of all 37 terrain types:

| Type ID | Name | Description |
|---------|------|-------------|
| 0 | STONE | Solid rock, undiggable in some areas |
| 1 | VWALL | Vertical wall |
| 2 | HWALL | Horizontal wall |
| 3 | TLCORNER | Top-left corner wall |
| 4 | TRCORNER | Top-right corner wall |
| 5 | BLCORNER | Bottom-left corner wall |
| 6 | BRCORNER | Bottom-right corner wall |
| 7 | CROSSWALL | Cross wall junction |
| 8 | TUWALL | T-junction up wall |
| 9 | TDWALL | T-junction down wall |
| 10 | TLWALL | T-junction left wall |
| 11 | TRWALL | T-junction right wall |
| 12 | DBWALL | Drawbridge wall |
| 13 | TREE | Tree (passable by certain creatures) |
| 14 | SDOOR | Secret door |
| 15 | SCORR | Secret corridor |
| 16 | POOL | Water pool |
| 17 | MOAT | Moat (water around castle) |
| 18 | WATER | Deep water |
| 19 | DRAWBRIDGE_UP | Raised drawbridge |
| 20 | LAVAPOOL | Lava pool |
| 21 | LAVAWALL | Lava wall |
| 22 | IRONBARS | Iron bars |
| 23 | DOOR | Door (open or closed) |
| 24 | CORR | Corridor |
| 25 | ROOM | Room floor |
| 26 | STAIRS | Stairs up or down |
| 27 | LADDER | Ladder up or down |
| 28 | FOUNTAIN | Fountain |
| 29 | THRONE | Throne |
| 30 | SINK | Sink |
| 31 | GRAVE | Grave |
| 32 | ALTAR | Altar |
| 33 | ICE | Ice (slippery) |
| 34 | DRAWBRIDGE_DOWN | Lowered drawbridge |
| 35 | AIR | Air (elemental planes) |
| 36 | CLOUD | Cloud (elemental planes) |

## Dungeon Branches

Main structure from dungeon.lua:

### Dungeons of Doom (Main Dungeon)
- **Levels**: 25 + 5 random
- **Entry**: Surface level
- **Description**: Primary dungeon containing most special levels

### Gnomish Mines
- **Levels**: 8 + 2 random
- **Branch point**: DL 2-5 of Dungeons of Doom
- **Entry**: Down stairs
- **Contains**: Mines' End with luckstone or candelabrum

### Sokoban
- **Levels**: 4
- **Branch point**: From oracle level upward in Dungeons of Doom
- **Entry**: Up stairs
- **Contains**: Puzzle levels with prizes (bag of holding, amulet of reflection)

### The Quest
- **Levels**: 5 + 2 random
- **Branch point**: Portal on DL 11-13 of Dungeons of Doom
- **Entry**: Magic portal
- **Requirement**: Level 14+, specific alignment
- **Contains**: Quest artifact

### Fort Ludios
- **Levels**: 1
- **Branch point**: Portal on DL 18-22 of Dungeons of Doom
- **Entry**: Magic portal (rare)
- **Contains**: Massive treasure vault guarded by soldiers

### Gehennom
- **Levels**: 20 + 5 random
- **Branch point**: Below the Castle in Dungeons of Doom
- **Entry**: Down from castle
- **Description**: Hell, no teleport zone, contains demon lairs

### Vlad's Tower
- **Levels**: 3
- **Branch point**: Branches upward in Gehennom
- **Entry**: Up stairs
- **Contains**: Vlad the Impaler, candelabrum of Invocation

### Elemental Planes
- **Levels**: 6 (Earth, Air, Fire, Water, Astral)
- **Entry**: Endgame after escaping Gehennom with Amulet
- **Description**: Final challenge sequence

## Special Levels

### Oracle
- **Location**: DL 5-10 of Dungeons of Doom
- **Features**: Oracle NPC, consultations, fountains

### Rogue Level
- **Location**: DL 15-19 of Dungeons of Doom
- **Features**: ASCII graphics tribute to Rogue, different display

### Medusa's Island
- **Location**: DL (end-5) +/-4 levels of Dungeons of Doom
- **Features**: Surrounded by water, Medusa boss, stash of statues

### Castle
- **Location**: DL (end-1) of Dungeons of Doom
- **Features**: Wand of wishing, drawbridge, heavy defenses

### Valley of the Dead
- **Location**: First level of Gehennom
- **Features**: Graveyard, no-teleport zone begins

### Sanctum
- **Location**: Final level of Gehennom
- **Features**: High Altar of Moloch, High Priest, Amulet of Yendor

## Navigation Notes

- Most levels connected by stairs (up and down)
- Some branches use magic portals (Quest, Fort Ludios)
- Gehennom and below: no-teleport zone (requires cursed scroll or quest artifact)
- Return from branches always possible
- Elemental planes are one-way sequence (cannot return)
