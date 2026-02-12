import jax.numpy as jnp
from enum import IntEnum


# ============================================================================
# Tile types (only those used in MiniHack arenas)
# ============================================================================
class TileType(IntEnum):
    VOID = 0
    FLOOR = 1
    VWALL = 2     # Vertical wall |
    HWALL = 3     # Horizontal wall -
    TLCORN = 4    # Top-left corner
    TRCORN = 5    # Top-right corner
    BLCORN = 6    # Bottom-left corner
    BRCORN = 7    # Bottom-right corner
    ALTAR = 8
    UPSTAIR = 9   # Upstaircase <
    DOWNSTAIR = 10    # Downstaircase >
    LAVA = 11         # Lava (damages on step)
    CLOUD = 12        # Cloud/fog (walkable, blocks vision in NetHack)
    TREE = 13         # Tree (blocks movement)
    IRON_BARS = 14    # Iron bars (blocks movement)
    CORRIDOR = 15     # Corridor # (walkable)
    DOOR_CLOSED = 16  # Closed door +
    DOOR_OPEN = 17    # Open door (walkable)
    DOOR_LOCKED = 18  # Locked door (blocks until kicked/unlocked)
    BOULDER = 19      # Boulder (Sokoban, blocks until pushed)
    PIT = 20          # Pit trap (Sokoban)
    PIT_FILLED = 21   # Filled pit (boulder pushed in, walkable)
    TRAP_BOARD = 22   # Board trap (instant death in memento)

NUM_TILE_TYPES = len(TileType)

SOLID_TILES = jnp.array([
    TileType.VOID, TileType.VWALL, TileType.HWALL,
    TileType.TLCORN, TileType.TRCORN, TileType.BLCORN, TileType.BRCORN,
    TileType.TREE, TileType.IRON_BARS,
    TileType.DOOR_CLOSED, TileType.DOOR_LOCKED, TileType.BOULDER,
], dtype=jnp.int32)


# ============================================================================
# Monster types
# ============================================================================
class MonsterType(IntEnum):
    NONE = 0
    HUMAN_ZOMBIE = 1
    PRIEST = 2
    # Tier 2 monsters
    GIANT_RAT = 3          # 'r' — quest_medium
    COCKATRICE = 4         # 'L' — hidenseek
    NAGA_HATCHLING = 5     # 'N' — hidenseek
    MINOTAUR = 6           # 'H' — quest, quest_hard, hidenseek
    OGRE = 7               # 'O' — hidenseek
    BABY_RED_DRAGON = 8    # 'D' — hidenseek
    TROLL = 9              # 'T' — hidenseek
    # Tier 3 monsters (memento)
    BLUE_JELLY = 10        # 'j' — memento_easy/short
    SPOTTED_JELLY = 11     # 'j' — memento_hard
    LICHEN = 12            # 'F' — memento_easy/short
    RED_MOLD = 13          # 'F' — memento_hard
    GREEN_MOLD = 14        # 'F' — memento_hard
    GRID_BUG = 15          # 'x' — memento

NUM_MONSTER_TYPES = len(MonsterType)

# Monster symbols for text rendering
MONSTER_SYMBOLS = {
    int(MonsterType.NONE): ' ',
    int(MonsterType.HUMAN_ZOMBIE): 'Z',
    int(MonsterType.PRIEST): '@',
    int(MonsterType.GIANT_RAT): 'r',
    int(MonsterType.COCKATRICE): 'L',
    int(MonsterType.NAGA_HATCHLING): 'N',
    int(MonsterType.MINOTAUR): 'H',
    int(MonsterType.OGRE): 'O',
    int(MonsterType.BABY_RED_DRAGON): 'D',
    int(MonsterType.TROLL): 'T',
    int(MonsterType.BLUE_JELLY): 'j',
    int(MonsterType.SPOTTED_JELLY): 'j',
    int(MonsterType.LICHEN): 'F',
    int(MonsterType.RED_MOLD): 'F',
    int(MonsterType.GREEN_MOLD): 'F',
    int(MonsterType.GRID_BUG): 'x',
}

# Stats: [level, speed, ac, mr, atk_dice, atk_sides, max_hp, xp_value]
# Human zombie from NetHack 3.7: level 4, speed 6, AC 8, MR 0, attack 1d8
# HD 4 -> avg HP ~= 4*4.5 = 18, XP ~20
# Priest from NetHack: level 5, speed 12, AC 2, attack 1d6, peaceful temple guardian
MONSTER_STATS = jnp.array([
    # [level, speed, ac, mr, atk_dice, atk_sides, max_hp, xp_value]
    [0, 0, 10, 0, 0, 0, 0, 0],       # NONE
    [4, 6, 8, 0, 1, 8, 18, 20],      # HUMAN_ZOMBIE
    [5, 12, 2, 0, 1, 6, 25, 0],      # PRIEST (peaceful, no XP)
    [1, 12, 7, 0, 1, 3, 5, 2],       # GIANT_RAT
    [5, 6, 6, 30, 1, 6, 25, 50],     # COCKATRICE
    [3, 10, 6, 0, 1, 4, 15, 20],     # NAGA_HATCHLING
    [6, 12, 6, 0, 3, 10, 50, 100],   # MINOTAUR
    [5, 9, 5, 0, 2, 6, 30, 60],      # OGRE
    [5, 9, 2, 20, 2, 6, 40, 60],     # BABY_RED_DRAGON
    [7, 12, 4, 0, 3, 6, 50, 80],     # TROLL
    [4, 0, 8, 10, 0, 0, 20, 15],     # BLUE_JELLY (passive)
    [5, 0, 8, 10, 0, 0, 25, 20],     # SPOTTED_JELLY (passive)
    [0, 1, 9, 0, 0, 0, 5, 1],        # LICHEN
    [1, 0, 9, 0, 0, 0, 10, 5],       # RED_MOLD (sessile)
    [1, 0, 9, 0, 0, 0, 10, 5],       # GREEN_MOLD (sessile)
    [0, 12, 9, 0, 1, 1, 1, 1],       # GRID_BUG
], dtype=jnp.int32)

MONSTER_MAX_HP = MONSTER_STATS[:, 6]

# Flags
MF_HOSTILE = 0x020
MF_UNDEAD = 0x008
MF_PEACEFUL = 0x040

MONSTER_FLAGS = jnp.array([
    0x000,                           # NONE
    MF_HOSTILE | MF_UNDEAD,         # HUMAN_ZOMBIE
    MF_PEACEFUL,                    # PRIEST
    MF_HOSTILE,                     # GIANT_RAT
    MF_HOSTILE,                     # COCKATRICE
    MF_HOSTILE,                     # NAGA_HATCHLING
    MF_HOSTILE,                     # MINOTAUR
    MF_HOSTILE,                     # OGRE
    MF_HOSTILE,                     # BABY_RED_DRAGON
    MF_HOSTILE,                     # TROLL
    MF_HOSTILE,                     # BLUE_JELLY
    MF_HOSTILE,                     # SPOTTED_JELLY
    MF_HOSTILE,                     # LICHEN
    MF_HOSTILE,                     # RED_MOLD
    MF_HOSTILE,                     # GREEN_MOLD
    MF_HOSTILE,                     # GRID_BUG
], dtype=jnp.int32)


# ============================================================================
# Actions
# ============================================================================
class Action(IntEnum):
    MOVE_N = 0
    MOVE_S = 1
    MOVE_E = 2
    MOVE_W = 3
    MOVE_NE = 4
    MOVE_NW = 5
    MOVE_SE = 6
    MOVE_SW = 7
    SEARCH = 8
    EAT = 9
    GO_DOWN_STAIRS = 10
    PICKUP = 11
    USE_ITEM = 12
    KICK = 13
    OPEN_DOOR = 14
    UNLOCK_DOOR = 15

NUM_ACTIONS = len(Action)

# Per-tier action counts
NUM_ACTIONS_TIER1 = 11   # Movement (0-7) + SEARCH(8) + EAT(9) + GO_DOWN_STAIRS(10)
NUM_ACTIONS_TIER2 = 14   # Tier 1 + PICKUP(11) + USE_ITEM(12) + KICK(13)
NUM_ACTIONS_TIER3 = 16   # Tier 2 + OPEN_DOOR(14) + UNLOCK_DOOR(15)
NUM_ACTIONS_TIER4 = 9    # Movement (0-7) + SEARCH/WAIT(8)

# Direction vectors for movement actions [row_delta, col_delta]
# row increases downward, col increases rightward
DIRECTION_VECTORS = jnp.array([
    [-1, 0],   # N
    [1, 0],    # S
    [0, 1],    # E
    [0, -1],   # W
    [-1, 1],   # NE
    [-1, -1],  # NW
    [1, 1],    # SE
    [1, -1],   # SW
    [0, 0],    # SEARCH (no movement)
    [0, 0],    # EAT (no movement)
    [0, 0],    # GO_DOWN_STAIRS (no movement)
    [0, 0],    # PICKUP (no movement)
    [0, 0],    # USE_ITEM (no movement)
    [0, 0],    # KICK (direction-dependent, handled separately)
    [0, 0],    # OPEN_DOOR (direction-dependent)
    [0, 0],    # UNLOCK_DOOR (direction-dependent)
], dtype=jnp.int32)


# ============================================================================
# Item types
# ============================================================================
class ItemType(IntEnum):
    NONE = 0
    POTION_LEVITATION = 1
    RING_LEVITATION = 2
    BOOTS_LEVITATION = 3
    WAND_COLD = 4
    FROST_HORN = 5
    WAND_DEATH = 6
    SKELETON_KEY = 7
    APPLE = 8
    GOLD = 9      # Gold piece (auto-pickup in TreasureDash)

NUM_ITEM_TYPES = len(ItemType)

# Items that grant levitation
LEVITATION_ITEMS = jnp.array([
    ItemType.POTION_LEVITATION,
    ItemType.RING_LEVITATION,
    ItemType.BOOTS_LEVITATION,
], dtype=jnp.int32)

# Items that freeze lava
COLD_ITEMS = jnp.array([
    ItemType.WAND_COLD,
    ItemType.FROST_HORN,
], dtype=jnp.int32)


# ============================================================================
# Player defaults (Monk human neutral male, matching sol-main)
# ============================================================================
PLAYER_START_HP = 16
PLAYER_START_MAX_HP = 16
PLAYER_START_AC = 10
PLAYER_START_STRENGTH = 10
PLAYER_START_XP_LEVEL = 1

MAX_PLAYER_LEVEL = 30
NORMAL_SPEED = 12  # Movement points needed to act (NetHack standard)

# XP table from NetHack 3.7 newuexp() — XP_TABLE[i] = XP needed to reach level i+1
# get_xp_for_level(N) returns XP_TABLE[N-1], so XP_TABLE[1] = newuexp(1) = 20 (XP for L2)
# newuexp(lev): lev<10: 10*2^lev, lev<20: 10000*2^(lev-10), lev>=20: 10000000*(lev-19)
XP_TABLE = jnp.array([
    0, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120,           # L1-L10: [0, newuexp(1..9)]
    10000, 20000, 40000, 80000, 160000, 320000, 640000,        # L11-L17: newuexp(10..16)
    1280000, 2560000, 5120000, 10000000, 20000000, 30000000,   # L18-L23: newuexp(17..22)
    40000000, 50000000, 60000000, 70000000, 80000000,          # L24-L28: newuexp(23..27)
    90000000, 100000000,                                        # L29-L30: newuexp(28..29)
], dtype=jnp.int32)

# Score for killing a zombie (NetHack score: difficulty * 4 + 1)
# Human zombie difficulty ~5 -> score ~21
ZOMBIE_KILL_SCORE = 5


# ============================================================================
# Tile sprite indices into tiles.npy (bypasses glyph2tile for pixel rendering)
# ============================================================================
TILE_TYPE_SPRITES = jnp.array([
    850,  # VOID -> stone
    869,  # FLOOR -> room
    851,  # VWALL -> vertical wall |
    852,  # HWALL -> horizontal wall -
    853,  # TLCORN -> top-left corner
    854,  # TRCORN -> top-right corner
    855,  # BLCORN -> bottom-left corner
    856,  # BRCORN -> bottom-right corner
    877,  # ALTAR
    873,  # UPSTAIR -> upstaircase <
    874,  # DOWNSTAIR -> downstaircase >
    884,  # LAVA (molten lava = tile 44 in other.txt, +840 offset)
    890,  # CLOUD -> S_cloud
    868,  # TREE -> S_tree
    867,  # IRON_BARS -> S_bars
    872,  # CORRIDOR
    865,  # DOOR_CLOSED -> S_vcdoor (vertical closed, auto-tiled in renderer)
    863,  # DOOR_OPEN -> S_vodoor (vertical open, auto-tiled in renderer)
    865,  # DOOR_LOCKED (same sprite as closed, auto-tiled in renderer)
    869,  # BOULDER -> render as floor; boulder object sprite overlaid in renderer
    902,  # PIT -> S_pit
    869,  # PIT_FILLED (looks like floor)
    895,  # TRAP_BOARD -> S_squeaky_board
], dtype=jnp.int32)

SPRITE_PLAYER = 345    # Player @ (monk)
SPRITE_BOULDER = 844   # Boulder object sprite (orange circle)

# Item sprites — tile indices from NLE glyph2tile mapping
ITEM_SPRITES = jnp.array([
    0,    # NONE (unused)
    674,  # POTION_LEVITATION — levitation potion
    761,  # RING_LEVITATION — levitation ring
    545,  # BOOTS_LEVITATION — levitation boots
    800,  # WAND_COLD — cold wand
    621,  # FROST_HORN — frost horn tool
    802,  # WAND_DEATH — death wand
    592,  # SKELETON_KEY — skeleton key
    648,  # APPLE — apple food
    807,  # GOLD — gold piece
], dtype=jnp.int32)

MONSTER_SPRITES = jnp.array([
    0,    # NONE (unused)
    246,  # HUMAN_ZOMBIE
    277,  # PRIEST
    191,  # GIANT_RAT (placeholder)
    134,  # COCKATRICE (placeholder)
    166,  # NAGA_HATCHLING (placeholder)
    115,  # MINOTAUR (placeholder)
    175,  # OGRE (placeholder)
    62,   # BABY_RED_DRAGON (placeholder)
    219,  # TROLL (placeholder)
    127,  # BLUE_JELLY (placeholder)
    128,  # SPOTTED_JELLY (placeholder)
    80,   # LICHEN (placeholder)
    81,   # RED_MOLD (placeholder)
    82,   # GREEN_MOLD (placeholder)
    234,  # GRID_BUG (placeholder)
], dtype=jnp.int32)

TILE_SIZE = 16


# ============================================================================
# Tile characters for text rendering
# ============================================================================
TILE_CHARS = {
    int(TileType.VOID): ' ',
    int(TileType.FLOOR): '.',
    int(TileType.VWALL): '|',
    int(TileType.HWALL): '-',
    int(TileType.TLCORN): '-',
    int(TileType.TRCORN): '-',
    int(TileType.BLCORN): '-',
    int(TileType.BRCORN): '-',
    int(TileType.ALTAR): '_',
    int(TileType.UPSTAIR): '<',
    int(TileType.DOWNSTAIR): '>',
    int(TileType.LAVA): 'L',
    int(TileType.CLOUD): 'C',
    int(TileType.TREE): 'T',
    int(TileType.IRON_BARS): 'F',
    int(TileType.CORRIDOR): '#',
    int(TileType.DOOR_CLOSED): '+',
    int(TileType.DOOR_OPEN): '.',
    int(TileType.DOOR_LOCKED): '+',
    int(TileType.BOULDER): '0',
    int(TileType.PIT): '^',
    int(TileType.PIT_FILLED): '.',
    int(TileType.TRAP_BOARD): '^',
}


# ============================================================================
# ZombieHorde-specific constants
# ============================================================================
# Altar position in ZombieHorde (row, col) — priest mills around this
ALTAR_POSITION = jnp.array([2, 2], dtype=jnp.int32)

# Temple region: .des REGION:(1,1,3,3) + buffer row/col for ALLOW_SANCT check
# Hostile monsters can't enter while priest is alive (in_your_sanctuary)
TEMPLE_MIN = jnp.array([1, 1], dtype=jnp.int32)  # (row, col) inclusive
TEMPLE_MAX = jnp.array([4, 4], dtype=jnp.int32)  # (row, col) inclusive
