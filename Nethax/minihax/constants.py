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

# Stats: [level, speed, ac, mr, atk1_dice, atk1_sides, atk2_dice, atk2_sides, atk3_dice, atk3_sides, atk4_dice, atk4_sides, max_hp, xp_value]
# Human zombie from NetHack 3.7: level 4, speed 6, AC 8, MR 0, attack 1d8
# HD 4 -> avg HP ~= 4*4.5 = 18, XP ~20
# Priest from NetHack: level 5, speed 12, AC 2, attack 1d6, peaceful temple guardian
# Minotaur: level 15 (NetHack monst.c), 3 attacks: claw 3d10, claw 3d10, butt 2d8
MONSTER_STATS = jnp.array([
    #  lvl spd  ac  mr  a1d a1s a2d a2s a3d a3s a4d a4s  hp  xp
    [  0,  0, 10,  0,  0, 0,  0, 0,  0, 0,  0, 0,   0,  0],  # 0  NONE
    [  4,  6,  8,  0,  1, 8,  0, 0,  0, 0,  0, 0,  18, 20],  # 1  HUMAN_ZOMBIE
    [ 12, 12, 10, 50,  4,10,  0, 0,  0, 0,  0, 0,  54,  0],  # 2  PRIEST
    [  1, 12,  7,  0,  1, 3,  0, 0,  0, 0,  0, 0,   5,  2],  # 3  GIANT_RAT
    [  5,  6,  6, 30,  1, 3,  0, 0,  0, 0,  0, 0,  23, 50],  # 4  COCKATRICE (bite 1d3, petrify deferred)
    [  3, 10,  6,  0,  1, 4,  0, 0,  0, 0,  0, 0,  15, 20],  # 5  NAGA_HATCHLING
    [ 15, 12,  6,  0,  3,10,  3,10,  2, 8,  0, 0,  50,100],  # 6  MINOTAUR (claw 3d10, claw 3d10, butt 2d8)
    [  5,  9,  5,  0,  2, 6,  0, 0,  0, 0,  0, 0,  30, 60],  # 7  OGRE
    [  5,  9,  2, 20,  2, 6,  1, 4,  1, 4,  0, 0,  40, 60],  # 8  BABY_RED_DRAGON (bite 2d6, claw 1d4, claw 1d4)
    [  7, 12,  4,  0,  2, 6,  1, 4,  1, 6,  0, 0,  50, 80],  # 9  TROLL (weapon 2d6, claw 1d4, bite 1d6)
    [  4,  0,  8, 10,  0, 0,  0, 0,  0, 0,  0, 0,  20, 15],  # 10 BLUE_JELLY
    [  5,  0,  8, 10,  0, 0,  0, 0,  0, 0,  0, 0,  25, 20],  # 11 SPOTTED_JELLY
    [  0,  1,  9,  0,  0, 0,  0, 0,  0, 0,  0, 0,   5,  1],  # 12 LICHEN
    [  1,  0,  9,  0,  0, 0,  0, 0,  0, 0,  0, 0,  10,  5],  # 13 RED_MOLD
    [  1,  0,  9,  0,  0, 0,  0, 0,  0, 0,  0, 0,  10,  5],  # 14 GREEN_MOLD
    [  0, 12,  9,  0,  1, 1,  0, 0,  0, 0,  0, 0,   1,  1],  # 15 GRID_BUG
], dtype=jnp.int32)

MONSTER_MAX_HP = MONSTER_STATS[:, 12]

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
    APPLY = 12           # Apply self-targeted item — enters slot selection (Tier 3: 2-step, Tier 2: auto-select)
    KICK = 13
    OPEN_DOOR = 14
    UNLOCK_DOOR = 15
    ZAP = 16             # Zap a wand/horn — enters slot selection (Tier 3 only)
    SLOT_0 = 17          # Select inventory slot 0 (during item slot selection)
    SLOT_1 = 18          # Select inventory slot 1
    SLOT_2 = 19          # Select inventory slot 2

NUM_ACTIONS = len(Action)

# Per-tier action counts
NUM_ACTIONS_TIER1 = 11   # Movement (0-7) + OPEN(8) + KICK(9) + SEARCH/WAIT(10)  [matches MiniHack NAVIGATE_ACTIONS]
NUM_ACTIONS_TIER1_EXPLORE = 12  # Tier1 + EAT(11) for ExploreMaze envs
NUM_ACTIONS_TIER2 = 14   # Tier 1 + PICKUP(11) + APPLY(12) + KICK(13)
NUM_ACTIONS_TIER3 = 20   # Tier 2 + OPEN_DOOR(14) + UNLOCK_DOOR(15) + ZAP(16) + SLOT_0/1/2(17-19)
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
    [0, 0],    # APPLY (no movement)
    [0, 0],    # KICK (direction-dependent, handled separately)
    [0, 0],    # OPEN_DOOR (direction-dependent)
    [0, 0],    # UNLOCK_DOOR (direction-dependent)
    [0, 0],    # ZAP (no movement — enters slot selection)
    [0, 0],    # SLOT_0 (no movement — inventory slot select)
    [0, 0],    # SLOT_1
    [0, 0],    # SLOT_2
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

# Items that require directional zap (wands and horns)
DIRECTIONAL_ITEMS = jnp.array([
    ItemType.WAND_COLD,
    ItemType.FROST_HORN,
    ItemType.WAND_DEATH,
], dtype=jnp.int32)

# Items usable via APPLY (self-targeted: potions, rings, boots, key)
APPLY_ITEMS = jnp.array([
    ItemType.POTION_LEVITATION,
    ItemType.RING_LEVITATION,
    ItemType.BOOTS_LEVITATION,
    ItemType.SKELETON_KEY,
], dtype=jnp.int32)

# Items usable via EAT (food)
FOOD_ITEMS = jnp.array([
    ItemType.APPLE,
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

# Score for killing a zombie (NetHack exper.c + more_experienced)
# experience() = 1 + level^2 = 1 + 4^2 = 17, score = 4 * experience() = 68
ZOMBIE_KILL_SCORE = 68


# ============================================================================
# Player Roles (Classes) — from NetHack 3.7 role.c
# ============================================================================
class RoleType(IntEnum):
    ARCHEOLOGIST = 0
    BARBARIAN = 1
    CAVEMAN = 2
    HEALER = 3
    KNIGHT = 4
    MONK = 5
    PRIEST = 6
    RANGER = 7
    ROGUE = 8
    SAMURAI = 9
    TOURIST = 10
    VALKYRIE = 11
    WIZARD = 12

NUM_ROLES = 13


class RaceType(IntEnum):
    HUMAN = 0
    ELF = 1
    DWARF = 2
    GNOME = 3
    ORC = 4

NUM_RACES = 5


# Base stats per role: [STR, INT, WIS, DEX, CON, CHA] — from role.c
ROLE_BASE_STATS = jnp.array([
    [7, 10, 10, 7, 7, 7],    # Archeologist
    [16, 7, 7, 15, 16, 6],   # Barbarian
    [10, 7, 7, 7, 8, 6],     # Caveman
    [7, 7, 13, 7, 11, 16],   # Healer
    [13, 7, 14, 8, 10, 17],  # Knight
    [10, 7, 8, 8, 7, 7],     # Monk
    [7, 7, 10, 7, 7, 7],     # Priest
    [13, 13, 13, 9, 13, 7],  # Ranger
    [7, 7, 7, 10, 7, 6],     # Rogue
    [10, 8, 7, 10, 17, 6],   # Samurai
    [7, 10, 6, 7, 7, 10],    # Tourist
    [10, 7, 7, 7, 10, 7],    # Valkyrie
    [7, 10, 7, 7, 7, 7],     # Wizard
], dtype=jnp.int32)


# HP advancement per role: [infix, inrnd, lofix, lornd, hifix, hirnd]
# in = initial, lo = per-level below xlev, hi = per-level at/above xlev
# HP = fix + rnd(rnd_value) for each tier — from role.c hpadv fields
ROLE_HP_ADV = jnp.array([
    [11, 0, 0, 8, 1, 0],     # Archeologist
    [14, 0, 0, 10, 2, 0],    # Barbarian
    [14, 0, 0, 8, 2, 0],     # Caveman
    [11, 0, 0, 8, 1, 0],     # Healer
    [14, 0, 0, 8, 2, 0],     # Knight
    [12, 0, 0, 8, 1, 0],     # Monk
    [12, 0, 0, 8, 1, 0],     # Priest
    [13, 0, 0, 6, 1, 0],     # Ranger
    [10, 0, 0, 8, 1, 0],     # Rogue
    [13, 0, 0, 8, 1, 0],     # Samurai
    [8, 0, 0, 8, 0, 0],      # Tourist
    [14, 0, 0, 8, 2, 0],     # Valkyrie
    [10, 0, 0, 8, 1, 0],     # Wizard
], dtype=jnp.int32)


# Power/Energy advancement per role: [infix, inrnd, lofix, lornd, hifix, hirnd]
ROLE_PW_ADV = jnp.array([
    [1, 0, 0, 1, 0, 1],      # Archeologist
    [1, 0, 0, 1, 0, 1],      # Barbarian
    [1, 0, 0, 1, 0, 1],      # Caveman
    [1, 4, 0, 1, 0, 2],      # Healer
    [1, 4, 0, 1, 0, 2],      # Knight
    [2, 2, 0, 2, 0, 2],      # Monk
    [4, 3, 0, 2, 0, 2],      # Priest
    [1, 0, 0, 1, 0, 1],      # Ranger
    [1, 0, 0, 1, 0, 1],      # Rogue
    [1, 0, 0, 1, 0, 1],      # Samurai
    [1, 0, 0, 1, 0, 1],      # Tourist
    [1, 0, 0, 1, 0, 1],      # Valkyrie
    [4, 3, 0, 2, 0, 3],      # Wizard
], dtype=jnp.int32)


# Level threshold where HP/PW advancement switches from lo to hi tier
ROLE_XLEV = jnp.array([14, 10, 10, 20, 10, 10, 10, 12, 11, 11, 14, 10, 12], dtype=jnp.int32)

# Energy modifier percentage per role (applied to power gain)
# Priest/Wizard=200, Healer/Knight=150, Barbarian/Valkyrie=75, others=100
ROLE_ENERMOD = jnp.array([100, 75, 100, 150, 150, 100, 200, 100, 100, 100, 100, 75, 200], dtype=jnp.int32)


# ============================================================================
# Race data — from NetHack 3.7 role.c races[]
# ============================================================================

# Max stat values per race: [max_STR, max_INT, max_WIS, max_DEX, max_CON, max_CHA]
# Note: STR caps use our linear encoding (18=18, 18/50=23, 18/100=24)
RACE_STAT_CAPS = jnp.array([
    [18, 18, 18, 18, 18, 18],  # Human (STR 18/100 -> handled by RACE_STR_MAX)
    [18, 20, 20, 18, 16, 18],  # Elf
    [18, 16, 16, 20, 20, 16],  # Dwarf (STR 18/100)
    [18, 19, 18, 18, 18, 18],  # Gnome (STR 18/50)
    [18, 16, 16, 18, 18, 16],  # Orc (STR 18/50)
], dtype=jnp.int32)

# Maximum exceptional STR per race (in our linear encoding: 18/100=24, 18/50=23)
RACE_STR_MAX = jnp.array([24, 18, 24, 23, 23], dtype=jnp.int32)

# Race HP advancement: [infix, inrnd, lofix, lornd, hifix, hirnd]
RACE_HP_ADV = jnp.array([
    [2, 0, 0, 2, 1, 0],      # Human
    [1, 0, 0, 1, 1, 0],      # Elf
    [4, 0, 0, 3, 2, 0],      # Dwarf
    [1, 0, 0, 1, 0, 0],      # Gnome
    [1, 0, 0, 1, 0, 0],      # Orc
], dtype=jnp.int32)

# Race Power advancement: [infix, inrnd, lofix, lornd, hifix, hirnd]
RACE_PW_ADV = jnp.array([
    [1, 0, 0, 2, 0, 2],      # Human
    [2, 0, 0, 3, 0, 3],      # Elf
    [0, 0, 0, 0, 0, 0],      # Dwarf
    [2, 0, 0, 2, 0, 2],      # Gnome
    [1, 0, 0, 1, 0, 1],      # Orc
], dtype=jnp.int32)


# ============================================================================
# Combat bonus lookup tables — from NetHack 3.7 weapon.c
# ============================================================================

# STR to-hit bonus (abon() in weapon.c:950-984)
# Index by STR value (0-25). Values < 3 are unused but included for safe indexing.
# Indices 19-22 (18/01-18/50) get +1, only 23+ (18/51+) gets +2
ABON_STR = jnp.array([
    # 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
     -2, -2, -2, -2, -2, -2, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    # 17  18  19  20  21  22  23  24  25
      1,  1,  1,  1,  1,  1,  2,  3,  3,
], dtype=jnp.int32)

# DEX to-hit modifier (abon() in weapon.c:950-984)
# Index by DEX value (0-25).
ABON_DEX = jnp.array([
    # 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25
     -3, -3, -3, -3, -2, -2, -1, -1,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
], dtype=jnp.int32)

# STR damage bonus (dbon() in weapon.c:988-1011)
# Index by STR value (0-25).
DBON_STR = jnp.array([
    # 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
     -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,
    # 17  18  19  20  21  22  23  24  25
      1,  2,  3,  3,  3,  4,  5,  6,  6,
], dtype=jnp.int32)

# CON bonus to HP per level (newhp() in attrib.c:1076-1139)
# Index by CON value (0-25).
CON_HP_BONUS = jnp.array([
    # 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25
     -2, -2, -2, -2, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  2,  3,  4,  4,  4,  4,  4,  4,  4,
], dtype=jnp.int32)


# ============================================================================
# Monk martial arts damage scaling — from NetHack 3.7 uhitm.c
# ============================================================================

# Damage die sides by level bracket: L1-4=d4, L5-8=d6, L9-12=d8, L13-16=d10, L17+=d12
# bracket = min((level - 1) // 4, 4)
MONK_MARTIAL_SIDES = jnp.array([4, 6, 8, 10, 12], dtype=jnp.int32)

# Martial arts skill bonus (Basic bare-handed combat)
MONK_MARTIAL_BONUS = 3


# ============================================================================
# Intrinsic abilities — bitmask constants
# ============================================================================

INTRINSIC_POISON_RES = 1 << 0
INTRINSIC_FIRE_RES = 1 << 1
INTRINSIC_COLD_RES = 1 << 2
INTRINSIC_SLEEP_RES = 1 << 3
INTRINSIC_SHOCK_RES = 1 << 4
INTRINSIC_SEE_INVISIBLE = 1 << 5
INTRINSIC_STEALTH = 1 << 6
INTRINSIC_FAST = 1 << 7
INTRINSIC_WARNING = 1 << 8
INTRINSIC_SEARCH = 1 << 9
INTRINSIC_TELEPORT_CONTROL = 1 << 10
INTRINSIC_INFRAVISION = 1 << 11

NUM_INTRINSICS = 12


# Cumulative intrinsic bitmask per role at each level (0-30).
# ROLE_INTRINSICS[role_id, level] = bitmask of all intrinsics gained by that level.
# Level 0 is unused (placeholder), levels 1-30 are active.
def _build_role_intrinsics():
    """Build cumulative role intrinsic tables from NetHack 3.7 attrib.c."""
    import numpy as np
    table = np.zeros((13, 31), dtype=np.int32)

    # Per-role intrinsic gain levels (from attrib.c role_abil arrays):
    role_gains = {
        0: [(1, INTRINSIC_SEARCH), (5, INTRINSIC_STEALTH), (10, INTRINSIC_FAST)],  # Archeologist
        1: [(1, INTRINSIC_POISON_RES), (7, INTRINSIC_FAST), (15, INTRINSIC_STEALTH)],  # Barbarian
        2: [(7, INTRINSIC_FAST), (15, INTRINSIC_WARNING)],  # Caveman
        3: [(1, INTRINSIC_POISON_RES), (15, INTRINSIC_WARNING)],  # Healer
        4: [(7, INTRINSIC_FAST)],  # Knight
        5: [(1, INTRINSIC_FAST | INTRINSIC_SLEEP_RES | INTRINSIC_SEE_INVISIBLE),  # Monk L1
            (3, INTRINSIC_POISON_RES), (5, INTRINSIC_STEALTH),
            (7, INTRINSIC_WARNING), (9, INTRINSIC_SEARCH),
            (11, INTRINSIC_FIRE_RES), (13, INTRINSIC_COLD_RES),
            (15, INTRINSIC_SHOCK_RES), (17, INTRINSIC_TELEPORT_CONTROL)],
        6: [(15, INTRINSIC_WARNING), (20, INTRINSIC_FIRE_RES)],  # Priest
        7: [(1, INTRINSIC_SEARCH), (7, INTRINSIC_STEALTH), (15, INTRINSIC_SEE_INVISIBLE)],  # Ranger
        8: [(1, INTRINSIC_STEALTH), (10, INTRINSIC_SEARCH)],  # Rogue
        9: [(1, INTRINSIC_FAST), (15, INTRINSIC_STEALTH)],  # Samurai
        10: [(10, INTRINSIC_SEARCH), (20, INTRINSIC_POISON_RES)],  # Tourist
        11: [(1, INTRINSIC_COLD_RES), (4, INTRINSIC_STEALTH), (7, INTRINSIC_FAST)],  # Valkyrie
        12: [(15, INTRINSIC_WARNING), (17, INTRINSIC_TELEPORT_CONTROL)],  # Wizard
    }

    for role_id, gains in role_gains.items():
        cumulative = 0
        for level in range(31):
            for gain_level, bits in gains:
                if level >= gain_level:
                    cumulative |= bits
            table[role_id, level] = cumulative

    return jnp.array(table, dtype=jnp.int32)

ROLE_INTRINSICS = _build_role_intrinsics()

# Cumulative racial intrinsic bitmask per race at each level (0-30).
def _build_race_intrinsics():
    """Build cumulative race intrinsic tables from NetHack 3.7 attrib.c."""
    import numpy as np
    table = np.zeros((5, 31), dtype=np.int32)

    race_gains = {
        0: [],  # Human: no racial intrinsics
        1: [(1, INTRINSIC_INFRAVISION), (4, INTRINSIC_SLEEP_RES)],  # Elf
        2: [(1, INTRINSIC_INFRAVISION)],  # Dwarf
        3: [(1, INTRINSIC_INFRAVISION)],  # Gnome
        4: [(1, INTRINSIC_INFRAVISION | INTRINSIC_POISON_RES)],  # Orc
    }

    for race_id, gains in race_gains.items():
        cumulative = 0
        for level in range(31):
            for gain_level, bits in gains:
                if level >= gain_level:
                    cumulative |= bits
            table[race_id, level] = cumulative

    return jnp.array(table, dtype=jnp.int32)

RACE_INTRINSICS = _build_race_intrinsics()


# ============================================================================
# Per-monster-type score — NetHack score = 4 * experience(monster)
# experience() = 1 + level^2 + various adjustments
# ============================================================================

# Simplified: score = 4 * (1 + level^2) for each monster type
# Uses MONSTER_STATS[:, 0] for level
MONSTER_XP_SCORE = 4 * (1 + MONSTER_STATS[:, 0] ** 2)


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
    256,  # MINOTAUR — brown bull-headed humanoid (glyph 250)
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
