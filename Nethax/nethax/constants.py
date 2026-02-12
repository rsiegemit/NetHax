import jax.numpy as jnp
from enum import IntEnum

FULL_MAP_VIEW = True
NORMAL_SPEED = 12
MAX_PLAYER_LEVEL = 30


# ============================================================================
# Tile types (dungeon terrain)
# ============================================================================
class TileType(IntEnum):
    VOID = 0           # Out of bounds / unexplored
    FLOOR = 1          # Room floor (.)
    CORRIDOR = 2       # Corridor (#)
    WALL = 3           # Wall (|-)
    CLOSED_DOOR = 4    # Closed door (+)
    OPEN_DOOR = 5      # Open door (|)
    STAIRCASE_UP = 6   # Upstairs (<)
    STAIRCASE_DOWN = 7 # Downstairs (>)
    WATER = 8          # Pool/moat (~)
    LAVA = 9           # Lava (~)
    ALTAR = 10         # Altar (_)
    FOUNTAIN = 11      # Fountain ({)
    TRAP = 12          # Known trap (^)
    HIDDEN_TRAP = 13   # Hidden trap (looks like floor)
    THRONE = 14        # Throne (\)
    GRAVE = 15         # Grave (|)
    SHOP_FLOOR = 16    # Shop tile


NUM_TILE_TYPES = len(TileType)

# Tiles that block movement
SOLID_TILES = jnp.array([
    TileType.VOID,
    TileType.WALL,
    TileType.CLOSED_DOOR,
], dtype=jnp.int32)

# Tiles that block line of sight
OPAQUE_TILES = jnp.array([
    TileType.VOID,
    TileType.WALL,
    TileType.CLOSED_DOOR,
], dtype=jnp.int32)


# ============================================================================
# Resistance bitmasks (for monster intrinsic resistances)
# ============================================================================
RES_FIRE    = 0x001
RES_COLD    = 0x002
RES_POISON  = 0x004
RES_SLEEP   = 0x008
RES_SHOCK   = 0x010
RES_DISINT  = 0x020
RES_ACID    = 0x040
RES_STONE   = 0x080
RES_DRAIN   = 0x100


# ============================================================================
# Conveyance bitmasks (intrinsics conveyed by eating corpse)
# ============================================================================
CONV_FIRE         = 0x001
CONV_COLD         = 0x002
CONV_POISON       = 0x004
CONV_SLEEP        = 0x008
CONV_SHOCK        = 0x010
CONV_TELEPATHY    = 0x020
CONV_SEE_INVIS    = 0x040
CONV_TELEPORT     = 0x080
CONV_TELEPORT_CTRL = 0x100
CONV_ACID         = 0x200
CONV_DISINT       = 0x400
CONV_DRAIN        = 0x800


# ============================================================================
# Monster flags bitmask
# ============================================================================
MF_REGEN     = 0x001
MF_FLY       = 0x002
MF_SWIM      = 0x004
MF_UNDEAD    = 0x008
MF_DEMON     = 0x010
MF_HOSTILE   = 0x020
MF_POISONOUS = 0x040
MF_ACIDIC    = 0x080
MF_STONING   = 0x100
MF_INVISIBLE = 0x200
MF_SEE_INVIS = 0x400
MF_GROUP     = 0x800


# ============================================================================
# Item categories
# ============================================================================
class ItemCategory(IntEnum):
    NONE = 0
    WEAPON = 1
    ARMOR = 2
    POTION = 3
    SCROLL = 4
    WAND = 5
    RING = 6
    AMULET = 7
    FOOD = 8
    CORPSE = 9
    GOLD = 10
    TOOL = 11
    GEM = 12


# ============================================================================
# Armor slots
# ============================================================================
class ArmorSlot(IntEnum):
    SUIT = 0
    SHIRT = 1
    CLOAK = 2
    SHIELD = 3
    HELM = 4
    GLOVES = 5
    BOOTS = 6


# ============================================================================
# Weapon types
# ============================================================================
class WeaponType(IntEnum):
    NONE = 0
    DAGGER = 1
    ELVEN_DAGGER = 2
    ORCISH_DAGGER = 3
    SILVER_DAGGER = 4
    KNIFE = 5
    CRYSKNIFE = 6
    SHORT_SWORD = 7
    ELVEN_SHORT_SWORD = 8
    LONG_SWORD = 9
    KATANA = 10
    TWO_HANDED_SWORD = 11
    SCIMITAR = 12
    SILVER_SABER = 13
    AXE = 14
    BATTLE_AXE = 15
    SPEAR = 16
    ELVEN_SPEAR = 17
    JAVELIN = 18
    TRIDENT = 19
    LANCE = 20
    MACE = 21
    WAR_HAMMER = 22
    MORNING_STAR = 23
    FLAIL = 24
    CLUB = 25
    QUARTERSTAFF = 26
    BULLWHIP = 27
    BOW = 28
    ELVEN_BOW = 29
    CROSSBOW = 30
    ARROW = 31
    ELVEN_ARROW = 32
    CROSSBOW_BOLT = 33
    DART = 34
    SHURIKEN = 35


NUM_WEAPON_TYPES = len(WeaponType)

# Weapon stats: [sdam, ldam, hit_bonus, weight, cost, two_handed]
# sdam/ldam = die size for damage vs small/large monsters (1dN)
# Data from NetHack 3.7 include/objects.h
WEAPON_STATS = jnp.array([
    #  sdam  ldam  hit  weight  cost  2H
    [   0,    0,    0,    0,      0,   0],  # NONE
    [   4,    3,    2,   10,      4,   0],  # DAGGER
    [   5,    3,    2,   10,      4,   0],  # ELVEN_DAGGER
    [   3,    3,    2,   10,      4,   0],  # ORCISH_DAGGER
    [   4,    3,    2,   12,     40,   0],  # SILVER_DAGGER
    [   3,    2,    0,    5,      4,   0],  # KNIFE
    [  10,   10,    3,   20,    100,   0],  # CRYSKNIFE
    [   6,    8,    0,   30,     10,   0],  # SHORT_SWORD
    [   8,    8,    0,   30,     10,   0],  # ELVEN_SHORT_SWORD
    [   8,   12,    0,   40,     15,   0],  # LONG_SWORD
    [  10,   12,    1,   40,     80,   0],  # KATANA
    [  12,    6,    0,  150,     50,   1],  # TWO_HANDED_SWORD (+2d6 ldam)
    [   8,    8,    0,   40,     15,   0],  # SCIMITAR
    [   8,    8,    0,   40,     75,   0],  # SILVER_SABER
    [   6,    4,    0,   60,      8,   0],  # AXE
    [   8,    6,    0,  120,     40,   1],  # BATTLE_AXE
    [   6,    8,    0,   30,      3,   0],  # SPEAR
    [   7,    8,    0,   30,      3,   0],  # ELVEN_SPEAR
    [   6,    6,    0,   20,      3,   0],  # JAVELIN
    [   6,    4,    0,   25,      5,   0],  # TRIDENT (+1 sdam, +2d4 ldam)
    [   6,    8,    0,  180,     10,   0],  # LANCE
    [   6,    6,    0,   30,      5,   0],  # MACE (+1 sdam)
    [   4,    4,    0,   50,      5,   0],  # WAR_HAMMER (+1 sdam)
    [   4,    6,    0,  120,     10,   0],  # MORNING_STAR (+d4 sdam, +1 ldam)
    [   6,    4,    0,   15,      4,   0],  # FLAIL (+1 sdam, +1d4 ldam)
    [   6,    3,    0,   30,      3,   0],  # CLUB
    [   6,    6,    0,   40,      5,   1],  # QUARTERSTAFF
    [   2,    1,    0,   20,      4,   0],  # BULLWHIP
    [   0,    0,    0,   30,     60,   0],  # BOW (launcher, no melee damage)
    [   0,    0,    0,   30,     60,   0],  # ELVEN_BOW (launcher)
    [   0,    0,    0,   50,     40,   0],  # CROSSBOW (launcher)
    [   6,    6,    0,    1,      2,   0],  # ARROW
    [   7,    6,    0,    1,      2,   0],  # ELVEN_ARROW
    [   4,    6,    0,    1,      2,   0],  # CROSSBOW_BOLT
    [   3,    2,    0,    1,      2,   0],  # DART
    [   8,    6,    2,    1,      5,   0],  # SHURIKEN
], dtype=jnp.int32)

# Backward-compatible alias: first two columns (sdam, ldam)
WEAPON_DAMAGE = WEAPON_STATS[:, :2]


# ============================================================================
# Armor types
# ============================================================================
class ArmorType(IntEnum):
    NONE = 0
    # Body armor (suit) - ordered roughly by AC (best to worst)
    GRAY_DRAGON_SCALE_MAIL = 1
    SILVER_DRAGON_SCALE_MAIL = 2
    RED_DRAGON_SCALE_MAIL = 3
    WHITE_DRAGON_SCALE_MAIL = 4
    BLUE_DRAGON_SCALE_MAIL = 5
    GREEN_DRAGON_SCALE_MAIL = 6
    BLACK_DRAGON_SCALE_MAIL = 7
    PLATE_MAIL = 8
    CRYSTAL_PLATE_MAIL = 9
    BRONZE_PLATE_MAIL = 10
    SPLINT_MAIL = 11
    BANDED_MAIL = 12
    DWARVISH_MITHRIL_COAT = 13
    ELVEN_MITHRIL_COAT = 14
    CHAIN_MAIL = 15
    SCALE_MAIL = 16
    STUDDED_LEATHER_ARMOR = 17
    RING_MAIL = 18
    LEATHER_ARMOR = 19
    LEATHER_JACKET = 20
    # Shields
    SMALL_SHIELD = 21
    LARGE_SHIELD = 22
    SHIELD_OF_REFLECTION = 23
    # Helms
    HELMET = 24
    DWARVISH_IRON_HELM = 25
    HELM_OF_BRILLIANCE = 26
    HELM_OF_TELEPATHY = 27
    # Cloaks
    CLOAK_OF_PROTECTION = 28
    ELVEN_CLOAK = 29
    CLOAK_OF_INVISIBILITY = 30
    CLOAK_OF_MAGIC_RESISTANCE = 31
    CLOAK_OF_DISPLACEMENT = 32
    OILSKIN_CLOAK = 33
    ROBE = 34
    LEATHER_CLOAK = 35
    # Gloves
    LEATHER_GLOVES = 36
    GAUNTLETS_OF_POWER = 37
    GAUNTLETS_OF_DEXTERITY = 38
    # Boots
    LOW_BOOTS = 39
    IRON_SHOES = 40
    HIGH_BOOTS = 41
    SPEED_BOOTS = 42
    WATER_WALKING_BOOTS = 43
    ELVEN_BOOTS = 44
    LEVITATION_BOOTS = 45


NUM_ARMOR_TYPES = len(ArmorType)

# Armor stats: [ac, mc, weight, cost, slot]
# ac = effective AC value (lower = better, NetHack convention)
# mc = magic cancellation level (0-3)
# slot uses ArmorSlot enum values
ARMOR_STATS = jnp.array([
    #  ac   mc  weight  cost   slot
    [  10,   0,    0,      0,   0],  # NONE
    [   1,   0,   40,   1200,   0],  # GRAY_DRAGON_SCALE_MAIL
    [   1,   0,   40,   1200,   0],  # SILVER_DRAGON_SCALE_MAIL
    [   1,   0,   40,    900,   0],  # RED_DRAGON_SCALE_MAIL
    [   1,   0,   40,    900,   0],  # WHITE_DRAGON_SCALE_MAIL
    [   1,   0,   40,    900,   0],  # BLUE_DRAGON_SCALE_MAIL
    [   1,   0,   40,    900,   0],  # GREEN_DRAGON_SCALE_MAIL
    [   1,   0,   40,   1200,   0],  # BLACK_DRAGON_SCALE_MAIL
    [   3,   2,  450,    600,   0],  # PLATE_MAIL
    [   3,   2,  415,    820,   0],  # CRYSTAL_PLATE_MAIL
    [   4,   1,  450,    400,   0],  # BRONZE_PLATE_MAIL
    [   4,   1,  400,     80,   0],  # SPLINT_MAIL
    [   4,   1,  350,     90,   0],  # BANDED_MAIL
    [   4,   2,  150,    240,   0],  # DWARVISH_MITHRIL_COAT
    [   5,   2,  150,    240,   0],  # ELVEN_MITHRIL_COAT
    [   5,   1,  300,     75,   0],  # CHAIN_MAIL
    [   6,   1,  250,     45,   0],  # SCALE_MAIL
    [   7,   1,  200,     15,   0],  # STUDDED_LEATHER_ARMOR
    [   7,   1,  250,    100,   0],  # RING_MAIL
    [   8,   1,  150,      5,   0],  # LEATHER_ARMOR
    [   9,   0,   30,     10,   0],  # LEATHER_JACKET
    # Shields
    [   9,   0,   30,      3,   3],  # SMALL_SHIELD
    [   8,   0,  100,     10,   3],  # LARGE_SHIELD
    [   8,   0,   50,     50,   3],  # SHIELD_OF_REFLECTION
    # Helms
    [   9,   0,   30,     10,   4],  # HELMET
    [   8,   0,   40,     20,   4],  # DWARVISH_IRON_HELM
    [   9,   0,   40,     50,   4],  # HELM_OF_BRILLIANCE
    [   9,   0,   50,     50,   4],  # HELM_OF_TELEPATHY
    # Cloaks
    [   7,   3,   10,     50,   2],  # CLOAK_OF_PROTECTION
    [   9,   1,   10,     60,   2],  # ELVEN_CLOAK
    [   9,   1,   10,     60,   2],  # CLOAK_OF_INVISIBILITY
    [   9,   1,   10,     60,   2],  # CLOAK_OF_MAGIC_RESISTANCE
    [   9,   1,   10,     50,   2],  # CLOAK_OF_DISPLACEMENT
    [   9,   2,   10,     50,   2],  # OILSKIN_CLOAK
    [   8,   2,   15,     50,   2],  # ROBE
    [   9,   1,   15,     40,   2],  # LEATHER_CLOAK
    # Gloves
    [   9,   0,   10,      8,   5],  # LEATHER_GLOVES
    [   9,   0,   30,     50,   5],  # GAUNTLETS_OF_POWER
    [   9,   0,   10,     50,   5],  # GAUNTLETS_OF_DEXTERITY
    # Boots
    [   9,   0,   10,      8,   6],  # LOW_BOOTS
    [   8,   0,   50,     16,   6],  # IRON_SHOES
    [   8,   0,   20,     12,   6],  # HIGH_BOOTS
    [   9,   0,   20,     50,   6],  # SPEED_BOOTS
    [   9,   0,   15,     50,   6],  # WATER_WALKING_BOOTS
    [   9,   0,   15,      8,   6],  # ELVEN_BOOTS
    [   9,   0,   15,     30,   6],  # LEVITATION_BOOTS
], dtype=jnp.int32)

# Backward-compatible alias: first column (ac)
ARMOR_AC = ARMOR_STATS[:, 0]


# ============================================================================
# Potion types (appearance randomized per run)
# ============================================================================
class PotionType(IntEnum):
    HEALING = 0
    EXTRA_HEALING = 1
    FULL_HEALING = 2
    GAIN_ABILITY = 3
    RESTORE_ABILITY = 4
    CONFUSION = 5
    BLINDNESS = 6
    PARALYSIS = 7
    SPEED = 8
    LEVITATION = 9
    INVISIBILITY = 10
    SEE_INVISIBLE = 11
    MONSTER_DETECTION = 12
    OBJECT_DETECTION = 13
    SLEEPING = 14
    GAIN_LEVEL = 15
    GAIN_ENERGY = 16
    POLYMORPH = 17
    SICKNESS = 18
    ACID = 19
    WATER = 20
    BOOZE = 21
    HALLUCINATION = 22


NUM_POTION_TYPES = len(PotionType)

# Potion appearance names (shuffled per episode to randomize identification)
POTION_APPEARANCES = [
    "purple-red",       # HEALING
    "puce",             # EXTRA_HEALING
    "black",            # FULL_HEALING
    "ruby",             # GAIN_ABILITY
    "pink",             # RESTORE_ABILITY
    "orange",           # CONFUSION
    "yellow",           # BLINDNESS
    "emerald",          # PARALYSIS
    "dark green",       # SPEED
    "cyan",             # LEVITATION
    "brilliant blue",   # INVISIBILITY
    "magenta",          # SEE_INVISIBLE
    "bubbly",           # MONSTER_DETECTION
    "smoky",            # OBJECT_DETECTION
    "effervescent",     # SLEEPING
    "milky",            # GAIN_LEVEL
    "cloudy",           # GAIN_ENERGY
    "golden",           # POLYMORPH
    "fizzy",            # SICKNESS
    "white",            # ACID
    "clear",            # WATER
    "brown",            # BOOZE
    "sky blue",         # HALLUCINATION
]

# Base cost in zorkmids for each potion type
POTION_COSTS = jnp.array([
     20,  # HEALING
    100,  # EXTRA_HEALING
    200,  # FULL_HEALING
    300,  # GAIN_ABILITY
    100,  # RESTORE_ABILITY
    100,  # CONFUSION
    150,  # BLINDNESS
    300,  # PARALYSIS
    200,  # SPEED
    200,  # LEVITATION
    150,  # INVISIBILITY
     50,  # SEE_INVISIBLE
    150,  # MONSTER_DETECTION
    150,  # OBJECT_DETECTION
    100,  # SLEEPING
    300,  # GAIN_LEVEL
    150,  # GAIN_ENERGY
    200,  # POLYMORPH
     50,  # SICKNESS
    250,  # ACID
    100,  # WATER
     50,  # BOOZE
    100,  # HALLUCINATION
], dtype=jnp.int32)

# Relative generation probability (higher = more common)
POTION_PROBABILITIES = jnp.array([
    115,  # HEALING
     45,  # EXTRA_HEALING
     10,  # FULL_HEALING
     40,  # GAIN_ABILITY
     40,  # RESTORE_ABILITY
     40,  # CONFUSION
     30,  # BLINDNESS
     40,  # PARALYSIS
     40,  # SPEED
     40,  # LEVITATION
     40,  # INVISIBILITY
     40,  # SEE_INVISIBLE
     40,  # MONSTER_DETECTION
     40,  # OBJECT_DETECTION
     40,  # SLEEPING
     20,  # GAIN_LEVEL
     40,  # GAIN_ENERGY
     10,  # POLYMORPH
     40,  # SICKNESS
     10,  # ACID
     80,  # WATER
     40,  # BOOZE
     30,  # HALLUCINATION
], dtype=jnp.int32)


# ============================================================================
# Scroll types (appearance/label randomized per run)
# ============================================================================
class ScrollType(IntEnum):
    IDENTIFY = 0
    ENCHANT_WEAPON = 1
    ENCHANT_ARMOR = 2
    REMOVE_CURSE = 3
    TELEPORTATION = 4
    MAGIC_MAPPING = 5
    FIRE = 6
    LIGHT = 7
    CREATE_MONSTER = 8
    DESTROY_ARMOR = 9
    CONFUSE_MONSTER = 10
    SCARE_MONSTER = 11
    TAMING = 12
    GENOCIDE = 13
    CHARGING = 14
    BLANK = 15
    GOLD_DETECTION = 16
    FOOD_DETECTION = 17
    PUNISHMENT = 18
    EARTH = 19
    AMNESIA = 20
    STINKING_CLOUD = 21


NUM_SCROLL_TYPES = len(ScrollType)

# Scroll label appearances (shuffled per episode)
SCROLL_APPEARANCES = [
    "KERNOD WEL",                # IDENTIFY
    "DAIYEN FOOELS",             # ENCHANT_WEAPON
    "ZELGO MER",                 # ENCHANT_ARMOR
    "PRATYAVAYAH",               # REMOVE_CURSE
    "VENZAR BORGAVVE",           # TELEPORTATION
    "ELAM EBOW",                 # MAGIC_MAPPING
    "ANDOVA BEGARIN",            # FIRE
    "VERR YED HORRE",            # LIGHT
    "LEP GEX VEN ZEA",           # CREATE_MONSTER
    "JUYED AWK YACC",            # DESTROY_ARMOR
    "NR 9",                      # CONFUSE_MONSTER
    "XIXAXA XOXAXA XUXAXA",     # SCARE_MONSTER
    "PRIRUTSENIE",               # TAMING
    "ELBIB YLOH",                # GENOCIDE
    "HACKEM MUCHE",              # CHARGING
    "unlabeled",                 # BLANK
    "THARR",                     # GOLD_DETECTION
    "YUM YUM",                   # FOOD_DETECTION
    "VE FORBRYDERNE",            # PUNISHMENT
    "KIRJE",                     # EARTH
    "DUAM XNAHT",                # AMNESIA
    "VELOX NEB",                 # STINKING_CLOUD
]

# Base cost in zorkmids for each scroll type
SCROLL_COSTS = jnp.array([
     20,  # IDENTIFY
     60,  # ENCHANT_WEAPON
     80,  # ENCHANT_ARMOR
     80,  # REMOVE_CURSE
    100,  # TELEPORTATION
    100,  # MAGIC_MAPPING
    100,  # FIRE
     50,  # LIGHT
    200,  # CREATE_MONSTER
    100,  # DESTROY_ARMOR
    100,  # CONFUSE_MONSTER
    100,  # SCARE_MONSTER
    200,  # TAMING
    300,  # GENOCIDE
    300,  # CHARGING
     60,  # BLANK
    100,  # GOLD_DETECTION
    100,  # FOOD_DETECTION
    300,  # PUNISHMENT
    200,  # EARTH
    200,  # AMNESIA
    300,  # STINKING_CLOUD
], dtype=jnp.int32)

# Relative generation probability (higher = more common)
SCROLL_PROBABILITIES = jnp.array([
    180,  # IDENTIFY
     80,  # ENCHANT_WEAPON
     63,  # ENCHANT_ARMOR
     65,  # REMOVE_CURSE
     55,  # TELEPORTATION
     45,  # MAGIC_MAPPING
     30,  # FIRE
     90,  # LIGHT
     45,  # CREATE_MONSTER
     45,  # DESTROY_ARMOR
     53,  # CONFUSE_MONSTER
     35,  # SCARE_MONSTER
     15,  # TAMING
     15,  # GENOCIDE
     15,  # CHARGING
     28,  # BLANK
     33,  # GOLD_DETECTION
     25,  # FOOD_DETECTION
     15,  # PUNISHMENT
     18,  # EARTH
     35,  # AMNESIA
     15,  # STINKING_CLOUD
], dtype=jnp.int32)


# ============================================================================
# Wand types
# ============================================================================
class WandDirectionType(IntEnum):
    NODIR = 0       # Self-targeted, no direction needed
    IMMEDIATE = 1   # Affects target in chosen direction immediately
    RAY = 2         # Shoots a beam that can bounce


class WandType(IntEnum):
    # RAY wands
    DEATH = 0
    FIRE = 1
    COLD = 2
    SLEEP = 3
    MAGIC_MISSILE = 4
    DIGGING = 5
    LIGHTNING = 6
    # IMMEDIATE wands
    NOTHING = 7
    STRIKING = 8
    MAKE_INVISIBLE = 9
    SLOW_MONSTER = 10
    SPEED_MONSTER = 11
    UNDEAD_TURNING = 12
    POLYMORPH = 13
    CANCELLATION = 14
    TELEPORTATION = 15
    OPENING = 16
    LOCKING = 17
    PROBING = 18
    # NODIR wands
    LIGHT = 19
    SECRET_DOOR_DETECTION = 20
    CREATE_MONSTER = 21
    WISHING = 22


NUM_WAND_TYPES = len(WandType)

# Wand appearance names (shuffled per episode)
WAND_APPEARANCES = [
    "long",         # DEATH
    "hexagonal",    # FIRE
    "short",        # COLD
    "runed",        # SLEEP
    "steel",        # MAGIC_MISSILE
    "iron",         # DIGGING
    "curved",       # LIGHTNING
    "oak",          # NOTHING
    "ebony",        # STRIKING
    "marble",       # MAKE_INVISIBLE
    "tin",          # SLOW_MONSTER
    "brass",        # SPEED_MONSTER
    "copper",       # UNDEAD_TURNING
    "silver",       # POLYMORPH
    "platinum",     # CANCELLATION
    "iridium",      # TELEPORTATION
    "zinc",         # OPENING
    "aluminum",     # LOCKING
    "uranium",      # PROBING
    "glass",        # LIGHT
    "balsa",        # SECRET_DOOR_DETECTION
    "maple",        # CREATE_MONSTER
    "pine",         # WISHING
]

# Wand stats: [direction_type, base_charges_min, base_charges_max, cost]
WAND_STATS = jnp.array([
    #  dir  min  max  cost
    [   2,   4,   8,  500],  # DEATH (ray)
    [   2,   4,   8,  175],  # FIRE (ray)
    [   2,   4,   8,  175],  # COLD (ray)
    [   2,   4,   8,  175],  # SLEEP (ray)
    [   2,   4,   8,  150],  # MAGIC_MISSILE (ray)
    [   2,   4,   8,  150],  # DIGGING (ray)
    [   2,   4,   8,  175],  # LIGHTNING (ray)
    [   1,   4,   8,  100],  # NOTHING (immediate)
    [   1,   4,   8,  150],  # STRIKING (immediate)
    [   1,   4,   8,  150],  # MAKE_INVISIBLE (immediate)
    [   1,   4,   8,  150],  # SLOW_MONSTER (immediate)
    [   1,   4,   8,  150],  # SPEED_MONSTER (immediate)
    [   1,   4,   8,  150],  # UNDEAD_TURNING (immediate)
    [   1,   4,   8,  200],  # POLYMORPH (immediate)
    [   1,   4,   8,  200],  # CANCELLATION (immediate)
    [   1,   4,   8,  200],  # TELEPORTATION (immediate)
    [   1,   4,   8,  150],  # OPENING (immediate)
    [   1,   4,   8,  150],  # LOCKING (immediate)
    [   1,   4,   8,  150],  # PROBING (immediate)
    [   0,  11,  15,  100],  # LIGHT (nodir)
    [   0,  11,  15,  150],  # SECRET_DOOR_DETECTION (nodir)
    [   0,  11,  15,  200],  # CREATE_MONSTER (nodir)
    [   0,   1,   1,  500],  # WISHING (nodir, always 1 charge)
], dtype=jnp.int32)

# Relative generation probability (higher = more common)
WAND_PROBABILITIES = jnp.array([
      5,  # DEATH
     40,  # FIRE
     40,  # COLD
     50,  # SLEEP
     50,  # MAGIC_MISSILE
     55,  # DIGGING
     40,  # LIGHTNING
     25,  # NOTHING
     75,  # STRIKING
     45,  # MAKE_INVISIBLE
     50,  # SLOW_MONSTER
     50,  # SPEED_MONSTER
     50,  # UNDEAD_TURNING
     45,  # POLYMORPH
     45,  # CANCELLATION
     45,  # TELEPORTATION
     25,  # OPENING
     25,  # LOCKING
     30,  # PROBING
     95,  # LIGHT
     50,  # SECRET_DOOR_DETECTION
     45,  # CREATE_MONSTER
      5,  # WISHING
], dtype=jnp.int32)


# ============================================================================
# Ring types
# ============================================================================
class RingType(IntEnum):
    ADORNMENT = 0
    GAIN_STRENGTH = 1
    GAIN_CONSTITUTION = 2
    INCREASE_ACCURACY = 3
    INCREASE_DAMAGE = 4
    PROTECTION = 5
    REGENERATION = 6
    SEARCHING = 7
    STEALTH = 8
    SUSTAIN_ABILITY = 9
    LEVITATION = 10
    HUNGER = 11
    AGGRAVATE_MONSTER = 12
    CONFLICT = 13
    WARNING = 14
    POISON_RESISTANCE = 15
    FIRE_RESISTANCE = 16
    COLD_RESISTANCE = 17
    SHOCK_RESISTANCE = 18
    FREE_ACTION = 19
    SLOW_DIGESTION = 20
    TELEPORTATION = 21
    TELEPORT_CONTROL = 22
    POLYMORPH = 23
    POLYMORPH_CONTROL = 24
    INVISIBILITY = 25
    SEE_INVISIBLE = 26


NUM_RING_TYPES = len(RingType)

# Ring appearance names (shuffled per episode)
RING_APPEARANCES = [
    "wooden",       # ADORNMENT
    "granite",      # GAIN_STRENGTH
    "opal",         # GAIN_CONSTITUTION
    "clay",         # INCREASE_ACCURACY
    "coral",        # INCREASE_DAMAGE
    "black onyx",   # PROTECTION
    "moonstone",    # REGENERATION
    "tiger eye",    # SEARCHING
    "jade",         # STEALTH
    "bronze",       # SUSTAIN_ABILITY
    "agate",        # LEVITATION
    "topaz",        # HUNGER
    "sapphire",     # AGGRAVATE_MONSTER
    "ruby",         # CONFLICT
    "diamond",      # WARNING
    "pearl",        # POISON_RESISTANCE
    "iron",         # FIRE_RESISTANCE
    "brass",        # COLD_RESISTANCE
    "copper",       # SHOCK_RESISTANCE
    "twisted",      # FREE_ACTION
    "steel",        # SLOW_DIGESTION
    "silver",       # TELEPORTATION
    "gold",         # TELEPORT_CONTROL
    "ivory",        # POLYMORPH
    "emerald",      # POLYMORPH_CONTROL
    "wire",         # INVISIBILITY
    "engagement",   # SEE_INVISIBLE
]

# Base cost in zorkmids for each ring type
RING_COSTS = jnp.array([
    100,  # ADORNMENT
    150,  # GAIN_STRENGTH
    150,  # GAIN_CONSTITUTION
    150,  # INCREASE_ACCURACY
    150,  # INCREASE_DAMAGE
    100,  # PROTECTION
    200,  # REGENERATION
    200,  # SEARCHING
    100,  # STEALTH
    100,  # SUSTAIN_ABILITY
    200,  # LEVITATION
    100,  # HUNGER
    150,  # AGGRAVATE_MONSTER
    300,  # CONFLICT
    100,  # WARNING
    150,  # POISON_RESISTANCE
    200,  # FIRE_RESISTANCE
    150,  # COLD_RESISTANCE
    150,  # SHOCK_RESISTANCE
    200,  # FREE_ACTION
    200,  # SLOW_DIGESTION
    200,  # TELEPORTATION
    300,  # TELEPORT_CONTROL
    300,  # POLYMORPH
    300,  # POLYMORPH_CONTROL
    150,  # INVISIBILITY
    150,  # SEE_INVISIBLE
], dtype=jnp.int32)

# Whether each ring type is enchantable (+/- levels): 1=yes, 0=no
RING_ENCHANTABLE = jnp.array([
    1,  # ADORNMENT
    1,  # GAIN_STRENGTH
    1,  # GAIN_CONSTITUTION
    1,  # INCREASE_ACCURACY
    1,  # INCREASE_DAMAGE
    1,  # PROTECTION
    0,  # REGENERATION
    0,  # SEARCHING
    0,  # STEALTH
    0,  # SUSTAIN_ABILITY
    0,  # LEVITATION
    0,  # HUNGER
    0,  # AGGRAVATE_MONSTER
    0,  # CONFLICT
    0,  # WARNING
    0,  # POISON_RESISTANCE
    0,  # FIRE_RESISTANCE
    0,  # COLD_RESISTANCE
    0,  # SHOCK_RESISTANCE
    0,  # FREE_ACTION
    0,  # SLOW_DIGESTION
    0,  # TELEPORTATION
    0,  # TELEPORT_CONTROL
    0,  # POLYMORPH
    0,  # POLYMORPH_CONTROL
    0,  # INVISIBILITY
    0,  # SEE_INVISIBLE
], dtype=jnp.int32)


# ============================================================================
# Amulet types
# ============================================================================
class AmuletType(IntEnum):
    ESP = 0
    LIFE_SAVING = 1
    STRANGULATION = 2
    RESTFUL_SLEEP = 3
    VERSUS_POISON = 4
    CHANGE = 5
    UNCHANGING = 6
    REFLECTION = 7
    MAGICAL_BREATHING = 8
    GUARDING = 9
    FLYING = 10


NUM_AMULET_TYPES = len(AmuletType)

# Amulet appearance names (shuffled per episode)
AMULET_APPEARANCES = [
    "circular",      # ESP
    "spherical",     # LIFE_SAVING
    "oval",          # STRANGULATION
    "triangular",    # RESTFUL_SLEEP
    "pyramidal",     # VERSUS_POISON
    "square",        # CHANGE
    "concave",       # UNCHANGING
    "hexagonal",     # REFLECTION
    "octagonal",     # MAGICAL_BREATHING
    "perforated",    # GUARDING
    "cubical",       # FLYING
]

# Base cost in zorkmids for each amulet type (all 150 except Amulet of Yendor)
AMULET_COSTS = jnp.array([
    150,  # ESP
    150,  # LIFE_SAVING
    150,  # STRANGULATION
    150,  # RESTFUL_SLEEP
    150,  # VERSUS_POISON
    150,  # CHANGE
    150,  # UNCHANGING
    150,  # REFLECTION
    150,  # MAGICAL_BREATHING
    150,  # GUARDING
    150,  # FLYING
], dtype=jnp.int32)

# Relative generation probability (higher = more common)
AMULET_PROBABILITIES = jnp.array([
    120,  # ESP
     75,  # LIFE_SAVING
    115,  # STRANGULATION
    115,  # RESTFUL_SLEEP
    115,  # VERSUS_POISON
    115,  # CHANGE
     60,  # UNCHANGING
     75,  # REFLECTION
     75,  # MAGICAL_BREATHING
     75,  # GUARDING
     60,  # FLYING
], dtype=jnp.int32)


# ============================================================================
# Food types
# ============================================================================
class FoodType(IntEnum):
    FOOD_RATION = 0
    CRAM_RATION = 1
    LEMBAS_WAFER = 2
    APPLE = 3
    ORANGE = 4
    PEAR = 5
    MELON = 6
    BANANA = 7
    CARROT = 8
    SLIME_MOLD = 9
    FORTUNE_COOKIE = 10
    CANDY_BAR = 11
    CREAM_PIE = 12
    PANCAKE = 13
    EGG = 14
    CORPSE = 15         # Uses monster type for nutrition/effect
    TRIPE_RATION = 16
    TIN = 17


NUM_FOOD_TYPES = len(FoodType)

# Food stats: [nutrition, weight, eat_delay, probability]
# Data from NetHack 3.7 include/objects.h
FOOD_STATS = jnp.array([
    #  nutr  wt  delay  prob
    [ 800,   20,   5,   380],  # FOOD_RATION
    [ 600,   15,   3,    20],  # CRAM_RATION
    [ 800,    5,   2,    20],  # LEMBAS_WAFER
    [  50,    2,   1,    15],  # APPLE
    [  80,    2,   1,    10],  # ORANGE
    [  50,    2,   1,    10],  # PEAR
    [ 100,    5,   1,    10],  # MELON
    [  80,    2,   1,    10],  # BANANA
    [  50,    2,   1,    15],  # CARROT
    [ 250,    5,   1,    75],  # SLIME_MOLD
    [  40,    1,   1,    55],  # FORTUNE_COOKIE
    [ 100,    2,   1,    13],  # CANDY_BAR
    [ 100,   10,   1,    25],  # CREAM_PIE
    [ 200,    2,   2,    25],  # PANCAKE
    [  80,    1,   1,    85],  # EGG
    [   0,    0,   1,     0],  # CORPSE (varies by monster type)
    [ 200,   10,   2,   140],  # TRIPE_RATION
    [   0,   10,   0,    75],  # TIN (varies by contents)
], dtype=jnp.int32)

# Backward-compatible alias: first column (nutrition)
FOOD_NUTRITION = FOOD_STATS[:, 0]




# ============================================================================
# Monster types (expanded)
# ============================================================================
class MonsterType(IntEnum):
    NONE = 0

    # --- Early dungeon (depth 1-5) ---
    NEWT = 1
    JACKAL = 2
    KOBOLD = 3
    SEWER_RAT = 4
    GRID_BUG = 5
    LICHEN = 6
    GECKO = 7
    ACID_BLOB = 8
    GNOME = 9
    HOBBIT = 10
    KILLER_BEE = 11
    BAT = 12
    KITTEN = 13
    FLOATING_EYE = 14
    GIANT_ANT = 15
    SOLDIER_ANT = 16

    # --- Mid dungeon (depth 6-12) ---
    WOLF = 17
    WINTER_WOLF = 18
    COCKATRICE = 19
    ORC = 20
    DWARF = 21
    NYMPH = 22
    STALKER = 23
    TIGER = 24
    WRAITH = 25
    GELATINOUS_CUBE = 26
    MIND_FLAYER = 27
    HELL_HOUND_PUP = 28

    # --- Late dungeon (depth 13-20) ---
    TROLL = 29
    OGRE = 30
    LICH = 31
    MINOTAUR = 32
    VAMPIRE = 33
    VAMPIRE_LORD = 34
    JABBERWOCK = 35

    # --- Dragons ---
    RED_DRAGON = 36
    WHITE_DRAGON = 37
    BLUE_DRAGON = 38
    GREEN_DRAGON = 39
    BLACK_DRAGON = 40
    GRAY_DRAGON = 41
    SILVER_DRAGON = 42

    # --- Giants / Demons ---
    HILL_GIANT = 43
    FIRE_GIANT = 44
    FROST_GIANT = 45
    PIT_FIEND = 46
    BALROG = 47


NUM_MONSTER_TYPES = len(MonsterType)

# --------------------------------------------------------------------------
# MONSTER_STATS
# Columns: [level, speed, ac, mr, atk_dice, atk_sides, weight, nutrition, difficulty]
# Values sourced from NetHack 3.7 monst.c (see reference/monsters.md).
# --------------------------------------------------------------------------
MONSTER_STATS = jnp.array([
    # [lvl, spd, ac,  mr, dice, sides, weight, nutr, diff]
    [  0,   0,   0,   0,   0,   0,      0,     0,    0],  # NONE
    # --- Early (depth 1-5) ---
    [  0,   6,   8,   0,   1,   1,     10,    20,    1],  # NEWT
    [  0,  12,   7,   0,   1,   2,    300,   250,    1],  # JACKAL
    [  0,   6,  10,   0,   1,   4,    400,   200,    1],  # KOBOLD
    [  0,  12,   7,   0,   1,   3,     20,    12,    1],  # SEWER_RAT
    [  0,  12,   9,   0,   1,   1,     15,    10,    1],  # GRID_BUG
    [  0,   1,   9,   0,   0,   0,     20,   200,    1],  # LICHEN
    [  1,   6,   8,   0,   1,   3,     10,    20,    1],  # GECKO
    [  1,   3,   8,   0,   1,   8,     30,    30,    2],  # ACID_BLOB
    [  1,   6,  10,   4,   1,   6,    650,   100,    2],  # GNOME
    [  1,   9,  10,   0,   1,   6,    500,   200,    2],  # HOBBIT
    [  1,  18,  -1,   0,   1,   3,      1,     5,    2],  # KILLER_BEE
    [  2,  22,   8,   0,   1,   4,     20,    20,    2],  # BAT
    [  2,  18,   6,   0,   1,   6,    150,   150,    2],  # KITTEN
    [  2,   1,   9,  10,   0,   0,     10,    10,    2],  # FLOATING_EYE
    [  2,  18,   3,   0,   1,   4,     10,    10,    3],  # GIANT_ANT
    [  3,  18,   3,   0,   2,   4,     20,     5,    4],  # SOLDIER_ANT
    # --- Mid (depth 6-12) ---
    [  5,  12,   4,   0,   2,   4,    500,   250,    5],  # WOLF
    [  5,  12,   4,  20,   2,   6,    700,   300,    7],  # WINTER_WOLF
    [  5,   6,   6,  30,   1,   3,     30,    30,    8],  # COCKATRICE
    [  1,   9,  10,   0,   1,   8,    850,   150,    2],  # ORC
    [  2,   6,  10,  10,   1,   8,    900,   300,    3],  # DWARF
    [  3,  12,   9,  20,   0,   0,    600,   300,    6],  # NYMPH
    [  8,  12,   3,   0,   4,   4,    900,   400,    8],  # STALKER
    [  6,  12,   6,   0,   1,  10,    600,   300,    8],  # TIGER
    [  6,  12,   4,  15,   1,   6,      0,     0,    8],  # WRAITH
    [  6,   6,   8,   0,   2,   4,    600,   150,    6],  # GELATINOUS_CUBE
    [  9,  12,   5,  90,   2,   1,   1450,   400,   11],  # MIND_FLAYER
    [  7,  12,   4,  20,   2,   6,    200,   200,    8],  # HELL_HOUND_PUP
    # --- Late (depth 13-20) ---
    [  7,  12,   4,   0,   2,   6,    800,   350,    9],  # TROLL
    [  5,  10,   5,   0,   2,   5,   1600,   500,    7],  # OGRE
    [ 11,   6,   0,  30,   1,  10,   1200,     0,   13],  # LICH
    [ 15,  15,   6,   0,   3,  10,   1500,   700,   17],  # MINOTAUR
    [ 10,  12,   1,  25,   1,   6,   1450,   400,   12],  # VAMPIRE
    [ 12,  14,   0,  50,   1,   8,   1450,   400,   14],  # VAMPIRE_LORD
    [ 15,  12,  -2,  50,   2,  10,   1300,   600,   18],  # JABBERWOCK
    # --- Dragons ---
    [ 15,   9,  -1,  20,   6,   6,   4500,  1500,   18],  # RED_DRAGON
    [ 15,   9,  -1,  20,   4,   6,   4500,  1500,   18],  # WHITE_DRAGON
    [ 15,   9,  -1,  20,   4,   6,   4500,  1500,   18],  # BLUE_DRAGON
    [ 15,   9,  -1,  20,   4,   6,   4500,  1500,   18],  # GREEN_DRAGON
    [ 15,   9,  -1,  20,   4,  10,   4500,  1500,   18],  # BLACK_DRAGON
    [ 15,   9,  -1,  20,   4,   6,   4500,  1500,   18],  # GRAY_DRAGON
    [ 15,   9,  -1,  20,   4,   6,   4500,  1500,   18],  # SILVER_DRAGON
    # --- Giants / Demons ---
    [  8,  10,   6,   0,   2,   8,   2200,   750,    9],  # HILL_GIANT
    [  9,  12,   4,   5,   2,  10,   2250,   750,   11],  # FIRE_GIANT
    [ 10,  12,   3,  10,   2,  12,   2250,   750,   13],  # FROST_GIANT
    [ 13,   6,  -3,  65,   4,   2,   1500,   400,   16],  # PIT_FIEND
    [ 12,  15,  -2,  75,   2,   6,   1500,   400,   15],  # BALROG
], dtype=jnp.int32)

# Derived: max HP per monster type. NetHack uses level*8 with minimum of 4.
MONSTER_MAX_HP = jnp.maximum(MONSTER_STATS[:, 0] * 8, 4)


# --------------------------------------------------------------------------
# MONSTER_RESISTANCES  (bitmask per monster)
# --------------------------------------------------------------------------
MONSTER_RESISTANCES = jnp.array([
    0x000,  # NONE
    # --- Early ---
    0x000,                                  # NEWT
    0x000,                                  # JACKAL
    0x000,                                  # KOBOLD
    0x000,                                  # SEWER_RAT
    RES_SHOCK,                              # GRID_BUG  (shock)
    0x000,                                  # LICHEN
    0x000,                                  # GECKO
    RES_ACID | RES_STONE,                   # ACID_BLOB (acid, stone)
    0x000,                                  # GNOME
    0x000,                                  # HOBBIT
    0x000,                                  # KILLER_BEE
    0x000,                                  # BAT
    0x000,                                  # KITTEN
    0x000,                                  # FLOATING_EYE
    0x000,                                  # GIANT_ANT
    0x000,                                  # SOLDIER_ANT
    # --- Mid ---
    0x000,                                  # WOLF
    RES_COLD,                               # WINTER_WOLF (cold)
    RES_POISON | RES_STONE,                 # COCKATRICE (poison, stone)
    0x000,                                  # ORC
    0x000,                                  # DWARF
    0x000,                                  # NYMPH
    0x000,                                  # STALKER
    0x000,                                  # TIGER
    RES_SLEEP | RES_POISON | RES_DRAIN,     # WRAITH (sleep, poison, drain)
    RES_SHOCK | RES_POISON | RES_ACID | RES_COLD | RES_SLEEP | RES_STONE,  # GELATINOUS_CUBE
    0x000,                                  # MIND_FLAYER
    RES_FIRE,                               # HELL_HOUND_PUP (fire)
    # --- Late ---
    0x000,                                  # TROLL
    0x000,                                  # OGRE
    RES_COLD | RES_SLEEP | RES_POISON,      # LICH (cold, sleep, poison)
    0x000,                                  # MINOTAUR
    RES_SLEEP | RES_POISON | RES_DRAIN,     # VAMPIRE (sleep, poison, drain)
    RES_SLEEP | RES_POISON | RES_DRAIN,     # VAMPIRE_LORD (sleep, poison, drain)
    0x000,                                  # JABBERWOCK
    # --- Dragons ---
    RES_FIRE,                               # RED_DRAGON
    RES_COLD,                               # WHITE_DRAGON
    RES_SHOCK,                              # BLUE_DRAGON
    RES_POISON,                             # GREEN_DRAGON
    RES_DISINT,                             # BLACK_DRAGON
    0x000,                                  # GRAY_DRAGON
    0x000,                                  # SILVER_DRAGON
    # --- Giants / Demons ---
    0x000,                                  # HILL_GIANT
    RES_FIRE,                               # FIRE_GIANT
    RES_COLD,                               # FROST_GIANT
    RES_FIRE | RES_POISON,                  # PIT_FIEND (fire, poison)
    RES_FIRE | RES_POISON,                  # BALROG (fire, poison)
], dtype=jnp.int32)


# --------------------------------------------------------------------------
# MONSTER_CONVEYANCES  (intrinsics granted by eating corpse, bitmask)
# --------------------------------------------------------------------------
MONSTER_CONVEYANCES = jnp.array([
    0x000,  # NONE
    # --- Early ---
    0x000,              # NEWT
    0x000,              # JACKAL
    0x000,              # KOBOLD
    0x000,              # SEWER_RAT
    0x000,              # GRID_BUG
    0x000,              # LICHEN
    0x000,              # GECKO
    CONV_ACID,          # ACID_BLOB
    0x000,              # GNOME
    0x000,              # HOBBIT
    CONV_POISON,        # KILLER_BEE
    0x000,              # BAT
    0x000,              # KITTEN
    CONV_TELEPATHY,     # FLOATING_EYE
    0x000,              # GIANT_ANT
    CONV_POISON,        # SOLDIER_ANT
    # --- Mid ---
    0x000,              # WOLF
    CONV_COLD,          # WINTER_WOLF
    CONV_POISON,        # COCKATRICE  (stone effect handled in CORPSE_EFFECTS)
    0x000,              # ORC
    0x000,              # DWARF
    CONV_TELEPORT,      # NYMPH
    CONV_SEE_INVIS,     # STALKER
    0x000,              # TIGER
    CONV_DRAIN,         # WRAITH  (level gain handled in CORPSE_EFFECTS)
    0x000,              # GELATINOUS_CUBE
    CONV_TELEPATHY,     # MIND_FLAYER
    CONV_FIRE,          # HELL_HOUND_PUP
    # --- Late ---
    0x000,              # TROLL
    0x000,              # OGRE
    0x000,              # LICH  (no corpse)
    0x000,              # MINOTAUR
    CONV_DRAIN,         # VAMPIRE
    CONV_DRAIN,         # VAMPIRE_LORD
    0x000,              # JABBERWOCK
    # --- Dragons ---
    CONV_FIRE,          # RED_DRAGON
    CONV_COLD,          # WHITE_DRAGON
    CONV_SHOCK,         # BLUE_DRAGON
    CONV_POISON,        # GREEN_DRAGON
    CONV_DISINT,        # BLACK_DRAGON
    0x000,              # GRAY_DRAGON
    0x000,              # SILVER_DRAGON
    # --- Giants / Demons ---
    0x000,              # HILL_GIANT
    CONV_FIRE,          # FIRE_GIANT
    CONV_COLD,          # FROST_GIANT
    0x000,              # PIT_FIEND  (no corpse)
    0x000,              # BALROG  (no corpse)
], dtype=jnp.int32)


# --------------------------------------------------------------------------
# MONSTER_FLAGS  (behaviour bitmask)
# --------------------------------------------------------------------------
MONSTER_FLAGS = jnp.array([
    0x000,  # NONE
    # --- Early ---
    0x000,                                  # NEWT
    MF_HOSTILE | MF_GROUP,                  # JACKAL (hostile, pack)
    MF_HOSTILE,                             # KOBOLD (hostile)
    MF_HOSTILE | MF_GROUP,                  # SEWER_RAT (hostile, group)
    MF_HOSTILE,                             # GRID_BUG (hostile)
    0x000,                                  # LICHEN
    0x000,                                  # GECKO
    MF_ACIDIC,                              # ACID_BLOB (acidic)
    0x000,                                  # GNOME
    0x000,                                  # HOBBIT
    MF_FLY | MF_HOSTILE | MF_POISONOUS | MF_GROUP,  # KILLER_BEE
    MF_FLY,                                 # BAT (fly)
    0x000,                                  # KITTEN (domestic)
    0x000,                                  # FLOATING_EYE
    MF_HOSTILE | MF_GROUP,                  # GIANT_ANT (hostile, group)
    MF_HOSTILE | MF_POISONOUS | MF_GROUP,   # SOLDIER_ANT (hostile, poison, group)
    # --- Mid ---
    MF_HOSTILE | MF_GROUP,                  # WOLF (hostile, pack)
    MF_HOSTILE,                             # WINTER_WOLF (hostile)
    MF_HOSTILE | MF_STONING | MF_POISONOUS, # COCKATRICE (hostile, stoning, poisonous)
    MF_HOSTILE | MF_GROUP,                  # ORC (hostile, group)
    0x000,                                  # DWARF
    MF_HOSTILE,                             # NYMPH (hostile)
    MF_HOSTILE | MF_INVISIBLE | MF_SEE_INVIS,  # STALKER (hostile, invisible, see invis)
    MF_HOSTILE,                             # TIGER (hostile)
    MF_HOSTILE | MF_UNDEAD,                 # WRAITH (hostile, undead)
    MF_HOSTILE | MF_ACIDIC,                 # GELATINOUS_CUBE (hostile, acidic)
    MF_HOSTILE | MF_SEE_INVIS,              # MIND_FLAYER (hostile, see invis)
    MF_HOSTILE,                             # HELL_HOUND_PUP (hostile)
    # --- Late ---
    MF_HOSTILE | MF_REGEN,                  # TROLL (hostile, regenerates)
    MF_HOSTILE,                             # OGRE (hostile)
    MF_HOSTILE | MF_UNDEAD,                 # LICH (hostile, undead)
    MF_HOSTILE,                             # MINOTAUR (hostile)
    MF_HOSTILE | MF_UNDEAD,                 # VAMPIRE (hostile, undead)
    MF_HOSTILE | MF_UNDEAD,                 # VAMPIRE_LORD (hostile, undead)
    MF_HOSTILE,                             # JABBERWOCK (hostile)
    # --- Dragons ---
    MF_HOSTILE | MF_FLY | MF_SEE_INVIS,    # RED_DRAGON
    MF_HOSTILE | MF_FLY | MF_SEE_INVIS,    # WHITE_DRAGON
    MF_HOSTILE | MF_FLY | MF_SEE_INVIS,    # BLUE_DRAGON
    MF_HOSTILE | MF_FLY | MF_SEE_INVIS,    # GREEN_DRAGON
    MF_HOSTILE | MF_FLY | MF_SEE_INVIS,    # BLACK_DRAGON
    MF_HOSTILE | MF_FLY | MF_SEE_INVIS,    # GRAY_DRAGON
    MF_HOSTILE | MF_FLY | MF_SEE_INVIS,    # SILVER_DRAGON
    # --- Giants / Demons ---
    MF_HOSTILE,                             # HILL_GIANT
    MF_HOSTILE,                             # FIRE_GIANT
    MF_HOSTILE,                             # FROST_GIANT
    MF_HOSTILE | MF_FLY | MF_DEMON,        # PIT_FIEND (hostile, fly, demon)
    MF_HOSTILE | MF_FLY | MF_DEMON,        # BALROG (hostile, fly, demon)
], dtype=jnp.int32)


# --------------------------------------------------------------------------
# MONSTER_SPAWN_FLOORS  [min_floor, max_floor]
# --------------------------------------------------------------------------
MONSTER_SPAWN_FLOORS = jnp.array([
    [ 0,  0],  # NONE
    # --- Early ---
    [ 1,  5],  # NEWT
    [ 1,  5],  # JACKAL
    [ 1,  6],  # KOBOLD
    [ 1,  5],  # SEWER_RAT
    [ 1,  4],  # GRID_BUG
    [ 1,  5],  # LICHEN
    [ 1,  5],  # GECKO
    [ 1,  6],  # ACID_BLOB
    [ 1,  6],  # GNOME
    [ 1,  5],  # HOBBIT
    [ 1,  6],  # KILLER_BEE
    [ 1,  8],  # BAT
    [ 1,  5],  # KITTEN
    [ 1,  7],  # FLOATING_EYE
    [ 1,  7],  # GIANT_ANT
    [ 2,  8],  # SOLDIER_ANT
    # --- Mid ---
    [ 4, 10],  # WOLF
    [ 5, 12],  # WINTER_WOLF
    [ 5, 14],  # COCKATRICE
    [ 3, 10],  # ORC
    [ 4, 12],  # DWARF
    [ 6, 15],  # NYMPH
    [ 6, 15],  # STALKER
    [ 6, 14],  # TIGER
    [ 6, 16],  # WRAITH
    [ 5, 14],  # GELATINOUS_CUBE
    [ 8, 18],  # MIND_FLAYER
    [ 6, 14],  # HELL_HOUND_PUP
    # --- Late ---
    [ 8, 20],  # TROLL
    [ 6, 16],  # OGRE
    [10, 25],  # LICH
    [12, 25],  # MINOTAUR
    [10, 22],  # VAMPIRE
    [12, 25],  # VAMPIRE_LORD
    [14, 25],  # JABBERWOCK
    # --- Dragons ---
    [15, 30],  # RED_DRAGON
    [15, 30],  # WHITE_DRAGON
    [15, 30],  # BLUE_DRAGON
    [15, 30],  # GREEN_DRAGON
    [15, 30],  # BLACK_DRAGON
    [15, 30],  # GRAY_DRAGON
    [15, 30],  # SILVER_DRAGON
    # --- Giants / Demons ---
    [10, 22],  # HILL_GIANT
    [14, 28],  # FIRE_GIANT
    [14, 28],  # FROST_GIANT
    [18, 30],  # PIT_FIEND
    [18, 30],  # BALROG
], dtype=jnp.int32)


# --------------------------------------------------------------------------
# MONSTER_SYMBOLS  (ASCII glyph per the NetHack convention)
# --------------------------------------------------------------------------
MONSTER_SYMBOLS = [
    ' ',   # NONE
    # --- Early ---
    ':',   # NEWT         (lizard class)
    'd',   # JACKAL       (canine)
    'k',   # KOBOLD       (kobold)
    'r',   # SEWER_RAT    (rodent)
    'x',   # GRID_BUG     (xan class)
    'F',   # LICHEN       (fungus)
    ':',   # GECKO        (lizard class)
    'b',   # ACID_BLOB    (blob)
    'G',   # GNOME        (gnome)
    'h',   # HOBBIT       (humanoid)
    'a',   # KILLER_BEE   (ant class)
    'B',   # BAT          (bird)
    'f',   # KITTEN       (feline)
    'e',   # FLOATING_EYE (eye)
    'a',   # GIANT_ANT    (ant class)
    'a',   # SOLDIER_ANT  (ant class)
    # --- Mid ---
    'd',   # WOLF         (canine)
    'd',   # WINTER_WOLF  (canine)
    'c',   # COCKATRICE   (cockatrice)
    'o',   # ORC          (orc)
    'h',   # DWARF        (humanoid)
    'n',   # NYMPH        (nymph)
    'E',   # STALKER      (elemental)
    'f',   # TIGER        (feline)
    'W',   # WRAITH       (wraith)
    'b',   # GELATINOUS_CUBE (blob)
    'h',   # MIND_FLAYER  (humanoid)
    'd',   # HELL_HOUND_PUP (canine)
    # --- Late ---
    'T',   # TROLL        (troll)
    'O',   # OGRE         (ogre)
    'L',   # LICH         (lich)
    'H',   # MINOTAUR     (giant humanoid)
    'V',   # VAMPIRE      (vampire)
    'V',   # VAMPIRE_LORD (vampire)
    'J',   # JABBERWOCK   (jabberwock)
    # --- Dragons ---
    'D',   # RED_DRAGON
    'D',   # WHITE_DRAGON
    'D',   # BLUE_DRAGON
    'D',   # GREEN_DRAGON
    'D',   # BLACK_DRAGON
    'D',   # GRAY_DRAGON
    'D',   # SILVER_DRAGON
    # --- Giants / Demons ---
    'H',   # HILL_GIANT   (giant humanoid)
    'H',   # FIRE_GIANT   (giant humanoid)
    'H',   # FROST_GIANT  (giant humanoid)
    '&',   # PIT_FIEND    (major demon)
    '&',   # BALROG       (major demon)
]


# --------------------------------------------------------------------------
# CORPSE_EFFECTS  (simple integer code per monster)
#   -2 = instant death (cockatrice stoning)
#   -1 = poisonous
#    0 = none
#    1 = level gain (wraith)
#    2 = telepathy (floating eye, mind flayer)
#    3 = see_invis + invis (stalker)
#    4 = teleportitis (nymph)
# --------------------------------------------------------------------------
CORPSE_EFFECTS = jnp.array([
     0,  # NONE
    # --- Early ---
     0,  # NEWT
     0,  # JACKAL
    -1,  # KOBOLD (poisonous)
     0,  # SEWER_RAT
     0,  # GRID_BUG
     0,  # LICHEN
     0,  # GECKO
     0,  # ACID_BLOB
     0,  # GNOME
     0,  # HOBBIT
    -1,  # KILLER_BEE (grants poison res but stings are poisonous)
     0,  # BAT
     0,  # KITTEN
     2,  # FLOATING_EYE (telepathy)
     0,  # GIANT_ANT
    -1,  # SOLDIER_ANT (poisonous)
    # --- Mid ---
     0,  # WOLF
     0,  # WINTER_WOLF
    -2,  # COCKATRICE (instant death / stoning)
     0,  # ORC
     0,  # DWARF
     4,  # NYMPH (teleportitis)
     3,  # STALKER (see_invis + invisibility)
     0,  # TIGER
     1,  # WRAITH (level gain)
     0,  # GELATINOUS_CUBE
     2,  # MIND_FLAYER (telepathy)
     0,  # HELL_HOUND_PUP
    # --- Late ---
     0,  # TROLL
     0,  # OGRE
     0,  # LICH (no corpse left)
     0,  # MINOTAUR
     0,  # VAMPIRE
     0,  # VAMPIRE_LORD
     0,  # JABBERWOCK
    # --- Dragons ---
     0,  # RED_DRAGON
     0,  # WHITE_DRAGON
     0,  # BLUE_DRAGON
     0,  # GREEN_DRAGON
     0,  # BLACK_DRAGON
     0,  # GRAY_DRAGON
     0,  # SILVER_DRAGON
    # --- Giants / Demons ---
     0,  # HILL_GIANT
     0,  # FIRE_GIANT
     0,  # FROST_GIANT
     0,  # PIT_FIEND (no corpse)
     0,  # BALROG (no corpse)
], dtype=jnp.int32)


# ============================================================================
# Player actions
# ============================================================================
class Action(IntEnum):
    NOOP = 0
    # Movement (4 cardinal + 4 diagonal)
    MOVE_N = 1
    MOVE_S = 2
    MOVE_E = 3
    MOVE_W = 4
    MOVE_NE = 5
    MOVE_NW = 6
    MOVE_SE = 7
    MOVE_SW = 8
    # Interaction
    PICKUP = 9         # Pick up item from ground
    DROP = 10          # Drop item from inventory
    EAT = 11           # Eat food/corpse
    QUAFF = 12         # Drink potion
    READ = 13          # Read scroll
    ZAP = 14           # Zap wand
    WEAR = 15          # Put on armor
    WIELD = 16         # Wield weapon
    REMOVE = 17        # Remove armor
    THROW = 18         # Throw item
    OPEN_DOOR = 19     # Open a closed door
    CLOSE_DOOR = 20    # Close an open door
    KICK = 21          # Kick (door/monster)
    SEARCH = 22        # Search for hidden doors/traps
    WAIT = 23          # Wait one turn
    GO_UP = 24         # Climb stairs up
    GO_DOWN = 25       # Climb stairs down
    PRAY = 26          # Pray to your god
    APPLY = 27         # Use a tool


NUM_ACTIONS = len(Action)

# Direction vectors for movement actions
DIRECTION_VECTORS = jnp.array([
    [0, 0],   # NOOP
    [-1, 0],  # N
    [1, 0],   # S
    [0, 1],   # E
    [0, -1],  # W
    [-1, 1],  # NE
    [-1, -1], # NW
    [1, 1],   # SE
    [1, -1],  # SW
], dtype=jnp.int32)


# ============================================================================
# BUC (Blessed/Uncursed/Cursed) status
# ============================================================================
class BUCStatus(IntEnum):
    CURSED = 0
    UNCURSED = 1
    BLESSED = 2

# Unknown until identified
BUC_UNKNOWN = -1


# ============================================================================
# Intrinsics / Resistances
# ============================================================================
class Intrinsic(IntEnum):
    FIRE_RESISTANCE = 0
    COLD_RESISTANCE = 1
    POISON_RESISTANCE = 2
    SLEEP_RESISTANCE = 3
    SHOCK_RESISTANCE = 4
    SEE_INVISIBLE = 5
    TELEPATHY = 6
    SPEED = 7

NUM_INTRINSICS = len(Intrinsic)


# ============================================================================
# Hunger states
# ============================================================================
class HungerState(IntEnum):
    SATIATED = 0
    NOT_HUNGRY = 1
    HUNGRY = 2
    WEAK = 3
    FAINTING = 4
    STARVED = 5

# Nutrition thresholds for hunger states
HUNGER_THRESHOLDS = jnp.array([
    2000,  # SATIATED (above this)
    1000,  # NOT_HUNGRY
    300,   # HUNGRY
    150,   # WEAK
    0,     # FAINTING
    -1,    # STARVED (dead)
], dtype=jnp.int32)


# ============================================================================
# XP table (experience points required per level, levels 1-30)
# ============================================================================
# NetHack XP formula:
#   Levels  1- 9:  10 * 2^level       (level 1 needs 0, level 2 needs 20, ...)
#   Levels 10-19:  10000 * 2^(level-10)
#   Levels 20-30:  10000000 * (level-19)
# Index 0 = level 1 (always 0 XP needed).
XP_TABLE = jnp.array([
    0,            # Level  1
    20,           # Level  2  (10 * 2^1)
    40,           # Level  3  (10 * 2^2)
    80,           # Level  4  (10 * 2^3)
    160,          # Level  5  (10 * 2^4)
    320,          # Level  6  (10 * 2^5)
    640,          # Level  7  (10 * 2^6)
    1280,         # Level  8  (10 * 2^7)
    2560,         # Level  9  (10 * 2^8)
    5120,         # Level 10  (10 * 2^9)
    10000,        # Level 11  (10000 * 2^0)
    20000,        # Level 12  (10000 * 2^1)
    40000,        # Level 13  (10000 * 2^2)
    80000,        # Level 14  (10000 * 2^3)
    160000,       # Level 15  (10000 * 2^4)
    320000,       # Level 16  (10000 * 2^5)
    640000,       # Level 17  (10000 * 2^6)
    1280000,      # Level 18  (10000 * 2^7)
    2560000,      # Level 19  (10000 * 2^8)
    5120000,      # Level 20  (10000 * 2^9)
    10000000,     # Level 21  (10000000 * 1)
    20000000,     # Level 22  (10000000 * 2)
    30000000,     # Level 23  (10000000 * 3)
    40000000,     # Level 24  (10000000 * 4)
    50000000,     # Level 25  (10000000 * 5)
    60000000,     # Level 26  (10000000 * 6)
    70000000,     # Level 27  (10000000 * 7)
    80000000,     # Level 28  (10000000 * 8)
    90000000,     # Level 29  (10000000 * 9)
    100000000,    # Level 30  (10000000 * 10)
], dtype=jnp.int32)


# ============================================================================
# Speed constants (turns per 12-tick cycle)
# ============================================================================
SPEED_VERY_FAST = 24
SPEED_FAST = 18
SPEED_NORMAL = 12
SPEED_SLOW = 6
SPEED_VERY_SLOW = 3


# ============================================================================
# Dungeon branch / special levels
# ============================================================================
class DungeonBranch(IntEnum):
    MAIN = 0
    MINES = 1


# ============================================================================
# Achievements / milestones for RL reward
# ============================================================================
class Achievement(IntEnum):
    # Exploration
    REACH_FLOOR_2 = 0
    REACH_FLOOR_5 = 1
    REACH_FLOOR_10 = 2
    REACH_FLOOR_15 = 3
    REACH_FLOOR_20 = 4
    # Combat
    KILL_FIRST_MONSTER = 5
    KILL_10_MONSTERS = 6
    KILL_50_MONSTERS = 7
    # Items
    IDENTIFY_FIRST_ITEM = 8
    EQUIP_WEAPON = 9
    EQUIP_ARMOR = 10
    QUAFF_POTION = 11
    READ_SCROLL = 12
    ZAP_WAND = 13
    # Survival
    EAT_FOOD = 14
    PRAY_SUCCESSFULLY = 15
    FIND_GOLD = 16
    # Shopping
    ENTER_SHOP = 17
    BUY_ITEM = 18
    # Progression
    GAIN_LEVEL = 19
    REACH_XP_LEVEL_5 = 20
    REACH_XP_LEVEL_10 = 21
    REACH_XP_LEVEL_14 = 22
    # Special
    EAT_CORPSE = 23
    GAIN_INTRINSIC = 24
    OPEN_DOOR = 25
    FIND_TRAP = 26
    USE_FOUNTAIN = 27

NUM_ACHIEVEMENTS = len(Achievement)

# Reward weights per achievement tier
ACHIEVEMENT_REWARD_WEIGHTS = jnp.array([
    # Exploration
    1, 2, 3, 5, 8,
    # Combat
    1, 2, 3,
    # Items
    1, 1, 1, 1, 1, 1,
    # Survival
    1, 2, 1,
    # Shopping
    1, 1,
    # Progression
    1, 2, 3, 5,
    # Special
    1, 2, 1, 1, 1,
], dtype=jnp.float32)
