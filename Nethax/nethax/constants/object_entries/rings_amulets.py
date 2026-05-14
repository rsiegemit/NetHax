"""Rings and amulets — vendor/nethack/include/objects.h lines 736-876."""
from Nethax.nethax.constants.objects import (
    ObjectEntry,
    ObjectClass,
    Color,
    Material,
)

# ---------------------------------------------------------------------------
# Additional HI_* colour aliases not yet in Color enum
# Source: vendor/nethack/include/color.h
#   HI_COPPER = CLR_YELLOW  (44)
#   HI_GOLD   = CLR_YELLOW  (46)
#   HI_MINERAL = CLR_GRAY   (53)
# ---------------------------------------------------------------------------
HI_COPPER  = Color.CLR_YELLOW   # 11
HI_GOLD    = Color.CLR_YELLOW   # 11
HI_MINERAL = Color.CLR_GRAY     # 7

# Short aliases for readability
HI_METAL   = Color.HI_METAL     # CLR_CYAN  = 6
HI_SILVER  = Color.HI_SILVER    # CLR_GRAY  = 7

# ---------------------------------------------------------------------------
# RING macro expansion (objects.h line 736-740):
#   RING(name, stone, power, cost, mgc, spec, mohs, metal, color, sn)
#   → OBJECT(..., power, RING_CLASS, prob=1, stack=0, weight=3, cost,
#             sdam=0, ldam=0, oc1=0, oc2=0, nutrition=15, color, sn)
#
# ObjectEntry field mapping:
#   name        = name
#   description = stone  (appearance when unidentified)
#   class_      = ObjectClass.RING_CLASS
#   prob        = 1   (all rings have prob=1 in macro)
#   weight      = 3
#   cost        = cost arg
#   sdam        = (0, 0)
#   ldam        = (0, 0)
#   oc1         = 0
#   oc2         = 0
#   nutrition   = 15
#   color       = color arg
#   material    = metal arg
#
# AMULET macro expansion (objects.h line 831-834):
#   AMULET(name, desc, power, prob, sn)
#   → OBJECT(..., power, AMULET_CLASS, prob, stack=0, weight=20, cost=150,
#             sdam=0, ldam=0, oc1=0, oc2=0, nutrition=20, HI_METAL, sn)
#
# ObjectEntry field mapping:
#   name        = name
#   description = desc
#   class_      = ObjectClass.AMULET_CLASS
#   prob        = prob arg
#   weight      = 20
#   cost        = 150
#   sdam        = (0, 0)
#   ldam        = (0, 0)
#   oc1         = 0
#   oc2         = 0
#   nutrition   = 20
#   color       = HI_METAL
#   material    = Material.IRON
# ---------------------------------------------------------------------------

ENTRIES: tuple = (

    # --- Rings (28 entries) ---

    # RING("adornment", "wooden", ADORNED, 100, 1, 1, 2, WOOD, HI_WOOD, ...)
    ObjectEntry(
        name="adornment",
        description="wooden",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.HI_WOOD,
        material=Material.WOOD,
    ),

    # RING("gain strength", "granite", 0, 150, 1, 1, 7, MINERAL, HI_MINERAL, ...)
    ObjectEntry(
        name="gain strength",
        description="granite",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=HI_MINERAL,
        material=Material.MINERAL,
    ),

    # RING("gain constitution", "opal", 0, 150, 1, 1, 7, MINERAL, HI_MINERAL, ...)
    ObjectEntry(
        name="gain constitution",
        description="opal",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=HI_MINERAL,
        material=Material.MINERAL,
    ),

    # RING("increase accuracy", "clay", 0, 150, 1, 1, 4, MINERAL, CLR_RED, ...)
    ObjectEntry(
        name="increase accuracy",
        description="clay",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_RED,
        material=Material.MINERAL,
    ),

    # RING("increase damage", "coral", 0, 150, 1, 1, 4, MINERAL, CLR_ORANGE, ...)
    ObjectEntry(
        name="increase damage",
        description="coral",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_ORANGE,
        material=Material.MINERAL,
    ),

    # RING("protection", "black onyx", PROTECTION, 100, 1, 1, 7, MINERAL, CLR_BLACK, ...)
    ObjectEntry(
        name="protection",
        description="black onyx",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_BLACK,
        material=Material.MINERAL,
    ),

    # RING("regeneration", "moonstone", REGENERATION, 200, 1, 0, 6, MINERAL, HI_MINERAL, ...)
    ObjectEntry(
        name="regeneration",
        description="moonstone",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=HI_MINERAL,
        material=Material.MINERAL,
    ),

    # RING("searching", "tiger eye", SEARCHING, 200, 1, 0, 6, GEMSTONE, CLR_BROWN, ...)
    ObjectEntry(
        name="searching",
        description="tiger eye",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_BROWN,
        material=Material.GEMSTONE,
    ),

    # RING("stealth", "jade", STEALTH, 100, 1, 0, 6, GEMSTONE, CLR_GREEN, ...)
    ObjectEntry(
        name="stealth",
        description="jade",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_GREEN,
        material=Material.GEMSTONE,
    ),

    # RING("sustain ability", "bronze", FIXED_ABIL, 100, 1, 0, 4, COPPER, HI_COPPER, ...)
    ObjectEntry(
        name="sustain ability",
        description="bronze",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=HI_COPPER,
        material=Material.COPPER,
    ),

    # RING("levitation", "agate", LEVITATION, 200, 1, 0, 7, GEMSTONE, CLR_RED, ...)
    ObjectEntry(
        name="levitation",
        description="agate",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_RED,
        material=Material.GEMSTONE,
    ),

    # RING("hunger", "topaz", HUNGER, 100, 1, 0, 8, GEMSTONE, CLR_CYAN, ...)
    ObjectEntry(
        name="hunger",
        description="topaz",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_CYAN,
        material=Material.GEMSTONE,
    ),

    # RING("aggravate monster", "sapphire", AGGRAVATE_MONSTER, 150, 1, 0, 9, GEMSTONE, CLR_BLUE, ...)
    ObjectEntry(
        name="aggravate monster",
        description="sapphire",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_BLUE,
        material=Material.GEMSTONE,
    ),

    # RING("conflict", "ruby", CONFLICT, 300, 1, 0, 9, GEMSTONE, CLR_RED, ...)
    ObjectEntry(
        name="conflict",
        description="ruby",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=300,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_RED,
        material=Material.GEMSTONE,
    ),

    # RING("warning", "diamond", WARNING, 100, 1, 0, 10, GEMSTONE, CLR_WHITE, ...)
    ObjectEntry(
        name="warning",
        description="diamond",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_WHITE,
        material=Material.GEMSTONE,
    ),

    # RING("poison resistance", "pearl", POISON_RES, 150, 1, 0, 4, BONE, CLR_WHITE, ...)
    ObjectEntry(
        name="poison resistance",
        description="pearl",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_WHITE,
        material=Material.BONE,
    ),

    # RING("fire resistance", "iron", FIRE_RES, 200, 1, 0, 5, IRON, HI_METAL, ...)
    ObjectEntry(
        name="fire resistance",
        description="iron",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # RING("cold resistance", "brass", COLD_RES, 150, 1, 0, 4, COPPER, HI_COPPER, ...)
    ObjectEntry(
        name="cold resistance",
        description="brass",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=HI_COPPER,
        material=Material.COPPER,
    ),

    # RING("shock resistance", "copper", SHOCK_RES, 150, 1, 0, 3, COPPER, HI_COPPER, ...)
    ObjectEntry(
        name="shock resistance",
        description="copper",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=HI_COPPER,
        material=Material.COPPER,
    ),

    # RING("free action", "twisted", FREE_ACTION, 200, 1, 0, 6, IRON, HI_METAL, ...)
    ObjectEntry(
        name="free action",
        description="twisted",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # RING("slow digestion", "steel", SLOW_DIGESTION, 200, 1, 0, 8, IRON, HI_METAL, ...)
    ObjectEntry(
        name="slow digestion",
        description="steel",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # RING("teleportation", "silver", TELEPORT, 200, 1, 0, 3, SILVER, HI_SILVER, ...)
    ObjectEntry(
        name="teleportation",
        description="silver",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=HI_SILVER,
        material=Material.SILVER,
    ),

    # RING("teleport control", "gold", TELEPORT_CONTROL, 300, 1, 0, 3, GOLD, HI_GOLD, ...)
    ObjectEntry(
        name="teleport control",
        description="gold",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=300,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=HI_GOLD,
        material=Material.GOLD,
    ),

    # RING("polymorph", "ivory", POLYMORPH, 300, 1, 0, 4, BONE, CLR_WHITE, ...)
    ObjectEntry(
        name="polymorph",
        description="ivory",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=300,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_WHITE,
        material=Material.BONE,
    ),

    # RING("polymorph control", "emerald", POLYMORPH_CONTROL, 300, 1, 0, 8, GEMSTONE, CLR_BRIGHT_GREEN, ...)
    ObjectEntry(
        name="polymorph control",
        description="emerald",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=300,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_BRIGHT_GREEN,
        material=Material.GEMSTONE,
    ),

    # RING("invisibility", "wire", INVIS, 150, 1, 0, 5, IRON, HI_METAL, ...)
    ObjectEntry(
        name="invisibility",
        description="wire",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # RING("see invisible", "engagement", SEE_INVIS, 150, 1, 0, 5, IRON, HI_METAL, ...)
    ObjectEntry(
        name="see invisible",
        description="engagement",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # RING("protection from shape changers", "shiny", PROT_FROM_SHAPE_CHANGERS, 100, 1, 0, 5, IRON, CLR_BRIGHT_CYAN, ...)
    ObjectEntry(
        name="protection from shape changers",
        description="shiny",
        class_=ObjectClass.RING_CLASS,
        prob=1,
        weight=3,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_BRIGHT_CYAN,
        material=Material.IRON,
    ),

    # --- Amulets (11 regular + fake Amulet of Yendor + real Amulet of Yendor = 13 entries) ---

    # AMULET("amulet of ESP", "circular", TELEPAT, 120, ...)
    ObjectEntry(
        name="amulet of ESP",
        description="circular",
        class_=ObjectClass.AMULET_CLASS,
        prob=120,
        weight=20,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # AMULET("amulet of life saving", "spherical", LIFESAVED, 75, ...)
    ObjectEntry(
        name="amulet of life saving",
        description="spherical",
        class_=ObjectClass.AMULET_CLASS,
        prob=75,
        weight=20,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # AMULET("amulet of strangulation", "oval", STRANGLED, 115, ...)
    ObjectEntry(
        name="amulet of strangulation",
        description="oval",
        class_=ObjectClass.AMULET_CLASS,
        prob=115,
        weight=20,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # AMULET("amulet of restful sleep", "triangular", SLEEPY, 115, ...)
    ObjectEntry(
        name="amulet of restful sleep",
        description="triangular",
        class_=ObjectClass.AMULET_CLASS,
        prob=115,
        weight=20,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # AMULET("amulet versus poison", "pyramidal", POISON_RES, 115, ...)
    ObjectEntry(
        name="amulet versus poison",
        description="pyramidal",
        class_=ObjectClass.AMULET_CLASS,
        prob=115,
        weight=20,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # AMULET("amulet of change", "square", 0, 115, ...)
    ObjectEntry(
        name="amulet of change",
        description="square",
        class_=ObjectClass.AMULET_CLASS,
        prob=115,
        weight=20,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # AMULET("amulet of unchanging", "concave", UNCHANGING, 60, ...)
    ObjectEntry(
        name="amulet of unchanging",
        description="concave",
        class_=ObjectClass.AMULET_CLASS,
        prob=60,
        weight=20,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # AMULET("amulet of reflection", "hexagonal", REFLECTING, 75, ...)
    ObjectEntry(
        name="amulet of reflection",
        description="hexagonal",
        class_=ObjectClass.AMULET_CLASS,
        prob=75,
        weight=20,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # AMULET("amulet of magical breathing", "octagonal", MAGICAL_BREATHING, 75, ...)
    ObjectEntry(
        name="amulet of magical breathing",
        description="octagonal",
        class_=ObjectClass.AMULET_CLASS,
        prob=75,
        weight=20,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # AMULET("amulet of guarding", "perforated", PROTECTION, 75, ...)
    ObjectEntry(
        name="amulet of guarding",
        description="perforated",
        class_=ObjectClass.AMULET_CLASS,
        prob=75,
        weight=20,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # AMULET("amulet of flying", "cubical", FLYING, 60, ...)
    ObjectEntry(
        name="amulet of flying",
        description="cubical",
        class_=ObjectClass.AMULET_CLASS,
        prob=60,
        weight=20,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=HI_METAL,
        material=Material.IRON,
    ),

    # OBJECT("cheap plastic imitation of the Amulet of Yendor", "Amulet of Yendor", ...)
    # BITS(..., PLASTIC); prob=0, weight=20, cost=0, nutrition=1
    ObjectEntry(
        name="cheap plastic imitation of the Amulet of Yendor",
        description="Amulet of Yendor",
        class_=ObjectClass.AMULET_CLASS,
        prob=0,
        weight=20,
        cost=0,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=1,
        color=HI_METAL,
        material=Material.PLASTIC,
    ),

    # OBJECT("Amulet of Yendor", "Amulet of Yendor", ...)
    # BITS(..., MITHRIL); prob=0, weight=20, cost=30000, nutrition=20
    ObjectEntry(
        name="Amulet of Yendor",
        description="Amulet of Yendor",
        class_=ObjectClass.AMULET_CLASS,
        prob=0,
        weight=20,
        cost=30000,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=HI_METAL,
        material=Material.MITHRIL,
    ),
)
