"""Food, gems, rocks — vendor/nethack/include/objects.h FOOD/GEM/ROCK entries."""
from Nethax.nethax.constants.objects import (
    ObjectEntry, ObjectClass, Material, Color,
)

# ---------------------------------------------------------------------------
# Field mapping notes
# ---------------------------------------------------------------------------
# FOOD(name, prob, delay, wt, unk, tin, nutrition, color, sn)
#   → cost = nutrition/20 + 5 (integer); sdam=(0,0); ldam=(0,0); oc1=oc2=0
#   → material = tin arg (FLESH, VEGGY, or METAL)
#
# GEM(name, desc, prob, wt, gval, nutr, mohs, glass, color, sn)
#   → cost = gval; sdam=(3,3); ldam=(3,3); oc1=oc2=0
#   → glass arg = material (GEMSTONE or GLASS)
#   → description: desc value with " gem" appended (matching existing convention)
#
# ROCK(name, desc, kn, prob, wt, gval, sdam, ldam, mgc, nutr, mohs, glass, colr, sn)
#   → cost = gval; sdam=(sdam,sdam); ldam=(ldam,ldam); oc1=oc2=0
#   → glass arg = material (MINERAL)
#   → NoDes desc → None
#
# HI_ORGANIC = CLR_BROWN (3), HI_MINERAL = CLR_GRAY (7), HI_METAL = CLR_CYAN (6)
# ---------------------------------------------------------------------------

ENTRIES = (
    # -----------------------------------------------------------------------
    # FOOD entries
    # Source: vendor/nethack/include/objects.h lines 1048–1117
    # -----------------------------------------------------------------------

    # FOOD("tripe ration", 140, 2, 10, 0, FLESH, 200, CLR_BROWN, TRIPE_RATION)
    # cost = 200/20 + 5 = 15
    ObjectEntry(
        name="tripe ration",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=140,
        weight=10,
        cost=15,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=200,
        color=Color.CLR_BROWN,
        material=Material.FLESH,
    ),

    # FOOD("corpse", 0, 1, 0, 0, FLESH, 0, CLR_BROWN, CORPSE)
    # cost = 0/20 + 5 = 5
    ObjectEntry(
        name="corpse",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=0,
        weight=0,
        cost=5,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=0,
        color=Color.CLR_BROWN,
        material=Material.FLESH,
    ),

    # FOOD("egg", 85, 1, 1, 1, FLESH, 80, CLR_WHITE, EGG)
    # cost = 80/20 + 5 = 9
    ObjectEntry(
        name="egg",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=85,
        weight=1,
        cost=9,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=80,
        color=Color.CLR_WHITE,
        material=Material.FLESH,
    ),

    # FOOD("meatball", 0, 1, 1, 0, FLESH, 5, CLR_BROWN, MEATBALL)
    # cost = 5/20 + 5 = 5
    ObjectEntry(
        name="meatball",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=0,
        weight=1,
        cost=5,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=5,
        color=Color.CLR_BROWN,
        material=Material.FLESH,
    ),

    # FOOD("meat stick", 0, 1, 1, 0, FLESH, 5, CLR_BROWN, MEAT_STICK)
    # cost = 5/20 + 5 = 5
    ObjectEntry(
        name="meat stick",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=0,
        weight=1,
        cost=5,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=5,
        color=Color.CLR_BROWN,
        material=Material.FLESH,
    ),

    # FOOD("enormous meatball", 0, 20, 400, 0, FLESH, 2000, CLR_BROWN, ENORMOUS_MEATBALL)
    # cost = 2000/20 + 5 = 105
    ObjectEntry(
        name="enormous meatball",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=0,
        weight=400,
        cost=105,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=2000,
        color=Color.CLR_BROWN,
        material=Material.FLESH,
    ),

    # FOOD("glob of gray ooze", 0, 2, 20, 0, FLESH, 20, CLR_GRAY, GLOB_OF_GRAY_OOZE)
    # cost = 20/20 + 5 = 6
    ObjectEntry(
        name="glob of gray ooze",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=0,
        weight=20,
        cost=6,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=Color.CLR_GRAY,
        material=Material.FLESH,
    ),

    # FOOD("glob of brown pudding", 0, 2, 20, 0, FLESH, 20, CLR_BROWN, GLOB_OF_BROWN_PUDDING)
    # cost = 20/20 + 5 = 6
    ObjectEntry(
        name="glob of brown pudding",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=0,
        weight=20,
        cost=6,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=Color.CLR_BROWN,
        material=Material.FLESH,
    ),

    # FOOD("glob of green slime", 0, 2, 20, 0, FLESH, 20, CLR_GREEN, GLOB_OF_GREEN_SLIME)
    # cost = 20/20 + 5 = 6
    ObjectEntry(
        name="glob of green slime",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=0,
        weight=20,
        cost=6,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=Color.CLR_GREEN,
        material=Material.FLESH,
    ),

    # FOOD("glob of black pudding", 0, 2, 20, 0, FLESH, 20, CLR_BLACK, GLOB_OF_BLACK_PUDDING)
    # cost = 20/20 + 5 = 6
    ObjectEntry(
        name="glob of black pudding",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=0,
        weight=20,
        cost=6,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=Color.CLR_BLACK,
        material=Material.FLESH,
    ),

    # FOOD("kelp frond", 0, 1, 1, 0, VEGGY, 30, CLR_GREEN, KELP_FROND)
    # cost = 30/20 + 5 = 6
    ObjectEntry(
        name="kelp frond",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=0,
        weight=1,
        cost=6,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=30,
        color=Color.CLR_GREEN,
        material=Material.VEGGY,
    ),

    # FOOD("eucalyptus leaf", 3, 1, 1, 0, VEGGY, 1, CLR_GREEN, EUCALYPTUS_LEAF)
    # cost = 1/20 + 5 = 5
    ObjectEntry(
        name="eucalyptus leaf",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=3,
        weight=1,
        cost=5,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=1,
        color=Color.CLR_GREEN,
        material=Material.VEGGY,
    ),

    # FOOD("apple", 15, 1, 2, 0, VEGGY, 50, CLR_RED, APPLE)
    # cost = 50/20 + 5 = 7
    ObjectEntry(
        name="apple",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=15,
        weight=2,
        cost=7,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=50,
        color=Color.CLR_RED,
        material=Material.VEGGY,
    ),

    # FOOD("orange", 10, 1, 2, 0, VEGGY, 80, CLR_ORANGE, ORANGE)
    # cost = 80/20 + 5 = 9
    ObjectEntry(
        name="orange",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=10,
        weight=2,
        cost=9,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=80,
        color=Color.CLR_ORANGE,
        material=Material.VEGGY,
    ),

    # FOOD("pear", 10, 1, 2, 0, VEGGY, 50, CLR_BRIGHT_GREEN, PEAR)
    # cost = 50/20 + 5 = 7
    ObjectEntry(
        name="pear",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=10,
        weight=2,
        cost=7,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=50,
        color=Color.CLR_BRIGHT_GREEN,
        material=Material.VEGGY,
    ),

    # FOOD("melon", 10, 1, 5, 0, VEGGY, 100, CLR_BRIGHT_GREEN, MELON)
    # cost = 100/20 + 5 = 10
    ObjectEntry(
        name="melon",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=10,
        weight=5,
        cost=10,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=100,
        color=Color.CLR_BRIGHT_GREEN,
        material=Material.VEGGY,
    ),

    # FOOD("banana", 10, 1, 2, 0, VEGGY, 80, CLR_YELLOW, BANANA)
    # cost = 80/20 + 5 = 9
    ObjectEntry(
        name="banana",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=10,
        weight=2,
        cost=9,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=80,
        color=Color.CLR_YELLOW,
        material=Material.VEGGY,
    ),

    # FOOD("carrot", 15, 1, 2, 0, VEGGY, 50, CLR_ORANGE, CARROT)
    # cost = 50/20 + 5 = 7
    ObjectEntry(
        name="carrot",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=15,
        weight=2,
        cost=7,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=50,
        color=Color.CLR_ORANGE,
        material=Material.VEGGY,
    ),

    # FOOD("sprig of wolfsbane", 7, 1, 1, 0, VEGGY, 40, CLR_GREEN, SPRIG_OF_WOLFSBANE)
    # cost = 40/20 + 5 = 7
    ObjectEntry(
        name="sprig of wolfsbane",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=7,
        weight=1,
        cost=7,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=40,
        color=Color.CLR_GREEN,
        material=Material.VEGGY,
    ),

    # FOOD("clove of garlic", 7, 1, 1, 0, VEGGY, 40, CLR_WHITE, CLOVE_OF_GARLIC)
    # cost = 40/20 + 5 = 7
    ObjectEntry(
        name="clove of garlic",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=7,
        weight=1,
        cost=7,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=40,
        color=Color.CLR_WHITE,
        material=Material.VEGGY,
    ),

    # FOOD("slime mold", 75, 1, 5, 0, VEGGY, 250, HI_ORGANIC, SLIME_MOLD)
    # cost = 250/20 + 5 = 17; HI_ORGANIC = CLR_BROWN
    ObjectEntry(
        name="slime mold",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=75,
        weight=5,
        cost=17,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=250,
        color=Color.CLR_BROWN,    # HI_ORGANIC
        material=Material.VEGGY,
    ),

    # FOOD("lump of royal jelly", 0, 1, 2, 0, VEGGY, 200, CLR_YELLOW, LUMP_OF_ROYAL_JELLY)
    # cost = 200/20 + 5 = 15
    ObjectEntry(
        name="lump of royal jelly",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=0,
        weight=2,
        cost=15,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=200,
        color=Color.CLR_YELLOW,
        material=Material.VEGGY,
    ),

    # FOOD("cream pie", 25, 1, 10, 0, VEGGY, 100, CLR_WHITE, CREAM_PIE)
    # cost = 100/20 + 5 = 10
    ObjectEntry(
        name="cream pie",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=25,
        weight=10,
        cost=10,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=100,
        color=Color.CLR_WHITE,
        material=Material.VEGGY,
    ),

    # FOOD("candy bar", 13, 1, 2, 0, VEGGY, 100, CLR_BRIGHT_BLUE, CANDY_BAR)
    # cost = 100/20 + 5 = 10
    ObjectEntry(
        name="candy bar",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=13,
        weight=2,
        cost=10,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=100,
        color=Color.CLR_BRIGHT_BLUE,
        material=Material.VEGGY,
    ),

    # FOOD("fortune cookie", 55, 1, 1, 0, VEGGY, 40, CLR_YELLOW, FORTUNE_COOKIE)
    # cost = 40/20 + 5 = 7
    ObjectEntry(
        name="fortune cookie",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=55,
        weight=1,
        cost=7,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=40,
        color=Color.CLR_YELLOW,
        material=Material.VEGGY,
    ),

    # FOOD("pancake", 25, 2, 2, 0, VEGGY, 200, CLR_YELLOW, PANCAKE)
    # cost = 200/20 + 5 = 15
    ObjectEntry(
        name="pancake",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=25,
        weight=2,
        cost=15,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=200,
        color=Color.CLR_YELLOW,
        material=Material.VEGGY,
    ),

    # FOOD("lembas wafer", 20, 2, 5, 0, VEGGY, 800, CLR_WHITE, LEMBAS_WAFER)
    # cost = 800/20 + 5 = 45
    ObjectEntry(
        name="lembas wafer",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=20,
        weight=5,
        cost=45,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=800,
        color=Color.CLR_WHITE,
        material=Material.VEGGY,
    ),

    # FOOD("cram ration", 20, 3, 15, 0, VEGGY, 600, HI_ORGANIC, CRAM_RATION)
    # cost = 600/20 + 5 = 35; HI_ORGANIC = CLR_BROWN
    ObjectEntry(
        name="cram ration",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=20,
        weight=15,
        cost=35,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=600,
        color=Color.CLR_BROWN,    # HI_ORGANIC
        material=Material.VEGGY,
    ),

    # FOOD("food ration", 380, 5, 20, 0, VEGGY, 800, HI_ORGANIC, FOOD_RATION)
    # cost = 800/20 + 5 = 45; HI_ORGANIC = CLR_BROWN
    ObjectEntry(
        name="food ration",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=380,
        weight=20,
        cost=45,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=800,
        color=Color.CLR_BROWN,    # HI_ORGANIC
        material=Material.VEGGY,
    ),

    # FOOD("K-ration", 0, 1, 10, 0, VEGGY, 400, HI_ORGANIC, K_RATION)
    # cost = 400/20 + 5 = 25; HI_ORGANIC = CLR_BROWN
    ObjectEntry(
        name="K-ration",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=0,
        weight=10,
        cost=25,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=400,
        color=Color.CLR_BROWN,    # HI_ORGANIC
        material=Material.VEGGY,
    ),

    # FOOD("C-ration", 0, 1, 10, 0, VEGGY, 300, HI_ORGANIC, C_RATION)
    # cost = 300/20 + 5 = 20; HI_ORGANIC = CLR_BROWN
    ObjectEntry(
        name="C-ration",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=0,
        weight=10,
        cost=20,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=300,
        color=Color.CLR_BROWN,    # HI_ORGANIC
        material=Material.VEGGY,
    ),

    # FOOD("tin", 75, 0, 10, 1, METAL, 0, HI_METAL, TIN)
    # cost = 0/20 + 5 = 5; HI_METAL = CLR_CYAN
    ObjectEntry(
        name="tin",
        description=None,
        class_=ObjectClass.FOOD_CLASS,
        prob=75,
        weight=10,
        cost=5,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=0,
        color=Color.HI_METAL,     # CLR_CYAN
        material=Material.METAL,
    ),

    # -----------------------------------------------------------------------
    # GEM entries (precious stones + worthless glass)
    # Source: vendor/nethack/include/objects.h lines 1526–1590
    # GEM(name, desc, prob, wt, gval, nutr, mohs, glass, color, sn)
    #   → cost=gval, nutrition=nutr, sdam=(3,3), ldam=(3,3)
    #   → description = desc + " gem" (matching existing objects.py convention)
    # -----------------------------------------------------------------------

    # GEM("dilithium crystal", "white", 2, 1, 4500, 15, 5, GEMSTONE, CLR_WHITE, DILITHIUM_CRYSTAL)
    ObjectEntry(
        name="dilithium crystal",
        description="white gem",
        class_=ObjectClass.GEM_CLASS,
        prob=2,
        weight=1,
        cost=4500,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_WHITE,
        material=Material.GEMSTONE,
    ),

    # GEM("diamond", "white", 3, 1, 4000, 15, 10, GEMSTONE, CLR_WHITE, DIAMOND)
    ObjectEntry(
        name="diamond",
        description="white gem",
        class_=ObjectClass.GEM_CLASS,
        prob=3,
        weight=1,
        cost=4000,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_WHITE,
        material=Material.GEMSTONE,
    ),

    # GEM("ruby", "red", 4, 1, 3500, 15, 9, GEMSTONE, CLR_RED, RUBY)
    ObjectEntry(
        name="ruby",
        description="red gem",
        class_=ObjectClass.GEM_CLASS,
        prob=4,
        weight=1,
        cost=3500,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_RED,
        material=Material.GEMSTONE,
    ),

    # GEM("jacinth", "orange", 3, 1, 3250, 15, 9, GEMSTONE, CLR_ORANGE, JACINTH)
    ObjectEntry(
        name="jacinth",
        description="orange gem",
        class_=ObjectClass.GEM_CLASS,
        prob=3,
        weight=1,
        cost=3250,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_ORANGE,
        material=Material.GEMSTONE,
    ),

    # GEM("sapphire", "blue", 4, 1, 3000, 15, 9, GEMSTONE, CLR_BLUE, SAPPHIRE)
    ObjectEntry(
        name="sapphire",
        description="blue gem",
        class_=ObjectClass.GEM_CLASS,
        prob=4,
        weight=1,
        cost=3000,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_BLUE,
        material=Material.GEMSTONE,
    ),

    # GEM("black opal", "black", 3, 1, 2500, 15, 8, GEMSTONE, CLR_BLACK, BLACK_OPAL)
    ObjectEntry(
        name="black opal",
        description="black gem",
        class_=ObjectClass.GEM_CLASS,
        prob=3,
        weight=1,
        cost=2500,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_BLACK,
        material=Material.GEMSTONE,
    ),

    # GEM("emerald", "green", 5, 1, 2500, 15, 8, GEMSTONE, CLR_GREEN, EMERALD)
    ObjectEntry(
        name="emerald",
        description="green gem",
        class_=ObjectClass.GEM_CLASS,
        prob=5,
        weight=1,
        cost=2500,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_GREEN,
        material=Material.GEMSTONE,
    ),

    # GEM("turquoise", "green", 6, 1, 2000, 15, 6, GEMSTONE, CLR_GREEN, TURQUOISE)
    ObjectEntry(
        name="turquoise",
        description="green gem",
        class_=ObjectClass.GEM_CLASS,
        prob=6,
        weight=1,
        cost=2000,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_GREEN,
        material=Material.GEMSTONE,
    ),

    # GEM("citrine", "yellow", 4, 1, 1500, 15, 6, GEMSTONE, CLR_YELLOW, CITRINE)
    ObjectEntry(
        name="citrine",
        description="yellow gem",
        class_=ObjectClass.GEM_CLASS,
        prob=4,
        weight=1,
        cost=1500,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_YELLOW,
        material=Material.GEMSTONE,
    ),

    # GEM("aquamarine", "green", 6, 1, 1500, 15, 8, GEMSTONE, CLR_GREEN, AQUAMARINE)
    ObjectEntry(
        name="aquamarine",
        description="green gem",
        class_=ObjectClass.GEM_CLASS,
        prob=6,
        weight=1,
        cost=1500,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_GREEN,
        material=Material.GEMSTONE,
    ),

    # GEM("amber", "yellowish brown", 8, 1, 1000, 15, 2, GEMSTONE, CLR_BROWN, AMBER)
    ObjectEntry(
        name="amber",
        description="yellowish brown gem",
        class_=ObjectClass.GEM_CLASS,
        prob=8,
        weight=1,
        cost=1000,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_BROWN,
        material=Material.GEMSTONE,
    ),

    # GEM("topaz", "yellowish brown", 10, 1, 900, 15, 8, GEMSTONE, CLR_BROWN, TOPAZ)
    ObjectEntry(
        name="topaz",
        description="yellowish brown gem",
        class_=ObjectClass.GEM_CLASS,
        prob=10,
        weight=1,
        cost=900,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_BROWN,
        material=Material.GEMSTONE,
    ),

    # GEM("jet", "black", 6, 1, 850, 15, 7, GEMSTONE, CLR_BLACK, JET)
    ObjectEntry(
        name="jet",
        description="black gem",
        class_=ObjectClass.GEM_CLASS,
        prob=6,
        weight=1,
        cost=850,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_BLACK,
        material=Material.GEMSTONE,
    ),

    # GEM("opal", "white", 12, 1, 800, 15, 6, GEMSTONE, CLR_WHITE, OPAL)
    ObjectEntry(
        name="opal",
        description="white gem",
        class_=ObjectClass.GEM_CLASS,
        prob=12,
        weight=1,
        cost=800,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_WHITE,
        material=Material.GEMSTONE,
    ),

    # GEM("chrysoberyl", "yellow", 8, 1, 700, 15, 5, GEMSTONE, CLR_YELLOW, CHRYSOBERYL)
    ObjectEntry(
        name="chrysoberyl",
        description="yellow gem",
        class_=ObjectClass.GEM_CLASS,
        prob=8,
        weight=1,
        cost=700,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_YELLOW,
        material=Material.GEMSTONE,
    ),

    # GEM("garnet", "red", 12, 1, 700, 15, 7, GEMSTONE, CLR_RED, GARNET)
    ObjectEntry(
        name="garnet",
        description="red gem",
        class_=ObjectClass.GEM_CLASS,
        prob=12,
        weight=1,
        cost=700,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_RED,
        material=Material.GEMSTONE,
    ),

    # GEM("amethyst", "violet", 14, 1, 600, 15, 7, GEMSTONE, CLR_MAGENTA, AMETHYST)
    ObjectEntry(
        name="amethyst",
        description="violet gem",
        class_=ObjectClass.GEM_CLASS,
        prob=14,
        weight=1,
        cost=600,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_MAGENTA,
        material=Material.GEMSTONE,
    ),

    # GEM("jasper", "red", 15, 1, 500, 15, 7, GEMSTONE, CLR_RED, JASPER)
    ObjectEntry(
        name="jasper",
        description="red gem",
        class_=ObjectClass.GEM_CLASS,
        prob=15,
        weight=1,
        cost=500,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_RED,
        material=Material.GEMSTONE,
    ),

    # GEM("fluorite", "violet", 15, 1, 400, 15, 4, GEMSTONE, CLR_MAGENTA, FLUORITE)
    ObjectEntry(
        name="fluorite",
        description="violet gem",
        class_=ObjectClass.GEM_CLASS,
        prob=15,
        weight=1,
        cost=400,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_MAGENTA,
        material=Material.GEMSTONE,
    ),

    # GEM("obsidian", "black", 9, 1, 200, 15, 6, GEMSTONE, CLR_BLACK, OBSIDIAN)
    ObjectEntry(
        name="obsidian",
        description="black gem",
        class_=ObjectClass.GEM_CLASS,
        prob=9,
        weight=1,
        cost=200,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_BLACK,
        material=Material.GEMSTONE,
    ),

    # GEM("agate", "orange", 12, 1, 200, 15, 6, GEMSTONE, CLR_ORANGE, AGATE)
    ObjectEntry(
        name="agate",
        description="orange gem",
        class_=ObjectClass.GEM_CLASS,
        prob=12,
        weight=1,
        cost=200,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_ORANGE,
        material=Material.GEMSTONE,
    ),

    # GEM("jade", "green", 10, 1, 300, 15, 6, GEMSTONE, CLR_GREEN, JADE)
    ObjectEntry(
        name="jade",
        description="green gem",
        class_=ObjectClass.GEM_CLASS,
        prob=10,
        weight=1,
        cost=300,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.CLR_GREEN,
        material=Material.GEMSTONE,
    ),

    # GEM("worthless piece of white glass", "white", 77, 1, 0, 6, 5, GLASS, CLR_WHITE, WORTHLESS_WHITE_GLASS)
    ObjectEntry(
        name="worthless piece of white glass",
        description="white gem",
        class_=ObjectClass.GEM_CLASS,
        prob=77,
        weight=1,
        cost=0,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=6,
        color=Color.CLR_WHITE,
        material=Material.GLASS,
    ),

    # GEM("worthless piece of blue glass", "blue", 77, 1, 0, 6, 5, GLASS, CLR_BLUE, WORTHLESS_BLUE_GLASS)
    ObjectEntry(
        name="worthless piece of blue glass",
        description="blue gem",
        class_=ObjectClass.GEM_CLASS,
        prob=77,
        weight=1,
        cost=0,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=6,
        color=Color.CLR_BLUE,
        material=Material.GLASS,
    ),

    # GEM("worthless piece of red glass", "red", 77, 1, 0, 6, 5, GLASS, CLR_RED, WORTHLESS_RED_GLASS)
    ObjectEntry(
        name="worthless piece of red glass",
        description="red gem",
        class_=ObjectClass.GEM_CLASS,
        prob=77,
        weight=1,
        cost=0,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=6,
        color=Color.CLR_RED,
        material=Material.GLASS,
    ),

    # GEM("worthless piece of yellowish brown glass", "yellowish brown", 77, 1, 0, 6, 5, GLASS, CLR_BROWN, WORTHLESS_YELLOWBROWN_GLASS)
    ObjectEntry(
        name="worthless piece of yellowish brown glass",
        description="yellowish brown gem",
        class_=ObjectClass.GEM_CLASS,
        prob=77,
        weight=1,
        cost=0,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=6,
        color=Color.CLR_BROWN,
        material=Material.GLASS,
    ),

    # GEM("worthless piece of orange glass", "orange", 76, 1, 0, 6, 5, GLASS, CLR_ORANGE, WORTHLESS_ORANGE_GLASS)
    ObjectEntry(
        name="worthless piece of orange glass",
        description="orange gem",
        class_=ObjectClass.GEM_CLASS,
        prob=76,
        weight=1,
        cost=0,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=6,
        color=Color.CLR_ORANGE,
        material=Material.GLASS,
    ),

    # GEM("worthless piece of yellow glass", "yellow", 77, 1, 0, 6, 5, GLASS, CLR_YELLOW, WORTHLESS_YELLOW_GLASS)
    ObjectEntry(
        name="worthless piece of yellow glass",
        description="yellow gem",
        class_=ObjectClass.GEM_CLASS,
        prob=77,
        weight=1,
        cost=0,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=6,
        color=Color.CLR_YELLOW,
        material=Material.GLASS,
    ),

    # GEM("worthless piece of black glass", "black", 76, 1, 0, 6, 5, GLASS, CLR_BLACK, WORTHLESS_BLACK_GLASS)
    ObjectEntry(
        name="worthless piece of black glass",
        description="black gem",
        class_=ObjectClass.GEM_CLASS,
        prob=76,
        weight=1,
        cost=0,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=6,
        color=Color.CLR_BLACK,
        material=Material.GLASS,
    ),

    # GEM("worthless piece of green glass", "green", 77, 1, 0, 6, 5, GLASS, CLR_GREEN, WORTHLESS_GREEN_GLASS)
    ObjectEntry(
        name="worthless piece of green glass",
        description="green gem",
        class_=ObjectClass.GEM_CLASS,
        prob=77,
        weight=1,
        cost=0,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=6,
        color=Color.CLR_GREEN,
        material=Material.GLASS,
    ),

    # GEM("worthless piece of violet glass", "violet", 77, 1, 0, 6, 5, GLASS, CLR_MAGENTA, WORTHLESS_VIOLET_GLASS)
    ObjectEntry(
        name="worthless piece of violet glass",
        description="violet gem",
        class_=ObjectClass.GEM_CLASS,
        prob=77,
        weight=1,
        cost=0,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=6,
        color=Color.CLR_MAGENTA,
        material=Material.GLASS,
    ),

    # -----------------------------------------------------------------------
    # ROCK entries (gray stones)
    # Source: vendor/nethack/include/objects.h lines 1598–1607
    # ROCK(name, desc, kn, prob, wt, gval, sdam, ldam, mgc, nutr, mohs, glass, colr, sn)
    #   → cost=gval, sdam=(sdam,sdam), ldam=(ldam,ldam), nutrition=nutr
    #   → glass arg = material (MINERAL), colr = CLR_GRAY
    #   → NoDes desc → None
    # -----------------------------------------------------------------------

    # ROCK("luckstone", "gray", 0, 10, 10, 60, 3, 3, 1, 10, 7, MINERAL, CLR_GRAY, LUCKSTONE)
    ObjectEntry(
        name="luckstone",
        description="gray stone",
        class_=ObjectClass.GEM_CLASS,
        prob=10,
        weight=10,
        cost=60,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=10,
        color=Color.CLR_GRAY,
        material=Material.MINERAL,
    ),

    # ROCK("loadstone", "gray", 0, 10, 500, 1, 3, 3, 1, 10, 6, MINERAL, CLR_GRAY, LOADSTONE)
    ObjectEntry(
        name="loadstone",
        description="gray stone",
        class_=ObjectClass.GEM_CLASS,
        prob=10,
        weight=500,
        cost=1,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=10,
        color=Color.CLR_GRAY,
        material=Material.MINERAL,
    ),

    # ROCK("touchstone", "gray", 0, 8, 10, 45, 3, 3, 1, 10, 6, MINERAL, CLR_GRAY, TOUCHSTONE)
    ObjectEntry(
        name="touchstone",
        description="gray stone",
        class_=ObjectClass.GEM_CLASS,
        prob=8,
        weight=10,
        cost=45,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=10,
        color=Color.CLR_GRAY,
        material=Material.MINERAL,
    ),

    # ROCK("flint", "gray", 0, 10, 10, 1, 6, 6, 0, 10, 7, MINERAL, CLR_GRAY, FLINT)
    ObjectEntry(
        name="flint",
        description="gray stone",
        class_=ObjectClass.GEM_CLASS,
        prob=10,
        weight=10,
        cost=1,
        sdam=(6, 6),
        ldam=(6, 6),
        oc1=0,
        oc2=0,
        nutrition=10,
        color=Color.CLR_GRAY,
        material=Material.MINERAL,
    ),

    # ROCK("rock", NoDes, 1, 100, 10, 0, 3, 3, 0, 10, 7, MINERAL, CLR_GRAY, ROCK)
    ObjectEntry(
        name="rock",
        description=None,
        class_=ObjectClass.GEM_CLASS,
        prob=100,
        weight=10,
        cost=0,
        sdam=(3, 3),
        ldam=(3, 3),
        oc1=0,
        oc2=0,
        nutrition=10,
        color=Color.CLR_GRAY,
        material=Material.MINERAL,
    ),
)
