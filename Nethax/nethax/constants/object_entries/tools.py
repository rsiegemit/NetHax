"""All tools — vendor/nethack/include/objects.h TOOL/CONTAINER/INSTRUMENT entries."""
from Nethax.nethax.constants.objects import (
    ObjectEntry,
    ObjectClass,
    Color,
    Material,
)

# Color aliases from vendor/nethack/include/color.h
# HI_METAL  = CLR_CYAN           = 6
# HI_COPPER = CLR_YELLOW         = 11
# HI_SILVER = CLR_GRAY           = 7
# HI_GOLD   = CLR_YELLOW         = 11
# HI_LEATHER= CLR_BROWN          = 3
# HI_CLOTH  = CLR_BROWN          = 3
# HI_WOOD   = CLR_BROWN          = 3
# HI_GLASS  = CLR_BRIGHT_CYAN    = 14
# HI_MINERAL= CLR_GRAY           = 7

ENTRIES: tuple[ObjectEntry, ...] = (

    # --- CONTAINERS ---

    # CONTAINER("large box", NoDes, 1, 0, 0, 40, 350, 8, WOOD, HI_WOOD, LARGE_BOX)
    ObjectEntry(
        name="large box",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=40,
        weight=350,
        cost=8,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=350,
        color=Color.HI_WOOD,
        material=Material.WOOD,
    ),

    # CONTAINER("chest", NoDes, 1, 0, 0, 35, 600, 16, WOOD, HI_WOOD, CHEST)
    ObjectEntry(
        name="chest",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=35,
        weight=600,
        cost=16,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=600,
        color=Color.HI_WOOD,
        material=Material.WOOD,
    ),

    # CONTAINER("ice box", NoDes, 1, 0, 0, 5, 900, 42, PLASTIC, CLR_WHITE, ICE_BOX)
    ObjectEntry(
        name="ice box",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=5,
        weight=900,
        cost=42,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=900,
        color=Color.CLR_WHITE,
        material=Material.PLASTIC,
    ),

    # CONTAINER("sack", "bag", 0, 0, 0, 35, 15, 2, CLOTH, HI_CLOTH, SACK)
    ObjectEntry(
        name="sack",
        description="bag",
        class_=ObjectClass.TOOL_CLASS,
        prob=35,
        weight=15,
        cost=2,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.HI_CLOTH,
        material=Material.CLOTH,
    ),

    # CONTAINER("oilskin sack", "bag", 0, 0, 0, 5, 15, 100, CLOTH, HI_CLOTH, OILSKIN_SACK)
    ObjectEntry(
        name="oilskin sack",
        description="bag",
        class_=ObjectClass.TOOL_CLASS,
        prob=5,
        weight=15,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.HI_CLOTH,
        material=Material.CLOTH,
    ),

    # CONTAINER("bag of holding", "bag", 0, 1, 0, 20, 15, 100, CLOTH, HI_CLOTH, BAG_OF_HOLDING)
    ObjectEntry(
        name="bag of holding",
        description="bag",
        class_=ObjectClass.TOOL_CLASS,
        prob=20,
        weight=15,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.HI_CLOTH,
        material=Material.CLOTH,
    ),

    # CONTAINER("bag of tricks", "bag", 0, 1, 1, 20, 15, 100, CLOTH, HI_CLOTH, BAG_OF_TRICKS)
    ObjectEntry(
        name="bag of tricks",
        description="bag",
        class_=ObjectClass.TOOL_CLASS,
        prob=20,
        weight=15,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.HI_CLOTH,
        material=Material.CLOTH,
    ),

    # --- LOCK OPENING TOOLS ---

    # TOOL("skeleton key", "key", 0, 0, 0, 0, 80, 3, 10, IRON, HI_METAL, SKELETON_KEY)
    ObjectEntry(
        name="skeleton key",
        description="key",
        class_=ObjectClass.TOOL_CLASS,
        prob=80,
        weight=3,
        cost=10,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=3,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # TOOL("lock pick", NoDes, 1, 0, 0, 0, 60, 4, 20, IRON, HI_METAL, LOCK_PICK)
    ObjectEntry(
        name="lock pick",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=60,
        weight=4,
        cost=20,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=4,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # TOOL("credit card", NoDes, 1, 0, 0, 0, 15, 1, 10, PLASTIC, CLR_WHITE, CREDIT_CARD)
    ObjectEntry(
        name="credit card",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=15,
        weight=1,
        cost=10,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=1,
        color=Color.CLR_WHITE,
        material=Material.PLASTIC,
    ),

    # --- LIGHT SOURCES ---

    # TOOL("tallow candle", "candle", 0, 1, 0, 0, 20, 2, 10, WAX, CLR_WHITE, TALLOW_CANDLE)
    ObjectEntry(
        name="tallow candle",
        description="candle",
        class_=ObjectClass.TOOL_CLASS,
        prob=20,
        weight=2,
        cost=10,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=2,
        color=Color.CLR_WHITE,
        material=Material.WAX,
    ),

    # TOOL("wax candle", "candle", 0, 1, 0, 0, 5, 2, 20, WAX, CLR_WHITE, WAX_CANDLE)
    ObjectEntry(
        name="wax candle",
        description="candle",
        class_=ObjectClass.TOOL_CLASS,
        prob=5,
        weight=2,
        cost=20,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=2,
        color=Color.CLR_WHITE,
        material=Material.WAX,
    ),

    # TOOL("brass lantern", NoDes, 1, 0, 0, 0, 30, 30, 12, COPPER, CLR_YELLOW, BRASS_LANTERN)
    ObjectEntry(
        name="brass lantern",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=30,
        weight=30,
        cost=12,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=30,
        color=Color.CLR_YELLOW,
        material=Material.COPPER,
    ),

    # TOOL("oil lamp", "lamp", 0, 0, 0, 0, 45, 20, 10, COPPER, CLR_YELLOW, OIL_LAMP)
    ObjectEntry(
        name="oil lamp",
        description="lamp",
        class_=ObjectClass.TOOL_CLASS,
        prob=45,
        weight=20,
        cost=10,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=Color.CLR_YELLOW,
        material=Material.COPPER,
    ),

    # TOOL("magic lamp", "lamp", 0, 0, 1, 0, 15, 20, 50, COPPER, CLR_YELLOW, MAGIC_LAMP)
    ObjectEntry(
        name="magic lamp",
        description="lamp",
        class_=ObjectClass.TOOL_CLASS,
        prob=15,
        weight=20,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=Color.CLR_YELLOW,
        material=Material.COPPER,
    ),

    # --- OTHER TOOLS ---

    # TOOL("expensive camera", NoDes, 1, 0, 0, 1, 15, 12, 200, PLASTIC, CLR_BLACK, EXPENSIVE_CAMERA)
    ObjectEntry(
        name="expensive camera",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=15,
        weight=12,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=12,
        color=Color.CLR_BLACK,
        material=Material.PLASTIC,
    ),

    # TOOL("mirror", "looking glass", 0, 0, 0, 0, 45, 13, 10, GLASS, HI_SILVER, MIRROR)
    ObjectEntry(
        name="mirror",
        description="looking glass",
        class_=ObjectClass.TOOL_CLASS,
        prob=45,
        weight=13,
        cost=10,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=13,
        color=Color.HI_SILVER,
        material=Material.GLASS,
    ),

    # TOOL("crystal ball", "glass orb", 0, 0, 1, 1, 15, 150, 60, GLASS, HI_GLASS, CRYSTAL_BALL)
    ObjectEntry(
        name="crystal ball",
        description="glass orb",
        class_=ObjectClass.TOOL_CLASS,
        prob=15,
        weight=150,
        cost=60,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=150,
        color=Color.HI_GLASS,
        material=Material.GLASS,
    ),

    # --- EYEWEAR ---

    # EYEWEAR("lenses", NoDes, 1, 0, 5, 3, 80, GLASS, HI_GLASS, LENSES)
    ObjectEntry(
        name="lenses",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=5,
        weight=3,
        cost=80,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=3,
        color=Color.HI_GLASS,
        material=Material.GLASS,
    ),

    # EYEWEAR("blindfold", NoDes, 1, BLINDED, 50, 2, 20, CLOTH, CLR_BLACK, BLINDFOLD)
    ObjectEntry(
        name="blindfold",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=50,
        weight=2,
        cost=20,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=2,
        color=Color.CLR_BLACK,
        material=Material.CLOTH,
    ),

    # EYEWEAR("towel", NoDes, 1, BLINDED, 50, 5, 50, CLOTH, CLR_MAGENTA, TOWEL)
    ObjectEntry(
        name="towel",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=50,
        weight=5,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=5,
        color=Color.CLR_MAGENTA,
        material=Material.CLOTH,
    ),

    # --- STILL OTHER TOOLS ---

    # TOOL("saddle", NoDes, 1, 0, 0, 0, 5, 200, 150, LEATHER, HI_LEATHER, SADDLE)
    ObjectEntry(
        name="saddle",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=5,
        weight=200,
        cost=150,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=200,
        color=Color.HI_LEATHER,
        material=Material.LEATHER,
    ),

    # TOOL("leash", NoDes, 1, 0, 0, 0, 65, 12, 20, LEATHER, HI_LEATHER, LEASH)
    ObjectEntry(
        name="leash",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=65,
        weight=12,
        cost=20,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=12,
        color=Color.HI_LEATHER,
        material=Material.LEATHER,
    ),

    # TOOL("stethoscope", NoDes, 1, 0, 0, 0, 25, 4, 75, IRON, HI_METAL, STETHOSCOPE)
    ObjectEntry(
        name="stethoscope",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=25,
        weight=4,
        cost=75,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=4,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # TOOL("tinning kit", NoDes, 1, 0, 0, 1, 15, 100, 30, IRON, HI_METAL, TINNING_KIT)
    ObjectEntry(
        name="tinning kit",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=15,
        weight=100,
        cost=30,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=100,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # TOOL("tin opener", NoDes, 1, 0, 0, 0, 35, 4, 30, IRON, HI_METAL, TIN_OPENER)
    ObjectEntry(
        name="tin opener",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=35,
        weight=4,
        cost=30,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=4,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # TOOL("can of grease", NoDes, 1, 0, 0, 1, 15, 15, 20, IRON, HI_METAL, CAN_OF_GREASE)
    ObjectEntry(
        name="can of grease",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=15,
        weight=15,
        cost=20,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # TOOL("figurine", NoDes, 1, 0, 1, 0, 25, 50, 80, MINERAL, HI_MINERAL, FIGURINE)
    ObjectEntry(
        name="figurine",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=25,
        weight=50,
        cost=80,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=50,
        color=Color.HI_MINERAL,
        material=Material.MINERAL,
    ),

    # TOOL("magic marker", NoDes, 1, 0, 1, 1, 15, 2, 50, PLASTIC, CLR_RED, MAGIC_MARKER)
    ObjectEntry(
        name="magic marker",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=15,
        weight=2,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=2,
        color=Color.CLR_RED,
        material=Material.PLASTIC,
    ),

    # --- TRAPS ---

    # TOOL("land mine", NoDes, 1, 0, 0, 0, 0, 200, 180, IRON, CLR_RED, LAND_MINE)
    ObjectEntry(
        name="land mine",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=0,
        weight=200,
        cost=180,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=200,
        color=Color.CLR_RED,
        material=Material.IRON,
    ),

    # TOOL("beartrap", NoDes, 1, 0, 0, 0, 0, 200, 60, IRON, HI_METAL, BEARTRAP)
    ObjectEntry(
        name="beartrap",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=0,
        weight=200,
        cost=60,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=200,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # --- INSTRUMENTS ---

    # TOOL("tin whistle", "whistle", 0, 0, 0, 0, 100, 3, 10, METAL, HI_METAL, TIN_WHISTLE)
    ObjectEntry(
        name="tin whistle",
        description="whistle",
        class_=ObjectClass.TOOL_CLASS,
        prob=100,
        weight=3,
        cost=10,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=3,
        color=Color.HI_METAL,
        material=Material.METAL,
    ),

    # TOOL("magic whistle", "whistle", 0, 0, 1, 0, 30, 3, 10, METAL, HI_METAL, MAGIC_WHISTLE)
    ObjectEntry(
        name="magic whistle",
        description="whistle",
        class_=ObjectClass.TOOL_CLASS,
        prob=30,
        weight=3,
        cost=10,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=3,
        color=Color.HI_METAL,
        material=Material.METAL,
    ),

    # TOOL("wooden flute", "flute", 0, 0, 0, 0, 4, 5, 12, WOOD, HI_WOOD, WOODEN_FLUTE)
    ObjectEntry(
        name="wooden flute",
        description="flute",
        class_=ObjectClass.TOOL_CLASS,
        prob=4,
        weight=5,
        cost=12,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=5,
        color=Color.HI_WOOD,
        material=Material.WOOD,
    ),

    # TOOL("magic flute", "flute", 0, 0, 1, 1, 2, 5, 36, WOOD, HI_WOOD, MAGIC_FLUTE)
    ObjectEntry(
        name="magic flute",
        description="flute",
        class_=ObjectClass.TOOL_CLASS,
        prob=2,
        weight=5,
        cost=36,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=5,
        color=Color.HI_WOOD,
        material=Material.WOOD,
    ),

    # TOOL("tooled horn", "horn", 0, 0, 0, 0, 5, 18, 15, BONE, CLR_WHITE, TOOLED_HORN)
    ObjectEntry(
        name="tooled horn",
        description="horn",
        class_=ObjectClass.TOOL_CLASS,
        prob=5,
        weight=18,
        cost=15,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=18,
        color=Color.CLR_WHITE,
        material=Material.BONE,
    ),

    # TOOL("frost horn", "horn", 0, 0, 1, 1, 2, 18, 50, BONE, CLR_WHITE, FROST_HORN)
    ObjectEntry(
        name="frost horn",
        description="horn",
        class_=ObjectClass.TOOL_CLASS,
        prob=2,
        weight=18,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=18,
        color=Color.CLR_WHITE,
        material=Material.BONE,
    ),

    # TOOL("fire horn", "horn", 0, 0, 1, 1, 2, 18, 50, BONE, CLR_WHITE, FIRE_HORN)
    ObjectEntry(
        name="fire horn",
        description="horn",
        class_=ObjectClass.TOOL_CLASS,
        prob=2,
        weight=18,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=18,
        color=Color.CLR_WHITE,
        material=Material.BONE,
    ),

    # TOOL("horn of plenty", "horn", 0, 0, 1, 1, 2, 18, 50, BONE, CLR_WHITE, HORN_OF_PLENTY)
    ObjectEntry(
        name="horn of plenty",
        description="horn",
        class_=ObjectClass.TOOL_CLASS,
        prob=2,
        weight=18,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=18,
        color=Color.CLR_WHITE,
        material=Material.BONE,
    ),

    # TOOL("wooden harp", "harp", 0, 0, 0, 0, 4, 30, 50, WOOD, HI_WOOD, WOODEN_HARP)
    ObjectEntry(
        name="wooden harp",
        description="harp",
        class_=ObjectClass.TOOL_CLASS,
        prob=4,
        weight=30,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=30,
        color=Color.HI_WOOD,
        material=Material.WOOD,
    ),

    # TOOL("magic harp", "harp", 0, 0, 1, 1, 2, 30, 50, WOOD, HI_WOOD, MAGIC_HARP)
    ObjectEntry(
        name="magic harp",
        description="harp",
        class_=ObjectClass.TOOL_CLASS,
        prob=2,
        weight=30,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=30,
        color=Color.HI_WOOD,
        material=Material.WOOD,
    ),

    # TOOL("bell", NoDes, 1, 0, 0, 0, 2, 30, 50, COPPER, HI_COPPER, BELL)
    ObjectEntry(
        name="bell",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=2,
        weight=30,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=30,
        color=Color.HI_COPPER,
        material=Material.COPPER,
    ),

    # TOOL("bugle", NoDes, 1, 0, 0, 0, 4, 10, 15, COPPER, HI_COPPER, BUGLE)
    ObjectEntry(
        name="bugle",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=4,
        weight=10,
        cost=15,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=10,
        color=Color.HI_COPPER,
        material=Material.COPPER,
    ),

    # TOOL("leather drum", "drum", 0, 0, 0, 0, 4, 25, 25, LEATHER, HI_LEATHER, LEATHER_DRUM)
    ObjectEntry(
        name="leather drum",
        description="drum",
        class_=ObjectClass.TOOL_CLASS,
        prob=4,
        weight=25,
        cost=25,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=25,
        color=Color.HI_LEATHER,
        material=Material.LEATHER,
    ),

    # TOOL("drum of earthquake", "drum", 0, 0, 1, 1, 2, 25, 25, LEATHER, HI_LEATHER, DRUM_OF_EARTHQUAKE)
    ObjectEntry(
        name="drum of earthquake",
        description="drum",
        class_=ObjectClass.TOOL_CLASS,
        prob=2,
        weight=25,
        cost=25,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=25,
        color=Color.HI_LEATHER,
        material=Material.LEATHER,
    ),

    # --- WEPTOOLS (tools usable as weapons) ---

    # WEPTOOL("pick-axe", NoDes, 1, 0, 0, 20, 100, 50, 6, 3, WHACK, P_PICK_AXE, IRON, HI_METAL, PICK_AXE)
    ObjectEntry(
        name="pick-axe",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=20,
        weight=100,
        cost=50,
        sdam=(1, 6),
        ldam=(1, 3),
        oc1=0,
        oc2=0,
        nutrition=100,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEPTOOL("grappling hook", NoDes, 1, 0, 0, 5, 30, 50, 2, 6, WHACK, P_FLAIL, IRON, HI_METAL, GRAPPLING_HOOK)
    ObjectEntry(
        name="grappling hook",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=5,
        weight=30,
        cost=50,
        sdam=(1, 2),
        ldam=(1, 6),
        oc1=0,
        oc2=0,
        nutrition=30,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEPTOOL("unicorn horn", NoDes, 1, 1, 1, 0, 20, 100, 12, 12, PIERCE, P_UNICORN_HORN, BONE, CLR_WHITE, UNICORN_HORN)
    ObjectEntry(
        name="unicorn horn",
        description=None,
        class_=ObjectClass.TOOL_CLASS,
        prob=0,
        weight=20,
        cost=100,
        sdam=(1, 12),
        ldam=(1, 12),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=Color.CLR_WHITE,
        material=Material.BONE,
    ),

    # --- UNIQUE TOOLS ---

    # OBJECT(OBJ("Candelabrum of Invocation", "candelabrum"), ..., TOOL_CLASS, 0, 0, 10, 5000, ..., 200, HI_GOLD, ...)
    ObjectEntry(
        name="Candelabrum of Invocation",
        description="candelabrum",
        class_=ObjectClass.TOOL_CLASS,
        prob=0,
        weight=10,
        cost=5000,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=200,
        color=Color.HI_GOLD,
        material=Material.GOLD,
    ),

    # OBJECT(OBJ("Bell of Opening", "silver bell"), ..., TOOL_CLASS, 0, 0, 10, 5000, ..., 50, HI_SILVER, ...)
    ObjectEntry(
        name="Bell of Opening",
        description="silver bell",
        class_=ObjectClass.TOOL_CLASS,
        prob=0,
        weight=10,
        cost=5000,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=50,
        color=Color.HI_SILVER,
        material=Material.SILVER,
    ),
)
