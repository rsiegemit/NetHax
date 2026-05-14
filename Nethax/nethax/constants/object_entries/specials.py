"""Special, ball, chain, venom, coins — vendor/nethack/include/objects.h."""
from Nethax.nethax.constants.objects import (
    ObjectEntry, ObjectClass, Material, Color,
)

# Color values not in the Color enum (from vendor/nethack/include/color.h):
#   HI_GOLD    = CLR_YELLOW  = 11
#   HI_ORGANIC = CLR_BROWN   =  3
#   HI_PAPER   = CLR_WHITE   = 15
#   HI_MINERAL = CLR_GRAY    =  7
# (HI_METAL = CLR_CYAN = 6 and HI_SILVER = CLR_GRAY = 7 are already aliases
#  in Color, but we use the integer literals here for the unlisted ones.)

ENTRIES = (

    # ------------------------------------------------------------------
    # Coins (COIN_CLASS)
    # COIN("gold piece", 1000, GOLD, 1, GOLD_PIECE)
    # → OBJECT(OBJ("gold piece", NoDes),
    #          BITS(1,1,0,0,0,0,0,0,0,0,0,P_NONE,GOLD),
    #          0, COIN_CLASS, 1000, 0, 1, 1, 0, 0, 0, 0, 0, HI_GOLD, GOLD_PIECE)
    # ------------------------------------------------------------------
    ObjectEntry(
        name="gold piece",
        description=None,
        class_=ObjectClass.COIN_CLASS,
        prob=1000,
        weight=1,
        cost=1,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=0,
        color=11,           # HI_GOLD = CLR_YELLOW
        material=Material.GOLD,
    ),

    # ------------------------------------------------------------------
    # Ball (BALL_CLASS)
    # OBJECT(OBJ("heavy iron ball", NoDes),
    #        BITS(1,0,0,0,0,0,0,0,0,0,WHACK,P_NONE,IRON), 0,
    #        BALL_CLASS, 1000, 0, 480, 10, 25, 25, 0, 0, 200, HI_METAL,
    #        HEAVY_IRON_BALL)
    # ------------------------------------------------------------------
    ObjectEntry(
        name="heavy iron ball",
        description=None,
        class_=ObjectClass.BALL_CLASS,
        prob=1000,
        weight=480,
        cost=10,
        sdam=(1, 25),
        ldam=(1, 25),
        oc1=0,
        oc2=0,
        nutrition=200,
        color=Color.HI_METAL,   # CLR_CYAN = 6
        material=Material.IRON,
    ),

    # ------------------------------------------------------------------
    # Chain (CHAIN_CLASS)
    # OBJECT(OBJ("iron chain", NoDes),
    #        BITS(1,0,0,0,0,0,0,0,0,0,WHACK,P_NONE,IRON), 0,
    #        CHAIN_CLASS, 1000, 0, 120, 0, 4, 4, 0, 0, 200, HI_METAL,
    #        IRON_CHAIN)
    # ------------------------------------------------------------------
    ObjectEntry(
        name="iron chain",
        description=None,
        class_=ObjectClass.CHAIN_CLASS,
        prob=1000,
        weight=120,
        cost=0,
        sdam=(1, 4),
        ldam=(1, 4),
        oc1=0,
        oc2=0,
        nutrition=200,
        color=Color.HI_METAL,   # CLR_CYAN = 6
        material=Material.IRON,
    ),

    # ------------------------------------------------------------------
    # Venom (VENOM_CLASS)
    # OBJECT(OBJ("splash of blinding venom", "splash of venom"),
    #        BITS(0,1,0,0,0,0,0,1,0,0,0,P_NONE,LIQUID), 0,
    #        VENOM_CLASS, 500, 0, 1, 0, 0, 0, 0, 0, 0, HI_ORGANIC,
    #        BLINDING_VENOM)
    # ------------------------------------------------------------------
    ObjectEntry(
        name="splash of blinding venom",
        description="splash of venom",
        class_=ObjectClass.VENOM_CLASS,
        prob=500,
        weight=1,
        cost=0,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=0,
        nutrition=0,
        color=3,            # HI_ORGANIC = CLR_BROWN
        material=Material.LIQUID,
    ),

    # OBJECT(OBJ("splash of acid venom", "splash of venom"),
    #        BITS(0,1,0,0,0,0,0,1,0,0,0,P_NONE,LIQUID), 0,
    #        VENOM_CLASS, 500, 0, 1, 0, 6, 6, 0, 0, 0, HI_ORGANIC,
    #        ACID_VENOM)
    ObjectEntry(
        name="splash of acid venom",
        description="splash of venom",
        class_=ObjectClass.VENOM_CLASS,
        prob=500,
        weight=1,
        cost=0,
        sdam=(1, 6),
        ldam=(1, 6),
        oc1=0,
        oc2=0,
        nutrition=0,
        color=3,            # HI_ORGANIC = CLR_BROWN
        material=Material.LIQUID,
    ),

    # ------------------------------------------------------------------
    # Special objects
    # ------------------------------------------------------------------

    # OBJECT(OBJ("cheap plastic imitation of the Amulet of Yendor",
    #            "Amulet of Yendor"),
    #        BITS(0,0,1,0,0,0,0,0,0,0,0,0,PLASTIC),
    #        0, AMULET_CLASS, 0, 0, 20, 0, 0, 0, 0, 0, 1, HI_METAL,
    #        FAKE_AMULET_OF_YENDOR)
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
        color=Color.HI_METAL,   # CLR_CYAN = 6
        material=Material.PLASTIC,
    ),

    # OBJECT(OBJ("Amulet of Yendor",  /* description == name */
    #            "Amulet of Yendor"),
    #        BITS(0,0,1,0,1,0,1,1,0,0,0,0,MITHRIL),
    #        0, AMULET_CLASS, 0, 0, 20, 30000, 0, 0, 0, 0, 20, HI_METAL,
    #        AMULET_OF_YENDOR)
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
        color=Color.HI_METAL,   # CLR_CYAN = 6
        material=Material.MITHRIL,
    ),

    # OBJECT(OBJ("Candelabrum of Invocation", "candelabrum"),
    #        BITS(0,0,1,0,1,0,1,1,0,0,0,P_NONE,GOLD),
    #        0, TOOL_CLASS, 0, 0, 10, 5000, 0, 0, 0, 0, 200, HI_GOLD,
    #        CANDELABRUM_OF_INVOCATION)
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
        color=11,           # HI_GOLD = CLR_YELLOW
        material=Material.GOLD,
    ),

    # OBJECT(OBJ("Bell of Opening", "silver bell"),
    #        BITS(0,0,1,0,1,1,1,1,0,0,0,P_NONE,SILVER),
    #        0, TOOL_CLASS, 0, 0, 10, 5000, 0, 0, 0, 0, 50, HI_SILVER,
    #        BELL_OF_OPENING)
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
        color=Color.HI_SILVER,  # CLR_GRAY = 7
        material=Material.SILVER,
    ),

    # OBJECT(OBJ("Book of the Dead", "papyrus"),
    #        BITS(0,0,1,0,1,0,1,1,0,0,0,P_NONE,PAPER),
    #        0, SPBOOK_CLASS, 0, 0, 50, 10000, 0, 0, 0, 7, 20, HI_PAPER,
    #        SPE_BOOK_OF_THE_DEAD)
    ObjectEntry(
        name="Book of the Dead",
        description="papyrus",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=0,
        weight=50,
        cost=10000,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,
        oc2=7,              # spell level
        nutrition=20,
        color=15,           # HI_PAPER = CLR_WHITE
        material=Material.PAPER,
    ),

    # OBJECT(OBJ("boulder", NoDes),
    #        BITS(1,0,0,0,0,0,0,0,1,0,0,P_NONE,MINERAL), 0,
    #        ROCK_CLASS, 100, 0, 6000, 0, 20, 20, 0, 0, 2000, HI_MINERAL,
    #        BOULDER)
    ObjectEntry(
        name="boulder",
        description=None,
        class_=ObjectClass.ROCK_CLASS,
        prob=100,
        weight=6000,
        cost=0,
        sdam=(1, 20),
        ldam=(1, 20),
        oc1=0,
        oc2=0,
        nutrition=2000,
        color=7,            # HI_MINERAL = CLR_GRAY
        material=Material.MINERAL,
    ),

    # OBJECT(OBJ("statue", NoDes),
    #        BITS(1,0,0,1,0,0,0,0,0,0,0,P_NONE,MINERAL), 0,
    #        ROCK_CLASS, 900, 0, 2500, 0, 20, 20, 0, 0, 2500, CLR_WHITE,
    #        STATUE)
    ObjectEntry(
        name="statue",
        description=None,
        class_=ObjectClass.ROCK_CLASS,
        prob=900,
        weight=2500,
        cost=0,
        sdam=(1, 20),
        ldam=(1, 20),
        oc1=0,
        oc2=0,
        nutrition=2500,
        color=Color.CLR_WHITE,
        material=Material.MINERAL,
    ),

)
