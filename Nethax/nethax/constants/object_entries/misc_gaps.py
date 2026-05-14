"""Misc object gaps — generics, missing scrolls/wands/tools, and NLE-canonical
raw-name variants for potions/scrolls/wands that already exist with verbose
"potion of X" names. vendor/nethack/include/objects.h."""
from Nethax.nethax.constants.objects import ObjectEntry, ObjectClass, Color, Material

# Color int aliases used below (from vendor/nethack/include/color.h)
# HI_METAL  = CLR_CYAN           = 6
# HI_WOOD   = CLR_BROWN          = 3
# HI_GLASS  = CLR_BRIGHT_CYAN    = 14
# HI_MINERAL= CLR_GRAY           = 7
# HI_COPPER = CLR_YELLOW         = 11
# HI_SILVER = CLR_GRAY           = 7
# HI_PAPER  = CLR_WHITE          = 15
# HI_ORGANIC= CLR_BROWN          = 3

ENTRIES: tuple[ObjectEntry, ...] = (

    # -----------------------------------------------------------------------
    # GENERIC placeholder objects (GENERIC macro, objects.h lines 72-107)
    # GENERIC(desc, class, gen_enum) →
    #   OBJECT(OBJ("generic <desc>", <desc>),
    #          BITS(0,0,0,0,0,0,0,1,0,0,0,P_NONE,0),
    #          0, <class>, 0,0,0,0,0,0,0,0,0, CLR_GRAY, gen_enum)
    # -----------------------------------------------------------------------

    # [1] GENERIC("strange", ILLOBJ_CLASS, GENERIC_ILLOBJ)
    ObjectEntry(
        name="generic strange", description="strange",
        class_=ObjectClass.ILLOBJ_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [2] GENERIC("weapon", WEAPON_CLASS, GENERIC_WEAPON)
    ObjectEntry(
        name="generic weapon", description="weapon",
        class_=ObjectClass.WEAPON_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [3] GENERIC("armor", ARMOR_CLASS, GENERIC_ARMOR)
    ObjectEntry(
        name="generic armor", description="armor",
        class_=ObjectClass.ARMOR_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [4] GENERIC("ring", RING_CLASS, GENERIC_RING)
    ObjectEntry(
        name="generic ring", description="ring",
        class_=ObjectClass.RING_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [5] GENERIC("amulet", AMULET_CLASS, GENERIC_AMULET)
    ObjectEntry(
        name="generic amulet", description="amulet",
        class_=ObjectClass.AMULET_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [6] GENERIC("tool", TOOL_CLASS, GENERIC_TOOL)
    ObjectEntry(
        name="generic tool", description="tool",
        class_=ObjectClass.TOOL_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [7] GENERIC("food", FOOD_CLASS, GENERIC_FOOD)
    ObjectEntry(
        name="generic food", description="food",
        class_=ObjectClass.FOOD_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [8] GENERIC("potion", POTION_CLASS, GENERIC_POTION)
    ObjectEntry(
        name="generic potion", description="potion",
        class_=ObjectClass.POTION_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [9] GENERIC("scroll", SCROLL_CLASS, GENERIC_SCROLL)
    ObjectEntry(
        name="generic scroll", description="scroll",
        class_=ObjectClass.SCROLL_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [10] GENERIC("spellbook", SPBOOK_CLASS, GENERIC_SPBOOK)
    ObjectEntry(
        name="generic spellbook", description="spellbook",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [11] GENERIC("wand", WAND_CLASS, GENERIC_WAND)
    ObjectEntry(
        name="generic wand", description="wand",
        class_=ObjectClass.WAND_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [12] GENERIC("coin", COIN_CLASS, GENERIC_COIN)
    ObjectEntry(
        name="generic coin", description="coin",
        class_=ObjectClass.COIN_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [13] GENERIC("gem", GEM_CLASS, GENERIC_GEM)
    ObjectEntry(
        name="generic gem", description="gem",
        class_=ObjectClass.GEM_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [14] GENERIC("large rock", ROCK_CLASS, GENERIC_ROCK)
    ObjectEntry(
        name="generic large rock", description="large rock",
        class_=ObjectClass.ROCK_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [15] GENERIC("iron ball", BALL_CLASS, GENERIC_BALL)
    ObjectEntry(
        name="generic iron ball", description="iron ball",
        class_=ObjectClass.BALL_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [16] GENERIC("iron chain", CHAIN_CLASS, GENERIC_CHAIN)
    ObjectEntry(
        name="generic iron chain", description="iron chain",
        class_=ObjectClass.CHAIN_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),
    # [17] GENERIC("venom", VENOM_CLASS, GENERIC_VENOM)
    ObjectEntry(
        name="generic venom", description="venom",
        class_=ObjectClass.VENOM_CLASS,
        prob=0, weight=0, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=0, color=Color.CLR_GRAY, material=Material.NO_MATERIAL,
    ),

    # -----------------------------------------------------------------------
    # POTIONS — raw NLE-canonical names (no "potion of " prefix).
    # All potions: weight=20, nutrition=10, material=GLASS.
    # Covers both raw variants for existing "potion of X" entries AND
    # potions not yet present in objects.py at all.
    # Source: objects.h lines 1125-1178.
    # -----------------------------------------------------------------------

    # POTION("gain ability", "ruby", 1, 0, 40, 300, CLR_RED, ...)
    ObjectEntry(
        name="gain ability", description="ruby",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=300, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_RED, material=Material.GLASS,
    ),
    # POTION("restore ability", "pink", 1, 0, 40, 100, CLR_BRIGHT_MAGENTA, ...)
    ObjectEntry(
        name="restore ability", description="pink",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_BRIGHT_MAGENTA, material=Material.GLASS,
    ),
    # POTION("confusion", "orange", 1, CONFUSION, 40, 100, CLR_ORANGE, ...)
    ObjectEntry(
        name="confusion", description="orange",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_ORANGE, material=Material.GLASS,
    ),
    # POTION("blindness", "yellow", 1, BLINDED, 30, 150, CLR_YELLOW, ...)
    ObjectEntry(
        name="blindness", description="yellow",
        class_=ObjectClass.POTION_CLASS,
        prob=30, weight=20, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_YELLOW, material=Material.GLASS,
    ),
    # POTION("paralysis", "emerald", 1, 0, 40, 300, CLR_BRIGHT_GREEN, ...)
    ObjectEntry(
        name="paralysis", description="emerald",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=300, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_BRIGHT_GREEN, material=Material.GLASS,
    ),
    # POTION("speed", "dark green", 1, FAST, 40, 200, CLR_GREEN, ...)
    ObjectEntry(
        name="speed", description="dark green",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=200, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_GREEN, material=Material.GLASS,
    ),
    # POTION("levitation", "cyan", 1, LEVITATION, 40, 200, CLR_CYAN, ...)
    ObjectEntry(
        name="levitation", description="cyan",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=200, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_CYAN, material=Material.GLASS,
    ),
    # POTION("hallucination", "sky blue", 1, HALLUC, 30, 100, CLR_CYAN, ...)
    ObjectEntry(
        name="hallucination", description="sky blue",
        class_=ObjectClass.POTION_CLASS,
        prob=30, weight=20, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_CYAN, material=Material.GLASS,
    ),
    # POTION("invisibility", "brilliant blue", 1, INVIS, 40, 150, CLR_BRIGHT_BLUE, ...)
    ObjectEntry(
        name="invisibility", description="brilliant blue",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_BRIGHT_BLUE, material=Material.GLASS,
    ),
    # POTION("see invisible", "magenta", 1, SEE_INVIS, 40, 50, CLR_MAGENTA, ...)
    ObjectEntry(
        name="see invisible", description="magenta",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=50, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_MAGENTA, material=Material.GLASS,
    ),
    # POTION("healing", "purple-red", 1, 0, 115, 20, CLR_MAGENTA, ...)
    ObjectEntry(
        name="healing", description="purple-red",
        class_=ObjectClass.POTION_CLASS,
        prob=115, weight=20, cost=20, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_MAGENTA, material=Material.GLASS,
    ),
    # POTION("extra healing", "puce", 1, 0, 45, 100, CLR_RED, ...)
    ObjectEntry(
        name="extra healing", description="puce",
        class_=ObjectClass.POTION_CLASS,
        prob=45, weight=20, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_RED, material=Material.GLASS,
    ),
    # POTION("gain level", "milky", 1, 0, 20, 300, CLR_WHITE, ...)
    ObjectEntry(
        name="gain level", description="milky",
        class_=ObjectClass.POTION_CLASS,
        prob=20, weight=20, cost=300, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_WHITE, material=Material.GLASS,
    ),
    # POTION("enlightenment", "swirly", 1, 0, 20, 200, CLR_BROWN, ...)
    ObjectEntry(
        name="enlightenment", description="swirly",
        class_=ObjectClass.POTION_CLASS,
        prob=20, weight=20, cost=200, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_BROWN, material=Material.GLASS,
    ),
    # POTION("monster detection", "bubbly", 1, 0, 40, 150, CLR_WHITE, ...)
    ObjectEntry(
        name="monster detection", description="bubbly",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_WHITE, material=Material.GLASS,
    ),
    # POTION("object detection", "smoky", 1, 0, 40, 150, CLR_GRAY, ...)
    ObjectEntry(
        name="object detection", description="smoky",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_GRAY, material=Material.GLASS,
    ),
    # POTION("gain energy", "cloudy", 1, 0, 40, 150, CLR_WHITE, ...)
    ObjectEntry(
        name="gain energy", description="cloudy",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_WHITE, material=Material.GLASS,
    ),
    # POTION("sleeping", "effervescent", 1, 0, 40, 100, CLR_GRAY, ...)
    ObjectEntry(
        name="sleeping", description="effervescent",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_GRAY, material=Material.GLASS,
    ),
    # POTION("full healing", "black", 1, 0, 10, 200, CLR_BLACK, ...)
    ObjectEntry(
        name="full healing", description="black",
        class_=ObjectClass.POTION_CLASS,
        prob=10, weight=20, cost=200, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_BLACK, material=Material.GLASS,
    ),
    # POTION("polymorph", "golden", 1, 0, 10, 200, CLR_YELLOW, ...)
    ObjectEntry(
        name="polymorph", description="golden",
        class_=ObjectClass.POTION_CLASS,
        prob=10, weight=20, cost=200, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_YELLOW, material=Material.GLASS,
    ),
    # POTION("booze", "brown", 0, 0, 40, 50, CLR_BROWN, ...)
    ObjectEntry(
        name="booze", description="brown",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=50, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_BROWN, material=Material.GLASS,
    ),
    # POTION("sickness", "fizzy", 0, 0, 40, 50, CLR_CYAN, ...)
    ObjectEntry(
        name="sickness", description="fizzy",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=50, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_CYAN, material=Material.GLASS,
    ),
    # POTION("fruit juice", "dark", 0, 0, 40, 50, CLR_BLACK, ...)
    ObjectEntry(
        name="fruit juice", description="dark",
        class_=ObjectClass.POTION_CLASS,
        prob=40, weight=20, cost=50, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_BLACK, material=Material.GLASS,
    ),
    # POTION("acid", "white", 0, 0, 10, 250, CLR_WHITE, ...)
    ObjectEntry(
        name="acid", description="white",
        class_=ObjectClass.POTION_CLASS,
        prob=10, weight=20, cost=250, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_WHITE, material=Material.GLASS,
    ),
    # POTION("oil", "murky", 0, 0, 30, 250, CLR_BROWN, ...)
    ObjectEntry(
        name="oil", description="murky",
        class_=ObjectClass.POTION_CLASS,
        prob=30, weight=20, cost=250, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_BROWN, material=Material.GLASS,
    ),
    # POTION("water", "clear", 0, 0, 80, 100, CLR_CYAN, ...)
    ObjectEntry(
        name="water", description="clear",
        class_=ObjectClass.POTION_CLASS,
        prob=80, weight=20, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=10, color=Color.CLR_CYAN, material=Material.GLASS,
    ),

    # -----------------------------------------------------------------------
    # SCROLLS — missing entries + raw NLE-canonical names for existing ones.
    # All scrolls: weight=5, nutrition=6, material=PAPER, color=CLR_WHITE(15).
    # Source: objects.h lines 1187-1265.
    # -----------------------------------------------------------------------

    # Raw names for scrolls already in objects.py (with "scroll of " prefix)
    ObjectEntry(
        name="identify", description="KERNOD WEL",
        class_=ObjectClass.SCROLL_CLASS,
        prob=180, weight=5, cost=20, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    ObjectEntry(
        name="light", description="VERR YED HORRE",
        class_=ObjectClass.SCROLL_CLASS,
        prob=90, weight=5, cost=50, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    ObjectEntry(
        name="enchant weapon", description="DAIYEN FOOELS",
        class_=ObjectClass.SCROLL_CLASS,
        prob=80, weight=5, cost=60, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    ObjectEntry(
        name="enchant armor", description="ZELGO MER",
        class_=ObjectClass.SCROLL_CLASS,
        prob=63, weight=5, cost=80, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    ObjectEntry(
        name="remove curse", description="PRATYAVAYAH",
        class_=ObjectClass.SCROLL_CLASS,
        prob=65, weight=5, cost=80, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    ObjectEntry(
        name="scare monster", description="XIXAXA XOXAXA XUXAXA",
        class_=ObjectClass.SCROLL_CLASS,
        prob=35, weight=5, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    ObjectEntry(
        name="teleportation", description="VENZAR BORGAVVE",
        class_=ObjectClass.SCROLL_CLASS,
        prob=55, weight=5, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    ObjectEntry(
        name="magic mapping", description="ELAM EBOW",
        class_=ObjectClass.SCROLL_CLASS,
        prob=45, weight=5, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    ObjectEntry(
        name="gold detection", description="THARR",
        class_=ObjectClass.SCROLL_CLASS,
        prob=33, weight=5, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    ObjectEntry(
        name="food detection", description="YUM YUM",
        class_=ObjectClass.SCROLL_CLASS,
        prob=25, weight=5, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    ObjectEntry(
        name="blank paper", description="unlabeled",
        class_=ObjectClass.SCROLL_CLASS,
        prob=28, weight=5, cost=60, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),

    # Missing scrolls not yet in objects.py at all
    # SCROLL("destroy armor", "JUYED AWK YACC", 1, 45, 100, SCR_DESTROY_ARMOR)
    ObjectEntry(
        name="destroy armor", description="JUYED AWK YACC",
        class_=ObjectClass.SCROLL_CLASS,
        prob=45, weight=5, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    # SCROLL("confuse monster", "NR 9", 1, 53, 100, SCR_CONFUSE_MONSTER)
    ObjectEntry(
        name="confuse monster", description="NR 9",
        class_=ObjectClass.SCROLL_CLASS,
        prob=53, weight=5, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    # SCROLL("create monster", "LEP GEX VEN ZEA", 1, 45, 200, SCR_CREATE_MONSTER)
    ObjectEntry(
        name="create monster", description="LEP GEX VEN ZEA",
        class_=ObjectClass.SCROLL_CLASS,
        prob=45, weight=5, cost=200, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    # SCROLL("taming", "PRIRUTSENIE", 1, 15, 200, SCR_TAMING)
    ObjectEntry(
        name="taming", description="PRIRUTSENIE",
        class_=ObjectClass.SCROLL_CLASS,
        prob=15, weight=5, cost=200, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    # SCROLL("genocide", "ELBIB YLOH", 1, 15, 300, SCR_GENOCIDE)
    ObjectEntry(
        name="genocide", description="ELBIB YLOH",
        class_=ObjectClass.SCROLL_CLASS,
        prob=15, weight=5, cost=300, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    # SCROLL("amnesia", "DUAM XNAHT", 1, 35, 200, SCR_AMNESIA)
    ObjectEntry(
        name="amnesia", description="DUAM XNAHT",
        class_=ObjectClass.SCROLL_CLASS,
        prob=35, weight=5, cost=200, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    # SCROLL("fire", "ANDOVA BEGARIN", 1, 30, 100, SCR_FIRE)
    ObjectEntry(
        name="fire", description="ANDOVA BEGARIN",
        class_=ObjectClass.SCROLL_CLASS,
        prob=30, weight=5, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    # SCROLL("earth", "KIRJE", 1, 18, 200, SCR_EARTH)
    ObjectEntry(
        name="earth", description="KIRJE",
        class_=ObjectClass.SCROLL_CLASS,
        prob=18, weight=5, cost=200, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    # SCROLL("punishment", "VE FORBRYDERNE", 1, 15, 300, SCR_PUNISHMENT)
    ObjectEntry(
        name="punishment", description="VE FORBRYDERNE",
        class_=ObjectClass.SCROLL_CLASS,
        prob=15, weight=5, cost=300, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    # SCROLL("charging", "HACKEM MUCHE", 1, 15, 300, SCR_CHARGING)
    ObjectEntry(
        name="charging", description="HACKEM MUCHE",
        class_=ObjectClass.SCROLL_CLASS,
        prob=15, weight=5, cost=300, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    # SCROLL("stinking cloud", "VELOX NEB", 1, 15, 300, SCR_STINKING_CLOUD)
    ObjectEntry(
        name="stinking cloud", description="VELOX NEB",
        class_=ObjectClass.SCROLL_CLASS,
        prob=15, weight=5, cost=300, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),
    # SCROLL("mail", "stamped", 0, 0, 0, SCR_MAIL) — #ifdef MAIL_STRUCTURES
    ObjectEntry(
        name="mail", description="stamped",
        class_=ObjectClass.SCROLL_CLASS,
        prob=0, weight=5, cost=0, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=6, color=Color.CLR_WHITE, material=Material.PAPER,
    ),

    # -----------------------------------------------------------------------
    # WANDS — missing entries + raw NLE-canonical names for existing ones.
    # All wands: weight=7, nutrition=30.
    # Source: objects.h lines 1449-1504.
    # -----------------------------------------------------------------------

    # Raw names for wands already in objects.py (with "wand of " prefix)
    ObjectEntry(
        name="light", description="glass",
        class_=ObjectClass.WAND_CLASS,
        prob=95, weight=7, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.CLR_BRIGHT_CYAN, material=Material.GLASS,
    ),
    ObjectEntry(
        name="nothing", description="oak",
        class_=ObjectClass.WAND_CLASS,
        prob=25, weight=7, cost=100, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_WOOD, material=Material.WOOD,
    ),
    ObjectEntry(
        name="secret door detection", description="balsa",
        class_=ObjectClass.WAND_CLASS,
        prob=50, weight=7, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_WOOD, material=Material.WOOD,
    ),
    ObjectEntry(
        name="opening", description="zinc",
        class_=ObjectClass.WAND_CLASS,
        prob=30, weight=7, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_METAL, material=Material.METAL,
    ),
    ObjectEntry(
        name="locking", description="aluminum",
        class_=ObjectClass.WAND_CLASS,
        prob=30, weight=7, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_METAL, material=Material.METAL,
    ),
    ObjectEntry(
        name="probing", description="uranium",
        class_=ObjectClass.WAND_CLASS,
        prob=30, weight=7, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_METAL, material=Material.METAL,
    ),
    ObjectEntry(
        name="magic missile", description="steel",
        class_=ObjectClass.WAND_CLASS,
        prob=50, weight=7, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_METAL, material=Material.IRON,
    ),
    ObjectEntry(
        name="striking", description="ebony",
        class_=ObjectClass.WAND_CLASS,
        prob=30, weight=7, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_WOOD, material=Material.WOOD,
    ),
    ObjectEntry(
        name="slow monster", description="tin",
        class_=ObjectClass.WAND_CLASS,
        prob=50, weight=7, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_METAL, material=Material.METAL,
    ),
    ObjectEntry(
        name="speed monster", description="brass",
        class_=ObjectClass.WAND_CLASS,
        prob=50, weight=7, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.CLR_YELLOW, material=Material.COPPER,
    ),
    ObjectEntry(
        name="cancellation", description="platinum",
        class_=ObjectClass.WAND_CLASS,
        prob=45, weight=7, cost=200, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.CLR_WHITE, material=Material.PLATINUM,
    ),
    ObjectEntry(
        name="polymorph", description="silver",
        class_=ObjectClass.WAND_CLASS,
        prob=45, weight=7, cost=200, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_SILVER, material=Material.SILVER,
    ),
    ObjectEntry(
        name="teleportation", description="iridium",
        class_=ObjectClass.WAND_CLASS,
        prob=45, weight=7, cost=200, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.CLR_BRIGHT_CYAN, material=Material.METAL,
    ),
    ObjectEntry(
        name="death", description="long",
        class_=ObjectClass.WAND_CLASS,
        prob=5, weight=7, cost=500, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_METAL, material=Material.IRON,
    ),
    ObjectEntry(
        name="sleep", description="runed",
        class_=ObjectClass.WAND_CLASS,
        prob=50, weight=7, cost=175, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_METAL, material=Material.IRON,
    ),
    ObjectEntry(
        name="cold", description="short",
        class_=ObjectClass.WAND_CLASS,
        prob=40, weight=7, cost=175, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_METAL, material=Material.IRON,
    ),
    ObjectEntry(
        name="fire", description="hexagonal",
        class_=ObjectClass.WAND_CLASS,
        prob=40, weight=7, cost=175, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_METAL, material=Material.IRON,
    ),
    ObjectEntry(
        name="lightning", description="curved",
        class_=ObjectClass.WAND_CLASS,
        prob=40, weight=7, cost=175, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_METAL, material=Material.IRON,
    ),
    ObjectEntry(
        name="digging", description="iron",
        class_=ObjectClass.WAND_CLASS,
        prob=40, weight=7, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_METAL, material=Material.IRON,
    ),

    # Missing wands — not yet in objects.py
    # WAND("enlightenment", "crystal", 15, 150, 1, NODIR, GLASS, HI_GLASS, WAN_ENLIGHTENMENT)
    ObjectEntry(
        name="enlightenment", description="crystal",
        class_=ObjectClass.WAND_CLASS,
        prob=15, weight=7, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.CLR_BRIGHT_CYAN, material=Material.GLASS,
    ),
    # WAND("create monster", "maple", 50, 200, 1, NODIR, WOOD, HI_WOOD, WAN_CREATE_MONSTER)
    ObjectEntry(
        name="create monster", description="maple",
        class_=ObjectClass.WAND_CLASS,
        prob=50, weight=7, cost=200, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_WOOD, material=Material.WOOD,
    ),
    # WAND("wishing", "pine", 5, 500, 1, NODIR, WOOD, HI_WOOD, WAN_WISHING)
    ObjectEntry(
        name="wishing", description="pine",
        class_=ObjectClass.WAND_CLASS,
        prob=5, weight=7, cost=500, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_WOOD, material=Material.WOOD,
    ),
    # WAND("stasis", "redwood", 45, 150, 1, NODIR, WOOD, CLR_RED, WAN_STASIS)
    ObjectEntry(
        name="stasis", description="redwood",
        class_=ObjectClass.WAND_CLASS,
        prob=45, weight=7, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.CLR_RED, material=Material.WOOD,
    ),
    # WAND("make invisible", "marble", 45, 150, 1, IMMEDIATE, MINERAL, HI_MINERAL, WAN_MAKE_INVISIBLE)
    ObjectEntry(
        name="make invisible", description="marble",
        class_=ObjectClass.WAND_CLASS,
        prob=45, weight=7, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.HI_MINERAL, material=Material.MINERAL,
    ),
    # WAND("undead turning", "copper", 50, 150, 1, IMMEDIATE, COPPER, HI_COPPER, WAN_UNDEAD_TURNING)
    ObjectEntry(
        name="undead turning", description="copper",
        class_=ObjectClass.WAND_CLASS,
        prob=50, weight=7, cost=150, sdam=(0, 0), ldam=(0, 0),
        oc1=0, oc2=0, nutrition=30, color=Color.CLR_YELLOW, material=Material.COPPER,
    ),
)
