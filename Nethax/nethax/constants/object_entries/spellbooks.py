"""Spellbooks — vendor/nethack/include/objects.h SPELL() entries.

SPELL macro signature:
    SPELL(name, desc, sub, prob, delay, level, mgc, dir, color, sn)
    expands to:
        OBJECT(OBJ(name, desc),
               BITS(0,0,0,0,mgc,0,0,0,0,0,dir,sub,PAPER),
               0, SPBOOK_CLASS, prob, delay, 50, level*100,
               0, 0, 0, level, 20, color, sn)
    So: weight=50, cost=level*100, sdam=(0,0), ldam=(0,0),
        oc1=sub (spell school), oc2=level, nutrition=20,
        material=LEATHER for parchment/vellum, PAPER for all others.

Spell school values (from vendor/nethack/include/skills.h):
    P_NONE              =  0
    P_ATTACK_SPELL      = 28
    P_HEALING_SPELL     = 29
    P_DIVINATION_SPELL  = 30
    P_ENCHANTMENT_SPELL = 31
    P_CLERIC_SPELL      = 32
    P_ESCAPE_SPELL      = 33
    P_MATTER_SPELL      = 34

Color constants (from vendor/nethack/include/color.h via objects.py):
    HI_PAPER   = CLR_WHITE          = 15
    HI_LEATHER = CLR_BROWN          =  3
    HI_CLOTH   = CLR_BROWN          =  3
    HI_COPPER  = CLR_YELLOW         = 11
    HI_SILVER  = CLR_GRAY           =  7
    HI_GOLD    = CLR_YELLOW         = 11
"""

from Nethax.nethax.constants.objects import (
    Color,
    Material,
    ObjectClass,
    ObjectEntry,
)

# Spell school constants (vendor/nethack/include/skills.h)
P_NONE             =  0
P_ATTACK_SPELL     = 28
P_HEALING_SPELL    = 29
P_DIVINATION_SPELL = 30
P_ENCHANTMENT_SPELL = 31
P_CLERIC_SPELL     = 32
P_ESCAPE_SPELL     = 33
P_MATTER_SPELL     = 34

ENTRIES = (
    # SPELL("dig", "parchment", P_MATTER_SPELL, 20, 6, 5, 1, RAY, HI_LEATHER, SPE_DIG)
    # Note: parchment/vellum use LEATHER material (#define PAPER LEATHER)
    ObjectEntry(
        name="dig",
        description="parchment",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=20,
        weight=50,
        cost=500,           # level 5 * 100
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_MATTER_SPELL,
        oc2=5,              # spell level
        nutrition=20,
        color=Color.CLR_BROWN,   # HI_LEATHER
        material=Material.LEATHER,
    ),

    # SPELL("magic missile", "vellum", P_ATTACK_SPELL, 45, 2, 2, 1, RAY, HI_LEATHER, SPE_MAGIC_MISSILE)
    ObjectEntry(
        name="magic missile",
        description="vellum",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=45,
        weight=50,
        cost=200,           # level 2 * 100
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ATTACK_SPELL,
        oc2=2,
        nutrition=20,
        color=Color.CLR_BROWN,   # HI_LEATHER
        material=Material.LEATHER,
    ),

    # SPELL("fireball", "ragged", P_ATTACK_SPELL, 20, 4, 4, 1, RAY, HI_PAPER, SPE_FIREBALL)
    ObjectEntry(
        name="fireball",
        description="ragged",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=20,
        weight=50,
        cost=400,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ATTACK_SPELL,
        oc2=4,
        nutrition=20,
        color=Color.CLR_WHITE,   # HI_PAPER
        material=Material.PAPER,
    ),

    # SPELL("cone of cold", "dog eared", P_ATTACK_SPELL, 10, 7, 4, 1, RAY, HI_PAPER, SPE_CONE_OF_COLD)
    ObjectEntry(
        name="cone of cold",
        description="dog eared",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=10,
        weight=50,
        cost=400,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ATTACK_SPELL,
        oc2=4,
        nutrition=20,
        color=Color.CLR_WHITE,   # HI_PAPER
        material=Material.PAPER,
    ),

    # SPELL("sleep", "mottled", P_ENCHANTMENT_SPELL, 30, 1, 3, 1, RAY, HI_PAPER, SPE_SLEEP)
    ObjectEntry(
        name="sleep",
        description="mottled",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=30,
        weight=50,
        cost=300,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ENCHANTMENT_SPELL,
        oc2=3,
        nutrition=20,
        color=Color.CLR_WHITE,   # HI_PAPER
        material=Material.PAPER,
    ),

    # SPELL("finger of death", "stained", P_ATTACK_SPELL, 5, 10, 7, 1, RAY, HI_PAPER, SPE_FINGER_OF_DEATH)
    ObjectEntry(
        name="finger of death",
        description="stained",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=5,
        weight=50,
        cost=700,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ATTACK_SPELL,
        oc2=7,
        nutrition=20,
        color=Color.CLR_WHITE,   # HI_PAPER
        material=Material.PAPER,
    ),

    # SPELL("light", "cloth", P_DIVINATION_SPELL, 45, 1, 1, 1, NODIR, HI_CLOTH, SPE_LIGHT)
    ObjectEntry(
        name="light",
        description="cloth",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=45,
        weight=50,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_DIVINATION_SPELL,
        oc2=1,
        nutrition=20,
        color=Color.CLR_BROWN,   # HI_CLOTH
        material=Material.PAPER,
    ),

    # SPELL("detect monsters", "leathery", P_DIVINATION_SPELL, 43, 1, 1, 1, NODIR, HI_LEATHER, SPE_DETECT_MONSTERS)
    ObjectEntry(
        name="detect monsters",
        description="leathery",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=43,
        weight=50,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_DIVINATION_SPELL,
        oc2=1,
        nutrition=20,
        color=Color.CLR_BROWN,   # HI_LEATHER
        material=Material.PAPER,
    ),

    # SPELL("healing", "white", P_HEALING_SPELL, 40, 2, 1, 1, IMMEDIATE, CLR_WHITE, SPE_HEALING)
    ObjectEntry(
        name="healing",
        description="white",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=40,
        weight=50,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_HEALING_SPELL,
        oc2=1,
        nutrition=20,
        color=Color.CLR_WHITE,
        material=Material.PAPER,
    ),

    # SPELL("knock", "pink", P_MATTER_SPELL, 25, 1, 1, 1, IMMEDIATE, CLR_BRIGHT_MAGENTA, SPE_KNOCK)
    ObjectEntry(
        name="knock",
        description="pink",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=25,
        weight=50,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_MATTER_SPELL,
        oc2=1,
        nutrition=20,
        color=Color.CLR_BRIGHT_MAGENTA,
        material=Material.PAPER,
    ),

    # SPELL("force bolt", "red", P_ATTACK_SPELL, 30, 2, 1, 1, IMMEDIATE, CLR_RED, SPE_FORCE_BOLT)
    ObjectEntry(
        name="force bolt",
        description="red",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=30,
        weight=50,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ATTACK_SPELL,
        oc2=1,
        nutrition=20,
        color=Color.CLR_RED,
        material=Material.PAPER,
    ),

    # SPELL("confuse monster", "orange", P_ENCHANTMENT_SPELL, 49, 2, 1, 1, IMMEDIATE, CLR_ORANGE, SPE_CONFUSE_MONSTER)
    ObjectEntry(
        name="confuse monster",
        description="orange",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=49,
        weight=50,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ENCHANTMENT_SPELL,
        oc2=1,
        nutrition=20,
        color=Color.CLR_ORANGE,
        material=Material.PAPER,
    ),

    # SPELL("cure blindness", "yellow", P_HEALING_SPELL, 25, 2, 2, 1, IMMEDIATE, CLR_YELLOW, SPE_CURE_BLINDNESS)
    ObjectEntry(
        name="cure blindness",
        description="yellow",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=25,
        weight=50,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_HEALING_SPELL,
        oc2=2,
        nutrition=20,
        color=Color.CLR_YELLOW,
        material=Material.PAPER,
    ),

    # SPELL("drain life", "velvet", P_ATTACK_SPELL, 10, 2, 2, 1, IMMEDIATE, CLR_MAGENTA, SPE_DRAIN_LIFE)
    ObjectEntry(
        name="drain life",
        description="velvet",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=10,
        weight=50,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ATTACK_SPELL,
        oc2=2,
        nutrition=20,
        color=Color.CLR_MAGENTA,
        material=Material.PAPER,
    ),

    # SPELL("slow monster", "light green", P_ENCHANTMENT_SPELL, 30, 2, 2, 1, IMMEDIATE, CLR_BRIGHT_GREEN, SPE_SLOW_MONSTER)
    ObjectEntry(
        name="slow monster",
        description="light green",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=30,
        weight=50,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ENCHANTMENT_SPELL,
        oc2=2,
        nutrition=20,
        color=Color.CLR_BRIGHT_GREEN,
        material=Material.PAPER,
    ),

    # SPELL("wizard lock", "dark green", P_MATTER_SPELL, 25, 3, 2, 1, IMMEDIATE, CLR_GREEN, SPE_WIZARD_LOCK)
    ObjectEntry(
        name="wizard lock",
        description="dark green",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=25,
        weight=50,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_MATTER_SPELL,
        oc2=2,
        nutrition=20,
        color=Color.CLR_GREEN,
        material=Material.PAPER,
    ),

    # SPELL("create monster", "turquoise", P_CLERIC_SPELL, 35, 3, 2, 1, NODIR, CLR_BRIGHT_CYAN, SPE_CREATE_MONSTER)
    ObjectEntry(
        name="create monster",
        description="turquoise",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=35,
        weight=50,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_CLERIC_SPELL,
        oc2=2,
        nutrition=20,
        color=Color.CLR_BRIGHT_CYAN,
        material=Material.PAPER,
    ),

    # SPELL("detect food", "cyan", P_DIVINATION_SPELL, 30, 3, 2, 1, NODIR, CLR_CYAN, SPE_DETECT_FOOD)
    ObjectEntry(
        name="detect food",
        description="cyan",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=30,
        weight=50,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_DIVINATION_SPELL,
        oc2=2,
        nutrition=20,
        color=Color.CLR_CYAN,
        material=Material.PAPER,
    ),

    # SPELL("cause fear", "light blue", P_ENCHANTMENT_SPELL, 25, 3, 3, 1, NODIR, CLR_BRIGHT_BLUE, SPE_CAUSE_FEAR)
    ObjectEntry(
        name="cause fear",
        description="light blue",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=25,
        weight=50,
        cost=300,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ENCHANTMENT_SPELL,
        oc2=3,
        nutrition=20,
        color=Color.CLR_BRIGHT_BLUE,
        material=Material.PAPER,
    ),

    # SPELL("clairvoyance", "dark blue", P_DIVINATION_SPELL, 15, 3, 3, 1, NODIR, CLR_BLUE, SPE_CLAIRVOYANCE)
    ObjectEntry(
        name="clairvoyance",
        description="dark blue",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=15,
        weight=50,
        cost=300,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_DIVINATION_SPELL,
        oc2=3,
        nutrition=20,
        color=Color.CLR_BLUE,
        material=Material.PAPER,
    ),

    # SPELL("cure sickness", "indigo", P_HEALING_SPELL, 32, 3, 3, 1, NODIR, CLR_BLUE, SPE_CURE_SICKNESS)
    ObjectEntry(
        name="cure sickness",
        description="indigo",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=32,
        weight=50,
        cost=300,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_HEALING_SPELL,
        oc2=3,
        nutrition=20,
        color=Color.CLR_BLUE,
        material=Material.PAPER,
    ),

    # SPELL("charm monster", "magenta", P_ENCHANTMENT_SPELL, 20, 3, 5, 1, IMMEDIATE, CLR_MAGENTA, SPE_CHARM_MONSTER)
    ObjectEntry(
        name="charm monster",
        description="magenta",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=20,
        weight=50,
        cost=500,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ENCHANTMENT_SPELL,
        oc2=5,
        nutrition=20,
        color=Color.CLR_MAGENTA,
        material=Material.PAPER,
    ),

    # SPELL("haste self", "purple", P_ESCAPE_SPELL, 33, 4, 3, 1, NODIR, CLR_MAGENTA, SPE_HASTE_SELF)
    ObjectEntry(
        name="haste self",
        description="purple",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=33,
        weight=50,
        cost=300,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ESCAPE_SPELL,
        oc2=3,
        nutrition=20,
        color=Color.CLR_MAGENTA,
        material=Material.PAPER,
    ),

    # SPELL("detect unseen", "violet", P_DIVINATION_SPELL, 20, 4, 3, 1, NODIR, CLR_MAGENTA, SPE_DETECT_UNSEEN)
    ObjectEntry(
        name="detect unseen",
        description="violet",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=20,
        weight=50,
        cost=300,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_DIVINATION_SPELL,
        oc2=3,
        nutrition=20,
        color=Color.CLR_MAGENTA,
        material=Material.PAPER,
    ),

    # SPELL("levitation", "tan", P_ESCAPE_SPELL, 20, 4, 4, 1, NODIR, CLR_BROWN, SPE_LEVITATION)
    ObjectEntry(
        name="levitation",
        description="tan",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=20,
        weight=50,
        cost=400,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ESCAPE_SPELL,
        oc2=4,
        nutrition=20,
        color=Color.CLR_BROWN,
        material=Material.PAPER,
    ),

    # SPELL("extra healing", "plaid", P_HEALING_SPELL, 27, 5, 3, 1, IMMEDIATE, CLR_GREEN, SPE_EXTRA_HEALING)
    ObjectEntry(
        name="extra healing",
        description="plaid",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=27,
        weight=50,
        cost=300,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_HEALING_SPELL,
        oc2=3,
        nutrition=20,
        color=Color.CLR_GREEN,
        material=Material.PAPER,
    ),

    # SPELL("restore ability", "light brown", P_HEALING_SPELL, 25, 5, 4, 1, NODIR, CLR_BROWN, SPE_RESTORE_ABILITY)
    ObjectEntry(
        name="restore ability",
        description="light brown",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=25,
        weight=50,
        cost=400,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_HEALING_SPELL,
        oc2=4,
        nutrition=20,
        color=Color.CLR_BROWN,
        material=Material.PAPER,
    ),

    # SPELL("invisibility", "dark brown", P_ESCAPE_SPELL, 20, 5, 4, 1, NODIR, CLR_BROWN, SPE_INVISIBILITY)
    ObjectEntry(
        name="invisibility",
        description="dark brown",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=20,
        weight=50,
        cost=400,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ESCAPE_SPELL,
        oc2=4,
        nutrition=20,
        color=Color.CLR_BROWN,
        material=Material.PAPER,
    ),

    # SPELL("detect treasure", "gray", P_DIVINATION_SPELL, 20, 5, 4, 1, NODIR, CLR_GRAY, SPE_DETECT_TREASURE)
    ObjectEntry(
        name="detect treasure",
        description="gray",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=20,
        weight=50,
        cost=400,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_DIVINATION_SPELL,
        oc2=4,
        nutrition=20,
        color=Color.CLR_GRAY,
        material=Material.PAPER,
    ),

    # SPELL("remove curse", "wrinkled", P_CLERIC_SPELL, 25, 5, 3, 1, NODIR, HI_PAPER, SPE_REMOVE_CURSE)
    ObjectEntry(
        name="remove curse",
        description="wrinkled",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=25,
        weight=50,
        cost=300,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_CLERIC_SPELL,
        oc2=3,
        nutrition=20,
        color=Color.CLR_WHITE,   # HI_PAPER
        material=Material.PAPER,
    ),

    # SPELL("magic mapping", "dusty", P_DIVINATION_SPELL, 18, 7, 5, 1, NODIR, HI_PAPER, SPE_MAGIC_MAPPING)
    ObjectEntry(
        name="magic mapping",
        description="dusty",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=18,
        weight=50,
        cost=500,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_DIVINATION_SPELL,
        oc2=5,
        nutrition=20,
        color=Color.CLR_WHITE,   # HI_PAPER
        material=Material.PAPER,
    ),

    # SPELL("identify", "bronze", P_DIVINATION_SPELL, 20, 6, 3, 1, NODIR, HI_COPPER, SPE_IDENTIFY)
    ObjectEntry(
        name="identify",
        description="bronze",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=20,
        weight=50,
        cost=300,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_DIVINATION_SPELL,
        oc2=3,
        nutrition=20,
        color=Color.CLR_YELLOW,  # HI_COPPER
        material=Material.PAPER,
    ),

    # SPELL("turn undead", "copper", P_CLERIC_SPELL, 16, 8, 6, 1, IMMEDIATE, HI_COPPER, SPE_TURN_UNDEAD)
    ObjectEntry(
        name="turn undead",
        description="copper",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=16,
        weight=50,
        cost=600,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_CLERIC_SPELL,
        oc2=6,
        nutrition=20,
        color=Color.CLR_YELLOW,  # HI_COPPER
        material=Material.PAPER,
    ),

    # SPELL("polymorph", "silver", P_MATTER_SPELL, 10, 8, 6, 1, IMMEDIATE, HI_SILVER, SPE_POLYMORPH)
    ObjectEntry(
        name="polymorph",
        description="silver",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=10,
        weight=50,
        cost=600,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_MATTER_SPELL,
        oc2=6,
        nutrition=20,
        color=Color.CLR_GRAY,    # HI_SILVER
        material=Material.PAPER,
    ),

    # SPELL("teleport away", "gold", P_ESCAPE_SPELL, 15, 6, 6, 1, IMMEDIATE, HI_GOLD, SPE_TELEPORT_AWAY)
    ObjectEntry(
        name="teleport away",
        description="gold",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=15,
        weight=50,
        cost=600,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ESCAPE_SPELL,
        oc2=6,
        nutrition=20,
        color=Color.CLR_YELLOW,  # HI_GOLD
        material=Material.PAPER,
    ),

    # SPELL("create familiar", "glittering", P_CLERIC_SPELL, 10, 7, 6, 1, NODIR, CLR_WHITE, SPE_CREATE_FAMILIAR)
    ObjectEntry(
        name="create familiar",
        description="glittering",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=10,
        weight=50,
        cost=600,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_CLERIC_SPELL,
        oc2=6,
        nutrition=20,
        color=Color.CLR_WHITE,
        material=Material.PAPER,
    ),

    # SPELL("cancellation", "shining", P_MATTER_SPELL, 15, 8, 7, 1, IMMEDIATE, CLR_WHITE, SPE_CANCELLATION)
    ObjectEntry(
        name="cancellation",
        description="shining",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=15,
        weight=50,
        cost=700,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_MATTER_SPELL,
        oc2=7,
        nutrition=20,
        color=Color.CLR_WHITE,
        material=Material.PAPER,
    ),

    # SPELL("protection", "dull", P_CLERIC_SPELL, 18, 3, 1, 1, NODIR, HI_PAPER, SPE_PROTECTION)
    ObjectEntry(
        name="protection",
        description="dull",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=18,
        weight=50,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_CLERIC_SPELL,
        oc2=1,
        nutrition=20,
        color=Color.CLR_WHITE,   # HI_PAPER
        material=Material.PAPER,
    ),

    # SPELL("jumping", "thin", P_ESCAPE_SPELL, 20, 3, 1, 1, IMMEDIATE, HI_PAPER, SPE_JUMPING)
    ObjectEntry(
        name="jumping",
        description="thin",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=20,
        weight=50,
        cost=100,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ESCAPE_SPELL,
        oc2=1,
        nutrition=20,
        color=Color.CLR_WHITE,   # HI_PAPER
        material=Material.PAPER,
    ),

    # SPELL("stone to flesh", "thick", P_HEALING_SPELL, 15, 1, 3, 1, IMMEDIATE, HI_PAPER, SPE_STONE_TO_FLESH)
    ObjectEntry(
        name="stone to flesh",
        description="thick",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=15,
        weight=50,
        cost=300,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_HEALING_SPELL,
        oc2=3,
        nutrition=20,
        color=Color.CLR_WHITE,   # HI_PAPER
        material=Material.PAPER,
    ),

    # SPELL("chain lightning", "checkered", P_ATTACK_SPELL, 25, 4, 2, 1, NODIR, CLR_GRAY, SPE_CHAIN_LIGHTNING)
    ObjectEntry(
        name="chain lightning",
        description="checkered",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=25,
        weight=50,
        cost=200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_ATTACK_SPELL,
        oc2=2,
        nutrition=20,
        color=Color.CLR_GRAY,
        material=Material.PAPER,
    ),

    # SPELL("blank paper", "plain", P_NONE, 18, 0, 0, 0, 0, HI_PAPER, SPE_BLANK_PAPER)
    ObjectEntry(
        name="blank paper",
        description="plain",
        class_=ObjectClass.SPBOOK_CLASS,
        prob=18,
        weight=50,
        cost=0,             # level 0 * 100
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=P_NONE,
        oc2=0,
        nutrition=20,
        color=Color.CLR_WHITE,   # HI_PAPER
        material=Material.PAPER,
    ),
)
