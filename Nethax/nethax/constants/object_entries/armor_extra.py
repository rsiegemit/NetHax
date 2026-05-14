"""Missing armor — vendor/nethack/include/objects.h ARMOR/HELM/CLOAK/SHIELD/GLOVES/BOOTS/DRGN_ARMR entries."""
from Nethax.nethax.constants.objects import ObjectEntry, ObjectClass, Color, Material

# Color aliases (vendor/nethack/include/color.h)
# HI_METAL      = CLR_CYAN          = 6
# HI_SILVER     = CLR_GRAY          = 7
# HI_GOLD       = CLR_YELLOW        = 11
# HI_LEATHER    = CLR_BROWN         = 3
# HI_CLOTH      = CLR_BROWN         = 3
# HI_WOOD       = CLR_BROWN         = 3
# DRAGON_SILVER = CLR_BRIGHT_CYAN   = 14

# ARMOR macro: oc1 = 10 - ac,  oc2 = can,  nutrition = weight
# DRGN_ARMR:   weight=40, prob=0, delay=5

ENTRIES: tuple[ObjectEntry, ...] = (

    # -----------------------------------------------------------------------
    # HELMS
    # HELM(name, desc, kn, mgc, power, prob, delay, wt, cost, ac, can, metal, c, sn)
    # oc1 = 10 - ac, oc2 = can, nutrition = wt
    # -----------------------------------------------------------------------

    # HELM("elven leather helm", "leather hat", 0, 0, 0, 6, 1, 3, 8, 9, 0, LEATHER, HI_LEATHER, ...)
    ObjectEntry(
        name="elven leather helm",
        description="leather hat",
        class_=ObjectClass.ARMOR_CLASS,
        prob=6,
        weight=3,
        cost=8,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=3,
        color=Color.HI_LEATHER,
        material=Material.LEATHER,
    ),

    # HELM("fedora", NoDes, 1, 0, 0, 0, 0, 3, 1, 10, 0, CLOTH, CLR_BROWN, ...)
    ObjectEntry(
        name="fedora",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=3,
        cost=1,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,   # 10 - 10
        oc2=0,
        nutrition=3,
        color=Color.CLR_BROWN,
        material=Material.CLOTH,
    ),

    # HELM("cornuthaum", "conical hat", 0, 1, CLAIRVOYANT, 5, 1, 4, 80, 10, 1, CLOTH, CLR_BLUE, ...)
    ObjectEntry(
        name="cornuthaum",
        description="conical hat",
        class_=ObjectClass.ARMOR_CLASS,
        prob=5,
        weight=4,
        cost=80,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,   # 10 - 10
        oc2=1,
        nutrition=4,
        color=Color.CLR_BLUE,
        material=Material.CLOTH,
    ),

    # HELM("dunce cap", "conical hat", 0, 1, 0, 5, 1, 4, 1, 10, 0, CLOTH, CLR_BLUE, ...)
    ObjectEntry(
        name="dunce cap",
        description="conical hat",
        class_=ObjectClass.ARMOR_CLASS,
        prob=5,
        weight=4,
        cost=1,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,   # 10 - 10
        oc2=0,
        nutrition=4,
        color=Color.CLR_BLUE,
        material=Material.CLOTH,
    ),

    # HELM("dented pot", NoDes, 1, 0, 0, 2, 0, 10, 8, 9, 0, IRON, CLR_BLACK, ...)
    ObjectEntry(
        name="dented pot",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=2,
        weight=10,
        cost=8,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=10,
        color=Color.CLR_BLACK,
        material=Material.IRON,
    ),

    # HELM("helm of brilliance", "crystal helmet", 0, 1, 0, 6, 1, 40, 50, 9, 0, GLASS, CLR_WHITE, ...)
    ObjectEntry(
        name="helm of brilliance",
        description="crystal helmet",
        class_=ObjectClass.ARMOR_CLASS,
        prob=6,
        weight=40,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=40,
        color=Color.CLR_WHITE,
        material=Material.GLASS,
    ),

    # HELM("helm of caution", "etched helmet", 0, 1, WARNING, 6, 1, 50, 50, 9, 0, IRON, CLR_GREEN, ...)
    ObjectEntry(
        name="helm of caution",
        description="etched helmet",
        class_=ObjectClass.ARMOR_CLASS,
        prob=6,
        weight=50,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=50,
        color=Color.CLR_GREEN,
        material=Material.IRON,
    ),

    # HELM("helm of opposite alignment", "crested helmet", 0, 1, 0, 10, 1, 50, 50, 9, 0, IRON, HI_METAL, ...)
    ObjectEntry(
        name="helm of opposite alignment",
        description="crested helmet",
        class_=ObjectClass.ARMOR_CLASS,
        prob=10,
        weight=50,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=50,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # HELM("helm of telepathy", "visored helmet", 0, 1, TELEPAT, 4, 1, 50, 50, 9, 0, IRON, HI_METAL, ...)
    ObjectEntry(
        name="helm of telepathy",
        description="visored helmet",
        class_=ObjectClass.ARMOR_CLASS,
        prob=4,
        weight=50,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=50,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # -----------------------------------------------------------------------
    # DRAGON SCALE MAIL  (11 entries)
    # DRGN_ARMR: weight=40, prob=0, delay=5, can=0, oc1=10-ac, sub=ARM_SUIT, material=DRAGON_HIDE
    # -----------------------------------------------------------------------

    # DRGN_ARMR("gray dragon scale mail", 1, ANTIMAGIC, 1200, 1, CLR_GRAY, ...)
    ObjectEntry(
        name="gray dragon scale mail",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=1200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=9,   # 10 - 1
        oc2=0,
        nutrition=40,
        color=Color.CLR_GRAY,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("gold dragon scale mail", 1, 0, 900, 1, HI_GOLD, ...)
    ObjectEntry(
        name="gold dragon scale mail",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=900,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=9,   # 10 - 1
        oc2=0,
        nutrition=40,
        color=Color.HI_GOLD,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("silver dragon scale mail", 1, REFLECTING, 1200, 1, DRAGON_SILVER, ...)
    ObjectEntry(
        name="silver dragon scale mail",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=1200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=9,   # 10 - 1
        oc2=0,
        nutrition=40,
        color=Color.CLR_BRIGHT_CYAN,   # DRAGON_SILVER
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("shimmering dragon scale mail", 1, DISPLACED, 1200, 1, CLR_CYAN, ...)
    ObjectEntry(
        name="shimmering dragon scale mail",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=1200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=9,   # 10 - 1
        oc2=0,
        nutrition=40,
        color=Color.CLR_CYAN,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("red dragon scale mail", 1, FIRE_RES, 900, 1, CLR_RED, ...)
    ObjectEntry(
        name="red dragon scale mail",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=900,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=9,   # 10 - 1
        oc2=0,
        nutrition=40,
        color=Color.CLR_RED,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("white dragon scale mail", 1, COLD_RES, 900, 1, CLR_WHITE, ...)
    ObjectEntry(
        name="white dragon scale mail",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=900,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=9,   # 10 - 1
        oc2=0,
        nutrition=40,
        color=Color.CLR_WHITE,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("orange dragon scale mail", 1, SLEEP_RES, 900, 1, CLR_ORANGE, ...)
    ObjectEntry(
        name="orange dragon scale mail",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=900,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=9,   # 10 - 1
        oc2=0,
        nutrition=40,
        color=Color.CLR_ORANGE,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("black dragon scale mail", 1, DISINT_RES, 1200, 1, CLR_BLACK, ...)
    ObjectEntry(
        name="black dragon scale mail",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=1200,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=9,   # 10 - 1
        oc2=0,
        nutrition=40,
        color=Color.CLR_BLACK,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("blue dragon scale mail", 1, SHOCK_RES, 900, 1, CLR_BLUE, ...)
    ObjectEntry(
        name="blue dragon scale mail",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=900,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=9,   # 10 - 1
        oc2=0,
        nutrition=40,
        color=Color.CLR_BLUE,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("green dragon scale mail", 1, POISON_RES, 900, 1, CLR_GREEN, ...)
    ObjectEntry(
        name="green dragon scale mail",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=900,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=9,   # 10 - 1
        oc2=0,
        nutrition=40,
        color=Color.CLR_GREEN,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("yellow dragon scale mail", 1, ACID_RES, 900, 1, CLR_YELLOW, ...)
    ObjectEntry(
        name="yellow dragon scale mail",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=900,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=9,   # 10 - 1
        oc2=0,
        nutrition=40,
        color=Color.CLR_YELLOW,
        material=Material.DRAGON_HIDE,
    ),

    # -----------------------------------------------------------------------
    # DRAGON SCALES  (11 entries)
    # DRGN_ARMR: weight=40, prob=0, delay=5, can=0, oc1=10-7=3, sub=ARM_SUIT
    # -----------------------------------------------------------------------

    # DRGN_ARMR("gray dragon scales", 0, ANTIMAGIC, 700, 7, CLR_GRAY, ...)
    ObjectEntry(
        name="gray dragon scales",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=700,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=3,   # 10 - 7
        oc2=0,
        nutrition=40,
        color=Color.CLR_GRAY,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("gold dragon scales", 0, 0, 500, 7, HI_GOLD, ...)
    ObjectEntry(
        name="gold dragon scales",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=500,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=3,   # 10 - 7
        oc2=0,
        nutrition=40,
        color=Color.HI_GOLD,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("silver dragon scales", 0, REFLECTING, 700, 7, DRAGON_SILVER, ...)
    ObjectEntry(
        name="silver dragon scales",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=700,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=3,   # 10 - 7
        oc2=0,
        nutrition=40,
        color=Color.CLR_BRIGHT_CYAN,   # DRAGON_SILVER
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("shimmering dragon scales", 0, DISPLACED, 700, 7, CLR_CYAN, ...)
    ObjectEntry(
        name="shimmering dragon scales",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=700,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=3,   # 10 - 7
        oc2=0,
        nutrition=40,
        color=Color.CLR_CYAN,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("red dragon scales", 0, FIRE_RES, 500, 7, CLR_RED, ...)
    ObjectEntry(
        name="red dragon scales",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=500,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=3,   # 10 - 7
        oc2=0,
        nutrition=40,
        color=Color.CLR_RED,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("white dragon scales", 0, COLD_RES, 500, 7, CLR_WHITE, ...)
    ObjectEntry(
        name="white dragon scales",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=500,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=3,   # 10 - 7
        oc2=0,
        nutrition=40,
        color=Color.CLR_WHITE,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("orange dragon scales", 0, SLEEP_RES, 500, 7, CLR_ORANGE, ...)
    ObjectEntry(
        name="orange dragon scales",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=500,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=3,   # 10 - 7
        oc2=0,
        nutrition=40,
        color=Color.CLR_ORANGE,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("black dragon scales", 0, DISINT_RES, 700, 7, CLR_BLACK, ...)
    ObjectEntry(
        name="black dragon scales",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=700,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=3,   # 10 - 7
        oc2=0,
        nutrition=40,
        color=Color.CLR_BLACK,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("blue dragon scales", 0, SHOCK_RES, 500, 7, CLR_BLUE, ...)
    ObjectEntry(
        name="blue dragon scales",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=500,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=3,   # 10 - 7
        oc2=0,
        nutrition=40,
        color=Color.CLR_BLUE,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("green dragon scales", 0, POISON_RES, 500, 7, CLR_GREEN, ...)
    ObjectEntry(
        name="green dragon scales",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=500,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=3,   # 10 - 7
        oc2=0,
        nutrition=40,
        color=Color.CLR_GREEN,
        material=Material.DRAGON_HIDE,
    ),

    # DRGN_ARMR("yellow dragon scales", 0, ACID_RES, 500, 7, CLR_YELLOW, ...)
    ObjectEntry(
        name="yellow dragon scales",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=40,
        cost=500,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=3,   # 10 - 7
        oc2=0,
        nutrition=40,
        color=Color.CLR_YELLOW,
        material=Material.DRAGON_HIDE,
    ),

    # -----------------------------------------------------------------------
    # ARMOR suits (ARM_SUIT)
    # ARMOR(name, desc, kn, mgc, blk, power, prob, delay, wt, cost, ac, can, sub, metal, c, sn)
    # -----------------------------------------------------------------------

    # ARMOR("orcish chain mail", "crude chain mail", 0, 0, 0, 0, 19, 5, 300, 75, 6, 1, ARM_SUIT, IRON, CLR_BLACK, ...)
    ObjectEntry(
        name="orcish chain mail",
        description="crude chain mail",
        class_=ObjectClass.ARMOR_CLASS,
        prob=19,
        weight=300,
        cost=75,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=4,   # 10 - 6
        oc2=1,
        nutrition=300,
        color=Color.CLR_BLACK,
        material=Material.IRON,
    ),

    # ARMOR("orcish ring mail", "crude ring mail", 0, 0, 0, 0, 19, 5, 250, 80, 8, 1, ARM_SUIT, IRON, CLR_BLACK, ...)
    ObjectEntry(
        name="orcish ring mail",
        description="crude ring mail",
        class_=ObjectClass.ARMOR_CLASS,
        prob=19,
        weight=250,
        cost=80,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=2,   # 10 - 8
        oc2=1,
        nutrition=250,
        color=Color.CLR_BLACK,
        material=Material.IRON,
    ),

    # ARMOR("leather jacket", NoDes, 1, 0, 0, 0, 11, 0, 30, 10, 9, 0, ARM_SUIT, LEATHER, CLR_BLACK, ...)
    ObjectEntry(
        name="leather jacket",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=11,
        weight=30,
        cost=10,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=30,
        color=Color.CLR_BLACK,
        material=Material.LEATHER,
    ),

    # ARMOR("Hawaiian shirt", NoDes, 1, 0, 0, 0, 8, 0, 5, 3, 10, 0, ARM_SHIRT, CLOTH, CLR_MAGENTA, ...)
    ObjectEntry(
        name="Hawaiian shirt",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=8,
        weight=5,
        cost=3,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,   # 10 - 10
        oc2=0,
        nutrition=5,
        color=Color.CLR_MAGENTA,
        material=Material.CLOTH,
    ),

    # ARMOR("T-shirt", NoDes, 1, 0, 0, 0, 2, 0, 5, 2, 10, 0, ARM_SHIRT, CLOTH, CLR_WHITE, ...)
    ObjectEntry(
        name="T-shirt",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=2,
        weight=5,
        cost=2,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,   # 10 - 10
        oc2=0,
        nutrition=5,
        color=Color.CLR_WHITE,
        material=Material.CLOTH,
    ),

    # -----------------------------------------------------------------------
    # CLOAKS (ARM_CLOAK)
    # CLOAK(name, desc, kn, mgc, power, prob, delay, wt, cost, ac, can, metal, c, sn)
    # -----------------------------------------------------------------------

    # CLOAK("mummy wrapping", NoDes, 1, 0, 0, 0, 0, 3, 2, 10, 1, CLOTH, CLR_GRAY, ...)
    ObjectEntry(
        name="mummy wrapping",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=0,
        weight=3,
        cost=2,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,   # 10 - 10
        oc2=1,
        nutrition=3,
        color=Color.CLR_GRAY,
        material=Material.CLOTH,
    ),

    # CLOAK("orcish cloak", "coarse mantelet", 0, 0, 0, 8, 0, 10, 40, 10, 1, CLOTH, CLR_BLACK, ...)
    ObjectEntry(
        name="orcish cloak",
        description="coarse mantelet",
        class_=ObjectClass.ARMOR_CLASS,
        prob=8,
        weight=10,
        cost=40,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,   # 10 - 10
        oc2=1,
        nutrition=10,
        color=Color.CLR_BLACK,
        material=Material.CLOTH,
    ),

    # CLOAK("dwarvish cloak", "hooded cloak", 0, 0, 0, 8, 0, 10, 50, 10, 1, CLOTH, HI_CLOTH, ...)
    ObjectEntry(
        name="dwarvish cloak",
        description="hooded cloak",
        class_=ObjectClass.ARMOR_CLASS,
        prob=8,
        weight=10,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=0,   # 10 - 10
        oc2=1,
        nutrition=10,
        color=Color.HI_CLOTH,
        material=Material.CLOTH,
    ),

    # CLOAK("oilskin cloak", "slippery cloak", 0, 0, 0, 8, 0, 10, 50, 9, 2, CLOTH, HI_CLOTH, ...)
    ObjectEntry(
        name="oilskin cloak",
        description="slippery cloak",
        class_=ObjectClass.ARMOR_CLASS,
        prob=8,
        weight=10,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=2,
        nutrition=10,
        color=Color.HI_CLOTH,
        material=Material.CLOTH,
    ),

    # CLOAK("robe", NoDes, 1, 1, 0, 6, 0, 15, 50, 8, 2, CLOTH, CLR_RED, ...)
    ObjectEntry(
        name="robe",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=6,
        weight=15,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=2,   # 10 - 8
        oc2=2,
        nutrition=15,
        color=Color.CLR_RED,
        material=Material.CLOTH,
    ),

    # CLOAK("alchemy smock", "apron", 0, 1, POISON_RES, 11, 0, 10, 50, 9, 1, CLOTH, CLR_WHITE, ...)
    ObjectEntry(
        name="alchemy smock",
        description="apron",
        class_=ObjectClass.ARMOR_CLASS,
        prob=11,
        weight=10,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=1,
        nutrition=10,
        color=Color.CLR_WHITE,
        material=Material.CLOTH,
    ),

    # CLOAK("leather cloak", NoDes, 1, 0, 0, 8, 0, 15, 40, 9, 1, LEATHER, CLR_BROWN, ...)
    ObjectEntry(
        name="leather cloak",
        description=None,
        class_=ObjectClass.ARMOR_CLASS,
        prob=8,
        weight=15,
        cost=40,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=1,
        nutrition=15,
        color=Color.CLR_BROWN,
        material=Material.LEATHER,
    ),

    # CLOAK("cloak of invisibility", "opera cloak", 0, 1, INVIS, 12, 0, 10, 60, 9, 1, CLOTH, CLR_BRIGHT_MAGENTA, ...)
    ObjectEntry(
        name="cloak of invisibility",
        description="opera cloak",
        class_=ObjectClass.ARMOR_CLASS,
        prob=12,
        weight=10,
        cost=60,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=1,
        nutrition=10,
        color=Color.CLR_BRIGHT_MAGENTA,
        material=Material.CLOTH,
    ),

    # CLOAK("cloak of magic resistance", "ornamental cope", 0, 1, ANTIMAGIC, 6, 0, 10, 60, 9, 1, CLOTH, CLR_WHITE, ...)
    ObjectEntry(
        name="cloak of magic resistance",
        description="ornamental cope",
        class_=ObjectClass.ARMOR_CLASS,
        prob=6,
        weight=10,
        cost=60,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=1,
        nutrition=10,
        color=Color.CLR_WHITE,
        material=Material.CLOTH,
    ),

    # CLOAK("cloak of displacement", "piece of cloth", 0, 1, DISPLACED, 12, 0, 10, 50, 9, 1, CLOTH, HI_CLOTH, ...)
    ObjectEntry(
        name="cloak of displacement",
        description="piece of cloth",
        class_=ObjectClass.ARMOR_CLASS,
        prob=12,
        weight=10,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=1,
        nutrition=10,
        color=Color.HI_CLOTH,
        material=Material.CLOTH,
    ),

    # -----------------------------------------------------------------------
    # SHIELDS (ARM_SHIELD)
    # SHIELD(name, desc, kn, mgc, blk, pow, prob, delay, wt, cost, ac, can, metal, c, sn)
    # -----------------------------------------------------------------------

    # SHIELD("shield of drain resistance", "wooden shield", 0, 1, 0, DRAIN_RES, 12, 0, 30, 50, 9, 0, WOOD, HI_WOOD, ...)
    ObjectEntry(
        name="shield of drain resistance",
        description="wooden shield",
        class_=ObjectClass.ARMOR_CLASS,
        prob=12,
        weight=30,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=30,
        color=Color.HI_WOOD,
        material=Material.WOOD,
    ),

    # SHIELD("shield of shock resistance", "wooden shield", 0, 1, 0, SHOCK_RES, 12, 0, 30, 50, 9, 0, WOOD, HI_WOOD, ...)
    ObjectEntry(
        name="shield of shock resistance",
        description="wooden shield",
        class_=ObjectClass.ARMOR_CLASS,
        prob=12,
        weight=30,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=30,
        color=Color.HI_WOOD,
        material=Material.WOOD,
    ),

    # SHIELD("Uruk-hai shield", "white-handed shield", 0, 0, 0, 0, 2, 0, 50, 7, 9, 0, IRON, HI_METAL, ...)
    ObjectEntry(
        name="Uruk-hai shield",
        description="white-handed shield",
        class_=ObjectClass.ARMOR_CLASS,
        prob=2,
        weight=50,
        cost=7,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=50,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # SHIELD("orcish shield", "red-eyed shield", 0, 0, 0, 0, 2, 0, 50, 7, 9, 0, IRON, CLR_RED, ...)
    ObjectEntry(
        name="orcish shield",
        description="red-eyed shield",
        class_=ObjectClass.ARMOR_CLASS,
        prob=2,
        weight=50,
        cost=7,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=50,
        color=Color.CLR_RED,
        material=Material.IRON,
    ),

    # SHIELD("shield of reflection", "polished silver shield", 0, 1, 0, REFLECTING, 7, 0, 50, 50, 8, 0, SILVER, HI_SILVER, ...)
    ObjectEntry(
        name="shield of reflection",
        description="polished silver shield",
        class_=ObjectClass.ARMOR_CLASS,
        prob=7,
        weight=50,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=2,   # 10 - 8
        oc2=0,
        nutrition=50,
        color=Color.HI_SILVER,
        material=Material.SILVER,
    ),

    # -----------------------------------------------------------------------
    # GLOVES (ARM_GLOVES)
    # GLOVES(name, desc, kn, mgc, power, prob, delay, wt, cost, ac, can, metal, c, sn)
    # -----------------------------------------------------------------------

    # GLOVES("gauntlets of power", "riding gloves", 0, 1, 0, 8, 1, 30, 50, 9, 0, IRON, CLR_BROWN, ...)
    ObjectEntry(
        name="gauntlets of power",
        description="riding gloves",
        class_=ObjectClass.ARMOR_CLASS,
        prob=8,
        weight=30,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=30,
        color=Color.CLR_BROWN,
        material=Material.IRON,
    ),

    # GLOVES("gauntlets of dexterity", "fencing gloves", 0, 1, 0, 8, 1, 10, 50, 9, 0, LEATHER, HI_LEATHER, ...)
    ObjectEntry(
        name="gauntlets of dexterity",
        description="fencing gloves",
        class_=ObjectClass.ARMOR_CLASS,
        prob=8,
        weight=10,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=10,
        color=Color.HI_LEATHER,
        material=Material.LEATHER,
    ),

    # -----------------------------------------------------------------------
    # BOOTS (ARM_BOOTS)
    # BOOTS(name, desc, kn, mgc, power, prob, delay, wt, cost, ac, can, metal, c, sn)
    # -----------------------------------------------------------------------

    # BOOTS("speed boots", "combat boots", 0, 1, FAST, 12, 2, 20, 50, 9, 0, LEATHER, HI_LEATHER, ...)
    ObjectEntry(
        name="speed boots",
        description="combat boots",
        class_=ObjectClass.ARMOR_CLASS,
        prob=12,
        weight=20,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=20,
        color=Color.HI_LEATHER,
        material=Material.LEATHER,
    ),

    # BOOTS("water walking boots", "jungle boots", 0, 1, WWALKING, 12, 2, 15, 50, 9, 0, LEATHER, HI_LEATHER, ...)
    ObjectEntry(
        name="water walking boots",
        description="jungle boots",
        class_=ObjectClass.ARMOR_CLASS,
        prob=12,
        weight=15,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=15,
        color=Color.HI_LEATHER,
        material=Material.LEATHER,
    ),

    # BOOTS("jumping boots", "hiking boots", 0, 1, JUMPING, 12, 2, 20, 50, 9, 0, LEATHER, HI_LEATHER, ...)
    ObjectEntry(
        name="jumping boots",
        description="hiking boots",
        class_=ObjectClass.ARMOR_CLASS,
        prob=12,
        weight=20,
        cost=50,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=20,
        color=Color.HI_LEATHER,
        material=Material.LEATHER,
    ),

    # BOOTS("elven boots", "mud boots", 0, 1, STEALTH, 12, 2, 15, 8, 9, 0, LEATHER, HI_LEATHER, ...)
    ObjectEntry(
        name="elven boots",
        description="mud boots",
        class_=ObjectClass.ARMOR_CLASS,
        prob=12,
        weight=15,
        cost=8,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=15,
        color=Color.HI_LEATHER,
        material=Material.LEATHER,
    ),

    # BOOTS("kicking boots", "buckled boots", 0, 1, 0, 12, 2, 50, 8, 9, 0, IRON, CLR_BROWN, ...)
    ObjectEntry(
        name="kicking boots",
        description="buckled boots",
        class_=ObjectClass.ARMOR_CLASS,
        prob=12,
        weight=50,
        cost=8,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=50,
        color=Color.CLR_BROWN,
        material=Material.IRON,
    ),

    # BOOTS("fumble boots", "riding boots", 0, 1, FUMBLING, 12, 2, 20, 30, 9, 0, LEATHER, HI_LEATHER, ...)
    ObjectEntry(
        name="fumble boots",
        description="riding boots",
        class_=ObjectClass.ARMOR_CLASS,
        prob=12,
        weight=20,
        cost=30,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=20,
        color=Color.HI_LEATHER,
        material=Material.LEATHER,
    ),

    # BOOTS("levitation boots", "snow boots", 0, 1, LEVITATION, 12, 2, 15, 30, 9, 0, LEATHER, HI_LEATHER, ...)
    ObjectEntry(
        name="levitation boots",
        description="snow boots",
        class_=ObjectClass.ARMOR_CLASS,
        prob=12,
        weight=15,
        cost=30,
        sdam=(0, 0),
        ldam=(0, 0),
        oc1=1,   # 10 - 9
        oc2=0,
        nutrition=15,
        color=Color.HI_LEATHER,
        material=Material.LEATHER,
    ),
)
