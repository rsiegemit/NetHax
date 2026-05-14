"""Missing weapons — vendor/nethack/include/objects.h WEAPON/BOW entries."""
from Nethax.nethax.constants.objects import ObjectEntry, ObjectClass, Material, Color

# ---------------------------------------------------------------------------
# Field mapping notes (WEAPON macro)
# WEAPON(name,desc,kn,mg,bi,prob,wt,cost,sdam,ldam,hitbon,typ,sub,metal,color,sn)
#   → sdam=(1,sdam_val), ldam=(1,ldam_val)
#   → oc1=hitbon, oc2=0
#   → nutrition=wt  (weapons store wt in oc_nutrition)
#
# BOW(name,desc,kn,prob,wt,cost,hitbon,metal,sub,color,sn)
#   → sdam=(1,2), ldam=(1,2)  (hardcoded 2,2 in BOW macro)
#   → oc1=hitbon, oc2=0
#   → nutrition=wt
#
# Color aliases:
#   HI_METAL  = CLR_CYAN (6)
#   HI_WOOD   = CLR_BROWN (3)
#   HI_SILVER = CLR_GRAY (7)
#   HI_LEATHER= CLR_BROWN (3)
# ---------------------------------------------------------------------------

ENTRIES = (

    # WEAPON("boomerang", NoDes, 1, 1, 0, 15, 5, 20, 9, 9, 0, 0, -P_BOOMERANG, WOOD, HI_WOOD, BOOMERANG)
    ObjectEntry(
        name="boomerang",
        description=None,
        class_=ObjectClass.WEAPON_CLASS,
        prob=15,
        weight=5,
        cost=20,
        sdam=(1, 9),
        ldam=(1, 9),
        oc1=0,
        oc2=0,
        nutrition=5,
        color=Color.HI_WOOD,
        material=Material.WOOD,
    ),

    # WEAPON("silver dagger", NoDes, 1, 1, 0, 3, 12, 40, 4, 3, 2, P, P_DAGGER, SILVER, HI_SILVER, SILVER_DAGGER)
    ObjectEntry(
        name="silver dagger",
        description=None,
        class_=ObjectClass.WEAPON_CLASS,
        prob=3,
        weight=12,
        cost=40,
        sdam=(1, 4),
        ldam=(1, 3),
        oc1=2,
        oc2=0,
        nutrition=12,
        color=Color.HI_SILVER,
        material=Material.SILVER,
    ),

    # WEAPON("stiletto", NoDes, 1, 1, 0, 5, 5, 4, 3, 2, 0, P|S, P_KNIFE, IRON, HI_METAL, STILETTO)
    ObjectEntry(
        name="stiletto",
        description=None,
        class_=ObjectClass.WEAPON_CLASS,
        prob=5,
        weight=5,
        cost=4,
        sdam=(1, 3),
        ldam=(1, 2),
        oc1=0,
        oc2=0,
        nutrition=5,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("worm tooth", NoDes, 1, 1, 0, 0, 20, 2, 2, 2, 0, 0, P_KNIFE, BONE, CLR_WHITE, WORM_TOOTH)
    ObjectEntry(
        name="worm tooth",
        description=None,
        class_=ObjectClass.WEAPON_CLASS,
        prob=0,
        weight=20,
        cost=2,
        sdam=(1, 2),
        ldam=(1, 2),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=Color.CLR_WHITE,
        material=Material.BONE,
    ),

    # WEAPON("crysknife", NoDes, 1, 1, 0, 0, 20, 100, 10, 10, 3, P, P_KNIFE, BONE, CLR_WHITE, CRYSKNIFE)
    ObjectEntry(
        name="crysknife",
        description=None,
        class_=ObjectClass.WEAPON_CLASS,
        prob=0,
        weight=20,
        cost=100,
        sdam=(1, 10),
        ldam=(1, 10),
        oc1=3,
        oc2=0,
        nutrition=20,
        color=Color.CLR_WHITE,
        material=Material.BONE,
    ),

    # WEAPON("scimitar", "curved sword", 0, 0, 0, 15, 40, 15, 8, 8, 0, S, P_SABER, IRON, HI_METAL, SCIMITAR)
    ObjectEntry(
        name="scimitar",
        description="curved sword",
        class_=ObjectClass.WEAPON_CLASS,
        prob=15,
        weight=40,
        cost=15,
        sdam=(1, 8),
        ldam=(1, 8),
        oc1=0,
        oc2=0,
        nutrition=40,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("silver saber", NoDes, 1, 0, 0, 6, 40, 75, 8, 8, 0, S, P_SABER, SILVER, HI_SILVER, SILVER_SABER)
    ObjectEntry(
        name="silver saber",
        description=None,
        class_=ObjectClass.WEAPON_CLASS,
        prob=6,
        weight=40,
        cost=75,
        sdam=(1, 8),
        ldam=(1, 8),
        oc1=0,
        oc2=0,
        nutrition=40,
        color=Color.HI_SILVER,
        material=Material.SILVER,
    ),

    # WEAPON("elven broadsword", "runed broadsword", 0, 0, 0, 4, 70, 10, 6, 6, 0, S, P_BROAD_SWORD, WOOD, HI_WOOD, ELVEN_BROADSWORD)
    ObjectEntry(
        name="elven broadsword",
        description="runed broadsword",
        class_=ObjectClass.WEAPON_CLASS,
        prob=4,
        weight=70,
        cost=10,
        sdam=(1, 6),
        ldam=(1, 6),
        oc1=0,
        oc2=0,
        nutrition=70,
        color=Color.HI_WOOD,
        material=Material.WOOD,
    ),

    # WEAPON("runesword", "runed broadsword", 0, 0, 0, 0, 40, 300, 4, 6, 0, S, P_BROAD_SWORD, IRON, CLR_BLACK, RUNESWORD)
    ObjectEntry(
        name="runesword",
        description="runed broadsword",
        class_=ObjectClass.WEAPON_CLASS,
        prob=0,
        weight=40,
        cost=300,
        sdam=(1, 4),
        ldam=(1, 6),
        oc1=0,
        oc2=0,
        nutrition=40,
        color=Color.CLR_BLACK,
        material=Material.IRON,
    ),

    # WEAPON("partisan", "vulgar polearm", 0, 0, 1, 5, 80, 10, 6, 6, 0, P, P_POLEARMS, IRON, HI_METAL, PARTISAN)
    ObjectEntry(
        name="partisan",
        description="vulgar polearm",
        class_=ObjectClass.WEAPON_CLASS,
        prob=5,
        weight=80,
        cost=10,
        sdam=(1, 6),
        ldam=(1, 6),
        oc1=0,
        oc2=0,
        nutrition=80,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("ranseur", "hilted polearm", 0, 0, 1, 5, 50, 6, 4, 4, 0, P, P_POLEARMS, IRON, HI_METAL, RANSEUR)
    ObjectEntry(
        name="ranseur",
        description="hilted polearm",
        class_=ObjectClass.WEAPON_CLASS,
        prob=5,
        weight=50,
        cost=6,
        sdam=(1, 4),
        ldam=(1, 4),
        oc1=0,
        oc2=0,
        nutrition=50,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("spetum", "forked polearm", 0, 0, 1, 5, 50, 5, 6, 6, 0, P, P_POLEARMS, IRON, HI_METAL, SPETUM)
    ObjectEntry(
        name="spetum",
        description="forked polearm",
        class_=ObjectClass.WEAPON_CLASS,
        prob=5,
        weight=50,
        cost=5,
        sdam=(1, 6),
        ldam=(1, 6),
        oc1=0,
        oc2=0,
        nutrition=50,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("glaive", "single-edged polearm", 0, 0, 1, 8, 75, 6, 6, 10, 0, S, P_POLEARMS, IRON, HI_METAL, GLAIVE)
    ObjectEntry(
        name="glaive",
        description="single-edged polearm",
        class_=ObjectClass.WEAPON_CLASS,
        prob=8,
        weight=75,
        cost=6,
        sdam=(1, 6),
        ldam=(1, 10),
        oc1=0,
        oc2=0,
        nutrition=75,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("halberd", "angled poleaxe", 0, 0, 1, 8, 150, 10, 10, 6, 0, P|S, P_POLEARMS, IRON, HI_METAL, HALBERD)
    ObjectEntry(
        name="halberd",
        description="angled poleaxe",
        class_=ObjectClass.WEAPON_CLASS,
        prob=8,
        weight=150,
        cost=10,
        sdam=(1, 10),
        ldam=(1, 6),
        oc1=0,
        oc2=0,
        nutrition=150,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("bardiche", "long poleaxe", 0, 0, 1, 4, 120, 7, 4, 4, 0, S, P_POLEARMS, IRON, HI_METAL, BARDICHE)
    ObjectEntry(
        name="bardiche",
        description="long poleaxe",
        class_=ObjectClass.WEAPON_CLASS,
        prob=4,
        weight=120,
        cost=7,
        sdam=(1, 4),
        ldam=(1, 4),
        oc1=0,
        oc2=0,
        nutrition=120,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("voulge", "pole cleaver", 0, 0, 1, 4, 125, 5, 4, 4, 0, S, P_POLEARMS, IRON, HI_METAL, VOULGE)
    ObjectEntry(
        name="voulge",
        description="pole cleaver",
        class_=ObjectClass.WEAPON_CLASS,
        prob=4,
        weight=125,
        cost=5,
        sdam=(1, 4),
        ldam=(1, 4),
        oc1=0,
        oc2=0,
        nutrition=125,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("fauchard", "pole sickle", 0, 0, 1, 6, 60, 5, 6, 8, 0, P|S, P_POLEARMS, IRON, HI_METAL, FAUCHARD)
    ObjectEntry(
        name="fauchard",
        description="pole sickle",
        class_=ObjectClass.WEAPON_CLASS,
        prob=6,
        weight=60,
        cost=5,
        sdam=(1, 6),
        ldam=(1, 8),
        oc1=0,
        oc2=0,
        nutrition=60,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("guisarme", "pruning hook", 0, 0, 1, 6, 80, 5, 4, 8, 0, S, P_POLEARMS, IRON, HI_METAL, GUISARME)
    ObjectEntry(
        name="guisarme",
        description="pruning hook",
        class_=ObjectClass.WEAPON_CLASS,
        prob=6,
        weight=80,
        cost=5,
        sdam=(1, 4),
        ldam=(1, 8),
        oc1=0,
        oc2=0,
        nutrition=80,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("bill-guisarme", "hooked polearm", 0, 0, 1, 4, 120, 7, 4, 10, 0, P|S, P_POLEARMS, IRON, HI_METAL, BILL_GUISARME)
    ObjectEntry(
        name="bill-guisarme",
        description="hooked polearm",
        class_=ObjectClass.WEAPON_CLASS,
        prob=4,
        weight=120,
        cost=7,
        sdam=(1, 4),
        ldam=(1, 10),
        oc1=0,
        oc2=0,
        nutrition=120,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("lucern hammer", "pronged polearm", 0, 0, 1, 5, 150, 7, 4, 6, 0, B|P, P_POLEARMS, IRON, HI_METAL, LUCERN_HAMMER)
    ObjectEntry(
        name="lucern hammer",
        description="pronged polearm",
        class_=ObjectClass.WEAPON_CLASS,
        prob=5,
        weight=150,
        cost=7,
        sdam=(1, 4),
        ldam=(1, 6),
        oc1=0,
        oc2=0,
        nutrition=150,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("bec de corbin", "beaked polearm", 0, 0, 1, 4, 100, 8, 8, 6, 0, B|P, P_POLEARMS, IRON, HI_METAL, BEC_DE_CORBIN)
    ObjectEntry(
        name="bec de corbin",
        description="beaked polearm",
        class_=ObjectClass.WEAPON_CLASS,
        prob=4,
        weight=100,
        cost=8,
        sdam=(1, 8),
        ldam=(1, 6),
        oc1=0,
        oc2=0,
        nutrition=100,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("dwarvish mattock", "broad pick", 0, 0, 1, 13, 120, 50, 12, 8, -1, B, P_PICK_AXE, IRON, HI_METAL, DWARVISH_MATTOCK)
    ObjectEntry(
        name="dwarvish mattock",
        description="broad pick",
        class_=ObjectClass.WEAPON_CLASS,
        prob=13,
        weight=120,
        cost=50,
        sdam=(1, 12),
        ldam=(1, 8),
        oc1=-1,
        oc2=0,
        nutrition=120,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("lance", NoDes, 1, 0, 0, 4, 180, 10, 6, 8, 0, P, P_LANCE, IRON, HI_METAL, LANCE)
    ObjectEntry(
        name="lance",
        description=None,
        class_=ObjectClass.WEAPON_CLASS,
        prob=4,
        weight=180,
        cost=10,
        sdam=(1, 6),
        ldam=(1, 8),
        oc1=0,
        oc2=0,
        nutrition=180,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("silver mace", NoDes, 1, 0, 0, 2, 36, 60, 6, 6, 0, B, P_MACE, SILVER, HI_SILVER, SILVER_MACE)
    ObjectEntry(
        name="silver mace",
        description=None,
        class_=ObjectClass.WEAPON_CLASS,
        prob=2,
        weight=36,
        cost=60,
        sdam=(1, 6),
        ldam=(1, 6),
        oc1=0,
        oc2=0,
        nutrition=36,
        color=Color.HI_SILVER,
        material=Material.SILVER,
    ),

    # WEAPON("rubber hose", NoDes, 1, 0, 0, 0, 20, 3, 4, 3, 0, B, P_WHIP, PLASTIC, CLR_BROWN, RUBBER_HOSE)
    ObjectEntry(
        name="rubber hose",
        description=None,
        class_=ObjectClass.WEAPON_CLASS,
        prob=0,
        weight=20,
        cost=3,
        sdam=(1, 4),
        ldam=(1, 3),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=Color.CLR_BROWN,
        material=Material.PLASTIC,
    ),

    # WEAPON("quarterstaff", "staff", 0, 0, 1, 11, 40, 5, 6, 6, 0, B, P_QUARTERSTAFF, WOOD, HI_WOOD, QUARTERSTAFF)
    ObjectEntry(
        name="quarterstaff",
        description="staff",
        class_=ObjectClass.WEAPON_CLASS,
        prob=11,
        weight=40,
        cost=5,
        sdam=(1, 6),
        ldam=(1, 6),
        oc1=0,
        oc2=0,
        nutrition=40,
        color=Color.HI_WOOD,
        material=Material.WOOD,
    ),

    # WEAPON("flail", NoDes, 1, 0, 0, 40, 15, 4, 6, 4, 0, B, P_FLAIL, IRON, HI_METAL, FLAIL)
    ObjectEntry(
        name="flail",
        description=None,
        class_=ObjectClass.WEAPON_CLASS,
        prob=40,
        weight=15,
        cost=4,
        sdam=(1, 6),
        ldam=(1, 4),
        oc1=0,
        oc2=0,
        nutrition=15,
        color=Color.HI_METAL,
        material=Material.IRON,
    ),

    # WEAPON("bullwhip", NoDes, 1, 0, 0, 2, 20, 4, 2, 1, 0, 0, P_WHIP, LEATHER, CLR_BROWN, BULLWHIP)
    ObjectEntry(
        name="bullwhip",
        description=None,
        class_=ObjectClass.WEAPON_CLASS,
        prob=2,
        weight=20,
        cost=4,
        sdam=(1, 2),
        ldam=(1, 1),
        oc1=0,
        oc2=0,
        nutrition=20,
        color=Color.CLR_BROWN,
        material=Material.LEATHER,
    ),

    # BOW("sling", NoDes, 1, 40, 3, 20, 0, LEATHER, P_SLING, HI_LEATHER, SLING)
    # BOW macro: sdam=2, ldam=2 (hardcoded); HI_LEATHER = CLR_BROWN
    ObjectEntry(
        name="sling",
        description=None,
        class_=ObjectClass.WEAPON_CLASS,
        prob=40,
        weight=3,
        cost=20,
        sdam=(1, 2),
        ldam=(1, 2),
        oc1=0,
        oc2=0,
        nutrition=3,
        color=Color.HI_LEATHER,
        material=Material.LEATHER,
    ),
)
