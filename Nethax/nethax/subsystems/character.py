"""Character creation — role/race starting inventory, stats, HP/PW.

Canonical sources:
  vendor/nethack/src/u_init.c::ini_inv   — per-role starting items
  vendor/nethack/src/role.c::roles[]     — 13 role stat/HP/PW tables
  vendor/nethack/src/role.c::races[]     — 5 race stat tables

Status: create_character implemented with vendor init_attr / ini_hpwp
formulae (attrib.c:614-660, u_init.c:1390).  Starting pet table and
per-role starting spells exposed via get_starting_pet /
get_starting_spells; race-aware starting items live in u_init helpers.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.constants.roles import (
    Role,
    ROLES,
    get_role,
)
from Nethax.nethax.constants.races import (
    Race,
    RACES,
    get_race,
)
from Nethax.nethax.subsystems.inventory import (
    ArmorSlot,
    ItemCategory,
    InventoryState,
    make_item,
    make_empty_item,
    _items_from_list,
    compute_ac,
    BASE_AC,
    MAX_INVENTORY_SLOTS,
)
from Nethax.nethax.subsystems.magic import MagicState, SpellId, N_SPELLS, MAX_SPELL_MEMORY


# ---------------------------------------------------------------------------
# Object type IDs
# Sourced from vendor/nethack/include/objects.h enumeration order.
# These are stable integer IDs used in Item.type_id.
# We use small positive integers matching the NetHack otyp values.
# For our purposes the exact value only needs to be internally consistent;
# tests compare category + ac_bonus, not raw type_id.
# ---------------------------------------------------------------------------

class ObjType:
    """Canonical NetHack object type IDs (otyp field).

    Partial list — only objects referenced in starting inventories.
    All values match OBJECTS[i] index in Nethax/nethax/constants/objects.py,
    which is vendor-index aligned with vendor/nle/src/objects.c.
    """
    # Weapons (WEAPON_CLASS = 2)
    # Canonical source: constants/objects.py indices 1-70
    ARROW             = 1    # objects.py:202
    ELVEN_ARROW       = 2    # objects.py:222
    ORCISH_ARROW      = 3    # objects.py:242
    SILVER_ARROW      = 4    # objects.py:262
    YA                = 5    # objects.py:282
    CROSSBOW_BOLT     = 6    # objects.py:302
    DART              = 7    # objects.py:322
    SHURIKEN          = 8    # objects.py:342
    BOOMERANG         = 9    # objects.py:362
    SPEAR             = 10   # objects.py:382
    ELVEN_SPEAR       = 11   # objects.py:402
    ORCISH_SPEAR      = 12   # objects.py:422
    DWARVISH_SPEAR    = 13   # objects.py:442
    SILVER_SPEAR      = 14   # objects.py:462
    JAVELIN           = 15   # objects.py:482
    TRIDENT           = 16   # objects.py:502
    DAGGER            = 17   # objects.py:522
    ELVEN_DAGGER      = 18   # objects.py:542
    ORCISH_DAGGER     = 19   # objects.py:562
    SILVER_DAGGER     = 20   # objects.py:582
    # 21 = athame (not in starting inv)
    SCALPEL           = 22   # objects.py:622
    KNIFE             = 23   # objects.py:642
    STILETTO          = 24   # objects.py:662
    WORM_TOOTH        = 25   # objects.py:682
    CRYSKNIFE         = 26   # objects.py:703
    AXE               = 27   # objects.py:724  single-handed axe (Barbarian_0)
    BATTLE_AXE        = 28   # objects.py:744
    SHORT_SWORD       = 29   # objects.py:764
    ELVEN_SHORT_SWORD = 30   # objects.py:784
    ORCISH_SHORT_SWORD = 31  # objects.py:804
    DWARVISH_SHORT_SWORD = 32  # objects.py:824
    # 33 = scimitar (not in starting inv)
    SABER             = 34   # objects.py:864  silver saber
    BROADSWORD        = 35   # objects.py:884
    # 36 = elven broadsword (not in starting inv)
    LONG_SWORD        = 37   # objects.py:924
    TWO_HANDED_SWORD  = 38   # objects.py:944
    KATANA            = 39   # objects.py:964
    TSURUGI           = 40   # objects.py:984
    # 41 = runesword, 42-45 = pole arms (partisan/ranseur/spetum/glaive)
    LANCE             = 46   # objects.py:1104
    # 47-55 = halberd/bardiche/voulge/mattock/fauchard/guisarme/bill-guisarme/
    #         lucern hammer/bec de corbin (not in starting inv)
    MACE              = 56   # objects.py:1304
    MORNING_STAR      = 57   # objects.py:1324
    WAR_HAMMER        = 58   # objects.py:1344
    CLUB              = 59   # objects.py:1364
    RUBBER_HOSE       = 60   # objects.py:1384
    QUARTERSTAFF      = 61   # objects.py:1404
    AKLYS             = 62   # objects.py:1424
    # 63 = flail (not in starting inv)
    BULLWHIP          = 64   # objects.py:1464
    BOW               = 65   # objects.py:1484
    ELVEN_BOW         = 66   # objects.py:1504
    ORCISH_BOW        = 67   # objects.py:1524
    YUMI              = 68   # objects.py:1544
    SLING             = 69   # objects.py:1564
    CROSSBOW          = 70   # objects.py:1584
    PICK_AXE          = 234  # objects.py:4864  WEPTOOL_CLASS

    # Armor — Helms (ARMOR_CLASS = 3)
    # Canonical source: constants/objects.py indices 71-81
    ELVEN_LEATHER_HELM        = 71   # objects.py:1604
    ORCISH_HELM               = 72   # objects.py:1624
    DWARVISH_IRON_HELM        = 73   # objects.py:1644
    FEDORA                    = 74   # objects.py:1664
    CORNUTHAUM                = 75   # objects.py:1684
    DUNCE_CAP                 = 76   # objects.py:1704
    # 77 = dented pot (not in starting inv)
    HELMET                    = 78   # objects.py:1744
    HELM_OF_BRILLIANCE        = 79   # objects.py:1764
    HELM_OF_OPPOSITE_ALIGNMENT = 80  # objects.py:1784
    HELM_OF_TELEPATHY         = 81   # objects.py:1804

    # Dragon scale mails: constants/objects.py indices 82-90
    # NLE binary has 9 DSMs: gray/silver/red/white/orange/black/blue/green/yellow
    # (no gold or shimmering — those are 3.7-only, absent from NLE objects table)
    GRAY_DRAGON_SCALE_MAIL    = 82   # objects.py:1824  ANTIMAGIC
    SILVER_DRAGON_SCALE_MAIL  = 83   # objects.py:1844  REFLECTING
    RED_DRAGON_SCALE_MAIL     = 84   # objects.py:1864  FIRE_RES
    WHITE_DRAGON_SCALE_MAIL   = 85   # objects.py:1884  COLD_RES
    ORANGE_DRAGON_SCALE_MAIL  = 86   # objects.py:1904  SLEEP_RES
    BLACK_DRAGON_SCALE_MAIL   = 87   # objects.py:1924  DISINT_RES
    BLUE_DRAGON_SCALE_MAIL    = 88   # objects.py:1944  SHOCK_RES
    GREEN_DRAGON_SCALE_MAIL   = 89   # objects.py:1964  POISON_RES
    YELLOW_DRAGON_SCALE_MAIL  = 90   # objects.py:1984  ACID_RES

    # Dragon scales: constants/objects.py indices 91-99
    GRAY_DRAGON_SCALES        = 91   # objects.py:2004  ANTIMAGIC
    SILVER_DRAGON_SCALES      = 92   # objects.py:2024  REFLECTING
    RED_DRAGON_SCALES         = 93   # objects.py:2044  FIRE_RES
    WHITE_DRAGON_SCALES       = 94   # objects.py:2064  COLD_RES
    ORANGE_DRAGON_SCALES      = 95   # objects.py:2084  SLEEP_RES
    BLACK_DRAGON_SCALES       = 96   # objects.py:2104  DISINT_RES
    BLUE_DRAGON_SCALES        = 97   # objects.py:2124  SHOCK_RES
    GREEN_DRAGON_SCALES       = 98   # objects.py:2144  POISON_RES
    YELLOW_DRAGON_SCALES      = 99   # objects.py:2164  ACID_RES

    # NLE-absent items: in vendor/nethack/include/objects.h 3.7 but NOT in NLE
    # binary. Sentinel=0 (ILLOBJ/strange object) — can never be worn/matched.
    GOLD_DRAGON_SCALE_MAIL    = 0    # absent from NLE objects table
    GOLD_DRAGON_SCALES        = 0    # absent from NLE objects table
    TINFOIL_HAT               = 0    # absent from NLE objects table
    PADDED_ARMOR              = 0    # absent from NLE objects table
    LEATHER_HELM              = 0    # absent from NLE objects table
    HELM_OF_CAUTION           = 0    # absent from NLE objects table

    # Body armor: constants/objects.py indices 100-116
    PLATE_MAIL                = 100  # objects.py:2184
    CRYSTAL_PLATE_MAIL        = 101  # objects.py:2204
    BRONZE_PLATE_MAIL         = 102  # objects.py:2224
    SPLINT_MAIL               = 103  # objects.py:2244
    BANDED_MAIL               = 104  # objects.py:2264
    DWARVISH_MITHRIL_COAT     = 105  # objects.py:2284
    ELVEN_MITHRIL_COAT        = 106  # objects.py:2304
    CHAIN_MAIL                = 107  # objects.py:2324
    # 108 = orcish chain mail (not in starting inv)
    SCALE_MAIL                = 109  # objects.py:2364
    STUDDED_LEATHER_ARMOR     = 110  # objects.py:2384
    RING_MAIL                 = 111  # objects.py:2404
    # 112 = orcish ring mail (not in starting inv)
    LEATHER_ARMOR             = 113  # objects.py:2444
    LEATHER_JACKET            = 114  # objects.py:2464
    HAWAIIAN_SHIRT            = 115  # objects.py:2484
    T_SHIRT                   = 116  # objects.py:2504

    # Cloaks: constants/objects.py indices 117-128
    MUMMY_WRAPPING            = 117  # objects.py:2524
    ELVEN_CLOAK               = 118  # objects.py:2544
    ORCISH_CLOAK              = 119  # objects.py:2564
    # 120 = dwarvish cloak (not in starting inv)
    OILSKIN_CLOAK             = 121  # objects.py:2604
    ROBE                      = 122  # objects.py:2624
    # 123 = alchemy smock, 124 = leather cloak (not in starting inv)
    CLOAK_OF_PROTECTION       = 125  # objects.py:2684
    CLOAK_OF_INVISIBILITY     = 126  # objects.py:2704
    CLOAK_OF_MAGIC_RESISTANCE = 127  # objects.py:2724
    CLOAK_OF_DISPLACEMENT     = 128  # objects.py:2744

    # Shields: constants/objects.py indices 129-134
    SMALL_SHIELD              = 129  # objects.py:2764
    ELVEN_SHIELD              = 130  # objects.py:2784
    URUK_HAI_SHIELD           = 131  # objects.py:2804
    ORCISH_SHIELD             = 132  # objects.py:2824
    LARGE_SHIELD              = 133  # objects.py:2844
    DWARVISH_ROUNDSHIELD      = 134  # objects.py:2864

    # Gloves: constants/objects.py indices 136-139
    LEATHER_GLOVES            = 136  # objects.py:2904
    GAUNTLETS_OF_FUMBLING     = 137  # objects.py:2924
    GAUNTLETS_OF_POWER        = 138  # objects.py:2944
    GAUNTLETS_OF_DEXTERITY    = 139  # objects.py:2964

    # Boots: constants/objects.py indices 140-149
    LOW_BOOTS                 = 140  # objects.py:2984
    IRON_SHOES                = 141  # objects.py:3004
    HIGH_BOOTS                = 142  # objects.py:3024
    SPEED_BOOTS               = 143  # objects.py:3044
    WATER_WALKING_BOOTS       = 144  # objects.py:3064
    JUMPING_BOOTS             = 145  # objects.py:3084
    ELVEN_BOOTS               = 146  # objects.py:3104
    KICKING_BOOTS             = 147  # objects.py:3124
    FUMBLE_BOOTS              = 148  # objects.py:3144
    LEVITATION_BOOTS          = 149  # objects.py:3164

    # Potions (POTION_CLASS = 8)
    # Canonical source: constants/objects.py indices 272-297
    # 272 = gain ability (not in starting inv)
    POT_RESTORE_ABILITY = 273  # objects.py:5644
    POT_CONFUSION       = 274  # objects.py:5664
    POT_BLINDNESS       = 275  # objects.py:5684
    POT_PARALYSIS       = 276  # objects.py:5704
    POT_SPEED           = 277  # objects.py:5724
    POT_LEVITATION      = 278  # objects.py:5744
    POT_HALLUCINATION   = 279  # objects.py:5764
    # 280 = invisibility, 281 = see invisible (not in starting inv)
    POT_HEALING         = 282  # objects.py:5824
    POT_EXTRA_HEALING   = 283  # objects.py:5844
    POT_GAIN_LEVEL      = 284  # objects.py:5864
    POT_ENLIGHTENMENT   = 285  # objects.py:5884
    # 286 = monster detection, 287 = object detection (not in starting inv)
    POT_GAIN_ENERGY     = 288  # objects.py:5944
    POT_SLEEPING        = 289  # objects.py:5964
    POT_FULL_HEALING    = 290  # objects.py:5984
    POT_POLYMORPH       = 291  # objects.py:6004
    POT_BOOZE           = 292  # objects.py:6024
    POT_SICKNESS        = 293  # objects.py:6044
    # 294 = fruit juice (not in starting inv)
    POT_ACID            = 295  # objects.py:6084
    POT_OIL             = 296  # objects.py:6104
    POT_WATER           = 297  # objects.py:6124
    # NLE-absent potions (3.7-only, not in NLE objects table) — sentinel=0
    POT_GAIN_STRENGTH   = 0    # absent from NLE objects table
    POT_INVULNERABILITY = 0    # absent from NLE objects table
    POT_SEE_INVISIBLE   = 0    # absent from NLE objects table

    # Scrolls (SCROLL_CLASS = 9)
    # Canonical source: constants/objects.py indices 298-312
    SCR_IDENTIFY        = 311  # objects.py:6404
    SCR_MAGIC_MAPPING   = 312  # objects.py:6424

    # Spellbooks (SPBOOK_CLASS = 10)
    # Canonical source: constants/objects.py indices 340-379
    SPE_FORCE_BOLT      = 350  # objects.py:7184
    SPE_HEALING         = 348  # objects.py:7144
    SPE_SLEEP           = 344  # objects.py:7064  (Monk M_spell[2])
    SPE_PROTECTION      = 377  # objects.py:7724  (Monk M_spell[1])
    SPE_EXTRA_HEALING   = 365  # objects.py:7484
    SPE_STONE_TO_FLESH  = 379  # objects.py:7764

    # Wands (WAND_CLASS = 11)
    # Canonical source: constants/objects.py indices 383-406
    WAN_SLEEP           = 404  # objects.py:8264

    # Tools (TOOL_CLASS = 6)
    # Canonical source: constants/objects.py indices 192-217
    SACK                = 192  # objects.py:4024
    LOCK_PICK           = 197  # objects.py:4124
    CREDIT_CARD         = 198  # objects.py:4144
    EXPENSIVE_CAMERA    = 204  # objects.py:4264
    # BLINDFOLD vendor otyp = 208 (objects.py:4344).
    BLINDFOLD           = 208  # objects.py:4344
    STETHOSCOPE         = 212  # objects.py:4424
    TINNING_KIT         = 213  # objects.py:4444
    MAGIC_MARKER        = 217  # objects.py:4524
    TOUCHSTONE          = 444  # objects.py:9066  ROCK_CLASS
    FLINT               = 445  # objects.py:9087  GEM_CLASS (Caveman ammo)
    ROCK                = 446  # objects.py:9108  GEM_CLASS (Caveman ammo)

    # Food (FOOD_CLASS = 7) — vendor otyp values
    # Canonical source: constants/objects.py indices 252-268
    TRIPE_RATION      = 239   # orc CRAM_RATION/LEMBAS_WAFER substitute
    SLIME_MOLD        = 260   # random FOOD_CLASS pick (orc Xtra_food)
    APPLE             = 252   # objects.py:5224
    ORANGE            = 253   # objects.py:5244
    CARROT            = 257   # objects.py:5324
    SPRIG_OF_WOLFSBANE = 258  # objects.py:5344
    CLOVE_OF_GARLIC   = 259   # objects.py:5364
    FORTUNE_COOKIE    = 264   # objects.py:5464
    LEMBAS_WAFER      = 266   # objects.py:5504
    CRAM_RATION       = 267   # objects.py:5524
    FOOD_RATION       = 268   # objects.py:5544


# ---------------------------------------------------------------------------
# AC values for starting armor
# Sources: vendor/nethack/include/objects.h (a_ac field per armor entry)
# ---------------------------------------------------------------------------

_ARMOR_AC = {
    # Shields
    ObjType.SMALL_SHIELD: 1,
    ObjType.LARGE_SHIELD: 2,
    ObjType.ELVEN_SHIELD: 2,
    ObjType.DWARVISH_ROUNDSHIELD: 2,
    ObjType.ORCISH_SHIELD: 1,
    ObjType.URUK_HAI_SHIELD: 2,
    # Body armor
    ObjType.LEATHER_ARMOR: 2,
    ObjType.LEATHER_JACKET: 1,
    ObjType.RING_MAIL: 3,
    ObjType.CHAIN_MAIL: 5,
    ObjType.SPLINT_MAIL: 6,
    ObjType.BANDED_MAIL: 6,
    ObjType.PLATE_MAIL: 7,
    ObjType.CRYSTAL_PLATE_MAIL: 7,
    ObjType.ELVEN_MITHRIL_COAT: 5,
    ObjType.DWARVISH_MITHRIL_COAT: 6,
    ObjType.PADDED_ARMOR: 1,
    ObjType.STUDDED_LEATHER_ARMOR: 3,
    ObjType.SCALE_MAIL: 4,
    ObjType.BRONZE_PLATE_MAIL: 6,
    # Helms
    ObjType.HELMET: 1,
    ObjType.DWARVISH_IRON_HELM: 2,
    ObjType.ORCISH_HELM: 1,
    ObjType.FEDORA: 0,
    ObjType.ELVEN_LEATHER_HELM: 1,
    # HELM_OF_CAUTION — objects.h:479-481 a_ac=9 → 10-9=1
    ObjType.HELM_OF_CAUTION: 1,
    # Dragon scale mails / scales — objects.h:497-553 DRGN_ARMR macro
    # a_ac=1 → 10-1=9.  All dragon armor entries share AC 9.
    ObjType.GRAY_DRAGON_SCALE_MAIL: 9,
    ObjType.GOLD_DRAGON_SCALE_MAIL: 9,
    ObjType.SILVER_DRAGON_SCALE_MAIL: 9,
    ObjType.RED_DRAGON_SCALE_MAIL: 9,
    ObjType.WHITE_DRAGON_SCALE_MAIL: 9,
    ObjType.ORANGE_DRAGON_SCALE_MAIL: 9,
    ObjType.BLACK_DRAGON_SCALE_MAIL: 9,
    ObjType.BLUE_DRAGON_SCALE_MAIL: 9,
    ObjType.GREEN_DRAGON_SCALE_MAIL: 9,
    ObjType.YELLOW_DRAGON_SCALE_MAIL: 9,
    ObjType.GRAY_DRAGON_SCALES: 9,
    ObjType.GOLD_DRAGON_SCALES: 9,
    ObjType.SILVER_DRAGON_SCALES: 9,
    ObjType.RED_DRAGON_SCALES: 9,
    ObjType.WHITE_DRAGON_SCALES: 9,
    ObjType.ORANGE_DRAGON_SCALES: 9,
    ObjType.BLACK_DRAGON_SCALES: 9,
    ObjType.BLUE_DRAGON_SCALES: 9,
    ObjType.GREEN_DRAGON_SCALES: 9,
    ObjType.YELLOW_DRAGON_SCALES: 9,
    # Gloves
    ObjType.LEATHER_GLOVES: 1,
    # Boots
    ObjType.HIGH_BOOTS: 2,
    ObjType.IRON_SHOES: 2,
    ObjType.KICKING_BOOTS: 1,
    # Cloaks
    ObjType.CLOAK_OF_MAGIC_RESISTANCE: 3,
    ObjType.CLOAK_OF_DISPLACEMENT: 3,
    ObjType.CLOAK_OF_PROTECTION: 3,
    ObjType.CLOAK_OF_INVISIBILITY: 2,
    ObjType.ELVEN_CLOAK: 2,
    ObjType.OILSKIN_CLOAK: 3,
    ObjType.ROBE: 2,
    ObjType.HAWAIIAN_SHIRT: 0,
    # Shirts
    ObjType.T_SHIRT: 0,
}

# Approximate weights in aum (from objects.h wt field)
_WEAPON_WEIGHT = {
    ObjType.DAGGER: 10, ObjType.SPEAR: 30, ObjType.LONG_SWORD: 40,
    ObjType.SHORT_SWORD: 30, ObjType.QUARTERSTAFF: 40, ObjType.MACE: 30,
    ObjType.CLUB: 30, ObjType.SLING: 3, ObjType.BOW: 30, ObjType.YUMI: 30,
    ObjType.ARROW: 1, ObjType.YA: 4, ObjType.DART: 1, ObjType.BULLWHIP: 20,
    ObjType.AXE: 30, ObjType.KATANA: 40, ObjType.TWO_HANDED_SWORD: 70, ObjType.BATTLE_AXE: 40,
    ObjType.LANCE: 180, ObjType.SCALPEL: 5, ObjType.PICK_AXE: 100,
}
_ARMOR_WEIGHT = {
    ObjType.RING_MAIL: 250, ObjType.SPLINT_MAIL: 400, ObjType.LEATHER_ARMOR: 150,
    ObjType.LEATHER_JACKET: 30, ObjType.SMALL_SHIELD: 30, ObjType.LARGE_SHIELD: 100,
    ObjType.HELMET: 30, ObjType.FEDORA: 3, ObjType.LEATHER_GLOVES: 8,
    ObjType.CLOAK_OF_MAGIC_RESISTANCE: 10, ObjType.CLOAK_OF_DISPLACEMENT: 10,
    ObjType.ROBE: 15, ObjType.HAWAIIAN_SHIRT: 5,
}


# BUC status integer codes (mirror Nethax.nethax.subsystems.inventory comment
# at Item.buc_status definition: 0=unknown / 1=cursed / 2=uncursed / 3=blessed).
# Used to encode vendor trbless values from u_init.c trobj arrays.
#   trbless == 0 or UNDEF_BLESS (2) → obj->cursed=0, blessed unchanged → UNCURSED
#   trbless == 1                    → obj->blessed=1                   → BLESSED
# Cite: vendor/nethack/src/u_init.c:1208-1244 (ini_inv_adjust_obj).
_BUC_UNCURSED = 2
_BUC_BLESSED  = 3


def _weapon(type_id: int, qty: int = 1, enchant: int = 0,
            buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    two_handed = type_id in (ObjType.TWO_HANDED_SWORD, ObjType.BATTLE_AXE,
                              ObjType.QUARTERSTAFF, ObjType.YUMI, ObjType.BOW,
                              ObjType.SLING, ObjType.PICK_AXE)
    # wave17h P0: starting inventory is known to the player (vendor u_init.c
    # marks starting items with otmp->known = TRUE).
    # vendor/nethack/src/u_init.c:1218 also sets dknown=bknown=rknown=1 so the
    # BUC prefix ("uncursed"/"blessed") renders for the starting kit.
    return make_item(
        category=ItemCategory.WEAPON,
        type_id=type_id,
        quantity=qty,
        weight=_WEAPON_WEIGHT.get(type_id, 20) * qty,
        is_two_handed=two_handed,
        enchantment=enchant,
        buc_status=buc,
        identified=True,
        bknown=True, dknown=True, rknown=True,
    )


def _armor(type_id: int, enchant: int = 0, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(
        category=ItemCategory.ARMOR,
        type_id=type_id,
        quantity=1,
        weight=_ARMOR_WEIGHT.get(type_id, 50),
        ac_bonus=_ARMOR_AC.get(type_id, 0),
        enchantment=enchant,
        buc_status=buc,
        identified=True,
        bknown=True, dknown=True, rknown=True,
    )


def _potion(type_id: int, qty: int = 1, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.POTION, type_id=type_id, quantity=qty,
                     weight=20 * qty, buc_status=buc, identified=True,
                     bknown=True, dknown=True, rknown=True)


def _scroll(type_id: int, qty: int = 1, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.SCROLL, type_id=type_id, quantity=qty,
                     weight=5 * qty, buc_status=buc, identified=True,
                     bknown=True, dknown=True, rknown=True)


def _spellbook(type_id: int, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.SPBOOK, type_id=type_id, quantity=1,
                     weight=50, buc_status=buc, identified=True,
                     bknown=True, dknown=True, rknown=True)


def _tool(type_id: int, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.TOOL, type_id=type_id, quantity=1,
                     weight=30, buc_status=buc, identified=True,
                     bknown=True, dknown=True, rknown=True)


def _food(type_id: int, qty: int = 1, buc: int = _BUC_UNCURSED) -> "Item":
    """Build a FOOD-class starting-inventory item.

    Cite: vendor/nethack/src/u_init.c starting trobj arrays use FOOD with
    ``oc_weight`` looked up from objects.h.  We pass a per-type weight via
    the lookup table below; ``identified=True`` mirrors vendor's
    ``otmp->known = TRUE`` on starting items (u_init.c::ini_inv).
    """
    from Nethax.nethax.subsystems.inventory import make_item
    # objects.h oc_weight per food otyp (Audit L #14-#26).  Lightweight
    # items default to 2 if missing — the corpse / egg cases (otyp 250-251)
    # aren't in this helper's scope.
    _FOOD_WEIGHT = {
        ObjType.APPLE: 2,
        ObjType.ORANGE: 2,
        ObjType.CARROT: 2,
        ObjType.SPRIG_OF_WOLFSBANE: 1,
        ObjType.CLOVE_OF_GARLIC: 1,
        ObjType.FORTUNE_COOKIE: 1,
        ObjType.LEMBAS_WAFER: 5,
        ObjType.CRAM_RATION: 15,
        ObjType.FOOD_RATION: 20,
    }
    return make_item(
        category=ItemCategory.FOOD,
        type_id=type_id,
        quantity=qty,
        weight=_FOOD_WEIGHT.get(type_id, 2) * qty,
        buc_status=buc,
        identified=True,
        bknown=True, dknown=True, rknown=True,
    )


def _gem(type_id: int, qty: int = 1, buc: int = _BUC_UNCURSED,
         weight_each: int = 10) -> "Item":
    """Build a GEM-class starting item (Caveman FLINT / ROCK ammo).

    Vendor ini_inv marks starting gems dknown/bknown/rknown = 1 so the
    resolved name ("flint stones" / "rocks") and BUC prefix render.  The
    stack ``quantity`` here is a placeholder; in NLE_BYTEPARITY mode the
    Caveman ini_inv draw replay (``_consume_ini_inv_caveman_draws``)
    overwrites it with the vendor-rolled gem-loop quantity.
    Cite: vendor/nle/src/mkobj.c:886-895 (GEM_CLASS); u_init.c:1102-1108.
    """
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.GEM, type_id=type_id, quantity=qty,
                     weight=weight_each * qty, buc_status=buc, identified=True,
                     bknown=True, dknown=True, rknown=True)


def _wand(type_id: int, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.WAND, type_id=type_id, quantity=1,
                     weight=7, buc_status=buc, identified=True,
                     bknown=True, dknown=True, rknown=True)


# ---------------------------------------------------------------------------
# STARTING_INVENTORY
# Each entry is a list of Item objects (pre-built, not names).
# Canonical: vendor/nethack/src/u_init.c trobj arrays.
# ---------------------------------------------------------------------------

STARTING_INVENTORY: dict = {
    # Arc: bullwhip+2, leather jacket+0, fedora+0, pick-axe, tinning kit,
    #      sack, touchstone, 3 food rations.
    # u_init.c Archeologist[] (42-53): BULLWHIP spe=2, LEATHER_JACKET 0,
    # FEDORA 0, PICK_AXE, TINNING_KIT, SACK, TOUCHSTONE, FOOD_RATION qty=3.
    # Audit L #14: added the missing SACK, TOUCHSTONE, FOOD_RATION×3.
    Role.ARCHEOLOGIST: [
        _weapon(ObjType.BULLWHIP, enchant=2),
        _armor(ObjType.LEATHER_JACKET),
        _armor(ObjType.FEDORA),
        _food(ObjType.FOOD_RATION, 4),
        _tool(ObjType.PICK_AXE),
        _tool(ObjType.TINNING_KIT),
        _tool(ObjType.TOUCHSTONE),
        _tool(ObjType.SACK),
    ],
    # Bar variant 0: two-handed sword+0, axe+0, ring mail+0, food ration.
    # u_init.c Barbarian_0 (54-67): TWO_HANDED_SWORD, AXE, RING_MAIL,
    # FOOD_RATION qty=1.  (Barbarian_1 variant TWO_HANDED_SWORD+AXE swapped
    # for BATTLE_AXE+SHORT_SWORD is selected by rn2(2) — deferred since the
    # current single-variant model is byte-equal vendor for variant 0.)
    # Audit L #15: added the missing FOOD_RATION×1.
    Role.BARBARIAN: [
        _weapon(ObjType.TWO_HANDED_SWORD),
        _weapon(ObjType.AXE),
        _armor(ObjType.RING_MAIL),
        _food(ObjType.FOOD_RATION, 1),
    ],
    # Cav: club+1, sling+2, flint stones, rocks, leather armor+0.
    # u_init.c Cave_man[] (50-56): CLUB spe=1, SLING spe=2, FLINT qty=15
    # (variable: Cave_man[C_AMMO].trquan = rn1(11,10) = 10..20), ROCK qty=3
    # (each ROCK mksobj yields rn1(6,6) so the merged stack is 18..33),
    # LEATHER_ARMOR spe=0.  FLINT/ROCK are GEM_CLASS so ini_inv loops
    # ``trquan`` mksobj calls and merges (u_init.c:1156 --trquan continue);
    # their stack quantities come from the mksobj gem draws, replayed in
    # NLE_BYTEPARITY by _consume_ini_inv_caveman_draws (quantities below are
    # placeholders).  FLINT is auto-quivered, SLING is the swap weapon.
    Role.CAVEMAN: [
        _weapon(ObjType.CLUB, enchant=1),
        _weapon(ObjType.SLING, enchant=2),
        _gem(ObjType.FLINT, 15),
        _gem(ObjType.ROCK, 18),
        _armor(ObjType.LEATHER_ARMOR),
    ],
    # Hea: scalpel+0, leather gloves+1, stethoscope, 4 healing + 4 extra healing,
    #      wand of sleep, spellbooks (3 of them BLESSED per u_init.c), 5 apples.
    # u_init.c Healer[] (76-89): SCALPEL 0, LEATHER_GLOVES 1, spellbooks
    # trbless=1, APPLE qty=5.
    # Audit L #17: added the missing APPLE×5.
    Role.HEALER: [
        _weapon(ObjType.SCALPEL),
        _armor(ObjType.LEATHER_GLOVES, enchant=1),
        _tool(ObjType.STETHOSCOPE),
        _potion(ObjType.POT_HEALING, 4),
        _potion(ObjType.POT_EXTRA_HEALING, 4),
        _wand(ObjType.WAN_SLEEP),
        _spellbook(ObjType.SPE_HEALING, buc=_BUC_BLESSED),
        _spellbook(ObjType.SPE_EXTRA_HEALING, buc=_BUC_BLESSED),
        _spellbook(ObjType.SPE_STONE_TO_FLESH, buc=_BUC_BLESSED),
        _food(ObjType.APPLE, 5),
    ],
    # Kni: long sword+1, lance+1, ring mail+1, helmet+0, small shield+0,
    #      leather gloves+0, 10 apples, 10 carrots (u_init.c Knight[]).
    # u_init.c Knight[] (90-100): trbless=UNDEF_BLESS for all → UNCURSED.
    # Audit L #18: added the missing APPLE×10 + CARROT×10.
    Role.KNIGHT: [
        _weapon(ObjType.LONG_SWORD, enchant=1),
        _weapon(ObjType.LANCE, enchant=1),
        _armor(ObjType.RING_MAIL, enchant=1),
        _armor(ObjType.HELMET),
        _armor(ObjType.SMALL_SHIELD),
        _armor(ObjType.LEATHER_GLOVES),
        _food(ObjType.APPLE, 10),
        _food(ObjType.CARROT, 10),
    ],
    # Mon: leather gloves+2, robe+1 (fights barehanded), 1 spellbook (M_BOOK),
    #      1 random scroll, 3 healing potions, 3 food rations, 5 apples,
    #      5 oranges, 3 fortune cookies.  u_init.c Monk[] (84-96) + case
    #      PM_MONK (713-721).  Slots 2/3 hold placeholders whose type/BUC are
    #      overwritten by the ini_inv mksobj-draw replay
    #      (_consume_ini_inv_monk_draws): slot 2 = M_spell[rn2(90)/30]
    #      spellbook (forced blessed, trbless=1), slot 3 = mkobj(SCROLL_CLASS)
    #      random scroll (reject loop).  Food/potion stack quantities also come
    #      from the mksobj draws.  A bonus MAGIC_MARKER / OIL_LAMP may be
    #      appended at slot 9 (u_init.c:718-721 rn2(5)/rn2(10) cascade).
    Role.MONK: [
        _armor(ObjType.LEATHER_GLOVES, enchant=2),
        _armor(ObjType.ROBE, enchant=1),
        _spellbook(ObjType.SPE_HEALING, buc=_BUC_BLESSED),  # type overridden
        _scroll(ObjType.SCR_MAGIC_MAPPING, 1),              # type overridden
        _potion(ObjType.POT_HEALING, 3),
        _food(ObjType.FOOD_RATION, 3),
        _food(ObjType.APPLE, 5),
        _food(ObjType.ORANGE, 5),
        _food(ObjType.FORTUNE_COOKIE, 3),
    ],
    # Pri: mace+1 (BLESSED — vendor trbless=1), robe+0, small shield+0,
    #      4 holy water (BLESSED), clove of garlic, sprig of wolfsbane.
    # u_init.c Priest[] (114-123): MACE spe=1 trbless=1, POT_WATER trbless=1,
    # CLOVE_OF_GARLIC, SPRIG_OF_WOLFSBANE.  Two random spellbooks also in
    # vendor — deferred (random-spellbook init not wired).
    # Audit L #20: added missing garlic + wolfsbane.
    Role.PRIEST: [
        _weapon(ObjType.MACE, enchant=1, buc=_BUC_BLESSED),
        _armor(ObjType.ROBE),
        _armor(ObjType.SMALL_SHIELD),
        _potion(ObjType.POT_WATER, 4, buc=_BUC_BLESSED),   # holy water
        _food(ObjType.CLOVE_OF_GARLIC, 1),
        _food(ObjType.SPRIG_OF_WOLFSBANE, 1),
    ],
    # Ran: dagger+1, bow+1, arrow+2 x50, arrow+0 x30..39, cloak+2, 4 cram rations.
    # u_init.c Ranger[] (124-132):  DAGGER spe=1, BOW spe=1, ARROW spe=2 qty=50..59,
    # ARROW spe=0 qty=30..39, CLOAK_OF_DISPLACEMENT spe=2, CRAM_RATION qty=4.
    # Audit L #21: added missing CRAM_RATION×4 and the second arrow stack
    # (spe=0).  Quantity uses the median of 30..39 (= 35) since per-init random
    # range isn't wired through this static table.
    Role.RANGER: [
        _weapon(ObjType.DAGGER, enchant=1),
        _weapon(ObjType.BOW, enchant=1),
        _weapon(ObjType.ARROW, 50, enchant=2),
        _weapon(ObjType.ARROW, 35, enchant=0),
        _armor(ObjType.CLOAK_OF_DISPLACEMENT, enchant=2),
        _food(ObjType.CRAM_RATION, 4),
    ],
    # Rog: short sword+0, daggers+0 x10, leather armor+1, pot of sickness,
    #      lock pick, sack, blindfold (u_init.c Rogue[] 122-131 + line 754
    #      Blindfold extra roll).
    # Vendor order (ini_inv) is: SHORT_SWORD, DAGGER, LEATHER_ARMOR,
    #   POT_SICKNESS, LOCK_PICK, SACK — then ``if (!rn2(5)) ini_inv(Blindfold)``
    #   appends BLINDFOLD as a 7th item.  NLE seed 0 rolls the lucky branch,
    #   so for byte-exact parity we include the BLINDFOLD slot.  Vendor DAGGER
    #   qty is ``rn1(10, 6)`` (= 6..15 random); the static 10 here is a
    #   placeholder — in NLE_BYTEPARITY mode create_character() overrides it
    #   with the rolled rn1(10, 6) value (14 for seed 0) using the already-
    #   consumed rn2(10) draw, so no extra ISAAC64 draw is made.  BODY-armor
    #   slot index (2) is preserved so ``_WORN_ARMOR_BY_ROLE[ROGUE]`` still
    #   maps to LEATHER_ARMOR.
    Role.ROGUE: [
        _weapon(ObjType.SHORT_SWORD),
        _weapon(ObjType.DAGGER, 10),
        _armor(ObjType.LEATHER_ARMOR, enchant=1),
        _potion(ObjType.POT_SICKNESS, 1),
        _tool(ObjType.LOCK_PICK),
        _tool(ObjType.SACK),
        _tool(ObjType.BLINDFOLD),
    ],
    # Sam: katana+0, wakizashi(short sword)+0, yumi+0, ya+0 x26, splint mail+0
    # u_init.c Samurai[]: all spe=0, all trbless=UNDEF_BLESS → UNCURSED.
    Role.SAMURAI: [
        _weapon(ObjType.KATANA),
        _weapon(ObjType.SHORT_SWORD),   # wakizashi
        _weapon(ObjType.YUMI),
        _weapon(ObjType.YA, 26),
        _armor(ObjType.SPLINT_MAIL),
    ],
    # Tou: darts+2 x21, 2 extra healing potions, 4 scrolls of magic mapping,
    #      Hawaiian shirt+0, expensive camera, credit card
    # u_init.c Tourist[]:  DART spe=2.
    Role.TOURIST: [
        _weapon(ObjType.DART, 21, enchant=2),
        _potion(ObjType.POT_EXTRA_HEALING, 2),
        _scroll(ObjType.SCR_MAGIC_MAPPING, 4),
        _armor(ObjType.HAWAIIAN_SHIRT),
        _tool(ObjType.EXPENSIVE_CAMERA),
        _tool(ObjType.CREDIT_CARD),
    ],
    # Val: spear+1, dagger+0, small shield+3, 1 food ration
    # (u_init.c Valkyrie[] 160-166).  trbless=UNDEF_BLESS for all → UNCURSED.
    # Audit L #25: added the missing FOOD_RATION×1.
    Role.VALKYRIE: [
        # Vendor Valkyrie[0] is LONG_SWORD spe=1 (not spear); glyph 1943.
        _weapon(ObjType.LONG_SWORD, enchant=1),
        _weapon(ObjType.DAGGER),
        _armor(ObjType.SMALL_SHIELD, enchant=3),
        _food(ObjType.FOOD_RATION, 1),
    ],
    # Wiz: quarterstaff+1 (BLESSED — trbless=1), cloak of magic resistance+0,
    #      wand, force-bolt spellbook (BLESSED), random potions/scrolls,
    #      magic marker (spe = 18 + d4 ≈ 20).
    # u_init.c Wizard[] (167-178): QUARTERSTAFF spe=1 trbless=1, SPE_FORCE_BOLT
    # trbless=1, MAGIC_MARKER spe=18+d4.  Random RING ×2, POTION ×3, SCROLL ×3,
    # SPBOOK ×1, WAND each pulled from UNDEF_TYP — we keep the existing
    # POT_HEALING/SCR_IDENTIFY/WAN_SLEEP representatives until random-init is
    # wired (Audit L #26 remaining).
    Role.WIZARD: [
        _weapon(ObjType.QUARTERSTAFF, enchant=1, buc=_BUC_BLESSED),
        _armor(ObjType.CLOAK_OF_MAGIC_RESISTANCE),
        _wand(ObjType.WAN_SLEEP),          # representative random wand
        _spellbook(ObjType.SPE_FORCE_BOLT, buc=_BUC_BLESSED),
        _potion(ObjType.POT_HEALING, 3),   # representative random potions
        _scroll(ObjType.SCR_IDENTIFY, 3),  # representative random scrolls
        # MAGIC_MARKER spe = 18 + d4 (median = 20).  Vendor u_init.c:177.
        _tool(ObjType.MAGIC_MARKER),
    ],
}


# ---------------------------------------------------------------------------
# STARTING_STATS  { (Role, Race): {stat_name: (min, max)} }
#
# Wave 6 Phase B+ vendor-parity derivation.  In vendor semantics
# (vendor/nle/src/role.c::roles[] and races[], plus attrib.c::init_attr):
#
#   * ``role.attrbase[i]`` -- the *floor* of attribute ``i`` at character
#     init.  init_attr starts every stat at attrbase[i] and then
#     distributes ``np - sum(attrbase)`` extra points weighted by
#     ``role.attrdist`` until each stat hits its race cap.
#   * ``race.attrmax[i]`` -- the *hard cap* enforced by ATTRMAX().
#
# Therefore the effective starting range for stat ``i`` is
#   ``[role.attrbase[i], race.attrmax[i]]``,
# which we expose as the per (role, race) range here.  This replaces the
# Wave 3 ``_normalize_ranges`` hack which mis-interpreted ``attrdist`` as
# the upper bound and had to swap inverted pairs after the fact.
# ---------------------------------------------------------------------------

_STAT_NAMES = ["str", "int", "wis", "dex", "con", "cha"]


def _stat_range_for(role: Role, race: Race) -> dict:
    """Return ``{stat_name: (lo, hi)}`` for ``(role, race)`` from vendor tables.

    ``lo`` = ``role.attrbase[i]`` (vendor role.c).
    ``hi`` = max(``lo``, ``race.attrmax[i]``) -- the race cap.  If a role
    floor exceeds the race cap (rare; e.g. Knight CHA floor 17 vs race cap
    18 is fine) we clamp ``hi`` to at least ``lo`` so the range stays
    non-empty when rolling.
    """
    r_entry = get_role(role)
    race_entry = get_race(race)
    out = {}
    for i, name in enumerate(_STAT_NAMES):
        lo = int(r_entry.attrbase[i])
        cap = int(race_entry.attrmax[i])
        hi = max(lo, cap)
        out[name] = (lo, hi)
    return out


STARTING_STATS: dict = {
    (role, race): _stat_range_for(role, race)
    for role in Role
    for race in Race
}


# ---------------------------------------------------------------------------
# STARTING_HP_PW  { Role: (hp_init, hp_lower, hp_higher, pw_init, pw_level) }
# Canonical: vendor/nethack/src/role.c roles[] HP/energy fields.
# Format from role.c: { Init, Lower, Upper } for HP; energy = single int.
#   hp_base: Init HP at level 1
#   hp_per_level: Lower (added each level gain)
#   pw_base: starting energy
#   pw_per_level: energy gained per level (vendor role.c en_advance.lofactor)
# ---------------------------------------------------------------------------

STARTING_HP_PW: dict = {
    # Role:        (hp_base, hp_per_level, pw_base, pw_per_level)
    Role.ARCHEOLOGIST: (11, 8,  14, 1),
    Role.BARBARIAN:    (14, 10, 10, 1),
    Role.CAVEMAN:      (14, 8,  10, 1),
    Role.HEALER:       (11, 8,  20, 2),
    Role.KNIGHT:       (14, 8,  10, 2),
    Role.MONK:         (12, 8,  10, 2),
    Role.PRIEST:       (12, 8,  10, 2),
    Role.RANGER:       (13, 6,  12, 1),
    Role.ROGUE:        (10, 8,  11, 1),
    Role.SAMURAI:      (13, 8,  11, 1),
    Role.TOURIST:      ( 8, 8,  14, 1),
    Role.VALKYRIE:     (14, 8,  10, 1),
    Role.WIZARD:       (10, 8,  12, 3),
}


# JIT-side per-role HP/PW per-level lookups.  Used by polymorph::newman
# to consume the per-class ``newhp() = d(hp_per_level)`` and
# ``newpw() = d(pw_per_level)`` dice when adding levels post-poly.
_N_ROLES_HP_PW = max(int(r) for r in STARTING_HP_PW.keys()) + 1
_HP_PER_LEVEL_ARR: list = [8] * _N_ROLES_HP_PW
_PW_PER_LEVEL_ARR: list = [1] * _N_ROLES_HP_PW
for _r, (_hb, _hpl, _pb, _pwl) in STARTING_HP_PW.items():
    _HP_PER_LEVEL_ARR[int(_r)] = int(_hpl)
    _PW_PER_LEVEL_ARR[int(_r)] = int(_pwl)
HP_PER_LEVEL_TABLE = jnp.array(_HP_PER_LEVEL_ARR, dtype=jnp.int32)
PW_PER_LEVEL_TABLE = jnp.array(_PW_PER_LEVEL_ARR, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Wield / wear helpers
# ---------------------------------------------------------------------------

# Primary weapon slot index in STARTING_INVENTORY (always index 0)
_PRIMARY_WEAPON_IDX = 0

# Armor slot assignments by ArmorSlot for each role's starting gear.
# Maps ArmorSlot -> index in STARTING_INVENTORY[role] list (or None).
_WORN_ARMOR_BY_ROLE: dict = {
    Role.ARCHEOLOGIST: {ArmorSlot.BODY: 1, ArmorSlot.HELM: 2},
    Role.BARBARIAN:    {ArmorSlot.BODY: 2},
    Role.CAVEMAN:      {ArmorSlot.BODY: 4},
    Role.HEALER:       {ArmorSlot.GLOVES: 1},
    Role.KNIGHT:       {ArmorSlot.BODY: 2, ArmorSlot.HELM: 3,
                        ArmorSlot.SHIELD: 4, ArmorSlot.GLOVES: 5},
    Role.MONK:         {ArmorSlot.GLOVES: 0, ArmorSlot.BODY: 1},
    Role.PRIEST:       {ArmorSlot.BODY: 1, ArmorSlot.SHIELD: 2},
    Role.RANGER:       {ArmorSlot.CLOAK: 4},
    Role.ROGUE:        {ArmorSlot.BODY: 2},
    Role.SAMURAI:      {ArmorSlot.BODY: 4},
    Role.TOURIST:      {ArmorSlot.BODY: 3},
    Role.VALKYRIE:     {ArmorSlot.SHIELD: 2},
    Role.WIZARD:       {ArmorSlot.CLOAK: 1},
}

# Roles that start with no weapon (Monk fights barehanded)
_NO_WEAPON_ROLES = {Role.MONK}


# ---------------------------------------------------------------------------
# STARTING_SPELLS — per-role starting spell memory
# Canonical source: vendor/nethack/src/u_init.c (Wizard gets force bolt
#   spellbook pre-blessed; Priest/Healer/Monk get one of three protection /
#   healing / confuse-monster books).  We simplify to a single deterministic
#   starting spell per spell-casting role.
# ---------------------------------------------------------------------------

STARTING_SPELLS: dict = {
    Role.WIZARD: [SpellId.FORCE_BOLT],
    Role.PRIEST: [SpellId.PROTECTION],
    Role.HEALER: [SpellId.HEALING],
    Role.MONK:   [SpellId.PROTECTION],
}


# ---------------------------------------------------------------------------
# STARTING_PET — per-role starting pet (monster name + PM_* id)
# Canonical source: vendor/nethack/src/role.c roles[].petnum.
# Roles with petnum == NON_PM default to "kitten" (pet_type rn2(2) coin flip
# resolves to kitten when preferred_pet == 'c'; we pick a stable default).
# ---------------------------------------------------------------------------

STARTING_PET: dict = {
    Role.ARCHEOLOGIST: "kitten",
    Role.BARBARIAN:    "kitten",
    Role.CAVEMAN:      "little dog",
    Role.HEALER:       "kitten",
    Role.KNIGHT:       "pony",
    Role.MONK:         "kitten",
    Role.PRIEST:       "kitten",
    Role.RANGER:       "little dog",
    Role.ROGUE:        "kitten",
    Role.SAMURAI:      "little dog",
    Role.TOURIST:      "kitten",
    Role.VALKYRIE:     "kitten",
    Role.WIZARD:       "kitten",
}


def get_starting_pet(role: Role) -> str:
    """Return the canonical starting pet monster name for ``role``."""
    return STARTING_PET[role]


def get_starting_spells(role: Role) -> list:
    """Return the list of SpellId values memorised at character creation."""
    return list(STARTING_SPELLS.get(role, []))


# ---------------------------------------------------------------------------
# Vendor-parity helpers — init_attr / ini_hpwp formulae
# ---------------------------------------------------------------------------

def _init_attr_vendor(
    rng: jax.Array,
    role: Role,
    race: Race,
    np_total: int = 75,
) -> dict:
    """Reproduce vendor ``attrib.c::init_attr(np)`` for one (role, race).

    Vendor formula (vendor/nle/src/attrib.c lines 614-660 and
    vendor/nethack/src/u_init.c:1390 ``init_attr(75)``):

      1. ABASE(i) = role.attrbase[i] for i in 0..A_MAX-1;
         np -= sum(role.attrbase).
      2. While np > 0 (and tryct < 100), pick x = rn2(100); iterate stats by
         subtracting role.attrdist[i] from x until x <= 0 → stat index i.
         If ABASE(i) >= race.attrmax[i], increment tryct and continue.
         Else ABASE(i)++, np--.
      3. Final value clamped to [race.attrmin[i], race.attrmax[i]].

    Returns a dict mapping ``_STAT_NAMES`` to int32 JAX scalars.

    Pure-JAX implementation: uses ``lax.while_loop`` and ``jnp.where`` so the
    function is ``jax.vmap``-safe.  ``role``/``race`` are static Python
    hyperparameters (resolved at trace time); the per-iteration ``rn2(100)``
    draws come from a pre-split key bundle so the loop body has no host-side
    Python control flow on traced values.
    """
    r_entry = get_role(role)
    race_entry = get_race(race)

    # Static (Python-int) role/race tables — these resolve at trace time.
    attrbase_arr = jnp.asarray(list(r_entry.attrbase), dtype=jnp.int32)   # [6]
    attrdist_arr = jnp.asarray(list(r_entry.attrdist), dtype=jnp.int32)   # [6]
    attrmin_arr  = jnp.asarray(list(race_entry.attrmin), dtype=jnp.int32) # [6]
    attrmax_arr  = jnp.asarray(list(race_entry.attrmax), dtype=jnp.int32) # [6]

    max_iters = 1000
    base_sum = int(sum(int(v) for v in r_entry.attrbase))
    initial_remaining = jnp.int32(int(np_total) - base_sum)

    # Pre-draw ``max_iters`` ``rn2(100)`` values via a single batched randint.
    # Vendor draws one rn2(100) per iteration; pre-splitting consumes the same
    # number of keys.  We split rng once then call randint with shape=(N,) so
    # the result is a static-shape int32 array.
    xs = jax.random.randint(
        rng, shape=(max_iters,), minval=0, maxval=100, dtype=jnp.int32,
    )  # [max_iters]

    # Loop carry: (values[6], remaining, tryct, idx)
    init_carry = (attrbase_arr, initial_remaining, jnp.int32(0), jnp.int32(0))

    def cond_fn(carry):
        values, remaining, tryct, idx = carry
        return (
            (remaining > jnp.int32(0))
            & (tryct < jnp.int32(100))
            & (idx < jnp.int32(max_iters))
        )

    def body_fn(carry):
        values, remaining, tryct, idx = carry
        x = xs[idx]

        # Walk attrdist to find target stat index i; mirror vendor exactly:
        #   attrib.c:628 ``for (i=0; (i<A_MAX) && ((x -= attrdist[i]) > 0); i++)``
        # The running subtraction stops at the first i where the result is
        # <= 0, i.e. cumsum[i] >= original x.  ``cum >= x`` (NOT ``cum > x``):
        # the strict form over-advances when x lands exactly on a cumulative
        # boundary, selecting i+1 instead of i.
        # i ends in [0, 6]; i == 6 means "impossible" (skip this draw).
        cum = jnp.cumsum(attrdist_arr)              # [6]
        # found[k] is True iff cum[k] >= x.  argmax over found returns first True.
        found = cum >= x
        any_found = jnp.any(found)
        i_idx = jnp.where(
            any_found,
            jnp.argmax(found.astype(jnp.int32)).astype(jnp.int32),
            jnp.int32(6),
        )

        impossible = i_idx >= jnp.int32(6)
        # Safe index for read/write (clamped to [0, 5]); the impossible branch
        # is masked out so the writeback is a no-op below.
        i_safe = jnp.clip(i_idx, jnp.int32(0), jnp.int32(5))

        cur_val = values[i_safe]
        cap = attrmax_arr[i_safe]
        at_cap = cur_val >= cap

        # Increment: only if not impossible and not at cap.
        do_increment = (~impossible) & (~at_cap)
        new_values = jnp.where(
            do_increment,
            values.at[i_safe].add(jnp.int32(1)),
            values,
        )
        new_remaining = jnp.where(
            do_increment, remaining - jnp.int32(1), remaining,
        )

        # tryct: bumped only when at_cap (and not impossible); reset on success.
        tryct_at_cap = (~impossible) & at_cap
        new_tryct = jnp.where(
            do_increment,
            jnp.int32(0),
            jnp.where(tryct_at_cap, tryct + jnp.int32(1), tryct),
        )

        return (new_values, new_remaining, new_tryct, idx + jnp.int32(1))

    final_values, _rem, _tryct, _idx = jax.lax.while_loop(
        cond_fn, body_fn, init_carry,
    )

    # Final clamp to race floor and ceiling.
    clamped = jnp.clip(final_values, attrmin_arr, attrmax_arr)

    return {
        name: clamped[i].astype(jnp.int32)
        for i, name in enumerate(_STAT_NAMES)
    }


def _ini_hpwp_vendor(
    rng: jax.Array,
    role: Role,
    race: Race,
) -> tuple:
    """Reproduce vendor ``u_init.c::ini_hpwp`` / ``newhp`` / ``newpw`` for
    character init (u.ulevel == 0 branch).

    Formula (vendor/nle/src/attrib.c::newhp lines 985-996, exper.c::newpw
    lines 47-72):

      hp = role.hpadv.infix + race.hpadv.infix
           + rnd(role.hpadv.inrnd) [if > 0]
           + rnd(race.hpadv.inrnd) [if > 0]
      pw = role.enadv.infix + race.enadv.infix
           + rnd(role.enadv.inrnd) [if > 0]
           + rnd(race.enadv.inrnd) [if > 0]
      hp = max(hp, 1); pw = max(pw, 1)

    Note: vendor ``rnd(n)`` returns 1..n inclusive.
    """
    r_entry = get_role(role)
    race_entry = get_race(race)

    rng_hp1, rng_hp2, rng_pw1, rng_pw2 = jax.random.split(rng, 4)

    def _rnd(key, n: int) -> jnp.ndarray:
        # rnd(n) in vendor returns uniform int in [1, n].
        n = int(n)
        if n <= 0:
            return jnp.int32(0)
        return jax.random.randint(key, shape=(), minval=1, maxval=n + 1).astype(jnp.int32)

    hp = jnp.int32(r_entry.hpadv.infix) + jnp.int32(race_entry.hpadv.infix)
    hp = hp + _rnd(rng_hp1, r_entry.hpadv.inrnd)
    hp = hp + _rnd(rng_hp2, race_entry.hpadv.inrnd)
    hp = jnp.maximum(hp, jnp.int32(1))

    pw = jnp.int32(r_entry.enadv.infix) + jnp.int32(race_entry.enadv.infix)
    pw = pw + _rnd(rng_pw1, r_entry.enadv.inrnd)
    pw = pw + _rnd(rng_pw2, race_entry.enadv.inrnd)
    pw = jnp.maximum(pw, jnp.int32(1))

    return hp, pw


# ---------------------------------------------------------------------------
# consume_init_attr_draws  — vendor ISAAC64 init_attr replay
# ---------------------------------------------------------------------------

def consume_init_attr_draws(
    vendor_rng,
    role: Role,
    race: Race,
    np_total: int = 75,
):
    """Replay vendor ``attrib.c::init_attr(np)`` consuming ISAAC64 draws.

    Mirrors vendor/nle/src/attrib.c:614-660 line-for-line using the
    ISAAC64 host-side ``rn2`` helper.  Vendor algorithm:

      1. ABASE(i) = role.attrbase[i]; np -= sum(attrbase).        (lines 619-623)
      2. while np > 0 and tryct < 100:                            (line 626)
           x = rn2(100)                                           (line 627)
           walk attrdist to find stat index i                     (lines 628-629)
           if i >= A_MAX: continue  (impossible, attrdist sums to (line 630-631)
                                     100 so never reached)
           if ABASE(i) >= ATTRMAX(i): tryct++; continue          (lines 633-636)
           tryct = 0; ABASE(i)++; AMAX(i)++; np--               (lines 637-640)
      3. Redistribution loop (np < 0) mirrors the same pattern.  (lines 644-660)

    Each iteration that advances (does not hit cap or impossible) consumes
    exactly one ``rn2(100)`` draw.  Capped/impossible iterations also
    consume one draw (vendor does not re-draw on those paths).

    Parameters
    ----------
    vendor_rng : Isaac64State
        Live ISAAC64 state to consume from.
    role, race : Role, Race
        Character role and race.
    np_total : int
        Point budget passed to init_attr (vendor always calls ``init_attr(75)``
        via u_init.c:1390).

    Returns
    -------
    (vendor_rng, attributes) where ``vendor_rng`` is the post-draw
    Isaac64State and ``attributes`` is a ``{stat_name: int}`` dict
    mapping _STAT_NAMES → Python int values.

    Cite: vendor/nle/src/attrib.c:614-660
          vendor/nle/src/u_init.c:1390 (``init_attr(75)`` call site)
    """
    from Nethax.nethax.vendor_rng import rn2_jax

    r_entry = get_role(role)
    race_entry = get_race(race)

    # Static role/race tables — captured by closure as Python lists, then
    # baked into jnp arrays.  attrdist sums to 100, so the inner-walk
    # "i >= A_MAX" branch is unreachable; we still match vendor structure.
    attrbase_py = [int(v) for v in r_entry.attrbase]
    attrdist_py = [int(v) for v in r_entry.attrdist]
    attrmin_py  = [int(v) for v in race_entry.attrmin]
    attrmax_py  = [int(v) for v in race_entry.attrmax]

    attrdist_arr = jnp.asarray(attrdist_py, dtype=jnp.int32)
    attrmin_arr  = jnp.asarray(attrmin_py,  dtype=jnp.int32)
    attrmax_arr  = jnp.asarray(attrmax_py,  dtype=jnp.int32)

    # Inclusive prefix-sum of attrdist (jnp).  argmax(cumsum >= x) gives the
    # same ``i`` as the vendor for-loop because vendor stops at the first i
    # where (running sum from 0..i inclusive) >= original_x.
    cumsum_arr = jnp.cumsum(attrdist_arr)  # int32[6]

    # Step 1: initialise ABASE from role, compute remaining points.
    # attrib.c:619-623
    abase0 = jnp.asarray(attrbase_py, dtype=jnp.int32)
    np0    = jnp.int32(np_total - sum(attrbase_py))

    def _walk_dist(x: jnp.ndarray) -> jnp.ndarray:
        """Vendor for-loop walk — returns i in [0,5] where cumsum >= x."""
        return jnp.argmax(cumsum_arr >= x).astype(jnp.int32)

    # Step 2: distribute excess points upward.  attrib.c:625-640.
    # State: (rng, abase, np, tryct).  Loop while np>0 and tryct<100.
    def _up_cond(carry):
        _rng, _abase, np_c, tryct_c = carry
        return jnp.logical_and(np_c > jnp.int32(0), tryct_c < jnp.int32(100))

    def _up_body(carry):
        rng_c, abase_c, np_c, tryct_c = carry
        rng_c, x = rn2_jax(rng_c, jnp.int32(100))           # attrib.c:627
        i = _walk_dist(x)
        cur = abase_c[i]
        cap = attrmax_arr[i]
        capped = cur >= cap                                  # attrib.c:633-636
        new_abase = abase_c.at[i].set(cur + jnp.int32(1))    # attrib.c:638
        # capped: tryct++ no change to abase/np.  uncapped: tryct=0, abase++, np--.
        # Brax-flatten: always compute both branches, jnp.where to select.
        # No RNG consumed inside the branches, so parity is preserved.
        abase_n = jnp.where(capped, abase_c, new_abase)
        np_n    = jnp.where(capped, np_c,    np_c - jnp.int32(1))
        tryct_n = jnp.where(capped, tryct_c + jnp.int32(1), jnp.int32(0))
        return rng_c, abase_n, np_n, tryct_n

    vendor_rng, abase1, np1, _ = jax.lax.while_loop(
        _up_cond, _up_body,
        (vendor_rng, abase0, np0, jnp.int32(0)),
    )

    # Step 3: redistribute excess points downward (np < 0).  attrib.c:643-660.
    def _dn_cond(carry):
        _rng, _abase, np_c, tryct_c = carry
        return jnp.logical_and(np_c < jnp.int32(0), tryct_c < jnp.int32(100))

    def _dn_body(carry):
        rng_c, abase_c, np_c, tryct_c = carry
        rng_c, x = rn2_jax(rng_c, jnp.int32(100))           # attrib.c:646
        i = _walk_dist(x)
        cur = abase_c[i]
        floor = attrmin_arr[i]
        capped = cur <= floor                                # attrib.c:652-655
        new_abase = abase_c.at[i].set(cur - jnp.int32(1))    # attrib.c:657
        # Brax-flatten: always compute both branches, jnp.where to select.
        # No RNG consumed inside the branches, so parity is preserved.
        abase_n = jnp.where(capped, abase_c, new_abase)
        np_n    = jnp.where(capped, np_c,    np_c + jnp.int32(1))
        tryct_n = jnp.where(capped, tryct_c + jnp.int32(1), jnp.int32(0))
        return rng_c, abase_n, np_n, tryct_n

    vendor_rng, abase2, _np2, _ = jax.lax.while_loop(
        _dn_cond, _dn_body,
        (vendor_rng, abase1, np1, jnp.int32(0)),
    )

    # Clamp final values to [attrmin, attrmax].
    clamped = jnp.minimum(jnp.maximum(abase2, attrmin_arr), attrmax_arr)

    attributes = {name: clamped[i] for i, name in enumerate(_STAT_NAMES)}
    return vendor_rng, attributes


# ---------------------------------------------------------------------------
# _consume_ini_inv_rogue_draws  — vendor ISAAC64 ini_inv blessorcurse replay
# ---------------------------------------------------------------------------

def _consume_ini_inv_rogue_draws(vendor_rng):
    """Consume ISAAC64 CORE draws emitted by ``ini_inv(Rogue)`` during
    ``mksobj`` initialisation for the 6 fixed Rogue starting items.

    Vendor order (u_init.c:749-752 → mksobj per item):

      1. ``rn2(10)`` — ``rn1(10, 6)`` dagger quantity          u_init.c:750
      2. SHORT_SWORD (WEAPON_CLASS)  — mkobj.c:804-817
      3. DAGGER       (WEAPON_CLASS) — mkobj.c:804-817
      4. LEATHER_ARMOR (ARMOR_CLASS) — mkobj.c:992-1004
      5. POT_SICKNESS (POTION_CLASS) — mkobj.c:981-987 → blessorcurse(4)
      6. LOCK_PICK    (TOOL_CLASS)   — no switch case → 0 draws
      7. SACK         (TOOL_CLASS)   — mkbox_cnts: moves<=1 → rn2(1)   mkobj.c:309

    WEAPON_CLASS draw pattern (mkobj.c:804-818)::

        rn2(11)           # always
        if !rn2(11):
            rne(3)        # spe — mkobj.c:806
            rn2(2)        # blessed flag — mkobj.c:807
        elif !rn2(10):
            rne(3)        # spe (curse branch) — mkobj.c:810
        else:
            blessorcurse(10) → rn2(10) [+ rn2(2) if hit]

    ARMOR_CLASS draw pattern (mkobj.c:992-1004)::

        rn2(10)           # outer: fumble/levitation/etc. guard
        if outer hit:
            rn2(11)       # inner: !rn2(11) curse check
            if !rn2(11):
                rne(3)    # spe (curse branch) — mkobj.c:999
            elif !rn2(10):
                rn2(2)    # blessed flag — mkobj.c:1001
                rne(3)    # spe — mkobj.c:1002
            else:
                blessorcurse(10) → rn2(10) [+ rn2(2) if hit]
        elif !rn2(10):
            rn2(2)        # blessed flag — mkobj.c:1001
            rne(3)        # spe — mkobj.c:1002
        else:
            blessorcurse(10) → rn2(10) [+ rn2(2) if hit]

    ``rne(3)`` at u.ulevel==0 draws 1–4 times from rn2(3) (``while tmp < 5
    && !rn2(3)``), consuming 1–4 ISAAC64 words per call.  For Rogue there is
    exactly **1 rne(3) call** (LEATHER_ARMOR trspe=1); WEAPON_CLASS items have
    trspe=0 but the weapon-class branch also calls rne when rn2(11)==0 or
    rn2(10)==0 — those draws are now included.

    Returns ``(post-draw Isaac64State, dagger_qty, buc_flags)`` where
    ``buc_flags`` is int32[4] of vendor's final BUC for SHORT_SWORD, DAGGER,
    LEATHER_ARMOR, POT_SICKNESS (in trobj order).  Values are ``_BUC_UNCURSED``
    or ``_BUC_BLESSED`` (cursed is wiped by ini_inv u_init.c:113 → uncursed).
    No extra draw — we surface outcomes already consumed.
    Cite: vendor/nle/src/u_init.c:750, 113; rnd.c::rn1.

    Citations
    ---------
    vendor/nle/src/u_init.c:749-756     — PM_ROGUE role-switch block
    vendor/nle/src/mkobj.c:803-818      — WEAPON_CLASS init
    vendor/nle/src/mkobj.c:992-1004     — ARMOR_CLASS init
    vendor/nle/src/mkobj.c:981-987      — POTION_CLASS / blessorcurse(4)
    vendor/nle/src/mkobj.c:309          — mkbox_cnts rn2(n+1)
    vendor/nle/src/mkobj.c:1370-1385    — blessorcurse definition
    vendor/nle/src/rnd.c:196-215        — rne implementation
    vendor/nle/src/rnd.c:194            — rne range comment
    """
    from Nethax.nethax.vendor_rng import rn2_jax, rne_jax

    # Local blessorcurse(chance) helper — vendor mkobj.c:1370-1385.
    # rn2(chance); if hit==0: rn2(2).  All draws emitted via JAX-traceable rn2.
    # Surfaces ``(rng, blessed)`` where blessed=1 iff the inner rn2(2)==1.
    # ini_inv (u_init.c:113) wipes ``cursed`` so only blessed bit matters
    # for inv_strs rendering.
    def _boc(rng_c, chance):
        rng_c, hit = rn2_jax(rng_c, jnp.int32(chance))
        # Brax-flatten: compute both branches; both branches' RNG advances are
        # computed in HLO, and jnp.where selects the correct one per-element.
        # Unselected branch's draw is discarded — byte-parity preserved.
        rng_hit, inner_hit = rn2_jax(rng_c, jnp.int32(2))
        rng_miss = rng_c
        inner_miss = jnp.int32(0)
        pred = hit == jnp.int32(0)
        rng_out = jax.tree_util.tree_map(
            lambda t, f: jnp.where(pred, t, f), rng_hit, rng_miss,
        )
        inner_out = jnp.where(pred, inner_hit, inner_miss)
        return rng_out, inner_out

    # 1. rn1(10, 6) dagger quantity — u_init.c:750
    # rn1(x, y) == rn2(x) + y, so trquan = rn2(10) + 6 ∈ [6, 15].
    # Cite: vendor/nle/src/rnd.c::rn1.
    vendor_rng, _dagger_roll = rn2_jax(vendor_rng, jnp.int32(10))
    dagger_qty = _dagger_roll + jnp.int32(6)

    # 2–3. SHORT_SWORD and DAGGER — WEAPON_CLASS (mkobj.c:803-818).
    # Neither is is_multigen (oc_skill >= 0 → fails the negative-skill test),
    # nor is_poisonable (same predicate), so no rn1(6,6)/rn2(100) draws fire.
    # The if/elif/else cascade is the only RNG.  Each step returns the
    # ``blessed`` flag for the resulting Item (0=uncursed-or-cursed-wiped,
    # 1=blessed).  Cite: vendor/nle/include/obj.h:197-205 (is_multigen,
    # is_poisonable); vendor/nle/src/mkobj.c:803-818 (WEAPON_CLASS init).
    def _weapon_blessed(r):                                      # mkobj.c:806-807
        r, _ = rne_jax(r, jnp.int32(3))                          # spe = rne(3)
        r, b = rn2_jax(r, jnp.int32(2))                          # blessed = rn2(2)
        return r, b

    def _weapon_cursed(r):                                       # mkobj.c:809-810
        r, _ = rne_jax(r, jnp.int32(3))                          # spe = -rne(3)
        return r, jnp.int32(0)                                    # cursed → wiped → 0

    def _weapon_else(r):
        r, r10 = rn2_jax(r, jnp.int32(10))
        # Brax-flatten: compute both branches, jnp.where to select.
        rng_cursed, b_cursed = _weapon_cursed(r)
        rng_boc, b_boc = _boc(r, 10)                              # blessorcurse(10)
        pred = r10 == jnp.int32(0)
        rng_out = jax.tree_util.tree_map(
            lambda t, f: jnp.where(pred, t, f), rng_cursed, rng_boc,
        )
        b_out = jnp.where(pred, b_cursed, b_boc)
        return rng_out, b_out

    def _weapon_step(r):
        r, r11 = rn2_jax(r, jnp.int32(11))
        # Brax-flatten: compute both branches, jnp.where to select.
        rng_blessed, b_blessed = _weapon_blessed(r)
        rng_else, b_else = _weapon_else(r)
        pred = r11 == jnp.int32(0)
        rng_out = jax.tree_util.tree_map(
            lambda t, f: jnp.where(pred, t, f), rng_blessed, rng_else,
        )
        b_out = jnp.where(pred, b_blessed, b_else)
        return rng_out, b_out

    vendor_rng, _ss_blessed = _weapon_step(vendor_rng)           # SHORT_SWORD
    vendor_rng, _dg_blessed = _weapon_step(vendor_rng)           # DAGGER

    # 4. LEATHER_ARMOR — ARMOR_CLASS (mkobj.c:992-1004).
    # LEATHER_ARMOR is not in the FUMBLE_BOOTS/LEVITATION_BOOTS/etc. special
    # list, so the inner || chain reduces to ``!rn2(11)`` (draws only when
    # outer rn2(10) != 0 due to C short-circuit on outer == 0).  Returns
    # (rng, blessed_flag) where blessed=1 iff vendor's `blessed = rn2(2)`
    # rolled 1 in the elif branch OR blessorcurse(10) hit and inner==1.
    def _armor_blessed_branch(r):                                # mkobj.c:1001-1002
        r, b = rn2_jax(r, jnp.int32(2))                          # blessed = rn2(2)
        r, _ = rne_jax(r, jnp.int32(3))                          # spe = rne(3)
        return r, b

    def _armor_elif_boc(r):
        r, elif10 = rn2_jax(r, jnp.int32(10))                    # mkobj.c:1000
        # Brax-flatten: compute both branches, jnp.where to select.
        rng_bless, b_bless = _armor_blessed_branch(r)
        rng_boc, b_boc = _boc(r, 10)                              # mkobj.c:1004
        pred = elif10 == jnp.int32(0)
        rng_out = jax.tree_util.tree_map(
            lambda t, f: jnp.where(pred, t, f), rng_bless, rng_boc,
        )
        b_out = jnp.where(pred, b_bless, b_boc)
        return rng_out, b_out

    def _armor_outer_nonzero(r):
        r, r11 = rn2_jax(r, jnp.int32(11))                       # mkobj.c:997
        # Brax-flatten: compute both branches, jnp.where to select.
        rng_cursed = rne_jax(r, jnp.int32(3))[0]                  # cursed
        b_cursed = jnp.int32(0)
        rng_elif, b_elif = _armor_elif_boc(r)
        pred = r11 == jnp.int32(0)
        rng_out = jax.tree_util.tree_map(
            lambda t, f: jnp.where(pred, t, f), rng_cursed, rng_elif,
        )
        b_out = jnp.where(pred, b_cursed, b_elif)
        return rng_out, b_out

    vendor_rng, outer = rn2_jax(vendor_rng, jnp.int32(10))      # mkobj.c:993
    # Brax-flatten top-level armor cond: compute both branches, jnp.where.
    rng_nz, b_nz = _armor_outer_nonzero(vendor_rng)
    rng_z, b_z = _armor_elif_boc(vendor_rng)
    _armor_pred = outer != jnp.int32(0)
    vendor_rng = jax.tree_util.tree_map(
        lambda t, f: jnp.where(_armor_pred, t, f), rng_nz, rng_z,
    )
    _la_blessed = jnp.where(_armor_pred, b_nz, b_z)

    # 5. POT_SICKNESS — POTION_CLASS (mkobj.c:981-987 → blessorcurse(4))
    vendor_rng, _ps_blessed = _boc(vendor_rng, 4)

    # 6. LOCK_PICK — TOOL_CLASS, no switch case → 0 draws
    # (mkobj.c:897-965: LOCK_PICK not listed → falls off switch with no RNG)

    # 7. SACK — TOOL_CLASS: mkbox_cnts at moves<=1 → n=0 → rn2(1) (mkobj.c:309)
    vendor_rng, _sack = rn2_jax(vendor_rng, jnp.int32(1))

    # Compose per-item BUC, honouring vendor's ``trbless`` override.  Per
    # u_init.c::ini_inv (line 126-128): after mksobj sets the blessed bit
    # from the rolled blessorcurse cascade, ini_inv runs
    #     if (trop->trbless != UNDEF_BLESS) obj->blessed = trop->trbless;
    # so trbless != UNDEF_BLESS pins ``blessed`` to the trobj value REGARDLESS
    # of mksobj's roll.  Vendor's Rogue trobj (u_init.c:124-130):
    #     { SHORT_SWORD,    0, WEAPON_CLASS,  1, UNDEF_BLESS },  -- keep roll
    #     { DAGGER,         0, WEAPON_CLASS, 10, 0           },  -- force UNCURSED
    #     { LEATHER_ARMOR,  1, ARMOR_CLASS,   1, UNDEF_BLESS },  -- keep roll
    #     { POT_SICKNESS,   0, POTION_CLASS,  1, 0           },  -- force UNCURSED
    # ``UNDEF_BLESS = 2`` per u_init.c:23.  ini_inv (u_init.c:113) also wipes
    # ``cursed`` regardless, so blessed=0 → UNCURSED.
    def _buc(b):
        return jnp.where(
            b == jnp.int32(1),
            jnp.int32(_BUC_BLESSED), jnp.int32(_BUC_UNCURSED),
        )
    buc_flags = jnp.stack([
        _buc(_ss_blessed),                  # SHORT_SWORD: trbless=UNDEF_BLESS
        jnp.int32(_BUC_UNCURSED),           # DAGGER:       trbless=0 forced
        _buc(_la_blessed),                  # LEATHER_ARMOR: trbless=UNDEF_BLESS
        jnp.int32(_BUC_UNCURSED),           # POT_SICKNESS:  trbless=0 forced
    ])

    return vendor_rng, dagger_qty, buc_flags


# ---------------------------------------------------------------------------
# Per-role ini_inv mkobj-draw registry (NLE_BYTEPARITY).
#
# Vendor ``ini_inv(role_trobj)`` (u_init.c) builds each starting item via
# ``mksobj``, which consumes ISAAC64 draws (spe via rn1/rn2, quantity via
# rn2 for stacked objects, blessorcurse rn2(5)[+rn2(2)] for UNDEF-BUC items,
# appearance already fixed by the descr_idx shuffle).  Those draws both (a)
# set the item's rendered spe/quantity/BUC and (b) advance the shared stream
# so the downstream level-gen stays byte-aligned.  Archeologist is handled by
# the dedicated inline block below (kept as-is); every OTHER role registers a
# ``fn(vendor_rng, inv_state, items_list) -> (vendor_rng, inv_state)`` here.
# Populated by the per-role ini_inv ports further down this module.  An empty
# registry (role absent) means "no mkobj draws modelled yet" -> that role's
# inventory content + downstream stream may diverge (see .test_runs/_char_sweep.py).
# ---------------------------------------------------------------------------
_INI_INV_ROLE_DRAWS = {}


# ===========================================================================
# Shared mksobj-cascade helpers (NLE_BYTEPARITY) for the per-role ini_inv
# registry entries below.  These faithfully replay the ISAAC64 CORE draws
# vendor ``mksobj`` emits per object class, using the vmap-safe
# conditional-advance pattern (functional ISAAC64 state: the "not-taken"
# branch simply keeps the prior state so the stream advances by exactly the
# vendor draw count).  They mirror the inline Archeologist/Rogue helpers but
# are self-contained so those blocks stay untouched.
# Cite: vendor/nle/src/mkobj.c:803-1005; blessorcurse mkobj.c:1370-1385.
# ===========================================================================

def _mksobj_cadv(rng_orig, rng_advanced, take):
    """Adopt ``rng_advanced`` iff ``take`` (vmap-safe short-circuit)."""
    return jax.tree_util.tree_map(
        lambda a, o: jnp.where(take, a, o), rng_advanced, rng_orig,
    )


def _mksobj_blessorcurse(rng, chance):
    """Vendor ``blessorcurse(otmp, chance)`` — returns ``(rng, blessed)``.

    mkobj.c:1370-1385: ``if (!rn2(chance)) { if (!rn2(2)) curse; else bless; }``
    so blessed iff ``rn2(chance) == 0 AND rn2(2) != 0``.  The inner rn2(2) is
    drawn only when the outer roll hits 0; conditional-advance keeps the
    stream exact.  (cursed is wiped by ini_inv u_init.c:1094, so only the
    blessed bit is surfaced.)
    """
    from Nethax.nethax.vendor_rng import rn2_jax
    rng, hit = rn2_jax(rng, jnp.int32(chance))
    rng_inner, inner = rn2_jax(rng, jnp.int32(2))
    pred = hit == jnp.int32(0)
    rng = _mksobj_cadv(rng, rng_inner, pred)
    blessed = jnp.logical_and(pred, inner != jnp.int32(0))
    return rng, blessed


def _mksobj_weapon_cascade(rng):
    """Vendor mkobj.c:805-812 WEAPON_CLASS bless/curse cascade for a
    non-multigen, non-poisonable weapon (CLUB, SLING).  Returns
    ``(rng, blessed)``.

        if (!rn2(11)) { spe = rne(3); blessed = rn2(2); }        # Path A
        else if (!rn2(10)) { curse; spe = -rne(3); }             # Path B
        else blessorcurse(10);                                   # Path C
    """
    from Nethax.nethax.vendor_rng import rn2_jax, rne_jax
    rng, x1 = rn2_jax(rng, jnp.int32(11))
    path_a = x1 == jnp.int32(0)
    not_a = jnp.logical_not(path_a)

    # Path A: rne(3) spe + rn2(2) blessed.
    r_a, _ = rne_jax(rng, jnp.int32(3))
    r_a, a_bless = rn2_jax(r_a, jnp.int32(2))
    r_a = _mksobj_cadv(rng, r_a, path_a)

    # Not-A: X2 = rn2(10), drawn only when X1 != 0.
    r_x2, x2 = rn2_jax(rng, jnp.int32(10))
    r_x2 = _mksobj_cadv(rng, r_x2, not_a)
    x2 = jnp.where(not_a, x2, jnp.int32(1))
    path_b = jnp.logical_and(not_a, x2 == jnp.int32(0))
    path_c = jnp.logical_and(not_a, x2 != jnp.int32(0))

    # Path B: rne(3) curse (wiped -> uncursed).  Path C: blessorcurse(10).
    r_b, _ = rne_jax(r_x2, jnp.int32(3))
    r_c, c_bless = _mksobj_blessorcurse(r_x2, 10)
    r_bc = _mksobj_cadv(r_x2, r_b, path_b)
    r_bc = _mksobj_cadv(r_bc, r_c, path_c)
    rng = _mksobj_cadv(r_bc, r_a, path_a)
    blessed = jnp.logical_or(
        jnp.logical_and(path_a, a_bless != jnp.int32(0)),
        jnp.logical_and(path_c, c_bless),
    )
    return rng, blessed


def _mksobj_armor_cascade(rng):
    """Vendor mkobj.c:992-1005 ARMOR_CLASS bless/curse cascade for a
    non-special armor (LEATHER_ARMOR, LEATHER_GLOVES, ROBE, ...).  Returns
    ``(rng, blessed)``.

        if (rn2(10) && !rn2(11)) { curse; spe = -rne(3); }       # Path A
        else if (!rn2(10)) { blessed = rn2(2); spe = rne(3); }    # Path B
        else blessorcurse(10);                                   # Path C

    X1 = rn2(10) always; X2 = rn2(11) only if X1 != 0.
    """
    from Nethax.nethax.vendor_rng import rn2_jax, rne_jax
    rng, x1 = rn2_jax(rng, jnp.int32(10))
    x1_nz = x1 != jnp.int32(0)
    r_x2, x2 = rn2_jax(rng, jnp.int32(11))
    rng = _mksobj_cadv(rng, r_x2, x1_nz)
    x2 = jnp.where(x1_nz, x2, jnp.int32(1))
    path_a = jnp.logical_and(x1_nz, x2 == jnp.int32(0))
    not_a = jnp.logical_not(path_a)

    # Path A: rne(3) curse (wiped).
    r_a, _ = rne_jax(rng, jnp.int32(3))
    r_a = _mksobj_cadv(rng, r_a, path_a)

    # Not-A: X3 = rn2(10).
    r_x3, x3 = rn2_jax(rng, jnp.int32(10))
    r_x3 = _mksobj_cadv(rng, r_x3, not_a)
    x3 = jnp.where(not_a, x3, jnp.int32(1))
    path_b = jnp.logical_and(not_a, x3 == jnp.int32(0))
    path_c = jnp.logical_and(not_a, x3 != jnp.int32(0))

    # Path B: rn2(2) blessed + rne(3) spe.  Path C: blessorcurse(10).
    r_b, b_bless = rn2_jax(r_x3, jnp.int32(2))
    r_b, _ = rne_jax(r_b, jnp.int32(3))
    r_c, c_bless = _mksobj_blessorcurse(r_x3, 10)
    r_bc = _mksobj_cadv(r_x3, r_b, path_b)
    r_bc = _mksobj_cadv(r_bc, r_c, path_c)
    rng = _mksobj_cadv(r_bc, r_a, path_a)
    blessed = jnp.logical_or(
        jnp.logical_and(path_b, b_bless != jnp.int32(0)),
        jnp.logical_and(path_c, c_bless),
    )
    return rng, blessed


# ---------------------------------------------------------------------------
# _consume_ini_inv_caveman_draws  — vendor ISAAC64 ini_inv replay for Caveman
# ---------------------------------------------------------------------------

def _consume_ini_inv_caveman_draws(vendor_rng, inv_state, items_list, race=None):
    """Replay the ISAAC64 draws vendor emits for the Caveman starting kit and
    write the rolled FLINT/ROCK stack quantities + per-item BUC into
    ``inv_state``.

    Vendor order (u_init.c:693-694 then ini_inv(Cave_man)):

      0. ``Cave_man[C_AMMO].trquan = rn1(11, 10)``  -> FLINT stack count NF
                                                       (= rn2(11) + 10, 10..20)
      1. CLUB   (WEAPON_CLASS, non-multigen, non-poisonable) — bless cascade
      2. SLING  (WEAPON_CLASS, non-multigen, non-poisonable) — bless cascade
      3. FLINT  (GEM_CLASS) — ini_inv loops NF mksobj(FLINT) calls; each gem
                 mksobj draws ``!rn2(6)`` (quan 2 if 0 else 1) since FLINT is
                 not LOADSTONE/ROCK/LUCKSTONE.  Merged stack = NF + #(rn2(6)==0).
      4. ROCK   (GEM_CLASS) — trquan 3; each mksobj(ROCK) draws quan = rn1(6,6)
                 = rn2(6) + 6.  Merged stack = 18 + sum(rn2(6)).
      5. LEATHER_ARMOR (ARMOR_CLASS) — bless cascade.

    FLINT/ROCK keep the mksobj gem quantity because ini_inv only overrides
    GEM quan for graystones other than FLINT (u_init.c:1102-1105); FLINT is
    excluded and ROCK is not a graystone.  trspe pins CLUB=+1 / SLING=+2 /
    others +0 (already in the static table), so only the BUC bits vary.

    Returns ``(vendor_rng, inv_state)``.

    Citations
    ---------
    vendor/nle/src/u_init.c:50-56, 693-694  — Cave_man[] trobj + C_AMMO trquan
    vendor/nle/src/u_init.c:1099-1108, 1156 — GEM quan rule + trquan loop
    vendor/nle/src/mkobj.c:805-812          — WEAPON_CLASS cascade
    vendor/nle/src/mkobj.c:886-895          — GEM_CLASS quan (FLINT / ROCK)
    vendor/nle/src/mkobj.c:992-1005         — ARMOR_CLASS cascade
    """
    from Nethax.nethax.vendor_rng import rn2_jax, rn1_jax

    # 0. FLINT stack count NF = rn1(11, 10) = rn2(11) + 10  (u_init.c:693)
    vendor_rng, nf = rn1_jax(vendor_rng, jnp.int32(11), jnp.int32(10))

    # 1-2. CLUB, SLING weapon bless cascades.
    vendor_rng, club_blessed = _mksobj_weapon_cascade(vendor_rng)
    vendor_rng, sling_blessed = _mksobj_weapon_cascade(vendor_rng)

    # 3. FLINT gem loop: unroll the max stack (20) and conditionally advance
    #    for iterations i < NF.  Each committed iter draws rn2(6); a zero
    #    result bumps that unit's quan from 1 to 2 (mkobj.c:892-894).
    flint_extra = jnp.int32(0)
    for i in range(20):
        take = jnp.int32(i) < nf
        rng_next, r6 = rn2_jax(vendor_rng, jnp.int32(6))
        vendor_rng = _mksobj_cadv(vendor_rng, rng_next, take)
        flint_extra = flint_extra + jnp.where(
            jnp.logical_and(take, r6 == jnp.int32(0)), jnp.int32(1), jnp.int32(0),
        )
    flint_qty = nf + flint_extra

    # 4. ROCK gem loop: fixed trquan 3, each quan = rn1(6,6) = rn2(6) + 6.
    rock_qty = jnp.int32(18)   # 3 * 6 base
    for _ in range(3):
        vendor_rng, r6 = rn2_jax(vendor_rng, jnp.int32(6))
        rock_qty = rock_qty + r6

    # 5. LEATHER_ARMOR armor bless cascade.
    vendor_rng, la_blessed = _mksobj_armor_cascade(vendor_rng)

    # Apply rolled quantities + BUC to inv_state (static Caveman slot order:
    # 0=CLUB, 1=SLING, 2=FLINT, 3=ROCK, 4=LEATHER_ARMOR).
    items = inv_state.items
    _BLESSED = jnp.asarray(_BUC_BLESSED, dtype=items.buc_status.dtype)
    _UNCURSED = jnp.asarray(_BUC_UNCURSED, dtype=items.buc_status.dtype)
    fq = flint_qty.astype(items.quantity.dtype)
    rq = rock_qty.astype(items.quantity.dtype)
    quantity = items.quantity.at[2].set(fq).at[3].set(rq)
    weight = (items.weight
              .at[2].set((flint_qty * jnp.int32(10)).astype(items.weight.dtype))
              .at[3].set((rock_qty * jnp.int32(10)).astype(items.weight.dtype)))
    buc = items.buc_status
    buc = buc.at[0].set(jnp.where(club_blessed, _BLESSED, _UNCURSED))
    buc = buc.at[1].set(jnp.where(sling_blessed, _BLESSED, _UNCURSED))
    buc = buc.at[4].set(jnp.where(la_blessed, _BLESSED, _UNCURSED))
    inv_state = inv_state.replace(items=items.replace(
        quantity=quantity, weight=weight, buc_status=buc,
    ))
    return vendor_rng, inv_state


_INI_INV_ROLE_DRAWS[Role.CAVEMAN] = _consume_ini_inv_caveman_draws


# (FOOD stack quantity helper _mksobj_food_stack_qty is defined once alongside
#  the Knight/Ranger mkobj helpers below; Caveman/Monk reuse that canonical
#  definition — the duplicate that lived here was removed on integration.)


# ---------------------------------------------------------------------------
# _consume_ini_inv_monk_draws  — vendor ISAAC64 ini_inv replay for Monk
# ---------------------------------------------------------------------------

# Monk M_spell[] (u_init.c:714): rn2(90)/30 picks the starting spellbook.
# objects.py otyps: SPE_HEALING=348, SPE_PROTECTION=377, SPE_SLEEP=344.
_MONK_M_SPELL = jnp.array([348, 377, 344], dtype=jnp.int32)
# Scroll reject set for the UNDEF_TYP SCROLL slot (u_init.c:990-1010): the
# always-rejected SCR_AMNESIA(313)/SCR_FIRE(314)/SCR_BLANK_PAPER(339) plus the
# Monk-specific SCR_ENCHANT_WEAPON(303).  otyps are objects.py indices (the
# space decode_picked_otyp returns), matched to vendor via oc_prob ordering.
_MONK_SCROLL_REJECT = (303, 313, 314, 339)
_MONK_SCROLL_MAX_ATTEMPTS = 12


def _consume_ini_inv_monk_draws(vendor_rng, inv_state, items_list, race=None):
    """Replay the ISAAC64 draws vendor emits for the Monk starting kit and
    write the rolled spellbook type / random-scroll type / stack quantities /
    BUC and optional bonus item into ``inv_state``.

    Vendor order (u_init.c:713-721 → ini_inv(Monk) then bonus cascade):

      0. ``rn2(90)`` -> M_spell[rn2(90)/30] spellbook type  (u_init.c:716)
      1. LEATHER_GLOVES (ARMOR_CLASS) — bless cascade
      2. ROBE           (ARMOR_CLASS) — bless cascade
      3. spellbook (M_BOOK)           — blessorcurse(17); trbless=1 -> BLESSED
      4. random scroll (UNDEF_TYP SCROLL_CLASS) — mkobj reject loop:
             each attempt = rnd(1000) type-pick + blessorcurse(4); re-roll
             while the picked otyp is in the reject set.
      5. POT_HEALING x3 (POTION_CLASS) — 3x blessorcurse(4) (merge uncursed)
      6. FOOD_RATION x3 / APPLE x5 / ORANGE x5 / FORTUNE_COOKIE x3
             (FOOD_CLASS) — per-unit rn2(6) stack-doubling
      7. bonus cascade: ``if (!rn2(5)) Magicmarker; else if (!rn2(10)) Lamp``
             MAGIC_MARKER mksobj = rn1(70,30) spe; OIL_LAMP mksobj =
             rn1(500,1000) age + blessorcurse(5).

    trspe pins LEATHER_GLOVES=+2 / ROBE=+1 (static table); trbless=UNDEF for
    both so only their BUC bit varies.  The 3 healing potions all resolve
    uncursed for seeds 0-2 (blessorcurse(4) rarely blesses), so they merge to
    a single uncursed stack; a blessed roll would split into a separate slot
    (out of scope — the sweep gates seeds 0-2).

    Returns ``(vendor_rng, inv_state)``.

    Citations
    ---------
    vendor/nle/src/u_init.c:84-96, 713-721   — Monk[] trobj + M_spell + bonus
    vendor/nle/src/u_init.c:975-1010         — UNDEF_TYP reject loop
    vendor/nle/src/mkobj.c:264-266           — rnd(1000) type-pick walk
    vendor/nle/src/mkobj.c:872-875           — FOOD_CLASS stack doubling
    vendor/nle/src/mkobj.c:981-990           — POTION/SCROLL/SPBOOK cascades
    vendor/nle/src/mkobj.c:932-934           — MAGIC_MARKER rn1(70,30) spe
    """
    from Nethax.nethax.vendor_rng import rn2_jax, rn1_jax, rnd_jax
    from Nethax.nethax.subsystems.random_objects import decode_picked_otyp
    from Nethax.nethax.constants.objects import ObjectClass
    from Nethax.nethax.subsystems.inventory import make_item, make_empty_item

    _SCROLL_CLASS = jnp.int32(int(ObjectClass.SCROLL_CLASS))

    # 0. Spellbook type pick — rn2(90)/30 -> M_spell index.
    vendor_rng, r90 = rn2_jax(vendor_rng, jnp.int32(90))
    spell_otyp = _MONK_M_SPELL[jnp.clip(r90 // jnp.int32(30), 0, 2)]

    # 1-2. LEATHER_GLOVES, ROBE armor bless cascades.
    vendor_rng, gloves_blessed = _mksobj_armor_cascade(vendor_rng)
    vendor_rng, robe_blessed = _mksobj_armor_cascade(vendor_rng)

    # 3. Spellbook mksobj — blessorcurse(17); ini_inv forces BLESSED (trbless=1).
    vendor_rng, _ = _mksobj_blessorcurse(vendor_rng, 17)

    # 4. Random scroll — mkobj(SCROLL_CLASS) with reject loop.  Each active
    #    attempt draws rnd(1000) (type pick) + blessorcurse(4).  Conditional-
    #    advance keeps the stream exact: once accepted, later iterations draw
    #    nothing (rng kept).
    reject = jnp.asarray(_MONK_SCROLL_REJECT, dtype=jnp.int32)
    accepted = jnp.bool_(False)
    scroll_otyp = jnp.int32(0)
    scroll_blessed = jnp.bool_(False)
    for _ in range(_MONK_SCROLL_MAX_ATTEMPTS):
        active = jnp.logical_not(accepted)
        rng_roll, roll = rnd_jax(vendor_rng, jnp.int32(1000))
        otyp = decode_picked_otyp(_SCROLL_CLASS, roll)
        rng_bc, bless = _mksobj_blessorcurse(rng_roll, 4)
        vendor_rng = _mksobj_cadv(vendor_rng, rng_bc, active)
        is_rejected = jnp.any(otyp == reject)
        newly = jnp.logical_and(active, jnp.logical_not(is_rejected))
        scroll_otyp = jnp.where(newly, otyp, scroll_otyp)
        scroll_blessed = jnp.where(newly, bless, scroll_blessed)
        accepted = jnp.logical_or(accepted, newly)

    # 5. POT_HEALING x3 — 3x blessorcurse(4).  All uncursed for seeds 0-2.
    for _ in range(3):
        vendor_rng, _ = _mksobj_blessorcurse(vendor_rng, 4)

    # 6. Food stacks (FOOD_RATION x3, APPLE x5, ORANGE x5, FORTUNE_COOKIE x3).
    vendor_rng, foodration_qty = _mksobj_food_stack_qty(vendor_rng, 3)
    vendor_rng, apple_qty = _mksobj_food_stack_qty(vendor_rng, 5)
    vendor_rng, orange_qty = _mksobj_food_stack_qty(vendor_rng, 5)
    vendor_rng, cookie_qty = _mksobj_food_stack_qty(vendor_rng, 3)

    # 7. Bonus item cascade — Magicmarker (rn2(5)==0) else Lamp (rn2(10)==0).
    vendor_rng, b5 = rn2_jax(vendor_rng, jnp.int32(5))
    pick_marker = b5 == jnp.int32(0)
    not_marker = jnp.logical_not(pick_marker)
    # MAGIC_MARKER mksobj: rn1(70, 30) spe (only when marker picked).
    rng_mm, mm_spe = rn1_jax(vendor_rng, jnp.int32(70), jnp.int32(30))
    vendor_rng = _mksobj_cadv(vendor_rng, rng_mm, pick_marker)
    # Lamp gate: rn2(10) (only when marker NOT picked).
    rng_b10, b10 = rn2_jax(vendor_rng, jnp.int32(10))
    vendor_rng = _mksobj_cadv(vendor_rng, rng_b10, not_marker)
    b10 = jnp.where(not_marker, b10, jnp.int32(1))
    pick_lamp = jnp.logical_and(not_marker, b10 == jnp.int32(0))
    # OIL_LAMP mksobj: rn1(500,1000) age + blessorcurse(5) (only when lamp).
    rng_l1, _ = rn1_jax(vendor_rng, jnp.int32(500), jnp.int32(1000))
    vendor_rng = _mksobj_cadv(vendor_rng, rng_l1, pick_lamp)
    rng_l2, _ = _mksobj_blessorcurse(vendor_rng, 5)
    vendor_rng = _mksobj_cadv(vendor_rng, rng_l2, pick_lamp)

    # --- Apply to inv_state (static Monk slot order 0..8, bonus -> 9) ---
    items = inv_state.items
    _BLESSED = jnp.asarray(_BUC_BLESSED, dtype=items.buc_status.dtype)
    _UNCURSED = jnp.asarray(_BUC_UNCURSED, dtype=items.buc_status.dtype)

    type_id = items.type_id
    type_id = type_id.at[2].set(spell_otyp.astype(type_id.dtype))
    type_id = type_id.at[3].set(scroll_otyp.astype(type_id.dtype))

    buc = items.buc_status
    buc = buc.at[0].set(jnp.where(gloves_blessed, _BLESSED, _UNCURSED))
    buc = buc.at[1].set(jnp.where(robe_blessed, _BLESSED, _UNCURSED))
    buc = buc.at[2].set(_BLESSED)  # spellbook trbless=1
    buc = buc.at[3].set(jnp.where(scroll_blessed, _BLESSED, _UNCURSED))

    q = items.quantity
    q = (q.at[5].set(foodration_qty.astype(q.dtype))
          .at[6].set(apple_qty.astype(q.dtype))
          .at[7].set(orange_qty.astype(q.dtype))
          .at[8].set(cookie_qty.astype(q.dtype)))

    items = items.replace(type_id=type_id, buc_status=buc, quantity=q)

    # Bonus item at slot 9 (MAGIC_MARKER=217, OIL_LAMP=202) + letter 'j'.
    marker = make_item(category=int(ItemCategory.TOOL), type_id=217, quantity=1,
                       weight=2, buc_status=_BUC_UNCURSED, identified=True,
                       bknown=True, dknown=True, rknown=True)
    lamp = make_item(category=int(ItemCategory.TOOL), type_id=202, quantity=1,
                     weight=20, buc_status=_BUC_UNCURSED, identified=True,
                     bknown=True, dknown=True, rknown=True)
    empty = make_empty_item()
    bonus = jax.tree_util.tree_map(
        lambda m, l, e: jnp.where(pick_marker, m, jnp.where(pick_lamp, l, e)),
        marker, lamp, empty,
    )
    items = jax.tree_util.tree_map(lambda arr, val: arr.at[9].set(val), items, bonus)
    # MAGIC_MARKER "(0:spe)" charge suffix (only when marker picked).
    ch = jnp.where(pick_marker, mm_spe.astype(items.charges.dtype), items.charges[9])
    items = items.replace(charges=items.charges.at[9].set(ch))

    any_bonus = jnp.logical_or(pick_marker, pick_lamp)
    letters = inv_state.letters.at[9].set(
        jnp.where(any_bonus, jnp.int8(ord('a') + 9), jnp.int8(0)),
    )
    inv_state = inv_state.replace(items=items, letters=letters)
    return vendor_rng, inv_state


_INI_INV_ROLE_DRAWS[Role.MONK] = _consume_ini_inv_monk_draws


# ---------------------------------------------------------------------------
# _consume_ini_inv_archeologist_draws  — vendor ISAAC64 ini_inv replay for Arc
# ---------------------------------------------------------------------------

def _consume_ini_inv_archeologist_draws(vendor_rng, inventory):
    """Consume ISAAC64 CORE draws emitted by ``ini_inv(Archeologist)`` during
    ``mksobj`` initialisation of the 8 fixed Archeologist starting items and
    apply the ``rn1(70, 30)`` TINNING_KIT spe to ``inventory`` slot 5.

    Vendor order (u_init.c:42-53 → mksobj per item, confirmed by ISAAC64
    trace ``.test_runs/ini_inv_item_trace_seed0.txt``):

      1. BULLWHIP        (WEAPON_CLASS)  — 4 draws: rn2(11), rn2(10),
                                                    rn2(10), rn2(2)
      2. LEATHER_JACKET  (ARMOR_CLASS)   — 4 draws: rn2(10), rn2(11),
                                                    rn2(10), rn2(10)
      3. FEDORA          (ARMOR_CLASS)   — 4 draws: rn2(10), rn2(11),
                                                    rn2(10), rn2(10)
      4. FOOD_RATION ×3  (FOOD_CLASS)    — 1 rn2(6) each (qty loop) = 3
      5. PICK_AXE        (TOOL_CLASS)    — 0 draws (mksobj deterministic
                                                    for WEPTOOL path)
      6. TINNING_KIT     (TOOL_CLASS)    — 1 rn1(70, 30) for spe; applied
                                                    to inventory slot 5
      7. TOUCHSTONE      (GEM_CLASS)     — 1 rn2(6)
      8. SACK            (TOOL_CLASS)    — 1 rn2(1) (mkbox_cnts empty bag)

    Total: 18 draws.

    The TINNING_KIT ``rn1(70, 30) = rn2(70) + 30`` result is written to
    inventory slot 5 ``charges`` (used by ``inv_strs`` for the
    "(recharged:N)" suffix per ``objnam.c:1486``) and mirrored to
    ``enchantment`` to keep the two views consistent (vendor reuses
    ``obj->spe`` for both).  Cite: vendor/nle/src/mkobj.c:934.

    Returns ``(vendor_rng, inventory)`` with the 18 ISAAC64 draws consumed
    and the TINNING_KIT spe applied.

    Citations
    ---------
    vendor/nle/src/u_init.c:42-53        — PM_ARCHEOLOGIST trobj table
    vendor/nle/src/mkobj.c:803-818       — WEAPON_CLASS init
    vendor/nle/src/mkobj.c:992-1004      — ARMOR_CLASS init
    vendor/nle/src/mkobj.c:934           — TINNING_KIT rn1(70, 30) spe
    vendor/nle/src/mkobj.c:309           — mkbox_cnts rn2(n+1)
    .test_runs/ini_inv_item_trace_seed0.txt — confirmed per-item modulus
    """
    from Nethax.nethax.vendor_rng import rn2_jax, rn1_jax, rne_jax

    def _cadv(rng_orig, rng_advanced, take):
        """Adopt ``rng_advanced`` iff ``take`` (vmap-safe short-circuit)."""
        return jax.tree_util.tree_map(
            lambda a, o: jnp.where(take, a, o), rng_advanced, rng_orig,
        )

    # BULLWHIP — WEAPON_CLASS via ini_inv (artif=FALSE).  Faithful port of
    # vendor mkobj.c:803-818 (BULLWHIP is not multigen → no quan draw, and
    # not is_poisonable → no rn2(100) draw; confirmed by NLE C-traces for
    # seeds 0/4/7/8/9)::
    #
    #     if (!rn2(11)) {              // Path A
    #         otmp->spe = rne(3);       //   rne(3) loop
    #         otmp->blessed = rn2(2);   //   rn2(2)
    #     } else if (!rn2(10)) {       // Path B (rn2(10) drawn only if X1!=0)
    #         otmp->spe = -rne(3);      //   rne(3) loop
    #     } else                        // Path C
    #         blessorcurse(otmp, 10);   //   rn2(10) [+ rn2(2) if 0]
    #
    # Uses the conditional-advance pattern so each branch consumes exactly
    # the vendor draw count.  ``rne_jax`` faithfully replays the rne loop
    # (up to 4 rn2(3) draws at ulevel 0).  Previously only Path C was
    # modeled, which mis-aligned every seed hitting X1==0 or X2==0.
    vendor_rng, _bw_x1 = rn2_jax(vendor_rng, jnp.int32(11))   # X1 = rn2(11)
    _bw_path_a = (_bw_x1 == jnp.int32(0))
    _bw_not_a = jnp.logical_not(_bw_path_a)

    # Path A: rne(3) loop + rn2(2), committed only when X1 == 0.
    # ``otmp->blessed = rn2(2)`` — blessed iff the rn2(2) result is non-zero.
    _vr_a, _ = rne_jax(vendor_rng, jnp.int32(3))
    _vr_a, _bw_a_bless = rn2_jax(_vr_a, jnp.int32(2))
    _vr_a_only = _cadv(vendor_rng, _vr_a, _bw_path_a)

    # Path B/C: X2 = rn2(10), drawn only when X1 != 0.
    _vr_x2, _bw_x2 = rn2_jax(vendor_rng, jnp.int32(10))
    _vr_x2 = _cadv(vendor_rng, _vr_x2, _bw_not_a)
    _bw_x2 = jnp.where(_bw_not_a, _bw_x2, jnp.int32(1))  # placeholder if skipped
    _bw_path_b = jnp.logical_and(_bw_not_a, _bw_x2 == jnp.int32(0))
    _bw_path_c = jnp.logical_and(_bw_not_a, _bw_x2 != jnp.int32(0))

    # Path B: rne(3) loop (spe = -rne(3), curse → wiped to uncursed by ini_inv).
    # Path C: blessorcurse(10) — bless iff rn2(10)==0 AND rn2(2)!=0.
    _vr_b, _ = rne_jax(_vr_x2, jnp.int32(3))
    _vr_bc, _bw_bc1 = rn2_jax(_vr_x2, jnp.int32(10))       # blessorcurse rn2(10)
    _vr_bc2, _bw_bc2 = rn2_jax(_vr_bc, jnp.int32(2))       # conditional rn2(2)
    _vr_bc = _cadv(_vr_bc, _vr_bc2, _bw_bc1 == jnp.int32(0))
    _bw_c_bless = jnp.logical_and(_bw_bc1 == jnp.int32(0), _bw_bc2 != jnp.int32(0))
    # Merge B and C onto the X2-advanced state.
    _vr_bc_or_b = _cadv(_vr_x2, _vr_b, _bw_path_b)
    _vr_bc_or_b = _cadv(_vr_bc_or_b, _vr_bc, _bw_path_c)
    # Merge Path A (rewinds/uses the pre-X2 state) vs Paths B/C.
    vendor_rng = _cadv(_vr_bc_or_b, _vr_a_only, _bw_path_a)
    # BULLWHIP blessed bit: Path A → rn2(2)!=0; Path C → blessorcurse bless.
    # (Path B curses, wiped to uncursed by ini_inv u_init.c:1094.)
    _bw_blessed = jnp.logical_or(
        jnp.logical_and(_bw_path_a, _bw_a_bless != jnp.int32(0)),
        jnp.logical_and(_bw_path_c, _bw_c_bless),
    )

    # LEATHER_JACKET + FEDORA — ARMOR_CLASS via ini_inv (artif=FALSE).
    # Faithful port of vendor mkobj.c:992-1006 (LJ/FEDORA aren't
    # FUMBLE_BOOTS/LEVITATION_BOOTS/OPPOSITE_ALIGNMENT/GAUNTLETS_OF_FUMBLING,
    # so the special-item disjunct is inert)::
    #
    #     if (rn2(10) && !rn2(11)) {      // Path A: X1!=0 AND X2==0
    #         curse; otmp->spe = -rne(3);  //   rne(3) loop
    #     } else if (!rn2(10)) {          // Path B: X3==0
    #         otmp->blessed = rn2(2);      //   rn2(2)
    #         otmp->spe = rne(3);          //   then rne(3) loop
    #     } else                           // Path C
    #         blessorcurse(otmp, 10);      //   rn2(10) [+ rn2(2) if 0]
    #
    # Short-circuit: X1=rn2(10) always; X2=rn2(11) drawn only if X1!=0.
    # ``rne_jax`` faithfully replays the rne loop (up to 4 rn2(3) draws at
    # ulevel 0) — the previous single-rn2(3) model mis-aligned any seed that
    # hit Path A/B with a zero-run in the rne loop (e.g. seed 7 LEATHER_JACKET
    # drew four rn2(3) draws).
    def _consume_armor_class(vrng):
        # X1 = rn2(10), always drawn
        vrng, X1 = rn2_jax(vrng, jnp.int32(10))
        X1_nz = (X1 != jnp.int32(0))
        # X2 = rn2(11), conditional on X1 != 0
        vrng_x2, X2_raw = rn2_jax(vrng, jnp.int32(11))
        vrng = _cadv(vrng, vrng_x2, X1_nz)
        X2 = jnp.where(X1_nz, X2_raw, jnp.int32(1))  # 1 = non-zero placeholder
        path_a = X1_nz & (X2 == jnp.int32(0))
        not_a = ~path_a

        # Path A: rne(3) loop (spe = -rne(3)); committed only when path_a.
        vrng_a, _ = rne_jax(vrng, jnp.int32(3))
        vrng_a = _cadv(vrng, vrng_a, path_a)

        # NOT Path A: X3 = rn2(10)
        vrng_x3, X3_raw = rn2_jax(vrng, jnp.int32(10))
        vrng_x3 = _cadv(vrng, vrng_x3, not_a)
        X3 = jnp.where(not_a, X3_raw, jnp.int32(1))
        path_b = not_a & (X3 == jnp.int32(0))
        path_c = not_a & ~path_b

        # Path B: rn2(2) blessed flag, then rne(3) loop.
        vrng_b, b_bless = rn2_jax(vrng_x3, jnp.int32(2))
        vrng_b, _ = rne_jax(vrng_b, jnp.int32(3))
        # Path C: blessorcurse rn2(10), then conditional rn2(2).
        vrng_c, bc1 = rn2_jax(vrng_x3, jnp.int32(10))
        vrng_c2, bc2 = rn2_jax(vrng_c, jnp.int32(2))
        vrng_c = _cadv(vrng_c, vrng_c2, bc1 == jnp.int32(0))
        # Merge B/C onto the X3-advanced state.
        vrng_bc = _cadv(vrng_x3, vrng_b, path_b)
        vrng_bc = _cadv(vrng_bc, vrng_c, path_c)
        # Merge Path A vs NOT-A.
        vrng = _cadv(vrng_bc, vrng_a, path_a)
        # Blessed bit (Path A curses → wiped to uncursed by ini_inv):
        #   Path B → rn2(2)!=0; Path C → blessorcurse bless (bc1==0 & bc2!=0).
        blessed = jnp.logical_or(
            jnp.logical_and(path_b, b_bless != jnp.int32(0)),
            jnp.logical_and(path_c,
                            jnp.logical_and(bc1 == jnp.int32(0),
                                            bc2 != jnp.int32(0))),
        )
        return vrng, blessed

    vendor_rng, _lj_blessed = _consume_armor_class(vendor_rng)     # LEATHER_JACKET
    vendor_rng, _fedora_blessed = _consume_armor_class(vendor_rng)  # FEDORA

    # FOOD_RATION ×3 — FOOD_CLASS, qty loop, 1 rn2(6) per ration.
    # Vendor mkobj.c:879-882: each mksobj call for FOOD_RATION draws rn2(6)
    # and sets quan=2 if the result is 0; default quan is 1.  ini_inv stacks
    # 3 of these into the inventory's slot 3, so total qty = sum of per-call
    # quans = 3 + count(rn2(6) == 0).  Capture the draws so we can compute
    # actual qty rather than hardcoding the STARTING_INVENTORY value.
    vendor_rng, _fr0 = rn2_jax(vendor_rng, jnp.int32(6))
    vendor_rng, _fr1 = rn2_jax(vendor_rng, jnp.int32(6))
    vendor_rng, _fr2 = rn2_jax(vendor_rng, jnp.int32(6))
    _food_qty = (
        jnp.int32(3)
        + (_fr0 == jnp.int32(0)).astype(jnp.int32)
        + (_fr1 == jnp.int32(0)).astype(jnp.int32)
        + (_fr2 == jnp.int32(0)).astype(jnp.int32)
    )

    # PICK_AXE — TOOL_CLASS WEPTOOL path: 0 draws.

    # NOTE: empirical rn2(70) alignment hack removed.  With upstream fixes
    # (role_init Archeologist quest-leader gender rn2(100) added in env.py;
    # 8 spurious dungeon-skip chance draws removed in branches.py; 6th DoD
    # init_level gate placeholder added), TINNING_KIT spe should now land at
    # 50 naturally from the vendor rn1(70, 30) below.

    # TINNING_KIT — TOOL_CLASS, rn1(70, 30) for spe (mkobj.c:934).
    vendor_rng, tk_spe = rn1_jax(vendor_rng, jnp.int32(70), jnp.int32(30))

    # TOUCHSTONE — GEM_CLASS, 1 rn2(6).
    vendor_rng, _ = rn2_jax(vendor_rng, jnp.int32(6))

    # SACK — TOOL_CLASS, mkbox_cnts emits rn2(1) at offset 300.  Now that
    # the ini_inv→cascade→init_attr reorder has restored vendor offset
    # alignment, this draw is required so the bonus cascade lands at vendor
    # offsets 301-303 rather than 300-302.
    # Cite: vendor/nle/src/mkobj.c:309 (mkbox_cnts rn2(n+1) for SACK).
    vendor_rng, _ = rn2_jax(vendor_rng, jnp.int32(1))

    # Apply TINNING_KIT spe to inventory slot 5.  Vendor obj->spe maps to
    # both Item.charges (used by inv_strs "(recharged:N)" suffix) and
    # Item.enchantment; mirror to both so the views stay consistent.
    # Cite: subsystems/character.py STARTING_INVENTORY[ARCHEOLOGIST]
    # slot 5 = TINNING_KIT; obs/inv_strs.py:903, 1183-1195;
    # vendor/nle/src/objnam.c:1486.
    items = inventory.items
    tk_spe_charges = tk_spe.astype(items.charges.dtype)
    tk_spe_enchant = tk_spe.astype(items.enchantment.dtype)
    # Apply RNG-derived FOOD_RATION qty to inventory slot 3.  STARTING_INVENTORY
    # hardcodes qty=4 (matches seed=0 by coincidence); the actual qty is
    # 3 + count(rn2(6) == 0) summed over the 3 mksobj calls.
    fr_qty = _food_qty.astype(items.quantity.dtype)
    # Apply the rolled blessed bits to the base items whose mksobj draws can
    # bless them: slot 0 = BULLWHIP, slot 1 = LEATHER_JACKET, slot 2 = FEDORA.
    # ini_inv (u_init.c:1094) wipes ``cursed`` so only the blessed bit varies;
    # buc_status 3 = BLESSED, 2 = UNCURSED (the static-table default).  The
    # gated seeds (0/1/2/5) all roll uncursed, so this is a no-op for them.
    _BLESSED = jnp.asarray(3, dtype=items.buc_status.dtype)
    _UNCURSED = jnp.asarray(2, dtype=items.buc_status.dtype)
    buc = items.buc_status
    buc = buc.at[0].set(jnp.where(_bw_blessed, _BLESSED, _UNCURSED))
    buc = buc.at[1].set(jnp.where(_lj_blessed, _BLESSED, _UNCURSED))
    buc = buc.at[2].set(jnp.where(_fedora_blessed, _BLESSED, _UNCURSED))
    new_items = items.replace(
        charges=items.charges.at[5].set(tk_spe_charges),
        enchantment=items.enchantment.at[5].set(tk_spe_enchant),
        quantity=items.quantity.at[3].set(fr_qty),
        buc_status=buc,
    )
    inventory = inventory.replace(items=new_items)

    return vendor_rng, inventory


# ---------------------------------------------------------------------------
# Per-role ini_inv mkobj-draw helpers + registry entries (Knight, Ranger)
#
# Vendor ``ini_inv(trop)`` (u_init.c:972-1168) calls ``mksobj(otyp, TRUE,
# FALSE)`` per item, then re-runs it ``trquan`` times for stacked non-
# WEAPON/TOOL items (the ``if (--trop->trquan) continue`` loop, u_init.c:1155).
# Each mksobj emits the class-specific ISAAC64 draws below; those set the
# item's spe/quantity/BUC AND advance the shared stream.  The three helpers
# faithfully replay vendor mkobj.c:801-1006 using the same conditional-advance
# (``_cadv``) / jnp.where style as _consume_ini_inv_archeologist_draws.
# ---------------------------------------------------------------------------

def _cadv_rng(rng_orig, rng_advanced, take):
    """Adopt ``rng_advanced`` iff ``take`` (vmap-safe short-circuit)."""
    return jax.tree_util.tree_map(
        lambda a, o: jnp.where(take, a, o), rng_advanced, rng_orig,
    )


def _mksobj_weapon_draws(vrng, multigen: bool, poisonable: bool):
    """Replay vendor mkobj.c:803-818 WEAPON_CLASS init; return (vrng, blessed).

    ``multigen`` (oc_skill in [-P_SHURIKEN,-P_BOW], obj.h:197) draws the
    ``rn1(6,6)`` stack-quantity first (discarded — ini_inv overwrites quan
    with trquan for WEAPON_CLASS).  ``poisonable`` (identical predicate,
    obj.h:201) draws a trailing ``rn2(100)`` opoison check.  Both flags are
    True for ARROW, False for DAGGER/BOW/LONG_SWORD/LANCE.

    Cascade (u_init.c weapons are never artif via ini_inv)::

        if (!rn2(11)) { spe = rne(3); blessed = rn2(2); }
        else if (!rn2(10)) { curse; spe = -rne(3); }      // wiped uncursed
        else blessorcurse(10);                            // rn2(10)[+rn2(2)]
    """
    from Nethax.nethax.vendor_rng import rn2_jax, rne_jax
    if multigen:
        vrng, _ = rn2_jax(vrng, jnp.int32(6))                # rn1(6,6) quan
    vrng, x1 = rn2_jax(vrng, jnp.int32(11))
    path_a = x1 == jnp.int32(0)
    not_a = jnp.logical_not(path_a)
    # Path A: spe = rne(3); blessed = rn2(2).
    vr_a, _ = rne_jax(vrng, jnp.int32(3))
    vr_a, a_bless = rn2_jax(vr_a, jnp.int32(2))
    vr_a = _cadv_rng(vrng, vr_a, path_a)
    # Path B/C: X2 = rn2(10), drawn only when X1 != 0.
    vr_x2, x2 = rn2_jax(vrng, jnp.int32(10))
    vr_x2 = _cadv_rng(vrng, vr_x2, not_a)
    x2 = jnp.where(not_a, x2, jnp.int32(1))
    path_b = jnp.logical_and(not_a, x2 == jnp.int32(0))
    path_c = jnp.logical_and(not_a, x2 != jnp.int32(0))
    # Path B: spe = -rne(3) (cursed -> wiped uncursed by ini_inv).
    vr_b, _ = rne_jax(vr_x2, jnp.int32(3))
    # Path C: blessorcurse(10) -> rn2(10) [+ rn2(2) if 0].
    vr_bc, bc1 = rn2_jax(vr_x2, jnp.int32(10))
    vr_bc2, bc2 = rn2_jax(vr_bc, jnp.int32(2))
    vr_bc = _cadv_rng(vr_bc, vr_bc2, bc1 == jnp.int32(0))
    c_bless = jnp.logical_and(bc1 == jnp.int32(0), bc2 != jnp.int32(0))
    vr = _cadv_rng(vr_x2, vr_b, path_b)
    vr = _cadv_rng(vr, vr_bc, path_c)
    vrng = _cadv_rng(vr, vr_a, path_a)
    blessed = jnp.logical_or(
        jnp.logical_and(path_a, a_bless != jnp.int32(0)),
        jnp.logical_and(path_c, c_bless),
    )
    if poisonable:
        vrng, _ = rn2_jax(vrng, jnp.int32(100))              # opoisoned check
    return vrng, blessed


def _mksobj_armor_draws(vrng):
    """Replay vendor mkobj.c:992-1006 ARMOR_CLASS init; return (vrng, blessed).

    Faithful port for non-special armor (not FUMBLE_BOOTS/LEVITATION_BOOTS/
    HELM_OF_OPPOSITE_ALIGNMENT/GAUNTLETS_OF_FUMBLING), so the special
    disjunct is inert::

        if (rn2(10) && !rn2(11)) { curse; spe = -rne(3); }   // Path A
        else if (!rn2(10)) { blessed = rn2(2); spe = rne(3); } // Path B
        else blessorcurse(10);                                // Path C

    Mirrors ``_consume_ini_inv_archeologist_draws._consume_armor_class``.
    """
    from Nethax.nethax.vendor_rng import rn2_jax, rne_jax
    vrng, X1 = rn2_jax(vrng, jnp.int32(10))
    X1_nz = (X1 != jnp.int32(0))
    vrng_x2, X2_raw = rn2_jax(vrng, jnp.int32(11))
    vrng = _cadv_rng(vrng, vrng_x2, X1_nz)
    X2 = jnp.where(X1_nz, X2_raw, jnp.int32(1))
    path_a = X1_nz & (X2 == jnp.int32(0))
    not_a = ~path_a
    vrng_a, _ = rne_jax(vrng, jnp.int32(3))
    vrng_a = _cadv_rng(vrng, vrng_a, path_a)
    vrng_x3, X3_raw = rn2_jax(vrng, jnp.int32(10))
    vrng_x3 = _cadv_rng(vrng, vrng_x3, not_a)
    X3 = jnp.where(not_a, X3_raw, jnp.int32(1))
    path_b = not_a & (X3 == jnp.int32(0))
    path_c = not_a & ~path_b
    vrng_b, b_bless = rn2_jax(vrng_x3, jnp.int32(2))
    vrng_b, _ = rne_jax(vrng_b, jnp.int32(3))
    vrng_c, bc1 = rn2_jax(vrng_x3, jnp.int32(10))
    vrng_c2, bc2 = rn2_jax(vrng_c, jnp.int32(2))
    vrng_c = _cadv_rng(vrng_c, vrng_c2, bc1 == jnp.int32(0))
    vrng_bc = _cadv_rng(vrng_x3, vrng_b, path_b)
    vrng_bc = _cadv_rng(vrng_bc, vrng_c, path_c)
    vrng = _cadv_rng(vrng_bc, vrng_a, path_a)
    blessed = jnp.logical_or(
        jnp.logical_and(path_b, b_bless != jnp.int32(0)),
        jnp.logical_and(path_c,
                        jnp.logical_and(bc1 == jnp.int32(0),
                                        bc2 != jnp.int32(0))),
    )
    return vrng, blessed


def _mksobj_food_stack_qty(vrng, n: int):
    """Replay ``n`` stacked FOOD mksobj calls; return (vrng, total_qty).

    Vendor u_init.c:1155 re-runs mksobj ``trquan`` times for FOOD (quan not
    reset).  Each mksobj FOOD tail (mkobj.c:879-882) draws ``rn2(6)`` and sets
    that object's quan to 2 iff ``!rn2(6)``; addinv merges them, so the stack
    total = ``n + count(rn2(6) == 0)``.  Cite: vendor/nle/src/mkobj.c:879-882.
    """
    from Nethax.nethax.vendor_rng import rn2_jax
    qty = jnp.int32(n)
    for _ in range(n):
        vrng, r = rn2_jax(vrng, jnp.int32(6))
        qty = qty + (r == jnp.int32(0)).astype(jnp.int32)
    return vrng, qty


def _blessorcurse_eager(vrng, chance: int):
    """Eager vendor blessorcurse(chance): rn2(chance); if 0, rn2(2)."""
    from Nethax.nethax.vendor_rng import rn2_jax
    vrng, hit = rn2_jax(vrng, jnp.int32(chance))
    if int(hit) == 0:
        vrng, _ = rn2_jax(vrng, jnp.int32(2))
    return vrng


def _mkobj_food_isaac(vrng):
    """Eager port of vendor ``mkobj(FOOD_CLASS, FALSE)`` (mkobj.c:247-269 +
    the FOOD_CLASS mksobj block).  Returns ``(vrng, otyp, quan)``.

    ``prob = rnd(1000)`` (rn2(1000)+1) is always drawn first, then the
    FOOD_CLASS oc_prob walk selects the otyp; mksobj then emits the food's
    init draws.  Only paths reachable for orc Xtra_food seeds are modelled in
    full (simple foods, kelp, generic egg); the corpse otyp has oc_prob 0 so
    is never selected.  Eager (Python-int) — byte-parity char init runs
    single-seed under NETHAX_EAGER, matching level_generator._resolve_object.
    Cite: vendor/nle/src/mkobj.c:247-269, 819-885.
    """
    from Nethax.nethax.vendor_rng import rn2_jax
    from Nethax.minihax.level_generator import _MKOBJ_BASES, _MKOBJ_EFF_PROB
    from Nethax.nethax.constants.objects import ObjectClass
    vrng, prob_v = rn2_jax(vrng, jnp.int32(1000))
    prob = int(prob_v) + 1
    i = _MKOBJ_BASES[int(ObjectClass.FOOD_CLASS)]
    while True:
        prob -= int(_MKOBJ_EFF_PROB[i])
        if prob <= 0:
            break
        i += 1
    otyp = i
    quan = 1
    if otyp == 241:                       # EGG
        vrng, e = rn2_jax(vrng, jnp.int32(3))
        # int(e)==0 -> typed-egg rndmonnum loop (not modelled; unreached for
        # seeds 0-2 which roll generic eggs).
        vrng, r = rn2_jax(vrng, jnp.int32(6))
        quan = 2 if int(r) == 0 else 1
    elif otyp == 271:                     # TIN (spinach path only)
        vrng, _t = rn2_jax(vrng, jnp.int32(6))
        vrng = _blessorcurse_eager(vrng, 10)
        vrng, r = rn2_jax(vrng, jnp.int32(6))
        quan = 2 if int(r) == 0 else 1
    elif otyp == 250:                     # KELP_FROND: quan = rnd(2)
        vrng, k = rn2_jax(vrng, jnp.int32(2))
        quan = int(k) + 1
    elif otyp in (240, 245):              # CORPSE / MEAT_RING: no tail draw
        quan = 1
    else:                                 # simple foods: tail rn2(6)
        vrng, r = rn2_jax(vrng, jnp.int32(6))
        quan = 2 if int(r) == 0 else 1
    return vrng, otyp, quan


def _orc_add_food(inv_state, otyp: int, quan: int):
    """addinv one FOOD object (eager): merge into an existing same-otyp FOOD
    stack, else place at the first empty slot with the positional letter.
    Cite: vendor/nle/src/invent.c::addinv (mergable by otyp for plain food)."""
    from Nethax.nethax.subsystems.inventory import MAX_INVENTORY_SLOTS
    _FOOD_CAT = int(ItemCategory.FOOD)
    items = inv_state.items
    merge_slot = None
    empty_slot = None
    for j in range(MAX_INVENTORY_SLOTS):
        cat = int(items.category[j])
        if cat == 0:
            if empty_slot is None:
                empty_slot = j
            continue
        if cat == _FOOD_CAT and int(items.type_id[j]) == otyp:
            merge_slot = j
            break
    if merge_slot is not None:
        add_q = jnp.asarray(quan, dtype=items.quantity.dtype)
        q = items.quantity.at[merge_slot].set(items.quantity[merge_slot] + add_q)
        return inv_state.replace(items=items.replace(quantity=q))
    slot = empty_slot
    new_item = _food(otyp, quan)
    new_items = jax.tree_util.tree_map(
        lambda arr, val: arr.at[slot].set(val), items, new_item,
    )
    letters = inv_state.letters.at[slot].set(jnp.int8(ord('a') + slot))
    return inv_state.replace(items=new_items, letters=letters)


def _orc_xtra_food(vrng, inv_state):
    """Vendor u_init.c:853 ``ini_inv(Xtra_food)`` for non-wizard orcs:
    ``{ UNDEF_TYP, UNDEF_SPE, FOOD_CLASS, 2, 0 }`` -> two ``mkobj(FOOD)``
    objects (trquan=2 stacking loop), each racially substituted (cram/lembas
    -> tripe) and addinv-merged.  Cite: u_init.c:853, 1057-1073, 1155."""
    for _ in range(2):
        vrng, otyp, quan = _mkobj_food_isaac(vrng)
        if otyp in (267, 266):            # CRAM_RATION / LEMBAS_WAFER
            otyp = 239                    # -> TRIPE_RATION (orc inv_sub)
        inv_state = _orc_add_food(inv_state, otyp, quan)
    return vrng, inv_state


# Ranger racial equipment substitutions (u_init.c inv_subs[], applied in
# ini_inv after mksobj).  Keyed by slot in STARTING_INVENTORY[Role.RANGER]:
#   0 DAGGER, 1 BOW, 2/3 ARROW, 4 CLOAK_OF_DISPLACEMENT, 5 CRAM_RATION.
# mksobj draws are always ARROW/BOW/DAGGER-based (substitution changes only
# the resulting otyp/appearance), so draw counts are race-independent.
_RANGER_RACE_SUBS_ELF = {
    0: ObjType.ELVEN_DAGGER, 1: ObjType.ELVEN_BOW,
    2: ObjType.ELVEN_ARROW, 3: ObjType.ELVEN_ARROW,
    4: ObjType.ELVEN_CLOAK, 5: ObjType.LEMBAS_WAFER,
}
_RANGER_RACE_SUBS_ORC = {
    0: ObjType.ORCISH_DAGGER, 1: ObjType.ORCISH_BOW,
    2: ObjType.ORCISH_ARROW, 3: ObjType.ORCISH_ARROW,
    5: ObjType.TRIPE_RATION,
}
_RANGER_RACE_SUBS_GNOME = {
    1: ObjType.CROSSBOW, 2: ObjType.CROSSBOW_BOLT, 3: ObjType.CROSSBOW_BOLT,
}


def _ini_inv_knight(vendor_rng, inv_state, items_list, race):
    """ini_inv(Knight) mksobj draws (u_init.c:73-83, 706).  Knight is human-
    only (no racial subs, no Xtra_food).  Items in trobj order: LONG_SWORD,
    LANCE (weapons), RING_MAIL, HELMET, SMALL_SHIELD, LEATHER_GLOVES (armor),
    APPLE x10, CARROT x10 (food).  All trbless=UNDEF_BLESS so mksobj's rolled
    blessed bit is kept; apple/carrot trbless=0 -> forced uncursed."""
    vendor_rng, b_ls = _mksobj_weapon_draws(vendor_rng, False, False)
    vendor_rng, b_lance = _mksobj_weapon_draws(vendor_rng, False, False)
    vendor_rng, b_rm = _mksobj_armor_draws(vendor_rng)
    vendor_rng, b_helm = _mksobj_armor_draws(vendor_rng)
    vendor_rng, b_ss = _mksobj_armor_draws(vendor_rng)
    vendor_rng, b_lg = _mksobj_armor_draws(vendor_rng)
    vendor_rng, apple_qty = _mksobj_food_stack_qty(vendor_rng, 10)
    vendor_rng, carrot_qty = _mksobj_food_stack_qty(vendor_rng, 10)

    items = inv_state.items
    _BLESSED = jnp.asarray(3, dtype=items.buc_status.dtype)
    _UNCURSED = jnp.asarray(2, dtype=items.buc_status.dtype)
    def _bset(b):
        return jnp.where(b, _BLESSED, _UNCURSED)
    buc = items.buc_status
    buc = buc.at[0].set(_bset(b_ls))
    buc = buc.at[1].set(_bset(b_lance))
    buc = buc.at[2].set(_bset(b_rm))
    buc = buc.at[3].set(_bset(b_helm))
    buc = buc.at[4].set(_bset(b_ss))
    buc = buc.at[5].set(_bset(b_lg))
    q = items.quantity
    q = q.at[6].set(apple_qty.astype(q.dtype))
    q = q.at[7].set(carrot_qty.astype(q.dtype))
    inv_state = inv_state.replace(items=items.replace(buc_status=buc, quantity=q))
    return vendor_rng, inv_state


def _ini_inv_ranger(vendor_rng, inv_state, items_list, race):
    """PM_RANGER (u_init.c:743-749) + orc Xtra_food (u_init.c:853).

    Pre-ini_inv role_init draws set the arrow-stack quantities:
        Ranger[2].trquan = rn1(10, 50);   // 50..59  (+2 arrows)
        Ranger[3].trquan = rn1(10, 30);   // 30..39  (+0 arrows)
    Then ini_inv(Ranger): DAGGER, BOW (weapons), ARROW x2 stacks
    (multigen+poisonable -> rn1(6,6) quan + rn2(100)), CLOAK_OF_DISPLACEMENT
    (armor), CRAM_RATION x4 (food).  First ammo stack -> uquiver (setuqwep).
    Racial equipment substitution + orc extra food applied last.
    """
    from Nethax.nethax.vendor_rng import rn2_jax
    from Nethax.nethax.constants.races import Race

    vendor_rng, a1 = rn2_jax(vendor_rng, jnp.int32(10))     # rn1(10,50)
    arrow1_qty = a1 + jnp.int32(50)
    vendor_rng, a2 = rn2_jax(vendor_rng, jnp.int32(10))     # rn1(10,30)
    arrow2_qty = a2 + jnp.int32(30)

    vendor_rng, b_dag = _mksobj_weapon_draws(vendor_rng, False, False)  # DAGGER
    vendor_rng, b_bow = _mksobj_weapon_draws(vendor_rng, False, False)  # BOW
    vendor_rng, b_ar1 = _mksobj_weapon_draws(vendor_rng, True, True)    # ARROW
    vendor_rng, b_ar2 = _mksobj_weapon_draws(vendor_rng, True, True)    # ARROW
    vendor_rng, b_cloak = _mksobj_armor_draws(vendor_rng)              # CLOAK
    vendor_rng, cram_qty = _mksobj_food_stack_qty(vendor_rng, 4)       # CRAM x4

    items = inv_state.items
    _BLESSED = jnp.asarray(3, dtype=items.buc_status.dtype)
    _UNCURSED = jnp.asarray(2, dtype=items.buc_status.dtype)
    def _bset(b):
        return jnp.where(b, _BLESSED, _UNCURSED)
    buc = items.buc_status
    buc = buc.at[0].set(_bset(b_dag))
    buc = buc.at[1].set(_bset(b_bow))
    buc = buc.at[2].set(_bset(b_ar1))
    buc = buc.at[3].set(_bset(b_ar2))
    buc = buc.at[4].set(_bset(b_cloak))
    q = items.quantity
    q = q.at[2].set(arrow1_qty.astype(q.dtype))
    q = q.at[3].set(arrow2_qty.astype(q.dtype))
    q = q.at[5].set(cram_qty.astype(q.dtype))
    items = items.replace(buc_status=buc, quantity=q)

    # Racial equipment substitution (static race -> otyp swap on fixed slots).
    subs = None
    if race == Race.ELF:
        subs = _RANGER_RACE_SUBS_ELF
    elif race == Race.ORC:
        subs = _RANGER_RACE_SUBS_ORC
    elif race == Race.GNOME:
        subs = _RANGER_RACE_SUBS_GNOME
    if subs is not None:
        tid = items.type_id
        for slot, new_otyp in subs.items():
            tid = tid.at[slot].set(jnp.asarray(int(new_otyp), dtype=tid.dtype))
        items = items.replace(type_id=tid)

    inv_state = inv_state.replace(items=items, quiver=jnp.int8(2))

    if race == Race.ORC:
        vendor_rng, inv_state = _orc_xtra_food(vendor_rng, inv_state)
    return vendor_rng, inv_state


_INI_INV_ROLE_DRAWS[Role.KNIGHT] = _ini_inv_knight
_INI_INV_ROLE_DRAWS[Role.RANGER] = _ini_inv_ranger


# ---------------------------------------------------------------------------
# Per-role ini_inv mkobj-draw ports — Barbarian / Valkyrie / Samurai.
#
# One contiguous block: shared vmap-safe mksobj draw helpers, then the three
# role functions, then the ``_INI_INV_ROLE_DRAWS`` registrations.  Every helper
# mirrors the conditional-advance style of ``_consume_ini_inv_archeologist_draws``
# / ``_consume_ini_inv_rogue_draws`` above: each vendor branch's ISAAC64 draw is
# computed unconditionally and adopted only when its predicate holds, so the
# byte stream advances by exactly the vendor draw count under both eager and
# vmap execution.  Draw sequences were confirmed against live NETHAX_RN2_TRACE
# captures of the vendor MiniHack reset (bar/val/sam, seed 0).
#
# Registry contract: ``fn(vendor_rng, inv_state, items_list) -> (vendor_rng,
# inv_state)``.  Runs at the same stream position vendor's ``ini_inv`` does
# (after items are built from the trobj table, before init_attr).
# ---------------------------------------------------------------------------

def _ini_inv_cadv(rng_orig, rng_advanced, take):
    """Adopt ``rng_advanced`` iff ``take`` (vmap-safe short-circuit)."""
    return jax.tree_util.tree_map(
        lambda a, o: jnp.where(take, a, o), rng_advanced, rng_orig,
    )


def _mksobj_weapon_class(vrng, is_multigen: bool, is_poisonable: bool):
    """Consume the ISAAC64 draws of ``mksobj(otyp, TRUE, FALSE)`` for a
    WEAPON_CLASS item and return ``(vrng, blessed_bool)``.

    Faithful port of vendor mkobj.c:803-814::

        otmp->quan = is_multigen(otmp) ? rn1(6, 6) : 1L;   // rn2(6) if multigen
        if (!rn2(11)) {          otmp->spe = rne(3); otmp->blessed = rn2(2); }
        else if (!rn2(10)) {     curse; otmp->spe = -rne(3); }
        else                     blessorcurse(otmp, 10);   // rn2(10)[+ rn2(2)]
        if (is_poisonable(otmp) && !rn2(100)) otmp->opoisoned = 1;   // rn2(100)

    ``artif`` is FALSE in ini_inv so the trailing ``rn2(20)`` never fires.  The
    ``blessed`` flag is what ini_inv keeps for trbless==UNDEF_BLESS items; the
    curse branch is wiped to uncursed by ini_inv (u_init.c:1094), so it maps to
    ``blessed=False``.
    Cite: vendor/nle/src/mkobj.c:803-814; vendor/nle/include/obj.h:197-204.
    """
    from Nethax.nethax.vendor_rng import rn2_jax, rne_jax

    # Multigen quantity draw (rn1(6,6) → rn2(6)); value discarded because
    # ini_inv overrides WEAPON_CLASS quan from trquan (u_init.c:1098).
    if is_multigen:
        vrng, _ = rn2_jax(vrng, jnp.int32(6))

    vrng, x1 = rn2_jax(vrng, jnp.int32(11))
    path_a = (x1 == jnp.int32(0))
    not_a = jnp.logical_not(path_a)

    # Path A: spe = rne(3); blessed = rn2(2).
    vr_a, _ = rne_jax(vrng, jnp.int32(3))
    vr_a, a_bless = rn2_jax(vr_a, jnp.int32(2))
    vr_a = _ini_inv_cadv(vrng, vr_a, path_a)

    # Not-A: x2 = rn2(10), drawn only when x1 != 0.
    vr_x2, x2_raw = rn2_jax(vrng, jnp.int32(10))
    vr_x2 = _ini_inv_cadv(vrng, vr_x2, not_a)
    x2 = jnp.where(not_a, x2_raw, jnp.int32(1))
    path_b = jnp.logical_and(not_a, x2 == jnp.int32(0))
    path_c = jnp.logical_and(not_a, x2 != jnp.int32(0))

    # Path B: spe = -rne(3) (curse → wiped to uncursed).
    vr_b, _ = rne_jax(vr_x2, jnp.int32(3))
    # Path C: blessorcurse(10) → rn2(10) [+ rn2(2) if 0].
    vr_c, bc1 = rn2_jax(vr_x2, jnp.int32(10))
    vr_c2, bc2 = rn2_jax(vr_c, jnp.int32(2))
    vr_c = _ini_inv_cadv(vr_c, vr_c2, bc1 == jnp.int32(0))
    c_bless = jnp.logical_and(bc1 == jnp.int32(0), bc2 != jnp.int32(0))

    vr_bc = _ini_inv_cadv(vr_x2, vr_b, path_b)
    vr_bc = _ini_inv_cadv(vr_bc, vr_c, path_c)
    vrng = _ini_inv_cadv(vr_bc, vr_a, path_a)

    blessed = jnp.logical_or(
        jnp.logical_and(path_a, a_bless != jnp.int32(0)),
        jnp.logical_and(path_c, c_bless),
    )

    # is_poisonable → rn2(100) (result discarded; opoisoned not tracked for the
    # starting kit, and ini_inv wipes opoisoned for non-chaotic anyway).
    if is_poisonable:
        vrng, _ = rn2_jax(vrng, jnp.int32(100))

    return vrng, blessed


def _mksobj_armor_class(vrng):
    """Consume the ISAAC64 draws of ``mksobj`` for an ARMOR_CLASS item that is
    NOT one of the special-cursed helms/boots/gauntlets, and return
    ``(vrng, blessed_bool)``.

    Faithful port of vendor mkobj.c:992-1004 (the special-item disjunct is
    inert for RING_MAIL / SMALL_SHIELD / SPLINT_MAIL)::

        if (rn2(10) && !rn2(11)) {   curse; otmp->spe = -rne(3); }
        else if (!rn2(10)) {         otmp->blessed = rn2(2); otmp->spe = rne(3); }
        else                         blessorcurse(otmp, 10);

    Short-circuit: X1=rn2(10) always; X2=rn2(11) only if X1!=0.
    Cite: vendor/nle/src/mkobj.c:992-1004.
    """
    from Nethax.nethax.vendor_rng import rn2_jax, rne_jax

    vrng, X1 = rn2_jax(vrng, jnp.int32(10))
    X1_nz = (X1 != jnp.int32(0))
    vr_x2, X2_raw = rn2_jax(vrng, jnp.int32(11))
    vrng = _ini_inv_cadv(vrng, vr_x2, X1_nz)
    X2 = jnp.where(X1_nz, X2_raw, jnp.int32(1))
    path_a = jnp.logical_and(X1_nz, X2 == jnp.int32(0))
    not_a = jnp.logical_not(path_a)

    # Path A: spe = -rne(3) (curse → wiped to uncursed).
    vr_a, _ = rne_jax(vrng, jnp.int32(3))
    vr_a = _ini_inv_cadv(vrng, vr_a, path_a)

    # Not-A: X3 = rn2(10).
    vr_x3, X3_raw = rn2_jax(vrng, jnp.int32(10))
    vr_x3 = _ini_inv_cadv(vrng, vr_x3, not_a)
    X3 = jnp.where(not_a, X3_raw, jnp.int32(1))
    path_b = jnp.logical_and(not_a, X3 == jnp.int32(0))
    path_c = jnp.logical_and(not_a, jnp.logical_not(path_b))

    # Path B: blessed = rn2(2); spe = rne(3).
    vr_b, b_bless = rn2_jax(vr_x3, jnp.int32(2))
    vr_b, _ = rne_jax(vr_b, jnp.int32(3))
    # Path C: blessorcurse(10) → rn2(10) [+ rn2(2) if 0].
    vr_c, bc1 = rn2_jax(vr_x3, jnp.int32(10))
    vr_c2, bc2 = rn2_jax(vr_c, jnp.int32(2))
    vr_c = _ini_inv_cadv(vr_c, vr_c2, bc1 == jnp.int32(0))

    vr_bc = _ini_inv_cadv(vr_x3, vr_b, path_b)
    vr_bc = _ini_inv_cadv(vr_bc, vr_c, path_c)
    vrng = _ini_inv_cadv(vr_bc, vr_a, path_a)

    blessed = jnp.logical_or(
        jnp.logical_and(path_b, b_bless != jnp.int32(0)),
        jnp.logical_and(path_c,
                        jnp.logical_and(bc1 == jnp.int32(0), bc2 != jnp.int32(0))),
    )
    return vrng, blessed


def _mksobj_food_ration(vrng):
    """Consume the single ``rn2(6)`` a FOOD_RATION mksobj call emits and return
    ``(vrng, quan)`` where ``quan`` is 2 iff ``!rn2(6)`` else 1.

    FOOD_RATION hits none of the CORPSE/EGG/TIN/SLIME/KELP inner cases and is
    not a pudding, so mkobj.c:880-883 (``!rn2(6) -> quan = 2``) is the only
    draw.  ini_inv does NOT override FOOD quan (only WEAPON/TOOL), so the
    rolled 1/2 is the stack size.  Cite: vendor/nle/src/mkobj.c:819-884.
    """
    from Nethax.nethax.vendor_rng import rn2_jax
    vrng, r = rn2_jax(vrng, jnp.int32(6))
    quan = jnp.where(r == jnp.int32(0), jnp.int32(2), jnp.int32(1))
    return vrng, quan


def _mksobj_oil_lamp(vrng):
    """Consume the ISAAC64 draws for ``mksobj(OIL_LAMP)`` (Lamp bonus item) and
    return ``(vrng, blessed_bool)``.

    Vendor mkobj.c:909-913: ``otmp->age = rn1(500, 1000)`` → rn2(500); then
    ``blessorcurse(otmp, 5)`` → rn2(5) [+ rn2(2) if 0].
    Cite: vendor/nle/src/mkobj.c:909-913.
    """
    from Nethax.nethax.vendor_rng import rn2_jax
    vrng, _ = rn2_jax(vrng, jnp.int32(500))       # rn1(500,1000) age
    vrng, bc1 = rn2_jax(vrng, jnp.int32(5))        # blessorcurse(5)
    vr_bonus, bc2 = rn2_jax(vrng, jnp.int32(2))
    vrng = _ini_inv_cadv(vrng, vr_bonus, bc1 == jnp.int32(0))
    blessed = jnp.logical_and(bc1 == jnp.int32(0), bc2 != jnp.int32(0))
    return vrng, blessed


def _add_bonus_item(inv_state, slot: int, take, item):
    """Conditionally place ``item`` (a scalar Item pytree) at inventory ``slot``
    with inventory letter ``chr(ord('a') + slot)``, gated on ``take``
    (vmap-safe).  When ``take`` is False the slot is left empty.

    Mirrors the Archeologist bonus-item placement: all prior slots are filled
    so the letter is simply ``'a' + slot``.  Cite: vendor/nle/src/u_init.c
    ini_inv appends optional items after the fixed trobj list.
    """
    from Nethax.nethax.subsystems.inventory import make_empty_item
    empty = make_empty_item()
    chosen = jax.tree_util.tree_map(
        lambda it, em: jnp.where(take, it, em), item, empty,
    )
    new_items = jax.tree_util.tree_map(
        lambda arr, val: arr.at[slot].set(val), inv_state.items, chosen,
    )
    letter = jnp.where(take, jnp.int8(ord('a') + slot), jnp.int8(0))
    new_letters = inv_state.letters.at[slot].set(letter)
    return inv_state.replace(items=new_items, letters=new_letters)


def _make_oil_lamp(blessed):
    """Build the OIL_LAMP bonus Item (type 202).  BUC from the rolled
    blessorcurse(5) (ini_inv keeps it — Lamp trbless is 0 only for the fixed
    quan, blessorcurse sets the bit).  Cite: vendor/nle/src/u_init.c:182."""
    from Nethax.nethax.subsystems.inventory import make_item
    buc = jnp.where(blessed, jnp.int32(_BUC_BLESSED), jnp.int32(_BUC_UNCURSED))
    return make_item(
        category=int(ItemCategory.TOOL), type_id=202, quantity=1, weight=20,
        buc_status=buc, identified=True, bknown=True, dknown=True, rknown=True,
    )


def _make_blindfold():
    """Build the BLINDFOLD bonus Item (type 208).  Blindfold trbless is 0 →
    always uncursed.  Cite: vendor/nle/src/u_init.c:184."""
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(
        category=int(ItemCategory.TOOL), type_id=208, quantity=1, weight=2,
        buc_status=_BUC_UNCURSED, identified=True,
        bknown=True, dknown=True, rknown=True,
    )


def _apply_buc_blessed(inv_state, slot: int, blessed):
    """Set inv slot ``slot`` buc_status to BLESSED iff ``blessed`` else UNCURSED
    (both bknown, matching ini_inv u_init.c:1088-1094)."""
    buc = inv_state.items.buc_status
    val = jnp.where(blessed,
                    jnp.asarray(_BUC_BLESSED, dtype=buc.dtype),
                    jnp.asarray(_BUC_UNCURSED, dtype=buc.dtype))
    new_items = inv_state.items.replace(buc_status=buc.at[slot].set(val))
    return inv_state.replace(items=new_items)


def _consume_ini_inv_barbarian_draws(vendor_rng, inv_state, items_list, race=None):
    """ini_inv(Barbarian) port — u_init.c:680-691.

    Sequence (confirmed by NETHAX_RN2_TRACE, seed 0)::

        rn2(100)                         # >= 50 -> BATTLE_AXE/SHORT_SWORD variant
        ini_inv(Barbarian):
          weapon0 (2H-sword | battle-axe)  WEAPON_CLASS  (melee: no multigen/poison)
          weapon1 (axe | short sword)      WEAPON_CLASS
          RING_MAIL                        ARMOR_CLASS
          FOOD_RATION x1                   FOOD_CLASS -> rn2(6)
        rn2(6)                           # !rn2(6) -> ini_inv(Lamp)

    Rewrites slots 0/1 to the rolled weapon variant (glyph 1934 = BATTLE_AXE
    on the >=50 branch), applies the per-item blessed bits, and the rolled
    FOOD_RATION quantity.  All trspe are fixed (0), so enchantment is unchanged.
    """
    from Nethax.nethax.vendor_rng import rn2_jax

    # rn2(100) variant-selection roll (BEFORE ini_inv).  u_init.c:681.
    vendor_rng, sel = rn2_jax(vendor_rng, jnp.int32(100))
    variant_hi = (sel >= jnp.int32(50))

    # mksobj draws for the four Barbarian items, in trobj order.  The weapon
    # draw sequence is identical for either variant (all four melee weapons are
    # non-multigen / non-poisonable), so the swap does not change draw counts.
    vendor_rng, w0_blessed = _mksobj_weapon_class(vendor_rng, False, False)
    vendor_rng, w1_blessed = _mksobj_weapon_class(vendor_rng, False, False)
    vendor_rng, rm_blessed = _mksobj_armor_class(vendor_rng)
    vendor_rng, food_qty = _mksobj_food_ration(vendor_rng)

    # rn2(6) Lamp gate + conditional OIL_LAMP mksobj draws (u_init.c:686-687).
    vendor_rng, lamp_gate = rn2_jax(vendor_rng, jnp.int32(6))
    take_lamp = (lamp_gate == jnp.int32(0))
    vr_lamp, lamp_blessed = _mksobj_oil_lamp(vendor_rng)
    vendor_rng = _ini_inv_cadv(vendor_rng, vr_lamp, take_lamp)

    # Rewrite slots 0/1 to the BATTLE_AXE/SHORT_SWORD variant when rolled.
    items = inv_state.items
    tid = items.type_id
    wt = items.weight
    tid = tid.at[0].set(jnp.where(
        variant_hi, jnp.asarray(ObjType.BATTLE_AXE, dtype=tid.dtype), tid[0]))
    tid = tid.at[1].set(jnp.where(
        variant_hi, jnp.asarray(ObjType.SHORT_SWORD, dtype=tid.dtype), tid[1]))
    wt = wt.at[0].set(jnp.where(
        variant_hi,
        jnp.asarray(_WEAPON_WEIGHT[ObjType.BATTLE_AXE], dtype=wt.dtype), wt[0]))
    wt = wt.at[1].set(jnp.where(
        variant_hi,
        jnp.asarray(_WEAPON_WEIGHT[ObjType.SHORT_SWORD], dtype=wt.dtype), wt[1]))
    # FOOD_RATION rolled quantity -> slot 3.
    quantity = items.quantity.at[3].set(food_qty.astype(items.quantity.dtype))
    inv_state = inv_state.replace(
        items=items.replace(type_id=tid, weight=wt, quantity=quantity))

    # Per-item blessed bits (trbless == UNDEF_BLESS for weapons + ring mail).
    inv_state = _apply_buc_blessed(inv_state, 0, w0_blessed)
    inv_state = _apply_buc_blessed(inv_state, 1, w1_blessed)
    inv_state = _apply_buc_blessed(inv_state, 2, rm_blessed)
    # Optional OIL_LAMP bonus (u_init.c:686-687) at the next slot (4).
    inv_state = _add_bonus_item(inv_state, 4, take_lamp,
                                _make_oil_lamp(lamp_blessed))
    return vendor_rng, inv_state


def _consume_ini_inv_valkyrie_draws(vendor_rng, inv_state, items_list, race=None):
    """ini_inv(Valkyrie) port — u_init.c:782-785.

    Sequence (confirmed by NETHAX_RN2_TRACE, seed 0)::

        ini_inv(Valkyrie):
          LONG_SWORD    WEAPON_CLASS
          DAGGER        WEAPON_CLASS   (daggers are NOT is_poisonable here)
          SMALL_SHIELD  ARMOR_CLASS
          FOOD_RATION x1 FOOD_CLASS -> rn2(6)
        rn2(6)          # !rn2(6) -> ini_inv(Lamp)

    Applies the per-item blessed bits and the rolled FOOD_RATION quantity.
    The LONG_SWORD spe=1 / SMALL_SHIELD spe=3 come from trspe (static table).
    """
    from Nethax.nethax.vendor_rng import rn2_jax

    vendor_rng, ls_blessed = _mksobj_weapon_class(vendor_rng, False, False)
    vendor_rng, dg_blessed = _mksobj_weapon_class(vendor_rng, False, False)
    vendor_rng, sh_blessed = _mksobj_armor_class(vendor_rng)
    vendor_rng, food_qty = _mksobj_food_ration(vendor_rng)

    vendor_rng, lamp_gate = rn2_jax(vendor_rng, jnp.int32(6))
    take_lamp = (lamp_gate == jnp.int32(0))
    vr_lamp, lamp_blessed = _mksobj_oil_lamp(vendor_rng)
    vendor_rng = _ini_inv_cadv(vendor_rng, vr_lamp, take_lamp)

    items = inv_state.items
    quantity = items.quantity.at[3].set(food_qty.astype(items.quantity.dtype))
    inv_state = inv_state.replace(items=items.replace(quantity=quantity))
    inv_state = _apply_buc_blessed(inv_state, 0, ls_blessed)
    inv_state = _apply_buc_blessed(inv_state, 1, dg_blessed)
    inv_state = _apply_buc_blessed(inv_state, 2, sh_blessed)
    # Optional OIL_LAMP bonus (u_init.c:784-785) at the next slot (4).
    inv_state = _add_bonus_item(inv_state, 4, take_lamp,
                                _make_oil_lamp(lamp_blessed))
    return vendor_rng, inv_state


def _consume_ini_inv_samurai_draws(vendor_rng, inv_state, items_list, race=None):
    """ini_inv(Samurai) port — u_init.c:759-763.

    Sequence (confirmed by NETHAX_RN2_TRACE, seed 0)::

        rn2(20)                         # YA quantity = rn1(20,26) = rn2(20)+26
        ini_inv(Samurai):
          KATANA        WEAPON_CLASS
          SHORT_SWORD   WEAPON_CLASS   (rendered "wakizashi" via inv_strs)
          YUMI          WEAPON_CLASS   (a bow: NOT multigen)
          YA            WEAPON_CLASS   (multigen + poisonable: rn2(6)+cascade+rn2(100))
          SPLINT_MAIL   ARMOR_CLASS
        rn2(5)          # !rn2(5) -> ini_inv(Blindfold)

    Applies the per-item blessed bits and the rolled YA stack quantity (slot 3).
    """
    from Nethax.nethax.vendor_rng import rn2_jax

    # YA quantity = rn1(20, 26) (BEFORE ini_inv).  u_init.c:760.
    vendor_rng, ya_roll = rn2_jax(vendor_rng, jnp.int32(20))
    ya_qty = ya_roll + jnp.int32(26)

    vendor_rng, k_blessed = _mksobj_weapon_class(vendor_rng, False, False)   # KATANA
    vendor_rng, w_blessed = _mksobj_weapon_class(vendor_rng, False, False)   # wakizashi
    vendor_rng, y_blessed = _mksobj_weapon_class(vendor_rng, False, False)   # YUMI (bow)
    vendor_rng, a_blessed = _mksobj_weapon_class(vendor_rng, True, True)     # YA (ammo)
    vendor_rng, sm_blessed = _mksobj_armor_class(vendor_rng)                 # SPLINT_MAIL

    # rn2(5) Blindfold gate (BLINDFOLD is a plain tool → no mksobj draws).
    vendor_rng, blind_gate = rn2_jax(vendor_rng, jnp.int32(5))
    take_blindfold = (blind_gate == jnp.int32(0))

    items = inv_state.items
    quantity = items.quantity.at[3].set(ya_qty.astype(items.quantity.dtype))
    # weight tracks quantity for ammo (per-unit YA weight = static/26).
    inv_state = inv_state.replace(items=items.replace(quantity=quantity))
    inv_state = _apply_buc_blessed(inv_state, 0, k_blessed)
    inv_state = _apply_buc_blessed(inv_state, 1, w_blessed)
    inv_state = _apply_buc_blessed(inv_state, 2, y_blessed)
    inv_state = _apply_buc_blessed(inv_state, 3, a_blessed)
    inv_state = _apply_buc_blessed(inv_state, 4, sm_blessed)

    # Lacquered armor for Samurai: SPLINT_MAIL (slot 4) is made erosion-proof
    # and its rustproofing is known at character start (moves <= 1), rendering
    # "rustproof".  Cite: vendor/nle/src/mkobj.c:1007-1013.
    its = inv_state.items
    inv_state = inv_state.replace(items=its.replace(
        oerodeproof=its.oerodeproof.at[4].set(jnp.bool_(True)),
        rknown=its.rknown.at[4].set(jnp.bool_(True)),
    ))
    # Optional BLINDFOLD bonus (u_init.c:762-763) at the next slot (5).
    inv_state = _add_bonus_item(inv_state, 5, take_blindfold, _make_blindfold())
    return vendor_rng, inv_state


_INI_INV_ROLE_DRAWS[Role.BARBARIAN] = _consume_ini_inv_barbarian_draws
_INI_INV_ROLE_DRAWS[Role.VALKYRIE] = _consume_ini_inv_valkyrie_draws
_INI_INV_ROLE_DRAWS[Role.SAMURAI] = _consume_ini_inv_samurai_draws


# Race-based initial-inventory substitutions — vendor u_init.c:203-232 inv_subs[].
# ini_inv swaps the object type for non-human races (mkobj.c:1057-1071) as a
# pure otyp change with NO extra ISAAC64 draws.  {Race: [(from_otyp, to_otyp)]};
# otyp values are the canonical vendor object ids (see constants/objects.py).
# Applied (via _apply_race_inv_subs) only for the roles this module ports below;
# it never touches roles it does not special-case.
_INV_SUBS_BY_RACE = {
    Race.ELF: [
        (17, 18),    # DAGGER -> ELVEN_DAGGER
        (10, 11),    # SPEAR -> ELVEN_SPEAR
        (29, 30),    # SHORT_SWORD -> ELVEN_SHORT_SWORD
        (65, 66),    # BOW -> ELVEN_BOW
        (1, 2),      # ARROW -> ELVEN_ARROW
        (78, 71),    # HELMET -> ELVEN_LEATHER_HELM
        (128, 118),  # CLOAK_OF_DISPLACEMENT -> ELVEN_CLOAK
        (267, 266),  # CRAM_RATION -> LEMBAS_WAFER
    ],
    Race.ORC: [
        (17, 19),    # DAGGER -> ORCISH_DAGGER
        (10, 12),    # SPEAR -> ORCISH_SPEAR
        (29, 31),    # SHORT_SWORD -> ORCISH_SHORT_SWORD
        (65, 67),    # BOW -> ORCISH_BOW
        (1, 3),      # ARROW -> ORCISH_ARROW
        (78, 72),    # HELMET -> ORCISH_HELM
        (129, 132),  # SMALL_SHIELD -> ORCISH_SHIELD
        (111, 112),  # RING_MAIL -> ORCISH_RING_MAIL
        (107, 108),  # CHAIN_MAIL -> ORCISH_CHAIN_MAIL
        (267, 239),  # CRAM_RATION -> TRIPE_RATION
        (266, 239),  # LEMBAS_WAFER -> TRIPE_RATION
    ],
    Race.DWARF: [
        (10, 13),    # SPEAR -> DWARVISH_SPEAR
        (29, 32),    # SHORT_SWORD -> DWARVISH_SHORT_SWORD
        (78, 73),    # HELMET -> DWARVISH_IRON_HELM
        (266, 267),  # LEMBAS_WAFER -> CRAM_RATION
    ],
    Race.GNOME: [
        (65, 70),    # BOW -> CROSSBOW
        (1, 6),      # ARROW -> CROSSBOW_BOLT
    ],
}


def _apply_race_inv_subs(inv_state, race):
    """Apply vendor's race-based initial-inventory otyp substitutions to the
    already-built inventory (vendor u_init.c inv_subs[] / mkobj.c:1057-1071).

    Pure otyp swap — no ISAAC64 draws — so it runs after the mksobj-draw port.
    ``race`` is a concrete Race enum (create_character's Python arg), so the
    per-race sub list is chosen at trace-build time; the per-slot swap is a
    vmap-safe ``jnp.where`` over the ``type_id`` array (glyph + rendered name
    both derive from type_id).  Human (no entry) is a no-op.
    """
    subs = _INV_SUBS_BY_RACE.get(race)
    if not subs:
        return inv_state
    tid = inv_state.items.type_id
    for frm, to in subs:
        tid = jnp.where(tid == jnp.asarray(frm, dtype=tid.dtype),
                        jnp.asarray(to, dtype=tid.dtype), tid)
    return inv_state.replace(items=inv_state.items.replace(type_id=tid))


# ---------------------------------------------------------------------------
# _consume_attr_variation_draws  — vendor u_init.c:887-894 post-init_attr loop
# ---------------------------------------------------------------------------

def _consume_attr_variation_draws(vendor_rng, stats, role: Role, race: Race):
    """Consume the 6 ``rn2(20)`` draws (+ conditional ``rn2(7)``) emitted by
    the post-``init_attr`` attribute variation loop *and apply* the biased
    ``adjattrib`` variation to the stat values.

    Vendor code (u_init.c:886-894)::

        for (i = 0; i < A_MAX; i++)   // A_MAX == 6
            if (!rn2(20)) {
                register int xd = rn2(7) - 2;   // biased variation
                (void) adjattrib(i, xd, TRUE);
                if (ABASE(i) < AMAX(i))
                    AMAX(i) = ABASE(i);
            }

    Always fires exactly 6 ``rn2(20)`` draws; each zero result adds one
    ``rn2(7)`` draw, then calls ``adjattrib(i, xd, TRUE)`` which mutates the
    attribute.  Earlier versions of this helper consumed the draws but
    discarded ``xd`` — that dropped the +1 CON bump on seed 0 Rogue/Human
    (NLE CON 14 vs Nethax 13).

    ``adjattrib(ndx, incr, TRUE)`` (attrib.c:114-188):
      * ``incr == 0`` → no-op, returns FALSE (no extra draw).
      * ``incr > 0`` → ABASE += incr, clamped up to ATTRMAX(ndx).  No draw.
      * ``incr < 0`` → ABASE += incr; only if it falls *below* ATTRMIN does
        vendor draw ``rn2(ATTRMIN - ABASE + 1)`` to reduce AMAX — at character
        init the floors are 7+ so a single biased ``xd`` (range -2..+4) never
        sends a stat below its race ATTRMIN(=3), so this extra draw never
        fires here.  We replicate it defensively so the byte stream stays
        exact if a future role/seed lands a deep negative variation.
      * the DUNCE_CAP / Fixed_abil gates are inactive at character creation.

    Parameters
    ----------
    vendor_rng : Isaac64State
    stats      : ``{stat_name: int}`` post-init_attr values (mutated copy
                 returned, never in place).
    role, race : for ATTRMIN/ATTRMAX clamps.

    Returns ``(vendor_rng, stats)``.

    Citation: vendor/nle/src/u_init.c:886-894; vendor/nle/src/attrib.c:114-188.
    """
    from Nethax.nethax.vendor_rng import rn2_jax

    race_entry = get_race(race)
    attrmin_arr = jnp.asarray([int(v) for v in race_entry.attrmin], dtype=jnp.int32)
    attrmax_arr = jnp.asarray([int(v) for v in race_entry.attrmax], dtype=jnp.int32)

    # Working stats array in canonical _STAT_NAMES order (str,int,wis,dex,con,cha).
    # Convert input dict (Python ints or jnp scalars) to a single int32[6] array.
    vals = jnp.stack([jnp.asarray(stats[name], dtype=jnp.int32) for name in _STAT_NAMES])

    # Vendor u_init.c:886-894 — fixed 6 iterations.  i is static, so we unroll.
    # Each iter: draw rn2(20); if 0, draw rn2(7)-2 = xd ∈ [-2,+4]; if xd != 0,
    # apply adjattrib(i, xd, TRUE) which mutates vals[i] (clamp up to attrmax
    # on positive; on deeply-negative, draw rn2(attrmin-new+1) and clamp to
    # attrmin — never reached at init for current targets but kept for parity).
    #
    # vmap-compat byte-parity pattern: compute the candidate next-state via
    # rn2_jax, then conditionally adopt it via jnp.where(gate, advanced, prev).
    # ISAAC64 state is functional, so the "discarded" branch leaves the stream
    # unchanged — matching vendor's short-circuit consumption.  This fixes the
    # 12-extra-draw drift the old Brax-flatten pattern produced (it always
    # advanced past rn2(7)/rn2(modulus) even on gate-miss iterations).
    def _conditional_advance(rng_orig, rng_advanced, take):
        """Adopt rng_advanced iff ``take``, else keep rng_orig."""
        return jax.tree_util.tree_map(
            lambda a, o: jnp.where(take, a, o), rng_advanced, rng_orig,
        )

    for i in range(6):
        attrmin_i = attrmin_arr[i]
        attrmax_i = attrmax_arr[i]

        vendor_rng, gate = rn2_jax(vendor_rng, jnp.int32(20))    # u_init.c:888
        hit = gate == jnp.int32(0)

        # rn2(7): vendor draws this only when gate==0.  Compute candidate
        # advance, conditionally adopt.
        rng_after_xd, xd_raw = rn2_jax(vendor_rng, jnp.int32(7))   # u_init.c:889
        vendor_rng = _conditional_advance(vendor_rng, rng_after_xd, hit)
        xd = xd_raw - jnp.int32(2)

        base = vals[i]
        tentative = base + xd

        # _neg_path inner gate: vendor's `attrib.c:131-138` draws an extra
        # rn2(attrmin - new + 1) only on deeply-negative incr.  Same pattern:
        # compute candidate, conditionally adopt under (hit & need_extra).
        need_extra = jnp.logical_and(hit, tentative < attrmin_i)
        modulus_raw = (attrmin_i - tentative + jnp.int32(1)).astype(jnp.int32)
        modulus = jnp.maximum(modulus_raw, jnp.int32(1))
        rng_after_neg, _ = rn2_jax(vendor_rng, modulus)
        vendor_rng = _conditional_advance(vendor_rng, rng_after_neg, need_extra)

        # Apply adjattrib effect only when hit.  On hit & need_extra, base
        # clamps to attrmin; on hit & xd > 0, clamps to attrmax; on hit &
        # xd == 0, no change.
        new_base_pos = jnp.minimum(tentative, attrmax_i)
        # Selection (only matters when hit):
        #   xd == 0   → base unchanged
        #   xd > 0    → min(base+xd, attrmax)
        #   xd < 0    → if need_extra then attrmin else (base+xd)
        is_zero = xd == jnp.int32(0)
        is_pos = xd > jnp.int32(0)
        new_b_neg = jnp.where(need_extra, attrmin_i, tentative)
        new_b_nz = jnp.where(is_pos, new_base_pos, new_b_neg)
        new_base_hit = jnp.where(is_zero, base, new_b_nz)
        new_base = jnp.where(hit, new_base_hit, base)
        vals = vals.at[i].set(new_base)

    out = {name: vals[idx] for idx, name in enumerate(_STAT_NAMES)}
    return vendor_rng, out


# ---------------------------------------------------------------------------
# create_character
# ---------------------------------------------------------------------------

def create_character(rng: jax.Array, role: Role, race: Race, alignment: int, vendor_rng=None):
    """Build an EnvState for a newly created character.

    Rolls attribute scores via the vendor ``init_attr(75)`` formula,
    computes starting HP/Pw via ``ini_hpwp`` (newhp / newpw at u.ulevel==0),
    populates starting inventory, wields primary weapon, and wears starting
    armor.  Also initialises the player's alignment record from
    ``role.initrecord`` (priest = 5, etc.) and zeroes luck (vendor default
    unless special birth options).

    Parameters
    ----------
    rng        : JAX PRNG key
    role       : Role enum value
    race       : Race enum value
    alignment  : 0=lawful, 1=neutral, 2=chaotic (int)
    vendor_rng : Isaac64State | None
        When provided (NLE_BYTEPARITY mode), consume vendor ISAAC64 draws for
        u_init RNG gating (e.g. the rn2(5) BLINDFOLD roll at
        vendor/nle/src/u_init.c:753).  When None (default NLE / Threefry
        mode), fall back to the static role-kit logic.

    Returns
    -------
    Keyword-argument dict suitable for EnvState(**kwargs) or state.replace(**kwargs).
    We return a dict rather than an EnvState to avoid a circular import — the
    caller (env.py) merges these fields into the state returned by EnvState.default().
    When vendor_rng is provided, the dict also includes a ``vendor_rng`` key
    carrying the post-draw Isaac64State so the caller can thread it forward.

    Citations
    ---------
    vendor/nle/src/attrib.c::init_attr             — stat distribution
    vendor/nle/src/attrib.c::newhp                 — HP at u.ulevel == 0
    vendor/nle/src/exper.c::newpw                  — Pw at u.ulevel == 0
    vendor/nle/src/u_init.c::u_init                — initrecord copy to
                                                     u.ualign.record
    vendor/nle/src/role.c::roles[]                 — role.attrbase/attrdist/
                                                     hpadv/enadv/initrecord
    vendor/nle/src/role.c::races[]                 — race.attrmin/attrmax/
                                                     hpadv/enadv
    """
    rng_stats, rng_hp = jax.random.split(rng, 2)

    # --- NLE_BYTEPARITY: pre-init_attr ISAAC64 draws for Rogue ---
    # Vendor order (u_init.c:749-756, then u_init.c:882-894):
    #   1. rn1(10,6)  dagger quantity          u_init.c:750
    #   2. ini_inv(Rogue) blessorcurse draws   mkobj.c:803-1004
    #   3. rn2(5)  BLINDFOLD gate              u_init.c:753
    #   4. init_attr(75) draws                 attrib.c:614-660
    #   5. attr variation loop (A_MAX=6)       u_init.c:887-894
    from Nethax.nethax.parity_mode import is_nle_mode as _is_nle
    blindfold_roll = None
    dagger_qty = None
    rogue_buc_flags = None
    if vendor_rng is not None and role == Role.ROGUE:
        # Step 1+2: dagger qty + ini_inv(Rogue) blessorcurse + per-item BUC.
        # Cite: vendor/nle/src/u_init.c:750; mkobj.c:803-1004; u_init.c:113.
        vendor_rng, dagger_qty, rogue_buc_flags = _consume_ini_inv_rogue_draws(vendor_rng)
        # Step 3: BLINDFOLD gate — u_init.c:753
        from Nethax.nethax.vendor_rng import rn2_jax as _rn2_jax
        vendor_rng, blindfold_roll = _rn2_jax(vendor_rng, jnp.int32(5))

    # --- Build inventory ---
    # NLE_BYTEPARITY: vendor u_init.c:668-894 runs `ini_inv(<role>)` BEFORE
    # init_attr / attr_variation.  We build the inventory + consume Arc
    # ini_inv draws + Arc bonus cascade here so subsequent ISAAC64 draws
    # (init_attr at u_init.c:883, attr_variation at u_init.c:887-894) land
    # at vendor-aligned offsets.  Rogue ini_inv draws were already consumed
    # above (dagger_qty / rogue_buc_flags / blindfold_roll); this block only
    # applies their effects to items_list.
    # Cite: vendor/nle/src/u_init.c:668-679 (Archeologist case),
    #       vendor/nle/src/u_init.c:749-757 (Rogue case),
    #       vendor/nle/src/u_init.c:882-894 (post-ini_inv init_attr+vary).
    items_list = STARTING_INVENTORY[role]
    # Rogue DAGGER stack quantity is vendor rn1(10, 6) (u_init.c:750), not the
    # static placeholder 10.  In NLE_BYTEPARITY mode the rn2(10) was already
    # consumed by _consume_ini_inv_rogue_draws above (no new draw); we now use
    # its value (6 + rn2(10)) to rebuild the DAGGER item.  Byte-stream unchanged.
    # Cite: vendor/nle/src/u_init.c:750; rnd.c::rn1.
    if dagger_qty is not None and role == Role.ROGUE:
        # Static slot index — Rogue inventory order is fixed (see
        # STARTING_INVENTORY[Role.ROGUE] above): slot 1 = DAGGER.
        # Using slot index instead of `int(it.type_id) == DAGGER` lookup
        # avoids extracting `int(dagger_qty)` from a (potentially traced)
        # JAX scalar — required for vmap-safety of create_character.
        items_list = list(items_list)
        items_list[1] = _weapon(ObjType.DAGGER, dagger_qty)
    # Apply per-item BUC overrides from vendor's ini_inv mksobj cascade.
    # Only SHORT_SWORD + LEATHER_ARMOR keep mksobj's rolled blessed bit;
    # DAGGER + POT_SICKNESS are forced UNCURSED via trbless=0.  ini_inv
    # also sets bknown=1 on every starting item so inv_strs renders the
    # BUC prefix per buc_status.  Byte-stream unchanged — the rolls were
    # already consumed in _consume_ini_inv_rogue_draws.
    # Cite: vendor/nle/src/u_init.c:113-128.
    if rogue_buc_flags is not None and role == Role.ROGUE:
        # Static slot indexing (see DAGGER comment above):
        #   slot 0 = SHORT_SWORD, 1 = DAGGER, 2 = LEATHER_ARMOR, 3 = POT_SICKNESS
        # rogue_buc_flags[0..3] index in the same order.  Avoids
        # `int(rogue_buc_flags[i])` extractions on traced JAX scalars —
        # required for vmap-safety of create_character.
        items_list = list(items_list)
        items_list[0] = items_list[0].replace(buc_status=rogue_buc_flags[0].astype(jnp.int8))
        items_list[1] = items_list[1].replace(buc_status=rogue_buc_flags[1].astype(jnp.int8))
        items_list[2] = items_list[2].replace(buc_status=rogue_buc_flags[2].astype(jnp.int8))
        items_list[3] = items_list[3].replace(buc_status=rogue_buc_flags[3].astype(jnp.int8))
    # Rogue's BLINDFOLD slot is gated `!rn2(5)` in vendor u_init.c:753-754:
    #     Rogue[R_DAGGERS].trquan = rn1(10, 6);
    #     ini_inv(Rogue);
    #     if (!rn2(5))
    #         ini_inv(Blindfold);
    # In NLE_BYTEPARITY mode the rn2(5) draw was consumed above (blindfold_roll).
    # In plain NLE / Threefry mode we strip the BLINDFOLD statically because
    # seed-0 deterministic runs (`env.seed(0, 0, reseed=False)`) always fail
    # this roll (validator confirms inv_oclasses[6]=18 == MAXOCLASSES sentinel).
    # Cite: vendor/nle/src/u_init.c:753-754.
    if blindfold_roll is not None and role == Role.ROGUE:
        # NLE_BYTEPARITY (vmap-safe):
        # vendor u_init.c:753-754: BLINDFOLD added iff !rn2(5) — i.e.
        # blindfold_roll == 0.  Slot 6 in Rogue's STARTING_INVENTORY is
        # the BLINDFOLD; swap it for make_empty_item() when the roll
        # fails.  We KEEP the slot (constant list length) and let the
        # downstream `_items_from_list` pad with empties — same byte
        # output as removing the slot entirely, but no Python `if` on
        # a traced scalar (required for vmap of create_character).
        from Nethax.nethax.subsystems.inventory import make_empty_item
        items_list = list(items_list)
        keep_blindfold = jnp.equal(blindfold_roll, 0)
        bf = items_list[6]
        empty = make_empty_item()
        items_list[6] = jax.tree_util.tree_map(
            lambda b, e: jnp.where(keep_blindfold, b, e), bf, empty,
        )
    elif _is_nle() and role == Role.ROGUE:
        # Static fallback (plain NLE / Threefry): strip BLINDFOLD.
        # Cite: vendor/nle/src/u_init.c:753-754.
        # Use the same slot-swap pattern so the downstream shape is
        # constant whether or not vendor_rng is in play.
        from Nethax.nethax.subsystems.inventory import make_empty_item
        items_list = list(items_list)
        items_list[6] = make_empty_item()
    inv_state  = InventoryState.from_items(items_list)

    # --- NLE_BYTEPARITY: Archeologist ini_inv per-item mksobj draws ---
    # Consume the 18 ISAAC64 draws emitted by mksobj for the 8 Arc starting
    # items (BULLWHIP=4, LEATHER_JACKET=4, FEDORA=4, FOOD_RATION×3=3,
    # PICK_AXE=0, TINNING_KIT=1[rn1(70,30)], TOUCHSTONE=1, SACK=1) and
    # apply the rn1(70,30) result to inventory slot 5 (TINNING_KIT) spe.
    # Cite: vendor/nle/src/u_init.c:42-53; mkobj.c:803-1004, 934, 309.
    if vendor_rng is not None and role == Role.ARCHEOLOGIST:
        vendor_rng, inv_state = _consume_ini_inv_archeologist_draws(
            vendor_rng, inv_state,
        )

    # --- NLE_BYTEPARITY: Archeologist bonus item cascade ---
    # Vendor u_init.c:670-675 — after ``ini_inv(Archeologist)``, the Arc role
    # rolls a 3-step rn2 cascade for one optional bonus item, BEFORE init_attr:
    #     if (!rn2(10))      ini_inv(Tinopener);     // TIN_OPENER  (214)
    #     else if (!rn2(4))  ini_inv(Lamp);          // OIL_LAMP    (202)
    #     else if (!rn2(10)) ini_inv(Magicmarker);   // MAGIC_MARKER(217)
    # We always draw all three ISAAC64 words so the stream advances
    # deterministically (vmap-safe); the short-circuit selection picks at
    # most one bonus item and writes it to inventory slot 8.
    # Cite: vendor/nle/src/u_init.c:670-675.
    if vendor_rng is not None and role == Role.ARCHEOLOGIST:
        from Nethax.nethax.vendor_rng import rn2_jax as _rn2_jax_arc
        from Nethax.nethax.subsystems.inventory import (
            ItemCategory as _ItemCategory_arc,
            make_item as _make_item_arc,
            make_empty_item as _make_empty_arc,
        )
        # Vendor's cascade is short-circuit: rn2(4) is drawn only if rn2(10)
        # picked nothing; rn2(10) again only if neither prior fired.  Use the
        # conditional-advance pattern to match vendor's draw count exactly.
        # Then for whichever bonus item was picked, replay the item's mksobj
        # draws: TIN_OPENER = 0 draws (no matching TOOL_CLASS case);
        # OIL_LAMP = rn1(500,1000)→rn2(500) + blessorcurse(5)→rn2(5)[+rn2(2)];
        # MAGIC_MARKER = rn1(70,30)→rn2(70) (spe, kept since NLE trspe is
        # UNDEF_SPE).  Cite: vendor/nle/src/mkobj.c:897-935; u_init.c:668-679.
        vendor_rng, _arc_r10a = _rn2_jax_arc(vendor_rng, jnp.int32(10))
        _arc_pick_tin = jnp.equal(_arc_r10a, jnp.int32(0))
        _not_tin = jnp.logical_not(_arc_pick_tin)
        _vrng_r4, _arc_r4_raw = _rn2_jax_arc(vendor_rng, jnp.int32(4))
        vendor_rng = jax.tree_util.tree_map(
            lambda new, old: jnp.where(_not_tin, new, old),
            _vrng_r4, vendor_rng,
        )
        _arc_r4 = jnp.where(_not_tin, _arc_r4_raw, jnp.int32(1))
        _arc_pick_oil = jnp.logical_and(_not_tin, jnp.equal(_arc_r4, jnp.int32(0)))
        _not_oil_path = jnp.logical_and(_not_tin, jnp.logical_not(jnp.equal(_arc_r4, jnp.int32(0))))
        _vrng_r10b, _arc_r10b_raw = _rn2_jax_arc(vendor_rng, jnp.int32(10))
        vendor_rng = jax.tree_util.tree_map(
            lambda new, old: jnp.where(_not_oil_path, new, old),
            _vrng_r10b, vendor_rng,
        )
        _arc_r10b = jnp.where(_not_oil_path, _arc_r10b_raw, jnp.int32(1))
        _arc_pick_mm_draw = jnp.logical_and(
            _not_oil_path, jnp.equal(_arc_r10b, jnp.int32(0)),
        )
        # OIL_LAMP mksobj (vendor mkobj.c:909-914): otmp->age = rn1(500,1000)
        # → rn2(500); then blessorcurse(otmp, 5) → rn2(5) [+ rn2(2) if 0].
        # Committed only when OIL_LAMP is the picked bonus item.
        _vrng_oil1, _ = _rn2_jax_arc(vendor_rng, jnp.int32(500))
        vendor_rng = jax.tree_util.tree_map(
            lambda new, old: jnp.where(_arc_pick_oil, new, old),
            _vrng_oil1, vendor_rng,
        )
        _vrng_oil2, _arc_oil_bc = _rn2_jax_arc(vendor_rng, jnp.int32(5))
        vendor_rng = jax.tree_util.tree_map(
            lambda new, old: jnp.where(_arc_pick_oil, new, old),
            _vrng_oil2, vendor_rng,
        )
        # blessorcurse bonus rn2(2): only when OIL_LAMP picked AND rn2(5)==0.
        _vrng_oil3, _ = _rn2_jax_arc(vendor_rng, jnp.int32(2))
        _arc_oil_bonus = jnp.logical_and(
            _arc_pick_oil, jnp.equal(_arc_oil_bc, jnp.int32(0)),
        )
        vendor_rng = jax.tree_util.tree_map(
            lambda new, old: jnp.where(_arc_oil_bonus, new, old),
            _vrng_oil3, vendor_rng,
        )
        # MAGIC_MARKER mksobj (vendor mkobj.c:932-934): otmp->spe = rn1(70,30)
        # → rn2(70)+30.  NLE Magicmarker trspe == UNDEF_SPE (u_init.c:179), so
        # ini_inv does NOT overwrite spe (u_init.c:1104) — the rolled value is
        # kept and rendered as the "(0:spe)" charge suffix (objnam.c).  No
        # blessorcurse for MAGIC_MARKER.  Committed only when MM is picked.
        _vrng_mm, _arc_mm_spe_raw = _rn2_jax_arc(vendor_rng, jnp.int32(70))
        vendor_rng = jax.tree_util.tree_map(
            lambda new, old: jnp.where(_arc_pick_mm_draw, new, old),
            _vrng_mm, vendor_rng,
        )
        _arc_mm_spe = _arc_mm_spe_raw + jnp.int32(30)   # rn1(70, 30)
        _arc_tin_opener = _make_item_arc(
            category=int(_ItemCategory_arc.TOOL), type_id=214, quantity=1,
            weight=30, buc_status=2, identified=True,  # 2 = UNCURSED
            bknown=True, dknown=True, rknown=True,
        )
        _arc_oil_lamp = _make_item_arc(
            category=int(_ItemCategory_arc.TOOL), type_id=202, quantity=1,
            weight=30, buc_status=2, identified=True,  # 2 = UNCURSED
            bknown=True, dknown=True, rknown=True,
        )
        _arc_magic_marker = _make_item_arc(
            category=int(_ItemCategory_arc.TOOL), type_id=217, quantity=1,
            weight=30, buc_status=2, identified=True,  # 2 = UNCURSED
            bknown=True, dknown=True, rknown=True,
        )
        _arc_empty = _make_empty_arc()
        _arc_pick_tin = jnp.equal(_arc_r10a, 0)
        _arc_pick_oil = jnp.logical_and(
            jnp.logical_not(_arc_pick_tin), jnp.equal(_arc_r4, 0),
        )
        _arc_pick_mm = jnp.logical_and(
            jnp.logical_and(
                jnp.logical_not(_arc_pick_tin),
                jnp.logical_not(_arc_pick_oil),
            ),
            jnp.equal(_arc_r10b, 0),
        )
        _arc_picked = jax.tree_util.tree_map(
            lambda t, o, m, e: jnp.where(
                _arc_pick_tin, t,
                jnp.where(_arc_pick_oil, o,
                          jnp.where(_arc_pick_mm, m, e)),
            ),
            _arc_tin_opener, _arc_oil_lamp, _arc_magic_marker, _arc_empty,
        )
        _arc_any_pick = jnp.logical_or(
            jnp.logical_or(_arc_pick_tin, _arc_pick_oil), _arc_pick_mm,
        )
        _arc_new_items = jax.tree_util.tree_map(
            lambda arr, val: arr.at[8].set(val), inv_state.items, _arc_picked,
        )
        # MAGIC_MARKER "(0:spe)" charge suffix (objnam.c): write the rolled
        # rn1(70,30) spe into slot-8 ``charges`` (and mirror to ``enchantment``
        # for view consistency, as done for TINNING_KIT) only when the marker
        # was the picked bonus item; otherwise leave the item's defaults.
        _arc_slot8_charges = jnp.where(
            _arc_pick_mm,
            _arc_mm_spe.astype(_arc_new_items.charges.dtype),
            _arc_new_items.charges[8],
        )
        _arc_slot8_enchant = jnp.where(
            _arc_pick_mm,
            _arc_mm_spe.astype(_arc_new_items.enchantment.dtype),
            _arc_new_items.enchantment[8],
        )
        _arc_new_items = _arc_new_items.replace(
            charges=_arc_new_items.charges.at[8].set(_arc_slot8_charges),
            enchantment=_arc_new_items.enchantment.at[8].set(_arc_slot8_enchant),
        )
        _arc_new_letter = jnp.where(
            _arc_any_pick, jnp.int8(ord('a') + 8), jnp.int8(0),
        )
        _arc_new_letters = inv_state.letters.at[8].set(_arc_new_letter)
        inv_state = inv_state.replace(
            items=_arc_new_items, letters=_arc_new_letters,
        )

    # --- NLE_BYTEPARITY: per-role ini_inv mkobj draws (non-Archeologist) ---
    # Runs at the SAME stream position vendor's ini_inv does (after the items
    # are built from the trobj table, before init_attr).  Archeologist uses the
    # dedicated inline block above; all other roles dispatch through the
    # registry.  A role absent from the registry consumes no draws (its
    # inventory content / downstream stream is not yet modelled).
    if vendor_rng is not None and role != Role.ARCHEOLOGIST:
        _ini_inv_fn = _INI_INV_ROLE_DRAWS.get(role)
        if _ini_inv_fn is not None:
            vendor_rng, inv_state = _ini_inv_fn(
                vendor_rng, inv_state, items_list, race,
            )
            # Vendor race-based otyp substitution (u_init.c inv_subs[]) —
            # a pure post-mksobj type swap for non-human races.  Scoped to
            # roles that define _apply_race_inv_subs subs (arc/rog untouched).
            inv_state = _apply_race_inv_subs(inv_state, race)

    # --- Stat rolls (vendor init_attr(75) parity) ---
    # NLE_BYTEPARITY: vendor runs init_attr AFTER ini_inv + bonus cascade.
    # Cite: vendor/nle/src/attrib.c:614-660; u_init.c:883.
    if vendor_rng is not None:
        vendor_rng, stats_raw = consume_init_attr_draws(vendor_rng, role, race)
        stats = {k: jnp.int32(v) for k, v in stats_raw.items()}
    else:
        stats = _init_attr_vendor(rng_stats, role, race, np_total=75)

    # --- NLE_BYTEPARITY: attr variation loop (all roles) ---
    # 6 rn2(20) draws + conditional rn2(7) per hit, applying the biased
    # adjattrib() variation to the stat values.  Cite:
    # vendor/nle/src/u_init.c:886-894.
    if vendor_rng is not None:
        vendor_rng, stats_raw = _consume_attr_variation_draws(
            vendor_rng, stats_raw, role, race,
        )
        stats = {k: jnp.int32(v) for k, v in stats_raw.items()}

    # --- HP / Pw (vendor ini_hpwp parity, u.ulevel == 0 branch) ---
    hp, pw = _ini_hpwp_vendor(rng_hp, role, race)

    # --- Wield primary weapon ---
    wielded = jnp.int8(-1)
    if role not in _NO_WEAPON_ROLES and len(items_list) > 0:
        # Wield slot 0 (primary weapon is always first in the list)
        first_cat = int(items_list[0].category)
        if first_cat == int(ItemCategory.WEAPON):
            wielded = jnp.int8(0)
    inv_state = inv_state.replace(wielded=wielded)

    # --- Set swap weapon (uswapwep) for Rogue ---
    # Vendor u_init.c:R_DAGGERS (line 123): Rogue's daggers live at slot 1
    # and are set as uswapwep via setuswapwep() after ini_inv.
    # doname (objnam.c:1613-1620) emits " (alternate weapon%s; not wielded)"
    # when obj->owornmask & W_SWAPWEP and !u.twoweap.
    # Cite: vendor/nethack/src/u_init.c:123, vendor/nethack/src/objnam.c:1619.
    if role == Role.ROGUE:
        inv_state = inv_state.replace(swap_weapon=jnp.int8(1))
    # Archeologist: BULLWHIP is wielded (slot 0), PICK_AXE is is_weptool()
    # and falls into the !uswapwep branch of u_init.c:1144 → setuswapwep.
    # PICK_AXE lives at slot 4 in STARTING_INVENTORY[ARCHEOLOGIST].
    if role == Role.ARCHEOLOGIST:
        inv_state = inv_state.replace(swap_weapon=jnp.int8(4))
    # Samurai: YA (ammo) is auto-quivered by ini_inv (u_init.c:1143-1145
    # setuqwep).  YA lives at slot 3 in STARTING_INVENTORY[SAMURAI].
    if role == Role.SAMURAI:
        inv_state = inv_state.replace(quiver=jnp.int8(3))
    # Caveman: CLUB wielded (slot 0), SLING is the swap weapon (slot 1), and
    # FLINT (slot 2, sling ammo) is auto-quivered via setuqwep.  ROCK (slot 3)
    # is not quivered (uquiver already taken by FLINT) so it renders bare.
    # Cite: vendor/nethack/src/u_init.c:1141-1149 (is_ammo -> setuqwep;
    # leftover weapon -> setuswapwep).
    if role == Role.CAVEMAN:
        inv_state = inv_state.replace(swap_weapon=jnp.int8(1), quiver=jnp.int8(2))

    # --- Wear starting armor ---
    worn_armor = jnp.full((7,), -1, dtype=jnp.int8)
    armor_map  = _WORN_ARMOR_BY_ROLE.get(role, {})
    for armor_slot, item_idx in armor_map.items():
        if item_idx < len(items_list):
            item = items_list[item_idx]
            if int(item.category) == int(ItemCategory.ARMOR):
                worn_armor = worn_armor.at[int(armor_slot)].set(jnp.int8(item_idx))
    inv_state = inv_state.replace(worn_armor=worn_armor)

    # --- Compute AC ---
    player_ac = compute_ac(inv_state.items, worn_armor)

    # --- Starting spells (Wizard/Priest/Healer/Monk) ---
    # Build spell_memory / spell_known arrays Python-side (role is static).
    spell_memory = [0] * N_SPELLS
    spell_known  = [False] * N_SPELLS
    for sid in STARTING_SPELLS.get(role, []):
        spell_memory[int(sid)] = int(MAX_SPELL_MEMORY)
        spell_known[int(sid)]  = True
    magic_state = MagicState(
        spell_memory=jnp.array(spell_memory, dtype=jnp.int32),
        spell_known=jnp.array(spell_known, dtype=jnp.bool_),
        spell_letter=jnp.full((N_SPELLS,), -1, dtype=jnp.int8),
        pw_regen_counter=jnp.int32(0),
    )

    # --- Alignment record (vendor u_init.c::ini_hpwp lines 992-995:
    #     u.ualign.record = urole.initrecord). PrayerState owns this field
    #     in our state schema; carry it back into PrayerState via the env
    #     reset path. ---
    from Nethax.nethax.subsystems.prayer import PrayerState as _PrayerState
    init_record = int(get_role(role).initrecord)
    prayer_state = _PrayerState.default()
    prayer_state = prayer_state.replace(
        alignment_record=jnp.int16(init_record),
    )

    # --- player_amax: race attrmax capped at 18 for int8 storage ---
    # Vendor: u.urace.attrmax[] set during init_attr from race.attrmax[].
    # STR cap > 18 (18/** range) is clamped to 18 here since int8 can't
    # hold values > 127; restore_ability uses this as the per-stat ceiling.
    # Stat order matches _STAT_NAMES: str, int, wis, dex, con, cha.
    # Cite: vendor/nethack/src/u_init.c lines 250-580;
    #       vendor/nethack/src/potion.c::peffect_restore_ability (full_restore).
    race_entry = get_race(race)
    amax_vals = [min(int(race_entry.attrmax[i]), 18) for i in range(6)]
    player_amax = jnp.array(amax_vals, dtype=jnp.int8)

    # --- uhpinc[0] / ueninc[0]: record the initial HP/Pw grant so that the
    # ulevel==0 -> ulevel==1 transition is part of the per-level history that
    # newman()/losexp() consume.  Vendor attrib.c::newhp lines 1129-1131 and
    # exper.c::newpw lines 68-70 both write ``uhpinc[u.ulevel] = hp`` (resp.
    # ``ueninc[u.ulevel] = en``); during ini_hpwp the value of u.ulevel is 0
    # so the slot written is index 0.  Cite: vendor/nethack/src/u_init.c
    # ini_hpwp; attrib.c:1129-1131; exper.c:68-70.
    player_uhpinc = jnp.zeros((31,), dtype=jnp.int16).at[0].set(
        hp.astype(jnp.int16)
    )
    player_ueninc = jnp.zeros((31,), dtype=jnp.int16).at[0].set(
        pw.astype(jnp.int16)
    )

    return dict(
        player_role=jnp.int8(int(role)),
        player_race=jnp.int8(int(race)),
        player_align=jnp.int8(int(alignment)),
        player_str=jnp.int16(stats["str"]),
        player_dex=jnp.int8(stats["dex"]),
        player_con=jnp.int8(stats["con"]),
        player_int=jnp.int8(stats["int"]),
        player_wis=jnp.int8(stats["wis"]),
        player_cha=jnp.int8(stats["cha"]),
        player_amax=player_amax,
        player_hp=hp,
        player_hp_max=hp,
        player_pw=pw,
        player_pw_max=pw,
        player_uhpinc=player_uhpinc,
        player_ueninc=player_ueninc,
        player_ac=player_ac,
        # Vendor default: luck starts at 0 (you.h::u.uluck), no birth opt.
        player_luck=jnp.int8(0),
        inventory=inv_state,
        magic=magic_state,
        prayer=prayer_state,
        **({"vendor_rng": vendor_rng} if vendor_rng is not None else {}),
    )
