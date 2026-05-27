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

    # Food (FOOD_CLASS = 7) — vendor otyp values
    # Canonical source: constants/objects.py indices 252-268
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
    return make_item(
        category=ItemCategory.WEAPON,
        type_id=type_id,
        quantity=qty,
        weight=_WEAPON_WEIGHT.get(type_id, 20) * qty,
        is_two_handed=two_handed,
        enchantment=enchant,
        buc_status=buc,
        identified=True,
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
    )


def _potion(type_id: int, qty: int = 1, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.POTION, type_id=type_id, quantity=qty,
                     weight=20 * qty, buc_status=buc, identified=True)


def _scroll(type_id: int, qty: int = 1, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.SCROLL, type_id=type_id, quantity=qty,
                     weight=5 * qty, buc_status=buc, identified=True)


def _spellbook(type_id: int, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.SPBOOK, type_id=type_id, quantity=1,
                     weight=50, buc_status=buc, identified=True)


def _tool(type_id: int, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.TOOL, type_id=type_id, quantity=1,
                     weight=30, buc_status=buc, identified=True)


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
    )


def _wand(type_id: int, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.WAND, type_id=type_id, quantity=1,
                     weight=7, buc_status=buc, identified=True)


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
        _tool(ObjType.PICK_AXE),
        _tool(ObjType.TINNING_KIT),
        _tool(ObjType.SACK),
        _tool(ObjType.TOUCHSTONE),
        _food(ObjType.FOOD_RATION, 3),
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
    # Cav: club+1, sling+2, rocks, leather armor+0 (u_init.c Caveman[])
    Role.CAVEMAN: [
        _weapon(ObjType.CLUB, enchant=1),
        _weapon(ObjType.SLING, enchant=2),
        make_item(category=ItemCategory.GEM, type_id=1, quantity=18, weight=18,
                  buc_status=_BUC_UNCURSED, identified=True),  # rocks
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
    # Mon: leather gloves+2, robe+1 (fights barehanded), 3 healing potions,
    #      3 food rations, 5 apples, 5 oranges, 3 fortune cookies.
    # u_init.c Monk[] (101-113): LEATHER_GLOVES spe=2, ROBE spe=1, POT_HEALING
    # qty=3, FOOD_RATION qty=3, APPLE qty=5, ORANGE qty=5, FORTUNE_COOKIE qty=3.
    # Audit L #19: added missing potions + food.  Random SCROLL also in vendor —
    # deferred since the random-scroll-on-init pipeline isn't wired yet.
    Role.MONK: [
        _armor(ObjType.LEATHER_GLOVES, enchant=2),
        _armor(ObjType.ROBE, enchant=1),
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
    #   qty is ``6..15`` random; we keep the static 10 (median) until per-init
    #   random ranges are threaded.  BODY-armor slot index (2) is preserved
    #   so ``_WORN_ARMOR_BY_ROLE[ROGUE]`` still maps to LEATHER_ARMOR.
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
        _weapon(ObjType.SPEAR, enchant=1),
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
    Role.CAVEMAN:      {ArmorSlot.BODY: 3},
    Role.HEALER:       {ArmorSlot.GLOVES: 1},
    Role.KNIGHT:       {ArmorSlot.BODY: 2, ArmorSlot.HELM: 3,
                        ArmorSlot.SHIELD: 4, ArmorSlot.GLOVES: 5},
    Role.MONK:         {ArmorSlot.GLOVES: 0, ArmorSlot.BODY: 1},
    Role.PRIEST:       {ArmorSlot.BODY: 1, ArmorSlot.SHIELD: 2},
    Role.RANGER:       {ArmorSlot.CLOAK: 3},
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

        # Walk attrdist to find target stat index i; mirror vendor:
        #   i = 0
        #   while i < 6:
        #     x -= attrdist[i]
        #     if x < 0: break
        #     i += 1
        # i ends in [0, 6]; i == 6 means "impossible" (skip this draw).
        # Cumulative sum: i = first index where cumsum[i] > x; if none, 6.
        cum = jnp.cumsum(attrdist_arr)              # [6]
        # found[k] is True iff cum[k] > x.  argmax over found returns first True.
        found = cum > x
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
    from Nethax.nethax import vendor_rng as _vrng

    r_entry = get_role(role)
    race_entry = get_race(race)

    attrbase = list(int(v) for v in r_entry.attrbase)   # [6] ints
    attrdist = list(int(v) for v in r_entry.attrdist)   # [6] ints
    attrmin  = list(int(v) for v in race_entry.attrmin) # [6] ints
    attrmax  = list(int(v) for v in race_entry.attrmax) # [6] ints
    a_max_idx = 6  # A_MAX — vendor include/attrib.h

    # Step 1: initialise ABASE from role, compute remaining points.
    # attrib.c:619-623
    abase = list(attrbase)
    np = np_total - sum(attrbase)

    # Step 2: distribute excess points upward.
    # attrib.c:625-640
    tryct = 0
    while np > 0 and tryct < 100:
        vendor_rng, x = _vrng.rn2(vendor_rng, 100)  # attrib.c:627
        # Walk attrdist: i = first index where cumulative attrdist > x.
        # attrib.c:628-629: ``for (i=0; (i<A_MAX) && ((x -= attrdist[i]) > 0); i++)``
        i = 0
        while i < a_max_idx and x >= 0:
            x -= attrdist[i]
            if x >= 0:
                i += 1
        if i >= a_max_idx:          # attrib.c:630-631 impossible branch
            continue                 # (never reached when attrdist sums to 100)
        if abase[i] >= attrmax[i]:  # attrib.c:633-636
            tryct += 1
            continue
        tryct = 0                   # attrib.c:637-640
        abase[i] += 1
        np -= 1

    # Step 3: redistribute excess points downward (np < 0).
    # attrib.c:643-660
    tryct = 0
    while np < 0 and tryct < 100:
        vendor_rng, x = _vrng.rn2(vendor_rng, 100)  # attrib.c:646
        i = 0
        while i < a_max_idx and x >= 0:
            x -= attrdist[i]
            if x >= 0:
                i += 1
        if i >= a_max_idx:          # attrib.c:649-651
            continue
        if abase[i] <= attrmin[i]:  # attrib.c:652-655
            tryct += 1
            continue
        tryct = 0                   # attrib.c:656-659
        abase[i] -= 1
        np += 1

    # Clamp final values to [attrmin, attrmax].
    clamped = [max(attrmin[i], min(attrmax[i], abase[i])) for i in range(a_max_idx)]

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
            rn2(2)        # blessed flag
            # rne(3) omitted — jit-inexact; draws suppressed (see note)
        elif !rn2(10):
            pass          # curse branch — rne(3) suppressed
        else:
            blessorcurse(10) → rn2(10) [+ rn2(2) if hit]

    ARMOR_CLASS draw pattern (mkobj.c:992-1004)::

        rn2(10)           # outer: fumble/levitation/etc. guard
        if outer hit:
            rn2(11)       # inner: !rn2(11) blessed check
            # rne(3) suppressed
        elif !rn2(10):
            rn2(2)        # blessed flag
            # rne(3) suppressed
        else:
            blessorcurse(10) → rn2(10) [+ rn2(2) if hit]

    Note on rne(3): ``rne(3)`` at u.ulevel==0 draws 1–5 times (``while tmp <
    5 && !rn2(3)``).  The number of draws is seed-dependent and at this stage
    we have no way to correctly branch without running the full game logic.
    The audit (PRE_MKLEV_RNG_AUDIT.md §4c) counts these draws inside the
    minimum ~11-draw estimate; omitting them is tracked as a known gap.

    Returns the post-draw ``Isaac64State``.

    Citations
    ---------
    vendor/nle/src/u_init.c:749-756     — PM_ROGUE role-switch block
    vendor/nle/src/mkobj.c:803-818      — WEAPON_CLASS init
    vendor/nle/src/mkobj.c:992-1004     — ARMOR_CLASS init
    vendor/nle/src/mkobj.c:981-987      — POTION_CLASS / blessorcurse(4)
    vendor/nle/src/mkobj.c:309          — mkbox_cnts rn2(n+1)
    vendor/nle/src/mkobj.c:1370-1385    — blessorcurse definition
    """
    from Nethax.nethax import vendor_rng as _vrng

    # 1. rn1(10, 6) dagger quantity — u_init.c:750
    vendor_rng, _dagger_qty = _vrng.rn2(vendor_rng, 10)

    # 2–3. SHORT_SWORD and DAGGER — WEAPON_CLASS (mkobj.c:803-818)
    for _ in range(2):
        vendor_rng, r11 = _vrng.rn2(vendor_rng, 11)
        if r11 == 0:
            # !rn2(11) branch: spe=rne(3), blessed=rn2(2)
            # rne(3) omitted (variable; tracked as known gap)
            vendor_rng, _blessed = _vrng.rn2(vendor_rng, 2)
        else:
            vendor_rng, r10 = _vrng.rn2(vendor_rng, 10)
            if r10 == 0:
                # elif !rn2(10): curse + rne(3) — rne(3) omitted
                pass
            else:
                # else: blessorcurse(10) → rn2(10) [+ rn2(2) if 0]
                vendor_rng, bc = _vrng.rn2(vendor_rng, 10)
                if bc == 0:
                    vendor_rng, _bc2 = _vrng.rn2(vendor_rng, 2)

    # 4. LEATHER_ARMOR — ARMOR_CLASS (mkobj.c:992-1004)
    # C code: if (rn2(10) && (FUMBLE_BOOTS||...||!rn2(11))) { curse; rne }
    #         else if (!rn2(10))                              { bless; rne }
    #         else                                            { blessorcurse(10) }
    # LEATHER_ARMOR is none of the named special types, so the inner || chain
    # reaches !rn2(11) — that draw always fires when outer rn2(10) != 0.
    # When outer rn2(10) == 0, C short-circuits and rn2(11) is NOT drawn.
    vendor_rng, outer = _vrng.rn2(vendor_rng, 10)   # mkobj.c:993
    if outer != 0:
        # outer rn2(10) non-zero → evaluate || chain → reaches !rn2(11)
        vendor_rng, r11 = _vrng.rn2(vendor_rng, 11)  # mkobj.c:997
        if r11 == 0:
            # if-branch taken: curse + rne(3) — rne(3) omitted (known gap)
            pass
        else:
            # if-condition False → fall to elif !rn2(10)
            vendor_rng, elif10 = _vrng.rn2(vendor_rng, 10)  # mkobj.c:1000
            if elif10 == 0:
                # elif branch: blessed = rn2(2) + rne(3) — rne(3) omitted
                vendor_rng, _blessed = _vrng.rn2(vendor_rng, 2)
            else:
                # else: blessorcurse(10)
                vendor_rng, bc = _vrng.rn2(vendor_rng, 10)
                if bc == 0:
                    vendor_rng, _bc2 = _vrng.rn2(vendor_rng, 2)
    else:
        # outer rn2(10) == 0 → if-condition False (short-circuit, no rn2(11))
        # fall to elif !rn2(10)
        vendor_rng, elif10 = _vrng.rn2(vendor_rng, 10)  # mkobj.c:1000
        if elif10 == 0:
            vendor_rng, _blessed = _vrng.rn2(vendor_rng, 2)
        else:
            vendor_rng, bc = _vrng.rn2(vendor_rng, 10)
            if bc == 0:
                vendor_rng, _bc2 = _vrng.rn2(vendor_rng, 2)

    # 5. POT_SICKNESS — POTION_CLASS (mkobj.c:981-987 → blessorcurse(4))
    vendor_rng, bc = _vrng.rn2(vendor_rng, 4)
    if bc == 0:
        vendor_rng, _bc2 = _vrng.rn2(vendor_rng, 2)

    # 6. LOCK_PICK — TOOL_CLASS, no switch case → 0 draws
    # (mkobj.c:897-965: LOCK_PICK not listed → falls off switch with no RNG)

    # 7. SACK — TOOL_CLASS: mkbox_cnts at moves<=1 → n=0 → rn2(1) (mkobj.c:309)
    vendor_rng, _sack = _vrng.rn2(vendor_rng, 1)

    return vendor_rng


# ---------------------------------------------------------------------------
# _consume_attr_variation_draws  — vendor u_init.c:887-894 post-init_attr loop
# ---------------------------------------------------------------------------

def _consume_attr_variation_draws(vendor_rng):
    """Consume the 6 ``rn2(20)`` draws (+ conditional ``rn2(7)``) emitted by
    the post-``init_attr`` attribute variation loop.

    Vendor code (u_init.c:887-894)::

        for (i = 0; i < A_MAX; i++)   // A_MAX == 6
            if (!rn2(20)) {
                int xd = rn2(7) - 2;
                adjattrib(i, xd, TRUE);
                ...
            }

    Always fires exactly 6 ``rn2(20)`` draws; each zero result adds one
    ``rn2(7)`` draw.  ``adjattrib`` itself does not draw from CORE.

    Returns the post-draw ``Isaac64State``.

    Citation: vendor/nle/src/u_init.c:887-894
    """
    from Nethax.nethax import vendor_rng as _vrng

    for _ in range(6):  # A_MAX = 6
        vendor_rng, gate = _vrng.rn2(vendor_rng, 20)  # u_init.c:888
        if gate == 0:
            vendor_rng, _xd = _vrng.rn2(vendor_rng, 7)  # u_init.c:889

    return vendor_rng


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
    if vendor_rng is not None and role == Role.ROGUE:
        # Step 1+2: dagger qty + ini_inv(Rogue) blessorcurse
        # Cite: vendor/nle/src/u_init.c:750; mkobj.c:803-1004
        vendor_rng = _consume_ini_inv_rogue_draws(vendor_rng)
        # Step 3: BLINDFOLD gate — u_init.c:753
        from Nethax.nethax import vendor_rng as _vendor_rng_mod
        vendor_rng, blindfold_roll = _vendor_rng_mod.rn2(vendor_rng, 5)

    # --- Stat rolls (vendor init_attr(75) parity) ---
    # Step 4: NLE_BYTEPARITY mode: consume ISAAC64 draws via consume_init_attr_draws.
    # Plain mode: fall back to Threefry-based _init_attr_vendor.
    # Cite: vendor/nle/src/attrib.c:614-660; u_init.c:882.
    if vendor_rng is not None:
        vendor_rng, stats_raw = consume_init_attr_draws(vendor_rng, role, race)
        stats = {k: jnp.int32(v) for k, v in stats_raw.items()}
    else:
        stats = _init_attr_vendor(rng_stats, role, race, np_total=75)

    # --- NLE_BYTEPARITY: attr variation loop (all roles) ---
    # Step 5: 6 rn2(20) draws + conditional rn2(7) per hit.
    # Cite: vendor/nle/src/u_init.c:887-894
    if vendor_rng is not None:
        vendor_rng = _consume_attr_variation_draws(vendor_rng)

    # --- HP / Pw (vendor ini_hpwp parity, u.ulevel == 0 branch) ---
    hp, pw = _ini_hpwp_vendor(rng_hp, role, race)

    # --- Build inventory ---
    items_list = STARTING_INVENTORY[role]
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
        # NLE_BYTEPARITY: blindfold_roll was consumed above
        if blindfold_roll != 0:
            # rn2(5) != 0 → !rn2(5) is False → skip BLINDFOLD
            items_list = [
                it for it in items_list
                if int(it.type_id) != int(ObjType.BLINDFOLD)
            ]
        # else: rn2(5) == 0 → !rn2(5) is True → keep BLINDFOLD in kit
    elif _is_nle() and role == Role.ROGUE:
        # Static fallback (plain NLE / Threefry): strip BLINDFOLD.
        # Cite: vendor/nle/src/u_init.c:753-754.
        items_list = [
            it for it in items_list
            if int(it.type_id) != int(ObjType.BLINDFOLD)
        ]
    inv_state  = InventoryState.from_items(items_list)

    # --- Wield primary weapon ---
    wielded = jnp.int8(-1)
    if role not in _NO_WEAPON_ROLES and len(items_list) > 0:
        # Wield slot 0 (primary weapon is always first in the list)
        first_cat = int(items_list[0].category)
        if first_cat == int(ItemCategory.WEAPON):
            wielded = jnp.int8(0)
    inv_state = inv_state.replace(wielded=wielded)

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
