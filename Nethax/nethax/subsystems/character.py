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
    Values are sequential within each class as they appear in objects.h.
    """
    # Weapons (WEAPON_CLASS = 2)
    ARROW             = 1
    ELVEN_ARROW       = 2
    ORCISH_ARROW      = 3
    SILVER_ARROW      = 4
    YA                = 5
    CROSSBOW_BOLT     = 6
    DART              = 7
    SHURIKEN          = 8
    BOOMERANG         = 9
    SPEAR             = 10
    ELVEN_SPEAR       = 11
    ORCISH_SPEAR      = 12
    DWARVISH_SPEAR    = 13
    SILVER_SPEAR      = 14
    JAVELIN           = 15
    TRIDENT           = 16
    DAGGER            = 17
    ELVEN_DAGGER      = 18
    ORCISH_DAGGER     = 19
    SILVER_DAGGER     = 20
    SCALPEL           = 21
    KNIFE             = 22
    STILETTO          = 23
    WORM_TOOTH        = 24
    CRYSKNIFE         = 25
    AXE               = 26  # single-handed axe (Barbarian_0)
    BATTLE_AXE        = 27
    SHORT_SWORD       = 28
    ELVEN_SHORT_SWORD = 29
    ORCISH_SHORT_SWORD= 30
    DWARVISH_SHORT_SWORD=31
    SABER             = 32
    BROADSWORD        = 33
    LONG_SWORD        = 34
    TWO_HANDED_SWORD  = 35
    KATANA            = 36
    TSURUGI           = 37
    MACE              = 38
    MORNING_STAR      = 39
    WAR_HAMMER        = 40
    CLUB              = 41
    RUBBER_HOSE       = 42
    QUARTERSTAFF      = 43
    AKLYS             = 44
    LANCE             = 45
    BULLWHIP          = 46
    BOW               = 47
    ELVEN_BOW         = 48
    ORCISH_BOW        = 49
    YUMI              = 50
    SLING             = 51
    CROSSBOW          = 52
    PICK_AXE          = 53

    # Armor (ARMOR_CLASS = 3)
    LEATHER_GLOVES    = 60
    PADDED_ARMOR      = 61
    LEATHER_ARMOR     = 113  # live-NLE otyp; vendor/nle/src/objects.c sequential count
    LEATHER_JACKET    = 63
    RING_MAIL         = 64
    STUDDED_LEATHER_ARMOR = 65
    CHAIN_MAIL        = 66
    SCALE_MAIL        = 67
    SPLINT_MAIL       = 68
    BANDED_MAIL       = 69
    DWARVISH_MITHRIL_COAT = 70
    ELVEN_MITHRIL_COAT = 71
    BRONZE_PLATE_MAIL = 72
    PLATE_MAIL        = 73
    CRYSTAL_PLATE_MAIL = 74
    SMALL_SHIELD      = 75
    ELVEN_SHIELD      = 76
    URUK_HAI_SHIELD   = 77
    ORCISH_SHIELD     = 78
    DWARVISH_ROUNDSHIELD = 79
    LARGE_SHIELD      = 80
    DWARVISH_IRON_HELM = 81
    ORCISH_HELM       = 82
    HELMET            = 83
    ELVEN_LEATHER_HELM = 84
    FEDORA            = 85
    LEATHER_HELM      = 86
    LEVITATION_BOOTS  = 87
    JUMPING_BOOTS     = 88
    ELVEN_BOOTS       = 89
    KICKING_BOOTS     = 90
    FUMBLE_BOOTS      = 91
    SPEED_BOOTS       = 92
    IRON_SHOES        = 93
    HIGH_BOOTS        = 94
    CLOAK_OF_PROTECTION = 95
    CLOAK_OF_INVISIBILITY = 96
    OILSKIN_CLOAK     = 97
    CLOAK_OF_MAGIC_RESISTANCE = 98
    CLOAK_OF_DISPLACEMENT = 99
    ELVEN_CLOAK       = 100
    ROBE              = 101
    ORCISH_CLOAK      = 102
    HAWAIIAN_SHIRT    = 103
    T_SHIRT           = 104
    HELM_OF_BRILLIANCE        = 105
    HELM_OF_OPPOSITE_ALIGNMENT = 106
    HELM_OF_TELEPATHY         = 107
    DUNCE_CAP                 = 108
    CORNUTHAUM                = 109
    TINFOIL_HAT               = 110
    MUMMY_WRAPPING            = 111
    GAUNTLETS_OF_FUMBLING     = 112
    GAUNTLETS_OF_POWER        = 113
    GAUNTLETS_OF_DEXTERITY    = 114
    LOW_BOOTS                 = 115
    WATER_WALKING_BOOTS       = 116

    # Helm of caution (vendor/nethack/include/objects.h:479-481 — oc_oprop=WARNING)
    HELM_OF_CAUTION           = 117

    # Dragon scale mails (vendor/nethack/include/objects.h:502-526)
    # Order mirrors vendor DRGN_ARMR block; powers from oc_oprop column.
    GRAY_DRAGON_SCALE_MAIL    = 118   # ANTIMAGIC
    GOLD_DRAGON_SCALE_MAIL    = 119   # (light source — no intrinsic)
    SILVER_DRAGON_SCALE_MAIL  = 120   # REFLECTING
    RED_DRAGON_SCALE_MAIL     = 121   # FIRE_RES
    WHITE_DRAGON_SCALE_MAIL   = 122   # COLD_RES
    ORANGE_DRAGON_SCALE_MAIL  = 123   # SLEEP_RES
    BLACK_DRAGON_SCALE_MAIL   = 124   # DISINT_RES
    BLUE_DRAGON_SCALE_MAIL    = 125   # SHOCK_RES
    GREEN_DRAGON_SCALE_MAIL   = 126   # POISON_RES
    YELLOW_DRAGON_SCALE_MAIL  = 127   # ACID_RES

    # Dragon scales (vendor/nethack/include/objects.h:530-553) — same powers
    GRAY_DRAGON_SCALES        = 128   # ANTIMAGIC
    GOLD_DRAGON_SCALES        = 129
    SILVER_DRAGON_SCALES      = 130   # REFLECTING
    RED_DRAGON_SCALES         = 131   # FIRE_RES
    WHITE_DRAGON_SCALES       = 132   # COLD_RES
    ORANGE_DRAGON_SCALES      = 133   # SLEEP_RES
    BLACK_DRAGON_SCALES       = 134   # DISINT_RES
    BLUE_DRAGON_SCALES        = 135   # SHOCK_RES
    GREEN_DRAGON_SCALES       = 136   # POISON_RES
    YELLOW_DRAGON_SCALES      = 137   # ACID_RES

    # Potions (POTION_CLASS = 8)
    POT_WATER         = 200
    POT_BOOZE         = 201
    POT_SICKNESS      = 277  # live-NLE C enum otyp; vendor/nle/src/u_init.c:127
    POT_CONFUSION     = 203
    POT_EXTRA_HEALING = 204
    POT_HEALING       = 205
    POT_SPEED         = 206
    POT_BLINDNESS     = 207
    POT_GAIN_ENERGY   = 208
    POT_LEVITATION    = 209
    POT_HALLUCINATION = 210
    POT_RESTORE_ABILITY = 211
    POT_GAIN_LEVEL    = 212
    POT_INVULNERABILITY = 213
    POT_POLYMORPH     = 214
    POT_OIL           = 215
    POT_GAIN_STRENGTH = 216
    POT_FULL_HEALING  = 217
    POT_ENLIGHTENMENT = 218
    POT_SEE_INVISIBLE = 219
    POT_ACID          = 220
    POT_SLEEPING      = 221
    POT_PARALYSIS     = 222

    # Scrolls (SCROLL_CLASS = 9)
    SCR_MAGIC_MAPPING = 300
    SCR_IDENTIFY      = 301

    # Spellbooks (SPBOOK_CLASS = 10)
    SPE_FORCE_BOLT    = 400
    SPE_HEALING       = 401
    SPE_EXTRA_HEALING = 402
    SPE_STONE_TO_FLESH= 403

    # Wands (WAND_CLASS = 11)
    WAN_SLEEP         = 500

    # Tools (TOOL_CLASS = 6)
    LOCK_PICK         = 600
    SACK              = 601
    TINNING_KIT       = 602
    TOUCHSTONE        = 603
    STETHOSCOPE       = 604
    EXPENSIVE_CAMERA  = 605
    CREDIT_CARD       = 606
    MAGIC_MARKER      = 607
    # BLINDFOLD vendor otyp = 208 (see subsystems/prayer.py:437).  Kept under
    # the TOOL_CLASS bucket here because vendor classifies it as TOOL_CLASS in
    # u_init.c::Blindfold[] (line 184).  The numeric value matches vendor.
    BLINDFOLD         = 208

    # Food (FOOD_CLASS = 7) — vendor otyp values (objects.py:252-268).
    # Audit L #14-#26 references for starting inventory items.
    APPLE             = 252   # objects.py:252
    ORANGE            = 253   # objects.py:253
    CARROT            = 257   # objects.py:257
    SPRIG_OF_WOLFSBANE = 258  # objects.py:258
    CLOVE_OF_GARLIC   = 259   # objects.py:259
    FORTUNE_COOKIE    = 264   # objects.py:264
    LEMBAS_WAFER      = 266   # objects.py:266
    CRAM_RATION       = 267   # objects.py:267
    FOOD_RATION       = 268   # objects.py:268


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
# create_character
# ---------------------------------------------------------------------------

def create_character(rng: jax.Array, role: Role, race: Race, alignment: int):
    """Build an EnvState for a newly created character.

    Rolls attribute scores via the vendor ``init_attr(75)`` formula,
    computes starting HP/Pw via ``ini_hpwp`` (newhp / newpw at u.ulevel==0),
    populates starting inventory, wields primary weapon, and wears starting
    armor.  Also initialises the player's alignment record from
    ``role.initrecord`` (priest = 5, etc.) and zeroes luck (vendor default
    unless special birth options).

    Parameters
    ----------
    rng       : JAX PRNG key
    role      : Role enum value
    race      : Race enum value
    alignment : 0=lawful, 1=neutral, 2=chaotic (int)

    Returns
    -------
    Keyword-argument dict suitable for EnvState(**kwargs) or state.replace(**kwargs).
    We return a dict rather than an EnvState to avoid a circular import — the
    caller (env.py) merges these fields into the state returned by EnvState.default().

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

    # --- Stat rolls (vendor init_attr(75) parity) ---
    stats = _init_attr_vendor(rng_stats, role, race, np_total=75)

    # --- HP / Pw (vendor ini_hpwp parity, u.ulevel == 0 branch) ---
    hp, pw = _ini_hpwp_vendor(rng_hp, role, race)

    # --- Build inventory ---
    items_list = STARTING_INVENTORY[role]
    # Rogue's BLINDFOLD slot is gated `!rn2(5)` in vendor u_init.c:754 —
    # we can't reproduce vendor C's exact ISAAC64 consumption order, so
    # in NLE_BYTEPARITY mode we OMIT the BLINDFOLD slot (which matches
    # NLE on the dominant 4/5 of seeds where the roll fails).  In the
    # default NLE mode we ALSO omit it, since the validator showed
    # NLE seed 0 also fails the roll.  Only NETHACK mode keeps the
    # always-on BLINDFOLD.
    from Nethax.nethax.parity_mode import is_nle_mode as _is_nle
    if _is_nle() and role == Role.ROGUE:
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
    )
