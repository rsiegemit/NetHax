"""Character creation — role/race starting inventory, stats, HP/PW.

Canonical sources:
  vendor/nethack/src/u_init.c::ini_inv   — per-role starting items
  vendor/nethack/src/role.c::roles[]     — 13 role stat/HP/PW tables
  vendor/nethack/src/role.c::races[]     — 5 race stat tables

Status: Wave 3 — create_character implemented.

TODO (Wave 4):
  - Race-based item substitutions (elven dagger instead of dagger for elves, etc.)
  - Luck, alignment-altar prayer starting bonus
  - Pet initialization
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
    LEATHER_ARMOR     = 62
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

    # Potions (POTION_CLASS = 8)
    POT_WATER         = 200
    POT_BOOZE         = 201
    POT_SICKNESS      = 202
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
    return make_item(
        category=ItemCategory.WEAPON,
        type_id=type_id,
        quantity=qty,
        weight=_WEAPON_WEIGHT.get(type_id, 20) * qty,
        is_two_handed=two_handed,
        enchantment=enchant,
        buc_status=buc,
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
    )


def _potion(type_id: int, qty: int = 1, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.POTION, type_id=type_id, quantity=qty,
                     weight=20 * qty, buc_status=buc)


def _scroll(type_id: int, qty: int = 1, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.SCROLL, type_id=type_id, quantity=qty,
                     weight=5 * qty, buc_status=buc)


def _spellbook(type_id: int, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.SPBOOK, type_id=type_id, quantity=1,
                     weight=50, buc_status=buc)


def _tool(type_id: int, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.TOOL, type_id=type_id, quantity=1,
                     weight=30, buc_status=buc)


def _wand(type_id: int, buc: int = _BUC_UNCURSED) -> "Item":
    from Nethax.nethax.subsystems.inventory import make_item
    return make_item(category=ItemCategory.WAND, type_id=type_id, quantity=1,
                     weight=7, buc_status=buc)


# ---------------------------------------------------------------------------
# STARTING_INVENTORY
# Each entry is a list of Item objects (pre-built, not names).
# Canonical: vendor/nethack/src/u_init.c trobj arrays.
# ---------------------------------------------------------------------------

STARTING_INVENTORY: dict = {
    # Arc: bullwhip+2, leather jacket+0, fedora+0, pick-axe, tinning kit
    # u_init.c Archeologist[]:  BULLWHIP spe=2, LEATHER_JACKET 0, FEDORA 0.
    Role.ARCHEOLOGIST: [
        _weapon(ObjType.BULLWHIP, enchant=2),
        _armor(ObjType.LEATHER_JACKET),
        _armor(ObjType.FEDORA),
        _tool(ObjType.PICK_AXE),
        _tool(ObjType.TINNING_KIT),
    ],
    # Bar variant 0: two-handed sword+0, axe+0, ring mail+0 (u_init.c Barbarian_0)
    Role.BARBARIAN: [
        _weapon(ObjType.TWO_HANDED_SWORD),
        _weapon(ObjType.AXE),
        _armor(ObjType.RING_MAIL),
    ],
    # Cav: club+1, sling+2, rocks, leather armor+0 (u_init.c Caveman[])
    Role.CAVEMAN: [
        _weapon(ObjType.CLUB, enchant=1),
        _weapon(ObjType.SLING, enchant=2),
        make_item(category=ItemCategory.GEM, type_id=1, quantity=18, weight=18,
                  buc_status=_BUC_UNCURSED),  # rocks
        _armor(ObjType.LEATHER_ARMOR),
    ],
    # Hea: scalpel+0, leather gloves+1, stethoscope, 4 healing + 4 extra healing,
    #      wand of sleep, spellbooks (3 of them BLESSED per u_init.c).
    # u_init.c Healer[]:  SCALPEL 0, LEATHER_GLOVES 1, spellbooks trbless=1.
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
    ],
    # Kni: long sword+1, lance+1, ring mail+1, helmet+0, small shield+0,
    #      leather gloves+0 (u_init.c Knight[]).  trbless=UNDEF_BLESS for all
    #      → cursed=0; default blessed flag is 0 → UNCURSED.
    Role.KNIGHT: [
        _weapon(ObjType.LONG_SWORD, enchant=1),
        _weapon(ObjType.LANCE, enchant=1),
        _armor(ObjType.RING_MAIL, enchant=1),
        _armor(ObjType.HELMET),
        _armor(ObjType.SMALL_SHIELD),
        _armor(ObjType.LEATHER_GLOVES),
    ],
    # Mon: leather gloves+2, robe+1 (fights barehanded — no weapon).
    # u_init.c Monk[]:  LEATHER_GLOVES spe=2, ROBE spe=1.
    Role.MONK: [
        _armor(ObjType.LEATHER_GLOVES, enchant=2),
        _armor(ObjType.ROBE, enchant=1),
    ],
    # Pri: mace+1 (BLESSED — vendor trbless=1), robe+0, small shield+0,
    #      4 holy water (BLESSED — vendor trbless=1).
    # u_init.c Priest[]:  MACE spe=1 trbless=1, POT_WATER trbless=1.
    Role.PRIEST: [
        _weapon(ObjType.MACE, enchant=1, buc=_BUC_BLESSED),
        _armor(ObjType.ROBE),
        _armor(ObjType.SMALL_SHIELD),
        _potion(ObjType.POT_WATER, 4, buc=_BUC_BLESSED),   # holy water
    ],
    # Ran: dagger+1, bow+1, arrow+2 x50, cloak of displacement+2
    # u_init.c Ranger[]:  DAGGER spe=1, BOW spe=1, ARROW spe=2,
    #                     CLOAK_OF_DISPLACEMENT spe=2.
    Role.RANGER: [
        _weapon(ObjType.DAGGER, enchant=1),
        _weapon(ObjType.BOW, enchant=1),
        _weapon(ObjType.ARROW, 50, enchant=2),
        _armor(ObjType.CLOAK_OF_DISPLACEMENT, enchant=2),
    ],
    # Rog: short sword+0, daggers+0 x10, leather armor+1 (u_init.c Rogue[]).
    Role.ROGUE: [
        _weapon(ObjType.SHORT_SWORD),
        _weapon(ObjType.DAGGER, 10),
        _armor(ObjType.LEATHER_ARMOR, enchant=1),
        _tool(ObjType.LOCK_PICK),
        _tool(ObjType.SACK),
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
    # Val: spear+1, dagger+0, small shield+3 (u_init.c Valkyrie[]).
    #      trbless=UNDEF_BLESS for all → UNCURSED.
    Role.VALKYRIE: [
        _weapon(ObjType.SPEAR, enchant=1),
        _weapon(ObjType.DAGGER),
        _armor(ObjType.SMALL_SHIELD, enchant=3),
    ],
    # Wiz: quarterstaff+1 (BLESSED — trbless=1), cloak of magic resistance+0,
    #      wand, force-bolt spellbook (BLESSED), random potions/scrolls.
    # u_init.c Wizard[]:  QUARTERSTAFF spe=1 trbless=1, SPE_FORCE_BOLT trbless=1.
    Role.WIZARD: [
        _weapon(ObjType.QUARTERSTAFF, enchant=1, buc=_BUC_BLESSED),
        _armor(ObjType.CLOAK_OF_MAGIC_RESISTANCE),
        _wand(ObjType.WAN_SLEEP),          # representative random wand
        _spellbook(ObjType.SPE_FORCE_BOLT, buc=_BUC_BLESSED),
        _potion(ObjType.POT_HEALING, 3),   # representative random potions
        _scroll(ObjType.SCR_IDENTIFY, 3),  # representative random scrolls
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
#   pw_per_level: energy gained per level (approximation)
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

    JIT note: this is called from ``env.reset`` (non-jit path); we use Python
    loops over a fixed iteration cap. Each call uses fresh PRNG keys from
    ``jax.random.split`` so it remains deterministic for a given seed.
    """
    r_entry = get_role(role)
    race_entry = get_race(race)

    attrbase = list(r_entry.attrbase)
    attrdist = list(r_entry.attrdist)
    attrmin  = list(race_entry.attrmin)
    attrmax  = list(race_entry.attrmax)

    # Start each stat at attrbase; remaining points to distribute.
    values = [int(attrbase[i]) for i in range(6)]
    remaining = int(np_total) - sum(values)

    # Distribute extras weighted by attrdist, capped by race attrmax.
    # tryct caps run-away when every stat is already at race-cap.
    max_iters = 1000  # generous upper bound to ensure determinism
    tryct = 0
    keys = jax.random.split(rng, max_iters)
    key_idx = 0
    while remaining > 0 and tryct < 100 and key_idx < max_iters:
        # x = rn2(100) — vendor's weighted selection.
        x = int(jax.random.randint(
            keys[key_idx], shape=(), minval=0, maxval=100
        ))
        key_idx += 1
        i = 0
        x_left = x
        while i < 6:
            x_left -= int(attrdist[i])
            if x_left < 0:
                break
            i += 1
        if i >= 6:
            continue  # vendor: "impossible"
        if values[i] >= int(attrmax[i]):
            tryct += 1
            continue
        tryct = 0
        values[i] += 1
        remaining -= 1

    # Final clamp to race floor (paranoia: floor is 3 for all races).
    for i in range(6):
        if values[i] < int(attrmin[i]):
            values[i] = int(attrmin[i])
        if values[i] > int(attrmax[i]):
            values[i] = int(attrmax[i])

    return {
        name: jnp.int32(values[i])
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
        player_hp=hp,
        player_hp_max=hp,
        player_pw=pw,
        player_pw_max=pw,
        player_ac=player_ac,
        # Vendor default: luck starts at 0 (you.h::u.uluck), no birth opt.
        player_luck=jnp.int8(0),
        inventory=inv_state,
        magic=magic_state,
        prayer=prayer_state,
    )
