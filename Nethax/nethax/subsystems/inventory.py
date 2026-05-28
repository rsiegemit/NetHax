"""Inventory subsystem — slot management, wear/wield/pickup/drop.

Canonical sources:
  vendor/nethack/src/invent.c  — inventory display, slot management (a-zA-Z)
  vendor/nethack/src/pickup.c  — auto-pickup, weight checking
  vendor/nethack/src/do_wear.c — wear/take-off armor
  vendor/nethack/src/wield.c   — wield weapon, two-weapon combat
  vendor/nethack/src/worn.c    — track intrinsics granted by worn gear

Status: full vendor-parity inventory operations.  Cursed-item locking,
enchantment to-hit/damage/AC contributions, erosion penalties, and the
container open/close/put/take pipeline (including bag-of-holding nested
weight reduction) are all wired — see ``compute_ac`` here and the
subsystems/containers.py module for the latter.
"""
from dataclasses import field
from enum import IntEnum

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import struct


# ---------------------------------------------------------------------------
# Item category enum  (vendor/nethack/include/defsym.h OBJCLASS entries)
# RANDOM_CLASS = 0 → used as "empty slot" sentinel here too.
# ---------------------------------------------------------------------------

class ItemCategory(IntEnum):
    NONE    = 0   # empty slot (RANDOM_CLASS in NetHack)
    ILLOBJ  = 1
    WEAPON  = 2
    ARMOR   = 3
    RING    = 4
    AMULET  = 5
    TOOL    = 6
    FOOD    = 7
    POTION  = 8
    SCROLL  = 9
    SPBOOK  = 10
    WAND    = 11
    COIN    = 12
    GEM     = 13
    ROCK    = 14
    BALL    = 15
    CHAIN   = 16
    VENOM   = 17


# ---------------------------------------------------------------------------
# Constants (vendor/nethack/src/invent.c, include/hack.h)
# ---------------------------------------------------------------------------

# NetHack uses letters a-z A-Z — exactly 52 slots.
# Cite: vendor/nethack/include/hack.h line 584 (invlet_basic = 52).
MAX_INVENTORY_SLOTS = 52  # a-zA-Z

# Weight-cap constants for encumbrance refusal.
# Cite: vendor/nethack/include/weight.h lines 12-25
#       WT_WEIGHTCAP_STRCON = 25, WT_WEIGHTCAP_SPARE = 50, MAX_CARR_CAP = 1000.
# vendor hack.c::weight_cap (lines 4295-4346) formula:
#   carrcap = 25*(STR + CON) + 50, capped at MAX_CARR_CAP = 1000.
WT_WEIGHTCAP_STRCON: int = 25
WT_WEIGHTCAP_SPARE:  int = 50
MAX_CARR_CAP:        int = 1000

# Loadstone otyp — special-cased by lift_object (pickup.c:1721) so it
# can always be picked up even when the 52-slot or weight cap would
# otherwise refuse the lift.
# Cite: vendor/nethack/include/objects.h LOADSTONE; constants/objects.py:9045
_LOADSTONE_TYPE_ID: int = 443

# User-given name length per slot (Wave 6).  Mirrors NetHack's ONAME_MAX in
# vendor/nethack/include/obj.h (capped at 16 chars + null for inv display).
USER_NAME_LEN = 16

# Body armor slots tracked by worn_armor array.
# vendor/nethack/include/obj.h: ARM_SUIT, ARM_SHIELD, ARM_HELM,
#   ARM_GLOVES, ARM_BOOTS, ARM_CLOAK, ARM_SHIRT
N_ARMOR_SLOTS = 7  # body, shield, helm, gloves, boots, cloak, shirt

# Additional worn/wielded slots outside the armor array.
# weapon (wielded), off-hand (two-weapon), amulet, ring[0], ring[1]
# quiver is tracked separately; that makes 5 + 1 = 6 non-armor slots.
# (The spec asked us to verify: invent.c treats rings as two separate slots.)
N_WORN_NON_ARMOR_SLOTS = 5  # wielded, off_hand, amulet, ring_L, ring_R

# Ground stack depth per tile — max items on one floor tile.
MAX_GROUND_STACK = 8

# Base AC with no armor worn (NetHack: u.uac starts at 10).
BASE_AC = 10


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ArmorSlot(IntEnum):
    """Indices into InventoryState.worn_armor.

    Matches the ARM_* sub-field ordering in vendor/nethack/include/obj.h.
    """
    BODY   = 0  # ARM_SUIT
    SHIELD = 1  # ARM_SHIELD
    HELM   = 2  # ARM_HELM
    GLOVES = 3  # ARM_GLOVES
    BOOTS  = 4  # ARM_BOOTS
    CLOAK  = 5  # ARM_CLOAK
    SHIRT  = 6  # ARM_SHIRT


# ---------------------------------------------------------------------------
# Item struct
# ---------------------------------------------------------------------------

@struct.dataclass
class Item:
    """A single item slot. Empty slots have ``category == 0``.

    Migrated from legacy ``Nethax.nethax.nethax_state.Item`` during Wave 2.

    Field map vs vendor ``include/obj.h::struct obj`` (Wave 6 parity polish):
      category       <- oclass
      type_id        <- otyp
      buc_status     <- cursed/blessed bitfield pair (collapsed)
      enchantment    <- spe (general slot for enchant/charges)
      charges        <- spe (split for clarity; vendor reuses spe)
      identified     <- known
      quantity       <- quan
      weight         <- owt
      ac_bonus       <- derived from objects table (not in vendor struct)
      is_two_handed  <- derived from objects table (not in vendor struct)
      greased        <- greased bitfield     (vendor obj.h L#172)
      oeroded        <- oeroded:2 bitfield   (vendor obj.h L#162-164)
      oeroded2       <- oeroded2:2 bitfield  (rust/corrode tiers)
      oerodeproof    <- oerodeproof bitfield (vendor obj.h L#166)
      bknown         <- bknown bitfield      (BUC-awareness)
      lamplit        <- lamplit bitfield     (light source state)
      olocked        <- olocked bitfield     (container lock state)
    """
    category: jnp.ndarray       # ItemCategory enum value (0 = empty)
    type_id: jnp.ndarray        # weapon/armor/potion/scroll/... type id
    buc_status: jnp.ndarray     # 0=unknown / 1=cursed / 2=uncursed / 3=blessed
    enchantment: jnp.ndarray    # +/- enchantment level
    charges: jnp.ndarray        # remaining charges (wands)
    identified: jnp.ndarray     # bool — known type
    quantity: jnp.ndarray       # stack count (arrows, gold, etc.)
    weight: jnp.ndarray         # item weight in aum (avoirdupois units)
    ac_bonus: jnp.ndarray       # base AC contribution (armor only; 0 for non-armor)
    is_two_handed: jnp.ndarray  # bool — two-handed weapon flag
    # Vendor obj.h gameplay bitfields — default False/0 so existing call sites
    # that pre-date this expansion still work.
    greased: jnp.ndarray = field(default_factory=lambda: jnp.bool_(False))
    oeroded: jnp.ndarray = field(default_factory=lambda: jnp.int8(0))
    oeroded2: jnp.ndarray = field(default_factory=lambda: jnp.int8(0))
    oerodeproof: jnp.ndarray = field(default_factory=lambda: jnp.bool_(False))
    bknown: jnp.ndarray = field(default_factory=lambda: jnp.bool_(False))
    lamplit: jnp.ndarray = field(default_factory=lambda: jnp.bool_(False))
    olocked: jnp.ndarray = field(default_factory=lambda: jnp.bool_(False))
    # Corpse tracking: index into MONSTERS table (-1 = not a corpse).
    corpse_entry_idx: jnp.ndarray = field(default_factory=lambda: jnp.int16(-1))
    # Recharge counter — vendor read.c::seffect_charging: wand explodes at 7.
    recharged: jnp.ndarray = field(default_factory=lambda: jnp.int8(0))
    # Corpse age: game turn when corpse was created (-1 = not a corpse/unknown).
    # cite: vendor/nethack/src/eat.c::eatcorpse line 1885 peek_at_iced_corpse_age
    corpse_creation_turn: jnp.ndarray = field(
        default_factory=lambda: jnp.int32(-1)
    )
    # Poisoned-tin flag: tin is poisoned even when sealed.
    # cite: vendor/nethack/src/eat.c::consume_tin line 1537 tin->otrapped check
    tin_poisoned: jnp.ndarray = field(default_factory=lambda: jnp.bool_(False))
    # dknown: description known (item seen "up close").  Set TRUE on first
    # pickup/sight via observe_object (vendor o_init.c::observe_object lines
    # 441-451 — sets obj->dknown=1 and discovers the object type).  Vendor
    # objnam.c::xname uses dknown to gate "potion" vs "ruby potion" rendering.
    # cite: vendor/nethack/include/obj.h lines 109-112.
    dknown: jnp.ndarray = field(default_factory=lambda: jnp.bool_(False))
    # rknown: rustproofing status known.  Set TRUE when an identify path
    # reveals erodeproof / poison / charge state.  Vendor objnam.c:1183
    # ("rknown && oerodeproof") gates rustproof/fireproof display on this.
    # cite: vendor/nethack/include/obj.h line 114.
    rknown: jnp.ndarray = field(default_factory=lambda: jnp.bool_(False))
    # age: vendor obj->age (long) — repurposed by arti_invoke as a
    # per-object "tired until move N" cooldown timestamp.  Default 0 means
    # "never invoked" (age <= moves => ready).
    # cite: vendor/nethack/include/obj.h obj->age;
    #       vendor/nethack/src/artifact.c::arti_invoke_cost lines 2106-2127
    #       (obj->age vs svm.moves "tired" gate; obj->age = svm.moves + rnz(100)).
    age: jnp.ndarray = field(default_factory=lambda: jnp.int32(0))
    # artifact_idx: per-slot artifact identity (-1 == not an artifact).
    # Mirrors vendor obj->oartifact (uchar) which tags ordinary objects with
    # their artifact list index (artilist[oartifact]).  Needed so carried
    # artifact extrinsics (cspfx) can be ORed across the whole inventory
    # rather than only the wielded slot.
    # cite: vendor/nethack/include/obj.h obj->oartifact;
    #       vendor/nethack/src/artifact.c::set_artifact_intrinsic
    #       (loops over all worn/carried artifacts).
    artifact_idx: jnp.ndarray = field(default_factory=lambda: jnp.int8(-1))
    # oeaten: residual nutrition bytes remaining on a partially-eaten food
    # item (vendor obj.h ``Bitfield(oeaten, 5)``).  Nonzero ⇒ "partly eaten "
    # prefix in objnam.c::doname_base line 1506.  We only need the
    # zero/nonzero distinction for inv_strs rendering; gameplay use of the
    # counter is out of scope for this field.
    oeaten: jnp.ndarray = field(default_factory=lambda: jnp.int8(0))
    # opoisoned: weapon/ammo coated in poison.  Vendor obj.h
    # ``Bitfield(opoisoned, 1)``; renders as "poisoned " prefix in
    # objnam.c::doname_base line 1420 (WEAPON_CLASS branch).
    opoisoned: jnp.ndarray = field(default_factory=lambda: jnp.bool_(False))


def make_empty_item() -> Item:
    """Return a zeroed Item representing an empty inventory slot."""
    return Item(
        category=jnp.int8(0), type_id=jnp.int16(0), buc_status=jnp.int8(0),
        enchantment=jnp.int8(0), charges=jnp.int8(0), identified=jnp.bool_(False),
        quantity=jnp.int16(0), weight=jnp.int32(0), ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False), greased=jnp.bool_(False),
        oeroded=jnp.int8(0), oeroded2=jnp.int8(0), oerodeproof=jnp.bool_(False),
        bknown=jnp.bool_(False), lamplit=jnp.bool_(False), olocked=jnp.bool_(False),
        corpse_entry_idx=jnp.int16(-1),
        recharged=jnp.int8(0),
        corpse_creation_turn=jnp.int32(-1),
        tin_poisoned=jnp.bool_(False),
        dknown=jnp.bool_(False),
        rknown=jnp.bool_(False),
        age=jnp.int32(0),
        artifact_idx=jnp.int8(-1),
        oeaten=jnp.int8(0),
        opoisoned=jnp.bool_(False),
    )


def make_item(
    category: int,
    type_id: int,
    quantity: int = 1,
    weight: int = 0,
    ac_bonus: int = 0,
    enchantment: int = 0,
    is_two_handed: bool = False,
    buc_status: int = 0,
    oeroded: int = 0,
    oeroded2: int = 0,
    oerodeproof: bool = False,
    corpse_entry_idx: int = -1,
    corpse_creation_turn: int = -1,
    tin_poisoned: bool = False,
    identified: bool = False,
    bknown: bool = False,
    dknown: bool = False,
    rknown: bool = False,
) -> Item:
    """Construct a concrete Item with given fields (Python-side helper).

    wave17h P0 (IDENTIFICATION #3): default identified=False to match vendor
    starting-item behavior (cite invent.c:2637-2647). Existing callers that
    rely on starting inventory being identified can pass identified=True.

    bknown/dknown/rknown default False (freshly-generated objects).  Vendor
    marks starting inventory with ``obj->dknown = obj->bknown = obj->rknown
    = 1`` (vendor/nethack/src/u_init.c:1218); starting-item helpers pass
    these True so the BUC prefix ("uncursed"/"blessed") renders.
    """
    return Item(
        category=jnp.int8(category),
        type_id=jnp.int16(type_id),
        buc_status=jnp.int8(buc_status),
        enchantment=jnp.int8(enchantment),
        charges=jnp.int8(0),
        identified=jnp.bool_(identified),
        quantity=jnp.int16(quantity),
        weight=jnp.int32(weight),
        ac_bonus=jnp.int8(ac_bonus),
        is_two_handed=jnp.bool_(is_two_handed),
        greased=jnp.bool_(False),
        oeroded=jnp.int8(oeroded),
        oeroded2=jnp.int8(oeroded2),
        oerodeproof=jnp.bool_(oerodeproof),
        bknown=jnp.bool_(bknown),
        lamplit=jnp.bool_(False),
        olocked=jnp.bool_(False),
        corpse_entry_idx=jnp.int16(corpse_entry_idx),
        recharged=jnp.int8(0),
        corpse_creation_turn=jnp.int32(corpse_creation_turn),
        tin_poisoned=jnp.bool_(tin_poisoned),
        dknown=jnp.bool_(dknown),
        rknown=jnp.bool_(rknown),
        age=jnp.int32(0),
        artifact_idx=jnp.int8(-1),
        oeaten=jnp.int8(0),
        opoisoned=jnp.bool_(False),
    )


# ---------------------------------------------------------------------------
# Helpers to build batched Item arrays from a list of items
# ---------------------------------------------------------------------------

def _items_from_list(item_list: list) -> Item:
    """Stack a Python list of Items into a batched Item of shape [N]."""
    padded = list(item_list) + [make_empty_item()] * (MAX_INVENTORY_SLOTS - len(item_list))
    return _stack_items(padded)


def _stack_items(items: list) -> Item:
    """Stack a fixed-length list of Items into a single batched Item."""
    return Item(
        category=jnp.array([int(it.category) for it in items], dtype=jnp.int8),
        type_id=jnp.array([int(it.type_id) for it in items], dtype=jnp.int16),
        buc_status=jnp.array([int(it.buc_status) for it in items], dtype=jnp.int8),
        enchantment=jnp.array([int(it.enchantment) for it in items], dtype=jnp.int8),
        charges=jnp.array([int(it.charges) for it in items], dtype=jnp.int8),
        identified=jnp.array([bool(it.identified) for it in items], dtype=jnp.bool_),
        quantity=jnp.array([int(it.quantity) for it in items], dtype=jnp.int16),
        weight=jnp.array([int(it.weight) for it in items], dtype=jnp.int32),
        ac_bonus=jnp.array([int(it.ac_bonus) for it in items], dtype=jnp.int8),
        is_two_handed=jnp.array([bool(it.is_two_handed) for it in items], dtype=jnp.bool_),
        greased=jnp.array([bool(it.greased) for it in items], dtype=jnp.bool_),
        oeroded=jnp.array([int(it.oeroded) for it in items], dtype=jnp.int8),
        oeroded2=jnp.array([int(it.oeroded2) for it in items], dtype=jnp.int8),
        oerodeproof=jnp.array([bool(it.oerodeproof) for it in items], dtype=jnp.bool_),
        bknown=jnp.array([bool(it.bknown) for it in items], dtype=jnp.bool_),
        lamplit=jnp.array([bool(it.lamplit) for it in items], dtype=jnp.bool_),
        olocked=jnp.array([bool(it.olocked) for it in items], dtype=jnp.bool_),
        corpse_entry_idx=jnp.array(
            [int(it.corpse_entry_idx) for it in items], dtype=jnp.int16
        ),
        recharged=jnp.array([int(it.recharged) for it in items], dtype=jnp.int8),
        corpse_creation_turn=jnp.array(
            [int(it.corpse_creation_turn) for it in items], dtype=jnp.int32
        ),
        tin_poisoned=jnp.array(
            [bool(it.tin_poisoned) for it in items], dtype=jnp.bool_
        ),
        dknown=jnp.array([bool(it.dknown) for it in items], dtype=jnp.bool_),
        rknown=jnp.array([bool(it.rknown) for it in items], dtype=jnp.bool_),
        age=jnp.array([int(it.age) for it in items], dtype=jnp.int32),
        artifact_idx=jnp.array(
            [int(it.artifact_idx) for it in items], dtype=jnp.int8
        ),
        oeaten=jnp.array([int(it.oeaten) for it in items], dtype=jnp.int8),
        opoisoned=jnp.array(
            [bool(it.opoisoned) for it in items], dtype=jnp.bool_
        ),
    )


def _empty_items_array() -> Item:
    """Return a batched Item array of shape [MAX_INVENTORY_SLOTS], all empty."""
    n = MAX_INVENTORY_SLOTS
    return Item(
        category=jnp.zeros((n,), dtype=jnp.int8),
        type_id=jnp.zeros((n,), dtype=jnp.int16),
        buc_status=jnp.zeros((n,), dtype=jnp.int8),
        enchantment=jnp.zeros((n,), dtype=jnp.int8),
        charges=jnp.zeros((n,), dtype=jnp.int8),
        identified=jnp.zeros((n,), dtype=jnp.bool_),
        quantity=jnp.zeros((n,), dtype=jnp.int16),
        weight=jnp.zeros((n,), dtype=jnp.int32),
        ac_bonus=jnp.zeros((n,), dtype=jnp.int8),
        is_two_handed=jnp.zeros((n,), dtype=jnp.bool_),
        greased=jnp.zeros((n,), dtype=jnp.bool_),
        oeroded=jnp.zeros((n,), dtype=jnp.int8),
        oeroded2=jnp.zeros((n,), dtype=jnp.int8),
        oerodeproof=jnp.zeros((n,), dtype=jnp.bool_),
        bknown=jnp.zeros((n,), dtype=jnp.bool_),
        lamplit=jnp.zeros((n,), dtype=jnp.bool_),
        olocked=jnp.zeros((n,), dtype=jnp.bool_),
        corpse_entry_idx=jnp.full((n,), -1, dtype=jnp.int16),
        recharged=jnp.zeros((n,), dtype=jnp.int8),
        corpse_creation_turn=jnp.full((n,), -1, dtype=jnp.int32),
        tin_poisoned=jnp.zeros((n,), dtype=jnp.bool_),
        dknown=jnp.zeros((n,), dtype=jnp.bool_),
        rknown=jnp.zeros((n,), dtype=jnp.bool_),
        age=jnp.zeros((n,), dtype=jnp.int32),
        artifact_idx=jnp.full((n,), -1, dtype=jnp.int8),
        oeaten=jnp.zeros((n,), dtype=jnp.int8),
        opoisoned=jnp.zeros((n,), dtype=jnp.bool_),
    )


def _empty_ground_items_array(n_branches: int, max_levels: int, map_h: int, map_w: int) -> Item:
    """Return a ground_items array of shape [n_branches, max_levels, map_h, map_w, MAX_GROUND_STACK]."""
    shape = (n_branches, max_levels, map_h, map_w, MAX_GROUND_STACK)
    return Item(
        category=jnp.zeros(shape, dtype=jnp.int8),
        type_id=jnp.zeros(shape, dtype=jnp.int16),
        buc_status=jnp.zeros(shape, dtype=jnp.int8),
        enchantment=jnp.zeros(shape, dtype=jnp.int8),
        charges=jnp.zeros(shape, dtype=jnp.int8),
        identified=jnp.zeros(shape, dtype=jnp.bool_),
        quantity=jnp.zeros(shape, dtype=jnp.int16),
        weight=jnp.zeros(shape, dtype=jnp.int32),
        ac_bonus=jnp.zeros(shape, dtype=jnp.int8),
        is_two_handed=jnp.zeros(shape, dtype=jnp.bool_),
        greased=jnp.zeros(shape, dtype=jnp.bool_),
        oeroded=jnp.zeros(shape, dtype=jnp.int8),
        oeroded2=jnp.zeros(shape, dtype=jnp.int8),
        oerodeproof=jnp.zeros(shape, dtype=jnp.bool_),
        bknown=jnp.zeros(shape, dtype=jnp.bool_),
        lamplit=jnp.zeros(shape, dtype=jnp.bool_),
        olocked=jnp.zeros(shape, dtype=jnp.bool_),
        corpse_entry_idx=jnp.full(shape, -1, dtype=jnp.int16),
        recharged=jnp.zeros(shape, dtype=jnp.int8),
        corpse_creation_turn=jnp.full(shape, -1, dtype=jnp.int32),
        tin_poisoned=jnp.zeros(shape, dtype=jnp.bool_),
        dknown=jnp.zeros(shape, dtype=jnp.bool_),
        rknown=jnp.zeros(shape, dtype=jnp.bool_),
        age=jnp.zeros(shape, dtype=jnp.int32),
        artifact_idx=jnp.full(shape, -1, dtype=jnp.int8),
        oeaten=jnp.zeros(shape, dtype=jnp.int8),
        opoisoned=jnp.zeros(shape, dtype=jnp.bool_),
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@struct.dataclass
class InventoryState:
    """Persistent inventory state for a single hero.

    All slot-index fields use -1 to mean "empty / bare hands / none".

    Fields
    ------
    items        : Item array of length MAX_INVENTORY_SLOTS.
                   Empty slots have category == 0 (NONE).
    wielded      : index into items for the wielded weapon (-1 = bare hands).
    off_hand     : index into items for the off-hand weapon (-1 = none).
                   Two-weapon combat (wield.c:dotwoweapon).
    alternate_weapon_slot : Wave 5 — index into items for the two-weapon
                   alternate weapon (-1 = none).  Distinct from off_hand
                   so callers can preserve the bookkeeping that 'X' toggle
                   updates without disturbing off-hand wear semantics.
    swap_weapon  : index into items for the uswapwep slot (-1 = none).
                   vendor/nethack/src/wield.c::setuswapwep; doname emits
                   " (alternate weapon; not wielded)" when set and
                   !u.twoweap.  Cite: vendor/nethack/src/objnam.c:1613-1620.
    worn_armor   : int8[N_ARMOR_SLOTS] — items index per ArmorSlot (-1 = none).
    worn_armor_ac_bonus : int8[N_ARMOR_SLOTS] — cached AC bonus contribution
                   per worn armor slot.  Mirrors vendor/nethack/src/do_wear.c::
                   find_ac which sums each slot's ARM_BONUS.  Updated by
                   wear_armor / take_off_armor.  Used by combat.compute_ac.
    worn_amulet  : items index for worn amulet (-1 = none).
    worn_rings   : int8[2] — items index for left/right ring finger (-1 = none).
    quiver       : items index for the auto-quiver slot (-1 = none).
                   (pickup.c / dothrow.c: auto-select ammunition)
    total_weight : cached sum of all carried item weights (int32, aum units).
    user_names   : int8[MAX_INVENTORY_SLOTS, USER_NAME_LEN] — per-slot
                   user-given names (Wave 6).  Mirrors NetHack's oname /
                   oextra storage (vendor/nethack/src/objnam.c::doname which
                   appends " named <name>").  A slot's name is "unset" when
                   user_names[slot, 0] == 0.
    """

    items: Item                    # [MAX_INVENTORY_SLOTS]
    wielded: jnp.ndarray           # scalar int8
    off_hand: jnp.ndarray          # scalar int8
    alternate_weapon_slot: jnp.ndarray  # scalar int8 (Wave 5)
    swap_weapon: jnp.ndarray       # scalar int8 — uswapwep slot (-1=none)
    worn_armor: jnp.ndarray        # [N_ARMOR_SLOTS] int8
    worn_armor_ac_bonus: jnp.ndarray  # [N_ARMOR_SLOTS] int8 (Wave 5)
    armor_stat_bonus: jnp.ndarray  # [6] int8 — additive [str,dex,con,int,wis,cha]
                                   # bonus sourced only from worn armor.
                                   # Recomputed by armor_effects.apply_armor_effects
                                   # after every wear/take-off (Wave 31).
    worn_amulet: jnp.ndarray       # scalar int8
    worn_rings: jnp.ndarray        # [2] int8
    quiver: jnp.ndarray            # scalar int8
    total_weight: jnp.ndarray      # scalar int32
    user_names: jnp.ndarray        # [MAX_INVENTORY_SLOTS, USER_NAME_LEN] int8
    wielded_artifact_idx: jnp.ndarray  # scalar int8 — wish._ARTIFACTS index (-1=none)
    # Cursed-stuck (welded) flags.
    # Cite: vendor/nethack/src/wield.c::welded() lines 1051-1058 —
    # a cursed wielded weapon is welded to the hand; do_wear.c line 1900
    # applies the same logic to armor/amulet/rings.
    welded: jnp.ndarray             # scalar bool — wielded weapon stuck
    worn_armor_welded: jnp.ndarray  # [N_ARMOR_SLOTS] bool
    worn_amulet_welded: jnp.ndarray  # scalar bool
    worn_rings_welded: jnp.ndarray   # [2] bool
    letters: jnp.ndarray            # int8[MAX_INVENTORY_SLOTS] — ASCII letter
                                    # per slot, 0=empty.  Vendor
                                    # invent.c::assigninvlet allocates a..z
                                    # first then A..Z; the letter sticks with
                                    # the item record while it's in inventory.

    @classmethod
    def empty(cls) -> "InventoryState":
        """Return a fully-empty InventoryState for a freshly created character."""
        return cls(
            items=_empty_items_array(),
            wielded=jnp.int8(-1),
            off_hand=jnp.int8(-1),
            alternate_weapon_slot=jnp.int8(-1),
            swap_weapon=jnp.int8(-1),
            worn_armor=jnp.full((N_ARMOR_SLOTS,), -1, dtype=jnp.int8),
            worn_armor_ac_bonus=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.int8),
            armor_stat_bonus=jnp.zeros((6,), dtype=jnp.int8),
            worn_amulet=jnp.int8(-1),
            worn_rings=jnp.full((2,), -1, dtype=jnp.int8),
            quiver=jnp.int8(-1),
            total_weight=jnp.int32(0),
            user_names=jnp.zeros((MAX_INVENTORY_SLOTS, USER_NAME_LEN), dtype=jnp.int8),
            wielded_artifact_idx=jnp.int8(-1),
            welded=jnp.bool_(False),
            worn_armor_welded=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.bool_),
            worn_amulet_welded=jnp.bool_(False),
            worn_rings_welded=jnp.zeros((2,), dtype=jnp.bool_),
            letters=jnp.zeros(MAX_INVENTORY_SLOTS, dtype=jnp.int8),
        )

    @classmethod
    def from_items(cls, item_list: list) -> "InventoryState":
        """Build an InventoryState pre-populated with a list of Items.

        Pads with empty items to MAX_INVENTORY_SLOTS.

        Per vendor invent.c::assigninvlet, each item entering inventory
        receives a persistent letter — lowercase a..z first, then A..Z.
        We assign letters positionally to the first N occupied slots,
        matching the canonical NetHack startup where slot 0 -> 'a',
        slot 1 -> 'b', etc.
        """
        # Build letters[] positionally: occupied slots 0..N-1 get a..z then A..Z.
        letters_list = []
        for i, item in enumerate(item_list):
            if i >= MAX_INVENTORY_SLOTS:
                break
            # Empty items (category 0) get letter 0.
            if int(item.category) == 0:
                letters_list.append(0)
            elif i < 26:
                letters_list.append(ord('a') + i)
            else:
                letters_list.append(ord('A') + (i - 26))
        # Pad to MAX_INVENTORY_SLOTS with 0 (empty letter).
        letters_list += [0] * (MAX_INVENTORY_SLOTS - len(letters_list))
        letters_arr = jnp.array(letters_list, dtype=jnp.int8)

        return cls(
            items=_items_from_list(item_list),
            wielded=jnp.int8(-1),
            off_hand=jnp.int8(-1),
            alternate_weapon_slot=jnp.int8(-1),
            swap_weapon=jnp.int8(-1),
            worn_armor=jnp.full((N_ARMOR_SLOTS,), -1, dtype=jnp.int8),
            worn_armor_ac_bonus=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.int8),
            armor_stat_bonus=jnp.zeros((6,), dtype=jnp.int8),
            worn_amulet=jnp.int8(-1),
            worn_rings=jnp.full((2,), -1, dtype=jnp.int8),
            quiver=jnp.int8(-1),
            total_weight=jnp.int32(0),
            user_names=jnp.zeros((MAX_INVENTORY_SLOTS, USER_NAME_LEN), dtype=jnp.int8),
            wielded_artifact_idx=jnp.int8(-1),
            welded=jnp.bool_(False),
            worn_armor_welded=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.bool_),
            worn_amulet_welded=jnp.bool_(False),
            worn_rings_welded=jnp.zeros((2,), dtype=jnp.bool_),
            letters=letters_arr,
        )


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def total_weight(items: Item) -> jnp.ndarray:
    """Sum carried item weights across all MAX_INVENTORY_SLOTS.

    JIT-compatible via lax.scan — no Python control flow over slots.

    Parameters
    ----------
    items : batched Item of shape [MAX_INVENTORY_SLOTS]

    Returns
    -------
    int32 total weight in aum units.
    """
    def _add_weight(acc, idx):
        occupied = items.category[idx] != 0
        w = jnp.where(occupied, items.weight[idx].astype(jnp.int32), jnp.int32(0))
        return acc + w, None

    total, _ = lax.scan(_add_weight, jnp.int32(0), jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32))
    return total


# ---------------------------------------------------------------------------
# AC computation
# ---------------------------------------------------------------------------

def compute_ac(items: Item, worn_armor: jnp.ndarray,
               worn_armor_ac_bonus: jnp.ndarray | None = None) -> jnp.ndarray:
    """Compute player AC from currently worn armor slots.

    NetHack formula (do_wear.c::find_ac): uac = 10 - sum(ARM_BONUS for each
    worn piece).  ARM_BONUS = objects[otyp].a_ac + spe (enchantment) -
    erosion_penalty.  Here we use item.ac_bonus + item.enchantment as the
    bonus, ignoring erosion (Wave 4).

    Wave 5: an optional ``worn_armor_ac_bonus`` array overrides the per-slot
    contribution.  When a slot's bonus is non-zero in this array, it is used
    in place of items[worn_armor[slot]].ac_bonus.  This allows callers to
    inject AC bonuses directly without populating Item records (used by
    test_armor_reduces_damage) and is the canonical cache updated by
    wear_armor / take_off_armor.

    Parameters
    ----------
    items      : batched Item of shape [MAX_INVENTORY_SLOTS]
    worn_armor : int8[N_ARMOR_SLOTS] — slot index per armor slot (-1 = empty)
    worn_armor_ac_bonus : optional int8[N_ARMOR_SLOTS] — per-slot AC bonus
                          override (default: derive from items.ac_bonus)

    Returns
    -------
    int32 player AC
    """
    use_cache = worn_armor_ac_bonus is not None

    def _sum_bonus(acc, slot_idx):
        item_idx = worn_armor[slot_idx].astype(jnp.int32)
        equipped = item_idx >= 0
        safe_idx = jnp.clip(item_idx, 0, MAX_INVENTORY_SLOTS - 1)
        item_bonus = items.ac_bonus[safe_idx].astype(jnp.int32)
        enchant = items.enchantment[safe_idx].astype(jnp.int32)
        # Use cached per-slot bonus when provided AND non-zero; otherwise
        # derive from the item record.  This lets callers set AC bonuses
        # directly via worn_armor_ac_bonus.
        if use_cache:
            cached = worn_armor_ac_bonus[slot_idx].astype(jnp.int32)
            chosen = jnp.where(cached != 0, cached, item_bonus)
            bonus = chosen + jnp.where(equipped, enchant, jnp.int32(0))
            contribution = jnp.where(
                equipped | (cached != 0),
                bonus,
                jnp.int32(0),
            )
        else:
            bonus = item_bonus + enchant
            contribution = jnp.where(equipped, bonus, jnp.int32(0))
        return acc + contribution, None

    total_bonus, _ = lax.scan(_sum_bonus, jnp.int32(0), jnp.arange(N_ARMOR_SLOTS, dtype=jnp.int32))
    return jnp.int32(BASE_AC) - total_bonus


# ---------------------------------------------------------------------------
# Weight cap helper (Audit L #12)
# ---------------------------------------------------------------------------

def weight_cap(state) -> jnp.ndarray:
    """Compute the player's carrying capacity in aum units.

    Mirrors vendor/nethack/src/hack.c::weight_cap lines 4295-4346:
        carrcap = WT_WEIGHTCAP_STRCON * (STR + CON) + WT_WEIGHTCAP_SPARE
        carrcap = min(carrcap, MAX_CARR_CAP)
        carrcap = max(carrcap, 1)  -- never return 0
    Levitation / Upolyd / Wounded_legs adjustments are not modeled here
    (parity gap documented in inventory.py).

    JIT-pure: arithmetic only.
    """
    strv = state.player_str.astype(jnp.int32)
    conv = state.player_con.astype(jnp.int32)
    cap  = jnp.int32(WT_WEIGHTCAP_STRCON) * (strv + conv) + jnp.int32(WT_WEIGHTCAP_SPARE)
    cap  = jnp.minimum(cap, jnp.int32(MAX_CARR_CAP))
    cap  = jnp.maximum(cap, jnp.int32(1))
    return cap


def _find_merge_slot(items: Item, in_cat, in_tid, in_buc, in_ench, in_oerodeproof) -> tuple:
    """Find an inventory slot that is mergeable with the incoming item.

    Mirrors vendor/nethack/src/invent.c::mergable (lines 4379-4460):
    same otyp, same cursed/blessed, same spe (enchantment), same
    oerodeproof.  We collapse to (category, type_id, buc, ench,
    oerodeproof) — the subset relevant for byte-equal pickup merging
    of common consumables (potions, scrolls, arrows, gems).

    Returns (found, slot) where slot is the chosen inventory index.
    """
    def _scan(carry, idx):
        found, slot = carry
        occupied = items.category[idx] != jnp.int8(0)
        match = (
            occupied
            & (items.category[idx]   == in_cat)
            & (items.type_id[idx]    == in_tid)
            & (items.buc_status[idx] == in_buc)
            & (items.enchantment[idx] == in_ench)
            & (items.oerodeproof[idx] == in_oerodeproof)
        )
        slot  = jnp.where(~found & match, idx, slot)
        found = found | match
        return (found, slot), None

    (found, slot), _ = lax.scan(
        _scan,
        (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32),
    )
    return found, slot


# ---------------------------------------------------------------------------
# Core inventory operations
# ---------------------------------------------------------------------------

def pickup(state, rng, ground_items: Item, branch: int, level: int) -> tuple:
    """Pick up the top item from the ground tile at player_pos.

    Reads item at ground_items[branch, level, row, col, 0] (top of stack).
    Stack-merging (Audit L #12):
      - Pre-scan inventory for a slot with matching (category, type_id,
        buc, enchantment, oerodeproof) — if found, add quantity instead
        of consuming a new slot.
      - Otherwise, use the lowest-index empty slot.
    Encumbrance refusal:
      - Compute new_weight = total_weight + item.weight*qty.  If
        new_weight > weight_cap (25*(STR+CON)+50, capped at 1000) AND the
        item is not a loadstone, refuse the pickup (state unchanged).
        Cite: vendor/nethack/src/pickup.c::lift_object lines 1705-1789.
    52-slot test:
      - inv_cnt >= invlet_basic AND no merge slot → refuse.  This is
        enforced implicitly by ``found`` since a full inventory yields
        no empty slot and no merge slot.  Loadstone (otyp 443) bypasses
        the 52-slot test (pickup.c:1721-1734).

    Canonical: vendor/nethack/src/pickup.c::pickup,
               vendor/nethack/src/invent.c::addinv,
               vendor/nethack/src/invent.c::merged 814-905,
               vendor/nethack/src/pickup.c::lift_object 1705-1789.

    Parameters
    ----------
    state        : EnvState
    rng          : JAX PRNG key (unused now; reserved for future weight checks)
    ground_items : Item of shape [n_branches, max_levels, map_h, map_w, MAX_GROUND_STACK]
    branch, level: current branch/level (Python ints for indexing)

    Returns
    -------
    (new_state, new_ground_items)
    """
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)

    # Ground item at top of stack (index 0)
    ground_cat  = ground_items.category[branch, level, row, col, 0]
    ground_tid  = ground_items.type_id[branch, level, row, col, 0]
    ground_buc  = ground_items.buc_status[branch, level, row, col, 0]
    ground_ench = ground_items.enchantment[branch, level, row, col, 0]
    ground_eprf = ground_items.oerodeproof[branch, level, row, col, 0]
    ground_wt   = ground_items.weight[branch, level, row, col, 0].astype(jnp.int32)
    ground_qty  = ground_items.quantity[branch, level, row, col, 0].astype(jnp.int32)

    has_item    = ground_cat != 0
    # Vendor pickup.c::pickup — gold is handled via add_to_money(quan), never
    # consumes an inventory letter. Detect COIN_CLASS here so we route the
    # quantity into u.umoney0 (state.player_gold) instead of an inventory slot.
    is_gold = has_item & (ground_cat == jnp.int8(ItemCategory.COIN))
    gold_qty = jnp.where(
        is_gold,
        ground_qty,
        jnp.int32(0),
    )
    is_loadstone = has_item & (ground_tid == jnp.int16(_LOADSTONE_TYPE_ID))

    # Merge-target scan (vendor invent.c::merged + mergable).
    merge_found, merge_slot = _find_merge_slot(
        state.inventory.items,
        ground_cat, ground_tid, ground_buc, ground_ench, ground_eprf,
    )

    # First-empty-slot scan
    def _find_slot(carry, idx):
        found, slot = carry
        is_empty = state.inventory.items.category[idx] == 0
        slot  = jnp.where(~found & is_empty, idx, slot)
        found = found | is_empty
        return (found, slot), None

    (empty_found, empty_slot), _ = lax.scan(
        _find_slot,
        (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32),
    )

    # Chosen target slot: merge slot wins; otherwise first empty.
    target_slot = jnp.where(merge_found, merge_slot, empty_slot)
    # Encumbrance: refuse if new total weight exceeds cap.
    # vendor pickup.c::lift_object 1756-1789; loadstone bypasses (1718-1734).
    # Note: ground_wt is the stack-total weight (objects[otyp].oc_weight*quan)
    # already, matching vendor obj->owt convention.
    cap = weight_cap(state)
    cur_wt = state.inventory.total_weight.astype(jnp.int32)
    new_total_wt_if_lifted = cur_wt + jnp.where(is_gold, jnp.int32(0), ground_wt)
    over_cap = new_total_wt_if_lifted > cap
    weight_ok = (~over_cap) | is_loadstone | is_gold

    # Slot availability: merge or empty; gold bypasses.  Loadstone bypasses
    # the empty-slot test only if no slot is free AND merge fails — vendor
    # pickup.c:1723 carrying(LOADSTONE) || merge_choice grants the lift.
    slot_ok = merge_found | empty_found | is_gold | is_loadstone

    can_pickup = has_item & slot_ok & weight_ok

    # Write ground item into the chosen inventory slot (skip for gold).
    safe_slot = jnp.clip(target_slot, 0, MAX_INVENTORY_SLOTS - 1)
    new_items = state.inventory.items
    write_slot = can_pickup & ~is_gold
    # Merge writes only update quantity + weight (vendor merged() lines 836-842).
    merge_write = write_slot & merge_found
    # Non-merge writes copy the full item record into target_slot.
    fresh_write = write_slot & ~merge_found

    # Quantity: merge → existing_qty + ground_qty; fresh → ground_qty.
    existing_qty = new_items.quantity[safe_slot].astype(jnp.int32)
    merged_qty   = existing_qty + ground_qty
    new_qty_val  = jnp.where(merge_write, merged_qty.astype(jnp.int16),
                   jnp.where(fresh_write, ground_qty.astype(jnp.int16),
                             new_items.quantity[safe_slot]))
    # Weight: merge → existing_wt + ground_wt (ground_wt is the stack-total
    # already, matching vendor obj->owt convention from weight()).
    existing_wt  = new_items.weight[safe_slot].astype(jnp.int32)
    merged_wt    = existing_wt + ground_wt
    new_wt_val   = jnp.where(merge_write, merged_wt,
                   jnp.where(fresh_write, ground_wt,
                             new_items.weight[safe_slot]))

    new_items = new_items.replace(
        category   = new_items.category.at[safe_slot].set(
            jnp.where(fresh_write, ground_cat, new_items.category[safe_slot])
        ),
        type_id    = new_items.type_id.at[safe_slot].set(
            jnp.where(fresh_write, ground_tid, new_items.type_id[safe_slot])
        ),
        buc_status = new_items.buc_status.at[safe_slot].set(
            jnp.where(fresh_write, ground_buc, new_items.buc_status[safe_slot])
        ),
        enchantment = new_items.enchantment.at[safe_slot].set(
            jnp.where(fresh_write, ground_ench, new_items.enchantment[safe_slot])
        ),
        charges    = new_items.charges.at[safe_slot].set(
            jnp.where(fresh_write, ground_items.charges[branch, level, row, col, 0], new_items.charges[safe_slot])
        ),
        identified = new_items.identified.at[safe_slot].set(
            jnp.where(fresh_write, ground_items.identified[branch, level, row, col, 0], new_items.identified[safe_slot])
        ),
        quantity   = new_items.quantity.at[safe_slot].set(new_qty_val),
        weight     = new_items.weight.at[safe_slot].set(new_wt_val),
        ac_bonus   = new_items.ac_bonus.at[safe_slot].set(
            jnp.where(fresh_write, ground_items.ac_bonus[branch, level, row, col, 0], new_items.ac_bonus[safe_slot])
        ),
        is_two_handed = new_items.is_two_handed.at[safe_slot].set(
            jnp.where(fresh_write, ground_items.is_two_handed[branch, level, row, col, 0], new_items.is_two_handed[safe_slot])
        ),
        # dknown: vendor pickup.c::pickup_object line 1818 calls
        # observe_object(obj) when !Blind, which sets obj->dknown=1
        # (o_init.c::observe_object lines 441-451).  We mirror that
        # here unconditionally — the item is "seen up close" on pickup.
        dknown = new_items.dknown.at[safe_slot].set(
            jnp.where(write_slot, jnp.bool_(True), new_items.dknown[safe_slot])
        ),
        # Preserve obj->oartifact across drop-and-pickup so the artifact's
        # cspfx extrinsics keep firing once carried.  Cite:
        # vendor/nethack/include/obj.h obj->oartifact.
        # Audit K wire-up follow-up: previously this field defaulted to -1
        # on every pickup, losing artifact identity.
        artifact_idx = new_items.artifact_idx.at[safe_slot].set(
            jnp.where(
                fresh_write,
                ground_items.artifact_idx[branch, level, row, col, 0],
                new_items.artifact_idx[safe_slot],
            )
        ),
    )

    # Letter assignment per vendor invent.c::assigninvlet (lines 693-732).
    # Mark all letters currently in use across the (existing) inventory,
    # then pick the lowest unused letter (a..z, then A..Z).  Merge writes
    # don't claim a new letter — the destination slot's letter is preserved.
    inuse_lower = jnp.zeros((26,), dtype=jnp.bool_)
    inuse_upper = jnp.zeros((26,), dtype=jnp.bool_)
    cur_letters = state.inventory.letters.astype(jnp.int32)
    def _mark_letters(carry, idx):
        il, iu = carry
        ch = cur_letters[idx]
        is_lower = (ch >= jnp.int32(ord('a'))) & (ch <= jnp.int32(ord('z')))
        is_upper = (ch >= jnp.int32(ord('A'))) & (ch <= jnp.int32(ord('Z')))
        l_idx = jnp.clip(ch - jnp.int32(ord('a')), 0, 25)
        u_idx = jnp.clip(ch - jnp.int32(ord('A')), 0, 25)
        il = jnp.where(is_lower, il.at[l_idx].set(True), il)
        iu = jnp.where(is_upper, iu.at[u_idx].set(True), iu)
        return (il, iu), None
    (inuse_lower, inuse_upper), _ = lax.scan(
        _mark_letters,
        (inuse_lower, inuse_upper),
        jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32),
    )
    # First unused lowercase, else first unused uppercase.
    def _first_free(carry, idx):
        found, slot = carry
        free_l = ~inuse_lower[idx]
        slot = jnp.where(~found & free_l, jnp.int32(ord('a')) + idx, slot)
        found = found | free_l
        return (found, slot), None
    (low_found, low_letter), _ = lax.scan(
        _first_free, (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(26, dtype=jnp.int32),
    )
    def _first_free_upper(carry, idx):
        found, slot = carry
        free_u = ~inuse_upper[idx]
        slot = jnp.where(~found & free_u, jnp.int32(ord('A')) + idx, slot)
        found = found | free_u
        return (found, slot), None
    (up_found, up_letter), _ = lax.scan(
        _first_free_upper, (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(26, dtype=jnp.int32),
    )
    chosen_letter = jnp.where(low_found, low_letter, up_letter).astype(jnp.int8)
    # Only stamp the letter on fresh writes; merge keeps the existing letter.
    new_letters = state.inventory.letters.at[safe_slot].set(
        jnp.where(fresh_write, chosen_letter, state.inventory.letters[safe_slot])
    )

    # Clear ground tile (set category to 0)
    new_ground_items = ground_items.replace(
        category=ground_items.category.at[branch, level, row, col, 0].set(
            jnp.where(can_pickup, jnp.int8(0), ground_items.category[branch, level, row, col, 0])
        )
    )

    new_inv = state.inventory.replace(
        items=new_items,
        total_weight=total_weight(new_items),
        letters=new_letters,
    )
    # Vendor pickup.c::pickup — gold goes to u.umoney0 (state.player_gold).
    new_gold = state.player_gold + gold_qty
    new_state = state.replace(inventory=new_inv, player_gold=new_gold)

    # Vault-guard witness hook (vendor vault.c::vault_gd_watching, line 1278).
    # If the player picks up gold from the vault floor while the guard can
    # see them, the guard treats it as the GD_EATGOLD activity and turns
    # hostile.  We approximate "on vault floor" by checking the per-level
    # vault_pos: when the pickup tile is within the 2x2 vault interior
    # (matches ``check_invault``'s Chebyshev<=1 gate at vault.py:138-140).
    from Nethax.nethax.subsystems.vault import (
        vault_gd_watching as _vault_witness,
        GD_EATGOLD as _GD_EATGOLD,
    )
    from Nethax.nethax.dungeon.branches import (
        MAX_LEVELS_PER_BRANCH as _MAX_LV,
    )
    flat_lv = (
        new_state.dungeon.current_branch.astype(jnp.int32) * jnp.int32(_MAX_LV)
        + (new_state.dungeon.current_level - jnp.int8(1)).astype(jnp.int32)
    )
    vp = new_state.features.vault_pos[flat_lv]
    vr, vc = vp[0].astype(jnp.int32), vp[1].astype(jnp.int32)
    pr_i = new_state.player_pos[0].astype(jnp.int32)
    pc_i = new_state.player_pos[1].astype(jnp.int32)
    in_vault = (vr >= jnp.int32(0)) & (
        (jnp.abs(pr_i - vr) <= jnp.int32(1))
        & (jnp.abs(pc_i - vc) <= jnp.int32(1))
    )
    witnessed = in_vault & (gold_qty > jnp.int32(0))
    new_state = jax.lax.cond(
        witnessed,
        lambda s: _vault_witness(s, _GD_EATGOLD),
        lambda s: s,
        new_state,
    )

    return new_state, new_ground_items


def drop(state, rng, ground_items: Item, branch: int, level: int, slot_idx: int) -> tuple:
    """Drop the item in inventory slot ``slot_idx`` onto the ground at player_pos.

    Drop preconditions (Audit L #13) — mirrors vendor do.c::drop 714-780
    and do.c::canletgo 665-711:
      - Cursed loadstone cannot be dropped (canletgo 685-699).
      - Welded uwep cannot be dropped (canletgo 672-684): if slot_idx is
        the wielded slot AND ``inventory.welded`` is True, refuse.
      - Levitation: if status.intrinsics[LEVITATION] is set, refuse
        (do.c:758-772 ``can_reach_floor`` returns False under Levitation,
        which in vendor calls ``hitfloor``; we approximate by refusing
        the drop to leave the slot intact).
      - Altar tile: call ``features.drop_at_altar`` which mutates the
        item's BUC per vendor pray.c::doaltar (called from dropx 786-796).
        The item is still placed on the ground stack as normal.
      - Ring-on-sink: vendor do.c:753-756 routes RING on SINK tile to
        ``dosinkring`` (silent removal + identification side-effects).
        DEFERRED — TileType has no SINK entry in the internal enum
        (constants/tiles.py:18-50); routing is documented but not
        functional until the SINK tile is modeled.

    Ground-stack merging (Audit L #13 closing): when the drop target tile
    already holds a matching item (same category/type_id/BUC/enchantment),
    merge into the existing stack rather than consuming a new ground slot.
    Cite: vendor/nethack/src/invent.c::merged (ground side of the same
    mergable() predicate).

    Canonical: vendor/nethack/src/do.c::drop 714-780,
               vendor/nethack/src/do.c::canletgo 665-711,
               vendor/nethack/src/do.c::dropx 786-796.

    Returns
    -------
    (new_state, new_ground_items)
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intrinsic
    from Nethax.nethax.constants.tiles import TileType as _TileType
    from Nethax.nethax.subsystems.features import drop_at_altar as _drop_at_altar

    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)
    slot_idx = jnp.int32(slot_idx)

    has_item = state.inventory.items.category[slot_idx] != 0

    # Cursed loadstone cannot be dropped.
    # Cite: vendor/nethack/src/do.c::canletgo line 685-699 — cursed loadstone
    # refuses with "For some reason, you cannot drop the stone!".
    LOADSTONE_TYPE_ID = jnp.int16(_LOADSTONE_TYPE_ID)
    CURSED = jnp.int8(1)
    is_cursed_loadstone = (
        (state.inventory.items.type_id[slot_idx] == LOADSTONE_TYPE_ID)
        & (state.inventory.items.buc_status[slot_idx] == CURSED)
    )

    # Welded uwep: if slot is the wielded weapon and the welded flag is on,
    # refuse the drop.  Cite: vendor/nethack/src/do.c::canletgo 672-684 and
    # do.c::drop 722-728.
    is_wielded_slot = (
        slot_idx == state.inventory.wielded.astype(jnp.int32)
    ) & (state.inventory.wielded.astype(jnp.int32) >= jnp.int32(0))
    welded_block = is_wielded_slot & state.inventory.welded

    # Levitation block: vendor do.c:758 — under Levitation, drop calls
    # hitfloor() and the item leaves inventory but never lands on the
    # current tile.  We refuse the drop to keep state predictable
    # (parity gap documented above).
    levitating = state.status.intrinsics[int(_Intrinsic.LEVITATION)]

    has_item = has_item & ~is_cursed_loadstone & ~welded_block & ~levitating

    # Item identity for ground-stack merge match.
    in_cat  = state.inventory.items.category[slot_idx]
    in_tid  = state.inventory.items.type_id[slot_idx]
    in_buc  = state.inventory.items.buc_status[slot_idx]
    in_ench = state.inventory.items.enchantment[slot_idx]
    in_eprf = state.inventory.items.oerodeproof[slot_idx]
    in_qty  = state.inventory.items.quantity[slot_idx].astype(jnp.int32)
    in_wt   = state.inventory.items.weight[slot_idx].astype(jnp.int32)

    # Scan ground stack for (a) first empty slot, (b) first mergeable slot.
    def _scan(carry, stack_idx):
        empty_found, empty_pos, merge_found, merge_pos = carry
        cat_here = ground_items.category[branch, level, row, col, stack_idx]
        is_empty = cat_here == jnp.int8(0)
        is_match = (
            (~is_empty)
            & (cat_here == in_cat)
            & (ground_items.type_id[branch, level, row, col, stack_idx]    == in_tid)
            & (ground_items.buc_status[branch, level, row, col, stack_idx] == in_buc)
            & (ground_items.enchantment[branch, level, row, col, stack_idx] == in_ench)
            & (ground_items.oerodeproof[branch, level, row, col, stack_idx] == in_eprf)
        )
        empty_pos = jnp.where(~empty_found & is_empty, stack_idx, empty_pos)
        empty_found = empty_found | is_empty
        merge_pos = jnp.where(~merge_found & is_match, stack_idx, merge_pos)
        merge_found = merge_found | is_match
        return (empty_found, empty_pos, merge_found, merge_pos), None

    (g_empty_found, g_empty_pos, g_merge_found, g_merge_pos), _ = lax.scan(
        _scan,
        (jnp.bool_(False), jnp.int32(0), jnp.bool_(False), jnp.int32(0)),
        jnp.arange(MAX_GROUND_STACK, dtype=jnp.int32),
    )

    # Target ground slot: merge first, else empty.
    g_target = jnp.where(g_merge_found, g_merge_pos, g_empty_pos)
    g_slot_ok = g_merge_found | g_empty_found
    can_drop = has_item & g_slot_ok
    safe_gs  = jnp.clip(g_target, 0, MAX_GROUND_STACK - 1)

    # Altar BUC mutation (vendor do.c::dropx 786-796 -> doaltarobj).
    # ``drop_at_altar`` mutates inventory.items.buc_status BEFORE the item
    # leaves inventory; we read the (possibly updated) BUC back into the
    # ground stack below.  Only fires on ALTAR tile and when can_drop.
    here_tile = state.terrain[branch, level, row, col].astype(jnp.int32)
    on_altar = (here_tile == jnp.int32(int(_TileType.ALTAR))) & can_drop
    state_altared = jax.lax.cond(
        on_altar,
        lambda s: _drop_at_altar(s, slot_idx),
        lambda s: s,
        state,
    )
    inv = state_altared.inventory.items

    # For merge writes: only quantity + weight need update; identity already matches.
    merge_write = can_drop & g_merge_found
    fresh_write = can_drop & ~g_merge_found

    # Helper: write inventory field into ground stack at safe_gs.
    def _set_ground(field_ground, field_inv):
        return field_ground.at[branch, level, row, col, safe_gs].set(
            jnp.where(fresh_write, field_inv[slot_idx],
                      field_ground[branch, level, row, col, safe_gs])
        )

    # Quantity/weight: merge adds, fresh copies.
    g_existing_qty = ground_items.quantity[branch, level, row, col, safe_gs].astype(jnp.int32)
    g_existing_wt  = ground_items.weight[branch, level, row, col, safe_gs].astype(jnp.int32)
    merged_qty = (g_existing_qty + in_qty).astype(jnp.int16)
    merged_wt  = (g_existing_wt + in_wt).astype(jnp.int32)

    new_qty_at_pos = jnp.where(
        merge_write, merged_qty,
        jnp.where(fresh_write, inv.quantity[slot_idx],
                  ground_items.quantity[branch, level, row, col, safe_gs])
    )
    new_wt_at_pos = jnp.where(
        merge_write, merged_wt,
        jnp.where(fresh_write, inv.weight[slot_idx],
                  ground_items.weight[branch, level, row, col, safe_gs])
    )

    new_ground = ground_items.replace(
        category    = _set_ground(ground_items.category,    inv.category),
        type_id     = _set_ground(ground_items.type_id,     inv.type_id),
        buc_status  = _set_ground(ground_items.buc_status,  inv.buc_status),
        enchantment = _set_ground(ground_items.enchantment, inv.enchantment),
        charges     = _set_ground(ground_items.charges,     inv.charges),
        identified  = _set_ground(ground_items.identified,  inv.identified),
        quantity    = ground_items.quantity.at[branch, level, row, col, safe_gs].set(new_qty_at_pos),
        weight      = ground_items.weight.at[branch, level, row, col, safe_gs].set(new_wt_at_pos),
        ac_bonus    = _set_ground(ground_items.ac_bonus,    inv.ac_bonus),
        is_two_handed = _set_ground(ground_items.is_two_handed, inv.is_two_handed),
        # Preserve obj->oartifact across the drop (Audit K wire-up).  Without
        # this, dropping an artifact loses its identity — subsequent pickup
        # would not re-grant cspfx extrinsics.  Cite: vendor/nethack/include/
        # obj.h obj->oartifact.
        artifact_idx = _set_ground(ground_items.artifact_idx, inv.artifact_idx),
    )

    # Zero the inventory slot (item leaves inventory regardless of merge/fresh).
    new_items = inv.replace(
        category   = inv.category.at[slot_idx].set(jnp.where(can_drop, jnp.int8(0), inv.category[slot_idx])),
        type_id    = inv.type_id.at[slot_idx].set(jnp.where(can_drop, jnp.int16(0), inv.type_id[slot_idx])),
        buc_status = inv.buc_status.at[slot_idx].set(jnp.where(can_drop, jnp.int8(0), inv.buc_status[slot_idx])),
        enchantment= inv.enchantment.at[slot_idx].set(jnp.where(can_drop, jnp.int8(0), inv.enchantment[slot_idx])),
        charges    = inv.charges.at[slot_idx].set(jnp.where(can_drop, jnp.int8(0), inv.charges[slot_idx])),
        identified = inv.identified.at[slot_idx].set(jnp.where(can_drop, jnp.bool_(False), inv.identified[slot_idx])),
        quantity   = inv.quantity.at[slot_idx].set(jnp.where(can_drop, jnp.int16(0), inv.quantity[slot_idx])),
        weight     = inv.weight.at[slot_idx].set(jnp.where(can_drop, jnp.int32(0), inv.weight[slot_idx])),
        ac_bonus   = inv.ac_bonus.at[slot_idx].set(jnp.where(can_drop, jnp.int8(0), inv.ac_bonus[slot_idx])),
        is_two_handed = inv.is_two_handed.at[slot_idx].set(jnp.where(can_drop, jnp.bool_(False), inv.is_two_handed[slot_idx])),
        # Clear the dropped slot's artifact_idx back to -1.
        artifact_idx = inv.artifact_idx.at[slot_idx].set(
            jnp.where(can_drop, jnp.int8(-1), inv.artifact_idx[slot_idx])
        ),
    )

    # Clear the letter for the now-empty slot (vendor invent.c::freeinv
    # nulls the obj's invlet implicitly as the obj detaches from gi.invent).
    # The next pickup of any item will receive the lowest free letter, so
    # the cleared letter naturally becomes available again.
    new_letters = state_altared.inventory.letters.at[slot_idx].set(
        jnp.where(can_drop, jnp.int8(0), state_altared.inventory.letters[slot_idx])
    )

    new_inv = state_altared.inventory.replace(
        items=new_items,
        total_weight=total_weight(new_items),
        letters=new_letters,
    )
    new_state = state_altared.replace(inventory=new_inv)
    return new_state, new_ground


# Cockatrice / chickatrice monster indices (Nethax PM ordering — see
# constants/monster_entries/chunk1.py lines 196, 215).  These are the
# only entries for which vendor's touch_petrifies() returns TRUE.
# Cite: vendor/nethack/include/mondata.h lines 200-201.
_PM_CHICKATRICE: int = 9
_PM_COCKATRICE:  int = 10

# CORPSE otyp (vendor/nethack/include/objects.h; mirrored in
# apply_tools._CORPSE_TYPE_ID).
_CORPSE_TYPE_ID: int = 240

# Sunsword artifact index (wish._ARTIFACTS row 32; artilist.h line 209).
# artifact_light(uwep) returns TRUE only for SUNSWORD on the weapon path.
# Cite: vendor/nethack/src/artifact.c::artifact_light lines 2264-2275.
_ARTI_SUNSWORD: int = 32


def wield(state, slot_idx: int):
    """Wield the item in slot_idx as the primary weapon.

    If the item is two-handed and a shield is equipped (worn_armor[SHIELD]),
    the shield is unequipped (worn_armor[SHIELD] set to -1).

    If the item is cursed (buc_status == 1), sets inventory.welded = True —
    the weapon is stuck until uncursed.
    Cite: vendor/nethack/src/wield.c::welded() lines 1051-1058.

    Petrification gate: wielding a cockatrice/chickatrice corpse with bare
    hands and no Stone_resistance instakills the hero (STONING death).
    Cite: vendor/nethack/src/wield.c::cant_wield_corpse lines 137-153 and
          ready_weapon line 183.

    Sunsword begin_burn: wielding Sunsword sets lamplit=True on the slot
    (artifact_light(wep) && !wep->lamplit → begin_burn(wep, FALSE)).
    Cite: vendor/nethack/src/wield.c::ready_weapon lines 245-250;
          vendor/nethack/src/artifact.c::artifact_light lines 2264-2275.

    Canonical: vendor/nethack/src/wield.c::wieldwep

    Parameters
    ----------
    state    : EnvState
    slot_idx : inventory slot index to wield

    Returns
    -------
    new_state
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic, TimedStatus
    from Nethax.nethax.subsystems.scoring import DeathCause

    slot_idx = jnp.int8(slot_idx)
    slot_i32 = slot_idx.astype(jnp.int32)
    has_item = state.inventory.items.category[slot_i32] != 0

    # ---- Petrification gate (vendor wield.c:183 + cant_wield_corpse:142) -----
    items = state.inventory.items
    is_corpse = has_item & (items.type_id[slot_i32].astype(jnp.int32)
                            == jnp.int32(_CORPSE_TYPE_ID))
    cnm = items.corpse_entry_idx[slot_i32].astype(jnp.int32)
    is_petrify_corpse = is_corpse & (
        (cnm == jnp.int32(_PM_COCKATRICE))
        | (cnm == jnp.int32(_PM_CHICKATRICE))
    )
    gloves_slot = state.inventory.worn_armor[int(ArmorSlot.GLOVES)].astype(jnp.int32)
    has_gloves = gloves_slot >= jnp.int32(0)
    stone_res = (
        state.status.intrinsics[int(Intrinsic.RESIST_STONE)]
        | (state.status.timed_intrinsics[int(Intrinsic.RESIST_STONE)] > 0)
    )
    petrify = is_petrify_corpse & (~has_gloves) & (~stone_res)

    # Instapetrify when petrify gate trips: zero HP + done + STONING cause.
    new_hp = jnp.where(petrify, jnp.int32(0), state.player_hp)
    new_done = state.done | petrify
    new_cause = jnp.where(
        petrify,
        jnp.int8(int(DeathCause.STONING)),
        state.scoring.death_cause,
    )
    new_scoring = state.scoring.replace(death_cause=new_cause)
    cur_stoned = state.status.timed_statuses[int(TimedStatus.STONED)].astype(jnp.int32)
    new_stoned = jnp.where(petrify, jnp.int32(1), cur_stoned)

    # Block the wield if the corpse is petrifying (vendor cant_wield_corpse
    # returns TRUE on the petrify path, so wield is skipped entirely).
    can_wield = has_item & ~is_petrify_corpse

    new_wielded = jnp.where(can_wield, slot_idx, state.inventory.wielded)

    # Two-handed: unequip shield if present
    is_two_handed = state.inventory.items.is_two_handed[slot_i32]
    shield_slot   = jnp.int32(ArmorSlot.SHIELD)
    new_worn_armor = jnp.where(
        can_wield & is_two_handed,
        state.inventory.worn_armor.at[shield_slot].set(jnp.int8(-1)),
        state.inventory.worn_armor,
    )

    # Cursed weapon welds to hand.
    CURSED = jnp.int8(1)
    is_cursed = state.inventory.items.buc_status[slot_i32] == CURSED
    new_welded = jnp.where(can_wield & is_cursed, jnp.bool_(True), state.inventory.welded)

    # Sunsword begin_burn (vendor wield.c:245-250).  artifact_light(wep)
    # returns TRUE only for SUNSWORD on the weapon path.
    is_sunsword = (
        state.inventory.items.artifact_idx[slot_i32].astype(jnp.int32)
        == jnp.int32(_ARTI_SUNSWORD)
    )
    cur_lamplit = state.inventory.items.lamplit[slot_i32]
    new_lamplit = state.inventory.items.lamplit.at[slot_i32].set(
        jnp.where(can_wield & is_sunsword, jnp.bool_(True), cur_lamplit)
    )
    new_items = state.inventory.items.replace(lamplit=new_lamplit)

    new_statuses = state.status.timed_statuses.at[int(TimedStatus.STONED)].set(new_stoned)
    new_status = state.status.replace(timed_statuses=new_statuses)

    new_inv = state.inventory.replace(
        items=new_items,
        wielded=new_wielded,
        worn_armor=new_worn_armor,
        welded=new_welded,
    )
    return state.replace(
        inventory=new_inv,
        player_hp=new_hp,
        done=new_done,
        scoring=new_scoring,
        status=new_status,
    )


def unwield(state):
    """Lay down the wielded weapon (return to bare hands).

    No-op if inventory.welded == True (cursed weapon stuck to hand).
    Cite: vendor/nethack/src/wield.c::welded() — cannot unwield while welded.

    Sunsword end_burn: unwielding Sunsword clears lamplit on the slot.
    Cite: vendor/nethack/src/wield.c::setuwep lines 114-119.

    Returns
    -------
    new_state
    """
    can_unwield = ~state.inventory.welded
    # End-burn the previously-wielded slot if it was Sunsword.
    prev_slot = state.inventory.wielded.astype(jnp.int32)
    prev_safe = jnp.clip(prev_slot, 0, MAX_INVENTORY_SLOTS - 1)
    was_sunsword = (prev_slot >= jnp.int32(0)) & (
        state.inventory.items.artifact_idx[prev_safe].astype(jnp.int32)
        == jnp.int32(_ARTI_SUNSWORD)
    )
    clear_lamp = can_unwield & was_sunsword
    cur_lamp = state.inventory.items.lamplit[prev_safe]
    new_lamp_at_slot = jnp.where(clear_lamp, jnp.bool_(False), cur_lamp)
    new_lamplit = state.inventory.items.lamplit.at[prev_safe].set(new_lamp_at_slot)
    new_items = state.inventory.items.replace(lamplit=new_lamplit)

    new_wielded = jnp.where(can_unwield, jnp.int8(-1), state.inventory.wielded)
    new_inv = state.inventory.replace(items=new_items, wielded=new_wielded)
    return state.replace(inventory=new_inv)


def wear_armor(state, slot_idx: int, armor_slot: ArmorSlot):
    """Wear the item in slot_idx in the given armor_slot.

    Updates player_ac via AC computation and caches the worn item's
    ac_bonus into inventory.worn_armor_ac_bonus[armor_slot] (Wave 5).

    If the item is cursed, sets worn_armor_welded[armor_slot] = True.
    Cite: vendor/nethack/src/do_wear.c line 1900 cursed check.

    Canonical: vendor/nethack/src/do_wear.c::dowearx

    Returns
    -------
    new_state with updated inventory.worn_armor, worn_armor_ac_bonus,
    worn_armor_welded, and player_ac.
    """
    slot_idx   = jnp.int8(slot_idx)
    slot_i32   = slot_idx.astype(jnp.int32)
    armor_i32  = jnp.int32(int(armor_slot))

    has_item   = state.inventory.items.category[slot_i32] != 0
    is_armor   = state.inventory.items.category[slot_i32] == jnp.int8(ItemCategory.ARMOR)
    can_wear   = has_item & is_armor

    new_worn_armor = jnp.where(
        can_wear,
        state.inventory.worn_armor.at[armor_i32].set(slot_idx),
        state.inventory.worn_armor,
    )
    # Cache per-slot AC bonus (vendor/nethack/src/do_wear.c::find_ac sums
    # each ARM_BONUS).  When unequipped (-1), bonus is 0.
    item_bonus = state.inventory.items.ac_bonus[slot_i32].astype(jnp.int8)
    new_worn_ac_bonus = jnp.where(
        can_wear,
        state.inventory.worn_armor_ac_bonus.at[armor_i32].set(item_bonus),
        state.inventory.worn_armor_ac_bonus,
    )
    # Cursed armor becomes stuck.
    CURSED = jnp.int8(1)
    is_cursed = state.inventory.items.buc_status[slot_i32] == CURSED
    new_worn_armor_welded = jnp.where(
        can_wear & is_cursed,
        state.inventory.worn_armor_welded.at[armor_i32].set(jnp.bool_(True)),
        state.inventory.worn_armor_welded,
    )
    new_ac = compute_ac(state.inventory.items, new_worn_armor)

    # wave17h P0 (IDENTIFICATION #2): use-ID on donning unknown armor.
    # Cite: vendor/nethack/src/do_wear.c lines 121-460 — wearing identifies
    # the type when its effect is observable (e.g. gauntlets of dex/str).
    new_items_id = jnp.where(
        can_wear,
        state.inventory.items.identified.at[slot_i32].set(jnp.bool_(True)),
        state.inventory.items.identified,
    )
    # rknown: wearing reveals erodeproof / poison-coating / charge state
    # (vendor objnam.c:1183 — rknown gates rustproof display).  When the
    # armor identifies on donning, rknown is also revealed.
    new_items_rknown = jnp.where(
        can_wear,
        state.inventory.items.rknown.at[slot_i32].set(jnp.bool_(True)),
        state.inventory.items.rknown,
    )
    item_type_id = state.inventory.items.type_id[slot_i32].astype(jnp.int32)
    type_mask    = state.identification.identified
    t_clip       = jnp.clip(item_type_id, jnp.int32(0), jnp.int32(type_mask.shape[0] - 1))
    new_type_mask = jnp.where(
        can_wear,
        type_mask.at[t_clip].set(jnp.bool_(True)),
        type_mask,
    )

    new_inv = state.inventory.replace(
        worn_armor=new_worn_armor,
        worn_armor_ac_bonus=new_worn_ac_bonus,
        worn_armor_welded=new_worn_armor_welded,
        items=state.inventory.items.replace(
            identified=new_items_id,
            rknown=new_items_rknown,
        ),
    )
    new_state = state.replace(
        inventory=new_inv,
        player_ac=new_ac,
        identification=state.identification.replace(identified=new_type_mask),
    )
    # Wave 31b: recompute armor-sourced intrinsics + stat bonuses.
    # cite: vendor/nethack/src/do_wear.c Boots_on/Cloak_on/Helmet_on/Gloves_on.
    from Nethax.nethax.subsystems.armor_effects import apply_armor_effects
    return apply_armor_effects(new_state)


def take_off_armor(state, armor_slot: ArmorSlot):
    """Remove the armor in armor_slot.

    No-op if worn_armor_welded[armor_slot] is True (cursed armor stuck).
    Cite: vendor/nethack/src/do_wear.c line 1900 — cursed armor blocked.

    Updates player_ac and zeros the cached AC bonus slot.

    Canonical: vendor/nethack/src/do_wear.c::dotakeoff

    Returns
    -------
    new_state with updated inventory.worn_armor, worn_armor_ac_bonus,
    and player_ac.
    """
    armor_i32 = jnp.int32(int(armor_slot))
    is_welded = state.inventory.worn_armor_welded[armor_i32]
    can_remove = ~is_welded

    new_worn_armor = jnp.where(
        can_remove,
        state.inventory.worn_armor.at[armor_i32].set(jnp.int8(-1)),
        state.inventory.worn_armor,
    )
    new_worn_ac_bonus = jnp.where(
        can_remove,
        state.inventory.worn_armor_ac_bonus.at[armor_i32].set(jnp.int8(0)),
        state.inventory.worn_armor_ac_bonus,
    )
    new_ac = compute_ac(state.inventory.items, new_worn_armor)

    new_inv = state.inventory.replace(
        worn_armor=new_worn_armor,
        worn_armor_ac_bonus=new_worn_ac_bonus,
    )
    new_state = state.replace(inventory=new_inv, player_ac=new_ac)
    # Wave 31b: recompute armor-sourced intrinsics + stat bonuses.
    # cite: vendor/nethack/src/do_wear.c Boots_off/Cloak_off/Helmet_off/Gloves_off.
    from Nethax.nethax.subsystems.armor_effects import apply_armor_effects
    return apply_armor_effects(new_state)


# ---------------------------------------------------------------------------
# Action handlers (top-level dispatch targets)
# ---------------------------------------------------------------------------

def handle_pickup(state, rng, ground_items: Item, branch: int, level: int) -> tuple:
    """Pickup action handler — pickup from current tile.

    Calls quest.on_artifact_picked_up when the picked-up item is the role's
    quest artifact (quest.c::artitouch ~127-134; Qstat(touched_artifact)=TRUE).

    Returns (new_state, new_ground_items).
    """
    new_state, new_gi = pickup(state, rng, ground_items, branch, level)

    # Quest artifact check: compare the ground item's type_id to the role's
    # artifact index before pickup.  JIT-pure: jax.lax.cond gates the update.
    from Nethax.nethax.subsystems.quest import on_artifact_picked_up, _ARTIFACT_IDX_BY_ROLE
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)
    picked_type_id = ground_items.type_id[branch, level, row, col, 0].astype(jnp.int16)
    role_idx = jnp.clip(state.player_role.astype(jnp.int32), 0, _ARTIFACT_IDX_BY_ROLE.shape[0] - 1)
    quest_art_id = _ARTIFACT_IDX_BY_ROLE[role_idx].astype(jnp.int16)
    is_quest_artifact = (picked_type_id == quest_art_id) & (picked_type_id > jnp.int16(0))
    new_state = jax.lax.cond(
        is_quest_artifact,
        on_artifact_picked_up,
        lambda s: s,
        new_state,
    )
    return new_state, new_gi


def handle_drop(state, rng, ground_items: Item, branch: int, level: int) -> tuple:
    """Drop action handler — drop first occupied inventory slot.

    Wave 4 will add item selection UI.

    Returns (new_state, new_ground_items).
    """
    # Find first occupied slot
    def _find_occupied(carry, idx):
        found, slot = carry
        occupied = state.inventory.items.category[idx] != 0
        slot  = jnp.where(~found & occupied, idx, slot)
        found = found | occupied
        return (found, slot), None

    (_, first_slot), _ = lax.scan(
        _find_occupied, (jnp.bool_(False), jnp.int32(0)), jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32)
    )
    return drop(state, rng, ground_items, branch, level, first_slot)


def handle_wield(state, rng):
    """Wield action handler — wield first weapon in inventory."""
    def _find_weapon(carry, idx):
        found, slot = carry
        is_weapon = state.inventory.items.category[idx] == jnp.int8(ItemCategory.WEAPON)
        slot  = jnp.where(~found & is_weapon, idx, slot)
        found = found | is_weapon
        return (found, slot), None

    (found_weapon, first_weapon), _ = lax.scan(
        _find_weapon, (jnp.bool_(False), jnp.int32(0)), jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32)
    )

    # NLE multi-key: prefer agent-chosen slot if it's a valid weapon.
    # Cite: vendor/nethack/src/wield.c::dowield calls getobj().
    from Nethax.nethax.subsystems.pending_action import resolve_slot
    chosen = resolve_slot(state, first_weapon)
    safe_chosen = jnp.clip(chosen, 0, MAX_INVENTORY_SLOTS - 1)
    chosen_is_weapon = (
        state.inventory.items.category[safe_chosen]
        == jnp.int8(ItemCategory.WEAPON)
    ) & (state.inventory.items.quantity[safe_chosen] > jnp.int16(0))
    weapon_slot = jnp.where(chosen_is_weapon, safe_chosen, first_weapon).astype(jnp.int32)
    new_state = wield(state, weapon_slot)
    # Clear wielded_artifact_idx when wielding via the action handler (no
    # artifact context available here; callers that grant artifacts set it
    # directly via state.inventory.replace(wielded_artifact_idx=...)).
    # Cite: vendor/nethack/src/artifact.c lines 880-885 (setworn clears
    # the W_WEP extrinsic for inv_prop when a new weapon is wielded).
    new_inv = new_state.inventory.replace(wielded_artifact_idx=jnp.int8(-1))
    new_state = new_state.replace(inventory=new_inv)
    from Nethax.nethax.subsystems.artifact_powers import apply_artifact_intrinsics
    new_state = apply_artifact_intrinsics(new_state)
    # Conduct: vendor/nethack/src/wield.c::wieldwep — WEAPONLESS broken when a
    # non-bare-hand weapon is wielded (insight.c ~2137, u.uconduct.weaphit).
    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated_if
    return mark_violated_if(new_state, int(Conduct.WEAPONLESS), found_weapon)


def handle_unwield(state, rng):
    """Unwield action handler — lay down wielded weapon (bare hands).

    No-op if the weapon is cursed-welded.
    Cite: vendor/nethack/src/wield.c::welded() — blocked while welded.
    """
    return unwield(state)


def handle_wear(state, rng):
    """Wear action handler — wear armor at pending_action_slot (or first
    armor in inventory if no slot specified; vendor/nethack/src/do_wear.c
    calls getobj() for the letter prompt).

    NLE multi-key compat: if state.pending_action_slot >= 0 AND that slot
    holds an armor, wear it.  Otherwise fall back to argmax-over-ARMOR.
    """
    def _find_armor(carry, idx):
        found, slot = carry
        is_armor = state.inventory.items.category[idx] == jnp.int8(ItemCategory.ARMOR)
        slot  = jnp.where(~found & is_armor, idx, slot)
        found = found | is_armor
        return (found, slot), None

    (_, first_armor), _ = lax.scan(
        _find_armor, (jnp.bool_(False), jnp.int32(0)), jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32)
    )

    # Multi-key follow-up: prefer agent-chosen slot if it's a valid armor.
    from Nethax.nethax.subsystems.pending_action import resolve_slot
    chosen = resolve_slot(state, first_armor)
    safe_chosen = jnp.clip(chosen, 0, MAX_INVENTORY_SLOTS - 1)
    chosen_is_armor = (
        state.inventory.items.category[safe_chosen]
        == jnp.int8(ItemCategory.ARMOR)
    ) & (state.inventory.items.quantity[safe_chosen] > jnp.int16(0))
    slot_idx = jnp.where(chosen_is_armor, safe_chosen, first_armor).astype(jnp.int32)
    return wear_armor(state, slot_idx, ArmorSlot.BODY)


def step(state, rng):
    """Per-turn inventory upkeep (ring/amulet tick effects, etc.).

    No-op for Wave 3. Wave 4: tick worn ring/amulet duration effects.
    """
    return state


# ---------------------------------------------------------------------------
# Naming (Wave 6) — vendor/nethack/src/do_name.c::do_oname
# ---------------------------------------------------------------------------

def handle_name(state, rng, slot_idx, name_bytes) -> "object":
    """Assign a user-given name to the inventory slot ``slot_idx``.

    Mirrors NetHack's ``do_oname`` (do_name.c): writes the chosen name into
    the slot's onamebuf so subsequent ``doname`` calls emit
    ``" named <name>"`` after the canonical item name.

    Parameters
    ----------
    state      : EnvState (must contain ``inventory`` field).
    rng        : JAX PRNG key (unused; kept for handler-signature symmetry).
    slot_idx   : int / jnp.int32 — inventory slot to name.
    name_bytes : sequence of length USER_NAME_LEN (Python list, bytes, or
                 ndarray).  Zero-terminated; first 0 byte marks the end.

    Returns
    -------
    new_state with ``inventory.user_names[slot_idx]`` updated.
    """
    slot_i32 = jnp.int32(slot_idx)
    safe_slot = jnp.clip(slot_i32, 0, MAX_INVENTORY_SLOTS - 1)

    # Normalize name_bytes to a length-USER_NAME_LEN int8 array.
    if isinstance(name_bytes, (bytes, bytearray)):
        padded = bytes(name_bytes)[:USER_NAME_LEN]
        padded = padded + b"\x00" * (USER_NAME_LEN - len(padded))
        name_row = jnp.array(list(padded), dtype=jnp.int8)
    elif isinstance(name_bytes, str):
        b = name_bytes.encode("ascii")[:USER_NAME_LEN]
        b = b + b"\x00" * (USER_NAME_LEN - len(b))
        name_row = jnp.array(list(b), dtype=jnp.int8)
    else:
        name_row = jnp.asarray(name_bytes, dtype=jnp.int8)
        # Ensure exact length
        cur_len = name_row.shape[0] if hasattr(name_row, "shape") else len(name_row)
        if cur_len < USER_NAME_LEN:
            pad = jnp.zeros((USER_NAME_LEN - cur_len,), dtype=jnp.int8)
            name_row = jnp.concatenate([name_row, pad], axis=0)
        elif cur_len > USER_NAME_LEN:
            name_row = name_row[:USER_NAME_LEN]

    new_user_names = state.inventory.user_names.at[safe_slot].set(name_row)
    new_inv = state.inventory.replace(user_names=new_user_names)
    return state.replace(inventory=new_inv)


# ---------------------------------------------------------------------------
# Helmet classification (vendor do_wear.c::hard_helmet)
# ---------------------------------------------------------------------------
# Vendor:  hard_helmet := is_metallic(obj) || is_crackable(obj)
# Soft (non-protective vs ballfall / digging debris):
#   71 elven leather helm  (leather)
#   74 fedora              (cloth)
#   75 cornuthaum          (cloth, conical hat)
#   76 dunce cap           (cloth)
# Hard (protective):
#   72 orcish helm         (iron)
#   73 dwarvish iron helm  (iron)
#   77 dented pot          (metal)
#   78 helmet              (plumed metal)
#   79 helm of brilliance  (crystal — crackable)
#   80 helm of opposite alignment (metal)
#   81 helm of telepathy   (metal)
# Cite: vendor/nethack/src/do_wear.c::hard_helmet,
#       vendor/nethack/include/objects.h HELM() entries 71-81.
_HARD_HELM_TIDS: tuple = (72, 73, 77, 78, 79, 80, 81)


def is_hard_helmet(type_id: jnp.ndarray) -> jnp.ndarray:
    """Return True if the helmet type_id is metallic/crackable (hard).

    JIT-pure; bool scalar.  Type IDs outside the helmet range return False.
    """
    tid = type_id.astype(jnp.int32)
    result = jnp.bool_(False)
    for hid in _HARD_HELM_TIDS:
        result = result | (tid == jnp.int32(hid))
    return result

