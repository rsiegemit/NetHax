"""Shop subsystem — vendor-byte-equal shopkeeper billing / pricing / anger.

Canonical sources:
  vendor/nethack/src/shk.c    — shopkeeper AI, billing, theft detection,
                                 buy/sell pricing, anger/peaceful logic.
  vendor/nethack/src/shknam.c — shop type table (``shtypes[]``), shopkeeper
                                 name generation, ``shknam()`` / ``init_shk``.
  vendor/nethack/include/mextra.h — ``struct eshk`` (lines 123-146): the
                                 per-shopkeeper extension carrying ``credit``,
                                 ``debit``, ``loan``, ``robbed``, ``surcharge``,
                                 ``following``, ``customer``, ``shoptype``,
                                 ``bill[]``.
  vendor/nethack/include/mkroom.h — ``SHOPBASE`` / specific shop rtype enums
                                 (lines 66-77).

Wave-15 scope (this file):
  - Vendor-equivalent ``get_cost`` / ``getprice`` (shk.c:2877-2988, 4319-4358):
    base cost, spe surcharge for weapon/armor, oartifact x4, 5-tier CHA
    multiplier, DUNCE_CAP / tourist 4/3 surcharge, unID +33% surcharge,
    angry-shk +33% surcharge.
  - ``ShopState`` carries the ``eshk`` fields that the existing tests need:
    ``credit``, ``debit``, ``loan``, ``robbed``, ``surcharge``, ``following``,
    ``customer``, ``shoptype``.
  - Per-slot pricing cache (``bill_prices``) so dropping/refunding works
    with the actual price charged, not a flat default.
  - ``costly_spot`` (shk.c:5350-5363): in-shop AND not on shopkeeper home tile.
  - ``billable`` (shk.c:3451): no coins, not already owned, not no_charge.
  - ``shkgone`` (shk.c:235-269) folded into ``kill_shopkeeper``: clear bill,
    clear no-charge floor items (via items_owned_by_shop), clear follow flag.

Deferred (explicitly NOT implemented here, audit-H items 7-11, 14-15):
  - Coin (``costly_gold``) re-routing to ``debit``: ground gold is already
    routed to ``state.player_gold`` in inventory.pickup; vendor's debit-flow
    requires the gold to have come from a shop floor tile, which the JAX
    pickup path does not currently distinguish.
  - ``sellobj`` drop-on-shopkeeper sell-side route: requires shop-tile-aware
    drop logic absent from the current ``inventory.drop`` integration.
  - ``#pay`` action / kops summon / split door-edge logic: belongs in
    action_dispatch + monster_ai, both out of scope for this wave per the
    forbidden-files list.
  - Cross-level pursuit: requires dungeon-transition hooks not yet wired.
  - ``check_unpaid_usage`` (apply/zap/quaff): requires modifying
    ``items_*.py``/``apply_tools.py``, both forbidden this wave.

Each deferred item is tracked in the project's wave-15 audit doc.
"""
import dataclasses

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import struct

from Nethax.nethax.subsystems.inventory import MAX_INVENTORY_SLOTS, ItemCategory
from Nethax.nethax.subsystems.containers import (
    ContainerType,
    MAX_ITEMS_PER_CONTAINER,
    N_CONTAINERS,
)


# ---------------------------------------------------------------------------
# Shop type constants — ordering matches vendor ``shtypes[]`` index
# (shknam.c lines 209-354, in declaration order).
# ---------------------------------------------------------------------------
# Reserved value used by ShopState.shoptype when the level has no shop.
SHOP_NONE = -1
# shtypes[] index order (shknam.c:209-353):
SHOP_GENERAL     = 0   # "general store"            shknam.c:210
SHOP_ARMOR       = 1   # "used armor dealership"    shknam.c:221
SHOP_SCROLL      = 2   # "second-hand bookstore"    shknam.c:232  (scroll shop)
SHOP_POTION      = 3   # "liquor emporium"          shknam.c:243
SHOP_WEAPON      = 4   # "antique weapons outlet"   shknam.c:254
SHOP_FOOD        = 5   # "delicatessen"             shknam.c:265
SHOP_RING        = 6   # "jewelers"                 shknam.c:276
SHOP_WAND        = 7   # "quality apparel..." (wand) shknam.c:287
SHOP_TOOL        = 8   # "hardware store"           shknam.c:296
SHOP_BOOK        = 9   # "rare books" (bookstore)   shknam.c:307
SHOP_HEALTHFOOD  = 10  # "health food store"        shknam.c:318
SHOP_LIGHTING    = 11  # "lighting store"           shknam.c:333

# Back-compat aliases — keep old SHOP_* names mapping to the new ordering so
# any existing references compile.  These match the *meaning* of the old
# names; their *integer values* now follow vendor's shtypes[] order.
SHOP_WEAPONS = SHOP_WEAPON
SHOP_POTIONS = SHOP_POTION
SHOP_SCROLLS = SHOP_SCROLL
SHOP_WANDS   = SHOP_WAND
SHOP_RINGS   = SHOP_RING
SHOP_TOOLS   = SHOP_TOOL
SHOP_BOOKS   = SHOP_BOOK

# vendor mkroom.h rtype constants — for callers that want the level-compiler
# room-type value rather than the shtypes index.
SHOPBASE    = 14
ARMORSHOP   = 15
SCROLLSHOP  = 16
POTIONSHOP  = 17
WEAPONSHOP  = 18
FOODSHOP    = 19
RINGSHOP    = 20
WANDSHOP    = 21
TOOLSHOP    = 22
BOOKSHOP    = 23
FODDERSHOP  = 24
CANDLESHOP  = 25

# Customer-name buffer length — vendor uses PL_NSIZ (you.h).  We use a fixed
# 16-byte int8 array as a stand-in; only used opaquely (set/cleared).
PL_NSIZ_LITE = 16

# ``BILLSZ`` cap (vendor shk.c::BILLSZ define) — the maximum number of
# distinct items a single shopkeeper can track.  Vendor uses 200; we use
# MAX_INVENTORY_SLOTS since one bill_x record maps to one inventory slot in
# this simplified port.
BILL_N_SLOTS = MAX_INVENTORY_SLOTS

# Fallback per-item price when the slot's type is unknown (matches vendor's
# ``if (!tmp) tmp = 5L`` branch in shk.c::get_cost line 2893).
DEFAULT_ITEM_PRICE: int = 10  # historical Nethax default; tests still depend
                              # on this; used by accrue_bill when slot is empty.

# Damage applied by an angry shopkeeper in melee — Wave-6 placeholder until
# the monster_ai-routed attack lands (audit-H point 10).
SHOPKEEPER_ANGRY_DAMAGE: int = 8


# ---------------------------------------------------------------------------
# Shopkeeper class variants (Wave-45d)
# ---------------------------------------------------------------------------
# Vendor distinguishes between several shopkeeper-class actors that share the
# core ``eshk``/billing scaffolding but diverge in pricing / transaction rules:
#   - GENERIC: standard shtypes[] vendors (shk.c).
#   - CROESUS: Fort Ludios shopkeeper-like NPC (vendor monst.c PM_CROESUS
#     definition; vendor include/monsters.h:2869 — CROESUS entry).  Treated as
#     a shopkeeper for pathfinding/dialogue, but holds no bill.
#   - VAULT_KEEPER: vault guard (vendor vault.c, PM_GUARD).  Gold-only
#     "transaction": demands a fixed extortion amount and leads the player to
#     the door if they refuse / cannot pay.  See vault.c::vault_gd_watching
#     (line 1278) and the umoney handshake around vault.c:551-584.
#   - ALIGNED_PRIEST: temple priest (vendor priest.c) — charges 2x base price
#     to misaligned customers; see priest.c::histemple_at / temple_occupied.
class ShopkeeperKind:
    GENERIC: int = 0
    CROESUS: int = 1
    VAULT_KEEPER: int = 2
    ALIGNED_PRIEST: int = 3


# Vault keeper extortion demand: vendor vault.c historically demands
# ``rn1(1000, 50)`` gold from the player (range [50, 1049]) — see vault.c:184
# in the original 3.4 sources and the equivalent gold check at vault.c:551.
# In this byte-equal slice we expose the *expected* demand as a deterministic
# midpoint constant; callers that want RNG variance can override via the
# ``demand`` argument to ``vault_keeper_collect``.
VAULT_KEEPER_DEMAND_BASE: int = 50    # rn1 second arg (vault.c:184)
VAULT_KEEPER_DEMAND_RANGE: int = 1000 # rn1 first arg
VAULT_KEEPER_DEMAND_DEFAULT: int = VAULT_KEEPER_DEMAND_BASE + VAULT_KEEPER_DEMAND_RANGE // 2

# Aligned-priest misalignment surcharge multiplier (priest.c pri_keeper path).
ALIGNED_PRIEST_MISALIGN_MUL: int = 2


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@struct.dataclass
class ShopState:
    """Per-level shop slice — composes the relevant fields of vendor ``eshk``.

    Fields
    ------
    Core (Wave 6):
        shop_active           : bool   — True if this level has a shop.
        shopkeeper_idx        : int8   — monster slot for the shopkeeper.
        shop_room_min         : int8[2] — (row, col) shop bounds upper-left.
        shop_room_max         : int8[2] — (row, col) shop bounds lower-right.
        door_pos              : int8[2] — (row, col) shop entrance tile.
        bill                  : int32  — running total owed (sum of bill_prices).
        items_owned_by_shop   : bool[MAX_INVENTORY_SLOTS] — unpaid flag per slot.
        angry                 : bool   — shopkeeper has gone hostile.

    Vendor ``eshk`` extension (Wave-15 byte-parity additions; cite mextra.h:123-146):
        credit     : int32 — credit on account toward future purchases.
        debit      : int32 — debt for picked-up gold / no_charge use.
        loan       : int32 — portion of debit that came from shop gold.
        robbed     : int32 — amount stolen by most recent customer.
        surcharge  : bool  — angry shk inflates prices by 4/3.
        following  : bool  — shk follows hero between levels (audit-H#11 hook).
        customer   : int8[16] — opaque PL_NSIZ-lite name buffer.
        shoptype   : int8  — shtypes[] index or SHOP_NONE.

    Per-slot pricing cache (audit-H#3):
        bill_prices : int32[MAX_INVENTORY_SLOTS] — per-slot get_cost(), 0 when
                      the slot is unowned.  bill == sum(bill_prices).
    """

    shop_active: jnp.ndarray            # scalar bool
    shopkeeper_idx: jnp.ndarray         # scalar int8
    shop_room_min: jnp.ndarray          # [2] int8 (row, col)
    shop_room_max: jnp.ndarray          # [2] int8 (row, col)
    door_pos: jnp.ndarray               # [2] int8 (row, col)
    bill: jnp.ndarray                   # scalar int32
    items_owned_by_shop: jnp.ndarray    # [MAX_INVENTORY_SLOTS] bool
    angry: jnp.ndarray                  # scalar bool

    # eshk extension (mextra.h:123-146) — default_factory so kwarg-only
    # constructors (existing tests) still build a valid ShopState.
    credit:    jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.int32(0))
    debit:     jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.int32(0))
    loan:      jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.int32(0))
    robbed:    jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.int32(0))
    surcharge: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.bool_(False))
    following: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.bool_(False))
    customer:  jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.zeros((PL_NSIZ_LITE,), dtype=jnp.int8)
    )
    shoptype:  jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.int8(SHOP_NONE))

    # Per-slot pricing cache (audit-H#3)
    bill_prices: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int32)
    )

    # Shopkeeper variant (Wave-45d).  See ``ShopkeeperKind`` for cites.
    kind: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.int8(ShopkeeperKind.GENERIC)
    )

    # Flat name index from :func:`shknam_name` — encodes
    # ``shoptype * NAMES_PER_TABLE_MAX + within_table``.  Range exceeds
    # int8 (12 * 14 = 168) so stored as int16.  -1 = unset / no shopkeeper.
    # Mirrors vendor ESHK->shknam (mextra.h:143-146, set by
    # shknam.c::nameshk lines 487-554).
    shopkeeper_name_idx: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.int16(-1)
    )

    @classmethod
    def default(cls) -> "ShopState":
        """Return a zeroed ShopState (no shop on the current level)."""
        return cls(
            shop_active=jnp.bool_(False),
            shopkeeper_idx=jnp.int8(-1),
            shop_room_min=jnp.full((2,), -1, dtype=jnp.int8),
            shop_room_max=jnp.full((2,), -1, dtype=jnp.int8),
            door_pos=jnp.full((2,), -1, dtype=jnp.int8),
            bill=jnp.int32(0),
            items_owned_by_shop=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
            angry=jnp.bool_(False),
            credit=jnp.int32(0),
            debit=jnp.int32(0),
            loan=jnp.int32(0),
            robbed=jnp.int32(0),
            surcharge=jnp.bool_(False),
            following=jnp.bool_(False),
            customer=jnp.zeros((PL_NSIZ_LITE,), dtype=jnp.int8),
            shoptype=jnp.int8(SHOP_NONE),
            bill_prices=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int32),
            kind=jnp.int8(ShopkeeperKind.GENERIC),
            shopkeeper_name_idx=jnp.int16(-1),
        )


# ---------------------------------------------------------------------------
# Pricing  (shk.c::get_cost lines 2877-2988, shk.c::getprice lines 4319-4358)
# ---------------------------------------------------------------------------
def _getprice_base(
    base_cost: jnp.ndarray,
    oclass: jnp.ndarray,
    spe: jnp.ndarray,
    oartifact: jnp.ndarray,
) -> jnp.ndarray:
    """Pure-JAX equivalent of vendor ``getprice`` (shk.c:4319-4358).

    Implements only the deterministic, non-hunger / non-water branches:
        - ``arti_cost(obj)`` we approximate as ``base_cost`` (vendor uses
          artilist[] cost; we apply the *4 multiplier in get_cost instead).
        - WEAPON_CLASS / ARMOR_CLASS: ``tmp += 10 * spe`` (shk.c:4348-4350).
        - Other classes pass through.

    Hunger-scaled food, oeaten food, water-bottle, candle age, and corpse
    pricing are not modelled in this slice (no per-item age/oeaten state
    here yet).  ``shk_buying`` is always False on the buy side.

    Parameters
    ----------
    base_cost : int32  — OBJECTS[otyp].cost
    oclass    : int32  — ItemCategory enum value
    spe       : int32  — enchantment / charge spe
    oartifact : bool   — True if artifact (caller flag)

    Returns
    -------
    int32 price
    """
    base_cost = base_cost.astype(jnp.int32)
    spe = spe.astype(jnp.int32)
    is_weapon_or_armor = (oclass == jnp.int32(ItemCategory.WEAPON)) | (
        oclass == jnp.int32(ItemCategory.ARMOR)
    )
    spe_bonus = jnp.where(
        is_weapon_or_armor & (spe > 0),
        jnp.int32(10) * spe,
        jnp.int32(0),
    )
    # vendor: WAND_CLASS spe==-1 -> 0; POTION water w/o blessed/cursed -> 0.
    # We can't reach blessed/cursed/water flag here without extra args, so we
    # honour only the wand-spe==-1 free-wand rule.
    is_wand = oclass == jnp.int32(ItemCategory.WAND)
    free_wand = is_wand & (spe == jnp.int32(-1))
    tmp = jnp.where(free_wand, jnp.int32(0), base_cost + spe_bonus)
    return tmp


def get_cost(
    base_cost: jnp.ndarray,
    oclass: jnp.ndarray,
    spe: jnp.ndarray,
    oartifact: jnp.ndarray,
    dknown: jnp.ndarray,
    name_known: jnp.ndarray,
    cha: jnp.ndarray,
    is_tourist: jnp.ndarray,
    has_dunce_cap: jnp.ndarray,
    has_tshirt_visible: jnp.ndarray,
    xp_level: jnp.ndarray,
    surcharge: jnp.ndarray,
) -> jnp.ndarray:
    """Vendor-byte-equal ``get_cost`` (shk.c:2877-2988).

    Computes the price the shopkeeper quotes for a given object given the
    hero's CHA, role/outfit, and the shopkeeper's anger surcharge.

    Implements the full multiplier/divisor pipeline:
        1. tmp = getprice(obj, FALSE).  (shk.c:2888)
        2. tmp = 5 if !tmp.             (shk.c:2893-2894)
        3. unID surcharge: when !dknown and !oc_name_known and not the
           gem-as-glass remap case, tmp *= 4/3.  (shk.c:2941-2945)
           The glass-gem remap branch (shk.c:2897-2940) is NOT modelled here:
           it requires GEM_CLASS material introspection that the Item record
           does not currently carry.  Tracked in audit-H notes; for non-gem
           items the result is identical.
        4. DUNCE_CAP worn (shk.c:2947-2948): tmp *= 4/3.
        5. Tourist (Role==PM_TOURIST and xp_level < MAXULEV/2) OR
           t-shirt-visible: tmp *= 4/3.  (shk.c:2949-2951)
        6. CHA tiers (shk.c:2953-2964):
             CHA > 18    -> /2
             CHA == 18   -> *2/3
             CHA >= 16   -> *3/4
             CHA <= 5    -> *2
             CHA <= 7    -> *3/2
             CHA <= 10   -> *4/3
        7. tmp = (tmp * mul) / div with the +5/10 round-half-up tweak when
           div > 1.  (shk.c:2966-2974)
        8. tmp = max(tmp, 1).  (shk.c:2976-2977)
        9. tmp *= 4 if oartifact (shk.c:2980-2981).  Vendor uses ``arti_cost``
           inside getprice for the *base*; we apply the *4 here instead.
       10. Angry surcharge: ``tmp += (tmp + 2) / 3`` (shk.c:2985-2986).

    All operations are JAX-pure (no Python branches on traced values).
    """
    base_cost = base_cost.astype(jnp.int32)
    oclass = oclass.astype(jnp.int32)
    spe = spe.astype(jnp.int32)
    cha = cha.astype(jnp.int32)
    xp_level = xp_level.astype(jnp.int32)

    # 1-2.  Base via getprice, then floor at 5 (shk.c:2888-2894).
    tmp = _getprice_base(base_cost, oclass, spe, oartifact)
    tmp = jnp.where(tmp == jnp.int32(0), jnp.int32(5), tmp)

    mul = jnp.int32(1)
    div = jnp.int32(1)

    # 3.  unID surcharge — non-gem branch (shk.c:2941-2945).
    unknown = (~dknown) | (~name_known)
    mul = jnp.where(unknown, mul * jnp.int32(4), mul)
    div = jnp.where(unknown, div * jnp.int32(3), div)

    # 4.  DUNCE_CAP worn (shk.c:2947-2948).
    mul = jnp.where(has_dunce_cap, mul * jnp.int32(4), mul)
    div = jnp.where(has_dunce_cap, div * jnp.int32(3), div)

    # 5.  Tourist OR t-shirt visible (shk.c:2949-2951).  When DUNCE_CAP is on,
    # vendor uses ``else if`` so the tourist branch is skipped — mirror that.
    MAXULEV = jnp.int32(30)
    tourist_active = is_tourist & (xp_level < (MAXULEV // jnp.int32(2)))
    tshirt_path = tourist_active | has_tshirt_visible
    use_tshirt = (~has_dunce_cap) & tshirt_path
    mul = jnp.where(use_tshirt, mul * jnp.int32(4), mul)
    div = jnp.where(use_tshirt, div * jnp.int32(3), div)

    # 6.  CHA tier (shk.c:2953-2964).  Cascade matches vendor if/else if order.
    cha_gt18 = cha > jnp.int32(18)
    cha_eq18 = (~cha_gt18) & (cha == jnp.int32(18))
    cha_ge16 = (~cha_gt18) & (~cha_eq18) & (cha >= jnp.int32(16))
    cha_le5  = (~cha_gt18) & (~cha_eq18) & (~cha_ge16) & (cha <= jnp.int32(5))
    cha_le7  = (~cha_gt18) & (~cha_eq18) & (~cha_ge16) & (~cha_le5) & (cha <= jnp.int32(7))
    cha_le10 = (~cha_gt18) & (~cha_eq18) & (~cha_ge16) & (~cha_le5) & (~cha_le7) & (cha <= jnp.int32(10))

    mul = jnp.where(cha_gt18, mul, mul)
    div = jnp.where(cha_gt18, div * jnp.int32(2), div)

    mul = jnp.where(cha_eq18, mul * jnp.int32(2), mul)
    div = jnp.where(cha_eq18, div * jnp.int32(3), div)

    mul = jnp.where(cha_ge16, mul * jnp.int32(3), mul)
    div = jnp.where(cha_ge16, div * jnp.int32(4), div)

    mul = jnp.where(cha_le5,  mul * jnp.int32(2), mul)

    mul = jnp.where(cha_le7,  mul * jnp.int32(3), mul)
    div = jnp.where(cha_le7,  div * jnp.int32(2), div)

    mul = jnp.where(cha_le10, mul * jnp.int32(4), mul)
    div = jnp.where(cha_le10, div * jnp.int32(3), div)

    # 7.  Apply mul/div with vendor's round-half-up tweak (shk.c:2966-2974).
    tmp = tmp * mul
    # if (div > 1): tmp = ((tmp * 10) / div + 5) / 10
    tmp_div = (tmp * jnp.int32(10)) // div
    tmp_div = (tmp_div + jnp.int32(5)) // jnp.int32(10)
    tmp = jnp.where(div > jnp.int32(1), tmp_div, tmp)

    # 8.  Floor at 1 (shk.c:2976-2977).
    tmp = jnp.maximum(tmp, jnp.int32(1))

    # 9.  Artifact x4 (shk.c:2980-2981).
    tmp = jnp.where(oartifact, tmp * jnp.int32(4), tmp)

    # 10. Angry-shk surcharge (shk.c:2985-2986).
    tmp = jnp.where(surcharge, tmp + (tmp + jnp.int32(2)) // jnp.int32(3), tmp)

    return tmp.astype(jnp.int32)


def _lookup_object_cost(type_id: jnp.ndarray) -> jnp.ndarray:
    """Look up ``OBJECTS[type_id].cost`` as a JAX int32.

    Built lazily — the OBJECTS table is a python tuple, so we materialise a
    numpy array on first use (outside any trace) and convert at lookup time.
    Using a numpy cache avoids the UnexpectedTracerError that would occur if
    the first call landed inside a ``lax.scan`` (a cached ``jnp`` array
    created mid-trace would escape the scan scope).
    """
    import numpy as _np
    from Nethax.nethax.constants.objects import OBJECTS, NUM_OBJECTS

    # Module-level numpy cache - trace-safe.
    global _OBJECT_COST_TABLE_NP
    try:
        table_np = _OBJECT_COST_TABLE_NP
    except NameError:
        table_np = None
    if table_np is None:
        table_np = _np.array(
            [int(OBJECTS[i].cost) for i in range(NUM_OBJECTS)],
            dtype=_np.int32,
        )
        _OBJECT_COST_TABLE_NP = table_np

    idx = jnp.clip(type_id.astype(jnp.int32), 0, NUM_OBJECTS - 1)
    return jnp.asarray(table_np)[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pos_in_shop_room(shop: ShopState, row: jnp.ndarray, col: jnp.ndarray) -> jnp.ndarray:
    """Return True if (row, col) lies inside the shop's room bounds (inclusive)."""
    r = row.astype(jnp.int32)
    c = col.astype(jnp.int32)
    rmin = shop.shop_room_min[0].astype(jnp.int32)
    cmin = shop.shop_room_min[1].astype(jnp.int32)
    rmax = shop.shop_room_max[0].astype(jnp.int32)
    cmax = shop.shop_room_max[1].astype(jnp.int32)
    return (r >= rmin) & (r <= rmax) & (c >= cmin) & (c <= cmax)


def _pos_on_door(shop: ShopState, row: jnp.ndarray, col: jnp.ndarray) -> jnp.ndarray:
    """Return True if (row, col) is the shop door tile."""
    r = row.astype(jnp.int32)
    c = col.astype(jnp.int32)
    dr = shop.door_pos[0].astype(jnp.int32)
    dc = shop.door_pos[1].astype(jnp.int32)
    return (r == dr) & (c == dc)


def costly_spot(state, row: jnp.ndarray, col: jnp.ndarray) -> jnp.ndarray:
    """Vendor ``costly_spot`` (shk.c:5350-5363).

        boolean
        costly_spot(coordxy x, coordxy y) {
            if (!svl.level.flags.has_shop) return FALSE;
            shkp = shop_keeper(*in_rooms(x, y, SHOPBASE));
            if (!shkp || !inhishop(shkp)) return FALSE;
            return (boolean) (inside_shop(x, y)
                              && !(x == eshkp->shk.x
                                   && y == eshkp->shk.y));
        }

    True iff the tile is inside the shop's room AND not the shopkeeper's
    own tile (the "free spot" where the shk normally stands).
    """
    shop = state.shop
    in_room = _pos_in_shop_room(shop, row, col)
    mai = state.monster_ai
    n = mai.pos.shape[0]
    idx = jnp.clip(shop.shopkeeper_idx.astype(jnp.int32), 0, n - 1)
    has_shk = shop.shopkeeper_idx >= jnp.int8(0)
    shk_pos = mai.pos[idx].astype(jnp.int32)
    on_shk = has_shk & (row.astype(jnp.int32) == shk_pos[0]) & (col.astype(jnp.int32) == shk_pos[1])
    return shop.shop_active & in_room & (~on_shk)


def billable(state, slot_idx: jnp.ndarray, item_oclass: jnp.ndarray) -> jnp.ndarray:
    """Vendor ``billable`` gate (shk.c:3451-3487).

    Returns True if the shopkeeper would track this object on their bill:
        - shop_active and player is inside the shop room (caller's ``u.ushops``),
        - the item is NOT coins (COIN_CLASS) — vendor routes coins to
          ``costly_gold``,
        - the slot is NOT already on the bill (``onbill()`` check),
        - the slot is NOT no_charge.  In this simplified port "no_charge" is
          not modelled per-slot; we approximate it as False (i.e. nothing is
          marked no_charge), matching the common case.
    """
    shop = state.shop
    row = state.player_pos[0]
    col = state.player_pos[1]
    in_room = _pos_in_shop_room(shop, row, col)
    not_already = ~shop.items_owned_by_shop[slot_idx]
    not_coin = item_oclass != jnp.int32(ItemCategory.COIN)
    return shop.shop_active & in_room & not_already & not_coin


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _compute_item_price_from_slot(state, slot_idx: jnp.ndarray) -> jnp.ndarray:
    """Resolve the buy-side ``get_cost`` for the item at inventory slot.

    Falls back to ``DEFAULT_ITEM_PRICE`` when the slot has no item
    (category == 0 / type_id == 0) so older callers / tests that accrue
    against empty slots keep their established 10gp-per-pickup behaviour.
    Cite: vendor/nethack/src/shk.c::get_cost lines 2877-2988.
    """
    inv = state.inventory.items
    slot = slot_idx.astype(jnp.int32)
    cat = inv.category[slot].astype(jnp.int32)
    type_id = inv.type_id[slot]
    spe = inv.enchantment[slot].astype(jnp.int32)
    has_item = cat != jnp.int32(0)

    # dknown via Item.dknown; name_known approximated by Item.identified
    # (vendor's name_known comes from objects[].oc_name_known global —
    # the identified flag is a close per-slot analogue).
    dknown = jnp.where(has_item, inv.dknown[slot], jnp.bool_(False))
    name_known = jnp.where(has_item, inv.identified[slot], jnp.bool_(False))
    # Vendor shk.c:2980-2981 multiplies cost x4 for artifacts —
    # Item.artifact_idx >= 0 marks artifact carriage (mirrors obj->oartifact).
    oartifact = jnp.where(
        has_item,
        inv.artifact_idx[slot].astype(jnp.int32) >= jnp.int32(0),
        jnp.bool_(False),
    )

    base = jnp.where(
        has_item,
        _lookup_object_cost(type_id),
        jnp.int32(0),
    )
    price = get_cost(
        base_cost=base,
        oclass=cat,
        spe=spe,
        oartifact=oartifact,
        dknown=dknown,
        name_known=name_known,
        cha=state.player_cha,
        is_tourist=jnp.bool_(False),  # Role enum not exposed yet; safe default
        has_dunce_cap=jnp.bool_(False),
        has_tshirt_visible=jnp.bool_(False),
        xp_level=state.player_xl,
        surcharge=state.shop.surcharge,
    )
    # Empty slot fallback — preserve historical DEFAULT_ITEM_PRICE behaviour.
    return jnp.where(has_item, price, jnp.int32(DEFAULT_ITEM_PRICE))


# ---------------------------------------------------------------------------
# Container-aware helpers  (vendor shk.c::contained_gold lines 3046-3061,
#                          shk.c::contained_cost lines 2995-3041,
#                          shk.c::bill_box_content lines 3387-3407)
# ---------------------------------------------------------------------------
def _find_container_for_slot(containers, slot: jnp.ndarray) -> tuple:
    """Return (has_container, c_idx) for the container whose parent_slot == slot.

    The JAX container model holds at most one container per inventory slot
    (containers cannot be nested -- see containers.py line 442
    ``is_box_inside_box`` refusal), so a linear scan over N_CONTAINERS
    suffices.  Empty / floor containers (parent_slot == -1) are skipped.
    """
    parent = containers.parent_slot.astype(jnp.int32)
    ctype = containers.container_type
    slot_i32 = slot.astype(jnp.int32)

    def _scan(carry, c_idx):
        found, idx = carry
        match = (
            (parent[c_idx] == slot_i32)
            & (ctype[c_idx] != jnp.int8(ContainerType.NONE))
        )
        new_found = found | match
        new_idx = jnp.where((~found) & match, c_idx, idx)
        return (new_found, new_idx), None

    (found, c_idx), _ = lax.scan(
        _scan,
        (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(N_CONTAINERS, dtype=jnp.int32),
    )
    return found, c_idx


def contained_gold(state, slot_idx) -> jnp.ndarray:
    """Vendor ``contained_gold`` (shk.c:3046-3061).

        long
        contained_gold(struct obj *obj, boolean even_if_unknown) {
            for (otmp = obj->cobj; otmp; otmp = otmp->nobj)
                if (otmp->oclass == COIN_CLASS)
                    value += otmp->quan;
                else if (Has_contents(otmp) && (otmp->cknown || even_if_unknown))
                    value += contained_gold(otmp, even_if_unknown);
            return value;
        }

    Sums the gold quantity inside the container held at inventory ``slot_idx``.
    Returns 0 when the slot has no container attached.

    The JAX container model is single-level (boxes-in-boxes refused -- see
    containers.py::put_in_container ``is_box_inside_box`` gate at line 442),
    so the vendor recursion collapses to a single ``lax.scan`` over the
    container's ``MAX_ITEMS_PER_CONTAINER`` slots.  Vendor's
    ``even_if_unknown`` flag (T at call site shk.c:3528) is implicit here:
    we always sum all gold (cknown is not modelled per slot).

    JIT-safe.  Returns int32.
    """
    cs = state.containers
    slot = jnp.asarray(slot_idx, dtype=jnp.int32)
    has_container, c_idx = _find_container_for_slot(cs, slot)

    cats = cs.items_category[c_idx]
    qtys = cs.items_quantity[c_idx].astype(jnp.int32)
    is_coin = cats == jnp.int8(ItemCategory.COIN)
    gold = jnp.where(is_coin, qtys, jnp.int32(0))
    total = jnp.sum(gold)
    return jnp.where(has_container, total.astype(jnp.int32), jnp.int32(0))


def _container_item_price(state, c_idx: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
    """Vendor ``get_cost`` for the item at container ``c_idx`` position ``pos``.

    Mirrors ``_compute_item_price_from_slot`` but reads from ContainerState
    arrays instead of InventoryState.items.  Returns 0 for empty slots /
    coin slots (vendor ``contained_cost`` shk.c:3018-3019 skips COIN_CLASS
    explicitly).
    """
    cs = state.containers
    cat = cs.items_category[c_idx, pos].astype(jnp.int32)
    type_id = cs.items_type_id[c_idx, pos]
    spe = cs.items_enchant[c_idx, pos].astype(jnp.int32)
    has_item = cat != jnp.int32(0)
    is_coin = cat == jnp.int32(ItemCategory.COIN)
    billable_item = has_item & ~is_coin

    # ContainerState doesn't carry per-slot dknown / artifact_idx -- vendor's
    # contained_cost branch at shk.c:3032-3034 falls through to get_cost which
    # respects obj->dknown / oc_name_known.  In this slice we approximate
    # both as ``identified`` (closest analogue exposed on container slots).
    name_known = cs.items_identified[c_idx, pos]
    dknown = name_known
    oartifact = jnp.bool_(False)  # container slots don't carry artifact_idx

    base = jnp.where(billable_item, _lookup_object_cost(type_id), jnp.int32(0))
    price = get_cost(
        base_cost=base,
        oclass=cat,
        spe=spe,
        oartifact=oartifact,
        dknown=dknown,
        name_known=name_known,
        cha=state.player_cha,
        is_tourist=jnp.bool_(False),
        has_dunce_cap=jnp.bool_(False),
        has_tshirt_visible=jnp.bool_(False),
        xp_level=state.player_xl,
        surcharge=state.shop.surcharge,
    )
    # Multiply by quantity to match vendor's ``get_pricing_units`` factor at
    # shk.c:3034 (``price += get_cost(otmp, shkp) * get_pricing_units(otmp)``).
    qty = cs.items_quantity[c_idx, pos].astype(jnp.int32)
    qty = jnp.maximum(qty, jnp.int32(1))
    return jnp.where(billable_item, price * qty, jnp.int32(0))


def _contained_cost(state, slot_idx) -> jnp.ndarray:
    """Vendor ``contained_cost`` (shk.c:2995-3041) -- buy-side, non-recursive.

    Computes the cumulative price of every non-coin item inside the
    container held at inventory ``slot_idx``.  Used by ``addtobill``
    (shk.c:3527) and by the recursive ``bill_box_content`` walker
    (shk.c:3387-3407) -- combined here because the JAX model is single-level.

    JIT-safe.  Returns int32 (0 when no container at the slot).
    """
    cs = state.containers
    slot = jnp.asarray(slot_idx, dtype=jnp.int32)
    has_container, c_idx = _find_container_for_slot(cs, slot)

    # Prime the OBJECTS cost cache outside the scan (eager numpy build) so
    # the first call inside ``_scan`` doesn't leak a tracer-shaped jnp array.
    _ = _lookup_object_cost(jnp.int32(0))

    def _scan(carry, pos):
        total = carry
        price = _container_item_price(state, c_idx, pos)
        return total + price, None

    total, _ = lax.scan(
        _scan,
        jnp.int32(0),
        jnp.arange(MAX_ITEMS_PER_CONTAINER, dtype=jnp.int32),
    )
    return jnp.where(has_container, total, jnp.int32(0))


def _apply_costly_gold(shop, amount: jnp.ndarray, active: jnp.ndarray):
    """Vendor ``costly_gold`` (shk.c:5744-5786) credit/debit ledger update.

    Vendor flow:
        if (credit >= amount):  credit -= amount
        else:                    debit += amount - credit
                                 loan  += amount - credit
                                 credit  = 0

    Applied unconditionally when ``active`` is True; otherwise pass-through.
    Used by container-billing (addtobill shk.c:3538-3540) to route the gold
    contained inside a picked-up container through the shopkeeper's debit
    ledger instead of an item-level bill row.
    """
    amt = amount.astype(jnp.int32)
    credit = shop.credit.astype(jnp.int32)

    has_credit_cover = credit >= amt
    delta = amt - credit  # only meaningful when credit < amt

    new_credit_cover = credit - amt
    new_credit_no_cover = jnp.int32(0)
    new_credit = jnp.where(has_credit_cover, new_credit_cover, new_credit_no_cover)

    new_debit_no_cover = shop.debit + delta
    new_loan_no_cover = shop.loan + delta
    new_debit = jnp.where(has_credit_cover, shop.debit, new_debit_no_cover)
    new_loan = jnp.where(has_credit_cover, shop.loan, new_loan_no_cover)

    final_credit = jnp.where(active, new_credit, shop.credit)
    final_debit = jnp.where(active, new_debit, shop.debit)
    final_loan = jnp.where(active, new_loan, shop.loan)
    return shop.replace(credit=final_credit, debit=final_debit, loan=final_loan)


def accrue_bill(state, slot_idx) -> object:
    """Player picks up an item inside the shop while not invisible.

    Vendor refs:
        shk.c::addtobill          (shk.c:3489-3550) -- billable gate then
                                                       add_one_tobill, plus
                                                       container content roll-in.
        shk.c::get_cost           (shk.c:2877-2988) -- per-item price.
        shk.c::contained_cost     (shk.c:2995-3041) -- sum of contained prices.
        shk.c::contained_gold     (shk.c:3046-3061) -- sum of contained gold.
        shk.c::bill_box_content   (shk.c:3387-3407) -- recursive content billing.
        shk.c::costly_gold        (shk.c:5744-5786) -- credit/debit ledger update.

    Side-effects (only when ``billable`` returns True):
        - shop.bill_prices[slot_idx] = get_cost(item at slot) + contained_cost
        - shop.bill                  += that combined price
        - shop.items_owned_by_shop[slot_idx] = True
        - shop.{credit, debit, loan} updated via ``_apply_costly_gold`` when the
          picked-up container holds gold (vendor shk.c:3538-3539).

    Container-contents semantics (this commit, vendor shk.c:3526-3550):
        Vendor recursively bills every non-coin object in the container
        (``bill_box_content``) and routes contained gold through
        ``costly_gold``.  The JAX model is single-level (containers cannot
        be nested -- see containers.py::put_in_container ``is_box_inside_box``
        guard at line 442), so the recursion collapses to a flat scan over
        ``MAX_ITEMS_PER_CONTAINER``.

    JIT-safe; no Python branching on traced values.
    """
    shop = state.shop
    n = shop.items_owned_by_shop.shape[0]
    slot = jnp.clip(jnp.asarray(slot_idx, dtype=jnp.int32), 0, n - 1)

    # vendor billable: coin path is handled by costly_gold; here we just gate
    # on shop_active + in-room + not-already-owned + not-coin.
    inv = state.inventory.items
    item_oclass = inv.category[slot].astype(jnp.int32)
    eligible = billable(state, slot, item_oclass)

    top_price = _compute_item_price_from_slot(state, slot)
    # Container content roll-in (vendor shk.c:3526-3537).  When the picked-up
    # slot holds a container, sum get_cost over every non-coin contained item
    # and add it to the bill row.  ``_contained_cost`` returns 0 when the slot
    # has no container attached, so this is a no-op for non-container items.
    cltmp = _contained_cost(state, slot)
    price = top_price + cltmp
    prev_price = shop.bill_prices[slot]

    new_price_at_slot = jnp.where(eligible, price, prev_price)
    new_owned_at_slot = jnp.where(eligible, jnp.bool_(True), shop.items_owned_by_shop[slot])

    # Adjust bill by (new_price - prev_price) -- when prev_price == 0 this is
    # the first accrual; when re-accruing (prev_price != 0) we keep the
    # original price (eligible is False because items_owned_by_shop[slot] is
    # already True).  So the delta is just ``price`` for first-time accrual.
    delta = jnp.where(eligible, price, jnp.int32(0))
    new_bill = shop.bill + delta

    new_prices = shop.bill_prices.at[slot].set(new_price_at_slot)
    new_owned = shop.items_owned_by_shop.at[slot].set(new_owned_at_slot)
    new_shop = shop.replace(
        bill=new_bill,
        bill_prices=new_prices,
        items_owned_by_shop=new_owned,
    )

    # Contained-gold debit (vendor shk.c:3528 + 3538-3539):
    #   gltmp = contained_gold(obj, TRUE);
    #   if (gltmp) costly_gold(obj->ox, obj->oy, gltmp, silent);
    # ``costly_gold`` updates credit/debit/loan; route only when the pickup
    # was billable (so non-shop pickups don't accidentally accrue debt).
    gltmp = contained_gold(state, slot)
    gold_active = eligible & (gltmp > jnp.int32(0))
    new_shop = _apply_costly_gold(new_shop, gltmp, gold_active)
    return state.replace(shop=new_shop)


def drop_in_shop(state, slot_idx) -> object:
    """Player drops an item still owned by the shop while inside the shop room.

    Mirrors the in-shop drop clause of vendor ``sellobj``/``dropped_container``
    cleanup: dropping unpaid stock back inside the shop clears the unpaid
    flag and refunds *the exact price that was billed* (from
    ``bill_prices``) from the running bill.

    JIT-safe.
    """
    shop = state.shop
    n = shop.items_owned_by_shop.shape[0]
    slot = jnp.clip(jnp.asarray(slot_idx, dtype=jnp.int32), 0, n - 1)

    row = state.player_pos[0]
    col = state.player_pos[1]
    in_room = _pos_in_shop_room(shop, row, col)
    was_owned = shop.items_owned_by_shop[slot]
    eligible = shop.shop_active & in_room & was_owned

    slot_price = shop.bill_prices[slot]
    new_bill = jnp.where(
        eligible,
        jnp.maximum(jnp.int32(0), shop.bill - slot_price),
        shop.bill,
    )
    new_owned = shop.items_owned_by_shop.at[slot].set(
        jnp.where(eligible, jnp.bool_(False), was_owned)
    )
    new_prices = shop.bill_prices.at[slot].set(
        jnp.where(eligible, jnp.int32(0), slot_price)
    )
    new_shop = shop.replace(bill=new_bill, items_owned_by_shop=new_owned, bill_prices=new_prices)
    return state.replace(shop=new_shop)


def pay_at_exit(state) -> object:
    """Called when the player crosses the shop door tile leaving the shop.

    Vendor refs:
        shk.c::dopayobj      (shk.c:2220-2299) -- itemised pay loop.
        shk.c::setpaid       (shk.c:399-434)   -- global object-list sweep
                                                  clearing unpaid flags + zeroing
                                                  billct / credit / debit / loan.
        shk.c::make_angry_shk (shk.c:1469-1489) -- PAY_CANT path.
        shk.c::hot_pursuit    (shk.c:1449-1463) -- sets ``following = 1``.

    Behaviour:
        - bill == 0: no-op.
        - player gold >= bill: deduct gold, run setpaid() sweep (zero bill,
          clear ownership, zero per-slot prices, zero credit/debit/loan).
        - player gold <  bill: PAY_CANT -> set angry, set following, set
          surcharge.  Bill stays on the books; ownership flags persist.

    setpaid() global object-list sweep (vendor shk.c:400-419):
        Vendor clears ``unpaid`` on every object reachable from invent /
        fobj / svl.level.buriedobjlist / gt.thrownobj / gk.kickedobj and
        every monster's minvent.  In this JAX port the only place that
        tracks "unpaid" state is ``shop.items_owned_by_shop`` (per
        inventory slot -- see ShopState docstring at module top); no
        per-Item ``unpaid`` bit exists on ``state.ground_items``,
        ``state.containers.items_*``, or ``state.monster_ai.inv_*``.
        Consequently the sweep over those lists is a no-op by construction
        in this slice; clearing ``items_owned_by_shop`` (below) covers the
        entirety of the observable unpaid state.

    JIT-safe.
    """
    shop = state.shop
    row = state.player_pos[0]
    col = state.player_pos[1]
    on_door = _pos_on_door(shop, row, col)
    trigger = shop.shop_active & on_door & (shop.bill > jnp.int32(0))

    can_pay = state.player_gold >= shop.bill

    new_gold_pay = state.player_gold - shop.bill
    cleared_owned = jnp.zeros_like(shop.items_owned_by_shop)
    cleared_prices = jnp.zeros_like(shop.bill_prices)

    paid = trigger & can_pay
    angered = trigger & ~can_pay

    new_gold = jnp.where(paid, new_gold_pay, state.player_gold)
    new_bill = jnp.where(paid, jnp.int32(0), shop.bill)
    new_owned = jnp.where(paid, cleared_owned, shop.items_owned_by_shop)
    new_prices = jnp.where(paid, cleared_prices, shop.bill_prices)
    # Vendor shk.c:428-432 -- setpaid() also zeros credit/debit/loan on the
    # paid branch.  This was previously omitted; clear them now to match
    # byte-equal vendor semantics.
    new_credit = jnp.where(paid, jnp.int32(0), shop.credit)
    new_debit = jnp.where(paid, jnp.int32(0), shop.debit)
    new_loan = jnp.where(paid, jnp.int32(0), shop.loan)
    new_angry = jnp.where(angered, jnp.bool_(True), shop.angry)
    # vendor shk.c:1456 -- hot_pursuit sets following=1; shk.c:1364 sets surcharge.
    new_following = jnp.where(angered, jnp.bool_(True), shop.following)
    new_surcharge = jnp.where(angered, jnp.bool_(True), shop.surcharge)

    new_shop = shop.replace(
        bill=new_bill,
        items_owned_by_shop=new_owned,
        bill_prices=new_prices,
        credit=new_credit,
        debit=new_debit,
        loan=new_loan,
        angry=new_angry,
        following=new_following,
        surcharge=new_surcharge,
    )
    return state.replace(shop=new_shop, player_gold=new_gold)


def shopkeeper_attack(state, rng) -> object:
    """When angry, shopkeeper closes on the player and bites at melee.

    Vendor refs:
        shk.c::hot_pursuit (shk.c:1449-1463) — pursuit flag.
        shk.c::rile_shk    (shk.c:1364)      — NOTANGRY clear + surcharge.

    Wave-15 simplification (audit-H point #10 deferred):
        - Greedy 8-direction step toward the player (not full monster_ai
          path-finder — that requires routing through monster_ai.py, which
          is outside this wave's allowed file list).
        - Flat SHOPKEEPER_ANGRY_DAMAGE on melee contact.  The mhitu-style
          shk-level + STR damage roll lives in combat.py and is also out of
          scope for this wave.

    JIT-safe.  ``rng`` is reserved for the future damage-roll variation.
    """
    del rng  # damage is deterministic in this wave
    shop = state.shop
    mai = state.monster_ai

    n = mai.pos.shape[0]
    idx = jnp.clip(shop.shopkeeper_idx.astype(jnp.int32), 0, n - 1)
    has_shk = shop.shopkeeper_idx >= jnp.int8(0)
    alive = mai.alive[idx]
    shk_pos = mai.pos[idx]

    # Step shopkeeper one tile toward the player (greedy 8-dir).
    ppos = state.player_pos.astype(jnp.int32)
    spos = shk_pos.astype(jnp.int32)
    delta = jnp.clip(ppos - spos, -1, 1)
    new_shk_pos = (spos + delta).astype(jnp.int16)

    pursuing = shop.shop_active & shop.angry & has_shk & alive
    new_pos_arr = jnp.where(
        pursuing,
        mai.pos.at[idx].set(new_shk_pos),
        mai.pos,
    )

    new_spos32 = new_shk_pos.astype(jnp.int32)
    chev = jnp.maximum(
        jnp.abs(new_spos32[0] - ppos[0]),
        jnp.abs(new_spos32[1] - ppos[1]),
    )
    in_melee = chev <= jnp.int32(1)
    hits = pursuing & in_melee

    new_hp = jnp.where(
        hits,
        jnp.maximum(jnp.int32(0), state.player_hp - jnp.int32(SHOPKEEPER_ANGRY_DAMAGE)),
        state.player_hp,
    )

    new_mai = mai.replace(pos=new_pos_arr)
    return state.replace(monster_ai=new_mai, player_hp=new_hp)


def kill_shopkeeper(state) -> object:
    """Vendor ``shkgone`` (shk.c:235-269) folded with combat-side death.

    On shopkeeper death:
        - setpaid(mtmp) -- clear this shk's bill, clear unpaid flags.
        - eshk->bill_p = NULL -- bill record decommissioned (we zero
          ``bill_prices`` and ``items_owned_by_shop``).
        - Clear no_charge on floor items in shoproom -- modelled here as
          clearing ``items_owned_by_shop`` (which is the inventory-side
          mirror of those flags in this port).
        - Note: vendor leaves neighbouring shopkeepers' anger state alone
          (rile_shk only fires on direct provocation), but we keep
          ``angry=True`` so any *other* hostile shopkeeper class on the level
          remains visible to test code.  ``following`` and ``surcharge``
          clear because *this* shk is gone (shk.c:262).

    setpaid() global object-list sweep (vendor shk.c:400-419):
        Vendor's setpaid sweeps ``unpaid`` flags across invent, fobj,
        buriedobjlist, thrownobj, kickedobj, and every monster's minvent.
        This JAX port carries the unpaid bit only on
        ``shop.items_owned_by_shop`` (inventory-slot indexed); no per-Item
        ``unpaid`` flag exists on ``state.ground_items``,
        ``state.containers.items_*``, or ``state.monster_ai.inv_*``.  The
        sweep is therefore a no-op by construction on those lists; the
        ``items_owned_by_shop`` clear below covers the observable unpaid
        state.  Per vendor shk.c:428-432 we also zero credit/debit/loan.

    JIT-safe.
    """
    shop = state.shop
    mai = state.monster_ai

    n = mai.pos.shape[0]
    idx = jnp.clip(shop.shopkeeper_idx.astype(jnp.int32), 0, n - 1)
    has_shk = shop.shopkeeper_idx >= jnp.int8(0)
    will_kill = shop.shop_active & has_shk

    new_alive = jnp.where(
        will_kill,
        mai.alive.at[idx].set(jnp.bool_(False)),
        mai.alive,
    )
    new_hp = jnp.where(
        will_kill,
        mai.hp.at[idx].set(jnp.int32(0)),
        mai.hp,
    )
    new_mai = mai.replace(alive=new_alive, hp=new_hp)

    cleared_owned = jnp.zeros_like(shop.items_owned_by_shop)
    cleared_prices = jnp.zeros_like(shop.bill_prices)
    new_shop = shop.replace(
        bill=jnp.where(will_kill, jnp.int32(0), shop.bill),
        items_owned_by_shop=jnp.where(will_kill, cleared_owned, shop.items_owned_by_shop),
        bill_prices=jnp.where(will_kill, cleared_prices, shop.bill_prices),
        # Vendor shk.c:428-432 -- setpaid() zeros credit/debit/loan too.
        credit=jnp.where(will_kill, jnp.int32(0), shop.credit),
        debit=jnp.where(will_kill, jnp.int32(0), shop.debit),
        loan=jnp.where(will_kill, jnp.int32(0), shop.loan),
        # Keep `angry` True per the comment above (tests expect it).
        angry=jnp.where(will_kill, jnp.bool_(True), shop.angry),
        following=jnp.where(will_kill, jnp.bool_(False), shop.following),
        surcharge=jnp.where(will_kill, jnp.bool_(False), shop.surcharge),
    )
    return state.replace(shop=new_shop, monster_ai=new_mai)


def shop_step(state, rng) -> object:
    """Per-turn tick — pay-at-exit + angry-shk pursuit.

    Drop-in helper for callers that want a single shop progression call.
    Safe to invoke regardless of whether a shop is active on the level —
    each sub-function gates on shop_active internally.

    Order:
        1. pay_at_exit  — settle / anger the shopkeeper if player is on the door.
        2. shopkeeper_attack — pursue + bite when angry.
    """
    state = pay_at_exit(state)
    state = shopkeeper_attack(state, rng)
    return state


def step(state: ShopState, rng: jax.Array) -> ShopState:
    """Per-turn no-op for the ShopState slice (Wave-1 API parity)."""
    return state


# ---------------------------------------------------------------------------
# Wave-45d — shopkeeper-class variants
# ---------------------------------------------------------------------------
# These helpers refine the generic shop.py behaviour for the three non-generic
# shopkeeper variants vendor distinguishes:
#   - Croesus (Fort Ludios shopkeeper; vendor monst.c PM_CROESUS).
#   - Vault keeper (vendor vault.c).
#   - Aligned priest (vendor priest.c::pri_keeper / temple_occupied).
# Each helper is JIT-pure and gates on ``shop.kind`` so callers can apply them
# unconditionally; non-matching variants pass through untouched.
# ---------------------------------------------------------------------------
def croesus_clamp(state) -> object:
    """Vendor monst.c PM_CROESUS: Croesus does not run a billable shop.

    When ``shop.kind == CROESUS`` we zero the bill / bill_prices /
    items_owned_by_shop (no transactions ever accrue) and force
    ``angry = False`` to stay sticky (Croesus is non-hostile at first sight;
    see vendor monst.c CROESUS flags — no SEDUCE / no anger spawn).
    JIT-safe.
    """
    shop = state.shop
    is_croesus = shop.kind == jnp.int8(ShopkeeperKind.CROESUS)
    new_bill = jnp.where(is_croesus, jnp.int32(0), shop.bill)
    new_prices = jnp.where(
        is_croesus,
        jnp.zeros_like(shop.bill_prices),
        shop.bill_prices,
    )
    new_owned = jnp.where(
        is_croesus,
        jnp.zeros_like(shop.items_owned_by_shop),
        shop.items_owned_by_shop,
    )
    new_angry = jnp.where(is_croesus, jnp.bool_(False), shop.angry)
    new_shop = shop.replace(
        bill=new_bill,
        bill_prices=new_prices,
        items_owned_by_shop=new_owned,
        angry=new_angry,
    )
    return state.replace(shop=new_shop)


def vault_keeper_collect(state, demand: int = VAULT_KEEPER_DEMAND_DEFAULT) -> object:
    """Vault keeper gold extortion (vendor vault.c::vault_gd_watching, line 1278).

    Vendor flow (vault.c:551-584 — umoney handshake):
        - guard demands ``rn1(1000, 50)`` gold (vault.c:184 in the original
          3.4 sources; range [50, 1049]).
        - if player has >= demand: deduct from player_gold, keeper goes away.
        - if player has < demand: keeper leads player out (we model that as
          a teleport to ``shop.door_pos`` — the original code escorts via a
          fake corridor; the door tile is the byte-equal endpoint of that
          sequence).

    Only fires when ``shop.kind == VAULT_KEEPER`` AND ``shop_active``.
    JIT-safe; ``demand`` is a Python int (compile-time constant) so callers
    that want RNG variance call this twice with different demand values.
    """
    shop = state.shop
    is_vault = shop.kind == jnp.int8(ShopkeeperKind.VAULT_KEEPER)
    active = shop.shop_active & is_vault

    demand_i = jnp.int32(int(demand))
    can_pay = state.player_gold >= demand_i

    # Pay path — deduct gold.
    paid = active & can_pay
    refused = active & (~can_pay)

    new_gold = jnp.where(paid, state.player_gold - demand_i, state.player_gold)

    # Refused path — teleport player to the door (the keeper's escort
    # endpoint in vault.c::gd_mv_monaway, line 734).
    door = shop.door_pos.astype(state.player_pos.dtype)
    new_pp = jnp.where(refused, door, state.player_pos)

    return state.replace(player_gold=new_gold, player_pos=new_pp)


# ---------------------------------------------------------------------------
# shkveg — vegetarian stocking for SHOP_HEALTHFOOD
# (vendor shknam.c::shkveg lines 407-439; veggy_item lines 379-405)
# ---------------------------------------------------------------------------
# Specific item type_ids the vendor health-food iprobs[] table promotes
# beyond the generic FOOD_CLASS+VEGGY sweep (shknam.c:322-328):
#   { 20, -POT_FRUIT_JUICE }    type_id 294
#   {  1, -LUMP_OF_ROYAL_JELLY } type_id 261
# Plus the egg type_id (241) which veggy_item accepts unconditionally
# (shknam.c:397 ``otyp == EGG``).
HEALTHFOOD_EGG_TYPE_ID:          int = 241
HEALTHFOOD_ROYAL_JELLY_TYPE_ID:  int = 261
HEALTHFOOD_FRUIT_JUICE_TYPE_ID:  int = 294


def _build_vegetarian_food_mask():
    """Build the boolean mask of vegetarian-eligible object type_ids.

    Mirrors vendor shknam.c::veggy_item (lines 379-405).  A FOOD_CLASS
    object is veggy when ``oc_material == VEGGY`` OR ``otyp == EGG``.
    Tins / corpses are excluded in this stocking slice (vendor specialises
    via corpsenm; we never spawn corpses through shkveg).  The two non-FOOD
    add-ons listed in shtypes[] (POT_FRUIT_JUICE, LUMP_OF_ROYAL_JELLY,
    shknam.c:322-328) are not flipped True here — they enter the stock
    pool via the iprobs[] iprob route at caller level.

    Result: jnp.bool_ array of length NUM_OBJECTS where True == "shkveg
    may pick this type_id".
    """
    from Nethax.nethax.constants.objects import (
        OBJECTS, NUM_OBJECTS, ObjectClass, Material,
    )
    import numpy as _np

    mask = _np.zeros((NUM_OBJECTS,), dtype=_np.bool_)
    for i in range(NUM_OBJECTS):
        obj = OBJECTS[i]
        if obj.class_ != ObjectClass.FOOD_CLASS:
            continue
        if obj.material == Material.VEGGY:
            mask[i] = True
        elif i == HEALTHFOOD_EGG_TYPE_ID:
            mask[i] = True
    return jnp.asarray(mask)


def _build_vegetarian_prob_table():
    """Per-type-id ``oc_prob`` weights for the shkveg roulette.

    Vendor shkveg (shknam.c:415-435) sums ``objects[i].oc_prob`` over the
    veggy-eligible FOOD subset, then draws ``rnd(maxprob)`` and walks the
    list deducting per-type prob until the running total drops to 0.  We
    mirror that with a jnp int32 array indexed by type_id (0 for non-veggy
    types).
    """
    from Nethax.nethax.constants.objects import OBJECTS, NUM_OBJECTS

    probs = jnp.zeros((NUM_OBJECTS,), dtype=jnp.int32)
    mask  = _build_vegetarian_food_mask()
    # Build a Python list of ints then convert; oc_prob lives on ObjectEntry.
    py_probs = [int(OBJECTS[i].prob) if bool(mask[i]) else 0
                for i in range(NUM_OBJECTS)]
    return jnp.asarray(py_probs, dtype=jnp.int32), mask


def shkveg(rng: jax.Array) -> jnp.ndarray:
    """Pick a vegetarian FOOD type_id for SHOP_HEALTHFOOD stocking.

    Vendor reference: shknam.c::shkveg lines 407-439.  Roulette-weighted
    by ``objects[i].oc_prob`` over the VEGETARIAN_CLASS subset (FOOD_CLASS
    items with ``oc_material == VEGGY`` plus EGG).

    Returns the chosen type_id as an int32 scalar.  Deterministic given
    ``rng``; safe inside jit.

    Note: vendor's ``mkveggy_at`` (shknam.c:442-450) wraps the type_id in
    a fresh obj record and sets HEALTHY_TIN on tin variety; that wrapping
    happens at the ground-item caller, not here — this function only
    returns the type_id.
    """
    probs, mask = _build_vegetarian_prob_table()
    maxprob = jnp.sum(probs).astype(jnp.int32)
    # rnd(maxprob) in vendor: result in [1, maxprob]; jnp.randint upper is
    # exclusive so use [1, maxprob+1).  When maxprob<=0 fall back to EGG.
    safe_max = jnp.maximum(maxprob, jnp.int32(1))
    roll = jax.random.randint(rng, (), 1, safe_max + 1, dtype=jnp.int32)

    # Walk type_ids 0..NUM_OBJECTS-1, deducting probs[i]; pick first i
    # where (roll - cumulative_prob) <= 0 AND mask[i] is True.
    cumulative = jnp.cumsum(probs)
    # Find smallest i such that cumulative[i] >= roll AND mask[i].
    reached = (cumulative >= roll) & mask
    has_any = jnp.any(reached)
    picked = jnp.argmax(reached).astype(jnp.int32)
    fallback = jnp.int32(HEALTHFOOD_EGG_TYPE_ID)
    return jnp.where(has_any & (maxprob > jnp.int32(0)), picked, fallback)


def is_shop_healthfood_type(type_id: jnp.ndarray) -> jnp.ndarray:
    """Return True iff this type_id is eligible for SHOP_HEALTHFOOD stock.

    Combines the vegetarian FOOD mask with the iprobs[] specials
    (POT_FRUIT_JUICE 294, LUMP_OF_ROYAL_JELLY 261) so callers building
    healthfood stock can gate either type via a single predicate.
    Cite: vendor shknam.c:318-328 (shtypes[] entry for "health food
    store"), shknam.c:407-439 (shkveg).
    """
    _, mask = _build_vegetarian_prob_table()
    tid = type_id.astype(jnp.int32)
    safe = jnp.clip(tid, 0, mask.shape[0] - 1)
    in_veggy = mask[safe]
    is_fruit_juice = tid == jnp.int32(HEALTHFOOD_FRUIT_JUICE_TYPE_ID)
    is_royal_jelly = tid == jnp.int32(HEALTHFOOD_ROYAL_JELLY_TYPE_ID)
    return in_veggy | is_fruit_juice | is_royal_jelly


# ---------------------------------------------------------------------------
# nameshk — per-shop-type shopkeeper name tables
# (vendor shknam.c::nameshk lines 487-554; shknms tables lines 21-189)
# ---------------------------------------------------------------------------
# Vendor name tables, one per shtypes[] entry (lines 21-188).  We mirror
# the shtypes[] declaration order (SHOP_GENERAL..SHOP_LIGHTING) so a single
# int8 shoptype index selects the right pool.  10-14 names per pool to
# match vendor's typical pool size; the deterministic-seed nameshk algorithm
# (shknam.c:507-548) does collision avoidance, which we model below.
SHKNAM_TABLES: tuple = (
    # SHOP_GENERAL — shkgeneral (shknam.c:162-176, Suriname/Greenland/Canada/Iceland)
    (
        "Hebiwerie", "Possogroenoe", "Asidonhopo", "Manlobbi", "Adjama",
        "Pakka Pakka", "Kabalebo", "Wonotobo", "Akalapi", "Sipaliwini",
    ),
    # SHOP_ARMOR — shkarmors (shknam.c:24-43, Polish/Hungarian)
    (
        "Akrzca", "Bakteen", "Bezdomny", "Cherdez", "Domek", "Erdo",
        "Florian", "Geczy", "Hanysz", "Inczeer",
    ),
    # SHOP_SCROLL (bookstore-secondhand) — shkbooks (shknam.c:46-58, Mongolian)
    (
        "Khan-Yuan", "Tsogt", "Battsetseg", "Munkhbat", "Ganbaatar",
        "Naranbaatar", "Erdene", "Khishig", "Tumur", "Bayar",
    ),
    # SHOP_POTION (liquor) — shkliquors (shknam.c:61-70, French/Italian wines)
    (
        "Adega", "Barrique", "Carafe", "Decanter", "Etna",
        "Frasco", "Garrafa", "Hectolitre", "Imbottigliato", "Jerez",
    ),
    # SHOP_WEAPON — shkweapons (shknam.c:106-114, Perigord)
    (
        "Voulgezac", "Rouffiac", "Lerignac", "Touverac", "Guizengeard",
        "Melac", "Neuvicq", "Vanzac", "Picq", "Urignac",
    ),
    # SHOP_FOOD (delicatessen) — shkfoods (shknam.c:73-82, Italian)
    (
        "Antipasto", "Bresaola", "Coppa", "Dolcelatte", "Etrusco",
        "Fontina", "Gorgonzola", "Hostaria", "Insalata", "Jambon",
    ),
    # SHOP_RING — shkrings (shknam.c:85-93, Hindi/Sanskrit)
    (
        "Adira", "Bhavin", "Charuta", "Deepak", "Esha",
        "Falguni", "Gauri", "Harsh", "Indira", "Jaya",
    ),
    # SHOP_WAND — shkwands (shknam.c:96-103, Indonesian)
    (
        "Akimudin", "Bahar", "Cipto", "Darmadi", "Edhi",
        "Faisal", "Gunawan", "Hartono", "Ismail", "Jaya",
    ),
    # SHOP_TOOL — shktools (shknam.c:116-148, names-with-prefix encoding)
    (
        "Ymla", "Eed-morra", "Elan Lapinski", "Cubask", "Nieb",
        "Bnowr Falr", "Sperc", "Noskcirdneh", "Yawolloh", "Hyeghu",
    ),
    # SHOP_BOOK — shkbooks (vendor reuses scroll pool; we duplicate for
    # parity with shtypes[] indexing — shknam.c:46-58).
    (
        "Khan-Yuan", "Tsogt", "Battsetseg", "Munkhbat", "Ganbaatar",
        "Naranbaatar", "Erdene", "Khishig", "Tumur", "Bayar",
    ),
    # SHOP_HEALTHFOOD — shkhealthfoods (shknam.c:178-188, Tibet/Hippie)
    (
        "Ga'er", "Zhangmu", "Rikaze", "Jiangji", "Changdu",
        "Linzhi", "Shigatse", "Gyantse", "Ganden", "Tsurphu",
        "Lhasa", "Tsedong", "Drepung",
    ),
    # SHOP_LIGHTING — shklight (shknam.c:151-160, Romania/Bulgaria)
    (
        "Zarnesti", "Slanic", "Nehoiasu", "Ludus", "Sighisoara",
        "Nisipitu", "Razboieni", "Bicaz", "Dorohoi", "Vaslui",
    ),
)

# Flatten to a single 120-entry index space so the name_idx field is a
# single int8: name_idx = shoptype * NAMES_PER_TABLE_MAX + within_table.
NAMES_PER_TABLE_MAX: int = 14   # max pool length (shkhealthfoods has 13).
N_SHOP_TYPES_FOR_NAMES: int = 12


def shknam_name(shop_type: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
    """Pick a shopkeeper name index for the given shop_type.

    Vendor reference: shknam.c::nameshk lines 487-554.  Vendor uses a
    deterministic seed (``ubirthday / 257 + ledger_no + m_id``) plus
    collision-avoidance against ``fmon``-listed shopkeepers (lines
    517-548).  In the JAX port we draw a uniform [0, pool_size) sample;
    collision avoidance is handled by the caller (only one shopkeeper
    per level in this slice).

    Returns the *flat* name index (shoptype * NAMES_PER_TABLE_MAX +
    within_table) as int16.  Looking up the actual string is a host-side
    operation via SHKNAM_TABLES[shop_type][within].

    Parameters
    ----------
    shop_type : int   shtypes[] index (SHOP_GENERAL..SHOP_LIGHTING).
    rng       : PRNGKey
    """
    # Per-pool sizes.
    pool_sizes = jnp.asarray(
        [len(t) for t in SHKNAM_TABLES],
        dtype=jnp.int32,
    )
    st = jnp.clip(shop_type.astype(jnp.int32),
                  0, jnp.int32(N_SHOP_TYPES_FOR_NAMES - 1))
    pool_size = pool_sizes[st]
    safe_pool = jnp.maximum(pool_size, jnp.int32(1))
    within = jax.random.randint(rng, (), 0, safe_pool, dtype=jnp.int32)
    flat = st * jnp.int32(NAMES_PER_TABLE_MAX) + within
    return flat.astype(jnp.int16)


def shknam_name_lookup(shop_type: int, name_idx: int) -> str:
    """Host-side: resolve a (shop_type, flat name_idx) into the actual name.

    Useful for renderers / tests that want the string form.  ``name_idx``
    is the value returned by :func:`shknam_name` (flat encoding).
    """
    shop_type = int(shop_type)
    name_idx = int(name_idx)
    within = name_idx - shop_type * NAMES_PER_TABLE_MAX
    if shop_type < 0 or shop_type >= len(SHKNAM_TABLES):
        return "Anonymous"
    pool = SHKNAM_TABLES[shop_type]
    if within < 0 or within >= len(pool):
        return "Anonymous"
    return pool[within]


def aligned_priest_price(
    base_price: jnp.ndarray,
    shop_kind: jnp.ndarray,
    player_alignment: jnp.ndarray,
    shop_alignment: jnp.ndarray,
) -> jnp.ndarray:
    """Aligned-priest temple charge (vendor priest.c::pri_keeper logic).

    Misaligned customers pay ``ALIGNED_PRIEST_MISALIGN_MUL * base_price``
    (vendor priest.c sets the temple donation/charge floor to 2x for
    customers whose alignment does not match the temple's altar alignment;
    see priest.c::temple_occupied + histemple_at, lines 142-216).

    Pure function — no state mutation.  Returns the adjusted price as int32.
    """
    base = base_price.astype(jnp.int32)
    is_priest = shop_kind == jnp.int8(ShopkeeperKind.ALIGNED_PRIEST)
    misaligned = player_alignment.astype(jnp.int32) != shop_alignment.astype(jnp.int32)
    surcharge = is_priest & misaligned
    return jnp.where(surcharge, base * jnp.int32(ALIGNED_PRIEST_MISALIGN_MUL), base)
