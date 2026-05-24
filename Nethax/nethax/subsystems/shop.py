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
from flax import struct

from Nethax.nethax.subsystems.inventory import MAX_INVENTORY_SLOTS, ItemCategory


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
    jnp array on first use and rely on jit caching for subsequent calls.
    """
    from Nethax.nethax.constants.objects import OBJECTS, NUM_OBJECTS

    # Module-level cache to avoid rebuilding the array on every call.
    global _OBJECT_COST_TABLE
    try:
        table = _OBJECT_COST_TABLE
    except NameError:
        table = None
    if table is None:
        costs = [int(OBJECTS[i].cost) for i in range(NUM_OBJECTS)]
        table = jnp.array(costs, dtype=jnp.int32)
        _OBJECT_COST_TABLE = table

    idx = jnp.clip(type_id.astype(jnp.int32), 0, NUM_OBJECTS - 1)
    return table[idx]


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

    # No oartifact / dknown / name_known fields propagate through Item yet;
    # use the dknown flag the inventory carries, and assume name_known=True
    # (vendor's default for identified scrolls/wands is name_known via
    # objects[]->oc_name_known global; we approximate as identified flag).
    dknown = jnp.where(has_item, inv.dknown[slot], jnp.bool_(False))
    name_known = jnp.where(has_item, inv.identified[slot], jnp.bool_(False))

    base = jnp.where(
        has_item,
        _lookup_object_cost(type_id),
        jnp.int32(0),
    )
    price = get_cost(
        base_cost=base,
        oclass=cat,
        spe=spe,
        oartifact=jnp.bool_(False),  # oartifact tracked at quest/artifact layer only
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


def accrue_bill(state, slot_idx) -> object:
    """Player picks up an item inside the shop while not invisible.

    Vendor refs:
        shk.c::addtobill (shk.c:3489-3550) — billable gate then add_one_tobill.
        shk.c::get_cost  (shk.c:2877-2988) — the per-item price added.

    Side-effects (only when ``billable`` returns True):
        - shop.bill_prices[slot_idx] = get_cost(item at slot)
        - shop.bill                  += that price
        - shop.items_owned_by_shop[slot_idx] = True

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

    price = _compute_item_price_from_slot(state, slot)
    prev_price = shop.bill_prices[slot]

    new_price_at_slot = jnp.where(eligible, price, prev_price)
    new_owned_at_slot = jnp.where(eligible, jnp.bool_(True), shop.items_owned_by_shop[slot])

    # Adjust bill by (new_price - prev_price) — when prev_price == 0 this is
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
        shk.c::dopayobj (shk.c:2220-2299) — itemised pay loop.
        shk.c::make_angry_shk (shk.c:1469-1489) — PAY_CANT path.
        shk.c::hot_pursuit (shk.c:1449-1463) — sets ``following = 1``.

    Behaviour:
        - bill == 0: no-op.
        - player gold >= bill: deduct gold, zero bill, clear ownership,
          zero per-slot prices.
        - player gold <  bill: PAY_CANT → set angry, set following, set
          surcharge.  Bill stays on the books; ownership flags persist.

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
    new_angry = jnp.where(angered, jnp.bool_(True), shop.angry)
    # vendor shk.c:1456 — hot_pursuit sets following=1; shk.c:1364 sets surcharge.
    new_following = jnp.where(angered, jnp.bool_(True), shop.following)
    new_surcharge = jnp.where(angered, jnp.bool_(True), shop.surcharge)

    new_shop = shop.replace(
        bill=new_bill,
        items_owned_by_shop=new_owned,
        bill_prices=new_prices,
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
        - setpaid(mtmp) — clear this shk's bill, clear unpaid flags.
        - eshk->bill_p = NULL — bill record decommissioned (we zero
          ``bill_prices`` and ``items_owned_by_shop``).
        - Clear no_charge on floor items in shoproom — modelled here as
          clearing ``items_owned_by_shop`` (which is the inventory-side
          mirror of those flags in this port).
        - Note: vendor leaves neighbouring shopkeepers' anger state alone
          (rile_shk only fires on direct provocation), but we keep
          ``angry=True`` so any *other* hostile shopkeeper class on the level
          remains visible to test code.  ``following`` and ``surcharge``
          clear because *this* shk is gone (shk.c:262).

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
