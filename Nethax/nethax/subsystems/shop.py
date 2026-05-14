"""Shop subsystem — simplified shopkeeper bill / pay-at-exit / angry mode.

Canonical sources:
  vendor/nethack/src/shk.c    — shopkeeper AI, billing, theft detection,
                                 buy/sell pricing, anger/peaceful logic (~6125 LOC)
  vendor/nethack/src/shknam.c — shop type table, shopkeeper name generation,
                                 shknam() / init_shk(); shop rtype constants
                                 (SHOPBASE, GENERAL_SHOP … CANDLESHOP)

Scope (Wave 6 — simplified shops):
    SUPPORTED
      - bill accrual on pickup inside the shop region
      - pay-at-exit when player crosses the door tile (or a designated
        threshold tile) leaving the shop
      - angry shopkeeper mode (theft / kill shopkeeper / unpaid exit):
        sticks to the player on the level and attacks at melee with
        boosted damage

    OUT OF SCOPE (deferred)
      - Haggling / price negotiation
      - Theft detection (other than unpaid-exit)
      - Multi-shop levels (Wave 6 supports max one shop per level)
      - Chat dialogue (shknam.c: shkname)
      - Robbery / bounty system
      - Pricing tables — all items are billed at a flat default price.
"""
import jax
import jax.numpy as jnp
from flax import struct

from Nethax.nethax.subsystems.inventory import MAX_INVENTORY_SLOTS


# ---------------------------------------------------------------------------
# Shop type constants (shknam.c rtype values, zero-indexed for this project)
# ---------------------------------------------------------------------------
SHOP_NONE = 0        # no shop on this level
SHOP_GENERAL = 1     # general store  (shknam.c: GENERAL_SHOP)
SHOP_WEAPONS = 2     # weapon shop    (WEAPON_SHOP)
SHOP_ARMOR = 3       # armor shop     (ARMOR_SHOP)
SHOP_POTIONS = 4     # potion shop    (POTION_SHOP)
SHOP_SCROLLS = 5     # scroll shop    (SCROLL_SHOP)
SHOP_WANDS = 6       # wand shop      (WAND_SHOP)
SHOP_RINGS = 7       # ring shop      (RING_SHOP)
SHOP_TOOLS = 8       # tool shop      (TOOL_SHOP)
SHOP_FOOD = 9        # delicatessen   (FOOD_SHOP)
SHOP_LIGHTING = 10   # lighting shop  (LIGHT_SHOP)
SHOP_BOOKS = 11      # bookstore      (BOOK_SHOP)


# Default price used by accrue_bill — Wave 6 simplification (no real pricing
# tables yet; mirrors vendor/nethack/src/shk.c::get_cost flat default of 5gp).
DEFAULT_ITEM_PRICE: int = 10

# Damage applied by the shopkeeper each turn in angry mode.
SHOPKEEPER_ANGRY_DAMAGE: int = 8


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@struct.dataclass
class ShopState:
    """One shop per level (Wave 6 simplification: max 1 shop per level).

    Fields
    ------
    shop_active           : bool  — True if the current level has a shop
    shopkeeper_idx        : int8  — monster slot for the shopkeeper
                                    (-1 when no shop)
    shop_room_min         : int8[2] — (row, col) of the shop bounds upper-left
    shop_room_max         : int8[2] — (row, col) of the shop bounds lower-right
    door_pos              : int8[2] — (row, col) of the shop entrance tile;
                                      used as the pay-at-exit trigger
    bill                  : int32 — total gold owed by the player
    items_owned_by_shop   : bool[MAX_INVENTORY_SLOTS] — True if picked up in
                                      shop and not yet paid for
    angry                 : bool  — angry mode (theft / killed shopkeeper /
                                    skipped payment)
    """

    shop_active: jnp.ndarray            # scalar bool
    shopkeeper_idx: jnp.ndarray         # scalar int8
    shop_room_min: jnp.ndarray          # [2] int8 (row, col)
    shop_room_max: jnp.ndarray          # [2] int8 (row, col)
    door_pos: jnp.ndarray               # [2] int8 (row, col)
    bill: jnp.ndarray                   # scalar int32
    items_owned_by_shop: jnp.ndarray    # [MAX_INVENTORY_SLOTS] bool
    angry: jnp.ndarray                  # scalar bool

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
        )


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def accrue_bill(state, slot_idx) -> object:
    """Player picks up an item inside the shop while not invisible.

    Mirrors vendor/nethack/src/shk.c::addtobill (shk.c:3489-3550):
        - on pickup in a shop, addtobill() flags the item as unpaid and
          adds its cost (via get_cost, shk.c:3517) to the shopkeeper's
          running bill via add_one_tobill (shk.c:3548).
        - we approximate the cost with DEFAULT_ITEM_PRICE (vendor's
          flat-fallback price branch in get_cost).

    Side-effects (only when the shop is active AND the player tile is
    inside the shop room):
        - shop.bill += DEFAULT_ITEM_PRICE
        - shop.items_owned_by_shop[slot_idx] = True

    JIT-safe; no Python branching on traced values.

    Returns: new EnvState.
    """
    shop = state.shop
    n = shop.items_owned_by_shop.shape[0]
    slot = jnp.clip(jnp.asarray(slot_idx, dtype=jnp.int32), 0, n - 1)

    row = state.player_pos[0]
    col = state.player_pos[1]
    in_room = _pos_in_shop_room(shop, row, col)
    eligible = shop.shop_active & in_room

    new_bill = jnp.where(
        eligible,
        shop.bill + jnp.int32(DEFAULT_ITEM_PRICE),
        shop.bill,
    )
    new_owned_at_slot = jnp.where(
        eligible,
        jnp.bool_(True),
        shop.items_owned_by_shop[slot],
    )
    new_owned = shop.items_owned_by_shop.at[slot].set(new_owned_at_slot)
    new_shop = shop.replace(bill=new_bill, items_owned_by_shop=new_owned)
    return state.replace(shop=new_shop)


def drop_in_shop(state, slot_idx) -> object:
    """Player drops an item still owned by the shop while inside the shop room.

    Mirrors the relevant clause in vendor/nethack/src/shk.c::dropped_container /
    sellobj_state cleanup: dropping unpaid stock back inside the shop clears
    the unpaid flag and refunds the line item from the running bill.

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

    new_bill = jnp.where(
        eligible,
        jnp.maximum(jnp.int32(0), shop.bill - jnp.int32(DEFAULT_ITEM_PRICE)),
        shop.bill,
    )
    new_owned = shop.items_owned_by_shop.at[slot].set(
        jnp.where(eligible, jnp.bool_(False), was_owned)
    )
    new_shop = shop.replace(bill=new_bill, items_owned_by_shop=new_owned)
    return state.replace(shop=new_shop)


def pay_at_exit(state) -> object:
    """Called when the player crosses the shop door tile leaving the shop.

    Mirrors vendor/nethack/src/shk.c::dopayobj (shk.c:2220-2299) and the
    PAY_CANT path (shk.c:2283-2285 → insufficient_funds → make_angry_shk
    at shk.c:1469-1489) for the simplified Wave 6 shop:

        - if bill == 0: no-op (handles zero-bill walk-out cleanly).
        - if player gold >= bill: deduct gold via pay() (shk.c:2288),
          zero bill via setpaid (shk.c:400), clear all
          items_owned_by_shop flags.
        - if player gold <  bill: PAY_CANT → make_angry_shk → hot_pursuit
          (shopkeeper goes ANGRY).  Bill stays on the books; item
          ownership flags persist.

    Trigger condition: shop_active AND player at door tile AND bill > 0.

    JIT-safe.
    """
    shop = state.shop
    row = state.player_pos[0]
    col = state.player_pos[1]
    on_door = _pos_on_door(shop, row, col)
    trigger = shop.shop_active & on_door & (shop.bill > jnp.int32(0))

    can_pay = state.player_gold >= shop.bill

    # Pay path: deduct gold, zero bill, clear ownership.
    new_gold_pay = state.player_gold - shop.bill
    cleared = jnp.zeros_like(shop.items_owned_by_shop)

    # Resolve gold & bill via where so JIT trace is single-branch.
    paid = trigger & can_pay
    angered = trigger & ~can_pay

    new_gold = jnp.where(paid, new_gold_pay, state.player_gold)
    new_bill = jnp.where(paid, jnp.int32(0), shop.bill)
    new_owned = jnp.where(paid, cleared, shop.items_owned_by_shop)
    new_angry = jnp.where(angered, jnp.bool_(True), shop.angry)

    new_shop = shop.replace(
        bill=new_bill,
        items_owned_by_shop=new_owned,
        angry=new_angry,
    )
    return state.replace(shop=new_shop, player_gold=new_gold)


def shopkeeper_attack(state, rng) -> object:
    """When angry, shopkeeper moves toward player and attacks at melee.

    Mirrors vendor/nethack/src/shk.c::hot_pursuit (shk.c:1449-1463):
        - rile_shk sets NOTANGRY(shkp) = FALSE (shk.c:1364).
        - ESHK(shkp)->following = 1 marks pursuit (shk.c:1456).
        - angry shopkeepers track the player anywhere on the level.
        - in melee range (Chebyshev distance <= 1) they hit for boosted
          damage (vendor's mattackq path uses mhitu, attack power scales
          with shopkeeper level + STR; Wave 6 uses a flat
          SHOPKEEPER_ANGRY_DAMAGE).

    Wave 6 simplification:
        - The shopkeeper's position is read from monster_ai.pos[shopkeeper_idx]
          and *snapped* to the player's adjacent tile (we don't run a full
          pathfinder for the simplified shop NPC).
        - On melee contact, player HP is reduced by SHOPKEEPER_ANGRY_DAMAGE.
        - No HP / death tracking for the shopkeeper here; killing the
          shopkeeper is handled via kill_shopkeeper (called from combat).

    JIT-safe.  ``rng`` is reserved for future damage-roll variation.
    """
    del rng  # damage is deterministic in Wave 6
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

    # Only update the shopkeeper slot when actively pursuing.
    pursuing = shop.shop_active & shop.angry & has_shk & alive
    new_pos_arr = jnp.where(
        pursuing,
        mai.pos.at[idx].set(new_shk_pos),
        mai.pos,
    )

    # Adjacent after the step?  (Chebyshev dist <= 1.)
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
    """Player killed the shopkeeper.

    Mirrors vendor/nethack/src/shk.c shopkeeper-death cleanup (shk.c:258-
    267: when ``mtmp`` is shk and ``ESHK(mtmp)->bill_p`` is non-null,
    setpaid(mtmp) clears the bill and resets unpaid flags).  The death of
    a shopkeeper does not pacify other shopkeepers — vendor's
    ``angry_guards`` / ``rile_shk`` flow keeps remaining shopkeepers
    hostile.  In our simplified one-shop-per-level model that becomes:
    keep angry=True so any remaining shopkeeper class on the level
    (e.g. Mine Town watchmen treated as neighbours by tests) stays
    hostile.

    Vendor refs:
      shk.c:261 setpaid(mtmp);            # clear this shk's bill
      shk.c:262 eshk->bill_p = (struct bill_x *) 0;
      shk.c:1469 make_angry_shk()         # related shopkeepers stay angry

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

    cleared = jnp.zeros_like(shop.items_owned_by_shop)
    new_shop = shop.replace(
        bill=jnp.where(will_kill, jnp.int32(0), shop.bill),
        items_owned_by_shop=jnp.where(will_kill, cleared, shop.items_owned_by_shop),
        angry=jnp.where(will_kill, jnp.bool_(True), shop.angry),
    )
    return state.replace(shop=new_shop, monster_ai=new_mai)


def shop_step(state, rng) -> object:
    """Per-turn tick — handles automatic pay-at-exit + angry shopkeeper pursuit.

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


# ---------------------------------------------------------------------------
# Legacy stubs preserved for backwards compatibility with Wave 1 callers.
# ---------------------------------------------------------------------------
def enter_shop(state: ShopState, level: int) -> ShopState:
    """No-op enter-shop stub (shk.c: inshop / you_enter_shop).

    Retained from Wave 1 for callers that take only the ShopState slice.
    Wave 6 uses accrue_bill / pay_at_exit directly on EnvState.
    """
    return state


def pickup_in_shop(state: ShopState, slot: int, level: int) -> ShopState:
    """Legacy no-op — Wave 6 callers should use accrue_bill(EnvState, slot)."""
    return state


def pay_bill(state: ShopState, level: int) -> ShopState:
    """Legacy no-op — Wave 6 callers should use pay_at_exit(EnvState)."""
    return state


def attack_shopkeeper(state: ShopState, level: int) -> ShopState:
    """Legacy no-op — Wave 6 callers should use kill_shopkeeper(EnvState)."""
    return state


def step(state: ShopState, rng: jax.Array) -> ShopState:
    """Per-turn no-op for the ShopState slice.

    Note: the new Wave 6 per-turn shop tick lives in ``shop_step`` and
    takes a full EnvState.  This ShopState-only ``step`` is preserved as a
    no-op so the Wave 1 test (tests/test_no_op_step.py::test_shop_step_noop)
    keeps passing.
    """
    return state
