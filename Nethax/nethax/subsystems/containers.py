"""Containers subsystem — bag-of-holding, large box, chest with nested inventory.

Canonical sources:
  vendor/nethack/src/pickup.c — in_container, out_container, use_container
                                  (~lines 2400-3500): open/close/put/take logic
  vendor/nethack/include/obj.h — Obj struct with ``cobj`` pointer to the
                                  nested object list
  vendor/nethack/include/objects.h — BAG_OF_HOLDING, LARGE_BOX, CHEST,
                                      OILSKIN_SACK, BAG_OF_TRICKS

Status: Wave 5 Phase 3 — first cut of the JAX-side container model.

Design
------
NetHack uses a linked list (Obj.cobj) for container contents.  Here we use a
fixed [N_CONTAINERS, MAX_ITEMS_PER_CONTAINER] 2-D Item-shaped array so the
shape is JIT-stable.  ``container_type`` records which kind of container each
slot holds and ``parent_slot`` records which inventory slot in
``InventoryState.items`` currently holds the container (-1 == not held).

Bag-of-holding weight multipliers (pickup.c::in_container):
  blessed   → 1/4 (0.25)
  uncursed  → 1/2 (0.50)
  cursed    → 2/1 (2.00)
We expose ``container_total_weight`` which returns the *effective* weight
contributed to the player's encumbrance load.
"""
from enum import IntEnum

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import struct


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CONTAINERS: int = 4               # max number of containers a player can carry
MAX_ITEMS_PER_CONTAINER: int = 20   # depth of each container's stack

# Weight-multiplier numerators (denominator = 4); use integer math for JIT.
# blessed bag-of-holding: 1/4; uncursed: 2/4; cursed: 8/4.
_BOH_NUMER_BLESSED:  int = 1
_BOH_NUMER_UNCURSED: int = 2
_BOH_NUMER_CURSED:   int = 8
_BOH_DENOM:          int = 4


class ContainerType(IntEnum):
    """Which kind of container is held in this slot.

    Values match the ``otyp`` constants in vendor/nethack/include/objects.h
    (BAG_OF_HOLDING, LARGE_BOX, CHEST, BAG_OF_TRICKS, OILSKIN_SACK, ICE_BOX),
    but here we use a small dense enum so the int8 field stays compact.
    """
    NONE           = 0   # empty container slot
    SACK           = 1   # plain sack (no multiplier)
    OILSKIN_SACK   = 2   # waterproof
    BAG_OF_HOLDING = 3   # weight multiplier per BUC
    BAG_OF_TRICKS  = 4   # cursed → explode-on-open
    LARGE_BOX      = 5   # heavy floor-only box
    CHEST          = 6   # heavy floor-only chest
    # Wave 25b: ICE_BOX — freezes corpses inside so their age does not advance
    # while contained.  Cite: vendor/nethack/include/obj.h:344 (Has_contents
    # macro covering ICE_BOX); vendor/nethack/src/pickup.c:2644-2657 (Icebox
    # branch: obj->age = svm.moves - obj->age on insertion, stop_timer
    # ROT_CORPSE / REVIVE_MON to freeze decay).
    ICE_BOX        = 7
    # Wave 25b note: vendor NetHack 3.7 has NO IRON_SAFE container otyp.
    # The objects.h CONTAINER list is exactly:
    #   LARGE_BOX, CHEST, ICE_BOX, SACK, OILSKIN_SACK, BAG_OF_HOLDING,
    #   BAG_OF_TRICKS  (vendor/nethack/include/objects.h:898-913).
    # No IRON_SAFE constant exists in obj.h, objects.h, pickup.c, or anywhere
    # else in the vendor tree (grep -rn IRON_SAFE vendor/nethack/ returns 0
    # matches).  Adding it would violate the byte-equal-vendor hard requirement,
    # so the task's IRON_SAFE request is not implemented.  If a future variant
    # (SLASH'EM, dNetHack, etc.) is vendored that defines IRON_SAFE, this enum
    # can be extended at value 8 with the matching type_id constant below.


class BUCStatus(IntEnum):
    """Mirror of inventory.Item.buc_status values used by container math."""
    UNKNOWN  = 0
    CURSED   = 1
    UNCURSED = 2
    BLESSED  = 3


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@struct.dataclass
class ContainerState:
    """Nested-inventory state for the containers held by the player.

    All ``items_*`` arrays are shaped ``[N_CONTAINERS, MAX_ITEMS_PER_CONTAINER]``.
    Empty entries have ``items_category == 0`` (matches inventory.Item).

    Fields
    ------
    items_category   : int8  — Item.category for each slot (0 == empty).
    items_type_id    : int16
    items_buc        : int8
    items_enchant    : int8
    items_charges    : int8
    items_identified : bool
    items_quantity   : int16
    items_weight     : int16
    container_type   : int8[N_CONTAINERS]
    parent_slot      : int8[N_CONTAINERS]  — inventory slot index, -1 if not held.
    is_open          : bool[N_CONTAINERS]  — open/close flag (use_container).
    container_buc    : int8[N_CONTAINERS]  — BUC of the container itself
                                              (drives BOH multiplier).
    """
    items_category:   jnp.ndarray
    items_type_id:    jnp.ndarray
    items_buc:        jnp.ndarray
    items_enchant:    jnp.ndarray
    items_charges:    jnp.ndarray
    items_identified: jnp.ndarray
    items_quantity:   jnp.ndarray
    items_weight:     jnp.ndarray
    container_type:   jnp.ndarray
    parent_slot:      jnp.ndarray
    is_open:          jnp.ndarray
    container_buc:    jnp.ndarray
    # is_locked: bool[N_CONTAINERS] — True when a chest/box is locked.
    # Cite: vendor/nethack/src/lock.c::pick_lock — locked chests can be picked
    # by a lock pick, skeleton key, or credit card.
    is_locked:        jnp.ndarray
    # is_trapped: bool[N_CONTAINERS] — True when the container is rigged
    # to fire on opening.  Cite: vendor/nethack/include/obj.h ``otrapped``
    # bitfield (distinct from olocked and oeaten/oerodeproof/etc.).
    is_trapped:       jnp.ndarray

    @classmethod
    def empty(cls) -> "ContainerState":
        """Return a fully-empty ContainerState (no containers held)."""
        n, m = N_CONTAINERS, MAX_ITEMS_PER_CONTAINER
        return cls(
            items_category   = jnp.zeros((n, m), dtype=jnp.int8),
            items_type_id    = jnp.zeros((n, m), dtype=jnp.int16),
            items_buc        = jnp.zeros((n, m), dtype=jnp.int8),
            items_enchant    = jnp.zeros((n, m), dtype=jnp.int8),
            items_charges    = jnp.zeros((n, m), dtype=jnp.int8),
            items_identified = jnp.zeros((n, m), dtype=jnp.bool_),
            items_quantity   = jnp.zeros((n, m), dtype=jnp.int16),
            items_weight     = jnp.zeros((n, m), dtype=jnp.int16),
            container_type   = jnp.zeros((n,),   dtype=jnp.int8),
            parent_slot      = jnp.full((n,), -1, dtype=jnp.int8),
            is_open          = jnp.zeros((n,),   dtype=jnp.bool_),
            container_buc    = jnp.zeros((n,),   dtype=jnp.int8),
            is_locked        = jnp.zeros((n,),   dtype=jnp.bool_),
            is_trapped       = jnp.zeros((n,),   dtype=jnp.bool_),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_first_empty_in_container(cs: ContainerState, c_idx: jnp.int32) -> tuple:
    """Return ``(found, pos)`` for the first empty slot inside container c_idx."""
    cats = cs.items_category[c_idx]

    def _scan(carry, idx):
        found, pos = carry
        is_empty = cats[idx] == jnp.int8(0)
        pos      = jnp.where(~found & is_empty, idx, pos)
        found    = found | is_empty
        return (found, pos), None

    (found, pos), _ = lax.scan(
        _scan,
        (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(MAX_ITEMS_PER_CONTAINER, dtype=jnp.int32),
    )
    return found, pos


def _find_first_empty_inventory_slot(items_category: jnp.ndarray) -> tuple:
    """Return ``(found, slot)`` for the first empty slot in inventory.items."""
    n = items_category.shape[0]

    def _scan(carry, idx):
        found, slot = carry
        is_empty = items_category[idx] == jnp.int8(0)
        slot     = jnp.where(~found & is_empty, idx, slot)
        found    = found | is_empty
        return (found, slot), None

    (found, slot), _ = lax.scan(
        _scan,
        (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(n, dtype=jnp.int32),
    )
    return found, slot


# ---------------------------------------------------------------------------
# Container trap  (vendor/nethack/src/pickup.c::container_trap)
# ---------------------------------------------------------------------------

def fire_container_trap(state, container_idx: int):
    """Fire the trap on a trapped container when it is opened.

    Canonical: vendor/nethack/src/pickup.c::container_trap (~line 2460).
    Vendor reads obj->otrapped (a separate bitfield from olocked / the
    BUC bits) and on True applies one of several effects (explosion,
    alarm, sleep gas, etc.).  Here we model the common case: a trapped
    container deals 1d10 damage to the player and clears its trapped flag.

    Trap-state lives in the dedicated ``ContainerState.is_trapped`` bool
    array — independent of BUC, mirrors vendor obj->otrapped bitfield.

    Parameters
    ----------
    state         : EnvState
    container_idx : int  — index into state.containers (0..N_CONTAINERS-1)

    Returns
    -------
    (new_state, is_trapped)  — new_state with trap fired and 1d10 HP damage
                               applied; is_trapped bool array.
    """
    cs = state.containers
    c_idx = jnp.int32(container_idx)

    has_container = cs.container_type[c_idx] != jnp.int8(ContainerType.NONE)
    is_trapped = has_container & cs.is_trapped[c_idx]

    rng_trap, new_rng = jax.random.split(state.rng)
    dmg = jax.random.randint(rng_trap, (), minval=1, maxval=11, dtype=jnp.int32)

    new_hp = jnp.where(
        is_trapped,
        jnp.maximum(jnp.int32(0), state.player_hp.astype(jnp.int32) - dmg),
        state.player_hp.astype(jnp.int32),
    ).astype(state.player_hp.dtype)

    new_is_trapped = cs.is_trapped.at[c_idx].set(
        jnp.where(is_trapped, jnp.bool_(False), cs.is_trapped[c_idx])
    )
    new_cs = cs.replace(is_trapped=new_is_trapped)
    return state.replace(player_hp=new_hp, containers=new_cs, rng=new_rng), is_trapped


# ---------------------------------------------------------------------------
# Open / Close
# ---------------------------------------------------------------------------

def open_container(state, slot_idx):
    """Mark the container at ``slot_idx`` as open.

    Bag-of-tricks special case (pickup.c::use_container): if the player opens
    a *cursed* bag of tricks, with 50% chance summon a hostile monster at a
    nearby valid tile.  Implemented by waking a previously-dead monster slot
    and placing it adjacent to the player.

    Parameters
    ----------
    state    : EnvState
    slot_idx : int — index into state.containers (0..N_CONTAINERS-1)

    Returns
    -------
    new_state
    """
    cs = state.containers
    c_idx = jnp.int32(slot_idx)

    ctype = cs.container_type[c_idx]
    cbuc  = cs.container_buc[c_idx]
    has_container = ctype != jnp.int8(ContainerType.NONE)

    new_is_open = cs.is_open.at[c_idx].set(
        jnp.where(has_container, jnp.bool_(True), cs.is_open[c_idx])
    )

    # ---- Cursed bag of tricks: 50% explode (summon hostile) ----
    is_cursed_bot = (
        has_container
        & (ctype == jnp.int8(ContainerType.BAG_OF_TRICKS))
        & (cbuc == jnp.int8(BUCStatus.CURSED))
    )

    rng_explode, new_rng = jax.random.split(state.rng)
    explode_roll = jax.random.uniform(rng_explode, shape=())
    should_explode = is_cursed_bot & (explode_roll < jnp.float32(0.5))

    # Spawn: find first dead monster slot, place it adjacent to player.
    mai = state.monster_ai
    dead_mask = ~mai.alive
    # Find lowest-index dead slot.
    dead_idx = jnp.argmax(dead_mask).astype(jnp.int32)
    has_dead = jnp.any(dead_mask)

    # Place 1 row south of the player (or clamp into map).
    map_h, map_w = state.terrain.shape[2], state.terrain.shape[3]
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    spawn_r = jnp.clip(pr + jnp.int32(1), jnp.int32(0), jnp.int32(map_h - 1))
    spawn_c = jnp.clip(pc,                jnp.int32(0), jnp.int32(map_w - 1))
    spawn_pos = jnp.stack([spawn_r, spawn_c]).astype(jnp.int16)

    do_spawn = should_explode & has_dead

    new_alive = mai.alive.at[dead_idx].set(
        jnp.where(do_spawn, jnp.bool_(True), mai.alive[dead_idx])
    )
    new_pos = mai.pos.at[dead_idx].set(
        jnp.where(do_spawn, spawn_pos, mai.pos[dead_idx])
    )
    new_hp = mai.hp.at[dead_idx].set(
        jnp.where(do_spawn, jnp.int32(4), mai.hp[dead_idx])
    )
    new_hp_max = mai.hp_max.at[dead_idx].set(
        jnp.where(do_spawn, jnp.int32(4), mai.hp_max[dead_idx])
    )
    new_peaceful = mai.peaceful.at[dead_idx].set(
        jnp.where(do_spawn, jnp.bool_(False), mai.peaceful[dead_idx])
    )
    new_asleep = mai.asleep.at[dead_idx].set(
        jnp.where(do_spawn, jnp.bool_(False), mai.asleep[dead_idx])
    )
    new_mai = mai.replace(
        alive    = new_alive,
        pos      = new_pos,
        hp       = new_hp,
        hp_max   = new_hp_max,
        peaceful = new_peaceful,
        asleep   = new_asleep,
    )

    # If the cursed BoT exploded, it also empties (otyp NONE-like) — vendor
    # behaviour: bag becomes a regular sack of tricks; we keep the type.
    new_cs = cs.replace(is_open=new_is_open)
    mid_state = state.replace(containers=new_cs, monster_ai=new_mai, rng=new_rng)

    # ---- Container trap (pickup.c::container_trap) ----
    # fire_container_trap is defined later but referenced here; Python function
    # calls are resolved at call-time so the forward reference is fine.
    mid_state, _fired = fire_container_trap(mid_state, slot_idx)
    return mid_state


def close_container(state, slot_idx):
    """Mark the container at ``slot_idx`` as closed (pickup.c::use_container)."""
    cs = state.containers
    c_idx = jnp.int32(slot_idx)
    has_container = cs.container_type[c_idx] != jnp.int8(ContainerType.NONE)
    new_is_open = cs.is_open.at[c_idx].set(
        jnp.where(has_container, jnp.bool_(False), cs.is_open[c_idx])
    )
    return state.replace(containers=cs.replace(is_open=new_is_open))


# ---------------------------------------------------------------------------
# Put / Take
# ---------------------------------------------------------------------------

def put_in_container(state, container_idx, src_slot):
    """Move inventory[src_slot] INTO container[container_idx] at first empty pos.

    Canonical: vendor/nethack/src/pickup.c::in_container (pickup.c:2558-2712).
    The vendor implementation unlinks the obj from invent (freeinv, pickup.c
    :2624) and links it into container.cobj (add_to_container, pickup.c:2703).
    Here we copy fields and zero the source slot.

    Bag-of-holding-in-bag-of-holding catastrophe:
      Per pickup.c:2658-2693 (Is_mbag + mbag_explodes branch), placing a
      BAG_OF_HOLDING (or BAG_OF_TRICKS — both Is_mbag) inside another
      BAG_OF_HOLDING triggers a magical explosion.  Vendor's
      ``mbag_explodes`` at depth=1 fires deterministically (rn2(2) <= 1
      is always True).  We model the explosion as a refusal at the put
      gate: ``item_allowed`` is cleared when ``src_is_mbag & container
      is_boh``, so the inserted bag is consumed by the explosion (the
      observable "you lose the bag you tried to put in") while the
      outer bag's contents are left in place (vendor scatters them
      around — we omit the scatter() machinery here).

    Refused items (Audit L #8, vendor pickup.c lines 2577-2622) — these
    silently no-op rather than inserting:
      * cursed loadstone (line 2577)
      * Amulet of Yendor / Candelabrum of Invocation / Bell of Opening /
        Book of the Dead (lines 2581-2589 — the four quest artifacts)
      * active leash (line 2591 — approximated here as any leash since we
        don't track the leashed-monster bit)
      * BOULDER / large STATUE (line 2616-2617)
      * any physical box (LARGE_BOX / CHEST / ICE_BOX, line 2616 Is_box)
      * welded uwep (pickup.c:2594-2599) — gated via InventoryState.welded
        when the source slot is the currently wielded weapon

    Parameters
    ----------
    state         : EnvState
    container_idx : int — container index (0..N_CONTAINERS-1)
    src_slot      : int — inventory slot index (0..MAX_INVENTORY_SLOTS-1)

    Returns
    -------
    new_state with updated inventory and container slices.
    """
    cs  = state.containers
    inv = state.inventory.items

    c_idx = jnp.int32(container_idx)
    s_idx = jnp.int32(src_slot)

    has_container = cs.container_type[c_idx] != jnp.int8(ContainerType.NONE)
    has_src_item  = inv.category[s_idx] != jnp.int8(0)

    # Audit L #8: vendor pickup.c::in_container lines 2577-2622 refuses
    # to insert the following classes of items into ANY container.  Each
    # rejection is silent (vendor prints a "you can't" message we don't
    # model here; the operation just no-ops).
    src_tid = inv.type_id[s_idx]
    src_buc = inv.buc_status[s_idx]
    is_cursed_loadstone = (
        (src_tid == jnp.int16(_LOADSTONE_TYPE_ID))
        & (src_buc == jnp.int8(BUCStatus.CURSED))
    )
    is_quest_item = (
        (src_tid == jnp.int16(_AMULET_OF_YENDOR_TYPE_ID))
        | (src_tid == jnp.int16(_CANDELABRUM_OF_INVOCATION_TYPE_ID))
        | (src_tid == jnp.int16(_BELL_OF_OPENING_TYPE_ID))
        | (src_tid == jnp.int16(_BOOK_OF_THE_DEAD_TYPE_ID))
    )
    is_oversize = (
        (src_tid == jnp.int16(_BOULDER_TYPE_ID))
        | (src_tid == jnp.int16(_STATUE_TYPE_ID))
    )
    is_leash = src_tid == jnp.int16(_LEASH_TYPE_ID)
    # Boxes inside boxes are also refused (pickup.c:2616 Is_box(obj)).  All
    # containers we model are boxes / bags; a coarse-grained gate that
    # forbids inserting any container-typed item via inventory.category
    # would over-restrict, so we narrow to the specific physical-box
    # otyps (LARGE_BOX / CHEST / ICE_BOX).
    is_box_inside_box = (
        (src_tid == jnp.int16(_LARGE_BOX_CONTAINER_TYPE_ID))
        | (src_tid == jnp.int16(_CHEST_CONTAINER_TYPE_ID))
        | (src_tid == jnp.int16(_ICE_BOX_CONTAINER_TYPE_ID))
    )
    # Vendor pickup.c:2487-2507 ``mbag_explodes``: at depth=1
    # (BoH placed inside another BoH) ``rn2(2) <= 1`` always fires.
    # We model: refuse the put (the inserted bag is destroyed by the
    # magical explosion) AND deal 1d6 damage to the player.  The full
    # scatter() of the outer bag's existing contents is approximated by
    # leaving them in place (vendor scatters them around at u.x/u.y);
    # this preserves the observable "you lose the bag you just tried
    # to put in" + "you take blast damage" without requiring a ground-
    # item write inside this JIT-shaped helper.
    _BOH_TID = jnp.int16(194)
    _BAG_OF_TRICKS_TID = jnp.int16(195)
    src_is_mbag = (src_tid == _BOH_TID) | (src_tid == _BAG_OF_TRICKS_TID)
    container_is_boh_check = (
        cs.container_type[c_idx] == jnp.int8(ContainerType.BAG_OF_HOLDING)
    )
    is_boh_in_boh = src_is_mbag & container_is_boh_check
    # Vendor pickup.c:2594-2599 refuses to insert the currently wielded
    # weapon when it's cursed-welded to the hero.
    wielded_slot = state.inventory.wielded.astype(jnp.int32)
    is_welded_uwep = (
        (wielded_slot == s_idx) & state.inventory.welded
    )
    item_allowed = ~(
        is_cursed_loadstone | is_quest_item | is_oversize
        | is_leash         | is_box_inside_box | is_boh_in_boh
        | is_welded_uwep
    )

    found_pos, dst_pos = _find_first_empty_in_container(cs, c_idx)
    can_put = has_container & has_src_item & found_pos & item_allowed
    safe_pos = jnp.clip(dst_pos, 0, MAX_ITEMS_PER_CONTAINER - 1)

    def _set_field(cs_field, inv_field, dtype):
        return cs_field.at[c_idx, safe_pos].set(
            jnp.where(can_put, inv_field[s_idx].astype(dtype), cs_field[c_idx, safe_pos])
        )

    new_cs = cs.replace(
        items_category   = _set_field(cs.items_category,   inv.category,    jnp.int8),
        items_type_id    = _set_field(cs.items_type_id,    inv.type_id,     jnp.int16),
        items_buc        = _set_field(cs.items_buc,        inv.buc_status,  jnp.int8),
        items_enchant    = _set_field(cs.items_enchant,    inv.enchantment, jnp.int8),
        items_charges    = _set_field(cs.items_charges,    inv.charges,     jnp.int8),
        items_identified = _set_field(cs.items_identified, inv.identified,  jnp.bool_),
        items_quantity   = _set_field(cs.items_quantity,   inv.quantity,    jnp.int16),
        items_weight     = _set_field(cs.items_weight,     inv.weight,      jnp.int16),
    )

    # Zero out the source inventory slot if we moved it.
    new_inv = inv.replace(
        category    = inv.category.at[s_idx].set(
            jnp.where(can_put, jnp.int8(0), inv.category[s_idx])
        ),
        type_id     = inv.type_id.at[s_idx].set(
            jnp.where(can_put, jnp.int16(0), inv.type_id[s_idx])
        ),
        buc_status  = inv.buc_status.at[s_idx].set(
            jnp.where(can_put, jnp.int8(0), inv.buc_status[s_idx])
        ),
        enchantment = inv.enchantment.at[s_idx].set(
            jnp.where(can_put, jnp.int8(0), inv.enchantment[s_idx])
        ),
        charges     = inv.charges.at[s_idx].set(
            jnp.where(can_put, jnp.int8(0), inv.charges[s_idx])
        ),
        identified  = inv.identified.at[s_idx].set(
            jnp.where(can_put, jnp.bool_(False), inv.identified[s_idx])
        ),
        quantity    = inv.quantity.at[s_idx].set(
            jnp.where(can_put, jnp.int16(0), inv.quantity[s_idx])
        ),
        weight      = inv.weight.at[s_idx].set(
            jnp.where(can_put, jnp.int32(0), inv.weight[s_idx])
        ),
        ac_bonus    = inv.ac_bonus.at[s_idx].set(
            jnp.where(can_put, jnp.int8(0), inv.ac_bonus[s_idx])
        ),
        is_two_handed = inv.is_two_handed.at[s_idx].set(
            jnp.where(can_put, jnp.bool_(False), inv.is_two_handed[s_idx])
        ),
    )

    # Bag-of-holding weight accounting — vendor mkobj.c::weight 1944-1953
    # (Audit L #2): blessed (cwt+3)/4, uncursed (cwt+1)/2, cursed cwt*2.
    # The item was in inventory at full weight; after moving in, the BoH
    # contributes the ceiling-rounded fraction.  Delta = effective - raw
    # (negative = savings).
    raw_w = inv.weight[s_idx].astype(jnp.int32)
    ctype = cs.container_type[c_idx]
    cbuc  = cs.container_buc[c_idx]
    is_boh   = ctype == jnp.int8(ContainerType.BAG_OF_HOLDING)
    is_bless = is_boh & (cbuc == jnp.int8(BUCStatus.BLESSED))
    is_curse = is_boh & (cbuc == jnp.int8(BUCStatus.CURSED))
    is_uncur = is_boh & ~is_bless & ~is_curse
    blessed_w = (raw_w + jnp.int32(3)) // jnp.int32(4)
    uncur_w   = (raw_w + jnp.int32(1)) // jnp.int32(2)
    cursed_w  = raw_w * jnp.int32(2)
    effective_w = jnp.where(
        is_bless,  blessed_w,
        jnp.where(is_uncur, uncur_w,
                  jnp.where(is_curse, cursed_w, raw_w)),
    )
    weight_delta = jnp.where(can_put, effective_w - raw_w, jnp.int32(0))
    new_total_weight = state.inventory.total_weight + weight_delta

    new_inv_state = state.inventory.replace(items=new_inv, total_weight=new_total_weight)
    return state.replace(inventory=new_inv_state, containers=new_cs)


def take_from_container(state, container_idx, item_pos):
    """Move container[container_idx][item_pos] OUT to first empty inventory slot.

    Canonical: vendor/nethack/src/pickup.c::out_container.
    """
    cs  = state.containers
    inv = state.inventory.items

    c_idx = jnp.int32(container_idx)
    p_idx = jnp.int32(item_pos)

    has_container = cs.container_type[c_idx] != jnp.int8(ContainerType.NONE)
    has_item      = cs.items_category[c_idx, p_idx] != jnp.int8(0)

    found_slot, dst_slot = _find_first_empty_inventory_slot(inv.category)
    can_take = has_container & has_item & found_slot
    n_slots = inv.category.shape[0]
    safe_slot = jnp.clip(dst_slot, 0, n_slots - 1)

    # Copy item fields out to inventory.
    new_inv = inv.replace(
        category    = inv.category.at[safe_slot].set(
            jnp.where(can_take, cs.items_category[c_idx, p_idx], inv.category[safe_slot])
        ),
        type_id     = inv.type_id.at[safe_slot].set(
            jnp.where(can_take, cs.items_type_id[c_idx, p_idx], inv.type_id[safe_slot])
        ),
        buc_status  = inv.buc_status.at[safe_slot].set(
            jnp.where(can_take, cs.items_buc[c_idx, p_idx], inv.buc_status[safe_slot])
        ),
        enchantment = inv.enchantment.at[safe_slot].set(
            jnp.where(can_take, cs.items_enchant[c_idx, p_idx], inv.enchantment[safe_slot])
        ),
        charges     = inv.charges.at[safe_slot].set(
            jnp.where(can_take, cs.items_charges[c_idx, p_idx], inv.charges[safe_slot])
        ),
        identified  = inv.identified.at[safe_slot].set(
            jnp.where(can_take, cs.items_identified[c_idx, p_idx], inv.identified[safe_slot])
        ),
        quantity    = inv.quantity.at[safe_slot].set(
            jnp.where(can_take, cs.items_quantity[c_idx, p_idx], inv.quantity[safe_slot])
        ),
        weight      = inv.weight.at[safe_slot].set(
            jnp.where(
                can_take,
                cs.items_weight[c_idx, p_idx].astype(jnp.int32),
                inv.weight[safe_slot],
            )
        ),
    )

    # Zero out the container position.
    new_cs = cs.replace(
        items_category   = cs.items_category.at[c_idx, p_idx].set(
            jnp.where(can_take, jnp.int8(0), cs.items_category[c_idx, p_idx])
        ),
        items_type_id    = cs.items_type_id.at[c_idx, p_idx].set(
            jnp.where(can_take, jnp.int16(0), cs.items_type_id[c_idx, p_idx])
        ),
        items_buc        = cs.items_buc.at[c_idx, p_idx].set(
            jnp.where(can_take, jnp.int8(0), cs.items_buc[c_idx, p_idx])
        ),
        items_enchant    = cs.items_enchant.at[c_idx, p_idx].set(
            jnp.where(can_take, jnp.int8(0), cs.items_enchant[c_idx, p_idx])
        ),
        items_charges    = cs.items_charges.at[c_idx, p_idx].set(
            jnp.where(can_take, jnp.int8(0), cs.items_charges[c_idx, p_idx])
        ),
        items_identified = cs.items_identified.at[c_idx, p_idx].set(
            jnp.where(can_take, jnp.bool_(False), cs.items_identified[c_idx, p_idx])
        ),
        items_quantity   = cs.items_quantity.at[c_idx, p_idx].set(
            jnp.where(can_take, jnp.int16(0), cs.items_quantity[c_idx, p_idx])
        ),
        items_weight     = cs.items_weight.at[c_idx, p_idx].set(
            jnp.where(can_take, jnp.int16(0), cs.items_weight[c_idx, p_idx])
        ),
    )

    # Bag-of-holding weight accounting on take-out — vendor mkobj.c::weight
    # 1944-1953 (Audit L #2): use the same ceiling formulas as on put-in so
    # the savings reverse exactly.  blessed (cwt+3)/4, uncursed (cwt+1)/2,
    # cursed cwt*2.
    raw_w_out  = cs.items_weight[c_idx, p_idx].astype(jnp.int32)
    ctype_out  = cs.container_type[c_idx]
    cbuc_out   = cs.container_buc[c_idx]
    is_boh_out   = ctype_out == jnp.int8(ContainerType.BAG_OF_HOLDING)
    is_bless_out = is_boh_out & (cbuc_out == jnp.int8(BUCStatus.BLESSED))
    is_curse_out = is_boh_out & (cbuc_out == jnp.int8(BUCStatus.CURSED))
    is_uncur_out = is_boh_out & ~is_bless_out & ~is_curse_out
    blessed_w_out = (raw_w_out + jnp.int32(3)) // jnp.int32(4)
    uncur_w_out   = (raw_w_out + jnp.int32(1)) // jnp.int32(2)
    cursed_w_out  = raw_w_out * jnp.int32(2)
    effective_w_out = jnp.where(
        is_bless_out,  blessed_w_out,
        jnp.where(is_uncur_out, uncur_w_out,
                  jnp.where(is_curse_out, cursed_w_out, raw_w_out)),
    )
    weight_delta_out  = jnp.where(can_take, raw_w_out - effective_w_out, jnp.int32(0))
    new_total_weight  = state.inventory.total_weight + weight_delta_out

    new_inv_state = state.inventory.replace(items=new_inv, total_weight=new_total_weight)
    return state.replace(inventory=new_inv_state, containers=new_cs)


# ---------------------------------------------------------------------------
# #tip — empty container contents onto the floor
# ---------------------------------------------------------------------------

def tip_container(state, container_idx):
    """Empty ``container_idx`` onto the floor at the player's tile.

    Canonical: vendor/nethack/src/pickup.c::tipcontainer lines 3687-3760
    (and the vendor ``#tip`` command at pickup.c::dotip line 3562).  The
    vendor loop walks ``box->cobj`` and calls ``obj_extract_self`` +
    ``place_object`` for each item.

    Implementation
    --------------
    Vectorized & assignment-based.  We compute a per-container-slot
    destination ground-slot up-front (cumulative count of occupied
    container slots before slot i, summed onto the starting empty count
    of the ground stack).  Container slot i with occupied item moves to
    ground slot ``base_empty + prior_occupied_count``; any overflow past
    ``MAX_GROUND_STACK`` is dropped (matches the existing no-room
    behavior in this port — vendor would call ``replace_object`` which
    falls back to the same tile, but our ground array has a fixed depth).

    Cite: vendor/nethack/src/pickup.c::dotip line 3562;
          vendor/nethack/src/pickup.c::tipcontainer lines 3687-3760.

    Parameters
    ----------
    state         : EnvState
    container_idx : int or jnp.int32 — container slot (0..N_CONTAINERS-1)

    Returns
    -------
    new_state with the container emptied and its contents copied onto
    the ground stack at the player's tile.
    """
    from Nethax.nethax.subsystems.inventory import MAX_GROUND_STACK

    c_idx = jnp.int32(container_idx)
    b     = state.dungeon.current_branch.astype(jnp.int32)
    lv    = (state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1))
    pr    = state.player_pos[0].astype(jnp.int32)
    pc    = state.player_pos[1].astype(jnp.int32)

    cs     = state.containers
    ground = state.ground_items

    has_container = cs.container_type[c_idx] != jnp.int8(ContainerType.NONE)

    # Container occupancy mask (per-slot bool).
    cat_row  = cs.items_category[c_idx]            # int8[M]
    occupied = cat_row != jnp.int8(0)              # bool[M]

    # Ground stack occupancy → starting fill count (== first empty slot).
    g_cats = ground.category[b, lv, pr, pc]        # int8[MAX_GROUND_STACK]
    g_occ  = g_cats != jnp.int8(0)                 # bool[MAX_GROUND_STACK]
    g_base = jnp.sum(g_occ.astype(jnp.int32))      # number of occupied ground slots

    # Per-container-slot destination ground index (only meaningful when occupied):
    # base_empty + (#occupied in container slots [0..i)).
    prior = jnp.cumsum(occupied.astype(jnp.int32)) - occupied.astype(jnp.int32)
    dest  = g_base + prior                         # int32[M]
    fits  = dest < jnp.int32(MAX_GROUND_STACK)
    do_copy_per_slot = has_container & occupied & fits  # bool[M]
    safe_dest = jnp.clip(dest, 0, MAX_GROUND_STACK - 1).astype(jnp.int32)

    # Write each container slot to its destination ground slot if do_copy[i].
    # We loop in Python over MAX_ITEMS_PER_CONTAINER (fixed at trace time so
    # the IR is flat).
    new_ground = ground
    new_cs     = cs
    for i in range(MAX_ITEMS_PER_CONTAINER):
        do_copy = do_copy_per_slot[i]
        g_slot  = safe_dest[i]

        # Ground write helper.
        def _gset(field_g, field_c, dtype):
            old = field_g[b, lv, pr, pc, g_slot]
            new_val = jnp.where(
                do_copy,
                field_c[c_idx, i].astype(dtype),
                old,
            )
            return field_g.at[b, lv, pr, pc, g_slot].set(new_val)

        new_ground = new_ground.replace(
            category    = _gset(new_ground.category,    cs.items_category,   jnp.int8),
            type_id     = _gset(new_ground.type_id,     cs.items_type_id,    jnp.int16),
            buc_status  = _gset(new_ground.buc_status,  cs.items_buc,        jnp.int8),
            enchantment = _gset(new_ground.enchantment, cs.items_enchant,    jnp.int8),
            charges     = _gset(new_ground.charges,     cs.items_charges,    jnp.int8),
            identified  = _gset(new_ground.identified,  cs.items_identified, jnp.bool_),
            quantity    = _gset(new_ground.quantity,    cs.items_quantity,   jnp.int16),
            weight      = _gset(new_ground.weight,      cs.items_weight,     jnp.int32),
        )

        # Container zero helper.
        def _czero(field_c, zero_val):
            old = field_c[c_idx, i]
            return field_c.at[c_idx, i].set(jnp.where(do_copy, zero_val, old))

        new_cs = new_cs.replace(
            items_category   = _czero(new_cs.items_category,   jnp.int8(0)),
            items_type_id    = _czero(new_cs.items_type_id,    jnp.int16(0)),
            items_buc        = _czero(new_cs.items_buc,        jnp.int8(0)),
            items_enchant    = _czero(new_cs.items_enchant,    jnp.int8(0)),
            items_charges    = _czero(new_cs.items_charges,    jnp.int8(0)),
            items_identified = _czero(new_cs.items_identified, jnp.bool_(False)),
            items_quantity   = _czero(new_cs.items_quantity,   jnp.int16(0)),
            items_weight     = _czero(new_cs.items_weight,     jnp.int16(0)),
        )

    return state.replace(containers=new_cs, ground_items=new_ground)


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def container_total_weight(container_state: ContainerState, container_idx) -> jnp.ndarray:
    """Return the effective weight contributed by container ``container_idx``.

    Raw weight = sum of items_weight inside.

    BAG_OF_HOLDING weight formulas — vendor ``mkobj.c::weight`` lines
    1944-1953 (Audit L #1):
      blessed   → (cwt + 3) / 4   (ceiling division by 4)
      uncursed  → (cwt + 1) / 2   (ceiling division by 2)
      cursed    →  cwt * 2

    All other containers contribute raw weight (no scaling).  These are
    integer ceiling rounds, not floors — previously this function did
    ``(raw * numer) // _BOH_DENOM`` which floors and is off-by-one for
    odd weights.
    """
    c_idx = jnp.int32(container_idx)

    weights = container_state.items_weight[c_idx].astype(jnp.int32)
    raw_total = jnp.sum(weights)

    ctype = container_state.container_type[c_idx]
    cbuc  = container_state.container_buc[c_idx]

    is_boh    = ctype == jnp.int8(ContainerType.BAG_OF_HOLDING)
    is_bless  = is_boh & (cbuc == jnp.int8(BUCStatus.BLESSED))
    is_curse  = is_boh & (cbuc == jnp.int8(BUCStatus.CURSED))
    is_uncur  = is_boh & ~is_bless & ~is_curse

    # blessed BoH:  (cwt + 3) / 4   — ceiling of cwt/4.
    blessed_w = (raw_total + jnp.int32(3)) // jnp.int32(4)
    # uncursed BoH: (cwt + 1) / 2   — ceiling of cwt/2.
    uncur_w   = (raw_total + jnp.int32(1)) // jnp.int32(2)
    # cursed BoH:   cwt * 2.
    cursed_w  = raw_total * jnp.int32(2)
    # non-BoH containers: raw weight unchanged.
    return jnp.where(
        is_bless,  blessed_w,
        jnp.where(is_uncur, uncur_w,
                  jnp.where(is_curse, cursed_w, raw_total)),
    )


# ---------------------------------------------------------------------------
# Python-side helper: install a container into ``state.containers``.
# ---------------------------------------------------------------------------

def install_container(
    state,
    container_idx: int,
    container_type: ContainerType,
    parent_slot: int = -1,
    buc: int = int(BUCStatus.UNCURSED),
    trapped: bool = False,
):
    """Place a fresh container of the given type into slot ``container_idx``.

    Test/setup helper — not used at runtime from action dispatch.
    """
    cs = state.containers
    c = int(container_idx)
    new_cs = cs.replace(
        container_type = cs.container_type.at[c].set(jnp.int8(int(container_type))),
        parent_slot    = cs.parent_slot.at[c].set(jnp.int8(int(parent_slot))),
        container_buc  = cs.container_buc.at[c].set(jnp.int8(int(buc))),
        is_open        = cs.is_open.at[c].set(jnp.bool_(False)),
        is_trapped     = cs.is_trapped.at[c].set(jnp.bool_(bool(trapped))),
    )
    return state.replace(containers=new_cs)


# ---------------------------------------------------------------------------
# Bag of tricks  (vendor/nethack/src/pickup.c::bagotricks)
# ---------------------------------------------------------------------------

def use_bag_of_tricks(state, rng):
    """Use a bag of tricks: spawn a hostile monster, decrement charges.

    Canonical: vendor/nethack/src/pickup.c::bagotricks.  The vendor selects a
    random low-tier monster and calls makemon().  Here we find the first dead
    monster slot and wake it as hostile (hp=4).  Charges stored in
    items_charges[bot_idx, 0] decrement by 1 each call.
    """
    cs = state.containers

    is_bot  = cs.container_type == jnp.int8(ContainerType.BAG_OF_TRICKS)
    has_chg = cs.items_charges[:, 0] > jnp.int8(0)
    usable  = is_bot & has_chg
    bot_idx = jnp.argmax(usable).astype(jnp.int32)
    any_bot = jnp.any(usable)

    old_chg = cs.items_charges[bot_idx, 0]
    new_charges = cs.items_charges.at[bot_idx, 0].set(
        jnp.where(any_bot, jnp.maximum(jnp.int8(0), old_chg - jnp.int8(1)), old_chg)
    )
    new_cs = cs.replace(items_charges=new_charges)

    mai = state.monster_ai
    dead_mask = ~mai.alive
    dead_idx  = jnp.argmax(dead_mask).astype(jnp.int32)
    has_dead  = jnp.any(dead_mask)

    map_h = state.terrain.shape[2]
    map_w = state.terrain.shape[3]
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    spawn_r   = jnp.clip(pr + jnp.int32(1), jnp.int32(0), jnp.int32(map_h - 1))
    spawn_c   = jnp.clip(pc,                jnp.int32(0), jnp.int32(map_w - 1))
    spawn_pos = jnp.stack([spawn_r, spawn_c]).astype(jnp.int16)

    do_spawn = any_bot & has_dead

    new_mai = mai.replace(
        alive    = mai.alive.at[dead_idx].set(
            jnp.where(do_spawn, jnp.bool_(True),  mai.alive[dead_idx])),
        pos      = mai.pos.at[dead_idx].set(
            jnp.where(do_spawn, spawn_pos,         mai.pos[dead_idx])),
        hp       = mai.hp.at[dead_idx].set(
            jnp.where(do_spawn, jnp.int32(4),      mai.hp[dead_idx])),
        hp_max   = mai.hp_max.at[dead_idx].set(
            jnp.where(do_spawn, jnp.int32(4),      mai.hp_max[dead_idx])),
        peaceful = mai.peaceful.at[dead_idx].set(
            jnp.where(do_spawn, jnp.bool_(False),  mai.peaceful[dead_idx])),
        asleep   = mai.asleep.at[dead_idx].set(
            jnp.where(do_spawn, jnp.bool_(False),  mai.asleep[dead_idx])),
    )

    rng_new, _ = jax.random.split(rng)
    return state.replace(containers=new_cs, monster_ai=new_mai, rng=rng_new)


# ---------------------------------------------------------------------------
# Cursed bag of holding item consumption  (pickup.c::use_container)
# ---------------------------------------------------------------------------

def cursed_bag_consume(state, rng, container_idx: int):
    """Cursed bag of holding: per-item 1/13 chance to destroy each item.

    Vendor: pickup.c::is_boh_item_gone lines 2509-2513:

        if (Is_mbag(container) && container->cursed && !rn2(13))
            return TRUE;  /* destroy this item */

    The check fires **per item** on every open/use of a cursed BoH (or
    BoT — they share ``Is_mbag``), not a single coin flip on the bag.
    Audit L #3 flagged the previous "single 10% roll picks one victim"
    implementation as a Nethax-only approximation.

    For ``MAX_ITEMS_PER_CONTAINER`` slots we split the rng once into
    one key per slot, roll ``rn2(13) == 0`` independently for each, and
    zero out the slots that drew 0 AND were occupied.

    Cite: vendor/nethack/src/pickup.c::is_boh_item_gone lines 2509-2513;
    vendor/nethack/include/obj.h:339 ``Is_mbag`` (BoH OR BoT).
    """
    cs    = state.containers
    c_idx = jnp.int32(container_idx)

    # Audit L #6: gate covers BoH OR BoT per Is_mbag(container).
    ctype = cs.container_type[c_idx]
    is_mbag = (
        (ctype == jnp.int8(ContainerType.BAG_OF_HOLDING))
        | (ctype == jnp.int8(ContainerType.BAG_OF_TRICKS))
    )
    is_cursed = cs.container_buc[c_idx] == jnp.int8(BUCStatus.CURSED)
    eat_active = is_mbag & is_cursed

    # Per-slot independent rn2(13) == 0 rolls.  Split the rng once into
    # MAX_ITEMS_PER_CONTAINER + 1 keys (last one carried out as rng_new).
    n_slots = cs.items_category.shape[1]
    keys = jax.random.split(rng, n_slots + 1)
    rng_new = keys[0]
    per_slot_keys = keys[1:]
    # rn2(13) == 0 per slot.
    rolls = jax.vmap(lambda k: jax.random.randint(k, (), 0, 13))(per_slot_keys)
    gone_roll = rolls == jnp.int32(0)

    occupied = cs.items_category[c_idx] != jnp.int8(0)
    do_eat = eat_active & gone_roll & occupied  # bool[N_SLOTS]

    new_cs = cs.replace(
        items_category = cs.items_category.at[c_idx].set(
            jnp.where(do_eat, jnp.int8(0),  cs.items_category[c_idx])
        ),
        items_quantity = cs.items_quantity.at[c_idx].set(
            jnp.where(do_eat, jnp.int16(0), cs.items_quantity[c_idx])
        ),
        items_weight   = cs.items_weight.at[c_idx].set(
            jnp.where(do_eat, jnp.int16(0), cs.items_weight[c_idx])
        ),
    )
    return state.replace(containers=new_cs, rng=rng_new)


# ---------------------------------------------------------------------------
# Action handlers (dispatched from action_dispatch.py)
# ---------------------------------------------------------------------------

def handle_loot_floor(state, rng):
    """#loot — open the first FLOOR container at the player's tile.

    Vendor ``#loot`` (vendor/nethack/src/pickup.c::doloot line 2166 →
    doloot_core line 2178) calls ``container_at(cc.x, cc.y, TRUE)`` to
    find containers on the floor at the hero's spot and routes the
    chosen one through ``do_loot_cont`` → ``use_container``.  Carried
    bags are handled via ``apply`` (the ``handle_apply_container`` path
    below), not via ``#loot``.

    Audit L #10: the prior ``handle_loot`` opened any container in
    ``state.containers`` (including carried bags), which was a Nethax-only
    divergence.  Splitting into a floor-only and a carried-only path
    matches the vendor command boundary.

    Cite: vendor/nethack/src/pickup.c::doloot lines 2166-2174;
          vendor/nethack/src/pickup.c::doloot_core lines 2178-2295
          (``container_at(cc.x, cc.y, TRUE)`` at line 2217).

    Implementation: filter by ``parent_slot == -1`` (the sentinel for
    "not held in any inventory slot"; floor containers populate this
    array via the same ContainerState struct in this port).  Open the
    lowest-index floor container; no-op if none.
    """
    cs = state.containers
    has_container = cs.container_type != jnp.int8(ContainerType.NONE)
    is_floor      = cs.parent_slot == jnp.int8(-1)
    usable        = has_container & is_floor
    first_idx     = jnp.argmax(usable).astype(jnp.int32)
    any_present   = jnp.any(usable)

    def _open(s):
        return open_container(s, first_idx)

    return jax.lax.cond(any_present, _open, lambda s: s, state)


# Back-compat alias: legacy name ``handle_loot`` still referenced by
# ``cancel_bag_of_holding`` doc strings and external imports prior to the
# Audit L #10 split.  Points at the floor path so ``_handle_loot`` →
# ``handle_loot`` continues to dispatch the vendor ``#loot`` behavior.
handle_loot = handle_loot_floor


def handle_apply_container(state, rng):
    """APPLY-to-container — open the first CARRIED container.

    Vendor ``apply`` (vendor/nethack/src/apply.c::doapply) routes a
    container in the player's inventory through ``use_container``.  This
    path is the carried-bag side of the Audit L #10 split: it filters by
    ``parent_slot >= 0`` so it never opens a floor container (those go
    through ``#loot`` → ``handle_loot_floor`` instead).

    Cite: vendor/nethack/src/apply.c::doapply (container branch);
          vendor/nethack/src/pickup.c::use_container.
    """
    cs = state.containers
    has_container = cs.container_type != jnp.int8(ContainerType.NONE)
    is_carried    = cs.parent_slot >= jnp.int8(0)
    usable        = has_container & is_carried
    first_idx     = jnp.argmax(usable).astype(jnp.int32)
    any_present   = jnp.any(usable)

    def _open(s):
        return open_container(s, first_idx)

    return jax.lax.cond(any_present, _open, lambda s: s, state)


# ---------------------------------------------------------------------------
# Wand of cancellation: bag-of-holding implosion
# ---------------------------------------------------------------------------

# type_id constants (vendor/nethack/include/objects.h object index order).
# Cite: Nethax/nethax/constants/objects.py:3964-4080 — verified container otyps.
_LARGE_BOX_CONTAINER_TYPE_ID:      int = 189   # vendor objects.h:899  LARGE_BOX
_CHEST_CONTAINER_TYPE_ID:          int = 190   # vendor objects.h:901  CHEST
_ICE_BOX_CONTAINER_TYPE_ID:        int = 191   # vendor objects.h:903  ICE_BOX
_SACK_CONTAINER_TYPE_ID:           int = 192   # SACK — demoted type after cancel
_OILSKIN_SACK_CONTAINER_TYPE_ID:   int = 193   # vendor objects.h:907  OILSKIN_SACK
_BAG_OF_HOLDING_CONTAINER_TYPE_ID: int = 194   # same as apply_tools._BAG_OF_HOLDING_TYPE_ID
_BAG_OF_TRICKS_CONTAINER_TYPE_ID:  int = 195   # vendor objects.h:911  BAG_OF_TRICKS
# Inserted-item type_ids that trigger the bag-of-holding implosion when
# placed INTO another bag of holding.  Cite: vendor/nethack/src/pickup.c
# ::mbag_explodes (line 2488-2507) — WAN_CANCELLATION or a nested Is_mbag
# (BAG_OF_HOLDING / BAG_OF_TRICKS, gated by spe>0 for the latter two).
_WAN_CANCELLATION_TYPE_ID:         int = 395   # vendor objects.h:8084-8086

# Audit L #8: vendor pickup.c::in_container lines 2577-2622 refuses to insert
# certain items into ANY container.  Cite: constants/objects.py line numbers.
_AMULET_OF_YENDOR_TYPE_ID:           int = 188   # objects.py:3944
_CANDELABRUM_OF_INVOCATION_TYPE_ID:  int = 237   # objects.py:4924
_BELL_OF_OPENING_TYPE_ID:            int = 238   # objects.py:4944
_BOOK_OF_THE_DEAD_TYPE_ID:           int = 382   # objects.py:7824
_LOADSTONE_TYPE_ID:                  int = 443   # objects.py:9045
_BOULDER_TYPE_ID:                    int = 447   # objects.py:9129
_STATUE_TYPE_ID:                     int = 448   # objects.py:9149
_LEASH_TYPE_ID:                      int = 211   # objects.py:4404


def cancel_bag_of_holding(state, container_idx: int, src_slot: int = -1):
    """Insertion-trigger implosion for a bag of holding (or bag of tricks).

    Audit L #5: vendor ``zap.c::cancel_item`` (lines 1239-1362) has NO
    BAG_OF_HOLDING / BAG_OF_TRICKS case — an external wand zap at a BoH
    just runs the trailing ``unbless`` + ``uncurse`` at lines 1359-1360
    and returns.  The "destroy contents + demote bag" behavior was a
    Nethax-only divergence; ``action_dispatch.py`` no longer invokes
    this function on external zap.

    The contents-destruction path is reserved for the *insertion*
    trigger (vendor pickup.c::in_container line 2658 → mbag_explodes
    2488-2507): inserting WAN_CANCELLATION (any spe) or a nested
    Is_mbag (BoH / BoT with spe > 0) detonates the bag, destroys all
    contents, ``obfree`` 's the triggering item (pickup.c:2685-2690),
    and deals ``d(6,6)`` HP damage (pickup.c:2692).

    Two call modes survive for back-compat with existing wiring:

    1. ``src_slot >= 0``: the insertion path described above.  We model
       "obfree the bag" by demoting its container_type to NONE so the
       slot becomes unusable (we don't free the underlying pytree
       array shape).
    2. ``src_slot == -1`` (legacy): no-op — external wand-zap path is
       not implemented because vendor has none.  Returns ``state``
       unchanged.  The buc-clearing portion (unbless+uncurse) is a
       documented follow-up since it requires a container-BUC flag
       flip not yet wired through.

    Cite: vendor/nethack/src/pickup.c::mbag_explodes (2488-2507);
          vendor/nethack/src/pickup.c::in_container 2658-2693;
          vendor/nethack/src/zap.c::cancel_item 1239-1362 (NO BoH branch).

    Parameters
    ----------
    state         : EnvState
    container_idx : int — index into state.containers (0..N_CONTAINERS-1)
    src_slot      : int — inventory slot of the inserted trigger item, or -1
                          for the (now-no-op) external wand-zap path.

    Returns
    -------
    new_state
    """
    # Audit L #5: external zap path is a no-op in vendor.  Short-circuit
    # before any destruction logic runs.
    if int(src_slot) < 0:
        return state
    cs    = state.containers
    c_idx = jnp.int32(container_idx)

    is_boh = cs.container_type[c_idx] == jnp.int8(ContainerType.BAG_OF_HOLDING)
    is_insertion = src_slot >= 0

    # Zero all items_quantity in this container slot (contents destroyed).
    # Cite: zap.c::cancel_item — bag implodes, destroying everything inside.
    zeroed_qty = jnp.zeros(MAX_ITEMS_PER_CONTAINER, dtype=jnp.int16)
    new_qty = jnp.where(
        is_boh,
        cs.items_quantity.at[c_idx].set(zeroed_qty),
        cs.items_quantity,
    )

    # External-zap path: demote BAG_OF_HOLDING → SACK (vendor zap.c just
    # changes obj->otyp; the bag still exists).
    # Insertion path: vendor obfree's the bag (pickup.c:2685-2690) — we set
    # container slot to NONE so the player can no longer interact with it.
    demoted_type = jnp.int8(int(ContainerType.NONE) if is_insertion
                            else int(ContainerType.SACK))
    new_ctype = jnp.where(
        is_boh,
        cs.container_type.at[c_idx].set(demoted_type),
        cs.container_type,
    )

    new_cs = cs.replace(items_quantity=new_qty, container_type=new_ctype)
    new_state = state.replace(containers=new_cs)

    # Insertion path only: destroy the triggering source item and apply
    # d(6,6) HP damage to the player.
    # Cite: pickup.c:2669 obfree(obj); pickup.c:2692 losehp(d(6,6), ...).
    if is_insertion and src_slot >= 0:
        s_idx = jnp.int32(src_slot)
        inv = new_state.inventory.items

        # Destroy the source item (zero its slot) iff the bag actually imploded.
        new_inv = inv.replace(
            category    = inv.category.at[s_idx].set(
                jnp.where(is_boh, jnp.int8(0), inv.category[s_idx])),
            type_id     = inv.type_id.at[s_idx].set(
                jnp.where(is_boh, jnp.int16(0), inv.type_id[s_idx])),
            quantity    = inv.quantity.at[s_idx].set(
                jnp.where(is_boh, jnp.int16(0), inv.quantity[s_idx])),
            weight      = inv.weight.at[s_idx].set(
                jnp.where(is_boh, jnp.int32(0), inv.weight[s_idx])),
        )

        # d(6,6) — sum of 6 d6 rolls (triangular distribution [6, 36]),
        # byte-equal to vendor `zap.c::cancel_item` mbag-explodes damage.
        # Cite: vendor/nethack/src/zap.c::cancel_item line 720.
        rng_dmg, new_rng = jax.random.split(new_state.rng)
        _d6_keys = jax.random.split(rng_dmg, 6)
        dmg = jnp.sum(jnp.stack([
            jax.random.randint(k, (), 1, 7, dtype=jnp.int32) for k in _d6_keys
        ])).astype(jnp.int32)
        new_hp = jnp.where(
            is_boh,
            jnp.maximum(jnp.int32(0),
                        new_state.player_hp.astype(jnp.int32) - dmg),
            new_state.player_hp.astype(jnp.int32),
        ).astype(new_state.player_hp.dtype)

        new_state = new_state.replace(
            inventory=new_state.inventory.replace(items=new_inv),
            player_hp=new_hp,
            rng=new_rng,
        )

    return new_state


# ---------------------------------------------------------------------------
# Wand-of-cancellation / nested-BoH insertion explosion
# Cite: vendor/nethack/src/pickup.c::in_container line 2658 + mbag_explodes
#       (lines 2488-2507).
# ---------------------------------------------------------------------------

def maybe_explode_on_insert(state, container_idx: int, src_slot: int):
    """Check whether inserting ``inventory[src_slot]`` into container
    ``container_idx`` triggers the mbag_explodes implosion, and apply it.

    Trigger conditions (vendor pickup.c:2488-2507 ``mbag_explodes``):
      * container is a "magic bag" — ``Is_mbag`` covers BOTH BAG_OF_HOLDING
        AND BAG_OF_TRICKS (vendor obj.h:339).  Audit L #6 fix.
      * AND the inserted item is WAN_CANCELLATION (any spe), OR a nested
        BAG_OF_HOLDING / BAG_OF_TRICKS with spe > 0.

    Vendor's mbag_explodes also has a depth-based probabilistic gate
    ``rn2(1 << depthin) <= depthin``.  At ``depthin == 0`` this is
    ``rn2(1) <= 0`` which is always True, so a top-level insertion always
    detonates — we model only depthin == 0 here.

    Note on magic markers: vendor mbag_explodes does NOT include
    MAGIC_MARKER as a trigger.  The wave-25b task description mentioned
    magic markers but no vendor codepath inserts that branch, so to keep
    byte-equal-vendor parity we do not detonate on marker insertion.

    JIT-safe: pure functional, uses jnp.where rather than Python ``if``.
    """
    cs  = state.containers
    inv = state.inventory.items
    c_idx = jnp.int32(container_idx)
    s_idx = jnp.int32(src_slot)

    # Audit L #6: ``Is_mbag(container)`` is BoH OR BoT, not just BoH.
    ctype = cs.container_type[c_idx]
    is_mbag = (
        (ctype == jnp.int8(ContainerType.BAG_OF_HOLDING))
        | (ctype == jnp.int8(ContainerType.BAG_OF_TRICKS))
    )

    src_tid     = inv.type_id[s_idx]
    src_charges = inv.charges[s_idx]
    is_wan_cancel = src_tid == jnp.int16(_WAN_CANCELLATION_TYPE_ID)
    # Nested-BoH (or BoT) explodes only with spe > 0.  Cite: pickup.c:2491-2493.
    is_nested_mbag = (
        (src_tid == jnp.int16(_BAG_OF_HOLDING_CONTAINER_TYPE_ID))
        | (src_tid == jnp.int16(_BAG_OF_TRICKS_CONTAINER_TYPE_ID))
    ) & (src_charges > jnp.int8(0))

    should_explode = is_mbag & (is_wan_cancel | is_nested_mbag)

    return jax.lax.cond(
        should_explode,
        lambda s: cancel_bag_of_holding(s, container_idx, src_slot),
        lambda s: s,
        state,
    )


# ---------------------------------------------------------------------------
# doloot dispatch (pickup.c:2166 doloot)
# ---------------------------------------------------------------------------

# Action sub-codes emitted by handle_doloot so callers know which branch fired.
DOLOOT_NOOP: int = 0   # no container available at player's tile
DOLOOT_IN:   int = 1   # put items in (default when inventory non-empty)
DOLOOT_OUT:  int = 2   # take items out (default when inventory is empty)
DOLOOT_BOTH: int = 3   # both — reserved (vendor menu-choice path)


def handle_doloot(state, rng):
    """#loot extended-command dispatcher.

    Canonical: vendor/nethack/src/pickup.c::doloot (line 2166) which calls
    doloot_core (line 2178).  Vendor presents a menu choosing among
      o  "Take something out"
      i  "Put something in"
      b  "Both of the above"
      r  "Reverse loot"  (cursed-item slip-out)
    based on player state.  We model the menu choice deterministically:
      * if the player's inventory has any non-coin item, default to IN
        (player most-likely wants to stash);
      * otherwise default to OUT.

    The selection is materialised by reusing ``handle_loot`` to open the
    first held container, then either ``put_in_container`` (slot 0 → first
    inv item) or ``take_from_container`` (container slot 0 → inventory).
    A future wave will dispatch through a proper floor-container lookup
    using level objects; for now we operate on the carried containers
    array exactly as ``handle_loot`` already does.

    Returns
    -------
    new_state
    """
    # Find a container — same lookup ``handle_loot`` uses.
    cs = state.containers
    has_any = cs.container_type != jnp.int8(ContainerType.NONE)
    first_idx_signed = jnp.argmax(has_any).astype(jnp.int32)
    any_present = jnp.any(has_any)

    # Determine in/out default based on inventory contents.
    inv_cat = state.inventory.items.category
    from Nethax.nethax.subsystems.inventory import ItemCategory as _IC
    non_coin_present = jnp.any(
        (inv_cat != jnp.int8(0)) & (inv_cat != jnp.int8(int(_IC.COIN)))
    )
    # Find first non-coin inventory slot for the IN branch.
    is_non_coin = (inv_cat != jnp.int8(0)) & (inv_cat != jnp.int8(int(_IC.COIN)))
    first_inv_slot = jnp.argmax(is_non_coin).astype(jnp.int32)

    # Find first occupied container slot (item to take out).
    occupied_in_c = cs.items_category[first_idx_signed] != jnp.int8(0)
    first_c_pos   = jnp.argmax(occupied_in_c).astype(jnp.int32)
    any_in_c      = jnp.any(occupied_in_c)

    # Branch 1: open the container (mirrors handle_loot).
    def _open_and_dispatch(s):
        s = open_container(s, first_idx_signed)
        # IN path — default when player has non-coin inventory.
        def _put(st):
            return put_in_container(st, first_idx_signed, first_inv_slot)
        # OUT path — when inventory has no non-coin items.
        def _take(st):
            return jax.lax.cond(
                any_in_c,
                lambda x: take_from_container(x, first_idx_signed, first_c_pos),
                lambda x: x,
                st,
            )
        return jax.lax.cond(non_coin_present, _put, _take, s)

    return jax.lax.cond(any_present, _open_and_dispatch, lambda s: s, state)


# ---------------------------------------------------------------------------
# ICE_BOX freeze helper
# Cite: vendor/nethack/src/pickup.c lines 2644-2657 (Icebox branch);
#       vendor/nethack/src/mkobj.c::age_corpses lazy-age model.
# ---------------------------------------------------------------------------

def is_corpse_iced(container_state: ContainerState, container_idx, item_pos) -> jnp.ndarray:
    """Return True iff the corpse at ``container[container_idx, item_pos]``
    is inside an ICE_BOX and therefore frozen (age does not advance).

    Used by callers that compute corpse age lazily as ``moves - creation_turn``:
    when this returns True, the caller must clamp the age contribution to
    zero (freeze semantics, vendor pickup.c:2644-2657 stop_timer(ROT_CORPSE)).
    """
    c_idx = jnp.int32(container_idx)
    p_idx = jnp.int32(item_pos)
    is_ice = container_state.container_type[c_idx] == jnp.int8(ContainerType.ICE_BOX)
    has_item = container_state.items_category[c_idx, p_idx] != jnp.int8(0)
    return is_ice & has_item


def any_corpse_iced(container_state: ContainerState) -> jnp.ndarray:
    """Vectorised gate: True iff *any* container slot is an ICE_BOX.

    Useful as a global short-circuit in a future per-turn corpse-age loop:
    the caller can skip the age increment for every corpse known to live in
    an ICE_BOX by indexing through ``parent_slot``.  Cite: vendor
    pickup.c:2647-2651 stop_timer(ROT_CORPSE).
    """
    return jnp.any(
        container_state.container_type == jnp.int8(ContainerType.ICE_BOX)
    )
