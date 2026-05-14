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
    (BAG_OF_HOLDING, LARGE_BOX, CHEST, BAG_OF_TRICKS, OILSKIN_SACK), but here
    we use a small dense enum so the int8 field stays compact.
    """
    NONE           = 0   # empty container slot
    SACK           = 1   # plain sack (no multiplier)
    OILSKIN_SACK   = 2   # waterproof
    BAG_OF_HOLDING = 3   # weight multiplier per BUC
    BAG_OF_TRICKS  = 4   # cursed → explode-on-open
    LARGE_BOX      = 5   # heavy floor-only box
    CHEST          = 6   # heavy floor-only chest


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
    return state.replace(containers=new_cs, monster_ai=new_mai, rng=new_rng)


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
      BAG_OF_HOLDING inside another BAG_OF_HOLDING triggers a magical
      explosion (do_boh_explosion); the source uses ``mbag_explodes`` to
      probabilistically trigger.  Wave 6 simplification: the put still
      succeeds but ``parent_slot`` won't be updated for the nested bag
      (deferred to a full explosion handler in a future wave).

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

    found_pos, dst_pos = _find_first_empty_in_container(cs, c_idx)
    can_put = has_container & has_src_item & found_pos
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

    new_inv_state = state.inventory.replace(items=new_inv)
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

    new_inv_state = state.inventory.replace(items=new_inv)
    return state.replace(inventory=new_inv_state, containers=new_cs)


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def container_total_weight(container_state: ContainerState, container_idx) -> jnp.ndarray:
    """Return the effective weight contributed by container ``container_idx``.

    Raw weight = sum of items_weight inside.

    BAG_OF_HOLDING weight multipliers (pickup.c::in_container):
      blessed   → 1/4
      uncursed  → 1/2
      cursed    → 2/1
    All other containers contribute raw weight (multiplier = 1).
    """
    c_idx = jnp.int32(container_idx)

    weights = container_state.items_weight[c_idx].astype(jnp.int32)
    raw_total = jnp.sum(weights)

    ctype = container_state.container_type[c_idx]
    cbuc  = container_state.container_buc[c_idx]

    is_boh = ctype == jnp.int8(ContainerType.BAG_OF_HOLDING)

    numer = jnp.where(
        is_boh & (cbuc == jnp.int8(BUCStatus.BLESSED)),
        jnp.int32(_BOH_NUMER_BLESSED),
        jnp.where(
            is_boh & (cbuc == jnp.int8(BUCStatus.CURSED)),
            jnp.int32(_BOH_NUMER_CURSED),
            jnp.where(
                is_boh,
                jnp.int32(_BOH_NUMER_UNCURSED),
                jnp.int32(_BOH_DENOM),  # multiplier == 1 for non-BoH
            ),
        ),
    )
    return (raw_total * numer) // jnp.int32(_BOH_DENOM)


# ---------------------------------------------------------------------------
# Python-side helper: install a container into ``state.containers``.
# ---------------------------------------------------------------------------

def install_container(
    state,
    container_idx: int,
    container_type: ContainerType,
    parent_slot: int = -1,
    buc: int = int(BUCStatus.UNCURSED),
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
    )
    return state.replace(containers=new_cs)


# ---------------------------------------------------------------------------
# Action handlers (dispatched from action_dispatch.py)
# ---------------------------------------------------------------------------

def handle_loot(state, rng):
    """LOOT — open the first held container (vendor/nethack/src/pickup.c::doloot).

    Wave 5 stand-in: pick the lowest-index slot that has a container and call
    ``open_container``.  If none, no-op.
    """
    cs = state.containers
    has_any = cs.container_type != jnp.int8(ContainerType.NONE)
    first_idx = jnp.argmax(has_any).astype(jnp.int32)
    any_present = jnp.any(has_any)

    def _open(s):
        return open_container(s, first_idx)

    return jax.lax.cond(any_present, _open, lambda s: s, state)


def handle_apply_container(state, rng):
    """APPLY-to-container — if the player APPLYs while holding a container,
    open it (vendor/nethack/src/apply.c::doapply routes containers to
    use_container).  This is the JIT-safe handler hooked from action dispatch.
    """
    return handle_loot(state, rng)
