"""Multi-turn occupation / ``afternmv`` state machine.

Vendor ``ga.afternmv`` is a function pointer that fires when ``gm.multi``
turns of occupation finish (e.g. taking off slow-to-remove armor while a
nymph is stealing).  See vendor/nethack/src/steal.c::stealarm,
unstolenarm, do_steal_armor — and similar uses in do_wear.c, dig.c,
read.c.

The vendor C model doesn't translate cleanly to JIT-pure JAX (function
pointers), so we replace it with an enum dispatch:

  OccupationKind:
    NONE         = 0  — player not occupied
    STEAL_ARM    = 1  — multi-turn armor doffing while a thief is
                        adjacent / dead (vendor stealarm + unstolenarm)

Public API
----------
``start_occupation(state, kind, target, turns)`` — set the player's
    current occupation; future per-turn actions are blocked by the
    caller (``_occupied_blocks_action`` helper).

``tick_occupation(state)`` — decrement ``occupation_remaining``; when it
    hits zero, fire the associated callback once and clear the slot.

``is_occupied(state)`` — bool gate for action_dispatch.

State plumbing
--------------
EnvState gains three int8 fields (see state.py):
    occupation_kind:      int8  — OccupationKind enum
    occupation_target:    int32 — monster slot / item index (kind-specific)
    occupation_remaining: int8  — turns left before callback fires

Cite: vendor/nethack/include/decl.h ga.afternmv;
      vendor/nethack/src/steal.c::stealarm lines 165-207;
      vendor/nethack/src/steal.c::unstolenarm lines 147-161;
      vendor/nethack/src/cmd.c (multi-turn occupation tick).
"""
from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp


class OccupationKind(IntEnum):
    NONE      = 0
    STEAL_ARM = 1  # nymph multi-turn armor undress (steal.c:165)


def start_occupation(
    state,
    kind: OccupationKind,
    target: jnp.ndarray,
    turns: jnp.ndarray,
):
    """Set the player's current occupation.

    Overrides any previously-set occupation (vendor's afternmv assignment
    is destructive too — the new occupation replaces the previous one,
    which is fine because vendor only sets afternmv at the start of an
    occupation that the player isn't otherwise busy with).
    """
    return state.replace(
        occupation_kind=jnp.int8(int(kind)),
        occupation_target=target.astype(jnp.int32),
        occupation_remaining=turns.astype(jnp.int8),
    )


def is_occupied(state) -> jnp.ndarray:
    """Return True iff player has an active occupation (action blocked).

    JIT-pure bool scalar.
    """
    return state.occupation_kind > jnp.int8(0)


def _callback_steal_arm(state):
    """Fire when STEAL_ARM occupation completes.

    Vendor stealarm: hero finishes doffing the targeted body armor; if
    the thief monster is still alive, the armor transfers to its
    inventory.  Otherwise (unstolenarm path) the armor simply ends up
    not-worn and stays in hero's inventory.

    JAX implementation:
      - clear ``inventory.worn_armor[BODY]`` (item is now not-worn)
      - if target monster still alive, copy the item to its inv slot 0
        (mirrors steal.c add_to_minv)
    """
    from Nethax.nethax.subsystems.inventory import ArmorSlot

    body_slot = jnp.int32(int(ArmorSlot.BODY))
    inv = state.inventory
    worn_idx = inv.worn_armor[body_slot]
    has_worn = worn_idx >= jnp.int8(0)
    safe_inv_idx = jnp.clip(worn_idx.astype(jnp.int32), 0, inv.items.category.shape[0] - 1)

    # Clear the worn-armor mapping (item is no longer worn).
    new_worn_armor = inv.worn_armor.at[body_slot].set(jnp.int8(-1))
    new_inv = inv.replace(worn_armor=new_worn_armor)
    state = state.replace(inventory=new_inv)

    # Transfer to target monster if alive.
    target = state.occupation_target.astype(jnp.int32)
    n_slots = state.monster_ai.alive.shape[0]
    safe_target = jnp.clip(target, 0, n_slots - 1)
    target_alive = state.monster_ai.alive[safe_target] & has_worn & (target >= jnp.int32(0))

    item_cat = inv.items.category[safe_inv_idx]
    item_tid = inv.items.type_id[safe_inv_idx]
    item_qty = inv.items.quantity[safe_inv_idx]
    mai = state.monster_ai
    new_mcat = jnp.where(target_alive, item_cat, mai.inv_category[safe_target, 0])
    new_mtid = jnp.where(target_alive, item_tid, mai.inv_type_id[safe_target, 0])
    new_mqty = jnp.where(target_alive, item_qty, mai.inv_quantity[safe_target, 0])
    new_mai = mai.replace(
        inv_category=mai.inv_category.at[safe_target, 0].set(new_mcat),
        inv_type_id=mai.inv_type_id.at[safe_target, 0].set(new_mtid),
        inv_quantity=mai.inv_quantity.at[safe_target, 0].set(new_mqty),
    )

    # Also remove from player's inventory (vendor freeinv).
    new_items = state.inventory.items.replace(
        category=state.inventory.items.category.at[safe_inv_idx].set(
            jnp.where(target_alive, jnp.int8(0), state.inventory.items.category[safe_inv_idx])
        ),
        quantity=state.inventory.items.quantity.at[safe_inv_idx].set(
            jnp.where(target_alive, jnp.int16(0), state.inventory.items.quantity[safe_inv_idx])
        ),
    )
    return state.replace(
        inventory=state.inventory.replace(items=new_items),
        monster_ai=new_mai,
    )


def _callback_noop(state):
    return state


_OCCUPATION_CALLBACKS = (
    _callback_noop,       # 0 NONE
    _callback_steal_arm,  # 1 STEAL_ARM
)


def tick_occupation(state):
    """Per-turn occupation tick.

    Decrements ``occupation_remaining``.  When it transitions from 1 to
    0 on this turn, the associated callback fires once.  After firing,
    ``occupation_kind`` resets to NONE.

    Cite: vendor/nethack/src/cmd.c — afternmv invocation when gm.multi
          reaches 0.
    """
    rem = state.occupation_remaining.astype(jnp.int32)
    kind = state.occupation_kind.astype(jnp.int32)
    active = kind > jnp.int32(0)
    new_rem = jnp.where(active, jnp.maximum(rem - jnp.int32(1), jnp.int32(0)), rem)
    firing = active & (rem == jnp.int32(1))

    def _fire(s):
        return jax.lax.switch(s.occupation_kind.astype(jnp.int32), _OCCUPATION_CALLBACKS, s)

    state_post = jax.lax.cond(firing, _fire, lambda s: s, state)
    new_kind = jnp.where(firing, jnp.int8(0), state_post.occupation_kind)
    return state_post.replace(
        occupation_kind=new_kind,
        occupation_remaining=new_rem.astype(jnp.int8),
    )
