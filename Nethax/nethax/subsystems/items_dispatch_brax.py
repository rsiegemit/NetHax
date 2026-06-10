"""Brax-style flattened rewrites of the heavy items dispatch entry points.

Why this file exists
--------------------
``Nethax.nethax.subsystems.items_potions.quaff_potion`` and
``Nethax.nethax.subsystems.items_scrolls.read_scroll`` are the two largest
single contributors to HLO size in ``action_dispatch`` (they sit under
``_handle_quaff`` / ``_handle_zap``/``_handle_read``).  Each runs a
``jax.lax.switch`` over 20-30 per-effect handler functions.  Under
``jax.vmap`` over seeds, ``lax.switch`` inlines **every** branch into the
compiled HLO graph anyway -- so paying for an additional masked compute
costs no extra HLO traces, while removing the switch dispatch overhead.

``apply_potion_to_monster`` is the throw-on-monster variant which suffers
the same blow-up.

Following the same Brax / Craftax pattern already used by
``combat_helpers_brax.py``, ``mattackm_brax.py``,
``monster_attack_player_brax.py`` and ``handlers_brax.py``:

  * **Always compute every branch on the same operand.**
  * **Select the chosen branch's result with ``jnp.where``** via
    ``jax.tree.map`` (so we get pytree-where for free).

For ``lax.cond`` we likewise compute both branches and select with
``jax.tree.map(jnp.where, ...)`` over the resulting state pytrees.

We keep ``lax.scan`` calls with fixed iteration counts (e.g. the
``_mark_up_to_n`` scan inside ``_effect_identify``) untouched -- those do
not blow up the HLO under vmap because the scan body is traced once.

Byte-parity contract
--------------------
1. **RNG draw order preserved exactly.**  Each branch is invoked with the
   identical PRNGKey the original would have routed to it, so internal
   ``jax.random.split`` chains run with the same input key and produce the
   same draws.  The original ``lax.switch`` only runs *one* branch but in
   Brax-style we run *all* branches on the same key; selecting one branch
   via mask preserves which draw sequence is observable in the output.
2. **Mutations byte-identical via ``jnp.where``** over pytrees with the
   same scalar mask as the original ``lax.switch`` selector / ``lax.cond``
   predicate.
3. **State pytree shape preserved** -- every branch returns an EnvState
   pytree of the same shape, so ``jax.tree.map`` selection is structural.

Conditionals flattened (per public entry point)
-----------------------------------------------
- ``quaff_potion_brax``            : 1 ``lax.cond`` + 1 ``lax.switch`` (26
                                     branches) -> 1 tree-where + 1
                                     26-way Python-loop tree-where.
- ``apply_potion_to_monster_brax`` : 1 ``lax.switch`` (26 branches) ->
                                     1 26-way Python-loop tree-where.
- ``read_scroll_brax``             : 1 ``lax.cond`` + 2 ``lax.switch``
                                     (23 branches each, confused + sane)
                                     -> 1 tree-where + 2 23-way
                                     Python-loop tree-wheres.

Signatures match the originals so the Brax versions are drop-in.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Reuse the existing per-effect handler tables and constants from the
# canonical implementations.  We are NOT modifying those modules; we only
# *read* their already-defined effect tables and helper functions.
from Nethax.nethax.subsystems import items_potions as _potions
from Nethax.nethax.subsystems import items_scrolls as _scrolls

from Nethax.nethax.subsystems.status_effects import TimedStatus
from Nethax.nethax.constants.objects import ObjectClass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tree_where(pred, on_true, on_false):
    """``jax.lax.cond`` analogue: select between two pytrees by scalar mask.

    ``pred`` is a JAX scalar boolean (traced).  ``on_true`` and ``on_false``
    must be pytrees with identical structure and per-leaf shape/dtype.
    """
    return jax.tree.map(
        lambda a, b: jnp.where(pred, a, b),
        on_true,
        on_false,
    )


def _select_branch_states(selector, branch_states):
    """Select one EnvState pytree from a list by integer ``selector``.

    ``selector``      : JAX scalar int32 -- the branch index to pick.
    ``branch_states`` : list of EnvState pytrees, one per branch.  Every
                        entry must share identical pytree structure /
                        per-leaf shape / per-leaf dtype.

    Implementation: fold right-to-left over the list using ``_tree_where``
    with the per-index equality mask.  The compiled HLO is a chain of
    ``select`` ops -- the same shape ``lax.switch`` lowers to, minus the
    dispatch overhead.
    """
    # Walk the list once; carry the running "selected" pytree.
    # Start from the last branch as the default; for each earlier index i,
    # overwrite if selector == i.
    result = branch_states[-1]
    n = len(branch_states)
    for i in range(n - 2, -1, -1):
        pred = selector == jnp.int32(i)
        result = _tree_where(pred, branch_states[i], result)
    return result


# ---------------------------------------------------------------------------
# Potions -- player-targeted dispatch
# ---------------------------------------------------------------------------

def _quaff_drop_glib_brax(state, slot_idx):
    """Glib-fingers drop branch -- byte-identical to the original
    ``_drop_glib`` inside ``items_potions.quaff_potion``.

    Decrements the potion's quantity and clears category if exhausted;
    does NOT apply the effect.
    """
    _sidx = jnp.int32(slot_idx)
    items = state.inventory.items
    old_qty = items.quantity[_sidx]
    new_qty = jnp.maximum(old_qty - jnp.int16(1), jnp.int16(0))
    new_cat = jnp.where(
        new_qty == jnp.int16(0),
        jnp.int8(0),
        items.category[_sidx],
    )
    new_quantity = items.quantity.at[_sidx].set(new_qty)
    new_category = items.category.at[_sidx].set(new_cat)
    new_items = items.replace(quantity=new_quantity, category=new_category)
    return state.replace(inventory=state.inventory.replace(items=new_items))


def _quaff_do_effect_brax(state, rng, slot_idx):
    """Active-quaff path -- effect dispatch via per-branch compute + mask.

    Replaces the inner ``jax.lax.switch(effect_id, _SWITCH_BRANCHES, ...)``
    with a Python loop over all ``_potions._EFFECT_TABLE`` entries.  Each
    entry runs on the same ``(state, rng, buc)`` operand the original
    branch would have received.

    Post-dispatch mutations (use-identification + quantity decrement +
    "You quaff the potion." message) are byte-identical to the original.
    """
    _sidx = jnp.int32(slot_idx)
    items = state.inventory.items
    type_id = items.type_id[_sidx].astype(jnp.int32)
    buc = items.buc_status[_sidx]

    effect_id = jnp.clip(
        type_id - jnp.int32(_potions._POTION_BASE_ID),
        0,
        _potions.N_POTIONS - 1,
    )

    # Run every effect handler on the same input; select by effect_id.
    # Each handler takes (state, rng, buc) -> state.
    branch_results = [
        fn(state, rng, buc) for fn in _potions._EFFECT_TABLE
    ]
    new_state = _select_branch_states(effect_id, branch_results)

    # Use-identification: per-item flag + per-type oc_name_known mask.
    new_items_id = new_state.inventory.items.identified.at[_sidx].set(
        jnp.bool_(True)
    )
    type_mask = new_state.identification.identified
    type_id_clipped = jnp.clip(
        type_id, jnp.int32(0), jnp.int32(type_mask.shape[0] - 1)
    )
    new_type_mask = type_mask.at[type_id_clipped].set(jnp.bool_(True))
    new_state = new_state.replace(
        inventory=new_state.inventory.replace(
            items=new_state.inventory.items.replace(identified=new_items_id),
        ),
        identification=new_state.identification.replace(identified=new_type_mask),
    )

    # Decrement quantity; clear category when exhausted.
    old_qty = new_state.inventory.items.quantity[_sidx]
    new_qty = jnp.maximum(old_qty - jnp.int16(1), jnp.int16(0))
    new_cat = jnp.where(
        new_qty == jnp.int16(0),
        jnp.int8(0),
        new_state.inventory.items.category[_sidx],
    )
    new_quantity = new_state.inventory.items.quantity.at[_sidx].set(new_qty)
    new_category = new_state.inventory.items.category.at[_sidx].set(new_cat)
    new_items = new_state.inventory.items.replace(
        quantity=new_quantity, category=new_category
    )
    new_inv = new_state.inventory.replace(items=new_items)

    # "You quaff the potion." message.
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    return new_state.replace(
        inventory=new_inv,
        messages=_msg_emit(new_state.messages, int(_MsgId.YOU_QUAFF_POTION)),
    )


def quaff_potion_brax(state, rng, slot_idx):
    """Brax-style flattened ``quaff_potion``.

    Original (``items_potions.quaff_potion``):
        - 1 ``lax.cond`` over ``glib_drop`` (drop vs do-quaff).
        - 1 ``lax.switch`` over 26 potion effects inside the do-quaff branch.

    Flattened: both ``lax.cond`` branches and all 26 ``lax.switch`` branches
    are always computed; selection is by ``jnp.where`` over the
    corresponding mask.

    Byte-parity: the RNG ``rng`` is the same key the original cond/switch
    would have routed to whichever branch fired.  All branches consume RNG
    from the same input key, so the selected branch's RNG draw sequence is
    identical to the original.
    """
    # Glib roll -- vendor/nethack/src/status.c::glibs() -- runs unconditionally
    # in the original (split happens before the cond), so we mirror that.
    is_glib = state.status.timed_statuses[int(TimedStatus.GLIB)] > jnp.int32(0)
    rng, rng_glib = jax.random.split(rng)
    glib_roll = jax.random.randint(rng_glib, (), 0, 5, dtype=jnp.int32)
    glib_drop = is_glib & (glib_roll == jnp.int32(0))

    # Always compute both branches; select by glib_drop.
    drop_state = _quaff_drop_glib_brax(state, slot_idx)
    quaff_state = _quaff_do_effect_brax(state, rng, slot_idx)

    return _tree_where(glib_drop, drop_state, quaff_state)


# ---------------------------------------------------------------------------
# Potions -- monster-targeted dispatch (thrown / shattered)
# ---------------------------------------------------------------------------

def apply_potion_to_monster_brax(state, rng, type_id, m_slot):
    """Brax-style flattened ``apply_potion_to_monster``.

    Original (``items_potions.apply_potion_to_monster``):
        - 1 ``lax.switch`` over 26 monster-targeted potion effects.

    Flattened: all 26 branches run on the same ``(state, m_slot, rng)``
    operand; selection by ``jnp.where`` over the ``effect_id`` mask.
    """
    effect_id = jnp.clip(
        type_id.astype(jnp.int32) - jnp.int32(_potions._POTION_BASE_ID),
        0,
        _potions.N_POTIONS - 1,
    )
    m_slot_i = m_slot.astype(jnp.int32)

    branch_results = [
        fn(state, m_slot_i, rng) for fn in _potions._MONSTER_EFFECT_TABLE
    ]
    return _select_branch_states(effect_id, branch_results)


# ---------------------------------------------------------------------------
# Scrolls -- public read dispatch
# ---------------------------------------------------------------------------

# Cache the boolean "has confused-branch handler" mask from the canonical
# scrolls module.  Mirrors items_scrolls._HAS_CONFUSED but accessed via the
# already-built table to avoid duplicating its construction.
_HAS_CONFUSED_BRAX = _scrolls._HAS_CONFUSED


def _read_call_effect(fn, state, rng, buc, slot_idx):
    """Invoke a scroll effect handler with the right number of args.

    Mirrors ``items_scrolls._make_branch``: handlers with 4 parameters take
    ``slot_idx`` (e.g. ``_effect_identify``); others take only 3.
    """
    import inspect
    sig = inspect.signature(fn)
    if len(sig.parameters) >= 4:
        return fn(state, rng, buc, slot_idx)
    return fn(state, rng, buc)


def _read_call_confused(fn_or_none, state, rng, slot_idx):
    """Invoke a confused-branch handler or pass-through state.

    Mirrors the lambdas built into ``items_scrolls._CONFUSED_BRANCHES``:
    if the effect has no confused handler, return state unchanged.
    """
    if fn_or_none is None:
        return state
    return fn_or_none(state, rng, slot_idx)


def _read_do_brax(state, rng, slot_idx):
    """Active scroll-read path -- both sane and confused dispatches flattened.

    Replaces the two inner ``lax.switch`` calls (confused + sane) with
    Python loops over all ``_scrolls._EFFECT_TABLE`` entries.  Each pair
    of branch results is then selected with ``effect_id``; the chosen
    confused state vs sane state is then selected with ``use_confused``.
    """
    _slot_idx = jnp.int32(slot_idx)
    items = state.inventory.items
    type_id = items.type_id[_slot_idx].astype(jnp.int32)
    buc = items.buc_status[_slot_idx]

    effect_id = jnp.clip(
        type_id - jnp.int32(_scrolls._SCROLL_BASE_ID),
        0,
        _scrolls.N_SCROLLS - 1,
    )

    confused = state.status.timed_statuses[int(TimedStatus.CONFUSION)] > jnp.int32(0)
    has_confused = _HAS_CONFUSED_BRAX[effect_id]
    use_confused = confused & has_confused

    # Emit "You read the scroll." BEFORE dispatching the effect, matching
    # vendor read.c:387 doread() pline() order.
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    state = state.replace(
        messages=_msg_emit(state.messages, int(_MsgId.YOU_READ_SCROLL)),
    )

    # --- Sane branches: run every effect handler with (state, rng, buc[, slot_idx]).
    sane_results = [
        _read_call_effect(fn, state, rng, buc, _slot_idx)
        for fn in _scrolls._EFFECT_TABLE
    ]
    sane_state = _select_branch_states(effect_id, sane_results)

    # --- Confused branches: run every confused handler (or pass-through).
    n = _scrolls.N_SCROLLS
    confused_handlers = [
        _scrolls._CONFUSED_HANDLER_MAP.get(i) for i in range(n)
    ]
    confused_results = [
        _read_call_confused(h, state, rng, _slot_idx)
        for h in confused_handlers
    ]
    confused_state = _select_branch_states(effect_id, confused_results)

    # Pick confused vs sane via mask -- mirrors the original jax.tree.map.
    new_state = _tree_where(use_confused, confused_state, sane_state)

    # Use-identification: per-item flag + per-type oc_name_known mask.
    new_items_id = new_state.inventory.items.identified.at[_slot_idx].set(
        jnp.bool_(True)
    )
    type_mask = new_state.identification.identified
    safe_otyp = jnp.clip(
        type_id, jnp.int32(0), jnp.int32(type_mask.shape[0] - 1)
    )
    new_type_mask = type_mask.at[safe_otyp].set(jnp.bool_(True))
    new_state = new_state.replace(
        inventory=new_state.inventory.replace(
            items=new_state.inventory.items.replace(identified=new_items_id),
        ),
        identification=new_state.identification.replace(identified=new_type_mask),
    )

    # Decrement quantity; clear category when exhausted.
    old_qty = new_state.inventory.items.quantity[_slot_idx]
    new_qty = jnp.maximum(old_qty - jnp.int16(1), jnp.int16(0))
    new_cat = jnp.where(
        new_qty == jnp.int16(0),
        jnp.int8(0),
        new_state.inventory.items.category[_slot_idx],
    )
    new_quantity = new_state.inventory.items.quantity.at[_slot_idx].set(new_qty)
    new_category = new_state.inventory.items.category.at[_slot_idx].set(new_cat)
    new_items = new_state.inventory.items.replace(
        quantity=new_quantity, category=new_category
    )
    new_inv = new_state.inventory.replace(items=new_items)

    return new_state.replace(inventory=new_inv)


def read_scroll_brax(state, rng, slot_idx):
    """Brax-style flattened ``read_scroll``.

    Original (``items_scrolls.read_scroll``):
        - 1 ``lax.cond`` over ``can_read`` (blind/stunned -> no-op).
        - 1 ``lax.switch`` over 23 confused-branch handlers.
        - 1 ``lax.switch`` over 23 sane-branch effect handlers.

    Flattened: every branch is always computed; selection is via masks.

    Byte-parity: the original sane/confused dispatch already ran *both*
    switches (see vendor-style ``jax.tree.map(jnp.where, confused_state,
    sane_state)`` in items_scrolls.read_scroll), so flattening the outer
    ``can_read`` cond is the only semantically new compute path -- and
    the blind/stunned no-op branch is just ``state`` itself.
    """
    is_blind = state.status.timed_statuses[int(TimedStatus.BLIND)] > jnp.int32(0)
    is_stunned = state.status.timed_statuses[int(TimedStatus.STUNNED)] > jnp.int32(0)
    can_read = ~(is_blind | is_stunned)

    # Always compute the "do read" branch; if blind/stunned, fall back to
    # unmodified ``state``.
    read_state = _read_do_brax(state, rng, slot_idx)
    return _tree_where(can_read, read_state, state)


# ---------------------------------------------------------------------------
# Notes on entry points already Brax-clean
# ---------------------------------------------------------------------------
#
# * ``items_potions.handle_quaff`` -- still uses an outer ``jax.lax.cond``
#   over ``found`` (any potion present?).  The cond's branches are
#   ``quaff_potion(...)`` and ``_quaff_no_potion(...)``.  Flattening this
#   would require unconditionally running ``quaff_potion`` with a sentinel
#   slot_idx, which is unsafe (slot may be -1 / out-of-bounds, item is
#   not a potion, BUC undefined, etc.).  Recommend a wrapper that calls
#   ``quaff_potion_brax`` instead of ``quaff_potion`` once we are ready
#   to flip the call site -- the outer cond can stay until then because
#   it is a *single* cond and not a 26-way switch, so HLO impact is minor.
#
# * ``items_scrolls.handle_read`` -- same shape as ``handle_quaff``.  Same
#   recommendation.
#
# * Per-effect handlers (e.g. ``_effect_identify``, ``_effect_charging``,
#   ``_effect_water``) still contain their own internal ``lax.cond`` /
#   ``lax.scan`` calls.  The scans are fixed-iteration so they trace once
#   and do **not** blow up under vmap; we leave them as-is per the task
#   constraint ("Keep lax.scan with fixed iteration counts").  Internal
#   ``lax.cond`` flattening inside individual handlers is a separate
#   follow-up if HLO size remains a concern after the outer dispatchers
#   are switched over.
