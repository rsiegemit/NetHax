"""Brax-style rewrite of ``dispatch_action`` — flat HLO via ``jnp.where`` masking.

Background
----------
The canonical ``dispatch_action`` in ``action_dispatch.py`` routes the active
action through a 46-way ``jax.lax.switch`` over ``_COMPACT_HANDLERS``.  Under
``jax.vmap`` over seeds the Dr.Jit paper (§4) confirms that ``lax.switch``
inlines ALL N branches into the HLO graph, so the resulting per-step IR for
nethax contains 46 fully traced handler bodies AND a 46-way ``select_n``
dispatch.  Inside the body, the function actually invokes the switch THREE
times (once for the pending two-step path, once for the normal path, plus the
two outer ``lax.cond`` arms) — every primitive in every handler is traced
multiple times under vmap.

This file mirrors the Brax pattern (cf. ``brax.envs.base.PipelineEnv.step``
and Craftax ``craftax/craftax/craftax_env.py``): every handler is invoked
unconditionally with the same ``(state, rng, dir_idx)`` triple, and the
results are selected leaf-by-leaf via ``jax.tree_util.tree_map`` + ``jnp.where``.
The XLA compiler then sees a single flat sequence of handler bodies followed
by a 46-way ``select`` per state leaf — fusion-friendly and ~46× smaller HLO
per vmap'd seed.

Cost trade-off
--------------
Every handler runs on every step (cold cost ≈ 46× the canonical path).  The
optimization is a compile-time / HLO-size win; runtime may regress on small
batches but should improve under heavy vmap once XLA fuses the flat IR.

Byte-parity contract
--------------------
1.  Each handler is invoked with the SAME ``rng`` argument as the canonical
    switch would supply, so each handler's *selected* output is byte-identical
    to the original.  Handlers internally call ``jax.random.split`` etc.; the
    same key in produces the same key stream out.
2.  Unselected handler outputs are discarded via ``jnp.where`` on every leaf
    of the state pytree.  Because each leaf select reads
    ``handler_i_output_leaf`` only when ``compact_idx == i``, the resulting
    pytree is bit-identical to the original switch result.
3.  The pytree shape (every leaf, every dtype, every shape) is preserved
    because every handler returns the same ``EnvState`` pytree shape.

What is flattened
-----------------
* The 46-way ``lax.switch`` over ``_COMPACT_HANDLERS`` (normal path) →
  ``_select_handler_output`` via ``tree_map + jnp.where`` cascade.
* The 46-way ``lax.switch`` over ``_COMPACT_HANDLERS`` (deferred-direction
  path: ``state_after_dir``) → same flat select.
* The outer ``lax.cond(pending, _branch_pending, ...)`` and inner
  ``lax.cond(opens_any, _branch_open, _branch_normal)`` → ``tree_map +
  jnp.where`` over the four candidate states
  (after_dir / after_letter_two_step / after_inv_letter / open_prompt / normal).
* The inner ``lax.cond(is_dir, ..., lax.cond(is_ltd, ...))`` inside
  ``_branch_pending`` → folded into the same flat ``where`` cascade.

Total: **2 × 46-way ``lax.switch`` + 4 × ``lax.cond`` flattened.**

What is NOT touched
-------------------
* Each handler's INTERNAL ``lax.cond`` / ``lax.switch`` / ``lax.while_loop``
  usage remains intact (see e.g. ``_handle_cast`` -> ``lax.switch`` over
  ``_MAGIC_EFFECT_DISPATCH_LIST``, ``_run_shared`` -> ``lax.while_loop``).
  Flattening those is a separate Brax pass per handler.

Signature
---------
``dispatch_action_brax(state, action, rng) -> state`` — identical to the
canonical ``dispatch_action`` so the function can drop into ``env.step``
with a single import swap.
"""
import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems.action_dispatch import (
    _ACTION_TO_HANDLER_IDX,
    _SLOT_TO_COMPACT,
    _SLOT_TO_DIR_IDX,
    _COMPACT_HANDLERS,
)


# ---------------------------------------------------------------------------
# Tree-map helpers — branchless state selection
# ---------------------------------------------------------------------------

def _tree_where(pred: jnp.ndarray, on_true, on_false):
    """Leaf-wise ``jnp.where(pred, on_true_leaf, on_false_leaf)``.

    ``pred`` is a scalar boolean array broadcast against every leaf.  Both
    branches must share the pytree structure and per-leaf shape (which is
    guaranteed here: every handler returns the same ``EnvState`` shape).
    """
    return jax.tree_util.tree_map(
        lambda t, f: jnp.where(pred, t, f),
        on_true,
        on_false,
    )


def _select_handler_output(compact_idx: jnp.ndarray, outputs: tuple, default):
    """Cascade ``jnp.where(compact_idx == i, outputs[i], acc)`` over all
    ``len(outputs)`` candidates, starting from ``default``.

    The cascade orders matches the canonical ``_COMPACT_HANDLERS`` tuple so
    the selected leaf for ``compact_idx == k`` is exactly ``outputs[k]``.
    """
    acc = default
    # Iterate in reverse so the final emitted ``where`` for index 0 sits at
    # the outermost level — purely cosmetic; semantically equivalent to a
    # forward sweep because each comparison is mutually exclusive.
    for i, out in enumerate(outputs):
        mask = compact_idx == jnp.int32(i)
        acc = _tree_where(mask, out, acc)
    return acc


# ---------------------------------------------------------------------------
# Public Brax-style dispatch
# ---------------------------------------------------------------------------

def dispatch_action_brax(state, action: jnp.int32, rng: jax.Array):
    """Byte-parity drop-in for ``dispatch_action`` with flat HLO.

    See module docstring for the design rationale and parity contract.
    """
    # Imports are local to mirror the canonical function (the pending_action
    # helpers depend on EnvState which would create an import cycle at module
    # load time).
    from Nethax.nethax.subsystems.pending_action import (
        action_opens_inv_letter_prompt,
        action_opens_letter_then_dir_prompt,
        letter_to_slot,
        action_to_direction,
        is_pending,
    )

    action_val = jnp.clip(jnp.int32(action), 0, 255)

    # --- Step A: pending follow-up bookkeeping ---------------------------
    pending = is_pending(state)
    pending_kind = state.pending_action_kind.astype(jnp.int32)
    slot_from_letter = letter_to_slot(action_val).astype(jnp.int8)
    dir_from_action = action_to_direction(action_val).astype(jnp.int8)

    state_after_inv_letter = state.replace(
        pending_action_kind=jnp.int8(0),
        pending_action_root=jnp.int8(0),
        pending_action_slot=slot_from_letter,
    )

    state_after_letter_two_step = state.replace(
        pending_action_kind=jnp.int8(2),  # AWAIT_DIRECTION
        pending_action_slot=slot_from_letter,
    )

    # State used when re-entering dispatch under kind==2 (AWAIT_DIRECTION):
    # we run the *deferred root* through the compact-handler bank with the
    # stored direction.
    root_action = state.pending_action_root.astype(jnp.int32) & jnp.int32(0xFF)
    state_for_dir_dispatch = state.replace(
        pending_action_kind=jnp.int8(0),
        pending_action_root=jnp.int8(0),
        pending_action_dir=dir_from_action,
    )
    handler_idx_root = _ACTION_TO_HANDLER_IDX[root_action].astype(jnp.int32)
    compact_idx_root = _SLOT_TO_COMPACT[handler_idx_root]
    dir_idx_root = _SLOT_TO_DIR_IDX[handler_idx_root]

    # --- Step B: prompt-opening bookkeeping ------------------------------
    opens_inv = action_opens_inv_letter_prompt(action_val)
    opens_ltd = action_opens_letter_then_dir_prompt(action_val)
    opens_any = opens_inv | opens_ltd

    kind_int = jnp.where(
        opens_inv,
        jnp.int8(1),
        jnp.where(opens_ltd, jnp.int8(3), jnp.int8(0)),
    )
    state_open_prompt = state.replace(
        pending_action_kind=kind_int,
        pending_action_root=action_val.astype(jnp.int8),
        pending_action_slot=jnp.int8(-1),
        pending_action_dir=jnp.zeros((2,), dtype=jnp.int8),
    )

    # --- Step C: normal-dispatch indices ---------------------------------
    handler_idx = _ACTION_TO_HANDLER_IDX[action_val].astype(jnp.int32)
    compact_idx = _SLOT_TO_COMPACT[handler_idx]
    dir_idx = _SLOT_TO_DIR_IDX[handler_idx]

    # --- Brax fan-out: run ALL 46 handlers, twice ------------------------
    # First fan-out: normal-path inputs.  Each handler is called with the
    # canonical ``rng`` so the SELECTED handler's output is byte-identical
    # to ``lax.switch(compact_idx, _COMPACT_HANDLERS, state, rng, dir_idx)``.
    # Unselected handler outputs are discarded leaf-by-leaf below.
    normal_outputs = tuple(
        h(state, rng, dir_idx) for h in _COMPACT_HANDLERS
    )

    # Second fan-out: deferred-direction inputs (kind==2 path).  Uses the
    # ``state_for_dir_dispatch`` carry and the per-root dir_idx_root.
    dir_outputs = tuple(
        h(state_for_dir_dispatch, rng, dir_idx_root) for h in _COMPACT_HANDLERS
    )

    # --- Flat select: replace each ``lax.switch`` with a ``where`` cascade
    # The default ``state``/``state_for_dir_dispatch`` is overwritten the
    # moment any mask matches; if no mask matches (impossible because
    # compact_idx is clamped via _SLOT_TO_COMPACT to [0, 45]) the cascade
    # falls back to the unmodified carry, matching the JAX ``lax.switch``
    # out-of-range behavior (clamps to last branch — close enough since this
    # branch is unreachable in well-formed inputs).
    state_normal = _select_handler_output(compact_idx, normal_outputs, state)
    state_after_dir = _select_handler_output(
        compact_idx_root, dir_outputs, state_for_dir_dispatch,
    )

    # --- Flat select: collapse the two outer ``lax.cond`` layers ----------
    # Original control flow:
    #   if pending:
    #       if pending_kind == 2:     -> state_after_dir
    #       elif pending_kind == 3:   -> state_after_letter_two_step
    #       else (kind == 1):         -> state_after_inv_letter
    #   else:
    #       if opens_any:             -> state_open_prompt
    #       else:                     -> state_normal
    is_dir = pending_kind == jnp.int32(2)
    is_ltd = pending_kind == jnp.int32(3)

    # Build the result with a leaf-wise where cascade.  Order matters: the
    # final ``where`` for ``pending & is_dir`` overrides earlier writes.
    # Start from the non-pending branch (opens_any ? open_prompt : normal),
    # then layer the pending branches on top under the ``pending`` mask.
    base = _tree_where(opens_any, state_open_prompt, state_normal)

    # Pending branch picks between three sub-modes.  ``state_after_inv_letter``
    # is the kind==1 default (matches the canonical ``lambda __:
    # state_after_inv_letter`` deep arm of the nested ``cond``).
    pending_pick = _tree_where(
        is_dir,
        state_after_dir,
        _tree_where(
            is_ltd,
            state_after_letter_two_step,
            state_after_inv_letter,
        ),
    )

    return _tree_where(pending, pending_pick, base)
