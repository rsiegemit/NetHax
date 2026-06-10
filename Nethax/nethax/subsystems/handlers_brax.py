"""Brax-style flat rewrites of the dominant non-movement action handlers.

Background
----------
Under ``dispatch_action_brax``'s 46-way flat fan-out every handler computes on
every step (the dispatcher then masks with ``jnp.where`` at the state-tree
level).  This means each handler's HLO footprint contributes additively to the
compiled program — the Brax pattern only wins if individual handlers are
themselves free of ``lax.cond`` / ``lax.switch`` branching, otherwise the
compiler still emits all branches for each one.

This file collects the most common non-movement handlers and exposes them with
a ``_brax`` suffix for the flat dispatcher to import uniformly.

Audit results
-------------
All five requested handlers are *already Brax-clean at the wrapper level*:

* ``_wait`` (action_dispatch.py:1802) — single ``state.replace`` emitting the
  ``YOU_WAIT`` message.  No control flow.

* ``_handle_quaff`` (action_dispatch.py:1958) — one-line direct delegate to
  ``potions._handle_quaff``.  No wrapper-level branching.

* ``_handle_zap`` (action_dispatch.py:1968) — ~220 lines but every conditional
  is ``jnp.where`` (direction decoding, wand-slot fallback, BoH cancellation,
  use-identification, enlightenment message).  Zero ``lax.cond`` / ``lax.switch``
  in the wrapper body.  The internal ``jax.tree.map(jnp.where, ...)`` over
  ``messages`` is the canonical Brax masked-mutation pattern.

* ``_handle_pickup`` (action_dispatch.py:2239) — projects ``(branch, level)``,
  delegates to ``inventory.pickup``, writes back.  No wrapper-level control
  flow.

* ``_handle_search`` (action_dispatch.py:2466) — RNG split + direct delegate to
  ``features.handle_search``.  No wrapper-level control flow.

Conds flattened per function
----------------------------
* ``_wait``          : 0 (already flat)
* ``_handle_quaff``  : 0 (already flat — pure delegate)
* ``_handle_zap``    : 0 (already uses ``jnp.where`` exclusively)
* ``_handle_pickup`` : 0 (already flat — pure delegate)
* ``_handle_search`` : 0 (already flat — pure delegate)

Where the real HLO cost lives
-----------------------------
The wrapper bodies are thin.  The compile-size dominator is inside the
subsystem delegates that the wrappers call:

* ``potions._handle_quaff``       → potion-effect ``lax.switch`` over ~30 types
* ``items_wands.handle_zap``      → wand-effect ``lax.switch`` over ~25 types
* ``features.handle_search``      → 3x3 sweep with per-tile branching
* ``inventory.pickup``            → per-slot ``lax.scan`` plus stacking logic

Flattening those is the next-tier intervention and is intentionally out of
scope for this file — it requires touching subsystem modules, not the
dispatch-layer wrappers.

Byte-parity
-----------
Because each re-export is the *same callable* as the original (no rewrite
performed), byte parity is preserved by construction:
    1. RNG draw order is identical (same function object).
    2. State mutations are identical.
    3. State pytree shape is identical.

Usage
-----
Import the ``*_brax`` aliases from this module instead of the originals when
wiring the flat ``dispatch_action_brax`` fan-out, so a future rewrite of any
single handler can be made by editing only this file:

    from Nethax.nethax.subsystems.handlers_brax import (
        _wait_brax,
        _handle_quaff_brax,
        _handle_zap_brax,
        _handle_pickup_brax,
        _handle_search_brax,
    )
"""
from Nethax.nethax.subsystems.action_dispatch import (
    _wait as _wait_orig,
    _handle_quaff as _handle_quaff_orig,
    _handle_zap as _handle_zap_orig,
    _handle_pickup as _handle_pickup_orig,
    _handle_search as _handle_search_orig,
)

# ---------------------------------------------------------------------------
# Re-exports — wrappers are already Brax-clean (no lax.cond / lax.switch).
# Rebinding only (no rewrite) keeps byte parity by construction.
# ---------------------------------------------------------------------------
_wait_brax = _wait_orig
_handle_quaff_brax = _handle_quaff_orig
_handle_zap_brax = _handle_zap_orig
_handle_pickup_brax = _handle_pickup_orig
_handle_search_brax = _handle_search_orig


__all__ = [
    "_wait_brax",
    "_handle_quaff_brax",
    "_handle_zap_brax",
    "_handle_pickup_brax",
    "_handle_search_brax",
]
