"""NLE-action-index ↔ Nethax-ASCII-ord bidirectional lookup.

Two action conventions live side-by-side in this repo:

1. **NLE convention.**  ``env.step(action)`` takes an *integer index* into
   ``env.actions`` (the 86-entry tuple ``USEFUL_ACTIONS`` returned by the
   default NLE constructor).  An NLE-trained policy emits action ``0`` to
   mean *CompassDirection.N* — not the ASCII character ``\\x00``.

2. **Nethax convention.**  ``NethaxEnv.step(state, action, rng)`` routes
   ``action`` through ``dispatch_action``'s 256-entry ASCII LUT
   (``_ACTION_TO_HANDLER_IDX[action]``).  Nethax expects the *ord*
   (e.g. ``ord('k') = 107`` for north), matching vendor cmd.c's char
   dispatch in ``rhack``.

Without bridging the two, an NLE policy emitting ``action=0`` becomes a
no-op in Nethax (slot 0 of the ASCII LUT is undefined/_SLOT_NOOP).  This
module provides the bridge.

The mapping is built once at module load as a **static** ``jnp.array``
so callers can index into it inside ``jax.lax.cond`` / ``vmap`` without
re-tracing.  See ``Nethax.nethax.env.NethaxEnv.step`` for the wired-in
caller (auto-detects index vs ord by range).

Cite: ``vendor/nle/nle/nethack/actions.py:215-251`` (ACTIONS,
USEFUL_ACTIONS); ``vendor/nle/nle/env/base.py:359``
(``observation, done = self.nethack.step(self.actions[action])`` — NLE
remaps index → vendor char-code internally).
"""
from __future__ import annotations

import jax.numpy as jnp

from Nethax.nethax.constants.actions import USEFUL_ACTIONS, ACTIONS


# ---------------------------------------------------------------------------
# NLE-INDEX → ASCII-ORD lookup table
# ---------------------------------------------------------------------------
#
# Layout: a 1-D ``jnp.int32`` array of length len(USEFUL_ACTIONS) == 86.
# ``NLE_INDEX_TO_ASCII[i]`` is the ASCII int that vendor cmd.c receives
# when NLE forwards ``self.actions[i]``.  ``USEFUL_ACTIONS`` is the
# default action tuple NLE constructs (vendor base.py:235).
#
# Example mappings (verified against vendor/nle/nle/nethack/actions.py):
#   0  → 107  (CompassDirection.N   = ord('k'))
#   1  → 108  (CompassDirection.E   = ord('l'))
#   2  → 106  (CompassDirection.S   = ord('j'))
#   3  → 104  (CompassDirection.W   = ord('h'))
#   4  → 117  (CompassDirection.NE  = ord('u'))
#   5  → 110  (CompassDirection.SE  = ord('n'))
#   6  →  98  (CompassDirection.SW  = ord('b'))
#   7  → 121  (CompassDirection.NW  = ord('y'))
#   16 →  60  (MiscDirection.UP     = ord('<'))
#   17 →  62  (MiscDirection.DOWN   = ord('>'))
#   18 →  46  (MiscDirection.WAIT   = ord('.'))
#   19 →  13  (MiscAction.MORE      = ord('\\r'))
#   61 → 115  (Command.SEARCH       = ord('s'))
#
NLE_INDEX_TO_ASCII: jnp.ndarray = jnp.asarray(
    [int(a) for a in USEFUL_ACTIONS], dtype=jnp.int32
)

# Public size constant — every consumer should use this rather than
# hard-coding 86, in case NLE adds/removes actions.
N_NLE_ACTIONS: int = int(NLE_INDEX_TO_ASCII.shape[0])
assert N_NLE_ACTIONS == 86, (
    f"Unexpected NLE useful-action count {N_NLE_ACTIONS}; expected 86 "
    "(vendor/nle/nle/nethack/actions.py USEFUL_ACTIONS)."
)


# ---------------------------------------------------------------------------
# Full 121-entry index → ASCII (matches NLE's ACTIONS tuple verbatim).
# Provided for callers that construct NLE with the full action space.
# ---------------------------------------------------------------------------

NLE_FULL_INDEX_TO_ASCII: jnp.ndarray = jnp.asarray(
    [int(a) for a in ACTIONS], dtype=jnp.int32
)
N_NLE_FULL_ACTIONS: int = int(NLE_FULL_INDEX_TO_ASCII.shape[0])
assert N_NLE_FULL_ACTIONS == 121, (
    f"Unexpected NLE full-action count {N_NLE_FULL_ACTIONS}; expected 121."
)


# ---------------------------------------------------------------------------
# Caller helper — JIT-pure, vmap-friendly.
# ---------------------------------------------------------------------------

def nle_index_to_ascii(action: jnp.ndarray) -> jnp.ndarray:
    """Convert an NLE useful-action index → Nethax ASCII ord.

    Uses :data:`NLE_INDEX_TO_ASCII` (a static array) for a single
    gather op.  Safe inside ``jax.jit`` / ``vmap``.  Out-of-range
    indices are clipped to ``[0, N_NLE_ACTIONS-1]`` rather than
    raising, so the caller can pre-mask without branching on traced
    values.

    Cite: vendor/nle/nle/env/base.py:359 — ``self.actions[action]``
    is exactly this gather (NumPy-side in vendor; JAX-side here).
    """
    a = jnp.clip(jnp.asarray(action, dtype=jnp.int32),
                 jnp.int32(0), jnp.int32(N_NLE_ACTIONS - 1))
    return NLE_INDEX_TO_ASCII[a]


def maybe_remap_action(action: jnp.ndarray) -> jnp.ndarray:
    """Auto-detect index vs ord and return the equivalent ASCII ord.

    Heuristic (matches the task brief in NLE_TERMINATION_ALIGNMENT.md):

      * ``0 <= action < N_NLE_ACTIONS (86)``  → treat as NLE index, gather.
      * ``action >= N_NLE_ACTIONS``           → treat as raw ASCII ord, pass through.

    The cutoff is unambiguous because ``USEFUL_ACTIONS`` includes
    actions with ASCII values in the high range (``>= 86`` — e.g.
    ``ord('k') = 107``); any input in ``[0, 85]`` is overwhelmingly
    likely to be an NLE index rather than a printable character.

    JIT-pure: implemented as ``jnp.where`` over a single boolean mask.
    Returns ``jnp.int32`` so ``dispatch_action``'s downstream ``clip``
    sees the expected dtype.

    Cite: vendor/nle/nle/env/base.py:359 (the ``self.actions[action]``
    remap NLE does numpy-side).
    """
    a = jnp.asarray(action, dtype=jnp.int32)
    is_index = a < jnp.int32(N_NLE_ACTIONS)
    # Use clipped gather even on the ord branch — the ``where`` below
    # masks it out, but the gather must succeed for both branches under
    # JAX tracing.
    gathered = NLE_INDEX_TO_ASCII[
        jnp.clip(a, jnp.int32(0), jnp.int32(N_NLE_ACTIONS - 1))
    ]
    return jnp.where(is_index, gathered, a)
