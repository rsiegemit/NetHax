"""Multi-key action state machine — NLE prompt parity.

Vendor NetHack and NLE expose certain commands as multi-step prompts:

  * ``WEAR`` (W)  → "What do you want to wear?" → inv letter
  * ``WIELD`` (w) → inv letter
  * ``QUAFF`` (q) → inv letter
  * ``EAT`` (e)   → inv letter
  * ``READ`` (r)  → inv letter
  * ``PUT_ON`` (P) → inv letter
  * ``TAKE_OFF`` (T) → inv letter
  * ``ZAP`` (z)   → inv letter → direction
  * ``THROW`` (t) → inv letter → direction
  * ``APPLY`` (a) → inv letter (and direction for digging tools)
  * ``CAST`` (Z)  → spell letter → direction

NLE-trained policies emit these as **integer-action sequences** — the
follow-up letters/directions are themselves NLE action enum values
(`CompassDirection.N` for north, `Command.LETTER_A` for slot a, etc.).
Without a state machine that consumes the follow-up, Nethax's auto-pick
behavior breaks transfer for any policy that learned the canonical NLE
two-step pattern.

This module wires the state machine.  ``EnvState`` carries three new
fields (added in state.py):

  * ``pending_action_kind``: which prompt is open (see :class:`PendingActionKind`).
  * ``pending_action_root``: the original action int (so the deferred
     handler knows whether to call wear/wield/quaff/...).
  * ``pending_action_slot``: filled at step 1 of a two-step prompt
     (ZAP/THROW) so step 2's direction can be applied to the chosen item.

Dispatcher integration (action_dispatch.py): before the main switch,
check ``state.pending_action_kind != NONE``.  If so:

  * **AWAIT_INV_LETTER**: decode action arg as a letter (NLE's
    ``Command.LETTER_*`` enums map a→0..z→25, A→26..Z→51).  Clamp to
    a valid inv slot.  Invoke the deferred handler with that slot.
    Clear pending.

  * **AWAIT_DIRECTION**: decode action arg as a compass direction
    (NLE's CompassDirection enum: N/E/S/W/NE/NW/SE/SW/..).  Invoke the
    deferred handler with that direction (and the previously-stored
    slot if relevant).  Clear pending.

  * **AWAIT_LETTER_THEN_DIR**: first step — decode letter, store in
    ``pending_action_slot``, transition kind→AWAIT_DIRECTION.

Cite: vendor/nethack/src/cmd.c (getobj/getdir prompts);
      vendor/nle/nle/nethack/actions.py (LETTER_A..LETTER_Z enums).
"""

from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp


class PendingActionKind(IntEnum):
    """Which prompt the dispatcher is waiting on, if any."""

    NONE                  = 0
    AWAIT_INV_LETTER      = 1  # one-step: WEAR/WIELD/QUAFF/EAT/READ/PUT_ON/TAKE_OFF
    AWAIT_DIRECTION       = 2  # one-step: standalone direction prompt
    AWAIT_LETTER_THEN_DIR = 3  # two-step: ZAP/THROW (letter, then direction)
    AWAIT_DIRECTION_THEN  = 4  # ZAP/THROW step 2: direction-only after letter


# ---------------------------------------------------------------------------
# NLE action enum mappings.
#
# Action int values come from vendor/nle/nle/nethack/actions.py.  We list the
# ones the state machine needs to recognize.  These values are stable across
# NLE versions.
# ---------------------------------------------------------------------------

# NLE Command enum — actions that OPEN a prompt.
PROMPT_OPENERS_INV_LETTER: tuple[int, ...] = (
    # WEAR=87, WIELD=119, QUAFF=113, EAT=101, READ=114, PUT_ON=80, TAKE_OFF=84
    87, 119, 113, 101, 114, 80, 84,
)

PROMPT_OPENERS_LETTER_THEN_DIR: tuple[int, ...] = (
    # ZAP=122, THROW=116, APPLY=97 (some apply targets need a direction)
    122, 116, 97,
)

# NLE Command.LETTER_A..LETTER_Z map to action ints in a contiguous range.
# Per vendor/nle/nle/nethack/actions.py: LETTER_A = ord('a') = 97.  But
# many of those collide with command codes (APPLY=97 = 'a').  NLE
# distinguishes by *context*: when a prompt is open, an 'a'..'z' value
# means a slot letter, not a command.  That's exactly the contract we
# implement here.
def letter_to_slot(action_val: jax.Array) -> jax.Array:
    """Decode an action int as an inventory slot (0..51).

    Lowercase 'a'..'z' → 0..25.  Uppercase 'A'..'Z' → 26..51.  Other
    values clamp to slot 0 (a safe default — the caller is expected to
    only call this when ``pending_action_kind == AWAIT_INV_LETTER``).
    """
    v = action_val.astype(jnp.int32)
    lower = (v >= jnp.int32(ord("a"))) & (v <= jnp.int32(ord("z")))
    upper = (v >= jnp.int32(ord("A"))) & (v <= jnp.int32(ord("Z")))
    slot_lower = v - jnp.int32(ord("a"))
    slot_upper = v - jnp.int32(ord("A")) + jnp.int32(26)
    return jnp.where(
        lower, slot_lower,
        jnp.where(upper, slot_upper, jnp.int32(0))
    ).astype(jnp.int8)


# NLE CompassDirection enum: N=107('k'), E=108('l'), S=106('j'), W=104('h'),
# NE=117('u'), NW=121('y'), SE=110('n'), SW=98('b').
DIR_KEY_N, DIR_KEY_E, DIR_KEY_S, DIR_KEY_W = ord("k"), ord("l"), ord("j"), ord("h")
DIR_KEY_NE, DIR_KEY_NW, DIR_KEY_SE, DIR_KEY_SW = ord("u"), ord("y"), ord("n"), ord("b")


def action_to_direction(action_val: jax.Array) -> jax.Array:
    """Decode an action int as a direction.  Returns a (dy, dx) int8 pair.

    North = (-1, 0); East = (0, +1); etc.  Other values → (0, 0).
    """
    v = action_val.astype(jnp.int32)
    dy = (
        jnp.where(v == jnp.int32(DIR_KEY_N), jnp.int32(-1), jnp.int32(0))
        + jnp.where(v == jnp.int32(DIR_KEY_S), jnp.int32(1), jnp.int32(0))
        + jnp.where(v == jnp.int32(DIR_KEY_NE), jnp.int32(-1), jnp.int32(0))
        + jnp.where(v == jnp.int32(DIR_KEY_NW), jnp.int32(-1), jnp.int32(0))
        + jnp.where(v == jnp.int32(DIR_KEY_SE), jnp.int32(1), jnp.int32(0))
        + jnp.where(v == jnp.int32(DIR_KEY_SW), jnp.int32(1), jnp.int32(0))
    )
    dx = (
        jnp.where(v == jnp.int32(DIR_KEY_E), jnp.int32(1), jnp.int32(0))
        + jnp.where(v == jnp.int32(DIR_KEY_W), jnp.int32(-1), jnp.int32(0))
        + jnp.where(v == jnp.int32(DIR_KEY_NE), jnp.int32(1), jnp.int32(0))
        + jnp.where(v == jnp.int32(DIR_KEY_NW), jnp.int32(-1), jnp.int32(0))
        + jnp.where(v == jnp.int32(DIR_KEY_SE), jnp.int32(1), jnp.int32(0))
        + jnp.where(v == jnp.int32(DIR_KEY_SW), jnp.int32(-1), jnp.int32(0))
    )
    return jnp.array([dy, dx], dtype=jnp.int8)


def open_prompt(state, prompt_kind: int, root_action: int):
    """Transition state to wait for a follow-up action.  No turn consumed."""
    return state.replace(
        pending_action_kind=jnp.int8(int(prompt_kind)),
        pending_action_root=jnp.int8(int(root_action)),
    )


def clear_prompt(state):
    """Drop any pending prompt state (return-to-NONE)."""
    return state.replace(
        pending_action_kind=jnp.int8(0),
        pending_action_root=jnp.int8(0),
        pending_action_slot=jnp.int8(-1),
    )


# Lookup tables built at module load — host-side, no JAX tracing.
_OPENERS_INV_LETTER = frozenset(PROMPT_OPENERS_INV_LETTER)
_OPENERS_LETTER_THEN_DIR = frozenset(PROMPT_OPENERS_LETTER_THEN_DIR)


def is_inv_letter_opener_table() -> jax.Array:
    """Return a uint8[256] mask: 1 if action int opens an inv-letter prompt."""
    mask = [1 if i in _OPENERS_INV_LETTER else 0 for i in range(256)]
    return jnp.array(mask, dtype=jnp.uint8)


def is_letter_then_dir_opener_table() -> jax.Array:
    """Return a uint8[256] mask: 1 if action opens a letter-then-dir prompt."""
    mask = [1 if i in _OPENERS_LETTER_THEN_DIR else 0 for i in range(256)]
    return jnp.array(mask, dtype=jnp.uint8)


_INV_LETTER_OPENER_MASK = is_inv_letter_opener_table()
_LETTER_THEN_DIR_OPENER_MASK = is_letter_then_dir_opener_table()


def action_opens_inv_letter_prompt(action_val: jax.Array) -> jax.Array:
    """True if `action_val` is one of WEAR/WIELD/QUAFF/EAT/READ/PUT_ON/TAKE_OFF."""
    safe = jnp.clip(action_val.astype(jnp.int32), 0, 255)
    return _INV_LETTER_OPENER_MASK[safe] != jnp.uint8(0)


def action_opens_letter_then_dir_prompt(action_val: jax.Array) -> jax.Array:
    """True if `action_val` is one of ZAP/THROW/APPLY (the two-step actions)."""
    safe = jnp.clip(action_val.astype(jnp.int32), 0, 255)
    return _LETTER_THEN_DIR_OPENER_MASK[safe] != jnp.uint8(0)


def is_pending(state) -> jax.Array:
    """True if the state is waiting on a follow-up key."""
    return state.pending_action_kind.astype(jnp.int8) != jnp.int8(0)
