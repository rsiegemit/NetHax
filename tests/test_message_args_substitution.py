"""Verify emit() substitutes printf-style args into reserved template slots.

Mirrors vendor pline.c::pline (lines 103-130) which accepts a printf format
string and writes the formatted result into a BUFSZ buffer via vsprintf.

Three messages are wired with arg slots:
  YOU_KILL_MONSTER (3)  -> monster name (from MONSTERS[entry_idx].name)
  FIND_GOLD        (4)  -> decimal numeric (10-char right-aligned)
  YOU_HIT_MONSTER  (11) -> monster name

Other messages have no arg slot; passing args to them must be a no-op so
existing call sites that already pass *args remain backward compatible
(cite messages.py emit(*args) signature).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems.messages import (
    MessageState,
    MessageId,
    emit,
)
from Nethax.nethax.constants.monsters import MONSTERS


def _decode(state: MessageState, n: int = 64) -> str:
    """Read message_buffer[1:n] as ASCII (skip byte 0 which is the msg_id)."""
    raw = bytes(int(b) for b in state.message_buffer[1:n])
    return raw.rstrip(b"\x00").decode("ascii", errors="replace")


def test_find_gold_substitutes_decimal():
    """FIND_GOLD: "You find <N> gold pieces." — arg writes right-aligned digits.

    Cite: vendor hack.c::pickup_gold line ~150 — "%ld gold piece%s".
    """
    s = MessageState.default()
    s = emit(s, int(MessageId.FIND_GOLD), 123)
    text = _decode(s)
    # 10-char right-aligned slot: "        123"
    assert text.startswith("You find "), text
    assert " 123 gold pieces." in text, text


def test_find_gold_zero_renders_as_zero():
    """FIND_GOLD with arg=0 should still render the digit '0' (not blank)."""
    s = MessageState.default()
    s = emit(s, int(MessageId.FIND_GOLD), 0)
    text = _decode(s)
    assert " 0 gold pieces." in text, text


def test_you_kill_monster_substitutes_name():
    """YOU_KILL_MONSTER: monster name from MONSTERS[entry_idx].name.

    Cite: vendor uhitm.c::killed — "You kill the %s%s!" with mon_nam().
    """
    # entry_idx 0 == first vendor monst.c entry (giant ant).
    target_name = MONSTERS[0].name
    s = MessageState.default()
    s = emit(s, int(MessageId.YOU_KILL_MONSTER), 0)
    text = _decode(s)
    assert text.startswith("You kill the "), text
    assert target_name in text, f"missing {target_name!r} in {text!r}"
    assert text.rstrip().endswith("!"), text


def test_you_hit_monster_substitutes_name():
    """YOU_HIT_MONSTER: same monster-name substitution as KILL.

    Cite: vendor uhitm.c::hmon — "You hit %s." with mon_nam().
    """
    target_name = MONSTERS[1].name
    s = MessageState.default()
    s = emit(s, int(MessageId.YOU_HIT_MONSTER), 1)
    text = _decode(s)
    assert text.startswith("You hit the "), text
    assert target_name in text, f"missing {target_name!r} in {text!r}"
    assert text.rstrip().endswith("."), text


def test_no_args_message_ignores_args_silently():
    """Messages without arg slots must accept *args and emit unchanged text."""
    s = MessageState.default()
    s_with_args = emit(s, int(MessageId.YOU_WAIT), 999)
    s_no_args   = emit(s, int(MessageId.YOU_WAIT))
    assert _decode(s_with_args) == _decode(s_no_args)


def test_jit_safe():
    """emit() must trace cleanly under jax.jit with jnp.int32 args."""
    @jax.jit
    def step(state, mid, a):
        return emit(state, mid, a)

    s = MessageState.default()
    s = step(s, jnp.int32(int(MessageId.YOU_KILL_MONSTER)), jnp.int32(2))
    text = _decode(s)
    assert text.startswith("You kill the "), text
    # entry_idx 2 == third monst.c entry.
    assert MONSTERS[2].name in text, text


def test_history_rotation_preserves_args():
    """Successive emit() calls must rotate prior substituted lines into history."""
    s = MessageState.default()
    s = emit(s, int(MessageId.FIND_GOLD), 7)
    s = emit(s, int(MessageId.YOU_KILL_MONSTER), 0)
    # history[1] holds the FIND_GOLD line (prior buffer).  history[0] holds
    # the original empty buffer (rotated first).
    hist1 = bytes(int(b) for b in s.message_history[1][1:64])
    hist1 = hist1.rstrip(b"\x00").decode("ascii", errors="replace")
    assert " 7 gold pieces." in hist1, hist1
