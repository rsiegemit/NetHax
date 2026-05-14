"""Wave 5 Phase 4 engrave tests.

Covers:
  - ENGRAVE action: sets engraving text + kind at the player's position.
  - ENGRAVE action: marks the ELBERETHLESS conduct violated.
  - Engraving persistence: a fresh engrave at a *new* tile leaves the old
    engraving intact; an engrave at the SAME tile overwrites the text.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.engrave import (
    handle_engrave,
    ENGR_DUST,
    ENGRAVE_TEXT_LEN,
)
from Nethax.nethax.subsystems.conduct import Conduct
from Nethax.nethax.constants.actions import Command
from Nethax.nethax.subsystems.action_dispatch import dispatch_action


_RNG = jax.random.PRNGKey(0)


def _fresh_state(row: int = 5, col: int = 7) -> EnvState:
    state = EnvState.default(_RNG)
    return state.replace(player_pos=jnp.array([row, col], dtype=jnp.int16))


def test_engrave_action_sets_engraving_at_player_pos():
    """handle_engrave writes Elbereth (ASCII bytes) at the player's tile."""
    state = _fresh_state(row=5, col=7)
    new_state = handle_engrave(state, _RNG)

    assert bool(new_state.engrave.has_engraving[5, 7]) is True
    assert int(new_state.engrave.engraving_kind[5, 7]) == int(ENGR_DUST)
    # First 8 bytes spell out 'Elbereth'
    expected = list(b"Elbereth")
    actual = [int(b) for b in new_state.engrave.text[5, 7, :8]]
    assert actual == expected, f"Expected Elbereth bytes, got {actual}"
    # No engraving on other tiles
    assert bool(new_state.engrave.has_engraving[0, 0]) is False


def test_engrave_action_violates_elberethless():
    """Engraving flips the ELBERETHLESS conduct (insight.c ~2206)."""
    state = _fresh_state()
    assert bool(state.conduct.violations[int(Conduct.ELBERETHLESS)]) is False
    new_state = handle_engrave(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.ELBERETHLESS)]) is True


def test_engrave_persists_until_overwritten():
    """Engraving at tile A persists when engraving at a different tile B;
    a second engrave at tile A overwrites (still Elbereth in dust)."""
    # First engrave at (5, 7).
    state = _fresh_state(row=5, col=7)
    state = handle_engrave(state, _RNG)
    assert bool(state.engrave.has_engraving[5, 7]) is True

    # Move and engrave at (3, 4) — original should persist.
    state = state.replace(player_pos=jnp.array([3, 4], dtype=jnp.int16))
    state = handle_engrave(state, _RNG)
    assert bool(state.engrave.has_engraving[5, 7]) is True
    assert bool(state.engrave.has_engraving[3, 4]) is True

    # Re-engrave at (5, 7): still Elbereth in dust (same text).
    state = state.replace(player_pos=jnp.array([5, 7], dtype=jnp.int16))
    state = handle_engrave(state, _RNG)
    assert bool(state.engrave.has_engraving[5, 7]) is True
    assert int(state.engrave.engraving_kind[5, 7]) == int(ENGR_DUST)


def test_engrave_action_via_dispatch():
    """The Command.ENGRAVE keypress also fires the engrave handler."""
    state = _fresh_state(row=10, col=12)
    new_state = dispatch_action(state, jnp.int32(int(Command.ENGRAVE)), _RNG)
    assert bool(new_state.engrave.has_engraving[10, 12]) is True
    assert bool(new_state.conduct.violations[int(Conduct.ELBERETHLESS)]) is True
