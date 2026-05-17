"""Engraving display parity tests — look.py vs vendor pager.c / engrave.c.

Vendor citations:
  engrave.c::read_engr_at lines 328-397 — per-kind intro + "You read: …"
  pager.c::lookat line 1612 — add_quoted_engraving called for engraving tiles
  pager.c::add_quoted_engraving lines 1629-1667 — appends quoted text
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.engrave import (
    ENGRAVE_TEXT_LEN,
    ENGR_DUST,
    ENGR_BURN,
    ENGR_ENGRAVE,
)
from Nethax.nethax.obs.look import build_look_here_text, build_look_text

_RNG = jax.random.PRNGKey(0)


def _state_with_engraving(row: int, col: int, text: str, kind: int) -> EnvState:
    """Return a state with an engraving written at (row, col)."""
    state = EnvState.default(_RNG)
    state = state.replace(player_pos=jnp.array([row, col], dtype=jnp.int16))

    raw = text.encode("ascii")
    padded = list(raw[:ENGRAVE_TEXT_LEN]) + [0] * (ENGRAVE_TEXT_LEN - len(raw))
    bytes_vec = jnp.array(padded, dtype=jnp.int8)

    eng = state.engrave
    new_engrave = eng.replace(
        text=eng.text.at[row, col, :].set(bytes_vec),
        has_engraving=eng.has_engraving.at[row, col].set(jnp.bool_(True)),
        engraving_kind=eng.engraving_kind.at[row, col].set(jnp.int8(kind)),
    )
    return state.replace(engrave=new_engrave)


def test_look_here_shows_elbereth_dust():
    """build_look_here_text includes 'Elbereth' and 'dust' for DUST engraving.

    Vendor: engrave.c::read_engr_at line 332 — "Something is written here in
    the dust."  Line 396 — 'You read: "Elbereth".'
    """
    state = _state_with_engraving(row=5, col=7, text="Elbereth", kind=ENGR_DUST)
    output = build_look_here_text(state)
    assert "Elbereth" in output, f"Expected 'Elbereth' in output, got: {output!r}"
    assert "dust" in output, f"Expected 'dust' in output, got: {output!r}"


def test_look_at_shows_engraving():
    """build_look_text(state, r, c) also surfaces the engraving at a non-player tile.

    Vendor: pager.c::lookat line 1612 calls add_quoted_engraving.
    Player is placed at (5,7); engraving is at (6,8) so look_text doesn't
    short-circuit to "yourself".
    """
    # Player at (5,7), engraving at a different tile (6,8).
    state = EnvState.default(_RNG)
    state = state.replace(player_pos=jnp.array([5, 7], dtype=jnp.int16))
    raw = "Elbereth".encode("ascii")
    padded = list(raw[:ENGRAVE_TEXT_LEN]) + [0] * (ENGRAVE_TEXT_LEN - len(raw))
    bytes_vec = jnp.array(padded, dtype=jnp.int8)
    eng = state.engrave
    new_engrave = eng.replace(
        text=eng.text.at[6, 8, :].set(bytes_vec),
        has_engraving=eng.has_engraving.at[6, 8].set(jnp.bool_(True)),
        engraving_kind=eng.engraving_kind.at[6, 8].set(jnp.int8(ENGR_DUST)),
    )
    state = state.replace(engrave=new_engrave)

    output = build_look_text(state, 6, 8)
    assert "Elbereth" in output, f"Expected 'Elbereth' in look_text, got: {output!r}"
    assert "dust" in output, f"Expected 'dust' in look_text, got: {output!r}"


def test_no_engraving_no_output():
    """An untouched tile produces no engraving lines.

    Vendor: engrave.c::read_engr_at returns immediately when no engr_at(x,y).
    """
    state = EnvState.default(_RNG)
    pr, pc = int(state.player_pos[0]), int(state.player_pos[1])
    output = build_look_here_text(state)
    assert "dust" not in output
    assert "burned" not in output
    assert "engraved" not in output
    assert "You read" not in output


def test_burn_engraving_string():
    """BURN kind produces "burned" in the output.

    Vendor: engrave.c::read_engr_at line 346-347 — "Some text has been burned
    into the floor here."
    """
    state = _state_with_engraving(row=5, col=7, text="Elbereth", kind=ENGR_BURN)
    output = build_look_here_text(state)
    assert "Elbereth" in output, f"Expected 'Elbereth', got: {output!r}"
    assert "burned" in output, f"Expected 'burned', got: {output!r}"
