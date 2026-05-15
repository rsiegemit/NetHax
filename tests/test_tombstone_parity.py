"""Wave 8c — tombstone byte-equality vs vendor rip.c::genl_outrip.

Cite: vendor/nethack/src/rip.c (rip_txt lines 27-43, center lines 75-83,
      genl_outrip lines 86-163).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.obs.tombstone import build_tombstone, _RIP_TXT


@pytest.fixture(scope="module")
def initial_state():
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    return state


def test_tombstone_15_lines(initial_state):
    """Vendor rip_txt has exactly 15 non-NULL lines."""
    lines = build_tombstone(initial_state, name="Roy")
    assert len(lines) == 15


def test_tombstone_preserves_rip_frame(initial_state):
    """Lines 0-5 (frame top) and 13-14 (frame bottom) must be byte-equal
    to vendor rip_txt."""
    lines = build_tombstone(initial_state, name="Roy")
    # Top of stone (rip.c lines 27-32) — frame, not substituted.
    for i in range(6):
        assert lines[i] == _RIP_TXT[i], f"line {i} differs"
    # Bottom of stone (rip.c lines 41-42) — frame.
    assert lines[13] == _RIP_TXT[13]
    assert lines[14] == _RIP_TXT[14]


def test_tombstone_name_appears(initial_state):
    """NAME_LINE (index 6) must contain the player's name."""
    lines = build_tombstone(initial_state, name="Roy")
    assert "Roy" in lines[6]


def test_tombstone_gold_au(initial_state):
    """GOLD_LINE (index 7) must contain '<N> Au' (rip.c line 109)."""
    lines = build_tombstone(initial_state, name="X", gold=42)
    assert "42 Au" in lines[7]


def test_tombstone_year_4_digit(initial_state):
    """YEAR_LINE (index 12) must contain the 4-digit year (rip.c L138-140)."""
    lines = build_tombstone(initial_state, name="X", year=2027)
    assert "2027" in lines[12]


def test_tombstone_killer_in_death_lines(initial_state):
    """DEATH_LINE (index 8) onward must contain the killer description.
    Vendor splits long killer strings across lines 8..11 at space boundaries
    (rip.c lines 116-135)."""
    lines = build_tombstone(initial_state, name="X",
                            killer="killed by a giant ant")
    combined = "".join(lines[8:12])
    assert "killed" in combined
    assert "ant" in combined


def test_tombstone_center_preserves_substituted_line_width(initial_state):
    """Vendor center() (rip.c lines 75-83) copies bytes IN-PLACE — the
    substituted line's length must equal the template line's length.
    Note: the RIP card has uneven line widths overall (template's last two
    rows are wider for the stone base), so we only check the four lines
    that ARE substituted: NAME, GOLD, DEATH (4 lines), YEAR."""
    lines = build_tombstone(initial_state, name="Stripling",
                            killer="killed by a kobold", year=2026, gold=100)
    SUBSTITUTED = [6, 7, 8, 9, 10, 11, 12]
    for i in SUBSTITUTED:
        assert len(lines[i]) == len(_RIP_TXT[i]), (
            f"line {i} width drift: template={len(_RIP_TXT[i])} "
            f"actual={len(lines[i])}"
        )
