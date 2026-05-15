"""Wave 8c — grouped inventory `i` menu vendor parity tests.

Vendor groups inventory slots by ObjectClass with class headers, in the order
from vendor/nethack/src/options.c::def_inv_order.  Headers come from
vendor/nethack/src/invent.c::names[] (line 4789).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.obs.inv_strs import build_grouped_inv_text


@pytest.fixture(scope="module")
def initial_state():
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    return state


def test_grouped_has_headers(initial_state):
    """The Valkyrie starting kit has 2 weapons + 1 armor; output must have
    both 'Weapons' and 'Armor' headers."""
    lines = build_grouped_inv_text(initial_state)
    assert "Weapons" in lines
    assert "Armor" in lines


def test_grouped_weapons_then_armor(initial_state):
    """Vendor def_inv_order: WEAPON_CLASS precedes ARMOR_CLASS — header
    'Weapons' must appear before 'Armor'."""
    lines = build_grouped_inv_text(initial_state)
    w = lines.index("Weapons")
    a = lines.index("Armor")
    assert w < a


def test_grouped_item_lines_follow_header(initial_state):
    """An item line ("a - ...") must follow its class header."""
    lines = build_grouped_inv_text(initial_state)
    idx = lines.index("Weapons")
    assert lines[idx + 1].startswith("a - "), (
        f"expected first weapon line to be 'a - ...', got {lines[idx+1]!r}"
    )


def test_grouped_no_empty_class_headers(initial_state):
    """We must NOT emit a class header when no slots of that class exist."""
    lines = build_grouped_inv_text(initial_state)
    # The Valkyrie starts with weapons + armor only — no Potions/Scrolls/etc.
    assert "Potions" not in lines
    assert "Scrolls" not in lines
    assert "Spellbooks" not in lines


def test_grouped_returns_list_of_strings(initial_state):
    """Return type contract: list of plain str (not bytes/uint8)."""
    lines = build_grouped_inv_text(initial_state)
    assert isinstance(lines, list)
    assert all(isinstance(L, str) for L in lines)
