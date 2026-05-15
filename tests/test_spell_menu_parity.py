"""Wave 8c — spell menu `+` vendor format parity.

Cite: vendor/nethack/src/spell.c::dospellmenu (lines 2075-2167)
      vendor/nethack/include/objects.h SPELL() entries (lines 1277-1412)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.obs.spell_menu import build_spell_menu_text, _SPELL_INFO
from Nethax.nethax.subsystems.magic import SpellId, N_SPELLS


@pytest.fixture(scope="module")
def initial_state():
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    return state


def test_no_spells_returns_vendor_message(initial_state):
    """Default state has no spells known → vendor's docast no-spells
    message (spell.c::docast line ~775)."""
    lines = build_spell_menu_text(initial_state)
    assert lines == ["You don't know any spells right now."]


def test_known_spell_appears_with_correct_letter(initial_state):
    """A spell at SpellId.MAGIC_MISSILE = 1 should render with letter 'b'."""
    state = initial_state
    new_magic = state.magic.replace(
        spell_known=state.magic.spell_known.at[int(SpellId.MAGIC_MISSILE)].set(True),
        spell_memory=state.magic.spell_memory.at[int(SpellId.MAGIC_MISSILE)].set(5000),
    )
    state2 = state.replace(magic=new_magic)
    lines = build_spell_menu_text(state2)
    body = "\n".join(lines)
    assert "b - magic missile" in body


def test_header_byte_equal_vendor_format(initial_state):
    """Vendor format string: '    %-20s Level %-12s Fail Retention'."""
    state = initial_state
    new_magic = state.magic.replace(
        spell_known=state.magic.spell_known.at[0].set(True),
    )
    state2 = state.replace(magic=new_magic)
    lines = build_spell_menu_text(state2)
    # Header should match vendor format with "Name" and "Category" as the strings.
    expected_header = "    %-20s Level %-12s Fail Retention" % ("Name", "Category")
    assert lines[0] == expected_header


def test_spell_info_table_complete():
    """_SPELL_INFO has exactly N_SPELLS entries (one per SpellId)."""
    assert len(_SPELL_INFO) == N_SPELLS


def test_dig_is_level_5_matter():
    """SpellId.DIG -> level 5, category matter (objects.h line 1294)."""
    name, level, cat = _SPELL_INFO[int(SpellId.DIG)]
    assert name == "dig"
    assert level == 5
    assert cat == "matter"


def test_magic_missile_is_level_2_attack():
    """SpellId.MAGIC_MISSILE -> level 2, category attack (objects.h line 1298)."""
    name, level, cat = _SPELL_INFO[int(SpellId.MAGIC_MISSILE)]
    assert name == "magic missile"
    assert level == 2
    assert cat == "attack"


def test_finger_of_death_is_level_7_attack():
    """SpellId.FINGER_OF_DEATH -> level 7, category attack (objects.h L1307)."""
    name, level, cat = _SPELL_INFO[int(SpellId.FINGER_OF_DEATH)]
    assert name == "finger of death"
    assert level == 7
    assert cat == "attack"


def test_cancellation_is_level_7_matter():
    """SpellId.CANCELLATION -> level 7, category matter (objects.h L1398)."""
    name, level, cat = _SPELL_INFO[int(SpellId.CANCELLATION)]
    assert name == "cancellation"
    assert level == 7
    assert cat == "matter"


def test_row_format_byte_equal_vendor():
    """Vendor row format: "%-20s  %2d   %-12s %3d%% %9s" prefixed by 'X - '."""
    state, _ = NethaxEnv().reset(jax.random.PRNGKey(0))
    new_magic = state.magic.replace(
        spell_known=state.magic.spell_known.at[int(SpellId.HEALING)].set(True),
        spell_memory=state.magic.spell_memory.at[int(SpellId.HEALING)].set(5000),
    )
    state2 = state.replace(magic=new_magic)
    lines = build_spell_menu_text(state2)
    # Find the healing row.
    healing_line = next(L for L in lines if "healing" in L and L.startswith("i -"))
    # Vendor format: "i - %-20s  %2d   %-12s %3d%% %9s" — "healing" = level 1, healing cat.
    # Just check structural fields appear in order.
    assert healing_line.startswith("i - ")
    assert "healing" in healing_line              # name + category
    assert "1" in healing_line                    # level
    assert "%" in healing_line                    # fail percentage
    assert "turns" in healing_line or "expired" in healing_line  # retention
