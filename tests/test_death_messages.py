"""Wave 6 Phase A — death message tests.

Covers death_message + DeathCause enum per
    vendor/nethack/src/end.c::done()              (overall flow)
    vendor/nethack/src/end.c::deaths[] / ends[]   (cause-text tables, lines 44-61)
    vendor/nethack/include/hack.h::game_end_types (DIED..ASCENDED, lines 482-499)
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.messages import (
    death_cause_name,
    death_message,
    death_verb,
)
from Nethax.nethax.subsystems.scoring import (
    DeathCause,
    KILLED_BY_MONSTER,
    STARVATION,
    DROWNED,
    BURNED,
    FELL_INTO_LAVA,
    PETRIFIED,
    SLIMED,
    CHOKED,
    POISON,
    N_DEATH_CAUSES,
    mark_ascended,
)


_RNG = jax.random.PRNGKey(0)


def _state_on_level(level: int) -> EnvState:
    """EnvState with dungeon.current_level set to ``level``."""
    state = EnvState.default(_RNG)
    new_dungeon = state.dungeon.replace(current_level=jnp.int8(level))
    return state.replace(dungeon=new_dungeon)


# ---------------------------------------------------------------------------
# Enum sanity — values must match vendor/nethack/include/hack.h.
# ---------------------------------------------------------------------------

def test_death_cause_enum_matches_vendor_values():
    """DeathCause integer values mirror game_end_types exactly (hack.h:482-499)."""
    assert int(DeathCause.DIED)         == 0
    assert int(DeathCause.CHOKING)      == 1
    assert int(DeathCause.POISONING)    == 2
    assert int(DeathCause.STARVING)     == 3
    assert int(DeathCause.DROWNING)     == 4
    assert int(DeathCause.BURNING)      == 5
    assert int(DeathCause.DISSOLVED)    == 6
    assert int(DeathCause.CRUSHING)     == 7
    assert int(DeathCause.STONING)      == 8
    assert int(DeathCause.TURNED_SLIME) == 9
    assert int(DeathCause.GENOCIDED)    == 10
    assert int(DeathCause.PANICKED)     == 11
    assert int(DeathCause.TRICKED)      == 12
    assert int(DeathCause.QUIT)         == 13
    assert int(DeathCause.ESCAPED)      == 14
    assert int(DeathCause.ASCENDED)     == 15
    assert N_DEATH_CAUSES == 16


def test_death_cause_aliases_resolve_to_canonical_values():
    """Caller-friendly aliases (KILLED_BY_MONSTER, etc.) point at the canonical
    DeathCause entries from vendor game_end_types."""
    assert int(KILLED_BY_MONSTER) == int(DeathCause.DIED)
    assert int(STARVATION)        == int(DeathCause.STARVING)
    assert int(DROWNED)           == int(DeathCause.DROWNING)
    assert int(BURNED)            == int(DeathCause.BURNING)
    assert int(FELL_INTO_LAVA)    == int(DeathCause.DISSOLVED)
    assert int(PETRIFIED)         == int(DeathCause.STONING)
    assert int(SLIMED)            == int(DeathCause.TURNED_SLIME)
    assert int(CHOKED)            == int(DeathCause.CHOKING)
    assert int(POISON)            == int(DeathCause.POISONING)


def test_death_cause_name_table_matches_vendor():
    """death_cause_name returns vendor end.c::deaths[] strings (lines 44-50)."""
    # Spot-check the strings exposed in deaths[].
    assert death_cause_name(int(DeathCause.DIED))         == "died"
    assert death_cause_name(int(DeathCause.CHOKING))      == "choked"
    assert death_cause_name(int(DeathCause.STARVING))     == "starvation"
    assert death_cause_name(int(DeathCause.DROWNING))     == "drowning"
    assert death_cause_name(int(DeathCause.BURNING))      == "burning"
    assert death_cause_name(int(DeathCause.ASCENDED))     == "ascended"
    assert death_cause_name(int(DeathCause.ESCAPED))      == "escaped"


def test_death_verb_table_matches_vendor():
    """death_verb returns vendor end.c::ends[] strings (lines 52-61)."""
    assert death_verb(int(DeathCause.DIED))     == "died"
    assert death_verb(int(DeathCause.STARVING)) == "starved"
    assert death_verb(int(DeathCause.DROWNING)) == "drowned"
    assert death_verb(int(DeathCause.BURNING))  == "burned"
    assert death_verb(int(DeathCause.DISSOLVED)) == "dissolved in the lava"


# ---------------------------------------------------------------------------
# Message generation
# ---------------------------------------------------------------------------

def test_killed_by_monster_message_includes_monster_name():
    """DIED + monster_name => 'Killed by a <monster> on dungeon level N'."""
    state = _state_on_level(3)
    msg = death_message(state, DeathCause.DIED, monster_name="giant rat")
    assert "Killed by" in msg
    assert "giant rat" in msg
    assert "dungeon level 3" in msg


def test_killed_by_monster_uses_an_for_vowel_initial_name():
    """KILLED_BY_AN format picks 'an' for vowel-initial monster names."""
    state = _state_on_level(2)
    msg = death_message(state, DeathCause.DIED, monster_name="orc")
    assert "an orc" in msg
    # Consonant-initial uses 'a'.
    msg2 = death_message(state, DeathCause.DIED, monster_name="kobold")
    assert "a kobold" in msg2


def test_killed_by_monster_without_name_uses_fallback():
    """No monster_name => 'something' placeholder, not a crash."""
    state = _state_on_level(1)
    msg = death_message(state, DeathCause.DIED)
    assert "Killed by" in msg
    assert "dungeon level 1" in msg


def test_starvation_message():
    """STARVING => 'Starved to death on dungeon level N'."""
    state = _state_on_level(2)
    msg = death_message(state, DeathCause.STARVING)
    assert msg == "Starved to death on dungeon level 2"


def test_drowned_message():
    """DROWNING => 'Drowned on dungeon level N'."""
    state = _state_on_level(4)
    msg = death_message(state, DeathCause.DROWNING)
    assert msg == "Drowned on dungeon level 4"


def test_burned_message():
    """BURNING => 'Burned to death on dungeon level N'."""
    state = _state_on_level(5)
    msg = death_message(state, DeathCause.BURNING)
    assert msg == "Burned to death on dungeon level 5"


def test_lava_message():
    """DISSOLVED (lava death) => 'Fell into lava on dungeon level N'."""
    state = _state_on_level(6)
    msg = death_message(state, DeathCause.DISSOLVED)
    assert msg == "Fell into lava on dungeon level 6"


def test_petrified_message():
    """STONING => 'Petrified on dungeon level N'."""
    state = _state_on_level(7)
    msg = death_message(state, DeathCause.STONING)
    assert "Petrified" in msg
    assert "level 7" in msg


def test_slimed_message():
    """TURNED_SLIME => 'Turned to slime on dungeon level N'."""
    state = _state_on_level(8)
    msg = death_message(state, DeathCause.TURNED_SLIME)
    assert "slime" in msg.lower()
    assert "level 8" in msg


def test_ascension_message_includes_score():
    """ASCENDED => 'Ascended to demigod status with <score> points'."""
    state = _state_on_level(5)
    # Inject a final_score for the message to pick up.
    state = state.replace(
        scoring=state.scoring.replace(final_score=jnp.int32(12345)),
    )
    msg = death_message(state, DeathCause.ASCENDED)
    assert "Ascended" in msg
    assert "demigod" in msg
    assert "12345" in msg


def test_ascension_message_falls_back_to_running_score():
    """If final_score==0, ascension message uses scoring.score."""
    state = _state_on_level(5)
    state = state.replace(
        scoring=state.scoring.replace(
            score=jnp.int32(777),
            final_score=jnp.int32(0),
        ),
    )
    msg = death_message(state, DeathCause.ASCENDED)
    assert "777" in msg


def test_escaped_message_includes_score():
    """ESCAPED => 'Escaped the dungeon with <score> points'."""
    state = _state_on_level(3)
    state = state.replace(
        scoring=state.scoring.replace(final_score=jnp.int32(500)),
    )
    msg = death_message(state, DeathCause.ESCAPED)
    assert "Escaped" in msg
    assert "500" in msg


def test_quit_message_mentions_level():
    """QUIT => 'Quit the game on dungeon level N'."""
    state = _state_on_level(1)
    msg = death_message(state, DeathCause.QUIT)
    assert "Quit" in msg
    assert "level 1" in msg
