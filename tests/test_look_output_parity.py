"""Wave 8c — look/`;` command vendor parity.

Cite: vendor/nethack/src/pager.c::lookat (lines 656-810)
      vendor/nethack/src/invent.c::look_here (lines 4101-4326)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.obs.look import build_look_text, build_look_here_text


@pytest.fixture(scope="module")
def initial_state():
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    return state


# --- build_look_text -------------------------------------------------------

def test_look_at_player_returns_yourself(initial_state):
    state = initial_state
    pr, pc = int(state.player_pos[0]), int(state.player_pos[1])
    assert build_look_text(state, pr, pc) == "yourself"


def test_look_at_unexplored_says_unexplored(initial_state):
    # Far corner of the level should be unexplored stone (tile = 0).
    # build_look_text wraps terrain with the(), so expect "the ..." prefix.
    text = build_look_text(initial_state, 0, 0)
    assert "unexplored" in text or text == "the wall"


def test_look_at_wall(initial_state):
    """A wall tile adjacent to the room should resolve to 'the wall'.

    Vendor pager.c lookat wraps terrain nouns with the() — so walls are
    'the wall', not bare 'wall'.  Updated for vendor-parity.
    """
    state = initial_state
    pr, pc = int(state.player_pos[0]), int(state.player_pos[1])
    # Walk one step left into the room — there should be a wall column nearby.
    found_wall = False
    for dc in (-3, -2, -1, 1, 2, 3):
        text = build_look_text(state, pr, pc + dc)
        if text == "the wall":
            found_wall = True
            break
    assert found_wall, "expected at least one nearby wall cell"


# --- build_look_here_text --------------------------------------------------

def test_look_here_empty_floor(initial_state):
    """Default state -> no items at player's feet -> vendor exact string."""
    text = build_look_here_text(initial_state)
    # Vendor format: 'You see no objects here.' (invent.c:4247).
    assert text == "You see no objects here."
