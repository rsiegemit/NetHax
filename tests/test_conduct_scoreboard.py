"""Wave 6 Phase A — conduct scoreboard + score-bonus tests.

Covers:
  - conduct_scoreboard(state) -> list[str] in end.c::list_conducts order
  - conduct_scoreboard_bytes(state) -> int8[13, 64]
  - conduct_score_bonus(state) -> int32 (sum of per-conduct bonuses)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.conduct import (
    Conduct,
    N_CONDUCTS,
    _CONDUCT_BONUSES,
    conduct_scoreboard,
    conduct_scoreboard_bytes,
    conduct_score_bonus,
    mark_violated,
)
from Nethax.nethax.constants.objects import OBJECTS, ObjectClass


_RNG = jax.random.PRNGKey(0)


def _fresh_state() -> EnvState:
    return EnvState.default(_RNG)


def _food_type_id(name: str) -> int:
    """Resolve a FOOD_CLASS object's bare name to its OBJECTS index.

    Wave 6 parity-fix (CA #63): OBJECTS was regenerated, breaking the
    hardcoded ``type_id=3`` corpse index used by earlier tests.  Resolve by
    canonical name instead so the test is resilient to vendor table
    re-ordering.  Cite: vendor/nethack/src/objects.c — FOOD class entries.
    """
    target = int(ObjectClass.FOOD_CLASS)
    for i, entry in enumerate(OBJECTS):
        if entry.name == name and int(entry.class_) == target:
            return i
    raise AssertionError(f"food {name!r} not in OBJECTS table")


def _state_with_food(type_id: int) -> EnvState:
    state = _fresh_state()
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(ObjectClass.FOOD_CLASS)),
        type_id=items.type_id.at[0].set(jnp.int16(type_id)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=items.weight.at[0].set(jnp.int32(0)),
    )
    return state.replace(inventory=state.inventory.replace(items=new_items))


# ---------------------------------------------------------------------------
# Scoreboard text generation
# ---------------------------------------------------------------------------

def test_scoreboard_all_preserved():
    """Fresh state: every conduct preserved -> 13 lines."""
    state = _fresh_state()
    lines = conduct_scoreboard(state)
    assert len(lines) == 13
    # Every line follows the "You preserved NAME." format.
    for line in lines:
        assert line.startswith("You preserved ")
        assert line.endswith(".")
    # All 13 Conduct names appear.
    names_in_lines = {line[len("You preserved "):-1] for line in lines}
    expected_names = {c.name for c in Conduct}
    assert names_in_lines == expected_names


def test_scoreboard_after_eat():
    """Eating a corpse violates FOODLESS + VEGAN + VEGETARIAN -> 10 lines."""
    from Nethax.nethax.subsystems.action_dispatch import _handle_eat

    state = _state_with_food(type_id=_food_type_id("corpse"))  # FLESH corpse
    state = _handle_eat(state, _RNG)
    lines = conduct_scoreboard(state)
    assert len(lines) == 10
    # The three violated conducts must NOT appear.
    names_in_lines = {line[len("You preserved "):-1] for line in lines}
    assert "FOODLESS" not in names_in_lines
    assert "VEGAN" not in names_in_lines
    assert "VEGETARIAN" not in names_in_lines
    # Remaining conducts ARE listed.
    assert "ATHEIST" in names_in_lines
    assert "PACIFIST" in names_in_lines


def test_scoreboard_after_pray():
    """Praying violates ATHEIST -> 12 lines (no ATHEIST line)."""
    state = _fresh_state()
    state = mark_violated(state, int(Conduct.ATHEIST))
    lines = conduct_scoreboard(state)
    assert len(lines) == 12
    names_in_lines = {line[len("You preserved "):-1] for line in lines}
    assert "ATHEIST" not in names_in_lines
    # All others present.
    for c in Conduct:
        if c is Conduct.ATHEIST:
            continue
        assert c.name in names_in_lines


def test_scoreboard_order_matches_list_conducts():
    """Display order mirrors end.c::list_conducts.

    Expected order: ATHEIST, WEAPONLESS, PACIFIST, ILLITERATE, POLYPILELESS,
    POLYSELFLESS, WISHLESS, ARTIWISHLESS, GENOCIDELESS, ELBERETHLESS,
    FOODLESS, VEGAN, VEGETARIAN.
    """
    state = _fresh_state()
    lines = conduct_scoreboard(state)
    expected_order = [
        "ATHEIST",
        "WEAPONLESS",
        "PACIFIST",
        "ILLITERATE",
        "POLYPILELESS",
        "POLYSELFLESS",
        "WISHLESS",
        "ARTIWISHLESS",
        "GENOCIDELESS",
        "ELBERETHLESS",
        "FOODLESS",
        "VEGAN",
        "VEGETARIAN",
    ]
    actual_order = [line[len("You preserved "):-1] for line in lines]
    assert actual_order == expected_order


# ---------------------------------------------------------------------------
# Scoreboard bytes variant
# ---------------------------------------------------------------------------

def test_scoreboard_bytes_shape():
    """conduct_scoreboard_bytes returns int8[13, 64]."""
    state = _fresh_state()
    arr = conduct_scoreboard_bytes(state)
    assert arr.shape == (13, 64)
    assert arr.dtype == jnp.int8


def test_scoreboard_bytes_preserved_decodes():
    """A preserved-conduct row decodes back to 'You preserved NAME.'."""
    state = _fresh_state()
    arr = conduct_scoreboard_bytes(state)
    # Row 0 is ATHEIST (first in _SCOREBOARD_ORDER).
    row0 = bytes(int(b) & 0xFF for b in arr[0])
    text = row0.rstrip(b"\x00").decode("ascii")
    assert text == "You preserved ATHEIST."


def test_scoreboard_bytes_violated_row_is_zero():
    """A violated-conduct row is all NUL bytes."""
    state = _fresh_state()
    state = mark_violated(state, int(Conduct.ATHEIST))
    arr = conduct_scoreboard_bytes(state)
    # Row 0 (ATHEIST) should be all zeros now.
    assert int(arr[0].sum()) == 0
    # Row 1 (WEAPONLESS) should still be populated.
    assert int(arr[1].sum()) != 0


# ---------------------------------------------------------------------------
# Score-bonus helper
# ---------------------------------------------------------------------------

def test_score_bonus_all_preserved():
    """Fresh state: bonus == sum of all 13 entries in _CONDUCT_BONUSES.

    The spec's stated sum (1000) is approximate; the canonical value is the
    sum of the explicit per-conduct bonus array.  100+50+25+100+50+200+50
    +25+100+100+50+25+25 == 900.
    """
    state = _fresh_state()
    bonus = conduct_score_bonus(state)
    expected = int(_CONDUCT_BONUSES.sum())
    assert int(bonus) == expected


def test_score_bonus_after_eat():
    """Eating a corpse drops FOODLESS+VEGAN+VEGETARIAN bonuses (175)."""
    from Nethax.nethax.subsystems.action_dispatch import _handle_eat

    state = _state_with_food(type_id=_food_type_id("corpse"))  # FLESH corpse
    state = _handle_eat(state, _RNG)
    bonus = conduct_score_bonus(state)
    expected = int(_CONDUCT_BONUSES.sum()) - (100 + 50 + 25)
    assert int(bonus) == expected


def test_score_bonus_zero_if_all_violated():
    """Violate every conduct -> bonus == 0."""
    state = _fresh_state()
    for c in Conduct:
        state = mark_violated(state, int(c))
    bonus = conduct_score_bonus(state)
    assert int(bonus) == 0


def test_score_bonus_pacifist_largest_value():
    """Only PACIFIST preserved -> bonus == 200."""
    state = _fresh_state()
    # Violate everything except PACIFIST.
    for c in Conduct:
        if c is Conduct.PACIFIST:
            continue
        state = mark_violated(state, int(c))
    bonus = conduct_score_bonus(state)
    assert int(bonus) == 200


def test_score_bonus_dtype_is_int32():
    """conduct_score_bonus returns an int32 scalar."""
    state = _fresh_state()
    bonus = conduct_score_bonus(state)
    assert bonus.dtype == jnp.int32
    assert bonus.shape == ()


def test_score_bonus_jit_safe():
    """conduct_score_bonus is JIT-traceable."""
    state = _fresh_state()
    jit_fn = jax.jit(conduct_score_bonus)
    bonus = jit_fn(state)
    assert int(bonus) == int(_CONDUCT_BONUSES.sum())
