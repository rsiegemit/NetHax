"""Parity tests for ascension scoring formula and per-type food nutrition.

Covers:
  1. compute_final_score ASCENDED branch — vendor end.c::really_done
     lines 1325-1352.
  2. _FOOD_NUTRITION table built from vendor/nethack/include/objects.h FOOD()
     macros (objects.py OBJECTS[i].nutrition).
  3. _CORPSE_NUTRITION table built from vendor/nethack/include/permonst.h
     cnutrit field (monsters.py MONSTERS[i].nutrition).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.scoring import (
    ScoringState,
    add_experience,
    compute_final_score,
    mark_ascended,
    record_deepest_level,
)
from Nethax.nethax.subsystems.inventory import (
    InventoryState,
    ItemCategory,
    MAX_INVENTORY_SLOTS,
    make_item,
)
from Nethax.nethax.subsystems.conduct import N_CONDUCTS
from Nethax.nethax.subsystems.action_dispatch import _FOOD_NUTRITION, _CORPSE_NUTRITION
from Nethax.nethax.subsystems.status_effects import MAX_NUTRITION


_RNG = jax.random.PRNGKey(42)

# Food type IDs from vendor/nethack/include/objects.h / constants/objects.py
_APPLE_TYPE_ID      = 252   # objects.h: APPLE, nutrition=50
_LEMBAS_TYPE_ID     = 266   # objects.h: LEMBAS_WAFER, nutrition=800
_FOOD_RATION_TYPE_ID = 268  # objects.h: FOOD_RATION, nutrition=800
_CORPSE_TYPE_ID     = 240   # objects.h: CORPSE, nutrition=0 (looked up via monster)

# Monster index for newt: index 0 in chunk1 is giant ant (nutrition=10),
# but we need the actual newt.  Vendor monsters.h: newt is not in the top
# entries — use giant ant (idx=0, nutrition=10) as a known-good corpse test.
# Cite: vendor/nethack/include/monsters.h line 89 (giant ant, cnutrit=10).
_GIANT_ANT_IDX = 0
_GIANT_ANT_NUTRITION = 10   # monsters.h: giant ant cnutrit=10


def _fresh_state() -> EnvState:
    return EnvState.default(_RNG)


def _violate_all_conducts(state: EnvState) -> EnvState:
    new_violations = jnp.ones((N_CONDUCTS,), dtype=jnp.bool_)
    return state.replace(
        conduct=state.conduct.replace(violations=new_violations)
    )


def _state_with_food(type_id: int, corpse_entry_idx: int = -1) -> EnvState:
    """Return a state with one food item of given type_id in slot 0."""
    item = make_item(
        category=int(ItemCategory.FOOD),
        type_id=type_id,
        quantity=1,
        corpse_entry_idx=corpse_entry_idx,
    )
    state = _fresh_state()
    return state.replace(inventory=InventoryState.from_items([item]))


# ---------------------------------------------------------------------------
# 1. Ascension score formula (vendor end.c:1325-1352)
# ---------------------------------------------------------------------------

def test_ascension_score_formula():
    """Vendor formula: ascension doubles (xp + gold_adj + travel + deep_b).

    Input:  XP=10000, gold=5000, deepest=25, ascended=True.
    gold_adj    = 5000                              (Audit G #3: ASCENDED skips
                                                      the death tax, end.c:1336
                                                      ``if (how < PANICKED) tmp-=tmp/10``)
    travel_b    = 50 * (25 - 1) = 1200              (end.c:1338-1339)
    deep_b      = 1000 * min(10, 25-20) = 5000      (end.c:1340)
    base        = 10000 + 5000 + 1200 + 5000 = 21200
    asc_b       = 21200                             (end.c:1344-1351 doubles)
    alignment_b = 0                                  (Audit G #2: vendor has no
                                                      flat align bonus; the 2x
                                                      doubling already captures
                                                      the aligned-ascension effect)
    conduct_b   = 900                               (local helper leaves counters
                                                      at 0 → all kept under the
                                                      counters==0 predicate)
    total       = 21200 + 21200 + 0 + 900 = 43300
    Cite: vendor/nethack/src/end.c::really_done lines 1325-1352.
    """
    state = _violate_all_conducts(_fresh_state())
    state = state.replace(
        scoring=add_experience(
            record_deepest_level(
                mark_ascended(state.scoring),
                jnp.int8(25),
            ),
            jnp.int32(10000),
        ),
        player_gold=jnp.int32(5000),
    )
    assert int(compute_final_score(state)) == 43300


def test_ascension_score_no_deep_bonus_below_20():
    """deepest <= 20 contributes 0 to the deep bonus even when ascended.

    XP=1000, gold=0, deepest=15, ascended=True.
    gold_adj=0, travel_b=50*14=700, deep_b=0.
    base=1700, asc_b=1700, alignment_b=0 (Audit G #2), conduct_b=900
       → total = 1700 + 1700 + 0 + 900 = 4300.
    """
    state = _violate_all_conducts(_fresh_state())
    state = state.replace(
        scoring=add_experience(
            record_deepest_level(
                mark_ascended(state.scoring),
                jnp.int8(15),
            ),
            jnp.int32(1000),
        ),
        player_gold=jnp.int32(0),
    )
    assert int(compute_final_score(state)) == 4300


# ---------------------------------------------------------------------------
# 2. Food nutrition table (_FOOD_NUTRITION)
# ---------------------------------------------------------------------------

def test_food_nutrition_table_apple():
    """_FOOD_NUTRITION[252] == 50 (apple, objects.h line 1080)."""
    assert int(_FOOD_NUTRITION[_APPLE_TYPE_ID]) == 50


def test_food_nutrition_table_lembas():
    """_FOOD_NUTRITION[266] == 800 (lembas wafer, objects.h line 1106)."""
    assert int(_FOOD_NUTRITION[_LEMBAS_TYPE_ID]) == 800


def test_food_nutrition_table_food_ration():
    """_FOOD_NUTRITION[268] == 800 (food ration, objects.h line 1110)."""
    assert int(_FOOD_NUTRITION[_FOOD_RATION_TYPE_ID]) == 800


def test_food_nutrition_lookup_apple():
    """Eating an apple raises nutrition by ~50.

    Cite: vendor/nethack/include/objects.h line 1080 FOOD("apple",...,50,...).
    """
    state = _violate_all_conducts(_state_with_food(_APPLE_TYPE_ID))
    before = int(state.status.nutrition)
    new_state = jax.jit(
        lambda s: __import__(
            "Nethax.nethax.subsystems.action_dispatch",
            fromlist=["_handle_eat"],
        )._handle_eat(s, _RNG)
    )(state)
    after = int(new_state.status.nutrition)
    assert after - before == 50


def test_food_nutrition_lookup_lembas():
    """Eating lembas raises nutrition by ~800.

    Cite: vendor/nethack/include/objects.h line 1106 FOOD("lembas wafer",...,800,...).
    """
    state = _violate_all_conducts(_state_with_food(_LEMBAS_TYPE_ID))
    # Start at 0 nutrition so we don't hit MAX cap early.
    state = state.replace(status=state.status.replace(nutrition=jnp.int32(0)))
    new_state = jax.jit(
        lambda s: __import__(
            "Nethax.nethax.subsystems.action_dispatch",
            fromlist=["_handle_eat"],
        )._handle_eat(s, _RNG)
    )(state)
    delta = int(new_state.status.nutrition) - 0
    assert delta == 800


# ---------------------------------------------------------------------------
# 3. Corpse nutrition per monster (_CORPSE_NUTRITION)
# ---------------------------------------------------------------------------

def test_corpse_nutrition_table_giant_ant():
    """_CORPSE_NUTRITION[0] == 10 (giant ant, monsters.h line 89 cnutrit=10)."""
    assert int(_CORPSE_NUTRITION[_GIANT_ANT_IDX]) == _GIANT_ANT_NUTRITION


def test_corpse_nutrition_per_monster():
    """Eating a giant-ant corpse raises nutrition by 10.

    Cite: vendor/nethack/include/permonst.h line 68 (cnutrit field);
          vendor/nethack/include/monsters.h line 89 (giant ant cnutrit=10).
    """
    state = _violate_all_conducts(
        _state_with_food(_CORPSE_TYPE_ID, corpse_entry_idx=_GIANT_ANT_IDX)
    )
    state = state.replace(status=state.status.replace(nutrition=jnp.int32(0)))
    new_state = jax.jit(
        lambda s: __import__(
            "Nethax.nethax.subsystems.action_dispatch",
            fromlist=["_handle_eat"],
        )._handle_eat(s, _RNG)
    )(state)
    delta = int(new_state.status.nutrition) - 0
    assert delta == _GIANT_ANT_NUTRITION
