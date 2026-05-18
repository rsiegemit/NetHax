"""Wave 6 Phase A — final scoring formula tests.

Covers compute_final_score per
    vendor/nethack/src/end.c::really_done (lines 1325-1352)
    vendor/nethack/src/topten.c (line 675 — t0->points = u.urexp)
    vendor/nethack/src/insight.c::show_conduct (per-conduct bonuses)
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
    AMULET_BONUS,
    ASCENSION_BONUS,
    DEEP_LEVEL_BONUS,
    DLEVEL_BONUS,
    ScoringState,
    add_experience,
    compute_conduct_bonus,
    compute_final_score,
    finalize_score,
    mark_ascended,
    record_deepest_level,
    _CONDUCT_BONUS,
)
from Nethax.nethax.subsystems.conduct import Conduct, N_CONDUCTS, mark_violated
from Nethax.nethax.subsystems.inventory import InventoryState, ItemCategory, make_item
from Nethax.nethax.subsystems.items_jewelry import AmuletEffect


_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_state() -> EnvState:
    """Fresh EnvState with deepest_level=1 (so the dlevel bonus is 0)."""
    return EnvState.default(_RNG)


def _violate_all_conducts(state: EnvState) -> EnvState:
    """Mark every conduct as violated so conduct_bonus collapses to 0."""
    new_violations = jnp.ones((N_CONDUCTS,), dtype=jnp.bool_)
    return state.replace(
        conduct=state.conduct.replace(violations=new_violations)
    )


def _give_amulet(state: EnvState) -> EnvState:
    """Place the Amulet of Yendor in inventory slot 0."""
    amulet = make_item(
        category=int(ItemCategory.AMULET),
        type_id=int(AmuletEffect.YENDOR),
        quantity=1,
        weight=20,
    )
    return state.replace(inventory=InventoryState.from_items([amulet]))


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------

def test_scoring_state_has_wave6_fields():
    """ScoringState.default exposes experience_points, deepest_level,
    ascended, final_score (Wave 6 Phase A schema extension)."""
    s = ScoringState.default()
    assert int(s.experience_points) == 0
    assert int(s.deepest_level) == 1
    assert bool(s.ascended) is False
    assert int(s.final_score) == 0


def test_compute_final_score_returns_int32():
    """compute_final_score returns a jnp int32 scalar."""
    state = _violate_all_conducts(_fresh_state())
    score = compute_final_score(state)
    assert score.dtype == jnp.int32
    assert score.shape == ()


# ---------------------------------------------------------------------------
# Core formula components
# ---------------------------------------------------------------------------

def test_score_includes_xp_and_gold():
    """Final score = experience_points + gold + dlevel_bonus + conduct + ..."""
    state = _violate_all_conducts(_fresh_state())
    # Inject XP and gold; deepest_level=1 (no dlevel bonus); no amulet/ascend.
    state = state.replace(
        scoring=add_experience(state.scoring, jnp.int32(123)),
        player_gold=jnp.int32(50),
    )
    assert int(compute_final_score(state)) == 123 + 50


def test_score_adds_dlevel_bonus():
    """deepest_level>20 adds DEEP_LEVEL_BONUS * (deepest-20) to the final score.

    Updated for vendor formula (end.c:1339-1340): bonus is 0 below level 20,
    DEEP_LEVEL_BONUS * (deepest - 20) above it.
    """
    state = _violate_all_conducts(_fresh_state())
    state = state.replace(
        scoring=record_deepest_level(state.scoring, jnp.int8(25)),
    )
    # No xp, no gold, no ascend, no kept conducts.
    expected = DEEP_LEVEL_BONUS * (25 - 20)
    assert int(compute_final_score(state)) == expected

    # Levels <= 20 yield zero depth bonus.
    state_shallow = _violate_all_conducts(_fresh_state())
    state_shallow = state_shallow.replace(
        scoring=record_deepest_level(state_shallow.scoring, jnp.int8(5)),
    )
    assert int(compute_final_score(state_shallow)) == 0


def test_score_amulet_bonus_when_holding():
    """Amulet of Yendor no longer adds a flat bonus in vendor formula.

    The vendor formula (end.c:1325-1352) does not include an amulet bonus;
    carrying the Amulet is a precondition for ascension, not a score addend.
    Score is 0 with zero XP/gold/depth even while holding it.
    """
    state = _violate_all_conducts(_fresh_state())
    state = _give_amulet(state)
    assert int(compute_final_score(state)) == 0


def test_score_no_amulet_bonus_without_amulet():
    """No amulet in inventory => score is 0 (no change either way)."""
    state = _violate_all_conducts(_fresh_state())
    assert int(compute_final_score(state)) == 0


def test_score_ascension_doubles_xp():
    """ascended=True doubles XP contribution (end.c:1344-1351).

    With xp=500 and ascended, score = 500 + 500 = 1000.
    Replaces the old flat ASCENSION_BONUS test.
    """
    state = _violate_all_conducts(_fresh_state())
    state = state.replace(
        scoring=add_experience(mark_ascended(state.scoring), jnp.int32(500)),
    )
    assert int(compute_final_score(state)) == 1000


# ---------------------------------------------------------------------------
# Conduct bonuses
# ---------------------------------------------------------------------------

def test_score_pacifist_bonus_largest():
    """Pacifist is the highest single-conduct bonus per insight.c."""
    assert _CONDUCT_BONUS[Conduct.PACIFIST] == 200
    # And it dominates every other entry.
    others = [v for k, v in _CONDUCT_BONUS.items() if k != Conduct.PACIFIST]
    assert all(_CONDUCT_BONUS[Conduct.PACIFIST] >= v for v in others)


def test_score_all_conducts_preserved_bonus_sum():
    """Fresh state has all conducts intact; conduct_bonus = sum of all entries."""
    state = _fresh_state()
    expected = sum(_CONDUCT_BONUS.values())
    assert int(compute_conduct_bonus(state)) == expected
    # Final score with deepest=1, xp=0, gold=0 = conduct sum.
    assert int(compute_final_score(state)) == expected


def test_score_violated_conduct_no_bonus():
    """Violating a conduct removes its bonus from the total."""
    state = _fresh_state()
    full_bonus = int(compute_conduct_bonus(state))

    state2 = mark_violated(state, int(Conduct.PACIFIST))
    after = int(compute_conduct_bonus(state2))
    assert after == full_bonus - _CONDUCT_BONUS[Conduct.PACIFIST]


def test_score_only_one_conduct_kept():
    """If only PACIFIST is kept, conduct bonus equals PACIFIST bonus alone."""
    state = _violate_all_conducts(_fresh_state())
    # Un-violate just PACIFIST.
    v = state.conduct.violations.at[int(Conduct.PACIFIST)].set(False)
    state = state.replace(conduct=state.conduct.replace(violations=v))
    assert int(compute_conduct_bonus(state)) == _CONDUCT_BONUS[Conduct.PACIFIST]


# ---------------------------------------------------------------------------
# Composition + finalize
# ---------------------------------------------------------------------------

def test_score_combined_terms():
    """All bonus channels combine additively per vendor formula (end.c:1325-1352)."""
    state = _violate_all_conducts(_fresh_state())
    state = state.replace(
        scoring=add_experience(state.scoring, jnp.int32(100)),
        player_gold=jnp.int32(40),
    )
    state = state.replace(
        scoring=record_deepest_level(state.scoring, jnp.int8(25)),
    )
    state = state.replace(scoring=mark_ascended(state.scoring))

    expected = (
        100                              # XP
        + 100                            # ascension doubles XP (end.c:1350)
        + 40                             # gold
        + DEEP_LEVEL_BONUS * (25 - 20)   # deepest > 20 bonus (end.c:1340)
    )
    assert int(compute_final_score(state)) == expected


def test_finalize_score_caches_on_scoring_slice():
    """finalize_score writes compute_final_score(...) into scoring.final_score."""
    state = _violate_all_conducts(_fresh_state())
    state = state.replace(
        scoring=add_experience(state.scoring, jnp.int32(77)),
        player_gold=jnp.int32(23),
    )
    new_state = finalize_score(state)
    assert int(new_state.scoring.final_score) == 100
    # Original state untouched (immutability).
    assert int(state.scoring.final_score) == 0


def test_compute_final_score_is_jittable():
    """compute_final_score must be JIT-traceable (no Python branches on tracer)."""
    state = _violate_all_conducts(_fresh_state())
    state = state.replace(
        scoring=add_experience(state.scoring, jnp.int32(10)),
        player_gold=jnp.int32(5),
    )
    jitted = jax.jit(compute_final_score)
    assert int(jitted(state)) == 15
