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
    """Mark every conduct as violated so conduct_bonus collapses to 0.

    Wave 29b-2 switched compute_conduct_bonus to the vendor-byte-equal
    ``counters == 0`` kept predicate (insight.c::show_conduct
    ``if (!u.uconduct.X)``), so the helper must bump counters too — the
    legacy violations mask alone leaves counters at 0 and the scorer
    reads every conduct as kept.  Mirrors wave33e fix to
    test_scoring_parity._set_violations.
    """
    return state.replace(
        conduct=state.conduct.replace(
            violations=jnp.ones((N_CONDUCTS,), dtype=jnp.bool_),
            counters=jnp.ones((N_CONDUCTS,), dtype=jnp.int32),
        )
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
    """Final score = experience_points + gold + dlevel_bonus + conduct + ...

    Wave 35: gold pays the vendor 10% death-tax (end.c:1337 ``tmp -= tmp/10``)
    when ``how < PANICKED``; for the non-ascended fixture here that gives
    ``50 - 50/10 = 45`` carried gold contribution.
    """
    state = _violate_all_conducts(_fresh_state())
    # Inject XP and gold; deepest_level=1 (no dlevel bonus); no amulet/ascend.
    # wave16a: compute_final_score now reads u.urexp (state.player_urexp);
    # mirror the test addend there too.
    state = state.replace(
        scoring=add_experience(state.scoring, jnp.int32(123)),
        player_urexp=jnp.int64(123),
        player_gold=jnp.int32(50),
    )
    # 123 (XP) + 45 (gold after 10% death tax) = 168.
    assert int(compute_final_score(state)) == 123 + (50 - 50 // 10)


def test_score_adds_dlevel_bonus():
    """deepest_level adds vendor end.c:1338-1340 depth bonuses to the score.

    Wave 35: ``compute_final_score`` now implements vendor's two-tier depth
    contribution byte-equal:
      travel_b = 50  * max(deepest - 1, 0)               (end.c:1338)
      deep_b   = 1000 * min(10, max(deepest - 20, 0))    (end.c:1340)
    Deep bonus caps at +10000 (deepest >= 30).
    """
    state = _violate_all_conducts(_fresh_state())
    state = state.replace(
        scoring=record_deepest_level(state.scoring, jnp.int8(25)),
    )
    # No xp, no gold, no ascend, no kept conducts.
    # travel_b = 50 * 24 = 1200; deep_b = 1000 * 5 = 5000 → 6200.
    expected_deep = 50 * (25 - 1) + DEEP_LEVEL_BONUS * (25 - 20)
    assert int(compute_final_score(state)) == expected_deep

    # deepest <= 20: only travel_b applies, deep_b = 0.
    state_shallow = _violate_all_conducts(_fresh_state())
    state_shallow = state_shallow.replace(
        scoring=record_deepest_level(state_shallow.scoring, jnp.int8(5)),
    )
    # travel_b = 50 * 4 = 200; deep_b = 0.
    assert int(compute_final_score(state_shallow)) == 50 * (5 - 1)


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
    """ascended=True doubles the (XP+gold+depth) base per vendor end.c:1344-1351.

    The vendor ascension multiplier (2x when kept original alignment, 1.5x
    when converted) is realised by ``asc_b = base if ascended`` in
    ``compute_final_score``.  There is **no flat alignment bonus** — Audit G
    #2 fix removed the spurious 5000 addend that diverged from vendor.

    With xp=500, no gold/depth:
        base = 500;  asc_b = base = 500
        total = 500 + 500 = 1000.
    """
    state = _violate_all_conducts(_fresh_state())
    state = state.replace(
        scoring=add_experience(mark_ascended(state.scoring), jnp.int32(500)),
        player_urexp=jnp.int64(500),
    )
    assert int(compute_final_score(state)) == 500 + 500


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
    """If only PACIFIST is kept, conduct bonus equals PACIFIST bonus alone.

    Wave 29b-2: scorer uses ``counters == 0`` as the kept predicate, so the
    helper must clear PACIFIST's counter too (un-violating the bool mask
    alone leaves the counter at 1 and the scorer reads it as violated).
    """
    state = _violate_all_conducts(_fresh_state())
    # Un-violate just PACIFIST: clear both the legacy bool mask AND the
    # vendor-byte-equal counter (insight.c::show_conduct ``if (!u.uconduct.X)``).
    pid = int(Conduct.PACIFIST)
    v = state.conduct.violations.at[pid].set(False)
    c = state.conduct.counters.at[pid].set(jnp.int32(0))
    state = state.replace(conduct=state.conduct.replace(violations=v, counters=c))
    assert int(compute_conduct_bonus(state)) == _CONDUCT_BONUS[Conduct.PACIFIST]


# ---------------------------------------------------------------------------
# Composition + finalize
# ---------------------------------------------------------------------------

def test_score_combined_terms():
    """All bonus channels combine additively per vendor formula (end.c:1325-1352).

    Wave 35 implements the full vendor base:
        gold_adj = max(gold,0) - max(gold,0)//10        (end.c:1337 death tax)
        travel_b = 50  * max(deepest-1, 0)              (end.c:1338)
        deep_b   = 1000 * min(10, max(deepest-20, 0))   (end.c:1340)
        base     = xp + gold_adj + travel_b + deep_b
        asc_b    = base if ascended else 0              (end.c:1344-1351)
        align_b  = 5000 if ascended (and aligned)       (end.c:1325-1352)
        total    = base + asc_b + artifact_b + align_b + conduct_b
    """
    state = _violate_all_conducts(_fresh_state())
    state = state.replace(
        scoring=add_experience(state.scoring, jnp.int32(100)),
        player_gold=jnp.int32(40),
    )
    state = state.replace(
        scoring=record_deepest_level(state.scoring, jnp.int8(25)),
    )
    state = state.replace(scoring=mark_ascended(state.scoring))

    gold_adj = 40 - 40 // 10            # 36
    travel_b = 50 * (25 - 1)            # 1200
    deep_b   = DEEP_LEVEL_BONUS * 5     # 5000
    base     = 100 + gold_adj + travel_b + deep_b
    expected = base + base               # ascended → base doubled (no flat align bonus)
    assert int(compute_final_score(state)) == expected


def test_finalize_score_caches_on_scoring_slice():
    """finalize_score writes compute_final_score(...) into scoring.final_score.

    Wave 35: gold pays the 10% death tax (end.c:1337) → 23 - 23//10 = 21.
    Total = 77 (XP) + 21 (gold_adj) = 98.
    """
    state = _violate_all_conducts(_fresh_state())
    state = state.replace(
        scoring=add_experience(state.scoring, jnp.int32(77)),
        player_gold=jnp.int32(23),
    )
    new_state = finalize_score(state)
    assert int(new_state.scoring.final_score) == 77 + (23 - 23 // 10)
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
