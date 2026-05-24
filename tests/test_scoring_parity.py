"""Wave 17 — bit-equal scoring parity vs vendor end.c::really_done.

For each sample game outcome below we hand-compute the expected final score
component-by-component and assert ``compute_final_score`` returns that exact
value.  This locks the Nethax formula to the vendor breakdown (end.c:1325-1352):

    gold_adj = gold - gold/10                # 10% death tax (how < PANICKED only)
    gold_adj = gold                          # ASCENDED skips the tax (end.c:1336)
    travel_b = 50 * (deepest - 1)
    deep_b   = 1000 * min(10, max(0, deepest - 20))
    base     = urexp + gold_adj + travel_b + deep_b
    asc_b    = base if ascended else 0       # ascension doubles base
    final    = base + asc_b + artifact_b + conduct_b

The old per-event ``AMULET_BONUS`` / ``ASCENSION_BONUS`` flat constants were
removed in wave16-followup21 in favor of the vendor doubling rule. Tests
update expected totals accordingly. The amulet path is now implicit via
the deepest-level travel/deep bonus (the player can only have the amulet
after reaching DL 26+).

# vendor/nethack/src/end.c:1325-1352 (really_done final score block)
# vendor/nethack/src/topten.c:675 (t0->points = u.urexp)
# vendor/nethack/src/insight.c::show_conduct (per-conduct bonus table)
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.scoring import (
    AMULET_BONUS,
    ASCENSION_BONUS,
    DLEVEL_BONUS,
    _CONDUCT_BONUS,
    add_experience,
    compute_final_score,
    mark_ascended,
    record_deepest_level,
)
from Nethax.nethax.subsystems.conduct import Conduct, N_CONDUCTS
from Nethax.nethax.subsystems.inventory import InventoryState, ItemCategory, make_item
from Nethax.nethax.subsystems.items_jewelry import AmuletEffect


_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Sanity: vendor constants
# ---------------------------------------------------------------------------

# Total of every conduct bonus when nothing has been violated.
_ALL_CONDUCTS_BONUS = sum(_CONDUCT_BONUS.values())


def test_conduct_bonus_table_sum_is_900():
    """Per the Nethax simplification documented in scoring.py, the flat
    per-conduct bonuses sum to 900 (matches insight.c per-conduct totals).
    """
    assert _ALL_CONDUCTS_BONUS == 900


def test_vendor_constants_match_audit_table():
    """Audit reference constants — AMULET=10000, ASCENSION=50000, DLEVEL=50.

    # vendor/nethack/src/end.c:1325-1352 (really_done bonuses).
    """
    assert AMULET_BONUS == 10000
    assert ASCENSION_BONUS == 50000
    assert DLEVEL_BONUS == 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh() -> EnvState:
    return EnvState.default(_RNG)


def _set_violations(state: EnvState, kept_conducts: list) -> EnvState:
    """Return state with only the conducts in ``kept_conducts`` preserved.

    Every other conduct is marked violated.  Sets both ``violations`` (legacy
    bool mask) and ``counters`` (vendor u.uconduct.<field> long counter) so
    the two fields stay in sync — matches ``violate()`` / ``increment_counter()``
    semantics in conduct.py.  Wave 29b-2 switched
    ``compute_conduct_bonus`` to use ``counters == 0`` as the kept predicate
    (insight.c byte-equal), so fixtures must bump counters too.
    """
    kept = set(int(c) for c in kept_conducts)
    violated = [i not in kept for i in range(N_CONDUCTS)]
    counters = [1 if v else 0 for v in violated]
    return state.replace(
        conduct=state.conduct.replace(
            violations=jnp.array(violated, dtype=jnp.bool_),
            counters=jnp.array(counters, dtype=jnp.int32),
        ),
    )


def _give_amulet(state: EnvState) -> EnvState:
    amulet = make_item(
        category=int(ItemCategory.AMULET),
        type_id=int(AmuletEffect.YENDOR),
        quantity=1,
        weight=20,
    )
    return state.replace(inventory=InventoryState.from_items([amulet]))


def _scenario(
    *,
    xp: int,
    gold: int,
    deepest: int,
    amulet: bool,
    ascended: bool,
    kept_conducts: list,
) -> EnvState:
    state = _fresh()
    state = _set_violations(state, kept_conducts)
    state = state.replace(
        scoring=add_experience(state.scoring, jnp.int32(xp)),
        player_gold=jnp.int32(gold),
    )
    state = state.replace(
        scoring=record_deepest_level(state.scoring, jnp.int8(deepest)),
    )
    if amulet:
        state = _give_amulet(state)
    if ascended:
        state = state.replace(scoring=mark_ascended(state.scoring))
    return state


# ---------------------------------------------------------------------------
# Sample 1: Quick death — XL 1, XP 5, gold 0, deepest 1, no amulet, no ascend,
# all conducts preserved.  Expected = 5 + 0 + 0 + 0 + 900 = 905.
# ---------------------------------------------------------------------------

def test_score_quick_death_905():
    """XP 5 + gold 0 + DL bonus 0 + no ascend + all conducts kept = 905.

    Vendor end.c:1325-1352 breakdown:
      gold_adj = 0 - 0  = 0
      travel_b = 50 * 0 = 0      (deepest=1)
      deep_b   = 0
      base     = 5 + 0 + 0 + 0 = 5
      asc_b    = 0               (not ascended)
      conduct_b= 900             (all kept)
      final    = 5 + 900 = 905
    """
    state = _scenario(
        xp=5, gold=0, deepest=1,
        amulet=False, ascended=False,
        kept_conducts=list(Conduct),
    )
    expected = 5 + 0 + DLEVEL_BONUS * 0 + 0 + 0 + _ALL_CONDUCTS_BONUS
    assert expected == 905
    assert int(compute_final_score(state)) == 905


# ---------------------------------------------------------------------------
# Sample 2: Mid-game — XL 12, XP 5000, gold 500, deepest 10, no amulet, no
# ascend, FOODLESS broken (all other conducts kept).
#
# Vendor end.c:1325-1352 breakdown:
#   gold_adj = 500 - 50 = 450   (10% death tax, end.c:1337)
#   travel_b = 50 * 9   = 450
#   deep_b   = 0                (deepest=10 < 20)
#   base     = 5000 + 450 + 450 = 5900
#   conduct_b= 900 - 100 = 800  (FOODLESS broken)
#   final    = 5900 + 0 + 800   = 6700
# ---------------------------------------------------------------------------

def test_score_mid_game_6700():
    """5000 XP + 450 gold-after-tax + 50*(10-1) + 800 conducts = 6700.

    # vendor/nethack/src/end.c:1337  ``tmp -= tmp/10`` death tax on gold.
    # vendor/nethack/src/end.c:1338  ``tmp += 50 * (deepest - 1)``.
    """
    kept = [c for c in Conduct if c != Conduct.FOODLESS]
    state = _scenario(
        xp=5000, gold=500, deepest=10,
        amulet=False, ascended=False,
        kept_conducts=kept,
    )
    gold_adj = 500 - 500 // 10  # 450
    expected = (
        5000
        + gold_adj
        + DLEVEL_BONUS * (10 - 1)
        + 0
        + 0
        + (_ALL_CONDUCTS_BONUS - _CONDUCT_BONUS[Conduct.FOODLESS])
    )
    assert expected == 6700
    assert int(compute_final_score(state)) == 6700


# ---------------------------------------------------------------------------
# Sample 3: Got amulet — XL 20, XP 30000, gold 2000, deepest 26, amulet
# carried, not ascended, only FOODLESS conduct kept.
#
# Vendor end.c:1325-1352 breakdown:
#   gold_adj = 2000 - 200    = 1800
#   travel_b = 50 * 25       = 1250
#   deep_b   = 1000 * min(10, 26-20) = 6000
#   base     = 30000+1800+1250+6000  = 39050
#   asc_b    = 0             (not ascended; the flat AMULET_BONUS was
#                             removed in wave16-followup21 — the deepest-
#                             level travel/deep bonus is the only carrier)
#   conduct_b= 100           (FOODLESS)
#   final    = 39050 + 0 + 100 = 39150
# ---------------------------------------------------------------------------

def test_score_got_amulet_39150():
    """Amulet carried but not ascended → 39150 under vendor doubling formula.

    # vendor/nethack/src/end.c:1334-1352 — base = urexp + gold_adj +
    # travel_b + deep_b; flat AMULET_BONUS does not exist in vendor.
    """
    state = _scenario(
        xp=30000, gold=2000, deepest=26,
        amulet=True, ascended=False,
        kept_conducts=[Conduct.FOODLESS],
    )
    gold_adj = 2000 - 2000 // 10  # 1800
    travel_b = DLEVEL_BONUS * (26 - 1)  # 1250
    deep_b   = 1000 * min(10, 26 - 20)  # 6000
    expected = (
        30000
        + gold_adj
        + travel_b
        + deep_b
        + 0   # no ascension bonus
        + _CONDUCT_BONUS[Conduct.FOODLESS]
    )
    assert expected == 39150
    assert int(compute_final_score(state)) == 39150


# ---------------------------------------------------------------------------
# Sample 4: Ascended pacifist — XL 25, XP 60000, gold 5000, deepest 53,
# amulet+ascend, only PACIFIST conduct kept.
#
# Vendor end.c:1325-1352 breakdown (with full doubling on ascension):
#   gold_adj = 5000             (Audit G #3: ASCENDED skips death tax,
#                                end.c:1336 ``if (how < PANICKED) tmp -= tmp/10``)
#   travel_b = 50 * 52          = 2600
#   deep_b   = 1000 * min(10,33)= 10000
#   base     = 60000+5000+2600+10000 = 77600
#   asc_b    = base             = 77600   (ascension doubles)
#   conduct_b= 200              (PACIFIST)
#   final    = 77600 + 77600 + 200 = 155400
# ---------------------------------------------------------------------------

def test_score_ascended_pacifist_155400():
    """Endgame ascension with pacifist preserved → 155400 under vendor doubling.

    # vendor/nethack/src/end.c:1336      — ASCENDED skips ``tmp -= tmp/10`` tax.
    # vendor/nethack/src/end.c:1344-1351 — ascended multiplies base.
    # vendor/nethack/src/insight.c PACIFIST has the 200-point conduct bonus.
    """
    state = _scenario(
        xp=60000, gold=5000, deepest=53,
        amulet=True, ascended=True,
        kept_conducts=[Conduct.PACIFIST],
    )
    gold_adj = 5000                      # ASCENDED → no death tax
    travel_b = DLEVEL_BONUS * (53 - 1)  # 2600
    deep_b   = 1000 * min(10, 53 - 20)  # 10000
    base     = 60000 + gold_adj + travel_b + deep_b  # 77600
    asc_b    = base                                  # full ascension double
    expected = base + asc_b + _CONDUCT_BONUS[Conduct.PACIFIST]
    assert expected == 155400
    assert int(compute_final_score(state)) == 155400


# ---------------------------------------------------------------------------
# Sample 5: Ascended atheist — XL 30, XP 90000, gold 8000, deepest 53,
# amulet+ascend, only ATHEIST conduct kept.
#
# Vendor end.c:1325-1352 breakdown:
#   gold_adj = 8000             (Audit G #3: ASCENDED skips death tax, end.c:1336)
#   travel_b = 50 * 52          = 2600
#   deep_b   = 1000 * min(10,33)= 10000
#   base     = 90000+8000+2600+10000 = 110600
#   asc_b    = base             = 110600
#   conduct_b= 100              (ATHEIST)
#   final    = 110600 + 110600 + 100 = 221300
# ---------------------------------------------------------------------------

def test_score_ascended_atheist_221300():
    """Endgame ascension with atheist preserved → 221300 under vendor doubling.

    # vendor/nethack/src/end.c:1336      — ASCENDED skips ``tmp -= tmp/10`` tax.
    # vendor/nethack/src/end.c:1344-1351 (ascended doubles base).
    """
    state = _scenario(
        xp=90000, gold=8000, deepest=53,
        amulet=True, ascended=True,
        kept_conducts=[Conduct.ATHEIST],
    )
    gold_adj = 8000                       # ASCENDED → no death tax
    travel_b = DLEVEL_BONUS * (53 - 1)   # 2600
    deep_b   = 1000 * min(10, 53 - 20)   # 10000
    base     = 90000 + gold_adj + travel_b + deep_b  # 110600
    asc_b    = base
    expected = base + asc_b + _CONDUCT_BONUS[Conduct.ATHEIST]
    assert expected == 221300
    assert int(compute_final_score(state)) == 221300


# ---------------------------------------------------------------------------
# Cross-check: compute_final_score remains JIT-traceable through every path.
# ---------------------------------------------------------------------------

def test_compute_final_score_jit_quick_death():
    """All 5 sample scenarios trace under jax.jit (JIT-safety guard)."""
    jitted = jax.jit(compute_final_score)
    state = _scenario(
        xp=5, gold=0, deepest=1,
        amulet=False, ascended=False,
        kept_conducts=list(Conduct),
    )
    assert int(jitted(state)) == 905
