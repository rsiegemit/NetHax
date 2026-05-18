"""Tests for to-hit math parity additions: XL, Luck, target-state, udaminc.

Vendor references:
  - vendor/nethack/src/uhitm.c:376  — u.uhitinc
  - vendor/nethack/src/uhitm.c:377  — Luck bonus sgn(Luck)*((|Luck|+2)/3)
  - vendor/nethack/src/uhitm.c:378  — XL contribution (player level)
  - vendor/nethack/src/uhitm.c:387-394 — target-state bonuses
  - vendor/nethack/src/uhitm.c:1450 — u.udaminc (ring of increase damage)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.combat import (
    SKILL_BASIC,
    melee_attack,
    to_hit_roll,
)

# Import these eagerly so module-level JAX arrays are created outside any JIT
# context (prevents pre-existing tracer-leak from lazy imports inside lax.cond).
import Nethax.nethax.subsystems.artifact_powers  # noqa: F401
import Nethax.nethax.subsystems.weapon_dice      # noqa: F401

_N_TRIALS = 500


def _base_state(rng_seed=0):
    """Fresh state with zeroed luck/uhitinc/udaminc, BASIC skill, AC=10 target."""
    state = EnvState.default(jax.random.PRNGKey(rng_seed)).replace(
        player_str=jnp.int16(18),   # strhitbon=1
        player_dex=jnp.int8(14),    # dexbon=0
        player_luck=jnp.int8(0),
        player_uhitinc=jnp.int8(0),
        player_udaminc=jnp.int8(0),
    )
    state = state.replace(
        combat=state.combat.replace(
            weapon_skill=state.combat.weapon_skill.at[0].set(jnp.int8(SKILL_BASIC)),
        )
    )
    return state


def _hit_rate(state, target_ac: int, n: int = _N_TRIALS, seed: int = 42) -> float:
    """Vectorised hit rate for ``to_hit_roll`` over ``n`` trials."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    ac = jnp.int32(target_ac)
    vroll = jax.jit(jax.vmap(lambda k: to_hit_roll(k, state, ac)))
    hits = vroll(keys)
    return float(jnp.mean(hits.astype(jnp.float32)))


# ---------------------------------------------------------------------------
# XL contribution (vendor/nethack/src/uhitm.c:378)
# ---------------------------------------------------------------------------

def test_xl_added_to_hit():
    """Higher XL should produce a strictly higher hit rate.

    vendor/nethack/src/uhitm.c:378:
        tmp += maybe_polyd(youmonst.data->mlevel, u.ulevel)
    i.e. XL is added directly into the to-hit accumulator.
    """
    target_ac = 5  # moderate AC — leaves room to see the effect

    state_xl1 = _base_state().replace(player_xl=jnp.int32(1))
    state_xl10 = _base_state().replace(player_xl=jnp.int32(10))

    rate_xl1 = _hit_rate(state_xl1, target_ac)
    rate_xl10 = _hit_rate(state_xl10, target_ac)

    assert rate_xl10 > rate_xl1, (
        f"XL=10 hit rate ({rate_xl10:.3f}) should exceed XL=1 ({rate_xl1:.3f})"
    )


# ---------------------------------------------------------------------------
# Luck bonus (vendor/nethack/src/uhitm.c:377)
# ---------------------------------------------------------------------------

def test_luck_positive_helps():
    """player_luck=+5 should produce a strictly higher hit rate than luck=-5.

    vendor/nethack/src/uhitm.c:377:
        tmp += sgn(Luck) * ((abs(Luck)+2)/3)
    luck=+5  → bonus = +1*((5+2)//3) = +2
    luck=-5  → bonus = -1*((5+2)//3) = -2
    """
    target_ac = 5
    xl = jnp.int32(1)

    state_lucky = _base_state().replace(player_xl=xl, player_luck=jnp.int8(5))
    state_unlucky = _base_state().replace(player_xl=xl, player_luck=jnp.int8(-5))

    rate_lucky = _hit_rate(state_lucky, target_ac)
    rate_unlucky = _hit_rate(state_unlucky, target_ac)

    assert rate_lucky > rate_unlucky, (
        f"luck=+5 rate ({rate_lucky:.3f}) should exceed luck=-5 ({rate_unlucky:.3f})"
    )


# ---------------------------------------------------------------------------
# Sleeping target bonus (vendor/nethack/src/uhitm.c:392: +2 if msleeping)
# ---------------------------------------------------------------------------

def _melee_hit_rate(state, monster_idx: int = 0, n: int = _N_TRIALS, seed: int = 42) -> float:
    """Hit rate via melee_attack (exercises _single_melee_strike path).

    Uses a fresh RNG split for each trial; returns fraction of trials where
    the returned hit flag is True.
    """
    rng = jax.random.PRNGKey(seed)
    hits = 0
    cur = state
    for _ in range(n):
        rng, sub = jax.random.split(rng)
        _new_state, _dmg, hit = melee_attack(cur, sub, jnp.int32(monster_idx))
        hits += int(hit)
        # Reset hp so the monster stays alive throughout.
        cur = cur  # keep original state (don't accumulate damage)
    return hits / n


@pytest.mark.timeout(1200)
def test_sleeping_target_bonus():
    """Sleeping target should have a higher hit rate than awake target.

    vendor/nethack/src/uhitm.c:392: if (mtmp->msleeping) tmp += 2;
    Modelled via MonsterAIState.asleep (bool field).
    Target-state bonuses live in _single_melee_strike; test via melee_attack.
    """
    xl = jnp.int32(1)

    def _state_with_target(asleep_val: bool):
        state = _base_state().replace(player_xl=xl, player_luck=jnp.int8(0))
        mai = state.monster_ai
        mai = mai.replace(
            alive=mai.alive.at[0].set(True),
            hp=mai.hp.at[0].set(jnp.int32(1000)),
            hp_max=mai.hp_max.at[0].set(jnp.int32(1000)),
            ac=mai.ac.at[0].set(jnp.int8(0)),   # AC=0: moderate difficulty
            asleep=mai.asleep.at[0].set(jnp.bool_(asleep_val)),
        )
        return state.replace(monster_ai=mai)

    state_awake = _state_with_target(False)
    state_asleep = _state_with_target(True)

    rate_awake = _melee_hit_rate(state_awake)
    rate_asleep = _melee_hit_rate(state_asleep)

    assert rate_asleep > rate_awake, (
        f"sleeping target hit rate ({rate_asleep:.3f}) should exceed "
        f"awake hit rate ({rate_awake:.3f})"
    )


# ---------------------------------------------------------------------------
# Immobile target bonus (vendor/nethack/src/uhitm.c:393-394: +4 if !mcanmove)
# We test the structural-immobility proxy: move_speed == 0.
# ---------------------------------------------------------------------------

@pytest.mark.timeout(1200)
def test_immobile_target_bonus():
    """Brown mold (speed=0) should have higher hit rate than newt (speed=6).

    vendor/nethack/src/uhitm.c:393-394:
        if (!mtmp->mcanmove) tmp += 4;
    Modelled here via the static _IS_IMMOBILE[entry_idx] mask (move_speed==0).
    brown mold = entry_idx 156 (speed 0), newt = entry_idx 318 (speed 6).
    Target-state bonuses live in _single_melee_strike; test via melee_attack.
    """
    _BROWN_MOLD_IDX = 156
    _NEWT_IDX = 318
    xl = jnp.int32(1)

    def _state_with_entry(entry_idx: int):
        state = _base_state().replace(player_xl=xl, player_luck=jnp.int8(0))
        mai = state.monster_ai
        mai = mai.replace(
            alive=mai.alive.at[0].set(True),
            hp=mai.hp.at[0].set(jnp.int32(1000)),
            hp_max=mai.hp_max.at[0].set(jnp.int32(1000)),
            ac=mai.ac.at[0].set(jnp.int8(0)),   # AC=0: moderate difficulty
            entry_idx=mai.entry_idx.at[0].set(jnp.int16(entry_idx)),
        )
        return state.replace(monster_ai=mai)

    state_mold = _state_with_entry(_BROWN_MOLD_IDX)
    state_newt = _state_with_entry(_NEWT_IDX)

    rate_mold = _melee_hit_rate(state_mold)
    rate_newt = _melee_hit_rate(state_newt)

    assert rate_mold > rate_newt, (
        f"immobile (brown mold) hit rate ({rate_mold:.3f}) should exceed "
        f"mobile (newt) hit rate ({rate_newt:.3f})"
    )
