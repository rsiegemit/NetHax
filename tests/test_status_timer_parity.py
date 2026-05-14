"""Wave 6 #77 closing-audit — status-effect timer durations vs vendor.

Each timed status's randomised duration is sampled many times and asserted
to fall within the documented vendor range.  Citations live on the
``apply_*`` helpers in ``Nethax/nethax/subsystems/status_effects.py``.

Lethal-status tests advance the step() loop the expected number of turns
and verify that exactly that many ticks elapse before death triggers.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from Nethax.nethax.subsystems.status_effects import (
    HungerState,
    Intrinsic,
    StatusState,
    TimedStatus,
    N_TIMED_STATUSES,
    apply_blind,
    apply_confuse,
    apply_fast,
    apply_food_poisoning_status,
    apply_glib,
    apply_hallucinate,
    apply_paralyze,
    apply_sleep,
    apply_slimed,
    apply_stoned,
    apply_strangled,
    apply_stun,
    step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_SAMPLES = 256  # enough to hit every bucket of a small uniform range


def _samples(apply_fn, status_idx, *, with_rng: bool = True, n=_N_SAMPLES):
    """Draw n samples of an apply_X duration from a fresh state.

    Returns a numpy int array of the rolled durations.
    """
    rolls = []
    for i in range(n):
        state = StatusState.default()
        if with_rng:
            rng = jax.random.PRNGKey(i + 1)
            state = apply_fn(state, rng)
        else:
            state = apply_fn(state)
        rolls.append(int(state.timed_statuses[status_idx]))
    return np.array(rolls, dtype=np.int32)


def _intrinsic_samples(apply_fn, intrinsic_idx, n=_N_SAMPLES):
    rolls = []
    for i in range(n):
        state = StatusState.default()
        rng = jax.random.PRNGKey(i + 1)
        state = apply_fn(state, rng)
        rolls.append(int(state.timed_intrinsics[intrinsic_idx]))
    return np.array(rolls, dtype=np.int32)


# ---------------------------------------------------------------------------
# Randomised duration ranges
# ---------------------------------------------------------------------------

class TestRandomisedDurations:
    def test_sleep_duration_range_1_to_50(self):
        """SLEEP — vendor zap.c:2864 ``fall_asleep(-rnd(50))`` → 1..50."""
        rolls = _samples(apply_sleep, TimedStatus.SLEEP)
        assert rolls.min() >= 1
        assert rolls.max() <= 50
        # Range should span more than a single value.
        assert rolls.max() - rolls.min() >= 20

    def test_stun_duration_range_3_to_7(self):
        """STUN — rn1(5, 3) → 3..7."""
        rolls = _samples(apply_stun, TimedStatus.STUNNED)
        assert rolls.min() >= 3
        assert rolls.max() <= 7
        # Must hit at least two distinct values.
        assert len(set(rolls.tolist())) >= 2

    def test_confuse_duration_range_16_to_22(self):
        """CONFUSE — vendor spell.c:153 ``rn1(7, 16)`` → 16..22."""
        rolls = _samples(apply_confuse, TimedStatus.CONFUSION)
        assert rolls.min() >= 16
        assert rolls.max() <= 22

    def test_blind_duration_range_250_to_349(self):
        """BLIND — vendor spell.c:146 ``rn1(100, 250)`` → 250..349."""
        rolls = _samples(apply_blind, TimedStatus.BLIND)
        assert rolls.min() >= 250
        assert rolls.max() <= 349

    def test_paralyze_duration_range_5_to_10(self):
        """PARALYZE — rn1(6, 5) → 5..10 (vendor potion.c / mhitu.c HOLD)."""
        rolls = _samples(apply_paralyze, TimedStatus.FROZEN)
        assert rolls.min() >= 5
        assert rolls.max() <= 10

    def test_hallucinate_duration_range_25_to_75(self):
        """HALLUCINATE — rnd(50)+25 ≡ rn1(50, 26) → 26..75 (spec 25..75)."""
        rolls = _samples(apply_hallucinate, TimedStatus.HALLUCINATION)
        # Spec window is 25..75 inclusive; our realised values fall in 26..75
        # which lies entirely inside the spec.
        assert rolls.min() >= 25
        assert rolls.max() <= 75

    def test_glib_duration_range_5_to_15(self):
        """GLIB — vendor apply.c:2643 ``rn1(11, 5)`` → 5..15."""
        rolls = _samples(apply_glib, TimedStatus.GLIB)
        assert rolls.min() >= 5
        assert rolls.max() <= 15

    def test_fast_duration_haste_self_range_100_to_200(self):
        """FAST (haste self) — rn1(100, 100) → 100..199 (spec 100..200).

        Cite: vendor spell.c::SPE_HASTE_SELF → speed_up() →
        ``incr_itimeout(&HFast, duration)`` (potion.c:2927).
        """
        rolls = _intrinsic_samples(apply_fast, int(Intrinsic.FAST))
        assert rolls.min() >= 100
        assert rolls.max() <= 200


# ---------------------------------------------------------------------------
# Lethal countdowns — exact turn count then death via step()
# ---------------------------------------------------------------------------

def _make_state_with_status(status_idx: int, turns: int,
                            sick_kind: int = 0) -> StatusState:
    base = StatusState.default()
    ts = jnp.zeros((N_TIMED_STATUSES,), dtype=jnp.int32)
    ts = ts.at[status_idx].set(turns)
    return base.replace(
        timed_statuses=ts,
        sick_kind=jnp.int8(sick_kind),
    )


def _step_once(state, hp=20, hp_max=20, key=0):
    rng = jax.random.PRNGKey(key)
    return step(
        state,
        rng,
        jnp.int32(hp),
        jnp.int32(hp_max),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(1),
        jnp.int8(0),
        jnp.bool_(False),
    )


class TestLethalCountdowns:
    """Vendor: STONED/STRANGLED/FOOD_POISONED kill in 5 turns; SLIMED in 10."""

    def test_stoned_lethal_in_5_turns(self):
        """STONED timer of 5 → death on turn 5 (apply_stoning fires at
        timer == 1, pre-decrement).

        Note: we assert ``done == True`` only; the HP-regen step inside
        ``step()`` runs *after* ``apply_stoning`` and may heal HP from 0
        back to 1 via the vendor-parity probabilistic regen path.  The
        canonical death signal is ``done``, not hp == 0 specifically.
        """
        base = StatusState.default()
        state = apply_stoned(base)
        assert int(state.timed_statuses[TimedStatus.STONED]) == 5

        hp = jnp.int32(20)
        for turn in range(1, 5):
            new_state, hp, _, done = _step_once(state, hp=int(hp), key=turn)
            state = new_state
            assert bool(done) is False, (
                f"died too early at turn {turn} with timer "
                f"{int(state.timed_statuses[TimedStatus.STONED])}"
            )
        # Turn 5: pre-decrement timer is 1 → death fires.
        _, _, _, done = _step_once(state, hp=int(hp), key=5)
        assert bool(done) is True

    def test_slimed_lethal_in_10_turns(self):
        base = StatusState.default()
        state = apply_slimed(base)
        assert int(state.timed_statuses[TimedStatus.SLIMED]) == 10

        hp = jnp.int32(20)
        for turn in range(1, 10):
            new_state, hp, _, done = _step_once(state, hp=int(hp), key=turn)
            state = new_state
            assert bool(done) is False, f"died too early at turn {turn}"
        _, _, _, done = _step_once(state, hp=int(hp), key=10)
        assert bool(done) is True

    def test_strangled_lethal_in_5_turns(self):
        base = StatusState.default()
        state = apply_strangled(base)
        assert int(state.timed_statuses[TimedStatus.STRANGLED]) == 5

        hp = jnp.int32(20)
        for turn in range(1, 5):
            new_state, hp, _, done = _step_once(state, hp=int(hp), key=turn)
            state = new_state
            assert bool(done) is False
        _, _, _, done = _step_once(state, hp=int(hp), key=5)
        assert bool(done) is True

    def test_food_poisoning_lethal_in_5_turns(self):
        base = StatusState.default()
        state = apply_food_poisoning_status(base)
        assert int(state.timed_statuses[TimedStatus.SICK]) == 5
        assert int(state.sick_kind) == 1

        hp = jnp.int32(20)
        for turn in range(1, 5):
            new_state, hp, _, done = _step_once(state, hp=int(hp), key=turn)
            state = new_state
            assert bool(done) is False
        _, _, _, done = _step_once(state, hp=int(hp), key=5)
        assert bool(done) is True


# ---------------------------------------------------------------------------
# JIT-safety smoke test — apply_* helpers must be jittable
# ---------------------------------------------------------------------------

class TestJITSafety:
    def test_apply_confuse_jits(self):
        jitted = jax.jit(apply_confuse)
        state = StatusState.default()
        out = jitted(state, jax.random.PRNGKey(0))
        v = int(out.timed_statuses[TimedStatus.CONFUSION])
        assert 16 <= v <= 22

    def test_apply_blind_jits(self):
        jitted = jax.jit(apply_blind)
        state = StatusState.default()
        out = jitted(state, jax.random.PRNGKey(0))
        v = int(out.timed_statuses[TimedStatus.BLIND])
        assert 250 <= v <= 349
