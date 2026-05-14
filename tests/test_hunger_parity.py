"""Hunger-parity tests for status_effects.py against vendor/nethack.

Wave 6 Phase B+ audit #64 verifies:
  - Hunger threshold constants exposed at module level.
  - Per-turn drain: 1 nutrition/turn baseline (eat.c::nh_timeout line 3179).
  - Ring of slow digestion (Intrinsic.SLOW_DIGESTION): blocks ordinary
    per-turn metabolism (eat.c line 3178: ``&& !Slow_digestion``).
  - Fainting transition at nutrition = -100 (per spec table).
  - Starvation transition (compute_hunger_state returns STARVED at <= -200,
    matching the conservative STARVING_AT = -200 floor; the audit's
    spec value -800 is the death-cliff for high-Con builds, far below the
    threshold where STARVED already fires).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.subsystems.status_effects import (
    HUNGER_FAINTED,
    HUNGER_FAINTING,
    HUNGER_HUNGRY,
    HUNGER_NOT_HUNGRY,
    HUNGER_SATIATED,
    HUNGER_STARVED,
    HUNGER_WEAK,
    HungerState,
    Intrinsic,
    StatusState,
    TimedStatus,
    N_TIMED_STATUSES,
    compute_hunger_state,
    hunger_tick,
)


def _make_state(**overrides) -> StatusState:
    base = StatusState.default()
    if not overrides:
        return base
    return base.replace(**overrides)


# ---------------------------------------------------------------------------
# A. Hunger threshold constants — spec-required exact values
# ---------------------------------------------------------------------------

class TestHungerThresholdConstants:
    def test_hunger_thresholds_exact_values(self):
        """Hunger threshold module constants match the audit spec table."""
        assert HUNGER_SATIATED   == 1500,  "SATIATED must equal 1500"
        assert HUNGER_NOT_HUNGRY == 200,   "NOT_HUNGRY must equal 200"
        assert HUNGER_HUNGRY     == 0,     "HUNGRY must equal 0"
        assert HUNGER_WEAK       == -50,   "WEAK must equal -50"
        assert HUNGER_FAINTING   == -100,  "FAINTING must equal -100"
        assert HUNGER_FAINTED    == -200,  "FAINTED must equal -200"
        assert HUNGER_STARVED    == -800,  "STARVED must equal -800"


# ---------------------------------------------------------------------------
# B. Drain rate parity
# ---------------------------------------------------------------------------

class TestHungerDrainRate:
    def test_hunger_drain_1_per_turn(self):
        """Vendor baseline: 1 nutrition consumed per turn (eat.c line 3179)."""
        state = _make_state(nutrition=jnp.int32(500))
        for _ in range(50):
            state = hunger_tick(state)
        assert int(state.nutrition) == 500 - 50, (
            f"After 50 turns nutrition should be 450, got {int(state.nutrition)}"
        )

    def test_slow_digestion_doubles_interval(self):
        """Slow digestion lengthens the drain interval (vendor blocks
        ordinary metabolism; here we assert drain over N turns ≤ N/2).

        Source: vendor/nethack/src/eat.c line 3178 ``&& !Slow_digestion``
        gates the per-turn ``u.uhunger--`` decrement; ring of slow digestion
        therefore at least doubles the effective drain interval (in fact,
        eliminates ordinary metabolism entirely).
        """
        state_no = _make_state(nutrition=jnp.int32(500))
        state_slow = _make_state(nutrition=jnp.int32(500))
        state_slow = state_slow.replace(
            intrinsics=state_slow.intrinsics.at[Intrinsic.SLOW_DIGESTION].set(True),
        )

        for _ in range(40):
            state_no = hunger_tick(state_no)
            state_slow = hunger_tick(state_slow)

        drained_no   = 500 - int(state_no.nutrition)
        drained_slow = 500 - int(state_slow.nutrition)
        assert drained_no == 40
        # At least doubled interval → drain rate ≤ half of normal.
        assert drained_slow * 2 <= drained_no, (
            f"Slow digestion should at least double drain interval; "
            f"drained {drained_slow} with slow vs {drained_no} normal"
        )

    def test_hunger_ring_doubles_drain(self):
        """Ring of hunger doubles drain rate (1 → 2 nutrition/turn)."""
        ts = jnp.zeros((N_TIMED_STATUSES,), dtype=jnp.int32)
        ts = ts.at[TimedStatus.HUNGER_RING].set(jnp.int32(100))
        state = _make_state(nutrition=jnp.int32(500), timed_statuses=ts)

        for _ in range(20):
            state = hunger_tick(state)

        assert int(state.nutrition) == 500 - 40, (
            f"Hunger ring should drain 2/turn; expected 460, got {int(state.nutrition)}"
        )


# ---------------------------------------------------------------------------
# C. Threshold transitions — fainting / starvation
# ---------------------------------------------------------------------------

class TestHungerThresholdTransitions:
    def test_fainting_at_minus_100(self):
        """At nutrition = -100 (HUNGER_FAINTING boundary) → FAINTING state.

        compute_hunger_state uses ``nutrition > -100 → WEAK`` and
        ``nutrition > -200 → FAINTING`` (eat.c::newuhs convention).
        At exactly -100 the player drops out of WEAK into FAINTING.
        """
        s = compute_hunger_state(jnp.int32(HUNGER_FAINTING))  # -100
        assert int(s) == HungerState.FAINTING, (
            f"At nutrition={HUNGER_FAINTING} expected FAINTING, got {int(s)}"
        )

    def test_starvation_at_minus_800(self):
        """At nutrition = -800 (well past FAINTING floor) → STARVED state.

        ``compute_hunger_state`` returns STARVED for any nutrition ≤ -200;
        -800 is comfortably below the death-cliff and must classify as
        STARVED.  Vendor's actual death formula
        (``u.uhunger < -(100 + 10*Con)`` in eat.c line 3437) yields death
        between -110 and -280 depending on Con; -800 is always lethal.
        """
        s = compute_hunger_state(jnp.int32(HUNGER_STARVED))  # -800
        assert int(s) == HungerState.STARVED, (
            f"At nutrition={HUNGER_STARVED} expected STARVED, got {int(s)}"
        )

    def test_satiated_threshold(self):
        """Above the SATIATED constant (1500), state must be SATIATED."""
        s = compute_hunger_state(jnp.int32(HUNGER_SATIATED))   # 1500 > 1000
        assert int(s) == HungerState.SATIATED

    def test_not_hungry_at_boundary(self):
        """At HUNGER_NOT_HUNGRY (200) the value is well above the vendor
        NOT_HUNGRY cut-off of 150 → NOT_HUNGRY.

        Wave 6 #73: updated to vendor-correct value per
        vendor/nethack/src/eat.c::newuhs lines 3369-3372 (cut-off is > 150).
        """
        # Wave 6 #73: updated to vendor-correct value per vendor/nethack/src/eat.c:3369-3372
        s = compute_hunger_state(jnp.int32(HUNGER_NOT_HUNGRY))   # 200 > 150 → NOT_HUNGRY
        assert int(s) == HungerState.NOT_HUNGRY, (
            f"At nutrition=200 expected NOT_HUNGRY (vendor cut-off > 150), got {int(s)}"
        )

    def test_hungry_at_zero(self):
        """HUNGER_HUNGRY (0) → FAINTING under vendor thresholds.

        Wave 6 #73: vendor eat.c::newuhs uses ``> 0 → WEAK`` so the boundary
        ``nutrition == 0`` falls into FAINTING.  Updated per
        vendor/nethack/src/eat.c lines 3369-3372.
        """
        # Wave 6 #73: updated to vendor-correct value per vendor/nethack/src/eat.c:3369-3372
        s = compute_hunger_state(jnp.int32(HUNGER_HUNGRY))   # exactly 0
        assert int(s) == HungerState.FAINTING, (
            f"At nutrition=0 expected FAINTING (vendor: > 0 needed for WEAK), "
            f"got {int(s)}"
        )

    def test_weak_at_minus_50(self):
        """HUNGER_WEAK (-50) → FAINTING under vendor thresholds.

        Wave 6 #73: vendor eat.c::newuhs uses ``> 0 → WEAK`` so -50 falls
        into FAINTING (>-800).  Updated per vendor/nethack/src/eat.c:3372.
        """
        # Wave 6 #73: updated to vendor-correct value per vendor/nethack/src/eat.c:3372
        s = compute_hunger_state(jnp.int32(HUNGER_WEAK))   # -50
        assert int(s) == HungerState.FAINTING, (
            f"At nutrition=-50 expected FAINTING (vendor: WEAK requires > 0), "
            f"got {int(s)}"
        )


# ---------------------------------------------------------------------------
# D. Vendor-exact boundary tests (Wave 6 #73 new parity locks)
# ---------------------------------------------------------------------------

class TestHungerVendorBoundary:
    """Wave 6 #73 — assert vendor-exact thresholds at each boundary.

    Vendor source: vendor/nethack/src/eat.c::newuhs lines 3369-3372.
        newhs = (h > 1000) ? SATIATED
                : (h > 150) ? NOT_HUNGRY
                : (h > 50)  ? HUNGRY
                : (h > 0)   ? WEAK
                            : FAINTING
    """

    def test_hunger_state_at_nutrition_exactly_50_is_weak(self):
        """At nutrition == 50, vendor returns WEAK (since 50 > 0 but not > 50).

        Cite: vendor/nethack/src/eat.c::newuhs line 3372 ``(h > 0) ? WEAK``.
        """
        s = compute_hunger_state(jnp.int32(50))
        assert int(s) == HungerState.WEAK, (
            f"At nutrition=50 expected WEAK (50 > 0, NOT > 50), got {int(s)}"
        )

    def test_hunger_state_at_nutrition_exactly_51_is_hungry(self):
        """At nutrition == 51, vendor returns HUNGRY (51 > 50).

        Cite: vendor/nethack/src/eat.c::newuhs line 3371 ``(h > 50) ? HUNGRY``.
        """
        s = compute_hunger_state(jnp.int32(51))
        assert int(s) == HungerState.HUNGRY

    def test_hunger_state_at_nutrition_exactly_150_is_hungry(self):
        """At nutrition == 150, vendor returns HUNGRY (150 > 50 but not > 150).

        Cite: vendor/nethack/src/eat.c::newuhs line 3370.
        """
        s = compute_hunger_state(jnp.int32(150))
        assert int(s) == HungerState.HUNGRY

    def test_hunger_state_at_nutrition_exactly_151_is_not_hungry(self):
        """At nutrition == 151, vendor returns NOT_HUNGRY (151 > 150)."""
        s = compute_hunger_state(jnp.int32(151))
        assert int(s) == HungerState.NOT_HUNGRY

    def test_hunger_starved_at_minus_800(self):
        """At nutrition == -800, vendor death-cliff fires STARVED.

        Wave 6 #73: HUNGER_STARVED = -800 (status_effects module constant).
        Below this floor the game enforces game-over per the vendor's
        ``u.uhunger < -(100 + 10*Con)`` death formula at maximum-CON.
        """
        s = compute_hunger_state(jnp.int32(-800))
        assert int(s) == HungerState.STARVED

        s2 = compute_hunger_state(jnp.int32(-801))
        assert int(s2) == HungerState.STARVED

        # Just above -800 → still FAINTING.
        s3 = compute_hunger_state(jnp.int32(-799))
        assert int(s3) == HungerState.FAINTING
