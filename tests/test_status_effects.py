"""Wave 3 tests for status_effects.py tick mechanics.

Covers:
  - Hunger transitions after many turns (SATIATED → NOT_HUNGRY)
  - Ring of regeneration → HP regens 2x faster
  - Eating an apple → nutrition +50
  - Strangulation timer at 1 → next step → player_hp = 0
  - Blindness timer counts down
  - compute_hunger_state threshold correctness
  - compute_encumbrance threshold correctness
  - Stoning expiry → death
  - Food poisoning expiry → death
  - Slow digestion blocks nutrition drain
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.subsystems.status_effects import (
    StatusState,
    TimedStatus,
    HungerState,
    Encumbrance,
    Intrinsic,
    N_TIMED_STATUSES,
    compute_hunger_state,
    compute_encumbrance,
    hunger_tick,
    tick_timers,
    hp_regen_tick,
    pw_regen_tick,
    apply_strangulation,
    apply_stoning,
    apply_food_poisoning,
    handle_eat,
    step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> StatusState:
    """Build a default StatusState with optional field overrides."""
    base = StatusState.default()
    if not overrides:
        return base
    return base.replace(**overrides)


def _run_step(state, player_hp, player_hp_max, player_pw, player_pw_max, xl, role, done, rng_key=0):
    rng = jax.random.PRNGKey(rng_key)
    return step(
        state,
        rng,
        jnp.int32(player_hp),
        jnp.int32(player_hp_max),
        jnp.int32(player_pw),
        jnp.int32(player_pw_max),
        jnp.int32(xl),
        jnp.int8(role),
        jnp.bool_(done),
    )


# ---------------------------------------------------------------------------
# compute_hunger_state — threshold correctness
# ---------------------------------------------------------------------------

class TestComputeHungerState:
    # Wave 6 #73: updated to vendor-correct thresholds per
    # vendor/nethack/src/eat.c::newuhs lines 3369-3372.
    #   nutrition > 1000 → SATIATED
    #   nutrition >  150 → NOT_HUNGRY
    #   nutrition >   50 → HUNGRY
    #   nutrition >    0 → WEAK
    #   nutrition > -800 → FAINTING
    #   nutrition ≤ -800 → STARVED
    def test_satiated(self):
        assert int(compute_hunger_state(jnp.int32(1001))) == HungerState.SATIATED
        assert int(compute_hunger_state(jnp.int32(1500))) == HungerState.SATIATED

    def test_boundary_satiated(self):
        # Exactly at 1000 → NOT_HUNGRY (> 1000 required for SATIATED)
        assert int(compute_hunger_state(jnp.int32(1000))) == HungerState.NOT_HUNGRY

    def test_not_hungry(self):
        assert int(compute_hunger_state(jnp.int32(900))) == HungerState.NOT_HUNGRY
        # Wave 6 #73: vendor cut-off is > 150 (not > 200).
        assert int(compute_hunger_state(jnp.int32(151))) == HungerState.NOT_HUNGRY

    def test_hungry(self):
        # Wave 6 #73: vendor cut-off is > 50 → HUNGRY (eat.c:3371).
        assert int(compute_hunger_state(jnp.int32(150))) == HungerState.HUNGRY
        assert int(compute_hunger_state(jnp.int32(100))) == HungerState.HUNGRY
        assert int(compute_hunger_state(jnp.int32(51)))  == HungerState.HUNGRY

    def test_weak(self):
        # Wave 6 #73: vendor cut-off is > 0 → WEAK (eat.c:3372).
        assert int(compute_hunger_state(jnp.int32(50)))  == HungerState.WEAK
        assert int(compute_hunger_state(jnp.int32(1)))   == HungerState.WEAK

    def test_fainting(self):
        # Wave 6 #73: FAINTING covers everything in (-800, 0].
        assert int(compute_hunger_state(jnp.int32(0)))    == HungerState.FAINTING
        assert int(compute_hunger_state(jnp.int32(-100))) == HungerState.FAINTING
        assert int(compute_hunger_state(jnp.int32(-799))) == HungerState.FAINTING

    def test_starved(self):
        # Wave 6 #73: STARVED death-cliff at -800 (HUNGER_STARVED constant).
        assert int(compute_hunger_state(jnp.int32(-800)))  == HungerState.STARVED
        assert int(compute_hunger_state(jnp.int32(-1000))) == HungerState.STARVED


# ---------------------------------------------------------------------------
# compute_encumbrance — threshold correctness
# ---------------------------------------------------------------------------

class TestComputeEncumbrance:
    def test_unencumbered(self):
        assert int(compute_encumbrance(jnp.int32(100), jnp.int32(100))) == Encumbrance.UNENCUMBERED

    def test_burdened(self):
        # 150 vs cap 100 → weight * 2 = 300 ≤ cap * 3 = 300 → BURDENED
        assert int(compute_encumbrance(jnp.int32(150), jnp.int32(100))) == Encumbrance.BURDENED

    def test_stressed(self):
        # 200 vs cap 100 → w*2=400 > 3*100=300, ≤ 5*100=500 → STRESSED
        assert int(compute_encumbrance(jnp.int32(200), jnp.int32(100))) == Encumbrance.STRESSED

    def test_strained(self):
        # 300 vs cap 100 → w*2=600 > 500, ≤ 900 → STRAINED
        assert int(compute_encumbrance(jnp.int32(300), jnp.int32(100))) == Encumbrance.STRAINED

    def test_overtaxed(self):
        # 500 vs cap 100 → w*2=1000 > 900, ≤ 1200 → OVERTAXED
        assert int(compute_encumbrance(jnp.int32(500), jnp.int32(100))) == Encumbrance.OVERTAXED

    def test_overloaded(self):
        # 700 vs cap 100 → w*2=1400 > 1200 → OVERLOADED
        assert int(compute_encumbrance(jnp.int32(700), jnp.int32(100))) == Encumbrance.OVERLOADED


# ---------------------------------------------------------------------------
# Hunger tick — transition SATIATED → NOT_HUNGRY after ~800 turns
# ---------------------------------------------------------------------------

class TestHungerTransition:
    def test_satiated_to_not_hungry_after_1000_turns(self):
        """After 1000 turns of draining, a SATIATED player should no longer be SATIATED."""
        # Start well into SATIATED range (nutrition=1500).
        state = _make_state(
            nutrition=jnp.int32(1500),
            hunger_state=jnp.int8(HungerState.SATIATED),
        )
        for _ in range(1000):
            state = hunger_tick(state)

        # 1000 drains at 1/turn → nutrition = 500, which is NOT_HUNGRY.
        assert int(state.hunger_state) == HungerState.NOT_HUNGRY
        assert int(state.nutrition) == 500


# ---------------------------------------------------------------------------
# Ring of regeneration → HP regens 2x faster
# ---------------------------------------------------------------------------

class TestHPRegenWithRing:
    """Vendor-parity HP regen (allmain.c::regen_hp, Wave 6 #78).

    The tests below were rewritten in Wave 6 #77 closing-audit to match the
    new ``hp_regen_tick`` signature: legacy deterministic-interval path was
    removed; callers must now supply CON, timestep and RNG.
    """

    def _tick(self, state, hp, hp_max, xl, role, con, timestep, key):
        """Convenience wrapper around hp_regen_tick with vendor inputs."""
        return hp_regen_tick(
            state, jnp.int32(hp), jnp.int32(hp_max),
            jnp.int32(xl), jnp.int8(role),
            jnp.int32(con), jnp.int32(timestep),
            jax.random.PRNGKey(key),
        )

    def test_regen_faster_with_ring(self):
        """Ring of REGEN heals every turn unconditionally (vendor:
        allmain.c lines 649-665 — ring bypasses the moves%20 and rn2(100)
        gates).  Without the ring, regen requires moves%20==0 AND
        XL+CON > rn2(100); we pick a non-multiple-of-20 timestep so the
        non-ring path is guaranteed to miss."""
        state_with_ring = _make_state()
        new_intrinsics = state_with_ring.intrinsics.at[Intrinsic.REGEN].set(True)
        state_with_ring = state_with_ring.replace(intrinsics=new_intrinsics)
        state_without_ring = _make_state()

        # timestep = 5 → moves % 20 != 0 → non-ring path can never heal.
        # Ring user heals every turn regardless.
        _, hp_ring = self._tick(state_with_ring, 8, 10, xl=1, role=0,
                                con=10, timestep=5, key=42)
        _, hp_no_ring = self._tick(state_without_ring, 8, 10, xl=1, role=0,
                                   con=10, timestep=5, key=42)
        assert int(hp_ring) == 9
        assert int(hp_no_ring) == 8

    def test_regen_fires_on_multiple_of_20_when_check_passes(self):
        """When moves % 20 == 0 and XL+CON > rn2(100), vendor regen
        fires.  We force the probability check to succeed by setting a
        very high (XL + CON), so the rn2(100) draw cannot beat it.
        """
        state = _make_state()
        # XL=50, CON=99 → XL+CON=149 > any rn2(100) ∈ [0,99] → always passes.
        _, hp_after = self._tick(state, 5, 10, xl=50, role=0,
                                 con=99, timestep=20, key=0)
        assert int(hp_after) == 6

    def test_regen_skipped_when_not_multiple_of_20(self):
        """Without ring, moves % 20 != 0 → no heal regardless of rn2(100)."""
        state = _make_state()
        _, hp_after = self._tick(state, 5, 10, xl=50, role=0,
                                 con=99, timestep=19, key=0)
        assert int(hp_after) == 5

    def test_regen_skipped_when_starving(self):
        """Regen is blocked when hunger_state >= WEAK (vendor: encumbrance_ok
        + WEAK gate).  Ring of regen is also blocked in that state."""
        state = _make_state(hunger_state=jnp.int8(HungerState.WEAK))
        new_intrinsics = state.intrinsics.at[Intrinsic.REGEN].set(True)
        state = state.replace(intrinsics=new_intrinsics)
        _, hp_after = self._tick(state, 5, 10, xl=50, role=0,
                                 con=99, timestep=20, key=0)
        assert int(hp_after) == 5


# ---------------------------------------------------------------------------
# Eating an apple → nutrition +50
# ---------------------------------------------------------------------------

class TestHandleEat:
    def test_eat_apple_adds_50_nutrition(self):
        """Eating an apple (nutrition=50, FOOD_CLASS=7) should add 50 nutrition."""
        state = _make_state(nutrition=jnp.int32(500))
        new_state = handle_eat(
            state,
            item_nutrition=jnp.int32(50),
            item_class=jnp.int8(7),     # ObjectClass.FOOD_CLASS
            item_present=jnp.bool_(True),
        )
        assert int(new_state.nutrition) == 550

    def test_eat_non_food_does_nothing(self):
        """Eating a non-food item (class != 7) should not change nutrition."""
        state = _make_state(nutrition=jnp.int32(500))
        new_state = handle_eat(
            state,
            item_nutrition=jnp.int32(50),
            item_class=jnp.int8(2),     # WEAPON_CLASS
            item_present=jnp.bool_(True),
        )
        assert int(new_state.nutrition) == 500

    def test_eat_no_item_does_nothing(self):
        state = _make_state(nutrition=jnp.int32(500))
        new_state = handle_eat(
            state,
            item_nutrition=jnp.int32(50),
            item_class=jnp.int8(7),
            item_present=jnp.bool_(False),
        )
        assert int(new_state.nutrition) == 500

    def test_eat_clamps_at_max(self):
        """Nutrition is clamped at MAX_NUTRITION (2000)."""
        state = _make_state(nutrition=jnp.int32(1980))
        new_state = handle_eat(
            state,
            item_nutrition=jnp.int32(50),
            item_class=jnp.int8(7),
            item_present=jnp.bool_(True),
        )
        assert int(new_state.nutrition) == 2000

    def test_eat_updates_hunger_state(self):
        """Eating when WEAK should update hunger_state."""
        state = _make_state(
            nutrition=jnp.int32(-60),
            hunger_state=jnp.int8(HungerState.WEAK),
        )
        new_state = handle_eat(
            state,
            item_nutrition=jnp.int32(300),
            item_class=jnp.int8(7),
            item_present=jnp.bool_(True),
        )
        # -60 + 300 = 240 → NOT_HUNGRY
        assert int(new_state.hunger_state) == HungerState.NOT_HUNGRY


# ---------------------------------------------------------------------------
# Strangulation timer at 1 → next step → player_hp = 0
# ---------------------------------------------------------------------------

class TestStrangulation:
    def test_strangled_timer_1_causes_death(self):
        """STRANGLED timer == 1: apply_strangulation fires death before decrement."""
        ts = jnp.zeros((N_TIMED_STATUSES,), dtype=jnp.int32)
        ts = ts.at[TimedStatus.STRANGLED].set(1)
        state = _make_state(timed_statuses=ts)

        _, new_hp, new_done = apply_strangulation(
            state, jnp.int32(20), jnp.bool_(False)
        )
        assert int(new_hp) == 0
        assert bool(new_done) is True

    def test_strangled_timer_2_does_not_kill(self):
        """STRANGLED timer == 2: player survives this turn."""
        ts = jnp.zeros((N_TIMED_STATUSES,), dtype=jnp.int32)
        ts = ts.at[TimedStatus.STRANGLED].set(2)
        state = _make_state(timed_statuses=ts)

        _, new_hp, new_done = apply_strangulation(
            state, jnp.int32(20), jnp.bool_(False)
        )
        assert int(new_hp) == 20
        assert bool(new_done) is False

    def test_strangled_via_step(self):
        """Full step() with STRANGLED=1 produces hp=0 and done=True."""
        ts = jnp.zeros((N_TIMED_STATUSES,), dtype=jnp.int32)
        ts = ts.at[TimedStatus.STRANGLED].set(1)
        state = _make_state(timed_statuses=ts)

        new_state, new_hp, new_pw, new_done = _run_step(
            state, player_hp=20, player_hp_max=20,
            player_pw=0, player_pw_max=0,
            xl=1, role=0, done=False,
        )
        assert int(new_hp) == 0
        assert bool(new_done) is True


# ---------------------------------------------------------------------------
# Blindness timer counts down
# ---------------------------------------------------------------------------

class TestBlindnessCountdown:
    def test_blind_timer_decrements(self):
        """BLIND timer decrements by 1 each call to tick_timers."""
        ts = jnp.zeros((N_TIMED_STATUSES,), dtype=jnp.int32)
        ts = ts.at[TimedStatus.BLIND].set(10)
        state = _make_state(timed_statuses=ts)

        for expected in range(9, -1, -1):
            state = tick_timers(state)
            assert int(state.timed_statuses[TimedStatus.BLIND]) == expected

    def test_blind_timer_clamps_at_zero(self):
        """BLIND timer does not go negative."""
        ts = jnp.zeros((N_TIMED_STATUSES,), dtype=jnp.int32)
        ts = ts.at[TimedStatus.BLIND].set(1)
        state = _make_state(timed_statuses=ts)

        state = tick_timers(state)
        state = tick_timers(state)
        assert int(state.timed_statuses[TimedStatus.BLIND]) == 0


# ---------------------------------------------------------------------------
# Stoning expiry → death
# ---------------------------------------------------------------------------

class TestStoning:
    def test_stoned_timer_1_causes_death(self):
        ts = jnp.zeros((N_TIMED_STATUSES,), dtype=jnp.int32)
        ts = ts.at[TimedStatus.STONED].set(1)
        state = _make_state(timed_statuses=ts)
        _, new_hp, new_done = apply_stoning(state, jnp.int32(15), jnp.bool_(False))
        assert int(new_hp) == 0
        assert bool(new_done) is True

    def test_stoned_timer_5_survives(self):
        ts = jnp.zeros((N_TIMED_STATUSES,), dtype=jnp.int32)
        ts = ts.at[TimedStatus.STONED].set(5)
        state = _make_state(timed_statuses=ts)
        _, new_hp, new_done = apply_stoning(state, jnp.int32(15), jnp.bool_(False))
        assert int(new_hp) == 15
        assert bool(new_done) is False


# ---------------------------------------------------------------------------
# Food poisoning expiry → death
# ---------------------------------------------------------------------------

class TestFoodPoisoning:
    def test_food_poison_timer_1_causes_death(self):
        ts = jnp.zeros((N_TIMED_STATUSES,), dtype=jnp.int32)
        ts = ts.at[TimedStatus.SICK].set(1)
        state = _make_state(timed_statuses=ts, sick_kind=jnp.int8(1))
        _, new_hp, new_done = apply_food_poisoning(state, jnp.int32(12), jnp.bool_(False))
        assert int(new_hp) == 0
        assert bool(new_done) is True

    def test_illness_timer_1_does_not_kill(self):
        """sick_kind == 2 (chronic illness) does not fire lethal expiry."""
        ts = jnp.zeros((N_TIMED_STATUSES,), dtype=jnp.int32)
        ts = ts.at[TimedStatus.SICK].set(1)
        state = _make_state(timed_statuses=ts, sick_kind=jnp.int8(2))
        _, new_hp, new_done = apply_food_poisoning(state, jnp.int32(12), jnp.bool_(False))
        assert int(new_hp) == 12
        assert bool(new_done) is False


# ---------------------------------------------------------------------------
# Slow digestion blocks nutrition drain
# ---------------------------------------------------------------------------

class TestSlowDigestion:
    def test_slow_digestion_intrinsic_blocks_drain(self):
        """SLOW_DIGESTION intrinsic should prevent nutrition loss."""
        state = _make_state(nutrition=jnp.int32(500))
        new_intrinsics = state.intrinsics.at[Intrinsic.SLOW_DIGESTION].set(True)
        state = state.replace(intrinsics=new_intrinsics)

        for _ in range(100):
            state = hunger_tick(state)

        assert int(state.nutrition) == 500

    def test_hunger_ring_doubles_drain(self):
        """HUNGER_RING active should drain 2 nutrition/turn."""
        ts = jnp.zeros((N_TIMED_STATUSES,), dtype=jnp.int32)
        ts = ts.at[TimedStatus.HUNGER_RING].set(100)
        state = _make_state(nutrition=jnp.int32(500), timed_statuses=ts)

        for _ in range(10):
            state = hunger_tick(state)

        # 10 turns × 2 drain = 20 drained → 480
        assert int(state.nutrition) == 480
