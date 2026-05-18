"""Parity tests for the four lethal status effects: STONED, SLIMED, STRANGLED, GLIB.

Vendor citations:
  timeout.c::stoned_dialogue  — STONED kills at countdown 0
  timeout.c::slime_dialogue   — SLIMED kills at countdown 0 (line 495)
  timeout.c::choked           — STRANGLED kills at countdown 0 (lines 890-894)
  status.c::glibs             — GLIB: 1-in-20 drop weapon per turn
  end.c::done                 — cursed AMULET_OF_STRANGULATION applies STRANGLED

Scope: status_effects.py (apply/cure/tick helpers) + items_jewelry.py (strangulation amulet).
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
    N_TIMED_STATUSES,
    apply_stoned,
    cure_stoned,
    apply_slimed,
    apply_strangled,
    tick_stoned_lethal,
    tick_slimed_lethal,
    tick_strangled_lethal,
    tick_glib,
    step,
)
from Nethax.nethax.subsystems.scoring import DeathCause
from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.items_jewelry import AmuletEffect, wear_amulet
from Nethax.nethax.subsystems.inventory import InventoryState, ItemCategory


_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_with_status(status_idx: int, turns: int) -> EnvState:
    """Build an EnvState with a single timed status set to ``turns``."""
    state = EnvState.default(_RNG)
    ts = state.status.timed_statuses.at[status_idx].set(jnp.int32(turns))
    return state.replace(status=state.status.replace(timed_statuses=ts))


def _step_status_once(status_state: StatusState, hp: int = 20, key: int = 0):
    """Run one turn of the StatusState step() loop."""
    rng = jax.random.PRNGKey(key)
    return step(
        status_state,
        rng,
        jnp.int32(hp),
        jnp.int32(hp),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(1),
        jnp.int8(0),
        jnp.bool_(False),
    )


def _env_with_amulet(effect: AmuletEffect, buc: int, slot: int = 0) -> EnvState:
    """Build an EnvState with one amulet in slot 0."""
    state = EnvState.default(_RNG)
    items = state.inventory.items
    new_category = items.category.at[slot].set(jnp.int8(int(ItemCategory.AMULET)))
    new_type_id = items.type_id.at[slot].set(jnp.int16(int(effect)))
    new_qty = items.quantity.at[slot].set(jnp.int16(1))
    new_buc = items.buc_status.at[slot].set(jnp.int8(buc))
    new_items = items.replace(
        category=new_category,
        type_id=new_type_id,
        quantity=new_qty,
        buc_status=new_buc,
    )
    return state.replace(inventory=state.inventory.replace(items=new_items))


# ---------------------------------------------------------------------------
# test_stoned_kills_at_zero
# ---------------------------------------------------------------------------

class TestStonedKillsAtZero:
    """apply_stoned(state, 1) → one tick → done=True, death_cause=STONING.

    Cite: vendor/nethack/src/timeout.c::stoned_dialogue line 200.
    tick_stoned_lethal fires when timer == 1 (pre-decrement).
    """

    def test_stoned_kills_at_zero(self):
        env = _env_with_status(int(TimedStatus.STONED), 1)
        assert int(env.status.timed_statuses[int(TimedStatus.STONED)]) == 1

        env2 = tick_stoned_lethal(env)
        assert bool(env2.done) is True
        assert int(env2.player_hp) == 0
        assert int(env2.scoring.death_cause) == int(DeathCause.STONING)

    def test_stoned_does_not_kill_early(self):
        """Timer > 1 must not trigger death."""
        for turns in (2, 3, 5):
            env = _env_with_status(int(TimedStatus.STONED), turns)
            env2 = tick_stoned_lethal(env)
            assert bool(env2.done) is False, f"died early at turns={turns}"

    def test_stoned_timer_zero_no_kill(self):
        """Timer already 0 (expired last turn) must not double-kill."""
        env = _env_with_status(int(TimedStatus.STONED), 0)
        env2 = tick_stoned_lethal(env)
        assert bool(env2.done) is False


# ---------------------------------------------------------------------------
# test_lizard_cures_stoned
# ---------------------------------------------------------------------------

class TestLizardCuresStoned:
    """apply_stoned then cure_stoned clears the timer.

    Cite: vendor/nethack/src/timeout.c — lizard corpse / prayer / acid clears
    stoning via make_stoned(0, ...).
    """

    def test_lizard_cures_stoned(self):
        base = StatusState.default()
        stoned = apply_stoned(base, turns=5)
        assert int(stoned.timed_statuses[int(TimedStatus.STONED)]) == 5

        cured = cure_stoned(stoned)
        assert int(cured.timed_statuses[int(TimedStatus.STONED)]) == 0

    def test_cure_stoned_idempotent_when_clear(self):
        base = StatusState.default()
        cured = cure_stoned(base)
        assert int(cured.timed_statuses[int(TimedStatus.STONED)]) == 0

    def test_apply_stoned_custom_turns(self):
        base = StatusState.default()
        state = apply_stoned(base, turns=3)
        assert int(state.timed_statuses[int(TimedStatus.STONED)]) == 3


# ---------------------------------------------------------------------------
# test_slimed_kills_at_zero
# ---------------------------------------------------------------------------

class TestSlimedKillsAtZero:
    """apply_slimed gives timer=10; tick_slimed_lethal fires at timer==1.

    Cite: vendor/nethack/src/timeout.c::slime_dialogue — done_timeout(TURNED_SLIME, SLIMED).
    """

    def test_slimed_kills_at_zero(self):
        env = _env_with_status(int(TimedStatus.SLIMED), 1)
        env2 = tick_slimed_lethal(env)
        assert bool(env2.done) is True
        assert int(env2.player_hp) == 0
        assert int(env2.scoring.death_cause) == int(DeathCause.TURNED_SLIME)

    def test_slimed_does_not_kill_early(self):
        for turns in (2, 5, 10):
            env = _env_with_status(int(TimedStatus.SLIMED), turns)
            env2 = tick_slimed_lethal(env)
            assert bool(env2.done) is False, f"died early at turns={turns}"

    def test_apply_slimed_sets_timer_10(self):
        base = StatusState.default()
        state = apply_slimed(base)
        assert int(state.timed_statuses[int(TimedStatus.SLIMED)]) == 10


# ---------------------------------------------------------------------------
# test_strangled_kills_at_zero
# ---------------------------------------------------------------------------

class TestStrangledKillsAtZero:
    """tick_strangled_lethal fires when STRANGLED timer == 1.

    Cite: vendor/nethack/src/timeout.c lines 890-894 — done_timeout(DIED, STRANGLED).
    DeathCause.CHOKING == 1 (vendor include/hack.h CHOKED).
    """

    def test_strangled_kills_at_zero(self):
        env = _env_with_status(int(TimedStatus.STRANGLED), 1)
        env2 = tick_strangled_lethal(env)
        assert bool(env2.done) is True
        assert int(env2.player_hp) == 0
        assert int(env2.scoring.death_cause) == int(DeathCause.CHOKING)

    def test_strangled_does_not_kill_early(self):
        for turns in (2, 3, 5, 6):
            env = _env_with_status(int(TimedStatus.STRANGLED), turns)
            env2 = tick_strangled_lethal(env)
            assert bool(env2.done) is False, f"died early at turns={turns}"

    def test_apply_strangled_sets_timer_5(self):
        base = StatusState.default()
        state = apply_strangled(base)
        assert int(state.timed_statuses[int(TimedStatus.STRANGLED)]) == 5


# ---------------------------------------------------------------------------
# test_strangulation_amulet_applies
# ---------------------------------------------------------------------------

class TestStrangulationAmuletApplies:
    """Wearing a cursed AMULET_OF_STRANGULATION sets the STRANGLED timer.

    Cite: vendor/nethack/src/end.c — cursed amulet of strangulation triggers
    the strangulation death path.
    items_jewelry.py::_AMULET_TO_TIMED maps STRANGULATION → (STRANGLED, 6).
    """

    def test_strangulation_amulet_applies_strangled(self):
        BUC_CURSED = 1
        state = _env_with_amulet(AmuletEffect.STRANGULATION, buc=BUC_CURSED)
        state2 = wear_amulet(state, _RNG, slot_idx=0)
        timer = int(state2.status.timed_statuses[int(TimedStatus.STRANGLED)])
        assert timer > 0, "STRANGLED timer should be set after wearing cursed strangulation amulet"

    def test_strangulation_timer_value(self):
        """Vendor default is 6 turns per _AMULET_TO_TIMED mapping."""
        BUC_CURSED = 1
        state = _env_with_amulet(AmuletEffect.STRANGULATION, buc=BUC_CURSED)
        state2 = wear_amulet(state, _RNG, slot_idx=0)
        timer = int(state2.status.timed_statuses[int(TimedStatus.STRANGLED)])
        assert timer == 6


# ---------------------------------------------------------------------------
# test_glib_drops_weapon_sometimes
# ---------------------------------------------------------------------------

class TestGlibDropsWeaponSometimes:
    """GLIB=10, tick 100 times — wielded must become -1 at least once.

    Cite: vendor/nethack/src/status.c::glibs — each turn rn2(20)==0 drops
    the wielded weapon; expected drop frequency ~5 per 100 turns.
    """

    def test_glib_drops_weapon_sometimes(self):
        state = EnvState.default(_RNG)

        # Set GLIB timer and a wielded weapon (slot 0).
        ts = state.status.timed_statuses.at[int(TimedStatus.GLIB)].set(jnp.int32(10))
        new_status = state.status.replace(timed_statuses=ts)
        state = state.replace(
            status=new_status,
            inventory=state.inventory.replace(wielded=jnp.int8(0)),
        )

        dropped_at_least_once = False
        for i in range(100):
            rng = jax.random.PRNGKey(i + 42)
            result = tick_glib(state, rng)
            if int(result.inventory.wielded) == -1:
                dropped_at_least_once = True
                break

        assert dropped_at_least_once, (
            "GLIB should drop the wielded weapon at least once in 100 trials "
            "(expected ~5 drops; probability of 0 drops ≈ (19/20)^100 < 1%)"
        )

    def test_glib_inactive_does_not_drop(self):
        """Timer == 0: no drops regardless of roll."""
        state = EnvState.default(_RNG)
        state = state.replace(
            inventory=state.inventory.replace(wielded=jnp.int8(0)),
        )
        # GLIB timer is 0 by default in StatusState.default()
        for i in range(50):
            rng = jax.random.PRNGKey(i)
            result = tick_glib(state, rng)
            assert int(result.inventory.wielded) == 0, "inactive GLIB must not drop weapon"

    def test_glib_no_weapon_no_crash(self):
        """wielded == -1 with active GLIB should not raise."""
        state = EnvState.default(_RNG)
        ts = state.status.timed_statuses.at[int(TimedStatus.GLIB)].set(jnp.int32(5))
        new_status = state.status.replace(timed_statuses=ts)
        state = state.replace(status=new_status)
        # wielded defaults to -1
        rng = jax.random.PRNGKey(99)
        result = tick_glib(state, rng)
        assert int(result.inventory.wielded) == -1
