"""Status-effects parity tests: confused direction, confuse-attack, poisoned,
sick-lethal, wounded-legs limp.

Vendor citations:
  hack.c:2424   — confused/stunned direction randomization
  spell.c       — SPE_CONFUSE_MONSTER confuse_attack_pending
  status.c      — POISONED HP tick, SICK lethal-on-zero
  hack.c        — WOUNDED_LEGS movement penalty
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
    apply_poisoned_tick,
    apply_sick_lethal,
    step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_status(**overrides) -> StatusState:
    base = StatusState.default()
    if not overrides:
        return base
    ts = base.timed_statuses
    for k, v in overrides.items():
        if k == "confuse_attack_pending":
            base = base.replace(confuse_attack_pending=jnp.bool_(v))
        else:
            idx = int(getattr(TimedStatus, k))
            ts = ts.at[idx].set(jnp.int32(v))
    base = base.replace(timed_statuses=ts)
    return base


def _run_step(status, player_hp, player_pw, done, rng_seed=0):
    rng = jax.random.PRNGKey(rng_seed)
    return step(
        status,
        rng,
        player_hp=jnp.int32(player_hp),
        player_hp_max=jnp.int32(100),
        player_pw=jnp.int32(player_pw),
        player_pw_max=jnp.int32(100),
        player_xl=jnp.int32(1),
        player_role=jnp.int8(0),
        done=jnp.bool_(done),
    )


# ---------------------------------------------------------------------------
# 1. Confused/stunned direction randomization
#    We cannot test action_dispatch in isolation without a full EnvState, so
#    we verify that the TimedStatus.CONFUSION slot exists and is readable —
#    the actual randomization is exercised in integration tests.  Here we
#    test the status field exists and can be set.
# ---------------------------------------------------------------------------

def test_confused_status_field_exists():
    """CONFUSION TimedStatus slot is accessible and settable."""
    s = _make_status(CONFUSION=10)
    assert int(s.timed_statuses[int(TimedStatus.CONFUSION)]) == 10


def test_stunned_status_field_exists():
    """STUNNED TimedStatus slot is accessible and settable."""
    s = _make_status(STUNNED=5)
    assert int(s.timed_statuses[int(TimedStatus.STUNNED)]) == 5


# ---------------------------------------------------------------------------
# 2. Confuse-attack-pending field
# ---------------------------------------------------------------------------

def test_confuse_attack_pending_default_false():
    """confuse_attack_pending starts False."""
    s = StatusState.default()
    assert not bool(s.confuse_attack_pending)


def test_confuse_attack_pending_settable():
    """confuse_attack_pending can be set to True."""
    s = _make_status(confuse_attack_pending=True)
    assert bool(s.confuse_attack_pending)


# ---------------------------------------------------------------------------
# 3. POISONED ticks damage
#    Cite: vendor/nethack/src/status.c POISONED tick.
# ---------------------------------------------------------------------------

def test_poisoned_ticks_damage():
    """POISONED active → 1 HP drained per step."""
    s = _make_status(POISONED=10)
    hp = jnp.int32(50)
    # Run 5 turns; each should drain 1 HP while POISONED > 0.
    for i in range(5):
        s, hp, _pw, _done = _run_step(s, int(hp), 10, False, rng_seed=i)
    # After 5 turns, HP should be less than 50 (at most 50 - 5 = 45).
    # Hunger tick drains no HP (nutrition stays in NOT_HUNGRY range), so
    # only POISONED reduces it.
    assert int(hp) <= 45


def test_poisoned_no_damage_when_inactive():
    """No POISONED status → HP unchanged by poisoned tick."""
    s = StatusState.default()
    hp = jnp.int32(50)
    _, new_hp = apply_poisoned_tick(s, hp)
    assert int(new_hp) == 50


def test_poisoned_drains_one_per_call():
    """apply_poisoned_tick drains exactly 1 HP when POISONED > 0."""
    s = _make_status(POISONED=5)
    hp = jnp.int32(30)
    _, new_hp = apply_poisoned_tick(s, hp)
    assert int(new_hp) == 29


def test_poisoned_clamps_at_zero():
    """apply_poisoned_tick does not go below 0 HP."""
    s = _make_status(POISONED=5)
    hp = jnp.int32(0)
    _, new_hp = apply_poisoned_tick(s, hp)
    assert int(new_hp) == 0


# ---------------------------------------------------------------------------
# 4. SICK (illness) kills at timer zero
#    Cite: vendor/nethack/src/status.c::sick lethal-on-zero.
# ---------------------------------------------------------------------------

def _make_sick_status(timer: int, kind: int) -> StatusState:
    base = StatusState.default()
    ts = base.timed_statuses.at[int(TimedStatus.SICK)].set(jnp.int32(timer))
    return base.replace(timed_statuses=ts, sick_kind=jnp.int8(kind))


def test_sick_kills_at_zero_illness():
    """SICK (sick_kind=2, illness) at timer=1 fires death on apply_sick_lethal."""
    s = _make_sick_status(1, kind=2)
    hp = jnp.int32(20)
    done = jnp.bool_(False)
    _, new_hp, new_done = apply_sick_lethal(s, hp, done)
    assert int(new_hp) == 0
    assert bool(new_done)


def test_sick_kills_via_step():
    """SICK illness timer=2 → after 2 step() turns, done=True."""
    s = _make_sick_status(2, kind=2)
    hp = jnp.int32(20)
    done = jnp.bool_(False)
    # Turn 1: timer=2 → not expiring yet (apply_sick_lethal checks == 1).
    s, hp, _pw, done = _run_step(s, int(hp), 10, bool(done), rng_seed=0)
    assert not bool(done), "should not die yet on turn 1 (timer was 2)"
    # Turn 2: timer decremented to 1 → lethal expiry fires.
    s, hp, _pw, done = _run_step(s, int(hp), 10, bool(done), rng_seed=1)
    assert bool(done), "should die when SICK timer expires"


def test_sick_food_poison_not_killed_by_illness_path():
    """SICK (sick_kind=1, food-poisoning) is NOT killed by apply_sick_lethal."""
    s = _make_sick_status(1, kind=1)
    hp = jnp.int32(20)
    done = jnp.bool_(False)
    _, new_hp, new_done = apply_sick_lethal(s, hp, done)
    # apply_sick_lethal only fires for kind==2; food-poisoning uses apply_food_poisoning.
    assert int(new_hp) == 20
    assert not bool(new_done)


# ---------------------------------------------------------------------------
# 5. WOUNDED_LEGS skips move with ~30% probability
#    Tested at the status level: the timer is set/readable.
#    The actual skip is in action_dispatch._try_step (integration test scope).
# ---------------------------------------------------------------------------

def test_wounded_legs_status_field_exists():
    """WOUNDED_LEGS TimedStatus slot is accessible and settable."""
    s = _make_status(WOUNDED_LEGS=10)
    assert int(s.timed_statuses[int(TimedStatus.WOUNDED_LEGS)]) == 10


def test_wounded_legs_ticks_down():
    """WOUNDED_LEGS timer decrements each step."""
    s = _make_status(WOUNDED_LEGS=5)
    hp = jnp.int32(50)
    s, hp, _pw, _done = _run_step(s, int(hp), 10, False, rng_seed=0)
    # After one step the timer should be 4 (or less if it ticked multiple times,
    # but tick_timers decrements by 1).
    assert int(s.timed_statuses[int(TimedStatus.WOUNDED_LEGS)]) == 4


# ---------------------------------------------------------------------------
# 6. VOMITING status exists and is readable
# ---------------------------------------------------------------------------

def test_vomiting_status_field_exists():
    """VOMITING TimedStatus slot is accessible and settable."""
    s = _make_status(VOMITING=3)
    assert int(s.timed_statuses[int(TimedStatus.VOMITING)]) == 3


# ---------------------------------------------------------------------------
# 7. N_TIMED_STATUSES includes new POISONED slot
# ---------------------------------------------------------------------------

def test_n_timed_statuses_includes_poisoned():
    """N_TIMED_STATUSES == 25 (includes new POISONED slot at index 24)."""
    assert N_TIMED_STATUSES == 25
    assert int(TimedStatus.POISONED) == 24
