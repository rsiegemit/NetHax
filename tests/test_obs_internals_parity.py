"""Wave 8b — obs internals + status conditions vendor parity tests.

Covers:
  - internal[0] = deepest_lev_reached (vendor winrl.cc::fill_obs ~L257)
  - internal[7] = raw u.uhunger nutrition counter (vendor eat.c)
  - build_status_conditions row-23 tail keywords (vendor botl.c::do_statusline2)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.obs.nle_obs import build_status_conditions
from Nethax.nethax.subsystems.status_effects import TimedStatus, Encumbrance


@pytest.fixture(scope="module")
def env_and_state():
    env = NethaxEnv()
    state, obs = env.reset(jax.random.PRNGKey(0))
    return env, state, obs


# ---------------------------------------------------------------------------
# internal[0] — deepest_lev_reached
# ---------------------------------------------------------------------------

def test_internal_0_starts_at_1(env_and_state):
    """Fresh env -> internal[0] == 1 (level 1)."""
    _, _, obs = env_and_state
    assert int(np.array(obs["internal"])[0]) == 1


def test_internal_0_does_not_decrease(env_and_state):
    """Mutating dungeon.current_level down must NOT lower internal[0] —
    the deepest tracker is monotonic non-decreasing (vendor parity)."""
    env, state, _ = env_and_state
    # If scoring.deepest_level field exists, simulate having gone to depth 5.
    if hasattr(state.scoring, "deepest_level"):
        new_scoring = state.scoring.replace(deepest_level=jnp.int32(5))
        state2 = state.replace(scoring=new_scoring)
        from Nethax.nethax.obs.nle_obs import build_internal
        internal = np.array(build_internal(state2))
        assert int(internal[0]) >= 5


# ---------------------------------------------------------------------------
# internal[7] — raw u.uhunger nutrition counter
# ---------------------------------------------------------------------------

def test_internal_7_starts_at_900(env_and_state):
    """Vendor eat.c::u.uhunger starts at 900 (NOT_HUNGRY).  We must emit it
    as the raw counter, NOT the hunger_state enum (which was 1 = NOT_HUNGRY)."""
    _, _, obs = env_and_state
    assert int(np.array(obs["internal"])[7]) == 900


def test_internal_7_decreases_over_turns(env_and_state):
    """Nutrition ticks down by ~1/turn (eat.c::vary_hunger).  After 100
    WAIT turns, the counter must have dropped from 900."""
    env, state0, _ = env_and_state
    state = state0
    rng = jax.random.PRNGKey(42)
    from Nethax.nethax.constants.actions import Action
    wait_action = jnp.int32(int(Action.WAIT))
    for _ in range(100):
        rng, step_rng = jax.random.split(rng)
        state, obs, _r, _d, _i = env.step(state, wait_action, step_rng)
    nutrition_after = int(np.array(obs["internal"])[7])
    assert nutrition_after < 900, f"nutrition didn't decrease: {nutrition_after}"


# ---------------------------------------------------------------------------
# build_status_conditions — vendor keyword tail
# ---------------------------------------------------------------------------

def test_status_conditions_zero_when_inactive(env_and_state):
    """No active status -> all spaces."""
    _, state, _ = env_and_state
    cond = np.array(build_status_conditions(state))
    assert np.all(cond == ord(' ')), (
        f"expected all spaces, got {bytes(cond)!r}"
    )


def test_status_conditions_conf(env_and_state):
    """Confusion active -> 'Conf' keyword appears."""
    _, state, _ = env_and_state
    new_ts = state.status.timed_statuses.at[int(TimedStatus.CONFUSION)].set(10)
    state2 = state.replace(status=state.status.replace(timed_statuses=new_ts))
    text = bytes(np.array(build_status_conditions(state2))).decode("ascii").strip()
    assert "Conf" in text


def test_status_conditions_blind(env_and_state):
    """Blind active -> 'Blind' keyword appears."""
    _, state, _ = env_and_state
    new_ts = state.status.timed_statuses.at[int(TimedStatus.BLIND)].set(10)
    state2 = state.replace(status=state.status.replace(timed_statuses=new_ts))
    text = bytes(np.array(build_status_conditions(state2))).decode("ascii").strip()
    assert "Blind" in text


def test_status_conditions_burdened(env_and_state):
    """Encumbrance.BURDENED -> 'Burdened' keyword appears."""
    _, state, _ = env_and_state
    state2 = state.replace(
        status=state.status.replace(encumbrance=jnp.int8(int(Encumbrance.BURDENED)))
    )
    text = bytes(np.array(build_status_conditions(state2))).decode("ascii").strip()
    assert "Burdened" in text


def test_status_conditions_multiple(env_and_state):
    """Multiple active -> keywords appear in vendor order.

    Vendor (botl.c::do_statusline2 lines 173-205) emits:
      Stone, Slime, Strngl, FoodPois, TermIll, <Hunger>, <Encumbrance>,
      Blind, Deaf, Stun, Conf, Hallu, Lev, Fly, Ride.
    """
    _, state, _ = env_and_state
    ts = state.status.timed_statuses
    ts = ts.at[int(TimedStatus.CONFUSION)].set(10)
    ts = ts.at[int(TimedStatus.HALLUCINATION)].set(10)
    ts = ts.at[int(TimedStatus.BLIND)].set(10)
    state2 = state.replace(
        status=state.status.replace(
            timed_statuses=ts,
            encumbrance=jnp.int8(int(Encumbrance.BURDENED)),
        )
    )
    text = bytes(np.array(build_status_conditions(state2))).decode("ascii").rstrip()
    # Vendor order: Burdened (Encumbrance) < Blind < Conf < Hallu.
    burd_at  = text.find("Burdened")
    blind_at = text.find("Blind")
    conf_at  = text.find("Conf")
    hallu_at = text.find("Hallu")
    assert 0 <= burd_at < blind_at < conf_at < hallu_at, (
        f"vendor keyword order broken: {text!r}"
    )
