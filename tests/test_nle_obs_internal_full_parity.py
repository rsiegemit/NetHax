"""Wave 13 — internal[9] and program_state[6] full vendor parity tests.

Vendor source: vendor/nle/win/rl/winrl.cc::fill_obs (lines 262-287).

internal layout (9 slots):
    [0] deepest_lev_reached
    [1] in_yn_function        (always 0)
    [2] in_getlin             (always 0)
    [3] xwaitingforspace      (always 0)
    [4] stairs_down
    [5] 0 (legacy core seed)
    [6] 0 (legacy disp seed)
    [7] u.uhunger
    [8] u.urexp

program_state layout (6 slots, winrl.cc:263-268):
    [0] gameover               (state.done)
    [1] panicking              (always 0)
    [2] exiting                (state.done)
    [3] in_moveloop            (always 1)
    [4] in_impossible          (always 0)
    [5] something_worth_saving (always 1)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from Nethax.nethax.obs.nle_obs import build_internal, build_program_state


@pytest.fixture(scope="module")
def base_state():
    from Nethax.nethax.env import NethaxEnv
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    return state


# ---------------------------------------------------------------------------
# program_state tests
# ---------------------------------------------------------------------------

def test_program_state_in_moveloop(base_state):
    """program_state[3] == 1 always (winrl.cc:266 program_state.in_moveloop)."""
    ps = np.array(build_program_state(base_state))
    assert int(ps[3]) == 1


def test_program_state_something_worth_saving(base_state):
    """program_state[5] == 1 always after reset (winrl.cc:268)."""
    ps = np.array(build_program_state(base_state))
    assert int(ps[5]) == 1


def test_program_state_gameover_zero_when_running(base_state):
    """program_state[0] == 0 while game is running (winrl.cc:263 gameover)."""
    ps = np.array(build_program_state(base_state))
    assert int(ps[0]) == 0


def test_program_state_stopped_on_done(base_state):
    """When state.done=True, program_state[2] (exiting) == 1."""
    done_state = base_state.replace(done=jnp.bool_(True))
    ps = np.array(build_program_state(done_state))
    assert int(ps[2]) == 1


def test_program_state_gameover_on_done(base_state):
    """When state.done=True, program_state[0] (gameover) == 1."""
    done_state = base_state.replace(done=jnp.bool_(True))
    ps = np.array(build_program_state(done_state))
    assert int(ps[0]) == 1


def test_program_state_panicking_always_zero(base_state):
    """program_state[1] == 0 always (no panic in nethax)."""
    ps = np.array(build_program_state(base_state))
    assert int(ps[1]) == 0


def test_program_state_in_impossible_always_zero(base_state):
    """program_state[4] == 0 always (winrl.cc:267)."""
    ps = np.array(build_program_state(base_state))
    assert int(ps[4]) == 0


# ---------------------------------------------------------------------------
# internal tests
# ---------------------------------------------------------------------------

def test_internal_deepest_lev(base_state):
    """internal[0] == deepest_level when explicitly set (winrl.cc:278)."""
    state2 = base_state.replace(
        scoring=base_state.scoring.replace(deepest_level=jnp.int32(10))
    )
    internal = np.array(build_internal(state2))
    assert int(internal[0]) == 10


def test_internal_hunger_raw(base_state):
    """internal[7] == raw nutrition counter (winrl.cc:285 u.uhunger)."""
    state2 = base_state.replace(
        status=base_state.status.replace(nutrition=jnp.int32(500))
    )
    internal = np.array(build_internal(state2))
    assert int(internal[7]) == 500


def test_internal_hunger_state(base_state):
    """internal slots 1-6 are not hunger_state enum; [7] is the raw counter.

    Verifies the raw counter is NOT in [1..6] range when nutrition=500,
    confirming we're not confusing uhunger with uhs enum (winrl.cc:285).
    """
    state2 = base_state.replace(
        status=base_state.status.replace(nutrition=jnp.int32(500))
    )
    internal = np.array(build_internal(state2))
    # [7] should be 500 (raw counter), not a small enum value like 2
    assert int(internal[7]) == 500
    assert int(internal[7]) > 6  # definitely not an enum


def test_internal_yn_function_zero(base_state):
    """internal[1] == 0 always (no y/n prompts in nethax, winrl.cc:279)."""
    internal = np.array(build_internal(base_state))
    assert int(internal[1]) == 0


def test_internal_getlin_zero(base_state):
    """internal[2] == 0 always (no text prompts in nethax, winrl.cc:280)."""
    internal = np.array(build_internal(base_state))
    assert int(internal[2]) == 0


def test_internal_legacy_seeds_zero(base_state):
    """internal[5] and [6] == 0 (legacy seed slots, winrl.cc:283-284)."""
    internal = np.array(build_internal(base_state))
    assert int(internal[5]) == 0
    assert int(internal[6]) == 0


def test_internal_quest_stage(base_state):
    """internal[11] would be quest stage — not present in vendor internal[9].

    The vendor internal array has only 9 slots. This test documents that
    quest stage is NOT in the NLE internal vector (winrl.cc:278-287).
    Shape must be (9,).
    """
    internal = np.array(build_internal(base_state))
    assert internal.shape == (9,), (
        f"internal must be shape (9,) per winrl.cc, got {internal.shape}"
    )


def test_internal_alignment_record(base_state):
    """internal has no alignment_record slot — vendor internal is (9,).

    Documents that alignment_record is not in NLE internal (winrl.cc:278-287).
    """
    internal = np.array(build_internal(base_state))
    assert internal.shape == (9,)


def test_internal_swallowed(base_state):
    """internal has no swallowed slot — vendor internal is (9,).

    Documents that uswallow is not in NLE internal (winrl.cc:278-287).
    """
    internal = np.array(build_internal(base_state))
    assert internal.shape == (9,)
