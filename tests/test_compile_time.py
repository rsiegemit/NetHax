"""Wave 8: assert that env._step_jit cold-compile stays within budget.

The Wave 8 structural refactor (action_dispatch: 43 → 29 compact slots with
shared move/run handlers) brought CPU compile time down from ~85s to ~10s
for a single jit, ~22s for vmap, ~9s for a 16-step lax.scan.

This test enforces a 5-minute (300 s) ceiling on the *cold* compile of
``jax.jit(_step_impl)`` plus a vmap variant.  300 s is the user-requested
hard cap; if a future change pushes us back over this floor the test
fails loudly.  Local-dev compile times should remain in the tens of
seconds — the test is a safety net, not a perf target.
"""
from __future__ import annotations

import os
import time

import pytest

# Force CPU before importing JAX (mirrors conftest.py).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from Nethax.nethax.env import NethaxEnv, _step_impl


# Hard ceiling (s).  User-stated requirement: cold compile ≤ 5 min on CPU.
_BUDGET_S = 300.0


def _build_state():
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    return state


def _measure(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out[0] if isinstance(out, tuple) else out)
    return time.perf_counter() - t0


@pytest.mark.timeout(360)
def test_step_impl_jit_compile_within_5min():
    """``jax.jit(_step_impl)`` cold-compile must finish within 300 s on CPU."""
    state = _build_state()
    fresh_jit = jax.jit(_step_impl)
    elapsed = _measure(fresh_jit, state, jnp.int32(0), jax.random.PRNGKey(1))
    assert elapsed <= _BUDGET_S, (
        f"jit(_step_impl) cold compile took {elapsed:.1f}s (> {_BUDGET_S:.0f}s budget). "
        "Wave 8 structural refactor regression?"
    )


@pytest.mark.timeout(360)
def test_step_impl_vmap_compile_within_5min():
    """``jax.jit(jax.vmap(_step_impl))`` cold-compile must finish within 300 s."""
    state = _build_state()
    batch = 4
    fresh = jax.jit(jax.vmap(_step_impl))
    batched_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (batch,) + x.shape), state
    )
    actions = jnp.zeros((batch,), dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(2), batch)
    elapsed = _measure(fresh, batched_state, actions, keys)
    assert elapsed <= _BUDGET_S, (
        f"jit(vmap(_step_impl)) batch={batch} cold compile took {elapsed:.1f}s "
        f"(> {_BUDGET_S:.0f}s budget). Wave 8 structural refactor regression?"
    )


@pytest.mark.timeout(360)
def test_step_impl_scan_compile_within_5min():
    """``jit(scan(_step_impl, length=16))`` cold-compile must finish within 300 s.

    This is the PPO-style rollout shape that was previously unbounded on GPU
    (>30 min, killed).  Post-Wave-8 the same shape compiles in seconds on CPU.
    """
    state = _build_state()
    length = 16

    def _body(carry, _):
        s, k = carry
        k, sub = jax.random.split(k)
        ns, _o, _r, _d = _step_impl(s, jnp.int32(0), sub)
        return (ns, k), None

    def _rollout(s, k):
        (final_s, _), _ = jax.lax.scan(_body, (s, k), None, length=length)
        return final_s

    fn = jax.jit(_rollout)
    elapsed = _measure(fn, state, jax.random.PRNGKey(3))
    assert elapsed <= _BUDGET_S, (
        f"jit(scan(_step_impl, length={length})) cold compile took {elapsed:.1f}s "
        f"(> {_BUDGET_S:.0f}s budget). Wave 8 structural refactor regression?"
    )
