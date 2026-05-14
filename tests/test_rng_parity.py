"""Wave 6 closing-audit: parity tests for Nethax/nethax/rng.py vs
vendor/nethack/src/rnd.c.

Vendor semantics (from rnd.c):
  rn2(N)   -> 0 <= result < N   (RND(N) = Rand() % N)
  rn1(N,x) -> x + rn2(N)  -> x <= result <= x + N - 1
  rnd(N)   -> 1 <= result <= N  (RND(N) + 1)
  d(n, N)  -> n <= result <= n*N  (sum of n rolls of [1..N])
  d(0, N)  -> 0   (loop body never executes; tmp stays 0)

Edge cases enforced:
  rn2(1) must always return 0 (mod-1 arithmetic).
  rnd(1) must always return 1.
  d(0, *) must return 0.

Bit-identical seed reproducibility is NOT a goal (we use jax.random Threefry,
vendor uses ISAAC-64).  We assert distributional and range semantics only.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.rng import dice_roll, rn1, rn2, rnd, split_n


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _key(seed: int = 0) -> jax.Array:
    return jax.random.PRNGKey(seed)


_N_SAMPLES_RANGE = 500
_N_SAMPLES_DIST = 1000


# ---------------------------------------------------------------------------
# rn2
# ---------------------------------------------------------------------------

def test_rn2_returns_0_to_n_minus_1():
    """rn2(N) must yield values in [0, N-1] over many trials."""
    keys = split_n(_key(0), _N_SAMPLES_RANGE)
    for n in (2, 5, 10, 100):
        results = jax.vmap(lambda k, nn=n: rn2(k, nn))(keys)
        assert int(jnp.min(results)) >= 0, f"rn2({n}) min < 0"
        assert int(jnp.max(results)) <= n - 1, f"rn2({n}) max > n-1"


def test_rn2_1_returns_zero():
    """Edge case: rn2(1) must always return 0 (vendor's mod-1 arithmetic)."""
    keys = split_n(_key(123), 200)
    results = jax.vmap(lambda k: rn2(k, 1))(keys)
    assert int(jnp.max(results)) == 0, "rn2(1) must always return 0."
    assert int(jnp.min(results)) == 0, "rn2(1) must always return 0."


# ---------------------------------------------------------------------------
# rnd
# ---------------------------------------------------------------------------

def test_rnd_returns_1_to_n():
    """rnd(N) must yield values in [1, N]."""
    keys = split_n(_key(11), _N_SAMPLES_RANGE)
    for n in (1, 2, 6, 20):
        results = jax.vmap(lambda k, nn=n: rnd(k, nn))(keys)
        assert int(jnp.min(results)) >= 1, f"rnd({n}) min < 1"
        assert int(jnp.max(results)) <= n, f"rnd({n}) max > n"

    # rnd(1) must always be 1.
    keys = split_n(_key(7), 100)
    ones = jax.vmap(lambda k: rnd(k, 1))(keys)
    assert int(jnp.min(ones)) == 1 and int(jnp.max(ones)) == 1


# ---------------------------------------------------------------------------
# d (dice_roll)
# ---------------------------------------------------------------------------

def test_d_returns_n_to_n_times_N():
    """d(n, N) must yield values in [n, n*N]."""
    keys = split_n(_key(31), _N_SAMPLES_RANGE)
    for n, sides in [(1, 6), (2, 6), (3, 8), (4, 4)]:
        results = jax.vmap(lambda k, nn=n, ss=sides: dice_roll(k, nn, ss))(keys)
        assert int(jnp.min(results)) >= n, f"d({n},{sides}) min < n"
        assert int(jnp.max(results)) <= n * sides, f"d({n},{sides}) max > n*N"


def test_d_zero_dice_returns_zero():
    """Vendor d(0, N): tmp stays 0, returns 0."""
    for seed in range(20):
        result = dice_roll(_key(seed), 0, 6)
        assert int(result) == 0, "d(0, N) must return 0 (no dice rolled)."


# ---------------------------------------------------------------------------
# rn1
# ---------------------------------------------------------------------------

def test_rn1_offset_correct():
    """rn1(N, x) must yield values in [x, x+N-1]."""
    keys = split_n(_key(53), _N_SAMPLES_RANGE)
    for n, x in [(5, 10), (3, -2), (8, 0), (1, 99)]:
        results = jax.vmap(lambda k, nn=n, xx=x: rn1(k, nn, xx))(keys)
        assert int(jnp.min(results)) >= x, f"rn1({n},{x}) min < x"
        assert int(jnp.max(results)) <= x + n - 1, f"rn1({n},{x}) max > x+n-1"


def test_rn1_n_equals_1_is_constant_x():
    """rn1(1, x) must always return x (since rn2(1) is always 0)."""
    keys = split_n(_key(91), 100)
    results = jax.vmap(lambda k: rn1(k, 1, 42))(keys)
    assert int(jnp.min(results)) == 42 and int(jnp.max(results)) == 42


# ---------------------------------------------------------------------------
# Distribution checks
# ---------------------------------------------------------------------------

def test_d_distribution_uniform_for_d1():
    """d(1, 6) should be uniform over {1,...,6} (expected freq ~ 1/6 each)."""
    keys = split_n(_key(2024), _N_SAMPLES_DIST)
    results = jax.vmap(lambda k: dice_roll(k, 1, 6))(keys)
    # All 6 faces should appear; each should be ~16.7% of samples.
    for face in range(1, 7):
        frac = float(jnp.mean(results == face))
        assert 0.10 < frac < 0.24, f"d(1,6) face {face} frequency {frac:.3f} too skewed."


def test_d_2d6_mean_7():
    """d(2, 6) expected mean is 7.0 (uniform sum)."""
    keys = split_n(_key(4096), _N_SAMPLES_DIST)
    results = jax.vmap(lambda k: dice_roll(k, 2, 6))(keys)
    mean = float(jnp.mean(results.astype(jnp.float32)))
    # 1000 samples of 2d6 (variance ~5.83) — 99% CI is roughly +/- 0.2.
    assert abs(mean - 7.0) < 0.4, f"d(2,6) mean {mean:.3f} far from 7.0"


def test_rn2_distribution_uniform():
    """rn2(10) should be approximately uniform over {0,...,9}."""
    keys = split_n(_key(2025), _N_SAMPLES_DIST)
    results = jax.vmap(lambda k: rn2(k, 10))(keys)
    for face in range(10):
        frac = float(jnp.mean(results == face))
        # Expected 0.10; allow [0.06, 0.16] over 1000 samples.
        assert 0.06 < frac < 0.16, f"rn2(10) face {face} frequency {frac:.3f} too skewed."


# ---------------------------------------------------------------------------
# JIT-safety
# ---------------------------------------------------------------------------

def test_rng_functions_jit():
    """All primitives must compile cleanly under jax.jit."""
    jit_rn2 = jax.jit(lambda k: rn2(k, 6))
    jit_rnd = jax.jit(lambda k: rnd(k, 6))
    jit_rn1 = jax.jit(lambda k: rn1(k, 5, 10))
    jit_d   = jax.jit(lambda k: dice_roll(k, 3, 6))
    k = _key(0)
    assert 0 <= int(jit_rn2(k)) < 6
    assert 1 <= int(jit_rnd(k)) <= 6
    assert 10 <= int(jit_rn1(k)) <= 14
    assert 3 <= int(jit_d(k)) <= 18
