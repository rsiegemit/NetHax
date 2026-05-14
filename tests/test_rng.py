"""Wave 2 tests for Nethax/nethax/rng.py — JAX PRNG dice primitives."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.rng import dice_roll, rn2, rnd, split_n, weighted_choice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _key(seed: int = 0) -> jax.Array:
    return jax.random.PRNGKey(seed)


# ---------------------------------------------------------------------------
# dice_roll
# ---------------------------------------------------------------------------

def test_dice_roll_range():
    result = dice_roll(_key(42), 3, 6)
    assert int(result) >= 3, "3d6 minimum is 3."
    assert int(result) <= 18, "3d6 maximum is 18."


def test_dice_roll_deterministic():
    key = _key(7)
    r1 = dice_roll(key, 2, 8)
    r2 = dice_roll(key, 2, 8)
    assert int(r1) == int(r2), "Same key must yield identical result."


def test_dice_roll_changes_with_key():
    results = {int(dice_roll(_key(i), 1, 20)) for i in range(50)}
    assert len(results) > 1, "Different keys should produce different rolls."


def test_dice_roll_jits():
    jit_roll = jax.jit(lambda k: dice_roll(k, 2, 6))
    result = jit_roll(_key(0))
    assert result.shape == ()
    assert result.dtype == jnp.int32
    assert 2 <= int(result) <= 12


# ---------------------------------------------------------------------------
# rn2
# ---------------------------------------------------------------------------

def test_rn2_in_range():
    keys = split_n(_key(0), 100)
    results = jax.vmap(lambda k: rn2(k, 10))(keys)
    assert jnp.all(results >= 0), "rn2 must be >= 0."
    assert jnp.all(results < 10), "rn2 must be < n."


def test_rn2_jits():
    jit_rn2 = jax.jit(lambda k: rn2(k, 6))
    result = jit_rn2(_key(1))
    assert result.shape == ()
    assert 0 <= int(result) < 6


# ---------------------------------------------------------------------------
# rnd
# ---------------------------------------------------------------------------

def test_rnd_in_range():
    keys = split_n(_key(5), 100)
    results = jax.vmap(lambda k: rnd(k, 6))(keys)
    assert jnp.all(results >= 1), "rnd must be >= 1."
    assert jnp.all(results <= 6), "rnd must be <= n."


# ---------------------------------------------------------------------------
# weighted_choice
# ---------------------------------------------------------------------------

def test_weighted_choice_extreme_weight():
    weights = jnp.array([100.0, 0.0, 0.0])
    for seed in range(20):
        result = weighted_choice(_key(seed), weights)
        assert int(result) == 0, "Weight of 100/0/0 must always select index 0."


def test_weighted_choice_jits():
    weights = jnp.array([1.0, 2.0, 3.0])
    jit_wc = jax.jit(lambda k: weighted_choice(k, weights))
    result = jit_wc(_key(0))
    assert result.shape == ()
    assert result.dtype == jnp.int32
    assert 0 <= int(result) < 3


def test_weighted_choice_distribution():
    """Heavy weight on index 2 should dominate over many samples."""
    weights = jnp.array([1.0, 1.0, 98.0])
    keys = split_n(_key(99), 200)
    results = jax.vmap(lambda k: weighted_choice(k, weights))(keys)
    frac_idx2 = jnp.mean(results == 2)
    assert float(frac_idx2) > 0.85, "Index 2 with weight 98% should win >85% of samples."
