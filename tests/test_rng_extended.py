"""Tests for extended NetHack RNG helpers (rne, rnl, rnz, rnf).

Wave 6 polish: ported vendor rnd.c helpers not covered by rng_parity CA #90.

Citations:
  vendor/nethack/src/rnd.c::rne — enchantment geometric roll
  vendor/nethack/src/rnd.c::rnl — luck-adjusted uniform
  vendor/nethack/src/rnd.c::rnz — time-scaling roll
  vendor/nethack/src/rnd.c::rnf — fractional probability check
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest
from Nethax.nethax.rng import rne, rnl, rnz, rnf


class TestRne:
    def test_rne_returns_at_least_one(self):
        """Vendor rne: tmp starts at 1, always returns >= 1."""
        for seed in range(50):
            v = int(rne(jax.random.PRNGKey(seed), 5))
            assert v >= 1, f"rne returned {v} (<1)"

    def test_rne_capped_at_cap(self):
        """Vendor caps tmp at cap=9 (default)."""
        for seed in range(50):
            v = int(rne(jax.random.PRNGKey(seed), 2))
            assert v <= 9, f"rne returned {v} (>9)"

    def test_rne_mostly_returns_1_for_high_x(self):
        """rne(x=20): chance of bumping is 1/20 per step → mean close to 1."""
        vals = [int(rne(jax.random.PRNGKey(seed), 20)) for seed in range(200)]
        mean = sum(vals) / len(vals)
        assert 1.0 <= mean <= 1.5, f"rne(20) mean={mean}; expected ~1.0-1.1"


class TestRnl:
    def test_rnl_returns_in_range_no_luck(self):
        for seed in range(50):
            v = int(rnl(jax.random.PRNGKey(seed), 20, 0))
            assert 0 <= v < 20

    def test_rnl_unbiased_at_zero_luck(self):
        """Without luck, rnl matches rn2 — uniform [0, x)."""
        vals = [int(rnl(jax.random.PRNGKey(seed), 20, 0)) for seed in range(2000)]
        mean = sum(vals) / len(vals)
        # Uniform [0, 19] mean = 9.5
        assert 8.5 <= mean <= 10.5, f"rnl mean={mean}; expected ~9.5"

    def test_rnl_clamped_in_range_with_luck(self):
        """Even with luck, rnl must stay in [0, x-1]."""
        for seed in range(50):
            v = int(rnl(jax.random.PRNGKey(seed), 20, 13))
            assert 0 <= v < 20, f"rnl returned {v} (out of [0,19])"


class TestRnz:
    def test_rnz_positive(self):
        """rnz(i) for i>0 should return positive values."""
        for seed in range(50):
            v = int(rnz(jax.random.PRNGKey(seed), 100))
            assert v >= 0, f"rnz(100) returned {v} (<0)"

    def test_rnz_scales_with_i(self):
        """rnz(1000) mean roughly > rnz(100) mean."""
        small = [int(rnz(jax.random.PRNGKey(seed), 100)) for seed in range(100)]
        large = [int(rnz(jax.random.PRNGKey(seed + 1000), 1000)) for seed in range(100)]
        assert sum(large) / 100 > sum(small) / 100 * 5, (
            "rnz(1000) should average significantly larger than rnz(100)"
        )

    def test_rnz_distribution_centered_near_input(self):
        """rnz tends to spread around the input value; median should be close to i."""
        vals = sorted(int(rnz(jax.random.PRNGKey(seed), 100)) for seed in range(500))
        median = vals[len(vals) // 2]
        # Vendor rnz is heavily skewed but should be roughly within [33, 300] for i=100.
        assert 50 <= median <= 200, f"rnz(100) median={median}; expected 50-200"


class TestRnf:
    def test_rnf_3_of_10_fires_roughly_30_percent(self):
        n_true = sum(bool(rnf(jax.random.PRNGKey(seed), 3, 10)) for seed in range(10000))
        rate = n_true / 10000
        assert 0.25 <= rate <= 0.35, f"rnf(3,10) fired {rate:.2%}; expected ~30%"

    def test_rnf_0_never_fires(self):
        for seed in range(100):
            assert not bool(rnf(jax.random.PRNGKey(seed), 0, 10))

    def test_rnf_all_always_fires(self):
        for seed in range(100):
            assert bool(rnf(jax.random.PRNGKey(seed), 10, 10))
