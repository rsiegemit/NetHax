"""Pytest configuration for nethax Wave 1 tests."""

import os

# Force JAX to use CPU so tests don't require a GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Enable 64-bit dtypes so NLE-parity int64 blstats actually allocate as int64.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402 — must come after env var is set

# Warm up JAX once at collection time.
_ = jax.numpy.zeros(1)
