"""Tests for symbolic_obs and text_obs implementations."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.obs.symbolic_obs import build_symbolic_observation, SYMBOLIC_OBS_DIM
from Nethax.nethax.obs.text_obs import build_text_observation

_RNG = jax.random.PRNGKey(0)


def _default_state() -> EnvState:
    return EnvState.default(rng=_RNG)


def test_symbolic_obs_shape_and_dtype():
    state = _default_state()
    obs = build_symbolic_observation(state)
    assert obs.shape == (SYMBOLIC_OBS_DIM,), f"expected ({SYMBOLIC_OBS_DIM},), got {obs.shape}"
    assert obs.dtype == jnp.float32, f"expected float32, got {obs.dtype}"


def test_symbolic_includes_hp():
    state = _default_state()
    # Replace player_hp with 42 via flax struct replace
    state = state.replace(player_hp=jnp.int32(42))
    obs = build_symbolic_observation(state)
    # Index 0 is player_hp
    assert float(obs[0]) == 42.0, f"expected hp=42 at index 0, got {float(obs[0])}"


def test_text_obs_24x80():
    state = _default_state()
    obs = build_text_observation(state)
    assert obs.shape == (24, 80), f"expected (24, 80), got {obs.shape}"
    assert obs.dtype == jnp.uint8, f"expected uint8, got {obs.dtype}"


def test_text_obs_status_row():
    """Row 23 (status line 2) contains HP: per botl.c do_statusline2."""
    state = _default_state()
    obs = build_text_observation(state)
    row23_bytes = bytes(int(b) for b in obs[23])
    row23_text = row23_bytes.decode("ascii", errors="replace")
    assert "HP:" in row23_text, f"'HP:' not found in row 23: {row23_text!r}"
