"""NLE integration smoke tests for the NLECompat shim.

Where a live ``nle.nethack`` install is available, these tests compare the
shim's outputs against the real NLE bindings (vendor ground truth).  If the
shared library can't be initialized (missing nhdat, no display, etc.) the
real-NLE checks fall back to static spec comparisons against
``nle.nethack.OBSERVATION_DESC`` / ``ACTIONS`` — still vendor-sourced.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import math
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.compat.nle_shim import NLECompat
from Nethax.nethax.obs.nle_obs import (
    NLE_OBSERVATION_KEYS,
    NLE_OBSERVATION_SHAPES,
    NLE_OBSERVATION_DTYPES,
)
from Nethax.nethax.constants.actions import ACTIONS, N_ACTIONS


# ---------------------------------------------------------------------------
# Helpers to grab the vendor spec without needing a live Nethack() instance.
# ---------------------------------------------------------------------------


def _real_nle_obs_desc():
    """Return ``nle.nethack.OBSERVATION_DESC`` or None if not importable."""
    try:
        import nle.nethack as real_nle  # type: ignore
        return real_nle.OBSERVATION_DESC
    except Exception:
        return None


def _real_nle_actions():
    """Return ``tuple(int(a) for a in nle.nethack.ACTIONS)`` or None."""
    try:
        import nle.nethack as real_nle  # type: ignore
        return tuple(int(a) for a in real_nle.ACTIONS)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_nle_compat_obs_dict_keys_match_real_nle():
    """NLECompat.reset() returns a dict with the same 17 keys as vendor NLE."""
    desc = _real_nle_obs_desc()
    if desc is None:
        pytest.skip("nle.nethack not importable — vendor spec unavailable")
    real_keys = set(desc.keys())

    nh = NLECompat(seed=0)
    our_obs, _info = nh.reset()
    our_keys = set(our_obs.keys())
    assert our_keys == real_keys, (
        f"Key drift: missing={real_keys - our_keys} extra={our_keys - real_keys}"
    )


def test_nle_compat_obs_shapes_match_real_nle():
    """Per-key obs shape matches vendor OBSERVATION_DESC[key]['shape']."""
    desc = _real_nle_obs_desc()
    if desc is None:
        pytest.skip("nle.nethack not importable")

    nh = NLECompat(seed=1)
    obs, _ = nh.reset()
    for key, spec in desc.items():
        expected_shape = tuple(spec["shape"])
        got_shape = tuple(obs[key].shape)
        assert got_shape == expected_shape, (
            f"{key}: shape {got_shape} != vendor {expected_shape}"
        )


def test_nle_compat_obs_dtypes_match_real_nle():
    """Per-key obs dtype matches vendor OBSERVATION_DESC[key]['dtype']."""
    desc = _real_nle_obs_desc()
    if desc is None:
        pytest.skip("nle.nethack not importable")

    nh = NLECompat(seed=2)
    obs, _ = nh.reset()
    for key, spec in desc.items():
        expected_dtype = np.dtype(spec["dtype"])
        got_dtype = np.dtype(jnp.dtype(obs[key].dtype))
        assert got_dtype == expected_dtype, (
            f"{key}: dtype {got_dtype} != vendor {expected_dtype}"
        )


def test_nle_compat_action_set_matches():
    """NLECompat.actions equals vendor nle.nethack.ACTIONS (or our local copy)."""
    real_actions = _real_nle_actions()
    nh = NLECompat(seed=3)
    if real_actions is not None:
        assert nh.actions == real_actions, (
            f"Action set drift: ours[:5]={nh.actions[:5]} vendor[:5]={real_actions[:5]}"
        )
    # Always assert against our pinned ACTIONS constant.
    expected = tuple(int(a) for a in ACTIONS)
    assert nh.actions == expected
    assert len(nh.actions) == N_ACTIONS == 121
    assert nh.action_set == nh.actions


def test_nle_compat_step_returns_5_tuple():
    """Gym 0.26+ API: step returns (obs, reward, terminated, truncated, info)."""
    nh = NLECompat(seed=4)
    nh.reset()
    out = nh.step(ord("."))  # "." = MiscDirection.WAIT
    assert isinstance(out, tuple)
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(obs, dict)
    assert set(obs.keys()) == set(NLE_OBSERVATION_KEYS)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert math.isfinite(reward), f"Non-finite reward: {reward}"


def test_nle_compat_reset_returns_obs_info_tuple():
    """Gym 0.26+ API: reset returns (obs, info)."""
    nh = NLECompat(seed=5)
    out = nh.reset()
    assert isinstance(out, tuple)
    assert len(out) == 2
    obs, info = out
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    # Per-key shapes/dtypes match our pinned spec.
    for key in NLE_OBSERVATION_KEYS:
        assert tuple(obs[key].shape) == NLE_OBSERVATION_SHAPES[key], (
            f"{key}: shape mismatch"
        )
        assert jnp.dtype(obs[key].dtype) == jnp.dtype(NLE_OBSERVATION_DTYPES[key]), (
            f"{key}: dtype mismatch"
        )


def test_nle_compat_100_step_smoke():
    """Run 100 random steps, verify no exceptions, no NaN/Inf, obs keys stable."""
    nh = NLECompat(seed=42)
    obs, _ = nh.reset()
    rng = np.random.default_rng(123)
    for i in range(100):
        action = int(rng.integers(0, N_ACTIONS))
        obs, reward, terminated, truncated, info = nh.step(action)
        assert isinstance(obs, dict)
        assert set(obs.keys()) == set(NLE_OBSERVATION_KEYS), (
            f"Step {i}: key drift -> {set(obs.keys())}"
        )
        assert math.isfinite(reward), f"Step {i}: non-finite reward {reward}"
        # Numerical sanity: glyphs are bounded ints, blstats are finite.
        glyphs = np.asarray(obs["glyphs"])
        assert glyphs.dtype == np.int16
        assert np.all(glyphs >= 0) and np.all(glyphs <= 5976), (
            f"Step {i}: glyphs out of [0,5976]"
        )
        blstats = np.asarray(obs["blstats"])
        assert np.all(np.isfinite(blstats)), f"Step {i}: blstats not finite"
        # Reset semantics: terminated allowed but shouldn't cause crash.
        if terminated:
            obs, _ = nh.reset()


def test_nle_compat_observation_space_is_dict():
    """observation_space is a gymnasium.spaces.Dict with 17 NLE keys."""
    pytest.importorskip("gymnasium")
    from gymnasium import spaces

    nh = NLECompat(seed=6)
    space = nh.observation_space
    assert isinstance(space, spaces.Dict), f"Expected Dict space, got {type(space)}"
    assert set(space.spaces.keys()) == set(NLE_OBSERVATION_KEYS)
    # Each entry is a Box with the right shape.
    for key in NLE_OBSERVATION_KEYS:
        sub = space.spaces[key]
        assert isinstance(sub, spaces.Box), f"{key}: expected Box, got {type(sub)}"
        assert tuple(sub.shape) == NLE_OBSERVATION_SHAPES[key]


def test_nle_compat_action_space_is_discrete_121():
    """action_space is gymnasium.spaces.Discrete(121)."""
    pytest.importorskip("gymnasium")
    from gymnasium import spaces

    nh = NLECompat(seed=7)
    space = nh.action_space
    assert isinstance(space, spaces.Discrete)
    assert int(space.n) == 121


def test_nle_compat_glyph_helpers():
    """Glyph predicate helpers cover the canonical NLE bands."""
    from Nethax.nethax.constants.glyphs import (
        GLYPH_MON_OFF, GLYPH_PET_OFF, GLYPH_INVIS_OFF,
        GLYPH_BODY_OFF, GLYPH_OBJ_OFF, GLYPH_CMAP_OFF,
        GLYPH_STATUE_OFF,
    )

    # Monsters
    assert NLECompat.nethack_glyph_is_monster(GLYPH_MON_OFF + 0)
    assert NLECompat.nethack_glyph_is_monster(GLYPH_PET_OFF + 0)
    assert NLECompat.nethack_glyph_is_normal_monster(GLYPH_MON_OFF + 0)
    assert not NLECompat.nethack_glyph_is_normal_monster(GLYPH_PET_OFF + 0)
    assert NLECompat.nethack_glyph_is_pet(GLYPH_PET_OFF + 5)

    # Invisible
    assert NLECompat.nethack_glyph_is_invisible(GLYPH_INVIS_OFF)
    assert not NLECompat.nethack_glyph_is_invisible(GLYPH_INVIS_OFF + 1)

    # Bodies (corpses)
    assert NLECompat.nethack_glyph_is_body(GLYPH_BODY_OFF + 0)
    assert not NLECompat.nethack_glyph_is_body(GLYPH_MON_OFF + 0)

    # Objects
    assert NLECompat.nethack_glyph_is_object(GLYPH_OBJ_OFF + 0)
    assert not NLECompat.nethack_glyph_is_object(GLYPH_MON_OFF + 0)

    # Cmap (terrain)
    assert NLECompat.nethack_glyph_is_cmap(GLYPH_CMAP_OFF + 0)
    assert not NLECompat.nethack_glyph_is_cmap(GLYPH_OBJ_OFF + 0)

    # Statue
    assert NLECompat.nethack_glyph_is_statue(GLYPH_STATUE_OFF + 0)
    assert not NLECompat.nethack_glyph_is_statue(GLYPH_STATUE_OFF - 1)


def test_nle_compat_character_string_parsing():
    """The character arg "mon-hum-neu-mal" resolves to Role.MONK / Race.HUMAN / neutral."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.constants.races import Race

    nh = NLECompat(seed=8, character="mon-hum-neu-mal")
    assert nh._role == Role.MONK
    assert nh._race == Race.HUMAN
    assert nh._alignment == 1  # neutral

    # Default character also lands somewhere sane.
    nh2 = NLECompat(seed=9)
    assert nh2._role is not None  # parsed from default "mon-hum-neu-mal"

    # Wildcard character produces (None, None, default-alignment).
    nh3 = NLECompat(seed=10, character="@")
    assert nh3._role is None
    assert nh3._race is None
