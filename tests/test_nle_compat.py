"""Tests for the NLE compatibility shim (``Nethax.nethax.compat.nle_shim``)."""
from __future__ import annotations

import pytest


def test_nle_compat_reset_returns_obs_info_tuple():
    """``NLECompat.reset`` returns ``(obs, info)`` per gymnasium 0.26+."""
    from Nethax.nethax.compat.nle_shim import NLECompat
    from Nethax.nethax.obs.nle_obs import NLE_OBSERVATION_KEYS

    nh = NLECompat(seed=0)
    out = nh.reset()
    assert isinstance(out, tuple)
    assert len(out) == 2
    obs, info = out
    assert isinstance(obs, dict)
    assert set(obs.keys()) == set(NLE_OBSERVATION_KEYS)
    assert len(obs) == 17
    assert isinstance(info, dict)


def test_nle_compat_step_returns_5_tuple():
    """``NLECompat.step(action)`` returns the gym-0.26+ 5-tuple."""
    from Nethax.nethax.compat.nle_shim import NLECompat
    from Nethax.nethax.obs.nle_obs import NLE_OBSERVATION_KEYS

    nh = NLECompat(seed=1)
    nh.reset()
    result = nh.step(ord("."))
    assert isinstance(result, tuple)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, dict)
    assert set(obs.keys()) == set(NLE_OBSERVATION_KEYS)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_nle_compat_action_set_matches_nle():
    """``NLECompat.actions`` is the canonical 121-tuple of NLE action ints."""
    from Nethax.nethax.compat.nle_shim import NLECompat
    from Nethax.nethax.constants.actions import ACTIONS, N_ACTIONS

    nh = NLECompat(seed=2)
    assert isinstance(nh.actions, tuple)
    assert len(nh.actions) == N_ACTIONS == 121
    assert tuple(int(a) for a in ACTIONS) == nh.actions
    # ``action_set`` is an alias.
    assert nh.action_set == nh.actions


def test_nle_compat_glyph_to_char_printable():
    """``nethack_glyph_to_char`` returns a printable char for ASCII range."""
    from Nethax.nethax.compat.nle_shim import NLECompat

    assert NLECompat.nethack_glyph_to_char(ord("@")) == "@"
    assert NLECompat.nethack_glyph_to_char(ord(".")) == "."
    # Out-of-range glyph collapses to '?'.
    assert NLECompat.nethack_glyph_to_char(5000) == "?"
