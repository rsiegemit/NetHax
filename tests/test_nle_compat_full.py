"""Full NLE drop-in compatibility validation for ``NLECompat``.

Goal: assert that ``Nethax.nethax.compat.nle_shim.NLECompat`` is a true
drop-in replacement for ``nle.env.NLE`` — anything an RL training script
expects from NLE should be present on our shim with byte-equivalent
shape/dtype contracts.

Tests skipped via ``pytest.importorskip("nle")`` when NLE is not present.

Citations:
    - vendor/nle/nle/env/base.py        — NLE class definition / step / reset.
    - vendor/nle/nle/nethack/nethack.py — OBSERVATION_DESC byte spec.
    - vendor/nle/nle/nethack/actions.py — 121-int ACTIONS tuple.
    - vendor/nle/include/nleobs.h       — C-level observation tensor sizes.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import math
import inspect
from typing import Any

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


# Most tests here only need vendor's static spec, not a live game; gate
# with ``importorskip`` once at module level so the entire file is skipped
# if NLE is unavailable.
nle_nethack = pytest.importorskip("nle.nethack")
nle_env_base = pytest.importorskip("nle.env.base")


# ---------------------------------------------------------------------------
# 1. Gym/Gymnasium API match
# ---------------------------------------------------------------------------


def test_inherits_from_gym_env():
    """``NLECompat`` must inherit from ``gymnasium.Env`` so wrappers work.

    Citation: vendor/nle/nle/env/base.py declares ``class NLE(gym.Env)``.
    """
    import gymnasium as gym

    nh = NLECompat(seed=0)
    assert isinstance(nh, gym.Env), (
        f"NLECompat must inherit gymnasium.Env, got bases={type(nh).__mro__}"
    )


def test_reset_signature_matches_gymnasium():
    """``reset(*, seed=None, options=None)`` per gymnasium >= 0.26."""
    sig = inspect.signature(NLECompat.reset)
    params = sig.parameters
    assert "seed" in params
    assert "options" in params
    # Both are keyword-only per gymnasium 0.26+.
    assert params["seed"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["options"].kind == inspect.Parameter.KEYWORD_ONLY


def test_reset_returns_obs_info_tuple():
    """Gymnasium 0.26+ contract: reset -> (obs, info)."""
    nh = NLECompat(seed=0)
    out = nh.reset()
    assert isinstance(out, tuple) and len(out) == 2
    obs, info = out
    assert isinstance(obs, dict)
    assert isinstance(info, dict)


def test_step_returns_5_tuple():
    """Gymnasium 0.26+ contract: step -> (obs, reward, terminated, truncated, info)."""
    nh = NLECompat(seed=1)
    nh.reset()
    out = nh.step(0)
    assert isinstance(out, tuple) and len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_close_method_present():
    """``close()`` must exist and return None (vendor parity)."""
    nh = NLECompat(seed=2)
    nh.reset()
    assert nh.close() is None


def test_seed_method_present_and_signature():
    """``seed(core, disp, reseed)`` signature matches vendor NLE.

    Citation: vendor/nle/nle/env/base.py::seed.
    """
    sig = inspect.signature(NLECompat.seed)
    names = list(sig.parameters)
    # ['self', 'core', 'disp', 'reseed']
    assert names[1:] == ["core", "disp", "reseed"], names

    nh = NLECompat(seed=3)
    out = nh.seed(42)
    assert isinstance(out, tuple) and len(out) == 3
    core, disp, reseed = out
    assert core == 42
    assert isinstance(disp, int)
    assert isinstance(reseed, bool)


def test_get_seeds_returns_triple():
    """``get_seeds()`` returns ``(core, disp, reseed)`` per vendor parity.

    Citation: vendor/nle/nle/env/base.py::get_seeds.
    """
    nh = NLECompat(seed=7)
    nh.seed(100, 200, False)
    out = nh.get_seeds()
    assert out == (100, 200, False), out


def test_render_modes_metadata_present():
    """``metadata`` must declare available render modes (gymnasium contract)."""
    nh = NLECompat(seed=4)
    assert hasattr(nh, "metadata")
    assert isinstance(nh.metadata, dict)
    # gymnasium 0.26+ key is "render_modes"
    modes = nh.metadata.get("render_modes") or nh.metadata.get("render.modes")
    assert modes is not None
    assert "human" in modes


def test_render_returns_string_for_ansi():
    """``render('ansi')`` returns a str representation of the tty frame."""
    nh = NLECompat(seed=5)
    nh.reset()
    s = nh.render("ansi")
    assert isinstance(s, str)
    assert "\n" in s  # multi-row tty frame


def test_print_action_meanings_present():
    """Vendor helper: ``print_action_meanings()`` enumerates actions."""
    nh = NLECompat(seed=6)
    assert hasattr(nh, "print_action_meanings")
    assert callable(nh.print_action_meanings)


def test_step_status_enum_present():
    """``StepStatus`` IntEnum (ABORTED=-1, RUNNING=0, DEATH=1) — vendor parity.

    Citation: vendor/nle/nle/env/base.py::NLE.StepStatus.
    """
    assert int(NLECompat.StepStatus.ABORTED) == -1
    assert int(NLECompat.StepStatus.RUNNING) == 0
    assert int(NLECompat.StepStatus.DEATH) == 1


# ---------------------------------------------------------------------------
# 2. Observation space byte-equivalence to vendor NLE
# ---------------------------------------------------------------------------


def test_obs_space_is_dict_with_17_keys():
    """``observation_space`` is ``Dict`` with 17 NLE-canonical keys."""
    import gymnasium as gym

    nh = NLECompat(seed=8)
    space = nh.observation_space
    assert isinstance(space, gym.spaces.Dict)
    assert set(space.spaces.keys()) == set(NLE_OBSERVATION_KEYS)
    assert len(space.spaces) == 17


def test_obs_space_shapes_match_vendor():
    """Per-key Box.shape matches vendor OBSERVATION_DESC."""
    nh = NLECompat(seed=9)
    space = nh.observation_space
    for key, spec in nle_nethack.OBSERVATION_DESC.items():
        expected_shape = tuple(spec["shape"])
        got_shape = tuple(space.spaces[key].shape)
        assert got_shape == expected_shape, (
            f"{key}: shape {got_shape} != vendor {expected_shape}"
        )


def test_obs_space_dtypes_match_vendor():
    """Per-key Box.dtype matches vendor OBSERVATION_DESC."""
    nh = NLECompat(seed=10)
    space = nh.observation_space
    for key, spec in nle_nethack.OBSERVATION_DESC.items():
        expected_dtype = np.dtype(spec["dtype"])
        got_dtype = np.dtype(space.spaces[key].dtype)
        assert got_dtype == expected_dtype, (
            f"{key}: dtype {got_dtype} != vendor {expected_dtype}"
        )


def test_runtime_obs_shapes_match_vendor():
    """Runtime obs (from reset) per-key shape matches vendor."""
    nh = NLECompat(seed=11)
    obs, _ = nh.reset()
    for key, spec in nle_nethack.OBSERVATION_DESC.items():
        expected_shape = tuple(spec["shape"])
        got_shape = tuple(obs[key].shape)
        assert got_shape == expected_shape, (
            f"{key}: shape {got_shape} != vendor {expected_shape}"
        )


def test_runtime_obs_dtypes_match_vendor():
    """Runtime obs per-key dtype matches vendor OBSERVATION_DESC."""
    nh = NLECompat(seed=12)
    obs, _ = nh.reset()
    for key, spec in nle_nethack.OBSERVATION_DESC.items():
        expected_dtype = np.dtype(spec["dtype"])
        got_dtype = np.dtype(jnp.dtype(obs[key].dtype))
        assert got_dtype == expected_dtype, (
            f"{key}: dtype {got_dtype} != vendor {expected_dtype}"
        )


def test_obs_keys_match_vendor_exactly():
    """Set of obs keys equals vendor OBSERVATION_DESC keys."""
    nh = NLECompat(seed=13)
    obs, _ = nh.reset()
    assert set(obs.keys()) == set(nle_nethack.OBSERVATION_DESC.keys())


# ---------------------------------------------------------------------------
# 3. Action space — byte-equivalence to vendor NLE
# ---------------------------------------------------------------------------


def test_action_space_discrete_121():
    """``action_space`` is Discrete(121)."""
    import gymnasium as gym

    nh = NLECompat(seed=14)
    assert isinstance(nh.action_space, gym.spaces.Discrete)
    assert int(nh.action_space.n) == 121


def test_actions_byte_identical_to_vendor():
    """Every entry in ``NLECompat.actions`` matches vendor ``nle.nethack.ACTIONS``.

    Citation: vendor/nle/nle/nethack/actions.py — 121-int action enum.
    """
    vendor = tuple(int(a) for a in nle_nethack.ACTIONS)
    nh = NLECompat(seed=15)
    assert nh.actions == vendor, (
        f"Action drift at first differing index: "
        f"{next((i, nh.actions[i], vendor[i]) for i in range(len(vendor)) if nh.actions[i] != vendor[i])}"
    )
    assert nh.action_set == vendor
    # Also assert against our pinned local copy.
    assert nh.actions == tuple(int(a) for a in ACTIONS)
    assert len(nh.actions) == N_ACTIONS == 121


# ---------------------------------------------------------------------------
# 4. Determinism — same seed -> same obs sequence
# ---------------------------------------------------------------------------


def test_determinism_same_seed_same_initial_obs():
    """Two NLECompats with same seed yield byte-equal initial glyphs."""
    nh1 = NLECompat(seed=999)
    nh2 = NLECompat(seed=999)
    obs1, _ = nh1.reset()
    obs2, _ = nh2.reset()
    # Glyphs MUST be equal — they're our deterministic ground truth.
    np.testing.assert_array_equal(
        np.asarray(obs1["glyphs"]), np.asarray(obs2["glyphs"])
    )
    # tty_chars deterministically derives from glyphs+chars.
    np.testing.assert_array_equal(
        np.asarray(obs1["tty_chars"]), np.asarray(obs2["tty_chars"])
    )


def test_determinism_same_seed_same_step_sequence():
    """Identical action sequences on same-seeded envs yield identical obs."""
    nh1 = NLECompat(seed=2024)
    nh2 = NLECompat(seed=2024)
    nh1.reset()
    nh2.reset()
    actions = [0, 1, 2, 3, 4]
    for a in actions:
        o1, r1, t1, tr1, _ = nh1.step(a)
        o2, r2, t2, tr2, _ = nh2.step(a)
        np.testing.assert_array_equal(
            np.asarray(o1["glyphs"]), np.asarray(o2["glyphs"])
        )
        assert r1 == r2
        assert t1 == t2
        assert tr1 == tr2


def test_reset_with_seed_kwarg_overrides_init_seed():
    """``reset(seed=K)`` reseeds and produces obs matching a fresh env(seed=K)."""
    nh1 = NLECompat(seed=0)
    obs1, _ = nh1.reset(seed=777)
    nh2 = NLECompat(seed=777)
    obs2, _ = nh2.reset()
    np.testing.assert_array_equal(
        np.asarray(obs1["glyphs"]), np.asarray(obs2["glyphs"])
    )


# ---------------------------------------------------------------------------
# 5. Wrapper compatibility — gymnasium wrappers
# ---------------------------------------------------------------------------


def test_wrapped_by_time_limit():
    """``gym.wrappers.TimeLimit`` wraps our shim without errors."""
    import gymnasium as gym
    from gymnasium.wrappers import TimeLimit

    nh = NLECompat(seed=16)
    wrapped = TimeLimit(nh, max_episode_steps=10)
    obs, info = wrapped.reset()
    assert isinstance(obs, dict)
    obs, r, term, trunc, info = wrapped.step(0)
    assert isinstance(r, float)


def test_wrapped_by_record_episode_statistics():
    """``gym.wrappers.RecordEpisodeStatistics`` wraps without errors."""
    from gymnasium.wrappers import RecordEpisodeStatistics

    nh = NLECompat(seed=17)
    wrapped = RecordEpisodeStatistics(nh, buffer_length=5)
    obs, info = wrapped.reset()
    assert isinstance(obs, dict)
    obs, r, term, trunc, info = wrapped.step(0)
    assert isinstance(r, float)


def test_max_episode_steps_truncates():
    """When ``max_episode_steps`` is small, ``truncated=True`` eventually fires.

    Citation: vendor/nle/nle/env/base.py::_check_abort.
    """
    nh = NLECompat(seed=18, max_episode_steps=3)
    nh.reset()
    truncated_seen = False
    for i in range(5):
        _, _, term, trunc, _ = nh.step(0)
        if trunc:
            truncated_seen = True
            break
        if term:
            break
    assert truncated_seen, "max_episode_steps=3 must trigger truncated=True"


# ---------------------------------------------------------------------------
# 6. NLE-specific extras
# ---------------------------------------------------------------------------


def test_character_attribute_present():
    """``nh.character`` reflects the constructor string (vendor parity)."""
    nh = NLECompat(seed=19, character="val-hum-law-fem")
    assert nh.character == "val-hum-law-fem"


def test_savedir_attribute_present():
    """``nh.savedir`` is present (None — we don't write ttyrecs)."""
    nh = NLECompat(seed=20)
    assert hasattr(nh, "savedir")
    assert nh.savedir is None


def test_last_observation_populated_after_reset():
    """``last_observation`` is a tuple in observation_keys order (vendor parity).

    Citation: vendor/nle/nle/env/base.py:238 ``self.last_observation = ()``.
    """
    nh = NLECompat(seed=21)
    assert nh.last_observation == ()  # empty before reset
    obs, _ = nh.reset()
    assert isinstance(nh.last_observation, tuple)
    assert len(nh.last_observation) == 17
    # First entry matches the "glyphs" key (vendor order).
    glyphs_idx = list(NLE_OBSERVATION_KEYS).index("glyphs")
    np.testing.assert_array_equal(
        np.asarray(nh.last_observation[glyphs_idx]),
        np.asarray(obs["glyphs"]),
    )


def test_step_info_has_end_status():
    """``info["end_status"]`` is populated with a ``StepStatus`` value."""
    nh = NLECompat(seed=22)
    nh.reset()
    _, _, _, _, info = nh.step(0)
    assert "end_status" in info
    assert isinstance(info["end_status"], NLECompat.StepStatus)


def test_observation_keys_subset_filter():
    """``observation_keys=`` subset arg returns only the requested keys.

    Citation: vendor/nle/nle/env/base.py::__init__ -- ``observation_keys`` arg.
    """
    keys = ("glyphs", "blstats", "message")
    nh = NLECompat(seed=23, observation_keys=keys)
    obs, _ = nh.reset()
    assert set(obs.keys()) == set(keys)
    obs, _, _, _, _ = nh.step(0)
    assert set(obs.keys()) == set(keys)


# ---------------------------------------------------------------------------
# 7. Observation byte-equivalence sweep (100 seeds, deterministic)
# ---------------------------------------------------------------------------


def test_glyphs_bounded_to_max_glyph_over_sweep():
    """Across 100 seeds the ``glyphs`` field never escapes [0, MAX_GLYPH].

    MAX_GLYPH=5976 comes from vendor/nle/include/nleobs.h MAX_GLYPH define
    (mirrored in vendor/nethack/include/display.h).
    """
    from Nethax.nethax.constants.glyphs import MAX_GLYPH

    for seed in range(100):
        nh = NLECompat(seed=seed)
        obs, _ = nh.reset()
        g = np.asarray(obs["glyphs"])
        assert g.dtype == np.int16
        assert int(g.min()) >= 0, f"seed {seed}: glyph below 0"
        assert int(g.max()) <= MAX_GLYPH, (
            f"seed {seed}: glyph {int(g.max())} exceeds MAX_GLYPH={MAX_GLYPH}"
        )


def test_obs_shape_stable_across_steps():
    """All 17 obs key shapes/dtypes stay stable across 50 steps."""
    nh = NLECompat(seed=33)
    obs, _ = nh.reset()
    rng = np.random.default_rng(11)
    for i in range(50):
        action = int(rng.integers(0, N_ACTIONS))
        obs, _, term, _, _ = nh.step(action)
        for key in NLE_OBSERVATION_KEYS:
            assert tuple(obs[key].shape) == NLE_OBSERVATION_SHAPES[key], (
                f"Step {i}, key {key}: shape drift"
            )
            assert jnp.dtype(obs[key].dtype) == jnp.dtype(NLE_OBSERVATION_DTYPES[key]), (
                f"Step {i}, key {key}: dtype drift"
            )
        if term:
            obs, _ = nh.reset()


# ---------------------------------------------------------------------------
# 8. glyph2tile parity — byte-equality with vendor (already established)
# ---------------------------------------------------------------------------


def test_glyph2tile_byte_equal_to_vendor():
    """Our glyph2tile table is byte-equal to vendor's 5976-entry table.

    Citation: vendor/nle/win/share/tiledata2.txt -> glyph2tile array of len 5976.
    """
    try:
        from nle.nethack import glyph2tile as vendor_glyph2tile
    except ImportError:
        pytest.skip("nle.nethack.glyph2tile not exposed in this build")
    from Nethax.tiles.tile_data import GLYPH2TILE

    vendor = np.asarray(vendor_glyph2tile, dtype=np.int32)
    ours = np.asarray(GLYPH2TILE, dtype=np.int32)
    assert ours.shape == vendor.shape, (
        f"glyph2tile length drift: ours={ours.shape}, vendor={vendor.shape}"
    )
    np.testing.assert_array_equal(ours, vendor)


# ---------------------------------------------------------------------------
# 9. Module-level glyph helpers + low-level constants
# ---------------------------------------------------------------------------


def test_module_level_glyph_helpers_exported():
    """All glyph predicate helpers are exported at module level."""
    from Nethax.nethax.compat import nle_shim as shim

    for name in (
        "nethack_glyph_to_char",
        "nethack_glyph_is_monster",
        "nethack_glyph_is_normal_monster",
        "nethack_glyph_is_pet",
        "nethack_glyph_is_body",
        "nethack_glyph_is_invisible",
        "nethack_glyph_is_object",
        "nethack_glyph_is_cmap",
        "nethack_glyph_is_swallow",
        "nethack_glyph_is_warning",
        "nethack_glyph_is_statue",
    ):
        assert hasattr(shim, name), f"Missing module-level helper: {name}"
        assert callable(getattr(shim, name))


def test_reset_advances_episode_counter():
    """Vendor parity: ``_episode`` counter increments on each reset."""
    nh = NLECompat(seed=24)
    assert nh._episode == -1
    nh.reset()
    assert nh._episode == 0
    nh.reset()
    assert nh._episode == 1


def test_step_counter_resets_on_reset():
    """``_steps`` resets to 0 on each ``reset()`` (vendor parity)."""
    nh = NLECompat(seed=25)
    nh.reset()
    nh.step(0)
    nh.step(0)
    assert nh._steps == 2
    nh.reset()
    assert nh._steps == 0
