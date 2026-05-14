"""Tests for NLE observation keys, shapes, and dtypes."""

import pytest
import jax.numpy as jnp


def test_17_keys():
    from Nethax.nethax.obs.nle_obs import NLE_OBSERVATION_KEYS
    assert len(NLE_OBSERVATION_KEYS) == 17, (
        f"Expected 17 NLE observation keys, got {len(NLE_OBSERVATION_KEYS)}"
    )


def test_empty_obs_keys():
    from Nethax.nethax.obs.nle_obs import NLE_OBSERVATION_KEYS, empty_nle_observation
    obs = empty_nle_observation()
    assert set(obs.keys()) == set(NLE_OBSERVATION_KEYS)


# Expected shapes per NLE spec.
_EXPECTED_SHAPES = {
    "glyphs":               (21, 79),
    "chars":                (21, 79),
    "colors":               (21, 79),
    "specials":             (21, 79),
    "blstats":              (27,),
    "message":              (256,),
    "program_state":        (6,),
    "internal":             (9,),
    "inv_glyphs":           (55,),
    "inv_letters":          (55,),
    "inv_oclasses":         (55,),
    "inv_strs":             (55, 80),
    "screen_descriptions":  (21, 79, 80),
    "tty_chars":            (24, 80),
    "tty_colors":           (24, 80),
    "tty_cursor":           (2,),
    "misc":                 (3,),
}


@pytest.mark.parametrize("key,expected_shape", list(_EXPECTED_SHAPES.items()))
def test_obs_shapes(key, expected_shape):
    from Nethax.nethax.obs.nle_obs import empty_nle_observation
    obs = empty_nle_observation()
    assert tuple(obs[key].shape) == expected_shape, (
        f"{key}: expected shape {expected_shape}, got {obs[key].shape}"
    )


# Expected dtypes per NLE spec.
_EXPECTED_DTYPES = {
    "glyphs":               jnp.int16,
    "chars":                jnp.uint8,
    "colors":               jnp.uint8,
    "specials":             jnp.uint8,
    "blstats":              jnp.int64,
    "message":              jnp.uint8,
    "program_state":        jnp.int32,
    "internal":             jnp.int32,
    "inv_glyphs":           jnp.int16,
    "inv_letters":          jnp.uint8,
    "inv_oclasses":         jnp.uint8,
    "inv_strs":             jnp.uint8,
    "screen_descriptions":  jnp.uint8,
    "tty_chars":            jnp.uint8,
    "tty_colors":           jnp.int8,
    "tty_cursor":           jnp.uint8,
    "misc":                 jnp.int32,
}


@pytest.mark.parametrize("key,expected_dtype", list(_EXPECTED_DTYPES.items()))
def test_obs_dtypes(key, expected_dtype):
    from Nethax.nethax.obs.nle_obs import empty_nle_observation
    obs = empty_nle_observation()
    assert obs[key].dtype == expected_dtype, (
        f"{key}: expected dtype {expected_dtype}, got {obs[key].dtype}"
    )
