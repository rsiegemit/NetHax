"""Tests for Nethax/nethax/save_load.py — Wave 6 save/load round-tripping."""
from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from Nethax.nethax import NethaxEnv
from Nethax.nethax.save_load import (
    IncompatibleSaveError,
    _NETHAX_SAVE_VERSION,
    check_version,
    load_state,
    save_state,
)
from Nethax.nethax.state import EnvState


def _fresh_state() -> EnvState:
    """Return a fresh reset state from NethaxEnv."""
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    return state


def _stepped_state(n_steps: int = 3) -> EnvState:
    """Return a state after running ``n_steps`` wait-actions."""
    env = NethaxEnv()
    rng = jax.random.PRNGKey(42)
    state, _ = env.reset(rng)
    for _ in range(n_steps):
        rng, step_rng = jax.random.split(rng)
        action = jnp.int32(ord("."))  # wait
        state, _, _, _, _ = env.step(state, action, step_rng)
    return state


def _assert_pytree_equal(a, b) -> None:
    """Assert two EnvState pytrees have equal leaves and identical treedefs."""
    leaves_a, td_a = jax.tree_util.tree_flatten(a)
    leaves_b, td_b = jax.tree_util.tree_flatten(b)
    assert td_a == td_b, "treedefs differ"
    assert len(leaves_a) == len(leaves_b)
    for i, (la, lb) in enumerate(zip(leaves_a, leaves_b)):
        np_a = np.asarray(la)
        np_b = np.asarray(lb)
        assert np_a.shape == np_b.shape, f"leaf {i} shape mismatch: {np_a.shape} vs {np_b.shape}"
        assert np_a.dtype == np_b.dtype, f"leaf {i} dtype mismatch: {np_a.dtype} vs {np_b.dtype}"
        assert np.array_equal(np_a, np_b), f"leaf {i} values differ"


def test_save_load_round_trip_default_state(tmp_path: Path) -> None:
    state = _fresh_state()
    path = tmp_path / "state.npz"
    save_state(state, path)
    loaded = load_state(path)
    _assert_pytree_equal(state, loaded)


def test_save_load_round_trip_after_steps(tmp_path: Path) -> None:
    state = _stepped_state(n_steps=5)
    path = tmp_path / "stepped.npz"
    save_state(state, path)
    loaded = load_state(path)
    _assert_pytree_equal(state, loaded)


def test_save_load_preserves_dtypes(tmp_path: Path) -> None:
    state = _fresh_state()
    path = tmp_path / "dtypes.npz"
    save_state(state, path)
    loaded = load_state(path)

    # Pick representative fields of distinct dtypes from the state schema.
    assert np.asarray(state.player_role).dtype == np.int8
    assert np.asarray(loaded.player_role).dtype == np.int8

    assert np.asarray(state.player_str).dtype == np.int16
    assert np.asarray(loaded.player_str).dtype == np.int16

    assert np.asarray(state.player_hp).dtype == np.int32
    assert np.asarray(loaded.player_hp).dtype == np.int32

    assert np.asarray(state.explored).dtype == np.bool_
    assert np.asarray(loaded.explored).dtype == np.bool_


def test_save_load_preserves_shapes(tmp_path: Path) -> None:
    state = _fresh_state()
    path = tmp_path / "shapes.npz"
    save_state(state, path)
    loaded = load_state(path)

    # monster_ai.pos has shape (MAX_MONSTERS=400, 2)
    assert state.monster_ai.pos.shape == (400, 2)
    assert loaded.monster_ai.pos.shape == (400, 2)

    # terrain shape (n_branches, max_levels, h, w)
    assert state.terrain.shape == loaded.terrain.shape
    assert state.visible.shape == loaded.visible.shape
    assert state.player_pos.shape == (2,)
    assert loaded.player_pos.shape == (2,)


def test_save_load_compressed_file_smaller_than_raw(tmp_path: Path) -> None:
    """Compressed .npz should be substantially smaller than the raw bytes
    of the leaves (most of EnvState is zero-initialised, so the compressor
    has lots of room).
    """
    state = _fresh_state()
    path = tmp_path / "compressed.npz"
    save_state(state, path)

    leaves, _ = jax.tree_util.tree_flatten(state)
    raw_bytes = sum(np.asarray(leaf).nbytes for leaf in leaves)
    compressed_bytes = path.stat().st_size

    assert compressed_bytes < raw_bytes, (
        f"compressed {compressed_bytes} bytes is not smaller than raw {raw_bytes} bytes"
    )


def test_load_wrong_version_raises(tmp_path: Path) -> None:
    """Hand-craft a .npz with a bogus _version → load_state must raise."""
    state = _fresh_state()
    leaves, treedef = jax.tree_util.tree_flatten(state)
    np_leaves = {f"leaf_{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    treedef_bytes = pickle.dumps(treedef)

    path = tmp_path / "wrong_version.npz"
    np.savez_compressed(
        str(path),
        _version=np.int32(999),
        treedef_str=np.frombuffer(treedef_bytes, dtype=np.uint8),
        **np_leaves,
    )

    with pytest.raises(IncompatibleSaveError):
        load_state(path)


def test_check_version_without_full_load(tmp_path: Path) -> None:
    state = _fresh_state()
    path = tmp_path / "version.npz"
    save_state(state, path)
    assert check_version(path) == _NETHAX_SAVE_VERSION


def test_load_corrupted_file_raises(tmp_path: Path) -> None:
    """Truncated / garbage file should raise (not silently return None)."""
    path = tmp_path / "corrupted.npz"
    path.write_bytes(b"this is not a valid npz file at all")
    with pytest.raises(Exception):
        load_state(path)


def test_save_load_preserves_player_pos(tmp_path: Path) -> None:
    state = _fresh_state()
    # Force a non-default player_pos so we can confirm the value round-trips.
    state = state.replace(player_pos=jnp.array([7, 13], dtype=jnp.int16))

    path = tmp_path / "player_pos.npz"
    save_state(state, path)
    loaded = load_state(path)

    assert np.array_equal(np.asarray(loaded.player_pos), np.array([7, 13], dtype=np.int16))
    assert np.asarray(loaded.player_pos).dtype == np.int16


def test_save_load_preserves_inventory_items(tmp_path: Path) -> None:
    state = _fresh_state()
    path = tmp_path / "inventory.npz"
    save_state(state, path)
    loaded = load_state(path)

    # Compare every leaf inside the inventory subtree.
    inv_leaves_a, inv_td_a = jax.tree_util.tree_flatten(state.inventory)
    inv_leaves_b, inv_td_b = jax.tree_util.tree_flatten(loaded.inventory)
    assert inv_td_a == inv_td_b
    assert len(inv_leaves_a) == len(inv_leaves_b)
    for la, lb in zip(inv_leaves_a, inv_leaves_b):
        assert np.array_equal(np.asarray(la), np.asarray(lb))


def test_save_load_accepts_str_and_path(tmp_path: Path) -> None:
    """Both ``str`` and ``pathlib.Path`` should be accepted as the path arg."""
    state = _fresh_state()
    path_obj = tmp_path / "as_path.npz"
    save_state(state, path_obj)
    loaded_a = load_state(path_obj)
    _assert_pytree_equal(state, loaded_a)

    path_str = str(tmp_path / "as_str.npz")
    save_state(state, path_str)
    loaded_b = load_state(path_str)
    _assert_pytree_equal(state, loaded_b)
