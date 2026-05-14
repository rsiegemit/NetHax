"""Save and load game state as serialised JAX pytrees.

Canonical sources:
  vendor/nethack/src/save.c    — savegame(), savelev(), saveobj(); binary
                                  struct serialisation of the entire game
                                  state (~4 500 lines)
  vendor/nethack/src/restore.c — restgame(), getlev(), restobj(); the
                                  symmetric deserialisation path

Design departure
----------------
NetHack's save.c writes raw C structs to disk in a platform-specific binary
format.  Nethax instead serialises the ``EnvState`` JAX pytree using numpy's
compressed ``.npz`` container:

  save   : jax.tree_util.tree_flatten(state)  →  list of numpy arrays
           → np.savez_compressed(path, leaf_0=..., leaf_1=..., ...,
                                 treedef_str=pickle.dumps(treedef),
                                 _version=_NETHAX_SAVE_VERSION)
  restore: np.load(path)  →  rebuild leaves and treedef
           → jax.tree_util.tree_unflatten(treedef, leaves)

Format choice rationale: ``.npz`` is JAX/numpy-native, supports per-array
compression, and stores each leaf as an addressable key — letting us read
just the version field without rehydrating the entire state.

Cross-version compatibility: the file embeds an integer ``_version`` field.
``load_state`` raises ``IncompatibleSaveError`` if the on-disk version does
not match ``_NETHAX_SAVE_VERSION``.  Bump the constant whenever ``EnvState``'s
field declaration order (and therefore the pytree treedef) changes.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

import jax
import numpy as np

from Nethax.nethax.state import EnvState

_NETHAX_SAVE_VERSION = 1  # bump when EnvState schema changes


class IncompatibleSaveError(Exception):
    """Raised when an on-disk save's version does not match the runtime."""


def save_state(state: EnvState, path: Union[str, Path]) -> None:
    """Flatten ``state`` (an EnvState pytree) and write to ``path`` as .npz.

    Each pytree leaf is stored under key ``leaf_<i>``; the treedef is pickled
    into ``treedef_str``; the format version is stored as ``_version``.

    Parameters
    ----------
    state : EnvState — the full game state to persist.
    path  : Destination file path (``.npz`` extension recommended).
    """
    leaves, treedef = jax.tree_util.tree_flatten(state)
    np_leaves = {f"leaf_{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    treedef_bytes = pickle.dumps(treedef)
    np.savez_compressed(
        str(path),
        _version=np.int32(_NETHAX_SAVE_VERSION),
        treedef_str=np.frombuffer(treedef_bytes, dtype=np.uint8),
        **np_leaves,
    )


def load_state(path: Union[str, Path]) -> EnvState:
    """Read an EnvState pytree previously written by ``save_state``.

    Parameters
    ----------
    path : Source ``.npz`` path.

    Returns
    -------
    state : EnvState restored from disk.

    Raises
    ------
    IncompatibleSaveError : if the on-disk version does not match the current
        ``_NETHAX_SAVE_VERSION``.
    """
    with np.load(str(path), allow_pickle=False) as npz:
        version = int(npz["_version"])
        if version != _NETHAX_SAVE_VERSION:
            raise IncompatibleSaveError(
                f"Save file version {version} does not match runtime "
                f"version {_NETHAX_SAVE_VERSION}.  Migrate the save or use "
                f"a matching Nethax build."
            )
        treedef_bytes = bytes(npz["treedef_str"])
        treedef = pickle.loads(treedef_bytes)

        leaf_keys = sorted(
            (k for k in npz.files if k.startswith("leaf_")),
            key=lambda k: int(k.split("_", 1)[1]),
        )
        leaves = [np.asarray(npz[k]) for k in leaf_keys]

    return jax.tree_util.tree_unflatten(treedef, leaves)


def check_version(path: Union[str, Path]) -> int:
    """Read just the ``_version`` field of a save file without rehydrating it.

    Useful for migration tooling that wants to detect old saves cheaply.
    """
    with np.load(str(path), allow_pickle=False) as npz:
        return int(npz["_version"])
