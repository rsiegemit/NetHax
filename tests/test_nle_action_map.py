"""Unit tests for the NLE-index ↔ Nethax-ASCII action map.

Cite: Nethax/nethax/nle_action_map.py; vendor/nle/nle/nethack/actions.py
ACTIONS/USEFUL_ACTIONS construction.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import sys
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax
import jax.numpy as jnp

from Nethax.nethax.nle_action_map import (
    NLE_INDEX_TO_ASCII,
    N_NLE_ACTIONS,
    NLE_FULL_INDEX_TO_ASCII,
    N_NLE_FULL_ACTIONS,
    maybe_remap_action,
    nle_index_to_ascii,
)


def test_table_sizes():
    assert N_NLE_ACTIONS == 86
    assert N_NLE_FULL_ACTIONS == 121
    assert NLE_INDEX_TO_ASCII.shape == (86,)
    assert NLE_FULL_INDEX_TO_ASCII.shape == (121,)
    assert NLE_INDEX_TO_ASCII.dtype == jnp.int32


def test_compass_directions():
    # Verified against vendor/nle/nle/nethack/actions.py:36-57.
    assert int(NLE_INDEX_TO_ASCII[0]) == ord("k")   # N
    assert int(NLE_INDEX_TO_ASCII[1]) == ord("l")   # E
    assert int(NLE_INDEX_TO_ASCII[2]) == ord("j")   # S
    assert int(NLE_INDEX_TO_ASCII[3]) == ord("h")   # W
    assert int(NLE_INDEX_TO_ASCII[4]) == ord("u")   # NE
    assert int(NLE_INDEX_TO_ASCII[5]) == ord("n")   # SE
    assert int(NLE_INDEX_TO_ASCII[6]) == ord("b")   # SW
    assert int(NLE_INDEX_TO_ASCII[7]) == ord("y")   # NW


def test_misc_directions():
    # Vendor lines 83-87.
    assert int(NLE_INDEX_TO_ASCII[16]) == ord("<")  # UP
    assert int(NLE_INDEX_TO_ASCII[17]) == ord(">")  # DOWN
    assert int(NLE_INDEX_TO_ASCII[18]) == ord(".")  # WAIT


def test_known_commands():
    # SEARCH = ord('s'); spot-check from the smoke-test action set.
    assert int(NLE_INDEX_TO_ASCII[61]) == ord("s")


def test_maybe_remap_index_path():
    # Index < 86 → gather from table.
    for idx in (0, 1, 7, 18, 61):
        out = int(maybe_remap_action(jnp.int32(idx)))
        assert out == int(NLE_INDEX_TO_ASCII[idx])


def test_maybe_remap_ord_path():
    # Action >= 86 → passthrough (already an ASCII ord).
    # Note: the brief's heuristic cuts at 86, so low ASCII (e.g. '.' = 46,
    # '<' = 60, '>' = 62) gets re-interpreted as an NLE index.  Callers
    # using ASCII for those directions must pass the NLE index instead
    # (see USEFUL_ACTIONS: idx 16='<', 17='>', 18='.').
    for ord_val in (107, 108, 115, 86, 90, 200, 255):
        assert int(maybe_remap_action(jnp.int32(ord_val))) == ord_val


def test_maybe_remap_is_jit_safe():
    # Critical for the task brief — the lookup must trace cleanly under
    # jax.jit / vmap.
    f = jax.jit(maybe_remap_action)
    assert int(f(jnp.int32(0))) == 107       # N
    assert int(f(jnp.int32(115))) == 115     # passthrough
    # vmap
    g = jax.vmap(maybe_remap_action)
    out = g(jnp.array([0, 1, 18, 61, 107, 115], dtype=jnp.int32))
    expected = jnp.array([107, 108, 46, 115, 107, 115], dtype=jnp.int32)
    assert jnp.array_equal(out, expected)


def test_smoke_test_action_set_round_trip():
    # The 6 actions from test_nle_return_distribution.py — verify the
    # _NLE_ACTION_INDICES → _NETHAX_ACTION_ORDS map matches the table.
    indices = (0, 1, 2, 3, 61, 18)
    ords    = (107, 108, 106, 104, 115, 46)
    for idx, expected_ord in zip(indices, ords):
        assert int(NLE_INDEX_TO_ASCII[idx]) == expected_ord, (
            f"smoke test index {idx} should map to ord {expected_ord} "
            f"but table has {int(NLE_INDEX_TO_ASCII[idx])}"
        )
