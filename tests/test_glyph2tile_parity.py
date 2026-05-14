"""Verify GLYPH2TILE mapping is byte-equal to live nle.nethack.glyph2tile.

Wave 6 polish: the GLYPH2TILE table is auto-generated from live NLE
(see Nethax/tiles/tile_data.py docstring).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

try:
    import nle.nethack as nh
    HAS_NLE = True
except ImportError:
    HAS_NLE = False


@pytest.mark.skipif(not HAS_NLE, reason="nle.nethack not installed")
def test_glyph2tile_length_matches_nle():
    from Nethax.tiles.tile_data import GLYPH2TILE
    nle_arr = np.array(nh.glyph2tile, dtype=np.int32)
    our = np.asarray(GLYPH2TILE)
    assert len(our) == len(nle_arr), (
        f"GLYPH2TILE length mismatch: ours={len(our)}, NLE={len(nle_arr)}"
    )


@pytest.mark.skipif(not HAS_NLE, reason="nle.nethack not installed")
def test_glyph2tile_byte_equal_nle():
    from Nethax.tiles.tile_data import GLYPH2TILE
    nle_arr = np.array(nh.glyph2tile, dtype=np.int32)
    our = np.asarray(GLYPH2TILE)
    assert np.array_equal(our, nle_arr), (
        "GLYPH2TILE not byte-equal to live nle.nethack.glyph2tile"
    )


def test_glyph2tile_length_5976():
    """MAX_GLYPH=5976 in NLE; our table must cover all glyphs."""
    from Nethax.tiles.tile_data import GLYPH2TILE
    assert len(GLYPH2TILE) == 5976, f"expected 5976 entries, got {len(GLYPH2TILE)}"
