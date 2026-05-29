"""Verify the cmap_indices module exposes both NLE 3.x and NetHack 5.x layouts.

Cite:
  - vendor/nle/include/rm.h:116-227         (3.x)
  - vendor/nethack/include/defsym.h:90-183  (5.x)
"""
from __future__ import annotations

import importlib
import os
import sys


def _reload_cmap(tree: str | None):
    """Reimport ``cmap_indices`` with the requested vendor tree env."""
    if tree is None:
        os.environ.pop("NETHAX_VENDOR_TREE", None)
    else:
        os.environ["NETHAX_VENDOR_TREE"] = tree
    sys.modules.pop("Nethax.nethax.constants.cmap_indices", None)
    return importlib.import_module("Nethax.nethax.constants.cmap_indices")


def test_default_layout_is_nle_3x():
    """Default (no env var) selects NLE 3.x; key indices match vendor/nle/rm.h."""
    ci = _reload_cmap(None)
    assert ci.VENDOR_TREE == "nle_3x"
    assert ci.is_nle_3x() is True
    assert ci.is_nethack_5x() is False
    # vendor/nle/include/rm.h:140-141
    assert ci.S_upstair == 23
    assert ci.S_dnstair == 24
    # vendor/nle/include/rm.h:138-139
    assert ci.S_corr == 21
    assert ci.S_litcorr == 22
    # vendor/nle/include/rm.h:144-148
    assert ci.S_altar == 27
    assert ci.S_fountain == 31
    # 3.x has no engraving / branch-stairs symbols.
    assert ci.S_engroom is None
    assert ci.S_engrcorr is None
    assert ci.S_brupstair is None


def test_nethack_5x_layout():
    """``NETHAX_VENDOR_TREE=nethack_5x`` selects the NetHack 3.7 layout."""
    ci = _reload_cmap("nethack_5x")
    assert ci.VENDOR_TREE == "nethack_5x"
    assert ci.is_nle_3x() is False
    assert ci.is_nethack_5x() is True
    # vendor/nethack/include/defsym.h:120-121
    assert ci.S_upstair == 25
    assert ci.S_dnstair == 26
    # vendor/nethack/include/defsym.h:116-118
    assert ci.S_corr == 22
    assert ci.S_litcorr == 23
    assert ci.S_engrcorr == 24
    # vendor/nethack/include/defsym.h:124-127 (new in 5.x)
    assert ci.S_brupstair == 29
    assert ci.S_brdnladder == 32
    # vendor/nethack/include/defsym.h:129
    assert ci.S_altar == 33
    # vendor/nethack/include/defsym.h:134
    assert ci.S_fountain == 37


def test_cmap_to_char_consistent_with_active_layout():
    """CMAP_TO_CHAR at index ACTIVE[sym] renders the same char in both layouts."""
    ci_3x = _reload_cmap(None)
    upstair_char_3x = chr(int(ci_3x.CMAP_TO_CHAR[ci_3x.S_upstair]))
    altar_char_3x = chr(int(ci_3x.CMAP_TO_CHAR[ci_3x.S_altar]))
    assert upstair_char_3x == "<"
    assert altar_char_3x == "_"

    ci_5x = _reload_cmap("nethack_5x")
    upstair_char_5x = chr(int(ci_5x.CMAP_TO_CHAR[ci_5x.S_upstair]))
    altar_char_5x = chr(int(ci_5x.CMAP_TO_CHAR[ci_5x.S_altar]))
    assert upstair_char_5x == "<"
    assert altar_char_5x == "_"
    # But indices differ between layouts.
    assert ci_3x.S_upstair != ci_5x.S_upstair


def test_unknown_value_falls_back_to_3x():
    """Unrecognised NETHAX_VENDOR_TREE values default to nle_3x."""
    ci = _reload_cmap("garbage_value")
    assert ci.VENDOR_TREE == "nle_3x"
    assert ci.S_upstair == 23
    # Reset env var so we don't poison the rest of the suite.
    _reload_cmap(None)
