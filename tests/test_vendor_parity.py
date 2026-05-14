"""Wave 2 tests for NLE vendor parity (action codes, blstats, glyph offsets).

All tests are skipped when NLE is not installed.  All imports are lazy
(inside test functions) so collection never fails if NLE is absent.
"""

import pytest

try:
    import nle.nethack  # noqa: F401
    nle_installed = True
except ImportError:
    nle_installed = False

_skip_no_nle = pytest.mark.skipif(
    not nle_installed, reason="NLE not installed"
)


@_skip_no_nle
def test_action_codes_match_nle():
    """Our ACTIONS int values must match nle.nethack.ACTIONS int values."""
    import nle.nethack as nnh
    from Nethax.nethax.constants.actions import ACTIONS

    nle_action_map = {a.__class__.__name__ + "." + a.name: int(a) for a in nnh.ACTIONS}

    mismatches = []
    for our_action in ACTIONS:
        key = our_action.__class__.__name__ + "." + our_action.name
        if key in nle_action_map:
            our_val = int(our_action)
            nle_val = nle_action_map[key]
            if our_val != nle_val:
                mismatches.append(
                    f"{key}: ours={our_val}, NLE={nle_val}"
                )

    assert not mismatches, "Action code mismatches:\n" + "\n".join(mismatches)


@_skip_no_nle
def test_blstats_indices_match_nle():
    """Our BL_* constants must match nle.nethack.NLE_BL_* values."""
    import nle.nethack as nnh
    from Nethax.nethax.constants import blstats as bl

    # Mapping from our constant name -> NLE attribute name
    pairs = [
        ("BL_X",         "NLE_BL_X"),
        ("BL_Y",         "NLE_BL_Y"),
        ("BL_STR25",     "NLE_BL_STR25"),
        ("BL_STR125",    "NLE_BL_STR125"),
        ("BL_DEX",       "NLE_BL_DEX"),
        ("BL_CON",       "NLE_BL_CON"),
        ("BL_INT",       "NLE_BL_INT"),
        ("BL_WIS",       "NLE_BL_WIS"),
        ("BL_CHA",       "NLE_BL_CHA"),
        ("BL_SCORE",     "NLE_BL_SCORE"),
        ("BL_HP",        "NLE_BL_HP"),
        ("BL_HPMAX",     "NLE_BL_HPMAX"),
        ("BL_DEPTH",     "NLE_BL_DEPTH"),
        ("BL_GOLD",      "NLE_BL_GOLD"),
        ("BL_ENE",       "NLE_BL_ENE"),
        ("BL_ENEMAX",    "NLE_BL_ENEMAX"),
        ("BL_AC",        "NLE_BL_AC"),
        ("BL_HD",        "NLE_BL_HD"),
        ("BL_XP",        "NLE_BL_XP"),
        ("BL_EXP",       "NLE_BL_EXP"),
        ("BL_TIME",      "NLE_BL_TIME"),
        ("BL_HUNGER",    "NLE_BL_HUNGER"),
        ("BL_CAP",       "NLE_BL_CAP"),
        ("BL_DNUM",      "NLE_BL_DNUM"),
        ("BL_DLEVEL",    "NLE_BL_DLEVEL"),
        ("BL_CONDITION", "NLE_BL_CONDITION"),
        ("BL_ALIGN",     "NLE_BL_ALIGN"),
    ]

    mismatches = []
    for our_name, nle_name in pairs:
        if not hasattr(bl, our_name):
            continue
        if not hasattr(nnh, nle_name):
            continue
        our_val = getattr(bl, our_name)
        nle_val = getattr(nnh, nle_name)
        if our_val != nle_val:
            mismatches.append(f"{our_name}={our_val} vs {nle_name}={nle_val}")

    assert not mismatches, "blstats index mismatches:\n" + "\n".join(mismatches)


@_skip_no_nle
def test_glyph_offsets_match_nle():
    """Our GLYPH_*_OFF constants must match nle.nethack.GLYPH_*_OFF values."""
    import nle.nethack as nnh
    from Nethax.nethax.constants import glyphs as g

    pairs = [
        ("GLYPH_MON_OFF",    "GLYPH_MON_OFF"),
        ("GLYPH_CMAP_OFF",   "GLYPH_CMAP_OFF"),
        ("NO_GLYPH",         "NO_GLYPH"),
    ]

    mismatches = []
    for our_name, nle_name in pairs:
        if not hasattr(g, our_name):
            continue
        if not hasattr(nnh, nle_name):
            continue
        our_val = getattr(g, our_name)
        nle_val = getattr(nnh, nle_name)
        if our_val != nle_val:
            mismatches.append(f"{our_name}={our_val} vs NLE {nle_name}={nle_val}")

    assert not mismatches, "Glyph offset mismatches:\n" + "\n".join(mismatches)
