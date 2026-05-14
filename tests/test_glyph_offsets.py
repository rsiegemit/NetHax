"""Tests for glyph offset constants and monotonicity."""


def test_mon_off_zero():
    from Nethax.nethax.constants.glyphs import GLYPH_MON_OFF
    assert GLYPH_MON_OFF == 0, f"GLYPH_MON_OFF must be 0, got {GLYPH_MON_OFF}"


def test_max_glyph_positive():
    from Nethax.nethax.constants.glyphs import MAX_GLYPH
    assert MAX_GLYPH > 0, f"MAX_GLYPH must be positive, got {MAX_GLYPH}"


def test_glyph_offsets_monotonic():
    """Each successive glyph segment must start after the previous one."""
    from Nethax.nethax.constants.glyphs import (
        GLYPH_MON_OFF,
        GLYPH_PET_OFF,
        GLYPH_INVIS_OFF,
        GLYPH_DETECT_OFF,
        GLYPH_BODY_OFF,
        GLYPH_RIDDEN_OFF,
        GLYPH_OBJ_OFF,
        GLYPH_CMAP_OFF,
        GLYPH_ZAP_OFF,
        GLYPH_SWALLOW_OFF,
        GLYPH_EXPLODE_OFF,
        GLYPH_WARNING_OFF,
        GLYPH_STATUE_OFF,
        MAX_GLYPH,
    )
    # Canonical NLE layout: EXPLODE comes before ZAP before SWALLOW
    # (see vendor/nle/win/rl/pynethack.cc and nle.nethack live binary).
    offsets = [
        GLYPH_MON_OFF,
        GLYPH_PET_OFF,
        GLYPH_INVIS_OFF,
        GLYPH_DETECT_OFF,
        GLYPH_BODY_OFF,
        GLYPH_RIDDEN_OFF,
        GLYPH_OBJ_OFF,
        GLYPH_CMAP_OFF,
        GLYPH_EXPLODE_OFF,
        GLYPH_ZAP_OFF,
        GLYPH_SWALLOW_OFF,
        GLYPH_WARNING_OFF,
        GLYPH_STATUE_OFF,
        MAX_GLYPH,
    ]
    for i in range(len(offsets) - 1):
        assert offsets[i] <= offsets[i + 1], (
            f"Glyph offsets not monotonic at index {i}: "
            f"{offsets[i]} > {offsets[i + 1]}"
        )
