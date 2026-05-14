"""Tests for blstats index layout and NLE conventions."""

import inspect


def test_n_blstats_27():
    from Nethax.nethax.constants.blstats import N_BLSTATS
    assert N_BLSTATS == 27


def test_bl_indices_unique_0_to_26():
    """All BL_* integer constants must form exactly {0, 1, ..., 26}."""
    import Nethax.nethax.constants.blstats as blstats_mod
    bl_values = [
        v for name, v in inspect.getmembers(blstats_mod)
        if name.startswith("BL_") and isinstance(v, int)
        and not name.startswith("BL_MASK_")
    ]
    assert sorted(bl_values) == list(range(27)), (
        f"BL_* indices are not a clean 0-26 range: {sorted(bl_values)}"
    )


def test_bl_hp_position():
    from Nethax.nethax.constants.blstats import BL_HP
    assert BL_HP == 10, f"NLE convention: BL_HP == 10, got {BL_HP}"


def test_bl_align_position():
    from Nethax.nethax.constants.blstats import BL_ALIGN
    assert BL_ALIGN == 26, f"NLE convention: BL_ALIGN == 26, got {BL_ALIGN}"
