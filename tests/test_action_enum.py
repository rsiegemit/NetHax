"""Tests for the Action enum and ACTIONS/USEFUL_ACTIONS constants."""


def test_action_count_121():
    """Canonical NLE count is 121 (verified against vendor/nle HEAD)."""
    from Nethax.nethax.constants.actions import N_ACTIONS
    assert N_ACTIONS == 121


def test_useful_action_count():
    """Canonical NLE USEFUL = 121 - 20 NON_RL_ACTIONS = 101."""
    from Nethax.nethax.constants.actions import USEFUL_ACTIONS
    assert len(USEFUL_ACTIONS) == 101


def test_action_tuple_canonical_length():
    """ACTIONS has 121 entries (matches vendor NLE).

    Cross-class int-value collisions are intentional in NetHack key bindings
    (e.g., direction key 'h' shares its code with Command.HELP via the same
    ord), so `len(set(ACTIONS))` is intentionally smaller than `len(ACTIONS)`
    — IntEnum members hash by int value across enum classes.
    """
    from Nethax.nethax.constants.actions import ACTIONS
    assert len(ACTIONS) == 121


def test_compass_present():
    """There must be exactly 8 compass directions (4 cardinal + 4 intercardinal)."""
    from Nethax.nethax.constants.actions import (
        CompassCardinalDirection,
        CompassIntercardinalDirection,
        ACTIONS,
    )
    cardinal_values = {int(v) for v in CompassCardinalDirection}
    intercardinal_values = {int(v) for v in CompassIntercardinalDirection}
    compass_values = cardinal_values | intercardinal_values
    assert len(compass_values) == 8, f"Expected 8 compass directions, got {len(compass_values)}"
    action_values = {int(a) for a in ACTIONS}
    missing = compass_values - action_values
    assert not missing, f"Compass directions missing from ACTIONS: {missing}"
