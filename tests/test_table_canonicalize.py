"""Tests for monster/object table canonicalization vs vendor NLE.

Wave 6 Phase B trimmed dual-named potions/scrolls and aligned MONSTERS with
vendor NLE monst.c (NUM_MONSTERS == 381).
"""

from Nethax.nethax.constants.objects import OBJECTS
from Nethax.nethax.constants.monsters import MONSTERS, NUMMONS


def test_num_objects_canonicalized():
    # Was 503 before canonicalization; must be well below 500.
    assert len(OBJECTS) < 500


def test_num_objects_close_to_target():
    # Target window after dual-name dedup pass.
    assert 450 <= len(OBJECTS) <= 470


def test_num_monsters_is_381():
    # Vendor NLE monst.c canonical count.
    assert NUMMONS == 381
    assert len(MONSTERS) == 381


def test_no_charon():
    # Charon is vendor #ifdef CHARON guarded; NLE default build excludes it.
    names = {m.name for m in MONSTERS}
    assert "Charon" not in names


def test_no_mail_daemon():
    # mail daemon is vendor #ifdef MAIL guarded; NLE default build excludes it.
    names = {m.name for m in MONSTERS}
    assert "mail daemon" not in names


def test_no_dual_named_potions():
    # Bare-name canonicalization: "healing" present (POTION_CLASS), but the
    # verbose dual "potion of healing" must have been dropped.
    names = {o.name for o in OBJECTS}
    if "healing" in names:
        assert "potion of healing" not in names
