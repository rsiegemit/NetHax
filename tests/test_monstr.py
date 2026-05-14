"""Tests for the full vendor monstr[] difficulty formula.

Canonical source: vendor/nethack/src/mondata.c::mstrength
The formula used here (see spawning._compute_monstr_full):
    monstr = level + min(move_speed//8, 5) + attk_count
             + (5 if any AT_BREA else 0)
             + (10 if any AD_STON else 0)
"""

from Nethax.nethax.constants.monsters import MONSTERS, NUMMONS
from Nethax.nethax.dungeon.spawning import MONSTR_DIFFICULTIES


def _idx_by_name(name: str) -> int:
    for i, m in enumerate(MONSTERS):
        if m.name == name:
            return i
    raise AssertionError(f"monster {name!r} not in MONSTERS table")


def test_monstr_low_level_monsters_low_difficulty():
    # Newt is a canonical L0 lizard; vendor mstrength gives a single-digit score.
    idx = _idx_by_name("newt")
    assert int(MONSTR_DIFFICULTIES[idx]) < 5


def test_monstr_dragon_has_high_difficulty_with_breath():
    # Red dragon: level 15 + breath bonus + multiple attacks -> >=20.
    idx = _idx_by_name("red dragon")
    assert int(MONSTR_DIFFICULTIES[idx]) >= 20


def test_monstr_cockatrice_has_petrify_bonus():
    # Cockatrice has AD_STON; petrify bonus pushes monstr above raw level.
    idx = _idx_by_name("cockatrice")
    assert int(MONSTR_DIFFICULTIES[idx]) > MONSTERS[idx].level


def test_monstr_table_size_matches_monsters_table():
    assert MONSTR_DIFFICULTIES.shape[0] == NUMMONS == len(MONSTERS)


def test_monstr_consistent_increases_with_level():
    # A high-level monster (Demogorgon) should outscore a low-level one (newt).
    hi = int(MONSTR_DIFFICULTIES[_idx_by_name("Demogorgon")])
    lo = int(MONSTR_DIFFICULTIES[_idx_by_name("newt")])
    assert hi > lo
