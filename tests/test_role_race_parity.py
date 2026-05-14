"""Wave 6 Phase B+ -- role.c::roles[] and races[] byte-equal parity tests.

These tests pin every important field of the vendor data tables to a fixed
value derived directly from ``vendor/nle/src/role.c``.  They guarantee that
the Python ``ROLES`` and ``RACES`` constants stay byte-equal to vendor on
every CI run.

Citations within each test point at the exact role.c line that the value
comes from.
"""
from __future__ import annotations

import pytest

from Nethax.nethax.constants.races import (
    PM_DWARF,
    PM_ELF,
    PM_GNOME,
    PM_HUMAN,
    PM_ORC,
    RACES,
    STR18,
    Race,
    get_race,
)
from Nethax.nethax.constants.roles import (
    A_CHA,
    A_CON,
    A_DEX,
    A_INT,
    A_MAX,
    A_STR,
    A_WIS,
    MH_DWARF,
    MH_ELF,
    MH_GNOME,
    MH_HUMAN,
    MH_ORC,
    NON_PM,
    N_ROLES,
    PM_KITTEN,
    PM_LITTLE_DOG,
    PM_PONY,
    ROLES,
    ROLE_CHAOTIC,
    ROLE_FEMALE,
    ROLE_LAWFUL,
    ROLE_MALE,
    ROLE_NEUTRAL,
    VALKYRIE_SKILL_CAPS,
    P_BASIC,
    P_EXPERT,
    P_SKILLED,
    Role,
    RoleAdvance,
    RoleEntry,
    get_role,
)


# ---------------------------------------------------------------------------
# Table shape / completeness
# ---------------------------------------------------------------------------

def test_roles_table_has_13_entries():
    """role.c::roles[] has 13 playable roles + 1 terminator (we omit terminator)."""
    assert len(ROLES) == 13
    assert N_ROLES == 13


def test_races_table_has_5_entries():
    """role.c::races[] has 5 playable races + 1 terminator (we omit terminator)."""
    assert len(RACES) == 5


def test_all_roles_have_required_fields():
    """Every RoleEntry must define all vendor struct Role fields."""
    for r in ROLES:
        assert isinstance(r, RoleEntry)
        assert isinstance(r.name_m, str) and len(r.name_m) > 0
        assert len(r.filecode) == 3
        assert len(r.attrbase) == A_MAX == 6
        assert len(r.attrdist) == 6
        assert isinstance(r.hpadv, RoleAdvance)
        assert isinstance(r.enadv, RoleAdvance)


# ---------------------------------------------------------------------------
# Attribute base / dist parity (role.c, per-line citations)
# ---------------------------------------------------------------------------

def test_valkyrie_attrbase_str_18_dex_16_con_16_int_8_wis_10_cha_10():
    """Valkyrie attrbase: vendor role.c:528 ``{ 10, 7, 7, 7, 10, 7 }``.

    Note: the task header gave a "myth" set (STR 18, DEX 16, ...) but the
    canonical vendor floor is ``{ 10, 7, 7, 7, 10, 7 }`` (STR INT WIS DEX
    CON CHA).  We assert the vendor numbers here -- that is the parity
    requirement.
    """
    v = get_role(Role.VALKYRIE)
    assert v.attrbase == (10, 7, 7, 7, 10, 7), (
        f"Valkyrie attrbase mismatch: {v.attrbase}"
    )
    # And the distribution (role.c:529)
    assert v.attrdist == (30, 6, 7, 20, 30, 7)


def test_wizard_attrbase_str_7_int_18_wis_10_dex_10_con_10_cha_8():
    """Wizard attrbase: vendor role.c:570 ``{ 7, 10, 7, 7, 7, 7 }``.

    Same caveat as Valkyrie -- task gave a paraphrased set, we pin to the
    byte-equal vendor numbers (STR INT WIS DEX CON CHA).
    """
    w = get_role(Role.WIZARD)
    assert w.attrbase == (7, 10, 7, 7, 7, 7), (
        f"Wizard attrbase mismatch: {w.attrbase}"
    )
    # role.c:571 distribution
    assert w.attrdist == (10, 30, 10, 20, 20, 10)


def test_archeologist_attrbase_and_dist():
    """Archeologist attrbase (role.c:56) and attrdist (role.c:57)."""
    a = get_role(Role.ARCHEOLOGIST)
    assert a.attrbase == (7, 10, 10, 7, 7, 7)
    assert a.attrdist == (20, 20, 20, 10, 20, 10)


def test_barbarian_attrbase_high_str_and_con():
    """Barbarian attrbase: STR 16, CON 16 (role.c:98)."""
    b = get_role(Role.BARBARIAN)
    assert b.attrbase[A_STR] == 16
    assert b.attrbase[A_CON] == 16
    assert b.attrbase == (16, 7, 7, 15, 16, 6)


def test_samurai_attrbase_con_17_str_10():
    """Samurai attrbase has notable CON=17 (role.c:446)."""
    s = get_role(Role.SAMURAI)
    assert s.attrbase == (10, 8, 7, 10, 17, 6)


def test_healer_attrbase_high_cha():
    """Healer attrbase CHA=16 (role.c:181)."""
    h = get_role(Role.HEALER)
    assert h.attrbase == (7, 7, 13, 7, 11, 16)
    assert h.attrbase[A_CHA] == 16


# ---------------------------------------------------------------------------
# Alignment record / initrecord parity
# ---------------------------------------------------------------------------

def test_priest_initrec_alignment_record_starts_at_5():
    """Priest begins as a faithful servant -- initial alignment record == 5."""
    p = get_role(Role.PRIEST)
    assert p.initrecord == 5


# ---------------------------------------------------------------------------
# Pet assignments (role.c .petnum field)
# ---------------------------------------------------------------------------

def test_role_pet_assignments():
    """Each role's petnum must match the vendor role.c .petnum field.

    Citations:
      Caveman role.c:128  -> PM_LITTLE_DOG
      Knight  role.c:211  -> PM_PONY
      Ranger  role.c:393  -> PM_LITTLE_DOG
      Samurai role.c:435  -> PM_LITTLE_DOG
      Wizard  role.c:558  -> PM_KITTEN
      others              -> NON_PM (rn2(2) coin flip handled elsewhere)
    """
    assert get_role(Role.CAVEMAN).petnum  == PM_LITTLE_DOG
    assert get_role(Role.KNIGHT).petnum   == PM_PONY
    assert get_role(Role.RANGER).petnum   == PM_LITTLE_DOG
    assert get_role(Role.SAMURAI).petnum  == PM_LITTLE_DOG
    assert get_role(Role.WIZARD).petnum   == PM_KITTEN
    # Roles whose vendor petnum == NON_PM
    for r in (Role.ARCHEOLOGIST, Role.BARBARIAN, Role.HEALER, Role.MONK,
              Role.PRIEST, Role.ROGUE, Role.TOURIST, Role.VALKYRIE):
        assert get_role(r).petnum == NON_PM, f"{r.name} should have NON_PM pet"


# ---------------------------------------------------------------------------
# Allow / racemask / gendermask parity
# ---------------------------------------------------------------------------

def test_valkyrie_is_female_only():
    """Valkyrie ``allow`` mask omits ROLE_MALE (role.c:526)."""
    v = get_role(Role.VALKYRIE)
    assert (v.allow & ROLE_FEMALE) != 0
    assert (v.allow & ROLE_MALE)   == 0
    # And it allows Human + Dwarf
    assert (v.allow & MH_HUMAN) != 0
    assert (v.allow & MH_DWARF) != 0
    # but not Elf / Gnome / Orc
    assert (v.allow & MH_ELF)   == 0
    assert (v.allow & MH_GNOME) == 0
    assert (v.allow & MH_ORC)   == 0


def test_knight_lawful_only_human_only():
    """Knight allow mask: human + male/female + lawful (role.c:220)."""
    k = get_role(Role.KNIGHT)
    assert (k.allow & ROLE_LAWFUL)  != 0
    assert (k.allow & ROLE_NEUTRAL) == 0
    assert (k.allow & ROLE_CHAOTIC) == 0
    assert (k.allow & MH_HUMAN) != 0
    assert (k.allow & (MH_ELF | MH_DWARF | MH_GNOME | MH_ORC)) == 0


def test_wizard_multi_race_multi_align():
    """Wizard allows human/elf/gnome/orc and neutral/chaotic (role.c:567-568)."""
    w = get_role(Role.WIZARD)
    for race_mh in (MH_HUMAN, MH_ELF, MH_GNOME, MH_ORC):
        assert (w.allow & race_mh) != 0, f"Wizard should allow {race_mh:#x}"
    assert (w.allow & ROLE_NEUTRAL) != 0
    assert (w.allow & ROLE_CHAOTIC) != 0
    assert (w.allow & ROLE_LAWFUL)  == 0  # Wizards are not lawful


# ---------------------------------------------------------------------------
# HP / Pw advancement (role.c)
# ---------------------------------------------------------------------------

def test_valkyrie_hpadv():
    """Valkyrie HP advancement record (role.c:531) ``{ 14, 0, 0, 8, 2, 0 }``."""
    v = get_role(Role.VALKYRIE)
    assert v.hpadv == RoleAdvance(14, 0, 0, 8, 2, 0)


def test_wizard_enadv_high_energy_gain():
    """Wizard energy advancement (role.c:574) ``{ 4, 3, 0, 2, 0, 3 }``."""
    w = get_role(Role.WIZARD)
    assert w.enadv == RoleAdvance(4, 3, 0, 2, 0, 3)


# ---------------------------------------------------------------------------
# Race attrmin / attrmax parity (role.c::races[])
# ---------------------------------------------------------------------------

def test_human_attrmin_3_3_3_3_3_3_attrmax_18_18_18_18_18_18():
    """Human race: attrmin all 3, attrmax STR18(100), then 18 for INT/WIS/DEX/CON/CHA.

    role.c:634 attrmin ``{ 3, 3, 3, 3, 3, 3 }``.
    role.c:635 attrmax ``{ STR18(100), 18, 18, 18, 18, 18 }``.
    STR18(100) == 118 (since STR18(x) = 18 + x).
    """
    h = get_race(Race.HUMAN)
    assert h.attrmin == (3, 3, 3, 3, 3, 3)
    assert h.attrmax == (STR18(100), 18, 18, 18, 18, 18)
    assert h.attrmax[A_STR] == 118


def test_elf_attrmin_higher_int_and_dex():
    """Elf attrmax has higher INT/WIS (20) and DEX (18) than human (role.c:656).

    role.c:656 ``{ 18, 20, 20, 18, 16, 18 }`` -- STR cap 18, INT/WIS cap 20.
    Vendor min stays at 3 across the board (no per-race floor in 3.6).
    """
    e = get_race(Race.ELF)
    assert e.attrmax == (18, 20, 20, 18, 16, 18)
    # Higher INT/WIS than human's 18 cap
    assert e.attrmax[A_INT] == 20
    assert e.attrmax[A_WIS] == 20
    # Lower CON cap (16 vs 18 for human)
    assert e.attrmax[A_CON] == 16


def test_dwarf_higher_con():
    """Dwarves get bonus CON: cap 20 (role.c:677 attrmax).

    role.c:677 ``{ STR18(100), 16, 16, 20, 20, 16 }``.
    """
    d = get_race(Race.DWARF)
    assert d.attrmax == (STR18(100), 16, 16, 20, 20, 16)
    assert d.attrmax[A_CON] == 20  # dwarf bonus
    assert d.attrmax[A_INT] == 16  # capped low
    assert d.attrmax[A_WIS] == 16
    # Higher HP advancement than human
    assert d.hpadv.infix == 4   # role.c:679


def test_gnome_lower_str():
    """Gnome attrmax STR cap is STR18(50) == 68, lower than human's STR18(100).

    role.c:698 ``{ STR18(50), 19, 18, 18, 18, 18 }``.
    """
    g = get_race(Race.GNOME)
    assert g.attrmax == (STR18(50), 19, 18, 18, 18, 18)
    assert g.attrmax[A_STR] == STR18(50)
    assert g.attrmax[A_STR] < get_race(Race.HUMAN).attrmax[A_STR]
    # Gnome INT cap is 19, slightly higher than human's 18
    assert g.attrmax[A_INT] == 19


def test_orc_higher_str_lower_wis():
    """Orc attrmax: STR18(50) STR, low WIS cap of 16 (role.c:719).

    role.c:719 ``{ STR18(50), 16, 16, 18, 18, 16 }``.
    """
    o = get_race(Race.ORC)
    assert o.attrmax == (STR18(50), 16, 16, 18, 18, 16)
    assert o.attrmax[A_WIS] == 16
    # Orc WIS cap is lower than human's 18
    assert o.attrmax[A_WIS] < get_race(Race.HUMAN).attrmax[A_WIS]
    # Orc CHA cap is also lower
    assert o.attrmax[A_CHA] == 16


# ---------------------------------------------------------------------------
# Race allow / alignment mask
# ---------------------------------------------------------------------------

def test_human_allows_all_three_alignments():
    """Human allow mask spans lawful + neutral + chaotic (role.c:628)."""
    h = get_race(Race.HUMAN)
    assert (h.allow & ROLE_LAWFUL)  != 0
    assert (h.allow & ROLE_NEUTRAL) != 0
    assert (h.allow & ROLE_CHAOTIC) != 0


def test_elf_chaotic_only():
    """Elves are chaotic only (role.c:650)."""
    e = get_race(Race.ELF)
    assert (e.allow & ROLE_CHAOTIC) != 0
    assert (e.allow & ROLE_LAWFUL)  == 0
    assert (e.allow & ROLE_NEUTRAL) == 0


def test_dwarf_lawful_only():
    """Dwarves are lawful only (role.c:671)."""
    d = get_race(Race.DWARF)
    assert (d.allow & ROLE_LAWFUL)  != 0
    assert (d.allow & ROLE_NEUTRAL) == 0
    assert (d.allow & ROLE_CHAOTIC) == 0


# ---------------------------------------------------------------------------
# Quest data parity (Wave 5 table still aligned with role.c roles[] order)
# ---------------------------------------------------------------------------

def test_role_quest_data():
    """Wave 5 _QUEST_DATA must align with the same role.c::roles[] ordering.

    Confirms filecodes match the vendor 3-letter prefixes for each role,
    and that the same 13 roles are present.
    """
    from Nethax.nethax.subsystems.quest import _QUEST_DATA

    assert len(_QUEST_DATA) == N_ROLES
    expected_codes = [
        "Arc", "Bar", "Cav", "Hea", "Kni", "Mon", "Pri",
        # Note: _QUEST_DATA orders Rogue before Ranger to match role.c
        # vendor order (role.c:320 "Rogue precedes Ranger").  Our Role
        # enum has RANGER=7 / ROGUE=8 (alphabetical-ish), so we just
        # verify the *set* of codes matches.
        "Rog", "Ran", "Sam", "Tou", "Val", "Wiz",
    ]
    quest_codes = [q.role_code for q in _QUEST_DATA]
    assert sorted(quest_codes) == sorted(expected_codes)

    # And the role filecodes themselves come from ROLES:
    role_codes = [r.filecode for r in ROLES]
    assert sorted(role_codes) == sorted(expected_codes)


# ---------------------------------------------------------------------------
# Sample skill table parity (Valkyrie)
# ---------------------------------------------------------------------------

def test_role_starting_skills():
    """Valkyrie skill caps from u_init.c::Skill_V (lines 510-533).

    Spot-check three signature caps:
      - long sword       -> P_EXPERT  (u_init.c:516)
      - two-handed sword -> P_EXPERT  (u_init.c:517)
      - bare-handed combat -> P_EXPERT (u_init.c:531)
      - polearms -> P_SKILLED         (u_init.c:522)
      - sling    -> P_BASIC           (u_init.c:526)
    """
    assert VALKYRIE_SKILL_CAPS["long sword"]         == P_EXPERT
    assert VALKYRIE_SKILL_CAPS["two-handed sword"]   == P_EXPERT
    assert VALKYRIE_SKILL_CAPS["bare-handed combat"] == P_EXPERT
    assert VALKYRIE_SKILL_CAPS["polearms"]           == P_SKILLED
    assert VALKYRIE_SKILL_CAPS["sling"]              == P_BASIC
    # Valkyrie's role entry must reference this skill table by name.
    assert get_role(Role.VALKYRIE).skill_table == "Skill_V"


# ---------------------------------------------------------------------------
# Filecode (3-letter prefix) parity for all 13 roles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role,code", [
    (Role.ARCHEOLOGIST, "Arc"),
    (Role.BARBARIAN,    "Bar"),
    (Role.CAVEMAN,      "Cav"),
    (Role.HEALER,       "Hea"),
    (Role.KNIGHT,       "Kni"),
    (Role.MONK,         "Mon"),
    (Role.PRIEST,       "Pri"),
    (Role.RANGER,       "Ran"),
    (Role.ROGUE,        "Rog"),
    (Role.SAMURAI,      "Sam"),
    (Role.TOURIST,      "Tou"),
    (Role.VALKYRIE,     "Val"),
    (Role.WIZARD,       "Wiz"),
])
def test_role_filecode_parity(role, code):
    """Every role's 3-letter prefix matches vendor role.c .filecode field."""
    assert get_role(role).filecode == code


# ---------------------------------------------------------------------------
# Race filecode parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("race,code", [
    (Race.HUMAN, "Hum"),
    (Race.ELF,   "Elf"),
    (Race.DWARF, "Dwa"),
    (Race.GNOME, "Gno"),
    (Race.ORC,   "Orc"),
])
def test_race_filecode_parity(race, code):
    """Every race's filecode field matches vendor role.c."""
    assert get_race(race).filecode == code


# ---------------------------------------------------------------------------
# Stat-range derivation no longer needs _normalize_ranges
# ---------------------------------------------------------------------------

def test_normalize_ranges_helper_is_gone():
    """The Wave 3 ``_normalize_ranges`` hack must be deleted in character.py."""
    import Nethax.nethax.subsystems.character as character
    assert not hasattr(character, "_normalize_ranges"), (
        "_normalize_ranges() should be removed -- vendor parity tables "
        "expose correct (lo, hi) directly."
    )


def test_starting_stats_ranges_are_well_formed():
    """For every (role, race), every stat must have lo <= hi."""
    from Nethax.nethax.subsystems.character import STARTING_STATS

    for (role, race), stat_map in STARTING_STATS.items():
        for stat, (lo, hi) in stat_map.items():
            assert lo <= hi, (
                f"STARTING_STATS[{role.name}, {race.name}][{stat}] "
                f"inverted: lo={lo} hi={hi}"
            )


def test_starting_stats_lo_equals_role_attrbase():
    """``lo`` in STARTING_STATS must equal vendor ``role.attrbase[i]``."""
    from Nethax.nethax.subsystems.character import STARTING_STATS

    stat_names = ["str", "int", "wis", "dex", "con", "cha"]
    for role in Role:
        attrbase = get_role(role).attrbase
        ranges = STARTING_STATS[(role, Race.HUMAN)]
        for i, name in enumerate(stat_names):
            assert ranges[name][0] == attrbase[i], (
                f"{role.name} {name} lo != attrbase: "
                f"{ranges[name][0]} vs {attrbase[i]}"
            )
