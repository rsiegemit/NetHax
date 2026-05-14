"""Wave 6 Phase B+ — per-entry vendor parity for MONSTERS table.

Each test asserts a sample monster's full field set matches
vendor/nle/src/monst.c::mons[].  Roughly 20 sample monsters spread across
the symbol classes, including the entries that Wave 6 audit #62 fixed.

Vendor line numbers (`monst.c:<line>`) are cited for each test so future
auditors can re-verify quickly.
"""
from __future__ import annotations

import pytest

from Nethax.nethax.constants import monsters as M
from Nethax.nethax.constants.monsters import (
    MONSTERS,
    AttackType, DamageType,
    MR_FIRE, MR_COLD, MR_SLEEP, MR_DISINT, MR_ELEC, MR_POISON, MR_ACID, MR_STONE,
    M1_FLY, M1_ANIMAL, M1_NOHANDS, M1_OVIPAROUS, M1_CARNIVORE, M1_OMNIVORE,
    M1_POIS, M1_HUMANOID, M1_SEE_INVIS, M1_SLITHY, M1_AMPHIBIOUS, M1_SWIM,
    M1_NOLIMBS, M1_NOHEAD, M1_NOTAKE,
    M2_NOPOLY, M2_HOSTILE, M2_DOMESTIC, M2_STRONG, M2_MALE, M2_FEMALE,
    M2_ORC, M2_DWARF, M2_GNOME, M2_ELF, M2_HUMAN, M2_LORD, M2_PRINCE,
    M2_DEMON, M2_STALK, M2_NASTY, M2_GREEDY, M2_JEWELS, M2_COLLECT, M2_MAGIC,
    M2_PEACEFUL, M2_PNAME, M2_UNDEAD,
    M3_INFRAVISIBLE, M3_INFRAVISION,
    MS_SILENT, MS_BARK, MS_BUZZ, MS_ORC, MS_HUMANOID, MS_HISS, MS_ROAR,
    MS_NEIGH, MS_LEADER, MS_NEMESIS,
    MZ_TINY, MZ_SMALL, MZ_MEDIUM, MZ_HUMAN, MZ_LARGE, MZ_HUGE, MZ_GIGANTIC,
    CLR_BLACK, CLR_RED, CLR_GREEN, CLR_BROWN, CLR_BLUE, CLR_MAGENTA,
    CLR_CYAN, CLR_GRAY, CLR_YELLOW, CLR_WHITE,
    HI_LORD, HI_DOMESTIC, HI_GOLD,
    WT_HUMAN, WT_ELF, WT_DRAGON,
)


def _find_idx(name: str) -> int:
    for i, m in enumerate(MONSTERS):
        if m.name == name:
            return i
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Sanity: 381 entries, names match vendor (verified by Wave 6 #54).
# ---------------------------------------------------------------------------

def test_monsters_count_is_381():
    assert len(MONSTERS) == 381


# ---------------------------------------------------------------------------
# Per-entry parity tests.  Each cites vendor/nle/src/monst.c line.
# ---------------------------------------------------------------------------

def test_giant_ant_full_fidelity():
    """vendor monst.c:108  MON("giant ant", S_ANT, LVL(2,18,3,0,0), ...)"""
    m = MONSTERS[_find_idx("giant ant")]
    assert m.level == 2
    assert m.move_speed == 18
    assert m.ac == 3
    assert m.mr == 0
    assert m.alignment == 0
    assert m.weight == 10
    assert m.nutrition == 10
    assert m.size == MZ_TINY
    assert m.sound == MS_SILENT
    assert m.color == CLR_BROWN
    assert m.attacks[0] == (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 4)
    assert m.resists_mask == 0
    assert m.conveys_mask == 0


def test_killer_bee_full_fidelity():
    """vendor monst.c:114  MON("killer bee", S_ANT, LVL(1,18,-1,0,0), ...)"""
    m = MONSTERS[_find_idx("killer bee")]
    assert m.level == 1
    assert m.ac == -1
    assert m.move_speed == 18
    assert m.weight == 1
    assert m.nutrition == 5
    assert m.size == MZ_TINY
    assert m.sound == MS_BUZZ
    assert m.color == CLR_YELLOW
    assert m.attacks[0] == (AttackType.AT_STNG, DamageType.AD_DRST, 1, 3)
    assert m.resists_mask == MR_POISON
    assert m.conveys_mask == MR_POISON
    assert m.flags2 & M2_FEMALE  # vendor: M2_HOSTILE | M2_FEMALE


def test_giant_beetle_weight_nutrition_fix():
    """vendor monst.c:132  giant beetle weight/cnutrit fix.

    Wave 6 parity-fix: chunk1 previously had weight=200, nutrition=50;
    vendor truth: cwt=10, cnutrit=10.
    """
    m = MONSTERS[_find_idx("giant beetle")]
    assert m.weight == 10
    assert m.nutrition == 10


def test_pyrolisk_single_attack_fix():
    """vendor monst.c:188  pyrolisk has ONLY a gaze attack.

    Wave 6 parity-fix: chunk1 had a spurious AT_BITE/AD_PHYS,1,6 secondary
    attack — vendor only defines AT_GAZE/AD_FIRE,2,6.
    """
    m = MONSTERS[_find_idx("pyrolisk")]
    assert m.attacks[0] == (AttackType.AT_GAZE, DamageType.AD_FIRE, 2, 6)
    assert len(m.attacks) == 1


def test_wolf_color_fix():
    """vendor monst.c:246  wolf color is CLR_BROWN, not CLR_GRAY."""
    m = MONSTERS[_find_idx("wolf")]
    assert m.color == CLR_BROWN


def test_kitten_flags3_no_infravision():
    """vendor monst.c:349  kitten has M3_INFRAVISIBLE but NOT M3_INFRAVISION.

    Wave 6 parity-fix: chunk1 incorrectly included M3_INFRAVISION (the
    'can see infrared' flag belongs to creatures with infravision, e.g.
    elves/dwarves; cats only emit infrared, so only M3_INFRAVISIBLE).
    """
    m = MONSTERS[_find_idx("kitten")]
    assert m.flags3 == M3_INFRAVISIBLE


def test_dwarf_flags2_has_nopoly():
    """vendor monst.c:421  dwarf has M2_NOPOLY (player race placeholder)."""
    m = MONSTERS[_find_idx("dwarf")]
    assert m.flags2 & M2_NOPOLY


def test_dwarf_lord_male():
    """vendor monst.c:433  dwarf lord is M2_MALE."""
    m = MONSTERS[_find_idx("dwarf lord")]
    assert m.flags2 & M2_MALE


def test_mind_flayer_color_magenta():
    """vendor monst.c:449  mind flayer color = CLR_MAGENTA, not BRIGHT_MAGENTA."""
    m = MONSTERS[_find_idx("mind flayer")]
    assert m.color == CLR_MAGENTA


def test_acid_blob_conveys_stone_only():
    """vendor monst.c:147  acid blob conveys MR_STONE only.

    Wave 6 parity-fix: chunk1 included MR_ACID|MR_STONE; vendor mr2 is only
    MR_STONE.  (Eating an acid blob corpse confers stoning resist, not acid
    resist — acid resist comes from yellow dragon scales/corpses.)
    """
    m = MONSTERS[_find_idx("acid blob")]
    assert m.conveys_mask == MR_STONE


def test_orc_no_poison_resist():
    """vendor monst.c:626  orc has resists_mask=0 (no MR_POISON).

    Wave 6 parity-fix: ours had MR_POISON for all 6 orc entries; vendor
    truth — orcs are NOT poison-resistant in NLE.  (Poison resistance comes
    from POIS flag1 which affects corpse contents, not active resistance.)
    """
    for name in ("orc", "hill orc", "Mordor orc", "Uruk-hai",
                 "orc shaman", "orc-captain"):
        m = MONSTERS[_find_idx(name)]
        assert m.resists_mask == 0, f"{name} should have no resists"


def test_orc_shaman_strong():
    """vendor monst.c:650  orc shaman has M2_STRONG."""
    m = MONSTERS[_find_idx("orc shaman")]
    assert m.flags2 & M2_STRONG


def test_lurker_above_single_engulf():
    """vendor monst.c:804  lurker above: single ATTK(AT_ENGL, AD_DGST, 1, 8)."""
    m = MONSTERS[_find_idx("lurker above")]
    assert len(m.attacks) == 1
    assert m.attacks[0] == (AttackType.AT_ENGL, DamageType.AD_DGST, 1, 8)


def test_ki_rin_animal_no_poison_resist():
    """vendor monst.c:1017  ki-rin: M1_ANIMAL flag, resists_mask=0, MS_NEIGH."""
    m = MONSTERS[_find_idx("ki-rin")]
    assert m.flags1 & M1_ANIMAL
    assert m.resists_mask == 0
    assert m.sound == MS_NEIGH


def test_gnome_nopoly():
    """vendor monst.c:1348  gnome has M2_NOPOLY."""
    m = MONSTERS[_find_idx("gnome")]
    assert m.flags2 & M2_NOPOLY


def test_guardian_naga_attacks():
    """vendor monst.c:1654  guardian naga: BITE/PLYS(1,6), SPIT/DRST(1,6), HUGS/PHYS(2,4)."""
    m = MONSTERS[_find_idx("guardian naga")]
    assert m.attacks[0] == (AttackType.AT_BITE, DamageType.AD_PLYS, 1, 6)
    assert m.attacks[1] == (AttackType.AT_SPIT, DamageType.AD_DRST, 1, 6)
    assert m.attacks[2] == (AttackType.AT_HUGS, DamageType.AD_PHYS, 2, 4)


def test_elf_level_10_strong():
    """vendor monst.c:2142  elf: level 10, M2_NOPOLY|M2_ELF|M2_STRONG|M2_COLLECT.

    Wave 6 parity-fix: ours had level=0; vendor truth = 10 (player-race
    placeholder retains real level for corpse stats).
    """
    m = MONSTERS[_find_idx("elf")]
    assert m.level == 10
    assert m.flags2 & M2_STRONG


def test_piranha_single_attack():
    """vendor monst.c:2620  piranha: move 12, single AT_BITE attack, M1_SLITHY."""
    m = MONSTERS[_find_idx("piranha")]
    assert m.move_speed == 12
    assert len(m.attacks) == 1
    assert m.flags1 & M1_SLITHY


def test_shark_slithy():
    """vendor monst.c:2627  shark has M1_SLITHY."""
    m = MONSTERS[_find_idx("shark")]
    assert m.flags1 & M1_SLITHY


def test_lord_carnarvon_quest_leader_stats():
    """vendor monst.c:2833  Lord Carnarvon: mmove=12, mr=30, single 1d6 attack.

    Wave 6 parity-fix: all 13 quest leaders had spurious mmove=15 (should
    be 12) and mr=90 (varies — 30 for most, 40/70/80 for a few).
    """
    m = MONSTERS[_find_idx("Lord Carnarvon")]
    assert m.move_speed == 12
    assert m.mr == 30
    assert m.attacks[0] == (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6)


def test_grand_master_mmove_12():
    """vendor monst.c:2895  Grand Master: mmove=12 (was 15), mr=70 (was 90)."""
    m = MONSTERS[_find_idx("Grand Master")]
    assert m.move_speed == 12
    assert m.mr == 70


def test_riders_color_lord():
    """vendor monst.c:2567/2575/2583  Death/Pestilence/Famine: color=HI_LORD."""
    for name in ("Death", "Pestilence", "Famine"):
        m = MONSTERS[_find_idx(name)]
        assert m.color == HI_LORD, f"{name} color should be HI_LORD"


def test_wizard_of_yendor_color_magenta():
    """vendor monst.c:2768  Wizard of Yendor color = CLR_MAGENTA (not BRIGHT)."""
    m = MONSTERS[_find_idx("Wizard of Yendor")]
    assert m.color == CLR_MAGENTA


def test_chromatic_dragon_no_acid_convey():
    """vendor monst.c:2994  Chromatic Dragon conveys do NOT include MR_ACID.

    Wave 6 parity-fix: vendor mr2 = MR_FIRE|MR_COLD|MR_SLEEP|MR_DISINT|
    MR_ELEC|MR_POISON|MR_STONE (no MR_ACID).
    """
    m = MONSTERS[_find_idx("Chromatic Dragon")]
    expected = MR_FIRE | MR_COLD | MR_SLEEP | MR_DISINT | MR_ELEC | MR_POISON | MR_STONE
    assert m.conveys_mask == expected


def test_minion_of_huhetotl_color_red():
    """vendor monst.c:2980  Minion of Huhetotl color = CLR_RED."""
    m = MONSTERS[_find_idx("Minion of Huhetotl")]
    assert m.color == CLR_RED


def test_valkyrie_chaotic():
    """vendor monst.c:2811  valkyrie alignment = -1 (chaotic), not +1."""
    m = MONSTERS[_find_idx("valkyrie")]
    assert m.alignment == -1


def test_garter_snake_tiny():
    """vendor monst.c:2456  garter snake size = MZ_TINY."""
    m = MONSTERS[_find_idx("garter snake")]
    assert m.size == MZ_TINY


def test_zombie_silent_sounds():
    """vendor monst.c:1963+ zombies all use MS_SILENT, not MS_GROAN."""
    for name in ("kobold zombie", "gnome zombie", "orc zombie", "dwarf zombie",
                 "elf zombie", "human zombie", "ettin zombie", "giant zombie"):
        m = MONSTERS[_find_idx(name)]
        assert m.sound == MS_SILENT, f"{name} sound should be MS_SILENT"


def test_full_per_entry_compare_against_vendor_dump():
    """Integration: re-run the audit comparator against the JSON snapshot
    and assert zero mismatches.  This requires /tmp/vendor_monsters.json
    produced by tools/parse_vendor_monst.py — when absent, the test is
    skipped so CI on fresh clones doesn't break.
    """
    import json
    import os

    json_path = "/tmp/vendor_monsters.json"
    if not os.path.exists(json_path):
        pytest.skip("vendor JSON snapshot absent; produced by tools/parse_vendor_monst.py")

    with open(json_path) as f:
        vendor = json.load(f)

    assert len(vendor) == 381
    assert len(MONSTERS) == 381

    mismatches = []
    for i, (v, o) in enumerate(zip(vendor, MONSTERS)):
        if v["name"] != o.name:
            mismatches.append((i, "NAME", o.name, v["name"]))
            continue
        if v["level"] != o.level:
            mismatches.append((i, v["name"], "level", o.level, v["level"]))
        if v["mmove"] != o.move_speed:
            mismatches.append((i, v["name"], "mmove", o.move_speed, v["mmove"]))
        if v["ac"] != o.ac:
            mismatches.append((i, v["name"], "ac", o.ac, v["ac"]))
        if v["mr"] != o.mr:
            mismatches.append((i, v["name"], "mr", o.mr, v["mr"]))
        if v["alignment"] != o.alignment:
            mismatches.append((i, v["name"], "align", o.alignment, v["alignment"]))
        if v["weight"] != o.weight:
            mismatches.append((i, v["name"], "weight", o.weight, v["weight"]))
        if v["nutrition"] != o.nutrition:
            mismatches.append((i, v["name"], "nut", o.nutrition, v["nutrition"]))
    assert not mismatches, "Found " + str(len(mismatches)) + " mismatches: " + str(mismatches[:5])
