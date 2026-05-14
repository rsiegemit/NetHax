"""Wave 6 Closing-Audit #74 — MonsterEntry.difficulty + ObjectEntry secondary
fields (oc_skill, oc_armor_class, oc_charge).

Ground truth:
    - vendor/nle/src/monst.c (MON() macro `d` arg, lines 47-50)
    - vendor/nle/src/objects.c (OBJECT/WEAPON/ARMOR/WAND/TOOL macros)
    - vendor/nle/src/mkobj.c    (wand charge initialisation, lines 1019-1027)
    - vendor/nle/src/read.c     (recharge ceilings, lines 478-480)
    - vendor/nethack/include/skills.h    (P_* enum)
    - vendor/nethack/include/objclass.h  (ARM_* enum, oc_subtyp definition)
"""
from __future__ import annotations

import pytest

from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.constants.objects import OBJECTS, ObjectClass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _monster_by_name(name: str):
    for m in MONSTERS:
        if m.name == name:
            return m
    raise KeyError(name)


def _object_by_name(name: str, cls: ObjectClass | None = None):
    """First OBJECTS entry matching ``name`` (and ``cls`` if given)."""
    for o in OBJECTS:
        if o.name == name and (cls is None or o.class_ == cls):
            return o
    raise KeyError(f"{name!r} (class={cls})")


# ---------------------------------------------------------------------------
# Mission 1: MonsterEntry.difficulty
# ---------------------------------------------------------------------------


def test_giant_rat_difficulty_matches_vendor():
    """vendor/nle/src/monst.c — giant rat MON() entry, trailing `d` arg = 2.

    Verifies the vendor difficulty rating is preserved on MonsterEntry.
    """
    m = _monster_by_name("giant rat")
    assert hasattr(m, "difficulty")
    # Vendor monst.c giant rat: MON("giant rat", ..., 2, CLR_BROWN)
    assert m.difficulty == 2, f"giant rat difficulty {m.difficulty} != vendor 2"


def test_gold_dragon_difficulty_higher_than_baby_dragon():
    """vendor/nle/src/monst.c — adult dragons (difficulty=20) outrank
    baby dragons (difficulty=13). Uses yellow dragon as the closest-to-gold
    variant present in NetHack 3.7 vendor (no literal 'gold dragon' species).
    """
    baby = _monster_by_name("baby yellow dragon")
    adult = _monster_by_name("yellow dragon")
    assert adult.difficulty > baby.difficulty, (
        f"adult {adult.difficulty} should be > baby {baby.difficulty}"
    )


def test_all_381_monsters_have_difficulty_field():
    """Every MonsterEntry exposes a difficulty attribute (Wave 6 closing audit
    adds this to all 381 vendor entries)."""
    assert len(MONSTERS) == 381
    missing = [m.name for m in MONSTERS if not hasattr(m, "difficulty")]
    assert not missing, f"entries without difficulty: {missing[:10]}"
    # And it must be an int (sentinel 0 allowed but typed).
    bad_type = [m.name for m in MONSTERS if not isinstance(m.difficulty, int)]
    assert not bad_type, f"non-int difficulty: {bad_type[:10]}"


def test_difficulty_used_for_spawn_eligibility():
    """``dungeon/spawning.py::MONSTR_DIFFICULTIES`` must reflect the vendor
    ``difficulty`` field when it's populated.

    Citation: vendor/nle/src/monst.c MON() macro `d` arg drives
    makemon.c::rndmonst() depth-window filtering.
    """
    from Nethax.nethax.dungeon.spawning import MONSTR_DIFFICULTIES

    # Pick a few well-known entries whose vendor difficulty is non-zero.
    samples = {
        "giant ant": 4,
        "killer bee": 5,
        "soldier ant": 6,
    }
    for name, expected in samples.items():
        idx = next(i for i, m in enumerate(MONSTERS) if m.name == name)
        actual = int(MONSTR_DIFFICULTIES[idx])
        assert actual == expected, (
            f"{name}: MONSTR_DIFFICULTIES[{idx}]={actual}, expected vendor {expected}"
        )


# ---------------------------------------------------------------------------
# Mission 2: OBJECT secondary fields (oc_skill)
# ---------------------------------------------------------------------------
# Citations: vendor/nethack/include/skills.h enum p_skills
#            vendor/nle/src/objects.c WEAPON() macro sub-arg (line 87-92)


def test_long_sword_skill_is_p_long_sword():
    """vendor/nle/src/objects.c — WEAPON("long sword", ...) has sub=P_LONG_SWORD.
    skills.h: P_LONG_SWORD = 7."""
    o = _object_by_name("long sword", ObjectClass.WEAPON_CLASS)
    assert o.oc_skill == 7, f"long sword oc_skill {o.oc_skill} != P_LONG_SWORD (7)"


def test_bow_skill_is_p_bow():
    """vendor/nle/src/objects.c — BOW("bow", ...) has sub=P_BOW.
    skills.h: P_BOW = 20."""
    o = _object_by_name("bow", ObjectClass.WEAPON_CLASS)
    assert o.oc_skill == 20, f"bow oc_skill {o.oc_skill} != P_BOW (20)"


def test_quarterstaff_skill_is_p_quarterstaff():
    """vendor/nle/src/objects.c — WEAPON("quarterstaff", ...) has
    sub=P_QUARTERSTAFF. skills.h: P_QUARTERSTAFF = 15."""
    o = _object_by_name("quarterstaff", ObjectClass.WEAPON_CLASS)
    assert o.oc_skill == 15, (
        f"quarterstaff oc_skill {o.oc_skill} != P_QUARTERSTAFF (15)"
    )


# ---------------------------------------------------------------------------
# Mission 2: OBJECT secondary fields (oc_armor_class)
# ---------------------------------------------------------------------------
# Citations: vendor/nethack/include/objclass.h enum obj_armor_types (line 37-45)
#            vendor/nle/src/objects.c HELM/SHIELD/ARMOR macros


def test_helmet_armor_class_arm_helm():
    """vendor/nle/src/objects.c — HELM("helmet", ...) passes ARM_HELM=2."""
    o = _object_by_name("helmet", ObjectClass.ARMOR_CLASS)
    assert o.oc_armor_class == 2, (
        f"helmet oc_armor_class {o.oc_armor_class} != ARM_HELM (2)"
    )


def test_chain_mail_armor_class_arm_suit():
    """vendor/nle/src/objects.c — ARMOR("chain mail", ..., ARM_SUIT=0, ...)."""
    o = _object_by_name("chain mail", ObjectClass.ARMOR_CLASS)
    assert o.oc_armor_class == 0, (
        f"chain mail oc_armor_class {o.oc_armor_class} != ARM_SUIT (0)"
    )


def test_small_shield_armor_class_arm_shield():
    """vendor/nle/src/objects.c — SHIELD("small shield", ...) passes
    ARM_SHIELD=1."""
    o = _object_by_name("small shield", ObjectClass.ARMOR_CLASS)
    assert o.oc_armor_class == 1, (
        f"small shield oc_armor_class {o.oc_armor_class} != ARM_SHIELD (1)"
    )


# ---------------------------------------------------------------------------
# Mission 2: OBJECT secondary fields (oc_charge)
# ---------------------------------------------------------------------------
# Citations: vendor/nle/src/mkobj.c lines 1019-1024 (wand init)
#            vendor/nle/src/read.c   lines 478-480 (recharge ceiling)


def test_wand_of_wishing_max_charges_3():
    """vendor/nle/src/mkobj.c:1020-1021 — WAN_WISHING -> rnd(3) => max 3.
    vendor/nle/src/read.c:478 — lim = (WAN_WISHING) ? 3 : ..."""
    o = _object_by_name("wishing", ObjectClass.WAND_CLASS)
    assert o.oc_charge == 3, (
        f"wand of wishing oc_charge {o.oc_charge} != 3 (vendor cap)"
    )


def test_magic_marker_max_charges_50():
    """vendor/nle/src/read.c:604 — blessed-recharge ceiling for magic marker
    is 50. (mkobj.c initialises rn1(70,30); recharge caps at 50.)"""
    o = _object_by_name("magic marker", ObjectClass.TOOL_CLASS)
    assert o.oc_charge == 50, (
        f"magic marker oc_charge {o.oc_charge} != 50 (read.c recharge ceiling)"
    )


def test_wand_of_light_max_charges_15():
    """vendor/nle/src/read.c:478-480 — NODIR wands have lim=15.
    `wand of light` is NODIR (objects.c WAND row 1)."""
    o = _object_by_name("light", ObjectClass.WAND_CLASS)
    assert o.oc_charge == 15, (
        f"wand of light oc_charge {o.oc_charge} != 15 (NODIR cap)"
    )
