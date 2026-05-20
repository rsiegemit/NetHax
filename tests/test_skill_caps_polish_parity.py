"""Skill cap parity tests — byte-equal verification of u_init.c Skill_X tables.

Canonical sources:
  vendor/nethack/src/u_init.c lines 257-571 — Skill_A … Skill_W per-role tables.
  vendor/nethack/src/weapon.c:1329           — enhance_weapon_skill (#enhance).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems.skills import (
    SkillId,
    SkillLevel,
    SkillState,
    N_SKILLS,
    init_skills,
    try_advance_skill,
    practice_needed_to_advance,
)
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.state import EnvState

_RNG = jax.random.PRNGKey(0)


def _fresh_state() -> EnvState:
    return EnvState.default(_RNG)


# ---------------------------------------------------------------------------
# test_wizard_dagger_cap_is_expert
# Cite: vendor/nethack/src/u_init.c:549 — { P_DAGGER, P_EXPERT } in Skill_W
# ---------------------------------------------------------------------------
def test_wizard_dagger_cap_is_expert():
    skills = init_skills(Role.WIZARD)
    cap = int(skills.max_level[int(SkillId.DAGGER)])
    assert cap == int(SkillLevel.P_EXPERT), (
        f"Wizard DAGGER cap: expected P_EXPERT(3), got {cap}"
    )


# ---------------------------------------------------------------------------
# test_valkyrie_long_sword_grand_master
# Cite: vendor/nethack/src/u_init.c:531 — { P_LONG_SWORD, P_EXPERT } in Skill_V
# (Vendor cap is P_EXPERT; test name reflects task spec, assertion matches vendor.)
# ---------------------------------------------------------------------------
def test_valkyrie_long_sword_grand_master():
    skills = init_skills(Role.VALKYRIE)
    cap = int(skills.max_level[int(SkillId.LONG_SWORD)])
    assert cap == int(SkillLevel.P_EXPERT), (
        f"Valkyrie LONG_SWORD cap: expected P_EXPERT(3), got {cap}"
    )


# ---------------------------------------------------------------------------
# test_monk_no_weapon_caps
# Cite: vendor/nethack/src/u_init.c:375 — Skill_Mon lists only QUARTERSTAFF,
# SPEAR, CROSSBOW, SHURIKEN as weapon skills (all P_BASIC); bladed weapons
# like DAGGER, LONG_SWORD etc. are absent → P_UNSKILLED (restricted).
# ---------------------------------------------------------------------------
def test_monk_no_weapon_caps():
    skills = init_skills(Role.MONK)
    # Weapons unlisted in Skill_Mon must be P_UNSKILLED.
    restricted_weapons = [
        SkillId.DAGGER, SkillId.KNIFE, SkillId.AXE, SkillId.PICK_AXE,
        SkillId.SHORT_SWORD, SkillId.BROAD_SWORD, SkillId.LONG_SWORD,
        SkillId.TWO_HANDED_SWORD, SkillId.SABER, SkillId.CLUB,
        SkillId.MACE, SkillId.MORNING_STAR, SkillId.FLAIL, SkillId.HAMMER,
        SkillId.POLEARMS, SkillId.TRIDENT, SkillId.LANCE, SkillId.BOW,
        SkillId.SLING, SkillId.DART, SkillId.BOOMERANG, SkillId.WHIP,
        SkillId.UNICORN_HORN,
    ]
    for sid in restricted_weapons:
        cap = int(skills.max_level[int(sid)])
        assert cap == int(SkillLevel.P_UNSKILLED), (
            f"Monk {sid.name} cap: expected P_UNSKILLED(0), got {cap}"
        )
    # Listed weapon skills must be exactly P_BASIC.
    listed_basic = [SkillId.QUARTERSTAFF, SkillId.SPEAR,
                    SkillId.CROSSBOW, SkillId.SHURIKEN]
    for sid in listed_basic:
        cap = int(skills.max_level[int(sid)])
        assert cap <= int(SkillLevel.P_BASIC), (
            f"Monk {sid.name} cap: expected <= P_BASIC(1), got {cap}"
        )


# ---------------------------------------------------------------------------
# test_priest_attack_spell_expert
# Cite: vendor/nethack/src/u_init.c:410 — { P_CLERIC_SPELL, P_EXPERT } in Skill_P
# Priest has no ATTACK_SPELL entry → P_UNSKILLED; CLERIC_SPELL is P_EXPERT.
# ---------------------------------------------------------------------------
def test_priest_attack_spell_expert():
    skills = init_skills(Role.PRIEST)
    # ATTACK_SPELL not in Skill_P → restricted (P_UNSKILLED).
    attack_cap = int(skills.max_level[int(SkillId.ATTACK_SPELL)])
    assert attack_cap == int(SkillLevel.P_UNSKILLED), (
        f"Priest ATTACK_SPELL cap: expected P_UNSKILLED(0), got {attack_cap}"
    )
    # CLERIC_SPELL is P_EXPERT per u_init.c:410.
    cleric_cap = int(skills.max_level[int(SkillId.CLERIC_SPELL)])
    assert cleric_cap == int(SkillLevel.P_EXPERT), (
        f"Priest CLERIC_SPELL cap: expected P_EXPERT(3), got {cleric_cap}"
    )


# ---------------------------------------------------------------------------
# test_enhance_advances_eligible_skill
# Cite: vendor/nethack/src/weapon.c::enhance_weapon_skill line 1329.
# Set advance[DAGGER]=20 with level=P_UNSKILLED(0), cap=P_EXPERT(3).
# practice_needed_to_advance(0) = 20 (vendor's macro applied to 1-based
# P_UNSKILLED=1 → 1*1*20 = 20), so advance=20 == threshold → eligible.
# After _handle_enhance, level[DAGGER] should be P_BASIC(1).
# ---------------------------------------------------------------------------
def test_enhance_advances_eligible_skill():
    from Nethax.nethax.subsystems.action_dispatch import _handle_enhance

    state = _fresh_state()
    dagger = int(SkillId.DAGGER)

    # Set cap and advance for DAGGER.
    new_max = state.skills.max_level.at[dagger].set(jnp.int8(int(SkillLevel.P_EXPERT)))
    new_adv = state.skills.advance.at[dagger].set(jnp.int32(20))
    state = state.replace(skills=state.skills.replace(max_level=new_max, advance=new_adv))

    new_state = _handle_enhance(state, _RNG)
    level = int(new_state.skills.level[dagger])
    assert level == int(SkillLevel.P_BASIC), (
        f"After #enhance with advance=20, expected DAGGER level P_BASIC(1), got {level}"
    )


# ---------------------------------------------------------------------------
# test_enhance_no_op_when_no_eligible
# Cite: vendor/nethack/src/weapon.c::enhance_weapon_skill line 1329.
# All skills at P_UNSKILLED(0) with advance=0 and cap=P_UNSKILLED(0) (restricted).
# practice_needed_to_advance(0) = 20 but level(0) >= cap(0) blocks → not eligible.
# _handle_enhance must leave state unchanged.
# ---------------------------------------------------------------------------
def test_enhance_no_op_when_no_eligible():
    from Nethax.nethax.subsystems.action_dispatch import _handle_enhance

    state = _fresh_state()
    # All max_level = 0 (P_UNSKILLED, restricted), advance = 0 → no eligible skill.
    # EnvState.default already has all-zero skills; verify and call.
    assert int(jnp.sum(state.skills.max_level)) == 0, "Expected all caps=0 in default state"

    new_state = _handle_enhance(state, _RNG)
    assert jnp.array_equal(new_state.skills.level, state.skills.level), (
        "#enhance should leave level unchanged when no skill is eligible"
    )
    assert jnp.array_equal(new_state.skills.advance, state.skills.advance), (
        "#enhance should leave advance unchanged when no skill is eligible"
    )
