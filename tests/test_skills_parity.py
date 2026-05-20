"""Skills subsystem parity tests.

Verifies that SkillState, use_skill, try_advance_skill, practice_needed_to_advance,
init_skills, and the combat/magic integrations behave per vendor semantics.

Canonical sources:
  vendor/nethack/include/skills.h:106  — practice_needed_to_advance formula
  vendor/nethack/src/weapon.c:1424     — use_skill
  vendor/nethack/src/weapon.c::skill_advance — try_advance_skill
  vendor/nethack/src/u_init.c          — per-role skill caps (Skill_V, Skill_W)
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
    practice_needed_to_advance,
    use_skill,
    try_advance_skill,
    init_skills,
    _SPELL_SCHOOL_TO_SKILL_ID,
)
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.state import EnvState

_RNG = jax.random.PRNGKey(42)


def _fresh_state() -> EnvState:
    return EnvState.default(_RNG)


# ---------------------------------------------------------------------------
# test_practice_needed_to_advance_formula
# vendor/nethack/include/skills.h:106 — level*level*20
# ---------------------------------------------------------------------------
def test_practice_needed_to_advance_formula():
    # Vendor's macro is (L)*(L)*20 keyed off vendor's 1-based level
    # (P_UNSKILLED=1 … P_GRAND_MASTER=6).  In our 0-based encoding we
    # evaluate at (level+1)^2 * 20 so the output sequence matches vendor
    # for every reachable Nethax level.
    assert int(practice_needed_to_advance(jnp.int8(0))) == 20   # P_UNSKILLED → P_BASIC
    assert int(practice_needed_to_advance(jnp.int8(1))) == 80   # P_BASIC → P_SKILLED
    assert int(practice_needed_to_advance(jnp.int8(2))) == 180  # P_SKILLED → P_EXPERT
    assert int(practice_needed_to_advance(jnp.int8(3))) == 320  # P_EXPERT → P_MASTER
    assert int(practice_needed_to_advance(jnp.int8(4))) == 500  # P_MASTER → P_GRAND_MASTER


# ---------------------------------------------------------------------------
# test_starting_wizard_skill_caps
# Wizard: LONG_SWORD max = BASIC, ATTACK_SPELL max = EXPERT
# Cite: vendor/nethack/src/u_init.c Skill_W (lines 548-571)
# ---------------------------------------------------------------------------
def test_starting_wizard_skill_caps():
    skills = init_skills(Role.WIZARD)
    long_sword_cap = int(skills.max_level[int(SkillId.LONG_SWORD)])
    attack_spell_cap = int(skills.max_level[int(SkillId.ATTACK_SPELL)])
    assert long_sword_cap == int(SkillLevel.P_UNSKILLED), (
        f"Wizard LONG_SWORD cap: expected P_UNSKILLED(0), got {long_sword_cap}"
    )
    assert attack_spell_cap == int(SkillLevel.P_EXPERT), (
        f"Wizard ATTACK_SPELL cap: expected P_EXPERT(3), got {attack_spell_cap}"
    )


# ---------------------------------------------------------------------------
# test_starting_valkyrie_skill_caps
# Valkyrie: LONG_SWORD max = EXPERT, ATTACK_SPELL max = BASIC
# (Valkyrie has P_ATTACK_SPELL: P_BASIC per Skill_V u_init.c:541)
# Cite: vendor/nethack/src/u_init.c Skill_V (lines 525-546)
# ---------------------------------------------------------------------------
def test_starting_valkyrie_skill_caps():
    skills = init_skills(Role.VALKYRIE)
    long_sword_cap = int(skills.max_level[int(SkillId.LONG_SWORD)])
    attack_spell_cap = int(skills.max_level[int(SkillId.ATTACK_SPELL)])
    assert long_sword_cap == int(SkillLevel.P_EXPERT), (
        f"Valkyrie LONG_SWORD cap: expected P_EXPERT(3), got {long_sword_cap}"
    )
    assert attack_spell_cap == int(SkillLevel.P_BASIC), (
        f"Valkyrie ATTACK_SPELL cap: expected P_BASIC(1), got {attack_spell_cap}"
    )


# ---------------------------------------------------------------------------
# test_use_skill_increments_advance
# Simulate 20 calls to use_skill for DAGGER; advance[DAGGER] == 20.
# Cite: vendor/nethack/src/weapon.c:1424
# ---------------------------------------------------------------------------
def test_use_skill_increments_advance():
    state = _fresh_state()
    dagger_id = jnp.int32(int(SkillId.DAGGER))
    # Vendor weapon.c:1428 gates use_skill on !P_RESTRICTED(skill); in our
    # encoding "restricted" means max_level <= P_UNSKILLED.  Unlock DAGGER
    # by raising its cap before practicing.
    new_max = state.skills.max_level.at[int(SkillId.DAGGER)].set(
        jnp.int8(int(SkillLevel.P_BASIC))
    )
    state = state.replace(skills=state.skills.replace(max_level=new_max))
    for _ in range(20):
        state = use_skill(state, dagger_id, 1)
    advance = int(state.skills.advance[int(SkillId.DAGGER)])
    assert advance == 20, f"Expected advance[DAGGER]==20, got {advance}"


# ---------------------------------------------------------------------------
# test_try_advance_at_threshold
# advance=20, level=P_UNSKILLED(0), max=P_SKILLED(2) → level becomes P_BASIC(1)
# practice_needed_to_advance(0) = 20 (vendor's macro applied to 1-based
# P_UNSKILLED=1 → 1*1*20 = 20), so advance==20 exactly meets the threshold.
# Cite: vendor/nethack/src/weapon.c::skill_advance + skills.h:106
# ---------------------------------------------------------------------------
def test_try_advance_at_threshold():
    state = _fresh_state()
    # Set advance[DAGGER]=20, max_level[DAGGER]=P_SKILLED
    dagger = int(SkillId.DAGGER)
    new_advance = state.skills.advance.at[dagger].set(jnp.int32(20))
    new_max = state.skills.max_level.at[dagger].set(jnp.int8(int(SkillLevel.P_SKILLED)))
    state = state.replace(skills=state.skills.replace(advance=new_advance, max_level=new_max))

    state = try_advance_skill(state, jnp.int32(dagger))
    level = int(state.skills.level[dagger])
    assert level == int(SkillLevel.P_BASIC), (
        f"Expected P_BASIC(1) after advance from P_UNSKILLED with 20 practice, got {level}"
    )


# ---------------------------------------------------------------------------
# test_try_advance_at_cap_blocked
# advance=400, level=P_EXPERT(3), max=P_BASIC(1) → level stays P_EXPERT(3) — capped.
# Cite: vendor/nethack/src/weapon.c::skill_advance — blocked when level >= max_skill
# ---------------------------------------------------------------------------
def test_try_advance_at_cap_blocked():
    state = _fresh_state()
    dagger = int(SkillId.DAGGER)
    new_level   = state.skills.level.at[dagger].set(jnp.int8(int(SkillLevel.P_EXPERT)))
    new_advance = state.skills.advance.at[dagger].set(jnp.int32(400))
    new_max     = state.skills.max_level.at[dagger].set(jnp.int8(int(SkillLevel.P_BASIC)))
    state = state.replace(skills=state.skills.replace(
        level=new_level, advance=new_advance, max_level=new_max
    ))
    state = try_advance_skill(state, jnp.int32(dagger))
    level = int(state.skills.level[dagger])
    assert level == int(SkillLevel.P_EXPERT), (
        f"Expected level to stay P_EXPERT(3) when capped at BASIC(1), got {level}"
    )


# ---------------------------------------------------------------------------
# test_spell_cast_increments_skill
# Cast MAGIC_MISSILE (SpellId=1, school=ATTACK_SPELL) once; advance[ATTACK_SPELL] >= 1.
# ---------------------------------------------------------------------------
def test_spell_cast_increments_skill():
    from Nethax.nethax.subsystems.magic import cast_spell, SpellId
    import jax.random as jr

    state = _fresh_state()
    # Give enough Pw to cast (magic missile level=2, cost=10)
    state = state.replace(
        player_pw=jnp.int32(50),
        player_pw_max=jnp.int32(50),
        player_xl=jnp.int32(10),
        player_int=jnp.int8(18),
    )
    # Give spell memory so cast can proceed
    new_mem = state.magic.spell_memory.at[int(SpellId.MAGIC_MISSILE)].set(jnp.int32(100))
    new_known = state.magic.spell_known.at[int(SpellId.MAGIC_MISSILE)].set(jnp.bool_(True))
    state = state.replace(magic=state.magic.replace(spell_memory=new_mem, spell_known=new_known))

    # Unlock ATTACK_SPELL skill (vendor weapon.c:1428 — use_skill skips
    # restricted skills, i.e. max_level <= P_UNSKILLED in our encoding).
    new_max = state.skills.max_level.at[int(SkillId.ATTACK_SPELL)].set(
        jnp.int8(int(SkillLevel.P_EXPERT))
    )
    state = state.replace(skills=state.skills.replace(max_level=new_max))

    rng = jr.PRNGKey(0)
    new_state, _success = cast_spell(state, rng, SpellId.MAGIC_MISSILE)
    advance = int(new_state.skills.advance[int(SkillId.ATTACK_SPELL)])
    assert advance >= 1, f"Expected advance[ATTACK_SPELL] >= 1 after casting magic missile, got {advance}"
