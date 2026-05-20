"""Parity tests for the three action-handler fixes.

  - _handle_cast  : wires _EFFECT_DISPATCH_LIST via jax.lax.switch
  - _handle_name  : labels unidentified inventory slots "Item N"
  - _handle_enhance: uniform level*level*20 threshold (vendor weapon.c:1329)

Canonical sources:
  vendor/nethack/src/spell.c::spelleffects  — cast effect dispatch
  vendor/nethack/src/do_name.c::do_oname    — item naming
  vendor/nethack/src/weapon.c::enhance_weapon_skill line 1329
  vendor/nethack/include/skills.h:106       — practice_needed_to_advance
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.magic import SpellId, _SPELL_LEVELS
from Nethax.nethax.subsystems.skills import SkillId, SkillLevel
from Nethax.nethax.subsystems.inventory import (
    ItemCategory,
    MAX_INVENTORY_SLOTS,
    USER_NAME_LEN,
)
from Nethax.nethax.constants.actions import Command
from Nethax.nethax.subsystems.action_dispatch import (
    _handle_cast,
    _handle_name,
    _handle_enhance,
)

_RNG = jax.random.PRNGKey(0)


def _base_state(**over) -> EnvState:
    s = EnvState.default(_RNG)
    s = s.replace(
        player_pw=jnp.int32(over.pop("player_pw", 200)),
        player_pw_max=jnp.int32(over.pop("player_pw_max", 200)),
        player_hp=jnp.int32(over.pop("player_hp", 10)),
        player_hp_max=jnp.int32(over.pop("player_hp_max", 100)),
        player_xl=jnp.int32(over.pop("player_xl", 10)),
        player_role=jnp.int8(over.pop("player_role", 12)),  # wizard
        player_int=jnp.int8(over.pop("player_int", 18)),
        player_wis=jnp.int8(over.pop("player_wis", 18)),
    )
    for k, v in over.items():
        s = s.replace(**{k: v})
    return s


def _state_with_spell(spell_id: SpellId, pw: int = 200) -> EnvState:
    """Return a state with spell_id known and memory > 0."""
    s = _base_state(player_pw=pw)
    sid = int(spell_id)
    new_known = s.magic.spell_known.at[sid].set(True)
    new_mem = s.magic.spell_memory.at[sid].set(jnp.int32(100))
    new_magic = s.magic.replace(spell_known=new_known, spell_memory=new_mem)
    return s.replace(magic=new_magic)


# ---------------------------------------------------------------------------
# _handle_cast tests
# ---------------------------------------------------------------------------

def test_handle_cast_applies_spell_effect():
    """Cast SPE_HEALING via _handle_cast; player HP must increase.

    Cite: vendor/nethack/src/spell.c::spelleffects HEALING branch.
    """
    spell = SpellId.HEALING
    state = _state_with_spell(spell, pw=200)
    state = state.replace(player_hp=jnp.int32(10))

    new_state = _handle_cast(state, _RNG)

    assert int(new_state.player_hp) > int(state.player_hp), (
        f"HP should increase after HEALING cast; got {int(new_state.player_hp)} "
        f"from {int(state.player_hp)}"
    )
    # Pw must have been deducted
    pw_cost = int(_SPELL_LEVELS[int(spell)]) * 5
    assert int(new_state.player_pw) == int(state.player_pw) - pw_cost, (
        f"Expected Pw={int(state.player_pw) - pw_cost}, got {int(new_state.player_pw)}"
    )


def test_handle_cast_no_effect_when_unknown():
    """No spell known — _handle_cast returns state unchanged.

    Cite: vendor/nethack/src/spell.c::docast early-exit when no spell selected.
    """
    state = _base_state()  # all spells unknown / no memory

    new_state = _handle_cast(state, _RNG)

    assert int(new_state.player_pw) == int(state.player_pw), (
        f"Pw should not change when no spell known; got {int(new_state.player_pw)}"
    )
    assert int(new_state.player_hp) == int(state.player_hp)


# ---------------------------------------------------------------------------
# _handle_name tests
# ---------------------------------------------------------------------------

def _state_with_item(slot: int = 0) -> EnvState:
    """Place a non-identified item in inventory slot ``slot``."""
    s = _base_state()
    inv = s.inventory
    new_cat = inv.items.category.at[slot].set(jnp.int8(int(ItemCategory.WEAPON)))
    # identified defaults to False in EnvState.default; ensure it stays unidentified
    new_items = inv.items.replace(category=new_cat)
    new_inv = inv.replace(items=new_items)
    return s.replace(inventory=new_inv)


def test_handle_name_writes_user_name():
    """C command labels slot-0 unidentified item; user_names[0, 0] must be non-zero.

    Cite: vendor/nethack/src/do_name.c::do_oname.
    """
    state = _state_with_item(slot=0)

    new_state = _handle_name(state, _RNG)

    first_byte = int(new_state.inventory.user_names[0, 0])
    assert first_byte != 0, (
        f"Expected non-zero first byte in user_names[0] after naming, got {first_byte}"
    )


def test_handle_name_skips_identified_items():
    """Identified items must NOT be overwritten by the generic label."""
    s = _state_with_item(slot=0)
    inv = s.inventory
    new_id = inv.items.identified.at[0].set(True)
    new_items = inv.items.replace(identified=new_id)
    state = s.replace(inventory=inv.replace(items=new_items))

    new_state = _handle_name(state, _RNG)

    # user_names[0] should remain all-zero (identified item skipped)
    first_byte = int(new_state.inventory.user_names[0, 0])
    assert first_byte == 0, (
        f"Identified item should not be named; got first_byte={first_byte}"
    )


# ---------------------------------------------------------------------------
# _handle_enhance tests
# ---------------------------------------------------------------------------

def _state_with_skill(skill_id: SkillId, level: int, advance: int, cap: int) -> EnvState:
    """Return a state with the given skill configuration."""
    s = _base_state()
    sk = s.skills
    i = int(skill_id)
    new_level   = sk.level.at[i].set(jnp.int8(level))
    new_advance = sk.advance.at[i].set(jnp.int32(advance))
    new_max     = sk.max_level.at[i].set(jnp.int8(cap))
    new_sk = sk.replace(level=new_level, advance=new_advance, max_level=new_max)
    return s.replace(skills=new_sk)


def test_handle_enhance_advances_eligible():
    """DAGGER at UNSKILLED(0) with advance=20 == threshold(0)=20; should advance to BASIC(1).

    Cite: vendor/nethack/src/weapon.c::enhance_weapon_skill line 1329.
    Threshold formula: vendor practice_needed_to_advance(P_UNSKILLED=1) = 1*1*20 = 20
    (vendor/nethack/include/skills.h:106); in our 0-based encoding the
    macro evaluated at level=0 returns (0+1)*(0+1)*20 = 20.
    """
    state = _state_with_skill(
        SkillId.DAGGER,
        level=int(SkillLevel.P_UNSKILLED),
        advance=20,
        cap=int(SkillLevel.P_SKILLED),
    )

    new_state = _handle_enhance(state, _RNG)

    new_level = int(new_state.skills.level[int(SkillId.DAGGER)])
    assert new_level == int(SkillLevel.P_BASIC), (
        f"Expected DAGGER to advance to BASIC(1), got {new_level}"
    )


def test_handle_enhance_no_advance_when_capped():
    """Skill already at cap; E must leave level unchanged.

    Cite: vendor/nethack/src/weapon.c::enhance_weapon_skill — blocked when
    level >= max_level.
    """
    state = _state_with_skill(
        SkillId.DAGGER,
        level=int(SkillLevel.P_BASIC),
        advance=500,
        cap=int(SkillLevel.P_BASIC),
    )

    new_state = _handle_enhance(state, _RNG)

    new_level = int(new_state.skills.level[int(SkillId.DAGGER)])
    assert new_level == int(SkillLevel.P_BASIC), (
        f"Expected level to stay BASIC(1) when capped, got {new_level}"
    )
