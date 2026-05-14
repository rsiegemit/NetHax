"""Wave 3 combat tests.

Tests:
    - AC computation: stripped player → 10
    - Wearing leather armor (AC bonus 2) → 8
    - To-hit: high-STR fighter vs AC=10 monster, expect >= 75% hit over 100 rolls
    - Damage roll: dagger sdam=(1,4), 100 rolls average ~2.5 (with no STR bonus)
    - Bump-attack on adjacent monster reduces HP
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.combat import (
    PLAYER_BASE_AC,
    compute_ac,
    to_hit_roll,
    damage_roll,
    bump_attack,
    melee_attack,
    practice_skill,
    SKILL_BASIC,
)
from Nethax.nethax.subsystems.inventory import ArmorSlot, ItemCategory


_RNG = jax.random.PRNGKey(0)


def _fresh_state() -> EnvState:
    return EnvState.default(_RNG)


# ---------------------------------------------------------------------------
# AC
# ---------------------------------------------------------------------------

def test_ac_stripped_player_is_10():
    state = _fresh_state()
    assert int(compute_ac(state)) == PLAYER_BASE_AC


def _equip_armor(state, slot: ArmorSlot, ac_bonus: int, inv_slot: int = 0):
    """Place an armor item at inv_slot[inv_slot] with given ac_bonus, then
    point worn_armor[slot] at it."""
    items = state.inventory.items
    items = items.replace(
        category=items.category.at[inv_slot].set(jnp.int8(ItemCategory.ARMOR)),
        ac_bonus=items.ac_bonus.at[inv_slot].set(jnp.int8(ac_bonus)),
    )
    worn = state.inventory.worn_armor.at[int(slot)].set(jnp.int8(inv_slot))
    return state.replace(
        inventory=state.inventory.replace(items=items, worn_armor=worn),
    )


def test_ac_leather_armor_drops_to_8():
    """Leather armor has AC bonus 2 (vendor objects.c).  AC = 10 - 2 = 8."""
    state = _fresh_state()
    state = _equip_armor(state, ArmorSlot.BODY, ac_bonus=2, inv_slot=0)
    assert int(compute_ac(state)) == 8


def test_ac_full_armor_stacks_across_slots():
    """Body 2 + shield 1 + helm 1 + cloak 1 = 5 → AC 5."""
    state = _fresh_state()
    state = _equip_armor(state, ArmorSlot.BODY,   ac_bonus=2, inv_slot=0)
    state = _equip_armor(state, ArmorSlot.SHIELD, ac_bonus=1, inv_slot=1)
    state = _equip_armor(state, ArmorSlot.HELM,   ac_bonus=1, inv_slot=2)
    state = _equip_armor(state, ArmorSlot.CLOAK,  ac_bonus=1, inv_slot=3)
    assert int(compute_ac(state)) == 5


# ---------------------------------------------------------------------------
# To-hit
# ---------------------------------------------------------------------------

def test_to_hit_high_str_fighter_vs_ac10_monster():
    """High-STR / high-DEX fighter vs AC=10 monster → >=75% hits over 100 rolls.

    # Wave 6 parity-fix: updated to match vendor/nethack/src/uhitm.c:709-710
    # (mhit = tmp > dieroll, strict greater-than — not <=) and
    # vendor/nethack/src/weapon.c:1571 (P_SKILLED → +2 to-hit, not +1).
    NetHack convention (uhitm.c:376, 709-710):
        tmp = 1 + abon + find_mac(mtmp) + skill_bonus + enchant
    Hit iff tmp > rnd(20).

    With STR=18/100 (118 internally), DEX=18, XL=5, SKILLED skill:
        abon  = 3 (STR) + (18-14) (DEX) = 7
        skill = +2 (SKILLED, vendor weapon.c:1571)
        tmp   = 1 + 7 + 10 + 2 + 0 = 20 → hit prob = 19/20.
    """
    from Nethax.nethax.subsystems.combat import SKILL_SKILLED

    state = _fresh_state().replace(
        player_str=jnp.int16(18 + 100),
        player_dex=jnp.int8(18),
        player_xl=jnp.int32(5),
    )
    new_skill = state.combat.weapon_skill.at[0].set(jnp.int8(SKILL_SKILLED))
    state = state.replace(combat=state.combat.replace(weapon_skill=new_skill))

    target_ac = jnp.int32(10)
    keys = jax.random.split(jax.random.PRNGKey(42), 100)
    hits = jnp.array([bool(to_hit_roll(k, state, target_ac)) for k in keys])
    hit_rate = float(hits.mean())
    assert hit_rate >= 0.75, f"expected >=75% hit rate, got {hit_rate:.2%}"


def test_to_hit_low_str_vs_ac_minus_10_is_hard():
    """Very-armored monster (low AC) is hard to hit."""
    state = _fresh_state().replace(
        player_str=jnp.int16(8),
        player_dex=jnp.int8(8),
        player_xl=jnp.int32(1),
    )
    target_ac = jnp.int32(-10)  # very well armored
    keys = jax.random.split(jax.random.PRNGKey(7), 200)
    hits = jnp.array([bool(to_hit_roll(k, state, target_ac)) for k in keys])
    # With tmp = 1 + 0 (abon) + (-10) + 0 + 0 = -9, hits should be 0.
    assert float(hits.mean()) < 0.05


# ---------------------------------------------------------------------------
# Damage
# ---------------------------------------------------------------------------

def test_damage_roll_dagger_avg_about_2_5():
    """1d4 average is 2.5 (with no STR bonus & no enchant)."""
    class _StubWeapon:
        enchantment = jnp.int32(0)

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    rolls = jnp.array(
        [
            int(
                damage_roll(
                    k,
                    _StubWeapon(),
                    jnp.bool_(False),
                    sdam_n=1,
                    sdam_sides=4,
                    ldam_n=1,
                    ldam_sides=4,
                    str_bonus=jnp.int32(0),
                )
            )
            for k in keys
        ]
    )
    avg = float(rolls.mean())
    assert 2.0 <= avg <= 3.0, f"1d4 mean expected ~2.5, got {avg:.3f}"


def test_damage_roll_uses_large_dice_against_big_target():
    """Large-target branch picks ldam over sdam."""
    class _StubWeapon:
        enchantment = jnp.int32(0)

    # sdam=1d1 (always 1), ldam=1d12 (avg 6.5)
    keys = jax.random.split(jax.random.PRNGKey(99), 50)
    smalls = jnp.array(
        [
            int(
                damage_roll(
                    k, _StubWeapon(), jnp.bool_(False),
                    sdam_n=1, sdam_sides=1, ldam_n=1, ldam_sides=12,
                    str_bonus=jnp.int32(0),
                )
            )
            for k in keys
        ]
    )
    larges = jnp.array(
        [
            int(
                damage_roll(
                    k, _StubWeapon(), jnp.bool_(True),
                    sdam_n=1, sdam_sides=1, ldam_n=1, ldam_sides=12,
                    str_bonus=jnp.int32(0),
                )
            )
            for k in keys
        ]
    )
    assert float(smalls.mean()) == 1.0
    assert float(larges.mean()) > 3.0


# ---------------------------------------------------------------------------
# Bump-attack
# ---------------------------------------------------------------------------

def test_bump_attack_reduces_monster_hp():
    """Bumping an adjacent monster damages it (with high-STR attacker)."""
    state = _fresh_state().replace(
        player_pos=jnp.array([5, 5], dtype=jnp.int16),
        player_str=jnp.int16(18 + 100),
        player_dex=jnp.int8(18),
        player_xl=jnp.int32(5),
    )
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        hp=mai.hp.at[0].set(jnp.int32(100)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(100)),
        pos=mai.pos.at[0].set(jnp.array([5, 6], dtype=jnp.int16)),
        ac=mai.ac.at[0].set(jnp.int8(10)),
    )
    state = state.replace(monster_ai=mai)

    initial_hp = int(state.monster_ai.hp[0])
    # Average several rolls to absorb miss variance; expect total dmg > 0.
    rng = jax.random.PRNGKey(1)
    total_dmg = 0
    cur_state = state
    for i in range(20):
        rng, sub = jax.random.split(rng)
        cur_state = bump_attack(
            cur_state, sub, jnp.array([5, 6], dtype=jnp.int32)
        )
    final_hp = int(cur_state.monster_ai.hp[0])
    assert final_hp < initial_hp, (
        f"expected hp to drop from {initial_hp}, got {final_hp}"
    )


def test_bump_attack_no_monster_is_noop_for_hp():
    """Bumping an empty tile shouldn't crash or change state HP fields."""
    state = _fresh_state().replace(
        player_pos=jnp.array([5, 5], dtype=jnp.int16),
    )
    initial_hp = state.monster_ai.hp
    rng = jax.random.PRNGKey(2)
    new_state = bump_attack(state, rng, jnp.array([5, 6], dtype=jnp.int32))
    assert jnp.array_equal(new_state.monster_ai.hp, initial_hp)


# ---------------------------------------------------------------------------
# Skill practice
# ---------------------------------------------------------------------------

def test_practice_skill_advances_at_threshold():
    """Basic→Skilled threshold is 1*1*20 = 20 practices."""
    state = _fresh_state()
    # Push tier up to BASIC manually.
    state = state.replace(
        combat=state.combat.replace(
            weapon_skill=state.combat.weapon_skill.at[0].set(jnp.int8(SKILL_BASIC)),
        )
    )
    weapon_type = jnp.int32(0)
    # 20 practices at BASIC (1*1*20 = 20) → advance to SKILLED.
    for _ in range(20):
        state = practice_skill(state, weapon_type)
    assert int(state.combat.weapon_skill[0]) == SKILL_BASIC + 1
    assert int(state.combat.weapon_practice[0]) == 0
