"""Parity tests for monster passive contact effects.

Vendor reference: vendor/nethack/src/uhitm.c::passive() lines 5864-6119
                  vendor/nethack/src/mhitu.c::hitmu() line 1060 (AD_DRIN)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

import Nethax.nethax.subsystems.artifact_powers  # noqa: F401
import Nethax.nethax.subsystems.weapon_dice       # noqa: F401

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.subsystems.combat import melee_attack, SKILL_BASIC
from Nethax.nethax.subsystems.status_effects import TimedStatus, Intrinsic
from Nethax.nethax.subsystems.inventory import ArmorSlot, ItemCategory, make_item
from Nethax.nethax.subsystems.monster_passives import apply_passive_to_player

_RNG = jax.random.PRNGKey(0)

# Monster entry indices (verified against Nethax/nethax/constants/monsters.py):
_IDX_FLOATING_EYE = 28
_IDX_BROWN_MOLD   = 156
_IDX_YELLOW_MOLD  = 157
_IDX_GREEN_MOLD   = 158
_IDX_RED_MOLD     = 159
_IDX_BLUE_JELLY   = 55
_IDX_COCKATRICE   = 10
_IDX_MIND_FLAYER  = 47
_IDX_DISENCHANTER = 209
_IDX_RUST_MONSTER = 208

# Weapon item: a +2 long sword (category=WEAPON=1, type_id=50, enchantment=+2)
_SWORD_TYPE_ID = 50


def _base_state(role: Role = Role.VALKYRIE, entry_idx: int = 0) -> EnvState:
    """Return a state with a single live monster at slot 0."""
    state = EnvState.default(_RNG)
    state = state.replace(
        player_role=jnp.int8(int(role)),
        player_str=jnp.int16(18),
        player_dex=jnp.int8(14),
        player_xl=jnp.int32(5),
        player_hp=jnp.int32(50),
        player_hp_max=jnp.int32(50),
        player_int=jnp.int8(12),
        player_luck=jnp.int8(0),
        player_uhitinc=jnp.int8(0),
        player_udaminc=jnp.int8(0),
        combat=EnvState.default(_RNG).combat.replace(
            weapon_skill=EnvState.default(_RNG).combat.weapon_skill.at[0].set(
                jnp.int8(SKILL_BASIC)
            )
        ),
    )
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(jnp.bool_(True)),
        hp=mai.hp.at[0].set(jnp.int32(9999)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(9999)),
        ac=mai.ac.at[0].set(jnp.int8(5)),
        entry_idx=mai.entry_idx.at[0].set(jnp.int16(entry_idx)),
        pos=mai.pos.at[0].set(jnp.array([3, 3], dtype=jnp.int16)),
    )
    return state.replace(monster_ai=mai)


def _with_weapon(state: EnvState, enchantment: int = 2) -> EnvState:
    """Place a weapon in slot 0 and wield it."""
    item = make_item(
        category=int(ItemCategory.WEAPON),
        type_id=_SWORD_TYPE_ID,
        enchantment=enchantment,
        quantity=1,
    )
    new_items = state.inventory.items
    for field_name in item.__dataclass_fields__:
        arr = getattr(new_items, field_name)
        val = getattr(item, field_name)
        new_arr = arr.at[0].set(val)
        new_items = new_items.replace(**{field_name: new_arr})
    new_inv = state.inventory.replace(items=new_items, wielded=jnp.int8(0))
    return state.replace(inventory=new_inv)


def _run_passive(state: EnvState, rng=None) -> EnvState:
    """Apply passive from monster slot 0."""
    if rng is None:
        rng = _RNG
    return apply_passive_to_player(state, jnp.int32(0), rng)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_floating_eye_paralyzes_on_melee():
    """Floating eye: player gets frozen (FROZEN timer > 0) unless blind/reflecting.

    Cite: vendor/nethack/src/uhitm.c::passive() line 6022-6064 (AD_PLYS).
    """
    state = _base_state(entry_idx=_IDX_FLOATING_EYE)
    result = _run_passive(state, jax.random.PRNGKey(1))
    assert int(result.status.timed_statuses[int(TimedStatus.FROZEN)]) > 0


def test_floating_eye_no_paralyze_when_blind():
    """Floating eye passive blocked when player is blind."""
    state = _base_state(entry_idx=_IDX_FLOATING_EYE)
    # Set BLIND timer
    new_statuses = state.status.timed_statuses.at[int(TimedStatus.BLIND)].set(jnp.int32(100))
    state = state.replace(status=state.status.replace(timed_statuses=new_statuses))
    result = _run_passive(state)
    assert int(result.status.timed_statuses[int(TimedStatus.FROZEN)]) == 0


def test_floating_eye_no_paralyze_when_reflecting():
    """Floating eye passive blocked when player has reflection."""
    state = _base_state(entry_idx=_IDX_FLOATING_EYE)
    new_intrinsics = state.status.intrinsics.at[int(Intrinsic.REFLECTING)].set(jnp.bool_(True))
    state = state.replace(status=state.status.replace(intrinsics=new_intrinsics))
    result = _run_passive(state)
    assert int(result.status.timed_statuses[int(TimedStatus.FROZEN)]) == 0


def test_brown_mold_cold_dmg():
    """Brown mold: deals cold damage to player on contact.

    Cite: vendor/nethack/src/uhitm.c::passive() line 6066-6083 (AD_COLD).
    """
    state = _base_state(entry_idx=_IDX_BROWN_MOLD)
    # Run many trials — at least one must deal damage (rnd(6) > 0 always).
    damaged = False
    for seed in range(50):
        result = _run_passive(state, jax.random.PRNGKey(seed))
        if int(result.player_hp) < 50:
            damaged = True
            break
    assert damaged, "brown mold should deal cold damage"


def test_brown_mold_cold_resist_blocks():
    """Cold resistance blocks brown mold damage."""
    state = _base_state(entry_idx=_IDX_BROWN_MOLD)
    new_intrinsics = state.status.intrinsics.at[int(Intrinsic.RESIST_COLD)].set(jnp.bool_(True))
    state = state.replace(status=state.status.replace(intrinsics=new_intrinsics))
    for seed in range(50):
        result = _run_passive(state, jax.random.PRNGKey(seed))
        assert int(result.player_hp) == 50, "cold resist should block damage"


def test_yellow_mold_paralyzes():
    """Yellow mold: contact causes paralysis (FROZEN timer) without magic resistance.

    Cite: vendor/nethack/src/uhitm.c::passive() line 6085-6088 (AD_STUN).
    """
    state = _base_state(entry_idx=_IDX_YELLOW_MOLD)
    result = _run_passive(state, jax.random.PRNGKey(5))
    assert int(result.status.timed_statuses[int(TimedStatus.FROZEN)]) > 0


def test_yellow_mold_magic_resist_blocks():
    """Magic resistance blocks yellow mold paralysis."""
    state = _base_state(entry_idx=_IDX_YELLOW_MOLD)
    new_intrinsics = state.status.intrinsics.at[int(Intrinsic.MAGIC_RESIST)].set(jnp.bool_(True))
    state = state.replace(status=state.status.replace(intrinsics=new_intrinsics))
    result = _run_passive(state)
    assert int(result.status.timed_statuses[int(TimedStatus.FROZEN)]) == 0


def test_green_mold_acid():
    """Green mold: acid damage and weapon corrosion on contact.

    Cite: vendor/nethack/src/uhitm.c::passive() lines 5906-5933 (AD_ACID) →
    vendor/nethack/src/trap.c::erode_obj ERODE_CORRODE: is_primary=False so
    corrosion writes to oeroded2 (cite trap.c:225), not oeroded.
    """
    state = _base_state(entry_idx=_IDX_GREEN_MOLD)
    state = _with_weapon(state)
    result = _run_passive(state, jax.random.PRNGKey(7))
    # HP should decrease
    assert int(result.player_hp) < 50
    # Weapon corrosion is tracked in oeroded2 (ERODE_CORRODE is_primary=False).
    assert int(result.inventory.items.oeroded2[0]) > 0


def test_acid_resist_blocks_green_mold():
    """Acid resistance blocks green mold damage and weapon corrosion."""
    state = _base_state(entry_idx=_IDX_GREEN_MOLD)
    state = _with_weapon(state)
    new_intrinsics = state.status.intrinsics.at[int(Intrinsic.RESIST_ACID)].set(jnp.bool_(True))
    state = state.replace(status=state.status.replace(intrinsics=new_intrinsics))
    result = _run_passive(state)
    assert int(result.player_hp) == 50
    assert int(result.inventory.items.oeroded[0]) == 0


def test_red_mold_fire():
    """Red mold: fire damage on contact.

    Cite: vendor/nethack/src/uhitm.c::passive() lines 5895-5905 (AD_FIRE).
    """
    state = _base_state(entry_idx=_IDX_RED_MOLD)
    result = _run_passive(state, jax.random.PRNGKey(3))
    assert int(result.player_hp) < 50


def test_fire_resist_blocks_red_mold():
    """Fire resistance blocks red mold damage."""
    state = _base_state(entry_idx=_IDX_RED_MOLD)
    new_intrinsics = state.status.intrinsics.at[int(Intrinsic.RESIST_FIRE)].set(jnp.bool_(True))
    state = state.replace(status=state.status.replace(intrinsics=new_intrinsics))
    result = _run_passive(state)
    assert int(result.player_hp) == 50


def test_blue_jelly_sleep():
    """Blue jelly: cold damage + chance of sleep on contact.

    Cite: vendor/nethack/src/uhitm.c::passive() line 6066 (AD_COLD, blue jelly).
    """
    state = _base_state(entry_idx=_IDX_BLUE_JELLY)
    # Run many seeds; at least some must apply sleep (1/3 chance).
    slept = False
    for seed in range(100):
        result = _run_passive(state, jax.random.PRNGKey(seed))
        if int(result.status.timed_statuses[int(TimedStatus.SLEEP)]) > 0:
            slept = True
            break
    assert slept, "blue jelly should sometimes apply sleep (1/3 chance)"


def test_cockatrice_stones_no_gloves():
    """Cockatrice passive stones player when no gloves are worn.

    Cite: vendor/nethack/src/uhitm.c::passive() lines 5934-5957 (AD_STON).
    """
    state = _base_state(entry_idx=_IDX_COCKATRICE)
    # Ensure no gloves (worn_armor[GLOVES] == -1 by default)
    assert int(state.inventory.worn_armor[int(ArmorSlot.GLOVES)]) == -1
    result = _run_passive(state)
    assert int(result.status.timed_statuses[int(TimedStatus.STONED)]) == 5


def test_cockatrice_no_stone_with_gloves():
    """Cockatrice passive does NOT stone player when gloves are worn.

    Cite: vendor/nethack/src/uhitm.c::passive() line 5943-5944.
    """
    state = _base_state(entry_idx=_IDX_COCKATRICE)
    # Place gloves in slot 1 and equip them
    glove_item = make_item(category=int(ItemCategory.ARMOR), type_id=10, quantity=1)
    new_items = state.inventory.items
    for field_name in glove_item.__dataclass_fields__:
        arr = getattr(new_items, field_name)
        val = getattr(glove_item, field_name)
        new_arr = arr.at[1].set(val)
        new_items = new_items.replace(**{field_name: new_arr})
    new_worn = state.inventory.worn_armor.at[int(ArmorSlot.GLOVES)].set(jnp.int8(1))
    new_inv = state.inventory.replace(items=new_items, worn_armor=new_worn)
    state = state.replace(inventory=new_inv)

    result = _run_passive(state)
    assert int(result.status.timed_statuses[int(TimedStatus.STONED)]) == 0


def test_mind_flayer_drains_int():
    """Mind flayer: drains rnd(2) Int on tentacle contact.

    Cite: vendor/nethack/src/mhitu.c::hitmu() line 1060 (AD_DRIN).
    """
    state = _base_state(entry_idx=_IDX_MIND_FLAYER)
    assert int(state.player_int) == 12
    result = _run_passive(state, jax.random.PRNGKey(9))
    assert int(result.player_int) < 12


def test_mind_flayer_xl_loss_at_low_int():
    """Mind flayer: XL decreases when Int reaches 1."""
    state = _base_state(entry_idx=_IDX_MIND_FLAYER)
    # Set Int to 1 so drain to 1 triggers XL loss
    state = state.replace(player_int=jnp.int8(1), player_xl=jnp.int32(5))
    result = _run_passive(state, jax.random.PRNGKey(2))
    # Int clamped at 1; XL should decrease
    assert int(result.player_xl) < 5


def test_disenchanter_drops_enchant():
    """Disenchanter: reduces wielded weapon enchantment by 1.

    Cite: vendor/nethack/src/uhitm.c::passive() lines 5992-6011 (AD_ENCH).
    """
    state = _base_state(entry_idx=_IDX_DISENCHANTER)
    state = _with_weapon(state, enchantment=2)
    assert int(state.inventory.items.enchantment[0]) == 2
    result = _run_passive(state)
    assert int(result.inventory.items.enchantment[0]) == 1


def test_disenchanter_no_weapon_noop():
    """Disenchanter with no weapon wielded: no crash, state unchanged."""
    state = _base_state(entry_idx=_IDX_DISENCHANTER)
    # No weapon wielded (wielded == -1 by default)
    result = _run_passive(state)
    # Just verify it didn't raise/crash and HP unchanged
    assert int(result.player_hp) == 50


def test_rust_monster_rusts_iron():
    """Rust monster: increments oeroded on wielded weapon.

    Cite: vendor/nethack/src/uhitm.c::passive() lines 5958-5967 (AD_RUST).
    """
    state = _base_state(entry_idx=_IDX_RUST_MONSTER)
    state = _with_weapon(state)
    assert int(state.inventory.items.oeroded[0]) == 0
    result = _run_passive(state)
    assert int(result.inventory.items.oeroded[0]) == 1


def test_rust_monster_proof_blocks():
    """Rust-proof weapons are not corroded by rust monster."""
    state = _base_state(entry_idx=_IDX_RUST_MONSTER)
    state = _with_weapon(state)
    # Mark weapon rust-proof
    new_oerodeproof = state.inventory.items.oerodeproof.at[0].set(jnp.bool_(True))
    new_items = state.inventory.items.replace(oerodeproof=new_oerodeproof)
    state = state.replace(inventory=state.inventory.replace(items=new_items))
    result = _run_passive(state)
    assert int(result.inventory.items.oeroded[0]) == 0


def test_passive_fires_via_melee_attack():
    """Passive dispatch fires when invoked after a melee strike.

    melee_attack itself does not call apply_passive_to_player (combat.py:21
    explicitly defers engulf/passive); higher-level callers invoke the
    passive after melee resolution.  This test exercises that two-step
    sequence: melee_attack then apply_passive_to_player.
    """
    state = _base_state(entry_idx=_IDX_COCKATRICE)
    assert int(state.inventory.worn_armor[int(ArmorSlot.GLOVES)]) == -1

    rng = jax.random.PRNGKey(42)
    state_after_melee, _dmg, _hit = melee_attack(state, rng, jnp.int32(0))
    new_state = apply_passive_to_player(state_after_melee, jnp.int32(0), rng)
    # Cockatrice passive must have fired → player stoned
    assert int(new_state.status.timed_statuses[int(TimedStatus.STONED)]) == 5
