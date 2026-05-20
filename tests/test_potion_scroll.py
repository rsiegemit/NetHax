"""Wave 3 tests for potion and scroll effect implementations.

Tests verify the core mechanics of each major effect category.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.inventory import (
    InventoryState,
    make_item,
    ItemCategory,
)
from Nethax.nethax.subsystems.items_potions import (
    PotionEffect,
    _POTION_BASE_ID,
    quaff_potion,
)
from Nethax.nethax.subsystems.items_scrolls import (
    ScrollEffect,
    _SCROLL_BASE_ID,
    read_scroll,
)
from Nethax.nethax.subsystems.status_effects import TimedStatus

_RNG = jax.random.PRNGKey(42)

_BUC_UNCURSED = 2
_BUC_CURSED   = 1
_BUC_BLESSED  = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_with_potion(effect: PotionEffect, buc: int = _BUC_UNCURSED,
                       hp: int = 5, hp_max: int = 20) -> EnvState:
    """Return an EnvState with a single potion in slot 0."""
    type_id = _POTION_BASE_ID + int(effect)
    item    = make_item(
        category=int(ItemCategory.POTION),
        type_id=type_id,
        quantity=1,
        buc_status=buc,
    )
    state = EnvState.default(_RNG)
    state = state.replace(
        player_hp=jnp.int32(hp),
        player_hp_max=jnp.int32(hp_max),
        inventory=InventoryState.from_items([item]),
    )
    return state


def _state_with_scroll(effect: ScrollEffect, buc: int = _BUC_UNCURSED) -> EnvState:
    """Return an EnvState with a single scroll in slot 0."""
    type_id = _SCROLL_BASE_ID + int(effect)
    item    = make_item(
        category=int(ItemCategory.SCROLL),
        type_id=type_id,
        quantity=1,
        buc_status=buc,
    )
    state = EnvState.default(_RNG)
    return state.replace(inventory=InventoryState.from_items([item]))


# ---------------------------------------------------------------------------
# Potion tests
# ---------------------------------------------------------------------------

def test_healing_increases_hp():
    """potion of healing → player HP increases."""
    state  = _state_with_potion(PotionEffect.HEALING, hp=5, hp_max=20)
    before = int(state.player_hp)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.player_hp) > before, (
        f"Expected HP > {before}, got {int(result.player_hp)}"
    )


def test_healing_does_not_exceed_max():
    """potion of healing → HP never exceeds player_hp_max."""
    state  = _state_with_potion(PotionEffect.HEALING, hp=19, hp_max=20)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.player_hp) <= int(result.player_hp_max), (
        f"HP {int(result.player_hp)} exceeds max {int(result.player_hp_max)}"
    )


def test_full_healing_sets_hp_to_max():
    """potion of full healing → player HP == player_hp_max."""
    state  = _state_with_potion(PotionEffect.FULL_HEALING, hp=3, hp_max=20)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.player_hp) == int(result.player_hp_max), (
        f"Expected HP == {int(result.player_hp_max)}, got {int(result.player_hp)}"
    )


def test_extra_healing_increases_hp():
    """potion of extra healing → HP increases and hallucination cleared."""
    state  = _state_with_potion(PotionEffect.EXTRA_HEALING, hp=5, hp_max=50)
    # pre-set hallucination timer
    new_ts = state.status.timed_statuses.at[int(TimedStatus.HALLUCINATION)].set(jnp.int32(100))
    state  = state.replace(status=state.status.replace(timed_statuses=new_ts))
    result = quaff_potion(state, _RNG, 0)
    assert int(result.player_hp) > 5, (
        f"HP not increased: {int(result.player_hp)}"
    )
    assert int(result.status.timed_statuses[int(TimedStatus.HALLUCINATION)]) == 0, (
        "Hallucination should be cleared by extra healing"
    )


def test_gain_energy_increases_pw():
    """potion of gain energy → player_pw and player_pw_max both increase."""
    state  = _state_with_potion(PotionEffect.GAIN_ENERGY)
    state  = state.replace(player_pw=jnp.int32(5), player_pw_max=jnp.int32(10))
    result = quaff_potion(state, _RNG, 0)
    assert int(result.player_pw_max) > 10, (
        f"pw_max not increased: {int(result.player_pw_max)}"
    )


def test_confusion_adds_timer():
    """potion of confusion → confusion timed_status > 0."""
    state  = _state_with_potion(PotionEffect.CONFUSION)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.status.timed_statuses[int(TimedStatus.CONFUSION)]) > 0, (
        "Confusion timer not set"
    )


def test_paralysis_adds_frozen_timer():
    """potion of paralysis → FROZEN timed_status > 0."""
    state  = _state_with_potion(PotionEffect.PARALYSIS)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.status.timed_statuses[int(TimedStatus.FROZEN)]) > 0, (
        "Frozen timer not set by paralysis"
    )


def test_blindness_adds_blind_timer():
    """potion of blindness → BLIND timed_status > 0."""
    state  = _state_with_potion(PotionEffect.BLINDNESS)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.status.timed_statuses[int(TimedStatus.BLIND)]) > 0, (
        "Blind timer not set"
    )


def test_sleeping_adds_sleep_timer():
    """potion of sleeping → SLEEP timed_status > 0."""
    state  = _state_with_potion(PotionEffect.SLEEPING)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.status.timed_statuses[int(TimedStatus.SLEEP)]) > 0, (
        "Sleep timer not set"
    )


def test_sickness_reduces_hp():
    """potion of sickness → player HP decreases."""
    state  = _state_with_potion(PotionEffect.SICKNESS, hp=20, hp_max=20)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.player_hp) < 20, (
        f"HP not reduced by sickness: {int(result.player_hp)}"
    )


def test_acid_reduces_hp():
    """potion of acid → player HP decreases."""
    state  = _state_with_potion(PotionEffect.ACID, hp=20, hp_max=20)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.player_hp) < 20, (
        f"HP not reduced by acid: {int(result.player_hp)}"
    )


def test_see_invisible_grants_intrinsic():
    """potion of see invisible → SEE_INVIS timed intrinsic > 0."""
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    state  = _state_with_potion(PotionEffect.SEE_INVISIBLE)
    result = quaff_potion(state, _RNG, 0)
    timer  = int(result.status.timed_intrinsics[Intrinsic.SEE_INVIS])
    perm   = bool(result.status.intrinsics[Intrinsic.SEE_INVIS])
    assert timer > 0 or perm, "see-invisible not granted"


def test_levitation_adds_timed_intrinsic():
    """potion of levitation → LEVITATION timed intrinsic > 0."""
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    state  = _state_with_potion(PotionEffect.LEVITATION)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.status.timed_intrinsics[Intrinsic.LEVITATION]) > 0, (
        "Levitation timer not set"
    )


def test_holy_water_cures_sickness():
    """blessed potion of water (holy water) → SICK timer cleared."""
    state = _state_with_potion(PotionEffect.WATER, buc=_BUC_BLESSED)
    new_ts = state.status.timed_statuses.at[int(TimedStatus.SICK)].set(jnp.int32(50))
    state  = state.replace(status=state.status.replace(timed_statuses=new_ts))
    result = quaff_potion(state, _RNG, 0)
    assert int(result.status.timed_statuses[int(TimedStatus.SICK)]) == 0, (
        "Holy water should clear sickness"
    )


def test_potion_decrements_quantity():
    """Quaffing a potion decrements its quantity by 1."""
    state  = _state_with_potion(PotionEffect.HEALING, hp=5, hp_max=20)
    result = quaff_potion(state, _RNG, 0)
    before = int(state.inventory.items.quantity[0])
    after  = int(result.inventory.items.quantity[0])
    assert after == before - 1, f"Quantity not decremented: {before} → {after}"


def test_potion_slot_cleared_when_last():
    """Quaffing the last potion clears the slot category to 0."""
    state  = _state_with_potion(PotionEffect.HEALING, hp=5, hp_max=20)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.inventory.items.category[0]) == 0, (
        "Slot should be cleared when quantity reaches 0"
    )


# ---------------------------------------------------------------------------
# Scroll tests
# ---------------------------------------------------------------------------

def test_magic_mapping_reveals_entire_level():
    """scroll of magic mapping → state.explored all True on current level."""
    state  = _state_with_scroll(ScrollEffect.MAGIC_MAPPING)
    result = read_scroll(state, _RNG, 0)
    b  = int(result.dungeon.current_branch)
    lv = int(result.dungeon.current_level) - 1
    explored_level = result.explored[b, lv]
    assert bool(jnp.all(explored_level)), (
        "Magic mapping should reveal the entire level"
    )


def test_remove_curse_uncurses_items():
    """scroll of remove curse (uncursed) → uncurses worn/wielded items only.

    Vendor seffect_remove_curse (read.c:1505-1602):
      uncursed → uncurse worn+wielded items only
      blessed  → uncurse ALL inventory items + unpunish

    Slot 1 weapon must be wielded for the uncursed scroll to affect it.
    Slot 2 weapon stays in pack and remains cursed.
    """
    # Slot 0: scroll of remove curse (uncursed)
    scroll_type_id = _SCROLL_BASE_ID + int(ScrollEffect.REMOVE_CURSE)
    scroll = make_item(
        category=int(ItemCategory.SCROLL),
        type_id=scroll_type_id,
        quantity=1,
        buc_status=_BUC_UNCURSED,
    )
    # Slot 1: a cursed weapon (will be wielded)
    wielded_cursed_weapon = make_item(
        category=int(ItemCategory.WEAPON),
        type_id=1,
        quantity=1,
        buc_status=_BUC_CURSED,
    )
    # Slot 2: another cursed weapon (in pack, not worn)
    packed_cursed_weapon = make_item(
        category=int(ItemCategory.WEAPON),
        type_id=2,
        quantity=1,
        buc_status=_BUC_CURSED,
    )
    state = EnvState.default(_RNG)
    inv   = InventoryState.from_items(
        [scroll, wielded_cursed_weapon, packed_cursed_weapon]
    )
    inv   = inv.replace(wielded=jnp.int8(1))
    state = state.replace(inventory=inv)
    result = read_scroll(state, _RNG, 0)
    # Wielded weapon (slot 1) gets uncursed.
    assert int(result.inventory.items.buc_status[1]) != _BUC_CURSED, (
        "Wielded item at slot 1 should be uncursed by uncursed scroll; "
        f"buc={int(result.inventory.items.buc_status[1])}"
    )
    # Pack weapon (slot 2) remains cursed under vendor worn-only scope.
    assert int(result.inventory.items.buc_status[2]) == _BUC_CURSED, (
        "Unworn item at slot 2 must stay cursed under vendor worn-only "
        f"scope; buc={int(result.inventory.items.buc_status[2])}"
    )


def test_remove_curse_blessed_uncurses_all_inventory():
    """blessed scroll of remove curse → uncurses ALL inventory items."""
    scroll_type_id = _SCROLL_BASE_ID + int(ScrollEffect.REMOVE_CURSE)
    scroll = make_item(
        category=int(ItemCategory.SCROLL),
        type_id=scroll_type_id,
        quantity=1,
        buc_status=_BUC_BLESSED,
    )
    packed_cursed_weapon = make_item(
        category=int(ItemCategory.WEAPON),
        type_id=2,
        quantity=1,
        buc_status=_BUC_CURSED,
    )
    state = EnvState.default(_RNG)
    state = state.replace(
        inventory=InventoryState.from_items([scroll, packed_cursed_weapon])
    )
    result = read_scroll(state, _RNG, 0)
    assert int(result.inventory.items.buc_status[1]) != _BUC_CURSED, (
        "Blessed remove-curse should uncurse pack items too; "
        f"buc={int(result.inventory.items.buc_status[1])}"
    )


def test_identify_marks_first_item_identified():
    """scroll of identify → first unidentified item's identified flag becomes True."""
    scroll_type_id = _SCROLL_BASE_ID + int(ScrollEffect.IDENTIFY)
    scroll = make_item(
        category=int(ItemCategory.SCROLL),
        type_id=scroll_type_id,
        quantity=1,
        buc_status=_BUC_UNCURSED,
        identified=True,  # wave17h: make_item now defaults identified=False
    )
    # Slot 1: an unidentified potion (make_item sets identified=True by default;
    # override by building manually)
    from Nethax.nethax.subsystems.inventory import Item
    unid_potion = Item(
        category=jnp.int8(int(ItemCategory.POTION)),
        type_id=jnp.int16(_POTION_BASE_ID + int(PotionEffect.HEALING)),
        buc_status=jnp.int8(_BUC_UNCURSED),
        enchantment=jnp.int8(0),
        charges=jnp.int8(0),
        identified=jnp.bool_(False),
        quantity=jnp.int16(1),
        weight=jnp.int32(20),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
    )
    state = EnvState.default(_RNG)
    state = state.replace(
        inventory=InventoryState.from_items([scroll, unid_potion])
    )
    assert not bool(state.inventory.items.identified[1]), "Pre-condition: item not identified"
    result = read_scroll(state, _RNG, 0)
    assert bool(result.inventory.items.identified[1]), (
        "identify scroll should mark first unidentified item as identified"
    )


def test_amnesia_clears_explored():
    """scroll of amnesia → explored array all False on current level."""
    state  = _state_with_scroll(ScrollEffect.AMNESIA)
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    # Pre-explore the level
    new_explored = state.explored.at[b, lv].set(jnp.ones_like(state.explored[b, lv]))
    state  = state.replace(explored=new_explored)
    result = read_scroll(state, _RNG, 0)
    explored_level = result.explored[b, lv]
    assert not bool(jnp.any(explored_level)), (
        "Amnesia should erase the explored map"
    )


def test_enchant_weapon_increases_enchantment():
    """scroll of enchant weapon → wielded weapon's enchantment increases."""
    scroll_type_id = _SCROLL_BASE_ID + int(ScrollEffect.ENCHANT_WEAPON)
    scroll = make_item(
        category=int(ItemCategory.SCROLL),
        type_id=scroll_type_id,
        quantity=1,
        buc_status=_BUC_UNCURSED,
    )
    weapon = make_item(
        category=int(ItemCategory.WEAPON),
        type_id=1,
        quantity=1,
        enchantment=0,
    )
    state = EnvState.default(_RNG)
    inv   = InventoryState.from_items([scroll, weapon])
    # Mark slot 1 as wielded
    inv   = inv.replace(wielded=jnp.int8(1))
    state = state.replace(inventory=inv)
    result = read_scroll(state, _RNG, 0)
    assert int(result.inventory.items.enchantment[1]) > 0, (
        f"Enchantment not increased: {int(result.inventory.items.enchantment[1])}"
    )


def test_fire_scroll_deals_damage():
    """scroll of fire → player HP decreases."""
    state  = EnvState.default(_RNG)
    state  = state.replace(player_hp=jnp.int32(50), player_hp_max=jnp.int32(50))
    scroll_type_id = _SCROLL_BASE_ID + int(ScrollEffect.FIRE)
    item   = make_item(
        category=int(ItemCategory.SCROLL),
        type_id=scroll_type_id,
        quantity=1,
        buc_status=_BUC_UNCURSED,
    )
    state  = state.replace(inventory=InventoryState.from_items([item]))
    result = read_scroll(state, _RNG, 0)
    assert int(result.player_hp) < 50, (
        f"Fire scroll should deal damage; HP={int(result.player_hp)}"
    )


def test_scroll_decrements_quantity():
    """Reading a scroll decrements its quantity by 1."""
    state  = _state_with_scroll(ScrollEffect.LIGHT)
    result = read_scroll(state, _RNG, 0)
    before = int(state.inventory.items.quantity[0])
    after  = int(result.inventory.items.quantity[0])
    assert after == before - 1, f"Quantity not decremented: {before} → {after}"


def test_teleportation_moves_player():
    """scroll of teleportation → player_pos changes.

    Vendor seffect_teleportation rejection-samples to a walkable
    (FLOOR/CORRIDOR) tile; default EnvState terrain is all VOID, so we
    must paint a floor swath before reading the scroll or the player
    stays put (no valid landing site).
    """
    from Nethax.nethax.constants.tiles import TileType
    state  = _state_with_scroll(ScrollEffect.TELEPORTATION)
    state  = state.replace(
        player_pos=jnp.array([10, 10], dtype=jnp.int16)
    )
    # Paint the current level fully walkable so rejection sampling
    # can find a destination.
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    floor_level = jnp.full_like(state.terrain[b, lv], int(TileType.FLOOR))
    state = state.replace(
        terrain=state.terrain.at[b, lv].set(floor_level)
    )
    # Run a few times to reduce flakiness (random pos might land on same tile).
    moved = False
    for seed in range(10):
        rng    = jax.random.PRNGKey(seed)
        result = read_scroll(state, rng, 0)
        if not jnp.array_equal(result.player_pos, state.player_pos):
            moved = True
            break
    assert moved, "Teleportation should change player_pos"


def test_punishment_sets_is_punished():
    """scroll of punishment (uncursed) → is_punished=True.

    Updated from Wave-3 WOUNDED_LEGS stub to parity-correct ball+chain per
    vendor/nethack/src/read.c::seffect_punishment (~1976).
    """
    state  = _state_with_scroll(ScrollEffect.PUNISHMENT)
    result = read_scroll(state, _RNG, 0)
    assert bool(result.is_punished), (
        "Uncursed punishment must set is_punished=True"
    )


def test_stinking_cloud_adds_vomiting_timer():
    """scroll of stinking cloud → VOMITING timer > 0."""
    state  = _state_with_scroll(ScrollEffect.STINKING_CLOUD)
    result = read_scroll(state, _RNG, 0)
    assert int(result.status.timed_statuses[int(TimedStatus.VOMITING)]) > 0, (
        "Stinking cloud should set VOMITING timer"
    )
