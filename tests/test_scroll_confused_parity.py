"""Confused-reader parity tests for scroll effects.

Each test sets the CONFUSION timed status, reads a scroll, and asserts
vendor-correct confused-branch behavior per vendor/nethack/src/read.c::seffect_*.
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
    Item,
    make_item,
    ItemCategory,
)
from Nethax.nethax.subsystems.items_scrolls import (
    ScrollEffect,
    _SCROLL_BASE_ID,
    read_scroll,
)
from Nethax.nethax.subsystems.items_potions import _POTION_BASE_ID, PotionEffect
from Nethax.nethax.subsystems.status_effects import TimedStatus

_RNG = jax.random.PRNGKey(0)
_BUC_UNCURSED = 2
_BUC_CURSED   = 1
_BUC_BLESSED  = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scroll_item(effect: ScrollEffect, buc: int = _BUC_UNCURSED) -> Item:
    return make_item(
        category=int(ItemCategory.SCROLL),
        type_id=_SCROLL_BASE_ID + int(effect),
        quantity=1,
        buc_status=buc,
    )


def _state_confused(items, confusion: int = 50) -> EnvState:
    """Return a default EnvState with the given items and CONFUSION set."""
    state = EnvState.default(_RNG)
    state = state.replace(inventory=InventoryState.from_items(items))
    new_ts = state.status.timed_statuses.at[int(TimedStatus.CONFUSION)].set(
        jnp.int32(confusion)
    )
    return state.replace(status=state.status.replace(timed_statuses=new_ts))


def _state_clear(items) -> EnvState:
    """Return a default EnvState with given items and no confusion."""
    state = EnvState.default(_RNG)
    return state.replace(inventory=InventoryState.from_items(items))


# ---------------------------------------------------------------------------
# test_confused_teleport_changes_level
# vendor/nethack/src/read.c::seffect_teleport — confused branch: level tele
# ---------------------------------------------------------------------------

def test_confused_teleport_changes_level():
    """Confused teleport: current_level changes (level teleport).

    Sane teleport only moves player_pos within the level; confused teleport
    changes current_level.  vendor read.c::seffect_teleport confused branch.
    """
    scroll = _scroll_item(ScrollEffect.TELEPORTATION)
    state  = _state_confused([scroll])
    before_level = int(state.dungeon.current_level)

    # Try multiple seeds to ensure at least one changes level.
    changed = False
    for seed in range(20):
        rng    = jax.random.PRNGKey(seed)
        result = read_scroll(state, rng, 0)
        if int(result.dungeon.current_level) != before_level:
            changed = True
            break

    assert changed, (
        f"Confused teleport should change current_level (was {before_level})"
    )


def test_sane_teleport_does_not_change_level():
    """Sane teleport (no confusion): current_level stays the same."""
    scroll = _scroll_item(ScrollEffect.TELEPORTATION)
    state  = _state_clear([scroll])
    before_level = int(state.dungeon.current_level)

    for seed in range(10):
        rng    = jax.random.PRNGKey(seed)
        result = read_scroll(state, rng, 0)
        assert int(result.dungeon.current_level) == before_level, (
            "Sane teleport must NOT change current_level"
        )


# ---------------------------------------------------------------------------
# test_confused_remove_curse_mixed
# vendor/nethack/src/read.c::seffect_remove_curse — confused: random BUC
# ---------------------------------------------------------------------------

def test_confused_remove_curse_mixed():
    """Confused remove curse: BUC states get mixed (both cursed and blessed appear).

    Over 100 independent trials the random 50/50 bless/curse should produce
    both outcomes.  vendor read.c::seffect_remove_curse confused branch.
    """
    scroll = _scroll_item(ScrollEffect.REMOVE_CURSE)
    weapon = make_item(category=int(ItemCategory.WEAPON), type_id=1,
                       quantity=1, buc_status=_BUC_UNCURSED)
    state  = _state_confused([scroll, weapon])

    saw_blessed = False
    saw_cursed  = False
    for seed in range(100):
        rng    = jax.random.PRNGKey(seed)
        result = read_scroll(state, rng, 0)
        buc    = int(result.inventory.items.buc_status[1])
        if buc == _BUC_BLESSED:
            saw_blessed = True
        if buc == _BUC_CURSED:
            saw_cursed = True
        if saw_blessed and saw_cursed:
            break

    assert saw_blessed, "Confused remove-curse should sometimes bless items"
    assert saw_cursed,  "Confused remove-curse should sometimes curse items"


def test_sane_remove_curse_always_uncurses():
    """Sane remove curse: a cursed item always becomes uncursed (not cursed)."""
    scroll = _scroll_item(ScrollEffect.REMOVE_CURSE)
    weapon = make_item(category=int(ItemCategory.WEAPON), type_id=1,
                       quantity=1, buc_status=_BUC_CURSED)
    state  = _state_clear([scroll, weapon])
    result = read_scroll(state, _RNG, 0)
    assert int(result.inventory.items.buc_status[1]) != _BUC_CURSED, (
        "Sane remove-curse must uncurse cursed items"
    )


# ---------------------------------------------------------------------------
# test_confused_identify_only_self
# vendor/nethack/src/read.c::seffect_identify — confused: identify only scroll
# ---------------------------------------------------------------------------

def test_confused_identify_only_self():
    """Confused identify: only the scroll itself (slot 0) becomes identified.

    vendor read.c::seffect_identify confused branch: return after identifying
    only the scroll being read; other items remain unidentified.
    """
    scroll = _scroll_item(ScrollEffect.IDENTIFY)
    # Build two unidentified potions in slots 1 and 2.
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
    state  = _state_confused([scroll, unid_potion, unid_potion])

    result = read_scroll(state, _RNG, 0)

    # Slot 0 (the scroll) should now be identified.
    assert bool(result.inventory.items.identified[0]), (
        "Confused identify should identify the scroll itself (slot 0)"
    )
    # Slots 1 and 2 must remain unidentified.
    assert not bool(result.inventory.items.identified[1]), (
        "Confused identify must NOT identify other items (slot 1)"
    )
    assert not bool(result.inventory.items.identified[2]), (
        "Confused identify must NOT identify other items (slot 2)"
    )


def test_sane_identify_marks_other_items():
    """Sane identify (no confusion): identifies the first unidentified item."""
    scroll = _scroll_item(ScrollEffect.IDENTIFY)
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
    state  = _state_clear([scroll, unid_potion])
    result = read_scroll(state, _RNG, 0)
    assert bool(result.inventory.items.identified[1]), (
        "Sane identify should identify first unidentified item (slot 1)"
    )


# ---------------------------------------------------------------------------
# test_confused_magic_mapping_extra_confusion
# vendor/nethack/src/read.c::seffect_magic_mapping — confused: +30 confusion
# ---------------------------------------------------------------------------

def test_confused_magic_mapping_extra_confusion():
    """Confused magic mapping: CONFUSION timer extended by 30 turns.

    vendor read.c::seffect_magic_mapping confused branch: reading while
    confused causes the confusion timer to increase by 30 turns.
    """
    scroll = _scroll_item(ScrollEffect.MAGIC_MAPPING)
    state  = _state_confused([scroll], confusion=10)
    before_conf = int(state.status.timed_statuses[int(TimedStatus.CONFUSION)])
    assert before_conf == 10

    result      = read_scroll(state, _RNG, 0)
    after_conf  = int(result.status.timed_statuses[int(TimedStatus.CONFUSION)])

    assert after_conf >= before_conf + 30, (
        f"Confused magic mapping should add >=30 confusion turns; "
        f"before={before_conf}, after={after_conf}"
    )


def test_magic_mapping_reveals_level():
    """Magic mapping (confused or not) always reveals the level map."""
    scroll = _scroll_item(ScrollEffect.MAGIC_MAPPING)
    state  = _state_confused([scroll], confusion=10)
    result = read_scroll(state, _RNG, 0)
    b  = int(result.dungeon.current_branch)
    lv = int(result.dungeon.current_level) - 1
    assert bool(jnp.all(result.explored[b, lv])), (
        "Magic mapping should reveal the entire level even when confused"
    )


# ---------------------------------------------------------------------------
# test_confused_charging_drains_pw
# vendor/nethack/src/read.c::seffect_charging — confused: charge MP instead
# ---------------------------------------------------------------------------

def test_confused_charging_increases_pw():
    """Confused charging: player_pw increases (MP charged instead of wand).

    vendor read.c::seffect_charging confused branch: "You feel a mild tingle."
    The player's power (player_pw) increases rather than a wand being charged.
    """
    scroll = _scroll_item(ScrollEffect.CHARGING)
    state  = _state_confused([scroll])
    # Give the player some max_pw headroom.
    state  = state.replace(player_pw=jnp.int32(0), player_pw_max=jnp.int32(50))

    result = read_scroll(state, _RNG, 0)

    assert int(result.player_pw) > 0, (
        f"Confused charging should increase player_pw; got {int(result.player_pw)}"
    )


def test_sane_charging_does_not_change_pw():
    """Sane charging (no confusion, no wand): player_pw unchanged."""
    scroll = _scroll_item(ScrollEffect.CHARGING)
    state  = _state_clear([scroll])
    state  = state.replace(player_pw=jnp.int32(5), player_pw_max=jnp.int32(50))

    result = read_scroll(state, _RNG, 0)

    assert int(result.player_pw) == 5, (
        f"Sane charging (no wand) should not change player_pw; "
        f"got {int(result.player_pw)}"
    )
