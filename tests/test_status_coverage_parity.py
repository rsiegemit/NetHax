"""Status-coverage parity tests: blind/stun block reads, deaf silences shrieker,
glib drops potions.

Vendor citations:
  read.c::doread     — HBlinded / HStun early-return gates
  sounds.c           — deaf gate suppresses sound-based events
  status.c::glib     — slippery fingers drop items on use
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.status_effects import TimedStatus
from Nethax.nethax.subsystems.inventory import ItemCategory
from Nethax.nethax.subsystems.items_scrolls import read_scroll, _SCROLL_BASE_ID
from Nethax.nethax.subsystems.items_spellbooks import read_spellbook
from Nethax.nethax.subsystems.items_potions import quaff_potion, _POTION_BASE_ID
from Nethax.nethax.subsystems.monster_ai import shrieker_summon
from Nethax.nethax.constants.monsters import MONSTERS, MS_SHRIEK

# Resolve shrieker entry index at import time by scanning MONSTERS for MS_SHRIEK.
_SHRIEKER_ENTRY_IDX: int = next(
    i for i, m in enumerate(MONSTERS) if int(m.sound) == MS_SHRIEK
)

_RNG = jax.random.PRNGKey(0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_status(state: EnvState, status_key: TimedStatus, turns: int) -> EnvState:
    ts = state.status.timed_statuses.at[int(status_key)].set(jnp.int32(turns))
    return state.replace(status=state.status.replace(timed_statuses=ts))


def _put_scroll(state: EnvState, slot: int = 0) -> EnvState:
    """Place an enchant-weapon scroll (effect 5) in slot `slot`."""
    type_id = _SCROLL_BASE_ID + 5   # SCR_ENCHANT_WEAPON
    items = state.inventory.items.replace(
        category=state.inventory.items.category.at[slot].set(jnp.int8(int(ItemCategory.SCROLL))),
        type_id=state.inventory.items.type_id.at[slot].set(jnp.int16(type_id)),
        quantity=state.inventory.items.quantity.at[slot].set(jnp.int16(1)),
        buc_status=state.inventory.items.buc_status.at[slot].set(jnp.int8(2)),  # uncursed
    )
    return state.replace(inventory=state.inventory.replace(items=items))


def _put_spellbook(state: EnvState, slot: int = 0) -> EnvState:
    """Place a spellbook of healing (spell_id=0) in slot `slot`."""
    items = state.inventory.items.replace(
        category=state.inventory.items.category.at[slot].set(jnp.int8(int(ItemCategory.SPBOOK))),
        type_id=state.inventory.items.type_id.at[slot].set(jnp.int16(0)),  # spell_id 0 = HEALING
        quantity=state.inventory.items.quantity.at[slot].set(jnp.int16(1)),
        buc_status=state.inventory.items.buc_status.at[slot].set(jnp.int8(2)),  # uncursed
    )
    return state.replace(inventory=state.inventory.replace(items=items))


def _put_potion(state: EnvState, slot: int = 0) -> EnvState:
    """Place a potion of healing (effect 10) in slot `slot`."""
    type_id = _POTION_BASE_ID + 10   # POT_HEALING
    items = state.inventory.items.replace(
        category=state.inventory.items.category.at[slot].set(jnp.int8(int(ItemCategory.POTION))),
        type_id=state.inventory.items.type_id.at[slot].set(jnp.int16(type_id)),
        quantity=state.inventory.items.quantity.at[slot].set(jnp.int16(1)),
        buc_status=state.inventory.items.buc_status.at[slot].set(jnp.int8(2)),
    )
    return state.replace(inventory=state.inventory.replace(items=items))


def _place_shrieker_adjacent(state: EnvState) -> EnvState:
    """Put a shrieker (entry_idx=167) at the tile south of the player."""
    pr = int(state.player_pos[0])
    pc = int(state.player_pos[1])
    map_h = state.terrain.shape[2]
    spawn_r = min(pr + 1, map_h - 1)
    mai = state.monster_ai
    # Chain all updates through a single replace to avoid losing earlier sets.
    mai = mai.replace(
        alive=mai.alive.at[0].set(jnp.bool_(True)),
        pos=mai.pos.at[0].set(jnp.array([spawn_r, pc], dtype=jnp.int16)),
        entry_idx=mai.entry_idx.at[0].set(jnp.int16(_SHRIEKER_ENTRY_IDX)),  # shrieker
        hp=mai.hp.at[0].set(jnp.int32(10)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(10)),
        peaceful=mai.peaceful.at[0].set(jnp.bool_(False)),
        asleep=mai.asleep.at[0].set(jnp.bool_(False)),
    )
    # Slot 1 stays dead (all slots start dead) — summon target is already there.
    return state.replace(monster_ai=mai)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_blind_blocks_scroll_read():
    """BLIND player cannot read a scroll — state unchanged.

    Cite: vendor/nethack/src/read.c::doread — HBlinded early return.
    """
    state = EnvState.default(_RNG)
    state = _put_scroll(state, slot=0)
    state = _set_status(state, TimedStatus.BLIND, 20)

    qty_before = int(state.inventory.items.quantity[0])
    after = read_scroll(state, _RNG, slot_idx=0)
    qty_after = int(after.inventory.items.quantity[0])

    assert qty_before == qty_after, "Blind player should not consume the scroll"
    # Inventory otherwise identical
    assert after is state or (
        int(after.inventory.items.quantity[0]) == qty_before
    )


def test_blind_blocks_spellbook():
    """BLIND player cannot study a spellbook — spell not learned.

    Cite: vendor/nethack/src/read.c::doread — HBlinded early return.
    """
    state = EnvState.default(_RNG)
    state = _put_spellbook(state, slot=0)
    state = _set_status(state, TimedStatus.BLIND, 20)
    # Ensure high INT so without blind it would succeed
    state = state.replace(player_int=jnp.int16(18))

    after = read_spellbook(state, _RNG, slot_idx=0)
    assert not bool(after.magic.spell_known[0]), "Blind player should not learn the spell"


def test_stun_blocks_read():
    """STUNNED player cannot read a scroll — state unchanged.

    Cite: vendor/nethack/src/read.c::doread — HStun early return.
    """
    state = EnvState.default(_RNG)
    state = _put_scroll(state, slot=0)
    state = _set_status(state, TimedStatus.STUNNED, 5)

    qty_before = int(state.inventory.items.quantity[0])
    after = read_scroll(state, _RNG, slot_idx=0)
    qty_after = int(after.inventory.items.quantity[0])

    assert qty_before == qty_after, "Stunned player should not consume the scroll"


def test_stun_blocks_spellbook():
    """STUNNED player cannot study a spellbook.

    Cite: vendor/nethack/src/read.c::doread — HStun early return.
    """
    state = EnvState.default(_RNG)
    state = _put_spellbook(state, slot=0)
    state = _set_status(state, TimedStatus.STUNNED, 5)
    state = state.replace(player_int=jnp.int16(18))

    after = read_spellbook(state, _RNG, slot_idx=0)
    assert not bool(after.magic.spell_known[0]), "Stunned player should not learn the spell"


def test_deaf_silences_shrieker():
    """DEAF player does not hear the shrieker — no monster summoned.

    Cite: vendor/nethack/src/sounds.c — deaf gate suppresses shrieker alarm.
    """
    state = EnvState.default(_RNG)
    state = _place_shrieker_adjacent(state)
    state = _set_status(state, TimedStatus.DEAF, 30)

    alive_before = int(jnp.sum(state.monster_ai.alive))

    # Run shrieker_summon 20 times; with p=0.25 per roll, at least one would
    # fire without the deaf gate.
    s = state
    for seed in range(20):
        s_candidate = shrieker_summon(state, jax.random.PRNGKey(seed))
        alive_after = int(jnp.sum(s_candidate.monster_ai.alive))
        assert alive_after == alive_before, (
            f"Deaf player should not trigger shrieker summon (seed={seed})"
        )


def test_shrieker_summons_without_deaf():
    """Sanity: without DEAF, shrieker CAN summon (probabilistic over many seeds).

    Cite: vendor/nethack/src/mon.c::shrieker — 25% chance per turn.
    """
    state = EnvState.default(_RNG)
    state = _place_shrieker_adjacent(state)
    # No deaf status

    alive_before = int(jnp.sum(state.monster_ai.alive))
    summoned_any = False
    for seed in range(200):
        after = shrieker_summon(state, jax.random.PRNGKey(seed))
        if int(jnp.sum(after.monster_ai.alive)) > alive_before:
            summoned_any = True
            break
    assert summoned_any, "Shrieker should summon at least once across 200 seeds without deaf"


def test_glib_drops_potion():
    """GLIB player drops the potion instead of quaffing with 1/5 probability.

    Cite: vendor/nethack/src/status.c::glib — slippery fingers drop items on use.
    Strategy: give the player reduced HP and high max so a quaff would heal.
    A drop = quantity decrements but HP stays the same.
    """
    state = EnvState.default(_RNG)
    state = _put_potion(state, slot=0)
    state = _set_status(state, TimedStatus.GLIB, 10)
    # Damage player so a successful quaff would clearly change HP.
    state = state.replace(player_hp=jnp.int32(5), player_hp_max=jnp.int32(50))

    qty_before = int(state.inventory.items.quantity[0])
    hp_before  = int(state.player_hp)

    dropped_any = False
    for seed in range(30):
        after = quaff_potion(state, jax.random.PRNGKey(seed), slot_idx=0)
        qty_after = int(after.inventory.items.quantity[0])
        hp_after  = int(after.player_hp)
        # Drop: qty decremented, HP unchanged (item never consumed).
        if qty_after == qty_before - 1 and hp_after == hp_before:
            dropped_any = True
            break

    assert dropped_any, "GLIB should drop the potion (hp unchanged) at least once in 30 seeds"


def test_no_glib_quaff_heals():
    """Without GLIB, quaffing a healing potion changes HP (not dropped)."""
    state = EnvState.default(_RNG)
    state = _put_potion(state, slot=0)
    # Damage player and raise max so healing has room to restore HP.
    state = state.replace(
        player_hp=jnp.int32(5),
        player_hp_max=jnp.int32(50),
    )

    after = quaff_potion(state, _RNG, slot_idx=0)
    assert int(after.player_hp) > 5, "Without GLIB, healing potion should restore HP"
