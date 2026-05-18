"""Wand polish parity tests — vendor/nethack/src/zap.c gaps filled in Wave N.

Tests verify corrected/new wand behaviours against vendor zap.c logic:
  WAN_PROBING        — probe_monster ~line 4700
  WAN_SLOW_MONSTER   — mon_adjust_speed ~line 4400
  WAN_SPEED_MONSTER  — mon_adjust_speed ~line 4400
  WAN_CANCELLATION   — cancel_monst ~line 4500
  WAN_ENLIGHTENMENT  — do_enlightenment: no map reveal
  WAN_OPENING        — wand_unlock ~line 4600: at most 50% of doors open
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.subsystems.items_wands import (
    WandEffect,
    WandState,
    ITEM_CATEGORY_WAND,
    zap_wand,
)
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.inventory import InventoryState, Item
from Nethax.nethax.subsystems.monster_ai import MAX_MONSTERS_PER_LEVEL

MAP_H, MAP_W = 21, 80
_RNG = jax.random.PRNGKey(7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(player_row: int = 10, player_col: int = 10) -> WandState:
    state = WandState.empty(map_h=MAP_H, map_w=MAP_W)
    terrain = jnp.full((MAP_H, MAP_W), int(TileType.FLOOR), dtype=jnp.int8)
    return state.replace(
        terrain=terrain,
        player_pos=jnp.array([player_row, player_col], dtype=jnp.int16),
    )


def _place_monster(
    state: WandState,
    slot: int,
    row: int,
    col: int,
    hp: int = 20,
    mon_type: int = 1,
) -> WandState:
    new_pos   = state.mon_pos.at[slot].set(jnp.array([row, col], dtype=jnp.int16))
    new_hp    = state.mon_hp.at[slot].set(jnp.int32(hp))
    new_type  = state.mon_type.at[slot].set(jnp.int16(mon_type))
    new_alive = state.mon_alive.at[slot].set(jnp.bool_(True))
    return state.replace(mon_pos=new_pos, mon_hp=new_hp,
                         mon_type=new_type, mon_alive=new_alive)


def _with_wand(state: WandState, effect: WandEffect, charges: int = 5) -> WandState:
    wand_item = Item(
        category=jnp.int8(ITEM_CATEGORY_WAND),
        type_id=jnp.int16(int(effect)),
        buc_status=jnp.int8(2),
        enchantment=jnp.int8(0),
        charges=jnp.int8(charges),
        identified=jnp.bool_(True),
        quantity=jnp.int16(1),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
    )
    return state.replace(inventory=InventoryState.from_items([wand_item]))


# ---------------------------------------------------------------------------
# WAN_PROBING — vendor zap.c::probe_monster ~line 4700
# ---------------------------------------------------------------------------

def test_wan_probing_stores_target_info():
    """WAN_PROBING must store the hit monster's HP in state.last_probed_hp."""
    state = _make_state(player_row=10, player_col=10)
    monster_hp = 42
    state = _place_monster(state, slot=1, row=10, col=11, hp=monster_hp)
    state = _with_wand(state, WandEffect.PROBING)

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    assert int(result.probed_hp) == monster_hp, (
        f"probed_hp expected {monster_hp}, got {int(result.probed_hp)}"
    )
    assert int(result.probed_idx) == 1, (
        f"probed_idx expected 1, got {int(result.probed_idx)}"
    )


# ---------------------------------------------------------------------------
# WAN_SLOW_MONSTER — vendor zap.c::mon_adjust_speed ~line 4400
# ---------------------------------------------------------------------------

def test_wan_slow_monster_decrements_speed():
    """WAN_SLOW_MONSTER must set mon_speed_mod[idx] < 0 on a hit monster."""
    state = _make_state(player_row=10, player_col=10)
    state = _place_monster(state, slot=1, row=10, col=11)
    state = _with_wand(state, WandEffect.SLOW_MONSTER)

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    speed_mod = int(result.mon_speed_mod[1])
    assert speed_mod < 0, (
        f"WAN_SLOW_MONSTER should produce mon_speed_mod < 0, got {speed_mod}"
    )


# ---------------------------------------------------------------------------
# WAN_SPEED_MONSTER — vendor zap.c::mon_adjust_speed ~line 4400
# ---------------------------------------------------------------------------

def test_wan_speed_monster_increments_speed():
    """WAN_SPEED_MONSTER must set mon_speed_mod[idx] > 0 on a hit monster."""
    state = _make_state(player_row=10, player_col=10)
    state = _place_monster(state, slot=1, row=10, col=11)
    state = _with_wand(state, WandEffect.SPEED_MONSTER)

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    speed_mod = int(result.mon_speed_mod[1])
    assert speed_mod > 0, (
        f"WAN_SPEED_MONSTER should produce mon_speed_mod > 0, got {speed_mod}"
    )


# ---------------------------------------------------------------------------
# WAN_CANCELLATION — vendor zap.c::cancel_monst ~line 4500
# ---------------------------------------------------------------------------

def test_wan_cancellation_flag_set():
    """WAN_CANCELLATION must set mon_cancelled[idx] = True on the hit monster."""
    state = _make_state(player_row=10, player_col=10)
    state = _place_monster(state, slot=1, row=10, col=11)
    state = _with_wand(state, WandEffect.CANCELLATION)

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    assert bool(result.mon_cancelled[1]), (
        "WAN_CANCELLATION must set mon_cancelled[1] = True"
    )


def test_wan_cancellation_does_not_affect_other_slots():
    """WAN_CANCELLATION (BEAM) must not cancel monsters in other slots."""
    state = _make_state(player_row=10, player_col=10)
    state = _place_monster(state, slot=1, row=10, col=11)
    state = _place_monster(state, slot=2, row=10, col=15)
    state = _with_wand(state, WandEffect.CANCELLATION)

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    # Slot 2 is beyond the first target; BEAM stops at first hit.
    assert not bool(result.mon_cancelled[2]), (
        "WAN_CANCELLATION (BEAM) must not cancel a monster behind the first target"
    )


# ---------------------------------------------------------------------------
# WAN_ENLIGHTENMENT — vendor zap.c::do_enlightenment: no map state change
# ---------------------------------------------------------------------------

def test_wan_enlightenment_does_not_reveal_map():
    """WAN_ENLIGHTENMENT must NOT change state.explored (deferred UI-only effect)."""
    state = _make_state()
    # Leave explored as all-False (zero-initialised).
    state = _with_wand(state, WandEffect.ENLIGHTENMENT)
    explored_before = state.explored.copy()

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    assert jnp.array_equal(result.explored, explored_before), (
        "WAN_ENLIGHTENMENT must not modify state.explored "
        "(vendor: intrinsics display only, not map reveal)"
    )


# ---------------------------------------------------------------------------
# WAN_OPENING — vendor zap.c::wand_unlock: single-door; 50% approx here
# ---------------------------------------------------------------------------

def test_wan_opening_does_not_open_all_doors():
    """WAN_OPENING must open at most 1 of 2 doors in most trials (50% per door).

    With p=0.5 per door, P(both open) = 0.25.  We run 20 independent seeds
    and assert that at least once fewer than both doors opened.  This fails
    only with probability 0.25^20 ≈ 9e-13.
    """
    both_open_count = 0
    for seed in range(20):
        state = _make_state()
        terrain = state.terrain
        # Place two closed doors at known positions.
        terrain = terrain.at[5, 5].set(jnp.int8(int(TileType.CLOSED_DOOR)))
        terrain = terrain.at[15, 15].set(jnp.int8(int(TileType.CLOSED_DOOR)))
        state = state.replace(terrain=terrain)
        state = _with_wand(state, WandEffect.OPENING)

        rng = jax.random.PRNGKey(seed + 100)
        result = zap_wand(state, rng, slot_idx=jnp.int32(0), direction=jnp.int32(2))

        door1_open = int(result.terrain[5, 5]) == int(TileType.OPEN_DOOR)
        door2_open = int(result.terrain[15, 15]) == int(TileType.OPEN_DOOR)
        if door1_open and door2_open:
            both_open_count += 1

    assert both_open_count < 20, (
        "WAN_OPENING opened both doors in all 20 trials — "
        "should be 50% per door, not 100% (vendor: single-door targeted effect)"
    )
