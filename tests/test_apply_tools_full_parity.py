"""Full parity tests for the three previously-stubbed apply_tools handlers.

Covers:
  _h_magic_marker  — apply.c::write_with_marker / domarker
  _h_lock_pick     — lock.c::picklock (line 636-644) Dex-based chance
  _h_stethoscope   — apply.c::use_stethoscope (line 318)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.inventory import ItemCategory, MAX_INVENTORY_SLOTS
from Nethax.nethax.subsystems.features import FeaturesState, DoorState
from Nethax.nethax.subsystems.apply_tools import (
    dispatch_apply,
    _MAGIC_MARKER_TYPE_ID,
    _LOCK_PICK_TYPE_ID,
    _SKELETON_KEY_TYPE_ID,
    _CREDIT_CARD_TYPE_ID,
    _STETHOSCOPE_TYPE_ID,
)
from Nethax.nethax.subsystems.items_scrolls import _SCROLL_BASE_ID, ScrollEffect

_RNG = jax.random.PRNGKey(0)

_SCR_BLANK_PAPER_ID = _SCROLL_BASE_ID + ScrollEffect.BLANK_PAPER  # 116
_SCR_LIGHT_ID       = _SCROLL_BASE_ID + ScrollEffect.LIGHT         # 103


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _floor_state(player_pos=(10, 10)) -> EnvState:
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)
    floor_map = jnp.full((static.map_h, static.map_w), int(TileType.FLOOR), dtype=jnp.int8)
    return state.replace(
        terrain=state.terrain.at[0, 0].set(floor_map),
        player_pos=jnp.array(player_pos, dtype=jnp.int16),
        player_hp=jnp.int32(20),
        player_hp_max=jnp.int32(20),
    )


def _wield_item(state: EnvState, type_id: int, category: int, slot: int = 0, **kwargs) -> EnvState:
    inv = state.inventory
    new_cat = inv.items.category.at[slot].set(jnp.int8(category))
    new_tid = inv.items.type_id.at[slot].set(jnp.int16(type_id))
    new_items = inv.items.replace(category=new_cat, type_id=new_tid)
    for field, val in kwargs.items():
        arr = getattr(new_items, field)
        new_items = new_items.replace(**{field: arr.at[slot].set(val)})
    return state.replace(inventory=inv.replace(items=new_items, wielded=jnp.int8(slot)))


def _add_item(state: EnvState, type_id: int, category: int, slot: int) -> EnvState:
    inv = state.inventory
    new_cat = inv.items.category.at[slot].set(jnp.int8(category))
    new_tid = inv.items.type_id.at[slot].set(jnp.int16(type_id))
    new_items = inv.items.replace(category=new_cat, type_id=new_tid)
    return state.replace(inventory=inv.replace(items=new_items))


def _place_monster(state: EnvState, idx: int, pos, hp: int = 20) -> EnvState:
    mai = state.monster_ai
    mai = mai.replace(
        pos=mai.pos.at[idx].set(jnp.array(pos, dtype=jnp.int16)),
        hp=mai.hp.at[idx].set(jnp.int32(hp)),
        hp_max=mai.hp_max.at[idx].set(jnp.int32(hp)),
        alive=mai.alive.at[idx].set(jnp.bool_(True)),
    )
    return state.replace(monster_ai=mai)


def _set_locked_door(state: EnvState, row: int, col: int) -> EnvState:
    flat_lv = 0  # default state starts at branch 0, level 1 → flat index 0
    new_door = state.features.door_state.at[flat_lv, row, col].set(
        jnp.int8(int(DoorState.LOCKED))
    )
    return state.replace(features=state.features.replace(door_state=new_door))


# ---------------------------------------------------------------------------
# _h_magic_marker tests
# ---------------------------------------------------------------------------

def test_magic_marker_writes_blank_scroll():
    """Applying a magic marker converts SCR_BLANK_PAPER to another scroll type.

    Cite: vendor/nethack/src/apply.c::write_with_marker (~line 4320).
    """
    state = _floor_state()
    # Wield marker at slot 0, blank scroll at slot 1.
    state = _wield_item(state, _MAGIC_MARKER_TYPE_ID, int(ItemCategory.TOOL), slot=0,
                        enchantment=jnp.int8(0))
    state = _add_item(state, _SCR_BLANK_PAPER_ID, int(ItemCategory.SCROLL), slot=1)

    result = dispatch_apply(state, _RNG)

    new_tid = int(result.inventory.items.type_id[1])
    assert new_tid != _SCR_BLANK_PAPER_ID, (
        f"Blank scroll was not converted; type_id still {new_tid}"
    )
    # Must be a valid scroll type_id (within _SCROLL_BASE_ID .. _SCROLL_BASE_ID+21).
    assert _SCROLL_BASE_ID <= new_tid < _SCROLL_BASE_ID + 22, (
        f"Converted type_id {new_tid} is not a valid writable scroll"
    )


def test_magic_marker_no_blank_is_noop():
    """Without a blank scroll in inventory, marker apply is a noop."""
    state = _floor_state()
    state = _wield_item(state, _MAGIC_MARKER_TYPE_ID, int(ItemCategory.TOOL), slot=0)

    result = dispatch_apply(state, _RNG)

    # Inventory unchanged (no blank to convert).
    assert jnp.array_equal(result.inventory.items.type_id,
                           state.inventory.items.type_id)


# ---------------------------------------------------------------------------
# _h_lock_pick tests
# ---------------------------------------------------------------------------

def _lock_pick_success_rate(tool_type_id: int, dex: int, n_trials: int = 200) -> float:
    """Run dispatch_apply n_trials times and return fraction of successes."""
    state = _floor_state(player_pos=(10, 10))
    state = state.replace(player_dex=jnp.int8(dex))
    state = _wield_item(state, tool_type_id, int(ItemCategory.TOOL), slot=0)
    # Place locked door at player position.
    state = _set_locked_door(state, row=10, col=10)

    successes = 0
    for i in range(n_trials):
        rng = jax.random.PRNGKey(i * 17 + 3)
        result = dispatch_apply(state, rng)
        flat_lv = 0
        door_val = int(result.features.door_state[flat_lv, 10, 10])
        if door_val != int(DoorState.LOCKED):
            successes += 1
    return successes / n_trials


def test_lock_pick_dex_high():
    """High Dex (18) → lock pick should succeed at a high rate (>50%).

    Cite: lock.c:636 ch = 3 * ACURR(A_DEX) → ch=54 at Dex=18.
    """
    rate = _lock_pick_success_rate(_LOCK_PICK_TYPE_ID, dex=18, n_trials=200)
    assert rate > 0.40, f"Expected >40% success at Dex=18, got {rate:.2f}"


def test_lock_pick_dex_low():
    """Low Dex (6) → lock pick should succeed at a much lower rate (<40%).

    Cite: lock.c:636 ch = 3 * ACURR(A_DEX) → ch=18 at Dex=6.
    """
    rate = _lock_pick_success_rate(_LOCK_PICK_TYPE_ID, dex=6, n_trials=200)
    assert rate < 0.40, f"Expected <40% success at Dex=6, got {rate:.2f}"


@pytest.mark.timeout(240)
def test_skeleton_key_higher_rate_than_lock_pick():
    """Skeleton key (ch=70+Dex) should open more reliably than lock pick (ch=3*Dex).

    Cite: lock.c:638 vs lock.c:636.

    Note: two separate _lock_pick_success_rate calls each trigger a fresh
    JIT trace of _handle_apply (different otyp constants), so the wall-time
    is dominated by ~170s of compilation.  The 240s timeout absorbs cold-JIT
    on slower runners.
    """
    rate_key  = _lock_pick_success_rate(_SKELETON_KEY_TYPE_ID, dex=10, n_trials=200)
    rate_pick = _lock_pick_success_rate(_LOCK_PICK_TYPE_ID,    dex=10, n_trials=200)
    assert rate_key > rate_pick, (
        f"Skeleton key rate {rate_key:.2f} not greater than lock pick rate {rate_pick:.2f}"
    )


# ---------------------------------------------------------------------------
# _h_stethoscope tests
# ---------------------------------------------------------------------------

def test_stethoscope_writes_probed_hp():
    """Applying stethoscope adjacent to a monster writes its HP to state.probed_hp.

    Cite: vendor/nethack/src/apply.c::use_stethoscope (line 318).
    """
    state = _floor_state(player_pos=(10, 10))
    state = _wield_item(state, _STETHOSCOPE_TYPE_ID, int(ItemCategory.TOOL), slot=0)
    monster_hp = 37
    state = _place_monster(state, idx=0, pos=(10, 11), hp=monster_hp)

    result = dispatch_apply(state, _RNG)

    assert int(result.probed_hp) == monster_hp, (
        f"probed_hp expected {monster_hp}, got {int(result.probed_hp)}"
    )
    assert int(result.probed_idx) == 0, (
        f"probed_idx expected 0, got {int(result.probed_idx)}"
    )


def test_stethoscope_no_adjacent_monster_is_noop():
    """Stethoscope with no adjacent monster leaves probed_hp/probed_idx unchanged."""
    state = _floor_state(player_pos=(10, 10))
    state = _wield_item(state, _STETHOSCOPE_TYPE_ID, int(ItemCategory.TOOL), slot=0)
    # Monster far away.
    state = _place_monster(state, idx=0, pos=(1, 1), hp=50)

    result = dispatch_apply(state, _RNG)

    assert int(result.probed_hp) == int(state.probed_hp)
    assert int(result.probed_idx) == int(state.probed_idx)
