"""Underwater mechanics parity tests.

Vendor references:
  vendor/nethack/src/hack.c::pooleffects line 3304  (enter water sets flag)
  vendor/nethack/src/hack.c lines 3237-3268          (leave water clears flag)
  vendor/nethack/src/hack.c lines 1016-1023           (no diagonal underwater)
  vendor/nethack/src/trap.c::drown() lines 5059-5195 (drowning damage)
  vendor/nethack/src/display.c line 944               (blind FOV underwater)
  vendor/nethack/src/trap.c::water_damage() line 5086 (inventory rust)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.state import EnvState
from Nethax.nethax.constants import TileType
from Nethax.nethax.constants.actions import (
    CompassCardinalDirection,
    CompassIntercardinalDirection,
)
from Nethax.nethax.subsystems.action_dispatch import _apply_fov, _try_step
from Nethax.nethax.subsystems.inventory import (
    MAX_INVENTORY_SLOTS, ItemCategory, make_item,
)
from Nethax.nethax.subsystems.water import water_step

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_floor_state(player_pos=(10, 10)):
    """Open all-FLOOR map, player at given position, monsters cleared."""
    env = NethaxEnv()
    state, _ = env.reset(_RNG)

    floor_terrain = jnp.full(
        state.terrain[0, 0].shape, int(TileType.FLOOR), dtype=state.terrain.dtype
    )
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor_terrain),
        player_pos=jnp.array(player_pos, dtype=jnp.int16),
    )
    # Clear all monsters to avoid interference.
    mai = state.monster_ai.replace(
        alive=jnp.zeros_like(state.monster_ai.alive),
        pos=jnp.full_like(state.monster_ai.pos, -1),
    )
    state = state.replace(monster_ai=mai)
    return env, state


def _place_tile(state, row, col, tile):
    """Place a tile at (branch=0, level=0, row, col)."""
    return state.replace(
        terrain=state.terrain.at[0, 0, row, col].set(jnp.int8(tile))
    )


def _set_player_water(state, in_water: bool):
    return state.replace(player_in_water=jnp.bool_(in_water))


# ---------------------------------------------------------------------------
# test 1: step onto POOL tile sets player_in_water
# ---------------------------------------------------------------------------

def test_step_into_pool_sets_in_water():
    """Stepping onto a POOL tile must set player_in_water=True.

    Cite: vendor/nethack/src/hack.c::pooleffects line 3304.
    """
    _env, state = _make_floor_state(player_pos=(10, 10))
    # Place a POOL one step east.
    state = _place_tile(state, 10, 11, TileType.POOL)
    state = _set_player_water(state, False)

    new_state = _try_step(state, 0, 1, _RNG)

    assert bool(new_state.player_in_water), (
        "Stepping onto POOL should set player_in_water=True "
        "(hack.c::pooleffects line 3304)"
    )
    assert tuple(new_state.player_pos.tolist()) == (10, 11), (
        "Player should have moved to the POOL tile."
    )


# ---------------------------------------------------------------------------
# test 2: step out of POOL clears player_in_water
# ---------------------------------------------------------------------------

def test_step_out_of_pool_clears_in_water():
    """After leaving a water tile, player_in_water must become False.

    Cite: vendor/nethack/src/hack.c lines 3237-3268.
    """
    _env, state = _make_floor_state(player_pos=(10, 10))
    # Player stands on a POOL tile and is marked as in-water.
    state = _place_tile(state, 10, 10, TileType.POOL)
    state = _set_player_water(state, True)
    # FLOOR to the east.
    assert int(state.terrain[0, 0, 10, 11]) == int(TileType.FLOOR)

    new_state = _try_step(state, 0, 1, _RNG)

    assert not bool(new_state.player_in_water), (
        "Stepping off a POOL tile should clear player_in_water "
        "(hack.c lines 3237-3268)"
    )
    assert tuple(new_state.player_pos.tolist()) == (10, 11)


# ---------------------------------------------------------------------------
# test 3: diagonal move blocked underwater
# ---------------------------------------------------------------------------

def test_underwater_blocks_diagonal():
    """Diagonal steps must be blocked while player_in_water is True.

    Cite: vendor/nethack/src/hack.c lines 1016-1023.
    """
    _env, state = _make_floor_state(player_pos=(10, 10))
    state = _set_player_water(state, True)
    original_pos = tuple(state.player_pos.tolist())

    # Attempt NE diagonal (dy=-1, dx=1).
    new_state = _try_step(state, -1, 1, _RNG)

    assert tuple(new_state.player_pos.tolist()) == original_pos, (
        "Diagonal move should be blocked underwater (hack.c:1016)"
    )


# ---------------------------------------------------------------------------
# test 4: drowning damage
# ---------------------------------------------------------------------------

def test_underwater_drowning_damage():
    """One-shot ``drown()`` on entry kills the player without water breathing.

    Vendor drown is a one-shot event (trap.c:5057-5198); after
    ``water_damage_chain`` and the safe-creature early return it tries
    ``emergency_disrobe`` + ``crawl_out`` and otherwise calls
    ``done(DROWNING)``.  Cite: vendor/nethack/src/trap.c::drown lines
    5151-5187.

    Calling ``drown(state, rng)`` directly with no breath/swim/amphib
    intrinsics produces either a successful crawl-out (player_in_water
    cleared) or death (hp=0, done=True).  Over many trials at least one
    death must occur.
    """
    from Nethax.nethax.subsystems.water import drown as _drown_fn

    died = False
    rng = _RNG
    for _ in range(50):
        rng, sub = jax.random.split(rng)
        _env, state = _make_floor_state(player_pos=(10, 10))
        state = _set_player_water(state, True)
        state = state.replace(
            player_hp=jnp.int32(200), player_hp_max=jnp.int32(200)
        )
        state = _drown_fn(state, sub)
        if int(state.player_hp) == 0 and bool(state.done):
            died = True
            break

    assert died, (
        "drown() should produce at least one DROWNING death in 50 trials "
        "(trap.c::drown line 5187)"
    )


# ---------------------------------------------------------------------------
# test 5: underwater blind FOV
# ---------------------------------------------------------------------------

def test_underwater_blind_fov():
    """FOV radius must drop when player_in_water is True.

    Cite: vendor/nethack/src/display.c line 944.
    """
    _env, state_surface = _make_floor_state(player_pos=(10, 10))

    state_water = _set_player_water(state_surface, True)
    state_surface = _set_player_water(state_surface, False)

    vis_surface = _apply_fov(state_surface).visible
    vis_water   = _apply_fov(state_water).visible

    count_surface = int(jnp.sum(vis_surface))
    count_water   = int(jnp.sum(vis_water))

    assert count_water < count_surface, (
        f"Underwater FOV ({count_water}) should be smaller than surface FOV "
        f"({count_surface}) — display.c:944"
    )
    # Underwater radius=1 → at most 3×3=9 visible tiles on open floor.
    assert count_water <= 9, (
        f"Underwater FOV radius should be 1 (≤9 tiles), got {count_water}"
    )


# ---------------------------------------------------------------------------
# test 6: water rusts iron items
# ---------------------------------------------------------------------------

def test_water_rusts_iron():
    """Iron items in inventory gain oeroded after 10 turns underwater.

    Cite: vendor/nethack/src/trap.c::water_damage() line 5086.
    """
    from Nethax.nethax.subsystems.inventory import (
        InventoryState, make_item, ItemCategory,
    )
    _env, state = _make_floor_state(player_pos=(10, 10))
    state = _set_player_water(state, True)
    state = state.replace(player_hp=jnp.int32(1000), player_hp_max=jnp.int32(1000))

    # Place an iron weapon (long sword = type_id 41, WEAPON class, iron material).
    # We use a WEAPON with type_id that maps to rustprone (emat_class=1).
    # From obs/inv_strs.py the OBJECTS table: pick type_id=41 (long sword, iron).
    from Nethax.nethax.obs.inv_strs import _OBJECT_EROSION_CLASS
    # Find a rustprone type_id (emat_class == 1).
    rustprone_ids = [i for i, v in enumerate(_OBJECT_EROSION_CLASS.tolist()) if v == 1]
    assert rustprone_ids, "Need at least one rustprone item type."
    iron_type_id = rustprone_ids[0]

    weapon = make_item(
        category=int(ItemCategory.WEAPON),
        type_id=iron_type_id,
        buc_status=2,
        enchantment=0,
        quantity=1,
        weight=40,
        ac_bonus=0,
        is_two_handed=False,
        oeroded=0,
        oerodeproof=False,
    )
    inv = InventoryState.from_items([weapon])
    state = state.replace(inventory=inv)

    rng = _RNG
    for _ in range(10):
        rng, sub = jax.random.split(rng)
        state = water_step(state, sub)

    oeroded_val = int(state.inventory.items.oeroded[0])
    assert oeroded_val >= 1, (
        f"Iron item should have oeroded >= 1 after 10 turns in water, got {oeroded_val} "
        "(trap.c::water_damage line 5086)"
    )
