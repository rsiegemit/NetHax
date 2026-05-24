"""Underwater polish parity tests — MAGIC_BREATHING, SWIMMING, per-turn rust.

Vendor references:
  vendor/nethack/include/you.h   MAGIC_BREATHING (MAGICAL_BREATHING = 52)
  vendor/nethack/include/prop.h  SWIMMING = 51, MAGICAL_BREATHING = 52
  vendor/nethack/src/trap.c      drown() lines 5059-5195 (drowning damage)
  vendor/nethack/src/trap.c      water_damage() line 5086 (inventory rust)
  vendor/nethack/src/hack.c      pooleffects() line 3304 (enter water)
  vendor/nethack/src/hack.c      swimeffect() line 3237 (leave water / surface)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants import TileType
from Nethax.nethax.subsystems.action_dispatch import _try_step
from Nethax.nethax.subsystems.status_effects import Intrinsic, add_intrinsic
from Nethax.nethax.subsystems.water import water_step

_RNG = jax.random.PRNGKey(7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(player_pos=(10, 10)):
    env = NethaxEnv()
    state, _ = env.reset(_RNG)
    floor_terrain = jnp.full(
        state.terrain[0, 0].shape, int(TileType.FLOOR), dtype=state.terrain.dtype
    )
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor_terrain),
        player_pos=jnp.array(player_pos, dtype=jnp.int16),
    )
    mai = state.monster_ai.replace(
        alive=jnp.zeros_like(state.monster_ai.alive),
        pos=jnp.full_like(state.monster_ai.pos, -1),
    )
    return state.replace(monster_ai=mai)


def _place_tile(state, row, col, tile):
    return state.replace(
        terrain=state.terrain.at[0, 0, row, col].set(jnp.int8(tile))
    )


def _set_in_water(state, flag: bool):
    return state.replace(player_in_water=jnp.bool_(flag))


def _grant(state, intrinsic: Intrinsic):
    new_status = add_intrinsic(state.status, int(intrinsic))
    return state.replace(status=new_status)


# ---------------------------------------------------------------------------
# test 1: MAGIC_BREATHING suppresses drowning damage
# ---------------------------------------------------------------------------

def test_magic_breathing_skips_drowning_damage():
    """With MAGIC_BREATHING, water_step must not reduce HP.

    Cite: vendor/nethack/include/you.h MAGIC_BREATHING;
          vendor/nethack/include/prop.h MAGICAL_BREATHING = 52.
    """
    state = _make_state()
    state = _set_in_water(state, True)
    state = state.replace(player_hp=jnp.int32(100), player_hp_max=jnp.int32(100))
    state = _grant(state, Intrinsic.MAGIC_BREATHING)

    rng = _RNG
    for _ in range(10):
        rng, sub = jax.random.split(rng)
        state = water_step(state, sub)

    assert int(state.player_hp) == 100, (
        "MAGIC_BREATHING should suppress drowning damage (you.h MAGIC_BREATHING / prop.h:52)"
    )


# ---------------------------------------------------------------------------
# test 2: SWIMMING keeps player on surface when stepping into POOL
# ---------------------------------------------------------------------------

def test_swimming_stays_on_surface():
    """With SWIMMING, stepping onto POOL must NOT set player_in_water.

    Cite: vendor/nethack/src/hack.c::pooleffects line 3304 — swimming hero
    stays on the surface rather than submerging (hack.c swimeffect / prop.h:51).
    """
    state = _make_state(player_pos=(10, 10))
    state = _place_tile(state, 10, 11, TileType.POOL)
    state = _set_in_water(state, False)
    state = _grant(state, Intrinsic.SWIMMING)

    new_state = _try_step(state, 0, 1, _RNG)

    assert tuple(new_state.player_pos.tolist()) == (10, 11), (
        "Player with SWIMMING should move onto the POOL tile."
    )
    assert not bool(new_state.player_in_water), (
        "SWIMMING hero on POOL tile should NOT be underwater (hack.c swimeffect / prop.h:51)"
    )


# ---------------------------------------------------------------------------
# test 3: No breathing — stepping into pool then water_step decreases HP
# ---------------------------------------------------------------------------

def test_no_breath_still_drowns():
    """Without any water-breathing, stepping into pool then ticking causes HP loss.

    Cite: vendor/nethack/src/hack.c::pooleffects line 3304 (enter),
          vendor/nethack/src/trap.c::drown() lines 5059-5195 (damage).

    Vendor drowning is binary: ``drown()`` either lets the hero crawl out
    or calls ``done(DROWNING)`` (instakill).  Our ``water_step`` models
    this with a per-5-turns rnl(50)-vs-turns_underwater check; the kill
    probability grows from 12% at turn 5 to ~100% at turn 50.  We tick
    50 turns so the insta-drown fires deterministically.
    """
    state = _make_state(player_pos=(10, 10))
    state = _place_tile(state, 10, 11, TileType.POOL)
    state = _set_in_water(state, False)
    state = state.replace(player_hp=jnp.int32(200), player_hp_max=jnp.int32(200))

    # Step into pool — sets player_in_water.
    state = _try_step(state, 0, 1, _RNG)
    assert bool(state.player_in_water), "Stepping into POOL should set player_in_water."

    # Apply 50 drowning ticks; by turn 50 the rnl(50) <= turns_underwater
    # check at turn 50 is guaranteed to trigger insta-drown.
    rng = _RNG
    for _ in range(50):
        rng, sub = jax.random.split(rng)
        state = water_step(state, sub)

    assert int(state.player_hp) < 200, (
        "No water-breathing: drowning damage should reduce HP (trap.c:5059)"
    )


# ---------------------------------------------------------------------------
# test 4: Iron items rust per turn while underwater
# ---------------------------------------------------------------------------

def test_water_rusts_per_turn():
    """Iron items gain oeroded after 10 turns underwater (water_damage_chain).

    Cite: vendor/nethack/src/trap.c::water_damage() line 5086 — each iron item
    has a 50% per-turn rust chance.
    """
    from Nethax.nethax.subsystems.inventory import InventoryState, make_item, ItemCategory
    from Nethax.nethax.obs.inv_strs import _OBJECT_EROSION_CLASS

    state = _make_state()
    state = _set_in_water(state, True)
    state = state.replace(player_hp=jnp.int32(1000), player_hp_max=jnp.int32(1000))

    # Find a rustprone type_id (emat_class == 1).
    rustprone_ids = [i for i, v in enumerate(_OBJECT_EROSION_CLASS.tolist()) if v == 1]
    assert rustprone_ids, "Need at least one rustprone item type for this test."
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
