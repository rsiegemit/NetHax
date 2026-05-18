"""Wand damage parity tests — vendor/nethack/src/zap.c::ubuzz / zhitm.

Tests verify that ray wand damage dice match vendor:
  WAN_FIRE / COLD / LIGHTNING: d(6, 6) = 6d6, range 6-36, mean 21.
  WAN_MAGIC_MISSILE:            d(2, 6) = 2d6, range 2-12, mean 7.

Also verifies resistance gates from zhitm():
  FIRE  resistance  → full immunity (0 damage).
  MAGIC resistance  → full immunity for magic missile.
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
    MR_MAGIC,
    zap_wand,
)
from Nethax.nethax.constants.monsters import MR_FIRE, MR_COLD, MR_ELEC
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.inventory import InventoryState, Item, MAX_INVENTORY_SLOTS
from Nethax.nethax.subsystems.monster_ai import MAX_MONSTERS_PER_LEVEL

MAP_H, MAP_W = 21, 80
_BASE_RNG = jax.random.PRNGKey(0)


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
    hp: int = 200,
    resists: int = 0,
) -> WandState:
    new_pos     = state.mon_pos.at[slot].set(jnp.array([row, col], dtype=jnp.int16))
    new_hp      = state.mon_hp.at[slot].set(jnp.int32(hp))
    new_alive   = state.mon_alive.at[slot].set(jnp.bool_(True))
    new_resists = state.mon_resists.at[slot].set(jnp.int32(resists))
    return state.replace(
        mon_pos=new_pos, mon_hp=new_hp,
        mon_alive=new_alive, mon_resists=new_resists,
    )


def _with_wand(state: WandState, effect: WandEffect, charges: int = 10) -> WandState:
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
    inv = InventoryState.from_items([wand_item])
    return state.replace(inventory=inv)


def _avg_damage(effect: WandEffect, n_trials: int = 100, resists: int = 0) -> float:
    """Zap effect at a monster n_trials times, return mean damage dealt."""
    base = _make_state()
    base = _place_monster(base, slot=1, row=10, col=11, hp=10_000, resists=resists)
    base = _with_wand(base, effect, charges=n_trials + 5)

    initial_hp = int(base.mon_hp[1])
    total_dmg = 0

    for i in range(n_trials):
        rng = jax.random.PRNGKey(i + 1)
        result = zap_wand(base, rng, slot_idx=jnp.int32(0), direction=jnp.int32(2))
        total_dmg += initial_hp - int(result.mon_hp[1])

    return total_dmg / n_trials


# ---------------------------------------------------------------------------
# 6d6 range tests (mean = 21, 95% CI roughly [18, 26] over 100 trials)
# ---------------------------------------------------------------------------

def test_wan_fire_average_damage_in_6d6_range():
    """WAN_FIRE average damage over 100 trials must be in [18, 26] (true mean 21)."""
    avg = _avg_damage(WandEffect.FIRE)
    assert 18 <= avg <= 26, f"WAN_FIRE mean damage {avg:.1f} outside [18, 26]"


def test_wan_cold_average_damage_in_6d6_range():
    """WAN_COLD average damage over 100 trials must be in [18, 26] (true mean 21)."""
    avg = _avg_damage(WandEffect.COLD)
    assert 18 <= avg <= 26, f"WAN_COLD mean damage {avg:.1f} outside [18, 26]"


def test_wan_lightning_average_damage_in_6d6_range():
    """WAN_LIGHTNING average damage over 100 trials must be in [18, 26] (true mean 21)."""
    avg = _avg_damage(WandEffect.LIGHTNING)
    assert 18 <= avg <= 26, f"WAN_LIGHTNING mean damage {avg:.1f} outside [18, 26]"


# ---------------------------------------------------------------------------
# 2d6 range test (mean = 7, 95% CI roughly [5, 9] over 100 trials)
# ---------------------------------------------------------------------------

def test_wan_magic_missile_average_damage_in_2d6_range():
    """WAN_MAGIC_MISSILE average damage over 100 trials must be in [5, 9] (true mean 7)."""
    avg = _avg_damage(WandEffect.MAGIC_MISSILE)
    assert 5 <= avg <= 9, f"WAN_MAGIC_MISSILE mean damage {avg:.1f} outside [5, 9]"


# ---------------------------------------------------------------------------
# Resistance gate tests
# ---------------------------------------------------------------------------

def test_fire_resist_blocks_damage():
    """Fire-resistant monster must take 0 damage from WAN_FIRE (vendor: full immunity)."""
    avg = _avg_damage(WandEffect.FIRE, n_trials=100, resists=MR_FIRE)
    assert avg == 0, f"Fire-resistant monster took mean {avg:.1f} damage (expected 0)"


def test_fire_resist_blocks_or_halves():
    """Fire-resistant monster mean damage must be < 14 (< half of true mean 21)."""
    avg = _avg_damage(WandEffect.FIRE, n_trials=100, resists=MR_FIRE)
    assert avg < 14, f"WAN_FIRE vs fire-resistant: mean {avg:.1f} >= 14"


def test_magic_resistance_blocks_missile():
    """Magic-resistant monster must take 0 damage from WAN_MAGIC_MISSILE in all 100 trials."""
    base = _make_state()
    base = _place_monster(base, slot=1, row=10, col=11, hp=10_000, resists=MR_MAGIC)
    base = _with_wand(base, WandEffect.MAGIC_MISSILE, charges=105)
    initial_hp = int(base.mon_hp[1])

    for i in range(100):
        rng = jax.random.PRNGKey(i + 1)
        result = zap_wand(base, rng, slot_idx=jnp.int32(0), direction=jnp.int32(2))
        dmg = initial_hp - int(result.mon_hp[1])
        assert dmg == 0, (
            f"Trial {i}: magic-resistant monster took {dmg} damage from WAN_MAGIC_MISSILE"
        )
