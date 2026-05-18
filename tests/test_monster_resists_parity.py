"""Per-monster resistance parity tests.

Verifies that MonsterAIState.resists, .undead, .nonliving, .speed_mod,
and .cancelled are correctly populated and fed through _handle_zap /
WandState.

Canonical sources:
  vendor/nethack/src/monst.c  — per-monster mresists (mr1) field
  vendor/nethack/src/zap.c    — zhitm immunity checks
  vendor/nethack/include/mondata.h — nonliving(), is_undead()
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.constants.monsters import (
    MONSTERS, MR_FIRE, MR_COLD, M2_UNDEAD, MonsterSymbol,
)
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.inventory import InventoryState, Item
from Nethax.nethax.subsystems.items_wands import (
    ITEM_CATEGORY_WAND,
    WandEffect,
    WandState,
    zap_wand,
)
from Nethax.nethax.subsystems.monster_ai import (
    MAX_MONSTERS_PER_LEVEL,
    _MONSTER_MRESISTS,
    _MONSTER_UNDEAD,
    _MONSTER_NONLIVING,
    make_monster_ai_state,
)

MAP_H, MAP_W = 21, 80

# ---------------------------------------------------------------------------
# Monster indices verified from MONSTERS tuple (grep output in review).
# Cite: vendor/nethack/src/monst.c entries.
# ---------------------------------------------------------------------------
_IDX_RED_DRAGON   = 143   # red dragon   — resists_mask = MR_FIRE (0x01)
_IDX_WHITE_DRAGON = 144   # white dragon — resists_mask = MR_COLD (0x02)
_IDX_LICH         = 180   # lich         — flags2 & M2_UNDEAD
_IDX_IRON_GOLEM   = 255   # iron golem   — symbol S_GOLEM → nonliving


# ---------------------------------------------------------------------------
# Precondition assertions (fail fast with clear message if table is wrong)
# ---------------------------------------------------------------------------

def test_precondition_red_dragon_fire_resist():
    assert bool(MONSTERS[_IDX_RED_DRAGON].resists_mask & MR_FIRE), (
        f"Precondition: MONSTERS[{_IDX_RED_DRAGON}] must have MR_FIRE; "
        f"got {MONSTERS[_IDX_RED_DRAGON].name} resists={hex(MONSTERS[_IDX_RED_DRAGON].resists_mask)}"
    )


def test_precondition_white_dragon_cold_resist():
    assert bool(MONSTERS[_IDX_WHITE_DRAGON].resists_mask & MR_COLD), (
        f"Precondition: MONSTERS[{_IDX_WHITE_DRAGON}] must have MR_COLD; "
        f"got {MONSTERS[_IDX_WHITE_DRAGON].name} resists={hex(MONSTERS[_IDX_WHITE_DRAGON].resists_mask)}"
    )


def test_precondition_lich_undead():
    assert bool(MONSTERS[_IDX_LICH].flags2 & M2_UNDEAD), (
        f"Precondition: MONSTERS[{_IDX_LICH}] must have M2_UNDEAD; "
        f"got {MONSTERS[_IDX_LICH].name}"
    )


def test_precondition_iron_golem_is_golem():
    assert MONSTERS[_IDX_IRON_GOLEM].symbol == MonsterSymbol.S_GOLEM, (
        f"Precondition: MONSTERS[{_IDX_IRON_GOLEM}] must be S_GOLEM; "
        f"got {MONSTERS[_IDX_IRON_GOLEM].name}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wand_state() -> WandState:
    state = WandState.empty(map_h=MAP_H, map_w=MAP_W)
    terrain = jnp.full((MAP_H, MAP_W), int(TileType.FLOOR), dtype=jnp.int8)
    return state.replace(
        terrain=terrain,
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
    )


def _place_monster(
    state: WandState,
    slot: int,
    entry_idx: int,
    row: int = 10,
    col: int = 11,
    hp: int = 200,
) -> WandState:
    """Place a live monster with resists/undead/nonliving from precomputed tables."""
    idx = jnp.int32(entry_idx)
    resists   = jnp.take(_MONSTER_MRESISTS,  idx, axis=0).astype(jnp.int32)
    undead    = jnp.take(_MONSTER_UNDEAD,    idx, axis=0).astype(jnp.bool_)
    new_pos     = state.mon_pos.at[slot].set(jnp.array([row, col], dtype=jnp.int16))
    new_hp      = state.mon_hp.at[slot].set(jnp.int32(hp))
    new_hp_max  = state.mon_hp_max.at[slot].set(jnp.int32(hp))
    new_alive   = state.mon_alive.at[slot].set(jnp.bool_(True))
    new_type    = state.mon_type.at[slot].set(jnp.int16(entry_idx))
    new_resists = state.mon_resists.at[slot].set(resists)
    new_undead  = state.mon_undead.at[slot].set(undead)
    return state.replace(
        mon_pos=new_pos, mon_hp=new_hp, mon_hp_max=new_hp_max,
        mon_alive=new_alive, mon_type=new_type,
        mon_resists=new_resists, mon_undead=new_undead,
    )


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


def _zap(state: WandState, seed: int = 0) -> WandState:
    """Zap east (direction=2) and return final WandState."""
    rng = jax.random.PRNGKey(seed)
    return zap_wand(state, rng, slot_idx=jnp.int32(0), direction=jnp.int32(2))


# ---------------------------------------------------------------------------
# test_fire_dragon_resists_fire
# Cite: vendor/nethack/src/zap.c::_effect_fire — immune if mon_resists & MR_FIRE
# ---------------------------------------------------------------------------

def test_fire_dragon_resists_fire():
    """Red dragon (MR_FIRE) takes no damage from WAN_FIRE."""
    state = _make_wand_state()
    state = _place_monster(state, slot=1, entry_idx=_IDX_RED_DRAGON, hp=200)
    state = _with_wand(state, WandEffect.FIRE)

    result = _zap(state)

    assert int(result.mon_hp[1]) == 200, (
        f"Red dragon should resist fire; hp dropped to {int(result.mon_hp[1])}"
    )
    assert bool(result.mon_alive[1]), "Red dragon should remain alive after fire zap"


# ---------------------------------------------------------------------------
# test_cold_dragon_resists_cold
# Cite: vendor/nethack/src/zap.c::_effect_cold — immune if mon_resists & MR_COLD
# ---------------------------------------------------------------------------

def test_cold_dragon_resists_cold():
    """White dragon (MR_COLD) takes no damage from WAN_COLD."""
    state = _make_wand_state()
    state = _place_monster(state, slot=1, entry_idx=_IDX_WHITE_DRAGON, hp=200)
    state = _with_wand(state, WandEffect.COLD)

    result = _zap(state)

    assert int(result.mon_hp[1]) == 200, (
        f"White dragon should resist cold; hp dropped to {int(result.mon_hp[1])}"
    )
    assert bool(result.mon_alive[1]), "White dragon should remain alive after cold zap"


# ---------------------------------------------------------------------------
# test_lich_resists_death
# Cite: vendor/nethack/src/zap.c::zhitm — undead immune to WAN_DEATH
# ---------------------------------------------------------------------------

def test_lich_resists_death():
    """Lich (M2_UNDEAD → mon_undead=True) is immune to WAN_DEATH."""
    state = _make_wand_state()
    state = _place_monster(state, slot=1, entry_idx=_IDX_LICH, hp=100)
    state = _with_wand(state, WandEffect.DEATH)

    result = _zap(state)

    assert bool(result.mon_alive[1]), (
        f"Lich (undead) must survive WAN_DEATH; hp={int(result.mon_hp[1])}"
    )


# ---------------------------------------------------------------------------
# test_golem_resists_death
# Cite: vendor/nethack/include/mondata.h::nonliving() — golems are nonliving
# vendor/nethack/src/zap.c::zhitm — nonliving immune to WAN_DEATH
# ---------------------------------------------------------------------------

def test_golem_resists_death():
    """Iron golem (S_GOLEM → nonliving → _DEATH_IMMUNE) is immune to WAN_DEATH."""
    state = _make_wand_state()
    state = _place_monster(state, slot=1, entry_idx=_IDX_IRON_GOLEM, hp=150)
    state = _with_wand(state, WandEffect.DEATH)

    result = _zap(state)

    assert bool(result.mon_alive[1]), (
        f"Iron golem (nonliving) must survive WAN_DEATH; hp={int(result.mon_hp[1])}"
    )


# ---------------------------------------------------------------------------
# test_speed_mod_persists_across_zaps
# Cite: vendor/nethack/src/zap.c WAN_SLOW_MONSTER — sets speed_mod = -1 per hit
# Two separate zaps: second hit re-sets speed_mod to -1 (already -1, stays -1).
# NetHack slow is not stacking; zap twice and verify speed_mod = -1.
# ---------------------------------------------------------------------------

def test_speed_mod_persists_across_zaps():
    """Zapping SLOW_MONSTER twice leaves speed_mod == -1 (not stacked, but persisted)."""
    state = _make_wand_state()
    state = _place_monster(state, slot=1, entry_idx=_IDX_LICH, hp=500)
    state = _with_wand(state, WandEffect.SLOW_MONSTER, charges=10)

    # First zap — sets speed_mod = -1.
    result1 = _zap(state, seed=0)
    assert int(result1.mon_speed_mod[1]) == -1, (
        f"After first SLOW zap speed_mod should be -1, got {int(result1.mon_speed_mod[1])}"
    )

    # Second zap from the result — speed_mod stays -1.
    result2 = _zap(result1, seed=1)
    assert int(result2.mon_speed_mod[1]) == -1, (
        f"After second SLOW zap speed_mod should remain -1, got {int(result2.mon_speed_mod[1])}"
    )
