"""Polish-parity tests for 5 scroll special effects.

Covers:
  1. SCR_FIRE blessed    — AoE explosion damages adjacent monsters (fire-res=0).
  2. SCR_EARTH           — boulders dropped at N/E/S/W cardinal tiles.
  3. SCR_STINKING_CLOUD  — positional cloud state set on player pos.
  4. SCR_GOLD_DETECTION  — confused/cursed reveals traps on current level.
  5. SCR_FOOD_DETECTION  — counts FOOD items; stores in last_food_count.

Vendor citations:
  read.c::seffect_fire ~1850, seffect_earth ~1919,
  do_stinking_cloud ~3082, seffect_gold_detection ~2035,
  seffect_food_detection ~2046.
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
    ItemCategory,
    make_item,
    _empty_ground_items_array,
)
from Nethax.nethax.subsystems.items_scrolls import (
    ScrollEffect,
    _SCROLL_BASE_ID,
    BOULDER_TYPE_ID,
    read_scroll,
)
from Nethax.nethax.subsystems.status_effects import TimedStatus
from Nethax.nethax.dungeon.branches import N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W

_RNG = jax.random.PRNGKey(42)
_BUC_UNCURSED = 2
_BUC_CURSED   = 1
_BUC_BLESSED  = 3


def _scroll(effect: ScrollEffect, buc: int = _BUC_UNCURSED):
    return make_item(
        category=int(ItemCategory.SCROLL),
        type_id=_SCROLL_BASE_ID + int(effect),
        quantity=1,
        buc_status=buc,
    )


def _state(items) -> EnvState:
    state = EnvState.default(_RNG)
    return state.replace(inventory=InventoryState.from_items(items))


# ---------------------------------------------------------------------------
# 1. SCR_FIRE blessed — AoE explosion damages adjacent monsters
# vendor/nethack/src/read.c::seffect_fire ~1850
# ---------------------------------------------------------------------------

def test_blessed_fire_aoe_damage():
    """Blessed SCR_FIRE deals AoE damage to all 3 adjacent (non-fire-res) monsters.

    vendor read.c::seffect_fire ~1893: blessed path calls explode() centered on
    player, hitting the 3x3 neighbourhood.  Fire-resistant monsters take 0.
    """
    scroll = _scroll(ScrollEffect.FIRE, buc=_BUC_BLESSED)
    state  = _state([scroll])

    # Place player at (5, 5)
    state = state.replace(player_pos=jnp.array([5, 5], dtype=jnp.int16))

    # Inject 3 monsters adjacent to player (non-fire-resistant entry_idx=0)
    mai = state.monster_ai
    new_pos = mai.pos.at[0].set(jnp.array([4, 5], dtype=jnp.int16))  # N
    new_pos = new_pos.at[1].set(jnp.array([5, 6], dtype=jnp.int16))  # E
    new_pos = new_pos.at[2].set(jnp.array([6, 5], dtype=jnp.int16))  # S
    new_hp  = mai.hp.at[0].set(jnp.int32(30))
    new_hp  = new_hp.at[1].set(jnp.int32(30))
    new_hp  = new_hp.at[2].set(jnp.int32(30))
    new_alive = mai.alive.at[0].set(jnp.bool_(True))
    new_alive = new_alive.at[1].set(jnp.bool_(True))
    new_alive = new_alive.at[2].set(jnp.bool_(True))
    # entry_idx=0: first monster in MONSTERS table — not fire-resistant
    new_entry = mai.entry_idx.at[0].set(jnp.int16(0))
    new_entry = new_entry.at[1].set(jnp.int16(0))
    new_entry = new_entry.at[2].set(jnp.int16(0))
    new_mai = mai.replace(pos=new_pos, hp=new_hp, alive=new_alive, entry_idx=new_entry)
    state = state.replace(monster_ai=new_mai)

    result = read_scroll(state, _RNG, 0)

    # All 3 monsters must have taken damage (hp < 30)
    assert int(result.monster_ai.hp[0]) < 30, "Monster N must take fire damage"
    assert int(result.monster_ai.hp[1]) < 30, "Monster E must take fire damage"
    assert int(result.monster_ai.hp[2]) < 30, "Monster S must take fire damage"


# ---------------------------------------------------------------------------
# 2. SCR_EARTH — boulders at 4 cardinal tiles
# vendor/nethack/src/read.c::seffect_earth ~1919
# ---------------------------------------------------------------------------

def test_earth_drops_boulders_at_cardinals():
    """SCR_EARTH places boulder ground items at N/E/S/W of player position.

    vendor read.c::seffect_earth ~1953: loops surrounding squares, calls
    drop_boulder_on_monster.  Boulders persist as ground items.
    """
    scroll = _scroll(ScrollEffect.EARTH)
    state  = _state([scroll])
    # Place player at (10, 40) — well away from edges
    state  = state.replace(player_pos=jnp.array([10, 40], dtype=jnp.int16))

    result = read_scroll(state, _RNG, 0)

    b  = int(result.dungeon.current_branch)
    lv = int(result.dungeon.current_level) - 1

    # Cardinal positions: N=(9,40), E=(10,41), S=(11,40), W=(10,39)
    cardinals = [(9, 40), (10, 41), (11, 40), (10, 39)]
    for (r, c) in cardinals:
        cat = int(result.ground_items.category[b, lv, r, c, 0])
        tid = int(result.ground_items.type_id[b, lv, r, c, 0])
        assert cat == int(ItemCategory.ROCK), (
            f"Expected ROCK_CLASS boulder at ({r},{c}), got category={cat}"
        )
        assert tid == BOULDER_TYPE_ID, (
            f"Expected type_id={BOULDER_TYPE_ID} at ({r},{c}), got {tid}"
        )


# ---------------------------------------------------------------------------
# 3. SCR_STINKING_CLOUD — positional cloud state
# vendor/nethack/src/read.c::do_stinking_cloud ~3082
# ---------------------------------------------------------------------------

def test_stinking_cloud_sets_state():
    """SCR_STINKING_CLOUD sets cloud_pos == player_pos and cloud_turns > 0.

    vendor read.c::do_stinking_cloud ~3082: create_gas_cloud(cc.x, cc.y,
    15+10*bcsign, 8+4*bcsign).  We record cloud_pos and cloud_turns in state.
    """
    scroll = _scroll(ScrollEffect.STINKING_CLOUD)
    state  = _state([scroll])
    state  = state.replace(player_pos=jnp.array([7, 20], dtype=jnp.int16))

    result = read_scroll(state, _RNG, 0)

    assert int(result.cloud_turns) > 0, (
        f"cloud_turns must be > 0 after reading stinking cloud, got {int(result.cloud_turns)}"
    )
    assert int(result.cloud_pos[0]) == 7, (
        f"cloud_pos row must match player_pos row (7), got {int(result.cloud_pos[0])}"
    )
    assert int(result.cloud_pos[1]) == 20, (
        f"cloud_pos col must match player_pos col (20), got {int(result.cloud_pos[1])}"
    )
    assert int(result.cloud_radius) == 3, (
        f"cloud_radius must be 3, got {int(result.cloud_radius)}"
    )


# ---------------------------------------------------------------------------
# 4. SCR_GOLD_DETECTION confused/cursed — reveals traps
# vendor/nethack/src/read.c::seffect_gold_detection ~2035
# ---------------------------------------------------------------------------

def test_gold_detection_cursed_reveals_traps():
    """Cursed SCR_GOLD_DETECTION reveals all traps on the current level.

    vendor read.c::seffect_gold_detection ~2041:
      if (confused || scursed): trap_detect(sobj) — reveal trap positions.
    """
    scroll = _scroll(ScrollEffect.GOLD_DETECTION, buc=_BUC_CURSED)
    state  = _state([scroll])

    # Pre-condition: no traps revealed
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    max_lv = state.terrain.shape[1]
    flat_lv = b * max_lv + lv
    assert not bool(jnp.any(state.traps.revealed[flat_lv])), (
        "Pre-condition: traps.revealed must be all False"
    )

    result = read_scroll(state, _RNG, 0)

    assert bool(jnp.all(result.traps.revealed[flat_lv])), (
        "Cursed gold detection must set traps.revealed[:,:] = True on current level"
    )


# ---------------------------------------------------------------------------
# 5. SCR_FOOD_DETECTION — counts FOOD items on level
# vendor/nethack/src/read.c::seffect_food_detection ~2046
# ---------------------------------------------------------------------------

def test_food_detection_counts():
    """SCR_FOOD_DETECTION sets last_food_count equal to food items on current level.

    vendor read.c::seffect_food_detection ~2046: food_detect() reveals food.
    Simplification: count FOOD_CLASS ground items; store in last_food_count.
    """
    scroll = _scroll(ScrollEffect.FOOD_DETECTION)
    state  = _state([scroll])

    # Place 3 food items on the current level at distinct tiles
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    food_cat = jnp.int8(int(ItemCategory.FOOD))
    food_tid = jnp.int16(1)

    gi = state.ground_items
    gi = gi.replace(
        category=gi.category
            .at[b, lv, 3, 5, 0].set(food_cat)
            .at[b, lv, 4, 6, 0].set(food_cat)
            .at[b, lv, 5, 7, 0].set(food_cat),
        type_id=gi.type_id
            .at[b, lv, 3, 5, 0].set(food_tid)
            .at[b, lv, 4, 6, 0].set(food_tid)
            .at[b, lv, 5, 7, 0].set(food_tid),
        quantity=gi.quantity
            .at[b, lv, 3, 5, 0].set(jnp.int16(1))
            .at[b, lv, 4, 6, 0].set(jnp.int16(1))
            .at[b, lv, 5, 7, 0].set(jnp.int16(1)),
    )
    state = state.replace(ground_items=gi)

    result = read_scroll(state, _RNG, 0)

    assert int(result.last_food_count) == 3, (
        f"SCR_FOOD_DETECTION must set last_food_count=3, got {int(result.last_food_count)}"
    )
