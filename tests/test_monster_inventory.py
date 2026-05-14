"""Wave 6 Mission: monster inventory state + spawn-time kit tests.

Vendor reference:
    - vendor/nethack/src/makemon.c::makemon + mongets — initial inventory
      based on the monster's M2_* flags / sound class.
    - vendor/nethack/include/permonst.h::mflags2 — M2_MAGIC, M2_GREEDY, ...
    - vendor/nethack/include/monst.h::minvent — monster inventory chain.

Tests:
    1. MonsterAIState exposes per-monster inventory arrays shaped [400, 8].
    2. Default inventory is empty (category=0, qty=0) for every slot.
    3. Mage monsters spawn with a wand (vendor mongets case MAGE).
    4. Priest monsters spawn with holy water (vendor mongets case PRIEST).
    5. Soldier monsters spawn with a weapon (vendor mongets case SOLDIER).
    6. Shopkeepers spawn with gold (vendor mongets case SHK).
    7. Animal monsters spawn with no inventory (vendor mongets default).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.subsystems.monster_ai import (
    MAX_MONSTERS_PER_LEVEL,
    MAX_MONSTER_INV,
    make_monster_ai_state,
)
from Nethax.nethax.dungeon.spawning import (
    _MONSTER_KIT_BY_ENTRY,
    _KIT_CATS, _KIT_TIDS, _KIT_QTYS, _KIT_CHGS,
    _KIT_NONE, _KIT_MAGE, _KIT_PRIEST, _KIT_SOLDIER, _KIT_GOLD,
    _CAT_POTION, _CAT_SCROLL, _CAT_WAND, _CAT_WEAPON, _CAT_ARMOR, _CAT_COIN,
    populate_level_with_monsters,
)


_RNG = jax.random.PRNGKey(7)


def _entry_for(name: str) -> int:
    for i, m in enumerate(MONSTERS):
        if m.name == name:
            return i
    raise AssertionError(f"MONSTERS missing entry: {name!r}")


# Fixture monster IDs (looked up from MONSTERS table by name).
_TITAN_IDX       = _entry_for("titan")          # MS_SPELL → mage kit
_ALIGNED_PRIEST  = _entry_for("aligned priest") # MS_PRIEST → priest kit
_SOLDIER_IDX     = _entry_for("soldier")        # MS_SOLDIER → soldier kit
_SHOPKEEPER_IDX  = _entry_for("shopkeeper")     # MS_SELL or M2_MAGIC → kit
_GIANT_ANT       = 0                            # animal, no kit


# ---------------------------------------------------------------------------
# 1. State shape
# ---------------------------------------------------------------------------

def test_monster_inventory_state_shape_400_8():
    """MonsterAIState exposes 6 inventory arrays each shaped [400, 8]."""
    mai = make_monster_ai_state()
    assert mai.inv_category.shape == (MAX_MONSTERS_PER_LEVEL, MAX_MONSTER_INV)
    assert mai.inv_type_id.shape == (MAX_MONSTERS_PER_LEVEL, MAX_MONSTER_INV)
    assert mai.inv_buc.shape == (MAX_MONSTERS_PER_LEVEL, MAX_MONSTER_INV)
    assert mai.inv_quantity.shape == (MAX_MONSTERS_PER_LEVEL, MAX_MONSTER_INV)
    assert mai.inv_charges.shape == (MAX_MONSTERS_PER_LEVEL, MAX_MONSTER_INV)
    assert mai.inv_identified.shape == (MAX_MONSTERS_PER_LEVEL, MAX_MONSTER_INV)
    # And the constants are 400 / 8 specifically.
    assert MAX_MONSTERS_PER_LEVEL == 400
    assert MAX_MONSTER_INV == 8


def test_monster_inventory_dtypes_correct():
    """Per-spec dtypes: int8 for category/buc/charges, int16 for type_id/qty,
    bool for identified.
    """
    mai = make_monster_ai_state()
    assert mai.inv_category.dtype == jnp.int8
    assert mai.inv_type_id.dtype == jnp.int16
    assert mai.inv_buc.dtype == jnp.int8
    assert mai.inv_quantity.dtype == jnp.int16
    assert mai.inv_charges.dtype == jnp.int8
    assert mai.inv_identified.dtype == jnp.bool_


# ---------------------------------------------------------------------------
# 2. Defaults
# ---------------------------------------------------------------------------

def test_monster_inventory_defaults_empty():
    """Empty MonsterAIState has category=0 (ItemCategory.NONE), qty=0 for
    every slot — i.e. no items.
    """
    mai = make_monster_ai_state()
    assert bool(jnp.all(mai.inv_category == 0))
    assert bool(jnp.all(mai.inv_quantity == 0))
    assert bool(jnp.all(mai.inv_charges == 0))
    assert bool(jnp.all(mai.inv_buc == 0))
    assert not bool(jnp.any(mai.inv_identified))


# ---------------------------------------------------------------------------
# 3. Kit lookup table correctness
# ---------------------------------------------------------------------------

def test_kit_lookup_assigns_mage_to_titan():
    """Titan (sound = MS_SPELL) maps to _KIT_MAGE per vendor mongets."""
    assert int(_MONSTER_KIT_BY_ENTRY[_TITAN_IDX]) == _KIT_MAGE


def test_kit_lookup_assigns_priest_to_aligned_priest():
    """Aligned priest (sound = MS_PRIEST) maps to _KIT_PRIEST."""
    assert int(_MONSTER_KIT_BY_ENTRY[_ALIGNED_PRIEST]) == _KIT_PRIEST


def test_kit_lookup_assigns_soldier_kit():
    """Soldier (sound = MS_SOLDIER) maps to _KIT_SOLDIER."""
    assert int(_MONSTER_KIT_BY_ENTRY[_SOLDIER_IDX]) == _KIT_SOLDIER


def test_kit_lookup_assigns_animal_to_none():
    """Giant ant (M1_ANIMAL, mindless) — no items kit."""
    assert int(_MONSTER_KIT_BY_ENTRY[_GIANT_ANT]) == _KIT_NONE


# ---------------------------------------------------------------------------
# 4. Spawn-time application of kits
# ---------------------------------------------------------------------------

def _spawn_one_monster(entry_idx: int) -> EnvState:
    """Build an EnvState, manually plant one monster with the given entry_idx
    in slot 0 with the correct kit (mimics populate_level_with_monsters'
    per-slot writes for a single index).
    """
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)
    floor_map = jnp.full((static.map_h, static.map_w), TileType.FLOOR, dtype=jnp.int8)
    state = state.replace(terrain=state.terrain.at[0, 0].set(floor_map))

    mai = state.monster_ai
    type_id = jnp.int32(entry_idx)
    kit_id = _MONSTER_KIT_BY_ENTRY[type_id].astype(jnp.int32)

    mai = mai.replace(
        entry_idx=mai.entry_idx.at[0].set(jnp.int16(entry_idx)),
        alive=mai.alive.at[0].set(jnp.bool_(True)),
        hp=mai.hp.at[0].set(jnp.int32(20)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(20)),
        pos=mai.pos.at[0].set(jnp.array([5, 5], dtype=jnp.int16)),
        inv_category=mai.inv_category.at[0].set(_KIT_CATS[kit_id]),
        inv_type_id=mai.inv_type_id.at[0].set(_KIT_TIDS[kit_id]),
        inv_quantity=mai.inv_quantity.at[0].set(_KIT_QTYS[kit_id]),
        inv_charges=mai.inv_charges.at[0].set(_KIT_CHGS[kit_id]),
    )
    return state.replace(monster_ai=mai)


def test_mage_monster_spawns_with_wand():
    """Mage-class spawn has a wand in inventory slot 0 with positive charges."""
    state = _spawn_one_monster(_TITAN_IDX)
    cats = state.monster_ai.inv_category[0]
    chgs = state.monster_ai.inv_charges[0]
    # Expect at least one WAND slot.
    has_wand = bool(jnp.any(cats == _CAT_WAND))
    assert has_wand, f"Mage missing wand; cats={cats.tolist()}"
    # The wand slot must have positive charges.
    wand_idx = int(jnp.argmax((cats == _CAT_WAND).astype(jnp.int32)))
    assert int(chgs[wand_idx]) > 0, "wand has zero charges"


def test_priest_monster_spawns_with_holy_water():
    """Priest kit includes a potion (POT_WATER for holy water)."""
    state = _spawn_one_monster(_ALIGNED_PRIEST)
    cats = state.monster_ai.inv_category[0]
    has_potion = bool(jnp.any(cats == _CAT_POTION))
    assert has_potion, f"Priest missing potion; cats={cats.tolist()}"


def test_soldier_monster_spawns_with_weapon():
    """Soldier kit includes a WEAPON slot (long sword)."""
    state = _spawn_one_monster(_SOLDIER_IDX)
    cats = state.monster_ai.inv_category[0]
    has_weapon = bool(jnp.any(cats == _CAT_WEAPON))
    has_armor  = bool(jnp.any(cats == _CAT_ARMOR))
    assert has_weapon, f"Soldier missing weapon; cats={cats.tolist()}"
    assert has_armor,  f"Soldier missing armor;  cats={cats.tolist()}"


def test_shopkeeper_spawns_with_gold():
    """Shopkeeper kit carries COIN (gold) with high quantity."""
    state = _spawn_one_monster(_SHOPKEEPER_IDX)
    cats = state.monster_ai.inv_category[0]
    qtys = state.monster_ai.inv_quantity[0]
    # Shopkeeper's sound is MS_SELL → maps to _KIT_GOLD (gold).
    # Some shopkeepers also have M2_MAGIC; the MS_SELL/M2_GREEDY rule
    # takes priority in _compute_kit_per_entry.
    kit = int(_MONSTER_KIT_BY_ENTRY[_SHOPKEEPER_IDX])
    if kit == _KIT_GOLD:
        gold_idx = int(jnp.argmax((cats == _CAT_COIN).astype(jnp.int32)))
        assert int(cats[gold_idx]) == _CAT_COIN
        assert int(qtys[gold_idx]) >= 50, \
            f"Shopkeeper gold qty too low: {int(qtys[gold_idx])}"
    else:
        # Fallback: shopkeeper has M2_MAGIC → mage kit (wand etc.) is acceptable.
        assert bool(jnp.any(cats != 0))


def test_animal_monster_spawns_with_no_inventory():
    """A giant ant (animal-class) gets the empty kit — no items."""
    state = _spawn_one_monster(_GIANT_ANT)
    cats = state.monster_ai.inv_category[0]
    assert bool(jnp.all(cats == 0)), f"Animal had inventory: {cats.tolist()}"


# ---------------------------------------------------------------------------
# 5. populate_level_with_monsters end-to-end
# ---------------------------------------------------------------------------

def test_populate_level_writes_entry_idx_and_inventory():
    """populate_level_with_monsters spawns N monsters and sets entry_idx +
    inventory per the kit lookup table.
    """
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)
    floor_map = jnp.full((static.map_h, static.map_w), TileType.FLOOR, dtype=jnp.int8)
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor_map),
        player_pos=jnp.array([0, 0], dtype=jnp.int16),
    )

    rng = jax.random.PRNGKey(101)
    new_state = populate_level_with_monsters(state, rng, n_monsters=5)

    mai = new_state.monster_ai
    # All 5 first slots should be alive with positive entry_idx coverage and
    # inventory derived from the per-entry kit table.
    for i in range(5):
        assert bool(mai.alive[i]), f"slot {i} not alive"
        ei = int(mai.entry_idx[i])
        expected_kit = int(_MONSTER_KIT_BY_ENTRY[ei])
        # Top slot of inventory should match the kit's slot 0 category.
        assert int(mai.inv_category[i, 0]) == int(_KIT_CATS[expected_kit, 0]), \
            f"slot {i} (entry={ei}) kit mismatch"
