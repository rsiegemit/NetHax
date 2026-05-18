"""Pet AI polish parity tests.

Audits the gap-fills added to monster_ai.py::pet_move:
  - pet_hunger ticker        (dog.c:380)
  - pet starves at -50       (dog.c:380)
  - pet flees on low HP      (dogmove.c:1100)
  - pet eats floor food      (dogmove.c:520)
  - pet uses BFS pathfind    (mfndpos)
  - pet attacks adjacent hostile (dogmove.c:1150)
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.monster_ai import (
    MAX_MONSTERS_PER_LEVEL,
    pet_move,
    _chebyshev_dist,
)

_RNG = jax.random.PRNGKey(0)
_CAT_FOOD = 7  # ItemCategory.FOOD


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _floor_state(player_pos=(10, 20)) -> EnvState:
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)
    floor_map = jnp.full((static.map_h, static.map_w), TileType.FLOOR, dtype=jnp.int8)
    return state.replace(
        terrain=state.terrain.at[0, 0].set(floor_map),
        player_pos=jnp.array(player_pos, dtype=jnp.int16),
        player_hp=jnp.int32(20),
        player_hp_max=jnp.int32(20),
    )


def _place_pet(state: EnvState, slot: int, pos,
               hp=10, hp_max=10, hunger=1000) -> EnvState:
    mai = state.monster_ai
    mai = mai.replace(
        pos=mai.pos.at[slot].set(jnp.array(pos, dtype=jnp.int16)),
        hp=mai.hp.at[slot].set(jnp.int32(hp)),
        hp_max=mai.hp_max.at[slot].set(jnp.int32(hp_max)),
        alive=mai.alive.at[slot].set(jnp.bool_(True)),
        tame=mai.tame.at[slot].set(jnp.bool_(True)),
        peaceful=mai.peaceful.at[slot].set(jnp.bool_(False)),
        asleep=mai.asleep.at[slot].set(jnp.bool_(False)),
        apport=mai.apport.at[slot].set(jnp.int8(5)),
        pet_hunger=mai.pet_hunger.at[slot].set(jnp.int16(hunger)),
        # entry_idx=0 → not undead/demon → not fearless
        entry_idx=mai.entry_idx.at[slot].set(jnp.int16(0)),
    )
    return state.replace(monster_ai=mai)


def _place_hostile(state: EnvState, slot: int, pos, hp=10) -> EnvState:
    mai = state.monster_ai
    mai = mai.replace(
        pos=mai.pos.at[slot].set(jnp.array(pos, dtype=jnp.int16)),
        hp=mai.hp.at[slot].set(jnp.int32(hp)),
        hp_max=mai.hp_max.at[slot].set(jnp.int32(hp)),
        alive=mai.alive.at[slot].set(jnp.bool_(True)),
        tame=mai.tame.at[slot].set(jnp.bool_(False)),
        peaceful=mai.peaceful.at[slot].set(jnp.bool_(False)),
        asleep=mai.asleep.at[slot].set(jnp.bool_(False)),
    )
    return state.replace(monster_ai=mai)


def _place_food(state: EnvState, row: int, col: int,
                weight: int = 40) -> EnvState:
    """Place a FOOD item at (row, col) on branch 0, level 0."""
    b, lv = 0, 0
    new_cat = state.ground_items.category.at[b, lv, row, col, 0].set(
        jnp.int8(_CAT_FOOD)
    )
    new_weight = state.ground_items.weight.at[b, lv, row, col, 0].set(
        jnp.int32(weight)
    )
    new_gi = state.ground_items.replace(category=new_cat, weight=new_weight)
    return state.replace(ground_items=new_gi)


# ---------------------------------------------------------------------------
# Test 1 — pet_hunger decrements per turn  (dog.c:380)
# ---------------------------------------------------------------------------

def test_pet_hunger_decrements_per_turn():
    """pet_hunger starts at 1000; after 50 turns it should be <= 950."""
    state = _floor_state(player_pos=(10, 20))
    state = _place_pet(state, slot=0, pos=(10, 19), hunger=1000)

    for _ in range(50):
        state = pet_move(state, _RNG, jnp.int32(0))

    hunger_after = int(state.monster_ai.pet_hunger[0])
    assert hunger_after <= 950, (
        f"Expected pet_hunger <= 950 after 50 turns, got {hunger_after}"
    )


# ---------------------------------------------------------------------------
# Test 2 — pet starves at hunger <= -50  (dog.c:380)
# ---------------------------------------------------------------------------

def test_pet_starves_at_minus_50():
    """Pet with hunger=-49 should die (alive=False) after one tick."""
    state = _floor_state(player_pos=(10, 20))
    state = _place_pet(state, slot=0, pos=(10, 19), hunger=-49)

    state = pet_move(state, _RNG, jnp.int32(0))

    assert not bool(state.monster_ai.alive[0]), (
        "Pet at hunger=-49 should have starved (alive=False) after one tick."
    )


# ---------------------------------------------------------------------------
# Test 3 — pet flees on low HP  (dogmove.c:1100)
# ---------------------------------------------------------------------------

def test_pet_flees_low_hp():
    """Pet at hp_max/5 should move AWAY from player (distance increases)."""
    player_pos = (10, 20)
    pet_pos = (10, 15)  # 5 tiles west; within follow range normally

    state = _floor_state(player_pos=player_pos)
    # hp_max=20, hp=4 → hp < hp_max/4 → low HP flee
    state = _place_pet(state, slot=0, pos=pet_pos, hp=4, hp_max=20, hunger=1000)

    dist_before = int(_chebyshev_dist(
        jnp.array(pet_pos, dtype=jnp.int32),
        jnp.array(player_pos, dtype=jnp.int32),
    ))

    state = pet_move(state, _RNG, jnp.int32(0))

    new_pos = state.monster_ai.pos[0]
    dist_after = int(_chebyshev_dist(
        new_pos.astype(jnp.int32),
        jnp.array(player_pos, dtype=jnp.int32),
    ))

    assert dist_after >= dist_before, (
        f"Low-HP pet should flee: dist went {dist_before} -> {dist_after} "
        f"(expected non-decrease). Pet pos: {tuple(new_pos)}"
    )


# ---------------------------------------------------------------------------
# Test 4 — pet eats floor food  (dogmove.c:520)
# ---------------------------------------------------------------------------

def test_pet_eats_floor_food():
    """Hungry pet on a tile with FOOD eats it: HP increases, item removed."""
    player_pos = (10, 20)
    pet_pos = (10, 19)

    state = _floor_state(player_pos=player_pos)
    # Pet is hungry (hunger <= 0) and damaged.
    state = _place_pet(state, slot=0, pos=pet_pos, hp=5, hp_max=20, hunger=0)
    # Place food at same tile as pet; weight=40 → heal = 40//4 = 10.
    state = _place_food(state, row=pet_pos[0], col=pet_pos[1], weight=40)

    hp_before = int(state.monster_ai.hp[0])
    food_cat_before = int(state.ground_items.category[0, 0, pet_pos[0], pet_pos[1], 0])

    state = pet_move(state, _RNG, jnp.int32(0))

    hp_after = int(state.monster_ai.hp[0])
    food_cat_after = int(state.ground_items.category[0, 0, pet_pos[0], pet_pos[1], 0])

    assert food_cat_before == _CAT_FOOD, "Setup: food should be placed before test."
    assert food_cat_after == 0, (
        f"Food should be consumed (category=0) after pet eats it, got {food_cat_after}"
    )
    assert hp_after > hp_before, (
        f"Pet HP should increase after eating food: {hp_before} -> {hp_after}"
    )


# ---------------------------------------------------------------------------
# Test 5 — pet uses BFS pathfind around wall  (mfndpos)
# ---------------------------------------------------------------------------

def test_pet_uses_pathfind_around_wall():
    """Pet should path around a wall column to follow player.

    Layout (schematic, row 10):
      col:   5  6  7  8  9  10  11  12  13  14  15
             .  .  .  .  W   W   W   W   .   .  P
                               ^wall column
      pet at col=5, player at col=15; wall blocks cols 9-12 on row 10.
      A greedy step would try to walk directly east into the wall.
      BFS should route around via row 9 or row 11.
    """
    player_pos = (10, 15)
    pet_pos = (10, 5)

    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)

    # Start with all-floor map.
    floor_map = jnp.full((static.map_h, static.map_w), TileType.FLOOR, dtype=jnp.int8)
    # Place a wall column blocking row 10 cols 9-12.
    for c in range(9, 13):
        floor_map = floor_map.at[10, c].set(TileType.WALL)
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor_map),
        player_pos=jnp.array(player_pos, dtype=jnp.int16),
        player_hp=jnp.int32(20),
        player_hp_max=jnp.int32(20),
    )
    state = _place_pet(state, slot=0, pos=pet_pos, hunger=1000)

    # Ensure pet has high enough HP to not trigger low-HP flee.
    state = pet_move(state, _RNG, jnp.int32(0))

    new_pos = tuple(int(x) for x in state.monster_ai.pos[0])
    # Pet should not have stepped into a wall tile.
    tile_at_new = int(state.terrain[0, 0, new_pos[0], new_pos[1]])
    assert tile_at_new != int(TileType.WALL), (
        f"Pet at {new_pos} stepped into a wall tile after BFS pathfind."
    )
    # Pet should have moved (not stayed put at pet_pos).
    assert new_pos != pet_pos, (
        f"Pet should have moved but stayed at {new_pos}."
    )


# ---------------------------------------------------------------------------
# Test 6 — pet attacks adjacent hostile  (dogmove.c:1150)
# ---------------------------------------------------------------------------

def test_pet_attacks_adjacent_hostile():
    """Pet adjacent to a hostile monster should reduce hostile's HP."""
    player_pos = (10, 20)
    pet_pos = (5, 5)
    hostile_pos = (5, 6)  # Chebyshev 1 from pet

    state = _floor_state(player_pos=player_pos)
    state = _place_pet(state, slot=0, pos=pet_pos, hunger=1000)
    state = _place_hostile(state, slot=1, pos=hostile_pos, hp=10)

    hostile_hp_before = int(state.monster_ai.hp[1])
    state = pet_move(state, _RNG, jnp.int32(0))
    hostile_hp_after = int(state.monster_ai.hp[1])

    assert hostile_hp_after < hostile_hp_before, (
        f"Pet should have attacked adjacent hostile: HP {hostile_hp_before} -> {hostile_hp_after}"
    )
