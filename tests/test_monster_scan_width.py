"""Wave 5 Phase 0c: verify MAX_MONSTERS_PER_LEVEL expanded from 200 to 400.

Tests:
    - Constant value is 400.
    - Fresh MonsterAIState arrays have leading dim 400.
    - Spawning 300 monsters works (would have hit cap at 200).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.monster_ai import (
    MAX_MONSTERS_PER_LEVEL,
    make_monster_ai_state,
)
from Nethax.nethax.dungeon.spawning import populate_level_with_monsters


def test_max_monsters_is_400():
    assert MAX_MONSTERS_PER_LEVEL == 400


def test_monster_ai_state_pos_shape_400():
    mai = make_monster_ai_state()
    assert mai.pos.shape == (400, 2)


def test_monster_ai_state_alive_shape_400():
    mai = make_monster_ai_state()
    assert mai.alive.shape == (400,)


def test_can_spawn_300_monsters():
    """Spawn 300 monsters — would have been clipped at the old 200-slot cap."""
    static = StaticParams()
    state = EnvState.default(rng=jax.random.PRNGKey(0), static=static)
    floor_map = jnp.full(
        (static.map_h, static.map_w), TileType.FLOOR, dtype=jnp.int8
    )
    state = state.replace(terrain=state.terrain.at[0, 0].set(floor_map))
    state = state.replace(player_pos=jnp.array([5, 5], dtype=jnp.int16))

    new_state = populate_level_with_monsters(
        state, jax.random.PRNGKey(123), n_monsters=300
    )
    alive_count = int(jnp.sum(new_state.monster_ai.alive))
    assert alive_count == 300, f"Expected 300 alive, got {alive_count}"
