"""Wave 3 spawning tests.

Tests:
    - MONSTR_DIFFICULTIES has length NUMMONS
    - Spawn 5 monsters at depth=3 → all type difficulties in expected range
    - After spawn, state.monster_ai.alive.sum() == 5
    - After spawn, no monster on same tile as player
    - eligible_monsters_for_depth excludes G_NOGEN entries
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants.monsters import NUMMONS, G_NOGEN, G_UNIQ
from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.dungeon.spawning import (
    MONSTR_DIFFICULTIES,
    eligible_monsters_for_depth,
    pick_monster_for_level,
    spawn_initial_monsters,
    populate_level_with_monsters,
)

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# MONSTR_DIFFICULTIES table
# ---------------------------------------------------------------------------

def test_monstr_difficulties_length():
    assert MONSTR_DIFFICULTIES.shape == (NUMMONS,), (
        f"Expected ({NUMMONS},), got {MONSTR_DIFFICULTIES.shape}"
    )


def test_monstr_difficulties_values_are_nonnegative():
    assert bool(jnp.all(MONSTR_DIFFICULTIES >= 0))


# ---------------------------------------------------------------------------
# eligible_monsters_for_depth
# ---------------------------------------------------------------------------

def test_eligible_excludes_nogen():
    mask = eligible_monsters_for_depth(depth=3)
    for i, m in enumerate(MONSTERS):
        if (m.generation_mask & G_NOGEN) != 0:
            assert not bool(mask[i]), f"Monster {m.name} (idx={i}) has G_NOGEN but was marked eligible"


def test_eligible_excludes_uniq():
    mask = eligible_monsters_for_depth(depth=3)
    for i, m in enumerate(MONSTERS):
        if (m.generation_mask & G_UNIQ) != 0:
            assert not bool(mask[i]), f"Monster {m.name} (idx={i}) has G_UNIQ but was marked eligible"


def test_eligible_at_depth3_has_entries():
    mask = eligible_monsters_for_depth(depth=3)
    assert bool(jnp.any(mask)), "No eligible monsters at depth=3"


def test_eligible_depth_window():
    """All eligible monsters have difficulty in [depth-6, depth+5]."""
    depth = 3
    mask = eligible_monsters_for_depth(depth)
    diffs = MONSTR_DIFFICULTIES
    eligible_diffs = diffs[mask]
    if eligible_diffs.shape[0] > 0:
        assert bool(jnp.all(eligible_diffs >= depth - 6))
        assert bool(jnp.all(eligible_diffs <= depth + 5))


# ---------------------------------------------------------------------------
# spawn_initial_monsters
# ---------------------------------------------------------------------------

def _make_floor_mask(h: int = 21, w: int = 80) -> jnp.ndarray:
    """Simple all-floor valid mask."""
    return jnp.ones((h, w), dtype=jnp.bool_)


def test_spawn_returns_correct_count():
    rng = jax.random.PRNGKey(1)
    positions, type_ids, hps, max_hps, count = spawn_initial_monsters(
        rng, depth=3, n_monsters=5, valid_tiles_mask=_make_floor_mask(),
        map_h=21, map_w=80,
    )
    assert int(count) == 5


def test_spawn_type_difficulties_in_range():
    """All spawned type difficulties should be in [depth-6, depth+5]."""
    depth = 3
    rng = jax.random.PRNGKey(2)
    _, type_ids, _, _, _ = spawn_initial_monsters(
        rng, depth=depth, n_monsters=5, valid_tiles_mask=_make_floor_mask(),
        map_h=21, map_w=80,
    )
    diffs = MONSTR_DIFFICULTIES[type_ids]
    assert bool(jnp.all(diffs >= depth - 6)), f"Some difficulties < {depth - 6}: {diffs}"
    assert bool(jnp.all(diffs <= depth + 5)), f"Some difficulties > {depth + 5}: {diffs}"


def test_spawn_hps_positive():
    rng = jax.random.PRNGKey(3)
    _, _, hps, max_hps, _ = spawn_initial_monsters(
        rng, depth=3, n_monsters=5, valid_tiles_mask=_make_floor_mask(),
        map_h=21, map_w=80,
    )
    assert bool(jnp.all(hps > 0))
    assert bool(jnp.all(max_hps > 0))


def test_spawn_positions_on_valid_tiles():
    """All spawned positions should be within the valid tile mask."""
    h, w = 21, 80
    rng = jax.random.PRNGKey(4)
    mask = _make_floor_mask(h, w)
    positions, _, _, _, _ = spawn_initial_monsters(
        rng, depth=3, n_monsters=5, valid_tiles_mask=mask,
        map_h=h, map_w=w,
    )
    for i in range(5):
        r, c = int(positions[i, 0]), int(positions[i, 1])
        assert bool(mask[r, c]), f"Monster {i} spawned at non-valid tile ({r},{c})"


# ---------------------------------------------------------------------------
# populate_level_with_monsters
# ---------------------------------------------------------------------------

def _state_with_floor() -> EnvState:
    """EnvState with a fully-floor level 1 (branch 0, level index 0)."""
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)
    # Fill branch=0, level=0 with FLOOR tiles
    floor_map = jnp.full(
        (static.map_h, static.map_w), TileType.FLOOR, dtype=jnp.int8
    )
    state = state.replace(terrain=state.terrain.at[0, 0].set(floor_map))
    # Place player at (5, 5) so we can check no monster lands there
    state = state.replace(player_pos=jnp.array([5, 5], dtype=jnp.int16))
    return state


def test_populate_alive_count():
    state = _state_with_floor()
    new_state = populate_level_with_monsters(state, jax.random.PRNGKey(10), n_monsters=5)
    alive_count = int(jnp.sum(new_state.monster_ai.alive))
    assert alive_count == 5, f"Expected 5 alive, got {alive_count}"


def test_populate_no_monster_on_player():
    state = _state_with_floor()
    new_state = populate_level_with_monsters(state, jax.random.PRNGKey(11), n_monsters=5)
    mai = new_state.monster_ai
    player_pos = new_state.player_pos.astype(jnp.int32)
    alive_mask = mai.alive[:5]
    mon_pos = mai.pos[:5].astype(jnp.int32)
    on_player = (
        (mon_pos[:, 0] == player_pos[0]) &
        (mon_pos[:, 1] == player_pos[1]) &
        alive_mask
    )
    assert not bool(jnp.any(on_player)), "A monster was spawned on the player tile"


def test_populate_hps_positive():
    state = _state_with_floor()
    new_state = populate_level_with_monsters(state, jax.random.PRNGKey(12), n_monsters=5)
    alive_hps = new_state.monster_ai.hp[:5]
    assert bool(jnp.all(alive_hps > 0)), f"Some spawned monsters have hp <= 0: {alive_hps}"


def test_populate_does_not_disturb_other_slots():
    """Slots beyond n_monsters should remain dead (alive=False)."""
    state = _state_with_floor()
    new_state = populate_level_with_monsters(state, jax.random.PRNGKey(13), n_monsters=5)
    tail_alive = new_state.monster_ai.alive[5:]
    assert not bool(jnp.any(tail_alive)), "Slots beyond n_monsters were set to alive"
