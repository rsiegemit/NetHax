"""Parity tests for boulder pushing, Sokoban pit detection, piercer drop,
shrieker summon, and container trap.

Canonical sources:
  vendor/nethack/src/hack.c::moverock   (boulder push)
  vendor/nethack/src/sokoban.c          (Sokoban prize)
  vendor/nethack/src/mon.c::trapped     (piercer drop)
  vendor/nethack/src/mon.c::shrieker    (shrieker alarm)
  vendor/nethack/src/pickup.c::container_trap (container trap)
"""
import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.boulders import (
    BOULDER_CATEGORY, BOULDER_TYPE_ID,
    SOKOBAN_BRANCH_IDX, SOKOBAN_PITS_TO_FILL,
    try_push_boulder,
)
from Nethax.nethax.subsystems.traps import TrapType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state():
    rng = jax.random.PRNGKey(0)
    return EnvState.default(rng=rng)


def _place_boulder(state, row, col):
    """Write a boulder into ground_items at the player's current branch/level."""
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    gi = state.ground_items
    gi2 = gi.replace(
        category   = gi.category.at[b, lv, row, col, 0].set(jnp.int8(BOULDER_CATEGORY)),
        type_id    = gi.type_id.at[b, lv, row, col, 0].set(jnp.int16(BOULDER_TYPE_ID)),
        quantity   = gi.quantity.at[b, lv, row, col, 0].set(jnp.int16(1)),
        weight     = gi.weight.at[b, lv, row, col, 0].set(jnp.int32(1000)),
        identified = gi.identified.at[b, lv, row, col, 0].set(jnp.bool_(True)),
    )
    return state.replace(ground_items=gi2)


def _set_terrain(state, row, col, tile):
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain.at[b, lv, row, col].set(jnp.int8(int(tile)))
    return state.replace(terrain=new_terrain)


def _set_player(state, row, col):
    return state.replace(player_pos=jnp.array([row, col], dtype=jnp.int16))


def _set_pit_trap(state, row, col):
    """Write a PIT trap into the traps layer at the current level."""
    b   = int(state.dungeon.current_branch)
    lv  = int(state.dungeon.current_level) - 1
    max_lv = state.terrain.shape[1]
    flat   = b * max_lv + lv
    new_tt = state.traps.trap_type.at[flat, row, col].set(jnp.int8(int(TrapType.PIT)))
    return state.replace(traps=state.traps.replace(trap_type=new_tt))


# ---------------------------------------------------------------------------
# 1. Boulder pushed into open tile succeeds
# ---------------------------------------------------------------------------

def test_push_boulder_into_open_succeeds():
    """Pushing a boulder with an open tile beyond moves it one step.

    Cite: hack.c::moverock — beyond tile is FLOOR → push succeeds.
    """
    state = _make_state()
    # Layout (rows 5..7, col 5): player@(5,5), boulder@(6,5), floor@(7,5)
    state = _set_player(state, 5, 5)
    state = _set_terrain(state, 6, 5, TileType.FLOOR)
    state = _set_terrain(state, 7, 5, TileType.FLOOR)
    state = _place_boulder(state, 6, 5)

    from_pos = jnp.array([5, 5], dtype=jnp.int32)
    to_pos   = jnp.array([6, 5], dtype=jnp.int32)

    new_state, pushed = try_push_boulder(state, from_pos, to_pos, 1, 0)

    assert bool(pushed), "push should succeed into open floor tile"

    b  = int(new_state.dungeon.current_branch)
    lv = int(new_state.dungeon.current_level) - 1

    # Boulder must now be at (7,5).
    beyond_cat = int(new_state.ground_items.category[b, lv, 7, 5, 0])
    assert beyond_cat == BOULDER_CATEGORY, "boulder should be at beyond tile (7,5)"

    # Original tile (6,5) must be clear.
    orig_cat = int(new_state.ground_items.category[b, lv, 6, 5, 0])
    assert orig_cat == 0, "boulder's original tile should be empty after push"


# ---------------------------------------------------------------------------
# 2. Boulder blocked by wall
# ---------------------------------------------------------------------------

def test_push_boulder_into_wall_blocks():
    """Push fails when beyond tile is a wall.

    Cite: hack.c::moverock — WALL in beyond → push blocked, player stays.
    """
    state = _make_state()
    state = _set_player(state, 5, 5)
    state = _set_terrain(state, 6, 5, TileType.FLOOR)
    state = _set_terrain(state, 7, 5, TileType.WALL)
    state = _place_boulder(state, 6, 5)

    from_pos = jnp.array([5, 5], dtype=jnp.int32)
    to_pos   = jnp.array([6, 5], dtype=jnp.int32)

    new_state, pushed = try_push_boulder(state, from_pos, to_pos, 1, 0)

    assert not bool(pushed), "push into wall should fail"

    b  = int(new_state.dungeon.current_branch)
    lv = int(new_state.dungeon.current_level) - 1

    # Boulder must remain at (6,5).
    orig_cat = int(new_state.ground_items.category[b, lv, 6, 5, 0])
    assert orig_cat == BOULDER_CATEGORY, "boulder should stay when push fails"


# ---------------------------------------------------------------------------
# 3. Boulder pushed into pit fills it (Sokoban)
# ---------------------------------------------------------------------------

def test_push_into_pit_fills_it_sokoban():
    """Boulder pushed into a pit fills it; sokoban_boulders_pitted increments.

    Cite: sokoban.c::sokoban_in_play — boulder fills pit, trap disarmed.
    Tested in Sokoban branch (branch index 2).
    """
    state = _make_state()
    # Switch to Sokoban branch.
    new_dungeon = state.dungeon.replace(
        current_branch=jnp.int8(SOKOBAN_BRANCH_IDX),
        current_level=jnp.int8(1),
    )
    state = state.replace(dungeon=new_dungeon)

    state = _set_player(state, 5, 5)
    state = _set_terrain(state, 6, 5, TileType.FLOOR)
    state = _set_terrain(state, 7, 5, TileType.TRAP)
    state = _place_boulder(state, 6, 5)
    state = _set_pit_trap(state, 7, 5)

    from_pos = jnp.array([5, 5], dtype=jnp.int32)
    to_pos   = jnp.array([6, 5], dtype=jnp.int32)

    new_state, pushed = try_push_boulder(state, from_pos, to_pos, 1, 0)

    assert bool(pushed), "push into pit should succeed"

    # sokoban_boulders_pitted must have incremented.
    assert int(new_state.sokoban_boulders_pitted) == 1

    # Boulder should NOT be placed at pit tile (it filled the pit).
    b  = int(new_state.dungeon.current_branch)
    lv = int(new_state.dungeon.current_level) - 1
    beyond_cat = int(new_state.ground_items.category[b, lv, 7, 5, 0])
    assert beyond_cat != BOULDER_CATEGORY, "boulder absorbed into pit should not remain as item"

    # Trap at (7,5) should be disarmed (trap_type == 0).
    max_lv = new_state.terrain.shape[1]
    flat   = SOKOBAN_BRANCH_IDX * max_lv + 0
    trap_t = int(new_state.traps.trap_type[flat, 7, 5])
    assert trap_t == 0, "pit trap should be disarmed after boulder fill"


# ---------------------------------------------------------------------------
# 4. Piercer spawns in corridor
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="maybe_spawn_piercer / _PIERCER_ENTRY_INDICES not implemented in "
    "monster_ai yet — piercer ceiling-drop is vendor mon.c::trapped, "
    "deferred to a future wave.",
    strict=True,
)
def test_piercer_drops_in_corridor():
    """maybe_spawn_piercer eventually spawns a piercer when player is in corridor.

    Cite: mon.c::trapped — piercers drop from ceiling in corridors.
    We run 200 iterations with a fixed RNG seed; expect at least one spawn.
    """
    from Nethax.nethax.subsystems.monster_ai import maybe_spawn_piercer, _PIERCER_ENTRY_INDICES

    state = _make_state()
    state = _set_player(state, 5, 5)
    # Set player's tile to CORRIDOR.
    state = _set_terrain(state, 5, 5, TileType.CORRIDOR)

    spawned = False
    rng = jax.random.PRNGKey(7)
    for i in range(200):
        rng, sub = jax.random.split(rng)
        new_state = maybe_spawn_piercer(state, sub)
        alive_count_new = int(jnp.sum(new_state.monster_ai.alive))
        if alive_count_new > 0:
            spawned = True
            # Verify entry_idx is one of the piercer types.
            alive_idx = int(jnp.argmax(new_state.monster_ai.alive))
            entry = int(new_state.monster_ai.entry_idx[alive_idx])
            assert entry in _PIERCER_ENTRY_INDICES, (
                f"spawned monster entry {entry} is not a piercer"
            )
            break

    assert spawned, "expected at least one piercer spawn in 200 corridor steps"


# ---------------------------------------------------------------------------
# 5. Shrieker summons monsters when adjacent
# ---------------------------------------------------------------------------

def test_shrieker_summons_monsters():
    """shrieker_summon activates a dead slot when a shrieker is adjacent.

    Cite: mon.c::shrieker — adjacent shrieker triggers summon with prob 0.25.
    We run 100 iterations and expect at least one summon.
    """
    from Nethax.nethax.subsystems.monster_ai import (
        shrieker_summon, _MS_SHRIEK_AI, MAX_MONSTERS_PER_LEVEL,
    )
    from Nethax.nethax.constants.monsters import MONSTERS

    # Find shrieker entry index.
    shrieker_idx = next(
        i for i, m in enumerate(MONSTERS) if m.name == "shrieker"
    )

    state = _make_state()
    state = _set_player(state, 5, 5)

    # Place a live shrieker adjacent to player at (5,6).
    mai = state.monster_ai
    mai = mai.replace(
        alive     = mai.alive.at[0].set(jnp.bool_(True)),
        pos       = mai.pos.at[0].set(jnp.array([5, 6], dtype=jnp.int16)),
        hp        = mai.hp.at[0].set(jnp.int32(8)),
        hp_max    = mai.hp_max.at[0].set(jnp.int32(8)),
        peaceful  = mai.peaceful.at[0].set(jnp.bool_(False)),
        entry_idx = mai.entry_idx.at[0].set(jnp.int16(shrieker_idx)),
    )
    state = state.replace(monster_ai=mai)

    initial_alive = int(jnp.sum(state.monster_ai.alive))

    summoned = False
    rng = jax.random.PRNGKey(13)
    for i in range(100):
        rng, sub = jax.random.split(rng)
        new_state = shrieker_summon(state, sub)
        new_alive = int(jnp.sum(new_state.monster_ai.alive))
        if new_alive > initial_alive:
            summoned = True
            break

    assert summoned, "expected shrieker to summon at least one monster in 100 steps"
