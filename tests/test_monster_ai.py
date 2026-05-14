"""Wave 3 monster AI tests.

Tests:
    - Sleeping monster does not move after monster_turn
    - Peaceful monster does not move after monster_turn
    - Dead monster (alive=False) is skipped
    - Monster adjacent to player → attack fires (player_hp may change)
    - Non-adjacent monster moves one step closer to player
    - wake_monsters_near wakes monsters within radius, leaves far ones asleep
    - monsters_step_all runs without error over all slots
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
    MoveStrategy,
    MAX_MONSTERS_PER_LEVEL,
    monster_turn,
    monsters_step_all,
    wake_monsters_near,
)

_RNG = jax.random.PRNGKey(7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_with_floor_and_monster(
    monster_pos=(10, 10),
    player_pos=(10, 15),
    hp: int = 20,
    asleep: bool = False,
    peaceful: bool = False,
    alive: bool = True,
) -> EnvState:
    """Build an EnvState with one live monster at slot 0."""
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)

    # All-floor terrain so movement is unrestricted
    floor_map = jnp.full((static.map_h, static.map_w), TileType.FLOOR, dtype=jnp.int8)
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor_map),
        player_pos=jnp.array(player_pos, dtype=jnp.int16),
        player_hp=jnp.int32(30),
        player_hp_max=jnp.int32(30),
    )

    mai = state.monster_ai
    mai = mai.replace(
        pos=mai.pos.at[0].set(jnp.array(monster_pos, dtype=jnp.int16)),
        hp=mai.hp.at[0].set(jnp.int32(hp)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(hp)),
        alive=mai.alive.at[0].set(jnp.bool_(alive)),
        asleep=mai.asleep.at[0].set(jnp.bool_(asleep)),
        peaceful=mai.peaceful.at[0].set(jnp.bool_(peaceful)),
        ac=mai.ac.at[0].set(jnp.int8(10)),
        attack_dice_n=mai.attack_dice_n.at[0].set(jnp.int8(1)),
        attack_dice_sides=mai.attack_dice_sides.at[0].set(jnp.int8(4)),
    )
    return state.replace(monster_ai=mai)


# ---------------------------------------------------------------------------
# Sleeping monster
# ---------------------------------------------------------------------------

def test_sleeping_monster_does_not_move():
    state = _state_with_floor_and_monster(
        monster_pos=(10, 10), player_pos=(10, 15), asleep=True
    )
    rng = jax.random.PRNGKey(1)
    new_state = monster_turn(state, rng, jnp.int32(0))

    orig_pos = state.monster_ai.pos[0]
    new_pos  = new_state.monster_ai.pos[0]
    assert bool(jnp.all(orig_pos == new_pos)), (
        f"Sleeping monster moved: {orig_pos} → {new_pos}"
    )


def test_sleeping_monster_does_not_attack():
    state = _state_with_floor_and_monster(
        monster_pos=(10, 10), player_pos=(10, 11), asleep=True
    )
    orig_hp = int(state.player_hp)
    rng = jax.random.PRNGKey(2)
    new_state = monster_turn(state, rng, jnp.int32(0))
    assert int(new_state.player_hp) == orig_hp, "Sleeping monster attacked player"


# ---------------------------------------------------------------------------
# Peaceful monster
# ---------------------------------------------------------------------------

def test_peaceful_monster_does_not_move():
    state = _state_with_floor_and_monster(
        monster_pos=(10, 10), player_pos=(10, 15), peaceful=True
    )
    rng = jax.random.PRNGKey(3)
    new_state = monster_turn(state, rng, jnp.int32(0))

    orig_pos = state.monster_ai.pos[0]
    new_pos  = new_state.monster_ai.pos[0]
    assert bool(jnp.all(orig_pos == new_pos)), "Peaceful monster moved"


# ---------------------------------------------------------------------------
# Dead monster
# ---------------------------------------------------------------------------

def test_dead_monster_is_skipped():
    state = _state_with_floor_and_monster(alive=False)
    rng = jax.random.PRNGKey(4)
    new_state = monster_turn(state, rng, jnp.int32(0))

    # State should be completely unchanged (alive stays False, hp stays 0)
    assert not bool(new_state.monster_ai.alive[0])


# ---------------------------------------------------------------------------
# Adjacent monster attacks
# ---------------------------------------------------------------------------

def test_adjacent_monster_can_affect_player_hp():
    """Monster at Chebyshev dist=1 from player runs melee; player_hp may drop."""
    # Place monster directly adjacent (distance = 1)
    state = _state_with_floor_and_monster(
        monster_pos=(10, 10), player_pos=(10, 11),
        hp=20, asleep=False, peaceful=False,
    )
    # Run many trials to ensure at least one hit (monster to-hit should be decent)
    any_damage = False
    for seed in range(20):
        rng = jax.random.PRNGKey(seed + 100)
        new_state = monster_turn(state, rng, jnp.int32(0))
        if int(new_state.player_hp) < int(state.player_hp):
            any_damage = True
            break

    assert any_damage, (
        "Adjacent monster never dealt damage across 20 seeds "
        "(expected at least one hit)"
    )


def test_adjacent_monster_strategy_becomes_hunt():
    state = _state_with_floor_and_monster(
        monster_pos=(10, 10), player_pos=(10, 11),
    )
    rng = jax.random.PRNGKey(200)
    new_state = monster_turn(state, rng, jnp.int32(0))
    strategy = int(new_state.monster_ai.mstrategy[0])
    assert strategy == MoveStrategy.HUNT, f"Expected HUNT, got {strategy}"


# ---------------------------------------------------------------------------
# Non-adjacent monster moves toward player
# ---------------------------------------------------------------------------

def test_monster_moves_toward_player():
    """Monster at (10, 10), player at (10, 15): after one turn, col should increase."""
    state = _state_with_floor_and_monster(
        monster_pos=(10, 10), player_pos=(10, 15),
    )
    rng = jax.random.PRNGKey(5)
    new_state = monster_turn(state, rng, jnp.int32(0))

    orig_col = int(state.monster_ai.pos[0, 1])
    new_col  = int(new_state.monster_ai.pos[0, 1])
    assert new_col > orig_col, (
        f"Monster should have moved right (toward player at col=15), "
        f"but col went {orig_col} → {new_col}"
    )


def test_monster_moves_one_step():
    """Greedy step should move exactly 1 tile (Chebyshev) per turn."""
    state = _state_with_floor_and_monster(
        monster_pos=(10, 10), player_pos=(10, 15),
    )
    rng = jax.random.PRNGKey(6)
    new_state = monster_turn(state, rng, jnp.int32(0))

    orig = state.monster_ai.pos[0].astype(jnp.int32)
    new  = new_state.monster_ai.pos[0].astype(jnp.int32)
    delta = jnp.abs(orig - new)
    cheb = int(jnp.maximum(delta[0], delta[1]))
    assert cheb == 1, f"Expected Chebyshev step of 1, got {cheb}"


def test_monster_strategy_is_hunt_after_move():
    state = _state_with_floor_and_monster(
        monster_pos=(5, 5), player_pos=(5, 15),
    )
    rng = jax.random.PRNGKey(7)
    new_state = monster_turn(state, rng, jnp.int32(0))
    strategy = int(new_state.monster_ai.mstrategy[0])
    assert strategy == MoveStrategy.HUNT


# ---------------------------------------------------------------------------
# wake_monsters_near
# ---------------------------------------------------------------------------

def _state_with_sleeping_monsters() -> EnvState:
    """EnvState with 3 sleeping monsters at various distances from (10, 10)."""
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)
    mai = state.monster_ai

    # Monster 0: (10, 10) — distance 0 from disturbance
    # Monster 1: (10, 13) — distance 3 (exactly at radius boundary)
    # Monster 2: (10, 14) — distance 4 (just outside default radius=3)
    positions = [
        (10, 10),
        (10, 13),
        (10, 14),
    ]
    for i, (r, c) in enumerate(positions):
        mai = mai.replace(
            pos=mai.pos.at[i].set(jnp.array([r, c], dtype=jnp.int16)),
            alive=mai.alive.at[i].set(jnp.bool_(True)),
            asleep=mai.asleep.at[i].set(jnp.bool_(True)),
            hp=mai.hp.at[i].set(jnp.int32(10)),
            hp_max=mai.hp_max.at[i].set(jnp.int32(10)),
        )
    return state.replace(monster_ai=mai)


def test_wake_monsters_wakes_within_radius():
    state = _state_with_sleeping_monsters()
    disturbance = jnp.array([10, 10], dtype=jnp.int32)
    new_state = wake_monsters_near(state, disturbance, radius=3)

    # Monsters 0 and 1 should be awake
    assert not bool(new_state.monster_ai.asleep[0]), "Monster 0 (dist=0) should be awake"
    assert not bool(new_state.monster_ai.asleep[1]), "Monster 1 (dist=3) should be awake"


def test_wake_monsters_leaves_far_ones_asleep():
    state = _state_with_sleeping_monsters()
    disturbance = jnp.array([10, 10], dtype=jnp.int32)
    new_state = wake_monsters_near(state, disturbance, radius=3)

    # Monster 2 (dist=4) should still be asleep
    assert bool(new_state.monster_ai.asleep[2]), "Monster 2 (dist=4) should still be asleep"


# ---------------------------------------------------------------------------
# monsters_step_all
# ---------------------------------------------------------------------------

def test_monsters_step_all_runs():
    """Smoke test: monsters_step_all should complete without error."""
    state = _state_with_floor_and_monster(
        monster_pos=(10, 10), player_pos=(10, 15),
    )
    rng = jax.random.PRNGKey(99)
    new_state = monsters_step_all(state, rng)
    # Monster 0 should have moved (was alive and non-sleeping)
    orig_pos = state.monster_ai.pos[0]
    new_pos  = new_state.monster_ai.pos[0]
    assert not bool(jnp.all(orig_pos == new_pos)), (
        "monsters_step_all: live monster didn't move"
    )


def test_monsters_step_all_skips_dead_slots():
    """All slots initialized as dead; none should change position."""
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)
    rng = jax.random.PRNGKey(100)
    new_state = monsters_step_all(state, rng)
    # All positions should still be -1 (default)
    assert bool(jnp.all(new_state.monster_ai.pos == -1)), (
        "Dead slots changed position during monsters_step_all"
    )
