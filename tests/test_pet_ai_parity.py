"""Wave 8d pet AI vendor-parity tests.

Audits Nethax/nethax/subsystems/monster_ai.py::pet_move against:
    vendor/nethack/src/dogmove.c::dog_move lines 566-644, 1014

Key invariants tested:
  - Follow mode (Chebyshev dist < 6): pet steps toward player
  - Explore mode (Chebyshev dist >= 6): random walk differs from follow
  - Already adjacent (dist == 0): pet stays adjacent / does not overshoot
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

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Helpers
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


def _place_pet(state: EnvState, slot: int, pos) -> EnvState:
    """Place a tame pet at pos in the given slot."""
    mai = state.monster_ai
    mai = mai.replace(
        pos=mai.pos.at[slot].set(jnp.array(pos, dtype=jnp.int16)),
        hp=mai.hp.at[slot].set(jnp.int32(10)),
        hp_max=mai.hp_max.at[slot].set(jnp.int32(10)),
        alive=mai.alive.at[slot].set(jnp.bool_(True)),
        tame=mai.tame.at[slot].set(jnp.bool_(True)),
        peaceful=mai.peaceful.at[slot].set(jnp.bool_(False)),
        asleep=mai.asleep.at[slot].set(jnp.bool_(False)),
        apport=mai.apport.at[slot].set(jnp.int8(5)),
    )
    return state.replace(monster_ai=mai)


def _chebyshev(a, b) -> int:
    return int(jnp.maximum(abs(a[0] - b[0]), abs(a[1] - b[1])))


# ---------------------------------------------------------------------------
# Test 1: pet within 2 tiles → follows (distance does not increase)
# Vendor: dog_move udist<9 (squared), appr=1 → moves toward player.
# ---------------------------------------------------------------------------

def test_pet_close_follows_player():
    """Pet placed 2 tiles from player closes the gap after one step.

    Vendor dogmove.c line 573: appr=1 when udist>1, moves toward player.
    Chebyshev dist 2 < 6 → follow mode; after one step dist <= 2.
    """
    player_pos = (10, 20)
    pet_start  = (10, 18)  # 2 tiles west, Chebyshev = 2

    state = _floor_state(player_pos)
    state = _place_pet(state, 0, pet_start)

    new_state = pet_move(state, _RNG, jnp.int32(0))
    new_pos = new_state.monster_ai.pos[0]
    dist_after = _chebyshev(new_pos, player_pos)

    # After one follow step the pet must be no farther than 2 away.
    assert dist_after <= 2, (
        f"Pet at {tuple(new_pos)} should have followed player at {player_pos}, "
        f"but Chebyshev dist = {dist_after} > 2"
    )


# ---------------------------------------------------------------------------
# Test 2: pet far away (dist 15) → explore mode differs from follow-only
# Vendor: dog_goal line 629 sets gx=FARAWAY (random wander) when pet is far
# from player and player not in sight — explore mode picks random direction.
# ---------------------------------------------------------------------------

def test_pet_far_away_explores_not_always_follow():
    """Pet 15 tiles away explores (random), not always moving toward player.

    We run 50 steps with different RNG keys and record positions.  A purely
    follow-only policy would monotonically decrease distance every step; the
    explore mode should produce moves that don't always decrease distance.
    Vendor dogmove.c line 629: ``gx = gg.gy = FARAWAY`` → random wander.
    """
    player_pos = (10, 10)
    pet_start  = (10, 25)  # Chebyshev = 15, well beyond the 6-tile threshold

    state = _floor_state(player_pos)
    state = _place_pet(state, 0, pet_start)

    positions = []
    for seed in range(50):
        rng = jax.random.PRNGKey(seed + 100)
        s = pet_move(state, rng, jnp.int32(0))
        positions.append(tuple(int(x) for x in s.monster_ai.pos[0]))

    # In follow-only all positions would be (10, 24) — greedy step toward player.
    # In explore mode we should see some variation (multiple distinct positions).
    unique_positions = set(positions)
    assert len(unique_positions) > 1, (
        f"Expected explore mode to produce varied positions, "
        f"but all 50 steps gave: {unique_positions}"
    )


# ---------------------------------------------------------------------------
# Test 3: pet at distance 0 (same tile as player) → stays adjacent
# Vendor: dog_move line 1022 — udist==0 returns MMOVE_NOTHING.
# In our model: dist<6 triggers follow, but greedy step from dist 0 → no move
# (delta is all zeros), so pet remains on same tile.
# ---------------------------------------------------------------------------

def test_pet_adjacent_stays_adjacent():
    """Pet at distance 0 does not leap away from the player.

    Vendor dogmove.c line 1022: ``if (!udist) return MMOVE_NOTHING;``
    Our greedy-step with delta=0 leaves position unchanged.
    """
    player_pos = (10, 20)
    pet_start  = (10, 20)  # same tile, Chebyshev = 0

    state = _floor_state(player_pos)
    state = _place_pet(state, 0, pet_start)

    new_state = pet_move(state, _RNG, jnp.int32(0))
    new_pos = new_state.monster_ai.pos[0]
    dist_after = _chebyshev(new_pos, player_pos)

    # Pet should remain on the same tile or move at most 1 tile away.
    assert dist_after <= 1, (
        f"Pet at {tuple(new_pos)} should stay adjacent to player at {player_pos}, "
        f"but Chebyshev dist = {dist_after}"
    )


# ---------------------------------------------------------------------------
# Test 4: non-pet (tame=False) is unaffected by pet_move
# ---------------------------------------------------------------------------

def test_non_pet_unaffected():
    """pet_move must not move a hostile (non-tame) monster."""
    player_pos = (10, 20)
    mon_start  = (10, 18)

    state = _floor_state(player_pos)
    mai = state.monster_ai
    mai = mai.replace(
        pos=mai.pos.at[0].set(jnp.array(mon_start, dtype=jnp.int16)),
        hp=mai.hp.at[0].set(jnp.int32(10)),
        alive=mai.alive.at[0].set(jnp.bool_(True)),
        tame=mai.tame.at[0].set(jnp.bool_(False)),  # not tame
    )
    state = state.replace(monster_ai=mai)

    new_state = pet_move(state, _RNG, jnp.int32(0))
    new_pos = new_state.monster_ai.pos[0]

    assert tuple(int(x) for x in new_pos) == mon_start, (
        f"Non-pet at {mon_start} was moved to {tuple(new_pos)} by pet_move"
    )
