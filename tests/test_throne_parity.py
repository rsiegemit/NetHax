"""Parity tests for throne sit effects.

Canonical source: vendor/nethack/src/sit.c::throne_sit_effect (lines 38-233).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest


_RNG = jax.random.PRNGKey(42)


def _make_state_on_throne():
    """Return a default EnvState with a THRONE tile at player position (0,0)."""
    from Nethax.nethax.state import EnvState
    from Nethax.nethax.constants.tiles import TileType
    state = EnvState.default(_RNG)
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain.at[b, lv, 0, 0].set(
        jnp.int8(int(TileType.THRONE))
    )
    return state.replace(
        terrain=new_terrain,
        player_pos=jnp.array([0, 0], dtype=jnp.int16),
    )


def _force_outcome(state, outcome_idx: int):
    """Run sit_throne with a seed that deterministically hits outcome_idx.

    Scans PRNGKey seeds until rn2(14) on the rng_outcome split matches.
    sit_throne does: rng, rng_outcome, rng_effect, rng_remove = split(rng, 4)
    """
    from Nethax.nethax.subsystems.throne import sit_throne, _N_OUTCOMES
    from Nethax.nethax.rng import rn2
    for seed in range(500):
        key = jax.random.PRNGKey(seed)
        splits = jax.random.split(key, 4)
        rng_outcome = splits[1]
        roll = int(rn2(rng_outcome, _N_OUTCOMES))
        if roll == outcome_idx:
            return sit_throne(state, key)
    pytest.skip(f"Could not find seed for outcome {outcome_idx} in 500 tries")


# ---------------------------------------------------------------------------
# test_sit_throne_gold_outcome
# Outcome 0 = _gain_gold: player_gold must increase.
# Cite: sit.c case 5 / case 6 gold-gain path.
# ---------------------------------------------------------------------------

def test_sit_throne_gold_outcome():
    state = _make_state_on_throne()
    initial_gold = int(state.player_gold)
    new_state = _force_outcome(state, outcome_idx=0)
    assert int(new_state.player_gold) > initial_gold, (
        "Gold outcome (idx 0) should increase player_gold; "
        "cite: sit.c line 103"
    )


# ---------------------------------------------------------------------------
# test_sit_throne_lightning_damage
# Outcome 12 = _electrocute: player_hp must decrease.
# Cite: sit.c line 77 — losehp(Shock_resistance ? rnd(6) : rnd(30), ...).
# ---------------------------------------------------------------------------

def test_sit_throne_lightning_damage():
    state = _make_state_on_throne()
    state = state.replace(player_hp=jnp.int32(200), player_hp_max=jnp.int32(200))
    initial_hp = int(state.player_hp)
    new_state = _force_outcome(state, outcome_idx=12)
    assert int(new_state.player_hp) < initial_hp, (
        "Electrocute outcome (idx 12) should reduce player_hp; "
        "cite: sit.c line 77"
    )


# ---------------------------------------------------------------------------
# test_sit_throne_eventually_disappears
# Sit 30 times; the throne tile should become FLOOR at least once.
# Cite: sit.c lines 224-226 — if (!rn2(3)) levl[tx][ty].typ = ROOM.
# ---------------------------------------------------------------------------

def test_sit_throne_eventually_disappears():
    from Nethax.nethax.subsystems.throne import sit_throne
    from Nethax.nethax.constants.tiles import TileType

    floors_seen = 0
    for seed in range(30):
        state = _make_state_on_throne()
        b  = int(state.dungeon.current_branch)
        lv = int(state.dungeon.current_level) - 1
        key = jax.random.PRNGKey(seed + 1000)
        new_state = sit_throne(state, key)
        tile = int(new_state.terrain[b, lv, 0, 0])
        if tile == int(TileType.FLOOR):
            floors_seen += 1

    assert floors_seen > 0, (
        "After 30 sits the throne should have become FLOOR at least once "
        "(expected ~10 at 1/3 probability); cite: sit.c line 224"
    )
