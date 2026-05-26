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


def _force_case(state, case_num: int):
    """Run sit_throne with a seed that deterministically hits vendor case_num.

    Scans PRNGKey seeds until BOTH:
      (a) the outer gate `rnd(6) > 4` fires, AND
      (b) `rnd(13)` returns case_num.

    sit_throne does:
        rng, rng_gate, rng_outcome, rng_effect, rng_remove = split(rng, 5)
    """
    from Nethax.nethax.subsystems.throne import sit_throne
    from Nethax.nethax.rng import rnd
    for seed in range(2000):
        key = jax.random.PRNGKey(seed)
        splits = jax.random.split(key, 5)
        rng_gate    = splits[1]
        rng_outcome = splits[2]
        gate_roll = int(rnd(rng_gate, 6))
        if gate_roll <= 4:
            continue
        roll = int(rnd(rng_outcome, 13))
        if roll == case_num:
            return sit_throne(state, key)
    pytest.skip(f"Could not find seed for case {case_num} in 2000 tries")


# ---------------------------------------------------------------------------
# test_sit_throne_take_gold (vendor case 5)
# Vendor case 5 = take_gold(): hero loses all gold.
# Cite: sit.c lines 102-104.
# ---------------------------------------------------------------------------

def test_sit_throne_gold_outcome():
    state = _make_state_on_throne()
    state = state.replace(player_gold=jnp.int32(500))
    new_state = _force_case(state, case_num=5)
    assert int(new_state.player_gold) == 0, (
        "Vendor case 5 (take_gold) should zero player_gold; "
        "cite: sit.c lines 102-104"
    )


# ---------------------------------------------------------------------------
# test_sit_throne_lightning_damage (vendor case 3)
# Case 3 = electric shock: player_hp must decrease.
# Cite: sit.c lines 77-82 — losehp(Shock_resistance ? rnd(6) : rnd(30), ...).
# ---------------------------------------------------------------------------

def test_sit_throne_lightning_damage():
    state = _make_state_on_throne()
    state = state.replace(player_hp=jnp.int32(200), player_hp_max=jnp.int32(200))
    initial_hp = int(state.player_hp)
    new_state = _force_case(state, case_num=3)
    assert int(new_state.player_hp) < initial_hp, (
        "Vendor case 3 (electrocute) should reduce player_hp; "
        "cite: sit.c lines 77-82"
    )


# ---------------------------------------------------------------------------
# test_sit_throne_eventually_disappears
# Sit many times; the throne tile should become FLOOR at least once.
# With outer gate P=1/3 and removal P=1/3, expected ~1/9 per sit.
# Cite: sit.c lines 224-226 — if (!rn2(3)) levl[tx][ty].typ = ROOM.
# ---------------------------------------------------------------------------

def test_sit_throne_eventually_disappears():
    from Nethax.nethax.subsystems.throne import sit_throne
    from Nethax.nethax.constants.tiles import TileType

    floors_seen = 0
    for seed in range(90):
        state = _make_state_on_throne()
        b  = int(state.dungeon.current_branch)
        lv = int(state.dungeon.current_level) - 1
        key = jax.random.PRNGKey(seed + 1000)
        new_state = sit_throne(state, key)
        tile = int(new_state.terrain[b, lv, 0, 0])
        if tile == int(TileType.FLOOR):
            floors_seen += 1

    assert floors_seen > 0, (
        "After 90 sits the throne should have become FLOOR at least once "
        "(expected ~10 at 1/9 probability); cite: sit.c lines 224-226"
    )
