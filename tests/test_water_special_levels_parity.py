"""Parity tests for vendor drown formula and special level factories.

Vendor references:
  vendor/nethack/src/trap.c::drown()  lines 5059-5195
  vendor/nethack/dat/soko1-1.lua      (Sokoban floor 1 — pit traps)
  vendor/nethack/dat/soko4-1.lua      (Sokoban floor 4 — pit traps)
  vendor/nethack/dat/castle.lua       (Castle — moat / POOL tiles)
  vendor/nethack/dat/oracle.lua       (Oracle — Oracle monster)
  vendor/nethack/dat/astral.lua       (Astral Plane — 3 altars)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.subsystems.water import water_step
from Nethax.nethax.dungeon.special_levels import (
    generate_oracle_level,
    generate_castle_level,
    generate_sokoban_floor_1,
    generate_sokoban_floor_4,
    generate_astral_plane,
    _MON_ORACLE,
    _T_WATER,
    _T_TRAP,
    _T_ALTAR,
)

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_state():
    env = NethaxEnv()
    state, _ = env.reset(_RNG)
    return state


def _set_in_water(state, in_water: bool, turns: int = 0):
    return state.replace(
        player_in_water=jnp.bool_(in_water),
        turns_underwater=jnp.int16(turns),
    )


# ---------------------------------------------------------------------------
# test_turns_underwater_increments
# ---------------------------------------------------------------------------

def test_turns_underwater_increments():
    """Each water_step call while in water must increment turns_underwater.

    Cite: vendor/nethack/src/trap.c::drown() line 5059 — the vendor tracks
    turns-in-water to compute drowning probability.
    """
    state = _fresh_state()
    state = _set_in_water(state, True, turns=0)
    # Give the player enough HP to survive 10 ticks.
    state = state.replace(player_hp=jnp.int32(10_000), player_hp_max=jnp.int32(10_000))

    rng = _RNG
    for _ in range(10):
        rng, sub = jax.random.split(rng)
        state = water_step(state, sub)

    assert int(state.turns_underwater) == 10, (
        f"Expected turns_underwater==10 after 10 water ticks, "
        f"got {int(state.turns_underwater)}"
    )


# ---------------------------------------------------------------------------
# test_turns_underwater_resets_on_leaving
# ---------------------------------------------------------------------------

def test_turns_underwater_resets_on_leaving():
    """water_step while NOT in water must reset turns_underwater to 0.

    This is the leave-water branch in the revised water_step.
    """
    state = _fresh_state()
    state = _set_in_water(state, False, turns=15)

    rng, sub = jax.random.split(_RNG)
    state = water_step(state, sub)

    assert int(state.turns_underwater) == 0, (
        f"Expected turns_underwater==0 after leaving water, "
        f"got {int(state.turns_underwater)}"
    )


# ---------------------------------------------------------------------------
# test_insta_drown_chance
# ---------------------------------------------------------------------------

def test_insta_drown_chance():
    """With turns_underwater > 20, insta-drown must kill the player within
    at most 100 five-turn check cycles (statistically near-certain at N=25+).

    Vendor formula: trap.c::drown line 5059 — rnl(50) <= turns_underwater.
    At turns_underwater=25 the probability per check is 26/50 > 50 %.
    """
    state = _fresh_state()
    state = state.replace(player_hp=jnp.int32(100), player_hp_max=jnp.int32(100))

    # Start at turns_underwater=20 so the very next check (turn 25) has
    # p(drown) = 26/50 > 50 %.  After 100 ticks (~20 check cycles) the
    # player is almost certainly dead.
    state = _set_in_water(state, True, turns=20)

    rng = _RNG
    died = False
    for _ in range(100):
        rng, sub = jax.random.split(rng)
        state = water_step(state, sub)
        if int(state.player_hp) == 0:
            died = True
            break

    assert died, (
        "Player should have drowned within 100 ticks with turns_underwater > 20 "
        "(trap.c::drown line 5059)"
    )


# ---------------------------------------------------------------------------
# test_oracle_level_has_oracle_monster
# ---------------------------------------------------------------------------

def test_oracle_level_has_oracle_monster():
    """ORACLE special level must place the Oracle NPC monster.

    Cite: vendor/nethack/dat/oracle.lua line 23 — des.monster("Oracle", 1,1).
    """
    _, monsters, _ = generate_oracle_level(_RNG)
    monster_types = monsters[:, 2]
    assert int(jnp.any(monster_types == _MON_ORACLE)), (
        "Oracle level must place the Oracle monster (oracle.lua line 23)"
    )


# ---------------------------------------------------------------------------
# test_castle_has_moat
# ---------------------------------------------------------------------------

def test_castle_has_moat():
    """CASTLE terrain must contain WATER (moat) tiles.

    Cite: vendor/nethack/dat/castle.lua — '}}' moat tiles (mapped to WATER).
    """
    terrain, _, _ = generate_castle_level(_RNG)
    n_water = int(jnp.sum(terrain == _T_WATER))
    assert n_water >= 1, (
        f"Castle terrain should have >=1 WATER (moat) tile, got {n_water} "
        "(castle.lua moat '}}' rows)"
    )


# ---------------------------------------------------------------------------
# test_sokoban_1_has_pits
# ---------------------------------------------------------------------------

def test_sokoban_1_has_pits():
    """SOKOBAN_1 terrain must contain PIT trap tiles.

    Cite: vendor/nethack/dat/soko1-1.lua lines 65-82 — hole traps along col 1.
    """
    terrain, _, _ = generate_sokoban_floor_1(_RNG)
    n_traps = int(jnp.sum(terrain == _T_TRAP))
    assert n_traps >= 1, (
        f"Sokoban floor 1 must have >=1 trap (pit/hole) tile, got {n_traps} "
        "(soko1-1.lua lines 65-82)"
    )


# ---------------------------------------------------------------------------
# test_sokoban_4_has_pits
# ---------------------------------------------------------------------------

def test_sokoban_4_has_pits():
    """SOKOBAN_4 terrain must contain PIT trap tiles.

    Cite: vendor/nethack/dat/soko4-1.lua lines 79-91 — pit traps.
    """
    terrain, _, _ = generate_sokoban_floor_4(_RNG)
    n_traps = int(jnp.sum(terrain == _T_TRAP))
    assert n_traps >= 1, (
        f"Sokoban floor 4 must have >=1 trap tile, got {n_traps} "
        "(soko4-1.lua lines 79-91)"
    )


# ---------------------------------------------------------------------------
# test_astral_has_3_altars
# ---------------------------------------------------------------------------

def test_astral_has_3_altars():
    """ASTRAL_PLANE terrain must contain exactly 3 altars (one per alignment).

    Cite: vendor/nethack/dat/astral.lua lines 89-91 — three sanctum altars.
    """
    terrain, _, _ = generate_astral_plane(_RNG)
    n_altars = int(jnp.sum(terrain == _T_ALTAR))
    assert n_altars == 3, (
        f"Astral Plane must have exactly 3 altars (Law/Neutral/Chaos), "
        f"got {n_altars} (astral.lua lines 89-91)"
    )
