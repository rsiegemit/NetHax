"""Wave 5 Phase 0b — bump-attack bridge in action_dispatch._try_step.

Verifies that movement into a tile occupied by an alive monster routes
through ``combat.melee_attack`` (vendor/nethack/src/hack.c::domove) and
does NOT move the player.  Also covers the run-loop short-circuit and
JIT-compatibility of the new code path.
"""
import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants import TileType
from Nethax.nethax.constants.actions import (
    CompassCardinalDirection,
    CompassCardinalDirectionLonger,
)


# ---------------------------------------------------------------------------
# Shared fixture builder — place a monster east of the player on FLOOR tiles.
# ---------------------------------------------------------------------------
def _env_with_optional_monster(monster_hp=10, monster_present=True,
                                monster_ac=10, player_hp=20):
    rng = jax.random.PRNGKey(7)
    env = NethaxEnv()
    state, _ = env.reset(rng)
    state = state.replace(player_hp=jnp.int32(player_hp))

    p_row = int(state.player_pos[0])
    p_col = int(state.player_pos[1])
    branch = int(state.dungeon.current_branch)
    level_idx = int(state.dungeon.current_level) - 1

    # Carve a horizontal floor corridor around the player (cols p_col-1 .. p_col+5).
    new_terrain = state.terrain
    for c in range(max(0, p_col - 1), min(state.terrain.shape[3], p_col + 6)):
        new_terrain = new_terrain.at[branch, level_idx, p_row, c].set(
            jnp.int8(TileType.FLOOR)
        )
    state = state.replace(terrain=new_terrain)

    # Clear all monsters first so any seeded population can't collide with
    # the path east of the player.
    mai = state.monster_ai
    n = mai.alive.shape[0]
    mai = mai.replace(
        alive=jnp.zeros_like(mai.alive),
        hp=jnp.zeros_like(mai.hp),
        hp_max=jnp.zeros_like(mai.hp_max),
        pos=jnp.full_like(mai.pos, -1),
    )

    if monster_present:
        m_row, m_col = p_row, p_col + 1
        mai = mai.replace(
            alive=mai.alive.at[0].set(True),
            hp=mai.hp.at[0].set(jnp.int32(monster_hp)),
            hp_max=mai.hp_max.at[0].set(jnp.int32(monster_hp)),
            pos=mai.pos.at[0].set(jnp.array([m_row, m_col], dtype=jnp.int16)),
            ac=mai.ac.at[0].set(jnp.int8(monster_ac)),
        )
    state = state.replace(monster_ai=mai)
    return env, state, rng


# ---------------------------------------------------------------------------
# Test 1: bump-attack on adjacent monster — monster HP drops or it dies;
# player stays put.
# ---------------------------------------------------------------------------
def test_player_attacks_adjacent_monster_via_move():
    env, state, rng = _env_with_optional_monster(
        monster_hp=200, monster_ac=10, player_hp=50,
    )
    p_row = int(state.player_pos[0])
    p_col = int(state.player_pos[1])
    monster_hp_before = int(state.monster_ai.hp[0])

    action = jnp.int32(int(CompassCardinalDirection.E))
    # Drive the engine with several attacks; at least one should land
    # against AC10 with default Valkyrie stats.
    landed = False
    for _ in range(20):
        rng, step_rng = jax.random.split(rng)
        state, _, _, done, _ = env.step(state, action, step_rng)
        if int(state.monster_ai.hp[0]) < monster_hp_before:
            landed = True
            break
        if bool(done):
            break

    assert landed, "Expected at least one attack to reduce monster HP"
    # Player should NOT have moved while monster occupies the tile.
    assert int(state.player_pos[0]) == p_row
    assert int(state.player_pos[1]) == p_col


# ---------------------------------------------------------------------------
# Test 2: open path (no monster) — player_pos.col += 1.
# ---------------------------------------------------------------------------
def test_player_moves_if_no_monster():
    env, state, rng = _env_with_optional_monster(monster_present=False)
    p_row = int(state.player_pos[0])
    p_col = int(state.player_pos[1])

    action = jnp.int32(int(CompassCardinalDirection.E))
    rng, step_rng = jax.random.split(rng)
    state, _, _, _, _ = env.step(state, action, step_rng)

    assert int(state.player_pos[0]) == p_row
    assert int(state.player_pos[1]) == p_col + 1


# ---------------------------------------------------------------------------
# Test 3: full kill via bump-attack — player_xp increases.
# ---------------------------------------------------------------------------
def test_bump_attack_kills_monster_grants_xp():
    env, state, rng = _env_with_optional_monster(monster_hp=3, monster_ac=10)
    xp_before = int(state.player_xp)

    action = jnp.int32(int(CompassCardinalDirection.E))
    for _ in range(30):
        rng, step_rng = jax.random.split(rng)
        state, _, _, done, _ = env.step(state, action, step_rng)
        if not bool(state.monster_ai.alive[0]):
            break
        if bool(done):
            break

    assert not bool(state.monster_ai.alive[0]), "Monster should be dead"
    assert int(state.player_xp) > xp_before, (
        f"Expected XP > {xp_before}, got {int(state.player_xp)}"
    )


# ---------------------------------------------------------------------------
# Test 4: _run_e halts when a monster blocks — player ends adjacent to monster.
# ---------------------------------------------------------------------------
def test_run_stops_at_monster():
    # Place monster 2 tiles east of player; run east should advance 1 then halt.
    env, state, rng = _env_with_optional_monster(monster_present=False)

    p_row = int(state.player_pos[0])
    p_col = int(state.player_pos[1])

    # Inject monster two tiles east.
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        hp=mai.hp.at[0].set(jnp.int32(10)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(10)),
        pos=mai.pos.at[0].set(jnp.array([p_row, p_col + 2], dtype=jnp.int16)),
        ac=mai.ac.at[0].set(jnp.int8(10)),
    )
    state = state.replace(monster_ai=mai)

    # Run east — should stop adjacent to monster (one tile east) without
    # entering the monster's tile.  Capital-E is the run-east action.
    action = jnp.int32(int(CompassCardinalDirectionLonger.E))
    rng, step_rng = jax.random.split(rng)
    state, _, _, _, _ = env.step(state, action, step_rng)

    # Player should have advanced to (p_row, p_col+1) and stopped there.
    assert int(state.player_pos[0]) == p_row
    assert int(state.player_pos[1]) == p_col + 1, (
        f"Run should stop adjacent to monster, got col={int(state.player_pos[1])}"
    )
    # Monster should still be at its original tile (run halted before bump-attack).
    assert int(state.monster_ai.pos[0, 1]) == p_col + 2


# ---------------------------------------------------------------------------
# Test 5: dispatch is JIT-compilable with the bump-attack branch.
# ---------------------------------------------------------------------------
def test_jit_compile_dispatch_with_bump_attack():
    env, state, rng = _env_with_optional_monster(monster_hp=5)

    @jax.jit
    def jit_step(s, a, r):
        return env.step(s, a, r)

    action = jnp.int32(int(CompassCardinalDirection.E))
    rng, step_rng = jax.random.split(rng)
    state2, _, _, _, _ = jit_step(state, action, step_rng)
    # Compile + first call must succeed and return a valid state.
    assert state2.player_pos.shape == state.player_pos.shape
    # Player tile should be unchanged (monster occupies adjacent tile).
    assert int(state2.player_pos[0]) == int(state.player_pos[0])
    assert int(state2.player_pos[1]) == int(state.player_pos[1])
