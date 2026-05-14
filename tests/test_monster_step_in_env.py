"""Wave 5 Phase 0a tests — monster_ai.step is wired into env.step.

Verifies that env.step now invokes the monster AI tick after the player
action and before status_effects.step, so monsters move/attack each turn.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp


def _env_with_adjacent_monster(
    player_pos=(10, 10),
    monster_pos=(10, 11),
    player_hp: int = 20,
    monster_hp: int = 50,
    monster_dice_n: int = 1,
    monster_dice_sides: int = 6,
):
    """Build a NethaxEnv state with one live, awake, hostile monster adjacent
    to the player and an all-floor map so movement is unrestricted.
    """
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.tiles import TileType

    rng = jax.random.PRNGKey(123)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    static = env.static
    # Carve full floor for branch 0 / level 0 so monster + player movement
    # is unrestricted.
    floor_map = jnp.full(
        (static.map_h, static.map_w), TileType.FLOOR, dtype=jnp.int8
    )
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor_map),
        player_pos=jnp.array(player_pos, dtype=jnp.int16),
        player_hp=jnp.int32(player_hp),
        player_hp_max=jnp.int32(max(player_hp, 20)),
    )

    # Clear all monster slots, then inject one at slot 0.
    mai = state.monster_ai
    mai = mai.replace(
        alive=jnp.zeros_like(mai.alive),
        asleep=jnp.zeros_like(mai.asleep),
        peaceful=jnp.zeros_like(mai.peaceful),
    )
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        asleep=mai.asleep.at[0].set(False),
        peaceful=mai.peaceful.at[0].set(False),
        hp=mai.hp.at[0].set(jnp.int32(monster_hp)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(monster_hp)),
        pos=mai.pos.at[0].set(jnp.array(monster_pos, dtype=jnp.int16)),
        ac=mai.ac.at[0].set(jnp.int8(10)),
        attack_dice_n=mai.attack_dice_n.at[0].set(jnp.int8(monster_dice_n)),
        attack_dice_sides=mai.attack_dice_sides.at[0].set(
            jnp.int8(monster_dice_sides)
        ),
    )
    state = state.replace(monster_ai=mai)
    return env, state, rng


def test_env_step_calls_monster_ai_step():
    """env.step should advance the monster: monster either moved closer or
    attacked the player (HP dropped). Either way, some monster-driven state
    change occurs on a WAIT step."""
    env, state, rng = _env_with_adjacent_monster(
        player_pos=(10, 10),
        monster_pos=(10, 12),  # 2 tiles away -> monster should move closer
        player_hp=20,
    )

    mpos_before = (int(state.monster_ai.pos[0, 0]), int(state.monster_ai.pos[0, 1]))
    hp_before = int(state.player_hp)

    rng, step_rng = jax.random.split(rng)
    action = jnp.int32(ord("."))  # WAIT
    state, _obs, _r, _done, _info = env.step(state, action, step_rng)

    mpos_after = (int(state.monster_ai.pos[0, 0]), int(state.monster_ai.pos[0, 1]))
    hp_after = int(state.player_hp)

    moved = mpos_after != mpos_before
    attacked = hp_after < hp_before

    assert moved or attacked, (
        f"Monster should have moved or attacked; "
        f"pos {mpos_before} -> {mpos_after}, hp {hp_before} -> {hp_after}"
    )


def test_monster_attacks_waiting_player():
    """Place a hostile monster adjacent to the player; after several WAIT
    steps the player's HP should decrease (monster_ai.step is firing)."""
    env, state, rng = _env_with_adjacent_monster(
        player_pos=(10, 10),
        monster_pos=(10, 11),  # adjacent
        player_hp=20,
        monster_hp=100,
        monster_dice_n=1,
        monster_dice_sides=6,
    )

    hp_before = int(state.player_hp)
    action = jnp.int32(ord("."))  # WAIT
    for _ in range(15):
        rng, step_rng = jax.random.split(rng)
        state, _obs, _r, done, _info = env.step(state, action, step_rng)
        if bool(done):
            break
        if int(state.player_hp) < hp_before:
            break

    hp_after = int(state.player_hp)
    assert hp_after < hp_before, (
        f"Adjacent hostile monster should damage waiting player: "
        f"hp_before={hp_before}, hp_after={hp_after}"
    )


def test_monster_ai_step_jit_compatible():
    """jax.jit(env.step) should compile cleanly with monster_ai.step in the
    pipeline and produce a state on the first call."""
    from Nethax.nethax.env import NethaxEnv

    rng = jax.random.PRNGKey(0)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    jitted_step = jax.jit(env.step)
    rng, step_rng = jax.random.split(rng)
    action = jnp.int32(ord("."))
    new_state, _obs, _r, _done, _info = jitted_step(state, action, step_rng)
    # Force materialization to ensure compilation actually ran end-to-end.
    _ = int(new_state.timestep)
    assert int(new_state.timestep) == int(state.timestep) + 1
