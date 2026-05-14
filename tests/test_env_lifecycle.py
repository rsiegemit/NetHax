"""Wave 2 end-to-end environment lifecycle tests.

Tests reset, step, JIT compilation, movement, and dungeon visibility.
All imports are lazy (inside test functions) so collection never fails
if a module is still being implemented by another agent.
"""

import pytest


def test_reset_returns_valid_state():
    """reset() must return (EnvState, obs) where obs has exactly 17 keys."""
    import jax
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.state import EnvState

    rng = jax.random.PRNGKey(0)
    env = NethaxEnv()
    state, obs = env.reset(rng)

    assert isinstance(state, EnvState), f"Expected EnvState, got {type(state)}"
    assert len(obs) == 17, f"Expected 17 obs keys, got {len(obs)}: {list(obs.keys())}"


def test_step_returns_5_tuple():
    """step() must return a 5-tuple (state, obs, reward, done, info) with correct types."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.state import EnvState

    rng = jax.random.PRNGKey(0)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    rng, step_rng = jax.random.split(rng)
    action = jnp.int32(ord("."))  # wait action
    result = env.step(state, action, step_rng)

    assert len(result) == 5, f"step() should return 5-tuple, got length {len(result)}"
    new_state, obs, reward, done, info = result

    assert isinstance(new_state, EnvState)
    assert len(obs) == 17
    assert reward.dtype == jnp.float32, f"reward dtype={reward.dtype}, expected float32"
    assert done.dtype == jnp.bool_, f"done dtype={done.dtype}, expected bool"
    assert isinstance(info, dict)


def test_step_advances_timestep():
    """Each step must increment the timestep by exactly 1.

    Wave 3 alignment: this invariant must hold even after Wave 3 subsystem
    dispatchers (combat, doors, traps, hunger) are wired into env.step.
    The timestep increment lives in env.py and must not be conditional on
    action type or game state.
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv

    rng = jax.random.PRNGKey(0)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    t0 = int(state.timestep)

    rng, step_rng = jax.random.split(rng)
    action = jnp.int32(ord("."))
    state2, _, _, _, _ = env.step(state, action, step_rng)

    assert int(state2.timestep) == t0 + 1, (
        f"Expected timestep {t0+1}, got {int(state2.timestep)}"
    )

    rng, step_rng2 = jax.random.split(rng)
    state3, _, _, _, _ = env.step(state2, action, step_rng2)
    assert int(state3.timestep) == t0 + 2

    # Wave 3: also verify with a movement action (not just wait)
    from Nethax.nethax.constants.actions import CompassCardinalDirection
    t3 = int(state3.timestep)
    rng, step_rng3 = jax.random.split(rng)
    move_action = jnp.int32(int(CompassCardinalDirection.N))
    state4, _, _, _, _ = env.step(state3, move_action, step_rng3)
    assert int(state4.timestep) == t3 + 1, (
        f"Expected timestep {t3+1} after move action, got {int(state4.timestep)}"
    )


def test_env_jit_compiles():
    """env.step wrapped in jax.jit must compile and execute without error."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv

    rng = jax.random.PRNGKey(0)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    rng, step_rng = jax.random.split(rng)
    action = jnp.int32(ord("."))

    jitted_step = jax.jit(env.step)
    new_state, obs, reward, done, info = jitted_step(state, action, step_rng)

    # Force materialisation to confirm the compiled function ran
    assert new_state.timestep.shape == ()


def test_movement_action_changes_pos():
    """Moving east (action 'l') onto a floor tile should increment player col by 1."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants import TileType
    from Nethax.nethax.constants.actions import CompassCardinalDirection

    rng = jax.random.PRNGKey(0)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    # Manually carve a floor tile to the east of the player so movement is not blocked.
    branch = int(state.dungeon.current_branch)
    level_idx = int(state.dungeon.current_level) - 1
    r = int(state.player_pos[0])
    c = int(state.player_pos[1])
    east_c = c + 1

    if east_c < state.terrain.shape[3]:
        # Set the east tile to FLOOR so the move is guaranteed to succeed.
        new_terrain = state.terrain.at[branch, level_idx, r, east_c].set(
            jnp.int8(TileType.FLOOR)
        )
        state = state.replace(terrain=new_terrain)

        action = jnp.int32(int(CompassCardinalDirection.E))  # ord('l')
        rng, step_rng = jax.random.split(rng)
        new_state, _, _, _, _ = env.step(state, action, step_rng)

        assert int(new_state.player_pos[1]) == east_c, (
            f"Expected col {east_c} after east move, got {int(new_state.player_pos[1])}"
        )
    else:
        pytest.skip("Player already at eastern map boundary; cannot test east movement")


def test_dungeon_visible_after_reset():
    """After reset the terrain array must contain some non-VOID tiles (map was generated)."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants import TileType

    rng = jax.random.PRNGKey(0)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    # terrain: [N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]
    branch = int(state.dungeon.current_branch)
    level_idx = int(state.dungeon.current_level) - 1
    level_terrain = state.terrain[branch, level_idx]

    n_non_void = int(jnp.sum(level_terrain != jnp.int8(TileType.VOID)))
    assert n_non_void > 0, (
        "All terrain tiles are VOID after reset — dungeon generation may not be wired in. "
        f"terrain shape={state.terrain.shape}, branch={branch}, level_idx={level_idx}"
    )
