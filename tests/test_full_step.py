"""Wave 3 integration tests — end-to-end env.step pipeline.

Tests cover the full lifecycle: reset -> step loop, JIT stability, and
observation completeness after Wave 3 state writes.

All imports are lazy (inside test functions) so collection never fails if
a module is still being implemented by another agent.
"""

import pytest


def test_full_step_lifecycle():
    """reset -> 10 steps with random actions -> no errors, state shapes invariant."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.state import EnvState

    rng = jax.random.PRNGKey(0)
    env = NethaxEnv()
    state, obs = env.reset(rng)

    assert isinstance(state, EnvState)

    initial_leaves = jax.tree.leaves(state)
    initial_shapes = [leaf.shape for leaf in initial_leaves]
    initial_dtypes = [leaf.dtype for leaf in initial_leaves]

    from Nethax.nethax.constants import ACTIONS
    action_values = [int(a) for a in ACTIONS]
    n_actions = len(action_values)

    for i in range(10):
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        # Pick a random action from the valid action set
        idx = int(jax.random.randint(action_rng, shape=(), minval=0, maxval=n_actions))
        action = jnp.int32(action_values[idx])
        state, obs, reward, done, info = env.step(state, action, step_rng)

        step_leaves = jax.tree.leaves(state)
        step_shapes = [leaf.shape for leaf in step_leaves]
        step_dtypes = [leaf.dtype for leaf in step_leaves]

        assert len(step_shapes) == len(initial_shapes), (
            f"Step {i+1}: leaf count changed from {len(initial_shapes)} to {len(step_shapes)}"
        )
        shape_mismatches = [
            (j, s0, s1)
            for j, (s0, s1) in enumerate(zip(initial_shapes, step_shapes))
            if s0 != s1
        ]
        assert not shape_mismatches, (
            f"Step {i+1}: shape changes: "
            + ", ".join(f"[{j}] {s0}->{s1}" for j, s0, s1 in shape_mismatches[:5])
        )
        dtype_mismatches = [
            (j, d0, d1)
            for j, (d0, d1) in enumerate(zip(initial_dtypes, step_dtypes))
            if d0 != d1
        ]
        assert not dtype_mismatches, (
            f"Step {i+1}: dtype changes: "
            + ", ".join(f"[{j}] {d0}->{d1}" for j, d0, d1 in dtype_mismatches[:5])
        )

        assert obs.keys() == set(
            ["glyphs", "chars", "colors", "specials", "blstats", "message",
             "program_state", "internal", "inv_glyphs", "inv_letters",
             "inv_oclasses", "inv_strs", "screen_descriptions",
             "tty_chars", "tty_colors", "tty_cursor", "misc"]
        )


@pytest.mark.timeout(300)
def test_jit_full_step():
    """jax.jit(env.step) runs without retracing across 5 steps.

    Wave34e: cold compile of the full env.step graph is ~200s on CPU;
    timeout bumped to 300s so the test isn't killed by the 120s default.
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv

    rng = jax.random.PRNGKey(0)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    jitted_step = jax.jit(env.step)

    # Compile on the first call.
    action = jnp.int32(ord("."))  # wait action
    rng, step_rng = jax.random.split(rng)
    state, obs, reward, done, info = jitted_step(state, action, step_rng)
    # Force materialization
    _ = int(state.timestep)

    # Subsequent calls should not retrace (no shape/dtype change).
    for i in range(4):
        rng, step_rng = jax.random.split(rng)
        state, obs, reward, done, info = jitted_step(state, action, step_rng)
        ts = int(state.timestep)
        assert ts == i + 2, f"Expected timestep {i+2}, got {ts}"


def test_obs_all_keys_populated():
    """After a step, no obs key has an obviously wrong shape or dtype mismatch.

    We check that all 17 keys are present and have the canonical NLE shapes/dtypes.
    Keys that are legitimately all-zero (Wave 2 stubs) are accepted; we only
    verify structure, not values.
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.obs.nle_obs import NLE_OBSERVATION_SHAPES, NLE_OBSERVATION_DTYPES

    rng = jax.random.PRNGKey(0)
    env = NethaxEnv()
    state, _ = env.reset(rng)
    rng, step_rng = jax.random.split(rng)
    action = jnp.int32(ord("."))
    _, obs, _, _, _ = env.step(state, action, step_rng)

    assert len(obs) == 17, f"Expected 17 obs keys, got {len(obs)}: {sorted(obs)}"

    for key in NLE_OBSERVATION_SHAPES:
        assert key in obs, f"Missing obs key: {key}"
        expected_shape = NLE_OBSERVATION_SHAPES[key]
        expected_dtype = NLE_OBSERVATION_DTYPES[key]
        actual_shape = obs[key].shape
        actual_dtype = obs[key].dtype
        assert actual_shape == expected_shape, (
            f"obs['{key}']: shape {actual_shape} != expected {expected_shape}"
        )
        assert actual_dtype == expected_dtype, (
            f"obs['{key}']: dtype {actual_dtype} != expected {expected_dtype}"
        )

    # Keys that should have at least some non-zero content after reset + step:
    # glyphs — map was generated, player glyph is placed
    assert int(jnp.any(obs["glyphs"] != 0)), (
        "obs['glyphs'] is all zeros after a step — map generation may not be wired"
    )
    # blstats — player HP, position, etc. are non-zero
    assert int(jnp.any(obs["blstats"] != 0)), (
        "obs['blstats'] is all zeros — blstat builder may be broken"
    )
