"""Wave 2/3 tests for EnvState pytree health and structural invariants.

Verifies that the state is a valid JAX pytree, that dtypes are stable
across steps, and that array shapes are constant across steps.

Wave 3 addition: test_dtypes_stable_after_wave3_writes verifies that
Wave 3 subsystem writes (combat, status, inventory, monster_ai) do not
introduce dtype promotion or shape changes.

All imports are lazy (inside test functions) so collection never fails.
"""

import pytest


def _make_state_and_env():
    """Helper: return (env, state) after one reset."""
    import jax
    from Nethax.nethax.env import NethaxEnv

    rng = jax.random.PRNGKey(0)
    env = NethaxEnv()
    state, _ = env.reset(rng)
    return env, state, rng


def test_state_is_pytree():
    """jax.tree.leaves(state) should yield only jax.Array / jnp.ndarray instances."""
    import jax
    import jax.numpy as jnp

    _env, state, _rng = _make_state_and_env()

    leaves = jax.tree.leaves(state)
    assert len(leaves) > 0, "State has no pytree leaves"

    non_array = [
        (i, type(leaf))
        for i, leaf in enumerate(leaves)
        if not isinstance(leaf, jax.Array)
    ]
    assert not non_array, (
        f"Non-array leaves found at indices: "
        + ", ".join(f"{i}: {t}" for i, t in non_array[:5])
    )


def test_state_dtypes_stable():
    """Dtypes of all state leaves must be identical before and after a step."""
    import jax
    import jax.numpy as jnp

    env, state, rng = _make_state_and_env()

    def leaf_dtypes(s):
        return [leaf.dtype for leaf in jax.tree.leaves(s)]

    before_dtypes = leaf_dtypes(state)

    rng, step_rng = jax.random.split(rng)
    action = jnp.int32(ord("."))
    state2, _, _, _, _ = env.step(state, action, step_rng)

    after_dtypes = leaf_dtypes(state2)

    assert len(before_dtypes) == len(after_dtypes), (
        f"Leaf count changed: {len(before_dtypes)} before, {len(after_dtypes)} after"
    )

    mismatches = [
        (i, str(b), str(a))
        for i, (b, a) in enumerate(zip(before_dtypes, after_dtypes))
        if b != a
    ]
    assert not mismatches, (
        "Dtype changes after step (leaf_index, before, after):\n"
        + "\n".join(f"  [{i}] {b} -> {a}" for i, b, a in mismatches[:10])
    )


def test_state_shapes_static():
    """Array shapes of all state leaves must be identical before and after multiple steps."""
    import jax
    import jax.numpy as jnp

    env, state, rng = _make_state_and_env()

    def leaf_shapes(s):
        return [leaf.shape for leaf in jax.tree.leaves(s)]

    initial_shapes = leaf_shapes(state)

    # Take 3 steps and verify shapes are unchanged at each step
    action = jnp.int32(ord("."))
    current = state
    for step_num in range(3):
        rng, step_rng = jax.random.split(rng)
        current, _, _, _, _ = env.step(current, action, step_rng)
        step_shapes = leaf_shapes(current)

        assert len(initial_shapes) == len(step_shapes), (
            f"Step {step_num+1}: leaf count changed "
            f"from {len(initial_shapes)} to {len(step_shapes)}"
        )

        mismatches = [
            (i, str(s0), str(s1))
            for i, (s0, s1) in enumerate(zip(initial_shapes, step_shapes))
            if s0 != s1
        ]
        assert not mismatches, (
            f"Shape changes at step {step_num+1} (leaf_index, initial, after):\n"
            + "\n".join(f"  [{i}] {s0} -> {s1}" for i, s0, s1 in mismatches[:10])
        )


def test_dtypes_stable_after_wave3_writes():
    """Wave 3: direct subsystem state writes must not change any leaf dtype.

    Simulates the kind of targeted writes Wave 3 agents make:
      - monster_ai.hp / alive / pos
      - status.nutrition / hunger_state
      - inventory.worn_armor / wielded
      - combat.last_hit_landed / last_attack_kind

    After each write the full leaf-dtype list must be identical to the
    pre-write baseline.
    """
    import jax
    import jax.numpy as jnp

    env, state, rng = _make_state_and_env()

    def leaf_dtypes(s):
        return [leaf.dtype for leaf in jax.tree.leaves(s)]

    baseline = leaf_dtypes(state)

    # --- monster_ai writes ---
    mai = state.monster_ai
    mai = mai.replace(
        hp=mai.hp.at[0].set(jnp.int32(20)),
        alive=mai.alive.at[0].set(True),
        pos=mai.pos.at[0].set(jnp.array([5, 5], dtype=jnp.int16)),
    )
    state1 = state.replace(monster_ai=mai)
    assert leaf_dtypes(state1) == baseline, "monster_ai writes changed leaf dtypes"

    # --- status writes ---
    st = state.status
    st = st.replace(
        nutrition=jnp.int32(500),
        hunger_state=jnp.int8(2),
    )
    state2 = state.replace(status=st)
    assert leaf_dtypes(state2) == baseline, "status writes changed leaf dtypes"

    # --- inventory writes ---
    inv = state.inventory
    inv = inv.replace(
        wielded=jnp.int8(0),
        worn_armor=inv.worn_armor.at[0].set(jnp.int8(3)),
    )
    state3 = state.replace(inventory=inv)
    assert leaf_dtypes(state3) == baseline, "inventory writes changed leaf dtypes"

    # --- combat writes ---
    cbt = state.combat
    cbt = cbt.replace(
        last_hit_landed=jnp.bool_(True),
        last_attack_kind=jnp.int32(1),
    )
    state4 = state.replace(combat=cbt)
    assert leaf_dtypes(state4) == baseline, "combat writes changed leaf dtypes"
