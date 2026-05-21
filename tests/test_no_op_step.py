"""Tests that each subsystem's step() returns a JAX-equal state (Wave 1 no-op).

Each test:
  1. Constructs a default state.
  2. Calls step(state, rng).
  3. Asserts that every leaf of the returned pytree equals the input leaf.
"""

import jax
import jax.numpy as jnp
import pytest


_RNG = jax.random.PRNGKey(0)


def _trees_equal(a, b) -> bool:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    if len(leaves_a) != len(leaves_b):
        return False
    return all(jnp.array_equal(x, y) for x, y in zip(leaves_a, leaves_b))


# ---------------------------------------------------------------------------
# Subsystems with classmethod default() + step(state, rng)
# ---------------------------------------------------------------------------

def test_combat_step_noop():
    from Nethax.nethax.subsystems.combat import CombatState, step
    state = CombatState.default()
    assert _trees_equal(step(state, _RNG), state)


def test_magic_step_noop():
    """Wave 3: magic step now takes EnvState (needs player_xl for regen formula).
    With no Pw, no spells, step leaves magic slice unchanged.
    """
    import jax
    from Nethax.nethax import NethaxEnv
    from Nethax.nethax.subsystems.magic import step as magic_step
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    new_state = magic_step(state, jax.random.PRNGKey(1))
    # spell_known should be unchanged across one tick
    assert (new_state.magic.spell_known == state.magic.spell_known).all()


def test_shop_step_noop():
    from Nethax.nethax.subsystems.shop import ShopState, step
    state = ShopState.default()
    assert _trees_equal(step(state, _RNG), state)


def test_prayer_step_noop():
    from Nethax.nethax.subsystems.prayer import PrayerState, step
    state = PrayerState.default()
    assert _trees_equal(step(state, _RNG), state)


def test_conduct_step_noop():
    from Nethax.nethax.subsystems.conduct import ConductState, step
    state = ConductState.default()
    assert _trees_equal(step(state, _RNG), state)


@pytest.mark.timeout(300)
def test_status_effects_step_noop():
    """Wave 3: step() now takes full EnvState (hunger / regen ticks need player_hp etc).
    No-op semantics: input action contributes nothing; status state unchanged when
    player has not taken damage and is well-fed.

    Wave34e: bump timeout — first call to env.step is an eager retrace
    of the full graph (~200s) since this test doesn't jit-compile.
    """
    import jax
    from Nethax.nethax import NethaxEnv
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    new_state, *_ = env.step(state, 0, jax.random.PRNGKey(1))
    # status timed_intrinsics & timed_statuses should be unchanged (no triggers)
    assert (new_state.status.intrinsics == state.status.intrinsics).all()


# ---------------------------------------------------------------------------
# Subsystems with module-level factory functions + step(state, rng)
# ---------------------------------------------------------------------------

def test_polymorph_step_noop():
    from Nethax.nethax.subsystems.polymorph import make_polymorph_state, step
    state = make_polymorph_state()
    assert _trees_equal(step(state, _RNG), state)


def test_monster_ai_step_noop():
    """Wave 3: step() now takes full EnvState. With no monsters spawned, step is no-op."""
    import jax
    from Nethax.nethax import NethaxEnv
    from Nethax.nethax.subsystems.monster_ai import monsters_step_all
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    # Clear any spawned monsters so step is truly no-op
    state = state.replace(monster_ai=state.monster_ai.replace(
        alive=state.monster_ai.alive.at[:].set(False)
    ))
    new_state = monsters_step_all(state, jax.random.PRNGKey(1))
    assert (new_state.monster_ai.alive == state.monster_ai.alive).all()


# ---------------------------------------------------------------------------
# Subsystems with dimension-parameterized default()
# ---------------------------------------------------------------------------

def test_traps_step_noop():
    from Nethax.nethax.subsystems.traps import TrapState, step
    state = TrapState.default(num_levels=1, map_h=21, map_w=79)
    assert _trees_equal(step(state, _RNG), state)


def test_features_step_noop():
    from Nethax.nethax.subsystems.features import FeaturesState, step
    state = FeaturesState.default(num_levels=1, map_h=21, map_w=79)
    assert _trees_equal(step(state, _RNG), state)


# ---------------------------------------------------------------------------
# Inventory (uses .empty() factory)
# ---------------------------------------------------------------------------

def test_inventory_step_noop():
    from Nethax.nethax.subsystems.inventory import InventoryState, step
    state = InventoryState.empty()
    assert _trees_equal(step(state, _RNG), state)
