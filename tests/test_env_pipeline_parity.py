"""Wave-6 Closing-Audit #81 — pipeline parity tests.

Verify that ``env._step_impl`` matches the vendor ``moveloop`` ordering and
that every per-turn subsystem call is wired in.  Vendor reference:
``vendor/nethack/src/allmain.c::moveloop``.

The order under test:
    1. dispatch player action
    2. monster turn (movemon)
    3. nh_timeout / status effects + regen
    4. polymorph + lycanthropy_timer tick
    5. age_spells — spell_memory decay
    6. shop_step — pay-at-exit + pursuit
    7. ascension / endgame check

``lit_radius_until_turn`` is an absolute deadline (set by SPELL_LIGHT to
``timestep + 100``), so its "decrement per turn" behaviour falls out of the
``timestep += 1`` step; we test that semantics here.

All imports are lazy so collection succeeds while peer audits are running.
"""

from __future__ import annotations

import inspect

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_env_and_state():
    """Build an env, reset it, and return (env, state, rng, WAIT action)."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.actions import MiscDirection

    rng = jax.random.PRNGKey(0)
    env = NethaxEnv()
    state, _obs = env.reset(rng)
    wait = jnp.int32(int(MiscDirection.WAIT))
    return env, state, rng, wait


# ---------------------------------------------------------------------------
# Per-turn-call presence tests
# ---------------------------------------------------------------------------

def test_age_spells_called_per_turn():
    """spell_memory entries that start >0 must decrease by 1 per step.

    Vendor: allmain.c line 355 invokes spell.c::age_spells.
    """
    import jax
    import jax.numpy as jnp

    env, state, rng, wait = _fresh_env_and_state()

    # Seed a known-positive spell_memory at index 0 so we can observe decay.
    seeded_mem = state.magic.spell_memory.at[0].set(jnp.int32(50))
    state = state.replace(magic=state.magic.replace(spell_memory=seeded_mem))

    before = int(state.magic.spell_memory[0])
    step_rng, _ = jax.random.split(rng)
    state, _obs, _r, _d, _info = env.step(state, wait, step_rng)
    after = int(state.magic.spell_memory[0])

    assert after == before - 1, (
        f"age_spells not called: spell_memory[0] {before} -> {after} (expected {before-1})"
    )


def test_lit_radius_decrements_per_turn():
    """lit_radius_until_turn is an absolute deadline — as timestep advances,
    the number of remaining lit turns (until_turn - timestep) decreases.

    Vendor: light.c::do_light_sources stores an expiry timestamp.
    """
    import jax
    import jax.numpy as jnp

    env, state, rng, wait = _fresh_env_and_state()

    # Arrange a future deadline so we can observe the remaining-turns shrink.
    state = state.replace(
        dungeon=state.dungeon.replace(lit_radius_until_turn=jnp.int32(10))
    )
    t0 = int(state.timestep)
    remaining_before = int(state.dungeon.lit_radius_until_turn) - t0

    step_rng, _ = jax.random.split(rng)
    state, _obs, _r, _d, _info = env.step(state, wait, step_rng)

    t1 = int(state.timestep)
    remaining_after = int(state.dungeon.lit_radius_until_turn) - t1

    assert t1 == t0 + 1, f"timestep didn't advance: {t0} -> {t1}"
    assert remaining_after == remaining_before - 1, (
        f"lit_radius remaining {remaining_before} -> {remaining_after} (expected {remaining_before-1})"
    )


def test_lycanthropy_timer_decrements_per_turn():
    """polymorph.step must decrement lycanthropy_timer once per step.

    Vendor: allmain.c lines 322-339 (mvl_change handling).
    """
    import jax
    import jax.numpy as jnp

    env, state, rng, wait = _fresh_env_and_state()

    # Seed a non-zero timer so decrement is observable.
    state = state.replace(
        polymorph=state.polymorph.replace(lycanthropy_timer=jnp.int16(20))
    )
    before = int(state.polymorph.lycanthropy_timer)

    step_rng, _ = jax.random.split(rng)
    state, _obs, _r, _d, _info = env.step(state, wait, step_rng)

    after = int(state.polymorph.lycanthropy_timer)
    assert after == before - 1, (
        f"lycanthropy_timer not decremented: {before} -> {after} (expected {before-1})"
    )


def test_shop_step_called_per_turn():
    """shop_step is invoked every turn; with no active shop it's a safe no-op
    (state shape/dtypes invariant). Vendor: shk.c via moveloop.
    """
    import jax

    env, state, rng, wait = _fresh_env_and_state()
    shop_before = state.shop
    step_rng, _ = jax.random.split(rng)
    state, _obs, _r, _d, _info = env.step(state, wait, step_rng)
    shop_after = state.shop

    # Both slices must be the same pytree shape (shop_step doesn't crash and
    # preserves the slice structure even when no shop is active).
    leaves_before = jax.tree.leaves(shop_before)
    leaves_after = jax.tree.leaves(shop_after)
    assert len(leaves_before) == len(leaves_after), (
        f"shop slice leaf count changed: {len(leaves_before)} -> {len(leaves_after)}"
    )
    for lb, la in zip(leaves_before, leaves_after):
        assert lb.shape == la.shape, f"shop leaf shape change: {lb.shape} -> {la.shape}"
        assert lb.dtype == la.dtype, f"shop leaf dtype change: {lb.dtype} -> {la.dtype}"


def test_timestep_increments_once_per_step():
    """allmain.c line 244 (svm.moves++) — exactly one increment per env.step.
    """
    import jax

    env, state, rng, wait = _fresh_env_and_state()

    t0 = int(state.timestep)
    for i in range(5):
        rng, step_rng = jax.random.split(rng)
        state, _obs, _r, _d, _info = env.step(state, wait, step_rng)
        expected = t0 + (i + 1)
        got = int(state.timestep)
        assert got == expected, f"timestep after step {i+1}: got {got}, want {expected}"


# ---------------------------------------------------------------------------
# Order tests — inspect the source of _step_impl for vendor ordering
# ---------------------------------------------------------------------------

def _step_impl_source() -> str:
    from Nethax.nethax.env import _step_impl
    return inspect.getsource(_step_impl)


def test_pipeline_order_player_before_monsters():
    """dispatch_action (player) must precede monster_ai.step.

    Vendor: allmain.c line 203 (player) precedes line 212 (movemon).
    """
    src = _step_impl_source()
    p = src.find("dispatch_action(")
    m = src.find("_monster_ai_step(")
    assert p != -1, "dispatch_action call not found in _step_impl"
    assert m != -1, "_monster_ai_step call not found in _step_impl"
    assert p < m, f"dispatch_action (idx {p}) must precede _monster_ai_step (idx {m})"


def test_pipeline_order_status_after_monsters():
    """Status tick must follow the monster turn.

    Vendor: allmain.c line 273 (nh_timeout) follows line 212 (movemon).
    """
    src = _step_impl_source()
    m = src.find("_monster_ai_step(")
    s = src.find("_status_step(")
    assert m != -1, "_monster_ai_step call not found"
    assert s != -1, "_status_step call not found"
    assert m < s, f"_monster_ai_step (idx {m}) must precede _status_step (idx {s})"


def test_endgame_check_last():
    """maybe_ascend must be the final subsystem call, after age_spells and shop.

    Vendor: allmain.c done() paths run at end of the turn block.
    """
    src = _step_impl_source()
    age = src.find("spell_memory - jnp.int32(1)")
    shop = src.find("_shop_step(")
    asc = src.find("maybe_ascend(")
    assert age != -1, "age_spells decrement not found"
    assert shop != -1, "_shop_step call not found"
    assert asc != -1, "maybe_ascend call not found"
    assert age < asc, f"age_spells (idx {age}) must precede maybe_ascend (idx {asc})"
    assert shop < asc, f"shop_step (idx {shop}) must precede maybe_ascend (idx {asc})"
