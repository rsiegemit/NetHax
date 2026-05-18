"""Swallowed-player parity tests.

Canonical source: vendor/nethack/src/mhitu.c::gulpmu (lines 1287-1434)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.subsystems.swallow import (
    SwallowState,
    try_engulf,
    digest_tick,
    release_from_engulf,
    _IS_ENGULFER,
)

# Purple worm MONSTERS index (has AT_ENGL + AD_DGST) — verified above.
PURPLE_WORM_IDX: int = 114


def _make_state(rng=None):
    """Return a fresh EnvState with one purple worm in slot 0."""
    if rng is None:
        rng = jax.random.PRNGKey(42)
    state = EnvState.default(rng=rng)
    # Place a live purple worm in monster slot 0.
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        entry_idx=mai.entry_idx.at[0].set(jnp.int16(PURPLE_WORM_IDX)),
        hp=mai.hp.at[0].set(jnp.int32(60)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(60)),
        pos=mai.pos.at[0].set(jnp.array([5, 5], dtype=jnp.int16)),
    )
    state = state.replace(monster_ai=mai)
    return state


# ---------------------------------------------------------------------------
# test_initial_state_not_swallowed
# ---------------------------------------------------------------------------
def test_initial_state_not_swallowed():
    """env.reset → swallow.swallowed == False."""
    state = _make_state()
    assert bool(state.swallow.swallowed) is False
    assert int(state.swallow.engulfer_slot) == -1
    assert int(state.swallow.digest_timer) == 0
    assert int(state.swallow.total_timer) == 0


# ---------------------------------------------------------------------------
# test_engulf_attack_swallows
# ---------------------------------------------------------------------------
def test_engulf_attack_swallows():
    """Calling try_engulf with a purple worm slot sets swallowed=True."""
    state = _make_state()
    rng = jax.random.PRNGKey(0)

    # Confirm purple worm is marked as an engulfer in the table.
    assert bool(_IS_ENGULFER[PURPLE_WORM_IDX]) is True

    new_state = try_engulf(state, jnp.int32(0), rng)

    assert bool(new_state.swallow.swallowed) is True
    assert int(new_state.swallow.engulfer_slot) == 0
    assert int(new_state.swallow.digest_timer) == 10
    assert int(new_state.swallow.total_timer) >= 26  # 25 + rnd(75) >= 26

    # Player position must now equal the engulfer's position.
    assert tuple(new_state.player_pos.tolist()) == (5, 5)


# ---------------------------------------------------------------------------
# test_digest_tick_damages
# ---------------------------------------------------------------------------
def test_digest_tick_damages():
    """After 10 digest_ticks, player HP decreases (digestion damage fires)."""
    state = _make_state()
    rng = jax.random.PRNGKey(1)
    state = try_engulf(state, jnp.int32(0), rng)
    initial_hp = int(state.player_hp)

    # Tick 10 times — digest_timer starts at 10, fires when it hits 0.
    for i in range(10):
        rng, sub = jax.random.split(rng)
        state = digest_tick(state, sub)

    assert int(state.player_hp) < initial_hp, (
        f"Expected HP to decrease after 10 ticks, got {state.player_hp} == {initial_hp}"
    )


# ---------------------------------------------------------------------------
# test_release_on_engulfer_death
# ---------------------------------------------------------------------------
def test_release_on_engulfer_death():
    """Setting the engulfer's alive=False then ticking once releases the player."""
    state = _make_state()
    rng = jax.random.PRNGKey(2)
    state = try_engulf(state, jnp.int32(0), rng)
    assert bool(state.swallow.swallowed) is True

    # Kill the engulfer.
    mai = state.monster_ai.replace(
        alive=state.monster_ai.alive.at[0].set(False),
    )
    state = state.replace(monster_ai=mai)

    rng, sub = jax.random.split(rng)
    state = digest_tick(state, sub)

    assert bool(state.swallow.swallowed) is False
    assert int(state.swallow.engulfer_slot) == -1


# ---------------------------------------------------------------------------
# test_movement_blocked_while_swallowed
# ---------------------------------------------------------------------------
def test_movement_blocked_while_swallowed():
    """While swallowed, movement actions are no-ops (player_pos unchanged)."""
    from Nethax.nethax.subsystems.action_dispatch import _try_step

    state = _make_state()
    rng = jax.random.PRNGKey(3)
    state = try_engulf(state, jnp.int32(0), rng)
    pos_before = tuple(state.player_pos.tolist())

    rng, sub = jax.random.split(rng)
    new_state = _try_step(state, -1, 0, sub)  # try to move north

    assert tuple(new_state.player_pos.tolist()) == pos_before, (
        f"Expected pos unchanged {pos_before}, got {new_state.player_pos.tolist()}"
    )


# ---------------------------------------------------------------------------
# test_total_timer_release
# ---------------------------------------------------------------------------
def test_total_timer_release():
    """When total_timer == 1, one digest_tick releases the player."""
    state = _make_state()
    rng = jax.random.PRNGKey(4)
    state = try_engulf(state, jnp.int32(0), rng)

    # Force total_timer to 1 so next tick triggers release.
    state = state.replace(
        swallow=state.swallow.replace(total_timer=jnp.int32(1))
    )

    rng, sub = jax.random.split(rng)
    state = digest_tick(state, sub)

    assert bool(state.swallow.swallowed) is False, (
        "Expected release after total_timer expired"
    )
    assert int(state.swallow.engulfer_slot) == -1
