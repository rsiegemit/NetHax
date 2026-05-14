"""Tests for ``Nethax.minihax.reward_manager.RewardManager``."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants.tiles import TileType
from Nethax.minihax.reward_manager import RewardManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fresh_state(seed: int = 0) -> EnvState:
    return EnvState.default(rng=jax.random.PRNGKey(seed), static=StaticParams())


def _move_player(state: EnvState, row: int, col: int) -> EnvState:
    """Return a new state with the player at (row, col)."""
    return state.replace(
        player_pos=jnp.array([row, col], dtype=jnp.int16),
    )


def _set_tile_under(state: EnvState, tile: TileType) -> EnvState:
    """Set the terrain tile at the player's current position."""
    branch = int(state.dungeon.current_branch)
    level = int(state.dungeon.current_level) - 1
    row = int(state.player_pos[0])
    col = int(state.player_pos[1])
    new_terrain = state.terrain.at[branch, level, row, col].set(int(tile))
    return state.replace(terrain=new_terrain)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_empty_manager_returns_zero():
    """A manager with no events yields zero reward and done=False."""
    rm = RewardManager()
    s = _fresh_state()
    fired = rm.initial_fired_mask()
    reward, done, new_fired = rm.compute_reward(s, s, fired)
    assert float(reward) == 0.0
    assert bool(done) is False
    assert bool(jnp.any(new_fired)) is False


def test_positional_event_fires_on_match():
    """Stepping onto the target tile fires the event exactly once."""
    rm = RewardManager()
    rm.add_positional_event((5, 5), reward=1.5, terminal_sufficient=True)
    fired = rm.initial_fired_mask()

    prev = _fresh_state()
    new = _move_player(prev, 5, 5)
    reward, done, new_fired = rm.compute_reward(prev, new, fired)
    assert float(reward) == pytest.approx(1.5)
    assert bool(done) is True
    assert bool(new_fired[0]) is True


def test_positional_event_not_repeatable_only_once():
    """Second visit yields no reward when repeatable=False."""
    rm = RewardManager()
    rm.add_positional_event((5, 5), reward=1.0, repeatable=False,
                              terminal_required=False,
                              terminal_sufficient=False)
    fired = rm.initial_fired_mask()

    prev = _fresh_state()
    on = _move_player(prev, 5, 5)
    r1, _, fired = rm.compute_reward(prev, on, fired)
    assert float(r1) == 1.0
    assert bool(fired[0]) is True

    # Move away and come back.
    off = _move_player(on, 3, 3)
    r2, _, fired = rm.compute_reward(on, off, fired)
    assert float(r2) == 0.0

    on2 = _move_player(off, 5, 5)
    r3, _, fired = rm.compute_reward(off, on2, fired)
    assert float(r3) == 0.0


def test_positional_event_repeatable_fires_every_visit():
    """Repeatable events fire every time the predicate is True."""
    rm = RewardManager()
    rm.add_positional_event((5, 5), reward=2.0, repeatable=True,
                              terminal_required=False,
                              terminal_sufficient=False)
    fired = rm.initial_fired_mask()

    prev = _fresh_state()
    on = _move_player(prev, 5, 5)
    r1, _, fired = rm.compute_reward(prev, on, fired)
    assert float(r1) == 2.0
    # mask still False because repeatable.
    assert bool(fired[0]) is False

    off = _move_player(on, 0, 0)
    r2, _, fired = rm.compute_reward(on, off, fired)
    assert float(r2) == 0.0

    on2 = _move_player(off, 5, 5)
    r3, _, fired = rm.compute_reward(off, on2, fired)
    assert float(r3) == 2.0  # fires again


def test_terminal_sufficient_event_sets_done():
    """A terminal_sufficient event alone is enough to end the episode."""
    rm = RewardManager()
    rm.add_positional_event((1, 1), terminal_required=False,
                              terminal_sufficient=True)
    fired = rm.initial_fired_mask()

    prev = _fresh_state()
    new = _move_player(prev, 1, 1)
    _, done, _ = rm.compute_reward(prev, new, fired)
    assert bool(done) is True


def test_terminal_required_and_terminal_sufficient_combination():
    """`done` only when EITHER sufficient fires OR all required fire."""
    rm = RewardManager()
    # Two required events; both must fire.
    rm.add_positional_event((1, 1), terminal_required=True,
                              terminal_sufficient=False)
    rm.add_positional_event((2, 2), terminal_required=True,
                              terminal_sufficient=False)
    fired = rm.initial_fired_mask()

    prev = _fresh_state()
    s1 = _move_player(prev, 1, 1)
    r, done, fired = rm.compute_reward(prev, s1, fired)
    assert float(r) == 1.0
    assert bool(done) is False  # second required event still pending

    s2 = _move_player(s1, 2, 2)
    r, done, fired = rm.compute_reward(s1, s2, fired)
    assert float(r) == 1.0
    assert bool(done) is True  # all required satisfied


def test_coordinate_event_fires_on_match():
    """`add_coordinate_event(x, y)` matches when player_pos == (y, x)."""
    rm = RewardManager()
    rm.add_coordinate_event(x=3, y=7, reward=4.0, terminal_sufficient=True)
    fired = rm.initial_fired_mask()

    prev = _fresh_state()
    # x=3, y=7 => row=7, col=3
    new = _move_player(prev, 7, 3)
    r, done, _ = rm.compute_reward(prev, new, fired)
    assert float(r) == 4.0
    assert bool(done) is True


def test_location_event_fires_on_stairs_down():
    """Stepping onto a STAIRCASE_DOWN tile fires the location event."""
    rm = RewardManager()
    rm.add_location_event("stairs_down", reward=1.0, terminal_sufficient=True)
    fired = rm.initial_fired_mask()

    prev = _fresh_state()
    moved = _move_player(prev, 4, 4)
    on_stairs = _set_tile_under(moved, TileType.STAIRCASE_DOWN)
    r, done, _ = rm.compute_reward(prev, on_stairs, fired)
    assert float(r) == 1.0
    assert bool(done) is True


def test_custom_reward_fn():
    """`add_custom_reward_fn` pays the value returned by the callable."""
    def shaping(prev: EnvState, new: EnvState) -> jnp.ndarray:
        # Pay HP-recovered.
        return (new.player_hp - prev.player_hp).astype(jnp.float32)

    rm = RewardManager()
    rm.add_custom_reward_fn(shaping)
    fired = rm.initial_fired_mask()

    prev = _fresh_state().replace(player_hp=jnp.int32(5))
    new = prev.replace(player_hp=jnp.int32(8))
    r, _, _ = rm.compute_reward(prev, new, fired)
    assert float(r) == 3.0

    # If no change, no reward.
    same, _, _ = rm.compute_reward(new, new, fired)
    assert float(same) == 0.0


def test_chained_events_sum():
    """Multiple events firing on the same step sum their rewards."""
    rm = RewardManager()
    rm.add_positional_event((5, 5), reward=1.0, terminal_required=False)
    rm.add_coordinate_event(x=5, y=5, reward=2.0, terminal_required=False)
    fired = rm.initial_fired_mask()

    prev = _fresh_state()
    new = _move_player(prev, 5, 5)
    r, _, _ = rm.compute_reward(prev, new, fired)
    # Both fire (row=5,col=5 matches positional AND coord(x=5,y=5)).
    assert float(r) == 3.0


def test_jit_compute_reward():
    """`compute_reward` must run under jax.jit."""
    rm = RewardManager()
    rm.add_positional_event((5, 5), reward=1.0, terminal_sufficient=True)
    fired = rm.initial_fired_mask()

    @jax.jit
    def step(prev, new, fired):
        return rm.compute_reward(prev, new, fired)

    prev = _fresh_state()
    new = _move_player(prev, 5, 5)
    r, done, _ = step(prev, new, fired)
    assert float(r) == 1.0
    assert bool(done) is True


def test_kill_event_registers_but_does_not_fire():
    """Kill events stub out as no-ops pending Wave 5 message wiring."""
    rm = RewardManager()
    rm.add_kill_event("newt", reward=10.0, terminal_sufficient=True)
    assert len(rm) == 1
    assert rm.events[0].implemented is False

    fired = rm.initial_fired_mask()
    prev = _fresh_state()
    r, done, _ = rm.compute_reward(prev, prev, fired)
    assert float(r) == 0.0
    # Stub event has terminal_required=True (default) and never fires,
    # so done must stay False.
    assert bool(done) is False


def test_eat_event_registers_but_does_not_fire():
    """Eat events stub out as no-ops pending Wave 5 message wiring."""
    rm = RewardManager()
    rm.add_eat_event("apple", reward=5.0)
    assert rm.events[0].implemented is False

    fired = rm.initial_fired_mask()
    prev = _fresh_state()
    r, done, _ = rm.compute_reward(prev, prev, fired)
    assert float(r) == 0.0
    assert bool(done) is False
