"""Tests for :class:`Nethax.minihax.minihax_env.MinihaxEnv`.

Wave 4 Phase 1, agent A4 deliverable.

Verifies:
* The canonical env registry holds at least 30 envs.
* ``MinihaxEnv`` round-trips for a representative env from each category.
* ``reset`` returns a valid ``EnvState`` pytree.
* ``step`` returns ``(state, reward, done, info)``.
* Reaching the goal yields ``reward=1.0`` and ``done=True``.
* Unknown env_id raises ``KeyError``.
* Custom ``RewardManager`` overrides the default.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants.tiles import TileType
from Nethax.minihax.minihax_env import MinihaxEnv
from Nethax.minihax.registry import MINIHACK_ENV_REGISTRY
from Nethax.minihax.reward_manager import RewardManager


def _rng(seed: int = 0) -> jax.Array:
    return jax.random.PRNGKey(seed)


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------
def test_registry_has_at_least_30_envs():
    """The canonical registry must expose at least 30 environments."""
    assert len(MINIHACK_ENV_REGISTRY) >= 30, (
        f"only {len(MINIHACK_ENV_REGISTRY)} envs registered; "
        f"expected at least 30"
    )


def test_registry_covers_all_major_categories():
    """At least one env per major category is registered."""
    categories = {spec.category for spec in MINIHACK_ENV_REGISTRY.values()}
    expected = {
        "Room", "Corridor", "MazeWalk", "HideNSeek", "KeyRoom",
        "LavaCross", "Sokoban", "Labyrinth", "River", "MultiRoom",
        "Quest", "Memento", "WoD", "Boxoban", "Skill",
    }
    missing = expected - categories
    assert not missing, f"missing categories: {missing}"


# ---------------------------------------------------------------------------
# Core env_id round-trips
# ---------------------------------------------------------------------------
def test_minihack_room_5x5_creates_env():
    """``MinihaxEnv("MiniHack-Room-5x5-v0")`` instantiates without error."""
    env = MinihaxEnv("MiniHack-Room-5x5-v0")
    assert env.env_id == "MiniHack-Room-5x5-v0"
    assert env.category == "Room"
    assert env.max_steps == 5 * 20


def test_minihack_room_5x5_reset():
    """``env.reset(rng)`` returns a valid EnvState pytree."""
    env = MinihaxEnv("MiniHack-Room-5x5-v0")
    state, info = env.reset(_rng(0))
    assert isinstance(state, EnvState)
    # Map dimensions are the static defaults; the sub-region is populated.
    assert state.terrain.shape == (7, 32, 21, 80)
    # The fired_mask in info is a bool array.
    assert "fired_mask" in info
    assert info["step_count"] == 0


def test_minihack_room_5x5_step():
    """``env.step`` returns (state, reward, done, info)."""
    env = MinihaxEnv("MiniHack-Room-5x5-v0")
    state, info = env.reset(_rng(1))
    new_state, reward, done, new_info = env.step(
        state, action=0, rng=_rng(2),
        fired_mask=info["fired_mask"],
        step_count=info["step_count"],
    )
    assert isinstance(new_state, EnvState)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "fired_mask" in new_info
    assert new_info["step_count"] == 1


def test_minihack_room_5x5_terminal_on_goal():
    """Standing on the stairs_down tile yields reward=1.0 and done=True."""
    env = MinihaxEnv("MiniHack-Room-5x5-v0")
    state, info = env.reset(_rng(3))

    # Force the player onto the stairs_down tile.  We scan the terrain for
    # the goal cell and teleport the player there.
    sub = state.terrain[0, 0]
    flat = sub.reshape(-1)
    target = jnp.int8(int(TileType.STAIRCASE_DOWN))
    goal_mask = (flat == target)
    if not bool(jnp.any(goal_mask)):
        pytest.skip("no stairs_down in this generated level")
    idx = int(jnp.argmax(goal_mask))
    row = idx // 80
    col = idx % 80

    teleported = state.replace(
        player_pos=jnp.array([row, col], dtype=jnp.int16),
    )

    new_state, reward, done, new_info = env.step(
        teleported, action=10, rng=_rng(4),    # SEARCH = noop-ish
        fired_mask=info["fired_mask"],
        step_count=0,
    )
    assert reward == pytest.approx(1.0)
    assert done is True


# ---------------------------------------------------------------------------
# Other category smoke tests
# ---------------------------------------------------------------------------
def test_minihack_corridor_r2_creates_env():
    env = MinihaxEnv("MiniHack-Corridor-R2-v0")
    assert env.env_id == "MiniHack-Corridor-R2-v0"
    state, _ = env.reset(_rng(5))
    assert isinstance(state, EnvState)


def test_minihack_lavacross_creates_env():
    """Any LavaCross variant should construct + reset."""
    env = MinihaxEnv("MiniHack-LavaCross-Levitate-Potion-Pickup-Full-v0")
    state, _ = env.reset(_rng(6))
    assert isinstance(state, EnvState)


def test_minihack_sokoban_creates_env():
    env = MinihaxEnv("MiniHack-Sokoban1a-v0")
    state, _ = env.reset(_rng(7))
    assert isinstance(state, EnvState)


def test_minihack_multiroom_creates_env():
    env = MinihaxEnv("MiniHack-MultiRoom-N2-v0")
    state, _ = env.reset(_rng(8))
    assert isinstance(state, EnvState)


# ---------------------------------------------------------------------------
# Error handling + customization
# ---------------------------------------------------------------------------
def test_unknown_env_id_raises_keyerror():
    """Unknown env_id triggers a ``KeyError``."""
    with pytest.raises(KeyError):
        MinihaxEnv("MiniHack-Does-Not-Exist-v0")


def test_custom_reward_manager_overrides_default():
    """A user-supplied ``RewardManager`` replaces the canonical sparse one."""
    custom = RewardManager()
    custom.add_coordinate_event(2, 2, reward=42.0, terminal_sufficient=True)

    env = MinihaxEnv("MiniHack-Room-5x5-v0", reward_manager=custom)
    assert env.reward_manager is custom

    state, info = env.reset(_rng(9))
    # Move the player to (2, 2): the custom reward fires.
    teleported = state.replace(
        player_pos=jnp.array([2, 2], dtype=jnp.int16),
    )
    _new_state, reward, done, _info = env.step(
        teleported, action=10, rng=_rng(10),
        fired_mask=info["fired_mask"],
        step_count=0,
    )
    assert reward == pytest.approx(42.0)
    assert done is True
