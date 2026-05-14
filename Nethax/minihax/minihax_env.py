"""``MinihaxEnv`` — high-level wrapper for the canonical MiniHack envs.

Wave 4 Phase 1, agent A4 deliverable.

Drop-in wrapper around :class:`Nethax.nethax.env.NethaxEnv` that exposes each
of the 150+ canonical MiniHack env_ids as a self-contained environment.

Usage
-----
.. code-block:: python

    from Nethax.minihax.minihax_env import MinihaxEnv
    import jax

    env = MinihaxEnv("MiniHack-Room-5x5-v0")
    rng = jax.random.PRNGKey(0)
    state, _ = env.reset(rng)
    state, reward, done, info = env.step(state, action=0, rng=rng)

The ``env_id`` is looked up in
:data:`Nethax.minihax.registry.MINIHACK_ENV_REGISTRY`.  Each entry encodes:

* a ``level_factory`` callable (``rng -> EnvState``),
* a default ``RewardManager`` (matching MiniHack's sparse +1 goal reward),
* the canonical ``max_steps`` budget.

Users can pass a custom :class:`~Nethax.minihax.reward_manager.RewardManager`
to override the default reward shape:

.. code-block:: python

    from Nethax.minihax.reward_manager import RewardManager
    rm = RewardManager()
    rm.add_coordinate_event(4, 4, reward=2.0, terminal_sufficient=True)
    env = MinihaxEnv("MiniHack-Room-5x5-v0", reward_manager=rm)

Notes
-----
``reset`` and ``step`` are Python-side methods so they can dispatch to the
factory closure (which runs once at reset) and to ``NethaxEnv.step`` (which
is fully JIT-friendly).  Building env instances is therefore NOT JIT-able;
calling ``env.step`` IS.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.env import NethaxEnv
from Nethax.minihax.reward_manager import RewardManager
from Nethax.minihax.registry import (
    MINIHACK_ENV_REGISTRY,
    EnvSpec,
    get_spec,
)


class MinihaxEnv:
    """Wrapper that delivers any of the canonical MiniHack environments."""

    def __init__(
        self,
        env_id: str,
        *,
        reward_manager: Optional[RewardManager] = None,
    ) -> None:
        # Lookup raises KeyError on unknown env_id — matches the contract
        # documented in tests/test_minihax_envs.py.
        spec: EnvSpec = get_spec(env_id)
        self._env_id = env_id
        self._spec = spec
        self._level_factory = spec.level_factory
        self._reward_manager = (
            reward_manager if reward_manager is not None else spec.reward_manager
        )
        self._max_steps = int(spec.max_steps)
        # NethaxEnv is reused across reset/step calls; it holds only the
        # StaticParams pytree-shape, so it is safe to share.
        self._engine = NethaxEnv()

    # ------------------------------------------------------------------
    # Properties / inspection
    # ------------------------------------------------------------------
    @property
    def env_id(self) -> str:
        return self._env_id

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    @property
    def reward_manager(self) -> RewardManager:
        return self._reward_manager

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def category(self) -> str:
        return self._spec.category

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> Tuple[EnvState, Dict[str, Any]]:
        """Build a fresh ``EnvState`` for this env.

        Returns ``(state, info)`` where ``info`` carries the initial
        ``fired_mask`` for the configured ``RewardManager`` so callers can
        thread it through subsequent ``step`` calls.
        """
        state = self._level_factory(rng)
        info: Dict[str, Any] = {
            "fired_mask": self._reward_manager.initial_fired_mask(),
            "step_count": 0,
        }
        return state, info

    def step(
        self,
        state: EnvState,
        action: Any,
        rng: jax.Array,
        *,
        fired_mask: Optional[jax.Array] = None,
        step_count: int = 0,
    ) -> Tuple[EnvState, float, bool, Dict[str, Any]]:
        """Apply ``action``, evaluate reward, return updated state.

        Parameters
        ----------
        state : EnvState
            The current game state.
        action : int | jax.Array
            Action index, dispatched to :func:`NethaxEnv.step`.
        rng : jax.Array
            PRNG key for stochastic action effects.
        fired_mask : jax.Array, optional
            Per-event ``fired`` mask returned by a previous ``step``.  If
            ``None``, we build a fresh mask via the reward manager.
        step_count : int
            Current step count for max-steps termination check.

        Returns
        -------
        (new_state, reward, done, info)
            ``info`` carries the updated ``fired_mask`` and ``step_count``.
        """
        if fired_mask is None:
            fired_mask = self._reward_manager.initial_fired_mask()

        new_state, _obs, _engine_reward, _engine_done, _engine_info = (
            self._engine.step(state, action, rng)
        )

        # Evaluate reward shape against the (prev_state, new_state) transition.
        reward, rm_done, new_fired = self._reward_manager.compute_reward(
            state, new_state, fired_mask,
        )

        # Truncation: cap on max_steps.
        new_step_count = int(step_count) + 1
        truncated = new_step_count >= self._max_steps

        # Combine: episode ends if reward-manager says done, or engine done,
        # or we hit max_steps.
        done = bool(rm_done) or bool(_engine_done) or truncated

        info: Dict[str, Any] = {
            "fired_mask": new_fired,
            "step_count": new_step_count,
            "truncated": truncated,
            "engine_done": bool(_engine_done),
            "reward_manager_done": bool(rm_done),
        }
        return new_state, float(reward), done, info

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"MinihaxEnv(env_id={self._env_id!r}, "
            f"category={self._spec.category!r}, "
            f"max_steps={self._max_steps})"
        )


__all__ = ["MinihaxEnv", "MINIHACK_ENV_REGISTRY"]
