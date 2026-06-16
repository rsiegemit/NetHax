"""Canonical MiniHack environment registry.

Wave 4 Phase 1, agent A4 deliverable.

Maps each canonical MiniHack ``env_id`` (e.g. ``MiniHack-Room-5x5-v0``) to an
``EnvSpec`` describing how to produce the level and shape the reward.

Each spec carries:
* ``level_factory``: a callable ``(rng) -> EnvState`` that builds the initial
  level.  Built once at registration time via ``LevelGenerator``.
* ``reward_manager``: a ``RewardManager`` instance defining the reward shape.
* ``max_steps``: the canonical step budget from vendor MiniHack.
* ``category``: human-readable category (e.g. ``"Room"``).

Building factories at module-import time means env construction is pure
Python; ``MinihaxEnv.step`` itself remains JIT-friendly because it delegates
to ``NethaxEnv.step``.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Optional

import jax

from Nethax.nethax.state import EnvState
from Nethax.minihax.reward_manager import RewardManager


# ---------------------------------------------------------------------------
# Public spec dataclass
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class EnvSpec:
    """Static description of one canonical MiniHack env.

    Wave17i: extended with ``reward_win`` / ``reward_lose`` / ``penalty_step``
    / ``penalty_time`` fields matching vendor MiniHack base.py:142-167.
    Defaults preserve previous behaviour (win=1, lose=0, no per-step penalty).
    """
    env_id: str
    level_factory: Callable[[jax.Array], EnvState]
    reward_manager: RewardManager
    max_steps: int
    category: str
    reward_win: float = 1.0
    reward_lose: float = 0.0
    penalty_step: float = 0.0
    penalty_time: float = 0.0


# ---------------------------------------------------------------------------
# Registry — populated by ``canonical.register_all``
# ---------------------------------------------------------------------------
MINIHACK_ENV_REGISTRY: Dict[str, EnvSpec] = {}


def register(spec: EnvSpec) -> None:
    """Insert a spec into the global registry, replacing any existing entry."""
    MINIHACK_ENV_REGISTRY[spec.env_id] = spec


def get_spec(env_id: str) -> EnvSpec:
    """Look up a spec; raises ``KeyError`` for unknown ids."""
    if env_id not in MINIHACK_ENV_REGISTRY:
        raise KeyError(f"Unknown MiniHack env_id: {env_id!r}")
    return MINIHACK_ENV_REGISTRY[env_id]


def list_envs(category: Optional[str] = None) -> list:
    """Return env_ids, optionally filtered by category."""
    if category is None:
        return sorted(MINIHACK_ENV_REGISTRY.keys())
    return sorted(
        env_id
        for env_id, spec in MINIHACK_ENV_REGISTRY.items()
        if spec.category == category
    )


# Populate the registry on import.
from Nethax.minihax.envs import canonical as _canonical  # noqa: E402

_canonical.register_all()

# Override SimpleCrossing entries with the N-strip builders that honour
# the ``N`` parameter in env_ids like ``MiniHack-SimpleCrossingS9N3-v0``.
# Cite: vendor/minihack/minihack/envs/minigrid.py:511-529.
from Nethax.minihax.world_gen.simple_crossing import (  # noqa: E402
    register_simplecrossing_envs as _register_simplecrossing_nstrip,
)

_register_simplecrossing_nstrip()
