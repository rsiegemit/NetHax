"""Nethax — JAX reimplementation of NetHack (5.0 / 3.7-branch).

Public API:
    EnvState, StaticParams       — master pytree
    NethaxEnv                    — top-level env with NLE-style reset/step
    Action, ACTIONS, N_ACTIONS   — full 119-action NLE-parity enum
    NLE_OBSERVATION_KEYS         — 17-key observation dict contract

See README.md for project scope and current wave status.
"""

# ---------------------------------------------------------------------------
# JAX persistent compilation cache — turn 10-minute cold compiles into
# 10-second warm replays after the first run.  Env vars are read by JAX on
# first import; setting them BEFORE ``import jax`` is the only reliable way
# to enable the cache.  We default to a project-local cache directory but
# honour any caller-supplied JAX_COMPILATION_CACHE_DIR.
#
# JAX docs: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
# The min_compile_time_secs and min_entry_size_bytes defaults (1s / 1KB)
# skip caching for fast/small compiles; we set both to 0 so EVERY traced
# function is cached.  The repeated dungeon-gen JIT (which dominates cold
# compile time) is large but breaks the threshold inconsistently across
# JAX versions, so explicit zero is the safe choice.
# ---------------------------------------------------------------------------
import os as _os
_os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR",
    _os.path.expanduser("~/.cache/nethax_jax"),
)
_os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
_os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.env import NethaxEnv

from Nethax.nethax.constants.actions import Action, ACTIONS, N_ACTIONS, USEFUL_ACTIONS
from Nethax.nethax.obs.nle_obs import NLE_OBSERVATION_KEYS

__all__ = [
    "EnvState",
    "StaticParams",
    "NethaxEnv",
    "Action",
    "ACTIONS",
    "N_ACTIONS",
    "USEFUL_ACTIONS",
    "NLE_OBSERVATION_KEYS",
]
