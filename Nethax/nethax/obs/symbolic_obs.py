"""Symbolic (flat-vector) observation builder for nethax.

Produces a compact fixed-size vector suitable for MLP-based RL baselines.
Avoids the overhead of rendering glyphs or ASCII; encodes structured game
state directly.

Canonical reference:
  - vendor/nle/nle/nethack/nethack.py  (for NLE parity context)

Wave 1 status: returns a zero placeholder of the target size. The vector
dimensionality (1024) is a provisional reservation; Wave 2 will fix it to
the actual concatenated feature count once projection is implemented.
"""

# TODO Wave 2: project map one-hot (tile type per cell, 21*79 cells) +
#              player position + monster locations + blstats vector into a
#              fixed-size vector. Pin the final dimension once layout is known.

import jax.numpy as jnp

# Provisional output dimension — will be tightened in Wave 2 once the full
# feature layout is defined.
SYMBOLIC_OBS_DIM: int = 1024


def build_symbolic_observation(env_state) -> jnp.ndarray:
    """Build a flat symbolic observation vector from nethax EnvState.

    Wave 1: returns jnp.zeros((SYMBOLIC_OBS_DIM,), float32) as a placeholder.

    Wave 2 will concatenate:
      - one-hot tile encoding for the 21x79 map
      - player (row, col) position
      - visible monster locations
      - blstats features (27 values)
    into a fixed-size float32 vector.

    Args:
        env_state: nethax EnvState (unused in Wave 1).

    Returns:
        jnp.ndarray of shape (SYMBOLIC_OBS_DIM,) float32.
    """
    return jnp.zeros((SYMBOLIC_OBS_DIM,), dtype=jnp.float32)
