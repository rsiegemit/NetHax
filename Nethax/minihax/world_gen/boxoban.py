"""Boxoban dataset infrastructure for MiniHax.

This module provides the data-access layer for Boxoban-style fixed-corpus
puzzle levels (the MiniHack `MiniHack-Boxoban-*` family). In MiniHack, Boxoban
levels are not bundled with the source tree — they are pulled at runtime by
``vendor/minihack/minihack/scripts/download_boxoban_levels.py`` from the
DeepMind Boxoban release. That fetch is intentionally out of scope here.

What is here:
    - ``_LEVELS``: a tuple of ASCII level strings. Seeded with ONE small
      placeholder level (8x8 room with two boulders, a pit, and a downstair)
      derived from the top-left room of vendor ``soko1a.des``. This is enough
      to wire callers and tests against a real interface without a network
      dependency.
    - ``boxoban_map(idx)``: deterministic accessor returning the level rows
      for ``idx`` (mod ``len(_LEVELS)``) as a ``tuple[str, ...]``.
    - ``random_boxoban_idx(rng)``: JAX RNG-driven random index, suitable for
      use inside a jitted reset path.

Followup work (NOT done here):
    - Run ``download_boxoban_levels.py`` (or vendor a snapshot) to obtain the
      full DeepMind Boxoban corpus (~900k levels across unfiltered/medium/
      hard splits).
    - Parse those text files (10x10 grids with ``# $ . @`` glyphs) into the
      MiniHack tile vocabulary used by ``CHAR_TO_TILE`` in ``sokoban.py`` and
      populate ``_LEVELS`` (or a sharded equivalent) from the corpus.
    - Decide whether levels should live in-memory as Python strings or be
      packed into a ``jnp.ndarray`` once at import time, and wire that into
      ``make_boxoban`` analogous to ``make_soko1a`` in ``sokoban.py``.
"""
import jax
import jax.numpy as jnp


# Placeholder corpus. One 8x8 level lifted from the top-left room of
# vendor/minihack/minihack/dat/soko1a.des: walled room, two boulders on the
# upper rows, a pit on row 6, and a downstair ('>') as the goal tile.
# Glyphs follow the MiniHack/.des convention used by sokoban.py:
#   '-' horizontal wall, '|' vertical wall, '.' floor, '`' boulder,
#   '^' pit (trap), '>' downstair.
_LEVELS: tuple[tuple[str, ...], ...] = (
    (
        "--------",
        "|......|",
        "|.``...|",
        "|......|",
        "|......|",
        "|...^..|",
        "|.....>|",
        "--------",
    ),
)


def boxoban_map(idx: int) -> tuple[str, ...]:
    """Return the level rows for ``idx`` (mod ``len(_LEVELS)``).

    Args:
        idx: integer level index. Wraps via modulo so callers can pass any
            non-negative int without bounds-checking.

    Returns:
        Tuple of equal-length ASCII row strings for the selected level.
    """
    if not _LEVELS:
        raise RuntimeError("Boxoban corpus is empty; populate _LEVELS first.")
    return _LEVELS[idx % len(_LEVELS)]


def random_boxoban_idx(rng) -> jnp.ndarray:
    """Sample a uniform random level index using a JAX PRNG key.

    Args:
        rng: a ``jax.random`` PRNG key.

    Returns:
        Scalar ``int32`` ``jnp.ndarray`` in ``[0, len(_LEVELS))``.
    """
    return jax.random.randint(rng, (), 0, len(_LEVELS), dtype=jnp.int32)
