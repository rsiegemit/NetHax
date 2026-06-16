"""SimpleCrossing N-strip world generator.

Mirrors vendor MiniGrid ``SimpleCrossing`` (vendor/minihack/minihack/envs/
minigrid.py:511-529), which parametrises the level by ``size`` and
``num_crossings`` (the ``N`` in env_ids like ``MiniHack-SimpleCrossingS9N3-v0``).

Layout:
* ``size x size`` open room.
* ``N`` vertical **wall** strips at evenly-spaced interior columns, each
  spanning rows 1..(h-2).
* Each strip has a single open gap so the level is solvable.  The gap
  row is chosen deterministically per strip-index so wall-count
  scales linearly with ``N``:

    walls_per_strip = (h - 2) - 1
    total_walls     = N * walls_per_strip

* Player starts at (0, 0); downstair at ``(w-1, h-1)``.

This module mirrors the analogous LavaCrossing N-strip pattern but emits
``"|"`` (TileType.VWALL) instead of ``"L"`` (TileType.LAVA).

We override the four canonical SimpleCrossing entries via
:func:`register_simplecrossing_envs`, which is invoked at registry
load time (see ``Nethax/minihax/registry.py``).
"""
from __future__ import annotations

import re
from typing import Callable, Tuple

import jax

from Nethax.nethax.state import EnvState
from Nethax.minihax.level_generator import LevelGenerator


# Vendor MiniGrid SimpleCrossing variants (size, num_crossings).
# Cite: vendor/minihack/minihack/envs/minigrid.py:511-529.
SIMPLECROSSING_VARIANTS: Tuple[Tuple[str, int, int, int], ...] = (
    ("MiniHack-SimpleCrossingS9N1-v0",  9,  9, 1),
    ("MiniHack-SimpleCrossingS9N2-v0",  9,  9, 2),
    ("MiniHack-SimpleCrossingS9N3-v0",  9,  9, 3),
    ("MiniHack-SimpleCrossingS11N5-v0", 11, 11, 5),
)


def _strip_columns(w: int, n: int) -> list:
    """Return ``n`` evenly-spaced interior column indices for the wall strips.

    Strips live in columns ``1..w-2`` (interior).  For ``n`` strips we
    place them at ``round((i + 1) * (w - 1) / (n + 1))`` for ``i`` in
    ``range(n)``, then clamp + de-duplicate to keep them strictly
    inside the interior.
    """
    interior_lo, interior_hi = 1, w - 2
    cols = []
    for i in range(n):
        c = int(round((i + 1) * (w - 1) / (n + 1)))
        c = max(interior_lo, min(interior_hi, c))
        cols.append(c)
    # De-duplicate while preserving order; if collisions push us short,
    # fill the next-available interior columns.
    seen = set()
    unique = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    if len(unique) < n:
        for c in range(interior_lo, interior_hi + 1):
            if c not in seen:
                seen.add(c)
                unique.append(c)
                if len(unique) == n:
                    break
    return unique


def _gap_row(h: int, strip_idx: int) -> int:
    """Deterministic gap row for the ``strip_idx``-th strip.

    The vendor MiniGrid generator chooses gap rows randomly per reset,
    but :class:`LevelGenerator` directives are recorded once at build
    time and ``fill_terrain`` takes deterministic rectangles only.
    Varying the gap row by strip index keeps each strip individually
    passable while preserving total wall count.
    """
    # Strips span rows 1..h-2 (inclusive); pick a gap row inside that.
    lo, hi = 1, h - 2
    span = hi - lo + 1
    # Stagger by strip index so adjacent strips don't share a gap row.
    return lo + (strip_idx * 2 + (h // 2 - 1)) % span


def build_simple_crossing(lg: LevelGenerator, w: int, h: int, n: int) -> None:
    """Emit ``n`` vertical wall strips with one gap each into ``lg``.

    Mirrors the LavaCrossing N-strip pattern but uses ``"|"`` (VWALL)
    in place of ``"L"`` (LAVA).
    """
    cols = _strip_columns(w, n)
    for idx, col in enumerate(cols):
        gap = _gap_row(h, idx)
        # Top segment: rows 1..gap-1 (skip if gap == 1).
        if gap > 1:
            lg.fill_terrain("|", col, 1, col, gap - 1)
        # Bottom segment: rows gap+1..h-2 (skip if gap == h-2).
        if gap < h - 2:
            lg.fill_terrain("|", col, gap + 1, col, h - 2)
    lg.set_start_pos(0, 0)
    lg.add_stair_down(x=w - 1, y=h - 1)


def _make_factory(
    builder: Callable[[LevelGenerator], None], w: int, h: int,
) -> Callable[[jax.Array], EnvState]:
    lg = LevelGenerator(w=w, h=h)
    builder(lg)
    return lg.get_factory()


def register_simplecrossing_envs() -> None:
    """Override the canonical SimpleCrossing registry entries with the
    N-strip builders defined above.

    Called from :mod:`Nethax.minihax.registry` after the canonical
    registrar runs, so this strictly replaces the prior single-strip
    placeholders without touching ``envs/canonical.py``.
    """
    # Import lazily so this module stays importable in isolation
    # (e.g. for unit tests that don't want the full registry side-effects).
    from Nethax.minihax.registry import (
        MINIHACK_ENV_REGISTRY, EnvSpec, register,
    )
    from Nethax.minihax.envs.canonical import _default_goal_reward_manager

    for env_id, w, h, n in SIMPLECROSSING_VARIANTS:
        def _builder(lg: LevelGenerator, _w=w, _h=h, _n=n) -> None:
            build_simple_crossing(lg, _w, _h, _n)
        factory = _make_factory(_builder, w=w, h=h)
        prev = MINIHACK_ENV_REGISTRY.get(env_id)
        max_steps = prev.max_steps if prev is not None else w * h
        category = prev.category if prev is not None else "Crossing"
        register(EnvSpec(
            env_id=env_id,
            level_factory=factory,
            reward_manager=_default_goal_reward_manager(),
            max_steps=max_steps,
            category=category,
        ))


__all__ = [
    "SIMPLECROSSING_VARIANTS",
    "build_simple_crossing",
    "register_simplecrossing_envs",
]
