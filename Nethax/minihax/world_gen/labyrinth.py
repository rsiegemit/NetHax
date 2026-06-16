"""Spiral-maze layouts for the MiniHack Labyrinth envs.

Source of truth: ``vendor/minihack/minihack/envs/lab.py`` — the vendor
``MiniHackLabyrinth`` / ``MiniHackLabyrinthSmall`` classes hand-author an
ASCII spiral maze (concentric rectangles, each with a single gap that
opens the corridor inward toward a center room).  Both maps are
deterministic: vendor does not call ``MAZEWALK`` or any RNG carving — the
ASCII grid IS the level.

This module exposes the exact vendor ASCII grids plus the matching start
and goal coordinates so the Labyrinth builders in
``envs/canonical.py:_labyrinth_builder`` can stamp them via
``LevelGenerator.set_map``.

Coordinate conventions
----------------------
MiniHack uses ``(x, y)`` = (column, row), zero-indexed within the MAP
block.  ``LABYRINTH_BIG_MAP[0]`` is the top wall ``-----...---`` row
(y=0), and the leftmost ``|`` of each interior row is x=0.
"""
from __future__ import annotations

from typing import Tuple


# Vendor source: vendor/minihack/minihack/envs/lab.py:8-30
# 37 columns × 21 rows, concentric-rectangle spiral with single-cell
# wall gaps connecting each ring to the next.
LABYRINTH_BIG_MAP: Tuple[str, ...] = (
    "-------------------------------------",
    "|.................|.|...............|",
    "|.|-------------|.|.|.------------|.|",
    "|.|.............|.|.|.............|.|",
    "|.|.|----------.|.|.|------------.|.|",
    "|.|.|...........|.|.............|.|.|",
    "|.|.|.|----------.|-----------|.|.|.|",
    "|.|.|.|...........|.......|...|.|.|.|",
    "|.|.|.|.|----------------.|.|.|.|.|.|",
    "|.|.|.|.|.................|.|.|.|.|.|",
    "|.|.|.|.|.-----------------.|.|.|.|.|",
    "|.|.|.|.|...................|.|.|.|.|",
    "|.|.|.|.|--------------------.|.|.|.|",
    "|.|.|.|.......................|.|.|.|",
    "|.|.|.|-----------------------|.|.|.|",
    "|.|.|...........................|.|.|",
    "|.|.|---------------------------|.|.|",
    "|.|...............................|.|",
    "|.|-------------------------------|.|",
    "|...................................|",
    "-------------------------------------",
)

# Vendor source: vendor/minihack/minihack/envs/lab.py:47-59
# 20 columns × 11 rows, same spiral pattern scaled down.
LABYRINTH_SMALL_MAP: Tuple[str, ...] = (
    "--------------------",
    "|.......|.|........|",
    "|.-----.|.|.-----|.|",
    "|.|...|.|.|......|.|",
    "|.|.|.|.|.|-----.|.|",
    "|.|.|...|....|.|.|.|",
    "|.|.--------.|.|.|.|",
    "|.|..........|...|.|",
    "|.|--------------|.|",
    "|..................|",
    "--------------------",
)

# Vendor start/goal coords from lab.py:32-33 (big) and lab.py:61-62 (small).
# Both use MiniHack (x, y) = (col, row) convention.
LABYRINTH_BIG_START: Tuple[int, int] = (19, 1)
LABYRINTH_BIG_GOAL: Tuple[int, int] = (19, 7)
LABYRINTH_SMALL_START: Tuple[int, int] = (9, 1)
LABYRINTH_SMALL_GOAL: Tuple[int, int] = (14, 5)


def labyrinth_map(big: bool) -> Tuple[Tuple[str, ...], Tuple[int, int], Tuple[int, int]]:
    """Return ``(map_rows, start_xy, goal_xy)`` for a Labyrinth env.

    ``big=True`` selects MiniHack-Labyrinth-Big-v0 (37×21);
    ``big=False`` selects MiniHack-Labyrinth-Small-v0 (20×11).
    """
    if big:
        return LABYRINTH_BIG_MAP, LABYRINTH_BIG_START, LABYRINTH_BIG_GOAL
    return LABYRINTH_SMALL_MAP, LABYRINTH_SMALL_START, LABYRINTH_SMALL_GOAL


__all__ = [
    "LABYRINTH_BIG_MAP",
    "LABYRINTH_SMALL_MAP",
    "LABYRINTH_BIG_START",
    "LABYRINTH_BIG_GOAL",
    "LABYRINTH_SMALL_START",
    "LABYRINTH_SMALL_GOAL",
    "labyrinth_map",
]
