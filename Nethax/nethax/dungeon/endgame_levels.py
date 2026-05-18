"""Endgame plane level generators — stub dispatch.

The five endgame-plane level factories live in dungeon/endgame.py (the
Wave-5 Phase-4b deliverable).  This module re-exports them under the
``generate_plane_*`` naming convention requested by the task spec, and
adds the BRANCH_PLANE_* constants that label each plane within the
ENDGAME branch.

Citations:
    vendor/nethack/dat/earth.lua, air.lua, fire.lua, water.lua, astral.lua
    vendor/nethack/src/endgame.c   — plane entry / level ordering
    vendor/nethack/include/dungeon.h — BRANCH_PLANE_EARTH … BRANCH_PLANE_ASTRAL
"""

from __future__ import annotations

from Nethax.nethax.dungeon.endgame import (
    generate_earth_plane,
    generate_air_plane,
    generate_fire_plane,
    generate_water_plane,
    generate_astral_plane,
    generate_endgame_level,
)

# ---------------------------------------------------------------------------
# Branch-plane index constants  (1-based Endgame level numbers)
# Citation: vendor/nethack/include/dungeon.h BRANCH_PLANE_* constants
# ---------------------------------------------------------------------------

BRANCH_PLANE_EARTH:  int = 1   # Endgame Dlvl 1 — Plane of Earth
BRANCH_PLANE_AIR:    int = 2   # Endgame Dlvl 2 — Plane of Air
BRANCH_PLANE_FIRE:   int = 3   # Endgame Dlvl 3 — Plane of Fire
BRANCH_PLANE_WATER:  int = 4   # Endgame Dlvl 4 — Plane of Water
BRANCH_PLANE_ASTRAL: int = 5   # Endgame Dlvl 5 — Astral Plane (ascension)

# ---------------------------------------------------------------------------
# Re-export the generators under the generate_plane_* naming convention.
# The factories themselves live in dungeon/endgame.py; this module is a
# thin alias layer so callers can do:
#
#   from Nethax.nethax.dungeon.endgame_levels import generate_plane_earth_level
# ---------------------------------------------------------------------------

generate_plane_earth_level  = generate_earth_plane
generate_plane_air_level    = generate_air_plane
generate_plane_fire_level   = generate_fire_plane
generate_plane_water_level  = generate_water_plane
generate_plane_astral_level = generate_astral_plane

__all__ = [
    "BRANCH_PLANE_EARTH",
    "BRANCH_PLANE_AIR",
    "BRANCH_PLANE_FIRE",
    "BRANCH_PLANE_WATER",
    "BRANCH_PLANE_ASTRAL",
    "generate_plane_earth_level",
    "generate_plane_air_level",
    "generate_plane_fire_level",
    "generate_plane_water_level",
    "generate_plane_astral_level",
    "generate_endgame_level",
]
