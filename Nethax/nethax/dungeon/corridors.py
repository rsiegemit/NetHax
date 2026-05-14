"""Corridor generation and door placement.

Purpose:
    Connects rooms with L-shaped corridors and places doors at
    room/corridor junctions.  Mirrors the corridor logic in mklev.c and the
    door-placement logic in mklev.c / mkmap.c.

Citation:
    vendor/nethack/src/mklev.c  — doconnect(), makedog(), add_door(),
        dig_corridor(); L-shaped corridor with random bend point.
    vendor/nethack/src/mkmap.c  — coordinate allocation, door placement.

Wave 2: connect_segments carves an L-shaped corridor between two rooms
        onto a terrain array; place_doors stamps CLOSED_DOOR tiles at
        the room-wall/corridor boundary with ~50% probability.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax

from Nethax.nethax.dungeon.rooms import Room

# ---------------------------------------------------------------------------
# Tile type constants (from constants.py TileType)
# ---------------------------------------------------------------------------

_TILE_VOID:        int = 0
_TILE_FLOOR:       int = 1
_TILE_CORRIDOR:    int = 2
_TILE_WALL:        int = 3
_TILE_CLOSED_DOOR: int = 4

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A corridor segment is represented as int16[N, 4] where each row is
# (y1, x1, y2, x2) — the two endpoints of a straight horizontal or vertical
# run.  L-shaped corridors decompose into two such segments sharing a bend.
CorridorSegments = jnp.ndarray  # int16[N, 4]

# A door entry is int16[M, 3]: (y, x, door_state) where door_state is
# 0=open, 1=closed, 2=locked, 3=secret (matching NLE glyph categories).
DoorArray = jnp.ndarray  # int16[M, 3]

# Fixed capacity for door arrays per level (2 per corridor = 2*(MAX_ROOMS-1))
_MAX_DOORS: int = 80


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def connect_segments(
    rng: jnp.ndarray,
    room_a: Room,
    room_b: Room,
    terrain: jnp.ndarray,
    index: int = 0,
) -> jnp.ndarray:
    """Carve a corridor from room_a to room_b, orthogonal-first then L-bend.

    Vendor (`sp_lev.c::dig_corridor` lines 2571-2660): the digger walks
    toward the target one step at a time, only turning when forced to.
    When both endpoints share a row or column, a single straight run
    suffices; otherwise an L-bend is required.  We follow the same
    priority: straight when possible, L-shape when not.

    Corridor width is exactly one tile (vendor: a single `levl[xx][yy].typ
    = ftyp` per step).  Diagonals are NOT supported — corridor cells are
    placed only on cardinal-aligned segments.

    Args:
        rng:     JAX PRNG key (unused here; vendor bend is deterministic).
        room_a:  source Room (scalar Room — a slice from the rooms array).
        room_b:  destination Room.
        terrain: int8[MAP_H, MAP_W] to carve into.
        index:   slot index (unused; kept for call-site consistency).

    Returns:
        Updated terrain int8[MAP_H, MAP_W].
    """
    h, w = terrain.shape
    rows = jnp.arange(h, dtype=jnp.int32)
    cols = jnp.arange(w, dtype=jnp.int32)

    y_a = ((room_a.y1 + room_a.y2) // 2).astype(jnp.int32)
    x_a = ((room_a.x1 + room_a.x2) // 2).astype(jnp.int32)
    y_b = ((room_b.y1 + room_b.y2) // 2).astype(jnp.int32)
    x_b = ((room_b.x1 + room_b.x2) // 2).astype(jnp.int32)

    same_row = y_a == y_b
    same_col = x_a == x_b

    # Horizontal leg: row y_a, columns min(x_a,x_b)..max(x_a,x_b)
    horiz_row = rows == y_a
    horiz_col = (cols >= jnp.minimum(x_a, x_b)) & (cols <= jnp.maximum(x_a, x_b))
    horiz = horiz_row[:, None] & horiz_col[None, :]

    # Vertical leg: col x_b, rows min(y_a,y_b)..max(y_a,y_b)
    vert_col = cols == x_b
    vert_row = (rows >= jnp.minimum(y_a, y_b)) & (rows <= jnp.maximum(y_a, y_b))
    vert = vert_row[:, None] & vert_col[None, :]

    # Straight: use whichever single leg lines the rooms up.
    straight = jnp.where(same_row, horiz, vert)
    # L-bend fallback.
    l_shape = horiz | vert
    corridor = jnp.where(same_row | same_col, straight, l_shape)

    is_floor = terrain == jnp.int8(_TILE_FLOOR)
    terrain_out = jnp.where(corridor & ~is_floor, jnp.int8(_TILE_CORRIDOR), terrain)
    return terrain_out


def place_doors(
    rng: jnp.ndarray,
    terrain: jnp.ndarray,
    rooms: Room,
    active: jnp.ndarray,
) -> jnp.ndarray:
    """Place CLOSED_DOOR tiles at room/corridor boundaries.

    For each active room, scan the perimeter (the ring one cell outside the
    interior bounding box).  Any perimeter cell that is currently CORRIDOR
    becomes a CLOSED_DOOR with probability ~0.5.

    Citation: vendor/nethack/src/mklev.c add_door(), doconnect().

    Args:
        rng:     JAX PRNG key.
        terrain: int8[MAP_H, MAP_W] with rooms + corridors already carved.
        rooms:   Room pytree from generate_rooms().
        active:  bool[MAX_ROOMS_PER_LEVEL] mask.

    Returns:
        Updated terrain int8[MAP_H, MAP_W] with doors stamped.
    """
    from Nethax.nethax.dungeon.rooms import MAX_ROOMS_PER_LEVEL
    h, w = terrain.shape

    # One door-flip coin per room perimeter cell — pre-sample a full h×w mask.
    rng, key_door = jax.random.split(rng)
    door_coin = jax.random.bernoulli(key_door, 0.5, shape=(h, w))

    rows = jnp.arange(h, dtype=jnp.int32)
    cols = jnp.arange(w, dtype=jnp.int32)

    def stamp_doors_for_room(terrain_, i):
        y1 = rooms.y1[i].astype(jnp.int32)
        x1 = rooms.x1[i].astype(jnp.int32)
        y2 = rooms.y2[i].astype(jnp.int32)
        x2 = rooms.x2[i].astype(jnp.int32)
        act = active[i]

        # Perimeter = wall ring just outside the interior.
        row_border = (rows >= y1 - 1) & (rows <= y2 + 1)
        col_border = (cols >= x1 - 1) & (cols <= x2 + 1)
        row_inner  = (rows >= y1)     & (rows <= y2)
        col_inner  = (cols >= x1)     & (cols <= x2)
        perimeter  = (row_border[:, None] & col_border[None, :]) & \
                     ~(row_inner[:, None] & col_inner[None, :])

        is_corridor = terrain_ == jnp.int8(_TILE_CORRIDOR)
        place = act & perimeter & is_corridor & door_coin
        terrain_new = jnp.where(place, jnp.int8(_TILE_CLOSED_DOOR), terrain_)
        return terrain_new, None

    terrain_out, _ = lax.scan(
        stamp_doors_for_room,
        terrain,
        jnp.arange(MAX_ROOMS_PER_LEVEL, dtype=jnp.int32),
    )
    return terrain_out


# ---------------------------------------------------------------------------
# TODO blocks
# ---------------------------------------------------------------------------
# Wave 4:
#   - Secret corridors: some segments should be hidden (TILE_WALL until
#     searched); controlled by dungeon depth and level flags.
#   - Trapdoors / holes between levels placed in corridors (mklev.c).
#   - Vault corridors: short single-segment corridors to vault rooms
#     (vault.c).
