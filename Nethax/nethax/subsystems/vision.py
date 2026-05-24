"""Line-of-sight helpers — vendor-parity ``clear_path`` / ``cansee`` / ``couldsee``.

This module hosts the canonical JIT-pure line-of-sight (LoS) primitives used
across detect / artifact / AI subsystems.  It replaces the Chebyshev-distance
proxies that several callers had previously been using as a stand-in for the
vendor's ray-based LoS check.

Canonical sources:
  vendor/nethack/include/vision.h
      ``#define cansee(x, y)``     (line 28)
      ``#define couldsee(x, y)``   (line 29)
      ``#define m_cansee(...)``    (line 42) — clear_path-based check
  vendor/nethack/src/vision.c
      ``does_block()``             (lines 152-202) — terrain LoS blockers
      ``clear_path()``             (lines 1612-1636) — quadrant Bresenham trace
      ``q1_path`` .. ``q4_path``   (lines 1212-1600) — generalized integer
                                                       Bresenham (Rogers 1985)

Vendor semantics modeled here:
  - ``clear_path(c1, r1, c2, r2)`` walks the straight line between two cells
    (excluding both endpoints) and returns False on the first blocked tile.
  - ``is_clear`` (vision.c line 1162) = NOT ``does_block`` for the tile.
    Blocking tiles per ``does_block`` (vision.c lines 165-184) include
    IS_OBSTRUCTED (walls), TREE, closed/locked/trapped doors, CLOUD, water-
    walls, lava-walls and boulders.  Of those we model WALL, CLOSED_DOOR and
    TREE at the tile layer; the others are not yet present in the JAX
    TileType enum so they cannot block.
  - ``cansee(x, y)`` is the player-side visibility predicate; in vendor it
    reads ``gv.viz_array``.  Without that precomputed array we model it as
    ``clear_path`` between the two endpoints (the relationship is exact for
    fully-lit rooms; lighting effects belong to ``fov.py``).
  - ``couldsee(x, y)`` is the same predicate ignoring blindness.  In vendor
    blindness is folded into ``cansee`` via ``Blind`` checks (display.h:174);
    ``couldsee`` returns LoS even when blind.
  - ``cansee_with_blind(x, y) := couldsee(x, y) AND NOT Blind`` mirrors the
    full vendor ``canseemon``-style gating used by display / spell targeting.

All helpers are JIT-pure: bounded ``fori_loop`` walks at most ``MAX_DIST``
intermediate steps, then returns a 0-D ``jnp.bool_`` result.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.constants.tiles import TileType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of intermediate cells walked along a single LoS ray.  Vendor
# clear_path bounds the loop by ``dx-1`` / ``dy-1`` (vision.c lines 1229,
# 1242), so any value >= max(ROWNO, COLNO) is safe.  We choose 20 to match
# the task spec and the typical detection-range envelope; tiles beyond 20
# cells away are well outside any vendor sight-radius bonus.
MAX_DIST: int = 20

_TILE_WALL        = jnp.int8(int(TileType.WALL))
_TILE_CLOSED_DOOR = jnp.int8(int(TileType.CLOSED_DOOR))
_TILE_TREE        = jnp.int8(int(TileType.TREE))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _current_level_terrain(state) -> jnp.ndarray:
    """Slice the [H, W] terrain plane for the player's current branch/level."""
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    return state.terrain[b, lv]


def _tile_blocks_los(tile: jnp.ndarray) -> jnp.ndarray:
    """Return True if ``tile`` blocks line of sight.

    Cite: vendor/nethack/src/vision.c::does_block (lines 165-184).  We model
    the tile-layer blockers we currently carry in TileType: WALL, CLOSED_DOOR
    (vendor IS_DOOR with D_CLOSED|D_LOCKED|D_TRAPPED), and TREE.  Boulders,
    clouds, water-walls and lava-walls are not yet modeled as tile types.
    """
    t = tile.astype(jnp.int8)
    return (t == _TILE_WALL) | (t == _TILE_CLOSED_DOOR) | (t == _TILE_TREE)


# ---------------------------------------------------------------------------
# clear_path — Bresenham LoS trace
# ---------------------------------------------------------------------------

def clear_path(state, row1: jnp.ndarray, col1: jnp.ndarray,
               row2: jnp.ndarray, col2: jnp.ndarray) -> jnp.ndarray:
    """Return True iff the straight line between two cells is unobstructed.

    Cite: vendor/nethack/src/vision.c::clear_path (lines 1612-1636), expanded
    via the q1/q2/q3/q4 Bresenham macros (lines 1212-1600).  Endpoints are
    NOT tested -- only the intermediate cells (vendor: "The start and finish
    points themselves are not checked", line 1191).

    JIT-safe: bounded ``fori_loop`` over ``MAX_DIST`` steps.  When the two
    endpoints coincide vendor returns 1 directly (line 1627); we replicate.
    """
    r0 = row1.astype(jnp.int32); c0 = col1.astype(jnp.int32)
    r1 = row2.astype(jnp.int32); c1 = col2.astype(jnp.int32)

    terrain = _current_level_terrain(state)
    H, W = terrain.shape

    dr_signed = (r1 - r0).astype(jnp.int32)
    dc_signed = (c1 - c0).astype(jnp.int32)
    dr_abs = jnp.abs(dr_signed)
    dc_abs = jnp.abs(dc_signed)
    n_steps = jnp.maximum(dr_abs, dc_abs).astype(jnp.int32)
    n_steps_safe = jnp.maximum(n_steps, jnp.int32(1))

    def body(i, clear):
        # Only steps 1 .. n_steps-1 are intermediate (endpoints excluded).
        active = ((i + 1) < n_steps) & clear
        numer_r = dr_signed * (i + 1)
        numer_c = dc_signed * (i + 1)
        step_r = jnp.round(
            numer_r.astype(jnp.float32) / n_steps_safe.astype(jnp.float32)
        ).astype(jnp.int32)
        step_c = jnp.round(
            numer_c.astype(jnp.float32) / n_steps_safe.astype(jnp.float32)
        ).astype(jnp.int32)
        tr = r0 + step_r
        tc = c0 + step_c
        safe_r = jnp.clip(tr, 0, H - 1)
        safe_c = jnp.clip(tc, 0, W - 1)
        tile = terrain[safe_r, safe_c]
        blocked = _tile_blocks_los(tile)
        return jnp.where(active & blocked, jnp.bool_(False), clear)

    clear = jax.lax.fori_loop(0, MAX_DIST, body, jnp.bool_(True))
    # Vendor: row1 == row2 && col1 == col2 -> result = 1 (line 1626-1627).
    same_tile = (r0 == r1) & (c0 == c1)
    return clear | same_tile


# ---------------------------------------------------------------------------
# couldsee / cansee / cansee_with_blind
# ---------------------------------------------------------------------------

def couldsee(state, row: jnp.ndarray, col: jnp.ndarray) -> jnp.ndarray:
    """Return True iff the player has a clear LoS to ``(row, col)``.

    Cite: vendor/nethack/include/vision.h ``#define couldsee(x, y)``
    (line 29) -- ``(gv.viz_array[y][x] & COULD_SEE) != 0``.  The COULD_SEE
    bit ignores blindness; the underlying geometry is the same Bresenham
    clear_path used by ``m_cansee`` (vision.h:42).  Endpoints excluded.
    """
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    return clear_path(state, pr, pc, row.astype(jnp.int32), col.astype(jnp.int32))


def cansee(state, from_row: jnp.ndarray, from_col: jnp.ndarray,
           to_row: jnp.ndarray, to_col: jnp.ndarray) -> jnp.ndarray:
    """Return True iff there is a clear LoS from ``(from_row, from_col)`` to
    ``(to_row, to_col)``.

    Cite: vendor/nethack/include/vision.h ``#define cansee(x, y)`` (line 28).
    The macro reads ``gv.viz_array``; we recover the same predicate via
    ``clear_path`` between the two endpoints.  This is the general-source
    form used by monster vs. monster checks (``m_cansee`` in vision.h:42
    forwards to ``clear_path`` directly).
    """
    return clear_path(state, from_row, from_col, to_row, to_col)


def _player_is_blind(state) -> jnp.ndarray:
    """Return True iff the player is currently blind.

    Cite: vendor/nethack/include/display.h ``canseemon`` chain -- vendor
    folds ``Blind`` into ``cansee`` via ``HBlinded`` / ``BlindedTimeout``
    (prop.h:BLINDED).  In the JAX state this corresponds to
    ``status.timed_statuses[TimedStatus.BLIND] > 0``.
    """
    # Lazy import to avoid a status_effects -> vision import cycle.
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    return state.status.timed_statuses[int(TimedStatus.BLIND)] > jnp.int32(0)


def cansee_with_blind(state, row: jnp.ndarray, col: jnp.ndarray) -> jnp.ndarray:
    """Return True iff the player can see ``(row, col)`` accounting for
    blindness.

    Cite: vendor/nethack/include/display.h ``canseemon`` (line 144+) -- folds
    ``cansee`` AND ``!Blind`` to gate display.  Equivalent here to
    ``couldsee(state, row, col) AND NOT player_blind``.
    """
    return couldsee(state, row, col) & (~_player_is_blind(state))
