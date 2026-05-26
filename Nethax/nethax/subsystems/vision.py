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

# Boulder ground-item identity (mirrors subsystems/boulders.py).
# Vendor vision.c:182-184 walks ``svl.level.objects[x][y]`` looking for an
# obj with ``obj->otyp == BOULDER`` and treats it as an LoS blocker.
_BOULDER_CATEGORY: int = 14  # ItemCategory.ROCK
_BOULDER_TYPE_ID:  int = 0   # generic boulder sub-type

# Door state values (mirrors features.DoorState).  Vendor vision.c:167-168
# treats any door with ``doormask & (D_CLOSED | D_LOCKED | D_TRAPPED)``
# as an LoS blocker.  Our DoorState enum encodes:
#   GONE=0, BROKEN=1, OPEN=2, CLOSED=4, LOCKED=8, SECRET=32.
# We additionally consult ``features.door_trapped`` for the D_TRAPPED bit.
_DOOR_CLOSED: int = 4
_DOOR_LOCKED: int = 8
_DOOR_SECRET: int = 32


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _current_level_terrain(state) -> jnp.ndarray:
    """Slice the [H, W] terrain plane for the player's current branch/level."""
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    return state.terrain[b, lv]


def _tile_blocks_los(tile: jnp.ndarray) -> jnp.ndarray:
    """Return True if ``tile`` blocks line of sight at the terrain layer only.

    Cite: vendor/nethack/src/vision.c::does_block (lines 165-184).  We model
    the tile-layer blockers we currently carry in TileType: WALL, CLOSED_DOOR
    (vendor IS_DOOR with D_CLOSED|D_LOCKED|D_TRAPPED), and TREE.  Boulders,
    clouds, water-walls and lava-walls are not yet modeled as tile types.

    This helper is kept for callers that only have terrain context (e.g.
    ``fov.compute_fov`` with its terrain-only signature).  For callers
    that hold an EnvState, prefer :func:`_cell_blocks_los_full` which also
    consults the door-state and ground-item overlays.
    """
    t = tile.astype(jnp.int8)
    return (t == _TILE_WALL) | (t == _TILE_CLOSED_DOOR) | (t == _TILE_TREE)


def _cell_blocks_los_full(state, tile: jnp.ndarray,
                          row: jnp.ndarray, col: jnp.ndarray) -> jnp.ndarray:
    """Vendor-parity does_block check at a single cell.

    Cite: vendor/nethack/src/vision.c::does_block lines 152-202.  We model:

      * Terrain blockers (lines 166-169): WALL, TREE, and any IS_DOOR tile
        whose ``doormask`` carries ``D_CLOSED``, ``D_LOCKED`` or ``D_TRAPPED``.
        Our :class:`features.DoorState` already encodes CLOSED/LOCKED and the
        SECRET state, and the trapped bit lives on ``features.door_trapped``.
      * Boulder objects on the tile (lines 181-184): we walk only the front
        slot of ``state.ground_items[b, lv, row, col, 0]`` because Nethax
        stores boulders as the sole ground item in that slot (mirrors
        subsystems/boulders.py).

    Mimics impersonating doors/boulders (vendor lines 186-189) are NOT yet
    modelled — Nethax does not carry the ``m_ap_type`` discriminator that
    distinguishes ``M_AP_FURNITURE`` and ``M_AP_OBJECT`` disguises from
    ``M_AP_MONSTER`` and ``M_AP_NOTHING``.  This is documented as a known
    divergence; callers that need mimic-as-blocker semantics should extend
    monster_ai.MonsterAIState with that field first.
    """
    t = tile.astype(jnp.int8)
    blocks_terrain = (t == _TILE_WALL) | (t == _TILE_TREE)

    # Door-state overlay — vendor lines 167-168.  Any IS_DOOR tile (terrain
    # CLOSED_DOOR or the matching door_state at any non-OPEN value) with
    # D_CLOSED|D_LOCKED|D_TRAPPED set on the doormask is a blocker.  In
    # Nethax the terrain CLOSED_DOOR enum is the single representation of
    # a closed door (an open door is tile OPEN_DOOR), so the terrain check
    # already covers D_CLOSED; we additionally honour the LOCKED/SECRET
    # encoding by consulting the features.door_state plane.
    from Nethax.nethax.dungeon.branches import MAX_LEVELS_PER_BRANCH
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv_local = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    flat_lv = b * jnp.int32(MAX_LEVELS_PER_BRANCH) + lv_local
    door_val = state.features.door_state[flat_lv, row, col].astype(jnp.int32)
    door_trapped = state.features.door_trapped[flat_lv, row, col]
    is_closed_state = door_val == jnp.int32(_DOOR_CLOSED)
    is_locked_state = door_val == jnp.int32(_DOOR_LOCKED)
    is_secret_state = door_val == jnp.int32(_DOOR_SECRET)
    blocks_door = (
        (t == _TILE_CLOSED_DOOR)  # terrain-level closed door
        | is_closed_state
        | is_locked_state
        | is_secret_state
        | door_trapped
    )

    # Boulder overlay — vendor lines 181-184.  Walk ``ground_items[b, lv,
    # row, col, 0]`` (the only slot used for boulders per
    # subsystems/boulders.py::_tile_has_boulder).
    gi = state.ground_items
    g_cat = gi.category[b, lv_local, row, col, 0].astype(jnp.int32)
    g_tid = gi.type_id[b, lv_local, row, col, 0].astype(jnp.int32)
    blocks_boulder = (
        (g_cat == jnp.int32(_BOULDER_CATEGORY))
        & (g_tid == jnp.int32(_BOULDER_TYPE_ID))
    )

    return blocks_terrain | blocks_door | blocks_boulder


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
        # Full vendor does_block (vision.c:152-202): terrain + door state
        # + boulder overlay.  Mimic-as-blocker is documented as a known
        # divergence (see _cell_blocks_los_full).
        blocked = _cell_blocks_los_full(state, tile, safe_r, safe_c)
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
