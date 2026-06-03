"""Field-of-view computation — visibility mask and explored-tile tracking.

Canonical sources:
  vendor/nethack/src/vision.c — circle_data table, vision_recalc(), seenV(),
                                 recheck_pos() (radius tables, quadrant offsets)
  vendor/nethack/src/light.c  — vision_full_recalc(), light radius management,
                                 canseemon() / cansee()

Status: Wave 2 — real Bresenham-line raycast FOV, JIT-compatible.
"""
import jax
import jax.numpy as jnp

from Nethax.nethax.constants import OPAQUE_TILES

# ---------------------------------------------------------------------------
# Constants (vision.c / light.c conventions)
# ---------------------------------------------------------------------------

# "Normal" line-of-sight radius used in lit rooms and outdoors.
DEFAULT_SIGHT_RADIUS: int = 7

# Reduced sight radius in dark (unlit) rooms — see adjacent tiles only.
# Vendor: vision.c line 321 — "If dark, set COULD_SEE so various spells work".
# In a dark room the player can only perceive immediately adjacent tiles.
DARK_ROOM_SIGHT_RADIUS: int = 2

# Minimum sight radius when the player is blind — adjacent tiles only.
BLIND_SIGHT_RADIUS: int = 1

# Maximum number of steps we walk along a single ray.  A ray of length R
# crosses at most 2*R cells (conservative bound used to keep scan static).
_MAX_RAY_STEPS: int = 2 * DEFAULT_SIGHT_RADIUS + 2  # 16 for default radius


# ---------------------------------------------------------------------------
# Build a static boolean opacity lookup table at import time.
# shape: (NUM_TILE_TYPES_BOUND,) bool
# ---------------------------------------------------------------------------
_OPAQUE_TABLE_SIZE: int = 32  # larger than any TileType value


def _build_opaque_table() -> jnp.ndarray:
    tbl = jnp.zeros(_OPAQUE_TABLE_SIZE, dtype=jnp.bool_)
    for v in OPAQUE_TILES:
        tbl = tbl.at[int(v)].set(True)
    return tbl


_OPAQUE_TABLE: jnp.ndarray = _build_opaque_table()


# ---------------------------------------------------------------------------
# Internal: cast one ray from player toward a target offset
# ---------------------------------------------------------------------------

def _cast_ray(
    visible: jnp.ndarray,
    terrain: jnp.ndarray,
    opaque_overlay: jnp.ndarray,
    pr: jnp.int32,
    pc: jnp.int32,
    dr: jnp.int32,
    dc: jnp.int32,
    max_steps: int,
) -> jnp.ndarray:
    """Walk a Bresenham line from (pr, pc) toward (pr+dr, pc+dc).

    Marks each tile visible as we walk.  Stops (without marking further tiles)
    after the first opaque tile is encountered — the opaque tile itself IS
    marked visible (you see the wall blocking you).

    The ``opaque_overlay`` is an optional per-cell boolean mask whose True
    entries augment the terrain-only opacity table.  Used to fold vendor
    ``does_block`` overlays (boulders on the tile, ``D_LOCKED`` / ``D_TRAPPED``
    doors) into ray-stop decisions.  Cite: vendor/nethack/src/vision.c::
    does_block lines 152-202.

    Returns updated visible mask.
    """
    h, w = terrain.shape
    abs_dr = jnp.abs(dr)
    abs_dc = jnp.abs(dc)
    # Bresenham state
    # We walk in the dominant direction, accumulating error in the minor one.
    # The loop body is kept shape-static by always iterating `max_steps` times
    # but gating writes via a `still_going` flag carried through lax.scan.

    def step_fn(carry, _):
        cur_r, cur_c, err, still_going, vis = carry

        # Only update if we haven't hit an opaque tile yet.
        in_bounds = (cur_r >= 0) & (cur_r < h) & (cur_c >= 0) & (cur_c < w)
        tile_idx = jnp.where(in_bounds, terrain[cur_r, cur_c], jnp.int32(0))
        # Clip to lookup table size for safety.
        tile_idx_clipped = jnp.clip(tile_idx, 0, _OPAQUE_TABLE_SIZE - 1)
        is_opaque_terrain = _OPAQUE_TABLE[tile_idx_clipped]
        # Fold in the overlay (boulders + locked/trapped doors).  Bounds-
        # check matches the terrain probe above.
        safe_r = jnp.clip(cur_r, 0, h - 1)
        safe_c = jnp.clip(cur_c, 0, w - 1)
        is_opaque_overlay = jnp.where(in_bounds, opaque_overlay[safe_r, safe_c],
                                       jnp.bool_(False))
        is_opaque = is_opaque_terrain | is_opaque_overlay

        # Mark current cell visible if we're still going and in bounds.
        new_vis = jnp.where(
            still_going & in_bounds,
            vis.at[cur_r, cur_c].set(True),
            vis,
        )
        # After marking an opaque tile, stop advancing.
        new_still_going = still_going & (~is_opaque) & in_bounds

        # Advance position with Bresenham.
        # When |dr| >= |dc|: row is dominant axis.
        dominant_row = abs_dr >= abs_dc
        new_err = jnp.where(dominant_row, err + abs_dc, err + abs_dr)
        step_dominant_r = jnp.where(dominant_row, jnp.sign(dr), jnp.int32(0))
        step_dominant_c = jnp.where(dominant_row, jnp.int32(0), jnp.sign(dc))
        step_minor_r    = jnp.where(dominant_row, jnp.int32(0), jnp.sign(dr))
        step_minor_c    = jnp.where(dominant_row, jnp.sign(dc), jnp.int32(0))
        dominant_len    = jnp.where(dominant_row, abs_dr, abs_dc)
        # Minor step when error exceeds the dominant length.
        do_minor = new_err * 2 >= dominant_len
        new_err  = jnp.where(do_minor, new_err - dominant_len, new_err)
        next_r   = cur_r + step_dominant_r + jnp.where(do_minor, step_minor_r, jnp.int32(0))
        next_c   = cur_c + step_dominant_c + jnp.where(do_minor, step_minor_c, jnp.int32(0))

        return (next_r, next_c, new_err, new_still_going, new_vis), None

    # Start one step away from the player (player is already marked visible
    # by the caller before any rays are cast).
    sign_dr = jnp.sign(dr)
    sign_dc = jnp.sign(dc)
    start_r = pr + sign_dr
    start_c = pc + sign_dc
    init_err = jnp.int32(0)

    (_, _, _, _, visible_out), _ = jax.lax.scan(
        step_fn,
        (start_r, start_c, init_err, jnp.bool_(True), visible),
        None,
        length=max_steps,
    )
    return visible_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_fov(
    terrain: jnp.ndarray,
    player_pos: jnp.ndarray,
    sight_radius: int = DEFAULT_SIGHT_RADIUS,
    opaque_overlay: jnp.ndarray | None = None,
    lit_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute field-of-view mask via Bresenham-line raycast.

    Args:
        terrain: int[H, W] tile grid (current level only).
        player_pos: int[2] (row, col).
        sight_radius: max line-of-sight distance.
        opaque_overlay: optional bool[H, W] mask whose True cells augment
            the terrain-only opacity table.  Used to fold vendor
            ``does_block`` overlays into the FOV computation: boulders on
            the tile (vendor vision.c:181-184) and door state bits
            ``D_LOCKED|D_TRAPPED|D_SECRET`` (vendor vision.c:167-168) that
            are not already encoded in the TileType enum.  Defaults to an
            all-False mask (terrain-only opacity, current behaviour).
        lit_mask: optional bool[H, W] mask of LIT cells.  When provided, the
            raycast result is gated so a cell is kept visible only if it is
            LIT *or* within the hero's light radius (Chebyshev distance <= 1).
            This mirrors vendor ``vision_recalc`` / ``recheck_pos`` dark-cell
            handling: a DARK cell (unlit corridor / dark-room tile) is shown
            only when adjacent to the hero (inside the light radius) — being on
            a line of sight through a doorway is NOT enough.  Cite:
            vendor/nethack/src/vision.c::vision_recalc lines 320-335 (rlit gates
            IN_SIGHT) and the per-cell could-see-but-unlit case.  When ``None``
            the gate is skipped entirely (terrain-only LOS — current behaviour),
            so existing callers (the per-step ``_apply_fov`` path, which already
            limits reach via ``sight_radius``) are unaffected.

    Returns:
        bool[H, W] mask, True where player can see this turn.

    Algorithm:
        Cast rays from the player to every tile on the perimeter of a square
        bounding box of radius `sight_radius`.  Each ray walks a Bresenham
        line; tiles are marked visible until the ray hits an opaque tile
        (inclusive — you see the wall that blocks you).

    JAX compatibility:
        All loops use `jax.lax.scan` / `jax.lax.fori_loop` with static sizes,
        so the function is `jit`-compatible with no Python-side data-dependent
        branches.
    """
    h, w = terrain.shape
    pr = player_pos[0].astype(jnp.int32)
    pc = player_pos[1].astype(jnp.int32)
    if opaque_overlay is None:
        opaque_overlay = jnp.zeros((h, w), dtype=jnp.bool_)

    # Start with the player's own tile visible.
    visible = jnp.zeros((h, w), dtype=jnp.bool_)
    visible = visible.at[pr, pc].set(True)

    # Enumerate all target offsets on the perimeter of the bounding square.
    # We always use DEFAULT_SIGHT_RADIUS for static array sizes (required by
    # jnp.arange inside JIT), then mask out tiles beyond sight_radius at the
    # end.  This keeps the function JIT-pure even when sight_radius is a traced
    # value (e.g. jnp.where for blindness).
    # Vendor: vision.c — blindness forces radius=1.
    R_static = DEFAULT_SIGHT_RADIUS  # static: used only for loop bounds
    sight_radius_i32 = jnp.int32(sight_radius)  # may be traced
    diameter = 2 * R_static + 1
    max_steps = R_static  # upper bound; rays beyond actual radius get masked

    # Build (dr, dc) pairs for all cells in [-R_static, R_static]^2.
    rows = jnp.arange(-R_static, R_static + 1, dtype=jnp.int32)  # (diameter,)
    cols = jnp.arange(-R_static, R_static + 1, dtype=jnp.int32)  # (diameter,)
    dr_grid, dc_grid = jnp.meshgrid(rows, cols, indexing="ij")  # (D, D)
    dr_flat = dr_grid.reshape(-1)   # (D*D,)
    dc_flat = dc_grid.reshape(-1)   # (D*D,)

    # Zero-offset (player tile) is already marked — skip it in the scan body
    # by keeping the cast trivially harmless (ray of length 0).

    def cast_one(vis, idx):
        dr = dr_flat[idx]
        dc = dc_flat[idx]
        # Skip the player's own cell (offset 0,0) — already marked.
        # Also skip rays whose L∞ distance exceeds the actual sight_radius.
        beyond_radius = jnp.maximum(jnp.abs(dr), jnp.abs(dc)) > sight_radius_i32
        skip = ((dr == 0) & (dc == 0)) | beyond_radius
        new_vis = jax.lax.cond(
            skip,
            lambda v: v,
            lambda v: _cast_ray(v, terrain, opaque_overlay,
                                pr, pc, dr, dc, max_steps),
            vis,
        )
        return new_vis, None

    visible, _ = jax.lax.scan(cast_one, visible, jnp.arange(diameter * diameter))
    # Additionally mask out any tiles whose Chebyshev distance from player
    # exceeds sight_radius (guards against diagonal rays slightly overreaching).
    rows_all = jnp.arange(h, dtype=jnp.int32)
    cols_all = jnp.arange(w, dtype=jnp.int32)
    rr, cc = jnp.meshgrid(rows_all, cols_all, indexing="ij")
    chebyshev = jnp.maximum(jnp.abs(rr - pr), jnp.abs(cc - pc))
    visible = visible & (chebyshev <= sight_radius_i32)

    # Dark-cell gate (vendor vision.c::vision_recalc dark-tile handling).
    # When a per-cell ``lit_mask`` is supplied, a cell that the Bresenham rays
    # reached is only actually SEEN if it is LIT or sits within the hero's own
    # light radius (Chebyshev distance <= 1).  A far DARK corridor / dark-room
    # cell on a line of sight through a doorway is NOT seen until the hero steps
    # adjacent to it — vendor only sets IN_SIGHT for unlit cells inside the
    # light radius, not for every couldsee() cell.  Without this gate the raycast
    # over-reveals dark corridor tiles distal to the hero at level entry.
    # Cite: vendor/nethack/src/vision.c:320-335 (rlit gate) + recheck_pos.
    if lit_mask is not None:
        within_light = chebyshev <= jnp.int32(1)
        visible = visible & (lit_mask | within_light)
    return visible


def lit_room_flood(
    player_pos: jnp.ndarray,
    room_x1: jnp.ndarray,
    room_y1: jnp.ndarray,
    room_x2: jnp.ndarray,
    room_y2: jnp.ndarray,
    room_active: jnp.ndarray,
    room_lit: jnp.ndarray,
    h: int,
    w: int,
) -> jnp.ndarray:
    """Flood-visibility mask for the hero's containing LIT room.

    Vendor ``vision_recalc`` (vendor/nethack/src/vision.c:320-335): when the
    hero stands inside a room, the whole room region — interior cells PLUS the
    one-cell bounding wall ring ``[lx-1..hx+1] x [ly-1..hy+1]`` — is set
    ``COULD_SEE | IN_SIGHT`` and the walls are stamped ``seenv = SVALL`` *iff*
    the room is lit (``rooms[rnum].rlit``).  A dark room only gets ``COULD_SEE``
    (no IN_SIGHT flood); its visible tiles come from the per-step LOS / adjacent
    cells instead.  This helper returns the IN_SIGHT flood for the lit case so
    the caller can OR it into the Bresenham LOS visibility — Bresenham still
    handles corridors and dark areas, this only adds the far walls/corners of a
    lit room that rays can't reach.

    Containing-room test mirrors vendor ``inside_room`` (mkroom.c:653):
        lx-1 <= px <= hx+1  &&  ly-1 <= py <= hy+1
    on the interior bounding box (Room.x1/x2/y1/y2).

    All inputs are fixed-size arrays over ``MAX_ROOMS_PER_LEVEL`` so this stays
    JIT-pure: no Python branches on traced values, vectorised over the rooms.

    Args:
        player_pos: int[2] (row, col).
        room_x1/y1/x2/y2: int arrays [n_rooms] — interior bounding box.
        room_active: bool[n_rooms] — placed-room mask.
        room_lit:    bool[n_rooms] — per-room lit flag (rlit).
        h, w: map dimensions (static).

    Returns:
        bool[h, w] mask, True for cells flooded by the hero's lit room.
    """
    pr = player_pos[0].astype(jnp.int32)
    pc = player_pos[1].astype(jnp.int32)

    x1 = room_x1.astype(jnp.int32)
    y1 = room_y1.astype(jnp.int32)
    x2 = room_x2.astype(jnp.int32)
    y2 = room_y2.astype(jnp.int32)

    # inside_room (mkroom.c:653) over the wall-inclusive ring; gate on active+lit.
    contains = (
        room_active
        & room_lit
        & (pc >= x1 - 1) & (pc <= x2 + 1)
        & (pr >= y1 - 1) & (pr <= y2 + 1)
    )  # bool[n_rooms]

    rows = jnp.arange(h, dtype=jnp.int32)[:, None]   # (h, 1)
    cols = jnp.arange(w, dtype=jnp.int32)[None, :]   # (1, w)

    def room_mask(i):
        # Cells of room i's wall-inclusive ring, only if it contains the hero.
        in_box = (
            (cols >= x1[i] - 1) & (cols <= x2[i] + 1)
            & (rows >= y1[i] - 1) & (rows <= y2[i] + 1)
        )
        return in_box & contains[i]

    masks = jax.vmap(room_mask)(jnp.arange(x1.shape[0]))  # (n_rooms, h, w)
    return jnp.any(masks, axis=0)


def update_explored(
    explored: jnp.ndarray,
    fov: jnp.ndarray,
) -> jnp.ndarray:
    """Return updated explored mask: explored OR fov.

    A tile is permanently marked explored once it enters the player's FOV.
    Equivalent to NetHack's ``levl[x][y].seenv`` flag set in vision_recalc().
    """
    return explored | fov


# ===========================================================================
# view_from — JAX port of vendor's Algorithm C (vendor/nle/src/vision.c).
#
# Vendor reference:
#   ``view_from`` lines 2640-2731 (Algorithm C, the one NLE compiles with —
#   VISION_TABLES is undefined per vendor/nle/include/config.h:458).
#   ``right_side`` lines 2312-2498.
#   ``left_side``  lines 2504-2633.
#   ``q1_path`` / ``q2_path`` / ``q3_path`` / ``q4_path`` lines 1160-1352
#   (MACRO_CPATH expansion — Bresenham line-clear tests).
#   ``viz_clear`` / ``left_ptrs`` / ``right_ptrs`` table build in
#   ``vision_reset`` lines 206-243.
#   ``does_block`` opacity predicate lines 156-184.
#
# The vendor algorithm is a row-by-row sweep outward from the source, with
# per-row "fingers of light" (column ranges) recursively passed to the next
# row.  Each finger uses Bresenham ``q?_path`` to verify visibility at its
# endpoints.  The result is a ``could_see`` bool grid matching the bits the
# vendor sets via ``set_cs``.
#
# JAX strategy:
#   * Pre-build ``viz_clear`` / ``left_ptrs`` / ``right_ptrs`` tables from
#     the terrain + opaque_overlay (one vectorised pass per row).
#   * Bresenham ``q?_path`` is a ``jax.lax.fori_loop`` over a fixed number
#     of steps (bounded by row/col distance).
#   * The recursive row-by-row segment cascade is flattened to a fixed-size
#     work queue: each entry is ``(row, left, right_mark, side, step,
#     active)``.  Per quadrant we run a bounded ``fori_loop`` that pops one
#     segment per iteration and may push 0..K new segments for the next row.
#     With map size 21x80 and per-row segment count bounded by ~COLNO/2,
#     a queue of ~4*ROWNO*COLNO/2 entries is safely over-provisioned.
# ===========================================================================


# Maximum sight radius vendor supports (circle_data terminator at index 135
# stores MAX_RADIUS+1 = 16, so MAX_RADIUS = 15).  Cite vision.c lines 26-43.
MAX_RADIUS: int = 15

# Vendor circle_data table (vendor/nle/src/vision.c lines 26-43).  For a given
# radius R, ``circle_data[circle_start[R] + dy]`` is the maximum column delta
# permitted at row distance dy from the source.  Used by ``view_from`` to
# clamp per-row sight extent when range>0; with range=0 (unlimited) we skip
# the clamp entirely, matching vendor.
_CIRCLE_DATA: tuple[int, ...] = (
    1, 1,
    2, 2, 1,
    3, 3, 2, 1,
    4, 4, 4, 3, 2,
    5, 5, 5, 4, 3, 2,
    6, 6, 6, 5, 5, 4, 2,
    7, 7, 7, 6, 6, 5, 4, 2,
    8, 8, 8, 7, 7, 6, 6, 4, 2,
    9, 9, 9, 9, 8, 8, 7, 6, 5, 3,
    10, 10, 10, 10, 9, 9, 8, 7, 6, 5, 3,
    11, 11, 11, 11, 10, 10, 9, 9, 8, 7, 5, 3,
    12, 12, 12, 12, 11, 11, 10, 10, 9, 8, 7, 5, 3,
    13, 13, 13, 13, 12, 12, 12, 11, 10, 10, 9, 7, 6, 3,
    14, 14, 14, 14, 13, 13, 13, 12, 12, 11, 10, 9, 8, 6, 3,
    15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3,
    16,
)

_CIRCLE_START: tuple[int, ...] = (
    0,   # radius 0 unused
    0,   # 1
    2,   # 2
    5,   # 3
    9,   # 4
    14,  # 5
    20,  # 6
    27,  # 7
    35,  # 8
    44,  # 9
    54,  # 10
    65,  # 11
    77,  # 12
    90,  # 13
    104, # 14
    119, # 15
)


def _build_terrain_opaque(terrain: jnp.ndarray) -> jnp.ndarray:
    """Vendor ``IS_ROCK(typ) || does_block(...)`` terrain check (vision.c:215).

    Returns bool[H, W] True where the cell terrain blocks sight.  Boulder /
    locked-door overlays are passed in separately as ``opaque_overlay``.
    """
    tile_idx = jnp.clip(terrain, 0, _OPAQUE_TABLE_SIZE - 1).astype(jnp.int32)
    return _OPAQUE_TABLE[tile_idx]


def _build_viz_tables(opaque: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray,
                                                     jnp.ndarray]:
    """Build ``viz_clear``, ``left_ptrs``, ``right_ptrs`` from opacity grid.

    Mirrors vendor ``vision_reset`` lines 206-243.  Walks each row left to
    right tracking ``dig_left`` (start of current run) and ``block`` (TRUE
    when current run is opaque).  At each opaque<->clear transition, fills
    in the prior run's pointer entries.

    Semantics (per vendor):
      * For a CLEAR cell at (y, x): ``left_ptrs[y][x]`` = column of opaque
        cell bounding the clear run on the left (or 0 if the run touches
        the left boundary), ``right_ptrs[y][x]`` = column of opaque cell
        bounding on the right (or COLNO-1 at right boundary).
      * For an OPAQUE cell at (y, x): ``left_ptrs[y][x]`` / ``right_ptrs``
        bracket the contiguous opaque run.

    Column 0 is always treated as opaque (vendor's ``!isok(0, y)`` — line
    212), so the iteration starts at x=1 with ``dig_left=0, block=TRUE``.
    """
    h, w = opaque.shape
    # Force col 0 to "always opaque" — vendor: location (0,y) is always stone.
    opaque0 = opaque.at[:, 0].set(True)

    def per_row(row_opq: jnp.ndarray):
        """Process one row.  Returns (viz_clear_row, left_row, right_row)."""
        # Iterate x = 1..w-1 with carry (dig_left, block, left_arr, right_arr).
        # We accumulate left_ptrs/right_ptrs into work arrays.
        init_viz = jnp.zeros((w,), dtype=jnp.bool_)
        init_left = jnp.zeros((w,), dtype=jnp.int32)
        init_right = jnp.zeros((w,), dtype=jnp.int32)

        def body(x, carry):
            dig_left, block, vizr, lp, rp = carry
            cur_block = row_opq[x]
            transition = block != cur_block
            # If transition AND prior run was opaque (block==True), close it:
            # for i in [dig_left, x-1]: left=dig_left, right=x-1, viz=0.
            # If transition AND prior run was clear (block==False), close it:
            # i = dig_left; if dig_left: dig_left--; for i in [start, x-1]:
            #   left=dig_left_dec, right=x, viz=1.
            # Then dig_left = x, block = !block.
            #
            # Vectorised version: compute the index mask, fill columns where
            # transition + within run.
            run_start = jnp.where(transition & ~block,
                                  jnp.where(dig_left > 0, dig_left - 1,
                                            dig_left),
                                  dig_left)
            # For the close-out, "fill" columns [dig_left .. x-1] (when block)
            # or [dig_left .. x-1] (when !block) with new left/right values.
            cols = jnp.arange(w, dtype=jnp.int32)
            in_run = transition & (cols >= dig_left) & (cols < x)

            new_left_val = jnp.where(block, dig_left, run_start)
            new_right_val = jnp.where(block, x - 1, x)
            new_viz_val = jnp.where(block, jnp.bool_(False), jnp.bool_(True))

            lp2 = jnp.where(in_run, new_left_val, lp)
            rp2 = jnp.where(in_run, new_right_val, rp)
            vizr2 = jnp.where(in_run, new_viz_val, vizr)

            new_dig_left = jnp.where(transition, x, dig_left)
            new_block = jnp.where(transition, ~block, block)
            return (new_dig_left, new_block, vizr2, lp2, rp2)

        final = jax.lax.fori_loop(
            1, w,
            body,
            (jnp.int32(0), jnp.bool_(True), init_viz, init_left, init_right),
        )
        dig_left, block, vizr, lp, rp = final

        # Right-boundary close-out (lines 235-242).
        # i = dig_left; if !block and dig_left: dig_left--;
        # for i in [start..COLNO-1]: left=dig_left, right=COLNO-1, viz=!block.
        start_idx = dig_left
        dig_left_dec = jnp.where(~block & (dig_left > 0), dig_left - 1, dig_left)
        cols = jnp.arange(w, dtype=jnp.int32)
        in_tail = cols >= start_idx
        lp_final = jnp.where(in_tail, dig_left_dec, lp)
        rp_final = jnp.where(in_tail, w - 1, rp)
        viz_final = jnp.where(in_tail, ~block, vizr)
        return viz_final, lp_final, rp_final

    viz_clear, left_ptrs, right_ptrs = jax.vmap(per_row)(opaque0)
    return viz_clear, left_ptrs, right_ptrs


def _bresenham_clear(
    viz_clear: jnp.ndarray,
    srow: jnp.int32,
    scol: jnp.int32,
    erow: jnp.int32,
    ecol: jnp.int32,
) -> jnp.bool_:
    """Generalised q?_path Bresenham clear-line test (vision.c:1163-1352).

    Walks the integer Bresenham line from (srow, scol) to (erow, ecol).
    The endpoints are NOT checked, only the cells strictly between them.
    Returns True if every intermediate cell is clear, False on first hit.

    Vendor uses 4 quadrant-specialised macros (q1..q4) chosen by the relative
    position of (erow, ecol) to (srow, scol).  All four are the same Bresenham
    algorithm with different signs; we unify them by computing absolute deltas
    and signed step directions, exactly like vendor's clear_path() forwarder
    at vision.c lines 1568-1593 (which dispatches to q1/q2/q3/q4 by quadrant).
    """
    dy = jnp.abs(srow - erow).astype(jnp.int32)
    dx = jnp.abs(scol - ecol).astype(jnp.int32)
    step_y = jnp.sign(erow - srow).astype(jnp.int32)
    step_x = jnp.sign(ecol - scol).astype(jnp.int32)
    dxs = dx << 1
    dys = dy << 1

    # Two branches in vendor: dy > dx (vertical-dominant) vs else (horizontal-
    # dominant).  Both have a (max(dy, dx) - 1)-iteration loop.  We unify into
    # a single fixed-iter fori_loop sized to ROWNO+COLNO and gate by k.
    # MAP_H=21, MAP_W=80 -> max steps ~80.
    max_iters = 80  # safe upper bound for map width

    def body_vert(i, state):
        x, y, err, ok = state
        # Apply vendor "if (err >= 0): x += step_x; err -= dys" then y += step_y; err += dxs
        do_x_step = err >= 0
        x = x + jnp.where(do_x_step, step_x, jnp.int32(0))
        err = err - jnp.where(do_x_step, dys, jnp.int32(0))
        y = y + step_y
        err = err + dxs
        # Test current cell (vendor's "if !is_clear(y, x) goto label").
        h, w = viz_clear.shape
        in_bounds = (y >= 0) & (y < h) & (x >= 0) & (x < w)
        cell_clear = jnp.where(
            in_bounds,
            viz_clear[jnp.clip(y, 0, h - 1), jnp.clip(x, 0, w - 1)],
            jnp.bool_(False),
        )
        # k iterations are dy-1; for i >= dy-1 we skip the work
        active = (i < (dy - 1)) & ok
        new_ok = ok & (~active | cell_clear)
        # Don't advance state when inactive (so further iters stay quiescent).
        # But we've already mutated x/y; that's fine because new_ok will stay
        # False if we hit an opaque, and inactive iters can't change new_ok.
        return (x, y, err, new_ok)

    def body_horiz(i, state):
        x, y, err, ok = state
        do_y_step = err >= 0
        y = y + jnp.where(do_y_step, step_y, jnp.int32(0))
        err = err - jnp.where(do_y_step, dxs, jnp.int32(0))
        x = x + step_x
        err = err + dys
        h, w = viz_clear.shape
        in_bounds = (y >= 0) & (y < h) & (x >= 0) & (x < w)
        cell_clear = jnp.where(
            in_bounds,
            viz_clear[jnp.clip(y, 0, h - 1), jnp.clip(x, 0, w - 1)],
            jnp.bool_(False),
        )
        active = (i < (dx - 1)) & ok
        new_ok = ok & (~active | cell_clear)
        return (x, y, err, new_ok)

    vert_dominant = dy > dx

    # Vertical branch initial state
    init_err_vert = dxs - dy
    init_state_vert = (scol, srow, init_err_vert, jnp.bool_(True))
    final_vert = jax.lax.fori_loop(0, max_iters, body_vert, init_state_vert)
    ok_vert = final_vert[3]

    # Horizontal branch initial state
    init_err_horiz = dys - dx
    init_state_horiz = (scol, srow, init_err_horiz, jnp.bool_(True))
    final_horiz = jax.lax.fori_loop(0, max_iters, body_horiz, init_state_horiz)
    ok_horiz = final_horiz[3]

    # If both dx and dy are 0 (caller passes same source/dest), vendor would
    # not call q?_path at all (precondition lines 1142-1144); we treat this
    # as clear to be defensive.
    return jnp.where(vert_dominant, ok_vert, ok_horiz)


# ---------------------------------------------------------------------------
# view_from — JAX port of vendor Algorithm C row-segment sweep.
# ---------------------------------------------------------------------------

# Fixed-size work-queue per quadrant.  Each row generates at most ~COLNO/2
# segments (alternating clear/opaque), and we have up to ROWNO=21 rows per
# quadrant.  A 512-entry queue is comfortably over-provisioned for a single
# quadrant pass.
_MAX_QUEUE: int = 512

# Maximum iterations for the per-quadrant row sweep work loop.  Each iteration
# pops one queue entry; with up to ~21 * 40 = 840 segments worst-case we use
# a slightly larger bound.
_MAX_QUEUE_ITERS: int = 1024


def _process_segment(
    visible: jnp.ndarray,
    viz_clear: jnp.ndarray,
    left_ptrs: jnp.ndarray,
    right_ptrs: jnp.ndarray,
    srow: jnp.int32,
    scol: jnp.int32,
    row: jnp.int32,
    left: jnp.int32,
    right_mark: jnp.int32,
    step: jnp.int32,
    side_right: jnp.bool_,
    lim_max: jnp.int32,
    lim_min: jnp.int32,
    deeper: jnp.bool_,
    next_queue_left: jnp.ndarray,
    next_queue_right: jnp.ndarray,
    next_queue_count: jnp.int32,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.int32]:
    """Process one segment on one row, mirroring vendor right_side/left_side
    body (vision.c:2353-2497 / 2536-2632).

    For ``side_right=True`` we run the right_side body (sweeping cols from
    ``left`` upward toward ``right_mark``).  For ``side_right=False`` we run
    the mirror-image left_side body (sweeping ``right`` downward toward
    ``left_mark``); the parameters are interpreted symmetrically: ``left``
    plays the role of ``right`` (sweep start), ``right_mark`` plays the role
    of ``left_mark`` (sweep end).

    Returns updated (visible, next_queue_left, next_queue_right,
    next_queue_count) tuple.  May push 0..N new segments onto the next-row
    queue for recursive ``deeper`` follow-up.
    """
    h, w = visible.shape
    # Per-row inner while-loop: ``while left <= right_mark`` (right) or
    # ``while right >= left_mark`` (left).  Bounded by w iterations.
    MAX_INNER = 80

    def inner_body(i, state):
        (cur, vis, qL, qR, qN, done) = state

        def do_iter(_):
            # Right-side: cur = "left" cursor; bound = right_mark; sweep up.
            # Left-side: cur = "right" cursor; bound = left_mark; sweep down.
            cur_in_bounds = jnp.where(
                side_right,
                cur <= right_mark,
                cur >= right_mark,  # in left-side, right_mark = left_mark
            )

            def in_iter(_):
                # right_edge = right_side ? right_ptrs[row][cur] : left_ptrs[row][cur]
                rr = jnp.clip(row, 0, h - 1)
                cc = jnp.clip(cur, 0, w - 1)
                rp_val = right_ptrs[rr, cc].astype(jnp.int32)
                lp_val = left_ptrs[rr, cc].astype(jnp.int32)
                edge = jnp.where(side_right, rp_val, lp_val)
                # Clamp to limit.
                edge = jnp.where(side_right,
                                 jnp.minimum(edge, lim_max),
                                 jnp.maximum(edge, lim_min))

                cell_clear = viz_clear[rr, cc]

                # Case 1: cell at `cur` is opaque — "Jump to the far side of
                # a stone wall".
                def opaque_case(_):
                    # edge_overshoot test: right_side: edge > right_mark;
                    # left_side: edge < right_mark (where right_mark = left_mark)
                    overshoot = jnp.where(side_right,
                                          edge > right_mark,
                                          edge < right_mark)
                    # Corner kludge: if cell on prev row at right_mark is clear,
                    # extend edge by one step.
                    prev_row = row - step
                    pr_in_bounds = (prev_row >= 0) & (prev_row < h)
                    pr_idx = jnp.clip(prev_row, 0, h - 1)
                    mark_idx = jnp.clip(right_mark, 0, w - 1)
                    prev_clear = jnp.where(
                        pr_in_bounds,
                        viz_clear[pr_idx, mark_idx],
                        jnp.bool_(False),
                    )
                    edge2 = jnp.where(
                        overshoot,
                        jnp.where(side_right,
                                  jnp.where(prev_clear, right_mark + 1, right_mark),
                                  jnp.where(prev_clear, right_mark - 1, right_mark)),
                        edge,
                    )
                    # Mark cells [cur..edge2] (right) or [edge2..cur] (left) visible.
                    cols = jnp.arange(w, dtype=jnp.int32)
                    in_range = jnp.where(
                        side_right,
                        (cols >= cur) & (cols <= edge2),
                        (cols >= edge2) & (cols <= cur),
                    )
                    row_idx = jnp.clip(row, 0, h - 1)
                    new_row = jnp.where(in_range, jnp.bool_(True), vis[row_idx])
                    vis2 = vis.at[row_idx].set(new_row)
                    # Advance cur past the wall: right: cur = edge2 + 1; left: cur = edge2 - 1.
                    new_cur = jnp.where(side_right, edge2 + 1, edge2 - 1)
                    # No recursion in the wall-jump branch.
                    return (new_cur, vis2, qL, qR, qN, jnp.bool_(False))

                def clear_case(_):
                    # Vendor: "if (left != start_col)" — skip the left-finding
                    # search when our cursor is already on the source column.
                    on_source_col = cur == scol
                    # ============================================
                    # Find the visible-from-source end of the segment.
                    # ============================================
                    # For right_side:  left = first col in [cur..edge] such
                    #   that q?_path(srow, scol, row, left) succeeds.
                    # For left_side:   right = first col in [cur..edge] (cur
                    #   sweeping down) such that q?_path succeeds.
                    def find_visible(_):
                        # Sweep from `cur` toward `edge` looking for first clear path.
                        # Right: cur..edge ascending; Left: cur..edge descending.
                        MAX_FIND = 80
                        def find_body(i2, st):
                            (pos, found, done2) = st
                            def chk(_):
                                cond = jnp.where(
                                    side_right,
                                    pos <= edge,
                                    pos >= edge,
                                )
                                def do_test(_):
                                    ok = _bresenham_clear(
                                        viz_clear, srow, scol, row, pos
                                    )
                                    return jax.lax.cond(
                                        ok,
                                        lambda _: (pos, jnp.bool_(True),
                                                   jnp.bool_(True)),
                                        lambda _: (
                                            jnp.where(side_right,
                                                      pos + 1, pos - 1),
                                            found,
                                            done2,
                                        ),
                                        operand=None,
                                    )
                                return jax.lax.cond(cond, do_test,
                                                     lambda _: (pos, found,
                                                                jnp.bool_(True)),
                                                     operand=None)
                            return jax.lax.cond(done2, lambda _: st, chk,
                                                operand=None)
                        init = (cur, jnp.bool_(False), jnp.bool_(False))
                        return jax.lax.fori_loop(0, MAX_FIND, find_body, init)

                    found_pos, found_any, _ = jax.lax.cond(
                        on_source_col,
                        lambda _: (cur, jnp.bool_(True), jnp.bool_(True)),
                        find_visible,
                        operand=None,
                    )
                    # Vendor only does the boundary checks (lines 2414-2435 /
                    # 2575-2590) inside the ``if (left != start_col)`` block
                    # (lines 2391-2435).  When ``on_source_col`` is True we
                    # skip find_visible entirely and proceed directly to the
                    # find-right-side step.  Encode that here: ``past_limit``
                    # / ``at_limit`` / ``backed_up`` only fire when we actually
                    # ran the find_visible sweep.
                    # Boundary checks (lines 2414-2424 / 2575-2585):
                    # right_side: if (left > lim_max) return; if (left == lim_max) {mark; return}
                    # left_side:  if (right < lim_min) return; if (right == lim_min) {mark; return}
                    past_limit = (~on_source_col) & jnp.where(
                        side_right,
                        found_pos > lim_max,
                        found_pos < lim_min,
                    )
                    at_limit = (~on_source_col) & jnp.where(
                        side_right,
                        found_pos == lim_max,
                        found_pos == lim_min,
                    )

                    def early_return(_):
                        # Return from the entire while-loop: set done=True.
                        return (cur, vis, qL, qR, qN, jnp.bool_(True))

                    def at_limit_branch(_):
                        # Mark cell (row, found_pos) visible and return.
                        row_idx = jnp.clip(row, 0, h - 1)
                        col_idx = jnp.clip(found_pos, 0, w - 1)
                        vis2 = vis.at[row_idx, col_idx].set(True)
                        return (cur, vis2, qL, qR, qN, jnp.bool_(True))

                    # Backed-up case (lines 2432-2435 / 2587-2590):
                    # right_side: if (left >= right_edge): left = right_edge; continue.
                    # left_side:  if (right <= left_edge): right = left_edge; continue.
                    backed_up = (~on_source_col) & jnp.where(
                        side_right,
                        found_pos >= edge,
                        found_pos <= edge,
                    )

                    def backed_up_branch(_):
                        new_cur = edge
                        return (new_cur, vis, qL, qR, qN, jnp.bool_(False))

                    def normal_branch(_):
                        # ``found_pos`` is the leftmost (right_side) / rightmost
                        # (left_side) visible-from-source col.  Now find the
                        # far end:
                        # right_side: right = right_mark if right_mark >= edge
                        #             else sweep right_mark..edge until q?_path fails.
                        # left_side:  left  = left_mark  if left_mark  <= edge
                        #             else sweep left_mark..edge descending.
                        mark_inside_edge = jnp.where(
                            side_right,
                            right_mark < edge,
                            right_mark > edge,
                        )

                        def far_via_sweep(_):
                            MAX_FAR = 80
                            def far_body(i3, st):
                                (pos, last_ok, done3) = st
                                def chk(_):
                                    cond = jnp.where(
                                        side_right,
                                        pos <= edge,
                                        pos >= edge,
                                    )
                                    def do_test(_):
                                        ok = _bresenham_clear(
                                            viz_clear, srow, scol, row, pos
                                        )
                                        return jax.lax.cond(
                                            ok,
                                            lambda _: (
                                                jnp.where(side_right,
                                                          pos + 1, pos - 1),
                                                pos,
                                                done3,
                                            ),
                                            lambda _: (pos, last_ok,
                                                       jnp.bool_(True)),
                                            operand=None,
                                        )
                                    return jax.lax.cond(cond, do_test,
                                                         lambda _: (pos, last_ok,
                                                                    jnp.bool_(True)),
                                                         operand=None)
                                return jax.lax.cond(done3, lambda _: st, chk,
                                                    operand=None)
                            # Vendor sweeps from right_mark to edge; we replicate.
                            # last_ok starts as right_mark - step (so if no
                            # iteration succeeds, the loop produces empty range
                            # and the outer test left <= right kicks in).
                            init_last = jnp.where(side_right,
                                                  right_mark - jnp.int32(1),
                                                  right_mark + jnp.int32(1))
                            init = (right_mark, init_last, jnp.bool_(False))
                            _, last_ok_out, _ = jax.lax.fori_loop(
                                0, MAX_FAR, far_body, init
                            )
                            return last_ok_out

                        far_pos = jax.lax.cond(
                            mark_inside_edge,
                            far_via_sweep,
                            lambda _: edge,
                            operand=None,
                        )

                        # right_side: left=found_pos, right=far_pos
                        # left_side:  right=found_pos, left=far_pos
                        seg_lo = jnp.where(side_right, found_pos, far_pos)
                        seg_hi = jnp.where(side_right, far_pos, found_pos)

                        # Ugly special case (lines 2475-2477 / 2611-2613):
                        # right_side: if (left == right == start_col
                        #             && start_col < COLNO-1 && !is_clear(row, start_col+1))
                        #                 right = start_col + 1
                        # left_side:  if (left == right == start_col
                        #             && start_col > 0 && !is_clear(row, start_col-1))
                        #                 left = start_col - 1
                        sc_idx = jnp.clip(scol, 0, w - 1)
                        sc_plus_idx = jnp.clip(scol + 1, 0, w - 1)
                        sc_minus_idx = jnp.clip(scol - 1, 0, w - 1)
                        row_idx0 = jnp.clip(row, 0, h - 1)
                        adj_plus_opaque = ~viz_clear[row_idx0, sc_plus_idx]
                        adj_minus_opaque = ~viz_clear[row_idx0, sc_minus_idx]
                        special_right = (
                            side_right
                            & (seg_lo == seg_hi) & (seg_lo == scol)
                            & (scol < (w - 1)) & adj_plus_opaque
                        )
                        special_left = (
                            (~side_right)
                            & (seg_lo == seg_hi) & (seg_hi == scol)
                            & (scol > 0) & adj_minus_opaque
                        )
                        seg_hi = jnp.where(special_right, scol + 1, seg_hi)
                        seg_lo = jnp.where(special_left, scol - 1, seg_lo)

                        # Clamp to limits (lines 2479-2480 / 2615-2616).
                        seg_lo = jnp.maximum(seg_lo, lim_min)
                        seg_hi = jnp.minimum(seg_hi, lim_max)

                        valid_seg = seg_lo <= seg_hi

                        def emit_seg(_):
                            # Mark [seg_lo..seg_hi] in `row` visible.
                            cols = jnp.arange(w, dtype=jnp.int32)
                            in_range = (cols >= seg_lo) & (cols <= seg_hi)
                            row_idx = jnp.clip(row, 0, h - 1)
                            new_row = jnp.where(in_range, jnp.bool_(True),
                                                vis[row_idx])
                            vis2 = vis.at[row_idx].set(new_row)
                            # Push (seg_lo, seg_hi) onto next-row queue if deeper.
                            should_push = deeper
                            push_idx = qN
                            qL2 = jax.lax.cond(
                                should_push,
                                lambda _: qL.at[push_idx].set(seg_lo),
                                lambda _: qL,
                                operand=None,
                            )
                            qR2 = jax.lax.cond(
                                should_push,
                                lambda _: qR.at[push_idx].set(seg_hi),
                                lambda _: qR,
                                operand=None,
                            )
                            qN2 = jnp.where(should_push, qN + 1, qN)
                            # Advance cur past the segment.
                            new_cur = jnp.where(side_right, seg_hi + 1,
                                                seg_lo - 1)
                            return (new_cur, vis2, qL2, qR2, qN2,
                                    jnp.bool_(False))

                        def skip_seg(_):
                            # seg_lo > seg_hi — nothing to mark; advance past edge.
                            new_cur = jnp.where(side_right, edge + 1, edge - 1)
                            return (new_cur, vis, qL, qR, qN, jnp.bool_(False))

                        return jax.lax.cond(valid_seg, emit_seg, skip_seg,
                                             operand=None)

                    return jax.lax.cond(
                        past_limit,
                        early_return,
                        lambda _: jax.lax.cond(
                            at_limit,
                            at_limit_branch,
                            lambda _: jax.lax.cond(
                                backed_up,
                                backed_up_branch,
                                normal_branch,
                                operand=None,
                            ),
                            operand=None,
                        ),
                        operand=None,
                    )

                return jax.lax.cond(cell_clear, clear_case, opaque_case,
                                     operand=None)

            return jax.lax.cond(cur_in_bounds, in_iter,
                                 lambda _: (cur, vis, qL, qR, qN,
                                            jnp.bool_(True)),
                                 operand=None)

        return jax.lax.cond(done, lambda _: state, do_iter, operand=None)

    init = (left, visible, next_queue_left, next_queue_right, next_queue_count,
            jnp.bool_(False))
    final = jax.lax.fori_loop(0, MAX_INNER, inner_body, init)
    return final[1], final[2], final[3], final[4]


def _view_from_quadrant(
    visible: jnp.ndarray,
    viz_clear: jnp.ndarray,
    left_ptrs: jnp.ndarray,
    right_ptrs: jnp.ndarray,
    srow: jnp.int32,
    scol: jnp.int32,
    start_row: jnp.int32,    # initial row to process
    start_left: jnp.int32,   # initial left mark
    start_right: jnp.int32,  # initial right mark
    step: jnp.int32,         # +1 (down) or -1 (up)
    side_right: jnp.bool_,
    range_val: jnp.int32,    # 0 = unlimited
) -> jnp.ndarray:
    """Process one quadrant (up/down x right/left) outward from source.

    Iterates row-by-row, processing all segments seeded for that row.
    Each segment may push 0..N segments for the next row.
    """
    h, w = visible.shape

    # Row distance from source — used to look up circle_data when range>0.
    # We iterate up to ROWNO rows.
    MAX_ROWS = 21

    # Per-row segment buffer: arrays of size MAX_SEGS_PER_ROW.
    MAX_SEGS = 64
    init_left_buf = jnp.full((MAX_SEGS,), -1, dtype=jnp.int32)
    init_right_buf = jnp.full((MAX_SEGS,), -1, dtype=jnp.int32)
    init_left_buf = init_left_buf.at[0].set(start_left)
    init_right_buf = init_right_buf.at[0].set(start_right)
    init_count = jnp.int32(1)

    # Build circle_data array for range>0 limiting.
    circle_data_arr = jnp.array(_CIRCLE_DATA, dtype=jnp.int32)
    circle_start_arr = jnp.array(_CIRCLE_START, dtype=jnp.int32)

    def row_body(i, state):
        (vis, seg_L, seg_R, seg_N, done) = state

        def do_row(_):
            row = start_row + step * i
            row_in_bounds = (row >= 0) & (row < h)
            # Compute limits for this row based on range.
            # row_dist = abs(row - srow); circle index = circle_start[range] + (row_dist - 1).
            row_dist = jnp.abs(row - srow).astype(jnp.int32)
            # When range=0 (unlimited): lim_max = w-1, lim_min = 0.
            unlimited = range_val == 0
            range_clamped = jnp.clip(range_val, 0, MAX_RADIUS)
            cs_idx = circle_start_arr[range_clamped] + (row_dist - 1)
            cs_idx_safe = jnp.clip(cs_idx, 0, len(_CIRCLE_DATA) - 1)
            limit_delta = circle_data_arr[cs_idx_safe]
            # deeper test (vendor: limits || *limits >= *(limits+1))
            # We compute deeper as: next row is in bounds AND (unlimited OR
            # next row's row_dist <= range).
            next_row_dist = row_dist + 1
            deeper = (
                ((row + step) >= 0) & ((row + step) < h)
                & (unlimited | (next_row_dist.astype(jnp.int32) <= range_val))
            )

            lim_max_full = jnp.where(unlimited, jnp.int32(w - 1),
                                     jnp.minimum(scol + limit_delta,
                                                 jnp.int32(w - 1)))
            lim_min_full = jnp.where(unlimited, jnp.int32(0),
                                     jnp.maximum(scol - limit_delta,
                                                 jnp.int32(0)))

            # Process all segments for this row, collecting new segments for next row.
            next_L = jnp.full((MAX_SEGS,), -1, dtype=jnp.int32)
            next_R = jnp.full((MAX_SEGS,), -1, dtype=jnp.int32)
            next_N = jnp.int32(0)

            def seg_body(j, sstate):
                (v, nL, nR, nN) = sstate
                def do_seg(_):
                    L_j = seg_L[j]   # segment low column
                    R_j = seg_R[j]   # segment high column
                    # _process_segment's first slot ``cur_start`` is the cursor
                    # initial value, second slot ``mark`` is the bound:
                    #   right_side: cursor = seg_lo (sweep up to right_mark=seg_hi)
                    #   left_side:  cursor = seg_hi (sweep down to left_mark=seg_lo)
                    cur_start = jnp.where(side_right, L_j, R_j)
                    mark = jnp.where(side_right, R_j, L_j)
                    # Vendor clamps the mark by limits (right_side: right_mark
                    # by lim_max; left_side: left_mark by lim_min) before the loop.
                    mark = jnp.where(
                        side_right,
                        jnp.minimum(mark, lim_max_full),
                        jnp.maximum(mark, lim_min_full),
                    )
                    return _process_segment(
                        v, viz_clear, left_ptrs, right_ptrs,
                        srow, scol, row, cur_start, mark, step,
                        side_right, lim_max_full, lim_min_full, deeper,
                        nL, nR, nN,
                    )
                return jax.lax.cond(j < seg_N, do_seg,
                                     lambda _: (v, nL, nR, nN),
                                     operand=None)

            (v_out, nL_out, nR_out, nN_out) = jax.lax.cond(
                row_in_bounds,
                lambda _: jax.lax.fori_loop(
                    0, MAX_SEGS, seg_body, (vis, next_L, next_R, next_N)
                ),
                lambda _: (vis, next_L, next_R, jnp.int32(0)),
                operand=None,
            )
            # If no new segments produced (or row was out of bounds), stop.
            stop = (nN_out == 0) | (~deeper) | (~row_in_bounds)
            return (v_out, nL_out, nR_out, nN_out, stop)

        return jax.lax.cond(done | (i >= MAX_ROWS), lambda _: state, do_row,
                             operand=None)

    final = jax.lax.fori_loop(
        0, MAX_ROWS, row_body,
        (visible, init_left_buf, init_right_buf, init_count, jnp.bool_(False)),
    )
    return final[0]


def view_from(
    terrain: jnp.ndarray,
    player_pos: jnp.ndarray,
    max_radius: int = 0,
    opaque_overlay: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """JAX port of vendor's ``view_from`` (vendor/nle/src/vision.c:2640-2731).

    Computes the could_see / IN_SIGHT mask from (srow, scol) using the
    vendor Algorithm C row-segment sweep.  Returns bool[H, W] mask.

    Args:
        terrain: int[H, W] tile grid (current level).
        player_pos: int[2] (row, col).
        max_radius: 0 = unlimited (vendor default); 1..15 limits sight using
            the vendor ``circle_data`` table (vision.c:26-43).
        opaque_overlay: optional bool[H, W] mask whose True cells augment
            the terrain-only opacity (boulders + locked-door overlays per
            vendor ``does_block`` lines 156-184).  Default: all-False.

    Returns:
        bool[H, W] mask, True for cells the vendor would mark IN_SIGHT.

    Vendor algorithm structure:
        1. Build ``viz_clear`` / ``left_ptrs`` / ``right_ptrs`` from opacity
           (vision_reset lines 206-243).
        2. Determine starting-row extent using left_ptrs/right_ptrs at srow.
        3. Sweep up and down, processing one row at a time.  Each row's
           ``right_side`` / ``left_side`` call may recursively spawn calls
           for the next row, one per "finger of light" (visibility segment).
        4. Each finger uses Bresenham ``q?_path`` to find its endpoints.

    JAX implementation:
        Algorithm C's recursion is flattened into 4 quadrant passes (up-right,
        up-left, down-right, down-left).  Within each quadrant, rows are
        processed in order; per-row the active "fingers" are popped from a
        fixed-size buffer, processed via ``_process_segment``, and new
        fingers for the next row are pushed onto a fresh buffer.
    """
    h, w = terrain.shape
    if opaque_overlay is None:
        opaque_overlay = jnp.zeros((h, w), dtype=jnp.bool_)

    terrain_opq = _build_terrain_opaque(terrain)
    opaque = terrain_opq | opaque_overlay

    viz_clear, left_ptrs, right_ptrs = _build_viz_tables(opaque)

    srow = player_pos[0].astype(jnp.int32)
    scol = player_pos[1].astype(jnp.int32)
    range_val = jnp.int32(max_radius)

    # Determine the starting-row extent (vendor view_from lines 2669-2693).
    srow_clamped = jnp.clip(srow, 0, h - 1)
    scol_clamped = jnp.clip(scol, 0, w - 1)
    is_clear_src = viz_clear[srow_clamped, scol_clamped]
    # Left/right from src column (vendor's "if (is_clear(srow, scol))" branch
    # is the dominant one — player typically stands on a clear cell).
    left_src = left_ptrs[srow_clamped, scol_clamped].astype(jnp.int32)
    right_src = right_ptrs[srow_clamped, scol_clamped].astype(jnp.int32)
    # Opaque-source branch (rare; player in stone): "you can only see adjacent
    # squares unless on a boundary or stone/clear boundary" (lines 2672-2683).
    # We replicate by looking at adjacent columns.
    sc_minus_idx = jnp.clip(scol - 1, 0, w - 1)
    sc_plus_idx = jnp.clip(scol + 1, 0, w - 1)
    left_alt = jnp.where(
        scol == 0,
        jnp.int32(0),
        jnp.where(
            viz_clear[srow_clamped, sc_minus_idx],
            left_ptrs[srow_clamped, sc_minus_idx].astype(jnp.int32),
            scol - 1,
        ),
    )
    right_alt = jnp.where(
        scol == w - 1,
        jnp.int32(w - 1),
        jnp.where(
            viz_clear[srow_clamped, sc_plus_idx],
            right_ptrs[srow_clamped, sc_plus_idx].astype(jnp.int32),
            scol + 1,
        ),
    )
    left = jnp.where(is_clear_src, left_src, left_alt)
    right = jnp.where(is_clear_src, right_src, right_alt)

    # Apply range clamp to starting row (lines 2686-2693).
    unlimited = range_val == 0
    left = jnp.where(unlimited, left, jnp.maximum(left, scol - range_val))
    right = jnp.where(unlimited, right, jnp.minimum(right, scol + range_val))

    # Mark source row [left..right] visible.
    visible = jnp.zeros((h, w), dtype=jnp.bool_)
    cols_all = jnp.arange(w, dtype=jnp.int32)
    src_row_mask = (cols_all >= left) & (cols_all <= right)
    visible = visible.at[srow_clamped].set(src_row_mask)

    # Quadrant launches (lines 2716-2730).
    # Down half (step=+1):
    #   right_side(srow+1, scol, right, limits) -- if scol < COLNO-1
    #   left_side (srow+1, left, scol, limits) -- if scol > 0
    # Up half (step=-1):
    #   right_side(srow-1, scol, right, limits)
    #   left_side (srow-1, left, scol, limits)

    # Quadrant seeds.  The buffer stores segments as (seg_lo, seg_hi) where
    # seg_lo <= seg_hi.  The seg_body in _view_from_quadrant decodes per side:
    #   side_right: cursor = seg_lo, bound = seg_hi
    #   side_left : cursor = seg_hi, bound = seg_lo
    # Vendor call sites (lines 2716-2730):
    #   right_side(nrow, scol, right, ...) -> cursor=scol, bound=right -> (scol, right)
    #   left_side (nrow, left,  scol, ...) -> cursor=scol, bound=left  -> (left, scol)
    # Down + right
    visible = _view_from_quadrant(
        visible, viz_clear, left_ptrs, right_ptrs, srow, scol,
        srow + jnp.int32(1), scol, right, jnp.int32(1), jnp.bool_(True),
        range_val,
    )
    # Down + left  (seg_lo = left, seg_hi = scol)
    visible = _view_from_quadrant(
        visible, viz_clear, left_ptrs, right_ptrs, srow, scol,
        srow + jnp.int32(1), left, scol, jnp.int32(1), jnp.bool_(False),
        range_val,
    )
    # Up + right
    visible = _view_from_quadrant(
        visible, viz_clear, left_ptrs, right_ptrs, srow, scol,
        srow - jnp.int32(1), scol, right, jnp.int32(-1), jnp.bool_(True),
        range_val,
    )
    # Up + left
    visible = _view_from_quadrant(
        visible, viz_clear, left_ptrs, right_ptrs, srow, scol,
        srow - jnp.int32(1), left, scol, jnp.int32(-1), jnp.bool_(False),
        range_val,
    )
    return visible
