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
    terrain: jnp.ndarray | None = None,
    los_mask: jnp.ndarray | None = None,
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
        terrain: optional int[h, w] tile grid.  When provided together with
            ``los_mask``, transparent door cells (DOORWAY / OPEN_DOOR) in the
            perimeter ring are gated on Bresenham LOS so vendor-blocked
            doorways stop being flooded as visible.  See "Door gate" below.
        los_mask: optional bool[h, w] of Bresenham LOS from the hero (i.e.
            ``compute_fov(... lit_mask=None)``).  Gates transparent doors when
            paired with ``terrain``.

    Door gate (vendor parity).  Vendor's main-loop branch in vision_recalc
    (vendor/nle/src/vision.c:744-785) treats lit-room perimeter cells in two
    distinct ways:

      * Walls and ``!viz_clear`` doors (closed/locked/trapped) — opaque cells
        whose room-interior neighbour is lit — are made ``IN_SIGHT`` via the
        "fake LOS from inward neighbour" rule (lines 749-774).  Every wall on
        a lit room's bounding ring qualifies; the flood matches that exactly.

      * ``viz_clear`` doors (doorless DOORWAY, OPEN_DOOR) take the else-branch
        at line 776: they receive ``IN_SIGHT`` only when the shadow-caster
        actually reached them (``COULD_SEE`` set in ``next_array``).  A
        diagonally-adjacent wall corner can shadow such a doorway out of
        ``COULD_SEE`` (e.g. seed=4 dlvl 1: hero @ internal (8, 7), HWALL @
        (9, 8) shadows the DOORWAY @ (9, 9), so NLE renders (9, 9) as
        ``S_stone``).  Bresenham LOS catches that case because the ray
        ``(8, 7) -> (9, 9)`` lands on the wall (9, 8) first and stops.

    The gate: a perimeter cell is kept in the flood IFF it is **not** a
    transparent door, OR ``los_mask`` reaches it.  Opaque doors and walls
    are unaffected.  Cite: vendor/nle/src/vision.c::does_block lines 164-167
    (D_CLOSED/D_LOCKED/D_TRAPPED block, D_NODOOR/D_BROKEN do not) and
    vision_recalc lines 744-785 (the two-branch lit/IN_SIGHT decision).

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
    flood = jnp.any(masks, axis=0)

    # Door gate: transparent doors (DOORWAY/OPEN_DOOR) in the flood require
    # actual Bresenham LOS.  See vendor reference in the docstring above.
    # Import here to avoid a circular import with the TileType enum module
    # being itself imported in fov.py at top.
    if terrain is not None and los_mask is not None:
        from Nethax.nethax.constants import TileType
        is_transparent_door = (
            (terrain == jnp.int32(int(TileType.DOORWAY)))
            | (terrain == jnp.int32(int(TileType.OPEN_DOOR)))
        )
        # Keep the cell in the flood unless it's a transparent door without LOS.
        flood = flood & ((~is_transparent_door) | los_mask)
    return flood


def update_explored(
    explored: jnp.ndarray,
    fov: jnp.ndarray,
) -> jnp.ndarray:
    """Return updated explored mask: explored OR fov.

    A tile is permanently marked explored once it enters the player's FOV.
    Equivalent to NetHack's ``levl[x][y].seenv`` flag set in vision_recalc().
    """
    return explored | fov
