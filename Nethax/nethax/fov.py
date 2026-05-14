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
        is_opaque = _OPAQUE_TABLE[tile_idx_clipped]

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
) -> jnp.ndarray:
    """Compute field-of-view mask via Bresenham-line raycast.

    Args:
        terrain: int[H, W] tile grid (current level only).
        player_pos: int[2] (row, col).
        sight_radius: max line-of-sight distance.

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

    # Start with the player's own tile visible.
    visible = jnp.zeros((h, w), dtype=jnp.bool_)
    visible = visible.at[pr, pc].set(True)

    # Enumerate all target offsets on the perimeter of the bounding square.
    # The side length is (2*R+1); offsets range from -R to R inclusive.
    R = sight_radius
    diameter = 2 * R + 1
    # Walk at most R steps per ray. Starting from one tile away from the
    # player (handled in _cast_ray), R iterations cover cells at Chebyshev
    # distance 1..R inclusive. Any extra steps would mark cells beyond
    # sight_radius — that was the Wave 2 first-pass bug.
    max_steps = R

    # Build (dr, dc) pairs for all cells in [-R, R] x [-R, R] excluding (0,0).
    # We cast a ray to every such cell so the FOV is "filled in" — rays to
    # interior cells fill the interior, rays to the border bound the radius.
    rows = jnp.arange(-R, R + 1, dtype=jnp.int32)  # (diameter,)
    cols = jnp.arange(-R, R + 1, dtype=jnp.int32)  # (diameter,)
    dr_grid, dc_grid = jnp.meshgrid(rows, cols, indexing="ij")  # (D, D)
    dr_flat = dr_grid.reshape(-1)   # (D*D,)
    dc_flat = dc_grid.reshape(-1)   # (D*D,)

    # Zero-offset (player tile) is already marked — skip it in the scan body
    # by keeping the cast trivially harmless (ray of length 0).

    def cast_one(vis, idx):
        dr = dr_flat[idx]
        dc = dc_flat[idx]
        # Skip the player's own cell (offset 0,0) — already marked.
        skip = (dr == 0) & (dc == 0)
        # Also skip if offset is beyond sight radius (L∞ check already
        # satisfied by construction, but guard against non-perimeter interior
        # rays that go nowhere useful when dr=dc=0 exactly).
        new_vis = jax.lax.cond(
            skip,
            lambda v: v,
            lambda v: _cast_ray(v, terrain, pr, pc, dr, dc, max_steps),
            vis,
        )
        return new_vis, None

    visible, _ = jax.lax.scan(cast_one, visible, jnp.arange(diameter * diameter))
    return visible


def update_explored(
    explored: jnp.ndarray,
    fov: jnp.ndarray,
) -> jnp.ndarray:
    """Return updated explored mask: explored OR fov.

    A tile is permanently marked explored once it enters the player's FOV.
    Equivalent to NetHack's ``levl[x][y].seenv`` flag set in vision_recalc().
    """
    return explored | fov
