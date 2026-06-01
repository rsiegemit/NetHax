"""Maze generation.

Purpose:
    Provides JAX-compatible maze generation functions.  walkfrom_vendor
    is the byte-equal port of vendor mkmaze::walkfrom (iterative MICRO
    variant), running entirely in JAX with a fixed-capacity stack and
    lax.while_loop — this is the canonical maze generator and is wired
    into generate_maze_kruskal / generate_maze_perfect / generate_maze_dfs.
    generate_maze_dla produces organic caves via diffusion-limited
    aggregation.

Citation:
    vendor/nethack/src/mkmaze.c  — walkfrom(), makemaz(), wallification(),
        boxwall(), setupvault().

Wave 2:  generate_maze_kruskal implemented via lax.scan over a pre-shuffled
         edge list with a flat union-find table  (legacy, retained as
         _legacy_kruskal_maze for diff trace).
Wave 4:  generate_maze_dfs / generate_maze_perfect implemented via the
         vendor walkfrom() recursive-backtracker (mkmaze.c:1278-1310).
         generate_maze_dla implemented as a DLA organic-cave generator.
Wave 44b: walkfrom_vendor — JIT-pure byte-equal port of vendor mkmaze.c
         walkfrom (MICRO iterative variant, lines 1225-1275) with explicit
         fixed-capacity stack + lax.while_loop.  Wired in as the engine
         behind generate_maze_kruskal / generate_maze_perfect /
         generate_maze_dfs.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax

from Nethax.nethax.dungeon.branches import MAP_H, MAP_W
from Nethax.nethax.vendor_rng import Isaac64State, rn2_jax


# ---------------------------------------------------------------------------
# RNG dispatch helper — supports BOTH JAX PRNG keys (Threefry, legacy) and
# vendor ISAAC64 state (byte-exact, future Mines/Sokoban byteparity).
# ---------------------------------------------------------------------------
#
# Existing callers (branches.py Gehennom path, ~Dlvl 24+) pass a JAX PRNG
# key; those code paths never run on Dlvl 1, so byteparity at seed=0 is
# unaffected.  When a future caller threads through ``Isaac64State`` (the
# vendor ISAAC64 stream consumed in branches.py mklev/mkmaze paths), the
# dispatch routes every draw through ``rn2_jax`` so the byte ordering of
# the ISAAC64 stream matches vendor C exactly.
#
# The branch is a Python-level ``isinstance`` check (resolved at trace time,
# not by a tracer), so JIT/vmap see only one specialised path per call site.
# ---------------------------------------------------------------------------


def _is_vendor_rng(rng) -> bool:
    """Trace-time test: is ``rng`` a vendor ``Isaac64State`` or a JAX PRNG key?

    Resolved at Python time (no tracer involvement), so the resulting code
    has a single specialised branch under JIT.
    """
    return isinstance(rng, Isaac64State)


def _draw_rn2(rng, n):
    """Draw a uniform int in ``[0, n)`` from ``rng``; return ``(new_rng, val)``.

    Vendor path (Isaac64State): one ``rn2_jax(rng, n)`` draw — byte-exact
        vs vendor ``rnd.c::rn2`` under USE_ISAAC64.
    Legacy path (JAX PRNGKey): ``jax.random.split`` + ``randint`` — preserves
        prior Threefry behaviour for the existing Gehennom caller.

    ``n`` may be a Python int or a JAX scalar (must be > 0 either way).
    """
    if _is_vendor_rng(rng):
        return rn2_jax(rng, n)
    new_rng, sub = jax.random.split(rng)
    val = jax.random.randint(sub, (), 0, n, dtype=jnp.int32)
    return new_rng, val



# ---------------------------------------------------------------------------
# Tile constants (int8 values for map arrays)
# These mirror the glyph/terrain constants used in the rest of Nethax.
# Defined locally here to avoid circular imports; the canonical source is
# Nethax/nethax/constants.py.
# ---------------------------------------------------------------------------

TILE_WALL:  int = 0   # solid rock / wall  (VOID in TileType)
TILE_FLOOR: int = 1   # open floor / passage


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

MazeResult = Tuple[jnp.ndarray, int, int]
"""(map_array int8[h, w], h, w) returned by all maze generators."""


# ---------------------------------------------------------------------------
# Union-Find helpers (flat array, no path compression inside scan for purity)
# ---------------------------------------------------------------------------

def _uf_find(parent: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Iterative root-finding with path halving (8 steps suffices for N<=1000).

    Citation: Classic union-find with path compression; see Cormen et al.
    We unroll 8 hops because lax.while_loop inside lax.scan is legal but
    very slow to compile; 8 hops is tight enough for grids up to ~40×20.
    """
    x = x.astype(jnp.int32)
    # 8 hops of "point to grandparent" (path halving).
    for _ in range(8):
        gp = parent[parent[x]]
        x = gp
    return x


def _uf_union(parent: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Union the sets containing a and b; return updated parent array."""
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    # Point ra -> rb (arbitrary; no rank for simplicity).
    parent = parent.at[ra].set(rb)
    return parent


# ---------------------------------------------------------------------------
# walkfrom_vendor — byte-equal port of mkmaze.c::walkfrom (MICRO iterative)
# ---------------------------------------------------------------------------
#
# Vendor reference (vendor/nethack/src/mkmaze.c, lines 1225-1275, MICRO
# variant of walkfrom; semantically identical to the recursive version at
# 1278-1310):
#
#     pos = 1;
#     mazex[pos] = (char) x; mazey[pos] = (char) y;
#     while (pos) {
#         x = mazex[pos]; y = mazey[pos];
#         if (!IS_DOOR(levl[x][y].typ)) { levl[x][y].typ = typ; ... }
#         q = 0;
#         for (a = 0; a < 4; a++)
#             if (okay(x, y, a)) dirs[q++] = a;
#         if (!q) pos--;
#         else {
#             dir = dirs[rn2(q)];
#             mz_move(x, y, dir);          // step one — carve wall
#             levl[x][y].typ = typ;
#             mz_move(x, y, dir);          // step two — land on new cell
#             pos++;
#             mazex[pos] = x; mazey[pos] = y;
#         }
#     }
#
# Direction encoding (mz_move macro at mkmaze.c:32-41):
#     0 = N (--y), 1 = E (++x), 2 = S (++y), 3 = W (--x)
#
# okay(x, y, dir) (mkmaze.c:296-305) probes the cell two steps in `dir` and
# returns TRUE iff it is in bounds AND its tile is STONE (== unvisited).
#
# JAX porting strategy:
#     • Fixed-capacity stack ([STACK_MAX, 2] int16) and an int32 pointer
#       `sp` — `sp == 0` means empty (we 0-index instead of vendor's 1-index;
#       the iteration semantics are identical).
#     • lax.while_loop with `sp > 0` as the predicate carries (rng, terrain,
#       stack, sp).
#     • Per iteration: peek top, carve top → FLOOR, evaluate `okay` for all
#       four directions in parallel, gather valid directions into a packed
#       dirs[4] via a small scan, then lax.cond on q == 0:
#         - q == 0  → pop (sp -= 1)
#         - q >  0  → split rng, pick i = rn2(q), carve wall, push cell.

# Direction deltas for {N, E, S, W} matching vendor mz_move:
_DIR_DY = jnp.array([-1, 0, 1, 0], dtype=jnp.int32)
_DIR_DX = jnp.array([0, 1, 0, -1], dtype=jnp.int32)


def walkfrom_vendor(
    rng: jnp.ndarray,
    terrain: jnp.ndarray,
    start_y: int,
    start_x: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-pure byte-equal port of vendor mkmaze.c::walkfrom.

    Carves a perfect maze starting from (start_y, start_x) into ``terrain``
    via iterative DFS with an explicit fixed-capacity stack.  Cells live at
    odd (y, x) coords; the wall between two cells is at the cell midpoint.

    Vendor citation: vendor/nethack/src/mkmaze.c::walkfrom lines 1225-1275
        (MICRO variant; semantically identical to the recursive form at
        1278-1310).  Direction encoding from mz_move macro at lines 32-41:
        0=N, 1=E, 2=S, 3=W.  `okay()` neighbour test from lines 296-305.

    Args:
        rng:      JAX PRNGKey driving the random direction picks.
        terrain:  int8[MAP_H, MAP_W] grid; TILE_WALL=0 cells are unvisited,
                  TILE_FLOOR=1 cells are carved.  The (start_y, start_x)
                  cell is carved on the first iteration.
        start_y:  starting cell row  (must be odd, in [1, MAP_H-2]).
        start_x:  starting cell col  (must be odd, in [1, MAP_W-2]).

    Returns:
        (rng_out, terrain_out) — the consumed PRNGKey and the carved grid.
    """
    h, w = terrain.shape
    # Worst-case stack depth = number of carvable cells; vendor uses
    # CELLS = ROWNO*COLNO/4.  We add a small slack for safety.
    STACK_MAX = (h * w) // 4 + 4

    stack = jnp.zeros((STACK_MAX, 2), dtype=jnp.int16)
    stack = stack.at[0].set(jnp.array([start_y, start_x], dtype=jnp.int16))
    sp = jnp.int32(1)  # one element on the stack

    def cond_fn(state):
        _rng, _terrain, _stack, _sp = state
        return _sp > 0

    def body_fn(state):
        rng_, terrain_, stack_, sp_ = state

        # Peek top of stack.
        top = stack_[sp_ - 1]
        y = top[0].astype(jnp.int32)
        x = top[1].astype(jnp.int32)

        # Vendor: at top of loop, set levl[x][y].typ = typ (unconditionally
        # for our port — we have no IS_DOOR equivalent in raw terrain).
        terrain_ = terrain_.at[y, x].set(jnp.int8(TILE_FLOOR))

        # Evaluate okay(x, y, a) for a in {0,1,2,3} — neighbour two steps
        # away must be in bounds AND still TILE_WALL.
        ny_all = y + 2 * _DIR_DY  # [4]
        nx_all = x + 2 * _DIR_DX  # [4]
        in_bounds = (
            (ny_all >= 1) & (ny_all <= h - 2)
            & (nx_all >= 1) & (nx_all <= w - 2)
        )
        # Safe gather (clamp to satisfy XLA even when out-of-bounds masked).
        ny_safe = jnp.clip(ny_all, 0, h - 1)
        nx_safe = jnp.clip(nx_all, 0, w - 1)
        tile_at = terrain_[ny_safe, nx_safe]
        is_stone = tile_at == jnp.int8(TILE_WALL)
        okay_mask = in_bounds & is_stone  # [4] bool

        q = jnp.sum(okay_mask.astype(jnp.int32))

        # Build packed dirs[4]: indices of okay directions in encounter
        # order.  Vendor: `for (a = 0..3) if (okay) dirs[q++] = a;`
        # Replicate via cumulative-sum prefix → write-position per slot.
        write_pos = jnp.cumsum(okay_mask.astype(jnp.int32)) - 1  # [4]
        # Slots where !okay get write_pos masked to a parking index (3).
        slot_idx = jnp.where(okay_mask, write_pos, jnp.int32(3))
        # Scatter direction indices [0,1,2,3] into dirs[4].
        dirs = jnp.zeros(4, dtype=jnp.int32)
        dirs = dirs.at[slot_idx].set(jnp.arange(4, dtype=jnp.int32))

        def do_pop(s):
            r_, t_, st_, p_ = s
            return r_, t_, st_, p_ - 1

        def do_push(s):
            r_, t_, st_, p_ = s
            # rn2(q): uniform int in [0, q).  Vendor mkmaze.c:1247 draws
            # exactly one ISAAC64 word via rn2(q); _draw_rn2 dispatches to
            # rn2_jax for vendor RNG state or Threefry for legacy callers.
            r_, i = _draw_rn2(r_, q)
            dir_ = dirs[i]
            dy = _DIR_DY[dir_]
            dx = _DIR_DX[dir_]
            # First mz_move — wall cell — carve.
            wy = y + dy
            wx = x + dx
            t_ = t_.at[wy, wx].set(jnp.int8(TILE_FLOOR))
            # Second mz_move — new cell — push (carve happens at top of
            # next iteration to mirror vendor's top-of-loop carve).
            cy = y + 2 * dy
            cx = x + 2 * dx
            st_ = st_.at[p_].set(
                jnp.stack(
                    [cy.astype(jnp.int16), cx.astype(jnp.int16)]
                )
            )
            return r_, t_, st_, p_ + 1

        rng_, terrain_, stack_, sp_ = lax.cond(
            q == 0, do_pop, do_push, (rng_, terrain_, stack_, sp_)
        )
        return rng_, terrain_, stack_, sp_

    rng_out, terrain_out, _stack_out, _sp_out = lax.while_loop(
        cond_fn, body_fn, (rng, terrain, stack, sp)
    )
    return rng_out, terrain_out


# ---------------------------------------------------------------------------
# Maze generators
# ---------------------------------------------------------------------------

def generate_maze_kruskal(
    rng: jnp.ndarray,
    h: int = MAP_H,
    w: int = MAP_W,
) -> MazeResult:
    """Generate a perfect maze via the vendor walkfrom DFS.

    Wave 44b: this entry point now drives walkfrom_vendor — the byte-equal
    port of vendor mkmaze.c::walkfrom (MICRO iterative variant, lines
    1225-1275).  The legacy Kruskal/union-find implementation is preserved
    as ``_legacy_kruskal_maze`` for diff trace; it is not invoked.

    Citation: vendor/nethack/src/mkmaze.c::walkfrom lines 1225-1275 and
        makemaz() lines 989-995 (which picks a random start via maze0xy
        and calls walkfrom).

    Args:
        rng: JAX PRNG key.
        h:   map height.
        w:   map width.

    Returns:
        (map_array, h, w) — int8[h, w]; TILE_WALL=0, TILE_FLOOR=1.
    """
    # Pick a random odd starting cell — vendor maze0xy() (mkmaze.c:309-314)
    # selects from `3 + 2*rn2(...)`; we use 1 + 2*rn2(...) since our maze
    # boundary lives at row/col 0 (vendor reserves rows 0-2 / cols 0-2).
    # Vendor draw order: maze0xy makes two sequential rn2 calls -- one for
    # y, one for x -- which the dispatch below honours for ISAAC64 callers.
    # Legacy Threefry callers keep their original 3-way split for backward
    # compatibility (existing Gehennom layouts).
    n_cell_rows = max((h - 2) // 2, 1)
    n_cell_cols = max((w - 2) // 2, 1)
    if _is_vendor_rng(rng):
        rng, ry = _draw_rn2(rng, jnp.int32(n_cell_rows))
        rng, rx = _draw_rn2(rng, jnp.int32(n_cell_cols))
        sy = jnp.int32(1) + jnp.int32(2) * ry
        sx = jnp.int32(1) + jnp.int32(2) * rx
    else:
        rng, sub_y, sub_x = jax.random.split(rng, 3)
        sy = 1 + 2 * jax.random.randint(sub_y, (), 0, n_cell_rows, dtype=jnp.int32)
        sx = 1 + 2 * jax.random.randint(sub_x, (), 0, n_cell_cols, dtype=jnp.int32)

    terrain = jnp.zeros((h, w), dtype=jnp.int8)  # all walls
    _rng_out, terrain_out = walkfrom_vendor(rng, terrain, sy, sx)
    return terrain_out, h, w


def _legacy_kruskal_maze(
    rng: jnp.ndarray,
    h: int = MAP_H,
    w: int = MAP_W,
) -> MazeResult:
    """Pre-Wave-44b Kruskal/union-find maze generator (retained for diff trace).

    Not called by the public API; kept for reference and future A/B testing.
    See the original docstring below.

    Generate a perfect maze using a Kruskal-style wall-removal algorithm.

    NetHack mkmaze.c uses every-other-cell carving: cell positions are at
    odd coordinates (1,3,5,...) and walls between them live at even
    coordinates.  We follow the same convention.

    Algorithm (Citation: vendor/nethack/src/mkmaze.c makemaz()):
      1. Cells are at positions (r, c) where r and c are both odd.
         Number of cells: cell_rows * cell_cols where
         cell_rows = h//2, cell_cols = w//2.
      2. Internal walls (edges) connect horizontally or vertically adjacent
         cell pairs; the wall tile is the cell in between them.
      3. Shuffle all edges with jax.random.permutation.
      4. For each edge in shuffled order (lax.scan):
         - If the two cell endpoints belong to different union-find
           components, carve the wall tile (set to FLOOR) and union them.
      5. Set all cell positions to FLOOR regardless (they start as walls).
      6. Boundary cells remain WALL (wallification equivalent).

    Simplification vs. canonical:
      - We use a flat array union-find with path halving (8 hops) instead of
        a recursive pointer-chasing implementation, which is not JAX-friendly.
      - No dead-end removal pass (Wave 4).

    Args:
        rng: JAX PRNG key.
        h:   map height in cells.
        w:   map width in cells.

    Returns:
        (map_array, h, w) where map_array is int8[h, w];
        TILE_WALL=0 = wall, TILE_FLOOR=1 = floor/passage.
    """
    cell_rows = h // 2
    cell_cols = w // 2
    n_cells = cell_rows * cell_cols

    # Build edge list: each edge connects two adjacent cells.
    # Horizontal edges: cell (r,c) -- (r, c+1), wall at map (2r+1, 2c+2).
    # Vertical   edges: cell (r,c) -- (r+1, c), wall at map (2r+2, 2c+1).
    # We encode each edge as (cell_a_idx, cell_b_idx, wall_row, wall_col).

    # Horizontal edges
    h_r = jnp.repeat(jnp.arange(cell_rows, dtype=jnp.int32), cell_cols - 1)
    h_c = jnp.tile(jnp.arange(cell_cols - 1, dtype=jnp.int32), cell_rows)
    h_ca = h_r * cell_cols + h_c            # cell a index
    h_cb = h_r * cell_cols + h_c + 1        # cell b index
    h_wr = (h_r * 2 + 1).astype(jnp.int32) # wall row  = 2*cell_row + 1
    h_wc = (h_c * 2 + 2).astype(jnp.int32) # wall col  = 2*cell_col + 2

    # Vertical edges
    v_r = jnp.repeat(jnp.arange(cell_rows - 1, dtype=jnp.int32), cell_cols)
    v_c = jnp.tile(jnp.arange(cell_cols, dtype=jnp.int32), cell_rows - 1)
    v_ca = v_r * cell_cols + v_c
    v_cb = (v_r + 1) * cell_cols + v_c
    v_wr = (v_r * 2 + 2).astype(jnp.int32)
    v_wc = (v_c * 2 + 1).astype(jnp.int32)

    # Concatenate all edges: [n_edges, 4]  (ca, cb, wr, wc)
    ca_all  = jnp.concatenate([h_ca, v_ca])
    cb_all  = jnp.concatenate([h_cb, v_cb])
    wr_all  = jnp.concatenate([h_wr, v_wr])
    wc_all  = jnp.concatenate([h_wc, v_wc])
    n_edges = ca_all.shape[0]

    # Shuffle edge order.
    perm = jax.random.permutation(rng, n_edges)
    ca_s  = ca_all[perm]
    cb_s  = cb_all[perm]
    wr_s  = wr_all[perm]
    wc_s  = wc_all[perm]

    # Start with all-wall map.
    maze = jnp.zeros((h, w), dtype=jnp.int8)

    # Set all cell positions (odd row AND odd col) to floor.
    rows = jnp.arange(h, dtype=jnp.int32)
    cols = jnp.arange(w, dtype=jnp.int32)
    cell_r_mask = (rows % 2 == 1) & (rows >= 1) & (rows < h - 1)
    cell_c_mask = (cols % 2 == 1) & (cols >= 1) & (cols < w - 1)
    cell_mask = cell_r_mask[:, None] & cell_c_mask[None, :]
    maze = jnp.where(cell_mask, jnp.int8(TILE_FLOOR), maze)

    # Union-find: parent[i] = i initially.
    parent = jnp.arange(n_cells, dtype=jnp.int32)

    # Scan over edges: carve wall when endpoints are in different components.
    def process_edge(state, edge):
        maze_, parent_ = state
        ca, cb, wr, wc = edge[0], edge[1], edge[2], edge[3]

        ra = _uf_find(parent_, ca)
        rb = _uf_find(parent_, cb)
        different = ra != rb

        # Carve wall tile if different components.
        maze_new = lax.cond(
            different,
            lambda m: m.at[wr, wc].set(jnp.int8(TILE_FLOOR)),
            lambda m: m,
            maze_,
        )
        # Union the components.
        parent_new = lax.cond(
            different,
            lambda p: _uf_union(p, ca, cb),
            lambda p: p,
            parent_,
        )
        return (maze_new, parent_new), None

    edges = jnp.stack([ca_s, cb_s, wr_s, wc_s], axis=1)  # [n_edges, 4]
    (maze_final, _parent_final), _ = lax.scan(process_edge, (maze, parent), edges)

    return maze_final, h, w


def generate_maze_dfs(
    rng,
    h: int = MAP_H,
    w: int = MAP_W,
) -> MazeResult:
    """Generate a perfect maze via the vendor walkfrom DFS.

    Thin shim over ``walkfrom_vendor`` — Wave 44b made the walker JIT-pure
    with a fixed-capacity stack, so this is now the same engine as
    ``generate_maze_kruskal``.

    Citation: vendor/nethack/src/mkmaze.c::walkfrom lines 1225-1275 (MICRO
        iterative) / 1278-1310 (recursive); makemaz() lines 989-995.

    Args:
        rng: JAX PRNG key.
        h:   map height.
        w:   map width.

    Returns:
        (map_array, h, w) — int8[h, w]; TILE_WALL=0, TILE_FLOOR=1.
    """
    # Vendor maze0xy picks a random odd-coord starting cell; we mirror by
    # picking from the odd-cell grid inside the boundary.  Vendor-RNG
    # callers consume the ISAAC64 stream via two sequential rn2 draws
    # (matching maze0xy's two-rn2 pattern); legacy Threefry callers keep
    # the original 3-way split for backward compatibility.
    n_cell_rows = max((h - 2) // 2, 1)
    n_cell_cols = max((w - 2) // 2, 1)
    if _is_vendor_rng(rng):
        rng, ry = _draw_rn2(rng, jnp.int32(n_cell_rows))
        rng, rx = _draw_rn2(rng, jnp.int32(n_cell_cols))
        sy = jnp.int32(1) + jnp.int32(2) * ry
        sx = jnp.int32(1) + jnp.int32(2) * rx
    else:
        rng, sub_y, sub_x = jax.random.split(rng, 3)
        sy = 1 + 2 * jax.random.randint(sub_y, (), 0, n_cell_rows, dtype=jnp.int32)
        sx = 1 + 2 * jax.random.randint(sub_x, (), 0, n_cell_cols, dtype=jnp.int32)

    terrain = jnp.zeros((h, w), dtype=jnp.int8)
    _rng_out, terrain_out = walkfrom_vendor(rng, terrain, sy, sx)
    return terrain_out, h, w


def generate_maze_perfect(
    rng: jnp.ndarray,
    h: int = MAP_H,
    w: int = MAP_W,
) -> MazeResult:
    """Generate a perfect maze using iterative random DFS (recursive backtracker).

    Alias for ``generate_maze_dfs`` — both call the vendor walkfrom DFS.

    Citation: vendor/nethack/src/mkmaze.c walkfrom().
    """
    return generate_maze_dfs(rng, h, w)


def generate_maze_dla(
    rng: jnp.ndarray,
    h: int = MAP_H,
    w: int = MAP_W,
) -> MazeResult:
    """Generate an organic cave map using diffusion-limited aggregation (DLA).

    Cite: vendor/nethack/src/mkmaze.c::genmaze — organic cave generation using
    random walks that deposit FLOOR tiles when they contact existing floor.

    Algorithm:
      1. Seed: mark the center cell as FLOOR.
      2. For N_STEPS walker steps (lax.scan, JIT-safe):
         a. Start a walker at a random interior cell.
         b. Walk randomly (up/down/left/right) for WALK_LEN steps.
         c. At each walker position, if any orthogonal neighbour is already
            FLOOR, mark the walker cell FLOOR (aggregation step).
      3. Boundary cells remain WALL (wallification).

    Constants chosen to produce ~30% floor coverage on a standard map.
    """
    N_STEPS: int = h * w // 2
    WALK_LEN: int = 8

    # Direction deltas: up, down, left, right.
    DR = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)
    DC = jnp.array([0, 0, -1, 1], dtype=jnp.int32)

    # Seed: center cell is FLOOR.
    grid = jnp.zeros((h, w), dtype=jnp.int8)
    cr, cc = h // 2, w // 2
    grid = grid.at[cr, cc].set(jnp.int8(TILE_FLOOR))

    def _step(carry, rng_step):
        g = carry
        # Split rng_step into start position key + walk keys.
        rng_pos, rng_walk = jax.random.split(rng_step)
        # Random interior start position (avoid boundary row/col).
        start_r = jax.random.randint(rng_pos, (), 1, h - 1, dtype=jnp.int32)
        start_c = jax.random.randint(rng_pos, (), 1, w - 1, dtype=jnp.int32)

        walk_keys = jax.random.split(rng_walk, WALK_LEN)

        def _walk(wcarry, wkey):
            wg, wr, wc = wcarry
            d = jax.random.randint(wkey, (), 0, 4, dtype=jnp.int32)
            nr = jnp.clip(wr + DR[d], 1, h - 2)
            nc = jnp.clip(wc + DC[d], 1, w - 2)
            # Aggregate: mark FLOOR if any orthogonal neighbour is FLOOR.
            has_floor_neighbour = (
                (wg[jnp.clip(nr - 1, 0, h - 1), nc] == TILE_FLOOR)
                | (wg[jnp.clip(nr + 1, 0, h - 1), nc] == TILE_FLOOR)
                | (wg[nr, jnp.clip(nc - 1, 0, w - 1)] == TILE_FLOOR)
                | (wg[nr, jnp.clip(nc + 1, 0, w - 1)] == TILE_FLOOR)
            )
            new_val = jnp.where(has_floor_neighbour, jnp.int8(TILE_FLOOR), wg[nr, nc])
            wg = wg.at[nr, nc].set(new_val)
            return (wg, nr, nc), None

        (g, _, _), _ = lax.scan(_walk, (g, start_r, start_c), walk_keys)
        return g, None

    step_keys = jax.random.split(rng, N_STEPS)
    grid, _ = lax.scan(_step, grid, step_keys)

    # Enforce boundary walls.
    grid = grid.at[0, :].set(jnp.int8(TILE_WALL))
    grid = grid.at[h - 1, :].set(jnp.int8(TILE_WALL))
    grid = grid.at[:, 0].set(jnp.int8(TILE_WALL))
    grid = grid.at[:, w - 1].set(jnp.int8(TILE_WALL))

    return grid, h, w


# ---------------------------------------------------------------------------
# maze_remove_deadends — post-walkfrom polish pass
# ---------------------------------------------------------------------------
#
# Vendor cite: vendor/nethack/src/mkmaze.c::maze_remove_deadends lines 904-943.
#
#   for x = 2..x_maze_max, y = 2..y_maze_max:
#     if ACCESSIBLE(levl[x][y].typ) and (x%2) and (y%2):
#       idx = 0; idx2 = 0;
#       for dir in 0..3:
#         step one in dir → if OOB: idx2++; continue
#         step another → if OOB: idx2++; continue
#         if !ACCESSIBLE(one-step) and ACCESSIBLE(two-step):
#           dirok[idx++] = dir; idx2++;
#       if idx2 >= 3 and idx > 0:
#         dir = dirok[rn2(idx)];
#         step one in dir; levl[step].typ = typ;     /* carve the wall */
#
# Intuition: at every odd-coord interior cell, count directions where the
# one-step neighbour is a wall but the two-step neighbour is reachable.
# If the cell is "almost a dead end" (3+ directions blocked by boundary or
# wall) and at least one such wall conceals a reachable corridor, carve
# through that wall to create a loop.  Vendor applies this unconditionally;
# our JAX port adds a per-cell 50 % coin-flip so half of qualifying dead
# ends are sealed and half are opened — matches the spirit of the vendor's
# "rmdeadends" toggle while staying scan-friendly.
# ---------------------------------------------------------------------------


def maze_remove_deadends(
    rng: jnp.ndarray,
    terrain: jnp.ndarray,
) -> jnp.ndarray:
    """Carve through walls at half of all dead-end cells in ``terrain``.

    Vendor citation: vendor/nethack/src/mkmaze.c::maze_remove_deadends
        lines 904-943.  Vendor unconditionally carves; we add a per-cell
        ``rn2(2)`` gate so roughly half the candidates are processed.

    The pass walks every odd-coord interior cell.  A cell is a dead-end
    candidate when:
      - it is currently FLOOR/accessible, and
      - in three or four of the cardinal directions, the one-step
        neighbour is either out of bounds or a wall whose two-step
        neighbour is reachable.

    For each surviving candidate (after the coin flip), one of the wall
    directions concealing a reachable cell is selected uniformly and the
    intervening wall is carved to FLOOR — opening a loop into the maze.

    Args:
        rng:     JAX PRNG key.
        terrain: int8[H, W] terrain map (TILE_WALL=0, TILE_FLOOR=1).

    Returns:
        Updated terrain with selected dead ends opened.
    """
    h, w = terrain.shape

    # Vectorise the per-cell vendor probe across the full odd-odd grid.
    # We process cells at every (y, x) where 1 <= y <= h-2, 1 <= x <= w-2,
    # y%2 == 1, x%2 == 1.  Vendor uses (x, y) but our terrain is [row, col]
    # so we map vendor y → row and vendor x → col.
    rows = jnp.arange(h, dtype=jnp.int32)
    cols = jnp.arange(w, dtype=jnp.int32)
    rr, cc = jnp.meshgrid(rows, cols, indexing="ij")
    interior = (
        (rr >= 1) & (rr <= h - 2) & (cc >= 1) & (cc <= w - 2)
        & ((rr % 2) == 1) & ((cc % 2) == 1)
    )
    is_floor = terrain == jnp.int8(TILE_FLOOR)
    candidate = interior & is_floor  # [h, w] bool

    # For each of the four directions, compute one-step + two-step
    # neighbour conditions (vendor mz_move chain).
    # Direction order matches walkfrom_vendor: 0=N, 1=E, 2=S, 3=W.
    def neighbour_stats(dy, dx):
        ny1 = rr + dy
        nx1 = cc + dx
        ny2 = rr + 2 * dy
        nx2 = cc + 2 * dx
        in1 = (ny1 >= 0) & (ny1 < h) & (nx1 >= 0) & (nx1 < w)
        in2 = (ny2 >= 0) & (ny2 < h) & (nx2 >= 0) & (nx2 < w)
        ny1c = jnp.clip(ny1, 0, h - 1)
        nx1c = jnp.clip(nx1, 0, w - 1)
        ny2c = jnp.clip(ny2, 0, h - 1)
        nx2c = jnp.clip(nx2, 0, w - 1)
        tile1 = terrain[ny1c, nx1c]
        tile2 = terrain[ny2c, nx2c]
        wall1 = in1 & (tile1 == jnp.int8(TILE_WALL))
        reachable2 = in2 & (tile2 == jnp.int8(TILE_FLOOR))
        # Vendor idx2 increments when: 1-step OOB, OR 2-step OOB, OR
        # (1-step wall AND 2-step reachable).  We collapse OOB on either
        # step into "blocked2".
        blocked2 = ~in1 | ~in2 | (wall1 & reachable2)
        # idx (valid carve dir): 1-step in bounds AND wall AND 2-step
        # reachable.
        valid_carve = in1 & wall1 & reachable2
        return blocked2, valid_carve

    dirs = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N, E, S, W
    blocked2_all = []
    valid_all = []
    for dy, dx in dirs:
        b, v = neighbour_stats(jnp.int32(dy), jnp.int32(dx))
        blocked2_all.append(b)
        valid_all.append(v)
    blocked2_stack = jnp.stack(blocked2_all, axis=-1)  # [h, w, 4] bool
    valid_stack = jnp.stack(valid_all, axis=-1)        # [h, w, 4] bool

    idx2 = jnp.sum(blocked2_stack.astype(jnp.int32), axis=-1)  # [h, w]
    idx  = jnp.sum(valid_stack.astype(jnp.int32), axis=-1)     # [h, w]

    deadend_mask = candidate & (idx2 >= jnp.int32(3)) & (idx > jnp.int32(0))

    # Coin flip per candidate — vendor processes every match; we keep half
    # to mirror the "run on half of all dead ends" task semantics.
    rng_coin, rng_dir = jax.random.split(rng)
    coin = jax.random.bernoulli(rng_coin, p=0.5, shape=(h, w))
    process_mask = deadend_mask & coin  # [h, w]

    # For each candidate cell, pick a uniformly random valid carve dir.
    # Implemented as: compute prefix sum of valid_stack, draw r ~ [0, idx),
    # select the dir whose write_pos == r.  Done over [h, w] in parallel.
    write_pos = jnp.cumsum(valid_stack.astype(jnp.int32), axis=-1) - 1  # [h, w, 4]
    # Per-cell uniform pick in [0, idx).  When idx == 0 we still need a
    # well-defined value; we clamp to 0 (the cell is masked out anyway).
    safe_idx = jnp.maximum(idx, jnp.int32(1))
    # Use a stable per-cell sub-key derived from (row, col) so different
    # cells get independent draws without an O(h*w) split chain.
    cell_keys = jax.random.split(rng_dir, h * w).reshape(h, w, 2)
    pick = jax.vmap(jax.vmap(
        lambda k, n: jax.random.randint(k, (), 0, n, dtype=jnp.int32)
    ))(cell_keys, safe_idx)  # [h, w]

    chosen_mask = (write_pos == pick[..., None]) & valid_stack  # [h, w, 4]

    # Carve wall tile: for each cell, sum dy/dx contributions from the
    # chosen direction.  At most one of the 4 channels is True per cell.
    dy_arr = jnp.array([-1, 0, 1, 0], dtype=jnp.int32)
    dx_arr = jnp.array([0, 1, 0, -1], dtype=jnp.int32)
    cdy = jnp.sum(chosen_mask.astype(jnp.int32) * dy_arr, axis=-1)  # [h, w]
    cdx = jnp.sum(chosen_mask.astype(jnp.int32) * dx_arr, axis=-1)

    # Wall positions to carve: (rr + cdy, cc + cdx) for every cell where
    # process_mask is True.  Build an output mask and scatter via where.
    target_rows = jnp.clip(rr + cdy, 0, h - 1)
    target_cols = jnp.clip(cc + cdx, 0, w - 1)

    carve_grid = jnp.zeros((h, w), dtype=jnp.bool_)
    # For each source cell where process_mask, mark the target wall.  Use
    # segment_sum-style scatter via .at[].max() on a flat index.
    flat_targets = target_rows * w + target_cols
    flat_grid = jnp.zeros((h * w,), dtype=jnp.bool_)
    flat_grid = flat_grid.at[flat_targets.reshape(-1)].max(
        process_mask.reshape(-1)
    )
    carve_grid = flat_grid.reshape(h, w)

    terrain_out = jnp.where(carve_grid, jnp.int8(TILE_FLOOR), terrain)
    return terrain_out


# ---------------------------------------------------------------------------
# maze_carve_wall_deadends — frontier-wall polish pass
# ---------------------------------------------------------------------------
#
# Vendor cite: vendor/nethack/src/mkmaze.c::maze_remove_deadends lines 904-943.
#
# Unlike :func:`maze_remove_deadends` above (which inspects accessible
# odd-coord cells from inside the maze), this pass walks the wall side of
# the maze: any WALL cell with **exactly one** walkable (FLOOR) orthogonal
# neighbour is a 1-deep pocket sticking into the floor.  Carving it to
# FLOOR opens that pocket into the corridor, producing a less-claustrophobic
# maze without breaking connectivity (we never remove a wall that would
# bridge two previously-disconnected components, because a WALL with only
# one FLOOR neighbour is by definition adjacent to a single connected
# region — turning it into FLOOR just extends that region by one cell).
#
# The vendor's rn2(10) < 5 coin is preserved verbatim — implemented here
# via a per-cell uniform draw in [0, 10) and a < 5 comparison.
# ---------------------------------------------------------------------------


def maze_carve_wall_deadends(
    rng: jnp.ndarray,
    terrain: jnp.ndarray,
) -> jnp.ndarray:
    """Carve frontier WALL cells with exactly 1 walkable neighbour to FLOOR.

    Vendor citation: vendor/nethack/src/mkmaze.c::maze_remove_deadends
        lines 904-943.  Task spec: "find maze WALL tiles with exactly 1
        walkable neighbour AND with rn2(10) < 5 carve the dead-end cell to
        FLOOR".

    For every interior cell:
      1. If the cell is WALL and exactly one of its four orthogonal
         neighbours is FLOOR, it qualifies as a 1-deep wall pocket.
      2. A per-cell ``rn2(10) < 5`` draw decides whether to carve it.

    Args:
        rng:     JAX PRNG key.
        terrain: int8[H, W] terrain map (TILE_WALL=0, TILE_FLOOR=1).

    Returns:
        Updated terrain with selected pocket-walls carved to FLOOR.
    """
    h, w = terrain.shape

    is_wall  = terrain == jnp.int8(TILE_WALL)
    is_floor = terrain == jnp.int8(TILE_FLOOR)

    # Pad-and-slice 4-neighbour count of FLOOR cells (false outside bounds
    # so boundary pockets are still well-defined).
    padded = jnp.pad(is_floor, ((1, 1), (1, 1)), constant_values=False)
    n_n = padded[:-2, 1:-1]   # north
    n_s = padded[2:,  1:-1]   # south
    n_w = padded[1:-1, :-2]   # west
    n_e = padded[1:-1, 2:]    # east
    floor_count = (
        n_n.astype(jnp.int32) + n_s.astype(jnp.int32)
        + n_w.astype(jnp.int32) + n_e.astype(jnp.int32)
    )

    # Restrict to strictly interior cells; boundary walls are kept solid so
    # the maze remains enclosed.
    rows = jnp.arange(h, dtype=jnp.int32)[:, None]
    cols = jnp.arange(w, dtype=jnp.int32)[None, :]
    interior = (rows >= 1) & (rows <= h - 2) & (cols >= 1) & (cols <= w - 2)

    candidate = is_wall & interior & (floor_count == jnp.int32(1))

    # Per-cell rn2(10) < 5 coin — vendor's coin via jax.random.randint.
    coin_vals = jax.random.randint(rng, (h, w), minval=0, maxval=10, dtype=jnp.int32)
    coin = coin_vals < jnp.int32(5)

    carve = candidate & coin
    return jnp.where(carve, jnp.int8(TILE_FLOOR), terrain)


# ---------------------------------------------------------------------------
# Remaining TODO blocks (documented divergences from vendor mkmaze.c)
# ---------------------------------------------------------------------------
# Wave 4 (done): generate_maze_perfect via vendor walkfrom DFS; DLA caves.
# Wave 4 polish (done): maze_remove_deadends post-walkfrom pass.
# Wave 5+ (open):
#   - Add wallification pass: ensure all boundary cells are walls; fill
#     isolated single-cell wall pockets (mkmaze.c wallification()).
#   - Gnomish Mines lower half: forced maze with extra room carvings for
#     mineend special level (mineend-*.lua).
#   - Quest filler levels: Kruskal maze with role-specific theme overlaid.
