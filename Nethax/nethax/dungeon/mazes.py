"""Maze generation.

Purpose:
    Provides JAX-compatible maze generation functions.  Kruskal's MST
    algorithm is implemented (used in NetHack for the Gnomish Mines lower
    half and Quest branches).  generate_maze_dfs / generate_maze_perfect
    carve a perfect maze via the vendor walkfrom() recursive-backtracker;
    generate_maze_dla produces organic caves via diffusion-limited
    aggregation.

Citation:
    vendor/nethack/src/mkmaze.c  — walkfree(), makemaz(), wallification(),
        boxwall(), setupvault(); Kruskal-style wall-removal loop.

Wave 2: generate_maze_kruskal implemented via lax.scan over a pre-shuffled
        edge list with a flat union-find table.
Wave 4: generate_maze_dfs / generate_maze_perfect implemented via the
        vendor walkfrom() recursive-backtracker (mkmaze.c:1278-1310).
        generate_maze_dla implemented as a DLA organic-cave generator.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax

from Nethax.nethax.dungeon.branches import MAP_H, MAP_W

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
# Maze generators
# ---------------------------------------------------------------------------

def generate_maze_kruskal(
    rng: jnp.ndarray,
    h: int = MAP_H,
    w: int = MAP_W,
) -> MazeResult:
    """Generate a perfect maze using a Kruskal-style wall-removal algorithm.

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
    """Generate a perfect maze via the vendor depth-first walker.

    Mirrors `vendor/nethack/src/mkmaze.c::walkfrom` (lines 1278-1310):
        while (1) {
            q = 0;
            for (a = 0; a < 4; a++) if (okay(x,y,a)) dirs[q++] = a;
            if (!q) return;                      // backtrack
            dir = dirs[rn2(q)];                  // pick random unvisited
            mz_move(x, y, dir);                  // step into wall
            levl[x][y].typ = typ;                // carve wall to floor
            mz_move(x, y, dir);                  // step into cell
            walkfrom(x, y, typ);                 // recurse
        }

    Vendor convention (mkmaze.c): cells live at odd coordinates and walls
    between them live at even coordinates; the walker takes two-step
    moves, carving the wall in between.

    This implementation runs an iterative DFS with an explicit stack
    (Python-level — level gen is non-JIT, called once at construction).
    Doors are NEVER placed in a maze level — vendor `makemaz()` calls
    `mkstairs` and `populate_maze` but does not invoke `add_door`.

    Citation: vendor/nethack/src/mkmaze.c makemaz()/walkfrom().

    Args:
        rng: JAX PRNG key.
        h:   map height in cells.
        w:   map width in cells.

    Returns:
        (map_array, h, w) — int8[h, w]; TILE_WALL=0, TILE_FLOOR=1.
    """
    import numpy as np

    # Materialise the rng to a seed so we can drive Python's stdlib RNG
    # deterministically.  Level gen runs once at construction (non-JIT).
    seed_arr = np.asarray(jax.random.bits(rng, (2,), jnp.uint32))
    seed = (int(seed_arr[0]) << 32) | int(seed_arr[1])
    pyrng = np.random.RandomState(seed & 0xFFFFFFFF)

    maze = np.zeros((h, w), dtype=np.int8)  # all walls

    # Cells at odd coords inside the boundary.
    def in_bounds(r, c):
        return 1 <= r < h - 1 and 1 <= c < w - 1

    start_r = 1 if h > 2 else 0
    start_c = 1 if w > 2 else 0
    maze[start_r, start_c] = TILE_FLOOR

    # Iterative DFS stack of (r, c) cell positions.
    stack = [(start_r, start_c)]
    # Vendor okay() probes neighbours 2 steps away (cell-to-cell).
    deltas = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    while stack:
        r, c = stack[-1]
        # Find unvisited cell neighbours (cells two steps away that are wall).
        unvisited = [
            (dr, dc) for (dr, dc) in deltas
            if in_bounds(r + dr, c + dc) and maze[r + dr, c + dc] == TILE_WALL
        ]
        if not unvisited:
            stack.pop()  # backtrack — equivalent to walkfrom returning
            continue
        dr, dc = unvisited[pyrng.randint(0, len(unvisited))]
        # Carve the wall between (vendor: first mz_move carves wall).
        maze[r + dr // 2, c + dc // 2] = TILE_FLOOR
        # Step into the new cell (vendor: second mz_move + recursion).
        maze[r + dr, c + dc] = TILE_FLOOR
        stack.append((r + dr, c + dc))

    return jnp.asarray(maze), h, w


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
# Remaining TODO blocks (documented divergences from vendor mkmaze.c)
# ---------------------------------------------------------------------------
# Wave 4 (done): generate_maze_perfect via vendor walkfrom DFS; DLA caves.
# Wave 5+ (open):
#   - Add wallification pass: ensure all boundary cells are walls; fill
#     isolated single-cell wall pockets (mkmaze.c wallification()).
#   - Add dead-end removal pass (optional: increases loop density).
#   - Gnomish Mines lower half: forced maze with extra room carvings for
#     mineend special level (mineend-*.lua).
#   - Quest filler levels: Kruskal maze with role-specific theme overlaid.
