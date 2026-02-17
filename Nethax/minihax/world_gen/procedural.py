"""Procedural maze generation using MAZEWALK (randomized DFS).

JAX-compatible implementation using lax.while_loop with fixed-size stack.
Replicates NetHack's sp_lev.c MAZEWALK algorithm.
"""
import jax
import jax.numpy as jnp
from jax import lax

from Nethax.minihax.constants import TileType


# Cardinal directions for maze carving: [row_delta, col_delta]
# Each step is 2 tiles (wall + cell)
_MAZE_DIRS = jnp.array([
    [-2, 0],   # North
    [2, 0],    # South
    [0, 2],    # East
    [0, -2],   # West
], dtype=jnp.int32)


def mazewalk(rng, game_map, start_pos, map_h, map_w, carve_tile=TileType.CORRIDOR):
    """Carve a maze using randomized DFS (MAZEWALK from NetHack sp_lev.c).

    Algorithm:
    1. Start at given position, mark as carve_tile
    2. Push onto fixed-size stack
    3. While stack not empty:
       a. Peek at top of stack (current position)
       b. Check 4 cardinal neighbors 2 tiles away -- which are VOID and unvisited?
       c. If any valid: pick random one, carve both the neighbor AND
          the wall between (midpoint) to carve_tile, push neighbor onto stack
       d. If none valid: pop from stack (backtrack)

    Matches NetHack's walkfrom() + okay(): only carves into VOID (STONE) tiles,
    preserving walls, pre-placed tiles, and borders.

    Args:
        rng: JAX PRNG key
        game_map: [map_h, map_w] -- initial map (walls/VOID everywhere except border)
        start_pos: [2] -- (row, col) starting position
        map_h, map_w: ints -- map dimensions
        carve_tile: tile type to carve (default CORRIDOR, sprite 872)

    Returns:
        game_map with carved maze pattern
    """
    max_stack = map_h * map_w

    # Initialize visited array
    visited = jnp.zeros((map_h, map_w), dtype=jnp.bool_)

    # Mark start as visited and carved
    game_map = game_map.at[start_pos[0], start_pos[1]].set(carve_tile)
    visited = visited.at[start_pos[0], start_pos[1]].set(True)

    # Initialize stack with start position
    stack = jnp.zeros((max_stack, 2), dtype=jnp.int32)
    stack = stack.at[0].set(start_pos)
    stack_ptr = jnp.int32(1)  # points to next free slot

    carry_init = (rng, stack, stack_ptr, game_map, visited)

    def cond_fn(carry):
        _, _, stack_ptr, _, _ = carry
        return stack_ptr > 0

    def body_fn(carry):
        rng, stack, stack_ptr, game_map, visited = carry
        rng, rng_perm = jax.random.split(rng)

        # Peek at top of stack
        current = stack[stack_ptr - 1]
        cr, cc = current[0], current[1]

        # Check all 4 neighbors (2 tiles away)
        neighbors = current + _MAZE_DIRS  # [4, 2]
        nr = neighbors[:, 0]
        nc = neighbors[:, 1]

        # Check bounds for each neighbor
        in_bounds = (
            (nr >= 0) & (nr < map_h) &
            (nc >= 0) & (nc < map_w)
        )

        # Check if unvisited and VOID (safe indexing)
        safe_r = jnp.clip(nr, 0, map_h - 1)
        safe_c = jnp.clip(nc, 0, map_w - 1)
        is_unvisited = ~visited[safe_r, safe_c]
        # Only carve into VOID tiles (NetHack's okay() checks typ == STONE)
        is_void = game_map[safe_r, safe_c] == TileType.VOID

        # Valid = in bounds AND unvisited AND destination is VOID
        valid = in_bounds & is_unvisited & is_void  # [4]

        has_valid = valid.any()

        # Shuffle direction order randomly
        perm = jax.random.permutation(rng_perm, 4)

        # Find the first valid direction in shuffled order
        # Use lax.scan to find first valid in permuted order
        shuffled_valid = valid[perm]

        # Pick the first valid index from the shuffled order
        # cumsum trick: first True has cumsum == 1
        cumsum = jnp.cumsum(shuffled_valid)
        first_mask = shuffled_valid & (cumsum == 1)
        # Map back to original direction index
        chosen_shuffled_idx = jnp.argmax(first_mask)
        chosen_dir = perm[chosen_shuffled_idx]

        # Compute neighbor and midpoint for chosen direction
        chosen_neighbor = current + _MAZE_DIRS[chosen_dir]
        midpoint = current + _MAZE_DIRS[chosen_dir] // 2

        # CASE 1: has_valid -- carve and push
        new_game_map_carve = game_map.at[
            chosen_neighbor[0], chosen_neighbor[1]
        ].set(carve_tile)
        new_game_map_carve = new_game_map_carve.at[
            midpoint[0], midpoint[1]
        ].set(carve_tile)

        new_visited_carve = visited.at[
            chosen_neighbor[0], chosen_neighbor[1]
        ].set(True)

        new_stack_carve = stack.at[stack_ptr].set(chosen_neighbor)
        new_stack_ptr_carve = stack_ptr + 1

        # CASE 2: no valid -- pop (backtrack)
        new_stack_ptr_pop = stack_ptr - 1

        # Select between cases
        game_map = jnp.where(has_valid, new_game_map_carve, game_map)
        visited = jnp.where(has_valid, new_visited_carve, visited)
        stack = jnp.where(has_valid, new_stack_carve, stack)
        stack_ptr = jnp.where(has_valid, new_stack_ptr_carve, new_stack_ptr_pop)

        return (rng, stack, stack_ptr, game_map, visited)

    rng, stack, stack_ptr, game_map, visited = lax.while_loop(
        cond_fn, body_fn, carry_init
    )

    return game_map


def wall_cleanup(game_map, map_h, map_w):
    """Remove wall tiles surrounded by solid tiles (NetHack's wall_cleanup).

    A wall tile is converted to VOID if ALL 8 neighbors (including diagonals)
    are "solid" (VOID or any wall type). If any neighbor is non-solid
    (CORRIDOR, FLOOR, door, stair, etc.), the wall survives.
    """
    is_wall = (
        (game_map == TileType.HWALL) | (game_map == TileType.VWALL) |
        (game_map == TileType.TLCORN) | (game_map == TileType.TRCORN) |
        (game_map == TileType.BLCORN) | (game_map == TileType.BRCORN)
    )

    # "Solid" = VOID or any wall type (NetHack's IS_STWALL: STONE through wall types)
    is_solid = (game_map == TileType.VOID) | is_wall

    # Check ALL 8 neighbors; out-of-bounds treated as solid (True)
    padded = jnp.pad(is_solid, 1, constant_values=True)
    all_neighbors_solid = (
        padded[:-2, :-2] &   # NW
        padded[:-2, 1:-1] &  # N
        padded[:-2, 2:] &    # NE
        padded[1:-1, :-2] &  # W
        padded[1:-1, 2:] &   # E
        padded[2:, :-2] &    # SW
        padded[2:, 1:-1] &   # S
        padded[2:, 2:]        # SE
    )

    should_void = is_wall & all_neighbors_solid
    return jnp.where(should_void, TileType.VOID, game_map)


# Spine lookup table: 4-bit mask [N=1, S=2, E=4, W=8] → wall type
# Matches NetHack's spine_array in fix_wall_spines()
_SPINE_MAP = jnp.array([
    TileType.HWALL,   # 0b0000: isolated → HWALL
    TileType.VWALL,   # 0b0001: N only → VWALL
    TileType.VWALL,   # 0b0010: S only → VWALL
    TileType.VWALL,   # 0b0011: N+S → VWALL
    TileType.HWALL,   # 0b0100: E only → HWALL
    TileType.BLCORN,  # 0b0101: N+E → BLCORN (└)
    TileType.TLCORN,  # 0b0110: S+E → TLCORN (┌)
    TileType.VWALL,   # 0b0111: N+S+E → TLWALL → VWALL (no TLWALL type)
    TileType.HWALL,   # 0b1000: W only → HWALL
    TileType.BRCORN,  # 0b1001: N+W → BRCORN (┘)
    TileType.TRCORN,  # 0b1010: S+W → TRCORN (┐)
    TileType.VWALL,   # 0b1011: N+S+W → TRWALL → VWALL (no TRWALL type)
    TileType.HWALL,   # 0b1100: E+W → HWALL
    TileType.HWALL,   # 0b1101: N+E+W → TUWALL → HWALL (no TUWALL type)
    TileType.HWALL,   # 0b1110: S+E+W → TDWALL → HWALL (no TDWALL type)
    TileType.HWALL,   # 0b1111: all → CROSSWALL → HWALL (no CROSSWALL type)
], dtype=jnp.int32)


def fix_wall_spines(game_map, map_h, map_w):
    """Convert wall tiles to proper types based on orthogonal wall neighbors.

    Uses NetHack's spine_array approach: compute 4-bit mask from N/S/E/W
    wall neighbors, look up correct wall type (HWALL, VWALL, or corner).
    """
    is_wall = (
        (game_map == TileType.HWALL) | (game_map == TileType.VWALL) |
        (game_map == TileType.TLCORN) | (game_map == TileType.TRCORN) |
        (game_map == TileType.BLCORN) | (game_map == TileType.BRCORN)
    )

    # Build 4-bit neighbor mask: N=1, S=2, E=4, W=8
    padded = jnp.pad(is_wall, 1, constant_values=False)
    has_N = padded[:-2, 1:-1].astype(jnp.int32)   # north neighbor is wall
    has_S = padded[2:, 1:-1].astype(jnp.int32)     # south
    has_E = padded[1:-1, 2:].astype(jnp.int32)     # east
    has_W = padded[1:-1, :-2].astype(jnp.int32)    # west

    spine_idx = has_N * 1 + has_S * 2 + has_E * 4 + has_W * 8  # [map_h, map_w]

    new_types = _SPINE_MAP[spine_idx]  # lookup for every position

    # Only apply to wall tiles
    return jnp.where(is_wall, new_types, game_map)


def wallify_map(game_map, map_h, map_w):
    """Add walls around walkable tiles (NetHack's wallify_map from sp_lev.c).

    For each VOID (STONE) tile, check 8 neighbors. If any neighbor is a
    walkable/room tile, convert VOID to HWALL (if the neighbor has a different
    row) or VWALL (if same row). Vertical neighbors have priority, matching
    NetHack's scan order (yy outer, xx inner).
    """
    # NetHack IS_ROOM: typ >= ROOM && typ <= WATERTUNNEL
    # CORR (23) is below ROOM (24) — corridors do NOT get wall borders
    is_room = (
        (game_map == TileType.FLOOR) |
        (game_map == TileType.DOWNSTAIR) | (game_map == TileType.UPSTAIR) |
        (game_map == TileType.LAVA)
    )
    is_void = (game_map == TileType.VOID)

    # Pad is_room to handle boundary (out-of-bounds = False)
    padded = jnp.pad(is_room, 1, constant_values=False)

    # Vertical neighbors (yy != y -> HWALL): N, S, NW, NE, SW, SE
    has_vert = (
        padded[:-2, 1:-1] |   # N
        padded[2:, 1:-1] |    # S
        padded[:-2, :-2] |    # NW
        padded[:-2, 2:] |     # NE
        padded[2:, :-2] |     # SW
        padded[2:, 2:]        # SE
    )

    # Horizontal-only neighbors (yy == y -> VWALL): W, E
    has_horiz = (
        padded[1:-1, :-2] |   # W
        padded[1:-1, 2:]      # E
    )

    # VOID + any vertical room neighbor -> HWALL
    # VOID + horizontal-only room neighbor -> VWALL
    new_map = jnp.where(is_void & has_vert, TileType.HWALL, game_map)
    new_map = jnp.where(is_void & ~has_vert & has_horiz, TileType.VWALL, new_map)
    return new_map


def wallification(game_map, map_h, map_w):
    """NetHack's wallification: wallify_map, wall_cleanup, fix_wall_spines."""
    game_map = wallify_map(game_map, map_h, map_w)
    game_map = wall_cleanup(game_map, map_h, map_w)
    game_map = fix_wall_spines(game_map, map_h, map_w)
    return game_map


# ============================================================================
# RANDOM_CORRIDORS: rooms connected by biased random walk corridors (NetHack style)
# ============================================================================

# Maximum number of rooms (fixed array size for JAX)
_MAX_ROOMS = 10
# Maximum rectangles in subdivision pool
_MAX_RECTS = 30
# Minimum gap between rooms (wall-to-wall), matching NetHack's check_room()
XLIM = 4  # Horizontal margin
YLIM = 3  # Vertical margin
_CORRIDOR_KEYS = _MAX_ROOMS * 3  # 3 keys per connection: door1 pos, door2 pos, walk
_MAX_CORRIDOR_STEPS = 200

# Extra random corridors (NetHack makecorridors Phase 2b)
_MAX_EXTRA_CORRIDORS = 13  # max for rn2(10) + 4 when _MAX_ROOMS=10
_EXTRA_CORR_KEYS_PER = 5   # keys per extra corridor: room_a, room_b, door1, door2, walk

# RNG key layout indices
_CONSEC_BASE = _MAX_ROOMS                          # consecutive corridor keys start here
_EXTRA_COUNT_IDX = _CONSEC_BASE + _CORRIDOR_KEYS   # index of the "how many extras" key
_EXTRA_BASE = _EXTRA_COUNT_IDX + 1                  # extra corridor keys start here
_STAIR_BASE = _EXTRA_BASE + _MAX_EXTRA_CORRIDORS * _EXTRA_CORR_KEYS_PER  # stair keys


def _is_door(tile):
    """Check if a tile is any type of door."""
    return (tile == TileType.DOOR_OPEN) | (tile == TileType.DOOR_CLOSED) | (tile == TileType.DOOR_LOCKED)


def _bydoor(game_map, r, c, map_h, map_w):
    """Check if any adjacent tile is already a door (NetHack's bydoor()).

    Returns True if a door exists within Manhattan distance 1.
    """
    sr = jnp.clip(r, 0, map_h - 1)
    sc = jnp.clip(c, 0, map_w - 1)
    above = _is_door(game_map[jnp.clip(r - 1, 0, map_h - 1), sc])
    below = _is_door(game_map[jnp.clip(r + 1, 0, map_h - 1), sc])
    left = _is_door(game_map[sr, jnp.clip(c - 1, 0, map_w - 1)])
    right = _is_door(game_map[sr, jnp.clip(c + 1, 0, map_w - 1)])
    return above | below | left | right


def _okdoor(game_map, r, c, map_h, map_w):
    """Check if a door can be placed at (r, c) (NetHack's okdoor()).

    Valid if: tile is HWALL or VWALL, AND no adjacent door exists.
    """
    sr = jnp.clip(r, 0, map_h - 1)
    sc = jnp.clip(c, 0, map_w - 1)
    tile = game_map[sr, sc]
    is_wall = (tile == TileType.HWALL) | (tile == TileType.VWALL)
    near_door = _bydoor(game_map, r, c, map_h, map_w)
    return is_wall & ~near_door


def _dodoor_tile(rng):
    """Determine door tile type per NetHack's dodoor()/dosdoor() probabilities.

    NetHack mkroom.c dodoor():
      2/3 D_NODOOR (open doorway, walkable)
      1/6 D_ISOPEN (open door, walkable)
      1/6 D_CLOSED (closed door, requires OPEN action)

    Both D_NODOOR and D_ISOPEN map to DOOR_OPEN (walkable).
    D_CLOSED maps to DOOR_CLOSED (blocks movement until opened).
    """
    rng1, rng2 = jax.random.split(rng)
    has_door = jax.random.randint(rng1, (), 0, 3) == 0  # 1/3 chance of actual door
    is_closed = jax.random.randint(rng2, (), 0, 2) == 0  # 50/50 open vs closed
    tile = jnp.where(has_door & is_closed, TileType.DOOR_CLOSED, TileType.DOOR_OPEN)
    return jnp.int32(tile)


def _dig_corridor(rng, game_map, org_r, org_c, dest_r, dest_c, map_h, map_w,
                   nxcor=jnp.bool_(False)):
    """Carve a corridor via biased random walk (NetHack's dig_corridor algorithm).

    The walk starts at (org_r, org_c) and aims for (dest_r, dest_c).
    At each step it carves CORRIDOR through VOID tiles, can pass through
    existing CORRIDOR, and fails on anything else.

    Direction bias: mostly closes the larger remaining distance first,
    with distance-proportional random turns for variety.

    Args:
        rng: JAX PRNG key
        game_map: [map_h, map_w] int32 tile array
        org_r, org_c: start position (one step outside source door, in VOID)
        dest_r, dest_c: end position (one step outside dest door, in VOID)
        map_h, map_w: map dimensions
        nxcor: if True, 1/35 chance of random abort per step (extra corridors)

    Returns:
        (game_map, success): updated map and whether corridor completed
    """
    # Initial direction (NetHack priority: right > down > left > up)
    go_right = dest_c > org_c
    go_down = ~go_right & (dest_r > org_r)
    go_left = ~go_right & ~go_down & (dest_c < org_c)

    init_dc = jnp.where(go_right, jnp.int32(1),
              jnp.where(go_down, jnp.int32(0),
              jnp.where(go_left, jnp.int32(-1), jnp.int32(0))))
    init_dr = jnp.where(go_right, jnp.int32(0),
              jnp.where(go_down, jnp.int32(1),
              jnp.where(go_left, jnp.int32(0), jnp.int32(-1))))

    # Back up one step so first iteration advances to org
    start_r = org_r - init_dr
    start_c = org_c - init_dc

    # State: (rng, r, c, dr, dc, step, game_map, success)
    init_state = (rng, start_r, start_c, init_dr, init_dc,
                  jnp.int32(0), game_map, jnp.bool_(True))

    def cond_fn(state):
        _, r, c, _, _, step, _, success = state
        at_dest = (r == dest_r) & (c == dest_c)
        return success & ~at_dest & (step < _MAX_CORRIDOR_STEPS)

    def body_fn(state):
        rng, r, c, dr, dc, step, game_map, success = state
        rng, k1, k2, k3 = jax.random.split(rng, 4)

        # nxcor random abort: 1/35 chance per step (NetHack's extra corridor flag)
        nxcor_abort = nxcor & (jax.random.randint(k3, (), 0, 35) == 0)
        success = success & ~nxcor_abort

        # Advance
        r = r + dr
        c = c + dc
        step = step + 1

        # Bounds check
        in_bounds = (r >= 1) & (r < map_h - 1) & (c >= 1) & (c < map_w - 1)
        safe_r = jnp.clip(r, 0, map_h - 1)
        safe_c = jnp.clip(c, 0, map_w - 1)
        tile = game_map[safe_r, safe_c]

        # Diggable = VOID or CORRIDOR
        is_void = tile == TileType.VOID
        is_corr = tile == TileType.CORRIDOR
        diggable = (is_void | is_corr) & in_bounds

        # Carve VOID -> CORRIDOR
        new_tile = jnp.where(is_void, TileType.CORRIDOR, tile)
        game_map = game_map.at[safe_r, safe_c].set(
            jnp.where(diggable & success, new_tile, tile)
        )

        # Fail if not diggable
        success = success & diggable

        # Remaining distances
        dix = jnp.abs(c - dest_c)  # horizontal
        diy = jnp.abs(r - dest_r)  # vertical

        # Bias randomization (distance-proportional random override)
        # 1/(dix-diy+1) chance to pretend horizontal distance is 0
        h_gap = jnp.maximum(dix - diy + 1, 1)
        v_gap = jnp.maximum(diy - dix + 1, 1)
        override_h = (dix > diy) & (diy > 0) & (jax.random.randint(k1, (), 0, h_gap) == 0)
        override_v = (diy > dix) & (dix > 0) & (jax.random.randint(k2, (), 0, v_gap) == 0)
        eff_dix = jnp.where(override_h, jnp.int32(0), dix)
        eff_diy = jnp.where(override_v, jnp.int32(0), diy)

        # Direction toward destination
        toward_dc = jnp.where(dest_c > c, jnp.int32(1), jnp.int32(-1))
        toward_dr = jnp.where(dest_r > r, jnp.int32(1), jnp.int32(-1))

        going_v = (dc == 0)   # currently moving vertically
        going_h = (dr == 0)   # currently moving horizontally

        # Helper: is tile at (cr, cc) diggable?
        def _check(cr, cc):
            ib = (cr >= 1) & (cr < map_h - 1) & (cc >= 1) & (cc < map_w - 1)
            sr = jnp.clip(cr, 0, map_h - 1)
            sc = jnp.clip(cc, 0, map_w - 1)
            t = game_map[sr, sc]
            return ib & ((t == TileType.VOID) | (t == TileType.CORRIDOR))

        # --- 4-step direction cascade ---

        # Attempt 1: Turn toward axis with larger remaining distance
        a1_turn_h = going_v & (eff_dix > eff_diy)
        a1_turn_v = going_h & (eff_diy > eff_dix)
        a1_applicable = a1_turn_h | a1_turn_v
        a1_dr = jnp.where(a1_turn_h, jnp.int32(0),
                jnp.where(a1_turn_v, toward_dr, dr))
        a1_dc = jnp.where(a1_turn_h, toward_dc,
                jnp.where(a1_turn_v, jnp.int32(0), dc))
        a1_ok = a1_applicable & _check(r + a1_dr, c + a1_dc)

        # Attempt 2: Continue straight
        a2_ok = _check(r + dr, c + dc)

        # Attempt 3: Turn toward dest on other axis
        a3_dr = jnp.where(going_h, toward_dr, jnp.int32(0))
        a3_dc = jnp.where(going_h, jnp.int32(0), toward_dc)
        a3_ok = _check(r + a3_dr, c + a3_dc)

        # Attempt 4: Reverse of attempt 3
        a4_dr = -a3_dr
        a4_dc = -a3_dc

        # Priority selection
        use1 = a1_ok
        use2 = ~use1 & a2_ok
        use3 = ~use1 & ~use2 & a3_ok

        new_dr = jnp.where(use1, a1_dr,
                 jnp.where(use2, dr,
                 jnp.where(use3, a3_dr, a4_dr)))
        new_dc = jnp.where(use1, a1_dc,
                 jnp.where(use2, dc,
                 jnp.where(use3, a3_dc, a4_dc)))

        # Only update direction if still walking
        at_dest = (r == dest_r) & (c == dest_c)
        dr = jnp.where(~at_dest & success, new_dr, dr)
        dc = jnp.where(~at_dest & success, new_dc, dc)

        return (rng, r, c, dr, dc, step, game_map, success)

    result = lax.while_loop(cond_fn, body_fn, init_state)
    return result[6], result[7]  # game_map, success


def random_corridors(rng, num_rooms, map_h, map_w):
    """Generate a map with rooms placed via rectangle subdivision, connected by
    biased random walk corridors, matching NetHack's makecorridors/dig_corridor
    algorithms.

    Algorithm:
    1. Start with the full map as a single rectangle in a pool.
    2. For each room, pick a random rectangle from the pool, size/place a room
       within it (NetHack-accurate sizes: interior width 3-10, height 3-6,
       area <= 50), then split the containing rectangle into up to 4
       sub-rectangles around the placed room. Early rooms have top-of-map bias
       (NetHack style).
    3. Connect consecutive valid rooms via biased random walk corridors with
       NetHack join() wall selection (priority: right > above > left > below).
    4. Place UPSTAIR in room 0, DOWNSTAIR in room 1.

    All loops use lax.fori_loop with fixed iteration counts for JAX compatibility.

    Args:
        rng: JAX PRNG key
        num_rooms: number of rooms to place (2..10)
        map_h, map_w: padded map dimensions

    Returns:
        game_map: [map_h, map_w] int32 tile array
        player_pos: [2] (row, col) at UPSTAIR
        stair_pos: [2] (row, col) at DOWNSTAIR
    """
    # Pre-split all RNG keys (layout defined by module-level constants):
    #   [0.._MAX_ROOMS-1]: room placement
    #   [_CONSEC_BASE .. +_CORRIDOR_KEYS-1]: consecutive corridor connections
    #   [_EXTRA_COUNT_IDX]: extra corridor count random
    #   [_EXTRA_BASE .. +_MAX_EXTRA_CORRIDORS*5-1]: extra corridor keys
    #   [_STAIR_BASE, _STAIR_BASE+1]: stair placement
    total_keys = _STAIR_BASE + 2
    rngs = jax.random.split(rng, total_keys)

    # Start with VOID map
    game_map = jnp.full((map_h, map_w), TileType.VOID, dtype=jnp.int32)

    # Room storage: rooms[i] = (interior_top, interior_left, interior_h, interior_w)
    rooms = jnp.zeros((_MAX_ROOMS, 4), dtype=jnp.int32)
    # Track which rooms were successfully placed
    room_valid = jnp.zeros(_MAX_ROOMS, dtype=jnp.bool_)

    # Pre-compute index grids (used by room drawing and corridor connection)
    row_idx = jnp.arange(map_h)[:, None]  # [map_h, 1]
    col_idx = jnp.arange(map_w)[None, :]  # [1, map_w]

    # ---- Phase 1: Place rooms via rectangle subdivision (NetHack-style) ----
    # Rectangle pool: (ly, lx, hy, hx) = (top_row, left_col, bottom_row, right_col)
    rects = jnp.zeros((_MAX_RECTS, 4), dtype=jnp.int32)
    rects = rects.at[0].set(jnp.array([0, 0, map_h - 1, map_w - 1]))
    rect_count = jnp.int32(1)

    def place_room_body(i, carry):
        game_map, rooms, room_valid, rects, rect_count = carry
        active = (i < num_rooms) & (rect_count > 0)

        room_rng = rngs[i]

        # ---- Retry loop: try up to 50 placements (NetHack does 100) ----
        # State: (rng, found, attempt, int_top, int_left, int_h, int_w, chosen_rect_idx)
        def retry_cond(state):
            _, found, attempt, _, _, _, _, _ = state
            return active & ~found & (attempt < 50)

        def retry_body(state):
            rng, found, attempt, best_top, best_left, best_h, best_w, best_ridx = state
            rng, k_rect, k_dx, k_dy, k_x, k_y, k_bias = jax.random.split(rng, 7)

            # Pick random rectangle from pool
            ridx = jax.random.randint(k_rect, (), 0, jnp.maximum(rect_count, 1))
            ridx = jnp.minimum(ridx, jnp.maximum(rect_count - 1, 0))
            chosen_rect = rects[ridx]

            ly, lx, hy, hx = chosen_rect[0], chosen_rect[1], chosen_rect[2], chosen_rect[3]
            rect_w = hx - lx
            rect_h = hy - ly

            # Border requirements
            xlim = jnp.int32(XLIM)
            ylim = jnp.int32(YLIM)
            xborder = jnp.where((lx > 0) & (hx < map_w - 1), 2 * xlim, xlim + 1)
            yborder = jnp.where((ly > 0) & (hy < map_h - 1), 2 * ylim, ylim + 1)

            # Room size (NetHack formula)
            wide_rect = rect_w > 28
            dx = jnp.where(wide_rect,
                            2 + jax.random.randint(k_dx, (), 0, 12),
                            2 + jax.random.randint(k_dx, (), 0, 8))
            dy = 2 + jax.random.randint(k_dy, (), 0, 4)
            dy = jnp.where(dx * dy > 50, 50 // dx, dy)

            # Clamp to rectangle
            max_dx = rect_w - 3 - xborder
            max_dy = rect_h - 3 - yborder
            dx = jnp.minimum(dx, jnp.maximum(max_dx, 2))
            dy = jnp.minimum(dy, jnp.maximum(max_dy, 2))
            size_ok = (max_dx >= 2) & (max_dy >= 2)

            int_w = dx + 1
            int_h = dy + 1

            # Position within rectangle
            x_offset = jnp.where(lx > 0, xlim, jnp.int32(3))
            y_offset = jnp.where(ly > 0, ylim, jnp.int32(2))

            x_base = jnp.where(lx > 0, lx, jnp.int32(3))
            y_base = jnp.where(ly > 0, ly, jnp.int32(2))
            x_range = jnp.maximum(hx - x_base - dx - xborder + 1, 1)
            y_range = jnp.maximum(hy - y_base - dy - yborder + 1, 1)

            xabs = lx + x_offset + jax.random.randint(k_x, (), 0, x_range)
            yabs = ly + y_offset + jax.random.randint(k_y, (), 0, y_range)

            # Top-of-map bias (NetHack style)
            full_height = (ly == 0) & (hy >= map_h - 1)
            bias_chance = jnp.where(i == 0, jnp.bool_(True),
                          jax.random.randint(k_bias, (), 0, jnp.maximum(i, 1)) == 0)
            in_bottom = (yabs + dy) > (map_h // 2)
            apply_bias = full_height & bias_chance & in_bottom
            biased_yabs = 2 + jax.random.randint(k_bias, (), 0, 3)
            yabs = jnp.where(apply_bias, biased_yabs, yabs)
            dy = jnp.where(apply_bias & (i < 4) & (dy > 1), dy - 1, dy)
            int_h = dy + 1  # recompute after potential shrink

            # ---- check_room: scan buffer zone for non-STONE tiles ----
            # NetHack scans from (lowx-XLIM, lowy-YLIM) to (hix+XLIM, hiy+YLIM)
            # and checks `typ != STONE`.  Only STONE (our VOID) is acceptable;
            # walls mean there's already a room nearby and ARE a conflict.
            buf_top = jnp.maximum(yabs - YLIM, 0)
            buf_bot = jnp.minimum(yabs + int_h + YLIM, map_h)
            buf_left = jnp.maximum(xabs - XLIM, 0)
            buf_right = jnp.minimum(xabs + int_w + XLIM, map_w)

            in_buffer = ((row_idx >= buf_top) & (row_idx < buf_bot) &
                         (col_idx >= buf_left) & (col_idx < buf_right))
            is_stone = (game_map == TileType.VOID)
            has_conflict = jnp.any(in_buffer & ~is_stone)

            check_ok = size_ok & ~has_conflict

            # Keep best placement if this attempt succeeded (first success wins)
            new_top = jnp.where(check_ok & ~found, yabs, best_top)
            new_left = jnp.where(check_ok & ~found, xabs, best_left)
            new_h = jnp.where(check_ok & ~found, int_h, best_h)
            new_w = jnp.where(check_ok & ~found, int_w, best_w)
            new_ridx = jnp.where(check_ok & ~found, ridx, best_ridx)
            new_found = found | check_ok

            return (rng, new_found, attempt + 1, new_top, new_left, new_h, new_w, new_ridx)

        retry_init = (room_rng, jnp.bool_(False), jnp.int32(0),
                      jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.int32(0))

        result = lax.while_loop(retry_cond, retry_body, retry_init)
        _, found, _, int_top, int_left, int_h, int_w, rect_idx = result

        should_place = active & found

        # Retrieve the chosen rectangle for splitting
        chosen_rect = rects[rect_idx]
        ly, lx, hy, hx = chosen_rect[0], chosen_rect[1], chosen_rect[2], chosen_rect[3]

        # Room expanded bounds (for rectangle splitting) = room + 1 cell border
        r2_ly = int_top - 1
        r2_lx = int_left - 1
        r2_hy = int_top + int_h
        r2_hx = int_left + int_w

        # ---- Draw room ----
        total_h = int_h + 2
        total_w = int_w + 2
        wall_top = int_top - 1
        wall_left = int_left - 1

        in_box = (
            (row_idx >= wall_top) & (row_idx < wall_top + total_h) &
            (col_idx >= wall_left) & (col_idx < wall_left + total_w)
        )
        in_interior = (
            (row_idx >= int_top) & (row_idx < int_top + int_h) &
            (col_idx >= int_left) & (col_idx < int_left + int_w)
        )

        is_top_wall = in_box & (row_idx == wall_top) & (col_idx > wall_left) & (col_idx < wall_left + total_w - 1)
        is_bot_wall = in_box & (row_idx == wall_top + total_h - 1) & (col_idx > wall_left) & (col_idx < wall_left + total_w - 1)
        is_left_wall = in_box & (col_idx == wall_left) & (row_idx > wall_top) & (row_idx < wall_top + total_h - 1)
        is_right_wall = in_box & (col_idx == wall_left + total_w - 1) & (row_idx > wall_top) & (row_idx < wall_top + total_h - 1)

        is_tl = (row_idx == wall_top) & (col_idx == wall_left)
        is_tr = (row_idx == wall_top) & (col_idx == wall_left + total_w - 1)
        is_bl = (row_idx == wall_top + total_h - 1) & (col_idx == wall_left)
        is_br = (row_idx == wall_top + total_h - 1) & (col_idx == wall_left + total_w - 1)

        new_tiles = game_map
        new_tiles = jnp.where(in_interior, TileType.FLOOR, new_tiles)
        new_tiles = jnp.where(is_top_wall, TileType.HWALL, new_tiles)
        new_tiles = jnp.where(is_bot_wall, TileType.HWALL, new_tiles)
        new_tiles = jnp.where(is_left_wall, TileType.VWALL, new_tiles)
        new_tiles = jnp.where(is_right_wall, TileType.VWALL, new_tiles)
        new_tiles = jnp.where(is_tl, TileType.TLCORN, new_tiles)
        new_tiles = jnp.where(is_tr, TileType.TRCORN, new_tiles)
        new_tiles = jnp.where(is_bl, TileType.BLCORN, new_tiles)
        new_tiles = jnp.where(is_br, TileType.BRCORN, new_tiles)

        game_map = jnp.where(should_place, new_tiles, game_map)

        # Store room data
        room_data = jnp.array([int_top, int_left, int_h, int_w])
        new_rooms = rooms.at[i].set(room_data)
        rooms = jnp.where(should_place, new_rooms, rooms)
        new_room_valid = room_valid.at[i].set(True)
        room_valid = jnp.where(should_place, new_room_valid, room_valid.at[i].set(False))

        # ---- Rectangle splitting (NetHack-style: split ALL overlapping rects) ----
        # NetHack's split_rects() recursively splits every rectangle in the pool
        # that overlaps with the placed room, not just the chosen one.  We implement
        # this as a single rebuild pass: for each rect, if it overlaps with the
        # room bounds, replace it with up to 4 non-overlapping sub-rects.
        new_rects = jnp.zeros((_MAX_RECTS, 4), dtype=jnp.int32)
        new_rc = jnp.int32(0)

        def _split_rect_pass(j, carry):
            nr, nc = carry
            is_active = j < rect_count
            r = rects[j]
            ry1, rx1, ry2, rx2 = r[0], r[1], r[2], r[3]

            # Check overlap with room bounds (r2_ly, r2_lx, r2_hy, r2_hx)
            overlaps = ((rx1 <= r2_hx) & (rx2 >= r2_lx) &
                        (ry1 <= r2_hy) & (ry2 >= r2_ly))
            do_split = is_active & overlaps & should_place
            do_keep = is_active & ~do_split

            # Keep: copy rect unchanged
            nr = nr.at[nc].set(r)
            nc = nc + jnp.where(do_keep, jnp.int32(1), jnp.int32(0))

            # Split: add up to 4 sub-rects around the room
            # TOP: rows above room, full width of this rect
            top_r = jnp.array([ry1, rx1, r2_ly - 2, rx2])
            top_space = r2_ly - ry1 - 1
            top_th = jnp.where(ry2 < map_h - 1, 2 * YLIM, YLIM + 1) + 4
            top_ok = do_split & (top_space > top_th) & (nc < _MAX_RECTS)
            nr = nr.at[nc].set(top_r)
            nc = nc + jnp.where(top_ok, jnp.int32(1), jnp.int32(0))

            # LEFT: cols left of room, full height of this rect
            left_r = jnp.array([ry1, rx1, ry2, r2_lx - 2])
            left_space = r2_lx - rx1 - 1
            left_th = jnp.where(rx2 < map_w - 1, 2 * XLIM, XLIM + 1) + 4
            left_ok = do_split & (left_space > left_th) & (nc < _MAX_RECTS)
            nr = nr.at[nc].set(left_r)
            nc = nc + jnp.where(left_ok, jnp.int32(1), jnp.int32(0))

            # BOTTOM: rows below room, full width of this rect
            bot_r = jnp.array([r2_hy + 2, rx1, ry2, rx2])
            bot_space = ry2 - r2_hy - 1
            bot_th = jnp.where(ry1 > 0, 2 * YLIM, YLIM + 1) + 4
            bot_ok = do_split & (bot_space > bot_th) & (nc < _MAX_RECTS)
            nr = nr.at[nc].set(bot_r)
            nc = nc + jnp.where(bot_ok, jnp.int32(1), jnp.int32(0))

            # RIGHT: cols right of room, full height of this rect
            right_r = jnp.array([ry1, r2_hx + 2, ry2, rx2])
            right_space = rx2 - r2_hx - 1
            right_th = jnp.where(rx1 > 0, 2 * XLIM, XLIM + 1) + 4
            right_ok = do_split & (right_space > right_th) & (nc < _MAX_RECTS)
            nr = nr.at[nc].set(right_r)
            nc = nc + jnp.where(right_ok, jnp.int32(1), jnp.int32(0))

            return nr, nc

        new_rects, new_rc = lax.fori_loop(
            0, _MAX_RECTS, _split_rect_pass, (new_rects, new_rc))
        rects = new_rects
        rect_count = new_rc

        return game_map, rooms, room_valid, rects, rect_count

    game_map, rooms, room_valid, _, _ = lax.fori_loop(
        0, _MAX_ROOMS, place_room_body,
        (game_map, rooms, room_valid, rects, rect_count)
    )

    # ---- Phase 2: Connect consecutive rooms via biased random walk ----
    def connect_rooms_body(i, game_map):
        r_i = rooms[i]       # (interior_top, interior_left, interior_h, interior_w)
        r_next = rooms[i + 1]

        # Room i bounds (wall coordinates)
        i_top, i_left, i_h, i_w = r_i[0], r_i[1], r_i[2], r_i[3]
        i_wall_left = i_left - 1
        i_wall_right = i_left + i_w
        i_wall_top = i_top - 1
        i_wall_bot = i_top + i_h

        # Room i+1 bounds (wall coordinates)
        n_top, n_left, n_h, n_w = r_next[0], r_next[1], r_next[2], r_next[3]
        n_wall_left = n_left - 1
        n_wall_right = n_left + n_w
        n_wall_top = n_top - 1
        n_wall_bot = n_top + n_h

        # RNG keys
        base_key = _MAX_ROOMS + i * 3
        rng_d1 = rngs[base_key]
        rng_d2 = rngs[base_key + 1]

        # --- Direction selection (NetHack join() uses interior bounds: lx/hx/ly/hy) ---
        # NetHack priority: right > above > left > below
        i_int_right = i_left + i_w - 1   # croom->hx
        n_int_bottom = n_top + n_h - 1    # troom->hy
        n_int_right = n_left + n_w - 1    # troom->hx
        is_right = n_left > i_int_right              # troom->lx > croom->hx
        is_above = ~is_right & (n_int_bottom < i_top) # troom->hy < croom->ly
        is_left = ~is_right & ~is_above & (n_int_right < i_left)  # troom->hx < croom->lx

        # Direction from room i toward room i+1
        toward_dr = jnp.where(is_right, jnp.int32(0),
                    jnp.where(is_above, jnp.int32(-1),
                    jnp.where(is_left, jnp.int32(0), jnp.int32(1))))
        toward_dc = jnp.where(is_right, jnp.int32(1),
                    jnp.where(is_above, jnp.int32(0),
                    jnp.where(is_left, jnp.int32(-1), jnp.int32(0))))

        # --- Door 1 position (on room i wall facing room i+1) ---
        # Right/Left wall: random row in interior, col = wall col
        # Top/Bottom wall: row = wall row, random col in interior
        d1_row_rl = i_top + jax.random.randint(rng_d1, (), 0, jnp.maximum(i_h, 1))
        d1_col_tb = i_left + jax.random.randint(rng_d1, (), 0, jnp.maximum(i_w, 1))

        d1_r = jnp.where(is_right | is_left, d1_row_rl,
               jnp.where(is_above, i_wall_top, i_wall_bot))
        d1_c = jnp.where(is_right, i_wall_right,
               jnp.where(is_left, i_wall_left,
               d1_col_tb))

        # --- Door 2 position (on room i+1 wall facing room i) ---
        d2_row_rl = n_top + jax.random.randint(rng_d2, (), 0, jnp.maximum(n_h, 1))
        d2_col_tb = n_left + jax.random.randint(rng_d2, (), 0, jnp.maximum(n_w, 1))

        d2_r = jnp.where(is_right | is_left, d2_row_rl,
               jnp.where(is_above, n_wall_bot, n_wall_top))
        d2_c = jnp.where(is_right, n_wall_left,
               jnp.where(is_left, n_wall_right,
               d2_col_tb))

        # --- bydoor check: if chosen position has adjacent door, shift along wall ---
        # Matches NetHack's finddpos() which calls bydoor()
        d1_has_adj = _bydoor(game_map, d1_r, d1_c, map_h, map_w)
        alt_d1_r = jnp.where(is_right | is_left, jnp.minimum(d1_r + 1, i_wall_bot - 1), d1_r)
        alt_d1_c = jnp.where(~(is_right | is_left), jnp.minimum(d1_c + 1, i_wall_right - 1), d1_c)
        d1_r = jnp.where(d1_has_adj, alt_d1_r, d1_r)
        d1_c = jnp.where(d1_has_adj, alt_d1_c, d1_c)

        d2_has_adj = _bydoor(game_map, d2_r, d2_c, map_h, map_w)
        alt_d2_r = jnp.where(is_right | is_left, jnp.minimum(d2_r + 1, n_wall_bot - 1), d2_r)
        alt_d2_c = jnp.where(~(is_right | is_left), jnp.minimum(d2_c + 1, n_wall_right - 1), d2_c)
        d2_r = jnp.where(d2_has_adj, alt_d2_r, d2_r)
        d2_c = jnp.where(d2_has_adj, alt_d2_c, d2_c)

        # --- Corridor start/end (one step outside each door into VOID) ---
        org_r = d1_r + toward_dr
        org_c = d1_c + toward_dc
        dest_r = d2_r - toward_dr
        dest_c = d2_c - toward_dc

        # Only connect if both rooms valid and i < num_rooms - 1
        active = (i < (num_rooms - 1)) & room_valid[i] & room_valid[i + 1]

        # Door type randomization (NetHack dodoor probabilities)
        rng_walk_full = rngs[base_key + 2]
        rng_walk, rng_dt = jax.random.split(rng_walk_full)
        rng_dt1, rng_dt2 = jax.random.split(rng_dt)
        d1_tile = _dodoor_tile(rng_dt1)
        d2_tile = _dodoor_tile(rng_dt2)

        # Dig corridor
        new_map, success = _dig_corridor(rng_walk, game_map, org_r, org_c,
                                          dest_r, dest_c, map_h, map_w)

        # Place doors at wall positions (only if corridor succeeded)
        should_place = active & success
        safe_d1_r = jnp.clip(d1_r, 0, map_h - 1)
        safe_d1_c = jnp.clip(d1_c, 0, map_w - 1)
        safe_d2_r = jnp.clip(d2_r, 0, map_h - 1)
        safe_d2_c = jnp.clip(d2_c, 0, map_w - 1)

        new_map = new_map.at[safe_d1_r, safe_d1_c].set(
            jnp.where(should_place, d1_tile, new_map[safe_d1_r, safe_d1_c]))
        new_map = new_map.at[safe_d2_r, safe_d2_c].set(
            jnp.where(should_place, d2_tile, new_map[safe_d2_r, safe_d2_c]))

        game_map = jnp.where(active, new_map, game_map)

        return game_map

    game_map = lax.fori_loop(0, _MAX_ROOMS - 1, connect_rooms_body, game_map)

    # ---- Phase 2b: Extra random corridors (NetHack makecorridors style) ----
    # NetHack: if (nroom > 2) for (i = rn2(nroom) + 4; i; i--) { ... }
    # Extra corridors only when more than 2 rooms (mklev.c:534)
    n_extra = jnp.where(
        num_rooms > 2,
        jax.random.randint(rngs[_EXTRA_COUNT_IDX], (), 0, jnp.maximum(num_rooms, 1)) + 4,
        jnp.int32(0),
    )

    def extra_corridor_body(i, game_map):
        active = i < n_extra

        # RNG keys for this extra corridor
        base = _EXTRA_BASE + i * _EXTRA_CORR_KEYS_PER
        k_a = rngs[base]
        k_b = rngs[base + 1]
        rng_d1 = rngs[base + 2]
        rng_d2 = rngs[base + 3]

        # Pick random room pair (NetHack style: skip a and a+1)
        a = jax.random.randint(k_a, (), 0, jnp.maximum(num_rooms, 1))
        b_raw = jax.random.randint(k_b, (), 0, jnp.maximum(num_rooms - 2, 1))
        b = jnp.where(b_raw >= a, b_raw + 2, b_raw)
        b = b % jnp.maximum(num_rooms, 1)

        # Get room data
        r_a = rooms[a]
        r_b = rooms[b]

        # Room a bounds (wall coordinates)
        a_top, a_left, a_h, a_w = r_a[0], r_a[1], r_a[2], r_a[3]
        a_wall_left = a_left - 1
        a_wall_right = a_left + a_w
        a_wall_top = a_top - 1
        a_wall_bot = a_top + a_h

        # Room b bounds (wall coordinates)
        b_top, b_left, b_h, b_w = r_b[0], r_b[1], r_b[2], r_b[3]
        b_wall_left = b_left - 1
        b_wall_right = b_left + b_w
        b_wall_top = b_top - 1
        b_wall_bot = b_top + b_h

        # Direction selection (NetHack join() uses interior bounds: lx/hx/ly/hy)
        a_int_right = a_left + a_w - 1   # croom->hx
        b_int_bottom = b_top + b_h - 1    # troom->hy
        b_int_right = b_left + b_w - 1    # troom->hx
        is_right = b_left > a_int_right              # troom->lx > croom->hx
        is_above = ~is_right & (b_int_bottom < a_top) # troom->hy < croom->ly
        is_left = ~is_right & ~is_above & (b_int_right < a_left)  # troom->hx < croom->lx

        toward_dr = jnp.where(is_right, jnp.int32(0),
                    jnp.where(is_above, jnp.int32(-1),
                    jnp.where(is_left, jnp.int32(0), jnp.int32(1))))
        toward_dc = jnp.where(is_right, jnp.int32(1),
                    jnp.where(is_above, jnp.int32(0),
                    jnp.where(is_left, jnp.int32(-1), jnp.int32(0))))

        # Door 1 (on room a wall facing room b)
        d1_row_rl = a_top + jax.random.randint(rng_d1, (), 0, jnp.maximum(a_h, 1))
        d1_col_tb = a_left + jax.random.randint(rng_d1, (), 0, jnp.maximum(a_w, 1))

        d1_r = jnp.where(is_right | is_left, d1_row_rl,
               jnp.where(is_above, a_wall_top, a_wall_bot))
        d1_c = jnp.where(is_right, a_wall_right,
               jnp.where(is_left, a_wall_left, d1_col_tb))

        # Door 2 (on room b wall facing room a)
        d2_row_rl = b_top + jax.random.randint(rng_d2, (), 0, jnp.maximum(b_h, 1))
        d2_col_tb = b_left + jax.random.randint(rng_d2, (), 0, jnp.maximum(b_w, 1))

        d2_r = jnp.where(is_right | is_left, d2_row_rl,
               jnp.where(is_above, b_wall_bot, b_wall_top))
        d2_c = jnp.where(is_right, b_wall_left,
               jnp.where(is_left, b_wall_right, d2_col_tb))

        # Corridor endpoints (one step outside each door)
        org_r = d1_r + toward_dr
        org_c = d1_c + toward_dc
        dest_r = d2_r - toward_dr
        dest_c = d2_c - toward_dc

        # Only connect if both rooms valid
        should_connect = active & room_valid[a] & room_valid[b]

        # nxcor STONE check: first tile outside croom door must be VOID
        # NetHack only checks org side, NOT dest side
        org_tile = game_map[jnp.clip(org_r, 0, map_h - 1), jnp.clip(org_c, 0, map_w - 1)]
        stone_ok = (org_tile == TileType.VOID)
        should_connect = should_connect & stone_ok

        # Door type randomization (NetHack dodoor probabilities)
        rng_walk_full = rngs[base + 4]
        rng_walk, rng_dt = jax.random.split(rng_walk_full)
        rng_dt1, rng_dt2 = jax.random.split(rng_dt)
        d1_tile = _dodoor_tile(rng_dt1)
        d2_tile = _dodoor_tile(rng_dt2)

        # Dig corridor (nxcor=True: 1/35 random abort per step)
        new_map, success = _dig_corridor(rng_walk, game_map, org_r, org_c,
                                          dest_r, dest_c, map_h, map_w,
                                          nxcor=jnp.bool_(True))

        # Apply corridor tiles unconditionally if should_connect
        # (NetHack: corridor tiles persist regardless of door okdoor results)
        game_map = jnp.where(should_connect, new_map, game_map)

        # Place each door independently based on okdoor (nxcor requires okdoor)
        d1_ok = _okdoor(game_map, d1_r, d1_c, map_h, map_w)
        d2_ok = _okdoor(game_map, d2_r, d2_c, map_h, map_w)
        safe_d1_r = jnp.clip(d1_r, 0, map_h - 1)
        safe_d1_c = jnp.clip(d1_c, 0, map_w - 1)
        safe_d2_r = jnp.clip(d2_r, 0, map_h - 1)
        safe_d2_c = jnp.clip(d2_c, 0, map_w - 1)

        # croom door: place if corridor was attempted and okdoor passes
        place_d1 = should_connect & d1_ok
        game_map = game_map.at[safe_d1_r, safe_d1_c].set(
            jnp.where(place_d1, d1_tile, game_map[safe_d1_r, safe_d1_c]))
        # troom door: place if corridor succeeded and okdoor passes
        place_d2 = should_connect & success & d2_ok
        game_map = game_map.at[safe_d2_r, safe_d2_c].set(
            jnp.where(place_d2, d2_tile, game_map[safe_d2_r, safe_d2_c]))

        return game_map

    game_map = lax.fori_loop(0, _MAX_EXTRA_CORRIDORS, extra_corridor_body, game_map)

    # ---- Phase 3: Place stairs ----
    rng_up = rngs[_STAIR_BASE]
    rng_down = rngs[_STAIR_BASE + 1]

    # Place UPSTAIR at random position in room 0 interior
    room0 = rooms[0]
    r0_top, r0_left, r0_h, r0_w = room0[0], room0[1], room0[2], room0[3]
    up_r = r0_top + jax.random.randint(rng_up, (), 0, jnp.maximum(r0_h, 1))
    up_c = r0_left + jax.random.randint(
        jax.random.split(rng_up, 2)[1], (), 0, jnp.maximum(r0_w, 1)
    )
    player_pos = jnp.array([up_r, up_c], dtype=jnp.int32)
    game_map = game_map.at[up_r, up_c].set(TileType.UPSTAIR)

    # Place DOWNSTAIR at random position in room 1 interior
    room1 = rooms[1]
    r1_top, r1_left, r1_h, r1_w = room1[0], room1[1], room1[2], room1[3]
    down_r = r1_top + jax.random.randint(rng_down, (), 0, jnp.maximum(r1_h, 1))
    down_c = r1_left + jax.random.randint(
        jax.random.split(rng_down, 2)[1], (), 0, jnp.maximum(r1_w, 1)
    )
    stair_pos = jnp.array([down_r, down_c], dtype=jnp.int32)
    game_map = game_map.at[down_r, down_c].set(TileType.DOWNSTAIR)

    return game_map, player_pos, stair_pos


# ============================================================================
# REPLACE_TERRAIN: random tile replacement within a region
# ============================================================================

def replace_terrain_random(rng, game_map, region, old_tile, new_tile, percent, map_h, map_w):
    """Replace old_tile with new_tile at given percentage within a region.

    Equivalent to NetHack .des REPLACE_TERRAIN:(r_min_c, r_min_r, r_max_c, r_max_r), old, new, pct%

    Args:
        rng: JAX PRNG key
        game_map: [map_h, map_w] int32 tile array
        region: tuple (r_min, c_min, r_max, c_max) — inclusive bounds (row, col)
        old_tile: int — TileType to replace
        new_tile: int — TileType to replace with
        percent: float — replacement probability (0-100)
        map_h, map_w: int

    Returns:
        new_map: [map_h, map_w] with replacements applied
    """
    r_min, c_min, r_max, c_max = region
    rows = jnp.arange(map_h)[:, None]
    cols = jnp.arange(map_w)[None, :]
    in_region = (rows >= r_min) & (rows <= r_max) & (cols >= c_min) & (cols <= c_max)
    is_old = game_map == old_tile
    rand_vals = jax.random.uniform(rng, (map_h, map_w))
    should_replace = in_region & is_old & (rand_vals < percent / 100.0)
    return jnp.where(should_replace, new_tile, game_map)


# ============================================================================
# RANDLINE: jagged 1-tile-wide line via recursive midpoint displacement
# ============================================================================

def randline(rng, game_map, start, end, roughness, tile, map_h, map_w):
    """Carve a jagged 1-tile-wide line via recursive midpoint displacement.

    Matches NetHack's selection_do_randline: recursively bisects the segment,
    displacing each midpoint randomly. Uses INTEGER arithmetic throughout
    (matching the original C implementation):
    - Integer midpoints via // division
    - Per-sub-segment roughness clamping to segment distance
    - Integer roughness decay: (rough * 2) // 3

    Args:
        rng: JAX PRNG key
        game_map: [map_h, map_w] int32 tile array
        start: [2] (row, col) start point
        end: [2] (row, col) end point
        roughness: int — max random displacement (NOT thickness)
        tile: int — TileType to place
        map_h, map_w: int

    Returns:
        new_map: [map_h, map_w]
    """
    max_depth = 12  # NetHack uses rec=12
    n = 2 ** max_depth  # 4096

    # Integer coordinate arrays (matching NetHack's int coords)
    px = jnp.zeros(n + 1, dtype=jnp.int32)
    py = jnp.zeros(n + 1, dtype=jnp.int32)

    # Set endpoints: start/end are (row, col)
    px = px.at[0].set(start[1])      # col
    py = py.at[0].set(start[0])      # row
    px = px.at[n].set(end[1])
    py = py.at[n].set(end[0])

    # Clamp roughness (NetHack: rough = min(rough, max(|dx|,|dy|)))
    rough = jnp.int32(roughness)
    max_dist = jnp.maximum(
        jnp.abs(end[1] - start[1]),
        jnp.abs(end[0] - start[0])
    )
    rough = jnp.minimum(rough, max_dist)

    # Iterative midpoint displacement (12 levels)
    # Python for loop: level values are compile-time constants
    rng_counter = jnp.int32(0)

    for level in range(max_depth):
        step = n >> level       # compile-time: 4096, 2048, ...
        half = step >> 1        # compile-time: 2048, 1024, ...
        n_segs = 1 << level     # compile-time: 1, 2, 4, ...

        def seg_body(seg, carry, _step=step, _half=half):
            px, py, rng_counter, cur_rough = carry
            left = seg * _step
            mid = left + _half
            right = left + _step

            # Per-sub-segment roughness clamping (NetHack does this per recursive call)
            seg_dist = jnp.maximum(
                jnp.abs(px[right] - px[left]),
                jnp.abs(py[right] - py[left])
            )
            local_rough = jnp.minimum(cur_rough, seg_dist)

            # Integer midpoint (matching C integer division)
            base_x = (px[left] + px[right]) // 2
            base_y = (py[left] + py[right]) // 2

            # Random displacement (only if local_rough >= 2)
            k = jax.random.fold_in(rng, rng_counter)
            k1, k2 = jax.random.split(k)
            r_bound = jnp.maximum(local_rough, 1)
            dx = jax.random.randint(k1, (), 0, r_bound) - (local_rough // 2)
            dy = jax.random.randint(k2, (), 0, r_bound) - (local_rough // 2)

            should_displace = local_rough >= 2
            mx = jnp.where(should_displace, base_x + dx, base_x)
            my = jnp.where(should_displace, base_y + dy, base_y)

            # Clamp to map bounds
            mx = jnp.clip(mx, 0, map_w - 1)
            my = jnp.clip(my, 0, map_h - 1)

            # When endpoints are identical, set midpoint = endpoint (skip marking)
            same_endpoints = (px[left] == px[right]) & (py[left] == py[right])
            mx = jnp.where(same_endpoints, px[left], mx)
            my = jnp.where(same_endpoints, py[left], my)

            px = px.at[mid].set(mx)
            py = py.at[mid].set(my)

            return px, py, rng_counter + 1, cur_rough

        px, py, rng_counter, rough = lax.fori_loop(
            0, n_segs, seg_body, (px, py, rng_counter, rough)
        )
        rough = (rough * 2) // 3  # Integer roughness decay (matching NetHack)

    # Mark all points on the game map
    xi = jnp.clip(px, 0, map_w - 1)
    yi = jnp.clip(py, 0, map_h - 1)
    marked = jnp.zeros((map_h, map_w), dtype=jnp.bool_)
    marked = marked.at[yi, xi].set(True)

    return jnp.where(marked, tile, game_map)
