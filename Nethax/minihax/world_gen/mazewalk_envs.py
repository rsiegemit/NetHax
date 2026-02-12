"""World generators for Tier 1 MAZEWALK-based navigation environments.

Implements 5 maze environments from MiniHack .des files:
- mazewalk.des          -> generate_mazewalk
- exploremazeeasy.des   -> generate_explore_maze_easy
- exploremazeeasy_premapped.des -> generate_explore_maze_easy_premapped
- exploremazehard.des   -> generate_explore_maze_hard
- exploremazehard_premapped.des -> generate_explore_maze_hard_premapped

.des coordinate convention: (col, row). Our code: (row, col). Converted here.
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import TileType, ItemType
from Nethax.minihax.states import NavigationState, NavigationStaticParams, GroundItems
from Nethax.minihax.primitives.visibility import compute_visible
from Nethax.minihax.world_gen.procedural import mazewalk, wallification


def _empty_ground_items(max_gi):
    """Create empty GroundItems (no items on ground)."""
    return GroundItems(
        position=jnp.zeros((max_gi, 2), dtype=jnp.int32),
        type_id=jnp.zeros(max_gi, dtype=jnp.int32),
        mask=jnp.zeros(max_gi, dtype=jnp.bool_),
    )


def _place_apples(rng, game_map, col, row_min, row_max, num_apples, map_w, offset):
    """Place apples on random CORRIDOR tiles in a given column.

    Args:
        rng: JAX PRNG key
        game_map: padded map (after _pad_to_static)
        col: column in LOCAL map coords (before offset)
        row_min, row_max: inclusive row range in local coords
        num_apples: number of apples to place (4)
        map_w: local map width
        offset: [row_offset, col_offset] from padding

    Returns:
        GroundItems with num_apples entries
    """
    abs_col = col + offset[1]
    # Place apples at random rows within the valid range
    rngs = jax.random.split(rng, num_apples)
    rows = jax.vmap(lambda k: jax.random.randint(k, (), row_min, row_max + 1))(rngs)
    abs_rows = rows + offset[0]
    positions = jnp.stack([abs_rows, jnp.full(num_apples, abs_col, dtype=jnp.int32)], axis=1)
    return GroundItems(
        position=positions,
        type_id=jnp.full(num_apples, ItemType.APPLE, dtype=jnp.int32),
        mask=jnp.ones(num_apples, dtype=jnp.bool_),
    )


def _make_border_box(map_h, map_w):
    """Create a map with wall borders and VOID interior.

    The original MiniHack .des files do NOT use lvlfill_maze_grid for
    MAZEWALK envs. The MAP places explicit borders, the interior stays
    STONE (VOID), and walkfrom carves corridors through VOID. After
    wallification, only border walls adjacent to corridors survive.
    """
    game_map = jnp.full((map_h, map_w), TileType.VOID, dtype=jnp.int32)

    # Top and bottom walls (HWALL)
    game_map = game_map.at[0, :].set(TileType.HWALL)
    game_map = game_map.at[map_h - 1, :].set(TileType.HWALL)

    # Left and right walls (VWALL)
    game_map = game_map.at[:, 0].set(TileType.VWALL)
    game_map = game_map.at[:, map_w - 1].set(TileType.VWALL)

    return game_map


def _pad_to_static(game_map, active_h, active_w, static_params,
                    row_offset=0, col_offset=0):
    """Pad map into NavigationStaticParams dimensions at given offset.

    Returns (padded_map, row_offset, col_offset).
    """
    sh = static_params.map_height
    sw = static_params.map_width
    padded = jnp.full((sh, sw), TileType.VOID, dtype=jnp.int32)
    padded = padded.at[row_offset:row_offset + active_h,
                       col_offset:col_offset + active_w].set(game_map)
    return padded, row_offset, col_offset


def _pad_to_static_centered(game_map, active_h, active_w, static_params):
    """Pad map centered within NavigationStaticParams dimensions.

    Matches NetHack's GEOMETRY:center,center formula from sp_lev.c:
        xstart = 2 + ((x_maze_max - 2 - xsize) / 2), then ensure odd
        ystart = 2 + ((y_maze_max - 2 - ysize) / 2), then ensure odd

    Returns (padded_map, row_offset, col_offset).
    """
    sh = static_params.map_height   # ROWNO = 21
    sw = static_params.map_width    # COLNO-1 = 79
    y_maze_max = sh - 1             # 20
    x_maze_max = sw                 # 79 (COLNO - 1)
    offset_r = 2 + ((y_maze_max - 2 - active_h) // 2)
    offset_c = 2 + ((x_maze_max - 2 - active_w) // 2)
    # Ensure odd parity (NetHack maze grid alignment)
    offset_r = offset_r | 1
    offset_c = offset_c | 1
    padded = jnp.full((sh, sw), TileType.VOID, dtype=jnp.int32)
    padded = padded.at[offset_r:offset_r + active_h,
                       offset_c:offset_c + active_w].set(game_map)
    return padded, offset_r, offset_c


def _place_stair_random(rng, game_map, tile_type, stair_tile, map_h, map_w):
    """Place a stair on a random CORRIDOR tile.

    Scans all tiles, builds a mask of valid positions, picks one randomly.

    Args:
        rng: JAX PRNG key
        game_map: [map_h, map_w]
        tile_type: tile type to search for placement (CORRIDOR)
        stair_tile: TileType to place (UPSTAIR or DOWNSTAIR)
        map_h, map_w: map dimensions

    Returns:
        (new_map, position_rowcol)
    """
    flat = game_map.reshape(-1)
    is_target = (flat == tile_type)
    # Count valid positions
    n_valid = is_target.sum()
    # Create cumulative indices for valid positions
    cumsum = jnp.cumsum(is_target)
    # Pick random index among valid
    chosen_idx = jax.random.randint(rng, (), 0, jnp.maximum(n_valid, 1))
    # Find the flat index where cumsum == chosen_idx + 1 and is_target
    match = (cumsum == chosen_idx + 1) & is_target
    flat_idx = jnp.argmax(match)
    row = flat_idx // map_w
    col = flat_idx % map_w
    new_map = game_map.at[row, col].set(stair_tile)
    return new_map, jnp.array([row, col], dtype=jnp.int32)


def _place_stair_in_region(rng, game_map, tile_type, stair_tile,
                            col_min, col_max, row_min, row_max, map_w):
    """Place a stair on a random CORRIDOR tile within a rectangular region.

    Args:
        col_min, col_max: inclusive column range
        row_min, row_max: inclusive row range
    """
    flat = game_map.reshape(-1)
    h = game_map.shape[0]
    w = game_map.shape[1]

    # Build position arrays
    indices = jnp.arange(flat.shape[0])
    rows = indices // w
    cols = indices % w

    is_target = (flat == tile_type)
    in_region = (
        (rows >= row_min) & (rows <= row_max) &
        (cols >= col_min) & (cols <= col_max)
    )
    valid = is_target & in_region

    n_valid = valid.sum()
    cumsum = jnp.cumsum(valid)
    chosen_idx = jax.random.randint(rng, (), 0, jnp.maximum(n_valid, 1))
    match = (cumsum == chosen_idx + 1) & valid
    flat_idx = jnp.argmax(match)
    row = flat_idx // w
    col = flat_idx % w
    new_map = game_map.at[row, col].set(stair_tile)
    return new_map, jnp.array([row, col], dtype=jnp.int32)


# ============================================================================
# mazewalk.des: 12x13 room, MAZEWALK from (col=5,row=5), stairs random
# ============================================================================

def generate_mazewalk(rng, params, static_params):
    """Generate a 12x13 maze with random stair placement.

    .des spec:
        MAP: 13 wide x 12 tall bordered room
        MAZEWALK:(5,5),east  -> start at (row=5, col=5)
        STAIR:random,up
        STAIR:random,down
    """
    rng, rng_maze, rng_up, rng_down = jax.random.split(rng, 4)

    active_h, active_w = 12, 13

    # Create bordered room
    game_map = _make_border_box(active_h, active_w)

    # MAZEWALK:(5,5),east — .des col=5, row=5
    # Original: level coords (5+35, 5+5) = (40, 10), step east → (41, 10),
    # parity adjust x=41 (odd, ok), y=10 (even, y--→9) → walkfrom(41, 9)
    # = MAP (row=4, col=6). Even row/col parity for DFS cells.
    start = jnp.array([4, 6], dtype=jnp.int32)
    game_map = mazewalk(rng_maze, game_map, start, active_h, active_w)

    # Remove walls not adjacent to any walkable tile (NetHack's wall_cleanup)
    game_map = wallification(game_map, active_h, active_w)

    # Place upstair (player start) on random floor tile
    game_map, upstair_pos = _place_stair_random(
        rng_up, game_map, TileType.CORRIDOR, TileType.UPSTAIR,
        active_h, active_w
    )

    # Place downstair (goal) on random floor tile (different from upstair)
    # After placing upstair, that tile is UPSTAIR, so picking from FLOOR avoids it
    game_map, downstair_pos = _place_stair_random(
        rng_down, game_map, TileType.CORRIDOR, TileType.DOWNSTAIR,
        active_h, active_w
    )

    # Pad to static params (GEOMETRY:center,center)
    game_map, off_r, off_c = _pad_to_static_centered(
        game_map, active_h, active_w, static_params
    )
    offset = jnp.array([off_r, off_c], dtype=jnp.int32)
    upstair_pos = upstair_pos + offset
    downstair_pos = downstair_pos + offset

    visible_map = compute_visible(upstair_pos, game_map, static_params.map_height, static_params.map_width)
    return NavigationState(
        map=game_map,
        player_position=upstair_pos,
        downstair_position=downstair_pos,
        ground_items=_empty_ground_items(static_params.max_ground_items),
        seen_map=visible_map,
        visible_map=visible_map,
        timestep=0,
        terminal=False,
        state_rng=rng,
    )


# ============================================================================
# exploremazeeasy.des: 21x11 map, vertical divider, two MAZEWALKs
# ============================================================================

def _make_explore_easy_map(rng):
    """Build the explore-maze-easy map (21 wide x 11 tall).

    .des MAP (21 cols x 11 rows including borders):
        Row 0:  |-------------------|      (VWALL + HWALL*19 + VWALL)
        Row 1-9:  |.       ..        .|    left corridor, divider cols 9-10, right corridor
        Row 10: ---------------------      (HWALL*21)

    The divider at columns 9 and 10 is VOID (blocks passage between halves).
    The '.' at cols 1 and 19 (rows 1-9) are pre-placed CORRIDOR markers.

    Two MAZEWALKs carve each half:
        Left:  start at col=2, random row 1..8
        Right: start at col=10, random row 1..8  (actually col=10 in .des)
    """
    active_h, active_w = 11, 21

    game_map = jnp.full((active_h, active_w), TileType.VOID, dtype=jnp.int32)

    # Top border: VWALL at corners, HWALL in between
    game_map = game_map.at[0, 0].set(TileType.VWALL)
    game_map = game_map.at[0, active_w - 1].set(TileType.VWALL)
    game_map = game_map.at[0, 1:active_w - 1].set(TileType.HWALL)

    # Bottom border: all HWALL
    game_map = game_map.at[active_h - 1, :].set(TileType.HWALL)

    # Side walls
    game_map = game_map.at[1:active_h - 1, 0].set(TileType.VWALL)
    game_map = game_map.at[1:active_h - 1, active_w - 1].set(TileType.VWALL)

    # Pre-placed corridor markers at cols 1 and 19 (rows 1-9)
    # .des has '.' at these positions — in original, these are ROOM tiles
    game_map = game_map.at[1:active_h - 1, 1].set(TileType.CORRIDOR)
    game_map = game_map.at[1:active_h - 1, 19].set(TileType.CORRIDOR)

    # Divider at cols 9 and 10: pre-placed CORRIDOR (matching .des '.' markers)
    # These prevent mazewalk from carving across the divider while remaining walkable
    game_map = game_map.at[1:active_h - 1, 9].set(TileType.CORRIDOR)
    game_map = game_map.at[1:active_h - 1, 10].set(TileType.CORRIDOR)

    # Split RNG
    rng, rng_left_start, rng_right_start, rng_maze_l, rng_maze_r = jax.random.split(rng, 5)

    # Left MAZEWALK: .des col=2, row random 1..8
    # In original NetHack, MAP is placed at (xstart=3, ystart=3). Parity
    # adjustment operates on LEVEL coords. With odd offsets, even MAP coords
    # become odd level coords. DFS cells land at even MAP rows/cols:
    #   cols 2,4,6,8 (midpoints 3,5,7) — fills all cols between pre-placed 1 and 9
    #   rows 0,2,4,6,8 (midpoints 1,3,5,7) — leaves row 9 as stone
    # Row: .des [1,8] → level [4,11] → parity adjust → even MAP rows {0,2,4,6,8}
    left_row = jax.random.randint(rng_left_start, (), 1, 9)  # [1, 8] inclusive
    left_row = jnp.maximum(left_row & ~jnp.int32(1), 2)  # even parity, min 2 (avoid border)
    left_start = jnp.array([left_row, 2], dtype=jnp.int32)
    game_map = mazewalk(rng_maze_l, game_map, left_start, active_h, active_w)

    # Right MAZEWALK: .des col=10, row random 1..8
    right_row = jax.random.randint(rng_right_start, (), 1, 9)
    right_row = jnp.maximum(right_row & ~jnp.int32(1), 2)  # even parity, min 2
    right_start = jnp.array([right_row, 10], dtype=jnp.int32)
    game_map = mazewalk(rng_maze_r, game_map, right_start, active_h, active_w)

    # Remove walls not adjacent to any walkable tile (NetHack's wall_cleanup)
    game_map = wallification(game_map, active_h, active_w)

    return rng, game_map, active_h, active_w


def generate_explore_maze_easy(rng, params, static_params):
    """ExploreMazeEasy: 21x11 two-half maze, not premapped.

    .des spec:
        STAIR:(09,01,09,09),(0,0,0,0),down  -> col=9, row 1..9
        BRANCH:(01,01,01,09),(0,0,0,0)      -> col=1, row 1..9 (player start)
        LOOP [4] { OBJECT:('%',"apple"),rndcoord(fillrect(19,1,19,09)) }
            -> 4 apples at col=19, rows 1..9
    """
    rng, rng_stair, rng_branch, rng_apples = jax.random.split(rng, 4)
    rng_rest, game_map, active_h, active_w = _make_explore_easy_map(rng)

    # Place downstair: col=9, row 1..9 on FLOOR
    game_map, downstair_pos = _place_stair_in_region(
        rng_stair, game_map, TileType.CORRIDOR, TileType.DOWNSTAIR,
        col_min=9, col_max=9, row_min=1, row_max=9, map_w=active_w
    )

    # Place upstair (player start): col=1, row 1..9 on FLOOR
    game_map, upstair_pos = _place_stair_in_region(
        rng_branch, game_map, TileType.CORRIDOR, TileType.UPSTAIR,
        col_min=1, col_max=1, row_min=1, row_max=9, map_w=active_w
    )

    # GEOMETRY:left,top without INIT_MAP -> offset (3, 3) per sp_lev.c
    game_map, off_r, off_c = _pad_to_static(
        game_map, active_h, active_w, static_params, row_offset=3, col_offset=3
    )
    offset = jnp.array([off_r, off_c], dtype=jnp.int32)
    upstair_pos = upstair_pos + offset
    downstair_pos = downstair_pos + offset

    # Place 4 apples at col=19, rows 1..9 (local coords)
    ground_items = _place_apples(rng_apples, game_map, col=19, row_min=1, row_max=9,
                                  num_apples=4, map_w=active_w, offset=offset)

    visible_map = compute_visible(upstair_pos, game_map, static_params.map_height, static_params.map_width)
    return NavigationState(
        map=game_map,
        player_position=upstair_pos,
        downstair_position=downstair_pos,
        ground_items=ground_items,
        seen_map=visible_map,
        visible_map=visible_map,
        timestep=0,
        terminal=False,
        state_rng=rng_rest,
    )


def generate_explore_maze_easy_premapped(rng, params, static_params):
    """ExploreMazeEasy premapped -- same generation, flag only."""
    return generate_explore_maze_easy(rng, params, static_params)


# ============================================================================
# exploremazehard.des: 29x15 map, vertical divider, two MAZEWALKs
# ============================================================================

def _make_explore_hard_map(rng):
    """Build the explore-maze-hard map (29 wide x 15 tall).

    .des MAP (29 cols x 15 rows including borders):
        Row 0:  |---------------------------|
        Row 1-13: |.           ..            .|
        Row 14: -----------------------------

    Divider at columns 13 and 14.
    Pre-placed corridor at cols 1 and 27 (rows 1-13).
    """
    active_h, active_w = 15, 29

    game_map = jnp.full((active_h, active_w), TileType.VOID, dtype=jnp.int32)

    # Top border
    game_map = game_map.at[0, 0].set(TileType.VWALL)
    game_map = game_map.at[0, active_w - 1].set(TileType.VWALL)
    game_map = game_map.at[0, 1:active_w - 1].set(TileType.HWALL)

    # Bottom border
    game_map = game_map.at[active_h - 1, :].set(TileType.HWALL)

    # Side walls
    game_map = game_map.at[1:active_h - 1, 0].set(TileType.VWALL)
    game_map = game_map.at[1:active_h - 1, active_w - 1].set(TileType.VWALL)

    # Pre-placed corridor markers at cols 1 and 27 (rows 1-13)
    # .des has '.' at these positions — in original, these are ROOM tiles
    game_map = game_map.at[1:active_h - 1, 1].set(TileType.CORRIDOR)
    game_map = game_map.at[1:active_h - 1, 27].set(TileType.CORRIDOR)

    # Divider at cols 13 and 14: pre-placed CORRIDOR (matching .des '.' markers)
    # These prevent mazewalk from carving across the divider while remaining walkable
    game_map = game_map.at[1:active_h - 1, 13].set(TileType.CORRIDOR)
    game_map = game_map.at[1:active_h - 1, 14].set(TileType.CORRIDOR)

    rng, rng_left_start, rng_right_start, rng_maze_l, rng_maze_r = jax.random.split(rng, 5)

    # Left MAZEWALK: .des col=2, row random 1..13
    # Same parity logic as easy map: even MAP row/col matching original level offsets
    left_row = jax.random.randint(rng_left_start, (), 1, 14)
    left_row = jnp.maximum(left_row & ~jnp.int32(1), 2)  # even parity, min 2 (avoid border)
    left_start = jnp.array([left_row, 2], dtype=jnp.int32)
    game_map = mazewalk(rng_maze_l, game_map, left_start, active_h, active_w)

    # Right MAZEWALK: .des col=14, row random 1..13
    right_row = jax.random.randint(rng_right_start, (), 1, 14)
    right_row = jnp.maximum(right_row & ~jnp.int32(1), 2)  # even parity, min 2
    right_start = jnp.array([right_row, 14], dtype=jnp.int32)
    game_map = mazewalk(rng_maze_r, game_map, right_start, active_h, active_w)

    # Remove walls not adjacent to any walkable tile (NetHack's wall_cleanup)
    game_map = wallification(game_map, active_h, active_w)

    return rng, game_map, active_h, active_w


def generate_explore_maze_hard(rng, params, static_params):
    """ExploreMazeHard: 29x15 two-half maze, not premapped.

    .des spec:
        STAIR:(14,01,14,09),(0,0,0,0),down  -> col=14, row 1..9
        BRANCH:(01,01,01,13),(0,0,0,0)      -> col=1, row 1..13
        LOOP [4] { OBJECT:('%',"apple"),rndcoord(fillrect(27,1,27,13)) }
            -> 4 apples at col=27, rows 1..13
    """
    rng, rng_stair, rng_branch, rng_apples = jax.random.split(rng, 4)
    rng_rest, game_map, active_h, active_w = _make_explore_hard_map(rng)

    # Place downstair: col=14, row 1..9 on FLOOR
    game_map, downstair_pos = _place_stair_in_region(
        rng_stair, game_map, TileType.CORRIDOR, TileType.DOWNSTAIR,
        col_min=14, col_max=14, row_min=1, row_max=9, map_w=active_w
    )

    # Place upstair (player start): col=1, row 1..13 on FLOOR
    game_map, upstair_pos = _place_stair_in_region(
        rng_branch, game_map, TileType.CORRIDOR, TileType.UPSTAIR,
        col_min=1, col_max=1, row_min=1, row_max=13, map_w=active_w
    )

    # GEOMETRY:left,top without INIT_MAP -> offset (3, 3) per sp_lev.c
    game_map, off_r, off_c = _pad_to_static(
        game_map, active_h, active_w, static_params, row_offset=3, col_offset=3
    )
    offset = jnp.array([off_r, off_c], dtype=jnp.int32)
    upstair_pos = upstair_pos + offset
    downstair_pos = downstair_pos + offset

    # Place 4 apples at col=27, rows 1..13 (local coords)
    ground_items = _place_apples(rng_apples, game_map, col=27, row_min=1, row_max=13,
                                  num_apples=4, map_w=active_w, offset=offset)

    visible_map = compute_visible(upstair_pos, game_map, static_params.map_height, static_params.map_width)
    return NavigationState(
        map=game_map,
        player_position=upstair_pos,
        downstair_position=downstair_pos,
        ground_items=ground_items,
        seen_map=visible_map,
        visible_map=visible_map,
        timestep=0,
        terminal=False,
        state_rng=rng_rest,
    )


def generate_explore_maze_hard_premapped(rng, params, static_params):
    """ExploreMazeHard premapped -- same generation, flag only."""
    return generate_explore_maze_hard(rng, params, static_params)
