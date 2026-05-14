"""Wave 6 Closing-Audit #82 — vendor-parity tests for dungeon generation.

Each test asserts a specific invariant from the NetHack vendor sources
(`mklev.c`, `mkroom.c`, `mkmaze.c`, `sp_lev.c`).  Vendor citations are
inline in the assertion docstrings.

Test command (SCOPED):
    .venv/bin/python -m pytest \
        tests/test_dungeon_gen_parity.py tests/test_dungeon_gen.py -v --timeout=180
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_np(arr):
    return arr.__array__() if hasattr(arr, "__array__") else np.asarray(arr)


def _active_indices(active_np):
    return [i for i in range(len(active_np)) if bool(active_np[i])]


# ---------------------------------------------------------------------------
# Part A — Rooms
# ---------------------------------------------------------------------------


def test_room_count_per_level_in_5_to_9():
    """Vendor `mklev.c::makerooms` (lines 403-426) places ~5..9 rooms per level.

    With n_rooms=-1 sentinel we draw a vendor-style random target in [5, 9];
    all 10 seeds should produce that many active rooms.
    """
    import jax
    from Nethax.nethax.dungeon.rooms import generate_rooms

    counts = []
    for seed in range(10):
        rng = jax.random.PRNGKey(seed)
        _rooms, active = generate_rooms(rng, 21, 80, n_rooms=-1)
        n_active = int(np.asarray(_to_np(active)).sum())
        counts.append(n_active)
        assert 5 <= n_active <= 9, (
            f"seed={seed}: got {n_active} active rooms, expected 5..9"
        )
    # At least some variation across 10 seeds
    assert len(set(counts)) >= 2, f"no variation across seeds: counts={counts}"


def test_room_min_size_2x2():
    """Vendor `sp_lev.c::create_room` line 1548-1549: dx=2+rn2(8), dy=2+rn2(4).

    Interior min size is therefore 2×2 cells.
    """
    import jax
    from Nethax.nethax.dungeon.rooms import generate_rooms

    rng = jax.random.PRNGKey(0)
    rooms, active = generate_rooms(rng, 21, 80, n_rooms=-1)
    active_np = _to_np(active)
    y1, x1, y2, x2 = (_to_np(rooms.y1), _to_np(rooms.x1),
                      _to_np(rooms.y2), _to_np(rooms.x2))

    for i in _active_indices(active_np):
        h_int = int(y2[i]) - int(y1[i]) + 1
        w_int = int(x2[i]) - int(x1[i]) + 1
        assert h_int >= 2, f"room {i}: interior height {h_int} < 2"
        assert w_int >= 2, f"room {i}: interior width {w_int} < 2"


def test_room_max_size_within_bounds():
    """Vendor: max interior width = 9 (2+rn2(8)), max height = 5 (2+rn2(4)).

    Also: all active rooms must fit inside the (h-2, w-2) interior of the map
    so the perimeter wall ring lies on valid cells.
    """
    import jax
    from Nethax.nethax.dungeon.rooms import generate_rooms

    h, w = 21, 80
    for seed in range(5):
        rng = jax.random.PRNGKey(seed)
        rooms, active = generate_rooms(rng, h, w, n_rooms=-1)
        active_np = _to_np(active)
        y1, x1, y2, x2 = (_to_np(rooms.y1), _to_np(rooms.x1),
                          _to_np(rooms.y2), _to_np(rooms.x2))
        for i in _active_indices(active_np):
            ih = int(y2[i]) - int(y1[i]) + 1
            iw = int(x2[i]) - int(x1[i]) + 1
            assert ih <= 5, f"seed={seed} room {i}: interior height {ih} > vendor max 5"
            assert iw <= 9, f"seed={seed} room {i}: interior width {iw} > vendor max 9"
            assert int(y1[i]) >= 1 and int(y2[i]) <= h - 2
            assert int(x1[i]) >= 1 and int(x2[i]) <= w - 2


def test_rooms_do_not_overlap():
    """Vendor `mklev.c::check_room` rejects overlapping placements.

    Two rooms (including 1-cell wall margin) must not intersect.
    """
    import jax
    from Nethax.nethax.dungeon.rooms import generate_rooms

    for seed in range(5):
        rng = jax.random.PRNGKey(seed)
        rooms, active = generate_rooms(rng, 21, 80, n_rooms=-1)
        active_np = _to_np(active)
        y1, x1, y2, x2 = (_to_np(rooms.y1), _to_np(rooms.x1),
                          _to_np(rooms.y2), _to_np(rooms.x2))
        idxs = _active_indices(active_np)
        margin = 1
        for pos, i in enumerate(idxs):
            for j in idxs[pos + 1:]:
                no_overlap = (
                    int(x2[i]) + margin < int(x1[j]) or
                    int(x2[j]) + margin < int(x1[i]) or
                    int(y2[i]) + margin < int(y1[j]) or
                    int(y2[j]) + margin < int(y1[i])
                )
                assert no_overlap, f"seed={seed} rooms {i},{j} overlap"


def test_room_walls_present_around_perimeter():
    """Vendor `mklev.c::do_room_or_subroom` (lines 277-296) stamps HWALL/VWALL/
    corner tiles around every room's interior bounding box.

    After `carve_rooms_into_terrain`, every cell in the 1-ring perimeter of
    each active room should be WALL (tile value 3).
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.rooms import (
        generate_rooms, carve_rooms_into_terrain,
    )

    rng = jax.random.PRNGKey(7)
    rooms, active = generate_rooms(rng, 21, 80, n_rooms=-1)
    terrain = jnp.zeros((21, 80), dtype=jnp.int8)
    terrain = carve_rooms_into_terrain(terrain, rooms, active)

    terrain_np = _to_np(terrain)
    active_np = _to_np(active)
    y1, x1, y2, x2 = (_to_np(rooms.y1), _to_np(rooms.x1),
                      _to_np(rooms.y2), _to_np(rooms.x2))

    WALL = 3
    for i in _active_indices(active_np):
        ay1, ax1 = int(y1[i]) - 1, int(x1[i]) - 1
        ay2, ax2 = int(y2[i]) + 1, int(x2[i]) + 1
        for r in range(ay1, ay2 + 1):
            for c in range(ax1, ax2 + 1):
                interior = (int(y1[i]) <= r <= int(y2[i]) and
                            int(x1[i]) <= c <= int(x2[i]))
                if interior:
                    continue
                assert terrain_np[r, c] == WALL, (
                    f"room {i}: perimeter cell ({r},{c}) = {terrain_np[r,c]} "
                    f"(want {WALL})"
                )


# ---------------------------------------------------------------------------
# Part B — Corridors
# ---------------------------------------------------------------------------


def _make_room_pair(y1a, x1a, y2a, x2a, y1b, x1b, y2b, x2b):
    """Build a 2-room Room pytree + active mask for corridor testing."""
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.rooms import Room, MAX_ROOMS_PER_LEVEL

    y1 = np.full(MAX_ROOMS_PER_LEVEL, -1, dtype=np.int16)
    x1 = np.full(MAX_ROOMS_PER_LEVEL, -1, dtype=np.int16)
    y2 = np.full(MAX_ROOMS_PER_LEVEL, -1, dtype=np.int16)
    x2 = np.full(MAX_ROOMS_PER_LEVEL, -1, dtype=np.int16)
    y1[0], x1[0], y2[0], x2[0] = y1a, x1a, y2a, x2a
    y1[1], x1[1], y2[1], x2[1] = y1b, x1b, y2b, x2b
    active = np.zeros(MAX_ROOMS_PER_LEVEL, dtype=bool)
    active[0] = True
    active[1] = True

    room_type = np.zeros(MAX_ROOMS_PER_LEVEL, dtype=np.int8)
    is_lit = np.zeros(MAX_ROOMS_PER_LEVEL, dtype=bool)
    rooms = Room(
        y1=jnp.asarray(y1), x1=jnp.asarray(x1),
        y2=jnp.asarray(y2), x2=jnp.asarray(x2),
        room_type=jnp.asarray(room_type), is_lit=jnp.asarray(is_lit),
    )
    return rooms, jnp.asarray(active)


def test_corridor_orthogonal_first():
    """Vendor `sp_lev.c::dig_corridor` (lines 2622-2660) prefers a straight
    cardinal run when the destination is already aligned on one axis;
    L-bends only appear when both row and column differ.

    Construct two rooms whose centres share a row, carve corridors, then
    assert the resulting corridor is a single horizontal segment (no
    extra vertical cells beyond the shared row).
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.rooms import (
        carve_rooms_into_terrain, connect_rooms,
    )

    # Two rooms centred on the same row.  Centre of [y1=5..y2=7] = 6.
    rooms, active = _make_room_pair(5, 5, 7, 8,    # centre row 6, col ~6
                                    5, 30, 7, 33)  # centre row 6, col ~31

    h, w = 21, 80
    terrain = jnp.zeros((h, w), dtype=jnp.int8)
    terrain = carve_rooms_into_terrain(terrain, rooms, active)
    terrain = connect_rooms(jax.random.PRNGKey(0), rooms, active, terrain)

    terrain_np = _to_np(terrain)
    CORRIDOR = 2
    corr_cells = np.argwhere(terrain_np == CORRIDOR)
    rows_used = set(int(r) for r, _ in corr_cells)
    # Straight corridor at row 6 -> only that row should have corridor cells.
    assert rows_used.issubset({6}), (
        f"orthogonal-first failed: corridor used rows {rows_used}, "
        f"expected only row 6"
    )
    assert len(corr_cells) > 0, "no corridor cells carved at all"


def test_corridor_width_one_tile():
    """Vendor `sp_lev.c::dig_corridor` writes one cell per step
    (line 2603: `crm->typ = ftyp`).  Corridor segments are 1 tile wide.

    Carve a corridor between two non-aligned rooms; verify no 2×2 block
    of CORRIDOR tiles exists.
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.rooms import (
        carve_rooms_into_terrain, connect_rooms,
    )

    rooms, active = _make_room_pair(2, 2, 4, 5,
                                    12, 40, 14, 45)
    h, w = 21, 80
    terrain = jnp.zeros((h, w), dtype=jnp.int8)
    terrain = carve_rooms_into_terrain(terrain, rooms, active)
    terrain = connect_rooms(jax.random.PRNGKey(1), rooms, active, terrain)

    terrain_np = _to_np(terrain)
    CORRIDOR = 2
    # Search for any 2x2 block all-corridor (would indicate width > 1).
    is_c = (terrain_np == CORRIDOR)
    for r in range(h - 1):
        for c in range(w - 1):
            block = is_c[r:r + 2, c:c + 2]
            assert not block.all(), (
                f"2x2 corridor block at ({r},{c}) — corridor wider than 1"
            )


# ---------------------------------------------------------------------------
# Part C — Mazes
# ---------------------------------------------------------------------------


def test_maze_uses_dfs_walker():
    """Vendor `mkmaze.c::walkfrom` (lines 1278-1310) is the DFS walker.

    A DFS-carved perfect maze on an odd×odd grid must:
      - have all floor cells connected (single component),
      - have NO 2×2 floor blocks (DFS only carves single-wide passages).
    """
    import jax
    from collections import deque
    from Nethax.nethax.dungeon.mazes import generate_maze_dfs, TILE_FLOOR

    rng = jax.random.PRNGKey(0)
    maze, h, w = generate_maze_dfs(rng, 11, 21)
    maze_np = _to_np(maze)

    floor_cells = [
        (r, c) for r in range(h) for c in range(w)
        if int(maze_np[r, c]) == TILE_FLOOR
    ]
    assert floor_cells, "DFS maze has no floor cells"

    # Connectivity
    visited = {floor_cells[0]}
    queue = deque([floor_cells[0]])
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if int(maze_np[nr, nc]) == TILE_FLOOR:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    assert set(floor_cells) == visited, "DFS maze is not fully connected"

    # No 2x2 floor block — vendor walkfrom only carves single-wide passages.
    is_f = (maze_np == TILE_FLOOR)
    for r in range(h - 1):
        for c in range(w - 1):
            block = is_f[r:r + 2, c:c + 2]
            assert not block.all(), (
                f"2x2 floor block at ({r},{c}) — not a single-pass DFS maze"
            )


def test_maze_no_doors_inside():
    """Vendor `mkmaze.c::makemaz` (lines 1197-1223) places stairs and monsters
    but never calls `add_door`.  A pure maze level has zero CLOSED_DOOR /
    OPEN_DOOR tiles.
    """
    import jax
    from Nethax.nethax.dungeon.mazes import generate_maze_dfs

    rng = jax.random.PRNGKey(3)
    maze, h, w = generate_maze_dfs(rng, 11, 21)
    maze_np = _to_np(maze)

    CLOSED_DOOR = 4
    OPEN_DOOR = 5
    assert not np.any(maze_np == CLOSED_DOOR), "DFS maze contains CLOSED_DOOR tile"
    assert not np.any(maze_np == OPEN_DOOR), "DFS maze contains OPEN_DOOR tile"


def test_maze_dead_ends_present():
    """Vendor `mkmaze.c::walkfrom` produces a perfect maze, which by graph
    theory contains at least one dead end (degree-1 vertex) on any non-trivial
    grid.  Verify ≥ 1 floor cell has exactly 1 floor neighbour.
    """
    import jax
    from Nethax.nethax.dungeon.mazes import generate_maze_dfs, TILE_FLOOR

    rng = jax.random.PRNGKey(0)
    maze, h, w = generate_maze_dfs(rng, 11, 21)
    maze_np = _to_np(maze)

    dead_ends = 0
    for r in range(h):
        for c in range(w):
            if int(maze_np[r, c]) != TILE_FLOOR:
                continue
            n_floor = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if int(maze_np[nr, nc]) == TILE_FLOOR:
                        n_floor += 1
            if n_floor == 1:
                dead_ends += 1

    assert dead_ends >= 1, f"perfect maze should have ≥ 1 dead end, got {dead_ends}"
