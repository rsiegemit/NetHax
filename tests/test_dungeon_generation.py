"""Wave 2 tests for dungeon generation.

Tests room placement, corridor connectivity, JIT compilation, and Kruskal
maze connectivity.  All imports are lazy (inside test functions) so test
collection succeeds even if a module is still being implemented.
"""

import pytest


def test_rooms_are_non_overlapping():
    """No two active rooms should have overlapping bounding boxes."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.rooms import generate_rooms

    rng = jax.random.PRNGKey(0)
    rooms, active = generate_rooms(rng, 21, 80, 8)

    # active is bool[MAX_ROOMS_PER_LEVEL]; rooms.y1/x1/y2/x2 are int16[MAX_ROOMS_PER_LEVEL]
    active_np = active.__array__() if hasattr(active, "__array__") else active
    y1 = rooms.y1.__array__() if hasattr(rooms.y1, "__array__") else rooms.y1
    x1 = rooms.x1.__array__() if hasattr(rooms.x1, "__array__") else rooms.x1
    y2 = rooms.y2.__array__() if hasattr(rooms.y2, "__array__") else rooms.y2
    x2 = rooms.x2.__array__() if hasattr(rooms.x2, "__array__") else rooms.x2

    active_indices = [i for i in range(len(active_np)) if active_np[i]]

    margin = 1
    for i_pos, i in enumerate(active_indices):
        for j in active_indices[i_pos + 1:]:
            # Two rooms overlap if their bounding boxes (plus margin) intersect
            no_overlap = (
                int(x2[i]) + margin < int(x1[j]) or
                int(x2[j]) + margin < int(x1[i]) or
                int(y2[i]) + margin < int(y1[j]) or
                int(y2[j]) + margin < int(y1[i])
            )
            assert no_overlap, (
                f"Rooms {i} and {j} overlap: "
                f"room {i} = (y1={y1[i]}, x1={x1[i]}, y2={y2[i]}, x2={x2[i]}), "
                f"room {j} = (y1={y1[j]}, x1={x1[j]}, y2={y2[j]}, x2={x2[j]})"
            )


def test_rooms_within_bounds():
    """All active rooms must fit within the (21, 80) map."""
    import jax
    from Nethax.nethax.dungeon.rooms import generate_rooms

    rng = jax.random.PRNGKey(0)
    rooms, active = generate_rooms(rng, 21, 80, 8)

    active_np = active.__array__() if hasattr(active, "__array__") else active
    y1 = rooms.y1.__array__() if hasattr(rooms.y1, "__array__") else rooms.y1
    x1 = rooms.x1.__array__() if hasattr(rooms.x1, "__array__") else rooms.x1
    y2 = rooms.y2.__array__() if hasattr(rooms.y2, "__array__") else rooms.y2
    x2 = rooms.x2.__array__() if hasattr(rooms.x2, "__array__") else rooms.x2

    h, w = 21, 80
    for i in range(len(active_np)):
        if not active_np[i]:
            continue
        assert int(y1[i]) >= 1,      f"Room {i} top edge y1={y1[i]} < 1"
        assert int(x1[i]) >= 1,      f"Room {i} left edge x1={x1[i]} < 1"
        assert int(y2[i]) <= h - 2,  f"Room {i} bottom edge y2={y2[i]} > {h-2}"
        assert int(x2[i]) <= w - 2,  f"Room {i} right edge x2={x2[i]} > {w-2}"


def test_corridor_connectivity():
    """BFS from the up-stair tile should reach the down-stair tile."""
    import jax
    import jax.numpy as jnp
    from collections import deque
    from Nethax.nethax.dungeon.branches import generate_main_branch_l1
    from Nethax.nethax.state import StaticParams

    rng = jax.random.PRNGKey(0)
    static = StaticParams()
    terrain, _rooms, _active, up_pos, dn_pos, *_rest = generate_main_branch_l1(rng, static)

    terrain_np = terrain.__array__() if hasattr(terrain, "__array__") else terrain
    up_r, up_c = int(up_pos[0]), int(up_pos[1])
    dn_r, dn_c = int(dn_pos[0]), int(dn_pos[1])

    # Tiles that are walkable: FLOOR=1, CORRIDOR=2, OPEN_DOOR=5,
    # STAIRCASE_UP=6, STAIRCASE_DOWN=7, SHOP_FLOOR=16
    WALKABLE = {1, 2, 5, 6, 7, 16}

    h, w = terrain_np.shape
    visited = set()
    queue = deque([(up_r, up_c)])
    visited.add((up_r, up_c))

    while queue:
        r, c = queue.popleft()
        if (r, c) == (dn_r, dn_c):
            break
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if int(terrain_np[nr, nc]) in WALKABLE:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    assert (dn_r, dn_c) in visited, (
        f"Down-stair at ({dn_r},{dn_c}) not reachable from up-stair at ({up_r},{up_c}). "
        f"Up-stair tile={int(terrain_np[up_r, up_c])}, "
        f"Down-stair tile={int(terrain_np[dn_r, dn_c])}"
    )


def test_dungeon_jits():
    """generate_main_branch_l1 should compile under jax.jit."""
    import jax
    from Nethax.nethax.dungeon.branches import generate_main_branch_l1
    from Nethax.nethax.state import StaticParams

    rng = jax.random.PRNGKey(0)
    static = StaticParams()

    jitted = jax.jit(generate_main_branch_l1, static_argnums=(1,))
    result = jitted(rng, static)
    # Unpack to force materialisation
    terrain, rooms, active, up_pos, dn_pos = result
    assert terrain.shape == (static.map_h, static.map_w)


def test_maze_kruskal_connected():
    """Every floor cell in the Kruskal maze should be reachable from every other floor cell."""
    import jax
    from collections import deque
    from Nethax.nethax.dungeon.mazes import generate_maze_kruskal, TILE_FLOOR

    rng = jax.random.PRNGKey(0)
    maze, h, w = generate_maze_kruskal(rng, 11, 11)

    maze_np = maze.__array__() if hasattr(maze, "__array__") else maze

    # Collect all floor cells
    floor_cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(maze_np[r, c]) == TILE_FLOOR
    ]

    assert len(floor_cells) > 0, "Maze has no floor cells"

    # BFS from the first floor cell
    start = floor_cells[0]
    visited = {start}
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if int(maze_np[nr, nc]) == TILE_FLOOR:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    unreachable = [cell for cell in floor_cells if cell not in visited]
    assert len(unreachable) == 0, (
        f"{len(unreachable)} floor cell(s) not reachable from {start}: "
        f"first few = {unreachable[:5]}"
    )
