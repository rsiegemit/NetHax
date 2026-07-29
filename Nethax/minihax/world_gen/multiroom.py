"""Byte-exact MiniHack-MultiRoom level generation.

Ports gym_minigrid ``MultiRoomEnv._gen_grid``
(``.venv/.../gym_minigrid/envs/multiroom.py``) plus the base
``MiniGridEnv`` helpers (``_rand_int`` / ``_rand_elem`` / ``place_obj`` /
``place_agent``) using the bit-exact numpy ``RandomState`` port
(:mod:`Nethax.minihax.rng.numpy_randomstate`).  The vendor MiniGrid layout
is generated in the EXACT draw order, translated to a MiniHack des ``MAP``
block via ``MiniGridHack.get_env_map``
(``vendor/minihack/minihack/envs/minigrid.py``) semantics, then stamped onto
the 80x21 NetHack level at the ``GEOMETRY:center,center`` origin — reproducing
the vendor obs cell-for-cell.

The MiniGrid env is seeded with the NLE ``core`` seed via
``MiniGridHack.seed(core) -> minigrid_env.seed(core)`` -> gym-0.21
``np_random(core)``; :meth:`RandomState.seed_gym` reproduces that chain.  The
integer seed is recovered from the per-episode JAX PRNG key (``key(seed)`` has
``key_data == [0, seed]``).

The recursive room placement, entry-door colour draw, goal rejection sampling
and agent placement all consume ``RandomState`` in vendor order, so the layout
generalises to N4/N6/N10 and the Lava/Locked/Monster/OpenDoor variants without
any hardcoded coordinates.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import jax

from Nethax.minihax.level_generator import LevelGenerator
from Nethax.minihax.rng.numpy_randomstate import RandomState
from Nethax.nethax.state import EnvState

# MiniGrid MultiRoomEnv uses a fixed 25x25 grid (grid_size=25).
_GRID = 25

# COLOR_NAMES = sorted(COLORS.keys()) in gym_minigrid.minigrid.
_COLOR_NAMES = ["blue", "green", "grey", "purple", "red", "yellow"]

# Interior/child room size lower bound (vendor minSz=4).
_MIN_SIZE = 4


def _max_room_size(n_rooms: int) -> int:
    """Vendor ``maxRoomSize`` per registered MultiRoom size class.

    N2 -> MultiRoomEnvN2S4 (maxRoomSize=4); N4 -> MultiRoomEnvN4S5 (5);
    N6/N10 -> default maxRoomSize=10.
    """
    if n_rooms == 2:
        return 4
    if n_rooms == 4:
        return 5
    return 10


# --- gym_minigrid MiniGridEnv random helpers (exact) -----------------------

def _rand_int(rs: RandomState, low: int, high: int) -> int:
    """``self.np_random.randint(low, high)`` — half-open [low, high)."""
    return rs.randint(low, high)


def _rand_elem(rs: RandomState, iterable) -> object:
    """``lst[_rand_int(0, len(lst))]``."""
    lst = list(iterable)
    return lst[rs.randint(0, len(lst))]


class _Room:
    __slots__ = ("top", "size", "entryDoorPos", "exitDoorPos")

    def __init__(self, top, size, entryDoorPos, exitDoorPos):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos


class _Grid:
    """Minimal stand-in for gym_minigrid ``Grid`` — cells hold ``None`` or a
    marker string (``"wall"`` / ``"door"`` / ``"goal"``)."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = [[None] * width for _ in range(height)]

    def set(self, i, j, v):
        self.cells[j][i] = v

    def get(self, i, j):
        return self.cells[j][i]


def _place_room(rs, numLeft, roomList, minSz, maxSz, entryDoorWall,
                entryDoorPos, width, height):
    """Exact port of ``MultiRoomEnv._placeRoom``."""
    sizeX = _rand_int(rs, minSz, maxSz + 1)
    sizeY = _rand_int(rs, minSz, maxSz + 1)

    if len(roomList) == 0:
        topX, topY = entryDoorPos
    elif entryDoorWall == 0:
        topX = entryDoorPos[0] - sizeX + 1
        y = entryDoorPos[1]
        topY = _rand_int(rs, y - sizeY + 2, y)
    elif entryDoorWall == 1:
        x = entryDoorPos[0]
        topX = _rand_int(rs, x - sizeX + 2, x)
        topY = entryDoorPos[1] - sizeY + 1
    elif entryDoorWall == 2:
        topX = entryDoorPos[0]
        y = entryDoorPos[1]
        topY = _rand_int(rs, y - sizeY + 2, y)
    elif entryDoorWall == 3:
        x = entryDoorPos[0]
        topX = _rand_int(rs, x - sizeX + 2, x)
        topY = entryDoorPos[1]
    else:
        assert False, entryDoorWall

    if topX < 0 or topY < 0:
        return False
    if topX + sizeX > width or topY + sizeY >= height:
        return False

    for room in roomList[:-1]:
        nonOverlap = (
            topX + sizeX < room.top[0]
            or room.top[0] + room.size[0] <= topX
            or topY + sizeY < room.top[1]
            or room.top[1] + room.size[1] <= topY
        )
        if not nonOverlap:
            return False

    roomList.append(_Room((topX, topY), (sizeX, sizeY), entryDoorPos, None))

    if numLeft == 1:
        return True

    for _i in range(0, 8):
        wallSet = set((0, 1, 2, 3))
        wallSet.remove(entryDoorWall)
        exitDoorWall = _rand_elem(rs, sorted(wallSet))
        nextEntryWall = (exitDoorWall + 2) % 4

        if exitDoorWall == 0:
            exitDoorPos = (topX + sizeX - 1,
                           topY + _rand_int(rs, 1, sizeY - 1))
        elif exitDoorWall == 1:
            exitDoorPos = (topX + _rand_int(rs, 1, sizeX - 1),
                           topY + sizeY - 1)
        elif exitDoorWall == 2:
            exitDoorPos = (topX, topY + _rand_int(rs, 1, sizeY - 1))
        elif exitDoorWall == 3:
            exitDoorPos = (topX + _rand_int(rs, 1, sizeX - 1), topY)
        else:
            assert False

        success = _place_room(rs, numLeft - 1, roomList, minSz, maxSz,
                              nextEntryWall, exitDoorPos, width, height)
        if success:
            break

    return True


def _place_obj(rs, grid, agent_pos, top, size):
    """Port of ``MiniGridEnv.place_obj`` (rejection sampling); returns pos."""
    top = (max(top[0], 0), max(top[1], 0))
    while True:
        pos = (
            _rand_int(rs, top[0], min(top[0] + size[0], grid.width)),
            _rand_int(rs, top[1], min(top[1] + size[1], grid.height)),
        )
        if grid.get(*pos) is not None:
            continue
        if agent_pos is not None and pos == agent_pos:
            continue
        break
    return pos


def _gen_grid(seed: int, min_rooms: int, max_rooms: int, max_room_size: int):
    """Exact port of ``MultiRoomEnv._gen_grid``.

    Returns ``(grid, agent_pos, agent_dir, goal_pos)`` where ``grid`` marks
    wall/door/goal cells (agent tracked separately, matching vendor which
    places ``None`` under the agent).
    """
    rs = RandomState()
    rs.seed_gym(seed)
    width = height = _GRID

    roomList: List[_Room] = []
    numRooms = _rand_int(rs, min_rooms, max_rooms + 1)

    while len(roomList) < numRooms:
        curRoomList: List[_Room] = []
        entryDoorPos = (_rand_int(rs, 0, width - 2),
                        _rand_int(rs, 0, width - 2))
        _place_room(rs, numRooms, curRoomList, _MIN_SIZE, max_room_size,
                    2, entryDoorPos, width, height)
        if len(curRoomList) > len(roomList):
            roomList = curRoomList

    assert len(roomList) > 0

    grid = _Grid(width, height)
    prevDoorColor = None

    for idx, room in enumerate(roomList):
        topX, topY = room.top
        sizeX, sizeY = room.size
        for i in range(0, sizeX):
            grid.set(topX + i, topY, "wall")
            grid.set(topX + i, topY + sizeY - 1, "wall")
        for j in range(0, sizeY):
            grid.set(topX, topY + j, "wall")
            grid.set(topX + sizeX - 1, topY + j, "wall")

        if idx > 0:
            doorColors = set(_COLOR_NAMES)
            if prevDoorColor:
                doorColors.remove(prevDoorColor)
            doorColor = _rand_elem(rs, sorted(doorColors))
            grid.set(*room.entryDoorPos, "door")
            prevDoorColor = doorColor
            roomList[idx - 1].exitDoorPos = room.entryDoorPos

    # place_agent(roomList[0].top, roomList[0].size)
    agent_pos = _place_obj(rs, grid, None,
                           roomList[0].top, roomList[0].size)
    agent_dir = _rand_int(rs, 0, 4)

    # place_obj(Goal(), roomList[-1].top, roomList[-1].size)
    goal_pos = _place_obj(rs, grid, agent_pos,
                          roomList[-1].top, roomList[-1].size)
    grid.set(*goal_pos, "goal")

    return grid, agent_pos, agent_dir, goal_pos


def _get_env_map(grid, agent_pos, wall_char: str):
    """Port of ``MiniGridHack.get_env_map`` (crop + floor trim).

    Returns ``(env_map_lines, start_pos, goal_pos, door_pos)`` in cropped-map
    (col, row) coordinates.
    """
    door_pos: List[Tuple[int, int]] = []
    goal_pos = None
    empty_strs = 0
    empty_str = True
    env_map: List[str] = []

    for j in range(grid.height):
        s = ""
        for i in range(grid.width):
            c = grid.get(i, j)
            if c is None:
                s += "."
                continue
            empty_str = False
            if c == "wall":
                s += wall_char
            elif c == "door":
                s += "+"
                door_pos.append((i, j - empty_strs))
            elif c == "goal":
                goal_pos = (i, j - empty_strs)
                s += "."
        if not empty_str and j < grid.height - 1:
            if set(s) != {"."}:
                s = s.replace(".", " ", s.index(wall_char))
                inv = s[::-1]
                s = inv.replace(".", " ", inv.index(wall_char))[::-1]
                env_map.append(s)
        elif empty_str:
            empty_strs += 1

    start_pos = (int(agent_pos[0]), int(agent_pos[1]) - empty_strs)
    return env_map, start_pos, goal_pos, door_pos


# sp_lev.c CENTER formula (GEOMETRY:center,center) — internal terrain origin.
def _geometry_center(w: int, h: int) -> Tuple[int, int]:
    x_maze_max = 78  # COLNO - 1
    y_maze_max = 20  # ROWNO - 1
    xstart = 2 + ((x_maze_max - 2 - w) // 2)
    ystart = 2 + ((y_maze_max - 2 - h) // 2)
    if (xstart % 2) == 0:
        xstart += 1
    if (ystart % 2) == 0:
        ystart += 1
    return xstart, ystart


def _recover_seed(rng: jax.Array) -> int:
    """Recover the integer NLE seed from a ``jax.random.key(seed)`` key."""
    try:
        kd = jax.random.key_data(rng)
    except (TypeError, ValueError):
        kd = rng
    kd = [int(x) for x in kd.reshape(-1)]
    if len(kd) >= 2:
        return (kd[0] << 32) | kd[1]
    return kd[0]


def multiroom_factory(n_rooms: int, *, lava_walls: bool,
                      locked: bool, monster: bool,
                      open_door: bool, extreme: bool
                      ) -> Callable[[jax.Array], EnvState]:
    """Build a per-reset ``(rng) -> EnvState`` factory for MultiRoom envs."""
    door_state = "locked" if (locked or extreme) else (
        "open" if open_door else "closed"
    )
    wall_char = "L" if (lava_walls or extreme) else "|"
    max_room_size = _max_room_size(n_rooms)
    # num_mon per vendor (minigrid.py): Monster N2/N4/N6=3/6/9; Extreme same.
    num_mon = 0
    if monster or extreme:
        num_mon = {2: 3, 4: 6, 6: 9, 10: 9}.get(n_rooms, 3)

    def factory(rng: jax.Array) -> EnvState:
        seed = _recover_seed(rng)
        grid, agent_pos, _agent_dir, _goal = _gen_grid(
            seed, n_rooms, n_rooms, max_room_size,
        )
        env_map, start_pos, goal_pos, door_pos = _get_env_map(
            grid, agent_pos, wall_char,
        )

        # Vendor LevelGenerator(map=env_map): W = max line length (each line is
        # grid.width wide including trailing VOID), H = number of lines.
        map_w = max(len(line) for line in env_map)
        map_h = len(env_map)
        dx, dy = _geometry_center(map_w, map_h)

        # Stamp the cropped MAP onto a full 80x21 VOID grid at the centered
        # internal origin, then hand to set_map (mirrors CorridorBattle path).
        grid_rows: List[str] = []
        for gy in range(21):
            row = [" "] * 80
            my = gy - dy
            if 0 <= my < map_h:
                line = env_map[my]
                for cx, ch in enumerate(line):
                    ax = cx + dx
                    if 0 <= ax < 80:
                        row[ax] = ch
            grid_rows.append("".join(row))

        lg = LevelGenerator(w=80, h=21, fill=" ", lit=True)
        lg.set_map(grid_rows)
        lg.set_start_pos(start_pos[0] + dx, start_pos[1] + dy)
        if goal_pos is not None:
            lg.add_stair_down(x=goal_pos[0] + dx, y=goal_pos[1] + dy)
        for (dcx, dcy) in door_pos:
            lg.add_door(door_state, place=(dcx + dx, dcy + dy))
        for _ in range(num_mon):
            lg.add_monster()

        return lg.get_factory()(rng)

    return factory
