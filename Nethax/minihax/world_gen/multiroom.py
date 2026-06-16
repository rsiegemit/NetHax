"""Procedural recursive room+door placement for MiniHack-MultiRoom envs.

Source: vendor/minihack/minihack/envs/minigrid.py wraps gym_minigrid's
``MultiRoomEnv`` (gym_minigrid/envs/multiroom.py). The reference algorithm
places the first room at a random position, then for each subsequent room
picks an existing wall and grows a fresh adjacent room with random size,
carving a door at the shared wall.  Topology is re-randomised every reset.

JIT note: the returned factory runs at reset time on the host (the LG
factory itself runs Python-side per ``LevelGenerator._apply_directives``).
We use a bounded Python loop over N rooms with JAX PRNG randomness for
each random choice, so this code never lands inside a JIT trace.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import jax

from Nethax.minihax.level_generator import LevelGenerator
from Nethax.nethax.state import EnvState


# Map dims match canonical.py vendor MultiRoom (was: w=76, h=21).
_MAP_W = 76
_MAP_H = 21

# Room interior size bounds.  Mirrors gym_minigrid MultiRoomEnv minRoomSize=4.
# Smaller upper bound for larger N so all rooms fit without overlap.
_MIN_SIZE = 4
_MAX_SIZE_BY_N = {2: 7, 4: 6, 6: 5, 10: 4}

# Per-room placement attempts before giving up on the child room.
_PLACE_ATTEMPTS = 24


def _max_size(n_rooms: int) -> int:
    """Pick the upper bound on a room's interior side length for ``n_rooms``."""
    for n_threshold in sorted(_MAX_SIZE_BY_N):
        if n_rooms <= n_threshold:
            return _MAX_SIZE_BY_N[n_threshold]
    return _MIN_SIZE


def _jrandint(rng: jax.Array, lo: int, hi: int) -> int:
    """Inclusive ``[lo, hi]`` sample from a JAX subkey."""
    if hi <= lo:
        return lo
    return int(jax.random.randint(rng, (), lo, hi + 1))


Bbox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) interior, inclusive
Door = Tuple[int, int]             # (x, y)


def _rects_overlap(a: Bbox, b: Bbox) -> bool:
    """True if interior bboxes ``a`` / ``b`` overlap or touch.

    Rooms sharing only a wall (gap == 1) don't count as overlapping; they're
    valid neighbours that can be joined via a door.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # Allow a 1-cell separation (the shared wall).
    return not (ax2 + 1 < bx1 or bx2 + 1 < ax1
                or ay2 + 1 < by1 or by2 + 1 < ay1)


def _place_first_room(rng_pool: jax.Array,
                      n_rooms: int) -> Tuple[jax.Array, Bbox]:
    """Pick a random bbox for room 0 anywhere on the map."""
    max_sz = _max_size(n_rooms)
    rng_pool, k_w, k_h, k_x, k_y = jax.random.split(rng_pool, 5)
    rw = _jrandint(k_w, _MIN_SIZE, max_sz)
    rh = _jrandint(k_h, _MIN_SIZE, max_sz)
    x1 = _jrandint(k_x, 1, _MAP_W - rw - 2)
    y1 = _jrandint(k_y, 1, _MAP_H - rh - 2)
    return rng_pool, (x1, y1, x1 + rw - 1, y1 + rh - 1)


def _try_child(rng_pool: jax.Array, parent: Bbox, side: int,
               n_rooms: int) -> Tuple[jax.Array, Optional[Bbox], Optional[Door]]:
    """Sample a candidate child rect attached to ``parent`` on ``side``.

    ``side``: 0=N, 1=S, 2=W, 3=E (relative to parent).  Returns ``(rng,
    child_bbox, door_xy)`` or ``(rng, None, None)`` when the child cannot
    fit the map bounds.  Overlap with other rooms is checked by the caller.
    """
    px1, py1, px2, py2 = parent
    max_sz = _max_size(n_rooms)
    rng_pool, k_w, k_h, k_off, k_door = jax.random.split(rng_pool, 5)
    cw = _jrandint(k_w, _MIN_SIZE, max_sz)
    ch = _jrandint(k_h, _MIN_SIZE, max_sz)

    if side in (0, 1):
        # North/South: shared wall is horizontal.  Pick a column offset so
        # the child interior overlaps the parent interior by ≥ 1 cell on
        # the x-axis (required for a reachable door).
        ov_lo = max(px1 - cw + 1, 1)
        ov_hi = min(px2, _MAP_W - cw - 2)
        if ov_hi < ov_lo:
            return rng_pool, None, None
        cx1 = _jrandint(k_off, ov_lo, ov_hi)
        cx2 = cx1 + cw - 1
        if side == 0:
            cy2 = py1 - 2
            cy1 = cy2 - ch + 1
        else:
            cy1 = py2 + 2
            cy2 = cy1 + ch - 1
        if cy1 < 1 or cy2 > _MAP_H - 2:
            return rng_pool, None, None
        door_y = py1 - 1 if side == 0 else py2 + 1
        dx_lo = max(px1, cx1)
        dx_hi = min(px2, cx2)
        door_x = _jrandint(k_door, dx_lo, dx_hi)
        return rng_pool, (cx1, cy1, cx2, cy2), (door_x, door_y)

    # East/West: shared wall is vertical.
    ov_lo = max(py1 - ch + 1, 1)
    ov_hi = min(py2, _MAP_H - ch - 2)
    if ov_hi < ov_lo:
        return rng_pool, None, None
    cy1 = _jrandint(k_off, ov_lo, ov_hi)
    cy2 = cy1 + ch - 1
    if side == 2:
        cx2 = px1 - 2
        cx1 = cx2 - cw + 1
    else:
        cx1 = px2 + 2
        cx2 = cx1 + cw - 1
    if cx1 < 1 or cx2 > _MAP_W - 2:
        return rng_pool, None, None
    door_x = px1 - 1 if side == 2 else px2 + 1
    dy_lo = max(py1, cy1)
    dy_hi = min(py2, cy2)
    door_y = _jrandint(k_door, dy_lo, dy_hi)
    return rng_pool, (cx1, cy1, cx2, cy2), (door_x, door_y)


def _layout(rng: jax.Array, n_rooms: int) -> Tuple[List[Bbox], List[Door]]:
    """Build randomised ``(rooms, doors)`` for one reset.

    Each room is ``(x1, y1, x2, y2)`` interior bbox; each door is ``(x, y)``.
    """
    rng_pool, room0 = _place_first_room(rng, n_rooms)
    rooms: List[Bbox] = [room0]
    doors: List[Door] = []
    for _ in range(n_rooms - 1):
        placed = False
        for _attempt in range(_PLACE_ATTEMPTS):
            rng_pool, k_parent, k_side = jax.random.split(rng_pool, 3)
            parent_idx = int(jax.random.randint(
                k_parent, (), 0, len(rooms),
            ))
            side = int(jax.random.randint(k_side, (), 0, 4))
            rng_pool, child, door = _try_child(
                rng_pool, rooms[parent_idx], side, n_rooms,
            )
            if child is None:
                continue
            if any(_rects_overlap(child, r) for r in rooms):
                continue
            rooms.append(child)
            doors.append(door)
            placed = True
            break
        if not placed:
            # No room left for further children — return what we have; the
            # env is still playable, just smaller than requested.
            break
    return rooms, doors


def multiroom_factory(n_rooms: int, *, lava_walls: bool,
                      locked: bool, monster: bool,
                      open_door: bool, extreme: bool
                      ) -> Callable[[jax.Array], EnvState]:
    """Build a per-reset ``(rng) -> EnvState`` factory for MultiRoom envs.

    ``door_state`` resolution:
      * ``locked``/``extreme`` → "locked"
      * ``open_door``         → "open"
      * else                   → "closed"
    """
    door_state = "locked" if (locked or extreme) else (
        "open" if open_door else "closed"
    )

    def factory(rng: jax.Array) -> EnvState:
        k_layout, k_build = jax.random.split(rng)
        rooms, doors = _layout(k_layout, n_rooms)

        lg = LevelGenerator(w=_MAP_W, h=_MAP_H)
        for (x1, y1, x2, y2) in rooms:
            lg.add_room(x=x1, y=y1, w=x2 - x1 + 1, h=y2 - y1 + 1)
        for (dx, dy) in doors:
            lg.add_door(dx, dy, state=door_state)

        # Lava strip on the floor of the first room (mirrors vendor
        # MiniHack-MultiRoom-Lava layout: a horizontal lava bar near the
        # room centre that the agent must navigate around).
        if lava_walls or extreme:
            x1, y1, x2, y2 = rooms[0]
            if x2 - x1 >= 2:
                ly = (y1 + y2) // 2
                lg.fill_terrain(
                    "L", x1 + 1, ly, min(x1 + 3, x2 - 1), ly,
                )

        # Monsters: spawn near the goal so they oppose the final approach.
        if monster or extreme:
            gx1, gy1, _gx2, _gy2 = rooms[-1]
            for _ in range(min(3, n_rooms)):
                lg.add_monster(place=(gx1, gy1))

        # Start in centre of first room, goal in centre of last room.
        sx1, sy1, sx2, sy2 = rooms[0]
        gx1, gy1, gx2, gy2 = rooms[-1]
        lg.set_start_pos((sx1 + sx2) // 2, (sy1 + sy2) // 2)
        lg.add_stair_down(x=(gx1 + gx2) // 2, y=(gy1 + gy2) // 2)

        return lg.get_factory()(k_build)

    return factory
