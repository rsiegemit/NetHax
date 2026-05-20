"""MiniHack-compatible ``LevelGenerator`` API ported to nethax.

This module provides the Python builder API used by MiniHack at environment
construction time to author levels.  It mirrors the public surface of
``vendor/minihack/minihack/level_generator.py`` so existing MiniHack level
scripts can be ported with minimal edits, while emitting a JAX ``EnvState``
suitable for the nethax engine.

Coordinate conventions
----------------------
MiniHack uses ``(x, y)`` = (column, row); nethax uses ``(row, col)``.  The
public API of this module accepts MiniHack ``(x, y)`` arguments to match the
vendor API; the factory converts to nethax row/col when writing into JAX
arrays.

Status
------
Wave 4 Phase 1, agent A1 deliverable.  Implements the builder + factory
without modifying ``EnvState`` schema.  Goal positions are recorded as
``STAIRCASE_DOWN`` tiles (consistent with MiniHack's
``add_stair_down``/``add_goal_pos`` aliasing).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.constants.objects import OBJECTS, ObjectClass, OBJECT_NAME_ALIASES
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.dungeon.spawning import (
    _ATK_DICE_N,
    _ATK_DICE_S,
    _BASE_AC,
    _IS_LARGE,
)
from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.subsystems.inventory import MAX_GROUND_STACK
from Nethax.nethax.subsystems.traps import TrapType


# ---------------------------------------------------------------------------
# Public maps mirroring MiniHack's vendor API
# ---------------------------------------------------------------------------

#: MiniHack terrain-character to nethax ``TileType``.
#: Source: ``vendor/minihack/minihack/level_generator.py`` ``MAP_CHARS``.
TERRAIN_CHAR_TO_TILE: dict = {
    " ": TileType.VOID,
    "#": TileType.CORRIDOR,
    ".": TileType.FLOOR,
    "-": TileType.WALL,
    "|": TileType.WALL,
    "+": TileType.CLOSED_DOOR,
    "}": TileType.WATER,
    "P": TileType.WATER,
    "W": TileType.WATER,
    "L": TileType.LAVA,
    "{": TileType.FOUNTAIN,
    "\\": TileType.THRONE,
    "<": TileType.STAIRCASE_UP,
    ">": TileType.STAIRCASE_DOWN,
}

#: MiniHack trap-name to nethax ``TrapType``.
#: Source: ``vendor/minihack/minihack/level_generator.py`` ``TRAP_NAMES``.
TRAP_NAME_TO_TYPE: dict = {
    "anti magic":      TrapType.ANTI_MAGIC,
    "arrow":           TrapType.ARROW_TRAP,
    "bear":            TrapType.BEAR_TRAP,
    "board":           TrapType.SQKY_BOARD,
    "dart":            TrapType.DART_TRAP,
    "falling rock":    TrapType.ROCKTRAP,
    "fire":            TrapType.FIRE_TRAP,
    "hole":            TrapType.HOLE,
    "land mine":       TrapType.LANDMINE,
    "level teleport":  TrapType.LEVEL_TELEP,
    "magic portal":    TrapType.MAGIC_PORTAL,
    "magic":           TrapType.MAGIC_TRAP,
    "pit":             TrapType.PIT,
    "polymorph":       TrapType.POLY_TRAP,
    "rolling boulder": TrapType.ROLLING_BOULDER_TRAP,
    "rust":            TrapType.RUST_TRAP,
    "sleep gas":       TrapType.SLP_GAS_TRAP,
    "spiked pit":      TrapType.SPIKED_PIT,
    "statue":          TrapType.STATUE_TRAP,
    "teleport":        TrapType.TELEP_TRAP,
    "trap door":       TrapType.TRAPDOOR,
    "web":             TrapType.WEB,
}


# ---------------------------------------------------------------------------
# Name → table-index lookups (one-time at import)
# ---------------------------------------------------------------------------

def _build_monster_name_lookup() -> dict:
    table = {}
    for idx, entry in enumerate(MONSTERS):
        table.setdefault(entry.name, idx)
    return table


def _build_object_name_lookup() -> dict:
    """Map MiniHack-style object names to OBJECTS indices.

    Wave 6 parity-fix (CA #63): OBJECTS regenerated from vendor objects.c
    contains anonymous separator rows (``name is None``).  Skip them and
    merge ``OBJECT_NAME_ALIASES`` so MiniHack scripts can still ask for
    "potion of levitation" (now stored bare as "levitation" + alias).
    Cite: vendor/nethack/src/objects.c — bare canonical names per class.
    """
    table: dict = {}
    for idx, entry in enumerate(OBJECTS):
        if entry.name is None:
            continue
        table.setdefault(entry.name, idx)
    # Merge "<prefix> <name>" aliases (e.g. "potion of levitation" -> 248).
    for alias, idx in OBJECT_NAME_ALIASES.items():
        table.setdefault(alias, idx)
    return table


_MONSTER_NAME_TO_IDX: dict = _build_monster_name_lookup()
_OBJECT_NAME_TO_IDX: dict = _build_object_name_lookup()


# ---------------------------------------------------------------------------
# Directive dataclasses
# ---------------------------------------------------------------------------

# Place specification: either a (col, row) tuple, a string room_id, or None.
Place = Union[None, Tuple[int, int], str]


@dataclasses.dataclass
class _RoomDirective:
    room_id: str
    x: int           # left col; -1 = random
    y: int           # top row;  -1 = random
    w: int           # width;    -1 = random
    h: int           # height;   -1 = random
    lit: bool


@dataclasses.dataclass
class _CorridorDirective:
    src: Tuple[int, int]   # (col, row)
    dst: Tuple[int, int]


@dataclasses.dataclass
class _DoorDirective:
    x: int
    y: int
    state: str   # 'closed' | 'open' | 'locked' | 'nodoor' | 'random'


@dataclasses.dataclass
class _MonsterDirective:
    name: str
    symbol: Optional[str]
    place: Place
    args: tuple


@dataclasses.dataclass
class _TrapDirective:
    name: str
    place: Place


@dataclasses.dataclass
class _ObjectDirective:
    name: str
    symbol: Optional[str]
    place: Place
    cursestate: str   # 'random' | 'blessed' | 'uncursed' | 'cursed'


@dataclasses.dataclass
class _StairDirective:
    direction: str    # 'up' | 'down'
    x: int            # -1 = random / use place
    y: int
    place: Place


@dataclasses.dataclass
class _FillTerrainDirective:
    terrain: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclasses.dataclass
class _StartPosDirective:
    x: int
    y: int


@dataclasses.dataclass
class _GoalPosDirective:
    x: int
    y: int


# ---------------------------------------------------------------------------
# Wave17i additions: directive types for add_altar / add_sink / add_gold /
# add_mazewalk.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _AltarOverride:
    x: int = -1
    y: int = -1
    place: Place = None


@dataclasses.dataclass
class _SinkOverride:
    place: Place = None


@dataclasses.dataclass
class _GoldDirective:
    amount: int
    place: Place = None


@dataclasses.dataclass
class _MazeWalkDirective:
    x: int
    y: int
    direction: str


# ---------------------------------------------------------------------------
# LevelGenerator
# ---------------------------------------------------------------------------

class LevelGenerator:
    """Python-side builder for MiniHack-style levels.

    Calling ``add_*`` appends a directive to an internal list.  ``get_factory``
    returns a closure that walks the directives and produces a fully populated
    ``EnvState`` using the supplied PRNG key.

    The builder is *not* JIT-traceable; it runs once on the Python side at
    environment-reset time.  The resulting ``EnvState`` is a plain Flax pytree
    that downstream ``env.step`` invocations can JIT-compile against.
    """

    def __init__(
        self,
        w: int = 80,
        h: int = 21,
        fill: str = ".",
        lit: bool = True,
    ) -> None:
        if w <= 0 or h <= 0:
            raise ValueError(f"map dimensions must be positive, got w={w} h={h}")
        if fill not in TERRAIN_CHAR_TO_TILE:
            raise ValueError(f"fill char {fill!r} is not a known terrain symbol")
        # nethax terrain arrays are sized to StaticParams (default 80×21).
        # Generated levels can be smaller; we only write into the top-left
        # (w × h) sub-region and leave the rest as VOID.
        static = StaticParams()
        if w > static.map_w or h > static.map_h:
            raise ValueError(
                f"requested map {w}x{h} exceeds static bounds "
                f"{static.map_w}x{static.map_h}"
            )

        self.w = w
        self.h = h
        self.fill = fill
        self.default_lit = lit
        self._static = static

        self._directives: List[Any] = []
        self._room_directives: dict = {}   # room_id -> _RoomDirective
        self._room_counter = 0

        # Build-trace metadata captured each time the factory runs.
        # Tests inspect these to verify name→index resolution.
        self.last_monster_entry_ids: List[int] = []
        self.last_object_entry_ids: List[int] = []
        self.last_trap_types: List[int] = []
        self.last_player_pos: Optional[Tuple[int, int]] = None
        self.last_goal_pos: Optional[Tuple[int, int]] = None

    # ---- Builder API -----------------------------------------------------

    def add_room(
        self,
        x: int = -1,
        y: int = -1,
        w: int = -1,
        h: int = -1,
        *,
        lit: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> str:
        """Reserve a rectangular room region.

        Coordinates use MiniHack convention: ``x`` is column, ``y`` is row.
        ``-1`` requests random placement / size at factory time.
        Returns a stable ``room_id`` string that can be passed to
        ``place=`` arguments on other directives.
        """
        if name is None:
            name = f"room_{self._room_counter}"
        self._room_counter += 1
        eff_lit = self.default_lit if lit is None else bool(lit)
        directive = _RoomDirective(room_id=name, x=x, y=y, w=w, h=h, lit=eff_lit)
        self._directives.append(directive)
        self._room_directives[name] = directive
        return name

    def add_corridor(self, src: Tuple[int, int], dst: Tuple[int, int]) -> None:
        """Carve an L-shaped corridor between two ``(x, y)`` endpoints."""
        self._directives.append(_CorridorDirective(src=tuple(src), dst=tuple(dst)))

    def add_door(self, *args, state: str = "closed", place=None) -> None:
        """Vendor-parity add_door.

        Two signatures supported (Wave17i):
          * Vendor (level_generator.py): ``add_door(state, place=(x, y))``
            where ``state`` is a string and ``place`` is a ``(col, row)``
            coord tuple.
          * Legacy nethax: ``add_door(x, y, state="closed")``.
        """
        # Decode positional args.
        x: int = -1
        y: int = -1
        if len(args) == 1 and isinstance(args[0], str):
            # Vendor form: add_door("closed", place=(x, y))
            state = args[0]
        elif len(args) == 1 and isinstance(args[0], tuple):
            # add_door((x, y), state=...)
            x, y = int(args[0][0]), int(args[0][1])
        elif len(args) == 2:
            a0, a1 = args
            if isinstance(a0, int) and isinstance(a1, int):
                # Legacy: add_door(x, y, state=...)
                x, y = int(a0), int(a1)
            elif isinstance(a0, str):
                # add_door("closed", (x, y))
                state = a0
                if isinstance(a1, tuple) and len(a1) == 2:
                    x, y = int(a1[0]), int(a1[1])
        elif len(args) == 3:
            # Legacy: add_door(x, y, state)
            x, y, state = int(args[0]), int(args[1]), str(args[2])
        elif len(args) == 0:
            pass  # state/place as kwargs only
        else:
            raise TypeError(f"add_door: too many positional args ({len(args)})")

        if place is not None:
            if isinstance(place, tuple) and len(place) == 2:
                x, y = int(place[0]), int(place[1])

        s = str(state)
        if s not in ("closed", "open", "locked", "nodoor", "random"):
            raise ValueError(f"unknown door state: {s!r}")
        self._directives.append(_DoorDirective(x=x, y=y, state=s))

    def add_monster(
        self,
        name: str = "random",
        symbol: Optional[str] = None,
        place: Place = None,
        args: tuple = (),
    ) -> None:
        """Spawn a monster on the level."""
        self._directives.append(_MonsterDirective(
            name=name, symbol=symbol, place=place, args=tuple(args),
        ))

    def add_trap(self, name: str = "teleport", place: Place = None) -> None:
        """Place a trap of the named kind."""
        if name != "random" and name not in TRAP_NAME_TO_TYPE:
            raise ValueError(
                f"unknown trap name {name!r}; valid: {sorted(TRAP_NAME_TO_TYPE)}"
            )
        self._directives.append(_TrapDirective(name=name, place=place))

    def add_object(
        self,
        name: str = "random",
        symbol: Optional[str] = None,
        place: Place = None,
        cursestate: str = "random",
    ) -> None:
        """Place an object (item) on the ground."""
        if cursestate not in ("random", "blessed", "uncursed", "cursed"):
            raise ValueError(f"unknown cursestate: {cursestate!r}")
        self._directives.append(_ObjectDirective(
            name=name, symbol=symbol, place=place, cursestate=cursestate,
        ))

    def add_stair_up(
        self,
        x: int = -1,
        y: int = -1,
        *,
        place: Place = None,
    ) -> None:
        """Add an up-staircase tile."""
        self._directives.append(_StairDirective(
            direction="up", x=x, y=y, place=place,
        ))

    def add_stair_down(
        self,
        x=-1,
        y: int = -1,
        *,
        place: Place = None,
    ) -> None:
        """Add a down-staircase tile (also the canonical 'goal' tile).

        Vendor-parity (Wave17i): accepts either ``add_stair_down((x, y))`` or
        ``add_stair_down(x, y)`` to match vendor level_generator.py which
        passes a ``coord`` tuple.
        """
        if isinstance(x, tuple) and len(x) == 2:
            cx, cy = int(x[0]), int(x[1])
        else:
            cx, cy = int(x), int(y)
        self._directives.append(_StairDirective(
            direction="down", x=cx, y=cy, place=place,
        ))

    def fill_terrain(
        self,
        terrain: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> None:
        """Fill the inclusive rectangle between ``(x1, y1)`` and ``(x2, y2)``."""
        if terrain not in TERRAIN_CHAR_TO_TILE:
            raise ValueError(f"unknown terrain char: {terrain!r}")
        self._directives.append(_FillTerrainDirective(
            terrain=terrain, x1=x1, y1=y1, x2=x2, y2=y2,
        ))

    def set_start_pos(self, x, y: int = -1) -> None:
        """Place the player at MiniHack ``(x, y)``.

        Vendor-parity (Wave17i): accepts ``set_start_pos((x, y))`` or
        ``set_start_pos(x, y)``.
        """
        if isinstance(x, tuple) and len(x) == 2:
            cx, cy = int(x[0]), int(x[1])
        else:
            cx, cy = int(x), int(y)
        self._directives.append(_StartPosDirective(x=cx, y=cy))

    def set_goal_pos(self, x, y: int = -1) -> None:
        """Mark a goal tile.  Stored as a STAIRCASE_DOWN tile for symmetry
        with MiniHack's ``add_goal_pos == add_stair_down`` alias.
        """
        if isinstance(x, tuple) and len(x) == 2:
            cx, cy = int(x[0]), int(x[1])
        else:
            cx, cy = int(x), int(y)
        self._directives.append(_GoalPosDirective(x=cx, y=cy))

    # ------------------------------------------------------------------
    # Wave17i: missing vendor methods
    # Cite: vendor/minihack/minihack/level_generator.py add_altar/add_sink/
    #       add_gold/add_boulder/add_mazewalk.
    # ------------------------------------------------------------------
    def add_altar(
        self,
        place: Place = None,
        align: str = "noalign",
        type: str = "altar",
    ) -> None:
        """Place an altar tile.  Vendor add_altar(place, align, type)."""
        del align, type  # nethax has a single altar tile
        # Resolve a concrete (x, y) at factory time; here we emit a
        # FillTerrainDirective covering a 1×1 region.
        if isinstance(place, tuple) and len(place) == 2:
            x, y = int(place[0]), int(place[1])
            self._directives.append(_FillTerrainDirective(
                terrain="\\",  # backslash maps to THRONE; altar uses '_' which
                                # we substitute via an inline directive below.
                x1=x, y1=y, x2=x, y2=y,
            ))
            # Replace the throne tile with ALTAR via a direct override
            # directive (handled by writing the proper tile in pass 2).
            self._directives.append(_AltarOverride(x=x, y=y))
        else:
            self._directives.append(_AltarOverride(x=-1, y=-1, place=place))

    def add_sink(self, place: Place = None) -> None:
        """Place a sink (vendor add_sink).  nethax uses FOUNTAIN as a stand-in
        because no dedicated SINK tile exists yet."""
        if isinstance(place, tuple) and len(place) == 2:
            x, y = int(place[0]), int(place[1])
            self._directives.append(_FillTerrainDirective(
                terrain="{", x1=x, y1=y, x2=x, y2=y,
            ))
        else:
            self._directives.append(_SinkOverride(place=place))

    def add_gold(
        self,
        amount: int = 1,
        place: Place = None,
    ) -> None:
        """Spawn a gold pile.  Vendor add_gold(amount, place=(x, y))."""
        # Gold maps to OBJECTS table entry "gold piece".
        # We dispatch through the existing _ObjectDirective with a custom
        # quantity annotation.
        self._directives.append(_GoldDirective(amount=int(amount), place=place))

    def add_boulder(self, place: Place = None) -> None:
        """Add a boulder.  Vendor add_boulder(place=(x, y))."""
        # Reuse add_object: vendor "boulder" exists in OBJECTS.
        self._directives.append(_ObjectDirective(
            name="boulder", symbol=None, place=place, cursestate="uncursed",
        ))

    def add_mazewalk(
        self,
        coord=None,
        dir: str = "east",
    ) -> None:
        """Carve a recursive-backtracker maze starting at ``coord``.

        Vendor MAZEWALK directive (level_generator.py + dat/lib des-file
        ``MAZEWALK: place,dir``) carves a perfect maze across the entire
        map starting from ``coord`` and propagating in ``dir``.

        nethax implementation (Wave17i): records a directive that triggers
        a recursive-backtracker carve in the factory pass (replaces the
        legacy "open room" stand-in in canonical.py:175-186).
        """
        if isinstance(coord, tuple) and len(coord) == 2:
            x, y = int(coord[0]), int(coord[1])
        else:
            x, y = 0, 0
        self._directives.append(_MazeWalkDirective(
            x=x, y=y, direction=str(dir),
        ))

    # ---- Factory --------------------------------------------------------

    def get_factory(self) -> Callable[[jax.Array], EnvState]:
        """Return a ``(rng) -> EnvState`` closure that materialises the level.

        Calling the closure multiple times with the same ``rng`` is
        deterministic: directives that involve randomness consume keys split
        from the input.
        """
        directives = list(self._directives)
        rooms_meta = dict(self._room_directives)
        w, h = self.w, self.h
        fill = self.fill
        static = self._static

        def factory(rng: jax.Array) -> EnvState:
            return _apply_directives(
                self, rng, directives, rooms_meta, w, h, fill, static,
            )

        return factory


# ---------------------------------------------------------------------------
# Factory implementation (Python-side, not JIT'd)
# ---------------------------------------------------------------------------

def _apply_directives(
    lg: "LevelGenerator",
    rng: jax.Array,
    directives: List[Any],
    rooms_meta: dict,
    w: int,
    h: int,
    fill: str,
    static: StaticParams,
) -> EnvState:
    """Walk the directive list and produce a populated ``EnvState``."""
    # Reset captured build-trace metadata so tests see a fresh snapshot.
    lg.last_monster_entry_ids = []
    lg.last_object_entry_ids = []
    lg.last_trap_types = []
    lg.last_player_pos = None
    lg.last_goal_pos = None

    # 1. Allocate a default EnvState.
    state = EnvState.default(rng, static)

    # 2. Initialise terrain[0, 0] sub-region with the fill character.
    fill_tile = int(TERRAIN_CHAR_TO_TILE[fill])
    terrain_np = jnp.asarray(state.terrain)
    fill_block = jnp.full((h, w), jnp.int8(fill_tile), dtype=jnp.int8)
    terrain_np = terrain_np.at[0, 0, :h, :w].set(fill_block)

    # Per-room resolved bounding boxes filled in during the room pass.
    # Stored as (y1_row, x1_col, y2_row, x2_col).
    resolved_rooms: dict = {}

    # 3. Walk directives.  We split the input rng repeatedly so each random
    # decision gets independent keys; this preserves reproducibility.
    rng_pool = rng

    def _next_key():
        nonlocal rng_pool
        rng_pool, sub = jax.random.split(rng_pool)
        return sub

    # Track ground-stack depth per (row, col) so multiple add_object calls on
    # the same tile stack into successive slots.
    stack_index: dict = {}

    # Trap state buffer (we modify state.traps once at the end).
    trap_type_arr = jnp.asarray(state.traps.trap_type)
    # Trap state stores [num_levels, map_h, map_w] flattened across branches:
    # num_levels == n_branches * max_levels_per_branch.  For branch=0 level=0
    # the flat index is 0.
    trap_lvl_idx = 0

    # Ground-items array (Item pytree).
    ground = state.ground_items

    # Pass 1: resolve rooms (room placements are needed before other directives
    # that reference them by id).
    for d in directives:
        if isinstance(d, _RoomDirective):
            terrain_np, bbox = _resolve_and_carve_room(
                terrain_np, d, w, h, _next_key,
            )
            resolved_rooms[d.room_id] = bbox

    # Pass 2: everything else.
    for d in directives:
        if isinstance(d, _RoomDirective):
            continue   # already handled
        elif isinstance(d, _CorridorDirective):
            terrain_np = _carve_corridor(terrain_np, d.src, d.dst, w, h)
        elif isinstance(d, _DoorDirective):
            terrain_np = _place_door(terrain_np, d, w, h)
        elif isinstance(d, _FillTerrainDirective):
            terrain_np = _fill_terrain_rect(terrain_np, d, w, h)
        elif isinstance(d, _StairDirective):
            terrain_np, pos = _place_stair(
                terrain_np, d, w, h, resolved_rooms, _next_key,
            )
            if d.direction == "down" and lg.last_goal_pos is None:
                lg.last_goal_pos = pos
        elif isinstance(d, _GoalPosDirective):
            terrain_np = _set_tile(
                terrain_np, d.y, d.x, int(TileType.STAIRCASE_DOWN), w, h,
            )
            lg.last_goal_pos = (d.x, d.y)
        elif isinstance(d, _StartPosDirective):
            lg.last_player_pos = (d.x, d.y)
        elif isinstance(d, _MonsterDirective):
            pos_rc, mon_idx = _resolve_monster(
                d, terrain_np, w, h, resolved_rooms, _next_key,
            )
            state = _write_monster(state, pos_rc, mon_idx)
            lg.last_monster_entry_ids.append(mon_idx)
        elif isinstance(d, _TrapDirective):
            pos_rc, trap_type = _resolve_trap(
                d, terrain_np, w, h, resolved_rooms, _next_key,
            )
            trap_type_arr = trap_type_arr.at[
                trap_lvl_idx, pos_rc[0], pos_rc[1]
            ].set(jnp.int8(trap_type))
            lg.last_trap_types.append(trap_type)
        elif isinstance(d, _ObjectDirective):
            pos_rc, obj_idx = _resolve_object(
                d, terrain_np, w, h, resolved_rooms, _next_key,
            )
            ground, stack_index = _write_ground_item(
                ground, stack_index, pos_rc, obj_idx,
            )
            lg.last_object_entry_ids.append(obj_idx)
        elif isinstance(d, _AltarOverride):
            # Place an ALTAR tile.  Resolve coordinates if needed.
            if d.x >= 0 and d.y >= 0:
                row, col = d.y, d.x
            else:
                rc = _resolve_place(
                    d.place, terrain_np, w, h, resolved_rooms, _next_key,
                )
                if rc is None:
                    continue
                row, col = rc
            terrain_np = _set_tile(
                terrain_np, row, col, int(TileType.ALTAR), w, h,
            )
        elif isinstance(d, _SinkOverride):
            # No dedicated SINK tile in nethax — use FOUNTAIN as analogue.
            rc = _resolve_place(
                d.place, terrain_np, w, h, resolved_rooms, _next_key,
            )
            if rc is None:
                continue
            terrain_np = _set_tile(
                terrain_np, rc[0], rc[1], int(TileType.FOUNTAIN), w, h,
            )
        elif isinstance(d, _GoldDirective):
            # Gold pile — emit as a ground item with type "gold piece".
            gold_idx = _OBJECT_NAME_TO_IDX.get(
                "gold piece", _OBJECT_NAME_TO_IDX.get("gold", 0),
            )
            rc = _resolve_place(
                d.place, terrain_np, w, h, resolved_rooms, _next_key,
            )
            if rc is None:
                continue
            ground, stack_index = _write_ground_item(
                ground, stack_index, rc, gold_idx,
            )
            lg.last_object_entry_ids.append(gold_idx)
        elif isinstance(d, _MazeWalkDirective):
            # Wave17i: recursive-backtracker maze starting at (d.x, d.y).
            # Carves CORRIDOR tiles through a WALL-filled region.
            terrain_np = _carve_maze(
                terrain_np, d.x, d.y, w, h, _next_key,
            )
        else:
            # Defensive: an unknown directive class signals a programming bug.
            raise RuntimeError(f"unhandled directive type: {type(d).__name__}")

    # 4. Commit accumulated terrain/traps/grounds.
    new_traps = state.traps.replace(trap_type=trap_type_arr)
    state = state.replace(
        terrain=terrain_np,
        traps=new_traps,
        ground_items=ground,
    )

    # 5. Apply player start position (default: any free floor tile).
    if lg.last_player_pos is not None:
        px, py = lg.last_player_pos
        state = state.replace(
            player_pos=jnp.array([py, px], dtype=jnp.int16),
        )
    else:
        # Pick the first FLOOR tile we can find within the (h, w) region.
        start_rc = _find_first_floor_tile(terrain_np, w, h)
        if start_rc is not None:
            r, c = start_rc
            state = state.replace(
                player_pos=jnp.array([r, c], dtype=jnp.int16),
            )
            lg.last_player_pos = (int(c), int(r))

    return state


# ---------------------------------------------------------------------------
# Terrain helpers
# ---------------------------------------------------------------------------

def _set_tile(
    terrain_np: jax.Array, row: int, col: int, tile: int, w: int, h: int,
) -> jax.Array:
    """Set ``terrain[0, 0, row, col]`` if the cell is inside (h, w)."""
    if not (0 <= row < h and 0 <= col < w):
        return terrain_np
    return terrain_np.at[0, 0, row, col].set(jnp.int8(tile))


def _resolve_and_carve_room(
    terrain_np: jax.Array,
    d: _RoomDirective,
    w: int,
    h: int,
    next_key,
) -> Tuple[jax.Array, Tuple[int, int, int, int]]:
    """Pick a concrete bbox for the room directive and carve it.

    Returns the updated terrain plus the (y1, x1, y2, x2) interior bbox in
    nethax row/col convention.
    """
    # MiniHack coords: x = col, y = row.  Random values use next_key.
    rw = d.w if d.w > 0 else int(jax.random.randint(next_key(), (), 3, min(7, max(4, w // 2))))
    rh = d.h if d.h > 0 else int(jax.random.randint(next_key(), (), 3, min(6, max(4, h // 2))))

    # Constrain room interior to (h, w) including a 1-cell wall margin.
    rw = max(1, min(rw, w - 2))
    rh = max(1, min(rh, h - 2))

    if d.x >= 0:
        x1 = d.x
    else:
        max_x = max(1, w - rw - 1)
        x1 = int(jax.random.randint(next_key(), (), 1, max_x + 1))
    if d.y >= 0:
        y1 = d.y
    else:
        max_y = max(1, h - rh - 1)
        y1 = int(jax.random.randint(next_key(), (), 1, max_y + 1))

    x2 = min(x1 + rw - 1, w - 2)
    y2 = min(y1 + rh - 1, h - 2)

    # Carve walls then floor.
    wall = int(TileType.WALL)
    floor = int(TileType.FLOOR)
    # Wall border (one cell outside interior).
    for r in range(max(0, y1 - 1), min(h, y2 + 2)):
        for c in range(max(0, x1 - 1), min(w, x2 + 2)):
            if r < y1 or r > y2 or c < x1 or c > x2:
                terrain_np = terrain_np.at[0, 0, r, c].set(jnp.int8(wall))
    # Floor interior.
    for r in range(y1, y2 + 1):
        for c in range(x1, x2 + 1):
            terrain_np = terrain_np.at[0, 0, r, c].set(jnp.int8(floor))

    return terrain_np, (y1, x1, y2, x2)


def _carve_corridor(
    terrain_np: jax.Array,
    src: Tuple[int, int],
    dst: Tuple[int, int],
    w: int,
    h: int,
) -> jax.Array:
    """L-shaped corridor between (x1, y1) and (x2, y2).

    Cells that are already FLOOR are left alone; everything else becomes
    CORRIDOR.
    """
    x1, y1 = src
    x2, y2 = dst
    corridor = int(TileType.CORRIDOR)
    floor = int(TileType.FLOOR)
    # Horizontal then vertical: row y1 from min(x1,x2) to max(x1,x2),
    # then column x2 from min(y1,y2) to max(y1,y2).
    for c in range(min(x1, x2), max(x1, x2) + 1):
        if 0 <= y1 < h and 0 <= c < w:
            existing = int(terrain_np[0, 0, y1, c])
            if existing != floor:
                terrain_np = terrain_np.at[0, 0, y1, c].set(jnp.int8(corridor))
    for r in range(min(y1, y2), max(y1, y2) + 1):
        if 0 <= r < h and 0 <= x2 < w:
            existing = int(terrain_np[0, 0, r, x2])
            if existing != floor:
                terrain_np = terrain_np.at[0, 0, r, x2].set(jnp.int8(corridor))
    return terrain_np


def _place_door(
    terrain_np: jax.Array, d: _DoorDirective, w: int, h: int,
) -> jax.Array:
    tile = TileType.CLOSED_DOOR if d.state != "open" else TileType.OPEN_DOOR
    if d.state == "nodoor":
        tile = TileType.FLOOR
    return _set_tile(terrain_np, d.y, d.x, int(tile), w, h)


def _fill_terrain_rect(
    terrain_np: jax.Array, d: _FillTerrainDirective, w: int, h: int,
) -> jax.Array:
    """Inclusive rectangle fill at terrain[0, 0]."""
    tile = int(TERRAIN_CHAR_TO_TILE[d.terrain])
    y_lo, y_hi = sorted((d.y1, d.y2))
    x_lo, x_hi = sorted((d.x1, d.x2))
    y_lo = max(0, y_lo); y_hi = min(h - 1, y_hi)
    x_lo = max(0, x_lo); x_hi = min(w - 1, x_hi)
    if y_lo > y_hi or x_lo > x_hi:
        return terrain_np
    block = jnp.full((y_hi - y_lo + 1, x_hi - x_lo + 1), jnp.int8(tile), dtype=jnp.int8)
    return terrain_np.at[0, 0, y_lo:y_hi + 1, x_lo:x_hi + 1].set(block)


def _place_stair(
    terrain_np: jax.Array,
    d: _StairDirective,
    w: int,
    h: int,
    resolved_rooms: dict,
    next_key,
) -> Tuple[jax.Array, Tuple[int, int]]:
    tile = (
        int(TileType.STAIRCASE_UP) if d.direction == "up"
        else int(TileType.STAIRCASE_DOWN)
    )
    # Coordinate priority: explicit (x, y) > place > random.
    if d.x >= 0 and d.y >= 0:
        col, row = d.x, d.y
    else:
        rc = _resolve_place(d.place, terrain_np, w, h, resolved_rooms, next_key)
        if rc is None:
            return terrain_np, (0, 0)
        row, col = rc
    terrain_np = _set_tile(terrain_np, row, col, tile, w, h)
    return terrain_np, (col, row)


def _find_first_floor_tile(
    terrain_np: jax.Array, w: int, h: int,
) -> Optional[Tuple[int, int]]:
    """Linear scan for the first FLOOR cell in terrain[0, 0, :h, :w]."""
    sub = terrain_np[0, 0, :h, :w]
    floor = int(TileType.FLOOR)
    mask = (sub == floor)
    flat = mask.reshape(-1)
    # jnp.argmax on bool returns first True index, or 0 if all False.
    any_true = bool(jnp.any(flat))
    if not any_true:
        return None
    idx = int(jnp.argmax(flat))
    return (idx // w, idx % w)


# ---------------------------------------------------------------------------
# Placement resolution
# ---------------------------------------------------------------------------

def _resolve_place(
    place: Place,
    terrain_np: jax.Array,
    w: int,
    h: int,
    resolved_rooms: dict,
    next_key,
) -> Optional[Tuple[int, int]]:
    """Convert a ``place`` spec to a concrete ``(row, col)`` cell.

    Returns ``None`` only if the resolution failed entirely (no candidate
    tile available).
    """
    if isinstance(place, tuple):
        col, row = place
        return (int(row), int(col))
    if isinstance(place, str) and place in resolved_rooms:
        y1, x1, y2, x2 = resolved_rooms[place]
        return _random_cell_in_rect(next_key(), y1, x1, y2, x2)
    # place is None or unknown string → random floor cell on the level.
    return _random_floor_cell(terrain_np, w, h, next_key())


def _random_cell_in_rect(
    rng: jax.Array, y1: int, x1: int, y2: int, x2: int,
) -> Tuple[int, int]:
    rh = y2 - y1 + 1
    rw = x2 - x1 + 1
    k1, k2 = jax.random.split(rng)
    dy = int(jax.random.randint(k1, (), 0, max(1, rh)))
    dx = int(jax.random.randint(k2, (), 0, max(1, rw)))
    return (y1 + dy, x1 + dx)


def _random_floor_cell(
    terrain_np: jax.Array, w: int, h: int, rng: jax.Array,
) -> Optional[Tuple[int, int]]:
    """Pick a uniformly-random FLOOR tile in the (h, w) sub-region."""
    sub = terrain_np[0, 0, :h, :w]
    floor = int(TileType.FLOOR)
    mask = (sub == floor).reshape(-1)
    count = int(jnp.sum(mask))
    if count == 0:
        return None
    probs = mask.astype(jnp.float32) / count
    idx = int(jax.random.choice(rng, h * w, p=probs))
    return (idx // w, idx % w)


# ---------------------------------------------------------------------------
# Monster / object / trap directive resolution
# ---------------------------------------------------------------------------

def _resolve_monster(
    d: _MonsterDirective,
    terrain_np: jax.Array,
    w: int,
    h: int,
    resolved_rooms: dict,
    next_key,
) -> Tuple[Tuple[int, int], int]:
    """Return ``((row, col), monster_idx)`` for a monster directive."""
    if d.name == "random":
        # Wave 5+ TODO: depth-aware random pick.  For Wave 4 we substitute a
        # deterministic fallback so the directive always produces a monster.
        idx = _MONSTER_NAME_TO_IDX.get("gnome", 0)
    else:
        if d.name not in _MONSTER_NAME_TO_IDX:
            raise KeyError(
                f"unknown monster name {d.name!r}; not present in MONSTERS table"
            )
        idx = _MONSTER_NAME_TO_IDX[d.name]
    rc = _resolve_place(d.place, terrain_np, w, h, resolved_rooms, next_key)
    if rc is None:
        rc = (0, 0)
    return rc, idx


def _resolve_object(
    d: _ObjectDirective,
    terrain_np: jax.Array,
    w: int,
    h: int,
    resolved_rooms: dict,
    next_key,
) -> Tuple[Tuple[int, int], int]:
    """Return ``((row, col), object_idx)`` for an object directive."""
    if d.name == "random":
        idx = _OBJECT_NAME_TO_IDX.get("apple", 0)
    else:
        if d.name not in _OBJECT_NAME_TO_IDX:
            raise KeyError(
                f"unknown object name {d.name!r}; not present in OBJECTS table"
            )
        idx = _OBJECT_NAME_TO_IDX[d.name]
    rc = _resolve_place(d.place, terrain_np, w, h, resolved_rooms, next_key)
    if rc is None:
        rc = (0, 0)
    return rc, idx


def _resolve_trap(
    d: _TrapDirective,
    terrain_np: jax.Array,
    w: int,
    h: int,
    resolved_rooms: dict,
    next_key,
) -> Tuple[Tuple[int, int], int]:
    if d.name == "random":
        trap_kind = int(TrapType.TELEP_TRAP)
    else:
        trap_kind = int(TRAP_NAME_TO_TYPE[d.name])
    rc = _resolve_place(d.place, terrain_np, w, h, resolved_rooms, next_key)
    if rc is None:
        rc = (0, 0)
    return rc, trap_kind


# ---------------------------------------------------------------------------
# EnvState writers
# ---------------------------------------------------------------------------

def _write_monster(
    state: EnvState, pos_rc: Tuple[int, int], mon_idx: int,
) -> EnvState:
    """Populate the first empty monster_ai slot with a freshly placed monster.

    Wave 4 simplification: we use a Python-side scan for the first
    ``alive=False`` slot (this whole function runs on the host).
    """
    mai = state.monster_ai
    alive_np = jnp.asarray(mai.alive)
    # Find first inactive slot.
    free_mask = ~alive_np
    if not bool(jnp.any(free_mask)):
        # No room — drop silently.  (Wave 5: surface a warning.)
        return state
    slot = int(jnp.argmax(free_mask.astype(jnp.int8)))

    row, col = pos_rc
    mon_idx_clipped = max(0, min(mon_idx, int(_BASE_AC.shape[0]) - 1))

    new_mai = mai.replace(
        pos=mai.pos.at[slot].set(jnp.array([row, col], dtype=jnp.int16)),
        hp=mai.hp.at[slot].set(jnp.int32(8)),
        hp_max=mai.hp_max.at[slot].set(jnp.int32(8)),
        alive=mai.alive.at[slot].set(jnp.bool_(True)),
        ac=mai.ac.at[slot].set(_BASE_AC[mon_idx_clipped]),
        is_large=mai.is_large.at[slot].set(_IS_LARGE[mon_idx_clipped]),
        attack_dice_n=mai.attack_dice_n.at[slot].set(
            _ATK_DICE_N[mon_idx_clipped].astype(jnp.int8)
        ),
        attack_dice_sides=mai.attack_dice_sides.at[slot].set(
            _ATK_DICE_S[mon_idx_clipped].astype(jnp.int8)
        ),
        asleep=mai.asleep.at[slot].set(jnp.bool_(False)),
        peaceful=mai.peaceful.at[slot].set(jnp.bool_(False)),
    )
    return state.replace(monster_ai=new_mai)


def _write_ground_item(
    ground: Any,
    stack_index: dict,
    pos_rc: Tuple[int, int],
    obj_idx: int,
) -> Tuple[Any, dict]:
    """Stamp an item into the top of the ground stack at ``pos_rc``.

    Stack overflow (> MAX_GROUND_STACK items on one tile) drops the new item.
    """
    row, col = pos_rc
    key = (row, col)
    depth = stack_index.get(key, 0)
    if depth >= MAX_GROUND_STACK:
        return ground, stack_index

    entry = OBJECTS[obj_idx] if 0 <= obj_idx < len(OBJECTS) else None
    cat_value = int(entry.class_) if entry is not None else int(ObjectClass.FOOD_CLASS)
    weight = entry.weight if entry is not None else 0

    new_ground = ground.replace(
        category=ground.category.at[0, 0, row, col, depth].set(jnp.int8(cat_value)),
        type_id=ground.type_id.at[0, 0, row, col, depth].set(jnp.int16(obj_idx)),
        buc_status=ground.buc_status.at[0, 0, row, col, depth].set(jnp.int8(0)),
        enchantment=ground.enchantment.at[0, 0, row, col, depth].set(jnp.int8(0)),
        charges=ground.charges.at[0, 0, row, col, depth].set(jnp.int8(0)),
        identified=ground.identified.at[0, 0, row, col, depth].set(jnp.bool_(False)),
        quantity=ground.quantity.at[0, 0, row, col, depth].set(jnp.int16(1)),
        weight=ground.weight.at[0, 0, row, col, depth].set(jnp.int32(weight)),
        ac_bonus=ground.ac_bonus.at[0, 0, row, col, depth].set(jnp.int8(0)),
        is_two_handed=ground.is_two_handed.at[0, 0, row, col, depth].set(jnp.bool_(False)),
    )
    new_stack = dict(stack_index)
    new_stack[key] = depth + 1
    return new_ground, new_stack


# ---------------------------------------------------------------------------
# Wave17i: recursive-backtracker maze carver for ``add_mazewalk``.
# Cite: vendor MiniHack uses NetHack's MAZEWALK des-file directive which
# triggers a recursive maze dig in mklev.c::makemaz / sp_lev.c::create_maze.
# We approximate the layout with a standard recursive-backtracker on a
# grid that walks in 2-cell strides (the same algorithm used by NetHack's
# walkfrom in mklev.c).
# ---------------------------------------------------------------------------


def _carve_maze(
    terrain_np: jax.Array,
    start_x: int,
    start_y: int,
    w: int,
    h: int,
    next_key,
) -> jax.Array:
    """Recursive-backtracker maze carve into the (h, w) top-left subregion.

    The maze is carved with WALL tiles separating CORRIDOR cells.  We walk in
    2-cell steps so each "stride" carves both the bridge cell and the target
    cell, matching the vendor's walkfrom() behaviour
    (vendor/nethack/src/mklev.c::walkfrom).

    Args:
        terrain_np: current terrain array.
        start_x:    starting column.
        start_y:    starting row.
        w, h:       active map extent.
        next_key:   PRNG factory for shuffling neighbour order.

    Returns:
        terrain_np with maze carved in.
    """
    import numpy as _np

    # Materialise the sub-region into a host-side numpy array for the
    # iterative carve (this entire function runs at Python init time, so
    # converting to numpy is fine).
    sub = _np.asarray(terrain_np[0, 0, :h, :w])
    wall = int(TileType.WALL)
    corridor = int(TileType.CORRIDOR)
    # Fill with WALL first.
    sub = _np.full_like(sub, wall, dtype=sub.dtype)

    sx = max(0, min(int(start_x), w - 1))
    sy = max(0, min(int(start_y), h - 1))
    # Align to odd coords so the 2-stride walk stays in-bounds.
    if sx % 2 == 0:
        sx = min(w - 1, sx + 1)
    if sy % 2 == 0:
        sy = min(h - 1, sy + 1)

    visited = _np.zeros((h, w), dtype=_np.bool_)
    stack = [(sy, sx)]
    visited[sy, sx] = True
    sub[sy, sx] = corridor

    # Use jax PRNG to derive a seed for the host-side shuffle so the
    # function stays deterministic with respect to the input key.
    key = next_key()
    rng_seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    pyrng = _np.random.RandomState(rng_seed)

    while stack:
        r, c = stack[-1]
        neighbours = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                neighbours.append((nr, nc, dr, dc))
        if not neighbours:
            stack.pop()
            continue
        idx = pyrng.randint(0, len(neighbours))
        nr, nc, dr, dc = neighbours[idx]
        # Carve bridge cell + neighbour.
        br, bc = r + dr // 2, c + dc // 2
        sub[br, bc] = corridor
        sub[nr, nc] = corridor
        visited[nr, nc] = True
        stack.append((nr, nc))

    # Write the carved sub-region back.
    new_terrain = terrain_np.at[0, 0, :h, :w].set(jnp.asarray(sub))
    return new_terrain
