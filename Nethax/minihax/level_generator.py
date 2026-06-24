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
from Nethax.nethax.subsystems.features import DoorState
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
    # HideNSeek line-of-sight overlays.  Canonical TileType has no walkable
    # CLOUD tile, so we map both vendor glyphs to TREE: TREE is walkable AND
    # opaque (vendor/nethack vision.c:166-169) which matches CLOUD's role as
    # a hide-mechanic occluder.  vendor des: hidenseek*.des REPLACE_TERRAIN.
    "T": TileType.TREE,
    "C": TileType.TREE,
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


#: MiniHack door-state string to nethax ``DoorState`` (vendor rm.h doormask).
#: ``random`` is treated as ``closed`` here (deterministic) — the LG does not
#: roll door states.  ``nodoor`` leaves the doorway as floor (state GONE).
_DOOR_STATE_VALUE: dict = {
    "open":   int(DoorState.OPEN),
    "closed": int(DoorState.CLOSED),
    "locked": int(DoorState.LOCKED),
    "random": int(DoorState.CLOSED),
    "nodoor": int(DoorState.GONE),
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


def _build_monster_group_flags():
    """Per-monster (G_SGROUP, G_LGROUP) booleans for m_initgrp triggering.

    Vendor makemon.c:1370-1377 — a freshly-made monster with G_SGROUP (or
    G_LGROUP) spawns a same-type group via m_initgrp.  Cite monst.c geno
    flags; mirrored on ``MonsterEntry.generation_mask``.
    """
    import numpy as _np
    from Nethax.nethax.constants.monsters import MONSTERS, G_SGROUP, G_LGROUP
    n = len(MONSTERS)
    sg = _np.zeros(n, dtype=bool)
    lg = _np.zeros(n, dtype=bool)
    for i, m in enumerate(MONSTERS):
        gm = int(m.generation_mask)
        sg[i] = bool(gm & G_SGROUP)
        lg[i] = bool(gm & G_LGROUP)
    return sg, lg


_MON_SGROUP, _MON_LGROUP = _build_monster_group_flags()


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
class _ReplaceTerrainDirective:
    """REPLACE_TERRAIN: probabilistic per-cell tile swap.

    Mirrors vendor des ``REPLACE_TERRAIN:(x1,y1,x2,y2), from, to, chance%``.
    Used by HideNSeek to scatter TREE/CLOUD line-of-sight occluders.
    """
    from_terrain: str
    to_terrain: str
    x1: int
    y1: int
    x2: int
    y2: int
    chance: int        # 0..100


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


@dataclasses.dataclass
class _StartingInventoryDirective:
    """Pre-populate a starting-inventory slot at reset time.

    Mirrors the vendor des ``INV:`` directive (placed-on-hero starting kit).
    Used by LavaCross-Levitate ``-Inv-`` variants where the levitation item
    must be carried at episode start rather than scattered on the floor.
    """
    category: int       # ItemCategory enum value (e.g. POTION=8, RING=4)
    type_id: int        # vendor object index (e.g. POT_LEVITATION=278)
    quantity: int
    weight: int
    buc_status: int     # 0=unknown / 1=cursed / 2=uncursed / 3=blessed
    identified: bool


@dataclasses.dataclass
class _SetMapDirective:
    """A literal ``MAP`` block from a vendor ``.des`` file.

    Source: vendor des-file ``MAP ... ENDMAP`` grids.  Each ``row`` is one
    terrain line in MiniHack ``(x=col, y=row)`` order; the level is stamped
    starting at the top-left of the active ``(h, w)`` region.  Unlike the
    default ``fill`` block, every cell — *including* spaces (which map to
    ``VOID`` per ``TERRAIN_CHAR_TO_TILE``) — is written, so the MAP block
    is authoritative and the level is correctly bounded by stone/void rather
    than leaking open FLOOR into the rest of the 80x21 grid.
    """
    rows: Tuple[str, ...]


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

    def replace_terrain(
        self,
        from_terrain: str,
        to_terrain: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        chance: int = 100,
    ) -> None:
        """Probabilistically replace ``from_terrain`` with ``to_terrain``.

        Mirrors vendor des ``REPLACE_TERRAIN:(x1,y1,x2,y2), from, to, chance%``.
        Per-cell Bernoulli sampling at factory time uses the directive-walk
        PRNG so generation is deterministic per reset key.
        """
        if from_terrain not in TERRAIN_CHAR_TO_TILE:
            raise ValueError(f"unknown from_terrain char: {from_terrain!r}")
        if to_terrain not in TERRAIN_CHAR_TO_TILE:
            raise ValueError(f"unknown to_terrain char: {to_terrain!r}")
        c = max(0, min(100, int(chance)))
        self._directives.append(_ReplaceTerrainDirective(
            from_terrain=from_terrain, to_terrain=to_terrain,
            x1=x1, y1=y1, x2=x2, y2=y2, chance=c,
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

    def set_map(self, rows) -> None:
        """Ingest a literal vendor ``MAP`` block.

        Source: vendor des-file ``MAP ... ENDMAP`` grid (e.g.
        ``vendor/minihack/minihack/dat/lava_crossing.des``).  ``rows`` is an
        iterable of strings, one per terrain line, in MiniHack ``(x, y)`` =
        (col, row) order.  The grid is stamped authoritatively at factory
        time: every glyph — including spaces, which resolve to ``VOID`` —
        is written into the terrain so the level is bounded by stone rather
        than the LG's default open-FLOOR fill.
        """
        clean = tuple(str(r) for r in rows)
        self._directives.append(_SetMapDirective(rows=clean))

    def add_random_corridors(self) -> None:
        """Vendor ``RANDOM_CORRIDORS`` directive (no-op stand-in).

        Source: vendor des-file ``RANDOM_CORRIDORS`` (e.g.
        ``vendor/minihack/minihack/dat/corridor2.des``) carves connecting
        corridors between every declared room using NetHack's
        ``join()``/``makecorridors`` (vendor/nethack/src/sp_lev.c).  nethax
        room placement already lays floor; explicit corridor carving is left
        to ``add_corridor`` directives.  This method exists so the des
        emitter drives the real LevelGenerator instead of falling back.
        """
        # No directive emitted: rooms are already navigable floor regions.
        return None

    def add_starting_inventory_item(
        self,
        category: int,
        type_id: int,
        *,
        quantity: int = 1,
        weight: int = 0,
        buc_status: int = 2,  # _BUC_UNCURSED — matches vendor ini_inv defaults
        identified: bool = True,
    ) -> None:
        """Place an item directly into the hero's starting inventory.

        Mirrors the vendor des ``INV:`` directive.  Used by LavaCross-Levitate
        ``-Inv-`` variants whose vendor counterparts ship with the levitation
        item already carried (vendor/minihack/minihack/envs/skills_lava.py
        ``MiniHackLCLevitatePotionInv`` / ``MiniHackLCLevitateRingInv``).

        Args mirror ``Nethax.nethax.subsystems.inventory.make_item``.
        """
        self._directives.append(_StartingInventoryDirective(
            category=int(category),
            type_id=int(type_id),
            quantity=int(quantity),
            weight=int(weight),
            buc_status=int(buc_status),
            identified=bool(identified),
        ))

    def mazewalk(self, row=None, col=None, direction: str = "east") -> None:
        """Vendor ``MAZEWALK`` directive via row/col emitter kwargs.

        Source: vendor des-file ``MAZEWALK: place,dir`` (e.g.
        ``vendor/minihack/minihack/dat/mazewalk.des``).  Adapter passes
        ``row``/``col`` (nethax convention); forward to ``add_mazewalk``
        which records a recursive-backtracker carve directive.
        """
        x = 0 if col is None else int(col)
        y = 0 if row is None else int(row)
        self.add_mazewalk(coord=(x, y), dir=direction)

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

    # 1. Allocate the base EnvState.
    #
    # Under NLE_BYTEPARITY, route the bootstrap through ``NethaxEnv.reset``
    # so the ISAAC64 CORE stream is advanced through the full vendor
    # init_objects -> role_init -> init_dungeons -> u_init -> mklev sequence
    # (339 pre-cascade draws + the Archeologist u_init optional-item rn2
    # cascade at u_init.c:652-660).  This produces a state whose
    # ``vendor_rng``, character stats, starting inventory (incl. the
    # OIL_LAMP / TIN_OPENER / MAGIC_MARKER bonus item at slot 8) and
    # DISP-stream offsets are byte-aligned with vendor MiniHack.  LG
    # directives below then overwrite the dungeon-shaped fields
    # (terrain, features, traps, ground_items, monster_ai, FOV) on top of
    # the vendor-aligned base — no inventory or RNG cascade work is
    # needed in this factory anymore.
    #
    # Outside NLE_BYTEPARITY we keep the lightweight default path: minihax
    # consumers in Threefry mode don't care about ISAAC64 alignment and
    # spinning up a full NethaxEnv.reset adds non-trivial latency.
    from Nethax.nethax.parity_mode import use_vendor_rng as _use_vrng_bootstrap
    if _use_vrng_bootstrap():
        from Nethax.nethax.env import NethaxEnv as _NethaxEnv
        from Nethax.nethax.constants.roles import Role as _Role
        from Nethax.nethax.constants.races import Race as _Race
        from Nethax.nethax.subsystems.features import FeaturesState as _FeaturesState
        from Nethax.nethax.subsystems.traps import TrapState as _TrapState
        from Nethax.nethax.subsystems.monster_ai import (
            make_monster_ai_state as _make_monster_ai_state,
        )
        from Nethax.nethax.state import _empty_ground_items_array as _empty_gi
        # NethaxEnv.reset's vendor-rng branch indexes rng[0]/rng[1] to
        # rebuild the uint64 ISAAC64 seed (env.py:168-170).  Callers from
        # the minihax harness pass a typed PRNGKey (jax.random.key(...))
        # which is 0-D and cannot be subscripted; unwrap to the raw
        # uint32 pair.
        try:
            _raw_key = jax.random.key_data(rng)
        except (TypeError, ValueError):
            _raw_key = rng
        _engine = _NethaxEnv(static=static)
        # Archeologist-Human-Lawful is the canonical MiniHack character
        # ("arc-hum-law-mal" — .test_runs/minihax_byteparity.py:149).
        # ``fast_reset=True``: skip mklev dungeon-gen / pet spawn / view_from
        # since LG directives below stamp the terrain authoritatively and the
        # ``default_lit`` block at the tail of this factory seeds FoV.
        # The ISAAC64 stream is still advanced through init_objects ->
        # role_init -> init_dungeons -> u_init so descr_idx + inventory
        # remain byte-aligned with vendor MiniHack.
        state, _ = _engine.reset(
            _raw_key,
            role=_Role.ARCHEOLOGIST,
            race=_Race.HUMAN,
            alignment=0,
            fast_reset=True,
        )
        # NethaxEnv.reset populated the state with a full vendor dungeon
        # level — rooms, fountains, sleeping monsters, dropped items,
        # traps.  LG owns terrain authorship in minihax, so wipe those
        # entity planes back to EnvState.default empties before applying
        # LG directives.  Vendor-aligned bits we want to KEEP are:
        #   - vendor_rng / vendor_rng_disp (ISAAC64 stream offsets)
        #   - descr_idx (object-description shuffle)
        #   - inventory (Archeologist ini_inv + u_init rn2 cascade bonus)
        #   - player stats (HP / AC / role / race / align / luck)
        #   - messages (role-intro line)
        _b = static.n_branches
        _l = static.max_levels_per_branch
        _hf = static.map_h
        _wf = static.map_w
        state = state.replace(
            terrain=jnp.zeros((_b, _l, _hf, _wf), dtype=jnp.int8),
            explored=jnp.zeros((_b, _l, _hf, _wf), dtype=jnp.bool_),
            visible=jnp.zeros((_hf, _wf), dtype=jnp.bool_),
            last_seen_terrain=jnp.full(
                (_b, _l, _hf, _wf), -1, dtype=jnp.int8,
            ),
            features=_FeaturesState.default(
                num_levels=_b * _l, map_h=_hf, map_w=_wf,
            ),
            traps=_TrapState.default(
                num_levels=_b * _l, map_h=_hf, map_w=_wf,
            ),
            ground_items=_empty_gi(_b, _l, _hf, _wf),
            monster_ai=_make_monster_ai_state(),
        )
    else:
        state = EnvState.default(rng, static)
        # Default (Threefry) mode: EnvState.default leaves ``vendor_rng`` as a
        # constant empty ISAAC64 stream.  The Room-Random / -Monster / -Dark
        # placement wrappers (canonical.py ``_wrap_*_room_placement``) draw the
        # player-spawn and stair cells from ``state.vendor_rng``, so an unseeded
        # stream makes EVERY reset produce an identical layout (player + stair
        # fixed regardless of the episode key) — unlike real MiniHack, which
        # randomizes placement each episode.  Seed the stream from the episode
        # ``rng`` so default-mode layouts vary per-episode.  (Byteparity mode
        # seeds ``vendor_rng`` via ``NethaxEnv.reset`` above and is untouched.)
        from Nethax.nethax import vendor_rng as _vrng_seed_mod
        _iso_seed = jax.random.randint(
            rng, (), 0, jnp.iinfo(jnp.int32).max, dtype=jnp.int32
        ).astype(jnp.uint64)
        state = state.replace(vendor_rng=_vrng_seed_mod.init_jax(_iso_seed))

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

    # Accumulate (row, col, DoorState) so doors get their open/closed/locked
    # status written into ``state.features.door_state`` at commit time.
    door_states: List[Tuple[int, int, int]] = []

    # Accumulate starting-inventory directives (vendor des INV: equivalent).
    # Materialised into ``state.inventory`` once at the end so item letters
    # are assigned positionally (see InventoryState.from_items).
    starting_inv: List[_StartingInventoryDirective] = []

    # Trap state buffer (we modify state.traps once at the end).
    trap_type_arr = jnp.asarray(state.traps.trap_type)
    # Trap state stores [num_levels, map_h, map_w] flattened across branches:
    # num_levels == n_branches * max_levels_per_branch.  For branch=0 level=0
    # the flat index is 0.
    trap_lvl_idx = 0

    # Ground-items array (Item pytree).
    ground = state.ground_items

    # Pass 0: stamp literal MAP blocks before anything else so subsequent
    # directives (stairs, objects) write on top of the authoritative grid.
    # Source: vendor des ``MAP ... ENDMAP`` (e.g. lava_crossing.des).  A MAP
    # block is authoritative: clear the default open-FLOOR fill to VOID first
    # (mirrors vendor ``INIT_MAP:solidfill,' '`` stone) so the level is bounded
    # by stone, then stamp the grid on top.
    has_map = any(isinstance(d, _SetMapDirective) for d in directives)
    if has_map:
        void_block = jnp.full((h, w), jnp.int8(int(TileType.VOID)), dtype=jnp.int8)
        terrain_np = terrain_np.at[0, 0, :h, :w].set(void_block)
        for d in directives:
            if isinstance(d, _SetMapDirective):
                terrain_np = _stamp_map_block(terrain_np, d.rows, w, h)

    # Pass 1: resolve rooms (room placements are needed before other directives
    # that reference them by id).
    for d in directives:
        if isinstance(d, _RoomDirective):
            terrain_np, bbox = _resolve_and_carve_room(
                terrain_np, d, w, h, _next_key,
            )
            resolved_rooms[d.room_id] = bbox

    # Vendor mklev opens with a 4-draw stair selection block
    # (rn2(3), rn2(2), rn2(W), rn2(W)) at offsets 339-342 — see
    # .test_runs/full_init_rn2_trace_room_ultimate_15x15_seed0.txt:344-347.
    # When this LG run is processing a single-room env with monsters AND
    # we're in vendor_rng mode, consume the prefix here so subsequent
    # ``_resolve_monster`` calls see the same vrng offset vendor's
    # makemon does.  Single-room + has-monster matches Room-Monster and
    # Room-Ultimate; Trap/Random/Dark wrappers handle the prefix
    # themselves and don't have monster directives.
    has_monster_dir = any(isinstance(d, _MonsterDirective) for d in directives)
    # Room envs carve via _FillTerrainDirective (not _RoomDirective) so
    # ``resolved_rooms`` is empty.  Derive the room bbox from the FLOOR-fill
    # directive's rect (the fill is applied later in pass 2, so we can't
    # read it off ``terrain_np`` yet — read it off the directive instead)
    # so the 4-prefix uses vendor's rn2(W) modulus and ``_resolve_monster``
    # lands monsters in-room (vs the (10, 39) map-center fallback when
    # room_w defaults to map_w=80).
    if not resolved_rooms:
        _floor_glyph = "."
        for _fd in directives:
            if (
                isinstance(_fd, _FillTerrainDirective)
                and _fd.terrain == _floor_glyph
            ):
                resolved_rooms["__carved_fill__"] = (
                    int(_fd.y1), int(_fd.x1), int(_fd.y2), int(_fd.x2),
                )
                break

    _mklev_stair_cell = None  # (row, col) of the vendor mkstairs down-stair
    # NOTE: not gated on ``_use_vendor_rng_dl()``.  The Monster/Ultimate room
    # wrappers (canonical.py ``_wrap_monster_room_placement`` etc.) consume
    # ``state.vendor_rng`` for layout in *both* parity modes — the down-stair
    # cell must be stamped in both too, else default (Threefry) mode produces a
    # goal-less level and the agent can never reach stairs-down (0% transfer).
    if (
        state is not None
        and has_monster_dir
        and len(resolved_rooms) == 1
    ):
        from Nethax.nethax import vendor_rng as _vendor_rng
        ry1, rx1, ry2, rx2 = next(iter(resolved_rooms.values()))
        room_w = max(1, rx2 - rx1 + 1)
        room_h = max(1, ry2 - ry1 + 1)
        vrng = state.vendor_rng
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))
        # mkstairs down-stair offset within the room (rn2(W), rn2(W)).
        vrng, _stair_xoff = _vendor_rng.rn2_jax(vrng, jnp.int32(room_w))
        vrng, _stair_yoff = _vendor_rng.rn2_jax(vrng, jnp.int32(room_h))
        state = state.replace(vendor_rng=vrng)
        _mklev_stair_cell = (ry1 + int(_stair_yoff), rx1 + int(_stair_xoff))

    # Pass 2: everything else.
    for d in directives:
        if isinstance(d, _RoomDirective):
            continue   # already handled
        elif isinstance(d, _SetMapDirective):
            continue   # stamped in pass 0
        elif isinstance(d, _CorridorDirective):
            terrain_np = _carve_corridor(terrain_np, d.src, d.dst, w, h)
        elif isinstance(d, _DoorDirective):
            terrain_np = _place_door(terrain_np, d, w, h)
            # Record the door's open/closed/locked status so the engine
            # treats it correctly.  Movement code reads
            # ``state.features.door_state`` (DoorState enum) — NOT just the
            # terrain tile — to decide whether a closed door is locked
            # (vendor: rm.h D_CLOSED/D_LOCKED; engine action_dispatch.py:676).
            # Without this the LG-authored locked doors in KeyRoom /
            # MultiRoom-Locked / LockedDoor would default to D_NODOOR (0) and
            # the agent could walk straight through.
            if 0 <= d.y < h and 0 <= d.x < w:
                door_states.append((d.y, d.x, _DOOR_STATE_VALUE[d.state]))
        elif isinstance(d, _FillTerrainDirective):
            terrain_np = _fill_terrain_rect(terrain_np, d, w, h)
        elif isinstance(d, _ReplaceTerrainDirective):
            terrain_np = _replace_terrain_rect(
                terrain_np, d, w, h, _next_key,
            )
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
            # Build the occupancy set (earlier monsters + the down-stair) so
            # enexto places m_initgrp members on free cells like vendor.
            _occ = set()
            if _mklev_stair_cell is not None:
                _occ.add(_mklev_stair_cell)
            import numpy as _np_occ
            _al = _np_occ.asarray(state.monster_ai.alive)
            _mp = _np_occ.asarray(state.monster_ai.pos)
            for _si in _np_occ.where(_al)[0]:
                _occ.add((int(_mp[_si, 0]), int(_mp[_si, 1])))
            pos_rc, mon_idx, state, members = _resolve_monster(
                d, terrain_np, w, h, resolved_rooms, _next_key, state,
                occupied=_occ,
            )
            state = _write_monster(state, pos_rc, mon_idx)
            lg.last_monster_entry_ids.append(mon_idx)
            # Write any m_initgrp group members as additional monsters.
            for _mpos, _midx in members:
                state = _write_monster(state, _mpos, _midx)
                lg.last_monster_entry_ids.append(_midx)
        elif isinstance(d, _TrapDirective):
            pos_rc, trap_type, state = _resolve_trap(
                d, terrain_np, w, h, resolved_rooms, _next_key, state,
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
        elif isinstance(d, _StartingInventoryDirective):
            # Buffer; committed after the walk so all items are assigned
            # contiguous letters via InventoryState.from_items.
            starting_inv.append(d)
        else:
            # Defensive: an unknown directive class signals a programming bug.
            raise RuntimeError(f"unhandled directive type: {type(d).__name__}")

    # Stamp the vendor mkstairs down-stair at its real (seed-dependent) cell,
    # computed from the rn2(W)/rn2(W) offsets consumed in the 4-prefix block
    # above (Monster/Ultimate envs).  Done AFTER the pass-2 FLOOR fill so it
    # isn't overwritten; replaces the per-wrapper hardcoded stair stamp.
    if _mklev_stair_cell is not None:
        _scy, _scx = _mklev_stair_cell
        if 0 <= _scy < h and 0 <= _scx < w:
            terrain_np = terrain_np.at[0, 0, _scy, _scx].set(
                jnp.int8(int(TileType.STAIRCASE_DOWN))
            )

    # 4. Commit accumulated terrain/traps/grounds.
    new_traps = state.traps.replace(trap_type=trap_type_arr)
    state = state.replace(
        terrain=terrain_np,
        traps=new_traps,
        ground_items=ground,
    )

    # 4b. Commit door open/closed/locked status into the features overlay.
    # Branch=0 level=0 flat index is 0 (same convention as traps above).
    if door_states:
        ds_arr = jnp.asarray(state.features.door_state)
        for row, col, dval in door_states:
            ds_arr = ds_arr.at[0, row, col].set(jnp.int8(dval))
        state = state.replace(
            features=state.features.replace(door_state=ds_arr),
        )

    # 4c. Materialise starting inventory.
    #
    # Vendor MiniHack envs run NetHack's full startup, so every hero spawns
    # with the role's ``ini_inv(...)`` items (vendor/nethack/src/u_init.c)
    # FIRST, and only then do des ``INV:`` directives append additional
    # carried items (vendor/nle/src/sp_lev.c::create_object with INV flag).
    #
    # Under NLE_BYTEPARITY, ``state.inventory`` was already populated by the
    # NethaxEnv.reset bootstrap at the top of this function — that runs the
    # full vendor u_init path (ini_inv + the Archeologist rn2(10)/rn2(4)/
    # rn2(5) optional-item cascade at u_init.c:652-660) reading the same
    # ISAAC64 stream offsets vendor C reads.  No further inventory work
    # needed here.  LG ``add_inventory_item`` directives are not used by
    # any canonical Room/Corridor/MazeWalk/etc env builder today (verified
    # by grep over Nethax/minihax/{envs,world_gen}); if a future env wires
    # INV: directives, extend this branch to read existing item slots out
    # of ``state.inventory.items`` and rebuild via InventoryState.from_items.
    #
    # Outside NLE_BYTEPARITY (legacy Threefry path) the state inventory is
    # the EnvState.default zero-init, so we still need to seed the
    # role-specific ini_inv items here.  No rn2 cascade in that mode — the
    # ISAAC64 stream is not modelled, so the optional bonus item is
    # deterministically omitted (matches Threefry behaviour before Lead
    # E/G's commits).
    from Nethax.nethax.subsystems.inventory import (
        InventoryState as _InventoryState,
        make_item as _make_item,
    )
    from Nethax.nethax.parity_mode import use_vendor_rng as _use_vendor_rng_inv
    if _use_vendor_rng_inv():
        assert not starting_inv, (
            "minihax NLE_BYTEPARITY path: LG add_inventory_item directives "
            "(starting_inv) are not wired through the NethaxEnv.reset "
            "bootstrap.  Extend _apply_directives to read existing inventory "
            "slots out of state.inventory.items before appending."
        )
    else:
        from Nethax.nethax.subsystems.character import (
            STARTING_INVENTORY as _STARTING_INVENTORY,
            Role as _Role,
        )
        items = list(_STARTING_INVENTORY[_Role.ARCHEOLOGIST])
        items.extend(
            _make_item(
                category=d.category,
                type_id=d.type_id,
                quantity=d.quantity,
                weight=d.weight,
                buc_status=d.buc_status,
                identified=d.identified,
                bknown=True, dknown=True, rknown=True,
            )
            for d in starting_inv
        )
        state = state.replace(inventory=_InventoryState.from_items(items))

    # 5. Apply player start position (default: any free floor tile).
    # Track whether the position came from an explicit ``set_start_pos`` so
    # step 6 can skip FoV seeding when the actual hero cell will be picked
    # later by a vendor-RNG wrapper (e.g. ``_wrap_random_room_placement`` in
    # canonical.py).  Without this guard, FoV seeds at the auto-found
    # top-left corner of the first room and over-lights its Chebyshev<=1
    # neighbourhood, which becomes wrong as soon as the wrapper rewrites
    # ``player_pos`` to the vendor-accepted random cell.
    explicit_start_pos = lg.last_player_pos is not None
    if explicit_start_pos:
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

    # 6. Seed initial FOV / last_seen_terrain so the starting room renders as
    # lit floor (S_room, cmap=19, glyph=2378) instead of S_stone (glyph=2359).
    # Skipped when the player position is not explicit — the random-room
    # wrappers in ``Nethax/minihax/envs/canonical.py`` call
    # :func:`seed_hero_fov` themselves after pinning ``player_pos`` to the
    # vendor-accepted cell.
    if explicit_start_pos:
        state = seed_hero_fov(state, lg.default_lit)

    return state


def seed_hero_fov(state: EnvState, default_lit: bool) -> EnvState:
    """Seed ``visible`` / ``explored`` / ``last_seen_terrain`` for the
    hero's current cell on level (branch=0, level=0).

    Mirrors ``Nethax/nethax/env.py:796-836`` (vendor ``vision_recalc`` on
    level entry).  Without this seed the engine-side ``fast_reset=True``
    bootstrap leaves ``last_seen_terrain`` at the -1 sentinel and every
    interior cell renders as stone (S_stone, glyph 2359) instead of lit
    floor (S_room, glyph 2378).

    Hero-radius (Chebyshev<=1) torchlight applies to both lit and dark
    rooms (vendor lights the hero's own 3x3 even in dark rooms).  The
    flood-fill ``lit_mask`` path is gated on ``default_lit`` because only
    ``LevelGenerator(lit=True)`` marks every carved tile as rlit=1.

    Per-cell visibility is gated by ``view_from`` so walls correctly block
    line-of-sight; this prevents over-lighting cells through a wall when
    the hero stands adjacent to the room boundary.
    """
    from Nethax.nethax.fov import view_from as _view_from
    terrain_l0 = state.terrain[0, 0]
    couldsee = _view_from(
        terrain_l0,
        state.player_pos.astype(jnp.int32),
        max_radius=0,
    )
    if default_lit:
        lit_mask = terrain_l0 != jnp.int8(int(TileType.VOID))
    else:
        lit_mask = jnp.zeros_like(terrain_l0, dtype=jnp.bool_)
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    _h_g, _w_g = terrain_l0.shape
    rows_g = jnp.arange(_h_g, dtype=jnp.int32)[:, None]
    cols_g = jnp.arange(_w_g, dtype=jnp.int32)[None, :]
    within_light = (
        (jnp.abs(rows_g - pr) <= jnp.int32(1))
        & (jnp.abs(cols_g - pc) <= jnp.int32(1))
    )
    vis = couldsee & (lit_mask | within_light)
    old_lst = state.last_seen_terrain[0, 0]
    new_lst = jnp.where(vis, terrain_l0.astype(jnp.int8), old_lst)
    new_explored = state.explored.at[0, 0].set(
        state.explored[0, 0] | vis
    )
    return state.replace(
        explored=new_explored,
        visible=vis,
        last_seen_terrain=state.last_seen_terrain.at[0, 0].set(new_lst),
    )


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


def _stamp_map_block(
    terrain_np: jax.Array, rows: Tuple[str, ...], w: int, h: int,
) -> jax.Array:
    """Write a literal vendor MAP grid into ``terrain[0, 0]``.

    Source: vendor des ``MAP ... ENDMAP`` blocks.  Each character is mapped
    through ``TERRAIN_CHAR_TO_TILE``; unknown glyphs (object/monster overlay
    symbols that the des places via separate directives) fall back to FLOOR
    so the tile is walkable.  Spaces resolve to ``VOID`` (vendor
    ``INIT_MAP:solidfill,' '`` stone), giving the level a hard boundary
    instead of the LG's default open-FLOOR fill.
    """
    floor = int(TileType.FLOOR)
    for y, line in enumerate(rows):
        if y >= h:
            break
        for x, ch in enumerate(line):
            if x >= w:
                break
            tile = TERRAIN_CHAR_TO_TILE.get(ch)
            if tile is None:
                # Glyph is an object/monster placement char (e.g. '!', '/');
                # the underlying terrain is open floor.
                tile = floor
            else:
                tile = int(tile)
            terrain_np = terrain_np.at[0, 0, y, x].set(jnp.int8(tile))
    return terrain_np


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


def _replace_terrain_rect(
    terrain_np: jax.Array,
    d: _ReplaceTerrainDirective,
    w: int,
    h: int,
    next_key,
) -> jax.Array:
    """Probabilistic per-cell tile swap (vendor REPLACE_TERRAIN).

    Bernoulli draws derived from the directive-walk PRNG; cells holding
    ``from_terrain`` flip to ``to_terrain`` when draw < chance.  Other
    cells (e.g. corridor-carved floor, stairs) are left untouched, matching
    vendor behaviour where REPLACE_TERRAIN runs before TERRAIN:randline.
    """
    from_tile = int(TERRAIN_CHAR_TO_TILE[d.from_terrain])
    to_tile = int(TERRAIN_CHAR_TO_TILE[d.to_terrain])
    y_lo, y_hi = sorted((d.y1, d.y2))
    x_lo, x_hi = sorted((d.x1, d.x2))
    y_lo = max(0, y_lo); y_hi = min(h - 1, y_hi)
    x_lo = max(0, x_lo); x_hi = min(w - 1, x_hi)
    if y_lo > y_hi or x_lo > x_hi or d.chance <= 0:
        return terrain_np
    rh = y_hi - y_lo + 1
    rw = x_hi - x_lo + 1
    key = next_key()
    draws = jax.random.uniform(key, (rh, rw), minval=0.0, maxval=100.0)
    flip = draws < float(d.chance)
    region = terrain_np[0, 0, y_lo:y_hi + 1, x_lo:x_hi + 1]
    eligible = region == jnp.int8(from_tile)
    new_region = jnp.where(eligible & flip, jnp.int8(to_tile), region)
    return terrain_np.at[0, 0, y_lo:y_hi + 1, x_lo:x_hi + 1].set(new_region)


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

def _enexto(terrain_np, occupied: set, xx: int, yy: int, w: int, h: int,
            vrng):
    """Replicate vendor ``enexto_core`` (teleport.c:126-218).

    Walk the borders of expanding squares centred on ``(xx, yy)`` (xx=col,
    yy=row), collecting ``goodpos`` cells; stop at the first range that has
    any, then pick one via ``rn2(num_good)``.  ``goodpos`` here = in-bounds
    FLOOR tile not in ``occupied`` (set of ``(row, col)``).  Returns
    ``((row, col)|None, vrng)``.  The ``rn2(num_good)`` draw is the same one
    vendor's m_initgrp consumes, so consuming it here keeps vrng aligned.
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    floor = int(TileType.FLOOR)
    sub = terrain_np[0, 0]
    MAX_GOOD = 15

    def goodpos(x, y):
        if not (0 <= y < h and 0 <= x < w):
            return False
        if (y, x) in occupied:
            return False
        return int(sub[y, x]) == floor

    xmax0 = max(xx - 1, (w - 1) - xx)
    ymax0 = max(yy - 0, (h - 1) - yy)
    rangemax = max(xmax0, ymax0)
    good = []
    full = False
    rng = 1
    while True:
        xmin = max(1, xx - rng)
        xmax = min(w - 1, xx + rng)
        ymin = max(0, yy - rng)
        ymax = min(h - 1, yy + rng)
        for x in range(xmin, xmax + 1):
            if goodpos(x, ymin):
                good.append((ymin, x))
                if len(good) == MAX_GOOD:
                    full = True
                    break
            if goodpos(x, ymax):
                good.append((ymax, x))
                if len(good) == MAX_GOOD:
                    full = True
                    break
        if not full:
            for y in range(ymin, ymax):
                if goodpos(xmin, y):
                    good.append((y, xmin))
                    if len(good) == MAX_GOOD:
                        full = True
                        break
                if goodpos(xmax, y):
                    good.append((y, xmax))
                    if len(good) == MAX_GOOD:
                        full = True
                        break
        rng += 1
        if full or good or rng > rangemax:
            break
    if not good:
        return None, vrng
    vrng, i = _vendor_rng.rn2_jax(vrng, jnp.int32(len(good)))
    return good[int(i)], vrng


def _resolve_monster(
    d: _MonsterDirective,
    terrain_np: jax.Array,
    w: int,
    h: int,
    resolved_rooms: dict,
    next_key,
    state: Optional[EnvState] = None,
    occupied: Optional[set] = None,
) -> Tuple[Tuple[int, int], int, Optional[EnvState], list]:
    """Resolve a monster directive to ``((row, col), monster_idx, new_state, members)``.

    ``members`` is a list of ``((row, col), member_idx)`` for any m_initgrp
    group members spawned by a group-flagged monster (empty otherwise).
    ``occupied`` is a set of ``(row, col)`` cells already taken by earlier
    monsters / the stair, used by ``enexto`` for member placement.

    Under ``use_vendor_rng()``, draws come from ``state.vendor_rng`` so that
    monster placement consumes the same ISAAC64 stream offsets vendor
    ``mkmonster`` consumes (vendor/nethack/src/makemon.c + mklev.c somxy
    loop).  Per the seed-0 5x5 trace diff vs the Trap variant
    (.test_runs/full_init_rn2_trace_room_monster_5x5_seed0.txt offsets
    345-369), the monster directive consumes:

    * 5× small-modulus draws for monster-type / makemon internal picks:
      ``rn2(5)``, ``rn2(2)``, ``rn2(50)``, ``rn2(100)``, ``rn2(100)``.
    * 10× ``(rn2(79), rn2(21))`` somxy() coordinate pairs (retry loop;
      we accept the first FLOOR cell, otherwise keep the last drawn pair —
      matches the bounded-retry pattern used by ``_resolve_trap``).

    The new ``state`` (with advanced ``vendor_rng``) is returned; callers
    MUST adopt it.
    """
    if d.name == "random":
        # Wave 5+ TODO: depth-aware random pick.  For Wave 4 we substitute a
        # deterministic fallback so the directive always produces a monster.
        # Vendor MiniHack-Room-Monster-5x5 seed=0 spawns a "newt" (glyph 318)
        # via Python random; use that as the byte-parity placeholder so the
        # glyph table matches vendor at the placement cell.
        idx = _MONSTER_NAME_TO_IDX.get("newt", 0)
    else:
        # vendor .des files capitalize monster names (e.g. "Minotaur") but
        # Nethax MONSTERS uses lowercase ("minotaur").  Try lowercase first
        # before raising KeyError.  Cite: vendor/minihack/minihack/dat/
        # quest_hard.des line 63 "MONSTER:('a',\"Minotaur\")".
        lookup = d.name if d.name in _MONSTER_NAME_TO_IDX else d.name.lower()
        if lookup not in _MONSTER_NAME_TO_IDX:
            raise KeyError(
                f"unknown monster name {d.name!r}; not present in MONSTERS table"
            )
        idx = _MONSTER_NAME_TO_IDX[lookup]

    from Nethax.nethax.parity_mode import use_vendor_rng as _use_vendor_rng
    if state is not None and _use_vendor_rng():
        from Nethax.nethax import vendor_rng as _vendor_rng
        vrng = state.vendor_rng
        # Resolve room geometry (vendor: croom->lx/hx/ly/hy).
        if resolved_rooms:
            ry1, rx1, ry2, rx2 = next(iter(resolved_rooms.values()))
        else:
            rx1, ry1, rx2, ry2 = 0, 0, w - 1, h - 1
        room_w = max(1, rx2 - rx1 + 1)
        room_h = max(1, ry2 - ry1 + 1)
        # Vendor per-monster 7-draw template (sp_lev.c:create_monster ->
        # get_location_coord -> mkroom.c:somexy + makemon.c:makemon ->
        # m_initweap).  Captured in
        # .test_runs/full_init_rn2_trace_room_ultimate_15x15_seed0.txt:343-349
        # and ..._room_monster_5x5_seed0.txt:343-349:
        #   rn2(3)        — mkclass mlet pick (3-class slice)
        #   rn2(room_w)   — somex(croom) (x offset in room)
        #   rn2(room_h)   — somey(croom) (y offset in room)
        #   rn2(2)        — somexy post-check / mk_roamer align
        #   rn2(50)       — m_initweap defensive item check (m_lev > rn2(50))
        #   rn2(100)      — m_initweap misc item check
        #   rn2(100)      — m_initweap follow-up (rnd_misc_item internal)
        # The monster lands at (rx1 + x_off, ry1 + y_off).  Variable-length
        # extras for grouping monsters (m_initgrp / m_initweap class
        # branches in makemon.c:163-800) are followup.
        # Per-monster template — GROUND-TRUTHED against the COMPLETE CORE
        # draw stream (NETHAX_RND, captures untraced rnd()/d() too).  See
        # .test_runs/full_rnd_stream_*_Monster_{5x5,15x15}_*_seed0.txt.
        # Monster-5x5 M1 (offsets 343-351) and Mon-15x15 M1 (343-351):
        #   rn2(3)   — mkclass mlet pick
        #   rn2(W)   — somex room x offset
        #   rn2(W)   — somey room y offset
        #   rnd(21)  — UNTRACED (RND#346; makemon mon setup)
        #   rnd(4)   — UNTRACED (RND#347)
        #   rn2(2)   — mk_roamer align / peace
        #   rn2(50)  — m_initweap defensive-item check
        #   rn2(100) — m_initweap misc-item check
        #   rn2(100) — m_initweap follow-up
        # Note: rn2_jax consumes exactly one ISAAC64 u64 per call regardless
        # of modulus (vendor RND = isaac64_next_uint64 % x, no rejection),
        # so the untraced fillers' moduli only matter for faithfulness.
        vrng, mkclass_val = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, mx_off = _vendor_rng.rn2_jax(vrng, jnp.int32(room_w))
        vrng, my_off = _vendor_rng.rn2_jax(vrng, jnp.int32(room_h))
        # The untraced rnd(21) IS vendor's rndmonst monster pick:
        # MONSTER:random -> create_monster(class=0) -> makemon(NULL) ->
        # rndmonst() draws rnd(choice_count) (choice_count==21 at depth 1,
        # matching the stream's rnd(21)) and walks the freq-weighted table.
        # Compute the real monster identity from it instead of a fixed newt.
        from Nethax.nethax.dungeon.spawning import pick_monster_for_level
        vrng, _picked_idx = pick_monster_for_level(None, 1, vendor_rng=vrng)
        if d.name == "random":
            idx = int(_picked_idx)
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(4))   # untraced rnd(4)
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))
        # Leader cell (vendor mkroom.c somexy: room-relative offsets).
        xi = rx1 + int(mx_off)
        yi = ry1 + int(my_off)
        floor = int(TileType.FLOOR)
        sub = terrain_np[0, 0, :h, :w]
        if 0 <= yi < h and 0 <= xi < w and int(sub[yi, xi]) == floor:
            rc = (yi, xi)
        else:
            rc = ((ry1 + ry2) // 2, (rx1 + rx2) // 2)
        # m_initgrp group spawn (vendor makemon.c:1369-1378).  A freshly-made
        # G_SGROUP / G_LGROUP monster spawns a same-type group:
        #     if ((geno & G_SGROUP) && rn2(2))       m_initgrp(n=3)
        #     else if (geno & G_LGROUP)              rn2(3) ? lgrp(10) : sgrp(3)
        # m_initgrp (makemon.c:79-144): cnt = rnd(n); cnt /= (ulevel<3)?4 ...;
        # if (!cnt) cnt++; then for each member: enexto() + makemon(member).
        # u.ulevel==1 at mklev so the divisor is 4.  Each member draws
        # enexto's rn2(num_good) + the member makemon draws; afterwards the
        # leader's own m_initweap runs.  All branch on the picked monster's
        # group flag (computed from MONSTERS.generation_mask) — no hardcodes.
        # Ground truth: full_rnd_stream_*_Monster_15x15_*_seed0.txt M2 (gridbug).
        _ULEVEL_DIV = 4  # u.ulevel == 1 at mklev: (ulevel<3)?4:(ulevel<5)?2:1
        members: list = []
        is_sgroup = bool(_MON_SGROUP[idx]) if 0 <= idx < _MON_SGROUP.shape[0] else False
        is_lgroup = bool(_MON_LGROUP[idx]) if 0 <= idx < _MON_LGROUP.shape[0] else False
        grp_n = 0
        if is_sgroup:
            vrng, _gate = _vendor_rng.rn2_jax(vrng, jnp.int32(2))
            if int(_gate) != 0:
                grp_n = 3
        elif is_lgroup:
            vrng, _lg = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
            grp_n = 10 if int(_lg) != 0 else 3
        if grp_n > 0:
            # cnt = rnd(grp_n) (== rn2(grp_n)+1), divided by the ulevel
            # factor, floored, min 1.
            vrng, _cnt_raw = _vendor_rng.rn2_jax(vrng, jnp.int32(grp_n))
            cnt = (int(_cnt_raw) + 1) // _ULEVEL_DIV
            if cnt < 1:
                cnt = 1
            occ = set(occupied) if occupied else set()
            occ.add(rc)  # leader occupies its own cell
            for _m in range(cnt):
                mpos, vrng = _enexto(terrain_np, occ, xi, yi, w, h, vrng)
                # member makemon draws (newmonhp/init + m_initweap):
                # rn2(4), rn2(2), rn2(50), rn2(100), rn2(100).
                vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(4))
                vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))
                vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(50))
                vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
                vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
                if mpos is not None:
                    members.append((mpos, idx))
                    occ.add(mpos)
            # Leader's own m_initweap (after the group is built).
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(50))
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
        else:
            # Non-group (or gate==0): just the leader's m_initweap.
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(50))
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
        new_state = state.replace(vendor_rng=vrng)
        return rc, idx, new_state, members

    # Default (Threefry) mode for ``MONSTER:random``: pick a level-appropriate
    # random monster identity (vendor rndmonst) instead of the fixed ``newt``
    # placeholder, so spawns vary by type+strength like real MiniHack.  Without
    # this, every Room-Monster/-Ultimate monster is the same tanky entry, which
    # a real-MiniHack-trained policy can't clear (it expects mostly weak depth-1
    # monsters) — Room-Monster-15x15 transfer collapses to ~30%.  ``vendor_rng``
    # is seeded per-episode in ``_apply_directives`` (default branch), so the
    # pick varies by seed.  Byte-parity mode is handled by the
    # ``use_vendor_rng()`` block above and never reaches here.
    if d.name == "random" and state is not None:
        from Nethax.nethax import vendor_rng as _vendor_rng_dm
        from Nethax.nethax.dungeon.spawning import pick_monster_for_level
        _vrng_dm = state.vendor_rng
        _vrng_dm, _picked_dm = pick_monster_for_level(None, 1, vendor_rng=_vrng_dm)
        idx = int(_picked_dm)
        state = state.replace(vendor_rng=_vrng_dm)

    rc = _resolve_place(d.place, terrain_np, w, h, resolved_rooms, next_key)
    if rc is None:
        rc = (0, 0)

    # m_initgrp group spawn in default (Threefry) mode too (vendor makemon.c
    # :1369-1378).  G_SGROUP / G_LGROUP species (jackal, sewer rat, gnome, …)
    # spawn a same-type pack around the leader; real MiniHack Room-Monster/
    # -Ultimate therefore has MORE, weaker monsters than a lone spawn.  Without
    # this the default path emitted singletons (e.g. 3 monsters where real has
    # 4), a per-spawn count/placement mismatch vs real.  Members are placed via
    # vendor ``enexto`` around the leader using the per-episode-seeded
    # ``vendor_rng``.  We skip the m_initweap rn2 alignment draws (only needed
    # for byte parity, which is handled by the use_vendor_rng() block above and
    # never reaches here).
    members: list = []
    if d.name == "random":
        is_sgroup = bool(_MON_SGROUP[idx]) if 0 <= idx < _MON_SGROUP.shape[0] else False
        is_lgroup = bool(_MON_LGROUP[idx]) if 0 <= idx < _MON_LGROUP.shape[0] else False
        if is_sgroup or is_lgroup:
            from Nethax.nethax import vendor_rng as _vrng_g
            vrng = state.vendor_rng
            grp_n = 0
            if is_sgroup:
                vrng, _gate = _vrng_g.rn2_jax(vrng, jnp.int32(2))
                if int(_gate) != 0:
                    grp_n = 3
            else:
                vrng, _lg = _vrng_g.rn2_jax(vrng, jnp.int32(3))
                grp_n = 10 if int(_lg) != 0 else 3
            if grp_n > 0:
                vrng, _cnt_raw = _vrng_g.rn2_jax(vrng, jnp.int32(grp_n))
                cnt = max(1, (int(_cnt_raw) + 1) // 4)  # u.ulevel==1 divisor
                occ = set(occupied) if occupied else set()
                occ.add(rc)
                xi, yi = rc[1], rc[0]
                for _m in range(cnt):
                    mpos, vrng = _enexto(terrain_np, occ, xi, yi, w, h, vrng)
                    if mpos is not None:
                        members.append((mpos, idx))
                        occ.add(mpos)
            state = state.replace(vendor_rng=vrng)
    return rc, idx, state, members


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
    state: Optional[EnvState] = None,
) -> Tuple[Tuple[int, int], int, Optional[EnvState]]:
    """Resolve a trap directive to ``((row, col), trap_kind, new_state)``.

    Under ``use_vendor_rng()``, draws come from ``state.vendor_rng`` so that
    trap placement consumes the same ISAAC64 stream offsets vendor
    ``mktrap`` consumes (vendor/nethack/src/mklev.c:1318-1366 trap-kind
    picker + somxy loop): 2× ``rn2(5)`` for type/internal selection, then
    up to 5× ``(rn2(79), rn2(21))`` coordinate pairs (somxy retry loop),
    accepting the first FLOOR cell.  The new ``state`` (with advanced
    ``vendor_rng``) is returned; callers MUST adopt it.
    """
    if d.name == "random":
        trap_kind = int(TrapType.TELEP_TRAP)
    else:
        trap_kind = int(TRAP_NAME_TO_TYPE[d.name])

    from Nethax.nethax.parity_mode import use_vendor_rng as _use_vendor_rng
    if state is not None and _use_vendor_rng():
        # Vendor-rng draws for the trap (2× rn2(5) kind + 5× somxy pairs)
        # are now consumed in ``_wrap_trap_room_placement`` AFTER the stair
        # stamp, matching vendor mklev order (mkstairs precedes mktrap).
        # We pick a deterministic placeholder position here without
        # touching the vendor stream; full trap-glyph parity is a follow-up.
        floor = int(TileType.FLOOR)
        sub = terrain_np[0, 0, :h, :w]
        rc: Optional[Tuple[int, int]] = None
        for yi in range(h):
            for xi in range(w):
                if int(sub[yi, xi]) == floor:
                    rc = (yi, xi)
                    break
            if rc is not None:
                break
        if rc is None:
            rc = (0, 0)
        return rc, trap_kind, state

    rc = _resolve_place(d.place, terrain_np, w, h, resolved_rooms, next_key)
    if rc is None:
        rc = (0, 0)
    return rc, trap_kind, state


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

    # HP from the monster's level (vendor makemon.c::newmonhp), NOT a flat 8.
    # A flat 8 made every monster equally tanky regardless of species, so a
    # real-MiniHack-trained policy that expects mostly weak depth-1 monsters
    # (newt/sewer-rat = 1..4 HP) couldn't clear Room-Monster levels and got
    # drained to death — Room-Monster-15x15 transfer ~30%.  Roll per-spawn with
    # a key folded from the episode rng + slot so it's reproducible and varies
    # by monster.  Monster HP is not part of the NLE obs (glyphs/blstats), so
    # byte parity is unaffected.
    from Nethax.nethax.subsystems.monster_ai import _newmonhp_roll as _nmhp
    _m_lev = int(MONSTERS[mon_idx_clipped].level)
    _hp_key = jax.random.fold_in(state.rng, jnp.int32(slot * 1009 + mon_idx_clipped + 1))
    _hp_val = int(_nmhp(_hp_key, jnp.int32(_m_lev)))

    new_mai = mai.replace(
        pos=mai.pos.at[slot].set(jnp.array([row, col], dtype=jnp.int16)),
        # entry_idx selects the monster glyph (GLYPH_MON_OFF + entry_idx ==
        # nle glyph; constants/glyphs.py: GLYPH_MON_OFF=0).  Previously left
        # default (0 = uninitialized), which rendered as NUL on the map and
        # produced glyph-table divergence for Monster-* room variants.
        entry_idx=mai.entry_idx.at[slot].set(jnp.int16(mon_idx_clipped)),
        hp=mai.hp.at[slot].set(jnp.int32(_hp_val)),
        hp_max=mai.hp_max.at[slot].set(jnp.int32(_hp_val)),
        m_lev=mai.m_lev.at[slot].set(jnp.int16(_m_lev)),
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
    # Materialise the sub-region as Python lists for the iterative carve.
    # This entire function runs at Python init time (level-build, not the
    # per-step JIT trace), so we use plain Python containers and pull
    # randomness from the JAX PRNG stream (via ``next_key``) instead of
    # round-tripping through numpy's RandomState.
    wall = int(TileType.WALL)
    corridor = int(TileType.CORRIDOR)
    # Fill with WALL first.
    sub = [[wall for _ in range(w)] for _ in range(h)]

    sx = max(0, min(int(start_x), w - 1))
    sy = max(0, min(int(start_y), h - 1))
    # Align to odd coords so the 2-stride walk stays in-bounds.
    if sx % 2 == 0:
        sx = min(w - 1, sx + 1)
    if sy % 2 == 0:
        sy = min(h - 1, sy + 1)

    visited = [[False for _ in range(w)] for _ in range(h)]
    stack = [(sy, sx)]
    visited[sy][sx] = True
    sub[sy][sx] = corridor

    while stack:
        r, c = stack[-1]
        neighbours = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc]:
                neighbours.append((nr, nc, dr, dc))
        if not neighbours:
            stack.pop()
            continue
        # Pure JAX-stream randomness: pull a fresh subkey from the
        # build-time PRNG factory each step.  Equivalent in distribution
        # to ``random.randrange(len(neighbours))`` but stays entirely on
        # the JAX side — no numpy RandomState round-trip.
        idx = int(jax.random.randint(next_key(), (), 0, len(neighbours)))
        nr, nc, dr, dc = neighbours[idx]
        # Carve bridge cell + neighbour.
        br, bc = r + dr // 2, c + dc // 2
        sub[br][bc] = corridor
        sub[nr][nc] = corridor
        visited[nr][nc] = True
        stack.append((nr, nc))

    # Write the carved sub-region back.
    sub_arr = jnp.asarray(sub, dtype=terrain_np.dtype)
    new_terrain = terrain_np.at[0, 0, :h, :w].set(sub_arr)
    return new_terrain
