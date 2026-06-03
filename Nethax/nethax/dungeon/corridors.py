"""Corridor generation and door placement.

Purpose:
    Connects rooms with L-shaped corridors and places doors at
    room/corridor junctions.  Mirrors the corridor logic in mklev.c and the
    door-placement logic in mklev.c / mkmap.c.

Citation:
    vendor/nethack/src/mklev.c  — doconnect(), makedog(), add_door(),
        dig_corridor(); L-shaped corridor with random bend point.
    vendor/nethack/src/mkmap.c  — coordinate allocation, door placement.

Wave 2: connect_segments carves an L-shaped corridor between two rooms
        onto a terrain array; place_doors stamps CLOSED_DOOR tiles at
        the room-wall/corridor boundary with ~50% probability.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import struct

from Nethax.nethax.dungeon.rooms import Room

# ---------------------------------------------------------------------------
# Tile type constants (from constants.py TileType)
# ---------------------------------------------------------------------------

_TILE_VOID:        int = 0
_TILE_FLOOR:       int = 1
_TILE_CORRIDOR:    int = 2
_TILE_WALL:        int = 3
_TILE_CLOSED_DOOR: int = 4

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A corridor segment is represented as int16[N, 4] where each row is
# (y1, x1, y2, x2) — the two endpoints of a straight horizontal or vertical
# run.  L-shaped corridors decompose into two such segments sharing a bend.
CorridorSegments = jnp.ndarray  # int16[N, 4]

# A door entry is int16[M, 3]: (y, x, door_state) where door_state is
# 0=open, 1=closed, 2=locked, 3=secret (matching NLE glyph categories).
DoorArray = jnp.ndarray  # int16[M, 3]

# Fixed capacity for door arrays per level (2 per corridor = 2*(MAX_ROOMS-1))
_MAX_DOORS: int = 80


# ---------------------------------------------------------------------------
# wipeout_text rubouts table (vendor/nle/src/engrave.c:27-78).
#
# Used by ``_wipe_one`` inside the SCORR niche-trap engraving step to model
# vendor wipeout_text's conditional inner ``rn2(strlen(wipeto))`` draw.
# Indexed by ASCII code 0..127.  Non-wipefrom chars have len=0; wipefrom
# chars store ``len`` plus the wipeto chars packed into a fixed-5-slot row.
# ---------------------------------------------------------------------------
def _build_rubouts_tables():
    import numpy as _np
    rubouts = [
        ('A', '^'), ('B', 'Pb['), ('C', '('), ('D', '|)['),
        ('E', '|FL[_'), ('F', '|-'), ('G', 'C('), ('H', '|-'),
        ('I', '|'), ('K', '|<'), ('L', '|_'), ('M', '|'),
        ('N', '|\\'), ('O', 'C('), ('P', 'F'), ('Q', 'C('),
        ('R', 'PF'), ('T', '|'), ('U', 'J'), ('V', '/\\'),
        ('W', 'V/\\'), ('Z', '/'), ('b', '|'), ('d', 'c|'),
        ('e', 'c'), ('g', 'c'), ('h', 'n'), ('j', 'i'),
        ('k', '|'), ('l', '|'), ('m', 'nr'), ('n', 'r'),
        ('o', 'c'), ('q', 'c'), ('w', 'v'), ('y', 'v'),
        (':', '.'), (';', ',:'), (',', '.'), ('=', '-'),
        ('+', '-|'), ('*', '+'), ('@', '0'), ('0', 'C('),
        ('1', '|'), ('6', 'o'), ('7', '/'), ('8', '3o'),
    ]
    lens = _np.zeros(128, dtype=_np.int32)
    chars = _np.zeros((128, 5), dtype=_np.int32)
    for src, dst in rubouts:
        i = ord(src)
        lens[i] = len(dst)
        for k, c in enumerate(dst):
            chars[i, k] = ord(c)
    return jnp.asarray(lens), jnp.asarray(chars)


_RUBOUT_LEN, _RUBOUT_CHARS = _build_rubouts_tables()

# Trivial wipe-to-space set: ``"?.,'`-|_"`` (engrave.c:110).
_TRIVIAL_WIPE_CHARS = jnp.asarray(
    [ord(c) for c in "?.,'`-|_"], dtype=jnp.int32
)

# Engraving template chars padded to length 13 (the longer of the two trap
# engravings).  Index 0..12.  For lth=11 ("ad aerarium") only [0..10] used.
def _engr_template(lth):
    """Return int32[13] of engraving char codes, padded with 0 beyond ``lth``.

    Vendor trap_engravings (engrave.c references trap.c):
        TRAPDOOR (14):    "Vlad was here"   lth=13
        TELEP_TRAP (15):  "ad aerarium"     lth=11
        LEVEL_TELEP(16):  "ad aerarium"     lth=11
    """
    import numpy as _np
    vlad = _np.array([ord(c) for c in "Vlad was here"], dtype=_np.int32)
    aer  = _np.array([ord(c) for c in "ad aerarium\0\0"], dtype=_np.int32)
    out_vlad = jnp.asarray(vlad)
    out_aer  = jnp.asarray(aer)
    # lth==13 -> vlad; lth==11 -> aer.
    return jnp.where(lth == jnp.int32(13), out_vlad, out_aer)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def connect_segments(
    rng: jnp.ndarray,
    room_a: Room,
    room_b: Room,
    terrain: jnp.ndarray,
    index: int = 0,
) -> jnp.ndarray:
    """Carve a corridor from room_a to room_b, orthogonal-first then L-bend.

    Vendor (`sp_lev.c::dig_corridor` lines 2571-2660): the digger walks
    toward the target one step at a time, only turning when forced to.
    When both endpoints share a row or column, a single straight run
    suffices; otherwise an L-bend is required.  We follow the same
    priority: straight when possible, L-shape when not.

    Corridor width is exactly one tile (vendor: a single `levl[xx][yy].typ
    = ftyp` per step).  Diagonals are NOT supported — corridor cells are
    placed only on cardinal-aligned segments.

    Args:
        rng:     JAX PRNG key (unused here; vendor bend is deterministic).
        room_a:  source Room (scalar Room — a slice from the rooms array).
        room_b:  destination Room.
        terrain: int8[MAP_H, MAP_W] to carve into.
        index:   slot index (unused; kept for call-site consistency).

    Returns:
        Updated terrain int8[MAP_H, MAP_W].
    """
    h, w = terrain.shape
    rows = jnp.arange(h, dtype=jnp.int32)
    cols = jnp.arange(w, dtype=jnp.int32)

    y_a = ((room_a.y1 + room_a.y2) // 2).astype(jnp.int32)
    x_a = ((room_a.x1 + room_a.x2) // 2).astype(jnp.int32)
    y_b = ((room_b.y1 + room_b.y2) // 2).astype(jnp.int32)
    x_b = ((room_b.x1 + room_b.x2) // 2).astype(jnp.int32)

    same_row = y_a == y_b
    same_col = x_a == x_b

    # Horizontal leg: row y_a, columns min(x_a,x_b)..max(x_a,x_b)
    horiz_row = rows == y_a
    horiz_col = (cols >= jnp.minimum(x_a, x_b)) & (cols <= jnp.maximum(x_a, x_b))
    horiz = horiz_row[:, None] & horiz_col[None, :]

    # Vertical leg: col x_b, rows min(y_a,y_b)..max(y_a,y_b)
    vert_col = cols == x_b
    vert_row = (rows >= jnp.minimum(y_a, y_b)) & (rows <= jnp.maximum(y_a, y_b))
    vert = vert_row[:, None] & vert_col[None, :]

    # Straight: use whichever single leg lines the rooms up.
    straight = jnp.where(same_row, horiz, vert)
    # L-bend fallback.
    l_shape = horiz | vert
    corridor = jnp.where(same_row | same_col, straight, l_shape)

    is_floor = terrain == jnp.int8(_TILE_FLOOR)
    terrain_out = jnp.where(corridor & ~is_floor, jnp.int8(_TILE_CORRIDOR), terrain)
    return terrain_out


def _finddpos_door_mask(terrain: jnp.ndarray) -> jnp.ndarray:
    """Return a bool[H, W] mask of preferred door positions.

    Vendor citation: vendor/nethack/src/mklev.c::finddpos lines 147-199 and
    its sibling ``good_rm_wall_doorpos`` (lines 80-102).  Vendor walks the
    room wall and only accepts a door at a wall cell whose room-interior
    neighbour is reachable and not adjacent to an existing door.  The key
    structural rule: doors land on WALL cells that border a corridor (or
    are themselves carved through by a corridor), not on interior FLOOR.

    Our JIT-pure equivalent: a cell qualifies when it is currently a WALL
    *and* at least one of its four orthogonal neighbours is CORRIDOR.  This
    captures vendor's "wall position with corridor leading into it"
    preference without needing per-room iteration — the mask is computed
    once for the whole map.

    Args:
        terrain: int8[H, W] with rooms + corridors already carved.

    Returns:
        bool[H, W] — True on wall cells adjacent to a corridor tile.
    """
    is_wall = terrain == jnp.int8(_TILE_WALL)
    is_corr = terrain == jnp.int8(_TILE_CORRIDOR)

    # Pad-and-slice orthogonal-neighbour check (cheaper than .at[] gathers,
    # and stays inside XLA's static-shape regime).
    padded = jnp.pad(is_corr, ((1, 1), (1, 1)), constant_values=False)
    corr_n = padded[:-2, 1:-1]   # north neighbour
    corr_s = padded[2:,  1:-1]   # south
    corr_w = padded[1:-1, :-2]   # west
    corr_e = padded[1:-1, 2:]    # east
    adj_corr = corr_n | corr_s | corr_w | corr_e

    return is_wall & adj_corr


def place_doors(
    rng: jnp.ndarray,
    terrain: jnp.ndarray,
    rooms: Room,
    active: jnp.ndarray,
) -> jnp.ndarray:
    """Place CLOSED_DOOR tiles at room/corridor boundaries.

    For each active room, scan the perimeter (the ring one cell outside the
    interior bounding box).  A perimeter cell becomes a CLOSED_DOOR when
    either:
      (a) the finddpos-preferred mask is set (WALL cell adjacent to a
          CORRIDOR — vendor-style structural choice), OR
      (b) the cell is currently CORRIDOR AND a 50/50 coin comes up — the
          legacy stamp that turns a carved-through corridor opening into a
          door.

    Branch (a) is the vendor preference (see :func:`_finddpos_door_mask`);
    branch (b) is retained so previously-tested behaviour is preserved when
    no wall-adjacent-to-corridor candidates exist for a given room.

    Citation: vendor/nethack/src/mklev.c::finddpos lines 147-199 and
        ``add_door()`` / ``doconnect()`` (preferred door positioning).

    Args:
        rng:     JAX PRNG key.
        terrain: int8[MAP_H, MAP_W] with rooms + corridors already carved.
        rooms:   Room pytree from generate_rooms().
        active:  bool[MAX_ROOMS_PER_LEVEL] mask.

    Returns:
        Updated terrain int8[MAP_H, MAP_W] with doors stamped.
    """
    from Nethax.nethax.dungeon.rooms import MAX_ROOMS_PER_LEVEL
    h, w = terrain.shape

    # Pre-compute the finddpos-preferred mask once for the whole map.
    preferred = _finddpos_door_mask(terrain)

    # One door-flip coin per cell — used only on the corridor-fallback branch.
    rng, key_door = jax.random.split(rng)
    door_coin = jax.random.bernoulli(key_door, 0.5, shape=(h, w))

    rows = jnp.arange(h, dtype=jnp.int32)
    cols = jnp.arange(w, dtype=jnp.int32)

    def stamp_doors_for_room(terrain_, i):
        y1 = rooms.y1[i].astype(jnp.int32)
        x1 = rooms.x1[i].astype(jnp.int32)
        y2 = rooms.y2[i].astype(jnp.int32)
        x2 = rooms.x2[i].astype(jnp.int32)
        act = active[i]

        # Perimeter = wall ring just outside the interior.
        row_border = (rows >= y1 - 1) & (rows <= y2 + 1)
        col_border = (cols >= x1 - 1) & (cols <= x2 + 1)
        row_inner  = (rows >= y1)     & (rows <= y2)
        col_inner  = (cols >= x1)     & (cols <= x2)
        perimeter  = (row_border[:, None] & col_border[None, :]) & \
                     ~(row_inner[:, None] & col_inner[None, :])

        is_corridor = terrain_ == jnp.int8(_TILE_CORRIDOR)
        # Branch (a): finddpos-preferred WALL cells adjacent to CORRIDOR
        # — stamp unconditionally (vendor structural preference).
        place_pref = act & perimeter & preferred
        # Branch (b): legacy CORRIDOR-cell stamp gated by 50/50 coin.
        place_fallback = act & perimeter & is_corridor & door_coin

        place = place_pref | place_fallback
        terrain_new = jnp.where(place, jnp.int8(_TILE_CLOSED_DOOR), terrain_)
        return terrain_new, None

    terrain_out, _ = lax.scan(
        stamp_doors_for_room,
        terrain,
        jnp.arange(MAX_ROOMS_PER_LEVEL, dtype=jnp.int32),
    )
    return terrain_out


# ===========================================================================
# Phase 4 of MKLEV_PORT_PLAN.md — vendor-exact makecorridors + make_niches.
#
# All functions in this section mirror vendor/nle/src/mklev.c byte-for-byte
# on the ISAAC64 stream.  They are independent of the Threefry-based
# connect_segments/place_doors above and operate on the dedicated
# :class:`LevelGenState` pytree defined below.
#
# Vendor citations:
#   vendor/nle/src/mklev.c:69-96    finddpos
#   vendor/nle/src/mklev.c:243-316  join
#   vendor/nle/src/mklev.c:319-348  makecorridors
#   vendor/nle/src/mklev.c:548-565  make_niches
#   vendor/nle/src/mklev.c:483-546  makeniche
#   vendor/nle/src/mklev.c:450-471  place_niche
#   vendor/nle/src/mklev.c:1240-1247 okdoor
#   vendor/nle/src/mklev.c:1249-1260 dodoor (delegates to dosdoor)
#   vendor/nle/src/mklev.c:383-447  dosdoor
#   vendor/nle/src/sp_lev.c:2215-2322 dig_corridor
# ===========================================================================

from Nethax.nethax.vendor_rng import (
    Isaac64State,
    rn2_jax,
    rnd_jax,
    rn1_jax,
)
from Nethax.nethax.constants.objects import ObjectClass
from Nethax.nethax.subsystems.random_objects import (
    consume_mksobj_init_draws,
    consume_mkobj_random_draws,
)

# ---------------------------------------------------------------------------
# Vendor tile-type constants (vendor/nle/include/rm.h).
# Only the subset the corridor/door/niche code touches.
# ---------------------------------------------------------------------------

VTILE_STONE:   int = 0    # STONE
VTILE_VWALL:   int = 1    # VWALL
VTILE_HWALL:   int = 2    # HWALL
VTILE_TLCORN:  int = 3    # TLCORNER (vendor rm.h: TLCORNER=3). Stamped by
                          # do_room_or_subroom (mklev.c:176) at (lx-1, ly-1).
VTILE_TRCORN:  int = 4    # TRCORNER (vendor rm.h: TRCORNER=4). Stamped by
                          # do_room_or_subroom (mklev.c:177) at (hx+1, ly-1).
VTILE_BLCORN:  int = 5    # BLCORNER (vendor rm.h: BLCORNER=5). Stamped by
                          # do_room_or_subroom (mklev.c:178) at (lx-1, hy+1).
VTILE_BRCORN:  int = 6    # BRCORNER (vendor rm.h: BRCORNER=6). Stamped by
                          # do_room_or_subroom (mklev.c:179) at (hx+1, hy+1).
VTILE_ROOM:    int = 11   # ROOM
VTILE_CORR:    int = 13   # CORR
VTILE_SCORR:   int = 14   # SCORR (secret corridor)
VTILE_DOOR:    int = 15   # DOOR
VTILE_SDOOR:   int = 16   # SDOOR (secret door)
VTILE_IRONBARS: int = 21  # IRONBARS
VTILE_STAIRS:   int = 25  # STAIRS (vendor rm.h: STAIRS=25).  Stamped by
                          # mkstairs (vendor mklev.c:1566) into ``levl[][].typ``;
                          # consulted by place_niche's !IS_FURNITURE check.
VTILE_ALTAR:    int = 31  # ALTAR (vendor IS_FURNITURE upper bound).

# Vendor door masks (rm.h: D_NODOOR=0, D_BROKEN=1, D_ISOPEN=2, D_CLOSED=4,
# D_LOCKED=8, D_TRAPPED=16).
DMASK_NODOOR:  int = 0
DMASK_ISOPEN:  int = 2
DMASK_CLOSED:  int = 4
DMASK_LOCKED:  int = 8
DMASK_TRAPPED: int = 16

# Map dimensions — vendor COLNO=80, ROWNO=21.
COLNO: int = 80
ROWNO: int = 21

# Vendor DOORMAX (vendor/nle/include/dungeon.h).
DOORMAX: int = 120

# MAXNROFROOMS — vendor/nle/include/global.h:385.
MAXNROFROOMS: int = 40


# ---------------------------------------------------------------------------
# LevelGenState — pytree carried through makecorridors/make_niches.
#
# This is the side-effect surface for the Phase 4 functions.  It deliberately
# stays separate from EnvState; Phase 5 will integrate it.  Shape contract:
#
#   typ          : int8 [COLNO, ROWNO]   tile type per cell (vendor rm.typ)
#   doormask     : int8 [COLNO, ROWNO]   door mask per cell (vendor rm.doormask)
#   door_x       : int8 [DOORMAX]        x-coord of doorindex doors
#   door_y       : int8 [DOORMAX]        y-coord of doorindex doors
#   doorindex    : int32 scalar          live door count
#   smeq         : int32 [MAXNROFROOMS]  union-find component per room
#                                          (vendor mklev.c:90 smeq[])
#   doorct       : int32 [MAXNROFROOMS]  door count per room
#                                          (vendor mkroom.doorct, bumped by
#                                          add_door at mklev.c:1198)
#
# Room data is provided as a separate Rooms input — see below.
# ---------------------------------------------------------------------------

@struct.dataclass
class LevelGenState:
    """Mutable level-generation surface for makecorridors/make_niches.

    Citation: vendor/nle/src/mklev.c globals — ``levl``, ``doors``,
    ``doorindex``, ``smeq``; per-room ``doorct`` from vendor
    ``struct mkroom`` (mkroom.h) bumped in ``add_door`` (mklev.c:1198).
    """
    typ:       jnp.ndarray  # int8 [COLNO, ROWNO]
    doormask:  jnp.ndarray  # int8 [COLNO, ROWNO]
    door_x:    jnp.ndarray  # int8 [DOORMAX]
    door_y:    jnp.ndarray  # int8 [DOORMAX]
    doorindex: jnp.ndarray  # int32 scalar
    smeq:      jnp.ndarray  # int32 [MAXNROFROOMS]
    doorct:    jnp.ndarray  # int32 [MAXNROFROOMS]


def make_empty_level_gen_state() -> LevelGenState:
    """Allocate a fresh LevelGenState with everything STONE and no doors.

    The smeq array is initialised to the identity ``smeq[i] = i`` so each
    room starts in its own union-find component (vendor mklev.c:198 — set
    in ``add_room`` to ``nroom + N_SMEQ`` then normalised; identity is the
    moral equivalent for our use).
    """
    return LevelGenState(
        typ=jnp.full((COLNO, ROWNO), VTILE_STONE, dtype=jnp.int8),
        doormask=jnp.zeros((COLNO, ROWNO), dtype=jnp.int8),
        door_x=jnp.zeros((DOORMAX,), dtype=jnp.int8),
        door_y=jnp.zeros((DOORMAX,), dtype=jnp.int8),
        doorindex=jnp.int32(0),
        smeq=jnp.arange(MAXNROFROOMS, dtype=jnp.int32),
        doorct=jnp.zeros((MAXNROFROOMS,), dtype=jnp.int32),
    )


def stamp_rooms_into_typ(gs: "LevelGenState", rooms: "RoomsBox") -> "LevelGenState":
    """Stamp each active room's walls + floor into ``gs.typ`` per vendor add_room.

    Vendor ``add_room`` (mklev.c:160-182) writes the level grid BEFORE
    makecorridors runs, so finddpos' ``okdoor`` check (which only accepts
    HWALL/VWALL cells) and dig_corridor's ``passable`` reads (STONE is the
    only diggable ``btyp``) see real room geometry.  Our LevelGenState
    grid started all-STONE, so okdoor always failed and every finddpos
    fell through to its ``(xl, yh)`` fallback — shifting door positions
    and diverging the corridor walk's rn2(dix-diy+1) bias spans.

    Vendor add_room stamping for a room (lowx,lowy)-(hix,hiy):
        HWALL on rows  y = lowy-1 and y = hiy+1   (x in lowx-1..hix+1)
        VWALL on cols  x = lowx-1 and x = hix+1   (y in lowy..hiy)
        ROOM  on interior (x in lowx..hix, y in lowy..hiy)
        corners (lowx-1,lowy-1) etc. = TLCORNER/.../BRCORNER (rm.h: 3-6)
    We stamp corners with their proper vendor TLCORNER/TRCORNER/BLCORNER/
    BRCORNER codes so the level grid round-trips through
    ``_vendor_grid_to_terrain`` (branches.py) cleanly: ``_VTYP_TO_TILE``
    maps codes 3-6 directly to WALL, eliminating the need for a
    geometric corner-promotion pass that previously misfired on interior
    ROOM cells flanked by doors on two perpendicular sides (see seed=4
    cell (col=26, row=5) where the W=SDOOR + S=DOOR pair tricked the
    promotion into marking the cell WALL, breaking the graffiti loop's
    ROOM check and over-drawing rn2(40) at draw 1833).
    okdoor remains correct (corners 3-6 are not HWALL/VWALL -> rejected,
    matching vendor) and so does dig_corridor (corners are not STONE/CORR/
    SCORR -> the walker treats them as "strange" and stops, matching
    vendor TLCORNER != btyp/ftyp/SCORR).

    Citation: vendor/nle/src/mklev.c:160-182 (add_room wall/floor stamp);
    mklev.c:175-179 (corners written AFTER the HWALL/VWALL pass).
    """
    xs = jnp.arange(COLNO, dtype=jnp.int32)[:, None]   # [COLNO, 1]
    ys = jnp.arange(ROWNO, dtype=jnp.int32)[None, :]   # [1, ROWNO]

    typ = gs.typ

    def stamp_one(typ_acc, i):
        lx = rooms.lx[i].astype(jnp.int32)
        ly = rooms.ly[i].astype(jnp.int32)
        hx = rooms.hx[i].astype(jnp.int32)
        hy = rooms.hy[i].astype(jnp.int32)
        act = rooms.active[i]

        # Wall band x in [lx-1, hx+1], y in [ly-1, hy+1].
        in_wall_x = (xs >= lx - 1) & (xs <= hx + 1)
        in_wall_y = (ys >= ly - 1) & (ys <= hy + 1)
        # Interior x in [lx, hx], y in [ly, hy].
        interior = (xs >= lx) & (xs <= hx) & (ys >= ly) & (ys <= hy)
        # Top/bottom HWALL rows (within the wall-x band).
        hwall = in_wall_x & ((ys == ly - 1) | (ys == hy + 1))
        # Left/right VWALL cols (only for y in [ly, hy], NOT the corners).
        vwall = ((xs == lx - 1) | (xs == hx + 1)) & (ys >= ly) & (ys <= hy)

        # Vendor writes HWALL/VWALL/ROOM only on success; mask by `act`.
        new = typ_acc
        new = jnp.where(act & hwall, jnp.int8(VTILE_HWALL), new)
        new = jnp.where(act & vwall, jnp.int8(VTILE_VWALL), new)
        new = jnp.where(act & interior, jnp.int8(VTILE_ROOM), new)
        # Corners: the four (lx-1,ly-1)/(hx+1,ly-1)/(lx-1,hy+1)/(hx+1,hy+1)
        # cells fall in the wall band but are neither hwall (they ARE on the
        # hwall rows).  Vendor overwrites them with the corner glyphs AFTER
        # the HWALL pass (mklev.c:175-179: TLCORNER/TRCORNER/BLCORNER/
        # BRCORNER = rm.h 3-6).  Stamp each corner with its proper code so
        # ``_vendor_grid_to_terrain`` (branches.py) can map them directly to
        # WALL via the static ``_VTYP_TO_TILE`` lookup — no geometric
        # corner-promotion required.  This avoids a false-positive on
        # interior ROOM cells whose W and S neighbours happen to be doors
        # (seed=4: cell (col=26, row=5) had a SDOOR on its W wall and a DOOR
        # on its S wall; the old geometric promotion treated both as
        # "wall continuations" and wrongly upgraded the interior cell to
        # WALL, breaking the graffiti loop's ROOM check at draw 1833).
        tl = (xs == lx - 1) & (ys == ly - 1)
        tr = (xs == hx + 1) & (ys == ly - 1)
        bl = (xs == lx - 1) & (ys == hy + 1)
        br = (xs == hx + 1) & (ys == hy + 1)
        new = jnp.where(act & tl, jnp.int8(VTILE_TLCORN), new)
        new = jnp.where(act & tr, jnp.int8(VTILE_TRCORN), new)
        new = jnp.where(act & bl, jnp.int8(VTILE_BLCORN), new)
        new = jnp.where(act & br, jnp.int8(VTILE_BRCORN), new)
        return new, None

    typ, _ = lax.scan(stamp_one, typ, jnp.arange(MAXNROFROOMS, dtype=jnp.int32))
    return gs.replace(typ=typ)


# ---------------------------------------------------------------------------
# Rooms input contract.
#
# makecorridors/make_niches take a Rooms pytree with these per-room fields:
#   lx, ly, hx, hy : int16 [MAXNROFROOMS]  interior bounding box (inclusive)
#   rtype          : int8  [MAXNROFROOMS]  RoomType
#   doorct         : int32 [MAXNROFROOMS]  door count per room
#   fdoor          : int32 [MAXNROFROOMS]  first door index in level doors[]
#
# The existing rooms.py::Room pytree exposes (y1, x1, y2, x2) — Phase 5 will
# expose lx/ly/hx/hy + doorct/fdoor.  For Phase 4 we accept either: the
# functions below ONLY read lx/ly/hx/hy.  We provide a tiny adapter that
# names the fields explicitly.
# ---------------------------------------------------------------------------

@struct.dataclass
class RoomsBox:
    """Minimal per-room rectangle input — vendor mkroom.h struct mkroom."""
    lx:    jnp.ndarray  # int16 [MAXNROFROOMS]
    ly:    jnp.ndarray  # int16 [MAXNROFROOMS]
    hx:    jnp.ndarray  # int16 [MAXNROFROOMS]
    hy:    jnp.ndarray  # int16 [MAXNROFROOMS]
    rtype: jnp.ndarray  # int8  [MAXNROFROOMS]  (0=OROOM)
    active: jnp.ndarray  # bool [MAXNROFROOMS]


# ---------------------------------------------------------------------------
# okdoor — vendor mklev.c:1240-1247.
# ---------------------------------------------------------------------------

def _is_wall(typ: jnp.ndarray) -> jnp.ndarray:
    """typ == HWALL or VWALL (vendor IS_WALL macro)."""
    return (typ == jnp.int8(VTILE_HWALL)) | (typ == jnp.int8(VTILE_VWALL))


def _is_door(typ: jnp.ndarray) -> jnp.ndarray:
    return (typ == jnp.int8(VTILE_DOOR)) | (typ == jnp.int8(VTILE_SDOOR))


def _bydoor(gs: "LevelGenState", x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Vendor mklev.c:1212-1234 ``bydoor`` — any orthogonal neighbour is a door.

    JAX-pure: 4 indexed gathers, OR them.  Out-of-bounds neighbours are
    treated as non-doors (vendor checks ``isok`` before reading).
    """
    def neighbour_is_door(dx, dy):
        nx = x + dx
        ny = y + dy
        in_bounds = (nx >= 0) & (nx < COLNO) & (ny >= 0) & (ny < ROWNO)
        t = gs.typ[jnp.clip(nx, 0, COLNO - 1), jnp.clip(ny, 0, ROWNO - 1)]
        return in_bounds & ((t == jnp.int8(VTILE_DOOR)) | (t == jnp.int8(VTILE_SDOOR)))
    return (neighbour_is_door(-1, 0) | neighbour_is_door(1, 0)
            | neighbour_is_door(0, -1) | neighbour_is_door(0, 1))


def okdoor(gs: "LevelGenState", x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Vendor mklev.c:1240-1247.

    Wall-cell, doorindex < DOORMAX, no adjacent door.  No RNG.
    """
    t = gs.typ[x, y]
    is_wall = (t == jnp.int8(VTILE_HWALL)) | (t == jnp.int8(VTILE_VWALL))
    has_capacity = gs.doorindex < jnp.int32(DOORMAX)
    return is_wall & has_capacity & ~_bydoor(gs, x, y)


# ---------------------------------------------------------------------------
# finddpos — vendor mklev.c:69-96.
#
# Vendor draws rn1(xh-xl+1, xl) then rn1(yh-yl+1, yl) — two scalar draws.
# If okdoor(x,y) succeeds, that's the answer.  Otherwise vendor falls back
# to two deterministic sweeps over the bounding box (no RNG).  We mirror
# this exactly: always draw the two rn1s, then mask in the sweep result.
# ---------------------------------------------------------------------------

def finddpos(
    rng: "Isaac64State",
    gs: "LevelGenState",
    xl: jnp.ndarray, yl: jnp.ndarray,
    xh: jnp.ndarray, yh: jnp.ndarray,
) -> tuple["Isaac64State", jnp.ndarray, jnp.ndarray]:
    """Vendor mklev.c:69-96 finddpos.

    Two RNG draws (always), then bounded sweep fallback (no RNG).
    Returns ``(new_rng, x, y)``.
    """
    xl_i = xl.astype(jnp.int32)
    yl_i = yl.astype(jnp.int32)
    xh_i = xh.astype(jnp.int32)
    yh_i = yh.astype(jnp.int32)

    # rn1(n, base) = base + rn2(n).  Vendor passes n = (xh - xl + 1).  When
    # the range is degenerate (xh < xl, which is the wall-strip slice case),
    # max(n, 1) keeps rn2 valid; the sweep below would still find a slot.
    span_x = jnp.maximum(xh_i - xl_i + jnp.int32(1), jnp.int32(1))
    span_y = jnp.maximum(yh_i - yl_i + jnp.int32(1), jnp.int32(1))
    rng, rx = rn1_jax(rng, span_x, xl_i)
    rng, ry = rn1_jax(rng, span_y, yl_i)

    primary_ok = okdoor(gs, rx, ry)

    # Sweep #1: bounded x in [xl, xh], y in [yl, yh], pick first okdoor cell.
    # Sweep #2: any IS_DOOR cell.  Sweep #3: fallback (xl, yh).
    # We compute all three statically with masks over the full COLNO x ROWNO
    # grid so shapes are static.
    xs = jnp.arange(COLNO, dtype=jnp.int32)
    ys = jnp.arange(ROWNO, dtype=jnp.int32)
    in_x = (xs >= xl_i) & (xs <= xh_i)
    in_y = (ys >= yl_i) & (ys <= yh_i)
    box = in_x[:, None] & in_y[None, :]

    t = gs.typ
    is_wall = (t == jnp.int8(VTILE_HWALL)) | (t == jnp.int8(VTILE_VWALL))

    # bydoor mask, vectorised across the map (4-neighbour OR of door cells).
    is_door_cell = (t == jnp.int8(VTILE_DOOR)) | (t == jnp.int8(VTILE_SDOOR))
    door_pad = jnp.pad(is_door_cell, ((1, 1), (1, 1)), constant_values=False)
    near_door = (door_pad[:-2, 1:-1] | door_pad[2:, 1:-1]
                 | door_pad[1:-1, :-2] | door_pad[1:-1, 2:])
    has_cap = gs.doorindex < jnp.int32(DOORMAX)
    ok_grid = is_wall & has_cap & ~near_door & box

    # First okdoor cell in vendor x-major order (x outer, y inner).
    flat_ok = ok_grid.reshape(-1)
    any_ok = jnp.any(flat_ok)
    first_idx_ok = jnp.argmax(flat_ok).astype(jnp.int32)
    fx_ok = first_idx_ok // jnp.int32(ROWNO)
    fy_ok = first_idx_ok %  jnp.int32(ROWNO)

    door_grid = is_door_cell & box
    flat_d = door_grid.reshape(-1)
    any_d = jnp.any(flat_d)
    first_idx_d = jnp.argmax(flat_d).astype(jnp.int32)
    fx_d = first_idx_d // jnp.int32(ROWNO)
    fy_d = first_idx_d %  jnp.int32(ROWNO)

    # Cascade: primary -> sweep1 -> sweep2 -> (xl, yh) fallback.
    use_primary = primary_ok
    out_x = jnp.where(use_primary, rx,
            jnp.where(any_ok, fx_ok,
            jnp.where(any_d,  fx_d, xl_i)))
    out_y = jnp.where(use_primary, ry,
            jnp.where(any_ok, fy_ok,
            jnp.where(any_d,  fy_d, yh_i)))
    return rng, out_x.astype(jnp.int32), out_y.astype(jnp.int32)


# ---------------------------------------------------------------------------
# dosdoor / dodoor — vendor mklev.c:1249-1260 and 383-447.
#
# dodoor() draws rn2(8): if non-zero use DOOR, else SDOOR.  dosdoor() then
# branches on type and draws rn2(3) gate.  We mirror the rn2 order exactly.
# add_door() is reduced to "append (x, y) at doorindex, doorindex++"; the
# fdoor reshuffle is a vendor optimisation for in-place insertion order that
# doesn't change ISAAC64 consumption.
# ---------------------------------------------------------------------------

def dodoor(
    rng: "Isaac64State",
    gs: "LevelGenState",
    x: jnp.ndarray, y: jnp.ndarray,
    aroom_idx: jnp.ndarray = jnp.int32(-1),
    shdoor: jnp.ndarray = jnp.bool_(False),
    depth: int = 1,
) -> tuple["Isaac64State", "LevelGenState"]:
    """Vendor mklev.c:1249-1260 ``dodoor`` -> mklev.c:383-447 ``dosdoor``.

    ``aroom_idx`` is the owning room (vendor ``aroom`` arg threaded into
    ``add_door``); its ``doorct`` is bumped by 1 (vendor mklev.c:1198).
    A negative index means "no owning room" (doorct untouched).

    Draws (in vendor order):
        rn2(8)   — DOOR vs SDOOR (mklev.c:1259).  UNCONDITIONAL — vendor
                   ``dodoor`` always calls ``dosdoor(x, y, aroom,
                   rn2(8) ? DOOR : SDOOR)``; there is no ``maybe_sdoor``
                   depth gate in this vendor tree.
        rn2(3)   — DOOR: locked/closed/open path vs NODOOR (mklev.c:395)
        rn2(5)   — DOOR&&path: open? (mklev.c:396)
        rn2(6)   — DOOR&&path&&!open: locked? (mklev.c:398)
        rn2(25)  — DOOR&&path&&!open: trapped? (mklev.c:404)
        rn2(5)   — SDOOR: locked vs closed (mklev.c:438)
        rn2(20)  — SDOOR: trapped? (mklev.c:443)

    NOTE: vendor's ``level_difficulty() >= 5`` / ``>= 4`` trap gates and
    ``Is_rogue_level`` override do NOT consume RNG when their LHS is false;
    on Main Dlvl 1 those gates are FALSE so the rn2(25)/rn2(20) draws are
    actually skipped.  Since we target the byte stream of vendor C, we
    encode that with ``lax.cond`` — Phase 5 will pass real level_difficulty.
    For Phase 4 we expose a ``level_difficulty`` int parameter; defaults
    skip the trap rolls (Dlvl 1 behaviour).

    Vendor cite: vendor/nle/src/mklev.c:1249-1260 dodoor + mklev.c:384-447
    dosdoor.
    """
    # rn2(8): DOOR (non-zero) vs SDOOR (zero).  Vendor ``dodoor``
    # unconditionally evaluates ``rn2(8) ? DOOR : SDOOR`` — there is no
    # depth gate, so this draw ALWAYS fires (even on Dlvl 1).
    # Vendor cite: vendor/nle/src/mklev.c:1259.
    rng, r8 = rn2_jax(rng, jnp.int32(8))
    is_door_kind = r8 != jnp.int32(0)

    # Avoid SDOORs on already-made doors: if !IS_WALL(typ), force DOOR.
    # Vendor IS_WALL(typ) ((typ) && (typ) <= DBWALL) covers the full wall
    # family VWALL(1)..DBWALL(12), including TLCORNER..BRCORNER(3..6),
    # CROSSWALL(7), and TUWALL..TRWALL(8..11).  A narrow HWALL/VWALL check
    # would miss corner cells (and other wall variants), causing
    # ``type_is_door`` to short-circuit true on corner-adjacent door sites
    # where vendor produced SDOOR — that emits a spurious DOORWAY tile.
    # Vendor cite: vendor/nle/include/rm.h:85 (IS_WALL) +
    # vendor/nle/src/mklev.c:391 (the !IS_WALL gate).
    cur = gs.typ[x, y]
    is_wall = (cur >= jnp.int8(VTILE_VWALL)) & (cur <= jnp.int8(12))
    type_is_door = is_door_kind | ~is_wall

    # --- DOOR branch — only drawn when type_is_door.  Vendor dosdoor
    # (mklev.c:395-398):
    #     if (!rn2(3)) {            /* doorway/closed/locked path */
    #         if (!rn2(5)) D_ISOPEN;
    #         else if (!rn2(6)) D_LOCKED;
    #         else D_CLOSED;
    #         ...
    #     } else { D_NODOOR; }
    # The rn2(5) is drawn ONLY when rn2(3)==0, and rn2(6) ONLY when
    # additionally rn2(5)!=0.  We mirror that nesting with lax.cond so the
    # ISAAC64 stream matches vendor's short-circuit exactly (an
    # unconditional triple-draw over-consumes when rn2(3)!=0).
    # rn2(25) is further gated by ``level_difficulty() >= 5`` (false on
    # Dlvl 1) so its draw is skipped.
    # Vendor cite: vendor/nle/src/mklev.c:394-404.
    def _draw_door_branch(r):
        r, r3_ = rn2_jax(r, jnp.int32(3))
        # rn2(5) / rn2(6) only when rn2(3)==0 (the !rn2(3) "path" branch).
        def _draw_path(rr):
            rr, r5_ = rn2_jax(rr, jnp.int32(5))
            # rn2(6) only when rn2(5)!=0 (the ``else if (!rn2(6))`` branch).
            def _draw_lock(rrr):
                return rn2_jax(rrr, jnp.int32(6))
            rr, r6_ = lax.cond(
                r5_ != jnp.int32(0),
                _draw_lock,
                lambda rrr: (rrr, jnp.int32(0)),
                rr,
            )
            return rr, r5_, r6_
        r, r5_, r6_ = lax.cond(
            r3_ == jnp.int32(0),
            _draw_path,
            lambda rr: (rr, jnp.int32(0), jnp.int32(0)),
            r,
        )
        return r, r3_, r5_, r6_

    rng, r3, r5d, r6d = lax.cond(
        type_is_door,
        _draw_door_branch,
        lambda r: (r, jnp.int32(0), jnp.int32(0), jnp.int32(0)),
        rng,
    )

    door_path = r3 == jnp.int32(0)
    door_open = door_path & (r5d == jnp.int32(0))
    door_lock = door_path & (~door_open) & (r6d == jnp.int32(0))
    door_mask_path = jnp.where(door_open, jnp.int8(DMASK_ISOPEN),
                     jnp.where(door_lock, jnp.int8(DMASK_LOCKED),
                                          jnp.int8(DMASK_CLOSED)))
    door_mask_nopath = jnp.where(shdoor, jnp.int8(DMASK_ISOPEN),
                                          jnp.int8(DMASK_NODOOR))
    door_mask_door = jnp.where(door_path, door_mask_path, door_mask_nopath)

    # --- SDOOR branch (rn2(5), rn2(20)) — only drawn when !type_is_door.
    # Vendor mklev.c:438-443: the rn2(5)/rn2(20) draws are inside
    # ``else { /* SDOOR */ ... }``.  rn2(20) is additionally gated by
    # ``level_difficulty() >= 4`` which is false on Dlvl 1 — skip draw.
    def _draw_sdoor_branch(r):
        r, r5s_ = rn2_jax(r, jnp.int32(5))
        # rn2(20) difficulty gate: false on Dlvl 1 — skip draw entirely.
        return r, r5s_

    rng, r5s = lax.cond(
        ~type_is_door,
        _draw_sdoor_branch,
        lambda r: (r, jnp.int32(0)),
        rng,
    )

    sdoor_locked = shdoor | (r5s == jnp.int32(0))
    door_mask_sdoor = jnp.where(sdoor_locked, jnp.int8(DMASK_LOCKED),
                                              jnp.int8(DMASK_CLOSED))

    new_typ_val = jnp.where(type_is_door, jnp.int8(VTILE_DOOR),
                                          jnp.int8(VTILE_SDOOR))
    new_mask_val = jnp.where(type_is_door, door_mask_door, door_mask_sdoor)

    new_typ      = gs.typ.at[x, y].set(new_typ_val)
    new_doormask = gs.doormask.at[x, y].set(new_mask_val)

    # add_door: append to doors[] at doorindex.  Vendor reorders to maintain
    # per-room fdoor blocks; that reshuffle does not consume RNG and we
    # ignore it for Phase 4 (Phase 5 integrates the per-room fdoor table).
    idx = jnp.clip(gs.doorindex, 0, DOORMAX - 1)
    new_dx = gs.door_x.at[idx].set(x.astype(jnp.int8))
    new_dy = gs.door_y.at[idx].set(y.astype(jnp.int8))
    new_di = gs.doorindex + jnp.int32(1)

    # Bump the owning room's doorct (vendor add_door mklev.c:1198
    # ``aroom->doorct++``).  A negative aroom_idx means no owning room.
    aroom_i = aroom_idx.astype(jnp.int32)
    has_room = aroom_i >= jnp.int32(0)
    room_slot = jnp.clip(aroom_i, 0, MAXNROFROOMS - 1)
    new_doorct = gs.doorct.at[room_slot].add(
        jnp.where(has_room, jnp.int32(1), jnp.int32(0))
    )

    return rng, LevelGenState(
        typ=new_typ,
        doormask=new_doormask,
        door_x=new_dx,
        door_y=new_dy,
        doorindex=new_di,
        smeq=gs.smeq,
        doorct=new_doorct,
    )


# ---------------------------------------------------------------------------
# dig_corridor — vendor sp_lev.c:2215-2322.
#
# Variable-length corridor walker.  Vendor bounds the loop at cct > 500.
# Per iteration RNG draws (in vendor order):
#   rn2(35)   — early-bail when nxcor (sp_lev.c:2248)
#   rn2(100)  — SCORR-vs-CORR (sp_lev.c:2259)
#   rn2(50)   — boulder placement when nxcor (sp_lev.c:2261); we skip
#                mksobj_at but consume the draw.
#   rn2(dix-diy+1) — bias direction (sp_lev.c:2275/2277), conditional on
#                |dix|>|diy| or |diy|>|dix| AND both non-zero.
#
# We implement as ``lax.while_loop`` with a tight static cap of 600 steps.
# State carries (rng, gs, xx, yy, dx, dy, cct, done, ok).
# ---------------------------------------------------------------------------

_DIG_MAX_STEPS: int = 600


def dig_corridor(
    rng: "Isaac64State",
    gs: "LevelGenState",
    ox: jnp.ndarray, oy: jnp.ndarray,
    tx: jnp.ndarray, ty: jnp.ndarray,
    nxcor: jnp.ndarray,
    ftyp: int = VTILE_CORR,
    btyp: int = VTILE_STONE,
) -> tuple["Isaac64State", "LevelGenState", jnp.ndarray]:
    """Vendor sp_lev.c:2215-2322 dig_corridor.

    Returns ``(new_rng, new_gs, success)``.  When success is False the caller
    (join) suppresses the destination-side dodoor.

    Variable-length inner loop bounded by vendor cct>500 (we use 600 for
    headroom); when the walker reaches (tx, ty) we early-exit by marking
    ``done`` and short-circuiting later iterations.
    """
    # Initial direction selection — vendor sp_lev.c:2234-2241.  This is an
    # if / else-if CHAIN, evaluated in order:
    #     if (tx > xx)      dx = 1;
    #     else if (ty > yy) dy = 1;
    #     else if (tx < xx) dx = -1;
    #     else              dy = -1;
    # The earlier predicates short-circuit the later ones, so e.g. when
    # ``tx < xx`` AND ``ty > yy`` vendor selects dy=1 (the second branch),
    # NOT dx=-1.  A naive ``where(tx>ox,1,where(tx<ox,-1,0))`` for dx would
    # wrongly fire the tx<xx branch in that case and pick the wrong walk
    # direction, diverging the corridor path (and its rn2(dix-diy+1) bias
    # spans).  Encode the chain faithfully.
    b1 = ox.astype(jnp.int32) < tx.astype(jnp.int32)   # tx > xx -> dx=1
    b2 = (~b1) & (oy.astype(jnp.int32) < ty.astype(jnp.int32))  # ty > yy -> dy=1
    b3 = (~b1) & (~b2) & (tx.astype(jnp.int32) < ox.astype(jnp.int32))  # tx<xx -> dx=-1
    # else -> dy=-1
    dx0 = jnp.where(b1, jnp.int32(1),
          jnp.where(b3, jnp.int32(-1), jnp.int32(0)))
    dy0 = jnp.where(b2, jnp.int32(1),
          jnp.where(b1 | b3, jnp.int32(0), jnp.int32(-1)))

    # vendor pre-decrements once before the loop (sp_lev.c:2243-2244).
    xx0 = ox.astype(jnp.int32) - dx0
    yy0 = oy.astype(jnp.int32) - dy0

    tx_i = tx.astype(jnp.int32)
    ty_i = ty.astype(jnp.int32)
    ftyp_i = jnp.int8(ftyp)
    btyp_i = jnp.int8(btyp)

    def cond_fn(carry):
        _, _, xx, yy, _, _, cct, done, _ = carry
        not_at_target = (xx != tx_i) | (yy != ty_i)
        not_capped = cct < jnp.int32(_DIG_MAX_STEPS)
        return not_at_target & not_capped & ~done

    def body_fn(carry):
        r, g, xx, yy, dx, dy, cct, done, ok = carry

        # cct++; cap gate (vendor sp_lev.c:2248 — `if (cct++ > 500 ...)`).
        # Vendor returns FALSE on cap with NO further draws this iteration.
        cct_new = cct + jnp.int32(1)
        cap_hit = cct_new > jnp.int32(500)

        # rn2(35) early-bail: vendor sp_lev.c:2248 draws only when nxcor
        # AND not already cap_hit (vendor's `||` short-circuits on cap_hit).
        # Vendor returns FALSE on bail with NO further draws this iteration.
        def _draw_r35(r_): return rn2_jax(r_, jnp.int32(35))
        r, r35 = lax.cond(nxcor & ~cap_hit, _draw_r35,
                          lambda r_: (r_, jnp.int32(1)), r)
        bail = nxcor & ~cap_hit & (r35 == jnp.int32(0))

        # Step (vendor sp_lev.c:2251-2252). xx_n/yy_n are the candidate cell;
        # OOB check at vendor:2254-2255 returns FALSE with no further draws.
        xx_n = xx + dx
        yy_n = yy + dy
        oob = (xx_n >= COLNO - 1) | (xx_n <= 0) | (yy_n <= 0) | (yy_n >= ROWNO - 1)

        # Vendor sp_lev.c:2257-2269 cell read.  rn2(100) is drawn ONLY when
        # crm->typ == btyp (vendor line 2259). rn2(50) (boulder) is drawn
        # ONLY when nxcor AND rn2(100) selected the ftyp branch (line 2261).
        # When crm->typ is ftyp/SCORR, NO rn2 draws fire.  When crm->typ is
        # strange (not btyp/ftyp/SCORR), vendor returns FALSE at line 2268
        # with no further draws.
        crm_typ = g.typ[jnp.clip(xx_n, 0, COLNO - 1), jnp.clip(yy_n, 0, ROWNO - 1)]
        is_btyp = crm_typ == btyp_i
        is_ftyp = crm_typ == ftyp_i
        is_scorr = crm_typ == jnp.int8(VTILE_SCORR)
        strange = ~is_btyp & ~is_ftyp & ~is_scorr & ~oob

        # Did vendor reach the rn2(100)/rn2(50) block this iter?
        # Only when we are NOT short-circuiting at cap/bail/oob, AND the cell
        # is btyp.  (Strange-cell path returns FALSE before any rn2.)
        reached_btyp_block = ~cap_hit & ~bail & ~oob & is_btyp

        def _draw_r100(r_): return rn2_jax(r_, jnp.int32(100))
        r, r100 = lax.cond(reached_btyp_block, _draw_r100,
                            lambda r_: (r_, jnp.int32(1)), r)

        # ftyp != CORR || rn2(100) — vendor line 2259.  ftyp_i is static here
        # but kept general; when ftyp==CORR the branch is purely r100!=0.
        write_ftyp = jnp.where(jnp.int8(ftyp_i) != jnp.int8(VTILE_CORR),
                                jnp.bool_(True),
                                r100 != jnp.int32(0))

        # rn2(50) boulder: vendor 2261 — only when btyp-block AND nxcor AND
        # write_ftyp (the SCORR branch at line 2264 does NOT draw rn2(50)).
        r50_gate = reached_btyp_block & nxcor & write_ftyp
        def _draw_r50(r_): return rn2_jax(r_, jnp.int32(50))
        r, _r50 = lax.cond(r50_gate, _draw_r50,
                            lambda r_: (r_, jnp.int32(0)), r)
        del _r50

        new_tile = jnp.where(write_ftyp, ftyp_i, jnp.int8(VTILE_SCORR))
        do_write = reached_btyp_block & ~done

        new_typ = lax.cond(
            do_write,
            lambda t: t.at[xx_n, yy_n].set(new_tile),
            lambda t: t,
            g.typ,
        )
        g_new = LevelGenState(
            typ=new_typ, doormask=g.doormask,
            door_x=g.door_x, door_y=g.door_y,
            doorindex=g.doorindex, smeq=g.smeq,
            doorct=g.doorct,
        )

        fail_now = oob | strange | bail | cap_hit

        # ----- direction biasing (vendor sp_lev.c:2272-2279) ----------------
        # Vendor: ``if (dix > diy && diy) { if (!rn2(dix-diy+1)) ... }
        #          else if (diy > dix && dix) { if (!rn2(diy-dix+1)) ... }``
        # Only reached when none of the above returns FALSE fired.
        dix = jnp.abs(xx_n - tx_i)
        diy = jnp.abs(yy_n - ty_i)

        cond_x = ~fail_now & (dix > diy) & (diy != jnp.int32(0))
        cond_y = ~fail_now & (diy > dix) & (dix != jnp.int32(0))

        def _draw_rbx(r_):
            span = jnp.maximum(dix - diy + jnp.int32(1), jnp.int32(1))
            return rn2_jax(r_, span)

        def _draw_rby(r_):
            span = jnp.maximum(diy - dix + jnp.int32(1), jnp.int32(1))
            return rn2_jax(r_, span)

        # If cond_x: draw rbx, skip rby.
        # Else if cond_y: skip rbx, draw rby.
        # Else: skip both.
        r, rbx, rby = lax.cond(
            cond_x,
            lambda r_: (lambda rr, rv: (rr, rv, jnp.int32(1)))(*_draw_rbx(r_)),
            lambda r_: lax.cond(
                cond_y,
                lambda r2: (lambda rr, rv: (rr, jnp.int32(1), rv))(*_draw_rby(r2)),
                lambda r2: (r2, jnp.int32(1), jnp.int32(1)),
                r_,
            ),
            r,
        )
        bias_x = cond_x & (rbx == jnp.int32(0))
        bias_y = cond_y & ~cond_x & (rby == jnp.int32(0))

        dix_eff = jnp.where(bias_x, jnp.int32(0), dix)
        diy_eff = jnp.where(bias_y, jnp.int32(0), diy)

        # Direction-change cascade (vendor:2281-2319).  We pick the new
        # (dx, dy) deterministically (no RNG) based on dix_eff/diy_eff and
        # adjacent cell types.
        def adj_typ(xa, ya):
            return g_new.typ[
                jnp.clip(xx_n + xa, 0, COLNO - 1),
                jnp.clip(yy_n + ya, 0, ROWNO - 1),
            ]
        passable = lambda t: (t == btyp_i) | (t == ftyp_i) | (t == jnp.int8(VTILE_SCORR))

        # Branch A: dy && dix > diy
        ddx = jnp.where(xx_n > tx_i, jnp.int32(-1), jnp.int32(1))
        a_ok = (dy != jnp.int32(0)) & (dix_eff > diy_eff) & passable(adj_typ(ddx, 0))

        # Branch B: dx && diy > dix
        ddy = jnp.where(yy_n > ty_i, jnp.int32(-1), jnp.int32(1))
        b_ok = (dx != jnp.int32(0)) & (diy_eff > dix_eff) & passable(adj_typ(0, ddy))

        # Branch C: continue straight
        c_ok = passable(adj_typ(dx, dy))

        # Branch D: rotate 90° (vendor:2308-2317)
        dx_d = jnp.where(dx != jnp.int32(0), jnp.int32(0),
                         jnp.where(tx_i < xx_n, jnp.int32(-1), jnp.int32(1)))
        dy_d = jnp.where(dx != jnp.int32(0),
                         jnp.where(ty_i < yy_n, jnp.int32(-1), jnp.int32(1)),
                         jnp.int32(0))
        d_ok = passable(adj_typ(dx_d, dy_d))

        # Final fallback: flip d (vendor:2318-2319)
        dx_e = -dx_d
        dy_e = -dy_d

        new_dx = jnp.where(a_ok, ddx,
                 jnp.where(b_ok, jnp.int32(0),
                 jnp.where(c_ok, dx,
                 jnp.where(d_ok, dx_d, dx_e))))
        new_dy = jnp.where(a_ok, jnp.int32(0),
                 jnp.where(b_ok, ddy,
                 jnp.where(c_ok, dy,
                 jnp.where(d_ok, dy_d, dy_e))))

        done_new = done | fail_now | ((xx_n == tx_i) & (yy_n == ty_i))
        ok_new = ok & ~fail_now

        return (r, g_new, xx_n, yy_n, new_dx, new_dy, cct_new, done_new, ok_new)

    init = (rng, gs, xx0, yy0, dx0, dy0, jnp.int32(0), jnp.bool_(False), jnp.bool_(True))
    rng_out, gs_out, _, _, _, _, _, _, ok_out = lax.while_loop(cond_fn, body_fn, init)
    return rng_out, gs_out, ok_out


# ---------------------------------------------------------------------------
# join — vendor mklev.c:243-316.
#
# Mirrors the four-way bounding-box comparison to pick wall slices, then
# calls finddpos twice, dodoor (possibly), dig_corridor, dodoor (possibly).
# smeq union update at the end.
# ---------------------------------------------------------------------------

def _smeq_union(smeq: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Vendor mklev.c:312-315 — ``smeq[max] = smeq[min]``."""
    sa = smeq[a]
    sb = smeq[b]
    new_a = jnp.where(sa < sb, sa, sb)
    new_b = jnp.where(sa < sb, sa, sb)
    # vendor: if smeq[a]<smeq[b] -> smeq[b]=smeq[a]; else smeq[a]=smeq[b]
    smeq = smeq.at[b].set(jnp.where(sa < sb, sa, sb))
    smeq = smeq.at[a].set(jnp.where(sa < sb, sa, sb))
    del new_a, new_b
    return smeq


def join(
    rng: "Isaac64State",
    gs: "LevelGenState",
    rooms: "RoomsBox",
    a_idx: jnp.ndarray,
    b_idx: jnp.ndarray,
    nxcor: jnp.ndarray,
    depth: int = 1,
) -> tuple["Isaac64State", "LevelGenState"]:
    """Vendor mklev.c:243-316 join.

    Skips entire body when either room is inactive (hx < 0) or doorindex
    has hit DOORMAX (vendor:259-260).  All four wall-slice branches draw
    finddpos exactly once each; the unused branch's RNG cost is the same
    bytes either way (vendor picks ONE branch — we do the same with
    lax.cond to keep the byte stream parity).
    """
    a = a_idx.astype(jnp.int32)
    b = b_idx.astype(jnp.int32)

    # Lookups
    c_lx = rooms.lx[a].astype(jnp.int32); c_ly = rooms.ly[a].astype(jnp.int32)
    c_hx = rooms.hx[a].astype(jnp.int32); c_hy = rooms.hy[a].astype(jnp.int32)
    t_lx = rooms.lx[b].astype(jnp.int32); t_ly = rooms.ly[b].astype(jnp.int32)
    t_hx = rooms.hx[b].astype(jnp.int32); t_hy = rooms.hy[b].astype(jnp.int32)

    # Early-out: either room inactive OR doorindex full.
    inactive = ~rooms.active[a] | ~rooms.active[b]
    full     = gs.doorindex >= jnp.int32(DOORMAX)
    skip     = inactive | full

    # Four-way wall-slice selection (vendor:261-289).
    cond_e = t_lx > c_hx                       # b is to the east of a
    cond_n = ~cond_e & (t_hy < c_ly)           # b is north of a
    cond_w = ~cond_e & ~cond_n & (t_hx < c_lx) # b is west of a
    # else south

    dx = jnp.where(cond_e, jnp.int32(1),
         jnp.where(cond_w, jnp.int32(-1), jnp.int32(0)))
    dy = jnp.where(cond_n, jnp.int32(-1),
         jnp.where(~cond_e & ~cond_n & ~cond_w, jnp.int32(1), jnp.int32(0)))

    # cc-side slice (croom edge)
    cc_xl = jnp.where(cond_e, c_hx + jnp.int32(1),
            jnp.where(cond_w, c_lx - jnp.int32(1), c_lx))
    cc_xh = jnp.where(cond_e, c_hx + jnp.int32(1),
            jnp.where(cond_w, c_lx - jnp.int32(1), c_hx))
    cc_yl = jnp.where(cond_e, c_ly,
            jnp.where(cond_n, c_ly - jnp.int32(1),
            jnp.where(cond_w, c_ly, c_hy + jnp.int32(1))))
    cc_yh = jnp.where(cond_e, c_hy,
            jnp.where(cond_n, c_ly - jnp.int32(1),
            jnp.where(cond_w, c_hy, c_hy + jnp.int32(1))))

    # tt-side slice (troom edge)
    tt_xl = jnp.where(cond_e, t_lx - jnp.int32(1),
            jnp.where(cond_w, t_hx + jnp.int32(1), t_lx))
    tt_xh = jnp.where(cond_e, t_lx - jnp.int32(1),
            jnp.where(cond_w, t_hx + jnp.int32(1), t_hx))
    tt_yl = jnp.where(cond_e, t_ly,
            jnp.where(cond_n, t_hy + jnp.int32(1),
            jnp.where(cond_w, t_ly, t_ly - jnp.int32(1))))
    tt_yh = jnp.where(cond_e, t_hy,
            jnp.where(cond_n, t_hy + jnp.int32(1),
            jnp.where(cond_w, t_hy, t_ly - jnp.int32(1))))

    # When skipping the whole join we still must NOT consume RNG — use
    # lax.cond.  Vendor returns immediately at line 260 with no draws.
    def do_join(carry):
        r, g = carry
        # finddpos cc, finddpos tt — always two pairs of rn1 draws each.
        r, ccx, ccy = finddpos(r, g, cc_xl, cc_yl, cc_xh, cc_yh)
        r, ttx, tty = finddpos(r, g, tt_xl, tt_yl, tt_xh, tt_yh)

        xx = ccx
        yy = ccy
        tx_adj = ttx - dx
        ty_adj = tty - dy

        # nxcor && levl[xx+dx][yy+dy].typ -> return without further draws
        next_typ = g.typ[
            jnp.clip(xx + dx, 0, COLNO - 1),
            jnp.clip(yy + dy, 0, ROWNO - 1),
        ]
        nxcor_block = nxcor & (next_typ != jnp.int8(VTILE_STONE))

        # First dodoor (cc) — only if okdoor || !nxcor, but vendor still
        # consumes the rn2(8)/rn2(3)/... RNG sequence ONLY when the call
        # actually fires.  We mirror with lax.cond.
        def call_first_dodoor(rg):
            r_, g_ = rg
            # cc-side door belongs to croom (room a) — vendor join passes
            # croom to dodoor at mklev.c:301.  Cite: mklev.c:299-301.
            return dodoor(r_, g_, xx, yy, aroom_idx=a, depth=depth)
        ok = okdoor(g, xx, yy) | ~nxcor
        first_gate = ok & ~nxcor_block
        r, g = lax.cond(first_gate, call_first_dodoor, lambda rg: rg, (r, g))

        # dig_corridor — only fires if not nxcor_block.  Capture its success
        # flag: vendor ``if (!dig_corridor(...)) return;`` (mklev.c:304-306)
        # aborts the join with NO second dodoor and NO smeq union when the
        # dig fails — including the ``nxcor && !rn2(35)`` early-bail
        # (sp_lev.c:2248), which consumes the rn2(35) draw but still returns
        # FALSE.  Approximating success with ``~nxcor_block`` wrongly fired
        # the second dodoor (an extra rn2(8)) after such a bail.
        def call_dig(rg):
            r_, g_ = rg
            r_, g_, ok_ = dig_corridor(
                r_, g_, xx + dx, yy + dy, tx_adj, ty_adj, nxcor)
            return r_, g_, ok_
        r, g, dig_ok = lax.cond(
            ~nxcor_block,
            call_dig,
            lambda rg: (rg[0], rg[1], jnp.bool_(False)),
            (r, g),
        )

        # Second dodoor (tt) — vendor mklev.c:309-310, only when the dig
        # succeeded (dig_ok) AND (okdoor(tt) || !nxcor).
        ok2 = okdoor(g, ttx, tty) | ~nxcor
        second_gate = dig_ok & ok2

        def call_second_dodoor(rg):
            r_, g_ = rg
            # tt-side door belongs to troom (room b) — vendor join passes
            # troom to dodoor at mklev.c:310.  Cite: mklev.c:309-310.
            return dodoor(r_, g_, ttx, tty, aroom_idx=b, depth=depth)
        r, g = lax.cond(second_gate, call_second_dodoor, lambda rg: rg, (r, g))

        # smeq union (vendor mklev.c:312-315) — no RNG.  Also gated on
        # dig_ok: vendor returns before this when the dig fails, so the
        # union-find merge must not happen on a failed join.
        def do_smeq(g_):
            new_smeq = _smeq_union(g_.smeq, a, b)
            return LevelGenState(
                typ=g_.typ, doormask=g_.doormask,
                door_x=g_.door_x, door_y=g_.door_y,
                doorindex=g_.doorindex, smeq=new_smeq,
                doorct=g_.doorct,
            )
        g = lax.cond(dig_ok, do_smeq, lambda g_: g_, g)
        return r, g

    rng_out, gs_out = lax.cond(skip, lambda rg: rg, do_join, (rng, gs))
    return rng_out, gs_out


# ---------------------------------------------------------------------------
# makecorridors — vendor mklev.c:319-348.
#
# Four sequential passes:
#   P1: join(a, a+1, FALSE) for a in [0, nroom-1), broken early on !rn2(50).
#   P2: join(a, a+2, FALSE) for a in [0, nroom-2) when smeq[a]!=smeq[a+2].
#   P3: while any { for b in [0, nroom): if smeq[a]!=smeq[b] join(a,b,FALSE) }
#       — bounded outer at nroom passes (vendor `any && a < nroom`).
#   P4: if nroom > 2: for i in rn2(nroom)+4 join(rn2(nroom), rn2(nroom-2)+>=a?2)
# ---------------------------------------------------------------------------

def makecorridors(
    rng: "Isaac64State",
    gs: "LevelGenState",
    rooms: "RoomsBox",
    nroom: jnp.ndarray,
    depth: int = 1,
) -> tuple["Isaac64State", "LevelGenState"]:
    """Vendor mklev.c:319-348 makecorridors.

    Args:
        rng:    ISAAC64 state.
        gs:     :class:`LevelGenState` — typ/doormask/doors/smeq carried.
        rooms:  :class:`RoomsBox` with per-room bounding boxes.
        nroom:  number of active rooms (int32 scalar).

    Returns:
        (new_rng, new_gs).
    """
    nroom_i = nroom.astype(jnp.int32)

    # ---- Pass 1: sequential a..a+1 with rn2(50) early-bail ---------------
    def p1_body(i, carry):
        r, g, broken = carry
        # Vendor draws rn2(50) AFTER the join (mklev.c:325-327).
        def do_step(rg):
            r_, g_ = rg
            r_, g_ = join(r_, g_, rooms, jnp.int32(i), jnp.int32(i) + jnp.int32(1), jnp.bool_(False), depth=depth)
            r_, r50 = rn2_jax(r_, jnp.int32(50))
            broken_new = r50 == jnp.int32(0)
            return r_, g_, broken_new
        r, g, broken_new = lax.cond(
            broken | (jnp.int32(i) >= nroom_i - jnp.int32(1)),
            lambda rg: (rg[0], rg[1], broken),
            do_step,
            (r, g),
        )
        return (r, g, broken_new)

    rng, gs, _ = lax.fori_loop(
        0, MAXNROFROOMS, p1_body, (rng, gs, jnp.bool_(False))
    )

    # ---- Pass 2: a..a+2 if smeq differs (no RNG gate) --------------------
    def p2_body(i, carry):
        r, g = carry
        active = jnp.int32(i) < nroom_i - jnp.int32(2)
        def do_step(rg):
            r_, g_ = rg
            differ = g_.smeq[i] != g_.smeq[i + 2]
            return lax.cond(
                differ,
                lambda rg2: join(rg2[0], rg2[1], rooms,
                                  jnp.int32(i), jnp.int32(i) + jnp.int32(2),
                                  jnp.bool_(False), depth=depth),
                lambda rg2: rg2,
                (r_, g_),
            )
        return lax.cond(active, do_step, lambda rg: rg, (r, g))

    rng, gs = lax.fori_loop(0, MAXNROFROOMS, p2_body, (rng, gs))

    # ---- Pass 3: "any" loop — bounded at nroom outer passes -------------
    # Vendor: for (a = 0; any && a < nroom; a++) { any=FALSE; for b ... }
    # The "any" flag resets per outer pass; we run MAXNROFROOMS outer passes
    # max (vendor's nroom).  Each (a, b) pair calls join when smeq differs.
    def p3_outer(a, carry_outer):
        r, g, any_flag = carry_outer
        outer_active = (jnp.int32(a) < nroom_i) & any_flag

        def do_outer(rg_any):
            r_, g_, _ = rg_any
            def p3_inner(b, carry_in):
                r_i, g_i, any_i = carry_in
                inner_active = jnp.int32(b) < nroom_i
                def do_inner(rg):
                    r2, g2, _any = rg
                    differ = g2.smeq[a] != g2.smeq[b]
                    def call_join(rg2):
                        return join(rg2[0], rg2[1], rooms,
                                    jnp.int32(a), jnp.int32(b),
                                    jnp.bool_(False), depth=depth)
                    r3, g3 = lax.cond(differ, call_join, lambda rg2: rg2, (r2, g2))
                    return r3, g3, _any | differ
                return lax.cond(inner_active, do_inner, lambda rg: rg, (r_i, g_i, any_i))
            r_, g_, any_after = lax.fori_loop(
                0, MAXNROFROOMS, p3_inner, (r_, g_, jnp.bool_(False))
            )
            return r_, g_, any_after

        return lax.cond(outer_active, do_outer, lambda rg: rg, (r, g, any_flag))

    rng, gs, _ = lax.fori_loop(
        0, MAXNROFROOMS, p3_outer, (rng, gs, jnp.bool_(True))
    )

    # ---- Pass 4: extra cross-connects if nroom > 2 -----------------------
    def p4_body(_, carry):
        r, g = carry
        # rn2(nroom), rn2(nroom-2), then b += 2 if b >= a.
        nroom_safe = jnp.maximum(nroom_i, jnp.int32(1))
        nm2_safe   = jnp.maximum(nroom_i - jnp.int32(2), jnp.int32(1))
        r, a = rn2_jax(r, nroom_safe)
        r, b = rn2_jax(r, nm2_safe)
        b = jnp.where(b >= a, b + jnp.int32(2), b)
        return join(r, g, rooms, a, b, jnp.bool_(True), depth=depth)

    def do_p4(carry):
        r, g = carry
        nroom_safe = jnp.maximum(nroom_i, jnp.int32(1))
        r, i_count = rn2_jax(r, nroom_safe)
        total = i_count + jnp.int32(4)
        return lax.fori_loop(0, total, p4_body, (r, g))

    rng, gs = lax.cond(nroom_i > jnp.int32(2), do_p4, lambda rg: rg, (rng, gs))

    return rng, gs


# ---------------------------------------------------------------------------
# make_niches — vendor mklev.c:548-565 + makeniche 483-546 + place_niche
# 450-471.
#
# Vendor RNG draws per niche attempt (rn2(nroom) for room pick, rn2(5) gate
# on doorct==1, then place_niche's rn2(2) + finddpos draws, then rn2(4) or
# 0/rn2(7)/rn2(5) cascade).  We mirror the byte-exact order; the side-effect
# part (rm.typ=SCORR, traps, mksobj_at, dosdoor) only the structural pieces
# we have surface for in LevelGenState — Phase 5 wires the rest.
#
# Phase 4 scope: byte-exact RNG consumption, plus dosdoor side-effect when
# the niche actually places.  Trap creation / corpse / mksobj are draws
# only; carved-out tile setting is performed.
# ---------------------------------------------------------------------------

def _place_niche(
    rng: "Isaac64State",
    gs: "LevelGenState",
    rooms: "RoomsBox",
    aroom_idx: jnp.ndarray,
) -> tuple["Isaac64State", jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Vendor mklev.c:450-471 place_niche.

    Draws rn2(2) to pick dy=±1, then finddpos along that wall strip.
    Returns ``(rng, ok, xx, yy, dy)``.
    """
    a = aroom_idx.astype(jnp.int32)
    lx = rooms.lx[a].astype(jnp.int32)
    ly = rooms.ly[a].astype(jnp.int32)
    hx = rooms.hx[a].astype(jnp.int32)
    hy = rooms.hy[a].astype(jnp.int32)

    rng, r2 = rn2_jax(rng, jnp.int32(2))
    dy = jnp.where(r2 != jnp.int32(0), jnp.int32(1), jnp.int32(-1))

    # finddpos along the wall strip.  Below room (dy=1) -> y=hy+1.
    # Above room (dy=-1) -> y=ly-1.
    strip_y = jnp.where(dy == jnp.int32(1), hy + jnp.int32(1), ly - jnp.int32(1))
    rng, xx, yy = finddpos(rng, gs, lx, strip_y, hx, strip_y)

    # Validity (vendor:466-470): isok(xx, yy+dy) && levl[][].typ==STONE
    # && isok(xx, yy-dy) && !IS_POOL && !IS_FURNITURE.
    out_x = xx
    out_y = yy
    in_a = (out_y + dy >= 0) & (out_y + dy < ROWNO)
    in_b = (out_y - dy >= 0) & (out_y - dy < ROWNO)
    typ_a = gs.typ[out_x, jnp.clip(out_y + dy, 0, ROWNO - 1)]
    typ_b = gs.typ[out_x, jnp.clip(out_y - dy, 0, ROWNO - 1)]
    # IS_FURNITURE(typ) := STAIRS<=typ<=ALTAR (vendor rm.h:104).  Without this
    # check the niche may try to back-onto the downstair cell, which vendor
    # rejects.  Seed=4 hits exactly this: room aidx=3 at (26,3)-(28,5) has
    # the downstair stamped at (28, 3); niche attempt with yy=2, dy=-1 has
    # yy-dy=3 (STAIRS), so vendor's place_niche returns false.
    # POOL check (vendor IS_POOL = POOL<=typ<=DRAWBRIDGE_UP, [16, 19]) is also
    # required by vendor; on Dlvl 1 main branch there are no pools yet, but
    # we include it for completeness.
    is_furniture_b = (typ_b >= jnp.int8(VTILE_STAIRS)) & (typ_b <= jnp.int8(VTILE_ALTAR))
    is_pool_b = (typ_b >= jnp.int8(16)) & (typ_b <= jnp.int8(19))
    ok = (in_a & in_b
          & (typ_a == jnp.int8(VTILE_STONE))
          & ~is_furniture_b
          & ~is_pool_b)
    return rng, ok, out_x, out_y, dy


def _makeniche(
    rng: "Isaac64State",
    gs: "LevelGenState",
    rooms: "RoomsBox",
    nroom: jnp.ndarray,
    trap_type: jnp.ndarray,
    depth: int = 1,
) -> tuple["Isaac64State", "LevelGenState"]:
    """Vendor mklev.c:483-546 makeniche.

    Loops up to vct=8 attempts.  Per attempt:
        rn2(nroom)         room pick
        rn2(5)             doorct==1 gate (drawn when room is OROOM)
        (place_niche draws: rn2(2) + finddpos rn1 pair)
        rn2(4)             SCORR-vs-CORR gate (vendor:504)

        SCORR branch: dosdoor(xx, yy, SDOOR)
            → dosdoor SDOOR draws: rn2(5) locked-vs-closed
              [+rn2(20) if depth>=4, skipped on shallow levels]

        CORR branch:
            rn2(7)         door-vs-inaccessible gate (vendor:525)
            if rn2(7)!=0:  rn2(5) SDOOR-vs-DOOR pick (vendor:526)
                           dosdoor(xx, yy, type)  — DOOR or SDOOR draws
            else:          rn2(5) ironbars gate (vendor:529)
                           [rn2(3) corpse if ironbars fires] (vendor:531)
                           rn2(3) mkobj gate (vendor:533)

    Vendor cite: vendor/nethack/src/mklev.c:483-546 makeniche.
    """
    nroom_i = nroom.astype(jnp.int32)
    OROOM = jnp.int8(0)  # vendor mkroom.h OROOM == 0

    def attempt(_, carry):
        r, g, done = carry

        # Vendor ``while (vct--)`` runs until the first successful placement
        # returns; iterations after that ``return`` draw NOTHING.  Gate the
        # whole body on ``~done`` so the ISAAC64 stream stops exactly when
        # vendor returns (mklev.c:539).  Vendor cite: vendor/nle/src/mklev.c:491-539.
        def _do_attempt(rg):
            r, g = rg

            # rn2(nroom) room pick — vendor mklev.c:493.
            nroom_safe = jnp.maximum(nroom_i, jnp.int32(1))
            r, aidx = rn2_jax(r, nroom_safe)

            # ``if (aroom->rtype != OROOM) continue;`` (mklev.c:495) — no
            # further draw for a non-OROOM room.
            is_oroom = rooms.rtype[aidx] == OROOM

            # ``if (aroom->doorct == 1 && rn2(5)) continue;`` (mklev.c:497).
            # C short-circuit: rn2(5) is drawn ONLY when the room is OROOM
            # AND doorct == 1.  Vendor cite: vendor/nle/src/mklev.c:497.
            doorct_is_one = g.doorct[aidx] == jnp.int32(1)
            draw_gate = is_oroom & doorct_is_one
            r, r5gate = lax.cond(
                draw_gate, lambda rr: rn2_jax(rr, jnp.int32(5)),
                lambda rr: (rr, jnp.int32(0)), r,
            )
            # Skip (continue) when non-OROOM, OR doorct==1 gate fired nonzero.
            gate_continue = (~is_oroom) | (doorct_is_one & (r5gate != jnp.int32(0)))

            # ``if (!place_niche(...)) continue;`` (mklev.c:499) — place_niche
            # draws rn2(2) + finddpos; reached only when not skipped.
            def _do_pn(rr):
                rr, pok_, xx_, yy_, dy_ = _place_niche(rr, g, rooms, aidx)
                return rr, pok_, xx_, yy_, dy_
            r, pok, xx, yy, dy_niche = lax.cond(
                ~gate_continue, _do_pn,
                lambda rr: (rr, jnp.bool_(False), jnp.int32(0), jnp.int32(0), jnp.int32(0)),
                r,
            )

            placed = (~gate_continue) & pok

            # rn2(4) — SCORR vs CORR branch (vendor mklev.c:504:
            # ``if (trap_type || !rn2(4))``).  C short-circuit: when
            # ``trap_type != 0`` the ``rn2(4)`` is NEVER drawn — the SCORR
            # branch fires unconditionally.  Only draw on a successful
            # placement AND when trap_type == 0.  Callers using a non-zero
            # trap_type (e.g. makevtele -> TELEP_TRAP=15) MUST see zero
            # draws here to keep the ISAAC64 stream byte-aligned.
            draw_r4 = placed & (trap_type == jnp.int32(0))
            r, r4 = lax.cond(
                draw_r4, lambda rr: rn2_jax(rr, jnp.int32(4)),
                lambda rr: (rr, jnp.int32(0)), r,
            )
            scorr_branch = (trap_type != jnp.int32(0)) | (r4 == jnp.int32(0))

            # SCORR branch: maketrap + (engraving) + dosdoor.  Vendor:
            # mklev.c:505-522.  Draws (in order):
            #   1. maketrap(trap_type) — zero draws for non-switch trap types
            #      (TELEP_TRAP=15, LEVEL_TELEP=16, TRAPDOOR=14 all fall through
            #      the maketrap switch with no RNG).  Vendor cite:
            #      vendor/nle/src/trap.c:315-443 (maketrap, only SQKY_BOARD /
            #      STATUE_TRAP / ROLLING_BOULDER / PIT/HOLE/TRAPDOOR-furniture
            #      paths draw).
            #   2. if trap_engravings[trap_type] != NULL:
            #        make_engr_at(..., DUST) — zero draws (e_type=DUST>0
            #          short-circuits the rnd(N_ENGRAVE-1) draw at
            #          engrave.c:412).
            #        wipe_engr_at(xx, yy-dy, 5, FALSE) -> wipeout_text(s, 5, 0)
            #          DUST engravings skip the cnt reduction at engrave.c:301,
            #          so wipeout_text loops 5 times drawing
            #          ``rn2(lth), rn2(4)`` per iteration (engrave.c:95-96).
            #          The string length is fixed per trap_type:
            #            TRAPDOOR (14):    "Vlad was here"   lth = 13
            #            TELEP_TRAP (15):  "ad aerarium"     lth = 11
            #            LEVEL_TELEP (16): "ad aerarium"     lth = 11
            #            other trap_type: NULL -> wipe_engr_at not called.
            #   3. dosdoor(xx, yy, aroom, SDOOR): rn2(5) locked-vs-closed
            #      [+rn2(20) if depth>=4]. add_door bumps the niche room's
            #      doorct.  Vendor cite: vendor/nle/src/mklev.c:516,1198.
            #
            # Caller's responsibility: ensure ``trap_type`` matches vendor.
            def _scorr_path(rg2):
                r_, g_ = rg2

                # Step 2: wipe_engr_at(5) draws for engraved traps.  The
                # engraving length is static per trap_type — pick it via
                # jnp.where so the draw count stays compile-time-constant (5)
                # but the modulus is data-dependent.  This is correct because
                # mod doesn't affect ISAAC64 stream advancement (each rn2
                # consumes exactly one uint64).
                # NB: ``has_engr`` could also be data-dependent but on the
                # Main Dlvl 1 makevtele path it is always True (TELEP_TRAP=15).
                # We gate with jnp.where instead of lax.cond so the trace is
                # straight-line — every ISAAC64 draw advances unconditionally
                # but masked out via lax.cond at the helper boundary below.
                _engr_lth = jnp.where(
                    (trap_type == jnp.int32(14)),
                    jnp.int32(13),  # "Vlad was here"
                    jnp.where(
                        (trap_type == jnp.int32(15))
                        | (trap_type == jnp.int32(16)),
                        jnp.int32(11),  # "ad aerarium"
                        jnp.int32(0),
                    ),
                )
                has_engr = _engr_lth != jnp.int32(0)

                # wipeout_text(s, 5, seed=0) — vendor engrave.c:80-142.  Each
                # iteration draws ``rn2(lth)`` then ``rn2(4)``.  When the
                # picked char is neither space nor in the trivial-rub set
                # ``"?.,'\`-|_"``, AND ``use_rubout != 0``, AND the char
                # matches a wipefrom in the rubouts table (engrave.c:27-78),
                # vendor draws an inner ``rn2(strlen(wipeto))`` and writes
                # ``wipeto[j]`` back into the engraving.  Otherwise the cell
                # is overwritten with ``'?'`` (the unreadable fallback) and
                # no inner draw fires.  Modelling the dynamic engraving is
                # required because the same cell can be picked across
                # iterations — its current contents drive whether the inner
                # draw fires (seed=7 hits this twice in 5 iterations).
                _engr_init = _engr_template(_engr_lth)
                _lth_safe = jnp.maximum(_engr_lth, jnp.int32(1))
                _SPACE = jnp.int32(ord(' '))
                _QMARK = jnp.int32(ord('?'))

                def _wipe_one(_, carry):
                    rr, engr = carry
                    rr, r1 = rn2_jax(rr, _lth_safe)
                    # Vendor wipeout_text draws BOTH rn2(lth) AND rn2(4)
                    # UNCONDITIONALLY (engrave.c:95-96), BEFORE the space
                    # and trivial-char `continue` checks at engrave.c:106
                    # and :110.  Per-iter draw count is always 2 outer +
                    # optional 1 inner (rubouts match).  Vendor cite:
                    # vendor/nle/src/engrave.c:91-103.
                    rr, r2 = rn2_jax(rr, jnp.int32(4))
                    ch = engr[r1]
                    is_space = ch == _SPACE
                    is_trivial = jnp.any(_TRIVIAL_WIPE_CHARS == ch)
                    use_rubout = r2 != jnp.int32(0)
                    ch_idx = jnp.clip(ch, jnp.int32(0), jnp.int32(127))
                    wipeto_len = _RUBOUT_LEN[ch_idx]
                    has_rubout = wipeto_len > jnp.int32(0)
                    # Vendor enters the rubouts loop only when NOT space AND
                    # NOT trivial AND use_rubout!=0.  Inside, the inner draw
                    # fires only when a wipefrom row matches.
                    do_inner = (~is_space) & (~is_trivial) & use_rubout & has_rubout
                    rr, r3 = lax.cond(
                        do_inner,
                        lambda rr_: rn2_jax(rr_, jnp.maximum(wipeto_len, jnp.int32(1))),
                        lambda rr_: (rr_, jnp.int32(0)),
                        rr,
                    )
                    # Compute the replacement char.  Branches (vendor order):
                    #   space        -> unchanged (continue, no replace)
                    #   trivial set  -> ' '
                    #   use_rubout=0 -> '?' (i=SIZE(rubouts) fallthrough)
                    #   matched     -> wipeto[r3]
                    #   no match    -> '?'
                    replaced_rubout = _RUBOUT_CHARS[ch_idx, jnp.clip(r3, 0, 4)]
                    new_ch = jnp.where(
                        is_space, ch,
                        jnp.where(
                            is_trivial, _SPACE,
                            jnp.where(
                                ~use_rubout, _QMARK,
                                jnp.where(has_rubout, replaced_rubout, _QMARK),
                            ),
                        ),
                    )
                    engr = engr.at[r1].set(new_ch)
                    return rr, engr

                def _do_wipe(rr):
                    rr_out, _engr_final = lax.fori_loop(
                        0, 5, _wipe_one, (rr, _engr_init),
                    )
                    return rr_out

                r_ = lax.cond(has_engr, _do_wipe, lambda rr: rr, r_)

                # Step 3: dosdoor(SDOOR) — rn2(5) locked-vs-closed.
                r_, r5s = rn2_jax(r_, jnp.int32(5))
                sdoor_locked = r5s == jnp.int32(0)
                mask_s = jnp.where(sdoor_locked, jnp.int8(DMASK_LOCKED), jnp.int8(DMASK_CLOSED))
                if depth >= 4:
                    r_, _r20 = rn2_jax(r_, jnp.int32(20))
                new_typ = g_.typ.at[xx, yy].set(jnp.int8(VTILE_SDOOR))
                new_dm  = g_.doormask.at[xx, yy].set(mask_s)
                idx = jnp.clip(g_.doorindex, 0, DOORMAX - 1)
                new_dx = g_.door_x.at[idx].set(xx.astype(jnp.int8))
                new_dy = g_.door_y.at[idx].set(yy.astype(jnp.int8))
                new_di = g_.doorindex + jnp.int32(1)
                slot = jnp.clip(aidx, 0, MAXNROFROOMS - 1)
                new_doorct = g_.doorct.at[slot].add(jnp.int32(1))
                return r_, LevelGenState(
                    typ=new_typ, doormask=new_dm,
                    door_x=new_dx, door_y=new_dy,
                    doorindex=new_di, smeq=g_.smeq, doorct=new_doorct,
                )

            # CORR branch: vendor mklev.c:518-540.
            def _corr_path(rg2):
                r_, g_ = rg2
                r_, r7 = rn2_jax(r_, jnp.int32(7))
                has_door = r7 != jnp.int32(0)

                # rn2(7)!=0 sub-path: rn2(5) SDOOR-vs-DOOR pick + dosdoor
                # (vendor:526).  add_door bumps doorct.
                def _corr_door(rg3):
                    r2, g2 = rg3
                    r2, r5d = rn2_jax(r2, jnp.int32(5))
                    use_sdoor = r5d != jnp.int32(0)

                    def _door_type(r3):
                        # Vendor dosdoor DOOR branch — mklev.c:394-405.  The
                        # locked/closed/doorway rolls (rn2(5), rn2(6)) and the
                        # depth-gated rn2(25) trapped roll are drawn ONLY inside
                        # the ``if (!rn2(3))`` true-branch.  When rn2(3)!=0 the
                        # else-branch sets D_NODOOR and draws NOTHING.  Gate the
                        # follow-on draws on the rn2(3) result so the ISAAC64
                        # stream matches vendor exactly.
                        r3, r3v = rn2_jax(r3, jnp.int32(3))
                        door_path = r3v == jnp.int32(0)

                        def _doorway(rr):
                            rr, r5v = rn2_jax(rr, jnp.int32(5))
                            rr, r6v = rn2_jax(rr, jnp.int32(6))
                            if depth >= 5:
                                rr, _r25 = rn2_jax(rr, jnp.int32(25))
                            door_open = r5v == jnp.int32(0)
                            door_lock = (~door_open) & (r6v == jnp.int32(0))
                            m = jnp.where(door_open, jnp.int8(DMASK_ISOPEN),
                                jnp.where(door_lock, jnp.int8(DMASK_LOCKED),
                                                     jnp.int8(DMASK_CLOSED)))
                            return rr, m

                        r3, mask_d = lax.cond(
                            door_path, _doorway,
                            lambda rr: (rr, jnp.int8(DMASK_NODOOR)), r3,
                        )
                        return r3, mask_d

                    def _sdoor_type(r3):
                        r3, r5s = rn2_jax(r3, jnp.int32(5))
                        if depth >= 4:
                            r3, _r20 = rn2_jax(r3, jnp.int32(20))
                        mask_s = jnp.where(r5s == jnp.int32(0),
                                           jnp.int8(DMASK_LOCKED), jnp.int8(DMASK_CLOSED))
                        return r3, mask_s

                    r2, door_mask = lax.cond(use_sdoor, _sdoor_type, _door_type, r2)
                    new_tile = jnp.where(use_sdoor, jnp.int8(VTILE_SDOOR), jnp.int8(VTILE_DOOR))
                    new_typ = g2.typ.at[xx, yy].set(new_tile)
                    new_dm  = g2.doormask.at[xx, yy].set(door_mask)
                    idx = jnp.clip(g2.doorindex, 0, DOORMAX - 1)
                    new_dx = g2.door_x.at[idx].set(xx.astype(jnp.int8))
                    new_dy = g2.door_y.at[idx].set(yy.astype(jnp.int8))
                    new_di = g2.doorindex + jnp.int32(1)
                    slot = jnp.clip(aidx, 0, MAXNROFROOMS - 1)
                    new_doorct = g2.doorct.at[slot].add(jnp.int32(1))
                    return r2, LevelGenState(
                        typ=new_typ, doormask=new_dm,
                        door_x=new_dx, door_y=new_dy,
                        doorindex=new_di, smeq=g2.smeq, doorct=new_doorct,
                    )

                # rn2(7)==0 sub-path: inaccessible niche (vendor:529-540).
                # No door -> no doorct bump.
                #
                # Vendor sequence (mklev.c:528-540):
                #   if (!rn2(5) && IS_WALL(...)) {           # ironbars gate
                #       levl[xx][yy].typ = IRONBARS;
                #       if (rn2(3)) mkcorpstat(CORPSE, ...);  # corpse roll
                #   }
                #   if (!level.flags.noteleport)
                #       mksobj_at(SCR_TELEPORTATION, ...);    # SCROLL_CLASS init
                #   if (!rn2(3)) mkobj_at(0, ...);            # mkobj gate
                #
                # The mksobj_at(SCR_TELEPORTATION) call invokes mksobj(otyp,
                # TRUE, FALSE) which runs the SCROLL_CLASS mksobj_init branch
                # (mkobj.c:981-987) = boc(4) = rn2(4) + cond rn2(2).  This
                # was missing prior to this fix, shifting the ISAAC64 stream
                # by 2 draws on the inaccessible-niche path.  Dlvl 1 in the
                # main dungeon has noteleport=False so the scroll always
                # spawns; we unconditionally consume the draws here (the
                # ``noteleport`` param is not threaded down to _makeniche).
                # Vendor cite: vendor/nle/src/mklev.c:536-538,
                # vendor/nle/src/mkobj.c:981-987 (SCROLL_CLASS boc(4)).
                def _corr_inaccessible(rg3):
                    r2, g2 = rg3
                    r2, r5i = rn2_jax(r2, jnp.int32(5))   # !rn2(5) ironbars gate
                    ironbars = r5i == jnp.int32(0)

                    def _ironbars_true(r3):
                        r3, r3c = rn2_jax(r3, jnp.int32(3))
                        del r3c
                        return r3

                    r2 = lax.cond(ironbars, _ironbars_true, lambda r3: r3, r2)

                    # mksobj_at(SCR_TELEPORTATION) → SCROLL_CLASS mksobj_init
                    # = blessorcurse(4) = rn2(4) [+ cond rn2(2)].
                    r2 = consume_mksobj_init_draws(
                        r2,
                        jnp.int32(int(ObjectClass.SCROLL_CLASS)),
                    )

                    # if (!rn2(3)) mkobj_at(0, xx, yy + dy, TRUE);
                    # When r3mk == 0, vendor invokes mkobj() with
                    # RANDOM_CLASS: rnd(1000) prob + rnd(100) class-pick +
                    # mksobj_init for the picked class.  Without consuming
                    # these draws Nethax desyncs by 2+ draws whenever the
                    # gate fires (seed=5 hits this on the first
                    # inaccessible-niche placement).
                    # Vendor cite: vendor/nle/src/mklev.c:539-540,
                    #              vendor/nle/src/mkobj.c:249-301 (mkobj).
                    r2, r3mk = rn2_jax(r2, jnp.int32(3))
                    do_mkobj = r3mk == jnp.int32(0)
                    r2 = lax.cond(
                        do_mkobj,
                        lambda r3: consume_mkobj_random_draws(r3),
                        lambda r3: r3,
                        r2,
                    )
                    return r2, g2

                r_, g_ = lax.cond(has_door, _corr_door, _corr_inaccessible, (r_, g_))
                return r_, g_

            # Carve niche tile at the pocket cell ONE STEP BEYOND the wall:
            # vendor sets ``rm = &levl[xx][yy + dy]`` then ``rm->typ = SCORR``
            # (mklev.c:503-505) or ``rm->typ = CORR`` (mklev.c:524).  The door
            # itself is later stamped at (xx, yy) by dosdoor — these are two
            # DISTINCT cells.  Writing the niche tile at (xx, yy) instead of
            # (xx, yy+dy) left the pocket STONE, so mineralize's all-STONE 3x3
            # scan counted those cells as eligible (over-drawing the rn2(1000)
            # gold/gem stream by 2 cells -> 4 extra rn2(1000) draws on seed 0).
            # Vendor cite: vendor/nle/src/mklev.c:503-505,524.
            ny = jnp.clip(yy + dy_niche, 0, ROWNO - 1)
            new_tile = jnp.where(scorr_branch, jnp.int8(VTILE_SCORR), jnp.int8(VTILE_CORR))
            g = LevelGenState(
                typ=lax.cond(
                    placed,
                    lambda t: t.at[xx, ny].set(new_tile),
                    lambda t: t,
                    g.typ,
                ),
                doormask=g.doormask,
                door_x=g.door_x, door_y=g.door_y,
                doorindex=g.doorindex, smeq=g.smeq, doorct=g.doorct,
            )

            def _place_draws(rg2):
                r_, g_ = rg2
                return lax.cond(scorr_branch, _scorr_path, _corr_path, (r_, g_))

            r, g = lax.cond(placed, _place_draws, lambda rg2: rg2, (r, g))
            return r, g, placed

        r2, g2, placed = lax.cond(
            ~done, _do_attempt,
            lambda rg: (rg[0], rg[1], jnp.bool_(False)), (r, g),
        )
        return (r2, g2, done | placed)

    rng, gs, _ = lax.fori_loop(0, 8, attempt, (rng, gs, jnp.bool_(False)))
    return rng, gs


def make_niches(
    rng: "Isaac64State",
    gs: "LevelGenState",
    rooms: "RoomsBox",
    nroom: jnp.ndarray,
    depth: int = 1,
    noteleport: bool = False,
) -> tuple["Isaac64State", "LevelGenState"]:
    """Vendor mklev.c:548-565 make_niches.

    Draws:
        rnd((nroom>>1)+1)  niche count
        per niche:
            rn2(6)         level-teleport gate (if depth>15 and !noteleport)
            rn2(6)         vampire trapdoor gate (if 5<depth<25)
            (then makeniche cascade)

    The ltptr/vamp flags flip off after first use; we model them as carry
    booleans in the loop.
    """
    nroom_i = nroom.astype(jnp.int32)
    ct_arg = jnp.maximum((nroom_i >> jnp.int32(1)) + jnp.int32(1), jnp.int32(1))
    rng, ct = rnd_jax(rng, ct_arg)

    # Vendor constants: LEVEL_TELEP, TRAPDOOR, NO_TRAP.  We use the trap.h
    # numeric values: LEVEL_TELEP=16, TRAPDOOR=14, NO_TRAP=0.
    NO_TRAP = jnp.int32(0)
    LEVEL_TELEP = jnp.int32(16)
    TRAPDOOR = jnp.int32(14)

    ltptr0 = jnp.bool_((not noteleport) and depth > 15)
    vamp0  = jnp.bool_(5 < depth < 25)

    def body(_, carry):
        r, g, ltptr, vamp = carry
        # rn2(6) ltptr gate — only DRAWN when ltptr is True (vendor:556).
        def draw_lt(rg):
            r_ = rg[0]
            r_, v = rn2_jax(r_, jnp.int32(6))
            return r_, v
        r, lt_roll = lax.cond(
            ltptr, draw_lt, lambda rg: (rg[0], jnp.int32(1)), (r,)
        )
        lt_fire = ltptr & (lt_roll == jnp.int32(0))

        # rn2(6) vamp gate — drawn when vamp AND !lt_fire (vendor:559 in
        # else-if branch).
        def draw_vamp(rg):
            r_ = rg[0]
            r_, v = rn2_jax(r_, jnp.int32(6))
            return r_, v
        r, vamp_roll = lax.cond(
            vamp & ~lt_fire, draw_vamp, lambda rg: (rg[0], jnp.int32(1)), (r,)
        )
        vamp_fire = vamp & ~lt_fire & (vamp_roll == jnp.int32(0))

        trap_type = jnp.where(lt_fire, LEVEL_TELEP,
                    jnp.where(vamp_fire, TRAPDOOR, NO_TRAP))
        r, g = _makeniche(r, g, rooms, nroom_i, trap_type, depth=depth)

        return (r, g, ltptr & ~lt_fire, vamp & ~vamp_fire)

    rng, gs, _, _ = lax.fori_loop(
        0, ct, body, (rng, gs, ltptr0, vamp0)
    )
    return rng, gs


# ---------------------------------------------------------------------------
# Public surface (Phase 4)
# ---------------------------------------------------------------------------
__all_phase4__ = [
    "LevelGenState",
    "RoomsBox",
    "make_empty_level_gen_state",
    "finddpos",
    "okdoor",
    "dodoor",
    "dig_corridor",
    "join",
    "makecorridors",
    "make_niches",
    "DOORMAX",
    "MAXNROFROOMS",
    "COLNO",
    "ROWNO",
]


# ---------------------------------------------------------------------------
# TODO blocks
# ---------------------------------------------------------------------------
# Wave 4:
#   - Secret corridors: some segments should be hidden (TILE_WALL until
#     searched); controlled by dungeon depth and level flags.
#   - Trapdoors / holes between levels placed in corridors (mklev.c).
#   - Vault corridors: short single-segment corridors to vault rooms
#     (vault.c).
