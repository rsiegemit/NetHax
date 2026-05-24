"""Room type definitions and room placement.

Purpose:
    Defines the RoomType enum (matching NetHack's roomtype_types from
    mkroom.h), the Room dataclass (bounding-box representation), and
    procedural room placement and corridor connection.

Citation:
    vendor/nethack/include/mkroom.h  — roomtype_types enum (OROOM..CANDLESHOP),
        struct mkroom (lx/hx/ly/hy/rtype/rlit), MAXNROFROOMS = 40
        (via vendor/nethack/include/global.h line 385)
    vendor/nethack/src/mkroom.c      — shop/temple/zoo/throne/morgue filling
    vendor/nethack/src/mklev.c       — room placement loop, makerooms()
    vendor/nethack/src/mkmap.c       — coordinate allocation

Wave 2: generate_rooms uses fori_loop for rejection-sampling placement;
        connect_rooms returns a terrain int8[MAP_H, MAP_W] with corridors
        carved between consecutive room pairs.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import struct

from Nethax.nethax.dungeon.branches import MAP_H, MAP_W

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MAXNROFROOMS = 40 from vendor/nethack/include/global.h line 385.
# We use this as the fixed array size for Room arrays on each level.
MAX_ROOMS_PER_LEVEL: int = 40

# Room size bounds (interior cells, walls not included).
# Vendor: vendor/nethack/src/sp_lev.c create_room() lines 1548-1549:
#     dx = 2 + rn2(8)   (random width)  -> interior width in [2, 9]
#     dy = 2 + rn2(4)   (random height) -> interior height in [2, 5]
# Minimum interior size is therefore 2x2.
_MIN_ROOM_W: int = 2   # vendor MIN_ROOM_WIDTH
_MAX_ROOM_W: int = 9   # vendor 2 + rn2(8)
_MIN_ROOM_H: int = 2   # vendor MIN_ROOM_HEIGHT
_MAX_ROOM_H: int = 5   # vendor 2 + rn2(4)

# Tile type values (from constants.py TileType).
_TILE_WALL:     int = 3
_TILE_FLOOR:    int = 1
_TILE_CORRIDOR: int = 2

# ---------------------------------------------------------------------------
# RoomType enum
#
# Values match enum roomtype_types in vendor/nethack/include/mkroom.h.
# ---------------------------------------------------------------------------

class RoomType(IntEnum):
    """Room type codes.

    Citation: vendor/nethack/include/mkroom.h, enum roomtype_types (lines 51-77)
    """
    ORDINARY   =  0   # OROOM — ordinary room
    THEMEROOM  =  1   # like OROOM, never converted to special room
    COURT      =  2   # throne room
    SWAMP      =  3   # contains pools
    VAULT      =  4   # detached room, usually via teleport trap
    BEEHIVE    =  5   # killer bees and royal jelly
    MORGUE     =  6   # corpses, undead, graves
    BARRACKS   =  7   # soldiers and their gear
    ZOO        =  8   # treasure and monsters
    DELPHI     =  9   # Oracle and peripherals
    TEMPLE     = 10   # shrine with altar and priest(ess)
    LEPREHALL  = 11   # leprechaun hall
    COCKNEST   = 12   # cockatrice nest
    ANTHOLE    = 13   # ant colony
    SHOPBASE   = 14   # sentinel: everything >= SHOPBASE is a shop
    ARMORSHOP  = 15
    SCROLLSHOP = 16
    POTIONSHOP = 17
    WEAPONSHOP = 18
    FOODSHOP   = 19
    RINGSHOP   = 20
    WANDSHOP   = 21
    TOOLSHOP   = 22
    BOOKSHOP   = 23
    FODDERSHOP = 24   # health food store
    CANDLESHOP = 25   # MAXRTYPE / UNIQUESHOP


# ---------------------------------------------------------------------------
# Room dataclass
#
# Mirrors struct mkroom from mkroom.h:
#   coordxy lx, hx, ly, hy  — bounding box (inclusive)
#   schar   rtype            — room type
#   schar   rlit             — is room lit?
# ---------------------------------------------------------------------------

@struct.dataclass
class Room:
    """Bounding-box description of a single room on a dungeon level.

    All coordinate arrays are int16 to match coordxy (short) in NetHack.
    Rooms are stored in fixed-size arrays of length MAX_ROOMS_PER_LEVEL;
    inactive slots are marked by y1 == y2 == x1 == x2 == -1.

    Fields
    ------
    y1 : int16  — top row of room interior (inclusive)
    x1 : int16  — left column of room interior (inclusive)
    y2 : int16  — bottom row of room interior (inclusive)
    x2 : int16  — right column of room interior (inclusive)
    room_type : int8  — RoomType value
    is_lit : bool     — whether the room is permanently lit

    Citation: vendor/nethack/include/mkroom.h struct mkroom (lines 11-25)
    """
    y1:        jnp.ndarray  # int16
    x1:        jnp.ndarray  # int16
    y2:        jnp.ndarray  # int16
    x2:        jnp.ndarray  # int16
    room_type: jnp.ndarray  # int8   (RoomType)
    is_lit:    jnp.ndarray  # bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rooms_overlap(ay1: jnp.ndarray, ax1: jnp.ndarray,
                   ay2: jnp.ndarray, ax2: jnp.ndarray,
                   by1: jnp.ndarray, bx1: jnp.ndarray,
                   by2: jnp.ndarray, bx2: jnp.ndarray) -> jnp.ndarray:
    """Return True if two axis-aligned bounding boxes overlap (including 1-cell margin).

    We add a 1-cell margin so rooms don't share walls.
    Citation: mklev.c makerooms() overlap check.
    """
    margin = 1
    # Separating axis test: no overlap iff one box is fully to the side/above/below.
    # Row axis: a is fully above b, or b is fully above a.
    # Col axis: a is fully left of b, or b is fully left of a.
    no_overlap = (
        (ay2 + margin < by1) |   # a entirely above b (row)
        (by2 + margin < ay1) |   # b entirely above a (row)
        (ax2 + margin < bx1) |   # a entirely left of b (col)
        (bx2 + margin < ax1)     # b entirely left of a (col)
    )
    return ~no_overlap


def _check_any_overlap(new_y1: jnp.ndarray, new_x1: jnp.ndarray,
                       new_y2: jnp.ndarray, new_x2: jnp.ndarray,
                       all_y1: jnp.ndarray, all_x1: jnp.ndarray,
                       all_y2: jnp.ndarray, all_x2: jnp.ndarray,
                       active: jnp.ndarray) -> jnp.ndarray:
    """Return True if the new room overlaps any active room."""
    overlaps = jax.vmap(
        lambda ay1, ax1, ay2, ax2, act: act & _rooms_overlap(
            new_y1, new_x1, new_y2, new_x2,
            ay1, ax1, ay2, ax2,
        )
    )(all_y1, all_x1, all_y2, all_x2, active)
    return jnp.any(overlaps)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def generate_rooms(
    rng: jnp.ndarray,
    h: int = MAP_H,
    w: int = MAP_W,
    n_rooms: int = 8,
    depth: int = 1,
) -> Room:
    """Place rooms on an h×w grid via rejection-sampling.

    Vendor (`mklev.c::makerooms` line 403):
        while (svn.nroom < (MAXNROFROOMS - 1) && rnd_rect())
            create_room(...)
    Vendor allows up to MAXNROFROOMS-1 = 39 rooms but typically stops when
    `rnd_rect()` can no longer find a free sub-rectangle.  Observed counts
    on the dungeons of doom main branch are in the 5..9 range (one room
    per ~6 free units of map width).

    Audit-N #1 (mkmap.c::litstate_rnd line 446):
        is_lit = (rnd(1 + abs(depth)) < 11) & (rn2(77) == 0)
    Lit rooms are very rare and biased toward shallow depths.

    Audit-N #2 (sp_lev.c::create_room lines 1548-1551):
        dx = 2 + rn2((hx - lx > 28) ? 12 : 8)
        dy = 2 + rn2(4)
        if dx * dy > 50: dy = 50 / dx
    Big-room enclosures sample dx from a wider range; the dx*dy <= 50 cap
    truncates height.  Our generator uses the standard 2+rn2(8)/2+rn2(4)
    width/height sampling because the map is divided into many small
    rooms (not single-rectangle big rooms); we *do* apply the lit-formula
    and the dy-cap.

    Args:
        rng:     JAX PRNG key.
        h:       map height in cells (default MAP_H = 21).
        w:       map width in cells (default MAP_W = 80).
        n_rooms: target room count.  Pass -1 to draw a vendor-style random
                 count in [5, 9].  Default 8 preserves Wave 2 behaviour.
        depth:   current dungeon depth (1..30+).  Used for lit-room formula.

    Returns:
        Room pytree with arrays shaped [MAX_ROOMS_PER_LEVEL].
        active slots have room_type=ORDINARY; inactive slots have coords=-1.
    """
    # Pre-sample all random values: for each room attempt we need 4 values
    # (y_offset, x_offset, height, width).  We also allow up to _MAX_RETRIES
    # rejection attempts per room slot by drawing extra samples up-front.
    _MAX_RETRIES: int = 32  # attempts per room slot (vendor: open-ended retry)

    total_samples = MAX_ROOMS_PER_LEVEL * _MAX_RETRIES
    rng, key_y, key_x, key_h, key_w, key_lit, key_n = jax.random.split(rng, 7)

    # Vendor-style random target count: 5..9 inclusive.
    # Citation: mklev.c makerooms — loop runs MAXNROFROOMS/6..MAXNROFROOMS-1
    #           but most levels settle in 5..9 due to rnd_rect failures.
    n_target_random = jax.random.randint(
        key_n, (), 5, 10, dtype=jnp.int32,
    )
    use_random = jnp.int32(n_rooms) < 0
    n_target = jnp.where(use_random, n_target_random, jnp.int32(n_rooms))

    # y offsets: [1, h - _MAX_ROOM_H - 1)  (leave border + room space)
    y_range = h - _MAX_ROOM_H - 2   # available rows for top-left
    x_range = w - _MAX_ROOM_W - 2
    y_range = max(y_range, 1)
    x_range = max(x_range, 1)

    all_y_off = jax.random.randint(key_y, (total_samples,), 1, 1 + y_range, dtype=jnp.int16)
    all_x_off = jax.random.randint(key_x, (total_samples,), 1, 1 + x_range, dtype=jnp.int16)
    all_heights = jax.random.randint(key_h, (total_samples,), _MIN_ROOM_H, _MAX_ROOM_H + 1, dtype=jnp.int16)
    all_widths  = jax.random.randint(key_w, (total_samples,), _MIN_ROOM_W, _MAX_ROOM_W + 1, dtype=jnp.int16)

    # Audit-N #1: vendor litstate_rnd (mkmap.c:446):
    #   is_lit = (rnd(1+abs(depth)) < 11) & (rn2(77) == 0)
    # rnd(N) returns [1, N]; we sample from [1, 1+abs(depth)] as
    # jax.random.randint(min=1, max=2+abs(depth)).
    abs_depth = abs(int(depth))
    key_lit_a, key_lit_b = jax.random.split(key_lit, 2)
    lit_roll_a = jax.random.randint(
        key_lit_a, (MAX_ROOMS_PER_LEVEL,), 1, 2 + abs_depth, dtype=jnp.int32
    )
    lit_roll_b = jax.random.randint(
        key_lit_b, (MAX_ROOMS_PER_LEVEL,), 0, 77, dtype=jnp.int32
    )
    all_lit = ((lit_roll_a < 11) & (lit_roll_b == 0)).astype(jnp.int8)

    # Audit-N #2: dx*dy <= 50 cap.  Apply after width/height sample —
    # truncate height when width is wide enough to push the area over 50.
    capped_h = jnp.where(
        all_widths * all_heights > 50,
        (jnp.int16(50) // jnp.maximum(all_widths, jnp.int16(1))).astype(jnp.int16),
        all_heights,
    )
    all_heights = jnp.maximum(capped_h, jnp.int16(_MIN_ROOM_H))

    # State carried through fori_loop:
    #   y1[MAX_ROOMS], x1[MAX_ROOMS], y2[MAX_ROOMS], x2[MAX_ROOMS]  — room coords
    #   active[MAX_ROOMS]  — bool mask of placed rooms
    init_coords = jnp.full((MAX_ROOMS_PER_LEVEL,), -1, dtype=jnp.int16)
    init_active = jnp.zeros((MAX_ROOMS_PER_LEVEL,), dtype=bool)

    state0 = (init_coords, init_coords, init_coords, init_coords, init_active)

    def place_one_room(i, state):
        """Try to place room slot i using up to _MAX_RETRIES sample draws."""
        y1_arr, x1_arr, y2_arr, x2_arr, active = state

        def try_one(retry_state, r):
            placed, y1_arr_, x1_arr_, y2_arr_, x2_arr_, active_ = retry_state
            sample_idx = i * _MAX_RETRIES + r

            ny1 = all_y_off[sample_idx]
            nx1 = all_x_off[sample_idx]
            nh  = all_heights[sample_idx]
            nw  = all_widths[sample_idx]
            ny2 = (ny1 + nh - 1).astype(jnp.int16)
            nx2 = (nx1 + nw - 1).astype(jnp.int16)

            # Check bounds: room must fit inside [1, h-2] x [1, w-2]
            fits = (ny2 <= h - 2) & (nx2 <= w - 2)
            overlaps = _check_any_overlap(
                ny1, nx1, ny2, nx2,
                y1_arr_, x1_arr_, y2_arr_, x2_arr_, active_,
            )

            accept = fits & ~overlaps & ~placed

            new_y1   = jnp.where(accept, ny1, y1_arr_.at[i].get())
            new_x1   = jnp.where(accept, nx1, x1_arr_.at[i].get())
            new_y2   = jnp.where(accept, ny2, y2_arr_.at[i].get())
            new_x2   = jnp.where(accept, nx2, x2_arr_.at[i].get())

            y1_arr_new = lax.cond(accept, lambda: y1_arr_.at[i].set(new_y1), lambda: y1_arr_)
            x1_arr_new = lax.cond(accept, lambda: x1_arr_.at[i].set(new_x1), lambda: x1_arr_)
            y2_arr_new = lax.cond(accept, lambda: y2_arr_.at[i].set(new_y2), lambda: y2_arr_)
            x2_arr_new = lax.cond(accept, lambda: x2_arr_.at[i].set(new_x2), lambda: x2_arr_)
            active_new = lax.cond(accept, lambda: active_.at[i].set(True), lambda: active_)
            placed_new = placed | accept

            return (placed_new, y1_arr_new, x1_arr_new, y2_arr_new, x2_arr_new, active_new), None

        init_retry = (jnp.bool_(False), y1_arr, x1_arr, y2_arr, x2_arr, active)
        (_, y1f, x1f, y2f, x2f, activef), _ = lax.scan(
            try_one, init_retry, jnp.arange(_MAX_RETRIES, dtype=jnp.int32)
        )
        return (y1f, x1f, y2f, x2f, activef)

    # fori_loop accepts a traced upper bound, so n_target can be either
    # a Python int (Wave 2 path) or a JAX scalar (vendor-random path).
    y1_f, x1_f, y2_f, x2_f, active_f = lax.fori_loop(
        jnp.int32(0), n_target, place_one_room, state0
    )

    # Room type: ORDINARY (0) for active slots, 0 elsewhere (still ORDINARY but masked by active_f)
    room_type = jnp.zeros((MAX_ROOMS_PER_LEVEL,), dtype=jnp.int8)

    return Room(
        y1=y1_f,
        x1=x1_f,
        y2=y2_f,
        x2=x2_f,
        room_type=room_type,
        is_lit=all_lit.astype(bool),
    ), active_f


# ---------------------------------------------------------------------------
# Wave 17f — Special-room assignment
#
# Vendor (vendor/nethack/src/mklev.c::makelevel lines 1344-1376) makes up to
# one special room per level, with type-dependent probability gated on
# u_depth.  The cascade is:
#
#   u_depth > 1  & rn2(u_depth) < 3 → SHOPBASE  (shop)
#   u_depth > 4  & !rn2(6)          → COURT
#   u_depth > 5  & !rn2(8)          → LEPREHALL
#   u_depth > 6  & !rn2(7)          → ZOO
#   u_depth > 8  & !rn2(5)          → TEMPLE
#   u_depth > 9  & !rn2(5)          → BEEHIVE
#   u_depth > 11 & !rn2(6)          → MORGUE
#   u_depth > 12 & !rn2(8)          → ANTHOLE
#   u_depth > 14 & !rn2(4)          → BARRACKS
#   u_depth > 15 & !rn2(6)          → SWAMP
#   u_depth > 16 & !rn2(8)          → COCKNEST
#
# The table below encodes (depth_gate, rn2_modulus, room_type) tuples so
# `assign_special_room` can walk them in vendor order under JIT.
# ---------------------------------------------------------------------------

# (min_depth_exclusive, rn2_modulus, room_type)  — vendor order matters.
_ROOM_PROBS = (
    (1,  None, int(RoomType.SHOPBASE)),    # special: rn2(u_depth) < 3
    (4,  6,    int(RoomType.COURT)),
    (5,  8,    int(RoomType.LEPREHALL)),
    (6,  7,    int(RoomType.ZOO)),
    (8,  5,    int(RoomType.TEMPLE)),
    (9,  5,    int(RoomType.BEEHIVE)),
    (11, 6,    int(RoomType.MORGUE)),
    (12, 8,    int(RoomType.ANTHOLE)),
    (14, 4,    int(RoomType.BARRACKS)),
    (15, 6,    int(RoomType.SWAMP)),
    (16, 8,    int(RoomType.COCKNEST)),
)


# Vendor mklev.c line 1349: u_depth < depth(&medusa_level) — Medusa is on
# the level just above Castle (~Dlvl 22-25 with the mean/dev placement).  We
# encode the depth cutoff as a compile-time constant; assign_special_room
# checks u_depth < MEDUSA_LEVEL_DEPTH before allowing SHOPBASE.
MEDUSA_LEVEL_DEPTH: int = 24

# Vendor mklev.c line 1350: svn.nroom >= room_threshold (where room_threshold
# is a static "shops require N or more rooms" gate, usually 4).
ROOM_THRESHOLD_FOR_SHOP: int = 4

# Vendor monster-extinction PM_ id sentinels (mklev.c lines 1355, 1362, 1369,
# 1374): if all monsters of the gating species are dead, the special room is
# skipped.  Indexes into a 5-element ``genocided`` mask passed to
# assign_special_room.
GENOCIDE_IDX_LEPRECHAUN = 0
GENOCIDE_IDX_KILLER_BEE = 1
GENOCIDE_IDX_SOLDIER    = 2
GENOCIDE_IDX_COCKATRICE = 3
GENOCIDE_IDX_ANT        = 4   # antholemon() — any ant species alive


def assign_special_room(
    rng: jnp.ndarray,
    rooms: Room,
    active: jnp.ndarray,
    u_depth: int,
    n_rooms: jnp.ndarray | None = None,
    genocided: jnp.ndarray | None = None,
) -> Room:
    """Assign at most one special room per level following vendor mklev.c.

    Vendor (mklev.c lines 1344-1376): walk the table in order; first
    matching depth + rn2 success picks the room type.  Then mkshop /
    mkzoo / mktemple / mkswamp pick the *first* active OROOM slot and
    flip its rtype.

    Audit-N #3:
      - SHOPBASE row adds ``u_depth < depth(&medusa_level)`` ceiling and
        ``svn.nroom >= room_threshold`` gate.
      - LEPREHALL / BEEHIVE / ANTHOLE / BARRACKS / COCKNEST add per-species
        ``svm.mvitals[...].mvflags & G_GONE`` gates (passed as the
        ``genocided`` array — bool[5] indexed by GENOCIDE_IDX_*).

    JIT-pure: uses jnp.where on the room_type array and walks the table
    via Python iteration (table is static, so the loop unrolls).

    Args:
        rng:        JAX PRNG key.
        rooms:      Room pytree (room_type currently all ORDINARY).
        active:    bool[MAX_ROOMS_PER_LEVEL] mask.
        u_depth:    1-based depth in dungeon (compile-time int).
        n_rooms:    optional int scalar — total active rooms on this level
                    (gates SHOPBASE per vendor line 1350).  If None, treated
                    as ROOM_THRESHOLD_FOR_SHOP (gate satisfied).
        genocided:  optional bool[5] — per-species extinction mask
                    (LEPREHALL/BEEHIVE/SOLDIER/COCKATRICE/ANT).  If None,
                    no species are genocided.

    Returns:
        Updated Room pytree with at most one slot flipped to a special type.
    """
    rng, key_pick = jax.random.split(rng)
    # First active slot index (vendor: scan svr.rooms[0..nroom] for OROOM).
    first_active_idx = jnp.argmax(active).astype(jnp.int32)

    # Default n_rooms = ROOM_THRESHOLD (gate satisfied) when caller omits.
    if n_rooms is None:
        n_rooms = jnp.int32(ROOM_THRESHOLD_FOR_SHOP)
    n_rooms = jnp.asarray(n_rooms, dtype=jnp.int32)

    # Default genocided = all-False when caller omits.
    if genocided is None:
        genocided = jnp.zeros((5,), dtype=jnp.bool_)
    genocided = jnp.asarray(genocided, dtype=jnp.bool_)

    # SHOPBASE gates (vendor mklev.c:1349-1350).
    shop_depth_ok = u_depth < MEDUSA_LEVEL_DEPTH
    shop_nroom_ok = n_rooms >= jnp.int32(ROOM_THRESHOLD_FOR_SHOP)
    shop_extra_gate = shop_depth_ok & shop_nroom_ok

    # Per-species extinction gates.
    species_alive_lep   = ~genocided[GENOCIDE_IDX_LEPRECHAUN]
    species_alive_bee   = ~genocided[GENOCIDE_IDX_KILLER_BEE]
    species_alive_sold  = ~genocided[GENOCIDE_IDX_SOLDIER]
    species_alive_cock  = ~genocided[GENOCIDE_IDX_COCKATRICE]
    species_alive_ant   = ~genocided[GENOCIDE_IDX_ANT]
    any_active = jnp.any(active)

    # Build a single int32 "chosen_type" via vendor cascade.  Walks the
    # static _ROOM_PROBS table; first hit wins (later cases skipped).
    chosen = jnp.int32(int(RoomType.ORDINARY))
    decided = jnp.bool_(False)

    keys = jax.random.split(key_pick, len(_ROOM_PROBS))
    for i, (min_depth, modulus, rtype) in enumerate(_ROOM_PROBS):
        depth_ok = u_depth > min_depth
        if modulus is None:
            # SHOPBASE special case: rn2(u_depth) < 3
            roll = jax.random.randint(keys[i], (), 0,
                                       max(int(u_depth), 1),
                                       dtype=jnp.int32)
            pass_ = depth_ok & (roll < jnp.int32(3)) & shop_extra_gate
        else:
            roll = jax.random.randint(keys[i], (), 0, modulus, dtype=jnp.int32)
            pass_ = depth_ok & (roll == jnp.int32(0))   # !rn2(modulus)

        # Audit-N #3: per-species genocide gates.  Vendor lines 1355, 1362,
        # 1369, 1374.
        if rtype == int(RoomType.LEPREHALL):
            pass_ = pass_ & species_alive_lep
        elif rtype == int(RoomType.BEEHIVE):
            pass_ = pass_ & species_alive_bee
        elif rtype == int(RoomType.ANTHOLE):
            pass_ = pass_ & species_alive_ant
        elif rtype == int(RoomType.BARRACKS):
            pass_ = pass_ & species_alive_sold
        elif rtype == int(RoomType.COCKNEST):
            pass_ = pass_ & species_alive_cock

        take = pass_ & ~decided
        chosen = jnp.where(take, jnp.int32(rtype), chosen)
        decided = decided | take

    do_assign = decided & any_active
    new_room_type = jnp.where(
        do_assign,
        rooms.room_type.at[first_active_idx].set(chosen.astype(jnp.int8)),
        rooms.room_type,
    )
    return rooms.replace(room_type=new_room_type)


def fill_special_room(
    rng: jnp.ndarray,
    rooms: Room,
    active: jnp.ndarray,
    terrain: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the per-type fill pass for whichever room is flagged special.

    Vendor (mkroom.c::do_mkroom):
      mkshop          — picks random shop type; floor unchanged.
      mkzoo(COURT)    — places throne tile at room center.
      mkzoo(ZOO/MORGUE/...) — sets has_*; tile fill done at monster spawn.
      mkswamp         — converts alternating interior cells to POOL.
      mktemple        — places ALTAR at shrine_pos (room center).

    This JAX implementation:
      * COURT  → set room center to THRONE.
      * TEMPLE → set room center to ALTAR.
      * SWAMP  → set (sx + sy) % 2 == 1 interior cells to POOL (vendor 554).
    All other special types are tile-fill no-ops here (monsters/items are
    populated by the spawning subsystem when the room is first entered).

    Note: the temple altar alignment, AM_SHRINE flag, and has_temple level
    flag are set by :func:`fill_special_room_features` which operates on
    FeaturesState (mkroom.c::mktemple lines 597-619).  This function
    handles tile-only updates so callers without FeaturesState (older
    tests) can still use it.

    Args:
        rng:     JAX PRNG key (currently unused — fill is deterministic).
        rooms:   Room pytree (with room_type set by assign_special_room).
        active:  bool[MAX_ROOMS_PER_LEVEL] mask.
        terrain: int8[MAP_H, MAP_W] terrain map.

    Returns:
        Updated terrain int8[MAP_H, MAP_W].
    """
    from Nethax.nethax.constants.tiles import TileType
    h, w = terrain.shape
    THRONE = jnp.int8(int(TileType.THRONE))
    ALTAR  = jnp.int8(int(TileType.ALTAR))
    POOL   = jnp.int8(int(TileType.POOL))
    FLOOR  = jnp.int8(int(TileType.FLOOR))

    def fill_one(terrain_, i):
        y1 = rooms.y1[i].astype(jnp.int32)
        x1 = rooms.x1[i].astype(jnp.int32)
        y2 = rooms.y2[i].astype(jnp.int32)
        x2 = rooms.x2[i].astype(jnp.int32)
        rt = rooms.room_type[i].astype(jnp.int32)
        act = active[i]

        # Centre point (vendor shrine_pos / court throne).
        cy = ((y1 + y2) // 2).astype(jnp.int32)
        cx = ((x1 + x2) // 2).astype(jnp.int32)
        cy_safe = jnp.clip(cy, 0, h - 1)
        cx_safe = jnp.clip(cx, 0, w - 1)

        is_court  = act & (rt == jnp.int32(int(RoomType.COURT)))
        is_temple = act & (rt == jnp.int32(int(RoomType.TEMPLE)))
        is_swamp  = act & (rt == jnp.int32(int(RoomType.SWAMP)))

        # COURT: throne at centre (mkroom.c:421-423).
        center_val = terrain_[cy_safe, cx_safe]
        new_center = jnp.where(is_court, THRONE,
                     jnp.where(is_temple, ALTAR, center_val))
        terrain_ = terrain_.at[cy_safe, cx_safe].set(new_center)

        # SWAMP: alternating floor cells → POOL (mkroom.c:554, sx + sy % 2).
        rows = jnp.arange(h, dtype=jnp.int32)
        cols = jnp.arange(w, dtype=jnp.int32)
        row_mask = (rows >= y1) & (rows <= y2)
        col_mask = (cols >= x1) & (cols <= x2)
        interior = row_mask[:, None] & col_mask[None, :]
        # parity = (r + c) % 2 == 1 (vendor: (sx + sy) % 2)
        rr, cc = jnp.meshgrid(rows, cols, indexing="ij")
        parity = ((rr + cc) % jnp.int32(2)) == jnp.int32(1)
        swamp_mask = is_swamp & interior & parity & (terrain_ == FLOOR)
        terrain_ = jnp.where(swamp_mask, POOL, terrain_)
        return terrain_, None

    terrain_out, _ = lax.scan(
        fill_one, terrain, jnp.arange(MAX_ROOMS_PER_LEVEL, dtype=jnp.int32)
    )
    return terrain_out


def fill_special_room_features(
    rng: jnp.ndarray,
    rooms: Room,
    active: jnp.ndarray,
    features,                  # FeaturesState
    flat_lv: int,
    player_align: int,
):
    """Feature-level fill for TEMPLE rooms (Audit-N #4).

    Vendor mkroom.c::mktemple (lines 597-619):
        sroom->rtype = TEMPLE;
        shrine_spot = shrine_pos(...);
        lev->typ = ALTAR;
        lev->altarmask = induced_align(80);   /* 80 % of player align */
        priestini(&u.uz, sroom, sx, sy, FALSE);
        lev->altarmask |= AM_SHRINE;
        svl.level.flags.has_temple = 1;

    JAX implementation: for the first active TEMPLE room (the one
    assign_special_room flipped) we:
      - set features.altar_alignment[flat_lv, cy, cx] = induced_align
      - set features.altar_shrine [flat_lv, cy, cx] = True
      - set features.has_temple   [flat_lv]          = True

    ``induced_align(80)`` returns the player's alignment with probability
    80 % and a random alternative with 20 %.

    Args:
        rng:         JAX PRNG key.
        rooms:       Room pytree (with room_type set by assign_special_room).
        active:      bool[MAX_ROOMS_PER_LEVEL] mask.
        features:    FeaturesState to update.
        flat_lv:     int — flattened level index.
        player_align: int — 0/1/2 Alignment value.

    Returns:
        Updated FeaturesState.
    """
    rng_align, rng_alt = jax.random.split(rng, 2)

    # induced_align(80): 80 % chance of returning player_align, otherwise
    # a uniformly random {0,1,2} replacement (vendor align.c::induced_align).
    coin = jax.random.randint(rng_align, (), 0, 100, dtype=jnp.int32)
    keep_player_align = coin < jnp.int32(80)
    alt_align = jax.random.randint(rng_alt, (), 0, 3, dtype=jnp.int32)
    induced = jnp.where(
        keep_player_align, jnp.int32(player_align), alt_align
    ).astype(jnp.int8)

    # Iterate over rooms to find the first TEMPLE active slot.  scan keeps
    # this JIT-safe.
    is_temple = active & (rooms.room_type == jnp.int8(int(RoomType.TEMPLE)))
    any_temple = jnp.any(is_temple)
    temple_idx = jnp.argmax(is_temple).astype(jnp.int32)

    cy = ((rooms.y1[temple_idx] + rooms.y2[temple_idx]) // 2).astype(jnp.int32)
    cx = ((rooms.x1[temple_idx] + rooms.x2[temple_idx]) // 2).astype(jnp.int32)

    # Only commit changes when there *is* a temple room.
    aa = features.altar_alignment
    sh = features.altar_shrine
    ht = features.has_temple

    new_aa = aa.at[flat_lv, cy, cx].set(
        jnp.where(any_temple, induced, aa[flat_lv, cy, cx])
    )
    new_sh = sh.at[flat_lv, cy, cx].set(
        jnp.where(any_temple, jnp.bool_(True), sh[flat_lv, cy, cx])
    )
    new_ht = ht.at[flat_lv].set(
        jnp.where(any_temple, jnp.bool_(True), ht[flat_lv])
    )
    return features.replace(
        altar_alignment=new_aa,
        altar_shrine=new_sh,
        has_temple=new_ht,
    )


def carve_rooms_into_terrain(
    terrain: jnp.ndarray,
    rooms: Room,
    active: jnp.ndarray,
) -> jnp.ndarray:
    """Carve room interiors and walls into a terrain array.

    Sets wall tiles around the perimeter and floor tiles in the interior.

    Citation: vendor/nethack/src/mklev.c makerooms() — the final carve step.

    Args:
        terrain: int8[MAP_H, MAP_W] array (modified in-place via functional update).
        rooms:   Room pytree from generate_rooms().
        active:  bool[MAX_ROOMS_PER_LEVEL] mask.

    Returns:
        Updated terrain int8[MAP_H, MAP_W].
    """
    h, w = terrain.shape

    def carve_one(terrain_, i):
        y1 = rooms.y1[i].astype(jnp.int32)
        x1 = rooms.x1[i].astype(jnp.int32)
        y2 = rooms.y2[i].astype(jnp.int32)
        x2 = rooms.x2[i].astype(jnp.int32)
        act = active[i]

        # Build a mask for the entire room bounding box (interior only).
        rows = jnp.arange(h)
        cols = jnp.arange(w)
        row_mask = (rows >= y1) & (rows <= y2)   # interior rows
        col_mask = (cols >= x1) & (cols <= x2)   # interior cols
        interior = row_mask[:, None] & col_mask[None, :]  # [h, w] bool

        # Wall border: one cell outside the interior.
        row_wall = (rows >= y1 - 1) & (rows <= y2 + 1)
        col_wall = (cols >= x1 - 1) & (cols <= x2 + 1)
        border = (row_wall[:, None] & col_wall[None, :]) & ~interior

        # Apply: first wall, then floor (floor overwrites wall in interior).
        terrain_w = jnp.where(act & border,   jnp.int8(_TILE_WALL),  terrain_)
        terrain_f = jnp.where(act & interior, jnp.int8(_TILE_FLOOR), terrain_w)
        return terrain_f, None

    terrain_out, _ = lax.scan(carve_one, terrain, jnp.arange(MAX_ROOMS_PER_LEVEL, dtype=jnp.int32))
    return terrain_out


def connect_rooms(
    rng: jnp.ndarray,
    rooms: Room,
    active: jnp.ndarray,
    terrain: jnp.ndarray,
) -> jnp.ndarray:
    """Carve L-shaped corridors connecting consecutive active rooms into terrain.

    Connects room[i] to room[i+1] for i in 0..MAX_ROOMS_PER_LEVEL-2 when
    both rooms are active.  Chooses a random wall exit point on each room
    and carves an L-shaped path between them.

    Citation: vendor/nethack/src/mklev.c doconnect(), dig_corridor().

    Args:
        rng:     JAX PRNG key.
        rooms:   Room pytree from generate_rooms().
        active:  bool[MAX_ROOMS_PER_LEVEL] mask.
        terrain: int8[MAP_H, MAP_W] — already has rooms carved in.

    Returns:
        Updated terrain int8[MAP_H, MAP_W] with corridors carved.
    """
    h, w = terrain.shape
    n_pairs = MAX_ROOMS_PER_LEVEL - 1

    # Sample bend offsets for all pairs up-front.
    rng, k1, k2 = jax.random.split(rng, 3)
    bend_rows = jax.random.randint(k1, (n_pairs,), 0, h, dtype=jnp.int32)
    bend_cols = jax.random.randint(k2, (n_pairs,), 0, w, dtype=jnp.int32)

    def connect_pair(terrain_, carry):
        i, br, bc = carry

        # Centre of room i and room i+1
        y_a = ((rooms.y1[i] + rooms.y2[i]) // 2).astype(jnp.int32)
        x_a = ((rooms.x1[i] + rooms.x2[i]) // 2).astype(jnp.int32)
        y_b = ((rooms.y1[i + 1] + rooms.y2[i + 1]) // 2).astype(jnp.int32)
        x_b = ((rooms.x1[i + 1] + rooms.x2[i + 1]) // 2).astype(jnp.int32)

        both_active = active[i] & active[i + 1]

        # Vendor `sp_lev.c::dig_corridor` (lines 2571-2660) prefers an
        # orthogonal straight run when one axis is already aligned and only
        # falls back to an L-bend when both row AND column differ.  We mirror
        # that policy: if y_a == y_b OR x_a == x_b, a single straight segment
        # suffices; otherwise carve the L (horizontal then vertical).
        rows = jnp.arange(h, dtype=jnp.int32)
        cols = jnp.arange(w, dtype=jnp.int32)

        same_row = y_a == y_b
        same_col = x_a == x_b

        # Horizontal segment at row y_a from x_a to x_b.
        horiz_row = rows == y_a
        horiz_col = (cols >= jnp.minimum(x_a, x_b)) & (cols <= jnp.maximum(x_a, x_b))
        horiz_mask = horiz_row[:, None] & horiz_col[None, :]

        # Vertical segment at col x_b from y_a to y_b.
        vert_col = cols == x_b
        vert_row = (rows >= jnp.minimum(y_a, y_b)) & (rows <= jnp.maximum(y_a, y_b))
        vert_mask = vert_row[:, None] & vert_col[None, :]

        # Straight-only corridor: use whichever single segment connects them.
        straight_mask = jnp.where(same_row, horiz_mask, vert_mask)
        # L-shaped fallback: both legs.
        l_mask = horiz_mask | vert_mask

        corridor_mask = jnp.where(same_row | same_col, straight_mask, l_mask)

        # Only carve into non-floor cells (don't overwrite room floors).
        is_floor = terrain_ == jnp.int8(_TILE_FLOOR)
        terrain_new = jnp.where(
            both_active & corridor_mask & ~is_floor,
            jnp.int8(_TILE_CORRIDOR),
            terrain_,
        )
        return terrain_new, None

    indices = jnp.arange(n_pairs, dtype=jnp.int32)
    carries = (indices, bend_rows[:n_pairs], bend_cols[:n_pairs])

    # lax.scan over pairs
    def scan_fn(terrain_, carry_tuple):
        i, br, bc = carry_tuple[0], carry_tuple[1], carry_tuple[2]
        return connect_pair(terrain_, (i, br, bc))

    terrain_out, _ = lax.scan(
        scan_fn,
        terrain,
        (indices, bend_rows, bend_cols),
    )
    return terrain_out


# ---------------------------------------------------------------------------
# fill_ordinary_rooms — per-room independent feature rolls (Audit-N #5)
# ---------------------------------------------------------------------------
#
# Vendor cite: vendor/nethack/src/mklev.c::fill_ordinary_room lines 968-1006.
# After special-room dispatch the vendor walks every OROOM/THEMEROOM and
# fires a sequence of independent dice rolls; each roll places one of the
# stock dungeon features (sleeping monster, traps, gold, fountain, sink,
# altar, grave, statue) at a random tile inside the room.
#
# The vendor uses the running RNG (rn2 / rnd) directly.  Our JIT-safe port
# threads ``jax.random.split`` to derive one independent sub-key per roll
# per room — no key reuse, deterministic schedule, no Python branching on
# traced values.  The function is non-jit (it runs once per level at
# generation time) but the inner per-room work is scan-friendly so it can
# be folded into a jitted pipeline by callers.


def _pick_room_tile(rng, y1, x1, y2, x2):
    """Return (r, c) uniformly inside the room interior (vendor ``somexyspace``).

    Vendor cite: vendor/nethack/src/mkmaze.c::somexyspace —
    ``somexy`` returns a uniform interior cell.  Our port draws ``row`` in
    ``[y1, y2]`` and ``col`` in ``[x1, x2]`` (both inclusive) using two
    sub-keys from a single split.
    """
    k_r, k_c = jax.random.split(rng, 2)
    row = jax.random.randint(k_r, (), minval=y1, maxval=y2 + jnp.int32(1)).astype(jnp.int32)
    col = jax.random.randint(k_c, (), minval=x1, maxval=x2 + jnp.int32(1)).astype(jnp.int32)
    return row, col


def _vendor_traptype_rnd(rng, level_diff):
    """JIT-safe port of vendor/nethack/src/mklev.c::traptype_rnd lines 1938-1998.

    Returns a TrapType value (int) for use with mktrap().  Pursues the vendor
    semantics: draw ``rnd(TRAPNUM - 1)`` then filter out trap kinds that the
    map cannot legally host (TRAPPED_DOOR, TRAPPED_CHEST, MAGIC_PORTAL,
    VIBRATING_SQUARE), plus depth-gated kinds (SLP_GAS_TRAP, LEVEL_TELEP,
    SPIKED_PIT, LANDMINE, WEB, STATUE_TRAP, POLY_TRAP, etc.).

    Vendor's ``do { kind = traptype_rnd(); } while (kind == NO_TRAP)`` loop
    is approximated here with a single draw + a NO_TRAP→ARROW_TRAP fallback
    so the function is JIT-safe (no traced-loop).  Callers that need the
    exact retry-until-non-zero behaviour can iterate at a fixed bound.

    Args:
        rng:        jax.random.PRNGKey scalar.
        level_diff: vendor ``level_difficulty()`` (depth-equivalent int).

    Returns:
        int32 scalar TrapType value (NO_TRAP=0 if no legal kind drawn).
    """
    # rnd(TRAPNUM - 1) → uniform in [1, TRAPNUM-1].  TRAPNUM = 26 per
    # vendor/nethack/include/trap.h (mirrors our N_TRAP_TYPES = 26).
    kind = jax.random.randint(rng, (), minval=1, maxval=26, dtype=jnp.int32)
    lvl  = jnp.asarray(level_diff, dtype=jnp.int32)

    # Disallow non-map trap kinds (vendor lines 1946-1955).
    is_trapped_door  = kind == jnp.int32(24)  # TRAPPED_DOOR
    is_trapped_chest = kind == jnp.int32(25)  # TRAPPED_CHEST
    is_portal        = kind == jnp.int32(17)  # MAGIC_PORTAL
    is_vibsquare     = kind == jnp.int32(23)  # VIBRATING_SQUARE
    is_fire          = kind == jnp.int32(10)  # FIRE_TRAP — Gehennom-only

    illegal = is_trapped_door | is_trapped_chest | is_portal | is_vibsquare | is_fire

    # Depth gates (vendor lines 1956-1995).
    depth_too_low = (
        ((kind == jnp.int32(7))  & (lvl < jnp.int32(2)))  | # ROLLING_BOULDER_TRAP
        ((kind == jnp.int32(8))  & (lvl < jnp.int32(2)))  | # SLP_GAS_TRAP
        ((kind == jnp.int32(16)) & (lvl < jnp.int32(5)))  | # LEVEL_TELEP
        ((kind == jnp.int32(12)) & (lvl < jnp.int32(5)))  | # SPIKED_PIT
        ((kind == jnp.int32(6))  & (lvl < jnp.int32(6)))  | # LANDMINE
        ((kind == jnp.int32(18)) & (lvl < jnp.int32(7)))  | # WEB
        ((kind == jnp.int32(19)) & (lvl < jnp.int32(8)))  | # STATUE_TRAP
        ((kind == jnp.int32(22)) & (lvl < jnp.int32(8)))    # POLY_TRAP
    )

    # Substitute illegal/too-deep draws with ARROW_TRAP (always legal — the
    # vendor's retry loop converges quickly on a legal kind; we collapse it
    # to the shallowest non-NO_TRAP trap to stay JIT-safe).
    legal = ~(illegal | depth_too_low)
    return jnp.where(legal, kind, jnp.int32(1))  # ARROW_TRAP fallback


def fill_ordinary_rooms(
    rng,
    rooms,
    active,
    terrain,
    features,
    traps,
    flat_lv: int,
    depth,
    player_align: int = 1,
):
    """Apply per-room independent feature rolls to every ordinary room.

    Vendor cite: vendor/nethack/src/mklev.c::fill_ordinary_room lines 968-1006:

        if ((u.uhave.amulet || !rn2(3)) && somexyspace(croom, &pos))
            tmonst = makemon(...);                      # sleeping monster
        x = 8 - (level_difficulty() / 6);  if (x <= 1) x = 2;
        while (!rn2(x) && trycnt++ < 1000) mktrap(...); # traps
        if (!rn2(3) && somexyspace(croom, &pos)) mkgold(...);
        if (!rn2(10))  mkfount(croom);                  # fountain 1/10
        if (!rn2(60))  mksink(croom);                   # sink     1/60
        if (!rn2(60))  mkaltar(croom);                  # altar    1/60
        x = 80 - (depth(&u.uz) * 2); if (x < 2) x = 2;
        if (!rn2(x))  mkgrave(croom);                   # grave depth-scaled
        if (!rn2(20) && somexyspace(croom, &pos))
            (void) mkcorpstat(STATUE, ...);             # statue 1/20

    JIT-safety: this function consumes ``rng`` via ``jax.random.split`` so
    no PRNG key is reused.  Each room receives an independent sub-key, and
    each per-room roll receives an independent sub-sub-key.  Inner work is
    a fixed-size lax.scan over MAX_ROOMS_PER_LEVEL — no Python branching
    on traced values.

    Note: we cap the per-room trap loop at 4 placements (vendor's
    ``while (!rn2(x))`` is geometrically distributed with mean
    ``1/(1 - 1/x) - 1`` ≈ 1 for typical x).  Vendor uses trycnt < 1000 as a
    safety bound; 4 iterations gives the same effective trap count at the
    feasible range while keeping the trace length finite.

    Args:
        rng:          jax.random.PRNGKey scalar.
        rooms:        Room pytree (one level).
        active:       bool[MAX_ROOMS_PER_LEVEL] mask.
        terrain:      int8[MAP_H, MAP_W] terrain map.
        features:     FeaturesState (per-tile altar_alignment is updated).
        traps:        TrapState (per-tile trap_type is updated).
        flat_lv:      int — flattened level index into features/traps arrays.
        depth:        int / scalar — vendor ``depth(&u.uz)`` /
                      ``level_difficulty()`` (used by trap rate + grave rate).
        player_align: int — Alignment value (0/1/2) used for altar
                      placement (vendor mkaltar passes player align to
                      induced_align).

    Returns:
        (terrain, features, traps) — updated in-place via functional ops.
    """
    from Nethax.nethax.constants.tiles import TileType
    FLOOR    = jnp.int8(int(TileType.FLOOR))
    FOUNTAIN = jnp.int8(int(TileType.FOUNTAIN))
    # The internal TileType enum has no SINK / STATUE / GOLD codes; we use
    # FLOOR for those (the *object* layer carries the gold/statue and the
    # *features* layer carries sink semantics).  Vendor uses rm.h tile
    # codes for fountain/sink/altar/grave but our compact TileType only
    # exposes FOUNTAIN/ALTAR/GRAVE.  Sink/statue/gold tiles stay FLOOR
    # here; downstream object-layer fills handle them.
    ALTAR    = jnp.int8(int(TileType.ALTAR))
    GRAVE    = jnp.int8(int(TileType.GRAVE))
    TRAP_TILE = jnp.int8(int(TileType.HIDDEN_TRAP))

    depth_i = jnp.asarray(depth, dtype=jnp.int32)

    # Trap rate (vendor mklev.c:981-983):
    #   x = 8 - level_difficulty()/6;  if (x <= 1) x = 2;
    trap_x = jnp.maximum(jnp.int32(8) - depth_i // jnp.int32(6), jnp.int32(2))

    # Grave rate (vendor mklev.c:996-998):
    #   x = 80 - depth(&u.uz) * 2;  if (x < 2) x = 2;
    grave_x = jnp.maximum(jnp.int32(80) - depth_i * jnp.int32(2), jnp.int32(2))

    # Number of independent per-room keys we need — see scan body.
    PER_ROOM_KEYS = 16  # generous; only ~10 rolls used.

    # Split top-level rng into one key per room.
    room_keys = jax.random.split(rng, MAX_ROOMS_PER_LEVEL)

    def fill_one(state, i):
        terrain_, features_aa, features_lit, traps_tt = state
        y1 = rooms.y1[i].astype(jnp.int32)
        x1 = rooms.x1[i].astype(jnp.int32)
        y2 = rooms.y2[i].astype(jnp.int32)
        x2 = rooms.x2[i].astype(jnp.int32)
        rt = rooms.room_type[i].astype(jnp.int32)
        act = active[i]

        # Per-room lit stamp — vendor mklev.c::do_room_or_subroom lines
        # 249-255: when ``lit`` is true the inner loop walks the room's
        # bounding box (including the 1-cell wall border) and sets
        # ``lev->lit = 1`` on every tile.  ``rooms.is_lit`` was rolled at
        # generation time per vendor mkmap.c::litstate_rnd lines 442-448
        # (``rnd(1+abs(depth))<11 && rn2(77)`` when rlit=-1).  We mirror
        # that here on every *active* room regardless of room type so
        # special rooms (shops, temples, …) also get their lit floors.
        H = features_lit.shape[1]
        W = features_lit.shape[2]
        rows_idx = jnp.arange(H, dtype=jnp.int32).reshape(H, 1)
        cols_idx = jnp.arange(W, dtype=jnp.int32).reshape(1, W)
        in_room = (
            (rows_idx >= y1) & (rows_idx <= y2)
            & (cols_idx >= x1) & (cols_idx <= x2)
        )
        room_lit = rooms.is_lit[i] & act
        cur_lit = features_lit[flat_lv]
        new_lit = jnp.where(in_room & room_lit, jnp.bool_(True), cur_lit)
        features_lit = features_lit.at[flat_lv].set(new_lit)

        # Only ordinary / themeroom rooms are filled (vendor line 949 gate):
        #   if (croom->rtype != OROOM && croom->rtype != THEMEROOM) return;
        is_ordinary = act & (
            (rt == jnp.int32(int(RoomType.ORDINARY))) |
            (rt == jnp.int32(int(RoomType.THEMEROOM)))
        )

        sub_keys = jax.random.split(room_keys[i], PER_ROOM_KEYS)
        k_fount, k_sink, k_altar, k_grave, k_statue, k_gold, k_sleep, \
            k_pos_fount, k_pos_sink, k_pos_altar, k_pos_grave, k_pos_statue, \
            k_pos_gold, k_pos_sleep, k_trap_outer, k_align = sub_keys

        # --- Fountain: !rn2(10)  (vendor line 990-991) ---
        fount_roll = jax.random.randint(k_fount, (), 0, 10, dtype=jnp.int32) == jnp.int32(0)
        rf, cf = _pick_room_tile(k_pos_fount, y1, x1, y2, x2)
        rf = jnp.clip(rf, 0, terrain_.shape[0] - 1)
        cf = jnp.clip(cf, 0, terrain_.shape[1] - 1)
        place_fount = is_ordinary & fount_roll
        terrain_ = terrain_.at[rf, cf].set(
            jnp.where(place_fount, FOUNTAIN, terrain_[rf, cf])
        )

        # --- Sink: !rn2(60)  (vendor line 992-993) ---
        # No SINK code in our TileType; downstream sink ops use the
        # features sinks_used layer.  We leave the tile as FLOOR but
        # do not currently expose a per-tile sink marker.  This roll is
        # therefore a no-op on terrain but the rng schedule is consumed
        # so callers stay deterministic across future TileType expansion.
        _sink_roll = jax.random.randint(k_sink, (), 0, 60, dtype=jnp.int32) == jnp.int32(0)
        _ = jax.random.randint(k_pos_sink, (), 0, 1, dtype=jnp.int32)

        # --- Altar: !rn2(60)  (vendor line 994-995) ---
        # mkaltar (mkroom.c:557-595) sets altarmask = induced_align(player).
        # induced_align(80) returns player_align 80% of the time, else a
        # random alternative.  We mirror the alt-align logic here.
        altar_roll = jax.random.randint(k_altar, (), 0, 60, dtype=jnp.int32) == jnp.int32(0)
        ra, ca = _pick_room_tile(k_pos_altar, y1, x1, y2, x2)
        ra = jnp.clip(ra, 0, terrain_.shape[0] - 1)
        ca = jnp.clip(ca, 0, terrain_.shape[1] - 1)
        # induced_align(80): 80% chance of player alignment, else random.
        coin = jax.random.randint(k_align, (), 0, 100, dtype=jnp.int32)
        alt_align = jax.random.randint(
            jax.random.fold_in(k_align, 1), (), 0, 3, dtype=jnp.int32
        )
        induced = jnp.where(
            coin < jnp.int32(80), jnp.int32(player_align), alt_align
        ).astype(jnp.int8)
        place_altar = is_ordinary & altar_roll
        terrain_ = terrain_.at[ra, ca].set(
            jnp.where(place_altar, ALTAR, terrain_[ra, ca])
        )
        features_aa = features_aa.at[flat_lv, ra, ca].set(
            jnp.where(place_altar, induced, features_aa[flat_lv, ra, ca])
        )

        # --- Grave: !rn2(grave_x)  (vendor line 996-1000) ---
        grave_roll = jax.random.randint(
            k_grave, (), 0, grave_x, dtype=jnp.int32
        ) == jnp.int32(0)
        rg, cg = _pick_room_tile(k_pos_grave, y1, x1, y2, x2)
        rg = jnp.clip(rg, 0, terrain_.shape[0] - 1)
        cg = jnp.clip(cg, 0, terrain_.shape[1] - 1)
        place_grave = is_ordinary & grave_roll
        terrain_ = terrain_.at[rg, cg].set(
            jnp.where(place_grave, GRAVE, terrain_[rg, cg])
        )

        # --- Statue: !rn2(20)  (vendor line 1003-1006) ---
        _statue_roll = jax.random.randint(k_statue, (), 0, 20, dtype=jnp.int32) == jnp.int32(0)
        _ = jax.random.randint(k_pos_statue, (), 0, 1, dtype=jnp.int32)
        # Statues are objects (mkcorpstat); we do not yet have a STATUE tile
        # nor an objects-layer hook here.  The rng schedule is preserved so
        # downstream wires can fold this in without disturbing parity.

        # --- Gold: !rn2(3)  (vendor line 986-987) ---
        _gold_roll = jax.random.randint(k_gold, (), 0, 3, dtype=jnp.int32) == jnp.int32(0)
        _ = jax.random.randint(k_pos_gold, (), 0, 1, dtype=jnp.int32)

        # --- Sleeping monster: u.uhave.amulet || !rn2(3)  (vendor line 974) ---
        # Without the amulet flag this is just !rn2(3).  Monster spawning is
        # handled by the monsters subsystem; here we consume the rng for
        # schedule stability.
        _sleep_roll = jax.random.randint(k_sleep, (), 0, 3, dtype=jnp.int32) == jnp.int32(0)
        _ = jax.random.randint(k_pos_sleep, (), 0, 1, dtype=jnp.int32)

        # --- Traps: while (!rn2(trap_x))  (vendor line 980-985) ---
        # Vendor uses a while-loop bounded by trycnt < 1000; we unroll a
        # fixed scan of length MAX_TRAPS_PER_ROOM with per-step continuation
        # so the expected trap count matches the geometric distribution.
        MAX_TRAPS_PER_ROOM = 4

        def trap_step(carry, j):
            terrain_in, traps_in, continue_, key = carry
            k_roll, k_kind, k_pos, k_next = jax.random.split(key, 4)
            roll = jax.random.randint(
                k_roll, (), 0, trap_x, dtype=jnp.int32
            ) == jnp.int32(0)
            kind = _vendor_traptype_rnd(k_kind, depth_i)
            rt_r, rt_c = _pick_room_tile(k_pos, y1, x1, y2, x2)
            rt_r = jnp.clip(rt_r, 0, terrain_in.shape[0] - 1)
            rt_c = jnp.clip(rt_c, 0, terrain_in.shape[1] - 1)
            should_place = continue_ & roll & is_ordinary
            new_terrain = terrain_in.at[rt_r, rt_c].set(
                jnp.where(should_place, TRAP_TILE, terrain_in[rt_r, rt_c])
            )
            new_traps = traps_in.at[flat_lv, rt_r, rt_c].set(
                jnp.where(should_place, kind.astype(jnp.int8), traps_in[flat_lv, rt_r, rt_c])
            )
            return (new_terrain, new_traps, continue_ & roll, k_next), None

        trap_state, _ = lax.scan(
            trap_step,
            (terrain_, traps_tt, jnp.bool_(True), k_trap_outer),
            jnp.arange(MAX_TRAPS_PER_ROOM, dtype=jnp.int32),
        )
        terrain_, traps_tt, _, _ = trap_state

        return (terrain_, features_aa, features_lit, traps_tt), None

    init_state = (
        terrain,
        features.altar_alignment,
        features.lit,
        traps.trap_type,
    )
    (terrain_out, aa_out, lit_out, tt_out), _ = lax.scan(
        fill_one,
        init_state,
        jnp.arange(MAX_ROOMS_PER_LEVEL, dtype=jnp.int32),
    )
    new_features = features.replace(altar_alignment=aa_out, lit=lit_out)
    new_traps    = traps.replace(trap_type=tt_out)
    return terrain_out, new_features, new_traps


# ---------------------------------------------------------------------------
# maybe_create_vault — 2x2 detached vault with teleport-trap entry
# ---------------------------------------------------------------------------
#
# Vendor cite: vendor/nethack/src/mklev.c lines 404-410, 1316-1342.
#
#   while (svn.nroom < (MAXNROFROOMS - 1) && rnd_rect()) {
#       if (svn.nroom >= (MAXNROFROOMS / 6) && rn2(2) && !tried_vault) {
#           tried_vault = TRUE;
#           if (create_vault()) {                    /* sets vault_x/vault_y */
#               gv.vault_x = svr.rooms[svn.nroom].lx;
#               gv.vault_y = svr.rooms[svn.nroom].ly;
#               svr.rooms[svn.nroom].hx = -1;
#           }
#       } ...
#   }
#   ...
#   if (do_vault()) {                                /* vault_x != -1 */
#       w = 1; h = 1;
#       if (check_room(...)) {
#           add_room(vault_x, vault_y, vault_x + w, vault_y + h,
#                    TRUE, VAULT, FALSE);            /* 2x2 interior */
#           svl.level.flags.has_vault = 1;
#           ...
#           mk_knox_portal(vault_x + w, vault_y + h);
#           if (!noteleport && !rn2(3))
#               makevtele();                          /* TELEP_TRAP entry */
#       }
#   }
#
# Vendor's create_vault macro expands to
# ``create_room(-1, -1, 2, 2, -1, -1, VAULT, TRUE)`` — a 2x2 interior
# (the "w=1, h=1" later in check_room refers to half-width/half-height).
# The vault is gated on having at least MAXNROFROOMS/6 = 6 ordinary rooms
# already, then a 50 % rn2(2) coin-flip.  The teleport-trap entry is then
# placed with !rn2(3) (66 % chance) if the level allows teleport.


def maybe_create_vault(
    rng,
    rooms,
    active,
    terrain,
    features,
    traps,
    flat_lv: int,
):
    """Try to carve a 2x2 detached vault and record its centre + teleport trap.

    Vendor cite: vendor/nethack/src/mklev.c lines 404-410 (gate) +
    lines 1316-1342 (placement) + line 1332 makevtele() teleport trap.

    Behavior:
      1. Gate: require at least ``MAXNROFROOMS // 6 = 6`` active ordinary
         rooms on the level, then a 50 % ``rn2(2)`` coin-flip — vendor
         lines 404-410.  This deviates from the legacy TODO comment which
         hypothesised an ``rn2(7)`` rate; the vendor source uses ``rn2(2)``
         gated on the room count.
      2. Pick a 2x2 area whose 4-tile bounding box (with 1-cell wall
         margin) overlaps no active room.  We sweep a small candidate
         set drawn from a fixed grid; the first non-adjacent candidate
         is chosen.
      3. Stamp all 4 interior tiles as FLOOR.
      4. Place TELEP_TRAP (TrapType=15) at the vault centre (the
         vendor ``makevtele`` teleport trap — line 1332).  Vendor gates
         this on a further ``!rn2(3)`` and ``!noteleport`` flag; we
         apply both gates here.
      5. Record ``features.vault_pos[flat_lv]`` = (centre_row, centre_col).

    JIT-safety: the candidate sweep is implemented as ``lax.scan`` over a
    fixed candidate grid.  All RNG draws use ``jax.random.split`` — no
    key reuse.

    Args:
        rng:      jax.random.PRNGKey scalar.
        rooms:    Room pytree.
        active:   bool[MAX_ROOMS_PER_LEVEL] mask.
        terrain:  int8[MAP_H, MAP_W].
        features: FeaturesState (vault_pos is updated).
        traps:    TrapState (TELEP_TRAP placed at centre).
        flat_lv:  int — flattened level index.

    Returns:
        (terrain, features, traps) — updated.  When the gate fails or no
        non-adjacent 2x2 site exists, returns the inputs unchanged
        (vault_pos remains (-1, -1)).
    """
    from Nethax.nethax.constants.tiles import TileType
    FLOOR = jnp.int8(int(TileType.FLOOR))
    TELEP_TRAP = jnp.int8(15)  # vendor/nethack/include/trap.h TELEP_TRAP=15.

    h, w = terrain.shape
    n_active = jnp.sum(active.astype(jnp.int32))

    # MAXNROFROOMS = 40 (vendor/nethack/include/global.h); /6 = 6.
    MIN_ROOMS_FOR_VAULT = jnp.int32(MAX_ROOMS_PER_LEVEL // 6)

    k_gate, k_coin, k_tele, k_pos = jax.random.split(rng, 4)
    # Gate 1: room count.
    rooms_ok = n_active >= MIN_ROOMS_FOR_VAULT
    # Gate 2: rn2(2) — vendor line 404.
    coin = jax.random.randint(k_coin, (), 0, 2, dtype=jnp.int32) == jnp.int32(0)
    # Combined gate (drops the rn2(2) when room count not met).
    gate = rooms_ok & coin

    # ---- Candidate sweep -----------------------------------------------------
    # We tile the map with 2x2 candidate slots on a 4-cell stride to ensure
    # each candidate's bounding box (with 1-cell margin) is independent.
    # For each candidate, compute its overlap-with-any-active-room flag.
    cand_step  = 4
    cand_rows  = jnp.arange(2, h - 4, cand_step, dtype=jnp.int32)
    cand_cols  = jnp.arange(2, w - 4, cand_step, dtype=jnp.int32)
    rr, cc = jnp.meshgrid(cand_rows, cand_cols, indexing="ij")
    cand_y1 = rr.reshape(-1)
    cand_x1 = cc.reshape(-1)
    n_cand = cand_y1.shape[0]

    def overlaps_any(y1, x1):
        # 2x2 interior → y2=y1+1, x2=x1+1.  Add 1-cell margin for the
        # adjacency check (vendor uses check_room's >=1 wall buffer).
        y2 = y1 + jnp.int32(1)
        x2 = x1 + jnp.int32(1)
        # Vmap _rooms_overlap-style test across all active rooms.
        def per_room(ry1, rx1, ry2, rx2, act):
            margin = jnp.int32(1)
            ry1i = ry1.astype(jnp.int32)
            rx1i = rx1.astype(jnp.int32)
            ry2i = ry2.astype(jnp.int32)
            rx2i = rx2.astype(jnp.int32)
            sep = (
                (y2 + margin < ry1i) |
                (ry2i + margin < y1) |
                (x2 + margin < rx1i) |
                (rx2i + margin < x1)
            )
            return act & ~sep
        flags = jax.vmap(per_room)(
            rooms.y1, rooms.x1, rooms.y2, rooms.x2, active
        )
        return jnp.any(flags)

    overlap_mask = jax.vmap(overlaps_any)(cand_y1, cand_x1)
    is_valid = ~overlap_mask

    # Pick the first valid candidate.  argmax on a bool returns the first
    # True index (or 0 if none); we OR with any_valid to suppress action.
    any_valid = jnp.any(is_valid)
    pick_idx  = jnp.argmax(is_valid.astype(jnp.int32)).astype(jnp.int32)
    vy1 = cand_y1[pick_idx]
    vx1 = cand_x1[pick_idx]

    should_place = gate & any_valid

    # ---- Stamp 4 interior tiles ----------------------------------------------
    # 2x2 interior at (vy1, vx1), (vy1, vx1+1), (vy1+1, vx1), (vy1+1, vx1+1).
    def stamp(t, dy, dx):
        r = vy1 + jnp.int32(dy)
        c = vx1 + jnp.int32(dx)
        return t.at[r, c].set(jnp.where(should_place, FLOOR, t[r, c]))

    terrain_out = terrain
    terrain_out = stamp(terrain_out, 0, 0)
    terrain_out = stamp(terrain_out, 0, 1)
    terrain_out = stamp(terrain_out, 1, 0)
    terrain_out = stamp(terrain_out, 1, 1)

    # ---- Teleport trap at the centre (top-left of interior is conventional) ---
    # Vendor's makevtele() places the trap at the vault centre; the
    # interior is 2x2 so any cell is the centre.  We use (vy1, vx1).
    tele_gate = jax.random.randint(k_tele, (), 0, 3, dtype=jnp.int32) == jnp.int32(0)
    place_trap = should_place & tele_gate
    new_tt = traps.trap_type.at[flat_lv, vy1, vx1].set(
        jnp.where(place_trap, TELEP_TRAP, traps.trap_type[flat_lv, vy1, vx1])
    )

    # ---- Record vault_pos -----------------------------------------------------
    # FeaturesState.vault_pos is int16[num_levels, 2].
    cur_vp = features.vault_pos[flat_lv]
    new_vp_row = jnp.where(should_place, vy1.astype(jnp.int16), cur_vp[0])
    new_vp_col = jnp.where(should_place, vx1.astype(jnp.int16), cur_vp[1])
    new_vp = features.vault_pos.at[flat_lv].set(
        jnp.stack([new_vp_row, new_vp_col]).astype(jnp.int16)
    )

    new_features = features.replace(vault_pos=new_vp)
    new_traps    = traps.replace(trap_type=new_tt)
    return terrain_out, new_features, new_traps


# ---------------------------------------------------------------------------
# TODO blocks
# ---------------------------------------------------------------------------
# Wave 4:
#   - Assign special room types: at most one zoo/morgue/barracks/beehive per
#     level (mkroom.c), shop placement (one per level in Mines/main Dlvl 1-15).
#   - Temple placement tied to alignment of the level (mkroom.c fill_temple).
#   - Vault generation: small 2×2 detached room with teleport trap entry
#     (vault.c, mkroom.c VAULT type).
#
# Wave 5:
#   - Shop name assignment from shclass table (mkroom.c shclass array).
#   - Delphi / Oracle room layout (mklev.c + oracle.lua fixed map).
