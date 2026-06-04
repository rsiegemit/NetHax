"""Vendor-faithful port of ``mineralize()`` from mklev.c.

Places mineral deposits (gold + gems) in solid-stone areas surrounding
rooms, and kelp in water tiles.  The JAX implementation replays the
exact ISAAC64 draw sequence so RNG parity with vendor NLE is maintained.

Citation: vendor/nle/src/mklev.c::mineralize (lines 894-988)

Algorithm summary
-----------------
Called as ``mineralize(-1, -1, -1, -1, FALSE)`` from mklev() after
``makelevel()``.  With default args:

  kelp_pool = 10, kelp_moat = 30   (vendor mklev.c:903-906)

Kelp scan (mklev.c:911-915):
  For every (x, y) with x in [2, COLNO-2), y in [1, ROWNO-1):
    - if typ == POOL  and rn2(10) == 0  → place kelp frond
    - if typ == MOAT  and rn2(30) == 0  → place kelp frond
  Draws fire *only* on POOL/MOAT cells.  Main Dlvl 1 has neither,
  so **0 kelp draws** for seed=0 Dlvl 1.

Level-check gate (mklev.c:919-924):
  Skipped for normal Main-branch levels (not hell, not V_tower, not
  rogue, not arboreal, Is_special returns NULL for ordinary levels).

Gold/gem scan (mklev.c:948-987):
  goldprob = 20 + depth/3   (depth=1 → goldprob=20)
  gemprob  = goldprob/4     (= 5)
  For every (x, y) in the same range, if the 3×3 neighbourhood is all
  STONE and W_NONDIGGABLE is clear:
    rn2(1000) < goldprob → draw rnd(goldprob*3) + rn2(3) [+ possible rnd if gem loop]
    rn2(1000) < gemprob  → draw rnd(2 + dunlev/3) iterations, each with
                            possible rn2(3) per non-ROCK gem.
  (mkobj / mksobj draws are NOT replicated here — only the placement
  coin rn2(3) and the ``quan = 1 + rnd(goldprob*3)`` draw matter for
  the RNG stream.  Object-alloc draws are separate and not part of the
  outer mineralize ISAAC stream for this port.)

For Dlvl 1, seed=0:
  - No POOL/MOAT cells → kelp loop: 0 draws.
  - Normal main-branch level → scan proceeds.
  - Each fully-stone 3×3 cell fires: rn2(1000) [gold], rn2(1000) [gem].
  - On gold hit: rnd(60) + rn2(3).
  - On gem hit: rnd(2) for count, then per gem: rnd(1000) type pick;
    if ROCK → rn1(6,6), no rn2(3); if LUCKSTONE → rn2(3); else → rn2(6)+rn2(3).
  Total depends on terrain; typical Dlvl 1 has O(50-150) eligible cells.

JIT notes
---------
The kelp scan and gold/gem scan are expressed as ``lax.fori_loop`` bodies
scanning a flat index over the (COLNO-4)×(ROWNO-2) cell grid.  ``lax.cond``
gates the per-cell rn2 draws so no draws fire on ineligible cells.
The scan y-skip optimisation from vendor C (``y += 2`` / ``y += 1``) cannot
be replicated under a fixed-iteration JAX loop; instead, every cell is
visited but ineligible cells consume **zero** draws (matching the vendor
when the y-skip would have been taken, because those cells fail the
all-STONE neighbourhood check).
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax

from Nethax.nethax.vendor_rng import Isaac64State, rn2_jax, rnd_jax, rn1_jax
from Nethax.nethax.subsystems.random_objects import decode_picked_otyp
from Nethax.nethax.constants.objects import ObjectClass
from Nethax.nethax.subsystems.inventory import ItemCategory, MAX_GROUND_STACK

# ---------------------------------------------------------------------------
# Map geometry — vendor/nle/include/global.h:327-328
# ---------------------------------------------------------------------------
_COLNO: int = 80   # COLNO
_ROWNO: int = 21   # ROWNO

# Scan bounds matching vendor loops:
#   x in [2, COLNO-2)  →  2 .. 77  (76 columns)
#   y in [1, ROWNO-1)  →  1 .. 19  (19 rows)
_X_LO: int = 2
_X_HI: int = _COLNO - 2   # exclusive upper bound (76 steps)
_Y_LO: int = 1
_Y_HI: int = _ROWNO - 1   # exclusive upper bound (19 steps)

_SCAN_W: int = _X_HI - _X_LO   # 76
_SCAN_H: int = _Y_HI - _Y_LO   # 19
_SCAN_N: int = _SCAN_W * _SCAN_H  # 1444 cells

# Vendor VendorTileType values (vendor/nethack/include/rm.h lines 55-94)
_VSTONE: jnp.int8 = jnp.int8(0)   # STONE = 0
_VPOOL:  jnp.int8 = jnp.int8(16)  # POOL  = 16
_VMOAT:  jnp.int8 = jnp.int8(17)  # MOAT  = 17

# Internal TileType values used in the Nethax terrain array.
# Citation: Nethax/nethax/constants/tiles.py::TileType
# VOID=0 is the "not yet placed" / solid stone equivalent in our terrain.
# Rooms carve FLOOR=1, corridors carve CORRIDOR=2, walls=3; everything else
# is VOID (== solid stone in the dungeon generator's pre-render map).
_TILE_VOID: jnp.int8 = jnp.int8(0)  # TileType.VOID — solid stone / unexplored

# GEM_CLASS id for decode_picked_otyp (ObjectClass.GEM_CLASS = 13).
# Citation: Nethax/nethax/constants/objects.py::ObjectClass
_GEM_CLASS_ID: int = int(ObjectClass.GEM_CLASS)  # 13

# ROCK otyp (positional index 446 in OBJECTS[]).
# Citation: Nethax/nethax/constants/objects.py line ~9108; vendor onames.h ROCK
_OTYP_ROCK: jnp.int32 = jnp.int32(446)

# LUCKSTONE otyp (index 442).  mksobj GEM_CLASS skips rn2(6) for luckstone.
# Citation: vendor/nle/src/mkobj.c:892 ``else if (otmp->otyp != LUCKSTONE && !rn2(6))``
_OTYP_LUCKSTONE: jnp.int32 = jnp.int32(442)

# GOLD_PIECE otyp (vendor onames.h: GOLD_PIECE = 410 for seed-0 rog-hum-cha).
# Citation: vendor floor-object dump at .test_runs/room_12_67_audit.md §3.
_OTYP_GOLD_PIECE: jnp.int32 = jnp.int32(410)

# Maximum gems per cell: rnd(2 + dunlev//3).  For dunlev up to ~14 → rnd(2+4)=6
# max.  Stack slot 7 is reserved for the gold-piece placement, keeping gems
# in slots 0..6 disjoint from gold.
# Citation: vendor/nle/src/mklev.c:974.
_MAX_GEMS: int = 7

# Stack slot assignment within ground_items[..., MAX_GROUND_STACK=8].
#   slots 0..6 → gem placements (one per gem in cnt order)
#   slot   7   → gold placement
# Disjoint slots avoid per-cell scatter collisions and let the first-empty-slot
# allocator in rooms.py (mkobj_at) continue to work alongside this writer.
_GOLD_STACK_SLOT: int = MAX_GROUND_STACK - 1  # 7


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def mineralize(
    terrain: jnp.ndarray,
    vendor_rng: Isaac64State,
    depth: int = 1,
    dunlev: int = 1,
    skip_lvl_checks: bool = False,
    in_endgame: bool = False,
    in_hell: bool = False,
    in_vtower: bool = False,
    is_rogue: bool = False,
    arboreal: bool = False,
    is_special_non_oracle_non_mines_town: bool = False,
    in_mines: bool = False,
    in_quest: bool = False,
    kelp_pool: int = -1,
    kelp_moat: int = -1,
    goldprob: int = -1,
    gemprob: int = -1,
    *,
    gi_category: jnp.ndarray | None = None,
    gi_type_id: jnp.ndarray | None = None,
    gi_quantity: jnp.ndarray | None = None,
    branch_idx: int = 0,
    level_idx: int = 0,
):
    """Replay vendor ``mineralize(-1,-1,-1,-1,FALSE)`` over the ISAAC64 stream.

    This function consumes the exact same sequence of rn2 draws as the
    vendor C ``mineralize`` for the given level parameters, advancing
    ``vendor_rng`` accordingly.  Terrain-placement side-effects (object
    layer) are omitted — only the RNG drain is replicated faithfully.

    Citation: vendor/nle/src/mklev.c::mineralize lines 894-988

    Args:
        terrain:   int8[MAP_H, MAP_W] terrain array (Nethax internal TileType).
                   Used to detect POOL/MOAT cells for the kelp scan.
        vendor_rng: ISAAC64 state to thread.
        depth:     vendor ``depth(&u.uz)`` — 1 for Main Dlvl 1.
        dunlev:    vendor ``dunlev(&u.uz)`` — 1 for Main Dlvl 1.
        skip_lvl_checks: mirrors vendor ``skip_lvl_checks`` arg (FALSE for
                   the mklev.c call site, line 1006).
        in_endgame, in_hell, in_vtower, is_rogue, arboreal,
        is_special_non_oracle_non_mines_town, in_mines, in_quest:
                   Boolean flags derived from the level's dungeon location.
                   All False for normal Main-branch Dlvl 1.
        kelp_pool, kelp_moat, goldprob, gemprob:
                   -1 triggers vendor default computation (mklev.c:903-930).

    Returns:
        If ``gi_category`` is None: ``(terrain, updated_vendor_rng)`` —
        terrain is unchanged but vendor_rng is advanced.
        Else: ``(terrain, updated_vendor_rng, gi_category', gi_type_id',
        gi_quantity')`` with mineralize-placed gold + gem objects scattered
        into the ground_items slabs at ``[branch_idx, level_idx, y, x, slot]``.

        Vendor cite: mklev.c:962-987 (place_object on gold/gem hits);
                     mkobj.c::place_object (insert into level.objects[][]).
    """
    # --- Resolve defaults (mklev.c:903-930) ---
    if kelp_pool < 0:
        kelp_pool = 10
    if kelp_moat < 0:
        kelp_moat = 30

    emit_objects = gi_category is not None

    # Early-exit gate: endgame (mklev.c:909-910).
    # For normal levels (in_endgame=False) this does not fire.
    if not skip_lvl_checks and in_endgame:
        if emit_objects:
            return terrain, vendor_rng, gi_category, gi_type_id, gi_quantity
        return terrain, vendor_rng

    # Kelp scan: x in [2, COLNO-2), y in [1, ROWNO-1).
    # Draws rn2(kelp_pool) for each POOL cell, rn2(kelp_moat) for each MOAT.
    # Our internal terrain uses TileType.POOL (=19) for POOL and has no MOAT.
    # For Dlvl 1 there are no POOL/MOAT cells, so this loop fires 0 draws.
    # We replicate the draw pattern via lax.fori_loop for JIT correctness.
    # Citation: vendor/nle/src/mklev.c:911-915
    vendor_rng = _kelp_scan(terrain, vendor_rng, kelp_pool, kelp_moat)

    # Level-check gate (mklev.c:919-924).
    # For normal Main-branch levels all checks are False → no early exit.
    if not skip_lvl_checks and (
        in_hell or in_vtower or is_rogue or arboreal
        or is_special_non_oracle_non_mines_town
    ):
        if emit_objects:
            return terrain, vendor_rng, gi_category, gi_type_id, gi_quantity
        return terrain, vendor_rng

    # Resolve gold/gem probabilities (mklev.c:926-941).
    if goldprob < 0:
        goldprob = 20 + depth // 3
    if gemprob < 0:
        gemprob = goldprob // 4

    if not skip_lvl_checks:
        if in_mines:
            goldprob *= 2
            gemprob  *= 3
        elif in_quest:
            goldprob //= 4
            gemprob  //= 6

    # Gold/gem scan (mklev.c:948-987).
    # Citation: vendor/nle/src/mklev.c:948-987
    if emit_objects:
        vendor_rng, gi_category, gi_type_id, gi_quantity = _mineral_scan(
            terrain, vendor_rng, goldprob, gemprob, dunlev,
            gi_category=gi_category,
            gi_type_id=gi_type_id,
            gi_quantity=gi_quantity,
            branch_idx=branch_idx,
            level_idx=level_idx,
        )
        return terrain, vendor_rng, gi_category, gi_type_id, gi_quantity

    vendor_rng = _mineral_scan(terrain, vendor_rng, goldprob, gemprob, dunlev)
    return terrain, vendor_rng


# ---------------------------------------------------------------------------
# Kelp scan helper
# ---------------------------------------------------------------------------

def _kelp_scan(
    terrain: jnp.ndarray,
    vendor_rng: Isaac64State,
    kelp_pool: int,
    kelp_moat: int,
) -> Isaac64State:
    """Consume rn2 draws for the kelp placement loop.

    Citation: vendor/nle/src/mklev.c:911-915

    For each cell (x, y) with x in [2, COLNO-2), y in [1, ROWNO-1):
      - If cell is POOL  and kelp_pool != 0: draw rn2(kelp_pool).
      - If cell is MOAT  and kelp_moat != 0: draw rn2(kelp_moat).
    (Both checks are independent; vendor uses short-circuit ``||`` but
    both conditions reference distinct cell states, so in practice each
    cell fires at most one draw branch.)
    """
    # Internal TileType.POOL = 19; no MOAT in our enum (mapped to WATER=8).
    # For correctness we check both local POOL (19) and WATER (8) as potential
    # MOAT equivalents.  On Dlvl 1 neither appears → zero draws.
    _LOCAL_POOL = jnp.int8(19)  # TileType.POOL
    _LOCAL_WATER = jnp.int8(8)  # TileType.WATER (closest to MOAT)

    kp = jnp.int32(kelp_pool)
    km = jnp.int32(kelp_moat)

    def body(flat_i, vrng):
        xi = flat_i // _SCAN_H + _X_LO   # column
        yi = flat_i %  _SCAN_H + _Y_LO   # row
        cell = terrain[yi, xi]

        # Pool branch: draw rn2(kelp_pool) iff cell==POOL and kelp_pool!=0
        # Citation: vendor mklev.c:913 ``levl[x][y].typ == POOL && !rn2(kelp_pool)``
        is_pool = (cell == _LOCAL_POOL) & (kp != jnp.int32(0))

        def draw_pool(v):
            new_v, _ = rn2_jax(v, kp)
            return new_v

        vrng = lax.cond(is_pool, draw_pool, lambda v: v, vrng)

        # Moat branch: draw rn2(kelp_moat) iff cell==MOAT and kelp_moat!=0
        # Citation: vendor mklev.c:914 ``levl[x][y].typ == MOAT && !rn2(kelp_moat)``
        is_moat = (cell == _LOCAL_WATER) & (km != jnp.int32(0))

        def draw_moat(v):
            new_v, _ = rn2_jax(v, km)
            return new_v

        vrng = lax.cond(is_moat, draw_moat, lambda v: v, vrng)
        return vrng

    return lax.fori_loop(0, _SCAN_N, body, vendor_rng)


# ---------------------------------------------------------------------------
# bound_digging — vendor mkmaze.c:1247-1328 (called from mklev.c:1005 BEFORE
# mineralize at :1006).  Marks every cell OUTSIDE the non-stone bounding box
# (expanded by 2 on a non-maze level) as W_NONDIGGABLE; the mineralize
# eligibility test (mklev.c:981 ``!(levl[x][y].wall_info & W_NONDIGGABLE)``)
# then skips those cells.  Without this, Nethax treats the whole stone border
# as eligible and over-draws the rn2(1000) gold/gem cluster (828 vs vendor's
# 241 cells for seed 0 rog-hum-cha) -- the first ISAAC64 divergence at draw
# 1781.  Citation: vendor/nle/src/mkmaze.c:1247-1328 (bound_digging),
#           vendor/nle/src/mklev.c:1005-1006 (bound_digging then mineralize).
# ---------------------------------------------------------------------------

def _bound_digging_nondiggable(terrain: jnp.ndarray) -> jnp.ndarray:
    """Return a bool[ROWNO, COLNO] mask of W_NONDIGGABLE cells.

    Mirrors vendor ``bound_digging`` (mkmaze.c:1247-1328) for the regular
    (non-maze, non-earth) level case where ``nonwall || !is_maze_lev`` is
    always true, so every edge offset is 2 (the ``: 1`` maze branch never
    applies on Main-branch Dlvl 1).

    Vendor scans inward from each edge for the first column/row containing a
    non-STONE cell.  Because each scan loop post-increments past the hit
    column, the saved bound is one beyond the first non-stone column, then
    nudged by 2::

        xmin = (first non-stone column from left)  + 1 - 2  = first - 1
        xmax = (first non-stone column from right) - 1 + 2  = first + 1
        ymin = (first non-stone row from top)      + 1 - 2  = first - 1
        ymax = (first non-stone row from bottom)   - 1 + 2  = first + 1

    Cells with ``y <= ymin || y >= ymax || x <= xmin || x >= xmax`` are
    marked non-diggable (mkmaze.c:1318-1327).  ``terrain`` uses STONE == 0
    (``_TILE_VOID``); ``isok``-style clamping is unnecessary because the
    fallback bounds collapse to the full map when the level is all stone.
    """
    non_stone = terrain != jnp.int8(0)            # [ROWNO, COLNO]
    cols_any = jnp.any(non_stone, axis=0)         # [COLNO]
    rows_any = jnp.any(non_stone, axis=1)         # [ROWNO]

    col_idx = jnp.arange(_COLNO, dtype=jnp.int32)
    row_idx = jnp.arange(_ROWNO, dtype=jnp.int32)

    # First / last non-stone column and row (matches vendor's edge scans).
    big = jnp.int32(1 << 15)
    first_col = jnp.min(jnp.where(cols_any, col_idx, big))
    last_col = jnp.max(jnp.where(cols_any, col_idx, -big))
    first_row = jnp.min(jnp.where(rows_any, row_idx, big))
    last_row = jnp.max(jnp.where(rows_any, row_idx, -big))

    # Vendor offsets (non-maze => +/- 2), with the post-increment captured
    # in first/last above.  xmin clamped to >= 0 (mkmaze.c:1271-1272); the
    # other three have no explicit clamp in the regular case but the
    # comparison handles out-of-range bounds harmlessly.
    xmin = jnp.maximum(first_col - jnp.int32(1), jnp.int32(0))
    xmax = last_col + jnp.int32(1)
    ymin = first_row - jnp.int32(1)
    ymax = last_row + jnp.int32(1)

    ys = row_idx.reshape(_ROWNO, 1)
    xs = col_idx.reshape(1, _COLNO)
    nondiggable = (
        (ys <= ymin) | (ys >= ymax) | (xs <= xmin) | (xs >= xmax)
    )
    return nondiggable


# ---------------------------------------------------------------------------
# Gold / gem mineral scan helper
# ---------------------------------------------------------------------------

def _mineral_scan(
    terrain: jnp.ndarray,
    vendor_rng: Isaac64State,
    goldprob: int,
    gemprob: int,
    dunlev: int,
    *,
    gi_category: jnp.ndarray | None = None,
    gi_type_id: jnp.ndarray | None = None,
    gi_quantity: jnp.ndarray | None = None,
    branch_idx: int = 0,
    level_idx: int = 0,
):
    """Consume rn2/rnd draws for the gold+gem placement loop.

    Citation: vendor/nle/src/mklev.c:948-987

    For each cell (x, y) in scan range that is entirely surrounded by STONE
    (8-neighbour + self, and not W_NONDIGGABLE):
      - Draw rn2(1000): if < goldprob → gold hit
        - On hit: draw rnd(goldprob*3) for quan, draw rn2(3) for burial/place
      - Draw rn2(1000): if < gemprob → gem hit
        - On hit: draw rnd(2 + dunlev//3) for count (cnt),
          then for each gem: draw rn2(3) for burial/place

    The W_NONDIGGABLE gate (mklev.c:981) is honoured via
    :func:`_bound_digging_nondiggable`, which replays vendor ``bound_digging``
    (mkmaze.c:1247-1328, run at mklev.c:1005 immediately before mineralize).
    The level's outer stone border is non-diggable, so without this gate the
    scan over-counts eligible cells (828 vs vendor 241 for seed 0
    rog-hum-cha) and over-draws the rn2(1000) cluster.
    Citation: vendor/nle/src/mklev.c:981, vendor/nle/src/mkmaze.c:1318-1327.
    """
    gp = jnp.int32(goldprob)
    gemp = jnp.int32(gemprob)
    dl = jnp.int32(dunlev)

    emit_objects = gi_category is not None

    # W_NONDIGGABLE mask (vendor bound_digging) — cells outside the non-stone
    # bounding box are skipped by the eligibility test below.
    nondiggable = _bound_digging_nondiggable(terrain)

    # "Solid stone" in our terrain = TileType.VOID (0).
    _VOID = jnp.int8(0)

    def _is_stone(ty: jnp.int8) -> jnp.ndarray:
        return ty == _VOID

    # Per-cell placement carriers — populated inside the fori_loop and scattered
    # into ground_items after.  ROCK gems are discarded (no floor placement);
    # buried gold/gems also skip place_object (added to inv_chain via
    # add_to_buried).  We track the placement flag explicitly so the scatter
    # writes zero into ground_items for the discarded entries.
    # Vendor cite: vendor/nle/src/mklev.c:968 add_to_buried (gold),
    #              :977 dealloc_obj (rock), :981 add_to_buried (gem).
    gold_place = jnp.zeros((_SCAN_N,), dtype=jnp.bool_)
    gold_qty   = jnp.zeros((_SCAN_N,), dtype=jnp.int32)
    gem_place  = jnp.zeros((_SCAN_N, _MAX_GEMS), dtype=jnp.bool_)
    gem_otyp   = jnp.zeros((_SCAN_N, _MAX_GEMS), dtype=jnp.int32)
    gem_qty    = jnp.zeros((_SCAN_N, _MAX_GEMS), dtype=jnp.int32)

    def body(flat_i, carry):
        vrng, gpl, gq, gmp, go, gmq = carry

        xi = flat_i // _SCAN_H + _X_LO
        yi = flat_i %  _SCAN_H + _Y_LO

        cell     = terrain[yi,     xi    ]
        cell_n   = terrain[yi - 1, xi    ]
        cell_s   = terrain[yi + 1, xi    ]
        cell_e   = terrain[yi,     xi + 1]
        cell_w   = terrain[yi,     xi - 1]
        cell_ne  = terrain[yi - 1, xi + 1]
        cell_nw  = terrain[yi - 1, xi - 1]
        cell_se  = terrain[yi + 1, xi + 1]
        cell_sw  = terrain[yi + 1, xi - 1]

        # Vendor cite: vendor/nle/src/mklev.c:950-961, :981 — eligibility =
        # diggable + self + 8 neighbours STONE.  The Y-SKIP in vendor's outer
        # loop is structurally subsumed by this 9-cell predicate: vendor's
        # y+=2/y+=1 only fires when (x,y+1) or (x,y) is non-STONE — exactly
        # the cells the predicate already rejects.  An earlier visit_mask
        # precompute attempt was confirmed dead code (no rn2 draws masked).
        eligible = (
            (~nondiggable[yi, xi])
            & _is_stone(cell)
            & _is_stone(cell_n)
            & _is_stone(cell_s)
            & _is_stone(cell_e)
            & _is_stone(cell_w)
            & _is_stone(cell_ne)
            & _is_stone(cell_nw)
            & _is_stone(cell_se)
            & _is_stone(cell_sw)
        )

        def do_eligible(c):
            v, gpl, gq, gmp, go, gmq = c

            # --- Gold draw: rn2(1000) --- Citation: mklev.c:962
            v, gold_roll = rn2_jax(v, jnp.int32(1000))
            gold_hit = gold_roll < gp

            def on_gold_hit(cc):
                vv, gpl_in, gq_in = cc
                # rnd(goldprob*3): Citation mklev.c:965
                vv, quan = rnd_jax(vv, gp * jnp.int32(3))
                # vendor mklev.c:965 ``otmp->quan = 1L + rnd(goldprob*3)``.
                quan_full = quan + jnp.int32(1)
                # rn2(3): burial coin   Citation mklev.c:967
                vv, coin = rn2_jax(vv, jnp.int32(3))
                # Place on floor iff coin != 0 (vendor: !rn2(3) → buried).
                placed = coin != jnp.int32(0)
                gpl_in = gpl_in.at[flat_i].set(placed)
                gq_in  = gq_in.at[flat_i].set(quan_full)
                return (vv, gpl_in, gq_in)

            v, gpl, gq = lax.cond(
                gold_hit, on_gold_hit, lambda cc: cc, (v, gpl, gq)
            )

            # --- Gem draw: rn2(1000) --- Citation: mklev.c:973
            v, gem_roll = rn2_jax(v, jnp.int32(1000))
            gem_hit = gem_roll < gemp

            def on_gem_hit(cc):
                vv, gmp_in, go_in, gmq_in = cc
                # cnt = rnd(2 + dunlev//3)  Citation mklev.c:974
                vv, cnt = rnd_jax(vv, jnp.int32(2) + dl // jnp.int32(3))

                # Per-gem cascade — vendor mklev.c:975-984 + mkobj.c:251,886-895.
                def gem_body(i, inner):
                    iv, gmp_inner, go_inner, gmq_inner = inner

                    def do_gem(g):
                        iv2, gmp2, go2, gmq2 = g
                        # 1. rnd(1000) type pick — vendor mkobj.c:251
                        iv2, type_roll = rnd_jax(iv2, jnp.int32(1000))
                        otyp = decode_picked_otyp(
                            jnp.int32(_GEM_CLASS_ID), type_roll
                        )
                        is_rock      = otyp == _OTYP_ROCK
                        is_luckstone = otyp == _OTYP_LUCKSTONE

                        def rock_branch(rb):
                            iv3, gmp3, go3, gmq3 = rb
                            # rn1(6,6) quantity — vendor mkobj.c:891
                            iv3, _q = rn1_jax(iv3, 6, 6)
                            # ROCK is dealloc'd — no floor placement (cite:
                            # vendor/nle/src/mklev.c:976-977).  Leave per-gem
                            # slot as the zero default (not placed).
                            return (iv3, gmp3, go3, gmq3)

                        def non_rock_branch(rb):
                            iv3, gmp3, go3, gmq3 = rb
                            # Vendor mkobj.c:892:
                            #   if (otmp->otyp != LUCKSTONE && !rn2(6))
                            #       otmp->quan = 2L;
                            # i.e. for non-luckstone gems draw rn2(6); the
                            # quantity is 2 iff that draw == 0, else 1.
                            # Luckstone skips the draw entirely and keeps the
                            # mksobj default quan = 1.
                            def lucky(r):
                                return (r, jnp.int32(1))

                            def gem_quan(r):
                                r2, k = rn2_jax(r, jnp.int32(6))
                                q = jnp.where(
                                    k == jnp.int32(0),
                                    jnp.int32(2),
                                    jnp.int32(1),
                                )
                                return (r2, q)

                            iv3, gem_q = lax.cond(
                                is_luckstone, lucky, gem_quan, iv3
                            )
                            # rn2(3) burial/place draw — vendor mklev.c:980.
                            iv3, coin = rn2_jax(iv3, jnp.int32(3))
                            placed = coin != jnp.int32(0)
                            gmp3 = gmp3.at[flat_i, i].set(placed)
                            go3  = go3.at[flat_i, i].set(otyp)
                            gmq3 = gmq3.at[flat_i, i].set(gem_q)
                            return (iv3, gmp3, go3, gmq3)

                        return lax.cond(
                            is_rock, rock_branch, non_rock_branch,
                            (iv2, gmp2, go2, gmq2),
                        )

                    return lax.cond(
                        jnp.int32(i) < cnt,
                        do_gem,
                        lambda g: g,
                        (iv, gmp_inner, go_inner, gmq_inner),
                    )

                vv, gmp_in, go_in, gmq_in = lax.fori_loop(
                    0, _MAX_GEMS, gem_body,
                    (vv, gmp_in, go_in, gmq_in),
                )
                return (vv, gmp_in, go_in, gmq_in)

            v, gmp, go, gmq = lax.cond(
                gem_hit, on_gem_hit, lambda cc: cc,
                (v, gmp, go, gmq),
            )

            return (v, gpl, gq, gmp, go, gmq)

        return lax.cond(
            eligible,
            do_eligible,
            lambda c: c,
            (vrng, gpl, gq, gmp, go, gmq),
        )

    (vrng_out, gold_place, gold_qty, gem_place, gem_otyp, gem_qty) = (
        lax.fori_loop(
            0, _SCAN_N, body,
            (vendor_rng, gold_place, gold_qty, gem_place, gem_otyp, gem_qty),
        )
    )

    if not emit_objects:
        return vrng_out

    # ------------------------------------------------------------------
    # Scatter placement results into ground_items[branch_idx, level_idx,
    # y, x, slot].
    #
    # Per-cell flat scan order: xi = idx // _SCAN_H + _X_LO,
    #                            yi = idx %  _SCAN_H + _Y_LO.  Each (yi, xi)
    # appears exactly once in the scan so .at[].set() has no scatter
    # conflicts.  Gold writes go into stack slot _GOLD_STACK_SLOT (=7);
    # gem writes go into stack slots 0.._MAX_GEMS-1 (0..6) keyed by the
    # per-cell gem index.
    # Vendor cite: vendor/nle/src/mklev.c:962-984 (place_object on gold/gem
    # hits); vendor/nle/src/mkobj.c::place_object inserts the object into
    # level.objects[x][y] at the head of the chain.
    # ------------------------------------------------------------------
    flat_idx = jnp.arange(_SCAN_N, dtype=jnp.int32)
    xis = flat_idx // _SCAN_H + jnp.int32(_X_LO)
    yis = flat_idx %  _SCAN_H + jnp.int32(_Y_LO)

    # --- Gold scatter ---
    gold_cat_writes = jnp.where(
        gold_place, jnp.int8(int(ItemCategory.COIN)), jnp.int8(0)
    )
    gold_typ_writes = jnp.where(
        gold_place, jnp.int16(int(_OTYP_GOLD_PIECE)), jnp.int16(0)
    )
    gold_qty_writes = jnp.where(
        gold_place,
        jnp.clip(gold_qty, 0, jnp.iinfo(jnp.int16).max).astype(jnp.int16),
        jnp.int16(0),
    )

    # Use ``mode='drop'`` semantics implicitly: each (yis, xis) is unique so
    # .at[].set is fine.  Only WRITE where gold_place is True — for
    # non-place cells, set() back to the current value to avoid corrupting
    # other writers (rooms.py's mkobj_at scatter ran earlier and may have
    # written to slot 7 of a room cell; mineralize cells are STONE so room
    # cells never collide, but be defensive).
    existing_cat = gi_category[branch_idx, level_idx, yis, xis, _GOLD_STACK_SLOT]
    existing_typ = gi_type_id [branch_idx, level_idx, yis, xis, _GOLD_STACK_SLOT]
    existing_qty = gi_quantity[branch_idx, level_idx, yis, xis, _GOLD_STACK_SLOT]
    gold_cat_writes = jnp.where(gold_place, gold_cat_writes, existing_cat)
    gold_typ_writes = jnp.where(gold_place, gold_typ_writes, existing_typ)
    gold_qty_writes = jnp.where(gold_place, gold_qty_writes, existing_qty)
    gi_category = gi_category.at[
        branch_idx, level_idx, yis, xis, _GOLD_STACK_SLOT
    ].set(gold_cat_writes)
    gi_type_id = gi_type_id.at[
        branch_idx, level_idx, yis, xis, _GOLD_STACK_SLOT
    ].set(gold_typ_writes)
    gi_quantity = gi_quantity.at[
        branch_idx, level_idx, yis, xis, _GOLD_STACK_SLOT
    ].set(gold_qty_writes)

    # --- Gem scatter (one stack slot per gem index 0.._MAX_GEMS-1) ---
    for slot in range(_MAX_GEMS):
        place_slot = gem_place[:, slot]
        otyp_slot  = gem_otyp[:, slot]
        qty_slot   = gem_qty[:, slot]

        cat_w = jnp.where(
            place_slot, jnp.int8(int(ItemCategory.GEM)), jnp.int8(0)
        )
        typ_w = jnp.where(
            place_slot,
            jnp.clip(otyp_slot, 0, jnp.iinfo(jnp.int16).max).astype(jnp.int16),
            jnp.int16(0),
        )
        qty_w = jnp.where(
            place_slot,
            jnp.clip(qty_slot, 0, jnp.iinfo(jnp.int16).max).astype(jnp.int16),
            jnp.int16(0),
        )
        exist_cat = gi_category[branch_idx, level_idx, yis, xis, slot]
        exist_typ = gi_type_id [branch_idx, level_idx, yis, xis, slot]
        exist_qty = gi_quantity[branch_idx, level_idx, yis, xis, slot]
        cat_w = jnp.where(place_slot, cat_w, exist_cat)
        typ_w = jnp.where(place_slot, typ_w, exist_typ)
        qty_w = jnp.where(place_slot, qty_w, exist_qty)
        gi_category = gi_category.at[
            branch_idx, level_idx, yis, xis, slot
        ].set(cat_w)
        gi_type_id = gi_type_id.at[
            branch_idx, level_idx, yis, xis, slot
        ].set(typ_w)
        gi_quantity = gi_quantity.at[
            branch_idx, level_idx, yis, xis, slot
        ].set(qty_w)

    return vrng_out, gi_category, gi_type_id, gi_quantity
