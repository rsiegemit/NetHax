"""Vendor-exact port of NetHack's ``sp_lev.c::create_room``.

Purpose:
    JIT-pure 1:1 port of the per-attempt room-placement loop in
    ``vendor/nle/src/sp_lev.c`` lines 1126-1292 (Phase 2 of the mklev.c
    port; see ``MKLEV_PORT_PLAN.md`` \xa71.4.1).  Only the *random* path
    (``xtmp/ytmp/wtmp/xaltmp/yaltmp all < 0`` OR ``vault``) is ported here
    -- the parametric/positioned branch (sp_lev.c:1219-1276) is reserved
    for special-level loading and not exercised by ``makerooms``.

Scope (random path, sp_lev.c:1172-1218):

    1. Pre-loop lit draws (sp_lev.c:1153-1154)::

           rlit = (rnd(1 + abs(depth)) < 11 && rn2(77)) ? TRUE : FALSE;

       Two RNG draws guarded by C's ``&&`` short-circuit: the ``rn2(77)``
       RHS fires *only* when ``rnd(1+abs(depth)) < 11`` is true.  Drawing
       unconditionally would shift every downstream draw by one slot --
       the bug class we already shipped twice (commits ``a783700`` /
       ``85ef963``, both reverted).  We honour the short-circuit with
       ``lax.cond``.

    2. Per-attempt loop body (sp_lev.c:1161-1277, ``trycnt <= 100``):

       =====  =======================================  ============
       Draw   Vendor expression                         Cite
       =====  =======================================  ============
       D1     ``rnd_rect()`` -> ``rn2(rect_cnt)``       sp_lev.c:1175
       D2     ``rn2((hx-lx > 28) ? 12 : 8)``           sp_lev.c:1188
       D3     ``rn2(4)``                               sp_lev.c:1189
       D4     ``rn2(hx - (lx>0?lx:3) - dx - xb + 1)``  sp_lev.c:1200
       D5     ``rn2(hy - (ly>0?ly:2) - dy - yb + 1)``  sp_lev.c:1202
       D6     ``rn2(nroom)``                           sp_lev.c:1203
       D7     ``rn1(3, 2)``                            sp_lev.c:1205
       =====  =======================================  ============

       Short-circuit gates:

       - D2/D3 fire only when ``!vault``; vault sets ``dx = dy = 1`` w/o
         drawing (sp_lev.c:1185-1186).
       - D4/D5 fire only when the rect-fits test (sp_lev.c:1195) passes;
         otherwise vendor ``continue``-s to the next attempt without
         further draws.
       - D6 is part of ``(!nroom || !rn2(nroom))``: the ``rn2(nroom)``
         RHS fires only when ``nroom != 0`` AND when ``ly == 0 &&
         hy >= ROWNO-1`` (the outer ``&&`` chain at sp_lev.c:1203).
       - D7 fires only when the full D6 conditional succeeds *and*
         ``yabs + dy > ROWNO/2``.

       ``check_room`` (sp_lev.c:1063-1120) scans the area around the
       candidate for non-stone cells; on a freshly-stoned makerooms
       level every cell has ``levl[x][y].typ == 0``, so the inner
       ``rn2(3)`` (sp_lev.c:1103) is never reached.  We mirror only the
       coordinate-clamp path here (no RNG).  Special-level loading,
       which *does* call create_room on a non-blank level, is out of
       scope for Phase 2.

Phase 3 integration (makerooms):
    The :func:`create_room_random` entry point takes the running
    ``Isaac64State`` and ``RectPool``, plus the host-side scalars
    ``depth``, ``nroom``, and ``vault``, and returns the post-call RNG,
    the post-``split_rects`` pool, a ``success`` flag, and the placed
    room's ``(xabs, yabs, wtmp, htmp, rlit)``.  Phase 3 calls this from
    its outer ``while (rnd_rect()) makerooms`` loop and writes the
    result into the level's room array.

Citation:
    vendor/nle/src/sp_lev.c:1126-1292  -- create_room()
    vendor/nle/src/sp_lev.c:1063-1120  -- check_room()
    vendor/nle/src/rect.c:88-92        -- rnd_rect()
    vendor/nle/include/global.h:327-328 -- COLNO=80, ROWNO=21
    MKLEV_PORT_PLAN.md \xa71.4.1            -- the 7+2 per-attempt draw table
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax

from Nethax.nethax.dungeon.rect_pool import (
    RectPool,
    rnd_rect,
    split_rects,
)
from Nethax.nethax.vendor_rng import Isaac64State, rn2_jax, rnd_jax, rn1_jax


# ---------------------------------------------------------------------------
# Vendor constants
# ---------------------------------------------------------------------------

# vendor/nle/include/global.h:327-328
_COLNO: int = 80
_ROWNO: int = 21

# vendor/nle/src/rect.c:17-18 -- XLIM/YLIM strip-padding.
# Vault path bumps both by 1 (sp_lev.c:1145-1146).
_XLIM: int = 4
_YLIM: int = 3

# vendor/nle/src/sp_lev.c:1161, 1218 -- ``do { ... } while (++trycnt <= 100);``.
# trycnt starts at 0 and the post-increment ``++trycnt`` runs *before* the
# ``<= 100`` test, so the body executes for trycnt values 0..100 inclusive --
# 101 iterations total.  Our previous value of 100 was off-by-one (~21 RNG
# bytes/level deficit).  Restoring vendor parity per sp_lev.c:1175,1218.
_MAX_TRYCNT: int = 101


# ---------------------------------------------------------------------------
# Result pytrees
# ---------------------------------------------------------------------------


class AttemptResult(NamedTuple):
    """Per-attempt outcome of one trip through the create_room do-while body.

    ``success`` mirrors vendor's ``r1 != 0`` exit condition at sp_lev.c:1277.
    On failure, the coordinate fields are zero (callers must mask on
    ``success``).  ``rect_*`` carries the parent rect from D1 so the caller
    can ``split_rects(parent, child)`` once the room is finally accepted.
    """

    rng: Isaac64State
    pool: RectPool
    success: jax.Array         # bool scalar
    xabs: jax.Array            # int16
    yabs: jax.Array            # int16
    wtmp: jax.Array            # int16  (room width incl. wall)
    htmp: jax.Array            # int16
    # r2 child rect for split_rects (vendor sp_lev.c:1215-1218):
    #   r2.lx = xabs - 1; r2.ly = yabs - 1;
    #   r2.hx = xabs + wtmp; r2.hy = yabs + htmp;
    r2_lx: jax.Array           # int16
    r2_ly: jax.Array           # int16
    r2_hx: jax.Array           # int16
    r2_hy: jax.Array           # int16
    # Parent rect (D1 result) so caller can pass it to split_rects.
    parent_lx: jax.Array       # int16
    parent_ly: jax.Array       # int16
    parent_hx: jax.Array       # int16
    parent_hy: jax.Array       # int16


class CreateRoomResult(NamedTuple):
    """Final outcome of one ``create_room`` call.

    Vendor's ``create_room`` returns ``boolean`` and side-effects the
    global ``rooms[nroom]`` + ``smeq[nroom]`` via ``add_room`` (sp_lev.c:
    1283-1290).  Phase 3 handles the room-array write; we just return the
    placed room's coordinates and the post-``split_rects`` pool / RNG.
    """

    rng: Isaac64State
    pool: RectPool
    success: jax.Array         # bool scalar
    xabs: jax.Array            # int16  -- room's leftmost interior column
    yabs: jax.Array            # int16  -- room's topmost interior row
    wtmp: jax.Array            # int16  -- room width (xabs..xabs+wtmp-1)
    htmp: jax.Array            # int16  -- room height
    rlit: jax.Array            # bool   -- room is lit


# ---------------------------------------------------------------------------
# Lit pre-loop draw -- vendor sp_lev.c:1153-1154
# ---------------------------------------------------------------------------


def _draw_rlit(
    rng: Isaac64State, depth: jax.Array
) -> Tuple[Isaac64State, jax.Array]:
    """Pre-loop lit-state draw (sp_lev.c:1153-1154).

    Vendor::

        if (rlit == -1)
            rlit = (rnd(1 + abs(depth(&u.uz))) < 11 && rn2(77)) ? TRUE : FALSE;

    Two draws separated by C's ``&&`` short-circuit -- the ``rn2(77)``
    RHS fires only when ``rnd(1+abs(depth)) < 11`` is true.

    Returns ``(new_rng, rlit_bool)``.
    """
    abs_depth = jnp.abs(depth).astype(jnp.int32)
    # C1: lit_A = rnd(1 + abs(depth))    -- always.
    rng, lit_a = rnd_jax(rng, jnp.int32(1) + abs_depth)
    lit_a_pass = lit_a < jnp.int32(11)

    # C2: lit_B = rn2(77)  -- only when lit_a_pass.  Brax-flatten the
    # short-circuit: compute both branches and select via ``jnp.where``.
    # Under vmap both branches already execute, so this is byte-identical
    # to ``lax.cond``.
    rng_draw, lit_b_draw = rn2_jax(rng, jnp.int32(77))
    rng_skip, lit_b_skip = rng, jnp.int32(0)
    rng = jax.tree_util.tree_map(
        lambda t, f: jnp.where(lit_a_pass, t, f), rng_draw, rng_skip
    )
    lit_b = jnp.where(lit_a_pass, lit_b_draw, lit_b_skip)

    # Vendor C: ``(... && rn2(77)) ? TRUE : FALSE``.  ``rn2(77)``
    # returns 0..76; the expression is TRUE iff *both* sub-tests are
    # nonzero/true -- i.e. ``lit_a < 11`` AND ``lit_b != 0``.
    rlit = lit_a_pass & (lit_b != jnp.int32(0))
    return rng, rlit


# ---------------------------------------------------------------------------
# Per-attempt loop body -- vendor sp_lev.c:1172-1218 (random/vault path).
# ---------------------------------------------------------------------------


def _scan_check_room(
    rng: Isaac64State,
    level_grid: jax.Array,   # int8[_ROWNO, _COLNO] -- 0 = STONE
    lowx: jax.Array,         # int16, post-clamp
    lowy: jax.Array,         # int16, post-clamp
    hix: jax.Array,          # int16, post-clamp
    hiy: jax.Array,          # int16, post-clamp
    vault: jax.Array,        # bool
    enable: jax.Array,       # bool -- skip entirely (and consume zero RNG) when False
) -> Tuple[Isaac64State, jax.Array]:
    """Vendor ``check_room`` cell scan -- sp_lev.c:1099-1113.

    Vendor (sp_lev.c::check_room) scans the padded bounding box
    ``[lowx-xlim .. hix+xlim] x [lowy-ylim .. hiy+ylim]`` (with x clamped to
    1..COLNO-1 and y clamped to 0..ROWNO-1), and for every non-stone cell
    (``levl[x][y].typ != 0``) draws ``rn2(3)`` and rejects the attempt when
    the roll is 0::

        for (x = *lowx - xlim; x <= hix + xlim; x++) {
            if (x <= 0 || x >= COLNO) continue;
            y = *lowy - ylim;
            ymax = hiy + ylim;
            if (y < 0) y = 0;
            if (ymax >= ROWNO) ymax = ROWNO - 1;
            lev = &levl[x][y];
            for (; y <= ymax; y++) {
                if (lev++->typ) {
                    if (!rn2(3))
                        return FALSE;
                    ...
                    goto chk;
                }
            }
        }

    On a freshly-stoned level (room 0) every cell is stone, so the inner
    ``rn2(3)`` never fires and the scan is a no-op.  On rooms 1+ the padded
    bounding boxes of previously placed rooms intrude into the candidate's
    scan window and consume RNG bytes -- audit MAKEROOMS_DEFICIT_AUDIT.md
    estimates "hundreds of draws/attempt x 100 attempts x 7 rooms = 1000+
    draws/level".  Prior to this fix that entire stream was skipped.

    Vendor semantics ported here (sp_lev.c:1063-1120):
        The outer ``chk:`` label is a goto-restart that shrinks the
        candidate bounds whenever a non-stone cell rolls a non-zero
        ``rn2(3)`` and restarts the scan.  At most ONE ``rn2(3)`` is
        drawn per outer iteration -- the cell that triggered the
        ``goto chk``.  We carry ``(rng, lowx, lowy, hix, hiy, status)``
        and drive the loop via a fixed-size ``lax.fori_loop`` whose
        upper bound (``_MAX_SHRINK_ITERS``) exceeds the worst-case
        number of restarts.  Each shrink reduces ``hix - lowx`` or
        ``hiy - lowy`` by at least one column/row, so the geometry
        collapses (``hix <= lowx || hiy <= lowy`` -> FALSE) within at
        most ``(hix_init - lowx_init) + (hiy_init - lowy_init)`` iters,
        which is bounded by ``COLNO + ROWNO``.

    JIT shape: outer ``fori_loop(0, _MAX_SHRINK_ITERS, ...)`` over the
    restart count; inside each iter, an inner ``fori_loop`` over
    ``_COLNO * _ROWNO`` cells finds the FIRST (x-major, y-minor) non-stone
    cell in the padded box.  Per-cell test mirrors vendor's clamp gates
    (``x > 0 && x < COLNO`` and ``y in [max(0,lowy-ylim), min(ROWNO-1,
    hiy+ylim)]``).

    Args:
        rng: ISAAC64 stream.
        level_grid: int8[_ROWNO, _COLNO] where cell == 0 means STONE
            (vendor ``levl[x][y].typ == 0``).  Indexed ``[y, x]``.
        lowx, lowy, hix, hiy: post-clamp interior bounds (sp_lev.c:1075-1082).
        vault: bumps xlim/ylim by 1 (sp_lev.c:1072-1073).
        enable: when False, skip the scan entirely (no RNG consumed).  Used
            when the attempt has already failed an earlier gate (``has_rect
            & fits``) so vendor would never have reached ``check_room``.

    Returns:
        ``(new_rng, check_ok)``.  ``check_ok`` is True iff the scan
        terminates with status == ACCEPTED (no non-stone cell remaining in
        the shrunken box).  Returns False on REJECTED (rolled 0) or on the
        ``hix<=lowx||hiy<=lowy`` collapse.
    """
    xlim = jnp.int16(_XLIM) + jnp.where(vault, jnp.int16(1), jnp.int16(0))
    ylim = jnp.int16(_YLIM) + jnp.where(vault, jnp.int16(1), jnp.int16(0))

    # Status codes: 0=SCANNING, 1=ACCEPTED, 2=REJECTED.
    SCANNING = jnp.int8(0)
    ACCEPTED = jnp.int8(1)
    REJECTED = jnp.int8(2)

    # Pre-collapse check (vendor sp_lev.c:1084 at the first ``chk:`` entry).
    init_status = jnp.where(
        ~enable,
        REJECTED,
        jnp.where(
            (hix <= lowx) | (hiy <= lowy),
            REJECTED,
            SCANNING,
        ),
    )

    n_cells = _COLNO * _ROWNO

    def _outer_body(_i, carry):
        rng_c, lowx_c, lowy_c, hix_c, hiy_c, status_c = carry

        def _do_iter(state):
            r, lx, ly, hx, hy, _st = state

            # Padded scan window for this iteration (sp_lev.c:1088-1096).
            scan_lowx = lx - xlim
            scan_lowy = ly - ylim
            scan_hix = hx + xlim
            scan_hiy = hy + ylim

            # Inner scan: find the FIRST (x-major, y-minor) non-stone cell
            # in the padded box.  Cell index ``i`` maps to ``x = i//_ROWNO,
            # y = i%_ROWNO`` so iteration order is x outer, y inner -- this
            # matches vendor's ``for (x = ...) for (y = ...) lev++``.
            def _inner_body(j, inner_carry):
                found_c, idx_c = inner_carry
                xj = jnp.int16(j // _ROWNO)
                yj = jnp.int16(j % _ROWNO)
                in_x = (
                    (xj >= scan_lowx)
                    & (xj <= scan_hix)
                    & (xj > jnp.int16(0))
                    & (xj < jnp.int16(_COLNO))
                )
                in_y = (
                    (yj >= scan_lowy)
                    & (yj <= scan_hiy)
                    & (yj >= jnp.int16(0))
                    & (yj < jnp.int16(_ROWNO))
                )
                cell = level_grid[yj, xj]
                hit = in_x & in_y & (cell != jnp.int8(0))
                # Latch the first hit -- subsequent hits are ignored.
                take = hit & (~found_c)
                idx_n = jnp.where(take, jnp.int32(j), idx_c)
                found_n = found_c | hit
                return (found_n, idx_n)

            found, idx = lax.fori_loop(
                0,
                n_cells,
                _inner_body,
                (jnp.bool_(False), jnp.int32(-1)),
            )

            # Brax-flatten: compute both _no_hit and _has_hit branches and
            # select via ``jnp.where`` on ``found``.
            # _no_hit branch: ACCEPTED, no RNG draw, bounds unchanged.
            no_hit_r, no_hit_lx, no_hit_ly, no_hit_hx, no_hit_hy = r, lx, ly, hx, hy
            no_hit_status = ACCEPTED

            # _has_hit branch: draw rn2(3) + shrink bounds.
            has_hit_r, roll = rn2_jax(r, jnp.int32(3))
            rejected = roll == jnp.int32(0)
            xh = jnp.int16(idx // _ROWNO)
            yh = jnp.int16(idx % _ROWNO)
            # Vendor shrink (sp_lev.c:1105-1112):
            #   if (x < *lowx) *lowx = x + xlim + 1;
            #   else           hix   = x - xlim - 1;
            #   if (y < *lowy) *lowy = y + ylim + 1;
            #   else           hiy   = y - ylim - 1;
            has_hit_lx = jnp.where(xh < lx, xh + xlim + jnp.int16(1), lx)
            has_hit_hx = jnp.where(xh < lx, hx, xh - xlim - jnp.int16(1))
            has_hit_ly = jnp.where(yh < ly, yh + ylim + jnp.int16(1), ly)
            has_hit_hy = jnp.where(yh < ly, hy, yh - ylim - jnp.int16(1))
            collapsed = (has_hit_hx <= has_hit_lx) | (has_hit_hy <= has_hit_ly)
            has_hit_status = jnp.where(
                rejected,
                REJECTED,
                jnp.where(collapsed, REJECTED, SCANNING),
            )

            iter_r = jax.tree_util.tree_map(
                lambda t, f: jnp.where(found, t, f), has_hit_r, no_hit_r
            )
            iter_lx = jnp.where(found, has_hit_lx, no_hit_lx)
            iter_ly = jnp.where(found, has_hit_ly, no_hit_ly)
            iter_hx = jnp.where(found, has_hit_hx, no_hit_hx)
            iter_hy = jnp.where(found, has_hit_hy, no_hit_hy)
            iter_status = jnp.where(found, has_hit_status, no_hit_status)
            return iter_r, iter_lx, iter_ly, iter_hx, iter_hy, iter_status

        # Brax-flatten the outer SCANNING gate: compute _do_iter result and
        # select against the unchanged carry via ``jnp.where``.
        do_r, do_lx, do_ly, do_hx, do_hy, do_status = _do_iter(
            (rng_c, lowx_c, lowy_c, hix_c, hiy_c, status_c)
        )
        is_scanning = status_c == SCANNING
        out_rng = jax.tree_util.tree_map(
            lambda t, f: jnp.where(is_scanning, t, f), do_r, rng_c
        )
        out_lx = jnp.where(is_scanning, do_lx, lowx_c)
        out_ly = jnp.where(is_scanning, do_ly, lowy_c)
        out_hx = jnp.where(is_scanning, do_hx, hix_c)
        out_hy = jnp.where(is_scanning, do_hy, hiy_c)
        out_status = jnp.where(is_scanning, do_status, status_c)
        return out_rng, out_lx, out_ly, out_hx, out_hy, out_status

    # Worst-case restart count: each shrink loses >=1 row or column in the
    # interior box.  Initial spans are bounded by COLNO + ROWNO = 101.
    _MAX_SHRINK_ITERS = _COLNO + _ROWNO

    rng_f, _, _, _, _, status_f = lax.fori_loop(
        0,
        _MAX_SHRINK_ITERS,
        _outer_body,
        (rng, lowx, lowy, hix, hiy, init_status),
    )
    return rng_f, status_f == ACCEPTED


def _try_one_attempt(
    rng: Isaac64State,
    pool: RectPool,
    level_grid: jax.Array,    # int8[_ROWNO, _COLNO]
    nroom: jax.Array,        # int32
    vault: jax.Array,        # bool
) -> AttemptResult:
    """Execute one trip through the random-path do-while body.

    Implements the vendor draw sequence ``D1 -> D2 -> D3 -> D4 -> D5
    -> (D6 -> D7)?`` honouring every short-circuit so that the ISAAC64
    stream consumed matches a vendor C run with identical inputs.

    Returns an :class:`AttemptResult` with ``success=True`` iff
    ``rnd_rect`` returned a rect *and* the rect-fits test passed *and*
    ``check_room`` would have returned TRUE.
    """
    # xlim/ylim from sp_lev.c:1138 + vault bump on :1145-1146.
    xlim = jnp.int16(_XLIM) + jnp.where(vault, jnp.int16(1), jnp.int16(0))
    ylim = jnp.int16(_YLIM) + jnp.where(vault, jnp.int16(1), jnp.int16(0))

    # ----- D1: rnd_rect() -- sp_lev.c:1175 ---------------------------------
    # Vendor: ``r1 = rnd_rect();  if (!r1) return FALSE;``
    # Our rnd_rect honours the empty-pool short-circuit internally
    # (returns rng unchanged when pool empty).
    pool, p_lx, p_ly, p_hx, p_hy, rng, has_rect = rnd_rect(pool, rng)
    p_lx = p_lx.astype(jnp.int16)
    p_ly = p_ly.astype(jnp.int16)
    p_hx = p_hx.astype(jnp.int16)
    p_hy = p_hy.astype(jnp.int16)

    # ----- D2, D3: dx, dy -- sp_lev.c:1185-1192 ---------------------------
    # Vendor::
    #     r1 = rnd_rect();
    #     if (!r1) {                   /* No more free rectangles! */
    #         debugpline0("No more rects...");
    #         return FALSE;            /* hard return from create_room */
    #     }
    #     ...
    #     if (vault) dx = dy = 1;
    #     else {
    #         dx = 2 + rn2((hx - lx > 28) ? 12 : 8);
    #         dy = 2 + rn2(4);
    #         if (dx * dy > 50) dy = 50 / dx;
    #     }
    # Two short-circuits to honour:
    #   (a) ``has_rect == False``  -> vendor's ``if (!r1) return FALSE``
    #       at sp_lev.c:1177-1180 exits create_room entirely *before*
    #       reaching the dx/dy assignment.  D2/D3 must NOT fire.
    #   (b) ``vault == True``      -> vendor sp_lev.c:1185-1186 sets
    #       ``dx = dy = 1`` without drawing.  D2/D3 must NOT fire.
    # Previously D2/D3 fired whenever ``!vault`` even on the empty-pool
    # branch -- one or two extra ISAAC64 bytes per stalled attempt.
    # Citation: vendor/nle/src/sp_lev.c:1175-1192.
    # Brax-flatten the nested has_rect / vault dx,dy short-circuits.
    # _draw_dxdy branch (has_rect & ~vault): consume D2 + D3.
    wide = (p_hx - p_lx) > jnp.int16(28)
    d2_mod = jnp.where(wide, jnp.int32(12), jnp.int32(8))
    rng_draw, d2 = rn2_jax(rng, d2_mod)
    rng_draw, d3 = rn2_jax(rng_draw, jnp.int32(4))
    draw_dx = jnp.int16(2) + d2.astype(jnp.int16)
    draw_dy_raw = jnp.int16(2) + d3.astype(jnp.int16)
    # dx*dy > 50 cap (sp_lev.c:1190-1191).
    area = draw_dx.astype(jnp.int32) * draw_dy_raw.astype(jnp.int32)
    draw_dy = jnp.where(
        area > jnp.int32(50),
        (jnp.int32(50) // jnp.maximum(draw_dx.astype(jnp.int32), jnp.int32(1))).astype(jnp.int16),
        draw_dy_raw,
    )

    # _vault_dxdy branch (has_rect & vault): no RNG, dx=dy=1.
    rng_vault, vault_dx, vault_dy = rng, jnp.int16(1), jnp.int16(1)

    # _skip_dxdy branch (~has_rect): no RNG, dx=dy=0.
    rng_skip, skip_dx, skip_dy = rng, jnp.int16(0), jnp.int16(0)

    # Honour both vault and no-rect short-circuits via nested where.
    inner_rng = jax.tree_util.tree_map(
        lambda t, f: jnp.where(vault, t, f), rng_vault, rng_draw
    )
    inner_dx = jnp.where(vault, vault_dx, draw_dx)
    inner_dy = jnp.where(vault, vault_dy, draw_dy)

    rng = jax.tree_util.tree_map(
        lambda t, f: jnp.where(has_rect, t, f), inner_rng, rng_skip
    )
    dx = jnp.where(has_rect, inner_dx, skip_dx)
    dy = jnp.where(has_rect, inner_dy, skip_dy)

    # ----- Borders -- sp_lev.c:1193-1194 ----------------------------------
    # xborder = (lx > 0 && hx < COLNO - 1) ? 2*xlim : xlim + 1
    # yborder = (ly > 0 && hy < ROWNO - 1) ? 2*ylim : ylim + 1
    xborder = jnp.where(
        (p_lx > jnp.int16(0)) & (p_hx < jnp.int16(_COLNO - 1)),
        jnp.int16(2) * xlim,
        xlim + jnp.int16(1),
    )
    yborder = jnp.where(
        (p_ly > jnp.int16(0)) & (p_hy < jnp.int16(_ROWNO - 1)),
        jnp.int16(2) * ylim,
        ylim + jnp.int16(1),
    )

    # ----- Rect-fits test -- sp_lev.c:1195 --------------------------------
    # if (hx - lx < dx + 3 + xborder || hy - ly < dy + 3 + yborder)
    #     { r1 = 0; continue; }
    fits = ((p_hx - p_lx) >= (dx + jnp.int16(3) + xborder)) & (
        (p_hy - p_ly) >= (dy + jnp.int16(3) + yborder)
    )
    # Combined gate: the rect must exist AND it must fit.  D4/D5 only
    # fire when both are true (short-circuit via lax.cond).
    can_place = has_rect & fits

    # ----- D4, D5: xabs, yabs -- sp_lev.c:1199-1202 -----------------------
    # xabs = lx + (lx > 0 ? xlim : 3)
    #        + rn2(hx - (lx > 0 ? lx : 3) - dx - xborder + 1);
    # yabs = ly + (ly > 0 ? ylim : 2)
    #        + rn2(hy - (ly > 0 ? ly : 2) - dy - yborder + 1);
    # Brax-flatten the can_place gate around D4/D5.  Both branches compute
    # unconditionally; result is selected via ``jnp.where`` on can_place.
    lx_branch = jnp.where(p_lx > jnp.int16(0), p_lx, jnp.int16(3))
    ly_branch = jnp.where(p_ly > jnp.int16(0), p_ly, jnp.int16(2))
    x_off = jnp.where(p_lx > jnp.int16(0), xlim, jnp.int16(3))
    y_off = jnp.where(p_ly > jnp.int16(0), ylim, jnp.int16(2))

    # Modulus must be >=1 even on the untaken branch to keep
    # rn2_jax (uint64 modulo) well-defined under JIT.
    x_mod_raw = (p_hx - lx_branch - dx - xborder + jnp.int16(1)).astype(jnp.int32)
    y_mod_raw = (p_hy - ly_branch - dy - yborder + jnp.int16(1)).astype(jnp.int32)
    x_mod = jnp.maximum(x_mod_raw, jnp.int32(1))
    y_mod = jnp.maximum(y_mod_raw, jnp.int32(1))

    rng_xy, d4 = rn2_jax(rng, x_mod)
    rng_xy, d5 = rn2_jax(rng_xy, y_mod)
    xa_draw = p_lx + x_off + d4.astype(jnp.int16)
    ya_draw = p_ly + y_off + d5.astype(jnp.int16)

    rng = jax.tree_util.tree_map(
        lambda t, f: jnp.where(can_place, t, f), rng_xy, rng
    )
    xabs = jnp.where(can_place, xa_draw, jnp.int16(0))
    yabs = jnp.where(can_place, ya_draw, jnp.int16(0))

    # ----- D6, D7: centre-yabs special case -- sp_lev.c:1203-1208 --------
    # Vendor::
    #     if (ly == 0 && hy >= (ROWNO - 1) && (!nroom || !rn2(nroom))
    #         && (yabs + dy > ROWNO / 2)) {
    #         yabs = rn1(3, 2);
    #         if (nroom < 4 && dy > 1) dy--;
    #     }
    #
    # C ``&&`` is left-associative + strictly left-to-right: the
    # condition is ``(((A && B) && (C||D)) && E)`` where
    #   A = (ly == 0), B = (hy >= ROWNO-1),
    #   C = (!nroom),  D = (!rn2(nroom)),
    #   E = (yabs + dy > ROWNO/2).
    # Evaluation order: A; if A then B; if A&&B then (C||D); if true
    # then E.  Crucially, ``rn2(nroom)`` (D6) is drawn when ``A && B``
    # is true AND ``!C`` (i.e. nroom != 0) -- *before* E is evaluated.
    # ``rn1(3,2)`` (D7) only fires when the *entire* condition is true.
    #
    # Previously we (incorrectly) included E in the D6 draw gate, which
    # under-drew rn2(nroom) on the (rare) attempts where A&&B&&!C held
    # but E was false.  Fixed to drop E from the D6 gate.
    # Citation: vendor/nle/src/sp_lev.c:1203-1208.
    #
    # Also: vendor only reaches this if-block after the rect-fits test
    # at sp_lev.c:1195 passed (otherwise ``continue;`` jumps back to
    # the while-test), so all gates also include ``can_place``.
    gate_ab = (
        can_place
        & (p_ly == jnp.int16(0))
        & (p_hy >= jnp.int16(_ROWNO - 1))
    )
    gate_e = (yabs + dy) > jnp.int16(_ROWNO // 2)
    gate_outer = gate_ab & gate_e
    nroom_pos = nroom > jnp.int32(0)

    # D6 fires iff gate_ab AND nroom > 0 -- E is evaluated AFTER the
    # rn2(nroom) draw in vendor's left-to-right ``&&`` chain.
    # sp_lev.c:1203.  Brax-flatten: compute both branches + jnp.where.
    d6_gate = gate_ab & nroom_pos
    rng_d6, d6_draw = rn2_jax(rng, jnp.maximum(nroom, jnp.int32(1)))
    rng = jax.tree_util.tree_map(
        lambda t, f: jnp.where(d6_gate, t, f), rng_d6, rng
    )
    d6_val = jnp.where(d6_gate, d6_draw, jnp.int32(0))

    # The full predicate that triggers D7 (rn1(3, 2)) is::
    #     gate_ab AND ((nroom == 0) OR (d6_val == 0)) AND gate_e
    d7_pred = gate_ab & ((~nroom_pos) | (d6_val == jnp.int32(0))) & gate_e

    # D7: rn1(3, 2) -- sp_lev.c:1205.  Fires only when d7_pred is True.
    # Brax-flatten: compute both branches + jnp.where.
    rng_d7, d7_draw = rn1_jax(rng, jnp.int32(3), jnp.int32(2))
    rng = jax.tree_util.tree_map(
        lambda t, f: jnp.where(d7_pred, t, f), rng_d7, rng
    )
    d7_val = jnp.where(d7_pred, d7_draw, jnp.int32(0))

    # Apply the centre-yabs override + dy-decrement (sp_lev.c:1205-1207).
    # ``if (nroom < 4 && dy > 1) dy--;`` -- no RNG.
    yabs_override = d7_val.astype(jnp.int16)
    yabs_final = jnp.where(d7_pred, yabs_override, yabs)
    dy_decremented = jnp.where(
        d7_pred & (nroom < jnp.int32(4)) & (dy > jnp.int16(1)),
        dy - jnp.int16(1),
        dy,
    )
    dy_final = dy_decremented

    # ----- check_room -- sp_lev.c:1063-1120 -------------------------------
    # Per-cell scan of the padded bounding box (sp_lev.c:1099-1113).  On
    # rooms 1+ the padded windows of previously placed rooms intersect the
    # scan and consume one ``rn2(3)`` draw per non-stone cell -- audit
    # MAKEROOMS_DEFICIT_AUDIT.md flags this as the top RNG deficit.
    # Pre-scan we apply the vendor clamps on lowx/lowy/hix/hiy
    # (sp_lev.c:1075-1082) and short-circuit when the geometry would
    # collapse (sp_lev.c:1084 ``hix <= lowx || hiy <= lowy`` -> FALSE).
    hix = xabs + dx
    hiy = yabs_final + dy_final
    lowx_clamped = jnp.maximum(xabs, jnp.int16(3))
    lowy_clamped = jnp.maximum(yabs_final, jnp.int16(2))
    hix_clamped = jnp.minimum(hix, jnp.int16(_COLNO - 3))
    hiy_clamped = jnp.minimum(hiy, jnp.int16(_ROWNO - 3))
    geom_ok = (hix_clamped > lowx_clamped) & (hiy_clamped > lowy_clamped)

    # Cell scan -- consumes ``rn2(3)`` per non-stone cell in the padded
    # window.  Gated on ``can_place & geom_ok`` so vendor's earlier exits
    # (no rect / rect-fits FALSE / collapsed geometry) skip the scan
    # without consuming RNG.  See sp_lev.c:1099-1113.
    scan_enable = can_place & geom_ok
    rng, scan_ok = _scan_check_room(
        rng,
        level_grid,
        lowx_clamped,
        lowy_clamped,
        hix_clamped,
        hiy_clamped,
        vault,
        scan_enable,
    )
    check_ok = geom_ok & scan_ok
    final_dx = hix_clamped - lowx_clamped
    final_dy = hiy_clamped - lowy_clamped

    success = can_place & check_ok

    # ----- Compute wtmp/htmp + r2 child rect -- sp_lev.c:1213-1218 -------
    # wtmp = dx + 1; htmp = dy + 1
    # r2.lx = xabs - 1; r2.ly = yabs - 1
    # r2.hx = xabs + wtmp; r2.hy = yabs + htmp
    final_xabs = lowx_clamped
    final_yabs = lowy_clamped
    wtmp = final_dx + jnp.int16(1)
    htmp = final_dy + jnp.int16(1)
    r2_lx = final_xabs - jnp.int16(1)
    r2_ly = final_yabs - jnp.int16(1)
    r2_hx = final_xabs + wtmp
    r2_hy = final_yabs + htmp

    return AttemptResult(
        rng=rng,
        pool=pool,
        success=success,
        xabs=jnp.where(success, final_xabs, jnp.int16(0)),
        yabs=jnp.where(success, final_yabs, jnp.int16(0)),
        wtmp=jnp.where(success, wtmp, jnp.int16(0)),
        htmp=jnp.where(success, htmp, jnp.int16(0)),
        r2_lx=jnp.where(success, r2_lx, jnp.int16(0)),
        r2_ly=jnp.where(success, r2_ly, jnp.int16(0)),
        r2_hx=jnp.where(success, r2_hx, jnp.int16(0)),
        r2_hy=jnp.where(success, r2_hy, jnp.int16(0)),
        parent_lx=p_lx,
        parent_ly=p_ly,
        parent_hx=p_hx,
        parent_hy=p_hy,
    )


# ---------------------------------------------------------------------------
# Top-level entry point -- vendor sp_lev.c:1126-1292 (random path).
# ---------------------------------------------------------------------------


def create_room_random(
    rng: Isaac64State,
    pool: RectPool,
    depth: jax.Array,    # int32 scalar (current dungeon depth, signed)
    nroom: jax.Array,    # int32 scalar (current room count, before insert)
    vault: jax.Array,    # bool scalar
    level_grid: jax.Array | None = None,  # int8[_ROWNO, _COLNO]
) -> CreateRoomResult:
    """Port of ``create_room`` for the random/vault path (sp_lev.c:1172-1218).

    Args:
        rng:    ISAAC64 stream (vendor-exact).
        pool:   :class:`RectPool` -- mutated via ``rnd_rect`` (D1) and a
                final ``split_rects`` once the room is accepted.
        depth: signed dungeon depth -- only ``abs(depth)`` matters
                (sp_lev.c:1154 ``rnd(1 + abs(depth(&u.uz)))``).
        nroom:  current room count, used by the D6 ``rn2(nroom)`` gate
                (sp_lev.c:1203).  Pass ``0`` on the first call.
        vault:  ``True`` iff the caller is placing a vault.  Bumps
                xlim/ylim by 1 (sp_lev.c:1145-1146) and skips D2/D3
                (sp_lev.c:1185-1186).
        level_grid: optional int8[_ROWNO, _COLNO] -- 0 means STONE, non-zero
                means an already-placed room/wall/etc.  Drives the vendor
                ``check_room`` per-cell ``rn2(3)`` scan (sp_lev.c:1099-1113).
                On room 0 (first placement) the grid is all stone and no
                draws fire; rooms 1+ may consume hundreds of bytes here.
                When ``None``, defaults to all-stone (back-compat for the
                Phase-2 smoke test and any caller that hasn't wired a grid
                yet).  Phase 3 ``makerooms`` is expected to maintain the
                grid by stamping each placed room's interior + 1-cell
                wall border to non-stone after success.

    Returns:
        :class:`CreateRoomResult`.  On ``success=True``, the pool has had
        the parent rect split out and ``(xabs, yabs, wtmp, htmp, rlit)``
        identifies the placed room.  On ``success=False`` (100 attempts
        all failed), the pool reflects whatever splits *did* land along
        the way; the room coords are zero.

    Phase 3 (makerooms) integration:
        Phase 3's outer ``while`` loop calls this per slot, then writes
        the returned ``(xabs, yabs, xabs+wtmp-1, yabs+htmp-1, rlit)`` into
        the level's Room array (vendor ``add_room`` at sp_lev.c:1285).
    """
    # Default level_grid to all-stone when caller doesn't pass one (back-
    # compat).  Phase 3's makerooms passes a real grid that reflects every
    # already-placed room's interior + wall-border footprint.
    if level_grid is None:
        level_grid = jnp.zeros((_ROWNO, _COLNO), dtype=jnp.int8)

    # ------------------------------------------------------------------
    # Pre-loop lit draws (sp_lev.c:1153-1154) -- ONCE per create_room.
    #
    # Vendor only draws lit_A/lit_B when ``rlit == -1`` (random) -- the
    # ``create_vault`` macro at mklev.c:38 passes ``rlit = TRUE`` so the
    # vault branch SKIPS these two draws entirely.  Forgetting this skip
    # caused Nethax to consume 2 extra ISAAC64 bytes per vault attempt
    # vs vendor on seed=1 (first divergence: draw 425).  Citation:
    # vendor/nle/src/mklev.c:38, vendor/nle/src/sp_lev.c:1153.
    # ------------------------------------------------------------------
    # Brax-flatten: compute both branches and select via ``jnp.where``.
    # Vault path: rlit is hard-wired TRUE (lit) per create_vault macro.
    rng_draw_rlit, rlit_draw = _draw_rlit(rng, depth)
    rng_skip_rlit, rlit_skip = rng, jnp.bool_(True)
    rng = jax.tree_util.tree_map(
        lambda t, f: jnp.where(vault, t, f), rng_skip_rlit, rng_draw_rlit
    )
    rlit = jnp.where(vault, rlit_skip, rlit_draw)

    # ------------------------------------------------------------------
    # do-while (trycnt <= 100) loop (sp_lev.c:1161-1277).
    #
    # Once an attempt succeeds (success=True) we still must execute the
    # remaining trycnt iterations under JIT for shape-stability, but they
    # must NOT consume any more RNG (vendor would have broken out of the
    # loop).  We gate the entire attempt on ``~done`` via ``lax.cond``.
    # ------------------------------------------------------------------
    def _body(_i, carry):
        rng_c, pool_c, done, x_c, y_c, w_c, h_c = carry

        # Brax-flatten: always run an attempt; select between the attempt
        # result and a zero-success sentinel via ``jnp.where`` on done.
        # Under vmap both branches already executed, so this is byte-
        # identical to the prior ``lax.cond``.
        do_res = _try_one_attempt(rng_c, pool_c, level_grid, nroom, vault)
        skip_res = AttemptResult(
            rng=rng_c,
            pool=pool_c,
            success=jnp.bool_(False),
            xabs=jnp.int16(0), yabs=jnp.int16(0),
            wtmp=jnp.int16(0), htmp=jnp.int16(0),
            r2_lx=jnp.int16(0), r2_ly=jnp.int16(0),
            r2_hx=jnp.int16(0), r2_hy=jnp.int16(0),
            parent_lx=jnp.int16(0), parent_ly=jnp.int16(0),
            parent_hx=jnp.int16(0), parent_hy=jnp.int16(0),
        )
        res = jax.tree_util.tree_map(
            lambda t, f: jnp.where(done, t, f), skip_res, do_res
        )

        # If this attempt succeeded (and we hadn't already finished),
        # commit the split_rects call (sp_lev.c:1281) and latch the
        # room coords.
        just_won = res.success & (~done)

        # Brax-flatten the split_rects gate: compute the split result and
        # the unchanged pool, then select via jnp.where on just_won.
        split_pool = split_rects(
            res.pool,
            res.parent_lx, res.parent_ly,
            res.parent_hx, res.parent_hy,
            res.r2_lx, res.r2_ly,
            res.r2_hx, res.r2_hy,
        )
        pool_n = jax.tree_util.tree_map(
            lambda t, f: jnp.where(just_won, t, f), split_pool, res.pool
        )

        # Latch outputs on the winning attempt; otherwise carry the
        # prior values forward.
        x_n = jnp.where(just_won, res.xabs, x_c)
        y_n = jnp.where(just_won, res.yabs, y_c)
        w_n = jnp.where(just_won, res.wtmp, w_c)
        h_n = jnp.where(just_won, res.htmp, h_c)
        done_n = done | res.success

        return (res.rng, pool_n, done_n, x_n, y_n, w_n, h_n)

    init = (
        rng,
        pool,
        jnp.bool_(False),
        jnp.int16(0), jnp.int16(0),
        jnp.int16(0), jnp.int16(0),
    )
    final = lax.fori_loop(0, _MAX_TRYCNT, _body, init)
    rng_f, pool_f, done_f, xabs_f, yabs_f, wtmp_f, htmp_f = final

    return CreateRoomResult(
        rng=rng_f,
        pool=pool_f,
        success=done_f,
        xabs=xabs_f,
        yabs=yabs_f,
        wtmp=wtmp_f,
        htmp=htmp_f,
        rlit=rlit,
    )


# ---------------------------------------------------------------------------
# Smoke-test (vendor parity quick check)
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    from Nethax.nethax.dungeon.rect_pool import init_rect
    from Nethax.nethax.vendor_rng import init as _rng_init

    pool0 = init_rect()
    rng0 = _rng_init(0)
    res = create_room_random(
        rng=rng0,
        pool=pool0,
        depth=jnp.int32(1),
        nroom=jnp.int32(0),
        vault=jnp.bool_(False),
    )
    print(
        "create_room_random smoke-test:",
        "success=", bool(res.success),
        "xabs=", int(res.xabs),
        "yabs=", int(res.yabs),
        "wtmp=", int(res.wtmp),
        "htmp=", int(res.htmp),
        "rlit=", bool(res.rlit),
        "rect_cnt=", int(res.pool.rect_cnt),
    )
