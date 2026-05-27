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

    # C2: lit_B = rn2(77)  -- only when lit_a_pass.  Short-circuit
    # via ``lax.cond``: untaken branch returns the RNG unchanged + a
    # sentinel-zero so the downstream ``(lit_a_pass & (lit_b == 0))``
    # test still works (when ``lit_a_pass`` is False the value is
    # masked out anyway).
    def _draw_b(r):
        return rn2_jax(r, jnp.int32(77))

    def _skip_b(r):
        return r, jnp.int32(0)

    rng, lit_b = lax.cond(lit_a_pass, _draw_b, _skip_b, rng)

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

    Simplification vs vendor:
        Vendor's ``goto chk`` shrinks the candidate room on a non-zero
        ``rn2(3)`` roll and restarts the scan from the new bounds; the
        post-shrink cells consume *more* RNG bytes.  We model the linear
        single-pass scan only -- one ``rn2(3)`` per non-stone cell, exit on
        the first ``0`` roll.  This captures the dominant per-attempt draw
        count; the shrink+retry refinement (additional ~tens of bytes when
        a roll is non-zero) is a future polish item.  See audit doc for
        the impact estimate.

    JIT shape: the scan iterates over the full padded box
    ``[lowx-xlim_max .. hix+xlim_max] x [lowy-ylim_max .. hiy+ylim_max]``
    -- both ranges are bounded by COLNO/ROWNO so we use a static-sized
    ``fori_loop`` over ``_COLNO x _ROWNO`` cells and mask out cells outside
    the per-attempt bounding box via ``jnp.where``.  Vendor's clamps
    (``x <= 0 || x >= COLNO`` and ``y < 0`` / ``ymax >= ROWNO``) are folded
    into the same mask.

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
        ``(new_rng, check_ok)``.  ``check_ok`` is True iff every non-stone
        cell in the scan window rolled a non-zero ``rn2(3)`` (or there were
        no non-stone cells).
    """
    xlim = jnp.int16(_XLIM) + jnp.where(vault, jnp.int16(1), jnp.int16(0))
    ylim = jnp.int16(_YLIM) + jnp.where(vault, jnp.int16(1), jnp.int16(0))

    # Padded scan window (pre-clamp -- the per-cell mask applies vendor's
    # ``x <= 0 || x >= COLNO`` / ``y < 0`` / ``ymax >= ROWNO`` clamps).
    scan_lowx = lowx - xlim
    scan_lowy = lowy - ylim
    scan_hix = hix + xlim
    scan_hiy = hiy + ylim

    n_cells = _COLNO * _ROWNO

    def _body(i, carry):
        rng_c, ok_c = carry
        # Cell coordinates -- iterate y inner (matches vendor's ``lev++``
        # which walks down the column).
        x = jnp.int16(i // _ROWNO)
        y = jnp.int16(i % _ROWNO)

        # Vendor mask: ``lowx-xlim <= x <= hix+xlim`` AND ``1 <= x < COLNO``
        # (vendor: ``if (x <= 0 || x >= COLNO) continue;``) AND
        # ``max(0, lowy-ylim) <= y <= min(ROWNO-1, hiy+ylim)``.
        in_x = (
            (x >= scan_lowx)
            & (x <= scan_hix)
            & (x > jnp.int16(0))
            & (x < jnp.int16(_COLNO))
        )
        in_y = (
            (y >= scan_lowy)
            & (y <= scan_hiy)
            & (y >= jnp.int16(0))
            & (y < jnp.int16(_ROWNO))
        )
        in_box = enable & in_x & in_y

        cell = level_grid[y, x]
        non_stone = cell != jnp.int8(0)

        # Draw rn2(3) iff (in_box AND non_stone) -- short-circuited via
        # lax.cond so the ISAAC64 stream is consumed only on cells vendor
        # would have hit.
        def _draw(r):
            return rn2_jax(r, jnp.int32(3))

        def _skip(r):
            return r, jnp.int32(0)

        # Once we've already rejected (ok_c=False) vendor would have
        # returned FALSE -- no further draws.  Gate on ok_c too.
        gate = in_box & non_stone & ok_c
        rng_n, roll = lax.cond(gate, _draw, _skip, rng_c)

        # Reject when this cell consumed a draw and rolled 0.
        ok_n = ok_c & ((~gate) | (roll != jnp.int32(0)))
        return (rng_n, ok_n)

    rng_f, ok_f = lax.fori_loop(0, n_cells, _body, (rng, jnp.bool_(True)))
    return rng_f, ok_f


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
    #     if (vault) dx = dy = 1;
    #     else {
    #         dx = 2 + rn2((hx - lx > 28) ? 12 : 8);
    #         dy = 2 + rn2(4);
    #         if (dx * dy > 50) dy = 50 / dx;
    #     }
    # Short-circuit: vault path skips both draws.  Within !vault, both
    # draws unconditionally happen (no inner short-circuit).
    def _draw_dxdy(carry):
        r = carry
        wide = (p_hx - p_lx) > jnp.int16(28)
        d2_mod = jnp.where(wide, jnp.int32(12), jnp.int32(8))
        r, d2 = rn2_jax(r, d2_mod)
        r, d3 = rn2_jax(r, jnp.int32(4))
        dx = jnp.int16(2) + d2.astype(jnp.int16)
        dy = jnp.int16(2) + d3.astype(jnp.int16)
        # dx*dy > 50 cap (sp_lev.c:1190-1191).
        area = dx.astype(jnp.int32) * dy.astype(jnp.int32)
        dy_capped = jnp.where(
            area > jnp.int32(50),
            (jnp.int32(50) // jnp.maximum(dx.astype(jnp.int32), jnp.int32(1))).astype(jnp.int16),
            dy,
        )
        return r, dx, dy_capped

    def _vault_dxdy(carry):
        r = carry
        return r, jnp.int16(1), jnp.int16(1)

    rng, dx, dy = lax.cond(vault, _vault_dxdy, _draw_dxdy, rng)

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
    def _draw_xyabs(carry):
        r = carry
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

        r, d4 = rn2_jax(r, x_mod)
        r, d5 = rn2_jax(r, y_mod)
        xa = p_lx + x_off + d4.astype(jnp.int16)
        ya = p_ly + y_off + d5.astype(jnp.int16)
        return r, xa, ya

    def _skip_xyabs(carry):
        r = carry
        return r, jnp.int16(0), jnp.int16(0)

    rng, xabs, yabs = lax.cond(can_place, _draw_xyabs, _skip_xyabs, rng)

    # ----- D6, D7: centre-yabs special case -- sp_lev.c:1203-1208 --------
    # Vendor::
    #     if (ly == 0 && hy >= (ROWNO - 1) && (!nroom || !rn2(nroom))
    #         && (yabs + dy > ROWNO / 2)) {
    #         yabs = rn1(3, 2);
    #         if (nroom < 4 && dy > 1) dy--;
    #     }
    #
    # Three nested ``&&`` short-circuits to honour:
    #
    #   gate_outer:  can_place AND (ly == 0) AND (hy >= ROWNO-1)
    #                AND (yabs + dy > ROWNO/2)
    #   gate_d6:     gate_outer AND (nroom != 0)   -- D6 fires here
    #   gate_d7:     gate_outer AND (nroom == 0 OR rn2(nroom) == 0)
    #                                              -- D7 fires here
    #
    # Subtlety: vendor's ``(!nroom || !rn2(nroom))`` is itself a short-
    # circuit -- ``rn2(nroom)`` is *not* drawn when ``nroom == 0`` --
    # so the D6 draw is conditional on ``nroom != 0`` even within the
    # gate_outer ``True`` branch.
    gate_outer = (
        can_place
        & (p_ly == jnp.int16(0))
        & (p_hy >= jnp.int16(_ROWNO - 1))
        & ((yabs + dy) > jnp.int16(_ROWNO // 2))
    )
    nroom_pos = nroom > jnp.int32(0)

    # D6 fires iff gate_outer AND nroom > 0.
    def _draw_d6(carry):
        r = carry
        r, v = rn2_jax(r, jnp.maximum(nroom, jnp.int32(1)))
        return r, v

    def _skip_d6(carry):
        r = carry
        return r, jnp.int32(0)

    rng, d6_val = lax.cond(
        gate_outer & nroom_pos, _draw_d6, _skip_d6, rng
    )

    # The full predicate that triggers D7 (rn1(3, 2)) is::
    #     gate_outer AND ((nroom == 0) OR (d6_val == 0))
    d7_pred = gate_outer & ((~nroom_pos) | (d6_val == jnp.int32(0)))

    # D7: rn1(3, 2) -- sp_lev.c:1205.  Fires only when d7_pred is True.
    def _draw_d7(carry):
        r = carry
        r, v = rn1_jax(r, jnp.int32(3), jnp.int32(2))
        return r, v

    def _skip_d7(carry):
        r = carry
        return r, jnp.int32(0)

    rng, d7_val = lax.cond(d7_pred, _draw_d7, _skip_d7, rng)

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
    # ------------------------------------------------------------------
    rng, rlit = _draw_rlit(rng, depth)

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

        def _do_attempt(state):
            r, p = state
            res = _try_one_attempt(r, p, level_grid, nroom, vault)
            return res

        def _skip_attempt(state):
            r, p = state
            # No RNG draw, no pool mutation -- vendor has already broken
            # out of the do-while.  Return a zero-success sentinel.
            return AttemptResult(
                rng=r,
                pool=p,
                success=jnp.bool_(False),
                xabs=jnp.int16(0), yabs=jnp.int16(0),
                wtmp=jnp.int16(0), htmp=jnp.int16(0),
                r2_lx=jnp.int16(0), r2_ly=jnp.int16(0),
                r2_hx=jnp.int16(0), r2_hy=jnp.int16(0),
                parent_lx=jnp.int16(0), parent_ly=jnp.int16(0),
                parent_hx=jnp.int16(0), parent_hy=jnp.int16(0),
            )

        res = lax.cond(done, _skip_attempt, _do_attempt, (rng_c, pool_c))

        # If this attempt succeeded (and we hadn't already finished),
        # commit the split_rects call (sp_lev.c:1281) and latch the
        # room coords.
        just_won = res.success & (~done)

        def _commit_split(p):
            return split_rects(
                p,
                res.parent_lx, res.parent_ly,
                res.parent_hx, res.parent_hy,
                res.r2_lx, res.r2_ly,
                res.r2_hx, res.r2_hy,
            )

        def _skip_split(p):
            return p

        pool_n = lax.cond(just_won, _commit_split, _skip_split, res.pool)

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
