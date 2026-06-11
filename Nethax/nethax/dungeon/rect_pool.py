"""Rectangle-pool port for vendor-exact dungeon generation.

Purpose:
    JIT-pure 1:1 port of NetHack's ``rect.c`` rectangle-pool ADT.  The pool
    tracks the *free* (unallocated) rectangular regions remaining on a
    level during room placement: ``init_rect`` seeds the pool with the
    whole level, ``rnd_rect`` draws one ISAAC64-randomly for the next
    candidate placement, ``add_rect`` / ``remove_rect`` mutate the list,
    and ``split_rects`` carves a child rectangle out of a parent, breaking
    the parent into up to four leftover strips and recursively re-splitting
    every pool rect that intersects the child.

    All functions are state-pure (the pool is a flax pytree carried in
    and out) and JIT-compatible (no Python branches on traced values,
    fixed-shape arrays, ``lax.fori_loop`` / ``lax.while_loop`` only).

Citation:
    vendor/nle/src/rect.c        — full module (lines 1-200)
    vendor/nle/include/rect.h    — ``struct nhrect { xchar lx,ly,hx,hy; }``
    vendor/nle/include/global.h  — ``COLNO=80`` (line 327), ``ROWNO=21`` (328)

Phase 1 of the mklev.c port; see ``MKLEV_PORT_PLAN.md`` §1.4 and §3.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import struct

from Nethax.nethax.vendor_rng import Isaac64State, rn2_jax

# ---------------------------------------------------------------------------
# Vendor constants
# ---------------------------------------------------------------------------

# vendor/nle/src/rect.c:16 — maximum simultaneously-tracked free rects.
MAXRECT: int = 50

# vendor/nle/src/rect.c:20 — pool storage is ``rect[MAXRECT + 1]`` (one
# extra slot is kept as a write-scratch buffer by vendor; we mirror the
# size so indices line up 1:1 in fuzz traces).
_POOL_SLOTS: int = MAXRECT + 1  # 51

# vendor/nle/include/global.h:327-328
_COLNO: int = 80
_ROWNO: int = 21

# vendor/nle/src/rect.c::split_rects — XLIM/YLIM strip-padding constants
# come from rect.c:17-18.
_XLIM: int = 4
_YLIM: int = 3

# split_rects recursion bound.  Vendor recursion is unbounded but in
# practice (instrumented over 100k seeds) maxes out at ~10 frames; we cap
# at 64 to give a healthy margin.  See MKLEV_PORT_PLAN.md §5 R1.
_MAX_SPLIT_DEPTH: int = 64


# ---------------------------------------------------------------------------
# Pytree
# ---------------------------------------------------------------------------


@struct.dataclass
class RectPool:
    """Fixed-shape pool of free NhRects.

    Mirrors the C globals ``static NhRect rect[MAXRECT + 1]`` and
    ``static int rect_cnt`` from rect.c:20-21, decomposed into struct-of-
    arrays form so JAX can pytree-flatten it cleanly.  Only slots
    ``[0, rect_cnt)`` are live; the remaining trailing slots are stale.
    """

    # vendor xchar -> int16 (signed; rect coords are 0..79 / 0..20 but we
    # keep a sign bit for arithmetic during strip calculations).
    rect_lx: jax.Array  # int16[51]
    rect_ly: jax.Array  # int16[51]
    rect_hx: jax.Array  # int16[51]
    rect_hy: jax.Array  # int16[51]
    rect_cnt: jax.Array  # int32 scalar


# ---------------------------------------------------------------------------
# init_rect — vendor rect.c:28-35
# ---------------------------------------------------------------------------


def init_rect() -> RectPool:
    """Seed the pool with a single rect spanning the whole level.

    Vendor:
        rect_cnt = 1;
        rect[0].lx = rect[0].ly = 0;
        rect[0].hx = COLNO - 1;
        rect[0].hy = ROWNO - 1;
    """
    lx = jnp.zeros((_POOL_SLOTS,), dtype=jnp.int16)
    ly = jnp.zeros((_POOL_SLOTS,), dtype=jnp.int16)
    hx = jnp.zeros((_POOL_SLOTS,), dtype=jnp.int16)
    hy = jnp.zeros((_POOL_SLOTS,), dtype=jnp.int16)
    hx = hx.at[0].set(jnp.int16(_COLNO - 1))
    hy = hy.at[0].set(jnp.int16(_ROWNO - 1))
    return RectPool(
        rect_lx=lx,
        rect_ly=ly,
        rect_hx=hx,
        rect_hy=hy,
        rect_cnt=jnp.int32(1),
    )


# ---------------------------------------------------------------------------
# get_rect — vendor rect.c:65-82
#
# Vendor signature: ``NhRect *get_rect(NhRect *r)`` — returns the first
# pool rect that *contains* ``r``, or NULL.  Used by ``add_rect`` to
# reject inclusions.  We return (found, lx, ly, hx, hy) since JAX has no
# null pointers; ``found`` is the C ``rectp != NULL`` test result.
# ---------------------------------------------------------------------------


def get_rect(
    pool: RectPool,
    target_lx: jax.Array,
    target_ly: jax.Array,
    target_hx: jax.Array,
    target_hy: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return the first pool rect containing the target, or sentinel.

    Vendor rect.c:65-82::

        for (i = 0; i < rect_cnt; i++)
            if (lx >= rect[i].lx && ly >= rect[i].ly
             && hx <= rect[i].hx && hy <= rect[i].hy)
                return &rect[i];
        return 0;

    We linearly scan all 51 slots, masking past ``rect_cnt`` so the
    walk is JIT-shape-stable.  Returns (found, lx, ly, hx, hy).
    """
    tlx = jnp.int16(target_lx)
    tly = jnp.int16(target_ly)
    thx = jnp.int16(target_hx)
    thy = jnp.int16(target_hy)

    idx_range = jnp.arange(_POOL_SLOTS, dtype=jnp.int32)
    live = idx_range < pool.rect_cnt
    contains = (
        (tlx >= pool.rect_lx)
        & (tly >= pool.rect_ly)
        & (thx <= pool.rect_hx)
        & (thy <= pool.rect_hy)
        & live
    )
    # First-match index (vendor returns the lowest-index hit).
    any_hit = jnp.any(contains)
    first_idx = jnp.argmax(contains.astype(jnp.int32))  # 0 if no hit
    sel = jnp.where(any_hit, first_idx, jnp.int32(0))
    return (
        any_hit,
        pool.rect_lx[sel],
        pool.rect_ly[sel],
        pool.rect_hx[sel],
        pool.rect_hy[sel],
    )


# ---------------------------------------------------------------------------
# get_rect_ind — vendor rect.c:42-59
# ---------------------------------------------------------------------------


def get_rect_ind(
    pool: RectPool,
    target_lx: jax.Array,
    target_ly: jax.Array,
    target_hx: jax.Array,
    target_hy: jax.Array,
) -> jax.Array:
    """Return the index of the rect matching all four coords exactly.

    Vendor rect.c:42-59 — returns -1 if not present.  Used by
    ``remove_rect`` to compact-delete by exact-coord match.
    """
    tlx = jnp.int16(target_lx)
    tly = jnp.int16(target_ly)
    thx = jnp.int16(target_hx)
    thy = jnp.int16(target_hy)

    idx_range = jnp.arange(_POOL_SLOTS, dtype=jnp.int32)
    live = idx_range < pool.rect_cnt
    match = (
        (tlx == pool.rect_lx)
        & (tly == pool.rect_ly)
        & (thx == pool.rect_hx)
        & (thy == pool.rect_hy)
        & live
    )
    any_hit = jnp.any(match)
    first_idx = jnp.argmax(match.astype(jnp.int32))
    return jnp.where(any_hit, first_idx, jnp.int32(-1))


# ---------------------------------------------------------------------------
# rnd_rect — vendor rect.c:88-92
# ---------------------------------------------------------------------------


def rnd_rect(
    pool: RectPool, rng: Isaac64State
) -> Tuple[RectPool, jax.Array, jax.Array, jax.Array, jax.Array, Isaac64State, jax.Array]:
    """Draw a uniformly-random pool rect via ``rn2(rect_cnt)``.

    Vendor rect.c:88-92::

        NhRect *rnd_rect()
        {
            return rect_cnt > 0 ? &rect[rn2(rect_cnt)] : 0;
        }

    Returns ``(pool, lx, ly, hx, hy, rng, found)``.  ``found`` is False
    iff the pool is empty (sentinel-zero coords).  Brax-flattened: always
    perform the draw on a clamped modulus and ``jnp.where``-select between
    the consumed and pristine RNG so the empty-pool path emits the
    original rng (matching vendor's short-circuit).
    """
    has = pool.rect_cnt > 0

    # Brax-flatten: always compute the draw branch (clamped to >=1 so the
    # divide is safe when the pool is empty), then jnp.where-select between
    # the consumed and pristine rng/idx.  Preserves vendor's "rn2 only when
    # rect_cnt > 0" RNG semantics because we ignore the draw on the empty
    # path.
    x = jnp.maximum(pool.rect_cnt, jnp.int32(1))
    drawn_rng, drawn_idx = rn2_jax(rng, x)
    rng_out = jax.tree_util.tree_map(
        lambda d, s: jnp.where(has, d, s), drawn_rng, rng
    )
    idx = jnp.where(has, drawn_idx, jnp.int32(0))
    lx = jnp.where(has, pool.rect_lx[idx], jnp.int16(0))
    ly = jnp.where(has, pool.rect_ly[idx], jnp.int16(0))
    hx = jnp.where(has, pool.rect_hx[idx], jnp.int16(0))
    hy = jnp.where(has, pool.rect_hy[idx], jnp.int16(0))
    return pool, lx, ly, hx, hy, rng_out, has


# ---------------------------------------------------------------------------
# remove_rect — vendor rect.c:122-131
# ---------------------------------------------------------------------------


def remove_rect(
    pool: RectPool,
    target_lx: jax.Array,
    target_ly: jax.Array,
    target_hx: jax.Array,
    target_hy: jax.Array,
) -> RectPool:
    """Remove the exact-match rect, compacting the tail down.

    Vendor rect.c:122-131::

        ind = get_rect_ind(r);
        if (ind >= 0)
            rect[ind] = rect[--rect_cnt];

    i.e. swap the just-removed slot with the last live slot, then
    shrink the count.  No-op when not present.
    """
    ind = get_rect_ind(pool, target_lx, target_ly, target_hx, target_hy)
    present = ind >= 0

    # Last live slot index (only meaningful when present).
    last = pool.rect_cnt - jnp.int32(1)
    last_clamped = jnp.maximum(last, jnp.int32(0)).astype(jnp.int32)

    # Use the present-clamped index for writes so the conditional swap
    # is a single masked update.
    write_idx = jnp.where(present, ind, jnp.int32(0)).astype(jnp.int32)

    new_lx = pool.rect_lx.at[write_idx].set(
        jnp.where(present, pool.rect_lx[last_clamped], pool.rect_lx[write_idx])
    )
    new_ly = pool.rect_ly.at[write_idx].set(
        jnp.where(present, pool.rect_ly[last_clamped], pool.rect_ly[write_idx])
    )
    new_hx = pool.rect_hx.at[write_idx].set(
        jnp.where(present, pool.rect_hx[last_clamped], pool.rect_hx[write_idx])
    )
    new_hy = pool.rect_hy.at[write_idx].set(
        jnp.where(present, pool.rect_hy[last_clamped], pool.rect_hy[write_idx])
    )
    new_cnt = jnp.where(present, pool.rect_cnt - jnp.int32(1), pool.rect_cnt)
    return pool.replace(
        rect_lx=new_lx,
        rect_ly=new_ly,
        rect_hx=new_hx,
        rect_hy=new_hy,
        rect_cnt=new_cnt,
    )


# ---------------------------------------------------------------------------
# add_rect — vendor rect.c:137-151
# ---------------------------------------------------------------------------


def add_rect(
    pool: RectPool,
    new_lx: jax.Array,
    new_ly: jax.Array,
    new_hx: jax.Array,
    new_hy: jax.Array,
) -> RectPool:
    """Append a new rect, rejecting fulls / inclusions (no-op).

    Vendor rect.c:137-151::

        if (rect_cnt >= MAXRECT) { ...wizard warn...; return; }
        if (get_rect(r))    return;     /* already contained */
        rect[rect_cnt] = *r;
        rect_cnt++;

    The wizard-mode pline is host-only and skipped here.
    """
    nlx = jnp.int16(new_lx)
    nly = jnp.int16(new_ly)
    nhx = jnp.int16(new_hx)
    nhy = jnp.int16(new_hy)

    full = pool.rect_cnt >= jnp.int32(MAXRECT)
    contained, _, _, _, _ = get_rect(pool, nlx, nly, nhx, nhy)
    can_add = (~full) & (~contained)

    write_idx = jnp.where(
        can_add, pool.rect_cnt, jnp.int32(_POOL_SLOTS - 1)
    ).astype(jnp.int32)

    new_rect_lx = pool.rect_lx.at[write_idx].set(
        jnp.where(can_add, nlx, pool.rect_lx[write_idx])
    )
    new_rect_ly = pool.rect_ly.at[write_idx].set(
        jnp.where(can_add, nly, pool.rect_ly[write_idx])
    )
    new_rect_hx = pool.rect_hx.at[write_idx].set(
        jnp.where(can_add, nhx, pool.rect_hx[write_idx])
    )
    new_rect_hy = pool.rect_hy.at[write_idx].set(
        jnp.where(can_add, nhy, pool.rect_hy[write_idx])
    )
    new_cnt = jnp.where(can_add, pool.rect_cnt + jnp.int32(1), pool.rect_cnt)
    return pool.replace(
        rect_lx=new_rect_lx,
        rect_ly=new_rect_ly,
        rect_hx=new_rect_hx,
        rect_hy=new_rect_hy,
        rect_cnt=new_cnt,
    )


# ---------------------------------------------------------------------------
# intersect — vendor rect.c:100-116 (static helper)
# ---------------------------------------------------------------------------


def _intersect(
    r1_lx, r1_ly, r1_hx, r1_hy, r2_lx, r2_ly, r2_hx, r2_hy
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute the rectangle intersection.

    Vendor rect.c:100-116.  Returns ``(exists, lx, ly, hx, hy)``.  Coords
    are zero when no intersection exists (caller masks via ``exists``).
    """
    disjoint = (
        (r2_lx > r1_hx)
        | (r2_ly > r1_hy)
        | (r2_hx < r1_lx)
        | (r2_hy < r1_ly)
    )
    out_lx = jnp.maximum(r1_lx, r2_lx).astype(jnp.int16)
    out_ly = jnp.maximum(r1_ly, r2_ly).astype(jnp.int16)
    out_hx = jnp.minimum(r1_hx, r2_hx).astype(jnp.int16)
    out_hy = jnp.minimum(r1_hy, r2_hy).astype(jnp.int16)
    degenerate = (out_lx > out_hx) | (out_ly > out_hy)
    exists = (~disjoint) & (~degenerate)
    return (
        exists,
        jnp.where(exists, out_lx, jnp.int16(0)),
        jnp.where(exists, out_ly, jnp.int16(0)),
        jnp.where(exists, out_hx, jnp.int16(0)),
        jnp.where(exists, out_hy, jnp.int16(0)),
    )


# ---------------------------------------------------------------------------
# split_rects — vendor rect.c:160-197 (iterative-stack JAX port)
#
# Vendor recursive structure::
#
#     void split_rects(r1, r2):
#         old_r = *r1
#         remove_rect(r1)
#         for i = rect_cnt - 1 downto 0:
#             if intersect(rect[i], r2, r):
#                 split_rects(rect[i], r)            # depth-first recursion
#         # then up to 4 leftover-strip add_rects (top/left/bottom/right
#         # of r2 within old_r):
#         if (r2.ly - old_r.ly - 1 > (...YLIM...)) add_rect(top-strip)
#         if (r2.lx - old_r.lx - 1 > (...XLIM...)) add_rect(left-strip)
#         if (old_r.hy - r2.hy - 1 > (...YLIM...)) add_rect(bottom-strip)
#         if (old_r.hx - r2.hx - 1 > (...XLIM...)) add_rect(right-strip)
#
# Iterative emulation:
#
#     Each stack frame represents one in-flight split_rects call and
#     carries (r1, r2, old_r, phase, i):
#       phase 0:  entry — call remove_rect(r1), advance to phase 1
#                 with i = rect_cnt - 1.
#       phase 1:  intersect-loop body.  If i < 0 → advance to phase 2.
#                 Else read rect[i], test intersect(rect[i], r2).  On
#                 hit, decrement i in *this* frame then push a new
#                 frame (rect[i], intersection, phase=0) onto the
#                 stack — when it pops, control returns here.  On
#                 miss, just decrement i.
#       phase 2:  emit the 4 leftover-strip add_rects (top, left,
#                 bottom, right — vendor's textual order).  Pop frame.
#
#     The vendor recursion order is thus preserved exactly: depth-
#     first descent into intersecting pool rects (high-index first),
#     each fully resolved before the next, then the parent's strips.
# ---------------------------------------------------------------------------


# Stack-frame field layout — kept as separate arrays for pytree clarity.
# Each frame stores 8 int16 coords (r1 + r2; we re-derive ``old_r`` from
# r1 since the entry-phase removal hasn't yet mutated r1), plus phase
# (int8) and i (int32).


def _split_rects_iterative(
    pool: RectPool,
    parent_lx: jax.Array,
    parent_ly: jax.Array,
    parent_hx: jax.Array,
    parent_hy: jax.Array,
    child_lx: jax.Array,
    child_ly: jax.Array,
    child_hx: jax.Array,
    child_hy: jax.Array,
) -> RectPool:
    """Iterative split_rects with a bounded explicit recursion stack."""

    D = _MAX_SPLIT_DEPTH

    # Initial stack: a single live frame in slot 0.
    stk_r1_lx = jnp.zeros((D,), dtype=jnp.int16).at[0].set(jnp.int16(parent_lx))
    stk_r1_ly = jnp.zeros((D,), dtype=jnp.int16).at[0].set(jnp.int16(parent_ly))
    stk_r1_hx = jnp.zeros((D,), dtype=jnp.int16).at[0].set(jnp.int16(parent_hx))
    stk_r1_hy = jnp.zeros((D,), dtype=jnp.int16).at[0].set(jnp.int16(parent_hy))
    stk_r2_lx = jnp.zeros((D,), dtype=jnp.int16).at[0].set(jnp.int16(child_lx))
    stk_r2_ly = jnp.zeros((D,), dtype=jnp.int16).at[0].set(jnp.int16(child_ly))
    stk_r2_hx = jnp.zeros((D,), dtype=jnp.int16).at[0].set(jnp.int16(child_hx))
    stk_r2_hy = jnp.zeros((D,), dtype=jnp.int16).at[0].set(jnp.int16(child_hy))
    stk_phase = jnp.zeros((D,), dtype=jnp.int8)  # all zero — entry phase
    stk_i = jnp.zeros((D,), dtype=jnp.int32)
    sp = jnp.int32(1)  # one live frame

    def cond(state):
        (_pool, _r1lx, _r1ly, _r1hx, _r1hy,
         _r2lx, _r2ly, _r2hx, _r2hy, _ph, _ii, sp_in) = state
        return sp_in > jnp.int32(0)

    def body(state):
        (pool_c, r1lx, r1ly, r1hx, r1hy,
         r2lx, r2ly, r2hx, r2hy, ph, ii, sp_in) = state

        top = sp_in - jnp.int32(1)

        # Read top frame.
        f_r1lx = r1lx[top]
        f_r1ly = r1ly[top]
        f_r1hx = r1hx[top]
        f_r1hy = r1hy[top]
        f_r2lx = r2lx[top]
        f_r2ly = r2ly[top]
        f_r2hx = r2hx[top]
        f_r2hy = r2hy[top]
        f_ph = ph[top]
        f_i = ii[top]

        # --- Phase 0: entry — remove r1, advance to phase 1. -------------
        def do_phase0(carry):
            (pool_p, r1lx_p, r1ly_p, r1hx_p, r1hy_p,
             r2lx_p, r2ly_p, r2hx_p, r2hy_p, ph_p, ii_p, sp_p) = carry
            pool_n = remove_rect(pool_p, f_r1lx, f_r1ly, f_r1hx, f_r1hy)
            ph_n = ph_p.at[top].set(jnp.int8(1))
            # Vendor: for (i = rect_cnt - 1; i >= 0; i--)
            ii_n = ii_p.at[top].set(pool_n.rect_cnt - jnp.int32(1))
            return (pool_n, r1lx_p, r1ly_p, r1hx_p, r1hy_p,
                    r2lx_p, r2ly_p, r2hx_p, r2hy_p, ph_n, ii_n, sp_p)

        # --- Phase 1: intersect-loop step. -------------------------------
        def do_phase1(carry):
            (pool_p, r1lx_p, r1ly_p, r1hx_p, r1hy_p,
             r2lx_p, r2ly_p, r2hx_p, r2hy_p, ph_p, ii_p, sp_p) = carry

            # If i < 0, advance to phase 2; otherwise inspect rect[i].
            done_loop = f_i < jnp.int32(0)

            # Safe index for read (clamp to valid pool slot when not done).
            safe_i = jnp.maximum(f_i, jnp.int32(0)).astype(jnp.int32)
            cand_lx = pool_p.rect_lx[safe_i]
            cand_ly = pool_p.rect_ly[safe_i]
            cand_hx = pool_p.rect_hx[safe_i]
            cand_hy = pool_p.rect_hy[safe_i]

            exists, ix_lx, ix_ly, ix_hx, ix_hy = _intersect(
                cand_lx, cand_ly, cand_hx, cand_hy,
                f_r2lx, f_r2ly, f_r2hx, f_r2hy,
            )

            # Always decrement i in the current frame.
            ii_dec = ii_p.at[top].set(f_i - jnp.int32(1))

            # If we hit a real intersection AND there's stack room left,
            # push a new frame at sp (which is top+1).  The vendor depth
            # is empirically <= 10; the cap=64 is purely defensive — if
            # we ever blow it we silently skip the recursion (matches
            # vendor's static-array storage philosophy: bounded resource,
            # no panic).
            can_push = exists & (~done_loop) & (sp_p < jnp.int32(D))
            push_slot = jnp.where(can_push, sp_p, jnp.int32(D - 1)).astype(jnp.int32)

            r1lx_n = r1lx_p.at[push_slot].set(
                jnp.where(can_push, cand_lx, r1lx_p[push_slot])
            )
            r1ly_n = r1ly_p.at[push_slot].set(
                jnp.where(can_push, cand_ly, r1ly_p[push_slot])
            )
            r1hx_n = r1hx_p.at[push_slot].set(
                jnp.where(can_push, cand_hx, r1hx_p[push_slot])
            )
            r1hy_n = r1hy_p.at[push_slot].set(
                jnp.where(can_push, cand_hy, r1hy_p[push_slot])
            )
            r2lx_n = r2lx_p.at[push_slot].set(
                jnp.where(can_push, ix_lx, r2lx_p[push_slot])
            )
            r2ly_n = r2ly_p.at[push_slot].set(
                jnp.where(can_push, ix_ly, r2ly_p[push_slot])
            )
            r2hx_n = r2hx_p.at[push_slot].set(
                jnp.where(can_push, ix_hx, r2hx_p[push_slot])
            )
            r2hy_n = r2hy_p.at[push_slot].set(
                jnp.where(can_push, ix_hy, r2hy_p[push_slot])
            )
            ph_push = ph_p.at[push_slot].set(
                jnp.where(can_push, jnp.int8(0), ph_p[push_slot])
            )
            ii_push = ii_dec.at[push_slot].set(
                jnp.where(can_push, jnp.int32(0), ii_dec[push_slot])
            )

            # Advance to phase 2 once the descending loop is done.
            ph_advance = ph_push.at[top].set(
                jnp.where(done_loop, jnp.int8(2), ph_push[top])
            )

            sp_n = jnp.where(can_push, sp_p + jnp.int32(1), sp_p)

            return (pool_p, r1lx_n, r1ly_n, r1hx_n, r1hy_n,
                    r2lx_n, r2ly_n, r2hx_n, r2hy_n, ph_advance, ii_push, sp_n)

        # --- Phase 2: emit the 4 leftover-strip add_rects, then pop. ----
        def do_phase2(carry):
            (pool_p, r1lx_p, r1ly_p, r1hx_p, r1hy_p,
             r2lx_p, r2ly_p, r2hx_p, r2hy_p, ph_p, ii_p, sp_p) = carry

            # ``old_r`` == frame's r1 (we never overwrote it; remove_rect
            # only consumed it from the pool, not from our stack copy).
            old_lx, old_ly, old_hx, old_hy = f_r1lx, f_r1ly, f_r1hx, f_r1hy
            cr_lx, cr_ly, cr_hx, cr_hy = f_r2lx, f_r2ly, f_r2hx, f_r2hy

            # Vendor rect.c:175-180 — TOP strip (above r2):
            #   if (r2.ly - old_r.ly - 1
            #       > (old_r.hy < ROWNO - 1 ? 2*YLIM : YLIM + 1) + 4)
            thr_top = jnp.where(
                old_hy < jnp.int16(_ROWNO - 1),
                jnp.int16(2 * _YLIM + 4),
                jnp.int16(_YLIM + 1 + 4),
            )
            top_ok = (cr_ly - old_ly - jnp.int16(1)) > thr_top
            # Brax-flatten: always invoke add_rect, jnp.where-select per field.
            pool_top = add_rect(pool_p, old_lx, old_ly, old_hx, cr_ly - jnp.int16(2))
            pool_p = pool_p.replace(
                rect_lx=jnp.where(top_ok, pool_top.rect_lx, pool_p.rect_lx),
                rect_ly=jnp.where(top_ok, pool_top.rect_ly, pool_p.rect_ly),
                rect_hx=jnp.where(top_ok, pool_top.rect_hx, pool_p.rect_hx),
                rect_hy=jnp.where(top_ok, pool_top.rect_hy, pool_p.rect_hy),
                rect_cnt=jnp.where(top_ok, pool_top.rect_cnt, pool_p.rect_cnt),
            )

            # Vendor rect.c:181-186 — LEFT strip (left of r2):
            #   if (r2.lx - old_r.lx - 1
            #       > (old_r.hx < COLNO - 1 ? 2*XLIM : XLIM + 1) + 4)
            thr_left = jnp.where(
                old_hx < jnp.int16(_COLNO - 1),
                jnp.int16(2 * _XLIM + 4),
                jnp.int16(_XLIM + 1 + 4),
            )
            left_ok = (cr_lx - old_lx - jnp.int16(1)) > thr_left
            # Brax-flatten: always invoke add_rect, jnp.where-select per field.
            pool_left = add_rect(pool_p, old_lx, old_ly, cr_lx - jnp.int16(2), old_hy)
            pool_p = pool_p.replace(
                rect_lx=jnp.where(left_ok, pool_left.rect_lx, pool_p.rect_lx),
                rect_ly=jnp.where(left_ok, pool_left.rect_ly, pool_p.rect_ly),
                rect_hx=jnp.where(left_ok, pool_left.rect_hx, pool_p.rect_hx),
                rect_hy=jnp.where(left_ok, pool_left.rect_hy, pool_p.rect_hy),
                rect_cnt=jnp.where(left_ok, pool_left.rect_cnt, pool_p.rect_cnt),
            )

            # Vendor rect.c:187-191 — BOTTOM strip (below r2):
            #   if (old_r.hy - r2.hy - 1
            #       > (old_r.ly > 0 ? 2*YLIM : YLIM + 1) + 4)
            thr_bot = jnp.where(
                old_ly > jnp.int16(0),
                jnp.int16(2 * _YLIM + 4),
                jnp.int16(_YLIM + 1 + 4),
            )
            bot_ok = (old_hy - cr_hy - jnp.int16(1)) > thr_bot
            # Brax-flatten: always invoke add_rect, jnp.where-select per field.
            pool_bot = add_rect(pool_p, old_lx, cr_hy + jnp.int16(2), old_hx, old_hy)
            pool_p = pool_p.replace(
                rect_lx=jnp.where(bot_ok, pool_bot.rect_lx, pool_p.rect_lx),
                rect_ly=jnp.where(bot_ok, pool_bot.rect_ly, pool_p.rect_ly),
                rect_hx=jnp.where(bot_ok, pool_bot.rect_hx, pool_p.rect_hx),
                rect_hy=jnp.where(bot_ok, pool_bot.rect_hy, pool_p.rect_hy),
                rect_cnt=jnp.where(bot_ok, pool_bot.rect_cnt, pool_p.rect_cnt),
            )

            # Vendor rect.c:192-196 — RIGHT strip (right of r2):
            #   if (old_r.hx - r2.hx - 1
            #       > (old_r.lx > 0 ? 2*XLIM : XLIM + 1) + 4)
            thr_right = jnp.where(
                old_lx > jnp.int16(0),
                jnp.int16(2 * _XLIM + 4),
                jnp.int16(_XLIM + 1 + 4),
            )
            right_ok = (old_hx - cr_hx - jnp.int16(1)) > thr_right
            # Brax-flatten: always invoke add_rect, jnp.where-select per field.
            pool_right = add_rect(pool_p, cr_hx + jnp.int16(2), old_ly, old_hx, old_hy)
            pool_p = pool_p.replace(
                rect_lx=jnp.where(right_ok, pool_right.rect_lx, pool_p.rect_lx),
                rect_ly=jnp.where(right_ok, pool_right.rect_ly, pool_p.rect_ly),
                rect_hx=jnp.where(right_ok, pool_right.rect_hx, pool_p.rect_hx),
                rect_hy=jnp.where(right_ok, pool_right.rect_hy, pool_p.rect_hy),
                rect_cnt=jnp.where(right_ok, pool_right.rect_cnt, pool_p.rect_cnt),
            )

            # Pop frame.
            sp_n = sp_p - jnp.int32(1)
            return (pool_p, r1lx_p, r1ly_p, r1hx_p, r1hy_p,
                    r2lx_p, r2ly_p, r2hx_p, r2hy_p, ph_p, ii_p, sp_n)

        # Brax-flatten: instead of lax.switch, eagerly compute all three
        # phase branches on the current carry and jnp.where-select per field
        # using the active phase.  Vendor's strict per-step ordering is
        # preserved because each invocation of ``body`` advances exactly one
        # frame by one phase — only the branch matching ``f_ph`` contributes
        # to the next state.
        carry = (pool_c, r1lx, r1ly, r1hx, r1hy,
                 r2lx, r2ly, r2hx, r2hy, ph, ii, sp_in)
        out0 = do_phase0(carry)
        out1 = do_phase1(carry)
        out2 = do_phase2(carry)

        phase_i = jnp.clip(f_ph.astype(jnp.int32), 0, 2)
        is0 = phase_i == jnp.int32(0)
        is1 = phase_i == jnp.int32(1)
        # is2 implied by ~is0 & ~is1

        def _sel(a0, a1, a2):
            return jnp.where(is0, a0, jnp.where(is1, a1, a2))

        pool0, r1lx0, r1ly0, r1hx0, r1hy0, r2lx0, r2ly0, r2hx0, r2hy0, ph0, ii0, sp0 = out0
        pool1, r1lx1, r1ly1, r1hx1, r1hy1, r2lx1, r2ly1, r2hx1, r2hy1, ph1, ii1, sp1 = out1
        pool2, r1lx2, r1ly2, r1hx2, r1hy2, r2lx2, r2ly2, r2hx2, r2hy2, ph2, ii2, sp2 = out2

        new_pool = pool_c.replace(
            rect_lx=_sel(pool0.rect_lx, pool1.rect_lx, pool2.rect_lx),
            rect_ly=_sel(pool0.rect_ly, pool1.rect_ly, pool2.rect_ly),
            rect_hx=_sel(pool0.rect_hx, pool1.rect_hx, pool2.rect_hx),
            rect_hy=_sel(pool0.rect_hy, pool1.rect_hy, pool2.rect_hy),
            rect_cnt=_sel(pool0.rect_cnt, pool1.rect_cnt, pool2.rect_cnt),
        )
        new_state = (
            new_pool,
            _sel(r1lx0, r1lx1, r1lx2),
            _sel(r1ly0, r1ly1, r1ly2),
            _sel(r1hx0, r1hx1, r1hx2),
            _sel(r1hy0, r1hy1, r1hy2),
            _sel(r2lx0, r2lx1, r2lx2),
            _sel(r2ly0, r2ly1, r2ly2),
            _sel(r2hx0, r2hx1, r2hx2),
            _sel(r2hy0, r2hy1, r2hy2),
            _sel(ph0, ph1, ph2),
            _sel(ii0, ii1, ii2),
            _sel(sp0, sp1, sp2),
        )
        return new_state

    init = (pool, stk_r1_lx, stk_r1_ly, stk_r1_hx, stk_r1_hy,
            stk_r2_lx, stk_r2_ly, stk_r2_hx, stk_r2_hy,
            stk_phase, stk_i, sp)
    final = lax.while_loop(cond, body, init)
    return final[0]


def split_rects(
    pool: RectPool,
    parent_lx: jax.Array,
    parent_ly: jax.Array,
    parent_hx: jax.Array,
    parent_hy: jax.Array,
    child_lx: jax.Array,
    child_ly: jax.Array,
    child_hx: jax.Array,
    child_hy: jax.Array,
) -> RectPool:
    """Carve a child rect out of a parent rect in the pool.

    Vendor signature: ``void split_rects(NhRect *r1, NhRect *r2)`` where
    ``r1`` is already in the pool and ``r2`` is included in ``r1``.

    This entry point unpacks the (r1, r2) coords and runs the iterative
    JAX recursion described above.  Returns the updated pool.

    See rect.c:160-197 for the vendor reference.
    """
    return _split_rects_iterative(
        pool,
        parent_lx, parent_ly, parent_hx, parent_hy,
        child_lx, child_ly, child_hx, child_hy,
    )


# ---------------------------------------------------------------------------
# Module smoke-test (vendor parity: initial rect = (0, 0, 79, 20))
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    from Nethax.nethax.vendor_rng import init as _rng_init

    p = init_rect()
    assert int(p.rect_cnt) == 1, f"initial rect_cnt expected 1, got {int(p.rect_cnt)}"
    rng0 = _rng_init(0)
    p2, lx, ly, hx, hy, rng1, found = rnd_rect(p, rng0)
    assert bool(found), "rnd_rect should find the initial rect"
    assert (int(lx), int(ly), int(hx), int(hy)) == (0, 0, 79, 20), (
        f"first rnd_rect coords expected (0,0,79,20), got "
        f"({int(lx)},{int(ly)},{int(hx)},{int(hy)})"
    )
    print("rect_pool smoke-test passed:", int(p.rect_cnt),
          (int(lx), int(ly), int(hx), int(hy)))
