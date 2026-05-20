"""Long-worm multi-tile body subsystem.

Vendor reference: vendor/nethack/src/worm.c (1001 lines).

In NetHack, "long worms" (PM_LONG_WORM) occupy multiple adjacent map tiles:
one head tile plus a chain of body segments that trail behind it.  Hits on
a non-head segment can sever the worm into two independent worms; hits on
any segment route HP damage to the worm's head (vendor: ``mtmp->mhp`` is a
single counter for the whole creature).

Vendor layout (worm.c:42-72):
    wtails[wormno]  → start of the segment chain (tail end)
    wheads[wormno]  → end of chain, co-located with the head monster
    wgrowtime[wormno] → next svm.moves at which to grow a new segment

JIT-pure JAX port (this module):
    WormState arrays are fixed-shape so they fit in a jitted EnvState.
    - ``MAX_NUM_WORMS``       maps worm slot ids (1..N-1; slot 0 unused, per
      vendor convention worm.c:90-94).
    - ``MAX_WSEGS_PER_WORM``  caps the per-worm segment chain length.

State arrays (all int16 / bool fixed-size):
    seg_pos      [W, S, 2]  (row, col) for each segment; (-1, -1) = empty
    seg_count    [W]        active segment count  (1..MAX_WSEGS_PER_WORM)
    wgrowtime    [W]        absolute move counter at which to grow next seg
    head_idx     [W]        owning monster slot id, or -1 if unused
    in_use       [W]        True if this worm slot is allocated

Functions:
    get_wormno(state)              → first free worm slot (0 if none)
    init_worm(state, slot, mon_id) → initialise a fresh worm with one seg
    worm_move(state, slot, hr, hc) → head moved to (hr, hc); add new seg
    grow_worm(state, slot, turn)   → extend worm by one segment when due
    cutworm(state, slot, x, y, ml) → split worm at segment (x, y); maybe new worm
    worm_at(state, r, c)           → (found, slot) for any seg at (r, c)
    tail_hit_to_head(state, slot, hp_lost) → route segment hit HP to head

JIT-pure: every operation uses ``jax.lax``/``jnp.where`` — no Python branching
on traced values, no Python lists.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


# ---------------------------------------------------------------------------
# Capacity constants
# ---------------------------------------------------------------------------
# Vendor MAX_NUM_WORMS (include/global.h) defaults to 32.  We mirror that.
MAX_NUM_WORMS: int = 32

# Vendor worms can have effectively unbounded segments (linked-list grows
# at most every few turns; see worm.c:218-238).  For a JIT-safe fixed
# array we cap at 16 segments which is enough for a long worm at m_lev≈10
# (vendor whplimit table tops out at ~33 segments — see worm.c:246-252).
MAX_WSEGS_PER_WORM: int = 16


# ---------------------------------------------------------------------------
# WormState
# ---------------------------------------------------------------------------

@struct.dataclass
class WormState:
    """Per-level long-worm bookkeeping.

    Cite: vendor/nethack/src/worm.c lines 27-79 (description of arrays).
    """

    # Position of each segment.  Index 0 is the tail end (oldest segment);
    # index ``seg_count[w]-1`` is the head dummy segment (vendor wheads).
    # Empty entries hold (-1, -1).
    seg_pos: jnp.ndarray       # [W, S, 2]  int16

    # Number of *valid* segments per worm slot.  An idle slot has 0.
    seg_count: jnp.ndarray     # [W]  int16

    # Vendor wgrowtime[wormno] — absolute svm.moves at which the worm should
    # add a new segment.  0 when the worm has not yet grown for the first time.
    # Cite: vendor/nethack/src/worm.c lines 218-238.
    wgrowtime: jnp.ndarray     # [W]  int32

    # Owning monster slot for each worm (or -1 when slot is unused).
    head_idx: jnp.ndarray      # [W]  int16

    # True iff this worm slot is allocated.
    in_use: jnp.ndarray        # [W]  bool


def make_worm_state() -> WormState:
    """Return a zero-initialised WormState (all slots empty)."""
    w = MAX_NUM_WORMS
    s = MAX_WSEGS_PER_WORM
    return WormState(
        seg_pos=jnp.full((w, s, 2), -1, dtype=jnp.int16),
        seg_count=jnp.zeros((w,), dtype=jnp.int16),
        wgrowtime=jnp.zeros((w,), dtype=jnp.int32),
        head_idx=jnp.full((w,), -1, dtype=jnp.int16),
        in_use=jnp.zeros((w,), dtype=jnp.bool_),
    )


# ---------------------------------------------------------------------------
# get_wormno — find a free slot
# ---------------------------------------------------------------------------

def get_wormno(state_or_worms) -> tuple:
    """Return (found, slot) — the first free worm slot ``1 <= slot < W``.

    Cite: vendor/nethack/src/worm.c::get_wormno lines 95-106.  Vendor
    convention reserves slot 0 ("0 = not a worm"); we mirror that by
    masking slot 0 below.
    """
    worms = _get_worms(state_or_worms)
    in_use = worms.in_use
    # Slot 0 is reserved (vendor: "the [0] positions of the arrays are never
    # used" — worm.c:90).  Mark slot 0 as 'in use' for the search so it is
    # never returned.
    mask = in_use.at[0].set(jnp.bool_(True))
    free = ~mask
    found = jnp.any(free)
    slot = jnp.argmax(free.astype(jnp.int32)).astype(jnp.int32)
    return found, slot


def _get_worms(state_or_worms) -> WormState:
    """Accept either an EnvState or WormState; return the WormState."""
    if isinstance(state_or_worms, WormState):
        return state_or_worms
    return state_or_worms.worm_state  # EnvState attribute (state.py)


def _replace_worms(state, new_worms: WormState):
    """Write ``new_worms`` back into ``state.worm_state`` if state is an EnvState."""
    if isinstance(state, WormState):
        return new_worms
    return state.replace(worm_state=new_worms)


# ---------------------------------------------------------------------------
# init_worm — set up a fresh worm at (r, c) owned by monster slot mon_id
# ---------------------------------------------------------------------------

def init_worm(state, slot: jnp.ndarray, mon_id: jnp.ndarray,
              r: jnp.ndarray, c: jnp.ndarray):
    """Initialise worm ``slot`` with a single head segment at (r, c).

    Cite: vendor/nethack/src/worm.c::initworm lines 119-137.  Vendor sets
    up the dummy head segment at (worm->mx, worm->my) and zeroes wgrowtime.
    """
    worms = _get_worms(state)
    s = slot.astype(jnp.int32)
    ri = r.astype(jnp.int16)
    ci = c.astype(jnp.int16)

    # Reset the segment chain to a single head entry at (r, c).
    new_seg_pos = worms.seg_pos.at[s, 0].set(jnp.stack([ri, ci]))
    # Zero the remaining segments (in case slot was previously used).
    zero_seg = jnp.full((MAX_WSEGS_PER_WORM - 1, 2), -1, dtype=jnp.int16)
    new_seg_pos = jax.lax.dynamic_update_slice(
        new_seg_pos,
        zero_seg[None, ...],
        (s, jnp.int32(1), jnp.int32(0)),
    )

    new_worms = worms.replace(
        seg_pos=new_seg_pos,
        seg_count=worms.seg_count.at[s].set(jnp.int16(1)),
        wgrowtime=worms.wgrowtime.at[s].set(jnp.int32(0)),
        head_idx=worms.head_idx.at[s].set(mon_id.astype(jnp.int16)),
        in_use=worms.in_use.at[s].set(jnp.bool_(True)),
    )
    return _replace_worms(state, new_worms)


# ---------------------------------------------------------------------------
# worm_move — head moves to (hr, hc); old head becomes a body segment
# ---------------------------------------------------------------------------

def worm_move(state, slot: jnp.ndarray, hr: jnp.ndarray, hc: jnp.ndarray,
              grow: jnp.ndarray):
    """Advance the worm one tile: enqueue (hr, hc) as new head segment.

    Vendor flow (worm.c::worm_move lines 196-278):
        seg = wheads[wnum];       /* old head, becomes a body seg */
        place_worm_seg(worm, seg->wx, seg->wy);
        new_seg = newseg();
        new_seg->wx = worm->mx; new_seg->wy = worm->my;
        seg->nseg = new_seg;
        wheads[wnum] = new_seg;
        if (wgrowtime[wnum] <= svm.moves) {
            /* keep extra seg, recalc growtime */
        } else {
            shrink_worm(wnum);    /* drop tail */
        }

    ``grow`` (bool) gates the shrink/keep choice.  When True, the worm's
    length increases by one; when False, the tail segment is dropped so
    the worm's body simply slides forward.
    """
    worms = _get_worms(state)
    s = slot.astype(jnp.int32)
    count = worms.seg_count[s].astype(jnp.int32)
    ri = hr.astype(jnp.int16)
    ci = hc.astype(jnp.int16)
    new_head = jnp.stack([ri, ci])

    cap = jnp.int32(MAX_WSEGS_PER_WORM)
    # When growing and we have room, simply append.  Otherwise, shift the
    # chain left (drop tail) and place new head at position ``count-1``.
    will_grow = grow & (count < cap)

    def _append(seg_pos):
        # Add at slot ``count``.
        idx = count
        return seg_pos.at[s, idx].set(new_head)

    def _shift(seg_pos):
        # Shift segments 1..count-1 to 0..count-2, write new head at count-1.
        # Use dynamic_update_slice rather than Python loops.
        old = seg_pos[s]                              # [S, 2]
        shifted = jnp.concatenate(
            [old[1:], jnp.full((1, 2), -1, dtype=jnp.int16)],
            axis=0,
        )                                              # [S, 2]
        # The new head goes at index (count-1) within the shifted array.
        head_idx_local = jnp.maximum(count - jnp.int32(1), jnp.int32(0))
        shifted = shifted.at[head_idx_local].set(new_head)
        return jax.lax.dynamic_update_slice(
            seg_pos,
            shifted[None, ...],
            (s, jnp.int32(0), jnp.int32(0)),
        )

    new_seg_pos = jax.lax.cond(will_grow, _append, _shift, worms.seg_pos)

    new_count = jnp.where(
        will_grow,
        (count + jnp.int32(1)).astype(jnp.int16),
        count.astype(jnp.int16),
    )

    new_worms = worms.replace(
        seg_pos=new_seg_pos,
        seg_count=worms.seg_count.at[s].set(new_count),
    )
    return _replace_worms(state, new_worms)


# ---------------------------------------------------------------------------
# grow_worm — extend by one segment when wgrowtime reached
# ---------------------------------------------------------------------------

def grow_worm(state, slot: jnp.ndarray, current_turn: jnp.ndarray,
              rng: jax.Array):
    """Optionally extend worm ``slot`` by one tail segment when its
    ``wgrowtime`` has been reached.

    Vendor logic (worm.c::worm_move lines 218-237 — the grow branch):

        if (wgrowtime[wnum] <= svm.moves) {
            if (!wgrowtime[wnum]) {
                wgrowtime[wnum] = svm.moves + rnd(5);
            } else {
                int incr = rn1(10, 2);   /* 2..12  */
                incr = (incr * NORMAL_SPEED) / max(mmove, 1);
                wgrowtime[wnum] = svm.moves + incr;
            }
            /* ... heal HP based on segment count ... */
        }

    We schedule the next growth turn (advance ``wgrowtime``) and grow when
    appropriate.  HP healing is left to the monster_ai layer (vendor reads
    ``worm->mhp`` directly from the head monst struct).
    """
    worms = _get_worms(state)
    s = slot.astype(jnp.int32)
    turn = current_turn.astype(jnp.int32)
    grow_due = worms.wgrowtime[s].astype(jnp.int32) <= turn
    in_use = worms.in_use[s]
    should_grow = grow_due & in_use

    # Schedule next growth.  Vendor uses rnd(5) on first growth and rn1(10,2)
    # (= randint [2..12]) thereafter; the JIT-safe path uses uniform [2..12].
    rng_first, rng_incr = jax.random.split(rng)
    first_inc = jax.random.randint(rng_first, (), 1, 6, dtype=jnp.int32)   # 1..5
    later_inc = jax.random.randint(rng_incr,  (), 2, 13, dtype=jnp.int32)  # 2..12
    is_first = worms.wgrowtime[s].astype(jnp.int32) == jnp.int32(0)
    inc = jnp.where(is_first, first_inc, later_inc)
    new_grow = jnp.where(should_grow, turn + inc, worms.wgrowtime[s])

    new_worms = worms.replace(
        wgrowtime=worms.wgrowtime.at[s].set(new_grow.astype(jnp.int32)),
    )
    return _replace_worms(state, new_worms), should_grow


# ---------------------------------------------------------------------------
# cutworm — segment hit may split the worm in two
# ---------------------------------------------------------------------------

def cutworm(state, slot: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray,
            cuttier: jnp.ndarray, m_lev: jnp.ndarray, rng: jax.Array):
    """Maybe split worm ``slot`` at segment (x, y); spawn a new worm.

    Cite: vendor/nethack/src/worm.c::cutworm lines 364-477.

    Vendor flow (paraphrased):
        if (x == worm->mx && y == worm->my) return;  /* hit on head */
        cut_chance = rnd(20);
        if (cuttier) cut_chance += 10;
        if (cut_chance < 17) return;                   /* not severe */
        /* find the segment */
        if (curr == wtails[wnum]) { shrink_worm(wnum); return; }
        /* split: old worm keeps the head end, new worm gets tail end */
        new_wnum = (m_lev >= 3 && !rn2(3)) ? get_wormno() : 0;
        if (new_wnum) { ... clone head, transfer tail segs ... }

    Our port matches the vendor RNG calls 1:1.

    Returns: (new_state, did_split, new_slot)
        new_slot is -1 when no split occurred (or when no free worm slot
        existed, mirroring vendor's "new_worm = clone_mon..." failure
        path on worm.c:432-449).
    """
    worms = _get_worms(state)
    s = slot.astype(jnp.int32)

    # Vendor: cut_chance = rnd(20) [+10 if cuttier].
    rng_chance, rng_new = jax.random.split(rng)
    cut_chance = jax.random.randint(rng_chance, (), 1, 21, dtype=jnp.int32)
    cut_chance = jnp.where(cuttier, cut_chance + jnp.int32(10), cut_chance)
    cuts = cut_chance >= jnp.int32(17)

    # Find the segment matching (x, y).
    target = jnp.stack([x.astype(jnp.int16), y.astype(jnp.int16)])
    pos = worms.seg_pos[s]                 # [S, 2]
    count = worms.seg_count[s].astype(jnp.int32)
    valid = jnp.arange(MAX_WSEGS_PER_WORM, dtype=jnp.int32) < count
    matches = jnp.all(pos == target[None, :], axis=1) & valid
    has_match = jnp.any(matches)
    seg_idx = jnp.argmax(matches.astype(jnp.int32)).astype(jnp.int32)

    # Vendor: head segment is at index seg_count-1.  Cut on head is no-op
    # (line 384: `if (x == worm->mx && y == worm->my) return;`).
    head_idx_local = jnp.maximum(count - jnp.int32(1), jnp.int32(0))
    is_head_hit = seg_idx == head_idx_local

    # Vendor: hit on tail-most segment just shrinks the worm (worm.c:407).
    is_tail_hit = seg_idx == jnp.int32(0)

    # Vendor: split requires m_lev >= 3 && !rn2(3) (worm.c:427).
    can_split_lvl = m_lev.astype(jnp.int32) >= jnp.int32(3)
    rn2_3 = jax.random.randint(rng_new, (), 0, 3, dtype=jnp.int32) == jnp.int32(0)
    spawn_new = cuts & has_match & ~is_head_hit & ~is_tail_hit & can_split_lvl & rn2_3

    # Try to allocate a new worm slot.
    found_new, new_slot = get_wormno(worms)
    can_alloc = spawn_new & found_new

    # Old worm keeps segments [seg_idx .. count-1]; new worm gets [0 .. seg_idx-1].
    # JIT-safe: build masks and gather.
    idx_arr = jnp.arange(MAX_WSEGS_PER_WORM, dtype=jnp.int32)
    keep_old = (idx_arr >= seg_idx) & valid    # mask of old-worm survivors
    keep_new = (idx_arr <  seg_idx) & valid    # mask of new-worm survivors

    # Compact arrays: shift kept segments down to indices 0..k-1.
    def _compact(pos_arr, keep_mask):
        # cumulative count of True up to (and including) each index.
        # New index of kept seg i = sum(keep[0..i-1]).  We use lax.scan.
        def body(carry, args):
            kp, p = args
            out_idx, out_arr = carry
            new_arr = jax.lax.cond(
                kp,
                lambda a: a.at[out_idx].set(p),
                lambda a: a,
                out_arr,
            )
            new_idx = jnp.where(kp, out_idx + 1, out_idx)
            return (new_idx, new_arr), None

        empty = jnp.full((MAX_WSEGS_PER_WORM, 2), -1, dtype=jnp.int16)
        (n_kept, packed), _ = jax.lax.scan(
            body,
            (jnp.int32(0), empty),
            (keep_mask, pos_arr),
        )
        return packed, n_kept

    old_packed, old_n = _compact(pos, keep_old)
    new_packed, new_n = _compact(pos, keep_new)

    # When splitting, write back both worms.  Otherwise just handle the
    # tail-shrink case (drop one tail segment).
    do_split = can_alloc

    def _do_split(w):
        seg_pos_after_old = jax.lax.dynamic_update_slice(
            w.seg_pos, old_packed[None, ...], (s, jnp.int32(0), jnp.int32(0)),
        )
        seg_pos_after = jax.lax.dynamic_update_slice(
            seg_pos_after_old, new_packed[None, ...],
            (new_slot, jnp.int32(0), jnp.int32(0)),
        )
        return w.replace(
            seg_pos=seg_pos_after,
            seg_count=w.seg_count.at[s].set(old_n.astype(jnp.int16))
                                  .at[new_slot].set(new_n.astype(jnp.int16)),
            in_use=w.in_use.at[new_slot].set(jnp.bool_(True)),
            head_idx=w.head_idx.at[new_slot].set(jnp.int16(-1)),
            wgrowtime=w.wgrowtime.at[new_slot].set(jnp.int32(0)),
        )

    def _do_shrink(w):
        # Shrink: drop the tail-most segment if a cut hit was on the tail.
        # Vendor: worm.c:174-186 shrink_worm.
        should_shrink = cuts & has_match & is_tail_hit
        shifted = jnp.concatenate(
            [pos[1:], jnp.full((1, 2), -1, dtype=jnp.int16)], axis=0,
        )
        new_pos = jnp.where(should_shrink, shifted, pos)
        new_count_val = jnp.where(
            should_shrink, jnp.maximum(count - 1, 0), count,
        ).astype(jnp.int16)
        return w.replace(
            seg_pos=jax.lax.dynamic_update_slice(
                w.seg_pos, new_pos[None, ...], (s, jnp.int32(0), jnp.int32(0)),
            ),
            seg_count=w.seg_count.at[s].set(new_count_val),
        )

    new_worms = jax.lax.cond(do_split, _do_split, _do_shrink, worms)
    final_slot = jnp.where(do_split, new_slot, jnp.int32(-1))
    return _replace_worms(state, new_worms), do_split, final_slot


# ---------------------------------------------------------------------------
# worm_at — return the worm slot occupying (r, c), if any
# ---------------------------------------------------------------------------

def worm_at(state, r: jnp.ndarray, c: jnp.ndarray):
    """Return (found, slot) — the first worm whose body covers tile (r, c).

    Cite: vendor/nethack/src/worm.c::wseg_at lines 945-965; this exposes the
    worm-id rather than the segment index because callers (combat, vision)
    only need to know which long-worm owns the tile.
    """
    worms = _get_worms(state)
    target = jnp.stack([r.astype(jnp.int16), c.astype(jnp.int16)])
    pos = worms.seg_pos                                       # [W, S, 2]
    matches = jnp.all(pos == target[None, None, :], axis=2)    # [W, S]
    # Mask out unused slots / segments outside seg_count.
    idx_arr = jnp.arange(MAX_WSEGS_PER_WORM, dtype=jnp.int32)
    seg_valid = idx_arr[None, :] < worms.seg_count[:, None].astype(jnp.int32)
    in_use = worms.in_use[:, None]
    real = matches & seg_valid & in_use
    any_per_worm = jnp.any(real, axis=1)                       # [W]
    found = jnp.any(any_per_worm)
    slot = jnp.argmax(any_per_worm.astype(jnp.int32)).astype(jnp.int32)
    return found, slot


# ---------------------------------------------------------------------------
# tail_hit_to_head — segment-hit damage routes to the head monster slot
# ---------------------------------------------------------------------------

def tail_hit_to_head(state, slot: jnp.ndarray) -> jnp.ndarray:
    """Return the monster slot id whose HP should absorb a hit on any
    segment of worm ``slot``.

    Cite: vendor/nethack/src/worm.c lines 37-41 — "hit point bookkeeping
    much easier" because all damage is applied to ``worm->mhp`` (the head
    monst struct).  We model that by returning the ``head_idx`` for the
    given worm; the combat layer then applies the HP delta to that slot.
    """
    worms = _get_worms(state)
    return worms.head_idx[slot.astype(jnp.int32)].astype(jnp.int32)


# ---------------------------------------------------------------------------
# wormgone — clear a worm slot (called when the head dies)
# ---------------------------------------------------------------------------

def wormgone(state, slot: jnp.ndarray):
    """Free worm ``slot`` — clears all segments and marks the slot unused.

    Cite: vendor/nethack/src/worm.c::wormgone lines 308-332.
    """
    worms = _get_worms(state)
    s = slot.astype(jnp.int32)
    empty_pos = jnp.full((MAX_WSEGS_PER_WORM, 2), -1, dtype=jnp.int16)
    new_seg_pos = jax.lax.dynamic_update_slice(
        worms.seg_pos, empty_pos[None, ...],
        (s, jnp.int32(0), jnp.int32(0)),
    )
    new_worms = worms.replace(
        seg_pos=new_seg_pos,
        seg_count=worms.seg_count.at[s].set(jnp.int16(0)),
        wgrowtime=worms.wgrowtime.at[s].set(jnp.int32(0)),
        head_idx=worms.head_idx.at[s].set(jnp.int16(-1)),
        in_use=worms.in_use.at[s].set(jnp.bool_(False)),
    )
    return _replace_worms(state, new_worms)


# ---------------------------------------------------------------------------
# count_wsegs — number of body segments (excludes the head dummy seg)
# ---------------------------------------------------------------------------

def count_wsegs(state, slot: jnp.ndarray) -> jnp.ndarray:
    """Return the number of *body* segments (head dummy excluded).

    Cite: vendor/nethack/src/worm.c::count_wsegs lines 835-846.
    """
    worms = _get_worms(state)
    s = slot.astype(jnp.int32)
    count = worms.seg_count[s].astype(jnp.int32)
    # Vendor excludes the dummy head segment, so subtract one.
    return jnp.maximum(count - jnp.int32(1), jnp.int32(0))


# ---------------------------------------------------------------------------
# worm_cross — diagonal pass-through-worm-body block
# ---------------------------------------------------------------------------

def worm_cross(state, x1: jnp.ndarray, y1: jnp.ndarray,
               x2: jnp.ndarray, y2: jnp.ndarray) -> jnp.ndarray:
    """Return True if a diagonal move from (x1,y1) to (x2,y2) would pass
    between two *consecutive* segments of the same long worm.

    Cite: vendor/nethack/src/worm.c::worm_cross lines 895-942.

    Vendor flow (paraphrased):
        if (distmin(x1,y1,x2,y2) != 1) return FALSE;     /* non-adjacent */
        if (x1 == x2 || y1 == y2)      return FALSE;     /* not diagonal */
        worm = m_at(x1, y2);
        if (!worm || m_at(x2, y1) != worm) return FALSE; /* not same worm */
        /* walk worm seg chain; if two consecutive segs occupy
           (x1,y2) and (x2,y1) in either order → return TRUE  */

    In the JAX port we don't have an m_at index, so we scan all worm slots
    in-graph: for each worm, check whether *consecutive* segments occupy the
    diagonal-cardinal pair (x1, y2) and (x2, y1).  If any worm has such a
    pair, the diagonal move is blocked.
    """
    worms = _get_worms(state)
    # Vendor only blocks for true diagonals.  (worm.c:917-919)
    is_diag = (x1.astype(jnp.int32) != x2.astype(jnp.int32)) & (
        y1.astype(jnp.int32) != y2.astype(jnp.int32)
    )

    a = jnp.stack([x1.astype(jnp.int16), y2.astype(jnp.int16)])  # (x1, y2)
    b = jnp.stack([x2.astype(jnp.int16), y1.astype(jnp.int16)])  # (x2, y1)

    pos = worms.seg_pos                                # [W, S, 2]
    # Pair-wise consecutive segments along axis=1.
    cur = pos[:, :-1, :]                               # [W, S-1, 2]
    nxt = pos[:,  1:, :]                               # [W, S-1, 2]

    idx_arr = jnp.arange(MAX_WSEGS_PER_WORM - 1, dtype=jnp.int32)
    seg_valid = (
        idx_arr[None, :] < (worms.seg_count[:, None].astype(jnp.int32) - jnp.int32(1))
    )                                                  # [W, S-1]
    in_use = worms.in_use[:, None]                     # [W, 1]

    cur_is_a = jnp.all(cur == a[None, None, :], axis=2)
    nxt_is_b = jnp.all(nxt == b[None, None, :], axis=2)
    cur_is_b = jnp.all(cur == b[None, None, :], axis=2)
    nxt_is_a = jnp.all(nxt == a[None, None, :], axis=2)

    consecutive = ((cur_is_a & nxt_is_b) | (cur_is_b & nxt_is_a)) & seg_valid & in_use
    return is_diag & jnp.any(consecutive)


# ---------------------------------------------------------------------------
# place_worm_tail_randomly — initial-placement BFS-style segment scatter
# ---------------------------------------------------------------------------

def place_worm_tail_randomly(state, slot: jnp.ndarray,
                             x: jnp.ndarray, y: jnp.ndarray,
                             rng: jax.Array):
    """Place worm ``slot``'s tail segments randomly around (x, y).

    Cite: vendor/nethack/src/worm.c::place_worm_tail_randomly lines 728-792.

    Vendor walks the segment chain and for each tail seg picks a random
    walkable adjacent tile via ``rnd_nextto_goodpos``; if no neighbour is
    available the chain is truncated (``toss_wsegs``).

    The JAX port is JIT-pure: for each segment slot 0..count-2 (tail end up
    to but excluding the head) it picks a random direction from the 8
    neighbours of the previous segment and writes it if walkable; otherwise
    the segment is truncated (set to (-1, -1)) and a running ``truncated``
    flag drops all later segs.
    """
    worms = _get_worms(state)
    s = slot.astype(jnp.int32)

    # Walkable mask = tiles that aren't solid.  We test against the *current*
    # level's terrain (the worm always lives on one level).
    terrain = state.terrain[
        state.dungeon.current_branch,
        state.dungeon.current_level - 1,
    ]                                                # [H, W]
    map_h, map_w = terrain.shape

    from Nethax.nethax.constants.tiles import TileType as _TT
    walkable = (
        (terrain == jnp.int8(_TT.FLOOR))
        | (terrain == jnp.int8(_TT.CORRIDOR))
        | (terrain == jnp.int8(_TT.OPEN_DOOR))
    )

    # 8 neighbour offsets (vendor rnd_nextto_goodpos uses any of 8 dirs).
    dirs = jnp.array(
        [(-1, -1), (-1, 0), (-1, 1),
         ( 0, -1),          ( 0, 1),
         ( 1, -1), ( 1, 0), ( 1, 1)],
        dtype=jnp.int32,
    )                                                # [8, 2]

    seg_pos = worms.seg_pos[s]                       # [S, 2]
    count = worms.seg_count[s].astype(jnp.int32)
    # Head segment is at index count-1; we keep it where it is and walk back
    # toward index 0 (the tail end), choosing a random adjacent tile each
    # step.  Vendor builds the chain head→tail; we mirror that.
    head_pos = jnp.stack([x.astype(jnp.int16), y.astype(jnp.int16)])

    def body(carry, key):
        i, prev_pos, pos_arr, truncated = carry
        # Pick a random direction.
        d_idx = jax.random.randint(key, (), 0, 8, dtype=jnp.int32)
        d = dirs[d_idx]
        candidate_r = prev_pos[0].astype(jnp.int32) + d[0]
        candidate_c = prev_pos[1].astype(jnp.int32) + d[1]
        in_bounds = (
            (candidate_r >= 0) & (candidate_r < map_h)
            & (candidate_c >= 0) & (candidate_c < map_w)
        )
        is_walk = walkable[
            jnp.clip(candidate_r, 0, map_h - 1),
            jnp.clip(candidate_c, 0, map_w - 1),
        ]
        is_active = (i < count - jnp.int32(1)) & ~truncated
        ok = is_active & in_bounds & is_walk
        new_seg = jnp.where(
            ok,
            jnp.stack([candidate_r.astype(jnp.int16),
                       candidate_c.astype(jnp.int16)]),
            jnp.array([-1, -1], dtype=jnp.int16),
        )
        # Compute target index: tail end is 0, walking up to count-2.
        # Vendor builds head→tail by walking back; we mirror by writing at
        # (count - 2 - i).
        tgt_idx = jnp.maximum(count - jnp.int32(2) - i, jnp.int32(0))
        write = is_active
        new_pos_arr = jax.lax.cond(
            write,
            lambda a: a.at[tgt_idx].set(new_seg),
            lambda a: a,
            pos_arr,
        )
        new_prev = jnp.where(ok, new_seg, prev_pos)
        new_truncated = truncated | (is_active & ~ok)
        return (i + jnp.int32(1), new_prev, new_pos_arr, new_truncated), None

    keys = jax.random.split(rng, MAX_WSEGS_PER_WORM)
    (_, _, new_seg_pos_slot, _), _ = jax.lax.scan(
        body,
        (jnp.int32(0), head_pos, seg_pos, jnp.bool_(False)),
        keys,
    )
    # Re-write head position (slot count-1) to (x, y) — vendor leaves it
    # there explicitly (worm.c:771-772).
    head_idx_local = jnp.maximum(count - jnp.int32(1), jnp.int32(0))
    new_seg_pos_slot = new_seg_pos_slot.at[head_idx_local].set(head_pos)

    new_seg_pos = jax.lax.dynamic_update_slice(
        worms.seg_pos,
        new_seg_pos_slot[None, ...],
        (s, jnp.int32(0), jnp.int32(0)),
    )
    new_worms = worms.replace(seg_pos=new_seg_pos)
    return _replace_worms(state, new_worms)


# ---------------------------------------------------------------------------
# wormhitu — each adjacent worm segment hits the player
# ---------------------------------------------------------------------------

def wormhitu(state, slot: jnp.ndarray):
    """Each non-head segment of worm ``slot`` that is adjacent to the player
    attempts to hit; HP delta is routed via tail_hit_to_head to the head
    monster.

    Cite: vendor/nethack/src/worm.c::wormhitu lines 334-362.

    Vendor flow:
        for (seg = wtails[wnum]; seg != wheads[wnum]; seg = seg->nseg)
            if (distu(seg->wx, seg->wy) < 3)         /* Chebyshev<=1 in tiles */
                if (mattacku(worm)) return 1;

    The JAX port counts adjacent body segments (excluding the head dummy at
    index count-1) and returns ``(state, n_hits, head_mon_slot)``.  The
    caller — combat layer — applies one mattacku-equivalent damage roll per
    hit to the head monster slot.

    distu(x, y) is sq-Euclidean distance in NetHack (vendor mondata.c).
    "distu(seg) < 3" means within one tile in either axis (Chebyshev <= 1).
    """
    worms = _get_worms(state)
    s = slot.astype(jnp.int32)

    pos = worms.seg_pos[s]                              # [S, 2]
    count = worms.seg_count[s].astype(jnp.int32)
    head_idx_local = jnp.maximum(count - jnp.int32(1), jnp.int32(0))
    idx_arr = jnp.arange(MAX_WSEGS_PER_WORM, dtype=jnp.int32)
    seg_valid = (idx_arr < count) & (idx_arr != head_idx_local)

    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    sr = pos[:, 0].astype(jnp.int32)
    sc = pos[:, 1].astype(jnp.int32)
    cheb = jnp.maximum(jnp.abs(sr - pr), jnp.abs(sc - pc))
    in_range = cheb <= jnp.int32(1)

    hits = jnp.sum((seg_valid & in_range).astype(jnp.int32))
    head_mon = worms.head_idx[s].astype(jnp.int32)
    return state, hits, head_mon
