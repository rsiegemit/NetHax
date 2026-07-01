"""Sparse (dense<->sparse) representation for ``ground_items``.

PHASE 1 of the sparse ground_items migration.  This module is the *foundation*:
a fully JAX-traceable dense<->sparse primitive plus a sparse-obs helper.  The
dense ``Item`` array remains the source of truth this phase — nothing here
shrinks state or changes any write site.  The point is to prove the traceable
primitive is byte-exact so a later phase can flip the representation.

Dense layout (see ``inventory._empty_ground_items_array``):
    each ``Item`` field is ``[B=n_branches, L=max_levels, H, W, S=MAX_GROUND_STACK]``
    empty slot  <=>  ``category == 0``.

Sparse layout (this module):
    ``SparseGroundItems.items`` — an ``Item`` whose every field is ``[B, L, K]``,
    holding the first ``K`` occupied (row, col, slot) cells per level in
    *row-major* order (flat index ``row*W*S + col*S + slot`` ascending).
    ``SparseGroundItems.pos`` — ``int16[B, L, K, 3]`` == (row, col, slot).
    Pad entries (fewer than K occupied) carry ``category == 0`` and canonical
    empty-fill values.  On overflow (more than K occupied) the LOWEST flat
    indices survive (row-major-first == earliest tiles, lowest slot first);
    later cells are dropped — see the fidelity note on ``dense_to_sparse``.

All three public functions are ``jax.jit``-traceable (no ``np.nonzero`` /
data-dependent shapes): selection uses fixed-shape ``jax.lax.top_k`` and
reconstruction uses fixed-shape scatter.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import struct

from Nethax.nethax.subsystems.inventory import Item, _empty_ground_items_array

# ---------------------------------------------------------------------------
# Field metadata harvested from a real empty Item (no hardcoded field list).
# FILLS[f] = the canonical empty-cell value for field f; DTYPES[f] its dtype.
# ---------------------------------------------------------------------------
_FIELDS = tuple(f.name for f in dataclasses.fields(Item))
_EMPTY_CELL = _empty_ground_items_array(1, 1, 1, 1)  # shape [1,1,1,1,S] per field
_FILLS = {
    f: jnp.asarray(getattr(_EMPTY_CELL, f)).reshape(-1)[0] for f in _FIELDS
}
_DTYPES = {f: jnp.asarray(getattr(_EMPTY_CELL, f)).dtype for f in _FIELDS}


@struct.dataclass
class SparseGroundItems:
    """Sparse dual of the dense ground_items array.

    ``items`` reuses the ``Item`` struct as a plain field container: every
    field is ``[B, L, K]`` (not ``[B, L, H, W, S]``).  ``pos`` records the
    dense (row, col, slot) each captured entry came from.  ``H/W/S/K`` are
    static (compile-time) so scatter shapes are constant.
    """

    items: Item
    pos: jnp.ndarray  # int16[B, L, K, 3] == (row, col, slot)
    H: int = struct.field(pytree_node=False)
    W: int = struct.field(pytree_node=False)
    S: int = struct.field(pytree_node=False)
    K: int = struct.field(pytree_node=False)


def dense_to_sparse(ground_items: Item, K: int) -> SparseGroundItems:
    """Dense ``Item[B,L,H,W,S]`` -> ``SparseGroundItems`` (JITTABLE).

    Selects, per level, the first ``K`` occupied cells (``category != 0``) in
    row-major order (flat index ``row*W*S + col*S + slot`` ascending) via
    ``jax.lax.top_k`` on the negated flat index.  Flat indices are unique per
    cell so there are no ties among occupied cells; ``top_k`` returns them
    strictly ascending.  Occupancy < K pads with canonical empty-fill entries
    (``category == 0``).  Occupancy > K keeps the LOWEST flat indices (earliest
    tiles / lowest slots) and drops the rest — the documented overflow edge.
    """
    cat = ground_items.category
    B, L, H, W, S = cat.shape
    HWS = H * W * S

    cat_flat = cat.reshape(B, L, HWS)
    occupied = cat_flat != 0                                   # [B,L,HWS]
    flat_pos = jnp.arange(HWS, dtype=jnp.int32)                # [HWS]
    # key = flat index if occupied else HWS (a sentinel strictly above every
    # real flat index in [0, HWS)).  top_k(-key) => K smallest keys => K
    # smallest occupied flat indices, ascending; pads fall to the sentinel.
    key = jnp.where(occupied, flat_pos[None, None, :], jnp.int32(HWS))
    _, sel_flat = jax.lax.top_k(-key, K)                       # [B,L,K] indices==flat pos

    sel_cat = jnp.take_along_axis(cat_flat, sel_flat, axis=2)  # [B,L,K]
    valid = sel_cat != 0                                       # [B,L,K]

    def _gather(name):
        vals = getattr(ground_items, name).reshape(B, L, HWS)
        g = jnp.take_along_axis(vals, sel_flat, axis=2)        # [B,L,K]
        # Force pad entries to canonical empty-fill (keeps sparse rep clean).
        return jnp.where(valid, g, _FILLS[name].astype(_DTYPES[name]))

    items = Item(**{f: _gather(f) for f in _FIELDS})

    row = (sel_flat // (W * S)).astype(jnp.int16)
    col = ((sel_flat // S) % W).astype(jnp.int16)
    slot = (sel_flat % S).astype(jnp.int16)
    pos = jnp.stack([row, col, slot], axis=-1)                # [B,L,K,3]
    pos = jnp.where(valid[..., None], pos, jnp.int16(0))       # pads -> (0,0,0)

    return SparseGroundItems(items=items, pos=pos, H=H, W=W, S=S, K=K)


def sparse_to_dense(sparse: SparseGroundItems) -> Item:
    """``SparseGroundItems`` -> dense ``Item[B,L,H,W,S]`` (JITTABLE inverse).

    Scatters each captured entry back to its flat cell; pad/invalid entries
    (``category == 0``) scatter to a trash cell (index ``HWS``) that is dropped.
    Round-trips ``sparse_to_dense(dense_to_sparse(gi, K)) == gi`` field-for-field
    whenever per-level occupancy <= K and empty dense cells are canonical fill.
    """
    items = sparse.items
    pos = sparse.pos
    H, W, S, K = sparse.H, sparse.W, sparse.S, sparse.K
    B, L = items.category.shape[0], items.category.shape[1]
    HWS = H * W * S

    valid = items.category != 0                               # [B,L,K]
    row = pos[..., 0].astype(jnp.int32)
    col = pos[..., 1].astype(jnp.int32)
    slot = pos[..., 2].astype(jnp.int32)
    flat = row * (W * S) + col * S + slot
    flat = jnp.where(valid, flat, jnp.int32(HWS))             # invalid -> trash

    bb = jnp.broadcast_to(jnp.arange(B)[:, None, None], (B, L, K))
    ll = jnp.broadcast_to(jnp.arange(L)[None, :, None], (B, L, K))

    out = {}
    for f in _FIELDS:
        fill = _FILLS[f].astype(_DTYPES[f])
        base = jnp.full((B, L, HWS + 1), fill, dtype=_DTYPES[f])
        base = base.at[bb, ll, flat].set(getattr(items, f))
        out[f] = base[:, :, :HWS].reshape(B, L, H, W, S)
    return Item(**out)


def sparse_slot0_maps(sparse: SparseGroundItems, b, lv):
    """Top-of-stack (slot 0) category + type_id maps for one level (JITTABLE).

    Returns ``(cat_map, typ_map)`` each ``[H, W]``; only ``slot == 0`` occupied
    entries are written (all other cells 0).  ``b`` / ``lv`` may be traced.
    """
    H, W = sparse.H, sparse.W
    cat = sparse.items.category[b, lv]                        # [K]
    typ = sparse.items.type_id[b, lv]                         # [K]
    pos = sparse.pos[b, lv]                                   # [K,3]
    is_slot0 = (cat != 0) & (pos[:, 2] == jnp.int16(0))
    row = pos[:, 0].astype(jnp.int32)
    col = pos[:, 1].astype(jnp.int32)
    flat = jnp.where(is_slot0, row * W + col, jnp.int32(H * W))
    cat_map = jnp.zeros((H * W + 1,), _DTYPES["category"]).at[flat].set(cat)
    typ_map = jnp.zeros((H * W + 1,), _DTYPES["type_id"]).at[flat].set(typ)
    return cat_map[:H * W].reshape(H, W), typ_map[:H * W].reshape(H, W)


# ===========================================================================
# PHASE 2 — JITTABLE, vmap-safe sparse MUTATION primitives.
#
# These replicate the dense engine's ``ground_items`` write semantics exactly
# on every ``category != 0`` cell (empties are never captured by the sparse
# rep, and every dense reader gates on ``category != 0``, so empty-cell field
# residue is irrelevant — this is the equivalence contract validated in
# ``.test_runs/_verify_sparse_mutations.py``).
#
# Convention: a ``SparseGroundItems`` field is ``[B, L, K]`` where ``B`` is the
# dungeon-branch axis and ``L`` the level axis (NOT an env batch — env batching
# is an *outer* ``jax.vmap`` over the whole state).  Every primitive takes a
# single ``(b, lv, r, c[, slot])`` locator (may be traced) and uses only
# dynamic gather / dynamic-update-slice on the leading ``(b, lv)`` dims, so an
# outer ``jax.vmap`` over the env batch lowers cleanly (no data-dependent
# shapes, fixed K).
#
# K-list invariant (preserved by every primitive): each occupied dense cell
# ``(r, c, slot)`` appears in AT MOST ONE valid (``category != 0``) K-entry.
# ``sparse_to_dense`` scatters valid entries by flat index, so the *position*
# of an entry within the K-list is irrelevant to the dense image — only the
# SET of valid entries matters.  This is what makes in-place free-pad insertion
# byte-exact.
#
# Overflow policy (matches Phase 1 ``dense_to_sparse``): when a level already
# holds K valid entries and a new (r, c, slot) must be inserted, the survivor
# set is the K entries with the LOWEST flat index (``row*W*S + col*S + slot``).
# Insertion therefore evicts the current MAX-flat valid entry iff the incoming
# flat index is lower; otherwise the incoming write is dropped.  Repeated
# min-flat-preserving evictions converge to exactly the same survivor set as a
# fresh ``dense_to_sparse`` top_k selection (flat indices are unique — no ties).
# ===========================================================================


def _flat_of(rp, W, S):
    """Row-major flat index ``row*W*S + col*S + slot`` for pos rows ``rp[...,3]``."""
    return (rp[..., 0].astype(jnp.int32) * (W * S)
            + rp[..., 1].astype(jnp.int32) * S
            + rp[..., 2].astype(jnp.int32))


def _write_entry(sparse, b, lv, r, c, slot, field_vals, do_write):
    """Insert-or-overwrite ONE ``(r, c, slot)`` entry on level ``(b, lv)``.

    ``field_vals`` maps Item field name -> scalar value; ONLY those fields are
    written at the chosen K slot (all others keep the target slot's current
    value — for a fresh insert the target is a canonical-empty-fill pad, so the
    unwritten fields stay at empty-fill, matching the dense engine which leaves
    a freshly-written empty ground cell's unmentioned fields at their prior
    empty-fill value).  ``do_write`` (scalar bool) gates the whole write.

    Target selection (see overflow policy above):
      1. an existing valid entry at ``(r, c, slot)``  -> overwrite in place;
      2. else the first free pad (``category == 0``)  -> fresh insert;
      3. else (level full) evict the MAX-flat valid entry iff the incoming
         flat index is lower, else drop the write.
    """
    items, pos = sparse.items, sparse.pos
    W, S, K = sparse.W, sparse.S, sparse.K
    r = jnp.asarray(r, jnp.int32)
    c = jnp.asarray(c, jnp.int32)
    slot = jnp.asarray(slot, jnp.int32)

    rp = pos[b, lv].astype(jnp.int32)                    # [K,3]
    valid = items.category[b, lv] != 0                   # [K]
    ks = jnp.arange(K, dtype=jnp.int32)

    at_cell = valid & (rp[:, 0] == r) & (rp[:, 1] == c) & (rp[:, 2] == slot)
    exists = jnp.any(at_cell)
    exists_k = jnp.argmax(at_cell)                       # first match (0 if none)

    free = ~valid
    has_free = jnp.any(free)
    free_k = jnp.argmax(free)                            # first free (0 if none)

    flat_valid = jnp.where(valid, _flat_of(rp, W, S), jnp.int32(-1))
    max_k = jnp.argmax(flat_valid)                       # entry with largest flat
    new_flat = r * (W * S) + c * S + slot
    can_evict = new_flat < flat_valid[max_k]             # incoming beats worst survivor

    target = jnp.where(exists, exists_k,
              jnp.where(has_free, free_k,
              jnp.where(can_evict, max_k, ks[0])))
    will_write = do_write & (exists | has_free | can_evict)
    safe_t = jnp.clip(target, 0, K - 1)

    new_items = items
    for name, val in field_vals.items():
        arr = getattr(new_items, name)
        cur = arr[b, lv, safe_t]
        nv = jnp.where(will_write, jnp.asarray(val, _DTYPES[name]), cur)
        new_items = new_items.replace(**{name: arr.at[b, lv, safe_t].set(nv)})

    cur_pos = pos[b, lv, safe_t]
    new_pos_row = jnp.stack([r, c, slot]).astype(jnp.int16)
    nvp = jnp.where(will_write, new_pos_row, cur_pos)
    new_pos = pos.at[b, lv, safe_t].set(nvp)
    return sparse.replace(items=new_items, pos=new_pos)


def _clear_cell(sparse, b, lv, r, c):
    """Mark EVERY valid entry at tile ``(r, c)`` as ``category == 0`` (freed)."""
    r = jnp.asarray(r, jnp.int32)
    c = jnp.asarray(c, jnp.int32)
    rp = sparse.pos[b, lv].astype(jnp.int32)             # [K,3]
    cat = sparse.items.category[b, lv]                   # [K]
    m = (cat != 0) & (rp[:, 0] == r) & (rp[:, 1] == c)
    new_cat_row = jnp.where(m, jnp.asarray(0, _DTYPES["category"]), cat)
    new_cat = sparse.items.category.at[b, lv].set(new_cat_row)
    return sparse.replace(items=sparse.items.replace(category=new_cat))


def sparse_read_tile(sparse: SparseGroundItems, b, lv, r, c) -> Item:
    """Return the ``Item[S]`` stack at tile ``(r, c)`` (JITTABLE gather).

    Slot ``s`` reads the (unique) valid K-entry with ``pos == (r, c, s)``; if
    absent, that slot is canonical empty-fill (``category == 0``).  Mirrors a
    dense ``ground_items[b, lv, r, c, :]`` read.
    """
    S, K = sparse.S, sparse.K
    r = jnp.asarray(r, jnp.int32)
    c = jnp.asarray(c, jnp.int32)
    rp = sparse.pos[b, lv].astype(jnp.int32)             # [K,3]
    valid = sparse.items.category[b, lv] != 0            # [K]
    at_cell = valid & (rp[:, 0] == r) & (rp[:, 1] == c)  # [K]
    ks = jnp.arange(K, dtype=jnp.int32)
    slots = jnp.arange(S, dtype=jnp.int32)
    match = at_cell[None, :] & (rp[None, :, 2] == slots[:, None])  # [S,K]
    kk = jnp.max(jnp.where(match, ks[None, :], jnp.int32(-1)), axis=1)  # [S]
    present = kk >= 0
    safe_kk = jnp.clip(kk, 0, K - 1)

    out = {}
    for f in _FIELDS:
        col = getattr(sparse.items, f)[b, lv]            # [K]
        out[f] = jnp.where(present, col[safe_kk],
                           _FILLS[f].astype(_DTYPES[f]))  # [S]
    return Item(**out)


def sparse_clear_slot(sparse: SparseGroundItems, b, lv, r, c, slot) -> SparseGroundItems:
    """Zero the ``category`` of the entry at ``(r, c, slot)`` (JITTABLE).

    Matches the dense timer / monster / pickup clear
    (``category.at[b, lv, r, c, slot].set(0)``): it ONLY changes ``category``,
    leaving every other field stale.  Because ``sparse_to_dense`` drops
    ``category == 0`` entries and every reader gates on ``category != 0``, a
    zeroed sparse entry reproduces a dense ``category == 0`` cell exactly.
    """
    r = jnp.asarray(r, jnp.int32)
    c = jnp.asarray(c, jnp.int32)
    slot = jnp.asarray(slot, jnp.int32)
    rp = sparse.pos[b, lv].astype(jnp.int32)
    cat = sparse.items.category[b, lv]
    m = (cat != 0) & (rp[:, 0] == r) & (rp[:, 1] == c) & (rp[:, 2] == slot)
    new_cat_row = jnp.where(m, jnp.asarray(0, _DTYPES["category"]), cat)
    new_cat = sparse.items.category.at[b, lv].set(new_cat_row)
    return sparse.replace(items=sparse.items.replace(category=new_cat))


def sparse_pickup(sparse: SparseGroundItems, b, lv, r, c) -> SparseGroundItems:
    """Pick up the top of stack: zero ``category`` at ``(r, c, slot 0)``.

    Dense ``pickup`` (inventory.py) does NOT shift the stack down — it only
    sets ``category[b, lv, r, c, 0] = 0`` (other fields left stale).  This is
    exactly ``sparse_clear_slot(..., slot=0)``.
    """
    return sparse_clear_slot(sparse, b, lv, r, c, 0)


def sparse_set_cell(sparse: SparseGroundItems, b, lv, r, c,
                    item_stack: Item) -> SparseGroundItems:
    """Replace the entire ``S``-deep stack at tile ``(r, c)`` with ``item_stack``.

    ``item_stack`` is an ``Item`` whose every field is shape ``[S]``.  General
    primitive of which pickup/drop/clear are special cases.  Implementation:
    free every existing entry at ``(r, c)``, then (re)insert each occupied
    (``category != 0``) slot with its FULL field set.  Handles the cell having
    fewer/more items than before; K-overflow follows the module overflow
    policy (evict max-flat).  Byte-exact on ``category != 0`` cells: occupied
    slots carry ``item_stack``'s exact values, empty slots are canonical fill.
    """
    S = sparse.S
    out = _clear_cell(sparse, b, lv, r, c)
    for s in range(S):
        vals = {f: getattr(item_stack, f)[s] for f in _FIELDS}
        do = getattr(item_stack, "category")[s] != 0
        out = _write_entry(out, b, lv, r, c, s, vals, do)
    return out


def sparse_drop(sparse: SparseGroundItems, b, lv, r, c,
                item: Item) -> SparseGroundItems:
    """Drop a single inventory ``item`` onto tile ``(r, c)`` (JITTABLE).

    Replicates dense ``inventory.drop`` ground-side semantics EXACTLY:
      * scan stack slots ``0..S-1`` ascending for (a) first empty slot and
        (b) first mergeable slot (identity match on
        ``category/type_id/buc_status/enchantment/oerodeproof``);
      * target = merge slot if found, else first empty slot;
      * MERGE: only ``quantity += item.quantity`` and ``weight += item.weight``
        (identity fields untouched — matches dense);
      * FRESH: write the item's fields into the empty slot (unwritten fields
        stay empty-fill, matching a dense write into a canonical-empty cell);
      * if the stack is full and nothing merges, the drop is a no-op.

    ``item`` is a single-slot ``Item`` (scalar fields).  Caller is responsible
    for the inventory-side clear and all drop preconditions (loadstone / weld /
    levitation / altar BUC) exactly as the dense ``drop`` does — this primitive
    is the ground-write half only.
    """
    tile = sparse_read_tile(sparse, b, lv, r, c)         # Item[S]
    S = sparse.S
    slots = jnp.arange(S, dtype=jnp.int32)

    empty = tile.category == 0                            # [S]
    is_match = ((~empty)
                & (tile.category == item.category)
                & (tile.type_id == item.type_id)
                & (tile.buc_status == item.buc_status)
                & (tile.enchantment == item.enchantment)
                & (tile.oerodeproof == item.oerodeproof))  # [S]

    empty_found = jnp.any(empty)
    empty_pos = jnp.argmax(empty)
    merge_found = jnp.any(is_match)
    merge_pos = jnp.argmax(is_match)

    g_target = jnp.where(merge_found, merge_pos, empty_pos)
    can_drop = merge_found | empty_found                  # caller pre-gates has_item

    # Existing ground values at the (clipped) target slot, for the merge sums.
    safe_gt = jnp.clip(g_target, 0, S - 1)
    g_qty = tile.quantity[safe_gt].astype(jnp.int32)
    g_wt = tile.weight[safe_gt].astype(jnp.int32)

    merged_qty = (g_qty + item.quantity.astype(jnp.int32)).astype(_DTYPES["quantity"])
    merged_wt = (g_wt + item.weight.astype(jnp.int32)).astype(_DTYPES["weight"])

    do_merge = can_drop & merge_found
    # MERGE branch: touch only quantity + weight of the existing entry.
    merged = _write_entry(
        sparse, b, lv, r, c, safe_gt,
        {"quantity": merged_qty, "weight": merged_wt},
        do_merge,
    )
    # FRESH branch: full field write into the empty slot.
    do_fresh = can_drop & (~merge_found)
    fresh_vals = {f: getattr(item, f) for f in _FIELDS}
    fresh = _write_entry(
        merged, b, lv, r, c, safe_gt, fresh_vals, do_fresh,
    )
    return fresh
