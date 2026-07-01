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
