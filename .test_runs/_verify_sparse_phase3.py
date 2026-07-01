"""PHASE 3 verification: sparse ground_items as SOLE EnvState storage.

Covers Deliverable A (bridge helpers, jit + vmap round-trip), Deliverable B
(EnvState.ground_items is a SparseGroundItems after reset), and Deliverable C
(build_glyphs runs on the reset state + state size shrink).

Run:
  JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python \
      .test_runs/_verify_sparse_phase3.py
"""
import numpy as np
import jax
import jax.numpy as jnp

import gym, gymnasium, gymnasium.spaces.dict as _g  # noqa: E402
_g.Space = (gymnasium.spaces.Space, gym.spaces.Space)

from Nethax.minihax.minihax_env import MinihaxEnv
from Nethax.nethax.obs.nle_obs import build_glyphs
from Nethax.nethax.subsystems.inventory import (
    _empty_dense_ground_items, MAX_GROUND_STACK,
)
from Nethax.nethax.subsystems.ground_items_sparse import (
    SparseGroundItems, dense_to_sparse, sparse_to_dense,
    sparse_to_dense_level, replace_level,
)

_FIELDS = list(_empty_dense_ground_items(1, 1, 1, 1).__dict__.keys())


def _cat_equal_on_occupied(a: SparseGroundItems, b: SparseGroundItems) -> bool:
    da = np.asarray(sparse_to_dense(a).category)
    db = np.asarray(sparse_to_dense(b).category)
    return bool(np.array_equal(da, db))


def _make_sparse():
    B, L, H, W = 2, 3, 6, 7
    d = _empty_dense_ground_items(B, L, H, W)
    d = d.replace(
        category=d.category.at[1, 2, 4, 5, 0].set(5).at[1, 2, 4, 5, 1].set(2)
                          .at[0, 1, 2, 3, 0].set(7),
        type_id=d.type_id.at[1, 2, 4, 5, 0].set(100).at[0, 1, 2, 3, 0].set(42),
        quantity=d.quantity.at[1, 2, 4, 5, 0].set(3),
    )
    return dense_to_sparse(d, 16)


def main():
    print("=== PHASE 3 sparse ground_items verification ===", flush=True)

    # ---------- (A) bridge-helper round-trip under jit AND vmap ----------
    sp = _make_sparse()

    f_lvl = jax.jit(sparse_to_dense_level)
    f_rep = jax.jit(replace_level)
    sp_rt = f_rep(sp, 1, 2, f_lvl(sp, 1, 2))
    a_jit = (
        _cat_equal_on_occupied(sp, sp_rt)
        and np.array_equal(np.asarray(sp.items.category),
                           np.asarray(sp_rt.items.category))
        and np.array_equal(np.asarray(sp.pos), np.asarray(sp_rt.pos))
    )

    batch = jax.tree_util.tree_map(lambda x: jnp.stack([x, x, x]), sp)
    def _rt(s):
        return replace_level(s, 1, 2, sparse_to_dense_level(s, 1, 2))
    out = jax.jit(jax.vmap(_rt))(batch)
    a_vmap = np.array_equal(np.asarray(batch.items.category),
                            np.asarray(out.items.category)) and \
             np.array_equal(np.asarray(batch.pos), np.asarray(out.pos))

    print(f"[A] bridge helper round-trip  jit={'PASS' if a_jit else 'FAIL'}  "
          f"vmap={'PASS' if a_vmap else 'FAIL'}", flush=True)

    # ---------- (B) reset -> SparseGroundItems ; (C) build_glyphs + size ----------
    e = MinihaxEnv("MiniHack-Room-5x5-v0")
    s, _ = e.reset(jax.random.key(0))
    gi = s.ground_items
    b_ok = isinstance(gi, SparseGroundItems)
    print(f"[B] reset .ground_items is SparseGroundItems: "
          f"{'PASS' if b_ok else 'FAIL'}  type={type(gi).__name__}", flush=True)
    print(f"    items.category shape={gi.items.category.shape} "
          f"pos shape={gi.pos.shape}  H,W,S,K={gi.H},{gi.W},{gi.S},{gi.K}",
          flush=True)

    mb = sum(l.nbytes for l in jax.tree_util.tree_leaves(s)) / 1e6
    print(f"[C] per-env state size = {mb:.2f} MB  (was 124.81; "
          f"ground_items alone was 111.97)", flush=True)

    g = np.asarray(build_glyphs(s))
    c_ok = g.shape == (21, 79)
    print(f"[C] build_glyphs on reset state: "
          f"{'PASS' if c_ok else 'FAIL'}  shape={g.shape}", flush=True)

    ok = a_jit and a_vmap and b_ok and c_ok
    print(f"\n[SUMMARY] {'ALL PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
