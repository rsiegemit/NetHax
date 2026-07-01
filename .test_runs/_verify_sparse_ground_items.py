"""PHASE 1 verification for sparse ground_items primitive + sparse obs path.

Run:
  JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python \
      .test_runs/_verify_sparse_ground_items.py
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

import gym, gymnasium, gymnasium.spaces.dict as _g  # noqa: E402
_g.Space = (gymnasium.spaces.Space, gym.spaces.Space)

from Nethax.minihax.minihax_env import MinihaxEnv
from Nethax.nethax.obs.nle_obs import build_glyphs
from Nethax.nethax.subsystems.inventory import (
    Item, MAX_GROUND_STACK, _empty_ground_items_array,
)
from Nethax.nethax.subsystems.ground_items_sparse import (
    dense_to_sparse, sparse_to_dense, SparseGroundItems,
)
from Nethax.nethax.dungeon.branches import (
    N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W,
)

FIELDS = [f for f in _empty_ground_items_array(1, 1, 1, 1).__dict__.keys()]
DTYPES = {f: np.asarray(getattr(_empty_ground_items_array(1, 1, 1, 1), f)).dtype
          for f in FIELDS}


def items_equal_full(a, b):
    for f in FIELDS:
        av, bv = np.asarray(getattr(a, f)), np.asarray(getattr(b, f))
        if not np.array_equal(av, bv):
            return False, f
    return True, None


# JITTED primitive: round-trip dense -> sparse -> dense
@jax.jit
def _roundtrip(gi):
    return sparse_to_dense(dense_to_sparse(gi, 64))


def inject_multi_and_stack(state, rng):
    """Inject several distinct items across cells + one full-depth stack."""
    br = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    gi = state.ground_items
    fields = {f: np.asarray(getattr(gi, f)).copy() for f in FIELDS}
    vis = np.asarray(state.visible[:MAP_H, :MAP_W])
    rows, cols = np.nonzero(vis)
    if len(rows) == 0:
        gr, gc = np.meshgrid(np.arange(MAP_H), np.arange(MAP_W), indexing="ij")
        rows, cols = gr.ravel(), gc.ravel()
    # scatter 20 single items over distinct visible cells
    order = np.asarray(jax.random.permutation(rng, len(rows)))
    for j in range(min(20, len(order))):
        i = order[j]
        r, c = int(rows[i]), int(cols[i])
        fields["category"][br, lv, r, c, 0] = (j % 7) + 1
        fields["type_id"][br, lv, r, c, 0] = (j * 13) % 400
        fields["quantity"][br, lv, r, c, 0] = (j % 3) + 1
        fields["enchantment"][br, lv, r, c, 0] = (j % 5) - 2
        fields["buc_status"][br, lv, r, c, 0] = (j % 4)
    # one cell with a full MAX_GROUND_STACK-deep pile
    i = order[min(20, len(order) - 1)]
    r, c = int(rows[i]), int(cols[i])
    for s in range(MAX_GROUND_STACK):
        fields["category"][br, lv, r, c, s] = (s % 7) + 1
        fields["type_id"][br, lv, r, c, s] = 100 + s * 7
        fields["quantity"][br, lv, r, c, s] = s + 1
        fields["weight"][br, lv, r, c, s] = (s + 1) * 3
    new_gi = Item(**{f: jnp.asarray(fields[f]) for f in FIELDS})
    return state.replace(ground_items=new_gi)


def occ(gi):
    return int((np.asarray(gi.category) != 0).sum())


def main():
    print(f"=== PHASE 1 sparse ground_items verification ===", flush=True)
    print(f"dims B={N_BRANCHES} L={MAX_LEVELS_PER_BRANCH} H={MAP_H} W={MAP_W} "
          f"S={MAX_GROUND_STACK}  K=64", flush=True)

    envs = ["MiniHack-Room-5x5-v0", "MiniHack-Room-Trap-5x5-v0"]

    # ---------- (1) round-trip under JIT ----------
    rt_ok = True
    n_rt = 0
    all_states = []
    for en in envs:
        e = MinihaxEnv(en)
        for seed in range(3):
            s0, _ = e.reset(jax.random.key(seed))
            states = [("reset", s0)]
            s = s0
            for t in range(3):
                s, _, _, _ = e.step(s, 0, jax.random.key(1000 + t))
                states.append((f"step{t+1}", s))
            states.append(("injected", inject_multi_and_stack(s0, jax.random.key(77 + seed))))
            for label, st in states:
                recon = _roundtrip(st.ground_items)          # JITTED
                eq, bad = items_equal_full(st.ground_items, recon)
                n_rt += 1
                if not eq:
                    rt_ok = False
                    print(f"  ROUND-TRIP FAIL {en}/{seed}/{label} field={bad} "
                          f"occ={occ(st.ground_items)}", flush=True)
                all_states.append((en, seed, label, st))
    print(f"[1] traceable round-trip (jax.jit) over {n_rt} states: "
          f"{'PASS' if rt_ok else 'FAIL'}", flush=True)

    # ---------- (2) obs byte-identical flag ON vs OFF ----------
    # build_glyphs reads os.environ at trace time -> jit twice (on/off graphs).
    os.environ.pop("NETHAX_SPARSE_GROUND", None)
    gf_off = jax.jit(build_glyphs)
    bp_ok = True
    n_bp = 0
    # Precompute OFF glyphs first (flag unset), then set flag and compute ON.
    off_glyphs = [np.asarray(gf_off(st)) for (_, _, _, st) in all_states]
    os.environ["NETHAX_SPARSE_GROUND"] = "1"
    os.environ["NETHAX_SPARSE_K"] = "64"
    gf_on = jax.jit(build_glyphs)
    for (en, seed, label, st), g_off in zip(all_states, off_glyphs):
        g_on = np.asarray(gf_on(st))
        n_bp += 1
        if not np.array_equal(g_off, g_on):
            bp_ok = False
            diff = int((g_off != g_on).sum())
            print(f"  OBS PARITY FAIL {en}/{seed}/{label} ({diff} cells)", flush=True)
    os.environ.pop("NETHAX_SPARSE_GROUND", None)
    os.environ.pop("NETHAX_SPARSE_K", None)
    print(f"[2] build_glyphs byte-identical flag ON vs OFF over {n_bp} states: "
          f"{'PASS' if bp_ok else 'FAIL'}", flush=True)

    # ---------- (4) size report ----------
    bytes_per_cell = sum(np.dtype(DTYPES[f]).itemsize for f in FIELDS)
    # canonical full dims [7,32,21,80,8] to match the 111.97 MB reference
    B, L, H, W, S = 7, 32, 21, 80, 8
    K = 64
    dense_cells = B * L * H * W * S
    dense_mb = dense_cells * bytes_per_cell / 1048576
    pos_bytes = 3 * 2
    sparse_mb = B * L * K * (bytes_per_cell + pos_bytes) / 1048576
    print(f"[4] bytes/cell(27 fields)={bytes_per_cell}  "
          f"canonical dims [7,32,21,80,8]", flush=True)
    print(f"[4] dense  ground_items = {dense_mb:.2f} MB", flush=True)
    print(f"[4] sparse ground_items = {sparse_mb:.4f} MB (K=64, incl pos)", flush=True)
    print(f"[4] shrink factor       = {dense_mb / sparse_mb:.1f}x", flush=True)

    print(f"\n[SUMMARY] roundtrip={'PASS' if rt_ok else 'FAIL'}  "
          f"obs_parity={'PASS' if bp_ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
