"""Empirical test: does compute_descr_shuffle work under jax.vmap?

Per Phase 0 audit: compute_descr_shuffle is described as "already
JAX-traceable" but the vmap path was never empirically validated. This
script does a direct A/B between a serial loop over compute_descr_shuffle
and jax.vmap(compute_descr_shuffle) on the same 4 seeds.

Signature:
    compute_descr_shuffle(rng: Isaac64State) -> (Isaac64State, int16[453])

Mirrors test_init_jax_vmap.py style.
"""
from __future__ import annotations

import os
import sys
import traceback

# Force CPU and 64-bit before importing jax.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
# NOTE: do NOT add REPO_ROOT/Nethax to sys.path — it would let Python
# import `nethax.*` (lowercase) AS WELL AS `Nethax.nethax.*`, creating
# two distinct module objects with two distinct `Isaac64State` classes.
# Inside `lax.cond`, the type mismatch then raises:
#   "true_fun output is Nethax.nethax.vendor_rng.Isaac64State but
#    false_fun output is nethax.vendor_rng.Isaac64State"

from Nethax.nethax.vendor_rng import init_jax, Isaac64State  # noqa: E402
from Nethax.nethax.obs.glyph_shuffle import compute_descr_shuffle  # noqa: E402


def extract_uint64_seed(rng: jax.Array) -> jax.Array:
    hi = rng[0].astype(jnp.uint64)
    lo = rng[1].astype(jnp.uint64)
    return (hi << jnp.uint64(32)) | lo


def report_rng_diff(
    serial_rng: Isaac64State, vmap_rng: Isaac64State, batch_size: int, label: str
) -> bool:
    fields = ["m", "r", "a", "b", "c", "n", "draws"]
    all_equal = True
    for f in fields:
        s = getattr(serial_rng, f)
        v = getattr(vmap_rng, f)
        if s.shape != v.shape:
            print(f"  [{label}] FIELD {f}: SHAPE MISMATCH serial={s.shape} vmap={v.shape}")
            all_equal = False
            continue
        if s.dtype != v.dtype:
            print(f"  [{label}] FIELD {f}: DTYPE MISMATCH serial={s.dtype} vmap={v.dtype}")
            all_equal = False
        eq = bool(jnp.array_equal(s, v))
        if eq:
            print(f"  [{label}] FIELD {f}: OK  (shape={s.shape} dtype={s.dtype})")
        else:
            all_equal = False
            print(f"  [{label}] FIELD {f}: DIVERGENT (shape={s.shape} dtype={s.dtype})")
            for i in range(batch_size):
                ei = bool(jnp.array_equal(s[i], v[i]))
                if not ei:
                    diff_count = int(jnp.sum(s[i] != v[i])) if s[i].ndim > 0 else int(s[i] != v[i])
                    print(f"    [{label}] replica {i}: differs ({diff_count} mismatched elements)")
    return all_equal


def report_descr_diff(serial_descr: jax.Array, vmap_descr: jax.Array, batch_size: int, label: str) -> bool:
    if serial_descr.shape != vmap_descr.shape:
        print(f"  [{label}] descr_idx SHAPE MISMATCH serial={serial_descr.shape} vmap={vmap_descr.shape}")
        return False
    if serial_descr.dtype != vmap_descr.dtype:
        print(f"  [{label}] descr_idx DTYPE MISMATCH serial={serial_descr.dtype} vmap={vmap_descr.dtype}")
    eq = bool(jnp.array_equal(serial_descr, vmap_descr))
    if eq:
        print(f"  [{label}] descr_idx: OK  (shape={serial_descr.shape} dtype={serial_descr.dtype})")
        return True
    print(f"  [{label}] descr_idx: DIVERGENT (shape={serial_descr.shape} dtype={serial_descr.dtype})")
    for i in range(batch_size):
        ei = bool(jnp.array_equal(serial_descr[i], vmap_descr[i]))
        if not ei:
            dc = int(jnp.sum(serial_descr[i] != vmap_descr[i]))
            print(f"    [{label}] replica {i}: descr_idx differs ({dc} mismatched elements)")
    return False


def main() -> int:
    print("=" * 60)
    print("Test: jax.vmap(compute_descr_shuffle) byte-parity vs serial loop")
    print("=" * 60)

    # 1. 4 PRNGKeys -> 4 uint64 seeds -> 4 Isaac64States.
    keys = jnp.stack([jax.random.PRNGKey(i) for i in range(4)])
    seeds = jnp.stack([extract_uint64_seed(keys[i]) for i in range(4)])
    print(f"seeds: {[int(s) for s in seeds]}")

    # Build batched Isaac64State via vmap(init_jax) (already verified).
    batched_state = jax.vmap(init_jax)(seeds)
    print(f"batched_state.m.shape = {batched_state.m.shape}")
    print()

    # 2. Serial: unbatch, call compute_descr_shuffle per replica, restack.
    print("-- Serial loop --")
    per_seed_inputs = []
    for i in range(4):
        per_seed_inputs.append(
            jax.tree_util.tree_map(lambda x, i=i: x[i], batched_state)
        )
    try:
        per_seed_results = [compute_descr_shuffle(per_seed_inputs[i]) for i in range(4)]
        per_rng = [r[0] for r in per_seed_results]
        per_descr = [r[1] for r in per_seed_results]
        serial_rng = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *per_rng)
        serial_descr = jnp.stack(per_descr, axis=0)
        print(f"  serial OK; rng.m shape={serial_rng.m.shape}, descr shape={serial_descr.shape}")
    except Exception as e:
        print(f"  serial compute_descr_shuffle FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 2
    print()

    # 3. eager vmap.
    print("-- jax.vmap(compute_descr_shuffle)(batched_state) --")
    eq_vmap_rng = None
    eq_vmap_descr = None
    try:
        vmap_rng, vmap_descr = jax.vmap(compute_descr_shuffle)(batched_state)
        print(f"  vmap call SUCCEEDED; rng.m shape={vmap_rng.m.shape}, descr shape={vmap_descr.shape}")
        eq_vmap_rng = report_rng_diff(serial_rng, vmap_rng, 4, "vmap")
        eq_vmap_descr = report_descr_diff(serial_descr, vmap_descr, 4, "vmap")
    except Exception as e:
        print(f"  vmap(compute_descr_shuffle) RAISED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 3
    print()

    # 4. JIT-compiled vmap.
    print("-- jax.jit(jax.vmap(compute_descr_shuffle))(batched_state) --")
    eq_jit_rng = None
    eq_jit_descr = None
    try:
        jit_vmap = jax.jit(jax.vmap(compute_descr_shuffle))
        jit_rng, jit_descr = jit_vmap(batched_state)
        print(f"  jit(vmap) call SUCCEEDED; rng.m shape={jit_rng.m.shape}, descr shape={jit_descr.shape}")
        eq_jit_rng = report_rng_diff(serial_rng, jit_rng, 4, "jit")
        eq_jit_descr = report_descr_diff(serial_descr, jit_descr, 4, "jit")
    except Exception as e:
        print(f"  jit(vmap) RAISED: {type(e).__name__}: {e}")
        traceback.print_exc()
    print()

    print("=" * 60)
    print(
        f"SUMMARY  vmap_rng_eq={eq_vmap_rng}  vmap_descr_eq={eq_vmap_descr}  "
        f"jit_rng_eq={eq_jit_rng}  jit_descr_eq={eq_jit_descr}"
    )
    print("=" * 60)

    ok = bool(eq_vmap_rng) and bool(eq_vmap_descr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
