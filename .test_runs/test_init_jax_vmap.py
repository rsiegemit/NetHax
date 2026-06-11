"""Empirical test: does vendor_rng.init_jax work under jax.vmap?

Per Phase 0 audit: init_jax is "verified bit-exact for seeds 0..1000" but
the vmap path was never tested. This script does a direct A/B between a
serial loop over init_jax and jax.vmap(init_jax) on the same 4 seeds.
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

# Make sure repo root is on path (PYTHONPATH=. should already do it, but be defensive).
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "Nethax"))

from nethax.vendor_rng import init_jax, Isaac64State  # noqa: E402


def extract_uint64_seed(rng: jax.Array) -> jax.Array:
    """Pack a 2xuint32 PRNGKey into one uint64 seed."""
    hi = rng[0].astype(jnp.uint64)
    lo = rng[1].astype(jnp.uint64)
    return (hi << jnp.uint64(32)) | lo


def report_pytree_diff(serial_state: Isaac64State, vmap_state: Isaac64State, batch_size: int) -> bool:
    """Returns True iff serial and vmap results are byte-equal for every replica."""
    fields = ["m", "r", "a", "b", "c", "n", "draws"]
    all_equal = True
    for f in fields:
        s = getattr(serial_state, f)
        v = getattr(vmap_state, f)
        if s.shape != v.shape:
            print(f"  FIELD {f}: SHAPE MISMATCH serial={s.shape} vmap={v.shape}")
            all_equal = False
            continue
        if s.dtype != v.dtype:
            print(f"  FIELD {f}: DTYPE MISMATCH serial={s.dtype} vmap={v.dtype}")
            all_equal = False
        eq = bool(jnp.array_equal(s, v))
        if eq:
            print(f"  FIELD {f}: OK  (shape={s.shape} dtype={s.dtype})")
        else:
            all_equal = False
            print(f"  FIELD {f}: DIVERGENT (shape={s.shape} dtype={s.dtype})")
            # per-replica diff
            for i in range(batch_size):
                ei = bool(jnp.array_equal(s[i], v[i]))
                if not ei:
                    diff_count = int(jnp.sum(s[i] != v[i])) if s[i].ndim > 0 else int(s[i] != v[i])
                    print(f"    replica {i}: differs ({diff_count} mismatched elements)")
    return all_equal


def main() -> int:
    print("=" * 60)
    print("Test: jax.vmap(init_jax) byte-parity vs serial loop")
    print("=" * 60)

    # 1. 4 PRNGKeys -> 4 uint64 seeds.
    keys = jnp.stack([jax.random.PRNGKey(i) for i in range(4)])
    print(f"keys shape={keys.shape} dtype={keys.dtype}")
    seeds = jnp.stack([extract_uint64_seed(keys[i]) for i in range(4)])
    print(f"seeds shape={seeds.shape} dtype={seeds.dtype}")
    print(f"seed values: {[int(s) for s in seeds]}")
    print()

    # 2. Serial: call init_jax per seed, stack pytrees.
    print("-- Serial loop --")
    try:
        per_seed_states = [init_jax(seeds[i]) for i in range(4)]
        serial_stacked = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *per_seed_states
        )
        print(f"  serial OK; m shape={serial_stacked.m.shape}")
    except Exception as e:
        print(f"  serial init_jax FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 2
    print()

    # 3. vmap test.
    print("-- jax.vmap(init_jax)(seeds) --")
    vmap_state = None
    try:
        vmap_state = jax.vmap(init_jax)(seeds)
        print(f"  vmap call SUCCEEDED; m shape={vmap_state.m.shape}")
    except Exception as e:
        print(f"  vmap(init_jax) RAISED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 3

    print("-- pytree comparison serial vs vmap --")
    eq_vmap = report_pytree_diff(serial_stacked, vmap_state, batch_size=4)
    print(f"  vmap == serial: {eq_vmap}")
    print()

    # 4. JIT-compiled vmap.
    print("-- jax.jit(jax.vmap(init_jax))(seeds) --")
    eq_jit = None
    try:
        jit_vmap = jax.jit(jax.vmap(init_jax))
        jit_state = jit_vmap(seeds)
        print(f"  jit(vmap(init_jax)) call SUCCEEDED; m shape={jit_state.m.shape}")
        eq_jit = report_pytree_diff(serial_stacked, jit_state, batch_size=4)
        print(f"  jit(vmap) == serial: {eq_jit}")
    except Exception as e:
        print(f"  jit(vmap(init_jax)) RAISED: {type(e).__name__}: {e}")
        traceback.print_exc()
    print()

    # 5. vmap over traced seeds (output of another jax op).
    print("-- vmap with traced seeds (from prior jax op) --")
    eq_traced = None
    try:
        def derive_seeds(base_seeds: jax.Array) -> jax.Array:
            # arbitrary deterministic transform: identity + 0 keeps values but
            # forces the seed to be a *traced* intermediate, not a literal.
            return base_seeds + jnp.zeros_like(base_seeds)
        traced_pipeline = jax.jit(lambda s: jax.vmap(init_jax)(derive_seeds(s)))
        traced_state = traced_pipeline(seeds)
        print(f"  jit(vmap(init_jax) o derive) SUCCEEDED; m shape={traced_state.m.shape}")
        eq_traced = report_pytree_diff(serial_stacked, traced_state, batch_size=4)
        print(f"  traced-seeds vmap == serial: {eq_traced}")
    except Exception as e:
        print(f"  traced-seeds path RAISED: {type(e).__name__}: {e}")
        traceback.print_exc()
    print()

    print("=" * 60)
    print(f"SUMMARY  eager_vmap_eq={eq_vmap}  jit_vmap_eq={eq_jit}  traced_vmap_eq={eq_traced}")
    print("=" * 60)

    return 0 if eq_vmap else 1


if __name__ == "__main__":
    sys.exit(main())
