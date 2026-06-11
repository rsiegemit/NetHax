"""Empirical test: does mineralize work under jax.vmap?

Per Phase 0 audit: mineralize is described as "already JAX-traceable" but
the vmap path was never empirically validated. This script does a direct
A/B between a serial loop over mineralize and jax.vmap(mineralize) on the
same 4 seeds.

Signature (from Nethax/nethax/dungeon/mineralize.py):
    mineralize(
        terrain,                # int8[ROWNO=21, COLNO=80]
        vendor_rng: Isaac64State,
        depth: int = 1,
        dunlev: int = 1,
        ...                     # static booleans and -1 sentinels
    ) -> (terrain, updated_vendor_rng)

In the no-emit-objects path mineralize only consumes RNG draws (it does
not mutate ground_items).  We test that path here — terrain is the only
non-RNG input; for vmap we vary terrain across replicas to verify the
function vmaps correctly under data-dependent eligibility predicates
inside its lax.fori_loop bodies.

Notes
-----
* We construct an empty all-VOID terrain plus a small floor-patched
  variant so the eligibility predicate yields a non-trivial mask.  Even
  with all-VOID terrain the loop still fires draws (cells in the inner
  scan region are all STONE → 100% eligible), so this is a meaningful
  stress test of the fori_loop / lax.cond paths under vmap.
* We do NOT exercise the ground_items emission path — keeps the test
  small and avoids constructing full EnvState pytrees.  The pure RNG
  consumption path is what the reset call site uses inside lax.cond, so
  this matches the vmap-safety question.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
# NOTE: do NOT add REPO_ROOT/Nethax to sys.path — see test_descr_shuffle_vmap.py.

from Nethax.nethax.vendor_rng import init_jax, Isaac64State  # noqa: E402
from Nethax.nethax.dungeon.mineralize import mineralize  # noqa: E402

# Map geometry — matches mineralize.py constants.
_ROWNO = 21
_COLNO = 80


def extract_uint64_seed(rng: jax.Array) -> jax.Array:
    hi = rng[0].astype(jnp.uint64)
    lo = rng[1].astype(jnp.uint64)
    return (hi << jnp.uint64(32)) | lo


def report_rng_diff(serial_rng: Isaac64State, vmap_rng: Isaac64State, batch_size: int, label: str) -> bool:
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


def report_terrain_diff(serial_terrain: jax.Array, vmap_terrain: jax.Array, batch_size: int, label: str) -> bool:
    if serial_terrain.shape != vmap_terrain.shape:
        print(f"  [{label}] terrain SHAPE MISMATCH serial={serial_terrain.shape} vmap={vmap_terrain.shape}")
        return False
    eq = bool(jnp.array_equal(serial_terrain, vmap_terrain))
    if eq:
        print(f"  [{label}] terrain: OK  (shape={serial_terrain.shape} dtype={serial_terrain.dtype})")
        return True
    print(f"  [{label}] terrain: DIVERGENT (shape={serial_terrain.shape} dtype={serial_terrain.dtype})")
    for i in range(batch_size):
        ei = bool(jnp.array_equal(serial_terrain[i], vmap_terrain[i]))
        if not ei:
            dc = int(jnp.sum(serial_terrain[i] != vmap_terrain[i]))
            print(f"    [{label}] replica {i}: terrain differs ({dc} mismatched elements)")
    return False


def make_test_terrain(replica: int) -> jax.Array:
    """Build a small int8[21,80] terrain that varies per replica.

    Replica 0: all VOID (every inner cell is fully-stone → 100% eligible).
    Replica 1: a 5x5 FLOOR room near the centre (carves out an interior).
    Replica 2: a vertical FLOOR strip down the middle.
    Replica 3: a single FLOOR cell at (10, 40).
    """
    terrain = jnp.zeros((_ROWNO, _COLNO), dtype=jnp.int8)
    FLOOR = jnp.int8(1)
    if replica == 1:
        terrain = terrain.at[8:13, 38:43].set(FLOOR)
    elif replica == 2:
        terrain = terrain.at[1:_ROWNO - 1, 40].set(FLOOR)
    elif replica == 3:
        terrain = terrain.at[10, 40].set(FLOOR)
    return terrain


def main() -> int:
    print("=" * 60)
    print("Test: jax.vmap(mineralize) byte-parity vs serial loop")
    print("=" * 60)

    # 1. Per-replica seeds and terrains.
    keys = jnp.stack([jax.random.PRNGKey(i) for i in range(4)])
    seeds = jnp.stack([extract_uint64_seed(keys[i]) for i in range(4)])
    print(f"seeds: {[int(s) for s in seeds]}")

    batched_rng = jax.vmap(init_jax)(seeds)
    print(f"batched_rng.m.shape = {batched_rng.m.shape}")

    per_terrain = [make_test_terrain(i) for i in range(4)]
    batched_terrain = jnp.stack(per_terrain, axis=0)
    print(f"batched_terrain.shape = {batched_terrain.shape}")
    print()

    # Wrap mineralize so we only vmap over (terrain, vendor_rng); all other
    # args are static defaults.  Keep depth/dunlev as Python int constants so
    # mineralize's interior int-path stays unchanged (it converts to jnp.int32
    # inside).
    def _mineralize_call(terrain, rng):
        return mineralize(terrain, rng, depth=1, dunlev=1)

    # 2. Serial.
    print("-- Serial loop --")
    per_seed_rng = [
        jax.tree_util.tree_map(lambda x, i=i: x[i], batched_rng) for i in range(4)
    ]
    try:
        per_results = [
            _mineralize_call(per_terrain[i], per_seed_rng[i]) for i in range(4)
        ]
        per_terr_out = [r[0] for r in per_results]
        per_rng_out = [r[1] for r in per_results]
        serial_terrain = jnp.stack(per_terr_out, axis=0)
        serial_rng = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *per_rng_out
        )
        print(
            f"  serial OK; terrain shape={serial_terrain.shape}, rng.m shape={serial_rng.m.shape}"
        )
        # Quick sanity: confirm draws differ across replicas (else the test is
        # trivially passing because mineralize didn't actually do anything).
        per_draws = [int(r.draws) for r in per_rng_out]
        print(f"  per-replica draws after mineralize: {per_draws}")
    except Exception as e:
        print(f"  serial mineralize FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 2
    print()

    # 3. eager vmap.
    print("-- jax.vmap(mineralize)(batched_terrain, batched_rng) --")
    eq_vmap_t = None
    eq_vmap_r = None
    try:
        vmap_call = jax.vmap(_mineralize_call, in_axes=(0, 0))
        vmap_terrain, vmap_rng = vmap_call(batched_terrain, batched_rng)
        print(
            f"  vmap call SUCCEEDED; terrain shape={vmap_terrain.shape}, rng.m shape={vmap_rng.m.shape}"
        )
        eq_vmap_t = report_terrain_diff(serial_terrain, vmap_terrain, 4, "vmap")
        eq_vmap_r = report_rng_diff(serial_rng, vmap_rng, 4, "vmap")
    except Exception as e:
        print(f"  vmap(mineralize) RAISED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 3
    print()

    # 4. jit(vmap(...)).
    print("-- jax.jit(jax.vmap(mineralize))(batched_terrain, batched_rng) --")
    eq_jit_t = None
    eq_jit_r = None
    try:
        jit_call = jax.jit(jax.vmap(_mineralize_call, in_axes=(0, 0)))
        jit_terrain, jit_rng = jit_call(batched_terrain, batched_rng)
        print(
            f"  jit(vmap) call SUCCEEDED; terrain shape={jit_terrain.shape}, rng.m shape={jit_rng.m.shape}"
        )
        eq_jit_t = report_terrain_diff(serial_terrain, jit_terrain, 4, "jit")
        eq_jit_r = report_rng_diff(serial_rng, jit_rng, 4, "jit")
    except Exception as e:
        print(f"  jit(vmap(mineralize)) RAISED: {type(e).__name__}: {e}")
        traceback.print_exc()
    print()

    print("=" * 60)
    print(
        f"SUMMARY  vmap_terrain_eq={eq_vmap_t}  vmap_rng_eq={eq_vmap_r}  "
        f"jit_terrain_eq={eq_jit_t}  jit_rng_eq={eq_jit_r}"
    )
    print("=" * 60)

    ok = bool(eq_vmap_t) and bool(eq_vmap_r)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
