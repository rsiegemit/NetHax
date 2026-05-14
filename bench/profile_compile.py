"""Compile-time profiler for ``env._step_jit``.

Times XLA cold-compile for three call shapes on CPU:

    1. ``jax.jit(_step_impl)`` — single-env step.
    2. ``jax.jit(jax.vmap(_step_impl))`` — batched step.
    3. ``jax.jit`` wrapping a 16-step ``lax.scan`` of ``_step_impl`` — the
       PPO-style rollout shape that triggers the pathological compile.

Each measurement uses a freshly-constructed function so XLA caches do not
leak between runs.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python bench/profile_compile.py

The script prints a small summary table and writes a JSON record to
``bench/results/compile_profile.json`` so successive runs can be diffed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make sure JAX picks CPU regardless of caller environment.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Project root on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import jax
import jax.numpy as jnp

from Nethax.nethax.env import NethaxEnv, _step_impl


def _wall(t0: float) -> float:
    return time.perf_counter() - t0


def _compile_single(state, action, key) -> float:
    """Cold-compile ``jit(_step_impl)`` and time it."""
    fn = jax.jit(_step_impl)
    t0 = time.perf_counter()
    out = fn(state, action, key)
    jax.block_until_ready(out[0])
    return _wall(t0)


def _compile_vmap(state, action, key, batch: int = 4) -> float:
    """Cold-compile ``jit(vmap(_step_impl))`` over a small batch."""
    fn = jax.jit(jax.vmap(_step_impl))
    batched_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (batch,) + x.shape), state
    )
    actions = jnp.full((batch,), int(action), dtype=jnp.int32)
    keys = jax.random.split(key, batch)
    t0 = time.perf_counter()
    out = fn(batched_state, actions, keys)
    jax.block_until_ready(out[0])
    return _wall(t0)


def _compile_scan(state, action, key, n_steps: int = 16) -> float:
    """Cold-compile ``jit(lax.scan(_step_impl, ...))``.

    ``n_steps`` is small (16) on purpose: the goal is to measure compile
    overhead, not execution.  XLA caches don't depend on the scan length.
    """

    def _body(carry, _):
        s, k = carry
        k, sub = jax.random.split(k)
        a = jnp.int32(int(action))
        ns, _obs, _r, _d = _step_impl(s, a, sub)
        return (ns, k), None

    def _rollout(s, k):
        (final_s, _), _ = jax.lax.scan(_body, (s, k), None, length=n_steps)
        return final_s

    fn = jax.jit(_rollout)
    t0 = time.perf_counter()
    out = fn(state, key)
    jax.block_until_ready(out)
    return _wall(t0)


def _compile_vmap_scan(state, key, n_steps: int = 16, batch: int = 4) -> float:
    """Cold-compile ``jit(vmap(scan(_step_impl, ...)))``.

    This is the PPO-style training shape: batch of envs each stepping a
    fixed-length rollout.  Previously unbounded (>30 min, killed); the
    Wave 8 refactor brings it down to tens of seconds.
    """

    def _body(carry, _):
        s, k = carry
        k, sub = jax.random.split(k)
        ns, _obs, _r, _d = _step_impl(s, jnp.int32(0), sub)
        return (ns, k), None

    def _rollout(s, k):
        (final_s, _), _ = jax.lax.scan(_body, (s, k), None, length=n_steps)
        return final_s

    fn = jax.jit(jax.vmap(_rollout))
    batched_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (batch,) + x.shape), state
    )
    keys = jax.random.split(key, batch)
    t0 = time.perf_counter()
    out = fn(batched_state, keys)
    jax.block_until_ready(out)
    return _wall(t0)


def _build_state():
    env = NethaxEnv()
    state, _obs = env.reset(jax.random.PRNGKey(0))
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-vmap", action="store_true",
        help="Skip the vmap measurement (debug; vmap can dwarf single-jit).",
    )
    ap.add_argument(
        "--skip-scan", action="store_true",
        help="Skip the scan measurement (debug; scan can dwarf single-jit).",
    )
    ap.add_argument(
        "--skip-vmap-scan", action="store_true",
        help="Skip the vmap(scan(...)) PPO-shape measurement.",
    )
    ap.add_argument(
        "--scan-steps", type=int, default=16,
        help="Length passed to lax.scan (default 16).",
    )
    ap.add_argument(
        "--vmap-batch", type=int, default=4,
        help="Batch size for the vmap measurement (default 4).",
    )
    ap.add_argument(
        "--out", type=str, default="bench/results/compile_profile.json",
        help="JSON output path (relative to repo root).",
    )
    args = ap.parse_args()

    print("=" * 72)
    print("env._step_jit compile-time profile")
    print("=" * 72)
    print(f"JAX     : {jax.__version__}  backend={jax.default_backend()}")

    state = _build_state()
    action = jnp.int32(0)
    key = jax.random.PRNGKey(1)

    results: dict = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
    }

    print("\n[1/3] jit(_step_impl) cold compile ...", flush=True)
    t_single = _compile_single(state, action, key)
    results["jit_single_s"] = t_single
    print(f"      {t_single:7.2f}s")

    if args.skip_vmap:
        print("\n[2/3] vmap: SKIPPED")
    else:
        print(f"\n[2/3] jit(vmap(_step_impl)) batch={args.vmap_batch} cold compile ...", flush=True)
        t_vmap = _compile_vmap(state, action, key, batch=args.vmap_batch)
        results["jit_vmap_s"] = t_vmap
        results["vmap_batch"] = args.vmap_batch
        print(f"      {t_vmap:7.2f}s")

    if args.skip_scan:
        print("\n[3/4] scan: SKIPPED")
    else:
        print(f"\n[3/4] jit(scan(_step_impl, length={args.scan_steps})) cold compile ...", flush=True)
        t_scan = _compile_scan(state, action, key, n_steps=args.scan_steps)
        results["jit_scan_s"] = t_scan
        results["scan_length"] = args.scan_steps
        print(f"      {t_scan:7.2f}s")

    if args.skip_vmap_scan:
        print("\n[4/4] vmap(scan): SKIPPED")
    else:
        print(
            f"\n[4/4] jit(vmap(scan(_step_impl, length={args.scan_steps}, batch={args.vmap_batch}))) "
            f"cold compile ...", flush=True,
        )
        t_vs = _compile_vmap_scan(
            state, key, n_steps=args.scan_steps, batch=args.vmap_batch,
        )
        results["jit_vmap_scan_s"] = t_vs
        print(f"      {t_vs:7.2f}s")

    out_path = _PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
