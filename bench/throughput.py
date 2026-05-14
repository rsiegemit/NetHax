"""Nethax throughput benchmark.

Measures steps-per-second across four scenarios:
  1. Single-env (CPU)
  2. Batched vmap (batch sizes 8, 64, 512, 4096) — CPU + GPU
  3. Long rollout via lax.scan (1000 steps, batch=64) — CPU + GPU
  4. Reset throughput — single + batched (batch=64)

Usage:
    # CPU
    JAX_PLATFORMS=cpu .venv/bin/python bench/throughput.py

    # GPU (skipped automatically if unavailable)
    JAX_PLATFORMS=gpu .venv/bin/python bench/throughput.py

    # Smoke mode (fast, used by tests)
    .venv/bin/python bench/throughput.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when invoked as `python bench/throughput.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from Nethax.nethax.env import NethaxEnv

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Nethax throughput benchmark")
parser.add_argument(
    "--smoke",
    action="store_true",
    help="Smoke mode: 1 warmup, 5 measured runs — fast sanity check.",
)
parser.add_argument(
    "--warmup",
    type=int,
    default=None,
    help="Override warmup iterations (default 5, smoke=1).",
)
parser.add_argument(
    "--runs",
    type=int,
    default=None,
    help="Override measured iterations (default 30, smoke=5).",
)
args = parser.parse_args()

SMOKE = args.smoke
N_WARMUP = args.warmup if args.warmup is not None else (1 if SMOKE else 5)
N_RUNS = args.runs if args.runs is not None else (5 if SMOKE else 30)

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

CPU_DEVICES = jax.devices("cpu")
try:
    GPU_DEVICES = jax.devices("gpu")
except RuntimeError:
    GPU_DEVICES = []

HAVE_GPU = len(GPU_DEVICES) > 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns() -> int:
    return time.perf_counter_ns()


def _block(x: Any) -> None:
    """Block until JAX computation is complete."""
    jax.block_until_ready(x)


def _stats(times_ns: list[int], n_steps: int) -> dict:
    """Compute mean/median/p95 steps-per-second from a list of ns timings."""
    sps = [n_steps / (t * 1e-9) for t in times_ns]
    return {
        "mean_sps": statistics.mean(sps),
        "median_sps": statistics.median(sps),
        "p95_sps": float(np.percentile(sps, 95)),
        "n_steps": n_steps,
        "n_runs": len(times_ns),
    }


def _fmt(label: str, d: dict) -> str:
    return (
        f"  {label:<45}  "
        f"mean={d['mean_sps']:>12,.0f}  "
        f"median={d['median_sps']:>12,.0f}  "
        f"p95={d['p95_sps']:>12,.0f}  sps"
    )


# ---------------------------------------------------------------------------
# Environment + JIT-compiled helpers
# ---------------------------------------------------------------------------

env = NethaxEnv()

# vmap(env.step) — maps over (states, actions, keys)
_vmap_step = jax.jit(jax.vmap(env._step_jit))

# JIT-compiled reset is not straightforward since env.reset() contains
# Python-level construction (EnvState.default) that isn't pure JAX.
# We time the Python-side reset as-is (it still calls JAX ops internally).
# For "batched reset" we measure N sequential resets (the realistic use case).


def _lax_scan_rollout(init_state, init_key, n_steps: int):
    """Run n_steps via lax.scan on a single state (or batched via vmap)."""

    def _scan_body(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        action = jnp.int32(0)
        new_state, _obs, _rew, _done = env._step_jit(state, action, subkey)
        return (new_state, key), None

    (final_state, _), _ = jax.lax.scan(_scan_body, (init_state, init_key), None, length=n_steps)
    return final_state


_jit_scan_rollout = jax.jit(_lax_scan_rollout, static_argnums=(2,))


def _make_vmap_scan(n_steps: int):
    """Return a jit+vmap scan rollout for a fixed n_steps (static for XLA)."""
    return jax.jit(jax.vmap(lambda s, k: _lax_scan_rollout(s, k, n_steps)))


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def _time_fn(fn, *args, n_warmup: int = N_WARMUP, n_runs: int = N_RUNS):
    """Warmup fn n_warmup times (not timed), then measure n_runs iterations.

    Returns list of elapsed nanoseconds per call.
    """
    for _ in range(n_warmup):
        _block(fn(*args))

    timings = []
    for _ in range(n_runs):
        t0 = _ns()
        _block(fn(*args))
        timings.append(_ns() - t0)
    return timings


def bench_single_env_cpu(device) -> dict:
    """Single env, CPU, action=0 step."""
    with jax.default_device(device):
        key = jax.random.PRNGKey(42)
        state, _ = env.reset(key)
        action = jnp.int32(0)
        step_key = jax.random.PRNGKey(1)

        def _step():
            return env._step_jit(state, action, step_key)

        timings = _time_fn(_step)
    return _stats(timings, n_steps=1)


def bench_batched(batch_size: int, device) -> dict:
    """vmap over batch_size envs, action=0 for all."""
    with jax.default_device(device):
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        # Build batched initial states
        states_list = [env.reset(k)[0] for k in keys]
        # Stack into a single batched pytree
        batched_state = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *states_list
        )
        actions = jnp.zeros(batch_size, dtype=jnp.int32)
        step_keys = jax.random.split(jax.random.PRNGKey(99), batch_size)

        def _step():
            return _vmap_step(batched_state, actions, step_keys)

        timings = _time_fn(_step)
    return _stats(timings, n_steps=batch_size)


def bench_scan_rollout(batch_size: int, n_steps: int, device) -> dict:
    """lax.scan rollout of n_steps, batched via vmap."""
    with jax.default_device(device):
        keys = jax.random.split(jax.random.PRNGKey(7), batch_size)
        states_list = [env.reset(k)[0] for k in keys]
        batched_state = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *states_list
        )
        vmap_scan = _make_vmap_scan(n_steps)

        def _rollout():
            return vmap_scan(batched_state, keys)

        timings = _time_fn(_rollout)
    return _stats(timings, n_steps=batch_size * n_steps)


def bench_reset_single(device) -> dict:
    """Single env reset throughput (Python-side, JAX ops inside)."""
    with jax.default_device(device):
        key = jax.random.PRNGKey(5)

        def _reset():
            state, obs = env.reset(key)
            _block(obs)
            return state

        timings = _time_fn(_reset)
    return _stats(timings, n_steps=1)


def bench_reset_batched(batch_size: int, device) -> dict:
    """Sequential batched reset: N independent env.reset calls, block at end."""
    with jax.default_device(device):
        keys = jax.random.split(jax.random.PRNGKey(6), batch_size)

        def _reset():
            states = [env.reset(k)[0] for k in keys]
            # block on last state's obs to ensure all JAX work is complete
            _block(states[-1])
            return states

        timings = _time_fn(_reset)
    return _stats(timings, n_steps=batch_size)


# ---------------------------------------------------------------------------
# JIT compile-time measurement
# ---------------------------------------------------------------------------


def _measure_compile_time(fn, *args) -> float:
    """Return compile time in seconds for the first call to fn(*args)."""
    # Force a fresh trace by calling with concrete args
    t0 = _ns()
    _block(fn(*args))
    return (_ns() - t0) * 1e-9


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


def _system_info() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "cpu_devices": len(CPU_DEVICES),
        "gpu_devices": len(GPU_DEVICES),
    }
    # Try to get RAM
    try:
        import subprocess
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            info["ram_gb"] = round(int(out) / 1024**3, 1)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / 1024**2, 1)
                        break
    except Exception:
        info["ram_gb"] = "unknown"

    # GPU info
    if HAVE_GPU:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                text=True,
            ).strip()
            info["gpu_model"] = out
        except Exception:
            info["gpu_model"] = "unknown"
    else:
        info["gpu_model"] = None

    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    sys_info = _system_info()
    results: dict = {"system": sys_info, "cpu": {}, "gpu": {}}

    print("=" * 85)
    print("Nethax Throughput Benchmark")
    print("=" * 85)
    print(f"  Platform : {sys_info['platform']}")
    print(f"  CPU      : {sys_info['cpu']}")
    print(f"  RAM      : {sys_info.get('ram_gb', 'unknown')} GB")
    print(f"  GPU      : {sys_info['gpu_model'] or 'none (CPU-only)'}")
    print(f"  JAX      : {sys_info['jax_version']}  backend={sys_info['jax_backend']}")
    print(f"  Smoke    : {SMOKE}  warmup={N_WARMUP}  runs={N_RUNS}")
    print()

    cpu_dev = CPU_DEVICES[0]

    # ---- Measure JIT compile time (once, on CPU) --------------------------
    # Import the raw un-jitted implementation and wrap it fresh so we get a
    # genuine first-trace cost (env._step_jit is already compiled by now).
    # Skipped in smoke mode — the fresh trace adds ~30-60s to the test.
    if not SMOKE:
        print("Measuring JIT compile time (first call) …")
        from Nethax.nethax.env import _step_impl as _raw_step_impl
        _compile_state, _ = env.reset(jax.random.PRNGKey(0))
        _fresh_jit = jax.jit(_raw_step_impl)
        compile_time_s = _measure_compile_time(
            _fresh_jit,
            _compile_state,
            jnp.int32(0),
            jax.random.PRNGKey(1),
        )
        results["jit_compile_time_s"] = compile_time_s
        print(f"  JIT compile time: {compile_time_s:.2f}s\n")
    else:
        results["jit_compile_time_s"] = None

    # ---- CPU benchmarks ---------------------------------------------------
    print("─" * 85)
    print("CPU Benchmarks")
    print("─" * 85)

    # 1. Single env
    d = bench_single_env_cpu(cpu_dev)
    results["cpu"]["single_env"] = d
    print(_fmt("single-env", d))

    if SMOKE:
        # Smoke mode is a true sanity check — single-env step only.  Each
        # additional batch / scan / reset benchmark adds a fresh JIT trace
        # (~30-60s each on CPU), which is too slow for a CI smoke gate.
        scan_batch = 0
        scan_steps = 0
    else:
        # 2. Batched
        for bs in [8, 64, 512, 4096]:
            d = bench_batched(bs, cpu_dev)
            results["cpu"][f"batch_{bs}"] = d
            print(_fmt(f"vmap batch={bs}", d))

        # 3. Scan rollout (batch=64, 1000 steps)
        scan_batch = 64
        scan_steps = 1000
        d = bench_scan_rollout(scan_batch, scan_steps, cpu_dev)
        results["cpu"][f"scan_batch{scan_batch}_steps{scan_steps}"] = d
        print(_fmt(f"lax.scan batch={scan_batch} steps={scan_steps}", d))

        # 4. Reset
        d = bench_reset_single(cpu_dev)
        results["cpu"]["reset_single"] = d
        print(_fmt("reset single", d))

        d = bench_reset_batched(scan_batch, cpu_dev)
        results["cpu"][f"reset_batch_{scan_batch}"] = d
        print(_fmt(f"reset batch={scan_batch}", d))

    # ---- GPU benchmarks (if available) ------------------------------------
    if HAVE_GPU and not SMOKE:
        gpu_dev = GPU_DEVICES[0]
        print()
        print("─" * 85)
        print("GPU Benchmarks")
        print("─" * 85)

        d = bench_single_env_cpu(gpu_dev)
        results["gpu"]["single_env"] = d
        print(_fmt("single-env", d))

        for bs in [8, 64, 512, 4096]:
            d = bench_batched(bs, gpu_dev)
            results["gpu"][f"batch_{bs}"] = d
            print(_fmt(f"vmap batch={bs}", d))

        d = bench_scan_rollout(scan_batch, scan_steps, gpu_dev)
        results["gpu"][f"scan_batch{scan_batch}_steps{scan_steps}"] = d
        print(_fmt(f"lax.scan batch={scan_batch} steps={scan_steps}", d))

        d = bench_reset_single(gpu_dev)
        results["gpu"]["reset_single"] = d
        print(_fmt("reset single", d))

        d = bench_reset_batched(scan_batch, gpu_dev)
        results["gpu"][f"reset_batch_{scan_batch}"] = d
        print(_fmt(f"reset batch={scan_batch}", d))
    elif SMOKE:
        results["gpu"]["status"] = "smoke mode: skipped"
    else:
        print()
        print("GPU: not available — skipping GPU benchmarks.")
        results["gpu"]["status"] = "no GPU available"

    # ---- Write JSON -------------------------------------------------------
    out_path = Path(__file__).parent / "results" / "throughput.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Results written to {out_path}")
    print("=" * 85)

    return results


if __name__ == "__main__":
    main()
