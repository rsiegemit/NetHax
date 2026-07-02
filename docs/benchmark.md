# Nethax Throughput Benchmark

## How to Reproduce

```bash
# Full benchmark (CPU)
JAX_PLATFORMS=cpu .venv/bin/python bench/throughput.py

# Full benchmark (GPU, auto-detected)
JAX_PLATFORMS=gpu .venv/bin/python bench/throughput.py

# Smoke mode (fast, ~5 warmup+measured runs — same as CI test)
.venv/bin/python bench/throughput.py --smoke

# Results are written to bench/results/throughput.json
```

The first invocation per process pays a JIT compile cost (~30–60 s for the
full dispatch / monster-AI / status pipeline). All reported numbers exclude
this compile step.

---

## System Information (reference run, smoke mode)

| Field    | Value                                |
|----------|--------------------------------------|
| Platform | macOS 14.6.1 arm64 (Apple Silicon)  |
| CPU      | Apple Silicon (arm64)                |
| RAM      | 48 GB                                |
| GPU      | none on reference machine            |
| JAX      | 0.10.0 (CPU backend)                 |

> Re-run `bench/throughput.py` to refresh these — the script overwrites
> `bench/results/throughput.json` with current-host numbers.

---

## Measured Throughput

All figures are **steps per second (sps)**, warmup-amortised.

### CPU — Smoke baseline (single-env, 1 warmup + 5 runs)

| Scenario                            | Mean sps | Median sps | p95 sps |
|-------------------------------------|----------|------------|---------|
| single-env (no vmap)                | 140      | 146        | 152     |

### CPU — Full benchmark (run `python bench/throughput.py` to populate)

| Scenario                            | Mean sps | Median sps | p95 sps |
|-------------------------------------|----------|------------|---------|
| single-env (no vmap)                | —        | —          | —       |
| vmap batch=8                        | —        | —          | —       |
| vmap batch=64                       | —        | —          | —       |
| vmap batch=512                      | —        | —          | —       |
| vmap batch=4096                     | —        | —          | —       |
| lax.scan rollout (batch=64, 1k stp) | —        | —          | —       |
| reset single                        | —        | —          | —       |
| reset batch=64                      | —        | —          | —       |

The full benchmark adds vmap traces at batch sizes 8/64/512/4096 plus a
1000-step lax.scan rollout — each is a fresh JIT compile (~30–90 s).
Expected total wall-time: ~20–40 min on CPU; significantly faster on GPU
where compiles parallelise well.

### GPU

*(skipped — no GPU on reference machine)*

> If a GPU is available, re-run with `JAX_PLATFORMS=gpu` and the table above
> will be populated in `bench/results/throughput.json`.

---

## JIT Compile Time

| Step                          | Time (s) |
|-------------------------------|----------|
| `env._step_jit` first call    | ~30–60 s |

The compile cost is paid once per Python process. Subsequent calls are
O(1 ms) or less. The benchmark measures only post-compile throughput.

---

## Comparison: NLE vs Nethax

| Environment | Single-env (CPU) | Batch-512 (CPU) | Batch-4096 (CPU) |
|-------------|-----------------|-----------------|-----------------|
| NLE (C ext) | ~10 000–20 000  | N/A (fork-per-env) | N/A           |
| Nethax (JAX)| *(measured)*    | *(measured)*    | *(measured)*   |

**NLE** achieves ~10 000–20 000 sps per environment on a modern x86 CPU but
is fundamentally **not batchable** — each environment is a separate forked
process with a C extension. There is no efficient vmap or lax.scan path.

**Nethax** pays a ~30–60 s JIT compile cost but then amortises it across
arbitrarily many rollouts. The key advantage is:

- `jax.vmap` over N environments is a single fused kernel, not N forked
  processes. At batch=512 or 4096 the aggregate sps typically exceeds NLE
  by 10–100×.
- `jax.lax.scan` eliminates Python-loop overhead for long rollouts; the
  entire 1 000-step trajectory is a single compiled XLA computation.
- On GPU, vmap batching can push aggregate throughput to millions of sps
  (environment-dependent — not yet measured on this machine).

The single-env no-vmap figure is expected to be **lower** than NLE (~1 000–
5 000 sps) because the JAX interpreter overhead dominates at batch=1. This
is the expected trade-off: JAX environments are not optimised for sequential
single-env interaction.

---

## Full-fidelity GPU measurements (2026-07, A100-80GB)

Measured on `MiniHack-Room-Monster-5x5-v0` via `MinihaxEnv` at **full fidelity**
(`MAX_MONSTERS=400`, all dungeon levels, no `SINGLE_LEVEL`), sparse `ground_items`
(8.04 MB/env vs 124.8 MB dense), movement-only restricted step. These are the
honest, load-bearing numbers — they supersede the optimistic "10–100× over NLE"
estimate above, which only holds for reduced/generic configs.

### Training-throughput flags

The RL/training path has two opt-in speedups, **both gated off the vendor /
byte-parity path** (they only fire when `use_vendor_rng()` is False), so the
48/48 multi-seed byte-parity gate is unaffected by construction:

| Flag | Effect |
|------|--------|
| `NETHAX_VEC_MONSTERS=1` (default on) | simultaneous-move vectorized monster turn (vmap over slots) instead of the serial 400-iteration scan |
| `NETHAX_FAST_POST=1` | trims the ~30 per-turn vendor status/timer ticks to the RL essentials (turn counters + status/HP/PW + polymorph) |

The vectorized monster turn additionally **hoists the player-rooted BFS distance
field** out of the per-monster vmap (computed once per env, shared) — a large
compute + memory win. All three are byte-parity-safe (serial vendor path
untouched; verified 48/48 12/12 across seeds 0/1/2/5).

### Measured curve (A100-80GB, `NETHAX_VEC_MONSTERS=1 NETHAX_FAST_POST=1`)

| Optimization stage | B=512 | B=1024 |
|--------------------|-------|--------|
| baseline (vec monsters only) | 929 sps | — |
| + fast post-monster (2.02×)  | 1878 sps | — |
| + BFS hoist (1.43×)          | 2678 sps | **3436 sps** |

**Cumulative: 929 → 3436 env-steps/s @ B=1024 = 3.7×.** Batch ceiling is **B=1024**
on 80 GB (B=2048 OOMs); the wall is a fused multi-broadcast transient in the
monster vmap, not the sparse state size.

### Honest comparison to NLE at full fidelity

A single NLE core does **~10,778 steps/s**. Full-fidelity Nethax on a *whole*
A100 tops out at **~3436 steps/s @ B=1024 — roughly 3× below one NLE core**, and
batching does **not** close this: throughput plateaus (574 → 929 → 1043 → 3436
after the optimizations) because the full-fidelity step graph is **compute-bound**
(monster AI + the per-move FOV/vision raycast), so the GPU saturates rather than
scaling linearly with B. The `jax.vmap` "beats NLE by 10–100×" advantage is real
for *reduced* configs (single level, few monsters) but does **not** hold for full
fidelity, where per-step cost dominates. Beating NLE at full fidelity remains open
and would require cutting the FOV raycast and monster-body cost further.

---

## Methodology

- Timing: `time.perf_counter_ns` (sub-millisecond resolution).
- Warmup: 5 calls (1 in smoke mode) before any timing begins.
- Measurement: 30 calls (5 in smoke mode); mean/median/p95 reported.
- `jax.block_until_ready` is called after each step to ensure async dispatch
  is fully complete before the timer stops.
- JIT compile time is measured separately by calling a fresh `jax.jit`
  wrapper on the step function once with concrete arguments.
- Batched states are constructed by calling `env.reset` per environment and
  stacking the resulting pytrees with `jnp.stack` — this is the correct way
  to build a batched initial state compatible with `jax.vmap`.
