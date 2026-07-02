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

> **See ["Honest comparison to the C baselines (MEASURED)"](#honest-comparison-to-the-c-baselines-measured-2026-07-02)
> below for the real, measured numbers.** The optimized-C NLE/MiniHack engines win on
> raw throughput at every scale; Nethax/Minihax do **not** beat them on steps/s. The
> paragraphs below described an *expected* vmap advantage that measurement did not bear
> out for these envs — kept only for context.

**NLE** achieves ~16 000–46 000 sps per environment on a CPU core (measured:
`NetHack-v0` 16,261; `NetHackChallenge-v0` 45,970) and, though each env is a separate
forked process, **scales ~linearly across CPU cores** — a 112-core node reaches
millions of sps.

**Nethax/Minihax** pay a JIT compile cost, then run N rollouts as one fused
`jax.vmap` kernel — but throughput **plateaus** (the step graph is compute-bound), so a
single GPU roughly matches *one* CPU core, not a whole node. The real advantages are
byte-exact parity, differentiability, and running the env natively on GPU/TPU — not
sample throughput.

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

**Cumulative: 929 → 3436 env-steps/s @ B=1024 = 3.7×.**

### Byte-neutral batch ceiling (donation + monster-vmap chunking)

The default batch ceiling is **B=1024** on 80 GB — the wall is a fused multi-broadcast
transient in the monster vmap (not the sparse state size). Two **byte-neutral** levers
(exact same outputs) raise it to **B=2048**:

- **Input→output donation** — `jax.jit(step, donate_argnums=(state,))` so XLA aliases
  the input state buffer into the output. Free for a scan-based training rollout (the
  caller must not reuse the donated state). *Alone* only saves ~2.7 GiB — the binding
  alloc is the monster transient, not a redundant state copy.
- **`NETHAX_VEC_CHUNK=64`** — processes the 400-monster vmap in chunks of 64
  (`jax.lax.map`), bounding the per-monster activation to `[64, …]` instead of
  `[400, …]`. Output is identical; the 51.66 GiB transient drops to ~14 GiB.

Together: **B=2048 fits at 3844 env-steps/s** (vs 3436 @ B=1024, **+12%** — the chunk
serialization is more than repaid by the larger batch). B=4096 still OOMs (needs ~33 GiB
more headroom → a state shrink such as bit-packing the bool feature planes). Recommended
full-fidelity training recipe on 80 GB: `NETHAX_VEC_CHUNK=64` + donated state at B=2048.

### Honest comparison to the C baselines (MEASURED 2026-07-02)

All baselines are **measured**, not cited — the optimized-C NLE/MiniHack envs win
decisively on raw throughput, at every scale. Two matched comparisons:

**① NetHack vs Nethax (full game).** Baseline: NLE running full NetHack, single-env
on one CPU core (Harvard, 112-core node).

| | throughput | at node scale |
|---|-----------|---------------|
| NetHack — `NetHackChallenge-v0` (NLE C) | **45,970 sps / core** | ~5.1M sps (×112 cores, fork-per-env) |
| NetHack — `NetHack-v0` (NLE C) | **16,261 sps / core** | ~1.8M sps (×112) |
| **Nethax (JAX, full fidelity, 1×A100)** | **3,844 sps @ B=2048** | one GPU |

Nethax on a *whole* A100 is **~4–12× below one NLE core** and ~500–1300× below a full
node. Batching does not close it: throughput plateaus (929 → 3436 → 3844 across the
optimizations) because the full-fidelity step is **compute-bound** (monster AI + the
per-move FOV/vision raycast), so the GPU saturates rather than scaling with B.

**② MiniHack vs Minihax (small RL scope).** Baseline: vendor MiniHack (NLE C) single-env.

| | throughput | at node scale |
|---|-----------|---------------|
| MiniHack — `Room-Monster-5x5` (NLE C, 1 core) | **~14,967 sps / core** | ~240k (16-core) … ~1.7M (112-core) sps, fork-per-env |
| **Minihax (JAX, matched reduced scope, 1×A100)** | **15,148 sps @ B=16,384** | one GPU |

At matched scope, Minihax on a full GPU ≈ **one** MiniHack CPU core, and loses to a
multi-core node (MiniHack forks scale ~linearly with cores; Minihax plateaus,
compute-bound). (Corrects an earlier draft that cited NLE ≈ 10,778 sps and claimed a
10–100× vmap advantage — both wrong.)

**Takeaway:** the value of the JAX reimplementation is **not** steps/s — the C engines
are faster. It is (a) **byte-exact vendor parity** (48/48 multi-seed), (b)
**differentiability** — gradients flow through `env.step`, impossible in the C envs,
(c) **GPU/TPU-native** single-kernel batching for research that must run the env on the
accelerator (no per-env host processes). For pure sample throughput on CPU, NLE/MiniHack
remain the better tool.

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
