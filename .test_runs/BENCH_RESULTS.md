# Nethax / MiniHax timing benchmarks

HEAD `1a057cb` (48/48 multi-seed Room byte-parity).  ParityMode.NLE.

Two execution paths are benchmarked because they serve different goals:

| path | what it's for | compile? | per-step |
|------|---------------|----------|----------|
| **eager** (`jax_disable_jit`) | byte-exact validation (the parity validator) | none | op-by-op, no fusion → very slow |
| **JIT** (`step_batched`, `static_action=0`) | RL training throughput | one heavy trace+compile | fused, fast warm steps + vmap scaling |

The static-action path is required: passing a *traced* action triggers the
46-branch dynamic dispatch whose trace/compile blows up.  The validator and
training both route a Python-int / static action through
`_dispatch_jit_validator`.

## Mac CPU (Apple Silicon, JAX_PLATFORMS=cpu)

### Init / reset (one-time)
| phase | Room-5x5 | Room-15x15 | Ult-15x15 | Nethax-full |
|-------|---------:|-----------:|----------:|------------:|
| import jax        | 0.33s | — | — | 0.33s |
| import env        | ~1–12s (cold fs) | | | 1.1s |
| construct env     | ~0.00s | | | 0.00s |
| reset **compile** | 2.7s | 2.8s | 5.6s | 19.5s |
| reset **warm**    | 1.3s | — | — | 3.5s |

### Eager per-step (validation path — no compile)
| env | per-step | single-env steps/s |
|-----|---------:|-------------------:|
| Room-5x5        | 7.9 s | 0.13 |
| Room-15x15      | 8.4 s | 0.12 |
| Ult-15x15       | 10.9 s | 0.09 |

→ Eager is op-by-op (thousands of unfused primitives per step).  Fine for
byte-parity validation, unusable for training.  This is *why* the JIT path
exists and why its one-time trace is large.

### JIT step (training path) — compile + warm + scaling
The first call traces+lowers the **entire fused step graph** into one HLO
module — and recompiles **once per distinct batch shape B**.  This trace+lower
is the documented bottleneck (task #16).  Measured with the persistent compile
cache (`JAX_COMPILATION_CACHE_DIR`) so each shape pays this **once ever**.

**No flag combination wins** — the trade-off is fundamental (B=1, Mac CPU,
measured TRACE/LOWER/COMPILE/EXEC separately):

| config | compile (B=1) | execution (1 step) | verdict |
|--------|--------------:|-------------------:|---------|
| XLA default (all opts) | **>1h08m, never completed** | (fast, presumed) | **compile wall** |
| `jax_disable_most_optimizations` | 34 s | **>4 min/step** | **exec wall** |
| `--xla_disable_hlo_passes` (keeps fusion) | 60 s | **217 s/step** | **exec wall** |

Reducing optimization to fix compile destroys execution, and vice-versa.
Constant across all three: TRACE ~35 s, LOWER ~5 s, HLO 12.9 MB / 113 K lines.

### OVERNIGHT PROBE 2026-06-22 — the diagnosis FLIPS
Full-opt compile via the **explicit `.trace().lower().compile()` path**:
```
TRACE 36s · LOWER 5s · COMPILE 39.2s (FULL OPT) · EXEC#1 162.7s · WARM 158.9 s/step
```
- **Compile is NOT the wall.** Full-opt compile terminates in **39 s** via the
  explicit lowered-compile path. The earlier ">1h08m, never finishes" was a
  pathology of the **implicit pjit function-call path** (`step_fn(...)`), not the
  fundamental compile. Compiling via `lower().compile()` + invoking the
  executable side-steps it entirely.
- **The real wall is EXECUTION: ~159 s/step at B=1, even fully optimized** —
  ~14× SLOWER than eager (11 s/step). The batched flat Brax graph (291k ops,
  computes all 400 monster slots + all branches unconditionally) is
  **CPU-hostile**: single-thread CPU walks 291k ops with long sequential scan
  chains. Eager wins on CPU because Python control flow skips dead monster slots.
- **Implication:** this graph is GPU-oriented. The flat compute-all-branches
  design is exactly what parallelizes on A100/H100 — 291k independent-ish ops map
  to the GPU, where 159 s on CPU should collapse to sub-second. **CPU is the
  wrong target for the JIT path; GPU is.** And the Harvard ">56 min compile" may
  ALSO be the pjit-path artifact — worth re-testing there with the explicit path.

### ROOT CAUSE & FIX (the real story)
The slow "compile" is **neither tracing nor lowering** — both finish in seconds:
- TRACE 35 s → jaxpr with 291,068 equations (recursive; top-level only 32 — the
  Craftax `lax.scan` collapses the monster loop into one `While` body).
- LOWER 5 s → StableHLO 12.9 MB / 113,383 lines.

The wall is **XLA's optimization pass pipeline**.  Evidence from
`TF_CPP_VMODULE=hlo_pass_pipeline=2`:
- The lowered module has **60,000+ distinct computations** (one per `lax.cond`
  branch), and XLA runs its full pass list **per computation**
  (`#called` climbs past 62,000).
- The `simplification` / `algsimp` family **runs to fixpoint**, and a single
  `simplification` call on the giant scan-body computation costs ~13.9 s
  (`post_scatter_expansion_simplification` ~7.2 s).  Super-linear × huge
  computation × tens of thousands of invocations = the >1 h.

**Fix (no code rewrite needed):** disable the run-to-fixpoint simplification
family while keeping kernel fusion:
```
XLA_FLAGS="--xla_disable_hlo_passes=algsimp,cse,conditional-canonicalizer,simplify-conditional,scatter-canonicalizer,dynamic-dimension-simplifier"
```
`flatten-call-graph` is **mandatory** (copy-insertion requires it — disabling it
raises `FAILED_PRECONDITION: Call graph must be flattened before copy
insertion`), so it is NOT in the list.  All disabled passes are
semantics-preserving → byte-parity unaffected (they change kernel efficiency,
never results).  This is the practical unblock for JIT benchmarking/training;
the `lax.cond`→`jnp.where` rewrite (`project_h100_compile_redesign`) remains the
longer-term way to shrink the 60k-computation count itself.

## Flags — the "speedy mode" question
| flag | default | effect |
|------|---------|--------|
| `NETHAX_CRAFTAX_SCAN` | **1 (ON)** | the Craftax speedy path (scan + jnp.where vs Python unroll); active in all runs |
| `NETHAX_PHASED_ORCH`  | 1 (ON)  | phased orchestration |
| `NETHAX_BRAX_ALL`     | 0 (off) | experimental pytree consolidation — **NET NEGATIVE**: 29 min stuck on B=1 compile, killed |
| `NETHAX_JIT_SPLIT`    | 0 (off) | split-compile |

There is no faster flag left unset.  `BRAX_ALL=1` is slower, not faster.

## Harvard A100 (gpu_test, commit 75228ef) — the EXPLICIT-PATH retest (2026-06-22)

Re-ran with the explicit `trace→lower→compile→invoke` path
(`.test_runs/_bench_jit_explicit.py`, job 24036479, A100-SXM4-40GB, B=1,
Room-15x15, full opt, no XLA_FLAGS). This is the handoff action #1 test.

| stage | A100 (explicit path) | Mac CPU (explicit path) |
|-------|---------------------:|------------------------:|
| reset compile | 8.4 s | 2.8 s |
| reset warm | 3.51 s | — |
| TRACE (B=1) | 68.2 s | 36 s |
| LOWER (B=1) | 5.8 s | 5 s |
| **COMPILE (B=1, full opt)** | **89.6 s — TERMINATES** | 39.2 s |
| **EXEC#1 (B=1)** | **>56 min, NEVER finished → SLURM-killed** | 162.7 s |

### The diagnosis flips AGAIN — compile was never the GPU wall; EXEC is, and GPU is WORSE
1. **Compile terminates fast on GPU too (89.6 s).** The explicit-path hypothesis
   from the handoff is CONFIRMED: the cancelled job 23964219 ">56 min compile"
   was the implicit-pjit pathology. Explicit `lower().compile()` finishes in 90 s
   on the A100 (~2.3× Mac, consistent with the documented ~2× cluster trace/
   compile regression). Compile is solved.
2. **But the GPU does NOT rescue execution — it makes it WORSE.** B=1 *first*
   execution ran **>56 min without completing** (heartbeats every 300 s out to
   +3600 s, no EXEC#1 line) before SLURM killed the 1 h job. That is *worse* than
   CPU's 162 s/step. **The original GPU "56 min" was EXEC, not compile.** The
   "GPU-shaped, collapses to sub-second" hypothesis is **REFUTED** for first-exec.
3. `ptxas` IS bundled (`.venv/.../nvidia/cuda_nvcc/bin/ptxas`) and used by XLA at
   compile → slow exec is NOT driver-side PTX-JIT fallback.
4. **Not autotuning either.** Diagnostic job 24044687
   (`--xla_gpu_autotune_level=0`): compile still 88.6 s, and B=1 exec1 STILL ran
   **>32 min unfinished** (cancelled). Disabling autotuning changed nothing →
   the exec wall is **fundamental**, not a first-exec tuning artifact.

### ROOT CAUSE of the GPU exec wall — the `lax.scan` monster loop serializes
The Craftax-style `lax.scan` over the 400-slot monster order is what made
*compile* tractable (1 `While` body in HLO vs 400× unrolled → the original
compile blowup). But a `scan` is **inherently sequential**: on GPU it runs 400
iterations of a deep (~68k-eqn) body strictly in order, each emitting many tiny
dependent kernels (~µs launch latency, no cross-iteration parallelism). 400 ×
(huge serial body) × per-kernel launch overhead = the >30 min single-step wall.
CPU avoids it: the scan body lowers to a native loop with no kernel-launch
overhead, so CPU "only" pays 162 s.

**The fundamental tension:** unroll the monster loop → 400× HLO → compile blows
up. Keep it as a scan → compile fine, but GPU exec serializes. Neither single-env
path is usable.

**The way out is batch-parallelism, not single-env speed.** Real RL training runs
thousands of envs; the GPU parallelizes the scan body across the **B** (env)
dimension while each env's 400-iter scan stays serial. Per-step *latency* stays
high, but *throughput* (env-steps/s) should scale with B if the wall is
launch-bound. **Whether throughput actually recovers at large B is the open
question** — and the reason the B-sweep (`gpu-explicit-bench.sbatch`, B up to 256)
matters. It is expensive (each B = ~90 s compile + a multi-min/hung exec), so it
needs a dedicated multi-hour job, not the 1 h `gpu_test` cap.

> NOTE: the `--exec-watchdog` SIGALRM in `_bench_jit_explicit.py` does **not**
> interrupt a blocked native exec (the main thread is inside XLA C++; the Python
> signal handler can't run until it returns). A hung exec1 rides to the SLURM
> wall regardless. To bound it, cap the job `--time`, not the watchdog.

## Notes
- `reset_batched` (vmap) is blocked by host-side `bool()` ops in monster
  spawning (`spawning.py::_populate_oroom_single`) — resets are benchmarked
  single; the step (the perf-critical path) is what's vmap'd / batched.
- Harvard cluster had a documented XLA trace/compile regression (7–10× vs
  2026-06-10); GPU compile numbers should be read against that.
- Persistent compile cache lives at `.jax_compile_cache/`; it is target-specific
  (CPU HLO ≠ CUDA HLO) so a Mac cache cannot prime the A100.
