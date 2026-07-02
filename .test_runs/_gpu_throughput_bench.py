"""Phase-0 GPU throughput profiler: measure Minihax env-steps/s vs batch size
and the memory ceiling, for reduced and full configs.

Run on a GPU node (Harvard). Persistent compile cache is on by default
(~/.cache/nethax_jax) so re-runs at the same B skip the ~14min cold compile.

Usage (GPU):
  NETHAX_VEC_MONSTERS=1 PYTHONPATH=. python .test_runs/_gpu_throughput_bench.py
  # reduced config:
  NETHAX_SINGLE_LEVEL=1 NETHAX_MAX_MONSTERS=16 NETHAX_VEC_MONSTERS=1 \
    PYTHONPATH=. python .test_runs/_gpu_throughput_bench.py

Env:
  BATCHES   comma list of batch sizes (default "256,1024,4096,16384")
  NSTEP     timed steps per batch (default 50)
"""
import os, sys, time, gc
import jax, jax.numpy as jnp, numpy as np
import gym, gymnasium, gymnasium.spaces.dict as _g
_g.Space = (gymnasium.spaces.Space, gym.spaces.Space)
from Nethax.minihax.minihax_env import MinihaxEnv

ENV = os.environ.get("BENCH_ENV", "MiniHack-Room-Monster-5x5-v0")
BATCHES = [int(x) for x in os.environ.get("BATCHES", "256,1024,4096,16384").split(",")]
NSTEP = int(os.environ.get("NSTEP", "50"))

dev = jax.devices()[0]
print(f"device={dev.platform}:{dev.device_kind}  env={ENV}  "
      f"cfg(SINGLE_LEVEL={os.environ.get('NETHAX_SINGLE_LEVEL')},"
      f"MAX_MON={os.environ.get('NETHAX_MAX_MONSTERS')})", flush=True)

env = MinihaxEnv(ENV)
s0, info0 = env.reset(jax.random.key(0))
state_mb = sum(int(l.nbytes) for l in jax.tree_util.tree_leaves(s0)) / 1e6
print(f"per-env state: {state_mb:.3f} MB", flush=True)

# Bench the movement-only FAST step (training path): dispatches only the move
# handler (no 46-way switch) -> ~47x faster compile, ~2.6x cheaper exec, and
# byte-identical to env.step for movement.  Set BENCH_FULL=1 to bench the full
# 46-handler step instead.
# Configurable restricted action set (agent emits index 0..N-1). Default = the
# 8 MiniHack Room compass moves. ACTION_ORDS overrides (comma-sep ASCII ords).
_ACTION_ORDS = [int(x) for x in os.environ.get(
    "ACTION_ORDS", "107,108,106,104,117,110,98,121").split(",")]
print(f"action set: {len(_ACTION_ORDS)} ords -> restricted dispatch", flush=True)
if os.environ.get("BENCH_FULL") == "1":
    def step_one(st, a, k, fm, sc, par):
        ns, r, d, _ = env.step(st, a, k, fired_mask=fm, step_count=sc, pits_at_reset=par)
        return ns, r, d
else:
    from Nethax.nethax.env import _make_restricted_step_impl
    _rstep = _make_restricted_step_impl(tuple(_ACTION_ORDS), True)  # byte-identical fast path
    def step_one(st, a, k, fm, sc, par):
        ns, obs, r, d = _rstep(st, a, k)               # a = action index 0..N-1
        return ns, r, d
# BENCH_DONATE=1: donate the batched state arg so XLA aliases input->output
# buffers (byte-neutral: pure buffer reuse) — frees a full ~state-sized copy,
# a byte-neutral lever for raising the batch ceiling.
if os.environ.get("BENCH_DONATE") == "1":
    vstep = jax.jit(jax.vmap(step_one), donate_argnums=(0,))
else:
    vstep = jax.jit(jax.vmap(step_one))

for B in BATCHES:
    try:
        # Throughput needs B valid envs to STEP, not B distinct layouts — tile
        # one host-built reset state to [B,...] via vmap (handles PRNGKey leaves),
        # avoiding reset_batch's host-side per-env loop (infeasible at B>=10k).
        t = time.time()
        bs = jax.vmap(lambda _: s0)(jnp.arange(B))
        bi = jax.vmap(lambda _: info0)(jnp.arange(B))
        jax.block_until_ready(bs.player_pos); t_reset = time.time() - t
        # full step wants an ASCII ord (107='k'=N); fast step wants dir_idx (0=N)
        acts = jnp.full((B,), 107 if os.environ.get("BENCH_FULL") == "1" else 0, dtype=jnp.int32)
        sc = jnp.zeros((B,), dtype=jnp.int32)
        # warm / compile (cached across runs)
        t = time.time()
        st, r, d = vstep(bs, acts, jax.random.split(jax.random.key(2), B), bi["fired_mask"], sc, bi["pits_at_reset"])
        jax.block_until_ready(st.player_pos); t_warm = time.time() - t
        # timed steady-state
        t = time.time()
        for i in range(NSTEP):
            kk = jax.random.split(jax.random.key(100 + i), B)
            st, r, d = vstep(st, acts, kk, bi["fired_mask"], sc, bi["pits_at_reset"])
        jax.block_until_ready(st.player_pos); dt = time.time() - t
        eps = NSTEP * B / dt
        print(f"B={B:7d}  reset={t_reset:6.1f}s  warm={t_warm:6.1f}s  "
              f"steady={eps:12.0f} env-steps/s  ({dt:.2f}s/{NSTEP} batch-steps)", flush=True)
        del bs, bi, st, r, d; gc.collect()
    except Exception as e:
        print(f"B={B:7d}  FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
        break
print("DONE", flush=True)
