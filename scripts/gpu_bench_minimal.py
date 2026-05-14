"""Minimal GPU throughput bench — just batch=512 vmap. One compile, one number."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jax, jax.numpy as jnp
from Nethax.nethax.env import NethaxEnv

env = NethaxEnv()
N = 512
print(f"  device: {jax.default_backend()} {jax.devices()}")
print(f"  batch:  {N}")

# Build batched initial state
keys = jax.random.split(jax.random.PRNGKey(0), N)
states_list = [env.reset(k)[0] for k in keys]
batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states_list)
actions = jnp.zeros(N, dtype=jnp.int32)
step_keys = jax.random.split(jax.random.PRNGKey(99), N)

vstep = jax.jit(jax.vmap(env._step_jit))
print("Compiling…", flush=True)
t0 = time.perf_counter()
out = vstep(batched, actions, step_keys)
jax.block_until_ready(out[0].turn if hasattr(out[0], "turn") else out)
print(f"  compile + first step: {time.perf_counter()-t0:.1f}s", flush=True)

# Warm-up
for _ in range(3):
    out = vstep(batched, actions, step_keys)
    jax.block_until_ready(out)

# Measure
ts = []
for _ in range(20):
    t0 = time.perf_counter_ns()
    out = vstep(batched, actions, step_keys)
    jax.block_until_ready(out)
    ts.append(time.perf_counter_ns() - t0)

import statistics
sps = [N / (t * 1e-9) for t in ts]
print(f"  mean  : {statistics.mean(sps):>12,.0f} steps/s")
print(f"  median: {statistics.median(sps):>12,.0f} steps/s")
print(f"  p95   : {sorted(sps)[int(len(sps)*0.95)]:>12,.0f} steps/s")
