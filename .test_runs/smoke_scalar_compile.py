"""One-shot smoke test: time how long ``env.step`` cold-compiles after Fix 1.

Single seed, single step.  Goal: prove that with ``_monster_jit`` un-jit'd
and ``monsters_step_all``'s 3 scans replaced by Python for-loops, the
H100 cold compile actually finishes in single-digit minutes (vs the 16+
hours of the original code).

Usage:
    PYTHONPATH=. .venv/bin/python -u .test_runs/smoke_scalar_compile.py
"""
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

print(f"[{time.strftime('%H:%M:%S')}] imports", flush=True)
import jax
import jax.numpy as jnp
from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.parity_mode import set_parity_mode, ParityMode
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.races import Race

print(f"[{time.strftime('%H:%M:%S')}] jax devices: {jax.devices()}", flush=True)

set_parity_mode(ParityMode.NLE_BYTEPARITY)
env = NethaxEnv()

print(f"[{time.strftime('%H:%M:%S')}] reset(seed=0)", flush=True)
t0 = time.time()
state, _ = env.reset(jax.random.PRNGKey(0), role=Role.ROGUE,
                     race=Race.HUMAN, alignment=2)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), state)
t_reset = time.time() - t0
print(f"[{time.strftime('%H:%M:%S')}] reset done in {t_reset:.1f}s", flush=True)

print(f"[{time.strftime('%H:%M:%S')}] step(action=0) — cold compile", flush=True)
t0 = time.time()
state, _, _, _, _ = env.step(state, 0, jax.random.PRNGKey(1))
jax.tree_util.tree_map(lambda x: x.block_until_ready(), state)
t_step_cold = time.time() - t0
print(f"[{time.strftime('%H:%M:%S')}] step COLD done in {t_step_cold:.1f}s",
      flush=True)

print(f"[{time.strftime('%H:%M:%S')}] step(action=0) — warm (cache hit)",
      flush=True)
t0 = time.time()
state, _, _, _, _ = env.step(state, 0, jax.random.PRNGKey(2))
jax.tree_util.tree_map(lambda x: x.block_until_ready(), state)
t_step_warm = time.time() - t0
print(f"[{time.strftime('%H:%M:%S')}] step WARM done in {t_step_warm*1000:.1f}ms",
      flush=True)

print(f"\n[SUMMARY] reset={t_reset:.1f}s cold_step={t_step_cold:.1f}s "
      f"warm_step={t_step_warm*1000:.1f}ms", flush=True)
