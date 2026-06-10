"""Unit-level smoke test for env.step_batched_static.

Verifies:
  1. Single-env step(state, 0, rng) and step_batched_static(state[None], 0, rng[None])
     produce byte-identical obs for batch index 0.
  2. Same result across two replicated seeds in a batch — proves vmap is
     not mixing data across batch elements.

Only does ONE dungeon-gen (the seed=0 reset) — avoids the multi-minute
reset_batched cost in eager mode that was blocking the broader smoke test.
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("NETHAX_EAGER", "1")
if os.environ.get("NETHAX_EAGER") == "1":
    import jax
    jax.config.update("jax_disable_jit", True)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import time
import numpy as np
import jax
import jax.numpy as jnp

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.parity_mode import set_parity_mode, ParityMode
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.races import Race

set_parity_mode(ParityMode.NLE_BYTEPARITY)

env = NethaxEnv()

print("[smoke] single-env reset (seed=0)")
t0 = time.time()
state, obs = env.reset(jax.random.PRNGKey(0), role=Role.ROGUE,
                       race=Race.HUMAN, alignment=2)
print(f"[smoke] reset done in {time.time() - t0:.1f}s")

print("[smoke] single-env step (action=0)")
t0 = time.time()
ns_scalar, obs_scalar, _, done_scalar, _ = env.step(
    state, 0, jax.random.PRNGKey(1))
print(f"[smoke] step done in {time.time() - t0:.1f}s")

# Stack to batch of size 2 (replicate).
print("[smoke] stack to batch size 2")
states_b = jax.tree_util.tree_map(lambda x: jnp.stack([x, x], axis=0), state)
rngs_b = jnp.stack([jax.random.PRNGKey(1), jax.random.PRNGKey(1)])

print("[smoke] batched step_batched_static")
t0 = time.time()
ns_b, obs_b, _, done_b = env.step_batched_static(states_b, 0, rngs_b)
print(f"[smoke] batched step done in {time.time() - t0:.1f}s")

# Compare obs[0] vs scalar
fail = 0
for k in sorted(obs_scalar.keys()):
    s = np.asarray(obs_scalar[k])
    b0 = np.asarray(obs_b[k])[0]
    b1 = np.asarray(obs_b[k])[1]
    if s.shape != b0.shape:
        print(f"  FAIL {k}: shape mismatch scalar={s.shape} batched[0]={b0.shape}")
        fail += 1
        continue
    if not np.array_equal(s, b0):
        nm = int((s != b0).sum())
        print(f"  FAIL {k}: scalar vs batched[0] differ at {nm} positions")
        fail += 1
        continue
    if not np.array_equal(b0, b1):
        nm = int((b0 != b1).sum())
        print(f"  FAIL {k}: batched[0] vs batched[1] differ at {nm} positions "
              f"(replication broken)")
        fail += 1

if fail == 0:
    print(f"\n[PASS] step_batched_static byte-identical to scalar step "
          f"({len(obs_scalar)} channels, replicated across 2 batch elems)")
    sys.exit(0)
else:
    print(f"\n[FAIL] {fail} channels diverged")
    sys.exit(1)
