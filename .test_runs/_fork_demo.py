"""DEMO: state snapshot / fork / rewind / parallel lookahead — the capability
NLE's opaque C process cannot offer. The env state is an immutable JAX pytree,
so cloning it is free and forking N counterfactual futures is one vmap kernel.

Run:  JAX_PLATFORMS=cpu NETHAX_VEC_MONSTERS=1 .venv/bin/python .test_runs/_fork_demo.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp
import gym, gymnasium, gymnasium.spaces.dict as _g
_g.Space = (gymnasium.spaces.Space, gym.spaces.Space)
from Nethax.minihax.minihax_env import MinihaxEnv
from Nethax.nethax.env import _make_restricted_step_impl

# 8 compass moves: N S E W NE NW SE SW  (dir_idx 0..7)
DIRS = ["k(N)", "j(S)", "l(E)", "h(W)", "u(NE)", "y(NW)", "n(SE)", "b(SW)"]
rstep = _make_restricted_step_impl((107, 108, 106, 104, 117, 121, 110, 98), True)

env = MinihaxEnv("MiniHack-Room-Monster-5x5-v0")
s0, info = env.reset(jax.random.key(0))
key = jax.random.key(7)
p0 = (int(s0.player_pos[0]), int(s0.player_pos[1]))
print(f"root state: player at {p0}\n")

# ---- 1) FORK: try ALL 8 actions from the SAME state, in parallel ----------
# One vmap = 8 counterfactual futures from one root. NLE would need 8 cloned
# C processes; here the root pytree is simply read 8 times.
def step_from_root(a):
    ns, obs, r, d = rstep(s0, a, key)
    return ns
branched = jax.jit(jax.vmap(step_from_root))(jnp.arange(8))
print("1) FORK — one-ply: 8 futures from the SAME root, computed in ONE vmap:")
for i, name in enumerate(DIRS):
    pr, pc = int(branched.player_pos[i, 0]), int(branched.player_pos[i, 1])
    moved = "" if (pr, pc) != p0 else "  (blocked/wall)"
    print(f"     action {name:6} -> player {(pr, pc)}{moved}")

# ---- 2) REWIND: the root is untouched — it's immutable ---------------------
print(f"\n2) REWIND — root still at {p0}: "
      f"{(int(s0.player_pos[0]), int(s0.player_pos[1])) == p0}  "
      "(snapshot = free; no re-simulation)")

# ---- 3) 2-PLY TREE: fork each of the 8 into 8 more -> 64 leaves, one shot --
def two_ply(a1, a2):
    s1, *_ = rstep(s0, a1, key)
    s2, *_ = rstep(s1, a2, key)
    return s2.player_pos
grid = jax.jit(jax.vmap(jax.vmap(two_ply, in_axes=(None, 0)), in_axes=(0, None)))(
    jnp.arange(8), jnp.arange(8))            # [8, 8, 2]
uniq = {(int(grid[i, j, 0]), int(grid[i, j, 1])) for i in range(8) for j in range(8)}
print(f"\n3) 2-PLY TREE — 8x8 = 64 leaf states from one root in a single kernel; "
      f"{len(uniq)} distinct reachable tiles.")

# ---- 4) PARALLEL 1-PLY LOOKAHEAD: pick the action maximizing a heuristic ----
# Toy heuristic: Chebyshev distance moved from the root (explore-outward).
def value_of(a):
    ns, *_ = rstep(s0, a, key)
    d = jnp.maximum(jnp.abs(ns.player_pos[0] - s0.player_pos[0]),
                    jnp.abs(ns.player_pos[1] - s0.player_pos[1]))
    return d.astype(jnp.int32)
vals = jax.jit(jax.vmap(value_of))(jnp.arange(8))
best = int(jnp.argmax(vals))
print(f"\n4) LOOKAHEAD — evaluate all 8 actions in parallel, pick best: "
      f"{DIRS[best]} (heuristic={int(vals[best])}).")
print("\nAll of the above is impossible to do cheaply in NLE: its running C game "
      "state cannot be snapshotted, forked, or rewound. Here it's just a pytree.")
