"""Isolate build_glyphs cost + test the multi-level-gather hypothesis.

VARIANT (env):
  A = vmap(build_glyphs) as-is  (reads terrain[branch,level] from [7,32,21,80])
  B = vmap(build_glyphs) on a state whose terrain/last_seen/explored have been
      PRE-SLICED to a single level ([1,1,21,80]) with branch=level=0, so the
      [branch,level] index can only hit one level -> no 224-way gather.
Delta A-B => cost of the multi-level dynamic gather under vmap.
"""
import os, time, jax, jax.numpy as jnp
import gym, gymnasium, gymnasium.spaces.dict as _g
_g.Space = (gymnasium.spaces.Space, gym.spaces.Space)
from Nethax.minihax.minihax_env import MinihaxEnv
from Nethax.nethax.obs.nle_obs import build_glyphs

V = os.environ.get("VARIANT", "A")
B = int(os.environ.get("B", "512"))
N = int(os.environ.get("NSTEP", "30"))
env = MinihaxEnv("MiniHack-Room-Monster-5x5-v0")
s0, info0 = env.reset(jax.random.key(0))

if V == "B":
    # Pre-slice the three multi-level [7,32,21,80] arrays to the current level
    # only, shape [1,1,21,80], and zero branch/level so [branch,level]->[0,0].
    br = int(s0.dungeon.current_branch); lv = int(s0.dungeon.current_level) - 1
    d = s0.dungeon.replace(
        current_branch=jnp.zeros_like(s0.dungeon.current_branch),
        current_level=jnp.ones_like(s0.dungeon.current_level))
    s0 = s0.replace(
        dungeon=d,
        terrain=s0.terrain[br:br+1, lv:lv+1],
        last_seen_terrain=s0.last_seen_terrain[br:br+1, lv:lv+1],
        explored=s0.explored[br:br+1, lv:lv+1],
    )

vg = jax.jit(jax.vmap(build_glyphs))
bs = jax.vmap(lambda _: s0)(jnp.arange(B))
dev = jax.devices()[0]
print(f"VARIANT={V} device={dev.platform}:{dev.device_kind} B={B}", flush=True)
t = time.time(); out = vg(bs); jax.block_until_ready(out); print(f"compile {time.time()-t:.1f}s", flush=True)
t = time.time()
for _ in range(N):
    out = vg(bs)
jax.block_until_ready(out); dt = time.time() - t
ms = jax.local_devices()[0].memory_stats() or {}
peak = (ms.get("peak_bytes_in_use", 0) or 0) / 1e9
print(f"  build_glyphs: {N*B/dt:.0f} calls/s  ({1000*dt/N:.1f} ms/batch @ B={B})  peak {peak:.2f} GB", flush=True)
