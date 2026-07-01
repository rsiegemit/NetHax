"""DCE-SAFE phase decomposition. The old _profile_phases returned a single field
(player_pos) per variant, so XLA dead-code-eliminated all state computation that
field didn't depend on -> the "obs=73%" reading was an artifact (build_glyphs is
actually 0.3ms in isolation). Here each variant returns a SCALAR REDUCTION over
EVERY leaf of the produced state, so nothing can be DCE'd: the timing reflects the
true cost of computing that phase's full state.

VARIANT: 1=dispatch 2=+pre-monster 3=+monster-turn 4=+post-monster 5=+obs(glyphs)
Threaded across N steps like the real bench. Cumulative deltas attribute cost.
"""
import os, time, jax, jax.numpy as jnp
import gym, gymnasium, gymnasium.spaces.dict as _g
_g.Space = (gymnasium.spaces.Space, gym.spaces.Space)
from Nethax.minihax.minihax_env import MinihaxEnv
from Nethax.nethax.env import (_clear_message, _pre_monster_jit, _monster_jit,
    _post_monster_jit, _obs_jit, _USE_JIT_SPLIT)
from Nethax.nethax.subsystems.action_dispatch import _move_shared

V = int(os.environ.get("VARIANT", "5")); B = int(os.environ.get("B", "512")); N = int(os.environ.get("NSTEP", "30"))
env = MinihaxEnv("MiniHack-Room-Monster-5x5-v0"); s0, info0 = env.reset(jax.random.key(0))


def _realize(state):
    """Scalar sum over every numeric/bool leaf -> forces full materialization."""
    tot = jnp.float32(0.0)
    for leaf in jax.tree_util.tree_leaves(state):
        if jnp.issubdtype(leaf.dtype, jnp.floating) or jnp.issubdtype(leaf.dtype, jnp.integer) or leaf.dtype == jnp.bool_:
            tot = tot + jnp.sum(leaf.astype(jnp.float32))
    return tot


def step(state, dir_idx, rng):
    rk = jax.random.split(rng, 9)
    rng_act, rng_monsters, rng_status, rng_poly, rng_shop, rng_swallow, rng_explvl, rng_regions, rng_astral = rk
    pw = jnp.any(state.monster_ai.alive & (state.monster_ai.entry_idx.astype(jnp.int32) == jnp.int32(281)))
    pb = state.dungeon.current_branch.astype(jnp.int32); pl = state.dungeon.current_level.astype(jnp.int32)
    ns0 = state.replace(messages=_clear_message(state.messages), action_consumed_turn=jnp.bool_(True))
    disp = _move_shared(ns0, rng_act, dir_idx)
    ns = jax.tree_util.tree_map(lambda n, o: jnp.where(state.done, o, n), disp, state)
    if V == 1: return ns
    ns = _pre_monster_jit(ns, state, rng_act, rng_astral, pb, pl)
    if V == 2: return ns
    ns = _monster_jit(ns, state, rng_monsters, rng_regions)
    if V == 3: return ns
    new = _post_monster_jit(ns, state, pw, rng_status, rng_poly, rng_shop, rng_swallow, rng_explvl)
    if V == 4: return new
    obs, reward = _obs_jit(state, new)
    return new  # obs computed but we thread `new`; realize() covers state


def threaded(state, rng):
    # V<5 return partial state (same pytree as input, safe to thread). V==5 also
    # returns `new` (full state) after computing obs. Realize forces materialization.
    ns = step(state, jnp.int32(0), rng)
    return ns

vstep = jax.jit(jax.vmap(lambda st, k: _realize(threaded(st, k))))
bs = jax.vmap(lambda _: s0)(jnp.arange(B))
dev = jax.devices()[0]
print(f"VARIANT={V} (1=disp 2=+pre 3=+monster 4=+post 5=+obs) device={dev.platform}:{dev.device_kind} B={B}", flush=True)
try:
    t = time.time(); out = vstep(bs, jax.random.split(jax.random.key(1), B)); jax.block_until_ready(out); print(f"compile {time.time()-t:.1f}s", flush=True)
    t = time.time()
    for i in range(N):
        out = vstep(bs, jax.random.split(jax.random.key(100 + i), B))
    jax.block_until_ready(out); dt = time.time() - t
    ms = jax.local_devices()[0].memory_stats() or {}
    peak = (ms.get("peak_bytes_in_use", 0) or 0) / 1e9
    print(f"  steady {N*B/dt:.0f} env-steps/s  ({1000*dt/N:.1f} ms/batch @ B={B})  peak {peak:.2f} GB", flush=True)
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:140]}", flush=True)
