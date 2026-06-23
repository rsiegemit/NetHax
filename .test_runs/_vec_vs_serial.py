"""Same RNG, same start state, vec(simultaneous) vs serial(sequential) monster
step. Any difference is PURE DYNAMICS (move order), not RNG — isolates whether
the vec path differs from the faithful path by more than randomness."""
import os, numpy as np, jax, jax.numpy as jnp
# default (Threefry) mode — same RNG for both
from Nethax.nethax.minihax_env import MinihaxEnv  # noqa
def run(vec):
    os.environ["NETHAX_VEC_MONSTERS"] = vec
    import importlib, Nethax.nethax.subsystems.vec_monster_turns as v
    importlib.reload(v)
    from Nethax.minihax.minihax_env import MinihaxEnv
    from Nethax.nethax.subsystems import monster_ai as MA
    env = MinihaxEnv("MiniHack-Room-Monster-15x15-v0")
    s,_ = env.reset(jax.random.key(0)); jax.block_until_ready(s)
    k = jax.random.key(12345)
    with jax.disable_jit():
        for t in range(3):
            s = MA.monsters_step_all(s, jax.random.key(t+1))
    return np.asarray(s.monster_ai.pos), np.asarray(s.monster_ai.alive), int(np.asarray(s.player_hp))
