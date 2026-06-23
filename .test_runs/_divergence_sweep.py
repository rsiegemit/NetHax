"""vec vs serial monster step, SAME RNG, per-(seed,step) position hash.
Run twice (NETHAX_VEC_MONSTERS=0/1) and diff -> divergence rate = fraction of
(seed,step) where simultaneous != sequential (the vec-mode transfer risk)."""
import os, sys, hashlib, numpy as np, jax
from Nethax.minihax.minihax_env import MinihaxEnv
from Nethax.nethax.subsystems import monster_ai as MA
ENVN = sys.argv[1] if len(sys.argv)>1 else "MiniHack-Room-Monster-15x15-v0"
NSEED = int(sys.argv[2]) if len(sys.argv)>2 else 8
NSTEP = int(sys.argv[3]) if len(sys.argv)>3 else 10
step = jax.jit(MA.monsters_step_all)
for seed in range(NSEED):
    env = MinihaxEnv(ENVN)
    s,_ = env.reset(jax.random.key(seed)); jax.block_until_ready(s)
    for t in range(NSTEP):
        s = step(s, jax.random.key(seed*1000+t)); jax.block_until_ready(s)
        mai=s.monster_ai; al=np.asarray(mai.alive)
        key=np.asarray(mai.pos)[al].tobytes()+np.asarray(mai.hp)[al].tobytes()
        print(f"{seed},{t},{hashlib.sha256(key).hexdigest()[:16]},{int(al.sum())}")
