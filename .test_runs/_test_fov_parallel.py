"""Compare view_from_parallel (vec) vs view_from (vendor shadow-cast): sanity that
the parallel FOV produces reasonable visibility (similar cell count, high overlap).
"""
import time
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def main():
    import jax, jax.numpy as jnp, numpy as np
    from Nethax.nethax.parity_mode import set_parity_mode, ParityMode
    set_parity_mode(ParityMode.NLE)
    from Nethax.minihax.minihax_env import MinihaxEnv
    from Nethax.nethax.fov import view_from, view_from_parallel
    env = MinihaxEnv("MiniHack-Room-Monster-15x15-v0")
    s0,_ = env.reset(jax.random.key(0)); jax.block_until_ready(s0)
    terr = s0.terrain[0,0]; ppos = s0.player_pos.astype(jnp.int32)
    a = np.asarray(view_from(terr, ppos, max_radius=0))
    b = np.asarray(view_from_parallel(terr, ppos, max_radius=0))
    inter = (a & b).sum(); union = (a | b).sum()
    log(f"vendor visible={a.sum()}  parallel visible={b.sum()}  IoU={inter/max(union,1):.3f}")
    log(f"in vendor-not-parallel={int((a&~b).sum())}  in parallel-not-vendor={int((b&~a).sum())}")
    log("DONE")
main()
