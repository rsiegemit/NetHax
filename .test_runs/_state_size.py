"""Report the biggest EnvState leaves by bytes — find the per-env memory hogs
that block large-batch (B>=1000) training."""
import time
def log(m): print(m, flush=True)
def main():
    import jax, jax.numpy as jnp, numpy as np
    from Nethax.nethax.parity_mode import set_parity_mode, ParityMode
    set_parity_mode(ParityMode.NLE)
    from Nethax.minihax.minihax_env import MinihaxEnv
    env = MinihaxEnv("MiniHack-Room-Monster-15x15-v0")
    s0,_ = env.reset(jax.random.key(0)); jax.block_until_ready(s0)
    leaves = jax.tree_util.tree_leaves_with_path(s0)
    rows=[]
    total=0
    for path, a in leaves:
        try:
            nb = int(a.size) * int(a.dtype.itemsize)
        except Exception:
            nb = 0
        total += nb
        rows.append((nb, jax.tree_util.keystr(path), str(a.dtype), tuple(a.shape)))
    rows.sort(reverse=True)
    log(f"TOTAL state bytes/env = {total/1e6:.1f} MB")
    log("TOP 20 leaves by bytes:")
    for nb,k,dt,sh in rows[:20]:
        log(f"  {nb/1e6:8.2f} MB  {k:45s} {dt:8s} {sh}")
    log("DONE")
main()
