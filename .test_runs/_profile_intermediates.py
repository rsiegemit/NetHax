"""Find the biggest INTERMEDIATE arrays in the step jaxpr (per-env, B=1) — the
~85MB/env activation that OOMs B>=256 after the state was shrunk."""
import time
def log(m): print(m, flush=True)
def main():
    import os
    import jax, jax.numpy as jnp
    from Nethax.nethax.parity_mode import use_vendor_rng
    from Nethax.minihax.minihax_env import MinihaxEnv
    from jax._src import source_info_util as siu
    env = MinihaxEnv("MiniHack-Room-Monster-15x15-v0")
    s0,_ = env.reset(jax.random.key(0)); jax.block_until_ready(s0)
    B=1
    st = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x,(B,)+x.shape), s0)
    rngs = jax.vmap(jax.random.key)(jnp.arange(B,dtype=jnp.uint32))
    acts = jnp.zeros((B,),jnp.int32)
    log(f"vendor={use_vendor_rng()} tracing ...")
    cj = jax.make_jaxpr(lambda s,a,r: env._engine.step_batched(s,a,r,static_action=0))(st,acts,rngs)
    best=[]
    def src(e):
        try:
            s=siu.summarize(e.source_info,num_frames=6)
            parts=[p.split("nethax/")[-1] for p in s.split() if "nethax/" in p and "vec_monster" not in p]
            return " <- ".join(parts[:2]) if parts else s[:60]
        except Exception: return "?"
    def walk(jx):
        for e in jx.eqns:
            for ov in e.outvars:
                av=getattr(ov,"aval",None)
                if av is not None and hasattr(av,"size"):
                    try: nb=int(av.size)*int(av.dtype.itemsize)
                    except Exception: nb=0
                    if nb>2_000_000:
                        best.append((nb,e.primitive.name,tuple(av.shape),str(av.dtype),src(e)))
            for v in e.params.values():
                sub=getattr(v,"jaxpr",None)
                if sub is not None: walk(sub.jaxpr if hasattr(sub,"jaxpr") else sub)
                elif isinstance(v,(tuple,list)):
                    for it in v:
                        s2=getattr(it,"jaxpr",None)
                        if s2 is not None: walk(s2.jaxpr if hasattr(s2,"jaxpr") else s2)
    walk(cj.jaxpr)
    best.sort(reverse=True)
    log("TOP intermediates >2MB (per-env, B=1):")
    seen=set()
    for nb,pr,sh,dt,s in best[:20]:
        key=(nb,sh,s)
        if key in seen: continue
        seen.add(key)
        log(f"  {nb/1e6:8.2f} MB  {pr:14s} {str(sh):22s} {dt:8s} {s}")
    log("DONE")
main()
