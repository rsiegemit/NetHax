"""Rank EVERY scan in the step by SERIAL cost = body_eqns * length (its serial
op-execution count) in NON-vendor + vec mode. Shows what remains serial after the
monster turn loop is vectorized — i.e. the next vectorization targets.

Usage:
  JAX_PLATFORMS=cpu PYTHONPATH=. NETHAX_VEC_MONSTERS=1 \
    .venv/bin/python -u .test_runs/_profile_scans.py
"""
import os, time


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def count_eqns(jx):
    n = 0
    for e in jx.eqns:
        n += 1
        for v in e.params.values():
            s = getattr(v, "jaxpr", None)
            if s is not None:
                n += count_eqns(s.jaxpr if hasattr(s, "jaxpr") else s)
            elif isinstance(v, (tuple, list)):
                for it in v:
                    s2 = getattr(it, "jaxpr", None)
                    if s2 is not None:
                        n += count_eqns(s2.jaxpr if hasattr(s2, "jaxpr") else s2)
    return n


def main():
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.parity_mode import use_vendor_rng
    from Nethax.minihax.minihax_env import MinihaxEnv

    log(f"vec_monsters={os.environ.get('NETHAX_VEC_MONSTERS','1')} use_vendor_rng={use_vendor_rng()}")
    env = MinihaxEnv("MiniHack-Room-Monster-15x15-v0")
    s0, _ = env.reset(jax.random.key(0))
    jax.block_until_ready(s0)
    B = 1
    st = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), s0)
    rngs = jax.vmap(jax.random.key)(jnp.arange(B, dtype=jnp.uint32))
    acts = jnp.zeros((B,), dtype=jnp.int32)
    log("tracing ...")
    cj = jax.make_jaxpr(
        lambda s, a, r: env._engine.step_batched(s, a, r, static_action=0))(st, acts, rngs)

    scans = []

    from jax._src import source_info_util as _siu

    def _src(e):
        # Walk the user-frame stack; return the first few Nethax frames so we can
        # see which subsystem owns the scan (skip the vec/profiler frames).
        try:
            tb = e.source_info.traceback
            frames = []
            for f in tb.raw_frames() if hasattr(tb, "raw_frames") else []:
                pass
        except Exception:
            pass
        try:
            s = _siu.summarize(e.source_info, num_frames=6)
            # keep the Nethax-relevant fragments
            parts = [p for p in s.split() if "/Nethax/" in p or "nethax/" in p]
            picks = [p.split("nethax/")[-1] for p in parts if "vec_monster" not in p]
            return " <- ".join(picks[:3]) if picks else s[:80]
        except Exception as ex:
            return f"?({ex})"

    def walk(jx, depth=0):
        for e in jx.eqns:
            if e.primitive.name == "scan":
                j = e.params.get("jaxpr")
                body = j.jaxpr if hasattr(j, "jaxpr") else j
                be = count_eqns(body)
                ln = e.params.get("length")
                serial = be * (ln if isinstance(ln, int) else 1)
                scans.append((serial, be, ln, depth, _src(e)))
            for v in e.params.values():
                s = getattr(v, "jaxpr", None)
                if s is not None:
                    walk(s.jaxpr if hasattr(s, "jaxpr") else s, depth + 1)
                elif isinstance(v, (tuple, list)):
                    for it in v:
                        s2 = getattr(it, "jaxpr", None)
                        if s2 is not None:
                            walk(s2.jaxpr if hasattr(s2, "jaxpr") else s2, depth + 1)
    walk(cj.jaxpr)
    scans.sort(reverse=True)
    total_serial = sum(s[0] for s in scans)
    log(f"total scans={len(scans)}  sum(serial=body_eqns*len)={total_serial:,}")
    log("TOP 20 by serial op-executions (serial, body_eqns, length, depth, src):")
    for s in scans[:20]:
        log(f"   serial={s[0]:>10,}  body={s[1]:>7}  len={str(s[2]):>5}  d={s[3]}  {s[4]}")
    log("DONE")


if __name__ == "__main__":
    main()
