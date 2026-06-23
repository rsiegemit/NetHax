"""VERIFY the '27M serial op-executions' claim with jaxpr + HLO evidence.

Proves, structurally (not by arithmetic):
  1. The step contains ONE dominant lax.scan over the monster slots (length=400).
  2. That scan's body contains NESTED serial loops (while/scan) — the BFS
     pathfinder and the Bresenham LoS trace — i.e. loops-inside-the-loop.
  3. The serial op-execution count = scan_length * recursive_body_eqns, and the
     lowered StableHLO while-loop count, both consistent with the ~8h GPU wall.

Usage:
  JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python -u .test_runs/_verify_serial.py
"""
import time


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def count(jaxpr):
    c = {"eqns": 0, "cond": 0, "scan": 0, "while": 0, "select": 0}
    _walk(jaxpr, c)
    return c


def _subjaxprs(eqn):
    out = []
    for v in eqn.params.values():
        j = getattr(v, "jaxpr", None)
        if j is not None:
            out.append(j.jaxpr if hasattr(j, "jaxpr") else j)
        elif isinstance(v, (tuple, list)):
            for it in v:
                j2 = getattr(it, "jaxpr", None)
                if j2 is not None:
                    out.append(j2.jaxpr if hasattr(j2, "jaxpr") else j2)
    return out


def _walk(jaxpr, c):
    for e in jaxpr.eqns:
        c["eqns"] += 1
        n = e.primitive.name
        if n in ("cond", "switch"):
            c["cond"] += 1
        elif n == "scan":
            c["scan"] += 1
        elif n == "while":
            c["while"] += 1
        elif n in ("select_n", "select"):
            c["select"] += 1
        for sub in _subjaxprs(e):
            _walk(sub, c)


def loops_inside(jaxpr, depth=0, acc=None):
    """List every scan/while with its body eqn-count and nesting depth."""
    if acc is None:
        acc = []
    for e in jaxpr.eqns:
        n = e.primitive.name
        if n in ("scan", "while"):
            body = None
            if n == "scan":
                jx = e.params.get("jaxpr")
                body = jx.jaxpr if hasattr(jx, "jaxpr") else jx
                length = e.params.get("length")
            else:  # while
                jx = e.params.get("body_jaxpr")
                body = jx.jaxpr if hasattr(jx, "jaxpr") else jx
                length = "data-dep"
            bc = count(body) if body is not None else {"eqns": 0}
            acc.append({"kind": n, "depth": depth, "length": length,
                        "body_eqns": bc["eqns"], "body_scan": bc.get("scan", 0),
                        "body_while": bc.get("while", 0)})
        for sub in _subjaxprs(e):
            loops_inside(sub, depth + 1, acc)
    return acc


def main():
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.parity_mode import set_parity_mode, ParityMode
    set_parity_mode(ParityMode.NLE)
    from Nethax.minihax.minihax_env import MinihaxEnv

    log(f"backend={jax.default_backend()}")
    env = MinihaxEnv("MiniHack-Room-15x15-v0")
    st1, _ = env.reset(jax.random.key(0))
    jax.block_until_ready(st1)
    log("reset done")

    B = 1
    st = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), st1)
    rngs = jax.vmap(jax.random.key)(jnp.arange(B, dtype=jnp.uint32))
    acts = jnp.zeros((B,), dtype=jnp.int32)
    fn = lambda s, a, r: env._engine.step_batched(s, a, r, static_action=0)

    log("tracing step jaxpr ...")
    cj = jax.make_jaxpr(fn)(st, acts, rngs)
    whole = count(cj.jaxpr)
    log(f"WHOLE STEP recursive: {whole}")

    loops = loops_inside(cj.jaxpr)
    loops.sort(key=lambda d: d["body_eqns"], reverse=True)
    log(f"total loop primitives (scan+while) = {len(loops)}")
    log("TOP 15 loops (kind, nest_depth, length, body_eqns, nested_scan, nested_while):")
    for d in loops[:15]:
        log(f"   {d['kind']:>5} depth={d['depth']} len={str(d['length']):>8} "
            f"body_eqns={d['body_eqns']:>7} nested_scan={d['body_scan']:>4} "
            f"nested_while={d['body_while']:>4}")

    # The monster scan = the long scan whose body itself contains nested loops.
    monster = next((d for d in loops if isinstance(d["length"], int)
                    and d["length"] >= 100 and (d["body_scan"] + d["body_while"]) > 0), None)
    if monster:
        prod = monster["length"] * monster["body_eqns"]
        log(f">>> MONSTER SCAN: length={monster['length']} body_eqns={monster['body_eqns']} "
            f"-> serial op-executions ~= {prod:,}")
        log(f">>> nested serial loops inside its body: scan={monster['body_scan']} while={monster['body_while']} "
            f"(this is the loops-inside-the-loop = the nesting proof)")

    log("lowering to StableHLO (no compile) to count while-loops ...")
    hlo = jax.jit(fn).lower(st, acts, rngs).as_text()
    n_while = hlo.count("while(")
    n_lines = hlo.count("\n")
    log(f"StableHLO: {n_lines:,} lines, 'while(' occurrences = {n_while}")
    log("VERIFY DONE")


if __name__ == "__main__":
    main()
