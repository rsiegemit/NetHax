"""Verify NETHAX_VEC_MONSTERS: (1) routing (non-vendor training mode), (2) jaxpr
collapse of the 400-monster serial scan, (3) CPU sanity over a few steps.

Run TWICE via the harness with NETHAX_VEC_MONSTERS=0 and =1 to compare.

Usage:
  JAX_PLATFORMS=cpu PYTHONPATH=. NETHAX_VEC_MONSTERS=1 \
    .venv/bin/python -u .test_runs/_verify_vec.py
"""
import os, time
import numpy as np


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def count_scans(jaxpr):
    """Return (total_scans, max_monster_scan) where max_monster_scan is the
    biggest (body_eqns, length) scan with length>=100 — the monster turn loop."""
    scans = []

    def walk(jx):
        for e in jx.eqns:
            if e.primitive.name == "scan":
                j = e.params.get("jaxpr")
                body = j.jaxpr if hasattr(j, "jaxpr") else j
                ne = _count_eqns(body)
                scans.append((ne, e.params.get("length")))
            for v in e.params.values():
                s = getattr(v, "jaxpr", None)
                if s is not None:
                    walk(s.jaxpr if hasattr(s, "jaxpr") else s)
                elif isinstance(v, (tuple, list)):
                    for it in v:
                        s2 = getattr(it, "jaxpr", None)
                        if s2 is not None:
                            walk(s2.jaxpr if hasattr(s2, "jaxpr") else s2)
    walk(jaxpr)
    big = [s for s in scans if isinstance(s[1], int) and s[1] >= 100]
    big.sort(reverse=True)
    return len(scans), (big[0] if big else None)


def _count_eqns(jx):
    n = 0
    for e in jx.eqns:
        n += 1
        for v in e.params.values():
            s = getattr(v, "jaxpr", None)
            if s is not None:
                n += _count_eqns(s.jaxpr if hasattr(s, "jaxpr") else s)
            elif isinstance(v, (tuple, list)):
                for it in v:
                    s2 = getattr(it, "jaxpr", None)
                    if s2 is not None:
                        n += _count_eqns(s2.jaxpr if hasattr(s2, "jaxpr") else s2)
    return n


def main():
    import jax
    import jax.numpy as jnp
    # DEFAULT (non-vendor / Threefry) mode — do NOT set ParityMode.NLE.
    from Nethax.nethax.parity_mode import use_vendor_rng
    from Nethax.minihax.minihax_env import MinihaxEnv

    vec = os.environ.get("NETHAX_VEC_MONSTERS", "1")
    log(f"NETHAX_VEC_MONSTERS={vec}  backend={jax.default_backend()}")
    log(f"use_vendor_rng()={use_vendor_rng()}  (must be False for the vec path)")

    env = MinihaxEnv("MiniHack-Room-Monster-15x15-v0")
    s0, _ = env.reset(jax.random.key(0))
    jax.block_until_ready(s0)
    n_alive = int(np.sum(np.asarray(s0.monster_ai.alive)))
    log(f"reset ok; alive monsters={n_alive}")

    # (2) jaxpr scan structure of the batched step.
    B = 1
    st = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), s0)
    rngs = jax.vmap(jax.random.key)(jnp.arange(B, dtype=jnp.uint32))
    acts = jnp.zeros((B,), dtype=jnp.int32)
    log("tracing step jaxpr ...")
    cj = jax.make_jaxpr(
        lambda s, a, r: env._engine.step_batched(s, a, r, static_action=0))(st, acts, rngs)
    total, monster = count_scans(cj.jaxpr)
    log(f"total scans={total}  biggest_len>=100 scan (body_eqns,len)={monster}")

    # (3) CPU sanity: eager monster step ONLY (avoids the slow action-dispatch
    # compile). Monsters should move; no two alive monsters share a tile.
    if os.environ.get("VEC_SANITY", "0") == "1":
        from Nethax.nethax.subsystems import monster_ai as MA
        log("CPU sanity: jit(monsters_step_all) x3 ...")
        step = jax.jit(MA.monsters_step_all)
        s = s0
        rng = jax.random.key(1)
        for t in range(3):
            rng, k = jax.random.split(rng)
            t0 = time.perf_counter()
            s = step(s, k)
            jax.block_until_ready(s)
            mai = s.monster_ai
            alive = np.asarray(mai.alive)
            pos = np.asarray(mai.pos)[alive]
            uniq = len({tuple(p) for p in pos})
            moved = int(np.sum(np.any(np.asarray(pos) != np.asarray(s0.monster_ai.pos)[alive], axis=1)))
            log(f"  step {t}: {time.perf_counter()-t0:6.1f}s alive={int(alive.sum())} "
                f"moved_from_start={moved} distinct_tiles={uniq}/{len(pos)} "
                f"player_hp={int(np.asarray(s.player_hp))}")
            if uniq != len(pos):
                log("  WARNING: monsters share a tile (collision-resolution gap)")
    log("DONE")


if __name__ == "__main__":
    main()
