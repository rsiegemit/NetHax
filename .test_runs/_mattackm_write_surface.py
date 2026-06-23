"""Write-surface of mattackm (m-vs-m strike) for the vectorized scatter-merge.

Forces an attack between two monster slots and prints which EnvState leaves change
(esp. which monster_ai per-slot fields, at attacker vs defender slot), so the
vectorized m-vs-m merge uses correct per-field reductions (hp: sum-diff; alive:
AND; etc.) and a correct scatter target.

Usage:
  JAX_PLATFORMS=cpu NETHAX_EAGER=1 PYTHONPATH=. .venv/bin/python -u .test_runs/_mattackm_write_surface.py
"""
import time
import numpy as np


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.parity_mode import set_parity_mode, ParityMode
    set_parity_mode(ParityMode.NLE)
    from Nethax.minihax.minihax_env import MinihaxEnv
    from Nethax.nethax.subsystems import monster_ai as MA

    env = MinihaxEnv("MiniHack-Room-Monster-15x15-v0")
    s0, _ = env.reset(jax.random.key(0))
    jax.block_until_ready(s0)
    alive = np.asarray(s0.monster_ai.alive)
    idxs = np.nonzero(alive)[0]
    log(f"alive slots: {idxs[:6].tolist()}")
    if len(idxs) < 2:
        log("need >=2 alive; aborting"); return
    atk, dfn = int(idxs[0]), int(idxs[1])

    # Force them adjacent + opposite faction so mattackm actually strikes.
    mai = s0.monster_ai
    apos = np.asarray(mai.pos)
    apos2 = apos.copy(); apos2[dfn] = apos[atk] + np.array([0, 1])
    mai = mai.replace(
        pos=jnp.asarray(apos2, mai.pos.dtype),
        tame=mai.tame.at[atk].set(False).at[dfn].set(True),     # hostile vs pet
        peaceful=mai.peaceful.at[atk].set(False),
    )
    s = s0.replace(monster_ai=mai)

    log(f"forced atk={atk} (hostile) dfn={dfn} (tame), adjacent. hp0={np.asarray(s.monster_ai.hp)[[atk,dfn]].tolist()}")
    changed = {}
    n_hit = 0
    for kk in range(40):
        out = MA.mattackm(s, jnp.int32(atk), jnp.int32(dfn), jax.random.key(kk))
        l0 = jax.tree_util.tree_leaves_with_path(s)
        l1 = jax.tree_util.tree_leaves_with_path(out)
        any_change = False
        for (path, a), (_p, b) in zip(l0, l1):
            if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
                a = jax.random.key_data(a); b = jax.random.key_data(b)
            a = np.asarray(a); b = np.asarray(b)
            if a.shape != b.shape or not np.array_equal(a, b):
                any_change = True
                key = jax.tree_util.keystr(path)
                where = ""
                if ".monster_ai." in key and a.ndim >= 1 and a.shape[0] == len(alive):
                    ds = np.nonzero(np.any((a != b).reshape(a.shape[0], -1), axis=1))[0]
                    where = f" @slots={ds.tolist()[:8]}"
                changed.setdefault(key, (str(a.dtype), where))
        if any_change:
            n_hit += 1
    log(f"=== {n_hit}/40 keys produced changes; union of changed leaves ===")
    for k in sorted(changed):
        dt, where = changed[k]
        tag = "  [monster_ai]" if ".monster_ai." in k else ""
        log(f"  {k:50s} dtype={dt:8s}{tag}{where}")
    log("DONE")


if __name__ == "__main__":
    main()
