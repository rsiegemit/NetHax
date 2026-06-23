"""Determine the EXACT shared-write surface of monster_turn for the vectorized
merge reduction spec. Runs one monster_turn on an acting monster and prints the
pytree paths of every leaf that changed vs the frozen start-of-turn state, with
shape/dtype — so the vec-merge spec is fact-based, not guessed.

Usage:
  JAX_PLATFORMS=cpu NETHAX_EAGER=1 PYTHONPATH=. .venv/bin/python -u .test_runs/_monster_write_surface.py
"""
import time


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from Nethax.nethax.parity_mode import set_parity_mode, ParityMode
    set_parity_mode(ParityMode.NLE)
    from Nethax.minihax.minihax_env import MinihaxEnv
    from Nethax.nethax.subsystems import monster_ai as MA

    env = MinihaxEnv("MiniHack-Room-Monster-15x15-v0")
    s0, _ = env.reset(jax.random.key(0))
    jax.block_until_ready(s0)
    mai = s0.monster_ai
    alive = np.asarray(mai.alive)
    idxs = np.nonzero(alive)[0]
    log(f"alive monsters: {len(idxs)} -> slots {idxs[:10].tolist()}")
    if len(idxs) == 0:
        log("no alive monster; aborting")
        return

    # Run a turn for the first few alive monsters; union the changed-leaf paths.
    changed = {}
    for slot in idxs[:5]:
        out = MA.monster_turn(s0, jax.random.key(int(slot) + 1), jnp.int32(int(slot)))
        l0 = jax.tree_util.tree_leaves_with_path(s0)
        l1 = jax.tree_util.tree_leaves_with_path(out)
        for (path, a), (_p, b) in zip(l0, l1):
            if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
                a = jax.random.key_data(a); b = jax.random.key_data(b)
            a = np.asarray(a); b = np.asarray(b)
            if a.shape != b.shape:
                key = jax.tree_util.keystr(path)
                changed.setdefault(key, ("SHAPE-CHANGE", a.shape, b.shape))
                continue
            if not np.array_equal(a, b):
                key = jax.tree_util.keystr(path)
                ndiff = int(np.sum(a != b))
                changed.setdefault(key, (str(a.dtype), a.shape, f"{ndiff} cells"))

    log(f"=== CHANGED LEAVES across {min(5,len(idxs))} acting monsters: {len(changed)} ===")
    for k in sorted(changed):
        dt, shp, info = changed[k]
        mai_tag = "  [monster_ai per-slot]" if ".monster_ai." in k else ""
        log(f"  {k:55s} dtype={dt:8s} shape={str(shp):20s} {info}{mai_tag}")
    log("DONE")


if __name__ == "__main__":
    main()
