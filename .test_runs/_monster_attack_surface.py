"""Full write surface of monster_turn INCLUDING the attack path — place a hostile
monster adjacent to the player so monster_turn bump-attacks, and capture every
changed EnvState leaf. Union with the move-only surface gives the complete set of
shared leaves the vec merge must handle (so the rest can be frozen -> OOM fix).

Usage:
  JAX_PLATFORMS=cpu NETHAX_EAGER=1 PYTHONPATH=. .venv/bin/python -u .test_runs/_monster_attack_surface.py
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
    mai = s0.monster_ai
    idxs = np.nonzero(np.asarray(mai.alive))[0]
    slot = int(idxs[0])
    ppos = np.asarray(s0.player_pos)
    # Place the monster directly adjacent to the player + hostile + awake.
    pos = np.asarray(mai.pos).copy()
    pos[slot] = ppos + np.array([0, 1])
    mai = mai.replace(
        pos=jnp.asarray(pos, mai.pos.dtype),
        tame=mai.tame.at[slot].set(False),
        peaceful=mai.peaceful.at[slot].set(False),
        asleep=mai.asleep.at[slot].set(False),
    )
    s = s0.replace(monster_ai=mai)
    log(f"player={ppos.tolist()} monster slot {slot} placed adjacent; player_hp0={int(np.asarray(s.player_hp))}")

    changed = {}
    for kk in range(40):
        out = MA.monster_turn(s, jax.random.key(kk + 100), jnp.int32(slot))
        for (path, a), (_p, b) in zip(jax.tree_util.tree_leaves_with_path(s),
                                      jax.tree_util.tree_leaves_with_path(out)):
            if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
                a = jax.random.key_data(a); b = jax.random.key_data(b)
            a = np.asarray(a); b = np.asarray(b)
            if a.shape != b.shape or not np.array_equal(a, b):
                changed.setdefault(jax.tree_util.keystr(path), str(a.dtype))
    log(f"=== union of changed leaves over 40 keys (attack scenario) ===")
    for k in sorted(changed):
        tag = "[monster_ai]" if ".monster_ai." in k else "[SHARED]"
        log(f"  {tag:14s} {k:48s} {changed[k]}")
    log("DONE")


if __name__ == "__main__":
    main()
