"""Isolate FOV on GPU: compile + time view_from(terrain, player_pos) alone.

Discriminator for the full-step hang: monsters_step_all is now 27ms (vectorized),
so the remaining full-step hang is FOV / timer / obs. view_from is directly
callable, so timing it in isolation says whether FOV is the culprit:
  * fast  -> FOV is NOT the bottleneck; the timer queue (big body) or obs is.
  * slow/hangs -> FOV's depth-3 nested sweep is the target; vectorize it next.

Usage (GPU):
  PYTHONPATH=. python -u .test_runs/_bench_fov.py --exec-watchdog 600
"""
import argparse, os, signal, statistics, time


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class _TO(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniHack-Room-Monster-15x15-v0")
    ap.add_argument("--exec-watchdog", type=int, default=0)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from Nethax.minihax.minihax_env import MinihaxEnv
    from Nethax.nethax.fov import view_from

    log(f"backend={jax.default_backend()}")
    env = MinihaxEnv(args.env)
    s0, _ = env.reset(jax.random.key(0))
    jax.block_until_ready(s0)
    terrain = s0.terrain[0, 0, :, :]
    ppos = s0.player_pos.astype(jnp.int32)
    log(f"terrain {terrain.shape} player {ppos.tolist()}")

    fn = jax.jit(lambda t, p: view_from(t, p, max_radius=0))
    t = time.perf_counter()
    comp = fn.trace(terrain, ppos).lower().compile()
    log(f"view_from compile {time.perf_counter()-t:.1f}s")

    if args.exec_watchdog:
        signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_TO()))
        signal.alarm(args.exec_watchdog)
    try:
        t = time.perf_counter()
        r = comp(terrain, ppos); jax.block_until_ready(r)
        log(f"view_from EXEC#1 {time.perf_counter()-t:.3f}s  visible_cells={int(r.sum())}")
        signal.alarm(0)
        ts = []
        for _ in range(10):
            t = time.perf_counter()
            r = comp(terrain, ppos); jax.block_until_ready(r)
            ts.append(time.perf_counter() - t)
        log(f"view_from warm {statistics.median(ts)*1e3:.3f}ms")
    except _TO:
        log(f"view_from EXEC WATCHDOG: exceeded {args.exec_watchdog}s -> FOV IS the hang")
    log("DONE")


if __name__ == "__main__":
    main()
