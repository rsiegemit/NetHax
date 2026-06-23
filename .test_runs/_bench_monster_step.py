"""Isolated GPU bench of monsters_step_all (NOT full step_batched, so no
FOV/timer/obs scans) — measures the actual per-op rate behind the projected
"~1ms/serial-op" GPU model, and the real payoff of NETHAX_VEC_MONSTERS.

Two configs, compared on the SAME compiled function:
  * vec-on  (NETHAX_VEC_MONSTERS=1): the monster step's serial chain is collapsed
    to wide vmap ops -> should compile + exec fast and COMPLETE.
  * vec-off (NETHAX_VEC_MONSTERS=0): the 27.3M-serial-op chain -> exec is the
    >8h wall. Bounded by --exec-watchdog so we can read its rate without hanging.

Runs in NON-vendor (Threefry/training) mode. monsters_step_all is single-env; we
optionally vmap it over B to check batch scaling of the vec path.

Usage (GPU):
  PYTHONPATH=. JAX_COMPILATION_CACHE_DIR=$PWD/.jax_compile_cache_gpu_ms \
  NETHAX_BENCH_NO_PARITY=1 NETHAX_VEC_MONSTERS=1 \
  python -u .test_runs/_bench_monster_step.py --batches 1,64,1024 --warm 20
"""
import argparse, json, os, signal, statistics, threading, time

_T0 = time.perf_counter()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} +{int(time.perf_counter()-_T0):>5}s] {m}", flush=True)


def _hb():
    import resource
    while True:
        time.sleep(120)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        log(f"... heartbeat alive rss={rss:.1f}")


class _Timeout(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniHack-Room-Monster-15x15-v0")
    ap.add_argument("--batches", default="1,64,1024")
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--exec-watchdog", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",") if b]
    threading.Thread(target=_hb, daemon=True).start()

    import jax
    import jax.numpy as jnp
    from Nethax.nethax.parity_mode import use_vendor_rng
    from Nethax.minihax.minihax_env import MinihaxEnv
    from Nethax.nethax.subsystems import monster_ai as MA

    vec = os.environ.get("NETHAX_VEC_MONSTERS", "1")
    log(f"backend={jax.default_backend()} vec_monsters={vec} vendor_rng={use_vendor_rng()}")
    env = MinihaxEnv(args.env)
    s1, _ = env.reset(jax.random.key(0))
    jax.block_until_ready(s1)
    log("reset done")

    out = {"backend": jax.default_backend(), "vec": vec, "scaling": []}
    for B in batches:
        rec = {"B": B}
        try:
            if B == 1:
                st = s1
                fn = jax.jit(MA.monsters_step_all)
                mk = lambda k: (st, k)
            else:
                st = jax.tree_util.tree_map(
                    lambda x: jnp.broadcast_to(x, (B,) + x.shape), s1)
                fn = jax.jit(jax.vmap(MA.monsters_step_all))
                ks = jax.vmap(jax.random.key)(jnp.arange(B, dtype=jnp.uint32))
                mk = lambda k: (st, ks)
            k0 = jax.random.key(1)
            args_t = mk(k0)

            t = time.perf_counter()
            traced = fn.trace(*args_t)
            lowered = traced.lower()
            comp = lowered.compile()
            rec["compile_s"] = time.perf_counter() - t
            log(f"  B={B} compile {rec['compile_s']:.1f}s")

            if args.exec_watchdog:
                signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_Timeout()))
                signal.alarm(args.exec_watchdog)
            t = time.perf_counter()
            try:
                r = comp(*args_t); jax.block_until_ready(r)
            finally:
                signal.alarm(0)
            rec["exec1_s"] = time.perf_counter() - t
            log(f"  B={B} exec1 {rec['exec1_s']:.3f}s")

            ts = []
            for _ in range(args.warm):
                t = time.perf_counter()
                r = comp(*args_t); jax.block_until_ready(r)
                ts.append(time.perf_counter() - t)
            warm = statistics.median(ts)
            rec["warm_ms"] = warm * 1e3
            rec["env_steps_per_s"] = B / warm
            log(f"  B={B} warm {warm*1e3:.3f}ms  env-steps/s={B/warm:.0f}")
        except _Timeout:
            rec["error"] = f"exec1 exceeded {args.exec_watchdog}s"
            log(f"  B={B} EXEC WATCHDOG {args.exec_watchdog}s")
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            log(f"  B={B} ERROR {rec['error']}")
        out["scaling"].append(rec)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
    log("DONE")


if __name__ == "__main__":
    main()
