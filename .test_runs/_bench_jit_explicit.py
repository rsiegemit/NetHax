"""JIT per-step env-scaling bench via the EXPLICIT trace->lower->compile->invoke
path (NOT the implicit pjit `step_fn(...)` call).

Why this file exists: the overnight probe (2026-06-22) proved the ">1h never
finishes" compile is a pathology of the implicit pjit function-call path. The
explicit `traced.lower().compile()` + `compiled(...)` invoke compiles in ~40 s
(full opt) and runs. `_bench_jit.py` still uses the stalling pjit path; this one
does not. Use THIS for reliable GPU numbers.

For each batch size B reports:
  * compile_s    — explicit lower().compile() wall (watchdog-bounded)
  * exec1_s      — first invoke
  * warm_ms      — median of N warm invokes
  * env_steps/s  — B / warm
  * us_per_env   — warm_ms*1000 / B

A heartbeat thread logs elapsed + RSS every 5 min. A SIGALRM watchdog bounds
EACH compile so one stalled shape can't hang the whole job.

Usage (GPU):
  PYTHONPATH=. JAX_COMPILATION_CACHE_DIR=$PWD/.jax_compile_cache_gpu \
  python -u .test_runs/_bench_jit_explicit.py \
      --env MiniHack-Room-15x15-v0 --batches 1,4,16,64,256 \
      --warm 20 --compile-watchdog 3600 --json gpu_explicit.json
"""
import argparse, json, os, resource, signal, statistics, threading, time

_T0 = time.perf_counter()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} +{int(time.perf_counter()-_T0):>6}s] {m}", flush=True)


def _heartbeat():
    while True:
        time.sleep(300)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        # Linux ru_maxrss is KB, macOS is bytes; report raw scaled both ways noted in log.
        log(f"... heartbeat: alive, ru_maxrss={rss:.1f} (GB on linux/KB-base)")


class _CompileTimeout(Exception):
    pass


class _ExecTimeout(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniHack-Room-15x15-v0")
    ap.add_argument("--batches", default="1,4,16,64,256")
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--compile-watchdog", type=int, default=3600)  # per-B compile bound
    ap.add_argument("--exec-watchdog", type=int, default=0)  # per-B exec1 bound (0=off)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",") if b]

    threading.Thread(target=_heartbeat, daemon=True).start()

    log(f"cache_dir={os.environ.get('JAX_COMPILATION_CACHE_DIR','<none>')}")
    log(f"XLA_FLAGS={os.environ.get('XLA_FLAGS','<none>')}")

    import jax
    import jax.numpy as jnp
    from Nethax.nethax.parity_mode import set_parity_mode, ParityMode, use_vendor_rng
    # NETHAX_BENCH_NO_PARITY=1 => default (Threefry) training mode, which is the
    # path the vectorized monster step (NETHAX_VEC_MONSTERS) runs in. Without it,
    # NLE byte-parity mode forces the serial vendor scan (the >8h path).
    if os.environ.get("NETHAX_BENCH_NO_PARITY", "0") != "1":
        set_parity_mode(ParityMode.NLE)
    from Nethax.minihax.minihax_env import MinihaxEnv

    out = {"env": args.env, "backend": jax.default_backend(),
           "device": str(jax.devices()[0]), "path": "explicit-lower-compile",
           "vendor_rng": bool(use_vendor_rng()),
           "vec_monsters": os.environ.get("NETHAX_VEC_MONSTERS", "1"),
           "scaling": []}
    log(f"vendor_rng={out['vendor_rng']} vec_monsters={out['vec_monsters']}")
    log(f"backend={out['backend']} device={out['device']} env={args.env}")

    env = MinihaxEnv(args.env)
    eng = env._engine
    t = time.perf_counter()
    st1, _ = env.reset(jax.random.key(0))
    jax.block_until_ready(st1)
    out["reset_compile_s"] = time.perf_counter() - t
    log(f"reset compile {out['reset_compile_s']:.1f}s")
    t = time.perf_counter()
    st_w, _ = env.reset(jax.random.key(1))
    jax.block_until_ready(st_w)
    out["reset_warm_s"] = time.perf_counter() - t
    log(f"reset warm {out['reset_warm_s']:.2f}s")

    fn = jax.jit(lambda s, a, r: eng.step_batched(s, a, r, static_action=0))

    log(f"{'B':>7} {'compile_s':>10} {'exec1_s':>9} {'warm_ms':>10} {'env-steps/s':>13} {'us/env':>9}")
    for B in batches:
        rec = {"B": B}
        try:
            st = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (B,) + x.shape), st1)
            rngs = jax.vmap(jax.random.key)(jnp.arange(B, dtype=jnp.uint32))
            acts = jnp.zeros((B,), dtype=jnp.int32)

            t = time.perf_counter()
            traced = fn.trace(st, acts, rngs)
            rec["trace_s"] = time.perf_counter() - t
            t = time.perf_counter()
            lowered = traced.lower()
            rec["lower_s"] = time.perf_counter() - t
            log(f"  B={B} trace {rec['trace_s']:.1f}s lower {rec['lower_s']:.1f}s -> compile (watchdog {args.compile_watchdog}s) ...")

            def _alarm(signum, frame):
                raise _CompileTimeout()
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(args.compile_watchdog)
            t = time.perf_counter()
            try:
                compiled = lowered.compile()
            finally:
                signal.alarm(0)
            rec["compile_s"] = time.perf_counter() - t
            log(f"  B={B} COMPILE DONE {rec['compile_s']:.1f}s")

            if args.exec_watchdog:
                def _exec_alarm(signum, frame):
                    raise _ExecTimeout()
                signal.signal(signal.SIGALRM, _exec_alarm)
                signal.alarm(args.exec_watchdog)
                log(f"  B={B} exec1 (watchdog {args.exec_watchdog}s) ...")
            t = time.perf_counter()
            try:
                r = compiled(st, acts, rngs)
                jax.block_until_ready(r)
            finally:
                signal.alarm(0)
            rec["exec1_s"] = time.perf_counter() - t
            log(f"  B={B} EXEC#1 DONE {rec['exec1_s']:.2f}s")

            st2 = r[0]
            ts = []
            for _ in range(args.warm):
                t = time.perf_counter()
                r = compiled(st2, acts, rngs)
                jax.block_until_ready(r)
                ts.append(time.perf_counter() - t)
            warm = statistics.median(ts)
            rec["warm_ms"] = warm * 1e3
            rec["env_steps_per_s"] = B / warm
            rec["us_per_env"] = warm * 1e6 / B
            log(f"{B:>7} {rec['compile_s']:>10.1f} {rec['exec1_s']:>9.2f} "
                f"{rec['warm_ms']:>10.3f} {rec['env_steps_per_s']:>13.0f} {rec['us_per_env']:>9.1f}")
        except _CompileTimeout:
            rec["error"] = f"compile exceeded {args.compile_watchdog}s watchdog"
            log(f"{B:>7}  COMPILE WATCHDOG: exceeded {args.compile_watchdog}s")
        except _ExecTimeout:
            rec["error"] = f"exec1 exceeded {args.exec_watchdog}s watchdog (exec is the wall, not compile)"
            log(f"{B:>7}  EXEC WATCHDOG: exec1 exceeded {args.exec_watchdog}s")
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            log(f"{B:>7}  ERROR: {rec['error']}")
        out["scaling"].append(rec)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        log(f"wrote {args.json}")
    log("EXPLICIT JIT BENCH DONE")


if __name__ == "__main__":
    main()
