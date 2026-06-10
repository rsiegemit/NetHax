"""Parallel multi-seed byte-parity validator.

Architecture:
  * NLE side: ``multiprocessing.Pool`` of ``n_workers`` subprocesses, one
    NLE instance per worker (NLE uses C globals, can't be in-process).
    Workers receive ``(seed, action)`` over a pipe and return the obs
    dict for that step.
  * Nethax side: ``env.reset_batched`` then ``env.step_batched_static``
    with action=0.  All ``n_seeds`` advance in lock-step on the device
    in one vmap'd graph.
  * Per step: gather the per-worker NLE obs dicts, stack them into a
    batched dict, diff against the Nethax batched obs.

Usage:
    PYTHONPATH=. .venv/bin/python .test_runs/parallel_byteparity.py [n_seeds] [n_steps] [n_workers]

Default: n_seeds=4, n_steps=5, n_workers=min(n_seeds, os.cpu_count()).
"""
from __future__ import annotations

import io
import os
import sys
import time
import contextlib
import multiprocessing as mp

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


# ---------------------------------------------------------------------------
# NLE worker subprocess
# ---------------------------------------------------------------------------

def _nle_worker(seed: int, n_steps: int, conn):
    """One NLE env, driven over a Connection.

    Protocol:
      send ('reset',)        -> conn.send(obs)
      send ('step', action)  -> conn.send((obs, done))
      send ('close',)        -> exit
    """
    try:
        from nle.env import NLE
        env = NLE(character="rog-hum-cha-mal", max_episode_steps=n_steps + 10,
                  savedir=None)
        env.seed(core=seed, disp=seed, reseed=False)
        while True:
            msg = conn.recv()
            if not msg:
                break
            kind = msg[0]
            if kind == "reset":
                obs = env.reset()
                conn.send({k: np.asarray(v) for k, v in obs.items()})
            elif kind == "step":
                _, action = msg
                step = env.step(int(action))
                if len(step) == 4:
                    obs, _, done, _ = step
                else:
                    obs, _, _, done, _ = step
                conn.send(({k: np.asarray(v) for k, v in obs.items()}, bool(done)))
            elif kind == "close":
                env.close()
                return
            else:
                conn.send(("error", f"unknown msg {kind}"))
    except Exception as e:
        try:
            conn.send(("error", repr(e)))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Diff helpers (channel-wise; per-seed reporting)
# ---------------------------------------------------------------------------

NETHAX_ONLY_OK = {"internal", "misc", "program_state"}


def _diff_batched(nle_per_seed: list[dict], nax_batched: dict, step_idx: int):
    """Return per-seed list of (seed_idx, diff_strings).

    nle_per_seed[i] is the NLE obs dict for seed i; nax_batched[k] has a
    leading [B] axis.
    """
    n = len(nle_per_seed)
    per_seed_diffs: list[list[str]] = [[] for _ in range(n)]

    # Channel-set check against seed 0 (NLE keys are stable across seeds).
    nle_keys = set(nle_per_seed[0].keys())
    nax_keys = set(nax_batched.keys())
    missing_in_nax = nle_keys - nax_keys
    missing_in_nle = (nax_keys - nle_keys) - NETHAX_ONLY_OK
    shared = sorted(nle_keys & nax_keys)

    if missing_in_nax:
        for i in range(n):
            per_seed_diffs[i].append(
                f"[step {step_idx}] Nethax MISSING channels: "
                f"{sorted(missing_in_nax)}")
    if missing_in_nle:
        for i in range(n):
            per_seed_diffs[i].append(
                f"[step {step_idx}] NLE missing channels (Nethax-only): "
                f"{sorted(missing_in_nle)}")

    for key in shared:
        nax_arr = np.asarray(nax_batched[key])
        # NLE per-seed arrays — assume all same shape; check shape on seed 0.
        nle0 = np.asarray(nle_per_seed[0][key])
        if nax_arr.shape[0] != n or nax_arr.shape[1:] != nle0.shape:
            for i in range(n):
                per_seed_diffs[i].append(
                    f"[step {step_idx}] {key}: SHAPE mismatch "
                    f"NLE={nle0.shape} Nethax[i]={nax_arr.shape[1:]}")
            continue

        try:
            nax_int = nax_arr.astype(np.int64)
        except (TypeError, ValueError):
            # Fall back to byte compare.
            for i in range(n):
                nle_b = np.asarray(nle_per_seed[i][key]).tobytes()
                nax_b = np.asarray(nax_arr[i]).tobytes()
                if nle_b != nax_b:
                    per_seed_diffs[i].append(
                        f"[step {step_idx}] {key}: byte mismatch")
            continue

        for i in range(n):
            nle_arr = np.asarray(nle_per_seed[i][key]).astype(np.int64)
            mask = nle_arr != nax_int[i]
            nm = int(mask.sum())
            if nm > 0:
                total = int(nle_arr.size)
                pct = 100.0 * nm / total
                flat = np.argwhere(mask.ravel())[:3].ravel().tolist()
                samples = "; ".join(
                    f"@{j}: NLE={int(nle_arr.ravel()[j])} "
                    f"Nethax={int(nax_int[i].ravel()[j])}"
                    for j in flat)
                per_seed_diffs[i].append(
                    f"[step {step_idx}] {key}: {nm}/{total} ({pct:.1f}%) "
                    f"bytes diverge ({samples})")

    return per_seed_diffs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(n_seeds: int, n_steps: int, n_workers: int | None = None):
    n_workers = n_workers or min(n_seeds, os.cpu_count() or 1)

    print(f"[parallel_byteparity] seeds=0..{n_seeds - 1}  steps={n_steps}  "
          f"workers={n_workers}")

    # Spawn NLE workers (one per seed) -------------------------------------
    ctx = mp.get_context("spawn")
    procs: list = []
    pipes: list = []
    t0 = time.time()
    for seed in range(n_seeds):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_nle_worker, args=(seed, n_steps, child_conn),
                        daemon=True)
        p.start()
        procs.append(p)
        pipes.append(parent_conn)
    print(f"[parallel_byteparity] spawned {n_seeds} NLE workers in "
          f"{time.time() - t0:.1f}s")

    # NLE reset (parallel) -------------------------------------------------
    t0 = time.time()
    for c in pipes:
        c.send(("reset",))
    nle_obs_per_seed = [c.recv() for c in pipes]
    if any(isinstance(o, tuple) and o and o[0] == "error" for o in nle_obs_per_seed):
        for o in nle_obs_per_seed:
            if isinstance(o, tuple) and o and o[0] == "error":
                print(f"[abort] NLE worker error: {o[1]}")
        return -1
    print(f"[parallel_byteparity] NLE reset done in {time.time() - t0:.1f}s")

    # Nethax setup ---------------------------------------------------------
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.parity_mode import set_parity_mode, ParityMode
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.constants.races import Race
    set_parity_mode(ParityMode.NLE_BYTEPARITY)

    env = NethaxEnv()
    rngs = jnp.stack([jax.random.PRNGKey(seed) for seed in range(n_seeds)])

    t0 = time.time()
    print(f"[parallel_byteparity] Nethax reset_batched starting")
    nax_states, nax_obs = env.reset_batched(rngs, role=Role.ROGUE,
                                            race=Race.HUMAN, alignment=2)
    # Materialise to force compile.
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), nax_states)
    print(f"[parallel_byteparity] Nethax reset_batched done in "
          f"{time.time() - t0:.1f}s")

    # Diff step 0 (after reset) -------------------------------------------
    nax_dict = {k: np.asarray(v) for k, v in nax_obs.items()}
    per_seed_diffs0 = _diff_batched(nle_obs_per_seed, nax_dict, step_idx=0)
    totals = [len(d) for d in per_seed_diffs0]

    # Per-step rollout ----------------------------------------------------
    action = 0
    done_mask = np.zeros((n_seeds,), dtype=bool)

    for step_idx in range(1, n_steps + 1):
        # Send step to all NLE workers in parallel.
        for i, c in enumerate(pipes):
            if not done_mask[i]:
                c.send(("step", action))
        # Receive while next step is in flight (later).
        nle_results = [None] * n_seeds
        for i, c in enumerate(pipes):
            if not done_mask[i]:
                msg = c.recv()
                if isinstance(msg, tuple) and msg and msg[0] == "error":
                    print(f"[abort] NLE worker seed={i}: {msg[1]}")
                    return -1
                obs, done = msg
                nle_results[i] = obs
                if done:
                    done_mask[i] = True

        # Nethax: one vmap'd step.
        # Generate per-seed step rngs.
        step_rngs = jnp.stack([jax.random.PRNGKey(seed + step_idx)
                               for seed in range(n_seeds)])
        t_nx0 = time.time()
        nax_states, nax_obs, _, nax_done = env.step_batched_static(
            nax_states, action, step_rngs)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), nax_states)
        nx_dt = time.time() - t_nx0

        nax_dict = {k: np.asarray(v) for k, v in nax_obs.items()}
        # Build a per-seed list aligned with NLE; for done seeds, skip diff
        # (use last available obs).  For simplicity diff all live seeds.
        nle_align = []
        for i in range(n_seeds):
            if nle_results[i] is None:
                nle_align.append(nle_obs_per_seed[i])  # last known obs
            else:
                nle_align.append(nle_results[i])
                nle_obs_per_seed[i] = nle_results[i]

        per_seed_diffs = _diff_batched(nle_align, nax_dict, step_idx)
        for i, ds in enumerate(per_seed_diffs):
            totals[i] += len(ds)

        n_passing = sum(1 for t in totals if t == 0)
        print(f"  step {step_idx:>4}: nax_step={nx_dt*1000:.1f}ms  "
              f"passing={n_passing}/{n_seeds}  "
              f"done={int(done_mask.sum())}/{n_seeds}")

        if done_mask.all():
            print(f"[parallel_byteparity] all seeds done at step {step_idx}")
            break

    # Shutdown workers ----------------------------------------------------
    for c in pipes:
        try:
            c.send(("close",))
        except Exception:
            pass
    for p in procs:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()

    # Summary -------------------------------------------------------------
    n_pass = sum(1 for t in totals if t == 0)
    print()
    print(f"[summary] {n_pass}/{n_seeds} seeds byte-parity PASS")
    for i, t in enumerate(totals):
        if t != 0:
            print(f"  seed={i}: {t} total divergences")
    return 0 if n_pass == n_seeds else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    n_seeds = int(args[0]) if len(args) > 0 else 4
    n_steps = int(args[1]) if len(args) > 1 else 5
    n_workers = int(args[2]) if len(args) > 2 else None
    sys.exit(run(n_seeds, n_steps, n_workers))
