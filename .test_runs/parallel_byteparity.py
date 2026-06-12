"""Parallel multi-seed byte-parity validator.

Architecture:
  * NLE side: ``multiprocessing.Pool`` of ``n_workers`` subprocesses, one
    NLE instance per worker (NLE uses C globals, can't be in-process).
    Workers receive ``(seed, action)`` over a pipe and return the obs
    dict for that step.
  * Nethax side: ``env.reset_batched`` then ``env.step_batched`` with
    a length-B int32 action array (broadcast 0).  All ``n_batch``
    advance in lock-step on the device in one vmap'd graph.
  * Per step: gather the per-worker NLE obs dicts, stack them into a
    batched dict, diff against the Nethax batched obs.

Batching across n_seeds:
  When ``n_seeds > n_workers``, we process seeds in successive batches
  of size ``n_workers``.  Each batch spawns its own pool, runs the
  rollout, then tears down before the next batch starts.  This keeps
  the open-file-descriptor + RAM footprint bounded regardless of
  total seed count.

Done-seed handling:
  We track a per-seed done mask.  Once a seed terminates on the NLE
  side, we stop sending it steps and SKIP the per-step diff for that
  seed (otherwise the running Nethax vmap'd state would diverge from
  the stale last-known NLE obs and produce spurious mismatches).  We
  also check that Nethax's ``done`` flag agrees with NLE's at the
  terminal step — disagreement is counted as a divergence.

Step-RNG construction:
  ``step_rngs[i] = jax.random.PRNGKey(seed_i + step_idx)`` — byte-for-
  byte the same construction as the SERIAL validator's
  ``run_validator`` in tests/test_nle_byte_parity.py:228.

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
# Persistent cache opt-in (was net-negative — write stalls on cluster NFS).
# Set NETHAX_USE_CACHE=1 to enable. Honors JAX_COMPILATION_CACHE_DIR if
# already set (cluster runs override to node-local scratch).
_USE_CACHE = os.environ.get("NETHAX_USE_CACHE") == "1"
if _USE_CACHE:
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_REPO, ".jax_cache"))
    os.environ.setdefault("JAX_EXPLAIN_CACHE_MISSES", "1")
    _CACHE_DIR = os.environ["JAX_COMPILATION_CACHE_DIR"]
    _pre = len(os.listdir(_CACHE_DIR)) if os.path.isdir(_CACHE_DIR) else 0
    print(f"[cache] dir={_CACHE_DIR} entries_before={_pre}", flush=True)
if os.environ.get("NETHAX_EAGER") == "1":
    import jax
    jax.config.update("jax_disable_jit", True)


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


def _diff_batched(
    nle_per_seed: list[dict | None],
    nax_batched: dict,
    step_idx: int,
    alive_mask: np.ndarray | None = None,
):
    """Return per-seed list of diff strings.

    Parameters
    ----------
    nle_per_seed : length-B list of NLE obs dicts (or None for seeds
        that have terminated and were not stepped this round).
    nax_batched : dict of Nethax batched obs (leading [B] axis).
    step_idx : current step index for diff message labels.
    alive_mask : optional boolean [B] array.  When False at index i,
        the diff for seed i is skipped (returns empty list).  This is
        the canonical way to suppress spurious post-terminal mismatches
        for seeds that have already died.
    """
    n = len(nle_per_seed)
    per_seed_diffs: list[list[str]] = [[] for _ in range(n)]
    if alive_mask is None:
        alive_mask = np.ones((n,), dtype=bool)

    alive_idx = [i for i in range(n) if alive_mask[i] and nle_per_seed[i] is not None]
    if not alive_idx:
        return per_seed_diffs

    # Channel-set check against the first alive seed.
    ref = nle_per_seed[alive_idx[0]]
    nle_keys = set(ref.keys())
    nax_keys = set(nax_batched.keys())
    missing_in_nax = nle_keys - nax_keys
    missing_in_nle = (nax_keys - nle_keys) - NETHAX_ONLY_OK

    if missing_in_nax:
        for i in alive_idx:
            per_seed_diffs[i].append(
                f"[step {step_idx}] Nethax MISSING channels: "
                f"{sorted(missing_in_nax)}")
    if missing_in_nle:
        for i in alive_idx:
            per_seed_diffs[i].append(
                f"[step {step_idx}] NLE missing channels (Nethax-only): "
                f"{sorted(missing_in_nle)}")

    shared = sorted(nle_keys & nax_keys)
    for key in shared:
        nax_arr = np.asarray(nax_batched[key])
        nle0 = np.asarray(ref[key])
        if nax_arr.shape[0] != n or nax_arr.shape[1:] != nle0.shape:
            for i in alive_idx:
                per_seed_diffs[i].append(
                    f"[step {step_idx}] {key}: SHAPE mismatch "
                    f"NLE={nle0.shape} Nethax[i]={nax_arr.shape[1:]}")
            continue

        try:
            nax_int = nax_arr.astype(np.int64)
        except (TypeError, ValueError):
            for i in alive_idx:
                nle_b = np.asarray(nle_per_seed[i][key]).tobytes()
                nax_b = np.asarray(nax_arr[i]).tobytes()
                if nle_b != nax_b:
                    per_seed_diffs[i].append(
                        f"[step {step_idx}] {key}: byte mismatch")
            continue

        for i in alive_idx:
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
# Single-batch driver — runs a contiguous slice of seeds in lock-step
# ---------------------------------------------------------------------------

def _run_batch(
    seed_offset: int,
    n_batch: int,
    n_steps: int,
    nethax_env,
    jax_mod,
    jnp_mod,
    role,
    race,
    alignment: int,
):
    """Run one batch of ``n_batch`` seeds [seed_offset .. seed_offset + n_batch).

    Returns per-seed (total_divergences, terminated_at_step) lists.
    """
    seeds = list(range(seed_offset, seed_offset + n_batch))
    print(f"[batch] seeds={seeds[0]}..{seeds[-1]} (n={n_batch} steps={n_steps})")

    # Spawn NLE workers — one process per seed in this batch.
    ctx = mp.get_context("spawn")
    procs: list = []
    pipes: list = []
    t0 = time.time()
    for s in seeds:
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_nle_worker, args=(s, n_steps, child_conn),
                        daemon=True)
        p.start()
        procs.append(p)
        pipes.append(parent_conn)
    print(f"[batch] spawned {n_batch} NLE workers in "
          f"{time.time() - t0:.1f}s")

    # NLE reset (parallel).
    t0 = time.time()
    for c in pipes:
        c.send(("reset",))
    nle_obs_per_seed = [c.recv() for c in pipes]
    if any(isinstance(o, tuple) and o and o[0] == "error" for o in nle_obs_per_seed):
        for o in nle_obs_per_seed:
            if isinstance(o, tuple) and o and o[0] == "error":
                print(f"[abort] NLE worker error: {o[1]}")
        _shutdown(procs, pipes)
        return None
    print(f"[batch] NLE reset done in {time.time() - t0:.1f}s")

    # Nethax reset_batched.
    rngs = jnp_mod.stack([jax_mod.random.PRNGKey(s) for s in seeds])
    t0 = time.time()
    nax_states, nax_obs = nethax_env.reset_batched(
        rngs, role=role, race=race, alignment=alignment)
    jax_mod.tree_util.tree_map(lambda x: x.block_until_ready(), nax_states)
    print(f"[batch] Nethax reset_batched done in {time.time() - t0:.1f}s")

    # Step 0 diff (post-reset).
    nax_dict = {k: np.asarray(v) for k, v in nax_obs.items()}
    per_seed_diffs0 = _diff_batched(nle_obs_per_seed, nax_dict, step_idx=0)
    totals = [len(d) for d in per_seed_diffs0]
    term_step = [-1] * n_batch  # -1 = never terminated

    # Rollout.
    action = 0
    action_batched = jnp_mod.zeros((n_batch,), dtype=jnp_mod.int32)
    done_mask = np.zeros((n_batch,), dtype=bool)

    for step_idx in range(1, n_steps + 1):
        alive_before = ~done_mask

        # Send step to alive NLE workers.
        for i, c in enumerate(pipes):
            if alive_before[i]:
                c.send(("step", action))
        nle_results: list[dict | None] = [None] * n_batch
        new_nle_done = np.zeros((n_batch,), dtype=bool)
        for i, c in enumerate(pipes):
            if alive_before[i]:
                msg = c.recv()
                if isinstance(msg, tuple) and msg and msg[0] == "error":
                    print(f"[abort] NLE worker seed={seeds[i]}: {msg[1]}")
                    _shutdown(procs, pipes)
                    return None
                obs, done = msg
                nle_results[i] = obs
                new_nle_done[i] = done

        # Nethax: one vmap'd step (done envs are no-op internally).
        # Step RNG construction matches tests/test_nle_byte_parity.py:228
        # (run_validator) exactly: PRNGKey(seed + step_idx) per env.
        step_rngs = jnp_mod.stack([jax_mod.random.PRNGKey(s + step_idx)
                                   for s in seeds])
        print(f"[step {step_idx}] calling step_batched...", flush=True)
        t_nx0 = time.time()
        nax_states, nax_obs, _, nax_done = nethax_env.step_batched(
            nax_states, action_batched, step_rngs, static_action=action)
        t_call = time.time() - t_nx0
        print(f"[step {step_idx}] step_batched returned in {t_call:.1f}s "
              f"(includes trace+compile+dispatch, NOT block)", flush=True)
        jax_mod.tree_util.tree_map(lambda x: x.block_until_ready(), nax_states)
        nx_dt = time.time() - t_nx0
        print(f"[step {step_idx}] block_until_ready done, total {nx_dt:.1f}s "
              f"(GPU exec = {nx_dt - t_call:.1f}s)", flush=True)

        nax_done_np = np.asarray(nax_done)
        nax_dict = {k: np.asarray(v) for k, v in nax_obs.items()}

        # Per-seed diff — only for envs alive at the START of this step.
        # (Newly-terminating seeds still get their terminal-step diff;
        # seeds already done before this step are skipped entirely.)
        per_seed_diffs = _diff_batched(
            nle_results, nax_dict, step_idx, alive_mask=alive_before)

        for i in range(n_batch):
            if alive_before[i]:
                totals[i] += len(per_seed_diffs[i])
                # Verify done agreement at the terminal step.
                if new_nle_done[i] != bool(nax_done_np[i]):
                    side_nle = "DONE" if new_nle_done[i] else "alive"
                    side_nax = "DONE" if nax_done_np[i] else "alive"
                    print(f"  [seed {seeds[i]}] done-flag disagreement at "
                          f"step {step_idx}: NLE={side_nle} Nethax={side_nax}")
                    totals[i] += 1
                # Update done mask for next step.
                if new_nle_done[i] or bool(nax_done_np[i]):
                    done_mask[i] = True
                    term_step[i] = step_idx

        alive_after = int((~done_mask).sum())
        n_passing = sum(1 for t in totals if t == 0)
        print(f"  step {step_idx:>4}: nax_step={nx_dt*1000:.1f}ms  "
              f"passing={n_passing}/{n_batch}  "
              f"alive={alive_after}/{n_batch}")

        if done_mask.all():
            print(f"[batch] all seeds done at step {step_idx}")
            break

    _shutdown(procs, pipes)
    return totals, term_step


def _shutdown(procs, pipes):
    for c in pipes:
        try:
            c.send(("close",))
        except Exception:
            pass
    for p in procs:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()


# ---------------------------------------------------------------------------
# Outer driver — batches across n_seeds
# ---------------------------------------------------------------------------

def run(n_seeds: int, n_steps: int, n_workers: int | None = None):
    """Run byte-parity validation over [0, n_seeds), n_steps steps per seed.

    ``n_workers`` caps the concurrent NLE process count.  When
    ``n_seeds > n_workers``, seeds are processed in successive batches
    of size ``n_workers``.  This keeps open-file-descriptor and RAM
    bounded regardless of n_seeds.
    """
    n_workers = n_workers or min(n_seeds, os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, n_seeds))

    print(f"[parallel_byteparity] seeds=0..{n_seeds - 1}  steps={n_steps}  "
          f"workers={n_workers}  batches={(n_seeds + n_workers - 1) // n_workers}")

    # Nethax / JAX setup — once, reused across batches so the JIT cache
    # warms after the first batch.
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.parity_mode import set_parity_mode, ParityMode
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.constants.races import Race
    set_parity_mode(ParityMode.NLE_BYTEPARITY)

    nethax_env = NethaxEnv()

    all_totals: list[int] = []
    all_term: list[int] = []
    t_total = time.time()
    for batch_start in range(0, n_seeds, n_workers):
        batch_size = min(n_workers, n_seeds - batch_start)
        result = _run_batch(
            seed_offset=batch_start,
            n_batch=batch_size,
            n_steps=n_steps,
            nethax_env=nethax_env,
            jax_mod=jax,
            jnp_mod=jnp,
            role=Role.ROGUE,
            race=Race.HUMAN,
            alignment=2,  # chaotic
        )
        if result is None:
            return -1
        totals, term_step = result
        all_totals.extend(totals)
        all_term.extend(term_step)
    print(f"[parallel_byteparity] total elapsed {time.time() - t_total:.1f}s")

    # Summary.
    n_pass = sum(1 for t in all_totals if t == 0)
    print()
    print(f"[summary] {n_pass}/{n_seeds} seeds byte-parity PASS")
    for i, t in enumerate(all_totals):
        if t != 0:
            print(f"  seed={i}: {t} total divergences "
                  f"(terminated at step {all_term[i]})")
    return 0 if n_pass == n_seeds else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    n_seeds = int(args[0]) if len(args) > 0 else 4
    n_steps = int(args[1]) if len(args) > 1 else 5
    n_workers = int(args[2]) if len(args) > 2 else None
    rc = run(n_seeds, n_steps, n_workers)
    if _USE_CACHE and os.path.isdir(_CACHE_DIR):
        _post = len(os.listdir(_CACHE_DIR))
        print(f"[cache] entries_after={_post} delta={_post - _pre}", flush=True)
    sys.exit(rc)
