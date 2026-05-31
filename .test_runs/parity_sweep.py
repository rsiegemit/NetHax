"""Many-seed × many-step lock-step parity sweep between NLE and Nethax.

For each seed, reset both envs with the same seed, then step in lock-step
with the same action sequence (seeded-random direction).  Compare obs +
reward + done + info at every step.  On first divergence, log and abort
that seed (no point continuing once streams diverge).

Aggregates: per-seed result (full match / first-divergence-step), top
diverging channels, reward delta stats.

CLI:
    parity_sweep.py [num_seeds] [num_steps] [--action-mode=<scripted|random>]

Defaults: 10 seeds × 100 steps, action-mode=random.
"""
import os, sys, time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/Users/rsiegelmann/Downloads/Projects/nethax")

import numpy as np
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from nle.env import NLE  # noqa: E402

from Nethax.nethax.parity_mode import set_parity_mode, ParityMode  # noqa: E402
set_parity_mode(ParityMode.NLE_BYTEPARITY)

from Nethax.nethax.env import NethaxEnv  # noqa: E402
from Nethax.nethax.constants.roles import Role  # noqa: E402
from Nethax.nethax.constants.races import Race  # noqa: E402


def _nax_to_dict(obs):
    if isinstance(obs, dict):
        return {k: np.asarray(v) for k, v in obs.items()}
    return {}


def _compare_obs(nle_obs, nax_obs):
    """Return list of (channel, bytes_diff) tuples — empty = full match."""
    diffs = []
    shared = set(nle_obs.keys()) & set(nax_obs.keys())
    for key in sorted(shared):
        a = np.asarray(nle_obs[key])
        b = np.asarray(nax_obs[key])
        if a.shape != b.shape:
            diffs.append((key, -1))   # shape mismatch
            continue
        bad = int(np.sum(a != b))
        if bad > 0:
            diffs.append((key, bad))
    return diffs


def run_one_seed(seed: int, num_steps: int, action_rng: np.random.RandomState):
    """Run a single seed, lock-step, until first divergence or num_steps.

    Returns a dict:
        seed, success, first_div_step, divs, total_reward_nle,
        total_reward_nax, info_mismatches, steps_completed
    """
    # NLE
    nle_env = NLE(
        character="rog-hum-cha-mal", max_episode_steps=10_000, savedir=None
    )
    nle_env.seed(core=seed, disp=seed, reseed=False)
    nle_obs = nle_env.reset()

    # Nethax
    nax_env = NethaxEnv()
    nax_state, nax_obs = nax_env.reset(
        jax.random.PRNGKey(seed),
        role=Role.ROGUE, race=Race.HUMAN, alignment=2,
    )

    nax_obs_d = _nax_to_dict(nax_obs)
    diffs = _compare_obs(nle_obs, nax_obs_d)
    if diffs:
        nle_env.close()
        return {
            "seed": seed, "success": False, "first_div_step": 0,
            "divs": diffs, "total_reward_nle": 0.0, "total_reward_nax": 0.0,
            "info_mismatches": 0, "steps_completed": 0,
        }

    total_nle = 0.0
    total_nax = 0.0
    info_mismatches = 0
    first_div_step = None
    final_div = []

    for step in range(1, num_steps + 1):
        # Action: 0..7 = 8-direction moves
        action = int(action_rng.randint(0, 8))

        try:
            nle_step = nle_env.step(action)
            if len(nle_step) == 4:
                nle_obs, nle_r, nle_done, nle_info = nle_step
            else:
                nle_obs, nle_r, _, nle_done, nle_info = nle_step
        except Exception as e:
            nle_env.close()
            return {
                "seed": seed, "success": False, "first_div_step": step,
                "divs": [("nle_step_error", str(e))],
                "total_reward_nle": total_nle, "total_reward_nax": total_nax,
                "info_mismatches": info_mismatches, "steps_completed": step - 1,
            }

        try:
            nax_state, nax_obs, nax_r, nax_done, nax_info = nax_env.step(
                nax_state, jnp.int32(action), jax.random.PRNGKey(seed * 1000 + step)
            )
        except Exception as e:
            nle_env.close()
            return {
                "seed": seed, "success": False, "first_div_step": step,
                "divs": [("nax_step_error", str(e))],
                "total_reward_nle": total_nle, "total_reward_nax": total_nax,
                "info_mismatches": info_mismatches, "steps_completed": step - 1,
            }

        nle_r = float(nle_r); nax_r = float(nax_r)
        nle_done = bool(nle_done); nax_done = bool(nax_done)
        total_nle += nle_r
        total_nax += nax_r

        # Compare obs
        nax_obs_d = _nax_to_dict(nax_obs)
        obs_diffs = _compare_obs(nle_obs, nax_obs_d)

        # Compare reward
        r_diff = abs(nle_r - nax_r) > 1e-6
        # Compare done
        d_diff = nle_done != nax_done

        # Compile per-step diffs
        step_divs = list(obs_diffs)
        if r_diff:
            step_divs.append(("reward", f"NLE={nle_r:.2f} vs Nax={nax_r:.2f}"))
        if d_diff:
            step_divs.append(("done", f"NLE={nle_done} vs Nax={nax_done}"))

        if step_divs and first_div_step is None:
            first_div_step = step
            final_div = step_divs
            break

        if nle_done or nax_done:
            break

    nle_env.close()
    success = first_div_step is None
    return {
        "seed": seed, "success": success,
        "first_div_step": first_div_step if not success else None,
        "divs": final_div, "total_reward_nle": total_nle,
        "total_reward_nax": total_nax, "info_mismatches": info_mismatches,
        "steps_completed": step,
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    num_seeds = int(args[0]) if args else 10
    num_steps = int(args[1]) if len(args) > 1 else 100

    print(
        f"\n[parity_sweep] num_seeds={num_seeds} num_steps={num_steps}\n"
        f"  comparing: obs (all channels) + reward + done\n"
    )

    results = []
    t_start = time.time()
    full_match = 0
    div_step_hist = []
    channel_freq = {}

    for seed in range(num_seeds):
        # Reproducible action rng per seed.
        action_rng = np.random.RandomState(seed * 12345 + 1)
        res = run_one_seed(seed, num_steps, action_rng)
        results.append(res)
        if res["success"]:
            full_match += 1
        else:
            div_step_hist.append(res["first_div_step"])
            for ch, _ in res["divs"]:
                channel_freq[ch] = channel_freq.get(ch, 0) + 1

        elapsed = time.time() - t_start
        status = "MATCH" if res["success"] else f"DIVERGE @ step {res['first_div_step']}"
        print(
            f"  seed={seed:>4} steps={res['steps_completed']:>4} "
            f"R_nle={res['total_reward_nle']:>8.1f} R_nax={res['total_reward_nax']:>8.1f}  "
            f"{status}  [{elapsed:.0f}s]"
        )
        if not res["success"]:
            print(f"           divs: {res['divs'][:5]}")

    elapsed = time.time() - t_start
    print(f"\n[parity_sweep] DONE in {elapsed:.0f}s")
    print(f"  full-match seeds: {full_match} / {num_seeds} "
          f"({100*full_match/num_seeds:.1f}%)")
    if div_step_hist:
        print(f"  divergence step stats: min={min(div_step_hist)} "
              f"max={max(div_step_hist)} "
              f"mean={sum(div_step_hist)/len(div_step_hist):.1f}")
    if channel_freq:
        print(f"  top diverging channels:")
        for ch, n in sorted(channel_freq.items(), key=lambda x: -x[1])[:10]:
            print(f"    {ch}: {n} seeds")

    sys.exit(0 if full_match == num_seeds else 1)


if __name__ == "__main__":
    main()
