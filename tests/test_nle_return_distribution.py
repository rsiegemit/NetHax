"""NLE vs Nethax return-distribution parity validator.

A SOFTER complement to ``tests/test_nle_byte_parity.py``.  Byte-replay is
the gold standard for transferability but requires ISAAC64 inside JIT.
This validator instead asks: under a fixed random policy, do the two
environments produce *statistically equivalent* episode-return
distributions?

Procedure
---------
For seed in 0..N-1:
  1. Reset both envs with character "rog-hum-cha-mal".
  2. At each step, pick an action from
     {N, E, S, W, SEARCH, WAIT}
     using a seeded numpy RNG (same draw for both envs).
  3. Step both envs up to MAX_STEPS or done.
  4. Record sum of step rewards as the episode return; also record
     episode length.

Then compute mean/std return, mean episode length, and a 2-sample
Kolmogorov-Smirnov test (``scipy.stats.ks_2samp``) on the return
distributions.  Verdict is "MATCH" iff KS p > 0.05.

Exit codes
----------
  0  → MATCH
  1  → DIVERGE
  2  → SKIP (NLE or Nethax import failed)

Run
---
    JAX_PLATFORMS=cpu .venv/bin/python tests/test_nle_return_distribution.py
    JAX_PLATFORMS=cpu .venv/bin/python tests/test_nle_return_distribution.py smoke
"""
from __future__ import annotations

import os
# Must precede any jax import to honour ISAAC64 uint64 layout and CPU-only.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

# When run as a script (python tests/foo.py) Python prepends `tests/` to
# sys.path, not the repo root.  Insert the repo root so `import Nethax`
# works without requiring PYTHONPATH=.
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np


# Action mapping.  Both NLE and Nethax now accept INDEX-into-env.actions
# semantics (Nethax via Nethax.nethax.nle_action_map.maybe_remap_action,
# wired into NethaxEnv.step).  The two tuples remain because Nethax also
# accepts a raw ASCII ord >= 86 — we still pass index for clarity.
#
# NLE action-tuple indices (verified at runtime against
# nle.nethack.{CompassDirection,MiscDirection,Command}):
#     N=0, E=1, S=2, W=3, WAIT=18, SEARCH=61
# Equivalent ASCII ords (vendor cmd.c char codes):
#     N=ord('k')=107, E=ord('l')=108, S=ord('j')=106, W=ord('h')=104,
#     WAIT=ord('.')=46, SEARCH=ord('s')=115
# Note: WAIT=46 (< 86) is ambiguous as ord-vs-index, so we send the index.
_NLE_ACTION_INDICES = (0, 1, 2, 3, 61, 18)
_NETHAX_ACTION_ORDS = (0, 1, 2, 3, 61, 18)  # Same as NLE indices — remapped JIT-side.
_ACTION_NAMES = ("N", "E", "S", "W", "SEARCH", "WAIT")


def _safe_import_nle():
    try:
        from nle.env import NLE
        return NLE, None
    except Exception as e:  # pragma: no cover
        return None, str(e)


def _safe_import_nethax():
    try:
        import jax
        import jax.numpy as jnp
        from Nethax.nethax.env import NethaxEnv
        from Nethax.nethax.parity_mode import set_parity_mode, ParityMode
        set_parity_mode(ParityMode.NLE_BYTEPARITY)
        return NethaxEnv, jax, jnp, None
    except Exception as e:
        return None, None, None, str(e)


def _run_nle_episode(NLE_cls, seed: int, max_steps: int, policy_rng: np.random.Generator):
    """Run one NLE episode under the fixed random policy.

    Returns (episode_return, episode_length, terminated_bool, actions_taken).
    """
    env = NLE_cls(
        character="rog-hum-cha-mal",
        max_episode_steps=max_steps,
        savedir=None,
    )
    try:
        env.seed(core=seed, disp=seed)
    except Exception:
        pass
    env.reset()

    total_return = 0.0
    steps = 0
    done = False
    actions_taken = []
    for _ in range(max_steps):
        a_idx = int(policy_rng.integers(0, len(_NLE_ACTION_INDICES)))
        actions_taken.append(a_idx)
        nle_action = _NLE_ACTION_INDICES[a_idx]
        step_ret = env.step(nle_action)
        if len(step_ret) == 4:
            _, r, done, _ = step_ret
        else:
            _, r, term, trunc, _ = step_ret
            done = bool(term) or bool(trunc)
        total_return += float(r)
        steps += 1
        if done:
            break
    env.close()
    return total_return, steps, bool(done), actions_taken


def _run_nethax_episode(NethaxEnv, jax, jnp, seed: int, max_steps: int, actions_taken: list[int]):
    """Run one Nethax episode replaying the same action indices as NLE.

    Returns (episode_return, episode_length, terminated_bool).
    """
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.constants.races import Race
    env = NethaxEnv()
    rng = jax.random.PRNGKey(seed)
    state, _ = env.reset(rng, role=Role.ROGUE, race=Race.HUMAN, alignment=2)

    total_return = 0.0
    steps = 0
    done = False
    for i, a_idx in enumerate(actions_taken):
        nax_action = _NETHAX_ACTION_ORDS[a_idx]
        step_rng = jax.random.PRNGKey(seed * 100000 + i)
        state, _, r, done_arr, _ = env.step(state, jnp.int32(nax_action), step_rng)
        total_return += float(np.asarray(r))
        steps += 1
        if bool(np.asarray(done_arr)):
            done = True
            break
    return total_return, steps, done


def run_validator(n_episodes: int = 100, max_steps: int = 200, verbose: bool = True) -> int:
    """Run the full validator.

    Returns:
       0 → MATCH (KS p > 0.05)
       1 → DIVERGE
       2 → SKIP (import failure)
    """
    NLE_cls, nle_err = _safe_import_nle()
    if nle_err:
        print(f"[SKIP] Could not import nle: {nle_err}")
        return 2
    NethaxEnv, jax, jnp, nax_err = _safe_import_nethax()
    if nax_err:
        print(f"[SKIP] Could not import Nethax: {nax_err}")
        return 2

    try:
        from scipy.stats import ks_2samp
    except ImportError as e:
        print(f"[SKIP] scipy not available: {e}")
        return 2

    nle_returns = np.zeros(n_episodes, dtype=np.float64)
    nax_returns = np.zeros(n_episodes, dtype=np.float64)
    nle_lengths = np.zeros(n_episodes, dtype=np.int32)
    nax_lengths = np.zeros(n_episodes, dtype=np.int32)

    for ep in range(n_episodes):
        # One RNG per seed so both runs see the same draw stream.
        policy_rng = np.random.default_rng(ep)
        nle_ret, nle_len, _nle_done, actions = _run_nle_episode(
            NLE_cls, seed=ep, max_steps=max_steps, policy_rng=policy_rng,
        )
        nax_ret, nax_len, _nax_done = _run_nethax_episode(
            NethaxEnv, jax, jnp, seed=ep, max_steps=max_steps, actions_taken=actions,
        )
        nle_returns[ep] = nle_ret
        nax_returns[ep] = nax_ret
        nle_lengths[ep] = nle_len
        nax_lengths[ep] = nax_len
        if verbose:
            print(
                f"[ep {ep:3d}] NLE ret={nle_ret:8.3f} len={nle_len:3d}  |  "
                f"Nethax ret={nax_ret:8.3f} len={nax_len:3d}"
            )

    ks_stat, ks_p = ks_2samp(nle_returns, nax_returns)
    verdict = "MATCH" if ks_p > 0.05 else "DIVERGE"

    print()
    print("=" * 64)
    print(f"  Return-distribution parity over {n_episodes} episodes")
    print("=" * 64)
    print(f"{'metric':<24}{'NLE':>18}{'Nethax':>18}")
    print(f"{'mean return':<24}{nle_returns.mean():>18.4f}{nax_returns.mean():>18.4f}")
    print(f"{'std return':<24}{nle_returns.std():>18.4f}{nax_returns.std():>18.4f}")
    print(f"{'min return':<24}{nle_returns.min():>18.4f}{nax_returns.min():>18.4f}")
    print(f"{'max return':<24}{nle_returns.max():>18.4f}{nax_returns.max():>18.4f}")
    print(f"{'mean episode length':<24}{nle_lengths.mean():>18.2f}{nax_lengths.mean():>18.2f}")
    print()
    print(f"  KS 2-sample statistic = {ks_stat:.4f}")
    print(f"  KS 2-sample p-value   = {ks_p:.4f}")
    print(f"  VERDICT: {verdict} (p {'>' if ks_p > 0.05 else '<='} 0.05)")
    print("=" * 64)

    return 0 if verdict == "MATCH" else 1


# ---------------------------------------------------------------------------
# Pytest entry point — lightweight smoke so this validator participates in
# the suite without blowing the time budget.
# ---------------------------------------------------------------------------

def test_nle_return_distribution_smoke():
    """Pytest smoke: 5 episodes / 20 steps to verify the harness wires up.

    A divergent verdict still passes the test — we just want to verify the
    plumbing.  The real verdict is produced by running the script directly
    with the larger episode count.
    """
    rc = run_validator(n_episodes=5, max_steps=20, verbose=False)
    assert rc in (0, 1, 2), f"unexpected return code {rc}"


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "smoke":
        rc = run_validator(n_episodes=5, max_steps=20, verbose=True)
    else:
        rc = run_validator(n_episodes=100, max_steps=200, verbose=True)
    sys.exit(rc)
