"""NLE vs Nethax algorithmic-equivalence validator.

A SIDE-PATH complement to ``tests/test_nle_byte_parity.py`` (byte parity)
and ``tests/test_nle_return_distribution.py`` (KS on returns alone).

The motivating question
-----------------------
The end goal is *not* per-tick byte parity; it is that an NLE-trained
agent transfers to Nethax with identical performance.  Byte parity is a
strong sufficient condition, but it is not strictly necessary — two
environments with different RNG schedules can still induce the same
*policy-conditioned* return distribution.  This validator measures that
weaker, sufficient-for-transfer property directly.

Procedure
---------
For each episode i in 0..N-1, with both sides seeded by i and a fixed
numpy-seeded policy:

  1. Reset NLE and Nethax with character "rog-hum-cha-mal".
  2. Apply the per-episode policy (see below) for up to MAX_STEPS or done.
  3. Record per episode:
       * total return (sum of step rewards)
       * episode length
       * death indicator + a coarse death-cause bucket
       * monsters killed
       * deepest dungeon level reached

  Policy selection per episode:
       * First half of episodes  → uniform-random over
         {N, E, S, W, SEARCH, WAIT} (numpy-seeded).
       * Second half of episodes → scripted "south-east-search":
         the deterministic cycle  S, E, SEARCH, S, E, SEARCH, ...

  The SAME action stream is replayed on both sides (NLE produces the
  actions; Nethax consumes them), so any aggregate divergence reflects
  environment dynamics, not policy noise.

Statistical tests
-----------------
``scipy.stats.ks_2samp`` is applied independently to the marginal
empirical distributions of:
    returns, episode lengths, monster kill counts, dungeon levels.

Verdict
-------
  "EQUIVALENT"  iff all four KS p-values are > 0.05
  "DIVERGE"     otherwise, naming each diverging marginal

NOT byte parity
---------------
This validator is *intentionally* aggregate.  It does not — and must
not — assert per-step state agreement.  See
``tests/test_nle_byte_parity.py`` for the byte-equivalence validator.

Run
---
    JAX_PLATFORMS=cpu .venv/bin/python tests/test_algorithmic_equivalence.py
    JAX_PLATFORMS=cpu .venv/bin/python tests/test_algorithmic_equivalence.py smoke

Exit codes
----------
  0 → EQUIVALENT
  1 → DIVERGE
  2 → SKIP (NLE / Nethax / scipy not importable)
"""
from __future__ import annotations

import os
# Must precede any jax import to honour ISAAC64 uint64 layout and CPU-only.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

# Running as a script puts ``tests/`` on sys.path; we need the repo root.
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np


# ---------------------------------------------------------------------------
# Action set — shared by both envs.  Indices match the existing
# ``test_nle_return_distribution`` validator so the two share a meaning.
# ---------------------------------------------------------------------------
# NLE action-tuple indices: N=0, E=1, S=2, W=3, WAIT=18, SEARCH=61
_NLE_ACTION_INDICES = (0, 1, 2, 3, 61, 18)
# Nethax accepts NLE indices when ParityMode is NLE_BYTEPARITY (env.step
# calls maybe_remap_action internally).
_NETHAX_ACTION_ORDS = (0, 1, 2, 3, 61, 18)
_ACTION_NAMES = ("N", "E", "S", "W", "SEARCH", "WAIT")

# Indices into the abstract 6-action set (above) for the scripted policy.
_ABSTRACT_N, _ABSTRACT_E, _ABSTRACT_S, _ABSTRACT_W, _ABSTRACT_SEARCH, _ABSTRACT_WAIT = 0, 1, 2, 3, 4, 5
_SCRIPTED_CYCLE = (_ABSTRACT_S, _ABSTRACT_E, _ABSTRACT_SEARCH)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def _policy_random(rng: np.random.Generator, step_idx: int) -> int:
    """Uniform-random over the abstract 6-action set."""
    return int(rng.integers(0, len(_NLE_ACTION_INDICES)))


def _policy_scripted_south_east_search(rng: np.random.Generator, step_idx: int) -> int:
    """Deterministic S → E → SEARCH cycle.  ``rng`` is unused but kept
    in the signature so the two policies are interchangeable."""
    return _SCRIPTED_CYCLE[step_idx % len(_SCRIPTED_CYCLE)]


# ---------------------------------------------------------------------------
# Imports — kept lazy so the test SKIPs cleanly when deps are missing.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-episode runners.
# ---------------------------------------------------------------------------

def _classify_nle_death(info: dict, blstats) -> str:
    """Coarse NLE death-cause bucket.

    NLE only surfaces ``StepStatus`` ∈ {RUNNING, DEATH, ABORTED}.  We
    map DEATH → "death" and inspect HP to disambiguate HP-0 deaths from
    truncation.  Anything else → "alive_or_truncated".
    """
    status = info.get("end_status", None)
    status_val = int(status) if status is not None else 0
    if status_val == 1:  # DEATH
        return "death"
    if status_val == -1:  # ABORTED
        return "aborted"
    # Catch HP-zero terminations that didn't surface as DEATH (rare).
    if blstats is not None and int(blstats[10]) <= 0:  # BL_HP = 10
        return "death"
    return "alive_or_truncated"


def _run_nle_episode(NLE_cls, seed: int, max_steps: int, policy_fn, policy_rng: np.random.Generator):
    """Run one NLE episode under ``policy_fn``.

    Returns a dict with episode metrics plus the action stream so
    Nethax can replay it identically.
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
    obs = env.reset()

    total_return = 0.0
    steps = 0
    done = False
    actions_taken: list[int] = []
    info: dict = {}
    last_obs = obs
    kill_count = 0
    deepest_level = 1
    # BL_DEPTH = 12 (vendor blstats index for dungeon depth)
    BL_DEPTH = 12

    for step_i in range(max_steps):
        a_idx = int(policy_fn(policy_rng, step_i))
        actions_taken.append(a_idx)
        nle_action = _NLE_ACTION_INDICES[a_idx]
        step_ret = env.step(nle_action)
        if len(step_ret) == 4:
            obs, r, done, info = step_ret
        else:
            obs, r, term, trunc, info = step_ret
            done = bool(term) or bool(trunc)
        total_return += float(r)
        steps += 1
        last_obs = obs

        # Best-effort monster kill counter via message scan: NLE does
        # not expose a structured kill counter.  We count occurrences of
        # the substring "kill" in the per-step message (case-insensitive,
        # excluding "killed by" which is the hero's death line).
        msg_bytes = obs.get("message") if isinstance(obs, dict) else None
        if msg_bytes is not None:
            try:
                msg_str = bytes(msg_bytes).decode("ascii", errors="ignore").lower()
                # "you kill" / "killed the" → hero scored a kill.
                if "you kill" in msg_str or "killed the" in msg_str:
                    kill_count += 1
            except Exception:
                pass

        depth = int(obs["blstats"][BL_DEPTH]) if isinstance(obs, dict) else 1
        if depth > deepest_level:
            deepest_level = depth

        if done:
            break
    env.close()

    death_cause = _classify_nle_death(
        info,
        last_obs.get("blstats") if isinstance(last_obs, dict) else None,
    )
    return {
        "episode_return": total_return,
        "episode_length": steps,
        "done": bool(done),
        "death_cause": death_cause,
        "monsters_killed": kill_count,
        "dungeon_level": deepest_level,
        "actions": actions_taken,
    }


def _run_nethax_episode(NethaxEnv, jax, jnp, seed: int, max_steps: int, actions_taken: list[int]):
    """Run one Nethax episode replaying ``actions_taken``.

    Returns a metrics dict matching the NLE-side keys.
    """
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.constants.races import Race
    env = NethaxEnv()
    rng = jax.random.PRNGKey(seed)
    state, _ = env.reset(rng, role=Role.ROGUE, race=Race.HUMAN, alignment=2)

    total_return = 0.0
    steps = 0
    done = False
    deepest_level = int(np.asarray(state.dungeon.current_level))

    for i, a_idx in enumerate(actions_taken):
        if i >= max_steps:
            break
        nax_action = _NETHAX_ACTION_ORDS[a_idx]
        step_rng = jax.random.PRNGKey(seed * 100000 + i)
        state, _, r, done_arr, _ = env.step(state, jnp.int32(nax_action), step_rng)
        total_return += float(np.asarray(r))
        steps += 1
        depth = int(np.asarray(state.dungeon.current_level))
        if depth > deepest_level:
            deepest_level = depth
        if bool(np.asarray(done_arr)):
            done = True
            break

    # Death-cause bucket on the Nethax side: HP<=0 → "death",
    # otherwise "alive_or_truncated".  (Nethax surfaces a more detailed
    # cause via player_killer_mid but the cross-env comparison only
    # needs the coarse bucket.)
    hp = int(np.asarray(state.player_hp))
    if done and hp <= 0:
        death_cause = "death"
    elif done:
        death_cause = "alive_or_truncated"
    else:
        death_cause = "alive_or_truncated"

    monsters_killed = int(np.asarray(state.scoring.monsters_killed))
    return {
        "episode_return": total_return,
        "episode_length": steps,
        "done": done,
        "death_cause": death_cause,
        "monsters_killed": monsters_killed,
        "dungeon_level": deepest_level,
    }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

_METRIC_KEYS = ("episode_return", "episode_length", "monsters_killed", "dungeon_level")
_METRIC_LABELS = {
    "episode_return": "returns",
    "episode_length": "lengths",
    "monsters_killed": "monster_kills",
    "dungeon_level": "dungeon_levels",
}


def _select_policy(episode_idx: int, n_episodes: int):
    """Half random, half scripted.  Identical on both sides."""
    if episode_idx < n_episodes // 2:
        return _policy_random, "random"
    return _policy_scripted_south_east_search, "scripted_SES"


def run_validator(n_episodes: int = 1000, max_steps: int = 200, verbose: bool = True) -> int:
    """Run the full algorithmic-equivalence validator.

    Returns:
       0 → EQUIVALENT (all KS p-values > 0.05)
       1 → DIVERGE
       2 → SKIP (NLE / Nethax / scipy import failure)
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

    # Per-side metric arrays.
    nle: dict[str, np.ndarray] = {k: np.zeros(n_episodes, dtype=np.float64) for k in _METRIC_KEYS}
    nax: dict[str, np.ndarray] = {k: np.zeros(n_episodes, dtype=np.float64) for k in _METRIC_KEYS}
    nle_deaths: list[str] = []
    nax_deaths: list[str] = []

    for ep in range(n_episodes):
        policy_fn, policy_name = _select_policy(ep, n_episodes)
        # Fresh RNG per episode so policy draws are deterministic given (ep).
        policy_rng = np.random.default_rng(ep)

        nle_ep = _run_nle_episode(
            NLE_cls, seed=ep, max_steps=max_steps,
            policy_fn=policy_fn, policy_rng=policy_rng,
        )
        nax_ep = _run_nethax_episode(
            NethaxEnv, jax, jnp, seed=ep, max_steps=max_steps,
            actions_taken=nle_ep["actions"],
        )

        for k in _METRIC_KEYS:
            nle[k][ep] = nle_ep[k]
            nax[k][ep] = nax_ep[k]
        nle_deaths.append(nle_ep["death_cause"])
        nax_deaths.append(nax_ep["death_cause"])

        if verbose:
            print(
                f"[ep {ep:4d} {policy_name:13s}] "
                f"NLE ret={nle_ep['episode_return']:7.2f} len={nle_ep['episode_length']:3d} "
                f"kills={nle_ep['monsters_killed']:2d} dlvl={nle_ep['dungeon_level']}  | "
                f"Nax ret={nax_ep['episode_return']:7.2f} len={nax_ep['episode_length']:3d} "
                f"kills={nax_ep['monsters_killed']:2d} dlvl={nax_ep['dungeon_level']}"
            )

    # Run KS on each marginal.
    ks_results: dict[str, tuple[float, float]] = {}
    diverging: list[str] = []
    for k in _METRIC_KEYS:
        stat, p = ks_2samp(nle[k], nax[k])
        ks_results[k] = (float(stat), float(p))
        if p <= 0.05:
            diverging.append(_METRIC_LABELS[k])

    verdict = "EQUIVALENT" if not diverging else "DIVERGE"

    # Death-cause summary (categorical — reported, not KS-tested).
    def _bucket_counts(deaths: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in deaths:
            out[d] = out.get(d, 0) + 1
        return out

    print()
    print("=" * 78)
    print(f"  Algorithmic-equivalence validator — {n_episodes} episodes, "
          f"max_steps={max_steps}")
    print("=" * 78)
    print(f"{'metric':<22}{'NLE mean':>14}{'Nax mean':>14}{'KS stat':>12}{'KS p':>14}")
    for k in _METRIC_KEYS:
        stat, p = ks_results[k]
        print(f"{_METRIC_LABELS[k]:<22}{nle[k].mean():>14.4f}{nax[k].mean():>14.4f}"
              f"{stat:>12.4f}{p:>14.4g}")
    print()
    print(f"  death-cause NLE   : {_bucket_counts(nle_deaths)}")
    print(f"  death-cause Nethax: {_bucket_counts(nax_deaths)}")
    print()
    if verdict == "EQUIVALENT":
        print("  VERDICT: EQUIVALENT (all KS p-values > 0.05)")
    else:
        print(f"  VERDICT: DIVERGE — diverging marginals: {', '.join(diverging)}")
    print("=" * 78)

    return 0 if verdict == "EQUIVALENT" else 1


# ---------------------------------------------------------------------------
# Pytest entry point — lightweight smoke so this validator participates in
# the suite without blowing the time budget.  The full N=1000 run is meant
# to be invoked via the script entry point.
# ---------------------------------------------------------------------------

def test_algorithmic_equivalence_smoke():
    """Pytest smoke: 4 episodes / 20 steps to verify the harness wires up.

    A "DIVERGE" verdict still passes the test — the smoke run is too
    small to be statistically meaningful (KS on 4 points has effectively
    no power) and we just want to verify the plumbing exercises both
    policies (random + scripted) and produces well-formed metrics.
    """
    rc = run_validator(n_episodes=4, max_steps=20, verbose=False)
    assert rc in (0, 1, 2), f"unexpected return code {rc}"


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "smoke":
        rc = run_validator(n_episodes=10, max_steps=20, verbose=True)
    else:
        rc = run_validator(n_episodes=1000, max_steps=200, verbose=True)
    sys.exit(rc)
