"""NLE side-by-side byte-diff validator.

THIS is the actual proof of NLE/MiniHack agent-transferability.  All other
parity work (RNG, obs encoding, multi-key state machine, reward managers)
is necessary but not sufficient — the only ground truth is:

  Given the same seed + same action sequence, do NLE and Nethax produce
  byte-identical observation tensors?

This script:
  1. Imports both ``nle`` (vendor) and ``Nethax.nethax.env`` (ours).
  2. Resets each env with seed=0.
  3. Steps each with the same scripted action sequence.
  4. After each step, diffs every observation field byte-for-byte.
  5. Exits non-zero on the first divergence, with a precise location.

When this passes, NLE-trained agents can plug into Nethax with the same
weights.  When it fails (today, expected!), the diff tells us EXACTLY
which channel/index/value diverges.

Run:
    JAX_PLATFORMS=cpu .venv/bin/python tests/test_nle_byte_parity.py

Pytest harness wraps this so it shows up in the suite.
"""
from __future__ import annotations

import sys
import numpy as np


def _safe_import_nle():
    try:
        from nle.env import NLE  # vendor/nle/nle/env/base.py::NLE
        # NLE class constructor accepts character spec like "rog-hum-cha-mal"
        # and a savedir.  Default reset returns an obs dict.
        env = NLE(
            character="rog-hum-cha-mal",
            max_episode_steps=100,
            savedir=None,
        )
        return env, None
    except Exception as e:  # pragma: no cover
        return None, str(e)


def _safe_import_nethax():
    try:
        import jax
        import jax.numpy as jnp
        from Nethax.nethax.env import NethaxEnv
        # Activate strict NLE byte-parity (ISAAC64 RNG, NLE bit layouts).
        # Without this, Nethax uses Threefry → entirely different dungeon
        # despite same seed.  Cite: Nethax/nethax/parity_mode.py:51 NLE_BYTEPARITY.
        from Nethax.nethax.parity_mode import set_parity_mode, ParityMode
        set_parity_mode(ParityMode.NLE_BYTEPARITY)
        return NethaxEnv, jax, jnp, None
    except Exception as e:
        return None, None, None, str(e)


def _diff_obs(nle_obs: dict, nax_obs: dict, step_idx: int) -> list[str]:
    """Diff each shared field byte-for-byte; return a list of human-
    readable divergence strings (one per failing channel)."""
    # Channels NLE doesn't expose in its default observation_shape but
    # Nethax builds anyway — not a divergence, just a superset.
    NETHAX_ONLY_OK = {"internal", "misc", "program_state"}

    diffs = []
    shared = set(nle_obs.keys()) & set(nax_obs.keys())
    missing_in_nax = set(nle_obs.keys()) - set(nax_obs.keys())
    missing_in_nle = (set(nax_obs.keys()) - set(nle_obs.keys())) - NETHAX_ONLY_OK

    if missing_in_nax:
        diffs.append(
            f"[step {step_idx}] Nethax MISSING channels: {sorted(missing_in_nax)}"
        )
    if missing_in_nle:
        diffs.append(
            f"[step {step_idx}] NLE missing channels (Nethax-only): "
            f"{sorted(missing_in_nle)}"
        )

    for key in sorted(shared):
        nle_arr = np.asarray(nle_obs[key])
        nax_arr = np.asarray(nax_obs[key])

        if nle_arr.shape != nax_arr.shape:
            diffs.append(
                f"[step {step_idx}] {key}: SHAPE mismatch "
                f"NLE={nle_arr.shape} Nethax={nax_arr.shape}"
            )
            continue
        if nle_arr.dtype != nax_arr.dtype:
            # dtype mismatch is a soft warning if values still match
            pass

        # Cast both to int64 for comparison (handles uint8 vs int8 etc).
        try:
            n1 = nle_arr.astype(np.int64)
            n2 = nax_arr.astype(np.int64)
        except (TypeError, ValueError):
            # Likely bytes; compare as raw bytes
            if nle_arr.tobytes() != nax_arr.tobytes():
                diffs.append(f"[step {step_idx}] {key}: byte mismatch")
            continue

        mismatch_mask = n1 != n2
        n_mismatch = int(mismatch_mask.sum())
        if n_mismatch > 0:
            total = int(n1.size)
            pct = 100.0 * n_mismatch / total
            # Pick first 3 diverging indices for trace
            flat_idx = np.argwhere(mismatch_mask.ravel())[:3].ravel().tolist()
            samples = [
                (int(i), int(n1.ravel()[i]), int(n2.ravel()[i])) for i in flat_idx
            ]
            sample_str = "; ".join(
                f"@{i}: NLE={a} Nethax={b}" for (i, a, b) in samples
            )
            diffs.append(
                f"[step {step_idx}] {key}: {n_mismatch}/{total} ({pct:.1f}%) "
                f"bytes diverge ({sample_str})"
            )

    return diffs


def run_validator(
    num_steps: int = 20,
    seed: int = 0,
    verbose: bool = True,
    show_all: bool = False,
) -> int:
    """Run the side-by-side validator.  Returns the number of total
    divergence strings collected across all steps.  Zero = full parity.

    If ``show_all`` is True, prints every per-step divergence line (no
    truncation).  Useful for trajectory analysis where summary-tail data
    matters as much as the first 8 channels.
    """
    nle_env, nle_err = _safe_import_nle()
    if nle_err:
        print(f"[SKIP] Could not import nle: {nle_err}")
        return -1

    nethax_cls, jax, jnp, nax_err = _safe_import_nethax()
    if nax_err:
        print(f"[SKIP] Could not import Nethax: {nax_err}")
        return -2

    # NLE reset.  NLE's `env.seed` signature is
    #   ``seed(core=None, disp=None, reseed=False)``
    # (vendor/nle/nle/env/base.py:441).  The previous call used
    # ``seeds=(seed, seed)`` (kwarg name ``seeds`` — invalid!).  Python
    # raised TypeError which the broad except silently swallowed, so NLE
    # fell back to its default wallclock/urandom seeding — hence the
    # NLE-side non-determinism observed across validator runs (player_x
    # flipping between 15, 32, 46, 57, ... for nominal seed=0).
    #
    # Pass explicit core+disp positional args AND reseed=False to disable
    # NLE's anti-TAS strong-reseed (vendor/nle/nle/env/base.py:455-466).
    if hasattr(nle_env, "seed"):
        nle_env.seed(core=seed, disp=seed, reseed=False)
    nle_obs = nle_env.reset()

    # Nethax reset — force same role as NLE side (rog-hum-cha-mal).
    nax_env = nethax_cls()
    try:
        from Nethax.nethax.constants.roles import Role
        from Nethax.nethax.constants.races import Race
        nax_state, nax_obs = nax_env.reset(
            jax.random.PRNGKey(seed),
            role=Role.ROGUE,
            race=Race.HUMAN,
            alignment=2,  # chaotic
        )
    except (ImportError, AttributeError):
        # Fallback: default role (Valkyrie) if Role enum import fails
        nax_state, nax_obs = nax_env.reset(jax.random.PRNGKey(seed))

    # Convert nethax obs (JAX arrays) to numpy dict.
    def _nax_to_dict(obs):
        if isinstance(obs, dict):
            return {k: np.asarray(v) for k, v in obs.items()}
        return {}

    nax_dict = _nax_to_dict(nax_obs)
    nle_dict = nle_obs if isinstance(nle_obs, dict) else {}

    all_diffs: list[str] = []
    diffs = _diff_obs(nle_dict, nax_dict, step_idx=0)
    all_diffs.extend(diffs)
    if verbose:
        if diffs:
            print(f"\n=== step 0 (after reset): {len(diffs)} divergences ===")
            limit = len(diffs) if show_all else 20
            for d in diffs[:limit]:
                print(f"  {d}")
        else:
            print("\n=== step 0 (after reset): MATCH ===")

    # Wait action = NLE Command.WAIT = ord('.')=46 in the legacy action map.
    # Nethax: also int 0 = wait in many configs; use the same int for both.
    # We will use a no-op or "search" (ord('s')=115) which is safe in both.
    action = 0

    for step_idx in range(1, num_steps + 1):
        # Step NLE
        try:
            nle_step = nle_env.step(action)
            # gym API: (obs, reward, done, info)
            if len(nle_step) == 4:
                nle_obs, _, nle_done, _ = nle_step
            else:
                nle_obs, _, _, nle_done, _ = nle_step
        except Exception as e:
            print(f"[abort] NLE step {step_idx} failed: {e}")
            break

        # Step Nethax
        try:
            # Pass ``action`` as a Python int so NethaxEnv routes through
            # the static-action fast path (env.py::_dispatch_jit_validator),
            # avoiding the 46-branch lax.switch in dispatch_action that
            # otherwise stalls JIT cold-compile.
            nax_state, nax_obs, _, nax_done, _ = nax_env.step(
                nax_state, action, jax.random.PRNGKey(seed + step_idx)
            )
        except Exception as e:
            print(f"[abort] Nethax step {step_idx} failed: {e}")
            break

        nax_dict = _nax_to_dict(nax_obs)
        nle_dict = nle_obs if isinstance(nle_obs, dict) else {}
        diffs = _diff_obs(nle_dict, nax_dict, step_idx)
        all_diffs.extend(diffs)
        if verbose:
            if diffs:
                print(
                    f"\n=== step {step_idx}: {len(diffs)} divergences ==="
                )
                limit = len(diffs) if show_all else 8
                for d in diffs[:limit]:
                    print(f"  {d}")
                if not show_all and len(diffs) > 8:
                    print(f"  ... ({len(diffs) - 8} more)")
            else:
                print(f"=== step {step_idx}: MATCH ===")

        if bool(nle_done):
            print(f"[done] NLE terminated at step {step_idx}")
            break

    nle_env.close()
    return len(all_diffs)


if __name__ == "__main__":
    # CLI: tests/test_nle_byte_parity.py [n_steps] [--all]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    n_steps = int(args[0]) if args else 5
    show_all = "--all" in flags
    total = run_validator(num_steps=n_steps, verbose=True, show_all=show_all)
    if total < 0:
        print("\n[skip] validator could not run")
        sys.exit(2)
    print(f"\n[summary] {total} total divergences across {n_steps} steps")
    sys.exit(0 if total == 0 else 1)
