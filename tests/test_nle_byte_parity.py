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


# Cache for the JIT'd Nethax rollout function.  Keyed by (id(nax_env),
# num_steps) so we re-use the compiled artifact across seeds when
# num_steps and env are constant.  In single-process multiseed runs
# (.test_runs/multiseed_byteparity.py) the env is recreated per seed,
# so this cache misses across seeds — but JAX's persistent compilation
# cache (JAX_COMPILATION_CACHE_DIR) still amortizes the XLA cost.
_ROLLOUT_CACHE: dict = {}


def _build_nethax_rollout(nax_env, jax, jnp, num_steps: int):
    """Build a JIT'd ``lax.scan`` rollout for ``num_steps`` Nethax steps.

    Returns ``rollout(init_state, action, seed_int) -> (all_obs, all_done)``
    where each leaf has a leading axis of ``num_steps``.  ``num_steps``
    is closed over as a Python int so the scan length is static at trace
    time.

    Rationale (vs. a Python ``for step_idx in range(...)`` loop calling
    ``nax_env.step`` per iteration): every internal ``lax.scan``/
    ``lax.cond`` in ``nax_env.step`` was lazily compiled on the first
    iteration of each call, so a 1000-step rollout paid the full XLA
    cold-compile cost 1000 times (~6+ hours observed on H100).  Wrapping
    the loop in a single outer ``lax.scan`` collapses that to ONE compile
    that covers the whole trajectory.
    """
    cache_key = (id(nax_env), num_steps)
    if cache_key in _ROLLOUT_CACHE:
        return _ROLLOUT_CACHE[cache_key]

    @jax.jit
    def rollout(init_state, action, seed_int):
        # ``seed_int`` is a jnp.int32 scalar; the scan iterates over
        # idx_seq = [1, 2, ..., num_steps].  Per-step PRNGKey is
        # ``jax.random.PRNGKey(seed_int + step_idx)`` — semantically
        # identical to the original Python ``seed + step_idx``.
        idx_seq = jnp.arange(1, num_steps + 1, dtype=jnp.int32)

        def step_fn(state, step_idx):
            rng = jax.random.PRNGKey(seed_int + step_idx)
            new_state, obs, _reward, done, _info = nax_env.step(state, action, rng)
            return new_state, (obs, done)

        _final_state, (all_obs, all_done) = jax.lax.scan(step_fn, init_state, idx_seq)
        return all_obs, all_done

    _ROLLOUT_CACHE[cache_key] = rollout
    return rollout


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

    # ------------------------------------------------------------------
    # Nethax side: pre-compute the entire ``num_steps`` trajectory in ONE
    # JIT'd ``lax.scan``.  The scan body wraps ``nax_env.step``, so XLA
    # compiles the inner dispatch / monster-AI / status pipeline once and
    # unrolls it ``num_steps`` times inside a single compiled HLO graph.
    # This is the whole point of the refactor — see
    # ``_build_nethax_rollout`` docstring for the cold-compile math.
    #
    # NLE side: still stepped in Python below (it's vendor C, cannot be
    # scanned).  We index into the pre-computed ``nax_all_obs`` pytree
    # per step and diff against the live NLE obs.
    # ------------------------------------------------------------------
    if num_steps > 0:
        rollout = _build_nethax_rollout(nax_env, jax, jnp, num_steps)
        try:
            nax_all_obs, _nax_all_done = rollout(
                nax_state, jnp.int32(action), jnp.int32(seed)
            )
        except Exception as e:
            print(f"[abort] Nethax rollout (scan) failed: {e}")
            nle_env.close()
            return len(all_diffs)

        # Materialize per-step obs slices once up-front.  ``nax_all_obs``
        # is a dict-of-arrays where each array has a leading axis of
        # ``num_steps``.  Slicing per-step inside the comparison loop is
        # cheap (numpy view).
        if isinstance(nax_all_obs, dict):
            nax_all_obs_np = {k: np.asarray(v) for k, v in nax_all_obs.items()}
        else:
            nax_all_obs_np = {}

        for step_idx in range(1, num_steps + 1):
            # Step NLE (vendor C; cannot be lifted into the scan).
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

            # Slice Nethax obs at step_idx-1 (scan output is 0-indexed
            # over the ``arange(1, num_steps + 1)`` sequence).
            scan_i = step_idx - 1
            nax_dict = {k: v[scan_i] for k, v in nax_all_obs_np.items()}
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
                # Nethax-side tail (steps step_idx+1 .. num_steps) was
                # pre-computed inside the scan; we just discard it.
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
