"""Compare reward AND done flag between NLE and Nethax for a 20-step
trajectory.  Same scenario as tests/test_nle_byte_parity.py (seed=0,
rog-hum-cha-mal, action=0 = N x 20).
"""
import os, sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/Users/rsiegelmann/Downloads/Projects/nethax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from nle.env import NLE  # noqa: E402

from Nethax.nethax.parity_mode import set_parity_mode, ParityMode  # noqa: E402
set_parity_mode(ParityMode.NLE_BYTEPARITY)

from Nethax.nethax.env import NethaxEnv  # noqa: E402
from Nethax.nethax.constants.roles import Role  # noqa: E402
from Nethax.nethax.constants.races import Race  # noqa: E402

SEED = 0
N = 20
ACTION = 0

nle_env = NLE(character="rog-hum-cha-mal", max_episode_steps=100, savedir=None)
nle_env.seed(core=SEED, disp=SEED, reseed=False)
_ = nle_env.reset()

nax_env = NethaxEnv()
nax_state, _ = nax_env.reset(
    jax.random.PRNGKey(SEED),
    role=Role.ROGUE,
    race=Race.HUMAN,
    alignment=2,
)

print(
    f"{'step':>4} {'NLE_r':>10} {'Nax_r':>10} "
    f"{'NLE_done':>9} {'Nax_done':>9} "
    f"{'cum_NLE':>12} {'cum_Nax':>12}  match"
)

nle_total = 0.0
nax_total = 0.0
mismatches = 0
for step in range(1, N + 1):
    nle_step = nle_env.step(ACTION)
    if len(nle_step) == 4:
        _, nle_r, nle_done, _ = nle_step
    else:
        _, nle_r, _, nle_done, _ = nle_step

    nax_state, _, nax_r, nax_done, _ = nax_env.step(
        nax_state, jnp.int32(ACTION), jax.random.PRNGKey(SEED + step)
    )
    nle_r = float(nle_r)
    nax_r = float(nax_r)
    nle_done = bool(nle_done)
    nax_done = bool(nax_done)
    nle_total += nle_r
    nax_total += nax_r
    same = abs(nle_r - nax_r) < 1e-6 and nle_done == nax_done
    if not same:
        mismatches += 1
    marker = "  YES" if same else "  NO  <--"
    print(
        f"{step:>4} {nle_r:>10.2f} {nax_r:>10.2f} "
        f"{str(nle_done):>9} {str(nax_done):>9} "
        f"{nle_total:>12.2f} {nax_total:>12.2f}{marker}"
    )
    if nle_done or nax_done:
        print(f"[done] step {step}")
        break

print(
    f"\nTotal: NLE={nle_total:.2f}  Nethax={nax_total:.2f}  "
    f"delta={nax_total - nle_total:+.2f}  per-step mismatches={mismatches}/{N}"
)
nle_env.close()
