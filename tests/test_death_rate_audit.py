"""Per-step death rate audit — measures why Nethax terminates ~3.5x
faster than NLE under random/WAIT policy.

Run: PYTHONPATH=. python tests/test_death_rate_audit.py
"""
import jax, jax.numpy as jnp
from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.races import Race

def audit(n_seeds=5, n_steps=50, role=Role.ROGUE, alignment=2):
    env = NethaxEnv()
    deaths_by_step = []
    for seed in range(n_seeds):
        state, _ = env.reset(jax.random.PRNGKey(seed), role=role, race=Race.HUMAN, alignment=alignment)
        hps = [int(state.player_hp)]
        for step in range(n_steps):
            state, _, _, done, _ = env.step(
                state, jnp.int32(46),  # WAIT
                jax.random.PRNGKey(seed * 1000 + step),
            )
            hps.append(int(state.player_hp))
            if bool(done):
                deaths_by_step.append((seed, step, hps))
                break
        else:
            deaths_by_step.append((seed, n_steps, hps))
    return deaths_by_step

if __name__ == "__main__":
    results = audit()
    for seed, last_step, hps in results:
        died = last_step < 50
        print(f"seed={seed} {'DIED' if died else 'survived'} step={last_step} hp_trace={hps[:10]}... last_hp={hps[-1]}")
    mean_len = sum(r[1] for r in results) / len(results)
    print(f"mean ep len: {mean_len}")
