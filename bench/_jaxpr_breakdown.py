"""Per-subsystem jaxpr op count: dispatch_action vs monster_ai.step vs status_step.

Helps localize the worst compile-time contributor.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp


def _count(jaxpr):
    n = 0
    for eqn in jaxpr.eqns:
        n += 1
        for v in eqn.params.values():
            if hasattr(v, "jaxpr"):
                n += _count(v.jaxpr)
            elif isinstance(v, tuple):
                for w in v:
                    if hasattr(w, "jaxpr"):
                        n += _count(w.jaxpr)
    return n


def main():
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.subsystems.action_dispatch import dispatch_action
    from Nethax.nethax.subsystems.monster_ai import step as monster_step
    from Nethax.nethax.subsystems.status_effects import step as status_step
    from Nethax.nethax.subsystems.polymorph import step as polymorph_step
    from Nethax.nethax.subsystems.shop import shop_step
    from Nethax.nethax.subsystems.ascension import maybe_ascend
    from Nethax.nethax.obs.nle_obs import build_nle_observation

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    rng = jax.random.PRNGKey(1)

    measurements = [
        ("dispatch_action", lambda: jax.make_jaxpr(dispatch_action)(state, jnp.int32(0), rng)),
        ("monster_ai.step", lambda: jax.make_jaxpr(monster_step)(state, rng)),
        ("status_effects.step", lambda: jax.make_jaxpr(status_step)(
            state.status, rng, state.player_hp, state.player_hp_max,
            state.player_pw, state.player_pw_max, state.player_xl,
            state.player_role, state.done,
        )),
        ("polymorph.step", lambda: jax.make_jaxpr(polymorph_step)(state, rng)),
        ("shop_step", lambda: jax.make_jaxpr(shop_step)(state, rng)),
        ("maybe_ascend", lambda: jax.make_jaxpr(maybe_ascend)(state)),
        ("build_nle_observation", lambda: jax.make_jaxpr(build_nle_observation)(state)),
    ]
    print(f"{'subsystem':<28}  {'total ops':>10}  {'top eqns':>10}")
    print("-" * 56)
    for name, mk in measurements:
        jp = mk()
        total = _count(jp.jaxpr)
        top = len(jp.jaxpr.eqns)
        print(f"{name:<28}  {total:>10}  {top:>10}")


if __name__ == "__main__":
    main()
