"""Per-handler jaxpr op count for the 43 dispatch handlers."""
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
    from Nethax.nethax.subsystems import action_dispatch

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    rng = jax.random.PRNGKey(1)

    handlers = action_dispatch._HANDLERS
    print(f"{'slot':>4}  {'handler':<24}  {'ops':>10}")
    print("-" * 48)
    total = 0
    rows = []
    for i, h in enumerate(handlers):
        try:
            jp = jax.make_jaxpr(h)(state, rng)
            c = _count(jp.jaxpr)
        except Exception as e:
            c = -1
        rows.append((i, h.__name__, c))
        total += max(c, 0)
    rows.sort(key=lambda r: -r[2])
    for slot, name, c in rows:
        print(f"{slot:>4}  {name:<24}  {c:>10}")
    print(f"\nTOTAL: {total}")


if __name__ == "__main__":
    main()
