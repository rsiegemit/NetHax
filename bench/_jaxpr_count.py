"""Quick jaxpr op-count for env._step_impl.

Counts top-level eqns + nested ops to spot the biggest compile contributors.
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
from Nethax.nethax.env import NethaxEnv, _step_impl


def _eqn_kinds(jaxpr, kinds=None, depth=0, max_depth=4):
    if kinds is None:
        kinds = {}
    for eqn in jaxpr.eqns:
        name = str(eqn.primitive)
        kinds[name] = kinds.get(name, 0) + 1
        # Recurse into nested call jaxprs.
        for p_name, p_val in eqn.params.items():
            if hasattr(p_val, "jaxpr"):
                _eqn_kinds(p_val.jaxpr, kinds, depth + 1, max_depth)
            elif isinstance(p_val, tuple):
                for v in p_val:
                    if hasattr(v, "jaxpr"):
                        _eqn_kinds(v.jaxpr, kinds, depth + 1, max_depth)
    return kinds


def main():
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    jp = jax.make_jaxpr(_step_impl)(state, jnp.int32(0), jax.random.PRNGKey(1))
    print(f"top-level eqns: {len(jp.jaxpr.eqns)}")
    kinds = _eqn_kinds(jp.jaxpr)
    items = sorted(kinds.items(), key=lambda kv: -kv[1])
    total = sum(kinds.values())
    print(f"total nested eqns: {total}")
    print("top 25 primitives:")
    for k, v in items[:25]:
        print(f"  {v:>7d}  {k}")


if __name__ == "__main__":
    main()
