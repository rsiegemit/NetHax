"""Probe the 3 FOOD_RATION rn2(6) draws across seeds to derive qty formula.

Vendor: seed=0 → 4 rations, seeds 1/2/5 → 3 rations.
If our formula `qty = 3 + sum(draw == 0)` is correct:
  seed=0 needs exactly one of 3 draws to be 0
  seeds 1/2/5 need zero of 3 draws to be 0
"""
import os, sys
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['NETHAX_EAGER'] = '1'

import jax
jax.config.update('jax_disable_jit', True)
import jax.numpy as jnp

sys.path.insert(0, '.')
from Nethax.nethax.parity_mode import ParityMode, set_parity_mode
set_parity_mode(ParityMode.NLE_BYTEPARITY)
from Nethax.nethax.vendor_rng import init as _init_isaac, rn2_jax

for seed in (0, 1, 2, 5):
    # Build an Isaac64State from the seed (matching how env.reset seeds it).
    vrng = _init_isaac(int(seed))

    # Replay the prefix: rn2(11), rn2(10), rn2(10), rn2(2) for BULLWHIP
    # then 8 more for LEATHER_JACKET + FEDORA armor.
    # See character.py:1452-1468.
    for mod in (11, 10, 10, 2,   10, 11, 10, 10,   10, 11, 10, 10):
        vrng, _ = rn2_jax(vrng, jnp.int32(mod))
    # Now we're at the FOOD_RATION block. Draw the 3 rn2(6) values.
    draws = []
    for _ in range(3):
        vrng, d = rn2_jax(vrng, jnp.int32(6))
        draws.append(int(d))
    n_zero = sum(1 for d in draws if d == 0)
    print(f"seed={seed}: 3× rn2(6) draws = {draws}, count(==0) = {n_zero}, "
          f"qty = 3 + {n_zero} = {3 + n_zero}")
