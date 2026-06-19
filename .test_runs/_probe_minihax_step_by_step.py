"""Step through MinihaxEnv.reset and track vendor_rng.draws + next rn2(6) values
at each major checkpoint to find where seed=1 stream converges to seed=0."""
import os, sys
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['NETHAX_EAGER'] = '1'

import jax
jax.config.update('jax_disable_jit', True)
import jax.numpy as jnp

sys.path.insert(0, '.')
from Nethax.nethax.parity_mode import ParityMode, set_parity_mode
set_parity_mode(ParityMode.NLE_BYTEPARITY)

from Nethax.nethax.vendor_rng import init_jax as _init, rn2_jax
from Nethax.nethax.dungeon.branches import (
    consume_init_dungeons_draws, consume_init_dungeons_variable_draws,
)

def peek3(label, vrng, seed):
    """Peek 3 rn2(6) draws without committing — fork the state."""
    v = vrng
    out = []
    for _ in range(3):
        v, d = rn2_jax(v, jnp.int32(6))
        out.append(int(d))
    print(f"  seed={seed} {label}: draws={int(vrng.draws)}, peek3 rn2(6) = {out}")

for seed in (0, 1, 2, 5):
    print(f"=== seed={seed} ===")
    vrng = _init(jnp.uint64(seed))
    peek3("after init_jax", vrng, seed)

    # Mimic env.py: Archeologist rn2(100) quest leader gender (line 311)
    vrng, _ = rn2_jax(vrng, jnp.int32(100))
    peek3("after arc_gender rn2(100)", vrng, seed)

    # init_dungeons
    vrng, dstate = consume_init_dungeons_draws(vrng)
    peek3("after init_dungeons_draws", vrng, seed)

    vrng, _ = consume_init_dungeons_variable_draws(vrng, dstate)
    peek3("after init_dungeons_variable_draws", vrng, seed)

    # Mimic _consume_ini_inv_archeologist_draws prefix (12 draws before FOOD).
    for mod in (11, 10, 10, 2,   10, 11, 10, 10,   10, 11, 10, 10):
        vrng, _ = rn2_jax(vrng, jnp.int32(mod))
    peek3("after 12 pre-FOOD draws", vrng, seed)
