"""Probe minihax monster_ai positions + glyphs for Mon/Ult 15x15."""
import os, sys
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['NETHAX_EAGER'] = '1'

import jax
jax.config.update('jax_disable_jit', True)
import gymnasium as _g
sys.modules['gym'] = _g
sys.modules['gym.spaces'] = _g.spaces
sys.modules['gym.envs'] = _g.envs
sys.modules['gym.envs.registration'] = _g.envs.registration
import numpy as np

sys.path.insert(0, '.')
from Nethax.nethax.parity_mode import ParityMode, set_parity_mode
set_parity_mode(ParityMode.NLE_BYTEPARITY)
from Nethax.minihax.minihax_env import MinihaxEnv

for env_id in ("MiniHack-Room-Monster-15x15-v0", "MiniHack-Room-Ultimate-15x15-v0"):
    env = MinihaxEnv(env_id)
    state, _info = env.reset(jax.random.key(0))
    mai = state.monster_ai
    alive = np.asarray(mai.alive)
    pos = np.asarray(mai.pos)
    species = np.asarray(mai.species) if hasattr(mai, 'species') else None
    print(f"\n{env_id}: player_pos = (y={int(state.player_pos[0])}, x={int(state.player_pos[1])})")
    for i in range(len(alive)):
        if alive[i]:
            sp = int(species[i]) if species is not None else "?"
            print(f"  slot {i}: alive, pos=(y={pos[i][0]}, x={pos[i][1]}), species={sp}")
