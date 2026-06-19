"""Probe minihax internal state for Ult-5x5/15x15 to find FOV bug."""
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
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, '.')
from Nethax.nethax.parity_mode import ParityMode, set_parity_mode
set_parity_mode(ParityMode.NLE_BYTEPARITY)
from Nethax.minihax.minihax_env import MinihaxEnv
from Nethax.nethax.obs.nle_obs import build_nle_observation

for env_id in ("MiniHack-Room-Ultimate-5x5-v0", "MiniHack-Room-Ultimate-15x15-v0", "MiniHack-Room-Monster-15x15-v0"):
    env = MinihaxEnv(env_id)
    state, _info = env.reset(jax.random.key(0))
    obs = build_nle_observation(state)
    glyphs = np.asarray(obs["glyphs"])
    pp = np.asarray(state.player_pos)
    print(f"\n{env_id}: player_pos = (y={int(pp[0])}, x={int(pp[1])})")
    # Find where 2378 (floor) and 2359 (stone) appear
    lst = np.asarray(state.last_seen_terrain[0, 0])
    vis = np.asarray(state.visible)
    # Print terrain + visibility around (y=10, x=37) for 5x5; or (y=9, x=36) for 15x15
    if "Ultimate-5x5" in env_id:
        y_focus, x_focus = 10, 37
    elif "Ultimate-15x15" in env_id:
        y_focus, x_focus = 9, 36
    else:  # Monster-15x15
        y_focus, x_focus = 4, 39
    print(f"  Focus around ({y_focus}, {x_focus}):")
    for yy in range(max(0, y_focus-2), min(21, y_focus+3)):
        terrain_row = " ".join(f"{int(state.terrain[0,0,yy,xx]):3d}" for xx in range(max(0,x_focus-3), min(80,x_focus+4)))
        vis_row = " ".join(f"{'V' if vis[yy,xx] else '.'}" for xx in range(max(0,x_focus-3), min(80,x_focus+4)))
        lst_row = " ".join(f"{int(lst[yy,xx]):3d}" for xx in range(max(0,x_focus-3), min(80,x_focus+4)))
        gly_row = " ".join(f"{int(glyphs[yy,xx]):4d}" for xx in range(max(0,x_focus-3), min(80,x_focus+4)))
        print(f"    y={yy} terrain: {terrain_row}")
        print(f"    y={yy} visible: {vis_row}")
        print(f"    y={yy} lst    : {lst_row}")
        print(f"    y={yy} glyph  : {gly_row}")
