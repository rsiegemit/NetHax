"""Probe vendor monster glyph position for Mon-5x5 and Mon-15x15."""
import os, sys, random
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import gymnasium as _g
sys.modules['gym'] = _g
sys.modules['gym.spaces'] = _g.spaces
sys.modules['gym.envs'] = _g.envs
sys.modules['gym.envs.registration'] = _g.envs.registration

import minihack  # noqa: F401
from minihack.envs.room import MiniHackRoom5x5Monster, MiniHackRoom15x15Monster

OBS = ("glyphs", "blstats")

for name, cls in [
    ("Mon-5x5", MiniHackRoom5x5Monster),
    ("Mon-15x15", MiniHackRoom15x15Monster),
]:
    random.seed(0)
    env = cls(observation_keys=OBS, character="arc-hum-law-mal")
    env.seed(0, 0, reseed=False)
    r = env.reset()
    obs = r[0] if isinstance(r, tuple) else r
    glyphs = np.asarray(obs["glyphs"])
    bl = np.asarray(obs["blstats"])
    hero_x, hero_y = int(bl[0]), int(bl[1])
    # Find non-hero monster glyphs (monster glyph range ~0-380 for monsters, hero=327)
    # MON range is 0..380, but inv items overlap. Look for any glyph 380 ≤ g but
    # in practice monsters are 0..377, hero=327. Just find unique glyphs in room.
    mon_locs = []
    for yy in range(21):
        for xx in range(79):
            g = int(glyphs[yy, xx])
            if 0 <= g <= 380 and g != 327:
                mon_locs.append((yy, xx, g))
    print(f"{name}: hero @ ({hero_y}, {hero_x}); non-hero monsters: {mon_locs}")
    try:
        env.close()
    except Exception:
        pass
