"""Probe vendor's actual hero position for failing envs at seed=0."""
import os, sys, random
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import gymnasium as _g
sys.modules['gym'] = _g
sys.modules['gym.spaces'] = _g.spaces
sys.modules['gym.envs'] = _g.envs
sys.modules['gym.envs.registration'] = _g.envs.registration

import minihack  # noqa: F401 - registers envs
from minihack.envs.room import (
    MiniHackRoom5x5Trap, MiniHackRoom15x15Trap,
    MiniHackRoom5x5Ultimate, MiniHackRoom15x15Ultimate,
    MiniHackRoom15x15Monster,
)

OBS = ("glyphs", "chars", "blstats")

for name, cls in [
    ("Trap-5x5", MiniHackRoom5x5Trap),
    ("Trap-15x15", MiniHackRoom15x15Trap),
    ("Ult-5x5", MiniHackRoom5x5Ultimate),
    ("Ult-15x15", MiniHackRoom15x15Ultimate),
    ("Mon-15x15", MiniHackRoom15x15Monster),
]:
    random.seed(0)
    env = cls(observation_keys=OBS, character="arc-hum-law-mal")
    env.seed(0, 0, reseed=False)
    r = env.reset()
    obs = r[0] if isinstance(r, tuple) else r
    glyphs = np.asarray(obs["glyphs"])
    bl = np.asarray(obs["blstats"])
    # blstats[0] = x, blstats[1] = y in vendor NLE
    hero_x, hero_y = int(bl[0]), int(bl[1])
    print(f"{name}: hero @ blstats (y={hero_y}, x={hero_x})  glyph[{hero_y},{hero_x}]={int(glyphs[hero_y, hero_x])}")
    # Also report row context
    yrange = range(max(0, hero_y - 2), min(21, hero_y + 3))
    xrange = range(max(0, hero_x - 5), min(79, hero_x + 6))
    for yy in yrange:
        row = " ".join(f"{int(glyphs[yy, xx]):4d}" for xx in xrange)
        print(f"  y={yy}: {row}")
    try:
        env.close()
    except Exception:
        pass
