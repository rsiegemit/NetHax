"""Compare vendor vs minihax monster entities for Mon-15x15 / Ult-15x15."""
import os, sys, random
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np
import gymnasium as _g
sys.modules['gym'] = _g
sys.modules['gym.spaces'] = _g.spaces
sys.modules['gym.envs'] = _g.envs
sys.modules['gym.envs.registration'] = _g.envs.registration
import minihack  # noqa
from minihack.envs.room import MiniHackRoom15x15Monster, MiniHackRoom15x15Ultimate

ENV = os.environ.get("ENV", "mon")
SEED = int(os.environ.get("SEED", "0"))
cls = MiniHackRoom15x15Monster if ENV == "mon" else MiniHackRoom15x15Ultimate

random.seed(SEED)
env = cls(observation_keys=("glyphs", "chars", "blstats"), character="arc-hum-law-mal")
env.seed(SEED, SEED, reseed=False)
r = env.reset()
obs = r[0] if isinstance(r, tuple) else r
g = np.asarray(obs["glyphs"]); bl = np.asarray(obs["blstats"])
hero = (int(bl[1]), int(bl[0]))
# Monster glyphs are [0, 381). Hero = 327.
ents = []
for y in range(21):
    for x in range(79):
        gv = int(g[y, x])
        if 0 <= gv < 381 and gv != 327:
            ents.append((y, x, gv))
print(f"VENDOR {ENV}-15x15 seed={SEED}: hero(y,x)={hero}  monsters={ents}")
env.close()
