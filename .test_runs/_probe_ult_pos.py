"""Probe vendor Ultimate envs: hero, monsters, traps, dnstair positions."""
import os, sys, random
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import gymnasium as _g
sys.modules['gym'] = _g
sys.modules['gym.spaces'] = _g.spaces
sys.modules['gym.envs'] = _g.envs
sys.modules['gym.envs.registration'] = _g.envs.registration

import minihack  # noqa: F401
from minihack.envs.room import (
    MiniHackRoom5x5Ultimate, MiniHackRoom15x15Ultimate,
    MiniHackRoom15x15Monster,
)

OBS = ("glyphs", "blstats")

# CMAP glyphs: 2378=S_room, 2383=S_dnstair, 2359=S_stone
# Trap glyphs: traps start around 2389 (S_arrow_trap) to 2410
# Monster glyphs: 0..380, hero=327
def classify(g: int) -> str:
    if g == 327: return "HERO"
    if 0 <= g <= 380: return f"MON({g})"
    if g == 2378: return "floor"
    if g == 2359: return "stone"
    if g == 2383: return "stair"
    if 2389 <= g <= 2410: return f"TRAP({g})"
    return f"g({g})"

for name, cls in [
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
    print(f"\n{name}: hero @ blstats (y={int(bl[1])}, x={int(bl[0])})")
    interesting = []
    for yy in range(21):
        for xx in range(79):
            g = int(glyphs[yy, xx])
            if g not in (2378, 2359, 2360):  # ignore floor, stone, dark_part_visible
                interesting.append((yy, xx, classify(g)))
    for loc in interesting:
        print(f"  ({loc[0]:2d},{loc[1]:2d}): {loc[2]}")
    try:
        env.close()
    except Exception:
        pass
