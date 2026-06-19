"""Capture vendor draws counts at major reset stages per seed."""
import os, sys, subprocess

PY = sys.executable
SCRIPT = """
import os, sys, random
sys.modules.setdefault('gym', __import__('gymnasium'))
import gymnasium as _g
sys.modules['gym.spaces'] = _g.spaces
sys.modules['gym.envs'] = _g.envs
sys.modules['gym.envs.registration'] = _g.envs.registration
import minihack
from minihack.envs.room import MiniHackRoom5x5
random.seed(int(os.environ['SEED']))
env = MiniHackRoom5x5(observation_keys=('inv_strs',), character='arc-hum-law-mal')
env.seed(int(os.environ['SEED']), int(os.environ['SEED']), reseed=False)
env.reset()
env.close()
"""

for seed in (0, 1, 2, 5):
    env = os.environ.copy()
    env["NETHAX_RN2_TRACE"] = "1"
    env["NETHAX_DUNGEON_TRACE"] = "1"
    env["SEED"] = str(seed)
    res = subprocess.run(
        [PY, "-c", SCRIPT], env=env, capture_output=True, text=True,
        timeout=120,
    )
    last_idx = -1
    stages = {}
    for line in res.stderr.splitlines():
        if line.startswith("NETHAX_RN2"):
            parts = line.split()
            last_idx = int(parts[1])
        elif (line.startswith("ROLE_INIT") or line.startswith("INIT_DUNGEONS")
              or line.startswith("ITEM_BEGIN") or line.startswith("U_INIT")
              or line.startswith("MKLEV") or line.startswith("INIT_OBJECTS")):
            tag = line.strip()
            if tag not in stages:
                stages[tag] = last_idx + 1
    print(f"=== seed={seed} ===")
    for tag, off in sorted(stages.items(), key=lambda kv: kv[1]):
        print(f"  draws={off:4d}  {tag}")
