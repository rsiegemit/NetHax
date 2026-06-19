"""Capture vendor's draws count at the FOOD_RATION block per seed."""
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
    env["SEED"] = str(seed)
    res = subprocess.run(
        [PY, "-c", SCRIPT], env=env, capture_output=True, text=True,
        timeout=120,
    )
    # Find draws count at the FIRST FOOD_RATION ITEM_BEGIN.
    food_pre_draws = None
    food_draws = []
    for line in res.stderr.splitlines():
        if line.startswith("NETHAX_RN2"):
            parts = line.split()
            if food_pre_draws is None:
                # parts: NETHAX_RN2 <idx> <max> <result>
                last_idx = int(parts[1])
        elif line.startswith("ITEM_BEGIN otyp=268") and food_pre_draws is None:
            food_pre_draws = last_idx + 1  # next index to draw at
        elif line.startswith("ITEM_END otyp=268"):
            break
    print(f"seed={seed}: vendor FOOD_RATION block starts at draws={food_pre_draws}")
