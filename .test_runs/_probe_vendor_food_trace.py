"""Capture vendor's rn2 trace at the FOOD_RATION block per seed.

Uses the built-in vendor NETHAX_RN2_TRACE instrumentation (rnd.c:108) plus
the per-item ITEM_BEGIN/ITEM_END markers in ini_inv (u_init.c:980-1078).
FOOD_RATION otyp = 268 (per Nethax/nethax/subsystems/character.py:296).
"""
import os, sys, subprocess

os.environ.pop("NETHAX_RN2_TRACE", None)  # we'll set per-subproc

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
    # The trace lands on stderr.  Parse out the FOOD_RATION block (otyp=268).
    in_food = False
    food_draws = []
    for line in res.stderr.splitlines():
        if line.startswith("ITEM_BEGIN otyp=268"):
            in_food = True
            continue
        if line.startswith("ITEM_END otyp=268"):
            in_food = False
            continue
        if in_food and line.startswith("NETHAX_RN2"):
            parts = line.split()
            # NETHAX_RN2 <idx> <max> <result>
            if len(parts) >= 4:
                food_draws.append((int(parts[2]), int(parts[3])))
    print(f"seed={seed}: FOOD_RATION block draws = {food_draws}")
