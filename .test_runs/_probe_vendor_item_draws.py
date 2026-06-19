"""Extract vendor's per-item rn2 draws for BULLWHIP/LJ/FEDORA per seed."""
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

# BULLWHIP=64, LEATHER_JACKET=114, FEDORA=74, FOOD_RATION=268
ITEMS = {64: "BULLWHIP", 114: "LEATHER_JACKET", 74: "FEDORA"}

for seed in (0, 1, 2, 5):
    env = os.environ.copy()
    env["NETHAX_RN2_TRACE"] = "1"
    env["SEED"] = str(seed)
    res = subprocess.run(
        [PY, "-c", SCRIPT], env=env, capture_output=True, text=True,
        timeout=120,
    )
    cur_item = None
    item_draws = {n: [] for n in ITEMS}
    for line in res.stderr.splitlines():
        if line.startswith("ITEM_BEGIN otyp="):
            otyp = int(line.split("=")[1])
            cur_item = otyp if otyp in ITEMS else None
        elif line.startswith("ITEM_END"):
            cur_item = None
        elif cur_item is not None and line.startswith("NETHAX_RN2"):
            parts = line.split()
            # NETHAX_RN2 idx max result
            item_draws[cur_item].append((int(parts[2]), int(parts[3])))
    print(f"=== seed={seed} ===")
    for otyp, name in ITEMS.items():
        print(f"  {name}: {item_draws[otyp]}")
