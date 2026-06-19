"""Probe vendor positions and inv_strs[3] across seeds {1,2,5} x 12 envs.

Emits JSON to .test_runs/vendor_seed_positions.json with records:
  {env_id, seed, entity_type, y_obs, x_obs, glyph}
Entity types: hero, stair, monster, trap, pet
Also emits inv_strs[slot=3] bytes as a separate top-level dict for sanity check.

NLE glyph ranges (per nle/include/display.h):
  MON: 0..380  (hero is a monster glyph)
  PET: 381..761
  INVIS: 762
  DETECT: 763..1143
  CORPSE/BODY: 1144..1524
  RIDDEN: 1525..1905
  OBJ: 1906..2358
  CMAP: 2359..2451 (stairs up=2374, stairs down=2375 approx)
  TRAP: 2452..2479
  ZAP/EXPLODE/etc: 2480+
  FLOOR/DARK/WALL: handled as "background"
"""
import os
import sys
import json
import random

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import gymnasium as _g

sys.modules["gym"] = _g
sys.modules["gym.spaces"] = _g.spaces
sys.modules["gym.envs"] = _g.envs
sys.modules["gym.envs.registration"] = _g.envs.registration

import minihack  # noqa: F401
from minihack.envs.room import (
    MiniHackRoom5x5,
    MiniHackRoom15x15,
    MiniHackRoom5x5Random,
    MiniHackRoom15x15Random,
    MiniHackRoom5x5Dark,
    MiniHackRoom15x15Dark,
    MiniHackRoom5x5Monster,
    MiniHackRoom15x15Monster,
    MiniHackRoom5x5Trap,
    MiniHackRoom15x15Trap,
    MiniHackRoom5x5Ultimate,
    MiniHackRoom15x15Ultimate,
)

ENVS = [
    ("Room-5x5",        MiniHackRoom5x5),
    ("Room-15x15",      MiniHackRoom15x15),
    ("Random-5x5",      MiniHackRoom5x5Random),
    ("Random-15x15",    MiniHackRoom15x15Random),
    ("Dark-5x5",        MiniHackRoom5x5Dark),
    ("Dark-15x15",      MiniHackRoom15x15Dark),
    ("Mon-5x5",         MiniHackRoom5x5Monster),
    ("Mon-15x15",       MiniHackRoom15x15Monster),
    ("Trap-5x5",        MiniHackRoom5x5Trap),
    ("Trap-15x15",      MiniHackRoom15x15Trap),
    ("Ult-5x5",         MiniHackRoom5x5Ultimate),
    ("Ult-15x15",       MiniHackRoom15x15Ultimate),
]
SEEDS = [1, 2, 5]

OBS = ("glyphs", "chars", "blstats", "inv_strs")

# Glyph range constants from NLE include/display.h
MON_END = 381        # 0..380 monsters (incl hero @327)
PET_END = 762        # 381..761 pets
INVIS = 762
CMAP_START = 2359
CMAP_END = 2452      # 2359..2451 dungeon features (stairs, doors, floor, walls)
TRAP_START = 2452
TRAP_END = 2480      # 2452..2479 traps
# Stair glyph offsets within CMAP: S_upstair=22, S_dnstair=23 typically
S_UPSTAIR = CMAP_START + 22  # 2381
S_DNSTAIR = CMAP_START + 23  # 2382

# Floor/wall/dark glyphs we treat as background (not "entities"):
# Conservative: anything in CMAP range that ISN'T a stair = background
def classify(g: int, hero_y: int, hero_x: int, y: int, x: int):
    if g == 327 and (y == hero_y and x == hero_x):
        return "hero"
    if 0 <= g < MON_END:
        return "monster"
    if MON_END <= g < PET_END:
        return "pet"
    if g == S_UPSTAIR or g == S_DNSTAIR:
        return "stair"
    if TRAP_START <= g < TRAP_END:
        return "trap"
    return None  # background / not interesting


records = []
inv_strs_slot3 = {}  # keyed by f"{env_id}|{seed}"

for env_id, cls in ENVS:
    for seed in SEEDS:
        random.seed(seed)
        env = cls(observation_keys=OBS, character="arc-hum-law-mal")
        env.seed(seed, seed, reseed=False)
        r = env.reset()
        obs = r[0] if isinstance(r, tuple) else r
        glyphs = np.asarray(obs["glyphs"])
        bl = np.asarray(obs["blstats"])
        inv_strs = np.asarray(obs["inv_strs"])
        hero_x, hero_y = int(bl[0]), int(bl[1])

        records.append({
            "env_id": env_id, "seed": seed, "entity_type": "hero",
            "y_obs": hero_y, "x_obs": hero_x,
            "glyph": int(glyphs[hero_y, hero_x]),
        })

        H, W = glyphs.shape
        for yy in range(H):
            for xx in range(W):
                if yy == hero_y and xx == hero_x:
                    continue
                g = int(glyphs[yy, xx])
                et = classify(g, hero_y, hero_x, yy, xx)
                if et is None:
                    continue
                records.append({
                    "env_id": env_id, "seed": seed, "entity_type": et,
                    "y_obs": yy, "x_obs": xx, "glyph": g,
                })

        # inv_strs[3] - slot 3, full row as bytes (list of ints for JSON)
        slot3 = inv_strs[3].tolist() if inv_strs.ndim >= 2 and inv_strs.shape[0] > 3 else None
        inv_strs_slot3[f"{env_id}|{seed}"] = slot3

        try:
            env.close()
        except Exception:
            pass

out_path = "/Users/rsiegelmann/Downloads/Projects/nethax/.test_runs/vendor_seed_positions.json"
payload = {
    "records": records,
    "inv_strs_slot3": inv_strs_slot3,
    "schema": {
        "records": "list of {env_id, seed, entity_type, y_obs, x_obs, glyph}",
        "inv_strs_slot3": "dict keyed 'env_id|seed' -> list[int] (bytes of inv_strs row 3)",
        "entity_types": ["hero", "stair", "monster", "trap", "pet"],
    },
}
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)

print(f"WROTE {out_path}: {len(records)} records, {len(inv_strs_slot3)} inv_strs entries")
