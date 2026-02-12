"""Per-tier BLStats normalization vectors for SymbolicGlyphNet.

Each tuple has 27 elements matching BLSTATS_SIZE in nle_obs.py.
Non-zero entries correspond to blstats fields actually populated
by the tier's build_blstats_*() function. Values are 1/max_expected_value.

Index reference:
  0:x  1:y  2:str25  3:str125  4:dex  5:con  6:int  7:wis  8:cha
  9:score  10:hp  11:hpmax  12:depth  13:gold  14:ene  15:enemax
  16:ac  17:hd  18:xp_level  19:xp  20:time  21:hunger  22:cap
  23:dnum  24:dlevel  25:pits_remaining  26:monsters_killed
"""
from Nethax.minihax.constants import (
    NUM_ACTIONS_TIER1, NUM_ACTIONS_TIER2, NUM_ACTIONS_TIER3, NUM_ACTIONS_TIER4,
)

# Tier 1: Navigation (map 21x79, hp=16 placeholder, max_timesteps~200)
BLSTATS_NORM_NAVIGATION = (
    1.0/79, 1.0/21,                     # x, y
    0, 0, 0, 0, 0, 0, 0,                # str..cha (unused)
    0,                                    # score
    1.0/16, 1.0/16,                      # hp, hpmax
    0, 0, 0, 0, 0, 0, 0, 0,             # depth..xp
    1.0/200,                             # time
    0, 0, 0, 0,                          # hunger..dlevel
    0, 0,                                # pits_remaining, monsters_killed
)

# Tier 2: Hazard (map 10x38, hp=16, max_timesteps~200)
BLSTATS_NORM_HAZARD = (
    1.0/38, 1.0/10,                      # x, y
    0, 0, 0, 0, 0, 0, 0,
    0,
    1.0/16, 1.0/16,                      # hp, hpmax
    0, 0, 0, 0, 0, 0, 0, 0,
    1.0/200,                             # time
    0, 0, 0, 0,
    0, 0,
)

# Tier 3: Combat (map 15x80, full combat stats)
BLSTATS_NORM_COMBAT = (
    1.0/80, 1.0/15,                      # x, y
    1.0/18, 1.0/18,                      # str25, str125
    0, 0, 0, 0, 0,                       # dex..cha (unused)
    1.0/1000,                            # score
    1.0/50, 1.0/50,                      # hp, hpmax
    0, 0, 0, 0,                          # depth, gold, ene, enemax
    1.0/10, 0,                           # ac, hd
    1.0/30, 1.0/1000,                    # xp_level, xp
    1.0/1000,                            # time
    0, 0, 0, 0,                          # hunger..dlevel
    0, 1.0/17,                           # pits_remaining, monsters_killed
)

# Tier 4: Sokoban (map 18x30, hp=16 placeholder, pits_remaining up to 8)
BLSTATS_NORM_SOKOBAN = (
    1.0/30, 1.0/18,                      # x, y
    0, 0, 0, 0, 0, 0, 0,
    0,
    1.0/16, 1.0/16,                      # hp, hpmax
    0, 0, 0, 0, 0, 0, 0, 0,
    1.0/1000,                            # time
    0, 0, 0, 0,
    1.0/8, 0,                            # pits_remaining
)


def get_encoder_config(tier_name):
    """Return (blstats_norm, num_actions) for a tier name.

    Args:
        tier_name: One of 'navigation', 'hazard', 'combat', 'sokoban'

    Returns:
        Tuple of (blstats_norm_tuple, num_actions_int)
    """
    configs = {
        'navigation': (BLSTATS_NORM_NAVIGATION, NUM_ACTIONS_TIER1),
        'hazard': (BLSTATS_NORM_HAZARD, NUM_ACTIONS_TIER2),
        'combat': (BLSTATS_NORM_COMBAT, NUM_ACTIONS_TIER3),
        'sokoban': (BLSTATS_NORM_SOKOBAN, NUM_ACTIONS_TIER4),
    }
    if tier_name not in configs:
        raise ValueError(f"Unknown tier: {tier_name}. Expected one of {list(configs.keys())}")
    return configs[tier_name]
