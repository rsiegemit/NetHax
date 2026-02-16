"""Full regression test for all minihax environments."""
import sys
import jax
from Nethax.nethax_env import make_minihax_env_from_name

# All symbolic envs
SYMBOLIC = [
    # Tier 1 Navigation
    'Minihax-Corridor2-v0','Minihax-Corridor3-v0','Minihax-Corridor4-v0',
    'Minihax-Corridor5-v0','Minihax-Corridor6-v0','Minihax-Corridor7-v0',
    'Minihax-Corridor8-v0','Minihax-Corridor9-v0','Minihax-Corridor10-v0',
    'Minihax-Mazewalk-v0',
    # Tier 2 Hazard
    'Minihax-LavaCrossing-v0','Minihax-HideNSeek-v0','Minihax-HideNSeekBig-v0',
    'Minihax-HideNSeekLava-v0','Minihax-HideNSeekMapped-v0',
    'Minihax-QuestEasy-v0','Minihax-QuestMedium-v0','Minihax-LockedDoor-v0',
    'Minihax-LockedDoorFixed-v0',
    # Tier 3 Combat
    'Minihax-ZombieHorde-Symbolic-v0','Minihax-Quest-v0','Minihax-QuestHard-v0',
    'Minihax-KeyAndDoor-v0','Minihax-ClosedDoor-v0','Minihax-Chest-v0',
    'Minihax-MementoEasy-v0','Minihax-MementoShort-v0','Minihax-MementoHard-v0',
    # Tier 4 Sokoban
    'Minihax-Soko1a-v0','Minihax-Soko1b-v0','Minihax-Soko2a-v0','Minihax-Soko2b-v0',
    'Minihax-Soko3a-v0','Minihax-Soko3b-v0','Minihax-Soko4a-v0','Minihax-Soko4b-v0',
]

# Pixel envs
PIXELS = [
    'Minihax-ZombieHorde-Pixels-v0',
    'Minihax-KeyAndDoor-Pixels-v0','Minihax-ClosedDoor-Pixels-v0',
    'Minihax-Chest-Pixels-v0','Minihax-MementoShort-Pixels-v0',
    'Minihax-MementoHard-Pixels-v0','Minihax-QuestHard-Pixels-v0',
    'Minihax-QuestEasy-Pixels-v0','Minihax-QuestMedium-Pixels-v0',
    'Minihax-LockedDoor-Pixels-v0',
    'Minihax-HideNSeek-Pixels-v0','Minihax-HideNSeekBig-Pixels-v0',
]

# NLE envs
NLE = [
    'Minihax-KeyAndDoor-NLE-v0','Minihax-ClosedDoor-NLE-v0',
    'Minihax-MementoHard-NLE-v0','Minihax-ZombieHorde-NLE-v0',
]

failed = []
passed = 0

for name in SYMBOLIC + PIXELS + NLE:
    try:
        env = make_minihax_env_from_name(name)
        obs, state = env.reset(jax.random.PRNGKey(42), env.default_params)
        obs2, state2, r, d, info = env.step(jax.random.PRNGKey(0), state, 0, env.default_params)
        print(f'OK {name}: r={float(r):.3f}, d={bool(d)}')
        passed += 1
    except Exception as e:
        print(f'FAIL {name}: {e}')
        failed.append((name, str(e)))

if failed:
    print(f'\n{len(failed)} FAILED:')
    for name, err in failed:
        print(f'  {name}: {err}')
else:
    print(f'\nALL {passed} PASSED')
