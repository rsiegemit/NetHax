"""List all MiniHack environments available in the sol3 conda env."""
import gym
import minihack
envs = [e for e in gym.envs.registry.keys() if 'MiniHack' in e]
for e in sorted(envs):
    print(e)
