"""Generate videos of our Nethax minihax implementations using praxis conda env.

Renders pixel observations from our envs and saves as mp4 videos.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/renos/nethax')

import jax
import jax.numpy as jnp
import numpy as np

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from Nethax.nethax_env import make_minihax_env_from_name


def generate_video(env_name, output_path, num_steps=50, seed=42):
    """Generate a full-map video of one of our minihax environments."""
    print(f"  Generating {env_name}...", flush=True)

    try:
        env = make_minihax_env_from_name(env_name)
    except Exception as e:
        print(f"  SKIP {env_name}: {e}")
        return False

    params = env.default_params
    rng = jax.random.PRNGKey(seed)

    rng, rng_reset = jax.random.split(rng)
    obs, state = env.reset(rng_reset, params)

    frames = []

    # Convert obs to image frame
    obs_np = np.array(obs)
    if obs_np.ndim == 3 and obs_np.shape[-1] == 3:
        # Pixel obs: already an image
        frames.append(obs_np)
    else:
        # Symbolic obs: skip (can't render as image directly)
        print(f"  SKIP {env_name}: symbolic obs, use Pixels variant")
        return False

    for step in range(num_steps):
        rng, rng_step = jax.random.split(rng)
        action = jax.random.randint(rng, (), 0, env.num_actions)
        obs, state, reward, done, info = env.step(rng_step, state, action, params)

        obs_np = np.array(obs)
        frames.append(obs_np)

        # Reset if done
        if bool(done):
            rng, rng_reset = jax.random.split(rng)
            obs, state = env.reset(rng_reset, params)
            obs_np = np.array(obs)
            frames.append(obs_np)

    # Save as mp4
    if HAS_IMAGEIO:
        imageio.mimsave(output_path, frames, fps=5)
    elif HAS_CV2:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 5, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
    else:
        print(f"  No video library! Saving frames as npz.")
        np.savez(output_path.replace('.mp4', '.npz'), frames=np.stack(frames))
        return True

    print(f"  Saved {output_path} ({len(frames)} frames)")
    return True


# Our pixel env names matching the originals we're comparing against
OUR_ENVS = {
    # Tier 1 Navigation
    'Corridor2': 'Minihax-Corridor2-Pixels-v0',
    'Corridor3': 'Minihax-Corridor3-Pixels-v0',
    'Corridor5': 'Minihax-Corridor5-Pixels-v0',
    'Mazewalk': 'Minihax-Mazewalk-Pixels-v0',
    'ExploreMazeEasy': 'Minihax-ExploreMazeEasy-Pixels-v0',
    'ExploreMazeEasyPremapped': 'Minihax-ExploreMazeEasyPremapped-Pixels-v0',
    'ExploreMazeHard': 'Minihax-ExploreMazeHard-Pixels-v0',
    'ExploreMazeHardPremapped': 'Minihax-ExploreMazeHardPremapped-Pixels-v0',
    # Tier 2 Hazards
    'LavaCrossing': 'Minihax-LavaCrossing-Pixels-v0',
    'HideNSeek': 'Minihax-HideNSeek-Pixels-v0',
    'HideNSeekBig': 'Minihax-HideNSeekBig-Pixels-v0',
    'HideNSeekLava': 'Minihax-HideNSeekLava-Pixels-v0',
    'HideNSeekMapped': 'Minihax-HideNSeekMapped-Pixels-v0',
    'QuestEasy': 'Minihax-QuestEasy-Pixels-v0',
    'LockedDoor': 'Minihax-LockedDoor-Pixels-v0',
    'LockedDoorFixed': 'Minihax-LockedDoorFixed-Pixels-v0',
    # Tier 3 Combat
    'QuestHard': 'Minihax-QuestHard-Pixels-v0',
    'ClosedDoor': 'Minihax-ClosedDoor-Pixels-v0',
    'MementoEasy': 'Minihax-MementoEasy-Pixels-v0',
    'MementoShort': 'Minihax-MementoShort-Pixels-v0',
    'MementoHard': 'Minihax-MementoHard-Pixels-v0',
    # Tier 4 Sokoban
    'Soko1a': 'Minihax-Soko1a-Pixels-v0',
    'Soko1b': 'Minihax-Soko1b-Pixels-v0',
    'Soko2a': 'Minihax-Soko2a-Pixels-v0',
    'Soko2b': 'Minihax-Soko2b-Pixels-v0',
    'Soko3a': 'Minihax-Soko3a-Pixels-v0',
    'Soko3b': 'Minihax-Soko3b-Pixels-v0',
    'Soko4a': 'Minihax-Soko4a-Pixels-v0',
    'Soko4b': 'Minihax-Soko4b-Pixels-v0',
}


def main():
    output_dir = '/home/renos/nethax/videos/ours'
    os.makedirs(output_dir, exist_ok=True)

    # If specific env passed as arg, only generate that one
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target in OUR_ENVS:
            mapping = {target: OUR_ENVS[target]}
        else:
            print(f"Unknown env: {target}")
            return
    else:
        mapping = OUR_ENVS

    ok, fail = 0, 0
    for our_name, env_name in mapping.items():
        output_path = os.path.join(output_dir, f'ours_{our_name}.mp4')
        success = generate_video(env_name, output_path, num_steps=50)
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} OK, {fail} FAIL")


if __name__ == '__main__':
    main()
