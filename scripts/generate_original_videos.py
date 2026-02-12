"""Generate reference videos of original MiniHack environments using sol3 conda env.

Uses the 'pixel' observation key for proper sprite-based tile rendering
(same NetHack tileset used in the game).
"""
import gymnasium
import minihack
import numpy as np
import os
import sys

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


def generate_video(env_name, output_path, num_steps=100, seed=42):
    """Generate a full-map video of an original MiniHack environment."""
    print(f"  Generating {env_name}...", flush=True)

    try:
        env = gymnasium.make(
            env_name,
            observation_keys=('pixel',),
            max_episode_steps=num_steps + 10,
        )
    except Exception as e:
        print(f"  SKIP {env_name}: {e}")
        return False

    frames = []
    obs, info = env.reset(seed=seed)
    frame = obs['pixel']
    frames.append(frame)

    rng = np.random.RandomState(seed)

    for step in range(num_steps):
        action = rng.randint(0, env.action_space.n)
        obs, reward, terminated, truncated, info = env.step(action)

        frame = obs['pixel']
        frames.append(frame)

        if terminated or truncated:
            obs, info = env.reset(seed=seed + step + 1)
            frame = obs['pixel']
            frames.append(frame)

    env.close()

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
        print(f"  No video library available! Saving frames as npz.")
        np.savez(output_path.replace('.mp4', '.npz'), frames=np.stack(frames))
        return True

    print(f"  Saved {output_path} ({len(frames)} frames, {frames[0].shape})")
    return True


# Mapping: our env name -> original MiniHack env name
ENV_MAPPING = {
    # Tier 1 Navigation
    'Corridor2': 'MiniHack-Corridor-R2-v0',
    'Corridor3': 'MiniHack-Corridor-R3-v0',
    'Corridor5': 'MiniHack-Corridor-R5-v0',
    'Mazewalk': 'MiniHack-MazeWalk-9x9-v0',
    'ExploreMazeEasy': 'MiniHack-ExploreMaze-Easy-v0',
    'ExploreMazeEasyPremapped': 'MiniHack-ExploreMaze-Easy-Mapped-v0',
    'ExploreMazeHard': 'MiniHack-ExploreMaze-Hard-v0',
    'ExploreMazeHardPremapped': 'MiniHack-ExploreMaze-Hard-Mapped-v0',
    # Tier 2 Hazards
    'LavaCrossing': 'MiniHack-LavaCross-Full-v0',
    'HideNSeek': 'MiniHack-HideNSeek-v0',
    'HideNSeekBig': 'MiniHack-HideNSeek-Big-v0',
    'HideNSeekLava': 'MiniHack-HideNSeek-Lava-v0',
    'HideNSeekMapped': 'MiniHack-HideNSeek-Mapped-v0',
    'QuestEasy': 'MiniHack-Quest-Easy-v0',
    'LockedDoor': 'MiniHack-LockedDoor-v0',
    'LockedDoorFixed': 'MiniHack-LockedDoor-Fixed-v0',
    # Tier 3 Combat
    'QuestHard': 'MiniHack-Quest-Hard-v0',
    'ClosedDoor': 'MiniHack-ClosedDoor-v0',
    'MementoEasy': 'MiniHack-Memento-F2-v0',
    'MementoShort': 'MiniHack-Memento-Short-F2-v0',
    'MementoHard': 'MiniHack-Memento-F4-v0',
    # Tier 4 Sokoban
    'Soko1a': 'MiniHack-Sokoban1a-v1',
    'Soko1b': 'MiniHack-Sokoban1b-v1',
    'Soko2a': 'MiniHack-Sokoban2a-v1',
    'Soko2b': 'MiniHack-Sokoban2b-v1',
    'Soko3a': 'MiniHack-Sokoban3a-v1',
    'Soko3b': 'MiniHack-Sokoban3b-v1',
    'Soko4a': 'MiniHack-Sokoban4a-v1',
    'Soko4b': 'MiniHack-Sokoban4b-v1',
}


def main():
    output_dir = '/home/renos/nethax/videos/originals'
    os.makedirs(output_dir, exist_ok=True)

    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target in ENV_MAPPING:
            mapping = {target: ENV_MAPPING[target]}
        else:
            print(f"Unknown env: {target}")
            return
    else:
        mapping = ENV_MAPPING

    ok, fail = 0, 0
    for our_name, orig_name in mapping.items():
        output_path = os.path.join(output_dir, f'original_{our_name}.mp4')
        success = generate_video(orig_name, output_path, num_steps=50)
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} OK, {fail} FAIL")


if __name__ == '__main__':
    main()
