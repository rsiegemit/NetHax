"""One-time script: convert minihack tiles.pkl to numpy arrays for JAX.

Usage: python -m tiles.convert_tiles

Reads: minihack-main/minihack/tiles/tiles.pkl
Writes: tiles/tiles.npy  (shape [num_tiles, 16, 16, 3], uint8)
"""
import pickle
import numpy as np
import os

def convert():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    pkl_path = os.path.join(repo_root, "minihack-main", "minihack", "tiles", "tiles.pkl")
    out_path = os.path.join(script_dir, "tiles.npy")

    with open(pkl_path, "rb") as f:
        tiles_dict = pickle.load(f)

    # tiles_dict maps tile_id -> ndarray(16, 16, 3)
    max_tile_id = max(tiles_dict.keys())
    num_tiles = max_tile_id + 1

    tiles_array = np.zeros((num_tiles, 16, 16, 3), dtype=np.uint8)
    for tile_id, tile_img in tiles_dict.items():
        tiles_array[tile_id] = tile_img

    np.save(out_path, tiles_array)
    print(f"Saved {num_tiles} tiles to {out_path} (shape: {tiles_array.shape})")

if __name__ == "__main__":
    convert()
