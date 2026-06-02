"""Dump Nethax terrain + ground_items at the seed=4 Bug B cell (9, 9)
and several nearby cells to characterise what's at (9, 9) and what
upstream code path put it there.

Usage:
    JAX_COMPILATION_CACHE_DIR=$HOME/.cache/nethax_jax \\
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \\
    JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 \\
    JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python .test_runs/diag_seed4_bugb.py
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/Users/rsiegelmann/Downloads/Projects/nethax")
sys.path.insert(0, "/Users/rsiegelmann/Downloads/Projects/nethax/tests")

import jax
import jax.numpy as jnp
import numpy as np

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.parity_mode import set_parity_mode, ParityMode
set_parity_mode(ParityMode.NLE_BYTEPARITY)

# Map TileType values to names for readability.
_TT = {int(v): k for k, v in TileType.__members__.items() if isinstance(v, int) or hasattr(v, "value")}
# Fallback: introspect enum
try:
    _TT = {int(t.value): t.name for t in TileType}
except Exception:
    pass


def main():
    env = NethaxEnv()
    rng = jax.random.PRNGKey(4)
    state, obs = env.reset(rng)
    terrain = np.asarray(state.terrain[0, 0])  # branch=0, level=0 [H, W]
    chars = np.asarray(obs["chars"])    # [21, 79]
    glyphs = np.asarray(obs["glyphs"])  # [21, 79]
    print(f"OBS chars[9, 7..12]: {chars[9, 7:13].tolist()}")
    print(f"OBS glyphs[9, 7..12]: {glyphs[9, 7:13].tolist()}")
    print(f"NLE glyph 2359 = S_stone; 2371 = ?")
    print()
    print(f"terrain shape={terrain.shape}  dtype={terrain.dtype}")
    print(f"terrain[9, 8..12]:")
    for c in range(8, 13):
        v = int(terrain[9, c])
        name = _TT.get(v, f"?{v}")
        print(f"  [9,{c}] = {v} ({name})")
    print()
    print(f"terrain[8..10, 7..11]:")
    for r in range(8, 11):
        row = " ".join(f"{int(terrain[r, c]):3d}" for c in range(7, 12))
        print(f"  r={r}: {row}")
    print()

    # ground_items for that level at row=9
    gi = state.ground_items
    cat = np.asarray(gi.category[0, 0])  # [H, W, MAX_STACK]
    typ = np.asarray(gi.type_id[0, 0])
    print(f"ground_items at (9, 7..12):")
    for c in range(7, 13):
        cats = cat[9, c].tolist()
        if any(cats):
            ts = typ[9, c].tolist()
            print(f"  [9,{c}] cats={cats}  type_ids={ts}")

    print()
    # Player position
    print(f"player_pos = {tuple(int(x) for x in state.player_pos)}")

    # Dump the entire level row 8-10 to find rooms
    print()
    print("Row 9 full (col 0..78):")
    print("  " + "".join("." if int(terrain[9, c]) in (1, 2) else
                          "#" if int(terrain[9, c]) == 7 else  # CORRIDOR
                          "+" if int(terrain[9, c]) == 23 else  # DOORWAY
                          "|" if int(terrain[9, c]) == 14 else  # VWALL
                          "-" if int(terrain[9, c]) == 15 else  # HWALL
                          str(int(terrain[9, c]) % 10) for c in range(0, 79)))


if __name__ == "__main__":
    main()
