import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp
import gym, gymnasium, gymnasium.spaces.dict as _g
_g.Space = (gymnasium.spaces.Space, gym.spaces.Space)
from Nethax.minihax.minihax_env import MinihaxEnv
from Nethax.nethax.env import _make_restricted_step_impl
from Nethax.nethax.constants.tiles import TileType

ORDS = (107, 108, 106, 104, 117, 121, 110, 98)  # k j l h u y n b
NAMES = ["k", "j", "l", "h", "u", "y", "n", "b"]
rstep = _make_restricted_step_impl(ORDS, True)
env = MinihaxEnv("MiniHack-Room-Monster-5x5-v0")
s, info = env.reset(jax.random.key(3))
KEY = jax.random.key(0)
b = int(s.dungeon.current_branch); lv = int(s.dungeon.current_level) - 1
terr = np.asarray(s.terrain[b, lv])
p = tuple(int(x) for x in s.player_pos)
print("player", p, " down-stair",
      [(int(y), int(x)) for y, x in zip(*np.where(terr == int(TileType.STAIRCASE_DOWN)))])
# actual movement per action index
for i, nm in enumerate(NAMES):
    ns, *_ = rstep(s, jnp.int32(i), KEY)
    np2 = tuple(int(x) for x in ns.player_pos)
    print(f"  idx {i} '{nm}': {p} -> {np2}  d={(np2[0]-p[0], np2[1]-p[1])}")
# print the walkable window around the player (FLOOR vs WALL)
FLOOR = int(TileType.FLOOR); WALL = int(TileType.WALL)
r0, c0 = p
print("\nterrain window (@=player, .=floor, #=wall, >=downstair, ?=other):")
for r in range(r0 - 4, r0 + 5):
    row = ""
    for c in range(c0 - 6, c0 + 7):
        if 0 <= r < terr.shape[0] and 0 <= c < terr.shape[1]:
            t = int(terr[r, c])
            ch = "@" if (r, c) == p else (">" if t == int(TileType.STAIRCASE_DOWN)
                  else "." if t == FLOOR else "#" if t == WALL else "?")
        else:
            ch = " "
        row += ch
    print("  " + row)
