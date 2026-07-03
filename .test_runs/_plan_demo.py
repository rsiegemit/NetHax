"""DEMO 2: depth-limited lookahead SEARCH driving real gameplay.

Receding-horizon planner: each turn, fork the current state into every
depth-D action sequence (8^D) with one vmap/scan, score each leaf by distance
to the level's down-staircase, take the first action of the best sequence, then
replan from the new state. This is model-predictive control / the MCTS inner
loop — on the real NetHack engine, only possible because the state is a
snapshot-able, forkable pytree.

Run: JAX_PLATFORMS=cpu NETHAX_SINGLE_LEVEL=1 NETHAX_MAX_MONSTERS=8 \
     NETHAX_VEC_MONSTERS=1 NETHAX_FAST_POST=1 .venv/bin/python .test_runs/_plan_demo.py
"""
import os, itertools
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp
import gym, gymnasium, gymnasium.spaces.dict as _g
_g.Space = (gymnasium.spaces.Space, gym.spaces.Space)
from Nethax.minihax.minihax_env import MinihaxEnv
from Nethax.nethax.env import _make_restricted_step_impl
from Nethax.nethax.constants.tiles import TileType

DIRS = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
rstep = _make_restricted_step_impl((107, 108, 106, 104, 117, 121, 110, 98), True)
DEPTH = 3
SEQS = jnp.asarray(list(itertools.product(range(8), repeat=DEPTH)), dtype=jnp.int32)  # [8^D, D]
KEY = jax.random.key(0)

env = MinihaxEnv("MiniHack-Room-Monster-5x5-v0")
state, info = env.reset(jax.random.key(3))

# Goal = down-staircase tile on the current level.
b = int(state.dungeon.current_branch); lv = int(state.dungeon.current_level) - 1
terr = np.asarray(state.terrain[b, lv])
ys, xs = np.where(terr == int(TileType.STAIRCASE_DOWN))
if len(ys):
    goal = (int(ys[0]), int(xs[0]))
else:
    goal = (int(state.player_pos[0]) + 6, int(state.player_pos[1]) + 6)  # fallback
goal_j = jnp.asarray(goal, jnp.int32)
print(f"start {tuple(int(x) for x in state.player_pos)}  ->  goal(down-stair) {goal}\n")


def _rollout_value(s_cur, seq):
    # Dense value: accumulate -distance at EVERY step of the rollout, so a plan
    # that makes progress EARLY beats one that wastes its first move (both may
    # reach the goal by depth D, but only the early-progress plan's first action
    # is worth executing under receding-horizon control).
    def body(carry, a):
        st, acc = carry
        ns, obs, r, d = rstep(st, a, KEY)
        dist = jnp.maximum(jnp.abs(ns.player_pos[0] - goal_j[0]),
                           jnp.abs(ns.player_pos[1] - goal_j[1]))
        return (ns, acc - dist.astype(jnp.int32)), dist
    (final, acc), dists = jax.lax.scan(body, (s_cur, jnp.int32(0)), seq)
    return acc, dists[-1]                    # (dense value, final dist)


@jax.jit
def plan(s_cur):
    # Fork into ALL 8^DEPTH futures in one kernel; dense-value each leaf.
    vals, finals = jax.vmap(lambda seq: _rollout_value(s_cur, seq))(SEQS)  # [8^D],[8^D]
    best = jnp.argmax(vals)
    return SEQS[best, 0], finals[best]       # first action of best plan, its final dist


traj = [tuple(int(x) for x in state.player_pos)]
reached = False
for t in range(15):
    prev = tuple(int(x) for x in state.player_pos)
    a, final_dist = plan(state)
    state, obs, r, d = rstep(state, a, KEY)
    pos = tuple(int(x) for x in state.player_pos)
    traj.append(pos)
    dcur = max(abs(pos[0] - goal[0]), abs(pos[1] - goal[1]))
    print(f"  turn {t:2d}: search picks action {int(a)} (move {prev}->{pos})  "
          f"dist-to-goal {dcur}  [best {DEPTH}-ply plan ends at dist {int(final_dist)}]")
    if pos == goal:
        print(f"\n  >>> REACHED the down-staircase in {t+1} planned turns. <<<")
        reached = True
        break
if not reached:
    print(f"\n  (stopped after 15 turns at {traj[-1]}, goal {goal})")

print(f"\ntrajectory: {traj}")
print(f"\nEach turn evaluated 8^{DEPTH} = {8**DEPTH} action sequences by forking the "
      f"REAL engine state — the MCTS/model-predictive inner loop NLE cannot run.")
