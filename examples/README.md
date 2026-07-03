# Examples — capabilities NLE cannot offer

Nethax runs NetHack as a pure-functional JAX **pytree**, not an opaque running C
process. That makes the game state **snapshot-able, forkable, rewindable, and
differentiable** — the basis for planning / search / model-based methods on the real
game. NLE (the C engine) is faster per step (see [`../docs/benchmark.md`](../docs/benchmark.md)),
but *cannot* do any of the below, because you can't cheaply clone or rewind a live C
game.

Both examples run on CPU in seconds at reduced scope:

```bash
JAX_PLATFORMS=cpu NETHAX_SINGLE_LEVEL=1 NETHAX_MAX_MONSTERS=8 \
  NETHAX_VEC_MONSTERS=1 NETHAX_FAST_POST=1 \
  PYTHONPATH=. .venv/bin/python examples/state_forking.py
```

## `state_forking.py` — snapshot / fork / rewind
From one root state: fork into all 8 action branches in a single `vmap` (8 divergent
futures, blocked moves detected), confirm the root is unchanged afterward (free
rewind), build an 8×8 = 64-leaf two-ply tree in one kernel, and pick the best action
by parallel one-ply lookahead.

## `lookahead_planner.py` — receding-horizon search driving real gameplay
A model-predictive planner: each turn it forks the current state into **all 8³ = 512
three-move futures** (one `vmap`/`scan`), scores each by distance to the level's
down-staircase, executes the first action of the best plan, then replans. It walks the
hero to the staircase in the minimum number of moves — an MCTS/MPC inner loop on the
real NetHack engine.

Both are themselves `jit`-compiled and `vmap` over a batch of games — search *and*
batching together.
