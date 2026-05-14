# Wave 1 — Foundation & Breadth-First Scaffolding

**Status:** ✅ Complete · 103 pytest tests passing · NLE-parity API surface in place.

Wave 1's mission was to **erect the entire scaffold of NetHack 5.0 / 3.7-branch in JAX, in breadth**. Every subsystem the real game depends on now has a typed Flax state slice, a no-op step function with the right signature, vendor source citations, and a TODO list pointing the way to later waves.

We deliberately did **not** implement game mechanics in Wave 1. Combat doesn't roll dice, dungeons don't generate, monsters don't move. What we built is the *shape* of the game — every pytree slot, every enum, every API contract — so later waves can fill mechanics in place without re-architecting.

---

## What shipped

| Area | Files | Lines | Status |
|---|---|---|---|
| `Nethax/nethax/constants/` | 8 | ~1,632 | NLE-parity enums (Action, Glyph offsets, BLStats, Role, Race) + monster/object data schemas |
| `Nethax/nethax/subsystems/` | 17 | ~2,300 | One module per subsystem — combat, magic, monster_ai, polymorph, inventory, items, identification, traps, features, prayer, conduct, shop, quest, status_effects, scoring, messages, action_dispatch |
| `Nethax/nethax/dungeon/` | 7 | ~947 | rooms, mazes, corridors, branches (Main/Mines/Sokoban/Quest/Vlad/Gehennom/Endgame), special_levels (28 named levels), level_memory |
| `Nethax/nethax/obs/` | 5 | ~358 | NLE-parity 17-key observation builder + symbolic / pixel / text variants |
| `Nethax/nethax/` (top-level new) | 5 | ~430 | `state.py` (master EnvState), `env.py` (NethaxEnv class), `fov.py`, `rng.py`, `save_load.py`, refreshed `__init__.py` |
| `tests/` | 9 | ~407 | pytest suite — imports, action enum, blstats layout, glyph offsets, NLE obs, state-slice construction, no-op step idempotency |
| `docs/wave-1/` | 9 | this set | Wave 1 documentation |

Total Wave 1 footprint: **~50 new Python modules, ~6,000 lines of code, 103 passing tests**.

---

## How to use

### Install + test (already wired)

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m pytest
```

### Smoke run the master env

```python
import jax
from Nethax.nethax import NethaxEnv

env = NethaxEnv()
state, obs = env.reset(jax.random.PRNGKey(0))
# `obs` is a 17-key NLE-compatible dict: glyphs (21,79) int16,
# blstats (27,) int64, message (256,) uint8, tty_chars (24,80) uint8, ...

state, obs, reward, done, info = env.step(state, action=0, rng=jax.random.PRNGKey(1))
# Wave 1: reward=0, done=False, state unchanged except timestep+=1.

step_jit = jax.jit(env.step)   # the whole step pipeline JITs cleanly.
```

---

## The doc set

| # | File | What it covers |
|---|---|---|
| 1 | [`README.md`](README.md) | This file — overview + TOC |
| 2 | [`architecture.md`](architecture.md) | Package layout, module dependency graph, design patterns |
| 3 | [`subsystems.md`](subsystems.md) | Every subsystem stub with state class, step fn signature, vendor citations, Wave-to-implement |
| 4 | [`nle-parity.md`](nle-parity.md) | NLE observation, action, blstats, glyph compatibility status with verification |
| 5 | [`state-schema.md`](state-schema.md) | Walkthrough of the master `EnvState` pytree |
| 6 | [`decisions.md`](decisions.md) | Design decisions taken this wave + their tradeoffs |
| 7 | [`gaps.md`](gaps.md) | Aggregated TODO list across all stubs — the game-design backlog |
| 8 | [`test-results.md`](test-results.md) | pytest output, coverage matrix, what's verified vs. asserted-only |
| 9 | [`next-wave.md`](next-wave.md) | Wave 2 scope preview |

---

## What's *not* in Wave 1 (deliberately)

- **No mechanics**: combat doesn't compute hit/damage, magic doesn't cast, traps don't trigger, monsters don't move, dungeon doesn't generate. All `step()` functions are no-ops.
- **No level memory wired**: `LevelMemoryState` exists but is not yet plumbed into the player's descent/ascent loop.
- **No real observation projection**: `build_nle_observation(state)` returns zero arrays of the correct shapes. Wave 2 wires the projection.
- **No agent baselines**: no random walker, no scripted agent. Wave 2 adds one.
- **No MiniHack envs**: Wave 5 ports the LevelGenerator + RewardManager + 170-env catalog.

These are documented in [`next-wave.md`](next-wave.md) and [`gaps.md`](gaps.md).
