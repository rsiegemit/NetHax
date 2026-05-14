# Wave 2 — Legacy code removal

Wave 1 was additive — it left every pre-existing file in `Nethax/nethax/` untouched. Wave 2 deleted that legacy code now that the new package fully subsumes it.

## What was deleted

| Path | What it was | Replaced by |
|---|---|---|
| `Nethax/nethax/nethax_state.py` | Old `EnvState` definition with `Item`, `Monsters`, `EnvParams`, `StaticEnvParams` | `Nethax/nethax/state.py` (new master `EnvState` + `StaticParams`) and `Nethax/nethax/subsystems/inventory.py` (now hosts `Item` directly) |
| `Nethax/nethax/game_logic.py` | Old step function with ~25 TODOs (do_melee_attack stub, is_game_over) | `Nethax/nethax/env.py::NethaxEnv.step` → `Nethax/nethax/subsystems/action_dispatch.py::dispatch_action` |
| `Nethax/nethax/renderer.py` | Old symbolic / text renderer using old `EnvState` | `Nethax/nethax/obs/{nle_obs,symbolic_obs,pixel_obs,text_obs}.py` |
| `Nethax/nethax/constants.py` (top-level file) | Old constants (`TileType`, `OPAQUE_TILES`, `RES_*`, `CONV_*`, etc.) | `Nethax/nethax/constants/` package — `tiles.py` (new) for `TileType` / `NUM_TILE_TYPES` / `SOLID_TILES` / `OPAQUE_TILES` |
| `Nethax/nethax/world_gen/` (directory) | Pre-Wave-1 procedural-gen utilities, all depending on old `EnvState` | `Nethax/nethax/dungeon/` (rooms, corridors, mazes, branches, level_memory) |
| `Nethax/nethax/envs/` (directory) | Old `nethax_symbolic_env.py` Gymnax wrapper | `Nethax/nethax/env.py::NethaxEnv` |
| `Nethax/nethax/util/` (directory) | Old `game_logic_utils.py` etc. | `Nethax/nethax/{fov,rng,save_load}.py` + subsystem helper functions |

## What was moved (not deleted)

| Path | Action |
|---|---|
| `Nethax/nethax/play_nethax.py` (legacy pygame UI, 285 lines) | Moved to `scripts/legacy/play_nethax.py`. **Currently broken** — imports `nethax_state.EnvParams`, `game_logic.nethax_step`, etc. that no longer exist. A future wave will rewrite it against `NethaxEnv`. |

## What was edited

| Path | Edit |
|---|---|
| `Nethax/nethax/subsystems/inventory.py` | Removed `from Nethax.nethax.nethax_state import Item`; defined `Item` Flax struct inline. |
| `Nethax/nethax/constants/__init__.py` | Removed the importlib bridge to legacy `constants.py`; now imports from new `tiles.py` directly. |

## What was NOT touched

- `Nethax/minihax/` — entire MiniHack curriculum package, untouched. It still uses its own state model and works independently. Wave 3 will migrate `minihax` to use the new env / state.
- `Nethax/environment_base/` — shared Gymnax-style base class. Unchanged.
- `Nethax/tiles/` — sprite atlas + GLYPH2TILE lookup. Wave 2 *consumed* this but didn't modify it.
- `reference/` — markdown spec extracts. Unchanged.
- `scripts/` — regression scripts, video generators, paper figures. Mostly unchanged (added `scripts/legacy/`).

## Verification

The legacy migration kept all 173 Wave 2 tests passing:
```
$ .venv/bin/python -m pytest tests/ -q
............... 173 passed in 14.34s
```

The only currently-broken code is `scripts/legacy/play_nethax.py`, which is deliberately not run anywhere — it's preserved as a reference for the future pygame rewrite.

## Why this matters

After Wave 2, every file under `Nethax/nethax/` is part of the new architecture. There's no "legacy island" to dance around. Wave 3+ subsystem implementers can:

- Read `Nethax/nethax/state.py` and know it's authoritative.
- Use `EnvState` slices without worrying about old-vs-new duality.
- Trust that `from Nethax.nethax.constants import TileType` resolves to the canonical file (not a synthetic re-export).
- Delete or refactor any file without checking for legacy callers.
