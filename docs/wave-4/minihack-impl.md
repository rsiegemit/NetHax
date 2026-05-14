# Wave 4 — MiniHack Implementation

The single largest Wave 4 deliverable: a `Nethax.minihax` package that exposes every canonical MiniHack `env_id` on top of `NethaxEnv`.

## Package layout

```
Nethax/minihax/
├── __init__.py
├── level_generator.py      ← procedural EnvState builder (LG API)
├── reward_manager.py       ← event-based reward shaping (RM API)
├── des_parser.py           ← parser/compiler for vendor *.des files
├── registry.py             ← (env_id) → EnvSpec
├── minihax_env.py          ← MinihaxEnv class (reset/step wrapper)
└── envs/canonical.py       ← 159 builder functions + register_all()
```

## `LevelGenerator`

The `LevelGenerator(w, h, fill, lit)` accumulates declarative `add_*` / `set_*` calls and produces a `(rng) -> EnvState` factory at the end:

```python
from Nethax.minihax.level_generator import LevelGenerator

lg = LevelGenerator(w=5, h=5, fill=".", lit=True)
lg.set_start_pos(0, 0)
lg.add_stair_down(x=4, y=4)
lg.add_monster()                # any monster, random valid floor tile
lg.add_trap()                   # any trap, random valid floor tile
factory = lg.get_factory()

state = factory(jax.random.PRNGKey(0))   # builds a populated EnvState
```

API surface (Wave 4 implementation):

| Method | Purpose |
|---|---|
| `add_room(x, y, w, h, fill, lit)` | Carve a rectangular room. |
| `add_corridor(start, end)` | Connect two points via L-shape. |
| `add_door(x, y, kind)` | Drop a door tile. |
| `add_monster(name=None, x=None, y=None)` | Place a monster (random valid tile if pos unspecified). |
| `add_trap(kind=None, x=None, y=None)` | Place a trap (random kind / position if unspecified). |
| `add_object(name, x=None, y=None)` | Drop an object on ground. |
| `add_stair_up(x=None, y=None)` / `add_stair_down(...)` | Place staircases. |
| `set_start_pos(x, y)` | Where the player spawns. |
| `set_goal_pos(x, y)` | Mark goal tile (used by RewardManager `add_location_event`). |
| `fill_terrain(rect, tile)` | Bulk-fill a region. |
| `get_factory()` | Compile to a `(rng) -> EnvState` closure. |

The factory closure builds the level deterministically given an rng; random elements (monster placement, trap kinds) consume rng splits.

## `RewardManager`

Event-based reward shaping with a per-event `(reward, terminal_sufficient, terminal_required, repeatable)` tuple. Multiple events can be active simultaneously; per-step evaluation compares previous-vs-current `EnvState` and fires events accordingly.

```python
from Nethax.minihax.reward_manager import RewardManager

rm = RewardManager()
rm.add_location_event("stairs_down", reward=1.0,
                      terminal_sufficient=True, terminal_required=True)
rm.add_kill_event(monster_name="dwarf",   reward=0.5, repeatable=True)
rm.add_pickup_event(object_name="apple", reward=0.1)
rm.add_message_event("you find a hidden passage", reward=0.05)
```

Event types implemented in Wave 4:

| Factory | Fires when |
|---|---|
| `add_eat_event(food_name, ...)` | Player eats matching food (via FOOD-class inv consumption). |
| `add_kill_event(monster_name, ...)` | A monster of given type died this step. |
| `add_message_event(substring, ...)` | The message buffer contains `substring`. |
| `add_pickup_event(object_name, ...)` | Player picked up matching object. |
| `add_wield_event(weapon_name, ...)` | Player wielded matching weapon. |
| `add_wear_event(armor_name, ...)` | Player wears matching armor. |
| `add_amulet_event(...)` | Player puts on a matching amulet. |
| `add_levitate_event(...)` | Levitation status flips on. |
| `add_positional_event(rect, ...)` | Player enters a coordinate range. |
| `add_coordinate_event(x, y, ...)` | Player lands on a specific tile. |
| `add_location_event(named_tile, ...)` | Player lands on `"stairs_down"`, `"stairs_up"`, `"altar"`, `"fountain"`, etc. |
| `add_custom_reward_fn(fn, ...)` | Caller-supplied `(prev_state, new_state) -> float`. |

`compute_reward(prev_state, new_state, fired_mask) → (reward, done, new_fired)` is JIT-friendly: it scans the event table inside `lax.scan` and combines per-event triggers with the `repeatable` / `terminal_*` flags.

## des-file parser

`Nethax/minihax/des_parser.py` (2267 LoC) implements a parser + compiler for the 36 canonical `*.des` files under `vendor/minihack/minihack/dat/`. Parsed AST nodes:

- `MAZE`, `ROOM`, `OBJECT`, `MONSTER`, `STAIR`, `TRAP`, `DOOR`, `FOUNTAIN`, `ALTAR`
- random selections (`RANDOM`), conditional blocks (`IF`/`ELSE`), named regions
- coordinate references (`SELECTION`, `RECT`)
- LUA-style number/string literals (most `.des` files preface their LG actions with a Lua-flavored header)

Parser → emits a Python `LevelGenerator` builder. Used by some canonical env factories where hand-translation would have been verbose (Sokoban variants, MultiRoom).

Files parsed: all 36 of `vendor/minihack/minihack/dat/{room.des, corridor.des, lavacross.des, sokoban*.des, multiroom.des, quest.des, …}`.

## 159-env registry

Each env_id maps to an `EnvSpec(env_id, level_factory, reward_manager, max_steps, category)`. Built at module import.

| Category | Count | Examples |
|---|---|---|
| Skill | 45 | `MiniHack-Eat-Distract-v0`, `MiniHack-Wield-Distract-v0`, `MiniHack-Levitate-Boots-Fixed-v0`, … |
| MultiRoom | 24 | `MiniHack-MultiRoom-N2-v0` through `MiniHack-MultiRoom-N10-Lava-OpenDoor-v0` |
| LavaCross | 18 | `MiniHack-LavaCross-Full-v0`, `MiniHack-LavaCross-Levitate-Potion-Pickup-Full-v0`, … |
| Room | 12 | `MiniHack-Room-5x5-v0`, `MiniHack-Room-Random-5x5-v0`, `MiniHack-Room-Trap-15x15-v0`, … |
| Sokoban | 8 | `MiniHack-Sokoban1a-v0` … `MiniHack-Sokoban4c-v0` |
| WoD | 8 | `MiniHack-Wodlevel-v0`, `MiniHack-WoD-Easy-v0`, … |
| MazeWalk | 6 | `MiniHack-MazeWalk-9x9-v0`, `MiniHack-MazeWalk-Mapped-9x9-v0`, … |
| Corridor | 5 | `MiniHack-Corridor-R2-v0` … `MiniHack-Corridor-R6-v0` |
| KeyRoom | 5 | `MiniHack-KeyRoom-Dark-S5-v0`, `MiniHack-KeyRoom-Fixed-S5-v0`, … |
| River | 5 | `MiniHack-River-Lava-v0`, `MiniHack-River-Narrow-v0`, … |
| HideNSeek | 4 | `MiniHack-HideNSeek-v0`, `MiniHack-HideNSeek-Mapped-v0`, … |
| Crossing | 4 | `MiniHack-LavaCrossingS9N1-v0`, `MiniHack-LavaCrossingS9N2-v0`, … |
| ExploreMaze | 4 | `MiniHack-ExploreMaze-Easy-v0` … `MiniHack-ExploreMaze-Hard-Mapped-v0` |
| Quest | 3 | `MiniHack-Quest-Easy-v0`, `MiniHack-Quest-Medium-v0`, `MiniHack-Quest-Hard-v0` |
| Memento | 3 | `MiniHack-Memento-Short-F2-v0`, …, `MiniHack-Memento-Hard-F4-v0` |
| Boxoban | 3 | `MiniHack-Boxoban-Unfiltered-v0`, `MiniHack-Boxoban-Hard-v0`, `MiniHack-Boxoban-Medium-v0` |
| Labyrinth | 2 | `MiniHack-Labyrinth-Big-v0`, `MiniHack-Labyrinth-Small-v0` |
| **Total** | **159** | Wave 4 ships ≥ 159 of the 170 canonical vendor envs (the remaining 11 are duplicates / aliases in the vendor table). |

## `MinihaxEnv`

```python
class MinihaxEnv:
    def __init__(self, env_id, *, reward_manager=None):
        spec = MINIHACK_ENV_REGISTRY[env_id]
        self._level_factory = spec.level_factory
        self._reward_manager = reward_manager or spec.reward_manager
        self._max_steps = spec.max_steps
        self._engine = NethaxEnv()           # shared NetHack engine

    def reset(self, rng):
        state = self._level_factory(rng)
        info = {"fired_mask": self._reward_manager.initial_fired_mask(),
                "step_count": 0}
        return state, info

    def step(self, state, action, rng, *, fired_mask, step_count):
        new_state, _obs, _r, _done, _info = self._engine.step(state, action, rng)
        reward, rm_done, new_fired = self._reward_manager.compute_reward(
            state, new_state, fired_mask)
        new_step_count = step_count + 1
        truncated = new_step_count >= self._max_steps
        done = bool(rm_done) or bool(_done) or truncated
        info = {"fired_mask": new_fired, "step_count": new_step_count,
                "truncated": truncated, "engine_done": bool(_done),
                "reward_manager_done": bool(rm_done)}
        return new_state, float(reward), done, info
```

`reset` and `step` are Python-side methods (not JIT-able themselves, because they hold the `RewardManager` Python object). But `NethaxEnv.step` underneath is fully `jax.jit`-compatible — wrappers over a JIT-friendly core, the standard MiniHack pattern.

## Reward shapes per canonical env

The default reward is **sparse +1 on `stairs_down`**, matching vendor MiniHack. A handful of envs ship custom defaults:

- **Sokoban / Boxoban**: `−0.001/step` + `+0.1` per pit filled (matching vendor `MiniHack-Sokoban` shaping).
- **LavaCross**: terminal +1 on goal; lava-death negative-terminal is Wave 5.
- **Skill envs**: each has a bespoke event mix (e.g. `Eat-Distract` rewards eating any food rather than reaching a stair).

Users override via the `reward_manager=` constructor kwarg; see [`test_minihax_envs.py::test_custom_reward_manager_overrides_default`](../../tests/test_minihax_envs.py).

## Test coverage

| File | Tests | Wave |
|---|---|---|
| `test_minihax_envs.py` | 11 | Wave 4 — registry, reset/step, custom reward |
| `test_minihax_level_generator.py` | 8 | Wave 4 — LG API |
| `test_minihax_reward_manager.py` | 14 | Wave 4 — all event factories |
| `test_minihax_des_parser.py` | 36 | Wave 4 — parses each vendor .des file |
| `test_wave4_integration.py` | 15 | Wave 4 — cross-subsystem |
| Total | **84** | Wave 4 MiniHack-adjacent tests |
