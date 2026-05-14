# Wave 1 — Architecture

## Package layout

```
Nethax/nethax/
├── __init__.py              public re-exports: EnvState, NethaxEnv, Action, ACTIONS, NLE_OBSERVATION_KEYS
├── state.py                 master EnvState (composes every subsystem slice)
├── env.py                   top-level NethaxEnv class — NLE-compatible reset/step
├── fov.py                   compute_fov, update_explored
├── rng.py                   JAX PRNG conventions (split_n, dice_roll, weighted_choice)
├── save_load.py             pytree serialization (stub)
│
├── constants/               NLE-parity & vendor data schemas
│   ├── __init__.py            re-exports everything
│   ├── actions.py             121-action enum + USEFUL_ACTIONS (101)
│   ├── blstats.py             27-field BL_* indices + BL_MASK_* condition flags
│   ├── glyphs.py              13 offset constants for glyph ID scheme
│   ├── roles.py               13 roles (Archeologist..Wizard)
│   ├── races.py               5 races (Human, Elf, Dwarf, Gnome, Orc)
│   ├── monsters.py            MonsterEntry schema + 10 canonical entries
│   └── objects.py             ObjectEntry schema + 10 canonical entries
│
├── subsystems/              one module per subsystem
│   ├── __init__.py
│   ├── combat.py              hit rolls, damage, weapon skills
│   ├── magic.py               43-spell SpellId, MagicState
│   ├── monster_ai.py          MoveStrategy, monster turns
│   ├── polymorph.py           PolymorphState (player + monster poly)
│   ├── inventory.py           InventoryState, slot management
│   ├── items.py               BUCStatus, Erosion, item effects
│   ├── identification.py      appearance shuffling, partial/full ID
│   ├── traps.py               26 TrapType, TrapState
│   ├── features.py            doors, fountains, altars, thrones, sinks
│   ├── prayer.py              alignment, prayer_timeout, luck
│   ├── conduct.py             13 Conduct flags
│   ├── shop.py                shopkeeper, bill (simplified by project scope)
│   ├── quest.py               QuestStage, leader/nemesis/artifact
│   ├── status_effects.py      Intrinsic, TimedStatus, HungerState, Encumbrance
│   ├── scoring.py             score, achievements, kills
│   ├── messages.py            MessageState ring buffer + MessageId enum
│   └── action_dispatch.py     dispatch_action — top-level routing
│
├── dungeon/                 dungeon generation & topology
│   ├── __init__.py
│   ├── rooms.py               RoomType (26 types), Room
│   ├── mazes.py               generate_maze_kruskal/perfect/dla
│   ├── corridors.py           connect_segments, place_doors
│   ├── branches.py            7-branch graph: Main/Mines/Sokoban/Quest/Vlad/Gehennom/Endgame
│   ├── special_levels.py      28 named levels (Oracle, Mine Town, Castle, Sanctum, ...)
│   └── level_memory.py        per-level state caching for descent/ascent
│
└── obs/                     observation builders
    ├── __init__.py
    ├── nle_obs.py             NLE-parity 17-key observation dict
    ├── symbolic_obs.py        flat vector (placeholder dim 1024)
    ├── pixel_obs.py           sprite render via Nethax/tiles/ atlas (stub)
    └── text_obs.py            24×80 tty char grid (stub)
```

The pre-existing files in `Nethax/nethax/` (`nethax_state.py`, `constants.py`, `game_logic.py`, `play_nethax.py`, `renderer.py`, `envs/`, `util/`, `world_gen/`) **were not modified** in Wave 1 — they continue to work as before. Wave 2 will begin migrating callers from the old `nethax_state.EnvState` to the new `state.EnvState`.

`Nethax/minihax/` and `Nethax/environment_base/` and `Nethax/tiles/` are likewise untouched — those will be addressed in Wave 5.

---

## Module-dependency graph (Wave 1)

```
                ┌─────────────────────────┐
                │   Nethax.nethax.__init__│  re-exports
                └────┬────────────────┬───┘
                     │                │
                     ▼                ▼
                  state.py         env.py
                     │                │
       ┌─────────────┼────────────────┘
       │             │                ↓
       ▼             ▼          action_dispatch.py
 ┌────────────┐  ┌──────────┐         ↓
 │ subsystems │  │ dungeon/ │     (Wave 2 will dispatch
 │   /*.py    │  │  *.py    │      into all subsystems)
 └─────┬──────┘  └──────┬───┘
       │                │
       └───────┬────────┘
               ▼
        constants/*.py
               ▲
               │
          obs/*.py
```

Hard rules enforced this wave:

1. **No cycles.** `subsystems/` imports from `constants/`. `state.py` imports from `subsystems/` + `dungeon/`. `env.py` imports from `state.py` + `obs/` + `subsystems/action_dispatch.py`. `obs/` imports from `constants/`.
2. **Each subsystem owns its slice.** A subsystem's Flax state class lives in its own file. `state.py` only composes — never defines fields a subsystem should own.
3. **Constants are pure-Python where possible.** `actions.py`, `blstats.py`, `glyphs.py`, `roles.py`, `races.py` import only `enum` and `typing` — no JAX dependency. This lets them be imported in environments without JAX installed.
4. **Step functions are pure.** Every `step(state, rng)` is a JAX-compatible pure function. Wave 1's are no-ops; the contract is in place for later waves.

---

## Master state composition

`EnvState` (in `state.py`) holds 17 subsystem state slices + 17 player-core scalars + 3 terrain layers + 3 game-loop scalars:

```python
@struct.dataclass
class EnvState:
    # Subsystem slices (17)
    combat, magic, monster_ai, polymorph, inventory, identification,
    traps, features, prayer, conduct, shop, quest, status, scoring,
    messages, dungeon, level_memory: <SubsystemName>State

    # Player core (17 scalars)
    player_pos, player_hp, player_hp_max, player_pw, player_pw_max,
    player_xp, player_xl, player_role, player_race, player_align,
    player_str, player_dex, player_con, player_int, player_wis,
    player_cha, player_gold

    # Terrain layers
    terrain: int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]
    explored: bool[N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]
    visible: bool[MAP_H, MAP_W]

    # Game loop
    rng: jax.PRNGKey
    timestep: int32
    done: bool
```

See [`state-schema.md`](state-schema.md) for the full pytree walkthrough.

---

## Design patterns adopted

- **Flax `@struct.dataclass`** for every state slice. Immutable pytree, fully JAX-compatible.
- **`classmethod default(cls)`** for zero-init constructors. Some take size params (e.g., `TrapState.default(num_levels, map_h, map_w)`); the rest are no-arg. State factories `make_<thing>_state()` for two subsystems that don't fit the classmethod pattern (`MonsterAIState`, `PolymorphState`, `LevelMemoryState`).
- **`enum.IntEnum`** for all enums. NetHack source uses C `#define` constants but every enum we use becomes an `IntEnum` so we get name semantics and round-trip with int storage.
- **Citation-driven**: every stub file's top docstring lists `vendor/nethack/src/*.c` files that contain the canonical implementation. When a Wave 2+ implementer touches a stub, they have a 30-second path to ground truth.
- **TODO blocks at bottom of every stub**, organised by wave: each subsystem knows what Wave 2, 3, 4, 5, 6 owe it.

---

## What the dispatcher looks like (and will look like)

`subsystems/action_dispatch.py` currently exposes:

```python
ACTION_HANDLERS: tuple[Callable, ...]  # length 121, all identity-fn

def dispatch_action(state, action, rng) -> EnvState:
    """Wave 1: returns state unchanged. Wave 2+ uses jax.lax.switch."""
    return state
```

Wave 2 will replace this with a `jax.lax.switch`-driven dispatch table where each handler is the actual per-action logic (move N, eat, quaff, cast, zap, etc.).

---

## Why this architecture survives the next 5 waves

Each later wave fills *behaviors*, not *shapes*. The pytree topology is fixed. That means:

- Re-`jit` invalidation will happen only when state types change — which by Wave 1 fiat will be rare.
- A `step()` implementer in Wave 3 only has to read & write that subsystem's slice; cross-subsystem coupling is explicit through state.
- Tests can pin shapes from Wave 1 forward; subsystem behavior tests can be added without breaking shape tests.
