# Nethax Architecture

System-level view of how the JAX NetHack reimplementation fits together. This doc captures the **shape** of nethax: data layout, dispatch flow, and the JAX functional contract every subsystem obeys.

For the per-subsystem parity matrix (what is bit-equal to vendor, what is simplified, what is deliberately divergent), see [`vendor_parity.md`](vendor_parity.md). For the RL-practitioner-oriented NLE migration story, see [`nle_migration.md`](nle_migration.md).

---

## Top-down

```
NethaxEnv.step(state, action, rng)
   |
   v
NethaxEnv._step_jit  =  jax.jit(_step_impl)
   |
   v
_step_impl  ---- mirrors vendor/nethack/src/allmain.c::moveloop ----
   1. dispatch_action(state, action, rng_act)          # player turn
   2. monster_ai.step(state, rng_monsters)             # movemon
   3. timestep += 1                                    # svm.moves++
   4. status_effects.step(...)                         # nh_timeout + regen
   5. polymorph.step(state, rng_poly)                  # were-form / timer decay
   6. magic.age_spells_decrement                       # spell_memory -= 1
   7. shop.shop_step(state, rng_shop)                  # pay-at-exit + pursuit
   8. ascension.maybe_ascend(state)                    # done() paths
```

The whole body is wrapped in a single `jax.lax.cond(state.done, no-op, do_step)`, so calling `step` on a finished episode is a one-cycle no-op.

The action-dispatch table (`subsystems/action_dispatch.py::_HANDLERS`) is a 43-entry `lax.switch`-friendly tuple:

| Slot | Range | Handlers |
|---|---|---|
| 0 | noop | wait / unhandled |
| 1-8 | move | 8 cardinals + intercardinals |
| 9-16 | run | 8 run variants |
| 17-18 | stairs | up / down |
| 19 | wait | `.` |
| 20-36 | core actions | eat, quaff, read, zap, cast, pickup, drop, wield, wear, putOn, remove, open, close, kick, fight, search, pray |
| 37-41 | Wave 5 actions | two-weapon, throw, loot, apply, engrave |
| 42 | Wave 6 action | name / call |

Every handler is a pure `(state, rng) -> state` function so the table dispatches via `jax.lax.switch` with no Python branching at trace time. Vendor citations live next to each slot in `action_dispatch.py`.

---

## `EnvState` pytree

`Nethax/nethax/state.py::EnvState` is a `flax.struct.dataclass` — registered as a JAX pytree, so it traverses `jit` / `vmap` / `scan` natively.

### Subsystem slices

Each subsystem owns a `*State` struct declared in its own module; `EnvState` simply composes them.

```
combat:         CombatState           subsystems/combat.py
magic:          MagicState            subsystems/magic.py
monster_ai:     MonsterAIState        subsystems/monster_ai.py        (200 slots)
polymorph:      PolymorphState        subsystems/polymorph.py
inventory:      InventoryState        subsystems/inventory.py         (52 slots)
identification: IdentificationState   subsystems/identification.py
traps:          TrapState             subsystems/traps.py
features:       FeaturesState         subsystems/features.py          (fountain/throne/sink/altar)
prayer:         PrayerState           subsystems/prayer.py
conduct:        ConductState          subsystems/conduct.py           (13 conducts)
shop:           ShopState             subsystems/shop.py
quest:          QuestState            subsystems/quest.py
status:         StatusState           subsystems/status_effects.py
scoring:        ScoringState          subsystems/scoring.py
messages:       MessageState          subsystems/messages.py
containers:     ContainerState        subsystems/containers.py        (4 containers x 20 items)
engrave:        EngraveState          subsystems/engrave.py           (per-tile)
dungeon:        DungeonState          dungeon/branches.py
level_memory:   LevelMemoryState      dungeon/level_memory.py
```

### Player core (kept at top level for fast access)

```
player_pos          int16[2]    (row, col)
player_hp, hp_max   int32
player_pw, pw_max   int32
player_xp, xl       int32       experience points / level
player_role         int8        Role enum
player_race         int8        Race enum
player_align        int8        Alignment enum (0=lawful, 1=neutral, 2=chaotic)
player_str          int16       0..125 (matches vendor exceptional STR encoding)
player_dex/con/int/wis/cha   int8
player_gold         int32
player_ac           int32       10 = unarmored, lower = better

# Wave 6 closing-audit additions (vendor u.* parity)
player_luck         int8        u.uluck     (you.h:460); [-10,10]
player_moreluck     int8        u.moreluck  (you.h:460); luckstone bonus
player_in_water     bool        u.uinwater  (you.h:431)
player_buried       bool        u.uburied   (you.h:436)
player_steed_mid    uint32      u.usteed_mid (you.h:494); 0 = not riding
player_killer_mid   uint32      svk.killer (last attacker)
player_mortality    int32       u.umortality (you.h:497)
```

### Terrain + ground items

```
terrain        int8[N_BRANCHES, MAX_LEVELS, MAP_H, MAP_W]   tile type
explored       bool[N_BRANCHES, MAX_LEVELS, MAP_H, MAP_W]
visible        bool[MAP_H, MAP_W]                            FOV current level only
ground_items   Item-pytree of shape [B, L, H, W, MAX_STACK]  per-tile item stack
```

Compile-time shape constants from `StaticParams`:

- `MAP_H = 21`, `MAP_W = 80` — matches NetHack `ROWNO=21`, `COLNO=80`
- `N_BRANCHES = 7` — Main, Mines, Sokoban, Quest, Vlad, Gehennom, Endgame
- `MAX_LEVELS_PER_BRANCH = 32` — vendor's deepest branch

Changing any `StaticParams` field changes pytree shapes and invalidates compiled functions.

### Game-loop bookkeeping

```
rng         jax.random.PRNGKey      threaded forwards through every step
timestep    int32                   incremented every step
done        bool                    terminal flag
```

---

## JAX functional contract

Every subsystem step function obeys the same signature pattern:

```python
def step(state: EnvState, ...args..., rng: jax.Array) -> EnvState: ...
```

Hard rules:

1. **No mutation.** `state = state.replace(field=new_value)` always returns a new pytree. The old one is unchanged.
2. **No Python `if` / `for` over traced data inside `_step_impl`.** Use `jax.lax.cond`, `jax.lax.switch`, `jax.lax.fori_loop`, `jax.lax.scan`, `jax.lax.select`.
3. **Static shapes.** Every array in `EnvState` has a shape that is a function of `StaticParams` only — no shape depends on runtime values. Monster slots, inventory slots, ground-item stack depth, container depth, etc., are all fixed at trace time. Empty slots are masked with sentinel values (`category=0`, `mid=0`, etc.).
4. **PRNG threading.** The only randomness source is `state.rng`; the env splits it into per-subsystem keys at the top of `_step_impl` via `jax.random.split(rng, k)`. Subsystem helpers (`rng.dice_roll`, `rng.rn2`, `rng.weighted_choice`) wrap `jax.random` directly.
5. **JIT-compatible.** `NethaxEnv.__init__` builds `self._step_jit = jax.jit(_step_impl)` once; every `env.step(...)` reuses it.

These rules let nethax compose cleanly with `jax.vmap` for batched rollouts and `jax.lax.scan` for unrolled trajectories — see [`benchmark.md`](benchmark.md).

---

## PRNG threading

`state.rng` is a JAX `PRNGKey`. The pattern at the top of `_step_impl`:

```python
rng_act, rng_monsters, rng_status, rng_poly, rng_shop = jax.random.split(rng, 5)
```

`state.rng` itself is **not** consumed by `_step_impl` — instead the caller passes a fresh `rng` per call (see `NethaxEnv.step`'s `rng` argument). The `state.rng` field exists so reset-time RNG can persist (dungeon-gen RNG, identification-shuffle RNG, etc.) without being threaded through every step.

Inside any subsystem, further splits happen at the call site as needed. Determinism is exact: same `(state, action, rng)` produces the same `state'` byte-for-byte. This is checked by `tests/test_nle_compat_full.py::test_determinism_*`.

---

## JIT compile cost + post-warmup throughput

The first call to `env.step` per Python process pays a one-time JIT compile of `_step_impl`:

| Stage | Time |
|---|---|
| `env._step_jit` first call | ~30-60 s (CPU, full dispatch + monster-AI + status pipeline) |
| Subsequent calls | sub-millisecond |

Post-warmup measured throughput (smoke baseline, single-env CPU, Apple Silicon arm64):

| Scenario | Mean sps | Median sps | p95 sps |
|---|---|---|---|
| single-env (no vmap) | 140 | 146 | 152 |

Full vmap / lax.scan numbers and the comparison vs NLE live in [`benchmark.md`](benchmark.md). The headline tradeoff:

- **Single-env**: nethax is slower than NLE's C extension (~140 sps vs NLE's ~10 000-20 000 sps).
- **Batched (vmap batch>=512)**: nethax wins because the batched env is a single fused XLA kernel, whereas NLE is fork-per-env and has no batched path.

---

## Cross-references

- `Nethax/nethax/env.py::_step_impl` — the canonical 8-step moveloop above.
- `Nethax/nethax/state.py::EnvState` — pytree composition.
- `Nethax/nethax/subsystems/action_dispatch.py::_HANDLERS` — 43-slot dispatch table.
- `vendor/nethack/src/allmain.c::moveloop` — vendor source of the per-turn pipeline order.
- [`vendor_parity.md`](vendor_parity.md) — per-subsystem parity + vendor citations.
- [`nle_migration.md`](nle_migration.md) — NLE drop-in story for RL users.
- [`benchmark.md`](benchmark.md) — throughput vs NLE.
