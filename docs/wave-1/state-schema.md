# Wave 1 — `EnvState` schema walkthrough

The master `EnvState` pytree composes every subsystem slice + player-core scalars + terrain layers + game-loop bookkeeping. Defined in `Nethax/nethax/state.py`.

## Top-level structure

```python
@struct.dataclass
class EnvState:
    # ── 17 subsystem state slices ──
    combat:        CombatState              # weapon skills, last attack
    magic:         MagicState               # spell memory, Pw regen
    monster_ai:    MonsterAIState           # per-monster behavior
    polymorph:     PolymorphState           # poly form + timer
    inventory:     InventoryState           # 52 slots + worn equipment
    identification:IdentificationState      # appearance shuffles + known flags
    traps:         TrapState                # per-tile trap type + revealed
    features:      FeaturesState            # doors, fountains, altars, sinks, thrones
    prayer:        PrayerState              # alignment, prayer_timeout, luck
    conduct:       ConductState             # 13 conduct violation flags
    shop:          ShopState                # shopkeeper state + bill (simplified)
    quest:         QuestState               # stage + leader/nemesis/artifact
    status:        StatusState              # intrinsics, hunger, encumbrance
    scoring:       ScoringState             # score, kills, achievements
    messages:      MessageState             # 256-byte buffer + 20-line history
    dungeon:       DungeonState             # branch graph, current location
    level_memory:  LevelMemoryState         # per-level caching for descent/ascent

    # ── Player core (frequently-accessed scalars) ──
    player_pos:    int16[2]                 # (row, col)
    player_hp:     int32
    player_hp_max: int32
    player_pw:     int32
    player_pw_max: int32
    player_xp:     int32
    player_xl:     int32                    # experience level 1..30
    player_role:   int8                     # Role enum
    player_race:   int8                     # Race enum
    player_align:  int8                     # Alignment enum
    player_str:    int16                    # 0..125 raw strength
    player_dex:    int8
    player_con:    int8
    player_int:    int8
    player_wis:    int8
    player_cha:    int8
    player_gold:   int32

    # ── Terrain layers (multi-branch, multi-level) ──
    terrain:  int8 [N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]
    explored: bool [N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]
    visible:  bool [MAP_H, MAP_W]           # FOV for current level only

    # ── Game loop ──
    rng:       jax.random.PRNGKey
    timestep:  int32
    done:      bool
```

## Pytree shape stats (with default `StaticParams`)

- `N_BRANCHES = 7`, `MAX_LEVELS_PER_BRANCH = 32`, `MAP_H = 21`, `MAP_W = 80`
- Total `terrain` cells: 7 × 32 × 21 × 80 = **376,320**
- Total `explored` cells: same — **376,320 bools**
- Per-tile trap-related arrays in `TrapState`: same shape — 2 arrays
- Per-tile feature arrays in `FeaturesState`: 5 arrays — same shape
- `LevelMemoryState.cached_map`: same — `int8[7, 32, 21, 80]`

So per-tile state uses roughly **3.4 MB** flat (one int8 per cell × ~9 arrays), which is well within JIT trace budgets.

## Subsystem-slice sizes (per default state)

| Slice | Notable arrays |
|---|---|
| `CombatState` | `weapon_skill[40]` int8 + `weapon_practice[40]` int32 → 200 B |
| `MagicState` | `spell_memory[43]` int32 + `spell_known[43]` bool + `spell_letter[43]` int8 → ~260 B |
| `MonsterAIState` | per-monster arrays × 200 max monsters: `target_pos[200,2]`, `mstrategy[200]`, etc. → ~3 KB |
| `InventoryState` | `items[52]: Item` → 7 fields × 52 = 364 scalars; `worn_armor[7]`, `worn_rings[2]` |
| `IdentificationState` | 6 appearance arrays totaling 184 entries + identified[NUM_OBJECTS] bool ≈ NUM_OBJECTS B |
| `TrapState` | `trap_type[7,32,21,80]` int8 + `revealed[7,32,21,80]` bool → ~750 KB |
| `FeaturesState` | 5 × `[7,32,21,80]` arrays → ~1.9 MB |
| `LevelMemoryState` | `cached_map[7,32,21,80]` int8 + `cached_explored[...]` bool → ~750 KB |
| `ShopState` | `bill[25, 52]` int32 → 5.2 KB |

The hot tiles (terrain/explored/traps/features/level_memory) dominate by an order of magnitude. Wave 4's level-memory wiring needs to be careful that `lax.scan` doesn't materialise per-step copies.

## `StaticParams`

```python
@struct.dataclass
class StaticParams:
    map_h: int = 21                          # NetHack ROWNO
    map_w: int = 80                          # NetHack COLNO
    n_branches: int = 7
    max_levels_per_branch: int = 32          # matches MAXLEVEL/global.h
```

These determine pytree shape (and therefore JIT cache identity). Changing any of them after first JIT trace causes a recompile. Wave 1 hard-codes them via defaults; Wave 2+ may expose them to RL config.

## Factory pattern

Each slice has one of these constructor patterns. We didn't unify them in Wave 1 because the existing code in `nethax_state.py` already established `Item`-style classmethods.

| Pattern | Used by |
|---|---|
| `<Slice>State.default()` no args | combat, magic, prayer, conduct, shop, quest, status, scoring, messages, features (note: features takes size args) |
| `<Slice>State.default(num_levels, map_h, map_w)` | traps, features |
| `<Slice>State.empty()` | inventory |
| `<Slice>State.unshuffled()` | identification |
| Module-level `make_<slice>_state()` factory | monster_ai, polymorph, level_memory |
| Hand-built in `state.py` | dungeon (no factory; `_default_dungeon_state()` helper) |

Wave 2 should consider unifying everything on `default()` — but it's not blocking.

## Pytree validation

`EnvState.default(rng=jax.random.PRNGKey(0))` produces a valid pytree that:
- contains only `jax.Array` leaves
- `jax.tree.map` walks every field
- round-trips through `jax.jit` with no errors (verified by smoke-running `jax.jit(env.step)`)
- has the right dtypes when `JAX_ENABLE_X64=1` is set (required for `blstats` int64 NLE-parity)
