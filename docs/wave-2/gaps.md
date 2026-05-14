# Wave 2 — Gaps

What's still missing or sketchy after Wave 2. Each item maps to a target wave.

## Mechanics that are still no-op

These are subsystems whose state slice + signatures are in place but the bodies return input unchanged. All assigned to Wave 3 or later.

### Wave 3 (next)
- **Combat**: `combat.melee_attack` / `ranged_attack` / `passive_attack` — return zero damage.
- **Magic**: `magic.cast_spell` / `magic.read_spellbook` — no effect dispatch.
- **Status effects**: `status_effects.tick_timers` / hunger threshold / encumbrance formula / HP+Pw regen — all no-op.
- **Traps**: `traps.trigger_trap` returns zero damage; no trap-specific effects.
- **Items**: `items.{apply_blessing,curse,erode,enchant}` — no state mutation.
- **Identification**: appearance shuffling exists in state but `init_shuffled_appearances` is no-op.
- **Inventory**: `inventory.{pickup,drop,wield,wear,take_off,put_on_ring}` — no slot mutation.
- **Object effect dispatch**: potion quaff, scroll read, wand zap — entries exist but nothing happens on use.

### Wave 4
- **Monster AI**: `monster_ai.{monster_turn,pet_turn,wake_sleeping}` — no movement.
- **Polymorph**: timer tick and form change.
- **Prayer + alignment + luck + altar**: outcome table.
- **Conduct trigger wiring** in 13 places.
- **Features**: fountain / throne / sink / altar / door / drawbridge effects.
- **Branches**: Mines, Sokoban, Quest, Vlad's, Gehennom, Endgame — branch graph initialization at game start.
- **Special levels**: Oracle, Mine Town, Mines End, Sokoban floors, Big Room.

### Wave 5
- **Quest**: per-role quest tables + level generation + nemesis fight.
- **MiniHack 170-env curriculum**: LevelGenerator + RewardManager port.

### Wave 6
- **Shop**: simplified buy/sell.
- **Save / load** pytree serialization.
- **Score**: final scoring formula.
- **Ascension**: Amulet + Astral Plane + altar offering.
- **Special levels**: Castle, Valley, Asmodeus, Baalzebub, Juiblex, Orcus, Vlad's Tower, Wizard's Tower, Sanctum, 4 elemental planes.
- **Conducts**: scoreboard display.

## Data-table coverage

- **Monsters**: 390 / canonical 381 (9 over — includes Charon, mail daemon, some `#if 0` entries we should remove). Wave 6 polish.
- **Objects**: 503 / canonical 453 (50 over — dual naming for potions/scrolls/wands). Wave 3 will canonicalize and drop verbose forms.
- **Tools**: 51 / canonical ~70 — missing ~19 instruments / traps-as-tools / specialty containers. Wave 3.
- **Spellbooks**: 44 / canonical 46 — 2 missing are `#if 0` deferred in vendor (`flame sphere`, `freeze sphere`). Acceptable.

## Observation projection still zero

| Key | Wave to wire |
|---|---|
| `colors`, `tty_colors` | Wave 3 (color lookup tables) |
| `specials` | Wave 3 (corpse / statue flags per tile) |
| `inv_glyphs`, `inv_letters`, `inv_oclasses`, `inv_strs` | Wave 3 (after inventory pickup wires up) |
| `screen_descriptions` | Wave 4 |
| `program_state` | Wave 4 (menu / dialog states) |
| `internal` | Wave 4 (agent-visible internals: stairs_down location, hunger state) |
| `misc` | Wave 4 |

## Top-level player core fields still placeholder

In `state.py::EnvState.default`:
- `player_role / player_race / player_align` = 0 — Wave 3 will accept role + race + alignment from `reset()` kwargs.
- `player_hp = player_hp_max = 10` — Wave 3 computes from role + race + level.
- `player_pw / pw_max = 0` — Wave 3 (Wizard / Priest / Healer roles).
- `player_str = 18` — fixed; Wave 3 should roll per role table.
- Other ability scores = 10 — same.

## Subsystem-specific TODOs

### Doors not placed
`dungeon/corridors.py::place_doors` is implemented but not called in `generate_main_branch_l1` because the FOV / movement haven't yet learned to bump-open closed doors. **Wave 3**: enable.

### `Nethax/minihax/` still uses pre-Wave-1 state
The MiniHack package has its own `state.py` / `game_logic/` and doesn't go through `Nethax.nethax.NethaxEnv`. **Wave 3 or Wave 5**: migrate `minihax` envs to the new `NethaxEnv` (or build a `MinihaxEnv` wrapper that subclasses it).

### `scripts/legacy/play_nethax.py` is broken
The 285-line pygame interactive UI was moved here when its imports (`nethax_state`, `game_logic`, `renderer`) were deleted. **Wave 6**: rewrite against `NethaxEnv`.

## API surface gaps vs NLE

| NLE feature | Status |
|---|---|
| `info["end_status"]` (RUNNING/DEATH/TASK_SUCCESSFUL/ABORTED) | empty info; Wave 6 |
| `info["is_ascended"]` | empty; Wave 6 |
| `wizkit_items` reset kwarg | not yet; Wave 4 |
| `seed(core, disp, reseed)` explicit method | env takes `rng` directly; Wave 3 if parity needed |

## Tests still skipped

- `test_vendor_parity.py` runs only if NLE installed. Should add a CI matrix entry to ensure it runs on at least one config.

## Documentation backlog

- `docs/wave-2/subsystem-readiness-matrix.md` (auto-generated table of every subsystem with its current step-fn behavior) — not blocking.
- Per-subsystem implementation notes (e.g., "Combat formula derivation from `uhitm.c::do_hit()`") — Wave 3 deliverable.
