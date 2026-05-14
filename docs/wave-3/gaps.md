# Wave 3 — Gaps

What's still missing after Wave 3. Most maps to Wave 4 or Wave 5.

## Wave 4 (next)

### MiniHack pull-forward (the big Wave 4 deliverable)
- `LevelGenerator` API port (add_monster, add_trap, add_object, fill_terrain, regions)
- `RewardManager` event system (add_eat_event, add_kill_event, add_message_event, terminal_sufficient)
- Port 36 canonical `*.des` files → Python factories
- Register all 170 canonical envs (`MiniHack-Room-5x5`, `MiniHack-Corridor-R*`, `MiniHack-LavaCross-*`, `MiniHack-MazeWalk-*`, `MiniHack-Sokoban*`, `MiniHack-Quest-*`, etc.)
- Implement `Nethax.minihax` package using new `NethaxEnv` infrastructure

### Action dispatch wiring (deferred from Wave 3)
- Wire `handle_eat / handle_quaff / handle_read / handle_zap / handle_cast / handle_pickup / handle_drop / handle_wield / handle_wear / handle_put_on / handle_remove / handle_open / handle_close / handle_kick / handle_fight` into `_HANDLERS` and `_ACTION_TO_HANDLER_IDX`
- After wiring: ~40 of 121 actions are real; rest fall through to no-op
- Add per-action tests verifying dispatch reaches each handler

### Monster AI completion
- LoS + pathfinding aware of walls
- `muse` (monster use of items: heal, escape, attack)
- Monster spell casting (integrate `mcastu` logic)
- Retreat behavior on low HP
- Pet recruitment / leashing / feeding
- Sleeping wake on visible-player trigger

### Dungeon branches
- Branch graph initialization at game start
- Place branch-entrance stairs in Main at canonical depths
- Mines (5 levels, entrance at Dlvl 2-4)
- Sokoban (4 levels, entrance after Oracle at Dlvl 6-10)
- Quest (5 levels, entrance at XL14 portal)
- Vlad's Tower (3 levels in Gehennom)
- Gehennom (16 levels below Castle)
- Endgame (5 Astral planes)

### Special levels (Wave 4 subset)
- Oracle level
- Mine Town
- Mines End
- Big Room

### Polymorph
- Player polymorph (controlled + random)
- Monster polymorph (poly trap, wand zap, poly pile)
- Form change → attack-set swap, AC recompute
- Lycanthropy (were-creatures)

### Prayer + alignment + luck
- `pray()` outcome table from `pray.c:500-1500`
- Divine intervention (heal / cure / gift / smite)
- Sacrifice on altar (corpse type → outcome)
- Luck adjustment from luckstone, BUC actions
- Prayer timeout management
- Altar BUC sense

### Conduct trigger wiring
- 13 conducts: FOODLESS, VEGAN, VEGETARIAN, ATHEIST, WEAPONLESS, PACIFIST, ILLITERATE, POLYPILELESS, POLYSELFLESS, WISHLESS, ARTIWISHLESS, GENOCIDELESS, ELBERETHLESS
- Each needs its violation trigger wired in the relevant action handler

### Observation polish (remaining 4 keys)
- `colors` overlay for monsters/items (currently terrain-only)
- `specials` per-tile flags (corpse, pile, trap, secret-door, invisible)
- `program_state`: menu/dialog/Y-N states
- `internal`: agent-visible internals (stairs_down position, hunger_state cache)
- `screen_descriptions`: per-glyph "what does this mean" descriptions
- `tty_colors`: per-tile colors in tty format

### Features
- Fountain quaff/dip effect tables (16 effects)
- Throne effect table (14 effects)
- Sink effect table (13 effects)
- Drawbridge open/close/destroy
- Secret door discovery via SEARCH action

## Wave 5

### Special levels (full set)
- Castle, Valley, Asmodeus, Baalzebub, Juiblex, Orcus
- Vlad's Tower (top + lower)
- Wizard's Tower + 3 fakes
- Sanctum (Amulet of Yendor placement)

### Quest
- Per-role quest tables (artifact, leader, nemesis from `role.c::Role` struct quest fields)
- Quest level generation (per-role unique layouts)
- Nemesis fight mechanics
- Artifact recovery + use
- Return-to-leader victory

### Vibrating square + Gehennom entrance
- VIBRATING_SQUARE trap → Gehennom gateway
- MAGIC_PORTAL branch jump

### Monster AI polish
- Stationary AI (shopkeeper guarding shop, priest guarding temple, vault guard escorting)
- Quest leader & nemesis behavior

### Bag-of-holding / containers
- Nested inventory
- Container open/close/put/take

## Wave 6

### Save / load
- `save_state(state, path)`: pytree flatten + numpy save
- `load_state(path)`: inverse
- Cross-version compatibility policy

### Scoring + ascension
- `compute_final_score` formula from `vendor/nethack/src/end.c`
- Death message text generation
- Ascension condition: Amulet + Astral altar offering

### Astral plane + 4 elemental planes
- ASTRAL_PLANE level with 3 altars
- EARTH / AIR / FIRE / WATER planes

### Shop simplified buy/sell
- Pick up in shop → bill accrual
- Pay bill at exit
- Angry shopkeeper attack mode (no haggling, no chat)

### `inv_strs` polish
- User-given names ("named Sting")
- Article "a"/"an" via vowel check
- Plural irregulars
- Two-weapon "alternate weapon" status

### Conduct scoreboard
- Display preserved conducts at end-game

### `scripts/legacy/play_nethax.py` rewrite
- Pygame interactive UI against new `NethaxEnv`

### Full monstr difficulty table
- Replace `entry.level` proxy with the full vendor formula

### Object table canonicalize
- Drop dual-naming ("potion of healing" + "healing") → keep only canonical bare names
- Bring OBJECTS count from 503 to ~453

### Monster table trim
- Remove Charon, mail daemon, and other `#ifdef`-guarded entries
- Bring MONSTERS count from 382 to canonical 381

### Property-based combat tests
- Hypothesis-style tests against vendor C reference for THAC0 + damage formulas
- Critical-hit-equivalents (lucky/unlucky d20 extremes)

### Specific role bonuses
- Monk martial arts damage scaling
- Samurai bushido bonus
- Knight chivalric morale
- Wave 6 polishes role-specific combat tweaks

## Out-of-scope (deliberately skipped, with confirmation)

- Wizard-mode debug commands
- Mail subsystem
- Music / sounds
- Real Lua integration (3.7 special levels use Python factories instead)
- Full shopkeeper haggling + dialogue
- Bones files (default off; may be flag-gated in Wave 6)

## TODOs visible in code

`grep -rEn "TODO|FIXME" Nethax/nethax --include='*.py' | wc -l` →  count is now in the low 200s (was ~80 at Wave 1; new TODOs added for Wave 4-6 items). Each TODO carries a wave assignment in the comment.

## Test gaps

- No property-based tests (Hypothesis) yet — would catch off-by-one in formulas
- No end-to-end "play to depth 5" benchmark — would catch perf regressions
- No NLE compatibility shim — would let real NLE agents run on `NethaxEnv`. Wave 4 should add a wrapper class
- No throughput benchmark (steps/sec on CPU vs GPU) — would validate the JAX-perf claim that's the entire reason for this project
