# Wave 1 — Gap list

Aggregated TODO list across every Wave 1 stub. These items are the **game-design backlog** that Waves 2-6 will burn down. Items are grouped by canonical NetHack subsystem.

`grep -rEn "TODO|FIXME|XXX|HACK" Nethax/nethax --include="*.py" | wc -l` → **~80 TODO comments** across the codebase as of Wave 1 close.

---

## Combat — assigned to Wave 3

- THAC0 / d20 to-hit roll (STR/DEX/luck/enchant vs defender AC)
- Weapon damage roll (small/large dice + enchant + STR bonus + role bonus)
- Weapon skill tiers (Basic/Skilled/Expert/Master/Grand-Master) + practice counters
- Two-weapon penalty rules
- Two-handed enforcement
- Cleave (Barbarian sword)
- Backstab (Rogue)

## Combat — assigned to Wave 4

- Monster passive attacks (cockatrice touch, fire elemental burn, etc.)
- Ranged attacks: throw, fire (auto-quiver), zap-style
- Breath weapons (dragon, hezrou, etc.)
- Engulf / swallow mechanic
- Polymorph combat

---

## Magic — assigned to Wave 3

- `cast_spell`: Pw check → d100 failure roll (role/role×school multipliers) → effect dispatch
- Per-spell effect implementations (damage formula, status app, target selection) — 43 spells
- Spellbook learning (`read_spellbook`) — fail roll + memory write
- Pw regen formula in `step()` (per role + XL)

## Magic — assigned to Wave 4

- Monster cast at hero (mcastu integration)
- Wand zap rays via `jax.lax.scan` along beam
- Spell memory decay (canonical decay table by school)

---

## Monster AI — assigned to Wave 3

- Per-monster pathfinding (greedy 8-dir + LoS + door-aware)
- Strategy selection by HP/range/items/spells
- Ranged-attack target picking
- Wake-on-disturbance logic

## Monster AI — assigned to Wave 4

- `muse` logic (monster uses items: heal, escape, attack)
- Spell selection (monster casting)
- Retreat behavior
- Item pickup
- Pet recruit / leash / feeding

## Monster AI — assigned to Wave 5

- Territory / stationary AI (shopkeeper, priest, vault guard)

---

## Polymorph — assigned to Wave 3

- Lycanthropy timer tick
- Racial change penalties
- AC recompute on form change
- Attack-set swap on form change

## Polymorph — assigned to Wave 4

- Monster polymorph
- Polytrap effect
- Polypiles (polymorph adjacent objects)

---

## Inventory — assigned to Wave 3

- Slot mutation: `pickup` / `drop` / `wield` / `wear` / `take_off`
- Weight calculation on every change
- Capacity check against STR
- Wear AC update
- Wield damage update

## Inventory — assigned to Wave 4

- BUC sensing on altar / priest
- Full enchantment delta tracking
- Erosion ticks (rust, corrode, burn, rot)
- Wand charge decrement

## Inventory — assigned to Wave 5

- Bag-of-holding nested inventory
- General container support

---

## Items / Identification — assigned to Wave 3

- Potion effect dispatch (healing, gain-X, paralysis, etc.)
- Scroll effect dispatch (identify, teleport, enchant, etc.)
- Wand effect dispatch (per wand type, ray vs target vs self)
- Ring intrinsic application
- Spellbook → spell learning
- Fisher-Yates shuffle of appearances at game start (`jax.random.permutation`)

## Items / Identification — assigned to Wave 4

- Partial ID by use (potion you drank that healed → known)
- Full ID by scroll
- Identification UI / message integration

---

## Traps — assigned to Wave 3

- Damage traps: ARROW_TRAP, DART_TRAP, ROCKTRAP, PIT, SPIKED_PIT, SLP_GAS_TRAP, FIRE_TRAP, LANDMINE
- Cosmetic traps: SQKY_BOARD
- Bear trap (immobilize)
- Web (immobilize)

## Traps — assigned to Wave 4

- TELEP_TRAP (local teleport)
- LEVEL_TELEP (level teleport)
- MAGIC_PORTAL (branch jump)
- POLY_TRAP (random polymorph)
- MAGIC_TRAP (random magic effect)
- ANTI_MAGIC (drain Pw)
- RUST_TRAP (item erosion)
- TRAPDOOR / HOLE (fall to lower level)

## Traps — assigned to Wave 5

- VIBRATING_SQUARE (Gehennom-end gateway)
- ROLLING_BOULDER_TRAP
- STATUE_TRAP

---

## Features — assigned to Wave 3

- Door open / close mechanics
- Door kick / break
- Lock pick / force / unlock with key

## Features — assigned to Wave 4

- Fountain effect table (16 effects)
- Dip-in-fountain effects
- Throne effect table (14 effects)
- Sink effect table (13 effects)
- Altar BUC check
- Altar sacrifice (delegates to prayer subsystem)
- Altar conversion (alignment shift)

## Features — assigned to Wave 5

- Drawbridge open / close / destroy
- Secret door discovery via SEARCH
- Magic chest (separate inventory persisted across levels)

---

## Prayer — assigned to Wave 4

- Full `pray()` outcome table (from `pray.c:500-1500`)
- Divine intervention (heal / cure / gift / smite)
- Sacrifice on altar (monster corpse type → outcome)
- Luck tracking from per-action triggers
- Prayer timeout management

---

## Conduct — assigned to Wave 4

- Wire each `Conduct` violation trigger:
  - FOODLESS / VEGAN / VEGETARIAN: in eat
  - ATHEIST: in prayer / altar
  - WEAPONLESS: in melee_attack
  - PACIFIST: in any kill
  - ILLITERATE: in read scroll / read spellbook / engrave (non-Elbereth)
  - POLYPILELESS: in poly-on-pile
  - POLYSELFLESS: in polymorph_player
  - WISHLESS: in wish-grant
  - ARTIWISHLESS: in artifact-wish
  - GENOCIDELESS: in genocide-cast
  - ELBERETHLESS: in engrave-elbereth

---

## Shop (simplified, per project scope) — assigned to Wave 4

- Buy / sell basic mechanics
- Pickup tracking → bill
- Pay bill mechanic
- Angry shopkeeper (combat-only response)
- Skip: haggling, theft detection, chat, locked doors, watchful neighbors

---

## Quest — assigned to Wave 4

- Per-role quest tables (artifact ID, leader monster ID, nemesis monster ID) from `vendor/nethack/src/role.c`
- Quest entrance portal placement (XL 14 trigger)
- Leader greeting → portal allow

## Quest — assigned to Wave 5

- Quest level generation (per-role unique layouts)
- Nemesis fight mechanics
- Artifact recovery
- Return-to-leader victory

---

## Status Effects — assigned to Wave 3

- Hunger threshold table from `eat.c` (nutrition → HungerState)
- Encumbrance formula from carry weight + STR
- HP/Pw regen formulas (per role, per XL)
- `tick_timers` expiry callbacks (Stoning → death, Strangle → death, etc.)

## Status Effects — assigned to Wave 4

- Sickness progression (food_poisoning → death in N turns)
- Slime progression (slimed → death cycle)
- Confusion: action remapping in dispatcher
- Hallucination: glyph remapping in observation builder

---

## Dungeon — assigned to Wave 2

- `rooms.py::generate_rooms`: non-overlapping rectangular room placement (BSP or rejection sampling)
- `corridors.py::connect_segments`: L-shaped corridor between rooms
- `mazes.py::generate_maze_kruskal`: perfect maze (used in Mines lower half + Quest)
- Simple maze for `mazes.py::generate_maze_perfect`, DLA for `generate_maze_dla`

## Dungeon — assigned to Wave 4

- Branch graph assembled at game start: place branch-entrance stairs in Main at canonical depths
- `traverse_stair` / `enter_branch`: full implementation with lazy level generation + level_memory caching
- Level-memory persistence on descent / ascent

## Dungeon — assigned to Wave 4

- Special levels: ORACLE, MINETOWN, MINES_END, BIG_ROOM
- Sokoban levels 1-4 (port `vendor/nethack/dat/sokoban*.des`)

## Dungeon — assigned to Wave 5

- Special levels: CASTLE, VALLEY, ASMODEUS, BAALZEBUB, JUIBLEX, ORCUS
- Vlad's Tower (top + lower)
- Wizard's Tower + FAKE_WIZARD_1/2/3
- SANCTUM (Amulet of Yendor placement)

## Dungeon — assigned to Wave 6

- ASTRAL_PLANE (ascension altar)
- EARTH_PLANE / AIR_PLANE / FIRE_PLANE / WATER_PLANE

---

## Observations — assigned to Wave 2

- `build_glyphs(env_state)`: project terrain + monsters + objects → glyph IDs using offset scheme
- `build_blstats(env_state)`: pack player_* fields into 27-vector at canonical indices
- `build_message(env_state)`: read `MessageState.message_buffer` into 256-byte obs slot
- `build_inventory_strings(env_state)`: render `objnam`-style strings with identification

## Observations — assigned to Wave 3

- `build_tty(env_state)`: render glyphs → ASCII 24×80 via objects/monsters/cmap lookup tables
- Pixel observation: use existing `Nethax/tiles/tiles.npy` atlas to render 16×16 sprites
- Symbolic observation: pack stats + map one-hot + inventory categorical into fixed-size vector

## Observations — assigned to Wave 4

- `screen_descriptions`: per-glyph "what does this mean" descriptions
- `program_state`: menu/dialog state
- `internal`: agent-visible internals (stairs_down, hunger_state, etc.)

---

## FOV (`fov.py`) — assigned to Wave 2

- Replace all-visible mask with ray-casting FOV using opacity table from tiles
- Shadow-casting algorithm for narrow corridors
- Blind mode: 1-tile radius regardless of map state

---

## Messages (`messages.py`) — assigned to Wave 4

- Template-based message-id system (JIT-friendly: each MessageId has a static template, dynamic args fill placeholders via JAX-arithmetic-on-bytes)
- Expand `MessageId` enum to cover all ~200 canonical NetHack messages
- Wire history ring buffer on every `emit`

---

## Save / Load — assigned to Wave 6

- Implement `save_state(state, path)`: pytree flatten → numpy save
- Implement `load_state(path)`: inverse
- Decide cross-version compatibility policy

---

## Scoring — assigned to Wave 6

- `compute_final_score(state, role, race, alignment, depth)`: NetHack's canonical formula
- Death message text generation
- Ascension condition check

---

## Action Dispatch — assigned to Wave 2 (movement) + Wave 3-6 (rest)

- Wave 2: 8 compass directions → tile-walk (collision, bump-attack, door-bump-open)
- Wave 2: `<` / `>` stair traversal
- Wave 3: `e` eat, `q` quaff, `r` read, `Z` cast, `z` zap
- Wave 3: `w` wield, `W` wear, `T` take-off, `P` put-on, `R` remove
- Wave 3: `d` drop, `,` pickup, `s` search, `o` open, `c` close, `^` get-trap-info
- Wave 4: `t` throw, `f` fire, `a` apply, `#pray`, `#sac`, `#chat`, `#dip`
- Wave 4: extended commands
- Wave 5: `>` traverse-stair-with-branch
- Wave 6: `S` save, `#quit`

---

## Top-level RNG / utilities — assigned to Wave 2

- `dice_roll(rng, n, sides)`: replace deterministic stub with `jax.random.randint` summed n times
- `weighted_choice(rng, weights)`: implement (currently returns 0)

---

## MiniHack parity (Wave 5)

Out of scope for current `Nethax/nethax/` package — lives in `Nethax/minihax/`. Will need:
- Port `LevelGenerator` API (`add_monster`, `add_trap`, `add_object`, `add_door`, `fill_terrain`, regions)
- Port `RewardManager` event system (`add_eat_event`, `add_kill_event`, `add_message_event`, `terminal_sufficient`)
- des-file → Python factory conversion (or des-file parser)
- 36 canonical des-files to port: `vendor/minihack/minihack/dat/{corridor*,exploremaze*,hidenseek*,lava_crossing,key_and_door,locked_door,quest*,soko*,memento*,closed_door}.des`
- Implement the 170 registered envs (`vendor/minihack/minihack/envs/__init__.py`)
