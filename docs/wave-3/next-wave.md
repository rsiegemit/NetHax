# Wave 3 → Wave 4 — Scope Preview

Wave 4's job: **deliver the RL benchmark surface that researchers actually use** — the 170-env MiniHack curriculum on top of `NethaxEnv`.

Plus: finish action dispatch wiring, ship dungeon branches, polymorph, prayer outcomes.

## Wave 4 deliverable

After Wave 4:
- `Nethax.minihax.MinihaxEnv(env_id: str)` creates any of the 170 canonical MiniHack envs.
- `gym.make("MiniHack-Room-5x5-v0")` (or similar) works after a thin adapter.
- A player can descend from the Main dungeon into Gnomish Mines, Sokoban, Quest.
- All 40+ action handlers are wired into `dispatch_action`.
- Prayer produces real outcomes.
- Polymorph works for player + monsters.

This is the **"RL-runnable"** wave. After Wave 4, the project can host actual benchmark training runs.

## Wave 4 breadth pass

### MiniHack package — biggest single item

The Wave 2 audit catalogued the entire MiniHack surface. Implementation plan:

1. **`LevelGenerator` API port** (~500 LoC)
   - `add_room`, `add_corridor`, `add_door`, `add_monster`, `add_trap`, `add_object`, `add_stair`, `fill_terrain`, `set_start_pos`, `set_goal_pos`
   - Returns a callable that produces a fully populated `EnvState`

2. **`RewardManager` API port** (~300 LoC)
   - 15 event factory methods: `add_eat_event`, `add_kill_event`, `add_message_event`, `add_pickup_event`, `add_wield_event`, `add_wear_event`, `add_amulet_event`, `add_levitate_event`, `add_positional_event`, `add_coordinate_event`, `add_location_event`, `add_custom_reward_fn`
   - Reward state with `repeatable`, `terminal_required`, `terminal_sufficient` flags
   - Per-step evaluation against current state

3. **`des-file` → Python factory** (~400 LoC)
   - Parser for the 36 canonical `*.des` files in `vendor/minihack/minihack/dat/`
   - OR: hand-translate each into Python factory functions (might be simpler given JAX constraints)

4. **170 env registrations** (~200 LoC)
   - Room (12 variants), Corridor (3), MazeWalk (6), HideNSeek (4), KeyRoom (7), LavaCross (15), Sokoban (8), Labyrinth (2), River (5), MultiRoom (16), MiniGrid ports (27), Quest (3), Memento (3), WoD (8), Boxoban (3), Skill suite (36), Crossing baselines (4), Navigation custom (2)

5. **Custom reward shapes**
   - Sokoban: -0.001/step + 0.1 per pit filled
   - LavaCross: terminal +1 on goal
   - Various: terminal on specific tile

### Action dispatch completion (Wave 4 first task)

Wire each subsystem's `handle_<action>` into `_HANDLERS` and `_ACTION_TO_HANDLER_IDX`:

- `handle_eat` → `Action.EAT` (ord('e'))
- `handle_quaff` → `Action.QUAFF` (ord('q'))
- `handle_read` → `Action.READ` (ord('r'))
- `handle_zap` → `Action.ZAP` (ord('z'))
- `handle_cast` → `Action.CAST` (ord('Z'))
- `handle_pickup` → `Action.PICKUP` (ord(','))
- `handle_drop` → `Action.DROP` (ord('d'))
- `handle_wield` → `Action.WIELD` (ord('w'))
- `handle_wear` → `Action.WEAR` (ord('W'))
- `handle_put_on` → `Action.PUTON` (ord('P'))
- `handle_remove` → `Action.REMOVE` (ord('R'))
- `handle_open` → `Action.OPEN` (ord('o'))
- `handle_close` → `Action.CLOSE` (ord('c'))
- `handle_kick` → `Action.KICK` (Ctrl-d)
- `handle_search` → `Action.SEARCH` (ord('s'))
- `handle_fight` → `Action.FIGHT` (ord('F'))

After wiring, every direct-callable handler becomes reachable from `env.step(state, action, rng)`.

### Dungeon branches

- `dungeon/branches.py::init_branch_graph(rng, static_params)` — at game start, build the `stair_links` array linking Main → Mines, Main → Sokoban, etc.
- `dungeon/branches.py::generate_mines_level(rng, depth)` — Mines style (irregular mazes + caves)
- `dungeon/branches.py::generate_sokoban_level(rng, floor_number)` — pick from 8 hand-designed Sokoban levels
- `dungeon/branches.py::generate_quest_level(rng, depth, role)` — per-role Quest layout
- `dungeon/level_memory.py::traverse_stair_cross_branch(state, rng, branch, direction)` — handles cross-branch stair traversal

### Polymorph

- `subsystems/polymorph.py::polymorph_player(state, rng, target_form, controlled)`:
  - Save original stats/inventory/attacks
  - Swap player_role / player_str / player_dex / player_con / etc. to target_form's
  - Update attack-set via monster_table
  - Recompute AC
- `polymorph.py::polymorph_monster(state, rng, monster_idx, target_form)` — same for monster slots
- Wire poly_trap, polymorph wand, polymorph spell
- Lycanthropy timer in `subsystems/polymorph.py::step`

### Prayer + alignment

- `subsystems/prayer.py::pray(state, rng)` — full outcome table from `pray.c:500-1500`
- Divine intervention paths: heal/cure/protection/gift_artifact/smite/anger_bolt
- `pray.c::god_zaps_you` simplified
- Sacrifice on altar: corpse type → outcome
- Conduct integration: ATHEIST violated on pray
- Wire `handle_pray` action

### Conduct trigger wiring

Each conduct check at its violation site:

- FOODLESS / VEGAN / VEGETARIAN: in `handle_eat`
- ATHEIST: in `handle_pray`
- WEAPONLESS: in `melee_attack`
- PACIFIST: in any kill event (combat.py monster-died branch)
- ILLITERATE: in `handle_read` (scroll/spellbook)
- POLYPILELESS: in poly-trap-affects-pile
- POLYSELFLESS: in `polymorph_player`
- WISHLESS: in wish-grant (handler for `WISH` action — Wave 6)
- ARTIWISHLESS: artifact-wish (Wave 6)
- GENOCIDELESS: in genocide-cast
- ELBERETHLESS: in engrave-Elbereth

### Observation polish (remaining 4 keys)

- `colors` overlay for monsters/items
- `specials` flags (trap/pile/corpse/secret/invisible)
- `program_state` (menu/dialog state, mostly zero in JAX env)
- `internal` (stairs_down location, hunger_state, dlevel)
- `screen_descriptions` (per-glyph descriptions for debugging)

### Features completion

- Fountain effect table (16 effects from `fountain.c::dryup`/`gushforth`)
- Throne effect table (14 effects from `sit.c::sit_on_throne`)
- Sink effect table (13 effects from `dokick.c::drinksink`)
- Altar BUC sense (Wave 4 integrates with prayer)

### Special levels (subset)

- Oracle level (~3 rooms with Oracle NPC; consult = pay gold for hint)
- Mine Town (~10 shops + temple + watchmen)
- Mines End (luckstone room)
- Big Room (large single room)

## Wave 4 risks

1. **MiniHack des-file parser**: NetHack's des-format is complex (random selections, conditional blocks, named regions). Hand-translation to Python factories might be safer than building a parser. Decision needed at Wave 4 start.

2. **Action dispatch wiring causes mass test changes**: many Wave 3 tests use direct function calls; after wiring, those should also work via `env.step`. Wave 4 should add an integration-only test layer that goes through dispatch.

3. **Polymorph state swap is invasive**: changing `player_role` mid-game cascades into combat (attack-set), HP_MAX recomputation, intrinsic propagation. Wave 4 should land polymorph late after MiniHack is stable.

4. **Cross-branch level memory**: descending into Mines from Main level 3, then ascending back, must preserve Main level 3 state exactly. Wave 2 has the infrastructure (`level_memory.enter_level/leave_level`), but cross-branch traversal hasn't been exercised end-to-end yet.

5. **Bottleneck: agents and shared dispatch table**: `action_dispatch.py` is one file. The action-wiring task can't be split across parallel agents without merge conflicts. Solution: dispatch-wiring is a SINGLE agent run before MiniHack agents start.

## Recommended Wave 4 launch shape

Sequential phases, with parallelism within each phase:

### Phase 0 — dispatch wiring (1 agent, blocker)
- Wire all 40+ handlers into action_dispatch
- Unskip relevant tests

### Phase 1 — MiniHack core (parallel agents)
- LevelGenerator API agent
- RewardManager API agent
- Des-file translation agent (or split: weapons-related des, navigation des, etc.)
- 170-env registration agent

### Phase 2 — game-mechanic depth (parallel agents)
- Dungeon branches + level memory cross-traversal
- Polymorph (player + monster)
- Prayer + alignment + altar interactions
- Conduct wiring
- Fountain/throne/sink effects
- Wave 4 special levels (Oracle, Mine Town)

### Phase 3 — obs polish (1 agent)
- Wire remaining 4 obs keys
- Add color overlays

### Phase 4 — integration tests (1 agent)
- End-to-end MiniHack env tests
- Cross-branch traversal tests
- Polymorph tests

~10-15 agents total.

## Open questions for Wave 4

1. **Des-file parser or hand-translate?** Hand-translate is safer for JAX-compatibility (full Python control) but means 36 ad-hoc files. Parser is more maintainable but adds 400+ LoC. **My recommendation: hand-translate, document each factory.** Open to redirect.

2. **Wrap or replace `Nethax/minihax/` package?** Current package uses pre-Wave-1 state. Wave 4 can either (a) build new `MinihaxEnv` class alongside, leaving old as legacy, or (b) refactor in place. **My recommendation: (a) — leave legacy until Wave 5/6 confidence.**

3. **MiniHack reward shape**: the canonical reward is sparse (terminal +1). Should we offer dense alternatives via `RewardManager` flags? Vendor does this. **My recommendation: match canonical, expose `RewardManager` for users to customize.**

4. **Polymorph fidelity**: full attack-set swap + intrinsic gain/loss vs simplified "swap role + stats only"? Full fidelity touches ~10 subsystems. **My recommendation: full fidelity since user picked it for inv_strs — be consistent.**

5. **Prayer god-anger probabilities**: vendor uses many random rolls. Implement exact d100 chains or approximate? **My recommendation: exact, since gen approach is identical to combat/spell-cast (already done).**

Defaults all marked ★. Reply with picks or "all defaults".
