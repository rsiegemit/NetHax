# Wave 1 — Subsystem inventory

Every NetHack subsystem that Wave 1 stubbed, with state class, vendor citations, and which wave will fill in mechanics.

## Combat (`subsystems/combat.py`)

- **State class:** `CombatState` (`default()`)
- **Fields:** `weapon_skill[40]` int8, `weapon_practice[40]` int32, `last_attack_kind` int32, `last_hit_landed` bool
- **Public fns:** `melee_attack`, `ranged_attack`, `passive_attack`, `step` — all no-op
- **Canonical source:** `vendor/nethack/src/{uhitm,mhitu,mhitm,mthrowu,weapon,dothrow}.c`
- **Wave to implement:** Wave 3 (THAC0/AC/skill/damage), Wave 4 (passive, ranged, breath, engulf, polymorph combat)

## Magic (`subsystems/magic.py`)

- **State class:** `MagicState` (`default()`)
- **Fields:** `spell_memory[43]` int32, `spell_known[43]` bool, `spell_letter[43]` int8, `pw_regen_counter` int32
- **Enums:** `SpellSchool` (7), `SpellId` (43 spells seeded: HEALING, EXTRA_HEALING, MAGIC_MISSILE, FIRE_BOLT, ..., DIG, LIGHT, CLAIRVOYANCE)
- **Public fns:** `cast_spell`, `read_spellbook`, `step`
- **Canonical source:** `vendor/nethack/src/{spell,zap,mcastu}.c`
- **Wave to implement:** Wave 3 (cast + Pw + fail roll), Wave 4 (monster cast via `mcastu`, wand zap rays)

## Monster AI (`subsystems/monster_ai.py`)

- **State class:** `MonsterAIState` (`make_monster_ai_state()`)
- **Fields:** `movement_points[200]` int16, `mstrategy[200]` int8, `target_pos[200,2]` int16, `last_seen_player_pos[200,2]` int16, `tame[200]` bool, `peaceful[200]` bool
- **Enum:** `MoveStrategy` (10 strategies: NONE, SLEEP, WANDER, HUNT, FLEE, PARALYZE, WAIT, RETREAT, SUMMON, CONFUSED)
- **Public fns:** `monster_turn`, `pet_turn`, `wake_sleeping`, `step`
- **Canonical source:** `vendor/nethack/src/{monmove,dogmove,dog,muse,minion}.c`
- **Wave to implement:** Wave 3 (pathfinding, basic strategy), Wave 4 (muse item use, spells, pets)

## Polymorph (`subsystems/polymorph.py`)

- **State class:** `PolymorphState` (`make_polymorph_state()`)
- **Fields:** `poly_form_id` int32 (-1 = human), `poly_turns` int32, `poly_controlled` bool
- **Public fns:** `polymorph_player`, `unpolymorph`, `polymorph_monster`, `step`
- **Canonical source:** `vendor/nethack/src/{polyself,were}.c`
- **Wave to implement:** Wave 3 (timer tick, AC/attack recompute), Wave 4 (monster poly, polytrap, polypiles)

## Inventory (`subsystems/inventory.py`)

- **State class:** `InventoryState` (`empty()`)
- **Fields:** `items[52]: Item`, `wielded` int8, `off_hand` int8, `worn_armor[7]` int8, `worn_amulet` int8, `worn_rings[2]` int8, `quiver` int8, `total_weight` int32
- **Enum:** `ArmorSlot` (BODY, SHIELD, HELM, GLOVES, BOOTS, CLOAK, SHIRT)
- **Public fns:** `pickup`, `drop`, `wield`, `wear`, `take_off`, `put_on_ring`, `step`
- **Canonical source:** `vendor/nethack/src/{invent,pickup,do_wear,wield,worn,mkobj,objnam,o_init}.c`
- **Wave to implement:** Wave 3 (slot mutation, weight calc, capacity check)

## Items (`subsystems/items.py`)

- **State class:** `ItemEffects` per-item helper struct
- **Enums:** `BUCStatus` (UNKNOWN/CURSED/UNCURSED/BLESSED), `Erosion` (NONE/RUSTY1-3/CORRODED1-3/BURNT1-3/ROTTED1-3)
- **Public fns:** `apply_blessing`, `apply_curse`, `erode`, `enchant`
- **Canonical source:** `vendor/nethack/src/{mkobj,objnam,potion,read}.c`
- **Wave to implement:** Wave 3 (potion/scroll/wand effect dispatch), Wave 4 (BUC sensing, full enchantment/erosion ticks)

## Identification (`subsystems/identification.py`)

- **State class:** `IdentificationState` (`unshuffled()`)
- **Fields:** `potion_appearance[26]`, `scroll_appearance[43]`, `wand_appearance[28]`, `ring_appearance[28]`, `amulet_appearance[13]`, `spellbook_appearance[46]`, `identified[NUM_OBJECTS]` bool
- **Public fns:** `init_shuffled_appearances`, `partial_identify`, `full_identify`, `check_known`
- **Canonical source:** `vendor/nethack/src/{o_init,insight,objnam}.c`
- **Wave to implement:** Wave 3 (Fisher-Yates shuffle at game-start via `jax.random.permutation`)

## Traps (`subsystems/traps.py`)

- **State class:** `TrapState` (`default(num_levels, map_h, map_w)`)
- **Fields:** `trap_type[L, H, W]` int8, `revealed[L, H, W]` bool
- **Enum:** `TrapType` — 26 entries (ARROW_TRAP, DART_TRAP, ROCKTRAP, BEAR_TRAP, LANDMINE, SLP_GAS, RUST, FIRE, PIT, SPIKED_PIT, HOLE, TRAPDOOR, TELEP, LEVEL_TELEP, MAGIC_PORTAL, WEB, STATUE_TRAP, MAGIC, ANTI_MAGIC, POLY, VIBRATING_SQUARE, ...)
- **Public fns:** `place_trap`, `trigger_trap`, `reveal_trap`, `step`
- **Canonical source:** `vendor/nethack/src/trap.c` (7,211 lines)
- **Wave to implement:** Wave 3 (damage traps), Wave 4 (teleport/poly/anti-magic), Wave 5 (vibrating square gateway)

## Features (`subsystems/features.py`)

- **State class:** `FeaturesState` (`default(num_levels, map_h, map_w)`)
- **Fields:** `fountains_used[L,H,W]`, `thrones_used[L,H,W]`, `sinks_used[L,H,W]`, `altar_alignment[L,H,W]` int8, `door_state[L,H,W]` int8
- **Enums:** `FountainEffect` (16), `ThroneEffect` (14), `SinkEffect` (13), `AltarAction` (5), `DoorState` (6)
- **Public fns:** `quaff_fountain`, `dip_fountain`, `sit_throne`, `kick_sink`, `sacrifice_on_altar`, `open_door`, `close_door`, `kick_door`, `unlock_door`, `step`
- **Canonical source:** `vendor/nethack/src/{fountain,sit,dokick,lock,dbridge}.c`
- **Wave to implement:** Wave 3 (door mechanics), Wave 4 (fountain/throne/sink/altar effect tables), Wave 5 (drawbridge, secret doors)

## Prayer (`subsystems/prayer.py`)

- **State class:** `PrayerState` (`default()`)
- **Fields:** `alignment` int32 (-1000..+1000), `prayer_timeout` int32, `luck` int32 (-13..+13), `lucky_stones` int32, `god_anger` int32
- **Enums:** `Alignment` (CHAOTIC=0/NEUTRAL=1/LAWFUL=2/UNALIGNED=3), `PrayerOutcome` (BLESSED, HEALED, IGNORED, CHASTISED, SMITTEN, ANGER_BOLT)
- **Public fns:** `pray`, `adjust_alignment`, `adjust_luck`, `step`
- **Canonical source:** `vendor/nethack/src/{pray,priest,minion}.c`
- **Wave to implement:** Wave 4 (divine intervention table from pray.c lines ~500-1500)

## Conduct (`subsystems/conduct.py`)

- **State class:** `ConductState` (`default()`)
- **Fields:** `violations[13]` bool
- **Enum:** `Conduct` (FOODLESS, VEGAN, VEGETARIAN, ATHEIST, WEAPONLESS, PACIFIST, ILLITERATE, POLYPILELESS, POLYSELFLESS, WISHLESS, ARTIWISHLESS, GENOCIDELESS, ELBERETHLESS)
- **Public fns:** `violate`, `step`
- **Canonical source:** `vendor/nethack/src/insight.c` + per-action trigger sites
- **Wave to implement:** Wave 4 (trigger wiring in each relevant subsystem)

## Shop (`subsystems/shop.py`)

- **State class:** `ShopState` (`default()`)
- **Fields:** `shopkeeper_pos[L,2]` int16, `shop_type[L]` int8, `shopkeeper_hp[L]` int16, `shopkeeper_angry[L]` bool, `bill[L, 52]` int32
- **Public fns:** `enter_shop`, `pickup_in_shop`, `pay_bill`, `attack_shopkeeper`, `step`
- **Canonical source:** `vendor/nethack/src/{shk,shknam}.c` (6,125 LOC — full fidelity out of scope)
- **Wave to implement:** Wave 4 simplified buy/sell only (no haggling, no theft detection, no chat). Confirmed deliberate scope reduction.

## Quest (`subsystems/quest.py`)

- **State class:** `QuestState` (`default()`)
- **Fields:** `stage` int8, `nemesis_alive` bool, `artifact_carried` bool, `leader_pos[2]` int16, `nemesis_pos[2]` int16
- **Enum:** `QuestStage` (NOT_STARTED, ENTERED_QUEST, LEADER_GREETED, NEMESIS_KILLED, ARTIFACT_RECOVERED, RETURNED_TO_LEADER)
- **Public fns:** `enter_quest_branch`, `talk_to_leader`, `slay_nemesis`, `pickup_artifact`, `step`
- **Canonical source:** `vendor/nethack/src/{quest,questpgr,role}.c`
- **Wave to implement:** Wave 4 (per-role quest tables), Wave 5 (quest level generation)

## Status Effects (`subsystems/status_effects.py`)

- **State class:** `StatusState` (`default()`) — initial nutrition = 900 per `eat.c:129`
- **Fields:** `intrinsics[69]` bool, `timed_intrinsics[69]` int32, `timed_statuses[24]` int32, `hunger_state` int8, `nutrition` int32, `encumbrance` int8, `sick_kind` int8, `hp_regen_counter` int32, `pw_regen_counter` int32
- **Enums:** `Intrinsic` (39 entries spanning resistances, vision, transport, attributes), `TimedStatus` (24 timed effects), `HungerState` (SATIATED..STARVED), `Encumbrance` (UNENCUMBERED..OVERLOADED)
- **Public fns:** `add_intrinsic`, `remove_intrinsic`, `add_timed`, `add_timed_intrinsic`, `tick_timers`, `compute_hunger_state`, `compute_encumbrance`, `step`
- **Canonical source:** `vendor/nethack/src/{attrib,timeout,eat,detect}.c`, `include/{prop,youprop}.h`
- **Wave to implement:** Wave 3 (threshold tables, regen formulas, expiry callbacks), Wave 4 (sickness/slime progression, confusion action remapping)

## Scoring (`subsystems/scoring.py`)

- **State class:** `ScoringState` (`default()`)
- **Fields:** `score` int32, `monsters_killed` int32, `achievements[N_ACHIEVEMENTS]` bool, `turns` int32
- **Enum:** `Achievement` (8 entries: ENTERED_GNOMISH_MINES, ENTERED_SOKOBAN, COMPLETED_SOKOBAN, GOT_LUCKSTONE, ENTERED_GEHENNOM, GOT_AMULET, ENTERED_ELEMENTAL_PLANES, ASCENDED)
- **Public fns:** `add_score`, `record_kill`, `record_achievement`, `compute_final_score` — `add_score`/`record_*` are fully implemented (trivial); `compute_final_score` is stub
- **Canonical source:** `vendor/nethack/src/{topten,end,exper}.c`
- **Wave to implement:** Wave 6 (NetHack scoring formula)

## Messages (`subsystems/messages.py`)

- **State class:** `MessageState` (`default()`)
- **Fields:** `message_buffer[256]` uint8, `message_history[20, 256]` uint8, `history_index` int32
- **Enum:** `MessageId` (placeholder: GAME_START, YOU_DIE, YOU_KILL_MONSTER, FIND_GOLD, OPEN_DOOR, EAT_FOOD)
- **Public fns:** `emit`, `clear_message`
- **Canonical source:** `vendor/nethack/src/pline.c`
- **Wave to implement:** Wave 4 — template-based message-id system (JIT-friendly, no Python format strings inside step)

## Action Dispatch (`subsystems/action_dispatch.py`)

- **Fn:** `dispatch_action(state, action, rng) -> state`
- **Module constant:** `ACTION_HANDLERS` — tuple of 121 identity functions, indexed by `_action_value_to_index(Action.value)`
- **Canonical source:** `vendor/nethack/src/cmd.c` (5,704 LOC)
- **Wave to implement:** Wave 2 movement, Wave 3-6 remaining 100+ actions
- **Pattern:** Wave 2+ will use `jax.lax.switch(handler_index, ACTION_HANDLERS, state, rng)`

---

## Dungeon (`dungeon/`)

| Module | State class | Key types | Fns | Source |
|---|---|---|---|---|
| `rooms.py` | `Room` (no factory; built per-level) | `RoomType` (26 values: OROOM..CANDLESHOP from `mkroom.h`) | `generate_rooms`, `connect_rooms` | `vendor/nethack/src/mklev.c`, `mkroom.c` |
| `mazes.py` | (no state) | — | `generate_maze_kruskal`, `generate_maze_perfect`, `generate_maze_dla` | `vendor/nethack/src/mkmaze.c` |
| `corridors.py` | (no state) | — | `connect_segments`, `place_doors` | `vendor/nethack/src/mklev.c` |
| `branches.py` | `DungeonState` (composed in master) | `Branch` (7), `BranchInfo`, `BRANCH_TABLE` | `current_dungeon_level`, `traverse_stair`, `enter_branch` | `vendor/nethack/src/dungeon.c`, `include/dungeon.h` |
| `special_levels.py` | (no state) | `SpecialLevel` (28: ORACLE..WATER_PLANE) | `generate_special_level` | `vendor/nethack/dat/*.des`, `vendor/nethack/src/sp_lev.c` |
| `level_memory.py` | `LevelMemoryState` (`make_empty_level_memory()`) | — | `enter_level`, `leave_level` | implicit in `vendor/nethack/src/dungeon.c` |

`MAX_LEVELS_PER_BRANCH = 32` (matches `MAXLEVEL`/`global.h`); `N_BRANCHES = 7`; `MAP_H = 21`, `MAP_W = 80` (NLE convention).

---

## Observation builders (`obs/`)

| Module | Purpose | Output | Status |
|---|---|---|---|
| `nle_obs.py` | NLE-parity 17-key observation dict | `Dict[str, jax.Array]` with shapes/dtypes matching `vendor/nle/include/nleobs.h` | Wave 1: returns zero arrays; Wave 2 wires projection |
| `symbolic_obs.py` | Flat float vector for compact baselines | `jnp.ndarray((1024,))` placeholder | Wave 2 projects map+stats+inventory |
| `pixel_obs.py` | RGB sprite render via `Nethax/tiles/tiles.npy` | `uint8[H*16, W*16, 3]` | Wave 2 uses existing tile atlas |
| `text_obs.py` | 24×80 ASCII tty grid | `uint8[24, 80]` | Wave 2 renders glyphs through `objects`/`monsters`/`cmap` lookups |

See [`nle-parity.md`](nle-parity.md) for the obs key contract.

---

## Constants (`constants/`)

| Module | Key exports | Verified against vendor |
|---|---|---|
| `actions.py` | `Action`, `ACTIONS` (121), `USEFUL_ACTIONS` (101), enum classes for each category | ✅ exec'd vendor/nle/nle/nethack/actions.py and confirmed counts |
| `glyphs.py` | 13 `GLYPH_*_OFF` constants, `MAX_GLYPH`, `NO_GLYPH`, category sizes | ⚠️ values from `pynethack.cc` — Wave 2 should cross-check with a live NLE build |
| `blstats.py` | 27 `BL_*` indices, `BL_MASK_*` condition flags, `N_BLSTATS=27` | ✅ matches `vendor/nle/include/nleobs.h:16-43` |
| `roles.py` | `Role` (13: ARCHEOLOGIST..WIZARD) | ✅ matches `vendor/nethack/src/role.c` |
| `races.py` | `Race` (5: HUMAN, ELF, DWARF, GNOME, ORC) | ✅ canonical |
| `monsters.py` | `MonsterEntry` schema (19 fields), 10 seed entries, `NUMMONS=394` | ⚠️ schema confirmed; full table pending Wave 2 |
| `objects.py` | `ObjectEntry` schema, 10 seed entries, `NUM_OBJECTS=459` | ⚠️ schema confirmed; full table pending Wave 2 |

---

## Top-level utilities

| Module | Purpose | Status |
|---|---|---|
| `state.py` | Master `EnvState` + `StaticParams` | ✅ wires all 17 subsystem slices |
| `env.py` | `NethaxEnv` class with NLE-style `reset()`/`step()` | ✅ reset+step+JIT all verified |
| `fov.py` | `compute_fov`, `update_explored`, sight radius constants | Wave 1: returns all-visible; Wave 2 implements raycast/shadowcast |
| `rng.py` | JAX PRNG conventions: `split_n`, `dice_roll`, `weighted_choice` | `split_n` works; others return deterministic placeholders |
| `save_load.py` | `save_state`/`load_state` for pytree persistence | Wave 6 |
