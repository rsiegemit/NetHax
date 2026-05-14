# Wave 4 — Mechanics Status

Per-subsystem status update from Wave 3 → Wave 4.  Each row records what moved from "stub" / "simplified" to "real," and what remains for Wave 5+.

Legend: ✅ real · ⚠️ simplified · ⏳ no-op (waiting on later wave)

## Action dispatch (`subsystems/action_dispatch.py`)

| Wave 3 status | Wave 4 status |
|---|---|
| Wired: movement (8 dir + 8 run) + stair_up + stair_down + wait. Action handlers existed in each subsystem but were NOT wired into `_HANDLERS`. | ✅ Wired: EAT (20), QUAFF (21), READ (22), ZAP (23), CAST (24), PICKUP (25), DROP (26), WIELD (27), WEAR (28), PUTON (29), REMOVE (30), OPEN (31), CLOSE (32), KICK (33), FIGHT (34), SEARCH (35), PRAY (36). Plus: `status_effects.step` now ticks every `env.step` (was missing in Wave 3). |
| `_ACTION_TO_HANDLER_IDX` only knew movement keys. | `_ACTION_TO_HANDLER_IDX` maps the 17 new ASCII / Command keys. |

37 handler slots total; ~50 % of the 121-action surface is now reachable.  The remaining 84 slots still fall through to the no-op (mostly menu / overview / debug commands like `INVENTORY`, `OPTIONS`, `WHATIS`, `VERSION`).

## Polymorph (`subsystems/polymorph.py`)

| Mechanic | Wave 3 | Wave 4 |
|---|---|---|
| `polymorph_player(state, rng, target_form, controlled)` | ⏳ stub | ✅ full fidelity: NATTK=6 attack-set swap, intrinsic mask from `MONSTERS[form].flags1/2`, HP_max from form's `level` roll, AC recompute via `_form_ac`, armor-drop if `~M1_HUMANOID \| M1_NOHANDS`, poly_timer ∈ [500, 1000), POLYSELFLESS conduct violated. (polyself.c::polymon) |
| `revert_polymorph(state, rng)` | ⏳ stub | ✅ Restores orig_str/dex/con/hp_max/ac, restores `orig_attack_*`, clears flags. (polyself.c::rehumanize) |
| `polymorph_monster(state, rng, slot, target_form)` | ⏳ stub | ✅ Per-slot form change, HP scaling, `orig_entry_idx` save. |
| Lycanthropy timer | ⏳ stub | ✅ `lycanthropy_timer` decremented every `step`; `trigger_lycanthropy` flips form when timer reaches 0. |
| `poly_trap_effect(state, rng)` | ⏳ stub | ✅ Uniform `rn2(NUMMONS)` form pick, dispatches `polymorph_player(controlled=False)`.  Not yet wired into `traps.dispatch` (Wave 5). |
| `step(state, rng)` | ⏳ no-op | ✅ Decrements poly_timer + lyc_timer; auto-reverts when poly_timer hits 0. |
| Tests | 0 | 20 |

## Prayer (`subsystems/prayer.py`)

| Mechanic | Wave 3 | Wave 4 |
|---|---|---|
| `pray(state, rng)` | ⏳ stub (returned state unchanged) | ✅ Full pipeline from pray.c:500-1500: detect-trouble path, angry gate (timeout > 0 ∨ record < threshold), d100 pleased buckets (0-29 HEAL_CURE, 30-49 PROTECTION, 50-64 REMOVE_CURSE, 65-74 GIFT_ARTIFACT, 75-99 no-op), `god_zaps_you` on angry. |
| `pray_timeout` updates | ⏳ static | ✅ Reset to `300 + rnz(700)` after pray; decremented in `step`. |
| `alignment_record ±1` adjust | ⏳ unwired | ✅ +1 on pleased, −1 on angry. |
| `sacrifice_on_altar(state, rng, slot)` | ⏳ stub | ✅ Corpse type → outcome (good, neutral, bad alignment match). Sets `pray_timeout`. |
| `altar_buc_sense(state)` | ⏳ no-op | ✅ Detects altar adjacency; reveals BUC of held item (objclass-restricted). |
| `handle_pray(state, rng)` (action-dispatch entry) | ⏳ no-op | ✅ Calls `pray`, then violates ATHEIST conduct. |
| Tests | 0 | 12 |

## Dungeon branches (`dungeon/branches.py`)

| Mechanic | Wave 3 | Wave 4 |
|---|---|---|
| `init_branch_graph(rng, static_params)` | ⏳ stub | ✅ Builds `stair_links` for Main↔Mines (Dlvl 3), Main↔Sokoban (Dlvl 6), Main↔Quest (Dlvl 12). |
| `generate_mines_level(rng, depth)` | ⏳ no-op | ✅ Cellular-automata cave layout, irregular boundary, monster placements (gnomes/dwarves at low depth), gems / luck items in deep levels. |
| `generate_sokoban_level(rng, floor_num)` | ⏳ no-op | ✅ One of 8 hand-encoded layouts per `floor_num`; placed boulders + pits per vendor `soko1.des`–`soko4.des`. |
| `generate_quest_level(rng, depth, role)` | ⏳ no-op | ✅ Per-role guardian monster (e.g., Knight → wraith); generic layout pending per-role artwork (Wave 5). |
| `apply_branch_graph_to_dungeon(dungeon, graph)` | — (new) | ✅ Projects `BranchGraphState.stair_links` onto `DungeonState.stair_links`, inferring up vs down direction. |
| `level_memory.traverse_stair_cross_branch` | ⏳ stub | ✅ Snapshots current level, generates/restores destination, repositions player onto matching stair tile. Tests in `test_dungeon_branches.py`. |
| Tests | 0 | 14 |

## Features (`subsystems/features.py`)

| Mechanic | Wave 3 | Wave 4 |
|---|---|---|
| Door open/close/kick/unlock | ✅ | ✅ unchanged |
| `quaff_fountain(state, rng)` | ⏳ stub | ✅ 16-outcome table from fountain.c::drinkfountain (wish, snake, water-demon, gold, gain stats, …). |
| `dip_fountain(state, rng, slot_idx)` | ⏳ stub | ✅ 8-outcome table from fountain.c::dipfountain (sword from stone, water-demon, identify, …). |
| `sit_on_throne(state, rng)` | ⏳ stub | ✅ 14-outcome table from sit.c::sit_on_throne (heal, energy, +XP, summon monsters, …). |
| `drink_sink(state, rng)` | ⏳ stub | ✅ 13-outcome table from dokick.c::drinksink (frog, dish, ring, gold, sleep gas, …). |
| `altar_buc_sense(state)` | ⏳ stub | ✅ Implemented in `subsystems/prayer.py`. |
| Tests | 0 | 35 (features + special levels) |

## Special levels (`dungeon/special_levels.py`)

| Level | Wave 3 | Wave 4 |
|---|---|---|
| Oracle | ⏳ | ✅ `generate_oracle_level` — 11×9 delphi + Oracle NPC + 4 fountains + treasure satellite rooms. (vendor/dat/oracle.lua) |
| Mine Town | ⏳ | ✅ `generate_mine_town` — irregular cave, 4 shop blocks (SHOP_FLOOR), defiled altar, 2 fountains, watchmen placements. (vendor/dat/minetn-1..7.lua) |
| Mines End | ⏳ | ✅ `generate_mines_end` — luckstone at center, 4 gnome-lord placements. (vendor/dat/minend.lua) |
| Big Room | ⏳ | ✅ `generate_big_room` — single 60×15 room with sparse pillars. |
| Castle | ⏳ | ⏳ Wave 5 |
| Vlad's Tower | ⏳ | ⏳ Wave 5 |
| Sanctum | ⏳ | ⏳ Wave 5 / 6 (ascension end-game) |

## Conduct (`subsystems/conduct.py`)

| Conduct | Wave 3 | Wave 4 | Wired at |
|---|---|---|---|
| FOODLESS | ⏳ | ✅ | `action_dispatch._handle_eat` |
| VEGAN | ⏳ | ✅ | same — material check via `food_material_for_type_id` |
| VEGETARIAN | ⏳ | ✅ | same |
| ATHEIST | ⏳ | ✅ | `prayer.handle_pray` |
| WEAPONLESS | ⏳ | ✅ | `combat.melee_attack` (with-weapon branch) |
| PACIFIST | ⏳ | ✅ | `combat.melee_attack` (monster-died branch) |
| ILLITERATE | ⏳ | ✅ (×2) | `scrolls.handle_read`, `spellbooks.handle_read` |
| POLYSELFLESS | ⏳ | ✅ | `polymorph.polymorph_player` |
| POLYPILELESS | ⏳ | ⏳ Wave 5 | (poly-trap-affects-pile branch not yet built) |
| GENOCIDELESS | ⏳ | ⏳ Wave 5 | (genocide scroll handler still TODO) |
| ELBERETHLESS | ⏳ | ⏳ Wave 5 | (engrave action not in dispatch yet) |
| WISHLESS | ⏳ | ⏳ Wave 6 | (wish handler is Wave 6) |
| ARTIWISHLESS | ⏳ | ⏳ Wave 6 | (artifact wish gated on wish handler) |

8 of 13 wired in Wave 4; the remaining 5 are gated on features that haven't been built.

## Observation (`obs/nle_obs.py`)

| Key | Wave 3 | Wave 4 |
|---|---|---|
| `glyphs` (21,79) int16 | ✅ | ✅ |
| `chars` (21,79) uint8 | ✅ | ✅ |
| `colors` (21,79) uint8 | ✅ terrain only | ✅ terrain + player tile (yellow 15) + ANSI per cmap |
| `specials` (21,79) uint8 | ⏳ zeros | ✅ 6-bit packed: corpse / pile / trap / secret-door / invis-monster / object-present |
| `blstats` (27,) int64 | ✅ | ✅ |
| `message` (256,) uint8 | ✅ | ✅ |
| `program_state` (6,) int32 | ⏳ zeros | ✅ (mostly zeros by design — nethax has no menus, but the field is populated for compatibility) |
| `internal` (9,) int32 | ⏳ zeros | ✅ stairs_down position, hunger_state, dlevel, encumbrance, score |
| `inv_glyphs` (55,) int16 | ✅ | ✅ |
| `inv_letters` (55,) uint8 | ✅ | ✅ |
| `inv_oclasses` (55,) uint8 | ✅ | ✅ |
| `inv_strs` (55, 80) uint8 | ✅ full fidelity | ✅ |
| `screen_descriptions` (21,79,80) | ⏳ zeros | ✅ per-glyph name lookup (terrain name, monster name) |
| `tty_chars` (24, 80) uint8 | ✅ | ✅ |
| `tty_colors` (24, 80) int8 | ✅ | ✅ |
| `tty_cursor` (2,) uint8 | ✅ | ✅ |
| `misc` (3,) int32 | ⏳ zeros | ⚠️ still zero in JAX env (Wave 5 — agent input flags, hard to project) |

**17 of 17 keys** now real-valued (Wave 3 had 13/17).  `misc` is the lone simplified key — it carries agent-side flags (in-menu / yn-prompt) that don't have analogues in the JAX env.

## Combat (`subsystems/combat.py`)

| Mechanic | Wave 3 | Wave 4 |
|---|---|---|
| `to_hit_roll`, `damage_roll`, `compute_ac`, `melee_attack`, `bump_attack`, `monster_attack_player`, `practice_skill` | ✅ | ✅ unchanged |
| **bump-attack bridge from `_try_step`** | ⏳ no | ⏳ no — still Wave 5; tests use direct `combat.melee_attack` calls |
| Two-weapon / ranged / breath / engulf | ⏳ | ⏳ Wave 5 |
| Polymorph combat (player attacks with polymorph form's attack set) | ⏳ | ⚠️ data is in place (`polymorph.attack_types` etc.) but `bump_attack` does not yet consult it; Wave 5 |

## Monster AI (`subsystems/monster_ai.py`)

Unchanged from Wave 3.  `monsters_step_all` is still NOT called from `env.step`.  Wave 5 priority.

## Status effects (`subsystems/status_effects.py`)

Unchanged formulas, but now ticked inside `env.step` (Phase 0 wiring).
