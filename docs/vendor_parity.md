# Vendor Parity Matrix

Per-subsystem record of which vendor source files each nethax module is ported from, and how close the parity is. "Full" means a formula / table / API surface ported byte-for-byte; "high" means structurally faithful with documented simplifications; "simplified" means deliberately reduced scope; "divergent" means an intentional design departure.

Vendor source paths are relative to `vendor/nethack/` (NetHack 3.7-branch HEAD) and `vendor/nle/` (NLE 1.3.0 series).

For the higher-level architecture, see [`architecture.md`](architecture.md). For NLE migration details, see [`nle_migration.md`](nle_migration.md).

---

## Core game loop + dispatch

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Env step (moveloop) | `Nethax/nethax/env.py::_step_impl` | full | `vendor/nethack/src/allmain.c::moveloop` lines ~200-360 |
| Action dispatch | `Nethax/nethax/subsystems/action_dispatch.py::_HANDLERS` | full surface (43 slots) | `vendor/nethack/src/cmd.c` (per-handler citations inline) |
| RNG primitives | `Nethax/nethax/rng.py` (`rn2`, `rnd`, `rn1`, `dice_roll`, `weighted_choice`) | full | `vendor/nethack/src/rnd.c` |

---

## Combat

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Melee to-hit (THAC0) | `subsystems/combat.py::bump_attack` | full | `vendor/nethack/src/uhitm.c::known_hitum`, `weapon.c::hitval` |
| Damage roll | `subsystems/combat.py::roll_damage` | full | `vendor/nethack/src/uhitm.c::damageum`, `weapon.c::dmgval` |
| AC computation (per-slot) | `subsystems/combat.py::compute_ac` | full | `vendor/nethack/src/do_wear.c::Armor` table |
| Two-weapon toggle | `subsystems/combat.py::handle_twoweapon` | full | `vendor/nethack/src/wield.c::dotwoweapon` |
| Thrown attack pipeline | `subsystems/combat.py::thrown_attack` | high (lodge-in-target deferred) | `vendor/nethack/src/dothrow.c` |
| Bump-attack from movement | `_try_step` in `action_dispatch.py` | full | `vendor/nethack/src/hack.c::domove` |
| Polymorph attack integration | `combat.bump_attack` reads `state.polymorph.attack_*` | full | `vendor/nethack/src/polyself.c` |

---

## Magic

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Spell casting (all 43 spells) | `subsystems/magic.py::cast_spell` + per-effect handlers | full | `vendor/nethack/src/spell.c::docast`, `spell.c::percent_success` |
| Pw regeneration | `subsystems/status_effects.py::pw_regen_tick` | full | `vendor/nethack/src/allmain.c` lines 305-320 |
| Spellbook study | `subsystems/items_spellbooks.py` | full | `vendor/nethack/src/read.c::study_book` |
| `age_spells` decay | `_step_impl` step 6 | full | `vendor/nethack/src/spell.c::age_spells` |
| Detection-spell timers (light, magic-mapping) | `subsystems/identification.py` + `dungeon/branches.py::lit_radius_until_turn` | full | `vendor/nethack/src/detect.c`, `light.c::do_light_sources` |

---

## Monsters

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Monster table (381 entries) | `Nethax/nethax/constants/monsters.py` + `monster_entries/chunk{1..6}.py` | full | `vendor/nethack/include/monst.h`, `monsters.h` |
| Monster AI (move + greedy 8-dir + LoS) | `subsystems/monster_ai.py::monster_turn` | high | `vendor/nethack/src/monmove.c`, `mon.c::movemon` |
| BFS pathfind (depth-12) | `monster_ai.py::_bfs_step` | high (vendor uses A*; nethax bounded BFS) | `vendor/nethack/src/mhitu.c` |
| Sleep / wake | `monster_ai.py` | full | `vendor/nethack/src/monmove.c` |
| Item use (muse) | `monster_ai.py::monster_use_item` | high (random init muse_init deferred) | `vendor/nethack/src/muse.c::find_misc` |
| Spell casting (mcastu) | `monster_ai.py::monster_cast_spell` | full (spell selection via `MS_SPELL` flag via `msound`) | `vendor/nethack/src/mcastu.c::buzzmu` |
| Retreat (HP < 1/7) | `monster_ai.py::maybe_retreat` | full | `vendor/nethack/src/monmove.c` |
| Pet AI (follow + peaceful) | `monster_ai.py::pet_move` | high (tameness decay deferred) | `vendor/nethack/src/dogmove.c`, `dog.c` |
| Spawning (`monstr` difficulty) | `dungeon/spawning.py` | full (uses canonical `MonsterEntry.difficulty`) | `vendor/nethack/src/makemon.c::monstr_init`, `mon.c::makemon` |
| Step-all 200 slots | `monster_ai.py::monsters_step_all` (single `lax.scan`) | full | n/a (JAX-native impl of vendor per-monster loop) |

---

## Objects

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Object table (453 canonical + 50 alias entries = 503 slots) | `Nethax/nethax/constants/objects.py` + `object_entries/` | full (alias dedup deferred) | `vendor/nethack/include/objects.h`, `objects.c` |
| Object class flags + `oc_subtyp` family | `constants/objects.py` | full (Wave 6 closing-audit fields) | `vendor/nethack/include/objclass.h` |
| Material (oc_material) | `constants/objects.py` field | full | `vendor/nethack/include/objclass.h::oc_material` |
| Pickup / drop | `subsystems/inventory.py` | full | `vendor/nethack/src/pickup.c::dopickup`, `pickup.c::dodrop` |
| Wield / wear / putOn / remove | `subsystems/inventory.py` | full | `vendor/nethack/src/wield.c`, `do_wear.c` |
| Item effects (potions, scrolls, wands, rings, amulets) | `subsystems/items_*.py` (5 files) | full (148 effect handlers) | `potion.c`, `read.c`, `zap.c`, `do_wear.c`, `mkobj.c` |
| Wand ray (Bresenham 8-step scan) | `subsystems/items_wands.py::ray_step` | full | `vendor/nethack/src/zap.c::buzz` |

---

## Spells

(Spell mechanics share infrastructure with Magic above.)

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| All 43 spell effects | `subsystems/magic.py` per-spell handlers | full | `vendor/nethack/src/spell.c`, `zap.c`, `detect.c` |
| Spell cost / success roll | `subsystems/magic.py::cast_spell` | full (Wave 6 #73 fix: `percent_success` returns SUCCESS pct) | `vendor/nethack/src/spell.c::percent_success` |
| Spell memory decay | step 6 of `_step_impl` | full | `vendor/nethack/src/spell.c::age_spells` |

---

## Status effects + timers

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Hunger thresholds + tick | `subsystems/status_effects.py::hunger_tick` | full (Wave 6 #73 vendor-exact thresholds) | `vendor/nethack/src/eat.c::newuhs` lines 3369-3372 |
| HP regen | `status_effects.py::hp_regen_tick` | full | `vendor/nethack/src/allmain.c` line 294 |
| Pw regen | `status_effects.py::pw_regen_tick` (canonical) | full | `vendor/nethack/src/allmain.c` line 305 |
| `nh_timeout` (status decay) | `status_effects.py::tick_timers` | full (Wave 6 #77 spec windows) | `vendor/nethack/src/timeout.c::nh_timeout` |
| Stoning / sliming / strangulation / food-poisoning death cycles | `status_effects.py` | full | `vendor/nethack/src/timeout.c::stoned_dialogue` etc. |
| Encumbrance formula | `status_effects.py` | full | `vendor/nethack/src/hack.c::inv_weight` |

---

## Hunger

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Hunger state + thresholds | `status_effects.py` (HUNGRY/WEAK/FAINTING/STARVED) | full | `vendor/nethack/include/hack.h` |
| Eat handler (all 33 food objects) | `subsystems/action_dispatch.py::_handle_eat` -> `items.py` | full | `vendor/nethack/src/eat.c::doeat` |
| Material-driven food conducts | `subsystems/conduct.py` (pulls `oc_material` from OBJECTS) | full | `vendor/nethack/src/eat.c`, `objects.h` |

---

## Prayer

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| `pray()` outcome chain | `subsystems/prayer.py::pray` | full (16-bucket d100 dispatch) | `vendor/nethack/src/pray.c::dopray` lines 500-1500 |
| Trouble detection | `prayer.py::_detect_trouble` | full (12 trouble flags) | `vendor/nethack/src/pray.c::in_trouble` |
| Sacrifice on altar | `prayer.py::sacrifice_on_altar` | full | `vendor/nethack/src/pray.c::dosacrifice` |
| Altar BUC sense | `prayer.py::altar_buc_sense` | full | `vendor/nethack/src/pray.c::altar_wrath` |
| Anger / wrath path | `prayer.py::god_zaps_you` | full | `vendor/nethack/src/pray.c::god_zaps_you` |
| `prayer.in_region` flag | `PrayerState.in_region` | full | `vendor/nethack/src/region.c` |

---

## Roles

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| 13 role table | `Nethax/nethax/constants/roles.py` (Wave 6 Phase B+ parity port) | full | `vendor/nethack/src/role.c::roles[]` |
| Starting stats | `constants/roles.py::STARTING_STATS` | full | `vendor/nethack/src/u_init.c` |
| Starting inventory | `subsystems/character.py::STARTING_INVENTORY` | full (13 roles) | `vendor/nethack/src/u_init.c::u_init` |
| Quest leader / nemesis / guardian / artifact | `dungeon/quest_levels.py` | full | `vendor/nethack/src/role.c::roles[]` (per-role fields) |

---

## Races

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| 5-race table (human, elf, dwarf, gnome, orc) | `Nethax/nethax/constants/races.py` (Wave 6 Phase B+ parity port) | full | `vendor/nethack/src/role.c::races[]` |
| Intrinsics (cold/fire/sleep/poison resist) per race | `constants/races.py` | full | `vendor/nethack/src/role.c::races[]` (intrinsic fields) |

---

## Character spawning

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| `create_character(rng, role, race, alignment)` | `subsystems/character.py::create_character` | full (stats + inventory + AC) | `vendor/nethack/src/u_init.c::u_init` |
| Role-stat-derived player init | `character.py` | full (Wave 6 Phase B+ vendor parity for HP/PW derivation) | `vendor/nethack/src/u_init.c` |

---

## Traps

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| 26 trap-type table | `subsystems/traps.py::TRAP_EFFECTS` | full | `vendor/nethack/include/trap.h`, `trap.c::trap_types` |
| `dotrap` dispatch (lax.switch) | `traps.py::trigger_trap` | full | `vendor/nethack/src/trap.c::dotrap` |
| POLY_TRAP / RUST_TRAP / STATUE_TRAP | `traps.py` (wired to subsystem effects) | full | `vendor/nethack/src/trap.c` |
| LEVEL_TELEP / MAGIC_PORTAL | `traps.py` -> dungeon transit | full | `vendor/nethack/src/trap.c::level_tele`, `dotele` |
| VIBRATING_SQUARE -> Gehennom portal | `traps.py` + `dungeon/branches.py::generate_gehennom_level` | full | `vendor/nethack/src/trap.c`, `mkmaze.c::mkgehennom` |

---

## Wish

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Wish handler | `subsystems/wish.py::makewish` | full (Wave 6 Phase B+ vendor `wishymatch` parser) | `vendor/nethack/src/do_wish.c::makewish`, `do_wish.c::wishymatch` |
| Wish modifier parser (blessed / cursed / +N / etc.) | `wish.py::wishymatch` | full | `vendor/nethack/src/do_wish.c::wishymatch` |
| Artifact alignment restriction (Excalibur lawful etc.) | `wish.py` | full | `vendor/nethack/src/artifact.c::SPFX_*` |
| WISHLESS / ARTIWISHLESS conducts | `wish.py` -> `conduct.py` | full | `vendor/nethack/src/insight.c::end_of_game` |

---

## Genocide

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Genocide scroll handler | `subsystems/items_scrolls.py::SCR_GENOCIDE` -> `apply_genocide` | full | `vendor/nethack/src/read.c::SCR_GENOCIDE` |
| GENOCIDELESS conduct | `subsystems/conduct.py` | full | `vendor/nethack/src/insight.c` |

---

## Polymorph

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| `polymorph_player` (NATTK=6 attack swap, intrinsic mask, AC recompute, armor-drop on no-hands) | `subsystems/polymorph.py::polymorph_player` | full | `vendor/nethack/src/polyself.c::polyself`, `polyself.c::skinback` |
| `polymorph_monster` | `polymorph.py::polymorph_monster` | full | `vendor/nethack/src/mon.c::newcham` |
| Lycanthropy timer + auto-revert | `polymorph.py::step` (decrements `poly_timer`, `lycanthropy_timer`) | full | `vendor/nethack/src/were.c` |
| `poly_trap_effect` | `polymorph.py::poly_trap_effect` | full | `vendor/nethack/src/trap.c::POLY_TRAP` |
| POLYSELFLESS conduct | `polymorph.py` -> `conduct.py` | full | `vendor/nethack/src/insight.c` |

---

## Special levels

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Oracle | `dungeon/special_levels.py::oracle_level` | full | `vendor/nethack/dat/oracle.lua` |
| Mine Town | `dungeon/special_levels.py::mine_town_level` | full (verbatim MAP string from `minetn-1.lua`) | `vendor/nethack/dat/minetn-1.lua` |
| Mines End | `dungeon/special_levels.py::mines_end_level` | full (verbatim MAP from `minend-1.lua`) | `vendor/nethack/dat/minend-1.lua` |
| Big Room | `dungeon/special_levels.py::big_room_level` | full | `vendor/nethack/dat/bigrm-*.lua` |
| Castle | `dungeon/special_levels.py::castle_level` | full (verbatim MAP from `castle.lua`) | `vendor/nethack/dat/castle.lua` |
| Vlad's Tower (3 floors) | `dungeon/special_levels.py::vlads_tower_level` | full (verbatim `tower1.lua` / `tower2.lua` / `tower3.lua`) | `vendor/nethack/dat/tower{1,2,3}.lua` |
| Wizard's Tower + 3 fakes | `dungeon/special_levels.py::wizards_tower_level` | full (verbatim `wizard1/2/3.lua` MAPs) | `vendor/nethack/dat/wizard{1,2,3}.lua` |
| Sanctum | `dungeon/special_levels.py::sanctum_level` | full (verbatim `sanctum.lua` MAP) | `vendor/nethack/dat/sanctum.lua` |
| Valley of the Dead | `dungeon/special_levels.py::valley_level` | full | `vendor/nethack/dat/valley.lua` |
| Demon lairs (6: Asmodeus, Baalzebub, Juiblex, Orcus, Yeenoghu, Demogorgon) | `dungeon/demon_lairs.py` | full (verbatim `*.lua` MAPs) | `vendor/nethack/dat/{asmodeus,baalz,juiblex,orcus}.lua` |
| 13 quest goal levels | `dungeon/quest_levels.py` | full (verbatim `*-goal.lua` MAPs) | `vendor/nethack/dat/{Arc,Bar,Cav,Hea,Kni,Mon,Pri,Ran,Rog,Sam,Tou,Val,Wiz}-goal.lua` |
| Endgame (Earth / Air / Fire / Water / Astral) | `dungeon/endgame.py` | full (verbatim MAPs from `{earth,air,fire,water,astral}.lua`) | `vendor/nethack/dat/{earth,air,fire,water,astral}.lua` |

> **Note on licensing**: the verbatim MAP strings are ported byte-equal from vendor `*.lua` files. See README "License + acknowledgments" — a full NGPL audit of this boundary is deferred.

---

## Material / spell flags

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| `oc_material` mapped through to conduct + corrode logic | `constants/objects.py` + `conduct.py` | full | `vendor/nethack/include/objclass.h::oc_material` |
| `MS_SPELL` mage-detection flag | `subsystems/monster_ai.py` (Wave 6 Phase B: reads `MonsterEntry.sound`) | full | `vendor/nethack/include/monst.h::MS_SPELL`, `mcastu.c` |
| Spell-school enum | `constants/spells.py` | full | `vendor/nethack/include/spell.h::SCHOOLS` |

---

## Identification

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Per-game description shuffle (potions / scrolls / wands / rings / amulets / gems) | `subsystems/identification.py` | full (Wave 6 #77 Fisher-Yates) | `vendor/nethack/src/o_init.c::init_objects` |
| Per-instance BUC discovery | `identification.py` | full | `vendor/nethack/src/pray.c::altar_wrath`, `do_wear.c` |
| Detection-spell timers | `identification.py` (Wave 6 #79: per-spell deadlines) | full | `vendor/nethack/src/detect.c` |

---

## Status timers (regen / decay)

Covered by **Status effects + timers** above; the unification is intentional (vendor's `timeout.c` covers both decay and regen).

---

## Env step

Covered by **Core game loop + dispatch** above (`_step_impl` mirrors `allmain.c::moveloop` step-by-step with citations inline in `env.py`).

---

## Dungeon generation

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Main branch L1 generation | `dungeon/branches.py::generate_main_branch_l1` | full (rooms + corridors + stair placement) | `vendor/nethack/src/mklev.c::makelevel` |
| Room placement (non-overlapping rect, fori_loop rejection sampling) | `dungeon/rooms.py::place_rooms` | full | `vendor/nethack/src/mkroom.c`, `mklev.c::do_room` |
| L-shaped corridors | `dungeon/corridors.py::connect_rooms` | full | `vendor/nethack/src/mklev.c::join` |
| Kruskal perfect maze | `dungeon/mazes.py::generate_maze` | full | `vendor/nethack/src/mkmaze.c::walkfrom` |
| Mines branch (CA caves) | `dungeon/branches.py::generate_mines_level` | full | `vendor/nethack/src/mkmaze.c::mkmines` |
| Sokoban (8 hand layouts) | `dungeon/branches.py::generate_sokoban_level` | full (from `soko*-*.lua`) | `vendor/nethack/dat/soko{1..4}-{1..2}.lua` |
| Gehennom (16-level procedural + 4 inserts) | `dungeon/branches.py::generate_gehennom_level` | full | `vendor/nethack/src/mkmaze.c::mkgehennom` |
| Cross-branch traversal + cache | `dungeon/level_memory.py::traverse_stair_cross_branch` | full | `vendor/nethack/src/dungeon.c::save_dungeon`, `do.c::next_level` |

---

## Doors

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Bump-to-open | `subsystems/action_dispatch.py::_try_step` -> door logic | full | `vendor/nethack/src/hack.c::domove` |
| Open / close action | `_handle_open`, `_handle_close` | full | `vendor/nethack/src/lock.c::doopen`, `doclose` |
| Kick (force open) | `_handle_kick` | full | `vendor/nethack/src/dokick.c::dokick` |
| Unlock | `lock.c` mirror in nethax | full | `vendor/nethack/src/lock.c::pick_lock` |

---

## Observations

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| 17-key NLE dict | `Nethax/nethax/obs/nle_obs.py::build_nle_observation` | full | `vendor/nle/nle/nethack/nethack.py::OBSERVATION_DESC`, `vendor/nle/include/nleobs.h` |
| `glyphs` (terrain + player overlay, fog-masked) | `obs/nle_obs.py` | full | `vendor/nle/include/nleobs.h::glyphs` |
| `blstats` (all 27 fields) | `obs/nle_obs.py::build_blstats` | full | `vendor/nle/include/nleobs.h::blstats`, `nle/nle/nethack/nethack.py::BLSTATS_FIELDS` |
| `tty_chars` / `tty_colors` / `tty_cursor` (24x80) | `obs/nle_obs.py` | full | `vendor/nle/include/nleobs.h::tty_*` |
| `inv_strs` (canonical NetHack inv-line text) | `obs/inv_strs.py` | full (Wave 6 Phase A: vowel-article, pluralization, name suffix) | `vendor/nethack/src/objnam.c::doname`, `objnam.c::an`, `objnam.c::makeplural` |
| `inv_glyphs`, `inv_letters`, `inv_oclasses` | `obs/nle_obs.py` | full | `vendor/nle/include/nleobs.h` |
| `colors`, `specials`, `internal`, `screen_descriptions` | `obs/nle_obs.py` (Wave 4 polish) | full | `vendor/nle/include/nleobs.h` |
| Symbolic / pixel / text variants | `obs/symbolic_obs.py`, `pixel_obs.py`, `text_obs.py` | full | n/a (nethax extension) |
| Pixel rendering (sprite atlas) | `obs/pixel_obs.py` + `Nethax/tiles/tiles.npy` | full | `vendor/nle/win/share/tiledata2.txt` |

---

## Scoring

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Score formula (`topten`) | `subsystems/scoring.py` | full (Wave 6 closing audit) | `vendor/nethack/src/end.c::topten`, `topten.c` |
| Per-conduct bonuses | `subsystems/conduct.py` + `scoring.py` | full | `vendor/nethack/src/insight.c::end_of_game` |
| End-of-game conduct scoreboard | `conduct.py` (Wave 6 Phase A) | full | `vendor/nethack/src/insight.c::end_of_game` |
| Ascension bonus (+50 000) | `subsystems/ascension.py` | full | `vendor/nethack/src/end.c::done_ascend` |

---

## AC

Covered by **Combat** -> AC computation above (`compute_ac` reads per-slot armor and produces vendor-equivalent AC value).

---

## BUC

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| BUC field per item slot | `Item` in `subsystems/inventory.py` | full | `vendor/nethack/include/obj.h::cursed`, `blessed` |
| Curse-status visibility (priest / altar / pray) | `identification.py`, `prayer.py::altar_buc_sense` | full | `vendor/nethack/src/pray.c::altar_wrath`, `priest.c` |
| Inventory rendering of BUC (`inv_strs`) | `obs/inv_strs.py` | full | `vendor/nethack/src/objnam.c::doname` |

---

## Action dispatch

Covered by **Core game loop + dispatch** above. 43-slot `lax.switch`-friendly tuple with per-slot vendor citations inline in `action_dispatch.py`.

---

## Branches

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| 7-branch graph (Main / Mines / Sokoban / Quest / Vlad / Gehennom / Endgame) | `dungeon/branches.py::init_branch_graph` | full | `vendor/nethack/src/dungeon.c`, `dungeon.lua` |
| Cross-branch ascent / descent | `dungeon/level_memory.py::traverse_stair_cross_branch` | full | `vendor/nethack/src/do.c::next_level`, `prev_level` |
| Per-level RNG seed cache (bit-equal restore on revisit) | `DungeonState.level_rng_seeds` | full | `vendor/nethack/src/dungeon.c::save_dungeon` |

---

## Messages

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| `MessageState` ring buffer | `subsystems/messages.py` | full | `vendor/nethack/src/pline.c::pline` |
| Death-message generation | `subsystems/messages.py` (Wave 6 Phase A) | full | `vendor/nethack/src/end.c::done`, `end.c::tombstone` |

---

## FOV

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| Bresenham raycast per ray | `Nethax/nethax/fov.py::compute_fov` | full | `vendor/nethack/src/vision.c` (Bresenham implementation) |
| `sight_radius` per-ray clip | `fov.py` | full | `vendor/nethack/src/vision.c` |

> nethax uses Bresenham raycast as a simplification of vendor's full shadow-cast; for standard play (radius 1-9) the visible set is equivalent.

---

## RNG

Covered by **Core game loop + dispatch** above. `Nethax/nethax/rng.py` ports `rn2`, `rnd`, `rn1`, `dice_roll`, `weighted_choice` from `vendor/nethack/src/rnd.c`. All wrap `jax.random` so determinism is byte-equal across CPU/GPU/TPU.

---

## Save / load

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| `save_state(state, path)` / `load_state(path)` | `Nethax/nethax/save_load.py` | high (pickle-based pytree roundtrip; not vendor binary format) | `vendor/nethack/src/save.c::dosave0`, `restore.c::dorecover` |

> **Divergent**: nethax uses JAX-pytree -> numpy -> pickle, not vendor's hand-rolled binary save format. Roundtrip preserves bit-equal state; vendor file format compatibility is out of scope.

---

## Item effects

Covered by **Objects** above. Each effect family (potion / scroll / wand / ring / amulet / spellbook) has its own `subsystems/items_*.py` module dispatched via `lax.switch` over operand tuples.

---

## Polymorph state swap

Covered by **Polymorph** above. Vendor's "swap form" logic (NATTK attack array, intrinsic mask, AC recompute, armor-drop on no-hands form) is fully wired in `polymorph.py::polymorph_player`.

---

## NLE shim

| Subsystem | nethax module | Parity | Vendor source |
|---|---|---|---|
| `NLECompat` drop-in class | `Nethax/nethax/compat/nle_shim.py` | full surface (see `docs/nle_compat.md` for per-feature matrix) | `vendor/nle/nle/env/base.py::NLE` |
| 121-action set | `nle_shim.py::ACTIONS` | byte-equal | `vendor/nle/nle/nethack/actions.py` |
| Observation space (17 keys, shapes, dtypes) | `nle_shim.py::observation_space` | byte-equal | `vendor/nle/nle/nethack/nethack.py::OBSERVATION_DESC` |
| `glyph2tile` (5976 entries) | `Nethax.tiles.tile_data.GLYPH2TILE` | byte-equal | `vendor/nle/win/share/tiledata2.txt` |
| `StepStatus` enum | `nle_shim.py::StepStatus` | byte-equal | `vendor/nle/nle/env/base.py::NLE.StepStatus` |
| Glyph predicate helpers (`glyph_is_monster` etc.) | `nle_shim.py` static + module aliases | full | `vendor/nethack/include/display.h` macros |

---

## Deliberate divergences (no fix planned)

1. **Shopkeepers** — simplified. nethax has a `ShopState` with pay-at-exit and angry-shopkeeper-pursuit tick but no full shopkeeper-as-monster dialogue tree. Vendor: `vendor/nethack/src/shk.c::shk_chat`, `shknam.c` — out of nethax scope. Items in shops are priced and tracked; theft triggers angry mode and the shopkeeper acts as a high-strength monster.
2. **Bones files** — simplified. nethax does not emit or read `vendor/nethack/src/bones.c`-format bones files. Player corpses on death are visible in the current episode only.
3. **Wizard-mode debug commands** — out of scope. `vendor/nethack/src/cmd.c::wiz_*` is intentionally not exposed.
4. **Mail daemon** — out of scope. `vendor/nethack/src/mail.c` (`MAIL` compile flag) is not ported.
5. **Music / sounds** — out of scope. `vendor/nethack/src/music.c` and `sounds.c` are TTY-side effects with no JAX equivalent.
6. **Lua integration** — out of scope at runtime. nethax ports vendor `*.lua` MAP strings statically into Python at build time; there is no in-game Lua VM. Vendor: `vendor/nethack/src/sp_lev.c::lspo_*`.
7. **Verbatim vendor MAP strings under NGPL** — see README. A full NGPL licensing audit of the verbatim-MAP-string boundary in `dungeon/special_levels.py`, `dungeon/demon_lairs.py`, `dungeon/quest_levels.py`, `dungeon/endgame.py` is deferred.
8. **Save format** — divergent. nethax uses pickle-of-pytree, not vendor's binary format. Game state is preserved bit-equal across save/load roundtrips, but the on-disk format is nethax-specific.
9. **RNG model in NLE shim** — vendor NLE uses two RNGs (`core` + `disp`) plus an Anti-TAS `reseed` flag. nethax has a single JAX `PRNGKey`. See `docs/nle_compat.md` for the round-trip semantics.

---

## Cross-references

- [`architecture.md`](architecture.md) — top-down system architecture.
- [`nle_migration.md`](nle_migration.md) — practical migration guide for NLE users.
- [`nle_compat.md`](nle_compat.md) — NLE drop-in compatibility status (per-feature matrix).
- [`benchmark.md`](benchmark.md) — throughput vs NLE.
- [`CHANGELOG.md`](../CHANGELOG.md) — wave-by-wave history.
