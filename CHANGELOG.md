# Changelog

Wave-by-wave history of the nethax reimplementation. Per-wave development logs (decisions, gaps, mechanics-status, next-wave plans) live under `docs/wave-{1..5}/`. This file is the high-level summary.

Test counts are taken from each wave's `README.md` headline.

---

## Wave 6 — Maximum Vendor Parity + Polish

**Status: complete. 1691 tests collected (`pytest --co -q`), all passing on CPU.**

Wave 6 is the parity-audit wave. Wave 5 had delivered a complete, end-to-end-playable JAX NetHack with monster AI, full special-level inventory, polymorph/prayer/quest fidelity, containers, engrave, genocide, and endgame ascension. Wave 6 picks up the remaining vendor-parity drift and fills it in.

**Phase A — polish surface**
- `inv_strs`: vowel-article mask, pluralization tables, name suffix, curse-status visibility, stack-quantity rendering (`vendor/nethack/src/objnam.c::doname`, `an`, `makeplural`).
- Conduct end-of-game scoreboard + per-conduct scoring bonuses (`insight.c::end_of_game`).
- Death-message generation in `subsystems/messages.py` (`vendor/nethack/src/end.c::done`, `end.c::tombstone`).
- Name (call) action handler (slot 42) — `vendor/nethack/src/do_name.c::do_oname`.

**Phase B / B+ — vendor-parity ports**
- `constants/roles.py`, `constants/races.py` — full Wave-6 Phase B+ parity ports of `vendor/nethack/src/role.c::roles[]` / `races[]`.
- `constants/objects.py` — full vendor-parity table including `oc_subtyp` family (Wave 6 closing-audit fields).
- `subsystems/wish.py::makewish` — full `wishymatch` parser, modifier handling (blessed/cursed/+N), artifact alignment restrictions (Excalibur lawful etc.); WISHLESS + ARTIWISHLESS conducts.
- `subsystems/monster_ai.py` — mage-class detection via real `MS_SPELL` flag (read through `MonsterEntry.sound` / `msound`), replacing Wave-5's `[LO, HI]` heuristic.
- `subsystems/conduct.py` — material-driven food conducts pull `oc_material` straight from canonical `OBJECTS` table.
- `subsystems/character.py` — vendor-parity HP/PW derivation in `create_character`.

**Phase C — closing audits (numbered #47, #73, #76, #77, #78, #79)**
- `#47` — shop pay-at-exit + angry shopkeeper pursuit tick in `_step_impl` step 7.
- `#73` — vendor-exact hunger thresholds (`eat.c::newuhs` lines 3369-3372); spell-success formula corrected (`spell.c::percent_success` returns SUCCESS pct).
- `#76` — back-compat `pw_regen_tick(state)` shim.
- `#77` — vendor-parity Fisher-Yates shuffle for identification description indices (`o_init.c::init_objects`); spec-exact status-effect windows (stoning/sliming/strangulation/food-poisoning death cycles, haste-self timer).
- `#78` — Closing-Audit additions: vendor `u.*` parity fields (`player_luck`, `player_moreluck`, `player_in_water`, `player_buried`, `player_steed_mid`, `player_killer_mid`, `player_mortality` mirroring `you.h` lines 360-510); `PrayerState` extensions (`punished`, `saddled_cursed`, `stuck_in_wall`, `in_region`); duplicated HP/Pw-regen interval path removed (canonical `status_effects.step` is the only caller).
- `#79` — detection-spell timers (per-spell deadlines including SPELL_LIGHT lit-radius timestamp).

**Phase D — documentation package**
- Top-level `README.md` rewrite with quickstart + NLE drop-in + test counts.
- `docs/architecture.md` — system architecture, EnvState shape, JAX functional contract.
- `docs/vendor_parity.md` — per-subsystem vendor parity matrix.
- `docs/nle_migration.md` — NLE -> nethax migration guide for RL practitioners.
- `CHANGELOG.md` (this file).

**Verbatim vendor MAP-string parity-fix sweep** — `dungeon/special_levels.py`, `dungeon/demon_lairs.py`, `dungeon/quest_levels.py`, `dungeon/endgame.py` all updated to parse byte-identical vendor `*.lua` MAP blocks (Mine Town, Mines End, Castle, Vlad's Tower 1-3, Wizard's Tower + 3 fakes, Sanctum, Asmodeus, Baalzebub, Juiblex, Orcus, Valley, Earth, Air, Fire, Water, Astral, all 13 Quest goals).

**Closing test surface**: 1691 collected (`JAX_PLATFORMS=cpu .venv/bin/python -m pytest --co -q`). Includes:
- 36 NLE-compat tests (`test_nle_compat_full.py`), 35 obs tests, 12 integration tests, 4 shim tests — 87 NLE-compat tests total.
- 1 opt-in deep Hypothesis sweep (`test_hypothesis_full.py`, gated on `RUN_HYPOTHESIS_FULL=1`).

---

## Wave 5 — Make It Alive

**Status: complete. ~790 tests passing.**

The cross-subsystem stubs from Wave 4 become a coherent moving world. Monsters act every step, traps polymorph piles, quests match per-role artwork, the Castle / Vlad / Wizard / Sanctum / 6-demon-lair / Gehennom / 5-Astral-plane endgame progression is end-to-end playable, containers and engrave land. The 5 deferred skips from Wave 4 (bump-attack bridge, trap dispatch, monster_kills_player, player_kills_monster, hunger 700-turn) are all unskipped.

Highlights: combat polish (per-slot AC, two-weapon, thrown pipeline, polymorph integration); monster AI depth (LoS Bresenham, BFS pathfind depth-12, muse, mcastu, retreat, pet); 35+ unique level factories; ascension wiring; 4-slot nested containers with bag-of-holding weight tables; engrave (Elbereth in dust) + ELBERETHLESS conduct; genocide scroll + GENOCIDELESS conduct; NLE compat shim. Five new dispatch slots (37-41: two-weapon, throw, loot, apply, engrave). ~10 K LoC added.

Wave 5 -> Wave 6 backlog: WISHLESS/ARTIWISHLESS conducts, full wish handler, save/load polish, scoring/ascension polish, shop simplified, `inv_strs` polish, monster + object table canonicalize, role-specific bonuses, property-based combat tests, NLE compat full validation.

---

## Wave 4 — RL Surface + Branches + Polymorph + Prayer

**Status: complete. 611 tests passing, 5 skipped, 0 failed.**

The RL-runnable wave. After Wave 3 produced a full mechanics-complete NetHack, Wave 4 stacks the canonical MiniHack 170-env benchmark surface, lights up the cross-branch dungeon (Main <-> Mines / Sokoban / Quest), gives polymorph + prayer real game-effect tables, ships the Wave-4 subset of special levels (Oracle, Mine Town, Mines End, Big Room), wires fountain / throne / sink / altar interactions, propagates 8 of 13 conducts, and finishes the obs surface (all 17 NLE keys real-valued).

Highlights: 16 new dispatch slots (eat/quaff/read/zap/cast/pickup/drop/wield/wear/putOn/remove/open/close/kick/fight/search/pray); `LevelGenerator` + `RewardManager` + des-file parser for the 36 canonical MiniHack `*.des` files; 159-env MiniHack registry; full polymorph (NATTK=6 attack set swap, intrinsic mask, AC recompute); full prayer outcome chain (d100 buckets, trouble detection, sacrifice, BUC sense, anger paths); fountain (16 outcomes) / throne (14) / sink (13); cross-branch traversal with per-level state caching. +158 tests.

---

## Wave 3 — Mechanics Wired

**Status: complete. 444 tests passing, 14 skipped.**

The scaffold from Waves 1-2 becomes a game that actually plays. After `reset()` the player has a starting inventory, monsters spawn around them, doors open on bump, traps trigger on step, and combat rolls real dice. Pw regen ticks, hunger ticks, intrinsics from rings apply, and all 17 NLE obs keys carry real values.

Strategic decision: instead of "MiniHack-first" (faster RL value), do "full drop-in NLE replacement" — every mechanic NLE supports works in JAX, JIT-compatible.

Highlights: combat (THAC0/AC/skill formulas from `uhitm.c` + `weapon.c`); magic (all 43 spells); monster AI + depth-curve spawning; full slot mutation (pickup/drop/wield/wear/put_on_ring); `STARTING_INVENTORY` for all 13 roles; potion (26) + scroll (23) + wand (28) + ring (28) + amulet (13) effects; full hunger/regen/timer status pipeline; 26 trap types; door bump-open/kick/unlock; full-fidelity `inv_strs` ("a - a +0 long sword (weapon in hand)"); 28 new integration tests. ~17 K LoC added. +271 tests.

---

## Wave 2 — Mechanics + Migration + Data Tables

**Status: complete. 173/173 tests passing.**

The "make a player walk around a real dungeon" wave. Three threads:

1. Fill the no-op stubs from Wave 1: dungeon generation (rooms + L-corridors + Kruskal mazes), action dispatch (21 of 121 actions via `lax.switch`), FOV (real Bresenham raycast), observation projection (all 17 NLE keys with real or fully-typed values), pixel observation (sprite-atlas tile rendering at 336x1264x3 uint8).
2. Populate the canonical NetHack 3.7 data tables: 390 monster entries (across 6 chunk files), 503 object entries (across 9 chunk files), all glyph offsets re-verified against a running NLE install (Wave 1's 119 -> Wave 2's 121 actions; estimated glyph offsets -> verified live).
3. Delete legacy code paths: removed `nethax_state.py`, `game_logic.py`, `renderer.py`, old `constants.py`, `world_gen/`, `envs/`, `util/`. Moved `play_nethax.py` -> `scripts/legacy/` pending Wave 6 rewrite.

New `tests/test_vendor_parity.py` asserts our `Action` codes, `BL_*` indices, and `GLYPH_*_OFF` constants are byte-equal to vendor NLE. +70 tests.

---

## Wave 1 — Foundation + Breadth-First Scaffold

**Status: complete. 103 tests passing.**

Mission: erect the entire scaffold of NetHack 3.7 in JAX, in breadth. Every subsystem the real game depends on has a typed Flax state slice, a no-op step function with the right signature, vendor source citations, and a TODO list pointing the way to later waves.

No game mechanics were implemented in Wave 1 — combat doesn't roll dice, dungeons don't generate, monsters don't move. What was built was the **shape** of the game: every pytree slot, every enum, every API contract — so later waves can fill mechanics in place without re-architecting.

Footprint: 50 new Python modules, ~6 000 LoC. Areas: `Nethax/nethax/constants/` (8 files; NLE-parity enums + monster/object schemas), `subsystems/` (17 modules — combat, magic, monster_ai, polymorph, inventory, items, identification, traps, features, prayer, conduct, shop, quest, status_effects, scoring, messages, action_dispatch), `dungeon/` (rooms, mazes, corridors, branches, special_levels with 28 named levels, level_memory), `obs/` (17-key NLE observation builder + symbolic / pixel / text variants), top-level (`state.py` master EnvState, `env.py` NethaxEnv class, `fov.py`, `rng.py`, `save_load.py`).

---

## Cross-references

- Per-wave dev logs: `docs/wave-{1..5}/README.md` + `decisions.md`, `gaps.md`, `mechanics-status.md`, `next-wave.md`, `test-results.md`.
- Architecture: [`docs/architecture.md`](docs/architecture.md).
- Vendor parity matrix: [`docs/vendor_parity.md`](docs/vendor_parity.md).
- NLE migration: [`docs/nle_migration.md`](docs/nle_migration.md).
- Throughput: [`docs/benchmark.md`](docs/benchmark.md).
- NLE compat status: [`docs/nle_compat.md`](docs/nle_compat.md).
