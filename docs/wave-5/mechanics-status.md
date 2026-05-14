# Wave 5 — Mechanics Status (per-subsystem delta)

Tracks each subsystem from "Wave 4 baseline" → "Wave 5 final".
A "real" row means the subsystem produces vendor-faithful state changes.

| Subsystem | Wave 4 | Wave 5 | Citation |
|---|---|---|---|
| **Combat (melee)** | basic to-hit + damage roll | + per-slot AC bonus, polymorph attacks, two-weapon toggle, bump-attack bridge | `vendor/nethack/src/uhitm.c::known_hitum`, `do_wear.c::Armor` |
| **Combat (ranged)** | stub | `thrown_attack` full pipeline (quiver → arc → land/break/lodge) | `vendor/nethack/src/dothrow.c` |
| **Monster AI (move)** | greedy 8-dir | + LoS Bresenham, BFS pathfind depth-12, retreat-when-HP<1/7, pet follow | `vendor/nethack/src/monmove.c`, `monster.c` |
| **Monster AI (use)** | none | `monster_use_item` (quaff potion / read scroll / zap wand) heuristic | `vendor/nethack/src/muse.c` |
| **Monster AI (cast)** | none | `monster_cast_spell` (mcastu) damage formula + mana drain | `vendor/nethack/src/mcastu.c::buzzmu` |
| **Polymorph (player)** | full | (unchanged) + combat integration: `bump_attack` reads `attack_*` when polymorphed | `vendor/nethack/src/polyself.c` |
| **Polymorph (monster)** | full | (unchanged) | `vendor/nethack/src/mon.c::newcham` |
| **Prayer / altar** | full | (unchanged from Wave 4) | `vendor/nethack/src/pray.c` |
| **Conduct (wired)** | 8 of 13 | **11 of 13** — adds ELBERETHLESS, GENOCIDELESS, POLYPILELESS | `vendor/nethack/src/insight.c` |
| **Conduct (deferred)** | 5 | 2 (WISHLESS, ARTIWISHLESS — gated on Wave-6 wish handler) | — |
| **Dungeon (main)** | full | (unchanged) | `vendor/nethack/src/mklev.c` |
| **Dungeon (Mines)** | full | (unchanged) | `vendor/nethack/src/mkmaze.c` |
| **Dungeon (Sokoban)** | 8 hand layouts | (unchanged) | `vendor/nethack/dat/soko*.des` |
| **Dungeon (Quest)** | single-role stub | **13 per-role layouts** + leader/nemesis table | `vendor/nethack/dat/qst*.lua`, `vendor/nethack/src/role.c::roles` |
| **Dungeon (Vlad's)** | none | 3 levels: lower / middle / top + Candelabrum | `vendor/nethack/dat/tower*.lua` |
| **Dungeon (Castle)** | none | full layout + drawbridge + wand of wishing | `vendor/nethack/dat/castle.lua` |
| **Dungeon (Wizard's)** | none | tower + 3 fakes (distinguish-by-search) | `vendor/nethack/src/wizard.c` |
| **Dungeon (Sanctum)** | none | Amulet of Yendor + high priest + 4 minions | `vendor/nethack/dat/sanctum.lua` |
| **Dungeon (Gehennom)** | none | 16-level branch (12 procedural + 4 inserts) | `vendor/nethack/src/mkmaze.c::mkgehennom` |
| **Dungeon (demon lairs)** | none | 6 unique layouts (Asmodeus, Baalzebub, Juiblex, Orcus, Yeenoghu, Demogorgon) | `vendor/nethack/dat/asmodeus.lua` + 5 |
| **Dungeon (Valley)** | none | Valley of the Dead L1 with vibrating square portal | `vendor/nethack/dat/valley.lua`, `trap.c::TRAP_VIBRATING_SQUARE` |
| **Dungeon (Endgame)** | none | 5 Astral planes (Earth / Air / Fire / Water / Astral) | `vendor/nethack/dat/{earth,air,fire,water,astral}.lua` |
| **Ascension** | none | `check_ascension`: Astral + matching altar + Amulet → done | `vendor/nethack/src/end.c::done_ascend` |
| **Special levels (Wave 4 set)** | Oracle, MineTown, MinesEnd, Big Room | (unchanged) | — |
| **Special-level total** | **4** | **35+** unique level factories | — |
| **Features (fountain/throne/sink/altar)** | full | (unchanged) | `vendor/nethack/src/fountain.c` |
| **Bump-attack bridge** | direct call only | wired into `_try_step` so movement into monster routes through `combat.bump_attack` | `vendor/nethack/src/hack.c::domove` |
| **Monster step in env** | not called | `monster_ai.step` called every `env.step` between dispatch and status | `vendor/nethack/src/allmain.c::moveloop` |
| **Traps (bridge to subsystems)** | enum only | `lax.switch` wide-carrier wires `POLY_TRAP / RUST_TRAP / STATUE_TRAP / LEVEL_TELEP / MAGIC_PORTAL / VIBRATING_SQUARE` | `vendor/nethack/src/trap.c::dotrap` |
| **Containers** | none | 4-slot nested inventory + bag-of-holding multiplier (1/4, 2/4, 8/4) + LOOT + APPLY | `vendor/nethack/src/pickup.c::use_container` |
| **Engrave** | none | `EngraveState` per-tile + `handle_engrave` (Elbereth in dust) | `vendor/nethack/src/engrave.c` |
| **Genocide** | none | `apply_genocide` scroll handler + GENOCIDELESS conduct | `vendor/nethack/src/read.c::SCR_GENOCIDE` |
| **Cross-branch round trip** | cache-preservation only | bit-equal terrain restore on revisit (`leave_level` fix) | `vendor/nethack/src/dungeon.c::save_dungeon` |
| **Obs (17 NLE keys)** | all 17 populated | (unchanged) | — |
| **NLE drop-in compat** | none | `Nethax.nethax.compat.nle_shim.NLECompat` wrapper | `vendor/nle/nle/nethack/nethack.py` |
| **Save / load** | stub | (unchanged — Wave 6) | — |
| **Scoring** | basic | + ASCENDED achievement + 50k bonus, full topten formula Wave 6 | `vendor/nethack/src/end.c::topten` |

---

## Headline counts

|  | Wave 4 | Wave 5 |
|---|---|---|
| Total subsystems | 19 | 19 (+ `containers`, `engrave`, `compat`) |
| Wired conducts | 8 / 13 | **11 / 13** |
| Special-level factories | 4 | **35+** |
| Dungeon branches in graph | 4 (Main, Mines, Sokoban, Quest) | **7** (+ Vlad, Gehennom, Endgame) |
| Action handler slots | 37 | **42** |
| MiniHack envs registered | 159 | 159 |
| Tests passing | 611 | ~790+ |
| Skipped | 5 | 0 |

---

## Move-from-stub list

The following Wave 4 stubs were upgraded to real in Wave 5:

- Monster AI step → wired in env.step (was unreachable).
- Bump-attack → wired in `_try_step` (was direct-call-only).
- Per-slot armor AC → was flat-bonus; now per-slot helmet small/medium/large + shield small/medium/large.
- Polymorph + combat → was separate; now bump-attack reads polymorph attack set.
- Trap dispatch → was enum-only; now bridges to 6 subsystem calls.
- Cross-branch terrain restore → was relaxed contract; now bit-equal round-trip.
- Containers → none → full 4-slot nested with BoH multiplier.
- Engrave → none → full per-tile state + Elbereth + ELBERETHLESS conduct.
- Endgame → none → 5 planes + ascension condition + done flag.
- Quest → single guardian → 13 role layouts + per-role table.
- Vlad's / Castle / Wizard / Sanctum / 6 demon lairs / Valley / Gehennom → all none → all real.

---

## Still simplified (Wave 6 candidates)

- **BFS pathfind**: bounded at depth 12 (vendor uses Dijkstra with no bound).  Sufficient for most map sizes but can fail on convoluted Sokoban shapes.
- **Mage detection** in `monster_use_item`: uses entry-index range heuristic instead of the real `MS_SPELL` flag.  Wave 6 should canonicalize against the monster table.
- **Pet AI**: tracks player but does not yet pick up items, attack hostile-near-player, or refuse to step into traps.
- **Scoring**: flat 50000-point ascension bonus; full topten formula (`u.urealtime`, `u.uhpmax`, hard-fought-multiplier) is Wave 6.
- **Death messages**: `done(state, KILLED_BY, "killer_name")` not wired.
- **Wish handler**: not yet built; WISHLESS / ARTIWISHLESS conducts deferred.
- **Quest layouts**: simplified-iconic versions of the vendor `.lua` (we hand-translated the dominant features but did not parse the full file).
