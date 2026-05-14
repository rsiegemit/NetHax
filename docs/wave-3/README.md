# Wave 3 — Mechanics Wired

**Status:** ✅ Complete · **444/444 pytest tests passing** · combat / magic / monster AI / items / status / traps / doors all live · 12 subsystem agents + 3 fix-up passes.

Wave 3 turned the breadth-first scaffold from Waves 1–2 into a game that actually plays. After `reset()`, the player has a starting inventory, monsters spawn around them, doors open on bump, traps trigger on step, and combat rolls real dice. Pw regen ticks, hunger ticks, intrinsics from rings apply, and the 17-key NLE observation dict carries real values for every key (including full-fidelity `inv_strs`).

The wave hit one strategic decision: instead of "MiniHack-first" (which would have been faster RL value), we did "full drop-in NLE replacement" — meaning every mechanic NLE supports works here, in JAX, JIT-compatible. MiniHack is now Wave 4.

---

## What shipped

| Area | Detail |
|---|---|
| **Combat** | THAC0/AC/skill formulas ported from `vendor/nethack/src/uhitm.c`+`weapon.c`. Real to-hit (d20 + STR/DEX/luck/enchant), damage (small/large dice + STR bonus), AC computation from worn armor, weapon skill practice/advance, bump-attack on movement. 527 + 259 lines. |
| **Magic** | All 43 spells: `cast_spell` with Pw cost + d100 success roll + dispatch. Per-spell effect handlers (healing, missile, fire bolt, detect, identify, levitation, ...). Pw regen tick. Spellbook read + study chance. 880 + 133 lines. |
| **Monster AI + spawning** | Depth-curve spawning via `monstr` difficulty table. `monster_turn` with greedy 8-dir pathfinding, sleep/wake, bump-attack on player. `monsters_step_all` runs 200 slots in a single `lax.scan`. 303 + 338 lines. |
| **Inventory + character init** | Full slot mutation (pickup/drop/wield/wear/take_off/put_on_ring). `STARTING_INVENTORY` for all 13 roles from `u_init.c`. `STARTING_STATS` per (Role, Race). `create_character` rolls stats, builds inv, computes AC. `env.reset(rng, role, race, alignment)` now creates a real Valkyrie/Wizard/etc. 695 + 617 lines. |
| **Potion + scroll effects** | All 26 potions + 23 scrolls dispatched via `lax.switch` over operand tuples. Healing/gain ability/levitation/teleport/identify/magic-mapping/remove curse/enchant/etc. 664 + 632 + 413 lines. |
| **Wand effects + ray** | All 28 wand effects: light/striking/cold/fire/lightning/sleep/death/poly/teleport/digging/wishing/etc. Bresenham ray dispatch via `lax.scan` over 8 steps. 560 + 260 lines. |
| **Ring + amulet** | 28 ring effects + 13 amulet effects. Wear/take-off applies/revokes intrinsics. Stat-adjust rings (gain STR, gain CON, adornment) modify player attrs. 430 + 315 lines. |
| **Status effects** | Full tick orchestrator: hunger_tick + hp_regen_tick + pw_regen_tick + tick_timers + starvation/strangulation/stoning/sliming/food-poisoning death cycles. Hunger threshold table, encumbrance formula, HP/Pw regen per role × XL. 549 + 286 lines. |
| **Traps + doors** | All 26 trap types' damage + side-effects. Door bump-to-open in `_try_step`. Kick / unlock. 326 + 408 + 563 (dispatch update) + 601 lines (tests). |
| **Observation polish** | Real `colors`, `tty_colors`, `specials`, `inv_glyphs`, `inv_letters`, `inv_oclasses` populated. All 17 NLE keys now project real values (no zeros remaining). 789 lines. |
| **Full-fidelity inv_strs** | NLE-canonical strings: `"a - a +0 long sword (weapon in hand)"`, `"b - a blessed +2 ring mail (being worn)"`, `"f - an unidentified violet potion"`, `"g - a +0 wand of striking (5:6)"`. Static byte tables built outside JIT, per-slot rendering via `lax.fori_loop`. 624 + 593 lines. |
| **Integration tests** | `test_full_step.py`, `test_combat_flow.py`, `test_item_use_flow.py`, `test_hunger_loop.py`, `test_movement_with_doors_traps.py`, `test_character_creation.py` — all multi-subsystem flows. 1,377 lines, 28 tests. |
| **Total Wave 3 footprint** | **~17,000 lines** of code + tests added across 12 subsystem files + 11 test files. 271 new tests (173 → 444). |

---

## How to use Wave 3

```python
import jax
from Nethax.nethax import NethaxEnv
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.races import Race

env = NethaxEnv()
state, obs = env.reset(jax.random.PRNGKey(0), role=Role.VALKYRIE, race=Race.HUMAN)

# Player has long sword wielded, ring mail worn, dagger in inv.
# 5 monsters are spawned around them.
# AC is real (player_ac < 10 because of armor).
print("starting HP:", state.player_hp, "AC:", state.player_ac)
print("inventory letters:", obs['inv_letters'][:5])
print("inv_strs[0]:", bytes(obs['inv_strs'][0]).rstrip(b'\x00').decode())
# → b'a - a +0 long sword (weapon in hand)'

# Movement bumps doors, triggers traps, attacks adjacent monsters.
state, obs, r, done, info = env.step(state, ord('l'), jax.random.PRNGKey(1))  # east

# JIT works for the whole step pipeline:
step_jit = jax.jit(env.step)
```

---

## Doc set

| # | File | Covers |
|---|---|---|
| 1 | [`README.md`](README.md) | This file |
| 2 | [`mechanics-status.md`](mechanics-status.md) | Per-subsystem status: what's real vs simplified vs still no-op |
| 3 | [`item-effects.md`](item-effects.md) | All ~150 item effects implemented (potions, scrolls, wands, rings, amulets, spells) |
| 4 | [`combat-formulas.md`](combat-formulas.md) | THAC0 / AC / damage / skill formulas with vendor citations |
| 5 | [`integration-issues.md`](integration-issues.md) | The 51 failures discovered post-agents and how they were fixed |
| 6 | [`decisions.md`](decisions.md) | Wave 3 design decisions + tradeoffs |
| 7 | [`gaps.md`](gaps.md) | Remaining TODOs (mostly Wave 4–6) |
| 8 | [`test-results.md`](test-results.md) | Full pytest output, 444 passing, 14 skipped (with reasons) |
| 9 | [`next-wave.md`](next-wave.md) | Wave 4 scope: MiniHack pull-forward + dungeon branches + polymorph |

---

## What's NOT in Wave 3 (deferred to later waves)

- **MiniHack 170-env curriculum** — Wave 4. The infrastructure (LevelGenerator + RewardManager port + des-file translation) is the big Wave 4 deliverable.
- **Dungeon branches** (Mines / Sokoban / Quest / Gehennom) — Wave 4. Single Main level only right now.
- **Polymorph** (player + monster) — Wave 4.
- **Prayer outcomes** — Wave 4 (alignment/luck/timeout state exists; pray returns IGNORED).
- **Special-level mechanics** (Oracle, Castle, Sanctum) — Wave 5.
- **Quest** — Wave 5.
- **Save/Load** — Wave 6.
- **Ascension** end-game — Wave 6.
- **Object-table canonicalization** (dropping "potion of X" duplicates) — Wave 4. OBJECTS count is 503, NLE canonical is 453; 50-entry overcount from dual naming.

See [`gaps.md`](gaps.md) for the full backlog.
