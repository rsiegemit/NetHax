# Wave 4 — Gaps

What's still missing after Wave 4.  Most maps to Wave 5 or Wave 6.

## Wave 5 (next)

### Combat polish (the biggest Wave 5 thread)

- **Bump-attack bridge in `_try_step`** — when the player tries to move onto a monster-occupied tile, `_try_step` should call `combat.bump_attack(target_pos)` instead of refusing the move. Currently bump-attack is callable directly from tests but the dispatch doesn't route to it.
- **`monster_ai.monsters_step_all` in `env.step`** — Wave 3 implemented monster AI; Wave 4 did not call it from the step loop. Monsters don't move during normal play yet.
- **Per-slot armor AC bonus** — `inventory.compute_ac` sums armor across 7 worn slots, but the per-slot bonuses (helmet vs shield vs boots) currently use a flat `ac_bonus` field. Wave 5 should add the small/medium/large bonus table.
- **Polymorph combat** — `polymorph.attack_types` / `attack_damage_types` etc. are populated, but `combat.bump_attack` does not consult them. Player-as-troll should bite for `_form_attacks(troll)`, not swing a long-sword.

### Monster AI completion

- **LoS + pathfinding aware of walls** — current monster AI greedy-pathfinds toward the player ignoring walls.
- **`muse` (monster item use)** — heal/escape/attack item-use selection from `vendor/nethack/src/muse.c`.
- **`mcastu` (monster spell casting)** — currently no-op.
- **Retreat behavior** — low-HP monsters flee.
- **Pet AI + leashing + feeding** — pet recruitment, follow distance, tameness decay.
- **Stationary AI** — shopkeeper guarding shop, priest guarding temple, vault guard escorting.

### Quest fidelity

- **Per-role artifacts / leaders / nemesis** — `vendor/nethack/src/role.c::Role` struct carries the per-role quest table (artifact, leader, nemesis, prefix); currently `generate_quest_level` only knows about per-role guardians.
- **Quest level generation per role** — vendor has bespoke `qst*.lua` layouts per role (Archeologist, Caveman, Healer, ..., Wizard). Wave 4 ships a generic per-role layout.
- **Nemesis fight mechanics** — special HP / regen / spell-cast rules.
- **Artifact recovery + return-to-leader victory condition.**

### Special levels (full set)

- Castle, Valley of the Dead, Asmodeus, Baalzebub, Juiblex, Orcus, Yeenoghu, Demogorgon — demon lair layouts.
- Vlad's Tower (top + lower) — Vlad's lair + Candelabrum room.
- Wizard's Tower + 3 fakes — distinguish-by-search puzzle.
- Sanctum — Amulet of Yendor placement, high priest fight.

### Branch additions

- **Vlad's Tower** — 3 levels in Gehennom.
- **Gehennom** — 16 levels below Castle.
- **Endgame planes** — 5 Astral planes.

### Vibrating square → Gehennom entrance

- `VIBRATING_SQUARE` trap reveals → MAGIC_PORTAL → Gehennom gateway.
- Currently the VIBRATING_SQUARE trap only sets a `revealed` flag.

### Bag-of-holding + containers

- Nested inventory state (containers hold sub-inventory).
- Container open / close / put / take actions.
- Bag-of-holding weight halving.

### 5 remaining conducts (when underlying features land)

- **POLYPILELESS** — needs poly-trap-affects-pile branch (when a POLY_TRAP fires on a ground-item pile, items polymorph too).
- **GENOCIDELESS** — needs `scroll of genocide` handler.
- **ELBERETHLESS** — needs `engrave` action + handler.
- **WISHLESS** — needs wish handler (Wave 6).
- **ARTIWISHLESS** — needs artifact-wish path (Wave 6).

### Cross-branch terrain restore-on-revisit bit-equality

Currently `traverse_stair_cross_branch` regenerates the destination level if `generated[dst_branch, dst_level-1]` is False. The descending leg writes the source level into `cached_map` but does NOT set `generated[src_branch, src_level-1]=True`. Result: a round-trip Main 3 → Mines 1 → Main 3 regenerates Main level 3 on the way back. Fix: `leave_level` should set the generated flag for the source level.

### POLY_TRAP-affects-pile branch

When `POLY_TRAP` triggers, vendor `trap.c::dotrap` may also polymorph items on the same tile (`POLYPILELESS` conduct fires). Wave 4's `poly_trap_effect` only handles the player-on-trap case.

### Genocide scroll effect

`SCR_GENOCIDE` handler in `items_scrolls.py` — read scroll, choose monster class, remove all instances. `GENOCIDELESS` conduct hook.

### Engrave action + handler

`Action.ENGRAVE` (ord('E')). Engrave string on floor. Special case: "Elbereth" repels monsters. `ELBERETHLESS` conduct hook.

### Wish handler

`Action.WISH` — currently no-op. Wave 6 will implement the wish parser + artifact-grant + `WISHLESS` / `ARTIWISHLESS` conduct hooks.

### Trap-effect → action-dispatch bridge

The trap dispatch (`traps._TRAP_EFFECTS`) currently treats POLY_TRAP / RUST_TRAP / STATUE_TRAP as no-op. Wave 5 should design a coherent bridge that lets each trap effect call into its owning subsystem (polymorph for POLY_TRAP, items for RUST_TRAP, monster_ai for STATUE_TRAP).

### Per-role starting kits beyond Wave 3 baseline

Wave 3 ships `STARTING_INVENTORY` for all 13 roles from `u_init.c::trobj`. Wave 5 should add the per-role starting-spell list (Healer starts knowing `cure light wounds`, etc.) and starting intrinsic table.

## Wave 6

### Save / Load
- `save_state(state, path)`: pytree flatten + numpy save.
- `load_state(path)`: inverse, with cross-version compatibility hash.

### Scoring + ascension
- `compute_final_score` formula from `vendor/nethack/src/end.c`.
- Death message text generation.
- Ascension condition: Amulet of Yendor + Astral altar offering with correct alignment.

### Shop simplified buy/sell
- Pick up in shop → bill accrual.
- Pay bill at exit.
- Angry shopkeeper mode.

### `inv_strs` polish
- User-given names ("named Sting").
- Article "a" / "an" via vowel check.
- Plural irregulars ("knives", "men").
- Two-weapon "alternate weapon" status.

### Conduct scoreboard
- Display preserved conducts at end-game.

### Wish handler
- Wish parser + object name → instantiation.
- Artifact wish path.
- WISHLESS / ARTIWISHLESS conduct hooks.

### `scripts/legacy/play_nethax.py` rewrite
- Pygame interactive UI against the new `NethaxEnv` + `MinihaxEnv`.

### Full monstr difficulty table
- Replace `entry.level` proxy with the full vendor formula (level + speed_bonus + attk_count + breath_bonus + petrify_bonus).

### Object table canonicalize
- Drop dual-naming ("potion of healing" + "healing") → keep only canonical bare names.

## Out-of-scope (deliberately skipped)

- Wizard-mode debug commands
- Mail subsystem
- Music / sounds
- Real Lua integration (3.7 special levels use Python factories)
- Full shopkeeper haggling + dialogue
- Bones files

## TODOs visible in code

`grep -rEn "TODO|FIXME" Nethax/ --include='*.py' | wc -l` is now in the high-200s (was low-200s after Wave 3; Wave 4 added new TODOs for trap-effect bridge, monster AI step in env.step, conduct backlog).

## Test gaps

- **Property-based tests (Hypothesis)** — would catch off-by-one in combat / spell formulas. Wave 6.
- **End-to-end "play to depth 5" benchmark** — would catch perf regressions and missing bridge wiring. Wave 5.
- **NLE compatibility shim** — would let real NLE agents run on `NethaxEnv` / `MinihaxEnv` without code changes. Wave 5.
- **Throughput benchmark** — steps/sec on CPU vs GPU. Wave 6.
- **Cross-branch round-trip terrain equality** — currently relaxed in `test_wave4_integration.py`; tighten when `leave_level` sets `generated=True`.
