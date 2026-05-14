# Wave 4 → Wave 5 — Scope Preview

Wave 5's job: **make the dungeon feel alive end-to-end.** Wave 4 delivered the RL benchmark surface (MiniHack 159 envs) and the breadth-first mechanics scaffold (polymorph, prayer, branches, special-level subset, obs polish). Wave 5 turns the remaining cross-subsystem stubs into a coherent moving world: monsters that act, traps that polymorph piles, quests that match per-role artwork, and a Castle / Vlad's / Gehennom progression.

After Wave 5 the only large remaining work is save/load + ascension end-game (Wave 6).

## Wave 5 deliverable

After Wave 5:
- Monsters move + attack + retreat + cast spells during normal play.
- Bump-attack works from `_try_step` — the standard "walk into monster" path.
- The trap-effect ↔ subsystem-call bridge is finished (POLY_TRAP, MAGIC_PORTAL, VIBRATING_SQUARE).
- Quest is per-role with vendor-faithful artifacts / leaders / nemesi.
- Castle, Vlad's, Wizard's Tower, Sanctum special levels are in.
- Gehennom branch exists; vibrating square reveals its entrance.
- Bag-of-holding + containers ship.
- The remaining 5 conducts trigger.

## Wave 5 breadth pass

### Monster AI completion — biggest single Wave 5 item

- **`monster_ai.step` in `env.step`** — call `monsters_step_all` between dispatch and status_effects.step. (~10-line wiring, but must thread RNG and update messages.)
- **LoS + path-aware target selection** — replace greedy 8-dir with bounded A* (cap 32 steps to keep JIT-friendly).
- **`muse` (monster item use)** — `vendor/nethack/src/muse.c`: heal / escape (potion of phasing) / attack (wand zap). Per-monster item slot, item-use heuristic.
- **`mcastu` (monster spell casting)** — port from `muse.c::mcastu`. Per-monster spell slot, mana drain.
- **Retreat behavior** — monsters with HP < 1/7 of max flee.
- **Pet AI** — recruit on tame, leashing distance, feeding restores tameness.
- **Stationary AI** — shopkeeper / priest / vault guard / Quest leader / Quest nemesis special behavior.

### Combat polish (the bridge layer)

- **Bump-attack bridge** in `subsystems/movement.py::_try_step`: when target tile has a live monster, call `combat.bump_attack(state, rng, target_pos)` instead of refusing the move.
- **Per-slot armor AC bonus table** — `vendor/nethack/src/do_wear.c::Armor` per-slot bonuses (helmet small/medium/large; shield small/medium/large).
- **Polymorph combat** — `combat.bump_attack` consults `state.polymorph.attack_*` when `is_polymorphed`.
- **Two-weapon combat** — `subsystems/combat.py::two_weapon_attack`.
- **Ranged / thrown** — `subsystems/combat.py::throw_attack`, `fire_attack`.

### Quest fidelity

- **Per-role quest tables** — port from `vendor/nethack/src/role.c::Role`: artifact, leader, nemesis, prefix string.
- **Per-role Quest layouts** — port from `vendor/dat/qst*.lua`: Archeologist mines temple, Caveman cave, Healer cave hospital, Knight tournament field, Monk monastery, Priest cathedral, Ranger forest, Rogue thieves' den, Samurai dojo, Tourist desert, Valkyrie hall, Wizard library, Barbarian cave.
- **Nemesis fight mechanics** — special HP / regen rules; on death, drops the artifact.
- **Return-to-leader victory** — artifact + leader-room → role flag set.

### Special levels (full set)

- Castle, Valley of the Dead, demon lairs (Asmodeus, Baalzebub, Juiblex, Orcus, Yeenoghu, Demogorgon).
- Vlad's Tower (top + lower): Vlad + Candelabrum.
- Wizard's Tower + 3 fakes — distinguish-by-search.
- Sanctum: Amulet of Yendor + high priest fight.

### Vibrating square → Gehennom entrance

- `VIBRATING_SQUARE` trap → MAGIC_PORTAL → Gehennom Dlvl 1.
- Gehennom branch (16 levels) added to `dungeon/branches.py`.
- Per-Gehennom-level Lord monsters.

### Bag-of-holding + containers

- Nested inventory state: per-container sub-inventory pytree.
- Container actions: `open`, `close`, `put-in`, `take-out`.
- Bag-of-holding: weight halving with curse-status flag.

### 5 remaining conducts

- **POLYPILELESS** — needs poly-trap-affects-pile branch.
- **GENOCIDELESS** — needs `scroll of genocide` handler (`items_scrolls.handle_genocide`).
- **ELBERETHLESS** — needs `engrave` action handler.
- **WISHLESS** — needs wish handler (deferred to Wave 6 if scope tight).
- **ARTIWISHLESS** — gated on wish (Wave 6).

### Trap-effect → subsystem-call bridge

Wave 5 designs a coherent bridge from `traps._TRAP_EFFECTS` to subsystem calls:
- `POLY_TRAP` → `polymorph.poly_trap_effect`
- `RUST_TRAP` → `items_inventory.rust_held_armor`
- `STATUE_TRAP` → `monster_ai.statue_animate`
- `LEVEL_TELEP` → `level_memory.teleport_cross_level`
- `MAGIC_PORTAL` → `level_memory.traverse_portal`
- `VIBRATING_SQUARE` → reveal flag (already done) + reveal Gehennom portal

The challenge: `lax.switch` branches must all return the same pytree shape. The bridge needs a "wide carrier" pattern where each branch reads/writes the same EnvState but mutates only its concerns.

### Cross-branch terrain restore-on-revisit

- `level_memory.leave_level` should set `generated[src_branch, src_level-1]=True` so symmetric descent / ascent restores from cache.
- Re-tighten the integration test `test_cross_branch_return_main_preserves_state` to assert bit-equal terrain.

### Per-role starting kits

- Beyond Wave 3's `STARTING_INVENTORY`, add:
  - Starting-spell table per role (Healer knows `cure light wounds`, etc.).
  - Starting intrinsic table per role (Monk has martial arts, Valkyrie has cold resistance).
  - Starting pet per role (cat, dog, pony, …).

## Wave 5 risks

1. **Monster AI step in env.step is invasive** — the step pipeline goes from "dispatch → status" to "dispatch → status → monsters → status_again?". Order matters: vendor does player-action → monster-action → status; we must match. Adding monster step in the middle may surface RNG-determinism issues from the parallel `lax.scan` over 200 monster slots.

2. **Bump-attack bridge needs careful RNG plumbing** — `_try_step` is currently RNG-free (movement is deterministic given action). Adding bump-attack means the movement subsystem must accept an RNG and pass it down. Touches every test that calls `_try_step` directly.

3. **Quest per-role data is large** — 13 roles × (artifact + leader + nemesis + prefix + layout) ≈ 65 distinct vendor data points. Likely a 200-LoC data-table port.

4. **Demon lair layouts are complex** — Asmodeus, Baalzebub, Juiblex are large with unique geometry (acid pits in Juiblex, fire pillars in Baalzebub). Hand-encoding > parsing; budget 1 agent per lair.

5. **Trap-effect bridge invasive** — `lax.switch`-on-trap-id with each branch calling into different subsystems means widening the bridge carrier to include all touched state slices. Risk: hidden RNG-thread bugs across branches.

## Recommended Wave 5 launch shape

Sequential phases, parallelism within each:

### Phase 0 — bridge wiring (2 agents, blocker)
- Monster AI step in env.step (1 agent).
- Bump-attack bridge in _try_step (1 agent).

### Phase 1 — combat + monster AI depth (parallel agents)
- LoS + path-aware target selection.
- muse / mcastu / retreat / pet AI.
- Per-slot armor AC bonus table.
- Polymorph combat (`combat.bump_attack` consults `polymorph.attack_*`).
- Two-weapon + ranged combat.

### Phase 2 — special levels + branches (parallel agents)
- Castle, Vlad's, Wizard's Tower, Sanctum.
- Demon lairs (1 agent each: Asmodeus, Baalzebub, Juiblex, Orcus, Yeenoghu, Demogorgon).
- Gehennom branch + 16 levels.
- Vibrating square → portal wiring.

### Phase 3 — Quest fidelity + containers (parallel)
- Per-role Quest tables + layouts.
- Bag-of-holding + containers.
- Nemesis fight mechanics.

### Phase 4 — trap-effect bridge + remaining conducts
- Trap-effect → subsystem-call bridge.
- POLYPILELESS, GENOCIDELESS, ELBERETHLESS wiring.

### Phase 5 — integration tests + cross-branch fixes (1 agent)
- End-to-end "play to depth 5" test.
- NLE compatibility shim.
- Cross-branch round-trip bit-equality.

~15-18 agents total.

## Open questions for Wave 5

1. **Monster-AI scan width**: vendor allows up to ~200 monsters per level (`MAX_NUM_DIFF_MONSTS`). Our current scan is 200 wide and JIT-friendly. Do we expand to 400 to leave headroom for summoning, or hold at 200 with a per-level cap? **My recommendation: hold at 200, soft-cap summoning. ★**

2. **Bump-attack RNG injection**: `_try_step` is currently RNG-free. Option A: thread RNG into every movement call site (10 callers). Option B: have movement consume a sub-key from a top-level RNG carrier. **My recommendation: B — `state.combat.rng` carrier that combat reads from in addition to the per-step rng. ★**

3. **Quest layout fidelity**: vendor's `qst*.lua` are detailed (~120 lines each per role). Hand-translate or parse? Wave 4 settled on parse-for-MiniHack. **My recommendation: hand-translate Quest specifically because the layouts are role-iconic and parser fidelity matters less than visual accuracy. ★**

4. **Trap-effect bridge: lax.switch vs cond cascade?** `lax.switch` is faster but requires all branches return identical pytree shape. `cond` cascade is more flexible. **My recommendation: `lax.switch` over a "wide carrier" pytree, accept the discipline. ★**

5. **Endgame planning**: should Wave 5 include the 5 Astral planes + ascension condition, or push to Wave 6? Vendor treats ascension as separate code in `end.c`. **My recommendation: push to Wave 6 along with save/load — keeps Wave 5 focused on dungeon-traversal mechanics. ★**

Defaults all marked ★. Reply with picks or "all defaults".
