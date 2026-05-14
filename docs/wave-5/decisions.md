# Wave 5 — Design Decisions

The decisions that shaped Wave 5.  Each entry: **decision** → **rationale** → **alternative not taken**.

## 1. Hold monster scan width at 200

User pick (high-fidelity option): **400 monster slots** for headroom.

**Decision:** 400 monster slots per level (chose Option A — high fidelity).

**Rationale:** Wave 5 introduces summoning paths in `monster_ai.muse` and in several demon lairs (Yeenoghu summons gnoll squires; Demogorgon summons mariliths).  200 was already near saturation on hostile-heavy floors.  Doubling to 400 leaves headroom for stacked summons + Gehennom set-pieces.

**Alternative:** Hold at 200 with soft-cap.  Lower memory + JIT cost (16 KB of state at 200 vs 32 KB at 400), but boxes off summoning roads.  Rejected for breadth-of-mechanics over byte-counting.

**Citation:** `vendor/nethack/include/monst.h::MAX_NUM_DIFF_MONSTS` ≈ 380; vendor uses a per-level cap of "as many as fit", which we approximate at 400.

## 2. RNG threaded everywhere (Option A)

User pick: thread RNG into every movement call site (10 callers).

**Decision:** `_try_step` and all bump-attack call sites accept an explicit RNG.

**Rationale:** Combat is RNG-driven (to-hit, damage).  Putting RNG on a state carrier (Option B) would have made `combat.rng` a state field, which couples combat to every step.  Option A keeps RNG as a per-call argument — closer to the rest of the codebase, easier to test with a fixed seed.

**Alternative:** `state.combat.rng` carrier.  Cleaner top-level call signature but messier per-step bookkeeping.

**Touched call sites:** `action_dispatch._try_step`, `combat.melee_attack`, `combat.bump_attack`, `monster_ai.monster_attack_player`, plus all run-move handlers.  ~10 sites.

## 3. `lax.switch` wide carrier for trap bridge

User pick: `lax.switch` over a "wide carrier" pytree.

**Decision:** `traps.dispatch_trap_effect` uses `lax.switch` with a uniform `(state, rng)` operand and each branch returns the same `EnvState` pytree.

**Rationale:** All trap subsystem calls already accept `(state, rng) -> state`, so the natural carrier is just the full `EnvState`.  This makes the bridge a 10-line dispatcher.  Performance: `lax.switch` is constant-time vs `lax.cond` cascade depth × cond-cost.

**Alternative:** `lax.cond` cascade.  More flexible (branches can have different shapes locally) but cascades to 6+ levels deep for our 6-effect bridge.  Rejected.

**Discipline:** Every new trap effect must accept `(state, rng) -> state`.  Sub-effects (e.g. rust-trap touching armor) need to look up their own state slice from `state` rather than receiving it as a separate arg.

## 4. Endgame included in Wave 5

User pick: include the 5 Astral planes + ascension in Wave 5.

**Decision:** Wave 5 ships `Nethax/nethax/dungeon/endgame.py` and `subsystems/ascension.py`.

**Rationale:** Endgame is a small, well-contained piece (~600 LoC for 5 plane factories + 150 LoC for ascension).  Including it means Wave 5 has a coherent "play to win" story end-to-end.  Wave 6 can then focus on polish (scoring, save/load, death messages).

**Alternative:** Push to Wave 6 (vendor treats ascension as separate code in `end.c`).  Rejected because the Wave-5 scope was already going to wire the Vibrating Square → Gehennom portal, which structurally implies the player can reach the Sanctum, which structurally implies they can reach the Endgame.  Stopping short of ascension would leave that progression dangling.

## 5. Hand-translate Quest layouts

User pick: hand-translate Quest specifically.

**Decision:** 13 role-specific quest levels are hand-encoded in `dungeon/quest_levels.py` rather than parsed from `vendor/nethack/dat/qst*.lua`.

**Rationale:** Quest layouts are role-iconic (the Archeologist mines temple, the Wizard library, the Samurai dojo).  Visual recognisability beats parser fidelity.  Vendor `.lua` is dense (~120 lines each); hand-translating the dominant features is faster and easier to debug.

**Alternative:** Parse `qst*.lua` like we already do for the MiniHack 36 envs.  More work, less control over which features survive.

**Wave 6 follow-up:** A future full-parse pass can replace these layouts; per-role function signatures are stable.

## 6. Per-slot AC bonus cached on InventoryState

**Decision:** Each worn-armor item caches its per-slot AC bonus on `InventoryState.worn_armor` (one int8 per slot).

**Rationale:** AC is read every step (monster to-hit needs it).  Recomputing the AC sum every step would be O(7 slots × switch on armor type) per step, materializing 7 ints from item.type_id lookups.  Caching collapses this to a single sum.

**Alternative:** Recompute every step from `inventory.items[slot].type_id`.  Cleaner but ~5x slower.

**Citation:** `vendor/nethack/src/worn.c::find_ac` (vendor caches on `u.uac`).

## 7. BFS pathfinding bounded at depth 12

**Decision:** `monster_ai.pathfind_step` uses BFS bounded to 12 iterations.

**Rationale:** `lax.while_loop` with an unbounded predicate would break JIT trace stability.  Depth 12 covers most paths between adjacent rooms in our 21x80 map.  Beyond depth 12, monsters fall back to greedy Chebyshev pursuit.

**Alternative:** Unbounded Dijkstra.  Vendor uses this in `monmove.c::m_move`.  We can't (JIT shape constraint).  A* with a heuristic could potentially be tighter but adds priority-queue infrastructure that's awkward in JAX.

**Wave 6 follow-up:** Profile real maps and decide if depth 16 or depth 24 is worth the JIT-compile cost.

## 8. Mage detection by entry_idx range heuristic

**Decision:** `_is_mage_entry(entry_idx)` returns True if `120 ≤ entry_idx < 165` (the spellcaster range in our monster table).

**Rationale:** Avoids the need to expand `MonsterAIState` with a per-monster `is_mage` flag.  The flag is implicit in the entry index.

**Alternative:** Add a per-monster `MS_SPELL` bit to `MonsterAIState` and populate it from the monster table at spawn time.  Cleaner but adds 200 ints to the state pytree.

**Wave 6 follow-up:** Move to the real `MS_SPELL` flag from `vendor/nethack/include/monst.h` — there are a few non-mages in the [120, 165) range (e.g. mind flayers) and a few mages outside it.

## 9. Endgame frozen via lax.cond on state.done

**Decision:** `env.step` wraps the whole pipeline in `lax.cond(state.done, identity, _do_step, ...)`.

**Rationale:** Post-ascension `env.step` calls should be no-ops (vendor's `moveloop` bails when `program_state.something_worth_saving` is cleared).  This keeps the pytree shape stable across pre- and post-done states — important for callers that batch or vmap across steps.

**Alternative:** Raise an exception on `step` after done.  Cleaner Python idiom but breaks JIT.

## 10. Parser kept for MiniHack des-files

**Decision:** MiniHack des-files are still parsed (`Nethax/minihax/des_parser.py`), but Quest layouts are hand-translated.

**Rationale:** Vendor itself uses lex/yacc to parse `.des` (the canonical pipeline). Mimicking the vendor approach for MiniHack means automatic compatibility with new vendor MiniHack envs.  Quest layouts (`.lua`) are different — they're hand-curated set pieces, not procedural.

This means our parsing surface matches vendor for the procedural side (MiniHack) and our hand-translation matches the curated side (Quest).  Consistent with vendor's own division of labor.

## Defaults summary

All 5 user-facing decisions (#1, #2, #3, #4, #5) were the high-fidelity / wider-scope option (★ defaults from `next-wave.md` were declined in favour of bigger Wave 5).

This makes Wave 5 substantively larger than originally planned, but Wave 6 substantially smaller.
