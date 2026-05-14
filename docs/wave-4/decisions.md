# Wave 4 — Design Decisions

The five Wave-3 → Wave-4 open questions were resolved by the user with all defaults (★). Plus a handful of decisions that surfaced during Phase 0 / 1 implementation.

## 1. ★ Des-file parser over hand-translation

**Resolved:** parser.  `Nethax/minihax/des_parser.py` (2267 LoC) reads each of the 36 canonical `*.des` files under `vendor/minihack/minihack/dat/` and compiles to `LevelGenerator`-shaped builders.

**Tradeoff:**
- ✅ Maintainable: when MiniHack upstream adds a new `.des` file, parsing absorbs it for free.
- ✅ Faithful: the parser preserves random-selection semantics (`RANDOM` blocks) that hand-translation would have flattened.
- ❌ +2267 LoC for the parser itself.
- ❌ The parser is now its own surface to test (36 round-trip tests against the vendor files).

The alternative (hand-translate each `.des` into a Python factory) was rejected as a higher long-term maintenance cost.

## 2. ★ New `MinihaxEnv` class alongside legacy minihax

**Resolved:** new class.  `Nethax/minihax/minihax_env.py::MinihaxEnv` is a Wave-4 class built on the Wave-4 `NethaxEnv`.  The legacy `Nethax/minihax/` package (Wave 1-2 stubs) lives alongside untouched; Wave 5 / 6 will retire it once we have RL benchmarks running against `MinihaxEnv` long enough to confirm no regressions.

**Tradeoff:**
- ✅ Zero risk of breaking Wave-1 minihax users.
- ✅ Clear separation between vendor-style minihax and the Wave-4-rebuilt one.
- ❌ Two `MinihaxEnv`-shaped classes coexist briefly.

## 3. ★ Match canonical sparse reward; expose `RewardManager` for customization

**Resolved:** sparse-by-default.  Every env's default `EnvSpec.reward_manager` carries a single `location_event` on `stairs_down` (terminal +1).  Users override via `MinihaxEnv(env_id, reward_manager=custom_rm)`.

**Tradeoff:**
- ✅ Matches vendor MiniHack's default reward exactly.
- ✅ Custom shaping is a single constructor kwarg away.
- ❌ Sparse rewards are notoriously hard for from-scratch RL — users will likely customize. (Vendor MiniHack documents this too; we mirror their stance.)

## 4. ★ Full-fidelity polymorph

**Resolved:** full fidelity.  `polymorph_player` does the full Wave-4 attack-set swap (NATTK=6 entries), intrinsic mask from `MONSTERS[form].flags1/2`, AC recompute, armor-drop on no-hands, lycanthropy timer.

**Tradeoff:**
- ✅ Consistent with the Wave-3 `inv_strs` decision: pay the fidelity cost once, ship a real implementation.
- ✅ Lycanthropy timer + polymorph_monster come along for free.
- ❌ +600 LoC vs the simpler "swap role + stats only" path.

## 5. ★ Exact d100 prayer chains

**Resolved:** exact.  `pray()` runs the d100 outcome chain matching `pray.c::doprayer`: pleased buckets (0-29 HEAL_CURE, 30-49 PROTECTION, 50-64 REMOVE_CURSE, 65-74 GIFT_ARTIFACT, 75-99 no-op), angry buckets via `god_zaps_you`.

**Tradeoff:**
- ✅ Same approach as combat / spell-cast (already done in Wave 3); consistent.
- ✅ Sacrifice-on-altar and altar-buc-sense come along.
- ❌ ~400 LoC for the outcome table.

## 6. Phase 0 addition — status_effects.step now ticks every env.step

**Decision (Phase 0):** The Wave 4 dispatch-wiring agent observed that `status_effects.step` was exposed in Wave 3 but never called from `NethaxEnv.step`.  The fix added two splits and a `status_effects.step(...)` call between dispatch and obs build:

```python
new_state = dispatch_action(state, action, rng_act)
new_status, new_hp, new_pw, new_done = _status_step(
    new_state.status, rng_status,
    new_state.player_hp, new_state.player_hp_max,
    new_state.player_pw, new_state.player_pw_max,
    new_state.player_xl, new_state.player_role, new_state.done)
new_state = new_state.replace(status=new_status, player_hp=new_hp,
                              player_pw=new_pw, done=new_done,
                              timestep=new_state.timestep + 1)
```

This makes Wave 3's hunger / regen / lethal-expiry / starvation cascade actually run during normal play. Wave 3 had the formulas implemented but tested in isolation; Wave 4 wires them into the live env-step pipeline.

## 7. Conduct 5 / 13 deferred — no underlying feature to violate

**Decision:** Wire 8 conducts now (FOODLESS, VEGAN, VEGETARIAN, ATHEIST, WEAPONLESS, PACIFIST, ILLITERATE ×2, POLYSELFLESS).  Defer 5 conducts (POLYPILELESS, WISHLESS, ARTIWISHLESS, GENOCIDELESS, ELBERETHLESS) until the features they gate ship.  E.g., `ELBERETHLESS` should fire from `engrave` handler — but engrave is Wave 5.

**Tradeoff:**
- ✅ No "wire a conduct that can never trigger" landmines.
- ❌ Tracking 5 deferred conducts adds to the Wave 5 backlog.

## 8. Branches uses procedural generation rather than des-file

**Decision:** `generate_mines_level` and `generate_quest_level` use procedural builders (cellular automata + room placement) rather than parsing `vendor/dat/mines*.lua` / `quest*.lua`.

**Why:** Vendor's Mines lua files use Lua control flow (`if level == 3 then ... end`, random-from-list selection) that the .des-parser does not currently target (.des is a subset of vendor's level-description format; Mines uses the full Lua DSL). Re-using `generate_main_branch_l1` patterns + small hand-tweaks is simpler than expanding the parser.

**Tradeoff:**
- ✅ Deterministic + JAX-friendly: every cell is `jnp.int8`-typed at build time.
- ✅ Easier to test (a "Mines layout has CA cave structure" assertion is straightforward).
- ❌ Lower vendor fidelity — Mine Town's actual `minetn-1.lua` shop layout is hand-encoded separately under `special_levels._MINETOWN_ROWS`.

## 9. Schema: `PolymorphState` + `PrayerState` extended; no top-level `EnvState` mutation

**Decision:** All Wave-4 schema growth went into the existing pytree slices (`PolymorphState`, `PrayerState`, `ConductState`), not the top-level `EnvState`.

`PolymorphState` gained:
- `orig_role_idx`, `orig_str`, `orig_dex`, `orig_con`, `orig_hp_max`, `orig_ac` — pre-polymorph snapshot
- `orig_attack_types`, `orig_attack_damage_types`, `orig_attack_n_dice`, `orig_attack_n_sides` — pre-polymorph attacks
- `lycanthropy_timer`, `were_form_idx`

`PrayerState` gained:
- `alignment_record` (replaces the Wave-3 `alignment` int)
- `pray_timeout` (already existed, semantics clarified)

**Tradeoff:**
- ✅ Existing `EnvState.default(...)` callers see no shape change.
- ✅ Wave 3 tests don't drift.
- ❌ Each slice grows; `PolymorphState` is now 25+ fields.

## 10. Cross-branch traversal uses `level_memory.enter_level` / `.leave_level` hooks

**Decision:** `traverse_stair_cross_branch` calls `leave_level(src_branch, src_level, current_terrain, current_explored)` to snapshot the current level into `cached_map`, then either `enter_level` (first visit) or restores from `cached_map` (revisit).

This re-uses the Wave-2 level-memory infrastructure rather than adding cross-branch-specific caches. Wave 5 will tighten the "set generated=True on leave" contract to make round-trip restore-on-revisit bit-equal (see [`integration-issues.md`](integration-issues.md) item 3).

## 11. POLY_TRAP not yet wired into traps.dispatch

**Decision:** Leave `poly_trap_effect` as a callable helper without wiring it into the trap-dispatch `lax.switch`. The trap-effect bridge is Wave 5.

Rationale: traps.dispatch currently treats POLY_TRAP as "no damage, no side-effects" (the trap is revealed but has no on-step effect). Adding the polymorph_player call inside the trap-dispatch switch needs careful RNG plumbing (`lax.switch` branches must all return the same pytree shape, and polymorph mutates many slices). Wave 5 will design the trap-effect → subsystem-call bridge as a coherent piece.
