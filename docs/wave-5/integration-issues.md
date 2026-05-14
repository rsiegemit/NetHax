# Wave 5 — Integration Issues

A running log of the issues caught during the Wave-5 integration pass that the parallel agents either missed or did not fully resolve.

## 1. TileType enum collision (Wave 5 Phase 2)

**Symptom:** Three Phase-2 agents (demon-lairs, endgame, major-special-levels) all needed new tile types: `DRAWBRIDGE_UP`, `ICE_FLOOR`, `POOL`.  Each agent added an entry to `Nethax/nethax/constants/tiles.py::TileType` concurrently and gave them overlapping integer values (each picked `17` first).

**Resolution:** Phase 5 integrator sequenced the three new entries:
- `DRAWBRIDGE_UP = 17`
- `ICE_FLOOR     = 18`
- `POOL          = 19`

Vendor uses different numbers (`DRAWBRIDGE_UP = 19, ICE = 33, POOL = 16` in `vendor/nethack/include/rm.h`), but to stay contiguous in our local enum we use sequential numbering and document the divergence in `tiles.py` comments.

This also forced `NUM_TILE_TYPES` to refresh.  All downstream consumers (`obs/nle_obs.py::CMAP_LOOKUP`, `fov.py::OPAQUE_TILES`) accept the new entries without modification because they're length-driven.

## 2. Polymorph + step() signature regression (Wave 5 Phase 1)

**Symptom:** Wave 4 introduced a `polymorph.step(state, rng)` signature.  The Phase-1 combat-polish agent added a Wave-5 `polymorph_combat_step` that took `(state, rng, attacker_idx)`.  The bridge layer assumed the new signature for the call from `env.step`, breaking `test_no_op_step::test_polymorph_step_noop` which still called the old signature.

**Resolution:** Kept `polymorph.step(state, rng)` as the canonical env-step entry; renamed the combat agent's new helper to `polymorph.combat_attack(state, rng, attacker_idx)`.  Bridge unchanged.

## 3. Engrave + Container both extending EnvState (clean coexistence)

**Risk:** Two Phase-3 agents (containers, engrave) both added new fields to `EnvState`:
- `containers: ContainerState`
- `engrave: EngraveState`

Concurrent edits to `EnvState` field order could have produced a merge conflict that breaks pytree identity (changing field order changes the JAX pytree treedef, invalidating all compiled functions).

**Resolution:** The two agents landed in different commits.  The integrator confirmed the merged ordering:
```python
containers: ContainerState
engrave: EngraveState
```
matches the order in the `EnvState.default` classmethod and that `flax.struct.dataclass` preserves declaration order in pytree leaves.

No compiled-function invalidation.

## 4. Action-dispatch slot ordering

**Risk:** Phase-1/3/4 agents added new handler slots concurrently.  Each agent picked "the next free slot" — but without coordination, multiple agents could pick the same slot.

**Resolution:** The Phase 5 integrator confirmed the final sequence:
- Slot 36 = PRAY (Wave 4)
- Slot 37 = TWOWEAPON (Wave 5 Phase 1 combat polish)
- Slot 38 = THROW (Wave 5 Phase 1)
- Slot 39 = LOOT (Wave 5 Phase 3 containers)
- Slot 40 = APPLY (Wave 5 Phase 3 containers)
- Slot 41 = ENGRAVE (Wave 5 Phase 4 engrave)

All confirmed in `Nethax/nethax/subsystems/action_dispatch.py::_HANDLERS`.

## 5. Cross-branch round-trip terrain bit-equality

**Symptom:** Wave 4 integration test `test_cross_branch_return_main_preserves_state` was relaxed to a "cache-preservation contract" (Mines remains cached after round-trip), not a bit-equal terrain check.

**Root cause:** `level_memory.leave_level` only wrote `cached_map` and `cached_explored`; it did not set `generated[src_branch, src_level-1] = True`.  As a result, the symmetric ascent re-generated the source level instead of restoring it from cache.

**Resolution (Wave 5 Phase 5):** A 1-line addition to `leave_level`:

```python
new_generated = state.generated.at[b, lv].set(True)
```

After the fix, `test_cross_branch_main_to_mines_to_main_terrain_preserved` (in `test_wave5_integration.py`) now asserts bit-equal terrain.

## 6. Endgame.done freezing the pipeline

**Risk:** After ascension sets `state.done=True`, subsequent `env.step` calls should be idempotent.  Without a guard, monster ticks and status ticks would continue, polluting the post-ascension state.

**Resolution:** `env.step` wraps the full pipeline in `lax.cond(state.done, lambda _: state, _do_step, operand=None)`.  Post-ascension steps return the state unchanged.  Confirmed by an explicit integration test (`test_endgame_ascension_full_flow`).

## 7. Quest dispatch role coverage

**Symptom:** The first cut of `dispatch_quest_level` only handled roles 0..12 but the player-role enum allowed `-1` (uninitialised) which fell through `lax.switch` and crashed with an out-of-range branch index.

**Resolution:** Clamp role to `[0, 12]` before dispatch:
```python
safe_role = jnp.clip(role, 0, 12)
```

This is in `quest_levels.py::dispatch_quest_level`.

## 8. Bag-of-holding weight at integer precision

**Risk:** Vendor uses floating-point weight multipliers (0.25 / 0.5 / 2.0).  In JAX with float32, accumulated weights can drift.

**Resolution:** Use integer math: store numerator (1, 2, 8) and denominator (4) as ints, compute `(raw_total * numer) // denom`.  Bit-exact across JIT compile.

See `containers.py::container_total_weight` (lines 460-490).

## What integration testing caught

The Phase-5 integration pass surfaced 5 bugs that the per-subsystem
agents missed:

1. Polymorph step signature regression (#2) — caught by `test_no_op_step`.
2. TileType enum collision (#1) — caught by `test_obs_polish::test_tile_type_count_matches_cmap`.
3. Cross-branch round-trip bit-equality (#5) — caught by the new
   `test_cross_branch_main_to_mines_to_main_terrain_preserved`.
4. Action-slot ordering (#4) — caught by `test_action_enum`.
5. Quest role clamping (#7) — caught by an early run of
   `test_quest_dispatch_returns_role_specific_layout`.

All five are now fixed; the integration test for each is in place.
