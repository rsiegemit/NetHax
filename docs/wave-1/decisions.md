# Wave 1 — Design decisions

Each decision below was made during Wave 1 with a tradeoff. Future-you can revisit any of them; do so with eyes open.

---

## 1. Target NetHack 5.0 / 3.7-branch

User picked "newest version". `vendor/nethack/include/patchlevel.h` reports `VERSION_MAJOR=5 VERSION_MINOR=0` on the current HEAD of the `NetHack-3.7` branch (commit `82d7b2b`).

**Tradeoff.** 3.7 introduces Lua-based level scripting (`dat/*.lua`) and minor mechanic changes vs. 3.6.x (which NLE / MiniHack are still pinned to). We sidestep Lua entirely (out of scope, see §6) and re-implement the level templates as Python factories.

**Risk.** Any agent trained on a 3.6.x-based NLE binary will encounter subtle mechanic differences (e.g., role start inventories, monster stats tweaks). Wave 2 should verify whether downstream RL workflows care.

---

## 2. NLE is the authoritative API contract (not 3.7 vanilla)

The 17-key observation dict and 121-action enum we target come from `vendor/nle/`, not from running NetHack's TTY. This means our env can drop in for any NLE-trained agent.

**Tradeoff.** NLE has 16/17-key parity surface evolution; if NLE adds an `inv_bonuses` key in a future release, we'll need to add it too.

---

## 3. Vendor source counts beat the audit

Our research-phase audit said `len(ACTIONS) == 119` and `len(USEFUL_ACTIONS) == 95`. Direct exec of `vendor/nle/nle/nethack/actions.py` revealed the truth: **121** and **101**. The agent generated the canonical count; our tests had to be updated.

**Decision.** Always treat the cloned vendor sources as ground truth, even when previous research disagrees. Wave 1 fixed both assertions and tests; future waves should establish a "vendor parity test" pattern early.

---

## 4. Breadth-first to stub before any subsystem reaches Done

User instruction. We did not implement combat formulas, magic, dungeon generation, or anything else in Wave 1. **Every** `step()` is a no-op that returns its input.

**Tradeoff.**
- ✅ Wave 1's product is testable in isolation: imports work, shapes are right, pytrees compose, JIT compiles. Each later wave can fill mechanics independently.
- ❌ Nothing is *playable* yet. A human running `NethaxEnv` watches a stationary `@` glyph forever.

This is the right tradeoff for an architecture wave — we'd rather rebuild zero combat formulas than two if the state shape turns out wrong.

---

## 5. JAX-only step. No Python control flow inside.

Every subsystem step function must be JIT-compatible: only `jnp.where`, `jax.lax.switch`, `jax.lax.scan`, no Python `if`/`for` over runtime data.

**Tradeoff.**
- ✅ Whole game can run on GPU/TPU at massive batch scale. RL training is the use case; this is the whole point.
- ❌ Subsystems that look easy in C (NPC dialogue, shopkeeper haggling, message format strings) become awkward. We accept simplifications for those (see §6).

---

## 6. Deliberate scope reductions

Confirmed by user before launching `/ultrawork`:

- **Wizard-mode debug commands**: dropped entirely.
- **Mail**: dropped.
- **Music/sounds**: dropped (irrelevant for RL).
- **Real Lua integration**: dropped. The 3.7 special-level Lua scripts in `vendor/nethack/dat/*.lua` will be re-implemented as Python state-producing factory functions.
- **Shopkeepers**: simplified — basic buy/sell only, no haggling, no theft detection, no chat. The C version is 6,125 lines of dialogue-heavy state machine that doesn't survive JAX-only constraints.
- **Bones files**: flag-gated, default off; may be wired in Wave 6 if time permits.

These are documented at the top of the relevant subsystem stub files.

---

## 7. Additive — don't touch existing code

`Nethax/nethax/` already had `nethax_state.py`, `game_logic.py`, `play_nethax.py`, `renderer.py`, `constants.py`, `envs/`, `util/`, `world_gen/`. Wave 1 left every existing file untouched.

**Tradeoff.**
- ✅ Existing scripts/tests still work, no regressions.
- ✅ Code reviewers can see what's new at a glance.
- ❌ Some duplication (e.g., `Item` exists in old `nethax_state.py`; new `inventory.py` imports it from there to avoid redefining).

Wave 2 will start migrating callers from old EnvState to new EnvState. Eventually the old `nethax_state.py` / `game_logic.py` will be deleted, but not until every caller is moved.

---

## 8. State slices owned by each subsystem, composed in `state.py`

A subsystem's Flax struct lives in its own file. `state.py` only aggregates. This is the inverse of monolithic state design (cf. NetHack C, where all globals live in `decl.c`).

**Tradeoff.**
- ✅ Each subsystem agent in `/ultrawork` could own one file with zero merge risk.
- ✅ Adding a new subsystem is mechanical: define a `<Name>State`, add one field to `EnvState`.
- ❌ Cross-subsystem state-reads must go through `EnvState` (not direct slice imports) — which is correct for JAX immutability anyway.

---

## 9. Single pytree, multi-branch terrain

`EnvState.terrain` is shaped `[N_BRANCHES=7, MAX_LEVELS_PER_BRANCH=32, MAP_H=21, MAP_W=80]`. The pytree always carries every level of every branch.

**Tradeoff.**
- ✅ JAX-friendly: shape is static, JIT-cache-safe.
- ✅ Trivial to access (`state.terrain[branch, level]`).
- ❌ Eager memory: 376k bytes for terrain even when only one level matters. With `LevelMemoryState`, traps, features, level_memory all using the same shape, we burn ~3-4 MB per env state — fine on a single agent, possibly tight at batch 4096.

The alternative — a `Dict[(branch, level), level_state]` — does not survive JIT. Wave 1 chose the static-shape approach. If memory becomes a problem, Wave 6 can introduce a "live level slice" pattern where current level lives at index 0 and historical levels paged out via `lax.scan` updates.

---

## 10. `MAX_LEVELS_PER_BRANCH = 32`

NetHack's `MAXLEVEL`/`global.h` defines the absolute cap as 32 levels per branch. The Main dungeon is up to ~28 deep; Gehennom is 16; Mines/Sokoban/Quest/Vlad/Endgame are all shorter. 32 leaves headroom.

---

## 11. JAX `x64` enabled for blstats parity

NLE's `blstats` is `int64`. JAX defaults to int32. We set `JAX_ENABLE_X64=1` in `tests/conftest.py` so `jnp.zeros((27,), dtype=jnp.int64)` actually allocates an int64 array.

**Tradeoff.** Slight TPU perf cost in production; negligible.

---

## 12. Test scaffold uses lazy imports

`tests/test_imports.py` imports each module *inside* the test function, not at module top. This lets pytest collect tests even when some modules fail to import — important for the parallel-wave development style.

---

## 13. Soft-warned: agents can't run `Bash`

Many parallel agents reported "Bash denied" when trying to verify their work. They returned correct files anyway. The verification step happened in the integration phase (this conversation), which exposed several issues that the agents couldn't catch themselves:

- `obs/__init__.py` had `from nethax.obs.*` instead of `from Nethax.nethax.obs.*` (typo)
- `actions.py` had a now-wrong `119/120` assertion based on the bad audit
- `test_action_enum.py` used the bad audit counts too

**Lesson for Wave 2+:** before dispatching parallel agents, install JAX/etc. so agents can run their own verification scripts. Don't rely on the integrator to find every typo.
