# Wave 2 — Design decisions

## 1. Trust the live NLE binary over header arithmetic

Wave 1 computed glyph offsets from formulas applied to NUMMONS / NUM_OBJECTS / MAXPCHARS in C headers. The estimates were wrong by ~70% in some places (`GLYPH_CMAP_OFF=4000 vs canonical 2359`). Wave 2 installed NLE in the project venv and read `nle.nethack.GLYPH_*_OFF` directly.

**Going forward**: any value NLE exposes through its public Python API is the source of truth. Don't compute from C macro expansion — let the compiled binary tell us.

## 2. Parallel chunked agents over single mega-agent for mechanical translation

The Wave 1 monsters and objects agents both stalled on the size of `vendor/nethack/include/monsters.h` (3,900 lines) and `objects.h` (1,659 lines). Trying to translate everything in one agent invocation = context exhaustion before completion.

Wave 2 split each table into chunks, dispatched 6 monster-chunk + 5 object-chunk + 3 gap-fill agents in parallel. Each owned a distinct file under `monster_entries/` or `object_entries/`. The master `monsters.py` / `objects.py` aggregates via simple imports + tuple concatenation.

**Tradeoff:**
- ✅ Parallel: 11 agents × 1–3 min wall clock each = ~3 min total.
- ✅ Failure isolation: a stalled chunk doesn't block other chunks.
- ❌ Slight duplication across chunks (food/tools/gems appeared in both `OBJECTS_BASE` and dedicated chunk files). Mitigated by `(name, class_)` dedup at aggregation time.

## 3. Aggregate-by-import, dedup-at-the-edge

Master files don't inline data; they import chunk `ENTRIES` tuples and concatenate. This means:

- Adding a new monster category is a new file in `monster_entries/` + one import line in `monsters.py`.
- Bugs are isolated to single chunks.
- The aggregation step does the dedup — chunks can over-supply without breaking anything.

The dedup key is `(name, int(class_))` for objects and falls back to source-order for monsters (no dedup needed there since chunks have non-overlapping ranges).

## 4. Dual-naming for objects accepted as Wave 2 debt

Existing `OBJECTS_BASE` entries use verbose names like `"potion of healing"`. NLE canonical is `"healing"` (the "potion of" prefix is added at render time via `obj_descr`). Wave 2's `misc_gaps` agent added the bare-name versions for parity.

After dedup, `OBJECTS` has **both** naming conventions — 503 total vs NLE's 453. This is **deliberately accepted as Wave 2 debt**: changing names in `OBJECTS_BASE` risks breaking any in-development consumer that already references the verbose names. Wave 3 (or whenever item effects get wired) will canonicalize to NLE bare names and drop the verbose forms.

## 5. Doors carved but not placed

`dungeon/corridors.py::place_doors` is implemented but **not called** by the default `generate_main_branch_l1` pipeline. The reason: in Wave 2 the movement handler treats `CLOSED_DOOR` as solid (correct), and we haven't yet wired the bump-to-open behavior, so placing doors would sever connectivity in the BFS connectivity test.

When Wave 3 lands bump-to-open, doors get re-enabled in one line.

## 6. `MAX_LEVELS_PER_BRANCH=32` static memory carries every level

Even with just one level generated, the pytree carries `terrain` shaped `[7 branches, 32 levels, 21, 80]` = 376k int8s ≈ 370 KB. At batch size 4096 on a GPU, that's 1.4 GB just for terrain.

**Why we keep it this way**: static shapes are mandatory for JIT. Going to "dynamic active level" would require either (a) introducing host-side level lookup (kills JIT) or (b) a more complex pytree that swaps active/inactive levels via index tricks.

Wave 4 will revisit when level memory wiring goes live. If batch sizes prove tight, the alternative is to carry only the current level + a host-side dict of inactive levels rolled back in at stair traversal (acceptable since stair traversal is rare).

## 7. Pixel rendering eager-loads tile atlas at import time

`obs/pixel_obs.py::_get_tiles()` uses a Python-level cache that loads `Nethax/tiles/tiles.npy` once on first call. This is **outside the JIT boundary** — at trace time, the atlas is a static `jnp.array` constant baked into the compiled XLA program.

**Why**: a `numpy.load` inside `build_pixel_observation` would fail JIT (file I/O isn't traceable). Hoisting the load outside is the standard JAX pattern for static resources.

## 8. Tests use lazy imports + `pytest.mark.skipif`

Pattern reused from Wave 1: every test imports inside its body, not at module top. New for Wave 2: `tests/test_vendor_parity.py` uses `@pytest.mark.skipif(not nle_installed)` so the suite runs cleanly on machines without NLE.

NLE installs cleanly via `pip install nle` (we did this in the project venv). If a contributor doesn't install NLE, vendor-parity tests skip silently.

## 9. Single-direction FOV walk-then-stop

Bresenham FOV cast walks each ray for at most `R` steps (where `R = sight_radius`), not the wave-1 first-pass `2R + 2`. The earlier bound over-walked rays past the radius edge.

Diagonal targets that are at Chebyshev distance < `R` still get over-marked along their continued direction, but those over-marks are within `R` anyway (the algorithm casts to every cell in the bounding box, not just the perimeter, so over-marks land on cells covered by other rays).

## 10. Run-length cap = 64 iterations

`dispatch_action._run` uses `jax.lax.while_loop` with a 64-iteration cap. NetHack's longest possible corridor is ~80 cells. 64 covers ~98% of realistic runs and keeps trace size bounded.

The cap also prevents pathological infinite loops if state somehow makes `_try_step` cycle without progress.

## 11. Static actions lookup vs `jnp.array` switch

`subsystems/action_dispatch.py::_ACTION_TO_HANDLER_IDX` is a 256-entry int8 lookup keyed by ASCII action value. `dispatch_action` reads the handler index from this lookup and runs `jax.lax.switch(idx, _HANDLERS, state, rng)`.

We chose `lax.switch` over a hash-map / `cond` cascade because:

- 256 is small, so the lookup is one indexed read.
- `lax.switch` compiles to a single jump table in XLA.
- Adding a new action handler is one line in `_HANDLERS` + the lookup table.

## 12. Agents can't run Bash; trust-but-verify at integration

Six Wave 2 agents reported "Bash denied" when trying to run their own verification scripts. The integrator (this conversation) re-verified each chunk by import + count.

**Lesson reinforced from Wave 1**: agents return code, not proof of working code. Integration tests are non-negotiable.
