# Wave 1 — Test results

## Headline

```
======================== 103 passed in 1.71s ========================
```

All 103 tests pass on `python3.12 + jax >= 0.4 + JAX_ENABLE_X64=1`.

## Test files

| File | Tests | Purpose |
|---|---|---|
| `test_imports.py` | 34 | Smoke import every module in `constants/`, `subsystems/`, `dungeon/`, `obs/`, plus `fov`, `rng`, `save_load` |
| `test_action_enum.py` | 4 | Verify 121 actions, 101 useful, 8 compass dirs present |
| `test_blstats_layout.py` | 4 | Verify 27 indices, no gaps, `BL_HP==10`, `BL_ALIGN==26` |
| `test_glyph_offsets.py` | 3 | Verify `GLYPH_MON_OFF==0`, `MAX_GLYPH>0`, offsets monotonic |
| `test_nle_observation.py` | 36 | 17-key parity + 17 shapes + 17 dtypes (parametrized) |
| `test_state_slices_construct.py` | 13 | Each subsystem state class constructs with its `.default()` / `.empty()` / `.unshuffled()` / factory |
| `test_no_op_step.py` | 12 | Each subsystem with a `step(state, rng)` returns input unchanged (JAX leaf equality via `jax.tree_util`) |

## What's verified

### NLE API parity (shape + dtype, not value)

The 17 observation keys, their shapes, and their dtypes all match `vendor/nle/include/nleobs.h:48-72` exactly. A drop-in NLE-trained agent that introspects observation space metadata will see exactly the structure it expects.

```python
NLE_OBSERVATION_SHAPES = {
    "glyphs": (21, 79),      "blstats": (27,),
    "chars": (21, 79),       "message": (256,),
    "colors": (21, 79),      "tty_chars": (24, 80),
    "specials": (21, 79),    ...
}
```

### Action enum integrity

`N_ACTIONS == 121` (matches `vendor/nle` direct exec). `USEFUL_ACTIONS` excludes the 20 `NON_RL_ACTIONS` → 101. The 8 compass directions all appear in `ACTIONS`.

### blstats layout integrity

`{BL_X, BL_Y, ..., BL_ALIGN}` = `{0, 1, ..., 26}` exactly (no gaps, no duplicates). `BL_HP==10`, `BL_ALIGN==26` per NLE convention.

### State pytree composition

Every subsystem state class can be instantiated via its canonical constructor and the result is a valid Flax pytree. `EnvState.default(rng=jax.random.PRNGKey(0))` produces a complete master pytree.

### JIT compatibility

(Verified out-of-band, not in pytest.) `jax.jit(env.step)` compiles the full step function. All `step()` functions are pure-functional and JIT-safe.

## What's NOT verified (yet)

- **Behaviour.** All `step()` functions are no-ops. There are no tests for "movement actually moves the player" because movement isn't implemented.
- **Cross-vendor consistency.** Glyph offset integer values come from `pynethack.cc` and should be cross-checked against a running NLE binary. Wave 2 should add a `tests/test_vendor_parity.py` that imports `nle.nethack` from a real install and compares offsets / action codes / blstats indices.
- **Monster table completeness.** `MONSTERS` has 10 entries vs. canonical 394. Wave 2 fills the rest; coverage will become testable then.
- **Object table completeness.** `OBJECTS` has 10 entries vs. canonical 459. Same.
- **Identification appearance shuffling.** State exists, mechanic does not.

## Test-execution environment

- Python: 3.12.13 (homebrew)
- JAX: latest (`>= 0.4.0`)
- Flax: latest
- pytest: 9.0.3
- `JAX_PLATFORMS=cpu` + `JAX_ENABLE_X64=1` set in `tests/conftest.py`

## Failures encountered during Wave 1 integration

For posterity — these were the bugs the integration phase found and fixed (none surfaced from the parallel agents' own validation, which couldn't run Bash):

1. **Wrong action count assertion.** Agent generated 121 actions (correct!) but the assertion in `actions.py` said `119 or 120`. Fixed assertion to `== 121`; corresponding test updated.

2. **Wrong USEFUL_ACTIONS construction.** Agent's logic was over-pruning (removed all `TextCharacters` then re-added `SPACE`), giving 86 instead of canonical 101. Fixed to match vendor exactly.

3. **`obs/__init__.py` import typo.** Agent used `from nethax.obs.*` instead of `from Nethax.nethax.obs.*`, dropping the package prefix. Fixed.

4. **JAX x64 not enabled.** `blstats` is supposed to be `int64`. JAX silently truncates to `int32` without `JAX_ENABLE_X64`. Added to `conftest.py`.

5. **Test for action-value uniqueness was wrong premise.** NLE intentionally has key-value collisions across enum classes (movement 'h' shares ord with Command members). Test now asserts canonical length, not set uniqueness.

## How to run

```sh
.venv/bin/python -m pytest                       # all 103 tests
.venv/bin/python -m pytest tests/test_imports.py # just import smoke
.venv/bin/python -m pytest -k "blstats"          # blstats-related
.venv/bin/python -m pytest -v                    # verbose
```
