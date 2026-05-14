# Wave 1 — NLE-parity status

The whole point of Wave 1 was to nail the API contract that lets agents trained on NLE run on `nethax` (and vice versa) without rewiring. This doc tracks how close we got.

## Observation dict — 17 keys ✅

Verified against `vendor/nle/nle/nethack/nethack.py:32-50` and `vendor/nle/include/nleobs.h:48-72`.

| Key | Shape | dtype | Status |
|---|---|---|---|
| `glyphs` | `(21, 79)` | `int16` | ✅ shape + dtype match. Zero-filled. |
| `chars` | `(21, 79)` | `uint8` | ✅ |
| `colors` | `(21, 79)` | `uint8` | ✅ |
| `specials` | `(21, 79)` | `uint8` | ✅ |
| `blstats` | `(27,)` | `int64` | ✅ requires `JAX_ENABLE_X64=1` (wired in `tests/conftest.py`) |
| `message` | `(256,)` | `uint8` | ✅ |
| `program_state` | `(6,)` | `int32` | ✅ |
| `internal` | `(9,)` | `int32` | ✅ |
| `inv_glyphs` | `(55,)` | `int16` | ✅ |
| `inv_letters` | `(55,)` | `uint8` | ✅ |
| `inv_oclasses` | `(55,)` | `uint8` | ✅ |
| `inv_strs` | `(55, 80)` | `uint8` | ✅ |
| `screen_descriptions` | `(21, 79, 80)` | `uint8` | ✅ |
| `tty_chars` | `(24, 80)` | `uint8` | ✅ |
| `tty_colors` | `(24, 80)` | `int8` | ✅ |
| `tty_cursor` | `(2,)` | `uint8` | ✅ |
| `misc` | `(3,)` | `int32` | ✅ |

Verification: 17 parametrized shape tests + 17 parametrized dtype tests pass.

**What's NOT done:** projection. Every value is currently `jnp.zeros(...)`. Wave 2 wires:
- `build_glyphs(env_state)` — project terrain + monsters + objects → glyph IDs via offset scheme
- `build_blstats(env_state)` — pack player_hp/pw/str/etc. into 27-vector at canonical indices
- `build_message(env_state)` — read `MessageState.message_buffer`
- `build_tty(env_state)` — render glyphs through `objects`/`monsters`/`cmap` lookup tables
- `build_inventory_strings(env_state)` — `objnam`-style with identification

## Action enum — 121 actions ✅

Verified by exec'ing `vendor/nle/nle/nethack/actions.py` directly:

```
Total ACTIONS: 121
  CompassDirection: 8
  CompassDirectionLonger: 8
  MiscDirection: 3
  MiscAction: 1
  Command: 85
  TextCharacters: 16
NON_RL_ACTIONS: 20
USEFUL_ACTIONS: 101
```

These match `Nethax/nethax/constants/actions.py` exactly. Tests:
- `test_action_count_121` — `N_ACTIONS == 121` ✅
- `test_useful_action_count` — `len(USEFUL_ACTIONS) == 101` ✅
- `test_action_tuple_canonical_length` — `len(ACTIONS) == 121` ✅
- `test_compass_present` — all 8 compass directions present in ACTIONS ✅

**Note** — the audit in our research phase incorrectly said NLE has 119 actions / 95 useful. The canonical vendor counts are **121 actions / 101 useful** (verified by direct exec). Wave 1 fixed both the assertion and the tests to match canonical.

## blstats layout — 27 fields ✅

Indices match `vendor/nle/include/nleobs.h:16-43`:

```
 0: BL_X         13: BL_GOLD
 1: BL_Y         14: BL_ENE
 2: BL_STR25     15: BL_ENEMAX
 3: BL_STR125    16: BL_AC
 4: BL_DEX       17: BL_HD
 5: BL_CON       18: BL_XP
 6: BL_INT       19: BL_EXP
 7: BL_WIS       20: BL_TIME
 8: BL_CHA       21: BL_HUNGER
 9: BL_SCORE     22: BL_CAP
10: BL_HP        23: BL_DNUM
11: BL_HPMAX     24: BL_DLEVEL
12: BL_DEPTH     25: BL_CONDITION
                 26: BL_ALIGN
```

Plus `BL_MASK_*` condition flags (stone, slime, strngl, foodpois, termill, blind, deaf, stun, conf, hallu, lev, fly, ride).

Tests:
- `test_n_blstats_27` ✅
- `test_bl_indices_unique_0_to_26` ✅ — all `BL_*` constants form `range(27)` exactly
- `test_bl_hp_position` — `BL_HP == 10` ✅
- `test_bl_align_position` — `BL_ALIGN == 26` ✅

## Glyph offset scheme — 13 offsets

Loaded as module-level int constants in `Nethax/nethax/constants/glyphs.py`:

```
GLYPH_MON_OFF       GLYPH_OBJ_OFF       GLYPH_ZAP_OFF
GLYPH_PET_OFF       GLYPH_CMAP_OFF      GLYPH_SWALLOW_OFF
GLYPH_INVIS_OFF     GLYPH_EXPLODE_OFF   GLYPH_WARNING_OFF
GLYPH_DETECT_OFF                        GLYPH_STATUE_OFF
GLYPH_BODY_OFF
GLYPH_RIDDEN_OFF
```

Plus `MAX_GLYPH`, `NO_GLYPH`, `NUMMONS`, `NUM_OBJECTS`, `EXPL_MAX`, `NUM_ZAP`, `WARNCOUNT`.

Tests:
- `test_mon_off_zero` — `GLYPH_MON_OFF == 0` ✅
- `test_max_glyph_positive` — `MAX_GLYPH > 0` ✅
- `test_glyph_offsets_monotonic` — offsets in ascending order ✅

**Caveat**: actual integer values were read from `vendor/nle/win/rl/pynethack.cc:477-492` but not cross-checked against a running NLE binary. Wave 2 should add a parity test that imports `nle.nethack` from a real install and compares.

## API surface — `NethaxEnv` ✅

```python
env = NethaxEnv(static: StaticParams | None = None)

state, obs = env.reset(rng: jax.Array)
# state: EnvState
# obs:   Dict[str, jax.Array] with 17 NLE-parity keys

state, obs, reward, done, info = env.step(state, action: jax.Array, rng: jax.Array)
# reward: float32 (Wave 1: always 0.0)
# done:   bool (Wave 1: always False)
# info:   Dict[str, Any] (Wave 1: empty)

# JIT works:
step_jit = jax.jit(env.step)
```

Verified manually with a smoke test that compiles `jax.jit(env.step)` and runs reset→step→step→step.

## What still differs from NLE

| NLE feature | Wave 1 status | Wave to address |
|---|---|---|
| Observation values projected from real state | ❌ all zeros | Wave 2 |
| Action dispatch produces side effects | ❌ no-op | Wave 2 (movement), 3-6 (rest) |
| `info["end_status"]` (`RUNNING`/`DEATH`/`TASK_SUCCESSFUL`/`ABORTED`) | ❌ empty info | Wave 6 |
| `info["is_ascended"]` | ❌ | Wave 6 |
| Auto-handling of `<More>` prompts and Y/N dialogs | n/a — JAX env has no prompts | n/a |
| `wizkit_items` reset arg | ❌ | Wave 4 (after inventory wired) |
| `seed()` method (core/disp split) | partially — env takes `rng` directly | Wave 2 (add explicit `seed()` if needed) |
| TTY rendering escape sequences for human view | ❌ | Wave 3 |
