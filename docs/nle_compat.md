# NLE drop-in compatibility status

`Nethax.nethax.compat.nle_shim.NLECompat` is the drop-in shim for
`nle.env.NLE`.  Existing RL training scripts that import NLE should run
unchanged if they import the shim instead.

This document is the source of truth for what is in parity, what is a
documented divergence, and where to look in the vendor source.

Validated against vendor NLE **1.3.0** (gymnasium 0.26+ API).

| Feature | nethax shim | NLE | Parity | Vendor source |
| --- | --- | --- | --- | --- |
| Inherits `gymnasium.Env` | yes | yes (NLE 1.3+) | full | `vendor/nle/nle/env/base.py:137` |
| `reset(*, seed, options) -> (obs, info)` | yes | yes | full | `vendor/nle/nle/env/base.py::reset` |
| `step(action) -> (obs, reward, terminated, truncated, info)` | yes | yes | full | `vendor/nle/nle/env/base.py::step` |
| `close()` | yes (no-op, pure JAX) | releases C handle | full surface | `vendor/nle/nle/env/base.py::close` |
| `seed(core, disp, reseed) -> (core, disp, reseed)` | yes (core seeds JAX PRNG, disp/reseed stored for parity) | yes (true RNG seeding) | partial (single RNG, no TAS reseed) | `vendor/nle/nle/env/base.py::seed` |
| `get_seeds()` | yes — last triple returned | yes — current vendor state | partial (echoes last `seed()` call) | `vendor/nle/nle/env/base.py::get_seeds` |
| `render(mode)`: human / ansi / full | yes | yes | full (renders tty buffer) | `vendor/nle/nle/env/base.py::render` |
| `print_action_meanings()` | yes | yes | full | `vendor/nle/nle/env/base.py::print_action_meanings` |
| `observation_space` `gym.spaces.Dict` | yes (17 keys) | yes (14-17 depending on `observation_keys` kwarg) | full superset | `vendor/nle/nle/env/base.py` `NLE_SPACE_ITEMS` |
| `action_space` `gym.spaces.Discrete(121)` | yes | yes | full | `vendor/nle/nle/env/base.py:324` |
| `metadata` | `{"render_modes": ["human","ansi","full"]}` | `{"render.modes": ["human","ansi"]}` (old key) | partial — we use the gymnasium-0.26+ key | `vendor/nle/nle/env/base.py:150` |
| `StepStatus` (ABORTED=-1 / RUNNING=0 / DEATH=1) | yes | yes | full | `vendor/nle/nle/env/base.py::NLE.StepStatus` |
| `last_observation` (tuple in obs-key order) | yes | yes | full surface; values from JAX obs builder | `vendor/nle/nle/env/base.py:238` |
| `character` (e.g. `"mon-hum-neu-mal"`) | yes — parsed to Role/Race/alignment | yes — passed to vendor Nethack | full surface; sub-token gender ("mal"/"fem") accepted but ignored | `vendor/nle/nle/env/base.py:228` |
| `observation_keys` subset filter | yes | yes | full | `vendor/nle/nle/env/base.py::__init__` |
| `max_episode_steps` truncation | yes — `truncated=True` once `_steps >= limit` | yes (`StepStatus.ABORTED`) | full | `vendor/nle/nle/env/base.py::_check_abort` |
| `savedir` / `save_ttyrec_every` | accepted in kwargs, ignored (`savedir` is always `None`) | writes ttyrec files | divergent — pure JAX env has no tty stream | `vendor/nle/nle/env/base.py:240-258` |
| `wizard` / `allow_all_yn_questions` / `allow_all_modes` / `spawn_monsters` / `options` / `fix_moon_phase` | accepted in kwargs, no-ops | actual game-mode flags | divergent — knobs irrelevant to nethax simulator | `vendor/nle/nle/env/base.py::__init__` |
| Action enum byte-identical to `nle.nethack.ACTIONS` (121 ints) | yes | yes | full — every entry equal | `vendor/nle/nle/nethack/actions.py` |
| `glyph2tile` table (5976 int16 entries) | yes — `Nethax.tiles.tile_data.GLYPH2TILE` byte-equal | yes | full | `vendor/nle/win/share/tiledata2.txt` |
| Per-key obs shape (vendor `OBSERVATION_DESC`) | full — 17/17 keys match | yes | full | `vendor/nle/nle/nethack/nethack.py::OBSERVATION_DESC`, `vendor/nle/include/nleobs.h` |
| Per-key obs dtype | full — 17/17 keys match | yes | full | same as above |
| Glyph predicate helpers (`glyph_is_monster`, …) | yes (static methods + module aliases) | yes | full | `vendor/nethack/include/display.h` macros |
| Determinism: same seed → same obs sequence | yes (JAX `PRNGKey`-backed) | yes (core+disp seeded) | full | n/a — covered by `test_nle_compat_full.py::test_determinism_*` |
| Gymnasium wrapper compatibility (`TimeLimit`, `RecordEpisodeStatistics`) | yes | yes | full | n/a — covered by `test_wrapped_by_*` |

## Deliberate divergences (no fix planned)

1. **`savedir` / `save_ttyrec_every`**: NethaxEnv is a pure JAX function; it has
   no underlying tty stream and produces no ttyrec files.  The kwargs are
   accepted for vendor-signature parity; `nh.savedir` is always `None`.
2. **`wizard`, `allow_all_yn_questions`, `allow_all_modes`, `spawn_monsters`,
   `options`, `fix_moon_phase`**: these flags toggle behaviour inside vendor's
   C-level NetHack binary.  NethaxEnv's behaviour is fixed at the JAX layer
   and has no equivalent runtime hooks.  Kwargs are accepted as no-ops so
   training scripts that pass them do not crash.
3. **RNG model**: vendor NLE uses two RNGs (`core` + `disp`) plus an Anti-TAS
   `reseed` flag.  NethaxEnv has a single JAX `PRNGKey`.  Our `seed(core,
   disp, reseed)` seeds the JAX key from `core` and stores `disp`/`reseed`
   so `get_seeds()` round-trips the values the caller supplied — but they do
   not influence simulation.
4. **`metadata` key name**: we publish `render_modes` (gymnasium 0.26+
   convention) instead of `render.modes` (gym 0.21 legacy).  All recent
   gymnasium wrappers consume the new key.

## Test coverage

See `tests/test_nle_compat_full.py` — 36 tests, all passing, covering:

- gymnasium API shape (reset/step/close/seed/render signatures + return types)
- `observation_space` and `action_space` byte-equality with vendor
  (`nle.nethack.OBSERVATION_DESC`)
- runtime obs key/shape/dtype byte-equality
- action enum byte-identical to `nle.nethack.ACTIONS`
- seed determinism (same seed → same `glyphs`, `tty_chars`, reward, term/trunc)
- `gym.wrappers.TimeLimit` and `RecordEpisodeStatistics` wrap without error
- `max_episode_steps` triggers `truncated=True`
- `last_observation` tuple parity (vendor obs-key order)
- `glyph2tile` 5976-entry byte-equality with `nle.nethack.glyph2tile`
- `StepStatus` enum values byte-equal to vendor
- `info["end_status"]` populated on every step
- 100-seed glyph-bounds sweep
- 50-step shape/dtype stability sweep
- module-level glyph predicate helpers exported

Plus `tests/test_nle_compat.py` (4), `tests/test_nle_observation.py` (35),
`tests/test_nle_integration.py` (12).  Total NLE-compat: **87 tests**.
