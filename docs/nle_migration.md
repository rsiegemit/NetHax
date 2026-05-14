# NLE -> Nethax Migration Guide

For RL practitioners who have an existing NLE training script and want to run it on nethax. The promise: change one import line, keep your agent.

For the per-feature compatibility matrix, see [`nle_compat.md`](nle_compat.md). For per-subsystem parity, see [`vendor_parity.md`](vendor_parity.md).

---

## Drop-in import

NLE:

```python
from nle.env import NLE

env = NLE(savedir=None, character="val-hum-law-fem")
```

nethax:

```python
from Nethax.nethax.compat.nle_shim import NLECompat as NLE

env = NLE(seed=0, character="val-hum-law-fem")
```

The shim accepts every NLE constructor kwarg (`savedir`, `wizard`, `allow_all_yn_questions`, `allow_all_modes`, `spawn_monsters`, `options`, `fix_moon_phase`, `max_episode_steps`, `observation_keys`, ...). Where a kwarg has no JAX equivalent, it is accepted as a no-op so existing scripts do not crash.

---

## Gym / Gymnasium API match

The shim implements `gymnasium.Env` (gymnasium 0.26+):

```python
obs, info        = env.reset(*, seed=None, options=None)
obs, reward, terminated, truncated, info = env.step(action)
env.close()
env.seed(core, disp, reseed) -> (core, disp, reseed)
env.get_seeds() -> (core, disp, reseed)
env.render(mode="human" | "ansi" | "full")
env.print_action_meanings()
```

Both `observation_space` and `action_space` are `gymnasium.spaces` instances — see below for shapes / dtypes.

`metadata = {"render_modes": ["human", "ansi", "full"]}` uses the gymnasium 0.26+ key name (`render_modes`) rather than the deprecated gym 0.21 key (`render.modes`).

All recent `gym.wrappers` (`TimeLimit`, `RecordEpisodeStatistics`, etc.) wrap the shim without modification. Coverage: `tests/test_nle_compat_full.py::test_wrapped_by_*`.

---

## Action space — 121 actions, byte-identical

```python
env.action_space  # gymnasium.spaces.Discrete(121)
env.actions       # tuple of 121 vendor action ints
```

The 121-tuple is byte-identical to `nle.nethack.ACTIONS`. Verified in `tests/test_nle_compat_full.py::test_action_space_matches_nle`.

Vendor source: `vendor/nle/nle/nethack/actions.py`.

Action indices are the raw vendor `cmd.c` codes (so `ord('.')` = 46 = wait, `ord('e')` = 101 = eat, etc.). The shim accepts both the action **index** (0..120) and the raw int directly; the underlying `NethaxEnv` only sees the raw int.

---

## Observation space — 17 keys, byte-identical

```python
env.observation_space  # gymnasium.spaces.Dict with 17 keys
```

| Key | Shape | Dtype | Source of truth |
|---|---|---|---|
| `glyphs` | (21, 79) | int16 | `vendor/nle/include/nleobs.h::glyphs` |
| `chars` | (21, 79) | uint8 | `nleobs.h::chars` |
| `colors` | (21, 79) | uint8 | `nleobs.h::colors` |
| `specials` | (21, 79) | uint8 | `nleobs.h::specials` |
| `blstats` | (27,) | int64 | `nleobs.h::blstats` |
| `message` | (256,) | uint8 | `nleobs.h::message` |
| `tty_chars` | (24, 80) | uint8 | `nleobs.h::tty_chars` |
| `tty_colors` | (24, 80) | int8 | `nleobs.h::tty_colors` |
| `tty_cursor` | (2,) | uint8 | `nleobs.h::tty_cursor` |
| `inv_glyphs` | (55,) | int16 | `nleobs.h::inv_glyphs` |
| `inv_letters` | (55,) | uint8 | `nleobs.h::inv_letters` |
| `inv_oclasses` | (55,) | uint8 | `nleobs.h::inv_oclasses` |
| `inv_strs` | (55, 80) | uint8 | `nleobs.h::inv_strs` |
| `screen_descriptions` | (21, 79, 80) | uint8 | `nleobs.h::screen_descriptions` |
| `internal` | (9,) | int32 | `nleobs.h::internal` |
| `misc` | (3,) | int32 | `nleobs.h::misc` |
| `program_state` | (6,) | int32 | `nleobs.h::program_state` |

Every key's shape and dtype is byte-equal to `nle.nethack.OBSERVATION_DESC` and `vendor/nle/include/nleobs.h`. Verified in `tests/test_nle_compat_full.py::test_observation_space_per_key`.

The `observation_keys=` constructor kwarg works exactly as in NLE: pass a subset to filter the obs dict at every step. The space narrows accordingly.

---

## `blstats` — 27-field layout, byte-identical

```python
obs["blstats"]  # shape (27,), int64
```

Field order (from `vendor/nle/include/nleobs.h`):

| Index | Field | Notes |
|---|---|---|
| 0 | NLE_BL_X | cursor x |
| 1 | NLE_BL_Y | cursor y |
| 2 | NLE_BL_STR25 | strength encoded as 3..25 |
| 3 | NLE_BL_STR125 | strength encoded as 3..125 (exceptional STR) |
| 4 | NLE_BL_DEX | |
| 5 | NLE_BL_CON | |
| 6 | NLE_BL_INT | |
| 7 | NLE_BL_WIS | |
| 8 | NLE_BL_CHA | |
| 9 | NLE_BL_SCORE | |
| 10 | NLE_BL_HP | |
| 11 | NLE_BL_HPMAX | |
| 12 | NLE_BL_DEPTH | dungeon depth (Dlvl) |
| 13 | NLE_BL_GOLD | |
| 14 | NLE_BL_ENE | power (Pw) |
| 15 | NLE_BL_ENEMAX | |
| 16 | NLE_BL_AC | armor class |
| 17 | NLE_BL_HD | monster hit dice (player polymorph form) |
| 18 | NLE_BL_XP | experience level |
| 19 | NLE_BL_EXP | experience points |
| 20 | NLE_BL_TIME | turn counter |
| 21 | NLE_BL_HUNGER | hunger state |
| 22 | NLE_BL_CAP | encumbrance level |
| 23 | NLE_BL_DNUM | dungeon number (branch id) |
| 24 | NLE_BL_DLEVEL | level number within branch |
| 25 | NLE_BL_CONDITION | bit mask of status conditions |
| 26 | NLE_BL_ALIGN | alignment |

Verified live against `vendor/nle/include/nleobs.h` and `nle.nethack.BLSTATS_FIELDS` in `tests/test_nle_observation.py`.

---

## Glyph table — 5976 entries, byte-identical

```python
from Nethax.tiles.tile_data import GLYPH2TILE   # shape (5976,), int16
```

Byte-equal to `nle.nethack.glyph2tile`. Verified in `tests/test_nle_compat_full.py::test_glyph2tile_matches_nle`.

Source: `vendor/nle/win/share/tiledata2.txt`.

The glyph predicate helpers (`glyph_is_monster`, `glyph_is_object`, `glyph_is_cmap`, `glyph_is_pet`, `glyph_is_body`, `glyph_is_invisible`, `glyph_is_statue`, `glyph_is_swallow`, `glyph_is_warning`) are byte-equivalent to vendor's `vendor/nethack/include/display.h` C macros and are exported both as static methods on `NLECompat` and as module-level functions in `Nethax.nethax.compat.nle_shim`.

---

## Determinism + seeding

```python
env.seed(core=42, disp=0, reseed=False)
obs, info = env.reset()
```

Same seed -> same obs sequence, byte-for-byte. Verified in `tests/test_nle_compat_full.py::test_determinism_*` and `test_determinism_glyphs_byte_equal`.

**Known divergence**: vendor NLE uses two RNGs (`core` for game logic and `disp` for display effects) plus an Anti-TAS `reseed` flag. nethax has a single JAX `PRNGKey`. The shim accepts the three-arg `seed(core, disp, reseed)` signature for source compatibility — it seeds the JAX key from `core` and round-trips `disp` / `reseed` through `get_seeds()` — but only `core` influences simulation.

For RL training this is rarely material; for replay / TAS workflows it is. See [`nle_compat.md`](nle_compat.md) "Deliberate divergences" #3.

---

## Other divergences vs vendor NLE

These are all enumerated with vendor source citations in [`vendor_parity.md`](vendor_parity.md) "Deliberate divergences" and [`nle_compat.md`](nle_compat.md). Headlines:

| Area | Status | Why |
|---|---|---|
| `savedir` / ttyrec recording | accepted, no-op | nethax is a pure JAX function with no tty stream. |
| `wizard` / `allow_all_modes` / `spawn_monsters` / `options` / `fix_moon_phase` | accepted, no-op | These flags configure vendor's C binary; nethax behaviour is fixed at the JAX layer. |
| Shopkeepers | simplified | No full dialogue tree; pay-at-exit and angry-pursuit are wired. |
| Bones files | not supported | Episode-local corpses only. |
| Save format | divergent | pytree-pickle, not vendor binary. State roundtrips bit-equal. |
| Wizard-mode debug commands | out of scope | `cmd.c::wiz_*` not exposed. |
| Mail / music / sounds | out of scope | TTY-side effects with no JAX equivalent. |
| Lua runtime | out of scope | nethax ports MAP strings statically; no in-game Lua VM. |

If your training script touches any of these areas explicitly, those calls will be silent no-ops on the nethax shim.

---

## Performance

nethax has a different performance profile than NLE. See [`benchmark.md`](benchmark.md) for full numbers — short version:

- **Single-env, post-warmup (CPU)**: nethax ~140 sps, NLE ~10 000-20 000 sps. NLE wins by ~100x at batch=1.
- **JIT compile cost**: nethax pays ~30-60 s once per Python process. NLE has no warmup.
- **vmap batch=512 (CPU)**: nethax wins. The batched env is a single fused XLA kernel; NLE is fork-per-env with no batched path.
- **vmap batch=4096+, or GPU**: nethax wins by 10-100x in aggregate throughput.
- **`lax.scan` long rollouts**: the entire trajectory is a single compiled XLA computation; Python-loop overhead is eliminated entirely.

**Recommendation**: if your training loop is `for env in range(N_ENVS): env.step(...)` at batch=1, NLE is faster. If you can refactor to `vmap(env.step)(batched_state, batched_action, batched_rng)` at batch >= 512, nethax wins.

Migration patterns:

```python
# Sequential (NLE-style, slow on nethax)
for i in range(n_envs):
    states[i], rewards[i], dones[i] = envs[i].step(actions[i])

# Batched (nethax-native, fast)
import jax
step_batched = jax.vmap(env._step_jit)
new_states, obs, rewards, dones = step_batched(states, actions, rngs)
```

For long rollouts:

```python
# Single XLA computation for an entire 1000-step trajectory.
def rollout(state, rngs):
    def body(state, rng):
        state, obs, reward, done = env._step_jit(state, action, rng)
        return state, (reward, done)
    return jax.lax.scan(body, state, rngs)
```

---

## Migration checklist

1. **Swap the import**: `from nle.env import NLE` -> `from Nethax.nethax.compat.nle_shim import NLECompat as NLE`.
2. **Run your existing eval script** at batch=1 — should produce results identical in shape / dtype to NLE.
3. **Verify determinism** with your seed-fixing logic.
4. **If your env loop is sequential**: expect ~100x slowdown. Consider refactoring to batched.
5. **If you use `savedir` / ttyrec / wizard mode / Lua hooks**: those become no-ops. Re-architect the workflow if they were load-bearing.
6. **Move to GPU** by setting `JAX_PLATFORMS=gpu`. No env code changes required.

---

## Cross-references

- [`nle_compat.md`](nle_compat.md) — exhaustive per-feature compat matrix.
- [`vendor_parity.md`](vendor_parity.md) — per-subsystem vendor parity + divergences.
- [`architecture.md`](architecture.md) — top-down system architecture (useful if you want to extend the env).
- [`benchmark.md`](benchmark.md) — throughput numbers.
- `tests/test_nle_compat_full.py` — 36 tests verifying byte-equality.
- `tests/test_nle_observation.py` — 35 tests covering observation shape + dtype.
- `tests/test_nle_integration.py` — 12 end-to-end integration tests.
