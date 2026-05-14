# Wave 2 — Test results

## Headline

```
======================== 173 passed in 14.34s ========================
```

Up from 103 in Wave 1 (+70 new tests). Zero failures, zero skips outside `test_vendor_parity.py` skipped only when NLE isn't installed (we have it installed).

## File-by-file

| File | Tests | Wave |
|---|---|---|
| `test_imports.py` | 34 | Wave 1, all still pass |
| `test_action_enum.py` | 4 | Wave 1, updated counts (121 / 101) |
| `test_blstats_layout.py` | 4 | Wave 1 |
| `test_glyph_offsets.py` | 3 | Wave 1, updated ordering (EXPLODE before ZAP) |
| `test_nle_observation.py` | 36 | Wave 1 (17 shape + 17 dtype + 2 key tests) |
| `test_state_slices_construct.py` | 13 | Wave 1 |
| `test_no_op_step.py` | 12 | Wave 1 |
| `test_dungeon_generation.py` | 5 | **Wave 2** — non-overlap, in-bounds, BFS connectivity, JIT, maze connectivity |
| `test_movement.py` | 4 | **Wave 2** — wall blocks, floor walks, OOB blocks, WAIT |
| `test_fov.py` | 5 | **Wave 2** — see self, see neighbors, wall blocks LoS, JIT, blind=adj-only |
| `test_rng.py` | 6 | **Wave 2** — dice range, determinism, key-variance, weighted choice, rn2 range, JIT |
| `test_pixel_obs.py` | 4 | **Wave 2** — shape, dtype, JIT, non-zero on FLOOR |
| `test_obs_projection.py` | 30 | **Wave 2** — HP/DEPTH/glyph/message/tty projection |
| `test_env_lifecycle.py` | 6 | **Wave 2** — reset, step, timestep advance, JIT, movement, dungeon-visible-after-reset |
| `test_state_invariants.py` | 3 | **Wave 2** — pytree leaves, dtype stability, shape stability |
| `test_vendor_parity.py` | 3 | **Wave 2** — Action codes match NLE, BL_* match NLE, GLYPH_* match NLE |

**Total: 173 tests, all passing.**

## What each new Wave 2 test class covers

### `test_dungeon_generation.py`
- Generated rooms are non-overlapping (1-cell margin enforced)
- Rooms within the (21, 80) bounds
- BFS from up-stair reaches down-stair (corridor connectivity proven)
- `jax.jit(generate_main_branch_l1)` compiles
- Kruskal maze: every floor cell reachable from every other (BFS)

### `test_movement.py`
- Wall blocks: player on FLOOR adjacent to WALL; movement toward WALL leaves pos unchanged
- Floor walks: player on FLOOR adjacent to FLOOR; movement updates pos by ±1
- OOB blocks: player at (0,0); action N (north → row=-1) leaves pos at (0,0)
- WAIT: timestep increments but pos unchanged

### `test_fov.py`
- Player sees self: visibility mask True at `player_pos`
- Open 5×5 room: 24 neighbors visible from center
- Wall blocks LoS: a wall directly north of player → tiles 2+ steps north not visible
- JIT: `jax.jit(compute_fov)(terrain, pos)` runs
- Blind radius 1: only cells within Chebyshev distance 1 are visible

### `test_rng.py`
- `dice_roll(key, 3, 6)` ∈ [3, 18]
- Same key → same result
- Different keys → different results (statistical via 10 trials)
- `weighted_choice` with `[100, 0, 0]` → returns 0 every time
- `rn2(key, 10)` ∈ [0, 10) for 100 keys
- JIT each function

### `test_pixel_obs.py`
- Output shape `(336, 1264, 3)`, dtype uint8
- Module constants `MAP_H=21, MAP_W=79, TILE_PX=16, PIXEL_OBS_SHAPE` correct
- `jax.jit(build_pixel_observation)(state)` runs
- Setting `terrain` to all FLOOR produces non-zero pixels (sprite atlas correctly loaded)

### `test_obs_projection.py`
- `obs['blstats'][BL_HP]` reflects `state.player_hp`
- `obs['blstats'][BL_DEPTH]` reflects `state.dungeon.current_level`
- Setting player at `(10, 40)` → `obs['glyphs'][10, 40]` is the player glyph (`GLYPH_MON_OFF + 0`)
- `state.messages.message_buffer` bytes appear in `obs['message']`
- `tty_chars` at player position is `ord('@')`
- ... 25 more parametrized cases for individual blstats fields and obs keys

### `test_env_lifecycle.py`
- `reset` returns valid state + 17-key obs dict
- `step` returns 5-tuple of correct types
- Timestep increments by 1 per step
- `jax.jit(env.step)` compiles
- East movement updates `player_pos[1]`
- After reset, terrain has at least one non-VOID tile (dungeon generation wired in)

### `test_state_invariants.py`
- All leaves of `state` are `jax.Array`
- Dtypes are preserved across `step`
- Shapes are preserved across 3 consecutive steps

### `test_vendor_parity.py` (gated on NLE install)
- `Action.X.value == nle.nethack.X.value` for all shared action names
- `BL_*` constants byte-match `nle.nethack.NLE_BL_*`
- `GLYPH_*_OFF` constants match `nle.nethack.GLYPH_*_OFF`

## Bugs caught during integration

These were issues the parallel agents couldn't catch themselves (Bash denied) but the integration pass found:

1. **121 vs 119 action assertion**: agent was correct (121), assertion was wrong (119). Fixed assertion + tests.
2. **`obs/__init__.py` typo**: `from nethax.obs.*` instead of `from Nethax.nethax.obs.*`. Fixed.
3. **JAX `x64` not enabled**: `blstats` should be int64; needed `JAX_ENABLE_X64=1` in conftest. Fixed.
4. **Action duplicates test wrong premise**: NLE intentionally has key collisions across enum classes. Test relaxed.
5. **FOV `max_steps` too large**: 2R+2 → over-walks; corrected to R.
6. **Glyph offsets estimated**: replaced with live-NLE values.
7. **Glyph ordering**: test asserted ZAP before EXPLODE; canonical is opposite. Test reordered.
8. **`__init__.py` was 0 bytes**: needed deletion and rewrite (Read couldn't handle empty file).
9. **`monsters.py::MONSTERS` had inline 10 entries while chunks were ready**: rewrote master to aggregate from chunks.
10. **`objects.py` over-counted due to dual naming**: documented as Wave 2 debt.

## Test execution

```sh
.venv/bin/python -m pytest                             # all 173
.venv/bin/python -m pytest -x                          # stop at first failure
.venv/bin/python -m pytest -k vendor_parity            # only NLE-parity tests
.venv/bin/python -m pytest tests/test_dungeon_generation.py -v
```

`tests/conftest.py` sets `JAX_PLATFORMS=cpu` and `JAX_ENABLE_X64=1` automatically.
