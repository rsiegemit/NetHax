# Wave 1 → Wave 2 — Scope preview

Wave 2's job: take the scaffolded API surface and **make a player who can walk around a randomly generated dungeon**. Same breadth approach as Wave 1, but now mechanics matter.

> **All choices below are proposals. Confirm or redirect before launching `/ultrawork` for Wave 2.**

## Wave 2 deliverable

A `NethaxEnv` where:

1. A real procedurally generated dungeon appears on `reset()`.
2. Movement actions (8 compass directions + stair traversal) actually move the player.
3. The NLE observation dict is no longer zero-filled — it projects the real `EnvState`.
4. Field-of-view masks the map properly.
5. The dice/RNG primitives produce JAX-randomized values.
6. Existing 103 tests still pass + new tests verify movement and observation projection.

This makes the env *playable but trivial*: no monsters react, no items, no combat. Just walking around a dungeon.

## Wave 2 breadth pass items

### Dungeon generation breadth (the big one)

- `rooms.py::generate_rooms` — non-overlapping rectangular rooms (5-10 per level, BSP or rejection sampling)
- `corridors.py::connect_segments` — L-shaped corridors connecting rooms with `place_doors`
- `mazes.py::generate_maze_kruskal` — perfect-maze for Mines lower half and Quest
- Stair placement: random valid floor in random rooms for up + down stairs
- Replace the all-zero `terrain` with actual generated levels for the Main branch

### Action dispatch breadth

Implement the dispatch table for **movement-class actions only**:

- `CompassDirection` × 8 (cardinal + intercardinal): walk if floor, bump if wall, open if closed door
- `CompassDirectionLonger` × 8: walk-until-blocked
- `MiscDirection.UP` / `DOWN`: stair traversal (only within same branch in Wave 2; Wave 4 wires branches)
- `MiscDirection.WAIT`: tick one turn

Use `jax.lax.switch(action_index, ACTION_HANDLERS, state, rng)` pattern in `subsystems/action_dispatch.py`.

### Observation projection

Replace the zero-fill stubs in `obs/nle_obs.py`:

- `build_glyphs(env_state)` — for each tile: terrain → `GLYPH_CMAP_OFF + cmap_index`; monster → `GLYPH_MON_OFF + monster_id`; object → `GLYPH_OBJ_OFF + obj_id`. Player is `GLYPH_MON_OFF + 0`-ish.
- `build_blstats(env_state)` — pack `EnvState.player_*` into the 27-vector.
- `build_message(env_state)` — read `messages.message_buffer` directly.
- `build_tty(env_state)` — render the glyph-to-char mapping into a 24×80 grid.
- `build_inventory_strings(env_state)` — placeholder OK if items are still empty.

### FOV

- Replace `fov.compute_fov` all-visible stub with a real raycast (start with simple 8-direction Bresenham-ish; shadowcasting is nicer but Wave 3 can upgrade).
- Plug into `step()` so `state.visible` updates on each move.
- Plug into `state.explored` (OR-accumulate).

### RNG primitives

- `rng.dice_roll(rng, n, sides)` → `jnp.sum(jax.random.randint(rng, (n,), 1, sides+1))`
- `rng.weighted_choice(rng, weights)` → `jax.random.choice(rng, len(weights), p=weights/weights.sum())`

### Item tables — data only

Populate the full `MONSTERS` and `OBJECTS` tuples in `constants/{monsters,objects}.py` from `vendor/nethack/src/{monst,objects}.c` macro tables. Schema is already defined; this is a mechanical port.

### Tests Wave 2 should add

- `test_dungeon_generation.py` — generated terrain has connectivity (BFS from up-stair reaches down-stair)
- `test_movement.py` — north action moves player up if floor; doesn't if wall
- `test_fov.py` — player at center of open room sees all tiles within radius 7
- `test_obs_projection.py` — `build_blstats` returns `BL_HP == state.player_hp`
- `test_vendor_parity.py` — import `nle.nethack` and assert our `Action` enum codes match (this requires installing NLE properly; mark `@pytest.mark.requires_nle`)

### Wave 2 deferred (still no-op after Wave 2)

- Combat (Wave 3)
- Magic (Wave 3)
- Status effect ticks (Wave 3)
- Monster AI (Wave 3)
- Traps trigger (Wave 3)
- Branch traversal (Wave 4)
- Special levels (Wave 4-5)

## Risks for Wave 2

1. **`jax.lax.switch` of 121 handlers** may be slow to compile. Mitigation: group movements into a single handler that takes the action as input and switches inside via `jnp.where`.
2. **Dungeon generation in JAX** is the hardest item. Non-overlapping room placement is hard to vectorize; we'll need `lax.while_loop` with a rejection-sampling pattern. Alternative: fixed grid of cells, randomly enable some.
3. **Monster/object tables are big.** 394 monsters × 19 fields + 459 objects × 13 fields = ~14k constants. Should be Python tuples at module load, but may slow import. Mitigation: lazy load via property.
4. **`build_tty` is rendering**, and rendering with `lax.scan` over 24×80 tiles is ~2000 ops per step. Should be fine but worth benchmarking.

## Recommended Wave 2 launch

Same shape as Wave 1: 8-12 parallel executor agents, each owning distinct files:

- Agent: `rooms.py` + `corridors.py` + `mazes.py` real algorithms
- Agent: branch-graph initialization in Main only
- Agent: monster table population (394 entries)
- Agent: object table population (459 entries)
- Agent: action dispatch movement handlers
- Agent: `build_glyphs` + `build_blstats`
- Agent: `build_tty` + `build_message`
- Agent: real FOV
- Agent: dice / weighted_choice in `rng.py`
- Agent: tests (test_dungeon_generation, test_movement, test_fov, test_obs_projection)

## Open questions for the user

1. **Should we install NLE in `.venv` for parity tests?** It's heavy (~1 GB build) but lets us write authoritative vendor-parity tests.
2. **Pixel observation priority — Wave 2 or Wave 3?** Symbolic + tty render covers most baselines; pixel is for human watching + some CNN agents.
3. **Migration of old `nethax_state.py` / `game_logic.py` — Wave 2 or Wave 6?** Recommend Wave 6: leave the legacy code working until the new env is full-featured, then delete in one pass.
4. **MiniHack curriculum (170 envs) — keep at Wave 5 or pull forward?** Pulling forward gives near-term RL benchmarks; deferring keeps the focus on `nethax` proper.
