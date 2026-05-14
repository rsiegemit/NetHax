# Wave 2 — Mechanics Status

What was a Wave 1 stub vs what's real after Wave 2.

Legend: ✅ implemented · ⚠️ partial · ⏳ still no-op (waiting on later wave)

## Core loop & dispatch

| Subsystem | Status | Notes |
|---|---|---|
| `subsystems/action_dispatch.py::dispatch_action` | ✅ | `jax.lax.switch` over 256-entry lookup table. 21 handlers wired (8 dir + 8 run + 2 stair + 1 wait + 1 no-op). All others fall through to no-op. |
| Movement (`_move_*`) | ✅ | Bounds check + tile-type solidity check via `jnp.where`. Walks onto FLOOR/CORRIDOR/OPEN_DOOR/STAIRCASE_*; blocked by WALL/CLOSED_DOOR/VOID. |
| Run-until-blocked (`_run_*`) | ✅ | `jax.lax.while_loop` with 64-iteration max cap. Terminates when `_try_step` leaves `player_pos` unchanged. |
| Stair traversal (`_stair_up/_stair_down`) | ⚠️ | Bumps `current_level` ±1 if player is on the matching stair tile. **Within-branch only** — `traverse_stair` reads `state.dungeon.stair_links` but multi-branch wiring is Wave 4. |
| Wait | ✅ | Returns state unchanged. Timestep ticks via outer `env.step`. |
| FOV update on move | ✅ | Each successful move calls `compute_fov` then `update_explored`. |

## Dungeon

| Subsystem | Status | Notes |
|---|---|---|
| `dungeon/rooms.py::generate_rooms` | ✅ | Rejection sampling of up to 8 non-overlapping rects (4–10 × 3–6) via `lax.fori_loop` + `lax.scan` over 16 retry candidates per slot. 1-cell margin enforced by `jax.vmap` overlap check. |
| `dungeon/rooms.py::carve_rooms_into_terrain` | ✅ | `lax.scan` over 40 room slots; carves WALL ring + FLOOR interior. |
| `dungeon/rooms.py::connect_rooms` | ✅ | L-shaped corridor between every consecutive active room pair. |
| `dungeon/corridors.py::place_doors` | ⚠️ | Implemented but **not called** in the default pipeline yet (treating doors as solid would sever connectivity until Wave 3 movement handles bumping doors open). |
| `dungeon/mazes.py::generate_maze_kruskal` | ✅ | Real Kruskal: edges shuffled by `jax.random.permutation`, processed by `lax.scan` over the full edge list, union-find with 8-hop path halving. |
| `dungeon/mazes.py::generate_maze_perfect/dla` | ⏳ | Still stubs returning zeros. Wave 3+. |
| `dungeon/branches.py::generate_main_branch_l1` | ✅ | Full pipeline: `generate_rooms → carve → connect → up_stair in room[0] → down_stair in last active room`. Returns `(terrain, rooms, active, up_pos, down_pos)`. |
| `dungeon/branches.py::traverse_stair` | ✅ | Reads `stair_links[branch, level-1, direction]` for `(dest_branch, dest_level)`. Direction 0=up, 1=down. |
| `dungeon/branches.py::enter_branch` | ⚠️ | Updates `current_branch` / `current_level` / `branch_levels[branch_id]`. Branch graph not yet *populated* — branches beyond Main are still empty. |
| `dungeon/level_memory.py::enter_level` | ✅ | If `generated[branch, level]` is False, generates the level and caches `terrain` + `level_rng_seed`. If True, restores from cache. |
| `dungeon/level_memory.py::leave_level` | ✅ | Writes current `terrain` + `explored` back to cache. |
| `dungeon/special_levels.py` | ⏳ | All 28 entries still produce zero arrays. Wave 4–5. |

## FOV & visibility

| Subsystem | Status | Notes |
|---|---|---|
| `fov.py::compute_fov` | ✅ | Bresenham raycast to every cell in `[-R, R] × [-R, R]` bounding box. Each ray walks ≤ `R` steps. Opaque tiles (VOID/WALL/CLOSED_DOOR) block continuation but are themselves marked visible. |
| `fov.py::update_explored` | ✅ | `explored OR fov`. |

## Observation projection (`obs/nle_obs.py`)

| Key | Status | Notes |
|---|---|---|
| `glyphs` | ✅ | TileType → cmap_index via static lookup, + `GLYPH_CMAP_OFF`. Player overlaid at `state.player_pos`. Unexplored tiles → `NO_GLYPH`. Shape `(21, 79)` int16. |
| `chars` | ✅ | Same pipeline as glyphs but via `_CMAP_TO_CHAR` lookup → ASCII. Shape `(21, 79)` uint8. |
| `blstats` | ✅ | All 27 fields populated. `BL_AC=10` and `BL_CONDITION=0` are placeholders (Wave 3 wires armor calc + condition bits). |
| `message` | ✅ | Direct read of `state.messages.message_buffer`. |
| `tty_chars` | ✅ | 24×80 grid: row 0 = message, rows 1-21 = glyph→char + `@` overlay, rows 22-23 = stats (status line). |
| `tty_cursor` | ✅ | At `(player_row + 1, player_col)`. |
| `tty_colors` | ⏳ | Zero. Wave 3. |
| `colors`, `specials` | ⏳ | Zero. Wave 3. |
| `inv_glyphs`, `inv_letters`, `inv_oclasses`, `inv_strs` | ⏳ | Zero. Wave 3 (after inventory wired). |
| `screen_descriptions`, `program_state`, `internal`, `misc` | ⏳ | Zero. Wave 4. |

## Pixel observation

| Function | Status | Notes |
|---|---|---|
| `obs/pixel_obs.py::build_pixel_observation` | ✅ | Uses `Nethax/tiles/tiles.npy` atlas + `tile_data.GLYPH2TILE` lookup. Output shape `(336, 1264, 3)` uint8 (21×16 × 79×16). JIT-compatible. |

## RNG (`rng.py`)

| Function | Status |
|---|---|
| `dice_roll(rng, n, sides)` | ✅ — `jnp.sum(jax.random.randint(...))` |
| `rnd(rng, n)` | ✅ — single die roll in [1, n] |
| `rn2(rng, n)` | ✅ — uniform int in [0, n) |
| `weighted_choice(rng, weights)` | ✅ — `jax.random.choice` with explicit probs |
| `split_n(rng, n)` | ✅ — wraps `jax.random.split` |

## What's still no-op

Combat, magic, monster AI, polymorph, item / inventory operations, traps, features (doors/fountains/altars/sinks/thrones), prayer, conduct violations, shop, quest, status-effect ticks, scoring, messages emit. All have correct state slices and function signatures — bodies return input unchanged.

The Wave 2 commitment was *breadth-first scaffolding remains intact, plus core mechanics needed to move around and observe a generated dungeon*. Behavior beyond that is Wave 3+.
