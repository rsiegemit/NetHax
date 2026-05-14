# Wave 2 — Mechanics, Migration, Full NLE-Parity Data

**Status:** ✅ Complete · **173/173 pytest tests passing** · Movement + dungeon + FOV + observation projection + pixel render all live · 390 monster entries · 503 object entries · legacy code removed.

Wave 2's mission was three-fold:
1. **Make a player who can walk around a real generated dungeon** — fill the no-op stubs from Wave 1.
2. **Populate the canonical NetHack 3.7 data tables in full** (all monsters and items).
3. **Delete the legacy `nethax_state.py` / `game_logic.py` / `renderer.py` / `world_gen/` / `envs/` / `util/` code paths now that the new package subsumes them.**

All three landed. Wave 2 also surfaced and fixed several Wave 1 inaccuracies — most notably the audit-reported "119 actions" should have been 121, and the glyph offset constants were synthetic estimates; both are now verified directly against a running NLE build.

---

## What shipped

| Area | Detail |
|---|---|
| **Dungeon generation** | `dungeon/rooms.py` non-overlapping rect placement (`lax.fori_loop` + rejection sampling), `dungeon/corridors.py` L-shaped corridors, `dungeon/mazes.py` Kruskal perfect-maze, `dungeon/branches.py::generate_main_branch_l1` full pipeline + `traverse_stair` / `enter_branch`, `dungeon/level_memory.py` with cache+restore on enter/leave |
| **Action dispatch** | 8 cardinal+intercardinal moves, 8 run variants, stair up/down, wait. `jax.lax.switch` over 256-entry ASCII-keyed table; 21 of 121 actions wired. |
| **FOV** | Real Bresenham raycast in `fov.py`. Per-ray length bounded by `sight_radius`. Verified at blind / standard / dark. |
| **Observation projection (NLE-parity)** | `obs/nle_obs.py` projects `EnvState` into the 17-key NLE dict: glyphs (terrain + player overlay, fog-masked), blstats (all 27 fields), message buffer, tty char grid (24×80 with status line), chars/colors stubs marked TODO for Wave 3. |
| **Pixel observation** | `obs/pixel_obs.py` uses the existing `Nethax/tiles/tiles.npy` atlas; renders `(336, 1264, 3)` uint8 image; jit-compatible. |
| **RNG primitives** | `rng.py` `dice_roll`, `rnd`, `rn2`, `weighted_choice` — all real `jax.random` calls. |
| **Constants — glyphs** | All 13 `GLYPH_*_OFF` offsets and `MAX_GLYPH=5976` re-verified by reading `nle.nethack` from a live install. Wave 1 had estimates; Wave 2 has truth. |
| **Constants — monsters** | 390 entries across `constants/monster_entries/chunk{1..6}.py`, aggregated into `constants/monsters.py::MONSTERS`. Covers all canonical entries minus `#if 0` / `#ifdef CHARON` / `#ifdef MAIL_STRUCTURES` blocks. |
| **Constants — objects** | 503 entries across 9 chunk files in `constants/object_entries/`, deduped+aggregated into `constants/objects.py::OBJECTS`. Includes weapons (72), armor (87), rings (29), amulets (14), tools (51), food (33), potions (43), scrolls (35), spellbooks (44), wands (45), gems (37), rocks (3), generics (16), and specials (Amulet of Yendor, Candelabrum, Bell, Book of the Dead, Heavy Iron Ball, statue, boulder, venoms, coins). |
| **Legacy removal** | Deleted: `nethax_state.py`, `game_logic.py`, `renderer.py`, old `constants.py`, `world_gen/`, `envs/`, `util/`. Moved `play_nethax.py` → `scripts/legacy/play_nethax.py` for future rewrite against new `NethaxEnv`. `Item` moved into `subsystems/inventory.py`. |
| **Vendor parity tests** | New `tests/test_vendor_parity.py` imports `nle.nethack` from the installed package and asserts our `Action` codes, `BL_*` indices, and `GLYPH_*_OFF` constants match NLE byte-for-byte (gated by `@pytest.mark.skipif(not nle_installed)`). |
| **Tests** | 70 new tests across `test_dungeon_generation.py`, `test_env_lifecycle.py`, `test_fov.py`, `test_movement.py`, `test_obs_projection.py`, `test_pixel_obs.py`, `test_rng.py`, `test_state_invariants.py`, `test_vendor_parity.py`. Total now 173. |

---

## Doc set

| # | File | Covers |
|---|---|---|
| 1 | [`README.md`](README.md) | This file |
| 2 | [`mechanics-status.md`](mechanics-status.md) | Status of each Wave 1 stub after Wave 2 — what's real, what's still a no-op |
| 3 | [`data-tables.md`](data-tables.md) | Monster / object table coverage, dedup decisions, count vs canonical NLE |
| 4 | [`nle-parity.md`](nle-parity.md) | Updated parity status — what's verified live against NLE |
| 5 | [`migration.md`](migration.md) | Legacy code removal — what was deleted, what was moved, why |
| 6 | [`decisions.md`](decisions.md) | Design decisions made this wave + tradeoffs |
| 7 | [`gaps.md`](gaps.md) | Remaining TODOs after Wave 2 |
| 8 | [`test-results.md`](test-results.md) | Full pytest output and what each test verifies |
| 9 | [`next-wave.md`](next-wave.md) | Wave 3 preview + open questions |

---

## How to use Wave 2

```python
import jax
from Nethax.nethax import NethaxEnv
from Nethax.nethax.constants.actions import Action, CompassCardinalDirection

env = NethaxEnv()
state, obs = env.reset(jax.random.PRNGKey(0))

# A real dungeon now exists at state.terrain[0, 0]
import jax.numpy as jnp
print(jnp.unique(state.terrain[0, 0]))   # → [VOID, FLOOR, CORRIDOR, WALL, CLOSED_DOOR, STAIRCASE_UP, STAIRCASE_DOWN]

# Move east
state, obs, r, done, info = env.step(state, int(CompassCardinalDirection.E), jax.random.PRNGKey(1))

# obs['glyphs'] is real (terrain projected); obs['blstats'][BL_HP] is the player's HP.
# JIT works:
step_jit = jax.jit(env.step)
state, *_ = step_jit(state, int(CompassCardinalDirection.W), jax.random.PRNGKey(2))
```

---

## What's NOT in Wave 2 (deferred to later waves)

- **Monsters don't move yet** — Wave 4. State slot exists; AI step is still no-op.
- **Combat is still damage-free** — Wave 3. Player can bump monsters; nothing happens.
- **No items on the ground** — Wave 3. Inventory state pytree is in place; spawn / pickup mechanics are TODO.
- **No magic** — Wave 3. Spell table populated; cast is no-op.
- **No traps trigger** — Wave 3.
- **No status-effect ticks** — Wave 3 (hunger, encumbrance, intrinsics).
- **Single level only** — Wave 4. Stair traversal works *within* a branch but Mines / Sokoban / Quest / Gehennom branches are still empty.
- **Inventory rendering in tty** — Wave 3 (currently just status line is rendered).
- **Identification system not active** — Wave 3 (appearance shuffle, partial / full ID).

These map to the TODOs aggregated in [`gaps.md`](gaps.md).
