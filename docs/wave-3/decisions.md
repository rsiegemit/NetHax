# Wave 3 — Design Decisions

## 1. Full drop-in NLE replacement, not MiniHack-first

After the data-table population pass surfaced the ~150-effect work involved in completing all NetHack mechanics, we briefly considered pivoting to "MiniHack-first" — wiring just enough mechanics for the 170 MiniHack tasks. The user chose to stay on the original plan: full NetHack fidelity, then add lightweight "presets" later.

**Tradeoff:**
- ✅ Anyone using NLE today can swap in `NethaxEnv` and keep training.
- ✅ Settings/presets at a higher layer can constrain features for narrower tasks.
- ❌ Slower path to the RL community's most-used benchmarks (MiniHack).
- Wave 4 puts MiniHack first now — it's the next deliverable.

## 2. Per-effect JAX functions, dispatched via `lax.switch`

Each potion/scroll/wand effect is its own pure function `(state, rng, buc) → state`. The top-level `quaff_potion`/`read_scroll`/`zap_wand` does:

```python
effect_id = item.type_id - BASE_OFFSET
state, rng = lax.switch(effect_id, _SWITCH_BRANCHES, state, rng, buc)
```

`_SWITCH_BRANCHES` is a tuple of small lambdas, one per effect. This compiles to a single XLA branch table — fast, no Python overhead at runtime.

**Tradeoff:** every effect must accept the same operand tuple shape. Effects that need extra context (e.g., direction for wands) carry it inside the operand. Awkward but JIT-friendly.

## 3. Spell vs wand effect sharing

`MAGIC_MISSILE` exists as both a spell and a wand. In vendor C, they share `bhit()` dispatch. In our code, they share via `cast_ray` helper functions imported from `items_wands.py`. This avoids duplicating the ~6 ray-effect implementations.

## 4. `inv_strs` full fidelity = static byte tables

The user explicitly chose full NLE fidelity over simplified inv strings. To make this JIT-compatible, we precompute static byte tables at module load:

- `_OBJECT_NAMES_BYTES[NUM_OBJECTS, 40]` — canonical names per object
- `_APPEARANCE_BYTES[NUM_OBJECTS, 30]` — appearance descriptions
- `_BUC_BYTES[4, 10]`, `_EQUIP_BYTES[8, 24]`, `_CLASS_PREFIX_BYTES[18, 14]`, `_CLASS_NOUN_BYTES[18, 12]`

Then `render_slot(state, slot_idx)` reads these tables inside JIT and assembles the 80-byte buffer via `lax.fori_loop` over each component (letter, quantity, BUC, enchant, name, equip status, charges).

**Simplifications kept:**
- Always uses "a" article (no vowel check for "an")
- Skips user-given names (no "named Sting" support)
- Plural is just appended 's' (no "knives" / "men" irregulars)
- Two-weapon "alternate weapon" status omitted

## 5. Depth-curve monster spawning via static table

`MONSTR_DIFFICULTIES: jnp.ndarray[381] int32` is computed once at import time from the canonical `MONSTERS` tuple. `eligible_monsters_for_depth(depth)` is a vectorized mask. `pick_monster_for_level(rng, depth)` uses `jax.random.choice` weighted by `gen_freq`.

**Simplification:** difficulty is just `entry.level` in Wave 3. The full `monstr[]` formula (level + speed_bonus + attk_count + breath_bonus + petrify_bonus) is Wave 5. For now, low-level dungeons get appropriately weak monsters, which is what matters for early-game RL.

## 6. Monster movement = greedy 8-dir, no LoS

Wave 3 monster AI doesn't compute its own FOV. It just steps toward `player_pos` greedily if within range. This gives wrong behavior in maze-like dungeons (monsters teleport through walls cognitively) but is fine for Wave 3's open-room test cases.

**Wave 4:** proper monster FOV + path-aware target selection.

## 7. Run-length cap on agents = bounded scan

Several effects (rays, scans, runs) use `lax.scan` or `lax.while_loop` with hard caps:

| Loop | Cap | Reason |
|---|---|---|
| `_run` movement | 64 steps | longest corridor ~80; 64 covers most |
| Wand ray | 8 tiles | NetHack's default ray range |
| FOV ray | sight_radius | bounded per-call |
| `lax.scan` over monsters | 200 | `MAX_MONSTERS_PER_LEVEL` |
| `lax.scan` over inventory | 52 | `MAX_INVENTORY_SLOTS` |

All bounds are compile-time constants for JIT correctness.

## 8. Action dispatch deferred to integration

Each Wave 3 subsystem exposes `handle_<action>(state, rng) -> state` functions, but the actual wiring of these into `_HANDLERS` and `_ACTION_TO_HANDLER_IDX` in `action_dispatch.py` is **deferred to Wave 4 integration**. The reason: 12 agents can't safely all edit the same dispatch table in parallel without merge conflicts. Each agent left their handler exposed; Wave 4 integrator wires them.

Right now, calling `env.step(state, action=ord('q'), rng)` for "quaff" runs the no-op handler — but `from Nethax.nethax.subsystems.items_potions import handle_quaff; handle_quaff(state, rng)` works directly. Tests use the direct form.

## 9. Status effect ticks run inside `subsystems/status_effects.step()`

The status step is called from `env.step()` (Wave 4 will wire this; currently the test that triggers status ticks calls it directly). The order matters:

1. lethal-expiry checks (strangulation/stoning/sliming/food poisoning)
2. tick_timers (decrement all timed statuses)
3. hunger_tick
4. hp_regen_tick
5. pw_regen_tick
6. apply_starvation

Each is its own pure function for testability.

## 10. Spawning writes use `lax.fori_loop` + scatter

`populate_level_with_monsters(state, rng, n=5)`:

1. Build valid-tile mask from `state.terrain[branch, level-1]`
2. `lax.fori_loop` over n slots, calling `spawn_initial_monsters` which uses inner `lax.scan` for the HP roll
3. Write 5 monsters into `state.monster_ai.{pos, hp, alive, ...}` via `lax.fori_loop` + `.at[i].set(...)`

This pattern (outer fori_loop for "n entities", inner scan/cond for entity construction) recurs throughout Wave 3.

## 11. Stat range table normalization

`_ROLE_STAT_RANGES[VALKYRIE]` had `(min=7, max=6)` for INT (negative range). The vendor `roles[]` table uses `attrdist` as a signed delta and `attrbase` as the minimum, which our agent didn't unpack correctly. Rather than re-derive every role/race from `role.c`, we wrote `_normalize_ranges()` to swap any inverted pair so `min <= max`. Documented as Wave 6 polish (full role.c port).

## 12. Test schema patches over source rewrites

When 8 test files had stale `Item(category=..., type_id=..., ...)` constructors missing the new 3 fields, the choice was:
1. Rewrite each test to use `make_item()` helper
2. Add defaults to Item fields (impossible with Flax struct.dataclass)
3. Patch every `Item(...)` block to add the 3 fields

We chose (3) via an automated regex pass. Faster and less invasive. Wave 4+ tests should use `make_item()` from the start.
