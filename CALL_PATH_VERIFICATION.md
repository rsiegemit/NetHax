# Wave-3 Helper Call-Path Verification

Audit date: 2026-05-27  
Branch: main  
Commits audited: 71d6519 (trap dead-predecessor), b18a098 (makemon deeper), 92cfa82 (TOOL+CONTAINER)

---

## Reachability Table

| Helper | Defined | Call site | Reachable from env.reset under use_vendor_rng()? |
|--------|---------|-----------|--------------------------------------------------|
| `_consume_makemon_post_hp_draws` | `spawning.py:940` | `spawning.py:1154` inside `spawn_initial_monsters` → called from `populate_level_with_monsters` → `env.py:318` | **YES** |
| `consume_mksobj_init_draws` (trap dead-predecessor path) | `random_objects.py:946` | `rooms.py:1809, 1823, 1832, 1867` inside `trap_step_isaac` → `lax.scan` at `rooms.py:1894` → `fill_one_isaac` → `fill_ordinary_rooms` → `branches.py:1054` → `env.reset` | **YES** |
| `consume_mksobj_init_draws` (mkobj_at loop path) | `random_objects.py:946` | `rooms.py:2090, 2103` inside `_mkobj_true` → `fill_one_isaac` → same chain | **YES** |
| `_mkbox_cnts_draws` | `random_objects.py:776` | Called by `_tool_chest_lbox_draws` (`random_objects.py:850`), `_tool_icebox_draws` (`random_objects.py:855`), `_tool_sack_draws` (`random_objects.py:867`) — all within `_tool_draws_dispatch` → `consume_mksobj_init_draws` (TOOL_CLASS path) | **CONDITIONAL — see note** |
| `trap_step_isaac` | `rooms.py:1770` (nested def) | `rooms.py:1895` via `lax.scan` inside `fill_one_isaac` | **YES** |

---

## Detailed Traces

### 1. `_consume_makemon_post_hp_draws`

```
env.reset (env.py:317-321)
  └─ populate_level_with_monsters(vendor_rng=state.vendor_rng)  [env.py:318]
       └─ spawn_initial_monsters(..., vendor_rng=vendor_rng)  [spawning.py:1268]
            └─ _spawn_one_v (fori_loop body)  [spawning.py:1141]
                 └─ _consume_makemon_post_hp_draws(vrng, type_id, ...)  [spawning.py:1154]
```

Gate: `if use_vendor_rng()` at `env.py:317` — fully guarded, fires on every monster in the loop.  
**Status: WIRED, REACHABLE.**

---

### 2. `consume_mksobj_init_draws` (trap dead-predecessor cascade — commit 71d6519)

```
env.reset (env.py:282-311)
  └─ generate_main_branch_l1_with_fill(vendor_rng=...)  [branches.py:1046-1058]
       └─ fill_ordinary_rooms(..., vendor_rng=vendor_rng)  [branches.py:1054]
            └─ lax.scan(fill_one_isaac, ...)  [rooms.py:2142]
                 └─ fill_one_isaac (per-room scan body)  [rooms.py:1716]
                      └─ lax.scan(trap_step_isaac, ...)  [rooms.py:1894]
                           └─ trap_step_isaac (per-trap scan body)  [rooms.py:1770]
                                └─ lax.cond(should_place & (depth_i <= rnd4_gate),
                                            _dead_pred_true, ...)  [rooms.py:1887]
                                     └─ consume_mksobj_init_draws(vi, item_cls)  [rooms.py:1809]
                                     └─ consume_mksobj_init_draws(vp, poss_cls)  [rooms.py:1823/1832]
                                     └─ consume_mksobj_init_draws(vic, TOOL_CLASS)  [rooms.py:1867]
```

Gate: `fill_one_isaac` branch is taken when `vendor_rng is not None` (`rooms.py:2134`).  
**Status: WIRED, REACHABLE.**

---

### 3. `_mkbox_cnts_draws` (commit 92cfa82)

`_mkbox_cnts_draws` is defined in `random_objects.py:776` and called only from:
- `_tool_chest_lbox_draws` (`random_objects.py:850`)
- `_tool_icebox_draws` (`random_objects.py:855`)
- `_tool_sack_draws` / bag helpers (`random_objects.py:867`)

These are dispatched via `_tool_draws_dispatch` → `consume_mksobj_init_draws` when:
1. `oclass_id == TOOL_CLASS` (6), AND
2. `otyp` is provided (not None)

**In `rooms.py` `_box_true` (box/chest placement, `rooms.py:1980-2010`):** The box spawn manually inlines 3 draws (`rn2(5)` locked, `rn2(10)` trapped, `rn2(100)` tknown) then runs a flat 8-iteration scan of `(rnd(100) class, rnd(100) type)` per item. It does **NOT** call `consume_mksobj_init_draws` or `_mkbox_cnts_draws` — the box content items get no per-item init cascade.

**In `rooms.py` `_mkobj_true` (mkobj_at loop, `rooms.py:2072-2112`):** Calls `consume_mksobj_init_draws(vrng_in, oclass0, otyp0)` with `otyp` present — so if `oclass0 == TOOL_CLASS` and `otyp0` is a container, `_mkbox_cnts_draws` **will** fire via `_tool_draws_dispatch`.

**Net reachability:**  
- Via the mkobj_at loop: **YES, conditional on TOOL_CLASS container being rolled.**  
- Via the box/chest placement path (`_box_true`): **NOT CALLED** — `_box_true` bypasses `consume_mksobj_init_draws` entirely and uses a hand-inlined approximation. `_mkbox_cnts_draws` is not reached on that path.

---

### 4. `trap_step_isaac` (commit 71d6519)

```
fill_one_isaac (rooms.py:1716)
  └─ lax.scan(trap_step_isaac, init_carry, jnp.arange(MAX_TRAPS_PER_ROOM))  [rooms.py:1894-1898]
```

`trap_step_isaac` is a nested def at `rooms.py:1770`, scanned unconditionally inside `fill_one_isaac`.  
`fill_one_isaac` is activated when `vendor_rng is not None` (`rooms.py:2134-2146`).  
**Status: WIRED, REACHABLE.**

---

## Missing Wires / Bugs

### BUG: `_box_true` bypasses `_mkbox_cnts_draws` (and `consume_mksobj_init_draws`)

**Location:** `rooms.py:1980-2010` (`_box_true` inside `fill_one_isaac`)

**What happens:** When a box/chest spawns (`!rn2(nroom*5/2)`), `_box_true` fires. It emits the 3 correct mksobj_init draws (`rn2(5)`, `rn2(10)`, `rn2(100)`) but then runs a simplified content loop that only draws `rnd(100)` class + `rnd(100)` type per item — **skipping the per-item `consume_mksobj_init_draws` cascade entirely**. The real vendor `mkbox_cnts` calls `mksobj_at(item, init=TRUE)` for each item, which fires the full mksobj_init branch per item (weapons get enchantment rolls, food gets rotten flag, wands get charge count, etc.).

**Impact:** Every game that spawns a box/chest on level 1 will diverge from the ISAAC64 stream from that point onward. This is a likely source of multiple divergences in the 66-count baseline.

**Proposed fix:** Replace the hand-inlined `_box_item_step` loop in `_box_true` with a call to the existing `_tool_chest_lbox_draws` (or call `consume_mksobj_init_draws` with `TOOL_CLASS` and the decoded otyp). Specifically:

```python
# After the 3 mksobj_init draws (locked/trapped/tknown), replace the
# hand-rolled item loop with:
box_otyp = jnp.where(_box_type == jnp.int32(0), jnp.int32(190), jnp.int32(189))
# _tool_chest_lbox_draws already emits rn2(5)+rn2(10)+mkbox_cnts;
# since we already emitted those 3 draws above, call _mkbox_cnts_draws directly:
from Nethax.nethax.subsystems.random_objects import _mkbox_cnts_draws
vrng_in = _mkbox_cnts_draws(vrng_in, box_otyp)
```

Note: `_mkbox_cnts_draws` is not currently imported in `rooms.py` — it would need to be added to the import from `random_objects`.

---

## Summary

| Helper | Reachable? |
|--------|-----------|
| `_consume_makemon_post_hp_draws` | YES |
| `consume_mksobj_init_draws` (trap cascade, commit 71d6519) | YES |
| `trap_step_isaac` (commit 71d6519) | YES |
| `_mkbox_cnts_draws` (commit 92cfa82, via mkobj_at loop) | CONDITIONAL (only if TOOL_CLASS container rolled in mkobj_at loop) |
| `_mkbox_cnts_draws` (commit 92cfa82, via box/chest placement) | **NOT REACHED** — `_box_true` uses a hand-inlined approximation that skips the full cascade |

**Root cause of remaining divergences:** `_box_true` in `fill_one_isaac` does not call `consume_mksobj_init_draws` / `_mkbox_cnts_draws` for box contents, causing stream divergence whenever a box/chest is placed.
