# Nethax ISAAC64 Draw Count — env.reset() NLE_BYTEPARITY

## Summary

| Metric | Count |
|--------|-------|
| **Nethax host-side draws** (this measurement) | **107** |
| Prior baseline | 107 |
| Vendor NLE total | 1789 |
| Ratio gap (vendor / Nethax host-side) | **16.7×** |

Seed: `jax.random.PRNGKey(0)`, role=Rogue, race=Human, alignment=chaotic.

## Method

Added `_DRAW_COUNT` global to `vendor_rng.py`, incremented unconditionally
in `_trace_isaac()` (which fires on every host-side ISAAC64 uint64 draw).
JAX-traced calls (`rn2_jax`, `rn1_jax`, `rn2_jax` inside `lax.fori_loop`, etc.)
bypass `_trace_isaac` and are **not** counted by this counter.

## Phase Breakdown (host-side draws only)

| Phase | Cumulative | Delta |
|-------|-----------|-------|
| `vrng.init()` | 0 | 0 |
| `compute_descr_shuffle()` (JAX-traced) | 0 | 0 |
| 3× `_vendor_draw_prngkey` | 3 | +3 |
| `create_character` (attribs + inventory) | 57 | +54 |
| `consume_init_dungeons_draws` (18 fixed) | 75 | +18 |
| `consume_init_dungeons_variable_draws` | **107** | +32 |

## JAX-Traced Draws (not in host count)

`compute_descr_shuffle` runs entirely via `rn2_jax` inside `lax.fori_loop`
and consumes **195** ISAAC64 draws that are invisible to the host counter.

Remaining gap vs vendor (1789 − 107 − 195 = **1487**) is attributable to:
- Dungeon-gen room/corridor/feature draws inside `generate_main_branch_l1_with_features`
  (`rn2_jax`, `rn1_jax`, `randint_jax` calls in `branches.py`, `rooms.py`, `spawning.py`)
- Monster HP rolls (`d_py` / `newmonhp`) and placement draws

## Conclusion

Nethax host-side draw count (107) matches the prior baseline exactly —
no regression. The 16.7× ratio vs vendor reflects the large fraction of
draws already ported to JAX-traced paths (`compute_descr_shuffle` alone
adds 195 JAX draws). Full vendor parity would require tracing all dungeon-gen
draws through the ISAAC64 stream rather than Threefry.
