# Brax-style flat-HLO integration

NetHax's `monster_turn`, `dispatch_action`, and ~20 subsystem entry points
have alternative "Brax-style" implementations that replace `lax.cond` and
`lax.switch` chains with `jnp.where` masking — both branches always
compute, the mask selects.

## Why

`lax.cond` under `vmap` lowers to `lax.select` with both branches inlined
into HLO. `lax.switch` does the same for every case. For graphs with
hundreds of nested conds (e.g. the 46-branch action dispatcher + the
2600-LOC monster_turn body), this produces an HLO module so large that
XLA's optimization passes do not finish on H100 in 6 hours.

The Brax pattern flattens this: every branch always computes, results are
selected by `jnp.where` on a mask. HLO grows linearly in branch count
instead of geometrically. Brax (Google's JAX physics engine), Craftax
(JAX NetHack-inspired env), and JaxMARL all use variants of this pattern.

## Environment variables

| Variable | Effect | Default |
|---|---|---|
| `NETHAX_BRAX_ALL=1` | Route subsystem callers through `*_brax` versions via PEP 562 module `__getattr__`. Includes monster phase, action dispatch, all item handlers, magic, features, combat, inventory. | `0` (use originals) |
| `NETHAX_CRAFTAX_SCAN=1` | Use `jax.lax.scan` over `fmon_order` with `(state, rng)` carry for the byte-parity monster loop, instead of the Fix-1 Python for-loop (which unrolls to 400× HLO under vmap). | `0` (Python loop) |
| `NETHAX_PHASED_ORCH=0` | Use the pre-surgery monolithic per-slot body (all of `monster_turn` in one jit) instead of the 8-phase orchestrator. | `1` (phased) |

For the validator (byte-parity verification on GPU), the recommended
combination is:

```
NETHAX_BRAX_ALL=1 NETHAX_CRAFTAX_SCAN=1 NETHAX_PHASED_ORCH=0
```

## Files

Each `*_brax.py` file lives next to its original in
`Nethax/nethax/subsystems/`. The 23 files cover:

- **Monster phase:** `monster_turn_brax.py`, `monster_attack_player_brax.py`,
  `pet_move_brax.py`, `monster_turn_brax.py` (combat helpers split into
  `combat_helpers_brax.py` + `mattackm_brax.py`), `use_cast_brax.py`
  (monster_use_item + monster_cast_spell), `postmov_brax.py`,
  `pathfind_step_brax.py`, `preloop_brax.py` (were/summon/qsteal/covetous +
  monster_can_see_player), `scan_bodies_brax.py`, `post_monster_brax.py`,
  `pre_monster_brax.py`.
- **Player action dispatch:** `dispatch_action_brax.py` (46-way switch
  flattened), `movement_brax.py` (_try_step / _move_branch), `handlers_brax.py`
  (identity re-exports).
- **Item & magic subsystems:** `items_dispatch_brax.py` (potions + scrolls),
  `wands_books_brax.py`, `items_misc_brax.py` (jewelry/corpses/items),
  `magic_brax.py`, `features_brax.py`, `inventory_brax.py`, `combat_brax.py`,
  `status_effects_brax.py`.

## Rollback

Set `NETHAX_BRAX_ALL=0` (or unset) to fall back to the original (lax.cond)
implementations. The `*_brax.py` files are never deleted; the PEP 562
`__getattr__` is only installed when the env var is set.

## Validation

Mac eager-mode `multiseed_byteparity.py` passes 10/10 with all four
integration rounds enabled. H100 cold-compile in progress at commit time.

## Trade-offs

- **HLO compile time:** ~10-100× smaller graph; expected to compile in
  single-digit minutes on H100 vs. >6h TIMEOUT pre-refactor.
- **Runtime FLOPs:** ~5-10× more arithmetic per env-step (every branch
  always computes, even if masked away). Acceptable for byte-parity
  validation; for RL training throughput, originals remain accessible.
- **Memory peak during compile:** also reduced (smaller HLO module).
- **Byte parity:** preserved exactly (RNG draw order is byte-identical;
  every conditional write replaced with `jnp.where`-masked equivalent).

## Research lineage

The Brax pattern was identified as the only viable architecture for our
constraints (byte-deterministic vendor ISAAC64 RNG + 400 monster slots +
nested per-monster AI) by two independent research agents on 2026-06-09.
JaxMARL's "group by agent class" pattern is incompatible with our byte-
parity requirement (reorders slot processing → desyncs vendor_rng).
Craftax's `update_mobs` pattern (`lax.scan` over slot indices with
`(state, rng)` carry) is preserved by `NETHAX_CRAFTAX_SCAN=1`.

## Memory notes (for future sessions)

- `project_craftax_pattern_2026_06_09.md` — root cause: `lax.cond` under
  vmap inlines both branches; need `lax.scan` outer + `jnp.where` inner.
- `project_h100_compile_redesign_2026_06_09.md` — XLA flag tweaks
  (`algsimp`, `cse`, etc.) are not the answer; architecture is.
- `project_fix1_h100_compile_2026_06_07.md` — Fix 1 (scan → Python loop)
  was the WRONG direction; reversed by the Craftax scan path.
