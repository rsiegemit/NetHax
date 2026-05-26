# NLE Byte-Parity Validator — First Run Report

Output from `tests/test_nle_byte_parity.py` after 1 step against
**NLE 1.3.0** with character `rog-hum-cha-mal`, seed=0.

## Divergence catalog (15 total)

### 1. Reset obs encoding mismatch (1 divergence, step 0)

NLE's `env.reset()` does not populate the dict the same way as
`env.step()`; the validator currently sees an empty dict at step 0.

**Action**: change validator to compare step 1+ only. (Filtered in v2.)

### 2. NLE-internal channels (3 channels)

`internal`, `misc`, `program_state` — Nethax emits these but NLE does
not include them in its default `observation_keys`.  **Not a bug**;
these are Nethax superset channels.  Filtered from comparison in v2.

### 3. Starting dungeon differs (1604/1659 = 96.7% glyphs)

**Root cause**: RNG mismatch.  Nethax uses Threefry for
dungeon-generation seeding; NLE uses ISAAC64 (vendor_rng).  Even with
the same integer seed, the two RNGs produce completely different
sequences → different room layouts, monster placements, item drops.

**Fix**: wire `Nethax.nethax.vendor_rng.Isaac64State` into
`Nethax.nethax.env.NethaxEnv.reset()` and replace `jax.random.PRNGKey`
seeding throughout `dungeon/branches.py`.  This is the largest single
parity gap.

### 4. blstats column 0 (player_x): NLE=3, Nethax=56

Different starting tile — consequence of (3) above; not a separate bug.

### 5. blstats columns 2/3 (STR25/STR125): NLE=14, Nethax=18

**Root cause**: Nethax defaults to `player_str=18` in
`state.py::EnvState.default()`; NLE rolls character stats per role
table.  For Rogue, NLE's `u_init` sets STR=14 (per vendor `attrib.c::A_STAT`
for PM_ROGUE).

**Fix**: populate starting attributes from
`vendor/nethack/src/attrib.c::A_STAT` table per role, not hardcoded 18.

### 6. chars/colors: 2.2% / 4.9% diverge at column ≥ 80

The first 79 columns (the map proper) diverge with the dungeon
(consequence of #3).  Columns 80+ are the status line — those diverge
because of #5 (different stats render different status text).

### 7. inv_letters all-zero in Nethax (NLE=[a,b,c,d,'e'])

**Root cause**: `state.inventory` doesn't track per-slot letters.
NLE assigns letters at game start so policies can reference items as
`'a', 'b', ...`.

**Fix**: add `inv_letters: jax.Array  # int8[55]` to `InventoryState`,
populate at character init with `'a'..'e'` for starting kit, and
maintain on pickup/drop (per vendor `invent.c::assigninvlet`).  Then
`build_inv_letters` reads from there.

### 8. inv_oclasses: NLE=[6,...] Nethax=[18,...]

NLE shows item class 6 (TOOL_CLASS — saddle? sack?), Nethax shows 18
(MAXOCLASSES = empty sentinel).

**Root cause**: Nethax doesn't populate starting inventory.  NLE does
(via vendor `u_init.c::u_init`).

**Fix**: populate `inventory.items` slots 0..4 at character init per
vendor role-specific starting kit.  For Rogue: lock-pick, dagger,
short sword, sling, etc.

### 9. inv_glyphs: 5/55 (9.1%) diverge

Consequence of #8 — Nethax has empty slots, NLE has items.

## Summary

**Real bugs (3)**: starting attributes (#5), inv_letters never
populated (#7), starting inventory missing (#8).

**Consequence of RNG mismatch (5)**: items #3, #4, #6 (chars/colors), #9.

**Comparison-tool artifacts (2)**: #1, #2.

## Priority order

1. **Wire `vendor_rng` (ISAAC64) into env seeding** — eliminates ~80% of
   per-step byte divergences by aligning dungeon + monster spawns.
2. **Populate starting kit + attributes from vendor `u_init.c`** —
   eliminates inv_oclasses + inv_glyphs + blstats divergences.
3. **Wire `inv_letters` into `InventoryState`** — fixes inv_letters all-zero.
4. **Filter NLE-internal channels in validator** (DONE in v2).

After (1)-(3), the validator should drop from 15→0 divergences (modulo
any further hidden encoding issues).
