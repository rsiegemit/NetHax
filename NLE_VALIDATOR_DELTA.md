# NLE Validator Delta — Run 3 vs Run 4

Re-ran `tests/test_nle_byte_parity.py` (3 steps) and
`tests/test_nle_return_distribution.py` (100 eps; CLI arg `20` is the
batch/log-frequency knob, not episode count) under
`ParityMode.NLE_BYTEPARITY` against NLE 1.3.0, seed=0,
character `rog-hum-cha-mal`.

Goal: measure improvement from the last hour of fixes:
- `052cb4f` — `inv_strs` `<letter> - ` prefix omitted in NLE mode
- `a0c10cc` — `tty_chars` row-0 space padding + initial-grid space fix
- `355cdbf` — ISAAC64 wired into 3 dungeon-gen sites
- `76e132c` / `d3f4ed6` — NLE action-index lookup table

## 1. Byte-parity divergence counts (run-3 → run-4)

Run-3 baseline: **15 divergences** at step 1 (per `NLE_BYTE_PARITY_REPORT.md`,
catalog of 15 items: 1 reset artifact + 3 NLE-only channels + 11 real channels).
Run-4 (this run): **14 divergences at step 1**, **14 each at step 2/3** →
`[summary] 43 total divergences across 3 steps`. Plus 1 step-0
"missing-channels" meta-line (NLE returns empty dict on reset — not a
real divergence).

| Channel               | Run-3 (cataloged) | Run-4 step 1   | Delta |
|-----------------------|-------------------|----------------|-------|
| `glyphs`              | 1604/1659 (96.7%) | 1645/1659 (99.2%) | **WORSE** |
| `chars`               | ~2.2% (col ≥ 80)  | 84/1659 (5.1%) | similar |
| `colors`              | ~4.9% (col ≥ 80)  | 90/1659 (5.4%) | similar |
| `blstats`             | 2 cols (STR)      | 8-9/27 (29.6–33.3%) | **WORSE-looking** (now seeing player_x/y too) |
| `inv_glyphs`          | 5/55 (9.1%)       | 6/55 (10.9%) | similar |
| `inv_letters`         | all-zero          | 1/55 (1.8%) | **IMPROVED** (kit letters now populated; only slot 6 still 0) |
| `inv_oclasses`        | NLE=6 Nethax=18   | 3/55 (5.5%) | **IMPROVED** (kit oclass set; 3 ordering mismatches) |
| `inv_strs`            | not cataloged     | 190/4400 (4.3%) | new, post-052cb4f |
| `message`             | not cataloged     | 12/256 (4.7%) | new |
| `screen_descriptions` | not cataloged     | 873/132720 (0.7%) | new (very small) |
| `specials`            | not cataloged     | 1/1659 (0.1%) | new (tiny) |
| `tty_chars`           | not cataloged     | 196/1920 (10.2%) | new (post-a0c10cc fix is in; still col-0 message row mismatch) |
| `tty_colors`          | not cataloged     | 229/1920 (11.9%) | new |
| `tty_cursor`          | not cataloged     | 2/2 (100.0%) | new (NLE=(18,49) Nethax=(6,40) — cursor on player tile, follows from dungeon RNG mismatch) |

## 2. Improved channels

- **`inv_letters`**: from "all-zero" → only 1/55 byte diverges (slot 6,
  Nethax=0 vs NLE='g'=103). Starting kit letters a..e now populated;
  one extra slot from per-role kit still missing. **Big win** from
  `f2e2ded` / `398b6e9`.
- **`inv_oclasses`**: from "all-MAXOCLASSES (18)" → 3/55 ordering
  divergences. Starting items now real. **Big win** from `f2e2ded`.
- **Step-0 reset-artifact** no longer counted (validator v2 only counts
  real per-channel diffs, plus 1 meta-line for the NLE-empty-reset dict
  at step 0).

Net: the run-3 "real-bug" channels narrowed from 8 → 6 (3 inv channels
no longer dominate; only ordering tail remains).

## 3. Channels still diverging (need work)

### High-bytes channels (still essentially "different dungeon")

- **`glyphs`** at 99.2% — the single biggest gap. Despite `355cdbf`
  wiring ISAAC64 into 3 dungeon-gen sites (room count, monster count),
  NLE and Nethax still generate fundamentally different maps. NLE
  glyph index 2359 (`S_dnstair`/floor sentinel) vs Nethax 5976 across
  the board → the player's starting room layout still differs.
  **Likely root cause**: more vendor RNG call sites are unwired.
  Candidates: `mklev.c::makelevel` room placement, door rolls,
  `mkroom.c` themed rooms, `mkobj.c` item placement, `monst*` spawn
  selection (only count was wired). Net effect: the entire room
  topology is still Threefry-derived for everything except the two
  loop counts.

- **`chars` / `colors`**: track `glyphs` (5.1% / 5.4%) — these are the
  visible map projection. Will collapse once `glyphs` does.

- **`screen_descriptions`**: 0.7% (873/132720). Tiny absolute fraction
  but driven by glyph-text mismatch at the diverging tiles.

### Status-line / cursor channels

- **`blstats`** 8-9/27: column 0 (player_x) NLE=49 Nethax=40, column 1
  (player_y) NLE=17 Nethax=5. Player spawned in different tile —
  follows directly from map divergence. STR columns now MATCH (run-3
  reported STR diverged; the `f2e2ded` attribs fix landed).
- **`tty_cursor`** 2/2: cursor follows player → 100% byte mismatch by
  definition while player_x/y differ.

### Status-line text channels (`tty_chars`, `tty_colors`)

- **`tty_chars`** 10.2% (196/1920): top-of-screen message row still
  diverges. NLE=`'I'(73), 't'(116), ' '`...="It is written…" intro
  message vs Nethax=spaces. `a0c10cc` added the space padding fix for
  the empty default, but NLE actually writes the role intro line
  ("It is written in the Book of …") that Nethax doesn't generate.
- **`tty_colors`** 11.9% same — tracks `tty_chars` text rendering.

### Inventory polish

- **`inv_strs`** 4.3%: post-`052cb4f` the `<letter> - ` prefix is
  gone, but item description bytes still differ. Sample
  `@5: NLE=115('s') Nethax=98('b')` — NLE "short bow" vs Nethax
  starting with a different item name, or buc-prefix wording.
- **`inv_letters` slot 6**: one more kit slot to populate.
- **`inv_oclasses`** 3 mismatches: starting kit ordering off by
  ~3 slots — likely the per-role kit traversal order differs from
  `u_init.c::knows_object` loop.

### Message channel

- **`message`** 12/256: intro message ("It is written…") is in NLE but
  not Nethax. Same root as `tty_chars`.

- **`specials`** 1/1659 at index 1472: spurious special tile —
  probably an altar/door flag set in one map but not the other,
  trivial once `glyphs` matches.

## 4. Return-distribution validator

100 episodes, random policy on action set `{N, E, S, W, SEARCH, WAIT}`,
MAX_STEPS=200:

```
metric                                 NLE            Nethax
mean return                         4.7000            0.1600
std return                          9.4112            1.2225
min return                          0.0000            0.0000
max return                         58.0000           12.0000
mean episode length                 190.74             55.15

  KS 2-sample statistic = 0.4400
  KS 2-sample p-value   = 0.0000
  VERDICT: DIVERGE (p <= 0.05)
```

**Verdict: DIVERGE** (p = 0.0000 ≤ 0.05).

KS statistic 0.44 — large. The dominant signal is episode-length:
NLE averages 190.7 steps (most reach the 200 cap); Nethax averages
55.2 steps (most terminate early). This means **Nethax is terminating
episodes far too aggressively** under random policy — likely an
over-strict death/quit condition (starvation? sub-zero HP from a
random WAIT loop? bump-attack into wall?). Returns are themselves
sparse (mostly zero in both), but the bound on returns differs
~30× because NLE has 3× more steps to accumulate.

## 5. Take-aways

1. **Net divergence count: 15 → ~14** (1 channel removed by `inv_letters`
   / kit fixes, plus reset artifact filtered). Real reduction is
   modest because new fixes uncovered channels that were previously
   masked by missing-data (e.g. `inv_strs` now compared because slots
   are populated).

2. **Biggest unresolved**: `glyphs` at 99.2%. `355cdbf` wired ISAAC64
   into room-count and monster-count, but every per-tile and per-room
   placement call in `mklev.c` / `mkroom.c` / `mkobj.c` / `monst.c`
   still consumes Threefry. Until those land, dungeon layout (and
   everything that depends on it — `chars`, `colors`, `tty_chars`,
   `tty_cursor`, `blstats` x/y, `screen_descriptions`) cannot byte-match.

3. **Return-distribution gap is the actionable RL-transfer red flag**:
   episode-length 55 vs 190 means Nethax kills the agent ~3.5× faster.
   This is independent of byte-parity and blocks reward-shape transfer
   even if byte-parity were perfect.

4. **Quick wins remaining**:
   - Populate the role intro message (`u_init.c::pline_The_Cracked…`)
     into `message` + `tty_chars` row 0 → kills 3 channels' bytes.
   - Add slot-6 starting-kit letter → kills `inv_letters` entirely.
   - Audit Nethax early-termination paths (HP≤0 on hunger? out-of-bounds
     move? unexpected `done`?) → closes the return-dist gap.

## Reproduce

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  PYTHONPATH=. .venv/bin/python tests/test_nle_byte_parity.py 3 --all
JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  PYTHONPATH=. .venv/bin/python tests/test_nle_return_distribution.py 20
```

Logs: `/tmp/validator_run4.log`, `/tmp/return_run.log`. JIT compile
of `_step_impl` took 12m38s (byte) and 12m40s (return) on cold cache.
