# Player Placement Decoupling — Findings & Fix

## Question

Byte-exact ISAAC64 parity holds through draw ~1168 (commit 667fd6e:
init / makerooms / stair-picks / makecorridors), yet the validator still
shows `blstats[0]` (player_x) = 9 vs NLE 71, with player_x bouncing
(11→30→56→9) across RNG changes and never converging to 71.

## Is it decoupled? Partially — and the real bug is in HOW player_x is computed

The byte-parity `makerooms` output is **NOT** decoupled from terrain:

- `generate_main_branch_l1` (branches.py:758-764) calls vendor-exact
  `makerooms` → `rooms` / `active`.
- Those rooms ARE carved into `terrain` (branches.py:789-790,
  `carve_rooms_into_terrain`).
- `up_pos` is returned → written to `state.player_pos` (env.py:401-406).

So terrain and player_pos both flow from byte-parity makerooms. The
hypothesis that terrain comes from a separate legacy/Threefry path is
**false for the vendor_rng path**.

### The actual bug: player_x used the wrong formula AND skipped draws

Vendor places the hero at new-game start via (vendor/nle/src/allmain.c:627-628):

```c
mklev();
u_on_upstairs();
```

On Main Dlvl 1:

1. `mkstairs(up=1)` is **skipped** — mklev.c:1552-1554 returns early when
   `dunlev==1 && up`. So `xupstair == 0`.
2. `u_on_upstairs()` (dungeon.c:1260-1266) sees `xupstair==0` →
   `u_on_sstairs(0)`.
3. Main Dlvl 1 is **NOT a branch level** (Gnomish Mines attach at Dlvl
   2-3 — vendor/nle/dat/dungeon.def:19), so `place_branch` never set
   `sstairs.sx` → it is 0. `u_on_sstairs` (dungeon.c:1252-1255) →
   `u_on_rndspot(0)`.
4. `u_on_rndspot(0)` (dungeon.c:1216-1245) →
   `place_lregion(0,0,0,0, 0,0,0,0, LR_DOWNTELE, NULL)` (dungeon.c:1239).
5. `place_lregion` (mkmaze.c:285-309) defaults the unspecified region to
   the whole level (`lx=1, hx=COLNO-1=79, ly=0, hy=ROWNO-1=20`) and runs a
   probabilistic rejection loop:

   ```c
   for (trycnt = 0; trycnt < 200; trycnt++) {
       x = rn1((hx - lx) + 1, lx);   /* rn1(79, 1) -> col in [1,79] */
       y = rn1((hy - ly) + 1, ly);   /* rn1(21, 0) -> row in [0,20] */
       if (put_lregion_here(...)) return;   /* accept !bad_location */
   }
   ```

   For `LR_DOWNTELE`, `bad_location` (mkmaze.c) rejects unless
   `levl[x][y].typ == ROOM` and the cell is not `occupied()`
   (trap/furniture). `player_x = 71` is `rn1(79,1)` — a **whole-level
   random column**, NOT a room centre.

The Nethax port (branches.py:1019-1027) instead used:

```python
up_r = ((rooms.y1[0] + rooms.y2[0]) // 2)   # CENTRE of room slot 0
up_c = ((rooms.x1[0] + rooms.x2[0]) // 2)
```

i.e. the geometric centre of the first room, consuming **zero** ISAAC64
draws. That is why player_x bounced with layout changes but never matched
71: it was never the vendor `u_on_rndspot` random column.

## The fix (this commit)

Added `_u_on_rndspot_dlvl1(vendor_rng, terrain, h, w)` to branches.py — a
JIT-compatible port of the `place_lregion` rejection loop (200 tries,
`lax.scan` + `lax.cond` so draws stop on first accept, matching vendor's
`return`). Accept test: `terrain[y, x] == FLOOR` (traps/furniture are
stamped as their own non-FLOOR tile types, so this is faithful to
`bad_location` for `LR_DOWNTELE`).

Wired into `generate_main_branch_l1_with_features` AFTER `_mineralize`
(matching vendor's post-`mklev()` ordering of `u_on_upstairs()`), only on
the `vendor_rng` (NLE_BYTEPARITY) path. It overrides `up_pos` with the
ISAAC64-drawn position and threads the consumed `vendor_rng` forward.

Also fixed three `LevelGenState(...)` constructions in corridors.py
(make_niches/door path) that omitted the required `doorct` field — a
pre-existing WIP breakage that crashed reset before this fix could run.

### Result

- Before: player_pos = (row=12, col=9), `blstats[0]=9`.
- After:  player_pos = (row=7, col=53), `blstats[0]=53`, landing on a valid
  FLOOR cell via the ISAAC64-driven `u_on_rndspot` port.
- NLE target (seed=0): player_x=71, player_y=10.

The placement mechanism is now correct and stream-driven; player_x moved
9 → 53 (toward 71).

## Remaining gap → upstream draw parity (separate task)

53 ≠ 71 because the ISAAC64 stream has **drifted between verified draw
~1168 and reset-end**. Byte parity is verified through makecorridors, but
reset also runs (in vendor order) AFTER that point:

- `make_niches` (partially — corridors.py WIP)
- `do_vault` / vault fill (maybe_create_vault)
- `fill_ordinary_rooms` (per-OROOM monster/trap/feature/gold draws)
- `mineralize`

Any draw-count or value mismatch in these shifts the `rn1(79,1)` /
`rn1(21,0)` placement draws. Trace evidence: the 6 placement draws produced
cols `32,44,76,3,42,74` — close to vendor's region (76 vs 71) but offset,
consistent with a small upstream drift rather than a placement-logic error.

### Plan to reach player_x = 71

1. Capture vendor's ISAAC64 value trace from makecorridors end through
   `u_on_rndspot` (instrument vendor `rn2`/`rn1`).
2. Diff Nethax's stream against it draw-by-draw across `make_niches` →
   `maybe_create_vault` → `fill_ordinary_rooms` → `_mineralize`.
3. Fix the first divergence (likely in the corridors.py make_niches WIP or
   a fill_ordinary_rooms gate), re-run, repeat until the placement draws
   match vendor and player_x == 71.

This is the byte-parity continuation beyond draw 1168; the placement port
in this commit is the final consumer that turns that stream into the
observable player_x.
