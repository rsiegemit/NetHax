# seed=5 byte-parity divergence diagnosis (rog-hum-cha-mal)

## Baseline (before this fix)
First ISAAC64 mod-divergence at draw index **542** in
`.test_runs/vendor_seed5_ops.trace`:

```
vendor: 542 rn2 mod=4 res=3  caller=0x10b37e874   (finddpos y rn1 — tt strip)
nethax: 542 rn2 mod=3 res=0  caller=corridors.py:555:finddpos
```

All 0..541 draws agreed exactly (op + mod + res).

## Bug #1 — Nethax stable sort vs vendor unstable qsort (FIXED)

Vendor `sort_rooms()` (mklev.c:707) calls libSystem `qsort`, which is **NOT
stable**.  Two rooms in seed=5 share `lx=68` (room w/ ly=4-6 was created
first; room w/ ly=14-17 was created last).  macOS qsort on the 8-element
input swaps these tied entries; Nethax used `jnp.argsort(stable=True)`,
which preserves creation order.

C verification (clang on macOS arm64):
```
8-element input [(68,0)(15,1)(37,2)(54,3)(51,4)(4,5)(20,6)(68,7)]
  qsort output: ... (68,7) (68,0)        # last two swapped
  stable sort:  ... (68,0) (68,7)        # creation order kept
```

That swap reverses Nethax's rooms 6 and 7 relative to vendor.  At Pass-1
`join(5, 6)` of makecorridors the tt-side strip is therefore taken from
the wrong room → wrong `tt_yh-tt_yl+1` (3 vs 4) → divergence at draw 542.

**Fix** (this commit): replace `jnp.argsort(stable=True)` in
`Nethax/nethax/dungeon/branches.py::_sort_rooms_by_lx` with a host-side
`pure_callback` that invokes libSystem `qsort` directly, matching the
vendor binary's tie-break byte-for-byte.

## Bug #2 — missing mkobj_at(RANDOM_CLASS) draws on inaccessible niches (FIXED)

With Bug #1 fixed, the next divergence appears at draw **1145**:

```
vendor: 1145 rnd mod=1000 res=902  caller=0x10b388410  (mkobj prob)
        1146 rnd mod=100  res=54   caller=0x10b388480  (mkobj tprob)
nethax: 1145 rn2 mod=8 ...                              (next iter of make_niches)
```

`_corr_inaccessible` in `corridors.py` mirrors vendor mklev.c:528-540 but
the **final** `if (!rn2(3)) mkobj_at(0, ...)` gate at mklev.c:539-540
drew the rn2(3) and discarded — it never consumed the `mkobj_at(0, ...)`
RANDOM_CLASS draws (rnd(1000) prob + rnd(100) class-pick + mksobj_init
per picked class).

**Fix** (this commit): added `consume_mkobj_random_draws` helper in
`Nethax/nethax/subsystems/random_objects.py` and wired it under
`lax.cond(r3mk == 0, ...)` in `_corr_inaccessible`.

## Bug #3 — fill_features mkfount somexy iteration count (UNFIXED)

After Bug #2, seed=5 byteparity drops from 18 → 14 divs but does NOT
close.  New first divergence at draw **1218**:

```
vendor sequence: rn2(10)=0 (mkfount fires)
                 somex/somey iter1  -> mod=7, mod=4
                 somex/somey iter2  -> mod=7, mod=4   <-- vendor needs 2 tries
                 rn2(7)             -> blessed-fount check
                 rn2(60)            -> mksink

nethax sequence: rn2(10)=0
                 somex/somey iter1 -> mod=7, mod=4
                 rn2(60)           -> mksink           <-- only 1 try
```

Cause: vendor's `mkfount` do-while loop (mklev.c:1578-1585) retries
`somexy` while the cell is `occupied(x,y) || bydoor(x,y)`.  Nethax's
fill-features `_fount_true` (rooms.py:2445) draws somexy once and
accepts; on this seed the first candidate hits an occupied/bydoor cell
in vendor but not in Nethax.

The mismatch could be:
- Nethax's `occupied` predicate misses something vendor tracks (e.g.
  the random object stamped earlier by the mklev.c:874 `mkobj_at`
  cascade that we now consume RNG for but DO NOT actually place on
  the typ/specials grid).
- Or `bydoor` differs because corridor/door placement is off by one
  cell from an earlier, deeper bug.

The required fix is either:
1. Stamp the random object onto the obj-track grid so `bydoor`/
   `occupied` matches, OR
2. Port vendor's `mkfount`/`mksink`/...'s full do-while around somexy
   in fill_features.

This is outside the 90-min budget for seed-5 alone.

## Seeds 6-8 still 18-19 divs after this commit

These seeds have NO tied `lx` rooms, so Bug #1's fix doesn't touch
them.  They diverge at a different mklev callsite that the seed=5 audit
hypothesised but did not pinpoint.  Their first ISAAC64 divergence
should be re-bisected with the same caller-trace flow — likely a
different missing-draw bug in fill_features or makeniche.

## Sweep results after this commit

```
seed= 0  PASS
seed= 1  FAIL (8)            (pre-existing, residual seed=1 bug)
seed= 2  PASS
seed= 3  PASS
seed= 4  FAIL (6)            (pre-existing seed=4 residual)
seed= 5  FAIL (14)           (down from 18 — Bugs 1+2 fixed, Bug 3 open)
seed= 6  FAIL (18)           (unchanged; different root cause)
seed= 7  FAIL (19)           (unchanged)
seed= 8  FAIL (19)           (unchanged)
seed= 9  FAIL (17)           (pet drift family, unchanged)
```

No regression on seeds 0, 2, 3 (the previously-passing ones).
