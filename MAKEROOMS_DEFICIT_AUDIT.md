# makerooms RNG Deficit Audit

## Summary

Vendor `mklev.c::makerooms` + `sp_lev.c::create_room` consume ~1789 ISAAC64 draws
per level for seed=0.  Our port consumes ~107.  This audit identifies the
structural causes.

---

## 1. Outer makerooms Loop (mklev.c:229)

**Vendor:**
```c
while (nroom < MAXNROFROOMS && rnd_rect()) {
    ...
    } else if (!create_room(-1,-1,-1,-1,-1,-1, OROOM, -1))
        return;
}
```

**Our port (`rooms.py::makerooms`):**
```python
lax.fori_loop(0, MAXNROFROOMS, body, carry0)
```

- Outer iteration count: `MAXNROFROOMS = 40` — correct.
- Per-iteration `rnd_rect()` draw at Step A, gated by `alive` — correct.
- Dead iterations (after `alive=False`) consume zero draws via `skip_rnd_rect` — correct.
- `still_alive = alive & has_rect & (nroom < MAXNROFROOMS)` kills loop on
  empty pool or OROOM failure — correct.

**Outer loop verdict: NO deficit here.**

---

## 2. Inner create_room do-while (sp_lev.c:1161, 1277)

**Vendor:**
```c
int trycnt = 0;
do {
    ...
} while (++trycnt <= 100 && !r1);
```

With `trycnt` starting at 0, the post-increment check `++trycnt <= 100` allows
`trycnt` to reach 101 before the condition becomes false — meaning the body
executes **101 times** in the all-fail case.

**Our port (`create_room.py`):**
```python
_MAX_TRYCNT: int = 100
...
lax.fori_loop(0, _MAX_TRYCNT, _body, init)  # 100 iterations
```

**Deficit: 1 iteration per `create_room` call (every call, unconditionally).**

---

## 3. Per-Attempt Draw Counts

### Pre-loop (once per `create_room` call, not per attempt)
| Draw | Vendor expression           | Always? |
|------|-----------------------------|---------|
| lit_A | `rnd(1 + abs(depth))`      | YES     |
| lit_B | `rn2(77)`                  | Only when `lit_A < 11` |

### Per-attempt body (random path, `!vault`)
| Draw | Vendor expression                        | Gate                        |
|------|------------------------------------------|-----------------------------|
| D1   | `rn2(rect_cnt)` via `rnd_rect()`         | Only when `rect_cnt > 0`    |
| D2   | `rn2((hx-lx>28)?12:8)` for dx           | Only when `has_rect`        |
| D3   | `rn2(4)` for dy                          | Only when `has_rect`        |
| D4   | `rn2(...)` for xabs                      | Only when rect fits         |
| D5   | `rn2(...)` for yabs                      | Only when rect fits         |
| D6   | `rn2(nroom)` centre-yabs gate            | Only when geometry triggers |
| D7   | `rn1(3,2)` yabs override                 | Only when D6 passes         |

Typical failed attempt (rect found, fails fit test): **3 draws** (D1+D2+D3).
Successful attempt (fits + check_room passes): **5–7 draws** (D1–D5 + maybe D6/D7).

---

## 4. Deficit Quantification

For seed=0, a typical level places ~7 rooms.  Each `create_room` call misses
exactly 1 iteration = ~3 draws (the last failed attempt would have drawn D1+D2+D3).

| Source                    | Per call | ~7 rooms | ~40 outer iters |
|---------------------------|----------|----------|-----------------|
| trycnt off-by-one (§2)    | ~3 draws | ~21 draws | n/a            |
| Outer loop structural issue | 0      | 0        | 0               |

**Estimated deficit from trycnt alone: ~21 draws** out of 1682 missing.

The dominant gap (1682 draws) indicates additional deficits not yet identified in
this audit pass.  Candidates (out of scope for this read-only audit):

- `check_room` RNG: our port unconditionally skips the `rn2(3)` draw at
  `sp_lev.c:1103` because it assumes a "fresh stone level."  If vendor's level
  is not fully stone-zero at makerooms time (e.g. boundary cells are set),
  every overlap scan fires `rn2(3)`.  On a level with previously-placed rooms,
  this is O(room_perimeter) draws per candidate — potentially hundreds of draws
  per create_room call.
- `makelevel` draws occurring **before** `makerooms` that shift the stream
  baseline (mkmap, wallification, OROOM vs special level detection, etc.)

---

## 5. Fix Proposals

### Fix 1 — trycnt off-by-one (confirmed)

In `create_room.py`, change:
```python
_MAX_TRYCNT: int = 100
...
lax.fori_loop(0, _MAX_TRYCNT, _body, init)
```
to:
```python
_MAX_TRYCNT: int = 101   # vendor: do { } while (++trycnt <= 100)
...
lax.fori_loop(0, _MAX_TRYCNT, _body, init)
```

This adds 1 attempt to every `create_room` call.  The `done` gate inside
`_body` ensures the extra iteration consumes zero draws when the room was
already placed on attempt ≤ 100.  For an all-fail result it adds ~3 draws.

### Fix 2 — check_room rn2(3) draws (suspected major source)

`check_room` (sp_lev.c:1099-1113) fires `rn2(3)` for every non-stone cell
inside the candidate's padded bounding box.  Our port assumes all cells are
stone and skips this draw entirely.  If vendor's level has any non-stone cells
when makerooms runs — even just the map border — every room attempt on a later
iteration can consume O(room_perimeter × xlim × ylim) draws.

Verify by instrumenting a vendor C run to count `rn2(3)` calls during
makerooms, then implement the full `check_room` scan in `_try_one_attempt`.

### Fix 3 — Investigate pre-makerooms draw baseline

Confirm that vendor stream position at the point `makerooms()` is called
matches our stream position at the point we call `makerooms()`.  A shift here
compounds multiplicatively across all subsequent draws.
