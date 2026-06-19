# Vendor Player Placement RNG Model

Source-of-truth files (vendor NLE 3.6.x, all paths under `vendor/nle/src/`):

- `mkmaze.c:275-319` — `place_lregion`
- `mkmaze.c:260-270` — `bad_location`
- `dungeon.c:1226-1256` — `u_on_rndspot`
- `include/hack.h:497` — `#define rn1(x, y) (rn2(x) + (y))`

## Headline facts

1. **The loop is 200 iterations, not 7.** `mkmaze.c:304` literally reads `for (trycnt = 0; trycnt < 200; trycnt++)`. Our wrapper's 7-iter assumption is wrong for any seed where the first 7 candidates all reject. Seed 0 just happens to accept early; seeds 1/2/5 do not.
2. **`u_on_rndspot` does NOT do its own RNG draws.** It is pure dispatch (`dungeon.c:1230-1252`): it picks one of three `place_lregion(...)` calls based on `upflag` and `On_W_tower_level`, then returns. All RNG comes from `place_lregion`.
3. **There is no "u_on_rndspot deterministic fallback".** The deterministic fallback lives inside `place_lregion` at `mkmaze.c:313-316`: row-major scan `for (x=lx..hx) for (y=ly..hy)`, first cell where `put_lregion_here(..., oneshot=TRUE, ...)` succeeds. Seed-0 hardcoded positions almost certainly come from an *early-accepting* probabilistic attempt, NOT from the deterministic scan — unless all 200 rolls reject, which is rare.

## Exact RNG sequence per iteration

Per `mkmaze.c:305-306`:

```c
x = rn1((hx - lx) + 1, lx);   // = rn2(hx-lx+1) + lx
y = rn1((hy - ly) + 1, ly);   // = rn2(hy-ly+1) + ly
```

For default-region calls (`lx==0` branch at `mkmaze.c:285-299`), the bounds are rewritten to:

- `lx = 1, hx = COLNO - 1 = 79`  → `rn2(79) + 1` → range `[1, 79]`
- `ly = 0, hy = ROWNO - 1 = 20`  → `rn2(21) + 0` → range `[0, 20]`

This matches the trace pairs (`79 ...`, `21 ...`) exactly. So `rn1` consumes **2 rn2 draws per iteration**: first arg 79, then arg 21.

## Accept/reject test (`put_lregion_here` + `bad_location`)

`mkmaze.c:331`: candidate accepted iff `!bad_location(x, y, nlx, nly, nhx, nhy)`.

`bad_location` (`mkmaze.c:265-269`) rejects when ANY of:

- `occupied(x, y)` — there is already a monster/feature there
- `within_bounded_area(x, y, nlx, nly, nhx, nhy)` — inside the exclusion sub-region (the `n*` rectangle)
- `levl[x][y].typ` is NOT `CORR` (only valid on maze levels), `ROOM`, or `AIR`

For a 5x5 single-room non-maze level, ONLY `ROOM` cells (the room's interior) accept. Wall ring, doors, corridors outside the room — all reject. `oneshot=FALSE` here because `lx != hx` (default region is 1..79), so the wall-clearing branch at lines 332-345 does NOT run during the probabilistic phase.

## Important: inarea / wall-ring question

The probabilistic phase draws from the **whole map** `[1..79] x [0..20]`, NOT from inside the room. The room only matters via `bad_location`'s `typ == ROOM` test rejecting non-room cells. So a 5x5 room sitting somewhere in the 79x21 grid will reject ~99% of throws and burn many of the 200 iterations.

## Pseudocode for the wrapper

```python
def place_player(state, lx=1, hx=79, ly=0, hy=20, nlx=0, nly=0, nhx=0, nhy=0):
    oneshot = (lx == hx and ly == hy)        # False for default region
    for trycnt in range(200):                # NOT 7
        x = rn2(state, hx - lx + 1) + lx     # consumes RNG
        y = rn2(state, hy - ly + 1) + ly     # consumes RNG
        if not bad_location(state, x, y, nlx, nly, nhx, nhy):
            u_on_newpos(state, x, y)
            return                           # early-accept; remaining draws NOT consumed
    # All 200 rejected — deterministic row-major scan, NO further rn2 draws:
    for x in range(lx, hx + 1):
        for y in range(ly, hy + 1):
            if not bad_location_oneshot(state, x, y, nlx, nly, nhx, nhy):
                u_on_newpos(state, x, y); return

def bad_location(s, x, y, nlx, nly, nhx, nhy):
    return (occupied(s, x, y)
            or (nlx <= x <= nhx and nly <= y <= nhy)
            or s.levl[x][y].typ not in (ROOM, AIR,
                                        CORR if s.is_maze_lev else _NEVER))
```

## RNG-desync implications

For every non-seed-0 case where vendor needs >7 iterations, our wrapper under-consumes by `2 * (actual_iters - 7)` rn2 draws. Every downstream subsystem (trap placement, monster gen, inventory) then reads off-by-N from the global ISAAC64 stream — explaining 0/12 at seeds 1/2/5.

**Action:** raise the loop to 200, implement `bad_location` against the room map, keep early-accept semantics. The deterministic fallback adds zero RNG draws, so wrapper RNG state is preserved whether the probabilistic phase succeeds or not.
