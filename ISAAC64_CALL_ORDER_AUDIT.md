# ISAAC64 Call-Order Divergence Audit — Nethax vs Vendor mklev.c

**Goal:** Explain why player start position diverges (NLE: x=37,y=3 vs Nethax: x=34,y=2) for seed=0.

---

## 1. Vendor ISAAC64 Consume Order — First 20 Calls

Traced through `mklev()` → `makelevel()` → `makerooms()` → per-room fill.

| # | File:Line | Vendor Call | Meaning |
|---|-----------|-------------|---------|
| 1 | `mklev.c:996` | `reseed_random(rn2)` | Re-seed ISAAC64 from the per-level seed at mklev() entry |
| 2 | `sp_lev.c:1154` | `rnd(1+abs(depth))` | Room 0 — lit-state roll A (`rnd` = `rn2+1`) |
| 3 | `sp_lev.c:1154` | `rn2(77)` | Room 0 — lit-state roll B |
| 4 | `sp_lev.c:1188` | `rn2(8)` | Room 0 — width `dx = 2 + rn2(8)` |
| 5 | `sp_lev.c:1189` | `rn2(4)` | Room 0 — height `dy = 2 + rn2(4)` |
| 6 | `sp_lev.c:1200` | `rn2(hx-lx-dx-xborder+1)` | Room 0 — x-placement within rect |
| 7 | `sp_lev.c:1202` | `rn2(hy-ly-dy-yborder+1)` | Room 0 — y-placement within rect |
| 8 | `sp_lev.c:1154` | `rnd(1+abs(depth))` | Room 1 — lit A |
| 9 | `sp_lev.c:1154` | `rn2(77)` | Room 1 — lit B |
| 10 | `sp_lev.c:1188` | `rn2(8)` | Room 1 — width |
| 11 | `sp_lev.c:1189` | `rn2(4)` | Room 1 — height |
| 12 | `sp_lev.c:1200` | `rn2(...)` | Room 1 — x-placement |
| 13 | `sp_lev.c:1202` | `rn2(...)` | Room 1 — y-placement |
| 14 | `mklev.c:710` | `rn2(nroom)` | Pick room for down-stair after all rooms placed |
| 15 | `mklev.c:715` | `rn2(nroom-1)` | Pick room for up-stair (different room) |
| 16 | `mklev.c:326` | `rn2(50)` | Corridor short-circuit randomness |
| 17 | `mklev.c:770` | `rn2(u_depth)` | Special-room type gate (SHOPBASE check) |
| 18 | `mklev.c:813` | `rn2(3)` | Room 0 fill — sleeping monster gate |
| 19 | `mklev.c:825` | `rn2(x)` | Room 0 fill — trap loop gate |
| 20 | `mklev.c:827` | `rn2(3)` | Room 0 fill — gold gate |

**Key observation for calls 2–7:** vendor `create_room()` (sp_lev.c:1127) draws
**lit-first, then width, then height, then x-pos, then y-pos** — in that order — for
each room, and the x/y offsets are derived from the bounding-rectangle (`hx-lx`,
`hy-ly`) which is only known *after* `rnd_rect()` is called (no RNG draw itself,
but it removes a rect from the pool in a state-dependent way).

---

## 2. Nethax ISAAC64 Consume Order — First 20 Calls

Traced through `env.reset()` → `generate_main_branch_l1_with_features()` →
`generate_rooms()` → `_isaac_draw_xywh()` + `_isaac_draw_lit()` →
`fill_ordinary_rooms()`.

| # | File:Line | Nethax Call | Meaning |
|---|-----------|-------------|---------|
| 1 | `env.py:144` | `rn2(5)` | Pre-draw room-count target: `room_target = rn2(5)` |
| 2 | `env.py:148` | `rn2(mc_upper)` | Pre-draw monster count: `rn2((n_rooms>>1)+1)` |
| 3 | `rooms.py:203` | `randint_jax(1, 1+y_range)` | Slot 0 — y-offset |
| 4 | `rooms.py:204` | `randint_jax(1, 1+x_range)` | Slot 0 — x-offset |
| 5 | `rooms.py:205` | `randint_jax(MIN_H, MAX_H+1)` | Slot 0 — height |
| 6 | `rooms.py:206` | `randint_jax(MIN_W, MAX_W+1)` | Slot 0 — width |
| 7–10 | `rooms.py:203–206` | (repeat for slot 1) | Slots drawn **y, x, h, w** order |
| … | … | (repeat for all `MAX_ROOMS * MAX_RETRIES = 1280` slots) | Full batch pre-drawn |
| 1281 | `rooms.py:227` | `randint_jax(1, 2+abs_depth)` | Lit roll A for slot 0 |
| 1282 | `rooms.py:228` | `randint_jax(0, 77)` | Lit roll B for slot 0 |
| … | … | (repeat lit A/B for all 40 slots) | All lit rolls after all xywh |
| +1 | `rooms.py:fill_one_isaac` | per-room monster/trap/gold/… | fill_ordinary_rooms draws |

---

## 3. First Divergence Point

**The divergence begins at Nethax call #1 (`env.py:144`) — before any room geometry is drawn.**

Vendor `mklev.c:996` calls `reseed_random(rn2)` which re-seeds ISAAC64 from a
per-level entropy source, then **immediately** calls `create_room()` which draws
lit-state first.  There is no pre-drawn room-count consume in vendor C; the room
count is determined by a `while (nroom < MAXNROFROOMS && rnd_rect())` loop that
terminates naturally when `rnd_rect()` exhausts available rectangles — no RNG
draw at all for the count.

Nethax does two pre-draws before any geometry (`rn2(5)` for room count,
`rn2(mc_upper)` for monster count), shifting every subsequent room-geometry draw
by at least 2 positions in the stream.

**Second structural divergence:** vendor order per room is `lit → dx → dy → x_pos → y_pos`
(sp_lev.c:1154,1188,1189,1200,1202).  Nethax `_isaac_draw_xywh` (rooms.py:203–206)
draws `y → x → h → w` — **lit and width/height are in different relative order,
and all `MAX_ROOMS * MAX_RETRIES` xywh draws are batched before any lit draw.**
Vendor interleaves lit with each room attempt before drawing dimensions; Nethax
batches all 1280 xywh values first, then all 80 lit pairs afterward.

**Third structural divergence:** stair placement. Vendor draws `rn2(nroom)` twice
(mklev.c:710,715) between room-loop and corridor-digging; Nethax places stairs
deterministically at room-0 / last-active-room centres with no RNG draw
(branches.py:731–753), so those two stream positions are never consumed.

---

## 4. Is Byte-Exact Dungeon-Gen Feasible?

**Verdict: feasible but requires a targeted structural refactor.** The ISAAC64
algorithm itself is already byte-exact (`vendor_rng.py` mirrors `isaac64.c`
correctly). The divergence is purely in *call order* and *call count*.

The parallel-scan architecture (`lax.scan` over pre-batched samples) is the
core obstacle.  Vendor C does:

```
for each room attempt:
    lit = rnd(...) + rn2(77)        ← drawn before dimensions
    dx  = 2 + rn2(...)
    dy  = 2 + rn2(4)
    x   = lx + offset + rn2(...)
    y   = ly + offset + rn2(...)
    if fits: accept; else retry
```

The room-count and monster-count are **not pre-drawn** from the stream in vendor C.

**What is NOT feasible without a large rewrite:**
- Keeping the current `_isaac_draw_xywh` / `_isaac_draw_lit` separation (all
  xywh first, all lit after) while matching vendor order.
- Pre-drawing room and monster counts from ISAAC64 before the first
  `create_room()` call.

---

## 5. Recommended Refactor

**Remove the two pre-draws in `env.reset()` (env.py:144–149).** Vendor never
pre-draws the room count; it is determined implicitly by `rnd_rect()` exhaustion.
Use a fixed room count of ~7 (the empirical mode for Dlvl 1) or sample it via
a host-side Python integer that does **not** consume the ISAAC64 stream.

**Reorder within `_isaac_draw_xywh` to match vendor per-room draw order:**
`lit_A → lit_B → dx → dy → x_pos → y_pos` (not `y → x → h → w`).  This
matches sp_lev.c:1154 before sp_lev.c:1188-1202.

**Thread `Isaac64State` through a `lax.fori_loop` (not a bulk scan) in
`generate_rooms`**, where each iteration draws all 6 values for one room attempt
in vendor order before testing overlap:

```python
def try_one_room(carry, _):
    vrng, placed, ... = carry
    vrng, lit_a = randint_jax(vrng, (), 1, 2+abs_depth)
    vrng, lit_b = randint_jax(vrng, (), 0, 77)
    vrng, dx    = randint_jax(vrng, (), 2, 10)  # 2+rn2(8)
    vrng, dy    = randint_jax(vrng, (), 2,  6)  # 2+rn2(4)
    vrng, x_pos = randint_jax(vrng, (), ...)
    vrng, y_pos = randint_jax(vrng, (), ...)
    ...
    return (vrng, ...), None
```

This is a pure `lax.scan`-compatible pattern (carry threading) and preserves
JIT-safety. The key constraint is the carry must thread `Isaac64State` —
exactly what the vendor-rng path already does in `_roll_hp` (spawning.py:664).

**Also remove stair-placement RNG draws** that vendor does (mklev.c:710,715)
but Nethax skips — currently those are two free stream positions that go
unconsumed in Nethax, causing further downstream offset.

With these three fixes (no pre-draws, reordered per-room draw sequence,
threaded fori_loop), the room-0 geometry — and therefore the up-stair
placement, which is room-0's centre — should match vendor C, resolving the
`player_x`/`player_y` divergence.

---

## 6. Cited Vendor Locations

- `vendor/nle/src/mklev.c:996` — `reseed_random(rn2)` at mklev() entry
- `vendor/nle/src/sp_lev.c:1154` — lit-state roll inside `create_room()`
- `vendor/nle/src/sp_lev.c:1188–1189` — `dx = 2 + rn2(8)`, `dy = 2 + rn2(4)`
- `vendor/nle/src/sp_lev.c:1200–1202` — x/y absolute placement draws
- `vendor/nle/src/mklev.c:710,715` — `rn2(nroom)` for stair room selection
- `vendor/nle/src/mklev.c:326` — `rn2(50)` corridor short-circuit
- `vendor/nle/src/mklev.c:813` — `rn2(3)` sleeping monster per-room gate
- `vendor/nle/src/mkmap.c:461` — lit-state formula (`rnd(1+abs(depth)) < 11 && rn2(77)`)
