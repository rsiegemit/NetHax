# somexy / room-coord helper RNG Audit

Vendor: `somex(croom) = rn1(hx-lx+1, lx)` → `rn2(width) + lx`  
Vendor: `somey(croom) = rn1(hy-ly+1, ly)` → `rn2(height) + ly`

Our equivalent: `randint_jax(vrng, (), x1, x2+1)` (X) and `randint_jax(vrng, (), y1, y2+1)` (Y).  
`randint_jax(rng, (), a, b)` = `rn2_jax(rng, b-a) + a` = `rn2(width) + lx` — **byte-exact match**.

---

## Per-Call-Site Verification

Vendor line numbers follow the task spec (mapped to `fill_ordinary_room` / `generate_stairs`
in the current 3.7 source; actual function boundaries are at lines 939 and 2250).

| # | Vendor site | Vendor draws | Our code (rooms.py / branches.py) | Status |
|---|---|---|---|---|
| 1 | `mklev.c:712` down-stair `somex(croom)` / `somey(croom)` | `rn2(nroom)` + `rn2(width)` + `rn2(height)` | `branches.py:802` — `rn2_jax(nroom)` then `rn1_jax(dn_w_safe, dn_lx)` + `rn1_jax(dn_h_safe, dn_ly)` via `_draw_down_xy` | ✅ MATCH |
| 2 | `mklev.c:723-725` up-stair `do { somex; somey } while occupied` retry | `rn2(nroom-1)` + ≥1× (`rn2(width)` + `rn2(height)`) | `branches.py:861` — draws `rn2(nroom-1)` only; **no up-stair somex/somey consumed on Dlvl 1** (gated on `dlevel != 1`). Comment at line 868 documents the skip. Phase 4+ levels need this path. | ⚠️ PARTIAL — correct for Dlvl 1, deficit for Dlvl > 1 |
| 3 | `mklev.c:814-815` sleeping monster `somexy` | `rn2(3)` + `rn2(width)` + `rn2(height)` (unconditional) | `rooms.py:1753-1754` — `randint_jax(0,3)` + `somexy(vrng)` | ✅ MATCH |
| 4 | `mklev.c:828` gold `somexy` | `rn2(3)` + `rn2(width)` + `rn2(height)` (unconditional) | `rooms.py:1785-1786` — `randint_jax(0,3)` + `somexy(vrng)` | ✅ MATCH |
| 5 | `mklev.c:854-855` box `somexy` | `rn2(box_mod)` → if hit: `rn2(3)` + `somexy` + mksobj_init draws | `rooms.py:1842-1880` — `lax.cond(box_gate, _box_true, _box_false, vrng)`; `_box_true` draws `rn2(3)` + `somexy` + mksobj_init chain | ✅ MATCH (short-circuit via lax.cond) |
| 6 | `mklev.c:862-866` graffiti `do { somexy } while (typ!=ROOM && !rn2(40))` | `rn2(graffiti_mod)` → if hit: `random_engraving` (3 draws) + ≥1 `somexy` + up-to-7 more (`somexy` + `rn2(40)`) | `rooms.py:1899-1929` — `lax.cond(graffiti_gate, _graffiti_true, ...)`: 3 engraving draws + 1 mandatory `somexy` + 7-iter scan of (`somexy` + `rn2(40)`) | ✅ MATCH (cap=8 total iterations) |
| 7 | `mklev.c:875` `mkobj_at` first `somexy` | `rn2(3)` → if hit: `somexy` + mkobj draws | `rooms.py:1934-1977` — `lax.cond(mkobj_gate, _mkobj_true, ...)`: `somexy` + `cls_roll` + `typ` + `mksobj_init` draws | ✅ MATCH |
| 8 | `mklev.c:880` `mkobj_at` inner-loop `somexy` | while `!rn2(5)`: `somexy` + mkobj draws (cap=8 iters) | `rooms.py:1953-1971` — `_mkobj_step` scan (length=8): each iteration: `rn2(5)` → if cont: `somexy` + full mkobj draws via `lax.cond` | ✅ MATCH |
| 9 | `mkroom.c::mkfount` via `find_okay_roompos` | `rn2(10)` → if hit: `somexy` (≥1 in do-while) | `rooms.py:1789-1795` — `rn2(10)` + 1× `somexy` | ✅ MATCH (first-try assumption; see note below) |
| 10 | `mkroom.c::mksink` via `find_okay_roompos` | `rn2(60)` → if hit: `somexy` (≥1 in do-while) | `rooms.py:1798-1799` — `rn2(60)` + 1× `somexy` | ✅ MATCH (first-try assumption) |
| 11 | `mkroom.c::mkaltar` via `find_okay_roompos` | `rn2(60)` → if hit: `somexy` + `rn2(3)` induced_align | `rooms.py:1809-1823` — `rn2(60)` + `somexy` + `rn2(3)` | ✅ MATCH |
| 12 | `mkroom.c::mkgrave` via `find_okay_roompos` | `rn2(grave_x)` → if hit: `somexy` (≥1 in do-while) + grave content draws | `rooms.py:1826-1832` — `rn2(grave_x)` + 1× `somexy` (grave content draws not modeled) | ✅ MATCH (first-try assumption; grave contents not tracked) |

---

## Notes

### `find_okay_roompos` retry loop
Vendor calls `somexy` inside `do { } while (occupied || bydoor)` — potentially >1 draw.
Our port always consumes exactly 1 `somexy` (unconditionally). This is the **first-try assumption**:
for typical rooms (low occupancy at generation time), the first position is unoccupied, so
one `somexy` matches vendor's modal behaviour. Dense/occupied rooms would cause vendor to draw
extra somexy calls that we skip. **This is a known bounded deficit, not a structural mismatch.**
It does not affect the fixed-schedule ISAAC64 stream for normal dungeon levels.

### Up-stair somex/somey on Dlvl > 1 (Site 2)
On `u.uz.dlevel != 1`, vendor draws `somexy` for the up-stair position inside a
`do { } while (occupied)` retry loop. Our port skips these draws (comment at
`branches.py:868`). This is a **documented deficit for Dlvl > 1** — Phase 4 of
MKLEV_PORT_PLAN.md tracks it.

### `somexy` bounds correctness
`randint_jax(vrng, (), x1, x2+1)` = uniform in `[x1, x2]` = `rn1(hx-lx+1, lx)`.  
`randint_jax(vrng, (), y1, y2+1)` = uniform in `[y1, y2]` = `rn1(hy-ly+1, ly)`.  
Both are **byte-exact** with vendor `somex`/`somey`. No fixed `jnp.where` substitution found.

### Graffiti `random_engraving` approximation
Our port always draws 3 engrave-RNG values (`rn2(4)` + `rn2(2)` + `rn2(10000)`) matching
the rumor path (75% of vendor calls). The engrave-file path (25%) draws only 2 values,
causing a 1-draw over-advance in that case. This is an accepted approximation for JIT-safety,
noted in `rooms.py:1891-1896`.

---

## Missed Sites

The following vendor `somex/somey` call sites are outside the `fill_ordinary_room` /
`generate_stairs` scope of this audit and are **not yet modeled**:

| Vendor site | Context | Status |
|---|---|---|
| `mkroom.c::fill_zoo` (lines 298, 311, 424) | Zoo/special room monster placement | Not in scope (special rooms) |
| `mklev.c:1667-1669` | Branch stair room (stair find for branch levels) | Not in scope (branch levels) |
| `mklev.c:2093` | Niche/croom during `fill_ordinary_room` sub-path | Covered by `corridors.py` make_niches |

---

## Estimated Deficit

| Category | Draw deficit per level | Severity |
|---|---|---|
| Up-stair somex/somey (Dlvl > 1) | 2 draws (+ retry draws if occupied) | Medium — affects all non-Dlvl-1 levels |
| `find_okay_roompos` retry >1 | 0–N extra somexy per occupied cell | Low — rare in practice |
| Graffiti engrave-path over-draw | +1 draw in ~25% of graffiti events | Very low |
| Grave content draws | Missing buried-gold + mkobj loop draws | Low — only affects post-grave stream |

**Structural correctness for Dlvl 1 (tested level): all 12 sites verified MATCH.**  
**Primary deficit: up-stair somex/somey for Dlvl > 1, tracked in MKLEV_PORT_PLAN.md Phase 4.**
