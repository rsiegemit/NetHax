# mklev Preamble + Trailer RNG Deficit Audit

_Vendor: `vendor/nle/src/mklev.c`. Nethax: `Nethax/nethax/dungeon/branches.py` + `rooms.py` + `corridors.py`._

---

## 1. Vendor preamble draws (before `makerooms()`)

`makelevel()` lines 662–706:

| Draw | Location | Notes |
|------|----------|-------|
| `rn2(5)` | mklev.c:693 | Medusa / hell short-circuit to `makemaz`. On Main Dlvl 1 (dnum != medusa_level.dnum at depth 1) this check is **always** reached but the `dnum` comparison fails so the draw fires and returns non-zero — 1 draw consumed before regular path. |

**Preamble total: 1 draw.**

`sort_rooms()` (line 707) uses `qsort` with `do_comp` — no RNG.

---

## 2. Vendor preamble draws between makerooms and fill_ordinary_rooms

These fire on Main Dlvl 1 after `makerooms()` returns and before the per-room fill loop (lines 710–796):

| # | Draw | Location | Condition on Dlvl 1 |
|---|------|----------|---------------------|
| 1 | `rn2(nroom)` | mklev.c:710 | Always (down-stair room pick) |
| 2 | `rn1(hx-lx+1, lx)` somex | mkroom.c:643 | Always (down-stair x) |
| 3 | `rn1(hy-ly+1, ly)` somey | mkroom.c:650 | Always (down-stair y) |
| 4 | `rn2(nroom-1)` | mklev.c:715 | When nroom > 1 |
| — | up-stair somex/somey | mklev.c:722–725 | **Skipped** on Dlvl 1 (`u.uz.dlevel != 1` is false) |

`makecorridors()` (line 734): heavy draw sequence — `rn2(50)` early-bail, `rn2(nroom)+4` random extra joins × (`rn2(nroom)` + `rn2(nroom-2)`) + per-join `finddpos` + `dodoor` (`rn2(8)`, `rn2(3)`, `rn2(5)`, `rn2(6)`, `rn2(25)`) + `dig_corridor` (`rn2(35)`, `rn2(100)`, `rn2(50)`, bias draws). Typical draw count: **50–150**.

`make_niches()` (line 735): `rnd((nroom>>1)+1)` niches, each consuming `rn2(nroom)`, `rn2(5)`, `place_niche`→`rn2(2)`, `rn2(4)`, `rn2(7)`, `rn2(5)`, `dodoor` draws. Typical: **20–60**.

`do_vault()` check (line 738): vault_x set inside `makerooms` — the `rnd_rect`+`rn2(2)` vault-gate draws are inside `makerooms`, not here. If vault was placed: `!rn2(3)` (line 752) for `makevtele` → `makeniche(TELEP_TRAP)` → more draws. **0–10**.

SHOPBASE / special-room cascade (lines 764–796): On Dlvl 1 (`u_depth=1`), all checks fail because `u_depth > 1` is the first gate. **0 draws** consumed.

**Between-makerooms-and-fill total: 4 (stair) + ~80 (corridors) + ~30 (niches) + 0–10 (vault post) = ~114–124 draws.**

---

## 3. Vendor trailer draws (after `fill_ordinary_rooms`)

`makelevel()` ends at line 886; `mklev()` continues (lines 1005–1036):

- `bound_digging()` — no RNG.
- `mineralize(-1,-1,-1,-1,FALSE)` — iterates over map cells; draws `rn2(1000)` per eligible STONE cell for gold, then `rn2(1000)` for gems, plus `rnd(goldprob*3)` / `rnd(2+dunlev/3)` / `rn2(3)` per placed object. On a typical 21×80 map (~1600 STONE cells passing the 8-neighbour check): **~50–200 draws**.
- `topologize()` — no RNG.
- `set_wall_state()` — no RNG.

**Trailer total: ~50–200 draws (mineralize).**

---

## 4. Nethax equivalent coverage

| Phase | Vendor | Nethax | Status |
|-------|--------|--------|--------|
| `rn2(5)` Medusa pre-check | 1 draw | **Not modelled** | **MISSING** |
| Down-stair room + somex/somey | 3 draws | Implemented in `generate_main_branch_l1` (vendor_rng path) | Covered |
| Up-stair room pick `rn2(nroom-1)` | 1 draw | Implemented | Covered |
| Up-stair somex/somey | 0 (skipped Dlvl 1) | Correctly skipped | Covered |
| `makecorridors` (corridors.py) | ~80 draws | Implemented (`makecorridors`, `join`, `dodoor`, `dig_corridor`) but **not called from `generate_main_branch_l1`** — only `connect_rooms` (Threefry L-shape) is used | **NOT WIRED** |
| `make_niches` (corridors.py) | ~30 draws | Implemented but **not wired** into `generate_main_branch_l1_with_features` (uses `_place_niches` with Threefry instead) | **NOT WIRED** |
| Vault `rn2(3)` for vtele | 0–1 draws | `maybe_create_vault` uses Threefry, not ISAAC64 | **NOT WIRED** |
| SHOPBASE cascade | 0 (Dlvl 1) | `assign_special_room` exists but **not called** from `generate_main_branch_l1_with_features` | Not needed on Dlvl 1 |
| `mineralize` | ~50–200 draws | **No equivalent exists** | **MISSING** |

---

## 5. Draw count summary

| Phase | Vendor draws | Nethax (ISAAC64 stream) | Deficit |
|-------|-------------|------------------------|---------|
| Pre-makerooms Medusa check | 1 | 0 | **1** |
| Stair picks + somex/somey | 4 | 4 | 0 |
| `makecorridors` | ~80–150 | 0 (not wired) | **~80–150** |
| `make_niches` | ~20–60 | 0 (not wired, Threefry used) | **~20–60** |
| Post-vault `rn2(3)` | 0–1 | 0 (Threefry) | **0–1** |
| `mineralize` (trailer) | ~50–200 | 0 (not implemented) | **~50–200** |
| **TOTAL** | **~155–416** | **~4** | **~151–412** |

---

## 6. Key findings

1. **`makecorridors` + `make_niches` (corridors.py)** are fully implemented with ISAAC64 byte-exact draws but are **never called** from the active level-gen pipeline. `generate_main_branch_l1` calls the Threefry `connect_rooms` instead, consuming zero ISAAC64 draws for ~80–150 vendor draws.

2. **`mineralize`** has no Nethax equivalent at all. It is the largest single contributor (~50–200 trailer draws) and completely absent.

3. **`rn2(5)` Medusa pre-check** (mklev.c:693) fires once before `makerooms` on every normal-level path. Not modelled.

4. **`maybe_create_vault`** uses Threefry for its gate coin, so the `rn2(3)` vtele draw is not ISAAC64-exact.

5. The vendor total of ~1789 draws per Dlvl 1 vs our ~107 implies an approximate **~1682 draw deficit**. The gaps above account for ~150–400; the remaining delta (~1280+) likely lives in `fill_ordinary_rooms` deep-paths (`makemon`, `mkgold`, `mktrap`, `mkobj_at`, `mkcorpstat`, `random_engraving`) which each expand into many sub-draws not yet counted here.
