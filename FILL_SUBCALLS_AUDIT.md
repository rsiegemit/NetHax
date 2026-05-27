# fill_ordinary_rooms Sub-Call RNG Deficit Audit

**Scope:** vendor `fill_ordinary_room` (mklev.c:939–1171) per-room helper cascade.
**Question:** Does Nethax `fill_one_isaac` model the *body* draws of each helper, or only the gate draws?

---

## 1. Vendor Per-Helper RNG Draw Counts

### Gate summary (draws in fill_ordinary_room itself)

| Line | Call | Gate draws | Note |
|------|------|-----------|------|
| 974 | `!rn2(3)` → monster | 1 | + somexy (2) |
| 984 | `while(!rn2(x))` × N | N+1 | + mktrap body per iter |
| 986 | `!rn2(3)` → mkgold | 1 | + somexy (2) |
| 990 | `!rn2(10)` → mkfount | 1 | + find_okay_roompos somexy |
| 992 | `!rn2(60)` → mksink | 1 | + find_okay_roompos somexy |
| 994 | `!rn2(60)` → mkaltar | 1 | + find_okay_roompos somexy |
| 999 | `!rn2(x)` → mkgrave | 1 | + find_okay_roompos somexy |
| 1003 | `!rn2(20)` → mkcorpstat | 1 | + somexy (2) |
| 1137 | `!rn2(nroom*5/2)` → box | 1 | + short-circuit body |
| 1142 | `!rn2(27+3*depth)` → graffiti | 1 | + short-circuit body |
| 1158 | `!rn2(3)` → mkobj loop | 1 | + short-circuit body |

### Helper bodies — draws **inside** each helper (not in gate)

#### `mkgold(0L, x, y)` — mkobj.c:2003
- `rnd(30/max(...))` **+** `rnd(level_difficulty()+2)` — **2 draws** (amount==0 path, which is always the case here)
- Then calls `mksobj_at(GOLD_PIECE, …)` → `mksobj(GOLD_PIECE, TRUE, FALSE)` → `mksobj_init` → COIN_CLASS does nothing in `mksobj_init`, but `mkobj_erosions` checks `may_generate_eroded` (metals: FALSE for gold) → **0 erosion draws**
- **Body draws: 2** (both unconditional when amount==0)

#### `mktrap(0, …)` — mklev.c:2036
Body draws per iteration (not counting the gate `!rn2(x)`):
1. `traptype_rnd`: `rnd(TRAPNUM-1)` = **1 draw**; conditionally `rn2(7)` for HOLE = **0–1 draw**
2. Location: `somexyspace` loop (≥1 draw per axis, up to ~200 retries; typical = **2 draws**)
3. Possibly `makemon` for WEB trap (monster subsystem, not counted here)
4. Possibly `mktrap_victim` path: `rnd(4)` = **1 draw** for victim check (lvl ≤ rnd(4))
- **Body draws per trap: 4–5 (typical)**; Nethax already models `rnd(TRAPNUM-1)` + somexy per step — this is **correctly modelled**.

#### `mkfount(croom)` — mklev.c:2285
- `find_okay_roompos` → `somexyspace` loop (typical **2 draws** for x+y)
- `!rn2(7)` for blessed fountain = **1 draw**
- **Body draws: 3**
- **Nethax models:** `somexy` (2 draws). **Missing: `rn2(7)` blessed-fountain draw = 1 draw/call.**

#### `mksink(croom)` — mklev.c:2317
- `find_okay_roompos` → somexyspace (**2 draws**)
- No other RNG.
- **Body draws: 2** (somexy already modelled — covered by Nethax).

#### `mkaltar(croom)` — mklev.c:2332
- `find_okay_roompos` → somexyspace (**2 draws** — modelled by Nethax somexy)
- `rn2((int)A_LAWFUL + 2) - 1` = `rn2(3)` for alignment = **1 draw**
- **Nethax models:** somexy + `rn2(100)` coin + `rn2(3)` alt_align (the `induced_align` logic). However, vendor mkaltar draws `rn2(3)` **directly** (unconditional). Nethax draws `rn2(100)` + `rn2(3)` — that is **1 extra draw** not in vendor's `mkaltar` body.
- **Net deficit for altar: +1 spurious draw in Nethax** (the `rn2(100)` coin comes from `induced_align` which is *not* called by vendor mkaltar; vendor just calls `rn2(3)` once).

#### `mkgrave(croom)` — mklev.c:2353
- `find_okay_roompos` → somexyspace (**2 draws**)
- `dobell = !rn2(10)` = **1 draw** (before find_okay_roompos — fires unconditionally)
- `!rn2(3)` → buried gold: **1 draw**; if taken: `mksobj(GOLD_PIECE)` (2 draws: rnd for class-prob, then mksobj_init COIN does nothing) + `rnd(20)` + `rnd(5)` = **2 body draws** for gold amount
- `rn2(5)` loop count = **1 draw**; then for each iteration: `mkobj(RANDOM_CLASS, TRUE)` = **2+ draws** (rnd(100) class + rnd(prob_total) type + mksobj_init body)
- Possibly `mksobj_at(BELL, …)` if dobell = **mksobj draws**
- **Minimum body draws (grave gate passes): dobell(1) + somexy(2) + rn2(3)(1) = 4 draws before any objects**
- **Nethax models:** only somexy. **Missing: `rn2(10)` dobell + `rn2(3)` buried-gold gate + loop-count `rn2(5)` + all mkobj/mksobj body draws.**

#### `mkcorpstat(STATUE, …, CORPSTAT_INIT)` — mkobj.c:2067
- Calls `mksobj_at(STATUE, x, y, TRUE, FALSE)` → `mksobj(STATUE, TRUE, FALSE)` → `mksobj_init`
  - ROCK_CLASS / STATUE path: `rndmonnum()` = **1 draw** (corpsenm)
  - Conditional: `rn2(level_difficulty()/2 + 10) > 10` → possibly `mkobj(SPBOOK_no_NOVEL)` = **1 draw** for that check + spellbook draws if taken
  - `mkobj_erosions`: stone → `may_generate_eroded` is FALSE for statues → **0 erosion draws**
  - `rn2(2)` for gender (if not neuter/male/female) = **0–1 draw**
- **Body draws: 2–4**
- **Nethax models:** only somexy (2 draws). **Missing: at minimum 2 draws from mksobj_init.**

#### `mksobj_at(rn2(3)?LARGE_BOX:CHEST, …)` — box/chest path (line 1138)
- Gate `!rn2(nroom*5/2)` already modelled with `lax.cond`.
- Body: `rn2(3)` type-pick + somexy + `mksobj(CHEST/LARGE_BOX, TRUE, FALSE)` → `mksobj_init` TOOL_CLASS:
  - CHEST/LARGE_BOX: `rn2(5)` locked + `rn2(10)` trapped + `rn2(100)` tknown = **3 draws**
  - Then `mkbox_cnts`: `rn2(n+1)` = **1 draw** for count; per item: `rnd(100)` class + mkobj body = **2+ draws per item**
  - `blessorcurse`: not called for containers
- **Nethax models inside `_box_true`:** `rn2(3)` type + somexy. **Missing: all mksobj_init draws for the container (3 draws) + mkbox_cnts (~1 + N×2 draws).**

#### `random_engraving` (line 1144)
- `rn2(4)` branch = **1 draw**
- If taken (3/4 chance): `getrumor` → `rn2(2)` truth + `rng(filechunksize)` = **2 draws**
- Else (1/4): `get_rnd_text` → `rng(filechunksize)` per try (up to 10) = **~1 draw**
- `wipeout_text` iterates over characters calling no RNG.
- **Body draws: 2–3**
- **Nethax comment at line 1853:** "random_engraving() is a table-lookup with no RNG draw in vendor C" — **THIS IS WRONG**. Vendor calls `rn2(4)`, `rn2(2)`, and `rng(filesize)` = **2–3 draws per graffiti call**. Nethax models 0 body draws for engraving. **Missing: 2–3 draws per graffiti event.**

#### `mkobj(RANDOM_CLASS, …)` object scatter (line 1159)
- `rnd(100)` for class + `rnd(oclass_prob_total)` for type = **2 draws**
- Then `mksobj_init` body (varies by class, ~3–8 draws typical) + `mkobj_erosions` (~1–3 draws)
- **Body draws: 6–13 per object**
- **Nethax models:** only somexy (2 draws per object). **Missing: 4–11 draws per mkobj call.**

---

## 2. Nethax `fill_one_isaac` — What Is and Isn't Modelled

| Helper | Gate draw | Nethax body draws | Vendor body draws | Deficit per call |
|--------|-----------|-------------------|-------------------|-----------------|
| sleeping monster | ✓ | somexy only | makemon (large) | large, but monster system separate |
| `mktrap` body | ✓ | `rnd(TRAPNUM-1)` + somexy | same + HOLE `rn2(7)` + victim `rnd(4)` | ~1–2 |
| `mkgold` body | ✓ | somexy only (2) | 2 amount draws + mksobj | **+2** |
| `mkfount` body | ✓ | somexy (2) | somexy + `rn2(7)` | **+1** |
| `mksink` body | ✓ | somexy (2) | somexy (2) | 0 |
| `mkaltar` body | ✓ | somexy + `rn2(100)` + `rn2(3)` (3) | somexy + `rn2(3)` (3) | **Nethax +1 spurious** |
| `mkgrave` body | ✓ | somexy (2) | `rn2(10)` + somexy + `rn2(3)` + `rn2(5)` + N×mkobj | **+3 minimum, +10–30 if objects** |
| `mkcorpstat/statue` body | ✓ | somexy (2) | somexy + 2–4 mksobj draws | **+2–4** |
| box/chest body | ✓ (lax.cond) | `rn2(3)` + somexy | + mksobj_init(3) + mkbox_cnts(1+N×2) | **+6–15** |
| `random_engraving` body | ✓ (lax.cond) | 0 | 2–3 | **+2–3** |
| `mkobj` scatter body | ✓ (lax.cond) | somexy per obj | somexy + 6–11 per obj | **+4–11 per obj** |

---

## 3. Total Deficit Per OROOM

Per room (ignoring probabilistic items that rarely trigger):

| Source | Prob of firing | Expected draws missed/room |
|--------|---------------|---------------------------|
| `mkgold` body (2 draws) | ~0.33 | **+0.67** |
| `mkfount` blessed rn2(7) | ~0.10 × 1 | **+0.10** |
| `mkaltar` spurious rn2(100) | ~0.017 | **−0.017** (over-draws) |
| `mkgrave` body (3+ draws) | ~1/grave_x × (3+) | **+0.5–2** |
| `mkcorpstat` body (2–4 draws) | ~0.05 × 3 | **+0.15** |
| `random_engraving` body (2–3 draws) | ~1/graffiti_mod × 2.5 | **+0.5–1** |
| box/chest body (6–15 draws) | ~1/(nroom×2.5) × 10 | **+1–4** |
| `mkobj` scatter body (6–11 per obj) | ~0.33 × (1 + geom(0.2)) × 8 | **+3–6** |
| `mktrap` HOLE rn2(7) + victim rnd(4) | ~0.33 × N_traps × 1.5 | **+0.5–1** |

**Estimated total deficit per OROOM: ~7–16 draws**
**Over 5–7 rooms per level: ~40–100 draws per level**

This aligns with the observed ~1682 draw deficit (1789 − 107) when combined with deficits from corridors, makemon, and check_room already identified.

---

## 4. Top 3 Fixes Ranked by Impact

### Fix 1 — `mkobj` scatter body draws (~3–6 draws/room, highest variance)
`fill_one_isaac` only calls `somexy` inside `_mkobj_true` but never models the `rnd(100)` class-pick + `rnd(prob_total)` type-pick + `mksobj_init` draws (~4–11 each). With ~1.25 expected objects per room when the gate passes (33% chance), this is **3–6 draws/room expected deficit**. Fix: add `rnd(100)` + `rnd(prob)` + class-specific `mksobj_init` draws inside `_mkobj_step`.

### Fix 2 — box/chest `mksobj_init` + `mkbox_cnts` body (~3–6 draws/call)
When the box gate passes, Nethax draws `rn2(3)` + somexy but skips `mksobj_init` for CHEST/LARGE_BOX (`rn2(5)` locked + `rn2(10)` trapped + `rn2(100)` tknown = 3 draws) and `mkbox_cnts` (`rn2(n+1)` count + `rnd(100)` + `rnd(prob)` per item). Fix: add TOOL_CLASS init draws and mkbox_cnts draws inside `_box_true`.

### Fix 3 — `random_engraving` body draws (~2–3 draws/call, currently 0 modelled)
The comment at rooms.py:1853 incorrectly states random_engraving has "no RNG draw". In reality vendor draws `rn2(4)` + either `rn2(2)` + `rng(filesize)` (rumor path) or `rng(filesize)` (engrave path). Fix: add `rn2(4)` unconditionally inside `_graffiti_true`, then conditionally `rn2(2)` + `rng(filesize)` or just `rng(filesize)`.

---

*Audit is read-only. No code was changed.*
