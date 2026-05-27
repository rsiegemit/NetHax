# mksobj RNG Cascade Audit

**Source:** `vendor/nethack/src/mkobj.c` (NetHack 3.7, rev 1.326)
**Auditor:** read-only, no code changes

---

## 1. RNG Draw Primitives

| Primitive | Draws | Notes |
|-----------|-------|-------|
| `rn2(n)` | 1 | single ISAAC64 word |
| `blessorcurse(n)` | 1–2 | `rn2(n)` + conditional `rn2(2)` |
| `rne(x)` | 1–5 | `while tmp < utmp && !rn2(x)` — at ulevel 0, utmp=5, so 1–5 draws of `rn2(x)` |
| erosions | 1–6 | see §2 |

---

## 2. `mkobj_erosions` (called for every non-initial-inventory damageable object)

```
rn2(100)                   # 1 draw always
  if != 0:
    rn2(80)                # +1 draw; if 0: loop rn2(9) up to 2 more times  (+0–3)
    rn2(80)                # +1 draw; if 0: loop rn2(9) up to 2 more times  (+0–3)
rn2(1000)                  # +1 draw (greased check)
```

**Expected draws:** ~1.03 (dominated by the `rn2(100)` + `rn2(1000)`, rare loops)
**Worst case:** 1+1+3+1+3+1 = **10 draws** (both erosion types loop to max)

At Dlvl 1 `may_generate_eroded` returns FALSE for initial hero inventory (`svm.moves <= 1 && !gi.in_mklev`), so **0 erosion draws for starting kit**. In-level generation always incurs erosion draws.

---

## 3. Per-class `mksobj_init` Draw Cascade

### WEAPON_CLASS (`mkobj.c:877–893`)
```
quan:   rn1(6,6) if multigen else 0             # 1 draw (arrows/darts/etc.)
        rn2(11)                                 # 1 draw
        branch 1 (!rn2(11)=0): rne(3) + rn2(2) # +1–6 draws
        branch 2 (!rn2(10)=0): rne(3)          # +1–5 draws
        branch 3 (else):       blessorcurse(10) # +1–2 draws
        rn2(100) poison check                   # +1 draw
        rn2(20+…) artifact check                # +1 draw
```
**Min:** 3 draws (non-multigen, branch 3, no poison, no artifact)
**Max (Dlvl1):** 1+1+5+1+1 ≈ **~9 draws** (multigen + branch 1 rne maxes)
**Expected:** ~4.5 draws (typical single weapon, branch 3 hits ~82%)

### ARMOR_CLASS (`mkobj.c:1086–1114`)
```
rn2(10)                                         # 1 draw
  if non-zero: rn2(11) inner                    # +1 draw
    branch 1 (inner=0):  rne(3)                 # +1–5 draws
    branch 2 (!rn2(10)): rn2(2) + rne(3)        # +2–6 draws
    branch 3 (else):     blessorcurse(10)        # +1–2 draws
  if zero: !rn2(10) check
    branch 2: rn2(2) + rne(3)                   # +2–6 draws
    branch 3: blessorcurse(10)                   # +1–2 draws
rn2(40+…) artifact check                        # +1 draw
```
**Min:** 3 draws
**Max:** ~10 draws (outer 10=0, elif=0, rne maxes)
**Expected:** ~4.5 draws

### POTION_CLASS & SCROLL_CLASS (`mkobj.c:1079`)
```
blessorcurse(4)   # 1–2 draws
```
**Min/Expected/Max:** 1–2 draws

### SPBOOK_CLASS (`mkobj.c:1083`)
```
blessorcurse(17)  # 1–2 draws
```
**Min/Expected/Max:** 1–2 draws

### WAND_CLASS (`mkobj.c:1116–1127`)
```
rn1(5, offset)    # 1 draw (charges)
blessorcurse(17)  # 1–2 draws
```
**Min:** 2 draws  **Max:** 3 draws

### RING_CLASS — uncharged (`mkobj.c:1143–1148`)
```
rn2(10) + conditional rn2(9)  # 1–2 draws
```
**Min:** 1 draw  **Max:** 2 draws

### RING_CLASS — charged (`mkobj.c:1129–1142`)
```
blessorcurse(3)               # 1–2 draws
rn2(10)                       # 1 draw
  if != 0: rn2(10)            # +1 draw
    bcsign branch: rne(3)     # +1–5 draws
    else: rn2(2) ? rne(3) : rne(3) # +1–6 draws
if spe==0: rn2(4) - rn2(3)   # +2 draws
if spe<0:  rn2(5)             # +1 draw
```
**Min:** 2 draws  **Max:** ~12 draws (charged ring, all branches hit, rne maxes)
**Expected:** ~5 draws

### AMULET_CLASS (`mkobj.c:1063–1069`)
```
rn2(10)                # 1 draw
  if != 0 and special: curse (no draw)
  else: blessorcurse(10) # 1–2 draws
```
**Min:** 1 draw  **Max:** 3 draws  **Expected:** 2 draws

### TOOL_CLASS (selected subtypes, `mkobj.c:988–1058`)
```
CANDLE:         rn2(2) + conditional rn2(7) + blessorcurse(5)   # 2–5 draws
LAMP/LANTERN:   rn1(500,1000) + blessorcurse(5)                  # 2–3 draws
MAGIC_LAMP:     blessorcurse(2)                                   # 1–2 draws
CHEST/LARGE_BOX: rn2(5)+rn2(10)+rn2(100) + mkbox_cnts()         # 3+ draws (recursive!)
MARKER/CAMERA:  rn1(70,30)                                        # 1 draw
CAN_OF_GREASE:  rn1(21,5) + blessorcurse(10)                     # 2–3 draws
CRYSTAL_BALL:   rn1(5,3) + blessorcurse(2)                       # 2–3 draws
HORN/BAG_OF_TRICKS: rn1(18,3)                                     # 1 draw
FIGURINE:       rndmonnum_adj loop + blessorcurse(4)              # 2–5 draws
MAGIC_INSTRUMENTS: rn1(5,4)                                       # 1 draw
LOCK_PICK etc.: 0 draws
```
**Containers (CHEST/LARGE_BOX/BAG_OF_HOLDING):** each calls `mkbox_cnts` which
recursively calls `mkobj` (and thus `mksobj`) for 0–7 contained objects — this
is an unbounded recursive draw cascade, easily 20–60+ draws per container.

### FOOD_CLASS (`mkobj.c:895–975`)
```
CORPSE: rndmonnum() loop (1+ draws) + rn2(2) sex
EGG:    rn2(3) + rndmonnum() loop
TIN:    rn2(6) + rndmonnum() loop + blessorcurse(10)
KELP:   rnd(2)
CANDY:  assign_candy_wrapper (no direct rn2)
default (most food): rn2(6) quan check
```
**Expected (ration/fruit/veggie):** 1 draw.  **Expected (corpse/egg/tin):** 3–10 draws.

### GEM_CLASS (`mkobj.c:977–986`)
```
LOADSTONE: curse (no draw)
ROCK:      rn1(6,6)   # 1 draw
other:     rn2(6)     # 1 draw
```
**Min/Max:** 0–1 draw

---

## 4. Weighted Average Per Random Object (Dlvl 1, mkobjprobs)

Using class weights from `mkobjprobs` (sums to 100):

| Class | Weight | Expected draws | Contribution |
|-------|--------|---------------|--------------|
| FOOD (20%) | 20 | 1.5 | 0.30 |
| POTION (16%) | 16 | 1.3 | 0.21 |
| SCROLL (16%) | 16 | 1.3 | 0.21 |
| ARMOR (11%) | 11 | 4.5 | 0.50 |
| WEAPON (10%) | 10 | 4.5 | 0.45 |
| TOOL (8%) | 8 | 3.5 | 0.28 |
| GEM (7%) | 7 | 1.0 | 0.07 |
| SPBOOK (4%) | 4 | 1.3 | 0.05 |
| WAND (4%) | 4 | 2.5 | 0.10 |
| RING (3%) | 3 | 3.5 | 0.11 |
| AMULET (1%) | 1 | 2.0 | 0.02 |

**Weighted class-pick draw:** +1 (the `rnd(100)` class roll in `mkobj`)
**Weighted type-pick draw within class:** +1 (`rnd(oclass_prob_totals)`)
**Erosion draw:** +~1.05 (always fires in-level)

**Total expected per random object (Dlvl 1):** ~1 + 1 + 2.30 + 1.05 ≈ **5.4 draws**
**Worst case single object:** ~12 draws (charged ring or complex weapon)
**Absolute worst (container):** 60–100+ draws (nested containers with contents)

---

## 5. `fill_ordinary_room` Object Call Frequency (Dlvl 1)

Per `fill_ordinary_room` (`mklev.c:939–1171`):

| Source | Calls | Condition |
|--------|-------|-----------|
| `mkobj_at(RANDOM_CLASS, …)` — first | 1 | `!rn2(3)` (33%) |
| `mkobj_at(RANDOM_CLASS, …)` — extras | 0–N | `while !rn2(5)`, avg 0.25 extra |
| `mksobj_at(CHEST or LARGE_BOX)` | 0–1 | `!rn2(nroom*5/2)` ≈ `!rn2(10)` at 4 rooms |
| `mkgold(0L,…)` | 0–1 | `!rn2(3)` |
| `mkcorpstat(STATUE,…)` | 0–1 | `!rn2(20)` |
| supply chest (bonus room only) | 2–6 | once per level, extra mksobj calls |

**Dlvl 1 baseline** (`depth=1`, `nroom` typically 4–8):
- Rooms per Dlvl 1: `rnd((nroom>>1)+1)` niches + fillable rooms. MAXNROFROOMS=40; Dlvl 1 typical nroom ≈ 5–8.
- Per room: 0.33 random objects avg + 0.1 chests + 0.05 statues.
- For 6 rooms: ~0.33×6 = **~2 random objects** placed per level via `fill_ordinary_room`.
- Each chest recursively spawns 0–5 contained objects.

**`fill_ordinary_room` total mksobj calls per Dlvl 1:** ~3–5 top-level calls, each ~5 draws = **~15–27 ISAAC64 draws for floor objects** (excluding monster inventories, shops, special rooms).

---

## 6. What Nethax Currently Models

| Pipeline | Status |
|---------|--------|
| `_consume_ini_inv_rogue_draws` (character.py:1145) | Models all 7 Rogue starting items fully, including WEAPON/ARMOR/POTION/SACK branches |
| `mkobj_random` (random_objects.py) | Models class selection (`rnd(100)`) and type selection within class — **no blessorcurse, no spe/rne, no erosion, no quantity draws** |
| `fill_ordinary_room` floor object placement | **Not modeled at all** |
| Monster inventory generation | **Not modeled at all** |
| Container contents (`mkbox_cnts`) | **Not modeled at all** |
| `mkgold` | **Not modeled** |

---

## 7. Deficit Summary

**Per Dlvl 1 level generation**, the unmodeled `mksobj_init` cascade + erosion draws
represent approximately:

- **Floor objects:** ~3–5 objects × 4.4 missing draws each ≈ **~13–22 untracked draws**
- **Monster inventories:** varies wildly; each monster with inventory calls mksobj ~1–4× per item
- **Containers:** each chest/box adds recursive draws; 1 chest ≈ 10–30 additional draws

**Total estimated unmodeled draw deficit for a full Dlvl 1 level generation:** **50–150 ISAAC64 draws**, dominated by monster inventories and containers.

The `mkobj_random` landing (random_objects.py) handles class and type selection correctly
but stubs quantity=1 and omits all `mksobj_init` RNG (blessorcurse, spe/rne, erosion,
subtype-specific rolls). This is the primary gap to close for byte-exact vendor replay.
