# TOOL_CLASS + CONTAINER mksobj Cascade Audit

**Source:** `vendor/nle/src/mkobj.c` (NetHack 3.6, rev 1.157)
**Scope:** TOOL_CLASS per-otyp RNG draws; CONTAINER `mkbox_cnts` recursive cascade.
**Mode:** Read-only audit. No code changes.

---

## 1. TOOL_CLASS Per-otyp Draw Counts

All line numbers reference `vendor/nle/src/mkobj.c`.

| otyp | Line(s) | RNG calls | Draw count |
|------|---------|-----------|------------|
| TALLOW_CANDLE / WAX_CANDLE | 905–906 | `rn2(2)` + conditional `rn2(7)` + `blessorcurse(5)` | 2–4 draws |
| BRASS_LANTERN / OIL_LAMP | 911–913 | `rn1(500,1000)` + `blessorcurse(5)` | 2–3 draws |
| MAGIC_LAMP | 918 | `blessorcurse(2)` | 1–2 draws |
| CHEST / LARGE_BOX | 922–929 | `rn2(5)` (olocked) + `rn2(10)` (otrapped) + `mkbox_cnts()` | 2 + recursive (see §2) |
| ICE_BOX / SACK / OILSKIN_SACK / BAG_OF_HOLDING | 925–929 | `mkbox_cnts()` only | recursive (see §2) |
| EXPENSIVE_CAMERA / TINNING_KIT / MAGIC_MARKER | 934 | `rn1(70, 30)` | **1 draw** |
| CAN_OF_GREASE | 937–938 | `rnd(25)` + `blessorcurse(10)` | 2–3 draws |
| CRYSTAL_BALL | 941–942 | `rnd(5)` + `blessorcurse(2)` | 2–3 draws |
| HORN_OF_PLENTY / BAG_OF_TRICKS | 946 | `rnd(20)` | **1 draw** |
| FIGURINE | 951–953 | `rndmonnum()` loop (1–30 draws) + `blessorcurse(4)` | 2–31 draws |
| BELL_OF_OPENING | 956 | `spe = 3` (constant) | **0 draws** |
| MAGIC_FLUTE / MAGIC_HARP / FROST_HORN / FIRE_HORN / DRUM_OF_EARTHQUAKE | 963 | `rn1(5, 4)` | **1 draw** |
| All other tools (LOCK_PICK, KEY, etc.) | — | no case match, falls through | **0 draws** |

### Corrections vs task brief

The brief assumed vendor-NLE 3.7 line numbers. This file is rev 1.157 (3.6):

- **BAG_OF_TRICKS:** `rnd(20)` at line 946 (not `rn1(20,1)` — range is 1–20, same distribution as `rn1(20,1)` — identical draw count: 1).
- **MAGIC_MARKER:** `rn1(70, 30)` at line 934 (range 30–99, **not** `rnd(50)` — 1 draw either way).
- **OIL_LAMP / BRASS_LANTERN:** `rn1(500, 1000)` at line 911 (range 1000–1499) + `blessorcurse(5)`. **MAGIC_LAMP has no age/spe charge, only `blessorcurse(2)`.** LANTERN is the same case as OIL_LAMP.
- **BAG_OF_HOLDING:** no dedicated blessorcurse — falls into the `mkbox_cnts()` branch at line 928–929. The brief noted "blessorcurse only" but the actual code calls `mkbox_cnts(otmp)` with n=1, so BAG_OF_HOLDING gets 0–1 contained items.
- **IRON_SAFE:** does not exist in this codebase (no grep hits anywhere in vendor/).

---

## 2. CONTAINER `mkbox_cnts` Recursive Cascade

### 2a. Count (`n`) selection — `mkobj.c:283–309`

```
switch (box->otyp):
  ICE_BOX:    n = 20
  CHEST:      n = (olocked ? 7 : 5)
  LARGE_BOX:  n = (olocked ? 5 : 3)
  SACK / OILSKIN_SACK:
              if (moves <= 1 && !in_mklev): n = 0; break
              /* FALLTHRU */
  BAG_OF_HOLDING: n = 1
  default:    n = 0

actual_items = rn2(n + 1)   # line 309: uniform 0..n inclusive
```

**Draw counts for count selection:**
- CHEST/LARGE_BOX: 0 draws (n is determined by `olocked` set before `mkbox_cnts` call).
- ICE_BOX/BAG_OF_HOLDING/SACK: 0 draws for n assignment.
- `rn2(n+1)`: **1 draw always** — gives actual item count.

### 2b. Per-item selection — `mkobj.c:309–352`

For each of the `actual_items` drawn above:

**ICE_BOX path (line 310–318):**
```
mksobj(CORPSE, TRUE, TRUE)   # full mksobj_init cascade for CORPSE
  → rndmonnum() 1+ draws + rn2(2) sex + erosion draws
```
Expected: ~4 draws per corpse.

**All other containers (line 321–349):**
```
rnd(100)              # class pick from boxiprobs (line 324) — 1 draw
  → iprob subtraction loop (no extra draws; linear scan)
mkobj(iclass, TRUE)   # full recursive mksobj + mksobj_init — 3–12 draws
  if COIN_CLASS:
    rnd(level_difficulty() + 2) * rnd(75)   # 2 draws for gold qty
  if ROCK:
    rnd_class(...)    # 1 draw; if quan > 2: assign quan=1 (no draw)
  if BAG_OF_HOLDING (otmp is bag):
    Is_mbag check → force SACK if true (no draw)
    else while WAN_CANCELLATION: rnd_class(WAN_LIGHT, WAN_LIGHTNING)  # 1+ draws
```

**Per-item draw breakdown (non-ICE_BOX, non-COIN, non-ROCK):**
- 1 draw (class pick `rnd(100)`)
- ~3–8 draws (`mksobj_init` for chosen class — see MKSOBJ_AUDIT.md §3)
- ~1 draw (erosion, always in-level)
- **Expected per item: ~5–6 draws**

### 2c. Locking/trapping for CHEST/LARGE_BOX — `mkobj.c:922–923`

These are set **before** `mkbox_cnts` is called, so they count against the container object init, not inside `mkbox_cnts`:

```
otmp->olocked  = !!(rn2(5));    # line 922 — 1 draw; P(locked)=4/5=80%
otmp->otrapped = !(rn2(10));    # line 923 — 1 draw; P(trapped)=1/10=10%
```

Note: `tknown` is **not set** in `mkbox_cnts` or `mksobj`; there is no `tknown` draw in this codebase.

### 2d. Full cascade summary table

| Container | olocked draw | otrapped draw | count draw | items (expected) | draws per item | Total expected draws |
|-----------|-------------|--------------|------------|-------------------|----------------|----------------------|
| CHEST (unlocked, n=5) | 1 | 1 | 1 | rn2(6)=2.5 avg | 5–6 | **16–19** |
| CHEST (locked, n=7) | 1 | 1 | 1 | rn2(8)=3.5 avg | 5–6 | **21–25** |
| LARGE_BOX (unlocked, n=3) | 1 | 1 | 1 | rn2(4)=1.5 avg | 5–6 | **11–13** |
| LARGE_BOX (locked, n=5) | 1 | 1 | 1 | rn2(6)=2.5 avg | 5–6 | **16–19** |
| ICE_BOX (n=20) | 0 | 0 | 1 | rn2(21)=10 avg | ~4 (corpse) | **~41** |
| BAG_OF_HOLDING (n=1) | 0 | 0 | 1 | rn2(2)=0.5 avg | 5–6 | **~4** |
| SACK (n=1, in_mklev) | 0 | 0 | 1 | rn2(2)=0.5 avg | 5–6 | **~4** |

**Worst-case single CHEST:** 80% locked → n=7 → `rn2(8)=7` items → 7 items × 6 draws + 3 overhead = **~45 draws**. If any contained item is itself a BAG_OF_HOLDING, add ~4 more draws recursively.

---

## 3. Per-Level CONTAINER Frequency Estimate (Dlvl 1)

From `fill_ordinary_room` (mklev.c, referenced in MKSOBJ_AUDIT.md §5):

```
P(chest per room) ≈ !rn2(nroom * 5/2) ≈ !rn2(10) = 10%
```

With ~6 fillable rooms at Dlvl 1:
- **Expected chests per Dlvl 1:** 6 × 0.10 = **~0.6 chests**
- At 0.6 × 18 expected draws per chest ≈ **~11 draws** from container contents per Dlvl 1 on average.
- In a worst-case level (3+ chests): **40–135 draws** from containers alone.

Containers are rare enough that the *expected* impact is modest, but the *variance* is very high — a single locked ICE_BOX would consume ~41 draws at once.

---

## 4. Recommended Implementation Strategy

### TOOL_CLASS

Most tool subtypes are straightforward: 0–3 scalar draws. Implement as a `jax.lax.switch` on `otyp`:

```python
# Pseudocode
def tool_init(otyp, rng):
    rng, k1 = split(rng)
    spe = lax.switch(otyp_index, [
        lambda k: rnd(k, 20),          # BAG_OF_TRICKS / HORN_OF_PLENTY
        lambda k: rn1(k, 70, 30),      # MAGIC_MARKER / CAMERA / TINNING_KIT
        lambda k: rnd(k, 5),           # CRYSTAL_BALL
        ...
    ], k1)
    rng, k2 = split(rng)
    b = blessorcurse(otyp_boc_chance[otyp], k2)
    return spe, b, rng
```

FIGURINE is special: the `rndmonnum()` loop needs a `lax.while_loop` (bounded at 30 tries).

### CONTAINER (mkbox_cnts)

The recursive cascade requires `jax.lax.fori_loop` with a **carried RNG + content array**:

```python
def mkbox_cnts(box_otyp, olocked, rng):
    n_max = lax.switch(box_otyp_index, [20, 7 if olocked else 5, 5 if olocked else 3, 1, 1, 0], ...)
    rng, k = split(rng)
    n_items = rn2(k, n_max + 1)   # 1 draw

    def body(i, carry):
        items, rng = carry
        rng, k = split(rng)
        iclass = class_pick_boxiprobs(k)   # 1 draw via rnd(100) lookup
        rng, item, rng = mksobj_init(iclass, rng)  # recursive — 3-12 draws
        items = items.at[i].set(item)
        return items, rng

    items, rng = lax.fori_loop(0, n_items, body, (empty_items, rng))
    return items, rng
```

Key constraints:
1. **Static upper bound:** `fori_loop` must run to `n_max` (max 20 for ICE_BOX, 7 for locked CHEST). Mask out items at index >= `n_items` with a validity flag.
2. **No nested recursion:** BAG_OF_HOLDING inside a CHEST calls `mkbox_cnts` again (n=1). This can be handled by inlining one level of recursion (bag-in-chest: max 1 item, safe to inline). Prevent bag-in-bag by the existing vendor guard (line 342–345: force SACK otyp).
3. **boxiprobs class pick:** Linear subtraction scan over 9 entries — implement as a fixed 9-step cumulative comparison, no loop needed.
4. **ICE_BOX corpse items:** Path is distinct — call `mksobj(CORPSE)` directly, skipping the `rnd(100)` class pick.

### Draw budget per container in JAX

Pre-allocate a fixed RNG key buffer per container type:
- CHEST: 3 (overhead) + 7 × 8 (items) = **59 keys max**
- ICE_BOX: 1 + 20 × 6 = **121 keys max**
- BAG_OF_HOLDING: 1 + 1 × 8 = **9 keys max**

Use `jax.random.split(rng, max_keys)` upfront, then index into the pre-split array inside `fori_loop` to keep the carry state purely positional.
