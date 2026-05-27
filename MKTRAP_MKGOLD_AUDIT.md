# RNG Cascade Audit: mktrap / mkgold / makedog

**Source:** `vendor/nle/src/mklev.c::mktrap` (lines 1274–1533),
`vendor/nle/src/mkobj.c::mkgold` (lines 1486–1504),
`vendor/nle/src/dog.c::makedog` + `pet_type` (lines 57–201).

---

## 1. `mktrap` — Internal Draw Count

The function lives in **mklev.c** (not trap.c — trap.c has no mktrap).

| Draw | Vendor code | Condition |
|------|-------------|-----------|
| `rn2(7)` — rogue-level trap type | line 1291 | `Is_rogue_level` path only |
| `rnd(TRAPNUM - 1)` — normal trap type | line 1319 | default path; loops until `!= NO_TRAP` |
| HOLE rejection: `rn2(7)` — extra retry | line 1362 | inside do-while, per HOLE draw |
| **Dead-predecessor block** (fired when `lvl <= rnd(4)`) | lines 1418–1533 | shallow levels only |
| └─ `rnd(4)` — level-difficulty gate | line 1418 | unconditional within block |
| └─ `mksobj(ARROW/DART/ROCK)` — ammo/debris item init | lines 1436–1446 | per trap type; mksobj pulls several rng draws (see §4) |
| └─ `rn2(4)` — possession class pick | line 1463 | |
| └─ `mkobj(poss_class)` — possession object | line 1478 | pulls full mksobj_init cascade |
| └─ `do { } while (!rn2(5))` — extra possession loop | line 1490 | geometric; expected ~1.25 items |
| └─ `rn2(15)` — victim race | line 1493 | |
| └─ `rn2(2)` — elf SLP_GAS exception | line 1501 | only when elf + SLP_GAS_TRAP |
| └─ `rn2(10)` — gnome candle roll | line 1514 | only when gnome victim |
| └─ `rn2(4)` — candle type | line 1515 | only when candle fires |
| └─ `mkcorpstat(CORPSE, …)` — corpse init | line 1529 | calls mksobj internally |

**Total per-trap (normal path, shallow, non-special victim):** ~5–10 draws beyond the type-selection draw.

---

## 2. `mkgold` — Internal Draw Count

When called with `amount == 0` (the fill_ordinary_rooms path):

| Draw | Vendor code |
|------|-------------|
| `rnd(30 / max(12 - depth, 2))` — multiplier | line 1493 |
| `rnd(level_difficulty() + 2)` — base amount | line 1495 |
| `mksobj_at(GOLD_PIECE, …)` — object init | line 1500 (when no existing pile) |

**Total:** 2 amount draws + mksobj_at object-init draws (~1–3 depending on class).

---

## 3. `makedog` / `pet_type` — Internal Draw Count

| Draw | Vendor code | Condition |
|------|-------------|-----------|
| `rn2(2)` — kitten vs. little dog | dog.c line 66 | only when `urole.petnum == NON_PM` and no `preferred_pet` override |
| `makemon(…, MM_EDOG)` | dog.c line 184 | calls full monster-spawn RNG cascade |
| `mksobj(SADDLE, …)` | dog.c line 191 | only for pony (Knight); pulls saddle mksobj_init |

**Total:** 0–1 draws in `pet_type` (role-dependent) + makemon cascade.

---

## 4. Coverage in Nethax

### mktrap

**Modeled (both Threefry and ISAAC64 paths):**
- Gate draw: `rn2(trap_x)` — ✓ (`trap_step` / `trap_step_isaac`)
- Type draw: `rnd(TRAPNUM - 1)` legalised against depth — ✓ (`_vendor_traptype_rnd` / `_isaac_legalise_trap_kind`)
- Position: `somexy` (x, y) — ✓

**Not modeled:**
- `rnd(4)` dead-predecessor level-gate — **MISSING**
- Dead-predecessor possession cascade (`rn2(4)`, `mkobj`, `rn2(5)` loop) — **MISSING**
- Dead-predecessor corpse cascade (`rn2(15)`, `rn2(2)`, `rn2(10)`, `rn2(4)`, `mkcorpstat`) — **MISSING**
- Rogue-level `rn2(7)` branch — **MISSING** (rogue level not yet implemented; low priority)
- HOLE `rn2(7)` rejection redraws — **MISSING** (loop approximated to single draw)

### mkgold

**Modeled:**
- Gate draw: `rn2(3)` — ✓
- Position: `somexy` — ✓ (ISAAC64 path); Threefry path draws 1 dummy scalar only (no x+y split)

**Not modeled:**
- `rnd(30/…)` multiplier draw — **MISSING**
- `rnd(level_difficulty() + 2)` base-amount draw — **MISSING**
- `mksobj_at(GOLD_PIECE)` object-init draws — **MISSING**

### makedog

**Modeled:**
- Pet type: resolved statically from role table — no `rn2(2)` draw emitted. Roles with `petnum == NON_PM` default to kitten without a coin flip — **1 DRAW MISSING** for those roles (Arch, Barb, Healer, Monk, Priest, Rogue, Tourist, Valkyrie, Wizard).
- Position: adjacency scan (pure-JAX, no RNG draw) — matches vendor when `preferred_pet` forces the type; **diverges for roles where vendor calls `rn2(2)`**.
- HP roll: `_roll_hp(dummy_rng, …)` uses a **hardcoded PRNGKey(0)** — does **not** consume the live RNG stream. This is a deliberate simplification but introduces a fixed, non-vendor HP value.
- Saddle for pony: `mksobj(SADDLE)` draws — **MISSING**.

---

## 5. Deficit Summary

| Function | Expected draws per call | Modeled | Missing |
|----------|-------------------------|---------|---------|
| mktrap type-select | 1 (+ retry) | ✓ (approx) | retry loop exact parity |
| mktrap dead-predecessor block | 5–12 | **0** | **~7–10 draws** |
| mkgold amount | 2 + mksobj | 0 | **2–4 draws** |
| makedog pet_type rn2(2) | 0–1 | 0 | **1 draw** (NON_PM roles) |
| makedog HP roll | live stream | dummy key | **not from live RNG** |
| makedog saddle mksobj | 1–3 | 0 | **1–3 draws** (Knight only) |

**Critical deficit:** The mktrap dead-predecessor cascade (~7–10 draws per shallow-level trap) and mkgold internal amount draws (~2–4) are entirely absent. These fire on dlvl ≤ ~4 for most traps, meaning **every shallow-level trap placement diverges the ISAAC64 stream** from the vendor sequence by 7–14 draws, compounding across rooms.

---

## 6. References

- `vendor/nle/src/mklev.c` lines 1274–1533 — `mktrap`
- `vendor/nle/src/mkobj.c` lines 1486–1505 — `mkgold`
- `vendor/nle/src/dog.c` lines 57–201 — `pet_type`, `makedog`
- `Nethax/nethax/dungeon/rooms.py` lines 1337–1421 — trap-type helpers
- `Nethax/nethax/dungeon/rooms.py` lines 1656–1686 (Threefry), 1756–1782 (ISAAC64) — trap/gold fill
- `Nethax/nethax/env.py` lines 506–586 — `_spawn_starting_pet`
- `Nethax/nethax/subsystems/character.py` lines 822–848 — `STARTING_PET` table
