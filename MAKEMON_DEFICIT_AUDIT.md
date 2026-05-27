# makemon() Per-Monster RNG Deficit Audit

**Context:** vendor mklev consumes ~1789 ISAAC64 draws/level vs our ~107.
This audit focuses on the per-monster draw gap in `makemon()`.

---

## Vendor draws per `makemon(NULL, x, y, MM_NOGRP)` call

| Step | Source | Draws |
|------|--------|-------|
| `rndmonst()` — `rnd(choice_count)` weighted pick | makemon.c:1591 | **1** |
| `adj_lev()` — no RNG, pure arithmetic | makemon.c:1757 | 0 |
| `newmonhp()` — `d(m_lev, 8)` = `m_lev × rn2(8)+1` | makemon.c:1011 | **m_lev** (typ. 1–3 for Dlvl1) |
| `mtmp->female = rn2(2)` — gender (non-neuter only) | makemon.c:1226 | **1** |
| `peace_minded()` — 2× `rn2` for co-aligned monsters | makemon.c:2039–2041 | **0–2** (≈1 avg) |
| S_JABBERWOCK/S_NYMPH: `rn2(5)` sleep roll | makemon.c:1265 | **0–1** |
| S_DEMON/S_NDEMON in mklev: `rn2(5)` sleep roll | makemon.c:1321 | **0–1** |
| PM_ANGEL: `rn2(3)` emin check | makemon.c:1355 | **0–1** |
| `rn2(2)` G_SGROUP group trigger | makemon.c:1370 | **1** (if G_SGROUP) |
| `rn2(3)` G_LGROUP group trigger | makemon.c:1373 | **1** (if G_LGROUP) |
| `m_initweap()` — armed monsters (`is_armed = AT_WEAP`) | makemon.c:1381–1382 | **5–30** (class-dependent) |
| `m_initinv()` — always called | makemon.c:1383 | **3–5** (m_lev vs rn2(50/100) × 2 + likes_gold rn2(5)) |
| `m_dowear()` — item wear decisions | makemon.c:1384 | **~3** (per armor slot) |
| `rn2(100)` saddle check | makemon.c:1386 | **1** |
| `mksobj()` per item created — bless/curse/erosion/spe | mkobj.c:805–1041 | **3–8 per item** |

**Typical Dlvl1 unarmed monster (rat, grid bug, newt):**
- rndmonst: 1 + newmonhp: 1 + female: 1 + peace_minded: 1 + m_initinv: 3 + saddle: 1 = **~8 draws**
- Plus `mksobj` calls from `m_initinv` rnd_defensive_item/rnd_misc_item if m_lev > threshold: **+3–8**
- **Subtotal: ~8–16 draws/monster**

**Typical Dlvl1 armed monster (soldier, orc):**
- Above + `m_initweap` (mercenary path): ~15–25 extra rn2 calls
- Plus 5–10 `mksobj` calls × 3–8 draws each = **~25–60 extra draws**
- **Subtotal: ~35–75 draws/monster**

---

## Nethax `spawning.py` draws per monster

| Step | Code | Draws |
|------|------|-------|
| `pick_monster_for_level()` — `isaac_weighted_choice` | spawning.py:563–565 | **1** |
| `_roll_hp()` — `d(level, 8)` via scan (level draws) | spawning.py:658–672 | **level** (typ. 1–3) |
| `_roll_hp()` — extra `rnd(4)` draw always consumed | spawning.py:670 | **1 (wasted)** |
| gender roll | **MISSING** | 0 |
| `peace_minded()` | precomputed table, no RNG | 0 |
| `m_initweap()` equivalent | **MISSING** | 0 |
| `m_initinv()` equivalent (rn2(50)/rn2(100) checks) | **MISSING** | 0 |
| `m_dowear()` | **MISSING** | 0 |
| `mksobj` per item | **MISSING** | 0 |

**Nethax draws per monster: ~2–4**

---

## Deficit Summary

| Scenario | Vendor draws | Nethax draws | Deficit/monster |
|----------|-------------|--------------|-----------------|
| Unarmed Dlvl1 (rat, newt) | ~10 | ~3 | **~7** |
| Armed Dlvl1 (orc, soldier) | ~50 | ~3 | **~47** |
| Weighted avg (mostly unarmed) | ~15 | ~3 | **~12** |

At 8–12 monsters per level: **~96–144 draws deficit from spawning alone**.

The remaining ~1500-draw gap (1789 total − ~107 current − ~120 spawn deficit) is
in room/corridor generation, object placement, and other mklev subsystems — but
spawning is a clear and measurable contributor.

---

## Top 3 Missing Draw Types to Add

1. **`m_initweap()` + `mksobj()` draws** — Armed monsters (orcs, soldiers, centaurs,
   giants, mercenaries) consume 15–25 rn2 calls in `m_initweap` plus 3–8 draws per
   item created via `mksobj` (bless/curse/erosion/enchantment/quantity rolls).
   This is the single largest per-monster draw source for non-trivial monsters.

2. **`m_initinv()` unconditional rn2 checks** — Every monster (after soldier guard)
   hits `rn2(50)` and `rn2(100)` checks for defensive/misc items, plus `rn2(5)`
   for gold-lovers. These fire for all monsters including rats/newts and contribute
   ~3 draws unconditionally regardless of monster type.

3. **`mtmp->female = rn2(2)` gender roll** — One draw per non-neuter, non-leader,
   non-nemesis monster (the majority of Dlvl1 spawns). Currently absent from
   `spawning.py`; easy fix, adds 1 draw per non-neuter monster.
