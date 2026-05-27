# makemon.c Deep RNG Sub-Cascade Audit

**Scope:** Every RNG draw inside `makemon()` and its direct callees, post-`newmonhp`.
**Baseline:** commit `dfe475c` — female + initweap + initinv HP rolls already merged.
**Canonical source:** `vendor/nle/src/makemon.c` lines 1105–1414.

---

## 1. Draw order inside `makemon()` (lines 1105–1414)

| # | Vendor line | Call / expression | Draw count | Coverage in Nethax |
|---|-------------|-------------------|------------|-------------------|
| 1 | 989 (inside `newmonhp`) | `adj_lev(ptr)` — **no RNG**; purely arithmetic on `level_difficulty()` + player level | 0 | N/A |
| 2 | 990–1013 | `newmonhp` special branches: golem (fixed), rider `d(10,8)`, mlevel>49 (fixed), adult dragon `d(m_lev,4)`, zero-level `rnd(4)`, normal `d(m_lev,8)` | 1–10 | **Covered** (`_roll_hp`) — but see deficit §3a |
| 3 | 1226 | `mtmp->female = rn2(2)` (non-neuter, non-leader/nemesis) | 1 | **Covered** |
| 4 | 1236 | `peace_minded(ptr)` — draws `rn2(16+record)` + `rn2(2+abs(mal))` for co-aligned non-minion monsters; all others no draw | 0–2 | **NOT consumed** |
| 5 | 1264 | `S_JABBERWOCK / S_NYMPH`: `rn2(5)` sleep check | 1 | **NOT consumed** |
| 6 | 1299 | `newcham()` for shapechangers — calls `select_newcham_form` which internally calls `rndmonst()` (multiple `rn2` draws); sets `allow_minvent=FALSE` skipping initweap/initinv | variable | **NOT consumed** |
| 7 | 1321 | `in_mklev` ndemon/wumpus/long-worm/giant-eel sleep: `rn2(5)` | 1 | **NOT consumed** |
| 8 | 1345 | `initworm(mtmp, rn2(5))` for PM_LONG_WORM | 1 | **NOT consumed** |
| 9 | 1355 | Angel `!rn2(3)` check — triggers `emin` alloc path | 1 | **NOT consumed** |
| 10 | 1362 | `eminp->min_align = rn2(3) - 1` (angel / priest emin path) | 1 | **NOT consumed** |
| 11 | 1363 | `eminp->renegade = !rn2(3)` (same path) | 1 | **NOT consumed** |
| 12 | 1368 | `set_malign(mtmp)` — **no RNG**; pure arithmetic | 0 | N/A |
| 13 | 1370 | G_SGROUP: `rn2(2)` before calling `m_initsgrp` | 1 | **NOT consumed** |
| 14 | 1373–1376 | G_LGROUP: `rn2(3)` to choose lgrp vs sgrp | 1 | **NOT consumed** |
| 15 | 1381–1382 | `m_initweap(mtmp)` cascade (is_armed gate) | 8 (capped) | **Covered** (capped at 8 + rn2(75)) |
| 16 | 1383 | `m_initinv(mtmp)` unconditional tail: `rn2(50)`, `rn2(100)`, `rn2(5)` | 3 | **Covered** |
| 17 | 1384 | `m_dowear(mtmp, TRUE)` — **no RNG**; selection only | 0 | N/A |
| 18 | 1386 | `!rn2(100)` saddle check for domestic monsters | 1 | **NOT consumed** |

---

## 2. Inside `m_initinv` class-specific draws (lines 575–800) — NOT consumed

The current code consumes only the 3-draw unconditional tail. The class branches before that tail also draw:

| Class branch | Key draws | Consumed? |
|---|---|---|
| `S_HUMAN` mercenary armor loop | up to 8 `rn2(5/3/2)` draws for armor tier + helmet/shield/boots/gloves | No |
| `S_HUMAN` shopkeeper | `rn2(4)` fallthrough | No |
| `S_HUMAN` priest robe | `rn2(7)`, `rn2(3)`, `rn1(10,20)` | No |
| `S_NYMPH` | `rn2(2)` × 2 | No |
| `S_GIANT` stones loop | `rn2(m_lev/2)` iterations × `rnd_class(…)` draws | No |
| `S_LICH` master/arch | `rn2(13)`, `rn2(7)`, `rn2(3)`, `rn2(13)`, `rn2(4)` | No |
| `S_MUMMY` | `rn2(7)` | No |
| `S_QUANTMECH` | `!rn2(20)` | No |
| `S_LEPRECHAUN` | `d(level_difficulty(), 30)` — ~1–30 draws | No |
| `S_GNOME` candle | `rn2(20 or 60)`, `rn2(4)` | No |
| `S_SOLDIER` early-return gate | `rn2(13)` | No |

---

## 3. Deficits by monster type

### 3a. `newmonhp` special cases — partial gap

`_roll_hp` only models the standard `d(m_lev, 8)` and `rnd(4)` paths.
Missing (draws consumed but not equivalent):

- **Adult dragons** (`mlet == S_DRAGON && mndx >= PM_GRAY_DRAGON`): vendor draws `d(m_lev, 4)` — that is `m_lev` dice of d4, not d8. Nethax uses the standard d8 path for all monsters (over-consumes by `m_lev` draws).
- **Riders** (`is_rider`): vendor draws `d(10, 8)` — fixed 10 dice. Nethax uses `MONSTR_DIFFICULTIES[type_id]` as hit-dice count, which may differ.
- **Golems** and **mlevel>49**: fixed HP — Nethax still rolls dice (wrong draw count).

### 3b. `peace_minded` — 0–2 draws skipped

`peace_minded` is called at line 1236 (before initweap) but Nethax does not consume its draws. For co-aligned, non-minion, non-special monsters it calls `rn2(16 + record)` and `rn2(2 + abs(mal))` — two draws. All other categories (always_peaceful, always_hostile, MS_LEADER, MS_GUARDIAN, MS_NEMESIS, race_peaceful, race_hostile, sgn-mismatch, amulet check) return without drawing.

**Deficit:** ~2 draws per co-aligned non-minion monster. These are common at shallow depths (co-aligned races).

### 3c. `m_initweap` cascade — undercounting for many species

The cap of 8 `rn2(2)` draws is a fixed approximation. Actual draw counts:

| Species | Actual draws in initweap | Capped at |
|---|---|---|
| Elf (elf-king) | 11–13 | 8 |
| Soldier/watchman | 5–7 | 8 |
| Angel (humanoid) | 4 fixed + rn2(4) | 8 |
| Orc captain (Uruk-hai) | 7–10 | 8 |
| Default demon/lord | `rnd(14-2*bias)` | 8 |

The cap is **under** for elves and orc-captains (1–5 draw deficit), **over** for simpler species (1–4 extra).

### 3d. `m_initinv` class body — entirely skipped

The 3-draw tail is consumed but the per-class draws (§2 above) are not. For mercenaries (soldiers, guards) this is 5–10 skipped draws. For leprechauns it is up to 30 skipped draws (`d(level_difficulty, 30)`).

### 3e. Monster-type-specific draws in `makemon()` body — entirely skipped

| Draw | Monster(s) | Count |
|---|---|---|
| `rn2(5)` sleep | Jabberwock, Nymph | 1 |
| `rn2(5)` sleep (in_mklev) | ndemon, wumpus, long worm, giant eel | 1 |
| `rn2(5)` for worm tail segments | Long Worm | 1 |
| Angel emin: `rn2(3)` gate + `rn2(3)` align + `rn2(3)` renegade | Angel | 3 |
| G_SGROUP `rn2(2)` | sgroup monsters | 1 |
| G_LGROUP `rn2(3)` | lgroup monsters | 1 |
| `!rn2(100)` saddle | domestic (horses, etc.) | 1 |
| `set_mimic_sym` `rn2(2)` × 1–2 | Mimic (maze/Delphi room) | 1–2 |
| `newcham` shape selection | Doppelganger, Vampire, Sandestin, Chameleon | variable (5–15) |

---

## 4. Summary

| Category | Status | Draw deficit (per spawn) |
|---|---|---|
| `newmonhp` standard path | Covered | 0 |
| `newmonhp` dragon/rider/golem paths | Wrong draw count | ±m_lev |
| `female` draw | Covered | 0 |
| `peace_minded` co-aligned draw | **Missing** | 2 |
| Jabberwock/Nymph sleep `rn2(5)` | **Missing** | 1 |
| in_mklev demon/worm sleep `rn2(5)` | **Missing** | 1 |
| Long Worm `initworm rn2(5)` | **Missing** | 1 |
| Angel emin path | **Missing** | 3 |
| G_SGROUP / G_LGROUP group roll | **Missing** | 1 |
| Saddle check `!rn2(100)` | **Missing** | 1 |
| `newcham` shapechangers | **Missing** | 5–15 |
| `m_initweap` cascade (armed) | Approximate (±5) | ±5 |
| `m_initinv` 3-draw tail | Covered | 0 |
| `m_initinv` class body draws | **Missing** | 0–30 (leprechaun worst) |
| `m_dowear` / `set_malign` / `adj_lev` | No RNG — N/A | 0 |

**Worst offenders by draw deficit:** leprechauns (~30), shapechangers (~15), elves (~5), angels (~3), co-aligned monsters (~2).
