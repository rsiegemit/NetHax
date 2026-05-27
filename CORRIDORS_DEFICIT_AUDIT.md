# RNG Deficit Audit: Corridors, dodoor, dig_corridor

**Scope:** vendor `mklev.c` + `sp_lev.c` vs `Nethax/nethax/dungeon/corridors.py`
**Baseline setup:** 5-corridor Dlvl 1 level (6 rooms, nroom=6).
**Vendor branch:** NetHack 3.6 (`vendor/nle/src/`).

---

## 1. Per-function vendor RNG draw counts

### 1a. `finddpos` (mklev.c:69-96)
Two `rn1` calls (= two `rn2` draws each, so 2 raw draws per call).

- **Vendor:** 2 draws per call.
- **Port:** 2 draws per call — **MATCHES**.

### 1b. `dodoor` / `dosdoor` (mklev.c:1249-1260 + 383-447) — per call, Dlvl 1

Vendor call stack: `dodoor` → `dosdoor(type=rn2(8)?DOOR:SDOOR)`.

| Draw | Condition in vendor (Dlvl 1) | Vendor draws? | Port draws? |
|------|------------------------------|---------------|-------------|
| `rn2(8)` | always | YES | YES |
| `rn2(3)` | type==DOOR (7/8 chance) | **conditional** | **UNCONDITIONAL** |
| `rn2(5)` (open?) | type==DOOR && rn2(3)==0 | **conditional** | **UNCONDITIONAL** |
| `rn2(6)` (lock?) | type==DOOR && rn2(3)==0 && !open | **conditional** | **UNCONDITIONAL** |
| `rn2(25)` (trap?) | level_difficulty()>=5 — FALSE at Dlvl 1 | **SKIPPED** | DRAWS (then `del`) |
| `rn2(5)` (SDOOR lock) | type==SDOOR (1/8 chance) | **conditional** | **UNCONDITIONAL** |
| `rn2(20)` (SDOOR trap) | difficulty>=4 — FALSE at Dlvl 1 | **SKIPPED** | DRAWS (then `del`) |

**Vendor expected draws per dodoor call (Dlvl 1):**
- DOOR path (p=7/8): 1 + 1 + (0 or 1) + (0 or 1) + 0 = **2–4 draws**
- SDOOR path (p=1/8): 1 + 1 + 0 = **2 draws**
- Typical average (DOOR path, all subbranches): ~**3 draws**

**Port draws per dodoor call (always):** `rn2(8)` + `rn2(3)` + `rn2(5)` + `rn2(6)` + `rn2(25)` + `rn2(5)` + `rn2(20)` = **7 draws unconditionally**.

**Deficit per dodoor call (Dlvl 1):**
- Port over-draws by **+4 draws** on average (7 vs ~3).
- Per 5-corridor level with 2 doors each = 10 dodoor calls: **+40 draws** over-consumption.

### 1c. `dig_corridor` (sp_lev.c:2215-2322) — per step in the walker loop

Vendor draws per iteration (nxcor=FALSE on main corridors):

| Draw | Condition |
|------|-----------|
| `rn2(35)` | `nxcor` only → **skipped** on main corridors |
| `rn2(100)` | `crm->typ == btyp` (cell is STONE) — typical case | **drawn** |
| `rn2(50)` | `nxcor` only → **skipped** on main corridors |
| `rn2(dix-diy+1)` | `dix>diy && diy` — direction bias | **conditional** |
| `rn2(diy-dix+1)` | `diy>dix && dix` — direction bias | **conditional** |

Port (`dig_corridor` body_fn) draws per iteration:

| Draw | Condition in port |
|------|-------------------|
| `rn2(35)` | **UNCONDITIONAL** (then gated by `nxcor &` in logic) |
| `rn2(100)` | UNCONDITIONAL |
| `rn2(50)` | **UNCONDITIONAL** (then `del r50`) |
| `rn2(dix-diy+1)` | UNCONDITIONAL (clamped to ≥1) |
| `rn2(diy-dix+1)` | UNCONDITIONAL (clamped to ≥1) |

**Per-step deficit on main corridors (nxcor=FALSE):**
- Vendor: 1 + 0–2 direction draws = ~1–3 draws/step
- Port: 5 draws/step always
- Over-consumption: **+2 to +4 draws per corridor step**.

For a typical 10-cell straight corridor (nxcor=FALSE):
- Vendor: ~20 draws (rn2(100) each cell + ~1 direction draw each)
- Port: ~50 draws (5 per cell)
- **Per-corridor deficit: ~+30 draws**

### 1d. `join` (mklev.c:243-316) — per call

`join` calls: `finddpos×2` + `dodoor×0-2` + `dig_corridor`.

Vendor per join (nxcor=FALSE, typical corridor succeeds):
- 2 × finddpos = **4 draws**
- 2 × dodoor ≈ **6 draws** average
- 1 × dig_corridor (10-cell) ≈ **20 draws**
- **Total: ~30 draws**

Port per join:
- 2 × finddpos = 4 draws
- 2 × dodoor = **14 draws** (7 each, unconditional)
- 1 × dig_corridor = **50 draws** (5/step × 10 steps)
- **Total: ~68 draws**

**Per-join over-consumption: ~+38 draws**

### 1e. `makecorridors` (mklev.c:319-348) — full 5-corridor scenario

For nroom=6, Pass 1 produces 5 joins + 1 `rn2(50)` early-bail draw per join.

| Pass | Vendor joins | Vendor `rn2` overhead | Port joins | Port overhead |
|------|-------------|----------------------|-----------|---------------|
| P1 | 5 joins + 5×rn2(50) | 5 draws | 5 joins + 5×rn2(50) | 5 draws |
| P2 | 0–2 joins | 0 | 0–2 joins | 0 |
| P3 | 0 (connected) | 0 | 0 (connected) | 0 |
| P4 | rn2(6)+4 extra joins, 2×rn2 each | ~16+rn2 draws | same | same |

P4 extra joins (nxcor=TRUE): vendor `rn2(35)` fires per step (early bail more likely), port draws it unconditionally = same overhead per call. Main deficit remains in the dodoor+dig_corridor body.

**Total makecorridors over-consumption estimate (Dlvl 1, 5 corridors + ~5 extra P4):**

| Component | Vendor draws | Port draws | Deficit |
|-----------|-------------|-----------|---------|
| finddpos (all joins, 10 total) | 20 | 20 | 0 |
| dodoor (10 main + ~10 extra) | ~60 | 140 | **+80** |
| dig_corridor (10 × ~10 steps) | ~200 | ~500 | **+300** |
| makecorridors RNG overhead (rn2(50)×5, rn2(nroom)×1, rn2(nroom-2)×1+4 extras) | ~12 | ~12 | 0 |
| **TOTAL** | **~292** | **~672** | **~+380** |

### 1f. `add_door` (mklev.c:350-381)

Pure bookkeeping (array shuffle). **Zero RNG draws in vendor. Zero in port. No deficit.**

### 1g. `do_room_or_subroom` (mklev.c:110-184)

No `rn2` calls. Zero draws in vendor and port. **No deficit.**

---

## 2. Per-function deficit summary table

| Function | Vendor draws (Dlvl 1) | Port draws | Deficit |
|----------|-----------------------|-----------|---------|
| `finddpos` (per call) | 2 | 2 | 0 |
| `dodoor` (per call) | ~3 | 7 | **+4** |
| `dig_corridor` (per step) | ~2 | 5 | **+3** |
| `add_door` | 0 | 0 | 0 |
| `do_room_or_subroom` | 0 | 0 | 0 |
| **makecorridors (5-corridor level)** | **~292** | **~672** | **~+380** |

---

## 3. Root causes and top 3 fixes

### Fix 1 (highest impact): `dodoor` — use `lax.cond` for DOOR vs SDOOR branches

**Impact: ~+40 draws per 5-corridor level (10 dodoor calls × 4 draws each)**

Vendor only draws `rn2(3)`, `rn2(5)`, `rn2(6)` when `type==DOOR`; only draws `rn2(5)`, `rn2(20)` when `type==SDOOR`. The port draws all 6 sub-rolls unconditionally every time.

Fix: wrap the DOOR sub-rolls in `lax.cond(type_is_door, draw_door_rolls, lambda r: (r, defaults), r)` and the SDOOR sub-rolls in `lax.cond(~type_is_door, ...)`. Also: `rn2(25)` and `rn2(20)` must NOT be drawn at Dlvl 1 at all (difficulty gates are false) — remove those draws entirely for Dlvl 1, or pass `level_difficulty` and guard with `lax.cond(difficulty >= 5, ...)`.

### Fix 2 (largest absolute deficit): `dig_corridor` — guard `rn2(35)` and `rn2(50)` behind `nxcor`

**Impact: ~+200 draws per 5-corridor level (2 wasted draws × ~10 steps × 10 corridors)**

Vendor only draws `rn2(35)` and `rn2(50)` when `nxcor=TRUE`. The port draws both unconditionally every step (then ignores them via `& nxcor`). Wrap them in `lax.cond(nxcor, lambda r: rn2_jax(r, 35), lambda r: (r, jnp.int32(1)), r)`.

### Fix 3 (medium): `dig_corridor` — direction-bias draws `rn2(dix-diy+1)` / `rn2(diy-dix+1)` are always drawn

**Impact: ~+100 draws per 5-corridor level**

Vendor only draws `rn2(dix-diy+1)` when `dix>diy && diy!=0`, and `rn2(diy-dix+1)` only in the `else if` branch. The port draws both unconditionally (clamped to ≥1). Fix with two `lax.cond` guards: `lax.cond((dix > diy) & (diy != 0), lambda r: rn2_jax(r, dix-diy+1), lambda r: (r, jnp.int32(1)), r)` and analogously for the y-bias — crucially the y-bias must be in an `else-if` (only drawn when x-bias condition was false).

---

## 4. Files to change

- `/Users/rsiegelmann/Downloads/Projects/nethax/Nethax/nethax/dungeon/corridors.py`
  - `dodoor` function (~lines 509-593): add `lax.cond` guards for DOOR/SDOOR sub-branches and remove unconditional `rn2(25)`/`rn2(20)` draws at Dlvl 1.
  - `dig_corridor` body_fn (~lines 661-716): guard `rn2(35)` and `rn2(50)` behind `nxcor`, and guard both direction-bias draws behind their respective vendor conditions.
