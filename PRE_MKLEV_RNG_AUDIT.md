# Pre-mklev RNG Audit — Rogue-Human-Chaotic-Male, seed=0

Covers all ISAAC64 CORE draws between the **end of `init_objects()`** and the
**first draw inside `mklev()`**, as executed by vendor NLE for the standard
seed-0 game (role/race/gender/alignment all pre-specified).

Vendor pipeline reference: `vendor/nle/src/allmain.c:604-627`

```
init_objects()   ← already accounted for (shuffle_all + GEM jitter + WAN_NOTHING)
role_init()      ← allmain.c:607
init_dungeons()  ← allmain.c:610
init_artifacts() ← allmain.c:613
u_init()         ← allmain.c:615
mklev()          ← allmain.c:627 (via makelevel)
```

---

## 1. `role_init` — `vendor/nle/src/role.c:2011`

All four selections (role, race, gender, alignment) are **pre-specified** via
command-line flags for a seeded game.  The validation branches all take the
"already valid" path and skip the random-pick helpers.

| Line | Call | Fires for Rogue-Human-Chaotic-Male? |
|------|------|-------------------------------------|
| 2026 | `randrole_filtered()` → `rn2(...)` | **NO** — `validrole(flags.initrole)` is true |
| 2036 | `randrace(flags.initrole)` → `rn2(n*100)/100` | **NO** — `validrace()` is true |
| 2051 | `randalign(...)` → `rn2(n)` | **NO** — `validalign()` is true |
| 2070 | `rn2(100) < 50` (quest leader gender) | **NO** — Rogue leader `PM_MASTER_OF_THIEVES` is explicitly male (`is_male(pm)` branch taken) |
| 2091 | `rn2(100) < 50` (nemesis gender) | **NO** — `PM_MASTER_ASSASSIN` is explicitly male |
| 2098 | `randrole(FALSE)` → `rn2(...)` (god fixup loop) | **NO** — Rogue has its own gods (Issek/Mog/Kos), `urole.lgod` is non-NULL |

**`role_init` total CORE draws: 0**

---

## 2. `init_dungeons` — `vendor/nle/src/dungeon.c:714`

Processes the compiled `dungeon` binary (from `dungeon.def`).  RNG draws occur
in three distinct sub-phases.

### 2a. Dungeon-chance gate — per dungeon with `tmpdungeon[i].chance > 0`

```c
// dungeon.c:775-776
if (!wizard && pd.tmpdungeon[i].chance && (pd.tmpdungeon[i].chance <= rn2(100)))
```

From `dungeon.def`, only **"Fort Ludios"** (portal branch to isolated vault) has
a `chance` field.  Every other dungeon in the standard definition has
`chance = 0`.  That is **1 `rn2(100)` draw** for Fort Ludios.

### 2b. Random dungeon depth — `tmpdungeon[i].lev.rand > 0`

```c
// dungeon.c:797-798
dungeons[i].num_dunlevs = (xchar) rn1(pd.tmpdungeon[i].lev.rand,
                                      pd.tmpdungeon[i].lev.base);
```

From `dungeon.def`:

| Dungeon | depth spec | lev.rand | Draw? |
|---------|-----------|----------|-------|
| Dungeons of Doom | (25, 5) | 5 | **YES — `rn2(5)` (via rn1)** |
| Gehennom | (20, 5) | 5 | **YES — `rn2(5)`** |
| Gnomish Mines | (8, 2) | 2 | **YES — `rn2(2)`** |
| The Quest | (5, 2) | 2 | **YES — `rn2(2)`** |
| Sokoban | (4, 0) | 0 | NO |
| Fort Ludios | (1, 0) | 0 | NO (also may be skipped by chance gate) |
| Vlad's Tower | (3, 0) | 0 | NO |
| Elemental Planes | (6, 0) | 0 | NO |

**4 `rn2` draws** from depth randomisation (DoD, Gehennom, Mines, Quest).

### 2c. Special-level chance gates — `init_level` / `place_level`

Each `LEVEL` and `RNDLEVEL` entry in `dungeon.def` that has a `chance` field
goes through `init_level` at `dungeon.c:548`:

```c
if (!wizard && tlevel->chance <= rn2(100))
    return; /* level skipped */
```

`RNDLEVEL` entries have a non-zero chance; plain `LEVEL` entries have
`chance = 0` (always placed) and therefore **do not** draw.

RNDLEVEL entries from `dungeon.def`:

| Name | chance | Draw |
|------|--------|------|
| bigrm (DoD) | 40 | **`rn2(100)`** |
| medusa (DoD) | 4 | **`rn2(100)`** |
| minetn (Mines) | 7 | **`rn2(100)`** |
| minend (Mines) | 3 | **`rn2(100)`** |
| soko1–soko4 (Sokoban) | 2 each | **4 × `rn2(100)`** |

**8 `rn2(100)` draws** from `init_level` chance gates.

### 2d. `place_level` random slot assignment — `dungeon.c:661`

```c
lev->dlevel.dlevel = pick_level(map, rn2(npossible));
```

One `rn2(npossible)` draw per successfully-created special level.  The exact
count depends on which levels survived the chance gates above (2c), but with
all 8 RNDLEVEL draws plus all unconditional LEVEL entries, approximately
**20–30 draws** occur here (one per special level that passes its chance gate
and has >1 placement candidate).  This range varies by seed.

### 2e. Branch attachment — `parent_dlevel` — `dungeon.c:398`

```c
i = j = rn2(num);
```

One `rn2(num)` draw per branch to pick attachment depth, where `num` is the
range of valid parent levels.  From `dungeon.def` there are **8 branches** in
total (Mines, Sokoban, Quest, Fort Ludios, Gehennom, Vlad's Tower, Elemental
Planes + Knox floating fixup).  Not all branches call `parent_dlevel` — only
non-chained ones do; chained branches (`CHAINBRANCH`, `CHAINLEVEL`) derive
their parent from the anchor level.  Approximately **6–7 `rn2` draws** from
branch placement.

### 2f. Tune string — `dungeon.c:917-918`

```c
for (i = 0; i < 5; i++)
    tune[i] = 'A' + rn2(7);
```

**5 `rn2(7)` draws** — always fires, unconditionally.

### `init_dungeons` total CORE draws (approximate)

| Sub-phase | Draws |
|-----------|-------|
| Fort Ludios chance gate | 1 |
| Dungeon depth (4 dungeons) | 4 |
| RNDLEVEL chance gates (8 entries) | 8 |
| `place_level` slot picks | ~20–30 |
| Branch `parent_dlevel` picks | ~6–7 |
| Tune string | 5 |
| **Total** | **~44–55** |

The `place_level` and branch counts are seed-dependent; the minimum guaranteed
fixed draws (Fort Ludios + depths + RNDLEVEL gates + tune) is **18**.

---

## 3. `init_artifacts` — `vendor/nle/src/artifact.c:81`

```c
init_artifacts() {
    memset(artiexist, 0, ...);
    memset(artidisco, 0, ...);
    hack_artifacts();   // artifact.c:57 — pure table fixup, no RNG
}
```

`hack_artifacts()` only rewrites alignment fields in `artilist[]`; no RNG
calls anywhere in this path.

**`init_artifacts` total CORE draws: 0**

---

## 4. `u_init` — `vendor/nle/src/u_init.c:582`

### 4a. HP and energy initialisation

Called at `u_init.c:635-636`:
```c
u.uhp = u.uhpmax = newhp();   // attrib.c:981
u.uen = u.uenmax = newpw();   // exper.c:47
```

For Rogue-Human at `u.ulevel == 0`:
- `newhp()`: `urole.hpadv.inrnd = 0`, `urace.hpadv.inrnd = 0` → **0 rnd() draws**
- `newpw()`: `urole.enadv.inrnd = 0`, `urace.enadv.inrnd = 0` → **0 rnd() draws**

### 4b. Role-switch block — `PM_ROGUE` — `u_init.c:749-756`

```c
case PM_ROGUE:
    Rogue[R_DAGGERS].trquan = rn1(10, 6);  // u_init.c:750  [rn2(10)+6 → 6..15]
    u.umoney0 = 0;
    ini_inv(Rogue);
    if (!rn2(5))                           // u_init.c:753  [BLINDFOLD gate]
        ini_inv(Blindfold);
    knows_object(SACK);
    skill_init(Skill_R);
    break;
```

**Draws in this block (before `ini_inv`):**
1. `rn2(10)` at line 750 — sets dagger quantity (always fires)
2. `rn2(5)` at line 753 — BLINDFOLD gate (always fires)

**Total: 2 draws before `ini_inv` calls.**

### 4c. `ini_inv(Rogue)` — per-item `mksobj` draws

Rogue `trobj` array (`u_init.c:122-131`):

| Slot | otyp | class | `mksobj` init draws |
|------|------|-------|---------------------|
| SHORT_SWORD | fixed | WEAPON_CLASS | `rn2(11)` + branch: `rn2(10)` (if !rn2(11)) **or** `blessorcurse(10)` = `rn2(10)` [+`rn2(2)` if hit] |
| DAGGER ×N | fixed | WEAPON_CLASS | same pattern as SHORT_SWORD |
| LEATHER_ARMOR | fixed | ARMOR_CLASS | `rn2(10)` + branch: if true → `rn2(11)` + curse/`rne(3)` **or** `rn2(10)` + bless/`rne(3)` **or** `blessorcurse(10)` |
| POT_SICKNESS | fixed | POTION_CLASS | `blessorcurse(4)` = `rn2(4)` [+`rn2(2)` if hit] |
| LOCK_PICK | fixed | TOOL_CLASS | no special case in switch → **0 draws** |
| SACK | fixed | TOOL_CLASS | `mkbox_cnts`: `moves<=1 && !in_mklev` → `n=0` → `rn2(1)` = 0 but **1 draw** |

**Minimum draws for `ini_inv(Rogue)` (no blessed/cursed hits):**
- SHORT_SWORD: `rn2(11)`, `rn2(10)`, `rn2(10)` = **3 draws**
- DAGGER: `rn2(11)`, `rn2(10)`, `rn2(10)` = **3 draws**
- LEATHER_ARMOR: `rn2(10)`, `rn2(10)`, `rn2(10)` = **3 draws** (outer + inner + blessorcurse)
- POT_SICKNESS: `rn2(4)` = **1 draw**
- LOCK_PICK: **0 draws**
- SACK: `rn2(1)` = **1 draw**

**Minimum: 11 draws from `ini_inv(Rogue)`.** Each `blessorcurse` hit adds +1
`rn2(2)`, and `rne(3)` adds a geometric series of draws; exact count is
seed-dependent.

### 4d. `ini_inv(Blindfold)` — conditional on line 753

If `rn2(5) == 0`: one `mksobj(BLINDFOLD, TRUE, FALSE)`.  BLINDFOLD is
TOOL_CLASS with no special case in the switch → **0 extra draws**.

### 4e. Race-switch block — `PM_HUMAN` — `u_init.c:804-806`

```c
case PM_HUMAN:
    /* Nothing special */
    break;
```

**0 draws.**

### 4f. `init_attr(75)` — `attrib.c:614`

```c
while (np > 0 && tryct < 100) {
    x = rn2(100);  // attrib.c:627
    ...
}
while (np < 0 && tryct < 100) {
    x = rn2(100);  // attrib.c:646
    ...
}
```

For Rogue: `urole.attrbase = {7, 7, 7, 10, 7, 6}` → sum = 44.  `np = 75 - 44
= 31`.  Each iteration draws `rn2(100)` to pick which attribute to increment;
up to 31 successful increments (each guarded by `ATTRMAX` cap) plus some
retries.  Typical: **31–40 draws** (exact count is seed-dependent via retry
logic).

### 4g. Post-`init_attr` random attribute variation loop — `u_init.c:888-894`

```c
for (i = 0; i < A_MAX; i++)      // A_MAX = 6 attributes
    if (!rn2(20)) {               // u_init.c:888
        int xd = rn2(7) - 2;     // u_init.c:889
        ...
    }
```

**6 `rn2(20)` draws** (always fired, one per attribute).  Each hit adds 1
`rn2(7)` draw.  Typical hits at seed=0: 0–2, so **6–8 total draws**.

### `u_init` total CORE draws (approximate)

| Sub-phase | Draws |
|-----------|-------|
| `newhp()` / `newpw()` | 0 |
| `rn1(10, 6)` dagger quantity | 1 |
| `rn2(5)` BLINDFOLD gate | 1 |
| `ini_inv(Rogue)` — 6 items | ~11–15 |
| `ini_inv(Blindfold)` if fired | 0 |
| `init_attr(75)` | ~31–40 |
| Attr variation loop (A_MAX=6) | ~6–8 |
| **Total** | **~50–65** |

---

## 5. Total pre-mklev draw count (seed=0 Rogue-Human-Chaotic-Male)

| Function | Min draws | Typical draws |
|----------|-----------|---------------|
| `role_init` | **0** | **0** |
| `init_dungeons` | 18 | **~44–55** |
| `init_artifacts` | **0** | **0** |
| `u_init` | 50 | **~50–65** |
| **Grand total** | **~68** | **~94–120** |

These are all draws on the **CORE ISAAC64 stream** that Nethax currently
performs **zero** of between `init_objects` and `mklev`.

---

## 6. Comparison to Nethax current implementation

From `Nethax/nethax/env.py:176-178`, the current pre-mklev block is:

```python
v_state, rng_level    = _vendor_draw_prngkey(v_state)   # 1 draw
v_state, rng_char     = _vendor_draw_prngkey(v_state)   # 1 draw
v_state, rng_monsters = _vendor_draw_prngkey(v_state)   # 1 draw
```

`_vendor_draw_prngkey` consumes **one ISAAC64 64-bit output** (= 2 × `rn2`
draws worth of entropy if each `rn2(N)` costs one 32-bit half).  In vendor C,
`rn2` calls the ISAAC64 engine which returns a 32-bit value; 3 draws of
`_vendor_draw_prngkey` = **3 CORE draws** total.

**Nethax draws 3.  Vendor draws ~94–120.  Gap: ~91–117 draws.**

Since mklev's room and stair placement begins consuming the CORE stream
immediately (via `rnd_rect` → `rn2` inside `makerooms`), this ~90-draw offset
shifts the entire dungeon-gen RNG stream, producing the 99.6% glyph divergence
and `player_x` offset of 25 tiles observed at seed=0.

---

## 7. Missing draws by priority

### Priority 1 — `u_init` attribute distribution (`init_attr`) — **~31–40 draws**

`attrib.c:614::init_attr(75)` is the single largest contributor and is
**completely absent** from Nethax.  The loop draws `rn2(100)` up to ~35 times
to distribute attribute points.  This alone accounts for roughly one-third of
the total gap.

**File:** `vendor/nle/src/attrib.c:626-660`
**Action:** Port `init_attr` as a JAX scan that advances `vendor_rng` by calling
`rn2(100)` until `np == 0` (capped at 100 iterations per phase).

### Priority 2 — `init_dungeons` place_level + branch picks — **~26–37 draws**

The recursive `place_level` calls (`dungeon.c:661`) and `parent_dlevel` branch
picks (`dungeon.c:398`) together contribute ~26–37 draws.  These depend on
how many special levels survive their chance gates, but the fixed part (tune
string: 5 draws, RNDLEVEL gates: 8 draws, depth draws: 4 draws, Fort Ludios
gate: 1 draw) = **18 guaranteed draws** that are trivially reproducible.

**File:** `vendor/nle/src/dungeon.c:398, 548, 661, 776-798, 917-918`
**Action (fast path):** Add a `_consume_init_dungeons_rng` function that
replays the 18 fixed draws (depths + RNDLEVEL gates + tune) deterministically,
then approximates the variable `place_level`/branch draws with a loop over the
expected number of placed levels.  The exact count can be derived by running
the vendor C once at seed=0 under an RNG counter.

### Priority 3 — `u_init` attr variation loop + `ini_inv` blessorcurse — **~17–23 draws**

The 6-attribute `rn2(20)` loop (`u_init.c:888`) is always exactly 6 draws and
is trivially portable.  The `ini_inv` blessorcurse draws are already partially
handled (the BLINDFOLD `rn2(5)` is ported); the remaining weapon/armor/potion
`blessorcurse` calls add ~11 more draws for the 6 fixed Rogue items.

**Files:** `vendor/nle/src/u_init.c:888-894`, `vendor/nle/src/mkobj.c:804-1004`
**Action:** Extend `create_character` in `character.py` to call `rn2` for each
`ini_inv` item's bless/curse path, and add the 6-draw attr variation loop.

---

## 8. Recommended fix order for `player_x` parity

1. **Port `init_attr(75)` draws** (~35 draws) — largest single block, pure
   arithmetic loop, no dungeon state dependencies.
2. **Port `init_dungeons` fixed draws** (18 guaranteed draws: depths ×4,
   RNDLEVEL gates ×8, Fort Ludios gate ×1, tune ×5) — no branching required,
   all values can be computed from the compiled dungeon data.
3. **Port attr variation loop** (6 draws) — trivial.
4. **Port `ini_inv` per-item blessorcurse draws** (~11 draws) — requires
   knowing exact item list but no dungeon data.
5. **Port variable `place_level` / branch draws** — last because they require
   tracking which special levels survived chance gates.

Steps 1–4 alone close ~65–80% of the draw gap and should bring `player_x` from
the current ±25-tile error to within the range of `place_level` variance.

---

*Audit performed against `vendor/nle/src/{role.c, dungeon.c, artifact.c,
u_init.c, attrib.c, mkobj.c, exper.c}` at commit HEAD.  No code changes were
made.*
