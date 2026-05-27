# Special Features & Dungeon-Gen RNG Audit

**Scope:** seed=0, Main Dlvl 1.  Read-only audit of vendor C source against
the JAX pipeline in `Nethax/nethax/dungeon/`.

---

## Per-System Draw Count (seed=0 Dlvl 1)

### 1. Vault placement (`do_vault` / `create_vault`)

`do_vault()` is the macro `(vault_x != -1)` — **0 RNG draws** to evaluate.

If vault is attempted (`vault_x` was set by `makerooms` `rn2(2)` gate, already
tracked), the fill path draws:
- `rnd_rect()` retry — 1 draw (inside `create_vault` → `rnd_rect`)
- `check_room` — 0 draws (pure geometry)
- `makevtele()` — gated on `!rn2(3)`: **1 draw**, then `makeniche(TELEP_TRAP)`
  → `makeniche` inner loop: up to 8×`rn2(nroom)` + `rn2(5)` + door draws

**Vault path draws not yet threaded through `vendor_rng`** in
`generate_main_branch_l1_with_features`: `maybe_create_vault` uses Threefry,
not the ISAAC64 stream.  Estimated deficit: **5–15 draws** when vault present.

Vendor cite: `mklev.c:738-762`, `mklev.c:568-571`.

---

### 2. Shop placement (`mkroom(SHOPBASE)` → `mkshop` → `stock_room`)

Gated at `mklev.c:769-771`:

```c
else if (u_depth > 1 && u_depth < depth(&medusa_level)
         && nroom >= room_threshold && rn2(u_depth) < 3)
    mkroom(SHOPBASE);
```

On Dlvl 1, `u_depth == 1`, so `u_depth > 1` is **false** — the entire cascade
short-circuits at the very first `if`.  **0 RNG draws fire before the gate.**

The remaining 10 special-room `rn2` checks (`COURT`, `LEPREHALL`, `ZOO`, …)
are all `else if` branches that are **never reached** on Dlvl 1 because the
entire block short-circuits at the shop gate.

**Status: 0 deficit on Dlvl 1.**

However, `assign_special_room` (in `rooms.py`) is **never called** from
`generate_main_branch_l1_with_features`.  On Dlvl 1 all 11 draws are skipped
in vendor C too, but on Dlvl 2+ this function must be wired in.

Vendor cite: `mklev.c:767-796`.

---

### 3. Special room setup (`mkroomtype` / `fill_zoo` / `mktemple` etc.)

Not reached on Dlvl 1 (all depth gates fail — see §2).  **0 draws.**

Vendor cite: `mkroom.c:234-244`, `mkroom.c:266-437`, `mkroom.c:574-597`.

---

### 4. `add_door` / `dosdoor` / `dodoor`

`add_door` (`mklev.c:350-381`) — **0 RNG draws** (pure bookkeeping).

`dosdoor` (`mklev.c:383-448`) draws:
- `rn2(3)` — door type gate
- `rn2(5)` — D_ISOPEN check (if gate passed)
- `rn2(6)` — D_LOCKED check (else)
- `rn2(25)` — D_TRAPPED (depth >= 5 only, gated away on Dlvl 1)

**Per `dosdoor` call: 2–3 draws.**  Called from `makecorridors` → `join` →
`dodoor` (1 draw `rn2(8)` to choose DOOR vs SDOOR, then `dosdoor`).

These draws are **already consumed** by `corridors.py::makecorridors` /
`corridors.py::dodoor` which thread `vendor_rng`.

Vendor cite: `mklev.c:350-448`, `mklev.c:1249-1260`.

---

### 5. `mkbranch` / `place_branch`

`place_branch` (`mklev.c:1149-1207`) is called at `mklev.c:800`:

```c
place_branch(branchp, 0, 0);
```

On Dlvl 1, `Is_branchlev(&u.uz)` is false (the Mines branch attaches at Dlvl
2–4).  `branchp` is `NULL`.  The very first line of `place_branch`:

```c
if (!br || made_branch) return;
```

returns immediately.  **0 RNG draws.**

`find_branch_room` (inside `place_branch`) uses `rn2(nroom)` but is never
reached.

`mk_knox_portal` (`mklev.c:1864-1898`) draws `rn2(3)` but is depth-gated
(`u_depth > 10`) — **0 draws on Dlvl 1.**

**Status: 0 deficit on Dlvl 1.**

Vendor cite: `mklev.c:1149-1207`, `mklev.c:1864-1898`.

---

### 6. `level_init` / `clear_level_structures`

`clear_level_structures` (`mklev.c:578-647`) — memset/zero operations only.
**0 RNG draws.**

`topologize` (`mklev.c:1038-1101`) — called from `mklev()` after `makelevel()`.
Pure grid traversal, no RNG.  **0 draws.**

`set_wall_state` — post-process pass.  **0 draws.**

Vendor cite: `mklev.c:578-647`, `mklev.c:990-1036`.

---

### 7. `u_on_upstairs`

`dungeon.c:1260-1266` — position assignment only (`u_on_newpos` or
`u_on_sstairs`), no RNG.  **0 draws.**

---

### 8. `docrt`

`display.c:1379-1430` — display refresh (vision_recalc, see_monsters,
newsym calls).  No RNG on the main dungeon RNG stream.  **0 draws.**

---

### 9. `makedog` / `pet_type`

Called from `allmain.c:636` (after `mklev` and `check_special_room`).
`pet_type` (`dog.c:57-67`): draws `rn2(2)` when no role/preference override.

**1 draw, NOT consumed by the JAX pipeline.**

This draw fires post-`mklev()`, after `reseed_random` re-seeds at
`mklev.c:1034-1035`, so it is in the **post-mklev seed region** — outside
the `mklev()` RNG window.  Verify whether the validator checks blstats at a
point where this draw has already shifted the stream.

Vendor cite: `dog.c:57-67`, `allmain.c:615-636`.

---

### 10. `check_special_room` (post-mklev)

`hack.c:2446-2510` — on game start, `uentered` and `ushops_entered` are empty
(player hasn't moved yet).  Returns at line 2458.  **0 draws.**

---

## Summary Table

| System | Draws (Dlvl 1) | JAX status |
|---|---|---|
| `do_vault` eval | 0 | N/A (macro) |
| Vault fill path (`makevtele`, niches) | 5–15 (when vault present) | Threefry only — **deficit** |
| Shop gate `rn2(u_depth)` | **0** (depth gate fails) | N/A |
| All other special-room `rn2` | **0** (else-if, never reached) | `assign_special_room` unwired |
| `add_door` | 0 | N/A |
| `dosdoor` / `dodoor` | 2–3 per corridor door | Already in `corridors.py` |
| `place_branch` | **0** (null branch) | N/A |
| `mk_knox_portal` | **0** (depth gate) | N/A |
| `clear_level_structures` / `topologize` | 0 | N/A |
| `u_on_upstairs` | 0 | N/A |
| `docrt` | 0 | N/A |
| `makedog` / `pet_type` | **1** (post-mklev) | Not ported |
| `check_special_room` | 0 | N/A |

---

## Estimated Additional Deficit

- **Vault fill (when vault created):** ~5–15 draws.  Vault placement probability
  on Dlvl 1 is ~50% (`rn2(2)` in `makerooms`).  Expected deficit: ~5–8 draws.
- **`makedog`/`pet_type`:** 1 draw, but post-`reseed_random`, so it shifts the
  stream only if the validator probes after `pet_type` runs.
- **`assign_special_room` unwired:** 0 draws on Dlvl 1, but will be a source
  of deficit on all deeper levels.

**Total estimated additional deficit (Dlvl 1):** ~6–9 draws (vault-path
dependent).  The bulk of the ~1680 unaccounted draws are earlier in the
pipeline (per-room fill loops, corridor door chains, makemon sub-draws).

---

## Top 5 Missing Sources (ranked by magnitude)

1. **Vault fill path not on ISAAC64 stream** (`maybe_create_vault` uses
   Threefry; `makevtele`→`makeniche` inner draws not threaded).
   ~5–15 draws per vault; affects ~50% of Dlvl 1 seeds.

2. **Per-room fill loop sub-draws not fully counted** — `makemon`,
   `mkgold`, `mktrap`, `mksobj` each recurse into deeper RNG consumers
   (object BUC rolls, monster stat rolls).  `fill_ordinary_rooms` caps loops
   at 4/8 but vendor loops are unbounded; mismatch in tail iterations.

3. **`makedog`/`pet_type` — 1 post-mklev draw** not consumed by JAX
   (post-`reseed_random` region; relevant if validator probes at turn 0
   before any action).

4. **`assign_special_room` unwired** — 0 draws on Dlvl 1 but 1–11 draws
   on all deeper levels.  Will compound deficit as soon as depth > 1 testing
   begins.

5. **`dosdoor` D_TRAPPED path on depth >= 5** — 1 extra `rn2(25)` draw
   not yet accounted for in `corridors.py::dosdoor` at higher depths.

Vendor cites:
- `mklev.c:738-762` (vault path)
- `mklev.c:803-885` (per-room fill)
- `dog.c:57-67` (makedog)
- `mklev.c:767-796` (special room gate)
- `mklev.c:395-405` (dosdoor trapped draw)
