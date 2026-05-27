# NLE DISP Stream Audit — CORE vs DISP RNG Split

**Date:** 2026-05-27  
**Scope:** Audit only — no code changes.  
**Goal:** Determine whether Nethax's current single-stream (`vendor_rng` = CORE only) model
causes byte-parity failures, and what it would take to fix it.

---

## 1. Stream Architecture in NLE

`vendor/nle/src/rnd.c:20–25` defines two independent ISAAC64 streams:

```c
enum { CORE = 0, DISP = 1 };

static struct rnglist_t rnglist[] = {
    { rn2,                FALSE, { 0 } },   /* CORE */
    { rn2_on_display_rng, FALSE, { 0 } },   /* DISP */
};
```

`rn2()` drains `rnglist[CORE]`; `rn2_on_display_rng()` drains `rnglist[DISP]`
(`rnd.c:62` and `rnd.c:69–73`).  Both streams are seeded independently via
`nle_set_seed` (`nle.c:527–536`) using separate `core` and `disp` integers passed
from Python.

`hacklib.c:854–855` stores both seeds:
```c
unsigned long nle_seeds[] = {0L, 0L};  /* [CORE], [DISP] */
```

**Current Nethax state:** `Nethax/nethax/state.py:342` holds a single
`vendor_rng: Isaac64State` — the CORE stream only.  There is no `disp_rng` field.
`env.py:140` seeds `vendor_rng` from the JAX PRNGKey but never seeds a DISP stream.

---

## 2. DISP Stream Call-Site Table

All sites in the vendor tree that call `rn2_on_display_rng`, classified by
frequency and trigger.

| # | File:Line | Context | Frequency | Trigger | CORE Pollution if Mis-routed |
|---|-----------|---------|-----------|---------|------------------------------|
| 1 | `display.c:465` | `map_object_or_mimic`: mimic glyph via `what_mon(…, rn2_on_display_rng)` | **Every render frame** that has a mimicking monster | Passive — called whenever the map tile is repainted | 1 draw/visible mimic/frame |
| 2 | `display.c:486` | `map_object_or_mimic`: pet glyph via `pet_to_glyph(…, rn2_on_display_rng)` | **Every render frame** with a visible tame monster | Passive — paint pass | 1 draw/visible pet/frame |
| 3 | `display.c:490,492,496,498` | Detected / normal monster glyph (`detected_mon_to_glyph`, `mon_to_glyph`) | **Every render frame** per visible monster (4 branches) | Passive — paint pass | Up to 4 draws/monster/frame |
| 4 | `display.c:522` | `display_warning`: Hallucination warning-level glyph | Every render frame, only under Hallucination | Passive — paint pass | 1 draw/warning monster/frame |
| 5 | `display.c:525` | `display_warning`: `mon_to_glyph` for WARN_OF_MON | Every render frame with matching warned monster | Passive | 1 draw/frame |
| 6 | `display.c:1841` | `swallow_to_glyph`: engulf display glyph | Every frame while swallowed | Passive — paint pass | 1 draw/frame |
| 7 | `detect.c:1629` | `detect_trap`: compare stored glyph against `trap_to_glyph` to decide if screen needs clearing | **Once per search/detect-traps action** that reveals a trap | Game event (search command, scroll/spell of detect traps) | 1 draw/trap-find |
| 8 | `detect.c:1867` | `run_vision`: `mon_to_glyph(u.ustuck, …)` in `sense_monsters_with_detection` | Once per detect-monsters cast while swallowed | Game event (detect-monsters spell/scroll) | 1 draw/cast |
| 9 | `detect.c:1876` | `trap_to_glyph` in `sense_monsters_with_detection` | Once per detect-monsters cast | Game event | 1 draw/seen-trap |
| 10 | `pager.c:407` | `do_look`: check if displayed glyph matches `mon_to_glyph(u.ustuck)` | **Every `;` look action**, or mouseover | Game event (look command) | 1 draw/look |
| 11 | `pager.c:461` | `do_look`: `what_trap(…, rn2_on_display_rng)` for trap tile description | Every look at a trap tile | Game event | 1 draw/look |
| 12 | `pager.c:1576` | `doidtrap` (auto-trap identification when stepping on): `what_trap` | Once per trap-step that shows name | Game event (movement) | 1 draw/trap-step |
| 13 | `hack.c:1504` | `run_corridor`: display trap name when stopping in front of trap during run | Each time autorun stops at a trap | Game event (movement) | 1 draw/run-stop |
| 14 | `hack.c:2762` | `domove` lateral scan for traps: display trap name | Each move scan near seen trap | Game event (movement) | 1 draw/adjacent-trap |
| 15 | `muse.c:1996` | `distfrommon`: reveal aggravating monster glyph via `show_glyph` | Once when aggravate monster triggers | Game event (aggravate property) | 1 draw/aggravate |
| 16 | `uhitm.c:2093` | `start_engulf`: player-monster engulf animation glyph | Once per engulf event | Game event (combat) | 1 draw/engulf |
| 17 | `dothrow.c:1082,1162,1389` | Projectile/tether glyph animation via `obj_to_glyph` | Once per throw/kick that triggers animation | Game event (throw/kick) | 1 draw/projectile |
| 18 | `mthrowu.c:551` | Monster throw animation glyph | Once per monster throw | Game event | 1 draw/mthrow |
| 19 | `trap.c:1834` | Trap-triggered throw animation glyph | Once per trap fire | Game event | 1 draw/trap-fire |
| 20 | `zap.c:3254,3256` | Zap beam animation glyph | Once per zap | Game event | 1 draw/zap |
| 21 | `invent.c:2720,2841` | Inventory menu glyph for each item | Every inventory open (`,` command) | Game event (inventory) | N draws/inventory-open |
| 22 | `pickup.c:921,946` | Pickup menu glyph | Every pickup with multiple items | Game event (pickup) | N draws/pickup |
| 23 | `do_name.c:1268` | Hallucination: `wipeout_text` scuffs artifact name during engraving | Rare (hallucinated engraving of artifact) | Game event | 1+ draws |
| 24 | `do_name.c:1568,1580` | Hallucination: pick random rank/fake name for "what's this object?" | Rare (hallucinated naming prompt) | Game event | 2 draws |
| 25 | `do_name.c:2029,2054,2111` | `bogusmon`/`rndmonnam`: fake monster/colour names under Hallucination | Every hallucinated monster display | Game event + passive render | N draws |
| 26 | `pray.c:2174,2189` | `halu_gname`: hallucinated god name in messages | Once per prayer while hallucinated | Game event (prayer) | 1–2 draws |
| 27 | `role.c:796` | `randrole(for_display=TRUE)`: pick display role for hallucinated title | Rare — hallucination-triggered name display | Passive (message) | 1 draw |
| 28 | `do.c:1458` | `goto_level`: `reseed_random(rn2_on_display_rng)` on level transition | Once per staircase/level-change | Game event | Full DISP re-init |
| 29 | `mklev.c:997,1035` | `mklev()`: `reseed_random(rn2_on_display_rng)` at start and end of level gen | Once per new level generated | Game event (level gen) | Full DISP re-init (×2) |
| 30 | `options.c:709` | `init_random(rn2_on_display_rng)` during options init | Once at game start | Init | Full DISP init |
| 31 | `winrl.cc:458` | `update_inventory_method`: `obj_to_glyph(otmp, rn2_on_display_rng)` for each inventory item | Every inventory-changed callback | Passive (RL obs build) | N draws/inv-update |

---

## 3. Per-Obs-Channel Impact

NLE's RL observation channels and which DISP draws occur during their construction:

### `glyphs` (map glyph array, `int16_t[(COLNO-1)*ROWNO]`)

Built by `winrl.cc:store_glyph` ← called from `print_glyph` (line 933).  The
`glyph` argument is computed **upstream** in `display.c` before reaching `winrl.cc`.
For every visible monster tile that passes through `map_object_or_mimic` or
`display_warning`, 1–4 DISP draws fire before the glyph lands in `glyphs_`.
Under Hallucination this includes `rndmonnam` loops (site 25).

**DISP draws per step:** ~1 per visible monster + 1 per visible mimic + 1 under
swallow + occasional Hallucination chains. Typical value: **0–10 DISP draws/step**
(can spike to 20+ in a crowded hall under Hallucination).

### `inv_glyphs` (inventory glyph array, `int16_t[MAX_INVENTORY]`)

Populated by `update_inventory_method` (`winrl.cc:445–461`), which calls
`obj_to_glyph(otmp, rn2_on_display_rng)` for every inventory item (site 31).
**Fires every time the inventory changes** (pickup, drop, use, combine).

**DISP draws per inventory-change event:** 1 per item in inventory (typically
4–20 draws; up to ~52 if slots are full).

### `chars` / `colors` / `specials` (tty character layers)

Filled by `store_mapped_glyph` ← `print_glyph`. These receive the *already-computed*
ch/color/special from `mapglyph()`, which does **not** call `rn2_on_display_rng`
itself. The DISP draws happen in the glyph-selection layer above.  So chars/colors
carry the *rendered side-effects* of DISP draws but do not add new ones.

### `screen_descriptions` (text descriptions per tile)

`store_screen_description` → `auto_describe` → `do_look` path.  `do_look` at
`pager.c:407` calls `mon_to_glyph(u.ustuck, rn2_on_display_rng)` (site 10) and
at `pager.c:461` calls `what_trap(…, rn2_on_display_rng)` (site 11). These fire
**once per tile** when `screen_descriptions` is enabled and the description is
refreshed. Given NLE refreshes the whole viewport each step: potentially
`(COLNO-1) * ROWNO` calls on tile types that have monster/trap glyphs, but in
practice only visible monster/trap tiles fire — roughly same order as `glyphs`.

### `message` / `tty_chars` (message log)

Hallucination-path text generation: `rndmonnam` (sites 24–25), `halu_gname`
(sites 26), `bogusmon` (sites 23). These are message-layer DISP draws, not map
draws.  Fire **0 times per step** in normal play; **1–5 per step** under
Hallucination.

---

## 4. CORE Pollution Analysis

### Current state: **one `vendor_rng` slot, seeded as CORE; DISP stream absent**

NLE's C code draws from `rnglist[DISP]` for all sites above. Nethax has no DISP
stream. Therefore:

- Any Nethax code path that re-implements a DISP-consuming function (glyph
  rendering, inventory menu building, look descriptions) and accidentally routes
  it through CORE will insert phantom draws into the CORE sequence.
- Conversely, if Nethax *ignores* DISP draws entirely (stubs them as
  `rn2_on_display_rng → 0` or equivalent), CORE is clean but DISP-dependent
  glyph values diverge from NLE on the same seed, causing `glyphs` and
  `inv_glyphs` obs channels to disagree byte-for-byte.

### Estimated CORE pollution per validator step (worst case)

| Scenario | Extra CORE draws |
|----------|-----------------|
| Empty room, no monsters, no inventory change | 0 |
| 5 visible monsters, no Hallucination | ~5–8 |
| Inventory open with 20 items | ~20 |
| Hallucination + 10 monsters + message | ~25–35 |
| Full crowded hall + inventory + screen_descriptions | ~50–80 |

In a standard walk-around step (move into corridor, no inventory change, a few
monsters), roughly **5–15 phantom CORE draws** would be injected if DISP calls
are accidentally routed through CORE.  Over a 1000-step episode this is
**5 000–15 000** extra CORE draws, fully desynchronising every subsequent CORE
roll (combat outcomes, monster AI, trap triggers, item generation).

---

## 5. Top 3 Sites Where DISP Matters Most for Byte Parity

### Site 1 — `display.c:486–498` (monster glyph selection, every render)

`mon_to_glyph`, `pet_to_glyph`, `detected_mon_to_glyph` all call
`rn2_on_display_rng` **once per visible monster per frame**.  This is the
highest-frequency DISP consumer.  In a typical dungeon with 5–15 monsters on
screen, 5–15 DISP draws fire every single step.  The returned glyph feeds
directly into the `glyphs` obs array — if DISP is not tracked, `glyphs` diverges
immediately, breaking any byte-parity validator that checks `glyphs`.

### Site 2 — `winrl.cc:458` / `invent.c:2720,2841` (inventory glyph, every inv change)

`update_inventory_method` calls `obj_to_glyph(otmp, rn2_on_display_rng)` per
item.  With a full inventory (52 slots) this is **52 DISP draws** in one callback.
`inv_glyphs` is the channel most directly populated by these draws.  Every pickup,
drop, or use event triggers this — in a combat-heavy episode that may mean
30–60 inv-change events ≈ **1 560–3 120 DISP draws per episode** flowing through
`inv_glyphs` alone.

### Site 3 — `mklev.c:997,1035` / `do.c:1458` (DISP reseed at level gen / level transition)

`reseed_random(rn2_on_display_rng)` is called twice inside `mklev()` and once
during `goto_level`.  This completely re-initialises the DISP ISAAC64 stream from
the current timestamp/state.  If Nethax does not model DISP reseeding, the DISP
stream position after level-gen is undefined relative to NLE — **all post-level-gen
glyph draws diverge** regardless of whether DISP was tracked up to that point.
This is the single most impactful structural gap: even a correctly seeded DISP
stream will drift on every `mklev` call unless `reseed_random(DISP)` is replicated.

---

## 6. Recommendation

### Add a second `Isaac64State` (DISP stream) to the state pytree: **YES**

Rationale:

1. `glyphs` and `inv_glyphs` are load-bearing RL observation channels.  Both are
   populated using DISP-stream draws on **every step** (glyph selection) or on
   every inventory change.  Byte-parity on these channels requires tracking the
   DISP state.

2. The DISP and CORE streams are **independent ISAAC64 instances** with separate
   seeds (`nle_seeds[0]` vs `nle_seeds[1]`, `hacklib.c:855`).  There is no way
   to derive DISP outputs from CORE state.  Stubs (constant 0 or hash tricks)
   will produce wrong glyphs.

3. The `reseed_random(rn2_on_display_rng)` calls in `mklev` mean the DISP stream
   is periodically re-initialised.  Modeling this correctly requires holding the
   DISP `Isaac64State` in the pytree so re-seeding can be applied at the same
   point as NLE.

**Empty-stub alternative (DISP → always 0):**  Acceptable only if the team
agrees to sacrifice byte-parity on `glyphs` and `inv_glyphs` and only needs
CORE parity (combat/AI outcomes).  This eliminates the pytree size penalty
but the obs channels will permanently disagree with NLE.

---

## 7. Implementation Sketch

### `Nethax/nethax/vendor_rng.py`

No structural changes to `Isaac64State` needed — it already represents one full
ISAAC64 context.  Two instances will be held separately in `EnvState`.

### `Nethax/nethax/state.py`

Add a second field alongside `vendor_rng`:

```python
vendor_rng: Isaac64State       # CORE stream — rnglist[CORE]
vendor_rng_disp: Isaac64State  # DISP stream — rnglist[DISP]
```

Both should default to `Isaac64State.empty()`.

### `Nethax/nethax/env.py` (`reset`)

Seed both streams, mirroring `nle.c:531–532`:

```python
v_state_core = _vendor_rng.init(int(seed_core))
v_state_disp  = _vendor_rng.init(int(seed_disp))
state = state.replace(vendor_rng=v_state_core, vendor_rng_disp=v_state_disp)
```

NLE uses the same integer for both streams by default (`env.seed(seeds=(s, s))`),
so `seed_disp = seed_core` is the correct initial value unless the caller passes
separate seeds.

### Call sites that should use DISP

Any Nethax helper that replicates a NLE function which internally calls
`rn2_on_display_rng` must be passed `state.vendor_rng_disp` instead of
`state.vendor_rng`.  Primary sites:

- `mon_to_glyph` / `pet_to_glyph` / `detected_mon_to_glyph` equivalents
  (called from obs builder that constructs `glyphs`)
- `obj_to_glyph` equivalent (called from inventory obs builder, `inv_glyphs`)
- `what_trap(…, rn2_on_display_rng)` equivalent (called from look/trap-step paths)
- `swallow_to_glyph` equivalent (called when player is swallowed)

Any game-logic function that uses `rn2` (not `rn2_on_display_rng`) — combat,
monster AI, item generation, movement — continues using `state.vendor_rng` (CORE).

### `mklev` / level-transition reseed

When Nethax's dungeon-gen replicates `mklev()`, it must call
`reseed_random(rn2_on_display_rng)` semantics on `vendor_rng_disp` at the same
two points (`mklev.c:997` and `mklev.c:1035`).  If `has_strong_rngseed` is true
(which it is for NLE seeded games), `reseed_random` calls `init_random` which
re-seeds from a platform entropy source — **Nethax must replicate this** or
accept that post-level-gen DISP diverges.  The safest approach: re-seed
`vendor_rng_disp` with the same entropy used by NLE at the equivalent point.

---

## 8. Summary

| Question | Answer |
|----------|--------|
| Does DISP affect byte-parity obs channels? | **Yes** — `glyphs`, `inv_glyphs`, `screen_descriptions` all depend on DISP draws |
| Highest-frequency DISP site | `display.c:486–498` — 1 DISP draw per visible monster per step |
| Most impactful structural gap | `mklev.c:997,1035` — full DISP re-seed destroys accumulated DISP state |
| Recommended fix | Add `vendor_rng_disp: Isaac64State` to `EnvState`; seed from same integer as CORE |
| Empty-stub acceptable? | Only if `glyphs`/`inv_glyphs` byte-parity is not required |
| Estimated Δ divergences if DISP added | ~1 divergence per step eliminated from `glyphs`; ~1 per inv-change from `inv_glyphs`; total ~**10–30 fewer divergences per episode** in typical play |
