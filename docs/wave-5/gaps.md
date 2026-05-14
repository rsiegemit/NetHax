# Wave 5 → Wave 6 — Backlog

The remaining work after Wave 5.  Organised by area, with vendor citations and effort estimates.

## Conduct surface

| Item | Why deferred | Vendor source |
|---|---|---|
| **WISHLESS** conduct | Wish handler not built (needs `do_wish.c::makewish`) | `vendor/nethack/src/do_wish.c` |
| **ARTIWISHLESS** conduct | Gated on wish | `vendor/nethack/src/do_wish.c` |
| **Wish action handler** | Not in dispatch table | `vendor/nethack/src/cmd.c::dowish` |

Effort: ~200 LoC for wish handler + 20 LoC for 2 conduct violations.

## Save / load

Currently `Nethax/nethax/save_load.py` is a skeleton.

| Item | Status | Vendor source |
|---|---|---|
| `save_state(state, path)` | stub | `vendor/nethack/src/save.c::dosave0` |
| `load_state(path) -> EnvState` | stub | `vendor/nethack/src/restore.c::dorecover` |
| Versioned format | none | — |
| Roundtrip test | none | — |

Effort: ~400 LoC.  JAX pytree → numpy → pickle → roundtrip is the simplest path.

## Scoring + ascension polish

| Item | Vendor source |
|---|---|
| Full `topten` formula | `vendor/nethack/src/end.c::topten` |
| `u.urealtime` tracking | `vendor/nethack/src/timeout.c` |
| Per-conduct score bonuses | `vendor/nethack/src/insight.c::end_of_game` |
| `#offer` action handler | `vendor/nethack/src/pray.c::dosacrifice` |
| Death message generation | `vendor/nethack/src/end.c::done` |
| `killer_name`, `how_killed` plumbing | `vendor/nethack/src/end.c::done2` |
| Tombstone display (text) | `vendor/nethack/src/end.c::tombstone` |

Effort: ~300 LoC.

## Shop

| Item | Notes |
|---|---|
| Simplified buy/sell handler | Wave 4 + 5 left this stubbed |
| Angry shopkeeper mode | Triggered by stealing / unpaid leave |
| Shopkeeper as monster (peaceful, special HP) | Touches monster spawning |
| Item-price computation | `vendor/nethack/src/shk.c::contained_cost` |

Effort: ~250 LoC.

## `inv_strs` polish

`Nethax/nethax/obs/inv_strs.py` produces inventory string lines but
has known gaps:

| Item | Notes |
|---|---|
| Named items ("Excalibur" vs "long sword") | needs artifact name table |
| Vowel article ("an apple" vs "a banana") | `vendor/nethack/src/objnam.c::an` |
| Irregular plurals ("octopi", "mice") | `objnam.c::makeplural` |
| Curse-status visibility for un-id'd items | `objnam.c::doname` |
| Stack quantity ("3 daggers" vs "a dagger") | `objnam.c::doname` |

Effort: ~150 LoC + lookup tables.

## Monster + object table polish

| Item | Notes |
|---|---|
| Full `monstr[]` difficulty formula | `vendor/nethack/src/makemon.c::monstr_init` |
| Object table canonicalize | drop dual-naming (`SIGNATURE` vs `SIGNATURE_OF`); 503 → 453 entries |
| Monster table trim | 382 → 381 entries (drop our placeholder PM_NUM) |
| `monstr[]` test data | parity with `vendor/include/monst.h::MONS_*` |

Effort: ~100 LoC + 2 days table cleanup.

## Combat + monster polish

| Item | Notes |
|---|---|
| Property-based combat tests (Hypothesis) | invariant: HP never negative; damage = sum-of-dice |
| Role-specific bonuses (Monk martial arts, Samurai bushido) | `vendor/nethack/src/uhitm.c::martial` |
| Throughput benchmark (steps/sec) | not measured; want > 500 steps/sec CPU |
| Real `MS_SPELL` flag from monst.h | replace entry-index heuristic |
| Monster muse-slot random init | `vendor/nethack/src/muse.c::find_misc` |
| Pet item pickup | `vendor/nethack/src/dog.c::dog_invent` |

Effort: ~400 LoC + 2 days benchmarking.

## NLE compat shim full validation

`Nethax/nethax/compat/nle_shim.py` ships in Wave 5 with a 3-test smoke.
Full validation needs:

| Item | Notes |
|---|---|
| Run an NLE-trained model against `NLECompat` | uses RLlib or sample_factory script |
| `ttyrec` recorder shim | `vendor/nle/nle/dat/nethackrc` |
| `character` arg parsing (e.g. "wiz-hum-mal-cha") | `vendor/nle/nle/env/base.py::NLE.__init__` |
| `nle.nle.NLE` gym wrapper | a thin wrapper over `NLECompat` |

Effort: ~300 LoC + 1 day NLE-side integration testing.

## `scripts/legacy/play_nethax.py` interactive rewrite

There's an old pygame-based interactive driver in `scripts/legacy/`.
Wave 6 should rewrite it against:

- `NLECompat` for the env loop.
- `obs/pixel_obs.py::build_pixel_obs` for the render.
- Modern pygame event loop (Wave 4-era version uses tkinter).

Effort: ~200 LoC.

## Property-based testing

Hypothesis-driven invariants:

- **Combat invariants:** HP never negative, damage = sum-of-dice, AC monotone in armor count.
- **Conduct invariants:** any wired conduct, once set, stays set (no false-clears).
- **Cross-branch invariants:** any round-trip (Main → X → Main) preserves Main terrain bit-equal.
- **Endgame invariants:** post-`done` step is identity.

Effort: ~150 LoC + 1 day fuzzing.

---

## Wave 6 budget

Rough size of the backlog above:

| Area | LoC | Days |
|---|---|---|
| Conducts (WISHLESS, ARTIWISHLESS, wish handler) | 220 | 1 |
| Save/load | 400 | 2 |
| Scoring + death messages | 300 | 1.5 |
| Shop | 250 | 1.5 |
| `inv_strs` polish | 150 | 1 |
| Monster + object table polish | 100 | 2 |
| Combat / monster polish | 400 | 3 |
| NLE compat full | 300 | 2 |
| Interactive driver | 200 | 1 |
| Property-based tests | 150 | 1 |
| **Total** | **~2470** | **~16** |

≈ 8-10 parallel agents, 2 weeks.  Wave 6 is the polish wave; substantially smaller than Wave 5 because the cross-subsystem mechanics are now in place.
