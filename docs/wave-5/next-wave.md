# Wave 5 → Wave 6 — Scope Preview

Wave 6 is the **polish wave**.  Wave 5 delivered the cross-subsystem
mechanics end-to-end (monster AI, full special-level inventory, Quest
fidelity, containers, engrave, genocide, endgame ascension).  Wave 6's
job is to harden the surface, add the missing bookkeeping, and ship
the playable interactive driver.

After Wave 6, Nethax is a complete drop-in replacement for NLE.

## Wave 6 deliverables

### Save / load

- `Nethax/nethax/save_load.py::save_state(state, path)` and `load_state(path)`.
- Versioned format (header + pytree serialization).
- JAX pytree → numpy → pickle → roundtrip test.
- Citation: `vendor/nethack/src/save.c::dosave0`, `restore.c::dorecover`.

### Scoring + ascension polish

- Full `topten` formula from `vendor/nethack/src/end.c::topten`.
- `u.urealtime` tracking via `vendor/nethack/src/timeout.c`.
- Per-conduct score bonuses (FOODLESS, ATHEIST, PACIFIST contribute).
- `#offer` action handler (Wave 5 auto-offers on contact; Wave 6 needs
  the explicit action).

### Death message generation

- `done(state, KILLED_BY, killer_name)` plumbing.
- `state.scoring.killer_name` populated by combat.
- Tombstone text generation per `vendor/nethack/src/end.c::tombstone`.

### Shop simplified

- Buy/sell handler: pick-up an item in a shop, owe gold.
- Drop / pay handler: clear the debt or trigger angry mode.
- Angry shopkeeper: full-strength monster, refuses payment thereafter.
- Item-price computation per `vendor/nethack/src/shk.c::contained_cost`.

### `inv_strs` polish

- Named items: lookup table for artifact names (Excalibur, Mjollnir, …).
- Vowel article: "a apple" → "an apple" per `objnam.c::an`.
- Irregular plurals: "octopi", "mice" per `objnam.c::makeplural`.
- Curse-status visibility flag per `objnam.c::doname`.
- Stack quantity rendering: "3 daggers" / "an arrow".

### Conduct scoreboard

- End-of-game render of all 13 conducts: ✓ / ✗ per conduct.
- Total achievement count from `vendor/nethack/src/insight.c::end_of_game`.

### `scripts/legacy/play_nethax.py` rewrite

- Pygame-based interactive driver.
- Wraps `NLECompat` for the env loop.
- Renders `obs/pixel_obs.py::build_pixel_obs` to surface.
- Modern pygame 2.0 event handling.

### Full `monstr[]` difficulty formula

- Port `vendor/nethack/src/makemon.c::monstr_init`.
- Drives spawning weights and EXP gains.

### Object table canonicalize

- Drop dual-naming (`SIGNATURE` vs `SIGNATURE_OF`): 503 → 453 entries.
- Stable parity with `vendor/nethack/include/objects.h`.

### Monster table trim

- 382 → 381 (drop our placeholder).
- Parity with `vendor/nethack/include/monst.h`.

### Property-based combat tests

- Hypothesis-driven invariants: HP never negative, damage = sum-of-dice, AC monotone in armor count.
- Conduct invariants: any wired conduct, once set, stays set.

### Role-specific bonuses

- Monk martial arts: `vendor/nethack/src/uhitm.c::martial`.
- Samurai bushido: alignment-bonus on naginata strike.
- Valkyrie cold resistance: at level 1 (Wave 5 ships character creation; Wave 6 wires intrinsic).

### NLE compat shim full validation

- Run an RLlib NLE-trained agent against `NLECompat`.
- `ttyrec` recorder shim.
- `character` arg parsing (e.g. "wiz-hum-mal-cha").
- Optional: `nle.nle.NLE` gym wrapper thin layer.

### WISHLESS + ARTIWISHLESS conducts + wish handler

- `Nethax/nethax/subsystems/wish.py`: `makewish(state, rng)`.
- Wish parser: a text string → an Item or stat-boost.
- Conducts: WISHLESS broken on any wish; ARTIWISHLESS broken on artifact wish.
- Citation: `vendor/nethack/src/do_wish.c::makewish`.

## Wave 6 risks

1. **Save / load pytree stability** — JAX pytree treedefs are defined by
   field declaration order in `flax.struct.dataclass`.  Cross-version
   migration needs a careful versioning scheme.

2. **Death message UX** — vendor's `done()` is highly state-dependent;
   killer-name plumbing through every combat path is invasive.

3. **`topten` formula sensitivity** — small bugs produce wildly
   different scores.  Want a golden-set test from a known vendor save.

4. **Wish handler scope creep** — wish parser is non-trivial because
   it accepts free-form text.  Wave 6 should restrict wishes to a
   canonical list (vendor's `Hallu_resist` and friends) or token-based.

## Recommended Wave 6 launch shape

Sequential phases, parallelism within each:

### Phase 0 — save/load + scoring (2 agents, blocker)
- save_state / load_state.
- Full topten + death messages.

### Phase 1 — polish surface (parallel)
- inv_strs polish.
- Conduct scoreboard.
- Object table canonicalize.
- Monster table trim.

### Phase 2 — shop + wish (parallel)
- Shop buy/sell + angry mode.
- Wish handler + WISHLESS / ARTIWISHLESS.

### Phase 3 — combat / monster polish (parallel)
- Role-specific bonuses.
- monstr[] difficulty.
- Property-based tests.
- Real MS_SPELL flag.

### Phase 4 — driver + NLE compat (parallel)
- Interactive pygame driver.
- NLE compat full validation.
- Throughput benchmark.

### Phase 5 — integration tests (1 agent)
- Round-trip save / load with all subsystem state.
- Vendor parity sanity (NLE-side comparison run).

~10 agents total.

## Open questions for Wave 6

1. **Save format**: JAX pytree → pickle, or a hand-rolled binary format
   that mirrors vendor's `save.c`?  Pickle is fast and easy but tied
   to Python version; binary is portable but more code.
   **My recommendation: pickle for v1, binary as Wave 7 polish. ★**

2. **Wish handler**: free-form text wish parser, or canonical-token-only
   ("wish for blessed greased fixed +3 gray dragon scale mail")?
   Free-form needs a real lexer.
   **My recommendation: canonical-token only.  Vendor's free-form is
   itself a small DSL but adds 200 LoC of parsing. ★**

3. **`topten` formula**: keep flat 50000 bonus + sum of partial bonuses,
   or port the exact vendor formula?  The exact formula has many
   special cases for cause-of-death.
   **My recommendation: port exact formula.  RL training is sensitive
   to reward shape and we want vendor parity. ★**

4. **Pygame interactive driver**: pure-pygame, or pygame + textual
   overlay (curses-style)?  Textual is closer to vendor look.
   **My recommendation: pure-pygame.  Modern wargames eschew tty
   overlays. ★**

5. **NLE compat depth**: just the `.reset / .step / .actions` triad,
   or full `nle.nle.NLE` gym wrapper compatibility?  Gym wrapper adds
   a config schema (character / role / race selection).
   **My recommendation: full gym wrapper.  Lets us run vendor's NLE
   training scripts unchanged. ★**

Defaults all marked ★.  Reply with picks or "all defaults".
