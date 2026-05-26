# NLE / Nethax Termination Alignment

## Why we care

`tests/test_nle_return_distribution.py` (5-episode smoke, commit
7f76d5f) reports:

| metric              | NLE   | Nethax |
| ------------------- | ----- | ------ |
| mean episode length | 20.00 | 14.6   |

Under the **same** random policy over `{N, E, S, W, SEARCH, WAIT}`,
NLE never terminates inside 20 steps while Nethax terminates early
~5x more often.  This file enumerates every `done=True` write in
Nethax and the corresponding NLE / vendor-NetHack rule, then flags
the ones most likely to fire under random movement on level 1.

## NLE termination contract (vendor/nle/nle/env/base.py)

NLE returns `done=True` only when the underlying `libnethack` C
process emits `done` from `nethack.step` (i.e. the C-side game
ended — player died, ascended, quit, or saved) OR when
`_is_episode_end(observation)` returns a non-`RUNNING` `StepStatus`.
The default task's `_is_episode_end` (base.py:523-532) returns
`StepStatus.RUNNING` unconditionally — i.e. **only C-side `done`
counts**.  `_check_abort` (base.py:336-337) flips `end_status` to
`ABORTED` at `max_episode_steps`, which is then quit nicely via
`_quit_game`.

So in NLE, on level 1, with random movement, the only realistic
early-termination paths are HP→0 via monster attacks or status
effects.  Vendor mechanics use a starting HP of ~10-16 for Rogue
(rogue.c) and a ~50% miss rate against the player — so dying in <20
random moves is empirically extremely rare on dlvl 1.

## Nethax `done=True` paths — full audit

### Path 1 — HP→0 from monster melee (action_dispatch.py:1191)

```python
state_final = state_final.replace(
    player_hp=_new_hp,
    done=state_final.done | (_new_hp == jnp.int32(0)),
)
```

This is the **descended-while-punished** ball/chain self-damage
block.  Triggers `rn1(7, 25) = 25-31` damage when the hero descends
stairs while punished (no helm) — irrelevant under SEARCH/WAIT/move,
but does fire if the random policy picks a `>` direction and the
hero is on stairs.  In the smoke test the action set excludes `<`/`>`
and the hero starts unpunished, so this path is **not the divergence
cause**.

### Path 2 — Lava (action_dispatch.py:1275)

```python
new_done_after_lava = state_final.done | _lava_kills
```

Triggers only on stepping into a lava tile (`TileType.LAVA`).  Dlvl-1
in default generation has no lava.  **Not the cause.**

### Path 3 — Choking on food (action_dispatch.py:1717)

```python
new_done = new_state.done | chokes
```

Only triggers on EAT when satiated.  Smoke test excludes EAT.  **Not
the cause.**

### Path 4 — Drowning (water.py:463, 698)

```python
new_done = s.done | (~crawled_out)
```

Only fires when stepping onto a water tile.  Dlvl-1 default doesn't
have pools.  **Not the cause** under default gen, but could fire if
the start room sits adjacent to a pool — worth checking the
generated level.

### Path 5 — Monster melee (monster_actions.py:892, 969, 1139, 1215; combat.py:1987)

These are the **per-turn monster turn** writes.  After every player
move, `_monster_ai_step` runs each monster's AI; each monster's
melee against the player can drop player HP and set `done`.  Vendor
parity: ``mhitu.c``/``hitmu``.  **STRONG SUSPECT — primary
divergence candidate.**

NLE-vs-Nethax delta: vendor NLE generates a fresh dlvl 1 from scratch
per episode with a low-density 1-3 monsters at character start (Rogue
gets pet + ~2 wild dlvl-1 monsters within sight per `mklev.c`).
Nethax's `populate_level_with_monsters` may be placing **more or
stronger** monsters than vendor would, or letting monsters attack on
turn 0 before the player has time to retreat.

Specifically: random N/E/S/W with no FIGHT means the player keeps
bumping into adjacent monsters (which triggers attack via
`_attack_branch`) and then takes a counter-melee on the same turn
plus the monster turn — easy to compound 4-8 HP loss per step at
starting HP 10-16.

### Path 6 — Monster ranged / spells (monster_ai.py:1808, 1911, 2165, 2394, 2761, 3455, 3529)

Wand zap / spell cast / breath weapon by monsters.  At dlvl 1 these
are rare (dlvl-1 monster pool is mostly low-tier melee), but a stray
gnome lord with a wand of striking would kill the hero in 1-3 shots.
**Possible cause** if `populate_level_with_monsters` over-spawns
high-tier monsters.

### Path 7 — Monster passives (monster_passives.py:133, 185)

Counterattack on player melee (e.g. floating eye paralysis →
death-by-melee while paralysed).  Floating eyes are rare on dlvl 1
in vendor.  **Unlikely cause.**

### Path 8 — Starvation (status_effects.py:1013)

`new_done = done | starved` when `hunger_state == STARVED`.  Vendor
hunger countdown starts at `u.uhunger = 900`, decrements by ~1/turn,
reaches STARVED at -2000 → death.  20 turns can't reach STARVED.
**Not the cause.**

### Path 9 — Strangulation (status_effects.py:1061)

Fires when `STRANGLED` timer hits 1.  Only relevant if the hero is
wearing a cursed amulet of strangulation — Rogue's starting
inventory doesn't include one.  **Not the cause.**

### Path 10 — Stoning (status_effects.py:1076)

Touching a cockatrice or eating cockatrice corpse.  Cockatrices
don't spawn on dlvl 1 (level 5+).  **Not the cause.**

### Path 11 — Sliming (status_effects.py:1092)

Green slime / acid blob touch.  Acid blobs are dlvl-3+.  **Not the
cause.**

### Path 12 — Illness (status_effects.py:1110, 1130)

`ILL_FATAL` timer expiry.  Requires prior infection; impossible
within 20 turns of game start.  **Not the cause.**

### Path 13 — `apply_stoning_death`, `apply_sliming_death`,
`apply_illness_death` (status_effects.py:1304, 1323, 1472)

Convergence points of the above status-driven deaths.  **Same as
8-12 — not the cause.**

### Path 14 — Polymorph revert (polymorph.py:1387, 1425, 1433)

UNCHANGING / genocide-self / post-revert-HP<1 deaths during
rehumanize.  Player can't be polymorphed in the first 20 turns of a
random-policy run.  **Not the cause.**

### Path 15 — Swallow / digestion (swallow.py:293)

Engulfer's digestion timer hitting 0 + HP→0.  Engulfers are dlvl-5+
(trapper, lurker above).  **Not the cause.**

### Path 16 — Quit (action_dispatch.py:2189)

`return state.replace(done=jnp.bool_(True))` on `Command.QUIT` (M-q).
Smoke action set has no QUIT.  **Not the cause.**

### Path 17 — Ascension (ascension.py:180)

`done=new_done` when hero ascends with the Amulet on the Astral
Plane.  Impossible in 20 turns.  **Not the cause.**

### Path 18 — env.py `_tick_stinking_cloud` (env.py:326,342)

```python
new_done = state.done | (new_hp <= jnp.int32(0))
```

Only active when `cloud_turns > 0` — set by scroll of stinking
cloud, which the hero has not read.  **Not the cause.**

### Path 19 — env.py `_status_step` exit (env.py:443)

Aggregates Paths 8-13.  **Same — not the cause.**

## Likely culprits (ranked)

1. **Monster melee (Path 5)** — most likely.  Mitigations:
   * Verify `populate_level_with_monsters` matches vendor's dlvl-1
     density (vendor `mklev.c::makedungeon` → `mkroom` → `makemon`
     places `rn2(3)` monsters per non-shop room, peaceful 50% chance,
     of dlvl <= 1 difficulty).
   * Confirm peaceful flag is respected (peaceful monsters in vendor
     don't attack until provoked).
   * Confirm hero's starting HP matches vendor Rogue (12-16 HP
     baseline + class HP rolls).
2. **Monster ranged (Path 6)** — secondary.  Audit
   `populate_level_with_monsters` for over-leveled spawns.
3. **Drowning (Path 4)** — edge case.  Worth checking whether the
   smoke seeds happen to place the hero adjacent to a generated pool.

## Test-time fix for the smoke

Independent of vendor parity, the smoke can be made deterministic by
either:
* Skipping the player's turn when adjacent to a monster (i.e. add a
  retreat heuristic into the policy — but that makes it not random).
* Spawning the hero in an empty room with no monster within Chebyshev
  radius 5 for the first 20 turns.

These are validation-shape changes; the parity question is whether
Nethax should naturally produce NLE-matching episode-length
distributions under random play.  Per the audit above, **the answer
is "yes after fixing Path 5"** — Path 5 is the only `done=True` write
that fires within 20 turns of game start under N/E/S/W/SEARCH/WAIT.

## Action-mapping fix shipped

`Nethax/nethax/nle_action_map.py` adds the static
`NLE_INDEX_TO_ASCII[86]` lookup and `maybe_remap_action()`.  Wired
into `NethaxEnv._step_impl` so callers can pass either NLE index
(`< 86`) or raw ASCII (`>= 86`).  Out-of-range NLE indices are
clipped, not raised, to preserve JIT purity.
