# Wave 5 — Endgame

The endgame ascension condition + the 5 Astral planes.  All in
`Nethax/nethax/dungeon/endgame.py` and
`Nethax/nethax/subsystems/ascension.py`.

## The five Astral planes

| Plane | depth | Theme | Citation |
|---|---|---|---|
| Earth  | 1 | Caverns in solid rock | `dat/earth.lua` |
| Air    | 2 | Almost-no-floor; flight needed | `dat/air.lua` |
| Fire   | 3 | Lava lake + floor islands | `dat/fire.lua` |
| Water  | 4 | Pool everywhere + floor bubbles | `dat/water.lua` |
| Astral | 5 | Open field + 3 altars | `dat/astral.lua` |

`generate_endgame_level(rng, depth)` dispatches by depth.

Internal Endgame ascent uses up-stair link between L_n and L_{n+1};
vendor uses portals (`des.levregion ... type="portal"`) but for
navigability we wire stairs.  This is captured in
`branches.py::init_branch_graph` lines 547-553.

## Ascension condition

`Nethax/nethax/subsystems/ascension.py::check_ascension`:

All three must hold (boolean AND):

1. `on_astral_plane(state)` — `current_branch == ENDGAME` and `current_level == 5`.
2. `on_matching_altar(state)` — player at one of the 3 canonical altar
   coordinates, AND `player_align` matches the altar's alignment.
3. `player_holds_amulet(state)` — `Amulet of Yendor` in inventory with
   quantity ≥ 1.

Vendor source: `vendor/nethack/src/pray.c::offer_real_amulet`,
`vendor/nethack/src/end.c::done_ascend`.

### Astral altar coords (vendor `dat/astral.lua` lines 41-43)

- Lawful  : `(9, 7)`
- Neutral : `(5, 37)`
- Chaotic : `(9, 67)`

These are baked into `endgame.py::ASTRAL_ALTAR_*` and consulted by
`ascension._altar_alignment_at`.

### Wave 5 simplification

Vendor additionally requires the player to type `#offer` while
standing on the altar.  We collapse this to: "standing on the altar
while holding the Amulet" auto-triggers ascension via
`maybe_ascend(state)` called from `env.step` at end of pipeline.

The full offer-handler will be wired through `dispatch_action.OFFER`
in Wave 6.

## Ascend

`ascend(state)`:

```python
new_done    = jnp.bool_(True)
new_scoring = record_achievement(state.scoring, Achievement.ASCENDED)
new_scoring = add_score(new_scoring, 50000)
return state.replace(done=new_done, scoring=new_scoring)
```

`maybe_ascend` is the JIT-safe entry: `lax.cond` on
`check_ascension(state)`.

## Scoring (Wave 5 basic)

Flat 50000-point ascension bonus.  Vendor topten formula
(`vendor/nethack/src/end.c::topten`) considers:

- `u.urealtime` (real elapsed time)
- `u.uhpmax`
- `u.umoves`
- hard-fought multiplier (per-level alive-at-end status)
- conduct-completion bonuses

All deferred to Wave 6.

## Post-ascension idempotency

`env.step` wraps the entire pipeline in:

```python
already_done = state.done
new_state = lax.cond(already_done, lambda _: state, _do_step, operand=None)
```

So after ascension sets `state.done=True`, all subsequent `env.step`
calls return the state unchanged (no monster ticks, no status ticks,
no further reward).

## Test coverage

`tests/test_endgame.py` — 19 tests across the 5 planes + ascension:

1. Earth plane terrain non-empty
2. Earth plane has caverns
3. Air plane is mostly void (passable count < 50%)
4. Fire plane has lava
5. Fire plane has floor islands
6. Water plane has pool / bubbles
7. Astral plane has exactly 3 altars
8. Astral altar coords match canonical
9. Astral altar alignments distinct
10. `generate_endgame_level(rng, depth)` dispatches by depth
11. Endgame factories produce correct shapes
12. `player_holds_amulet` returns True when Amulet present
13. `player_holds_amulet` returns False otherwise
14. `on_astral_plane` checks branch+level
15. `on_matching_altar` matches lawful at (9,7)
16. `check_ascension` requires all three conditions
17. `ascend` sets done + achievement + score
18. `maybe_ascend` is no-op when condition not met
19. Ascension from `env.step` end-to-end (this is the integration test
    `test_endgame_ascension_full_flow` in `test_wave5_integration.py`).

All 19 pass.

## Wave 6 follow-ups

- Full `topten` scoring formula.
- `#offer` action handler.
- Per-conduct bonus (FOODLESS, ATHEIST, PACIFIST add to score).
- "How did this character die" report from `done()` machinery.
- Tombstone display from `vendor/nethack/src/end.c::tombstone`.
