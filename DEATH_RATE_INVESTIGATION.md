# Death-Rate Investigation — Nethax kills 3.5x faster than NLE

P0 RL-transfer issue: under random / WAIT policy, mean episode length is
**55 (Nethax) vs 191 (NLE)** turns.  This document records the code-path
analysis (smoke data not gathered — see "smoke status" below) and the top
three hypotheses, ranked.

## Smoke status

`tests/test_death_rate_audit.py` (committed in 3406e18) was launched with
`JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`
and killed after **>76 s of CPU time** with no output produced —
**JIT compile of `_step_impl` exceeds the 60 s budget on a cold cache.**
The script is correct; running it once the cache is warm should produce
per-step HP traces in milliseconds.  We proceed by code reading.

## Top hypotheses (ranked)

### H1 — Generated monsters are NOT spawned asleep (PRIMARY)

`Nethax/nethax/dungeon/spawning.py::populate_level_with_monsters`
(lines 803-957) writes 5 monsters into slots [0, 5) on level 1 and **never
sets `asleep = True`**.  The default in
`Nethax/nethax/subsystems/monster_ai.py::make_monster_ai_state` (line 748)
is `asleep = jnp.zeros(n, dtype=bool)`.

In vendor NetHack, `mklev.c::mkmon` and `makemon.c::makemon` set
`mtmp->msleeping = 1` for every level-gen monster (the `MM_ASLEEP` path).
NLE inherits this — fresh level-1 monsters do not act until they
**wake** via `monmove.c::disturb` / `mon.c::disturb_mon` (LoS + sound +
the `rn2(7)` check in `monmove.c::monster_nearby`).

The wake-check exists in Nethax (`maybe_wake_monster`, line 4729) and
gates on `asleep`, but since `asleep` is never True at spawn time, the
gate is a no-op.  All 5 monsters act on turn 1, every turn, regardless
of distance / LoS.

### H2 — No minimum-distance constraint on monster spawn placement

`Nethax/nethax/dungeon/spawning.py::_pick_valid_tile` (lines 687-707)
samples uniformly over **all** walkable tiles excluding the player tile
— no Chebyshev / room-locality / `enexto`-style buffer.  Vendor
`mklev.c::mkmon` places monsters by-room (one of the level's rooms via
`somexy`) and the typical layout puts them ≥ 6–10 squares from the
player's start room.

Combined with H1, a uniform sample on a 21×80 map can place an awake
hostile monster adjacent to the player on turn 0 with non-trivial
probability.  Under WAIT, that monster melees the player every tick.

### H3 — Player faints far earlier than NLE (secondary, longer-tail)

`Nethax/nethax/subsystems/status_effects.py::apply_starvation`
(lines 997-1039) uses `HungerState.FAINTING` with a 1/10 (10%) faint
roll per turn — `randint(1, 11)` → triggers when the d10 == 1.
Vendor `eat.c::newuhs`/`weight_cap` faint chance is **rn2(20-u.uhunger/10)`
≈ 1/20** at the FAINTING threshold (lines 350-360 of eat.c).  Doubled
faint cadence + FAINTED state means a player rolling toward
starvation loses 1d10 turns to "fainted" idleness while hostiles
gather.  This is not lethal in 50 turns on its own but stacks on H1.

## Why ROGUE alignment=2 is the worst-case probe

- HP_max = 10 + 1d8 (line 720 character.py).  Two melee hits from a
  small hostile (1d6) is enough to kill.
- alignment=2 → chaotic → `_PEACE_MINDED_TABLE` returns hostile for
  most lawful/neutral mid-tier monsters spawned by
  `pick_monster_for_level(depth=1)`.

The same audit on VALKYRIE (HP_max 14 + 1d8, lawful) should show a much
smaller delta from NLE — useful confirmatory probe once JIT cache warm.

## Recommended fix order

1. **Set `asleep = True` for every slot written by
   `populate_level_with_monsters`** (`new_asleep = mai_carry.asleep.at[i]
   .set(jnp.bool_(True))` + a positive `sleep_timer`).  Vendor parity.
2. Constrain `_pick_valid_tile` to exclude an N-tile Chebyshev disc
   around the player on level 1 (`enexto`-style).
3. Halve `apply_starvation` faint probability from 1/10 to 1/20.

After each fix, re-run `tests/test_death_rate_audit.py` (warm cache) and
expect mean ep length to rise toward 191.

## Files & line refs (absolute)

- `/Users/rsiegelmann/Downloads/Projects/nethax/.claude/worktrees/agent-aa63ef9faeead379c/Nethax/nethax/dungeon/spawning.py:803`
- `/Users/rsiegelmann/Downloads/Projects/nethax/.claude/worktrees/agent-aa63ef9faeead379c/Nethax/nethax/dungeon/spawning.py:687` (`_pick_valid_tile`)
- `/Users/rsiegelmann/Downloads/Projects/nethax/.claude/worktrees/agent-aa63ef9faeead379c/Nethax/nethax/subsystems/monster_ai.py:748` (`asleep` default)
- `/Users/rsiegelmann/Downloads/Projects/nethax/.claude/worktrees/agent-aa63ef9faeead379c/Nethax/nethax/subsystems/monster_ai.py:4729` (`maybe_wake_monster`)
- `/Users/rsiegelmann/Downloads/Projects/nethax/.claude/worktrees/agent-aa63ef9faeead379c/Nethax/nethax/subsystems/status_effects.py:997` (`apply_starvation`)
- `/Users/rsiegelmann/Downloads/Projects/nethax/.claude/worktrees/agent-aa63ef9faeead379c/Nethax/nethax/env.py:566` (`_do_step` orchestration)
