# Wave 5 — Monster AI

The single largest Wave-5 deliverable.  Lives in `Nethax/nethax/subsystems/monster_ai.py`.  Called every `env.step` between dispatch and status_effects.

## Architecture

`monster_ai.step` runs `monster_turn` over all 200 monster slots
using `lax.scan`.  Each `monster_turn` invocation:

1. Checks `alive` (skip dead).
2. `maybe_wake_monster` (proximity-based wake).
3. Decides peaceful vs hostile branch.
4. **Pet branch** (peaceful + tame): `pet_move`.
5. **Hostile branch**:
   - `maybe_retreat` (HP<1/7 of max → flee away from player).
   - Else: `monster_use_item` (muse heuristic).
   - Else: `monster_cast_spell` (if mage-class + has spell).
   - Else: `pathfind_step` (BFS toward player).
   - Adjacent-to-player: `monster_attack_player`.

## Line of sight

`monster_can_see_player(state, monster_idx)` — Bresenham line
between `state.monster_ai.pos[monster_idx]` and `state.player_pos`.
Walks the line one tile at a time, stops on opaque tiles (WALL,
CLOSED_DOOR).  Returns bool.

Citation: `vendor/nethack/src/vision.c::couldsee`.

## Pathfind (BFS depth 12)

`pathfind_step(state, monster_idx)` — JIT-friendly BFS with a
hard upper bound of 12 iterations.

Approach: a 21x80 int16 distance field, initialised with the player
tile = 0 and all else = INF.  Each iteration shifts the field
in 4 cardinal directions, taking min(dist, neighbor+1) at passable
tiles.  After 12 iterations, the monster steps along the gradient
toward the player (or holds position if no descent is available).

Why bounded: `lax.while_loop` with an unbounded predicate would
break JIT trace stability.  Depth 12 covers most map shapes; if
the player is further away, the monster falls back to greedy
chebyshev pursuit.

Citation: `vendor/nethack/src/monmove.c::m_move`.  (Vendor uses
Dijkstra with a per-monster reachability cache; we use BFS as a
simpler JIT-friendly approximation.)

## Muse — monster item use

`monster_use_item(state, rng, monster_idx)`:

Per-monster slot for the muse item.  Heuristics from
`vendor/nethack/src/muse.c`:

- **Heal**: if monster HP < 1/2 of max, and it has a potion of
  full healing in its muse slot → `_quaff`.
- **Escape**: if HP < 1/4 of max and not yet retreating → `_quaff`
  a potion of phasing (if held).
- **Attack**: if player adjacent → `_zap` a wand at the player.
- **Read**: scroll of teleportation if held.

Wave 5 simplification: muse-slot population is stubbed.  Real
random `muse_init` (`muse.c::find_misc`) is Wave 6.

## Mcastu — monster spell casting

`monster_cast_spell(state, rng, monster_idx)`:

Mage detection (Wave 5 simplification): monster's `entry_idx` falls
in the "spellcaster" range — entries 130-160 cover most mages /
liches / priests.  Wave 6 should consult the real `MS_SPELL` flag
on `vendor/nethack/include/monst.h`.

Damage formula (from `vendor/nethack/src/mcastu.c::buzzmu`):

    damage = (level / 2 + 1) * d(6)
    capped at player_hp - 1 (vendor reserves the killing blow)

A successful cast drains 5 Pw from the caster (we track this on
`monster_ai.pw[monster_idx]`).  If Pw < 5, the cast is skipped.

## Retreat

`maybe_retreat(state, monster_idx)`:

Returns a target tile `(r, c)` to flee to.  Condition:
`monster_ai.hp[idx] < monster_ai.hp_max[idx] / 7`.

The flee direction is the opposite of the gradient toward the
player.  We take a single greedy step away (Wave 5 simplification —
vendor uses A* with the player as a "negative goal"; we just step
in the opposite direction of the toward-player vector).

Citation: `vendor/nethack/src/monmove.c::find_safe_dirs`.

## Pet AI

`pet_move(state, rng, monster_idx)`:

Tame monsters (peaceful + `tame=True`) follow the player.  The pet:

- If adjacent to player, holds.
- If LoS to player and Chebyshev distance ≤ 6, takes a greedy step
  toward player.
- Else, takes a small random step (vendor uses pet's own
  exploration logic; we simplify to a wander).

Tameness decay (`u.uctame_decay`) is Wave 6.

## Sleep / wake

`maybe_wake_monster(state, monster_idx)`:

If `asleep` and player within Chebyshev distance ≤ 1 → wake.
Asleep monsters do not act on their turn.

Citation: `vendor/nethack/src/monmove.c::dochug` (the sleep gate).

## Wave 5 simplifications (Wave 6 todo)

- BFS bounded at depth 12 (vendor Dijkstra is unbounded).
- Mage detection via entry-index range, not `MS_SPELL`.
- Muse slot fixed per-monster (no random init).
- No "passive-only" monster cases (we always attempt move).
- Pet doesn't pick up items.
- No `MM_SEEINVIS` / `MM_SEEMIMIC` flag handling.
- No `mfndpos` filter for traps (pets can walk into traps).

## Test coverage

`tests/test_monster_ai_depth.py` — 15 tests:

1. LoS clear between empty tiles
2. LoS blocked by wall (Bresenham termination)
3. BFS pathfind moves monster closer
4. BFS pathfind around a wall
5. BFS pathfind respects the depth-12 cap
6. `monster_use_item` quaffs healing potion when HP low
7. `monster_use_item` zaps wand when player adjacent
8. `monster_cast_spell` deals d6-scaled damage
9. `monster_cast_spell` drains 5 Pw
10. `maybe_retreat` fires when HP<1/7 of max
11. `pet_move` keeps pet alive on env.step
12. `maybe_wake_monster` wakes adjacent sleepers
13. Sleeping monster does not act
14. Hostile monster attacks player when adjacent
15. `monster_turn` JIT-compiles end-to-end

All 15 pass.

## Performance

200-slot scan + BFS depth 12 = O(200 * 12 * 21 * 80) ≈ 4 M ops per
step, fully vectorised across the JAX backend.  On CPU this is
≈ 5 ms / step; on GPU ≈ 0.5 ms.
