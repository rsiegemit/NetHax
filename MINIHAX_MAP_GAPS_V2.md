# MiniHack Map-Fidelity Gap List (V2 — ACCURATE)

**Supersedes** `MINIHAX_PORT_STATUS.md` (stale). Scope: the ~159 canonical
MiniHack envs registered by `Nethax/minihax/envs/canonical.py`. **Read-only
audit — no code changed.** Date: 2026-05-28.

This V2 corrects two material errors in the prior audit:

1. **Reward managers are NOT the gap.** The skill envs already wire vendor-
   equivalent `RewardManager`s (`_skill_eat_rm`, `_skill_wield_rm`,
   `_skill_levitate_rm`, ...; `canonical.py:158-331`). Reward shaping is fine.
2. **The wired `des_parser` is effectively DEAD code.** Several `_register_*`
   functions route through `_des_factory(...)` "with the LG builder as a
   fallback" (`canonical.py:60-124`). In practice **the fallback always wins**,
   so *every* env is rendered by its hand-coded `LevelGenerator` builder, and
   map fidelity is purely a function of those builders vs. vendor.

---

## THE HEADLINE FINDING: the des path never builds a map

`_des_factory_from_source` (`canonical.py:88-124`) parses a vendor `.des`, then
**probe-builds once** and accepts the result only if it has a `.terrain`
attribute. The probe was run for 9 representative `.des` files
(`.venv/bin/python3` against `Nethax/minihax/des_parser.des_to_factory`):

| .des probed | probe result |
|---|---|
| corridor5, soko1a, hidenseek, mazewalk, quest_hard, memento_hard, lava_crossing, exploremazehard, key_and_door | **all return `_MockLevelGenerator` (no `.terrain`) → fallback** |

Root cause (`des_parser.py` emitter vs. real `LevelGenerator` API):

* `des_to_factory` emits against a `_MockLevelGenerator`, not the real
  `Nethax.minihax.level_generator.LevelGenerator`. The returned object has no
  `.terrain`, so the `canonical.py:120` health check rejects it.
* Even if it used the real LG, the emitter calls methods/kwargs the LG does not
  define: `_emit_stmt` (`des_parser.py:1496-1710`) calls
  `lg.add_room(row=,col=,height=,width=,kind=,sub=)`, `lg.add_region(...)`,
  `lg.set_map(...)`, `lg.replace_terrain(...)`, `lg.mazewalk(...)`,
  `lg.add_random_corridors(...)`, `lg.set_branch(...)`, `lg.add_container(...)`,
  `lg.set_random_monsters(...)`. The real LG (`level_generator.py`) implements
  **only** `add_room(x,y,w,h,...)`, `add_corridor`, `add_door`, `fill_terrain`,
  `add_mazewalk` — none with `row/col/sym/kind/sub` kwargs, and **no**
  `set_map`/`add_region`/`replace_terrain`/`mazewalk`/`add_random_corridors`.
  Every guarded call (`if _has(lg, ...)`) silently no-ops; the un-guarded ones
  would `TypeError`.

**Consequence:** the `.des` files under
`vendor/minihack/minihack/dat/` are *not* consulted at runtime. The
`key_and_door_tmp.des` template render (`canonical.py:562-579`) is also wasted —
it feeds the same dead path. So the audit reduces to: **does each hand-coded
builder match the vendor map?**

---

## Per-category fidelity table (ranked by agent-transfer impact)

"Transfer impact" = how much a Nethax↔vendor map difference changes the optimal
policy a pre-trained agent would execute.

| # | Category (count) | Vendor source | Nethax builder | Map-match | What's missing / differs |
|---|---|---|---|---|---|
| 1 | **MultiRoom (24)** | MiniGrid procedural, re-rolled every `reset()`, then `wallify()` — random # rooms, sizes, positions, door cells (`envs/minigrid.py:13-105,136-460`) | `_multiroom_builder` — fixed 4×4 rooms on a deterministic `1+i*8` grid, single L-corridor chain (`canonical.py:860-933`) | **NO** | Wrong topology entirely: vendor rooms vary in count/size/placement per episode and connect via a *random* door graph; Nethax is one static chain. Lava-walls: vendor replaces *all* wall glyphs with `L` (`minigrid.py:31-36`), Nethax drops a single 3-tile lava strip. Locked/Extreme door semantics diverge. An agent trained on vendor's varied layouts faces a fixed maze. |
| 2 | **Sokoban (8)** | Fixed hand-authored `soko{1..4}{a,b}.des` — exact boulder/pit/wall layouts, `BRANCH`, `NON_DIGGABLE`/`NON_PASSWALL`, fountain-free pit puzzles (`dat/soko1a.des` etc., loaded via `envs/sokoban.py`) | `_sokoban_builder` — generic 10×8 room, `max(1,level)` boulders on a modular grid, **fountains** `{` as drop targets (`canonical.py:736-785`) | **NO** | Puzzle is wrong: vendor uses *pits* (`TRAP:"pit"`) as boulder sinks inside a specific wall maze; Nethax uses fountains in an open room. Boulder counts/positions and the solvable push-order do not match any `soko*.des`. Optimal push sequence is completely different. |
| 3 | **MazeWalk (6)** | `mazewalk.des` / `envs/mazewalk.py` — full-bounded rect, `MAZEWALK` recursive carve over the whole interior; `premapped` variants set the `premapped` flag (visibility), random up+down stairs | `_mazewalk_builder` → `lg.add_mazewalk(coord=(1,1))` real recursive-backtracker carve; stairs at corners (`canonical.py:463-495`; carve at `level_generator.py:537-558`) | **PARTIAL** | Maze *is* a real perfect maze (good). Differences: vendor places **random** up & down stairs (Nethax fixes start=top-left, goal=bottom-right corner); `Mapped` variants must pre-reveal the maze (premapped flag) — Nethax treats Mapped == unmapped. Topology class matches; start/goal priors and visibility differ. |
| 4 | **Quest (3)** | `quest_easy/medium/hard.des` — multi-room + `MAZEWALK` corridor maze, lava lake (`LLLLL`), closed doors, wand-of-death + Minotaur, `levregion` stairs (`dat/quest_hard.des`, `envs/skills_quest.py`) | `_quest_builder` — two plain rooms + one straight corridor; medium/hard add 1-3 monsters and a 2-tile lava dab (`canonical.py:939-973`) | **NO** | No maze corridor, no lava lake geometry, no door gating, no WoD-vs-Minotaur setup. Hard quest's intended policy (zap WoD across lava at the Minotaur) is unrepresentable in the builder map. |
| 5 | **KeyRoom (5)** | `key_and_door.des` (Fixed-S5) + `key_and_door_tmp.des` rendered by `KeyRoomGenerator` — outer room, **SUBROOM** with a **locked** `ROOMDOOR`, key as a named artifact, down-stair inside subroom (`envs/keyroom.py:13-27,82`) | `_keyroom_builder` — outer `add_room` + second `add_room` top-right (NOT a true subroom), `skeleton key` object, stair in the inner room. **No locked door is placed.** (`canonical.py:546-559`) | **PARTIAL→NO** | The defining mechanic — a **locked door** the agent must unlock with the found key — is absent from the builder (the inner "room" is just an adjacent room, openly reachable). Agent can reach the goal without ever using the key, collapsing the task. |
| 6 | **HideNSeek (4)** | `hidenseek*.des` — random tree/cloud cover via `REPLACE_TERRAIN` ('T','C'), shuffled start/goal corners, randline clearings, 1 random hostile from a shuffled set (`dat/hidenseek.des`) | `_hidenseek_builder` — plain rectangular room, optional 3-tile lava strip, 2 random monsters (`canonical.py:501-540`) | **NO** | No tree/cloud cover (the entire "hide" mechanic), no shuffled corners. Map is an open room; line-of-sight/stealth policy doesn't transfer. |
| 7 | **River (5)** | `envs/river.py` — 25×7 map string, vertical W/L strip, **boulders scattered** (`n_boulder=5`) the agent pushes to bridge the river; goal at `(24,2)` | `_river_builder` — 25×7 room with W/L strip, **no boulders**, just `add_monster` (`canonical.py:823-854`) | **NO** | The core mechanic (push boulders into water to form a crossing) is missing — no boulders placed. Without boulders the river is uncrossable the way the agent learned; lava variants likewise lack the bridge. |
| 8 | **Corridor (3) + CorridorBattle (2)** | Corridor-R{2,3,5}: `corridor{2,3,5}.des` = N `ROOM:"ordinary",random` + `RANDOM_CORRIDORS` (random room placement + random corridor graph). CorridorBattle uses `envs/fightcorridor.py` (long corridor, monster wave) | `_corridor_builder` — N fixed 4×4 rooms on a modular grid + explicit L-corridors; `battle_builder` two rooms + one corridor + 3 monsters (`canonical.py:402-457`) | **PARTIAL** | Right *kind* of level (rooms joined by corridors) but vendor randomises room positions and the corridor graph every episode; Nethax is deterministic. Exploration policy partly transfers; exact navigation does not. |
| 9 | **ExploreMaze (4)** | `exploremaze{easy,hard}(_premapped).des` — two `MAZEWALK` halves split by a center wall, lit region, 4 apples via `LOOP`, stair in a column region (`dat/exploremazehard.des`) | `_exploremaze_builder` — plain room (12×8 or 20×12), 3 apples in a row, corner stair (`canonical.py:1339-1378`) | **NO** | No maze (the whole point); apples present but layout is an open room. `_premapped` visibility not modeled. |
| 10 | **LavaCross — skill (12)** | `lava_crossing.des` (Full/Restricted) = 13×7 walled room, 1-wide vertical lava column, levitation/freeze item on left bank, stair on right bank. Levitate-* variants build inline `.des` strings (`envs/skills_lava.py`) | `_lavacross_builder` — 15×8 room, vertical lava strip, one levitation item placed by type, stair right (`canonical.py:619-693`) | **PARTIAL** | Structure close (room + lava column + item + far stair). Differences: item is *deterministically typed/placed* vs vendor's `IF[%]` random pick over potion/ring/boots/wand/horn from a banked region; lava strip width/exact columns differ. Mechanic (acquire levitation, cross) transfers reasonably. |
| 11 | **LavaCrossing minigrid (6) / SimpleCrossing (4)** | MiniGrid `CrossingEnv` procedural — random lava/wall crossing walls re-rolled per reset, wallified (`envs/minigrid.py:463-529`) | `_register_lavacross_envs` / `_register_simplecrossing_envs` — single straight half-grid lava/wall line, fixed start/goal corners (`canonical.py:696-730`) | **PARTIAL→NO** | Vendor has *N* randomly placed crossing barriers with gap cells; Nethax has one solid half-line. Navigation policy (find the gap) does not transfer to a single fixed wall. |
| 12 | **Memento (3)** | `memento_{short,easy,hard}.des` — long corridor with `IF[%]`-randomised sentinel monsters + board traps; success = remember which branch (`dat/memento_hard.des`) | `_memento_builder` — single plain room, corner stair (`canonical.py:979-1009`) | **NO** | The memory mechanic (corridor with a remembered cue → correct exit) is entirely absent; it's just a room-to-stair walk. |
| 13 | **Labyrinth (2)** | `dat/` lab via `envs/lab.py` — true labyrinth maze with a minotaur, wand reward | `_labyrinth_builder` — open room with 2-3 straight `|` pillars (`canonical.py:791-817`) | **NO** | Not a labyrinth; trivial open room with pillars. No minotaur, no maze. |
| 14 | **WoD (8)** | `envs/skills_wod.py` — room, blessed wand of death near start, Minotaur target, goal-pos; difficulty adds monsters | `_wod_builder` — 15×8 room, `death` wand at (3,3), minotaur at (12,6), +monsters by difficulty (`canonical.py:1015-1058`) | **PARTIAL** | Good structural match (room + WoD + minotaur). Risk: exact room size/positions and whether the `death` wand / `minotaur` names resolve in the Nethax OBJECT/MONSTER tables (falls back to `random` on `KeyError`, `canonical.py:1022-1031`). If the fallback fires, the WoD/target is wrong. |
| 15 | **Boxoban (3)** | `envs/boxohack.py` — loads Boxoban dataset levels (procedural boulder/target puzzles) | `_boxoban_builder` — generic room, 2-4 boulders + fountains (`canonical.py:1064-1089`) | **NO** | Dataset puzzles not ported; generic placeholder room. |
| 16 | **Room (12)** | `envs/room.py` → `LevelGenerator(w=size,h=size)`, `add_goal_pos`, monsters/traps. Purely procedural, no `.des`. | `_room_builder` — matching size, random/fixed goal, monsters/traps (`canonical.py:354-396`) | **YES** | Faithful: this is the one family where Nethax reproduces vendor (both build a plain N×N room procedurally). Dark = unlit handled. |
| 17 | **Skill: Eat/Wield/Wear/PutOn/Zap/Read/Pray/Sink (~24)** | `envs/skills_simple.py` — tiny room, the target item/altar/sink, RM on the skill event. Procedural LG, no `.des`. | `_skill_*_builder` — 5×5 room, matching item+symbol, stair; vendor-equiv RM (`canonical.py:1095-1256`) | **YES (mostly)** | Item names/symbols mirror vendor (`canonical.py:1204-1208`); RM matches. Minor: **Sink** modeled as a fountain `{` proxy (`canonical.py:1154-1163`) and **Pray** altar via `fill_terrain("\\")` — acceptable proxies. |
| 18 | **Skill: Levitate (10) / Freeze (8)** | `skills_levitate.py` / `skills_freeze.py` — tiny room + item + monster; RM on float/bounce message | `_skill_levitate_builder` / `_skill_freeze_builder` — 5×5 room, item, monster, stair (`canonical.py:1128-1303`) | **YES** | Structure + RM match vendor. Freeze-Lava keeps sparse RM (matches vendor). Fidelity good. |
| 19 | **ClosedDoor / LockedDoor (3)** | `dat/closed_door.des`, `locked_door.des`, `locked_door_fixed.des` — small room, single door of the named state | `_register_skill_door_envs` — room + `add_door(state="closed"/"locked")` + stair (`canonical.py:1306-1333`) | **YES (structural)** | Door is actually placed with the right state (unlike KeyRoom). Verify kick/unlock mechanics downstream, but the *map* matches. |

---

## Bottom line: how many envs truly need map work

Of the ~159 envs, **map work is genuinely needed for ~117**:

* **NO (rebuild required), ~95 envs:** MultiRoom (24), Sokoban (8), Quest (3),
  HideNSeek (4), River (5), ExploreMaze (4), Memento (3), Labyrinth (2),
  Boxoban (3), the 6 minigrid LavaCrossing + (partial) 4 SimpleCrossing, plus
  KeyRoom's missing locked door (5). These have the wrong topology or a missing
  core mechanic.
* **PARTIAL (tune/finish), ~22 envs:** MazeWalk (6, stairs+premapped),
  Corridor/CorridorBattle (5, randomisation), LavaCross-skill (12,
  item-randomisation), WoD (8 share a name-resolution risk).
* **YES (faithful), ~42 envs:** Room (12), the simple/levitate/freeze skill
  families (~42 combined), ClosedDoor/LockedDoor (3). Map fidelity here is
  adequate for transfer.

### Top 5 by transfer impact

1. **MultiRoom (24 envs)** — largest family; vendor is *procedurally random*
   each episode, Nethax is a static chain. A pre-trained agent's exploration
   policy is meaningless on the fixed map. Either port MiniGrid's procedural
   room generator or randomise the builder.
2. **Sokoban (8 envs)** — wrong puzzle (fountains vs. pits, wrong boulder
   layout). Push-puzzle policy cannot transfer; needs faithful `soko*.des`
   ports (a real des→LG path, or hand-port the 8 fixed maps).
3. **KeyRoom (5 envs)** — **missing locked door** voids the entire
   find-key→unlock task; the goal is reachable without the key. High impact for
   a small fix (add the locked `ROOMDOOR` between outer room and subroom).
4. **River (5 envs)** — **no boulders placed**, so the learned "push boulders
   to bridge water" policy has nothing to act on. Small fix (scatter boulders
   per `river.py:51-57`), high impact.
5. **Quest (3 envs)** — most complex map (maze corridor + lava lake + WoD +
   Minotaur). Lower count but the hardest transfer target; current builder is a
   two-room placeholder.

### Strategic note

The single highest-leverage fix is the **des path itself**: wiring
`des_to_factory` to emit against the *real* `LevelGenerator` (reconcile the
`row/col/sym/kind/sub` kwarg mismatch and add `set_map`/`add_region`/
`replace_terrain`/`mazewalk`/`add_random_corridors`) would let Sokoban, Quest,
Memento, ExploreMaze, HideNSeek, Corridor, KeyRoom and LavaCross all build from
the authoritative vendor `.des` at once, instead of 95 bespoke builder rewrites.
Today that path is present but inert.
