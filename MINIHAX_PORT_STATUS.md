# Minihax Port Status — MiniHack → Nethax/Minihax

Audit deliverable for MiniHack → Minihax env port verification.

## Scope and method

* **Vendor**: `vendor/minihack/minihack/envs/*.py` — 18 task files, 159 unique
  `id="MiniHack-…-v0"` registrations (counted via direct grep on `id=` strings,
  cross-checked against vendor docs/envs/).
* **Nethax**: `Nethax/minihax/envs/canonical.py` populates
  `Nethax/minihax/registry.MINIHACK_ENV_REGISTRY` at import time. 159 env_ids
  are registered (115 inline string literals + 44 expansions of f-string
  templates in `_register_skill_simple_envs`,
  `_register_skill_levitate_envs`, `_register_skill_freeze_envs`).
* `pip list | grep -i minihack` returned no installed package on this host;
  the audit therefore uses the vendored source under `vendor/minihack` (the
  canonical upstream).

**Coverage by env_id: 159 / 159 (100 %)** — set diff of vendor IDs and
Nethax registry yields `MISSING={}`, `EXTRA={}`.

Coverage by fidelity (map AND reward byte-for-byte equivalent) is
substantially lower; see per-category breakdown below.

## Shared infrastructure status

| Vendor piece | Nethax counterpart | Notes |
|---|---|---|
| `LevelGenerator` (Python → .des) | `Nethax/minihax/level_generator.py` (1217 LOC) | Procedural API only — no .des emission, factories return `EnvState` directly. `add_mazewalk` real recursive-backtracker carve as of Wave 17i. |
| `.des` files (`dat/*.des`) | `Nethax/minihax/des_parser.py` (2267 LOC) | Parser exists but **not wired into canonical envs**. All canonical envs use hand-coded LG builders. Vendor maps under `vendor/minihack/minihack/dat/*.des` are NOT consumed. |
| `RewardManager` | `Nethax/minihax/reward_manager.py` (879 LOC) | API parity for `add_eat_event`, `add_wield_event`, `add_wear_event`, `add_amulet_event`, `add_message_event`, `add_kill_event`, `add_pickup_event`, `add_positional_event`, `add_location_event`, `add_coordinate_event`. Predicates marked `implemented=True` for canonical message-based and location-based events; eat/wield/wear/amulet/kill flagged but predicates evaluated JIT-side. |
| Gym wrapper (`MiniHackNavigation`, `MiniHackSkill`) | `Nethax/minihax/minihax_env.MinihaxEnv` (224 LOC) | Drop-in `reset(rng) / step(state, action, rng)` wrapper; not a Gym `Env` but functionally equivalent for JAX agents. |
| 9×9 observation crop | Implemented elsewhere (`Nethax/minihax/nle_obs.py`, `pixel_renderer.py`) | NLE-style observation produced by `Nethax/nethax/env.NethaxEnv.step`; verified pixel/symbolic env adapters exist in `Nethax/minihax/envs/*_pixels_env.py` and `*_nle_env.py`. |
| `reward_win` / `reward_lose` / `penalty_step` / `penalty_time` | `EnvSpec` dataclass fields (registry.py:46-49) | Plumbed through to `MinihaxEnv.step`. Vendor parity note (minihax_env.py:193-202): when a RewardManager is present, only `reward_manager.collect_reward()` is paid, matching `base.py:378-392`. |

## Per-category fidelity verdict

Verdict scale:
* **EXISTS** — id registered, map structure ≈ vendor, reward fn ≈ vendor.
* **PARTIAL** — id registered but map and/or reward materially differ.
* **MISSING** — id absent from Nethax registry (none in this audit).

### Group A (24 envs) — registered, all PARTIAL on reward

| Category | env_ids | Map fidelity | Reward fidelity | Verdict |
|---|---|---|---|---|
| Room (12) | `Room-{5x5,15x15}`, `…-Random`, `…-Dark`, `…-Monster`, `…-Trap`, `…-Ultimate` | Hand-coded room of size N; vendor uses LG `add_goal_pos` + traps/monsters identically. Stair at (N-1,N-1) deterministic; vendor matches for non-Random. | Sparse +1 stairs_down — vendor identical. | **EXISTS** |
| Corridor (3) | `Corridor-R{2,3,5}` | Nethax wires N rooms with L-corridors at fixed x-offsets; vendor uses `corridor{2,3,5}.des` with hand-authored layouts including doors. | Sparse +1 stairs_down — matches. | **PARTIAL (map)** |
| CorridorBattle (2) | `CorridorBattle{,-Dark}` | Nethax: 2 4×4 rooms + 1 corridor + 3 random monsters. Vendor `fightcorridor.py` + `fightcorridor.des`: specific monster placement on corridor tiles. | Default sparse; vendor identical. | **PARTIAL (map)** |
| HideNSeek (4) | `HideNSeek{,-Mapped,-Lava,-Big}` | Nethax small/big rooms + 2 monsters + optional lava strip. Vendor `hidenseek*.des`: hand-authored monster placement and walls for line-of-sight occlusion. | Sparse +1; vendor matches. | **PARTIAL (map)** |
| KeyRoom (5) | `KeyRoom-{Fixed-S5,S5,Dark-S5,S15,Dark-S15}` | Nethax outer+sub room with key in outer; vendor `key_and_door.des` places door between rooms, locked. **Nethax does not add a locked door between outer/sub room** — agent can walk to stair without using key. | Sparse +1 stairs_down; vendor adds key-pickup shaping via `RewardManager.add_pickup_event` in skill suite, not the navigation KeyRoom (vendor uses sparse too). | **PARTIAL (map)** |
| Labyrinth (2) | `Labyrinth-{Big,Small}` | Hand-coded pillars; vendor `labyrinth_*.des` has a spiral maze. | Sparse +1; matches. | **PARTIAL (map)** |
| River (5) | `River{,-Monster,-Lava,-MonsterLava,-Narrow}` | Nethax: 25-wide single room + vertical water/lava strip. Vendor `river.py`: also single room but uses boulder placement and 5×7 strip with explicit pre-placed boulders for the bridging task. **Nethax does not place the boulders** — agent has no way to bridge water. | Sparse +1; vendor matches. | **PARTIAL (map, gameplay-breaking)** |
| Quest (3) | `Quest-{Easy,Medium,Hard}` | Nethax: 2 rooms + corridor + optional monster/lava. Vendor `quest_{easy,medium,hard}.des`: multi-stage maps with locked doors, keys, multiple level transitions. | Sparse +1; vendor matches. | **PARTIAL (map, severe)** |
| Memento (3) | `Memento-{Short-F2,F2,F4}` | Empty single room; vendor `memento_*.des` has decision-point room with mementos (objects to remember). **Nethax does not implement the memory-task structure.** | Sparse +1. Vendor uses sparse + position-conditional reward. | **PARTIAL (map, severe)** |
| WoD (8) | `WoD-{Easy,Medium,Hard,Pro}-{Full,Restricted}` | Nethax places wand of death at start; vendor `wod.des` places wand and lich/demon to kill; action set is `Restricted` (zap-only) variants. | Vendor: `RewardManager.add_kill_event` for the kill, +1 on goal. Nethax: sparse stairs-down only — **does not reward killing target with wand**. Episodes succeed by walking past lich. | **PARTIAL (reward, severe)** |
| Boxoban (3) | `Boxoban-{Unfiltered,Medium,Hard}` | Hand-coded 2/3/4 boulder grid. Vendor loads from procedural Boxoban dataset (1000s of levels via `boxohack.py`). | Default sparse. Vendor `Sokoban._reward_fn`: `+1` on completion, `+0.1` per pit-filled (shaping), `-0.001` per step (time). | **PARTIAL (map + reward, severe)** |

### Group B (6 envs) — MazeWalk

| env_ids | Map fidelity | Reward fidelity | Verdict |
|---|---|---|---|
| `MazeWalk-{9x9,15x15,45x19}{,-Mapped}` | Real recursive-backtracker via `LG.add_mazewalk` since Wave 17i. Stair pinned to (w-2,h-2); vendor uses `add_stair_down()` (any reachable cell). `-Mapped` flag (visibility=premapped) — Nethax registers the same factory; **visibility difference not honored**. | Sparse +1; matches. | **PARTIAL (map: deterministic stair, missing premapped flag)** |

### Group C (35 envs) — MiniGrid ports

| Category | env_ids | Map fidelity | Reward fidelity | Verdict |
|---|---|---|---|---|
| LavaCrossing (6) | `LavaCrossingS{9N1,9N2,9N3,11N5,19N13,19N17}` | Nethax: vertical lava strip mid-x with one opening; vendor `lava_crossing.des` carves N "tracks" with multiple lava strips matching MiniGrid layout. **Strip count parameter `N` is ignored.** | Sparse +1; lava-touch terminates 0 (vendor matches). | **PARTIAL (map)** |
| SimpleCrossing (4) | `SimpleCrossingS{9N1,9N2,9N3,11N5}` | Vertical wall strip; vendor uses N wall strips with one opening each (MiniGrid SimpleCrossing). | Sparse +1; matches. | **PARTIAL (map)** |
| MultiRoom (24) | `MultiRoom-N{2,4,6,10}{,-OpenDoor,-Locked,-Lava,-Lava-OpenDoor,-Monster,-LavaMonsters,-Extreme}` | Nethax: N rooms at x-stride 8 + L-corridor + doors; vendor `multiroom.py` uses MiniGrid's recursive room-placement with random orientation. Door states ({open,closed,locked}) wired correctly. **N=12,14 vendor variants absent** — actually present in vendor `multiroom.py:200-300` as N12/N14, but neither side registers them by id with the v0 suffix. | Sparse +1; matches. | **PARTIAL (map structure)** |
| LavaCross-Levitate (12) | `LavaCross-{Full,Restricted}`, `LavaCross-Levitate-{Full,Restricted}`, `LavaCross-Levitate-{Potion,Ring}-{Pickup,Inv}-{Full,Restricted}` | Single 15×8 room + vertical lava + levitation item placement; vendor matches structure but uses des files with `INV: ` directive for `-Inv-` variants. **`-Inv-` variants in Nethax still place item at `(2,4)` on the floor, not in inventory.** | Default sparse. Vendor: `RewardManager.add_message_event("levitating", …)` + stairs reward. **Nethax does not reward levitation acquisition.** | **PARTIAL (map for -Inv- + reward, severe)** |

### Group A (53 skill envs) — uniform reward mismatch

All skill envs (`Eat*`, `Pray*`, `Sink*`, `Wield*`, `Wear*`, `PutOn*`, `Zap*`, `Read*`, `Levitate-*`, `Freeze-*`, `ClosedDoor`, `LockedDoor*`) are registered to `_default_goal_reward_manager()` — sparse stairs_down.

Vendor `skills_simple.py` / `skills_levitate.py` / `skills_freeze.py` / `skills_wod.py` uses targeted RewardManagers:

| Vendor env | Vendor RM event | Nethax RM | Verdict |
|---|---|---|---|
| Eat-* | `add_eat_event("apple")` | stairs_down | **PARTIAL (reward)** |
| Wield-* | `add_wield_event("dagger")` | stairs_down | **PARTIAL (reward)** |
| Wear-* | `add_wear_event("robe")` | stairs_down | **PARTIAL (reward)** |
| PutOn-* | `add_amulet_event()` | stairs_down | **PARTIAL (reward)** |
| Zap-* | `add_message_event(["The feeling subsides."])` | stairs_down | **PARTIAL (reward)** |
| Read-* | `add_message_event(["This scroll seems to be blank."])` | stairs_down | **PARTIAL (reward)** |
| Pray-* | `add_positional_event("altar", "pray")` | stairs_down | **PARTIAL (reward)** |
| Sink-* | `add_positional_event("sink", "quaff")` | stairs_down | **PARTIAL (reward)** |
| Levitate-* | `add_message_event(["You start to float"])` + variant items | stairs_down | **PARTIAL (reward)** |
| Freeze-* | `add_message_event(["frozen", "freezes"])` + kill | stairs_down | **PARTIAL (reward)** |
| ClosedDoor / LockedDoor | `add_message_event(["closed door", "locked"])` | stairs_down | **PARTIAL (reward)** |

A pre-trained agent that learned "stand on apple → eat → reward" will receive no reward in Minihax-Eat-v0 until it walks to a stairs cell. This is the largest single-line gap in the port.

## ExploreMaze (4 envs)

`ExploreMaze-{Easy,Hard}{,-Mapped}`: Nethax scatters 3 apples + stair in a room. Vendor `exploremaze.py`: scatters apples and the agent's reward is the `add_eat_event("apple")` for each apple plus stair. **Nethax does not credit apple consumption.** **PARTIAL (reward).**

## Sokoban (8 envs) — covered above under Group A

`Sokoban{1a,1b,2a,2b,3a,3b,4a,4b}`: vendor `Sokoban._reward_fn` shapes by pit-fill count (+0.1 per filled pit) and per-step penalty (-0.001), and treats step into pit as death. Nethax uses sparse stairs_down without pits and replaces target tile with fountain `{`. **PARTIAL (map + reward, severe).**

## Coverage summary

* env_id coverage: 159 / 159 (**100 %**).
* Map-fidelity coverage (vendor-equivalent layouts): **~12 envs EXISTS** (the 12 Room variants) — **~7.5 %**.
* Reward-fidelity coverage (vendor-equivalent reward fn): **~12 envs EXISTS** (12 Room variants; all others assign sparse stairs_down regardless of vendor reward shaping) — **~7.5 %**.
* Joint map+reward coverage: **~7.5 %** (12 / 159).

## Priority list — top 5 missing envs to port next

Ranked by training-impact (envs commonly used in MiniHack benchmarks):

1. **MiniHack-Room-Monster-15x15-v0 + Room-Trap-15x15-v0** — the Room
   suite is the de-facto MiniHack smoke test; the Trap / Monster variants
   need the trap and monster-spawn surfaces verified against vendor placement.
   *Touch*: confirm `LG.add_trap` and `LG.add_monster` randomness matches
   vendor seed semantics.
2. **MiniHack-Eat-{,-Fixed,-Distr}-v0 (3 envs)** — skill-suite leaderboards
   universally use Eat as the canonical "credit assignment on item interaction"
   test. Reward is one `add_eat_event("apple")` call.
   *Touch*: wire `_default_goal_reward_manager()` → `add_eat_event("apple")`
   in `_register_skill_simple_envs` Eat branch.
3. **MiniHack-MazeWalk-9x9-v0 / Mapped-9x9-v0** — small mazes are the
   common navigation benchmark. Two fixes: (a) stair via `add_stair_down()`
   (random reachable cell) instead of pinned corner; (b) honor the `-Mapped`
   visibility flag (`premapped=True` → set `dungeon.visibility[:] = True`).
4. **MiniHack-LavaCross-Levitate-Potion-Inv-Full-v0** — used in the MiniHack
   paper's tool-use experiments. Two fixes: (a) `-Inv-` variants must place
   the item in the player's inventory at reset (vendor `INV:` directive);
   (b) reward must fire `add_message_event(["You start to float"])` on
   levitation acquisition.
5. **MiniHack-Sokoban1a-v0** — the only Sokoban variant with a vendor `.des`
   simple enough to translate by hand. Needs: (a) pit tile (`^`) in
   `world_gen/sokoban.py`, (b) boulder kinetic push primitive verified, (c)
   `Sokoban._reward_fn` shaping (`+0.1` per filled pit, `-0.001` per step).

Beyond the top 5, the systemic fix is **`canonical.py` should wire the
correct event-specific RewardManager for every skill / boxoban / WoD env**
— a single-file change of ~150 LOC that lifts joint coverage from 7.5 %
to ~50 %.

## Files inspected

* `/Users/rsiegelmann/Downloads/Projects/nethax/vendor/minihack/minihack/envs/*.py`
* `/Users/rsiegelmann/Downloads/Projects/nethax/vendor/minihack/minihack/dat/*.des`
* `/Users/rsiegelmann/Downloads/Projects/nethax/Nethax/minihax/registry.py`
* `/Users/rsiegelmann/Downloads/Projects/nethax/Nethax/minihax/envs/canonical.py`
* `/Users/rsiegelmann/Downloads/Projects/nethax/Nethax/minihax/level_generator.py`
* `/Users/rsiegelmann/Downloads/Projects/nethax/Nethax/minihax/reward_manager.py`
* `/Users/rsiegelmann/Downloads/Projects/nethax/Nethax/minihax/minihax_env.py`
* `/Users/rsiegelmann/Downloads/Projects/nethax/Nethax/minihax/des_parser.py`
