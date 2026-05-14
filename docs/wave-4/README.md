# Wave 4 — MiniHack + Branches + Polymorph + Prayer + Obs Polish

**Status:** ✅ Complete · **611 / 611 pytest tests passing** (5 skipped, 0 failed) · MiniHack curriculum + dungeon branches + polymorph + prayer + features + observation polish + conduct wiring · 9 parallel agents + Phase 0 dispatch wiring + integration pass.

Wave 4 is the **RL-runnable** wave. After Wave 3 produced a full mechanics-complete NetHack, Wave 4 stacks the canonical MiniHack 170-env benchmark surface on top, lights up the cross-branch dungeon (Main ↔ Mines / Sokoban / Quest), gives polymorph + prayer real game-effect tables, ships the Wave-4 subset of special levels (Oracle, Mine Town, Mines End, Big Room), wires fountain / throne / sink / altar interactions, propagates 8 of 13 conducts, and finishes the obs surface (all 17 NLE keys real-valued).

`MinihaxEnv("MiniHack-Room-5x5-v0")` now boots, resets to a populated `EnvState`, and `env.step(action, rng)` runs the full pipeline including action dispatch, status ticks, and reward-manager evaluation — all `jax.jit`-compatible.

---

## What shipped

| Area | Detail |
|---|---|
| **Dispatch wiring (Phase 0)** | 16 new handler slots in `subsystems/action_dispatch.py::_HANDLERS`: EAT (20), QUAFF (21), READ (22), ZAP (23), CAST (24), PICKUP (25), DROP (26), WIELD (27), WEAR (28), PUTON (29), REMOVE (30), OPEN (31), CLOSE (32), KICK (33), FIGHT (34), SEARCH (35), PRAY (36). `status_effects.step` ticks every `env.step`. |
| **LevelGenerator** | `Nethax/minihax/level_generator.py` — `add_room`, `add_monster`, `add_trap`, `add_object`, `add_stair_down/up`, `set_start_pos`, `fill_terrain`. Returns a callable `(rng) -> EnvState`. ~900 LoC + 8 tests. |
| **RewardManager** | `Nethax/minihax/reward_manager.py` — 12+ event factories: `add_eat_event`, `add_kill_event`, `add_message_event`, `add_pickup_event`, `add_wield_event`, `add_wear_event`, `add_coordinate_event`, `add_location_event`, `add_custom_reward_fn`, `add_amulet_event`, `add_levitate_event`, `add_positional_event`.  Per-event `repeatable`, `terminal_required`, `terminal_sufficient` flags. ~600 LoC. |
| **des-file parser** | `Nethax/minihax/des_parser.py` — full parser for the 36 canonical `*.des` files under `vendor/minihack/minihack/dat/`.  Compiles to `LevelGenerator`-shaped builders. 2267 LoC. |
| **159-env registry** | `Nethax/minihax/{registry,minihax_env,envs/canonical}.py` — every canonical MiniHack env_id registered, indexed by `MINIHACK_ENV_REGISTRY[env_id] -> EnvSpec`. Counts per category in [`minihack-impl.md`](minihack-impl.md). |
| **Dungeon branches** | `Nethax/nethax/dungeon/branches.py` — Main ↔ Mines ↔ Sokoban ↔ Quest with `init_branch_graph`, `generate_mines_level` (CA caves), `generate_sokoban_level` (8 hand layouts), `generate_quest_level` (per-role guardian). `level_memory.traverse_stair_cross_branch` wires cross-branch ascent/descent and caches per-level state. 1023 LoC. |
| **Polymorph** | `subsystems/polymorph.py` — full-fidelity `polymorph_player` (NATTK=6 attack set swap, intrinsic mask, AC recompute, armor-drop on no-hands, POLYSELFLESS conduct), `polymorph_monster` (per-slot form change), lycanthropy timer, `poly_trap_effect`, `step` orchestrator decrementing `poly_timer` and `lycanthropy_timer`, auto-revert at expiry. 20 tests. |
| **Prayer** | `subsystems/prayer.py` — `pray()` runs the full d100 outcome chain from `pray.c:500-1500`: `_detect_trouble` → fix path, pleased buckets (heal/protection/remove-curse/gift), god-anger paths (`god_zaps_you`), alignment threshold gate, pray-timeout management.  `sacrifice_on_altar`, `altar_buc_sense`, `handle_pray` (action-dispatch entry).  12 tests. |
| **Features + special levels** | `subsystems/features.py` — fountain (`quaff_fountain` 16-outcome, `dip_fountain` 8-outcome), throne (`sit_on_throne` 14-outcome), sink (`drink_sink` 13-outcome). `dungeon/special_levels.py` — Oracle (delphi + fountains), Mine Town (shops + altar + watchmen), Mines End (luckstone), Big Room. 34 tests across features + special-levels. |
| **Conduct wiring** | 8 of 13 conducts triggered: FOODLESS (eat), VEGAN/VEGETARIAN (eat by material), ATHEIST (pray), WEAPONLESS (melee w/ weapon), PACIFIST (any kill), ILLITERATE (read scroll/spellbook), POLYSELFLESS (polymorph_player). 5 still TODO (no underlying feature yet). |
| **Observation polish** | `obs/nle_obs.py` — last 4 keys real-valued: `colors` paints terrain + player tile, `specials` flags trap/pile/corpse/object bits, `internal` carries stairs_down/hunger/dlevel, `screen_descriptions` per-glyph names. 17/17 keys now project state. |
| **Integration tests** | `tests/test_wave4_integration.py` — 15 cross-subsystem tests: MinihaxEnv full episode, cross-branch descend/return, polymorph timer revert, prayer through env.step, conduct propagation, obs surface, custom reward manager, JIT-compile across 5 dispatched actions, Oracle factory, full step lifecycle.  All 15 passing. |
| **Total Wave 4 footprint** | **~158 new tests** (453 → 611). Roughly 9 K LoC across the new minihax package and the expanded subsystems. |

---

## How to use Wave 4

```python
import jax
from Nethax.minihax.minihax_env import MinihaxEnv

env = MinihaxEnv("MiniHack-Room-5x5-v0")
state, info = env.reset(jax.random.PRNGKey(0))

# Step through the env. Action ids match vendor cmd.c (ord('e') for EAT, etc.).
state, reward, done, info = env.step(
    state, action=ord("."), rng=jax.random.PRNGKey(1),
    fired_mask=info["fired_mask"], step_count=info["step_count"],
)

# All 17 NLE obs keys are populated via build_nle_observation(state).
from Nethax.nethax.obs.nle_obs import build_nle_observation
obs = build_nle_observation(state)
assert set(obs.keys()) >= {"glyphs", "blstats", "colors", "inv_strs", "internal"}
```

Custom reward shape:

```python
from Nethax.minihax.reward_manager import RewardManager
rm = RewardManager()
rm.add_kill_event(monster_name="dwarf", reward=1.0)
rm.add_coordinate_event(2, 4, reward=2.0, terminal_sufficient=True)
env = MinihaxEnv("MiniHack-Room-5x5-v0", reward_manager=rm)
```

Cross-branch traversal (Main → Mines):

```python
from Nethax.nethax.dungeon.level_memory import traverse_stair_cross_branch
state = traverse_stair_cross_branch(state, rng, target_branch=-1, direction=+1)
# state.dungeon.current_branch is now Branch.GNOMISH_MINES.
```

---

## Doc set

| # | File | Covers |
|---|---|---|
| 1 | [`README.md`](README.md) | This file |
| 2 | [`minihack-impl.md`](minihack-impl.md) | MiniHack delivery: LevelGenerator + RewardManager + des-parser + 159-env registry |
| 3 | [`mechanics-status.md`](mechanics-status.md) | Per-subsystem Wave-3 → Wave-4 deltas: what moved from stub / simplified to real |
| 4 | [`integration-issues.md`](integration-issues.md) | The 600s watchdog incident + polymorph step() regression caught by `test_no_op_step` |
| 5 | [`decisions.md`](decisions.md) | Wave 4 design decisions + tradeoffs |
| 6 | [`gaps.md`](gaps.md) | Remaining TODOs → Wave 5 / Wave 6 backlog |
| 7 | [`test-results.md`](test-results.md) | Full pytest breakdown: 611 passing, 5 skipped, 0 failed |
| 8 | [`next-wave.md`](next-wave.md) | Wave 5 scope preview: monster AI, combat polish, Quest fidelity, full special-level set |

---

## What's NOT in Wave 4 (deferred to Wave 5 / 6)

- **Monster AI step in `env.step`** — `monsters_step_all` exists but is not called from the env loop (Wave 5).
- **Bump-attack bridge** in `_try_step` for monster-occupied tiles (Wave 5; combat.melee_attack still callable directly).
- **5 conducts** (POLYPILELESS, WISHLESS, ARTIWISHLESS, GENOCIDELESS, ELBERETHLESS) — gated on features that aren't built yet (poly-pile path, wish handler, engrave action, genocide scroll).
- **Castle / Valley / Vlad's / Sanctum** — Wave 5.
- **Vibrating square + Gehennom entrance** — Wave 5.
- **Bag-of-holding / containers** — Wave 5.
- **Per-role starting kits beyond Wave 3's `STARTING_INVENTORY`** — Wave 5.
- **Save / Load** — Wave 6.
- **Ascension end-game** — Wave 6.

See [`gaps.md`](gaps.md) for the full backlog.
