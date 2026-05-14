# Wave 5 — Monster AI + Combat Polish + Full Special Levels + Containers + Endgame

**Status:** ✅ Complete · MiniHack 159-env baseline still passing; ~180+ new Wave-5 tests; all 5 Wave-4 deferred skips now unskipped · 10 parallel agents + Phase 0 dispatch wiring + Phase 5 integration pass.

Wave 5 is the **make-it-alive** wave.  Wave 4 delivered the RL benchmark surface (MiniHack) and the breadth-first mechanics scaffold (polymorph, prayer, branches, special-level subset).  Wave 5 finishes the cross-subsystem stubs into a coherent moving world: monsters that act on every step, traps that polymorph piles, quests that match per-role artwork, a full Castle / Vlad / Wizard / Sanctum / 6-demon-lair / Gehennom / 5-Astral-plane endgame progression, containers (bag of holding), and the engrave + genocide + conduct surface.

After Wave 5 the only large remaining work is save/load + ascension polish (Wave 6).

---

## What shipped

| Area | Detail |
|---|---|
| **Dispatch wiring (Phase 0)** | 5 new handler slots: 37 TWOWEAPON, 38 THROW, 39 LOOT, 40 APPLY, 41 ENGRAVE.  Monster-AI step is now called every `env.step` between dispatch and status_effects.  Bump-attack bridge wired into `_try_step` (moving into a monster-occupied tile now routes through `combat.bump_attack`). |
| **Combat polish** | `subsystems/combat.py` — per-slot armor AC bonus table (`do_wear.c::Armor`), `handle_twoweapon` toggle, `thrown_attack` full pipeline (quiver → arc → land), polymorph integration: `bump_attack` now reads `state.polymorph.attack_*` when polymorphed. 9 tests. |
| **Monster AI depth** | `subsystems/monster_ai.py` — LoS via Bresenham, BFS pathfind bounded to depth 12, `monster_use_item` (muse: quaff/read/zap), `monster_cast_spell` (mcastu), `maybe_retreat` (HP<1/7 max → flee), `pet_move` (peaceful pet tracking), sleep/wake. 15 tests. |
| **Major special levels** | `dungeon/special_levels.py` — Castle (drawbridge + wand of wishing), Vlad's Tower (3 levels: lower / middle / top), Wizard's Tower + 3 fakes, Sanctum (Amulet + high priest fight). 15 tests. |
| **Demon lairs** | `dungeon/demon_lairs.py` — Asmodeus (ice palace), Baalzebub (fire pillars), Juiblex (acid pits), Orcus (skeleton hall), Yeenoghu (war camp), Demogorgon (twin-tower swamp). 17 tests. |
| **Gehennom** | `dungeon/branches.py::generate_gehennom_level` — 16-level procedural branch; 12 procedural levels gated by depth, 4 unique inserts (Valley of the Dead L1, Asmodeus L4, Baalzebub L8, Demogorgon L16). Vibrating square → magic portal chain. 8 tests. |
| **Quest** | `dungeon/quest_levels.py` — 13 role-specific quest layouts (Archeologist mines temple, Caveman cave, Healer hospital, Knight tournament field, Monk monastery, Priest cathedral, Ranger forest, Rogue thieves' den, Samurai dojo, Tourist desert, Valkyrie hall, Wizard library, Barbarian cave).  Per-role artifact / leader / nemesis / prefix tables.  `subsystems/quest.py::nemesis_fight`, `return_to_leader`. 15 tests. |
| **Endgame** | `dungeon/endgame.py` — 5 Astral planes: Earth (caverns), Air (almost no floor), Fire (lava lake), Water (pool bubbles), Astral (3 altars).  `subsystems/ascension.py` — Amulet + Astral + matching altar → done + ASCENDED achievement + 50000-point bonus. 19 tests. |
| **Containers** | `subsystems/containers.py` — 4-slot nested inventory (per-container 20-deep stack), bag-of-holding weight multiplier table (blessed 1/4, uncursed 2/4, cursed 8/4), `open / close / put_in / take_from`, LOOT + APPLY action handlers. 10 tests. |
| **Trap bridge** | `subsystems/traps.py` — wide-carrier `lax.switch` from `TRAP_EFFECTS` to subsystem calls (POLY_TRAP, RUST_TRAP, STATUE_TRAP, LEVEL_TELEP, MAGIC_PORTAL, VIBRATING_SQUARE).  12 tests. |
| **Engrave** | `subsystems/engrave.py` — `EngraveState` (per-tile text + kind), `handle_engrave` (Elbereth in dust + ELBERETHLESS conduct).  Plus genocide-scroll handler + extended conduct. 11 tests. |
| **Bump-attack** | `_try_step` now routes movement into a monster tile through `combat.bump_attack`. 5 tests. |
| **Monster step in env** | `monster_ai.step` called from `env.step` between dispatch and status. 3 tests. |
| **Monster scan width** | Held at 200 per-level monster slots; 4 tests check soft-cap on summoning. |
| **NLE compat shim** | `Nethax/nethax/compat/nle_shim.py` — `NLECompat` class wraps `NethaxEnv` so it matches the `nle.nethack.Nethack` `.reset() / .step() / .actions` shape. 4 tests. |
| **Integration tests** | `tests/test_wave5_integration.py` — cross-subsystem smokes including bag-through-env-step, engrave-through-env-step, genocide-through-env-step, two-weapon, throw, monster pathfinds around walls, pet follows player, all 17 obs keys post-Wave-5, jit compile over all 8 new action ids, cross-branch round-trip bit-equal. |
| **Wave 4 → 0 deferred skips** | All 5 Wave-4 deferred-skip tests (bump-attack bridge, trap dispatch, monster_kills_player, player_kills_monster, hunger 700-turn) now pass. |
| **Total Wave 5 footprint** | ~180+ new tests, roughly 10 K LoC across the expanded subsystems.  611 (Wave 4) → 790+ (Wave 5). |

---

## How to use Wave 5

```python
import jax
import jax.numpy as jnp
from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants.actions import Command

env = NethaxEnv()
state, obs = env.reset(jax.random.PRNGKey(0))

# Monster AI now ticks every env.step.
rng = jax.random.PRNGKey(1)
state, obs, reward, done, info = env.step(state, jnp.int32(ord(".")), rng)

# Engrave Elbereth.
state, obs, reward, done, info = env.step(
    state, jnp.int32(int(Command.ENGRAVE)), jax.random.PRNGKey(2),
)
# state.conduct.violations[Conduct.ELBERETHLESS] == True
```

NLE drop-in:

```python
from Nethax.nethax.compat.nle_shim import NLECompat
nh = NLECompat(seed=0)
obs = nh.reset()           # 17-key dict
obs, reward, done, info = nh.step(action=ord("."))
```

Cross-branch portal:

```python
from Nethax.nethax.dungeon.level_memory import traverse_portal
from Nethax.nethax.dungeon.branches import Branch
state = traverse_portal(state, rng, target_branch=int(Branch.GEHENNOM), target_level=1)
```

---

## Doc set

| # | File | Covers |
|---|---|---|
| 1 | [`README.md`](README.md) | This file |
| 2 | [`mechanics-status.md`](mechanics-status.md) | Per-subsystem Wave-4 → Wave-5 deltas |
| 3 | [`combat-polish.md`](combat-polish.md) | Per-slot AC, two-weapon, thrown, polymorph integration |
| 4 | [`monster-ai.md`](monster-ai.md) | LoS / BFS / muse / mcastu / retreat / pet / sleep-wake |
| 5 | [`special-levels.md`](special-levels.md) | Full inventory of 35+ special-level factories |
| 6 | [`quest-impl.md`](quest-impl.md) | 13 role quests + per-role table + nemesis fight |
| 7 | [`endgame.md`](endgame.md) | 5 Astral planes + ascension condition + scoring |
| 8 | [`integration-issues.md`](integration-issues.md) | Issues caught during Wave 5 (TileType enum collision, polymorph regression, action-slot sequencing) |
| 9 | [`decisions.md`](decisions.md) | Wave 5 design decisions |
| 10 | [`gaps.md`](gaps.md) | Wave 5 → Wave 6 backlog |
| 11 | [`test-results.md`](test-results.md) | Per-file Wave 5 additions, the 5 → 0 Wave 4 skip transition |
| 12 | [`next-wave.md`](next-wave.md) | Wave 6 scope preview |

---

## What's NOT in Wave 5 (deferred to Wave 6)

- **Save / load** — `Nethax/nethax/save_load.py` skeleton exists, full serialization Wave 6.
- **Full scoring** — vendor's `end.c` topten formula; Wave 5 uses a flat 50000-point ascension bonus.
- **Death message generation** — `done()` with `killer_name` / `how_killed` plumbing.
- **Shop simplified buy/sell + angry shopkeeper** — Wave 6.
- **`inv_strs` polish** — named items / vowel article / irregular plurals.
- **WISHLESS + ARTIWISHLESS conducts** — gated on wish handler (Wave 6).
- **Object table canonicalize** — drop dual-naming, 503 → 453.
- **Monster table trim** — 382 → 381.
- **Per-role bonus tables** — Monk martial arts, Samurai bushido, etc.
- **Throughput benchmark** — steps/sec measurement.

See [`gaps.md`](gaps.md) for the full backlog.
