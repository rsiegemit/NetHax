# Wave 3 — Mechanics Status

What each subsystem does now, plus what's still simplified or deferred.

Legend: ✅ real · ⚠️ simplified · ⏳ still no-op (waiting on later wave)

## Combat (`subsystems/combat.py`)

| Mechanic | Status | Notes |
|---|---|---|
| `compute_ac(state)` | ✅ | Sums armor contributions over 7 worn slots via `lax.scan`. `ARM_BONUS = ac_bonus + enchantment` (do_wear.c:2473) |
| `to_hit_roll(rng, attacker, target_ac)` | ✅ | `tmp = 1 + abon + target_ac + skill_bonus + enchant`; hit iff `rnd(20) ≤ tmp`. (uhitm.c:365) |
| `damage_roll(rng, weapon, target_size_large, sdam, ldam, str_bonus)` | ✅ | sdam vs ldam by target size, + enchantment + STR bonus (weapon.c:215) |
| `_abon(str, dex, xl)` STR+DEX bonus | ✅ | Full table (weapon.c:950) |
| `_dbon(str)` STR damage bonus | ✅ | Full table (weapon.c:992) |
| `melee_attack(state, rng, target_idx)` | ✅ | Orchestrates roll + damage + monster HP write + skill practice |
| `bump_attack(state, rng, target_pos)` | ✅ | Looks up alive monster at target; calls melee_attack; player advances onto tile if monster died |
| `monster_attack_player(state, rng, monster_idx)` | ✅ | Symmetric: monster rolls vs player AC, applies damage |
| `practice_skill(state, weapon_type)` | ✅ | Counter advancement, tier promotion at threshold (weapon.c:1198) |
| Two-weapon | ⏳ Wave 4 |
| Ranged (throw/fire) | ⏳ Wave 4 |
| Breath weapons | ⏳ Wave 4 |
| Engulf / passive | ⏳ Wave 4 |
| Polymorph combat | ⏳ Wave 4 |

## Magic (`subsystems/magic.py`)

| Mechanic | Status | Notes |
|---|---|---|
| `cast_spell(state, rng, spell_id)` | ✅ | Pw check + d100 success roll + effect dispatch (spell.c:spelleffects) |
| `spell_fail_chance(role, spell_id, xl, INT, WIS)` | ✅ | Full formula (spell.c:percent_success) — skill assumed UNSKILLED |
| 43 spell effects | ✅ | Each is a small JAX function. Heal/missile/fire bolt/cold/death/teleport/poly/levitate/etc. |
| `pw_regen_tick(state, ...)` | ✅ | Interval `max(1, 30-XL)` (Wizard/Healer: `max(1, 20-XL)`); halved with ENERGY_REGEN |
| `read_spellbook(state, rng, slot)` | ✅ | d20 study check + INT bonus + book level (spell.c:study_book) |
| Spell memory decay | ⚠️ | Wave 3: −1 per cast; full per-school table = Wave 6 |
| Wishing | ⚠️ | Recharges wands to 15 (placeholder); full wish parser would need menu — Wave 6 |
| Monster casting (mcastu) | ⏳ Wave 4 |

## Monster AI (`subsystems/monster_ai.py` + `dungeon/spawning.py`)

| Mechanic | Status | Notes |
|---|---|---|
| `monster_turn(state, rng, idx)` | ✅ | Sleeping/peaceful guard, Chebyshev-≤1 → attack, otherwise greedy 8-dir step |
| `monsters_step_all(state, rng)` | ✅ | `lax.scan` over 200 slots |
| `wake_monsters_near(state, pos, radius=3)` | ✅ | Vectorized chebyshev check, flips `asleep` |
| `MONSTR_DIFFICULTIES` table | ⚠️ | Wave 3 uses `entry.level` as proxy; full monstr formula = Wave 5 |
| `eligible_monsters_for_depth(depth)` | ✅ | `difficulty ∈ [depth-6, depth+5]` + ~G_NOGEN + ~G_UNIQ |
| `pick_monster_for_level(rng, depth)` | ✅ | Weighted choice over gen_freq |
| `spawn_initial_monsters(rng, depth, n, valid_mask)` | ✅ | `lax.fori_loop` over n slots |
| `populate_level_with_monsters(state, rng, n=5)` | ✅ | Wired into `env.reset()` |
| Sleeping monsters | ✅ | Don't move; combat wakes adjacent |
| `muse` (monster item use) | ⏳ Wave 4 |
| Monster spell casting | ⏳ Wave 4 |
| Retreat / pickup behavior | ⏳ Wave 4 |
| Pet AI | ⏳ Wave 4 |
| Quest leader / nemesis behavior | ⏳ Wave 5 |

## Inventory + character (`subsystems/inventory.py` + `subsystems/character.py`)

| Mechanic | Status | Notes |
|---|---|---|
| `Item` schema | ✅ | 10 fields: category/type_id/buc_status/enchantment/charges/identified/quantity/weight/ac_bonus/is_two_handed |
| `InventoryState.empty()` / `from_items()` | ✅ | Batched [52] Item array |
| `pickup(state, rng, ground, b, l)` | ✅ | First-empty-slot scan, ground tile cleared |
| `drop(state, rng, slot)` | ✅ | Inv slot zeroed, ground stack updated |
| `wield(state, slot)` | ✅ | Updates `wielded`, swaps off-hand for two-handed |
| `wear_armor(state, slot, armor_slot)` | ✅ | Updates `worn_armor` + AC |
| `take_off_armor(state, armor_slot)` | ✅ | Reverse, with AC recompute |
| `total_weight(items)` | ✅ | `lax.scan` over 52 slots |
| Cursed item locking | ⏳ Wave 4 |
| Enchantment/erosion on durability | ⏳ Wave 4 |
| Bag-of-holding | ⏳ Wave 5 |
| `STARTING_INVENTORY` × 13 roles | ✅ | All from `u_init.c::trobj` arrays |
| `STARTING_STATS` × (Role, Race) | ✅ | Roll ranges from `role.c::roles[]` |
| `STARTING_HP_PW` × Role | ✅ | (hp_base, hp_per_level, pw_base, pw_per_level) |
| `create_character(rng, role, race, alignment)` | ✅ | Returns dict for `state.replace(**)` |

## Item effects

| Category | Count | Status |
|---|---|---|
| Potions | 26 | ✅ All effects (heal/gain/restore/detect/see-invis/levitate/paralyze/sleep/confuse/hallu/blind/sick/acid/oil/poly/water/booze/fruit-juice/enlighten/invisibility/speed/object-detection/monster-detection) |
| Scrolls | 23 | ✅ All effects (identify/light/enchant-weapon/enchant-armor/remove-curse/scare/teleport/magic-mapping/gold/food/blank/mail/destroy-armor/confuse/create-monster/taming/genocide/amnesia/fire/earth/punishment/charging/stinking-cloud) |
| Wands | 28 | ✅ All effects (light/nothing/secret-door/opening/locking/probing/magic-missile/striking/slow/speed/cancellation/poly/teleport/death/sleep/cold/fire/lightning/digging/enlightenment/create-monster/wishing/stasis/make-invisible/undead-turning/draining/acid/poison-gas) |
| Rings | 28 | ✅ All intrinsic-granters wear/take-off |
| Amulets | 13 | ✅ All intrinsic-granters; LIFE_SAVING flag; YENDOR for ascension |
| Spells | 43 | ✅ All cast effects (sharing dispatch with wands where applicable) |

Many simplifications individual to specific effects — see [`item-effects.md`](item-effects.md).

## Status effects (`subsystems/status_effects.py`)

| Mechanic | Status |
|---|---|
| `tick_timers` (decrement 24 timed statuses) | ✅ |
| `compute_hunger_state(nutrition)` | ✅ — threshold table from eat.c |
| `hunger_tick(state)` | ✅ — handles HUNGER_RING + SLOW_DIGESTION |
| `compute_encumbrance(weight, capacity)` | ✅ — STR-based capacity formula |
| `hp_regen_tick(state, ...)` | ✅ — interval `max(1, 20-XL)`; REGEN halves |
| `pw_regen_tick(state, ...)` | ✅ — interval `max(1, 30-XL)`; Wizard/Healer faster |
| `apply_starvation` → faint/death | ✅ |
| `apply_strangulation` → death | ✅ |
| `apply_stoning` → death | ✅ |
| `apply_sliming` → death | ✅ (transform to slime = Wave 4) |
| `apply_food_poisoning` → death | ✅ |
| `step(state, rng)` orchestrator | ✅ |
| `handle_eat(state, ...)` | ✅ |

## Traps (`subsystems/traps.py`)

All 26 trap types have damage + side-effects:

| Trap | Implementation |
|---|---|
| ARROW_TRAP | d6 piercing |
| DART_TRAP | d3 piercing |
| ROCKTRAP | d2 + d20 |
| SQKY_BOARD | wake_monsters_near |
| BEAR_TRAP / WEB / PIT | freeze player d4-d6 turns |
| LANDMINE | d6 + d6 + d2 fire |
| SLP_GAS | sleep d20 turns |
| FIRE_TRAP | d2 fire |
| SPIKED_PIT | freeze + d4 |
| TELEP_TRAP | random valid-tile teleport |
| LEVEL_TELEP | same-level teleport (Wave 4: cross-level) |
| MAGIC_TRAP | d20-roll outcome table (6 effects in Wave 3) |
| ANTI_MAGIC | drain d6 Pw |
| HOLE / TRAPDOOR | d6 damage (Wave 4 wires level-fall) |
| RUST_TRAP / STATUE_TRAP / POLY_TRAP | stub (Wave 4) |
| VIBRATING_SQUARE | revealed flag only |
| MAGIC_PORTAL | stub (Wave 5 wires branch jump) |

## Features (`subsystems/features.py`)

| Mechanic | Status |
|---|---|
| `open_door` (bump-to-open) | ✅ — wired in `_try_step` |
| `close_door` | ✅ |
| `kick_door` (d6 vs strength) | ✅ |
| `unlock_door` (key slot check) | ✅ |
| `door_blocks_movement` | ✅ |
| Fountain quaff / dip | ⏳ Wave 4 (effect table) |
| Sit on throne | ⏳ Wave 4 |
| Kick sink | ⏳ Wave 4 |
| Altar interactions | ⏳ Wave 4 (prayer integration) |
| Drawbridge / secret doors | ⏳ Wave 5 |

## Observation (`obs/`)

| Key | Status |
|---|---|
| `glyphs` (21,79) int16 | ✅ |
| `chars` (21,79) uint8 | ✅ |
| `colors` (21,79) uint8 | ✅ ANSI 0-15 lookup |
| `specials` (21,79) uint8 | ⚠️ zeros (trap/corpse/pile flags = Wave 4) |
| `blstats` (27,) int64 | ✅ all 27 fields populated |
| `message` (256,) uint8 | ✅ |
| `program_state` (6,) int32 | ⏳ Wave 4 |
| `internal` (9,) int32 | ⏳ Wave 4 |
| `inv_glyphs` (55,) int16 | ✅ batched item glyphs |
| `inv_letters` (55,) uint8 | ✅ slot index → 'a'-'z'A'-'Z' |
| `inv_oclasses` (55,) uint8 | ✅ from item.category |
| **`inv_strs` (55, 80) uint8** | ✅ **full NLE fidelity** — BUC word, enchant, name, equip status, charges |
| `screen_descriptions` (21,79,80) | ⏳ Wave 4 |
| `tty_chars` (24, 80) uint8 | ✅ |
| `tty_colors` (24, 80) int8 | ✅ |
| `tty_cursor` (2,) uint8 | ✅ |
| `misc` (3,) int32 | ⏳ Wave 4 |

13 of 17 keys are real-valued; 4 still zeros (program_state, internal, screen_descriptions, misc, specials — Wave 4 territory).

## Action dispatch (`subsystems/action_dispatch.py`)

Wired in Wave 3: movement (8 dir + 8 run + stairs + wait), trap-trigger on step, door-bump-open in `_try_step`. Other action handlers exist as `handle_*` functions in their respective subsystems but are not yet wired into `_HANDLERS` / `_ACTION_TO_HANDLER_IDX`. **Wave 4 integration task**: wire `handle_quaff`, `handle_read`, `handle_zap`, `handle_cast`, `handle_eat`, `handle_pickup`, `handle_drop`, `handle_wield`, `handle_wear`, `handle_put_on`, `handle_remove`, etc.
