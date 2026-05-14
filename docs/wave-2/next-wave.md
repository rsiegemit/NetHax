# Wave 2 → Wave 3 — Scope preview

Wave 3's job: **wire up the gameplay**. After Wave 2 the player walks around an empty dungeon; after Wave 3 there are items, combat, magic, monsters that move, and status effects that tick.

> Confirm or redirect before launching `/ultrawork` for Wave 3.

## Wave 3 deliverable

A `NethaxEnv` where, after `reset()`:
- Items spawn on the floor (a few, in random rooms).
- Player has a starting weapon and basic gear in inventory.
- Monsters spawn and **move toward / attack** the player (basic AI).
- Player can **bump-attack** monsters; damage is real (THAC0 + AC + damage dice + STR bonus).
- Player can `pickup`, `wield`, `wear`, `drop`, `eat`, `quaff`, `read`, `zap`, `cast`.
- Hunger ticks down; status effects (blind, conf, sleep, etc.) tick and expire.
- The 6 observation keys still zero in Wave 2 (`colors`, `tty_colors`, `inv_*`, `specials`) get real values.
- Doors open on bump; can be kicked.

This is "the game starts working" wave. After Wave 3, RL agents can train meaningfully on what's effectively an early-game-NetHack with full combat/magic/items.

## Wave 3 breadth pass

### Item effects (biggest single item)
- Potion effects: ~26 entries. Each is a small JAX function on `EnvState`.
- Scroll effects: ~23 entries. Some (identify, magic mapping) are state-mutating.
- Wand zap: 28 wands, each with a ray or self-target. `lax.scan` along beam.
- Ring intrinsics: 28 entries. Wearing grants intrinsic; removal revokes.
- Amulet intrinsics: 13 entries.
- Spellbook learning: 44 entries, fails by spell-failure formula.

### Combat fidelity
- THAC0 roll (`vendor/nethack/src/uhitm.c::find_roll_to_hit`)
- AC computation from worn armor + cloak + shield + helm + boots + gloves + shirt
- Damage roll (small / large monster dice + STR bonus + enchantment + role bonus)
- Weapon skill tiers (Basic / Skilled / Expert / Master / GrandMaster)
- Practice counter advancement on hit
- Two-weapon penalty
- Two-handed enforcement
- Ranged attacks: throw, fire (auto-quiver), wand zap
- Monster passive attacks (cockatrice touch, fire elemental burn)

### Magic
- `cast_spell`: Pw cost → d100 failure roll → effect dispatch
- Each spell's effect (parallel to potion effects)
- Spellbook reading: chance to learn vs fail vs lose page
- Spell memory decay (per-spell, per-turn)
- Pw regen formula (per role × XL)

### Monster AI
- Movement points accumulation
- Greedy 8-direction pathfinding toward player
- Bump-attack on player
- Sleeping monsters wake on disturbance
- Simple strategy selection (hunt if visible, wander otherwise)

### Inventory operations
- `pickup`: weight check, capacity check, slot assignment
- `drop`: remove from slot, place on `ground_items`
- `wield`: enforce two-handed, swap off-hand
- `wear`: AC update
- `take_off`: AC update, cursed check
- `put_on_ring`: cursed prevention, grant intrinsic
- `quiver`, `swap_weapons`

### Status effects
- Hunger threshold table (vendor `eat.c`)
- Encumbrance formula (carry weight + STR table)
- HP regen formula (per role × XL × CON)
- Pw regen formula
- `tick_timers` expiry callbacks: Stoning → death, Strangulation → death, Confusion → end, etc.
- Sickness progression (food poisoning → death in N turns)

### Traps
- Damage traps trigger: ARROW_TRAP, DART_TRAP, ROCKTRAP, PIT, SPIKED_PIT, SLP_GAS_TRAP, FIRE_TRAP, LANDMINE
- Web (immobilize)
- Bear trap (immobilize)
- (Teleport / poly / vibrating-square = Wave 4)

### Door bump-to-open
- Movement onto CLOSED_DOOR → open it (set tile to OPEN_DOOR), don't move
- Onto LOCKED_DOOR → need key / lockpick / force / kick (Wave 4 adds kick)
- Re-enable `place_doors` in dungeon pipeline

### Observation polish
- `colors` — per-tile color lookup
- `tty_colors` — same data, NLE format
- `specials` — corpse / statue / pile flags per tile
- `inv_glyphs` — glyph for each inventory item
- `inv_letters` — assigned letter per slot
- `inv_oclasses` — object class per slot
- `inv_strs` — rendered names ("blessed +2 long sword named Sting (weapon in hand)")

### Player initialization
- `reset(rng, role: Role, race: Race, alignment: Alignment)` — proper character creation
- Starting inventory per role (Valkyrie: long sword + small shield + ring mail, Wizard: quarterstaff + cloak of magic resistance + spellbooks, etc.)
- Starting HP / Pw / stats from role × race × XL=1 tables
- Starting alignment record

### Wave 3 tests
- `test_combat.py` — hit/miss/damage formulas vs vendor reference
- `test_items.py` — each item effect applied correctly
- `test_inventory.py` — pickup/drop/wield/wear all preserve invariants
- `test_status_effects.py` — hunger ticks, intrinsic application
- `test_monster_ai.py` — monsters move toward player
- `test_role_init.py` — each role produces correct starting state

### Object table cleanup
- Drop verbose-named potions/scrolls/wands from `OBJECTS_BASE`, keep only canonical bare names
- Expected: OBJECTS count drops from 503 → ~470

### Migrate `Nethax/minihax/` to new `NethaxEnv`
- Replace `minihax/envs/*.py` references to old state with new state
- Wave 5 reuses these for MiniHack curriculum

## Wave 3 risks

1. **Item effect dispatch is large**: 26 + 23 + 28 + 28 + 13 + 44 = **162 distinct effects** to implement. Each is small but the total surface is large. Mitigation: dispatch each category to its own agent in parallel.
2. **Combat formula precision**: NetHack to-hit / damage has many edge cases. Property tests against vendor C source are non-trivial. Mitigation: implement core path first; defer edge cases to Wave 6.
3. **Monster AI re-trace cost**: each monster turn becomes a `lax.switch` over strategies. At 200 monsters × 32 levels = 6400 potential turns per game tick. Compilation may slow. Mitigation: batch all monster turns into one `lax.scan` over monster indices.
4. **Hunger / Pw regen makes timestep ticks non-trivial**: each step ticks ~5 counters. Need to batch these per-tick updates rather than scattering across subsystems.

## Recommended Wave 3 launch shape

8–12 parallel agents:
1. **Potion effects** — sonnet
2. **Scroll effects** — sonnet
3. **Wand zap + rays** — sonnet
4. **Ring + amulet intrinsics** — sonnet
5. **Spell cast + spellbook learn** — sonnet
6. **Combat formulas** (to-hit, damage, AC, skills) — opus (complex)
7. **Monster AI** — opus (complex)
8. **Inventory ops** — sonnet
9. **Status effect ticks** — sonnet
10. **Trap damage + features doors** — sonnet
11. **Observation projection (inv, colors, tty_colors)** — sonnet
12. **Role / race / starting inventory** — sonnet

Plus integration: object table canonicalize, run tests, write Wave 3 docs.

## Open questions

1. **MiniHack pull-forward — Wave 3 or Wave 4?** You preferred Wave 3 originally; this plan defers to Wave 4 to keep Wave 3 focused on game-mechanic core. Confirm.

2. **Wave 3 testing depth**: should we add property-based tests (Hypothesis) for combat formulas against vendor C source, or rely on a small set of canonical example values? Property tests add high confidence but ~1 day of agent time.

3. **`obs['inv_strs']` fidelity**: the canonical NLE format is "the blessed +2 long sword named Sting (weapon in hand)". Full reproduction needs identification appearance, BUC visibility, enchantment, naming, equip status. Should Wave 3 ship a **simplified** form ("long sword (a)" or similar) and defer perfect-NLE-fidelity rendering to Wave 6?

4. **Monster spawning policy**: NetHack spawns monsters by level depth + role + role-quest stage. For Wave 3 should we use a simplified "5–10 random monsters from the canonical table per level" or implement the depth-based spawning curve precisely?

5. **Polymorph and lycanthropy — Wave 3 or 4?** Player polymorph involves swapping the entire attack-set + stat block. Easier as a Wave 4 unit after monster AI lands.

Reply with picks (or "all defaults"). My defaults are:
- ★ MiniHack pull-forward → Wave 4
- ★ No property tests yet; canonical example values + Wave 6 deepens
- ★ Simplified `inv_strs`; perfect fidelity in Wave 6
- ★ Simplified monster spawning (random N per level), depth-curve Wave 6
- ★ Polymorph → Wave 4
