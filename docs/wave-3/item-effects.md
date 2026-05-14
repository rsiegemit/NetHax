# Wave 3 — Item Effects

161 distinct item effects landed this wave across potions, scrolls, wands, rings, amulets, and spells. Most are real JAX implementations; some are simplified (documented below).

## Potions (26) — `subsystems/items_potions.py`

Canonical source: `vendor/nethack/src/potion.c::peffects`.

| Potion | Wave 3 behavior |
|---|---|
| HEALING | +d8 HP, +1 HP_MAX if blessed |
| EXTRA_HEALING | +d4×4 HP, +2 HP_MAX if blessed, clears hallucination |
| FULL_HEALING | HP→HP_MAX, +4 HP_MAX if blessed |
| GAIN_ENERGY | +d4×3 Pw, blessed extra |
| GAIN_LEVEL | +1 player_xl (Wave 3: simplified XP-curve bypass) |
| GAIN_ABILITY | +1 random ability score |
| RESTORE_ABILITY | clears attribute-loss flags |
| SEE_INVISIBLE | grants Intrinsic.SEE_INVIS |
| INVISIBILITY | timed invisibility status |
| MONSTER_DETECTION | reveals monster glyphs (Wave 3: no-op until obs builder has overlay) |
| OBJECT_DETECTION | reveals item glyphs (same caveat) |
| LEVITATION | grants Intrinsic.LEVITATION |
| SPEED | timed FAST status |
| PARALYSIS | timed PARALYZED |
| SLEEPING | timed SLEEP |
| CONFUSION | timed CONFUSION |
| HALLUCINATION | timed HALLUCINATION |
| BLINDNESS | timed BLIND |
| SICKNESS | sets sick_kind |
| ACID | d6 acid damage |
| OIL | item modifier (Wave 4: applies to weapons/lamps) |
| POLYMORPH | self-poly (Wave 4: wires polymorph subsystem) |
| WATER | holy water cures sick; cursed → adverse |
| BOOZE | timed CONFUSION + nutrition |
| FRUIT_JUICE | +50 nutrition |
| ENLIGHTENMENT | reveals intrinsics (Wave 4: produces message) |

Dispatch: `jax.lax.switch` over PotionEffect index with operand tuple `(state, rng, buc)`.

## Scrolls (23) — `subsystems/items_scrolls.py`

Canonical source: `vendor/nethack/src/read.c::seffects`.

| Scroll | Wave 3 behavior |
|---|---|
| IDENTIFY | scans inventory for first unidentified item, flips `identified` |
| LIGHT | floods current room with explored=True |
| ENCHANT_WEAPON | wielded weapon enchantment += 1..3 |
| ENCHANT_ARMOR | first worn armor enchantment += 1..3 |
| DESTROY_ARMOR | randomly destroys worn armor slot |
| REMOVE_CURSE | scans inventory, BUC: CURSED→UNCURSED |
| SCARE_MONSTER | nearby monsters set to FLEE strategy |
| TELEPORTATION | player_pos → random valid tile |
| MAGIC_MAPPING | `state.explored` all True on current level |
| GOLD_DETECTION | reveals gold piles (Wave 4 overlay) |
| FOOD_DETECTION | reveals food piles (Wave 4 overlay) |
| CONFUSE_MONSTER | next-attack confuses target (Wave 3: stub flag) |
| CREATE_MONSTER | spawn adjacent (Wave 3: no-op; needs monster spawn helper) |
| TAMING | makes adjacent monsters peaceful (Wave 3: stub) |
| GENOCIDE | wipes all monsters of one type (Wave 3: no-op for menu; full impl needs input) |
| AMNESIA | clears `state.explored`, forgets spells (`spell_known` cleared) |
| FIRE | d6 fire damage in radius |
| EARTH | spawns boulders adjacent (Wave 3: no-op; needs boulder physics) |
| PUNISHMENT | attaches iron ball+chain (Wave 3: timed status only) |
| CHARGING | wielded wand charges += d8 |
| STINKING_CLOUD | creates POISON_GAS region (Wave 3: timed area effect on player only) |
| MAIL | reveals "mail" message (joke scroll, no effect) |
| BLANK_PAPER | does nothing (intentional) |

## Wands (28) — `subsystems/items_wands.py`

Canonical source: `vendor/nethack/src/zap.c`.

| Wand | Class | Wave 3 behavior |
|---|---|---|
| LIGHT | SELF | floods explored |
| NOTHING | NODIR | no-op |
| SECRET_DOOR_DETECTION | SELF | full map explored |
| OPENING | NODIR | all CLOSED→OPEN doors on level |
| LOCKING | NODIR | all OPEN→CLOSED |
| PROBING | NODIR | stub (revelation only) |
| MAGIC_MISSILE | RAY | 1d6 per monster, range 8 |
| STRIKING | BEAM | 2d12 first monster |
| SLOW_MONSTER | RAY | sets slow flag (stub; Wave 4 speed array) |
| SPEED_MONSTER | RAY | sets fast flag |
| CANCELLATION | BEAM | stub (Wave 4 needs MR cancellation) |
| POLYMORPH | BEAM | monster.type_id → random eligible |
| TELEPORTATION | BEAM | monster.pos → random valid |
| DEATH | RAY | HP→0 (skips undead) |
| SLEEP | RAY | sets asleep |
| COLD | RAY | d6 cold; WATER tile → FLOOR (freeze) |
| FIRE | RAY | d6 fire |
| LIGHTNING | RAY | d6 elec |
| DIGGING | AT_LOCATION | WALL/VOID → CORRIDOR along ray |
| ENLIGHTENMENT | SELF | full map explored |
| CREATE_MONSTER | SELF | spawns adjacent monster |
| WISHING | SELF | recharges all wands to 15 (Wave 3 placeholder) |
| STASIS | BEAM | asleep flag |
| MAKE_INVISIBLE | BEAM | invisible flag |
| UNDEAD_TURNING | RAY | d8 to undead only |
| DRAINING | BEAM | 1d4 + drain XL stub |
| ACID | BEAM | d6 acid |
| POISON_GAS | RAY | d6 poison |

Ray walk: `jax.lax.scan` over 8 steps. Beam stops at first monster. Charges decrement before effect dispatch.

## Rings (28) — `subsystems/items_jewelry.py`

Canonical source: `vendor/nethack/src/do_wear.c::doputon` + `worn.c::setworn`.

| Ring | Wear effect |
|---|---|
| ADORNMENT | +1 player_cha (reverted on take-off) |
| GAIN_STRENGTH | +1 player_str |
| GAIN_CONSTITUTION | +1 player_con |
| INCREASE_ACCURACY | combat to-hit bonus (stub field; Wave 4 wires into to_hit) |
| INCREASE_DAMAGE | combat damage bonus (same caveat) |
| PROTECTION | -1 AC (stub; Wave 4 wires into AC calc) |
| REGENERATION | Intrinsic.REGEN |
| SEARCHING | Intrinsic.SEARCHING |
| STEALTH | Intrinsic.STEALTH |
| SUSTAIN_ABILITY | Intrinsic.FIXED_ABIL |
| LEVITATION | Intrinsic.LEVITATION |
| HUNGER | TimedStatus.HUNGER_RING sentinel (doubles hunger rate) |
| AGGRAVATE_MONSTER | Intrinsic.AGGRAVATE |
| CONFLICT | Intrinsic.CONFLICT |
| WARNING | Intrinsic.WARNING |
| POISON_RESISTANCE | Intrinsic.RESIST_POISON |
| FIRE_RESISTANCE | Intrinsic.RESIST_FIRE |
| COLD_RESISTANCE | Intrinsic.RESIST_COLD |
| SHOCK_RESISTANCE | Intrinsic.RESIST_SHOCK |
| FREE_ACTION | Intrinsic.FREE_ACTION |
| SLOW_DIGESTION | (hunger_tick halves rate) |
| TELEPORTATION | Intrinsic.TELEPORT |
| TELEPORT_CONTROL | Intrinsic.TELEPORT_CONTROL |
| POLYMORPH | Intrinsic.POLYMORPH |
| POLYMORPH_CONTROL | Intrinsic.POLYMORPH_CONTROL |
| INVISIBILITY | Intrinsic.INVIS |
| SEE_INVISIBLE | Intrinsic.SEE_INVIS |
| PROTECTION_FROM_SHAPE_CHANGERS | Intrinsic.PROT_FROM_SHAPE_CHANGERS |

Take-off reverses every effect symmetrically. Two rings supported.

## Amulets (13) — `subsystems/items_jewelry.py`

| Amulet | Wear effect |
|---|---|
| ESP | Intrinsic.TELEPATHY |
| LIFE_SAVING | LIFESAVED flag (Wave 4 triggers on death) |
| STRANGULATION | TimedStatus.STRANGLED = 6 → death cycle |
| RESTFUL_SLEEP | TimedStatus.SLEEPY = 50 → periodic random sleep |
| VERSUS_POISON | Intrinsic.RESIST_POISON |
| CHANGE | polymorph on wear (Wave 4 wires polymorph) |
| UNCHANGING | Intrinsic.UNCHANGING (resist poly) |
| REFLECTION | Intrinsic.REFLECTING |
| MAGICAL_BREATHING | Intrinsic.BREATHLESS |
| GUARDING | Intrinsic.PROTECTION |
| FLYING | Intrinsic.FLYING |
| CHEAP_IMITATION | no effect (joke amulet) |
| YENDOR | flag for ascension (Wave 6 ending) |

## Spells (43) — `subsystems/magic.py`

Spell cast dispatches via `_EFFECT_DISPATCH: dict[SpellId, fn]`. Each fn takes `(state_adapter, rng)` and returns the changes dict. Effects share dispatch with wands where applicable (e.g., MAGIC_MISSILE, FIRE_BOLT, CONE_OF_COLD, DEATH).

All 43 spells from `vendor/nethack/include/objects.h::SPELL()` have effect handlers. Highlights:

- HEALING / EXTRA_HEALING / CURE_BLINDNESS / CURE_SICKNESS — heal/cure
- MAGIC_MISSILE / FIRE_BOLT / FORCE_BOLT / CONE_OF_COLD / DRAIN_LIFE / FINGER_OF_DEATH — attack (uses ray cast helpers)
- DETECT_MONSTERS / DETECT_FOOD / DETECT_TREASURE / IDENTIFY / MAGIC_MAPPING / CLAIRVOYANCE — info
- CHARM_MONSTER / SLEEP / CONFUSE_MONSTER / CAUSE_FEAR / SLOW_MONSTER — status apply
- PROTECTION / REMOVE_CURSE / RESTORE_ABILITY — buff
- LEVITATION / HASTE_SELF / INVISIBILITY / JUMPING — self-buff
- TELEPORT_AWAY / POLYMORPH / CREATE_FAMILIAR / SUMMON_NASTIES — transformation
- KNOCK / WIZARD_LOCK / STONE_TO_FLESH / DIG / LIGHT — utility

## Dispatch wiring TODO (Wave 4)

Each item-effect module exposes a `handle_<action>(state, rng)` function:

- `items_potions.handle_quaff`
- `items_scrolls.handle_read` (also dispatches `handle_read_spellbook` for spellbooks)
- `items_wands.handle_zap`
- `items_jewelry.handle_put_on` / `handle_remove`
- `magic.handle_cast`
- `inventory.handle_pickup` / `handle_drop` / `handle_wield` / `handle_wear`
- `status_effects.handle_eat`

These are NOT yet wired into `action_dispatch.py::_HANDLERS`. Wave 4's first integration task is to import each `handle_*` and register it at the right index in `_ACTION_TO_HANDLER_IDX` and `_HANDLERS`.

Until then: the corresponding actions (EAT, QUAFF, READ, ZAP, CAST, etc.) fall through to no-op in `dispatch_action`. The mechanics work when called directly from tests; they're just not reachable from `env.step(state, action, rng)` yet.
