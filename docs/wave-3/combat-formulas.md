# Wave 3 — Combat Formulas

Direct ports of NetHack's combat math from `vendor/nethack/src/`. Every formula cites its vendor source.

## To-hit roll

**Vendor:** `uhitm.c:365-427::find_roll_to_hit` line 376.

```
tmp = 1 + abon(STR, DEX, XL) + target_ac + skill_bonus + enchant
hit iff rnd(20) <= tmp
```

Components:
- `abon(STR, DEX, XL)` — attribute bonus, see below
- `target_ac` — defender AC; NetHack convention: lower AC is better, so `target_ac=-2` → `tmp += -2` makes hit harder
- `skill_bonus` — `−4..+1` based on weapon skill tier (UNSKILLED=-4, BASIC=-2, SKILLED=-1, EXPERT=0, MASTER=+1, GRANDMASTER=+1)
- `enchant` — wielded weapon `enchantment` (typically -5..+7)

## STR/DEX attribute bonus `abon`

**Vendor:** `weapon.c:950-988`.

```
str_bonus = 0
if str < 6:  abon -= 3
elif str < 8:  abon -= 2
elif str < 17: abon += 0
elif str == 18:  abon += 1
elif str < 100:  abon += 2          # 18/01..18/99
else: abon += 3                     # 18/** (i.e. 18/100)

dex_bonus = ...                     # similar table on DEX

abon = str_bonus + dex_bonus + (xl < 3 ? -1 : 0)
```

## Damage roll

**Vendor:** `weapon.c:215-302::dmgval`. Large-target branch at line 225, STR-bonus clamp at 300-302.

```
dice = ldam if target_is_large else sdam
damage = d(dice) + enchantment + dbon(STR)
damage = max(damage, 0)             # weapon.c:300 — never negative
```

## STR damage bonus `dbon`

**Vendor:** `weapon.c:992-1015`.

```
if str < 6:    dbon = -1
elif str < 16: dbon =  0
elif str < 18: dbon = +1
elif str == 18: dbon = +2
elif str < 100: dbon = +3
else: dbon = +4
```

## AC computation

**Vendor:** `do_wear.c:2473-2525::find_ac`.

```
ac = 10                              # base unarmored

for armor_slot in [BODY, SHIELD, HELM, GLOVES, BOOTS, CLOAK, SHIRT]:
    if worn_armor[armor_slot] >= 0:
        item = inventory.items[worn_armor[armor_slot]]
        bonus = item.ac_bonus + item.enchantment    # ARM_BONUS macro
        ac -= bonus

# Wave 3 stops here.
# (Vendor: also subtracts u.uacbase, applies role/race AC base, etc. — Wave 4)
```

Implemented as `lax.scan` over 7 armor slots in `subsystems/inventory.py::compute_ac`.

## Weapon skill tier

**Vendor:** `weapon.c:1198::skill_advance`, `include/skills.h:106::practice_needed_to_advance`.

```
practice_needed(tier) = tier * tier * 20

skill_advance(skill_id) if practice[skill_id] >= practice_needed(current_tier):
    skill[skill_id] = min(skill[skill_id] + 1, P_GRAND_MASTER)
    # practice continues from current count
```

Wave 3 simplification: each successful melee hit calls `practice_skill(state, weapon_type)` which increments `combat.weapon_practice[skill_id]` by 1 and checks against threshold.

`weapon_type` mapping to `skill_id`: Wave 3 uses `type_id % N_WEAPON_SKILLS` placeholder. Wave 4 wires the canonical map from `weapon.c::weapon_skill_index`.

## Monster attack on player

**Vendor:** `mhitu.c::mattacku`.

```
tmp = 1 + mlev + (10 - player_ac)
hit iff rnd(20) <= tmp

damage = 0
for attack_slot in monster.attacks:    # up to 6
    if attack_slot != NO_ATTK:
        damage += d(n_dice, n_sides)
```

Implemented via `lax.scan` over the 8 attack-slot keys with mask in `combat.monster_attack_player`.

## Bump-attack

When player movement targets a tile with an alive monster, `_try_step` (in `subsystems/action_dispatch.py`) calls `combat.bump_attack(state, rng, target_pos)`. This:

1. Looks up `monster_idx` at `target_pos` via vectorized match
2. Calls `melee_attack(state, rng, monster_idx)`
3. If `monster_idx` was not found (no alive monster): no-op (uses `jax.tree_util.tree_map(jnp.where, ...)` with a sentinel to make the function shape-static)
4. If monster died: advances player onto tile

## Practice / skill advancement

After every successful melee `_practice_skill_on_hit`:

```
counter = combat.weapon_practice[skill_id] + 1
threshold = current_tier * current_tier * 20
new_tier = if counter >= threshold and current_tier < GRAND_MASTER:
              current_tier + 1
           else: current_tier
combat.weapon_skill[skill_id] = new_tier
combat.weapon_practice[skill_id] = counter
```

## Wave 3 simplifications

| Mechanism | Status |
|---|---|
| Two-weapon penalty / off-hand attack | NOT implemented (Wave 4) |
| Ranged: throw / fire (auto-quiver) | NOT implemented (Wave 4) |
| Breath weapons (dragon, hezrou) | NOT implemented (Wave 4) |
| Engulf / swallow | NOT implemented (Wave 4) |
| Passive attacks (cockatrice touch, fire elemental) | NOT implemented (Wave 4) |
| Polymorph combat (player as monster) | NOT implemented (Wave 4) |
| Riding | NOT implemented (Wave 5) |
| Critical hits (lucky/unlucky d20 extremes) | NOT implemented (Wave 6 polish) |
| Specific role bonuses (e.g., Monk martial arts) | NOT implemented (Wave 6) |
| Specific weapon-skill bonuses for skilled+ | Placeholder values; Wave 6 |

## Verification

`tests/test_combat.py` (9 tests):
- Unarmored AC = 10
- Wearing leather armor (ac_bonus=2) → AC = 8
- High-STR Fighter vs AC=10 → ≥75% hit rate over 100 rolls (sanity check on STR/DEX bonuses)
- Dagger sdam=(1,4) damage roll averages ~2.5 over 100 rolls
- Bump-attack on adjacent monster reduces its HP
- Skill practice counter advances on hit
- Skill tier advances when counter reaches threshold
