# NetHack 3.7 Weapons Reference

Complete weapon data extracted from `include/objects.h`.

**Strike Types:**
- P = Pierce
- S = Slash
- B = Whack (Bludgeon)

**Parameters Explained:**
- **Weight (wt):** Item weight in aum (NetHack units)
- **Cost:** Base price in zorkmids
- **Sdam:** Small damage die (vs small/medium monsters)
- **Ldam:** Large damage die (vs large monsters)
- **Hit:** To-hit bonus
- **Prob:** Probability of generation (higher = more common)
- **2H:** Two-handed weapon (1) or one-handed (0)

---

## Projectiles

Missiles that require a launcher (bow, sling, crossbow).

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|
| arrow | - | 1 | 2 | 6 | 6 | 0 | P | bow | iron | 55 |
| elven arrow | runed arrow | 1 | 2 | 7 | 6 | 0 | P | bow | wood | 20 |
| orcish arrow | crude arrow | 1 | 2 | 5 | 6 | 0 | P | bow | iron | 20 |
| silver arrow | - | 1 | 5 | 6 | 6 | 0 | P | bow | silver | 12 |
| ya | bamboo arrow | 1 | 4 | 7 | 7 | +1 | P | bow | metal | 15 |
| crossbow bolt | - | 1 | 2 | 4 | 6 | 0 | P | crossbow | iron | 55 |

---

## Thrown Weapons

Missiles that don't use a launcher.

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| dart | - | 1 | 2 | 3 | 2 | 0 | P | dart | iron | 60 | 0 | Stackable |
| shuriken | throwing star | 1 | 5 | 8 | 6 | +2 | P | shuriken | iron | 35 | 0 | Stackable |
| boomerang | - | 5 | 20 | 9 | 9 | 0 | - | boomerang | wood | 15 | 0 | Stackable, returns |

---

## Spears

Can be thrown or used in melee.

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| spear | - | 30 | 3 | 6 | 8 | 0 | P | spear | iron | 50 | 0 | Stackable |
| elven spear | runed spear | 30 | 3 | 7 | 8 | 0 | P | spear | wood | 10 | 0 | Stackable |
| orcish spear | crude spear | 30 | 3 | 5 | 8 | 0 | P | spear | iron | 13 | 0 | Stackable |
| dwarvish spear | stout spear | 35 | 3 | 8 | 8 | 0 | P | spear | iron | 12 | 0 | Stackable |
| silver spear | - | 36 | 40 | 6 | 8 | 0 | P | spear | silver | 2 | 0 | Stackable |
| javelin | throwing spear | 20 | 3 | 6 | 6 | 0 | P | spear | iron | 10 | 0 | Stackable |
| trident | - | 25 | 5 | 6 | 4 | 0 | P | trident | iron | 8 | 0 | **+1 sdam, +2d4 ldam** |

---

## Daggers and Knives

Short bladed weapons, all stackable.

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| dagger | - | 10 | 4 | 4 | 3 | +2 | P | dagger | iron | 30 | 0 | Stackable |
| elven dagger | runed dagger | 10 | 4 | 5 | 3 | +2 | P | dagger | wood | 10 | 0 | Stackable |
| orcish dagger | crude dagger | 10 | 4 | 3 | 3 | +2 | P | dagger | iron | 12 | 0 | Stackable |
| silver dagger | - | 12 | 40 | 4 | 3 | +2 | P | dagger | silver | 3 | 0 | Stackable |
| athame | - | 10 | 4 | 4 | 3 | +2 | S | dagger | iron | 0 | 0 | Stackable, ritual blade |
| scalpel | - | 5 | 6 | 3 | 3 | +2 | S | knife | metal | 0 | 0 | Stackable |
| knife | - | 5 | 4 | 3 | 2 | 0 | P\|S | knife | iron | 20 | 0 | Stackable |
| stiletto | - | 5 | 4 | 3 | 2 | 0 | P\|S | knife | iron | 5 | 0 | Stackable |
| worm tooth | - | 20 | 2 | 2 | 2 | 0 | - | knife | bone | 0 | 0 | Stackable, drops from worms |
| crysknife | - | 20 | 100 | 10 | 10 | +3 | P | knife | bone | 0 | 0 | Stackable, enchanted worm teeth |

---

## Axes

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| axe | - | 60 | 8 | 6 | 4 | 0 | S | axe | iron | 40 | 0 | |
| battle-axe | double-headed axe | 120 | 40 | 8 | 6 | 0 | S | axe | iron | 10 | **1** | Two-handed |

---

## Swords

### Short Swords

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| short sword | - | 30 | 10 | 6 | 8 | 0 | P | short sword | iron | 8 | 0 | |
| elven short sword | runed short sword | 30 | 10 | 8 | 8 | 0 | P | short sword | wood | 2 | 0 | |
| orcish short sword | crude short sword | 30 | 10 | 5 | 8 | 0 | P | short sword | iron | 3 | 0 | |
| dwarvish short sword | broad short sword | 30 | 10 | 7 | 8 | 0 | P | short sword | iron | 2 | 0 | |

### Sabers

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| scimitar | curved sword | 40 | 15 | 8 | 8 | 0 | S | saber | iron | 15 | 0 | |
| silver saber | - | 40 | 75 | 8 | 8 | 0 | S | saber | silver | 6 | 0 | |

### Broadswords

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| broadsword | - | 70 | 10 | 4 | 6 | 0 | S | broad sword | iron | 8 | 0 | **+d4 sdam, +1 ldam** |
| elven broadsword | runed broadsword | 70 | 10 | 6 | 6 | 0 | S | broad sword | wood | 4 | 0 | **+d4 sdam, +1 ldam** |

### Long Swords

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| long sword | - | 40 | 15 | 8 | 12 | 0 | S | long sword | iron | 50 | 0 | |
| katana | samurai sword | 40 | 80 | 10 | 12 | +1 | S | long sword | iron | 4 | 0 | |

### Two-Handed Swords

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| two-handed sword | - | 150 | 50 | 12 | 6 | 0 | S | two-handed sword | iron | 22 | **1** | **+2d6 ldam** |
| tsurugi | long samurai sword | 60 | 500 | 16 | 8 | +2 | S | two-handed sword | metal | 0 | **1** | **+2d6 ldam**, artifact base |

### Special Swords

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| runesword | runed broadsword | 40 | 300 | 4 | 6 | 0 | S | broad sword | iron | 0 | 0 | **+d4 sdam, +1 ldam**; Stormbringer: +5d2 +d8 from level drain |

---

## Polearms

All polearms are two-handed and use the polearms skill.

### Spear-Type Polearms

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| partisan | vulgar polearm | 80 | 10 | 6 | 6 | 0 | P | polearms | iron | 5 | **1** | **+1 ldam** |
| ranseur | hilted polearm | 50 | 6 | 4 | 4 | 0 | P | polearms | iron | 5 | **1** | **+d4 sdam, +d4 ldam** |
| spetum | forked polearm | 50 | 5 | 6 | 6 | 0 | P | polearms | iron | 5 | **1** | **+1 sdam, +d6 ldam** |
| glaive | single-edged polearm | 75 | 6 | 6 | 10 | 0 | S | polearms | iron | 8 | **1** | |

### Axe-Type Polearms

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| halberd | angled poleaxe | 150 | 10 | 10 | 6 | 0 | P\|S | polearms | iron | 8 | **1** | **+1d6 ldam** |
| bardiche | long poleaxe | 120 | 7 | 4 | 4 | 0 | S | polearms | iron | 4 | **1** | **+1d4 sdam, +2d4 ldam** |
| voulge | pole cleaver | 125 | 5 | 4 | 4 | 0 | S | polearms | iron | 4 | **1** | **+d4 sdam, +d4 ldam** |

### Curved/Hooked Polearms

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| fauchard | pole sickle | 60 | 5 | 6 | 8 | 0 | P\|S | polearms | iron | 6 | **1** | |
| guisarme | pruning hook | 80 | 5 | 4 | 8 | 0 | S | polearms | iron | 6 | **1** | **+1d4 sdam** |
| bill-guisarme | hooked polearm | 120 | 7 | 4 | 10 | 0 | P\|S | polearms | iron | 4 | **1** | **+1d4 sdam** |

### Other Polearms

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| lucern hammer | pronged polearm | 150 | 7 | 4 | 6 | 0 | B\|P | polearms | iron | 5 | **1** | **+1d4 sdam** |
| bec de corbin | beaked polearm | 100 | 8 | 8 | 6 | 0 | B\|P | polearms | iron | 4 | **1** | |

---

## Other Melee Weapons

### Pick/Lance

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| dwarvish mattock | broad pick | 120 | 50 | 12 | 8 | -1 | B | pick-axe | iron | 13 | **1** | Digging tool |
| lance | - | 180 | 10 | 6 | 8 | 0 | P | lance | iron | 4 | 0 | **+2d10 jousting (primary), +2d2 jousting (secondary)** |

### Bludgeons

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| mace | - | 30 | 5 | 6 | 6 | 0 | B | mace | iron | 40 | 0 | **+1 sdam** |
| silver mace | - | 36 | 60 | 6 | 6 | 0 | B | mace | silver | 2 | 0 | **+1 sdam** |
| morning star | - | 120 | 10 | 4 | 6 | 0 | B | morning star | iron | 12 | 0 | **+d4 sdam, +1 ldam** |
| war hammer | - | 50 | 5 | 4 | 4 | 0 | B | hammer | iron | 15 | 0 | **+1 sdam** |
| club | - | 30 | 3 | 6 | 3 | 0 | B | club | wood | 12 | 0 | |
| rubber hose | - | 20 | 3 | 4 | 3 | 0 | B | whip | plastic | 0 | 0 | Tourist starting weapon |

### Staves and Flails

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| quarterstaff | staff | 40 | 5 | 6 | 6 | 0 | B | quarterstaff | wood | 11 | **1** | |
| aklys | thonged club | 15 | 4 | 6 | 3 | 0 | B | club | iron | 8 | 0 | Returns when thrown |
| flail | - | 15 | 4 | 6 | 4 | 0 | B | flail | iron | 40 | 0 | **+1 sdam, +1d4 ldam** |

### Whips

| Name | Appearance | Weight | Cost | Sdam | Ldam | Hit | Type | Skill | Material | Prob | 2H | Notes |
|------|------------|--------|------|------|------|-----|------|-------|----------|------|----|-------|
| bullwhip | - | 20 | 4 | 2 | 1 | 0 | - | whip | leather | 2 | 0 | Can't be poisoned |

---

## Launchers

Weapons that fire projectiles.

| Name | Appearance | Weight | Cost | Hit | Skill | Material | Prob | 2H | Ammo |
|------|------------|--------|------|-----|-------|----------|------|----|------|
| bow | - | 30 | 60 | 0 | bow | wood | 24 | 0 | arrows |
| elven bow | runed bow | 30 | 60 | 0 | bow | wood | 12 | 0 | arrows |
| orcish bow | crude bow | 30 | 60 | 0 | bow | wood | 12 | 0 | arrows |
| yumi | long bow | 30 | 60 | 0 | bow | wood | 0 | 0 | arrows (especially ya) |
| sling | - | 3 | 20 | 0 | sling | leather | 40 | 0 | rocks, gems |
| crossbow | - | 50 | 40 | 0 | crossbow | wood | 45 | 0 | crossbow bolts |

---

## Notes

### Extra Damage Formulas

Many weapons deal bonus damage beyond their base die:

- **Broadswords** (broadsword, elven broadsword, runesword): +d4 small damage, +1 large damage
- **Two-handed swords** (two-handed sword, tsurugi): +2d6 large damage
- **Trident**: +1 small damage, +2d4 large damage
- **Maces** (mace, silver mace): +1 small damage
- **War hammer**: +1 small damage
- **Morning star**: +d4 small damage, +1 large damage
- **Flail**: +1 small damage, +1d4 large damage
- **Lance**: +2d10 when jousting (primary weapon), +2d2 when jousting (secondary in dual-wield)
- **Polearms**: See individual entries for specific bonuses

### Special Properties

- **Silver weapons**: Deal extra damage to silver-hating monsters (werewolves, demons, etc.)
- **Crysknife**: Created by enchanting a stack of worm teeth; entire stack reverts to teeth when dropped
- **Runesword**: Base for Stormbringer artifact (+5d2 +d8 from level drain)
- **Tsurugi**: Base for two-handed sword artifacts
- **Athame**: Ritual blade used by wizards for spell casting
- **Lance**: Special jousting mechanics when mounted
- **Aklys**: Returns to wielder when thrown
- **Boomerang**: Returns when thrown (if it misses)
- **Bullwhip**: Cannot be poisoned; used for disarming

### Skill Categories

- **Bow**: arrows (except crossbow bolts)
- **Crossbow**: crossbow bolts only
- **Sling**: rocks, gems, heavy projectiles
- **Thrown**: dart, shuriken, boomerang
- **Spear**: all spears, javelins
- **Trident**: trident only
- **Dagger**: dagger family
- **Knife**: knife, scalpel, worm tooth, crysknife
- **Axe**: axe, battle-axe
- **Sword Skills**: short sword, broad sword, long sword, two-handed sword, saber
- **Polearms**: all polearms
- **Lance**: lance only
- **Pick-axe**: dwarvish mattock
- **Mace**: mace, silver mace
- **Morning star**: morning star only
- **Hammer**: war hammer
- **Club**: club, aklys
- **Quarterstaff**: quarterstaff only
- **Flail**: flail only
- **Whip**: bullwhip, rubber hose

### Materials

Different materials provide varying effectiveness against certain monsters:
- **Iron/Metal**: Standard weapons
- **Silver**: Extra damage to lycanthropes and demons
- **Wood**: Elven weapons, lighter weight
- **Bone**: Worm tooth, crysknife (special properties)

---

**Source:** `include/objects.h` from NetHack 3.7
**Generated:** 2026-02-05
