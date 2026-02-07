# NetHack 3.7 Wands Reference

All wands in NetHack 3.7 with their properties and usage patterns.

**Note:** Wand appearances (materials) are randomized at the start of each game. The "appearance" column shows the default unidentified description.

## Direction Types

- **NODIR** - Self-targeted, no direction needed (11-15 charges on generation)
- **IMMEDIATE** - Affects target in chosen direction immediately (4-8 charges)
- **RAY** - Shoots a beam that can bounce and affect multiple targets (4-8 charges)

## Wands by Type

### NODIR Wands (No Direction Required)

| Name | Appearance | Probability | Cost | Charges | Effect |
|------|------------|-------------|------|---------|--------|
| light | glass | 95 | 100 | 11-15 | Lights up the area around you |
| secret door detection | balsa | 50 | 150 | 11-15 | Reveals secret doors and corridors nearby |
| enlightenment | crystal | 15 | 150 | 11-15 | Provides information about yourself and surroundings |
| create monster | maple | 45 | 200 | 11-15 | Summons random monsters |
| wishing | pine | 5 | 500 | **1** | Grants a wish (always generates with exactly 1 charge!) |

### IMMEDIATE Wands (Point and Use)

| Name | Appearance | Probability | Cost | Charges | Effect |
|------|------------|-------------|------|---------|--------|
| nothing | oak | 25 | 100 | 4-8 | Does nothing (used to identify other wands) |
| striking | ebony | 75 | 150 | 4-8 | Deals physical damage to target |
| make invisible | marble | 45 | 150 | 4-8 | Makes target invisible |
| slow monster | tin | 50 | 150 | 4-8 | Slows down target monster |
| speed monster | brass | 50 | 150 | 4-8 | Speeds up target monster |
| undead turning | copper | 50 | 150 | 4-8 | Damages undead, may flee or be destroyed |
| polymorph | silver | 45 | 200 | 4-8 | Transforms target into different creature |
| cancellation | platinum | 45 | 200 | 4-8 | Removes magical properties from target |
| teleportation | iridium | 45 | 200 | 4-8 | Teleports target to random location on level |
| opening | zinc | 25 | 150 | 4-8 | Opens doors, containers, and locks |
| locking | aluminum | 25 | 150 | 4-8 | Locks doors and containers |
| probing | uranium | 30 | 150 | 4-8 | Reveals target's stats, inventory, and condition |

### RAY Wands (Beam Effects)

| Name | Appearance | Probability | Cost | Charges | Effect |
|------|------------|-------------|------|---------|--------|
| digging | iron | 55 | 150 | 4-8 | Creates tunnels through walls, can dig pits |
| magic missile | steel | 50 | 150 | 4-8 | Fires magic projectiles that always hit |
| fire | hexagonal | 40 | 175 | 4-8 | Shoots fire beam, can burn items |
| cold | short | 40 | 175 | 4-8 | Shoots cold beam, can freeze potions |
| sleep | runed | 50 | 175 | 4-8 | Puts targets to sleep |
| death | long | 5 | 500 | 4-8 | Instant death ray (very rare, very powerful) |
| lightning | curved | 40 | 175 | 4-8 | Electric beam, can destroy rings and wands |

## Wand Properties

- **Weight:** All wands weigh 7 units
- **Nutrition:** All wands provide 30 nutrition when eaten
- **Recharging:** Wands can be recharged with scroll of charging
  - Maximum recharge limit: **7 times**
  - Recharged wands gain 1d3 charges (or more if blessed scroll used)
  - Exceeding recharge limit causes wand to explode!
- **Wresting:** When a wand has 0 charges, you may "wrest" one final charge with 1/(charges_used) chance of success

## Probability Distribution

### Very Common (75-95)
- light (95)
- striking (75)
- enchant weapon (80) [scroll, not wand]

### Common (50-55)
- secret door detection (50)
- slow monster (50)
- speed monster (50)
- undead turning (50)
- sleep (50)
- magic missile (50)
- digging (55)

### Uncommon (25-45)
- nothing (25)
- make invisible (45)
- polymorph (45)
- cancellation (45)
- teleportation (45)
- create monster (45)
- opening (25)
- locking (25)
- probing (30)
- fire (40)
- cold (40)
- lightning (40)

### Rare (5-15)
- enlightenment (15)
- wishing (5) ⭐
- death (5) ☠️

## Strategic Notes

### Most Valuable Wands
1. **wishing** - The most powerful item in the game; always has exactly 1 charge
2. **death** - Instant kill against most enemies
3. **teleportation** - Escape tool and tactical positioning
4. **digging** - Essential for accessing certain areas
5. **cancellation** - Disables dangerous monsters and protects items

### Wand Identification
- **nothing** wand is safe to use for testing
- Ray wands can bounce off walls
- IMMEDIATE wands work on adjacent targets
- NODIR wands work without selecting a direction

### Recharging Strategy
- Track recharge count carefully (maximum 7 recharges)
- Use blessed scroll of charging for more charges
- Never recharge a wand that's been recharged 7 times!
- Wishing and death wands are prime recharge candidates

### Combat Wands
Best offensive wands:
1. death (instant kill)
2. fire/cold/lightning (elemental damage)
3. magic missile (guaranteed hit)
4. striking (physical damage)
5. sleep (disable enemies)

### Utility Wands
Most useful non-combat wands:
1. digging (terrain modification)
2. teleportation (mobility)
3. opening/locking (access control)
4. cancellation (remove magic)
5. probing (information gathering)
