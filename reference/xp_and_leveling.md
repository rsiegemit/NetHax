# NetHack 3.7 Experience and Leveling Reference

## Experience Point Formula

From `exper.c: newuexp(lev)`:

```
if (lev < 1) return 0;
if (lev <= 9) return 10 * (2^lev);
if (lev <= 19) return 10000 * (2^(lev-10));
if (lev <= 30) return 10000000 * (lev - 19);
```

## Complete XP Table (Levels 1-30)

| Level | XP Required | XP Difference | Cumulative XP |
|-------|-------------|---------------|---------------|
| 1 | 0 | - | 0 |
| 2 | 20 | 20 | 20 |
| 3 | 40 | 20 | 60 |
| 4 | 80 | 40 | 140 |
| 5 | 160 | 80 | 300 |
| 6 | 320 | 160 | 620 |
| 7 | 640 | 320 | 1,260 |
| 8 | 1,280 | 640 | 2,540 |
| 9 | 2,560 | 1,280 | 5,100 |
| 10 | 5,120 | 2,560 | 10,220 |
| 11 | 10,000 | 4,880 | 20,220 |
| 12 | 20,000 | 10,000 | 40,220 |
| 13 | 40,000 | 20,000 | 80,220 |
| 14 | 80,000 | 40,000 | 160,220 |
| 15 | 160,000 | 80,000 | 320,220 |
| 16 | 320,000 | 160,000 | 640,220 |
| 17 | 640,000 | 320,000 | 1,280,220 |
| 18 | 1,280,000 | 640,000 | 2,560,220 |
| 19 | 2,560,000 | 1,280,000 | 5,120,220 |
| 20 | 5,120,000 | 2,560,000 | 10,240,220 |
| 21 | 10,000,000 | 4,879,780 | 20,240,220 |
| 22 | 20,000,000 | 10,000,000 | 40,240,220 |
| 23 | 30,000,000 | 10,000,000 | 70,240,220 |
| 24 | 40,000,000 | 10,000,000 | 110,240,220 |
| 25 | 50,000,000 | 10,000,000 | 160,240,220 |
| 26 | 60,000,000 | 10,000,000 | 220,240,220 |
| 27 | 70,000,000 | 10,000,000 | 290,240,220 |
| 28 | 80,000,000 | 10,000,000 | 370,240,220 |
| 29 | 90,000,000 | 10,000,000 | 460,240,220 |
| 30 | 100,000,000 | 10,000,000 | 560,240,220 |

## Level Progression Characteristics

### Levels 1-9: Exponential Growth (Early Game)
- Formula: `10 * 2^level`
- Doubles each level
- Rapid initial progression
- Covers dungeon exploration phase

### Levels 10-19: Steep Exponential (Mid Game)
- Formula: `10000 * 2^(level-10)`
- Much larger multiplier
- Quest and deeper dungeon access
- Preparation for Gehennom

### Levels 20-30: Linear Growth (End Game)
- Formula: `10000000 * (level - 19)`
- Constant 10M XP per level
- Gehennom and Elemental Planes
- Maximum level cap at 30

## Player Level Cap

- **MAXULEV**: 30 (maximum player level)
- Reaching level 30 requires 100,000,000 XP
- XP continues to accumulate but no further levels gained

## HP Gain Per Level

HP gain varies by role and constitution:

| Role | Base HP/Level | Con Bonus |
|------|---------------|-----------|
| Archeologist | 1d10 | +0 to +3 |
| Barbarian | 1d10 | +0 to +3 |
| Caveperson | 1d10 | +0 to +3 |
| Healer | 1d8 | +0 to +3 |
| Knight | 1d10 | +0 to +3 |
| Monk | 1d8 | +0 to +3 |
| Priest | 1d8 | +0 to +3 |
| Ranger | 1d10 | +0 to +3 |
| Rogue | 1d8 | +0 to +3 |
| Samurai | 1d10 | +0 to +3 |
| Tourist | 1d8 | +0 to +3 |
| Valkyrie | 1d10 | +0 to +3 |
| Wizard | 1d6 | +0 to +3 |

**Constitution modifiers**:
- Con 3-6: -2 HP/level
- Con 7-14: +0 HP/level
- Con 15-16: +1 HP/level
- Con 17: +2 HP/level
- Con 18-25: +3 HP/level

## Energy (Spell Power) Gain Per Level

Energy gain varies by role and wisdom/intelligence:

| Role | Base Energy/Level | Stat Bonus |
|------|-------------------|------------|
| Archeologist | 1d4 | Wisdom |
| Barbarian | 1d4 | Wisdom |
| Caveperson | 1d4 | Wisdom |
| Healer | 1d6 | Wisdom |
| Knight | 1d4 | Wisdom |
| Monk | 1d4+1 | Wisdom |
| Priest | 1d6 | Wisdom |
| Ranger | 1d4 | Wisdom |
| Rogue | 1d4 | Wisdom |
| Samurai | 1d4 | Wisdom |
| Tourist | 1d4 | Wisdom |
| Valkyrie | 1d4 | Wisdom |
| Wizard | 1d6 | Intelligence |

**Wisdom/Intelligence modifiers**:
- Similar to Constitution for HP
- Wizards use Intelligence instead of Wisdom

## Monster XP Value

Monsters grant XP when killed based on their difficulty:

**Base formula**: Experience scales with monster level, difficulty, and special abilities

**Factors**:
- Monster level (mlevel)
- Difficulty rating
- Special attacks (extra XP for dangerous abilities)
- Unique/named monsters grant bonus XP

## Game Speed Constants

- **NORMAL_SPEED**: 12 (standard movement speed)
- Speed affects action frequency (12 = 1 action per turn)
- Very fast: 24 (2 actions/turn)
- Fast: 18 (1.5 actions/turn)
- Slow: 6 (0.5 actions/turn)

## Combat Constants

- **NATTK**: 6 (maximum attacks per monster type)
- Monsters can have up to 6 different attack types
- Player typically has 1-2 attacks (more with martial arts, two-weapon combat)

## Strategic Leveling Notes

1. **Early game (1-10)**: Focus on survival, levels come quickly
2. **Mid game (10-20)**: Quest accessibility at 14, deeper dungeon exploration
3. **Late game (20+)**: Each level takes significant grinding, prioritize other goals
4. **Level 30**: Extremely rare, not required for ascension
5. **Wraith corpses**: Can accelerate leveling dramatically (gain 1 level per corpse)
6. **Genocide wraiths**: Common strategy to prevent easy leveling exploits
