# NetHack 3.7 Scrolls Reference

All scrolls in NetHack 3.7 with their properties and effects.

**Note:** Scroll labels are randomized at the start of each game. The "label" column shows the default unidentified text.

## Scroll List

| Name | Label | Probability | Cost | Effect |
|------|-------|-------------|------|--------|
| enchant armor | ZELGO MER | 63 | 80 | Increases the enchantment of worn armor by +1 |
| destroy armor | JUYED AWK YACC | 45 | 100 | Destroys a piece of worn armor (cursed: your armor, blessed: enemy's) |
| confuse monster | NR 9 | 53 | 100 | Next monster touched becomes confused |
| scare monster | XIXAXA XOXAXA XUXAXA | 35 | 100 | Creates a square that scares monsters (permanent if dropped while not cursed) |
| remove curse | PRATYAVAYAH | 65 | 80 | Removes curses from worn/wielded items |
| enchant weapon | DAIYEN FOOELS | 80 | 60 | Increases enchantment of wielded weapon by +1 |
| create monster | LEP GEX VEN ZEA | 45 | 200 | Summons one or more monsters nearby |
| taming | PRIRUTSENIE | 15 | 200 | Tames nearby monsters |
| genocide | ELBIB YLOH | 15 | 300 | Eliminates all monsters of a chosen species from the game |
| light | VERR YED HORRE | 90 | 50 | Lights up the area around you permanently |
| teleportation | VENZAR BORGAVVE | 55 | 100 | Teleports you to a random location on the level |
| gold detection | THARR | 33 | 100 | Reveals all gold on the current level |
| food detection | YUM YUM | 25 | 100 | Reveals all food on the current level |
| identify | KERNOD WEL | 180 | 20 | Identifies items in your inventory |
| magic mapping | ELAM EBOW | 45 | 100 | Reveals the entire map of the current level |
| amnesia | DUAM XNAHT | 35 | 200 | Causes you to forget parts of explored maps |
| fire | ANDOVA BEGARIN | 30 | 100 | Creates a fire explosion around you |
| earth | KIRJE | 18 | 200 | Creates earth/rock around you (can bury items or creatures) |
| punishment | VE FORBRYDERNE | 15 | 300 | Attaches a heavy iron ball to your leg |
| charging | HACKEM MUCHE | 15 | 300 | Recharges wands, rings, or tools |
| stinking cloud | VELOX NEB | 15 | 300 | Creates a toxic cloud that damages and blinds |
| blank paper | unlabeled | 28 | 60 | A blank scroll (can be written on with a magic marker) |
| mail | stamped | 0 | 0 | Special scroll (only appears via external mail system) |

## Usage Notes

- **Weight:** All scrolls weigh 5 units
- **Nutrition:** Scrolls provide 6 nutrition when eaten
- **Reading:** Most scrolls are consumed when read
- **Illiteracy:** Some roles cannot read scrolls without learning
- **Light:** You need light to read scrolls (unless blind and illiterate)
- **Ink:** Blank scrolls can be written on using a magic marker

## Probability Distribution

- **Very Common (80+):** enchant weapon (80), light (90), identify (180)
- **Common (45-65):** enchant armor (63), destroy armor (45), confuse monster (53), create monster (45), remove curse (65), teleportation (55), magic mapping (45)
- **Uncommon (25-35):** scare monster (35), gold detection (33), amnesia (35), fire (30), food detection (25), blank paper (28)
- **Rare (15-18):** taming (15), genocide (15), earth (18), punishment (15), charging (15), stinking cloud (15)
- **Special:** mail (0 - system-generated only)

## Important Scrolls

### High Value
- **identify** - Most abundant and useful for learning item properties
- **enchant weapon/armor** - Essential for improving equipment
- **remove curse** - Critical for dealing with cursed items
- **charging** - Extends the life of wands and tools

### Dangerous
- **destroy armor** - Can destroy your worn equipment if cursed
- **create monster** - Spawns enemies, potentially dangerous
- **amnesia** - Makes you forget explored areas
- **punishment** - Burdens you with a heavy ball and chain
- **genocide** - Powerful but limited uses; choose targets wisely

## Special Interactions

- **Blessed vs Cursed:**
  - Blessed scrolls often have enhanced effects (more charges, better bonuses)
  - Cursed scrolls may have reversed or harmful effects
- **Confused Reading:** Reading while confused can produce different, sometimes beneficial effects
- **Blank Scrolls:** Can be written with spells using a magic marker
