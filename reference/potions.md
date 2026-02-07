# NetHack 3.7 Potions Reference

All potions in NetHack 3.7 with their properties and effects.

**Note:** Potion appearances (colors) are randomized at the start of each game. The "appearance" column shows the default unidentified description.

## Potion List

| Name | Appearance | Probability | Cost | Effect |
|------|------------|-------------|------|--------|
| gain ability | ruby | 40 | 300 | Increases a random ability score (Str, Dex, Con, Int, Wis, Cha) by 1 point |
| restore ability | pink | 40 | 100 | Restores ability scores that have been drained or reduced |
| confusion | orange | 40 | 100 | Causes confusion status for several turns |
| blindness | yellow | 30 | 150 | Causes temporary blindness |
| paralysis | emerald | 40 | 300 | Paralyzes the drinker for several turns (dangerous!) |
| speed | dark green | 40 | 200 | Grants temporary very fast speed |
| levitation | cyan | 40 | 200 | Grants temporary levitation (float above ground/water/traps) |
| hallucination | sky blue | 30 | 100 | Causes hallucination (monsters and items appear different) |
| invisibility | brilliant blue | 40 | 150 | Grants temporary invisibility |
| see invisible | magenta | 40 | 50 | Grants ability to see invisible creatures |
| healing | purple-red | 115 | 20 | Restores HP (heals d8 + Constitution bonus) |
| extra healing | puce | 45 | 100 | Restores more HP than healing (heals d8+d8 + Con bonus) |
| gain level | milky | 20 | 300 | Increases experience level by 1 |
| enlightenment | swirly | 20 | 200 | Reveals information about yourself and the dungeon level |
| monster detection | bubbly | 40 | 150 | Reveals all monsters on the current level |
| object detection | smoky | 40 | 150 | Reveals all objects on the current level |
| gain energy | cloudy | 40 | 150 | Increases maximum magical energy (Pw) |
| sleeping | effervescent | 40 | 100 | Puts the drinker to sleep (dangerous!) |
| full healing | black | 10 | 200 | Fully restores HP and cures various ailments |
| polymorph | golden | 10 | 200 | Polymorphs the drinker into a random creature |
| booze | brown | 40 | 50 | Causes confusion and various drunkenness effects |
| sickness | fizzy | 40 | 50 | Causes sickness (vomiting, HP loss) |
| fruit juice | dark | 40 | 50 | Nutritious drink, no special effects |
| acid | white | 10 | 250 | Damages the drinker; useful for corroding items |
| oil | murky | 30 | 250 | Can be lit on fire or used for lamp fuel |
| water | clear | 80 | 100 | Plain water; used for dilution, blessing, or cursing |

## Usage Notes

- **Weight:** All potions weigh 20 units
- **Nutrition:** Potions provide 10 nutrition when quaffed
- **Dipping:** Many potions have additional effects when items are dipped into them
- **Throwing:** Potions shatter when thrown, creating various effects in the target area
- **Dilution:** Mixing potions with water can dilute them (usually into water)
- **Holy water:** Blessed water is one of the most valuable items in the game

## Probability Distribution

The probability values represent relative spawn chances. Higher probability = more common.

- **Very Common (80+):** water (80), healing (115), identify (180 for scrolls, but healing is most common potion)
- **Common (40-50):** gain ability, restore ability, confusion, paralysis, speed, levitation, invisibility, see invisible, booze, sickness, fruit juice
- **Uncommon (30):** blindness, hallucination, oil
- **Rare (10-20):** full healing, polymorph, acid

## Special Interactions

- **Blessed vs Cursed:** The blessing status of potions affects their power and can reverse negative effects
- **Polymorph:** Can transform potions into other potion types
- **Alchemy:** Certain potion combinations produce specific results when mixed
