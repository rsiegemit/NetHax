# NetHack 3.7 Food and Nutrition Reference

## Food Items

Complete list of food items with their properties:

| Food Item | Probability | Eat Delay | Weight | Nutrition |
|-----------|-------------|-----------|--------|-----------|
| tripe ration | 140 | 2 | 10 | 200 |
| corpse | 0 | 1 | 0 | varies |
| egg | 85 | 1 | 1 | 80 |
| meatball | 0 | 1 | 1 | 5 |
| meat stick | 0 | 1 | 1 | 5 |
| enormous meatball | 0 | 20 | 400 | 2000 |
| kelp frond | 0 | 1 | 1 | 30 |
| eucalyptus leaf | 3 | 1 | 1 | 1 |
| apple | 15 | 1 | 2 | 50 |
| orange | 10 | 1 | 2 | 80 |
| pear | 10 | 1 | 2 | 50 |
| melon | 10 | 1 | 5 | 100 |
| banana | 10 | 1 | 2 | 80 |
| carrot | 15 | 1 | 2 | 50 |
| sprig of wolfsbane | 7 | 1 | 1 | 40 |
| clove of garlic | 7 | 1 | 1 | 40 |
| slime mold | 75 | 1 | 5 | 250 |
| lump of royal jelly | 0 | 1 | 2 | 200 |
| cream pie | 25 | 1 | 10 | 100 |
| candy bar | 13 | 1 | 2 | 100 |
| fortune cookie | 55 | 1 | 1 | 40 |
| pancake | 25 | 2 | 2 | 200 |
| lembas wafer | 20 | 2 | 5 | 800 |
| cram ration | 20 | 3 | 15 | 600 |
| food ration | 380 | 5 | 20 | 800 |
| K-ration | 0 | 1 | 10 | 400 |
| C-ration | 0 | 1 | 10 | 300 |
| tin | 75 | 0 | 10 | varies |

## Special Food Effects

- **carrot**: Cures blindness
- **sprig of wolfsbane**: Cures lycanthropy
- **clove of garlic**: Repels undead

## Hunger States and Thresholds

| State | Nutrition Range | Description |
|-------|----------------|-------------|
| Satiated | > 2000 | Overfed, may eventually vomit |
| Not Hungry | 1000-2000 | Normal, comfortable state |
| Hungry | 300-1000 | "You are beginning to feel hungry" |
| Weak | 150-300 | "You are beginning to feel weak" |
| Fainting | 0-150 | Risk of collapsing |
| Starved | ≤ 0 | Death from starvation |

## Hunger Mechanics

- **Base hunger rate**: 1 nutrition per turn
- Hunger rate increases with:
  - Regeneration
  - High Constitution
  - Exertion (fighting, spell casting)
  - Being a lycanthrope in animal form

## Notes

- **Probability**: Relative generation probability (0 = special circumstances only)
- **Eat Delay**: Turns required to eat the item
- **Weight**: Item weight units
- **Nutrition**: Nutrition points gained when eaten
- **Corpses**: Nutrition varies by monster type and freshness
- **Tins**: Nutrition depends on contents (monster type)
