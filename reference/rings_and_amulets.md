# NetHack 3.7 Rings and Amulets Reference

## Rings

All rings in NetHack 3.7 with their properties. Ring appearances are randomized at the start of each game.

**Universal Properties:**
- Weight: 3 units
- Nutrition: 15 (when eaten)
- Two ring slots: left and right hands
- Some rings are chargeable (can be blessed/cursed to modify effect strength)

### Ring List

| Name | Appearance | Property | Cost | Enchantable | Effect |
|------|------------|----------|------|-------------|--------|
| adornment | wooden | Charisma modifier | 100 | Yes | Increases/decreases Charisma based on enchantment |
| gain strength | granite | Strength boost | 150 | Yes | Increases/decreases Strength based on enchantment |
| gain constitution | opal | Constitution boost | 150 | Yes | Increases/decreases Constitution based on enchantment |
| increase accuracy | clay | To-hit bonus | 150 | Yes | Improves/worsens attack accuracy based on enchantment |
| increase damage | coral | Damage bonus | 150 | Yes | Increases/decreases melee damage based on enchantment |
| protection | black onyx | AC and MC boost | 100 | Yes | Improves AC and grants +1 magic cancellation |
| regeneration | moonstone | HP regeneration | 200 | No | Regenerates HP faster (increases hunger rate) |
| searching | tiger eye | Find secrets | 200 | No | Automatic searching for traps and secret doors |
| stealth | jade | Stealth | 100 | No | Monsters less likely to notice you |
| sustain ability | bronze | Ability protection | 100 | No | Prevents ability score drain |
| levitation | agate | Float | 200 | No | Permanent levitation (float over ground/water/traps) |
| hunger | topaz | Increased hunger | 100 | No | **Cursed** - Rapidly increases hunger rate |
| aggravate monster | sapphire | Wake monsters | 150 | No | **Cursed** - All monsters aware of your location |
| conflict | ruby | Monster infighting | 300 | No | Causes monsters to fight each other |
| warning | diamond | Danger sense | 100 | No | Warns of nearby dangerous monsters |
| poison resistance | pearl | Poison immunity | 150 | No | Grants resistance to poison damage |
| fire resistance | iron | Fire immunity | 200 | No | Grants resistance to fire damage |
| cold resistance | brass | Cold immunity | 150 | No | Grants resistance to cold damage |
| shock resistance | copper | Electric immunity | 150 | No | Grants resistance to electricity damage |
| free action | twisted | Paralysis immunity | 200 | No | Prevents paralysis and being stuck |
| slow digestion | steel | Reduced hunger | 200 | No | Greatly reduces hunger rate (very useful!) |
| teleportation | silver | Random teleport | 200 | No | **Cursed** - Randomly teleports you periodically |
| teleport control | gold | Controlled teleport | 300 | No | Allows you to choose teleport destination |
| polymorph | ivory | Form change | 300 | No | **Cursed** - Randomly polymorphs you periodically |
| polymorph control | emerald | Controlled form | 300 | No | Allows you to choose polymorph target |
| invisibility | wire | Invisibility | 150 | No | Grants permanent invisibility |
| see invisible | engagement | Detect invisible | 150 | No | Allows you to see invisible creatures |
| protection from shape changers | shiny | Anti-polymorph | 100 | No | Prevents monsters from shapeshifting near you |

### Ring Categories

#### Enchantable Rings (Charged)
These rings have +/- enchantment levels that modify their effects:
- adornment (Cha modifier)
- gain strength (Str modifier)
- gain constitution (Con modifier)
- increase accuracy (to-hit bonus)
- increase damage (damage bonus)
- protection (AC/MC bonus)

**Note:** Blessed rings have positive enchantments, cursed rings have negative ones.

#### Always-On Rings
These provide constant benefits when worn:
- regeneration (faster healing)
- searching (auto-search)
- stealth (quieter movement)
- sustain ability (ability protection)
- All resistance rings (poison, fire, cold, shock)
- free action (paralysis immunity)
- slow digestion (less hunger)
- see invisible (detect invisible)
- warning (danger sense)
- protection from shape changers

#### Valuable Rings
Most useful rings for general play:
1. **slow digestion** - Extends food supply significantly
2. **free action** - Prevents deadly paralysis
3. **teleport control** - Incredible mobility when paired with teleportation
4. **conflict** - Makes monsters fight each other
5. **regeneration** - Faster healing (with adequate food)
6. **fire/cold resistance** - Essential for certain areas and enemies

#### Cursed/Dangerous Rings
These rings are harmful when worn:
- **hunger** - Rapidly drains nutrition
- **aggravate monster** - All monsters know your location
- **teleportation** - Random involuntary teleportation
- **polymorph** - Random involuntary form changes

---

## Amulets

All amulets in NetHack 3.7. Amulet descriptions are randomized at the start of each game.

**Universal Properties:**
- Weight: 20 units
- Cost: 150 gold (except Amulet of Yendor)
- Nutrition: 20 (when eaten)
- One amulet slot: worn around neck
- Cannot be enchanted (no +/- values)

### Amulet List

| Name | Appearance | Property | Probability | Effect |
|------|------------|----------|-------------|--------|
| amulet of ESP | circular | Telepathy | 120 | Sense the presence and location of nearby creatures |
| amulet of life saving | spherical | Auto-resurrect | 75 | Automatically revives you once when you die (consumed on use) |
| amulet of strangulation | oval | Choking | 115 | **Cursed** - Slowly strangles you to death |
| amulet of restful sleep | triangular | Narcolepsy | 115 | **Cursed** - Periodically puts you to sleep |
| amulet versus poison | pyramidal | Poison immunity | 115 | Grants resistance to poison damage |
| amulet of change | square | Polymorph once | 115 | Polymorphs you once when worn, then becomes normal amulet |
| amulet of unchanging | concave | Anti-polymorph | 60 | Prevents all polymorph (useful against polymorph traps) |
| amulet of reflection | hexagonal | Reflect beams | 75 | Reflects ray attacks and some projectiles |
| amulet of magical breathing | octagonal | Breathe anywhere | 75 | Allows breathing underwater and in other environments |
| amulet of guarding | perforated | +2 AC, +2 MC | 75 | Improves armor class by 2 and magic cancellation by 2 |
| amulet of flying | cubical | Flight | 60 | Grants flying ability (better than levitation) |
| Amulet of Yendor | Amulet of Yendor | Quest goal | unique | **THE** goal of NetHack - recover this and ascend! |

### Special Notes

#### Most Valuable Amulets
1. **Amulet of Yendor** - The ultimate quest objective (appears identified)
2. **life saving** - Get a second chance when killed (one-time use)
3. **reflection** - Essential defense against ray attacks
4. **ESP** - Incredible tactical advantage from sensing all creatures

#### Dangerous Amulets
- **strangulation** - Will kill you in ~5 turns if cursed and worn
- **restful sleep** - Randomly puts you to sleep (very dangerous in combat)

Both of these will auto-curse when worn, making them difficult to remove without remove curse.

#### Amulet of Yendor
- **Cost:** 30,000 gold (not 150 like other amulets)
- **Appearance:** Always "Amulet of Yendor" (never randomized)
- **Purpose:** Main quest goal - must be recovered and brought to the surface
- **Fake Amulets:** "cheap plastic imitation of the Amulet of Yendor" exists as a decoy

### Amulet Probability Distribution

- **Most Common:** ESP (120)
- **Common:** strangulation (115), restful sleep (115), versus poison (115), change (115)
- **Uncommon:** life saving (75), reflection (75), magical breathing (75), guarding (75)
- **Rare:** unchanging (60), flying (60)
- **Unique:** Amulet of Yendor (quest-specific)

### Strategic Amulet Usage

#### Early Game
- **ESP** - Game-changing awareness of all creatures
- **versus poison** - Protection from poison attacks
- **magical breathing** - Useful for water exploration

#### Mid Game
- **reflection** - Critical defense against dangerous rays
- **life saving** - Insurance policy against sudden death
- **guarding** - Solid defensive boost (+2 AC, +2 MC)

#### Late Game
- **reflection** - Essential for surviving high-level threats
- **life saving** - Extra lives for the most dangerous encounters
- **ESP** - Still valuable for tactical positioning
- **Amulet of Yendor** - Required for ascending and winning

#### Special Situations
- **unchanging** - Prevents polymorph (useful in specific situations)
- **flying** - Superior mobility (better than levitation)
- **magical breathing** - Water levels and certain environments

### Identification Priority
High priority to identify:
1. **strangulation/restful sleep** - Avoid these at all costs!
2. **life saving** - Know when you have emergency resurrection
3. **reflection** - Critical defensive item
4. **ESP** - Game-changing tactical advantage

Can wait to identify:
- versus poison (nice but not critical early)
- change (one-time effect, not urgent)
- guarding (useful but not game-changing)
