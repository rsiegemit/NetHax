# NetHack 3.7 Corpse Effects Reference

## Intrinsic Conveyance System

Monster corpses can grant intrinsics and special effects when eaten.

**Conveyance mechanism**:
- `mconveys` field determines which intrinsic a monster can grant
- `should_givit()` uses formula: `mlevel > rn2(chance)` where chance is typically 15
- Higher monster level = higher probability of granting intrinsic

## Resistances by Corpse Source

### Fire Resistance
- red dragon
- fire giant
- fire ant
- fire elemental
- hell hound
- pyrolisk

### Cold Resistance
- white dragon
- frost giant
- winter wolf
- blue jelly
- ice devil

### Sleep Resistance
- orange dragon
- gelatinous cube

### Disintegration Resistance
- black dragon

### Shock Resistance
- blue dragon
- electric eel

### Poison Resistance
- green dragon
- killer bee
- scorpion
- soldier ant
- pit viper
- snake

### Acid Resistance
- yellow dragon (temporary only)
- acid blob (temporary only)

### Stone Resistance
- lizard corpse (temporary)
- yellow dragon (temporary)

### Teleportation
- leprechaun (mlevel > rn2(10))
- tengu (mlevel > rn2(10))
- quantum mechanic (mlevel > rn2(10))

### Teleport Control
- tengu (mlevel > rn2(12))

## Special Corpse Effects (cpostfx)

These effects bypass the normal conveyance system:

### Level/Stat Gain
- **wraith**: Gain +1 experience level (always)
- **giant corpses** (all types): 50% chance of +1 Strength

### Healing/Curing
- **nurse**: Full HP restore + cure blindness
- **lizard**: Cures stun + confusion + grants temporary stone resistance

### Intrinsics
- **stalker**: Permanent invisibility + see invisible (always)
- **floating eye**: Telepathy (always works, chance=1)
- **mind flayer**: 50% chance of +1 Intelligence, otherwise telepathy
- **displacer beast**: Temporary displacement

### Polymorphing
- **chameleon**: Polymorph self
- **doppelganger**: Polymorph self
- **genetic engineer**: Polymorph self

### Status Effects
- **bat**: Stun
- **giant bat**: Stun
- **yellow light**: Stun
- **quantum mechanic**: Toggle speed (fast/slow)

### Harmful Effects
- **disenchanter**: Lose a random intrinsic
- **Death**: Fatal
- **Pestilence**: Fatal
- **Famine**: Fatal
- **human werewolf**: Catch lycanthropy
- **human werejackal**: Catch lycanthropy
- **human wererat**: Catch lycanthropy

## Notes on Corpse Freshness

- Corpses rot over time (age increases each turn)
- Rotten corpses may cause food poisoning
- Tainted corpses (from certain monsters) are always risky
- Eating old corpses reduces nutrition value
- Some corpses (like lizards) should be kept fresh for emergency use

## Strategic Considerations

1. **Priority corpses to eat early**:
   - Floating eye (guaranteed telepathy)
   - Stalker (guaranteed invisibility + see invisible)
   - Wraith (level gain)

2. **Keep for emergencies**:
   - Lizard (stone resistance cure)
   - Nurse (full healing)

3. **Avoid unless desperate**:
   - Disenchanter
   - Lycanthrope corpses
   - Rider corpses (Death, Pestilence, Famine)

4. **Monster level matters**:
   - Higher level monsters have better chance of granting intrinsics
   - Seek archons, titans, master mind flayers for better odds
