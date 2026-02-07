# NetHack 3.7 Monster Reference

Comprehensive reference for monster data structures, flags, and statistics for building a NetHack-like JAX RL environment.

---

## 1. Permonst Struct Layout

From `include/permonst.h` - the core monster definition structure:

```c
struct permonst {
    const char *pmnames;        /* Monster name(s) */
    char mlet;                  /* Symbol/glyph */
    schar mlevel;               /* Monster level (difficulty) */
    Bitfield(mmove, 10);        /* Base speed (12 = normal) */
    schar ac;                   /* Armor class */
    schar mr;                   /* Magic resistance % */
    aligntyp maligntyp;         /* Alignment: -128=chaotic, 0=neutral, 127=lawful */

    struct attack mattk[NATTK]; /* Attack array (up to 6 attacks) */
                                /* Each attack: { type, damage_type, ndice, ndsiz } */

    unsigned int msize;         /* Physical size (MZ_*) */
    unsigned int mresists;      /* Resistances (MR_*) */
    unsigned int mconveys;      /* Resistances conveyed by eating */

    unsigned long mflags1;      /* M1_* flags */
    unsigned long mflags2;      /* M2_* flags */
    unsigned long mflags3;      /* M3_* flags */

    unsigned int mcolor;        /* Color */
    unsigned int geno;          /* Generation flags (G_*) */

    short cnutrit;              /* Nutrition from corpse */
    short mweight;              /* Weight in zorkmids */
    short difficulty;           /* Difficulty rating for experience */
};
```

### Attack Structure
```c
struct attack {
    uchar aatyp;    /* Attack type (AT_*) */
    uchar adtyp;    /* Damage type (AD_*) */
    uchar damn;     /* Number of dice */
    uchar damd;     /* Dice size */
};
```

---

## 2. Attack Types (AT_*)

From `include/monattk.h`:

| Code | Name | Description |
|------|------|-------------|
| `AT_NONE` | No attack | Passive or non-combatant |
| `AT_CLAW` | Claw | Standard melee claw/scratch |
| `AT_BITE` | Bite | Standard melee bite |
| `AT_KICK` | Kick | Melee kick attack |
| `AT_BUTT` | Butt | Head butt or ram |
| `AT_TUCH` | Touch | Touch attack (no damage modifiers) |
| `AT_STNG` | Sting | Stinger/poisonous appendage |
| `AT_HUGS` | Hug | Crushing hug/constriction |
| `AT_SPIT` | Spit | Ranged spit attack |
| `AT_ENGL` | Engulf | Swallow/engulf whole |
| `AT_BREA` | Breath | Breath weapon |
| `AT_EXPL` | Explode | Explodes on death |
| `AT_BOOM` | Suicide | Explodes to attack (self-destructs) |
| `AT_GAZE` | Gaze | Passive gaze attack |
| `AT_TENT` | Tentacle | Tentacle attack |
| `AT_SCRE` | Scream | Sonic/scream attack |
| `AT_WEAP` | Weapon | Uses wielded weapon |
| `AT_MAGC` | Magic | Cast spell/magic attack |

---

## 3. Damage Types (AD_*)

From `include/monattk.h`:

| Code | Name | Description | Effect |
|------|------|-------------|--------|
| `AD_PHYS` | Physical | Normal physical damage | Standard HP damage |
| `AD_MAGM` | Magic missile | Pure magic damage | Ignores AC |
| `AD_FIRE` | Fire | Fire damage | Resistable, burns items |
| `AD_COLD` | Cold | Cold damage | Resistable, freezes potions |
| `AD_SLEE` | Sleep | Sleep attack | Puts player to sleep |
| `AD_DISN` | Disintegrate | Disintegration | Instant kill or massive damage |
| `AD_ELEC` | Electric | Lightning damage | Resistable, destroys rings/wands |
| `AD_DRST` | Poison (strength) | Strength poison | Drains strength |
| `AD_ACID` | Acid | Acid damage | Resistable, corrodes armor |
| `AD_BLND` | Blind | Blindness | Temporary blindness |
| `AD_STUN` | Stun | Stunning | Confusion/stun |
| `AD_SLOW` | Slow | Slowing | Reduces speed |
| `AD_PLYS` | Paralyze | Paralysis | Player cannot move |
| `AD_DRLI` | Drain life | Level drain | Permanent level loss |
| `AD_DREN` | Drain energy | Energy drain | Drains Pw (magic power) |
| `AD_LEGS` | Wound legs | Leg damage | Slows movement |
| `AD_STON` | Stone | Petrification | Instant death (turn to stone) |
| `AD_STCK` | Stick | Adhesion | Weapon sticks to monster |
| `AD_SGLD` | Steal gold | Gold theft | Steals gold |
| `AD_SITM` | Steal item | Item theft | Steals inventory item |
| `AD_SEDU` | Seduce | Seduction | Steals items via charm |
| `AD_TLPT` | Teleport | Random teleport | Teleports player |
| `AD_RUST` | Rust | Rusting | Damages ferrous items |
| `AD_CONF` | Confuse | Confusion | Player moves randomly |
| `AD_DGST` | Digest | Digestion | Damage while engulfed |
| `AD_HEAL` | Heal | Healing | Heals target |
| `AD_WRAP` | Wrap | Entangle/hold | Holds player in place |
| `AD_WERE` | Lycanthropy | Lycanthropy | Werewolf curse |
| `AD_DRDX` | Drain dexterity | Dexterity poison | Drains dexterity |
| `AD_DRCO` | Drain constitution | Constitution drain | Drains constitution |
| `AD_DRIN` | Drain intelligence | Intelligence drain | Drains intelligence |
| `AD_DISE` | Disease | Sickness | Terminal illness |
| `AD_DCAY` | Decay | Rot | Organic items rot |
| `AD_SSEX` | Succubus | Seduction | Special seduction attack |
| `AD_HALU` | Hallucination | Hallucination | Hallucination effect |
| `AD_DETH` | Death | Death magic | Instant death (save vs. death) |
| `AD_PEST` | Pestilence | Pestilence | Disease + attribute drain |
| `AD_FAMN` | Famine | Famine | Severe hunger |
| `AD_SLIM` | Slime | Sliming | Turn into green slime |
| `AD_ENCH` | Disenchant | Disenchantment | Removes enchantments |
| `AD_CORR` | Corrode | Corrosion | Corrodes armor/weapons |
| `AD_CLRC` | Clerical | Clerical spell | Random cleric spell |
| `AD_SPEL` | Magical | Wizard spell | Random wizard spell |
| `AD_RBRE` | Random breath | Random breath | Random breath weapon |
| `AD_SAMU` | Steal amulet | Amulet theft | Steals Amulet of Yendor |
| `AD_CURS` | Curse | Cursing | Curses items |

---

## 4. Monster Flags

### M1_* Flags (mflags1)

From `include/monflag.h` - Basic properties:

| Flag | Description |
|------|-------------|
| `M1_FLY` | Can fly (moves over water/lava) |
| `M1_SWIM` | Can swim (moves through water) |
| `M1_AMORPHOUS` | Amorphous (passes through bars) |
| `M1_WALLWALK` | Can phase through walls |
| `M1_CLING` | Can cling to ceiling |
| `M1_TUNNEL` | Can tunnel through rock |
| `M1_NEEDPICK` | Needs pickaxe to tunnel |
| `M1_CONCEAL` | Conceals itself (hides) |
| `M1_HIDE` | Hides under objects |
| `M1_AMPHIBIOUS` | Amphibious (water + land) |
| `M1_BREATHLESS` | Doesn't breathe (immune to drowning/choking) |
| `M1_NOTAKE` | Cannot pick up items |
| `M1_NOEYES` | Eyeless (immune to blinding) |
| `M1_NOHANDS` | No hands (can't wield/throw) |
| `M1_NOLIMBS` | No limbs at all |
| `M1_NOHEAD` | Headless (immune to beheading) |
| `M1_MINDLESS` | Mindless (immune to psychic) |
| `M1_HUMANOID` | Humanoid shape |
| `M1_ANIMAL` | Animal intelligence |
| `M1_SLITHY` | Slithy/serpentine |
| `M1_UNSOLID` | Incorporeal/unsolid |
| `M1_THICK_HIDE` | Thick hide (hard to hit) |
| `M1_OVIPAROUS` | Lays eggs |
| `M1_REGEN` | Regenerates HP |
| `M1_SEE_INVIS` | Can see invisible |
| `M1_TPORT` | Can teleport |
| `M1_TPORT_CNTRL` | Can control teleportation |
| `M1_ACID` | Acidic (damages attackers) |
| `M1_POIS` | Poisonous (to eat) |
| `M1_CARNIVORE` | Carnivorous |
| `M1_HERBIVORE` | Herbivorous |
| `M1_OMNIVORE` | Omnivorous |
| `M1_METALLIVORE` | Eats metal |

### M2_* Flags (mflags2)

Behavioral and special properties:

| Flag | Description |
|------|-------------|
| `M2_NOPOLY` | Cannot be polymorphed into |
| `M2_UNDEAD` | Undead creature |
| `M2_WERE` | Lycanthrope (werewolf type) |
| `M2_HUMAN` | Human or elf |
| `M2_ELF` | Elvish |
| `M2_DWARF` | Dwarven |
| `M2_GNOME` | Gnomish |
| `M2_ORC` | Orcish |
| `M2_DEMON` | Demonic |
| `M2_MERC` | Mercenary (can be chatted with) |
| `M2_LORD` | Demon lord or prince |
| `M2_PRINCE` | Demon prince |
| `M2_MINION` | Minion of a deity |
| `M2_GIANT` | Giant-type |
| `M2_MALE` | Always male |
| `M2_FEMALE` | Always female |
| `M2_NEUTER` | Neuter/genderless |
| `M2_PNAME` | Has proper name |
| `M2_HOSTILE` | Always hostile |
| `M2_PEACEFUL` | Can be peaceful |
| `M2_DOMESTIC` | Domestic animal |
| `M2_WANDER` | Wanders randomly |
| `M2_STALK` | Stalks player |
| `M2_NASTY` | Extra-nasty (generates w/ items) |
| `M2_STRONG` | Strong (carries more) |
| `M2_ROCKTHROW` | Throws boulders |
| `M2_GREEDY` | Collects gold |
| `M2_JEWELS` | Collects gems |
| `M2_COLLECT` | Collects miscellaneous items |
| `M2_MAGIC` | Picks up magic items |

### M3_* Flags (mflags3)

Additional special properties:

| Flag | Description |
|------|-------------|
| `M3_WANTSAMUL` | Wants Amulet of Yendor |
| `M3_WANTSBELL` | Wants Bell of Opening |
| `M3_WANTSBOOK` | Wants Book of the Dead |
| `M3_WANTSCAND` | Wants Candelabrum |
| `M3_WANTSARTI` | Wants artifacts |
| `M3_WANTSORB` | Wants Orb of Detection |
| `M3_WAITFORU` | Waits for player |
| `M3_CLOSE` | Follows closely |
| `M3_COVETOUS` | Covetous (seeks player) |
| `M3_WAITMASK` | Waiting behavior mask |
| `M3_INFRAVISION` | Has infravision |
| `M3_INFRAVISIBLE` | Visible via infravision |
| `M3_DISPLACED` | Appears displaced |
| `M3_NOTAME` | Cannot be tamed |

---

## 5. Resistance Flags (MR_*)

From `include/monflag.h`:

| Flag | Resistance |
|------|------------|
| `MR_FIRE` | Fire resistance |
| `MR_COLD` | Cold resistance |
| `MR_SLEEP` | Sleep resistance |
| `MR_DISINT` | Disintegration resistance |
| `MR_ELEC` | Shock resistance |
| `MR_POISON` | Poison resistance |
| `MR_ACID` | Acid resistance |
| `MR_STONE` | Petrification resistance |
| `MR_DRAIN` | Drain resistance |
| `MR_SICK` | Sickness resistance |

---

## 6. Monster Size (MZ_*) and Generation Flags (G_*)

### Size Classes (MZ_*)

| Size | Code | Description | Weight Range |
|------|------|-------------|--------------|
| Tiny | `MZ_TINY` | < 100 zorkmids | Insects, small rodents |
| Small | `MZ_SMALL` | 100-449 | Cats, small dogs |
| Medium | `MZ_MEDIUM` | 450-999 | Humans, wolves |
| Large | `MZ_LARGE` | 1000-2099 | Horses, large dogs |
| Huge | `MZ_HUGE` | 2100-4499 | Giants, elephants |
| Gigantic | `MZ_GIGANTIC` | ≥ 4500 | Dragons, purple worms |

### Generation Flags (G_*)

| Flag | Description |
|------|-------------|
| `G_NOGEN` | Never randomly generated |
| `G_SGROUP` | Generated in small groups (2-4) |
| `G_LGROUP` | Generated in large groups (4-6) |
| `G_GENO` | Can be genocided |
| `G_NOCORPSE` | Leaves no corpse |
| `G_HELL` | Only in Gehennom (hell) |
| `G_NOHELL` | Not in Gehennom |
| `G_UNIQ` | Unique monster (only one) |
| `G_VLGROUP` | Very large groups (6-10) |

---

## 7. Monster Symbol Map

Primary display glyphs for monster classes:

| Symbol | Class | Examples |
|--------|-------|----------|
| `a` | Giant ant | giant ant, soldier ant, fire ant, killer bee |
| `b` | Blob | acid blob, gelatinous cube, quivering blob |
| `c` | Cockatrice | cockatrice, pyrolisk, chickatrice |
| `d` | Dog/canine | jackal, fox, coyote, dog, wolf, warg |
| `e` | Eye | floating eye, gas spore |
| `f` | Feline | kitten, housecat, jaguar, tiger |
| `g` | Gremlin | gremlin, gargoyle |
| `h` | Humanoid | hobbit, dwarf, bugbear |
| `i` | Imp | tengu, homunculus |
| `j` | Jelly | blue jelly, spotted jelly, ochre jelly |
| `k` | Kobold | kobold, large kobold, kobold lord |
| `l` | Leprechaun | leprechaun |
| `m` | Mimic | small mimic, large mimic, giant mimic |
| `n` | Nymph | wood nymph, water nymph, mountain nymph |
| `o` | Orc | goblin, orc, uruk-hai, orc captain |
| `p` | Piercer | rock piercer, iron piercer, glass piercer |
| `q` | Quadruped | rothe, mumak, leocrotta, wumpus |
| `r` | Rodent | sewer rat, giant rat, rabid rat |
| `s` | Spider | cave spider, centipede, giant spider, scorpion |
| `t` | Trapper | lurker above, trapper |
| `u` | Unicorn | white unicorn, gray unicorn, black unicorn |
| `v` | Vortex | fog cloud, steam vortex, dust vortex, energy vortex |
| `w` | Worm | baby long worm, long worm, purple worm |
| `x` | Xan | grid bug, xan |
| `y` | Light | yellow light |
| `z` | Zombie | kobold zombie, gnome zombie, zombie, ghoul |
| `A` | Angelic | couatl, aleax, Angel, Archon |
| `B` | Bird | raven, vulture |
| `C` | Centaur | plains centaur, forest centaur, mountain centaur |
| `D` | Dragon | baby/adult red/white/blue/green/yellow/black/orange/silver dragon |
| `E` | Elemental | stalker, air/fire/earth/water elemental |
| `F` | Fungus | lichen, brown mold, yellow mold, green mold, red mold, shrieker |
| `G` | Gnome | gnome, gnome lord, gnomish wizard, gnome king |
| `H` | Giant humanoid | hill giant, stone giant, fire/frost/storm giant, titan, minotaur |
| `I` | Insect | giant beetle, locust |
| `J` | Jabberwock | jabberwock, vorpal jabberwock |
| `K` | Keystone Kop | Keystone Kop, Kop Sergeant, Kop Lieutenant, Kop Kaptain |
| `L` | Lich | lich, demilich, master lich, arch-lich |
| `M` | Mummy | kobold/gnome/orc/dwarf/elf/human/ettin/giant mummy |
| `N` | Naga | red/black/golden/guardian naga |
| `O` | Ogre | ogre, ogre lord, ogre king |
| `P` | Giant humanoid | glass golem, clay golem, stone golem, iron golem |
| `Q` | Quantum mechanic | quantum mechanic |
| `R` | Rust monster | rust monster, disenchanter |
| `S` | Snake | garter snake, snake, water moccasin, pit viper, python, cobra |
| `T` | Troll | troll, ice troll, rock troll, water troll, Olog-hai |
| `U` | Umber hulk | umber hulk |
| `V` | Vampire | vampire, vampire lord, Vlad the Impaler |
| `W` | Wraith | barrow wight, wraith, Nazgul |
| `X` | Xorn | xorn |
| `Y` | Yeti | ape, owlbear, yeti, sasquatch, carnivorous ape |
| `Z` | Zruty | zruty |
| `&` | Major demon | succubus, incubus, horned devil, ice devil, balrog, pit fiend, Juiblex, Yeenoghu, Orcus, Geryon, Dispater, Baalzebub, Asmodeus, Demogorgon |
| `'` | Golem | straw golem, rope golem, leather golem, wood golem, flesh golem |
| `;` | Sea monster | jellyfish, piranha, shark, giant eel, electric eel, kraken |
| `@` | Human | tourist, shopkeeper, guard, soldier, sergeant, lieutenant, captain, watchman, medusa, wizard of yendor |
| `:` | Lizard | newt, iguana, gecko, lizard |
| `~` | Worm tail | long worm tail, purple worm tail |
| `1` | Demon lord | Death, Pestilence, Famine |
| ` ` | Ghost | ghost, shade |

---

## 8. Monster Compendium by Depth

### Early Dungeon (Depth 1-5)

| Monster | Sym | Lvl | Spd | AC | MR% | Align | Attacks | Wt | Nutr | Resists | Conveys | Diff | Notes |
|---------|-----|-----|-----|----|----|-------|---------|----|----|---------|---------|------|-------|
| **newt** | `:` | 0 | 6 | 8 | 0 | N | bite/1d1 | 10 | 20 | none | none | 1 | Weakest monster |
| **jackal** | `d` | 0 | 12 | 7 | 0 | N | bite/1d2 | 300 | 250 | none | none | 1 | Pack animal |
| **kobold** | `k` | 0 | 6 | 10 | 0 | C | weap/1d4 | 400 | 200 | poison | none | 1 | Uses weapons |
| **sewer rat** | `r` | 0 | 12 | 7 | 0 | N | bite/1d3 | 20 | 12 | none | none | 1 | Disease carrier |
| **grid bug** | `x` | 0 | 12 | 9 | 0 | N | bite/1d1/elec | 15 | 10 | elec | none | 1 | Shock damage |
| **lichen** | `F` | 0 | 1 | 9 | 0 | N | touch/0d0 | 20 | 200 | none | none | 1 | Nearly immobile |
| **gecko** | `:` | 1 | 6 | 8 | 0 | N | bite/1d3 | 10 | 20 | none | none | 1 | Small lizard |
| **acid blob** | `b` | 1 | 3 | 8 | 0 | N | passive/1d8/acid | 30 | 30 | acid/stone | acid | 2 | Corrodes on contact |
| **gnome** | `G` | 1 | 6 | 10 | 4 | N | weap/1d6 | 650 | 100 | none | none | 2 | Dwells in mines |
| **hobbit** | `h` | 1 | 9 | 10 | 0 | N | weap/1d6 | 500 | 200 | none | none | 2 | Peaceful race |
| **killer bee** | `a` | 1 | 18 | -1 | 0 | N | sting/1d3/poison | 1 | 5 | none | poison | 2 | Fast poison sting |
| **bat** | `B` | 2 | 22 | 8 | 0 | N | bite/1d4 | 20 | 20 | none | none | 2 | Very fast |
| **kitten** | `f` | 2 | 18 | 6 | 0 | N | bite/1d6 | 150 | 150 | none | none | 2 | Domestic pet |
| **little dog** | `d` | 2 | 18 | 6 | 0 | N | bite/1d6 | 150 | 150 | none | none | 2 | Domestic pet |
| **floating eye** | `e` | 2 | 1 | 9 | 10 | N | passive/0d0/plys | 10 | 10 | none | telepathy | 2 | Paralyzes on contact |
| **giant ant** | `a` | 2 | 18 | 3 | 0 | N | bite/1d4 | 10 | 10 | none | none | 3 | Fast insect |
| **soldier ant** | `a` | 3 | 18 | 3 | 0 | N | bite/2d4, sting/3d4/poison | 20 | 5 | none | poison | 4 | Dangerous early |

### Mid Dungeon (Depth 6-12)

| Monster | Sym | Lvl | Spd | AC | MR% | Align | Attacks | Wt | Nutr | Resists | Conveys | Diff | Notes |
|---------|-----|-----|-----|----|----|-------|---------|----|----|---------|---------|------|-------|
| **wolf** | `d` | 5 | 12 | 4 | 0 | N | bite/2d4 | 500 | 250 | none | none | 5 | Wild canine |
| **winter wolf** | `d` | 5 | 12 | 4 | 20 | N | bite/2d4, breath/2d6/cold | 700 | 300 | cold | cold | 7 | Cold breath |
| **large dog** | `d` | 6 | 15 | 4 | 0 | N | bite/2d4 | 800 | 400 | none | none | 6 | Grown pet |
| **cockatrice** | `c` | 5 | 6 | 6 | 30 | N | bite/1d3, touch/0d0/stone, passive/0d0/stone | 30 | 30 | poison/stone | poison/stone | 8 | Petrifies on touch |
| **gelatinous cube** | `b` | 6 | 6 | 8 | 0 | N | passive/2d4/plys, engulf/1d6/dgst | 600 | 150 | elec/poison/acid/cold/sleep/stone | none | 6 | Paralyzes, engulfs |
| **quivering blob** | `b` | 5 | 1 | 8 | 0 | N | touch/1d8 | 200 | 100 | sleep/poison | none | 5 | Slow moving |
| **orc** | `o` | 1 | 9 | 10 | 0 | C | weap/1d8 | 850 | 150 | none | none | 2 | Common enemy |
| **dwarf** | `h` | 2 | 6 | 10 | 10 | N | weap/1d8 | 900 | 300 | none | none | 3 | Lawful humanoid |
| **nymph** | `n` | 3 | 12 | 9 | 20 | N | claw/0d0/sitm, claw/0d0/sitm, claw/0d0/sedu | 600 | 300 | none | teleport | 6 | Steals items |
| **stalker** | `E` | 8 | 12 | 3 | 0 | N | claw/4d4 | 900 | 400 | none | see_invis | 8 | Invisible |
| **displacer beast** | `q` | 6 | 15 | 4 | 20 | N | claw/2d6, claw/2d6 | 1500 | 750 | none | displace | 8 | Appears displaced |
| **tiger** | `f` | 6 | 12 | 6 | 0 | N | claw/1d8, claw/1d8, bite/1d10 | 600 | 300 | none | none | 8 | Powerful feline |
| **hell hound pup** | `d` | 7 | 12 | 4 | 20 | C | bite/2d6, breath/2d6/fire | 200 | 200 | fire | fire | 8 | Fire breath |
| **wraith** | `W` | 6 | 12 | 4 | 15 | C | touch/1d6/drli | 0 | 0 | sleep/poison/drain | drain | 8 | Drains levels |
| **air elemental** | `E` | 8 | 36 | 2 | 30 | N | engulf/1d10 | 0 | 0 | poison/stone | none | 8 | Very fast |
| **fire elemental** | `E` | 8 | 12 | 2 | 30 | N | claw/3d6, passive/0d0/fire | 0 | 0 | fire/poison/stone | none | 8 | Burns on contact |
| **earth elemental** | `E` | 8 | 6 | 2 | 30 | N | claw/4d6 | 2500 | 0 | poison/stone | none | 8 | Very strong |
| **water elemental** | `E` | 8 | 6 | 2 | 30 | N | claw/5d6 | 2500 | 0 | poison/stone | none | 8 | Aquatic |
| **mind flayer** | `h` | 9 | 12 | 5 | 90 | C | weap/1d4, tent/2d1/drin, tent/2d1/drin, tent/2d1/drin, tent/2d1/drin | 1450 | 400 | none | telepathy | 11 | Drains Int |

### Late Dungeon (Depth 13-20)

| Monster | Sym | Lvl | Spd | AC | MR% | Align | Attacks | Wt | Nutr | Resists | Conveys | Diff | Notes |
|---------|-----|-----|-----|----|----|-------|---------|----|----|---------|---------|------|-------|
| **troll** | `T` | 7 | 12 | 4 | 0 | C | weap/2d6, claw/1d4, bite/1d6 | 800 | 350 | none | none | 9 | Regenerates |
| **ogre** | `O` | 5 | 10 | 5 | 0 | C | weap/2d5 | 1600 | 500 | none | none | 7 | Strong humanoid |
| **lich** | `L` | 11 | 6 | 0 | 30 | C | touch/1d10/cold, magc/0d0/spel | 1200 | 0 | cold/sleep/poison | none | 13 | Powerful undead mage |
| **demilich** | `L` | 14 | 9 | -2 | 60 | C | touch/3d4/drli, magc/0d0/spel | 1200 | 0 | fire/cold/sleep/elec/poison/stone | none | 17 | Advanced lich |
| **master lich** | `L` | 17 | 9 | -4 | 70 | C | touch/3d6/drli, magc/0d0/spel | 1200 | 0 | fire/cold/sleep/elec/poison/stone | none | 21 | Greater lich |
| **arch-lich** | `L` | 20 | 9 | -6 | 80 | C | touch/4d6/drli, magc/0d0/spel | 1200 | 0 | fire/cold/sleep/elec/poison/stone | none | 25 | Most powerful lich |
| **minotaur** | `H` | 15 | 15 | 6 | 0 | N | claw/3d10, claw/3d10, butt/2d8 | 1500 | 700 | none | none | 17 | Maze dweller |
| **jabberwock** | `J` | 15 | 12 | -2 | 50 | N | bite/2d10, bite/2d10, claw/2d10, claw/2d10 | 1300 | 600 | none | none | 18 | Vorpal beast |
| **vampire** | `V` | 10 | 12 | 1 | 25 | C | claw/1d6, bite/1d6/drli | 1450 | 400 | sleep/poison/drain | drain | 12 | Shapeshifter |
| **vampire lord** | `V` | 12 | 14 | 0 | 50 | C | claw/1d8, bite/1d8/drli | 1450 | 400 | sleep/poison/drain | drain | 14 | Powerful vampire |
| **nurse** | `@` | 11 | 6 | 0 | 0 | N | heal/2d6/heal | 1500 | 400 | none | poison | 10 | Heals self when hitting |
| **quantum mechanic** | `Q` | 7 | 12 | 3 | 10 | N | claw/1d4/tlpt | 1450 | 400 | none | teleport_ctrl | 9 | Random teleport |
| **disenchanter** | `R` | 12 | 12 | -10 | 30 | N | claw/4d4/ench | 750 | 250 | none | none | 14 | Disenchants items |
| **kraken** | `;` | 20 | 3 | 6 | 0 | N | claw/2d4, claw/2d4, hug/2d6/wrap, bite/5d4 | 1800 | 1000 | none | none | 20 | Huge sea monster |
| **shopkeeper** | `@` | 12 | 18 | 0 | 50 | N | weap/4d4, weap/4d4 | 1450 | 400 | none | none | 15 | Protects shop |
| **ghost** | ` ` | 10 | 3 | -5 | 50 | C | touch/1d1/drli | 0 | 0 | sleep/poison/stone/drain | none | 11 | Player ghost |

### Dragons (All Depths 15+)

| Monster | Sym | Lvl | Spd | AC | MR% | Align | Attacks | Wt | Nutr | Resists | Conveys | Diff | Notes |
|---------|-----|-----|-----|----|----|-------|---------|----|----|---------|---------|------|-------|
| **baby gray dragon** | `D` | 12 | 9 | -1 | 10 | N | bite/2d6, claw/1d4, claw/1d4 | 1500 | 500 | none | none | 13 | No breath yet |
| **gray dragon** | `D` | 15 | 9 | -1 | 20 | N | breath/4d6/magm, bite/3d8, claw/1d4, claw/1d4 | 4500 | 1500 | none | magic_missile | 18 | Magic missile breath |
| **baby silver dragon** | `D` | 12 | 9 | -1 | 10 | N | bite/2d6, claw/1d4, claw/1d4 | 1500 | 500 | none | none | 13 | No breath yet |
| **silver dragon** | `D` | 15 | 9 | -1 | 20 | N | breath/4d6/rbre, bite/3d8, claw/1d4, claw/1d4 | 4500 | 1500 | none | reflection | 18 | Random breath |
| **baby red dragon** | `D` | 12 | 9 | -1 | 10 | C | bite/2d6, claw/1d4, claw/1d4 | 1500 | 500 | fire | fire | 13 | Fire resist |
| **red dragon** | `D` | 15 | 9 | -1 | 20 | C | breath/6d6/fire, bite/3d8, claw/1d4, claw/1d4 | 4500 | 1500 | fire | fire | 18 | Fire breath |
| **baby white dragon** | `D` | 12 | 9 | -1 | 10 | C | bite/2d6, claw/1d4, claw/1d4 | 1500 | 500 | cold | cold | 13 | Cold resist |
| **white dragon** | `D` | 15 | 9 | -1 | 20 | C | breath/4d6/cold, bite/3d8, claw/1d4, claw/1d4 | 4500 | 1500 | cold | cold | 18 | Cold breath |
| **baby orange dragon** | `D` | 12 | 9 | -1 | 10 | C | bite/2d6, claw/1d4, claw/1d4 | 1500 | 500 | sleep | sleep | 13 | Sleep resist |
| **orange dragon** | `D` | 15 | 9 | -1 | 20 | C | breath/4d25/slee, bite/3d8, claw/1d4, claw/1d4 | 4500 | 1500 | sleep | sleep | 18 | Sleep breath |
| **baby black dragon** | `D` | 12 | 9 | -1 | 10 | C | bite/2d6, claw/1d4, claw/1d4 | 1500 | 500 | disint | disint | 13 | Disint resist |
| **black dragon** | `D` | 15 | 9 | -1 | 20 | C | breath/4d10/disn, bite/3d8, claw/1d4, claw/1d4 | 4500 | 1500 | disint | disint | 18 | Disintegration |
| **baby blue dragon** | `D` | 12 | 9 | -1 | 10 | C | bite/2d6, claw/1d4, claw/1d4 | 1500 | 500 | elec | elec | 13 | Shock resist |
| **blue dragon** | `D` | 15 | 9 | -1 | 20 | C | breath/4d6/elec, bite/3d8, claw/1d4, claw/1d4 | 4500 | 1500 | elec | elec | 18 | Lightning breath |
| **baby green dragon** | `D` | 12 | 9 | -1 | 10 | C | bite/2d6, claw/1d4, claw/1d4 | 1500 | 500 | poison | poison | 13 | Poison resist |
| **green dragon** | `D` | 15 | 9 | -1 | 20 | C | breath/4d6/drst, bite/3d8, claw/1d4, claw/1d4 | 4500 | 1500 | poison | poison | 18 | Poison breath |
| **baby yellow dragon** | `D` | 12 | 9 | -1 | 10 | C | bite/2d6, claw/1d4, claw/1d4 | 1500 | 500 | acid | acid | 13 | Acid resist |
| **yellow dragon** | `D` | 15 | 9 | -1 | 20 | C | breath/4d6/acid, bite/3d8, claw/1d4, claw/1d4 | 4500 | 1500 | acid | acid | 18 | Acid breath |

### Giants (Depth 16+)

| Monster | Sym | Lvl | Spd | AC | MR% | Align | Attacks | Wt | Nutr | Resists | Conveys | Diff | Notes |
|---------|-----|-----|-----|----|----|-------|---------|----|----|---------|---------|------|-------|
| **stone giant** | `H` | 6 | 6 | 0 | 0 | N | weap/2d10 | 2200 | 750 | none | none | 8 | Throws boulders |
| **hill giant** | `H` | 8 | 10 | 6 | 0 | C | weap/2d8 | 2200 | 750 | none | none | 9 | Common giant |
| **fire giant** | `H` | 9 | 12 | 4 | 5 | C | weap/2d10 | 2250 | 750 | fire | fire | 11 | Fire resist |
| **frost giant** | `H` | 10 | 12 | 3 | 10 | C | weap/2d12 | 2250 | 750 | cold | cold | 13 | Cold resist |
| **storm giant** | `H` | 16 | 12 | 3 | 50 | N | weap/2d12 | 2250 | 750 | elec | elec | 19 | Shock resist |
| **titan** | `H` | 16 | 18 | -3 | 70 | N | weap/2d8, magc/0d0/spel | 2300 | 900 | none | none | 19 | Casts spells |

### Major Demons (Depth 18+, Gehennom)

| Monster | Sym | Lvl | Spd | AC | MR% | Align | Attacks | Wt | Nutr | Resists | Conveys | Diff | Notes |
|---------|-----|-----|-----|----|----|-------|---------|----|----|---------|---------|------|-------|
| **succubus** | `&` | 6 | 12 | 0 | 70 | C | claw/0d0/ssex | 1450 | 400 | fire/poison | none | 9 | Seduction attack |
| **incubus** | `&` | 6 | 12 | 0 | 70 | C | claw/0d0/ssex | 1450 | 400 | fire/poison | none | 9 | Seduction attack |
| **horned devil** | `&` | 6 | 9 | -5 | 50 | C | weap/1d4, claw/1d4, bite/2d3, sting/1d3 | 1500 | 400 | fire/poison | none | 9 | Multiple attacks |
| **bone devil** | `&` | 9 | 15 | -1 | 75 | C | weap/3d4, sting/2d4 | 1500 | 400 | fire/poison | none | 12 | Fast demon |
| **ice devil** | `&` | 11 | 6 | -4 | 55 | C | claw/1d4, claw/1d4, bite/2d4, sting/3d4/cold | 1500 | 400 | fire/cold/poison | none | 14 | Cold attacks |
| **nalfeshnee** | `&` | 11 | 9 | -1 | 65 | C | claw/1d4, claw/1d4, bite/2d4, magc/0d0/spel | 1500 | 400 | fire/poison | none | 14 | Casts spells |
| **pit fiend** | `&` | 13 | 6 | -3 | 65 | C | claw/4d2, claw/4d2, hug/2d4 | 1500 | 400 | fire/poison | none | 16 | Powerful demon |
| **marilith** | `&` | 7 | 12 | -6 | 80 | C | weap/2d4, weap/2d4, claw/2d4, claw/2d4, claw/2d4, claw/2d4 | 1500 | 400 | fire/poison | none | 11 | Six attacks |
| **vrock** | `&` | 8 | 12 | 0 | 50 | C | claw/1d4, claw/1d4, claw/1d8, claw/1d8, bite/1d6 | 1500 | 400 | fire/poison | none | 11 | Multiple claws |
| **balrog** | `&` | 12 | 15 | -2 | 75 | C | weap/2d6, weap/2d6 | 1500 | 400 | fire/poison | none | 15 | Dual wield |

### Demon Lords & Princes (Unique, Gehennom)

| Monster | Sym | Lvl | Spd | AC | MR% | Align | Attacks | Wt | Nutr | Resists | Conveys | Diff | Notes |
|---------|-----|-----|-----|----|----|-------|---------|----|----|---------|---------|------|-------|
| **Juiblex** | `&` | 50 | 3 | -7 | 65 | C | engulf/4d10/dgst, spit/3d6/acid | 1500 | 0 | fire/poison/acid/stone | none | 65 | Demon lord of ooze |
| **Yeenoghu** | `&` | 56 | 18 | -5 | 80 | C | weap/3d6, weap/2d8, claw/1d6, magc/2d6/conf | 1500 | 0 | fire/poison | none | 68 | Demon lord, gnolls |
| **Orcus** | `&` | 66 | 9 | -6 | 85 | C | weap/3d6, claw/3d4, claw/3d4, sting/2d4/drst, magc/0d0/spel | 1500 | 0 | fire/poison | none | 78 | Prince of undead |
| **Geryon** | `&` | 72 | 3 | -3 | 75 | C | claw/3d6, claw/3d6, sting/2d4/drst | 1500 | 0 | fire/poison | none | 80 | Arch-devil |
| **Dispater** | `&` | 78 | 15 | -2 | 80 | C | weap/4d6, magc/0d0/spel | 1500 | 0 | fire/poison | none | 84 | Arch-devil |
| **Baalzebub** | `&` | 89 | 9 | -5 | 85 | C | bite/2d6, gaze/2d6 | 1500 | 0 | fire/poison | none | 94 | Lord of flies |
| **Asmodeus** | `&` | 105 | 12 | -7 | 90 | C | claw/1d4, claw/1d4, magc/0d0/cold | 1500 | 0 | fire/cold/poison | none | 109 | Overlord of hell |
| **Demogorgon** | `&` | 106 | 15 | -8 | 95 | C | claw/1d6, claw/1d6, sting/1d4/sick, magc/0d0/spel | 1500 | 0 | fire/elec/poison | none | 110 | Prince of demons |

### The Riders (Unique, Endgame)

| Monster | Sym | Lvl | Spd | AC | MR% | Align | Attacks | Wt | Nutr | Resists | Conveys | Diff | Notes |
|---------|-----|-----|-----|----|----|-------|---------|----|----|---------|---------|------|-------|
| **Death** | `1` | 30 | 12 | -5 | 100 | N | touch/8d8/deth, touch/8d8/deth | 1450 | 0 | fire/cold/elec/sleep/poison/stone/drain | none | 50 | Instant death touch |
| **Pestilence** | `1` | 30 | 12 | -5 | 100 | N | touch/8d8/pest, touch/8d8/pest | 1450 | 0 | fire/cold/elec/sleep/poison/stone/drain | none | 50 | Disease attack |
| **Famine** | `1` | 30 | 12 | -5 | 100 | N | touch/8d8/famn, touch/8d8/famn | 1450 | 0 | fire/cold/elec/sleep/poison/stone/drain | none | 50 | Hunger attack |

---

## 9. Corpse and Eating Effects

Key monsters with special corpse/eating effects:

| Monster | Effect When Eaten |
|---------|-------------------|
| **floating eye** | Gain telepathy |
| **newt** | Restores 1 HP (always safe) |
| **cockatrice** | Petrification (instant death unless lucky) |
| **acid blob** | Gain acid resistance |
| **yellow dragon** | Gain acid resistance |
| **white/baby white dragon** | Gain cold resistance |
| **winter wolf** | Gain cold resistance |
| **blue/baby blue dragon** | Gain shock resistance |
| **red/baby red dragon** | Gain fire resistance |
| **hell hound** | Gain fire resistance |
| **green/baby green dragon** | Gain poison resistance |
| **killer bee** | Gain poison resistance |
| **stalker** | Gain see invisible |
| **nymph** | Gain teleportitis |
| **quantum mechanic** | Gain teleport control |
| **wraith** | Gain level (if not undead, gain 1 XP level) |
| **nurse** | Gain poison resistance |
| **mind flayer** | Gain telepathy, lose 1 Intelligence |
| **gray/silver dragon** | Various resistances |
| **displacer beast** | Gain displacement |
| **gelatinous cube** | Resistance but sickening |
| **lizard** | Cures stoning/sliming |
| **tengu** | Gain teleportitis |

---

## 10. Monster AI Behaviors

### Movement Patterns

| Behavior | Monsters | Description |
|----------|----------|-------------|
| **Covetous** | Demon lords, Wizard of Yendor | Actively seeks player and Amulet |
| **Stalker** | Stalkers, werebeasts | Follows player relentlessly |
| **Wanderer** | Most animals | Random movement |
| **Greedy** | Dragons, giants, leprechauns | Seeks gold/gems |
| **Peaceful** | Shopkeepers, guards, pets | Won't attack unless provoked |
| **Domestic** | Dogs, cats | Can be tamed, follows owner |

### Special Behaviors

- **Shapeshifters**: Vampires, chameleons, doppelgangers change form
- **Summoners**: Demons, liches summon minions
- **Spellcasters**: Liches, angels, wizards cast offensive/defensive spells
- **Regenerators**: Trolls regenerate HP constantly
- **Exploding**: Gas spores, exploding mushrooms damage on death

---

## Implementation Notes for JAX RL Environment

### Key Considerations

1. **State Representation**: Monster data can be encoded as fixed-size vectors:
   - Base stats: [level, speed, AC, MR%, alignment] (5 floats)
   - Resistances: 10-bit vector (fire, cold, shock, etc.)
   - Flags: 3x32-bit vectors (M1/M2/M3)
   - Attacks: 6x4 matrix (type, damage_type, dice_n, dice_size)

2. **Difficulty Scaling**: `difficulty` field determines XP and spawn depth
   - Early: difficulty 1-5
   - Mid: difficulty 6-15
   - Late: difficulty 16-30
   - Endgame: difficulty 40+

3. **Breath Weapons**: Dragons use `AT_BREA` with various `AD_*` damage types
   - Cooldown mechanics needed for balance
   - Line-of-effect targeting

4. **Death Effects**: Handle special corpse properties via reward/penalty system

5. **Generation**: Use `G_*` flags and depth ranges for spawn tables
   - Small groups: `G_SGROUP` (ants, orcs)
   - Large groups: `G_LGROUP` (bees, rats)
   - Unique: `G_UNIQ` (demon lords, riders)

6. **Performance**: Pre-compute attack damage distributions for faster rollouts

---

## References

- NetHack 3.7 source: `include/permonst.h`, `include/monattk.h`, `include/monflag.h`
- NetHack wiki: https://nethackwiki.com
- Monster data: `src/monst.c` (full monster database)

---

*This reference is intended for building JAX-based RL environments and does not cover all 400+ NetHack monsters. Focus is on representative samples across depth tiers and special mechanics.*
