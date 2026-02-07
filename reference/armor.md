# NetHack 3.7 Armor Reference

Comprehensive armor data from `include/objects.h`. AC values shown as both stored (source) and effective (actual game value).

**Note**: In the source, AC is stored as `10 - ac`. For example, plate mail stored as 3 means effective AC 3 (lower is better).

## Helmets

| Name | Appearance | Known? | Magic? | AC (stored/effective) | MC | Weight | Cost | Material | Special Property |
|------|------------|--------|--------|----------------------|-----|--------|------|----------|-----------------|
| elven leather helm | leather hat | No | No | 9/9 | 0 | 3 | 8 | LEATHER | - |
| orcish helm | iron skull cap | No | No | 9/9 | 0 | 30 | 10 | IRON | - |
| dwarvish iron helm | hard hat | No | No | 8/8 | 0 | 40 | 20 | IRON | - |
| fedora | - | Yes | No | 10/10 | 0 | 3 | 1 | CLOTH | - |
| cornuthaum | conical hat | No | Yes | 10/10 | 1 | 4 | 80 | CLOTH | CLAIRVOYANT (for wizards), blocks for others |
| dunce cap | conical hat | No | Yes | 10/10 | 0 | 4 | 1 | CLOTH | Sets Int/Wis to 6 |
| dented pot | - | Yes | No | 9/9 | 0 | 10 | 8 | IRON | - |
| helm of brilliance | crystal helmet | No | Yes | 9/9 | 0 | 40 | 50 | GLASS | - |
| helmet | plumed helmet | No | No | 9/9 | 0 | 30 | 10 | IRON | - |
| helm of caution | etched helmet | No | Yes | 9/9 | 0 | 50 | 50 | IRON | WARNING |
| helm of opposite alignment | crested helmet | No | Yes | 9/9 | 0 | 50 | 50 | IRON | Changes alignment |
| helm of telepathy | visored helmet | No | Yes | 9/9 | 0 | 50 | 50 | IRON | TELEPAT |

## Body Armor

### Dragon Scale Mails (all magical)

| Name | AC (stored/effective) | MC | Weight | Cost | Property |
|------|----------------------|-----|--------|------|----------|
| gray dragon scale mail | 1/1 | 0 | 40 | 1200 | ANTIMAGIC (magic resistance) |
| gold dragon scale mail | 1/1 | 0 | 40 | 900 | Light source |
| silver dragon scale mail | 1/1 | 0 | 40 | 1200 | REFLECTING |
| red dragon scale mail | 1/1 | 0 | 40 | 900 | FIRE_RES |
| white dragon scale mail | 1/1 | 0 | 40 | 900 | COLD_RES |
| orange dragon scale mail | 1/1 | 0 | 40 | 900 | SLEEP_RES |
| black dragon scale mail | 1/1 | 0 | 40 | 1200 | DISINT_RES |
| blue dragon scale mail | 1/1 | 0 | 40 | 900 | SHOCK_RES |
| green dragon scale mail | 1/1 | 0 | 40 | 900 | POISON_RES |
| yellow dragon scale mail | 1/1 | 0 | 40 | 900 | ACID_RES |

### Dragon Scales (non-magical but confer properties)

| Name | AC (stored/effective) | MC | Weight | Cost | Property |
|------|----------------------|-----|--------|------|----------|
| gray dragon scales | 7/7 | 0 | 40 | 700 | ANTIMAGIC |
| gold dragon scales | 7/7 | 0 | 40 | 500 | Light source |
| silver dragon scales | 7/7 | 0 | 40 | 700 | REFLECTING |
| red dragon scales | 7/7 | 0 | 40 | 500 | FIRE_RES |
| white dragon scales | 7/7 | 0 | 40 | 500 | COLD_RES |
| orange dragon scales | 7/7 | 0 | 40 | 500 | SLEEP_RES |
| black dragon scales | 7/7 | 0 | 40 | 700 | DISINT_RES |
| blue dragon scales | 7/7 | 0 | 40 | 500 | SHOCK_RES |
| green dragon scales | 7/7 | 0 | 40 | 500 | POISON_RES |
| yellow dragon scales | 7/7 | 0 | 40 | 500 | ACID_RES |

### Metal Armor

| Name | Known? | Magic? | AC (stored/effective) | MC | Weight | Cost | Material |
|------|--------|--------|----------------------|-----|--------|------|----------|
| plate mail | Yes | No | 3/3 | 2 | 450 | 600 | IRON |
| crystal plate mail | Yes | No | 3/3 | 2 | 415 | 820 | GLASS |
| bronze plate mail | Yes | No | 4/4 | 1 | 450 | 400 | COPPER |
| splint mail | Yes | No | 4/4 | 1 | 400 | 80 | IRON |
| banded mail | Yes | No | 4/4 | 1 | 350 | 90 | IRON |
| dwarvish mithril-coat | Yes | No | 4/4 | 2 | 150 | 240 | MITHRIL |
| elven mithril-coat | Yes | No | 5/5 | 2 | 150 | 240 | MITHRIL |
| chain mail | Yes | No | 5/5 | 1 | 300 | 75 | IRON |
| orcish chain mail | No (crude chain mail) | No | 6/6 | 1 | 300 | 75 | IRON |
| scale mail | Yes | No | 6/6 | 1 | 250 | 45 | IRON |
| studded leather armor | Yes | No | 7/7 | 1 | 200 | 15 | LEATHER |
| ring mail | Yes | No | 7/7 | 1 | 250 | 100 | IRON |
| orcish ring mail | No (crude ring mail) | No | 8/8 | 1 | 250 | 80 | IRON |
| leather armor | Yes | No | 8/8 | 1 | 150 | 5 | LEATHER |
| leather jacket | Yes | No | 9/9 | 0 | 30 | 10 | LEATHER |

### Shirts

| Name | Known? | AC (stored/effective) | MC | Weight | Cost | Material |
|------|--------|----------------------|-----|--------|------|----------|
| Hawaiian shirt | Yes | 10/10 | 0 | 5 | 3 | CLOTH |
| T-shirt | Yes | 10/10 | 0 | 5 | 2 | CLOTH |

## Cloaks

| Name | Appearance | Known? | Magic? | AC (stored/effective) | MC | Weight | Cost | Material | Special Property |
|------|------------|--------|--------|----------------------|-----|--------|------|----------|-----------------|
| mummy wrapping | - | Yes | No | 10/10 | 1 | 3 | 2 | CLOTH | Blocks invisibility |
| elven cloak | faded pall | No | Yes | 9/9 | 1 | 10 | 60 | CLOTH | STEALTH |
| orcish cloak | coarse mantelet | No | No | 10/10 | 1 | 10 | 40 | CLOTH | - |
| dwarvish cloak | hooded cloak | No | No | 10/10 | 1 | 10 | 50 | CLOTH | - |
| oilskin cloak | slippery cloak | No | No | 9/9 | 2 | 10 | 50 | CLOTH | - |
| robe | - | Yes | Yes | 8/8 | 2 | 15 | 50 | CLOTH | - |
| alchemy smock | apron | No | Yes | 9/9 | 1 | 10 | 50 | CLOTH | POISON_RES |
| leather cloak | - | Yes | No | 9/9 | 1 | 15 | 40 | LEATHER | - |
| cloak of protection | tattered cape | No | Yes | 7/7 | 3 | 10 | 50 | CLOTH | PROTECTION (only item with MC 3) |
| cloak of invisibility | opera cloak | No | Yes | 9/9 | 1 | 10 | 60 | CLOTH | INVIS |
| cloak of magic resistance | ornamental cope | No | Yes | 9/9 | 1 | 10 | 60 | CLOTH | ANTIMAGIC |
| cloak of displacement | piece of cloth | No | Yes | 9/9 | 1 | 10 | 50 | CLOTH | DISPLACED |

## Shields

| Name | Appearance | Known? | AC (stored/effective) | MC | Weight | Cost | Material | Special Property |
|------|------------|--------|----------------------|-----|--------|------|----------|-----------------|
| small shield | - | Yes | 9/9 | 0 | 30 | 3 | WOOD | - |
| elven shield | blue and green shield | No | 8/8 | 0 | 40 | 7 | WOOD | - |
| Uruk-hai shield | white-handed shield | No | 9/9 | 0 | 50 | 7 | IRON | - |
| orcish shield | red-eyed shield | No | 9/9 | 0 | 50 | 7 | IRON | - |
| large shield | - | Yes | 8/8 | 0 | 100 | 10 | IRON | - |
| dwarvish roundshield | large round shield | No | 8/8 | 0 | 100 | 10 | IRON | - |
| shield of reflection | polished silver shield | No | 8/8 | 0 | 50 | 50 | SILVER | REFLECTING |

## Gloves

**Note**: All gloves have their colors shuffled but not materials. IRON gloves remain CLR_BROWN (HI_LEATHER) for visual consistency.

| Name | Appearance | Known? | Magic? | AC (stored/effective) | MC | Weight | Cost | Material | Special Property |
|------|------------|--------|--------|----------------------|-----|--------|------|----------|-----------------|
| leather gloves | old gloves | No | No | 9/9 | 0 | 10 | 8 | LEATHER | - |
| gauntlets of fumbling | padded gloves | No | Yes | 9/9 | 0 | 10 | 50 | LEATHER | FUMBLING |
| gauntlets of power | riding gloves | No | Yes | 9/9 | 0 | 30 | 50 | IRON | Grants superhuman strength |
| gauntlets of dexterity | fencing gloves | No | Yes | 9/9 | 0 | 10 | 50 | LEATHER | Increases dexterity |

## Boots

| Name | Appearance | Known? | Magic? | AC (stored/effective) | MC | Weight | Cost | Material | Special Property |
|------|------------|--------|--------|----------------------|-----|--------|------|----------|-----------------|
| low boots | walking shoes | No | No | 9/9 | 0 | 10 | 8 | LEATHER | - |
| iron shoes | hard shoes | No | No | 8/8 | 0 | 50 | 16 | IRON | - |
| high boots | jackboots | No | No | 8/8 | 0 | 20 | 12 | LEATHER | - |
| speed boots | combat boots | No | Yes | 9/9 | 0 | 20 | 50 | LEATHER | FAST (very fast) |
| water walking boots | jungle boots | No | Yes | 9/9 | 0 | 15 | 50 | LEATHER | WWALKING |
| jumping boots | hiking boots | No | Yes | 9/9 | 0 | 20 | 50 | LEATHER | JUMPING |
| elven boots | mud boots | No | Yes | 9/9 | 0 | 15 | 8 | LEATHER | STEALTH |
| kicking boots | buckled boots | No | Yes | 9/9 | 0 | 50 | 8 | IRON | Enhanced kicking damage |
| fumble boots | riding boots | No | Yes | 9/9 | 0 | 20 | 30 | LEATHER | FUMBLING (cursed) |
| levitation boots | snow boots | No | Yes | 9/9 | 0 | 15 | 30 | LEATHER | LEVITATION |

## Armor Type Categories

- **ARM_SUIT**: Full body armor (dragon scale mail, plate mail, chain mail, etc.)
- **ARM_SHIRT**: Undergarments (Hawaiian shirt, T-shirt)
- **ARM_CLOAK**: Cloaks and robes
- **ARM_SHIELD**: Shields
- **ARM_HELM**: Helmets and helms
- **ARM_GLOVES**: Gloves and gauntlets
- **ARM_BOOTS**: Boots and shoes

## Magic Cancellation (MC) Levels

- **MC 0**: No magical protection
- **MC 1**: Basic magical protection (most armor)
- **MC 2**: Enhanced magical protection (plate mail, dwarvish mithril-coat, elven mithril-coat, robe, oilskin cloak)
- **MC 3**: Maximum magical protection (cloak of protection only)

**Note**: Ring of protection or amulet of guarding adds +1 to total MC from worn armor (caps at MC 3).

## AC (Armor Class) Guide

Lower AC values are better. Negative AC provides the best protection.

- **AC 1**: Dragon scale mails (best AC from single piece)
- **AC 3**: Plate mail, crystal plate mail
- **AC 4**: Bronze plate mail, splint mail, banded mail, dwarvish mithril-coat
- **AC 5**: Elven mithril-coat, chain mail
- **AC 6**: Orcish chain mail, scale mail
- **AC 7**: Studded leather, ring mail, dragon scales, cloak of protection
- **AC 8**: Orcish ring mail, leather armor, dwarvish iron helm, elven shield, large shield, dwarvish roundshield, shield of reflection, robe, iron shoes, high boots
- **AC 9**: Most helmets, most cloaks, most gloves, most boots
- **AC 10**: Fedora, cornuthaum, dunce cap, Hawaiian shirt, T-shirt, mummy wrapping, orcish cloak, dwarvish cloak

## Material Properties

- **IRON**: Heavy, conductive, rusts, interferes with spellcasting
- **LEATHER**: Light, flexible, burns easily
- **CLOTH**: Very light, burns easily
- **MITHRIL**: Light, strong, doesn't rust, excellent for spellcasting
- **DRAGON_HIDE**: Fireproof, special properties per dragon type
- **GLASS**: Fragile, shatters easily
- **COPPER/BRONZE**: Medium weight, corrodes
- **SILVER**: Effective against certain monsters
- **WOOD**: Light, burns easily

## Key Notes

1. **Dragon armor**: Scale mails are magical (created with magic), scales are non-magical (natural drops) but both confer resistances
2. **Spellcasting penalty**: Iron and metal armor interferes with spellcasting; mithril does not
3. **MC stacking**: Total MC = sum of all worn armor MC values, capped at 3
4. **Cloak of protection**: Only item providing MC 3 on its own
5. **Shuffled appearances**: Many items have randomized descriptions until identified
6. **Known status**: Some items start identified (fedora, plate mail), others must be discovered
7. **Cornuthaum**: Grants clairvoyance to Wizards, blocks it for other roles
8. **Dunce cap**: Fixes Intelligence and Wisdom to 6 (protects against Int drain death)
