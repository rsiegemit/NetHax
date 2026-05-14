"""Monster entries chunk 6 — vendor/nethack/include/monsters.h lines 3164-3914."""
from Nethax.nethax.constants.monsters import (
    MonsterEntry, MonsterSymbol, AttackType, DamageType,
    # M1_* flags
    M1_FLY, M1_SWIM, M1_AMPHIBIOUS, M1_BREATHLESS, M1_NOTAKE,
    M1_NOEYES, M1_NOHANDS, M1_NOLIMBS, M1_NOHEAD, M1_MINDLESS,
    M1_HUMANOID, M1_ANIMAL, M1_SLITHY, M1_THICK_HIDE, M1_OVIPAROUS,
    M1_REGEN, M1_SEE_INVIS, M1_TPORT_CNTRL, M1_POIS,
    M1_CARNIVORE, M1_HERBIVORE, M1_OMNIVORE,
    M1_TUNNEL, M1_NEEDPICK,
    # M2_* flags
    M2_NOPOLY, M2_HUMAN, M2_DEMON, M2_GIANT,
    M2_MALE, M2_FEMALE,
    M2_PNAME, M2_HOSTILE, M2_PEACEFUL, M2_STALK, M2_WANDER,
    M2_NASTY, M2_STRONG, M2_ROCKTHROW,
    M2_GREEDY, M2_JEWELS, M2_COLLECT, M2_MAGIC,
    M2_PRINCE, M2_SHAPESHIFTER, M2_ORC,
    # M3_* flags
    M3_WANTSAMUL, M3_WANTSARTI, M3_WAITFORU,
    M3_CLOSE, M3_INFRAVISION, M3_INFRAVISIBLE, M3_DISPLACES,
    # MR_* resistance bits
    MR_FIRE, MR_COLD, MR_SLEEP, MR_DISINT, MR_ELEC, MR_POISON, MR_ACID, MR_STONE,
    # MS_* sound constants
    MS_SILENT, MS_DJINNI, MS_MUMBLE, MS_BRIBE, MS_RIDER,
    MS_HUMANOID, MS_LEADER, MS_NEMESIS, MS_GUARDIAN,
    MS_BELLOW, MS_GROWL, MS_CHIRP, MS_SQEEK,
    # MZ_* size constants
    MZ_TINY, MZ_SMALL, MZ_MEDIUM, MZ_HUMAN, MZ_LARGE, MZ_HUGE, MZ_GIGANTIC,
    # G_* generation mask bits
    G_UNIQ, G_NOGEN, G_HELL, G_GENO, G_NOCORPSE, G_SGROUP, G_LGROUP,
    # Weight constants
    WT_HUMAN, WT_DRAGON,
    # Colour constants
    CLR_BLACK, CLR_RED, CLR_GREEN, CLR_BROWN, CLR_BLUE, CLR_MAGENTA,
    CLR_CYAN, CLR_GRAY, CLR_ORANGE, CLR_BRIGHT_BLUE, CLR_BRIGHT_GREEN,
    CLR_YELLOW, CLR_BRIGHT_MAGENTA, CLR_BRIGHT_CYAN, CLR_WHITE,
    HI_LORD, HI_DOMESTIC, HI_OVERLORD,
    # NO_ATTK sentinel
    NO_ATTK,
)

ENTRIES = (
    # --- other demons ---

    # djinni
    MonsterEntry(
        name="djinni",
        symbol=MonsterSymbol.S_DEMON,
        level=7, move_speed=12, ac=4, mr=30, alignment=0,
        generation_mask=G_NOGEN | G_NOCORPSE,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 8),
        ),
        weight=1500, nutrition=400, sound=MS_DJINNI, size=MZ_HUMAN,
        resists_mask=MR_POISON | MR_STONE,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_FLY | M1_POIS,
        flags2=M2_NOPOLY | M2_STALK | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=CLR_YELLOW,
        difficulty=8,
    ),

    # --- sea monsters ---

    # jellyfish
    MonsterEntry(
        name="jellyfish",
        symbol=MonsterSymbol.S_EEL,
        level=3, move_speed=3, ac=6, mr=0, alignment=0,
        generation_mask=G_GENO | G_NOGEN,
        attacks=(
            (AttackType.AT_STNG, DamageType.AD_DRST, 3, 3),
        ),
        weight=80, nutrition=20, sound=MS_SILENT, size=MZ_SMALL,
        resists_mask=MR_POISON,
        conveys_mask=MR_POISON,
        flags1=M1_SWIM | M1_AMPHIBIOUS | M1_SLITHY | M1_NOLIMBS | M1_NOHEAD | M1_NOTAKE | M1_POIS,
        flags2=M2_HOSTILE,
        flags3=0,
        color=CLR_BLUE,
        difficulty=5,
    ),

    # piranha
    MonsterEntry(
        name="piranha",
        symbol=MonsterSymbol.S_EEL,
        level=5, move_speed=12, ac=4, mr=0, alignment=0,
        generation_mask=G_GENO | G_NOGEN | G_SGROUP,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=60, nutrition=30, sound=MS_SILENT, size=MZ_SMALL,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_SWIM | M1_AMPHIBIOUS | M1_ANIMAL | M1_SLITHY | M1_NOLIMBS | M1_CARNIVORE | M1_OVIPAROUS | M1_NOTAKE,
        flags2=M2_HOSTILE,
        flags3=0,
        color=CLR_RED,
        difficulty=6,
    ),

    # shark
    MonsterEntry(
        name="shark",
        symbol=MonsterSymbol.S_EEL,
        level=7, move_speed=12, ac=2, mr=0, alignment=0,
        generation_mask=G_GENO | G_NOGEN,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 5, 6),
        ),
        weight=500, nutrition=350, sound=MS_SILENT, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_SWIM | M1_AMPHIBIOUS | M1_ANIMAL | M1_SLITHY | M1_NOLIMBS | M1_CARNIVORE | M1_OVIPAROUS | M1_THICK_HIDE | M1_NOTAKE,
        flags2=M2_HOSTILE,
        flags3=0,
        color=CLR_GRAY,
        difficulty=9,
    ),

    # giant eel
    MonsterEntry(
        name="giant eel",
        symbol=MonsterSymbol.S_EEL,
        level=5, move_speed=9, ac=-1, mr=0, alignment=0,
        generation_mask=G_GENO | G_NOGEN,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 3, 6),
            (AttackType.AT_TUCH, DamageType.AD_WRAP, 0, 0),
        ),
        weight=200, nutrition=250, sound=MS_SILENT, size=MZ_HUGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_SWIM | M1_AMPHIBIOUS | M1_ANIMAL | M1_SLITHY | M1_NOLIMBS | M1_CARNIVORE | M1_OVIPAROUS | M1_NOTAKE,
        flags2=M2_HOSTILE,
        flags3=M3_INFRAVISIBLE,
        color=CLR_CYAN,
        difficulty=7,
    ),

    # electric eel
    MonsterEntry(
        name="electric eel",
        symbol=MonsterSymbol.S_EEL,
        level=7, move_speed=10, ac=-3, mr=0, alignment=0,
        generation_mask=G_GENO | G_NOGEN,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_ELEC, 4, 6),
            (AttackType.AT_TUCH, DamageType.AD_WRAP, 0, 0),
        ),
        weight=200, nutrition=250, sound=MS_SILENT, size=MZ_HUGE,
        resists_mask=MR_ELEC,
        conveys_mask=MR_ELEC,
        flags1=M1_SWIM | M1_AMPHIBIOUS | M1_ANIMAL | M1_SLITHY | M1_NOLIMBS | M1_CARNIVORE | M1_OVIPAROUS | M1_NOTAKE,
        flags2=M2_HOSTILE,
        flags3=M3_INFRAVISIBLE,
        color=CLR_BRIGHT_BLUE,
        difficulty=10,
    ),

    # kraken
    MonsterEntry(
        name="kraken",
        symbol=MonsterSymbol.S_EEL,
        level=20, move_speed=3, ac=6, mr=0, alignment=-3,
        generation_mask=G_GENO | G_NOGEN,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 4),
            (AttackType.AT_HUGS, DamageType.AD_WRAP, 2, 6),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 5, 4),
        ),
        weight=1800, nutrition=1000, sound=MS_SILENT, size=MZ_HUGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_SWIM | M1_AMPHIBIOUS | M1_ANIMAL | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_NOPOLY | M2_HOSTILE | M2_STRONG,
        flags3=M3_INFRAVISIBLE,
        color=CLR_RED,
        difficulty=22,
    ),

    # --- lizards, &c ---

    # newt
    MonsterEntry(
        name="newt",
        symbol=MonsterSymbol.S_LIZARD,
        level=0, move_speed=6, ac=8, mr=0, alignment=0,
        generation_mask=G_GENO | 5,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 2),
        ),
        weight=10, nutrition=20, sound=MS_SILENT, size=MZ_TINY,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_SWIM | M1_AMPHIBIOUS | M1_ANIMAL | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_HOSTILE,
        flags3=0,
        color=CLR_YELLOW,
        difficulty=1,
    ),

    # gecko
    MonsterEntry(
        name="gecko",
        symbol=MonsterSymbol.S_LIZARD,
        level=1, move_speed=6, ac=8, mr=0, alignment=0,
        generation_mask=G_GENO | 5,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 3),
        ),
        weight=10, nutrition=20, sound=MS_SQEEK, size=MZ_TINY,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_ANIMAL | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_HOSTILE,
        flags3=0,
        color=CLR_GREEN,
        difficulty=2,
    ),

    # iguana
    MonsterEntry(
        name="iguana",
        symbol=MonsterSymbol.S_LIZARD,
        level=2, move_speed=6, ac=7, mr=0, alignment=0,
        generation_mask=G_GENO | 5,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 4),
        ),
        weight=30, nutrition=30, sound=MS_SILENT, size=MZ_TINY,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_ANIMAL | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_HOSTILE,
        flags3=0,
        color=CLR_BROWN,
        difficulty=3,
    ),

    # baby crocodile
    MonsterEntry(
        name="baby crocodile",
        symbol=MonsterSymbol.S_LIZARD,
        level=3, move_speed=6, ac=7, mr=0, alignment=0,
        generation_mask=G_GENO,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 4),
        ),
        weight=200, nutrition=200, sound=MS_SILENT, size=MZ_MEDIUM,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_SWIM | M1_AMPHIBIOUS | M1_ANIMAL | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_HOSTILE,
        flags3=0,
        color=CLR_BROWN,
        difficulty=4,
    ),

    # lizard
    MonsterEntry(
        name="lizard",
        symbol=MonsterSymbol.S_LIZARD,
        level=5, move_speed=6, ac=6, mr=10, alignment=0,
        generation_mask=G_GENO | 5,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 6),
        ),
        weight=10, nutrition=40, sound=MS_SILENT, size=MZ_TINY,
        resists_mask=MR_STONE,
        conveys_mask=MR_STONE,
        flags1=M1_ANIMAL | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_HOSTILE,
        flags3=0,
        color=CLR_GREEN,
        difficulty=6,
    ),

    # chameleon
    MonsterEntry(
        name="chameleon",
        symbol=MonsterSymbol.S_LIZARD,
        level=6, move_speed=5, ac=6, mr=10, alignment=0,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 4, 2),
        ),
        weight=100, nutrition=100, sound=MS_SILENT, size=MZ_TINY,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_ANIMAL | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_NOPOLY | M2_HOSTILE | M2_SHAPESHIFTER,
        flags3=0,
        color=CLR_BROWN,
        difficulty=7,
    ),

    # crocodile
    MonsterEntry(
        name="crocodile",
        symbol=MonsterSymbol.S_LIZARD,
        level=6, move_speed=9, ac=5, mr=0, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 4, 2),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 12),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_SILENT, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_SWIM | M1_AMPHIBIOUS | M1_ANIMAL | M1_THICK_HIDE | M1_NOHANDS | M1_OVIPAROUS | M1_CARNIVORE,
        flags2=M2_STRONG | M2_HOSTILE,
        flags3=0,
        color=CLR_BROWN,
        difficulty=7,
    ),

    # salamander
    MonsterEntry(
        name="salamander",
        symbol=MonsterSymbol.S_LIZARD,
        level=8, move_speed=12, ac=-1, mr=0, alignment=-9,
        generation_mask=G_HELL | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 8),
            (AttackType.AT_TUCH, DamageType.AD_FIRE, 1, 6),
            (AttackType.AT_HUGS, DamageType.AD_PHYS, 2, 6),
            (AttackType.AT_HUGS, DamageType.AD_FIRE, 3, 6),
        ),
        weight=1500, nutrition=400, sound=MS_MUMBLE, size=MZ_HUMAN,
        resists_mask=MR_SLEEP | MR_FIRE,
        conveys_mask=MR_FIRE,
        flags1=M1_HUMANOID | M1_SLITHY | M1_THICK_HIDE | M1_POIS,
        flags2=M2_STALK | M2_HOSTILE | M2_COLLECT | M2_MAGIC,
        flags3=M3_INFRAVISIBLE,
        color=CLR_ORANGE,
        difficulty=12,
    ),

    # --- dummy monster for visual interface ---

    # long worm tail
    MonsterEntry(
        name="long worm tail",
        symbol=MonsterSymbol.S_WORM_TAIL,
        level=0, move_speed=0, ac=0, mr=0, alignment=0,
        generation_mask=G_NOGEN | G_NOCORPSE | G_UNIQ,
        attacks=(),
        weight=0, nutrition=0, sound=0, size=0,
        resists_mask=0,
        conveys_mask=0,
        flags1=0,
        flags2=M2_NOPOLY,
        flags3=0,
        color=CLR_BROWN,
        difficulty=1,
    ),

    # --- character classes ---

    # archeologist
    MonsterEntry(
        name="archeologist",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=1, alignment=3,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_TUNNEL | M1_NEEDPICK | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=12,
    ),

    # barbarian
    MonsterEntry(
        name="barbarian",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=1, alignment=0,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=MR_POISON,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=12,
    ),

    # caveman (vendor parity: NLE has separate caveman/cavewoman)
    MonsterEntry(
        name="caveman",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=0, alignment=1,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 4),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_MALE | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=12,
    ),

    # cavewoman (vendor parity: NLE has separate caveman/cavewoman)
    MonsterEntry(
        name="cavewoman",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=0, alignment=1,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 4),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_FEMALE | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=12,
    ),

    # healer
    MonsterEntry(
        name="healer",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=1, alignment=0,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=MR_POISON,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=12,
    ),

    # knight
    MonsterEntry(
        name="knight",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=1, alignment=3,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=12,
    ),

    # monk
    MonsterEntry(
        name="monk",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=2, alignment=0,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 8),
            (AttackType.AT_KICK, DamageType.AD_PHYS, 1, 8),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_HERBIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_MALE | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=11,
    ),

    # priest (vendor parity: NLE has separate priest/priestess in PC class)
    MonsterEntry(
        name="priest",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=2, alignment=0,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_MALE | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=12,
    ),

    # priestess (vendor parity: NLE has separate priest/priestess in PC class)
    MonsterEntry(
        name="priestess",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=2, alignment=0,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_FEMALE | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=12,
    ),

    # ranger
    MonsterEntry(
        name="ranger",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=2, alignment=-3,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 4),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=12,
    ),

    # rogue
    MonsterEntry(
        name="rogue",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=1, alignment=-3,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_GREEDY | M2_JEWELS | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=12,
    ),

    # samurai
    MonsterEntry(
        name="samurai",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=1, alignment=3,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=12,
    ),

    # tourist
    MonsterEntry(
        name="tourist",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=1, alignment=0,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=12,
    ),

    # valkyrie
    MonsterEntry(
        name="valkyrie",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=1, alignment=-1,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=MR_COLD,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_FEMALE | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=12,
    ),

    # wizard (player class)
    MonsterEntry(
        name="wizard",
        symbol=MonsterSymbol.S_HUMAN,
        level=10, move_speed=12, ac=10, mr=3, alignment=0,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_COLLECT | M2_MAGIC,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=12,
    ),

    # --- quest leaders ---

    # Lord Carnarvon
    MonsterEntry(
        name="Lord Carnarvon",
        symbol=MonsterSymbol.S_HUMAN,
        level=20, move_speed=12, ac=0, mr=30, alignment=20,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_LEADER, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_TUNNEL | M1_NEEDPICK | M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PNAME | M2_PEACEFUL | M2_STRONG | M2_MALE | M2_COLLECT | M2_MAGIC,
        flags3=M3_CLOSE | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=22,
    ),

    # Pelias
    MonsterEntry(
        name="Pelias",
        symbol=MonsterSymbol.S_HUMAN,
        level=20, move_speed=12, ac=0, mr=30, alignment=0,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_LEADER, size=MZ_HUMAN,
        resists_mask=MR_POISON,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PNAME | M2_PEACEFUL | M2_STRONG | M2_MALE | M2_COLLECT | M2_MAGIC,
        flags3=M3_CLOSE | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=22,
    ),

    # Shaman Karnov
    MonsterEntry(
        name="Shaman Karnov",
        symbol=MonsterSymbol.S_HUMAN,
        level=20, move_speed=12, ac=0, mr=30, alignment=20,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 4),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_LEADER, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PNAME | M2_PEACEFUL | M2_STRONG | M2_MALE | M2_COLLECT | M2_MAGIC,
        flags3=M3_CLOSE | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=22,
    ),

    # Hippocrates
    MonsterEntry(
        name="Hippocrates",
        symbol=MonsterSymbol.S_HUMAN,
        level=20, move_speed=12, ac=0, mr=40, alignment=0,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_LEADER, size=MZ_HUMAN,
        resists_mask=MR_POISON,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PNAME | M2_PEACEFUL | M2_STRONG | M2_MALE | M2_COLLECT | M2_MAGIC,
        flags3=M3_CLOSE | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=22,
    ),

    # King Arthur
    MonsterEntry(
        name="King Arthur",
        symbol=MonsterSymbol.S_HUMAN,
        level=20, move_speed=12, ac=0, mr=40, alignment=20,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_LEADER, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PNAME | M2_PEACEFUL | M2_STRONG | M2_MALE | M2_COLLECT | M2_MAGIC,
        flags3=M3_CLOSE | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=23,
    ),

    # Grand Master
    MonsterEntry(
        name="Grand Master",
        symbol=MonsterSymbol.S_HUMAN,
        level=25, move_speed=12, ac=0, mr=70, alignment=0,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 4, 10),
            (AttackType.AT_KICK, DamageType.AD_PHYS, 2, 8),
            (AttackType.AT_MAGC, DamageType.AD_CLRC, 2, 8),
            (AttackType.AT_MAGC, DamageType.AD_CLRC, 2, 8),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_LEADER, size=MZ_HUMAN,
        resists_mask=MR_FIRE | MR_ELEC | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_SEE_INVIS | M1_HERBIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_MALE | M2_NASTY | M2_MAGIC,
        flags3=M3_CLOSE | M3_INFRAVISIBLE,
        color=CLR_BLACK,
        difficulty=30,
    ),

    # Arch Priest
    MonsterEntry(
        name="Arch Priest",
        symbol=MonsterSymbol.S_HUMAN,
        level=25, move_speed=12, ac=7, mr=70, alignment=0,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 4, 10),
            (AttackType.AT_KICK, DamageType.AD_PHYS, 2, 8),
            (AttackType.AT_MAGC, DamageType.AD_CLRC, 2, 8),
            (AttackType.AT_MAGC, DamageType.AD_CLRC, 2, 8),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_LEADER, size=MZ_HUMAN,
        resists_mask=MR_FIRE | MR_ELEC | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_SEE_INVIS | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_MALE | M2_COLLECT | M2_MAGIC,
        flags3=M3_CLOSE | M3_INFRAVISIBLE,
        color=CLR_WHITE,
        difficulty=30,
    ),

    # Orion
    MonsterEntry(
        name="Orion",
        symbol=MonsterSymbol.S_HUMAN,
        level=20, move_speed=12, ac=0, mr=30, alignment=0,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=2200, nutrition=700, sound=MS_LEADER, size=MZ_HUGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE | M1_SEE_INVIS | M1_SWIM | M1_AMPHIBIOUS,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PNAME | M2_PEACEFUL | M2_STRONG | M2_MALE | M2_COLLECT | M2_MAGIC,
        flags3=M3_CLOSE | M3_INFRAVISION | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=22,
    ),

    # Master of Thieves
    MonsterEntry(
        name="Master of Thieves",
        symbol=MonsterSymbol.S_HUMAN,
        level=20, move_speed=12, ac=0, mr=30, alignment=-20,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 6),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 6),
            (AttackType.AT_CLAW, DamageType.AD_SAMU, 2, 4),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_LEADER, size=MZ_HUMAN,
        resists_mask=MR_STONE,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_MALE | M2_GREEDY | M2_JEWELS | M2_COLLECT | M2_MAGIC,
        flags3=M3_CLOSE | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=24,
    ),

    # Lord Sato
    MonsterEntry(
        name="Lord Sato",
        symbol=MonsterSymbol.S_HUMAN,
        level=20, move_speed=12, ac=0, mr=30, alignment=20,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_LEADER, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PNAME | M2_PEACEFUL | M2_STRONG | M2_MALE | M2_COLLECT | M2_MAGIC,
        flags3=M3_CLOSE | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=23,
    ),

    # Twoflower
    MonsterEntry(
        name="Twoflower",
        symbol=MonsterSymbol.S_HUMAN,
        level=20, move_speed=12, ac=10, mr=20, alignment=0,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_LEADER, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PNAME | M2_PEACEFUL | M2_STRONG | M2_MALE | M2_COLLECT | M2_MAGIC,
        flags3=M3_CLOSE | M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=22,
    ),

    # Norn
    MonsterEntry(
        name="Norn",
        symbol=MonsterSymbol.S_HUMAN,
        level=20, move_speed=12, ac=0, mr=80, alignment=0,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=1800, nutrition=550, sound=MS_LEADER, size=MZ_HUGE,
        resists_mask=MR_COLD,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_FEMALE | M2_COLLECT | M2_MAGIC,
        flags3=M3_CLOSE | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=23,
    ),

    # Neferet the Green
    MonsterEntry(
        name="Neferet the Green",
        symbol=MonsterSymbol.S_HUMAN,
        level=20, move_speed=12, ac=0, mr=60, alignment=0,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 2, 8),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_LEADER, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_FEMALE | M2_PNAME | M2_PEACEFUL | M2_STRONG | M2_COLLECT | M2_MAGIC,
        flags3=M3_CLOSE | M3_INFRAVISIBLE,
        color=CLR_GREEN,
        difficulty=23,
    ),

    # --- quest nemeses ---

    # Minion of Huhetotl
    MonsterEntry(
        name="Minion of Huhetotl",
        symbol=MonsterSymbol.S_DEMON,
        level=16, move_speed=12, ac=-2, mr=75, alignment=-14,
        generation_mask=G_NOCORPSE | G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 8, 4),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 4, 6),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
            (AttackType.AT_CLAW, DamageType.AD_SAMU, 2, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_NEMESIS, size=MZ_LARGE,
        resists_mask=MR_FIRE | MR_POISON | MR_STONE,
        conveys_mask=0,
        flags1=M1_FLY | M1_SEE_INVIS | M1_POIS,
        flags2=M2_NOPOLY | M2_DEMON | M2_STALK | M2_HOSTILE | M2_STRONG | M2_NASTY | M2_COLLECT,
        flags3=M3_WANTSARTI | M3_WAITFORU | M3_INFRAVISION | M3_INFRAVISIBLE,
        color=CLR_RED,
        difficulty=23,
    ),

    # Thoth Amon
    MonsterEntry(
        name="Thoth Amon",
        symbol=MonsterSymbol.S_HUMAN,
        level=16, move_speed=12, ac=0, mr=10, alignment=-14,
        generation_mask=G_NOGEN | G_UNIQ | G_NOCORPSE,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
            (AttackType.AT_CLAW, DamageType.AD_SAMU, 1, 4),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_NEMESIS, size=MZ_HUMAN,
        resists_mask=MR_POISON | MR_STONE,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PNAME | M2_STRONG | M2_MALE | M2_STALK | M2_HOSTILE | M2_NASTY | M2_COLLECT | M2_MAGIC,
        flags3=M3_WANTSARTI | M3_WAITFORU | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=22,
    ),

    # Chromatic Dragon
    MonsterEntry(
        name="Chromatic Dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=16, move_speed=12, ac=0, mr=30, alignment=-14,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_BREA, DamageType.AD_RBRE, 6, 6),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
            (AttackType.AT_CLAW, DamageType.AD_SAMU, 2, 8),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 4, 8),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 4, 8),
            (AttackType.AT_STNG, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_DRAGON, nutrition=1700, sound=MS_NEMESIS, size=MZ_GIGANTIC,
        resists_mask=MR_FIRE | MR_COLD | MR_SLEEP | MR_DISINT | MR_ELEC | MR_POISON | MR_ACID | MR_STONE,
        conveys_mask=MR_FIRE | MR_COLD | MR_SLEEP | MR_DISINT | MR_ELEC | MR_POISON | MR_STONE,
        flags1=M1_THICK_HIDE | M1_NOHANDS | M1_CARNIVORE | M1_SEE_INVIS | M1_POIS,
        flags2=M2_NOPOLY | M2_HOSTILE | M2_FEMALE | M2_STALK | M2_STRONG | M2_NASTY | M2_GREEDY | M2_JEWELS | M2_MAGIC,
        flags3=M3_WANTSARTI | M3_WAITFORU | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=23,
    ),

    # Cyclops
    MonsterEntry(
        name="Cyclops",
        symbol=MonsterSymbol.S_GIANT,
        level=18, move_speed=12, ac=0, mr=0, alignment=-15,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 4, 8),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 4, 8),
            (AttackType.AT_CLAW, DamageType.AD_SAMU, 2, 6),
        ),
        weight=1900, nutrition=700, sound=MS_NEMESIS, size=MZ_HUGE,
        resists_mask=MR_STONE,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_GIANT | M2_STRONG | M2_ROCKTHROW | M2_STALK | M2_HOSTILE | M2_NASTY | M2_MALE | M2_JEWELS | M2_COLLECT,
        flags3=M3_WANTSARTI | M3_WAITFORU | M3_INFRAVISION | M3_INFRAVISIBLE,
        color=CLR_GRAY,
        difficulty=23,
    ),

    # Ixoth
    MonsterEntry(
        name="Ixoth",
        symbol=MonsterSymbol.S_DRAGON,
        level=15, move_speed=12, ac=-1, mr=20, alignment=-14,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_BREA, DamageType.AD_FIRE, 8, 6),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 4, 8),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 4),
            (AttackType.AT_CLAW, DamageType.AD_SAMU, 2, 4),
        ),
        weight=WT_DRAGON, nutrition=1600, sound=MS_NEMESIS, size=MZ_GIGANTIC,
        resists_mask=MR_FIRE | MR_STONE,
        conveys_mask=MR_FIRE,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_CARNIVORE | M1_SEE_INVIS,
        flags2=M2_NOPOLY | M2_MALE | M2_PNAME | M2_HOSTILE | M2_STRONG | M2_NASTY | M2_STALK | M2_GREEDY | M2_JEWELS | M2_MAGIC,
        flags3=M3_WANTSARTI | M3_WAITFORU | M3_INFRAVISIBLE,
        color=CLR_RED,
        difficulty=22,
    ),

    # Master Kaen
    MonsterEntry(
        name="Master Kaen",
        symbol=MonsterSymbol.S_HUMAN,
        level=25, move_speed=12, ac=-10, mr=10, alignment=-20,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 16, 2),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 16, 2),
            (AttackType.AT_MAGC, DamageType.AD_CLRC, 0, 0),
            (AttackType.AT_CLAW, DamageType.AD_SAMU, 1, 4),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_NEMESIS, size=MZ_HUMAN,
        resists_mask=MR_POISON | MR_STONE,
        conveys_mask=MR_POISON,
        flags1=M1_HUMANOID | M1_HERBIVORE | M1_SEE_INVIS,
        flags2=M2_NOPOLY | M2_HUMAN | M2_MALE | M2_PNAME | M2_HOSTILE | M2_STRONG | M2_NASTY | M2_STALK | M2_COLLECT | M2_MAGIC,
        flags3=M3_WANTSARTI | M3_WAITFORU | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=31,
    ),

    # Nalzok
    MonsterEntry(
        name="Nalzok",
        symbol=MonsterSymbol.S_DEMON,
        level=16, move_speed=12, ac=-2, mr=85, alignment=-127,
        generation_mask=G_NOGEN | G_UNIQ | G_NOCORPSE,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 8, 4),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 4, 6),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
            (AttackType.AT_CLAW, DamageType.AD_SAMU, 2, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_NEMESIS, size=MZ_LARGE,
        resists_mask=MR_FIRE | MR_POISON | MR_STONE,
        conveys_mask=0,
        flags1=M1_FLY | M1_SEE_INVIS | M1_POIS,
        flags2=M2_NOPOLY | M2_DEMON | M2_MALE | M2_PNAME | M2_HOSTILE | M2_STRONG | M2_STALK | M2_NASTY | M2_COLLECT,
        flags3=M3_WANTSARTI | M3_WAITFORU | M3_INFRAVISION | M3_INFRAVISIBLE,
        color=CLR_RED,
        difficulty=23,
    ),

    # Scorpius
    MonsterEntry(
        name="Scorpius",
        symbol=MonsterSymbol.S_SPIDER,
        level=15, move_speed=12, ac=10, mr=0, alignment=-15,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 6),
            (AttackType.AT_CLAW, DamageType.AD_SAMU, 2, 6),
            (AttackType.AT_STNG, DamageType.AD_DISE, 1, 4),
        ),
        weight=750, nutrition=350, sound=MS_NEMESIS, size=MZ_HUMAN,
        resists_mask=MR_POISON | MR_STONE,
        conveys_mask=MR_POISON,
        flags1=M1_ANIMAL | M1_NOHANDS | M1_OVIPAROUS | M1_POIS | M1_CARNIVORE,
        flags2=M2_NOPOLY | M2_MALE | M2_PNAME | M2_HOSTILE | M2_STRONG | M2_STALK | M2_NASTY | M2_COLLECT | M2_MAGIC,
        flags3=M3_WANTSARTI | M3_WAITFORU,
        color=HI_LORD,
        difficulty=17,
    ),

    # Master Assassin
    MonsterEntry(
        name="Master Assassin",
        symbol=MonsterSymbol.S_HUMAN,
        level=15, move_speed=12, ac=0, mr=30, alignment=18,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_DRST, 2, 6),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 8),
            (AttackType.AT_CLAW, DamageType.AD_SAMU, 2, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_NEMESIS, size=MZ_HUMAN,
        resists_mask=MR_STONE,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_MALE | M2_HOSTILE | M2_STALK | M2_NASTY | M2_COLLECT | M2_MAGIC,
        flags3=M3_WANTSARTI | M3_WAITFORU | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=20,
    ),

    # Ashikaga Takauji
    MonsterEntry(
        name="Ashikaga Takauji",
        symbol=MonsterSymbol.S_HUMAN,
        level=15, move_speed=12, ac=0, mr=40, alignment=-13,
        generation_mask=G_NOGEN | G_UNIQ | G_NOCORPSE,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 6),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 6),
            (AttackType.AT_CLAW, DamageType.AD_SAMU, 2, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_NEMESIS, size=MZ_HUMAN,
        resists_mask=MR_STONE,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PNAME | M2_HOSTILE | M2_STRONG | M2_STALK | M2_NASTY | M2_MALE | M2_COLLECT | M2_MAGIC,
        flags3=M3_WANTSARTI | M3_WAITFORU | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=19,
    ),

    # Lord Surtur
    MonsterEntry(
        name="Lord Surtur",
        symbol=MonsterSymbol.S_GIANT,
        level=15, move_speed=12, ac=2, mr=50, alignment=12,
        generation_mask=G_NOGEN | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 10),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 10),
            (AttackType.AT_CLAW, DamageType.AD_SAMU, 2, 6),
        ),
        weight=2250, nutrition=850, sound=MS_NEMESIS, size=MZ_HUGE,
        resists_mask=MR_FIRE | MR_STONE,
        conveys_mask=MR_FIRE,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_GIANT | M2_MALE | M2_PNAME | M2_HOSTILE | M2_STALK | M2_STRONG | M2_NASTY | M2_ROCKTHROW | M2_JEWELS | M2_COLLECT,
        flags3=M3_WANTSARTI | M3_WAITFORU | M3_INFRAVISION | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=19,
    ),

    # Dark One
    MonsterEntry(
        name="Dark One",
        symbol=MonsterSymbol.S_HUMAN,
        level=15, move_speed=12, ac=0, mr=80, alignment=-10,
        generation_mask=G_NOGEN | G_UNIQ | G_NOCORPSE,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_CLAW, DamageType.AD_SAMU, 1, 4),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_NEMESIS, size=MZ_HUMAN,
        resists_mask=MR_STONE,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_STRONG | M2_HOSTILE | M2_STALK | M2_NASTY | M2_COLLECT | M2_MAGIC,
        flags3=M3_WANTSARTI | M3_WAITFORU | M3_INFRAVISIBLE,
        color=CLR_BLACK,
        difficulty=20,
    ),

    # --- quest guardians ---

    # student
    MonsterEntry(
        name="student",
        symbol=MonsterSymbol.S_HUMAN,
        level=5, move_speed=12, ac=10, mr=10, alignment=3,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_GUARDIAN, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_TUNNEL | M1_NEEDPICK | M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=7,
    ),

    # chieftain
    MonsterEntry(
        name="chieftain",
        symbol=MonsterSymbol.S_HUMAN,
        level=5, move_speed=12, ac=10, mr=10, alignment=0,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_GUARDIAN, size=MZ_HUMAN,
        resists_mask=MR_POISON,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=7,
    ),

    # neanderthal
    MonsterEntry(
        name="neanderthal",
        symbol=MonsterSymbol.S_HUMAN,
        level=5, move_speed=12, ac=10, mr=10, alignment=1,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 4),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_GUARDIAN, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=7,
    ),

    # attendant
    MonsterEntry(
        name="attendant",
        symbol=MonsterSymbol.S_HUMAN,
        level=5, move_speed=12, ac=10, mr=10, alignment=3,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_GUARDIAN, size=MZ_HUMAN,
        resists_mask=MR_POISON,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=7,
    ),

    # page
    MonsterEntry(
        name="page",
        symbol=MonsterSymbol.S_HUMAN,
        level=5, move_speed=12, ac=10, mr=10, alignment=3,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_GUARDIAN, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=7,
    ),

    # abbot
    MonsterEntry(
        name="abbot",
        symbol=MonsterSymbol.S_HUMAN,
        level=5, move_speed=12, ac=10, mr=20, alignment=0,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 8, 2),
            (AttackType.AT_KICK, DamageType.AD_STUN, 3, 2),
            (AttackType.AT_MAGC, DamageType.AD_CLRC, 0, 0),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_GUARDIAN, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_HERBIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=8,
    ),

    # acolyte
    MonsterEntry(
        name="acolyte",
        symbol=MonsterSymbol.S_HUMAN,
        level=5, move_speed=12, ac=10, mr=20, alignment=0,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_MAGC, DamageType.AD_CLRC, 0, 0),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_GUARDIAN, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=8,
    ),

    # hunter
    MonsterEntry(
        name="hunter",
        symbol=MonsterSymbol.S_HUMAN,
        level=5, move_speed=12, ac=10, mr=10, alignment=-7,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 4),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_GUARDIAN, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_SEE_INVIS | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISION | M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=7,
    ),

    # thug
    MonsterEntry(
        name="thug",
        symbol=MonsterSymbol.S_HUMAN,
        level=5, move_speed=12, ac=10, mr=10, alignment=-3,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_GUARDIAN, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_GREEDY | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=7,
    ),

    # ninja
    MonsterEntry(
        name="ninja",
        symbol=MonsterSymbol.S_HUMAN,
        level=5, move_speed=12, ac=10, mr=10, alignment=3,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_HOSTILE | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=7,
    ),

    # roshi
    MonsterEntry(
        name="roshi",
        symbol=MonsterSymbol.S_HUMAN,
        level=5, move_speed=12, ac=10, mr=10, alignment=3,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_GUARDIAN, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=7,
    ),

    # guide
    MonsterEntry(
        name="guide",
        symbol=MonsterSymbol.S_HUMAN,
        level=5, move_speed=12, ac=10, mr=20, alignment=0,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_GUARDIAN, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_COLLECT | M2_MAGIC,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=8,
    ),

    # warrior
    MonsterEntry(
        name="warrior",
        symbol=MonsterSymbol.S_HUMAN,
        level=5, move_speed=12, ac=10, mr=10, alignment=-1,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_GUARDIAN, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_COLLECT | M2_FEMALE,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=7,
    ),

    # apprentice
    MonsterEntry(
        name="apprentice",
        symbol=MonsterSymbol.S_HUMAN,
        level=5, move_speed=12, ac=10, mr=30, alignment=0,
        generation_mask=G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
        ),
        weight=WT_HUMAN, nutrition=400, sound=MS_GUARDIAN, size=MZ_HUMAN,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_HUMAN | M2_PEACEFUL | M2_STRONG | M2_COLLECT | M2_MAGIC,
        flags3=M3_INFRAVISIBLE,
        color=HI_DOMESTIC,
        difficulty=8,
    ),
)
