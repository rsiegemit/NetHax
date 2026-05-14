"""
NetHack monster entries 131-195 (chunk 3).

Source: vendor/nethack/include/monsters.h lines 1276-1916
Covers: giant bat … gnome mummy
Excludes: all #if 0 / DEFERRED blocks (baby shimmering dragon, shimmering dragon,
          vorpal jabberwock).
"""

from Nethax.nethax.constants.monsters import (
    MonsterEntry,
    MonsterSymbol,
    AttackType,
    DamageType,
    # resistance bits
    MR_FIRE, MR_COLD, MR_SLEEP, MR_DISINT, MR_ELEC, MR_POISON, MR_ACID, MR_STONE,
    # generation mask bits
    G_GENO, G_NOGEN, G_NOCORPSE, G_SGROUP, G_LGROUP, G_HELL, G_NOHELL,
    # sound constants
    MS_SILENT, MS_ROAR, MS_SQEEK, MS_SQAWK, MS_HUMANOID, MS_MOO,
    MS_BURBLE, MS_SHRIEK, MS_ARREST, MS_SPELL, MS_BOAST, MS_MUMBLE,
    MS_GRUNT, MS_ORC,
    # size constants
    MZ_SMALL, MZ_LARGE, MZ_HUGE, MZ_GIGANTIC, MZ_HUMAN,
    # M1 flags
    M1_FLY, M1_SWIM, M1_AMPHIBIOUS, M1_BREATHLESS, M1_NOTAKE,
    M1_NOEYES, M1_NOHANDS, M1_NOLIMBS, M1_NOHEAD, M1_MINDLESS,
    M1_HUMANOID, M1_ANIMAL, M1_UNSOLID, M1_THICK_HIDE, M1_OVIPAROUS,
    M1_REGEN, M1_SEE_INVIS, M1_ACID, M1_POIS,
    M1_CARNIVORE, M1_OMNIVORE, M1_WALLWALK,
    # M2 flags
    M2_HOSTILE, M2_PEACEFUL, M2_WANDER, M2_STALK, M2_STRONG,
    M2_NASTY, M2_GREEDY, M2_JEWELS, M2_COLLECT, M2_MAGIC,
    M2_GIANT, M2_ROCKTHROW, M2_NEUTER, M2_GNOME, M2_HUMAN,
    M2_MALE, M2_FEMALE, M2_LORD, M2_PRINCE, M2_UNDEAD,
    M2_ORC, M2_DWARF, M2_NOPOLY,
    # M3 flags
    M3_INFRAVISIBLE, M3_INFRAVISION, M3_WANTSBOOK,
    # weight constants
    WT_ETHEREAL, WT_BABY_DRAGON, WT_DRAGON, WT_ELF, WT_HUMAN,
    # colour constants
    CLR_BLACK, CLR_RED, CLR_GREEN, CLR_BROWN, CLR_BLUE, CLR_MAGENTA,
    CLR_CYAN, CLR_GRAY, CLR_ORANGE, CLR_BRIGHT_GREEN, CLR_YELLOW,
    CLR_WHITE,
    HI_GOLD, HI_LORD, HI_ZAP, DRAGON_SILVER,
)
from typing import Tuple

NO_ATTK = (AttackType.AT_NONE, DamageType.AD_PHYS, 0, 0)

ENTRIES: Tuple[MonsterEntry, ...] = (

    # 131 — giant bat
    MonsterEntry(
        name="giant bat",
        symbol=MonsterSymbol.S_BAT,
        level=2, move_speed=22, ac=7, mr=0, alignment=0,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 6),
        ),
        weight=30, nutrition=30, sound=MS_SQEEK, size=MZ_SMALL,
        resists_mask=0, conveys_mask=0,
        flags1=M1_FLY | M1_ANIMAL | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_WANDER | M2_HOSTILE,
        flags3=M3_INFRAVISIBLE,
        color=CLR_RED,
        difficulty=3,
    ),

    # 132 — raven
    MonsterEntry(
        name="raven",
        symbol=MonsterSymbol.S_BAT,
        level=4, move_speed=20, ac=6, mr=0, alignment=0,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_CLAW, DamageType.AD_BLND, 1, 6),
        ),
        weight=40, nutrition=20, sound=MS_SQAWK, size=MZ_SMALL,
        resists_mask=0, conveys_mask=0,
        flags1=M1_FLY | M1_ANIMAL | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_WANDER | M2_HOSTILE,
        flags3=M3_INFRAVISIBLE,
        color=CLR_BLACK,
        difficulty=6,
    ),

    # 133 — vampire bat
    MonsterEntry(
        name="vampire bat",
        symbol=MonsterSymbol.S_BAT,
        level=5, move_speed=20, ac=6, mr=0, alignment=0,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_BITE, DamageType.AD_DRST, 0, 0),
        ),
        weight=30, nutrition=20, sound=MS_SQEEK, size=MZ_SMALL,
        resists_mask=MR_SLEEP | MR_POISON, conveys_mask=0,
        flags1=M1_FLY | M1_ANIMAL | M1_NOHANDS | M1_POIS | M1_REGEN | M1_OMNIVORE,
        flags2=M2_HOSTILE,
        flags3=M3_INFRAVISIBLE,
        color=CLR_BLACK,
        difficulty=7,
    ),

    # 134 — plains centaur
    MonsterEntry(
        name="plains centaur",
        symbol=MonsterSymbol.S_CENTAUR,
        level=4, move_speed=18, ac=4, mr=0, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_KICK, DamageType.AD_PHYS, 1, 6),
        ),
        weight=2500, nutrition=500, sound=MS_HUMANOID, size=MZ_LARGE,
        resists_mask=0, conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_STRONG | M2_GREEDY | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=CLR_BROWN,
        difficulty=6,
    ),

    # 135 — forest centaur
    MonsterEntry(
        name="forest centaur",
        symbol=MonsterSymbol.S_CENTAUR,
        level=5, move_speed=18, ac=3, mr=10, alignment=-1,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
            (AttackType.AT_KICK, DamageType.AD_PHYS, 1, 6),
        ),
        weight=2550, nutrition=600, sound=MS_HUMANOID, size=MZ_LARGE,
        resists_mask=0, conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_STRONG | M2_GREEDY | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=CLR_GREEN,
        difficulty=8,
    ),

    # 136 — mountain centaur
    MonsterEntry(
        name="mountain centaur",
        symbol=MonsterSymbol.S_CENTAUR,
        level=6, move_speed=20, ac=2, mr=10, alignment=-3,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 10),
            (AttackType.AT_KICK, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_KICK, DamageType.AD_PHYS, 1, 6),
        ),
        weight=2550, nutrition=500, sound=MS_HUMANOID, size=MZ_LARGE,
        resists_mask=0, conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_STRONG | M2_GREEDY | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=CLR_CYAN,
        difficulty=9,
    ),

    # 137 — baby gray dragon
    MonsterEntry(
        name="baby gray dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=12, move_speed=9, ac=2, mr=10, alignment=0,
        generation_mask=G_GENO,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=WT_BABY_DRAGON, nutrition=500, sound=MS_ROAR, size=MZ_HUGE,
        resists_mask=0, conveys_mask=0,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_GREEDY | M2_JEWELS,
        flags3=0,
        color=CLR_GRAY,
        difficulty=13,
    ),

    # (vendor parity: baby gold dragon not present in NLE — removed)

    # 139 — baby silver dragon
    MonsterEntry(
        name="baby silver dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=12, move_speed=9, ac=2, mr=10, alignment=0,
        generation_mask=G_GENO,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=WT_BABY_DRAGON, nutrition=500, sound=MS_ROAR, size=MZ_HUGE,
        resists_mask=0, conveys_mask=0,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_GREEDY | M2_JEWELS,
        flags3=0,
        color=DRAGON_SILVER,
        difficulty=13,
    ),

    # 140 — baby red dragon
    MonsterEntry(
        name="baby red dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=12, move_speed=9, ac=2, mr=10, alignment=0,
        generation_mask=G_GENO,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=WT_BABY_DRAGON, nutrition=500, sound=MS_ROAR, size=MZ_HUGE,
        resists_mask=MR_FIRE, conveys_mask=0,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_GREEDY | M2_JEWELS,
        flags3=M3_INFRAVISIBLE,
        color=CLR_RED,
        difficulty=13,
    ),

    # 141 — baby white dragon
    MonsterEntry(
        name="baby white dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=12, move_speed=9, ac=2, mr=10, alignment=0,
        generation_mask=G_GENO,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=WT_BABY_DRAGON, nutrition=500, sound=MS_ROAR, size=MZ_HUGE,
        resists_mask=MR_COLD, conveys_mask=0,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_GREEDY | M2_JEWELS,
        flags3=0,
        color=CLR_WHITE,
        difficulty=13,
    ),

    # 142 — baby orange dragon
    MonsterEntry(
        name="baby orange dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=12, move_speed=9, ac=2, mr=10, alignment=0,
        generation_mask=G_GENO,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=WT_BABY_DRAGON, nutrition=500, sound=MS_ROAR, size=MZ_HUGE,
        resists_mask=MR_SLEEP, conveys_mask=0,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_GREEDY | M2_JEWELS,
        flags3=0,
        color=CLR_ORANGE,
        difficulty=13,
    ),

    # 143 — baby black dragon
    MonsterEntry(
        name="baby black dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=12, move_speed=9, ac=2, mr=10, alignment=0,
        generation_mask=G_GENO,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=WT_BABY_DRAGON, nutrition=500, sound=MS_ROAR, size=MZ_HUGE,
        resists_mask=MR_DISINT, conveys_mask=0,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_GREEDY | M2_JEWELS,
        flags3=0,
        color=CLR_BLACK,
        difficulty=13,
    ),

    # 144 — baby blue dragon
    MonsterEntry(
        name="baby blue dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=12, move_speed=9, ac=2, mr=10, alignment=0,
        generation_mask=G_GENO,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=WT_BABY_DRAGON, nutrition=500, sound=MS_ROAR, size=MZ_HUGE,
        resists_mask=MR_ELEC, conveys_mask=0,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_GREEDY | M2_JEWELS,
        flags3=0,
        color=CLR_BLUE,
        difficulty=13,
    ),

    # 145 — baby green dragon
    MonsterEntry(
        name="baby green dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=12, move_speed=9, ac=2, mr=10, alignment=0,
        generation_mask=G_GENO,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=WT_BABY_DRAGON, nutrition=500, sound=MS_ROAR, size=MZ_HUGE,
        resists_mask=MR_POISON, conveys_mask=0,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_CARNIVORE | M1_POIS,
        flags2=M2_HOSTILE | M2_STRONG | M2_GREEDY | M2_JEWELS,
        flags3=0,
        color=CLR_GREEN,
        difficulty=13,
    ),

    # 146 — baby yellow dragon
    MonsterEntry(
        name="baby yellow dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=12, move_speed=9, ac=2, mr=10, alignment=0,
        generation_mask=G_GENO,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=WT_BABY_DRAGON, nutrition=500, sound=MS_ROAR, size=MZ_HUGE,
        resists_mask=MR_ACID | MR_STONE, conveys_mask=0,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_CARNIVORE | M1_ACID,
        flags2=M2_HOSTILE | M2_STRONG | M2_GREEDY | M2_JEWELS,
        flags3=0,
        color=CLR_YELLOW,
        difficulty=13,
    ),

    # 147 — gray dragon
    MonsterEntry(
        name="gray dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=15, move_speed=9, ac=-1, mr=20, alignment=4,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BREA, DamageType.AD_MAGM, 4, 6),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 3, 8),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
        ),
        weight=WT_DRAGON, nutrition=1500, sound=MS_ROAR, size=MZ_GIGANTIC,
        resists_mask=0, conveys_mask=0,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_SEE_INVIS | M1_OVIPAROUS | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_NASTY | M2_GREEDY | M2_JEWELS | M2_MAGIC,
        flags3=0,
        color=CLR_GRAY,
        difficulty=20,
    ),

    # (vendor parity: gold dragon not present in NLE — removed)

    # 149 — silver dragon
    MonsterEntry(
        name="silver dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=15, move_speed=9, ac=-1, mr=20, alignment=4,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BREA, DamageType.AD_COLD, 4, 6),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 3, 8),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
        ),
        weight=WT_DRAGON, nutrition=1500, sound=MS_ROAR, size=MZ_GIGANTIC,
        resists_mask=MR_COLD, conveys_mask=0,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_SEE_INVIS | M1_OVIPAROUS | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_NASTY | M2_GREEDY | M2_JEWELS | M2_MAGIC,
        flags3=0,
        color=DRAGON_SILVER,
        difficulty=20,
    ),

    # 150 — red dragon
    MonsterEntry(
        name="red dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=15, move_speed=9, ac=-1, mr=20, alignment=-4,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BREA, DamageType.AD_FIRE, 6, 6),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 3, 8),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
        ),
        weight=WT_DRAGON, nutrition=1500, sound=MS_ROAR, size=MZ_GIGANTIC,
        resists_mask=MR_FIRE, conveys_mask=MR_FIRE,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_SEE_INVIS | M1_OVIPAROUS | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_NASTY | M2_GREEDY | M2_JEWELS | M2_MAGIC,
        flags3=M3_INFRAVISIBLE,
        color=CLR_RED,
        difficulty=20,
    ),

    # 151 — white dragon
    MonsterEntry(
        name="white dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=15, move_speed=9, ac=-1, mr=20, alignment=-5,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BREA, DamageType.AD_COLD, 4, 6),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 3, 8),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
        ),
        weight=WT_DRAGON, nutrition=1500, sound=MS_ROAR, size=MZ_GIGANTIC,
        resists_mask=MR_COLD, conveys_mask=MR_COLD,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_SEE_INVIS | M1_OVIPAROUS | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_NASTY | M2_GREEDY | M2_JEWELS | M2_MAGIC,
        flags3=0,
        color=CLR_WHITE,
        difficulty=20,
    ),

    # 152 — orange dragon
    MonsterEntry(
        name="orange dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=15, move_speed=9, ac=-1, mr=20, alignment=5,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BREA, DamageType.AD_SLEE, 4, 25),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 3, 8),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
        ),
        weight=WT_DRAGON, nutrition=1500, sound=MS_ROAR, size=MZ_GIGANTIC,
        resists_mask=MR_SLEEP, conveys_mask=MR_SLEEP,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_SEE_INVIS | M1_OVIPAROUS | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_NASTY | M2_GREEDY | M2_JEWELS | M2_MAGIC,
        flags3=0,
        color=CLR_ORANGE,
        difficulty=20,
    ),

    # 153 — black dragon
    MonsterEntry(
        name="black dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=15, move_speed=9, ac=-1, mr=20, alignment=-6,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BREA, DamageType.AD_DISN, 1, 255),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 3, 8),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
        ),
        weight=WT_DRAGON, nutrition=1500, sound=MS_ROAR, size=MZ_GIGANTIC,
        resists_mask=MR_DISINT, conveys_mask=MR_DISINT,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_SEE_INVIS | M1_OVIPAROUS | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_NASTY | M2_GREEDY | M2_JEWELS | M2_MAGIC,
        flags3=0,
        color=CLR_BLACK,
        difficulty=20,
    ),

    # 154 — blue dragon
    MonsterEntry(
        name="blue dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=15, move_speed=9, ac=-1, mr=20, alignment=-7,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BREA, DamageType.AD_ELEC, 4, 6),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 3, 8),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
        ),
        weight=WT_DRAGON, nutrition=1500, sound=MS_ROAR, size=MZ_GIGANTIC,
        resists_mask=MR_ELEC, conveys_mask=MR_ELEC,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_SEE_INVIS | M1_OVIPAROUS | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_NASTY | M2_GREEDY | M2_JEWELS | M2_MAGIC,
        flags3=0,
        color=CLR_BLUE,
        difficulty=20,
    ),

    # 155 — green dragon
    MonsterEntry(
        name="green dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=15, move_speed=9, ac=-1, mr=20, alignment=6,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BREA, DamageType.AD_DRST, 4, 6),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 3, 8),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
        ),
        weight=WT_DRAGON, nutrition=1500, sound=MS_ROAR, size=MZ_GIGANTIC,
        resists_mask=MR_POISON, conveys_mask=MR_POISON,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_SEE_INVIS | M1_OVIPAROUS | M1_CARNIVORE | M1_POIS,
        flags2=M2_HOSTILE | M2_STRONG | M2_NASTY | M2_GREEDY | M2_JEWELS | M2_MAGIC,
        flags3=0,
        color=CLR_GREEN,
        difficulty=20,
    ),

    # 156 — yellow dragon
    MonsterEntry(
        name="yellow dragon",
        symbol=MonsterSymbol.S_DRAGON,
        level=15, move_speed=9, ac=-1, mr=20, alignment=7,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BREA, DamageType.AD_ACID, 4, 6),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 3, 8),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
        ),
        weight=WT_DRAGON, nutrition=1500, sound=MS_ROAR, size=MZ_GIGANTIC,
        resists_mask=MR_ACID | MR_STONE, conveys_mask=MR_STONE,
        flags1=M1_FLY | M1_THICK_HIDE | M1_NOHANDS | M1_SEE_INVIS | M1_OVIPAROUS | M1_CARNIVORE | M1_ACID,
        flags2=M2_HOSTILE | M2_STRONG | M2_NASTY | M2_GREEDY | M2_JEWELS | M2_MAGIC,
        flags3=0,
        color=CLR_YELLOW,
        difficulty=20,
    ),

    # 157 — stalker
    MonsterEntry(
        name="stalker",
        symbol=MonsterSymbol.S_ELEMENTAL,
        level=8, move_speed=12, ac=3, mr=0, alignment=0,
        generation_mask=G_GENO | 3,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 4, 4),
        ),
        weight=900, nutrition=400, sound=MS_SILENT, size=MZ_LARGE,
        resists_mask=0, conveys_mask=0,
        flags1=M1_ANIMAL | M1_FLY | M1_SEE_INVIS,
        flags2=M2_WANDER | M2_STALK | M2_HOSTILE | M2_STRONG,
        flags3=M3_INFRAVISION,
        color=CLR_WHITE,
        difficulty=9,
    ),

    # 158 — air elemental
    MonsterEntry(
        name="air elemental",
        symbol=MonsterSymbol.S_ELEMENTAL,
        level=8, move_speed=36, ac=2, mr=30, alignment=0,
        generation_mask=G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_ENGL, DamageType.AD_PHYS, 1, 10),
        ),
        weight=WT_ETHEREAL, nutrition=0, sound=MS_SILENT, size=MZ_HUGE,
        resists_mask=MR_POISON | MR_STONE, conveys_mask=0,
        flags1=M1_NOEYES | M1_NOLIMBS | M1_NOHEAD | M1_MINDLESS | M1_BREATHLESS | M1_UNSOLID | M1_FLY,
        flags2=M2_STRONG | M2_NEUTER,
        flags3=0,
        color=CLR_CYAN,
        difficulty=10,
    ),

    # 159 — fire elemental
    MonsterEntry(
        name="fire elemental",
        symbol=MonsterSymbol.S_ELEMENTAL,
        level=8, move_speed=12, ac=2, mr=30, alignment=0,
        generation_mask=G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_FIRE, 3, 6),
            (AttackType.AT_NONE, DamageType.AD_FIRE, 0, 4),
        ),
        weight=WT_ETHEREAL, nutrition=0, sound=MS_SILENT, size=MZ_HUGE,
        resists_mask=MR_FIRE | MR_POISON | MR_STONE, conveys_mask=0,
        flags1=M1_NOEYES | M1_NOLIMBS | M1_NOHEAD | M1_MINDLESS | M1_BREATHLESS | M1_UNSOLID | M1_FLY | M1_NOTAKE,
        flags2=M2_STRONG | M2_NEUTER,
        flags3=M3_INFRAVISIBLE,
        color=CLR_YELLOW,
        difficulty=10,
    ),

    # 160 — earth elemental
    MonsterEntry(
        name="earth elemental",
        symbol=MonsterSymbol.S_ELEMENTAL,
        level=8, move_speed=6, ac=2, mr=30, alignment=0,
        generation_mask=G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 4, 6),
        ),
        weight=2500, nutrition=0, sound=MS_SILENT, size=MZ_HUGE,
        resists_mask=MR_FIRE | MR_COLD | MR_POISON | MR_STONE, conveys_mask=0,
        flags1=M1_NOEYES | M1_NOLIMBS | M1_NOHEAD | M1_MINDLESS | M1_BREATHLESS | M1_WALLWALK | M1_THICK_HIDE,
        flags2=M2_STRONG | M2_NEUTER,
        flags3=0,
        color=CLR_BROWN,
        difficulty=10,
    ),

    # 161 — water elemental
    MonsterEntry(
        name="water elemental",
        symbol=MonsterSymbol.S_ELEMENTAL,
        level=8, move_speed=6, ac=2, mr=30, alignment=0,
        generation_mask=G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 5, 6),
        ),
        weight=2500, nutrition=0, sound=MS_SILENT, size=MZ_HUGE,
        resists_mask=MR_POISON | MR_STONE, conveys_mask=0,
        flags1=M1_NOEYES | M1_NOLIMBS | M1_NOHEAD | M1_MINDLESS | M1_BREATHLESS | M1_UNSOLID | M1_AMPHIBIOUS | M1_SWIM,
        flags2=M2_STRONG | M2_NEUTER,
        flags3=0,
        color=CLR_BLUE,
        difficulty=10,
    ),

    # 162 — lichen
    MonsterEntry(
        name="lichen",
        symbol=MonsterSymbol.S_FUNGUS,
        level=0, move_speed=1, ac=9, mr=0, alignment=0,
        generation_mask=G_GENO | 4,
        attacks=(
            (AttackType.AT_TUCH, DamageType.AD_STCK, 0, 0),
        ),
        weight=20, nutrition=200, sound=MS_SILENT, size=MZ_SMALL,
        resists_mask=0, conveys_mask=0,
        flags1=M1_BREATHLESS | M1_NOEYES | M1_NOLIMBS | M1_NOHEAD | M1_MINDLESS | M1_NOTAKE,
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=0,
        color=CLR_BRIGHT_GREEN,
        difficulty=1,
    ),

    # 163 — brown mold
    MonsterEntry(
        name="brown mold",
        symbol=MonsterSymbol.S_FUNGUS,
        level=1, move_speed=0, ac=9, mr=0, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_NONE, DamageType.AD_COLD, 0, 6),
        ),
        weight=50, nutrition=30, sound=MS_SILENT, size=MZ_SMALL,
        resists_mask=MR_COLD | MR_POISON, conveys_mask=MR_COLD | MR_POISON,
        flags1=M1_BREATHLESS | M1_NOEYES | M1_NOLIMBS | M1_NOHEAD | M1_MINDLESS | M1_NOTAKE,
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=0,
        color=CLR_BROWN,
        difficulty=2,
    ),

    # 164 — yellow mold
    MonsterEntry(
        name="yellow mold",
        symbol=MonsterSymbol.S_FUNGUS,
        level=1, move_speed=0, ac=9, mr=0, alignment=0,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_NONE, DamageType.AD_STUN, 0, 4),
        ),
        weight=50, nutrition=30, sound=MS_SILENT, size=MZ_SMALL,
        resists_mask=MR_POISON, conveys_mask=MR_POISON,
        flags1=M1_BREATHLESS | M1_NOEYES | M1_NOLIMBS | M1_NOHEAD | M1_MINDLESS | M1_POIS | M1_NOTAKE,
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=0,
        color=CLR_YELLOW,
        difficulty=2,
    ),

    # 165 — green mold
    MonsterEntry(
        name="green mold",
        symbol=MonsterSymbol.S_FUNGUS,
        level=1, move_speed=0, ac=9, mr=0, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_NONE, DamageType.AD_ACID, 0, 4),
        ),
        weight=50, nutrition=30, sound=MS_SILENT, size=MZ_SMALL,
        resists_mask=MR_ACID | MR_STONE, conveys_mask=MR_STONE,
        flags1=M1_BREATHLESS | M1_NOEYES | M1_NOLIMBS | M1_NOHEAD | M1_MINDLESS | M1_ACID | M1_NOTAKE,
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=0,
        color=CLR_GREEN,
        difficulty=2,
    ),

    # 166 — red mold
    MonsterEntry(
        name="red mold",
        symbol=MonsterSymbol.S_FUNGUS,
        level=1, move_speed=0, ac=9, mr=0, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_NONE, DamageType.AD_FIRE, 0, 4),
        ),
        weight=50, nutrition=30, sound=MS_SILENT, size=MZ_SMALL,
        resists_mask=MR_FIRE | MR_POISON, conveys_mask=MR_FIRE | MR_POISON,
        flags1=M1_BREATHLESS | M1_NOEYES | M1_NOLIMBS | M1_NOHEAD | M1_MINDLESS | M1_NOTAKE,
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=M3_INFRAVISIBLE,
        color=CLR_RED,
        difficulty=2,
    ),

    # 167 — shrieker
    MonsterEntry(
        name="shrieker",
        symbol=MonsterSymbol.S_FUNGUS,
        level=3, move_speed=1, ac=7, mr=0, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(),
        weight=100, nutrition=100, sound=MS_SHRIEK, size=MZ_SMALL,
        resists_mask=MR_POISON, conveys_mask=MR_POISON,
        flags1=M1_BREATHLESS | M1_NOEYES | M1_NOLIMBS | M1_NOHEAD | M1_MINDLESS | M1_NOTAKE,
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=0,
        color=CLR_MAGENTA,
        difficulty=2,
    ),

    # 168 — violet fungus
    MonsterEntry(
        name="violet fungus",
        symbol=MonsterSymbol.S_FUNGUS,
        level=3, move_speed=1, ac=7, mr=0, alignment=0,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_TUCH, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_TUCH, DamageType.AD_STCK, 0, 0),
        ),
        weight=100, nutrition=100, sound=MS_SILENT, size=MZ_SMALL,
        resists_mask=MR_POISON, conveys_mask=MR_POISON,
        flags1=M1_BREATHLESS | M1_NOEYES | M1_NOLIMBS | M1_NOHEAD | M1_MINDLESS | M1_NOTAKE,
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=0,
        color=CLR_MAGENTA,
        difficulty=5,
    ),

    # 169 — gnome
    MonsterEntry(
        name="gnome",
        symbol=MonsterSymbol.S_GNOME,
        level=1, move_speed=6, ac=10, mr=4, alignment=0,
        generation_mask=G_GENO | G_SGROUP | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=650, nutrition=100, sound=MS_ORC, size=MZ_SMALL,
        resists_mask=0, conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_NOPOLY | M2_GNOME | M2_COLLECT,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_BROWN,
        difficulty=3,
    ),

    # 170 — gnome lord  (NAMS: "gnome lord"/"gnome lady"/"gnome leader")
    MonsterEntry(
        name="gnome lord",
        symbol=MonsterSymbol.S_GNOME,
        level=3, move_speed=8, ac=10, mr=4, alignment=0,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
        ),
        weight=700, nutrition=120, sound=MS_ORC, size=MZ_SMALL,
        resists_mask=0, conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_GNOME | M2_LORD | M2_MALE | M2_COLLECT,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_BLUE,
        difficulty=4,
    ),

    # 171 — gnomish wizard
    MonsterEntry(
        name="gnomish wizard",
        symbol=MonsterSymbol.S_GNOME,
        level=3, move_speed=10, ac=4, mr=10, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
        ),
        weight=700, nutrition=120, sound=MS_ORC, size=MZ_SMALL,
        resists_mask=0, conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_GNOME | M2_MAGIC,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=HI_ZAP,
        difficulty=5,
    ),

    # 172 — gnome king  (NAMS: "gnome king"/"gnome queen"/"gnome ruler")
    MonsterEntry(
        name="gnome king",
        symbol=MonsterSymbol.S_GNOME,
        level=5, move_speed=10, ac=10, mr=20, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 6),
        ),
        weight=750, nutrition=150, sound=MS_ORC, size=MZ_SMALL,
        resists_mask=0, conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_GNOME | M2_PRINCE | M2_MALE | M2_COLLECT,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=HI_LORD,
        difficulty=6,
    ),

    # 173 — giant  (placeholder for zombie/mummy corpses)
    MonsterEntry(
        name="giant",
        symbol=MonsterSymbol.S_GIANT,
        level=6, move_speed=6, ac=0, mr=0, alignment=2,
        generation_mask=G_GENO | G_NOGEN | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 10),
        ),
        weight=2250, nutrition=750, sound=MS_BOAST, size=MZ_HUGE,
        resists_mask=0, conveys_mask=0,
        flags1=M1_HUMANOID | M1_CARNIVORE,
        flags2=M2_GIANT | M2_STRONG | M2_ROCKTHROW | M2_NASTY | M2_COLLECT | M2_JEWELS,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_RED,
        difficulty=8,
    ),

    # 174 — stone giant
    MonsterEntry(
        name="stone giant",
        symbol=MonsterSymbol.S_GIANT,
        level=6, move_speed=6, ac=0, mr=0, alignment=2,
        generation_mask=G_GENO | G_SGROUP | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 10),
        ),
        weight=2250, nutrition=750, sound=MS_BOAST, size=MZ_HUGE,
        resists_mask=0, conveys_mask=0,
        flags1=M1_HUMANOID | M1_CARNIVORE,
        flags2=M2_GIANT | M2_STRONG | M2_ROCKTHROW | M2_NASTY | M2_COLLECT | M2_JEWELS,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_GRAY,
        difficulty=8,
    ),

    # 175 — hill giant
    MonsterEntry(
        name="hill giant",
        symbol=MonsterSymbol.S_GIANT,
        level=8, move_speed=10, ac=6, mr=0, alignment=-2,
        generation_mask=G_GENO | G_SGROUP | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 8),
        ),
        weight=2200, nutrition=700, sound=MS_BOAST, size=MZ_HUGE,
        resists_mask=0, conveys_mask=0,
        flags1=M1_HUMANOID | M1_CARNIVORE,
        flags2=M2_GIANT | M2_STRONG | M2_ROCKTHROW | M2_NASTY | M2_COLLECT | M2_JEWELS,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_CYAN,
        difficulty=10,
    ),

    # 176 — fire giant
    MonsterEntry(
        name="fire giant",
        symbol=MonsterSymbol.S_GIANT,
        level=9, move_speed=12, ac=4, mr=5, alignment=2,
        generation_mask=G_GENO | G_SGROUP | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 10),
        ),
        weight=2250, nutrition=750, sound=MS_BOAST, size=MZ_HUGE,
        resists_mask=MR_FIRE, conveys_mask=MR_FIRE,
        flags1=M1_HUMANOID | M1_CARNIVORE,
        flags2=M2_GIANT | M2_STRONG | M2_ROCKTHROW | M2_NASTY | M2_COLLECT | M2_JEWELS,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_YELLOW,
        difficulty=11,
    ),

    # 177 — frost giant
    MonsterEntry(
        name="frost giant",
        symbol=MonsterSymbol.S_GIANT,
        level=10, move_speed=12, ac=3, mr=10, alignment=-3,
        generation_mask=G_NOHELL | G_GENO | G_SGROUP | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 12),
        ),
        weight=2250, nutrition=750, sound=MS_BOAST, size=MZ_HUGE,
        resists_mask=MR_COLD, conveys_mask=MR_COLD,
        flags1=M1_HUMANOID | M1_CARNIVORE,
        flags2=M2_GIANT | M2_STRONG | M2_ROCKTHROW | M2_NASTY | M2_COLLECT | M2_JEWELS,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_WHITE,
        difficulty=13,
    ),

    # 178 — ettin
    MonsterEntry(
        name="ettin",
        symbol=MonsterSymbol.S_GIANT,
        level=10, move_speed=12, ac=3, mr=0, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 8),
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 3, 6),
        ),
        weight=1700, nutrition=500, sound=MS_GRUNT, size=MZ_HUGE,
        resists_mask=0, conveys_mask=0,
        flags1=M1_ANIMAL | M1_HUMANOID | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_NASTY | M2_COLLECT,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_BROWN,
        difficulty=13,
    ),

    # 179 — storm giant
    MonsterEntry(
        name="storm giant",
        symbol=MonsterSymbol.S_GIANT,
        level=16, move_speed=12, ac=3, mr=10, alignment=-3,
        generation_mask=G_GENO | G_SGROUP | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 12),
        ),
        weight=2250, nutrition=750, sound=MS_BOAST, size=MZ_HUGE,
        resists_mask=MR_ELEC, conveys_mask=MR_ELEC,
        flags1=M1_HUMANOID | M1_CARNIVORE,
        flags2=M2_GIANT | M2_STRONG | M2_ROCKTHROW | M2_NASTY | M2_COLLECT | M2_JEWELS,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_BLUE,
        difficulty=19,
    ),

    # 180 — titan
    MonsterEntry(
        name="titan",
        symbol=MonsterSymbol.S_GIANT,
        level=16, move_speed=18, ac=-3, mr=70, alignment=9,
        generation_mask=1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 8),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
        ),
        weight=2300, nutrition=900, sound=MS_SPELL, size=MZ_HUGE,
        resists_mask=0, conveys_mask=0,
        flags1=M1_FLY | M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_STRONG | M2_ROCKTHROW | M2_NASTY | M2_COLLECT | M2_MAGIC,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_MAGENTA,
        difficulty=20,
    ),

    # 181 — minotaur
    MonsterEntry(
        name="minotaur",
        symbol=MonsterSymbol.S_GIANT,
        level=15, move_speed=15, ac=6, mr=0, alignment=0,
        generation_mask=G_GENO | G_NOGEN,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 3, 10),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 3, 10),
            (AttackType.AT_BUTT, DamageType.AD_PHYS, 2, 8),
        ),
        weight=1500, nutrition=700, sound=MS_SILENT, size=MZ_LARGE,
        resists_mask=0, conveys_mask=0,
        flags1=M1_ANIMAL | M1_HUMANOID | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_NASTY,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_BROWN,
        difficulty=17,
    ),

    # 182 — jabberwock
    MonsterEntry(
        name="jabberwock",
        symbol=MonsterSymbol.S_JABBERWOCK,
        level=15, move_speed=12, ac=-2, mr=50, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 10),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 10),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 10),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 10),
        ),
        weight=1300, nutrition=600, sound=MS_BURBLE, size=MZ_LARGE,
        resists_mask=0, conveys_mask=0,
        flags1=M1_ANIMAL | M1_FLY | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_NASTY | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=CLR_ORANGE,
        difficulty=18,
    ),

    # 183 — Keystone Kop
    MonsterEntry(
        name="Keystone Kop",
        symbol=MonsterSymbol.S_KOP,
        level=1, move_speed=6, ac=10, mr=10, alignment=9,
        generation_mask=G_GENO | G_LGROUP | G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 4),
        ),
        weight=WT_HUMAN, nutrition=200, sound=MS_ARREST, size=MZ_HUMAN,
        resists_mask=0, conveys_mask=0,
        flags1=M1_HUMANOID,
        flags2=M2_HUMAN | M2_WANDER | M2_HOSTILE | M2_MALE | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=CLR_BLUE,
        difficulty=3,
    ),

    # 184 — Kop Sergeant
    MonsterEntry(
        name="Kop Sergeant",
        symbol=MonsterSymbol.S_KOP,
        level=2, move_speed=8, ac=10, mr=10, alignment=10,
        generation_mask=G_GENO | G_SGROUP | G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=200, sound=MS_ARREST, size=MZ_HUMAN,
        resists_mask=0, conveys_mask=0,
        flags1=M1_HUMANOID,
        flags2=M2_HUMAN | M2_WANDER | M2_HOSTILE | M2_STRONG | M2_MALE | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=CLR_BLUE,
        difficulty=4,
    ),

    # 185 — Kop Lieutenant
    MonsterEntry(
        name="Kop Lieutenant",
        symbol=MonsterSymbol.S_KOP,
        level=3, move_speed=10, ac=10, mr=20, alignment=11,
        generation_mask=G_GENO | G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 1, 8),
        ),
        weight=WT_HUMAN, nutrition=200, sound=MS_ARREST, size=MZ_HUMAN,
        resists_mask=0, conveys_mask=0,
        flags1=M1_HUMANOID,
        flags2=M2_HUMAN | M2_WANDER | M2_HOSTILE | M2_STRONG | M2_MALE | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=CLR_CYAN,
        difficulty=5,
    ),

    # 186 — Kop Kaptain
    MonsterEntry(
        name="Kop Kaptain",
        symbol=MonsterSymbol.S_KOP,
        level=4, move_speed=12, ac=10, mr=20, alignment=12,
        generation_mask=G_GENO | G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 6),
        ),
        weight=WT_HUMAN, nutrition=200, sound=MS_ARREST, size=MZ_HUMAN,
        resists_mask=0, conveys_mask=0,
        flags1=M1_HUMANOID,
        flags2=M2_HUMAN | M2_WANDER | M2_HOSTILE | M2_STRONG | M2_MALE | M2_COLLECT,
        flags3=M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=6,
    ),

    # 187 — lich
    MonsterEntry(
        name="lich",
        symbol=MonsterSymbol.S_LICH,
        level=11, move_speed=6, ac=0, mr=30, alignment=-9,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_TUCH, DamageType.AD_COLD, 1, 10),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
        ),
        weight=1200, nutrition=100, sound=MS_MUMBLE, size=MZ_HUMAN,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON, conveys_mask=MR_COLD,
        flags1=M1_BREATHLESS | M1_HUMANOID | M1_POIS | M1_REGEN,
        flags2=M2_UNDEAD | M2_HOSTILE | M2_MAGIC,
        flags3=M3_INFRAVISION,
        color=CLR_BROWN,
        difficulty=14,
    ),

    # 188 — demilich
    MonsterEntry(
        name="demilich",
        symbol=MonsterSymbol.S_LICH,
        level=14, move_speed=9, ac=-2, mr=60, alignment=-12,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_TUCH, DamageType.AD_COLD, 3, 4),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
        ),
        weight=1200, nutrition=100, sound=MS_MUMBLE, size=MZ_HUMAN,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON, conveys_mask=MR_COLD,
        flags1=M1_BREATHLESS | M1_HUMANOID | M1_POIS | M1_REGEN,
        flags2=M2_UNDEAD | M2_HOSTILE | M2_MAGIC,
        flags3=M3_INFRAVISION,
        color=CLR_RED,
        difficulty=18,
    ),

    # 189 — master lich
    MonsterEntry(
        name="master lich",
        symbol=MonsterSymbol.S_LICH,
        level=17, move_speed=9, ac=-4, mr=90, alignment=-15,
        generation_mask=G_HELL | G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_TUCH, DamageType.AD_COLD, 3, 6),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
        ),
        weight=1200, nutrition=100, sound=MS_MUMBLE, size=MZ_HUMAN,
        resists_mask=MR_FIRE | MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=MR_FIRE | MR_COLD,
        flags1=M1_BREATHLESS | M1_HUMANOID | M1_POIS | M1_REGEN,
        flags2=M2_UNDEAD | M2_HOSTILE | M2_MAGIC,
        flags3=M3_WANTSBOOK | M3_INFRAVISION,
        color=HI_LORD,
        difficulty=21,
    ),

    # 190 — arch-lich
    MonsterEntry(
        name="arch-lich",
        symbol=MonsterSymbol.S_LICH,
        level=25, move_speed=9, ac=-6, mr=90, alignment=-15,
        generation_mask=G_HELL | G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_TUCH, DamageType.AD_COLD, 5, 6),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
        ),
        weight=1200, nutrition=100, sound=MS_MUMBLE, size=MZ_HUMAN,
        resists_mask=MR_FIRE | MR_COLD | MR_SLEEP | MR_ELEC | MR_POISON,
        conveys_mask=MR_FIRE | MR_COLD,
        flags1=M1_BREATHLESS | M1_HUMANOID | M1_POIS | M1_REGEN,
        flags2=M2_UNDEAD | M2_HOSTILE | M2_MAGIC,
        flags3=M3_WANTSBOOK | M3_INFRAVISION,
        color=HI_LORD,
        difficulty=29,
    ),

    # 191 — kobold mummy
    MonsterEntry(
        name="kobold mummy",
        symbol=MonsterSymbol.S_MUMMY,
        level=3, move_speed=8, ac=6, mr=20, alignment=-2,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
        ),
        weight=400, nutrition=50, sound=MS_SILENT, size=MZ_SMALL,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON, conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_POIS,
        flags2=M2_UNDEAD | M2_HOSTILE,
        flags3=M3_INFRAVISION,
        color=CLR_BROWN,
        difficulty=4,
    ),

    # 192 — gnome mummy
    MonsterEntry(
        name="gnome mummy",
        symbol=MonsterSymbol.S_MUMMY,
        level=4, move_speed=10, ac=6, mr=20, alignment=-3,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 6),
        ),
        weight=650, nutrition=50, sound=MS_SILENT, size=MZ_SMALL,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON, conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_POIS,
        flags2=M2_UNDEAD | M2_HOSTILE | M2_GNOME,
        flags3=M3_INFRAVISION,
        color=CLR_RED,
        difficulty=5,
    ),

)
