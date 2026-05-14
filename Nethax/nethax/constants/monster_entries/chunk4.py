"""Monster entries chunk 4 — vendor/nethack/include/monsters.h lines 1917-2537.

Sections: Mummies, Nagas, Ogres, Puddings, Quantum mechanics, Rust monsters,
          Snakes, Trolls, Umber hulk, Vampires, Wraiths, Xorn, Apelike beasts,
          Zombies, Golems (straw through gold).

Note: vampire mage (#if 0 / DEFERRED block, lines 2301-2312) is excluded.
Actual entry count: 64 active MON() entries.
"""

from Nethax.nethax.constants.monsters import (
    MonsterEntry,
    MonsterSymbol,
    AttackType,
    DamageType,
    NO_ATTK,
    # MR_* resistance bits
    MR_FIRE, MR_COLD, MR_SLEEP, MR_POISON, MR_ACID, MR_STONE, MR_ELEC,
    # G_* generation mask bits
    G_GENO, G_NOCORPSE, G_NOGEN, G_SGROUP, G_LGROUP, G_HELL, G_NOHELL,
    G_UNIQ,
    # MS_* sound constants
    MS_SILENT, MS_MUMBLE, MS_GRUNT, MS_HISS, MS_HUMANOID, MS_GROWL, MS_ROAR,
    MS_VAMPIRE, MS_SPELL, MS_GROAN, MS_BONES,
    # MZ_* size constants
    MZ_TINY, MZ_SMALL, MZ_MEDIUM, MZ_HUMAN, MZ_LARGE, MZ_HUGE,
    # M1_* flags
    M1_FLY, M1_SWIM, M1_BREATHLESS, M1_MINDLESS, M1_HUMANOID, M1_ANIMAL,
    M1_SLITHY, M1_NOLIMBS, M1_NOHANDS, M1_NOHEAD, M1_NOEYES, M1_NOTAKE,
    M1_THICK_HIDE, M1_OVIPAROUS, M1_REGEN, M1_SEE_INVIS, M1_TPORT,
    M1_UNSOLID, M1_POIS, M1_ACID, M1_CARNIVORE, M1_OMNIVORE, M1_AMORPHOUS,
    M1_TUNNEL, M1_METALLIVORE, M1_WALLWALK, M1_CONCEAL,
    # M2_* flags
    M2_UNDEAD, M2_HOSTILE, M2_STRONG, M2_ORC, M2_DWARF, M2_ELF, M2_GNOME,
    M2_GIANT, M2_NEUTER, M2_LORD, M2_PRINCE, M2_WANDER, M2_STALK,
    M2_NASTY, M2_GREEDY, M2_JEWELS, M2_COLLECT, M2_NOPOLY, M2_PNAME,
    M2_MALE, M2_SHAPESHIFTER,
    # M3_* flags
    M3_INFRAVISION, M3_INFRAVISIBLE, M3_WAITFORU, M3_WANTSCAND,
    # Weight constants
    WT_ELF, WT_HUMAN, WT_ETHEREAL,
    # Color constants
    CLR_BLACK, CLR_RED, CLR_GREEN, CLR_BROWN, CLR_BLUE, CLR_MAGENTA,
    CLR_CYAN, CLR_GRAY, CLR_YELLOW, CLR_WHITE,
    HI_LORD, HI_GOLD, HI_DOMESTIC, HI_PAPER,
)

ENTRIES = (

    # -----------------------------------------------------------------------
    # Mummies  (lines 1917-1968)
    # -----------------------------------------------------------------------

    # 196 — orc mummy
    MonsterEntry(
        name="orc mummy",
        symbol=MonsterSymbol.S_MUMMY,
        level=5, move_speed=10, ac=5, mr=20, alignment=-4,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 6),
        ),
        weight=850, nutrition=75,
        sound=MS_SILENT, size=MZ_HUMAN,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_POIS,
        flags2=M2_UNDEAD | M2_HOSTILE | M2_ORC | M2_GREEDY | M2_JEWELS,
        flags3=M3_INFRAVISION,
        color=CLR_GRAY,
        difficulty=6,
    ),

    # 197 — dwarf mummy
    MonsterEntry(
        name="dwarf mummy",
        symbol=MonsterSymbol.S_MUMMY,
        level=5, move_speed=10, ac=5, mr=20, alignment=-4,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 6),
        ),
        weight=900, nutrition=150,
        sound=MS_SILENT, size=MZ_HUMAN,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_POIS,
        flags2=M2_UNDEAD | M2_HOSTILE | M2_DWARF | M2_GREEDY | M2_JEWELS,
        flags3=M3_INFRAVISION,
        color=CLR_RED,
        difficulty=6,
    ),

    # 198 — elf mummy
    MonsterEntry(
        name="elf mummy",
        symbol=MonsterSymbol.S_MUMMY,
        level=6, move_speed=12, ac=4, mr=30, alignment=-5,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 4),
        ),
        weight=WT_ELF, nutrition=175,
        sound=MS_SILENT, size=MZ_HUMAN,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_POIS,
        flags2=M2_UNDEAD | M2_HOSTILE | M2_ELF,
        flags3=M3_INFRAVISION,
        color=CLR_GREEN,
        difficulty=7,
    ),

    # 199 — human mummy
    MonsterEntry(
        name="human mummy",
        symbol=MonsterSymbol.S_MUMMY,
        level=6, move_speed=12, ac=4, mr=30, alignment=-5,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 4),
        ),
        weight=WT_HUMAN, nutrition=200,
        sound=MS_SILENT, size=MZ_HUMAN,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_POIS,
        flags2=M2_UNDEAD | M2_HOSTILE,
        flags3=M3_INFRAVISION,
        color=CLR_GRAY,
        difficulty=7,
    ),

    # 200 — ettin mummy
    MonsterEntry(
        name="ettin mummy",
        symbol=MonsterSymbol.S_MUMMY,
        level=7, move_speed=12, ac=4, mr=30, alignment=-6,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 6),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 6),
        ),
        weight=1700, nutrition=250,
        sound=MS_SILENT, size=MZ_HUGE,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_POIS,
        flags2=M2_UNDEAD | M2_HOSTILE | M2_STRONG,
        flags3=M3_INFRAVISION,
        color=CLR_BLUE,
        difficulty=8,
    ),

    # 201 — giant mummy
    MonsterEntry(
        name="giant mummy",
        symbol=MonsterSymbol.S_MUMMY,
        level=8, move_speed=14, ac=3, mr=30, alignment=-7,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 3, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 3, 4),
        ),
        weight=2050, nutrition=375,
        sound=MS_SILENT, size=MZ_HUGE,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_POIS,
        flags2=M2_UNDEAD | M2_HOSTILE | M2_GIANT | M2_STRONG | M2_JEWELS,
        flags3=M3_INFRAVISION,
        color=CLR_CYAN,
        difficulty=10,
    ),

    # -----------------------------------------------------------------------
    # Nagas  (lines 1972-2048)
    # -----------------------------------------------------------------------

    # 202 — red naga hatchling
    MonsterEntry(
        name="red naga hatchling",
        symbol=MonsterSymbol.S_NAGA,
        level=3, move_speed=10, ac=6, mr=0, alignment=0,
        generation_mask=G_GENO,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 4),
        ),
        weight=500, nutrition=100,
        sound=MS_MUMBLE, size=MZ_LARGE,
        resists_mask=MR_FIRE | MR_POISON,
        conveys_mask=MR_FIRE | MR_POISON,
        flags1=M1_NOLIMBS | M1_SLITHY | M1_THICK_HIDE | M1_NOTAKE | M1_OMNIVORE,
        flags2=M2_STRONG,
        flags3=M3_INFRAVISIBLE,
        color=CLR_RED,
        difficulty=4,
    ),

    # 203 — black naga hatchling
    MonsterEntry(
        name="black naga hatchling",
        symbol=MonsterSymbol.S_NAGA,
        level=3, move_speed=10, ac=6, mr=0, alignment=0,
        generation_mask=G_GENO,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 4),
        ),
        weight=500, nutrition=100,
        sound=MS_MUMBLE, size=MZ_LARGE,
        resists_mask=MR_POISON | MR_ACID | MR_STONE,
        conveys_mask=MR_POISON | MR_STONE,
        flags1=M1_NOLIMBS | M1_SLITHY | M1_THICK_HIDE | M1_ACID | M1_NOTAKE | M1_CARNIVORE,
        flags2=M2_STRONG,
        flags3=0,
        color=CLR_BLACK,
        difficulty=4,
    ),

    # 204 — golden naga hatchling
    MonsterEntry(
        name="golden naga hatchling",
        symbol=MonsterSymbol.S_NAGA,
        level=3, move_speed=10, ac=6, mr=0, alignment=0,
        generation_mask=G_GENO,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 4),
        ),
        weight=500, nutrition=100,
        sound=MS_MUMBLE, size=MZ_LARGE,
        resists_mask=MR_POISON,
        conveys_mask=MR_POISON,
        flags1=M1_NOLIMBS | M1_SLITHY | M1_THICK_HIDE | M1_NOTAKE | M1_OMNIVORE,
        flags2=M2_STRONG,
        flags3=0,
        color=HI_GOLD,
        difficulty=4,
    ),

    # 205 — guardian naga hatchling
    MonsterEntry(
        name="guardian naga hatchling",
        symbol=MonsterSymbol.S_NAGA,
        level=3, move_speed=10, ac=6, mr=0, alignment=0,
        generation_mask=G_GENO,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 4),
        ),
        weight=500, nutrition=100,
        sound=MS_MUMBLE, size=MZ_LARGE,
        resists_mask=MR_POISON,
        conveys_mask=MR_POISON,
        flags1=M1_NOLIMBS | M1_SLITHY | M1_THICK_HIDE | M1_NOTAKE | M1_OMNIVORE,
        flags2=M2_STRONG,
        flags3=0,
        color=CLR_GREEN,
        difficulty=4,
    ),

    # 206 — red naga
    MonsterEntry(
        name="red naga",
        symbol=MonsterSymbol.S_NAGA,
        level=6, move_speed=12, ac=4, mr=0, alignment=-4,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 4),
            (AttackType.AT_BREA, DamageType.AD_FIRE, 2, 6),
        ),
        weight=2600, nutrition=400,
        sound=MS_MUMBLE, size=MZ_HUGE,
        resists_mask=MR_FIRE | MR_POISON,
        conveys_mask=MR_FIRE | MR_POISON,
        flags1=M1_NOLIMBS | M1_SLITHY | M1_THICK_HIDE | M1_OVIPAROUS | M1_NOTAKE | M1_OMNIVORE,
        flags2=M2_STRONG,
        flags3=M3_INFRAVISIBLE,
        color=CLR_RED,
        difficulty=8,
    ),

    # 207 — black naga
    MonsterEntry(
        name="black naga",
        symbol=MonsterSymbol.S_NAGA,
        level=8, move_speed=14, ac=2, mr=10, alignment=4,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
            (AttackType.AT_SPIT, DamageType.AD_ACID, 0, 0),
        ),
        weight=2600, nutrition=400,
        sound=MS_MUMBLE, size=MZ_HUGE,
        resists_mask=MR_POISON | MR_ACID | MR_STONE,
        conveys_mask=MR_POISON | MR_STONE,
        flags1=M1_NOLIMBS | M1_SLITHY | M1_THICK_HIDE | M1_OVIPAROUS | M1_ACID | M1_NOTAKE | M1_CARNIVORE,
        flags2=M2_STRONG,
        flags3=0,
        color=CLR_BLACK,
        difficulty=10,
    ),

    # 208 — golden naga
    MonsterEntry(
        name="golden naga",
        symbol=MonsterSymbol.S_NAGA,
        level=10, move_speed=14, ac=2, mr=70, alignment=5,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 4, 6),
        ),
        weight=2600, nutrition=400,
        sound=MS_MUMBLE, size=MZ_HUGE,
        resists_mask=MR_POISON,
        conveys_mask=MR_POISON,
        flags1=M1_NOLIMBS | M1_SLITHY | M1_THICK_HIDE | M1_OVIPAROUS | M1_NOTAKE | M1_OMNIVORE,
        flags2=M2_STRONG,
        flags3=0,
        color=HI_GOLD,
        difficulty=13,
    ),

    # 209 — guardian naga
    MonsterEntry(
        name="guardian naga",
        symbol=MonsterSymbol.S_NAGA,
        level=12, move_speed=16, ac=0, mr=50, alignment=7,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PLYS, 1, 6),
            (AttackType.AT_SPIT, DamageType.AD_DRST, 1, 6),
            (AttackType.AT_HUGS, DamageType.AD_PHYS, 2, 4),
        ),
        weight=2600, nutrition=400,
        sound=MS_MUMBLE, size=MZ_HUGE,
        resists_mask=MR_POISON,
        conveys_mask=MR_POISON,
        flags1=M1_NOLIMBS | M1_SLITHY | M1_THICK_HIDE | M1_OVIPAROUS | M1_POIS | M1_NOTAKE | M1_OMNIVORE,
        flags2=M2_STRONG,
        flags3=0,
        color=CLR_GREEN,
        difficulty=16,
    ),

    # -----------------------------------------------------------------------
    # Ogres  (lines 2052-2075)
    # -----------------------------------------------------------------------

    # 210 — ogre
    MonsterEntry(
        name="ogre",
        symbol=MonsterSymbol.S_OGRE,
        level=5, move_speed=10, ac=5, mr=0, alignment=-3,
        generation_mask=G_SGROUP | G_GENO | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 5),
        ),
        weight=1600, nutrition=500,
        sound=MS_GRUNT, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_CARNIVORE,
        flags2=M2_STRONG | M2_GREEDY | M2_JEWELS | M2_COLLECT,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_BROWN,
        difficulty=7,
    ),

    # 211 — ogre lord  (NAMS: "ogre lord" / "ogre lady" / "ogre leader")
    MonsterEntry(
        name="ogre lord",
        symbol=MonsterSymbol.S_OGRE,
        level=7, move_speed=12, ac=3, mr=30, alignment=-5,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 6),
        ),
        weight=1700, nutrition=700,
        sound=MS_GRUNT, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_CARNIVORE,
        flags2=M2_STRONG | M2_LORD | M2_MALE | M2_GREEDY | M2_JEWELS | M2_COLLECT,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_RED,
        difficulty=9,
    ),

    # 212 — ogre king  (NAMS: "ogre king" / "ogre queen" / "ogre tyrant")
    MonsterEntry(
        name="ogre king",
        symbol=MonsterSymbol.S_OGRE,
        level=9, move_speed=14, ac=4, mr=60, alignment=-7,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 3, 5),
        ),
        weight=1700, nutrition=750,
        sound=MS_GRUNT, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_CARNIVORE,
        flags2=M2_STRONG | M2_PRINCE | M2_MALE | M2_GREEDY | M2_JEWELS | M2_COLLECT,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=HI_LORD,
        difficulty=11,
    ),

    # -----------------------------------------------------------------------
    # Puddings  (lines 2081-2123)
    # -----------------------------------------------------------------------

    # 213 — gray ooze
    MonsterEntry(
        name="gray ooze",
        symbol=MonsterSymbol.S_PUDDING,
        level=3, move_speed=1, ac=8, mr=0, alignment=0,
        generation_mask=G_GENO | G_NOCORPSE | 2,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_RUST, 2, 8),
        ),
        weight=500, nutrition=250,
        sound=MS_SILENT, size=MZ_MEDIUM,
        resists_mask=MR_FIRE | MR_COLD | MR_POISON | MR_ACID | MR_STONE,
        conveys_mask=MR_FIRE | MR_COLD | MR_POISON,
        flags1=(M1_BREATHLESS | M1_AMORPHOUS | M1_NOEYES | M1_NOLIMBS | M1_NOHEAD
                | M1_MINDLESS | M1_OMNIVORE | M1_ACID),
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=0,
        color=CLR_GRAY,
        difficulty=4,
    ),

    # 214 — brown pudding
    MonsterEntry(
        name="brown pudding",
        symbol=MonsterSymbol.S_PUDDING,
        level=5, move_speed=3, ac=8, mr=0, alignment=0,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_DCAY, 0, 0),
        ),
        weight=500, nutrition=250,
        sound=MS_SILENT, size=MZ_MEDIUM,
        resists_mask=MR_COLD | MR_ELEC | MR_POISON | MR_ACID | MR_STONE,
        conveys_mask=MR_COLD | MR_ELEC | MR_POISON,
        flags1=(M1_BREATHLESS | M1_AMORPHOUS | M1_NOEYES | M1_NOLIMBS | M1_NOHEAD
                | M1_MINDLESS | M1_OMNIVORE | M1_ACID),
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=0,
        color=CLR_BROWN,
        difficulty=6,
    ),

    # 215 — green slime
    MonsterEntry(
        name="green slime",
        symbol=MonsterSymbol.S_PUDDING,
        level=6, move_speed=6, ac=6, mr=0, alignment=0,
        generation_mask=G_HELL | G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_TUCH, DamageType.AD_SLIM, 1, 4),
            (AttackType.AT_NONE, DamageType.AD_SLIM, 0, 0),
        ),
        weight=400, nutrition=150,
        sound=MS_SILENT, size=MZ_LARGE,
        resists_mask=MR_COLD | MR_ELEC | MR_POISON | MR_ACID | MR_STONE,
        conveys_mask=0,
        flags1=(M1_BREATHLESS | M1_AMORPHOUS | M1_NOEYES | M1_NOLIMBS | M1_NOHEAD
                | M1_MINDLESS | M1_OMNIVORE | M1_ACID | M1_POIS),
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=0,
        color=CLR_GREEN,
        difficulty=8,
    ),

    # 216 — black pudding
    MonsterEntry(
        name="black pudding",
        symbol=MonsterSymbol.S_PUDDING,
        level=10, move_speed=6, ac=6, mr=0, alignment=0,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_CORR, 3, 8),
            (AttackType.AT_NONE, DamageType.AD_CORR, 0, 0),
        ),
        weight=900, nutrition=250,
        sound=MS_SILENT, size=MZ_LARGE,
        resists_mask=MR_COLD | MR_ELEC | MR_POISON | MR_ACID | MR_STONE,
        conveys_mask=MR_COLD | MR_ELEC | MR_POISON,
        flags1=(M1_BREATHLESS | M1_AMORPHOUS | M1_NOEYES | M1_NOLIMBS | M1_NOHEAD
                | M1_MINDLESS | M1_OMNIVORE | M1_ACID),
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=0,
        color=CLR_BLACK,
        difficulty=12,
    ),

    # -----------------------------------------------------------------------
    # Quantum mechanics  (lines 2127-2143)
    # -----------------------------------------------------------------------

    # 217 — quantum mechanic
    MonsterEntry(
        name="quantum mechanic",
        symbol=MonsterSymbol.S_QUANTMECH,
        level=7, move_speed=12, ac=3, mr=10, alignment=0,
        generation_mask=G_GENO | 3,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_TLPT, 1, 4),
        ),
        weight=WT_HUMAN, nutrition=20,
        sound=MS_HUMANOID, size=MZ_HUMAN,
        resists_mask=MR_POISON,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_OMNIVORE | M1_POIS | M1_TPORT,
        flags2=M2_HOSTILE,
        flags3=M3_INFRAVISIBLE,
        color=CLR_CYAN,
        difficulty=9,
    ),

    # (vendor parity: genetic engineer not present in NLE — removed)

    # -----------------------------------------------------------------------
    # Rust monster / disenchanter  (lines 2147-2161)
    # -----------------------------------------------------------------------

    # 219 — rust monster
    MonsterEntry(
        name="rust monster",
        symbol=MonsterSymbol.S_RUSTMONST,
        level=5, move_speed=18, ac=2, mr=0, alignment=0,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_TUCH, DamageType.AD_RUST, 0, 0),
            (AttackType.AT_TUCH, DamageType.AD_RUST, 0, 0),
            (AttackType.AT_NONE, DamageType.AD_RUST, 0, 0),
        ),
        weight=1000, nutrition=250,
        sound=MS_SILENT, size=MZ_MEDIUM,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_SWIM | M1_ANIMAL | M1_NOHANDS | M1_METALLIVORE,
        flags2=M2_HOSTILE,
        flags3=M3_INFRAVISIBLE,
        color=CLR_BROWN,
        difficulty=8,
    ),

    # 220 — disenchanter
    MonsterEntry(
        name="disenchanter",
        symbol=MonsterSymbol.S_RUSTMONST,
        level=12, move_speed=12, ac=-10, mr=0, alignment=-3,
        generation_mask=G_HELL | G_GENO | 2,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_ENCH, 4, 4),
            (AttackType.AT_NONE, DamageType.AD_ENCH, 0, 0),
        ),
        weight=750, nutrition=200,
        sound=MS_GROWL, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_ANIMAL | M1_CARNIVORE,
        flags2=M2_HOSTILE,
        flags3=M3_INFRAVISIBLE,
        color=CLR_BLUE,
        difficulty=14,
    ),

    # -----------------------------------------------------------------------
    # Snakes  (lines 2167-2221)
    # -----------------------------------------------------------------------

    # 221 — garter snake
    MonsterEntry(
        name="garter snake",
        symbol=MonsterSymbol.S_SNAKE,
        level=1, move_speed=8, ac=8, mr=0, alignment=0,
        generation_mask=G_LGROUP | G_GENO | 1,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 2),
        ),
        weight=50, nutrition=60,
        sound=MS_HISS, size=MZ_TINY,
        resists_mask=0,
        conveys_mask=0,
        flags1=(M1_SWIM | M1_CONCEAL | M1_NOLIMBS | M1_ANIMAL | M1_SLITHY
                | M1_OVIPAROUS | M1_CARNIVORE | M1_NOTAKE),
        flags2=0,
        flags3=0,
        color=CLR_GREEN,
        difficulty=3,
    ),

    # 222 — snake
    MonsterEntry(
        name="snake",
        symbol=MonsterSymbol.S_SNAKE,
        level=4, move_speed=15, ac=3, mr=0, alignment=0,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_DRST, 1, 6),
        ),
        weight=100, nutrition=80,
        sound=MS_HISS, size=MZ_SMALL,
        resists_mask=MR_POISON,
        conveys_mask=MR_POISON,
        flags1=(M1_SWIM | M1_CONCEAL | M1_NOLIMBS | M1_ANIMAL | M1_SLITHY | M1_POIS
                | M1_OVIPAROUS | M1_CARNIVORE | M1_NOTAKE),
        flags2=M2_HOSTILE,
        flags3=0,
        color=CLR_BROWN,
        difficulty=6,
    ),

    # 223 — water moccasin
    MonsterEntry(
        name="water moccasin",
        symbol=MonsterSymbol.S_SNAKE,
        level=4, move_speed=15, ac=3, mr=0, alignment=0,
        generation_mask=G_GENO | G_NOGEN | G_LGROUP,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_DRST, 1, 6),
        ),
        weight=150, nutrition=80,
        sound=MS_HISS, size=MZ_SMALL,
        resists_mask=MR_POISON,
        conveys_mask=MR_POISON,
        flags1=(M1_SWIM | M1_CONCEAL | M1_NOLIMBS | M1_ANIMAL | M1_SLITHY | M1_POIS
                | M1_CARNIVORE | M1_OVIPAROUS | M1_NOTAKE),
        flags2=M2_HOSTILE,
        flags3=0,
        color=CLR_RED,
        difficulty=7,
    ),

    # 224 — python
    MonsterEntry(
        name="python",
        symbol=MonsterSymbol.S_SNAKE,
        level=6, move_speed=3, ac=5, mr=0, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_TUCH, DamageType.AD_PHYS, 0, 0),
            (AttackType.AT_HUGS, DamageType.AD_WRAP, 1, 4),
            (AttackType.AT_HUGS, DamageType.AD_PHYS, 2, 4),
        ),
        weight=250, nutrition=100,
        sound=MS_HISS, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=(M1_SWIM | M1_NOLIMBS | M1_ANIMAL | M1_SLITHY | M1_CARNIVORE
                | M1_OVIPAROUS | M1_NOTAKE),
        flags2=M2_HOSTILE | M2_STRONG,
        flags3=M3_INFRAVISION,
        color=CLR_MAGENTA,
        difficulty=8,
    ),

    # 225 — pit viper
    MonsterEntry(
        name="pit viper",
        symbol=MonsterSymbol.S_SNAKE,
        level=6, move_speed=15, ac=2, mr=0, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_DRST, 1, 4),
            (AttackType.AT_BITE, DamageType.AD_DRST, 1, 4),
        ),
        weight=100, nutrition=60,
        sound=MS_HISS, size=MZ_MEDIUM,
        resists_mask=MR_POISON,
        conveys_mask=MR_POISON,
        flags1=(M1_SWIM | M1_CONCEAL | M1_NOLIMBS | M1_ANIMAL | M1_SLITHY | M1_POIS
                | M1_CARNIVORE | M1_OVIPAROUS | M1_NOTAKE),
        flags2=M2_HOSTILE,
        flags3=M3_INFRAVISION,
        color=CLR_BLUE,
        difficulty=9,
    ),

    # 226 — cobra
    MonsterEntry(
        name="cobra",
        symbol=MonsterSymbol.S_SNAKE,
        level=6, move_speed=18, ac=2, mr=0, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_BITE, DamageType.AD_DRST, 2, 4),
            (AttackType.AT_SPIT, DamageType.AD_BLND, 0, 0),
        ),
        weight=250, nutrition=100,
        sound=MS_HISS, size=MZ_MEDIUM,
        resists_mask=MR_POISON,
        conveys_mask=MR_POISON,
        flags1=(M1_SWIM | M1_CONCEAL | M1_NOLIMBS | M1_ANIMAL | M1_SLITHY | M1_POIS
                | M1_CARNIVORE | M1_OVIPAROUS | M1_NOTAKE),
        flags2=M2_HOSTILE,
        flags3=0,
        color=CLR_BLUE,
        difficulty=10,
    ),

    # -----------------------------------------------------------------------
    # Trolls  (lines 2225-2266)
    # -----------------------------------------------------------------------

    # 227 — troll
    MonsterEntry(
        name="troll",
        symbol=MonsterSymbol.S_TROLL,
        level=7, move_speed=12, ac=4, mr=0, alignment=-3,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 4, 2),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 4, 2),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=800, nutrition=350,
        sound=MS_GRUNT, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_REGEN | M1_CARNIVORE,
        flags2=M2_STRONG | M2_STALK | M2_HOSTILE,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_BROWN,
        difficulty=9,
    ),

    # 228 — ice troll
    MonsterEntry(
        name="ice troll",
        symbol=MonsterSymbol.S_TROLL,
        level=9, move_speed=10, ac=2, mr=20, alignment=-3,
        generation_mask=G_NOHELL | G_GENO | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 6),
            (AttackType.AT_CLAW, DamageType.AD_COLD, 2, 6),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=1000, nutrition=300,
        sound=MS_GRUNT, size=MZ_LARGE,
        resists_mask=MR_COLD,
        conveys_mask=MR_COLD,
        flags1=M1_HUMANOID | M1_REGEN | M1_CARNIVORE,
        flags2=M2_STRONG | M2_STALK | M2_HOSTILE,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_WHITE,
        difficulty=12,
    ),

    # 229 — rock troll
    MonsterEntry(
        name="rock troll",
        symbol=MonsterSymbol.S_TROLL,
        level=9, move_speed=12, ac=0, mr=0, alignment=-3,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 3, 6),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 8),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=1200, nutrition=300,
        sound=MS_GRUNT, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_REGEN | M1_CARNIVORE,
        flags2=M2_STRONG | M2_STALK | M2_HOSTILE | M2_COLLECT,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_CYAN,
        difficulty=12,
    ),

    # 230 — water troll
    MonsterEntry(
        name="water troll",
        symbol=MonsterSymbol.S_TROLL,
        level=11, move_speed=14, ac=4, mr=40, alignment=-3,
        generation_mask=G_NOGEN | G_GENO,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 8),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 8),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=1200, nutrition=350,
        sound=MS_GRUNT, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_REGEN | M1_CARNIVORE | M1_SWIM,
        flags2=M2_STRONG | M2_STALK | M2_HOSTILE,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=CLR_BLUE,
        difficulty=13,
    ),

    # 231 — Olog-hai  (NAMS: "Olog-hai" — no gendered variants)
    MonsterEntry(
        name="Olog-hai",
        symbol=MonsterSymbol.S_TROLL,
        level=13, move_speed=12, ac=-4, mr=0, alignment=-7,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 3, 6),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 8),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 6),
        ),
        weight=1500, nutrition=400,
        sound=MS_GRUNT, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_HUMANOID | M1_REGEN | M1_CARNIVORE,
        flags2=M2_STRONG | M2_STALK | M2_HOSTILE | M2_COLLECT,
        flags3=M3_INFRAVISIBLE | M3_INFRAVISION,
        color=HI_LORD,
        difficulty=16,
    ),

    # -----------------------------------------------------------------------
    # Umber hulk  (lines 2270-2277)
    # -----------------------------------------------------------------------

    # 232 — umber hulk
    MonsterEntry(
        name="umber hulk",
        symbol=MonsterSymbol.S_UMBER,
        level=9, move_speed=6, ac=2, mr=25, alignment=0,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 3, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 3, 4),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 2, 5),
            (AttackType.AT_GAZE, DamageType.AD_CONF, 0, 0),
        ),
        weight=1200, nutrition=500,
        sound=MS_SILENT, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_TUNNEL | M1_CARNIVORE,
        flags2=M2_STRONG,
        flags3=M3_INFRAVISIBLE,
        color=CLR_BROWN,
        difficulty=12,
    ),

    # -----------------------------------------------------------------------
    # Vampires  (lines 2281-2322; #if 0 vampire mage excluded)
    # -----------------------------------------------------------------------

    # 233 — vampire
    MonsterEntry(
        name="vampire",
        symbol=MonsterSymbol.S_VAMPIRE,
        level=10, move_speed=12, ac=1, mr=25, alignment=-8,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_BITE, DamageType.AD_DRLI, 1, 6),
        ),
        weight=WT_HUMAN, nutrition=400,
        sound=MS_VAMPIRE, size=MZ_HUMAN,
        resists_mask=MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_FLY | M1_BREATHLESS | M1_HUMANOID | M1_POIS | M1_REGEN,
        flags2=(M2_UNDEAD | M2_STALK | M2_HOSTILE | M2_STRONG | M2_NASTY
                | M2_SHAPESHIFTER),
        flags3=M3_INFRAVISIBLE,
        color=CLR_RED,
        difficulty=12,
    ),

    # 234 — vampire lord  (NAMS: "vampire lord" / "vampire lady" / "vampire leader")
    MonsterEntry(
        name="vampire lord",
        symbol=MonsterSymbol.S_VAMPIRE,
        level=12, move_speed=14, ac=0, mr=50, alignment=-9,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 8),
            (AttackType.AT_BITE, DamageType.AD_DRLI, 1, 8),
        ),
        weight=WT_HUMAN, nutrition=400,
        sound=MS_VAMPIRE, size=MZ_HUMAN,
        resists_mask=MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_FLY | M1_BREATHLESS | M1_HUMANOID | M1_POIS | M1_REGEN,
        flags2=(M2_UNDEAD | M2_STALK | M2_HOSTILE | M2_STRONG | M2_NASTY | M2_LORD
                | M2_MALE | M2_SHAPESHIFTER),
        flags3=M3_INFRAVISIBLE,
        color=CLR_BLUE,
        difficulty=14,
    ),

    # 235 — Vlad the Impaler
    MonsterEntry(
        name="Vlad the Impaler",
        symbol=MonsterSymbol.S_VAMPIRE,
        level=28, move_speed=26, ac=-6, mr=80, alignment=-10,
        generation_mask=G_NOGEN | G_NOCORPSE | G_UNIQ,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 10),
            (AttackType.AT_BITE, DamageType.AD_DRLI, 1, 12),
        ),
        weight=WT_HUMAN, nutrition=400,
        sound=MS_VAMPIRE, size=MZ_HUMAN,
        resists_mask=MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_FLY | M1_BREATHLESS | M1_HUMANOID | M1_POIS | M1_REGEN,
        flags2=(M2_NOPOLY | M2_UNDEAD | M2_STALK | M2_HOSTILE | M2_PNAME | M2_STRONG
                | M2_NASTY | M2_PRINCE | M2_MALE | M2_SHAPESHIFTER),
        flags3=M3_WAITFORU | M3_WANTSCAND | M3_INFRAVISIBLE,
        color=HI_LORD,
        difficulty=32,
    ),

    # -----------------------------------------------------------------------
    # Wraiths  (lines 2326-2353)
    # -----------------------------------------------------------------------

    # 236 — barrow wight
    MonsterEntry(
        name="barrow wight",
        symbol=MonsterSymbol.S_WRAITH,
        level=3, move_speed=12, ac=5, mr=5, alignment=-3,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_DRLI, 0, 0),
            (AttackType.AT_MAGC, DamageType.AD_SPEL, 0, 0),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
        ),
        weight=1200, nutrition=0,
        sound=MS_SPELL, size=MZ_HUMAN,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_HUMANOID,
        flags2=M2_UNDEAD | M2_STALK | M2_HOSTILE | M2_COLLECT,
        flags3=0,
        color=CLR_GRAY,
        difficulty=7,
    ),

    # 237 — wraith
    MonsterEntry(
        name="wraith",
        symbol=MonsterSymbol.S_WRAITH,
        level=6, move_speed=12, ac=4, mr=15, alignment=-6,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_TUCH, DamageType.AD_DRLI, 1, 6),
        ),
        weight=WT_ETHEREAL, nutrition=0,
        sound=MS_SILENT, size=MZ_HUMAN,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON | MR_STONE,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_FLY | M1_HUMANOID | M1_UNSOLID,
        flags2=M2_UNDEAD | M2_STALK | M2_HOSTILE,
        flags3=0,
        color=CLR_BLACK,
        difficulty=8,
    ),

    # 238 — Nazgul
    MonsterEntry(
        name="Nazgul",
        symbol=MonsterSymbol.S_WRAITH,
        level=13, move_speed=12, ac=0, mr=25, alignment=-17,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_DRLI, 1, 4),
            (AttackType.AT_BREA, DamageType.AD_SLEE, 2, 25),
        ),
        weight=WT_HUMAN, nutrition=0,
        sound=MS_SPELL, size=MZ_HUMAN,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_HUMANOID,
        flags2=(M2_NOPOLY | M2_UNDEAD | M2_STALK | M2_STRONG | M2_HOSTILE | M2_MALE
                | M2_COLLECT),
        flags3=0,
        color=HI_LORD,
        difficulty=17,
    ),

    # -----------------------------------------------------------------------
    # Xorn  (lines 2357-2366)
    # -----------------------------------------------------------------------

    # 239 — xorn
    MonsterEntry(
        name="xorn",
        symbol=MonsterSymbol.S_XORN,
        level=8, move_speed=9, ac=-2, mr=20, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 3),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 3),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 3),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 4, 6),
        ),
        weight=1200, nutrition=700,
        sound=MS_ROAR, size=MZ_MEDIUM,
        resists_mask=MR_FIRE | MR_COLD | MR_STONE,
        conveys_mask=MR_STONE,
        flags1=M1_BREATHLESS | M1_WALLWALK | M1_THICK_HIDE | M1_METALLIVORE,
        flags2=M2_HOSTILE | M2_STRONG,
        flags3=0,
        color=CLR_BROWN,
        difficulty=11,
    ),

    # -----------------------------------------------------------------------
    # Apelike beasts  (lines 2372-2417)
    # -----------------------------------------------------------------------

    # 240 — monkey
    MonsterEntry(
        name="monkey",
        symbol=MonsterSymbol.S_YETI,
        level=2, move_speed=12, ac=6, mr=0, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_SITM, 0, 0),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 3),
        ),
        weight=100, nutrition=50,
        sound=MS_GROWL, size=MZ_SMALL,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_ANIMAL | M1_HUMANOID | M1_OMNIVORE,
        flags2=0,
        flags3=M3_INFRAVISIBLE,
        color=CLR_GRAY,
        difficulty=4,
    ),

    # 241 — ape
    MonsterEntry(
        name="ape",
        symbol=MonsterSymbol.S_YETI,
        level=4, move_speed=12, ac=6, mr=0, alignment=0,
        generation_mask=G_GENO | G_SGROUP | 2,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 3),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 3),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 6),
        ),
        weight=1100, nutrition=500,
        sound=MS_GROWL, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_ANIMAL | M1_HUMANOID | M1_OMNIVORE,
        flags2=M2_STRONG,
        flags3=M3_INFRAVISIBLE,
        color=CLR_BROWN,
        difficulty=6,
    ),

    # 242 — owlbear
    MonsterEntry(
        name="owlbear",
        symbol=MonsterSymbol.S_YETI,
        level=5, move_speed=12, ac=5, mr=0, alignment=0,
        generation_mask=G_GENO | 3,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_HUGS, DamageType.AD_PHYS, 2, 8),
        ),
        weight=1700, nutrition=700,
        sound=MS_ROAR, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_ANIMAL | M1_HUMANOID | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG | M2_NASTY,
        flags3=M3_INFRAVISIBLE,
        color=CLR_BROWN,
        difficulty=7,
    ),

    # 243 — yeti
    MonsterEntry(
        name="yeti",
        symbol=MonsterSymbol.S_YETI,
        level=5, move_speed=15, ac=6, mr=0, alignment=0,
        generation_mask=G_GENO | 2,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_BITE, DamageType.AD_PHYS, 1, 4),
        ),
        weight=1600, nutrition=700,
        sound=MS_GROWL, size=MZ_LARGE,
        resists_mask=MR_COLD,
        conveys_mask=MR_COLD,
        flags1=M1_ANIMAL | M1_HUMANOID | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG,
        flags3=M3_INFRAVISIBLE,
        color=CLR_WHITE,
        difficulty=7,
    ),

    # 244 — carnivorous ape
    MonsterEntry(
        name="carnivorous ape",
        symbol=MonsterSymbol.S_YETI,
        level=6, move_speed=12, ac=6, mr=0, alignment=0,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_HUGS, DamageType.AD_PHYS, 1, 8),
        ),
        weight=1250, nutrition=550,
        sound=MS_GROWL, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_ANIMAL | M1_HUMANOID | M1_CARNIVORE,
        flags2=M2_HOSTILE | M2_STRONG,
        flags3=M3_INFRAVISIBLE,
        color=CLR_BLACK,
        difficulty=8,
    ),

    # 245 — sasquatch
    MonsterEntry(
        name="sasquatch",
        symbol=MonsterSymbol.S_YETI,
        level=7, move_speed=15, ac=6, mr=0, alignment=2,
        generation_mask=G_GENO | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 6),
            (AttackType.AT_KICK, DamageType.AD_PHYS, 1, 8),
        ),
        weight=1550, nutrition=750,
        sound=MS_GROWL, size=MZ_LARGE,
        resists_mask=0,
        conveys_mask=0,
        flags1=M1_ANIMAL | M1_HUMANOID | M1_SEE_INVIS | M1_OMNIVORE,
        flags2=M2_STRONG,
        flags3=M3_INFRAVISIBLE,
        color=CLR_GRAY,
        difficulty=9,
    ),

    # -----------------------------------------------------------------------
    # Zombies  (lines 2421-2505)
    # -----------------------------------------------------------------------

    # 246 — kobold zombie
    MonsterEntry(
        name="kobold zombie",
        symbol=MonsterSymbol.S_ZOMBIE,
        level=0, move_speed=6, ac=10, mr=0, alignment=-2,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
        ),
        weight=400, nutrition=50,
        sound=MS_SILENT,size=MZ_SMALL,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_POIS,
        flags2=M2_UNDEAD | M2_STALK | M2_HOSTILE,
        flags3=M3_INFRAVISION,
        color=CLR_BROWN,
        difficulty=1,
    ),

    # 247 — gnome zombie
    MonsterEntry(
        name="gnome zombie",
        symbol=MonsterSymbol.S_ZOMBIE,
        level=1, move_speed=6, ac=10, mr=0, alignment=-2,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 5),
        ),
        weight=650, nutrition=50,
        sound=MS_SILENT,size=MZ_SMALL,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_POIS,
        flags2=M2_UNDEAD | M2_STALK | M2_HOSTILE | M2_GNOME,
        flags3=M3_INFRAVISION,
        color=CLR_BROWN,
        difficulty=2,
    ),

    # 248 — orc zombie
    MonsterEntry(
        name="orc zombie",
        symbol=MonsterSymbol.S_ZOMBIE,
        level=2, move_speed=6, ac=9, mr=0, alignment=-3,
        generation_mask=G_GENO | G_SGROUP | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 6),
        ),
        weight=850, nutrition=75,
        sound=MS_SILENT,size=MZ_HUMAN,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_POIS,
        flags2=M2_UNDEAD | M2_STALK | M2_HOSTILE | M2_ORC,
        flags3=M3_INFRAVISION,
        color=CLR_GRAY,
        difficulty=3,
    ),

    # 249 — dwarf zombie
    MonsterEntry(
        name="dwarf zombie",
        symbol=MonsterSymbol.S_ZOMBIE,
        level=2, move_speed=6, ac=9, mr=0, alignment=-3,
        generation_mask=G_GENO | G_SGROUP | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 6),
        ),
        weight=900, nutrition=150,
        sound=MS_SILENT,size=MZ_HUMAN,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_POIS,
        flags2=M2_UNDEAD | M2_STALK | M2_HOSTILE | M2_DWARF,
        flags3=M3_INFRAVISION,
        color=CLR_RED,
        difficulty=3,
    ),

    # 250 — elf zombie
    MonsterEntry(
        name="elf zombie",
        symbol=MonsterSymbol.S_ZOMBIE,
        level=3, move_speed=6, ac=9, mr=0, alignment=-3,
        generation_mask=G_GENO | G_SGROUP | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 7),
        ),
        weight=WT_ELF, nutrition=175,
        sound=MS_SILENT,size=MZ_HUMAN,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID,
        flags2=M2_UNDEAD | M2_STALK | M2_HOSTILE | M2_ELF,
        flags3=M3_INFRAVISION,
        color=CLR_GREEN,
        difficulty=4,
    ),

    # 251 — human zombie
    MonsterEntry(
        name="human zombie",
        symbol=MonsterSymbol.S_ZOMBIE,
        level=4, move_speed=6, ac=8, mr=0, alignment=-3,
        generation_mask=G_GENO | G_SGROUP | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 8),
        ),
        weight=WT_HUMAN, nutrition=200,
        sound=MS_SILENT,size=MZ_HUMAN,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID,
        flags2=M2_UNDEAD | M2_STALK | M2_HOSTILE,
        flags3=M3_INFRAVISION,
        color=HI_DOMESTIC,
        difficulty=5,
    ),

    # 252 — ettin zombie
    MonsterEntry(
        name="ettin zombie",
        symbol=MonsterSymbol.S_ZOMBIE,
        level=6, move_speed=8, ac=6, mr=0, alignment=-4,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 10),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 10),
        ),
        weight=1700, nutrition=250,
        sound=MS_SILENT,size=MZ_HUGE,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID,
        flags2=M2_UNDEAD | M2_STALK | M2_HOSTILE | M2_STRONG,
        flags3=M3_INFRAVISION,
        color=CLR_BLUE,
        difficulty=7,
    ),

    # 253 — ghoul
    MonsterEntry(
        name="ghoul",
        symbol=MonsterSymbol.S_ZOMBIE,
        level=3, move_speed=6, ac=10, mr=0, alignment=-2,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PLYS, 1, 2),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 3),
        ),
        weight=400, nutrition=50,
        sound=MS_SILENT, size=MZ_SMALL,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_POIS | M1_OMNIVORE,
        flags2=M2_UNDEAD | M2_WANDER | M2_HOSTILE,
        flags3=M3_INFRAVISION,
        color=CLR_BLACK,
        difficulty=5,
    ),

    # 254 — giant zombie
    MonsterEntry(
        name="giant zombie",
        symbol=MonsterSymbol.S_ZOMBIE,
        level=8, move_speed=8, ac=6, mr=0, alignment=-4,
        generation_mask=G_GENO | G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 8),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 8),
        ),
        weight=2050, nutrition=375,
        sound=MS_SILENT,size=MZ_HUGE,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID,
        flags2=M2_UNDEAD | M2_STALK | M2_HOSTILE | M2_GIANT | M2_STRONG,
        flags3=M3_INFRAVISION,
        color=CLR_CYAN,
        difficulty=9,
    ),

    # 255 — skeleton
    MonsterEntry(
        name="skeleton",
        symbol=MonsterSymbol.S_ZOMBIE,
        level=12, move_speed=8, ac=4, mr=0, alignment=0,
        generation_mask=G_NOCORPSE | G_NOGEN,
        attacks=(
            (AttackType.AT_WEAP, DamageType.AD_PHYS, 2, 6),
            (AttackType.AT_TUCH, DamageType.AD_SLOW, 1, 6),
        ),
        weight=300, nutrition=5,
        sound=MS_BONES, size=MZ_HUMAN,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON | MR_STONE,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_THICK_HIDE,
        flags2=(M2_UNDEAD | M2_WANDER | M2_HOSTILE | M2_STRONG | M2_COLLECT
                | M2_NASTY),
        flags3=M3_INFRAVISION,
        color=CLR_WHITE,
        difficulty=14,
    ),

    # -----------------------------------------------------------------------
    # Golems  (lines 2509-2537)
    # -----------------------------------------------------------------------

    # 256 — straw golem
    MonsterEntry(
        name="straw golem",
        symbol=MonsterSymbol.S_GOLEM,
        level=3, move_speed=12, ac=10, mr=0, alignment=0,
        generation_mask=G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 2),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 2),
        ),
        weight=400, nutrition=0,
        sound=MS_SILENT, size=MZ_LARGE,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID,
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=0,
        color=CLR_YELLOW,
        difficulty=4,
    ),

    # 257 — paper golem
    MonsterEntry(
        name="paper golem",
        symbol=MonsterSymbol.S_GOLEM,
        level=3, move_speed=12, ac=10, mr=0, alignment=0,
        generation_mask=G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 3),
        ),
        weight=400, nutrition=0,
        sound=MS_SILENT, size=MZ_LARGE,
        resists_mask=MR_COLD | MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID,
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=0,
        color=HI_PAPER,
        difficulty=4,
    ),

    # 258 — rope golem
    MonsterEntry(
        name="rope golem",
        symbol=MonsterSymbol.S_GOLEM,
        level=4, move_speed=9, ac=8, mr=0, alignment=0,
        generation_mask=G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 1, 4),
            (AttackType.AT_HUGS, DamageType.AD_PHYS, 6, 1),
        ),
        weight=450, nutrition=0,
        sound=MS_SILENT, size=MZ_LARGE,
        resists_mask=MR_SLEEP | MR_POISON,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID,
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=0,
        color=CLR_BROWN,
        difficulty=6,
    ),

    # 259 — gold golem
    MonsterEntry(
        name="gold golem",
        symbol=MonsterSymbol.S_GOLEM,
        level=5, move_speed=9, ac=6, mr=0, alignment=0,
        generation_mask=G_NOCORPSE | 1,
        attacks=(
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 3),
            (AttackType.AT_CLAW, DamageType.AD_PHYS, 2, 3),
        ),
        weight=450, nutrition=0,
        sound=MS_SILENT, size=MZ_LARGE,
        resists_mask=MR_SLEEP | MR_POISON | MR_ACID,
        conveys_mask=0,
        flags1=M1_BREATHLESS | M1_MINDLESS | M1_HUMANOID | M1_THICK_HIDE,
        flags2=M2_HOSTILE | M2_NEUTER,
        flags3=0,
        color=HI_GOLD,
        difficulty=6,
    ),

)
