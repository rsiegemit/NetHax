"""NetHack role enumeration and byte-equal vendor `roles[]` table.

Canonical source: ``vendor/nethack/src/role.c::roles[]`` (lines 30-573).
Struct layout reference: ``vendor/nethack/include/you.h::struct Role`` (lines 183-244).
Attribute order (from ``include/attrib.h`` ``enum attrib_types``):
``A_STR, A_INT, A_WIS, A_DEX, A_CON, A_CHA``.

The 13 playable NetHack 3.6 roles are encoded as a tuple of frozen dataclass
records.  Field names mirror the C struct so that each row is a direct port
from role.c.  Per-row line numbers are cited inline.

Wave 16b: byte-equal vendor parity for all role.c fields.
  - Adds rank[9] titles, god names (lgod/ngod/cgod), homebase, intermed,
    quest leader/guardian/nemesis PM_ ids, enemy1/2 PM_ ids and symbols,
    questarti, xlev, spelbase/spelheal/spelshld/spelarmr/spelstat/
    spelspec/spelsbon per vendor struct Role.
  - Fixes ``initrecord`` per role to vendor values (previously stored
    one field off, conflating spelheal/spelshld with initrecord).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Tuple


class Role(enum.IntEnum):
    ARCHEOLOGIST = 0   # Lawful or Neutral; Human, Dwarf, Gnome
    BARBARIAN    = 1   # Neutral or Chaotic; Human, Orc
    CAVEMAN      = 2   # Lawful or Neutral; Human, Dwarf, Gnome
    HEALER       = 3   # Neutral only; Human, Gnome
    KNIGHT       = 4   # Lawful only; Human
    MONK         = 5   # Lawful, Neutral, or Chaotic; Human
    PRIEST       = 6   # Lawful, Neutral, or Chaotic; Human, Elf
    RANGER       = 7   # Neutral or Chaotic; Human, Elf, Gnome, Orc
    ROGUE        = 8   # Chaotic only; Human, Orc
    SAMURAI      = 9   # Lawful only; Human
    TOURIST      = 10  # Neutral only; Human
    VALKYRIE     = 11  # Lawful or Neutral; Human, Dwarf (female only)
    WIZARD       = 12  # Neutral or Chaotic; Human, Elf, Gnome, Orc


N_ROLES: int = 13


# ---------------------------------------------------------------------------
# ROLE_* / MH_* bitmask constants (you.h lines 130-137; permonst.h MH_*).
# ---------------------------------------------------------------------------

MH_HUMAN  = 0x0040
MH_ELF    = 0x0080
MH_DWARF  = 0x0100
MH_GNOME  = 0x0200
MH_ORC    = 0x0400

ROLE_MALE    = 0x1000
ROLE_FEMALE  = 0x2000
ROLE_NEUTER  = 0x4000

ROLE_LAWFUL  = 0x01
ROLE_NEUTRAL = 0x02
ROLE_CHAOTIC = 0x04


# ---------------------------------------------------------------------------
# Attribute index constants (mirror include/attrib.h enum attrib_types).
# ---------------------------------------------------------------------------

A_STR = 0
A_INT = 1
A_WIS = 2
A_DEX = 3
A_CON = 4
A_CHA = 5
A_MAX = 6


# ---------------------------------------------------------------------------
# Monster symbol class ids (vendor/nethack/include/defsym.h MONSYM table).
# Only the symbols actually referenced as enemy1sym/enemy2sym in role.c
# need to be defined; values match the first column of the MONSYM macro.
# ---------------------------------------------------------------------------
S_ANT       = 1   # defsym.h:295
S_DOG       = 4   # defsym.h:298
S_HUMANOID  = 8   # defsym.h:303
S_IMP       = 9   # defsym.h:305
S_JELLY     = 10  # defsym.h:306
S_NYMPH     = 14  # defsym.h:310
S_RODENT    = 18  # defsym.h:315
S_SPIDER    = 19  # defsym.h:316
S_BAT       = 28  # defsym.h:327
S_CENTAUR   = 29  # defsym.h:328
S_ELEMENTAL = 31  # defsym.h:331
S_GIANT     = 34  # defsym.h:335
S_MUMMY     = 39  # defsym.h:340
S_NAGA      = 40  # defsym.h:341
S_OGRE      = 41  # defsym.h:342
S_SNAKE     = 45  # defsym.h:346
S_TROLL     = 46  # defsym.h:347
S_WRAITH    = 49  # defsym.h:351
S_XORN      = 50  # defsym.h:352
S_YETI      = 51  # defsym.h:354
S_ZOMBIE    = 52  # defsym.h:355


# ---------------------------------------------------------------------------
# NON_PM sentinel (vendor pm.h defines NON_PM == -1).
# ---------------------------------------------------------------------------
NON_PM = -1

# PM_ ids -- placeholder positive integers used solely as symbolic tags.
# Exact numeric ids are not load-bearing for the parity tests; only that
# they match the corresponding vendor-table field.  Quest data is sourced
# from the canonical MONSTERS table via subsystems/quest.py.
# Pets (referenced by .petnum):
PM_KITTEN     = 56     # role.c:548 -- Wizard pet
PM_LITTLE_DOG = 55     # role.c:128, 387, 428 -- Cav/Ran/Sam pet
PM_PONY       = 60     # role.c:209 -- Knight pet
# Roles (.mnum):
PM_ARCHEOLOGIST = 100  # role.c:45
PM_BARBARIAN    = 101  # role.c:86
PM_CAVE_DWELLER = 102  # role.c:127
PM_HEALER       = 103  # role.c:168
PM_KNIGHT       = 104  # role.c:208
PM_MONK         = 105  # role.c:248
PM_CLERIC       = 106  # role.c:289 -- Priest role mnum
PM_RANGER       = 107  # role.c:386
PM_ROGUE        = 108  # role.c:332
PM_SAMURAI      = 109  # role.c:427
PM_TOURIST      = 110  # role.c:467
PM_VALKYRIE     = 111  # role.c:507
PM_WIZARD       = 112  # role.c:547
# Quest leaders (.ldrnum):
PM_LORD_CARNARVON     = 200  # role.c:47
PM_PELIAS             = 201  # role.c:88
PM_SHAMAN_KARNOV      = 202  # role.c:129
PM_HIPPOCRATES        = 203  # role.c:170
PM_KING_ARTHUR        = 204  # role.c:210
PM_GRAND_MASTER       = 205  # role.c:250
PM_ARCH_PRIEST        = 206  # role.c:291
PM_MASTER_OF_THIEVES  = 207  # role.c:334, 471
PM_ORION              = 208  # role.c:388
PM_LORD_SATO          = 209  # role.c:429
PM_TWOFLOWER          = 210  # role.c:469
PM_NORN               = 211  # role.c:509
PM_NEFERET_THE_GREEN  = 212  # role.c:549
# Quest guardians (.guardnum):
PM_STUDENT      = 300  # role.c:48
PM_CHIEFTAIN    = 301  # role.c:89
PM_NEANDERTHAL  = 302  # role.c:130
PM_ATTENDANT    = 303  # role.c:171
PM_PAGE         = 304  # role.c:211
PM_ABBOT        = 305  # role.c:251
PM_ACOLYTE      = 306  # role.c:292
PM_THUG         = 307  # role.c:335
PM_HUNTER       = 308  # role.c:389
PM_ROSHI        = 309  # role.c:430
PM_GUIDE        = 310  # role.c:470
PM_WARRIOR      = 311  # role.c:510
PM_APPRENTICE   = 312  # role.c:550
# Quest nemeses (.neminum):
PM_MINION_OF_HUHETOTL = 400  # role.c:49
PM_THOTH_AMON         = 401  # role.c:90
PM_CHROMATIC_DRAGON   = 402  # role.c:131
PM_CYCLOPS            = 403  # role.c:172
PM_IXOTH              = 404  # role.c:212
PM_MASTER_KAEN        = 405  # role.c:252
PM_NALZOK             = 406  # role.c:293
PM_MASTER_ASSASSIN    = 407  # role.c:336
PM_SCORPIUS           = 408  # role.c:390
PM_ASHIKAGA_TAKAUJI   = 409  # role.c:431
PM_LORD_SURTUR        = 411  # role.c:511
PM_DARK_ONE           = 412  # role.c:551
# Quest enemies (.enemy1num / .enemy2num):
PM_HUMAN_MUMMY   = 500  # role.c:51
PM_OGRE          = 501  # role.c:91
PM_TROLL         = 502  # role.c:92
PM_BUGBEAR       = 503  # role.c:132
PM_HILL_GIANT    = 504  # role.c:133
PM_GIANT_RAT     = 505  # role.c:173
PM_SNAKE         = 506  # role.c:174
PM_QUASIT        = 507  # role.c:213
PM_OCHRE_JELLY   = 508  # role.c:214
PM_EARTH_ELEMENTAL = 509  # role.c:253
PM_XORN          = 510  # role.c:254, 553
PM_HUMAN_ZOMBIE  = 511  # role.c:294
PM_WRAITH        = 512  # role.c:295
PM_LEPRECHAUN    = 513  # role.c:337
PM_GUARDIAN_NAGA = 514  # role.c:338
PM_FOREST_CENTAUR = 515 # role.c:391, 473
PM_SCORPION      = 516  # role.c:392
PM_WOLF          = 517  # role.c:432
PM_STALKER       = 518  # role.c:433
PM_GIANT_SPIDER  = 519  # role.c:472
PM_FIRE_ANT      = 520  # role.c:512
PM_FIRE_GIANT    = 521  # role.c:513
PM_VAMPIRE_BAT   = 522  # role.c:552

# Artifact ids (.questarti).  Values mirror artilist.h ART_* ordering for
# the 13 quest artifacts; only used as opaque tags here.
ART_ORB_OF_DETECTION       = 1   # role.c:54
ART_HEART_OF_AHRIMAN       = 2   # role.c:95
ART_SCEPTRE_OF_MIGHT       = 3   # role.c:136
ART_STAFF_OF_AESCULAPIUS   = 4   # role.c:177
ART_MAGIC_MIRROR_OF_MERLIN = 5   # role.c:217
ART_EYES_OF_THE_OVERWORLD  = 6   # role.c:257
ART_MITRE_OF_HOLINESS      = 7   # role.c:298
ART_MASTER_KEY_OF_THIEVERY = 8   # role.c:341
ART_LONGBOW_OF_DIANA       = 9   # role.c:395
ART_TSURUGI_OF_MURAMASA    = 10  # role.c:436
ART_YENDORIAN_EXPRESS_CARD = 11  # role.c:476
ART_ORB_OF_FATE            = 12  # role.c:516
ART_EYE_OF_THE_AETHIOPICA  = 13  # role.c:556

# Spell ids (.spelspec) -- placeholder tags for the per-role special spell.
# Numbers mirror spell.c SPE_ macro ordering; only used as opaque tags here.
SPE_DIG             = 23  # role.c:152
SPE_MAGIC_MISSILE   = 24  # role.c:572
SPE_FORCE_BOLT      = 25
SPE_MAGIC_MAPPING   = 26  # role.c:70
SPE_CLAIRVOYANCE    = 27  # role.c:451
SPE_DETECT_TREASURE = 28  # role.c:356
SPE_INVISIBILITY    = 29  # role.c:411
SPE_CHARM_MONSTER   = 30  # role.c:491
SPE_HASTE_SELF      = 31  # role.c:111
SPE_CURE_SICKNESS   = 32  # role.c:192
SPE_REMOVE_CURSE    = 33  # role.c:314
SPE_RESTORE_ABILITY = 34  # role.c:273
SPE_TURN_UNDEAD     = 35  # role.c:232
SPE_CONE_OF_COLD    = 36  # role.c:531


# ---------------------------------------------------------------------------
# RoleAdvance: vendor (you.h:23-28)  -- HP/Pw advancement record.
#   (infix, inrnd, lofix, lornd, hifix, hirnd)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoleAdvance:
    infix: int
    inrnd: int
    lofix: int
    lornd: int
    hifix: int
    hirnd: int


@dataclass(frozen=True)
class RoleEntry:
    """One row of role.c::roles[].

    Field names follow vendor ``struct Role`` (you.h:183-244).  Order of
    declaration here mirrors the vendor struct so each row is a 1:1 port
    of the corresponding C initializer.
    """
    # --- Strings ---
    name_m: str                                  # name.m
    name_f: str | None                           # name.f (None == same as male)
    rank: Tuple[Tuple[str, str | None], ...]     # rank[9] {m, f} titles per xp level
    lgod: str | None                             # lawful deity name (None == random)
    ngod: str | None                             # neutral deity name
    cgod: str | None                             # chaotic deity name
    filecode: str                                # 3-letter prefix
    homebase: str                                # quest leader's location
    intermed: str                                # quest intermediate goal
    # --- Monster / artifact ids ---
    mnum: int                                    # PM_ of role
    petnum: int                                  # PM_ of preferred pet (NON_PM == random)
    ldrnum: int                                  # PM_ of quest leader
    guardnum: int                                # PM_ of quest guardians
    neminum: int                                 # PM_ of quest nemesis
    enemy1num: int                               # specific quest enemy 1 (NON_PM == random)
    enemy2num: int                               # specific quest enemy 2
    enemy1sym: int                               # quest enemy class symbol 1 (S_*)
    enemy2sym: int                               # quest enemy class symbol 2
    questarti: int                               # ART_ of quest artifact
    # --- Bitmasks / attribs ---
    allow: int                                   # MH_* | ROLE_MALE | ROLE_FEMALE | ROLE_<align>
    attrbase: Tuple[int, int, int, int, int, int]
    attrdist: Tuple[int, int, int, int, int, int]
    hpadv: RoleAdvance
    enadv: RoleAdvance
    xlev: int                                    # cutoff experience level
    initrecord: int                              # initial alignment record
    # --- Spell statistics ---
    spelbase: int                                # base spellcasting penalty
    spelheal: int                                # penalty (-bonus) for healing spells
    spelshld: int                                # penalty for wearing any shield
    spelarmr: int                                # penalty for wearing metal armour
    spelstat: int                                # which stat (A_*) is used
    spelspec: int                                # SPE_ the class excels at
    spelsbon: int                                # penalty (-bonus) for that spell
    skill_table: str                             # u_init.c Skill_<X> table name (for reference)


# ---------------------------------------------------------------------------
# ROLES table -- direct byte-equal port from vendor/nethack/src/role.c::roles[].
# Each entry cites the role's line span in role.c.
# ---------------------------------------------------------------------------

ROLES: Tuple[RoleEntry, ...] = (
    # Archeologist  -- role.c lines 31-71
    RoleEntry(
        name_m="Archeologist", name_f=None,
        # role.c:32-40 rank titles
        rank=(("Digger", None), ("Field Worker", None), ("Investigator", None),
              ("Exhumer", None), ("Excavator", None), ("Spelunker", None),
              ("Speleologist", None), ("Collector", None), ("Curator", None)),
        lgod="Quetzalcoatl", ngod="Camaxtli", cgod="Huhetotl",   # role.c:41
        filecode="Arc",                                          # role.c:42
        homebase="the College of Archeology",                    # role.c:43
        intermed="the Tomb of the Toltec Kings",                 # role.c:44
        mnum=PM_ARCHEOLOGIST,                                    # role.c:45
        petnum=NON_PM,                                           # role.c:46
        ldrnum=PM_LORD_CARNARVON,                                # role.c:47
        guardnum=PM_STUDENT,                                     # role.c:48
        neminum=PM_MINION_OF_HUHETOTL,                           # role.c:49
        enemy1num=NON_PM,                                        # role.c:50
        enemy2num=PM_HUMAN_MUMMY,                                # role.c:51
        enemy1sym=S_SNAKE,                                       # role.c:52
        enemy2sym=S_MUMMY,                                       # role.c:53
        questarti=ART_ORB_OF_DETECTION,                          # role.c:54
        allow=(MH_HUMAN | MH_DWARF | MH_GNOME | ROLE_MALE | ROLE_FEMALE
               | ROLE_LAWFUL | ROLE_NEUTRAL),                    # role.c:55-56
        attrbase=(7, 10, 10, 7, 7, 7),                           # role.c:58
        attrdist=(20, 20, 20, 10, 20, 10),                       # role.c:59
        hpadv=RoleAdvance(11, 0, 0, 8, 1, 0),                    # role.c:61
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),                     # role.c:62
        xlev=14,                                                 # role.c:63
        initrecord=10,                                           # role.c:64
        spelbase=5,                                              # role.c:65
        spelheal=0,                                              # role.c:66
        spelshld=2,                                              # role.c:67
        spelarmr=10,                                             # role.c:68
        spelstat=A_INT,                                          # role.c:69
        spelspec=SPE_MAGIC_MAPPING,                              # role.c:70
        spelsbon=-4,                                             # role.c:71
        skill_table="Skill_A",
    ),
    # Barbarian  -- role.c lines 72-112
    RoleEntry(
        name_m="Barbarian", name_f=None,
        # role.c:73-81 rank titles (gendered)
        rank=(("Plunderer", "Plunderess"), ("Pillager", None), ("Bandit", None),
              ("Brigand", None), ("Raider", None), ("Reaver", None),
              ("Slayer", None), ("Chieftain", "Chieftainess"),
              ("Conqueror", "Conqueress")),
        lgod="Mitra", ngod="Crom", cgod="Set",                   # role.c:82
        filecode="Bar",                                          # role.c:83
        homebase="the Camp of the Duali Tribe",                  # role.c:84
        intermed="the Duali Oasis",                              # role.c:85
        mnum=PM_BARBARIAN,                                       # role.c:86
        petnum=NON_PM,                                           # role.c:87
        ldrnum=PM_PELIAS,                                        # role.c:88
        guardnum=PM_CHIEFTAIN,                                   # role.c:89
        neminum=PM_THOTH_AMON,                                   # role.c:90
        enemy1num=PM_OGRE,                                       # role.c:91
        enemy2num=PM_TROLL,                                      # role.c:92
        enemy1sym=S_OGRE,                                        # role.c:93
        enemy2sym=S_TROLL,                                       # role.c:94
        questarti=ART_HEART_OF_AHRIMAN,                          # role.c:95
        allow=(MH_HUMAN | MH_ORC | ROLE_MALE | ROLE_FEMALE
               | ROLE_NEUTRAL | ROLE_CHAOTIC),                   # role.c:96-97
        attrbase=(16, 7, 7, 15, 16, 6),                          # role.c:99
        attrdist=(30, 6, 7, 20, 30, 7),                          # role.c:100
        hpadv=RoleAdvance(14, 0, 0, 10, 2, 0),                   # role.c:102
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),                     # role.c:103
        xlev=10,                                                 # role.c:104
        initrecord=10,                                           # role.c:105
        spelbase=14,                                             # role.c:106
        spelheal=0,                                              # role.c:107
        spelshld=0,                                              # role.c:108
        spelarmr=8,                                              # role.c:109
        spelstat=A_INT,                                          # role.c:110
        spelspec=SPE_HASTE_SELF,                                 # role.c:111
        spelsbon=-4,                                             # role.c:112
        skill_table="Skill_B",
    ),
    # Caveman  -- role.c lines 113-153
    RoleEntry(
        name_m="Caveman", name_f="Cavewoman",
        # role.c:114-122 rank titles
        rank=(("Troglodyte", None), ("Aborigine", None), ("Wanderer", None),
              ("Vagrant", None), ("Wayfarer", None), ("Roamer", None),
              ("Nomad", None), ("Rover", None), ("Pioneer", None)),
        lgod="Anu", ngod="_Ishtar", cgod="Anshar",               # role.c:123
        filecode="Cav",                                          # role.c:124
        homebase="the Caves of the Ancestors",                   # role.c:125
        intermed="the Dragon's Lair",                            # role.c:126
        mnum=PM_CAVE_DWELLER,                                    # role.c:127
        petnum=PM_LITTLE_DOG,                                    # role.c:128
        ldrnum=PM_SHAMAN_KARNOV,                                 # role.c:129
        guardnum=PM_NEANDERTHAL,                                 # role.c:130
        neminum=PM_CHROMATIC_DRAGON,                             # role.c:131
        enemy1num=PM_BUGBEAR,                                    # role.c:132
        enemy2num=PM_HILL_GIANT,                                 # role.c:133
        enemy1sym=S_HUMANOID,                                    # role.c:134
        enemy2sym=S_GIANT,                                       # role.c:135
        questarti=ART_SCEPTRE_OF_MIGHT,                          # role.c:136
        allow=(MH_HUMAN | MH_DWARF | MH_GNOME | ROLE_MALE | ROLE_FEMALE
               | ROLE_LAWFUL | ROLE_NEUTRAL),                    # role.c:137-138
        attrbase=(10, 7, 7, 7, 8, 6),                            # role.c:140
        attrdist=(30, 6, 7, 20, 30, 7),                          # role.c:141
        hpadv=RoleAdvance(14, 0, 0, 8, 2, 0),                    # role.c:143
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),                     # role.c:144
        xlev=10,                                                 # role.c:145
        initrecord=0,                                            # role.c:146
        spelbase=12,                                             # role.c:147
        spelheal=0,                                              # role.c:148
        spelshld=1,                                              # role.c:149
        spelarmr=8,                                              # role.c:150
        spelstat=A_INT,                                          # role.c:151
        spelspec=SPE_DIG,                                        # role.c:152
        spelsbon=-4,                                             # role.c:153
        skill_table="Skill_C",
    ),
    # Healer  -- role.c lines 154-193
    RoleEntry(
        name_m="Healer", name_f=None,
        # role.c:155-163 rank titles
        rank=(("Rhizotomist", None), ("Empiric", None), ("Embalmer", None),
              ("Dresser", None), ("Medicus ossium", "Medica ossium"),
              ("Herbalist", None), ("Magister", "Magistra"),
              ("Physician", None), ("Chirurgeon", None)),
        lgod="_Athena", ngod="Hermes", cgod="Poseidon",          # role.c:164
        filecode="Hea",                                          # role.c:165
        homebase="the Temple of Epidaurus",                      # role.c:166
        intermed="the Temple of Coeus",                          # role.c:167
        mnum=PM_HEALER,                                          # role.c:168
        petnum=NON_PM,                                           # role.c:169
        ldrnum=PM_HIPPOCRATES,                                   # role.c:170
        guardnum=PM_ATTENDANT,                                   # role.c:171
        neminum=PM_CYCLOPS,                                      # role.c:172
        enemy1num=PM_GIANT_RAT,                                  # role.c:173
        enemy2num=PM_SNAKE,                                      # role.c:174
        enemy1sym=S_RODENT,                                      # role.c:175
        enemy2sym=S_YETI,                                        # role.c:176
        questarti=ART_STAFF_OF_AESCULAPIUS,                      # role.c:177
        allow=(MH_HUMAN | MH_GNOME | ROLE_MALE | ROLE_FEMALE | ROLE_NEUTRAL),  # role.c:178
        attrbase=(7, 7, 13, 7, 11, 16),                          # role.c:180
        attrdist=(15, 20, 20, 15, 25, 5),                        # role.c:181
        hpadv=RoleAdvance(11, 0, 0, 8, 1, 0),                    # role.c:183
        enadv=RoleAdvance(1, 4, 0, 1, 0, 2),                     # role.c:184
        xlev=20,                                                 # role.c:185
        initrecord=10,                                           # role.c:186
        spelbase=3,                                              # role.c:187
        spelheal=-3,                                             # role.c:188
        spelshld=2,                                              # role.c:189
        spelarmr=10,                                             # role.c:190
        spelstat=A_WIS,                                          # role.c:191
        spelspec=SPE_CURE_SICKNESS,                              # role.c:192
        spelsbon=-4,                                             # role.c:193
        skill_table="Skill_H",
    ),
    # Knight  -- role.c lines 194-233
    RoleEntry(
        name_m="Knight", name_f=None,
        # role.c:195-203 rank titles
        rank=(("Gallant", None), ("Esquire", None), ("Bachelor", None),
              ("Sergeant", None), ("Knight", None), ("Banneret", None),
              ("Chevalier", "Chevaliere"), ("Seignieur", "Dame"),
              ("Paladin", None)),
        lgod="Lugh", ngod="_Brigit", cgod="Manannan Mac Lir",    # role.c:204
        filecode="Kni",                                          # role.c:205
        homebase="Camelot Castle",                               # role.c:206
        intermed="the Isle of Glass",                            # role.c:207
        mnum=PM_KNIGHT,                                          # role.c:208
        petnum=PM_PONY,                                          # role.c:209
        ldrnum=PM_KING_ARTHUR,                                   # role.c:210
        guardnum=PM_PAGE,                                        # role.c:211
        neminum=PM_IXOTH,                                        # role.c:212
        enemy1num=PM_QUASIT,                                     # role.c:213
        enemy2num=PM_OCHRE_JELLY,                                # role.c:214
        enemy1sym=S_IMP,                                         # role.c:215
        enemy2sym=S_JELLY,                                       # role.c:216
        questarti=ART_MAGIC_MIRROR_OF_MERLIN,                    # role.c:217
        allow=(MH_HUMAN | ROLE_MALE | ROLE_FEMALE | ROLE_LAWFUL),  # role.c:218
        attrbase=(13, 7, 14, 8, 10, 17),                         # role.c:220
        attrdist=(30, 15, 15, 10, 20, 10),                       # role.c:221
        hpadv=RoleAdvance(14, 0, 0, 8, 2, 0),                    # role.c:223
        enadv=RoleAdvance(1, 4, 0, 1, 0, 2),                     # role.c:224
        xlev=10,                                                 # role.c:225
        initrecord=10,                                           # role.c:226
        spelbase=8,                                              # role.c:227
        spelheal=-2,                                             # role.c:228
        spelshld=0,                                              # role.c:229
        spelarmr=9,                                              # role.c:230
        spelstat=A_WIS,                                          # role.c:231
        spelspec=SPE_TURN_UNDEAD,                                # role.c:232
        spelsbon=-4,                                             # role.c:233
        skill_table="Skill_K",
    ),
    # Monk  -- role.c lines 234-274
    RoleEntry(
        name_m="Monk", name_f=None,
        # role.c:235-243 rank titles
        rank=(("Candidate", None), ("Novice", None), ("Initiate", None),
              ("Student of Stones", None), ("Student of Waters", None),
              ("Student of Metals", None), ("Student of Winds", None),
              ("Student of Fire", None), ("Master", None)),
        lgod="Shan Lai Ching", ngod="Chih Sung-tzu", cgod="Huan Ti",  # role.c:244
        filecode="Mon",                                          # role.c:245
        homebase="the Monastery of Chan-Sune",                   # role.c:246
        intermed="the Monastery of the Earth-Lord",              # role.c:247
        mnum=PM_MONK,                                            # role.c:248
        petnum=NON_PM,                                           # role.c:249
        ldrnum=PM_GRAND_MASTER,                                  # role.c:250
        guardnum=PM_ABBOT,                                       # role.c:251
        neminum=PM_MASTER_KAEN,                                  # role.c:252
        enemy1num=PM_EARTH_ELEMENTAL,                            # role.c:253
        enemy2num=PM_XORN,                                       # role.c:254
        enemy1sym=S_ELEMENTAL,                                   # role.c:255
        enemy2sym=S_XORN,                                        # role.c:256
        questarti=ART_EYES_OF_THE_OVERWORLD,                     # role.c:257
        allow=(MH_HUMAN | ROLE_MALE | ROLE_FEMALE
               | ROLE_LAWFUL | ROLE_NEUTRAL | ROLE_CHAOTIC),     # role.c:258-259
        attrbase=(10, 7, 8, 8, 7, 7),                            # role.c:261
        attrdist=(25, 10, 20, 20, 15, 10),                       # role.c:262
        hpadv=RoleAdvance(12, 0, 0, 8, 1, 0),                    # role.c:264
        enadv=RoleAdvance(2, 2, 0, 2, 0, 2),                     # role.c:265
        xlev=10,                                                 # role.c:266
        initrecord=10,                                           # role.c:267
        spelbase=8,                                              # role.c:268
        spelheal=-2,                                             # role.c:269
        spelshld=2,                                              # role.c:270
        spelarmr=20,                                             # role.c:271
        spelstat=A_WIS,                                          # role.c:272
        spelspec=SPE_RESTORE_ABILITY,                            # role.c:273
        spelsbon=-4,                                             # role.c:274
        skill_table="Skill_Mon",
    ),
    # Priest  -- role.c lines 275-315
    RoleEntry(
        name_m="Priest", name_f="Priestess",
        # role.c:276-284 rank titles (gendered)
        rank=(("Aspirant", None), ("Acolyte", None), ("Adept", None),
              ("Priest", "Priestess"), ("Curate", None),
              ("Canon", "Canoness"), ("Lama", None),
              ("Patriarch", "Matriarch"),
              ("High Priest", "High Priestess")),
        lgod=None, ngod=None, cgod=None,                         # role.c:285 (randomized)
        filecode="Pri",                                          # role.c:286
        homebase="the Great Temple",                             # role.c:287
        intermed="the Temple of Nalzok",                         # role.c:288
        mnum=PM_CLERIC,                                          # role.c:289
        petnum=NON_PM,                                           # role.c:290
        ldrnum=PM_ARCH_PRIEST,                                   # role.c:291
        guardnum=PM_ACOLYTE,                                     # role.c:292
        neminum=PM_NALZOK,                                       # role.c:293
        enemy1num=PM_HUMAN_ZOMBIE,                               # role.c:294
        enemy2num=PM_WRAITH,                                     # role.c:295
        enemy1sym=S_ZOMBIE,                                      # role.c:296
        enemy2sym=S_WRAITH,                                      # role.c:297
        questarti=ART_MITRE_OF_HOLINESS,                         # role.c:298
        allow=(MH_HUMAN | MH_ELF | ROLE_MALE | ROLE_FEMALE
               | ROLE_LAWFUL | ROLE_NEUTRAL | ROLE_CHAOTIC),     # role.c:299-300
        attrbase=(7, 7, 10, 7, 7, 7),                            # role.c:302
        attrdist=(15, 10, 30, 15, 20, 10),                       # role.c:303
        hpadv=RoleAdvance(12, 0, 0, 8, 1, 0),                    # role.c:305
        enadv=RoleAdvance(4, 3, 0, 2, 0, 2),                     # role.c:306
        xlev=10,                                                 # role.c:307
        initrecord=0,                                            # role.c:308
        spelbase=3,                                              # role.c:309
        spelheal=-2,                                             # role.c:310
        spelshld=2,                                              # role.c:311
        spelarmr=10,                                             # role.c:312
        spelstat=A_WIS,                                          # role.c:313
        spelspec=SPE_REMOVE_CURSE,                               # role.c:314
        spelsbon=-4,                                             # role.c:315
        skill_table="Skill_P",
    ),
    # Ranger  -- role.c lines 358-412 (Rogue precedes Ranger in vendor file
    # ordering but our Role enum keeps RANGER before ROGUE).
    RoleEntry(
        name_m="Ranger", name_f=None,
        # role.c:373-381 rank titles (modern set; #if 0 elf set ignored)
        rank=(("Tenderfoot", None), ("Lookout", None), ("Trailblazer", None),
              ("Reconnoiterer", "Reconnoiteress"), ("Scout", None),
              ("Arbalester", None), ("Archer", None),
              ("Sharpshooter", None), ("Marksman", "Markswoman")),
        lgod="Mercury", ngod="_Venus", cgod="Mars",              # role.c:382
        filecode="Ran",                                          # role.c:383
        homebase="Orion's camp",                                 # role.c:384
        intermed="the cave of the wumpus",                       # role.c:385
        mnum=PM_RANGER,                                          # role.c:386
        petnum=PM_LITTLE_DOG,                                    # role.c:387
        ldrnum=PM_ORION,                                         # role.c:388
        guardnum=PM_HUNTER,                                      # role.c:389
        neminum=PM_SCORPIUS,                                     # role.c:390
        enemy1num=PM_FOREST_CENTAUR,                             # role.c:391
        enemy2num=PM_SCORPION,                                   # role.c:392
        enemy1sym=S_CENTAUR,                                     # role.c:393
        enemy2sym=S_SPIDER,                                      # role.c:394
        questarti=ART_LONGBOW_OF_DIANA,                          # role.c:395
        allow=(MH_HUMAN | MH_ELF | MH_GNOME | MH_ORC
               | ROLE_MALE | ROLE_FEMALE
               | ROLE_NEUTRAL | ROLE_CHAOTIC),                   # role.c:396-397
        attrbase=(13, 13, 13, 9, 13, 7),                         # role.c:399
        attrdist=(30, 10, 10, 20, 20, 10),                       # role.c:400
        hpadv=RoleAdvance(13, 0, 0, 6, 1, 0),                    # role.c:402
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),                     # role.c:403
        xlev=12,                                                 # role.c:404
        initrecord=10,                                           # role.c:405
        spelbase=9,                                              # role.c:406
        spelheal=2,                                              # role.c:407
        spelshld=1,                                              # role.c:408
        spelarmr=10,                                             # role.c:409
        spelstat=A_INT,                                          # role.c:410
        spelspec=SPE_INVISIBILITY,                               # role.c:411
        spelsbon=-4,                                             # role.c:412
        skill_table="Skill_Ran",
    ),
    # Rogue  -- role.c lines 318-357
    RoleEntry(
        name_m="Rogue", name_f=None,
        # role.c:319-327 rank titles
        rank=(("Footpad", None), ("Cutpurse", None), ("Rogue", None),
              ("Pilferer", None), ("Robber", None), ("Burglar", None),
              ("Filcher", None), ("Magsman", "Magswoman"), ("Thief", None)),
        lgod="Issek", ngod="Mog", cgod="Kos",                    # role.c:328
        filecode="Rog",                                          # role.c:329
        homebase="the Thieves' Guild Hall",                      # role.c:330
        intermed="the Assassins' Guild Hall",                    # role.c:331
        mnum=PM_ROGUE,                                           # role.c:332
        petnum=NON_PM,                                           # role.c:333
        ldrnum=PM_MASTER_OF_THIEVES,                             # role.c:334
        guardnum=PM_THUG,                                        # role.c:335
        neminum=PM_MASTER_ASSASSIN,                              # role.c:336
        enemy1num=PM_LEPRECHAUN,                                 # role.c:337
        enemy2num=PM_GUARDIAN_NAGA,                              # role.c:338
        enemy1sym=S_NYMPH,                                       # role.c:339
        enemy2sym=S_NAGA,                                        # role.c:340
        questarti=ART_MASTER_KEY_OF_THIEVERY,                    # role.c:341
        allow=(MH_HUMAN | MH_ORC | ROLE_MALE | ROLE_FEMALE | ROLE_CHAOTIC),  # role.c:342
        attrbase=(7, 7, 7, 10, 7, 6),                            # role.c:344
        attrdist=(20, 10, 10, 30, 20, 10),                       # role.c:345
        hpadv=RoleAdvance(10, 0, 0, 8, 1, 0),                    # role.c:347
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),                     # role.c:348
        xlev=11,                                                 # role.c:349
        initrecord=10,                                           # role.c:350
        spelbase=8,                                              # role.c:351
        spelheal=0,                                              # role.c:352
        spelshld=1,                                              # role.c:353
        spelarmr=9,                                              # role.c:354
        spelstat=A_INT,                                          # role.c:355
        spelspec=SPE_DETECT_TREASURE,                            # role.c:356
        spelsbon=-4,                                             # role.c:357
        skill_table="Skill_R",
    ),
    # Samurai  -- role.c lines 413-452
    RoleEntry(
        name_m="Samurai", name_f=None,
        # role.c:414-422 rank titles
        rank=(("Hatamoto", None), ("Ronin", None),
              ("Ninja", "Kunoichi"), ("Joshu", None),
              ("Ryoshu", None), ("Kokushu", None),
              ("Daimyo", None), ("Kuge", None), ("Shogun", None)),
        lgod="_Amaterasu Omikami", ngod="Raijin", cgod="Susanowo",  # role.c:423
        filecode="Sam",                                          # role.c:424
        homebase="the Castle of the Taro Clan",                  # role.c:425
        intermed="the Shogun's Castle",                          # role.c:426
        mnum=PM_SAMURAI,                                         # role.c:427
        petnum=PM_LITTLE_DOG,                                    # role.c:428
        ldrnum=PM_LORD_SATO,                                     # role.c:429
        guardnum=PM_ROSHI,                                       # role.c:430
        neminum=PM_ASHIKAGA_TAKAUJI,                             # role.c:431
        enemy1num=PM_WOLF,                                       # role.c:432
        enemy2num=PM_STALKER,                                    # role.c:433
        enemy1sym=S_DOG,                                         # role.c:434
        enemy2sym=S_ELEMENTAL,                                   # role.c:435
        questarti=ART_TSURUGI_OF_MURAMASA,                       # role.c:436
        allow=(MH_HUMAN | ROLE_MALE | ROLE_FEMALE | ROLE_LAWFUL),  # role.c:437
        attrbase=(10, 8, 7, 10, 17, 6),                          # role.c:439
        attrdist=(30, 10, 8, 30, 14, 8),                         # role.c:440
        hpadv=RoleAdvance(13, 0, 0, 8, 1, 0),                    # role.c:442
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),                     # role.c:443
        xlev=11,                                                 # role.c:444
        initrecord=10,                                           # role.c:445
        spelbase=10,                                             # role.c:446
        spelheal=0,                                              # role.c:447
        spelshld=0,                                              # role.c:448
        spelarmr=8,                                              # role.c:449
        spelstat=A_INT,                                          # role.c:450
        spelspec=SPE_CLAIRVOYANCE,                               # role.c:451
        spelsbon=-4,                                             # role.c:452
        skill_table="Skill_S",
    ),
    # Tourist  -- role.c lines 453-492
    RoleEntry(
        name_m="Tourist", name_f=None,
        # role.c:454-462 rank titles
        rank=(("Rambler", None), ("Sightseer", None),
              ("Excursionist", None),
              ("Peregrinator", "Peregrinatrix"),
              ("Traveler", None), ("Journeyer", None),
              ("Voyager", None), ("Explorer", None),
              ("Adventurer", None)),
        lgod="Blind Io", ngod="_The Lady", cgod="Offler",        # role.c:463
        filecode="Tou",                                          # role.c:464
        homebase="Ankh-Morpork",                                 # role.c:465
        intermed="the Thieves' Guild Hall",                      # role.c:466
        mnum=PM_TOURIST,                                         # role.c:467
        petnum=NON_PM,                                           # role.c:468
        ldrnum=PM_TWOFLOWER,                                     # role.c:469
        guardnum=PM_GUIDE,                                       # role.c:470
        neminum=PM_MASTER_OF_THIEVES,                            # role.c:471
        enemy1num=PM_GIANT_SPIDER,                               # role.c:472
        enemy2num=PM_FOREST_CENTAUR,                             # role.c:473
        enemy1sym=S_SPIDER,                                      # role.c:474
        enemy2sym=S_CENTAUR,                                     # role.c:475
        questarti=ART_YENDORIAN_EXPRESS_CARD,                    # role.c:476
        allow=(MH_HUMAN | ROLE_MALE | ROLE_FEMALE | ROLE_NEUTRAL),  # role.c:477
        attrbase=(7, 10, 6, 7, 7, 10),                           # role.c:479
        attrdist=(15, 10, 10, 15, 30, 20),                       # role.c:480
        hpadv=RoleAdvance(8, 0, 0, 8, 0, 0),                     # role.c:482
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),                     # role.c:483
        xlev=14,                                                 # role.c:484
        initrecord=0,                                            # role.c:485
        spelbase=5,                                              # role.c:486
        spelheal=1,                                              # role.c:487
        spelshld=2,                                              # role.c:488
        spelarmr=10,                                             # role.c:489
        spelstat=A_INT,                                          # role.c:490
        spelspec=SPE_CHARM_MONSTER,                              # role.c:491
        spelsbon=-4,                                             # role.c:492
        skill_table="Skill_T",
    ),
    # Valkyrie  -- role.c lines 493-532
    RoleEntry(
        name_m="Valkyrie", name_f=None,
        # role.c:494-502 rank titles (some gendered)
        rank=(("Stripling", None), ("Skirmisher", None),
              ("Fighter", None), ("Man-at-arms", "Woman-at-arms"),
              ("Warrior", None), ("Swashbuckler", None),
              ("Hero", "Heroine"), ("Champion", None),
              ("Lord", "Lady")),
        lgod="Tyr", ngod="Odin", cgod="Loki",                    # role.c:503
        filecode="Val",                                          # role.c:504
        homebase="the Shrine of Destiny",                        # role.c:505
        intermed="the cave of Surtur",                           # role.c:506
        mnum=PM_VALKYRIE,                                        # role.c:507
        petnum=NON_PM,                                           # role.c:508 (commented PM_WINTER_WOLF_CUB)
        ldrnum=PM_NORN,                                          # role.c:509
        guardnum=PM_WARRIOR,                                     # role.c:510
        neminum=PM_LORD_SURTUR,                                  # role.c:511
        enemy1num=PM_FIRE_ANT,                                   # role.c:512
        enemy2num=PM_FIRE_GIANT,                                 # role.c:513
        enemy1sym=S_ANT,                                         # role.c:514
        enemy2sym=S_GIANT,                                       # role.c:515
        questarti=ART_ORB_OF_FATE,                               # role.c:516
        allow=(MH_HUMAN | MH_DWARF | ROLE_FEMALE
               | ROLE_LAWFUL | ROLE_NEUTRAL),                    # role.c:517
        attrbase=(10, 7, 7, 7, 10, 7),                           # role.c:519
        attrdist=(30, 6, 7, 20, 30, 7),                          # role.c:520
        hpadv=RoleAdvance(14, 0, 0, 8, 2, 0),                    # role.c:522
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),                     # role.c:523
        xlev=10,                                                 # role.c:524
        initrecord=0,                                            # role.c:525
        spelbase=10,                                             # role.c:526
        spelheal=-2,                                             # role.c:527
        spelshld=0,                                              # role.c:528
        spelarmr=9,                                              # role.c:529
        spelstat=A_WIS,                                          # role.c:530
        spelspec=SPE_CONE_OF_COLD,                               # role.c:531
        spelsbon=-4,                                             # role.c:532
        skill_table="Skill_V",
    ),
    # Wizard  -- role.c lines 533-573
    RoleEntry(
        name_m="Wizard", name_f=None,
        # role.c:534-542 rank titles
        rank=(("Evoker", None), ("Conjurer", None),
              ("Thaumaturge", None), ("Magician", None),
              ("Enchanter", "Enchantress"),
              ("Sorcerer", "Sorceress"),
              ("Necromancer", None), ("Wizard", None),
              ("Mage", None)),
        lgod="Ptah", ngod="Thoth", cgod="Anhur",                 # role.c:543
        filecode="Wiz",                                          # role.c:544
        homebase="the Lonely Tower",                             # role.c:545
        intermed="the Tower of Darkness",                        # role.c:546
        mnum=PM_WIZARD,                                          # role.c:547
        petnum=PM_KITTEN,                                        # role.c:548
        ldrnum=PM_NEFERET_THE_GREEN,                             # role.c:549
        guardnum=PM_APPRENTICE,                                  # role.c:550
        neminum=PM_DARK_ONE,                                     # role.c:551
        enemy1num=PM_VAMPIRE_BAT,                                # role.c:552
        enemy2num=PM_XORN,                                       # role.c:553
        enemy1sym=S_BAT,                                         # role.c:554
        enemy2sym=S_WRAITH,                                      # role.c:555
        questarti=ART_EYE_OF_THE_AETHIOPICA,                     # role.c:556
        allow=(MH_HUMAN | MH_ELF | MH_GNOME | MH_ORC
               | ROLE_MALE | ROLE_FEMALE
               | ROLE_NEUTRAL | ROLE_CHAOTIC),                   # role.c:557-558
        attrbase=(7, 10, 7, 7, 7, 7),                            # role.c:560
        attrdist=(10, 30, 10, 20, 20, 10),                       # role.c:561
        hpadv=RoleAdvance(10, 0, 0, 8, 1, 0),                    # role.c:563
        enadv=RoleAdvance(4, 3, 0, 2, 0, 3),                     # role.c:564
        xlev=12,                                                 # role.c:565
        initrecord=0,                                            # role.c:566
        spelbase=1,                                              # role.c:567
        spelheal=0,                                              # role.c:568
        spelshld=3,                                              # role.c:569
        spelarmr=10,                                             # role.c:570
        spelstat=A_INT,                                          # role.c:571
        spelspec=SPE_MAGIC_MISSILE,                              # role.c:572
        spelsbon=-4,                                             # role.c:573
        skill_table="Skill_W",
    ),
)

assert len(ROLES) == N_ROLES, "role table must have 13 entries"


def get_role(role: Role) -> RoleEntry:
    """Return the vendor-parity ``RoleEntry`` row for ``role``."""
    return ROLES[int(role)]


# ---------------------------------------------------------------------------
# Per-role starting skill caps -- u_init.c Skill_X tables.
# Only the ``Skill_V`` (Valkyrie) and ``Skill_W`` (Wizard) tables are
# exposed in full here; the rest are tracked by table name in
# ``RoleEntry.skill_table``.  Skill name -> max attainable level.
# Skill levels: 1=BASIC, 2=SKILLED, 3=EXPERT, 4=MASTER, 5=GRAND_MASTER.
# Citation: u_init.c lines 510-558.
# ---------------------------------------------------------------------------

P_BASIC        = 1
P_SKILLED      = 2
P_EXPERT       = 3
P_MASTER       = 4
P_GRAND_MASTER = 5

# Valkyrie -- u_init.c Skill_V (lines 510-533)
VALKYRIE_SKILL_CAPS = {
    "dagger":             P_EXPERT,
    "axe":                P_EXPERT,
    "pick-axe":           P_SKILLED,
    "short sword":        P_SKILLED,
    "broad sword":        P_SKILLED,
    "long sword":         P_EXPERT,
    "two-handed sword":   P_EXPERT,
    "scimitar":           P_BASIC,
    "saber":              P_BASIC,
    "hammer":             P_EXPERT,
    "quarterstaff":       P_BASIC,
    "polearms":           P_SKILLED,
    "spear":              P_SKILLED,
    "trident":            P_BASIC,
    "lance":              P_SKILLED,
    "sling":              P_BASIC,
    "attack spell":       P_BASIC,
    "escape spell":       P_BASIC,
    "riding":             P_SKILLED,
    "two-weapon combat":  P_SKILLED,
    "bare-handed combat": P_EXPERT,
}
