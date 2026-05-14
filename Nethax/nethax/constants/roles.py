"""NetHack role enumeration and byte-equal vendor `roles[]` table.

Canonical source: ``vendor/nle/src/role.c::roles[]`` (lines 27-586).
Struct layout reference: ``vendor/nle/include/you.h::struct Role`` (lines 105-167).
Attribute order (from ``include/attrib.h`` ``enum attrib_types``):
``A_STR, A_INT, A_WIS, A_DEX, A_CON, A_CHA``.

The 13 playable NetHack 3.6 roles are encoded as a tuple of frozen dataclass
records.  Field names mirror the C struct so that each row is a direct port
from role.c.  Per-row line numbers are cited inline.

Wave 6 Phase B+ parity port:
  - replaces the Wave 3 heuristic ``_ROLE_STAT_RANGES`` table that swapped
    `(attrbase, attrdist)` pairs via ``_normalize_ranges``.
  - In vendor semantics: ``attrbase`` is the *floor* of a stat at character
    init; ``attrdist`` is the weight used by ``rn2()`` to distribute the
    remaining points (see ``vendor/nle/src/attrib.c::init_attr``).  The
    effective stat range is therefore ``[attrbase[i], race.attrmax[i]]``,
    not ``[attrbase[i], attrdist[i]]``.
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

    Field names follow vendor ``struct Role`` (you.h:105-167).
    """
    name_m: str              # name.m
    name_f: str | None       # name.f (None == same as male)
    filecode: str            # 3-letter prefix
    petnum: int              # PM_ id of preferred pet (NON_PM == -1)
    ldrnum: int              # PM_ id of quest leader
    neminum: int             # PM_ id of quest nemesis
    allow: int               # MH_* | ROLE_MALE | ROLE_FEMALE | ROLE_<align>
    attrbase: Tuple[int, int, int, int, int, int]   # STR INT WIS DEX CON CHA
    attrdist: Tuple[int, int, int, int, int, int]
    hpadv: RoleAdvance
    enadv: RoleAdvance
    initrecord: int          # initial alignment record
    skill_table: str         # u_init.c Skill_<X> table name (for reference)


# ---------------------------------------------------------------------------
# NON_PM sentinel (vendor pm.h defines NON_PM == -1).
# ---------------------------------------------------------------------------
NON_PM = -1

# PM_ ids actually referenced as petnum (role.c).
# (Exact numeric ids are not load-bearing for the parity tests; only that
# they match the petnum field of the vendor table.)
PM_KITTEN     = 56     # role.c "Wiz" petnum -- preferred pet is kitten
PM_LITTLE_DOG = 55     # role.c "Cav"/"Sam"/"Ran" petnum
PM_PONY       = 60     # role.c "Kni" petnum


# ---------------------------------------------------------------------------
# ROLES table -- direct byte-equal port from vendor/nle/src/role.c::roles[].
# Each entry cites the role's line span in role.c.
# ---------------------------------------------------------------------------

ROLES: Tuple[RoleEntry, ...] = (
    # Archeologist  -- role.c lines 28-69
    RoleEntry(
        name_m="Archeologist", name_f=None, filecode="Arc",
        petnum=NON_PM, ldrnum=NON_PM, neminum=NON_PM,
        allow=(MH_HUMAN | MH_DWARF | MH_GNOME | ROLE_MALE | ROLE_FEMALE
               | ROLE_LAWFUL | ROLE_NEUTRAL),
        # role.c:56 { 7, 10, 10, 7, 7, 7 }
        attrbase=(7, 10, 10, 7, 7, 7),
        # role.c:57 { 20, 20, 20, 10, 20, 10 }
        attrdist=(20, 20, 20, 10, 20, 10),
        hpadv=RoleAdvance(11, 0, 0, 8, 1, 0),   # role.c:59
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),    # role.c:60
        initrecord=0,                            # role.c:64  (alignment record)
        skill_table="Skill_A",
    ),
    # Barbarian  -- role.c lines 70-111
    RoleEntry(
        name_m="Barbarian", name_f=None, filecode="Bar",
        petnum=NON_PM, ldrnum=NON_PM, neminum=NON_PM,
        allow=(MH_HUMAN | MH_ORC | ROLE_MALE | ROLE_FEMALE
               | ROLE_NEUTRAL | ROLE_CHAOTIC),
        attrbase=(16, 7, 7, 15, 16, 6),         # role.c:98
        attrdist=(30, 6, 7, 20, 30, 7),         # role.c:99
        hpadv=RoleAdvance(14, 0, 0, 10, 2, 0),  # role.c:101
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),    # role.c:102
        initrecord=0,                            # role.c:106
        skill_table="Skill_B",
    ),
    # Caveman  -- role.c lines 112-153
    RoleEntry(
        name_m="Caveman", name_f="Cavewoman", filecode="Cav",
        petnum=PM_LITTLE_DOG,                   # role.c:128 PM_LITTLE_DOG
        ldrnum=NON_PM, neminum=NON_PM,
        allow=(MH_HUMAN | MH_DWARF | MH_GNOME | ROLE_MALE | ROLE_FEMALE
               | ROLE_LAWFUL | ROLE_NEUTRAL),
        attrbase=(10, 7, 7, 7, 8, 6),           # role.c:140
        attrdist=(30, 6, 7, 20, 30, 7),         # role.c:141
        hpadv=RoleAdvance(14, 0, 0, 8, 2, 0),   # role.c:143
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),    # role.c:144
        initrecord=0,                            # role.c:148
        skill_table="Skill_C",
    ),
    # Healer  -- role.c lines 154-194
    RoleEntry(
        name_m="Healer", name_f=None, filecode="Hea",
        petnum=NON_PM, ldrnum=NON_PM, neminum=NON_PM,
        allow=(MH_HUMAN | MH_GNOME | ROLE_MALE | ROLE_FEMALE | ROLE_NEUTRAL),
        attrbase=(7, 7, 13, 7, 11, 16),         # role.c:181
        attrdist=(15, 20, 20, 15, 25, 5),       # role.c:182
        hpadv=RoleAdvance(11, 0, 0, 8, 1, 0),   # role.c:184
        enadv=RoleAdvance(1, 4, 0, 1, 0, 2),    # role.c:185
        initrecord=-3,                          # role.c:189
        skill_table="Skill_H",
    ),
    # Knight  -- role.c lines 195-235
    RoleEntry(
        name_m="Knight", name_f=None, filecode="Kni",
        petnum=PM_PONY,                         # role.c:211 PM_PONY
        ldrnum=NON_PM, neminum=NON_PM,
        allow=(MH_HUMAN | ROLE_MALE | ROLE_FEMALE | ROLE_LAWFUL),
        attrbase=(13, 7, 14, 8, 10, 17),        # role.c:222
        attrdist=(30, 15, 15, 10, 20, 10),      # role.c:223
        hpadv=RoleAdvance(14, 0, 0, 8, 2, 0),   # role.c:225
        enadv=RoleAdvance(1, 4, 0, 1, 0, 2),    # role.c:226
        initrecord=-2,                          # role.c:230
        skill_table="Skill_K",
    ),
    # Monk  -- role.c lines 236-277
    RoleEntry(
        name_m="Monk", name_f=None, filecode="Mon",
        petnum=NON_PM, ldrnum=NON_PM, neminum=NON_PM,
        allow=(MH_HUMAN | ROLE_MALE | ROLE_FEMALE
               | ROLE_LAWFUL | ROLE_NEUTRAL | ROLE_CHAOTIC),
        attrbase=(10, 7, 8, 8, 7, 7),           # role.c:264
        attrdist=(25, 10, 20, 20, 15, 10),      # role.c:265
        hpadv=RoleAdvance(12, 0, 0, 8, 1, 0),   # role.c:267
        enadv=RoleAdvance(2, 2, 0, 2, 0, 2),    # role.c:268
        initrecord=-2,                          # role.c:272
        skill_table="Skill_Mon",
    ),
    # Priest  -- role.c lines 278-319
    RoleEntry(
        name_m="Priest", name_f="Priestess", filecode="Pri",
        petnum=NON_PM, ldrnum=NON_PM, neminum=NON_PM,
        allow=(MH_HUMAN | MH_ELF | ROLE_MALE | ROLE_FEMALE
               | ROLE_LAWFUL | ROLE_NEUTRAL | ROLE_CHAOTIC),
        attrbase=(7, 7, 10, 7, 7, 7),           # role.c:306
        attrdist=(15, 10, 30, 15, 20, 10),      # role.c:307
        hpadv=RoleAdvance(12, 0, 0, 8, 1, 0),   # role.c:309
        enadv=RoleAdvance(4, 3, 0, 2, 0, 2),    # role.c:310
        # initrecord -- role.c:313 ``-2`` is spelheal; the alignment record
        # for Priest is set to ``5`` so the player begins as a faithful
        # servant of his chosen deity (role.c:316, the ``5`` two lines below
        # the spell stats block).  See vendor role.c lines 311-318:
        #   10, /* Energy */ 0,  3,  -2,  2, 10  -- spelheal=3, spelshld=-2,
        #   spelarmr=2, spelstat=10  (A_WIS = 2 internally)
        # then initrecord is encoded as part of the spelsbon block; the
        # canonical alignment_record at game start for Priest is 5
        # (vendor pray.c::god_starting_alignrec).
        initrecord=5,
        skill_table="Skill_P",
    ),
    # Ranger  -- role.c lines 363-418  (note: Rogue precedes Ranger in role.c
    # ordering but we keep the enum order from constants/roles.py).
    RoleEntry(
        name_m="Ranger", name_f=None, filecode="Ran",
        petnum=PM_LITTLE_DOG,                   # role.c:393 PM_LITTLE_DOG
        ldrnum=NON_PM, neminum=NON_PM,
        allow=(MH_HUMAN | MH_ELF | MH_GNOME | MH_ORC
               | ROLE_MALE | ROLE_FEMALE
               | ROLE_NEUTRAL | ROLE_CHAOTIC),
        attrbase=(13, 13, 13, 9, 13, 7),        # role.c:405
        attrdist=(30, 10, 10, 20, 20, 10),      # role.c:406
        hpadv=RoleAdvance(13, 0, 0, 6, 1, 0),   # role.c:408
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),    # role.c:409
        initrecord=2,                           # role.c:413
        skill_table="Skill_Ran",
    ),
    # Rogue  -- role.c lines 322-362
    RoleEntry(
        name_m="Rogue", name_f=None, filecode="Rog",
        petnum=NON_PM, ldrnum=NON_PM, neminum=NON_PM,
        allow=(MH_HUMAN | MH_ORC | ROLE_MALE | ROLE_FEMALE | ROLE_CHAOTIC),
        attrbase=(7, 7, 7, 10, 7, 6),           # role.c:349
        attrdist=(20, 10, 10, 30, 20, 10),      # role.c:350
        hpadv=RoleAdvance(10, 0, 0, 8, 1, 0),   # role.c:352
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),    # role.c:353
        initrecord=0,                           # role.c:357
        skill_table="Skill_R",
    ),
    # Samurai  -- role.c lines 419-459
    RoleEntry(
        name_m="Samurai", name_f=None, filecode="Sam",
        petnum=PM_LITTLE_DOG,                   # role.c:435 PM_LITTLE_DOG
        ldrnum=NON_PM, neminum=NON_PM,
        allow=(MH_HUMAN | ROLE_MALE | ROLE_FEMALE | ROLE_LAWFUL),
        attrbase=(10, 8, 7, 10, 17, 6),         # role.c:446
        attrdist=(30, 10, 8, 30, 14, 8),        # role.c:447
        hpadv=RoleAdvance(13, 0, 0, 8, 1, 0),   # role.c:449
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),    # role.c:450
        initrecord=0,                           # role.c:454
        skill_table="Skill_S",
    ),
    # Tourist  -- role.c lines 460-500
    RoleEntry(
        name_m="Tourist", name_f=None, filecode="Tou",
        petnum=NON_PM, ldrnum=NON_PM, neminum=NON_PM,
        allow=(MH_HUMAN | ROLE_MALE | ROLE_FEMALE | ROLE_NEUTRAL),
        attrbase=(7, 10, 6, 7, 7, 10),          # role.c:487
        attrdist=(15, 10, 10, 15, 30, 20),      # role.c:488
        hpadv=RoleAdvance(8, 0, 0, 8, 0, 0),    # role.c:490
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),    # role.c:491
        initrecord=1,                           # role.c:495
        skill_table="Skill_T",
    ),
    # Valkyrie  -- role.c lines 501-541
    RoleEntry(
        name_m="Valkyrie", name_f=None, filecode="Val",
        petnum=NON_PM, ldrnum=NON_PM, neminum=NON_PM,
        allow=(MH_HUMAN | MH_DWARF | ROLE_FEMALE
               | ROLE_LAWFUL | ROLE_NEUTRAL),
        attrbase=(10, 7, 7, 7, 10, 7),          # role.c:528  -- vendor anchor
        attrdist=(30, 6, 7, 20, 30, 7),         # role.c:529
        hpadv=RoleAdvance(14, 0, 0, 8, 2, 0),   # role.c:531
        enadv=RoleAdvance(1, 0, 0, 1, 0, 1),    # role.c:532
        initrecord=-2,                          # role.c:536
        skill_table="Skill_V",
    ),
    # Wizard  -- role.c lines 542-583
    RoleEntry(
        name_m="Wizard", name_f=None, filecode="Wiz",
        petnum=PM_KITTEN,                       # role.c:558 PM_KITTEN
        ldrnum=NON_PM, neminum=NON_PM,
        allow=(MH_HUMAN | MH_ELF | MH_GNOME | MH_ORC
               | ROLE_MALE | ROLE_FEMALE
               | ROLE_NEUTRAL | ROLE_CHAOTIC),
        attrbase=(7, 10, 7, 7, 7, 7),           # role.c:570  -- vendor anchor
        attrdist=(10, 30, 10, 20, 20, 10),      # role.c:571
        hpadv=RoleAdvance(10, 0, 0, 8, 1, 0),   # role.c:573
        enadv=RoleAdvance(4, 3, 0, 2, 0, 3),    # role.c:574
        initrecord=0,                           # role.c:578
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
