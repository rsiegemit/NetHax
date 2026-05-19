"""NetHack race enumeration and byte-equal vendor ``races[]`` table.

Canonical source: ``vendor/nethack/src/role.c::races[]`` (lines 581-685).
Struct layout reference: ``vendor/nethack/include/you.h::struct Race`` (lines 257-294).

Attribute order (matches ``include/attrib.h``):
``A_STR, A_INT, A_WIS, A_DEX, A_CON, A_CHA``.

Wave 16b: byte-equal vendor parity -- adds ``individual`` (race-gendered
name), ``mummynum``, ``zombienum`` fields per vendor struct Race.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Tuple


class Race(enum.IntEnum):
    HUMAN = 0  # Any alignment; no innate resistances
    ELF   = 1  # Chaotic only; sleep resistance
    DWARF = 2  # Lawful only; no innate resistances
    GNOME = 3  # Neutral only; no innate resistances
    ORC   = 4  # Chaotic only; poison resistance


N_RACES: int = 5


# ---------------------------------------------------------------------------
# STR18(x) helper (vendor attrib.h:36) -- encodes 18/xx percentile strength.
#     STR18(100) == 118 (the max strength of 18/** for a male human).
#     STR18(50)  == 68  (cap for gnomes/orcs).
# ---------------------------------------------------------------------------

def STR18(x: int) -> int:
    return 18 + x


@dataclass(frozen=True)
class RaceAdvance:
    """``struct RoleAdvance`` (you.h:23-28) reused for race HP/Pw."""
    infix: int
    inrnd: int
    lofix: int
    lornd: int
    hifix: int
    hirnd: int


@dataclass(frozen=True)
class RaceEntry:
    """One row of role.c::races[]; mirrors vendor ``struct Race`` (you.h:257-294)."""
    noun: str
    adj: str
    coll: str
    filecode: str
    # Race-gendered individual name ("man"/"woman" for human; 0,0 == use noun).
    # Cite: you.h:263 ``struct RoleName individual;``  role.c:587 etc.
    individual: Tuple[str | None, str | None]
    # NLE-style id retained for back-compat with existing helpers; mnum in
    # vendor 3.7 collapses to a single short.  We keep both so callers can
    # query by either name.
    malenum: int
    femalenum: int
    mummynum: int                                   # PM_ as a mummy
    zombienum: int                                  # PM_ as a zombie
    allow: int
    selfmask: int
    lovemask: int
    hatemask: int
    attrmin: Tuple[int, int, int, int, int, int]   # STR INT WIS DEX CON CHA
    attrmax: Tuple[int, int, int, int, int, int]
    hpadv: RaceAdvance
    enadv: RaceAdvance


# ---------------------------------------------------------------------------
# Race / role bitmask sentinels (you.h:130-137 and permonst.h MH_*).
# Duplicated locally to keep this module self-contained.
# ---------------------------------------------------------------------------

MH_HUMAN  = 0x0040
MH_ELF    = 0x0080
MH_DWARF  = 0x0100
MH_GNOME  = 0x0200
MH_ORC    = 0x0400

ROLE_MALE    = 0x1000
ROLE_FEMALE  = 0x2000

ROLE_LAWFUL  = 0x01
ROLE_NEUTRAL = 0x02
ROLE_CHAOTIC = 0x04


# Convenient PM_ ids referenced as malenum (exact value not load-bearing).
PM_HUMAN = 354
PM_ELF   = 355
PM_DWARF = 356
PM_GNOME = 357
PM_ORC   = 358
NON_PM   = -1

# Mummy / zombie PM_ ids (role.c .mummynum / .zombienum). Opaque tags.
PM_HUMAN_MUMMY  = 600  # role.c:589
PM_ELF_MUMMY    = 601  # role.c:610
PM_DWARF_MUMMY  = 602  # role.c:630
PM_GNOME_MUMMY  = 603  # role.c:650
PM_ORC_MUMMY    = 604  # role.c:670
PM_HUMAN_ZOMBIE = 700  # role.c:590
PM_ELF_ZOMBIE   = 701  # role.c:611
PM_DWARF_ZOMBIE = 702  # role.c:631
PM_GNOME_ZOMBIE = 703  # role.c:651
PM_ORC_ZOMBIE   = 704  # role.c:671


# ---------------------------------------------------------------------------
# RACES table -- direct byte-equal port from vendor/nle/src/role.c::races[].
# Each entry cites the line span in role.c.
# ---------------------------------------------------------------------------

RACES: Tuple[RaceEntry, ...] = (
    # Human  -- role.c lines 582-602
    RaceEntry(
        noun="human", adj="human", coll="humanity", filecode="Hum",
        individual=("man", "woman"),                    # role.c:587
        malenum=PM_HUMAN, femalenum=NON_PM,
        mummynum=PM_HUMAN_MUMMY,                        # role.c:589
        zombienum=PM_HUMAN_ZOMBIE,                      # role.c:590
        allow=(MH_HUMAN | ROLE_MALE | ROLE_FEMALE
               | ROLE_LAWFUL | ROLE_NEUTRAL | ROLE_CHAOTIC),  # role.c:591-592
        selfmask=MH_HUMAN, lovemask=0, hatemask=(MH_GNOME | MH_ORC),  # role.c:593-595
        attrmin=(3, 3, 3, 3, 3, 3),                     # role.c:597
        attrmax=(STR18(100), 18, 18, 18, 18, 18),       # role.c:598
        hpadv=RaceAdvance(2, 0, 0, 2, 1, 0),            # role.c:600
        enadv=RaceAdvance(1, 0, 2, 0, 2, 0),            # role.c:601
    ),
    # Elf  -- role.c lines 603-622
    RaceEntry(
        noun="elf", adj="elven", coll="elvenkind", filecode="Elf",
        individual=(None, None),                        # role.c:608  {0, 0}
        malenum=PM_ELF, femalenum=NON_PM,
        mummynum=PM_ELF_MUMMY,                          # role.c:610
        zombienum=PM_ELF_ZOMBIE,                        # role.c:611
        allow=(MH_ELF | ROLE_MALE | ROLE_FEMALE | ROLE_CHAOTIC),  # role.c:612
        selfmask=MH_ELF, lovemask=MH_ELF, hatemask=MH_ORC,        # role.c:613-615
        attrmin=(3, 3, 3, 3, 3, 3),                     # role.c:617
        attrmax=(18, 20, 20, 18, 16, 18),               # role.c:618
        hpadv=RaceAdvance(1, 0, 0, 1, 1, 0),            # role.c:620
        enadv=RaceAdvance(2, 0, 3, 0, 3, 0),            # role.c:621
    ),
    # Dwarf  -- role.c lines 623-642
    RaceEntry(
        noun="dwarf", adj="dwarven", coll="dwarvenkind", filecode="Dwa",
        individual=(None, None),                        # role.c:628  {0, 0}
        malenum=PM_DWARF, femalenum=NON_PM,
        mummynum=PM_DWARF_MUMMY,                        # role.c:630
        zombienum=PM_DWARF_ZOMBIE,                      # role.c:631
        allow=(MH_DWARF | ROLE_MALE | ROLE_FEMALE | ROLE_LAWFUL),     # role.c:632
        selfmask=MH_DWARF, lovemask=(MH_DWARF | MH_GNOME), hatemask=MH_ORC,  # role.c:633-635
        attrmin=(3, 3, 3, 3, 3, 3),                     # role.c:637
        attrmax=(STR18(100), 16, 16, 20, 20, 16),       # role.c:638 -- CON cap 20
        hpadv=RaceAdvance(4, 0, 0, 3, 2, 0),            # role.c:640
        enadv=RaceAdvance(0, 0, 0, 0, 0, 0),            # role.c:641
    ),
    # Gnome  -- role.c lines 643-662
    RaceEntry(
        noun="gnome", adj="gnomish", coll="gnomehood", filecode="Gno",
        individual=(None, None),                        # role.c:648  {0, 0}
        malenum=PM_GNOME, femalenum=NON_PM,
        mummynum=PM_GNOME_MUMMY,                        # role.c:650
        zombienum=PM_GNOME_ZOMBIE,                      # role.c:651
        allow=(MH_GNOME | ROLE_MALE | ROLE_FEMALE | ROLE_NEUTRAL),    # role.c:652
        selfmask=MH_GNOME, lovemask=(MH_DWARF | MH_GNOME), hatemask=MH_HUMAN,  # role.c:653-655
        attrmin=(3, 3, 3, 3, 3, 3),                     # role.c:657
        attrmax=(STR18(50), 19, 18, 18, 18, 18),        # role.c:658 -- STR cap 18/50
        hpadv=RaceAdvance(1, 0, 0, 1, 0, 0),            # role.c:660
        enadv=RaceAdvance(2, 0, 2, 0, 2, 0),            # role.c:661
    ),
    # Orc  -- role.c lines 663-682
    RaceEntry(
        noun="orc", adj="orcish", coll="orcdom", filecode="Orc",
        individual=(None, None),                        # role.c:668  {0, 0}
        malenum=PM_ORC, femalenum=NON_PM,
        mummynum=PM_ORC_MUMMY,                          # role.c:670
        zombienum=PM_ORC_ZOMBIE,                        # role.c:671
        allow=(MH_ORC | ROLE_MALE | ROLE_FEMALE | ROLE_CHAOTIC),      # role.c:672
        selfmask=MH_ORC, lovemask=0, hatemask=(MH_HUMAN | MH_ELF | MH_DWARF),  # role.c:673-675
        attrmin=(3, 3, 3, 3, 3, 3),                     # role.c:677
        attrmax=(STR18(50), 16, 16, 18, 18, 16),        # role.c:678 -- WIS cap 16
        hpadv=RaceAdvance(1, 0, 0, 1, 0, 0),            # role.c:680
        enadv=RaceAdvance(1, 0, 1, 0, 1, 0),            # role.c:681
    ),
)

assert len(RACES) == N_RACES, "race table must have 5 entries"


def get_race(race: Race) -> RaceEntry:
    """Return the vendor-parity ``RaceEntry`` row for ``race``."""
    return RACES[int(race)]
