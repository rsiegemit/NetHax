"""NetHack race enumeration and byte-equal vendor ``races[]`` table.

Canonical source: ``vendor/nle/src/role.c::races[]`` (lines 617-726).
Struct layout reference: ``vendor/nle/include/you.h::struct Race`` (lines 181-219).

The vendor file ``vendor/nethack/src/role.c`` *declares* the symbol
``races[NUM_RACES + 1]`` at line 581 but the data definition was refactored
into NLE's ``vendor/nle/src/role.c`` (NetHack 3.6.2 release).  We use NLE as
the canonical source -- it is byte-equal to the 3.6 vendor distribution.

Attribute order (matches ``include/attrib.h``):
``A_STR, A_INT, A_WIS, A_DEX, A_CON, A_CHA``.

Wave 6 Phase B+ parity port -- replaces the Wave 1 stub.
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
    """One row of role.c::races[]; mirrors vendor ``struct Race`` (you.h:181-219)."""
    noun: str
    adj: str
    coll: str
    filecode: str
    malenum: int
    femalenum: int
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


# ---------------------------------------------------------------------------
# RACES table -- direct byte-equal port from vendor/nle/src/role.c::races[].
# Each entry cites the line span in role.c.
# ---------------------------------------------------------------------------

RACES: Tuple[RaceEntry, ...] = (
    # Human  -- role.c lines 618-639
    RaceEntry(
        noun="human", adj="human", coll="humanity", filecode="Hum",
        malenum=PM_HUMAN, femalenum=NON_PM,
        allow=(MH_HUMAN | ROLE_MALE | ROLE_FEMALE
               | ROLE_LAWFUL | ROLE_NEUTRAL | ROLE_CHAOTIC),
        selfmask=MH_HUMAN, lovemask=0, hatemask=(MH_GNOME | MH_ORC),
        attrmin=(3, 3, 3, 3, 3, 3),                     # role.c:634
        attrmax=(STR18(100), 18, 18, 18, 18, 18),       # role.c:635
        hpadv=RaceAdvance(2, 0, 0, 2, 1, 0),            # role.c:637
        enadv=RaceAdvance(1, 0, 2, 0, 2, 0),            # role.c:638
    ),
    # Elf  -- role.c lines 640-660
    RaceEntry(
        noun="elf", adj="elven", coll="elvenkind", filecode="Elf",
        malenum=PM_ELF, femalenum=NON_PM,
        allow=(MH_ELF | ROLE_MALE | ROLE_FEMALE | ROLE_CHAOTIC),
        selfmask=MH_ELF, lovemask=MH_ELF, hatemask=MH_ORC,
        attrmin=(3, 3, 3, 3, 3, 3),                     # role.c:655
        attrmax=(18, 20, 20, 18, 16, 18),               # role.c:656
        hpadv=RaceAdvance(1, 0, 0, 1, 1, 0),            # role.c:658
        enadv=RaceAdvance(2, 0, 3, 0, 3, 0),            # role.c:659
    ),
    # Dwarf  -- role.c lines 661-681
    RaceEntry(
        noun="dwarf", adj="dwarven", coll="dwarvenkind", filecode="Dwa",
        malenum=PM_DWARF, femalenum=NON_PM,
        allow=(MH_DWARF | ROLE_MALE | ROLE_FEMALE | ROLE_LAWFUL),
        selfmask=MH_DWARF, lovemask=(MH_DWARF | MH_GNOME), hatemask=MH_ORC,
        attrmin=(3, 3, 3, 3, 3, 3),                     # role.c:676
        attrmax=(STR18(100), 16, 16, 20, 20, 16),       # role.c:677 -- CON cap 20
        hpadv=RaceAdvance(4, 0, 0, 3, 2, 0),            # role.c:679
        enadv=RaceAdvance(0, 0, 0, 0, 0, 0),            # role.c:680
    ),
    # Gnome  -- role.c lines 682-702
    RaceEntry(
        noun="gnome", adj="gnomish", coll="gnomehood", filecode="Gno",
        malenum=PM_GNOME, femalenum=NON_PM,
        allow=(MH_GNOME | ROLE_MALE | ROLE_FEMALE | ROLE_NEUTRAL),
        selfmask=MH_GNOME, lovemask=(MH_DWARF | MH_GNOME), hatemask=MH_HUMAN,
        attrmin=(3, 3, 3, 3, 3, 3),                     # role.c:697
        attrmax=(STR18(50), 19, 18, 18, 18, 18),        # role.c:698 -- STR cap 18/50
        hpadv=RaceAdvance(1, 0, 0, 1, 0, 0),            # role.c:700
        enadv=RaceAdvance(2, 0, 2, 0, 2, 0),            # role.c:701
    ),
    # Orc  -- role.c lines 703-723
    RaceEntry(
        noun="orc", adj="orcish", coll="orcdom", filecode="Orc",
        malenum=PM_ORC, femalenum=NON_PM,
        allow=(MH_ORC | ROLE_MALE | ROLE_FEMALE | ROLE_CHAOTIC),
        selfmask=MH_ORC, lovemask=0, hatemask=(MH_HUMAN | MH_ELF | MH_DWARF),
        attrmin=(3, 3, 3, 3, 3, 3),                     # role.c:718
        attrmax=(STR18(50), 16, 16, 18, 18, 16),        # role.c:719 -- WIS cap 16
        hpadv=RaceAdvance(1, 0, 0, 1, 0, 0),            # role.c:721
        enadv=RaceAdvance(1, 0, 1, 0, 1, 0),            # role.c:722
    ),
)

assert len(RACES) == N_RACES, "race table must have 5 entries"


def get_race(race: Race) -> RaceEntry:
    """Return the vendor-parity ``RaceEntry`` row for ``race``."""
    return RACES[int(race)]
