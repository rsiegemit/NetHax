"""
NetHack monster table — complete static data for the JAX reimplementation.

Canonical source: vendor/nethack/include/monsters.h (entry data),
                  vendor/nethack/include/permonst.h  (struct permonst),
                  vendor/nethack/include/monattk.h   (AT_* / AD_* enums),
                  vendor/nethack/include/monflag.h   (MR_*, MS_*, MZ_* constants),
                  vendor/nethack/include/defsym.h    (S_ANT … S_MIMIC_DEF symbols).

Status: Wave 2 complete — all canonical entries (no #if 0 / #ifdef CHARON /
        #ifdef MAIL_STRUCTURES blocks; those are excluded by default).
"""

from __future__ import annotations

import dataclasses
from enum import IntEnum
from typing import Tuple


# ---------------------------------------------------------------------------
# Monster symbol classes  (from vendor/nethack/include/defsym.h MONSYM block)
# ---------------------------------------------------------------------------

class MonsterSymbol(IntEnum):
    S_ANT        =  1   # 'a' ant or other insect
    S_BLOB       =  2   # 'b' blob
    S_COCKATRICE =  3   # 'c' cockatrice
    S_DOG        =  4   # 'd' dog or other canine
    S_EYE        =  5   # 'e' eye or sphere
    S_FELINE     =  6   # 'f' cat or other feline
    S_GREMLIN    =  7   # 'g' gremlin
    S_HUMANOID   =  8   # 'h' humanoid
    S_IMP        =  9   # 'i' imp or minor demon
    S_JELLY      = 10   # 'j' jelly
    S_KOBOLD     = 11   # 'k' kobold
    S_LEPRECHAUN = 12   # 'l' leprechaun
    S_MIMIC      = 13   # 'm' mimic
    S_NYMPH      = 14   # 'n' nymph
    S_ORC        = 15   # 'o' orc
    S_PIERCER    = 16   # 'p' piercer
    S_QUADRUPED  = 17   # 'q' quadruped
    S_RODENT     = 18   # 'r' rodent
    S_SPIDER     = 19   # 's' arachnid or centipede
    S_TRAPPER    = 20   # 't' trapper or lurker above
    S_UNICORN    = 21   # 'u' unicorn or horse
    S_VORTEX     = 22   # 'v' vortex
    S_WORM       = 23   # 'w' worm
    S_XAN        = 24   # 'x' xan or other mythical/fantastic insect
    S_LIGHT      = 25   # 'y' light
    S_ZRUTY      = 26   # 'z' zruty
    S_ANGEL      = 27   # 'A' angelic being
    S_BAT        = 28   # 'B' bat or bird
    S_CENTAUR    = 29   # 'C' centaur
    S_DRAGON     = 30   # 'D' dragon
    S_ELEMENTAL  = 31   # 'E' elemental
    S_FUNGUS     = 32   # 'F' fungus or mold
    S_GNOME      = 33   # 'G' gnome
    S_GIANT      = 34   # 'H' giant humanoid
    S_INVISIBLE  = 35   # 'I' invisible monster
    S_JABBERWOCK = 36   # 'J' jabberwock
    S_KOP        = 37   # 'K' Keystone Kop
    S_LICH       = 38   # 'L' lich
    S_MUMMY      = 39   # 'M' mummy
    S_NAGA       = 40   # 'N' naga
    S_OGRE       = 41   # 'O' ogre
    S_PUDDING    = 42   # 'P' pudding or ooze
    S_QUANTMECH  = 43   # 'Q' quantum mechanic
    S_RUSTMONST  = 44   # 'R' rust monster or disenchanter
    S_SNAKE      = 45   # 'S' snake
    S_TROLL      = 46   # 'T' troll
    S_UMBER      = 47   # 'U' umber hulk
    S_VAMPIRE    = 48   # 'V' vampire
    S_WRAITH     = 49   # 'W' wraith
    S_XORN       = 50   # 'X' xorn
    S_YETI       = 51   # 'Y' apelike creature
    S_ZOMBIE     = 52   # 'Z' zombie
    S_HUMAN      = 53   # '@' human or elf
    S_GHOST      = 54   # ' ' ghost
    S_GOLEM      = 55   # "'" golem
    S_DEMON      = 56   # '&' major demon
    S_EEL        = 57   # ';' sea monster
    S_LIZARD     = 58   # ':' lizard
    S_WORM_TAIL  = 59   # '~' long worm tail
    S_MIMIC_DEF  = 60   # ']' mimic


# ---------------------------------------------------------------------------
# Attack types  (from vendor/nethack/include/monattk.h)
# ---------------------------------------------------------------------------

class AttackType(IntEnum):
    AT_NONE =   0   # passive
    AT_CLAW =   1   # claw / punch / hit
    AT_BITE =   2   # bite
    AT_KICK =   3   # kick
    AT_BUTT =   4   # head butt
    AT_TUCH =   5   # touches
    AT_STNG =   6   # sting
    AT_HUGS =   7   # crushing bearhug
    AT_SPIT =  10   # spits substance (ranged)
    AT_ENGL =  11   # engulf
    AT_BREA =  12   # breath (ranged)
    AT_EXPL =  13   # explodes — proximity
    AT_BOOM =  14   # explodes when killed
    AT_GAZE =  15   # gaze (ranged)
    AT_TENT =  16   # tentacles
    AT_WEAP = 254   # uses weapon
    AT_MAGC = 255   # uses magic spell


# ---------------------------------------------------------------------------
# Damage types  (from vendor/nethack/include/monattk.h)
# ---------------------------------------------------------------------------

class DamageType(IntEnum):
    AD_PHYS =   0   # ordinary physical
    AD_MAGM =   1   # magic missiles
    AD_FIRE =   2   # fire damage
    AD_COLD =   3   # frost damage
    AD_SLEE =   4   # sleep ray
    AD_DISN =   5   # disintegration / death ray
    AD_ELEC =   6   # shock damage
    AD_DRST =   7   # drains strength (poison)
    AD_ACID =   8   # acid damage
    AD_SPC1 =   9   # buzz() extension slot 1
    AD_SPC2 =  10   # buzz() extension slot 2
    AD_BLND =  11   # blinds
    AD_STUN =  12   # stuns
    AD_SLOW =  13   # slows
    AD_PLYS =  14   # paralyzes
    AD_DRLI =  15   # drains life levels
    AD_DREN =  16   # drains magic energy
    AD_LEGS =  17   # damages legs
    AD_STON =  18   # petrifies
    AD_STCK =  19   # sticks (mimic)
    AD_SGLD =  20   # steals gold
    AD_SITM =  21   # steals item
    AD_SEDU =  22   # seduces & steals multiple items
    AD_TLPT =  23   # teleports you
    AD_RUST =  24   # rusts armour
    AD_CONF =  25   # confuses
    AD_DGST =  26   # digests
    AD_HEAL =  27   # heals (nurse)
    AD_WRAP =  28   # wrap (eels)
    AD_WERE =  29   # confers lycanthropy
    AD_DRDX =  30   # drains dexterity
    AD_DRCO =  31   # drains constitution
    AD_DRIN =  32   # drains intelligence (mind flayer)
    AD_DISE =  33   # confers disease
    AD_DCAY =  34   # decays organics
    AD_SSEX =  35   # succubus seduction (extended)
    AD_HALU =  36   # hallucination
    AD_DETH =  37   # Death only
    AD_PEST =  38   # Pestilence only
    AD_FAMN =  39   # Famine only
    AD_SLIM =  40   # turns you into green slime
    AD_ENCH =  41   # removes enchantment
    AD_CORR =  42   # corrodes armor
    AD_POLY =  43   # polymorphs target
    AD_CLRC = 240   # random clerical spell
    AD_SPEL = 241   # random magic spell
    AD_RBRE = 242   # random breath weapon
    AD_SAMU = 252   # may steal Amulet (Wizard)
    AD_CURS = 253   # random curse


# ---------------------------------------------------------------------------
# Attack tuple type alias:  (AttackType, DamageType, n_dice, n_sides)
# ---------------------------------------------------------------------------

Attack = Tuple[AttackType, DamageType, int, int]

NO_ATTK: Attack = (AttackType.AT_NONE, DamageType.AD_PHYS, 0, 0)


# ---------------------------------------------------------------------------
# MonsterEntry dataclass  (mirrors struct permonst from permonst.h)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MonsterEntry:
    name: str
    """Common name string (from NAM macro first arg)."""

    symbol: MonsterSymbol
    """Monster glyph class (S_ANT, S_BLOB, …)."""

    level: int
    """Base monster level / hit dice (mlevel)."""

    move_speed: int
    """Movement rate (mmove); NORMAL_SPEED = 12."""

    ac: int
    """Base armour class (lower is better, can be negative)."""

    mr: int
    """Magic resistance percentage (0–100)."""

    alignment: int
    """Base alignment: negative = chaotic, 0 = neutral, positive = lawful."""

    generation_mask: int
    """Raw geno flags (G_GENO, G_NOGEN, G_SGROUP, G_LGROUP, frequency bits)."""

    attacks: Tuple[Attack, ...]
    """Up to NATTK=6 attacks; unused slots are NO_ATTK."""

    weight: int
    """Corpse weight (cwt), in 'cn' units (0.1 lb each)."""

    nutrition: int
    """Nutritional value when eaten (cnutrit)."""

    sound: int
    """Sound type (MS_* enum value from monflag.h)."""

    size: int
    """Physical size (MZ_TINY=0 … MZ_GIGANTIC=7)."""

    resists_mask: int
    """Inherent resistances bitmask (MR_FIRE=0x01, MR_COLD=0x02, …)."""

    conveys_mask: int
    """Properties conveyed to the player by eating this corpse."""

    flags1: int
    """M1_* boolean flags (M1_ANIMAL, M1_FLY, M1_NOHANDS, …)."""

    flags2: int
    """M2_* boolean flags (M2_HOSTILE, M2_FEMALE, M2_PRINCE, …)."""

    flags3: int
    """M3_* boolean flags (M3_INFRAVISIBLE, …)."""

    color: int
    """Display colour index (CLR_* values from color.h)."""

    difficulty: int = 0
    """Pre-computed difficulty rating (mons[i].difficulty).

    Source: vendor/nle/src/monst.c MON() macro's trailing `d` argument
    (see lines 47-50: ``#define MON(nam, sym, lvl, gen, atk, siz, mr1, mr2,
    flg1, flg2, flg3, d, col)``). Used by makemon.c::rndmonst() and
    mondata.c::mstrength to drive depth-based monster generation.

    Defaults to 0 for backward compatibility with entries that haven't been
    populated yet (Wave 6 closing audit adds this to all 381 entries).
    """


# ---------------------------------------------------------------------------
# Canonical count  (from permonst.h: NUMMONS enum = 394 in NetHack-3.7 HEAD)
# grep -c MON vendor/nethack/include/monsters.h → 394 entries
# ---------------------------------------------------------------------------

NUMMONS: int = 381  # active entries — matches vendor NLE monst.c exactly


# ---------------------------------------------------------------------------
# Resistance / conveyance bit constants  (from monflag.h)
# ---------------------------------------------------------------------------

MR_FIRE   = 0x01
MR_COLD   = 0x02
MR_SLEEP  = 0x04
MR_DISINT = 0x08
MR_ELEC   = 0x10
MR_POISON = 0x20
MR_ACID   = 0x40
MR_STONE  = 0x80

# ---------------------------------------------------------------------------
# Generation mask bits  (from monflag.h)
# ---------------------------------------------------------------------------
G_UNIQ     = 0x1000  # generated only once
G_NOHELL   = 0x0800  # not generated in hell
G_HELL     = 0x0400  # generated only in hell
G_NOGEN    = 0x0200  # generated only specially
G_SGROUP   = 0x0080  # appear in small groups normally
G_LGROUP   = 0x0040  # appear in large groups normally
G_GENO     = 0x0020  # can be genocided
G_NOCORPSE = 0x0010  # no corpse left ever

# ---------------------------------------------------------------------------
# Sound constants (MS_*)  (from monflag.h)
# ---------------------------------------------------------------------------
MS_SILENT   =  0
MS_BARK     =  1
MS_MEW      =  2
MS_ROAR     =  3
MS_BELLOW   =  4
MS_GROWL    =  5
MS_SQEEK    =  6
MS_SQAWK    =  7
MS_CHIRP    =  8
MS_HISS     =  9
MS_BUZZ     = 10
MS_GRUNT    = 11
MS_NEIGH    = 12
MS_MOO      = 13
MS_WAIL     = 14
MS_GURGLE   = 15
MS_BURBLE   = 16
MS_TRUMPET  = 17
MS_SHRIEK   = 18
MS_BONES    = 19
MS_LAUGH    = 20
MS_MUMBLE   = 21
MS_IMITATE  = 22
MS_WERE     = 23
MS_ORC      = 24
MS_HUMANOID = 25
MS_ARREST   = 26
MS_SOLDIER  = 27
MS_GUARD    = 28
MS_DJINNI   = 29
MS_NURSE    = 30
MS_SEDUCE   = 31
MS_VAMPIRE  = 32
MS_BRIBE    = 33
MS_CUSS     = 34
MS_RIDER    = 35
MS_LEADER   = 36
MS_NEMESIS  = 37
MS_GUARDIAN = 38
MS_SELL     = 39
MS_ORACLE   = 40
MS_PRIEST   = 41
MS_SPELL    = 42
MS_BOAST    = 43
MS_GROAN    = 44

# ---------------------------------------------------------------------------
# Size constants (MZ_*)  (from monflag.h)
# ---------------------------------------------------------------------------
MZ_TINY     = 0
MZ_SMALL    = 1
MZ_MEDIUM   = 2
MZ_HUMAN    = 2   # alias
MZ_LARGE    = 3
MZ_HUGE     = 4
MZ_GIGANTIC = 7

# ---------------------------------------------------------------------------
# M1_* flags  (from monflag.h)
# ---------------------------------------------------------------------------
M1_FLY         = 0x00000001
M1_SWIM        = 0x00000002
M1_AMORPHOUS   = 0x00000004
M1_WALLWALK    = 0x00000008
M1_CLING       = 0x00000010
M1_TUNNEL      = 0x00000020
M1_NEEDPICK    = 0x00000040
M1_CONCEAL     = 0x00000080
M1_HIDE        = 0x00000100
M1_AMPHIBIOUS  = 0x00000200
M1_BREATHLESS  = 0x00000400
M1_NOTAKE      = 0x00000800
M1_NOEYES      = 0x00001000
M1_NOHANDS     = 0x00002000
M1_NOLIMBS     = 0x00006000
M1_NOHEAD      = 0x00008000
M1_MINDLESS    = 0x00010000
M1_HUMANOID    = 0x00020000
M1_ANIMAL      = 0x00040000
M1_SLITHY      = 0x00080000
M1_UNSOLID     = 0x00100000
M1_THICK_HIDE  = 0x00200000
M1_OVIPAROUS   = 0x00400000
M1_REGEN       = 0x00800000
M1_SEE_INVIS   = 0x01000000
M1_TPORT       = 0x02000000
M1_TPORT_CNTRL = 0x04000000
M1_ACID        = 0x08000000
M1_POIS        = 0x10000000
M1_CARNIVORE   = 0x20000000
M1_HERBIVORE   = 0x40000000
M1_OMNIVORE    = 0x60000000
M1_METALLIVORE = 0x80000000

# ---------------------------------------------------------------------------
# M2_* flags  (from monflag.h)
# ---------------------------------------------------------------------------
M2_NOPOLY      = 0x00000001
M2_UNDEAD      = 0x00000002
M2_WERE        = 0x00000004
M2_HUMAN       = 0x00000008
M2_ELF         = 0x00000010
M2_DWARF       = 0x00000020
M2_GNOME       = 0x00000040
M2_ORC         = 0x00000080
M2_DEMON       = 0x00000100
M2_MERC        = 0x00000200
M2_LORD        = 0x00000400
M2_PRINCE      = 0x00000800
M2_MINION      = 0x00001000
M2_GIANT       = 0x00002000
M2_SHAPESHIFTER= 0x00004000
M2_MALE        = 0x00010000
M2_FEMALE      = 0x00020000
M2_NEUTER      = 0x00040000
M2_PNAME       = 0x00080000
M2_HOSTILE     = 0x00100000
M2_PEACEFUL    = 0x00200000
M2_DOMESTIC    = 0x00400000
M2_WANDER      = 0x00800000
M2_STALK       = 0x01000000
M2_NASTY       = 0x02000000
M2_STRONG      = 0x04000000
M2_ROCKTHROW   = 0x08000000
M2_GREEDY      = 0x10000000
M2_JEWELS      = 0x20000000
M2_COLLECT     = 0x40000000
M2_MAGIC       = 0x80000000

# ---------------------------------------------------------------------------
# M3_* flags  (from monflag.h)
# ---------------------------------------------------------------------------
M3_WANTSAMUL   = 0x0001
M3_WANTSBELL   = 0x0002
M3_WANTSBOOK   = 0x0004
M3_WANTSCAND   = 0x0008
M3_WANTSARTI   = 0x0010
M3_WANTSALL    = 0x001f
M3_WAITFORU    = 0x0040
M3_CLOSE       = 0x0080
M3_COVETOUS    = 0x001f
M3_INFRAVISION = 0x0100
M3_INFRAVISIBLE= 0x0200
M3_DISPLACES   = 0x0400

# ---------------------------------------------------------------------------
# Weight constants (from weight.h)
# ---------------------------------------------------------------------------
WT_ETHEREAL    = 0
WT_JELLY       = 50
WT_NYMPH       = 600
WT_ELF         = 800
WT_HUMAN       = 1450
WT_BABY_DRAGON = 1500
WT_DRAGON      = 4500

# ---------------------------------------------------------------------------
# Colour aliases (from color.h)
# ---------------------------------------------------------------------------
CLR_BLACK          =  0
CLR_RED            =  1
CLR_GREEN          =  2
CLR_BROWN          =  3
CLR_BLUE           =  4
CLR_MAGENTA        =  5
CLR_CYAN           =  6
CLR_GRAY           =  7
NO_COLOR           =  8
CLR_ORANGE         =  9
CLR_BRIGHT_GREEN   = 10
CLR_YELLOW         = 11
CLR_BRIGHT_BLUE    = 12
CLR_BRIGHT_MAGENTA = 13
CLR_BRIGHT_CYAN    = 14
CLR_WHITE          = 15

HI_LORD     = CLR_MAGENTA        # lord-tier monsters
HI_DOMESTIC = CLR_WHITE          # domestic animals / class chars
HI_OVERLORD = CLR_BRIGHT_MAGENTA # death/riders
HI_METAL    = CLR_CYAN
HI_GOLD     = CLR_YELLOW
HI_LEATHER  = CLR_BROWN
HI_WOOD     = CLR_BROWN
HI_PAPER    = CLR_WHITE
HI_GLASS    = CLR_BRIGHT_CYAN
HI_ZAP      = CLR_BRIGHT_BLUE
DRAGON_SILVER = CLR_BRIGHT_CYAN

# A_NONE from align.h
A_NONE = -128


# ---------------------------------------------------------------------------
# MONSTERS — all canonical entries (Wave 2 complete)
# Source: vendor/nethack/include/monsters.h
# Excludes: #if 0 / #ifdef CHARON / #ifdef MAIL_STRUCTURES blocks
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Chunk imports (Wave 2 — populated by 6 parallel sub-agents).
# ---------------------------------------------------------------------------
from Nethax.nethax.constants.monster_entries import (
    chunk1 as _chunk1,
    chunk2 as _chunk2,
    chunk3 as _chunk3,
    chunk4 as _chunk4,
    chunk5 as _chunk5,
    chunk6 as _chunk6,
)


MONSTERS: Tuple[MonsterEntry, ...] = (
    # Wave 2: aggregated from chunked agent output. Each chunk handles a
    # contiguous range of vendor/nethack/include/monsters.h MON() entries.
    # 381 active monster entries — matches vendor NLE monst.c exactly.
    *_chunk1.ENTRIES,
    *_chunk2.ENTRIES,
    *_chunk3.ENTRIES,
    *_chunk4.ENTRIES,
    *_chunk5.ENTRIES,
    *_chunk6.ENTRIES,
)


# ---------------------------------------------------------------------------
# TODO items
# ---------------------------------------------------------------------------
# Wave 2: populate the remaining ~384 entries from monsters.h
#         (entries 10 through NUMMONS-1: cockatrice through special PMs)
# Wave 3: glyph mapping integration — map MonsterSymbol → tile index
# Wave 4: spawn weights, difficulty-based generation logic
