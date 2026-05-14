"""NLE bottom-line stats (blstats) indices and condition bitmasks.

The `blstats` observation is a 27-element int32 array.  Each index
corresponds to one bottom-line stat field as defined in NLE's C header.
The `BL_MASK_*` constants are condition bitmasks applied to `blstats[BL_CONDITION]`.

Canonical source: vendor/nle/include/nleobs.h:16-43,
                  vendor/nethack/include/botl.h:107-134
Status: Wave 1 stub
"""

# ---------------------------------------------------------------------------
# blstats indices (from vendor/nle/include/nleobs.h:17-43)
# ---------------------------------------------------------------------------

BL_X         = 0
BL_Y         = 1
BL_STR25     = 2   # strength 3..25
BL_STR125    = 3   # strength 3..125
BL_DEX       = 4
BL_CON       = 5
BL_INT       = 6
BL_WIS       = 7
BL_CHA       = 8
BL_SCORE     = 9
BL_HP        = 10
BL_HPMAX     = 11
BL_DEPTH     = 12
BL_GOLD      = 13
BL_ENE       = 14
BL_ENEMAX    = 15
BL_AC        = 16
BL_HD        = 17  # monster level / hit-dice
BL_XP        = 18  # experience level
BL_EXP       = 19  # experience points
BL_TIME      = 20
BL_HUNGER    = 21  # hunger state
BL_CAP       = 22  # carrying capacity
BL_DNUM      = 23
BL_DLEVEL    = 24
BL_CONDITION = 25  # condition bitmask (see BL_MASK_* below)
BL_ALIGN     = 26

N_BLSTATS: int = 27

# ---------------------------------------------------------------------------
# Condition bitmasks for blstats[BL_CONDITION]
# Source: vendor/nethack/include/botl.h:107-134
#
# NOTE: The NLE pynethack.cc:530-544 exposes a subset of these.
# The full list from botl.h is included here for completeness.
# TODO(wave 2): cross-check that pynethack.cc exposed values match these.
# ---------------------------------------------------------------------------

BL_MASK_BAREH    = 0x00000001  # bare-handed
BL_MASK_BLIND    = 0x00000002  # blinded
BL_MASK_BUSY     = 0x00000004  # busy
BL_MASK_CONF     = 0x00000008  # confused
BL_MASK_DEAF     = 0x00000010  # deaf
BL_MASK_ELF_IRON = 0x00000020  # elf held by iron
BL_MASK_FLY      = 0x00000040  # flying
BL_MASK_FOODPOIS = 0x00000080  # food-poisoned
BL_MASK_GLOWHANDS = 0x00000100  # glowing hands
BL_MASK_GRAB     = 0x00000200  # grabbed
BL_MASK_HALLU    = 0x00000400  # hallucinating
BL_MASK_HELD     = 0x00000800  # held
BL_MASK_ICY      = 0x00001000  # icy
BL_MASK_INLAVA   = 0x00002000  # in lava
BL_MASK_LEV      = 0x00004000  # levitating
BL_MASK_PARLYZ   = 0x00008000  # paralyzed
BL_MASK_RIDE     = 0x00010000  # riding a steed
BL_MASK_SLEEPING = 0x00020000  # sleeping
BL_MASK_SLIME    = 0x00040000  # turning to slime
BL_MASK_SLIPPERY = 0x00080000  # slippery
BL_MASK_STONE    = 0x00100000  # turning to stone
BL_MASK_STRNGL   = 0x00200000  # strangling
BL_MASK_STUN     = 0x00400000  # stunned
BL_MASK_SUBMERGED = 0x00800000  # submerged
BL_MASK_TERMILL  = 0x01000000  # terminally ill
BL_MASK_TETHERED = 0x02000000  # tethered
BL_MASK_TRAPPED  = 0x04000000  # trapped
BL_MASK_UNCONSC  = 0x08000000  # unconscious

BL_MASK_BITS: int = 30  # number of mask bits that can be set (botl.h:137)

# ---------------------------------------------------------------------------
# TODO (Wave 2+):
#   - Add BL_HUNGER_* constants (HUNGRY=1, WEAK=2, FAINTING=3, FAINTED=4,
#     SATIATED=5, OVERSATIATED=6) for decoding BL_HUNGER.
#   - Add BL_ALIGN_* constants (LAWFUL, NEUTRAL, CHAOTIC).
#   - Add BL_CAP_* constants (UNENCUMBERED=0 .. OVERTAXED=5).
#   - Add helper `decode_condition(cond_int) -> list[str]` utility.
# ---------------------------------------------------------------------------
