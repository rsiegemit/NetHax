"""NLE action enumerations and action tuples.

Defines every action the NLE environment can receive as input.  The
`Action` enum merges all sub-classes into a single flat namespace; the
`ACTIONS` tuple matches the ordering used by NLE's `nle.nethack.ACTIONS`
exactly, and `USEFUL_ACTIONS` is the 95-action RL-friendly subset.

Canonical source: vendor/nle/nle/nethack/actions.py:215-222
Status: Wave 1 stub
"""

import enum


# ---------------------------------------------------------------------------
# Helper functions (mirrored from NLE for documentation clarity)
# ---------------------------------------------------------------------------

def _M(c: int | str) -> int:
    """Meta-key modifier: sets bit 7."""
    if isinstance(c, str):
        c = ord(c)
    return 0x80 | c


def _C(c: int | str) -> int:
    """Control-key modifier: masks to low 5 bits."""
    if isinstance(c, str):
        c = ord(c)
    return 0x1F & c


# ---------------------------------------------------------------------------
# Sub-enums (matching NLE exactly)
# ---------------------------------------------------------------------------

class TextCharacters(enum.IntEnum):
    PLUS   = ord("+")   # Also SEESPELLS.
    MINUS  = ord("-")
    SPACE  = ord(" ")
    APOS   = ord("'")
    QUOTE  = ord('"')   # Also SEEAMULET.
    NUM_0  = ord("0")
    NUM_1  = ord("1")
    NUM_2  = ord("2")
    NUM_3  = ord("3")
    NUM_4  = ord("4")
    NUM_5  = ord("5")
    NUM_6  = ord("6")
    NUM_7  = ord("7")
    NUM_8  = ord("8")
    NUM_9  = ord("9")
    DOLLAR = ord("$")   # Also SEEGOLD.


class CompassCardinalDirection(enum.IntEnum):
    N = ord("k")
    E = ord("l")
    S = ord("j")
    W = ord("h")


class CompassIntercardinalDirection(enum.IntEnum):
    NE = ord("u")
    SE = ord("n")
    SW = ord("b")
    NW = ord("y")


# Merged compass (cardinal + intercardinal) — 8 members.
CompassDirection = enum.IntEnum(
    "CompassDirection",
    {
        **CompassCardinalDirection.__members__,
        **CompassIntercardinalDirection.__members__,
    },
)


class CompassCardinalDirectionLonger(enum.IntEnum):
    N = ord("K")
    E = ord("L")
    S = ord("J")
    W = ord("H")


class CompassIntercardinalDirectionLonger(enum.IntEnum):
    NE = ord("U")
    SE = ord("N")
    SW = ord("B")
    NW = ord("Y")


# Merged longer compass — 8 members.
CompassDirectionLonger = enum.IntEnum(
    "CompassDirectionLonger",
    {
        **CompassCardinalDirectionLonger.__members__,
        **CompassIntercardinalDirectionLonger.__members__,
    },
)


class MiscDirection(enum.IntEnum):
    UP   = ord("<")   # go up a staircase
    DOWN = ord(">")   # go down a staircase
    WAIT = ord(".")   # rest one move / apply to self


class MiscAction(enum.IntEnum):
    MORE = ord("\r")  # read the next message


class UnsafeActions(enum.IntEnum):
    # These result in undesirable behaviour in RL environments.
    HELP    = ord("?")   # give a help message
    PREVMSG = _C("p")    # view recent game messages


class Command(enum.IntEnum):
    EXTCMD     = ord("#")   # perform an extended command
    EXTLIST    = _M("?")    # list all extended commands
    ADJUST     = _M("a")    # adjust inventory letters
    ANNOTATE   = _M("A")    # name current level
    APPLY      = ord("a")   # apply (use) a tool (pick-axe, key, lamp...)
    ATTRIBUTES = _C("x")    # show your attributes
    AUTOPICKUP = ord("@")   # toggle the pickup option on/off
    CALL       = ord("C")   # call (name) something
    CAST       = ord("Z")   # zap (cast) a spell
    CHAT       = _M("c")    # talk to someone
    CLOSE      = ord("c")   # close a door
    CONDUCT    = _M("C")    # list voluntary challenges you have maintained
    DIP        = _M("d")    # dip an object into something
    DROP       = ord("d")   # drop an item
    DROPTYPE   = ord("D")   # drop specific item types
    EAT        = ord("e")   # eat something
    ENGRAVE    = ord("E")   # engrave writing on the floor
    ENHANCE    = _M("e")    # advance or check weapon and spell skills
    ESC        = _C("[")    # escape from the current query/action
    FIGHT      = ord("F")   # Prefix: force fight even if you don't see a monster
    FIRE       = ord("f")   # fire ammunition from quiver
    FORCE      = _M("f")    # force a lock
    GLANCE     = ord(";")   # show what type of thing a map symbol corresponds to
    HISTORY    = ord("V")   # show long version and game history
    INVENTORY  = ord("i")   # show your inventory
    INVENTTYPE = ord("I")   # inventory specific item types
    INVOKE     = _M("i")    # invoke an object's special powers
    JUMP       = _M("j")    # jump to another location
    KICK       = _C("d")    # kick something
    KNOWN      = ord("\\")  # show what object types have been discovered
    KNOWNCLASS = ord("`")   # show discovered types for one class of objects
    LOOK       = ord(":")   # look at what is here
    LOOT       = _M("l")    # loot a box on the floor
    MONSTER    = _M("m")    # use monster's special ability
    MOVE       = ord("m")   # Prefix: move without picking up objects/fighting
    MOVEFAR    = ord("M")   # Prefix: run without picking up objects/fighting
    OFFER      = _M("o")    # offer a sacrifice to the gods
    OPEN       = ord("o")   # open a door
    OPTIONS    = ord("O")   # show option settings, possibly change them
    OVERVIEW   = _C("o")    # show a summary of the explored dungeon
    PAY        = ord("p")   # pay your shopping bill
    PICKUP     = ord(",")   # pick up things at the current location
    PRAY       = _M("p")    # pray to the gods for help
    PUTON      = ord("P")   # put on an accessory (ring, amulet, etc)
    QUAFF      = ord("q")   # quaff (drink) something
    QUIT       = _M("q")    # exit without saving current game
    QUIVER     = ord("Q")   # select ammunition for quiver
    READ       = ord("r")   # read a scroll or spellbook
    REDRAW     = _C("r")    # redraw screen
    REMOVE     = ord("R")   # remove an accessory (ring, amulet, etc)
    RIDE       = _M("R")    # mount or dismount a saddled steed
    RUB        = _M("r")    # rub a lamp or a stone
    RUSH       = ord("g")   # Prefix: rush until something interesting is seen
    RUSH2      = ord("G")   # Prefix: rush until something interesting is seen
    SAVE       = ord("S")   # save the game and exit
    SEARCH     = ord("s")   # search for traps and secret doors
    SEEALL     = ord("*")   # show all equipment in use
    SEEAMULET  = ord('"')   # show the amulet currently worn
    SEEARMOR   = ord("[")   # show the armor currently worn
    SEEGOLD    = ord("$")   # count your gold
    SEERINGS   = ord("=")   # show the ring(s) currently worn
    SEESPELLS  = ord("+")   # list and reorder known spells
    SEETOOLS   = ord("(")   # show the tools currently in use
    SEETRAP    = ord("^")   # show the type of adjacent trap
    SEEWEAPON  = ord(")")   # show the weapon currently wielded
    SHELL      = ord("!")   # do a shell escape (not enabled in NLE build)
    SIT        = _M("s")    # sit down
    SWAP       = ord("x")   # swap wielded and secondary weapons
    TAKEOFF    = ord("T")   # take off one piece of armor
    TAKEOFFALL = ord("A")   # remove all armor
    TELEPORT   = _C("t")    # teleport around the level
    THROW      = ord("t")   # throw something
    TIP        = _M("T")    # empty a container
    TRAVEL     = ord("_")   # travel to a specific location on the map
    TURN       = _M("t")    # turn undead away
    TWOWEAPON  = ord("X")   # toggle two-weapon combat
    UNTRAP     = _M("u")    # untrap something
    VERSION    = _M("v")    # list compile time options for this version of NetHack
    VERSIONSHORT = ord("v") # show version
    WEAR       = ord("W")   # wear a piece of armor
    WHATDOES   = ord("&")   # tell what a command does
    WHATIS     = ord("/")   # show what type of thing a symbol corresponds to
    WIELD      = ord("w")   # wield (put in use) a weapon
    WIPE       = _M("w")    # wipe off your face
    ZAP        = ord("z")   # zap a wand


# ---------------------------------------------------------------------------
# Flat `Action` enum merging every playable sub-enum
# ---------------------------------------------------------------------------

Action = enum.IntEnum(
    "Action",
    {
        **{f"COMPASS_{k}": v for k, v in CompassDirection.__members__.items()},
        **{f"COMPASSLONG_{k}": v for k, v in CompassDirectionLonger.__members__.items()},
        **{k: v for k, v in MiscDirection.__members__.items()},
        **{k: v for k, v in MiscAction.__members__.items()},
        **{k: v for k, v in Command.__members__.items()},
        **{f"TEXT_{k}": v for k, v in TextCharacters.__members__.items()},
    },
)

# ---------------------------------------------------------------------------
# ACTIONS tuple — matches NLE's nle.nethack.ACTIONS ordering exactly.
# Canonical source: vendor/nle/nle/nethack/actions.py:215-222
# ---------------------------------------------------------------------------

ACTIONS: tuple = tuple(
    list(CompassDirection)
    + list(CompassDirectionLonger)
    + list(MiscDirection)
    + list(MiscAction)
    + list(Command)
    + list(TextCharacters)
)

N_ACTIONS: int = len(ACTIONS)
# Verified against vendor/nle/nle/nethack/actions.py at NLE HEAD:
#   8 CompassDir + 8 CompassDirLonger + 3 MiscDir + 1 MiscAction + 85 Command + 16 TextChars = 121.
# The earlier "119" figure was an audit error; the canonical NLE count is 121.
assert N_ACTIONS == 121, (
    f"Unexpected action count {N_ACTIONS}; canonical NLE total is 121. "
    "Check for IntEnum duplicate-value aliases in Command."
)

# ---------------------------------------------------------------------------
# NON_RL_ACTIONS — actions excluded from USEFUL_ACTIONS
# ---------------------------------------------------------------------------

_NON_RL_ACTIONS = (
    Command.ANNOTATE,
    Command.AUTOPICKUP,
    Command.CONDUCT,
    Command.EXTCMD,
    Command.EXTLIST,
    Command.GLANCE,
    Command.HISTORY,
    Command.KNOWN,
    Command.KNOWNCLASS,
    Command.OPTIONS,
    Command.OVERVIEW,
    Command.TELEPORT,
    Command.QUIT,
    Command.REDRAW,
    Command.SAVE,
    Command.SEEALL,
    Command.TRAVEL,
    Command.VERSION,
    Command.WHATDOES,
    Command.WHATIS,
)

# USEFUL_ACTIONS: 101-action RL subset.
# Matches vendor/nle: drops only the 20 NON_RL_ACTIONS; keeps all TextCharacters.
_useful = [a for a in ACTIONS if a not in _NON_RL_ACTIONS]
USEFUL_ACTIONS: tuple = tuple(_useful)
del _useful

# ---------------------------------------------------------------------------
# TODO (Wave 2+):
#   - Expose WizardCommand enum if wizard-mode support is added
#   - Consider exposing UnsafeActions for debugging environments
# ---------------------------------------------------------------------------
