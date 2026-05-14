"""Message subsystem — in-game message buffer and scrollback history.

Canonical source:
  vendor/nethack/src/pline.c — pline(), vpline(), putmesg(), dumplogmsg();
                                the core message-printing pipeline that
                                formats a string and pushes it to the window
                                manager (pline.c lines 1-130).
                                Key constants: BUFSZ=256 (hack.h),
                                DUMPLOG_MSG_COUNT (saved_plines ring buffer,
                                pline.c lines 20-46).

Design departure
----------------
NetHack's pline() takes a printf-style format string and writes directly to
the terminal window.  Nethax cannot do this inside jit-compiled steps because
JAX traces require static shapes and no Python-side side-effects at trace
time.

Wave 4 plan: replace emit() with a message-id system.
  - Each message has a static integer ID (MessageId enum below).
  - Dynamic arguments (monster name, item name, damage number) are stored as
    fixed-width integer arrays in MessageState alongside the ID.
  - The renderer reads the ID + args and formats the human-readable string
    outside the jit boundary.

Status: Wave 1 stub — MessageState dataclass + no-op emit + clear_message.
emit() returns state unchanged; the Wave 4 message-id system will replace it.

TODO (Wave 4):
  - Replace emit(state, message: str) with
    emit(state, msg_id: MessageId, *args: jnp.int32) -> MessageState.
  - Implement ring-buffer rotation in emit: write msg_id + args into
    message_buffer, shift old buffer into message_history at history_index,
    increment history_index % HISTORY_LEN.
  - Expose get_current_message(state) -> MessageId + args tuple for renderer.
  - Grow MessageId as each subsystem comes online (one entry per pline call
    site, roughly matching pline.c call sites across the codebase).
"""
from enum import IntEnum

import jax.numpy as jnp
from flax import struct


# ---------------------------------------------------------------------------
# Buffer geometry (mirrors pline.c / hack.h constants)
# ---------------------------------------------------------------------------

MSG_BUF_LEN: int = 256   # BUFSZ in hack.h — max chars in one message line
HISTORY_LEN: int = 20    # pline.c saved_plines ring-buffer depth


# ---------------------------------------------------------------------------
# Message ID enumeration (placeholder — grows each wave)
# ---------------------------------------------------------------------------

class MessageId(IntEnum):
    """Static message identifiers for the JIT-friendly message system.

    Wave 1: bare minimum set.  Each subsequent wave adds IDs for newly
    implemented pline() call sites.  Keep IDs stable (never renumber);
    append new entries at the end of each wave block.

    Wave 1 IDs
    ----------
    """
    NONE              = 0   # sentinel — no message this turn
    GAME_START        = 1   # "Welcome to NetHack!"
    YOU_DIE           = 2   # "You die..."  (end.c: really_done)
    YOU_KILL_MONSTER  = 3   # "You kill the <monster>!"
    FIND_GOLD         = 4   # "You find <n> gold pieces."
    OPEN_DOOR         = 5   # "The door opens."
    EAT_FOOD          = 6   # "You eat the <food>."


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@struct.dataclass
class MessageState:
    """In-game message buffer and scrollback ring.

    Fields
    ------
    message_buffer  : current message line as uint8 bytes, shape (MSG_BUF_LEN,).
                      Null-terminated; unused bytes are 0.
                      Mirrors the single-line display at the top of the
                      NetHack terminal (pline.c: putmesg destination).
    message_history : ring buffer of previous messages, shape (HISTORY_LEN,
                      MSG_BUF_LEN), uint8.  Oldest entry is overwritten when
                      the buffer is full (pline.c: saved_plines / dumplogmsg).
    history_index   : next write position in message_history (int32, wraps
                      modulo HISTORY_LEN).
    """
    message_buffer:  jnp.ndarray  # (MSG_BUF_LEN,)          uint8
    message_history: jnp.ndarray  # (HISTORY_LEN, MSG_BUF_LEN)  uint8
    history_index:   jnp.ndarray  # scalar                   int32

    @classmethod
    def default(cls) -> "MessageState":
        """Return a zeroed MessageState (empty buffers) for a new game."""
        return cls(
            message_buffer=jnp.zeros((MSG_BUF_LEN,), dtype=jnp.uint8),
            message_history=jnp.zeros((HISTORY_LEN, MSG_BUF_LEN), dtype=jnp.uint8),
            history_index=jnp.int32(0),
        )


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def emit(state: MessageState, message: str) -> MessageState:
    """Queue a message for display (Wave 1 no-op).

    Parameters
    ----------
    message : Human-readable string — accepted for API compatibility but
              ignored in Wave 1 because string encoding inside a jit trace
              requires static shapes.

    Returns
    -------
    state unchanged.

    Wave 4 plan
    -----------
    Replace signature with emit(state, msg_id: MessageId, *args) and:
      1. Rotate current message_buffer into message_history at history_index.
      2. Increment history_index % HISTORY_LEN.
      3. Encode msg_id + args into message_buffer (fixed-width int layout).
    The renderer decodes msg_id + args back to a human string outside jit.
    """
    return state


def clear_message(state: MessageState) -> MessageState:
    """Zero out the current message buffer (called at the start of each turn).

    Mirrors the clear_nhwindow(WIN_MESSAGE) call at the top of NetHack's
    main loop (pline.c / cmd.c).
    """
    return state.replace(
        message_buffer=jnp.zeros((MSG_BUF_LEN,), dtype=jnp.uint8)
    )


# ---------------------------------------------------------------------------
# Wave 6 Phase A — death-message generation
#
# Mirrors vendor/nethack/src/end.c::done() and the two parallel tables there:
#   * deaths[]  (end.c lines 44-50) — short past-tense cause name
#                ("died", "choked", "poisoned", "starvation", "drowning",
#                 "burning", "dissolving under the heat and pressure",
#                 "crushed", "turned to stone", "turned into slime",
#                 "genocided", "panic", "trickery", "quit", "escaped",
#                 "ascended").
#   * ends[]    (end.c lines 52-61) — "when you %s" phrasing
#                ("died", "choked", "were poisoned", "starved", "drowned",
#                 "burned", ...).
# We reproduce both tables here so the renderer (which runs outside JIT) can
# format human-readable lines.  Per design departure docs at the top of this
# module, formatting strings inside a jit trace is not possible; death_message
# is therefore a pure-Python helper invoked at game-over.
#
# Citations:
#   vendor/nethack/src/end.c::done()              — overall game-over flow.
#   vendor/nethack/src/end.c::deaths[] / ends[]   — cause-text tables.
#   vendor/nethack/src/end.c::done_object_name    — formatkiller object naming.
#   vendor/nethack/include/hack.h::game_end_types — DIED..ASCENDED integer
#                                                    values (we mirror exactly).
# ---------------------------------------------------------------------------

# Indexed by DeathCause integer value (0..15).
# Past-tense cause name for the killer-bar / RIP tombstone.
_DEATH_CAUSE_NAME = (
    "died",                                  #  0 DIED
    "choked",                                #  1 CHOKING
    "poisoned",                              #  2 POISONING
    "starvation",                            #  3 STARVING
    "drowning",                              #  4 DROWNING
    "burning",                               #  5 BURNING
    "dissolving under the heat and pressure",#  6 DISSOLVED
    "crushed",                               #  7 CRUSHING
    "turned to stone",                       #  8 STONING
    "turned into slime",                     #  9 TURNED_SLIME
    "genocided",                             # 10 GENOCIDED
    "panic",                                 # 11 PANICKED
    "trickery",                              # 12 TRICKED
    "quit",                                  # 13 QUIT
    "escaped",                               # 14 ESCAPED
    "ascended",                              # 15 ASCENDED
)

# Indexed by DeathCause integer value; "when you %s" past-tense verb phrase.
_DEATH_VERB = (
    "died",                                  #  0 DIED
    "choked",                                #  1 CHOKING
    "were poisoned",                         #  2 POISONING
    "starved",                               #  3 STARVING
    "drowned",                               #  4 DROWNING
    "burned",                                #  5 BURNING
    "dissolved in the lava",                 #  6 DISSOLVED
    "were crushed",                          #  7 CRUSHING
    "turned to stone",                       #  8 STONING
    "turned into slime",                     #  9 TURNED_SLIME
    "were genocided",                        # 10 GENOCIDED
    "panicked",                              # 11 PANICKED
    "were tricked",                          # 12 TRICKED
    "quit",                                  # 13 QUIT
    "escaped",                               # 14 ESCAPED
    "ascended",                              # 15 ASCENDED
)


def death_cause_name(cause: int) -> str:
    """Return the vendor ``deaths[cause]`` string (past-tense cause name).

    Cite: vendor/nethack/src/end.c lines 44-50.
    """
    idx = int(cause)
    if idx < 0 or idx >= len(_DEATH_CAUSE_NAME):
        return "died"
    return _DEATH_CAUSE_NAME[idx]


def death_verb(cause: int) -> str:
    """Return the vendor ``ends[cause]`` string ("when you %s" phrasing).

    Cite: vendor/nethack/src/end.c lines 52-61.
    """
    idx = int(cause)
    if idx < 0 or idx >= len(_DEATH_VERB):
        return "died"
    return _DEATH_VERB[idx]


def death_message(state, cause: int, monster_name: str | None = None) -> str:
    """Generate the end-of-game text per vendor end.c::done().

    Parameters
    ----------
    state : EnvState — final state at game-over (reads dungeon.current_level
            and scoring.final_score / scoring.score).
    cause : DeathCause integer (or int matching game_end_types).
    monster_name : Optional killer-monster name for DIED.  When ``cause`` is
            DIED and monster_name is given, the message is
            ``"Killed by a <monster> on dungeon level <N>"``.

    Returns
    -------
    A single human-readable line.  Examples:

        Killed by a giant rat on dungeon level 3
        Starved to death on dungeon level 2
        Drowned on dungeon level 4
        Burned to death on dungeon level 5
        Fell into lava on dungeon level 6
        Ascended to demigod status with 12345 points
        Quit the game on dungeon level 1
        Escaped the dungeon with 500 points

    Cites:
        vendor/nethack/src/end.c::done()              (overall flow)
        vendor/nethack/src/end.c lines 1064-1070     (no-killer-prefix logic)
        vendor/nethack/src/end.c lines 1421-1481    (ascension / escape text)
    """
    # Import here to avoid a top-level cycle (scoring depends on conduct).
    from Nethax.nethax.subsystems.scoring import DeathCause

    cause_int = int(cause)
    level     = int(state.dungeon.current_level)

    if cause_int == int(DeathCause.ASCENDED):
        score = int(state.scoring.final_score)
        if score == 0:
            score = int(state.scoring.score)
        return f"Ascended to demigod status with {score} points"

    if cause_int == int(DeathCause.ESCAPED):
        score = int(state.scoring.final_score)
        if score == 0:
            score = int(state.scoring.score)
        return f"Escaped the dungeon with {score} points"

    if cause_int == int(DeathCause.QUIT):
        return f"Quit the game on dungeon level {level}"

    if cause_int == int(DeathCause.DIED):
        # vendor: "Killed by <killer name>" — KILLED_BY_AN prefix at
        # end.c line 201; svk.killer.name is the monster name from
        # done_object_name / mon_nam.
        name = monster_name or "something"
        # Use "a/an" prefix per KILLED_BY_AN format (end.c::formatkiller).
        article = "an" if name[:1].lower() in "aeiou" else "a"
        return f"Killed by {article} {name} on dungeon level {level}"

    if cause_int == int(DeathCause.STARVING):
        return f"Starved to death on dungeon level {level}"

    if cause_int == int(DeathCause.DROWNING):
        return f"Drowned on dungeon level {level}"

    if cause_int == int(DeathCause.BURNING):
        return f"Burned to death on dungeon level {level}"

    if cause_int == int(DeathCause.DISSOLVED):
        return f"Fell into lava on dungeon level {level}"

    if cause_int == int(DeathCause.STONING):
        return f"Petrified on dungeon level {level}"

    if cause_int == int(DeathCause.TURNED_SLIME):
        return f"Turned to slime on dungeon level {level}"

    if cause_int == int(DeathCause.POISONING):
        return f"Poisoned on dungeon level {level}"

    if cause_int == int(DeathCause.CHOKING):
        return f"Choked to death on dungeon level {level}"

    if cause_int == int(DeathCause.CRUSHING):
        return f"Crushed on dungeon level {level}"

    if cause_int == int(DeathCause.GENOCIDED):
        return f"Genocided on dungeon level {level}"

    if cause_int == int(DeathCause.PANICKED):
        return f"Panicked on dungeon level {level}"

    if cause_int == int(DeathCause.TRICKED):
        return f"Tricked on dungeon level {level}"

    # Fallback — unknown cause integer.
    return f"Died on dungeon level {level}"
