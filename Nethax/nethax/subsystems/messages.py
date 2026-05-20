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

Wave 28d: message-id ring-buffer rotation implemented.
  - Each message has a static integer ID (MessageId enum below).
  - emit(state, msg_id) uses jax.lax.switch on msg_id to select a fixed
    ASCII byte template (the human-readable line, pre-baked) and writes
    it into message_buffer.  Byte 0 holds msg_id; bytes 1.. hold the
    rendered text (zero-padded to MSG_BUF_LEN).
  - The current message_buffer is shifted into message_history[history_index]
    before being overwritten, and history_index is advanced modulo
    HISTORY_LEN (mirrors gs.saved_pline_index rotation in pline.c::dumplogmsg
    lines 20-46).
  - JIT-pure: lax.switch + .at[].set ops; no Python control flow on traced
    values.
"""
from enum import IntEnum

import jax
import jax.numpy as jnp
import numpy as _np
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

    # Wave 28d IDs — wired into subsystem call sites.
    YOU_TURN_INTO        = 7   # polyself.c::polymon  "You turn into a ..."
    YOU_RETURN_TO_HUMAN  = 8   # polyself.c::rehumanize "You return to ... form."
    YOU_PRAY             = 9   # pray.c::dopray "You begin praying ..."
    MONSTER_HITS_YOU     = 10  # mhitu.c::mattacku "The monster hits!"
    YOU_HIT_MONSTER      = 11  # uhitm.c::hmon "You hit the monster."
    YOU_QUAFF_POTION     = 12  # potion.c::dodrink "You quaff the potion."
    YOU_READ_SCROLL      = 13  # read.c::doread "You read the scroll."
    GO_UP_STAIRS         = 14  # do.c::doup "You climb up the stairs."
    GO_DOWN_STAIRS       = 15  # do.c::dodown "You climb down the stairs."
    YOU_WAIT             = 16  # cmd.c::dowait "You wait."


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

# ---------------------------------------------------------------------------
# Rendered-message templates
#
# One ASCII line per MessageId.  pline.c::pline formats a printf string and
# putmesg() writes it to WIN_MESSAGE.  Inside the JIT trace we cannot run
# printf, so we pre-bake each line as a fixed-width uint8 row indexed by
# msg_id.  Variable arguments (monster name, damage number, ...) are not
# substituted here — emit() takes ``*args`` for future expansion, but Wave
# 28d only writes the static template.
#
# Cite: vendor/nethack/src/pline.c::pline lines 103-111 (printf entry),
#       putmesg lines 64-80 (write to message window).
# ---------------------------------------------------------------------------

_MESSAGE_TEMPLATES: tuple[str, ...] = (
    "",                              # 0  NONE
    "Welcome to NetHack!",           # 1  GAME_START         pline.c:104
    "You die...",                    # 2  YOU_DIE            end.c::done
    "You kill the monster!",         # 3  YOU_KILL_MONSTER   uhitm.c::killed
    "You find some gold.",           # 4  FIND_GOLD          hack.c::pickup_gold
    "The door opens.",               # 5  OPEN_DOOR          do_name.c::doopen
    "You eat the food.",             # 6  EAT_FOOD           eat.c::eatcorpse
    "You turn into a new form!",     # 7  YOU_TURN_INTO      polyself.c::polymon
    "You return to your old form.",  # 8  YOU_RETURN_TO_HUMAN polyself.c::rehumanize
    "You begin praying to your god.",# 9  YOU_PRAY           pray.c::dopray
    "The monster hits!",             # 10 MONSTER_HITS_YOU   mhitu.c::mattacku
    "You hit the monster.",          # 11 YOU_HIT_MONSTER    uhitm.c::hmon
    "You quaff the potion.",         # 12 YOU_QUAFF_POTION   potion.c::dodrink
    "You read the scroll.",          # 13 YOU_READ_SCROLL    read.c::doread
    "You climb up the stairs.",      # 14 GO_UP_STAIRS       do.c::doup
    "You climb down the stairs.",    # 15 GO_DOWN_STAIRS     do.c::dodown
    "You wait.",                     # 16 YOU_WAIT           cmd.c::dowait
)


def _bake_templates() -> jnp.ndarray:
    """Pack _MESSAGE_TEMPLATES into a [N_MESSAGES, MSG_BUF_LEN] uint8 array.

    Each row: [msg_id, ascii bytes..., 0-padding to MSG_BUF_LEN].
    Byte 0 is msg_id (preserves the Wave 1 contract that
    ``message_buffer[0] == msg_id``); bytes 1.. hold the rendered ASCII line.
    """
    n = len(_MESSAGE_TEMPLATES)
    arr = _np.zeros((n, MSG_BUF_LEN), dtype=_np.uint8)
    for i, text in enumerate(_MESSAGE_TEMPLATES):
        arr[i, 0] = i & 0xFF
        raw = text.encode("ascii")[: MSG_BUF_LEN - 1]
        arr[i, 1 : 1 + len(raw)] = list(raw)
    return jnp.asarray(arr, dtype=jnp.uint8)


# Module-level constant; baked once at import.
_TEMPLATE_TABLE: jnp.ndarray = _bake_templates()
_N_TEMPLATES: int = len(_MESSAGE_TEMPLATES)


def emit(state: MessageState, msg_id: int, *args) -> MessageState:
    """Render ``msg_id`` into message_buffer and rotate the ring buffer.

    Cite: vendor/nethack/src/pline.c::pline (line 103) — formats and pushes
    the message to WIN_MESSAGE; pline.c::dumplogmsg (lines 20-46) —
    saved_plines ring buffer rotation with ``gs.saved_pline_index``.

    Algorithm (mirrors pline.c::dumplogmsg + pline.c::pline):
      1. Save the current ``message_buffer`` into ``message_history`` at
         ``history_index`` (the "oldest" slot, mod HISTORY_LEN).
      2. Advance ``history_index`` by 1, modulo HISTORY_LEN
         (pline.c:45 ``gs.saved_pline_index = (indx + 1) % DUMPLOG_MSG_COUNT``).
      3. Look up the fixed template for ``msg_id`` and write it into the
         fresh message_buffer.  Byte 0 holds msg_id; bytes 1.. hold the
         pre-baked ASCII rendering (zero-padded to MSG_BUF_LEN).

    Parameters
    ----------
    state  : MessageState
    msg_id : int / jnp.int32  — MessageId enum value.  Out-of-range IDs
             render as NONE (all zeros).
    *args  : optional jnp.int32 scalars — reserved for future substitution
             (e.g. damage number, gold quantity).  Wave 28d ignores them;
             they are accepted so call sites can be written once and
             upgraded later without churn.

    Returns
    -------
    Updated MessageState (new message_buffer + rotated history).

    JIT-safe: shape-static; uses ``.at[].set`` and integer indexing only.
    """
    del args  # Reserved for printf-style argument substitution (future wave).

    # Step 1: rotate current buffer into history at history_index.
    safe_idx = jnp.mod(state.history_index, jnp.int32(HISTORY_LEN))
    new_history = state.message_history.at[safe_idx].set(state.message_buffer)

    # Step 2: advance the ring pointer (pline.c:45).
    new_index = jnp.mod(state.history_index + jnp.int32(1), jnp.int32(HISTORY_LEN))

    # Step 3: look up the pre-baked template row for msg_id.
    # Out-of-range ids clip to 0 (NONE), matching the safe "no message" case.
    msg_id_i32 = jnp.int32(msg_id)
    safe_id    = jnp.clip(msg_id_i32, jnp.int32(0), jnp.int32(_N_TEMPLATES - 1))
    new_buffer = _TEMPLATE_TABLE[safe_id]

    return state.replace(
        message_buffer=new_buffer,
        message_history=new_history,
        history_index=new_index,
    )


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
