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

    # Wave 30b — per-prop expiry messages from nh_timeout.
    HALLU_BORING         = 17  # potion.c:381 / timeout.c:778-783
                               # "Everything looks SO boring now."
                               # Fired when HHallucination timer ticks 1 → 0
                               # via nh_timeout HALLU case.

    # Wave 32i IDs — swallow message variants + throne/altar flavor.
    SWALLOW_DIGESTS      = 18  # mhitu.c:1336 "swallows you whole" (digests)
    SWALLOW_ENFOLDS      = 19  # mhitu.c:1337 "folds itself around you"
    SWALLOW_ENGULFS      = 20  # mhitu.c:1338 "engulfs you" (default)
    THRONE_ATTR_LOSS     = 21  # sit.c throne effect 1
    THRONE_ATTR_GAIN     = 22  # sit.c throne effect 2
    THRONE_SHOCK         = 23  # sit.c throne effect 3
    THRONE_FULL_HEAL     = 24  # sit.c throne effect 4
    THRONE_TAKE_GOLD     = 25  # sit.c throne effect 5
    THRONE_WISH          = 26  # sit.c throne effect 6
    THRONE_COURT         = 27  # sit.c throne effect 7
    THRONE_GENOCIDE      = 28  # sit.c throne effect 8
    THRONE_CURSE_ITEMS   = 29  # sit.c throne effect 9
    THRONE_MAP_CONFUSE   = 30  # sit.c throne effect 10
    THRONE_TELEPORT      = 31  # sit.c throne effect 11
    THRONE_IDENTIFY      = 32  # sit.c throne effect 12
    THRONE_CONFUSE       = 33  # sit.c throne effect 13
    ALTAR_WRATH          = 34  # pray.c::altar_wrath same-aligned penalty
    ALTAR_LUCK_LOSS      = 35  # pray.c::altar_wrath different-aligned penalty
    SPELL_FIZZLES        = 36  # spell.c:1373 "You fail to cast the spell correctly."

    # Slime per-turn dialogue (vendor/nethack/src/timeout.c::slime_dialogue
    # lines 380-443).  Five message ticks fire as Slimed counts down from 9
    # to 1 (i = (Slimed & TIMEOUT) / 2, fires when timer is odd).
    SLIME_TURNING_COLOR  = 37  # i=4 / t=9  "You are turning a little green."
    SLIME_LIMBS_OOZY     = 38  # i=3 / t=7  "Your limbs are getting oozy."
    SLIME_SKIN_PEEL      = 39  # i=2 / t=5  "Your skin begins to peel away."
    SLIME_TURNING_INTO   = 40  # i=1 / t=3  "You are turning into green slime."
    LEVI_FLOAT_LOWER     = 41  # timeout.c:348 "You float slightly lower."
    LEVI_WOBBLE          = 42  # timeout.c:349 "You wobble unsteadily in the air."

    # Game-start role-specific intro line (vendor allmain.c::welcome lines
    # 920-922):
    #     pline("%s %s, welcome to NetHack!  You are a%s.",
    #           Hello((struct monst *) 0), svp.plname, buf);
    # The leading "Hello"-equivalent varies by role (role.c::Hello lines
    # 2119-2140): "Salutations" (Knight), "Konnichi wa" (Samurai),
    # "Aloha" (Tourist), "Velkommen" (Valkyrie), "Hello" otherwise.
    ROLE_INTRO           = 43  # allmain.c:920 per-role welcome pline()

    # Wave: MiniHack skill-env reward messages.
    WAND_FEELING_SUBSIDES = 44  # zap.c:2188 pline_The("feeling subsides.")
                                # WAN_ENLIGHTENMENT zap effect (zapnodir).
    SCROLL_SEEMS_BLANK    = 45  # read.c:1266 pline("This scroll seems to be
                                # blank.")  SCR_BLANK_PAPER read effect.


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
    # YOU_KILL_MONSTER: 32-byte monster-name slot starts at offset 13
    # (after "You kill the ").  Trailing '!' is appended by emit() after
    # the trimmed name.  Cite: vendor uhitm.c::killed line ~1015 —
    # "You kill the %s%s!" formatted with mon_nam().
    "You kill the " + (" " * 32) + "!",  # 3 YOU_KILL_MONSTER uhitm.c::killed
    # FIND_GOLD: 10-byte right-aligned numeric slot at offset 9.
    # Cite: vendor hack.c::pickup_gold line ~150 — "%ld gold piece%s".
    "You find " + (" " * 10) + " gold pieces.",  # 4 FIND_GOLD hack.c
    "The door opens.",               # 5  OPEN_DOOR          do_name.c::doopen
    "You eat the food.",             # 6  EAT_FOOD           eat.c::eatcorpse
    "You turn into a new form!",     # 7  YOU_TURN_INTO      polyself.c::polymon
    "You return to your old form.",  # 8  YOU_RETURN_TO_HUMAN polyself.c::rehumanize
    "You begin praying to your god.",# 9  YOU_PRAY           pray.c::dopray
    "The monster hits!",             # 10 MONSTER_HITS_YOU   mhitu.c::mattacku
    # YOU_HIT_MONSTER: 32-byte monster-name slot starts at offset 12
    # (after "You hit the ").  '.' is appended after the trimmed name.
    # Cite: vendor uhitm.c::hmon line ~1180 — "You hit %s." with mon_nam().
    "You hit the " + (" " * 32) + ".",  # 11 YOU_HIT_MONSTER uhitm.c::hmon
    "You quaff the potion.",         # 12 YOU_QUAFF_POTION   potion.c::dodrink
    "You read the scroll.",          # 13 YOU_READ_SCROLL    read.c::doread
    "You climb up the stairs.",      # 14 GO_UP_STAIRS       do.c::doup
    "You climb down the stairs.",    # 15 GO_DOWN_STAIRS     do.c::dodown
    "You wait.",                     # 16 YOU_WAIT           cmd.c::dowait
    "Everything looks SO boring now.", # 17 HALLU_BORING     potion.c:381
    "The monster swallows you whole!", # 18 SWALLOW_DIGESTS  mhitu.c:1336
    "The monster folds itself around you!", # 19 SWALLOW_ENFOLDS  mhitu.c:1337
    "The monster engulfs you!",        # 20 SWALLOW_ENGULFS  mhitu.c:1338
    "You feel weaker.",                # 21 THRONE_ATTR_LOSS sit.c effect 1
    "You feel a surge of power.",      # 22 THRONE_ATTR_GAIN sit.c effect 2
    "A shock runs through your body!", # 23 THRONE_SHOCK     sit.c effect 3
    "You feel much better!",           # 24 THRONE_FULL_HEAL sit.c effect 4
    "Your gold disappears!",           # 25 THRONE_TAKE_GOLD sit.c effect 5
    "A wish is granted!",              # 26 THRONE_WISH      sit.c effect 6
    "The court is summoned!",          # 27 THRONE_COURT     sit.c effect 7
    "A feeling of great wisdom comes over you.", # 28 THRONE_GENOCIDE sit.c effect 8
    "You feel a malignant aura surround you.", # 29 THRONE_CURSE_ITEMS sit.c effect 9
    "Your mind is filled with images!", # 30 THRONE_MAP_CONFUSE sit.c effect 10
    "You are teleported!",             # 31 THRONE_TELEPORT  sit.c effect 11
    "Your possessions are identified.", # 32 THRONE_IDENTIFY  sit.c effect 12
    "Your vision becomes unsteady.",   # 33 THRONE_CONFUSE   sit.c effect 13
    "You feel that you have transgressed.", # 34 ALTAR_WRATH  pray.c::altar_wrath
    "You feel your luck waver.",       # 35 ALTAR_LUCK_LOSS  pray.c::altar_wrath
    "You fail to cast the spell correctly.", # 36 SPELL_FIZZLES spell.c:1373
    "You are turning a little green.",       # 37 SLIME_TURNING_COLOR timeout.c:381
    "Your limbs are getting oozy.",          # 38 SLIME_LIMBS_OOZY    timeout.c:382
    "Your skin begins to peel away.",        # 39 SLIME_SKIN_PEEL     timeout.c:383
    "You are turning into green slime.",     # 40 SLIME_TURNING_INTO  timeout.c:384
    "You float slightly lower.",             # 41 LEVI_FLOAT_LOWER    timeout.c:348
    "You wobble unsteadily in the air.",     # 42 LEVI_WOBBLE         timeout.c:349
    # 43 ROLE_INTRO — rendered dynamically by emit_role_intro(); the static
    # row is an unused placeholder kept so list index == msg_id for entries
    # appended after it.  Cite: allmain.c:920 per-role welcome pline().
    "",                                      # 43 ROLE_INTRO          allmain.c:920
    # WAN_ENLIGHTENMENT zap: vendor renders pline_The("feeling subsides.")
    # which capitalises the leading article -> "The feeling subsides."
    # Cite: vendor/nle/src/zap.c:2188 (zapnodir WAN_ENLIGHTENMENT branch).
    "The feeling subsides.",                 # 44 WAND_FEELING_SUBSIDES zap.c:2188
    # SCR_BLANK_PAPER read.  Cite: vendor/nle/src/read.c:1266.
    "This scroll seems to be blank.",        # 45 SCROLL_SEEMS_BLANK   read.c:1266
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


# ---------------------------------------------------------------------------
# Argument-slot metadata (printf-style substitution into pre-baked templates).
#
# Mirrors vendor pline.c::pline(fmt, ...) which accepts a printf-style format
# string and writes the formatted result via vsprintf into a BUFSZ buffer
# (pline.c lines 103-130).  In our JIT-compatible model we can't run printf
# at trace time, so each MessageId reserves a fixed-width byte slot inside
# its template and emit() writes the argument value(s) into that slot.
#
# Kinds:
#   _ARG_KIND_NONE    = 0 -> no substitution (template is static)
#   _ARG_KIND_NUMERIC = 1 -> right-aligned decimal ASCII into the slot
#   _ARG_KIND_MONSTER = 2 -> monster-name bytes from MONSTERS[entry_idx].name
# ---------------------------------------------------------------------------

_ARG_KIND_NONE:    int = 0
_ARG_KIND_NUMERIC: int = 1
_ARG_KIND_MONSTER: int = 2

# Slot width for the monster-name placeholder (chosen to fit the longest
# vendor monster name comfortably; truncates if longer).
_MONSTER_NAME_SLOT_WIDTH: int = 32

# Numeric slot width — fits up to 10 ASCII digits (covers a 32-bit int).
_NUMERIC_SLOT_WIDTH: int = 10


def _bake_arg_metadata():
    """Build [N_MESSAGES] arrays describing each MessageId's arg slot.

    Returns three int32 vectors:
      - kind   : _ARG_KIND_*
      - offset : byte position (into the buffer row) where the arg writes
      - width  : slot width in bytes

    The offset is computed against the template buffer layout, which is:
        buffer[0]        = msg_id
        buffer[1..1+len] = template ASCII
    so an offset of "1 + python_index_in_template" yields the buffer offset.
    """
    n = _N_TEMPLATES
    kind   = _np.zeros((n,), dtype=_np.int32)
    offset = _np.zeros((n,), dtype=_np.int32)
    width  = _np.zeros((n,), dtype=_np.int32)

    # YOU_KILL_MONSTER (id=3): "You kill the <name>!" — name starts at template
    # column 13 ("You kill the " = 13 chars), then 32-byte slot.  Buffer
    # offset = 1 (msg_id) + 13 = 14.
    kind[3]   = _ARG_KIND_MONSTER
    offset[3] = 1 + len("You kill the ")
    width[3]  = _MONSTER_NAME_SLOT_WIDTH

    # FIND_GOLD (id=4): "You find <N> gold pieces." — N at column 9.
    kind[4]   = _ARG_KIND_NUMERIC
    offset[4] = 1 + len("You find ")
    width[4]  = _NUMERIC_SLOT_WIDTH

    # YOU_HIT_MONSTER (id=11): "You hit the <name>." — name at column 12.
    kind[11]   = _ARG_KIND_MONSTER
    offset[11] = 1 + len("You hit the ")
    width[11]  = _MONSTER_NAME_SLOT_WIDTH

    return (
        jnp.asarray(kind,   dtype=jnp.int32),
        jnp.asarray(offset, dtype=jnp.int32),
        jnp.asarray(width,  dtype=jnp.int32),
    )


_ARG_KIND, _ARG_OFFSET, _ARG_WIDTH = _bake_arg_metadata()


def _bake_monster_name_table() -> jnp.ndarray:
    """Build [N_MONSTERS, _MONSTER_NAME_SLOT_WIDTH] uint8 table of names.

    Indexed by entry_idx (matches MONSTERS tuple ordering).  Each row holds
    the monster name as ASCII bytes, right-padded with spaces (' ' = 0x20)
    so that the resulting line aligns naturally in the message buffer.
    Out-of-range indices are looked up via clip-to-bounds in emit().

    Cite: vendor monst.c MON() entries — monster_entry.mname strings.
    """
    # Local import to avoid a cycle at module-import time; constants/monsters
    # itself does not depend on messages.
    from Nethax.nethax.constants.monsters import MONSTERS

    n = len(MONSTERS)
    arr = _np.full((n, _MONSTER_NAME_SLOT_WIDTH), ord(" "), dtype=_np.uint8)
    for i, m in enumerate(MONSTERS):
        raw = m.name.encode("ascii")[: _MONSTER_NAME_SLOT_WIDTH]
        arr[i, : len(raw)] = list(raw)
    return jnp.asarray(arr, dtype=jnp.uint8)


_MONSTER_NAME_TABLE: jnp.ndarray = _bake_monster_name_table()
_N_MONSTERS: int = int(_MONSTER_NAME_TABLE.shape[0])


def _digits10(value: jnp.ndarray) -> jnp.ndarray:
    """Render ``value`` (int32 scalar) as 10 right-aligned ASCII digits.

    Returned shape: (_NUMERIC_SLOT_WIDTH,) uint8.  Leading positions are
    filled with ASCII space (0x20) until the most-significant non-zero
    digit.  Negative values are rendered as ``|v|`` with a leading '-'
    sign before the first digit (clamped: very large negatives still fit
    inside 10 chars).  Mirrors vendor pline.c "%ld" formatting for the
    gold-pickup line ("%ld gold piece%s").
    """
    v_signed = jnp.int32(value)
    is_neg   = v_signed < jnp.int32(0)
    v_abs    = jnp.where(is_neg, -v_signed, v_signed).astype(jnp.int32)
    space    = jnp.uint8(0x20)  # ' '
    zero     = jnp.uint8(ord("0"))
    minus    = jnp.uint8(ord("-"))

    digits = jnp.full((_NUMERIC_SLOT_WIDTH,), space, dtype=jnp.uint8)
    # Build digits from right (column 9) to left (column 0).
    def _step(i, carry):
        digits_in, val_in = carry
        # Position from the right: column index = (W-1) - i.
        col = jnp.int32(_NUMERIC_SLOT_WIDTH - 1) - jnp.int32(i)
        d = jnp.mod(val_in, jnp.int32(10)).astype(jnp.uint8)
        # If val_in==0 AND we've already written at least one digit (i>=1)
        # then we leave a space; otherwise write the digit.  At i==0 we
        # always write at least the ones digit (so value 0 renders "0").
        already_done = (val_in == jnp.int32(0)) & (jnp.int32(i) > jnp.int32(0))
        ch = jnp.where(already_done, space, zero + d)
        digits_out = digits_in.at[col].set(ch)
        val_out    = jnp.floor_divide(val_in, jnp.int32(10))
        return digits_out, val_out

    digits, _ = jax.lax.fori_loop(0, _NUMERIC_SLOT_WIDTH, _step, (digits, v_abs))
    # If negative, place '-' just before the leading digit.  We find the
    # leading digit by scanning left-to-right for the first non-space.
    def _find_leading(i, acc):
        found, pos = acc
        is_digit = (digits[i] != space) & (digits[i] != minus)
        pos = jnp.where(~found & is_digit, jnp.int32(i), pos)
        return (found | is_digit, pos)

    _, lead = jax.lax.fori_loop(
        0, _NUMERIC_SLOT_WIDTH, _find_leading,
        (jnp.bool_(False), jnp.int32(_NUMERIC_SLOT_WIDTH - 1)),
    )
    # Position to place '-' = max(lead-1, 0).
    minus_pos = jnp.maximum(lead - jnp.int32(1), jnp.int32(0))
    digits = jnp.where(
        is_neg,
        digits.at[minus_pos].set(minus),
        digits,
    )
    return digits


def emit(state: MessageState, msg_id: int, *args) -> MessageState:
    """Render ``msg_id`` (with optional printf-style args) into the buffer.

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
      4. If the MessageId has an argument slot (see _ARG_KIND), substitute
         the first ``arg`` into the reserved byte range — this mirrors
         vendor pline.c's printf vsprintf step.

    Parameters
    ----------
    state  : MessageState
    msg_id : int / jnp.int32  — MessageId enum value.  Out-of-range IDs
             render as NONE (all zeros).
    *args  : optional jnp.int32 scalars — printf-style arguments to be
             substituted into the template.  Currently three messages
             have arg slots wired:
                YOU_KILL_MONSTER (3)  -> arg0 = monster entry_idx
                FIND_GOLD        (4)  -> arg0 = decimal gold quantity
                YOU_HIT_MONSTER  (11) -> arg0 = monster entry_idx
             Additional args are ignored; templates without slots ignore
             ``args`` entirely.

    Returns
    -------
    Updated MessageState (new message_buffer + rotated history).

    JIT-safe: shape-static; uses ``.at[].set`` and integer indexing only.
    """
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

    # Step 4: substitute the first arg into the reserved slot (if any).
    # Mirrors vendor pline.c's vsprintf into the BUFSZ buffer.
    if len(args) >= 1:
        kind   = _ARG_KIND[safe_id]
        offset = _ARG_OFFSET[safe_id]
        width  = _ARG_WIDTH[safe_id]
        arg0   = jnp.int32(args[0])

        # Build the substitution bytes for both kinds; pick via lax.cond.
        # NUMERIC: 10-byte right-aligned ASCII via _digits10.
        # MONSTER: lookup MONSTERS[arg0].name from baked table.
        numeric_bytes = _digits10(arg0)
        safe_mon_idx  = jnp.clip(
            arg0, jnp.int32(0), jnp.int32(_N_MONSTERS - 1)
        )
        monster_bytes = _MONSTER_NAME_TABLE[safe_mon_idx]

        # Generic byte-range writer: writes `src[:width]` into
        # new_buffer[offset:offset+width].  Implemented via fori_loop so
        # the trace stays shape-static.
        def _write_slot(buf, src, off, w):
            def _wstep(i, b):
                col_in  = jnp.int32(i)
                col_out = jnp.int32(off) + col_in
                # Bounds-guard: keep col_out within MSG_BUF_LEN.
                col_out = jnp.clip(col_out, jnp.int32(0),
                                   jnp.int32(MSG_BUF_LEN - 1))
                active  = col_in < jnp.int32(w)
                ch = jnp.where(active, src[col_in], b[col_out])
                return b.at[col_out].set(ch)

            max_w = max(_NUMERIC_SLOT_WIDTH, _MONSTER_NAME_SLOT_WIDTH)
            return jax.lax.fori_loop(0, max_w, _wstep, buf)

        # Apply substitution conditional on kind.  NONE -> no-op.
        new_buffer = jax.lax.cond(
            kind == jnp.int32(_ARG_KIND_NUMERIC),
            lambda b: _write_slot(b, numeric_bytes, offset, width),
            lambda b: b,
            new_buffer,
        )
        new_buffer = jax.lax.cond(
            kind == jnp.int32(_ARG_KIND_MONSTER),
            lambda b: _write_slot(b, monster_bytes, offset, width),
            lambda b: b,
            new_buffer,
        )

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
# Game-start role-specific intro line (NLE pline parity).
#
# Cite: vendor/nethack/src/allmain.c::welcome lines 920-922 ::
#     pline("%s %s, welcome to NetHack!  You are a%s.",
#           Hello((struct monst *) 0), svp.plname, buf);
# Where Hello() (role.c lines 2119-2140) returns a role-specific greeting:
#     KNIGHT   -> "Salutations"
#     SAMURAI  -> "Konnichi wa"
#     TOURIST  -> "Aloha"
#     VALKYRIE -> "Velkommen"
#     other    -> "Hello"
# ``buf`` is built from align + race_adj + role_name; we omit the leading
# alignment/race composition here because the static table is keyed only
# by role (race + alignment vary independently and are emitted via the
# header-builder; mirror the role-specific portion of the line which is
# what the validator checks against ``tty_chars row 0``).
# ---------------------------------------------------------------------------

# Per-role greeting prefix from vendor role.c::Hello (lines 2119-2140).
_ROLE_HELLO: tuple[str, ...] = (
    "Hello",        # 0  ARCHEOLOGIST
    "Hello",        # 1  BARBARIAN
    "Hello",        # 2  CAVEMAN
    "Hello",        # 3  HEALER
    "Salutations",  # 4  KNIGHT       role.c:2123
    "Hello",        # 5  MONK
    "Hello",        # 6  PRIEST
    "Hello",        # 7  RANGER
    "Hello",        # 8  ROGUE
    "Konnichi wa",  # 9  SAMURAI      role.c:2125
    "Aloha",        # 10 TOURIST      role.c:2129
    "Velkommen",    # 11 VALKYRIE     role.c:2131
    "Hello",        # 12 WIZARD
)

# Per-role name (vendor role.c::roles[].name.m).
_ROLE_NAME: tuple[str, ...] = (
    "Archeologist", "Barbarian", "Caveman", "Healer", "Knight",
    "Monk", "Priest", "Ranger", "Rogue", "Samurai", "Tourist",
    "Valkyrie", "Wizard",
)

# Per-role female name (vendor role.c::roles[].name.f); None = same as .m.
# Used when the player is female and the role has a distinct female name.
# Cite: vendor/nethack/src/role.c roles[] table.
_ROLE_NAME_F: tuple[str | None, ...] = (
    None,           # 0  ARCHEOLOGIST  (no distinct f)
    None,           # 1  BARBARIAN     (no distinct f)
    "Cavewoman",    # 2  CAVEMAN       role.c: Cavewoman
    None,           # 3  HEALER
    None,           # 4  KNIGHT
    None,           # 5  MONK
    "Priestess",    # 6  PRIEST        role.c: Priestess
    None,           # 7  RANGER
    None,           # 8  ROGUE
    None,           # 9  SAMURAI
    None,           # 10 TOURIST
    None,           # 11 VALKYRIE      (female-only role; name.f is NULL)
    None,           # 12 WIZARD
)

# Roles that allow BOTH genders (ROLE_MALE | ROLE_FEMALE set in .allow).
# Vendor role.c::roles[].allow — gender bit-field.  When BOTH bits are set
# the sex adjective is shown in the welcome line for new games.
# Cite: vendor/nethack/src/allmain.c::welcome lines 682-686.
_ROLE_ALLOWS_BOTH_GENDERS: tuple[bool, ...] = (
    True,   # 0  ARCHEOLOGIST
    True,   # 1  BARBARIAN
    True,   # 2  CAVEMAN
    True,   # 3  HEALER
    True,   # 4  KNIGHT
    True,   # 5  MONK
    True,   # 6  PRIEST
    True,   # 7  RANGER
    True,   # 8  ROGUE
    False,  # 9  SAMURAI  (male-only)
    True,   # 10 TOURIST
    False,  # 11 VALKYRIE (female-only)
    True,   # 12 WIZARD
)

# Alignment adjective strings — vendor role.c::aligns[].adj (lines 762-764).
_ALIGN_ADJ: tuple[str, ...] = (
    "lawful",    # 0  A_LAWFUL
    "neutral",   # 1  A_NEUTRAL
    "chaotic",   # 2  A_CHAOTIC
)

# Race adjective strings — vendor role.c::races[].adj.
_RACE_ADJ: tuple[str, ...] = (
    "human",    # 0  HUMAN
    "elven",    # 1  ELF
    "dwarven",  # 2  DWARF
    "gnomish",  # 3  GNOME
    "orcish",   # 4  ORC
)

# Number of roles (matches Nethax.nethax.constants.roles.N_ROLES = 13).
_N_ROLES_INTRO: int = 13

# NLE default player name — vendor/nle/nle/env/base.py:306 sets
# ``playername="Agent-" + character``; nethack parses ``plname`` as the
# substring before the first '-', yielding "Agent".
# Cite: vendor/nle/nle/env/base.py line 306.
_NLE_PLAYER_NAME: str = "Agent"


def _build_role_intro_line(
    role: int,
    race: int = 0,
    alignment: int = 0,
    female: bool = False,
) -> str:
    """Build the vendor welcome pline text for the given character spec.

    Mirrors vendor/nethack/src/allmain.c::welcome (lines 679-691)::

        *buf = '\\0';
        if (new_game ...) Sprintf(eos(buf), " %s", align_str(...));
        if (!urole.name.f && (urole.allow & ROLE_GENDMASK) == BOTH_GENDERS)
            Sprintf(eos(buf), " %s", genders[currentgend].adj);
        pline("... You are a%s %s %s.", buf, urace.adj, role.name);

    Returns the rendered ASCII line (no trailing NUL).
    """
    r = max(0, min(role, _N_ROLES_INTRO - 1))
    hello     = _ROLE_HELLO[r]
    role_name = _ROLE_NAME[r]
    # Use female role name when player is female and role defines one.
    if female and _ROLE_NAME_F[r] is not None:
        role_name = _ROLE_NAME_F[r]  # type: ignore[assignment]

    align_adj = _ALIGN_ADJ[max(0, min(alignment, 2))]

    race_r    = max(0, min(race, len(_RACE_ADJ) - 1))
    race_adj  = _RACE_ADJ[race_r]

    # buf = " <align>" [ + " <sex>" if role allows both genders and name.f is None ]
    buf = f" {align_adj}"
    if _ROLE_ALLOWS_BOTH_GENDERS[r] and _ROLE_NAME_F[r] is None:
        sex_adj = "female" if female else "male"
        buf += f" {sex_adj}"

    return (
        f"{hello} {_NLE_PLAYER_NAME}, welcome to NetHack!"
        f"  You are a{buf} {race_adj} {role_name}."
    )


def emit_role_intro(
    state: MessageState,
    role: int,
    race: int = 0,
    alignment: int = 0,
    female: bool = False,
) -> MessageState:
    """Emit the role-specific game-start welcome line.

    Mirrors vendor/nethack/src/allmain.c::welcome (lines 679-691) — the
    ``pline("%s %s, welcome to NetHack!  You are a%s %s %s.", ...)`` call
    fired once at new-game start.

    The message text is built on the Python host (reset is not JIT-compiled)
    so it can incorporate the full alignment + sex + race + role descriptor,
    matching NLE's ``message`` obs byte-for-byte.

    Cite: vendor/nethack/src/allmain.c::welcome lines 679-691;
          vendor/nle/nle/env/base.py line 306 (plname = "Agent").

    Parameters
    ----------
    state     : MessageState
    role      : Role integer index (0..12).
    race      : Race integer index (0=Human, 1=Elf, 2=Dwarf, 3=Gnome, 4=Orc).
    alignment : 0=lawful, 1=neutral, 2=chaotic.
    female    : True if the player is female.

    Returns
    -------
    Updated MessageState with the welcome line in ``message_buffer`` and the
    previous buffer rotated into ``message_history``.
    """
    # Build the ASCII welcome line on the host.
    line = _build_role_intro_line(role, race=race, alignment=alignment, female=female)

    # Pack into the MSG_BUF_LEN buffer:
    #   byte 0       = MessageId.ROLE_INTRO  (internal msg_id tag)
    #   bytes 1..len = ASCII text
    #   remainder    = zero-padded
    arr = _np.zeros((MSG_BUF_LEN,), dtype=_np.uint8)
    arr[0] = int(MessageId.ROLE_INTRO) & 0xFF
    raw = line.encode("ascii")[: MSG_BUF_LEN - 1]
    arr[1 : 1 + len(raw)] = list(raw)
    new_buffer = jnp.asarray(arr, dtype=jnp.uint8)

    # Rotate current buffer into history (same contract as ``emit``).
    safe_idx    = jnp.mod(state.history_index, jnp.int32(HISTORY_LEN))
    new_history = state.message_history.at[safe_idx].set(state.message_buffer)
    new_index   = jnp.mod(state.history_index + jnp.int32(1), jnp.int32(HISTORY_LEN))

    return state.replace(
        message_buffer=new_buffer,
        message_history=new_history,
        history_index=new_index,
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
