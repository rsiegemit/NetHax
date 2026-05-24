"""Engrave subsystem — write text on the dungeon floor (Elbereth & friends).

Canonical sources:
  vendor/nethack/src/engrave.c::doengrave        — main ENGRAVE command handler
  vendor/nethack/src/engrave.c::write_engr_text  — per-tile write back / merge
  vendor/nethack/include/engrave.h               — engr struct fields

Status: Wave 5 Phase 4 — minimal Elbereth-in-dust simplification.

The vendor flow lets the player pick a writing implement (finger, wand,
athame, ...) and arbitrary text; the engraving kind controls whether the
inscription scares monsters (Elbereth on a non-dusted engraving) and how
quickly it fades.  For Wave 5 we collapse the action to "engrave 'Elbereth'
in dust at the player's current tile", since that's the only inscription
that has gameplay effects and the only one the ELBERETHLESS conduct cares
about.

Engraving kinds (engrave.h:26-31 enum):
    0 = none      — empty tile
    1 = dust      — finger in dust (fades quickly when stepped on)
    2 = engrave   — athame / digging / cold wand
    3 = burn      — fire-wand / fire-trap / Fire Brand
    4 = mark      — magic marker (semi-permanent)
    5 = blood     — vampire/demon finger; wears like dust
    6 = headstone — graveyard inscription (always considered "fresh")
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map dimensions (must match dungeon.branches.MAP_H/MAP_W).  Imported lazily
# inside default() so this module can be imported before the dungeon package.

# Max characters of inscription text stored per tile.
# Vendor: BUFSZ-1 = 255 in engrave.h; we use 80 to match the screen width
# bound used in obs/look.py::_engrave_descriptor for "You read: '<text>'".
# This is wide enough for any realistic engraving while keeping the per-level
# state under 1.3 MB (21*80*80 bytes).
# Cite: vendor/nethack/include/engrave.h struct engr.engr_txt.
ENGRAVE_TEXT_LEN: int = 80

# Engraving kinds (mirrors engrave.h ENGR_DUST / ENGR_BURN / ENGR_ENGRAVE / ENGR_MARK / ENGR_BLOOD).
ENGR_NONE: int    = 0
ENGR_DUST: int    = 1
ENGR_ENGRAVE: int = 2
ENGR_BURN: int    = 3
ENGR_MARK: int    = 4  # magic marker (engrave.h:MARK=4); semi-permanent, not eroded by wipe_engr_on_step
ENGR_BLOOD: int   = 5  # blood writing (engrave.h:ENGR_BLOOD=5); vampire/demon finger (engrave.c:doengrave line 573)
ENGR_HEADSTONE: int = 6  # graveyard headstone (engrave.h:HEADSTONE=6); excluded from is_elbereth_at

# ASCII byte sequence for "Elbereth" — padded with zeros to ENGRAVE_TEXT_LEN.
_ELBERETH_BYTES = tuple(b"Elbereth") + (0,) * (ENGRAVE_TEXT_LEN - 8)


# ---------------------------------------------------------------------------
# State struct
# ---------------------------------------------------------------------------

@struct.dataclass
class EngraveState:
    """Engravings on the current level.

    Wave 5 simplification: we only track engravings on the *current* level
    (one MAP_H x MAP_W grid).  Multi-level engravings would shape this as
    [N_BRANCHES * MAX_LEVELS, MAP_H, MAP_W, ENGRAVE_TEXT_LEN]; deferred to
    Wave 6 since no current test exercises level transitions.

    Fields
    ------
    text           : int8[MAP_H, MAP_W, ENGRAVE_TEXT_LEN] — ASCII bytes.
    has_engraving  : bool[MAP_H, MAP_W] — True where text is meaningful.
    engraving_kind : int8[MAP_H, MAP_W] — ENGR_NONE/DUST/ENGRAVE/BURN/MARK/BLOOD/HEADSTONE.
    engr_time      : int32[MAP_H, MAP_W] — turn at which the engraving becomes
                     readable (mirror of vendor engr.engr_time).  Monsters
                     consult ``engr_time <= moves`` before reacting (sengr_at
                     at engrave.c:250-261), so a half-dried-up inscription
                     doesn't yet repel them.  Defaults to 0 (immediately
                     readable) so existing tests / state defaults work
                     without a multi-turn write occupation.
    """

    text: jnp.ndarray            # int8[MAP_H, MAP_W, ENGRAVE_TEXT_LEN]
    has_engraving: jnp.ndarray   # bool[MAP_H, MAP_W]
    engraving_kind: jnp.ndarray  # int8[MAP_H, MAP_W]
    engr_time: jnp.ndarray       # int32[MAP_H, MAP_W]  — vendor engr.engr_time

    @classmethod
    def default(cls, map_h: int | None = None, map_w: int | None = None) -> "EngraveState":
        """Return a fresh empty EngraveState.

        Shape defaults to dungeon.branches.MAP_H x MAP_W when not provided.
        """
        if map_h is None or map_w is None:
            from Nethax.nethax.dungeon.branches import MAP_H, MAP_W
            map_h = map_h or MAP_H
            map_w = map_w or MAP_W
        return cls(
            text=jnp.zeros((map_h, map_w, ENGRAVE_TEXT_LEN), dtype=jnp.int8),
            has_engraving=jnp.zeros((map_h, map_w), dtype=jnp.bool_),
            engraving_kind=jnp.zeros((map_h, map_w), dtype=jnp.int8),
            engr_time=jnp.zeros((map_h, map_w), dtype=jnp.int32),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _elbereth_bytes_array() -> jnp.ndarray:
    """Return an int8[ENGRAVE_TEXT_LEN] vector with the ASCII for 'Elbereth'."""
    return jnp.array(_ELBERETH_BYTES, dtype=jnp.int8)


def _is_vampire_form(state) -> jnp.ndarray:
    """Return True if the player is polymorphed into a vampire form.

    Mirrors vendor/nethack/src/engrave.c::doengrave line 573 bloodtype check:
      Upolyd && is_bloodtype(youmonst.data)
    A vampire form is identified by M2_UNDEAD flag + S_VAMPIRE symbol.
    """
    from Nethax.nethax.constants.monsters import MONSTERS, M2_UNDEAD, MonsterSymbol

    poly = state.polymorph
    form_idx = poly.current_form_idx.astype(jnp.int32)

    # Build lookup tables for M2_UNDEAD flag and S_VAMPIRE symbol per monster.
    flags2_table = jnp.array([int(m.flags2) for m in MONSTERS], dtype=jnp.uint32)
    symbol_table = jnp.array([int(m.symbol) for m in MONSTERS], dtype=jnp.int32)

    form_flags2 = flags2_table[form_idx]
    form_symbol = symbol_table[form_idx]

    is_undead  = (form_flags2 & jnp.uint32(M2_UNDEAD)) != jnp.uint32(0)
    is_vampire_sym = form_symbol == jnp.int32(int(MonsterSymbol.S_VAMPIRE))

    return poly.is_polymorphed & is_undead & is_vampire_sym


def handle_engrave(state, rng):
    """Player engraves at the current position.

    Wave 5 simplification: always engrave 'Elbereth' in dust regardless of
    inventory (writing finger).  Mirrors the most-common ELBERETHLESS-
    violating action in vendor (engrave.c::doengrave's finger path).

    When the player is polymorphed into a vampire form, the engraving kind
    is ENGR_BLOOD instead of ENGR_DUST (engrave.c::doengrave line 573).

    Effects:
      - state.engrave.has_engraving[row, col]  = True
      - state.engrave.engraving_kind[row, col] = ENGR_BLOOD (vampire) or ENGR_DUST
      - state.engrave.text[row, col, :]        = 'Elbereth'
      - state.conduct.violations[ELBERETHLESS] = True

    Parameters
    ----------
    state : EnvState
    rng   : jax.random.PRNGKey (unused for the dust-Elbereth case).

    Returns
    -------
    Updated EnvState.
    """
    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated

    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)

    bytes_vec = _elbereth_bytes_array()

    # Vampire polymorph → blood writing (engrave.c:573).
    kind = jnp.where(_is_vampire_form(state), jnp.int8(ENGR_BLOOD), jnp.int8(ENGR_DUST))

    eng = state.engrave
    new_text  = eng.text.at[row, col, :].set(bytes_vec)
    new_has   = eng.has_engraving.at[row, col].set(jnp.bool_(True))
    new_kind  = eng.engraving_kind.at[row, col].set(kind)
    # vendor engrave.c::doengrave sets ``engr_time = moves`` on a finger-write
    # so monsters can react immediately.  Multi-turn occupation-based engraves
    # (athame inscription, etc.) would set engr_time = moves + delay; since
    # Nethax doesn't model the occupation loop, we treat all writes as
    # instantly readable.  Cite: engrave.c::doengrave write-back block.
    new_time  = eng.engr_time.at[row, col].set(state.timestep.astype(jnp.int32))

    new_engrave = eng.replace(
        text=new_text,
        has_engraving=new_has,
        engraving_kind=new_kind,
        engr_time=new_time,
    )
    new_state = state.replace(engrave=new_engrave)
    # Conduct: ELBERETHLESS broken on any engrave action (insight.c counter).
    return mark_violated(new_state, int(Conduct.ELBERETHLESS))


def step(
    state: EngraveState,
    rng: jax.Array,
    player_row: jnp.ndarray = None,
    player_col: jnp.ndarray = None,
) -> EngraveState:
    """Per-turn tick — erode DUST engraving at the player's tile.

    vendor/nethack/src/engrave.c::wipe_engr_at lines 270-290:
        wipe_engr_at(x, y, cnt, magical) — DUST engravings erode on every
        step that lands on them. ENGRAVE/BURN/MARK/BLOOD only erode when
        magical && !rn2(2) (line 278).

    Currently called as a no-op; now if `player_row`/`player_col` are
    given, invoke `_wipe_engr_tile` at the player's current tile to fade
    a DUST engraving by 2 chars 50% of the time. Backward-compatible: if
    the caller doesn't pass coords (e.g., env.py during a non-movement
    turn), do nothing.
    """
    if player_row is None or player_col is None:
        return state
    return _wipe_engr_tile(state, player_row, player_col, rng)


def engrave_text_at(eng: EngraveState, row, col) -> jnp.ndarray:
    """Return the raw int8[ENGRAVE_TEXT_LEN] text array at (row, col).

    Mirrors vendor/nethack/src/engrave.c::engr_at which returns a pointer
    to the engr struct; callers read ep->engr_txt directly.
    """
    r = jnp.asarray(row, dtype=jnp.int32)
    c = jnp.asarray(col, dtype=jnp.int32)
    return eng.text[r, c, :]


def _wipe_engr_tile(eng: EngraveState, row, col, rng: jax.Array) -> EngraveState:
    """Erode the engraving at (row, col) per vendor wear rules.

    Mirrors vendor/nethack/src/engrave.c::wipe_engr_at (lines 270-289):

      - DUST and ENGR_BLOOD wear quickly (vendor uses the full rubouts
        substitution table; we keep the existing 50% bernoulli + 2-byte
        trim as an approximation — full character substitution via the
        ``rubouts`` table is deferred).
      - ENGRAVE, MARK, and HEADSTONE wear via ``rn2(1 + 50/(cnt+1)) == 0``;
        with ``cnt=1`` (single wipe-attempt per step) that's
        ``rn2(26) == 0`` ≈ 1/26 per step.  cite engrave.c:280-285.
      - BURN only erodes when ``is_ice(x,y) || (magical && !rn2(2))``.
        Plain movement doesn't visit those branches, so BURN never wears
        via this path (matches the practical effect).
      - When all bytes become zero, clear ``has_engraving``.

    Previously: only DUST eroded; ENGRAVE/MARK/BURN/BLOOD were treated
    as fully permanent, which let bloodwriting and athame-engraved
    Elbereth last forever instead of degrading slowly.
    """
    r = jnp.asarray(row, dtype=jnp.int32)
    c = jnp.asarray(col, dtype=jnp.int32)

    kind = eng.engraving_kind[r, c].astype(jnp.int32)

    # vendor wipe_engr_at line 279: DUST and BLOOD pass into the fast
    # wear path; ENGRAVE/MARK/HEADSTONE wear via the slow per-call gate;
    # BURN is unaffected by movement.
    is_dust_class = (kind == ENGR_DUST) | (kind == ENGR_BLOOD)
    is_slow_class = (
        (kind == ENGR_ENGRAVE)
        | (kind == ENGR_MARK)
        | (kind == ENGR_HEADSTONE)
    )

    rng_fast, rng_slow = jax.random.split(rng)
    fast_roll = jax.random.bernoulli(rng_fast, p=0.5)
    # cnt=1 single wipe attempt: rn2(1 + 50/(1+1)) == rn2(26).
    slow_roll = jax.random.randint(
        rng_slow, (), 0, 26, dtype=jnp.int32
    ) == jnp.int32(0)

    should_erode = (is_dust_class & fast_roll) | (is_slow_class & slow_roll)

    text = eng.text[r, c, :]  # int8[ENGRAVE_TEXT_LEN]

    # Count non-zero bytes to find current text length.
    nonzero_mask = text != 0                     # bool[ENGRAVE_TEXT_LEN]
    text_len = jnp.sum(nonzero_mask.astype(jnp.int32))

    # New length after removing 2 chars (clamped to 0).
    new_len = jnp.maximum(text_len - 2, 0)

    # Build eroded text: keep first new_len bytes, zero the rest.
    indices = jnp.arange(ENGRAVE_TEXT_LEN, dtype=jnp.int32)
    eroded_text = jnp.where(indices < new_len, text, jnp.int8(0))

    # Apply only when should_erode.
    new_text = jnp.where(should_erode, eroded_text, text)
    still_has = (new_len > 0) | ~should_erode

    new_texts = eng.text.at[r, c, :].set(new_text)
    new_has   = eng.has_engraving.at[r, c].set(eng.has_engraving[r, c] & still_has)

    return eng.replace(text=new_texts, has_engraving=new_has)


def wipe_engr_on_step(state_or_eng, row_or_rng, col=None, rng=None):
    """Erode the DUST engraving at the player's tile (or an explicit tile).

    Two call signatures are supported:

    1. ``wipe_engr_on_step(state, rng)``
       Takes a full EnvState; erodes the engraving at state.player_pos.
       Returns an updated EnvState.

    2. ``wipe_engr_on_step(eng, row, col, rng)``
       Takes an EngraveState plus explicit tile coords.
       Returns an updated EngraveState.

    Mirrors vendor/nethack/src/engrave.c::wipe_engr_at (lines 270-290).
    """
    if col is None:
        # Signature 1: (state, rng)
        state = state_or_eng
        rng_key = row_or_rng
        row = state.player_pos[0].astype(jnp.int32)
        col_val = state.player_pos[1].astype(jnp.int32)
        new_eng = _wipe_engr_tile(state.engrave, row, col_val, rng_key)
        return state.replace(engrave=new_eng)
    else:
        # Signature 2: (eng, row, col, rng)
        eng = state_or_eng
        row = row_or_rng
        return _wipe_engr_tile(eng, row, col, rng)


def is_elbereth_at(eng: EngraveState, row, col, moves=None) -> jnp.ndarray:
    """Return True if the engraving at ``(row, col)`` is exactly 'Elbereth'.

    Mirrors vendor/nethack/src/engrave.c::sengr_at strict-mode (engrave.c:
    250-261), the test consulted by monster AI to decide whether to flee /
    avoid the tile.  The vendor function rejects an engraving in three
    additional cases beyond the strict-text match:
      1. ``ep->engr_time > svm.moves`` — the inscription is still being
         dried / not yet readable.
      2. ``ep->engr_type == HEADSTONE`` — graveyard inscriptions never
         repel monsters (they are not "Elbereth on the floor").
      3. ``ep->engr_txt[0] == '\0'`` — empty inscription; covered by
         ``has_engraving=False`` here.

    Parameters
    ----------
    eng   : EngraveState
    row   : int / scalar int32
    col   : int / scalar int32
    moves : optional scalar int32 — current turn counter (``state.timestep``).
            When provided, gate on ``engr_time <= moves``.  When omitted,
            we conservatively treat the engraving as readable (preserves
            legacy callers that only had the strict-text check).
    """
    r = jnp.asarray(row, dtype=jnp.int32)
    c = jnp.asarray(col, dtype=jnp.int32)
    has = eng.has_engraving[r, c]
    text_at = eng.text[r, c, :]
    target = _elbereth_bytes_array()
    matches = jnp.all(text_at == target)

    # HEADSTONE inscriptions are never read as Elbereth even if text matches.
    kind = eng.engraving_kind[r, c].astype(jnp.int32)
    not_headstone = kind != jnp.int32(ENGR_HEADSTONE)

    # Drying-delay gate.  When moves is not supplied, treat as readable.
    if moves is None:
        readable = jnp.bool_(True)
    else:
        m = jnp.asarray(moves, dtype=jnp.int32)
        readable = eng.engr_time[r, c].astype(jnp.int32) <= m

    return has & matches & not_headstone & readable
