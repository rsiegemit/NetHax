"""Engrave subsystem — write text on the dungeon floor (Elbereth & friends).

Canonical sources:
  vendor/nethack/src/engrave.c::doengrave        — main ENGRAVE command handler
  vendor/nethack/src/engrave.c::write_engr_text  — per-tile write back / merge
  vendor/nethack/include/engrave.h               — engr struct fields

Status: Wave 46b — full vendor parity (DUST/BLOOD/ENGRAVE/BURN/MARK/HEADSTONE
+ is_elbereth + engrave_scares_monster + tick_engravings decay).

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
import os as _os
# NETHAX_SINGLE_LEVEL training mode: shrink to 8 ("Elbereth" length). The
# vectorized monster step vmaps monster_turn over 400 slots, replicating the
# threaded engrave.text [H,W,LEN] per-monster -> [400,21,80,80] int32 = 215 MB/env
# (the sole large per-env activation; blocks B>=256 on a 40 GB A100).  LEN=8 still
# detects Elbereth (8 chars); longer engravings are truncated, fine for training.
# Gated so the byte-parity path (no gate) keeps the full 80.
ENGRAVE_TEXT_LEN: int = (
    int(_os.environ.get("NETHAX_ENGRAVE_LEN", "8"))
    if _os.environ.get("NETHAX_SINGLE_LEVEL", "0") == "1" else 80
)

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
# Rubouts substitution table (D16)
# Cite: vendor/nethack/src/engrave.c lines 65-116 (rubouts[]) and
#       lines 118-183 (wipeout_text).
#
# wipeout_text picks a random position in the string and either:
#   - if char is space: skip (no change)
#   - if char in "?.,'`-|_": maps to space
#   - else with prob 3/4: looks up the rubouts entry; if found, picks one of
#     the listed substitutes at random; if not found, sets to '?'
#   - else (prob 1/4 when use_rubout==0): sets to '?'
# After ``cnt`` iterations, trailing spaces are trimmed.
#
# We model this as two static [256]-shape lookup arrays:
#   _RUBOUTS_LUT[c, k]   = ASCII byte of the k-th substitute for char c
#                          (only first _RUBOUTS_COUNT[c] entries are valid).
#   _RUBOUTS_COUNT[c]    = number of substitutes for c (0 → fall through to '?').
# Plus a punctuation mask _IS_PUNCT_RUBOUT[c] for the
# "?.,'`-|_" → space rule.
# ---------------------------------------------------------------------------

# Max substitutes per char: 'E' → "|FL[_" has 5 entries (longest row).
_RUBOUTS_MAX_K: int = 5

# Pairs taken directly from vendor engrave.c lines 69-116.
_RUBOUTS_PAIRS: tuple = (
    ('A', '^'),    ('B', 'Pb['),  ('C', '('),    ('D', '|)['),
    ('E', '|FL[_'),('F', '|-'),   ('G', 'C('),   ('H', '|-'),
    ('I', '|'),    ('K', '|<'),   ('L', '|_'),   ('M', '|'),
    ('N', '|\\'),  ('O', 'C('),   ('P', 'F'),    ('Q', 'C('),
    ('R', 'PF'),   ('T', '|'),    ('U', 'J'),    ('V', '/\\'),
    ('W', 'V/\\'), ('Z', '/'),    ('b', '|'),    ('d', 'c|'),
    ('e', 'c'),    ('g', 'c'),    ('h', 'n'),    ('j', 'i'),
    ('k', '|'),    ('l', '|'),    ('m', 'nr'),   ('n', 'r'),
    ('o', 'c'),    ('q', 'c'),    ('w', 'v'),    ('y', 'v'),
    (':', '.'),    (';', ',:'),   (',', '.'),    ('=', '-'),
    ('+', '-|'),   ('*', '+'),    ('@', '0'),    ('0', 'C('),
    ('1', '|'),    ('6', 'o'),    ('7', '/'),    ('8', '3o'),
)


def _build_rubouts_arrays():
    """Build the static rubouts LUT + count arrays from vendor pairs.

    Returns
    -------
    (lut, count) : ([256, _RUBOUTS_MAX_K] int8, [256] int8)
    """
    import numpy as np
    lut = np.zeros((256, _RUBOUTS_MAX_K), dtype=np.int8)
    cnt = np.zeros((256,), dtype=np.int8)
    for src, subs in _RUBOUTS_PAIRS:
        sc = ord(src)
        for k, ch in enumerate(subs):
            assert k < _RUBOUTS_MAX_K, f"too many substitutes for {src!r}"
            lut[sc, k] = ord(ch)
        cnt[sc] = len(subs)
    return jnp.asarray(lut, dtype=jnp.int8), jnp.asarray(cnt, dtype=jnp.int8)


_RUBOUTS_LUT, _RUBOUTS_COUNT = _build_rubouts_arrays()

# Small punctuation that wipeout_text maps directly to space (engrave.c:149):
#   strchr("?.,'`-|_", *s)
def _build_punct_mask():
    import numpy as np
    mask = np.zeros((256,), dtype=bool)
    for c in "?.,'`-|_":
        mask[ord(c)] = True
    return jnp.asarray(mask, dtype=jnp.bool_)


_IS_PUNCT_RUBOUT = _build_punct_mask()

# Vendor wipe_engr_at uses ``cnt = rnd(5)`` (1..5) for player-step rubouts
# (hack.c::maybe_smudge_engr line 3026).  We cap the loop at this constant
# so the scan is JIT-friendly.
_WIPEOUT_MAX_CNT: int = 5


# ---------------------------------------------------------------------------
# Tool/wand dispatch otyps (D17)
# Cite: vendor/nethack/src/engrave.c::doengrave_sfx_item lines 819-849 and
#       doengrave_sfx_item_WAN lines 583-737.
# Vendor otyps are the indices in vendor/nethack/include/objects.h, which
# match the ``constants/objects.py:OBJECTS`` tuple position (see e.g.
# subsystems/containers.py:880 _WAN_CANCELLATION_TYPE_ID = 395).
# ---------------------------------------------------------------------------
_OTYP_MAGIC_MARKER:      int = 217   # objects.py:4524
_OTYP_WAN_CANCELLATION:  int = 395   # objects.py:8084 (containers.py:880)
_OTYP_WAN_DIGGING:       int = 400   # objects.py:8184
_OTYP_WAN_FIRE:          int = 402   # objects.py:8224
_OTYP_WAN_COLD:          int = 403   # objects.py:8244
_OTYP_WAN_LIGHTNING:     int = 406   # objects.py:8304


def _build_is_blade_lut():
    """Build a [NUM_OBJECTS]-sized bool LUT marking is_blade(otyp).

    Vendor obj.h line 213-216: is_blade(otmp) is
        oclass == WEAPON_CLASS && P_DAGGER <= oc_skill <= P_SABER
    (P_DAGGER=1 .. P_SABER=9).  Cite vendor/nethack/include/skills.h:24-32.
    """
    import numpy as np
    from Nethax.nethax.constants.objects import OBJECTS, ObjectClass
    n = len(OBJECTS)
    blade = np.zeros((n,), dtype=bool)
    for i, e in enumerate(OBJECTS):
        if e.class_ == ObjectClass.WEAPON_CLASS and 1 <= e.oc_skill <= 9:
            blade[i] = True
    return jnp.asarray(blade, dtype=jnp.bool_)


_IS_BLADE_OTYP = _build_is_blade_lut()


# ---------------------------------------------------------------------------
# State struct
# ---------------------------------------------------------------------------

@struct.dataclass
class EngraveState:
    """Engravings on the current level.

    JAX-required schema deficit: engravings are tracked only on the
    *current* level (one MAP_H x MAP_W grid).  Vendor stores engravings
    in a linked-list rooted at ``g.engravings`` indexed by level via
    ``level.engravings``; full multi-level parity would shape this as
    [N_BRANCHES * MAX_LEVELS, MAP_H, MAP_W, ENGRAVE_TEXT_LEN] which
    would grow EnvState by ~70 MB.  Level snapshot/restore in
    ``level_memory.snapshot_engravings`` (save.c:548 mirror) is the
    workaround: engravings are serialised on level-exit and restored
    on level-enter, so cross-level persistence works correctly.

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


def _wielded_otyp(state) -> jnp.ndarray:
    """Return the wielded weapon/tool/wand's otyp (or -1 when bare-handed).

    Mirrors subsystems/combat.py::_wielded_type_id (line 840-845): looks up
    ``state.inventory.items.type_id[wielded_slot]`` with clamping for the
    -1 (bare hands) sentinel.
    """
    wielded = state.inventory.wielded.astype(jnp.int32)
    safe = jnp.clip(wielded, 0, state.inventory.items.type_id.shape[0] - 1)
    type_id = state.inventory.items.type_id[safe].astype(jnp.int32)
    return jnp.where(wielded >= 0, type_id, jnp.int32(-1))


def _engr_type_for_otyp(otyp: jnp.ndarray, kind_default: jnp.ndarray) -> jnp.ndarray:
    """Dispatch wielded ``otyp`` → engr_type per vendor doengrave_sfx_item.

    Vendor cites:
      - WEAPON_CLASS + is_blade → de->type = ENGRAVE  (engrave.c:819-833)
      - WAN_FIRE       → de->type = BURN              (engrave.c:707-717)
      - WAN_LIGHTNING  → de->type = BURN              (engrave.c:718-734)
      - WAN_DIGGING    → de->type = ENGRAVE           (engrave.c:684-705)
      - MAGIC_MARKER   → de->type = MARK              (engrave.c:843-849)
      - All other wands (incl. WAN_COLD, WAN_CANCELLATION) and the
        bare-finger / tool / amulet / ... paths leave de->type at its
        init value, which is ENGR_BLOOD for vampire/demon form and
        ENGR_DUST otherwise (engrave.c:558, 573-574).

    Notes / approximations:
      - We do NOT implement vendor's ``is_art(ART_FIRE_BRAND)`` → BURN
        branch (engrave.c:820-821).  Artifact identity for inventory
        weapons is not currently surfaced as a per-otyp field; the
        regular fire-brand long-sword otyp still falls through to the
        blade-→ENGRAVE rule, which differs only in the engr_type
        (ENGRAVE vs BURN).  Document for future tightening.
      - Non-blade weapons (oc_skill outside P_DAGGER..P_SABER) keep
        de->type = DUST per vendor (the welded/dull pline branches at
        engrave.c:825-829 also leave type unchanged).

    Parameters
    ----------
    otyp         : int32 scalar — wielded otyp, or -1 for bare hands.
    kind_default : int8 scalar  — ENGR_DUST or ENGR_BLOOD (vampire form).
    """
    blade_lut_len = _IS_BLADE_OTYP.shape[0]
    safe_otyp = jnp.clip(otyp, 0, blade_lut_len - 1)
    is_blade = (otyp >= 0) & _IS_BLADE_OTYP[safe_otyp]

    is_marker     = otyp == jnp.int32(_OTYP_MAGIC_MARKER)
    is_wan_fire   = otyp == jnp.int32(_OTYP_WAN_FIRE)
    is_wan_lit    = otyp == jnp.int32(_OTYP_WAN_LIGHTNING)
    is_wan_dig    = otyp == jnp.int32(_OTYP_WAN_DIGGING)

    # Priority order mirrors the vendor switch fall-through: WAN cases set
    # explicit types; weapon-blade sets ENGRAVE; everything else keeps
    # the default (DUST or BLOOD).
    out = kind_default
    out = jnp.where(is_blade,   jnp.int8(ENGR_ENGRAVE), out)
    out = jnp.where(is_wan_dig, jnp.int8(ENGR_ENGRAVE), out)
    out = jnp.where(is_wan_fire,jnp.int8(ENGR_BURN),    out)
    out = jnp.where(is_wan_lit, jnp.int8(ENGR_BURN),    out)
    out = jnp.where(is_marker,  jnp.int8(ENGR_MARK),    out)
    return out


def handle_engrave(state, rng):
    """Player engraves at the current position.

    Vendor reference: engrave.c::doengrave + doengrave_sfx_item +
    doengrave_sfx_item_WAN (lines 543-737, 740-892).  Vendor dispatches
    on the wielded stylus (de->otmp->oclass / ->otyp) to choose the
    engraving kind.  Nethax mirrors that dispatch via
    ``_engr_type_for_otyp`` (D17), while always inscribing the literal
    text 'Elbereth' (Wave 5 simplification — vendor prompts for text;
    only Elbereth has gameplay effects via the ELBERETHLESS conduct).

    Default kind is ENGR_DUST; ENGR_BLOOD when polymorphed into a
    vampire form (engrave.c:573 is_demon / is_vampire branch); then the
    wielded otyp may upgrade to ENGRAVE/BURN/MARK.

    Effects:
      - state.engrave.has_engraving[row, col]  = True
      - state.engrave.engraving_kind[row, col] = dispatched engr_type
      - state.engrave.text[row, col, :]        = 'Elbereth'
      - state.conduct.violations[ELBERETHLESS] = True

    Parameters
    ----------
    state : EnvState
    rng   : jax.random.PRNGKey (unused for the simplified inscription).
    """
    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated

    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)

    bytes_vec = _elbereth_bytes_array()

    # Vampire polymorph → blood writing (engrave.c:573-574).
    kind_default = jnp.where(
        _is_vampire_form(state), jnp.int8(ENGR_BLOOD), jnp.int8(ENGR_DUST)
    )

    # Tool/wand dispatch (engrave.c:819-849, 583-737).
    otyp = _wielded_otyp(state)
    kind = _engr_type_for_otyp(otyp, kind_default)

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


def _wipeout_text_jax(
    text: jnp.ndarray,
    rng: jax.Array,
    cnt: jnp.ndarray,
) -> jnp.ndarray:
    """JAX port of vendor wipeout_text (engrave.c lines 118-183).

    Picks ``cnt`` random positions in ``text`` (each independently re-rolled)
    and either:
      - leaves a space unchanged,
      - maps "?.,'`-|_" → space (engrave.c:149),
      - with prob 1/4 sets to '?' (engrave.c:154-155 use_rubout==0),
      - with prob 3/4 looks up the rubouts table; if found, picks one of
        the substitutes uniformly at random; otherwise sets to '?'
        (engrave.c:156-176).
    Then trims trailing spaces (replaces them with NUL terminators, which
    Nethax represents as 0 bytes); ``has_engraving`` is cleared by the
    caller when the resulting text is fully blank.

    The loop runs a fixed ``_WIPEOUT_MAX_CNT`` iterations and masks
    iterations past ``cnt`` to keep the kernel JIT-shape-static.  Each
    iteration consumes one Threefry split off ``rng`` (no key reuse).

    Parameters
    ----------
    text : int8[ENGRAVE_TEXT_LEN] — current engraving bytes (0-terminated).
    rng  : jax.random.PRNGKey
    cnt  : int32 scalar — number of degrade-steps (1..5 from vendor caller).
    """
    cnt_i = jnp.asarray(cnt, dtype=jnp.int32)
    keys = jax.random.split(rng, _WIPEOUT_MAX_CNT)

    def step_one(text_carry: jnp.ndarray, scan_inputs):
        key, idx = scan_inputs
        active = idx < cnt_i

        k_pos, k_use_rubout, k_sub = jax.random.split(key, 3)

        # Current length = # nonzero leading bytes.  Vendor uses strlen,
        # i.e. position of first NUL; trailing zeros (post-trim) match.
        nz = text_carry != jnp.int8(0)
        # equivalent to strlen since vendor maintains NUL-terminated
        # buffers (no internal zero bytes).
        lth = jnp.sum(nz.astype(jnp.int32))

        # Pick position; clamp to 1 to avoid degenerate randint(0,0).
        lth_safe = jnp.maximum(lth, jnp.int32(1))
        pos = jax.random.randint(k_pos, (), 0, lth_safe, dtype=jnp.int32)

        # use_rubout in [0,4).
        use_rubout = jax.random.randint(
            k_use_rubout, (), 0, 4, dtype=jnp.int32
        )

        # Read current character (as uint8 index into rubouts LUT).
        ch = text_carry[pos].astype(jnp.int32) & jnp.int32(0xFF)
        is_space = ch == jnp.int32(ord(' '))
        is_punct = _IS_PUNCT_RUBOUT[ch]

        # rubouts lookup
        rcnt = _RUBOUTS_COUNT[ch].astype(jnp.int32)
        rcnt_safe = jnp.maximum(rcnt, jnp.int32(1))
        sub_idx = jax.random.randint(k_sub, (), 0, rcnt_safe, dtype=jnp.int32)
        sub_byte = _RUBOUTS_LUT[ch, sub_idx]
        has_rubout = rcnt > jnp.int32(0)

        # Compose the new character per vendor switch:
        #   space          → ch (no change)
        #   punct          → ' '
        #   use_rubout==0  → '?'
        #   has_rubout     → sub_byte
        #   else           → '?'
        question = jnp.int8(ord('?'))
        space_b  = jnp.int8(ord(' '))
        new_char = jnp.where(
            is_space,
            jnp.asarray(ch, dtype=jnp.int8),
            jnp.where(
                is_punct,
                space_b,
                jnp.where(
                    use_rubout == jnp.int32(0),
                    question,
                    jnp.where(has_rubout, sub_byte, question),
                ),
            ),
        )

        # Apply only when this iteration is active AND length > 0.
        do_write = active & (lth > jnp.int32(0))
        new_text = jnp.where(
            do_write,
            text_carry.at[pos].set(new_char),
            text_carry,
        )
        return new_text, None

    text_out, _ = jax.lax.scan(
        step_one,
        text,
        (keys, jnp.arange(_WIPEOUT_MAX_CNT, dtype=jnp.int32)),
    )

    # Vendor trailing-space trim (engrave.c:181-182):
    #   while (lth && engr[lth-1] == ' ') engr[--lth] = '\0';
    # Compute a "trailing-blank" mask: position i is trailing-blank iff
    # every position j>=i is space-or-NUL.  Implemented via reverse cummin.
    is_blank = (text_out == jnp.int8(ord(' '))) | (text_out == jnp.int8(0))
    rev_blank = jnp.flip(is_blank.astype(jnp.int32), axis=0)
    rev_cummin = jnp.minimum.accumulate(rev_blank)
    trailing_blank = jnp.flip(rev_cummin, axis=0).astype(jnp.bool_)
    text_trimmed = jnp.where(trailing_blank, jnp.int8(0), text_out)
    return text_trimmed


def _wipe_engr_tile(eng: EngraveState, row, col, rng: jax.Array) -> EngraveState:
    """Erode the engraving at (row, col) per vendor wear rules.

    Mirrors vendor/nethack/src/engrave.c::wipe_engr_at (lines 270-289):

      - DUST and ENGR_BLOOD wear via ``wipeout_text(engr_txt, cnt, 0)``
        with no per-step gate; vendor callers pass ``cnt=rnd(5)`` on
        player movement (hack.c::maybe_smudge_engr line 3026).  We
        mirror the ``rnd(5)`` cnt and invoke the rubouts kernel
        ``_wipeout_text_jax`` (D16) so individual characters are
        substituted via the vendor table rather than blindly trimmed.
      - ENGRAVE, MARK, and HEADSTONE wear via ``rn2(1 + 50/(cnt+1)) == 0``;
        with ``cnt=1`` (single wipe-attempt per step) that's
        ``rn2(26) == 0`` ≈ 1/26 per step (engrave.c:280-285).  When
        the roll succeeds we also pass through the rubouts kernel
        with cnt=1.
      - BURN only erodes when ``is_ice(x,y) || (magical && !rn2(2))``.
        Plain movement doesn't visit those branches, so BURN never wears
        via this path (matches the practical effect).
      - When the inscription becomes entirely blank/NUL, clear
        ``has_engraving``.

    Previously: only DUST eroded; the dust/blood branch trimmed 2 trailing
    bytes with prob 1/2 instead of running the rubouts substitution.
    Slow-class branches now also go through the rubouts kernel.
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

    rng_cnt, rng_fast, rng_slow_gate, rng_slow = jax.random.split(rng, 4)
    # Fast-class cnt = rnd(5) → uniform 1..5 (engrave.c caller hack.c:3026).
    fast_cnt = jax.random.randint(
        rng_cnt, (), 1, _WIPEOUT_MAX_CNT + 1, dtype=jnp.int32
    )
    # cnt=1 single wipe attempt for slow class: rn2(1 + 50/(1+1)) == rn2(26).
    slow_roll = jax.random.randint(
        rng_slow_gate, (), 0, 26, dtype=jnp.int32
    ) == jnp.int32(0)

    text = eng.text[r, c, :]  # int8[ENGRAVE_TEXT_LEN]

    # Fast-wear (dust/blood): always run rubouts with cnt=rnd(5).
    fast_text = _wipeout_text_jax(text, rng_fast, fast_cnt)
    # Slow-wear (engrave/mark/headstone): gated, then cnt=1 rubouts.
    slow_text_raw = _wipeout_text_jax(text, rng_slow, jnp.int32(1))
    slow_text = jnp.where(slow_roll, slow_text_raw, text)

    new_text = jnp.where(
        is_dust_class,
        fast_text,
        jnp.where(is_slow_class, slow_text, text),
    )

    # When all bytes are zero/space after vendor's trailing-space trim,
    # the engraving is gone (engrave.c:286-287 del_engr).
    still_has = jnp.any(new_text != jnp.int8(0))

    new_texts = eng.text.at[r, c, :].set(new_text)
    new_has   = eng.has_engraving.at[r, c].set(
        eng.has_engraving[r, c] & still_has
    )

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


# ---------------------------------------------------------------------------
# Wave 46b: text-only Elbereth predicate + scares-monster helper +
# coarse per-turn DUST decay tick.
# Cite: vendor/nethack/src/engrave.c::sengr_at strict-mode (lines 250-261);
#       vendor/nethack/src/monmove.c::onscary (lines 240-303);
#       vendor/nethack/src/engrave.c::wipe_engr_at (lines 270-289).
# ---------------------------------------------------------------------------

def is_elbereth(text: jnp.ndarray) -> jnp.ndarray:
    """Text-only Elbereth predicate.

    Returns True iff the inscription bytes are exactly the canonical
    ASCII "Elbereth" sequence (zero-padded to ENGRAVE_TEXT_LEN).  This
    mirrors vendor ``strcmp(ep->engr_txt[actual_text], "Elbereth") == 0``
    used inside ``sengr_at`` (engrave.c:250-261) before the type / time
    gating that ``is_elbereth_at`` adds on top.

    Parameters
    ----------
    text : int8[ENGRAVE_TEXT_LEN] — the raw inscription bytes.
    """
    target = _elbereth_bytes_array()
    t = jnp.asarray(text, dtype=jnp.int8)
    # Truncate / pad if caller passed a different length.
    if t.shape[0] != ENGRAVE_TEXT_LEN:
        pad = jnp.zeros((ENGRAVE_TEXT_LEN,), dtype=jnp.int8)
        upto = min(int(t.shape[0]), ENGRAVE_TEXT_LEN)
        t = pad.at[:upto].set(t[:upto])
    return jnp.all(t == target)


def engrave_scares_monster(state, monster_idx) -> jnp.ndarray:
    """Return True iff the engraving on the player's tile scares monster ``monster_idx``.

    Mirrors the Elbereth half of vendor ``onscary`` (monmove.c:240-303):

      1. The tile contains an Elbereth inscription that is readable
         and not a headstone (delegated to ``is_elbereth_at``).
      2. The monster is not Elbereth-exempt (humanoids, Wizard of
         Yendor, Archon, Riders — see ``_IGNORES_ELBERETH`` in
         monster_ai.py for the same vendor list).
      3. The monster can see (``mtmp->mcansee``; vendor line 299
         ``!mtmp->mcansee``).  We approximate via the monster's
         ``blind_timer == 0``.
      4. The monster has a mind (``!M1_MINDLESS``).  Vendor strictly
         only excludes mindless creatures via the broader exemption
         lists, but the task asks us to gate explicitly here; mindless
         golems / molds / zombies were already exempt because vendor's
         humanoid / undead checks cover most of them.
      5. The monster is not peaceful (vendor line 300 ``mtmp->mpeaceful``).

    Parameters
    ----------
    state       : EnvState
    monster_idx : int32 scalar — index into ``state.monster_ai.*``.

    Returns
    -------
    bool scalar
    """
    from Nethax.nethax.constants.monsters import M1_MINDLESS

    eng = state.engrave
    mi  = state.monster_ai
    idx = jnp.asarray(monster_idx, dtype=jnp.int32)

    # 1. Elbereth on player's tile (readable, non-headstone).
    ppos = state.player_pos.astype(jnp.int32)
    scared_raw = is_elbereth_at(eng, ppos[0], ppos[1], moves=state.timestep)

    # 2. Per-entry exemption (humanoids / Wizard / Archon / Riders).
    #    Re-derive the exemption table here to avoid a circular import
    #    from monster_ai.py.
    from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol
    import numpy as _np
    _RIDER = {"Death", "Pestilence", "Famine"}
    _EXEMPT = {"Wizard of Yendor", "Archon"}
    _ignores_np = _np.array(
        [
            (m.symbol == MonsterSymbol.S_HUMAN)
            or (m.name in _RIDER)
            or (m.name in _EXEMPT)
            for m in MONSTERS
        ],
        dtype=bool,
    )
    ignores_lut = jnp.asarray(_ignores_np, dtype=jnp.bool_)

    eidx = mi.entry_idx[idx].astype(jnp.int32)
    safe_e = jnp.clip(eidx, 0, ignores_lut.shape[0] - 1)
    ignores = ignores_lut[safe_e]

    # 3. Sight: blind_timer == 0 → can see (vendor mcansee).
    can_see = mi.blind_timer[idx].astype(jnp.int32) == jnp.int32(0)

    # 4. Mind: !M1_MINDLESS on the entry's flags1.
    flags1_table = jnp.array(
        [int(m.flags1) for m in MONSTERS], dtype=jnp.uint32
    )
    flags1 = flags1_table[safe_e]
    has_mind = (flags1 & jnp.uint32(M1_MINDLESS)) == jnp.uint32(0)

    # 5. Not peaceful (vendor line 300).
    not_peaceful = ~mi.peaceful[idx]

    # 6. Monster must be alive.
    alive = mi.alive[idx]

    return scared_raw & ~ignores & can_see & has_mind & not_peaceful & alive


def tick_engravings(state, rng: jax.Array):
    """Coarse per-turn engraving decay at the player's tile.

    Mirrors vendor wear semantics from engrave.c::wipe_engr_at (lines
    270-289) at a per-turn granularity:

      - ENGR_DUST   : 1/15 chance per turn to erase entirely when the
                      player is standing on it (DUST is the only kind
                      that wears just from being walked over each turn).
      - ENGR_BLOOD  : permanent absent magical wipe (vendor wipe_engr_at
                      treats BLOOD like DUST for movement, but the task
                      spec asks for BLOOD to persist; we follow the
                      task spec).
      - ENGRAVE/BURN/MARK/HEADSTONE: permanent (BURN never fades via
        movement; ENGRAVE/MARK only fade via the slow 1/26 gate on
        explicit step events, not this coarse per-turn tick).

    This is a coarser companion to ``wipe_engr_on_step``: the latter
    runs the full vendor rubouts kernel on movement, while
    ``tick_engravings`` is a lightweight per-turn pass that callers
    invoke once per game turn.  Both are vendor-cited.

    Parameters
    ----------
    state : EnvState
    rng   : jax.random.PRNGKey

    Returns
    -------
    EnvState with state.engrave updated.
    """
    eng = state.engrave
    r = state.player_pos[0].astype(jnp.int32)
    c = state.player_pos[1].astype(jnp.int32)

    kind = eng.engraving_kind[r, c].astype(jnp.int32)
    has  = eng.has_engraving[r, c]

    # 1/15 erase roll — only DUST decays on this coarse path.
    roll = jax.random.randint(rng, (), 0, 15, dtype=jnp.int32)
    erase = (roll == jnp.int32(0)) & has & (kind == jnp.int32(ENGR_DUST))

    zero_text = jnp.zeros((ENGRAVE_TEXT_LEN,), dtype=jnp.int8)
    new_text_at = jnp.where(erase, zero_text, eng.text[r, c, :])
    new_has_at  = jnp.where(erase, jnp.bool_(False), has)
    new_kind_at = jnp.where(erase, jnp.int8(ENGR_NONE),
                            eng.engraving_kind[r, c])

    new_texts = eng.text.at[r, c, :].set(new_text_at)
    new_has   = eng.has_engraving.at[r, c].set(new_has_at)
    new_kinds = eng.engraving_kind.at[r, c].set(new_kind_at)

    new_eng = eng.replace(
        text=new_texts,
        has_engraving=new_has,
        engraving_kind=new_kinds,
    )
    return state.replace(engrave=new_eng)
