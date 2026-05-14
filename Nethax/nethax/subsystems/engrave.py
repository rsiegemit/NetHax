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

Engraving kinds (engrave.h:ENGR_*):
    0 = none    — empty tile
    1 = dust    — finger in dust (default; fades quickly when stepped on)
    2 = burn    — fire-wand / fire-trap (permanent)
    3 = engrave — athame / digging (permanent until specifically erased)
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

# Max characters of inscription text stored per tile.  vendor stores
# variable-length strings; for JIT-friendly shape we cap at 8 bytes which
# fits 'Elbereth' exactly.
ENGRAVE_TEXT_LEN: int = 8

# Engraving kinds (mirrors engrave.h ENGR_DUST / ENGR_BURN / ENGR_ENGRAVE).
ENGR_NONE: int    = 0
ENGR_DUST: int    = 1
ENGR_BURN: int    = 2
ENGR_ENGRAVE: int = 3

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
    engraving_kind : int8[MAP_H, MAP_W] — ENGR_NONE/DUST/BURN/ENGRAVE.
    """

    text: jnp.ndarray            # int8[MAP_H, MAP_W, ENGRAVE_TEXT_LEN]
    has_engraving: jnp.ndarray   # bool[MAP_H, MAP_W]
    engraving_kind: jnp.ndarray  # int8[MAP_H, MAP_W]

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
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _elbereth_bytes_array() -> jnp.ndarray:
    """Return an int8[ENGRAVE_TEXT_LEN] vector with the ASCII for 'Elbereth'."""
    return jnp.array(_ELBERETH_BYTES, dtype=jnp.int8)


def handle_engrave(state, rng):
    """Player engraves at the current position.

    Wave 5 simplification: always engrave 'Elbereth' in dust regardless of
    inventory (writing finger).  Mirrors the most-common ELBERETHLESS-
    violating action in vendor (engrave.c::doengrave's finger path).

    Effects:
      - state.engrave.has_engraving[row, col]  = True
      - state.engrave.engraving_kind[row, col] = ENGR_DUST
      - state.engrave.text[row, col, :]        = 'Elbereth'
      - state.conduct.violations[ELBERETHLESS] = True

    The scares-undead-and-demons evaluation is performed elsewhere (Wave 6
    will wire it into monster_ai.step's move_toward_player branch).

    Parameters
    ----------
    state : EnvState
    rng   : jax.random.PRNGKey (unused for the dust-Elbereth case but kept
            in the action-handler signature for future RNG-driven kinds).

    Returns
    -------
    Updated EnvState.
    """
    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated

    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)

    bytes_vec = _elbereth_bytes_array()

    eng = state.engrave
    new_text  = eng.text.at[row, col, :].set(bytes_vec)
    new_has   = eng.has_engraving.at[row, col].set(jnp.bool_(True))
    new_kind  = eng.engraving_kind.at[row, col].set(jnp.int8(ENGR_DUST))

    new_engrave = eng.replace(
        text=new_text,
        has_engraving=new_has,
        engraving_kind=new_kind,
    )
    new_state = state.replace(engrave=new_engrave)
    # Conduct: ELBERETHLESS broken on any engrave action (insight.c counter).
    return mark_violated(new_state, int(Conduct.ELBERETHLESS))


def step(state: EngraveState, rng: jax.Array) -> EngraveState:
    """No-op per-turn tick for the engrave subsystem.

    vendor decays dust engravings when monsters/player step over them
    (engrave.c::wipe_engr_at lines 270-290 → wipeout_text engrave.c:120-
    183).  BURN engravings only erode with ``magical && !rn2(2)``
    (engrave.c:278); plain DUST engravings erode on contact.  We defer
    the per-step erode logic to a future wave.
    """
    return state


def is_elbereth_at(eng: EngraveState, row, col) -> jnp.ndarray:
    """Return True if the engraving at ``(row, col)`` is exactly 'Elbereth'.

    Mirrors vendor/nethack/src/engrave.c::sengr_at strict-mode usage
    (engrave.c:250-261), which is the test consulted by monster AI to
    decide whether to flee/avoid the tile (see monster move-toward-
    player gating in monster.c).  JIT-safe.

    Parameters
    ----------
    eng : EngraveState
    row : int / scalar int32
    col : int / scalar int32
    """
    r = jnp.asarray(row, dtype=jnp.int32)
    c = jnp.asarray(col, dtype=jnp.int32)
    has = eng.has_engraving[r, c]
    text_at = eng.text[r, c, :]
    target = _elbereth_bytes_array()
    matches = jnp.all(text_at == target)
    return has & matches
