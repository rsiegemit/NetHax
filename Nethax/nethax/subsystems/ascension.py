"""Wave 5 Phase 4b — Ascension condition (endgame).

Ascension is the win condition.  Per vendor/nethack/src/end.c the player
wins (how == ASCENDED, end.c:1064/1181/1300/1344/1422) when:

    1. They are on the Astral Plane (Branch.ENDGAME, level 5).
    2. They are standing on an altar whose alignment matches their own.
    3. They are carrying the Amulet of Yendor in their inventory.

Wave 6 simplification (documented divergence from vendor):
    Vendor requires the player to issue ``#offer`` on the altar
    (pray.c::dosacrifice → offer_real_amulet, which then sets ``done``
    with how=ASCENDED and routes through end.c::done()).  We accept the
    weaker "step onto matching altar while carrying the Amulet" trigger.
    The full offering action will be wired through dispatch_action.OFFER
    in a future wave.

Citations:
    vendor/nethack/src/end.c              — done()/really_done() drive
                                            the ASCENDED game-over path
                                            (line 1064: how == ASCENDED
                                            short-circuits killer fmt).
    vendor/nethack/src/pray.c::dosacrifice — vendor entry; #offer triggers
                                              ascension when on matching
                                              altar with Amulet.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.dungeon.branches import Branch
from Nethax.nethax.dungeon.endgame import (
    ASTRAL_ALTAR_LAWFUL,
    ASTRAL_ALTAR_NEUTRAL,
    ASTRAL_ALTAR_CHAOTIC,
    ASTRAL_ALIGN_LAWFUL,
    ASTRAL_ALIGN_NEUTRAL,
    ASTRAL_ALIGN_CHAOTIC,
)
from Nethax.nethax.subsystems.scoring import (
    Achievement,
    record_achievement,
    add_score,
)
from Nethax.nethax.subsystems.items_jewelry import AmuletEffect
from Nethax.nethax.subsystems.inventory import ItemCategory


# Endgame level number (1-based) for the Astral Plane.
ASTRAL_LEVEL: int = 5


def player_holds_amulet(state) -> jnp.ndarray:
    """Return bool — True if player carries the Amulet of Yendor.

    Scans the inventory for an Item with category=AMULET and
    type_id=AmuletEffect.YENDOR.
    """
    inv = state.inventory.items
    is_amulet = inv.category == jnp.int8(int(ItemCategory.AMULET))
    is_yendor = inv.type_id == jnp.int16(int(AmuletEffect.YENDOR))
    qty_ok    = inv.quantity > jnp.int16(0)
    return jnp.any(is_amulet & is_yendor & qty_ok)


def on_astral_plane(state) -> jnp.ndarray:
    """Return bool — True if player is on the Astral Plane."""
    in_endgame = state.dungeon.current_branch == jnp.int8(int(Branch.ENDGAME))
    on_astral  = state.dungeon.current_level == jnp.int8(ASTRAL_LEVEL)
    return in_endgame & on_astral


def _altar_alignment_at(row: int, col: int) -> int:
    """Return alignment code for the altar at (row, col) on Astral, or -1."""
    if (row, col) == ASTRAL_ALTAR_LAWFUL:
        return ASTRAL_ALIGN_LAWFUL
    if (row, col) == ASTRAL_ALTAR_NEUTRAL:
        return ASTRAL_ALIGN_NEUTRAL
    if (row, col) == ASTRAL_ALTAR_CHAOTIC:
        return ASTRAL_ALIGN_CHAOTIC
    return -1


def on_matching_altar(state) -> jnp.ndarray:
    """Return bool — True if player stands on an altar matching their alignment.

    Computed via JIT-safe ops: compare player_pos against each of the three
    canonical altar coords, derive the alignment via gather, then compare
    against player_align.
    """
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)

    # Build altar tables (rows, cols, alignments).
    altar_rows = jnp.array(
        [ASTRAL_ALTAR_LAWFUL[0],
         ASTRAL_ALTAR_NEUTRAL[0],
         ASTRAL_ALTAR_CHAOTIC[0]],
        dtype=jnp.int32,
    )
    altar_cols = jnp.array(
        [ASTRAL_ALTAR_LAWFUL[1],
         ASTRAL_ALTAR_NEUTRAL[1],
         ASTRAL_ALTAR_CHAOTIC[1]],
        dtype=jnp.int32,
    )
    altar_aligns = jnp.array(
        [ASTRAL_ALIGN_LAWFUL,
         ASTRAL_ALIGN_NEUTRAL,
         ASTRAL_ALIGN_CHAOTIC],
        dtype=jnp.int32,
    )

    # Find any altar that matches both the player position and alignment.
    pos_match    = (altar_rows == pr) & (altar_cols == pc)
    align_match  = altar_aligns == state.player_align.astype(jnp.int32)
    return jnp.any(pos_match & align_match)


def check_ascension(state) -> jnp.ndarray:
    """Return bool — True if the player has met the ascension condition.

    All three conditions must hold:
        1. on_astral_plane(state)
        2. on_matching_altar(state)
        3. player_holds_amulet(state)

    Per vendor/nethack/src/end.c (ASCENDED how-code branches at
    end.c:1064, 1181, 1300, 1344, 1422) and src/pray.c::dosacrifice
    (the #offer route that sets how=ASCENDED).

    JIT-safe — returns a jnp.bool_ scalar.
    """
    return on_astral_plane(state) & on_matching_altar(state) & player_holds_amulet(state)


def ascend(state):
    """Perform ascension.

    Sets state.done=True, records the ASCENDED achievement, and adds a
    basic ascension score bonus (Wave 5 placeholder; the full topten
    formula lives in Wave 6).

    Citation: vendor/nethack/src/end.c::done() path with how=ASCENDED,
    leading into really_done() (end.c:1132-) and topten scoring.  The
    flat 50000 bonus here mirrors vendor's bonus-loaded ASCENDED scoring
    (the full vendor formula adds: ualign + experience + gold +
    artifacts, see end.c::artifact_score).
    """
    # Mark game over.
    new_done = jnp.bool_(True)

    # Record the achievement + add a flat 50000-point ascension bonus
    # (vendor uses a complex multiplier; Wave 5 keeps it simple).
    new_scoring = record_achievement(state.scoring, int(Achievement.ASCENDED))
    new_scoring = add_score(new_scoring, jnp.int32(50000))

    return state.replace(done=new_done, scoring=new_scoring)


def maybe_ascend(state):
    """If ascension condition is met, perform ascension; else return state.

    Convenience wrapper for env.step.  JIT-safe via jax.lax.cond.
    """
    return jax.lax.cond(
        check_ascension(state),
        ascend,
        lambda s: s,
        state,
    )
