"""Teleport gating — vendor-faithful noteleport / level-tele parity helpers.

Canonical source:
    vendor/nethack/src/teleport.c
        noteleport_level()  — line 28-47
        scrolltele()        — line 848 (amulet-of-Yendor 1/3 disorient gate)
        level_tele()        — line 1164 (heaven/Cloud-9 + invocation gate)

This module is purely-static / JIT-friendly: no state mutation lives here,
only the predicates that callers (magic.py, items_scrolls.py) consult.
"""

from __future__ import annotations

import jax.numpy as jnp

from Nethax.nethax.dungeon.branches import Branch


# ---------------------------------------------------------------------------
# Static (branch, level) noteleport mask
# ---------------------------------------------------------------------------
#
# Vendor sets ``svl.level.flags.noteleport`` from each level's dat/*.lua
# ``des.level_flags(... "noteleport" ...)`` directive (vendor/nethack/dat/).
# The Nethax port does not yet replay that flag through DungeonState, so we
# encode the canonical noteleport set as a static lookup keyed by
# (Branch enum, 1-based dungeon level).
#
# Levels covered (vendor parity for the three branches the spec calls out):
#   - Branch.SOKOBAN : all dungeon levels (soko*-*.lua all carry noteleport).
#   - Branch.VLAD    : all 3 tower levels (wizard1/2/3.lua + tower*.lua).
#   - Branch.MAIN    : Dlvl 26 (castle.lua — "the Stronghold" / Castle).
#
# Note: we do NOT mark Gehennom-bottom (sanctum.lua) or the Elemental Planes
# here because those levels are gated by other mechanics (in_endgame /
# invocation) that level_tele handles separately below.
#
# Cite: vendor/nethack/dat/soko*-*.lua, wizard1.lua, wizard2.lua, wizard3.lua,
#       castle.lua — all carry ``des.level_flags("...","noteleport",...)``.
# ---------------------------------------------------------------------------

# Castle / Stronghold sits on Main Dlvl 26 (vendor/nethack/dat/dungeon.lua:
# castle base = -1 → deepest Main level; Nethax branches.py line 827).
_MAIN_NOTELEPORT_LEVEL: int = 26


def is_noteleport_level(branch: jnp.ndarray, level: jnp.ndarray) -> jnp.ndarray:
    """Return True when teleport is forbidden on (branch, level).

    Mirrors ``noteleport_level(&gy.youmonst)`` from vendor teleport.c:30 for
    the natural ``svl.level.flags.noteleport`` clause — the part that fires
    for Sokoban, Vlad's Tower, and the Stronghold.

    Parameters
    ----------
    branch : int  — Branch enum value (state.dungeon.current_branch).
    level  : int  — 1-based dungeon level within ``branch``.

    Returns
    -------
    jnp.bool_ scalar.

    JIT-pure: only jnp.equal / jnp.logical_or are used so the predicate can
    be embedded in jax.lax.cond gates without host-side branching.

    Cite: vendor/nethack/src/teleport.c::noteleport_level lines 38-39.
    """
    b = jnp.asarray(branch).astype(jnp.int32)
    lv = jnp.asarray(level).astype(jnp.int32)

    is_sokoban = b == jnp.int32(int(Branch.SOKOBAN))
    is_vlad    = b == jnp.int32(int(Branch.VLAD))
    is_castle  = (b == jnp.int32(int(Branch.MAIN))) & (
        lv == jnp.int32(_MAIN_NOTELEPORT_LEVEL)
    )
    return is_sokoban | is_vlad | is_castle


# ---------------------------------------------------------------------------
# Endgame predicate (vendor In_endgame(&u.uz))
# ---------------------------------------------------------------------------

def in_endgame(branch: jnp.ndarray) -> jnp.ndarray:
    """True when the hero is on any Elemental Plane.

    Vendor: ``In_endgame(&u.uz)`` checks u.uz.dnum == endgame_dnum.
    Cite: vendor/nethack/include/dungeon.h ``In_endgame`` macro.
    """
    return jnp.asarray(branch).astype(jnp.int32) == jnp.int32(int(Branch.ENDGAME))


def in_sokoban(branch: jnp.ndarray) -> jnp.ndarray:
    """True when the hero is anywhere in the Sokoban branch.

    Vendor: ``In_sokoban(&u.uz)``.
    Cite: vendor/nethack/include/dungeon.h ``In_sokoban`` macro.
    """
    return jnp.asarray(branch).astype(jnp.int32) == jnp.int32(int(Branch.SOKOBAN))


def in_hell(branch: jnp.ndarray) -> jnp.ndarray:
    """True when the hero is anywhere in Gehennom.

    Vendor: ``In_hell(&u.uz)`` / ``Inhell``.
    Cite: vendor/nethack/include/dungeon.h ``In_hell`` macro.
    """
    return jnp.asarray(branch).astype(jnp.int32) == jnp.int32(int(Branch.GEHENNOM))


def in_quest(branch: jnp.ndarray) -> jnp.ndarray:
    """True when the hero is anywhere in the Quest branch.

    Vendor: ``In_quest(&u.uz)``.
    Cite: vendor/nethack/include/dungeon.h ``In_quest`` macro.
    """
    return jnp.asarray(branch).astype(jnp.int32) == jnp.int32(int(Branch.QUEST))
