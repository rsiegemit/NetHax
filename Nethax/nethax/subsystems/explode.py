"""Explosion subsystem — 3x3 AoE damage with per-tile resistance routing.

Canonical source:
    vendor/nethack/src/explode.c::explode (lines 199-696) — 3x3 AoE damage
    around a center tile; per-tile resistance routing via explosionmask
    (lines 26-115).  Vendor's full implementation also handles visual glyphs,
    floor zap_over_floor side-effects, golemeffects, item destruction, shop
    damage, and various corner cases (engulfer, grabbing, etc.).

Wave coverage (this module):
    - Iterates the 3x3 neighbourhood (8 neighbours + center) around
      ``center_pos``.
    - Rolls ``n_dice``d``n_sides`` once per call (vendor: damage is rolled
      once and applied to every target in the AoE — see explode.c line 207
      ``damu = dam`` and the per-tile loop at line 457).
    - Applies that damage to every alive monster at any of the 9 tiles.
    - Applies the same damage to the player if ``player_pos`` is inside the
      3x3 ring.
    - Per-target resistance routing:
        AD_FIRE  -> RESIST_FIRE   / MR_FIRE
        AD_COLD  -> RESIST_COLD   / MR_COLD
        AD_ELEC  -> RESIST_SHOCK  / MR_ELEC
        AD_ACID  -> RESIST_ACID   / MR_ACID
        AD_MAGIC -> MAGIC_RESIST  / (no monster MR bit; no resist for monsters)
      Resisting targets take ``(dam + 1) // 2`` damage (vendor explode.c
      line 538: ``mdam = (dam + 1) / 2``).

Out of scope (not in vendor lines 199-696 either, or deferred):
    - Visual / shield-effect glyph rendering.
    - Item destruction, floor effects (zap_over_floor).
    - Killer/credit attribution, shop damage, gas-spore exposing.
    - Cross-resistance double-damage (resists_cold + AD_FIRE -> 2x dam).
    - inside_engulfer / grabber double-damage corner cases.

JIT-purity:
    - All control flow uses ``jnp.where`` and array masking — no Python
      conditionals on traced values.
    - The 3x3 loop is unrolled (9 fixed tiles) so the trace shape is
      constant.
"""
from __future__ import annotations
from enum import IntEnum

import jax
import jax.numpy as jnp

from Nethax.nethax.rng import dice_roll
from Nethax.nethax.constants.monsters import (
    MR_FIRE, MR_COLD, MR_ELEC, MR_ACID,
)
from Nethax.nethax.subsystems.status_effects import Intrinsic


class DamageType(IntEnum):
    """Subset of vendor AD_* (vendor/nethack/include/monattk.h).

    Only the explosion-relevant damage types are listed.  Values match
    vendor's enum so that callers can pass either the vendor constant or
    this enum interchangeably.
    """
    AD_PHYS  = 0
    AD_MAGIC = 1   # vendor AD_MAGM
    AD_FIRE  = 2
    AD_COLD  = 3
    AD_ELEC  = 4   # vendor AD_ELEC
    AD_ACID  = 5   # vendor AD_ACID


# Module-level aliases for ergonomic external use.
AD_PHYS  = int(DamageType.AD_PHYS)
AD_MAGIC = int(DamageType.AD_MAGIC)
AD_FIRE  = int(DamageType.AD_FIRE)
AD_COLD  = int(DamageType.AD_COLD)
AD_ELEC  = int(DamageType.AD_ELEC)
AD_ACID  = int(DamageType.AD_ACID)


# ---------------------------------------------------------------------------
# Resistance lookup tables — keyed by DamageType int.
# A value of 0 in the MR column means "no monster resist bit applies"
# (e.g. AD_MAGIC: vendor uses `resists_magm(m)` which is a flag predicate
# not represented in the resists_mask bitfield).  In that case the
# explode helper falls back to: monsters have no resistance, hero uses
# MAGIC_RESIST.  Vendor explode.c lines 78-99 handle this branching.
# ---------------------------------------------------------------------------

# Player-side intrinsic id per damage type (0 means "no intrinsic resist").
_PLAYER_RESIST_INTRINSIC = {
    AD_FIRE:  int(Intrinsic.RESIST_FIRE),
    AD_COLD:  int(Intrinsic.RESIST_COLD),
    AD_ELEC:  int(Intrinsic.RESIST_SHOCK),
    AD_ACID:  int(Intrinsic.RESIST_ACID),
    AD_MAGIC: int(Intrinsic.MAGIC_RESIST),
}

# Monster-side MR_* bit per damage type (0 means "no MR bit applies").
_MONSTER_RESIST_BIT = {
    AD_FIRE:  int(MR_FIRE),
    AD_COLD:  int(MR_COLD),
    AD_ELEC:  int(MR_ELEC),
    AD_ACID:  int(MR_ACID),
    AD_MAGIC: 0,
}


def _player_has_resist(state, dmg_type: int) -> jax.Array:
    """Return scalar bool: does the hero resist this damage type?

    Mirrors vendor explode.c explosionmask hero branch (lines 33-71):
    intrinsic + timed-intrinsic OR-merge per damage type.  Damage types
    not in the table (e.g. AD_PHYS) yield False — physical AoE has no
    resist source in vendor either.
    """
    intr_id = _PLAYER_RESIST_INTRINSIC.get(int(dmg_type), 0)
    if intr_id == 0:
        return jnp.bool_(False)
    perm  = state.status.intrinsics[intr_id]
    timed = state.status.timed_intrinsics[intr_id] > jnp.int32(0)
    return (perm | timed).astype(jnp.bool_)


def _monster_resist_mask(mai_resists: jax.Array, dmg_type: int) -> jax.Array:
    """Per-monster bool array: which monsters resist this damage type?

    Reads the per-slot ``resists`` bitmask (populated at spawn from
    MONSTERS[entry_idx].resists_mask — vendor/nethack/src/monst.c MON()).
    Damage types with no MR bit (AD_MAGIC, AD_PHYS) return all-False
    because no monsters bear a generic-magic-resist bit in that field;
    full vendor parity would consult resists_magm(m), which we defer.
    """
    bit = _MONSTER_RESIST_BIT.get(int(dmg_type), 0)
    if bit == 0:
        return jnp.zeros_like(mai_resists, dtype=jnp.bool_)
    return (mai_resists & jnp.int32(bit)) != jnp.int32(0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explode(
    state,
    rng: jax.Array,
    center_pos: jax.Array,
    dmg_type: int,
    n_dice: int,
    n_sides: int,
):
    """Apply a 3x3 AoE explosion centered on ``center_pos``.

    Parameters
    ----------
    state       : EnvState — must expose ``monster_ai`` (with ``pos``,
                  ``alive``, ``hp``, ``resists``) plus ``player_pos``,
                  ``player_hp``, and ``status``.
    rng         : JAX PRNG key (consumed).
    center_pos  : int16[2] (row, col) — explosion epicentre.
    dmg_type    : DamageType / AD_* int value.  Drives resistance routing.
    n_dice      : Python int — number of dice to roll for damage (static).
    n_sides     : Python int — sides per die (static).

    Returns
    -------
    new EnvState with updated ``monster_ai.hp`` / ``monster_ai.alive`` and
    ``player_hp``.

    Notes
    -----
    Vendor explode.c rolls damage once (``dam`` argument is supplied by
    the caller and reused at every tile — see line 207 ``damu = dam`` and
    line 511+ ``int mdam = dam``).  We reproduce that single-roll
    semantics: one ``dice_roll`` for the whole AoE.

    Resisting targets receive ``(dam + 1) // 2`` damage (vendor line 538).
    """
    # ---- Roll damage once ------------------------------------------------
    dam_full = dice_roll(rng, int(n_dice), int(n_sides)).astype(jnp.int32)
    dam_half = ((dam_full + jnp.int32(1)) // jnp.int32(2)).astype(jnp.int32)

    center_r = center_pos[0].astype(jnp.int32)
    center_c = center_pos[1].astype(jnp.int32)

    # ---- Monsters --------------------------------------------------------
    mai = state.monster_ai
    mon_r = mai.pos[:, 0].astype(jnp.int32)
    mon_c = mai.pos[:, 1].astype(jnp.int32)

    # In the 3x3 ring (vendor: 8 neighbours + center) iff Chebyshev <= 1.
    dr = mon_r - center_r
    dc = mon_c - center_c
    cheby = jnp.maximum(jnp.abs(dr), jnp.abs(dc))
    in_aoe = mai.alive & (cheby <= jnp.int32(1))

    mon_resists_mask = _monster_resist_mask(mai.resists, int(dmg_type))
    # Each monster gets dam_full or dam_half depending on resist.
    per_mon_dmg = jnp.where(mon_resists_mask, dam_half, dam_full)
    # Apply only to monsters in AoE.
    per_mon_dmg = jnp.where(in_aoe, per_mon_dmg, jnp.int32(0))

    new_hp    = jnp.maximum(mai.hp - per_mon_dmg, jnp.int32(0))
    new_alive = jnp.where(
        in_aoe & (new_hp <= jnp.int32(0)),
        jnp.bool_(False),
        mai.alive,
    )
    new_mai = mai.replace(hp=new_hp, alive=new_alive)

    # ---- Player ----------------------------------------------------------
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    pdr = pr - center_r
    pdc = pc - center_c
    pcheby = jnp.maximum(jnp.abs(pdr), jnp.abs(pdc))
    player_in_aoe = pcheby <= jnp.int32(1)
    player_resists = _player_has_resist(state, int(dmg_type))
    player_dmg_raw = jnp.where(player_resists, dam_half, dam_full)
    player_dmg = jnp.where(player_in_aoe, player_dmg_raw, jnp.int32(0))
    # Vendor: u.uhp -= damu (no floor at 1; death possible).
    new_player_hp = (state.player_hp - player_dmg).astype(state.player_hp.dtype)

    return state.replace(monster_ai=new_mai, player_hp=new_player_hp)
