"""Brax-style flattened rewrites of ``mattackm`` and ``monster_cast_damage``.

Why this file exists
--------------------
Under ``jax.vmap`` (the validator's per-seed fast path), every
``jax.lax.cond`` / ``jax.lax.switch`` lowers to ``lax.select``: BOTH
branches are always compiled into the HLO graph and the result is
masked.  When such branches are nested — as they are in
``monster_cast_damage`` (Python-``if`` chain over ``spellnum``) and in
the surrounding pet/cast pipelines that call ``mattackm`` from a per-slot
``lax.scan`` — the HLO blows up and the H100 fusion pass struggles.

Brax (and Craftax, the JAX NetHack-style env) avoid this by **always**
computing every branch and using ``jnp.where`` to pick the kept result.
The HLO is then flat and fusion-friendly.

Byte-parity contract
--------------------
Both rewrites are exact drop-in replacements for the originals:

* RNG draw order preserved — every ``jax.random.split`` happens at the
  same point in the same sequence as the original.
* Mutations are byte-identical because they are realised via
  ``jnp.where`` on precomputed values (same dtypes, same shapes).
* Returned state pytrees preserve the exact same shape and dtypes; only
  ``state.monster_ai.hp`` and ``state.monster_ai.alive`` are touched, at
  the same index, with the same value as the original.

Conditionals flattened
----------------------
``mattackm``           : the original already uses pure ``jnp.where`` for
                         every per-slot decision (hit/miss, slot-active,
                         confused bonus, elf-orc bonus, AC softening).
                         The only ``lax`` primitives present are two
                         **fixed-size** ``lax.scan`` calls (NATTK=6 outer,
                         8-die inner).  Per the Craftax memory note —
                         "lax.scan over fmon_order + jnp.where masking;
                         Python loop = 400× HLO, scan = 1×" — those
                         scans are preserved.  This Brax pass is a
                         structural mirror that makes the masking
                         pattern explicit (no hidden cond/switch
                         remains).

``monster_cast_damage``: the original is a Python ``if``-chain over a
                         **compile-time** ``int`` ``spellnum``.  In the
                         Brax form ``spellnum`` is accepted as a JAX
                         scalar; every formula is computed
                         unconditionally from the same ``rng`` and the
                         result is selected via a flat ``jnp.where``
                         tower.  Because every branch consumes ``rng``
                         the same way (split → vmap of randint), the RNG
                         contract is identical to the original — which
                         in the Python-``if`` form only ever evaluated
                         one branch but consumed exactly the same key.

The signatures match the originals byte-for-byte so the Brax versions
are drop-in callable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems.monster_ai import (
    MCAST_CLERIC,
    MCAST_FIRE_PILLAR,
    MCAST_GEYSER,
    MCAST_LIGHTNING,
    MCAST_PSI_BOLT,
    _MONSTER_ATTACK_AATYP_TABLE,
    _MONSTER_ATTACK_N_TABLE,
    _MONSTER_ATTACK_S_TABLE,
    _MONSTER_FLAGS2_TABLE,
    _MONSTER_LEVEL_TABLE,
    _NATTK,
    _roll_dice_dynamic,
)


# ---------------------------------------------------------------------------
# monster_cast_damage — Brax flattening of the Python-``if`` dispatch.
# ---------------------------------------------------------------------------

def monster_cast_damage_brax(rng: jax.Array, spellnum,
                             ml: jnp.ndarray) -> jnp.ndarray:
    """Brax-style flattened dispatch of vendor spell damage formulas.

    Byte-equal to ``monster_cast_damage``: same RNG order, same return
    dtype/shape.  ``spellnum`` may be a Python ``int`` (matches the
    original signature) **or** a ``jnp.ndarray`` scalar — both forms
    select the same branch through ``jnp.where`` masking.

    Vendor mapping (mcastu.c):
        MCAST_PSI_BOLT    → d((ml/2)+1, 6)
        MCAST_FIRE_PILLAR → d(8, 6)
        MCAST_GEYSER      → d(8, 6)
        MCAST_LIGHTNING   → d(8, 6)
        MCAST_CLERIC      → d((ml/2)+1, 6)

    All five formulas share a single ``rng`` (no split between
    branches), exactly as in the original — the original called one of
    five helpers that each consumed ``rng`` directly via
    ``_roll_dice_dynamic``.  The Brax form computes every helper using
    the same ``rng`` and masks; because the original only ever invoked
    one helper per call, and that helper's draw order is identical to
    each of the four branches we compute here, the chosen branch's bits
    are byte-identical.
    """
    # n_dice for the level-scaled branches (psi_bolt / cleric).  Vendor:
    #   n_dice = max(1, ml//2 + 1)
    ml32 = ml.astype(jnp.int32) if hasattr(ml, "astype") else jnp.int32(ml)
    n_dice_scaled = jnp.maximum(jnp.int32(1),
                                ml32 // jnp.int32(2) + jnp.int32(1))

    # All four distinct rolls — every branch uses the same ``rng`` so
    # the chosen branch is bit-for-bit identical to the original.
    dmg_psi   = _roll_dice_dynamic(rng, n_dice_scaled,   6)
    dmg_fire  = _roll_dice_dynamic(rng, jnp.int32(8),    6)
    dmg_geys  = _roll_dice_dynamic(rng, jnp.int32(8),    6)
    dmg_lite  = _roll_dice_dynamic(rng, jnp.int32(8),    6)
    dmg_cler  = _roll_dice_dynamic(rng, n_dice_scaled,   6)

    sn = jnp.asarray(spellnum, dtype=jnp.int32)
    is_fire = sn == jnp.int32(MCAST_FIRE_PILLAR)
    is_geys = sn == jnp.int32(MCAST_GEYSER)
    is_lite = sn == jnp.int32(MCAST_LIGHTNING)
    is_cler = sn == jnp.int32(MCAST_CLERIC)
    # MCAST_PSI_BOLT is the default branch (matches original "default: psi
    # bolt").

    out = jnp.where(
        is_fire, dmg_fire,
        jnp.where(
            is_geys, dmg_geys,
            jnp.where(
                is_lite, dmg_lite,
                jnp.where(is_cler, dmg_cler, dmg_psi),
            ),
        ),
    )
    return out.astype(jnp.int32)


# ---------------------------------------------------------------------------
# mattackm — Brax mirror of the monster-vs-monster melee NATTK loop.
# ---------------------------------------------------------------------------

def mattackm_brax(state, attacker_idx: jnp.ndarray, defender_idx: jnp.ndarray,
                  rng: jax.Array) -> object:
    """Brax-style flattened rewrite of ``mattackm``.

    Byte-equal to ``monster_ai.mattackm``: same RNG order, same return
    state shape, same ``hp``/``alive`` mutation at the defender index.

    The vendor NATTK=6 outer loop and the 8-die inner loop are both
    fixed-size and ``cond``-free, so we keep ``lax.scan`` for them
    (per the Craftax memory note — Python-unrolling a fixed scan blows
    HLO up 400×).  Every per-slot decision is realised via plain
    ``jnp.where`` masking; there is no ``lax.cond`` or ``lax.switch``
    anywhere in this function.

    Cite: vendor/nethack/src/mhitm.c::mattackm lines 293-592; permonst.h
    NATTK = 6 (line 48).
    """
    a = attacker_idx.astype(jnp.int32)
    d = defender_idx.astype(jnp.int32)
    mai = state.monster_ai

    # ---- gates: alive on both sides, distinct slots ----------------------
    same_slot = (a == d)
    both_alive = mai.alive[a] & mai.alive[d]
    can_strike_base = both_alive & ~same_slot

    # ---- attacker / defender table lookups -------------------------------
    a_entry = jnp.clip(
        mai.entry_idx[a].astype(jnp.int32),
        0, _MONSTER_ATTACK_AATYP_TABLE.shape[0] - 1,
    )
    d_entry = jnp.clip(
        mai.entry_idx[d].astype(jnp.int32),
        0, _MONSTER_ATTACK_AATYP_TABLE.shape[0] - 1,
    )
    a_lev = jnp.clip(_MONSTER_LEVEL_TABLE[a_entry].astype(jnp.int32), 1, 30)

    # ---- to-hit bonuses (pure jnp.where) ---------------------------------
    defender_confused = mai.confuse_timer[d] > jnp.int16(0)
    defender_helpless = (
        mai.asleep[d] | (mai.paralyzed_timer[d] > jnp.int16(0))
    )
    bonus_confused = jnp.where(defender_confused | defender_helpless,
                               jnp.int32(4), jnp.int32(0))

    _M2_ORC: int = 0x00000004
    _M2_ELF: int = 0x00000008
    a_flags2 = _MONSTER_FLAGS2_TABLE[a_entry]
    d_flags2 = _MONSTER_FLAGS2_TABLE[d_entry]
    is_elf = (a_flags2 & jnp.int32(_M2_ELF)) != 0
    is_orc = (d_flags2 & jnp.int32(_M2_ORC)) != 0
    bonus_elf_orc = jnp.where(is_elf & is_orc, jnp.int32(1), jnp.int32(0))

    # ---- AC softening for negative AC (single RNG draw, masked select) ---
    def_ac_raw = mai.ac[d].astype(jnp.int32)
    rng, key_ac = jax.random.split(rng)
    ac_neg_roll = jax.random.randint(
        key_ac, (), 1, jnp.maximum(-def_ac_raw + 1, 2), dtype=jnp.int32,
    )
    ac_value = jnp.where(def_ac_raw >= 0, def_ac_raw, -ac_neg_roll)
    base_tmp = jnp.maximum(
        ac_value + jnp.int32(10) + a_lev + bonus_confused + bonus_elf_orc,
        jnp.int32(1),
    )

    # ---- NATTK attack-slot table -----------------------------------------
    AT_NONE = jnp.int16(0)
    aatyp_row = _MONSTER_ATTACK_AATYP_TABLE[a_entry]
    n_row     = _MONSTER_ATTACK_N_TABLE[a_entry]
    s_row     = _MONSTER_ATTACK_S_TABLE[a_entry]

    # Fixed-size key split — matches original RNG order exactly.
    nattk_keys = jax.random.split(rng, _NATTK)

    # ---- per-slot scan body (no cond/switch — pure jnp.where) ------------
    def _attack_step(carry, idx):
        cur_def_hp, cur_def_alive, struck = carry
        key_i = nattk_keys[idx]
        aatyp_i = aatyp_row[idx]
        n_raw   = n_row[idx].astype(jnp.int32)
        s_raw   = s_row[idx].astype(jnp.int32)
        n_dice  = jnp.clip(n_raw, 1, 8)
        sides   = jnp.clip(s_raw, 1, 12)

        slot_active = (aatyp_i != AT_NONE) & (n_raw > 0)
        can_attack = can_strike_base & slot_active & cur_def_alive

        key_hit, key_dmg = jax.random.split(key_i)
        roll = jax.random.randint(
            key_hit, (), 1, jnp.int32(21) + idx, dtype=jnp.int32,
        )
        hit = (base_tmp > roll) & can_attack

        # Fixed 8-key inner scan (static size — Brax-compatible).
        keys_d = jax.random.split(key_dmg, 8)

        def _roll_one(c, k):
            sub = jax.random.randint(k, (), 1, sides + 1, dtype=jnp.int32)
            return c, sub

        _, rolls = jax.lax.scan(_roll_one, jnp.int32(0), keys_d)
        take = jnp.arange(8, dtype=jnp.int32) < n_dice
        raw_dmg = jnp.sum(
            jnp.where(take, rolls, jnp.int32(0))
        ).astype(jnp.int32)

        # Brax mask: hit gates the damage; alive gates HP underflow.
        dmg = jnp.where(hit, raw_dmg, jnp.int32(0))
        new_def_hp = jnp.maximum(cur_def_hp - dmg, jnp.int32(0))
        new_def_alive = cur_def_alive & (new_def_hp > jnp.int32(0))
        return (new_def_hp, new_def_alive, struck | hit), None

    init = (mai.hp[d].astype(jnp.int32), mai.alive[d], jnp.bool_(False))
    (final_def_hp, final_def_alive, _struck), _ = jax.lax.scan(
        _attack_step, init, jnp.arange(_NATTK, dtype=jnp.int32),
    )

    # ---- byte-identical defender mutation (.at[].set) --------------------
    new_hp_arr    = mai.hp.at[d].set(final_def_hp.astype(mai.hp.dtype))
    new_alive_arr = mai.alive.at[d].set(final_def_alive)

    new_mai = mai.replace(hp=new_hp_arr, alive=new_alive_arr)
    return state.replace(monster_ai=new_mai)
