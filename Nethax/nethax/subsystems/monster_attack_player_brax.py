"""Brax-style flattened rewrite of ``combat.monster_attack_player``.

Same algorithm, same byte-parity contract, same signature.  All
``jax.lax.cond`` / ``jax.lax.switch`` sites are replaced with **always-compute
both branches + ``jnp.where`` mask selection** (the Brax / Google physics
engine pattern).  Under ``jax.vmap`` over seeds, ``lax.cond`` lowers to
``lax.select`` (both branches in HLO) and nested cond chains cause
pathological HLO compile blowup; flat ``jnp.where`` over precomputed masks
produces fusion-friendly HLO that compiles in single-digit minutes.

The mutation pattern used throughout:

    # Original
    state = lax.cond(c, lambda s: s.replace(field=...), lambda s: s, state)
    # Brax flattened
    new   = state.replace(field=...)             # always compute
    state = jax.tree.map(lambda a, b: jnp.where(c, a, b), new, state)

For nested struct.dataclass pytrees this selects every leaf field-by-field
with the same scalar mask, which is byte-equivalent to the cond form.

RNG draws are taken in **exactly the same order** as the original:
    1.  ``split_n(rng, 2)`` -> ``key_hit, key_dmg``
    2.  ``jax.random.split(key_hit)`` -> ``key_hit, key_ac``
    3.  ``rnd(key_hit, 20)`` (to-hit d20)
    4.  ``jax.random.randint(key_ac, ...)``  (AC softening)
    5.  ``split_n(key_dmg, 8)`` + 8x ``randint`` via scan (damage dice)
    6.  ``jax.random.split(rng)`` -> ``rng_engulf, _``  (engulf rng, derived from outer rng)
    7.  ``trigger_lycanthropy(state, rng, ...)``    (uses outer rng)
    8.  ``try_engulf(state, idx, rng_engulf)``       (uses derived engulf key)

All eight RNG channels are consumed regardless of mask outcome, identical to
the original (where ``lax.cond`` under JIT/vmap also pre-traces both
branches, so each branch's draws are functionally pure with respect to
their input key).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.rng import rnd, split_n
from Nethax.nethax.subsystems.combat import (
    compute_ac,
    _MONSTER_PRIMARY_ADTYP_TABLE,
    _AD_WERE,
)
from Nethax.nethax.subsystems.swallow import (
    _IS_ENGULFER as _ENGULFER_TABLE,
)

# Choose Brax helpers when NETHAX_BRAX_ALL=1 (round 2 integration).
import os as _os_mab
if _os_mab.environ.get("NETHAX_BRAX_ALL", "0") == "1":
    from Nethax.nethax.subsystems.combat_helpers_brax import (
        try_engulf_brax as _try_engulf,
    )
else:
    from Nethax.nethax.subsystems.swallow import try_engulf as _try_engulf


def _select_tree(cond, on_true, on_false):
    """Pytree analogue of ``jnp.where(cond, on_true, on_false)``.

    Both ``on_true`` and ``on_false`` must be pytrees with identical
    structure (same Flax dataclass type, same field dtypes).
    """
    return jax.tree.map(lambda a, b: jnp.where(cond, a, b), on_true, on_false)


def monster_attack_player_brax(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Brax-style flattened rewrite of ``monster_attack_player``.

    Byte-equal to ``combat.monster_attack_player``; same RNG order, same
    return signature ``(new_state, damage_dealt)``.
    """
    # ------------------------------------------------------------------
    # RNG split — identical to original.
    # ------------------------------------------------------------------
    key_hit, key_dmg = split_n(rng, 2)
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    player_ac = compute_ac(state)
    alive = mai.alive[idx]

    # ------------------------------------------------------------------
    # To-hit roll (d20 + m_lev + AC_VALUE).  Pure jnp; no cond.
    # ------------------------------------------------------------------
    key_hit, key_ac = jax.random.split(key_hit)
    roll = rnd(key_hit, 20).astype(jnp.int32)
    from Nethax.nethax.subsystems.monster_ai import _monster_level
    species_lev = _monster_level(mai.entry_idx[idx])
    inst_mlev = mai.m_lev[idx].astype(jnp.int32)
    mlev = jnp.where(inst_mlev > jnp.int32(0), inst_mlev, species_lev)
    mlev = jnp.clip(mlev, jnp.int32(1), jnp.int32(49))

    ac_raw = player_ac.astype(jnp.int32)
    ac_neg_roll = jax.random.randint(
        key_ac, (), 1, jnp.maximum(-ac_raw + 1, 2), dtype=jnp.int32
    )
    ac_value = jnp.where(ac_raw >= 0, ac_raw, -ac_neg_roll)
    raw_tmp = ac_value + jnp.int32(10) + mlev
    tmp = jnp.maximum(raw_tmp, jnp.int32(1))
    hit = (tmp > roll) & alive

    # ------------------------------------------------------------------
    # Damage dice — keep the fixed 8-key scan (it's a static-size scan,
    # not a problematic cond).  Pure jnp masking on n_dice.
    # ------------------------------------------------------------------
    raw_n_dice = jnp.clip(mai.attack_dice_n[idx].astype(jnp.int32), 1, 8)
    raw_sides  = jnp.clip(mai.attack_dice_sides[idx].astype(jnp.int32), 1, 12)
    n_dice = jnp.where(mai.is_unwielded[idx], jnp.int32(1), raw_n_dice)
    sides  = jnp.where(mai.is_unwielded[idx], jnp.int32(2), raw_sides)

    def roll_one(carry, key):
        sub_roll = jax.random.randint(
            key, (), minval=1, maxval=sides + 1, dtype=jnp.int32
        )
        return carry, sub_roll

    keys = split_n(key_dmg, 8)
    _, rolls = jax.lax.scan(roll_one, jnp.int32(0), keys)
    take_mask = jnp.arange(8, dtype=jnp.int32) < n_dice
    raw_dmg = jnp.sum(jnp.where(take_mask, rolls, jnp.int32(0))).astype(jnp.int32)

    # ------------------------------------------------------------------
    # adtyp dispatch — flatten the 7-way lax.switch + outer lax.cond(hit).
    #
    # Strategy: compute every branch's effect on (state, base_dmg) up-front
    # and select via jnp.where on per-branch active masks.
    #   branch_idx mapping:  0=PHYS  1=FIRE  2=COLD  3=SLEE  4=ELEC  5=ACID  6=DREN
    # When hit=False, all branches are masked off (eff_dmg=0, state
    # unchanged), matching the original outer lax.cond(hit, switch, skip).
    # ------------------------------------------------------------------
    safe_entry = jnp.clip(
        mai.entry_idx[idx].astype(jnp.int32),
        0,
        _MONSTER_PRIMARY_ADTYP_TABLE.shape[0] - 1,
    )
    adtyp = _MONSTER_PRIMARY_ADTYP_TABLE[safe_entry]

    from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intr, TimedStatus as _TS
    intr = state.status.intrinsics
    fire_res  = intr[int(_Intr.RESIST_FIRE)]
    cold_res  = intr[int(_Intr.RESIST_COLD)]
    shock_res = intr[int(_Intr.RESIST_SHOCK)]
    acid_res  = intr[int(_Intr.RESIST_ACID)]
    sleep_res = intr[int(_Intr.RESIST_SLEEP)]

    _AD_PHYS_V = jnp.int32(0)
    _AD_FIRE_V = jnp.int32(2)
    _AD_COLD_V = jnp.int32(3)
    _AD_SLEE_V = jnp.int32(4)
    _AD_ELEC_V = jnp.int32(6)
    _AD_ACID_V = jnp.int32(8)
    _AD_DREN_V = jnp.int32(16)

    is_phys = (adtyp == _AD_PHYS_V) | (
        (adtyp != _AD_FIRE_V) & (adtyp != _AD_COLD_V) &
        (adtyp != _AD_SLEE_V) & (adtyp != _AD_ELEC_V) &
        (adtyp != _AD_ACID_V) & (adtyp != _AD_DREN_V)
    )  # PHYS is the default branch in the original switch.
    is_fire = adtyp == _AD_FIRE_V
    is_cold = adtyp == _AD_COLD_V
    is_slee = adtyp == _AD_SLEE_V
    is_elec = adtyp == _AD_ELEC_V
    is_acid = adtyp == _AD_ACID_V
    is_dren = adtyp == _AD_DREN_V

    # Per-branch damage values (state-free).  Each is the branch's resolved
    # base damage *before* the outer hit gate.
    dmg_phys = raw_dmg
    dmg_fire = jnp.where(fire_res,  raw_dmg // jnp.int32(2), raw_dmg)
    dmg_cold = jnp.where(cold_res,  raw_dmg // jnp.int32(2), raw_dmg)
    dmg_slee = raw_dmg
    dmg_elec = jnp.where(shock_res, raw_dmg // jnp.int32(2), raw_dmg)
    dmg_acid = jnp.where(acid_res,  raw_dmg // jnp.int32(2), raw_dmg)
    dmg_dren = raw_dmg

    eff_dmg = (
        jnp.where(is_phys, dmg_phys,
        jnp.where(is_fire, dmg_fire,
        jnp.where(is_cold, dmg_cold,
        jnp.where(is_slee, dmg_slee,
        jnp.where(is_elec, dmg_elec,
        jnp.where(is_acid, dmg_acid,
        jnp.where(is_dren, dmg_dren, raw_dmg))))))).astype(jnp.int32)
    )

    # -------- SLEE side-effect (sleep timer increment) --------
    # Original _b_slee always increments the sleep timer when sleep_res is
    # False; only applied when the outer hit gate is active AND adtyp==SLEE.
    slee_active = hit & is_slee
    sleep_dur = jnp.int32(10)
    cur_sleep_timer = state.status.timed_statuses[int(_TS.SLEEP)].astype(jnp.int32)
    incremented_sleep_timer = jnp.where(
        sleep_res, cur_sleep_timer, cur_sleep_timer + sleep_dur,
    )
    # Select new vs old at the leaf level (single int32 slot in the array).
    new_sleep_timer = jnp.where(slee_active, incremented_sleep_timer, cur_sleep_timer)
    new_timed = state.status.timed_statuses.at[int(_TS.SLEEP)].set(
        new_sleep_timer.astype(state.status.timed_statuses.dtype)
    )
    state_after_slee = state.replace(status=state.status.replace(timed_statuses=new_timed))

    # -------- DREN side-effect (level drain) --------
    # Original _b_dren: if drli_res, no-op; else call losexp.  Brax: always
    # call losexp, then tree.map-select on dren_active & ~drli_res.
    from Nethax.nethax.subsystems.experience import losexp as _losexp
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _DRIntr
    drli_res = (
        state_after_slee.status.intrinsics[int(_DRIntr.RESIST_DRAIN)]
        | (state_after_slee.status.timed_intrinsics[int(_DRIntr.RESIST_DRAIN)] > jnp.int32(0))
    )
    state_drained = _losexp(state_after_slee)
    dren_active = hit & is_dren & (~drli_res)
    state_post_dispatch = _select_tree(dren_active, state_drained, state_after_slee)

    # Outer hit gate on damage (original applied dmg = where(hit, eff_dmg, 0)).
    dmg = jnp.where(hit, eff_dmg, jnp.int32(0)).astype(jnp.int32)

    new_hp = jnp.maximum(state_post_dispatch.player_hp - dmg, jnp.int32(0)).astype(jnp.int32)
    new_done = state_post_dispatch.done | (new_hp <= 0)
    new_state = state_post_dispatch.replace(player_hp=new_hp, done=new_done)

    # ------------------------------------------------------------------
    # AD_WERE lycanthropy infection — flatten lax.cond(infect_cond,
    # _trigger_lycan, identity).
    # ------------------------------------------------------------------
    poly = new_state.polymorph
    already_lycan = poly.lycanthropy_form >= jnp.int8(0)
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    prot_shape = new_state.status.intrinsics[int(Intrinsic.PROT_FROM_SHAPE_CHANGERS)]

    infect_cond = (
        hit
        & (adtyp == jnp.int32(_AD_WERE))
        & (~already_lycan)
        & (~prot_shape)
    )

    if _os_mab.environ.get("NETHAX_BRAX_ALL", "0") == "1":
        from Nethax.nethax.subsystems.combat_helpers_brax import (
            trigger_lycanthropy_brax as _trigger_lycan,
        )
    else:
        from Nethax.nethax.subsystems.polymorph import trigger_lycanthropy as _trigger_lycan
    were_form = mai.entry_idx[idx].astype(jnp.int32)

    state_lycan = _trigger_lycan(new_state, rng, were_form)
    new_state = _select_tree(infect_cond, state_lycan, new_state)

    # ------------------------------------------------------------------
    # AT_ENGL engulf — flatten lax.cond(engulf_cond, _try_engulf, identity).
    # ------------------------------------------------------------------
    rng_engulf, _ = jax.random.split(rng)
    is_engulfer = _ENGULFER_TABLE[safe_entry]
    engulf_cond = hit & is_engulfer

    state_engulf = _try_engulf(new_state, idx, rng_engulf)
    new_state = _select_tree(engulf_cond, state_engulf, new_state)

    # ------------------------------------------------------------------
    # "The monster hits!" message — flatten lax.cond(hit, emit, identity).
    # ------------------------------------------------------------------
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    msgs_emitted = _msg_emit(new_state.messages, int(_MsgId.MONSTER_HITS_YOU))
    new_messages = _select_tree(hit, msgs_emitted, new_state.messages)
    new_state = new_state.replace(messages=new_messages)

    return new_state, dmg
