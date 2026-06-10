"""Brax-style flattened rewrites of player-side combat dispatchers.

This module mirrors ``combat.py`` for the player-attack call sites
(``_single_melee_strike``, ``melee_attack``, ``bump_attack``,
``handle_throw``).  Every ``jax.lax.cond`` / ``jax.lax.switch`` site is
replaced with **always-compute both branches + ``jnp.where`` mask
selection** (the Brax / Google physics engine pattern, also used in
Craftax and ``monster_attack_player_brax``).

Why
---
Under ``jax.vmap`` over seeds, ``lax.cond`` lowers to ``lax.select``
(both branches in HLO) and nested cond chains cause pathological HLO
compile blowup.  Flat ``jnp.where`` over precomputed masks produces
fusion-friendly HLO that compiles in single-digit minutes.

Player melee combat is called on every player attack action, so its
``lax.cond`` chain (skill practice, skill_use, XP award, kill recording,
corpse drop, nemesis hook, artifact-hit effects) directly contributes
to compile cost.

The mutation pattern used throughout:

    # Original
    state = lax.cond(c, lambda s: s.replace(field=...), lambda s: s, state)
    # Brax flattened
    new   = state.replace(field=...)             # always compute
    state = jax.tree.map(lambda a, b: jnp.where(c, a, b), new, state)

Byte-parity contract (must hold against ``combat.py``):

1. RNG draws are taken in *exactly* the same order as the originals.
   ``_single_melee_strike`` keeps the ``split_n(rng, 7)`` head split and
   the same ``key_dmg`` sub-split sequence (key_dmg_w, key_dmg_p,
   key_dmg_w2, key_dmg_arti = fold_in(0xA47F), key_axe =
   fold_in(0xAEC0), key_multi = fold_in(0xA771)).  ``melee_attack``
   keeps the same ``split_n(rng, 2)`` and routes the same per-strike
   keys to ``_single_melee_strike_brax``.

2. All ``lax.cond``/``lax.switch`` gates become ``jnp.where`` masks; both
   branches always execute, so mutations remain byte-identical when the
   helpers are themselves pure.

3. State pytree shape is preserved — ``_select_tree`` selects each leaf
   with the same scalar mask, byte-equivalent to the cond form.

DO NOT modify any existing combat helper; everything imported here is
re-used from ``combat.py`` so we inherit byte-parity wherever the
upstream helpers are already byte-clean.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.rng import rnd, split_n
from Nethax.nethax.subsystems.combat import (
    _abon,
    _dbon,
    _skill_hit_bonus,
    _compute_encumbrance,
    _wielded_enchant,
    _wielded_skill_id,
    _wielded_type_id,
    _polymorph_attack_dice,
    _roll_dice_sum,
    _knight_chivalric_bonus,
    _monk_martial_arts_bonus,
    _samurai_bushido_bonus,
    practice_skill,
    relobj_drop_monster_inventory,
    # Tables / constants reused verbatim.
    _IS_IMMOBILE,
    _OBJECT_MATERIAL,
    _OBJECT_IS_AXE,
    _IS_WOOD_GOLEM,
    _THICK_HIDE,
    _IS_SHADE,
    _HATES_SILVER,
    _KILLED_DROPS_CORPSE,
    _MONSTER_XP_TABLE,
    _MATERIAL_SILVER,
    _FOOD_CATEGORY,
    _CORPSE_TYPE_ID,
    _SKILL_DAM_BONUS,
    _SKILL_WEAPON_TYPE_TO_SKILL,
    N_SKILL_TIERS,
    # Helper-binding aliases (so we use the same identities as the
    # originals — keeps byte parity for fold_in keys, scoring, XP).
    _arti_bonus,
    _arti_idx,
    _arti_hit_effects,
    _wdd,
    _skills_use_skill,
    _xp_experience,
    _xp_more_experienced,
    _scoring_record_kill,
    _scoring_record_kill_pm,
    _scoring_died_for_pm,
)
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.objects import Material


# ---------------------------------------------------------------------------
# Pytree-where helper — same pattern as monster_attack_player_brax.
# ---------------------------------------------------------------------------
def _select_tree(cond, on_true, on_false):
    """Pytree analogue of ``jnp.where(cond, on_true, on_false)``.

    Both ``on_true`` and ``on_false`` must be pytrees with identical
    structure (same Flax dataclass type, same field dtypes).
    """
    return jax.tree.map(lambda a, b: jnp.where(cond, a, b), on_true, on_false)


# ---------------------------------------------------------------------------
# _single_melee_strike — Brax flattened
# ---------------------------------------------------------------------------
def _single_melee_strike_brax(
    state,
    rng: jax.Array,
    target_monster_idx: jnp.ndarray,
    hit_penalty: jnp.ndarray = None,
):
    """Brax-style rewrite of ``combat._single_melee_strike``.

    Returns ``(new_state, dmg, hit)`` byte-identical to the original.  All
    seven ``lax.cond`` sites in the original (skill practice on hit,
    skills_use_skill on hit, XP award on kill, scoring on kill, corpse
    placement on kill, ROT_CORPSE/REVIVE_MON timer on can_place, nemesis
    hook on is_nemesis_kill, artifact hit effects on hit&~poly) are
    replaced with ``jnp.where`` mask selection over precomputed branches.

    Flattened cond/switch count: 7 conds → 0.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    # ------------------------------------------------------------------
    # RNG split — identical to original.
    # ------------------------------------------------------------------
    key_hit, key_dmg, key_monk, key_samurai, key_backstab, key_ac, key_silver = split_n(rng, 7)
    idx = target_monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    target_ac = mai.ac[idx].astype(jnp.int32)
    target_large = mai.is_large[idx]
    target_alive = mai.alive[idx]

    # ------------------------------------------------------------------
    # To-hit roll — pure jnp arithmetic.  No cond.
    # ------------------------------------------------------------------
    roll = rnd(key_hit, 20).astype(jnp.int32)
    abon = _abon(state.player_str, state.player_dex, state.player_xl, state=state)
    skill_bonus = _skill_hit_bonus(state)
    enchant = _wielded_enchant(state)
    pen = jnp.int32(0) if hit_penalty is None else hit_penalty.astype(jnp.int32)
    knight_bonus = _knight_chivalric_bonus(state, idx)
    xl_bonus = state.player_xl.astype(jnp.int32)
    luck = state.player_luck.astype(jnp.int32)
    luck_bonus = jnp.sign(luck) * ((jnp.abs(luck) + 2) // 3)
    uhitinc = state.player_uhitinc.astype(jnp.int32)

    sleeping_bonus = jnp.where(
        (mai.sleep_timer[idx].astype(jnp.int32) > jnp.int32(0)) | mai.asleep[idx],
        jnp.int32(2), jnp.int32(0),
    )
    target_stun_bonus = jnp.where(
        mai.stun_timer[idx].astype(jnp.int32) > jnp.int32(0),
        jnp.int32(2), jnp.int32(0),
    )
    timestep = getattr(state, "timestep", jnp.int32(0))
    fleeing_timer_bonus = jnp.where(
        mai.flee_until_turn[idx].astype(jnp.int32) > timestep.astype(jnp.int32),
        jnp.int32(2), jnp.int32(0),
    )
    fleeing_strat_bonus = jnp.where(
        mai.mstrategy[idx].astype(jnp.int32) == jnp.int32(4),
        jnp.int32(2), jnp.int32(0),
    )
    fleeing_bonus = jnp.maximum(fleeing_timer_bonus, fleeing_strat_bonus)
    paralyzed_bonus = jnp.where(
        mai.paralyzed_timer[idx].astype(jnp.int32) > jnp.int32(0),
        jnp.int32(4), jnp.int32(0),
    )
    entry_i = jnp.clip(mai.entry_idx[idx].astype(jnp.int32), 0, _IS_IMMOBILE.shape[0] - 1)
    immobile_bonus = jnp.maximum(
        jnp.where(_IS_IMMOBILE[entry_i], jnp.int32(4), jnp.int32(0)),
        paralyzed_bonus,
    )

    stunned_timer = state.status.timed_statuses[int(TimedStatus.STUNNED)].astype(jnp.int32)
    stun_hit_penalty = jnp.where(
        stunned_timer > jnp.int32(0), jnp.int32(-1), jnp.int32(0),
    )

    enc = _compute_encumbrance(state)
    enc_penalty = jnp.where(
        enc > jnp.int32(0), -(jnp.int32(2) * enc - jnp.int32(1)), jnp.int32(0),
    )

    confused_timer = state.status.timed_statuses[int(TimedStatus.CONFUSION)].astype(jnp.int32)
    confusion_penalty = jnp.where(
        confused_timer > jnp.int32(0), jnp.int32(-1), jnp.int32(0),
    )

    trap_penalty = jnp.where(state.player_in_trap, jnp.int32(-3), jnp.int32(0))

    tmp = (jnp.int32(1) + abon + target_ac + skill_bonus + enchant + pen + knight_bonus
           + xl_bonus + luck_bonus + uhitinc
           + sleeping_bonus + target_stun_bonus + fleeing_bonus + immobile_bonus
           + stun_hit_penalty + enc_penalty + confusion_penalty + trap_penalty)
    hit = (tmp > roll) & target_alive

    # ------------------------------------------------------------------
    # Damage roll — pure jnp.where; no cond on poly.
    # ------------------------------------------------------------------
    poly_dice, poly_sides = _polymorph_attack_dice(state)
    poly = getattr(state, "polymorph", None)
    is_poly = (
        poly.is_polymorphed
        if (poly is not None and hasattr(poly, "is_polymorphed"))
        else jnp.bool_(False)
    )

    str_dmg = _dbon(state.player_str)
    weapon_enchant = _wielded_enchant(state)
    skill_dmg_id = _wielded_skill_id(state)
    skill_dmg_tier = jnp.clip(
        state.combat.weapon_skill[skill_dmg_id].astype(jnp.int32),
        0, N_SKILL_TIERS - 1,
    )
    skill_dmg_bonus = _SKILL_DAM_BONUS[skill_dmg_tier].astype(jnp.int32)

    wep_type = _wielded_type_id(state)
    dn1, ds1, dn2, ds2 = _wdd(wep_type, target_large)

    key_dmg_w, key_dmg_p, key_dmg_w2 = split_n(key_dmg, 3)
    raw1 = _roll_dice_sum(key_dmg_w, dn1, ds1)
    raw2 = jnp.where(ds2 > 0, _roll_dice_sum(key_dmg_w2, dn2, ds2), jnp.int32(0))
    str_bonus_total = (str_dmg + weapon_enchant + skill_dmg_bonus).astype(jnp.int32)
    weapon_dmg = jnp.maximum(raw1 + raw2 + str_bonus_total, jnp.int32(0)).astype(jnp.int32)

    poly_raw = _roll_dice_sum(key_dmg_p, poly_dice, poly_sides)
    poly_dmg = jnp.maximum(poly_raw + str_dmg, jnp.int32(0)).astype(jnp.int32)

    monk_bonus = _monk_martial_arts_bonus(state, key_monk)
    samurai_bonus = _samurai_bushido_bonus(state, key_samurai)
    role_bonus = (monk_bonus + samurai_bonus).astype(jnp.int32)

    udaminc = state.player_udaminc.astype(jnp.int32)
    base_dmg = jnp.where(is_poly, poly_dmg, weapon_dmg + role_bonus).astype(jnp.int32)
    base_dmg = jnp.maximum(base_dmg + udaminc, jnp.int32(0)).astype(jnp.int32)

    arti_idx = _arti_idx(state)
    target_entry = mai.entry_idx[idx].astype(jnp.int32)
    key_dmg_arti = jax.random.fold_in(key_dmg, jnp.uint32(0xA47F))
    arti_bonus = _arti_bonus(arti_idx, target_entry, key_dmg_arti).astype(jnp.int32)
    arti_bonus = jnp.where(is_poly, jnp.int32(0), arti_bonus)
    base_dmg = (base_dmg + arti_bonus).astype(jnp.int32)

    target_paralyzed = mai.paralyzed_timer[idx].astype(jnp.int32) > jnp.int32(0)
    paralyzed_dmg_bonus = jnp.where(target_paralyzed, jnp.int32(4), jnp.int32(0))
    base_dmg = (base_dmg + paralyzed_dmg_bonus).astype(jnp.int32)

    is_rogue = state.player_role == jnp.int8(int(Role.ROGUE))
    target_fleeing_bs = mai.mstrategy[idx].astype(jnp.int32) == jnp.int32(4)
    target_vulnerable = mai.asleep[idx] | target_fleeing_bs | target_paralyzed
    xl_clamped = jnp.maximum(state.player_xl.astype(jnp.int32), jnp.int32(1))
    backstab_roll = rnd(key_backstab, xl_clamped).astype(jnp.int32)
    backstab_bonus = jnp.where(
        is_rogue & target_vulnerable, backstab_roll, jnp.int32(0),
    )
    base_dmg = (base_dmg + backstab_bonus).astype(jnp.int32)

    stun_dmg_penalty = jnp.where(
        stunned_timer > jnp.int32(0), jnp.int32(-1), jnp.int32(0),
    )
    base_dmg = jnp.maximum(base_dmg + stun_dmg_penalty, jnp.int32(0)).astype(jnp.int32)

    # Silver weapon vs hates-silver target.
    wep_type_silver = _wielded_type_id(state)
    safe_wep_for_mat = jnp.clip(wep_type_silver, 0, _OBJECT_MATERIAL.shape[0] - 1)
    wep_material = _OBJECT_MATERIAL[safe_wep_for_mat].astype(jnp.int32)
    is_silver_weapon = (wep_type_silver >= jnp.int32(0)) & (
        wep_material == jnp.int32(_MATERIAL_SILVER)
    )
    silver_entry = jnp.clip(
        mai.entry_idx[idx].astype(jnp.int32), 0, _HATES_SILVER.shape[0] - 1,
    )
    target_hates_silver_m = _HATES_SILVER[silver_entry]
    silver_d20 = rnd(key_silver, 20).astype(jnp.int32)
    silver_bonus_m = jnp.where(
        is_silver_weapon & target_hates_silver_m & ~is_poly,
        silver_d20,
        jnp.int32(0),
    )
    base_dmg = (base_dmg + silver_bonus_m).astype(jnp.int32)

    # Axe vs wood-golem bonus.
    wep_type_axe = _wielded_type_id(state)
    safe_wep_for_axe = jnp.clip(wep_type_axe, 0, _OBJECT_IS_AXE.shape[0] - 1)
    is_axe_weapon = (wep_type_axe >= jnp.int32(0)) & _OBJECT_IS_AXE[safe_wep_for_axe]
    safe_tgt_for_wood = jnp.clip(
        mai.entry_idx[idx].astype(jnp.int32), 0, _IS_WOOD_GOLEM.shape[0] - 1,
    )
    target_is_wood = _IS_WOOD_GOLEM[safe_tgt_for_wood]
    key_axe = jax.random.fold_in(key_dmg, jnp.uint32(0xAEC0))
    axe_d4 = rnd(key_axe, 4).astype(jnp.int32)
    axe_bonus = jnp.where(
        is_axe_weapon & target_is_wood & ~is_poly,
        axe_d4,
        jnp.int32(0),
    )
    base_dmg = (base_dmg + axe_bonus).astype(jnp.int32)

    # Leather-or-softer vs thick-hide => 0 damage.
    target_thick = _THICK_HIDE[safe_tgt_for_wood]
    leather_or_softer = (wep_type_axe >= jnp.int32(0)) & (
        wep_material <= jnp.int32(int(Material.LEATHER))
    )
    nullify_leather = leather_or_softer & target_thick & ~is_poly
    base_dmg = jnp.where(nullify_leather, jnp.int32(0), base_dmg)

    # Shade no-glare nullification.
    target_is_shade = _IS_SHADE[safe_tgt_for_wood]
    shade_immune = target_is_shade & ~is_silver_weapon & ~is_poly
    base_dmg = jnp.where(shade_immune, jnp.int32(0), base_dmg)

    # Polymorph multi-attack extras (slots 1..NATTK-1).  Static Python
    # loop already; preserved verbatim.
    from Nethax.nethax.subsystems.polymorph import NATTK as _NATTK
    from Nethax.nethax.constants.monsters import AttackType as _AttackType
    AT_NONE_VAL = jnp.uint8(int(_AttackType.AT_NONE))
    poly_types = (
        poly.attack_types if (poly is not None and hasattr(poly, "attack_types"))
        else jnp.zeros((_NATTK,), dtype=jnp.uint8)
    )
    poly_ndice = (
        poly.attack_n_dice if (poly is not None and hasattr(poly, "attack_n_dice"))
        else jnp.zeros((_NATTK,), dtype=jnp.uint8)
    )
    poly_nsides = (
        poly.attack_n_sides if (poly is not None and hasattr(poly, "attack_n_sides"))
        else jnp.zeros((_NATTK,), dtype=jnp.uint8)
    )
    key_multi = jax.random.fold_in(key_dmg, jnp.uint32(0xA771))
    multi_keys = split_n(key_multi, _NATTK)
    extra_multi_total = jnp.int32(0)
    _MAX_DICE = 8
    for i in range(1, _NATTK):
        slot_type = poly_types[i]
        slot_dice = poly_ndice[i].astype(jnp.int32)
        slot_sides = poly_nsides[i].astype(jnp.int32)
        slot_active = (
            (slot_type != AT_NONE_VAL)
            & (slot_dice > jnp.int32(0))
            & (slot_sides > jnp.int32(0))
        )
        safe_sides = jnp.maximum(slot_sides, jnp.int32(1))
        slot_rolls = jax.random.randint(
            multi_keys[i], (_MAX_DICE,), 0, safe_sides, dtype=jnp.int32,
        ) + jnp.int32(1)
        active_mask = jnp.arange(_MAX_DICE, dtype=jnp.int32) < slot_dice
        slot_sum = jnp.sum(jnp.where(active_mask, slot_rolls, jnp.int32(0))).astype(jnp.int32)
        extra_multi_total = extra_multi_total + jnp.where(
            slot_active, slot_sum, jnp.int32(0),
        ).astype(jnp.int32)
    base_dmg = (base_dmg + jnp.where(is_poly, extra_multi_total, jnp.int32(0))).astype(jnp.int32)

    dmg = jnp.where(hit, base_dmg, jnp.int32(0)).astype(jnp.int32)

    new_hp = jnp.maximum(mai.hp[idx] - dmg, jnp.int32(0)).astype(jnp.int32)
    new_alive = (new_hp > 0) & target_alive
    killed = target_alive & ~new_alive

    new_hp_arr = mai.hp.at[idx].set(new_hp)
    new_alive_arr = mai.alive.at[idx].set(new_alive)
    new_mai = mai.replace(hp=new_hp_arr, alive=new_alive_arr)

    new_combat = state.combat.replace(last_hit_landed=hit)
    new_state = state.replace(monster_ai=new_mai, combat=new_combat)

    # ------------------------------------------------------------------
    # Flatten cond #1: skill practice on hit.
    # ------------------------------------------------------------------
    skill_id = _wielded_skill_id(new_state)
    practiced = practice_skill(new_state, skill_id)
    new_state = _select_tree(hit, practiced, new_state)

    # ------------------------------------------------------------------
    # Flatten cond #2: SkillState use_skill on hit.
    # ------------------------------------------------------------------
    wep_type_id = _wielded_type_id(new_state)
    safe_type_id = jnp.clip(
        wep_type_id.astype(jnp.int32), 0, _SKILL_WEAPON_TYPE_TO_SKILL.shape[0] - 1,
    )
    wep_skill_id = _SKILL_WEAPON_TYPE_TO_SKILL[safe_type_id]
    after_use_skill = _skills_use_skill(new_state, wep_skill_id, 1)
    new_state = _select_tree(hit, after_use_skill, new_state)

    # Pacifist conduct mark on killed.
    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated_if
    new_state = mark_violated_if(new_state, int(Conduct.PACIFIST), killed)

    # XP award + kill record.
    entry = jnp.clip(
        mai.entry_idx[idx].astype(jnp.int32),
        0, _MONSTER_XP_TABLE.shape[0] - 1,
    )
    kill_count = _scoring_died_for_pm(new_state.scoring, entry)
    mcloned = new_state.monster_ai.mcloned[idx]
    xp_award = _xp_experience(entry, kill_count, mcloned=mcloned)

    # ------------------------------------------------------------------
    # Flatten cond #3: XP more_experienced on kill.
    # ------------------------------------------------------------------
    more_xp_state = _xp_more_experienced(new_state, xp_award, jnp.int32(0))
    new_state = _select_tree(killed, more_xp_state, new_state)

    # ------------------------------------------------------------------
    # Flatten cond #4: scoring record_kill + record_kill_pm on killed.
    # ------------------------------------------------------------------
    scored_state = new_state.replace(
        scoring=_scoring_record_kill_pm(
            _scoring_record_kill(new_state.scoring, xp_award), entry,
        )
    )
    new_state = _select_tree(killed, scored_state, new_state)

    # ------------------------------------------------------------------
    # Flatten cond #5: corpse placement on killed (with inner cond #6 on
    # can_place becoming a mask over the corpse-timer call).
    # ------------------------------------------------------------------
    safe_entry_corpse = jnp.clip(entry, 0, _KILLED_DROPS_CORPSE.shape[0] - 1)
    drops_corpse = _KILLED_DROPS_CORPSE[safe_entry_corpse]

    # Inline _place_corpse with mask semantics: always compute the
    # ground-stack write *and* the corpse-rot timer, then select on
    # killed.  Inner ``can_place`` already gates the writes via
    # ``jnp.where`` masks in the original (the only inner cond is the
    # corpse-rot timer call).
    gi = new_state.ground_items
    death_pos = new_state.monster_ai.pos[idx].astype(jnp.int32)
    d_row = jnp.clip(death_pos[0], 0, new_state.terrain.shape[2] - 1)
    d_col = jnp.clip(death_pos[1], 0, new_state.terrain.shape[3] - 1)
    branch = new_state.dungeon.current_branch.astype(jnp.int32)
    level = new_state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    n_stack = gi.category.shape[-1]

    def _find_slot(carry, sidx):
        found, gs = carry
        is_empty = gi.category[branch, level, d_row, d_col, sidx] == jnp.int8(0)
        gs = jnp.where(~found & is_empty, sidx, gs)
        found = found | is_empty
        return (found, gs), None

    (gfound, gslot), _ = jax.lax.scan(
        _find_slot,
        (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(n_stack, dtype=jnp.int32),
    )
    # Note: ``can_place`` already requires ``killed`` to make the writes
    # meaningful — but to be safe under the outer select_tree we also
    # mask on ``killed`` so the ground-stack writes are *only* committed
    # when the kill landed.
    can_place = gfound & drops_corpse & killed
    safe_gs = jnp.clip(gslot, 0, n_stack - 1)
    corpse_entry = new_state.monster_ai.entry_idx[idx].astype(jnp.int16)
    placed_gi = gi.replace(
        category=gi.category.at[branch, level, d_row, d_col, safe_gs].set(
            jnp.where(can_place, jnp.int8(_FOOD_CATEGORY),
                      gi.category[branch, level, d_row, d_col, safe_gs])
        ),
        type_id=gi.type_id.at[branch, level, d_row, d_col, safe_gs].set(
            jnp.where(can_place, jnp.int16(_CORPSE_TYPE_ID),
                      gi.type_id[branch, level, d_row, d_col, safe_gs])
        ),
        quantity=gi.quantity.at[branch, level, d_row, d_col, safe_gs].set(
            jnp.where(can_place, jnp.int16(1),
                      gi.quantity[branch, level, d_row, d_col, safe_gs])
        ),
        corpse_entry_idx=gi.corpse_entry_idx.at[branch, level, d_row, d_col, safe_gs].set(
            jnp.where(can_place, corpse_entry,
                      gi.corpse_entry_idx[branch, level, d_row, d_col, safe_gs])
        ),
    )
    placed_state = new_state.replace(ground_items=placed_gi)

    # Flatten cond #6: corpse-rot / revive timer on can_place.
    from Nethax.nethax.subsystems.timer_queue import (
        start_corpse_timer as _start_corpse_timer,
    )
    timer_state = _start_corpse_timer(
        placed_state, branch, level, d_row, d_col, safe_gs,
        corpse_entry.astype(jnp.int32),
    )
    placed_state = _select_tree(can_place, timer_state, placed_state)

    # Outer cond #5 select: only apply corpse placement when killed.
    new_state = _select_tree(killed, placed_state, new_state)

    # Drop slain monster's inventory at the death tile (gated internally
    # by ``killed``; no cond needed).
    new_state = relobj_drop_monster_inventory(new_state, idx, killed)

    # ------------------------------------------------------------------
    # Flatten cond #7: quest nemesis-kill hook.
    # ------------------------------------------------------------------
    from Nethax.nethax.subsystems.quest import on_nemesis_killed, _NEMESIS_IDX_BY_ROLE
    role_idx_q = jnp.clip(
        new_state.player_role.astype(jnp.int32), 0, _NEMESIS_IDX_BY_ROLE.shape[0] - 1,
    )
    nemesis_entry = _NEMESIS_IDX_BY_ROLE[role_idx_q].astype(jnp.int32)
    is_nemesis_kill = killed & (mai.entry_idx[idx].astype(jnp.int32) == nemesis_entry)
    nemesis_state = on_nemesis_killed(new_state, mai.entry_idx[idx])
    new_state = _select_tree(is_nemesis_kill, nemesis_state, new_state)

    # ------------------------------------------------------------------
    # Flatten cond #8: artifact on-hit effects (Vorpal, Magicbane, ...)
    # Original returns ``(new_state, arti_killed)`` — flatten by always
    # computing both branches and selecting per-leaf, with arti_killed
    # selected via jnp.where on the same mask.
    # ------------------------------------------------------------------
    key_arti_hit = jax.random.fold_in(rng, jnp.uint32(0xB33F))
    arti_state, arti_killed_branch = _arti_hit_effects(new_state, idx, key_arti_hit)
    arti_active = hit & ~is_poly
    new_state = _select_tree(arti_active, arti_state, new_state)
    arti_killed = jnp.where(arti_active, arti_killed_branch, jnp.bool_(False))
    killed = killed | arti_killed

    return new_state, dmg, hit


# ---------------------------------------------------------------------------
# melee_attack — Brax flattened
# ---------------------------------------------------------------------------
def melee_attack_brax(
    state,
    rng: jax.Array,
    target_monster_idx: jnp.ndarray,
):
    """Brax-style rewrite of ``combat.melee_attack``.

    Flattens two ``lax.cond`` sites:

      * the ``two_weap`` cond selecting between single-strike and
        two-strike paths;
      * the post-strike message-emit cond on ``hit_landed``.

    RNG order: the ``split_n(rng, 2)`` head is preserved (so the
    per-strike keys match the original).  Both strike paths always
    execute under vmap; the result is selected by ``jnp.where``.

    Flattened cond/switch count: 2 conds → 0 (plus 7→0 conds flattened
    inside ``_single_melee_strike_brax``, so 9 total in the call chain).
    """
    two_weap = state.combat.two_weapon

    rng_a, rng_b = split_n(rng, 2)

    # ------------------------------------------------------------------
    # Compute BOTH strike paths up front, then select via _select_tree.
    # ------------------------------------------------------------------
    # Single path: one strike at zero penalty.
    s_single, dmg_single, hit_single = _single_melee_strike_brax(
        state, rng_a, target_monster_idx,
    )

    # Double path: two strikes, each at -1 hit penalty, threaded through
    # the same RNG keys as the original.
    s1, dmg1, hit1 = _single_melee_strike_brax(
        state, rng_a, target_monster_idx, hit_penalty=jnp.int32(-1),
    )
    s2, dmg2, hit2 = _single_melee_strike_brax(
        s1, rng_b, target_monster_idx, hit_penalty=jnp.int32(-1),
    )
    dmg_double = dmg1 + dmg2
    hit_double = hit1 | hit2

    new_state = _select_tree(two_weap, s2, s_single)
    dmg = jnp.where(two_weap, dmg_double, dmg_single).astype(jnp.int32)
    hit_landed = jnp.where(two_weap, hit_double, hit_single)

    # ------------------------------------------------------------------
    # Flatten "You hit the monster" message-emit cond on hit_landed.
    # ------------------------------------------------------------------
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    emitted = _msg_emit(new_state.messages, int(_MsgId.YOU_HIT_MONSTER))
    new_messages = _select_tree(hit_landed, emitted, new_state.messages)
    new_state = new_state.replace(messages=new_messages)
    return new_state, dmg, hit_landed


# ---------------------------------------------------------------------------
# bump_attack — Brax flattened
# ---------------------------------------------------------------------------
def bump_attack_brax(state, rng: jax.Array, target_pos: jnp.ndarray):
    """Brax-style rewrite of ``combat.bump_attack``.

    The original ``bump_attack`` already used a ``jax.tree_util.tree_map``
    over a ``jnp.where(found, ...)`` mask for the no-monster fallback —
    so the only structural change here is to dispatch into
    ``melee_attack_brax`` (which is itself fully cond-free).  The
    monster-died "step onto tile" logic is already pure ``jnp.where``.

    Flattened cond/switch count: 0 conds at this level (chain inherits 9
    from melee_attack_brax + _single_melee_strike_brax).
    """
    target_pos_i32 = target_pos.astype(jnp.int32)
    mai = state.monster_ai

    pos_i32 = mai.pos.astype(jnp.int32)
    matches = (
        (pos_i32[:, 0] == target_pos_i32[0])
        & (pos_i32[:, 1] == target_pos_i32[1])
        & mai.alive
    )
    idx = jnp.argmax(matches).astype(jnp.int32)
    found = jnp.any(matches)

    safe_idx = jnp.where(found, idx, jnp.int32(0))

    attacked_state, _dmg, _hit = melee_attack_brax(state, rng, safe_idx)

    new_state = _select_tree(found, attacked_state, state)

    monster_died = found & ~new_state.monster_ai.alive[safe_idx]
    new_player_pos = jnp.where(
        monster_died,
        target_pos.astype(jnp.int16),
        new_state.player_pos,
    )
    new_state = new_state.replace(player_pos=new_player_pos)

    return new_state


# ---------------------------------------------------------------------------
# handle_throw — Brax flattened
# ---------------------------------------------------------------------------
def handle_throw_brax(state, rng):
    """Brax-style rewrite of ``combat.handle_throw``.

    Flattens the single ``jax.lax.cond(can_throw, _do_throw, identity)``
    by always invoking ``thrown_attack`` (with a sentinel slot when
    nothing is throwable) and selecting via ``_select_tree``.  Note
    ``thrown_attack`` is invoked with the same RNG key regardless of
    branch, so under vmap both branches consume the same draws as the
    original (matching the cond's pre-trace semantics).

    Flattened cond/switch count: 1 cond → 0.
    """
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.subsystems.pending_action import resolve_slot
    # Import inside the function to avoid a circular import at module
    # load (combat.py imports throwing.py at module-top, throwing.py
    # imports combat_brax at runtime only).
    from Nethax.nethax.subsystems.combat import thrown_attack

    items = state.inventory.items
    quiver = state.inventory.quiver.astype(jnp.int32)
    has_quiver = quiver >= jnp.int32(0)

    is_weapon = items.category == jnp.int8(ItemCategory.WEAPON)
    in_stock = items.quantity > jnp.int16(0)
    valid_weap = is_weapon & in_stock
    first_weap = jnp.argmax(valid_weap).astype(jnp.int32)
    has_weap = jnp.any(valid_weap)

    fallback_slot = jnp.where(has_quiver, quiver, first_weap)
    can_throw = has_quiver | has_weap

    chosen = resolve_slot(state, fallback_slot)
    safe_chosen = jnp.clip(chosen, 0, valid_weap.shape[0] - 1)
    chosen_is_thrown = valid_weap[safe_chosen]
    slot = jnp.where(chosen_is_thrown, safe_chosen, fallback_slot).astype(jnp.int32)

    # Always invoke thrown_attack so both vmap branches compute the same
    # draws as the original cond pre-trace.  Slot is clamped to a valid
    # range; when can_throw is False we restore the original state via
    # _select_tree, so any side effects of thrown_attack on the
    # no-throw path are discarded.
    safe_slot = jnp.where(can_throw, slot, jnp.int32(0))
    direction = jnp.array([0, 1], dtype=jnp.int32)
    thrown_state = thrown_attack(state, rng, safe_slot, direction)

    return _select_tree(can_throw, thrown_state, state)
