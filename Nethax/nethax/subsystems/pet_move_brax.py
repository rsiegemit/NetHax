"""Brax-style flattening of ``pet_move`` (see monster_ai.py:4032 _pet_move_body).

Motivation
----------
Under ``jax.vmap`` over seeds, ``jax.lax.cond`` lowers to ``jax.lax.select``,
which evaluates BOTH branches and selects.  For nested cond chains this causes
pathological HLO compile blow-up (hours).  Google Brax's pattern — always
compute every branch, select with ``jnp.where`` on precomputed scalar masks —
produces flat, fusion-friendly HLO that compiles in single-digit minutes.

What this file does
-------------------
Rewrites the four ``jax.lax.cond`` sites in ``_pet_move_body``:

    1. flee-on-low-HP cond  → mask on position field
    2. outer is_pet gate     → tree-map mask between (transformed state) and
                               (entry state)
    3. has_target dispatch   → tree-map mask between (attack state) and
                               (no-target state)
    4. within_follow_range   → tree-map mask between (follow state) and
                               (explore state)

The 0a/0b/0b' "bookkeeping" phases at the top of pet_move are already
``jnp.where``-style; they are reproduced verbatim.

Byte-parity caveats
-------------------
* ``rng`` is a functional ``jax.Array`` — splitting it more than the original
  is byte-safe because downstream consumers receive the unchanged input.
* ``state.vendor_rng`` is a STATEFUL ISAAC64 stream.  ``vendor_pet_dog_move``
  (called by the FOLLOW branch when ``vendor_mode=True``) advances it.
  Eagerly calling it in non-follow seeds would corrupt the stream.
  → In ``vendor_mode=True`` we keep a single ``jax.lax.cond`` around the
    follow branch.  Everything else is Brax-flattened.
* ``mattackm`` uses only its rng argument (a fresh split of the input rng),
  so calling it speculatively is byte-safe.  We pass ``defender_idx=monster_idx``
  (self) when there is no target; ``mattackm``'s ``same_slot`` gate makes
  this a no-op.

Conds flattened: 3 of 4.  Skipped: vendor-mode follow dispatch
(``vendor_pet_dog_move`` mutates ``vendor_rng``).

Signature matches ``pet_move``: ``pet_move_brax(state, rng, monster_idx)``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from Nethax.nethax.parity_mode import use_vendor_rng as _use_vendor_rng
from Nethax.nethax.subsystems.monster_ai import (
    MAX_MONSTERS_PER_LEVEL,
    _DOG_SIZE_NUTRIT_MULT_TABLE,
    _DOG_STARVE,
    _DOG_WEAK,
    _M2_DEMON,
    _M2_UNDEAD,
    _MAP_H,
    _MAP_W,
    _MONSTER_LEVEL_TABLE,
    _MONSTER_SIZE_TABLE,
    _OBJ_NUTRITION_TABLE,
    _chebyshev_dist,
    _has_flag2,
    apply_confusion_to_step,
)

# Round 2 brax integration.
import os as _os_pm
if _os_pm.environ.get("NETHAX_BRAX_ALL", "0") == "1":
    from Nethax.nethax.subsystems.mattackm_brax import mattackm_brax as mattackm
    from Nethax.nethax.subsystems.pathfind_step_brax import pathfind_step_brax as pathfind_step
else:
    from Nethax.nethax.subsystems.monster_ai import mattackm, pathfind_step


def _select_state(mask: jnp.ndarray, s_then, s_else):
    """Tree-map ``jnp.where(mask, s_then_leaf, s_else_leaf)`` across pytree.

    ``mask`` is a scalar boolean; ``jnp.where`` broadcasts it across every
    leaf shape.  Semantically equivalent to ``jax.lax.cond(mask, lambda _:
    s_then, lambda _: s_else, None)`` but always-compute (Brax pattern).
    """
    return jax.tree.map(lambda a, b: jnp.where(mask, a, b), s_then, s_else)


def pet_move_brax(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Thin dispatcher mirroring ``pet_move`` (monster_ai.py:4005)."""
    return _pet_move_brax_jit(
        state, rng, monster_idx, bool(_use_vendor_rng())
    )


@functools.partial(jax.jit, static_argnames=("vendor_mode",))
def _pet_move_brax_jit(state, rng: jax.Array, monster_idx: jnp.ndarray,
                       vendor_mode: bool):
    return _pet_move_brax_body(state, rng, monster_idx, vendor_mode)


def _pet_move_brax_body(state, rng: jax.Array, monster_idx: jnp.ndarray,
                        vendor_mode: bool):
    """Brax-flattened body — see module docstring for diff vs ``_pet_move_body``."""
    _CAT_FOOD_LOCAL: int = 7

    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    is_pet = mai.tame[idx] & mai.alive[idx]

    # -----------------------------------------------------------------------
    # 0a. Hunger tick (verbatim from _pet_move_body — already jnp.where-style).
    # -----------------------------------------------------------------------
    timestep_i32 = state.timestep.astype(jnp.int32)
    hungrytime = mai.hungrytime[idx].astype(jnp.int32)
    elapsed = jnp.maximum(timestep_i32 - hungrytime, jnp.int32(0))

    is_weak    = is_pet & (elapsed > jnp.int32(_DOG_WEAK))
    is_starved = is_pet & (elapsed > jnp.int32(_DOG_STARVE))
    cur_hp_starve = mai.hp[idx].astype(jnp.int32)
    mhp_starve_revert = (
        is_pet & (elapsed > jnp.int32(500)) & (cur_hp_starve < jnp.int32(3))
    )

    cur_mhpmax = mai.hp_max[idx].astype(jnp.int32)
    cur_penalty = mai.mhpmax_penalty[idx].astype(jnp.int32)
    already_penalised = cur_penalty > jnp.int32(0)
    do_weak_apply = is_weak & ~already_penalised
    new_mhpmax_val = jnp.maximum(cur_mhpmax // jnp.int32(3), jnp.int32(1))
    new_penalty_val = cur_mhpmax - new_mhpmax_val
    new_mhpmax = jnp.where(do_weak_apply, new_mhpmax_val, cur_mhpmax)
    new_penalty = jnp.where(do_weak_apply, new_penalty_val, cur_penalty)
    cur_hp = mai.hp[idx].astype(jnp.int32)
    new_hp_capped = jnp.minimum(cur_hp, new_mhpmax)
    final_alive = jnp.where(is_starved, jnp.bool_(False), mai.alive[idx])

    cur_hunger = mai.pet_hunger[idx].astype(jnp.int32)
    new_hunger_val = cur_hunger - jnp.int32(1)
    new_hunger = jnp.where(is_pet, new_hunger_val, cur_hunger).astype(jnp.int16)
    legacy_starved = is_pet & (new_hunger_val <= jnp.int32(-50))
    final_alive = jnp.where(legacy_starved, jnp.bool_(False), final_alive)

    new_tame_starve     = jnp.where(mhp_starve_revert, jnp.bool_(False), mai.tame[idx])
    new_peaceful_starve = jnp.where(mhp_starve_revert, jnp.bool_(False), mai.peaceful[idx])

    mai_h = mai.replace(
        pet_hunger=mai.pet_hunger.at[idx].set(new_hunger),
        hp_max=mai.hp_max.at[idx].set(new_mhpmax),
        mhpmax_penalty=mai.mhpmax_penalty.at[idx].set(new_penalty),
        hp=mai.hp.at[idx].set(new_hp_capped),
        alive=mai.alive.at[idx].set(final_alive),
        confuse_timer=mai.confuse_timer.at[idx].set(
            jnp.where(do_weak_apply, jnp.int16(1), mai.confuse_timer[idx])
        ),
        tame=mai.tame.at[idx].set(new_tame_starve),
        peaceful=mai.peaceful.at[idx].set(new_peaceful_starve),
    )
    state = state.replace(monster_ai=mai_h)

    mai = state.monster_ai
    is_pet = mai.tame[idx] & mai.alive[idx]
    mpos = mai.pos[idx].astype(jnp.int32)

    # -----------------------------------------------------------------------
    # 0b. Eat floor food (verbatim — already jnp.where-style).
    # -----------------------------------------------------------------------
    b = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    pr = jnp.clip(mpos[0], 0, _MAP_H - 1)
    pc = jnp.clip(mpos[1], 0, _MAP_W - 1)
    food_cat = state.ground_items.category[b, lv, pr, pc, 0].astype(jnp.int32)
    has_food = food_cat == jnp.int32(_CAT_FOOD_LOCAL)
    is_hungry_legacy = mai.pet_hunger[idx].astype(jnp.int32) <= jnp.int32(0)
    can_eat = is_pet & is_hungry_legacy & has_food

    food_tid = state.ground_items.type_id[b, lv, pr, pc, 0].astype(jnp.int32)
    safe_tid = jnp.clip(food_tid, 0, _OBJ_NUTRITION_TABLE.shape[0] - 1)
    base_nutrit = _OBJ_NUTRITION_TABLE[safe_tid]
    pet_entry = jnp.clip(mai.entry_idx[idx].astype(jnp.int32),
                         0, _MONSTER_SIZE_TABLE.shape[0] - 1)
    pet_msize = jnp.clip(_MONSTER_SIZE_TABLE[pet_entry].astype(jnp.int32),
                         0, _DOG_SIZE_NUTRIT_MULT_TABLE.shape[0] - 1)
    size_mult = _DOG_SIZE_NUTRIT_MULT_TABLE[pet_msize]
    nutrit = jnp.maximum(base_nutrit * size_mult, jnp.int32(1))

    cur_hungrytime = mai.hungrytime[idx].astype(jnp.int32)
    moves_now = state.timestep.astype(jnp.int32)
    base_hungrytime = jnp.maximum(cur_hungrytime, moves_now)
    new_hungrytime = jnp.where(
        can_eat, base_hungrytime + nutrit, cur_hungrytime
    )

    cur_penalty_e = mai.mhpmax_penalty[idx].astype(jnp.int32)
    cur_mhpmax_e  = mai.hp_max[idx].astype(jnp.int32)
    restored_mhpmax = cur_mhpmax_e + cur_penalty_e
    new_mhpmax_eat = jnp.where(can_eat, restored_mhpmax, cur_mhpmax_e)
    new_penalty_eat = jnp.where(can_eat, jnp.int32(0), cur_penalty_e)

    cur_mtame = mai.mtame[idx].astype(jnp.int32)
    bumped_mtame = jnp.minimum(cur_mtame + jnp.int32(1), jnp.int32(20))
    new_mtame = jnp.where(can_eat, bumped_mtame, cur_mtame).astype(jnp.int8)

    new_confuse_eat = jnp.where(can_eat, jnp.int16(0), mai.confuse_timer[idx])

    new_ground_cat = state.ground_items.category.at[b, lv, pr, pc, 0].set(
        jnp.where(can_eat, jnp.int8(0), state.ground_items.category[b, lv, pr, pc, 0])
    )
    new_hunger_after_eat = jnp.where(can_eat, jnp.int16(1000), mai.pet_hunger[idx])

    has_drop = mai.last_drop_turn[idx] > jnp.int32(0)
    drop_r = mai.last_drop_pos[idx, 0].astype(jnp.int32)
    drop_c = mai.last_drop_pos[idx, 1].astype(jnp.int32)
    ppos_e = state.player_pos.astype(jnp.int32)
    e_dr = jnp.abs(drop_r - ppos_e[0])
    e_dc = jnp.abs(drop_c - ppos_e[1])
    dropdist_e = jnp.maximum(e_dr, e_dc)
    elapsed_e = jnp.maximum(
        state.timestep.astype(jnp.int32)
        - mai.last_drop_turn[idx].astype(jnp.int32),
        jnp.int32(0),
    )
    denom_e = jnp.maximum(dropdist_e + elapsed_e, jnp.int32(1))
    apport_credit = jnp.int32(200) // denom_e
    do_apport_credit = can_eat & has_drop
    cur_apport_e = mai.apport[idx].astype(jnp.int32)
    bumped_apport = jnp.clip(cur_apport_e + apport_credit, jnp.int32(1), jnp.int32(127))
    new_apport_e = jnp.where(do_apport_credit, bumped_apport, cur_apport_e).astype(jnp.int8)

    mai_e = mai.replace(
        hungrytime=mai.hungrytime.at[idx].set(new_hungrytime),
        hp_max=mai.hp_max.at[idx].set(new_mhpmax_eat),
        mhpmax_penalty=mai.mhpmax_penalty.at[idx].set(new_penalty_eat),
        mtame=mai.mtame.at[idx].set(new_mtame),
        confuse_timer=mai.confuse_timer.at[idx].set(new_confuse_eat),
        pet_hunger=mai.pet_hunger.at[idx].set(new_hunger_after_eat),
        apport=mai.apport.at[idx].set(new_apport_e),
    )
    new_ground = state.ground_items.replace(category=new_ground_cat)
    state = state.replace(monster_ai=mai_e, ground_items=new_ground)
    mai = state.monster_ai

    # -----------------------------------------------------------------------
    # 0b'. Pick up item at current tile (verbatim).
    # -----------------------------------------------------------------------
    mpos2 = mai.pos[idx].astype(jnp.int32)
    pr2 = jnp.clip(mpos2[0], 0, _MAP_H - 1)
    pc2 = jnp.clip(mpos2[1], 0, _MAP_W - 1)
    g_cat = state.ground_items.category[b, lv, pr2, pc2, 0].astype(jnp.int32)
    g_buc = state.ground_items.buc_status[b, lv, pr2, pc2, 0].astype(jnp.int32)
    has_item_here = g_cat != jnp.int32(0)
    not_food = g_cat != jnp.int32(_CAT_FOOD_LOCAL)
    not_cursed = g_buc >= jnp.int32(0)
    empty_mask = mai.inv_category[idx] == jnp.int8(0)
    has_empty = jnp.any(empty_mask)
    pick_slot = jnp.argmax(empty_mask.astype(jnp.int32)).astype(jnp.int32)
    can_pickup = is_pet & has_item_here & not_food & not_cursed & has_empty

    g_type = state.ground_items.type_id[b, lv, pr2, pc2, 0]
    g_qty  = state.ground_items.quantity[b, lv, pr2, pc2, 0]
    g_chg  = state.ground_items.charges[b, lv, pr2, pc2, 0]

    new_inv_cat = mai.inv_category.at[idx, pick_slot].set(
        jnp.where(can_pickup, g_cat.astype(jnp.int8),
                  mai.inv_category[idx, pick_slot])
    )
    new_inv_type = mai.inv_type_id.at[idx, pick_slot].set(
        jnp.where(can_pickup, g_type, mai.inv_type_id[idx, pick_slot])
    )
    new_inv_qty = mai.inv_quantity.at[idx, pick_slot].set(
        jnp.where(can_pickup, g_qty, mai.inv_quantity[idx, pick_slot])
    )
    new_inv_buc = mai.inv_buc.at[idx, pick_slot].set(
        jnp.where(can_pickup, g_buc.astype(jnp.int8),
                  mai.inv_buc[idx, pick_slot])
    )
    new_inv_chg = mai.inv_charges.at[idx, pick_slot].set(
        jnp.where(can_pickup, g_chg, mai.inv_charges[idx, pick_slot])
    )
    new_ground_cat2 = state.ground_items.category.at[b, lv, pr2, pc2, 0].set(
        jnp.where(can_pickup, jnp.int8(0),
                  state.ground_items.category[b, lv, pr2, pc2, 0])
    )
    mai_p = mai.replace(
        inv_category=new_inv_cat,
        inv_type_id=new_inv_type,
        inv_quantity=new_inv_qty,
        inv_buc=new_inv_buc,
        inv_charges=new_inv_chg,
    )
    new_ground2 = state.ground_items.replace(category=new_ground_cat2)
    state = state.replace(monster_ai=mai_p, ground_items=new_ground2)
    mai = state.monster_ai

    # -----------------------------------------------------------------------
    # 0c. Flee on low HP.
    # FLATTENED (was lax.cond at monster_ai.py:4349):
    #     state = lax.cond(should_flee_low_hp, _flee_move, lambda s: s, state)
    # Replaced with a single jnp.where on the pos field.
    # -----------------------------------------------------------------------
    pet_hp = mai.hp[idx].astype(jnp.int32)
    pet_hp_max = jnp.maximum(mai.hp_max[idx].astype(jnp.int32), jnp.int32(1))
    low_hp = pet_hp * jnp.int32(4) < pet_hp_max
    entry = mai.entry_idx[idx]
    fearless = _has_flag2(entry, _M2_UNDEAD) | _has_flag2(entry, _M2_DEMON)
    should_flee_low_hp = is_pet & low_hp & ~fearless

    ppos = state.player_pos.astype(jnp.int32)
    flee_delta = jnp.clip(mpos - ppos, -1, 1).astype(jnp.int32)
    flee_delta = jnp.where(
        jnp.all(flee_delta == 0),
        jnp.array([1, 0], dtype=jnp.int32),
        flee_delta,
    )
    flee_r = jnp.clip(mpos[0] + flee_delta[0], 0, _MAP_H - 1).astype(jnp.int16)
    flee_c = jnp.clip(mpos[1] + flee_delta[1], 0, _MAP_W - 1).astype(jnp.int16)
    flee_pos = jnp.stack([flee_r, flee_c])

    cur_pos_row = mai.pos[idx]
    masked_flee_pos = jnp.where(should_flee_low_hp, flee_pos, cur_pos_row)
    mai_flee = mai.replace(pos=mai.pos.at[idx].set(masked_flee_pos))
    state = state.replace(monster_ai=mai_flee)
    mai = state.monster_ai

    # Re-derive after flee.
    is_pet = mai.tame[idx] & mai.alive[idx]
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)

    # -----------------------------------------------------------------------
    # Compute attack target (verbatim — pure jnp ops).
    # -----------------------------------------------------------------------
    other_pos = mai.pos.astype(jnp.int32)
    dr = jnp.abs(other_pos[:, 0] - mpos[0])
    dc = jnp.abs(other_pos[:, 1] - mpos[1])
    cheb = jnp.maximum(dr, dc)
    self_mask = jnp.arange(MAX_MONSTERS_PER_LEVEL, dtype=jnp.int32) == idx

    m_lev_self = jnp.clip(
        _MONSTER_LEVEL_TABLE[
            jnp.clip(entry.astype(jnp.int32), 0, _MONSTER_LEVEL_TABLE.shape[0] - 1)
        ].astype(jnp.int32),
        1, 30,
    )
    safe_max = jnp.maximum(mai.hp_max[idx].astype(jnp.int32), jnp.int32(1))
    balk = m_lev_self + (jnp.int32(5) * mai.hp[idx].astype(jnp.int32)) // safe_max - jnp.int32(2)

    all_lev = jnp.clip(
        _MONSTER_LEVEL_TABLE[
            jnp.clip(mai.entry_idx.astype(jnp.int32), 0,
                     _MONSTER_LEVEL_TABLE.shape[0] - 1)
        ].astype(jnp.int32),
        1, 30,
    )
    is_low_hp_self = (mai.hp[idx].astype(jnp.int32) * jnp.int32(4)
                       < safe_max)
    not_balked = all_lev < balk
    not_peaceful_when_lowhp = ~(is_low_hp_self & mai.peaceful)

    hostile = (mai.alive & ~mai.tame & ~mai.peaceful & ~self_mask
               & (cheb <= jnp.int32(1)) & not_balked
               & not_peaceful_when_lowhp)
    has_target = jnp.any(hostile)
    target_idx = jnp.argmax(hostile.astype(jnp.int32)).astype(jnp.int32)

    rng_attack_local, _rng_after_attack = jax.random.split(rng)

    # -----------------------------------------------------------------------
    # Attack / follow / explore.
    # FLATTENED (was lax.cond at monster_ai.py:4477, 4474):
    #     _pet_act = lax.cond(has_target, _attack_hostile, _move_no_target, s)
    #     _move_no_target = lax.cond(within_follow_range, _follow_player,
    #                                _explore, s)
    # Replaced with always-compute + tree-mask, EXCEPT vendor-mode follow
    # which still uses lax.cond (vendor_pet_dog_move mutates vendor_rng;
    # eager always-call would corrupt the byte-parity stream).
    # -----------------------------------------------------------------------
    dist_to_player = _chebyshev_dist(mpos, ppos)
    within_follow_range = dist_to_player < jnp.int32(6)

    # --- Branch A: attack hostile.  mattackm draws only from rng_attack_local
    # (a local split), so calling it speculatively is byte-safe.  When there
    # is no real target we pass defender_idx=idx (self) — mattackm's
    # same_slot gate makes that a no-op.
    safe_target_idx = jnp.where(has_target, target_idx, idx)
    s_attack = mattackm(state, idx, safe_target_idx, rng_attack_local)
    _m_attack = s_attack.monster_ai
    killed = ~_m_attack.alive[safe_target_idx]
    target_lev = jnp.clip(
        _MONSTER_LEVEL_TABLE[
            jnp.clip(_m_attack.entry_idx[safe_target_idx].astype(jnp.int32), 0,
                     _MONSTER_LEVEL_TABLE.shape[0] - 1)
        ].astype(jnp.int32), 1, 30,
    )
    new_xp = _m_attack.mon_xp.at[idx].set(
        jnp.where(killed,
                  _m_attack.mon_xp[idx] + target_lev,
                  _m_attack.mon_xp[idx])
    )
    s_attack = s_attack.replace(monster_ai=_m_attack.replace(mon_xp=new_xp))

    # --- Branch B: explore (random 8-dir step).  Pure JAX RNG; byte-safe to
    # always compute.
    rng_dir, _ = jax.random.split(rng)
    dir_idx = jax.random.randint(rng_dir, (), 0, 8)
    dy = jnp.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=jnp.int32)[dir_idx]
    dx = jnp.array([-1,  0,  1,-1, 1,-1, 0, 1], dtype=jnp.int32)[dir_idx]
    cur = mai.pos[idx].astype(jnp.int32)
    new_r_explore = jnp.clip(cur[0] + dy, 0, _MAP_H - 1).astype(jnp.int16)
    new_c_explore = jnp.clip(cur[1] + dx, 0, _MAP_W - 1).astype(jnp.int16)
    new_pos_explore = jnp.array([new_r_explore, new_c_explore], dtype=jnp.int16)
    mai_explore = mai.replace(pos=mai.pos.at[idx].set(new_pos_explore))
    s_explore = state.replace(monster_ai=mai_explore)

    # --- Branch C: follow.  In NON-vendor mode this is pure (pathfind_step +
    # apply_confusion_to_step + pos update) — byte-safe to always compute.
    # In vendor mode it calls vendor_pet_dog_move which mutates vendor_rng;
    # we must NOT compute it speculatively, so we wrap a lax.cond on the
    # combined "should we run vendor follow?" mask.
    if vendor_mode:
        from Nethax.nethax.subsystems.pet_dog_move import vendor_pet_dog_move

        # The vendor follow branch only runs when: pet is alive & tame
        # (outer is_pet gate), no target (has_target is False), and within
        # follow range.  The original code reached this branch only via
        # the lax.cond cascade is_pet → ~has_target → within_follow_range.
        run_follow_mask = is_pet & ~has_target & within_follow_range

        def _do_vendor_follow(s):
            new_s, new_vrng = vendor_pet_dog_move(s, s.vendor_rng, idx)
            return new_s.replace(vendor_rng=new_vrng)

        s_follow = jax.lax.cond(
            run_follow_mask, _do_vendor_follow, lambda s: s, state,
        )
    else:
        # Non-vendor follow: pathfind_step + confusion + pos update.  Pure
        # JAX RNG (rng / fold_in), no stateful side effects → always-compute.
        step_delta = pathfind_step(state, idx)
        _rng_conf_pet = jax.random.fold_in(rng, jnp.int32(0x636F6E66))  # "conf"
        is_confused_pet = mai.confuse_timer[idx] > jnp.int16(0)
        step_delta = apply_confusion_to_step(
            step_delta, is_confused_pet, _rng_conf_pet,
        )
        cur_follow = mai.pos[idx].astype(jnp.int32)
        new_r_follow = jnp.clip(cur_follow[0] + step_delta[0],
                                0, _MAP_H - 1).astype(jnp.int16)
        new_c_follow = jnp.clip(cur_follow[1] + step_delta[1],
                                0, _MAP_W - 1).astype(jnp.int16)
        new_pos_follow = jnp.stack([new_r_follow, new_c_follow])
        mai_follow = mai.replace(pos=mai.pos.at[idx].set(new_pos_follow))
        s_follow = state.replace(monster_ai=mai_follow)

    # --- Compose: between follow and explore based on within_follow_range,
    # then between attack and that result based on has_target, then between
    # acted-state and untouched-state based on is_pet.
    s_move_no_target = _select_state(within_follow_range, s_follow, s_explore)
    s_pet_act = _select_state(has_target, s_attack, s_move_no_target)
    out = _select_state(is_pet, s_pet_act, state)

    return out
