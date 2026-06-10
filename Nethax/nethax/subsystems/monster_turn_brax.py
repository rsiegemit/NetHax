"""Brax-style rewrites of monster_ai.monster_turn and its small helpers.

Motivation
----------
Under ``jax.vmap``, ``jax.lax.cond`` lowers to ``jax.lax.select`` so BOTH
branches are emitted into HLO.  Deeply nested cond chains (as in
``monster_turn``) therefore blow up compile time.  Brax / Craftax style
sidesteps this by always evaluating both branches at the Python level and
selecting the result with ``jnp.where`` masking — yielding flat HLO that
compiles in O(branches) rather than O(2**depth).

These rewrites are NOT JIT-compiled here; integration code wraps the
``monster_turn_brax`` driver with ``jax.jit`` and ``jax.vmap`` once the
helper rewrites land.  For now the body calls the ORIGINAL non-Brax
helpers (``pet_move``, ``monster_use_item``, ``monster_cast_spell``,
``monster_attack_player``, ``_postmov_per_monster``, ``pathfind_step``,
``maybe_wake_monster``, ``maybe_retreat``, ``_m_search_items``).  When the
companion Brax rewrites land we will swap each call site to the
``*_brax`` variant in a single integration commit.

Byte-parity invariants preserved
--------------------------------
1. RNG split structure identical to the original ``monster_turn``:
       (rng_pet, rng_cast, rng_atk, rng_pick,
        rng_decay, rng_wake, rng_conf_step) = jax.random.split(rng, 7)
       rng_mconf, rng_mstun = jax.random.split(rng_decay)
2. Mutations are byte-identical via ``jnp.where`` masking on the array
   leaves of the monster_ai pytree.
3. State pytree shape / dtype preserved (no new fields, no resizes).

Brax-rewrite checklist for ``monster_turn``
-------------------------------------------
Original control-flow primitives → Brax replacement
* ``cond(is_pet, pet_branch, hostile_branch)``
      → run ``pet_move`` AND the hostile branch, then ``tree_map(where, ...)``
        with mask = ``is_pet``.
* ``cond(should_act, _act, identity)`` (hostile gate)
      → run ``_act`` body unconditionally, then select with ``should_act``.
* ``cond(cast_now, _maybe_cast, identity)``
      → run ``monster_cast_spell`` unconditionally, select with ``cast_now``.
* ``cond(steps_onto_player, _attack, _move)``
      → run BOTH attack and move state updates, select with mask.

These four flattened ``cond``/``switch`` constructs are the source of the
exponential HLO blow-up.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import os

from Nethax.nethax.subsystems import monster_ai as _ma
from Nethax.nethax.subsystems.monster_ai import (
    MoveStrategy,
    _IGNORES_ELBERETH,
    _MAP_H,
    _MAP_W,
    _M2_DEMON,
    _M2_UNDEAD,
    _SQSRCHRADIUS,
    _M_SEARCH_CAT_ROCK,
    _has_flag2,
    _is_mage_entry,
    _mon_wants_cat,
    _monster_level,
    apply_confusion_to_step,
)

# Choose Brax-style inner helpers when NETHAX_BRAX_ALL=1, else use originals.
# This lets the full chain run flat-HLO end-to-end.
if os.environ.get("NETHAX_BRAX_ALL", "0") == "1":
    from Nethax.nethax.subsystems.use_cast_brax import (
        monster_use_item_brax as monster_use_item,
        monster_cast_spell_brax as monster_cast_spell,
    )
    from Nethax.nethax.subsystems.pet_move_brax import pet_move_brax as pet_move
    from Nethax.nethax.subsystems.postmov_brax import (
        _postmov_per_monster_brax as _postmov_per_monster,
    )
    # Round 2: pathfind + monster_can_see_player.
    from Nethax.nethax.subsystems.pathfind_step_brax import pathfind_step_brax as pathfind_step
    from Nethax.nethax.subsystems.preloop_brax import (
        monster_can_see_player_brax as monster_can_see_player,
    )
else:
    from Nethax.nethax.subsystems.monster_ai import (
        monster_use_item,
        monster_cast_spell,
        pet_move,
        _postmov_per_monster,
        pathfind_step,
        monster_can_see_player,
    )


# ---------------------------------------------------------------------------
# Helper: tree-level select between two state pytrees.
# ---------------------------------------------------------------------------

def _state_select(mask: jnp.ndarray, on_true, on_false):
    """Leaf-wise ``jnp.where(mask, on_true, on_false)`` over matching pytrees.

    Both inputs must have the exact same pytree structure and matching leaf
    shapes / dtypes — the standard Brax-style assumption.  ``mask`` is a
    scalar bool that broadcasts against every leaf.
    """
    return jax.tree_util.tree_map(
        lambda t, f: jnp.where(mask, t, f), on_true, on_false,
    )


# ---------------------------------------------------------------------------
# 1. _m_search_items_brax
# ---------------------------------------------------------------------------
# Original has NO conds — the body is already pure mask arithmetic.  Brax
# version is byte-identical; included for API symmetry so callers can use
# ``*_brax`` uniformly.
# Conds flattened: 0.
# ---------------------------------------------------------------------------

def _m_search_items_brax(state, monster_idx: jnp.ndarray) -> tuple:
    """Brax-style mirror of ``_m_search_items`` (no cond changes needed)."""
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    mpos = mai.pos[idx].astype(jnp.int32)
    omx, omy = mpos[0], mpos[1]
    entry = mai.entry_idx[idx]

    b = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    cat_map = state.ground_items.category[b, lv, :, :, 0].astype(jnp.int32)

    rows = jnp.arange(_MAP_H, dtype=jnp.int32)[:, None]
    cols = jnp.arange(_MAP_W, dtype=jnp.int32)[None, :]
    drow = jnp.abs(rows - omx)
    dcol = jnp.abs(cols - omy)
    distmin = jnp.maximum(drow, dcol)
    in_box = (distmin <= jnp.int32(_SQSRCHRADIUS)) & (distmin > jnp.int32(0))

    wanted_cat = _mon_wants_cat(entry, cat_map.flatten()).reshape(cat_map.shape)
    is_rock = cat_map == jnp.int32(_M_SEARCH_CAT_ROCK)
    candidate = in_box & wanted_cat & ~is_rock

    NEG_INF = jnp.int32(-1_000_000)
    score = jnp.where(candidate, -distmin, NEG_INF)

    flat = score.flatten()
    bestf = jnp.argmax(flat)
    found = jnp.any(candidate)
    trow = (bestf // jnp.int32(_MAP_W)).astype(jnp.int32)
    tcol = (bestf % jnp.int32(_MAP_W)).astype(jnp.int32)
    trow_safe = jnp.where(found, trow, jnp.int32(0))
    tcol_safe = jnp.where(found, tcol, jnp.int32(0))
    return found, trow_safe, tcol_safe


# ---------------------------------------------------------------------------
# 2. maybe_retreat_brax
# ---------------------------------------------------------------------------
# Original has no cond — already pure ``jnp.where`` arithmetic.  Brax
# variant included for symmetry.
# Conds flattened: 0.
# ---------------------------------------------------------------------------

def maybe_retreat_brax(state, monster_idx: jnp.ndarray) -> jnp.ndarray:
    """Brax-style mirror of ``maybe_retreat`` (no cond changes needed)."""
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    hp = mai.hp[idx].astype(jnp.int32)
    # hp_max bound retained from original — unused here but preserves shape.
    _ = jnp.maximum(mai.hp_max[idx].astype(jnp.int32), jnp.int32(1))

    entry = mai.entry_idx[idx]
    mlev = _monster_level(entry)
    quarter = jnp.maximum(mlev // jnp.int32(4), jnp.int32(5))
    threshold = jnp.where(mlev >= jnp.int32(2), quarter, jnp.int32(1))
    low_hp = hp <= threshold

    is_demon = _has_flag2(entry, _M2_DEMON)
    is_undead = _has_flag2(entry, _M2_UNDEAD)
    fearless = is_demon | is_undead

    peaceful = mai.peaceful[idx]
    should_flee = low_hp & ~fearless & ~peaceful

    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    delta = jnp.clip(mpos - ppos, -1, 1).astype(jnp.int32)
    zero = jnp.zeros((2,), dtype=jnp.int32)
    return jnp.where(should_flee, delta, zero)


# ---------------------------------------------------------------------------
# 3. maybe_wake_monster_brax
# ---------------------------------------------------------------------------
# Original has a Python-level ``if rng is None`` (static, not traced) and
# no jax conds.  Brax variant: keep the Python static branch (it is static
# at trace time, not vmapped), use jnp.where for the dynamic gate.
# Conds flattened: 0 (already pure ``jnp.where``).
# ---------------------------------------------------------------------------

def maybe_wake_monster_brax(
    state,
    monster_idx: jnp.ndarray,
    rng: jax.Array = None,
):
    """Brax-style mirror of ``maybe_wake_monster`` (no cond changes needed)."""
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    asleep = mai.asleep[idx]
    alive = mai.alive[idx]
    in_los = monster_can_see_player(state, idx)

    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    dr = mpos[0] - ppos[0]
    dc = mpos[1] - ppos[1]
    distu_sq = dr * dr + dc * dc
    within_100 = distu_sq <= jnp.int32(100)

    # ``rng is None`` is a Python-static branch, not vmapped — leaving it
    # as a Python ``if`` is Brax-compatible (no HLO emission either way).
    if rng is None:
        rn2_7_pass = jnp.bool_(True)
    else:
        rng_key, _ = jax.random.split(rng)
        rn2_7_pass = jax.random.randint(rng_key, (), 0, 7) == 0

    should_wake = asleep & alive & in_los & within_100 & rn2_7_pass

    new_asleep = jnp.where(should_wake, jnp.bool_(False), mai.asleep[idx])
    new_mai = mai.replace(asleep=mai.asleep.at[idx].set(new_asleep))
    return state.replace(monster_ai=new_mai)


# ---------------------------------------------------------------------------
# 4. monster_turn_brax — the main driver.
# ---------------------------------------------------------------------------
# Brax flattening map (4 conds → 0 conds):
#   (A) cond(is_pet, _pet_branch, _hostile_branch)
#         → compute pet-branch state + hostile-branch state, then
#           _state_select(is_pet, pet_state, hostile_state).
#   (B) cond(should_act, _act, identity)  in hostile branch
#         → run the act body, then _state_select(should_act, act_state,
#           pre_act_state).
#   (C) cond(cast_now, _maybe_cast, identity)
#         → run monster_cast_spell unconditionally, then
#           _state_select(cast_now, cast_state, pre_cast_state).
#   (D) cond(steps_onto_player, _attack, _move)
#         → compute attack state and move state, then
#           _state_select(steps_onto_player, attack_state, move_state).
#
# RNG ordering preserved exactly:
#   jax.random.split(rng, 7) is unchanged, and the rng_decay sub-split is
#   identical.  rng_pet, rng_cast, rng_atk, rng_pick, rng_wake, rng_conf_step
#   are each consumed at most once on their respective code paths just like
#   the original; under Brax-style they are *also* consumed when their gate
#   is False, but because the original always splits these keys up-front
#   the consumption is purely cosmetic (no key is reused).
# ---------------------------------------------------------------------------

def monster_turn_brax(state, rng: jax.Array, monster_idx: jnp.ndarray) -> object:
    """Brax-style rewrite of ``monster_turn``.

    See module docstring for the flattening map.  Byte-parity guarantees:
      * RNG split layout identical to the original.
      * Final state is selected leaf-wise from one of the original branches'
        outputs, never a synthesised hybrid.
    """
    # Import here to mirror the original's late import of combat (avoids a
    # subsystems import cycle if the module is loaded in isolation).
    if os.environ.get("NETHAX_BRAX_ALL", "0") == "1":
        from Nethax.nethax.subsystems.monster_attack_player_brax import (
            monster_attack_player_brax as monster_attack_player,
        )
    else:
        from Nethax.nethax.subsystems.combat import monster_attack_player

    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    # --- RNG split (preserved verbatim) ---
    (rng_pet, rng_cast, rng_atk, rng_pick,
     rng_decay, rng_wake, rng_conf_step) = jax.random.split(rng, 7)
    rng_mconf, rng_mstun = jax.random.split(rng_decay)

    # ----- Branch (A) selector -----
    is_pet = mai.tame[idx] & mai.alive[idx]

    # =====================================================================
    # PET BRANCH — always computed (Brax style).
    # =====================================================================
    pet_state = pet_move(state, rng_pet, idx)

    # =====================================================================
    # HOSTILE BRANCH — always computed.
    # =====================================================================
    # --- monmove.c:717 mcanmove gate ---
    _m_pre = state.monster_ai
    is_paralyzed = _m_pre.paralyzed_timer[idx] > jnp.int16(0)
    is_waiting = _m_pre.mstrategy[idx] == jnp.int8(MoveStrategy.WAIT)
    cannot_move = is_paralyzed | is_waiting

    # --- monmove.c:737-742 stochastic confusion / stun decay ---
    rn50 = jax.random.randint(rng_mconf, (), 0, 50)
    rn10 = jax.random.randint(rng_mstun, (), 0, 10)
    decay_conf = (_m_pre.confuse_timer[idx] > 0) & (rn50 == 0)
    decay_stun = (_m_pre.stun_timer[idx] > 0) & (rn10 == 0)
    new_conf_v = jnp.where(decay_conf, jnp.int16(0), _m_pre.confuse_timer[idx])
    new_stun_v = jnp.where(decay_stun, jnp.int16(0), _m_pre.stun_timer[idx])

    _m_decay = _m_pre.replace(
        confuse_timer=_m_pre.confuse_timer.at[idx].set(new_conf_v),
        stun_timer=_m_pre.stun_timer.at[idx].set(new_stun_v),
    )
    s_after_decay = state.replace(monster_ai=_m_decay)

    # Record asleep BEFORE the wake check — wake-this-turn monsters don't
    # also act this turn (vendor disturb semantics).
    was_asleep = s_after_decay.monster_ai.asleep[idx]

    # 3: wake check.
    s_after_wake = _ma.maybe_wake_monster(s_after_decay, idx, rng_wake)

    # 4: gate on alive & not asleep (at start of turn) & not peaceful & mcanmove.
    m_after_wake = s_after_wake.monster_ai
    should_act = (
        m_after_wake.alive[idx]
        & ~was_asleep
        & ~m_after_wake.peaceful[idx]
        & ~cannot_move
    )

    # -------------------------------------------------------------------
    # _act body (always executed; selected by should_act later).
    # -------------------------------------------------------------------
    # 5: muse (call site preserved; helper is mostly stubs).
    s_after_muse = monster_use_item(s_after_wake, rng_cast, idx)

    # 6: optional spell cast for mage-class monsters.
    is_mage = _is_mage_entry(s_after_muse.monster_ai.entry_idx[idx])
    roll = jax.random.uniform(rng_pick, ())  # 0..1 — same key as original.
    cast_now = is_mage & (roll < 0.5)

    # Brax flatten (C): always run the cast, select.
    s_after_cast_run = monster_cast_spell(s_after_muse, rng_cast, idx)
    s_after_cast = _state_select(cast_now, s_after_cast_run, s_after_muse)

    # 7a: Elbereth fear check (onscary).
    from Nethax.nethax.subsystems.engrave import is_elbereth_at
    from Nethax.nethax.dungeon.branches import Branch as _Branch
    _ppos = s_after_cast.player_pos.astype(jnp.int32)
    _scared_raw = is_elbereth_at(s_after_cast.engrave, _ppos[0], _ppos[1])
    _eidx = s_after_cast.monster_ai.entry_idx[idx].astype(jnp.int32)
    _safe_e = jnp.clip(_eidx, 0, _IGNORES_ELBERETH.shape[0] - 1)
    _ignores = _IGNORES_ELBERETH[_safe_e]
    _in_gehennom = (
        s_after_cast.dungeon.current_branch.astype(jnp.int32)
        == jnp.int32(_Branch.GEHENNOM)
    )
    scared = _scared_raw & ~_ignores & ~_in_gehennom

    # 7b: movement decision.
    retreat_step = _ma.maybe_retreat(s_after_cast, idx)
    wants_retreat = jnp.any(retreat_step != 0)
    path_step = pathfind_step(s_after_cast, idx)

    # 7b': m_search_items override.
    _ms_found, _ms_r, _ms_c = _ma._m_search_items(s_after_cast, idx)
    _mpos_i32 = s_after_cast.monster_ai.pos[idx].astype(jnp.int32)
    _item_target = jnp.stack([_ms_r, _ms_c]).astype(jnp.int32)
    _item_step = jnp.clip(_item_target - _mpos_i32, -1, 1).astype(jnp.int32)
    path_step = jnp.where(_ms_found, _item_step, path_step)
    step_delta = jnp.where(wants_retreat, retreat_step, path_step)

    # Confusion override AFTER retreat (vendor parity).
    is_confused_mi = s_after_cast.monster_ai.confuse_timer[idx] > jnp.int16(0)
    step_delta = apply_confusion_to_step(
        step_delta, is_confused_mi, rng_conf_step,
    )
    step_delta = jnp.where(
        scared, jnp.zeros(2, dtype=jnp.int32), step_delta
    )

    cur_pos = s_after_cast.monster_ai.pos[idx].astype(jnp.int32)
    new_pos_i32 = cur_pos + step_delta
    ppos_i32 = s_after_cast.player_pos.astype(jnp.int32)
    steps_onto_player = jnp.all(new_pos_i32 == ppos_i32)

    # -------------------------------------------------------------------
    # Brax flatten (D): always compute BOTH attack and move states.
    # -------------------------------------------------------------------
    # Attack branch.
    s_attack_raw, _atk_dmg = monster_attack_player(s_after_cast, rng_atk, idx)
    _m_atk = s_attack_raw.monster_ai
    _m_atk_new = _m_atk.replace(
        last_seen_player_pos=_m_atk.last_seen_player_pos.at[idx].set(
            s_after_cast.player_pos.astype(jnp.int16)
        ),
        mstrategy=_m_atk.mstrategy.at[idx].set(jnp.int8(MoveStrategy.HUNT)),
    )
    s_attack = s_attack_raw.replace(monster_ai=_m_atk_new)

    # Move branch.
    _m_mv = s_after_cast.monster_ai
    _target_raw = jnp.where(steps_onto_player, cur_pos, new_pos_i32)
    _target_r = jnp.clip(_target_raw[0], 0, _MAP_H - 1)
    _target_c = jnp.clip(_target_raw[1], 0, _MAP_W - 1)
    final_pos = jnp.stack([_target_r, _target_c]).astype(jnp.int16)
    new_strategy = jnp.where(
        wants_retreat,
        jnp.int8(MoveStrategy.FLEE),
        jnp.int8(MoveStrategy.HUNT),
    )
    ppos_i16 = s_after_cast.player_pos.astype(jnp.int16)
    _m_mv_new = _m_mv.replace(
        pos=_m_mv.pos.at[idx].set(final_pos),
        last_seen_player_pos=_m_mv.last_seen_player_pos.at[idx].set(ppos_i16),
        mstrategy=_m_mv.mstrategy.at[idx].set(new_strategy),
    )
    s_move = s_after_cast.replace(monster_ai=_m_mv_new)

    # Select attack vs. move.
    s_after_move_or_attack = _state_select(steps_onto_player, s_attack, s_move)

    # postmov per-tile refresh — runs in the act path.
    s_act_final = _postmov_per_monster(s_after_move_or_attack, idx)

    # -------------------------------------------------------------------
    # Brax flatten (B): select between act-final and pre-act for hostile.
    # -------------------------------------------------------------------
    hostile_state = _state_select(should_act, s_act_final, s_after_wake)

    # -------------------------------------------------------------------
    # Brax flatten (A): select pet vs. hostile.
    # -------------------------------------------------------------------
    return _state_select(is_pet, pet_state, hostile_state)
