"""Brax-style flattened rewrites of ``try_engulf``, ``polymorph_player``, and
``trigger_lycanthropy``.

Why this file exists
--------------------
``monster_attack_player`` already has a Brax-style port (see
``monster_attack_player_brax.py``).  Under ``jax.vmap`` over seeds, the
remaining ``jax.lax.cond`` calls *inside* the helpers it dispatches to
(``try_engulf``, ``polymorph_player``, and the lycanthropy path) still
lower to ``lax.select`` with both branches present in the HLO graph.
The deep, nested cond chains in ``polymorph_player`` (snapshot, dismount,
same-race newman re-roll) plus the cond inside ``try_engulf`` (already
swallowed) cause the same compile blow-up the Brax pattern was designed
to eliminate.

Following the same pattern as ``monster_attack_player_brax`` /
``use_cast_brax``: **always compute both branches** and select with
``jnp.where`` over the precomputed mask (via ``jax.tree.map`` for full
pytree analogues of ``lax.cond``).

Byte-parity contract
--------------------
1. RNG draw order preserved exactly (see notes per function).
2. Mutations byte-identical via ``jnp.where`` over pytrees with the same
   scalar mask as the original ``lax.cond`` predicate.
3. State pytree shape preserved — the leaves selected by ``jnp.where``
   have identical dtype/shape on both branches.

Conditionals flattened
----------------------
- ``try_engulf_brax``         : 1 ``lax.cond`` → 1 pytree-where.
- ``polymorph_player_brax``   : 3 ``lax.cond`` → 3 pytree-where (snapshot,
                                dismount, same-race newman).
- ``trigger_lycanthropy_brax``: 0 direct conds; routes through
                                ``polymorph_player_brax`` so the nested
                                conds inside are also flattened.

The signatures match the originals so the Brax versions are drop-in.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.rng import rn2

from Nethax.nethax.subsystems.swallow import (
    _SWALLOW_VARIANT,
)
from Nethax.nethax.subsystems import polymorph as _poly
from Nethax.nethax.subsystems.polymorph import (
    _FORM_CAN_RIDE,
    _FORM_FLAGS2,
    _FORM_IS_GIANT,
    _FORM_IS_UNDEAD,
    _FORM_STRONGMONST,
    _POLY_TIMER_BASE,
    _POLY_TIMER_RANGE,
    _LYCANTHROPY_FORM_DURATION,
    _drop_worn_armor_per_slot,
    _form_attacks,
    _form_hp_max,
    _form_intrinsics,
    _monster_tables,
    _recompute_ac,
    _retouch_equipment_silver,
    newman,
)


def _select_tree(cond, on_true, on_false):
    """Pytree analogue of ``jnp.where(cond, on_true, on_false)``.

    Both ``on_true`` and ``on_false`` must be pytrees with identical
    structure (same Flax dataclass type, same field dtypes).
    """
    return jax.tree.map(lambda a, b: jnp.where(cond, a, b), on_true, on_false)


# ---------------------------------------------------------------------------
# try_engulf
# ---------------------------------------------------------------------------
def try_engulf_brax(state, monster_idx, rng):
    """Brax-style flattened rewrite of ``swallow.try_engulf``.

    Matches the *default* (``attack_is_dgst is None``) calling convention
    used by ``monster_attack_player_brax`` — i.e. the vendor 3.6
    ``rn1(25, 75)`` formula clamped to >= 2.  RNG draw order: a single
    ``jax.random.randint`` on ``rng`` (identical to the original's
    ``attack_is_dgst is None`` branch).

    Byte-equal to ``swallow.try_engulf(state, monster_idx, rng)``; the
    outer ``lax.cond(already, identity, apply_engulf, state)`` is
    replaced with always-compute + pytree ``where``.
    """
    already = state.swallow.swallowed

    # Vendor 3.6 non-AD_DGST: rn1(25, 75) = rn2(25) + 75 ∈ [75, 99].
    # Clamp to >= 2 mirrors mhitu.c:1395.
    total = jnp.maximum(
        jax.random.randint(rng, (), 0, 25, dtype=jnp.int32) + jnp.int32(75),
        jnp.int32(2),
    )

    # Player's pos moves to the engulfer's pos (they are now inside).
    # vendor/nethack/src/mhitu.c:1301.
    slot_i32 = monster_idx.astype(jnp.int32)
    engulfer_pos = state.monster_ai.pos[slot_i32]

    new_swallow = state.swallow.replace(
        swallowed=jnp.bool_(True),
        engulfer_slot=slot_i32,
        digest_timer=jnp.int32(10),
        total_timer=total,
    )
    new_pos = engulfer_pos.astype(jnp.int16)

    # Choose swallow-message variant by engulfer's AT_ENGL damage type.
    # vendor/nethack/src/mhitu.c:1335-1338.
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    entry_idx = state.monster_ai.entry_idx[slot_i32].astype(jnp.int32)
    safe_entry = jnp.clip(entry_idx, jnp.int32(0), jnp.int32(_SWALLOW_VARIANT.shape[0] - 1))
    variant = _SWALLOW_VARIANT[safe_entry].astype(jnp.int32)
    msg_id_engulfs = jnp.int32(int(_MsgId.SWALLOW_ENGULFS))
    msg_id_digests = jnp.int32(int(_MsgId.SWALLOW_DIGESTS))
    msg_id_enfolds = jnp.int32(int(_MsgId.SWALLOW_ENFOLDS))
    chosen_id = jnp.where(
        variant == jnp.int32(1), msg_id_digests,
        jnp.where(variant == jnp.int32(2), msg_id_enfolds, msg_id_engulfs),
    )
    new_messages = _msg_emit(state.messages, chosen_id)

    swallowed_state = state.replace(
        swallow=new_swallow,
        player_pos=new_pos,
        messages=new_messages,
    )

    # Flatten: lax.cond(already, identity, apply_engulf, state)
    #   == when already True → keep state; else use swallowed_state.
    return _select_tree(already, state, swallowed_state)


# ---------------------------------------------------------------------------
# polymorph_player
# ---------------------------------------------------------------------------
def polymorph_player_brax(state, rng, target_form_idx, controlled=False):
    """Brax-style flattened rewrite of ``polymorph.polymorph_player``.

    Three ``lax.cond`` sites flattened:
      1. orig_* snapshot (gated on ``already_poly``).
      2. force-dismount on incompatible new form (gated on ``do_dismount``).
      3. ``newman`` re-roll on same-race poly (gated on ``same_race``).

    RNG draw order preserved exactly:
      a. ``jax.random.split(rng)`` → (rng, sub)  for ``_form_hp_max``.
      b. ``jax.random.split(rng)`` → (rng, sub2) for ``poly_timer``.
      c. ``jax.random.split(rng)`` → (rng, sub_fall) for dismount fall roll.
      d. ``jax.random.split(rng)`` → (rng, sub_rt) for ``_retouch_equipment_silver``.
      e. ``jax.random.split(rng)`` → (rng, sub_nm) for ``newman``.
    These are unconditional in the original (the lax.cond branches consume
    them inside their lambdas regardless of which side is selected under
    JIT/vmap), so the Brax form is byte-identical.
    """
    # Coerce inputs to JAX scalars (same as original).
    form_i16 = jnp.int16(int(target_form_idx)) if isinstance(target_form_idx, int) \
        else target_form_idx.astype(jnp.int16)
    controlled_b = jnp.bool_(bool(controlled)) if isinstance(controlled, bool) \
        else controlled.astype(jnp.bool_)

    poly = state.polymorph
    already_poly = poly.is_polymorphed

    # --- 1. Snapshot originals (flattened cond).
    # Original: poly = lax.cond(already_poly, identity, _snap, poly)
    # Brax    : compute _snap unconditionally, select via tree-where.
    types_pre, dtyps_pre, nd_pre, ns_pre = _form_attacks(form_i16)
    snapped = poly.replace(
        orig_role_idx=state.player_role.astype(jnp.int8),
        orig_str=state.player_str.astype(jnp.int16),
        orig_dex=state.player_dex.astype(jnp.int8),
        orig_con=state.player_con.astype(jnp.int8),
        orig_hp_max=state.player_hp_max.astype(jnp.int32),
        orig_ac=state.player_ac.astype(jnp.int32),
        orig_attack_types=poly.attack_types,
        orig_attack_damage_types=poly.attack_damage_types,
        orig_attack_n_dice=poly.attack_n_dice,
        orig_attack_n_sides=poly.attack_n_sides,
    )
    poly = _select_tree(already_poly, poly, snapped)

    # --- 2/3. Set new form data + adopt attacks/intrinsics.
    types, dtyps, nd, ns = _form_attacks(form_i16)
    intr = _form_intrinsics(form_i16)
    rng, sub = jax.random.split(rng)
    in_endgame = state.dungeon.current_branch.astype(jnp.int32) == jnp.int32(6)
    new_hp_max = _form_hp_max(form_i16, sub, in_endgame=in_endgame)

    # poly_timer = rn1(500, 500) → [500, 1000).
    rng, sub2 = jax.random.split(rng)
    timer = (jnp.int16(_POLY_TIMER_BASE)
             + jax.random.randint(sub2, (), 0, _POLY_TIMER_RANGE).astype(jnp.int16))

    # vendor polyself.c:874 — short timer for low-level → high-level polys.
    tables_local = _monster_tables()
    mlvl_form = tables_local["level"][form_i16.astype(jnp.int32)].astype(jnp.int32)
    ulevel = state.player_xl.astype(jnp.int32)
    safe_mlvl = jnp.maximum(mlvl_form, jnp.int32(1))
    scaled_timer = (timer.astype(jnp.int32) * ulevel // safe_mlvl).astype(jnp.int16)
    timer = jnp.where(ulevel < mlvl_form, scaled_timer, timer)

    new_count = jnp.where(controlled_b,
                          poly.controlled_poly_count + jnp.int8(1),
                          poly.controlled_poly_count)

    poly = poly.replace(
        is_polymorphed=jnp.bool_(True),
        current_form_idx=form_i16,
        poly_timer=timer,
        poly_controlled=controlled_b,
        controlled_poly_count=new_count,
        attack_types=types,
        attack_damage_types=dtyps,
        attack_n_dice=nd,
        attack_n_sides=ns,
        intrinsics_mask=intr,
        poly_form_id=form_i16.astype(jnp.int32),
        poly_turns=timer.astype(jnp.int32),
        poly_controlled_legacy=controlled_b,
    )

    # --- 4. Recompute AC. Apply HP_max swap. Clamp Pw.
    state = state.replace(
        polymorph=poly,
        player_hp_max=new_hp_max,
        player_hp=jnp.minimum(state.player_hp, new_hp_max),
        player_pw=jnp.minimum(state.player_pw, state.player_pw_max),
    )
    state = _recompute_ac(state, form_i16)

    # --- 4a. uasmon_maxStr (polyself.c:1077-1119 + 820-832).
    form_i32   = form_i16.astype(jnp.int32)
    is_strong  = _FORM_STRONGMONST[form_i32]
    is_giant_f = _FORM_IS_GIANT[form_i32]
    is_undead_f= _FORM_IS_UNDEAD[form_i32]
    live_H     = is_giant_f & (~is_undead_f)
    new_max_str = jnp.where(
        is_strong & live_H,
        jnp.int16(119),
        jnp.where(is_strong, jnp.int16(118),
                  jnp.int16(18)),
    )
    cur_str = state.player_str.astype(jnp.int16)
    updated_str = jnp.where(is_strong,
                            new_max_str,
                            jnp.minimum(cur_str, new_max_str))
    state = state.replace(player_str=updated_str.astype(state.player_str.dtype))

    # --- 4b. Mount-on-poly: force dismount if new form cannot ride.
    # Original: state = lax.cond(do_dismount, _dismount, identity, state)
    # Brax    : compute dismount state unconditionally, select via tree-where.
    rng, sub_fall = jax.random.split(rng)
    fall_roll = jax.random.randint(sub_fall, (), 1, 7).astype(jnp.int32)
    was_riding = state.player_steed_mid != jnp.uint32(0)
    new_form_can_ride = _FORM_CAN_RIDE[form_i16.astype(jnp.int32)]
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intr
    levitating = state.status.intrinsics[int(_Intr.LEVITATION)]
    do_dismount = was_riding & (~new_form_can_ride)

    applied = jnp.where(levitating, jnp.int32(0), fall_roll)
    dismount_hp = jnp.maximum(state.player_hp - applied, jnp.int32(0))
    dismount_state = state.replace(
        player_steed_mid=jnp.uint32(0),
        player_hp=dismount_hp,
        done=state.done | (dismount_hp <= jnp.int32(0)),
    )
    state = _select_tree(do_dismount, dismount_state, state)

    # --- 5. Drop incompatible armor per-slot.
    state = _drop_worn_armor_per_slot(state, form_i16)

    # --- 5b. retouch_equipment: silver items burn silver-allergic forms.
    rng, sub_rt = jax.random.split(rng)
    state = _retouch_equipment_silver(state, form_i16, sub_rt)

    # --- 5c. newman(): re-roll on same-race poly.
    # Original: state = lax.cond(same_race, newman, identity, state)
    # Brax    : run newman unconditionally, select via tree-where.
    form_flags2 = _FORM_FLAGS2[form_i16.astype(jnp.int32)]
    player_race = state.player_race.astype(jnp.int32)
    race_to_m2 = jnp.array(
        [0x8, 0x10, 0x20, 0x40, 0x80], dtype=jnp.uint32
    )
    safe_race = jnp.clip(player_race, 0, 4)
    player_race_m2 = race_to_m2[safe_race]
    same_race = (form_flags2 & player_race_m2) != jnp.uint32(0)

    rng, sub_nm = jax.random.split(rng)
    newman_state = newman(state, sub_nm)
    state = _select_tree(same_race, newman_state, state)

    # --- 7. Conduct: POLYSELFLESS violated.
    from Nethax.nethax.subsystems.conduct import Conduct, increment_counter
    state = increment_counter(state, int(Conduct.POLYSELFLESS))

    # Emit "You turn into ..." message.
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    state = state.replace(messages=_msg_emit(state.messages, int(_MsgId.YOU_TURN_INTO)))

    return state


# ---------------------------------------------------------------------------
# trigger_lycanthropy
# ---------------------------------------------------------------------------
def trigger_lycanthropy_brax(state, rng, were_form_idx):
    """Brax-style flattened rewrite of ``polymorph.trigger_lycanthropy``.

    The function itself has no direct ``lax.cond`` / ``lax.switch`` sites,
    but it dispatches into ``polymorph_player`` whose three internal
    conds expand under ``vmap``.  Routing through
    ``polymorph_player_brax`` propagates the Brax flattening.

    RNG draw order is identical to the original (single ``rng`` handed
    down to the polymorph path).
    """
    form_i8 = jnp.int8(int(were_form_idx)) if isinstance(were_form_idx, int) \
        else were_form_idx.astype(jnp.int8)
    form_i16 = jnp.int16(int(were_form_idx)) if isinstance(were_form_idx, int) \
        else were_form_idx.astype(jnp.int16)

    state = polymorph_player_brax(state, rng, form_i16, False)

    # Override poly_timer with the shorter were-form duration.
    poly = state.polymorph.replace(
        poly_timer=jnp.int16(_LYCANTHROPY_FORM_DURATION),
        lycanthropy_form=form_i8,
    )
    return state.replace(polymorph=poly)
