"""Brax-style rewrite of the three movement helpers from ``action_dispatch``.

This module mirrors the byte-parity semantics of:

  * ``_try_step``       (action_dispatch.py:486)
  * ``_try_step_inner`` (action_dispatch.py:635)
  * ``_move_branch``    (action_dispatch.py:837)

with one structural change: every ``jax.lax.cond`` / ``jax.lax.switch`` /
``jax.lax.while_loop`` is replaced with an always-run path combined with
``jnp.where`` masking (Brax / Craftax pattern).  Fixed-iteration
``jax.lax.fori_loop`` / ``jax.lax.scan`` are kept where the original used them
(they are HLO-cheap; only data-dependent control flow gets flattened).

WHY: under the 46-branch ``action_dispatch_brax`` fan-out, every movement
handler is *always* traced.  The original nested ``lax.cond`` chain explodes
into separate HLO regions per branch; the Brax form collapses them into one
straight-line graph that the compiler can DCE / common-sub-expression-share.

BYTE-PARITY CONSTRAINTS
=======================

1. RNG draw order preserved exactly.  Every ``jax.random.split`` is performed
   in the same order as the original — both "active" and "skipped" branches
   draw their keys identically because we always-run them.
2. Mutations are byte-identical: where the original returned ``s`` unchanged
   (no-op branch), we always compute the "would-be" branch and gate the write
   with ``jnp.where`` so the resulting pytree equals ``s`` field-for-field.
3. State pytree shape is preserved: ``_select_state(pred, a, b)`` uses
   ``jax.tree_util.tree_map`` so output pytree topology equals input.

NOTE ON SHADOW RNG DRAWS
========================

The original ``_attack_branch`` / ``_pet_swap_branch`` / ``_do_drown`` /
``_do_open`` only consume the shared ``rng`` on the active path.  Because
JAX's ``random.split`` is pure-functional, calling it eagerly on every step
does not perturb the ``state.rng`` carry — the splits stay local to this
function frame and are simply discarded by the ``jnp.where`` selection when
the branch was inactive.  Vendor-side draws (state.vendor_rng) ARE part of
the state pytree, so we use the same tree_map gating to mask the vendor
counter advance on inactive branches.

CITES (vendor): hack.c::domove / hack.c::test_move / hack.c::pooleffects /
trap.c::dotrap / ball.c::drag_ball / ball.c::ballfall / engrave.c::wipe_engr_at
/ uhitm.c:474-502 (pet-swap "foo" gate) / lock.c::doopen.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.constants import TileType
from Nethax.nethax.subsystems.traps import trigger_trap, TrapType
from Nethax.nethax.subsystems.combat import melee_attack as _combat_melee_attack
from Nethax.nethax.subsystems.features import DoorState

# Re-use the module-private constants / helpers from action_dispatch verbatim.
# This module is a pure-additive rewrite — it does not redefine the originals.
from Nethax.nethax.subsystems.action_dispatch import (
    _IS_SOLID,
    _NUM_TILE_TYPES,
    _DIR_TABLE,
    _current_level_terrain,
    _flat_level_idx,
    _apply_fov,
)


# ---------------------------------------------------------------------------
# Pytree-aware where: select between two EnvStates field-by-field.
# ---------------------------------------------------------------------------

def _select_state(pred: jax.Array, on_true, on_false):
    """``jnp.where(pred, on_true, on_false)`` lifted over a pytree.

    Both ``on_true`` and ``on_false`` must have identical pytree shape — the
    same invariant the originals' ``jax.lax.cond`` branches enforce.
    """
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(pred, a, b),
        on_true,
        on_false,
    )


# ===========================================================================
# _move_branch_brax
# ===========================================================================

def _move_branch_brax(state, dy: int, dx: int, rng: jax.Array,
                      target, in_bounds, terrain_2d, map_h, map_w):
    """Brax-style standard movement (wall / door / trap).

    Flattened ``lax.cond``s in original:
      (a) features open-on-bump
      (b) terrain open-on-bump tile rewrite
      (c) trap trigger on new tile
      (d) drown on water entry
      (e) mention_walls message emit
    """
    pos = state.player_pos.astype(jnp.int32)

    safe_row = jnp.clip(target[0], 0, map_h - 1)
    safe_col = jnp.clip(target[1], 0, map_w - 1)
    tile_val = terrain_2d[safe_row, safe_col].astype(jnp.int32)

    # --- Door bump (lock.c doopen / test_move) -----------------------------
    flat_lv = _flat_level_idx(state)
    door_val = state.features.door_state[flat_lv, safe_row, safe_col].astype(jnp.int32)

    target_is_closed_door = (tile_val == jnp.int32(TileType.CLOSED_DOOR)) & in_bounds
    door_is_locked = door_val == jnp.int32(DoorState.LOCKED)

    open_on_bump = target_is_closed_door & ~door_is_locked
    blocked_by_lock = target_is_closed_door & door_is_locked  # noqa: F841 (parity field)

    # Pre-roll trap damage — RNG draw preserved exactly (always happens in
    # original, hoisted outside the cond there too).
    bump_trap_dmg_roll = jax.random.randint(rng, (), minval=1, maxval=11, dtype=jnp.int32)

    # ---- (a) features open-on-bump : Brax-flatten ------------------------
    f = state.features
    lv_, row_, col_ = flat_lv, safe_row, safe_col
    current_ = f.door_state[lv_, row_, col_].astype(jnp.int32)
    is_closed_ = current_ == jnp.int32(DoorState.CLOSED)
    is_trapped_ = f.door_trapped[lv_, row_, col_]
    new_val_ = jnp.where(
        is_closed_ & is_trapped_,
        jnp.int32(DoorState.GONE),
        jnp.where(is_closed_ & ~is_trapped_, jnp.int32(DoorState.OPEN), current_),
    ).astype(jnp.int8)
    damage_open = jnp.where(is_closed_ & is_trapped_, bump_trap_dmg_roll, jnp.int32(0))
    f_open_ds = f.door_state.at[lv_, row_, col_].set(new_val_)
    f_open_trapped = jnp.where(
        is_closed_ & is_trapped_,
        f.door_trapped.at[lv_, row_, col_].set(jnp.bool_(False)),
        f.door_trapped,
    )
    # Gate at field level — only apply when open_on_bump.
    new_door_state = jnp.where(open_on_bump, f_open_ds, f.door_state)
    new_door_trapped = jnp.where(open_on_bump, f_open_trapped, f.door_trapped)
    new_features = f.replace(door_state=new_door_state, door_trapped=new_door_trapped)
    open_damage = jnp.where(open_on_bump, damage_open, jnp.int32(0))

    # ---- (b) terrain rewrite to OPEN_DOOR on bump ------------------------
    t = state.terrain
    t_after_open = t.at[
        state.dungeon.current_branch,
        state.dungeon.current_level - 1,
        safe_row,
        safe_col,
    ].set(jnp.int8(TileType.OPEN_DOOR))
    new_terrain = jnp.where(open_on_bump, t_after_open, t)

    # Movement gates (unchanged from original — already where-based).
    safe_tile = jnp.clip(tile_val, 0, _NUM_TILE_TYPES - 1)
    is_solid = _IS_SOLID[safe_tile]
    door_blocked = target_is_closed_door

    is_diagonal_move = (jnp.int32(dy) != jnp.int32(0)) & (jnp.int32(dx) != jnp.int32(0))
    cardA_row = pos[0] + jnp.int32(dy)
    cardA_col = pos[1]
    cardB_row = pos[0]
    cardB_col = pos[1] + jnp.int32(dx)
    cardA_safe_r = jnp.clip(cardA_row, 0, map_h - 1)
    cardA_safe_c = jnp.clip(cardA_col, 0, map_w - 1)
    cardB_safe_r = jnp.clip(cardB_row, 0, map_h - 1)
    cardB_safe_c = jnp.clip(cardB_col, 0, map_w - 1)
    cardA_tile = jnp.clip(terrain_2d[cardA_safe_r, cardA_safe_c].astype(jnp.int32),
                          0, _NUM_TILE_TYPES - 1)
    cardB_tile = jnp.clip(terrain_2d[cardB_safe_r, cardB_safe_c].astype(jnp.int32),
                          0, _NUM_TILE_TYPES - 1)
    cardA_solid = _IS_SOLID[cardA_tile]
    cardB_solid = _IS_SOLID[cardB_tile]
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _PWIntr
    passes_walls = (
        state.status.intrinsics[int(_PWIntr.PASSES_WALLS)]
        | (state.status.timed_intrinsics[int(_PWIntr.PASSES_WALLS)] > jnp.int32(0))
    )
    diagonal_corner_blocked = (
        is_diagonal_move & cardA_solid & cardB_solid & ~passes_walls
    )

    from Nethax.nethax.subsystems.worm import worm_cross as _worm_cross
    _wc_x1 = pos[0]
    _wc_y1 = pos[1]
    _wc_x2 = pos[0] + jnp.int32(dy)
    _wc_y2 = pos[1] + jnp.int32(dx)
    worm_cross_blocked = (
        is_diagonal_move
        & _worm_cross(state, _wc_x1, _wc_y1, _wc_x2, _wc_y2)
        & ~passes_walls
    )

    src_tile = jnp.clip(terrain_2d[
        jnp.clip(pos[0], 0, map_h - 1),
        jnp.clip(pos[1], 0, map_w - 1),
    ].astype(jnp.int32), 0, _NUM_TILE_TYPES - 1)
    src_is_door = (
        (src_tile == jnp.int32(TileType.OPEN_DOOR))
        | (src_tile == jnp.int32(TileType.CLOSED_DOOR))
    )
    tgt_is_door = (
        (tile_val == jnp.int32(TileType.OPEN_DOOR))
        | (tile_val == jnp.int32(TileType.CLOSED_DOOR))
    )
    diagonal_door_blocked = (
        is_diagonal_move & (src_is_door | tgt_is_door) & ~passes_walls
    )

    can_move = (
        in_bounds & ~is_solid & ~door_blocked
        & ~diagonal_corner_blocked & ~diagonal_door_blocked
        & ~worm_cross_blocked
    )

    new_pos = jnp.where(can_move, target, pos).astype(jnp.int16)
    bump_hp = jnp.maximum(jnp.int32(0), state.player_hp - open_damage)

    # --- Drag ball-and-chain on punished movement -------------------------
    moved = can_move & jnp.any(new_pos != pos.astype(jnp.int16))
    _desired_ball_pos = pos.astype(jnp.int16)
    _desired_r = jnp.clip(_desired_ball_pos[0].astype(jnp.int32), 0, map_h - 1)
    _desired_c = jnp.clip(_desired_ball_pos[1].astype(jnp.int32), 0, map_w - 1)
    _desired_tile = jnp.clip(
        terrain_2d[_desired_r, _desired_c].astype(jnp.int32),
        0, _NUM_TILE_TYPES - 1,
    )
    _desired_solid = _IS_SOLID[_desired_tile]

    _ring_r = new_pos[0].astype(jnp.int32) + _DIR_TABLE[:, 0]
    _ring_c = new_pos[1].astype(jnp.int32) + _DIR_TABLE[:, 1]
    _ring_r_safe = jnp.clip(_ring_r, 0, map_h - 1)
    _ring_c_safe = jnp.clip(_ring_c, 0, map_w - 1)
    _ring_in_bounds = (
        (_ring_r >= 0) & (_ring_r < map_h)
        & (_ring_c >= 0) & (_ring_c < map_w)
    )
    _ring_tiles = jnp.clip(
        terrain_2d[_ring_r_safe, _ring_c_safe].astype(jnp.int32),
        0, _NUM_TILE_TYPES - 1,
    )
    _ring_walkable = _ring_in_bounds & ~_IS_SOLID[_ring_tiles]
    _ring_idx = jnp.argmax(_ring_walkable.astype(jnp.int32))
    _any_walkable = jnp.any(_ring_walkable)
    _reroute_pos = jnp.stack([
        _ring_r_safe[_ring_idx].astype(jnp.int16),
        _ring_c_safe[_ring_idx].astype(jnp.int16),
    ])
    _rerouted_ball = jnp.where(_any_walkable, _reroute_pos, state.ball_pos)
    _ball_when_moved = jnp.where(_desired_solid, _rerouted_ball, _desired_ball_pos)
    new_ball_pos = jnp.where(
        state.is_punished & moved,
        _ball_when_moved,
        state.ball_pos,
    )

    _ball_moved_now = state.is_punished & moved
    _nbp_r = jnp.clip(new_ball_pos[0].astype(jnp.int32), 0, map_h - 1)
    _nbp_c = jnp.clip(new_ball_pos[1].astype(jnp.int32), 0, map_w - 1)
    _new_ball_under = terrain_2d[_nbp_r, _nbp_c].astype(jnp.int8)
    new_ball_under_glyph = jnp.where(
        _ball_moved_now, _new_ball_under, state.ball_under_glyph,
    )
    _chain_r = jnp.clip(pos[0], 0, map_h - 1)
    _chain_c = jnp.clip(pos[1], 0, map_w - 1)
    _new_chain_under = terrain_2d[_chain_r, _chain_c].astype(jnp.int8)
    new_chain_under_glyph = jnp.where(
        _ball_moved_now, _new_chain_under, state.chain_under_glyph,
    )

    _bp_r = jnp.clip(new_ball_pos[0].astype(jnp.int32), 0, map_h - 1)
    _bp_c = jnp.clip(new_ball_pos[1].astype(jnp.int32), 0, map_w - 1)
    _ball_tile = jnp.clip(
        terrain_2d[_bp_r, _bp_c].astype(jnp.int32),
        0, _NUM_TILE_TYPES - 1,
    )
    _ball_on_water = (
        (_ball_tile == jnp.int32(int(TileType.WATER)))
        | (_ball_tile == jnp.int32(int(TileType.POOL)))
    )
    _ball_on_trap_tile = (
        (_ball_tile == jnp.int32(int(TileType.TRAP)))
        | (_ball_tile == jnp.int32(int(TileType.HIDDEN_TRAP)))
    )
    _ball_trap_kind = state.traps.trap_type[flat_lv, _bp_r, _bp_c].astype(jnp.int32)
    _ball_on_pit = _ball_on_trap_tile & (
        (_ball_trap_kind == jnp.int32(int(TrapType.PIT)))
        | (_ball_trap_kind == jnp.int32(int(TrapType.SPIKED_PIT)))
    )
    _pullback_active = state.is_punished & moved
    _pull_water = _pullback_active & _ball_on_water
    _pull_pit = _pullback_active & _ball_on_pit
    new_in_water = state.player_in_water | _pull_water
    new_in_trap = state.player_in_trap | _pull_pit
    new_trap_type = jnp.where(
        _pull_pit, jnp.int8(int(TrapType.PIT)), state.player_trap_type
    )
    new_trap_timer = jnp.where(
        _pull_pit, jnp.int16(4), state.player_trap_timer
    )

    state_mid = state.replace(
        player_pos=new_pos,
        features=new_features,
        terrain=new_terrain,
        player_hp=bump_hp,
        ball_pos=new_ball_pos,
        ball_under_glyph=new_ball_under_glyph,
        chain_under_glyph=new_chain_under_glyph,
        player_in_water=new_in_water,
        player_in_trap=new_in_trap,
        player_trap_type=new_trap_type,
        player_trap_timer=new_trap_timer,
    )

    # ---- (c) trap trigger : Brax-flatten ---------------------------------
    actually_moved = jnp.array_equal(new_pos, target.astype(jnp.int16))
    new_tile_val = terrain_2d[
        jnp.clip(new_pos[0].astype(jnp.int32), 0, map_h - 1),
        jnp.clip(new_pos[1].astype(jnp.int32), 0, map_w - 1),
    ].astype(jnp.int32)

    on_trap = actually_moved & (
        (new_tile_val == jnp.int32(TileType.TRAP)) |
        (new_tile_val == jnp.int32(TileType.HIDDEN_TRAP))
    )

    trap_pos = jnp.array(
        [flat_lv, new_pos[0].astype(jnp.int32), new_pos[1].astype(jnp.int32)],
        dtype=jnp.int32,
    )

    trap_rng, new_rng = jax.random.split(state_mid.rng)

    # Always run trigger_trap, then gate the result through tree_map(where).
    trap_traps, trap_dmg_val, trap_se_val = trigger_trap(state_mid.traps, trap_rng, trap_pos)
    # Identity outputs when no trap fires.
    id_traps = state_mid.traps
    id_dmg = jnp.int32(0)
    id_se = jnp.zeros(6, dtype=jnp.int32)
    new_traps = _select_state(on_trap, trap_traps, id_traps)
    trap_dmg = jnp.where(on_trap, trap_dmg_val, id_dmg)
    trap_se = jnp.where(on_trap, trap_se_val, id_se)

    new_hp = jnp.maximum(
        jnp.int32(0),
        state_mid.player_hp - trap_dmg,
    )

    from Nethax.nethax.subsystems.status_effects import TimedStatus
    from Nethax.nethax.subsystems.traps import _SE_LEVEL_DESCEND
    freeze_turns = trap_se[0]
    sleep_turns = trap_se[1]

    old_frozen = state_mid.status.timed_statuses[int(TimedStatus.FROZEN)]
    old_sleep = state_mid.status.timed_statuses[int(TimedStatus.SLEEP)]
    new_frozen = jnp.maximum(old_frozen, freeze_turns)
    new_sleep = jnp.maximum(old_sleep, sleep_turns)

    new_timed = state_mid.status.timed_statuses \
        .at[int(TimedStatus.FROZEN)].set(new_frozen) \
        .at[int(TimedStatus.SLEEP)].set(new_sleep)
    new_status = state_mid.status.replace(timed_statuses=new_timed)

    state_final = state_mid.replace(
        traps=new_traps,
        player_hp=new_hp,
        status=new_status,
        rng=new_rng,
    )

    # HOLE / TRAPDOOR descend (already where-based in original).
    descend = trap_se[_SE_LEVEL_DESCEND] > jnp.int32(0)
    _max_level_branch = jnp.int8(state_final.terrain.shape[1])
    from Nethax.nethax.dungeon.branches import Branch as _Branch
    _br = state_final.dungeon.current_branch.astype(jnp.int32)
    _is_sokoban = _br == jnp.int32(int(_Branch.SOKOBAN))
    _is_endgame = _br == jnp.int32(int(_Branch.ENDGAME))
    _not_branch_bottom = state_final.dungeon.current_level < _max_level_branch
    _can_fall_thru = _not_branch_bottom & ~_is_sokoban & ~_is_endgame
    descend = descend & _can_fall_thru
    _bumped_level = jnp.minimum(
        state_final.dungeon.current_level + jnp.int8(1),
        _max_level_branch,
    )
    _new_level = jnp.where(descend, _bumped_level, state_final.dungeon.current_level)
    _new_deepest = jnp.maximum(
        state_final.scoring.deepest_level, _new_level.astype(jnp.int8)
    )

    from Nethax.nethax.subsystems.traps import _SE_LEVEL_TELE
    level_tele_flag = trap_se[_SE_LEVEL_TELE] > jnp.int32(0)
    rng_lt, _new_rng2 = jax.random.split(state_final.rng)
    rand_lvl = jax.random.randint(
        rng_lt, (), 1, jnp.maximum(_max_level_branch.astype(jnp.int32) + 1, 2),
        dtype=jnp.int32,
    ).astype(jnp.int8)
    _new_level = jnp.where(level_tele_flag, rand_lvl, _new_level)
    _new_deepest = jnp.maximum(_new_deepest, _new_level.astype(jnp.int8))

    state_final = state_final.replace(
        dungeon=state_final.dungeon.replace(current_level=_new_level),
        scoring=state_final.scoring.replace(deepest_level=_new_deepest),
        rng=_new_rng2,
    )

    # --- ballfall (already where-based) -----------------------------------
    from Nethax.nethax.subsystems.inventory import ArmorSlot as _ArmorSlot
    from Nethax.nethax.subsystems.inventory import is_hard_helmet as _is_hard_helmet
    from Nethax.nethax.subsystems.inventory import (
        MAX_INVENTORY_SLOTS as _LIT_MAX_INV,
        MAX_GROUND_STACK as _LIT_MAX_GS,
    )
    descended_punished = descend & state_final.is_punished

    # --- drag_down litter (uses lax.scan with fixed length — kept) --------
    _lit_br = state_final.dungeon.current_branch.astype(jnp.int32)
    _lit_lv = (state_final.dungeon.current_level - jnp.int8(1)).astype(jnp.int32)
    _lit_r = jnp.clip(state_final.player_pos[0].astype(jnp.int32), 0, map_h - 1)
    _lit_c = jnp.clip(state_final.player_pos[1].astype(jnp.int32), 0, map_w - 1)
    _lit_rng_base, _new_rng_lit = jax.random.split(state_final.rng)

    def _litter_body(carry, slot_idx):
        gi, inv_items = carry
        slot_rng = jax.random.fold_in(_lit_rng_base, slot_idx)
        roll = jax.random.randint(slot_rng, (), 0, 4, dtype=jnp.int32)
        slot_occupied = inv_items.category[slot_idx] != jnp.int8(0)
        do_drop = descended_punished & slot_occupied & (roll == jnp.int32(0))

        def _find_empty(c, s):
            found, p = c
            cat = gi.category[_lit_br, _lit_lv, _lit_r, _lit_c, s]
            empty = (cat == jnp.int8(0))
            p = jnp.where(~found & empty, s, p)
            found = found | empty
            return (found, p), None
        (gs_found, gs_pos), _ = jax.lax.scan(
            _find_empty,
            (jnp.bool_(False), jnp.int32(0)),
            jnp.arange(_LIT_MAX_GS, dtype=jnp.int32),
        )
        write = do_drop & gs_found
        gs_safe = jnp.clip(gs_pos, 0, _LIT_MAX_GS - 1)

        def _w(g_arr, inv_arr):
            cur = g_arr[_lit_br, _lit_lv, _lit_r, _lit_c, gs_safe]
            return g_arr.at[_lit_br, _lit_lv, _lit_r, _lit_c, gs_safe].set(
                jnp.where(write, inv_arr[slot_idx], cur)
            )

        new_gi = gi.replace(
            category    = _w(gi.category,    inv_items.category),
            type_id     = _w(gi.type_id,     inv_items.type_id),
            buc_status  = _w(gi.buc_status,  inv_items.buc_status),
            enchantment = _w(gi.enchantment, inv_items.enchantment),
            charges     = _w(gi.charges,     inv_items.charges),
            identified  = _w(gi.identified,  inv_items.identified),
            quantity    = _w(gi.quantity,    inv_items.quantity),
            weight      = _w(gi.weight,      inv_items.weight),
            ac_bonus    = _w(gi.ac_bonus,    inv_items.ac_bonus),
            is_two_handed = _w(gi.is_two_handed, inv_items.is_two_handed),
            artifact_idx  = _w(gi.artifact_idx,  inv_items.artifact_idx),
        )

        def _z(inv_arr, zero_val):
            return inv_arr.at[slot_idx].set(
                jnp.where(write, zero_val, inv_arr[slot_idx])
            )
        new_inv_items = inv_items.replace(
            category   = _z(inv_items.category,    jnp.int8(0)),
            type_id    = _z(inv_items.type_id,     jnp.int16(0)),
            buc_status = _z(inv_items.buc_status,  jnp.int8(0)),
            enchantment= _z(inv_items.enchantment, jnp.int8(0)),
            charges    = _z(inv_items.charges,     jnp.int8(0)),
            identified = _z(inv_items.identified,  jnp.bool_(False)),
            quantity   = _z(inv_items.quantity,    jnp.int16(0)),
            weight     = _z(inv_items.weight,      jnp.int32(0)),
            ac_bonus   = _z(inv_items.ac_bonus,    jnp.int8(0)),
            is_two_handed = _z(inv_items.is_two_handed, jnp.bool_(False)),
            artifact_idx  = _z(inv_items.artifact_idx,  jnp.int8(-1)),
        )

        return (new_gi, new_inv_items), None

    (_lit_gi, _lit_inv_items), _ = jax.lax.scan(
        _litter_body,
        (state_final.ground_items, state_final.inventory.items),
        jnp.arange(_LIT_MAX_INV, dtype=jnp.int32),
    )
    state_final = state_final.replace(
        ground_items=_lit_gi,
        inventory=state_final.inventory.replace(items=_lit_inv_items),
        rng=_new_rng_lit,
    )

    _bf_rng_hit, _bf_rng_dmg, _new_rng3 = jax.random.split(state_final.rng, 3)
    _bf_hit_roll = jax.random.randint(_bf_rng_hit, (), 0, 5, dtype=jnp.int32)
    _bf_hits = descended_punished & (_bf_hit_roll != jnp.int32(0))
    _bf_dmg = (
        jax.random.randint(_bf_rng_dmg, (), 0, 7, dtype=jnp.int32) + jnp.int32(25)
    )
    _helm_slot = state_final.inventory.worn_armor[int(_ArmorSlot.HELM)]
    _has_helm = _helm_slot >= jnp.int8(0)
    _safe_slot = jnp.clip(_helm_slot.astype(jnp.int32), 0, state_final.inventory.items.type_id.shape[0] - 1)
    _helm_tid = state_final.inventory.items.type_id[_safe_slot]
    _hard_helm = _has_helm & _is_hard_helmet(_helm_tid)
    _bf_dmg = jnp.where(_hard_helm, jnp.int32(3), _bf_dmg)
    _bf_dmg = jnp.where(_bf_hits, _bf_dmg, jnp.int32(0))
    _new_hp = jnp.maximum(
        state_final.player_hp - _bf_dmg, jnp.int32(0)
    ).astype(state_final.player_hp.dtype)
    state_final = state_final.replace(
        player_hp=_new_hp,
        done=state_final.done | (_new_hp == jnp.int32(0)),
        rng=_new_rng3,
    )

    # --- Elbereth dust wipe (already where-based) -------------------------
    from Nethax.nethax.subsystems.engrave import is_elbereth_at, ENGR_DUST
    wipe_rng, _ = jax.random.split(state_final.rng)
    wipe_r = state_final.player_pos[0].astype(jnp.int32)
    wipe_c = state_final.player_pos[1].astype(jnp.int32)
    is_elb = is_elbereth_at(state_final.engrave, wipe_r, wipe_c)
    is_dust = (
        state_final.engrave.engraving_kind[wipe_r, wipe_c].astype(jnp.int32)
        == jnp.int32(ENGR_DUST)
    )
    wipe_roll = jax.random.uniform(wipe_rng)
    do_wipe = is_elb & is_dust & actually_moved & (wipe_roll < 0.25)
    new_has_engraving = jnp.where(
        do_wipe,
        state_final.engrave.has_engraving.at[wipe_r, wipe_c].set(jnp.bool_(False)),
        state_final.engrave.has_engraving,
    )
    state_final = state_final.replace(
        engrave=state_final.engrave.replace(has_engraving=new_has_engraving)
    )

    # --- Lava entry (already where-based) ---------------------------------
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _LavaIntr
    _lava_r = state_final.player_pos[0].astype(jnp.int32)
    _lava_c = state_final.player_pos[1].astype(jnp.int32)
    _lava_tile = _current_level_terrain(state_final)[
        jnp.clip(_lava_r, 0, map_h - 1),
        jnp.clip(_lava_c, 0, map_w - 1),
    ].astype(jnp.int32)
    _on_lava = (_lava_tile == jnp.int32(TileType.LAVA))
    _has_fire_res = (
        state_final.status.intrinsics[int(_LavaIntr.RESIST_FIRE)]
        | (state_final.status.timed_intrinsics[int(_LavaIntr.RESIST_FIRE)] > jnp.int32(0))
    )
    _has_levitation = (
        state_final.status.intrinsics[int(_LavaIntr.LEVITATION)]
        | (state_final.status.timed_intrinsics[int(_LavaIntr.LEVITATION)] > jnp.int32(0))
    )
    _has_flying = (
        state_final.status.intrinsics[int(_LavaIntr.FLYING)]
        | (state_final.status.timed_intrinsics[int(_LavaIntr.FLYING)] > jnp.int32(0))
    )
    _has_wwalk = (
        state_final.status.intrinsics[int(_LavaIntr.WWALKING)]
        | (state_final.status.timed_intrinsics[int(_LavaIntr.WWALKING)] > jnp.int32(0))
    )
    lava_rng, new_rng_after_lava = jax.random.split(state_final.rng)
    lava_rolls = jax.random.randint(lava_rng, (6,), 0, 6, dtype=jnp.int32) + jnp.int32(1)
    lava_dmg = jnp.sum(lava_rolls).astype(jnp.int32)
    _survives_lava = (
        _has_fire_res | _has_levitation | _has_flying
        | (_has_wwalk & (lava_dmg < state_final.player_hp))
    )
    _lava_kills = _on_lava & actually_moved & ~_survives_lava
    _wwalk_dmg = (
        _on_lava & actually_moved & _has_wwalk
        & ~_has_fire_res & ~_has_levitation & ~_has_flying
    )
    new_hp_after_lava = jnp.where(
        _lava_kills,
        jnp.int32(0),
        jnp.where(_wwalk_dmg,
                  jnp.maximum(state_final.player_hp - lava_dmg, jnp.int32(0)),
                  state_final.player_hp),
    )
    new_done_after_lava = state_final.done | _lava_kills
    state_final = state_final.replace(
        player_hp=new_hp_after_lava,
        done=new_done_after_lava,
        rng=new_rng_after_lava,
    )

    # --- Water entry/exit (already where-based) ---------------------------
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intrinsic
    _new_r = state_final.player_pos[0].astype(jnp.int32)
    _new_c = state_final.player_pos[1].astype(jnp.int32)
    _new_tile = _current_level_terrain(state_final)[
        jnp.clip(_new_r, 0, map_h - 1),
        jnp.clip(_new_c, 0, map_w - 1),
    ].astype(jnp.int32)
    _is_water_tile = (
        (_new_tile == jnp.int32(TileType.WATER))
        | (_new_tile == jnp.int32(TileType.POOL))
    )
    _has_swimming = (
        state_final.status.intrinsics[int(_Intrinsic.SWIMMING)]
        | (state_final.status.timed_intrinsics[int(_Intrinsic.SWIMMING)] > jnp.int32(0))
    )
    _new_in_water = jnp.where(
        actually_moved,
        _is_water_tile & ~_has_swimming,
        state_final.player_in_water,
    )
    state_final = state_final.replace(player_in_water=_new_in_water)

    # ---- (d) drown on entry : Brax-flatten -------------------------------
    from Nethax.nethax.subsystems.water import drown as _drown
    _entered_water = (
        actually_moved & _is_water_tile & ~_has_swimming & _new_in_water
    )
    _drown_rng, _drown_new_rng = jax.random.split(state_final.rng)
    # Always compute drowned-state; gate via pytree-where.
    drowned_state = _drown(state_final, _drown_rng).replace(rng=_drown_new_rng)
    state_final = _select_state(_entered_water, drowned_state, state_final)

    # ---- (e) mention_walls message : Brax-flatten ------------------------
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    _bump_wall = in_bounds & is_solid & ~target_is_closed_door
    _is_wall_tile = tile_val == jnp.int32(int(TileType.WALL))
    _bump_msg_id = jnp.where(
        _is_wall_tile,
        jnp.int32(int(_MsgId.ITS_A_WALL)),
        jnp.int32(int(_MsgId.ITS_SOLID_STONE)),
    )
    # Always emit, then gate via tree_map(where) on the messages substate.
    bumped_messages = _msg_emit(state_final.messages, _bump_msg_id)
    new_messages = _select_state(_bump_wall, bumped_messages, state_final.messages)
    state_final = state_final.replace(messages=new_messages)

    _move_blocked_no_time = (~can_move) & (~open_on_bump)
    state_final = state_final.replace(
        action_consumed_turn=jnp.where(
            _move_blocked_no_time,
            jnp.bool_(False),
            state_final.action_consumed_turn,
        ),
    )

    return _apply_fov(state_final)


# ===========================================================================
# _try_step_inner_brax
# ===========================================================================

def _try_step_inner_brax(state, dy: int, dx: int, rng: jax.Array):
    """Brax-style inner body (no no-op gate; runs the move/attack fan-out).

    Flattened ``lax.cond``s in original:
      (f) pet_swap   vs (g) peaceful_bump vs (h) attack vs (i) move
      Inside attack: kill_xp / scoring  (2 lax.cond)
      Inside pet_swap: foo_blocked / do_swap (1 lax.cond)
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS

    # Direction randomization (already where-based).
    confused = state.status.timed_statuses[int(_TS.CONFUSION)] > jnp.int32(0)
    stunned = state.status.timed_statuses[int(_TS.STUNNED)] > jnp.int32(0)
    impaired = confused | stunned

    rng, rng_imp, rng_dy, rng_dx = jax.random.split(rng, 4)
    do_randomize = impaired & (jax.random.uniform(rng_imp) < jnp.float32(0.5))
    rand_dy = jax.random.randint(rng_dy, (), -1, 2).astype(jnp.int32)
    rand_dx = jax.random.randint(rng_dx, (), -1, 2).astype(jnp.int32)
    eff_dy = jnp.where(do_randomize, rand_dy, jnp.int32(dy))
    eff_dx = jnp.where(do_randomize, rand_dx, jnp.int32(dx))

    terrain_2d = _current_level_terrain(state)
    map_h, map_w = terrain_2d.shape

    pos = state.player_pos.astype(jnp.int32)
    target = pos + jnp.stack([eff_dy, eff_dx])

    in_bounds = (
        (target[0] >= 0) & (target[0] < map_h) &
        (target[1] >= 0) & (target[1] < map_w)
    )

    mai = state.monster_ai
    monster_pos_i32 = mai.pos.astype(jnp.int32)
    monster_match = (
        (monster_pos_i32[:, 0] == target[0])
        & (monster_pos_i32[:, 1] == target[1])
        & mai.alive
        & in_bounds
    )
    monster_idx = jnp.argmax(monster_match).astype(jnp.int32)
    monster_present_raw = jnp.any(monster_match)
    target_peaceful = mai.peaceful[monster_idx]
    target_tame = mai.tame[monster_idx] > jnp.int8(0)
    is_pet_swap = monster_present_raw & target_tame
    is_peaceful_bump = monster_present_raw & target_peaceful & ~target_tame
    is_hostile_attack = monster_present_raw & ~target_peaceful & ~target_tame
    # When no monster present at all: take the move branch.
    is_move = ~monster_present_raw

    # ---- (h) attack branch : compute always --------------------------------
    was_alive = state.monster_ai.alive[monster_idx]
    attacked, _dmg, _hit = _combat_melee_attack(state, rng, monster_idx)
    killed = was_alive & ~attacked.monster_ai.alive[monster_idx]

    from Nethax.nethax.subsystems.experience import (
        experience as _xp_experience,
        more_experienced as _xp_more_experienced,
    )
    from Nethax.nethax.subsystems.scoring import (
        record_kill as _scoring_record_kill,
    )
    entry_post = attacked.monster_ai.entry_idx[monster_idx].astype(jnp.int32)
    kc = attacked.scoring.monsters_killed
    mcl = attacked.monster_ai.mcloned[monster_idx]
    xp_award = _xp_experience(entry_post, kc, mcloned=mcl)

    # XP grant: always compute, mask.
    attacked_xp = _xp_more_experienced(attacked, xp_award, jnp.int32(0))
    attacked = _select_state(killed, attacked_xp, attacked)

    # Scoring kill record: always compute, mask.
    attacked_scored = attacked.replace(
        scoring=_scoring_record_kill(attacked.scoring, xp_award)
    )
    attacked = _select_state(killed, attacked_scored, attacked)

    # Confuse-attack-on-hit (already where-based).
    pending = attacked.status.confuse_attack_pending
    apply_confuse_hit = pending & _hit & ~killed
    old_ct = attacked.monster_ai.confuse_timer[monster_idx].astype(jnp.int32)
    new_ct = jnp.where(apply_confuse_hit, jnp.maximum(old_ct, jnp.int32(15)), old_ct)
    new_ct_arr = attacked.monster_ai.confuse_timer.at[monster_idx].set(
        new_ct.astype(attacked.monster_ai.confuse_timer.dtype)
    )
    new_mai = attacked.monster_ai.replace(confuse_timer=new_ct_arr)
    new_pending = jnp.where(pending & _hit, jnp.bool_(False), pending)
    new_status = attacked.status.replace(confuse_attack_pending=new_pending)
    attack_state = attacked.replace(monster_ai=new_mai, status=new_status)

    # ---- (f) pet_swap branch : compute always ------------------------------
    from Nethax.nethax.vendor_rng import rn2_jax as _rn2j, rnd_jax as _rndj
    from Nethax.nethax.subsystems.messages import (
        emit as _msg_emit,
        MessageId as _MsgId,
    )

    vrng_in = state.vendor_rng
    vrng_after_rn7, rn7 = _rn2j(vrng_in, jnp.int64(7))
    foo = rn7 == jnp.int32(0)

    # foo_blocked sub-branch (vendor draws rnd(6) + emits message).
    vrng_after_flee, _flee_dur = _rndj(vrng_after_rn7, jnp.int64(6))
    msgs_after_pet_stop = _msg_emit(state.messages, int(_MsgId.YOU_STOP_PET_IN_WAY))
    foo_state = state.replace(
        vendor_rng=vrng_after_flee, messages=msgs_after_pet_stop
    )

    # do_swap sub-branch (positional swap + FOV).
    old_pos_i32 = state.player_pos.astype(jnp.int32)
    new_player_pos_swap = target.astype(jnp.int16)
    new_mon_pos_swap = state.monster_ai.pos.at[monster_idx].set(
        old_pos_i32.astype(state.monster_ai.pos.dtype)
    )
    new_mai_swap = state.monster_ai.replace(pos=new_mon_pos_swap)
    swap_pre = state.replace(
        player_pos=new_player_pos_swap,
        monster_ai=new_mai_swap,
        vendor_rng=vrng_after_rn7,
    )
    swap_state = _apply_fov(swap_pre)

    pet_swap_state = _select_state(foo, foo_state, swap_state)

    # ---- (i) move branch : compute always ----------------------------------
    move_state = _move_branch_brax(
        state, eff_dy, eff_dx, rng, target, in_bounds, terrain_2d, map_h, map_w
    )

    # ---- (g) peaceful_bump : state stays unchanged -------------------------
    peaceful_state = state

    # ---- Selection priority (matches original cond chain) ------------------
    # 1. pet_swap (is_pet_swap)
    # 2. peaceful (is_peaceful_bump)
    # 3. attack   (is_hostile_attack)
    # 4. move     (is_move)
    #
    # Start with move_state as the default, then layer higher-priority
    # branches via _select_state.  Because the four predicates are mutually
    # exclusive (by construction above), order between pet/peaceful/attack
    # doesn't matter — but we follow original order for clarity.
    out = move_state
    out = _select_state(is_hostile_attack, attack_state, out)
    out = _select_state(is_peaceful_bump, peaceful_state, out)
    out = _select_state(is_pet_swap, pet_swap_state, out)
    return out


# ===========================================================================
# _try_step_brax
# ===========================================================================

def _try_step_brax(state, dy: int, dx: int, rng: jax.Array):
    """Brax-style outer wrapper: no-op gate fan-out flattened.

    Flattened ``lax.cond``s in original:
      (j) noop_gate vs inner-body  -> compute inner always; gate via where.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS
    from Nethax.nethax.subsystems.occupation import is_occupied as _is_occupied

    is_vomiting = state.status.timed_statuses[int(_TS.VOMITING)] > jnp.int32(0)
    is_busy = _is_occupied(state)

    rng, rng_wl = jax.random.split(rng)
    wl_roll = jax.random.uniform(rng_wl)
    is_wounded = state.status.timed_statuses[int(_TS.WOUNDED_LEGS)] > jnp.int32(0)
    do_limp = is_wounded & (wl_roll < jnp.float32(0.3))

    is_diagonal = jnp.bool_((dy != 0) & (dx != 0))
    blocked_underwater = state.player_in_water & is_diagonal

    rng, rng_trap, rng_trap_pit = jax.random.split(rng, 3)
    from Nethax.nethax.subsystems.traps import TrapType as _TT

    timer = state.player_trap_timer.astype(jnp.int32)
    ttype = state.player_trap_type.astype(jnp.int32)
    has_timer = timer > jnp.int32(0)

    is_bear = ttype == jnp.int32(int(_TT.BEAR_TRAP))
    is_pit = (ttype == jnp.int32(int(_TT.PIT))) | (
        ttype == jnp.int32(int(_TT.SPIKED_PIT))
    )
    is_web = ttype == jnp.int32(int(_TT.WEB))
    _terr_2d = _current_level_terrain(state)
    _pr = state.player_pos[0].astype(jnp.int32)
    _pc = state.player_pos[1].astype(jnp.int32)
    _here_tile = _terr_2d[
        jnp.clip(_pr, 0, _terr_2d.shape[0] - 1),
        jnp.clip(_pc, 0, _terr_2d.shape[1] - 1),
    ].astype(jnp.int32)
    is_lava = _here_tile == jnp.int32(int(TileType.LAVA))
    is_floor = jnp.bool_(False)

    bear_rn5 = jax.random.randint(rng_trap, (), 0, 5, dtype=jnp.int32) == jnp.int32(0)
    bear_decr = is_bear & (is_diagonal | bear_rn5)

    pit_roll = jax.random.randint(rng_trap_pit, (), 0, 6, dtype=jnp.int32)
    pit_decr = is_pit & (pit_roll == jnp.int32(0))

    web_decr = is_web
    lava_decr = is_lava
    floor_decr = is_floor

    decr_now = has_timer & (bear_decr | pit_decr | web_decr | lava_decr | floor_decr)
    new_timer = jnp.where(
        decr_now,
        jnp.maximum(timer - jnp.int32(1), jnp.int32(0)).astype(jnp.int16),
        state.player_trap_timer,
    )
    timer_freed = decr_now & (new_timer == jnp.int16(0))

    legacy_escape = (
        jax.random.randint(rng_trap, (), 0, 4, dtype=jnp.int32) == jnp.int32(0)
    )
    legacy_blocks = state.player_in_trap & ~has_timer & ~legacy_escape

    new_timer_blocks = has_timer & ~timer_freed

    blocked_trap = legacy_blocks | new_timer_blocks

    noop_gate = (
        state.swallow.swallowed | is_vomiting | do_limp
        | blocked_underwater | blocked_trap | is_busy
    )

    cleared = (state.player_in_trap & ~has_timer & legacy_escape) | timer_freed
    state_after_escape = state.replace(
        player_in_trap=jnp.where(cleared, jnp.bool_(False), state.player_in_trap),
        player_trap_timer=new_timer,
        player_trap_type=jnp.where(
            timer_freed, jnp.int8(0), state.player_trap_type
        ),
    )

    # ---- (j) noop_gate fan-out flattened ---------------------------------
    # Always compute the inner body; mask the result via _select_state.
    inner_out = _try_step_inner_brax(state_after_escape, dy, dx, rng)
    return _select_state(noop_gate, state_after_escape, inner_out)
