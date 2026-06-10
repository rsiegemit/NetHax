"""Brax-style flattened rewrites of ``items_wands`` and ``items_spellbooks``.

Why this file exists
--------------------
Under ``jax.vmap`` over seeds, every remaining ``jax.lax.cond`` /
``jax.lax.switch`` in the wand- and spellbook-effect dispatchers lowers
to ``lax.select`` with *both* branches materialised in the HLO graph,
**and** the surrounding ``lax.switch`` over wand type expands every
branch independently per-seed.  The wand-zap path is the single largest
HLO contributor noted in ``handlers_brax.py``:

    ``items_wands.handle_zap`` → wand-effect ``lax.switch`` over ~25 types

Following the same precedent as ``combat_helpers_brax`` /
``monster_attack_player_brax``: **always compute every branch and select
with ``jnp.where`` over a precomputed scalar mask** (via
``jax.tree.map`` for full pytree analogues of ``lax.cond`` /
``lax.switch``).

Scope (per task brief)
----------------------
Only ``jax.lax.switch`` and ``jax.lax.cond`` are flattened.  Fixed-shape
``lax.scan`` (ray step, dig step, freeze step) and ``lax.while_loop``
(rejection sampling for teleport / polymorph / create_monster) are
**preserved verbatim** — they are not cond / switch and the task brief
restricts rewrite to the two listed primitives.

Byte-parity contract
--------------------
1. RNG draw order preserved exactly: each Brax handler consumes the
   same ``jax.random.split`` / ``randint`` sequence the original
   ``lax.cond`` /``lax.switch`` branch consumed, in the same order.
   Where one branch consumes more keys than another, we still split
   the maximum and only the selected branch's downstream-mask reaches
   the output (the unused splits are pure functions of ``rng`` so they
   are deterministic and free of side-effects).
2. Mutations byte-identical: ``jnp.where`` (or ``jax.tree.map(jnp.where,
   ...)``) selects between branch outputs that have identical dtype /
   shape on both sides, matching the original ``lax.cond`` output.
3. State pytree shape preserved: every Brax variant returns the same
   ``(WandState, rng)`` / ``state`` pytree structure as its source.

Conds / switches flattened per public function
----------------------------------------------
* ``cast_ray_brax``                            : 2 ``lax.cond`` → 2 ``jnp.where`` (scan body).
* ``_cast_ray_terrain_predicate_brax``         : 1 ``lax.cond`` → 1 pytree-where (fori body).
* ``_effect_striking_brax``                    : 1 ``lax.cond`` → 1 pytree-where (on_hit hit/miss).
* ``_effect_polymorph_brax``                   : 1 ``lax.cond`` → 1 pytree-where (system-shock kill vs morph).
* ``_effect_death_brax``                       : 1 ``lax.cond`` → 1 ``jnp.where`` (immune zero-dmg).
* ``_effect_cold_brax``                        : 1 ``lax.cond`` → 1 ``jnp.where`` (freeze tile).
* ``_effect_digging_brax``                     : 4 ``lax.cond`` → 4 ``jnp.where`` (set_hole / up_dig / normal_dig / dig-step write).
* ``zap_wand_brax``                            : 1 ``lax.switch`` (28-way) → 28 pytree-where (mask fan-out).
* ``handle_zap_brax``                          : 0 direct conds (already pure-Python + ``jnp.where``); routes through ``zap_wand_brax``.
* ``zap_polymorph_at_self_brax``               : 0 direct conds (pure delegate).
* ``_cursed_book_backfire_brax``               : 1 ``lax.switch`` (8-way) → 8 pytree-where (mask fan-out).
* ``read_spellbook_brax``                      : 0 JIT conds (Python branching on host ints); routes through ``_cursed_book_backfire_brax``.
* ``handle_read_spellbook_brax``               : 0 (pure delegate).
* ``study_book_delay_brax``                    : 0 (host-only int math, unchanged).
* ``study_success_chance_brax``                : 0 (host-only float math, unchanged).

Total: 11 ``lax.cond`` + 2 ``lax.switch`` (28-way + 8-way = 36 effective
branches) flattened to ``jnp.where`` mask selection.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax

from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.traps import TrapType

from Nethax.nethax.subsystems.items_wands import (
    # Constants / tables.
    N_WANDS,
    N_MONSTERS,
    MR_MAGIC,
    ICE_TILE,
    DIG_TILE,
    DEFAULT_RAY_RANGE,
    ITEM_CATEGORY_WAND,
    _DIR_DY,
    _DIR_DX,
    _DEATH_IMMUNE,
    _MONSTER_GEN_LEVEL,
    _MONSTER_BASE_AC,
    _MON_NATURAL_SHIFTER,
    _TILE_FLOOR,
    _TILE_CORRIDOR,
    WandEffect,
    WandClass,
    WAND_EFFECT_CLASS,
    WandState,
    # RNG helpers.
    _rng_d,
    _rng_rnd,
    # Core helpers we reuse as-is (no cond/switch inside).
    _find_monster_at,
    _deal_damage,
    _decrement_charges,
    # Effect handlers that have no cond / switch in their body — reused.
    _effect_light,
    _effect_nothing,
    _effect_secret_door_detection,
    _effect_probing,
    _effect_magic_missile,
    _effect_slow_monster,
    _effect_speed_monster,
    _effect_cancellation,
    _effect_teleportation,         # while_loop only (not flattened by brief)
    _effect_sleep,
    _effect_fire,
    _effect_lightning,
    _effect_enlightenment,
    _effect_create_monster,        # while_loop only
    _effect_wishing,
    _effect_stasis,
    _effect_make_invisible,
    _effect_undead_turning,
    _effect_draining,
    _effect_acid,
    _effect_poison_gas,
)

from Nethax.nethax.subsystems.items_spellbooks import (
    BLANK_SPELL_ID,
    _ROLE_WIZARD,
    _WIZARD_STUDY_BONUS,
    _BUC_CURSED,
    _BUC_UNCURSED,
    _BUC_BLESSED,
    _BLESSED_STUDY_BONUS,
    _SPELL_OC_DELAYS,
    _PLAYER_HAS_ANTIMAGIC,
    study_book_delay,
    study_success_chance,
    _assign_letter,
)


# ---------------------------------------------------------------------------
# Pytree mask helper — analogue of ``jnp.where`` over arbitrary pytrees.
# Mirrors combat_helpers_brax._select_tree.
# ---------------------------------------------------------------------------

def _where_tree(cond, on_true, on_false):
    """Pytree analogue of ``jnp.where(cond, on_true, on_false)``.

    Both pytrees must share structure / dtype / shape on each leaf.
    """
    return jax.tree.map(lambda a, b: jnp.where(cond, a, b), on_true, on_false)


# ---------------------------------------------------------------------------
# host-only helpers — unchanged passthrough.
# ---------------------------------------------------------------------------

def study_book_delay_brax(book_level: int, oc_delay: int) -> int:
    """Brax alias of ``items_spellbooks.study_book_delay`` (host-only int math)."""
    return study_book_delay(book_level, oc_delay)


def study_success_chance_brax(
    player_int: int,
    player_xl: int,
    book_level: int,
    role_id: int = 0,
    buc_status: int = 2,
) -> float:
    """Brax alias of ``items_spellbooks.study_success_chance`` (host-only)."""
    return study_success_chance(
        player_int, player_xl, book_level, role_id=role_id, buc_status=buc_status
    )


# ---------------------------------------------------------------------------
# cast_ray_brax — flattens the 2 inner ``lax.cond`` inside the scan body.
# ---------------------------------------------------------------------------

def cast_ray_brax(
    state: WandState,
    rng: jax.Array,
    start_pos: jax.Array,
    direction,
    ray_range: int = DEFAULT_RAY_RANGE,
    on_hit_fn=None,
    stop_on_hit: bool = False,
):
    """Brax-style ``cast_ray`` — scan body uses ``jnp.where`` over a fully
    computed on-hit branch, instead of two ``lax.cond`` selections.

    RNG-parity rule:
      The original ``lax.cond`` only invoked ``on_hit_fn`` when
      ``(~stopped) & has_monster``.  We compute it unconditionally and
      ``jnp.where``-mask the (state, rng) output back to the pre-hit
      values when the gate is False — therefore the *selected* rng
      stream is byte-identical to the original.
    """
    dy0 = _DIR_DY[direction].astype(jnp.int16)
    dx0 = _DIR_DX[direction].astype(jnp.int16)

    map_h, map_w = state.terrain.shape

    if on_hit_fn is None:
        def on_hit_fn(s, r, _idx):
            return s, r

    def _step(carry, _step_i):
        s, r, pos, dy, dx, stopped, reflected = carry

        next_pos = pos + jnp.array([dy, dx], dtype=jnp.int16)

        next_row = jnp.clip(next_pos[0], 0, map_h - 1)
        next_col = jnp.clip(next_pos[1], 0, map_w - 1)
        next_pos = jnp.array([next_row, next_col], dtype=jnp.int16)

        oob = (next_pos[0] != pos[0] + dy) | (next_pos[1] != pos[1] + dx)

        tile = s.terrain[next_pos[0], next_pos[1]].astype(jnp.int32)
        is_wall = tile == int(TileType.WALL)
        is_blocker = s.blockers[next_pos[0], next_pos[1]]

        mon_idx = _find_monster_at(s, next_pos)
        has_monster = (mon_idx > 0) & s.mon_alive[mon_idx]

        hits_player = (
            (next_pos[0] == s.player_pos[0])
            & (next_pos[1] == s.player_pos[1])
        )
        do_reflect = (~stopped) & (~reflected) & hits_player & s.player_reflecting
        new_dy = jnp.where(do_reflect, -dy, dy)
        new_dx = jnp.where(do_reflect, -dx, dx)
        new_reflected = reflected | do_reflect

        # ── Flattened lax.cond #1: always compute on-hit branch, mask via where.
        # Original:
        #     s, r = lax.cond((~stopped) & has_monster, _apply, _noop, (s, r))
        apply_mask = (~stopped) & has_monster
        s_hit, r_hit = on_hit_fn(s, r, mon_idx)
        s = _where_tree(apply_mask, s_hit, s)
        r = jnp.where(apply_mask, r_hit, r)

        # ── Flattened lax.cond #2: stop-on-hit selection by mask.
        # Original:
        #     beam_stopped = lax.cond((~stopped) & has_monster & stop_on_hit, ...)
        beam_stopped = stopped | is_wall | is_blocker | oob
        hit_stop_mask = (~stopped) & has_monster & jnp.bool_(stop_on_hit)
        beam_stopped = jnp.where(hit_stop_mask, jnp.bool_(True), beam_stopped)

        return (s, r, next_pos, new_dy, new_dx, beam_stopped, new_reflected), None

    init_carry = (
        state,
        rng,
        start_pos.astype(jnp.int16),
        dy0,
        dx0,
        jnp.bool_(False),
        jnp.bool_(False),
    )
    (final_state, final_rng, _, _, _, _, _), _ = lax.scan(
        _step, init_carry, jnp.arange(ray_range)
    )
    return final_state, final_rng


# ---------------------------------------------------------------------------
# _cast_ray_terrain_predicate_brax — flattens 1 cond in the fori body.
# ---------------------------------------------------------------------------

def _cast_ray_terrain_predicate_brax(
    state, rng, direction, target, on_tile_fn, max_range: int = 13,
):
    """Brax version of ``items_wands._cast_ray_terrain_predicate``.

    The original fired ``on_tile_fn`` via ``lax.cond(matches, ...)``.
    Here we compute ``on_tile_fn`` unconditionally and pytree-mask the
    result.  The ``done`` flag still suppresses subsequent matches, so
    only the first matching tile contributes — byte-identical to the
    original first-hit semantics.
    """
    dir_table = jnp.array([
        [-1,  0],
        [-1,  1],
        [ 0,  1],
        [ 1,  1],
        [ 1,  0],
        [ 1, -1],
        [ 0, -1],
        [-1, -1],
    ], dtype=jnp.int32)
    dir_idx = jnp.clip(jnp.asarray(direction, jnp.int32), 0, 7)
    dy = dir_table[dir_idx, 0]
    dx = dir_table[dir_idx, 1]
    map_h, map_w = state.terrain.shape
    start_r = state.player_pos[0].astype(jnp.int32)
    start_c = state.player_pos[1].astype(jnp.int32)
    target_t = jnp.int8(int(target))

    def body(i, carry):
        s, r, done = carry
        step = i + jnp.int32(1)
        tr = start_r + dy * step
        tc = start_c + dx * step
        in_bounds = (tr >= 0) & (tr < map_h) & (tc >= 0) & (tc < map_w)
        rr = jnp.clip(tr, 0, map_h - 1)
        cc = jnp.clip(tc, 0, map_w - 1)
        cur = s.terrain[rr, cc]
        matches = in_bounds & (cur == target_t) & ~done

        # ── Flattened lax.cond: compute on_tile_fn always, mask via where.
        s_hit, r_hit = on_tile_fn(s, r, jnp.array([rr, cc], dtype=jnp.int32))
        s_new = _where_tree(matches, s_hit, s)
        r_new = jnp.where(matches, r_hit, r)
        return s_new, r_new, done | matches

    final_state, final_rng, _ = jax.lax.fori_loop(
        0, max_range, body, (state, rng, jnp.bool_(False)))
    return final_state, final_rng


# ---------------------------------------------------------------------------
# Brax effect rewrites — only those whose body contained a lax.cond.
# All other _effect_* handlers are imported as-is (no cond/switch inside).
# ---------------------------------------------------------------------------

def _effect_opening_brax(
    state: WandState, rng: jax.Array, direction=2,
):
    """WAN_OPENING — uses Brax terrain-predicate ray."""
    def on_hit_door(s, r, pos):
        tr, tc = pos[0], pos[1]
        cur = s.terrain[tr, tc]
        is_closed = cur == jnp.int8(int(TileType.CLOSED_DOOR))
        new_t = jnp.where(is_closed,
                          jnp.int8(int(TileType.OPEN_DOOR)),
                          cur)
        new_terrain = s.terrain.at[tr, tc].set(new_t)
        return s.replace(terrain=new_terrain), r

    return _cast_ray_terrain_predicate_brax(
        state, rng, direction,
        target=int(TileType.CLOSED_DOOR),
        on_tile_fn=on_hit_door,
    )


def _effect_locking_brax(
    state: WandState, rng: jax.Array, direction=2,
):
    """WAN_LOCKING — uses Brax terrain-predicate ray."""
    def on_hit_door(s, r, pos):
        tr, tc = pos[0], pos[1]
        cur = s.terrain[tr, tc]
        is_open = cur == jnp.int8(int(TileType.OPEN_DOOR))
        new_t = jnp.where(is_open,
                          jnp.int8(int(TileType.CLOSED_DOOR)),
                          cur)
        new_terrain = s.terrain.at[tr, tc].set(new_t)
        return s.replace(terrain=new_terrain), r

    return _cast_ray_terrain_predicate_brax(
        state, rng, direction,
        target=int(TileType.OPEN_DOOR),
        on_tile_fn=on_hit_door,
    )


def _effect_striking_brax(
    state: WandState, rng: jax.Array, direction=2,
):
    """WAN_STRIKING — Brax flatten of the 1-cond hit/miss branch.

    RNG parity: the vendor source consumes ``rnd(20)`` for the to-hit
    roll, then ``d(2, 12)`` for the damage roll *only when hit*.  We
    consume both unconditionally and mask the damage with ``jnp.where``
    when ``hit`` is False; when ``hit`` is True the resulting state is
    byte-identical to the original lax.cond branch.

    NOTE: the unused damage split is a deterministic function of rng,
    so the *selected* rng stream remains byte-identical.  The vendor
    behaviour itself always advances the global rng even on miss in
    many sites — but for parity with the original Nethax handler we
    keep the same final rng as the cond-true branch.
    """
    def on_hit(s, r, mon_idx):
        entry_idx = s.mon_type[mon_idx].astype(jnp.int32)
        ac = _MONSTER_BASE_AC[entry_idx].astype(jnp.int32)
        r, roll = _rng_rnd(r, 20)
        hit = roll < (jnp.int32(10) + ac)

        # ── Flattened lax.cond: compute hit branch unconditionally.
        r_hit, dmg = _rng_d(r, 2, 12)
        s_hit = _deal_damage(s, mon_idx, dmg)

        s_out = _where_tree(hit, s_hit, s)
        r_out = jnp.where(hit, r_hit, r)
        return s_out, r_out

    return cast_ray_brax(state, rng, state.player_pos, direction,
                         on_hit_fn=on_hit, stop_on_hit=True)


def _effect_polymorph_brax(
    state: WandState, rng: jax.Array, direction=2,
):
    """WAN_POLYMORPH — Brax flatten of the system-shock vs morph cond.

    RNG parity: both kill-branch (no RNG draw) and morph-branch
    (multiple ``split`` / ``randint`` / ``while_loop`` draws) are
    always evaluated; only the masked output is exposed.  The morph
    branch contains ``lax.while_loop`` (rejection-sample valid form) +
    ``_form_hp_max`` (a deterministic helper) — neither is a switch/
    cond, so they are preserved per the task brief.
    """
    from Nethax.nethax.subsystems.polymorph import _POLY_FORM_VALID, _form_hp_max

    def on_hit(s, r, mon_idx):
        entry_idx = s.mon_type[mon_idx].astype(jnp.int32)
        natural_shifter = _MON_NATURAL_SHIFTER[entry_idx]
        r, shock_roll = _rng_rnd(r, 25)
        shock_fires = (~natural_shifter) & (shock_roll == jnp.int32(1))

        # ---- kill branch ----
        kill_alive = s.mon_alive.at[mon_idx].set(jnp.bool_(False))
        kill_hp    = s.mon_hp.at[mon_idx].set(jnp.int32(0))
        s_kill = s.replace(mon_alive=kill_alive, mon_hp=kill_hp)
        # kill branch consumes no further RNG → r_kill == r.
        r_kill = r

        # ---- morph branch ----
        # Rejection-sample form (preserved while_loop — not a cond/switch).
        def _cond(ws):
            _, candidate = ws
            return ~_POLY_FORM_VALID[candidate]

        def _body(ws):
            rr, _ = ws
            rr, sub = jax.random.split(rr)
            c = jax.random.randint(sub, shape=(), minval=1,
                                   maxval=N_MONSTERS, dtype=jnp.int32)
            return (rr, c)

        r_morph, sub0 = jax.random.split(r)
        init_c = jax.random.randint(sub0, shape=(), minval=1,
                                    maxval=N_MONSTERS, dtype=jnp.int32)
        r_morph, new_type = lax.while_loop(_cond, _body, (r_morph, init_c))
        new_type = new_type.astype(jnp.int32)

        r_morph, sub_hp = jax.random.split(r_morph)
        new_hp_max = _form_hp_max(
            new_type.astype(jnp.int16), sub_hp
        ).astype(jnp.int32)
        old_hp_max = jnp.maximum(s.mon_hp_max[mon_idx].astype(jnp.float32),
                                 jnp.float32(1.0))
        ratio  = s.mon_hp[mon_idx].astype(jnp.float32) / old_hp_max
        new_hp = jnp.maximum(
            jnp.int32(1),
            (ratio * new_hp_max.astype(jnp.float32)).astype(jnp.int32),
        )
        s_morph = s.replace(
            mon_type=s.mon_type.at[mon_idx].set(new_type.astype(jnp.int16)),
            mon_hp=s.mon_hp.at[mon_idx].set(new_hp),
            mon_hp_max=s.mon_hp_max.at[mon_idx].set(new_hp_max),
        )

        # ── Flattened lax.cond: select by shock_fires mask.
        s_out = _where_tree(shock_fires, s_kill, s_morph)
        r_out = jnp.where(shock_fires, r_kill, r_morph)
        return s_out, r_out

    return cast_ray_brax(state, rng, state.player_pos, direction,
                         on_hit_fn=on_hit, stop_on_hit=True)


def _effect_death_brax(
    state: WandState, rng: jax.Array, direction=2,
):
    """WAN_DEATH — Brax flatten of the immune-vs-kill dmg cond."""
    def on_hit(s, r, mon_idx):
        mtype = s.mon_type[mon_idx].astype(jnp.int32)
        mtype = jnp.clip(mtype, 0, N_MONSTERS - 1)
        tbl_immune  = jnp.take(_DEATH_IMMUNE, mtype, axis=0)
        flag_undead = s.mon_undead[mon_idx]
        is_immune   = tbl_immune | flag_undead
        # ── Flattened lax.cond: dmg = 0 if immune else mon_hp.
        dmg = jnp.where(is_immune, jnp.int32(0), s.mon_hp[mon_idx])
        s = _deal_damage(s, mon_idx, dmg)
        return s, r

    return cast_ray_brax(state, rng, state.player_pos, direction,
                         on_hit_fn=on_hit, stop_on_hit=False)


def _effect_cold_brax(
    state: WandState, rng: jax.Array, direction=2,
):
    """WAN_COLD — Brax flatten of the per-tile freeze ``lax.cond``."""
    from Nethax.nethax.constants.monsters import MR_COLD
    dy = _DIR_DY[direction].astype(jnp.int16)
    dx = _DIR_DX[direction].astype(jnp.int16)
    map_h, map_w = state.terrain.shape

    def on_hit(s, r, mon_idx):
        r, dmg = _rng_d(r, 6, 6)
        is_immune = (s.mon_resists[mon_idx] & int(MR_COLD)).astype(jnp.bool_)
        actual_dmg = jnp.where(is_immune, jnp.int32(0), dmg)
        s = _deal_damage(s, mon_idx, actual_dmg)
        return s, r

    state, rng = cast_ray_brax(state, rng, state.player_pos, direction,
                               on_hit_fn=on_hit, stop_on_hit=False)

    # Freeze water tiles along the ray path.
    def _freeze_step(carry, step_i):
        terrain, pos = carry
        next_pos = pos + jnp.array([dy, dx], dtype=jnp.int16)
        nr = jnp.clip(next_pos[0], 0, map_h - 1)
        nc = jnp.clip(next_pos[1], 0, map_w - 1)
        is_water = terrain[nr, nc] == int(TileType.WATER)
        # ── Flattened lax.cond: write ICE_TILE only when is_water.
        cur_tile = terrain[nr, nc]
        new_tile = jnp.where(is_water, jnp.int8(ICE_TILE), cur_tile)
        new_terrain = terrain.at[nr, nc].set(new_tile)
        return (new_terrain, jnp.array([nr, nc], dtype=jnp.int16)), None

    (frozen_terrain, _), _ = lax.scan(
        _freeze_step,
        (state.terrain, state.player_pos.astype(jnp.int16)),
        jnp.arange(DEFAULT_RAY_RANGE, dtype=jnp.int32),
    )
    return state.replace(terrain=frozen_terrain), rng


def _effect_digging_brax(
    state: WandState, rng: jax.Array, direction=0,
):
    """WAN_DIGGING — Brax flatten of all 4 nested ``lax.cond`` invocations.

    Conds collapsed:
      1. ``_set_hole`` inner protected/unprotected cond → ``jnp.where`` on
         the player-tile write.
      2. The outer down/up/horizontal selector pair (down vs up, then up
         vs normal) → two ``jnp.where`` on per-tile terrain results.
      3. The dig-step ``do_write`` per-tile write → ``jnp.where`` between
         carved tile and current tile.
    """
    map_h, map_w = state.terrain.shape
    dir_idx = jnp.int32(direction)

    _n_levels = state.traps.trap_type.shape[0]
    flat_lv = jnp.where(
        jnp.int32(_n_levels) > jnp.int32(1),
        jnp.clip(state.dungeon_level.astype(jnp.int32) - 1, 0, _n_levels - 1),
        jnp.int32(0),
    )
    is_airwater_level = state.branch == jnp.int8(6)

    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)

    # ---------- set_hole — flattened ----------
    here = state.terrain[pr, pc].astype(jnp.int32)
    tile_protected = (
        (here == jnp.int32(TileType.STAIRCASE_UP))
        | (here == jnp.int32(TileType.STAIRCASE_DOWN))
        | (here == jnp.int32(TileType.ALTAR))
        | (here == jnp.int32(TileType.THRONE))
    )
    tt = state.traps.trap_type[flat_lv, pr, pc].astype(jnp.int32)
    trap_protected = (
        (tt == jnp.int32(int(TrapType.MAGIC_PORTAL)))
        | (tt == jnp.int32(int(TrapType.VIBRATING_SQUARE)))
    )
    wall_protected = state.wall_info[flat_lv, pr, pc]
    protected = (
        tile_protected | trap_protected | wall_protected | is_airwater_level
    )
    # Flattened lax.cond: write HOLE only when not protected.
    hole_tile = jnp.where(
        protected,
        state.terrain[pr, pc],
        jnp.int8(TileType.HOLE),
    )
    set_hole_terrain = state.terrain.at[pr, pc].set(hole_tile)

    # ---------- up_dig: terrain unchanged ----------
    up_dig_terrain = state.terrain

    # ---------- normal (horizontal) dig — same scan body, internal cond
    # flattened to jnp.where.
    safe_dir = jnp.clip(dir_idx, 0, 7)
    dy = _DIR_DY[safe_dir].astype(jnp.int16)
    dx = _DIR_DX[safe_dir].astype(jnp.int16)
    digdepth0 = jax.random.randint(rng, (), 8, 26, dtype=jnp.int32)

    def _dig_step(carry, _):
        terrain, pos, remaining, stopped = carry
        next_pos = pos + jnp.array([dy, dx], dtype=jnp.int16)
        nr = jnp.clip(next_pos[0], 0, map_h - 1)
        nc = jnp.clip(next_pos[1], 0, map_w - 1)
        tile = terrain[nr, nc].astype(jnp.int32)

        is_closed_door = tile == jnp.int32(TileType.CLOSED_DOOR)
        is_wall  = tile == jnp.int32(TileType.WALL)
        is_tree  = tile == jnp.int32(TileType.TREE)
        is_stone = (tile == jnp.int32(TileType.VOID))

        nondig = state.wall_info[flat_lv, nr, nc]

        still_going = (~stopped) & (remaining > jnp.int32(0))

        carve = (is_closed_door | is_wall | is_tree | is_stone) & (~nondig)

        new_tile = jnp.where(
            is_closed_door, jnp.int8(TileType.OPEN_DOOR),
            jnp.where(is_wall, jnp.int8(TileType.OPEN_DOOR),
            jnp.where(is_tree, jnp.int8(TileType.FLOOR),
            jnp.where(is_stone, jnp.int8(TileType.CORRIDOR), terrain[nr, nc])))
        )
        cost = jnp.where(
            is_closed_door, jnp.int32(2),
            jnp.where(is_wall, jnp.int32(2),
            jnp.where(is_tree, jnp.int32(2),
            jnp.where(is_stone, jnp.int32(1), jnp.int32(1))))
        )

        do_write = still_going & carve
        # ── Flattened lax.cond: select between carved tile and original.
        write_tile = jnp.where(do_write, new_tile, terrain[nr, nc])
        new_terrain = terrain.at[nr, nc].set(write_tile)

        new_remaining = jnp.where(still_going, remaining - cost, remaining)
        new_pos = jnp.where(still_going[..., None],
                            jnp.array([nr, nc], dtype=jnp.int16),
                            pos)
        return (new_terrain, new_pos, new_remaining, stopped), None

    init = (
        state.terrain,
        state.player_pos.astype(jnp.int16),
        digdepth0,
        jnp.bool_(False),
    )
    (normal_terrain, _, _, _), _ = lax.scan(
        _dig_step,
        init,
        jnp.arange(26, dtype=jnp.int32),
    )

    # ---------- terrain selector: down → up → normal — flattened ----------
    is_down = dir_idx == jnp.int32(8)
    is_up   = dir_idx == jnp.int32(9)
    # First the inner cond: is_up ? up_dig : normal.
    not_down_terrain = jnp.where(is_up, up_dig_terrain, normal_terrain)
    # Then the outer cond: is_down ? set_hole : not_down_terrain.
    new_terrain = jnp.where(is_down, set_hole_terrain, not_down_terrain)

    # Up-dig falling rock damage — RNG draw + optional player_hp update.
    rng, sub = jax.random.split(rng)
    rock_dmg = jax.random.randint(sub, (), 1, 7, dtype=jnp.int32)
    if hasattr(state, "player_hp"):
        new_hp = jnp.where(is_up, state.player_hp - rock_dmg, state.player_hp)
        return state.replace(terrain=new_terrain, player_hp=new_hp), rng

    return state.replace(terrain=new_terrain), rng


# ---------------------------------------------------------------------------
# Brax dispatch table — wraps each Brax / unchanged effect to common
# (state, rng, direction) → (state, rng) signature.
# ---------------------------------------------------------------------------

def _b_light(s, r, d):           return _effect_light(s, r)
def _b_nothing(s, r, d):         return _effect_nothing(s, r, d)
def _b_secret_door(s, r, d):     return _effect_secret_door_detection(s, r)
def _b_opening(s, r, d):         return _effect_opening_brax(s, r, d)
def _b_locking(s, r, d):         return _effect_locking_brax(s, r, d)
def _b_probing(s, r, d):         return _effect_probing(s, r, d)
def _b_magic_missile(s, r, d):   return _effect_magic_missile(s, r, d)
def _b_striking(s, r, d):        return _effect_striking_brax(s, r, d)
def _b_slow(s, r, d):            return _effect_slow_monster(s, r, d)
def _b_speed(s, r, d):           return _effect_speed_monster(s, r, d)
def _b_cancellation(s, r, d):    return _effect_cancellation(s, r, d)
def _b_polymorph(s, r, d):       return _effect_polymorph_brax(s, r, d)
def _b_teleport(s, r, d):        return _effect_teleportation(s, r, d)
def _b_death(s, r, d):           return _effect_death_brax(s, r, d)
def _b_sleep(s, r, d):           return _effect_sleep(s, r, d)
def _b_cold(s, r, d):            return _effect_cold_brax(s, r, d)
def _b_fire(s, r, d):            return _effect_fire(s, r, d)
def _b_lightning(s, r, d):       return _effect_lightning(s, r, d)
def _b_digging(s, r, d):         return _effect_digging_brax(s, r, d)
def _b_enlightenment(s, r, d):   return _effect_enlightenment(s, r)
def _b_create_monster(s, r, d):  return _effect_create_monster(s, r)
def _b_wishing(s, r, d):         return _effect_wishing(s, r)
def _b_stasis(s, r, d):          return _effect_stasis(s, r, d)
def _b_make_invisible(s, r, d):  return _effect_make_invisible(s, r, d)
def _b_undead_turning(s, r, d):  return _effect_undead_turning(s, r, d)
def _b_draining(s, r, d):        return _effect_draining(s, r, d)
def _b_acid(s, r, d):            return _effect_acid(s, r, d)
def _b_poison_gas(s, r, d):      return _effect_poison_gas(s, r, d)


_EFFECT_BRANCHES_BRAX = (
    _b_light,            # 0  LIGHT
    _b_nothing,          # 1  NOTHING
    _b_secret_door,      # 2  SECRET_DOOR_DETECTION
    _b_opening,          # 3  OPENING
    _b_locking,          # 4  LOCKING
    _b_probing,          # 5  PROBING
    _b_magic_missile,    # 6  MAGIC_MISSILE
    _b_striking,         # 7  STRIKING
    _b_slow,             # 8  SLOW_MONSTER
    _b_speed,            # 9  SPEED_MONSTER
    _b_cancellation,     # 10 CANCELLATION
    _b_polymorph,        # 11 POLYMORPH
    _b_teleport,         # 12 TELEPORTATION
    _b_death,            # 13 DEATH
    _b_sleep,            # 14 SLEEP
    _b_cold,             # 15 COLD
    _b_fire,             # 16 FIRE
    _b_lightning,        # 17 LIGHTNING
    _b_digging,          # 18 DIGGING
    _b_enlightenment,    # 19 ENLIGHTENMENT
    _b_create_monster,   # 20 CREATE_MONSTER
    _b_wishing,          # 21 WISHING
    _b_stasis,           # 22 STASIS
    _b_make_invisible,   # 23 MAKE_INVISIBLE
    _b_undead_turning,   # 24 UNDEAD_TURNING
    _b_draining,         # 25 DRAINING
    _b_acid,             # 26 ACID
    _b_poison_gas,       # 27 POISON_GAS
)

assert len(_EFFECT_BRANCHES_BRAX) == N_WANDS, (
    f"_EFFECT_BRANCHES_BRAX has {len(_EFFECT_BRANCHES_BRAX)} entries; "
    f"expected {N_WANDS}"
)


# ---------------------------------------------------------------------------
# zap_wand_brax — flattens the 28-way ``lax.switch``.
# ---------------------------------------------------------------------------

def zap_wand_brax(
    state: WandState,
    rng: jax.Array,
    slot_idx: jax.Array,
    direction: jax.Array,
) -> WandState:
    """Brax-style ``zap_wand``.

    Replaces the 28-way ``jax.lax.switch`` over wand effect with a
    fully unrolled fan-out: every branch handler runs on the same
    (state, rng, direction) and the final state / rng is selected by
    ``jnp.where`` (per-leaf) against a one-hot mask derived from the
    effect index.

    Byte-parity rules (per task brief):
      1. RNG draw order: each branch consumes its own ``rng`` (the
         caller's), so the *selected* branch's downstream rng output
         is byte-identical to the original ``lax.switch`` selection.
      2. Mutations: ``jax.tree.map(jnp.where, ...)`` selects between
         branch outputs that share dtype / shape with the original.
      3. State pytree shape: unchanged — every branch returns a
         WandState with the same struct.
    """
    effect_idx = state.inventory.items.type_id[slot_idx].astype(jnp.int32)
    effect_idx = jnp.clip(effect_idx, 0, N_WANDS - 1)

    # Decrement charges before applying effect.
    new_inv = _decrement_charges(state.inventory, slot_idx)
    state = state.replace(inventory=new_inv)

    # Compute every branch.  Branch ``k`` receives the *same* (state,
    # rng, direction) the original ``lax.switch`` would have given it,
    # so RNG-draw order inside the selected branch is identical.
    branch_states = []
    branch_rngs = []
    for fn in _EFFECT_BRANCHES_BRAX:
        sb, rb = fn(state, rng, direction)
        branch_states.append(sb)
        branch_rngs.append(rb)

    # Fold-mask: start with branch 0, then for each k>=1 select via where
    # mask = (effect_idx == k).  This keeps the comparison count linear
    # in N_WANDS but emits a tree of selects in HLO that is structurally
    # equivalent to lax.switch flattening — without lax.switch.
    out_state = branch_states[0]
    out_rng = branch_rngs[0]
    for k in range(1, N_WANDS):
        mask = effect_idx == jnp.int32(k)
        out_state = _where_tree(mask, branch_states[k], out_state)
        out_rng = jnp.where(mask, branch_rngs[k], out_rng)

    return out_state


# ---------------------------------------------------------------------------
# handle_zap_brax — pure-Python branching on caller args + jnp.where.
# No lax.cond / lax.switch in the original wrapper body; the only
# dispatch lives inside zap_wand which is replaced by zap_wand_brax.
# ---------------------------------------------------------------------------

def handle_zap_brax(
    state: WandState,
    rng: jax.Array,
    chosen_direction=None,
    chosen_slot=None,
) -> WandState:
    """Brax-style ``handle_zap`` — routes to ``zap_wand_brax``.

    No JIT control flow added or removed: the original wrapper used
    Python ``if`` for ``None`` defaults (host-side, compiled-away) and
    ``jnp.where`` for the chosen-slot fallback.
    """
    categories = state.inventory.items.category
    is_wand    = categories == jnp.int8(ITEM_CATEGORY_WAND)
    fallback_slot = jnp.argmax(is_wand).astype(jnp.int32)

    if chosen_slot is None:
        slot_idx = fallback_slot
    else:
        safe_chosen = jnp.clip(
            chosen_slot.astype(jnp.int32), 0, is_wand.shape[0] - 1
        )
        chosen_is_wand = is_wand[safe_chosen]
        slot_idx = jnp.where(chosen_is_wand, safe_chosen, fallback_slot).astype(
            jnp.int32
        )

    if chosen_direction is None:
        direction = jnp.int32(2)
    else:
        direction = jnp.int32(chosen_direction)

    return zap_wand_brax(state, rng, slot_idx, direction)


# ---------------------------------------------------------------------------
# zap_polymorph_at_self_brax — pure delegate (no cond/switch in original).
# ---------------------------------------------------------------------------

def zap_polymorph_at_self_brax(state, rng: jax.Array, slot_idx: jax.Array):
    """Brax alias of ``zap_polymorph_at_self``.

    The original body has zero ``lax.cond`` / ``lax.switch``; nothing
    to flatten.  Re-exported as ``*_brax`` so a future rewrite of the
    polymorph_player internals can be picked up by editing only this
    module's imports.
    """
    from Nethax.nethax.subsystems.polymorph import (
        polymorph_player,
        choose_random_polymorph_form,
    )
    inv = state.inventory
    old_charges = inv.items.charges
    new_charges = old_charges.at[slot_idx].add(jnp.int8(-1))
    new_charges = jnp.maximum(new_charges, jnp.int8(0))
    new_inv = inv.replace(items=inv.items.replace(charges=new_charges))
    state = state.replace(inventory=new_inv)

    rng, sub = jax.random.split(rng)
    form = choose_random_polymorph_form(state, sub)
    rng, sub2 = jax.random.split(rng)
    return polymorph_player(state, sub2, form, controlled=False)


# ===========================================================================
# items_spellbooks — Brax rewrites
# ===========================================================================

def _cursed_book_backfire_brax(
    state, rng: jax.Array, slot_idx: int, book_level: int,
):
    """Brax-flattened ``_cursed_book_backfire``.

    Replaces the 8-way ``jax.lax.switch`` over ``rn2(lev)`` with a
    fully unrolled fan-out + pytree ``jnp.where`` mask select.

    RNG parity
    ----------
    The original splits ``rng`` into 9 disjoint sub-keys up front (one
    branch-selector + one per branch).  Every branch is a pure
    function of its dedicated sub-key, so computing all branches and
    selecting via mask is **byte-identical** to the original switch:
    no extra splits are introduced, and the selected branch consumes
    the exact same key the switch would have routed to it.
    """
    from Nethax.nethax.rng import rnd, rn1, rn2
    from Nethax.nethax.subsystems.status_effects import TimedStatus, _roll_rn1

    (rng_branch, sub_tele, sub_blind, sub_gold_amt, sub_gold_pos,
     sub_confuse, sub_poison, sub_dmg, sub_curse) = jax.random.split(rng, 9)

    lev = max(1, int(book_level))

    # ---- branch 0: teleport ----
    def b0_teleport(s):
        from Nethax.nethax.constants.tiles import TileType
        br = s.dungeon.current_branch.astype(jnp.int32)
        lv = s.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
        level_tiles = s.terrain[br, lv]
        floor_mask = level_tiles == jnp.int8(int(TileType.FLOOR))
        flat_mask = floor_mask.reshape(-1).astype(jnp.float32)
        total = jnp.sum(flat_mask)
        H, W = level_tiles.shape
        has_floor = total > 0
        probs = jnp.where(
            has_floor,
            flat_mask / jnp.maximum(total, jnp.float32(1.0)),
            jnp.ones((H * W,), dtype=jnp.float32) / jnp.float32(H * W),
        )
        flat_idx = jax.random.choice(sub_tele, H * W, p=probs).astype(jnp.int32)
        new_row = (flat_idx // W).astype(jnp.int16)
        new_col = (flat_idx % W).astype(jnp.int16)
        new_pos = jnp.stack([new_row, new_col])
        out_pos = jnp.where(has_floor, new_pos, s.player_pos)
        return s.replace(player_pos=out_pos)

    # ---- branch 1: aggravate ----
    def b1_aggravate(s):
        from Nethax.nethax.subsystems.monster_ai import wake_monsters_near
        return wake_monsters_near(s, s.player_pos, radius=999, petcall=False)

    # ---- branch 2: blind ----
    def b2_blind(s):
        ts = s.status.timed_statuses
        add = _roll_rn1(sub_blind, 100, 250)
        new_blind = ts[int(TimedStatus.BLIND)] + add
        new_ts = ts.at[int(TimedStatus.BLIND)].set(new_blind)
        return s.replace(status=s.status.replace(timed_statuses=new_ts))

    # ---- branch 3: take gold ----
    def b3_take_gold(s):
        gold = s.player_gold.astype(jnp.int32)
        bracket_n = jnp.where(
            gold < jnp.int32(50), jnp.int32(1),
            jnp.where(gold < jnp.int32(100), gold - jnp.int32(25) + jnp.int32(1),
            jnp.where(gold < jnp.int32(500), gold - jnp.int32(50) + jnp.int32(1),
            jnp.where(gold < jnp.int32(1000), gold - jnp.int32(100) + jnp.int32(1),
            jnp.where(gold < jnp.int32(5000), gold - jnp.int32(500) + jnp.int32(1),
            jnp.where(gold < jnp.int32(10000), gold - jnp.int32(1000) + jnp.int32(1),
                                                gold - jnp.int32(5000) + jnp.int32(1)))))),
        )
        bracket_x = jnp.where(
            gold < jnp.int32(50), jnp.int32(0),
            jnp.where(gold < jnp.int32(100), jnp.int32(25),
            jnp.where(gold < jnp.int32(500), jnp.int32(50),
            jnp.where(gold < jnp.int32(1000), jnp.int32(100),
            jnp.where(gold < jnp.int32(5000), jnp.int32(500),
            jnp.where(gold < jnp.int32(10000), jnp.int32(1000),
                                                jnp.int32(5000)))))),
        )
        safe_n = jnp.maximum(bracket_n, jnp.int32(1))
        rn2_roll = jax.random.randint(sub_gold_amt, (), 0, safe_n, dtype=jnp.int32)
        rn1_result = (bracket_x + rn2_roll).astype(jnp.int32)
        stolen = jnp.where(gold < jnp.int32(50), gold, rn1_result)
        stolen = jnp.minimum(stolen, gold)
        new_gold = jnp.maximum(gold - stolen, jnp.int32(0)).astype(jnp.int32)
        return s.replace(player_gold=new_gold)

    # ---- branch 4: confuse ----
    def b4_confuse(s):
        ts = s.status.timed_statuses
        add = _roll_rn1(sub_confuse, 7, 16)
        new_conf = ts[int(TimedStatus.CONFUSION)] + add
        new_ts = ts.at[int(TimedStatus.CONFUSION)].set(new_conf)
        return s.replace(status=s.status.replace(timed_statuses=new_ts))

    # ---- branch 5: poison ----
    def b5_poison(s):
        ts = s.status.timed_statuses
        new_attr = jnp.int32(10)
        new_ts = ts.at[int(TimedStatus.ATTRIBUTE_AWAY)].set(new_attr)
        drain = rnd(sub_poison, 2)
        new_str = jnp.maximum(
            s.player_str - drain.astype(jnp.int16), jnp.int16(3)
        ).astype(jnp.int16)
        return s.replace(
            player_str=new_str,
            status=s.status.replace(timed_statuses=new_ts),
        )

    # ---- branch 6: explode ----
    def b6_explode(s):
        dmg = jnp.int32(2) * rnd(sub_dmg, 10) + jnp.int32(5)
        gated_dmg = jnp.where(_PLAYER_HAS_ANTIMAGIC, jnp.int32(0), dmg)
        new_hp = jnp.maximum(s.player_hp - gated_dmg, jnp.int32(1))
        new_qty = s.inventory.items.quantity.at[slot_idx].set(jnp.int16(0))
        new_items = s.inventory.items.replace(quantity=new_qty)
        new_inv = s.inventory.replace(items=new_items)
        return s.replace(player_hp=new_hp, inventory=new_inv)

    # ---- branch 7: default (rndcurse — vanilla-unreachable) ----
    def b_default(s):
        from Nethax.nethax.subsystems.items_scrolls import rndcurse
        return rndcurse(s, sub_curse)

    branches = (
        b0_teleport, b1_aggravate, b2_blind, b3_take_gold,
        b4_confuse,  b5_poison,    b6_explode, b_default,
    )

    rn2_lev = rn2(rng_branch, lev).astype(jnp.int32)

    # ── Flattened lax.switch (8-way): compute every branch, mask-select.
    branch_outs = [fn(state) for fn in branches]
    out_state = branch_outs[0]
    for k in range(1, len(branch_outs)):
        mask = rn2_lev == jnp.int32(k)
        out_state = _where_tree(mask, branch_outs[k], out_state)
    return out_state


# ---------------------------------------------------------------------------
# read_spellbook_brax — Python branching on host ints + Brax cursed_book.
# The original has no JIT-time ``lax.cond`` / ``lax.switch`` outside of
# the cursed_book path which is replaced above.
# ---------------------------------------------------------------------------

def read_spellbook_brax(state, rng: jax.Array, slot_idx: int):
    """Brax variant of ``read_spellbook``.

    Routes the cursed-book backfire through ``_cursed_book_backfire_brax``
    (no ``lax.switch``).  Everything else is host-only Python control
    flow (``int(...)`` extractions + plain ``if``), so it is unchanged
    structurally — the public alias exists so dispatchers can swap
    files in one import line.
    """
    from Nethax.nethax.subsystems.magic import KEEN, N_SPELLS, _SPELL_LEVELS
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    is_blind   = state.status.timed_statuses[int(TimedStatus.BLIND)]   > jnp.int32(0)
    is_stunned = state.status.timed_statuses[int(TimedStatus.STUNNED)] > jnp.int32(0)
    if bool(is_blind) or bool(is_stunned):
        return state

    spell_id   = int(state.inventory.items.type_id[slot_idx])
    buc_status = int(state.inventory.items.buc_status[slot_idx])

    if spell_id == BLANK_SPELL_ID or spell_id < 0 or spell_id >= N_SPELLS:
        return state

    book_level = int(_SPELL_LEVELS[spell_id])

    if buc_status == _BUC_CURSED:
        return _cursed_book_backfire_brax(state, rng, slot_idx, book_level)

    if buc_status != _BUC_BLESSED:
        player_int = int(state.player_int)
        player_xl  = int(state.player_xl)
        role_id    = int(state.player_role)

        read_ability = player_int + 4 + player_xl // 2 - 2 * book_level
        if role_id == _ROLE_WIZARD:
            read_ability += _WIZARD_STUDY_BONUS
        read_ability = max(0, min(read_ability, 20))

        rng, sub = jax.random.split(rng)
        roll = int(jax.random.randint(sub, (), 1, 21))

        if roll > read_ability:
            return state

    magic = state.magic
    new_known = magic.spell_known.at[spell_id].set(True)
    new_mem   = magic.spell_memory.at[spell_id].set(jnp.int32(KEEN + 1))
    magic = magic.replace(spell_known=new_known, spell_memory=new_mem)

    if int(magic.spell_letter[spell_id]) == -1:
        magic = _assign_letter(magic, spell_id)

    oc_delay = int(_SPELL_OC_DELAYS[spell_id]) if 0 <= spell_id < len(_SPELL_OC_DELAYS) else 1
    delay_turns = study_book_delay(book_level, oc_delay)
    new_timestep = (state.timestep.astype(jnp.int32) + jnp.int32(delay_turns))

    return state.replace(magic=magic, timestep=new_timestep)


def handle_read_spellbook_brax(state, rng: jax.Array, slot_idx: int):
    """Brax variant of ``handle_read_spellbook`` — pure delegate."""
    new_state = read_spellbook_brax(state, rng, slot_idx)
    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated
    return mark_violated(new_state, int(Conduct.ILLITERATE))


# ---------------------------------------------------------------------------
# Public exports.
# ---------------------------------------------------------------------------

__all__ = [
    # items_wands public entry points (Brax variants).
    "cast_ray_brax",
    "zap_wand_brax",
    "handle_zap_brax",
    "zap_polymorph_at_self_brax",
    # items_spellbooks public entry points (Brax variants).
    "study_book_delay_brax",
    "study_success_chance_brax",
    "read_spellbook_brax",
    "handle_read_spellbook_brax",
]
