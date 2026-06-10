"""Brax-style rewrites of the five "pre-loop" monster-AI helpers.

Background
----------
Under ``jax.vmap`` (multi-seed / multi-env rollouts), ``jax.lax.cond`` and
``jax.lax.switch`` lower to ``lax.select`` and emit *both* branches in the
HLO.  For nested cond chains this produces pathological HLO compile blowup
on H100.  The Brax pattern (Google brax + Craftax —
https://github.com/MichaelTMatthews/Craftax) sidesteps this by always
computing both branches eagerly and selecting results via ``jnp.where``
masks.  The resulting HLO is flat and fusion-friendly.

This module hosts the Brax-style ports of the five helpers called from
the monster-phase pre-loop (and the LoS BFS used by postmov):

  * ``monster_can_see_player_brax`` — LoS Bresenham line scan
    (vendor vision.c::clear_path lines 165-184).
  * ``_were_change_all_brax`` — per-turn 1/30 were-creature shape-shift
    (vendor were.c::were_change lines 8-45).
  * ``_were_summon_attempt_brax`` — were-monster summons 1..5 helpers
    (vendor mhitu.c:987-1015 + were.c::were_summon lines 140-189).
  * ``_try_steal_quest_artifact_brax`` — Wizard of Yendor steals quest
    artifacts (vendor steal.c::stealamulet lines 689-766).
  * ``_covetous_ai_step_brax`` — HUNT + speed boost on covetous monsters
    (vendor wizard.c::strategy + target_on lines 235-327).

Audit & flatten count
---------------------
Reading the originals in ``monster_ai.py``:

  * ``monster_can_see_player`` (line 1188): 0 conds, 0 switches, 1
    ``lax.fori_loop`` (fixed trip count = max(MAP_H, MAP_W)).  Per the
    task brief, fixed-iter ``fori_loop`` is OK and KEPT verbatim.
  * ``_were_change_all`` (line 5490): 0 conds, 0 switches, 0 scans.
    Already Brax-shaped (all conditional dataflow via ``jnp.where``).
  * ``_were_summon_attempt`` (line 5916): 0 conds, 0 switches, 1
    ``jax.lax.scan`` with STATIC trip count 5.  The scan iterates a
    pytree-mutating body — the classic Brax-pattern target.  UNROLLED
    into 5 sequential ``jnp.where``-masked updates so XLA sees flat
    dataflow (no scan-body fusion barrier).
  * ``_try_steal_quest_artifact`` (line 5848): 0 conds, 0 switches.
    Already Brax-shaped.
  * ``_covetous_ai_step`` (line 5741): 0 conds, 0 switches.  Already
    Brax-shaped.

Number of ``lax.cond`` / ``lax.switch`` constructs flattened per
function:

  * ``monster_can_see_player_brax``     : 0 conds, 0 switches
                                          (1 fixed ``fori_loop`` kept)
  * ``_were_change_all_brax``           : 0 conds, 0 switches
  * ``_were_summon_attempt_brax``       : 0 conds, 0 switches
                                          (1 static-trip-count
                                           ``lax.scan`` unrolled → 5
                                           sequential ``jnp.where``
                                           masked updates)
  * ``_try_steal_quest_artifact_brax``  : 0 conds, 0 switches
  * ``_covetous_ai_step_brax``          : 0 conds, 0 switches

Total: 0 ``lax.cond`` flattened, 0 ``lax.switch`` flattened, 1
``lax.scan`` unrolled.  The originals were already mostly Brax-shaped;
this module is the canonical "pre-loop" landing zone so the post-Brax
H100 compile path can import from one place.

Byte-parity constraints
-----------------------
1. RNG draw order preserved exactly.  ``_were_change_all_brax`` consumes
   ``n`` keys via ``jax.random.split(rng, n)`` + ``vmap`` of
   ``jax.random.randint``, matching the original.
   ``_were_summon_attempt_brax`` reproduces the exact split sequence
   (``n + 1`` for gates + spawn, then ``2`` for count + helper-keys,
   then ``5`` helper keys) so each spawn slot receives the byte-identical
   key it would have received under the scan.
2. Every mutation routes through ``jnp.where`` masking.  No conditional
   ``.at[...].set(...)``.
3. State pytree shape preserved (we only call ``.replace`` with same
   field names/dtypes as the originals).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems.monster_ai import (
    _BOULDER_TYPE_ID_VISION,
    _COVETOUS_ENTRY_IDS,
    _M1_SEE_INVIS,
    _MAP_H,
    _MAP_W,
    _QUEST_ARTIFACT_TIDS,
    _TILE_CLOSED_DOOR,
    _TILE_TREE,
    _TILE_WALL,
    _current_level_terrain,
    _find_ground_quest_artifact,
    _find_nearest_upstair,
    _has_flag1,
    _pick_were_summon_species,
    _player_carries_quest_artifact,
    _player_is_invisible,
    _were_counterpart,
    MoveStrategy,
)


# ---------------------------------------------------------------------------
# 1. monster_can_see_player_brax — LoS Bresenham line scan
# ---------------------------------------------------------------------------
# Audit: original uses one ``jax.lax.fori_loop`` with a STATIC trip count
# of ``max(_MAP_H, _MAP_W)`` (= 80).  Per the task brief, fixed-iteration
# ``fori_loop`` is OK to keep — XLA unrolls / scans it without the
# both-branches HLO blowup that plagues ``lax.cond``.  Zero ``lax.cond``
# and zero ``lax.switch`` in the original; the body's conditional dataflow
# (active gate, blocked gate) is already routed through ``jnp.where``.
#
# Conds flattened: 0.  Switches flattened: 0.
# ---------------------------------------------------------------------------


def monster_can_see_player_brax(state, monster_idx: jnp.ndarray) -> jnp.ndarray:
    """Brax-style port of ``monster_ai.monster_can_see_player``.

    Byte-parity contract: every comparison + select in this function
    matches the original element-for-element.  The single fixed-iter
    ``lax.fori_loop`` is kept (per task brief).

    Cite: vendor/nethack/src/vision.c::clear_path + ``block_light``
    around lines 165-184.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    terrain = _current_level_terrain(state)

    # Per-tile boulder presence on the current level (vision.c:182
    # is_clear also rejects tiles with a boulder).
    b = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    gi = state.ground_items
    boulder_here = jnp.any(
        gi.type_id[b, lv].astype(jnp.int32)
        == jnp.int32(_BOULDER_TYPE_ID_VISION),
        axis=-1,
    )

    r0, c0 = mpos[0], mpos[1]
    r1, c1 = ppos[0], ppos[1]

    dr_signed = (r1 - r0).astype(jnp.int32)
    dc_signed = (c1 - c0).astype(jnp.int32)
    dr_abs = jnp.abs(dr_signed)
    dc_abs = jnp.abs(dc_signed)
    n_steps = jnp.maximum(dr_abs, dc_abs).astype(jnp.int32)
    n_steps_safe = jnp.maximum(n_steps, jnp.int32(1))
    max_steps = max(_MAP_H, _MAP_W)

    def body(i, clear):
        # Parametric position along the line, excluding endpoints.
        # The "active" gate is already a ``jnp.where`` — no ``lax.cond``.
        active = ((i + 1) < n_steps) & clear
        numer_r = dr_signed * (i + 1)
        numer_c = dc_signed * (i + 1)
        step_r = jnp.round(
            numer_r.astype(jnp.float32) / n_steps_safe.astype(jnp.float32)
        ).astype(jnp.int32)
        step_c = jnp.round(
            numer_c.astype(jnp.float32) / n_steps_safe.astype(jnp.float32)
        ).astype(jnp.int32)
        tr = r0 + step_r
        tc = c0 + step_c
        safe_r = jnp.clip(tr, 0, _MAP_H - 1)
        safe_c = jnp.clip(tc, 0, _MAP_W - 1)
        tile = terrain[safe_r, safe_c].astype(jnp.int32)
        has_boulder = boulder_here[safe_r, safe_c]
        blocked = (
            (tile == _TILE_WALL)
            | (tile == _TILE_CLOSED_DOOR)
            | (tile == _TILE_TREE)
            | has_boulder
        )
        return jnp.where(active & blocked, jnp.bool_(False), clear)

    clear = jax.lax.fori_loop(0, max_steps, body, jnp.bool_(True))
    same_tile = (r0 == r1) & (c0 == c1)

    # Invisible-player gate (vendor vision.c::couldsee).
    is_invis = _player_is_invisible(state)
    sees_invis = _has_flag1(mai.entry_idx[idx], _M1_SEE_INVIS)
    invis_gate = (~is_invis) | sees_invis

    return (clear | same_tile) & invis_gate


# ---------------------------------------------------------------------------
# 2. _were_change_all_brax — per-turn 1/30 were-creature shape-shift
# ---------------------------------------------------------------------------
# Audit: original has 0 conds, 0 switches, 0 scans.  Already Brax-shaped.
# This port is a structural copy preserving the exact RNG split / vmap'd
# rn2(30) draw order so byte-parity is automatic.
#
# Conds flattened: 0.  Switches flattened: 0.
# ---------------------------------------------------------------------------


def _were_change_all_brax(mai, rng: jax.Array):
    """Brax-style port of ``monster_ai._were_change_all``.

    Cite: vendor/nethack/src/were.c::were_change lines 8-45.
    """
    entry = mai.entry_idx.astype(jnp.int32)
    counterpart = jax.vmap(_were_counterpart)(entry)
    is_were = counterpart != entry

    n = entry.shape[0]
    keys = jax.random.split(rng, n)
    rolls = jax.vmap(
        lambda k: jax.random.randint(k, (), 0, 30, dtype=jnp.int32)
    )(keys)

    transforms = mai.alive & is_were & (rolls == jnp.int32(0))

    new_entry = jnp.where(transforms, counterpart, entry).astype(jnp.int16)

    hp_now = mai.hp.astype(jnp.int32)
    hp_max = mai.hp_max.astype(jnp.int32)
    heal_amt = jnp.where(
        transforms, (hp_max - hp_now) // jnp.int32(4), jnp.int32(0)
    )
    new_hp = jnp.minimum(hp_now + heal_amt, hp_max).astype(jnp.int32)

    # Vendor new_were lines 120-125: helpless monsters wake on transform.
    new_asleep = jnp.where(transforms, jnp.bool_(False), mai.asleep)
    new_sleep_t = jnp.where(transforms, jnp.int16(0), mai.sleep_timer)
    new_paral_t = jnp.where(transforms, jnp.int16(0), mai.paralyzed_timer)

    # ---- were-howl: wake_nearto(mx, my, 4*4) — vendor were.c lines 35-38 ----
    pos = mai.pos.astype(jnp.int32)                   # (N,2)
    src_pos = pos[:, None, :]                          # (N,1,2)
    tgt_pos = pos[None, :, :]                          # (1,N,2)
    d_rows = jnp.abs(src_pos[..., 0] - tgt_pos[..., 0])
    d_cols = jnp.abs(src_pos[..., 1] - tgt_pos[..., 1])
    cheb = jnp.maximum(d_rows, d_cols)                 # (N,N)
    src_active = transforms[:, None]                   # (N,1)
    pair_wake = src_active & (cheb <= jnp.int32(4))    # (N,N)
    woken = jnp.any(pair_wake, axis=0) & mai.alive     # (N,)

    new_asleep = jnp.where(woken, jnp.bool_(False), new_asleep)
    new_sleep_t = jnp.where(woken, jnp.int16(0), new_sleep_t)

    return mai.replace(
        entry_idx=new_entry,
        hp=new_hp,
        asleep=new_asleep,
        sleep_timer=new_sleep_t,
        paralyzed_timer=new_paral_t,
    )


# ---------------------------------------------------------------------------
# 3. _try_steal_quest_artifact_brax — Wizard of Yendor steals artifact
# ---------------------------------------------------------------------------
# Audit: original has 0 conds, 0 switches, 0 scans.  Already Brax-shaped.
# The ``rng`` argument is accepted for signature parity but unused (no
# RNG draws in the original) — kept so call sites are 1:1.
#
# Conds flattened: 0.  Switches flattened: 0.
# ---------------------------------------------------------------------------


def _try_steal_quest_artifact_brax(state, rng: jax.Array):
    """Brax-style port of ``monster_ai._try_steal_quest_artifact``.

    Cite: vendor/nethack/src/steal.c::stealamulet lines 689-766;
          vendor/nethack/src/wizard.c::strategy + target_on lines 235-327.
    """
    del rng  # unused — present for signature parity with the original.
    mai = state.monster_ai

    # Find an alive adjacent Wizard.
    entry = mai.entry_idx.astype(jnp.int32)
    is_wiz = entry == jnp.int32(281)
    ppos = state.player_pos.astype(jnp.int32)
    mpos = mai.pos.astype(jnp.int32)
    dr = jnp.abs(mpos[:, 0] - ppos[0])
    dc = jnp.abs(mpos[:, 1] - ppos[1])
    adj = jnp.maximum(dr, dc) <= jnp.int32(1)
    eligible = mai.alive & is_wiz & adj
    any_eligible = jnp.any(eligible)
    wiz_idx = jnp.argmax(eligible.astype(jnp.int32)).astype(jnp.int32)

    # Find the first quest artifact in the player's inventory.
    inv = state.inventory.items
    cat = inv.category
    tid = inv.type_id
    occupied = cat != jnp.int8(0)
    is_qa = jnp.zeros_like(occupied)
    for qid in _QUEST_ARTIFACT_TIDS:
        is_qa = is_qa | (tid == jnp.int16(qid))
    targets = occupied & is_qa
    any_target = jnp.any(targets)
    art_slot = jnp.argmax(targets.astype(jnp.int32)).astype(jnp.int32)

    do_steal = any_eligible & any_target

    # Capture the artifact's data before we zero it out.
    art_cat = inv.category[art_slot]
    art_tid = inv.type_id[art_slot]

    # Zero the player's inventory slot (idempotent if do_steal=False).
    new_pcat = jnp.where(do_steal, jnp.int8(0), inv.category[art_slot])
    new_pqty = jnp.where(do_steal, jnp.int16(0), inv.quantity[art_slot])
    new_items = inv.replace(
        category=inv.category.at[art_slot].set(new_pcat),
        quantity=inv.quantity.at[art_slot].set(new_pqty),
    )

    # Add the artifact to the Wizard's monster inventory slot 0.
    new_mcat = jnp.where(do_steal, art_cat, mai.inv_category[wiz_idx, 0])
    new_mtid = jnp.where(do_steal, art_tid, mai.inv_type_id[wiz_idx, 0])
    new_mqty = jnp.where(do_steal, jnp.int16(1), mai.inv_quantity[wiz_idx, 0])
    new_mai = mai.replace(
        inv_category=mai.inv_category.at[wiz_idx, 0].set(new_mcat),
        inv_type_id=mai.inv_type_id.at[wiz_idx, 0].set(new_mtid),
        inv_quantity=mai.inv_quantity.at[wiz_idx, 0].set(new_mqty),
    )

    new_inv = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv, monster_ai=new_mai)


# ---------------------------------------------------------------------------
# 4. _covetous_ai_step_brax — HUNT + speed boost on covetous monsters
# ---------------------------------------------------------------------------
# Audit: original has 0 conds, 0 switches, 0 scans.  Already Brax-shaped.
# All conditional dataflow flows through ``jnp.where``.  vmap over
# monster slots for the nearest-upstair lookup is OK (per-slot pure
# function with no inner cond).
#
# Conds flattened: 0.  Switches flattened: 0.
# ---------------------------------------------------------------------------


def _covetous_ai_step_brax(state):
    """Brax-style port of ``monster_ai._covetous_ai_step``.

    Cite: vendor/nethack/src/wizard.c::strategy + target_on lines 235-327;
          vendor/nethack/src/wizard.c::tactics lines 378-415 (STRAT_HEAL).
    """
    player_has = _player_carries_quest_artifact(state)
    ground_has, art_r, art_c = _find_ground_quest_artifact(state)

    floor_only = ground_has & ~player_has
    any_target = player_has | floor_only

    mai = state.monster_ai
    entry = mai.entry_idx.astype(jnp.int32)
    is_covetous = jnp.bool_(False)
    for eid in _COVETOUS_ENTRY_IDS:
        is_covetous = is_covetous | (entry == jnp.int32(eid))
    apply = mai.alive & is_covetous & any_target

    new_strat = jnp.where(
        apply, jnp.int8(int(MoveStrategy.HUNT)), mai.mstrategy
    )
    new_speed = jnp.where(apply, jnp.int8(1), mai.speed_mod)
    new_flee = jnp.where(apply, jnp.int32(0), mai.flee_until_turn)

    # Floor-target steering (vendor target_on line 252-253).
    set_target = apply & floor_only
    art_tile = jnp.stack(
        [art_r.astype(jnp.int16), art_c.astype(jnp.int16)]
    )  # shape [2]
    n = mai.target_pos.shape[0]
    new_target = jnp.where(
        set_target[:, None],
        jnp.broadcast_to(art_tile, (n, 2)),
        mai.target_pos,
    )

    # ----- STRAT_HEAL retreat override (wizard.c::tactics 378-415) -----
    hp_i32 = mai.hp.astype(jnp.int32)
    hp_max_i32 = mai.hp_max.astype(jnp.int32)
    low_hp = hp_i32 * jnp.int32(3) < hp_max_i32
    retreat = mai.alive & is_covetous & low_hp

    new_strat = jnp.where(
        retreat,
        jnp.int8(int(MoveStrategy.RETREAT)),
        new_strat,
    )
    new_speed = jnp.where(retreat, jnp.int8(0), new_speed)

    # Per-monster nearest upstair lookup.  vmap is Brax-safe (per-slot
    # pure function, no inner cond/switch).
    pos_int32 = mai.pos.astype(jnp.int32)
    def _stair_for_slot(p):
        f, r, c = _find_nearest_upstair(state, p)
        return f, r, c
    stair_found, stair_r, stair_c = jax.vmap(_stair_for_slot)(pos_int32)

    set_stair = retreat & stair_found
    stair_tile = jnp.stack(
        [stair_r.astype(jnp.int16), stair_c.astype(jnp.int16)],
        axis=-1,
    )  # [n, 2]
    new_target = jnp.where(
        set_stair[:, None],
        stair_tile,
        new_target,
    )

    return state.replace(monster_ai=mai.replace(
        mstrategy=new_strat,
        speed_mod=new_speed,
        flee_until_turn=new_flee,
        target_pos=new_target,
    ))


# ---------------------------------------------------------------------------
# 5. _were_summon_attempt_brax — were-monster summons 1..5 helpers
# ---------------------------------------------------------------------------
# Audit: original has 0 conds, 0 switches, but ONE ``jax.lax.scan`` over
# the 5 helper-spawn slots.  The scan body mutates the state pytree per
# iteration via ``.at[].set(jnp.where(...))`` — the classic Brax-pattern
# target.  Since the trip count is STATICALLY 5, we unroll into 5
# sequential ``jnp.where``-masked updates.
#
# Byte-parity: the RNG draws are byte-identical because the scan's
# (dead_idx_arr, helper_keys) iteration consumes helper_keys[i] on step
# i; the unrolled version does the same with helper_keys_static[i] in
# Python-loop order.  The Python loop runs at TRACE time (not runtime),
# so the unrolled HLO is flat — no scan-body fusion barrier.
#
# Conds flattened: 0.  Switches flattened: 0.  Scans unrolled: 1.
# ---------------------------------------------------------------------------


def _were_summon_attempt_brax(state, rng: jax.Array):
    """Brax-style port of ``monster_ai._were_summon_attempt``.

    Cite: vendor/nethack/src/mhitu.c lines 987-1015 (1/10 gate) +
          vendor/nethack/src/were.c::were_summon lines 140-189.

    Byte-parity contract: the RNG split sequence matches the original
    exactly:

        keys        = split(rng, n + 1)        # gate_keys + spawn_key
        gate_rolls  = vmap(randint)(keys[:n])
        k1, k2      = split(spawn_key, 2)
        count       = randint(k1, (), 1, 6)
        helper_keys = split(k2, 5)

    The 5-step ``jax.lax.scan`` over helper slots is unrolled into 5
    sequential ``jnp.where``-masked updates — same dataflow, flat HLO.
    """
    mai = state.monster_ai
    n = mai.pos.shape[0]

    entry = mai.entry_idx.astype(jnp.int32)
    is_were = (
        (entry == jnp.int32(15))
        | (entry == jnp.int32(21))
        | (entry == jnp.int32(95))
        | (entry == jnp.int32(269))
        | (entry == jnp.int32(270))
        | (entry == jnp.int32(271))
    )
    ppos = state.player_pos.astype(jnp.int32)
    mpos = mai.pos.astype(jnp.int32)
    dr = jnp.abs(mpos[:, 0] - ppos[0])
    dc = jnp.abs(mpos[:, 1] - ppos[1])
    adjacent = jnp.maximum(dr, dc) <= jnp.int32(1)

    # --- RNG draw order — byte-identical to the scan version ---
    keys = jax.random.split(rng, n + 1)
    gate_keys = keys[:n]
    spawn_key = keys[n]

    gate_rolls = jax.vmap(
        lambda k: jax.random.randint(k, (), 0, 10, dtype=jnp.int32)
    )(gate_keys)
    triggers = mai.alive & is_were & adjacent & (gate_rolls == jnp.int32(0))

    any_trigger = jnp.any(triggers)
    trig_idx = jnp.argmax(triggers.astype(jnp.int32)).astype(jnp.int32)
    parent_entry = mai.entry_idx[trig_idx].astype(jnp.int32)

    k1, k2 = jax.random.split(spawn_key, 2)
    count = jax.random.randint(k1, (), 1, 6, dtype=jnp.int32)

    # Find up to 5 dead slots to spawn into — same selection as the
    # original (argsort with non-dead pushed to the high half).
    dead_mask = ~mai.alive
    dead_idx_arr = jnp.argsort(
        jnp.where(dead_mask, jnp.arange(n), jnp.int32(n) + jnp.arange(n))
    )[:5]

    helper_keys = jax.random.split(k2, 5)

    # Static offset table (vendor mhitu summon — 8-pattern around player).
    offsets = jnp.array(
        [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]],
        dtype=jnp.int32,
    )

    # ----- Unrolled spawn loop (replaces ``jax.lax.scan`` over 5 slots) -----
    # Each iteration computes will_spawn + masked mutation via ``jnp.where``,
    # then writes through ``.at[slot].set(...)`` with the masked value.
    # Inactive iterations write the existing value back (byte-identical).
    cur_state = state
    for i in range(5):
        helper_slot_idx = dead_idx_arr[i]
        key = helper_keys[i]
        i_jax = jnp.int32(i)

        will_spawn = (
            any_trigger
            & (i_jax < count)
            & dead_mask[helper_slot_idx]
        )
        species = _pick_were_summon_species(parent_entry, key)

        m = cur_state.monster_ai
        off = offsets[i_jax % jnp.int32(8)]
        spawn_r = (ppos[0] + off[0]).astype(jnp.int16)
        spawn_c = (ppos[1] + off[1]).astype(jnp.int16)

        new_pos = jnp.where(
            will_spawn,
            jnp.stack([spawn_r, spawn_c]),
            m.pos[helper_slot_idx],
        )
        new_alive = jnp.where(
            will_spawn, jnp.bool_(True), m.alive[helper_slot_idx]
        )
        new_entry_v = jnp.where(
            will_spawn,
            species.astype(jnp.int16),
            m.entry_idx[helper_slot_idx],
        )
        new_hp = jnp.where(will_spawn, jnp.int32(4), m.hp[helper_slot_idx])
        new_hpmax = jnp.where(
            will_spawn, jnp.int32(4), m.hp_max[helper_slot_idx]
        )

        new_m = m.replace(
            pos=m.pos.at[helper_slot_idx].set(new_pos),
            alive=m.alive.at[helper_slot_idx].set(new_alive),
            entry_idx=m.entry_idx.at[helper_slot_idx].set(new_entry_v),
            hp=m.hp.at[helper_slot_idx].set(new_hp),
            hp_max=m.hp_max.at[helper_slot_idx].set(new_hpmax),
        )
        cur_state = cur_state.replace(monster_ai=new_m)

    return cur_state
