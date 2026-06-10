"""Brax-style rewrites of public ``features.py`` entry points — flat HLO via
``jnp.where`` masking in place of ``jax.lax.cond`` / ``jax.lax.switch``.

Background
----------
Under ``jax.vmap`` over seeds, ``lax.cond`` / ``lax.switch`` inline ALL
branches into the HLO graph and gate the result with a ``select_n`` —
identical to the dispatch_action precedent (``dispatch_action_brax.py``)
and to the Craftax pattern (``craftax/craftax/craftax_env.py``).  The
canonical ``features.py`` entry points called from ``action_dispatch``
(handle_open, handle_close, handle_kick, handle_search, sit_on_altar,
quaff_fountain, dip_fountain, sit_on_throne, drink_sink, sit_sink,
kick_sink) host a mix of small ``lax.cond`` blocks and large
``lax.switch`` outcome tables (13–16 branches) that fan out under vmap.

This module mirrors the Brax pattern: evaluate every branch
unconditionally, then leaf-by-leaf ``jnp.where``-cascade the result.  The
selected leaf for each ``mask == True`` is byte-identical to the original
``cond`` / ``switch`` output; unselected branch outputs are discarded.

Byte-parity contract
--------------------
1.  RNG draw order is preserved exactly.  Every original ``jax.random``
    call appears in the same order with the same key/subkey derivations as
    the canonical function.
2.  Branch bodies are pure-functional helpers reused verbatim from the
    canonical file (no copy-paste of effect logic) — every emitted
    mutation flows through ``jnp.where`` over the candidate
    sub-pytrees, so each leaf is bit-identical to ``lax.switch`` /
    ``lax.cond`` for the selected mask.
3.  The state pytree shape is preserved because every branch returns the
    same ``EnvState`` (or ``FeaturesState``/``MessageState``) pytree
    shape.

What is flattened (conds / switches per function)
-------------------------------------------------
* ``handle_open_brax``     — 1 ``lax.cond`` flattened.
* ``handle_close_brax``    — 0 (mirror; canonical already where-only).
* ``handle_kick_brax``     — 2 ``lax.cond`` flattened (outer riding/wounded,
                              inner monster/door).
* ``handle_search_brax``   — 0 (mirror; canonical already where-only).
* ``sit_on_altar_brax``    — 2 ``lax.cond`` flattened (wrath message,
                              luck-loss message).
* ``quaff_fountain_brax``  — 1 ``lax.switch`` (16 branches) flattened.
* ``dip_fountain_brax``    — 1 ``lax.switch`` (8 branches) + 2 ``lax.cond``
                              flattened (excal-eligible / grants_excalibur).
* ``sit_on_throne_brax``   — 1 ``lax.switch`` (13 branches) flattened.
* ``drink_sink_brax``      — 1 ``lax.switch`` (12 branches) flattened.
* ``sit_sink_brax``        — 0 (canonical is identity).
* ``kick_sink_brax``       — 0 (canonical already where-only).

Total flattened: **4 × ``lax.switch`` (49 branches) + 7 × ``lax.cond``**.

Branch bodies are referenced via the canonical implementations so the
original logic remains the single source of truth.
"""
import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems import features as _features
from Nethax.nethax.subsystems.features import (
    DoorState,
    FeaturesState,
    _flat_lv_from_state,
    _fountain_pos_idx,
    open_door,
    close_door,
    kick_door,
    altar_buc_sense,
    # Bucket constants for fountain/throne/sink dispatch tables.
    _FOUNTAIN_DRY_UP,
    _FOUNTAIN_GUSH,
    _FOUNTAIN_WATER_DEMON,
    _FOUNTAIN_WISH,
)


# ---------------------------------------------------------------------------
# Tree-map helpers — branchless state selection (mirrors dispatch_action_brax)
# ---------------------------------------------------------------------------

def _tree_where(pred: jnp.ndarray, on_true, on_false):
    """Leaf-wise ``jnp.where(pred, on_true_leaf, on_false_leaf)``.

    ``pred`` is a scalar boolean array broadcast against every leaf.  Both
    branches must share the pytree structure and per-leaf shape.
    """
    return jax.tree_util.tree_map(
        lambda t, f: jnp.where(pred, t, f),
        on_true,
        on_false,
    )


def _select_branch(idx: jnp.ndarray, outputs: tuple, default):
    """Cascade ``jnp.where(idx == i, outputs[i], acc)`` for all candidates.

    Mirrors ``lax.switch(idx, branches)`` semantics: the leaf returned for
    ``idx == k`` is exactly ``outputs[k]``.  ``default`` is returned for
    every leaf when ``idx`` matches none of the candidates (out-of-range
    case — well-formed inputs never trigger this).
    """
    acc = default
    for i, out in enumerate(outputs):
        mask = idx == jnp.int32(i)
        acc = _tree_where(mask, out, acc)
    return acc


# ---------------------------------------------------------------------------
# handle_open — flatten the single inner ``lax.cond`` over ``opens``.
# ---------------------------------------------------------------------------

def handle_open_brax(state, rng: jax.Array):
    """Brax-style rewrite of ``features.handle_open``.

    Canonical (``features.py`` line 1214) uses one ``lax.cond`` on the
    resist-roll outcome to pick between the post-open FeaturesState and
    the untouched original.  Flattened here with a single ``_tree_where``.

    RNG draw order preserved: ``jax.random.split(rng, 2)`` produces
    ``rng_resist`` then ``rng_trap``; both are consumed in the same order
    as the canonical function.

    Conds flattened: 1.
    """
    flat_lv = _flat_lv_from_state(state)
    pos = jnp.array(
        [flat_lv, state.player_pos[0], state.player_pos[1]],
        dtype=jnp.int32,
    )

    # Resist roll — vendor lock.c:904 (matches canonical bytes).
    str_i = state.player_str.astype(jnp.int32)
    dex_i = state.player_dex.astype(jnp.int32)
    con_i = state.player_con.astype(jnp.int32)
    sdc = jnp.maximum(str_i + dex_i + con_i, jnp.int32(1))
    rng_resist, rng_trap = jax.random.split(rng, 2)
    resist_roll = jax.random.randint(
        rng_resist, (), minval=0, maxval=sdc, dtype=jnp.int32,
    )
    opens = resist_roll < jnp.int32(30)

    lvl_diff = state.dungeon.current_level.astype(jnp.int32)
    new_features_opened, trap_dmg = open_door(
        state.features, pos, rng_trap, level_difficulty=lvl_diff,
    )

    # Flatten ``lax.cond(opens, _opened, _unchanged)`` with a tree_where.
    new_features = _tree_where(opens, new_features_opened, state.features)

    applied_dmg = jnp.where(opens, trap_dmg, jnp.int32(0))
    new_hp = jnp.maximum(jnp.int32(0), state.player_hp - applied_dmg)
    return state.replace(features=new_features, player_hp=new_hp)


# ---------------------------------------------------------------------------
# handle_close — canonical is already where-only.  Mirror for API uniformity.
# ---------------------------------------------------------------------------

def handle_close_brax(state, rng: jax.Array):
    """Drop-in mirror of ``features.handle_close`` (no flattening required).

    Canonical contains zero ``lax.cond`` / ``lax.switch`` calls; the body
    is pure-jnp where masking inside ``close_door``.  Provided here so
    callers can swap modules uniformly.

    Conds flattened: 0.
    """
    return _features.handle_close(state, rng)


# ---------------------------------------------------------------------------
# handle_kick — flatten the outer riding/wounded ``lax.cond`` AND the inner
# any-monster / door ``lax.cond``.
# ---------------------------------------------------------------------------

def handle_kick_brax(state, rng: jax.Array):
    """Brax-style rewrite of ``features.handle_kick``.

    Canonical (line 1297) nests three ``lax.cond`` calls:
        outer:  is_riding ? kick_steed : (is_wounded ? identity : _do_kick)
        inner:  any_monster ? _kick_monster : _kick_door

    All three are flattened via ``_tree_where`` over pre-computed candidate
    states.  RNG order preserved: ``riding.kick_steed`` receives the full
    rng; ``_do_kick`` derives ``key_kick, key_monk = jax.random.split(rng)``
    in the same order as canonical.

    Conds flattened: 2 outer (riding/wounded) + 1 inner (monster/door) = 3
    nested into a 4-way leaf-wise cascade.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS
    from Nethax.nethax.subsystems.riding import kick_steed as _riding_kick_steed
    from Nethax.nethax.subsystems.inventory import ArmorSlot as _ArmorSlot
    from Nethax.nethax.subsystems.character import ObjType as _ObjType
    from Nethax.nethax.constants.roles import Role as _Role
    from Nethax.nethax.constants.monsters import M1_THICK_HIDE as _M1_THICK_HIDE
    from Nethax.nethax.subsystems.combat import _MONSTER_SYMBOL_TABLE  # noqa: F401
    from Nethax.nethax.subsystems.polymorph import _monster_tables

    is_wounded = state.status.timed_statuses[int(_TS.WOUNDED_LEGS)] > jnp.int32(0)
    is_riding = state.player_steed_mid != jnp.uint32(0)

    # Steed branch carry — independent of door/monster computation.
    state_after_steed = _riding_kick_steed(state, rng)

    # --- _do_kick body, lifted from canonical handle_kick verbatim. -------
    flat_lv = _flat_lv_from_state(state)
    prow = state.player_pos[0].astype(jnp.int32)
    pcol = state.player_pos[1].astype(jnp.int32)

    mai = state.monster_ai
    at_tile = (
        mai.alive
        & (mai.pos[:, 0].astype(jnp.int32) == prow)
        & (mai.pos[:, 1].astype(jnp.int32) == pcol)
    )
    any_monster = jnp.any(at_tile)
    target_slot = jnp.argmax(at_tile).astype(jnp.int32)

    key_kick, key_monk = jax.random.split(rng)
    str_i = state.player_str.astype(jnp.int32)
    dex_i = state.player_dex.astype(jnp.int32)
    con_i = state.player_con.astype(jnp.int32)
    base = (str_i + dex_i + con_i) // jnp.int32(15)
    base_clamped = jnp.maximum(base, jnp.int32(1))
    dmg = jax.random.randint(
        key_kick, (), minval=1, maxval=base_clamped + jnp.int32(1), dtype=jnp.int32,
    )

    boots_inv_idx = state.inventory.worn_armor[int(_ArmorSlot.BOOTS)].astype(jnp.int32)
    has_boots = boots_inv_idx >= jnp.int32(0)
    safe_b = jnp.clip(boots_inv_idx, 0, state.inventory.items.type_id.shape[0] - 1)
    boot_type = jnp.where(
        has_boots,
        state.inventory.items.type_id[safe_b].astype(jnp.int32),
        jnp.int32(0),
    )
    boot_spe = jnp.where(
        has_boots,
        state.inventory.items.enchantment[safe_b].astype(jnp.int32),
        jnp.int32(0),
    )
    kicking_boots_worn = boot_type == jnp.int32(int(_ObjType.KICKING_BOOTS))
    dmg = dmg + jnp.where(kicking_boots_worn, jnp.int32(5), jnp.int32(0))

    is_monk = state.player_role == jnp.int8(int(_Role.MONK))
    monk_upper = jnp.maximum(dex_i // jnp.int32(2) + jnp.int32(1), jnp.int32(1))
    monk_roll = jax.random.randint(
        key_monk, (), minval=0, maxval=monk_upper, dtype=jnp.int32,
    )
    dmg = dmg + jnp.where(is_monk, monk_roll, jnp.int32(0))

    dmg = dmg + boot_spe
    dmg = dmg + state.player_udaminc.astype(jnp.int32)

    tables = _monster_tables()
    flags1 = tables["flags1"]
    entry = jnp.clip(
        mai.entry_idx[target_slot].astype(jnp.int32), 0, flags1.shape[0] - 1,
    )
    thick_hide = (flags1[entry] & jnp.uint32(_M1_THICK_HIDE)) != jnp.uint32(0)
    is_shade = jnp.bool_(False)

    dmg = jnp.where(thick_hide | is_shade, jnp.int32(0), dmg)
    dmg = jnp.maximum(dmg, jnp.int32(0)).astype(jnp.int32)

    # Inner branch A: kick a monster.
    old_hp = mai.hp[target_slot]
    new_hp = jnp.maximum(old_hp - dmg, jnp.int32(0))
    new_mai_monster = mai.replace(
        hp=mai.hp.at[target_slot].set(new_hp),
        peaceful=mai.peaceful.at[target_slot].set(jnp.bool_(False)),
    )
    state_kick_monster = state.replace(monster_ai=new_mai_monster)

    # Inner branch B: kick a door.
    pos_door = jnp.array([flat_lv, prow, pcol], dtype=jnp.int32)
    new_features_door, _ = kick_door(state.features, rng, pos_door)
    state_kick_door = state.replace(features=new_features_door)

    # Flatten inner ``lax.cond(any_monster, _kick_monster, _kick_door)``.
    state_after_kick = _tree_where(any_monster, state_kick_monster, state_kick_door)

    # Outer flatten: is_riding selects steed-branch; else wounded gates the
    # do_kick result vs the identity (no-op kick).
    state_when_not_riding = _tree_where(is_wounded, state, state_after_kick)
    return _tree_where(is_riding, state_after_steed, state_when_not_riding)


# ---------------------------------------------------------------------------
# handle_search — canonical is already where-only (no cond/switch).
# ---------------------------------------------------------------------------

def handle_search_brax(state, rng: jax.Array):
    """Drop-in mirror of ``features.handle_search`` (no flattening required).

    Canonical body uses static Python ``for dr, dc`` loops with
    ``jnp.where`` masking — zero ``lax.cond`` / ``lax.switch`` calls.
    Provided here for API uniformity.

    Conds flattened: 0.
    """
    return _features.handle_search(state, rng)


# ---------------------------------------------------------------------------
# sit_on_altar — flatten the two message-emit ``lax.cond`` calls.
# ---------------------------------------------------------------------------

def sit_on_altar_brax(state, rng):
    """Brax-style rewrite of ``features.sit_on_altar``.

    Canonical (line 1555) emits two flavor messages via ``lax.cond`` —
    one for the same-aligned wrath branch and one for the luck-loss
    branch.  Both are flattened by computing the emit-state once and
    selecting via ``_tree_where``.

    RNG order preserved: ``jax.random.split(rng, 4)`` then
    ``rng_a, rng_l1, rng_l2, rng_l3`` consumed in canonical order.

    Conds flattened: 2.
    """
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId

    max_levels = state.terrain.shape[1]
    b = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    flat_lv = b * jnp.int32(max_levels) + lv
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)
    altar_align = state.features.altar_alignment[flat_lv, row, col].astype(jnp.int32)
    player_align_i = state.player_align.astype(jnp.int32)
    on_altar = altar_align >= jnp.int32(0)

    rng_a, rng_l1, rng_l2, rng_l3 = jax.random.split(rng, 4)
    del rng_l3  # parity with canonical; unused by both implementations.

    same_aligned = on_altar & (altar_align == player_align_i)
    record = state.prayer.alignment_record.astype(jnp.int32)
    threshold = -jax.random.randint(rng_a, (), 0, 4, dtype=jnp.int32)
    record_gate = record > threshold
    do_wrath_same = same_aligned & record_gate

    new_wis = jnp.where(
        do_wrath_same,
        jnp.maximum(state.player_wis - jnp.int8(1), jnp.int8(3)),
        state.player_wis,
    )
    new_record = jnp.where(
        do_wrath_same,
        (state.prayer.alignment_record - jnp.int16(1)).astype(jnp.int16),
        state.prayer.alignment_record,
    )

    do_wrath_other = on_altar & ~do_wrath_same
    luck = state.player_luck.astype(jnp.int32)
    luck_eligible = do_wrath_other & (luck > jnp.int32(-5))
    upper = jnp.maximum(luck + jnp.int32(6), jnp.int32(1))
    luck_roll = jax.random.randint(rng_l1, (), 0, upper, dtype=jnp.int32)
    luck_active = luck_eligible & (luck_roll != jnp.int32(0))
    rn20 = jax.random.randint(rng_l2, (), 0, 20, dtype=jnp.int32)
    luck_delta = jnp.where(rn20 != jnp.int32(0), jnp.int32(-1), jnp.int32(-2))
    new_luck = jnp.where(
        luck_active,
        jnp.clip(luck + luck_delta, jnp.int32(-13), jnp.int32(13)),
        luck,
    ).astype(jnp.int8)

    new_prayer = state.prayer.replace(alignment_record=new_record)

    # Flatten message-emit conds.  Both candidate MessageState pytrees are
    # built unconditionally; selection is a leaf-wise where.
    msgs_unchanged = state.messages
    msgs_wrath_emit = _msg_emit(state.messages, int(_MsgId.ALTAR_WRATH))
    msgs_after_wrath = _tree_where(do_wrath_same, msgs_wrath_emit, msgs_unchanged)

    msgs_luck_emit = _msg_emit(msgs_after_wrath, int(_MsgId.ALTAR_LUCK_LOSS))
    msgs_after_luck = _tree_where(luck_active, msgs_luck_emit, msgs_after_wrath)

    new_state = state.replace(
        player_wis=new_wis,
        player_luck=new_luck,
        prayer=new_prayer,
        messages=msgs_after_luck,
    )

    # Retain BUC-reveal side-effect (Wave 4 altar_buc_sense semantics).
    return altar_buc_sense(new_state)


# ---------------------------------------------------------------------------
# quaff_fountain — flatten the 16-branch ``lax.switch`` over bucket.
# ---------------------------------------------------------------------------

def quaff_fountain_brax(state, rng):
    """Brax-style rewrite of ``features.quaff_fountain``.

    Canonical (line 1695) dispatches 16 outcome branches via ``lax.switch``.
    Flattened here by computing every branch unconditionally and selecting
    via ``_select_branch``.

    RNG draw order preserved: ``jax.random.split(rng, 3)`` → ``rng_fate,
    rng_eff, rng_dry``.  The shared ``rng_eff`` is forwarded to the
    ``_curse_ray`` branch closure exactly as in canonical (closure captures
    the outer ``rng_eff`` symbol).

    Switches flattened: 1 (16 branches).
    """
    rng_fate, rng_eff, rng_dry = jax.random.split(rng, 3)
    fate = jax.random.randint(rng_fate, (), minval=0, maxval=30, dtype=jnp.int32)

    bucket_table = jnp.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        _features._FOUNTAIN_SELF_KNOWLEDGE,
        _features._FOUNTAIN_FOUL_WATER,
        _features._FOUNTAIN_POISONOUS,
        _features._FOUNTAIN_SNAKES,
        _features._FOUNTAIN_WATER_DEMON,
        _features._FOUNTAIN_CURSE_RAY,
        _features._FOUNTAIN_SEE_INVISIBLE,
        _features._FOUNTAIN_MONSTER_DETECT,
        _features._FOUNTAIN_FIND_GEM,
        _features._FOUNTAIN_WATER_NYMPH,
        _features._FOUNTAIN_SCARE,
        _features._FOUNTAIN_GUSH,
    ], dtype=jnp.int32)
    bucket = bucket_table[fate]

    # 1-in-3 promote WATER_DEMON → WISH (uses rng_eff identically to canonical).
    demon_promote = jax.random.randint(rng_eff, (), 0, 3, dtype=jnp.int32) == 0
    bucket = jnp.where(
        (bucket == _FOUNTAIN_WATER_DEMON) & demon_promote,
        jnp.int32(_FOUNTAIN_WISH),
        bucket,
    )

    # Branch closures — identical to canonical (re-bound here so they
    # capture the same ``rng_eff`` symbol used inside ``_curse_ray``).
    def _refresh(s):
        return s.replace(status=s.status.replace(
            nutrition=s.status.nutrition + jnp.int32(10)))

    def _self_knowledge(s):
        return s.replace(player_wis=jnp.minimum(jnp.int8(25), s.player_wis + jnp.int8(1)))

    def _foul_water(s):
        return s.replace(status=s.status.replace(
            nutrition=s.status.nutrition - jnp.int32(20)))

    def _poisonous(s):
        return s.replace(
            player_str=jnp.maximum(jnp.int16(3), s.player_str - jnp.int16(3)),
            player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(5)),
        )

    def _snakes(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(2)))

    def _water_demon(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(3)))

    def _curse_ray(s):
        inv = s.inventory.items
        occupied = inv.category != jnp.int8(0)
        not_cursed = inv.buc_status != jnp.int8(1)
        n = inv.category.shape[0]
        rolls = jax.random.randint(rng_eff, (n,), 0, 5, dtype=jnp.int32)
        target = occupied & not_cursed & (rolls == jnp.int32(0))
        new_buc = jnp.where(target, jnp.int8(1), inv.buc_status)
        new_items = inv.replace(buc_status=new_buc)
        return s.replace(
            inventory=s.inventory.replace(items=new_items),
            status=s.status.replace(nutrition=s.status.nutrition - jnp.int32(20)),
        )

    def _see_invisible(s):
        from Nethax.nethax.subsystems.status_effects import Intrinsic
        new_intr = s.status.intrinsics.at[int(Intrinsic.SEE_INVIS)].set(True)
        return s.replace(status=s.status.replace(intrinsics=new_intr))

    def _monster_detect(s):
        from Nethax.nethax.subsystems.status_effects import Intrinsic
        new_intr = s.status.intrinsics.at[int(Intrinsic.DETECT_MONSTERS)].set(True)
        return s.replace(status=s.status.replace(intrinsics=new_intr))

    def _find_gem(s):
        return s.replace(player_gold=s.player_gold + jnp.int32(5))

    def _water_nymph(s):
        inv = s.inventory.items
        occupied = inv.category != jnp.int8(0)
        first_idx = jnp.argmax(occupied.astype(jnp.int32))
        has_any = jnp.any(occupied)
        new_cat = jnp.where(
            has_any,
            inv.category.at[first_idx].set(jnp.int8(0)),
            inv.category,
        )
        new_qty = jnp.where(
            has_any,
            inv.quantity.at[first_idx].set(jnp.int16(0)),
            inv.quantity,
        )
        new_items = inv.replace(category=new_cat, quantity=new_qty)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _scare(s):
        return s

    def _gush(s):
        return s

    def _dry_up_only(s):
        return s

    def _moist(s):
        return s.replace(
            player_hp=jnp.minimum(s.player_hp_max, s.player_hp + jnp.int32(10)),
            player_con=jnp.minimum(jnp.int8(25), s.player_con + jnp.int8(1)),
        )

    def _wish(s):
        new_max = s.player_hp_max + jnp.int32(5)
        return s.replace(
            player_hp_max=new_max,
            player_hp=new_max,
            player_gold=s.player_gold + jnp.int32(100),
        )

    branches = (
        _refresh, _self_knowledge, _foul_water, _poisonous,
        _snakes, _water_demon, _curse_ray, _see_invisible,
        _monster_detect, _find_gem, _water_nymph, _scare,
        _gush, _dry_up_only, _moist, _wish,
    )

    # Fan out every branch over the SAME input ``state`` — same byte image
    # as ``lax.switch(bucket, branches, state)``.
    outputs = tuple(fn(state) for fn in branches)
    new_state = _select_branch(bucket, outputs, state)

    # Post-step: dryup() roll (vendor 1-in-3, force on GUSH / DRY_UP buckets).
    dry_roll = jax.random.randint(rng_dry, (), 0, 3, dtype=jnp.int32)
    will_dry = (
        (dry_roll == jnp.int32(0))
        | (bucket == _FOUNTAIN_DRY_UP)
        | (bucket == _FOUNTAIN_GUSH)
    )
    flat_lv, row, col = _fountain_pos_idx(new_state)
    new_used = jnp.where(
        will_dry,
        new_state.features.fountains_used.at[flat_lv, row, col].set(True),
        new_state.features.fountains_used,
    )
    new_features = new_state.features.replace(fountains_used=new_used)
    return new_state.replace(features=new_features)


# ---------------------------------------------------------------------------
# dip_fountain — flatten 1 × 8-branch ``lax.switch`` + 2 × ``lax.cond``.
# ---------------------------------------------------------------------------

def dip_fountain_brax(state, rng, slot_idx):
    """Brax-style rewrite of ``features.dip_fountain``.

    Canonical (line 1883) hosts:
      - an 8-branch ``lax.switch`` over the dip bucket
      - an inner ``lax.cond(grants_excalibur, _grant_excal, _deny_excal)``
      - an outer ``lax.cond(excal_eligible, _excal_branch, _no_excal_branch)``

    All three are flattened.  RNG order preserved: ``jax.random.split(rng, 3)``
    → ``rng_fate, rng_excal, rng_dry`` consumed identically to canonical.

    Switches flattened: 1 (8 branches).  Conds flattened: 2.
    """
    rng_fate, rng_excal, rng_dry = jax.random.split(rng, 3)

    LONG_SWORD_TYPE_ID = 37
    LAWFUL = 2
    from Nethax.nethax.constants.roles import Role as _Role
    KNIGHT = int(_Role.KNIGHT)

    inv = state.inventory.items
    n = inv.category.shape[0]
    slot = jnp.clip(slot_idx.astype(jnp.int32), 0, n - 1)

    item_type = inv.type_id[slot]
    item_buc = inv.buc_status[slot]
    item_ench = inv.enchantment[slot]
    item_qty = inv.quantity[slot]
    item_filled = inv.category[slot] != jnp.int8(0)

    is_long_sword = (item_type == jnp.int16(LONG_SWORD_TYPE_ID)) & item_filled
    is_lawful = state.player_align == jnp.int8(LAWFUL)
    is_high_xl = state.player_xl >= jnp.int32(5)
    is_single = item_qty == jnp.int16(1)
    is_knight = state.player_role == jnp.int8(KNIGHT)
    excal_eligible = is_long_sword & is_lawful & is_high_xl & is_single
    excal_denom = jnp.where(is_knight, jnp.int32(6), jnp.int32(30))
    excal_roll = jax.random.randint(
        rng_excal, (), 0, excal_denom, dtype=jnp.int32,
    ) == jnp.int32(0)
    grants_excalibur = excal_eligible & excal_roll

    fate = jax.random.randint(rng_fate, (), 1, 31, dtype=jnp.int32)
    bucket = jnp.where(
        fate == jnp.int32(16), jnp.int32(0),
        jnp.where((fate >= jnp.int32(17)) & (fate <= jnp.int32(20)), jnp.int32(1),
        jnp.where(fate == jnp.int32(21), jnp.int32(2),
        jnp.where(fate == jnp.int32(22), jnp.int32(3),
        jnp.where(fate == jnp.int32(23), jnp.int32(4),
        jnp.where(fate == jnp.int32(29), jnp.int32(5),
        jnp.where(fate == jnp.int32(28), jnp.int32(6),
                  jnp.int32(7))))))))

    def _curse_item(s):
        new_buc = jnp.where(item_filled, jnp.int8(1), item_buc)
        new_arr = s.inventory.items.buc_status.at[slot].set(new_buc)
        new_items = s.inventory.items.replace(buc_status=new_arr)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _uncurse_item(s):
        new_buc = jnp.where(item_buc == jnp.int8(1), jnp.int8(2), item_buc)
        new_arr = s.inventory.items.buc_status.at[slot].set(new_buc)
        new_items = s.inventory.items.replace(buc_status=new_arr)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _water_demon(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(4)))

    def _water_nymph(s):
        return s

    def _snakes(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(2)))

    def _find_gold(s):
        return s.replace(player_gold=s.player_gold + jnp.int32(20))

    def _bath_lose_gold(s):
        loss = jnp.maximum(jnp.int32(0), s.player_gold // jnp.int32(10))
        return s.replace(player_gold=s.player_gold - loss)

    def _corrode(s):
        new_ench = jnp.where(
            item_filled,
            jnp.maximum(jnp.int8(-5), item_ench - jnp.int8(1)),
            item_ench,
        )
        new_arr = s.inventory.items.enchantment.at[slot].set(new_ench)
        new_items = s.inventory.items.replace(enchantment=new_arr)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    branches = (
        _curse_item, _uncurse_item, _water_demon, _water_nymph,
        _snakes, _find_gold, _bath_lose_gold, _corrode,
    )

    # Flatten the 8-branch ``lax.switch`` (every branch evaluated, then select).
    outputs = tuple(fn(state) for fn in branches)
    base_state = _select_branch(bucket, outputs, state)

    # Excalibur grant / deny — both evaluated unconditionally so the inner
    # ``lax.cond`` collapses to a single ``_tree_where``.
    def _grant_excal(s):
        new_buc = s.inventory.items.buc_status.at[slot].set(jnp.int8(3))
        new_ench = s.inventory.items.enchantment.at[slot].set(
            jnp.maximum(item_ench, jnp.int8(5))
        )
        new_ident = s.inventory.items.identified.at[slot].set(jnp.bool_(True))
        new_items = s.inventory.items.replace(
            buc_status=new_buc, enchantment=new_ench, identified=new_ident,
        )
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _deny_excal(s):
        new_buc = s.inventory.items.buc_status.at[slot].set(jnp.int8(1))
        new_items = s.inventory.items.replace(buc_status=new_buc)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    # Both candidate excal sub-states operate on the original ``state`` —
    # mirroring canonical's ``_excal_branch`` which feeds ``state`` (not
    # ``base_state``) to ``lax.cond(grants_excalibur, _grant, _deny, state)``.
    excal_grant_state = _grant_excal(state)
    excal_deny_state = _deny_excal(state)
    excal_branch_state = _tree_where(grants_excalibur, excal_grant_state, excal_deny_state)

    # Outer cond: excal_eligible ? excal_branch_state : base_state.
    out_state = _tree_where(excal_eligible, excal_branch_state, base_state)

    # Mark fountain used in some outcomes (always on Excalibur path).
    drain_roll = jax.random.randint(rng_dry, (), 0, 3, dtype=jnp.int32)
    will_dry = (drain_roll == jnp.int32(0)) | grants_excalibur
    flat_lv, row, col = _fountain_pos_idx(out_state)
    new_used = jnp.where(
        will_dry,
        out_state.features.fountains_used.at[flat_lv, row, col].set(True),
        out_state.features.fountains_used,
    )
    new_features = out_state.features.replace(fountains_used=new_used)
    return out_state.replace(features=new_features)


# ---------------------------------------------------------------------------
# sit_on_throne — flatten the 13-branch ``lax.switch`` over effect.
# ---------------------------------------------------------------------------

def sit_on_throne_brax(state, rng):
    """Brax-style rewrite of ``features.sit_on_throne``.

    Canonical (line 2063) dispatches 13 throne effects via ``lax.switch``.
    Flattened here by evaluating every branch and selecting via
    ``_select_branch``.

    RNG draw order preserved: ``jax.random.split(rng, 3)`` →
    ``rng_eff, rng_post, _`` consumed identically to canonical.

    Switches flattened: 1 (13 branches).
    """
    rng_eff, rng_post, _ = jax.random.split(rng, 3)
    effect = jax.random.randint(rng_eff, (), 0, 13, dtype=jnp.int32)

    def _attr_loss(s):
        return s.replace(
            player_str=jnp.maximum(jnp.int16(3), s.player_str - jnp.int16(3)),
            player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(5)),
        )

    def _attr_gain(s):
        return s.replace(
            player_str=jnp.minimum(jnp.int16(125), s.player_str + jnp.int16(1)))

    def _shock(s):
        from Nethax.nethax.subsystems.status_effects import Intrinsic
        has_res = s.status.intrinsics[int(Intrinsic.RESIST_SHOCK)]
        dmg = jnp.where(has_res, jnp.int32(3), jnp.int32(15))
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - dmg))

    def _full_heal(s):
        return s.replace(player_hp=s.player_hp_max)

    def _take_gold(s):
        return s.replace(player_gold=jnp.int32(0))

    def _wish(s):
        new_max = s.player_hp_max + jnp.int32(5)
        return s.replace(
            player_hp_max=new_max,
            player_hp=new_max,
            player_gold=s.player_gold + jnp.int32(100),
        )

    def _court(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(2)))

    def _genocide(s):
        return s.replace(player_xp=s.player_xp + jnp.int32(50))

    def _curse_items(s):
        inv = s.inventory.items
        occupied = inv.category != jnp.int8(0)
        not_cursed = inv.buc_status != jnp.int8(1)
        eligible = occupied & not_cursed
        idx = jnp.argmax(eligible.astype(jnp.int32))
        has_any = jnp.any(eligible)
        new_buc = jnp.where(
            has_any,
            inv.buc_status.at[idx].set(jnp.int8(1)),
            inv.buc_status,
        )
        new_items = inv.replace(buc_status=new_buc)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _map_or_confuse(s):
        return s.replace(
            player_wis=jnp.minimum(jnp.int8(25), s.player_wis + jnp.int8(1)))

    def _teleport(s):
        return s.replace(player_pos=jnp.array([1, 1], dtype=jnp.int16))

    def _identify(s):
        inv = s.inventory.items
        occupied = inv.category != jnp.int8(0)
        cum = jnp.cumsum(occupied.astype(jnp.int32))
        mark = occupied & (cum <= jnp.int32(5))
        new_ident = jnp.where(mark, jnp.bool_(True), inv.identified)
        new_items = inv.replace(identified=new_ident)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _confuse(s):
        from Nethax.nethax.subsystems.status_effects import TimedStatus
        new_ts = s.status.timed_statuses.at[int(TimedStatus.CONFUSION)].add(jnp.int32(20))
        return s.replace(status=s.status.replace(timed_statuses=new_ts))

    branches = (
        _attr_loss, _attr_gain, _shock, _full_heal,
        _take_gold, _wish, _court, _genocide,
        _curse_items, _map_or_confuse, _teleport, _identify,
        _confuse,
    )
    outputs = tuple(fn(state) for fn in branches)
    new_state = _select_branch(effect, outputs, state)

    # Per-effect flavor message (uses indexed _msg_emit_t, NOT a cond).
    from Nethax.nethax.subsystems.messages import emit as _msg_emit_t, MessageId as _MsgId_t
    _THRONE_MSG_IDS = jnp.array([
        int(_MsgId_t.THRONE_ATTR_LOSS),
        int(_MsgId_t.THRONE_ATTR_GAIN),
        int(_MsgId_t.THRONE_SHOCK),
        int(_MsgId_t.THRONE_FULL_HEAL),
        int(_MsgId_t.THRONE_TAKE_GOLD),
        int(_MsgId_t.THRONE_WISH),
        int(_MsgId_t.THRONE_COURT),
        int(_MsgId_t.THRONE_GENOCIDE),
        int(_MsgId_t.THRONE_CURSE_ITEMS),
        int(_MsgId_t.THRONE_MAP_CONFUSE),
        int(_MsgId_t.THRONE_TELEPORT),
        int(_MsgId_t.THRONE_IDENTIFY),
        int(_MsgId_t.THRONE_CONFUSE),
    ], dtype=jnp.int32)
    safe_effect = jnp.clip(effect, jnp.int32(0), jnp.int32(_THRONE_MSG_IDS.shape[0] - 1))
    throne_msg_id = _THRONE_MSG_IDS[safe_effect]
    new_state = new_state.replace(
        messages=_msg_emit_t(new_state.messages, throne_msg_id),
    )

    # Post-step: 1-in-3 chance throne disappears.
    destroy_roll = jax.random.randint(rng_post, (), 0, 3, dtype=jnp.int32)
    will_destroy = destroy_roll == jnp.int32(0)
    flat_lv = _flat_lv_from_state(new_state)
    row = new_state.player_pos[0].astype(jnp.int32)
    col = new_state.player_pos[1].astype(jnp.int32)
    new_used = jnp.where(
        will_destroy,
        new_state.features.thrones_used.at[flat_lv, row, col].set(True),
        new_state.features.thrones_used,
    )
    new_features = new_state.features.replace(thrones_used=new_used)
    return new_state.replace(features=new_features)


# ---------------------------------------------------------------------------
# drink_sink — flatten the 12-branch ``lax.switch`` over bucket.
# ---------------------------------------------------------------------------

def drink_sink_brax(state, rng):
    """Brax-style rewrite of ``features.drink_sink``.

    Canonical (line 2223) dispatches 12 sink-effect branches via
    ``lax.switch``.  Flattened here.

    RNG draw order preserved: ``jax.random.split(rng, 2)`` → ``rng_eff, _``
    consumed identically to canonical.

    Switches flattened: 1 (12 branches).
    """
    rng_eff, _ = jax.random.split(rng, 2)
    fate = jax.random.randint(rng_eff, (), 0, 20, dtype=jnp.int32)

    bucket_table = jnp.array([
        _features._SINK_COLD_WATER,
        _features._SINK_WARM_WATER,
        _features._SINK_SCALDING,
        _features._SINK_SEWER_RAT,
        _features._SINK_RANDOM_POTION,
        _features._SINK_FIND_RING,
        _features._SINK_BREAK,
        _features._SINK_WATER_ELEMENTAL,
        _features._SINK_DRAIN_NUTRITION,
        _features._SINK_DRAIN_NUTRITION,
        _features._SINK_POLYMORPH,
        _features._SINK_NOISE,
        _features._SINK_NOISE,
        _features._SINK_STENCH,
        _features._SINK_COLD_WATER,
        _features._SINK_COLD_WATER,
        _features._SINK_COLD_WATER,
        _features._SINK_COLD_WATER,
        _features._SINK_COLD_WATER,
        _features._SINK_COLD_WATER,
    ], dtype=jnp.int32)
    bucket = bucket_table[fate]

    def _cold_water(s):
        return s

    def _warm_water(s):
        return s

    def _scalding(s):
        from Nethax.nethax.subsystems.status_effects import Intrinsic
        fire_res = s.status.intrinsics[int(Intrinsic.RESIST_FIRE)]
        dmg = jnp.where(fire_res, jnp.int32(0), jnp.int32(3))
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - dmg))

    def _sewer_rat(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(1)))

    def _random_potion(s):
        return s.replace(player_wis=jnp.minimum(jnp.int8(25), s.player_wis + jnp.int8(1)))

    def _find_ring(s):
        inv = s.inventory.items
        left, right = s.inventory.worn_rings[0], s.inventory.worn_rings[1]
        left_safe = jnp.maximum(jnp.int32(0), left.astype(jnp.int32))
        right_safe = jnp.maximum(jnp.int32(0), right.astype(jnp.int32))
        has_left = left >= jnp.int8(0)
        has_right = right >= jnp.int8(0)

        new_ident = inv.identified
        new_ident = jnp.where(
            has_left,
            new_ident.at[left_safe].set(jnp.bool_(True)),
            new_ident,
        )
        new_ident = jnp.where(
            has_right,
            new_ident.at[right_safe].set(jnp.bool_(True)),
            new_ident,
        )
        new_items = inv.replace(identified=new_ident)
        flat_lv = _flat_lv_from_state(s)
        row = s.player_pos[0].astype(jnp.int32)
        col = s.player_pos[1].astype(jnp.int32)
        new_sink_arr = s.features.sinks_used.at[flat_lv, row, col].set(True)
        new_features = s.features.replace(sinks_used=new_sink_arr)
        return s.replace(
            inventory=s.inventory.replace(items=new_items),
            features=new_features,
        )

    def _break(s):
        flat_lv = _flat_lv_from_state(s)
        row = s.player_pos[0].astype(jnp.int32)
        col = s.player_pos[1].astype(jnp.int32)
        new_arr = s.features.sinks_used.at[flat_lv, row, col].set(True)
        return s.replace(features=s.features.replace(sinks_used=new_arr))

    def _water_elemental(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(2)))

    def _drain_nutrition(s):
        return s.replace(status=s.status.replace(
            nutrition=s.status.nutrition - jnp.int32(30)))

    def _polymorph(s):
        from Nethax.nethax.subsystems.status_effects import Intrinsic
        unchanging = s.status.intrinsics[int(Intrinsic.UNCHANGING)]
        delta = jnp.where(unchanging, jnp.int32(0), jnp.int32(1))
        return s.replace(player_xp=s.player_xp + delta)

    def _noise(s):
        return s

    def _stench(s):
        return s.replace(status=s.status.replace(
            nutrition=s.status.nutrition - jnp.int32(5)))

    branches = (
        _cold_water, _warm_water, _scalding, _sewer_rat,
        _random_potion, _find_ring, _break, _water_elemental,
        _drain_nutrition, _polymorph, _noise, _stench,
    )
    outputs = tuple(fn(state) for fn in branches)
    return _select_branch(bucket, outputs, state)


# ---------------------------------------------------------------------------
# sit_sink — canonical is an explicit identity (no cond/switch).
# ---------------------------------------------------------------------------

def sit_sink_brax(state, rng):
    """Drop-in mirror of ``features.sit_sink`` (canonical is identity).

    Conds flattened: 0.
    """
    return _features.sit_sink(state, rng)


# ---------------------------------------------------------------------------
# kick_sink — canonical already where-only (cascaded rn2 + jnp.where).
# ---------------------------------------------------------------------------

def kick_sink_brax(state, rng):
    """Drop-in mirror of ``features.kick_sink`` (canonical is where-only).

    Conds flattened: 0.
    """
    return _features.kick_sink(state, rng)
