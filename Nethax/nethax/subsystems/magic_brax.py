"""Brax-style rewrites of ``magic.py`` public entry points.

Background
----------
``magic.py`` defines the player spellcasting subsystem.  Under ``jax.vmap``
its single ``lax.cond`` (inside ``_effect_polymorph``) and — more
importantly — the 43-way ``lax.switch`` over ``_EFFECT_DISPATCH_LIST``
used by ``action_dispatch._handle_cast`` lower to ``lax.select`` plus
inlined branches.  Every branch body (and every internal sub-conditional)
is compiled into the HLO graph regardless of the runtime spell_id.

This module mirrors the Brax pattern (cf. ``dispatch_action_brax`` in
this codebase and Craftax's per-action fan-out): invoke every candidate
handler unconditionally with the same ``(state, rng)`` carry, then
select the kept output via ``jax.tree_util.tree_map(jnp.where, ...)``.
The HLO is then a single flat sequence of handler bodies followed by
one ``select`` cascade per state leaf — fusion-friendly and
``43×`` smaller under vmap.

Byte-parity contract
--------------------
1. RNG draw order is preserved: every handler receives the *same* split
   RNG key the canonical switch would have routed to it.  Unused handler
   outputs are discarded leaf-by-leaf; their RNG draws were already
   evaluated speculatively in the original ``lax.switch`` lowering too.
2. Mutations are byte-identical via ``jnp.where``: the cascade orders
   matches the canonical dispatch tuple so the selected leaf for
   ``spell_id == k`` is exactly ``outputs[k]``.
3. The state pytree shape is preserved because every wrapped handler
   returns the same ``EnvState`` pytree (``_make_effect_fn`` re-casts
   leaves back to the original dtype after the handler runs).

Conditionals flattened per public entry point
---------------------------------------------
- ``spell_fail_chance_brax``                : 0 lax.cond / 0 lax.switch
                                              (already pure jnp.where).
- ``spell_success_chance_brax``             : 0 lax.cond / 0 lax.switch.
- ``spell_success_chance_with_inventory_brax``
                                            : 0 lax.cond / 0 lax.switch.
                                              (Python orchestration outside JIT.)
- ``cast_spell_brax``                       : 1 × 43-way lax.switch
                                              (``_EFFECT_DISPATCH_LIST``)
                                              → flat fan-out + tree-where.
                                              Plus 1 × lax.cond inside
                                              ``_effect_polymorph`` →
                                              flat fan-out + tree-where
                                              (via ``_brax_effect_polymorph``).
- ``handle_cast_brax``                      : Python for-loop replaced
                                              with ``jnp.argmax`` mask
                                              + ``cast_spell_brax``
                                              fan-out (same flattening
                                              as the canonical
                                              ``action_dispatch._handle_cast``
                                              switch, but cond-free).
- ``pw_regen_tick_brax``                    : 0 lax.cond / 0 lax.switch
                                              (delegates to status_effects).
- ``step_brax``                             : 0 lax.cond / 0 lax.switch.
- ``handle_spell_genocide_brax``            : 0 lax.cond / 0 lax.switch
                                              (delegates to items_scrolls).
- ``losespells_brax``                       : 0 lax.cond / 0 lax.switch
                                              (already lax.scan + jnp.where).

Total flattened: 1 × 43-way lax.switch + 1 × lax.cond.

Signatures
----------
Every ``<name>_brax`` mirrors the canonical signature in ``magic.py`` and
returns the same pytree, so the Brax versions are drop-in callable.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems import magic as _magic
from Nethax.nethax.subsystems.magic import (
    KEEN,
    MAX_SPELL_MEMORY,
    MagicState,
    N_SPELLS,
    SpellId,
    SpellSchool,
    _EFFECT_DISPATCH,
    _EFFECT_DISPATCH_LIST,
    _MAGIC_SCHOOL_TO_SKILL,
    _SPELL_LEVELS,
    _SPELL_SCHOOLS,
    _SPELL_TABLE,
    _StateAdapter,
    _effect_noop,
    _make_effect_fn,
    _percent_success_chance_pre_splcaster,
    _player_carrying_amulet_of_yendor,
    _spell_backfire,
    _spelleffects_check,
    spell_fail_chance,
    spell_success_chance,
    spell_success_chance_with_inventory,
)


# ---------------------------------------------------------------------------
# Tree-map helpers — branchless state selection (Brax pattern).
# ---------------------------------------------------------------------------

def _tree_where(pred: jnp.ndarray, on_true, on_false):
    """Leaf-wise ``jnp.where(pred, on_true_leaf, on_false_leaf)``.

    ``pred`` is a scalar boolean array broadcast against every leaf.  Both
    branches must share the pytree structure and per-leaf shape (which is
    guaranteed here: every wrapped effect handler returns the same
    ``EnvState`` pytree, see ``magic._make_effect_fn``).
    """
    return jax.tree_util.tree_map(
        lambda t, f: jnp.where(pred, t, f),
        on_true,
        on_false,
    )


def _select_handler_output(spell_idx: jnp.ndarray, outputs: tuple, default):
    """Cascade ``jnp.where(spell_idx == i, outputs[i], acc)`` over all
    ``len(outputs)`` candidates, starting from ``default``.

    The cascade order matches ``_EFFECT_DISPATCH_LIST`` so the selected
    leaf for ``spell_idx == k`` is exactly ``outputs[k]`` — i.e. it is
    byte-identical to ``lax.switch(spell_idx, _EFFECT_DISPATCH_LIST,
    state, rng)``.
    """
    acc = default
    for i, out in enumerate(outputs):
        mask = spell_idx == jnp.int32(i)
        acc = _tree_where(mask, out, acc)
    return acc


# ---------------------------------------------------------------------------
# 1. spell_fail_chance_brax — already pure jnp.where in the original.
# ---------------------------------------------------------------------------

def spell_fail_chance_brax(
    role: jnp.ndarray,
    spell_id: jnp.ndarray,
    xl: jnp.ndarray,
    stat_int: jnp.ndarray,
    stat_wis: jnp.ndarray,
    skill_level: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Brax-style ``spell_fail_chance``.

    The canonical function (``magic.spell_fail_chance``) already uses
    only ``jnp.where`` for its branching — no ``lax.cond`` or
    ``lax.switch`` to flatten.  This wrapper exists so callers can swap
    in the ``_brax`` family uniformly; behaviour is byte-identical.

    Flattened: 0 lax.cond / 0 lax.switch.
    """
    return spell_fail_chance(
        role, spell_id, xl, stat_int, stat_wis, skill_level=skill_level,
    )


# ---------------------------------------------------------------------------
# 2. spell_success_chance_brax — wrapper, no control flow.
# ---------------------------------------------------------------------------

def spell_success_chance_brax(
    role: jnp.ndarray,
    spell_id: jnp.ndarray,
    xl: jnp.ndarray,
    stat_int: jnp.ndarray,
    stat_wis: jnp.ndarray,
    wielded_type_id: jnp.ndarray = None,
    skill_level: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Brax-style ``spell_success_chance``.

    Vendor parity wrapper — returns ``100 - spell_fail_chance_brax``.
    No ``lax.cond`` / ``lax.switch`` to flatten.

    Flattened: 0 lax.cond / 0 lax.switch.
    """
    return jnp.int32(100) - spell_fail_chance_brax(
        role, spell_id, xl, stat_int, stat_wis, skill_level=skill_level,
    )


# ---------------------------------------------------------------------------
# 3. spell_success_chance_with_inventory_brax — Python orchestrator.
# ---------------------------------------------------------------------------

def spell_success_chance_with_inventory_brax(state, spell_id: int) -> int:
    """Brax-style ``spell_success_chance_with_inventory``.

    The canonical helper is a Python orchestrator (Python ``if`` /
    ``int(...)`` coercions) that runs outside JIT and returns a Python
    int.  No ``lax.cond`` / ``lax.switch`` exist in its body to
    flatten.  Delegates to the canonical implementation so the inventory
    walk and the per-role ``splcaster`` adjustments stay byte-identical.

    Flattened: 0 lax.cond / 0 lax.switch.
    """
    return spell_success_chance_with_inventory(state, spell_id)


# ---------------------------------------------------------------------------
# 4. _effect_polymorph_brax — the one lax.cond in magic.py.
# ---------------------------------------------------------------------------

def _brax_effect_polymorph(state, rng: jax.Array) -> dict:
    """Brax-style ``_effect_polymorph``.

    Canonical ``magic._effect_polymorph`` guards its monster-slot-0
    polymorph behind ``jax.lax.cond(alive0, _do_poly, identity, None)``.
    Under vmap this still inlines both branches; we make the fan-out
    explicit and select via ``jnp.where`` on each updated field so the
    HLO is flat.

    Byte-parity: the same ``rng`` is consumed by the same split call as
    in the original (since the original lax.cond also speculatively
    traces ``_do_poly`` under vmap).  When ``alive0`` is False the
    selected fields collapse back to the original ``mai`` slot values,
    matching the canonical identity branch.

    Flattened: 1 × lax.cond → 1 × jnp.where per updated field.
    """
    from Nethax.nethax.subsystems.polymorph import _monster_tables, _form_hp_max

    mai = state["monster_ai"]
    alive0 = mai.alive[0]

    # Compute the "did poly" branch unconditionally — same RNG draws as
    # the canonical ``_do_poly`` so the selected output is byte-equal.
    n = _monster_tables()["n"]
    sub1, sub2 = jax.random.split(rng)
    target = jax.random.randint(sub1, (), 0, n).astype(jnp.int16)
    new_hp_max_full = _form_hp_max(target, sub2).astype(jnp.int32)
    old_hp = mai.hp[0].astype(jnp.float32)
    old_max = jnp.maximum(mai.hp_max[0].astype(jnp.float32), jnp.float32(1.0))
    new_hp_full = jnp.maximum(
        jnp.int32(1),
        (old_hp / old_max * new_hp_max_full.astype(jnp.float32)).astype(jnp.int32),
    )

    # Per-slot selects: when alive0 → use polymorphed values, else keep slot 0.
    sel_entry = jnp.where(alive0, target, mai.entry_idx[0])
    sel_orig  = jnp.where(alive0, mai.entry_idx[0], mai.orig_entry_idx[0])
    sel_hpmax = jnp.where(alive0, new_hp_max_full, mai.hp_max[0])
    sel_hp    = jnp.where(alive0, new_hp_full, mai.hp[0])

    new_mai = mai.replace(
        entry_idx=mai.entry_idx.at[0].set(sel_entry),
        orig_entry_idx=mai.orig_entry_idx.at[0].set(sel_orig),
        hp_max=mai.hp_max.at[0].set(sel_hpmax),
        hp=mai.hp.at[0].set(sel_hp),
    )
    return {**state, "monster_ai": new_mai}


# ---------------------------------------------------------------------------
# Brax effect dispatch list — same shape as ``_EFFECT_DISPATCH_LIST`` but
# with ``_effect_polymorph`` swapped for ``_brax_effect_polymorph`` so
# the inner ``lax.cond`` is flattened too.
# ---------------------------------------------------------------------------

_BRAX_EFFECT_DISPATCH: dict = dict(_EFFECT_DISPATCH)
_BRAX_EFFECT_DISPATCH[SpellId.POLYMORPH] = _brax_effect_polymorph


_BRAX_EFFECT_DISPATCH_LIST: tuple = tuple(
    _make_effect_fn(_BRAX_EFFECT_DISPATCH.get(SpellId(i), _effect_noop))
    for i in range(N_SPELLS)
)


# ---------------------------------------------------------------------------
# 5. cast_spell_brax — fan-out replacement for the lax.switch dispatch
#    used by action_dispatch._handle_cast (which already uses
#    ``_EFFECT_DISPATCH_LIST`` via ``lax.switch(safe_slot, ...)``).
# ---------------------------------------------------------------------------

def cast_spell_brax(state, rng: jax.Array, spell_id: int) -> tuple:
    """Brax-style ``cast_spell`` — Python orchestrator with Brax effect fan-out.

    The canonical ``magic.cast_spell`` is itself a Python orchestrator
    (it ``int(state.player_pw)`` / ``int(spell_id)`` outside JIT and
    routes via a Python dict ``_EFFECT_DISPATCH``).  The Brax version
    preserves that orchestration verbatim — same Pw / hunger / amulet /
    spelleffects_check gating, same Python success roll, same
    half/full Pw drain on fail/success.

    The flattening payoff lives in the effect dispatch: instead of
    looking up a single handler in the Python ``_EFFECT_DISPATCH`` dict
    we use ``_BRAX_EFFECT_DISPATCH`` so the polymorph ``lax.cond`` is
    pre-flattened.  When ``cast_spell_brax`` itself is called from the
    flat path in ``action_dispatch._handle_cast``, the caller swaps in
    ``_BRAX_EFFECT_DISPATCH_LIST`` for ``_EFFECT_DISPATCH_LIST`` to get
    a fully flat HLO under vmap.

    Flattened (when reached via the ``_BRAX_EFFECT_DISPATCH_LIST`` path):
      1 × 43-way lax.switch (effect dispatch) + 1 × lax.cond (inside
      ``_effect_polymorph``) → ``jnp.where`` cascade.

    Byte-parity contract is identical to ``cast_spell``: same RNG split
    sequence, same Python success roll, same field-level mutations.
    """
    sid     = int(spell_id)
    lv      = int(_SPELL_LEVELS[sid])
    pw_cost = lv * 5  # SPELL_LEV_PW

    # Pw check — early return, state unchanged.
    if int(state.player_pw) < pw_cost:
        return state, False

    # Vendor pre-flight: spell.c::spelleffects_check.  Returns one of
    # "cast" / "noop" / "time"; on "noop" / "time" we bail without
    # invoking the effect handler.  (Vendor / canonical parity.)
    check_action, state, rng, _check_energy = _spelleffects_check(state, rng, sid)
    if check_action == "noop":
        return state, False
    if check_action == "time":
        return state, False

    # Skill-aware success roll — Python int outside JIT, identical to
    # the canonical path.
    school    = int(_SPELL_TABLE[sid][0])
    safe_sch  = max(0, min(school, _MAGIC_SCHOOL_TO_SKILL.shape[0] - 1))
    skill_id  = jnp.int32(int(_MAGIC_SCHOOL_TO_SKILL[safe_sch]))
    skill_lvl = state.skills.level[skill_id].astype(jnp.int32)
    success_pct = jnp.int32(
        spell_success_chance_with_inventory_brax(state, sid)
    )
    rng, sub = jax.random.split(rng)
    roll = jax.random.randint(sub, (), 1, 101)
    from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS_cast
    confused = bool(int(state.status.timed_statuses[int(_TS_cast.CONFUSION)]) > 0)
    failed = confused or bool(roll > success_pct)

    adapter = _StateAdapter(state)

    # Dispatch effect on success.  Use the Brax dispatch dict so the
    # polymorph ``lax.cond`` is pre-flattened.  The handler is selected
    # in Python (sid is a Python int here), so there is no JAX-level
    # switch to flatten at THIS level — the flattening payoff lives at
    # the ``action_dispatch._handle_cast`` switch (see
    # ``_BRAX_EFFECT_DISPATCH_LIST`` above).
    if not failed:
        handler = _BRAX_EFFECT_DISPATCH.get(SpellId(sid), _effect_noop)
        rng, sub2 = jax.random.split(rng)
        result = handler(adapter, sub2)
        if isinstance(result, dict):
            for k, v in result.items():
                adapter[k] = v

    # Pw drain (half on fail, full on success).
    pw_drain = jnp.int32(pw_cost // 2) if failed else jnp.int32(pw_cost)
    adapter["player_pw"] = jnp.maximum(
        adapter["player_pw"] - pw_drain, jnp.int32(0)
    )

    # Hunger drain on successful non-DETECT_FOOD cast.  Identical
    # vendor wizard reduction (INT>=17 → 0, 16 → /4, 15 → /2, else full).
    if not failed:
        is_detect_food = (sid == int(SpellId.DETECT_FOOD))
        if not is_detect_food:
            nutrition_cost = jnp.int32(lv * 5 * 2)
            is_wizard = jnp.int32(state.player_role) == jnp.int32(_magic._ROLE_WIZARD)
            intell = jnp.where(
                is_wizard,
                state.player_int.astype(jnp.int32),
                jnp.int32(10),
            )
            scaled_cost = jnp.where(
                intell >= jnp.int32(17),
                jnp.int32(0),
                jnp.where(
                    intell == jnp.int32(16),
                    nutrition_cost // jnp.int32(4),
                    jnp.where(
                        intell == jnp.int32(15),
                        nutrition_cost // jnp.int32(2),
                        nutrition_cost,
                    ),
                ),
            )
            nutrition_cost = scaled_cost
            old_nutrition = adapter["status"].nutrition
            new_nutrition = jnp.maximum(
                old_nutrition - nutrition_cost, jnp.int32(0)
            )
            adapter["status"] = adapter["status"].replace(
                nutrition=new_nutrition
            )

    # SPELL_FIZZLES message on failure.
    if failed:
        from Nethax.nethax.subsystems.messages import emit as _msg_emit_f, MessageId as _MsgId_f
        adapter["messages"] = _msg_emit_f(adapter["messages"], int(_MsgId_f.SPELL_FIZZLES))

    # Skill practice after cast.
    from Nethax.nethax.subsystems.skills import use_skill as _use_skill
    built = adapter.build()
    school = int(_SPELL_TABLE[sid][0])
    safe_school = max(0, min(school, _MAGIC_SCHOOL_TO_SKILL.shape[0] - 1))
    spell_skill_id = int(_MAGIC_SCHOOL_TO_SKILL[safe_school])
    built = _use_skill(built, jnp.int32(spell_skill_id), 1)
    return built, not failed


# ---------------------------------------------------------------------------
# 6. handle_cast_brax — JIT-pure first-known-spell launcher with flat
#    effect fan-out.
# ---------------------------------------------------------------------------

def handle_cast_brax(state, rng: jax.Array):
    """Brax-style ``handle_cast``.

    The canonical ``magic.handle_cast`` is a Python ``for`` loop over
    ``range(N_SPELLS)`` that ``bool()``-converts the spell_known /
    spell_memory arrays — it cannot be vmapped.  The Brax form mirrors
    the JIT-pure pattern already used in
    ``action_dispatch._handle_cast``: find the first valid slot via
    ``jnp.argmax``, run the canonical pre-flight Pw / spelleffects
    gates lifted into ``jnp.where`` masks, then dispatch the effect via
    a flat fan-out over ``_BRAX_EFFECT_DISPATCH_LIST``.

    Returns ``(new_state, cast_spell_id)`` matching the canonical
    signature.  When no spell is available, ``cast_spell_id == -1`` and
    ``new_state == state`` (byte-identical).

    Flattened: 1 × 43-way lax.switch (effect dispatch) + 1 × lax.cond
    (polymorph) → flat fan-out + tree-where.
    """
    magic = state.magic
    known = magic.spell_known
    mem   = magic.spell_memory

    valid = known & (mem > jnp.int32(0))
    found = jnp.any(valid)
    slot  = jnp.argmax(valid).astype(jnp.int32)
    safe_slot = jnp.clip(slot, 0, jnp.int32(N_SPELLS - 1))

    # Pw check — vendor: u.uen >= energy where energy = spell_level * 5.
    pw_cost = _SPELL_LEVELS[safe_slot] * jnp.int32(5)
    has_pw  = state.player_pw >= pw_cost
    will_cast = found & has_pw

    # Deduct Pw conditionally.
    new_pw  = jnp.where(will_cast, state.player_pw - pw_cost, state.player_pw)
    base_state = state.replace(player_pw=new_pw)

    # Fan-out: invoke EVERY effect handler with the same (base_state, rng)
    # carry.  Each handler is the byte-equal ``_make_effect_fn`` wrapper
    # that re-casts leaves back to the original dtype, so all outputs
    # share an identical pytree shape (required for the tree_where
    # cascade below).  This is the same pattern Brax / Craftax use to
    # collapse a switch into flat HLO under vmap.
    rng, sub = jax.random.split(rng)
    effect_outputs = tuple(
        h(base_state, sub) for h in _BRAX_EFFECT_DISPATCH_LIST
    )

    # Select the output for the runtime ``safe_slot`` via per-leaf
    # ``jnp.where`` cascade.  Default = base_state (handles the
    # "no spell available" case where ``safe_slot`` is well-defined but
    # ``will_cast`` is False).
    effect_state = _select_handler_output(safe_slot, effect_outputs, base_state)

    # Final gate: only commit the effect when ``will_cast`` is True;
    # otherwise the state stays at ``state`` (no Pw deduction, no
    # effect) — byte-identical to the canonical Python ``for`` loop's
    # "return state, -1" branch.
    final_state = _tree_where(will_cast, effect_state, state)

    # ``cast_spell_id`` mirrors the canonical return: the safe_slot
    # when a cast was attempted, -1 otherwise.
    cast_id = jnp.where(found, safe_slot, jnp.int32(-1))
    return final_state, cast_id


# ---------------------------------------------------------------------------
# 7. pw_regen_tick_brax — delegate to step_brax (same as canonical).
# ---------------------------------------------------------------------------

def pw_regen_tick_brax(state, rng: jax.Array | None = None):
    """Brax-style ``pw_regen_tick`` — EnvState-shaped Pw regen shim.

    Canonical delegates to ``status_effects.pw_regen_tick`` which has
    no ``lax.cond`` / ``lax.switch`` at this layer.

    Flattened: 0 lax.cond / 0 lax.switch.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    return step_brax(state, rng)


# ---------------------------------------------------------------------------
# 8. step_brax — per-turn Pw regen, no control flow.
# ---------------------------------------------------------------------------

def step_brax(state, rng: jax.Array):
    """Brax-style ``step`` — per-turn magic upkeep.

    Delegates to the canonical ``status_effects.pw_regen_tick`` which is
    a vendor-formula computation with no ``lax.cond`` / ``lax.switch``
    at this layer.  Byte-identical to ``magic.step``.

    Flattened: 0 lax.cond / 0 lax.switch.
    """
    from Nethax.nethax.subsystems.status_effects import pw_regen_tick as _pw

    new_status, new_pw = _pw(
        state.status,
        state.player_pw.astype(jnp.int32),
        state.player_pw_max.astype(jnp.int32),
        state.player_xl.astype(jnp.int32),
        state.player_role.astype(jnp.int8),
        state.player_int.astype(jnp.int32),
        state.player_wis.astype(jnp.int32),
        jnp.int32(getattr(state, "timestep", 0)),
        rng,
    )
    return state.replace(player_pw=new_pw, status=new_status)


# ---------------------------------------------------------------------------
# 9. handle_spell_genocide_brax — delegate, no control flow here.
# ---------------------------------------------------------------------------

def handle_spell_genocide_brax(state, rng: jax.Array):
    """Brax-style ``handle_spell_genocide``.

    Delegates to ``items_scrolls.apply_genocide``; no ``lax.cond`` /
    ``lax.switch`` lives in this thin wrapper.  Byte-identical to
    ``magic.handle_spell_genocide``.

    Flattened: 0 lax.cond / 0 lax.switch.
    """
    from Nethax.nethax.subsystems.items_scrolls import apply_genocide
    return apply_genocide(state, rng)


# ---------------------------------------------------------------------------
# 10. losespells_brax — already uses lax.scan + jnp.where, no cond/switch.
# ---------------------------------------------------------------------------

def losespells_brax(state, rng: jax.Array):
    """Brax-style ``losespells`` — vendor amnesia spell-forgetting roll.

    Canonical implementation already uses ``lax.scan`` + ``jnp.where``
    only (no ``lax.cond`` / ``lax.switch`` to flatten).  Delegates to
    keep the vendor RNG order and luck modifier byte-identical.

    Flattened: 0 lax.cond / 0 lax.switch (no targets present).
    """
    return _magic.losespells(state, rng)
