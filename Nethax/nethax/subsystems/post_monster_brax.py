"""Brax-style rewrite of ``env._post_monster_body`` / ``_post_monster_jit_impl``.

Background
----------
Under ``jax.vmap`` (multi-seed/multi-env rollouts), ``jax.lax.cond`` lowers
to ``select`` and inlines *both* branches into the HLO graph.  Repeated
across the per-step phases this produces pathological HLO blow-up.  The
Brax pattern (Google's brax + Craftax —
https://github.com/MichaelTMatthews/Craftax) sidesteps this by always
computing both arms eagerly and selecting results leaf-by-leaf via
``jnp.where``.  The resulting HLO is flat and fusion-friendly.

This module hosts ``_post_monster_body_brax`` — a 1:1 byte-parity
replacement for ``env._post_monster_jit_impl`` rewritten in the Brax
style.

Audit of the original
---------------------
Reading ``env._post_monster_body`` (env.py:1564-1809) we find:

  * Zero ``jax.lax.cond`` calls *in the body itself*.
  * Zero ``jax.lax.switch`` calls *in the body itself*.
  * Zero ``jax.lax.fori_loop`` calls *in the body itself*.
  * Every conditional dataflow inside the body is already expressed via
    ``jnp.where`` over precomputed masks (ball-as-trap-escape, age_spells
    confusion-doubled decay, calendar moonphase tick, intervene
    rndcurse / aggravate / nasty / resurrect effects).

  All sub-helper subsystems invoked from the body (``_status_step``,
  ``_polymorph_step``, ``_shop_step``, ``maybe_ascend``, ``_digest_tick``,
  ``_newexplevel``, ``_tick_*``) *may* contain their own ``lax.cond`` /
  ``lax.switch`` internals.  Per the Brax-pass convention used elsewhere
  in this tree (see ``dispatch_action_brax.py`` "What is NOT touched"
  note), each subsystem's INTERNAL control flow is the target of a
  separate per-subsystem Brax pass — flattening them here would couple
  this module to every helper's implementation detail and double-trace
  their bodies.

Reading the wrapper ``env._post_monster_jit_impl`` (env.py:1812-1843) we
find exactly ONE ``jax.lax.cond`` — the ``state.done`` short-circuit
that returns the pre-step ``state`` unchanged when the env is already
terminal.  Under vmap this cond inlines BOTH arms (the entire
post-monster phase + the identity arm), which is precisely the HLO
blow-up we are paying for.  Flattening this single cond is the primary
goal of this file.

What is flattened
-----------------
* The ``jax.lax.cond(state.done, identity, body)`` short-circuit in
  ``_post_monster_jit_impl`` →  ``jax.tree.map(jnp.where(state.done,
  state_leaf, body_leaf))`` over the full ``EnvState`` pytree.

Total ``lax.cond / lax.switch / lax.fori_loop`` flattened in this file:
**1** (the outer done short-circuit).  The body itself was already
Brax-shaped at the top level; further flattening lives in the
sub-subsystem brax modules.

Byte-parity contract
--------------------
1.  RNG draw order is preserved by calling the body unconditionally with
    the SAME ``(rng_status, rng_poly, rng_shop, rng_swallow, rng_explvl)``
    tuple the canonical wrapper supplies.  Every sub-helper consumes its
    rng identically to the canonical path; the unselected (done==True)
    output is discarded leaf-wise via ``jnp.where`` so observable rng
    bytes are unchanged.
2.  Mutations are byte-identical: when ``state.done`` is False the
    ``jnp.where`` cascade selects the body's output leaf, when True it
    selects the pre-step ``state`` leaf — bit-for-bit identical to
    ``lax.cond``'s ``select`` lowering.
3.  The state pytree shape (every leaf, every dtype, every shape) is
    preserved because the body returns the same ``EnvState`` pytree
    shape as the input ``state``.

Signature
---------
``_post_monster_body_brax(ns, state, prev_wizard_alive, rng_status,
rng_poly, rng_shop, rng_swallow, rng_explvl) -> EnvState`` — identical
to ``env._post_monster_jit_impl`` so the function can drop into
``env._step_impl`` with a single import swap.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Import the helper subsystems exactly as env.py does so the unconditional
# evaluation here calls bit-identical code paths.
from Nethax.nethax.subsystems.status_effects import step as _status_step
from Nethax.nethax.subsystems.status_effects import (
    tick_hallu_expiry as _tick_hallu_expiry,
)
from Nethax.nethax.subsystems.status_effects import (
    tick_luck_drift as _tick_luck_drift,
)
from Nethax.nethax.subsystems.status_effects import (
    tick_slime_cancels_stoning as _tick_slime_cancels_stoning,
)
from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS
from Nethax.nethax.subsystems.timer_queue import tick_timers as _tick_timer_queue
from Nethax.nethax.subsystems.occupation import tick_occupation as _tick_occupation
from Nethax.nethax.subsystems.ascension import maybe_ascend
from Nethax.nethax.subsystems.polymorph import step as _polymorph_step
from Nethax.nethax.subsystems.shop import shop_step as _shop_step
from Nethax.nethax.subsystems.riding import tick_gallop as _tick_gallop
from Nethax.nethax.subsystems.riding import tick_saddle as _tick_saddle
from Nethax.nethax.subsystems.swallow import digest_tick as _digest_tick
from Nethax.nethax.subsystems.experience import newexplevel as _newexplevel


# Vendor permonst id of the Wizard of Yendor.  See vendor/include/monsters.h
# and the original env.py:1728 definition (``_PM_WIZARD_ENTRY``).
_PM_WIZARD_ENTRY = jnp.int32(281)

# Nasty-summon monster pool.  10-monster slice across demons/devils/giant
# beasts taken verbatim from env.py:1765-1767.
_NASTY_POOL = jnp.array(
    [297, 299, 300, 301, 302, 307, 150, 49, 182, 234], dtype=jnp.int16
)


def _post_monster_body_core(
    ns,
    state,
    prev_wizard_alive,
    rng_status,
    rng_poly,
    rng_shop,
    rng_swallow,
    rng_explvl,
):
    """Phase-3..8a body — branchless port of ``env._post_monster_body``.

    Cite parity: env.py:1564-1809.  See module docstring for the audit.
    """
    # --- Phase 3: turn counters ---------------------------------------
    # env.py:1586-1595
    ns = ns.replace(timestep=ns.timestep + jnp.int32(1))
    ns = ns.replace(
        game_moves=ns.game_moves
        + jnp.where(ns.action_consumed_turn, jnp.int32(1), jnp.int32(0))
    )

    # --- Phase 4: status-effect tick ----------------------------------
    # env.py:1602-1607
    ns = _tick_hallu_expiry(ns)
    ns = _tick_slime_cancels_stoning(ns)
    # env.py:1608-1624
    new_status, new_hp, new_pw, new_done = _status_step(
        ns.status,
        rng_status,
        ns.player_hp,
        ns.player_hp_max,
        ns.player_pw,
        ns.player_pw_max,
        ns.player_xl,
        ns.player_role,
        ns.done,
    )
    ns = ns.replace(
        status=new_status,
        player_hp=new_hp,
        player_pw=new_pw,
        done=new_done,
    )

    # --- Phase 4a-riding: gallop + saddle wear ------------------------
    # env.py:1631-1632
    ns = _tick_gallop(ns)
    ns = _tick_saddle(ns)

    # --- Phase 4a0: luck drift ---------------------------------------
    # env.py:1638
    ns = _tick_luck_drift(ns)

    # --- Phase 4a1: timer queue drain --------------------------------
    # env.py:1644
    ns = _tick_timer_queue(ns)

    # --- Phase 4a2: multi-turn occupation tick -----------------------
    # env.py:1650
    ns = _tick_occupation(ns)

    # --- Phase 4a3: ball-as-trap-escape ------------------------------
    # env.py:1659-1675 — already branchless via jnp.where.
    rng_ball_esc, _rng_after = jax.random.split(
        jax.random.fold_in(rng_status, jnp.int32(0xBA77)), 2
    )
    ball_roll = jax.random.randint(rng_ball_esc, (), 0, 4, dtype=jnp.int32)
    can_drop_escape = (
        ns.is_punished & ns.player_in_trap & (ball_roll == jnp.int32(0))
    )
    ns = ns.replace(
        player_in_trap=jnp.where(
            can_drop_escape, jnp.bool_(False), ns.player_in_trap
        ),
        player_trap_timer=jnp.where(
            can_drop_escape, jnp.int16(0), ns.player_trap_timer
        ),
        ball_pos=jnp.where(can_drop_escape, ns.player_pos, ns.ball_pos),
        ball_thrown_pos=jnp.where(
            can_drop_escape, ns.player_pos, ns.ball_thrown_pos
        ),
        ball_thrown_turns=jnp.where(
            can_drop_escape, jnp.int8(1), ns.ball_thrown_turns
        ),
    )
    # Decay ball_thrown_turns (env.py:1671-1675).
    ns = ns.replace(
        ball_thrown_turns=jnp.maximum(
            ns.ball_thrown_turns - jnp.int8(1), jnp.int8(0)
        ).astype(jnp.int8),
    )

    # --- Phase 4a: experience level promotion ------------------------
    # env.py:1680
    ns = _newexplevel(ns, rng_explvl)

    # --- Phase 4b: swallow/engulf digestion --------------------------
    # env.py:1683
    ns = _digest_tick(ns, rng_swallow)

    # --- Phase 5: were/polymorph timer tick --------------------------
    # env.py:1688
    ns = _polymorph_step(ns, rng_poly)

    # --- Phase 6: age_spells -----------------------------------------
    # env.py:1694-1699 — already branchless via jnp.where.
    magic = ns.magic
    is_confused = ns.status.timed_statuses[int(_TS.CONFUSION)] > jnp.int32(0)
    decrement = jnp.where(is_confused, jnp.int32(2), jnp.int32(1))
    new_mem = jnp.maximum(magic.spell_memory - decrement, jnp.int32(0))
    ns = ns.replace(magic=magic.replace(spell_memory=new_mem))

    # --- Phase 6b: moonphase calendar tick ---------------------------
    # env.py:1705-1711
    moon_advance = (ns.timestep % jnp.int32(250)) == jnp.int32(0)
    new_moon = jnp.where(
        moon_advance,
        (ns.calendar_moonphase.astype(jnp.int32) + jnp.int32(1)) % jnp.int32(4),
        ns.calendar_moonphase.astype(jnp.int32),
    ).astype(jnp.int8)
    ns = ns.replace(calendar_moonphase=new_moon)

    # --- Phase 7: shop tick ------------------------------------------
    # env.py:1715
    ns = _shop_step(ns, rng_shop)

    # --- Phase 8: ascension / endgame --------------------------------
    # env.py:1718
    ns = maybe_ascend(ns)

    # --- Phase 8a: Wizard intervene() --------------------------------
    # env.py:1728-1807 — already branchless via jnp.where; bit-faithful
    # transcription preserving rng split order (rng_iv → rng_iv2 → rng_iv3
    # → rng_iv4) so the rndcurse/aggravate/nasty/resurrect draws match.
    wiz_now = jnp.any(
        ns.monster_ai.alive
        & (ns.monster_ai.entry_idx.astype(jnp.int32) == _PM_WIZARD_ENTRY)
    )
    wiz_just_died = prev_wizard_alive & ~wiz_now

    rng_iv, rng_iv2 = jax.random.split(rng_status, 2)
    which = jax.random.randint(rng_iv, (), 0, 6, dtype=jnp.int32)

    # Effect 2: rndcurse.
    is_curse = wiz_just_died & (which == jnp.int32(2))
    items = ns.inventory.items
    n_slots = items.category.shape[0]
    cur_slot = jax.random.randint(rng_iv2, (), 0, n_slots, dtype=jnp.int32)
    occupied = items.category[cur_slot] != jnp.int8(0)
    new_buc = jnp.where(
        is_curse & occupied, jnp.int8(1), items.buc_status[cur_slot]
    )
    new_items = items.replace(
        buc_status=items.buc_status.at[cur_slot].set(new_buc)
    )

    # Effect 3: aggravate.
    is_aggr = wiz_just_died & (which == jnp.int32(3))
    mai = ns.monster_ai
    new_asleep = jnp.where(is_aggr, jnp.zeros_like(mai.asleep), mai.asleep)
    new_sleep_t = jnp.where(
        is_aggr, jnp.zeros_like(mai.sleep_timer), mai.sleep_timer
    )

    # Effect 4: nasty.
    is_nasty = wiz_just_died & (which == jnp.int32(4))
    rng_iv3, rng_iv4 = jax.random.split(rng_iv2, 2)
    nasty_pick_roll = jax.random.randint(
        rng_iv3, (), 0, _NASTY_POOL.shape[0], dtype=jnp.int32
    )
    nasty_entry = _NASTY_POOL[nasty_pick_roll]
    # NOTE: dead_slots is computed but unused in the canonical body
    # (env.py:1772).  Preserved here verbatim so the trace tree is
    # structurally identical.
    dead_slots = ~new_asleep & ~mai.alive  # logic-only; matches env.py:1772
    del dead_slots
    dead_mask = ~mai.alive
    any_dead = jnp.any(dead_mask)
    dead_idx = jnp.argmax(dead_mask.astype(jnp.int32)).astype(jnp.int32)
    do_nasty = is_nasty & any_dead

    nasty_alive = jnp.where(do_nasty, jnp.bool_(True), mai.alive[dead_idx])
    nasty_entry_v = jnp.where(
        do_nasty, nasty_entry, mai.entry_idx[dead_idx]
    )
    nasty_hp = jnp.where(do_nasty, jnp.int32(20), mai.hp[dead_idx])
    nasty_hpmax = jnp.where(do_nasty, jnp.int32(20), mai.hp_max[dead_idx])

    # Effect 5: resurrect Wizard.
    is_resurr = wiz_just_died & (which == jnp.int32(5))
    do_resurr = is_resurr & any_dead
    resurr_alive = jnp.where(do_resurr, jnp.bool_(True), nasty_alive)
    resurr_entry = jnp.where(do_resurr, jnp.int16(281), nasty_entry_v)
    resurr_hp = jnp.where(do_resurr, jnp.int32(50), nasty_hp)
    resurr_hpmax = jnp.where(do_resurr, jnp.int32(50), nasty_hpmax)

    new_alive_arr = mai.alive.at[dead_idx].set(resurr_alive)
    new_entry_arr = mai.entry_idx.at[dead_idx].set(resurr_entry)
    new_hp_arr = mai.hp.at[dead_idx].set(resurr_hp)
    new_hpmax_arr = mai.hp_max.at[dead_idx].set(resurr_hpmax)

    ns = ns.replace(
        inventory=ns.inventory.replace(items=new_items),
        monster_ai=mai.replace(
            asleep=new_asleep,
            sleep_timer=new_sleep_t,
            alive=new_alive_arr,
            entry_idx=new_entry_arr,
            hp=new_hp_arr,
            hp_max=new_hpmax_arr,
        ),
    )
    # rng_iv4 is reserved by the canonical body for future intervene
    # effects; reference it explicitly so the live-key tracking matches.
    del rng_iv4
    return ns


def _post_monster_body_brax(
    ns,
    state,
    prev_wizard_alive,
    rng_status,
    rng_poly,
    rng_shop,
    rng_swallow,
    rng_explvl,
):
    """Brax-style port of ``env._post_monster_jit_impl``.

    Signature matches the canonical wrapper at env.py:1812-1843.

    Implementation: unconditionally evaluate the post-monster body
    (eager Brax-pattern), then leaf-wise ``jnp.where(state.done,
    state_leaf, body_leaf)`` over the resulting pytree.  This flattens
    the single ``lax.cond(state.done, ...)`` short-circuit in the
    canonical wrapper while preserving byte-parity (rng draw order,
    mutations, pytree shape).
    """
    body_out = _post_monster_body_core(
        ns,
        state,
        prev_wizard_alive,
        rng_status,
        rng_poly,
        rng_shop,
        rng_swallow,
        rng_explvl,
    )
    # Leaf-wise select equivalent to lax.cond's lowering.
    return jax.tree.map(
        lambda done_leaf, body_leaf: jnp.where(state.done, done_leaf, body_leaf),
        state,
        body_out,
    )
