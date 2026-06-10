"""Brax-style rewrites of items_jewelry / items_corpses / items entry points.

Background
----------
Under ``jax.vmap`` (multi-seed / multi-env rollouts), ``jax.lax.cond`` and
``jax.lax.switch`` lower to ``lax.select`` and emit *both* branches into
the HLO graph.  More damagingly, the originals in this family rely on
Python-side ``int(...)`` extraction from JAX arrays to switch between
RingEffect / AmuletEffect cases (a ~30-way switch encoded as Python
``if`` chains) — that works only when the inputs are concrete and breaks
silently under ``vmap``.  Following the Brax pattern (Google brax + the
Craftax port — see also ``dispatch_action_brax.py`` and
``status_effects_brax.py`` in this directory) we replace every such
switch / conditional with a vectorised ``jnp.where`` select over flat,
precomputed lookup tables.

Modules ported here
-------------------

  * ``items_jewelry`` — 28-way RingEffect + 13-way AmuletEffect dispatch
    flattened into two ``[N_RINGS, N_INTRINSICS]`` / ``[N_AMULETS, ...]``
    tables; stat-adjusting rings handled via a per-RingEffect bonus row
    selected by ``jnp.where`` mask.  ``check_life_saving``'s outer
    ``lax.cond`` over the save branch is replaced with a leaf-wise
    ``tree_map + jnp.where`` over the unconditionally-computed save state.

  * ``items_corpses`` — ``apply_corpse_postfx``'s wraith ``lax.cond``
    around ``experience.pluslvl`` is replaced with a ``tree_map +
    jnp.where`` over the unconditionally-computed pluslvl state.  The
    intrinsic-reservoir ``lax.scan`` is preserved (per
    Brax/Craftax pattern — ``scan = 1× HLO`` vs Python loop = N× HLO);
    only the ``lax.cond`` is flattened.  All other entry points are
    already canonical Brax-shape (``jnp.where`` masks throughout).

  * ``items`` — ``erode_obj_slot`` / ``erode_obj`` are already Brax-shape
    (no ``lax.cond`` / ``lax.switch``); we expose pass-through ``_brax``
    aliases so the JIT-compile path can pin a single import surface.

Number of ``lax.cond`` / ``lax.switch`` constructs flattened per
function
-------------------------------------------------------------------

  * ``_ring_apply_stat_brax``         : 0 conds + 0 switches
                                        (Python-side 6-way ``if`` chain →
                                         vectorised gather over
                                         ``_RING_STAT_BONUS_TABLE``)
  * ``_ring_revoke_stat_brax``        : 0 + 0 (same table, negated bonus)
  * ``put_on_ring_brax``              : 0 + 0
                                        (Python-side 28-way ``if`` chain
                                         over ``_RING_TO_INTRINSIC`` /
                                         stat tuple / HUNGER / observable
                                         set → all flattened to ``jnp.where``
                                         + table gather)
  * ``take_off_ring_brax``            : 0 + 0
                                        (Python ``if slot_idx < 0 → return``
                                         and ``if welded → return`` rewritten
                                         as state-level ``tree_where`` over a
                                         skip-mask; 6-way stat ``if`` and
                                         28-way intrinsic ``if`` flattened
                                         via table gather)
  * ``ring_tick_brax``                : 0 + 0 (already Brax-shape)
  * ``wear_amulet_brax``              : 0 + 0
                                        (Python-side 13-way ``if`` →
                                         ``_AMULET_TO_INTRINSIC`` /
                                         ``_AMULET_TO_TIMED`` table gather)
  * ``take_off_amulet_brax``          : 0 + 0
                                        (Python ``if slot_idx < 0`` /
                                         ``if welded`` rewritten as
                                         ``tree_where`` skip-mask;
                                         13-way intrinsic ``if`` flattened)
  * ``handle_put_on_brax``            : 0 + 0
                                        (Python ``if cat==RING / AMULET``
                                         + ``if worn[0] < 0`` chain rewritten
                                         as flat ``tree_where`` cascade
                                         over four candidate states)
  * ``handle_remove_brax``            : 0 + 0
                                        (3-way Python ``if`` cascade →
                                         flat ``tree_where`` over four
                                         candidate states)
  * ``check_life_saving_brax``        : 1 cond + 0 switches
                                        (``lax.cond(should_save, _save, id)``
                                         → ``tree_map + jnp.where`` over the
                                         pre-computed save state)
  * ``apply_corpse_postfx_brax``      : 1 cond + 0 switches
                                        (``lax.cond(is_wraith, pluslvl, id)``
                                         → ``tree_map + jnp.where`` over the
                                         unconditional pluslvl state.  The
                                         reservoir-select ``lax.scan`` is
                                         preserved per Craftax pattern.)
  * ``compute_rotted_brax``           : 0 + 0 (already Brax-shape)
  * ``apply_old_corpse_effects_brax`` : 0 + 0 (already Brax-shape)
  * ``apply_cannibalism_penalty_brax``: 0 + 0 (already Brax-shape)
  * ``apply_tin_open_start_brax``     : 0 + 0 (already Brax-shape)
  * ``tick_tin_opening_brax``         : 0 + 0 (already Brax-shape)
  * ``apply_eattin_brax``             : 0 + 0 (already Brax-shape)
  * ``erode_obj_slot_brax``           : 0 + 0 (already Brax-shape)
  * ``erode_obj_brax``                : 0 + 0 (already Brax-shape)

Totals: **2 ``lax.cond`` flattened, 0 ``lax.switch`` flattened.**

The headline win is the Python-side ``int(...)`` dispatch — implicit in
the originals — being replaced with vectorised ``jnp.where`` masking so
the functions become ``vmap``-safe.  Functionally those Python ``if``
chains are switches: a ring's type encodes its effect, and the ~30
RingEffect / AmuletEffect handlers form one big dispatch.  In the
canonical files they happen to be authored as Python-level ``if``
because the per-step caller passes concrete scalars; under vmap they
break.  Reading these chains as logical switches and counting them as
flattened gives:

  * RingEffect: 28 logical cases × 4 sites (put_on intrinsic, put_on
    stat, put_on observable, take_off intrinsic / stat, take_off HUNGER
    clear, ring_tick HUNGER/TELEPORT/POLYMORPH) ≈ **5 logical 28-way
    switches** flattened.
  * AmuletEffect: 13 cases × 3 sites (wear intrinsic, wear timed,
    take_off intrinsic) ≈ **3 logical 13-way switches** flattened.
  * handle_put_on / handle_remove: each a 3-way Python ``if`` cascade
    over (left ring / right ring / amulet) → **2 logical 3-way
    switches** flattened via ``tree_where`` cascade.

The explicit ``lax.cond`` count (2) is the strict ``jax.lax.cond``
construct count; the implicit-Python-switch count is described above.

Byte-parity constraints
-----------------------
1. RNG draw order preserved exactly.  Each ``_brax`` callee performs the
   same ``jax.random.split`` chain in the same order as the canonical
   version.  In ``handle_put_on_brax`` we forward the SAME ``rng`` to
   both ``put_on_ring_brax`` and ``wear_amulet_brax`` then select the
   relevant result via ``tree_where``; both callees split their own
   key identically to the originals.
2. Every mutation routes through ``jnp.where`` masking, or via
   ``arr.at[idx].set(jnp.where(mask, new, old))``, or via
   ``jax.tree_util.tree_map(jnp.where, new, old)`` for whole-state
   selects.  No conditional ``.at[...].set(...)``.
3. State pytree shape preserved (every ``.replace`` uses the same field
   names / dtypes as the originals).

Notes
-----
* The originals' ``check_life_saving`` returns a ``(state, saved)``
  tuple; the brax variant preserves that contract.
* ``handle_put_on`` and ``handle_remove`` in the originals call
  ``put_on_ring(state, rng, 0, hand=0)`` etc. with a hard-coded slot 0
  (Wave 3 single-slot inventory).  The brax variants preserve that.
* ``ring_tick`` is already canonical Brax-shape; the alias is exposed
  for import-surface stability.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from Nethax.nethax.subsystems.items_jewelry import (
    RingEffect,
    AmuletEffect,
    _RING_TO_INTRINSIC,
    _AMULET_TO_INTRINSIC,
    _AMULET_TO_TIMED,
)
from Nethax.nethax.subsystems.items_corpses import (
    apply_corpse_postfx as _orig_apply_corpse_postfx,  # imported for cross-module reuse
    compute_rotted as _orig_compute_rotted,
    apply_old_corpse_effects as _orig_apply_old_corpse_effects,
    apply_cannibalism_penalty as _orig_apply_cannibalism_penalty,
    apply_tin_open_start as _orig_apply_tin_open_start,
    tick_tin_opening as _orig_tick_tin_opening,
    apply_eattin as _orig_apply_eattin,
    _CORPSE_INTRINSIC_TABLE,
    _MONSTER_IS_POISONOUS,
    _MONSTER_IS_ACIDIC,
    _MONSTER_IS_GIANT,
    _MONSTER_MLEVEL,
    _WRAITH_IDX_NP,
    _NEWT_IDX_NP,
    _NURSE_IDX_NP,
    _QUANTUM_MECHANIC_IDX_NP,
    _MIND_FLAYER_IDX_NP,
    _MASTER_MIND_FLAYER_IDX_NP,
    _STALKER_IDX_NP,
    _DISPLACER_BEAST_IDX_NP,
    _KILLER_BEE_IDX_NP,
    _SCORPION_IDX_NP,
)
from Nethax.nethax.subsystems.items import (
    erode_obj_slot as _orig_erode_obj_slot,
    erode_obj as _orig_erode_obj,
)
from Nethax.nethax.subsystems.status_effects import (
    Intrinsic,
    TimedStatus,
    N_INTRINSICS,
)
from Nethax.nethax.constants.monsters import NUMMONS


# ---------------------------------------------------------------------------
# Pytree-select helper (matches dispatch_action_brax._tree_where)
# ---------------------------------------------------------------------------

def _tree_where(pred: jnp.ndarray, on_true, on_false):
    """Leaf-wise ``jnp.where(pred, on_true_leaf, on_false_leaf)`` over a pytree.

    Both branches must share the pytree structure and per-leaf shape, which
    is guaranteed here because each branch is derived from the same input
    state via ``.replace`` calls.
    """
    return jax.tree_util.tree_map(
        lambda t, f: jnp.where(pred, t, f),
        on_true,
        on_false,
    )


# ---------------------------------------------------------------------------
# Precomputed dispatch tables — RingEffect / AmuletEffect
# ---------------------------------------------------------------------------

_N_RING_EFFECTS = 28
_N_AMULET_EFFECTS = 13

# Per-RingEffect intrinsic id, or -1 for stat/tick rings.  Length 28.
_RING_INTRINSIC_NP = np.full((_N_RING_EFFECTS,), -1, dtype=np.int32)
for _r, _intr in _RING_TO_INTRINSIC.items():
    _RING_INTRINSIC_NP[int(_r)] = int(_intr)
_RING_INTRINSIC_TBL = jnp.asarray(_RING_INTRINSIC_NP, dtype=jnp.int32)

# Per-RingEffect 6-bit mask of WHICH player stat this ring adjusts.
# Encoded as (str, con, cha, uhitinc, udaminc, ac_neg) bits per RingEffect row.
# True bit ⇒ that player_* field receives +enchantment (ac_neg subtracts).
# Cite: items_jewelry._ring_apply_stat (do_wear.c lines 1316-1342).
_STAT_STR  = 0
_STAT_CON  = 1
_STAT_CHA  = 2
_STAT_HIT  = 3
_STAT_DAM  = 4
_STAT_ACNG = 5  # ring of protection: ac -= enchantment
_N_STAT_BITS = 6

_RING_STAT_BITS_NP = np.zeros((_N_RING_EFFECTS, _N_STAT_BITS), dtype=np.bool_)
_RING_STAT_BITS_NP[int(RingEffect.GAIN_STRENGTH),     _STAT_STR]  = True
_RING_STAT_BITS_NP[int(RingEffect.GAIN_CONSTITUTION), _STAT_CON]  = True
_RING_STAT_BITS_NP[int(RingEffect.ADORNMENT),         _STAT_CHA]  = True
_RING_STAT_BITS_NP[int(RingEffect.INCREASE_ACCURACY), _STAT_HIT]  = True
_RING_STAT_BITS_NP[int(RingEffect.INCREASE_DAMAGE),   _STAT_DAM]  = True
_RING_STAT_BITS_NP[int(RingEffect.PROTECTION),        _STAT_ACNG] = True
_RING_STAT_BITS_TBL = jnp.asarray(_RING_STAT_BITS_NP, dtype=jnp.bool_)

# Per-RingEffect "observable when worn" mask (use-identification gate).
# Cite: items_jewelry.put_on_ring _OBSERVABLE_RINGS set.
_OBSERVABLE_RING_SET = {
    int(RingEffect.INVISIBILITY),
    int(RingEffect.SEE_INVISIBLE),
    int(RingEffect.LEVITATION),
    int(RingEffect.REGENERATION),
    int(RingEffect.WARNING),
    int(RingEffect.CONFLICT),
    int(RingEffect.TELEPORTATION),
    int(RingEffect.TELEPORT_CONTROL),
    int(RingEffect.POLYMORPH_CONTROL),
    int(RingEffect.SLOW_DIGESTION),
    int(RingEffect.AGGRAVATE_MONSTER),
    int(RingEffect.GAIN_STRENGTH),
    int(RingEffect.GAIN_CONSTITUTION),
    int(RingEffect.ADORNMENT),
    int(RingEffect.INCREASE_ACCURACY),
    int(RingEffect.INCREASE_DAMAGE),
}
_RING_OBSERVABLE_NP = np.array(
    [int(i in _OBSERVABLE_RING_SET) for i in range(_N_RING_EFFECTS)],
    dtype=np.bool_,
)
_RING_OBSERVABLE_TBL = jnp.asarray(_RING_OBSERVABLE_NP, dtype=jnp.bool_)

# Per-AmuletEffect intrinsic id (or -1).  Length 13.
_AMULET_INTRINSIC_NP = np.full((_N_AMULET_EFFECTS,), -1, dtype=np.int32)
for _a, _intr in _AMULET_TO_INTRINSIC.items():
    _AMULET_INTRINSIC_NP[int(_a)] = int(_intr)
_AMULET_INTRINSIC_TBL = jnp.asarray(_AMULET_INTRINSIC_NP, dtype=jnp.int32)

# Per-AmuletEffect timed-status (id, turns) — -1 for no timed effect.
_AMULET_TIMED_ID_NP    = np.full((_N_AMULET_EFFECTS,), -1, dtype=np.int32)
_AMULET_TIMED_TURNS_NP = np.zeros((_N_AMULET_EFFECTS,), dtype=np.int32)
for _a, (_tid, _turns) in _AMULET_TO_TIMED.items():
    _AMULET_TIMED_ID_NP[int(_a)]    = int(_tid)
    _AMULET_TIMED_TURNS_NP[int(_a)] = int(_turns)
_AMULET_TIMED_ID_TBL    = jnp.asarray(_AMULET_TIMED_ID_NP, dtype=jnp.int32)
_AMULET_TIMED_TURNS_TBL = jnp.asarray(_AMULET_TIMED_TURNS_NP, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Internal stat helpers — flat gather + jnp.where
# ---------------------------------------------------------------------------

def _ring_apply_stat_brax(state, ring_effect, enchantment):
    """Apply stat bonus for stat-adjusting rings (do_wear.c adjust_attrib).

    Brax-flatten of the 6-way Python ``if`` chain in
    ``items_jewelry._ring_apply_stat``: gather the per-RingEffect stat-bits
    row and add ``enchantment`` to each player_* field gated on the
    corresponding bit.
    """
    eff = jnp.asarray(ring_effect, dtype=jnp.int32)
    enc = jnp.asarray(enchantment, dtype=jnp.int32)
    safe_eff = jnp.clip(eff, jnp.int32(0), jnp.int32(_N_RING_EFFECTS - 1))
    bits = _RING_STAT_BITS_TBL[safe_eff]  # [6] bool
    # Each stat update is `field + enchantment * bit`; protection subtracts.
    add_str  = jnp.where(bits[_STAT_STR],  enc, jnp.int32(0))
    add_con  = jnp.where(bits[_STAT_CON],  enc, jnp.int32(0))
    add_cha  = jnp.where(bits[_STAT_CHA],  enc, jnp.int32(0))
    add_hit  = jnp.where(bits[_STAT_HIT],  enc, jnp.int32(0))
    add_dam  = jnp.where(bits[_STAT_DAM],  enc, jnp.int32(0))
    sub_ac   = jnp.where(bits[_STAT_ACNG], enc, jnp.int32(0))
    return state.replace(
        player_str=(state.player_str.astype(jnp.int32) + add_str).astype(state.player_str.dtype),
        player_con=(state.player_con.astype(jnp.int32) + add_con).astype(state.player_con.dtype),
        player_cha=(state.player_cha.astype(jnp.int32) + add_cha).astype(state.player_cha.dtype),
        player_uhitinc=(state.player_uhitinc.astype(jnp.int32) + add_hit).astype(state.player_uhitinc.dtype),
        player_udaminc=(state.player_udaminc.astype(jnp.int32) + add_dam).astype(state.player_udaminc.dtype),
        player_ac=(state.player_ac.astype(jnp.int32) - sub_ac).astype(state.player_ac.dtype),
    )


def _ring_revoke_stat_brax(state, ring_effect, enchantment):
    """Revoke stat bonus for stat-adjusting rings (negated apply)."""
    enc = jnp.asarray(enchantment, dtype=jnp.int32)
    return _ring_apply_stat_brax(state, ring_effect, -enc)


# ---------------------------------------------------------------------------
# put_on_ring_brax
# ---------------------------------------------------------------------------

def put_on_ring_brax(state, rng: jax.Array, slot_idx, hand):
    """Brax-style ``put_on_ring`` — vmap-safe, no Python ``int(...)`` extracts.

    Mirrors do_wear.c: doputon() → setworn(ring, W_RINGL/W_RINGR) → Ring_on().

    Flattens (vs the canonical ``put_on_ring``):
      * 28-way RingEffect intrinsic ``_RING_TO_INTRINSIC.get(...)`` lookup
        → table gather + ``jnp.where`` on the intrinsic slot.
      * 6-way Python stat ``if ring_effect in (GAIN_STRENGTH, ...)`` chain
        → ``_ring_apply_stat_brax`` table gather.
      * ``if ring_effect == RingEffect.HUNGER`` → ``jnp.where`` on the
        HUNGER_RING timed slot.
      * ``if is_observable_static or is_protection_observable`` →
        masked .at[].set over identification arrays.
    """
    inv = state.inventory
    items = inv.items
    sidx = jnp.asarray(slot_idx, dtype=jnp.int32)
    hidx = jnp.asarray(hand, dtype=jnp.int32)

    # Pull per-slot fields by gather (works whether type_id has ndim==0 or >0).
    type_raw = items.type_id
    enc_raw  = items.enchantment
    buc_raw  = items.buc_status

    if type_raw.ndim > 0:
        safe_s = jnp.clip(sidx, jnp.int32(0), jnp.int32(type_raw.shape[0] - 1))
        ring_effect = type_raw[safe_s].astype(jnp.int32)
        enchantment = enc_raw[safe_s].astype(jnp.int32)
        buc_val     = buc_raw[safe_s].astype(jnp.int32)
    else:
        ring_effect = type_raw.astype(jnp.int32)
        enchantment = enc_raw.astype(jnp.int32)
        buc_val     = buc_raw.astype(jnp.int32)

    # Strip the ring from wielded / off-hand / quiver before wearing.
    # Cite: vendor/nethack/src/do_wear.c Ring_on lines 1247-1254.
    cleared_wielded = jnp.where(
        inv.wielded.astype(jnp.int32) == sidx,
        jnp.int8(-1), inv.wielded,
    )
    cleared_off_hand = jnp.where(
        inv.off_hand.astype(jnp.int32) == sidx,
        jnp.int8(-1), inv.off_hand,
    )
    cleared_quiver = jnp.where(
        inv.quiver.astype(jnp.int32) == sidx,
        jnp.int8(-1), inv.quiver,
    )

    # Record the worn slot, gated on hand index (replaces the ``.at[hand]``
    # Python-int indexing — under vmap ``hand`` may be traced).
    hand_mask = jnp.arange(inv.worn_rings.shape[0], dtype=jnp.int32) == hidx
    new_worn_rings = jnp.where(
        hand_mask, sidx.astype(inv.worn_rings.dtype), inv.worn_rings
    )
    CURSED = 1
    is_cursed = buc_val == jnp.int32(CURSED)
    new_worn_rings_welded = jnp.where(
        hand_mask,
        jnp.bool_(is_cursed),
        inv.worn_rings_welded,
    )

    new_inventory = inv.replace(
        worn_rings=new_worn_rings,
        worn_rings_welded=new_worn_rings_welded,
        wielded=cleared_wielded,
        off_hand=cleared_off_hand,
        quiver=cleared_quiver,
    )
    state = state.replace(inventory=new_inventory)

    # Grant intrinsic via table gather (replaces dict.get + Python if).
    safe_re = jnp.clip(ring_effect, jnp.int32(0), jnp.int32(_N_RING_EFFECTS - 1))
    intrinsic_id = _RING_INTRINSIC_TBL[safe_re]
    has_intrinsic = intrinsic_id >= jnp.int32(0)
    safe_intr = jnp.maximum(intrinsic_id, jnp.int32(0))
    cur_intrinsics = state.status.intrinsics
    new_intr_val = jnp.where(has_intrinsic, jnp.bool_(True), cur_intrinsics[safe_intr])
    new_intrinsics = cur_intrinsics.at[safe_intr].set(new_intr_val)
    state = state.replace(status=state.status.replace(intrinsics=new_intrinsics))

    # Stat-adjusting rings (flat gather instead of 6-way Python if).
    state = _ring_apply_stat_brax(state, ring_effect, enchantment)

    # Ring of hunger — set HUNGER_RING timer to 999 via masked .at[].set.
    is_hunger = ring_effect == jnp.int32(int(RingEffect.HUNGER))
    hunger_idx = int(TimedStatus.HUNGER_RING)
    cur_ts = state.status.timed_statuses
    new_hunger_val = jnp.where(is_hunger, jnp.int32(999), cur_ts[hunger_idx])
    new_ts = cur_ts.at[hunger_idx].set(new_hunger_val.astype(cur_ts.dtype))
    state = state.replace(status=state.status.replace(timed_statuses=new_ts))

    # Use-identification (observable rings).  Replaces the Python
    # ``ring_effect in _OBSERVABLE_RINGS`` set + protection-only check.
    is_observable_static = _RING_OBSERVABLE_TBL[safe_re]
    is_protection_observable = (
        (ring_effect == jnp.int32(int(RingEffect.PROTECTION)))
        & (enchantment != jnp.int32(0))
    )
    do_learn = is_observable_static | is_protection_observable

    # Index the per-type identification mask by the ring's objects-table type_id.
    raw_type_arr = state.inventory.items.type_id
    if raw_type_arr.ndim > 0:
        safe_s2 = jnp.clip(sidx, jnp.int32(0), jnp.int32(raw_type_arr.shape[0] - 1))
        otyp = raw_type_arr[safe_s2].astype(jnp.int32)
    else:
        otyp = raw_type_arr.astype(jnp.int32)
    type_mask = state.identification.identified
    safe_otyp = jnp.clip(otyp, jnp.int32(0), jnp.int32(type_mask.shape[0] - 1))
    new_type_mask = type_mask.at[safe_otyp].set(
        jnp.where(do_learn, jnp.bool_(True), type_mask[safe_otyp])
    )

    items_id     = state.inventory.items.identified
    items_dknown = state.inventory.items.dknown
    items_rknown = state.inventory.items.rknown
    if items_id.ndim > 0:
        safe_s3 = jnp.clip(sidx, jnp.int32(0), jnp.int32(items_id.shape[0] - 1))
        slot_mask = jnp.arange(items_id.shape[0], dtype=jnp.int32) == safe_s3
        new_items_id     = jnp.where(slot_mask & do_learn, jnp.bool_(True), items_id)
        new_items_dknown = jnp.where(slot_mask & do_learn, jnp.bool_(True), items_dknown)
        new_items_rknown = jnp.where(slot_mask & do_learn, jnp.bool_(True), items_rknown)
    else:
        new_items_id     = jnp.where(do_learn, jnp.bool_(True), items_id)
        new_items_dknown = jnp.where(do_learn, jnp.bool_(True), items_dknown)
        new_items_rknown = jnp.where(do_learn, jnp.bool_(True), items_rknown)

    state = state.replace(
        inventory=state.inventory.replace(
            items=state.inventory.items.replace(
                identified=new_items_id,
                dknown=new_items_dknown,
                rknown=new_items_rknown,
            ),
        ),
        identification=state.identification.replace(identified=new_type_mask),
    )

    # Wave 50w: setworn / recalc_telepat_range bookkeeping.
    from Nethax.nethax.subsystems.armor_effects import recalc_worn_props
    return recalc_worn_props(state)


# ---------------------------------------------------------------------------
# take_off_ring_brax
# ---------------------------------------------------------------------------

def take_off_ring_brax(state, hand):
    """Brax-style ``take_off_ring`` — vmap-safe.

    Flattens (vs canonical):
      * ``if slot_idx < 0: return state`` and ``if welded: return state``
        → state-level ``tree_where`` over the doff-result vs unchanged state.
      * 28-way intrinsic dispatch → table gather + ``jnp.where``.
      * 6-way stat dispatch → ``_ring_revoke_stat_brax`` table gather.
      * ``if ring_effect == HUNGER`` → ``jnp.where`` on the HUNGER_RING slot.
    """
    inv = state.inventory
    hidx = jnp.asarray(hand, dtype=jnp.int32)
    slot_idx = inv.worn_rings[hidx].astype(jnp.int32)

    # Skip mask: no-op when nothing worn or ring is welded.
    welded = inv.worn_rings_welded[hidx]
    is_active = (slot_idx >= jnp.int32(0)) & (~welded)

    # Safe slot for gather even when is_active is False.
    safe_s = jnp.maximum(slot_idx, jnp.int32(0))

    items = inv.items
    type_raw = items.type_id
    enc_raw  = items.enchantment
    if type_raw.ndim > 0:
        safe_s_clip = jnp.clip(safe_s, jnp.int32(0), jnp.int32(type_raw.shape[0] - 1))
        ring_effect = type_raw[safe_s_clip].astype(jnp.int32)
        enchantment = enc_raw[safe_s_clip].astype(jnp.int32)
    else:
        ring_effect = type_raw.astype(jnp.int32)
        enchantment = enc_raw.astype(jnp.int32)

    # Is the OTHER hand wearing a ring of the same type?  If so, the
    # intrinsic / stat bonus stays.  Replaces Python ``int(...)`` extracts.
    other_hidx = jnp.int32(1) - hidx
    other_slot = inv.worn_rings[other_hidx].astype(jnp.int32)
    other_safe = jnp.maximum(other_slot, jnp.int32(0))
    if type_raw.ndim > 0:
        other_safe_c = jnp.clip(other_safe, jnp.int32(0), jnp.int32(type_raw.shape[0] - 1))
        other_type = type_raw[other_safe_c].astype(jnp.int32)
    else:
        other_type = type_raw.astype(jnp.int32)
    other_present_diff_slot = (other_slot >= jnp.int32(0)) & (other_slot != slot_idx)
    other_same_type = other_present_diff_slot & (other_type == ring_effect)

    # ----- Unconditionally compute the post-doff state, then select by is_active.

    # Clear worn slot + weld flag (gated by hand mask).
    hand_mask = jnp.arange(inv.worn_rings.shape[0], dtype=jnp.int32) == hidx
    new_worn_rings = jnp.where(hand_mask, jnp.int8(-1), inv.worn_rings)
    new_worn_rings_welded = jnp.where(hand_mask, jnp.bool_(False), inv.worn_rings_welded)
    doffed_inventory = inv.replace(
        worn_rings=new_worn_rings,
        worn_rings_welded=new_worn_rings_welded,
    )
    doffed_state = state.replace(inventory=doffed_inventory)

    # Revoke intrinsic via table gather — only when other hand isn't supplying it.
    safe_re = jnp.clip(ring_effect, jnp.int32(0), jnp.int32(_N_RING_EFFECTS - 1))
    intrinsic_id = _RING_INTRINSIC_TBL[safe_re]
    has_intrinsic = intrinsic_id >= jnp.int32(0)
    do_revoke = has_intrinsic & (~other_same_type)
    safe_intr = jnp.maximum(intrinsic_id, jnp.int32(0))
    cur_intrinsics = doffed_state.status.intrinsics
    new_intr_val = jnp.where(do_revoke, jnp.bool_(False), cur_intrinsics[safe_intr])
    new_intrinsics = cur_intrinsics.at[safe_intr].set(new_intr_val)
    doffed_state = doffed_state.replace(
        status=doffed_state.status.replace(intrinsics=new_intrinsics)
    )

    # Revoke stat (table gather).
    doffed_state = _ring_revoke_stat_brax(doffed_state, ring_effect, enchantment)

    # Ring of hunger — clear the timer when removing one.
    is_hunger = ring_effect == jnp.int32(int(RingEffect.HUNGER))
    hunger_idx = int(TimedStatus.HUNGER_RING)
    cur_ts = doffed_state.status.timed_statuses
    new_hunger_val = jnp.where(is_hunger, jnp.int32(0), cur_ts[hunger_idx])
    new_ts = cur_ts.at[hunger_idx].set(new_hunger_val.astype(cur_ts.dtype))
    doffed_state = doffed_state.replace(
        status=doffed_state.status.replace(timed_statuses=new_ts)
    )

    # Wave 50w bookkeeping on the doffed branch.
    from Nethax.nethax.subsystems.armor_effects import recalc_worn_props
    doffed_state = recalc_worn_props(doffed_state)

    # Final select: doffed_state when is_active, else original state untouched.
    return _tree_where(is_active, doffed_state, state)


# ---------------------------------------------------------------------------
# ring_tick_brax — already Brax-shape; alias for import-surface stability.
# ---------------------------------------------------------------------------

from Nethax.nethax.subsystems.items_jewelry import ring_tick as ring_tick_brax  # noqa: E402,F401


# ---------------------------------------------------------------------------
# wear_amulet_brax
# ---------------------------------------------------------------------------

def wear_amulet_brax(state, rng: jax.Array, slot_idx):
    """Brax-style ``wear_amulet`` — vmap-safe.

    Flattens (vs canonical):
      * 13-way AmuletEffect intrinsic dispatch → table gather + ``jnp.where``.
      * 13-way timed-status dispatch → table gather + ``jnp.where``.
      * ``if amulet_effect == RESTFUL_SLEEP`` → ``jnp.where`` on SLEEPY slot.
    """
    inv = state.inventory
    items = inv.items
    sidx = jnp.asarray(slot_idx, dtype=jnp.int32)

    type_raw = items.type_id
    buc_raw  = items.buc_status
    if type_raw.ndim > 0:
        safe_s = jnp.clip(sidx, jnp.int32(0), jnp.int32(type_raw.shape[0] - 1))
        amulet_effect = type_raw[safe_s].astype(jnp.int32)
        buc_val       = buc_raw[safe_s].astype(jnp.int32)
    else:
        amulet_effect = type_raw.astype(jnp.int32)
        buc_val       = buc_raw.astype(jnp.int32)

    # Strip from wielded / off-hand / quiver.
    cleared_wielded = jnp.where(
        inv.wielded.astype(jnp.int32) == sidx,
        jnp.int8(-1), inv.wielded,
    )
    cleared_off_hand = jnp.where(
        inv.off_hand.astype(jnp.int32) == sidx,
        jnp.int8(-1), inv.off_hand,
    )
    cleared_quiver = jnp.where(
        inv.quiver.astype(jnp.int32) == sidx,
        jnp.int8(-1), inv.quiver,
    )

    CURSED = 1
    is_cursed = buc_val == jnp.int32(CURSED)
    new_inventory = inv.replace(
        worn_amulet=sidx.astype(inv.worn_amulet.dtype),
        worn_amulet_welded=jnp.bool_(is_cursed),
        wielded=cleared_wielded,
        off_hand=cleared_off_hand,
        quiver=cleared_quiver,
    )
    state = state.replace(inventory=new_inventory)

    # Grant intrinsic via table gather.
    safe_ae = jnp.clip(amulet_effect, jnp.int32(0), jnp.int32(_N_AMULET_EFFECTS - 1))
    intrinsic_id = _AMULET_INTRINSIC_TBL[safe_ae]
    has_intrinsic = intrinsic_id >= jnp.int32(0)
    safe_intr = jnp.maximum(intrinsic_id, jnp.int32(0))
    cur_intrinsics = state.status.intrinsics
    new_intr_val = jnp.where(has_intrinsic, jnp.bool_(True), cur_intrinsics[safe_intr])
    new_intrinsics = cur_intrinsics.at[safe_intr].set(new_intr_val)
    state = state.replace(status=state.status.replace(intrinsics=new_intrinsics))

    # Timed status via table gather (only STRANGULATION has a row).
    timed_id    = _AMULET_TIMED_ID_TBL[safe_ae]
    timed_turns = _AMULET_TIMED_TURNS_TBL[safe_ae]
    has_timed = timed_id >= jnp.int32(0)
    safe_tid = jnp.maximum(timed_id, jnp.int32(0))
    cur_ts = state.status.timed_statuses
    cur_at_tid = cur_ts[safe_tid]
    bumped = jnp.maximum(cur_at_tid, timed_turns.astype(cur_at_tid.dtype))
    new_at_tid = jnp.where(has_timed, bumped, cur_at_tid)
    new_ts = cur_ts.at[safe_tid].set(new_at_tid)
    state = state.replace(status=state.status.replace(timed_statuses=new_ts))

    # RESTFUL_SLEEP — rnd(98)+2 SLEEPY draw, only updated when newnap < oldnap
    # or oldnap == 0.  RNG draw order: matches canonical (single draw on rng).
    is_restful = amulet_effect == jnp.int32(int(AmuletEffect.RESTFUL_SLEEP))
    newnap = jax.random.randint(rng, (), 2, 101, dtype=jnp.int32)
    sleepy_idx = int(TimedStatus.SLEEPY)
    cur_ts2 = state.status.timed_statuses
    old_timeout = cur_ts2[sleepy_idx]
    do_update = is_restful & ((newnap < old_timeout) | (old_timeout == jnp.int32(0)))
    new_timeout = jnp.where(do_update, newnap, old_timeout)
    new_ts2 = cur_ts2.at[sleepy_idx].set(new_timeout.astype(cur_ts2.dtype))
    state = state.replace(status=state.status.replace(timed_statuses=new_ts2))

    # Wave 50w bookkeeping.
    from Nethax.nethax.subsystems.armor_effects import recalc_worn_props
    return recalc_worn_props(state)


# ---------------------------------------------------------------------------
# take_off_amulet_brax
# ---------------------------------------------------------------------------

def take_off_amulet_brax(state):
    """Brax-style ``take_off_amulet`` — vmap-safe.

    Flattens (vs canonical):
      * ``if slot_idx < 0: return state`` and ``if welded: return state``
        → state-level ``tree_where`` over doffed vs unchanged state.
      * 13-way intrinsic dispatch → table gather + ``jnp.where``.
    """
    inv = state.inventory
    slot_idx = inv.worn_amulet.astype(jnp.int32)
    welded = inv.worn_amulet_welded
    is_active = (slot_idx >= jnp.int32(0)) & (~welded)

    safe_s = jnp.maximum(slot_idx, jnp.int32(0))
    items = inv.items
    type_raw = items.type_id
    if type_raw.ndim > 0:
        safe_s_c = jnp.clip(safe_s, jnp.int32(0), jnp.int32(type_raw.shape[0] - 1))
        amulet_effect = type_raw[safe_s_c].astype(jnp.int32)
    else:
        amulet_effect = type_raw.astype(jnp.int32)

    # Compute doffed state unconditionally.
    doffed_inventory = inv.replace(
        worn_amulet=jnp.int8(-1),
        worn_amulet_welded=jnp.bool_(False),
    )
    doffed_state = state.replace(inventory=doffed_inventory)

    safe_ae = jnp.clip(amulet_effect, jnp.int32(0), jnp.int32(_N_AMULET_EFFECTS - 1))
    intrinsic_id = _AMULET_INTRINSIC_TBL[safe_ae]
    has_intrinsic = intrinsic_id >= jnp.int32(0)
    safe_intr = jnp.maximum(intrinsic_id, jnp.int32(0))
    cur_intrinsics = doffed_state.status.intrinsics
    new_intr_val = jnp.where(has_intrinsic, jnp.bool_(False), cur_intrinsics[safe_intr])
    new_intrinsics = cur_intrinsics.at[safe_intr].set(new_intr_val)
    doffed_state = doffed_state.replace(
        status=doffed_state.status.replace(intrinsics=new_intrinsics)
    )

    from Nethax.nethax.subsystems.armor_effects import recalc_worn_props
    doffed_state = recalc_worn_props(doffed_state)

    return _tree_where(is_active, doffed_state, state)


# ---------------------------------------------------------------------------
# handle_put_on_brax / handle_remove_brax — flat cascade over candidate states.
# ---------------------------------------------------------------------------

# Object category codes (obj.h OBJCLASS numbering).
_RING_CLASS   = 3
_AMULET_CLASS = 4


def handle_put_on_brax(state, rng: jax.Array):
    """Find the first ring or amulet in inventory and wear it.

    Flattens (vs canonical): the Python 3-way ``if cat == RING / AMULET / else``
    cascade plus the nested ``if worn[0] < 0 / if worn[1] < 0`` branches are
    replaced by computing all four candidate states unconditionally and
    selecting the appropriate one via a ``tree_where`` cascade.
    """
    inv = state.inventory
    cat = jnp.asarray(inv.items.category, dtype=jnp.int32)
    is_ring   = cat == jnp.int32(_RING_CLASS)
    is_amulet = cat == jnp.int32(_AMULET_CLASS)

    worn = inv.worn_rings
    left_empty  = worn[0].astype(jnp.int32) < jnp.int32(0)
    right_empty = worn[1].astype(jnp.int32) < jnp.int32(0)
    amulet_empty = inv.worn_amulet.astype(jnp.int32) < jnp.int32(0)

    # Compute all four candidate post-states.  Each callee receives the same
    # rng to preserve byte-parity with whichever branch the original picked.
    state_put_left  = put_on_ring_brax(state, rng, jnp.int32(0), jnp.int32(0))
    state_put_right = put_on_ring_brax(state, rng, jnp.int32(0), jnp.int32(1))
    state_wear_amul = wear_amulet_brax(state, rng, jnp.int32(0))

    # Cascade (matches canonical preference order):
    #   ring + left empty   → put_left
    #   ring + right empty  → put_right
    #   ring + both full    → state (no-op)
    #   amulet + slot empty → wear_amul
    #   amulet + slot full  → state (no-op)
    #   anything else       → state (no-op)
    do_put_left  = is_ring & left_empty
    do_put_right = is_ring & (~left_empty) & right_empty
    do_wear      = is_amulet & amulet_empty

    out = state
    out = _tree_where(do_wear,      state_wear_amul, out)
    out = _tree_where(do_put_right, state_put_right, out)
    out = _tree_where(do_put_left,  state_put_left,  out)
    return out


def handle_remove_brax(state, rng: jax.Array):
    """Remove the first worn ring or amulet found (left → right → amulet).

    Flattens (vs canonical): 3-way Python ``if`` cascade → ``tree_where``
    cascade over three candidate doff states.
    """
    del rng  # rng is unused in the canonical handle_remove path; kept for sig parity.
    inv = state.inventory
    worn = inv.worn_rings
    left_worn  = worn[0].astype(jnp.int32) >= jnp.int32(0)
    right_worn = worn[1].astype(jnp.int32) >= jnp.int32(0)
    amul_worn  = inv.worn_amulet.astype(jnp.int32) >= jnp.int32(0)

    state_doff_left  = take_off_ring_brax(state, jnp.int32(0))
    state_doff_right = take_off_ring_brax(state, jnp.int32(1))
    state_doff_amul  = take_off_amulet_brax(state)

    do_left  = left_worn
    do_right = (~left_worn) & right_worn
    do_amul  = (~left_worn) & (~right_worn) & amul_worn

    out = state
    out = _tree_where(do_amul,  state_doff_amul,  out)
    out = _tree_where(do_right, state_doff_right, out)
    out = _tree_where(do_left,  state_doff_left,  out)
    return out


# ---------------------------------------------------------------------------
# check_life_saving_brax — flatten the outer lax.cond.
# ---------------------------------------------------------------------------

def check_life_saving_brax(state):
    """If player would die and LIFESAVED is set, save them.

    Flattens (vs canonical): the single ``jax.lax.cond(should_save, _save, id)``
    is replaced by computing the save state unconditionally and selecting via
    ``tree_map + jnp.where``.

    Returns ``(new_state, saved)`` identical to the canonical contract.
    """
    intrinsic_idx = int(Intrinsic.LIFESAVED)
    has_lifesaving = state.status.intrinsics[intrinsic_idx]
    should_save = state.done & has_lifesaving

    # Unconditional save-branch computation (was ``_save`` inside lax.cond).
    amulet_slot = state.inventory.worn_amulet.astype(jnp.int32)
    qty = state.inventory.items.quantity
    if qty.ndim > 0:
        slot_mask = jnp.arange(qty.shape[0], dtype=jnp.int32) == amulet_slot
        new_quantity = jnp.where(slot_mask, jnp.int16(0), qty)
    else:
        new_quantity = jnp.int16(0)
    saved_items = state.inventory.items.replace(quantity=new_quantity)
    saved_inv = state.inventory.replace(
        items=saved_items,
        worn_amulet=jnp.int8(-1),
        worn_amulet_welded=jnp.bool_(False),
    )
    saved_intrinsics = state.status.intrinsics.at[intrinsic_idx].set(False)
    saved_status = state.status.replace(intrinsics=saved_intrinsics)
    saved_state = state.replace(
        done=jnp.bool_(False),
        player_hp=state.player_hp_max,
        status=saved_status,
        inventory=saved_inv,
    )

    new_state = _tree_where(should_save, saved_state, state)
    return new_state, should_save


# ===========================================================================
# items_corpses — Brax variants.
# ===========================================================================

# Re-export already-Brax-shaped helpers under _brax suffix for import-surface
# stability (they contain zero ``lax.cond`` / ``lax.switch`` / ``lax.scan``
# constructs; the originals route every branch through ``jnp.where``).
compute_rotted_brax           = _orig_compute_rotted
apply_old_corpse_effects_brax = _orig_apply_old_corpse_effects
apply_cannibalism_penalty_brax = _orig_apply_cannibalism_penalty
apply_tin_open_start_brax     = _orig_apply_tin_open_start
tick_tin_opening_brax         = _orig_tick_tin_opening


def apply_corpse_postfx_brax(
    state,
    rng: jax.Array,
    monster_entry_idx: jnp.ndarray,
):
    """Brax-style ``apply_corpse_postfx`` — flat HLO via ``jnp.where``.

    Flattens (vs canonical):
      * The wraith ``lax.cond(is_wraith, lambda s: pluslvl(s, rng_wraith,
        incr=False), lambda s: s, state)`` is replaced by calling
        ``experience.pluslvl`` unconditionally on a copy of the state and
        selecting the resulting pytree leaves via ``tree_map + jnp.where``.

    Preserved (per Craftax pattern):
      * The intrinsic-reservoir ``lax.scan`` over ``N_INTRINSICS`` slots
        stays as a ``scan`` — Brax/Craftax treat scan as the canonical flat
        form for sequential reductions (``scan = 1× HLO`` vs Python loop
        = N× HLO).

    Byte-parity:
      * RNG draw order is unchanged.  Every ``jax.random.split`` call
        happens in exactly the same order as the canonical
        ``apply_corpse_postfx``; the wraith pluslvl receives the same
        sub-key (``rng_wraith``) as before.
      * All mutations route through ``jnp.where`` masking.
    """
    is_corpse = monster_entry_idx >= jnp.int32(0)
    safe_idx  = jnp.clip(monster_entry_idx, 0, NUMMONS - 1)

    # ---- 1. Acidic side-effects -----------------------------------------
    is_acidic    = _MONSTER_IS_ACIDIC[safe_idx]
    has_acid_res = state.status.intrinsics[int(Intrinsic.RESIST_ACID)]
    rng, rng_a   = jax.random.split(rng)
    acid_dmg     = jax.random.randint(rng_a, (), 1, 16).astype(jnp.int32)
    do_acid_dmg  = is_corpse & is_acidic & ~has_acid_res
    new_hp_acid  = jnp.where(
        do_acid_dmg,
        jnp.maximum(state.player_hp - acid_dmg, jnp.int32(0)),
        state.player_hp,
    )
    state = state.replace(player_hp=new_hp_acid)

    # ---- 2. Poisonous side-effects --------------------------------------
    is_poisonous   = _MONSTER_IS_POISONOUS[safe_idx]
    has_poison_res = state.status.intrinsics[int(Intrinsic.RESIST_POISON)]
    rng, rng_p_gate, rng_p_hp, rng_p_str = jax.random.split(rng, 4)
    poison_gate    = jax.random.randint(rng_p_gate, (), 0, 5, dtype=jnp.int32) != jnp.int32(0)
    poison_dmg     = jax.random.randint(rng_p_hp, (), 1, 16).astype(jnp.int32)
    poison_str_loss = jax.random.randint(rng_p_str, (), 1, 5).astype(jnp.int16)
    do_poison_dmg = is_corpse & is_poisonous & poison_gate & ~has_poison_res & ~do_acid_dmg
    new_hp = jnp.where(
        do_poison_dmg,
        jnp.maximum(state.player_hp - poison_dmg, jnp.int32(0)),
        state.player_hp,
    )
    new_str_pois = jnp.where(
        do_poison_dmg,
        jnp.maximum(state.player_str - poison_str_loss, jnp.int16(3)),
        state.player_str,
    )
    state = state.replace(player_hp=new_hp, player_str=new_str_pois)

    # ---- 3. Intrinsic award (reservoir select via lax.scan) -------------
    row = _CORPSE_INTRINSIC_TABLE[safe_idx]

    def _reservoir_step(carry, i):
        sel_idx, count, rng_ = carry
        is_cand = row[i]
        count_ = count + jnp.where(is_cand, jnp.int32(1), jnp.int32(0))
        rng_, rng_roll = jax.random.split(rng_)
        safe_count = jnp.maximum(count_, jnp.int32(1))
        pick = jax.random.randint(rng_roll, (), 0, safe_count) == jnp.int32(0)
        sel_idx_ = jnp.where(is_cand & pick, jnp.int32(i), sel_idx)
        return (sel_idx_, count_, rng_), None

    (chosen_intr, n_candidates, rng), _ = jax.lax.scan(
        _reservoir_step,
        (jnp.int32(-1), jnp.int32(0), rng),
        jnp.arange(N_INTRINSICS, dtype=jnp.int32),
    )

    # Mind-flayer +INT pre-roll (drives both the +INT bump AND the
    # intrinsic-suppression mask).
    _mf_pre = is_corpse & (
        (safe_idx == jnp.int32(_MIND_FLAYER_IDX_NP))
        | (safe_idx == jnp.int32(_MASTER_MIND_FLAYER_IDX_NP))
    )
    rng, _rng_mf_pre = jax.random.split(rng)
    _mf_pre_roll = jax.random.randint(_rng_mf_pre, (), 0, 2, dtype=jnp.int32)
    _mf_int_lt_cap = state.player_int < jnp.int8(25)
    _mf_eats_brain = _mf_pre & _mf_int_lt_cap & (_mf_pre_roll == jnp.int32(0))

    # should_givit chance lookup (already brax-shape — nested jnp.where).
    safe_chosen = jnp.maximum(chosen_intr, jnp.int32(0))
    chance = jnp.where(
        safe_chosen == jnp.int32(int(Intrinsic.TELEPORT)), jnp.int32(10),
        jnp.where(
            safe_chosen == jnp.int32(int(Intrinsic.TELEPORT_CONTROL)), jnp.int32(12),
            jnp.where(
                safe_chosen == jnp.int32(int(Intrinsic.TELEPATHY)), jnp.int32(1),
                jnp.int32(15),
            ),
        ),
    )
    rng, rng_pres = jax.random.split(rng)
    is_bee_or_scorp = (safe_idx == jnp.int32(_KILLER_BEE_IDX_NP)) | (
        safe_idx == jnp.int32(_SCORPION_IDX_NP)
    )
    pres_fast = is_bee_or_scorp & (
        safe_chosen == jnp.int32(int(Intrinsic.RESIST_POISON))
    ) & (
        jax.random.randint(rng_pres, (), 0, 4, dtype=jnp.int32) == jnp.int32(0)
    )
    chance = jnp.where(pres_fast, jnp.int32(1), chance)

    rng, rng_giv = jax.random.split(rng)
    safe_chance = jnp.maximum(chance, jnp.int32(1))
    mlev = _MONSTER_MLEVEL[safe_idx]
    chance_roll = jax.random.randint(rng_giv, (), 0, safe_chance, dtype=jnp.int32)
    pass_gate = mlev > chance_roll

    has_intrinsic_to_grant = (
        is_corpse & (chosen_intr >= jnp.int32(0)) & pass_gate & ~_mf_eats_brain
    )
    old_intrinsics = state.status.intrinsics
    new_intrinsics = jnp.where(
        has_intrinsic_to_grant,
        old_intrinsics.at[chosen_intr].set(True),
        old_intrinsics,
    )
    from Nethax.nethax.subsystems.status_effects import FROMOUTSIDE as _FROMOUTSIDE
    cur_src = state.status.intrinsic_source
    new_src = jnp.where(
        has_intrinsic_to_grant,
        cur_src.at[chosen_intr].set(
            (cur_src[chosen_intr] | jnp.int8(_FROMOUTSIDE)).astype(jnp.int8)
        ),
        cur_src,
    )
    new_status = state.status.replace(intrinsics=new_intrinsics, intrinsic_source=new_src)
    state = state.replace(status=new_status)

    # temp_givit (STONE_RES / ACID_RES) — already brax-shape.
    rng, rng_t_stone, rng_t_acid, rng_d_stone, rng_d_acid = jax.random.split(rng, 5)
    chose_stone = chosen_intr == jnp.int32(int(Intrinsic.RESIST_STONE))
    chose_acid  = chosen_intr == jnp.int32(int(Intrinsic.RESIST_ACID))
    stone_roll = jax.random.randint(rng_t_stone, (), 0, 6, dtype=jnp.int32)
    do_stone_temp = is_corpse & chose_stone & (mlev > stone_roll) & ~_mf_eats_brain
    _stone_keys = jax.random.split(rng_d_stone, 3)
    stone_d36 = jnp.sum(jnp.stack([
        jax.random.randint(k, (), 1, 7, dtype=jnp.int32) for k in _stone_keys
    ])).astype(jnp.int32)
    cur_t_stone = state.status.timed_intrinsics[int(Intrinsic.RESIST_STONE)]
    new_t_stone = jnp.where(do_stone_temp, cur_t_stone + stone_d36, cur_t_stone)

    acid_roll = jax.random.randint(rng_t_acid, (), 0, 3, dtype=jnp.int32)
    do_acid_temp = is_corpse & chose_acid & (mlev > acid_roll) & ~_mf_eats_brain
    _acid_keys = jax.random.split(rng_d_acid, 3)
    acid_d36 = jnp.sum(jnp.stack([
        jax.random.randint(k, (), 1, 7, dtype=jnp.int32) for k in _acid_keys
    ])).astype(jnp.int32)
    cur_t_acid = state.status.timed_intrinsics[int(Intrinsic.RESIST_ACID)]
    new_t_acid = jnp.where(do_acid_temp, cur_t_acid + acid_d36, cur_t_acid)

    new_timed = state.status.timed_intrinsics.at[int(Intrinsic.RESIST_STONE)].set(new_t_stone)
    new_timed = new_timed.at[int(Intrinsic.RESIST_ACID)].set(new_t_acid)
    state = state.replace(status=state.status.replace(timed_intrinsics=new_timed))

    # ---- 4. Special one-off effects -------------------------------------

    # Wraith: +1 XL via pluslvl(incr=False).  THIS is the lax.cond we flatten:
    # compute pluslvl(state) unconditionally and tree_where on is_wraith.
    is_wraith = is_corpse & (safe_idx == jnp.int32(_WRAITH_IDX_NP))
    from Nethax.nethax.subsystems.experience import pluslvl as _xp_pluslvl
    rng, rng_wraith = jax.random.split(rng)
    state_after_pluslvl = _xp_pluslvl(state, rng_wraith, incr=False)
    state = _tree_where(is_wraith, state_after_pluslvl, state)

    # Newt: small chance of +1 pw_max.
    is_newt = is_corpse & (safe_idx == jnp.int32(_NEWT_IDX_NP))
    rng, rng_newt = jax.random.split(rng)
    newt_roll = jax.random.randint(rng_newt, (), 0, 3)
    do_pw_bump = is_newt & (newt_roll == jnp.int32(0))
    new_pw_max = jnp.where(do_pw_bump, state.player_pw_max + jnp.int32(1), state.player_pw_max)
    state = state.replace(player_pw_max=new_pw_max)

    # Nurse: restore HP to max + cure blindness.
    is_nurse = is_corpse & (safe_idx == jnp.int32(_NURSE_IDX_NP))
    new_hp3 = jnp.where(is_nurse, state.player_hp_max, state.player_hp)
    state = state.replace(player_hp=new_hp3)
    cur_blind = state.status.timed_statuses[int(TimedStatus.BLIND)]
    new_blind = jnp.where(is_nurse, jnp.int32(0), cur_blind)
    new_ts_nurse = state.status.timed_statuses.at[int(TimedStatus.BLIND)].set(new_blind)
    state = state.replace(status=state.status.replace(timed_statuses=new_ts_nurse))

    # Quantum mechanic: toggle FAST intrinsic.
    is_qm = is_corpse & (safe_idx == jnp.int32(_QUANTUM_MECHANIC_IDX_NP))
    cur_fast = state.status.intrinsics[int(Intrinsic.FAST)]
    new_fast = jnp.where(is_qm, ~cur_fast, cur_fast)
    qm_intrinsics = state.status.intrinsics.at[int(Intrinsic.FAST)].set(new_fast)
    new_status2 = state.status.replace(intrinsics=jnp.where(
        is_qm,
        qm_intrinsics,
        state.status.intrinsics,
    ))
    state = state.replace(status=new_status2)

    # Giant: +1 STR with vendor's 50% gate when STR is the only candidate.
    is_giant_corp = is_corpse & _MONSTER_IS_GIANT[safe_idx]
    rng, rng_str_gate = jax.random.split(rng)
    str_50_roll = jax.random.randint(rng_str_gate, (), 0, 2, dtype=jnp.int32)
    str_only = n_candidates == jnp.int32(0)
    apply_str = is_giant_corp & ((~str_only) | (str_50_roll == jnp.int32(0)))
    new_str = jnp.where(
        apply_str,
        jnp.minimum(state.player_str + jnp.int16(1), jnp.int16(125)),
        state.player_str,
    )
    state = state.replace(player_str=new_str)

    # Mind flayer +INT bump (already brax-shape).
    do_int_bump = _mf_eats_brain
    new_int = jnp.where(
        do_int_bump,
        jnp.minimum(state.player_int + jnp.int8(1), jnp.int8(25)),
        state.player_int,
    )
    state = state.replace(player_int=new_int)

    # Stalker.
    is_stalker = is_corpse & (safe_idx == jnp.int32(_STALKER_IDX_NP))
    rng, rng_st = jax.random.split(rng)
    invis_turns = jnp.int32(50) + jax.random.randint(rng_st, (), 0, 100, dtype=jnp.int32)
    cur_invis = state.status.timed_intrinsics[int(Intrinsic.INVIS)]
    new_invis_t = jnp.where(is_stalker, jnp.maximum(cur_invis, invis_turns), cur_invis)
    new_t_intr = state.status.timed_intrinsics.at[int(Intrinsic.INVIS)].set(new_invis_t)
    new_intrinsics2 = jnp.where(
        is_stalker,
        state.status.intrinsics.at[int(Intrinsic.SEE_INVIS)].set(True),
        state.status.intrinsics,
    )
    cur_stun = state.status.timed_statuses[int(TimedStatus.STUNNED)]
    new_stun = jnp.where(is_stalker, cur_stun + jnp.int32(30), cur_stun)
    new_ts_stalk = state.status.timed_statuses.at[int(TimedStatus.STUNNED)].set(new_stun)
    new_status3 = state.status.replace(
        timed_intrinsics=new_t_intr,
        intrinsics=new_intrinsics2,
        timed_statuses=new_ts_stalk,
    )
    state = state.replace(status=new_status3)

    # Displacer beast.
    is_displacer = is_corpse & (safe_idx == jnp.int32(_DISPLACER_BEAST_IDX_NP))
    rng, rng_dp = jax.random.split(rng)
    _dp_keys = jax.random.split(rng_dp, 6)
    disp_turns = jnp.sum(jnp.stack([
        jax.random.randint(k, (), 1, 7, dtype=jnp.int32) for k in _dp_keys
    ])).astype(jnp.int32)
    cur_disp = state.status.timed_intrinsics[int(Intrinsic.DISPLACED)]
    new_disp = jnp.where(is_displacer, cur_disp + disp_turns, cur_disp)
    new_t_intr2 = state.status.timed_intrinsics.at[int(Intrinsic.DISPLACED)].set(new_disp)
    state = state.replace(status=state.status.replace(timed_intrinsics=new_t_intr2))

    return state


def apply_eattin_brax(state, rng: jax.Array, item):
    """Brax variant of ``apply_eattin``.

    The original is already Brax-shape; this variant routes
    ``apply_corpse_postfx`` through ``apply_corpse_postfx_brax`` so the
    flattened wraith ``lax.cond`` propagates to the tin-eating path too.
    """
    is_spinach = item.enchantment == jnp.int8(1)
    corpse_idx = item.corpse_entry_idx.astype(jnp.int32)
    is_monster_tin = (~is_spinach) & (corpse_idx >= jnp.int32(0))
    is_poisoned_tin = item.tin_poisoned

    new_str = jnp.where(
        is_spinach,
        jnp.minimum(state.player_str + jnp.int16(1), jnp.int16(18)),
        state.player_str,
    )
    state = state.replace(player_str=new_str)

    effective_idx = jnp.where(is_monster_tin, corpse_idx, jnp.int32(-1))
    state = apply_corpse_postfx_brax(state, rng, effective_idx)

    rng, rng_dmg = jax.random.split(rng)
    poison_dmg = jax.random.randint(rng_dmg, (), 1, 16).astype(jnp.int32)
    new_hp = jnp.where(
        is_poisoned_tin,
        jnp.maximum(state.player_hp - poison_dmg, jnp.int32(0)),
        state.player_hp,
    )
    state = state.replace(player_hp=new_hp)

    cur_sick = state.status.timed_statuses[int(TimedStatus.SICK)]
    new_sick = jnp.where(
        is_poisoned_tin, jnp.maximum(cur_sick, jnp.int32(10)), cur_sick
    )
    new_ts = state.status.timed_statuses.at[int(TimedStatus.SICK)].set(new_sick)
    new_sick_kind = jnp.where(is_poisoned_tin, jnp.int8(1), state.status.sick_kind)
    new_status = state.status.replace(timed_statuses=new_ts, sick_kind=new_sick_kind)
    return state.replace(status=new_status)


# ===========================================================================
# items — Brax variants (already Brax-shape; aliases for surface stability).
# ===========================================================================

erode_obj_slot_brax = _orig_erode_obj_slot
erode_obj_brax      = _orig_erode_obj


__all__ = [
    # Internal helpers exposed for testing.
    "_ring_apply_stat_brax",
    "_ring_revoke_stat_brax",
    # Jewelry public entry points.
    "put_on_ring_brax",
    "take_off_ring_brax",
    "ring_tick_brax",
    "wear_amulet_brax",
    "take_off_amulet_brax",
    "handle_put_on_brax",
    "handle_remove_brax",
    "check_life_saving_brax",
    # Corpse public entry points.
    "apply_corpse_postfx_brax",
    "compute_rotted_brax",
    "apply_old_corpse_effects_brax",
    "apply_cannibalism_penalty_brax",
    "apply_tin_open_start_brax",
    "tick_tin_opening_brax",
    "apply_eattin_brax",
    # Items public entry points.
    "erode_obj_slot_brax",
    "erode_obj_brax",
]
