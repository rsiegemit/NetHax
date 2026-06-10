"""Brax-style flattened rewrites of inventory + swallow public entry points.

Precedent: Brax, Craftax, and ``combat_helpers_brax.py``.  Every
``jax.lax.cond`` and ``jax.lax.switch`` in the original public entry
points is replaced with always-compute + ``jnp.where`` (or ``jax.tree.map
+ jnp.where`` for pytree-typed branches).  ``lax.scan`` over fixed-count
loops (inventory slots, ground stacks, armor slots) is preserved
verbatim — these are static-bound and compile to a single tight loop in
HLO, which is the desired Brax shape.

Byte-parity constraints
-----------------------
1. RNG draw order preserved exactly — every ``rn2`` / ``rnd`` /
   ``jax.random.*`` call fires in the same order on the same key as the
   original, regardless of which branch is selected.
2. Mutations byte-identical via ``jnp.where`` over pytrees with the same
   scalar mask as the original ``lax.cond`` predicate.
3. State pytree shape preserved — the leaves selected by ``jnp.where``
   share dtype with the originals; ``_select_tree`` walks the Flax
   dataclass via ``jax.tree.map``.

Flattening summary (conds / switches collapsed → ``jnp.where``)
---------------------------------------------------------------
Inventory:
  pickup_brax           : 1 lax.cond  → 1 pytree-where (vault witness).
  drop_brax             : 1 lax.cond  → 1 pytree-where (altar BUC).
  wield_brax            : 0 conds     → unchanged (already where-based).
  unwield_brax          : 0 conds     → unchanged.
  wear_armor_brax       : 1 lax.cond  → 1 scalar where (WEAR_ROBE msg).
  take_off_armor_brax   : 0 conds     → unchanged.
  handle_pickup_brax    : 1 lax.cond  → 1 pytree-where (quest artifact).
  handle_drop_brax      : 0 conds     → unchanged (scan-only).
  handle_wield_brax     : 1 lax.cond  → 1 pytree-where (wield msg).
  handle_unwield_brax   : 0 conds     → unchanged.
  handle_wear_brax      : 0 conds     → unchanged.
  step_brax             : 0 conds     → unchanged.
  handle_name_brax      : 0 conds     → unchanged.
  Total: 4 lax.cond  →  4 jnp.where flattenings.

Swallow (excluding ``try_engulf``, which lives in combat_helpers_brax):
  release_from_engulf_brax : 0 conds → unchanged (pure data update).
  digest_tick_brax         : 2 lax.cond → 2 pytree-where (release + outer
                             swallowed gate).
  Total: 2 lax.cond  →  2 jnp.where flattenings.

Grand total: 6 lax.cond sites flattened.  Zero ``lax.switch`` sites in
either source module.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax

from Nethax.nethax.rng import rnd
from Nethax.nethax.subsystems.inventory import (
    ArmorSlot,
    Item,
    ItemCategory,
    MAX_GROUND_STACK,
    MAX_INVENTORY_SLOTS,
    _LOADSTONE_TYPE_ID,
    _ARTI_SUNSWORD,
    _CORPSE_TYPE_ID,
    _PM_CHICKATRICE,
    _PM_COCKATRICE,
    _find_merge_slot,
    compute_ac,
    total_weight,
    weight_cap,
)
from Nethax.nethax.subsystems.swallow import (
    _SWALLOW_VARIANT,
    release_from_engulf,
)


# ---------------------------------------------------------------------------
# Pytree-where helper (same shape as combat_helpers_brax._select_tree)
# ---------------------------------------------------------------------------
def _select_tree(cond, on_true, on_false):
    """Pytree analogue of ``jnp.where(cond, on_true, on_false)``.

    Both branches must be pytrees with identical structure and dtypes.
    """
    return jax.tree.map(lambda a, b: jnp.where(cond, a, b), on_true, on_false)


# ===========================================================================
# INVENTORY
# ===========================================================================

# ---------------------------------------------------------------------------
# pickup_brax
# ---------------------------------------------------------------------------
def pickup_brax(state, rng, ground_items: Item, branch: int, level: int) -> tuple:
    """Brax-style flattened rewrite of ``inventory.pickup``.

    Conds flattened: 1
      - vault witness ``lax.cond(witnessed, vault_gd_watching, identity, …)``
        → ``_select_tree(witnessed, watched_state, new_state)``.

    The two ``lax.scan`` loops (``_find_merge_slot`` over inventory and
    the first-empty-slot / first-free-letter scans) are preserved — they
    iterate over a Python-static count (``MAX_INVENTORY_SLOTS`` or 26)
    and compile to a single tight loop.

    RNG draw order: identical to the original (``rng`` is currently
    unused in ``pickup``; this Brax rewrite reserves it for future weight
    checks the same way).
    """
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)

    ground_cat  = ground_items.category[branch, level, row, col, 0]
    ground_tid  = ground_items.type_id[branch, level, row, col, 0]
    ground_buc  = ground_items.buc_status[branch, level, row, col, 0]
    ground_ench = ground_items.enchantment[branch, level, row, col, 0]
    ground_eprf = ground_items.oerodeproof[branch, level, row, col, 0]
    ground_wt   = ground_items.weight[branch, level, row, col, 0].astype(jnp.int32)
    ground_qty  = ground_items.quantity[branch, level, row, col, 0].astype(jnp.int32)

    has_item = ground_cat != 0
    is_gold = has_item & (ground_cat == jnp.int8(ItemCategory.COIN))
    gold_qty = jnp.where(is_gold, ground_qty, jnp.int32(0))
    is_loadstone = has_item & (ground_tid == jnp.int16(_LOADSTONE_TYPE_ID))

    merge_found, merge_slot = _find_merge_slot(
        state.inventory.items,
        ground_cat, ground_tid, ground_buc, ground_ench, ground_eprf,
    )

    # First-empty-slot scan (kept — scan over MAX_INVENTORY_SLOTS).
    def _find_slot(carry, idx):
        found, slot = carry
        is_empty = state.inventory.items.category[idx] == 0
        slot  = jnp.where(~found & is_empty, idx, slot)
        found = found | is_empty
        return (found, slot), None

    (empty_found, empty_slot), _ = lax.scan(
        _find_slot,
        (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32),
    )

    target_slot = jnp.where(merge_found, merge_slot, empty_slot)
    cap = weight_cap(state)
    cur_wt = state.inventory.total_weight.astype(jnp.int32)
    new_total_wt_if_lifted = cur_wt + jnp.where(is_gold, jnp.int32(0), ground_wt)
    over_cap = new_total_wt_if_lifted > cap
    weight_ok = (~over_cap) | is_loadstone | is_gold

    slot_ok = merge_found | empty_found | is_gold | is_loadstone
    can_pickup = has_item & slot_ok & weight_ok

    safe_slot = jnp.clip(target_slot, 0, MAX_INVENTORY_SLOTS - 1)
    new_items = state.inventory.items
    write_slot = can_pickup & ~is_gold
    merge_write = write_slot & merge_found
    fresh_write = write_slot & ~merge_found

    existing_qty = new_items.quantity[safe_slot].astype(jnp.int32)
    merged_qty   = existing_qty + ground_qty
    new_qty_val  = jnp.where(
        merge_write, merged_qty.astype(jnp.int16),
        jnp.where(fresh_write, ground_qty.astype(jnp.int16),
                  new_items.quantity[safe_slot]),
    )
    existing_wt  = new_items.weight[safe_slot].astype(jnp.int32)
    merged_wt    = existing_wt + ground_wt
    new_wt_val   = jnp.where(
        merge_write, merged_wt,
        jnp.where(fresh_write, ground_wt, new_items.weight[safe_slot]),
    )

    new_items = new_items.replace(
        category   = new_items.category.at[safe_slot].set(
            jnp.where(fresh_write, ground_cat, new_items.category[safe_slot])
        ),
        type_id    = new_items.type_id.at[safe_slot].set(
            jnp.where(fresh_write, ground_tid, new_items.type_id[safe_slot])
        ),
        buc_status = new_items.buc_status.at[safe_slot].set(
            jnp.where(fresh_write, ground_buc, new_items.buc_status[safe_slot])
        ),
        enchantment = new_items.enchantment.at[safe_slot].set(
            jnp.where(fresh_write, ground_ench, new_items.enchantment[safe_slot])
        ),
        charges    = new_items.charges.at[safe_slot].set(
            jnp.where(fresh_write,
                      ground_items.charges[branch, level, row, col, 0],
                      new_items.charges[safe_slot])
        ),
        identified = new_items.identified.at[safe_slot].set(
            jnp.where(fresh_write,
                      ground_items.identified[branch, level, row, col, 0],
                      new_items.identified[safe_slot])
        ),
        quantity   = new_items.quantity.at[safe_slot].set(new_qty_val),
        weight     = new_items.weight.at[safe_slot].set(new_wt_val),
        ac_bonus   = new_items.ac_bonus.at[safe_slot].set(
            jnp.where(fresh_write,
                      ground_items.ac_bonus[branch, level, row, col, 0],
                      new_items.ac_bonus[safe_slot])
        ),
        is_two_handed = new_items.is_two_handed.at[safe_slot].set(
            jnp.where(fresh_write,
                      ground_items.is_two_handed[branch, level, row, col, 0],
                      new_items.is_two_handed[safe_slot])
        ),
        dknown = new_items.dknown.at[safe_slot].set(
            jnp.where(write_slot, jnp.bool_(True), new_items.dknown[safe_slot])
        ),
        artifact_idx = new_items.artifact_idx.at[safe_slot].set(
            jnp.where(
                fresh_write,
                ground_items.artifact_idx[branch, level, row, col, 0],
                new_items.artifact_idx[safe_slot],
            )
        ),
    )

    # Letter-assignment scans (kept — 26-element scans).
    inuse_lower = jnp.zeros((26,), dtype=jnp.bool_)
    inuse_upper = jnp.zeros((26,), dtype=jnp.bool_)
    cur_letters = state.inventory.letters.astype(jnp.int32)
    def _mark_letters(carry, idx):
        il, iu = carry
        ch = cur_letters[idx]
        is_lower = (ch >= jnp.int32(ord('a'))) & (ch <= jnp.int32(ord('z')))
        is_upper = (ch >= jnp.int32(ord('A'))) & (ch <= jnp.int32(ord('Z')))
        l_idx = jnp.clip(ch - jnp.int32(ord('a')), 0, 25)
        u_idx = jnp.clip(ch - jnp.int32(ord('A')), 0, 25)
        il = jnp.where(is_lower, il.at[l_idx].set(True), il)
        iu = jnp.where(is_upper, iu.at[u_idx].set(True), iu)
        return (il, iu), None
    (inuse_lower, inuse_upper), _ = lax.scan(
        _mark_letters,
        (inuse_lower, inuse_upper),
        jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32),
    )
    def _first_free(carry, idx):
        found, slot = carry
        free_l = ~inuse_lower[idx]
        slot = jnp.where(~found & free_l, jnp.int32(ord('a')) + idx, slot)
        found = found | free_l
        return (found, slot), None
    (low_found, low_letter), _ = lax.scan(
        _first_free, (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(26, dtype=jnp.int32),
    )
    def _first_free_upper(carry, idx):
        found, slot = carry
        free_u = ~inuse_upper[idx]
        slot = jnp.where(~found & free_u, jnp.int32(ord('A')) + idx, slot)
        found = found | free_u
        return (found, slot), None
    (up_found, up_letter), _ = lax.scan(
        _first_free_upper, (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(26, dtype=jnp.int32),
    )
    chosen_letter = jnp.where(low_found, low_letter, up_letter).astype(jnp.int8)
    new_letters = state.inventory.letters.at[safe_slot].set(
        jnp.where(fresh_write, chosen_letter, state.inventory.letters[safe_slot])
    )

    new_ground_items = ground_items.replace(
        category=ground_items.category.at[branch, level, row, col, 0].set(
            jnp.where(can_pickup, jnp.int8(0),
                      ground_items.category[branch, level, row, col, 0])
        )
    )

    new_inv = state.inventory.replace(
        items=new_items,
        total_weight=total_weight(new_items),
        letters=new_letters,
    )
    new_gold = state.player_gold + gold_qty
    new_state = state.replace(inventory=new_inv, player_gold=new_gold)

    # Vault-guard witness — flatten ``lax.cond(witnessed, vault_gd_watching, id)``.
    from Nethax.nethax.subsystems.vault import (
        vault_gd_watching as _vault_witness,
        GD_EATGOLD as _GD_EATGOLD,
    )
    from Nethax.nethax.dungeon.branches import (
        MAX_LEVELS_PER_BRANCH as _MAX_LV,
    )
    flat_lv = (
        new_state.dungeon.current_branch.astype(jnp.int32) * jnp.int32(_MAX_LV)
        + (new_state.dungeon.current_level - jnp.int8(1)).astype(jnp.int32)
    )
    vp = new_state.features.vault_pos[flat_lv]
    vr, vc = vp[0].astype(jnp.int32), vp[1].astype(jnp.int32)
    pr_i = new_state.player_pos[0].astype(jnp.int32)
    pc_i = new_state.player_pos[1].astype(jnp.int32)
    in_vault = (vr >= jnp.int32(0)) & (
        (jnp.abs(pr_i - vr) <= jnp.int32(1))
        & (jnp.abs(pc_i - vc) <= jnp.int32(1))
    )
    witnessed = in_vault & (gold_qty > jnp.int32(0))
    # Always-compute the watched branch; pytree-select on ``witnessed``.
    watched_state = _vault_witness(new_state, _GD_EATGOLD)
    new_state = _select_tree(witnessed, watched_state, new_state)

    return new_state, new_ground_items


# ---------------------------------------------------------------------------
# drop_brax
# ---------------------------------------------------------------------------
def drop_brax(state, rng, ground_items: Item, branch: int, level: int,
              slot_idx: int) -> tuple:
    """Brax-style flattened rewrite of ``inventory.drop``.

    Conds flattened: 1
      - altar BUC mutation ``lax.cond(on_altar, drop_at_altar, identity, state)``
        → ``_select_tree(on_altar, altared, state)``.

    The single ``lax.scan`` over the ground stack (``MAX_GROUND_STACK``
    elements) is preserved.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intrinsic
    from Nethax.nethax.constants.tiles import TileType as _TileType
    from Nethax.nethax.subsystems.features import drop_at_altar as _drop_at_altar

    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)
    slot_idx = jnp.int32(slot_idx)

    has_item = state.inventory.items.category[slot_idx] != 0

    LOADSTONE_TYPE_ID = jnp.int16(_LOADSTONE_TYPE_ID)
    CURSED = jnp.int8(1)
    is_cursed_loadstone = (
        (state.inventory.items.type_id[slot_idx] == LOADSTONE_TYPE_ID)
        & (state.inventory.items.buc_status[slot_idx] == CURSED)
    )

    is_wielded_slot = (
        slot_idx == state.inventory.wielded.astype(jnp.int32)
    ) & (state.inventory.wielded.astype(jnp.int32) >= jnp.int32(0))
    welded_block = is_wielded_slot & state.inventory.welded

    levitating = state.status.intrinsics[int(_Intrinsic.LEVITATION)]

    has_item = has_item & ~is_cursed_loadstone & ~welded_block & ~levitating

    in_cat  = state.inventory.items.category[slot_idx]
    in_tid  = state.inventory.items.type_id[slot_idx]
    in_buc  = state.inventory.items.buc_status[slot_idx]
    in_ench = state.inventory.items.enchantment[slot_idx]
    in_eprf = state.inventory.items.oerodeproof[slot_idx]
    in_qty  = state.inventory.items.quantity[slot_idx].astype(jnp.int32)
    in_wt   = state.inventory.items.weight[slot_idx].astype(jnp.int32)

    def _scan(carry, stack_idx):
        empty_found, empty_pos, merge_found, merge_pos = carry
        cat_here = ground_items.category[branch, level, row, col, stack_idx]
        is_empty = cat_here == jnp.int8(0)
        is_match = (
            (~is_empty)
            & (cat_here == in_cat)
            & (ground_items.type_id[branch, level, row, col, stack_idx]    == in_tid)
            & (ground_items.buc_status[branch, level, row, col, stack_idx] == in_buc)
            & (ground_items.enchantment[branch, level, row, col, stack_idx] == in_ench)
            & (ground_items.oerodeproof[branch, level, row, col, stack_idx] == in_eprf)
        )
        empty_pos = jnp.where(~empty_found & is_empty, stack_idx, empty_pos)
        empty_found = empty_found | is_empty
        merge_pos = jnp.where(~merge_found & is_match, stack_idx, merge_pos)
        merge_found = merge_found | is_match
        return (empty_found, empty_pos, merge_found, merge_pos), None

    (g_empty_found, g_empty_pos, g_merge_found, g_merge_pos), _ = lax.scan(
        _scan,
        (jnp.bool_(False), jnp.int32(0), jnp.bool_(False), jnp.int32(0)),
        jnp.arange(MAX_GROUND_STACK, dtype=jnp.int32),
    )

    g_target = jnp.where(g_merge_found, g_merge_pos, g_empty_pos)
    g_slot_ok = g_merge_found | g_empty_found
    can_drop = has_item & g_slot_ok
    safe_gs  = jnp.clip(g_target, 0, MAX_GROUND_STACK - 1)

    # Altar BUC mutation — flatten ``lax.cond(on_altar, drop_at_altar, id)``.
    here_tile = state.terrain[branch, level, row, col].astype(jnp.int32)
    on_altar = (here_tile == jnp.int32(int(_TileType.ALTAR))) & can_drop
    altared_full = _drop_at_altar(state, slot_idx)
    state_altared = _select_tree(on_altar, altared_full, state)
    inv = state_altared.inventory.items

    merge_write = can_drop & g_merge_found
    fresh_write = can_drop & ~g_merge_found

    def _set_ground(field_ground, field_inv):
        return field_ground.at[branch, level, row, col, safe_gs].set(
            jnp.where(fresh_write, field_inv[slot_idx],
                      field_ground[branch, level, row, col, safe_gs])
        )

    g_existing_qty = ground_items.quantity[branch, level, row, col, safe_gs].astype(jnp.int32)
    g_existing_wt  = ground_items.weight[branch, level, row, col, safe_gs].astype(jnp.int32)
    merged_qty = (g_existing_qty + in_qty).astype(jnp.int16)
    merged_wt  = (g_existing_wt + in_wt).astype(jnp.int32)

    new_qty_at_pos = jnp.where(
        merge_write, merged_qty,
        jnp.where(fresh_write, inv.quantity[slot_idx],
                  ground_items.quantity[branch, level, row, col, safe_gs])
    )
    new_wt_at_pos = jnp.where(
        merge_write, merged_wt,
        jnp.where(fresh_write, inv.weight[slot_idx],
                  ground_items.weight[branch, level, row, col, safe_gs])
    )

    new_ground = ground_items.replace(
        category    = _set_ground(ground_items.category,    inv.category),
        type_id     = _set_ground(ground_items.type_id,     inv.type_id),
        buc_status  = _set_ground(ground_items.buc_status,  inv.buc_status),
        enchantment = _set_ground(ground_items.enchantment, inv.enchantment),
        charges     = _set_ground(ground_items.charges,     inv.charges),
        identified  = _set_ground(ground_items.identified,  inv.identified),
        quantity    = ground_items.quantity.at[branch, level, row, col, safe_gs].set(new_qty_at_pos),
        weight      = ground_items.weight.at[branch, level, row, col, safe_gs].set(new_wt_at_pos),
        ac_bonus    = _set_ground(ground_items.ac_bonus,    inv.ac_bonus),
        is_two_handed = _set_ground(ground_items.is_two_handed, inv.is_two_handed),
        artifact_idx = _set_ground(ground_items.artifact_idx, inv.artifact_idx),
    )

    new_items = inv.replace(
        category   = inv.category.at[slot_idx].set(
            jnp.where(can_drop, jnp.int8(0), inv.category[slot_idx])),
        type_id    = inv.type_id.at[slot_idx].set(
            jnp.where(can_drop, jnp.int16(0), inv.type_id[slot_idx])),
        buc_status = inv.buc_status.at[slot_idx].set(
            jnp.where(can_drop, jnp.int8(0), inv.buc_status[slot_idx])),
        enchantment= inv.enchantment.at[slot_idx].set(
            jnp.where(can_drop, jnp.int8(0), inv.enchantment[slot_idx])),
        charges    = inv.charges.at[slot_idx].set(
            jnp.where(can_drop, jnp.int8(0), inv.charges[slot_idx])),
        identified = inv.identified.at[slot_idx].set(
            jnp.where(can_drop, jnp.bool_(False), inv.identified[slot_idx])),
        quantity   = inv.quantity.at[slot_idx].set(
            jnp.where(can_drop, jnp.int16(0), inv.quantity[slot_idx])),
        weight     = inv.weight.at[slot_idx].set(
            jnp.where(can_drop, jnp.int32(0), inv.weight[slot_idx])),
        ac_bonus   = inv.ac_bonus.at[slot_idx].set(
            jnp.where(can_drop, jnp.int8(0), inv.ac_bonus[slot_idx])),
        is_two_handed = inv.is_two_handed.at[slot_idx].set(
            jnp.where(can_drop, jnp.bool_(False), inv.is_two_handed[slot_idx])),
        artifact_idx = inv.artifact_idx.at[slot_idx].set(
            jnp.where(can_drop, jnp.int8(-1), inv.artifact_idx[slot_idx])),
    )

    new_letters = state_altared.inventory.letters.at[slot_idx].set(
        jnp.where(can_drop, jnp.int8(0), state_altared.inventory.letters[slot_idx])
    )

    new_inv = state_altared.inventory.replace(
        items=new_items,
        total_weight=total_weight(new_items),
        letters=new_letters,
    )
    new_state = state_altared.replace(inventory=new_inv)
    return new_state, new_ground


# ---------------------------------------------------------------------------
# wield_brax / unwield_brax — already ``jnp.where``-based in the original,
# replicated here verbatim so the Brax surface is complete.
# ---------------------------------------------------------------------------
def wield_brax(state, slot_idx: int):
    """Brax-style rewrite of ``inventory.wield``.

    Conds flattened: 0 — the original is already structured around
    ``jnp.where`` and ``.at[...]`` scatter, with no ``lax.cond`` /
    ``lax.switch`` sites.  Reproduced verbatim for surface completeness.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic, TimedStatus
    from Nethax.nethax.subsystems.scoring import DeathCause

    slot_idx = jnp.int8(slot_idx)
    slot_i32 = slot_idx.astype(jnp.int32)
    has_item = state.inventory.items.category[slot_i32] != 0

    items = state.inventory.items
    is_corpse = has_item & (items.type_id[slot_i32].astype(jnp.int32)
                            == jnp.int32(_CORPSE_TYPE_ID))
    cnm = items.corpse_entry_idx[slot_i32].astype(jnp.int32)
    is_petrify_corpse = is_corpse & (
        (cnm == jnp.int32(_PM_COCKATRICE))
        | (cnm == jnp.int32(_PM_CHICKATRICE))
    )
    gloves_slot = state.inventory.worn_armor[int(ArmorSlot.GLOVES)].astype(jnp.int32)
    has_gloves = gloves_slot >= jnp.int32(0)
    stone_res = (
        state.status.intrinsics[int(Intrinsic.RESIST_STONE)]
        | (state.status.timed_intrinsics[int(Intrinsic.RESIST_STONE)] > 0)
    )
    petrify = is_petrify_corpse & (~has_gloves) & (~stone_res)

    new_hp = jnp.where(petrify, jnp.int32(0), state.player_hp)
    new_done = state.done | petrify
    new_cause = jnp.where(
        petrify,
        jnp.int8(int(DeathCause.STONING)),
        state.scoring.death_cause,
    )
    new_scoring = state.scoring.replace(death_cause=new_cause)
    cur_stoned = state.status.timed_statuses[int(TimedStatus.STONED)].astype(jnp.int32)
    new_stoned = jnp.where(petrify, jnp.int32(1), cur_stoned)

    can_wield = has_item & ~is_petrify_corpse

    new_wielded = jnp.where(can_wield, slot_idx, state.inventory.wielded)

    is_two_handed = state.inventory.items.is_two_handed[slot_i32]
    shield_slot   = jnp.int32(ArmorSlot.SHIELD)
    new_worn_armor = jnp.where(
        can_wield & is_two_handed,
        state.inventory.worn_armor.at[shield_slot].set(jnp.int8(-1)),
        state.inventory.worn_armor,
    )

    CURSED = jnp.int8(1)
    is_cursed = state.inventory.items.buc_status[slot_i32] == CURSED
    new_welded = jnp.where(can_wield & is_cursed,
                           jnp.bool_(True), state.inventory.welded)

    is_sunsword = (
        state.inventory.items.artifact_idx[slot_i32].astype(jnp.int32)
        == jnp.int32(_ARTI_SUNSWORD)
    )
    cur_lamplit = state.inventory.items.lamplit[slot_i32]
    new_lamplit = state.inventory.items.lamplit.at[slot_i32].set(
        jnp.where(can_wield & is_sunsword, jnp.bool_(True), cur_lamplit)
    )
    new_items = state.inventory.items.replace(lamplit=new_lamplit)

    new_statuses = state.status.timed_statuses.at[int(TimedStatus.STONED)].set(new_stoned)
    new_status = state.status.replace(timed_statuses=new_statuses)

    new_inv = state.inventory.replace(
        items=new_items,
        wielded=new_wielded,
        worn_armor=new_worn_armor,
        welded=new_welded,
    )
    return state.replace(
        inventory=new_inv,
        player_hp=new_hp,
        done=new_done,
        scoring=new_scoring,
        status=new_status,
    )


def unwield_brax(state):
    """Brax-style rewrite of ``inventory.unwield``.

    Conds flattened: 0 — original is already ``jnp.where``-based.
    """
    can_unwield = ~state.inventory.welded
    prev_slot = state.inventory.wielded.astype(jnp.int32)
    prev_safe = jnp.clip(prev_slot, 0, MAX_INVENTORY_SLOTS - 1)
    was_sunsword = (prev_slot >= jnp.int32(0)) & (
        state.inventory.items.artifact_idx[prev_safe].astype(jnp.int32)
        == jnp.int32(_ARTI_SUNSWORD)
    )
    clear_lamp = can_unwield & was_sunsword
    cur_lamp = state.inventory.items.lamplit[prev_safe]
    new_lamp_at_slot = jnp.where(clear_lamp, jnp.bool_(False), cur_lamp)
    new_lamplit = state.inventory.items.lamplit.at[prev_safe].set(new_lamp_at_slot)
    new_items = state.inventory.items.replace(lamplit=new_lamplit)

    new_wielded = jnp.where(can_unwield, jnp.int8(-1), state.inventory.wielded)
    new_inv = state.inventory.replace(items=new_items, wielded=new_wielded)
    return state.replace(inventory=new_inv)


# ---------------------------------------------------------------------------
# wear_armor_brax
# ---------------------------------------------------------------------------
def wear_armor_brax(state, slot_idx: int, armor_slot: ArmorSlot):
    """Brax-style flattened rewrite of ``inventory.wear_armor``.

    Conds flattened: 1
      - ``lax.cond(can_wear, emit_WEAR_ROBE, identity, messages)``
        → scalar ``jnp.where``-merge over the (already
        ``jnp.where``-shaped) messages pytree.

    ``apply_armor_effects`` is called unconditionally at tail just like
    the original — it is a pure recompute, not a conditional branch.
    """
    slot_idx   = jnp.int8(slot_idx)
    slot_i32   = slot_idx.astype(jnp.int32)
    armor_i32  = jnp.int32(int(armor_slot))

    has_item   = state.inventory.items.category[slot_i32] != 0
    is_armor   = state.inventory.items.category[slot_i32] == jnp.int8(ItemCategory.ARMOR)
    can_wear   = has_item & is_armor

    new_worn_armor = jnp.where(
        can_wear,
        state.inventory.worn_armor.at[armor_i32].set(slot_idx),
        state.inventory.worn_armor,
    )
    item_bonus = state.inventory.items.ac_bonus[slot_i32].astype(jnp.int8)
    new_worn_ac_bonus = jnp.where(
        can_wear,
        state.inventory.worn_armor_ac_bonus.at[armor_i32].set(item_bonus),
        state.inventory.worn_armor_ac_bonus,
    )
    CURSED = jnp.int8(1)
    is_cursed = state.inventory.items.buc_status[slot_i32] == CURSED
    new_worn_armor_welded = jnp.where(
        can_wear & is_cursed,
        state.inventory.worn_armor_welded.at[armor_i32].set(jnp.bool_(True)),
        state.inventory.worn_armor_welded,
    )
    new_ac = compute_ac(state.inventory.items, new_worn_armor)

    new_items_id = jnp.where(
        can_wear,
        state.inventory.items.identified.at[slot_i32].set(jnp.bool_(True)),
        state.inventory.items.identified,
    )
    new_items_rknown = jnp.where(
        can_wear,
        state.inventory.items.rknown.at[slot_i32].set(jnp.bool_(True)),
        state.inventory.items.rknown,
    )
    item_type_id = state.inventory.items.type_id[slot_i32].astype(jnp.int32)
    type_mask    = state.identification.identified
    t_clip       = jnp.clip(item_type_id, jnp.int32(0),
                            jnp.int32(type_mask.shape[0] - 1))
    new_type_mask = jnp.where(
        can_wear,
        type_mask.at[t_clip].set(jnp.bool_(True)),
        type_mask,
    )

    new_inv = state.inventory.replace(
        worn_armor=new_worn_armor,
        worn_armor_ac_bonus=new_worn_ac_bonus,
        worn_armor_welded=new_worn_armor_welded,
        items=state.inventory.items.replace(
            identified=new_items_id,
            rknown=new_items_rknown,
        ),
    )

    # Flatten ``lax.cond(can_wear, emit, identity, messages)`` →
    # always-emit then pytree-where on ``can_wear``.
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    emitted = _msg_emit(state.messages, int(_MsgId.WEAR_ROBE))
    new_messages = _select_tree(can_wear, emitted, state.messages)

    new_state = state.replace(
        inventory=new_inv,
        player_ac=new_ac,
        identification=state.identification.replace(identified=new_type_mask),
        messages=new_messages,
    )
    from Nethax.nethax.subsystems.armor_effects import apply_armor_effects
    return apply_armor_effects(new_state)


# ---------------------------------------------------------------------------
# take_off_armor_brax
# ---------------------------------------------------------------------------
def take_off_armor_brax(state, armor_slot: ArmorSlot):
    """Brax-style rewrite of ``inventory.take_off_armor``.

    Conds flattened: 0 — original is already ``jnp.where``-shaped.
    """
    armor_i32 = jnp.int32(int(armor_slot))
    is_welded = state.inventory.worn_armor_welded[armor_i32]
    can_remove = ~is_welded

    new_worn_armor = jnp.where(
        can_remove,
        state.inventory.worn_armor.at[armor_i32].set(jnp.int8(-1)),
        state.inventory.worn_armor,
    )
    new_worn_ac_bonus = jnp.where(
        can_remove,
        state.inventory.worn_armor_ac_bonus.at[armor_i32].set(jnp.int8(0)),
        state.inventory.worn_armor_ac_bonus,
    )
    new_ac = compute_ac(state.inventory.items, new_worn_armor)

    new_inv = state.inventory.replace(
        worn_armor=new_worn_armor,
        worn_armor_ac_bonus=new_worn_ac_bonus,
    )
    new_state = state.replace(inventory=new_inv, player_ac=new_ac)
    from Nethax.nethax.subsystems.armor_effects import apply_armor_effects
    return apply_armor_effects(new_state)


# ---------------------------------------------------------------------------
# Action handlers (top-level dispatch targets)
# ---------------------------------------------------------------------------
def handle_pickup_brax(state, rng, ground_items: Item, branch: int, level: int) -> tuple:
    """Brax-style flattened rewrite of ``inventory.handle_pickup``.

    Conds flattened: 1
      - quest-artifact gate ``lax.cond(is_quest_artifact,
        on_artifact_picked_up, identity, state)``
        → ``_select_tree(is_quest_artifact, touched_state, new_state)``.
    """
    new_state, new_gi = pickup_brax(state, rng, ground_items, branch, level)

    from Nethax.nethax.subsystems.quest import (
        on_artifact_picked_up, _ARTIFACT_IDX_BY_ROLE,
    )
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)
    picked_type_id = ground_items.type_id[branch, level, row, col, 0].astype(jnp.int16)
    role_idx = jnp.clip(state.player_role.astype(jnp.int32),
                       0, _ARTIFACT_IDX_BY_ROLE.shape[0] - 1)
    quest_art_id = _ARTIFACT_IDX_BY_ROLE[role_idx].astype(jnp.int16)
    is_quest_artifact = ((picked_type_id == quest_art_id)
                         & (picked_type_id > jnp.int16(0)))
    touched_state = on_artifact_picked_up(new_state)
    new_state = _select_tree(is_quest_artifact, touched_state, new_state)
    return new_state, new_gi


def handle_drop_brax(state, rng, ground_items: Item, branch: int, level: int) -> tuple:
    """Brax-style rewrite of ``inventory.handle_drop``.

    Conds flattened: 0 — scan-only.
    """
    def _find_occupied(carry, idx):
        found, slot = carry
        occupied = state.inventory.items.category[idx] != 0
        slot  = jnp.where(~found & occupied, idx, slot)
        found = found | occupied
        return (found, slot), None

    (_, first_slot), _ = lax.scan(
        _find_occupied, (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32),
    )
    return drop_brax(state, rng, ground_items, branch, level, first_slot)


def handle_wield_brax(state, rng):
    """Brax-style flattened rewrite of ``inventory.handle_wield``.

    Conds flattened: 1
      - WIELD_DAGGER message ``lax.cond(found_weapon, emit, identity, msgs)``
        → ``_select_tree(found_weapon, emitted, messages)``.

    NOTE: ``mark_violated_if`` is called unconditionally at tail
    (delegates internally to ``jnp.where`` — no extra ``lax.cond`` site
    introduced by this function).
    """
    def _find_weapon(carry, idx):
        found, slot = carry
        is_weapon = state.inventory.items.category[idx] == jnp.int8(ItemCategory.WEAPON)
        slot  = jnp.where(~found & is_weapon, idx, slot)
        found = found | is_weapon
        return (found, slot), None

    (found_weapon, first_weapon), _ = lax.scan(
        _find_weapon, (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32),
    )

    from Nethax.nethax.subsystems.pending_action import resolve_slot
    chosen = resolve_slot(state, first_weapon)
    safe_chosen = jnp.clip(chosen, 0, MAX_INVENTORY_SLOTS - 1)
    chosen_is_weapon = (
        state.inventory.items.category[safe_chosen]
        == jnp.int8(ItemCategory.WEAPON)
    ) & (state.inventory.items.quantity[safe_chosen] > jnp.int16(0))
    weapon_slot = jnp.where(chosen_is_weapon, safe_chosen, first_weapon).astype(jnp.int32)
    new_state = wield_brax(state, weapon_slot)
    new_inv = new_state.inventory.replace(wielded_artifact_idx=jnp.int8(-1))
    new_state = new_state.replace(inventory=new_inv)
    from Nethax.nethax.subsystems.artifact_powers import apply_artifact_intrinsics
    new_state = apply_artifact_intrinsics(new_state)

    # Flatten: lax.cond(found_weapon, emit, identity, messages).
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    emitted = _msg_emit(new_state.messages, int(_MsgId.WIELD_DAGGER))
    wielded_msg = _select_tree(found_weapon, emitted, new_state.messages)
    new_state = new_state.replace(messages=wielded_msg)

    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated_if
    return mark_violated_if(new_state, int(Conduct.WEAPONLESS), found_weapon)


def handle_unwield_brax(state, rng):
    """Brax-style rewrite of ``inventory.handle_unwield``.

    Conds flattened: 0 — pure delegate to ``unwield_brax``.
    """
    return unwield_brax(state)


def handle_wear_brax(state, rng):
    """Brax-style rewrite of ``inventory.handle_wear``.

    Conds flattened: 0 — scan + delegate.
    """
    def _find_armor(carry, idx):
        found, slot = carry
        is_armor = state.inventory.items.category[idx] == jnp.int8(ItemCategory.ARMOR)
        slot  = jnp.where(~found & is_armor, idx, slot)
        found = found | is_armor
        return (found, slot), None

    (_, first_armor), _ = lax.scan(
        _find_armor, (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32),
    )

    from Nethax.nethax.subsystems.pending_action import resolve_slot
    chosen = resolve_slot(state, first_armor)
    safe_chosen = jnp.clip(chosen, 0, MAX_INVENTORY_SLOTS - 1)
    chosen_is_armor = (
        state.inventory.items.category[safe_chosen]
        == jnp.int8(ItemCategory.ARMOR)
    ) & (state.inventory.items.quantity[safe_chosen] > jnp.int16(0))
    slot_idx = jnp.where(chosen_is_armor, safe_chosen, first_armor).astype(jnp.int32)
    return wear_armor_brax(state, slot_idx, ArmorSlot.BODY)


def step_brax(state, rng):
    """Brax-style rewrite of ``inventory.step``.

    Conds flattened: 0 — no-op identity in the Wave 3 original; preserved.
    """
    return state


def handle_name_brax(state, rng, slot_idx, name_bytes) -> "object":
    """Brax-style rewrite of ``inventory.handle_name``.

    Conds flattened: 0 — the Python-side branching here (str / bytes /
    ndarray normalization) runs at trace time, not in the HLO graph, so
    no ``lax.cond`` site exists to flatten.
    """
    from Nethax.nethax.subsystems.inventory import USER_NAME_LEN

    slot_i32 = jnp.int32(slot_idx)
    safe_slot = jnp.clip(slot_i32, 0, MAX_INVENTORY_SLOTS - 1)

    if isinstance(name_bytes, (bytes, bytearray)):
        padded = bytes(name_bytes)[:USER_NAME_LEN]
        padded = padded + b"\x00" * (USER_NAME_LEN - len(padded))
        name_row = jnp.array(list(padded), dtype=jnp.int8)
    elif isinstance(name_bytes, str):
        b = name_bytes.encode("ascii")[:USER_NAME_LEN]
        b = b + b"\x00" * (USER_NAME_LEN - len(b))
        name_row = jnp.array(list(b), dtype=jnp.int8)
    else:
        name_row = jnp.asarray(name_bytes, dtype=jnp.int8)
        cur_len = name_row.shape[0] if hasattr(name_row, "shape") else len(name_row)
        if cur_len < USER_NAME_LEN:
            pad = jnp.zeros((USER_NAME_LEN - cur_len,), dtype=jnp.int8)
            name_row = jnp.concatenate([name_row, pad], axis=0)
        elif cur_len > USER_NAME_LEN:
            name_row = name_row[:USER_NAME_LEN]

    new_user_names = state.inventory.user_names.at[safe_slot].set(name_row)
    new_inv = state.inventory.replace(user_names=new_user_names)
    return state.replace(inventory=new_inv)


# ===========================================================================
# SWALLOW (all public entry points *besides* try_engulf, which lives in
# combat_helpers_brax.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# release_from_engulf_brax
# ---------------------------------------------------------------------------
def release_from_engulf_brax(state):
    """Brax-style rewrite of ``swallow.release_from_engulf``.

    Conds flattened: 0 — pure data update; no branches.
    """
    new_swallow = state.swallow.replace(
        swallowed=jnp.bool_(False),
        engulfer_slot=jnp.int32(-1),
        digest_timer=jnp.int32(0),
        total_timer=jnp.int32(0),
    )
    return state.replace(swallow=new_swallow)


# ---------------------------------------------------------------------------
# digest_tick_brax
# ---------------------------------------------------------------------------
def digest_tick_brax(state, rng: jax.Array):
    """Brax-style flattened rewrite of ``swallow.digest_tick``.

    Conds flattened: 2
      1. inner ``lax.cond(should_release, release_from_engulf, identity, s2)``
         → ``_select_tree(should_release, released, s2)``.
      2. outer ``lax.cond(state.swallow.swallowed, _tick, identity, state)``
         → ``_select_tree(state.swallow.swallowed, ticked, state)``.

    RNG draw order preserved: a single ``rnd(rng, 6)`` call fires
    unconditionally (matching the original's behavior under JIT/vmap,
    where the ``rnd`` inside the ``_tick`` branch traces regardless).
    """
    sw = state.swallow

    # --- Compute the "ticked" branch unconditionally. ---
    new_total = sw.total_timer - jnp.int32(1)
    new_digest = sw.digest_timer - jnp.int32(1)

    # Digestion damage — vendor mhitu.c:1418: rnd(6)+1 per digest tick.
    dmg = rnd(rng, 6) + jnp.int32(1)
    do_damage = new_digest <= jnp.int32(0)
    applied_dmg = jnp.where(do_damage, dmg, jnp.int32(0))
    reset_digest = jnp.where(do_damage, jnp.int32(10), new_digest)

    new_hp = jnp.maximum(state.player_hp - applied_dmg, jnp.int32(0))
    new_done = state.done | (new_hp <= jnp.int32(0))

    slot = sw.engulfer_slot.astype(jnp.int32)
    safe_slot = jnp.clip(slot, 0, state.monster_ai.alive.shape[0] - 1)
    engulfer_alive = jnp.where(
        slot >= jnp.int32(0),
        state.monster_ai.alive[safe_slot],
        jnp.bool_(False),
    )

    should_release = (new_total <= jnp.int32(0)) | (~engulfer_alive)

    s2 = state.replace(
        player_hp=new_hp,
        done=new_done,
        swallow=sw.replace(
            total_timer=new_total,
            digest_timer=reset_digest,
        ),
    )

    # Flatten inner cond: select between released and s2 via pytree-where.
    released = release_from_engulf_brax(s2)
    ticked = _select_tree(should_release, released, s2)

    # Flatten outer cond: select between ticked and original state on
    # ``state.swallow.swallowed``.
    return _select_tree(state.swallow.swallowed, ticked, state)
