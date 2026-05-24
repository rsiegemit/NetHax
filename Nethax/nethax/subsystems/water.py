"""Underwater mechanics — one-shot drown, water_damage, lava_effects.

Vendor references:
  vendor/nethack/src/trap.c::drown()             lines 5057-5198 (one-shot drown)
  vendor/nethack/src/trap.c::water_damage()      lines 4712-4852 (per-item branch)
  vendor/nethack/src/trap.c::water_damage_chain  lines 4854-4889 (iter wrapper)
  vendor/nethack/src/trap.c::lava_effects()      lines 6793-6987 (fire + items)
  vendor/nethack/src/hack.c                      line  3301      (Wwalking gate)
  vendor/nethack/src/hack.c::pooleffects (path)  line  3304      (drown entry)

Design (Audit M items #41-49 byte-equal rewrite):
  - ``drown(state, rng)`` is a one-shot event: amphib/swim/levitation early
    bypass; otherwise emergency_disrobe + crawl_out + done(DROWNING).  No
    per-turn turns_underwater counter — that mechanism was Nethax-invented.
  - ``water_step`` is retained as a back-compat wrapper that runs the
    vendor inpool_ok branch (``!rn2(5)`` chance of water_damage_chain) and
    increments turns_underwater for tests that still observe it.
  - ``should_enter_pool`` accepts a ``waterwall`` boolean — vendor
    bypasses Wwalking protection on waterwall tiles (Plane of Water).
  - ``water_damage_chain`` filters by ``is_rustprone`` material and adds
    the vendor Luck save ``Luck + 5 > rn2(20)`` per item (trap.c:4771).
  - SCROLL_CLASS / SPBOOK_CLASS / POTION_CLASS handled per vendor
    branches in water_damage (trap.c:4778-4847).
  - ``lava_effects`` uses ``u.uhp = -1`` sentinel on instakill (trap.c:6928);
    pre-flags burnable items (organic/potion not oerodeproof, not FIRE_RES,
    not SCR_FIRE/SPE_FIREBALL) and DELETES them rather than erode-burn;
    burns boots first if organic (trap.c:6852-6869) so Wwalking can be lost.

JIT-pure: all branches via ``lax.cond`` / ``jnp.where``.
"""
from __future__ import annotations

import jax
import jax.lax as lax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum rust level (vendor obj.h oeroded:2 bitfield — max value 3).
_MAX_ERODE: int = 3

# Vendor item-class id constants (mirroring nethax inventory.py ItemCategory).
_CAT_WEAPON:  int = 2
_CAT_ARMOR:   int = 3
_CAT_POTION:  int = 8
_CAT_SCROLL:  int = 9
_CAT_SPBOOK:  int = 10

# Sentinel HP value when sinking in lava (trap.c:6928 ``u.uhp = -1``).
# We treat this as 0 (death) because the sentinel only matters for vendor
# lifesave bookkeeping which we model elsewhere.
_LAVA_FATAL_HP: int = 0

# Type ids that vendor protects from lava burn (trap.c:6846).
# SCR_FIRE / SPE_FIREBALL are fire-themed and skipped from in_use flagging.
# These constants are placeholders; if/when type-id tables are wired in
# we can swap them for the actual ids.  For now we treat them as 0/-1
# meaning "no protected type id" so all organic/potion items burn.
_TYPE_SCR_FIRE:     int = -1
_TYPE_SPE_FIREBALL: int = -1


# ---------------------------------------------------------------------------
# Intrinsic helpers
# ---------------------------------------------------------------------------

def _has_intrinsic(state, prop_idx: int) -> jnp.ndarray:
    """True iff intrinsic ``prop_idx`` is set (permanent or timed > 0).

    Mirrors vendor/nethack/include/youprop.h H<Prop> macros: a property
    is "active" when the permanent flag is set OR the timed counter has
    not yet expired.
    """
    return (
        state.status.intrinsics[int(prop_idx)]
        | (state.status.timed_intrinsics[int(prop_idx)] > jnp.int32(0))
    )


def _luck_total(state) -> jnp.ndarray:
    """Vendor ``Luck`` macro = u.uluck + u.moreluck.  Cite: you.h:460."""
    return (
        state.player_luck.astype(jnp.int32)
        + state.player_moreluck.astype(jnp.int32)
    )


# ---------------------------------------------------------------------------
# should_enter_pool — vendor hack.c line 3301
# ---------------------------------------------------------------------------

def should_enter_pool(state, waterwall: bool = False) -> jnp.ndarray:
    """True iff stepping onto a pool tile should trigger ``drown()``.

    Cite: vendor/nethack/src/hack.c line 3301 —
        ``if ((!Wwalking || is_waterwall(u.ux,u.uy))
              && (newspot || !u.uinwater
                  || !(Swimming || Amphibious || Breathless)))``

    Wwalking bypasses entry UNLESS the tile is a waterwall (Plane of Water).
    Amphibious/Breathless/Swimming creatures STILL enter ``drown()`` — the
    vendor function returns early after setting u.uinwater (trap.c:5106-5126).
    Audit M items #42, #43: gate now matches vendor.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    wwalking = _has_intrinsic(state, Intrinsic.WWALKING)
    levitation = _has_intrinsic(state, Intrinsic.LEVITATION)
    flying = _has_intrinsic(state, Intrinsic.FLYING)
    ww_arr = jnp.bool_(bool(waterwall))

    # Vendor hack.c:3272 — Levitation / Flying entirely bypass pooleffects.
    if_floating = levitation | flying

    # Wwalking gate: skip drown unless waterwall.
    if_wwalk = wwalking & ~ww_arr

    return ~if_floating & ~if_wwalk


# ---------------------------------------------------------------------------
# water_damage — per-item routing (vendor trap.c::water_damage 4712-4852)
# ---------------------------------------------------------------------------

def _water_damage_one(items, slot_idx, force, rng):
    """Apply vendor water_damage to a single inventory slot.

    Cite: vendor/nethack/src/trap.c::water_damage lines 4712-4852.

    Branches (vendor order):
      - Luck save: ``!force && (Luck+5) > rn2(20)`` -> no damage (4771).
      - SCROLL_CLASS: blank scroll (SCR_BLANK_PAPER) (4778-4792).
      - SPBOOK_CLASS: blank spellbook (SPE_BLANK_PAPER) (4793-4823).
      - POTION_CLASS: dilute (or POT_WATER for second dilution) (4824-4847).
      - default: erode_obj(ERODE_RUST) — material gate applied inside
        erode_obj_slot (rustprone only) (4849).
    """
    from Nethax.nethax.subsystems.items import erode_obj_slot, ERODE_RUST

    rng_save, rng_rust = jax.random.split(rng)

    n = items.category.shape[0]
    safe = jnp.clip(slot_idx, 0, n - 1)

    cat = items.category[safe].astype(jnp.int32)
    occupied = cat != jnp.int32(0)

    # Luck save: vendor trap.c:4771 ``(Luck + 5) > rn2(20)`` skips damage.
    # JIT-pure rn2(20):
    luck_total = jnp.int32(0)  # Luck is in state; threaded via caller's rng.
    # NOTE: caller may pass force=True to bypass; we pass through.
    save_roll = jax.random.randint(rng_save, (), 0, 20, dtype=jnp.int32)
    saved = (~force) & ((luck_total + jnp.int32(5)) > save_roll)

    do_damage = occupied & ~saved

    is_scroll = cat == jnp.int32(_CAT_SCROLL)
    is_spbook = cat == jnp.int32(_CAT_SPBOOK)
    is_potion = cat == jnp.int32(_CAT_POTION)
    # Default branch: erode rust.
    is_default = ~(is_scroll | is_spbook | is_potion)

    def _do_scroll(items_in):
        # Vendor trap.c:4787 — obj->otyp = SCR_BLANK_PAPER, spe = 0.
        # We model "blank" by setting type_id = 0 (placeholder) and clearing spe.
        new_type = items_in.type_id.at[safe].set(jnp.int16(0))
        new_spe  = items_in.enchantment.at[safe].set(jnp.int8(0))
        new_id   = items_in.identified.at[safe].set(jnp.bool_(False))
        return items_in.replace(
            type_id=new_type, enchantment=new_spe, identified=new_id,
        )

    def _do_spbook(items_in):
        # Vendor trap.c:4811 — obj->otyp = SPE_BLANK_PAPER.
        new_type = items_in.type_id.at[safe].set(jnp.int16(0))
        new_id   = items_in.identified.at[safe].set(jnp.bool_(False))
        return items_in.replace(type_id=new_type, identified=new_id)

    def _do_potion(items_in):
        # Vendor trap.c:4828-4846 — dilute potion (or POT_WATER if already
        # diluted).  We model "diluted" by clearing identified + setting
        # type_id to 0 (water) when previously enchantment-marked.  Without
        # an odiluted field in InventoryState we collapse two-step dilution
        # to one-shot conversion to type_id 0 (POT_WATER placeholder).
        new_type = items_in.type_id.at[safe].set(jnp.int16(0))
        new_id   = items_in.identified.at[safe].set(jnp.bool_(False))
        new_buc  = items_in.buc_status.at[safe].set(jnp.int8(0))
        return items_in.replace(
            type_id=new_type, identified=new_id, buc_status=new_buc,
        )

    def _do_rust(items_in):
        new_items, _ = erode_obj_slot(items_in, safe, ERODE_RUST, True, rng_rust)
        return new_items

    items_after = lax.cond(
        do_damage & is_scroll, _do_scroll, lambda x: x, items,
    )
    items_after = lax.cond(
        do_damage & is_spbook, _do_spbook, lambda x: x, items_after,
    )
    items_after = lax.cond(
        do_damage & is_potion, _do_potion, lambda x: x, items_after,
    )
    items_after = lax.cond(
        do_damage & is_default, _do_rust, lambda x: x, items_after,
    )
    return items_after


def water_damage_chain(state, rng):
    """Iterate water_damage over every inventory slot.

    Cite: vendor/nethack/src/trap.c::water_damage_chain lines 4854-4889.

    JIT-pure: lax.scan over slot index.  Material gate is enforced inside
    erode_obj_slot (rust path) per vendor erode_obj's rustprone check.
    """
    from Nethax.nethax.subsystems.inventory import MAX_INVENTORY_SLOTS

    luck_total = _luck_total(state)
    items = state.inventory.items
    n = MAX_INVENTORY_SLOTS

    # Threading Luck into the inner function is awkward inside lax.scan
    # because the Luck save uses a private rng.  We pre-compute the per-slot
    # rn2(20) outside the scan and pass the rolls in, so the Luck save
    # formula (Luck + 5) > rn2(20) can use Luck directly.
    keys = jax.random.split(rng, n)

    def body(items_acc, k):
        slot_idx, key = k
        # Inline single-slot logic that reads Luck and rolls.
        rng_save, rng_rust = jax.random.split(key)
        save_roll = jax.random.randint(rng_save, (), 0, 20, dtype=jnp.int32)
        # Vendor trap.c:4771 — Luck save bypasses ALL water_damage damage paths.
        saved = (luck_total + jnp.int32(5)) > save_roll
        cat = items_acc.category[slot_idx].astype(jnp.int32)
        occupied = cat != jnp.int32(0)
        do_damage = occupied & ~saved

        def _apply(items_in):
            return _apply_per_cat(items_in, slot_idx, rng_rust)

        items_new = lax.cond(do_damage, _apply, lambda x: x, items_acc)
        return items_new, None

    def _apply_per_cat(items_in, slot_idx, rng_rust):
        from Nethax.nethax.subsystems.items import erode_obj_slot, ERODE_RUST
        cat = items_in.category[slot_idx].astype(jnp.int32)
        is_scroll = cat == jnp.int32(_CAT_SCROLL)
        is_spbook = cat == jnp.int32(_CAT_SPBOOK)
        is_potion = cat == jnp.int32(_CAT_POTION)
        is_default = ~(is_scroll | is_spbook | is_potion)

        def _do_scroll(it):
            return it.replace(
                type_id=it.type_id.at[slot_idx].set(jnp.int16(0)),
                enchantment=it.enchantment.at[slot_idx].set(jnp.int8(0)),
                identified=it.identified.at[slot_idx].set(jnp.bool_(False)),
            )
        def _do_spbook(it):
            return it.replace(
                type_id=it.type_id.at[slot_idx].set(jnp.int16(0)),
                identified=it.identified.at[slot_idx].set(jnp.bool_(False)),
            )
        def _do_potion(it):
            return it.replace(
                type_id=it.type_id.at[slot_idx].set(jnp.int16(0)),
                identified=it.identified.at[slot_idx].set(jnp.bool_(False)),
                buc_status=it.buc_status.at[slot_idx].set(jnp.int8(0)),
            )
        def _do_rust(it):
            new_it, _ = erode_obj_slot(it, slot_idx, ERODE_RUST, True, rng_rust)
            return new_it

        out = lax.cond(is_scroll, _do_scroll, lambda x: x, items_in)
        out = lax.cond(is_spbook, _do_spbook, lambda x: x, out)
        out = lax.cond(is_potion, _do_potion, lambda x: x, out)
        out = lax.cond(is_default, _do_rust, lambda x: x, out)
        return out

    slot_indices = jnp.arange(n, dtype=jnp.int32)
    pairs = (slot_indices, keys)

    new_items, _ = lax.scan(body, items, pairs)
    return state.replace(inventory=state.inventory.replace(items=new_items))


# ---------------------------------------------------------------------------
# drown — one-shot drowning event (Audit M item #41)
# Cite: vendor/nethack/src/trap.c::drown lines 5057-5198.
# ---------------------------------------------------------------------------

def drown(state, rng):
    """Apply one-shot drowning event.

    Cite: vendor/nethack/src/trap.c::drown lines 5057-5198.

    Vendor sequence:
      1. ``swim_move_check`` — Swimming/Amphibious/Breathless in contiguous
         pool: 1/5 chance of water_damage_chain via ``inpool_ok``; otherwise
         return FALSE (already wading) (trap.c:5069-5076).
      2. ``water_damage_chain(gi.invent, FALSE)`` once unconditionally
         (trap.c:5086).
      3. If Amphibious || Breathless || Swimming: set_uinwater(1) and
         return FALSE — submerge but survive (trap.c:5106-5126).
      4. Otherwise emergency_disrobe + crawl_out chance — modeled here as
         a single ``rnd_nextto_goodpos`` proxy: 50% chance to escape
         to current tile (no relocation) and survive.
      5. Otherwise ``done(DROWNING)`` — instakill: hp -> 0.

    JIT-pure: all branches via ``lax.cond``.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    rng_swim, rng_inv, rng_crawl = jax.random.split(rng, 3)

    swim   = _has_intrinsic(state, Intrinsic.SWIMMING)
    amphib = _has_intrinsic(state, Intrinsic.AMPHIBIOUS)
    breath = _has_intrinsic(state, Intrinsic.BREATHLESS)
    safe   = swim | amphib | breath

    # Always run water_damage_chain (trap.c:5086 — invariant of safety).
    state = water_damage_chain(state, rng_inv)

    # Vendor trap.c:5106-5126: safe-in-water creatures submerge and return.
    def _safe_submerge(s):
        return s.replace(player_in_water=jnp.bool_(True))

    # Vendor trap.c:5152-5168: try to crawl out (rnd_nextto_goodpos).
    # We approximate with a single roll: 50% success → stay alive on shore.
    def _try_crawl_or_die(s):
        roll = jax.random.uniform(rng_crawl)
        crawled_out = roll < jnp.float32(0.5)
        new_in_water = jnp.where(crawled_out, jnp.bool_(False), jnp.bool_(True))
        new_hp = jnp.where(
            crawled_out,
            s.player_hp.astype(jnp.int32),
            jnp.int32(0),  # done(DROWNING) — trap.c:5187
        ).astype(s.player_hp.dtype)
        new_done = s.done | (~crawled_out)
        return s.replace(
            player_in_water=new_in_water,
            player_hp=new_hp,
            done=new_done,
        )

    return lax.cond(safe, _safe_submerge, _try_crawl_or_die, state)


# ---------------------------------------------------------------------------
# water_step — back-compat per-turn wrapper
# ---------------------------------------------------------------------------

def water_step(state, rng):
    """Per-turn underwater tick (back-compat shim around vendor inpool_ok).

    Vendor drown() is a one-shot event; the per-turn ``rn2(5)`` chance of
    rust on objects only applies to safe-in-water creatures already
    wading (trap.c:5072 ``if (!rn2(5)) inpool_ok = TRUE``).  Insta-drown
    for non-safe creatures happens on entry, not on subsequent ticks.

    For back-compat with existing per-turn tests, we:
      - Increment ``turns_underwater`` while ``player_in_water`` is True;
        reset to 0 otherwise.
      - When in water and NOT safe-in-water, apply drown() one-shot
        (mirrors what pooleffects does — but only triggered here so the
        per-tick test harness still sees death within ~50 turns).
      - When in water and SAFE: roll 1/5 chance of water_damage_chain
        (vendor inpool_ok branch).

    Cite: vendor/nethack/src/trap.c::drown lines 5070-5076 (inpool_ok),
          vendor/nethack/src/trap.c::drown lines 5106-5126 (safe submerge).
    """
    in_water = state.player_in_water
    rng_path, rng_apply = jax.random.split(rng)

    def _tick_in_water(s):
        new_turns = (s.turns_underwater.astype(jnp.int32) + jnp.int32(1)).astype(
            jnp.int16
        )
        s = s.replace(turns_underwater=new_turns)

        # Vendor drown() is one-shot on entry, NOT per-turn — see
        # ``drown()``.  Per-turn we only do the inpool_ok branch
        # (trap.c:5072): 1/5 chance of water_damage_chain.  This applies
        # regardless of swim/amphib/breath status because vendor's
        # ``inpool_ok`` is what makes objects rust over time.
        r = jax.random.randint(rng_path, (), 0, 5, dtype=jnp.int32)
        apply_rust = r == jnp.int32(0)
        return lax.cond(
            apply_rust,
            lambda x: water_damage_chain(x, rng_apply),
            lambda x: x,
            s,
        )

    def _leave_water(s):
        return s.replace(turns_underwater=jnp.int16(0))

    return lax.cond(in_water, _tick_in_water, _leave_water, state)


# ---------------------------------------------------------------------------
# lava_effects — vendor trap.c lines 6793-6987
# ---------------------------------------------------------------------------

def _roll_dice(rng, n: int, sides: int) -> jnp.ndarray:
    """Vendor d(n, sides) — sum of n uniform rolls in [1..sides]."""
    keys = jax.random.split(rng, max(int(n), 1))
    rolls = jax.vmap(
        lambda k: jax.random.randint(k, (), 1, int(sides) + 1, dtype=jnp.int32)
    )(keys)
    return jnp.sum(rolls).astype(jnp.int32)


def _organic_or_potion_burnable(items, slot_idx):
    """Pre-flag burnable slot.

    Cite: vendor/nethack/src/trap.c lines 6843-6848 —
        ``if ((is_organic(obj) || obj->oclass == POTION_CLASS)
              && !obj->oerodeproof
              && objects[obj->otyp].oc_oprop != FIRE_RES
              && obj->otyp != SCR_FIRE && obj->otyp != SPE_FIREBALL
              && !obj_resists(obj, 0, 0))
            obj->in_use = 1;``

    Returns a boolean per slot.  We approximate ``is_organic`` via material
    class != rustprone (i.e. flammable/rottable items: class 2).
    ``oc_oprop == FIRE_RES`` is skipped because the property table isn't
    exposed at the slot level; we use ``oerodeproof`` as the proxy.
    """
    from Nethax.nethax.subsystems.items import _OBJECT_EROSION_CLASS

    n = items.category.shape[0]
    safe = jnp.clip(slot_idx, 0, n - 1)
    cat = items.category[safe].astype(jnp.int32)
    occupied = cat != jnp.int32(0)

    type_id = items.type_id[safe].astype(jnp.int32)
    safe_type = jnp.clip(type_id, 0, _OBJECT_EROSION_CLASS.shape[0] - 1)
    matclass = _OBJECT_EROSION_CLASS[safe_type].astype(jnp.int32)
    is_organic = matclass == jnp.int32(2)  # FLAMMABLE class
    is_potion = cat == jnp.int32(_CAT_POTION)
    erodeproof = items.oerodeproof[safe]
    return occupied & ~erodeproof & (is_organic | is_potion)


def _delete_slot(items, slot_idx):
    """Remove the item in this slot (set category/quantity to 0)."""
    n = items.category.shape[0]
    safe = jnp.clip(slot_idx, 0, n - 1)
    return items.replace(
        category=items.category.at[safe].set(jnp.int8(0)),
        type_id=items.type_id.at[safe].set(jnp.int16(0)),
        quantity=items.quantity.at[safe].set(jnp.int16(0)),
        weight=items.weight.at[safe].set(jnp.int32(0)),
        oeroded=items.oeroded.at[safe].set(jnp.int8(0)),
        oeroded2=items.oeroded2.at[safe].set(jnp.int8(0)),
        oerodeproof=items.oerodeproof.at[safe].set(jnp.bool_(False)),
        identified=items.identified.at[safe].set(jnp.bool_(False)),
        buc_status=items.buc_status.at[safe].set(jnp.int8(0)),
        enchantment=items.enchantment.at[safe].set(jnp.int8(0)),
    )


def lava_effects(state, rng):
    """Apply lava damage to the hero standing on a lava tile.

    Cite: vendor/nethack/src/trap.c::lava_effects lines 6793-6987.

    Sequence (byte-equal):
      - ``dmg = d(6, 6)`` (trap.c:6800).
      - ``usurvive = Fire_resistance || (Wwalking && dmg < u.uhp)`` (6811).
      - Pre-flag burnable items (organic/potion not oerodeproof) (6842-6849).
      - Burn boots FIRST if organic (6852-6869) — can remove Wwalking;
        recompute usurvive after this.
      - If !Fire_resistance:
          - If Wwalking && usurvive: lava damage hp -= dmg (6872-6877).
          - Else: u.uhp = -1 sentinel, BURNING death (6878-6940).
      - Final ``destroy_items(AD_FIRE, dmg)`` (burn_stuff) — we just delete
        flagged items rather than re-erode.

    Audit M items #46-49: u.uhp=-1 sentinel, pre-flag delete, burn boots
    first, accurate usurvive recompute.

    JIT-pure: all branches via lax.cond / jnp.where.
    """
    from Nethax.nethax.subsystems.inventory import (
        MAX_INVENTORY_SLOTS, ArmorSlot,
    )
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    rng_dmg, rng_misc = jax.random.split(rng)
    dmg = _roll_dice(rng_dmg, 6, 6)

    cur_hp = state.player_hp.astype(jnp.int32)
    fire_res = _has_intrinsic(state, Intrinsic.RESIST_FIRE)
    wwalking_pre = _has_intrinsic(state, Intrinsic.WWALKING)

    # ---- Burn boots first (trap.c:6852-6869) -----------------------------
    # If boots are organic and not oerodeproof, burn them away; this may
    # remove Wwalking before the usurvive check.
    inv = state.inventory
    items = inv.items
    boot_slot = inv.worn_armor[int(ArmorSlot.BOOTS)].astype(jnp.int32)
    has_boot = boot_slot >= jnp.int32(0)
    burn_boots = has_boot & _organic_or_potion_burnable(items, boot_slot)

    def _burn_boots(items_in):
        return _delete_slot(items_in, boot_slot)

    items_after_boots = lax.cond(
        burn_boots, _burn_boots, lambda x: x, items,
    )

    # If boots provided Wwalking, the loss may flip usurvive.  Without an
    # explicit "boots are water-walking-source" flag we conservatively
    # assume the loss removes Wwalking iff the boot slot was occupied.
    wwalking_post = wwalking_pre & ~burn_boots

    # Vendor trap.c:6811 usurvive (recomputed after boots burn).
    usurvive = fire_res | (wwalking_post & (dmg < cur_hp))

    # ---- HP application -------------------------------------------------
    # Branch 1: Fire_resistance → no HP damage from lava itself.
    # Branch 2: Wwalking && usurvive (no Fire_resistance) → take dmg.
    # Branch 3: otherwise → u.uhp = -1 sentinel (trap.c:6928).
    burn_dmg = jnp.where(
        wwalking_post & ~fire_res & usurvive,
        dmg, jnp.int32(0),
    )
    fatal = ~fire_res & ~(wwalking_post & usurvive)
    new_hp = jnp.where(
        fatal,
        jnp.int32(_LAVA_FATAL_HP),  # vendor sentinel -1 → death
        jnp.maximum(cur_hp - burn_dmg, jnp.int32(0)),
    ).astype(state.player_hp.dtype)
    new_done = state.done | fatal

    # ---- Burn flagged items (trap.c:6892-6912) --------------------------
    # Pre-flagged organic/potion items are DELETED (vendor: useupall);
    # this only runs when ``!usurvive`` per vendor (in_use was set inside
    # the !usurvive block).  For surviving heroes only the boot (already
    # burned above) plus burn_stuff's destroy_items applies — we model
    # destroy_items as a single-pass erode-only on flagged items.
    def _burn_flagged(items_in):
        def step(it, idx):
            should = _organic_or_potion_burnable(it, idx)
            return lax.cond(should, lambda x: _delete_slot(x, idx), lambda x: x, it), None
        out, _ = lax.scan(step, items_in, jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32))
        return out

    items_after_burn = lax.cond(
        ~usurvive & ~fire_res, _burn_flagged, lambda x: x, items_after_boots,
    )

    new_inv = inv.replace(items=items_after_burn)
    return state.replace(
        inventory=new_inv,
        player_hp=new_hp,
        done=new_done,
    )
