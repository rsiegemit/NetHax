"""Underwater mechanics — drowning damage and inventory rust.

Vendor references:
  vendor/nethack/src/trap.c::drown()        lines 5059-5195  (drowning tick)
  vendor/nethack/src/trap.c::water_damage() lines 5086       (inventory rust)
  vendor/nethack/src/hack.c::pooleffects()  line  3304       (enter-water flag)
  vendor/nethack/src/hack.c::swimeffect()   line  3237       (leave-water flag)

Design: all functions are JIT-pure (no Python control flow on traced values).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum rust level (vendor obj.h oeroded:2 bitfield — max value 3).
_MAX_ERODE: int = 3

# Probability that each iron item is damaged per turn underwater.
# Vendor trap.c:5086 — erosion check uses rn2(2) (50 % chance).
_RUST_PROB: float = 0.5


# ---------------------------------------------------------------------------
# water_damage_chain — rust iron inventory items
# ---------------------------------------------------------------------------

def water_damage_chain(state, rng):
    """Increment oeroded by 1 on each non-rustproof iron item with prob 0.5.

    Cite: vendor/nethack/src/trap.c::water_damage() line 5086.

    Items are "iron" when their erosion material class is 1 (rustprone).
    We approximate this by checking item category (WEAPON=1, ARMOR=3) and
    treating all such items as rust-susceptible unless oerodeproof.
    This matches the vendor approximation for the common case.

    JIT-pure: uses lax.scan over inventory slots.
    """
    from Nethax.nethax.subsystems.inventory import MAX_INVENTORY_SLOTS, ItemCategory
    from Nethax.nethax.obs.inv_strs import _OBJECT_EROSION_CLASS

    items = state.inventory.items
    n = MAX_INVENTORY_SLOTS

    def rust_one(carry, slot_idx):
        rng_cur, oeroded_arr = carry
        rng_cur, sub = jax.random.split(rng_cur)
        roll = jax.random.uniform(sub)

        cat = items.category[slot_idx].astype(jnp.int32)
        occupied = cat != jnp.int32(0)

        type_id = items.type_id[slot_idx].astype(jnp.int32)
        # Clip to erosion-class table bounds.
        safe_type = jnp.clip(type_id, 0, _OBJECT_EROSION_CLASS.shape[0] - 1)
        emat = _OBJECT_EROSION_CLASS[safe_type].astype(jnp.int32)
        is_rustprone = emat == jnp.int32(1)

        rustproof = items.oerodeproof[slot_idx]
        cur = oeroded_arr[slot_idx].astype(jnp.int32)

        do_rust = occupied & is_rustprone & ~rustproof & (roll < _RUST_PROB)
        new_val = jnp.where(
            do_rust,
            jnp.minimum(cur + jnp.int32(1), jnp.int32(_MAX_ERODE)),
            cur,
        ).astype(jnp.int8)
        new_arr = oeroded_arr.at[slot_idx].set(new_val)
        return (rng_cur, new_arr), None

    (_rng_out, new_oeroded), _ = jax.lax.scan(
        rust_one,
        (rng, items.oeroded),
        jnp.arange(n, dtype=jnp.int32),
    )

    new_items = items.replace(oeroded=new_oeroded)
    new_inv = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


# ---------------------------------------------------------------------------
# water_step — drowning damage tick
# ---------------------------------------------------------------------------

def water_step(state, rng):
    """Apply one-turn drowning tick if player is underwater.

    Each turn in water:
      - Deal rnd(6) HP damage (vendor trap.c:5059 drown()).
        Skipped when player has MAGIC_BREATHING intrinsic (vendor you.h
        MAGICAL_BREATHING = 52; amulet of magical breathing, air elemental form).
        Cite: vendor/nethack/include/you.h MAGIC_BREATHING; vendor prop.h:52.
        TODO: vendor drown() has a 1-in-N per-turn insta-drown check where N
        decreases with turns underwater (trap.c:5059); for now rnd(6) per turn.
      - Call water_damage_chain to rust iron inventory items (trap.c:5086).

    Cite: vendor/nethack/src/trap.c::drown() lines 5059-5195.

    JIT-pure: all branching via jnp.where / jax.lax.cond.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    in_water = state.player_in_water

    rng_dmg, rng_rust = jax.random.split(rng)

    # MAGIC_BREATHING (MAGICAL_BREATHING = 52, vendor prop.h) suppresses
    # drowning damage — amulet of magical breathing, air elemental polyform.
    # Cite: vendor/nethack/include/you.h MAGIC_BREATHING.
    has_magic_breath = (
        state.status.intrinsics[int(Intrinsic.MAGIC_BREATHING)]
        | (state.status.timed_intrinsics[int(Intrinsic.MAGIC_BREATHING)] > jnp.int32(0))
    )

    # rnd(6) = uniform [1, 6] — vendor trap.c:5059 uses rnd(6).
    dmg = jax.random.randint(rng_dmg, (), minval=1, maxval=7, dtype=jnp.int32)

    def _apply_drown(s):
        new_hp = jnp.where(
            has_magic_breath,
            s.player_hp,
            jnp.maximum(
                jnp.int32(0),
                s.player_hp.astype(jnp.int32) - dmg,
            ).astype(s.player_hp.dtype),
        )
        s = s.replace(player_hp=new_hp)
        s = water_damage_chain(s, rng_rust)
        return s

    return jax.lax.cond(in_water, _apply_drown, lambda s: s, state)
