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
      - Increment turns_underwater.  Reset to 0 on leaving water (handled by
        action_dispatch / _try_step when player_in_water transitions to False).
      - Every 5 turns, check insta-drown: roll rnl(50); drown if roll <=
        turns_underwater.  Vendor formula: trap.c::drown line 5059.
        "rnl(50)" = randint [0, 50) biased by luck; we approximate as
        randint [0, 50) (luck bias is small and requires StatusState).
        Cite: vendor/nethack/src/trap.c::drown() line 5059.
      - Skipped entirely when player has MAGIC_BREATHING intrinsic.
        Cite: vendor/nethack/include/you.h MAGIC_BREATHING; vendor prop.h:52.
      - Call water_damage_chain to rust iron inventory items (trap.c:5086).

    Cite: vendor/nethack/src/trap.c::drown() lines 5059-5195.

    JIT-pure: all branching via jnp.where / jax.lax.cond.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    in_water = state.player_in_water

    rng_drown, rng_rust = jax.random.split(rng)

    # MAGIC_BREATHING (MAGICAL_BREATHING = 52, vendor prop.h) suppresses
    # drowning damage — amulet of magical breathing, air elemental polyform.
    # Cite: vendor/nethack/include/you.h MAGIC_BREATHING.
    has_magic_breath = (
        state.status.intrinsics[int(Intrinsic.MAGIC_BREATHING)]
        | (state.status.timed_intrinsics[int(Intrinsic.MAGIC_BREATHING)] > jnp.int32(0))
    )

    def _apply_drown(s):
        # Increment turns_underwater each tick in water.
        new_turns = (s.turns_underwater.astype(jnp.int32) + jnp.int32(1)).astype(
            jnp.int16
        )
        s = s.replace(turns_underwater=new_turns)

        # Every 5 turns: insta-drown check.
        # Vendor trap.c::drown line 5059: if (rnl(50) <= turns_underwater) → drown.
        # We approximate rnl(50) as uniform [0, 50).
        on_check_turn = (new_turns % jnp.int16(5)) == jnp.int16(0)
        roll = jax.random.randint(rng_drown, (), minval=0, maxval=50, dtype=jnp.int32)
        insta_drown = on_check_turn & (roll <= new_turns.astype(jnp.int32))

        # Apply: insta-drown sets HP to 0; MAGIC_BREATHING suppresses both paths.
        new_hp = jnp.where(
            has_magic_breath,
            s.player_hp,
            jnp.where(
                insta_drown,
                jnp.int32(0),
                s.player_hp.astype(jnp.int32),
            ).astype(s.player_hp.dtype),
        )
        s = s.replace(player_hp=new_hp)
        s = water_damage_chain(s, rng_rust)
        return s

    def _leave_water(s):
        # Reset counter when not in water this tick.
        return s.replace(turns_underwater=jnp.int16(0))

    return jax.lax.cond(in_water, _apply_drown, _leave_water, state)
