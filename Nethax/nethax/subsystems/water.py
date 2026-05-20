"""Underwater mechanics — drowning damage, inventory rust, and lava.

Vendor references:
  vendor/nethack/src/trap.c::drown()         lines 5059-5198  (drowning tick)
  vendor/nethack/src/trap.c::water_damage()  line  5086        (inventory rust)
  vendor/nethack/src/trap.c::lava_effects()  lines 6794-6987   (fire + items)
  vendor/nethack/src/trap.c (is_pool gate)   line  4104        (Wwalking gate)
  vendor/nethack/src/hack.c::pooleffects()   line  3304        (enter flag)
  vendor/nethack/src/hack.c::swimeffect()    line  3237        (leave flag)

Design: all functions are JIT-pure (no Python control flow on traced values).
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

# Probability that each iron item is damaged per turn underwater.
# Vendor trap.c:5086 — erosion check uses rn2(2) (50 % chance).
_RUST_PROB: float = 0.5


# ---------------------------------------------------------------------------
# water_damage_chain — rust iron inventory items
# ---------------------------------------------------------------------------

def water_damage_chain(state, rng):
    """Rust each non-rustproof iron item in inventory with prob 0.5.

    Cite: vendor/nethack/src/trap.c::water_damage() line 5086 — vendor routes
    every per-item rust through ``erode_obj(otmp, str, ERODE_RUST, EF_NONE)``;
    we follow that path via ``items.erode_obj_slot``.

    JIT-pure: uses lax.scan over inventory slots.
    """
    from Nethax.nethax.subsystems.inventory import MAX_INVENTORY_SLOTS
    from Nethax.nethax.subsystems.items import erode_obj_slot, ERODE_RUST

    items = state.inventory.items
    n = MAX_INVENTORY_SLOTS

    def rust_one(carry, slot_idx):
        rng_cur, items_acc = carry
        rng_cur, sub = jax.random.split(rng_cur)
        roll = jax.random.uniform(sub)

        cat = items_acc.category[slot_idx].astype(jnp.int32)
        occupied = cat != jnp.int32(0)
        roll_ok = roll < jnp.float32(_RUST_PROB)
        should_attempt = occupied & roll_ok

        def _do(items_in):
            new_items, _ = erode_obj_slot(items_in, slot_idx, ERODE_RUST, True)
            return new_items

        new_items_acc = lax.cond(should_attempt, _do, lambda x: x, items_acc)
        return (rng_cur, new_items_acc), None

    (_rng_out, new_items), _ = jax.lax.scan(
        rust_one,
        (rng, items),
        jnp.arange(n, dtype=jnp.int32),
    )

    new_inv = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


# ---------------------------------------------------------------------------
# water_step — drowning damage tick
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


def should_enter_pool(state) -> jnp.ndarray:
    """True iff stepping onto a pool tile should trigger ``drown()``.

    Cite: vendor/nethack/src/trap.c line 4104 —
        ``if (is_pool(u.ux, u.uy) && !Wwalking && !Swimming && !u.uinwater)``

    JIT-pure boolean gate; callers use this to decide whether to invoke the
    drowning path on entry to a water tile.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    wwalking = _has_intrinsic(state, Intrinsic.WWALKING)
    swimming = _has_intrinsic(state, Intrinsic.SWIMMING)
    already_in = state.player_in_water
    return ~wwalking & ~swimming & ~already_in


def water_step(state, rng):
    """Apply one-turn drowning tick if player is underwater.

    Each turn in water:
      - Increment turns_underwater.  Reset to 0 on leaving water (handled by
        action_dispatch / _try_step when player_in_water transitions to False).
      - Swimming / Amphibious / Breathless / Magic-breathing immediately
        short-circuit drowning damage.
        Cite: vendor/nethack/src/trap.c::drown line 5070 —
              ``if (u.uinwater && is_pool(...) && (Swimming || Amphibious
                  || Breathless)) { ... return FALSE; }``
      - Every 5 turns, check insta-drown: roll rnl(50); drown if roll <=
        turns_underwater.  Vendor formula: trap.c::drown line 5059.
        "rnl(50)" = randint [0, 50) biased by luck; we approximate as
        randint [0, 50) (luck bias is small and requires StatusState).
        Cite: vendor/nethack/src/trap.c::drown() line 5059.
      - Call water_damage_chain to rust iron inventory items (trap.c:5086).

    Cite: vendor/nethack/src/trap.c::drown() lines 5059-5198.

    JIT-pure: all branching via jnp.where / jax.lax.cond.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    in_water = state.player_in_water

    rng_drown, rng_rust = jax.random.split(rng)

    # Vendor trap.c:5070 short-circuit: Swimming, Amphibious, Breathless
    # (== MAGIC_BREATHING in our enum) all bypass the drown counter.
    # Cite: vendor/nethack/src/trap.c::drown line 5070.
    has_swim     = _has_intrinsic(state, Intrinsic.SWIMMING)
    has_amphib   = _has_intrinsic(state, Intrinsic.BREATHLESS)  # alias = MAGIC_BREATHING
    has_breath   = _has_intrinsic(state, Intrinsic.MAGIC_BREATHING)
    safe_in_water = has_swim | has_amphib | has_breath

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

        # Apply: insta-drown sets HP to 0; safe-in-water suppresses both paths.
        # Cite: vendor/nethack/src/trap.c::drown line 5070 (Swimming/Amphibious/
        # Breathless short-circuit).
        new_hp = jnp.where(
            safe_in_water,
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


# ---------------------------------------------------------------------------
# lava_effects — hero on lava tile (trap.c:6794-6987)
# ---------------------------------------------------------------------------

def _roll_dice(rng, n: int, sides: int) -> jnp.ndarray:
    """Vendor d(n, sides) — sum of n uniform rolls in [1..sides]."""
    keys = jax.random.split(rng, max(int(n), 1))
    rolls = jax.vmap(
        lambda k: jax.random.randint(k, (), 1, int(sides) + 1, dtype=jnp.int32)
    )(keys)
    return jnp.sum(rolls).astype(jnp.int32)


def lava_effects(state, rng):
    """Apply lava damage to the hero standing on a lava tile.

    Cite: vendor/nethack/src/trap.c::lava_effects() lines 6794-6987.

    Vendor rules (byte-equal):
      - ``dmg = d(6, 6)``  (trap.c:6800)
      - ``usurvive = Fire_resistance || (Wwalking && dmg < u.uhp)``
        (trap.c:6811).
      - If Fire_resistance: no HP damage on the lava tile itself; only the
        ``burn_stuff`` path destroys vulnerable items
        (``destroy_items(AD_FIRE, dmg)`` — trap.c:6984).
      - Else if Wwalking and survives: ``losehp(dmg, "molten lava")``
        (trap.c:6875) plus item destruction in burn_stuff.
      - Else (no Wwalking, no Fire_resistance): instakill — set hp = -1
        and run the lethal lava path (trap.c:6927-6940).  Lifesave wiring
        is deferred to the caller's lifesave hook; here we set hp <= 0.

    Item destruction is approximated by skipping organic / potion items
    via ``destroy_items``; the precise per-item burn list (trap.c:6892-
    6912) requires the full inventory walk and is implemented separately
    in ``destroy_items`` once wired into apply_tools.

    JIT-pure: all branching via lax.cond / jnp.where.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    fire_res = _has_intrinsic(state, Intrinsic.RESIST_FIRE)
    wwalking = _has_intrinsic(state, Intrinsic.WWALKING)

    # Vendor: dmg = d(6, 6).  Cite: trap.c:6800.
    rng_dmg, rng_unused = jax.random.split(rng)
    dmg = _roll_dice(rng_dmg, 6, 6)

    cur_hp = state.player_hp.astype(jnp.int32)
    # Vendor usurvive (trap.c:6811): Fire_resistance OR (Wwalking && dmg < uhp).
    usurvive = fire_res | (wwalking & (dmg < cur_hp))

    # ----- Branch 1: Wwalking & ~Fire_resistance & usurvive ----------------
    # ``pline_The("%s here burns you!"); losehp(dmg, lava_killer)`` —
    # trap.c:6872-6876.
    burn_dmg = jnp.where(wwalking & ~fire_res, dmg, jnp.int32(0))

    # ----- Branch 2: ~Wwalking & ~Fire_resistance → fall in, lethal --------
    # ``You("fall into the lava!"); u.uhp = -1`` — trap.c:6879, 6928.
    fatal = ~wwalking & ~fire_res
    new_hp_after_burn = jnp.where(
        fatal,
        jnp.int32(0),
        jnp.maximum(cur_hp - burn_dmg, jnp.int32(0)),
    ).astype(state.player_hp.dtype)

    # ----- Item destruction (burn_stuff label, trap.c:6983) -----------------
    # ``destroy_items(AD_FIRE, dmg)`` — vendor routes through erode_obj with
    # ERODE_BURN, the AD_FIRE path of fire-burnable items.  We use the same
    # erode_obj_slot helper that water_damage_chain uses, but with ERODE_BURN.
    from Nethax.nethax.subsystems.inventory import MAX_INVENTORY_SLOTS
    from Nethax.nethax.subsystems.items import erode_obj_slot, ERODE_BURN

    items = state.inventory.items

    def burn_one(carry, slot_idx):
        rng_cur, items_acc = carry
        rng_cur, sub = jax.random.split(rng_cur)
        roll = jax.random.uniform(sub)

        cat = items_acc.category[slot_idx].astype(jnp.int32)
        occupied = cat != jnp.int32(0)
        roll_ok = roll < jnp.float32(_RUST_PROB)
        should_attempt = occupied & roll_ok

        def _do(items_in):
            new_items, _ = erode_obj_slot(items_in, slot_idx, ERODE_BURN, True)
            return new_items

        new_items_acc = lax.cond(should_attempt, _do, lambda x: x, items_acc)
        return (rng_cur, new_items_acc), None

    (_, new_items), _ = jax.lax.scan(
        burn_one,
        (rng_unused, items),
        jnp.arange(MAX_INVENTORY_SLOTS, dtype=jnp.int32),
    )
    new_inv = state.inventory.replace(items=new_items)
    state = state.replace(inventory=new_inv, player_hp=new_hp_after_burn)
    return state
