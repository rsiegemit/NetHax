"""Smoke tests for Nethax.nethax.subsystems.explode.

Vendor citations:
  vendor/nethack/src/explode.c::explode      (lines 199-696)
  vendor/nethack/src/explode.c::explosionmask (lines 26-115)

Coverage:
  * 3x3 AoE applies damage to all alive monsters in the ring.
  * Per-monster resistance halves damage (vendor line 538 (dam+1)/2).
  * Player damage when player tile is inside the ring.
  * Player resistance halves player damage.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.explode import (
    explode, AD_FIRE, AD_COLD,
)
from Nethax.nethax.constants.monsters import MR_FIRE
from Nethax.nethax.subsystems.status_effects import Intrinsic

_RNG = jax.random.PRNGKey(7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _place_monster(state, slot, row, col, hp=100, resists=0):
    """Place a live monster in slot ``slot`` with given hp and resist mask."""
    mai = state.monster_ai
    return state.replace(
        monster_ai=mai.replace(
            pos       = mai.pos.at[slot].set(jnp.array([row, col], dtype=jnp.int16)),
            hp        = mai.hp.at[slot].set(jnp.int32(hp)),
            hp_max    = mai.hp_max.at[slot].set(jnp.int32(hp)),
            alive     = mai.alive.at[slot].set(jnp.bool_(True)),
            resists   = mai.resists.at[slot].set(jnp.int32(resists)),
            entry_idx = mai.entry_idx.at[slot].set(jnp.int16(1)),
        )
    )


# ---------------------------------------------------------------------------
# Smoke test (task spec):
#   3 monsters at center + 2 neighbours, fire 6d6 with AD_FIRE.
#   All 3 take damage; resistant monster takes half.
# ---------------------------------------------------------------------------

def test_explode_fire_3x3_with_resist():
    """Three monsters in the 3x3, one fire-resistant.

    Damage is rolled once for the AoE; non-resist monsters take ``dam``;
    resist monster takes ``(dam + 1) // 2``.  We compare delta-HP ratios
    rather than the raw roll so the test is RNG-stable.
    """
    state = EnvState.default(_RNG)
    # Move player far away so the AoE doesn't catch the hero.
    state = state.replace(
        player_pos=jnp.array([0, 0], dtype=jnp.int16),
        player_hp=jnp.int32(50),
    )

    center = jnp.array([10, 10], dtype=jnp.int16)
    # Slot 1 = center, slot 2 = east neighbour, slot 3 = north neighbour
    # (we skip slot 0 since several spawn helpers reserve it).
    state = _place_monster(state, slot=1, row=10, col=10, hp=200)             # vulnerable
    state = _place_monster(state, slot=2, row=10, col=11, hp=200, resists=MR_FIRE)  # resistant
    state = _place_monster(state, slot=3, row=9,  col=10, hp=200)             # vulnerable

    out = explode(state, _RNG, center, AD_FIRE, n_dice=6, n_sides=6)

    hp_center = int(out.monster_ai.hp[1])
    hp_resist = int(out.monster_ai.hp[2])
    hp_north  = int(out.monster_ai.hp[3])

    dmg_center = 200 - hp_center
    dmg_resist = 200 - hp_resist
    dmg_north  = 200 - hp_north

    # All three were caught in the AoE.
    assert dmg_center > 0, f"center monster should take damage; dmg={dmg_center}"
    assert dmg_north  > 0, f"north monster should take damage; dmg={dmg_north}"
    assert dmg_resist > 0, "resistant monster should still take half damage"

    # Center and north must have taken the same full damage (single roll).
    assert dmg_center == dmg_north, (
        f"non-resistant monsters must share the same roll; "
        f"center={dmg_center}, north={dmg_north}"
    )

    # Resistant monster takes (dam+1)//2.
    assert dmg_resist == (dmg_center + 1) // 2, (
        f"resistant monster should take (dam+1)//2={(dmg_center+1)//2}; "
        f"got {dmg_resist} (full dam={dmg_center})"
    )


def test_explode_skips_monsters_outside_ring():
    """A monster two tiles away from the center must not be touched."""
    state = EnvState.default(_RNG)
    state = state.replace(player_pos=jnp.array([0, 0], dtype=jnp.int16))

    center = jnp.array([10, 10], dtype=jnp.int16)
    state = _place_monster(state, slot=1, row=10, col=10, hp=100)  # inside
    state = _place_monster(state, slot=2, row=10, col=13, hp=100)  # 3 cols away — outside

    out = explode(state, _RNG, center, AD_FIRE, n_dice=6, n_sides=6)

    assert int(out.monster_ai.hp[1]) < 100, "inside monster should be damaged"
    assert int(out.monster_ai.hp[2]) == 100, "outside monster must be untouched"


def test_explode_damages_player_when_in_ring():
    """Player at center takes damage; player far away does not."""
    state = EnvState.default(_RNG)
    state = state.replace(
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
        player_hp=jnp.int32(200),
    )

    center = jnp.array([10, 10], dtype=jnp.int16)
    out = explode(state, _RNG, center, AD_FIRE, n_dice=6, n_sides=6)
    assert int(out.player_hp) < 200, "player at center must take damage"


def test_explode_player_resist_halves_damage():
    """Player with RESIST_FIRE intrinsic takes (dam+1)//2."""
    rng = _RNG

    # Two parallel runs: with and without intrinsic; same RNG -> same roll.
    state_no = EnvState.default(rng).replace(
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
        player_hp=jnp.int32(200),
    )
    intr = state_no.status.intrinsics.at[int(Intrinsic.RESIST_FIRE)].set(True)
    state_yes = state_no.replace(status=state_no.status.replace(intrinsics=intr))

    center = jnp.array([10, 10], dtype=jnp.int16)
    out_no  = explode(state_no,  rng, center, AD_FIRE, n_dice=6, n_sides=6)
    out_yes = explode(state_yes, rng, center, AD_FIRE, n_dice=6, n_sides=6)

    dmg_no  = 200 - int(out_no.player_hp)
    dmg_yes = 200 - int(out_yes.player_hp)
    assert dmg_no > 0
    assert dmg_yes == (dmg_no + 1) // 2, (
        f"RESIST_FIRE should halve dam: dmg_yes={dmg_yes}, "
        f"expected (dmg_no+1)//2={(dmg_no+1)//2}"
    )


def test_explode_cold_routes_to_resist_cold():
    """AD_COLD damage is halved on monsters with MR_COLD."""
    from Nethax.nethax.constants.monsters import MR_COLD
    state = EnvState.default(_RNG)
    state = state.replace(player_pos=jnp.array([0, 0], dtype=jnp.int16))
    center = jnp.array([10, 10], dtype=jnp.int16)
    state = _place_monster(state, slot=1, row=10, col=10, hp=200)
    state = _place_monster(state, slot=2, row=10, col=11, hp=200, resists=MR_COLD)

    out = explode(state, _RNG, center, AD_COLD, n_dice=6, n_sides=6)

    dmg_full = 200 - int(out.monster_ai.hp[1])
    dmg_half = 200 - int(out.monster_ai.hp[2])
    assert dmg_full > 0
    assert dmg_half == (dmg_full + 1) // 2


def test_explode_jit():
    """JIT-pure: the call survives jax.jit compilation."""
    @jax.jit
    def _run(s, r):
        return explode(s, r, jnp.array([10, 10], dtype=jnp.int16),
                       AD_FIRE, 6, 6)

    state = EnvState.default(_RNG)
    state = state.replace(player_pos=jnp.array([0, 0], dtype=jnp.int16))
    state = _place_monster(state, slot=1, row=10, col=10, hp=100)

    out = _run(state, _RNG)
    assert int(out.monster_ai.hp[1]) < 100


# ---------------------------------------------------------------------------
# Scatter dispersal — vendor explode.c::scatter (lines 721-947).
# Each ground-stack slot inside the 3x3 ring has a 50% chance to be
# displaced into a random adjacent walkable tile.  POTION items shatter
# on impact and are zeroed out instead of being relocated.
# ---------------------------------------------------------------------------

def _scatter_state_with_items(weapon=True, potion=True):
    """Build a floor state with a weapon and/or potion at (10,10)/(10,11)."""
    from Nethax.nethax.state import StaticParams
    from Nethax.nethax.constants.tiles import TileType
    from Nethax.nethax.subsystems.inventory import ItemCategory

    static = StaticParams()
    state = EnvState.default(_RNG, static)
    floor_map = jnp.full((static.map_h, static.map_w), int(TileType.FLOOR),
                         dtype=jnp.int8)
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor_map),
        player_pos=jnp.array([0, 0], dtype=jnp.int16),  # hero out of AoE
    )
    gi = state.ground_items
    if weapon:
        gi = gi.replace(
            category=gi.category.at[0, 0, 10, 10, 0].set(
                jnp.int8(int(ItemCategory.WEAPON))),
            type_id=gi.type_id.at[0, 0, 10, 10, 0].set(jnp.int16(60)),
            quantity=gi.quantity.at[0, 0, 10, 10, 0].set(jnp.int16(1)),
        )
    if potion:
        gi = gi.replace(
            category=gi.category.at[0, 0, 10, 11, 0].set(
                jnp.int8(int(ItemCategory.POTION))),
            type_id=gi.type_id.at[0, 0, 10, 11, 0].set(jnp.int16(200)),
            quantity=gi.quantity.at[0, 0, 10, 11, 0].set(jnp.int16(1)),
        )
    return state.replace(ground_items=gi)


# JIT-compiled module-level helper — re-used across the scatter tests so
# we only compile once.  The center is fixed (10, 10).
@jax.jit
def _jit_scatter_explode(state, rng):
    return explode(state, rng, jnp.array([10, 10], dtype=jnp.int16),
                   AD_FIRE, 6, 6)


def test_explode_scatters_weapon_conserves_count():
    """A weapon inside the 3x3 ring is never destroyed; it may move.

    Vendor cite: explode.c::scatter (lines 721-947).  Non-fragile items
    (weapons, armor, scrolls, wands, ...) are flung but conserved.
    """
    from Nethax.nethax.subsystems.inventory import ItemCategory

    state = _scatter_state_with_items(weapon=True, potion=False)
    n_seeds = 40
    n_moved = 0
    n_preserved = 0
    for seed in range(n_seeds):
        o = _jit_scatter_explode(state, jax.random.PRNGKey(seed))
        if int(o.ground_items.category[0, 0, 10, 10, 0]) == 0:
            n_moved += 1
        # Weapon must be conserved on the level — one slot, anywhere.
        wcount = int(jnp.sum(
            (o.ground_items.category[0, 0]
             == int(ItemCategory.WEAPON))
            & (o.ground_items.type_id[0, 0] == 60)
        ))
        if wcount == 1:
            n_preserved += 1
    assert n_preserved == n_seeds, (
        f"weapon must be conserved every seed; got {n_preserved}/{n_seeds}"
    )
    # ~50% scatter probability — broad tolerance for small n.
    assert n_moved >= n_seeds // 5, (
        f"expected some scattering; observed {n_moved}/{n_seeds}"
    )


def test_explode_scatters_potion_breaks_in_place():
    """A POTION inside the 3x3 ring breaks (slot zeroed) when scatter fires.

    Vendor cite: explode.c::scatter line 808-813 + breaks() for POTION_CLASS.
    """
    from Nethax.nethax.subsystems.inventory import ItemCategory

    state = _scatter_state_with_items(weapon=False, potion=True)
    n_seeds = 40
    n_broken = 0
    for seed in range(n_seeds):
        o = _jit_scatter_explode(state, jax.random.PRNGKey(seed))
        pcount = int(jnp.sum(
            (o.ground_items.category[0, 0]
             == int(ItemCategory.POTION))
            & (o.ground_items.type_id[0, 0] == 200)
        ))
        # The potion is either intact (didn't trigger) or fully gone (broke).
        # It must NEVER appear at a NEW location (potions don't relocate).
        assert pcount in (0, 1), (
            f"potion must be 0 or 1; got {pcount}"
        )
        if pcount == 0:
            n_broken += 1
            assert int(o.ground_items.category[0, 0, 10, 11, 0]) == 0, (
                "broken potion source slot must be zeroed"
            )
    # ~50% break rate — broad tolerance.
    assert n_broken >= n_seeds // 5, (
        f"expected some breakage; observed {n_broken}/{n_seeds}"
    )


# ---------------------------------------------------------------------------
# Wand-backfire wiring: scroll-of-charging on a wand whose ``recharged``
# counter is already >=7 triggers a 3x3 AoE through ``_effect_charging``.
# Cite: vendor/nethack/src/read.c::wand_explode (line 2414);
#       vendor/nethack/src/zap.c retributive strike notes (lines 224-263).
# ---------------------------------------------------------------------------

def test_wand_backfire_aoe_damages_neighbour():
    """Scroll-of-charging on an overcharged WAN_FIRE: AoE hits a neighbour.

    Setup:
      * player at (10, 10)
      * neighbour monster at (10, 11) (Chebyshev=1 — inside the AoE)
      * far monster at (10, 15)      (outside the AoE)
      * inventory: scroll of charging + WAN_FIRE with recharged=7
        (next attempt overcharges and explodes)
    Expectation:
      * neighbour takes damage
      * far monster is untouched
      * the wand slot is cleared (category == 0)
    """
    from Nethax.nethax.subsystems.inventory import (
        InventoryState, make_item, ItemCategory,
    )
    from Nethax.nethax.subsystems.items_scrolls import (
        ScrollEffect, _SCROLL_BASE_ID, read_scroll,
    )
    from Nethax.nethax.subsystems.items_wands import WandEffect, ITEM_CATEGORY_WAND

    state = EnvState.default(_RNG)
    state = state.replace(
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
        player_hp=jnp.int32(200),
        player_hp_max=jnp.int32(200),
    )
    state = _place_monster(state, slot=1, row=10, col=11, hp=200)  # adjacent
    state = _place_monster(state, slot=2, row=10, col=15, hp=200)  # far

    # Inventory: slot 0 = scroll of charging, slot 1 = WAN_FIRE (recharged=7).
    scroll_item = make_item(
        category=int(ItemCategory.SCROLL),
        type_id=_SCROLL_BASE_ID + int(ScrollEffect.CHARGING),
        quantity=1,
        buc_status=2,
    )
    # The wand item: we need recharged=7 so the next charge overcharges.
    # ``make_item`` doesn't expose ``recharged`` directly; build the item then
    # post-edit the inventory.
    wand_item = make_item(
        category=int(ItemCategory.WAND),
        type_id=int(WandEffect.FIRE),
        quantity=1,
        buc_status=2,
    )
    state = state.replace(
        inventory=InventoryState.from_items([scroll_item, wand_item])
    )
    # Bump the wand slot's ``recharged`` field to 7 and give it 5 charges.
    items = state.inventory.items
    new_recharged = items.recharged.at[1].set(jnp.int8(7))
    new_charges   = items.charges.at[1].set(jnp.int8(5))
    state = state.replace(
        inventory=state.inventory.replace(
            items=items.replace(recharged=new_recharged, charges=new_charges),
        )
    )

    # Read the scroll (slot 0).
    out = read_scroll(state, _RNG, 0)

    # Adjacent monster must take damage; far monster untouched.
    assert int(out.monster_ai.hp[1]) < 200, (
        f"adjacent monster should be damaged by wand backfire AoE; "
        f"hp={int(out.monster_ai.hp[1])}"
    )
    assert int(out.monster_ai.hp[2]) == 200, (
        f"far monster must be untouched; hp={int(out.monster_ai.hp[2])}"
    )
    # Wand slot was cleared (category==0).
    assert int(out.inventory.items.category[1]) == 0, (
        "wand slot must be cleared after overcharge explosion"
    )
