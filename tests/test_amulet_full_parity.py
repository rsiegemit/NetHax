"""Full parity tests for amulet effects.

Canonical sources:
  vendor/nethack/src/do_wear.c  — Amulet_on(), Amulet_off()
  vendor/nethack/src/end.c      — done() lines 1084-1105 (life-saving)
  vendor/nethack/src/zap.c      — buzz() reflection path

Tests:
  test_amulet_of_esp_grants_telepathy
  test_amulet_life_saving_saves_on_death
  test_versus_poison_grants_resist_poison
  test_strangulation_kills
  test_unchanging_blocks_revert
  test_reflection_bounces_ray
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.status_effects import Intrinsic, TimedStatus
from Nethax.nethax.subsystems.items_jewelry import (
    AmuletEffect,
    wear_amulet,
    check_life_saving,
)
from Nethax.nethax.subsystems.polymorph import (
    revert_polymorph,
    polymorph_player,
    UNCHANGING_MASK,
)
from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.subsystems.items_wands import WandState, cast_ray

_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_with_amulet(effect: AmuletEffect, slot: int = 0) -> EnvState:
    """Default state with one amulet of the given type in slot ``slot``."""
    state = EnvState.default(_RNG)
    new_items = state.inventory.items.replace(
        category=state.inventory.items.category.at[slot].set(jnp.int8(4)),  # AMULET_CLASS
        type_id=state.inventory.items.type_id.at[slot].set(jnp.int16(int(effect))),
        quantity=state.inventory.items.quantity.at[slot].set(jnp.int16(1)),
    )
    new_inv = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


def _has(state: EnvState, intrinsic: Intrinsic) -> bool:
    return bool(state.status.intrinsics[int(intrinsic)])


# ---------------------------------------------------------------------------
# test_amulet_of_esp_grants_telepathy
# Cite: do_wear.c Amulet_on case AMULET_OF_ESP → set_itimeout TELEPAT
# ---------------------------------------------------------------------------

def test_amulet_of_esp_grants_telepathy():
    """Wearing AMULET_OF_ESP sets Intrinsic.TELEPATHY."""
    state = _state_with_amulet(AmuletEffect.ESP, slot=0)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert _has(state, Intrinsic.TELEPATHY), (
        "TELEPATHY must be set after wearing amulet of ESP "
        "(do_wear.c Amulet_on case AMULET_OF_ESP)"
    )


# ---------------------------------------------------------------------------
# test_amulet_life_saving_saves_on_death
# Cite: end.c::done() lines 1084-1105 — Lifesaved check, useup(uamul),
#       savelife() restores hp to hp_max, survive=TRUE.
# ---------------------------------------------------------------------------

def test_amulet_life_saving_saves_on_death():
    """check_life_saving: hp→hp_max, done→False, amulet quantity→0 on death."""
    state = _state_with_amulet(AmuletEffect.LIFE_SAVING, slot=0)
    state = wear_amulet(state, _RNG, slot_idx=0)

    # Confirm intrinsic is set after wear.
    assert _has(state, Intrinsic.LIFESAVED), "LIFESAVED intrinsic must be set after wear"

    # Simulate death: hp=0, done=True.
    state = state.replace(
        player_hp=jnp.int32(0),
        player_hp_max=jnp.int32(20),
        done=jnp.bool_(True),
    )

    state_after, lifesaved = check_life_saving(state)

    assert bool(lifesaved), "lifesaved_bool must be True"
    assert not bool(state_after.done), "done must be False after life-saving"
    assert int(state_after.player_hp) == 20, "HP must be restored to hp_max"
    assert int(state_after.inventory.items.quantity[0]) == 0, (
        "Amulet quantity must be 0 (consumed/crumbled — end.c useup(uamul))"
    )
    assert not _has(state_after, Intrinsic.LIFESAVED), (
        "LIFESAVED intrinsic must be revoked so a second death is fatal"
    )


def test_amulet_life_saving_no_save_when_not_worn():
    """check_life_saving: no effect when LIFESAVED intrinsic is absent."""
    state = EnvState.default(_RNG)
    state = state.replace(
        player_hp=jnp.int32(0),
        player_hp_max=jnp.int32(10),
        done=jnp.bool_(True),
    )
    state_after, lifesaved = check_life_saving(state)
    assert not bool(lifesaved), "lifesaved_bool must be False when no amulet worn"
    assert bool(state_after.done), "done must remain True"
    assert int(state_after.player_hp) == 0, "HP must stay 0"


# ---------------------------------------------------------------------------
# test_versus_poison_grants_resist_poison
# Cite: do_wear.c Amulet_on — no explicit case for VERSUS_POISON;
#       grants POISON_RES via extrinsic (prop.h POISON_RES = 6).
# ---------------------------------------------------------------------------

def test_versus_poison_grants_resist_poison():
    """Wearing AMULET_VERSUS_POISON sets Intrinsic.RESIST_POISON."""
    state = _state_with_amulet(AmuletEffect.VERSUS_POISON, slot=0)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert _has(state, Intrinsic.RESIST_POISON), (
        "RESIST_POISON must be set after wearing amulet versus poison"
    )


# ---------------------------------------------------------------------------
# test_strangulation_kills
# Cite: do_wear.c Amulet_on case AMULET_OF_STRANGULATION — starts STRANGLED timer.
#       status_effects.py apply_strangulation: timer==1 → hp=0, done=True.
# ---------------------------------------------------------------------------

def test_strangulation_kills():
    """Wearing a cursed AMULET_OF_STRANGULATION starts STRANGLED timer.

    With timer at 1 (about to expire), apply_strangulation sets hp=0, done=True.
    """
    from Nethax.nethax.subsystems.status_effects import apply_strangulation

    state = _state_with_amulet(AmuletEffect.STRANGULATION, slot=0)
    # Mark as cursed (buc_status=1 is cursed in this codebase).
    new_items = state.inventory.items.replace(
        buc_status=state.inventory.items.buc_status.at[0].set(jnp.int8(1))
    )
    state = state.replace(inventory=state.inventory.replace(items=new_items))
    state = wear_amulet(state, _RNG, slot_idx=0)

    # STRANGLED timer must be set.
    strangled_timer = int(state.status.timed_statuses[int(TimedStatus.STRANGLED)])
    assert strangled_timer > 0, "STRANGLED timer must be > 0 after wearing amulet of strangulation"

    # Drive timer to 1 to trigger lethal expiry on next apply_strangulation call.
    new_ts = state.status.timed_statuses.at[int(TimedStatus.STRANGLED)].set(jnp.int32(1))
    state = state.replace(status=state.status.replace(timed_statuses=new_ts))

    _status, new_hp, new_done = apply_strangulation(
        state.status, state.player_hp, state.done
    )
    assert int(new_hp) == 0, "HP must be 0 when STRANGLED timer expires"
    assert bool(new_done), "done must be True when STRANGLED timer expires"


# ---------------------------------------------------------------------------
# test_unchanging_blocks_revert
# Cite: polyself.c:1367 — rehumanize checks UNCHANGING → kills player.
#       do_wear.c Amulet_on AMULET_OF_UNCHANGING → sets UNCHANGING intrinsic.
# ---------------------------------------------------------------------------

def test_unchanging_blocks_revert():
    """Wearing AMULET_OF_UNCHANGING then reverting polymorph kills the player.

    Cite: vendor/nethack/src/polyself.c:1367 — rehumanize while Unchanging
    is a fatal condition: 'rehumanize: Unchanging → You die.'
    """
    # Find a valid polymorph form (not unique, not nopoly).
    from Nethax.nethax.constants.monsters import G_UNIQ, M2_NOPOLY
    form_idx = next(
        i for i, m in enumerate(MONSTERS)
        if not (m.generation_mask & G_UNIQ)
        and not (m.flags2 & M2_NOPOLY)
        and i > 0
    )

    state = _state_with_amulet(AmuletEffect.UNCHANGING, slot=0)
    state = wear_amulet(state, _RNG, slot_idx=0)

    assert bool(state.status.intrinsics[UNCHANGING_MASK]), (
        "UNCHANGING intrinsic must be set after wearing amulet of unchanging"
    )

    # Polymorph the player into some form so there is something to revert.
    state = polymorph_player(state, _RNG, form_idx)
    assert bool(state.polymorph.is_polymorphed), "Player must be polymorphed"

    # Revert — should kill due to Unchanging.
    state_after = revert_polymorph(state, _RNG)
    assert bool(state_after.done), (
        "done must be True: polyself.c:1367 Unchanging kills on rehumanize"
    )
    assert int(state_after.player_hp) == 0, "HP must be 0 on Unchanging-kill"


# ---------------------------------------------------------------------------
# test_reflection_bounces_ray
# Cite: vendor/nethack/src/zap.c::buzz — beam hits player tile with Reflecting,
#       direction reverses and the beam continues back toward the source.
# ---------------------------------------------------------------------------

def test_reflection_bounces_ray():
    """A ray aimed at the player reflects back and hits the source monster.

    Setup:
      - Monster at (5, 2), alive with 10 HP.
      - Player at (5, 5).
      - Ray fired East (direction=2, dx=+1) from monster position toward player.
      - player_reflecting=True.

    Expected: after cast_ray, the monster at (5,2) is dead (ray bounced back
    and dealt damage to it via on_hit_fn), player HP untouched.
    """
    from Nethax.nethax.subsystems.items_wands import WandState, cast_ray, _deal_damage
    from Nethax.nethax.constants.tiles import TileType

    MAP_H, MAP_W = 10, 10
    N = 4  # small monster table for test

    terrain = jnp.full((MAP_H, MAP_W), int(TileType.FLOOR), dtype=jnp.int8)

    # Monster at slot 1 (slot 0 is sentinel), position (5,2), HP=10.
    MON_SLOT = 1
    MON_ROW, MON_COL = 5, 2
    mon_pos = jnp.zeros((N, 2), dtype=jnp.int16)
    mon_pos = mon_pos.at[MON_SLOT].set(jnp.array([MON_ROW, MON_COL], dtype=jnp.int16))
    mon_hp = jnp.zeros(N, dtype=jnp.int32)
    mon_hp = mon_hp.at[MON_SLOT].set(jnp.int32(10))
    mon_alive = jnp.zeros(N, dtype=jnp.bool_)
    mon_alive = mon_alive.at[MON_SLOT].set(True)

    from Nethax.nethax.subsystems.inventory import InventoryState
    from Nethax.nethax.subsystems.traps import TrapState
    state = WandState(
        mon_pos=mon_pos,
        mon_hp=mon_hp,
        mon_hp_max=mon_hp,
        mon_type=jnp.zeros(N, dtype=jnp.int16),
        mon_alive=mon_alive,
        mon_asleep=jnp.zeros(N, dtype=jnp.bool_),
        mon_undead=jnp.zeros(N, dtype=jnp.bool_),
        mon_invisible=jnp.zeros(N, dtype=jnp.bool_),
        mon_resists=jnp.zeros(N, dtype=jnp.int32),
        mon_speed_mod=jnp.zeros(N, dtype=jnp.int8),
        mon_cancelled=jnp.zeros(N, dtype=jnp.bool_),
        mon_paralyzed_timer=jnp.zeros(N, dtype=jnp.int16),
        mon_sleep_timer=jnp.zeros(N, dtype=jnp.int16),
        terrain=terrain,
        explored=jnp.zeros((MAP_H, MAP_W), dtype=jnp.bool_),
        blockers=jnp.zeros((MAP_H, MAP_W), dtype=jnp.bool_),
        inventory=InventoryState.empty(),
        player_pos=jnp.array([5, 5], dtype=jnp.int16),
        dungeon_level=jnp.int8(1),
        probed_hp=jnp.int32(0),
        probed_idx=jnp.int32(0),
        player_reflecting=jnp.bool_(True),
        branch=jnp.int8(0),
        traps=TrapState.default(num_levels=1, map_h=MAP_H, map_w=MAP_W),
        wall_info=jnp.zeros((1, MAP_H, MAP_W), dtype=jnp.bool_),
    )

    # on_hit_fn: deal 15 damage (enough to kill the 10-HP monster).
    def on_hit(s, r, mon_idx):
        return _deal_damage(s, mon_idx, jnp.int32(15)), r

    # Fire ray East (direction=2) from monster position (5,2).
    # It travels East, hits player at (5,5), reflects West, returns to (5,2).
    start = jnp.array([MON_ROW, MON_COL], dtype=jnp.int16)
    direction = jnp.int32(2)  # East

    final_state, _ = cast_ray(
        state, _RNG, start, direction,
        ray_range=16,
        on_hit_fn=on_hit,
        stop_on_hit=False,
    )

    assert not bool(final_state.mon_alive[MON_SLOT]), (
        "Monster at (5,2) must be dead: reflected ray bounced back and hit it "
        "(zap.c::buzz reflection path)"
    )
