"""Apply-tools parity tests.

Verifies vendor/nethack/src/apply.c behavior for all tool handlers wired into
Nethax.nethax.subsystems.apply_tools.

Cite: vendor/nethack/src/apply.c::doapply (line 4214).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.inventory import ItemCategory, make_item, MAX_INVENTORY_SLOTS
from Nethax.nethax.subsystems.monster_ai import MAX_MONSTERS_PER_LEVEL
from Nethax.nethax.subsystems.apply_tools import (
    dispatch_apply,
    _MAGIC_WHISTLE_TYPE_ID,
    _TIN_WHISTLE_TYPE_ID,
    _OIL_LAMP_TYPE_ID,
    _MAGIC_LAMP_TYPE_ID,
    _STETHOSCOPE_TYPE_ID,
    _TINNING_KIT_TYPE_ID,
    _CAN_OF_GREASE_TYPE_ID,
    _HORN_OF_PLENTY_TYPE_ID,
    _TOWEL_TYPE_ID,
    _EXPENSIVE_CAMERA_TYPE_ID,
    _LEASH_TYPE_ID,
    _CRYSTAL_BALL_TYPE_ID,
    _CORPSE_TYPE_ID,
    _TRIPE_RATION_TYPE_ID,
)
from Nethax.nethax.subsystems.status_effects import TimedStatus

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _floor_state(player_pos=(10, 10)) -> EnvState:
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)
    floor_map = jnp.full((static.map_h, static.map_w), int(TileType.FLOOR), dtype=jnp.int8)
    return state.replace(
        terrain=state.terrain.at[0, 0].set(floor_map),
        player_pos=jnp.array(player_pos, dtype=jnp.int16),
        player_hp=jnp.int32(20),
        player_hp_max=jnp.int32(20),
    )


def _wield_item(state: EnvState, type_id: int, category: int = int(ItemCategory.TOOL),
                slot: int = 0, **kwargs) -> EnvState:
    """Place an item in the given slot and wield it."""
    inv = state.inventory
    new_cat = inv.items.category.at[slot].set(jnp.int8(category))
    new_tid = inv.items.type_id.at[slot].set(jnp.int16(type_id))
    new_items = inv.items.replace(category=new_cat, type_id=new_tid)
    for field, val in kwargs.items():
        arr = getattr(new_items, field)
        new_items = new_items.replace(**{field: arr.at[slot].set(val)})
    return state.replace(inventory=inv.replace(items=new_items, wielded=jnp.int8(slot)))


def _place_monster(state: EnvState, idx: int, pos, hp: int = 20,
                   tame: bool = False, asleep: bool = False) -> EnvState:
    mai = state.monster_ai
    mai = mai.replace(
        pos=mai.pos.at[idx].set(jnp.array(pos, dtype=jnp.int16)),
        hp=mai.hp.at[idx].set(jnp.int32(hp)),
        hp_max=mai.hp_max.at[idx].set(jnp.int32(hp)),
        alive=mai.alive.at[idx].set(jnp.bool_(True)),
        tame=mai.tame.at[idx].set(jnp.bool_(tame)),
        asleep=mai.asleep.at[idx].set(jnp.bool_(asleep)),
        entry_idx=mai.entry_idx.at[idx].set(jnp.int16(0)),
    )
    return state.replace(monster_ai=mai)


# ---------------------------------------------------------------------------
# test_magic_whistle_calls_pets
# Cite: apply.c::magic_whistled (line 518) — tame monsters move adjacent.
# ---------------------------------------------------------------------------
def test_magic_whistle_calls_pets():
    state = _floor_state(player_pos=(10, 10))
    # Place a tame pet far away at (1, 1).
    state = _place_monster(state, idx=0, pos=(1, 1), tame=True)
    state = _wield_item(state, type_id=_MAGIC_WHISTLE_TYPE_ID)

    new_state = dispatch_apply(state, _RNG)

    pet_pos = new_state.monster_ai.pos[0]
    pr, pc = int(new_state.player_pos[0]), int(new_state.player_pos[1])
    dy = abs(int(pet_pos[0]) - pr)
    dx = abs(int(pet_pos[1]) - pc)
    chebyshev = max(dy, dx)
    assert chebyshev <= 1, (
        f"After magic whistle, pet should be adjacent (Chebyshev<=1); got {chebyshev}. "
        f"pet_pos={pet_pos}, player_pos=({pr},{pc})"
    )


# ---------------------------------------------------------------------------
# test_oil_lamp_lights
# Cite: apply.c::use_lamp (called from doapply line 4347) — toggles lamplit.
# ---------------------------------------------------------------------------
def test_oil_lamp_lights():
    state = _floor_state()
    state = _wield_item(state, type_id=_OIL_LAMP_TYPE_ID)
    assert not bool(state.inventory.items.lamplit[0]), "precondition: lamp starts unlit"

    new_state = dispatch_apply(state, _RNG)

    assert bool(new_state.inventory.items.lamplit[0]), "Oil lamp should be lit after apply"


def test_oil_lamp_toggles_off():
    state = _floor_state()
    state = _wield_item(state, type_id=_OIL_LAMP_TYPE_ID, lamplit=jnp.bool_(True))
    assert bool(state.inventory.items.lamplit[0]), "precondition: lamp starts lit"

    new_state = dispatch_apply(state, _RNG)

    assert not bool(new_state.inventory.items.lamplit[0]), "Oil lamp should be extinguished"


# ---------------------------------------------------------------------------
# test_stethoscope_probes_monster
# Cite: apply.c::use_stethoscope (line 318) — examine adjacent monster.
# We verify that the monster's mtame field is set to the probed sentinel (20).
# ---------------------------------------------------------------------------
def test_stethoscope_probes_monster():
    state = _floor_state(player_pos=(10, 10))
    # Adjacent monster at (10, 11) — Chebyshev distance 1.
    state = _place_monster(state, idx=0, pos=(10, 11), hp=15)
    state = _wield_item(state, type_id=_STETHOSCOPE_TYPE_ID)

    new_state = dispatch_apply(state, _RNG)

    probed_mtame = int(new_state.monster_ai.mtame[0])
    assert probed_mtame == 20, (
        f"Stethoscoped monster should have mtame sentinel 20; got {probed_mtame}"
    )


def test_stethoscope_no_effect_on_non_adjacent():
    state = _floor_state(player_pos=(10, 10))
    # Monster far away at (1, 1).
    state = _place_monster(state, idx=0, pos=(1, 1), hp=15)
    state = _wield_item(state, type_id=_STETHOSCOPE_TYPE_ID)

    new_state = dispatch_apply(state, _RNG)

    # No adjacent monster → mtame unchanged (0, not sentinel 20).
    probed_mtame = int(new_state.monster_ai.mtame[0])
    assert probed_mtame != 20, "Non-adjacent monster should not be probed"


# ---------------------------------------------------------------------------
# test_tinning_kit_creates_tin
# Cite: apply.c::use_tinning_kit (line 2177) — convert corpse to tin.
# We mark the resulting tin by corpse_creation_turn==-2.
# ---------------------------------------------------------------------------
def test_tinning_kit_creates_tin():
    state = _floor_state()
    # Place a corpse in slot 1.
    inv = state.inventory
    new_cat = inv.items.category.at[1].set(jnp.int8(int(ItemCategory.FOOD)))
    new_tid = inv.items.type_id.at[1].set(jnp.int16(_CORPSE_TYPE_ID))
    new_cct = inv.items.corpse_creation_turn.at[1].set(jnp.int32(100))
    new_cei = inv.items.corpse_entry_idx.at[1].set(jnp.int16(5))
    new_items = inv.items.replace(category=new_cat, type_id=new_tid,
                                   corpse_creation_turn=new_cct,
                                   corpse_entry_idx=new_cei)
    state = state.replace(inventory=inv.replace(items=new_items))
    state = _wield_item(state, type_id=_TINNING_KIT_TYPE_ID, slot=0)

    new_state = dispatch_apply(state, _RNG)

    # The corpse at slot 1 should now have corpse_creation_turn == -2 (tinned sentinel).
    cct = int(new_state.inventory.items.corpse_creation_turn[1])
    assert cct == -2, f"Tinned corpse should have cct==-2; got {cct}"
    # corpse_entry_idx preserved.
    assert int(new_state.inventory.items.corpse_entry_idx[1]) == 5


# ---------------------------------------------------------------------------
# test_can_of_grease_greases_item
# Cite: apply.c::use_grease (line 2604) — grease a carried item.
# ---------------------------------------------------------------------------
def test_can_of_grease_greases_item():
    state = _floor_state()
    # Place a weapon in slot 1.
    inv = state.inventory
    new_cat = inv.items.category.at[1].set(jnp.int8(int(ItemCategory.WEAPON)))
    new_tid = inv.items.type_id.at[1].set(jnp.int16(1))
    new_items = inv.items.replace(category=new_cat, type_id=new_tid)
    state = state.replace(inventory=inv.replace(items=new_items))
    state = _wield_item(state, type_id=_CAN_OF_GREASE_TYPE_ID, slot=0)

    assert not bool(state.inventory.items.greased[1]), "precondition: weapon not greased"

    new_state = dispatch_apply(state, _RNG)

    assert bool(new_state.inventory.items.greased[1]), "Weapon should be greased"


# ---------------------------------------------------------------------------
# test_horn_of_plenty_food
# Cite: apply.c::hornoplenty (line 4385) — add food item to inventory.
# ---------------------------------------------------------------------------
def test_horn_of_plenty_food():
    state = _floor_state()
    state = _wield_item(state, type_id=_HORN_OF_PLENTY_TYPE_ID, slot=0)
    # All other slots empty.
    inv_before = state.inventory

    new_state = dispatch_apply(state, _RNG)

    # Should have a FOOD item with type_id == TRIPE_RATION in slot 1.
    inv_after = new_state.inventory
    food_slots = (inv_after.items.category == jnp.int8(int(ItemCategory.FOOD)))
    assert bool(jnp.any(food_slots)), "Horn of plenty should add food to inventory"
    food_idx = int(jnp.argmax(food_slots))
    assert int(inv_after.items.type_id[food_idx]) == _TRIPE_RATION_TYPE_ID, (
        f"Expected tripe ration type_id {_TRIPE_RATION_TYPE_ID}; "
        f"got {int(inv_after.items.type_id[food_idx])}"
    )


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------

def test_tin_whistle_wakes_adjacent_sleeper():
    """Cite: apply.c::use_whistle (line 476) — shrill sound may wake sleepers."""
    # Use a fixed RNG where randint < 1 for slot 0 → should wake (roll==0, 1/8 chance).
    state = _floor_state(player_pos=(10, 10))
    state = _place_monster(state, idx=0, pos=(10, 11), asleep=True)
    state = _wield_item(state, type_id=_TIN_WHISTLE_TYPE_ID)

    # Run several times; at least once it should wake (roll 0 out of 8).
    woke = False
    for seed in range(20):
        rng = jax.random.PRNGKey(seed)
        ns = dispatch_apply(state, rng)
        if not bool(ns.monster_ai.asleep[0]):
            woke = True
            break
    assert woke, "Tin whistle should wake adjacent sleeping monsters occasionally"


def test_magic_lamp_toggles_lamplit():
    """Cite: apply.c::use_lamp / doapply case MAGIC_LAMP (line 4344)."""
    state = _floor_state()
    state = _wield_item(state, type_id=_MAGIC_LAMP_TYPE_ID)
    assert not bool(state.inventory.items.lamplit[0])

    new_state = dispatch_apply(state, _RNG)
    assert bool(new_state.inventory.items.lamplit[0]), "Magic lamp should light up on apply"


def test_towel_clears_blind():
    """Cite: apply.c::use_towel (line 112) — removes face-covering item, unblind."""
    state = _floor_state()
    # Set player blind for 100 turns.
    new_ts = state.status.timed_statuses.at[int(TimedStatus.BLIND)].set(jnp.int32(100))
    state = state.replace(status=state.status.replace(timed_statuses=new_ts))
    state = _wield_item(state, type_id=_TOWEL_TYPE_ID)

    new_state = dispatch_apply(state, _RNG)

    blind_timer = int(new_state.status.timed_statuses[int(TimedStatus.BLIND)])
    assert blind_timer == 0, f"Towel should clear blind timer; got {blind_timer}"


def test_expensive_camera_blinds_player_when_no_target():
    """Cite: apply.c::use_camera (line 79) — flash blinds player if no target."""
    state = _floor_state(player_pos=(10, 10))
    # No monsters on the map.
    state = _wield_item(state, type_id=_EXPENSIVE_CAMERA_TYPE_ID)

    new_state = dispatch_apply(state, _RNG)

    blind_timer = int(new_state.status.timed_statuses[int(TimedStatus.BLIND)])
    assert blind_timer == 50, f"Camera should blind player (50 turns) when no target; got {blind_timer}"


def test_noop_for_unknown_tool():
    """Unknown type_id dispatches to noop and returns state unchanged."""
    state = _floor_state()
    state = _wield_item(state, type_id=9999)  # unknown tool

    new_state = dispatch_apply(state, _RNG)

    # State should be structurally identical (no crash, no mutation).
    assert int(new_state.player_hp) == int(state.player_hp)
