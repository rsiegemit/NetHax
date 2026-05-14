"""Wave 6 Mission: monster_use_item payload tests.

Vendor reference:
    - vendor/nethack/src/muse.c::find_defensive / use_defensive
      → MUSE_POT_HEALING quaff branch (monsters.c::healup).
    - vendor/nethack/src/muse.c::find_misc / use_misc
      → MUSE_SCR_TELEPORTATION read branch.
    - vendor/nethack/src/muse.c::find_offensive / use_offensive
      → MUSE_WAN_FIRE zap branch (via zap.c::buzz).

Tests:
    1. Low-HP mage with a POT_HEALING in inv → HP increases, qty decrements.
    2. Full-HP mage with POT_HEALING → no quaff, qty unchanged.
    3. Low-HP mage with NO POT_HEALING in inv → HP unchanged.
    4. Mage adjacent to player with SCR_TELEPORT → monster position changes,
       scroll qty decrements.
    5. Mage in LoS, dist 2..8, with wand of fire (charges > 0) → player_hp
       drops, wand charges decrement.
    6. Animal-class monster (M1_ANIMAL) never uses items even with full inv.
    7. Wand charges decrement exactly by 1 per zap; zero charges → no zap.
    8. Potion quantity decrement exactly by 1 per quaff.
    9. Empty-inventory eligible monster: no state change.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.subsystems.monster_ai import (
    MAX_MONSTERS_PER_LEVEL,
    MAX_MONSTER_INV,
    monster_use_item,
    _CAT_POTION, _CAT_SCROLL, _CAT_WAND,
    _POT_HEALING, _SCR_TELEPORT, _WAN_FIRE,
)


_RNG = jax.random.PRNGKey(7)


def _entry_for(name: str) -> int:
    for i, m in enumerate(MONSTERS):
        if m.name == name:
            return i
    raise AssertionError(f"MONSTERS missing entry: {name!r}")


_TITAN_IDX  = _entry_for("titan")      # MS_SPELL — eligible for muse
_GIANT_ANT  = 0                        # M1_ANIMAL — NOT eligible for muse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _floor_state() -> EnvState:
    """EnvState with all-floor terrain on the player's branch+level."""
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)
    floor_map = jnp.full((static.map_h, static.map_w), TileType.FLOOR, dtype=jnp.int8)
    return state.replace(
        terrain=state.terrain.at[0, 0].set(floor_map),
        player_hp=jnp.int32(30),
        player_hp_max=jnp.int32(30),
    )


def _set_monster(state, slot, pos, hp=20, hp_max=20,
                 entry_idx=0, peaceful=False, asleep=False, alive=True):
    mai = state.monster_ai
    mai = mai.replace(
        pos=mai.pos.at[slot].set(jnp.array(pos, dtype=jnp.int16)),
        hp=mai.hp.at[slot].set(jnp.int32(hp)),
        hp_max=mai.hp_max.at[slot].set(jnp.int32(hp_max)),
        alive=mai.alive.at[slot].set(jnp.bool_(alive)),
        asleep=mai.asleep.at[slot].set(jnp.bool_(asleep)),
        peaceful=mai.peaceful.at[slot].set(jnp.bool_(peaceful)),
        entry_idx=mai.entry_idx.at[slot].set(jnp.int16(entry_idx)),
    )
    return state.replace(monster_ai=mai)


def _give_item(state, slot, inv_slot, category, type_id, qty=1, charges=0):
    """Place an item at inventory slot of monster slot."""
    mai = state.monster_ai
    mai = mai.replace(
        inv_category=mai.inv_category.at[slot, inv_slot].set(jnp.int8(category)),
        inv_type_id=mai.inv_type_id.at[slot, inv_slot].set(jnp.int16(type_id)),
        inv_quantity=mai.inv_quantity.at[slot, inv_slot].set(jnp.int16(qty)),
        inv_charges=mai.inv_charges.at[slot, inv_slot].set(jnp.int8(charges)),
    )
    return state.replace(monster_ai=mai)


# ===========================================================================
# 1. Heal branch
# ===========================================================================

def test_low_hp_mage_quaffs_healing_potion():
    """Low-HP eligible monster with POT_HEALING in inv → HP rises and the
    potion stack decrements by 1.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=2, hp_max=40,
                         entry_idx=_TITAN_IDX)
    state = _give_item(state, 0, 0, _CAT_POTION, _POT_HEALING, qty=2)
    rng = jax.random.PRNGKey(12)
    out = monster_use_item(state, rng, jnp.int32(0))
    new_hp = int(out.monster_ai.hp[0])
    new_qty = int(out.monster_ai.inv_quantity[0, 0])
    assert new_hp > 2, f"healing did not increase HP: {new_hp}"
    assert new_hp <= 40, f"healing exceeded hp_max: {new_hp}"
    assert new_qty == 1, f"potion qty did not decrement: {new_qty}"


def test_no_quaff_when_full_hp():
    """A mage at full HP (HP == hp_max) does NOT quaff — vendor only quaffs
    when hurt.  Our gate fires on hp < hp_max/4 too; full HP fails both.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=40, hp_max=40,
                         entry_idx=_TITAN_IDX)
    state = _give_item(state, 0, 0, _CAT_POTION, _POT_HEALING, qty=1)
    rng = jax.random.PRNGKey(13)
    out = monster_use_item(state, rng, jnp.int32(0))
    assert int(out.monster_ai.hp[0]) == 40, "Full-HP monster quaffed anyway"
    assert int(out.monster_ai.inv_quantity[0, 0]) == 1, \
        "Full-HP monster consumed a potion"


def test_no_quaff_when_no_healing_in_inv():
    """Low HP but no POT_HEALING — _try_heal must leave state unchanged."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=2, hp_max=40,
                         entry_idx=_TITAN_IDX)
    # Inventory empty: no healing potion.
    rng = jax.random.PRNGKey(14)
    out = monster_use_item(state, rng, jnp.int32(0))
    assert int(out.monster_ai.hp[0]) == 2, "HP changed despite no potion"


def test_potion_quantity_decrement_on_quaff():
    """Quaffing pulls exactly 1 unit from the stack; a 5-stack becomes 4."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=2, hp_max=40,
                         entry_idx=_TITAN_IDX)
    state = _give_item(state, 0, 0, _CAT_POTION, _POT_HEALING, qty=5)
    rng = jax.random.PRNGKey(15)
    out = monster_use_item(state, rng, jnp.int32(0))
    assert int(out.monster_ai.inv_quantity[0, 0]) == 4


# ===========================================================================
# 2. Scroll teleport branch
# ===========================================================================

def test_player_adjacent_mage_reads_scroll_teleport():
    """A mage adjacent to player (dist == 1) with SCR_TELEPORT teleports.

    The monster's position changes and the scroll's quantity drops by 1.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 11], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=40, hp_max=40,
                         entry_idx=_TITAN_IDX)
    state = _give_item(state, 0, 0, _CAT_SCROLL, _SCR_TELEPORT, qty=1)
    rng = jax.random.PRNGKey(16)
    out = monster_use_item(state, rng, jnp.int32(0))

    orig_pos = state.monster_ai.pos[0]
    new_pos  = out.monster_ai.pos[0]
    # Scroll consumed.
    assert int(out.monster_ai.inv_quantity[0, 0]) == 0, \
        "scroll qty did not decrement"
    # Position changed (random teleport tile is overwhelmingly different
    # from (10, 10) given 21*80 grid).
    moved = bool(jnp.any(orig_pos != new_pos))
    assert moved, "Scroll teleport did not move monster"


# ===========================================================================
# 3. Wand zap branch
# ===========================================================================

def test_mage_in_los_zaps_wand_of_fire_at_player():
    """Mage in LoS at dist 4 with wand of fire (5 charges) zaps player.

    player_hp drops and wand charges decrement by 1.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 14], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=40, hp_max=40,
                         entry_idx=_TITAN_IDX)
    state = _give_item(state, 0, 0, _CAT_WAND, _WAN_FIRE, qty=1, charges=5)
    orig_hp = int(state.player_hp)
    rng = jax.random.PRNGKey(17)
    out = monster_use_item(state, rng, jnp.int32(0))
    new_hp = int(out.player_hp)
    new_charges = int(out.monster_ai.inv_charges[0, 0])
    assert new_hp < orig_hp, f"player_hp did not drop: {new_hp} (was {orig_hp})"
    assert new_charges == 4, f"charges did not decrement: {new_charges}"


def test_wand_charges_decrement_on_zap():
    """Single zap decrements charges by exactly 1."""
    state = _floor_state().replace(player_pos=jnp.array([10, 14], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=40, hp_max=40,
                         entry_idx=_TITAN_IDX)
    state = _give_item(state, 0, 0, _CAT_WAND, _WAN_FIRE, qty=1, charges=3)
    rng = jax.random.PRNGKey(18)
    out = monster_use_item(state, rng, jnp.int32(0))
    assert int(out.monster_ai.inv_charges[0, 0]) == 2


def test_zero_charge_wand_does_not_zap():
    """Wand with 0 charges should not zap; player_hp unchanged."""
    state = _floor_state().replace(player_pos=jnp.array([10, 14], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=40, hp_max=40,
                         entry_idx=_TITAN_IDX)
    state = _give_item(state, 0, 0, _CAT_WAND, _WAN_FIRE, qty=1, charges=0)
    orig_hp = int(state.player_hp)
    rng = jax.random.PRNGKey(19)
    out = monster_use_item(state, rng, jnp.int32(0))
    assert int(out.player_hp) == orig_hp
    # Charges stay at 0 — no underflow.
    assert int(out.monster_ai.inv_charges[0, 0]) == 0


# ===========================================================================
# 4. Animal-class gate
# ===========================================================================

def test_animal_never_uses_items_despite_inventory():
    """An animal-class monster (M1_ANIMAL) is excluded from muse per
    vendor muse.c:1428.  Even if it holds a healing potion at 1 HP, it
    does not quaff.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=1, hp_max=40,
                         entry_idx=_GIANT_ANT)
    state = _give_item(state, 0, 0, _CAT_POTION, _POT_HEALING, qty=3)
    rng = jax.random.PRNGKey(20)
    out = monster_use_item(state, rng, jnp.int32(0))
    assert int(out.monster_ai.hp[0]) == 1, \
        "Animal monster used potion (muse gate failed)"
    assert int(out.monster_ai.inv_quantity[0, 0]) == 3, \
        "Animal monster consumed potion"


def test_peaceful_monster_does_not_use_items():
    """Peaceful monsters skip muse entirely per vendor muse.c:1428 gate."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=1, hp_max=40,
                         entry_idx=_TITAN_IDX, peaceful=True)
    state = _give_item(state, 0, 0, _CAT_POTION, _POT_HEALING, qty=2)
    rng = jax.random.PRNGKey(21)
    out = monster_use_item(state, rng, jnp.int32(0))
    assert int(out.monster_ai.hp[0]) == 1
    assert int(out.monster_ai.inv_quantity[0, 0]) == 2


def test_empty_inventory_eligible_monster_is_noop():
    """A mage at low HP with EMPTY inventory: muse exits cleanly with
    no state change.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=2, hp_max=40,
                         entry_idx=_TITAN_IDX)
    rng = jax.random.PRNGKey(22)
    out = monster_use_item(state, rng, jnp.int32(0))
    assert int(out.monster_ai.hp[0]) == 2
    assert int(out.player_hp) == 30
