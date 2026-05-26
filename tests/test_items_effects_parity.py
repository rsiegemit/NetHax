"""Wave 6 Closing-Audit #92 — item-effect parity vs vendor NetHack.

Targets ~30+ tests covering 5 categories (potions, scrolls, wands, rings,
amulets).  Each test verifies the JAX implementation matches the spirit of
the vendor formula in vendor/nethack/src/{potion.c,read.c,zap.c,do_wear.c}.

Focus areas:
  - Formula direction (HP up/down, timer set, intrinsic granted).
  - Side effects (blindness cure on healing potions, etc.).
  - BUC modulation (blessed strongest, cursed weakest/inverted).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.inventory import (
    InventoryState,
    Item,
    ItemCategory,
    make_item,
)
from Nethax.nethax.subsystems.items_potions import (
    PotionEffect,
    _POTION_BASE_ID,
    quaff_potion,
)
from Nethax.nethax.subsystems.items_scrolls import (
    ScrollEffect,
    _SCROLL_BASE_ID,
    read_scroll,
)
from Nethax.nethax.subsystems.items_wands import (
    WandEffect,
    WandState,
    ITEM_CATEGORY_WAND,
    zap_wand,
)
from Nethax.nethax.subsystems.items_jewelry import (
    RingEffect,
    AmuletEffect,
    put_on_ring,
    wear_amulet,
)
from Nethax.nethax.subsystems.status_effects import (
    TimedStatus,
    Intrinsic,
)
from Nethax.nethax.constants.tiles import TileType


_RNG = jax.random.PRNGKey(42)

_BUC_CURSED = 1
_BUC_UNCURSED = 2
_BUC_BLESSED = 3

MAP_H, MAP_W = 21, 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_with_potion(effect, buc=_BUC_UNCURSED, hp=5, hp_max=50):
    type_id = _POTION_BASE_ID + int(effect)
    item = make_item(
        category=int(ItemCategory.POTION),
        type_id=type_id,
        quantity=1,
        buc_status=buc,
    )
    state = EnvState.default(_RNG)
    state = state.replace(
        player_hp=jnp.int32(hp),
        player_hp_max=jnp.int32(hp_max),
        inventory=InventoryState.from_items([item]),
    )
    return state


def _state_with_scroll(effect, buc=_BUC_UNCURSED):
    type_id = _SCROLL_BASE_ID + int(effect)
    item = make_item(
        category=int(ItemCategory.SCROLL),
        type_id=type_id,
        quantity=1,
        buc_status=buc,
    )
    state = EnvState.default(_RNG)
    return state.replace(inventory=InventoryState.from_items([item]))


def _wand_state_with_wand(effect, charges=5, player_row=10, player_col=10):
    """WandState helper for direct zap_wand calls."""
    state = WandState.empty(map_h=MAP_H, map_w=MAP_W)
    terrain = jnp.full((MAP_H, MAP_W), int(TileType.FLOOR), dtype=jnp.int8)
    wand_item = Item(
        category=jnp.int8(ITEM_CATEGORY_WAND),
        type_id=jnp.int16(int(effect)),
        buc_status=jnp.int8(_BUC_UNCURSED),
        enchantment=jnp.int8(0),
        charges=jnp.int8(charges),
        identified=jnp.bool_(True),
        quantity=jnp.int16(1),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
    )
    inv = InventoryState.from_items([wand_item])
    return state.replace(
        terrain=terrain,
        player_pos=jnp.array([player_row, player_col], dtype=jnp.int16),
        inventory=inv,
    )


def _place_monster(state, slot, row, col, hp=20, undead=False):
    new_pos = state.mon_pos.at[slot].set(jnp.array([row, col], dtype=jnp.int16))
    new_hp = state.mon_hp.at[slot].set(jnp.int32(hp))
    new_alive = state.mon_alive.at[slot].set(jnp.bool_(True))
    new_undead = state.mon_undead.at[slot].set(jnp.bool_(undead))
    return state.replace(
        mon_pos=new_pos,
        mon_hp=new_hp,
        mon_alive=new_alive,
        mon_undead=new_undead,
    )


def _state_with_ring(effect, enchantment=0):
    state = EnvState.default(_RNG)
    new_item = state.inventory.items.replace(
        category=jnp.int8(3),
        type_id=jnp.int16(int(effect)),
        enchantment=jnp.int8(enchantment),
    )
    return state.replace(inventory=state.inventory.replace(items=new_item))


def _state_with_amulet(effect):
    state = EnvState.default(_RNG)
    new_item = state.inventory.items.replace(
        category=jnp.int8(4),
        type_id=jnp.int16(int(effect)),
        enchantment=jnp.int8(0),
    )
    return state.replace(inventory=state.inventory.replace(items=new_item))


# ===========================================================================
# A. POTION TESTS — vendor/nethack/src/potion.c::peffects
# ===========================================================================

def test_potion_healing_cures_blindness_when_not_cursed():
    """POT_HEALING vendor: healup(..., !cursed, !cursed) — cureblind=!cursed."""
    state = _state_with_potion(PotionEffect.HEALING, buc=_BUC_UNCURSED, hp=5, hp_max=50)
    # Pre-set blindness.
    new_ts = state.status.timed_statuses.at[int(TimedStatus.BLIND)].set(jnp.int32(100))
    state = state.replace(status=state.status.replace(timed_statuses=new_ts))
    result = quaff_potion(state, _RNG, 0)
    assert int(result.status.timed_statuses[int(TimedStatus.BLIND)]) == 0, (
        "uncursed POT_HEALING should clear blindness (vendor cureblind=!cursed)"
    )


def test_potion_healing_cursed_does_not_cure_blindness():
    """POT_HEALING cursed should NOT cure blindness (vendor cureblind=!cursed=FALSE)."""
    state = _state_with_potion(PotionEffect.HEALING, buc=_BUC_CURSED, hp=5, hp_max=50)
    new_ts = state.status.timed_statuses.at[int(TimedStatus.BLIND)].set(jnp.int32(100))
    state = state.replace(status=state.status.replace(timed_statuses=new_ts))
    result = quaff_potion(state, _RNG, 0)
    assert int(result.status.timed_statuses[int(TimedStatus.BLIND)]) > 0, (
        "cursed POT_HEALING must not cure blindness"
    )


def test_potion_extra_healing_cures_blindness():
    """POT_EXTRA_HEALING vendor: healup(..., !cursed, TRUE) — always cures blindness."""
    state = _state_with_potion(PotionEffect.EXTRA_HEALING, buc=_BUC_CURSED, hp=5, hp_max=50)
    new_ts = state.status.timed_statuses.at[int(TimedStatus.BLIND)].set(jnp.int32(100))
    state = state.replace(status=state.status.replace(timed_statuses=new_ts))
    result = quaff_potion(state, _RNG, 0)
    assert int(result.status.timed_statuses[int(TimedStatus.BLIND)]) == 0, (
        "POT_EXTRA_HEALING must cure blindness even when cursed (vendor cureblind=TRUE)"
    )


def test_potion_full_healing_cures_blindness_and_hallucination():
    """POT_FULL_HEALING vendor: healup(400, ..., TRUE) + make_hallucinated(0)."""
    state = _state_with_potion(PotionEffect.FULL_HEALING, buc=_BUC_UNCURSED, hp=1, hp_max=50)
    ts = state.status.timed_statuses
    ts = ts.at[int(TimedStatus.BLIND)].set(jnp.int32(100))
    ts = ts.at[int(TimedStatus.HALLUCINATION)].set(jnp.int32(100))
    state = state.replace(status=state.status.replace(timed_statuses=ts))
    result = quaff_potion(state, _RNG, 0)
    assert int(result.player_hp) == int(result.player_hp_max), "Full healing → HP = HP_MAX"
    assert int(result.status.timed_statuses[int(TimedStatus.BLIND)]) == 0
    assert int(result.status.timed_statuses[int(TimedStatus.HALLUCINATION)]) == 0


def test_potion_full_healing_uncursed_cures_sickness():
    """POT_FULL_HEALING vendor: curesick=!cursed → uncursed cures sick + vomiting."""
    state = _state_with_potion(PotionEffect.FULL_HEALING, buc=_BUC_UNCURSED, hp=5, hp_max=50)
    ts = state.status.timed_statuses
    ts = ts.at[int(TimedStatus.SICK)].set(jnp.int32(50))
    ts = ts.at[int(TimedStatus.VOMITING)].set(jnp.int32(20))
    state = state.replace(status=state.status.replace(timed_statuses=ts))
    result = quaff_potion(state, _RNG, 0)
    assert int(result.status.timed_statuses[int(TimedStatus.SICK)]) == 0
    assert int(result.status.timed_statuses[int(TimedStatus.VOMITING)]) == 0


def test_potion_speed_grants_fast_intrinsic():
    """POT_SPEED — vendor speed_up(rn1(10, 100+60*bcsign))."""
    state = _state_with_potion(PotionEffect.SPEED, buc=_BUC_UNCURSED, hp=20, hp_max=50)
    result = quaff_potion(state, _RNG, 0)
    timer = int(result.status.timed_intrinsics[Intrinsic.FAST])
    perm = bool(result.status.intrinsics[Intrinsic.FAST])
    assert timer > 0 or perm, "POT_SPEED should grant FAST intrinsic"


def test_potion_levitation_sets_timer():
    """POT_LEVITATION — vendor incr_itimeout(&HLevitation, rn1(140,10))."""
    state = _state_with_potion(PotionEffect.LEVITATION)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.status.timed_intrinsics[Intrinsic.LEVITATION]) > 0


def test_potion_paralysis_sets_frozen_timer():
    """POT_PARALYSIS — vendor nomul(-(rn1(10,25-12*bcsign)))."""
    state = _state_with_potion(PotionEffect.PARALYSIS)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.status.timed_statuses[int(TimedStatus.FROZEN)]) > 0


def test_potion_hallucination_sets_timer():
    """POT_HALLUCINATION — vendor make_hallucinated(rn1(200,600-300*bcsign))."""
    state = _state_with_potion(PotionEffect.HALLUCINATION)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.status.timed_statuses[int(TimedStatus.HALLUCINATION)]) > 0


def test_potion_blindness_sets_timer():
    """POT_BLINDNESS — vendor make_blinded(rn1(200,250-125*bcsign))."""
    state = _state_with_potion(PotionEffect.BLINDNESS)
    result = quaff_potion(state, _RNG, 0)
    assert int(result.status.timed_statuses[int(TimedStatus.BLIND)]) > 0


def test_potion_blindness_blessed_shorter_than_cursed():
    """Blessed potion blindness must be shorter than cursed (vendor bcsign sign-flip)."""
    s_b = _state_with_potion(PotionEffect.BLINDNESS, buc=_BUC_BLESSED)
    s_c = _state_with_potion(PotionEffect.BLINDNESS, buc=_BUC_CURSED)
    r_b = quaff_potion(s_b, _RNG, 0)
    r_c = quaff_potion(s_c, _RNG, 0)
    b_blind = int(r_b.status.timed_statuses[int(TimedStatus.BLIND)])
    c_blind = int(r_c.status.timed_statuses[int(TimedStatus.BLIND)])
    assert b_blind < c_blind, (
        f"blessed blind timer {b_blind} should be < cursed {c_blind}"
    )


# ===========================================================================
# B. SCROLL TESTS — vendor/nethack/src/read.c::seffects
# ===========================================================================

def test_scroll_enchant_armor_increases_enchantment():
    """SCR_ENCHANT_ARMOR — vendor seffect_enchant_armor on worn body armor."""
    scroll = make_item(
        category=int(ItemCategory.SCROLL),
        type_id=_SCROLL_BASE_ID + int(ScrollEffect.ENCHANT_ARMOR),
        quantity=1,
        buc_status=_BUC_UNCURSED,
    )
    armor = make_item(
        category=int(ItemCategory.ARMOR),
        type_id=1,
        quantity=1,
        enchantment=0,
    )
    state = EnvState.default(_RNG)
    inv = InventoryState.from_items([scroll, armor])
    inv = inv.replace(worn_armor=inv.worn_armor.at[0].set(jnp.int8(1)))
    state = state.replace(inventory=inv)
    result = read_scroll(state, _RNG, 0)
    assert int(result.inventory.items.enchantment[1]) > 0, (
        f"Armor enchantment not increased: {int(result.inventory.items.enchantment[1])}"
    )


def test_scroll_identify_marks_item_identified():
    """SCR_IDENTIFY uncursed identifies first unidentified item."""
    scroll = make_item(
        category=int(ItemCategory.SCROLL),
        type_id=_SCROLL_BASE_ID + int(ScrollEffect.IDENTIFY),
        quantity=1,
        buc_status=_BUC_UNCURSED,
    )
    unid_potion = Item(
        category=jnp.int8(int(ItemCategory.POTION)),
        type_id=jnp.int16(_POTION_BASE_ID + int(PotionEffect.HEALING)),
        buc_status=jnp.int8(_BUC_UNCURSED),
        enchantment=jnp.int8(0),
        charges=jnp.int8(0),
        identified=jnp.bool_(False),
        quantity=jnp.int16(1),
        weight=jnp.int32(20),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
    )
    state = EnvState.default(_RNG)
    state = state.replace(
        inventory=InventoryState.from_items([scroll, unid_potion])
    )
    result = read_scroll(state, _RNG, 0)
    assert bool(result.inventory.items.identified[1]), (
        "SCR_IDENTIFY must mark first unid item identified"
    )


@pytest.mark.timeout(900)
def test_scroll_teleportation_moves_player():
    """SCR_TELEPORTATION — vendor tele().

    Wave 43d: the helper ``_state_with_scroll`` returns a default
    EnvState whose terrain is all ``TileType.UNKNOWN`` (=0), and
    ``_teleds`` only accepts walkable tiles — so the 40-try rejection
    loop never finds a destination and the player stays put.  Paint a
    floor patch on the current level so the teleport has somewhere
    valid to land.

    Cite: vendor/nethack/src/teleport.c::safe_teleds lines 716-770;
          vendor/nethack/src/read.c::seffect_teleportation.
    """
    state = _state_with_scroll(ScrollEffect.TELEPORTATION)
    state = state.replace(player_pos=jnp.array([10, 10], dtype=jnp.int16))
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    # Paint a wide floor band so safe_teleds finds a valid tile.
    floor_layer = jnp.full(
        state.terrain[b, lv].shape, int(TileType.FLOOR), dtype=state.terrain.dtype
    )
    state = state.replace(terrain=state.terrain.at[b, lv].set(floor_layer))

    moved = False
    for seed in range(10):
        rng = jax.random.PRNGKey(seed)
        result = read_scroll(state, rng, 0)
        if not jnp.array_equal(result.player_pos, state.player_pos):
            moved = True
            break
    assert moved, "SCR_TELEPORTATION should change player position"


def test_scroll_magic_mapping_reveals_level():
    """SCR_MAGIC_MAPPING — vendor level_mapalot()."""
    state = _state_with_scroll(ScrollEffect.MAGIC_MAPPING)
    result = read_scroll(state, _RNG, 0)
    b = int(result.dungeon.current_branch)
    lv = int(result.dungeon.current_level) - 1
    assert bool(jnp.all(result.explored[b, lv])), "Magic mapping must reveal level"


def test_scroll_light_marks_level_explored():
    """SCR_LIGHT — vendor seffect_light marks tiles lit/explored."""
    state = _state_with_scroll(ScrollEffect.LIGHT)
    result = read_scroll(state, _RNG, 0)
    b = int(result.dungeon.current_branch)
    lv = int(result.dungeon.current_level) - 1
    assert bool(jnp.all(result.explored[b, lv]))


def test_scroll_create_monster_is_noop_in_wave3():
    """SCR_CREATE_MONSTER — Wave 3 stub; must not crash and must decrement scroll qty."""
    state = _state_with_scroll(ScrollEffect.CREATE_MONSTER)
    before = int(state.inventory.items.quantity[0])
    result = read_scroll(state, _RNG, 0)
    after = int(result.inventory.items.quantity[0])
    assert after == before - 1


def test_scroll_charging_recharges_wand():
    """SCR_CHARGING — vendor recharge wand."""
    scroll = make_item(
        category=int(ItemCategory.SCROLL),
        type_id=_SCROLL_BASE_ID + int(ScrollEffect.CHARGING),
        quantity=1,
        buc_status=_BUC_UNCURSED,
    )
    # Slot 1: a wand with 2 charges.
    wand = Item(
        category=jnp.int8(int(ItemCategory.WAND)),
        type_id=jnp.int16(int(WandEffect.MAGIC_MISSILE)),
        buc_status=jnp.int8(_BUC_UNCURSED),
        enchantment=jnp.int8(0),
        charges=jnp.int8(2),
        identified=jnp.bool_(True),
        quantity=jnp.int16(1),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
    )
    state = EnvState.default(_RNG)
    state = state.replace(inventory=InventoryState.from_items([scroll, wand]))
    result = read_scroll(state, _RNG, 0)
    new_charges = int(result.inventory.items.charges[1])
    assert new_charges > 2, f"Wand charges should increase from 2, got {new_charges}"


def test_scroll_scare_monster_is_noop_in_wave3():
    """SCR_SCARE_MONSTER — Wave 3 stub; must not crash and must consume scroll."""
    state = _state_with_scroll(ScrollEffect.SCARE_MONSTER)
    before = int(state.inventory.items.quantity[0])
    result = read_scroll(state, _RNG, 0)
    after = int(result.inventory.items.quantity[0])
    assert after == before - 1


# ===========================================================================
# C. WAND TESTS — vendor/nethack/src/zap.c
# ===========================================================================

def test_wand_light_marks_all_explored():
    """WAN_LIGHT — vendor litroom() lights a radius-5 disc around @.

    Cite: vendor/nethack/src/read.c::litroom line 2601:
      do_clear_area(u.ux, u.uy, blessed_effect ? 9 : 5, set_lit, ...)
    so an uncursed wand of light lights tiles within disc radius 5 around
    the hero — NOT the entire map.
    """
    state = _wand_state_with_wand(WandEffect.LIGHT, player_row=10, player_col=10)
    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))
    # Player tile lit.
    assert bool(result.explored[10, 10])
    # Tile beyond radius 5 must stay dark.
    assert not bool(result.explored[0, 0])


@pytest.mark.timeout(900)
def test_wand_digging_carves_wall():
    """WAN_DIGGING — vendor dig_map turns WALL into OPEN_DOOR (D_NODOOR).

    Wave D12 rebalance: in non-maze levels vendor zap.c:1725 carves WALL
    tiles into D_NODOOR (open doorway), not CORRIDOR.  CORRIDOR is the
    outcome only for *stone* tiles (zap.c:1731).  Older assertions
    expected CORRIDOR; updated to match the per-tile outcomes in
    Nethax/nethax/subsystems/items_wands.py::_effect_digging lines
    1220-1230.

    Cite: vendor/nethack/src/dig.c lines 1714-1731.
    """
    state = _wand_state_with_wand(WandEffect.DIGGING, player_row=10, player_col=10)
    # Place WALL north of player.
    new_terrain = state.terrain
    for row in range(2, 10):
        new_terrain = new_terrain.at[row, 10].set(jnp.int8(int(TileType.WALL)))
    state = state.replace(terrain=new_terrain)
    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(0))
    changed = 0
    for row in range(3, 10):
        if int(result.terrain[row, 10]) == int(TileType.OPEN_DOOR):
            changed += 1
    assert changed > 0, "WAN_DIGGING should carve WALL tiles into OPEN_DOOR"


def test_wand_opening_unlocks_doors():
    """WAN_OPENING — vendor zap.c iterates map opening closed doors."""
    state = _wand_state_with_wand(WandEffect.OPENING)
    new_terrain = state.terrain.at[10, 12].set(jnp.int8(int(TileType.CLOSED_DOOR)))
    state = state.replace(terrain=new_terrain)
    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))
    assert int(result.terrain[10, 12]) == int(TileType.OPEN_DOOR), (
        "Closed door must be opened"
    )


def test_wand_magic_missile_deals_d6_damage():
    """WAN_MAGIC_MISSILE — vendor zap.c:3464 sets nd=2, so damage is 2d6 (2..12)."""
    state = _wand_state_with_wand(WandEffect.MAGIC_MISSILE)
    state = _place_monster(state, slot=1, row=10, col=11, hp=20)
    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))
    final_hp = int(result.mon_hp[1])
    assert final_hp < 20 and final_hp >= 8, (
        f"Magic missile 2d6 damage should be 2..12; HP went 20→{final_hp}"
    )


def test_wand_fire_damages_monster():
    """WAN_FIRE — vendor buzz() ZT_FIRE d(1,6)."""
    state = _wand_state_with_wand(WandEffect.FIRE)
    state = _place_monster(state, slot=1, row=10, col=11, hp=20)
    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))
    assert int(result.mon_hp[1]) < 20


def test_wand_cold_freezes_water_and_damages():
    """WAN_COLD — vendor buzz() ZT_COLD freezes water + d(1,6) damage."""
    state = _wand_state_with_wand(WandEffect.COLD)
    # Water tile east of player.
    new_terrain = state.terrain.at[10, 12].set(jnp.int8(int(TileType.WATER)))
    state = state.replace(terrain=new_terrain)
    state = _place_monster(state, slot=1, row=10, col=11, hp=20)
    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))
    assert int(result.mon_hp[1]) < 20, "Cold ray damages monster"
    assert int(result.terrain[10, 12]) != int(TileType.WATER), "Cold freezes water"


def test_wand_lightning_deals_damage():
    """WAN_LIGHTNING — vendor buzz() ZT_LIGHTNING d(1,6)."""
    state = _wand_state_with_wand(WandEffect.LIGHTNING)
    state = _place_monster(state, slot=1, row=10, col=11, hp=20)
    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))
    assert int(result.mon_hp[1]) < 20


def test_wand_sleep_sets_asleep_flag():
    """WAN_SLEEP — vendor sleep_monst()."""
    state = _wand_state_with_wand(WandEffect.SLEEP)
    state = _place_monster(state, slot=1, row=10, col=11, hp=20)
    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))
    assert bool(result.mon_asleep[1])


# ===========================================================================
# D. RING TESTS — vendor/nethack/src/do_wear.c::Ring_on
# ===========================================================================

def test_ring_protection_grants_no_intrinsic_but_records_worn():
    """RIN_PROTECTION — vendor stores AC bonus via worn mask, no intrinsic flag."""
    state = _state_with_ring(RingEffect.PROTECTION, enchantment=2)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert int(state.inventory.worn_rings[0]) == 0


def test_ring_regeneration_grants_regen_intrinsic():
    """RIN_REGENERATION — vendor Ring_on grants Intrinsic.REGEN."""
    state = _state_with_ring(RingEffect.REGENERATION)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert bool(state.status.intrinsics[int(Intrinsic.REGEN)])


def test_ring_gain_strength_increases_str():
    """RIN_GAIN_STRENGTH — vendor adjust_attrib(A_STR, +spe)."""
    state = _state_with_ring(RingEffect.GAIN_STRENGTH, enchantment=3)
    base = int(state.player_str)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert int(state.player_str) == base + 3


def test_ring_slow_digestion_grants_intrinsic():
    """RIN_SLOW_DIGESTION — vendor Ring_on grants Intrinsic.SLOW_DIGESTION."""
    state = _state_with_ring(RingEffect.SLOW_DIGESTION)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert bool(state.status.intrinsics[int(Intrinsic.SLOW_DIGESTION)])


def test_ring_fire_resistance_grants_intrinsic():
    """RIN_FIRE_RESISTANCE — vendor Ring_on grants Intrinsic.RESIST_FIRE."""
    state = _state_with_ring(RingEffect.FIRE_RESISTANCE)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert bool(state.status.intrinsics[int(Intrinsic.RESIST_FIRE)])


def test_ring_hunger_starts_hunger_drain():
    """RIN_HUNGER — vendor doubles hunger drain; flagged via HUNGER_RING timer."""
    state = _state_with_ring(RingEffect.HUNGER)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert int(state.status.timed_statuses[int(TimedStatus.HUNGER_RING)]) > 0


def test_ring_levitation_grants_intrinsic():
    """RIN_LEVITATION — vendor Ring_on grants Intrinsic.LEVITATION."""
    state = _state_with_ring(RingEffect.LEVITATION)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert bool(state.status.intrinsics[int(Intrinsic.LEVITATION)])


def test_ring_free_action_grants_intrinsic():
    """RIN_FREE_ACTION — vendor Ring_on grants Intrinsic.FREE_ACTION."""
    state = _state_with_ring(RingEffect.FREE_ACTION)
    state = put_on_ring(state, _RNG, slot_idx=0, hand=0)
    assert bool(state.status.intrinsics[int(Intrinsic.FREE_ACTION)])


# ===========================================================================
# E. AMULET TESTS — vendor/nethack/src/do_wear.c::Amulet_on
# ===========================================================================

def test_amulet_life_saving_grants_lifesaved():
    """AMULET_OF_LIFE_SAVING — vendor Amulet_on grants Intrinsic.LIFESAVED."""
    state = _state_with_amulet(AmuletEffect.LIFE_SAVING)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert bool(state.status.intrinsics[int(Intrinsic.LIFESAVED)])


def test_amulet_reflection_grants_reflecting():
    """AMULET_OF_REFLECTION — vendor Amulet_on grants Intrinsic.REFLECTING."""
    state = _state_with_amulet(AmuletEffect.REFLECTION)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert bool(state.status.intrinsics[int(Intrinsic.REFLECTING)])


def test_amulet_esp_grants_telepathy():
    """AMULET_OF_ESP — vendor Amulet_on grants Intrinsic.TELEPATHY."""
    state = _state_with_amulet(AmuletEffect.ESP)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert bool(state.status.intrinsics[int(Intrinsic.TELEPATHY)])


def test_amulet_magical_breathing_grants_breathless():
    """AMULET_OF_MAGICAL_BREATHING — vendor Amulet_on grants Intrinsic.BREATHLESS."""
    state = _state_with_amulet(AmuletEffect.MAGICAL_BREATHING)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert bool(state.status.intrinsics[int(Intrinsic.BREATHLESS)])


def test_amulet_strangulation_starts_strangled_timer():
    """AMULET_OF_STRANGULATION — vendor sets Strangled=6 on wear."""
    state = _state_with_amulet(AmuletEffect.STRANGULATION)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert int(state.status.timed_statuses[int(TimedStatus.STRANGLED)]) > 0


def test_amulet_unchanging_grants_unchanging_intrinsic():
    """AMULET_OF_UNCHANGING — vendor Amulet_on grants Intrinsic.UNCHANGING."""
    state = _state_with_amulet(AmuletEffect.UNCHANGING)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert bool(state.status.intrinsics[int(Intrinsic.UNCHANGING)])


def test_amulet_change_is_noop_in_wave3():
    """AMULET_OF_CHANGE — Wave 3 stub: must not crash; worn_amulet is set."""
    state = _state_with_amulet(AmuletEffect.CHANGE)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert int(state.inventory.worn_amulet) == 0


def test_amulet_restful_sleep_starts_sleepy_timer():
    """AMULET_OF_RESTFUL_SLEEP — vendor sets SLEEPY timer (rnd 2..100)."""
    state = _state_with_amulet(AmuletEffect.RESTFUL_SLEEP)
    state = wear_amulet(state, _RNG, slot_idx=0)
    assert int(state.status.timed_statuses[int(TimedStatus.SLEEPY)]) > 0
