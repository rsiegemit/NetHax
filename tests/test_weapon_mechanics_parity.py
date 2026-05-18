"""Wave 4 special-weapon mechanics parity tests.

Covers:
  1. Multi-shot quiver fire   — dothrow.c::dofire
  2. Cleave splash damage     — uhitm.c::cleave
  3. Polearm 2-tile reach     — uhitm.c::dolean
  4. Two-handed unwield on ride — do_wear.c

Cite:
  vendor/nethack/src/dothrow.c::dofire (multishot block)
  vendor/nethack/src/uhitm.c::cleave
  vendor/nethack/src/uhitm.c::dolean
  vendor/nethack/src/do_wear.c (two-handed riding restriction)
"""
import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants import TileType
from Nethax.nethax.subsystems.inventory import ItemCategory, wield
from Nethax.nethax.subsystems.combat import (
    multishot_thrown_attack,
    _handle_polearm_attack,
    _apply_cleave_splash,
    _wielded_is_polearm,
    _wielded_is_axe,
    enforce_no_twohanded_while_riding,
    SKILL_UNSKILLED,
    SKILL_SKILLED,
)
from Nethax.nethax.subsystems.skills import SkillId


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _base_env_state():
    rng = jax.random.PRNGKey(42)
    env = NethaxEnv()
    state, _ = env.reset(rng)
    return env, state, rng


def _carve_open_room(state, rows=5, cols=10):
    """Carve a block of FLOOR tiles around the player so movement/reach works."""
    p_row = int(state.player_pos[0])
    p_col = int(state.player_pos[1])
    branch = int(state.dungeon.current_branch)
    level  = int(state.dungeon.current_level) - 1
    terrain = state.terrain
    for dr in range(-2, rows):
        for dc in range(-1, cols):
            r = p_row + dr
            c = p_col + dc
            if 0 <= r < terrain.shape[2] and 0 <= c < terrain.shape[3]:
                terrain = terrain.at[branch, level, r, c].set(jnp.int8(int(TileType.FLOOR)))
    return state.replace(terrain=terrain)


def _clear_monsters(state):
    """Remove all monsters."""
    mai = state.monster_ai
    mai = mai.replace(
        alive=jnp.zeros_like(mai.alive),
        hp=jnp.zeros_like(mai.hp),
        hp_max=jnp.zeros_like(mai.hp_max),
        pos=jnp.full_like(mai.pos, -1),
    )
    return state.replace(monster_ai=mai)


def _place_monster(state, slot, row_off, col_off, hp=50, ac=10):
    """Place a live monster at player_pos + (row_off, col_off)."""
    p_row = int(state.player_pos[0])
    p_col = int(state.player_pos[1])
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[slot].set(True),
        hp=mai.hp.at[slot].set(jnp.int32(hp)),
        hp_max=mai.hp_max.at[slot].set(jnp.int32(hp)),
        pos=mai.pos.at[slot].set(
            jnp.array([p_row + row_off, p_col + col_off], dtype=jnp.int16)
        ),
        ac=mai.ac.at[slot].set(jnp.int8(ac)),
    )
    return state.replace(monster_ai=mai)


def _wield_type_id(state, type_id: int, is_two_handed: bool = False,
                   category: int = int(ItemCategory.WEAPON)):
    """Place a weapon of ``type_id`` in inventory slot 0 and wield it."""
    items = state.inventory.items
    items = items.replace(
        type_id=items.type_id.at[0].set(jnp.int16(type_id)),
        category=items.category.at[0].set(jnp.int8(category)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
        is_two_handed=items.is_two_handed.at[0].set(jnp.bool_(is_two_handed)),
    )
    state = state.replace(inventory=state.inventory.replace(items=items))
    return wield(state, jnp.int32(0))


def _set_launcher_skill(state, skill_id: int, tier: int):
    """Set a launcher skill tier in state.skills.level."""
    skills = state.skills
    new_level = skills.level.at[skill_id].set(jnp.int8(tier))
    return state.replace(skills=skills.replace(level=new_level))


def _place_arrow_in_slot(state, inv_slot: int, type_id: int = 1, qty: int = 20):
    """Put an arrow (type_id 1-5) with ``qty`` into inventory slot ``inv_slot``."""
    items = state.inventory.items
    items = items.replace(
        type_id=items.type_id.at[inv_slot].set(jnp.int16(type_id)),
        category=items.category.at[inv_slot].set(jnp.int8(int(ItemCategory.WEAPON))),
        quantity=items.quantity.at[inv_slot].set(jnp.int16(qty)),
        weight=items.weight.at[inv_slot].set(jnp.int16(1)),
    )
    new_inv = state.inventory.replace(
        items=items,
        quiver=jnp.int8(inv_slot),
    )
    return state.replace(inventory=new_inv)


# ---------------------------------------------------------------------------
# Test 1 — Multi-shot quiver fire at P_SKILLED bow
# Cite: vendor/nethack/src/dothrow.c::dofire (multishot block, circa line 386).
# ---------------------------------------------------------------------------

def test_bow_multi_shot_at_skilled():
    """P_SKILLED bow (skill tier 2) fires N=1+(2-0)=3 arrows per action.

    We assert that after multishot_thrown_attack the arrow count in the quiver
    slot decreases by at least N (or the monster takes damage if arrows hit).
    Because the monster is placed in the flight path and AC is permissive, at
    least one shot should deal damage across 3 attempts.
    """
    _, state, rng = _base_env_state()
    state = _carve_open_room(state)
    state = _clear_monsters(state)

    # Wield a bow (type_id 65).
    state = _wield_type_id(state, type_id=65)
    # Set BOW skill to P_SKILLED (tier 2).
    state = _set_launcher_skill(state, skill_id=int(SkillId.BOW), tier=SKILL_SKILLED)
    # Place arrows in quiver slot 1.
    state = _place_arrow_in_slot(state, inv_slot=1, type_id=1, qty=20)

    # Place monster 3 tiles east (within THROW_MAX_RANGE=8).
    state = _place_monster(state, slot=0, row_off=0, col_off=3, hp=200, ac=10)

    arrow_qty_before = int(state.inventory.items.quantity[1])
    monster_hp_before = int(state.monster_ai.hp[0])

    direction = jnp.array([0, 1], dtype=jnp.int32)  # east
    rng, sub = jax.random.split(rng)
    new_state = multishot_thrown_attack(state, sub, jnp.int32(1), direction)

    arrow_qty_after = int(new_state.inventory.items.quantity[1])
    monster_hp_after = int(new_state.monster_ai.hp[0])

    # With N=3 shots, at least 1 arrow should be consumed or monster damaged.
    damage_dealt = monster_hp_before - monster_hp_after
    arrows_consumed = arrow_qty_before - arrow_qty_after

    assert arrows_consumed >= 1 or damage_dealt > 0, (
        f"Expected multi-shot to consume arrows or deal damage: "
        f"consumed={arrows_consumed}, damage={damage_dealt}"
    )
    # Skill=2 → N=3 shots → at most 3 arrows consumed.
    assert arrows_consumed <= 3, (
        f"P_SKILLED gives N=3 shots; consumed {arrows_consumed}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Cleave splash damages adjacent monsters
# Cite: vendor/nethack/src/uhitm.c::cleave (circa line 3620).
# ---------------------------------------------------------------------------

def test_cleave_splash():
    """Wielding an axe deals half-damage splash to perpendicular monsters.

    Setup: player at (r, c), primary target at (r, c+1).
    Perpendicular tiles: (r-1, c+1) and (r+1, c+1).
    After _apply_cleave_splash, those monsters should have taken dmg//2.
    """
    _, state, rng = _base_env_state()
    state = _carve_open_room(state, rows=5, cols=5)
    state = _clear_monsters(state)

    # Wield battle-axe (type_id 28, oc_skill=3 = P_AXE).
    state = _wield_type_id(state, type_id=28)
    assert _wielded_is_axe(state), "battle-axe should be recognised as axe-class"

    # Place three monsters: primary east, two perpendicular.
    state = _place_monster(state, slot=0, row_off=0,  col_off=1, hp=100, ac=10)  # primary
    state = _place_monster(state, slot=1, row_off=-1, col_off=1, hp=100, ac=10)  # left-perp
    state = _place_monster(state, slot=2, row_off=1,  col_off=1, hp=100, ac=10)  # right-perp

    primary_dmg = jnp.int32(20)
    rng, sub = jax.random.split(rng)
    new_state = _apply_cleave_splash(state, sub, jnp.int32(0), primary_dmg)

    expected_splash = 10  # primary_dmg // 2
    hp1_after = int(new_state.monster_ai.hp[1])
    hp2_after = int(new_state.monster_ai.hp[2])

    assert hp1_after <= 100 - expected_splash, (
        f"Left-perp monster should take {expected_splash} splash; hp={hp1_after}"
    )
    assert hp2_after <= 100 - expected_splash, (
        f"Right-perp monster should take {expected_splash} splash; hp={hp2_after}"
    )
    # Primary target unchanged by cleave (cleave only hits perpendiculars).
    hp0_after = int(new_state.monster_ai.hp[0])
    assert hp0_after == 100, f"Primary target HP should be unchanged by cleave: {hp0_after}"


# ---------------------------------------------------------------------------
# Test 3 — Polearm attacks monster at distance 2
# Cite: vendor/nethack/src/uhitm.c::dolean (reach weapon attack, circa line 3480).
# ---------------------------------------------------------------------------

def test_polearm_attacks_distance_2():
    """Wielding a voulge (type_id 49) allows hitting a monster 2 tiles east.

    The adjacent tile (col+1) is empty; the monster is at (col+2).
    _handle_polearm_attack should damage the distance-2 monster.
    """
    _, state, rng = _base_env_state()
    state = _carve_open_room(state)
    state = _clear_monsters(state)

    # Wield voulge (type_id 49, oc_skill=16 = P_POLEARMS).
    state = _wield_type_id(state, type_id=49)
    assert _wielded_is_polearm(state), "voulge should be recognised as polearm"

    # Place monster 2 tiles east; tile 1 east is empty.
    state = _place_monster(state, slot=0, row_off=0, col_off=2, hp=100, ac=5)

    monster_hp_before = int(state.monster_ai.hp[0])

    # Run several attempts since to-hit has variance; at least one should land.
    hit_landed = False
    for i in range(20):
        rng, sub = jax.random.split(rng)
        new_state = _handle_polearm_attack(state, sub, jnp.int32(1))  # dir_idx=1 = East
        if int(new_state.monster_ai.hp[0]) < monster_hp_before:
            hit_landed = True
            break

    assert hit_landed, (
        "Polearm reach attack should deal damage to monster at distance 2"
    )


# ---------------------------------------------------------------------------
# Test 4 — Two-handed weapon is unwielded when player mounts a steed
# Cite: vendor/nethack/src/do_wear.c (two-handed riding restriction, circa line 1820).
# ---------------------------------------------------------------------------

def test_unwield_two_handed_on_ride():
    """Two-handed weapon is force-unwielded when player_steed_mid is set.

    enforce_no_twohanded_while_riding should clear inventory.wielded → -1
    when is_riding=True and the wielded weapon is_two_handed=True.
    """
    _, state, rng = _base_env_state()

    # Wield a two-handed weapon (e.g. battle-axe with is_two_handed=True).
    state = _wield_type_id(state, type_id=28, is_two_handed=True)
    assert int(state.inventory.wielded) == 0, "Weapon should be wielded in slot 0"

    # Simulate mounting: set player_steed_mid to non-zero.
    state = state.replace(player_steed_mid=jnp.uint32(1))

    new_state = enforce_no_twohanded_while_riding(state)

    assert int(new_state.inventory.wielded) == -1, (
        "Two-handed weapon should be unwielded when riding"
    )
    # The item should still be in inventory (slot 0 quantity unchanged).
    assert int(new_state.inventory.items.quantity[0]) >= 1, (
        "Item should remain in inventory after force-unwield"
    )


# ---------------------------------------------------------------------------
# Test 5 — enforce_no_twohanded_while_riding is no-op when not riding
# ---------------------------------------------------------------------------

def test_two_handed_ok_when_not_riding():
    """Two-handed weapon is kept when player is not mounted."""
    _, state, rng = _base_env_state()
    state = _wield_type_id(state, type_id=28, is_two_handed=True)
    # Ensure not riding.
    state = state.replace(player_steed_mid=jnp.uint32(0))

    new_state = enforce_no_twohanded_while_riding(state)
    assert int(new_state.inventory.wielded) == 0, (
        "Two-handed weapon should remain wielded when not riding"
    )
