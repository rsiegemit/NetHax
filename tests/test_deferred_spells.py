"""Wave 6 #79 — tests for the 19 spell effects that Wave 6 #76 left as no-ops.

Each test invokes the per-spell effect handler directly through
``_EFFECT_DISPATCH`` and asserts the vendor-correct state mutation.

Vendor sources are cited in each test's docstring.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.magic import (
    SpellId,
    _EFFECT_DISPATCH,
    _StateAdapter,
    _effect_level_teleport,
)
from Nethax.nethax.subsystems.inventory import make_item, _items_from_list
from Nethax.nethax.subsystems.features import DoorState
from Nethax.nethax.constants.tiles import TileType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(**over) -> EnvState:
    """Return a default EnvState with WIZARD role + full pw."""
    rng = jax.random.PRNGKey(0)
    state = EnvState.default(rng)
    state = state.replace(
        player_pw=jnp.int32(over.pop("player_pw", 200)),
        player_pw_max=jnp.int32(over.pop("player_pw_max", 200)),
        player_hp=jnp.int32(over.pop("player_hp", 50)),
        player_hp_max=jnp.int32(over.pop("player_hp_max", 100)),
        player_int=jnp.int8(over.pop("player_int", 18)),
        player_wis=jnp.int8(over.pop("player_wis", 18)),
        player_xl=jnp.int32(over.pop("player_xl", 10)),
        player_role=jnp.int8(over.pop("player_role", 12)),
    )
    for k, v in over.items():
        state = state.replace(**{k: v})
    return state


def _run_effect_obj(handler, state: EnvState, seed: int = 0) -> EnvState:
    adapter = _StateAdapter(state)
    rng = jax.random.PRNGKey(seed)
    result = handler(adapter, rng)
    if isinstance(result, dict):
        for k, v in result.items():
            adapter[k] = v
    return adapter.build()


def _run_effect(spell_id: SpellId, state: EnvState, seed: int = 0) -> EnvState:
    return _run_effect_obj(_EFFECT_DISPATCH[spell_id], state, seed=seed)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dig_carves_4_corridor_tiles_east():
    """DIG: 4 tiles east of player become CORRIDOR.

    Cite: vendor/nethack/src/dig.c::zap_dig.
    """
    state = _base_state()
    state = state.replace(player_pos=jnp.array([5, 5], dtype=jnp.int16))
    new_state = _run_effect(SpellId.DIG, state, seed=1)
    br = int(new_state.dungeon.current_branch)
    lv = int(new_state.dungeon.current_level) - 1
    row = 5
    for col in range(6, 10):
        tile = int(new_state.terrain[br, lv, row, col])
        assert tile == int(TileType.CORRIDOR), (
            f"tile at (row=5,col={col}) should be CORRIDOR (got {tile})"
        )


def test_knock_opens_adjacent_closed_door():
    """KNOCK: adjacent CLOSED_DOOR → OPEN_DOOR via features.door_state.

    Cite: vendor/nethack/src/lock.c::do_oclose.
    """
    state = _base_state()
    state = state.replace(player_pos=jnp.array([5, 5], dtype=jnp.int16))
    # Drop a CLOSED door one tile east of player at (5, 6) on current level.
    br = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    max_levels = state.dungeon.stair_links.shape[1]
    flat_lv = br * max_levels + lv
    new_door_state = state.features.door_state.at[flat_lv, 5, 6].set(
        jnp.int8(int(DoorState.CLOSED))
    )
    state = state.replace(features=state.features.replace(door_state=new_door_state))

    new_state = _run_effect(SpellId.KNOCK, state, seed=2)
    after = int(new_state.features.door_state[flat_lv, 5, 6])
    assert after == int(DoorState.OPEN), (
        f"door at (5, 6) should now be OPEN; got {after}"
    )


def test_wizard_lock_closes_and_locks_open_door():
    """WIZARD_LOCK: adjacent OPEN_DOOR → LOCKED.

    Cite: vendor/nethack/src/lock.c::do_oclose (inverse).
    """
    state = _base_state()
    state = state.replace(player_pos=jnp.array([5, 5], dtype=jnp.int16))
    br = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    max_levels = state.dungeon.stair_links.shape[1]
    flat_lv = br * max_levels + lv
    new_door_state = state.features.door_state.at[flat_lv, 5, 6].set(
        jnp.int8(int(DoorState.OPEN))
    )
    state = state.replace(features=state.features.replace(door_state=new_door_state))

    new_state = _run_effect(SpellId.WIZARD_LOCK, state, seed=3)
    after = int(new_state.features.door_state[flat_lv, 5, 6])
    assert after == int(DoorState.LOCKED), (
        f"door at (5, 6) should now be LOCKED; got {after}"
    )


def test_jumping_moves_player_east_2():
    """JUMPING: player_pos shifts by (0, +2) when target tile is FLOOR.

    Cite: vendor/nethack/src/cmd.c::dojump.
    """
    state = _base_state()
    state = state.replace(player_pos=jnp.array([5, 5], dtype=jnp.int16))
    # Make the destination tile (5, 7) FLOOR.
    br = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain.at[br, lv, 5, 7].set(jnp.int8(int(TileType.FLOOR)))
    state = state.replace(terrain=new_terrain)

    new_state = _run_effect(SpellId.JUMPING, state, seed=4)
    assert int(new_state.player_pos[0]) == 5
    assert int(new_state.player_pos[1]) == 7, (
        f"player should land at col=7; got {int(new_state.player_pos[1])}"
    )


def test_teleport_away_moves_player_to_floor():
    """TELEPORT_AWAY: player lands on a FLOOR tile.

    Cite: vendor/nethack/src/teleport.c::dotele.
    """
    state = _base_state()
    state = state.replace(player_pos=jnp.array([0, 0], dtype=jnp.int16))
    # Paint a small floor patch on the current level so destination exists.
    br = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    floor_layer = jnp.zeros_like(state.terrain[br, lv])
    floor_layer = floor_layer.at[10, 10].set(jnp.int8(int(TileType.FLOOR)))
    floor_layer = floor_layer.at[10, 11].set(jnp.int8(int(TileType.FLOOR)))
    floor_layer = floor_layer.at[10, 12].set(jnp.int8(int(TileType.FLOOR)))
    new_terrain = state.terrain.at[br, lv].set(floor_layer)
    state = state.replace(terrain=new_terrain)

    new_state = _run_effect(SpellId.TELEPORT_AWAY, state, seed=5)
    rr = int(new_state.player_pos[0])
    cc = int(new_state.player_pos[1])
    landed = int(new_state.terrain[br, lv, rr, cc])
    assert landed == int(TileType.FLOOR), (
        f"player should land on FLOOR; landed on tile {landed} at ({rr},{cc})"
    )


def test_identify_marks_first_slot_identified():
    """IDENTIFY: first unidentified inventory slot becomes identified.

    Cite: vendor/nethack/src/read.c::SCR_IDENTIFY.
    """
    state = _base_state()
    item = make_item(category=4, type_id=100, quantity=1)
    item = item.replace(identified=jnp.bool_(False))
    inv = state.inventory.replace(items=_items_from_list([item]))
    state = state.replace(inventory=inv)
    assert not bool(state.inventory.items.identified[0])

    new_state = _run_effect(SpellId.IDENTIFY, state, seed=6)
    assert bool(new_state.inventory.items.identified[0]), (
        "first inventory slot should be identified"
    )


def test_detect_monsters_sets_flag_until_100_turns_later():
    """DETECT_MONSTERS: identification.detect_monsters_until_turn = timestep + 100.

    Cite: vendor/nethack/src/detect.c::monster_detect.
    """
    state = _base_state()
    state = state.replace(timestep=jnp.int32(50))
    new_state = _run_effect(SpellId.DETECT_MONSTERS, state, seed=7)
    assert int(new_state.identification.detect_monsters_until_turn) == 150


def test_detect_food_sets_flag():
    """DETECT_FOOD: identification.detect_food_until_turn = timestep + 100.

    Cite: vendor/nethack/src/detect.c::food_detect.
    """
    state = _base_state()
    state = state.replace(timestep=jnp.int32(10))
    new_state = _run_effect(SpellId.DETECT_FOOD, state, seed=8)
    assert int(new_state.identification.detect_food_until_turn) == 110


def test_detect_treasure_sets_flag():
    """DETECT_TREASURE: identification.detect_treasure_until_turn = timestep + 100.

    Cite: vendor/nethack/src/detect.c::trap_detect (treasure branch).
    """
    state = _base_state()
    state = state.replace(timestep=jnp.int32(0))
    new_state = _run_effect(SpellId.DETECT_TREASURE, state, seed=9)
    assert int(new_state.identification.detect_treasure_until_turn) == 100


def test_clairvoyance_sets_all_explored_true():
    """CLAIRVOYANCE: explored[branch, level, :, :] = True.

    Cite: vendor/nethack/src/detect.c::do_vicinity_map (extended scope).
    """
    state = _base_state()
    new_state = _run_effect(SpellId.CLAIRVOYANCE, state, seed=10)
    br = int(new_state.dungeon.current_branch)
    lv = int(new_state.dungeon.current_level) - 1
    assert bool(jnp.all(new_state.explored[br, lv])), (
        "entire current level should be marked explored"
    )


def test_remove_curse_uncurses_worn_items():
    """REMOVE_CURSE: all worn/wielded items become buc_status=UNCURSED(2).

    Cite: vendor/nethack/src/read.c::SCR_REMOVE_CURSE.
    """
    state = _base_state()
    item = make_item(category=1, type_id=10, quantity=1, buc_status=1)  # CURSED
    inv = state.inventory.replace(items=_items_from_list([item]))
    # Wield it.
    inv = inv.replace(wielded=jnp.int8(0))
    state = state.replace(inventory=inv)
    assert int(state.inventory.items.buc_status[0]) == 1  # cursed

    new_state = _run_effect(SpellId.REMOVE_CURSE, state, seed=11)
    assert int(new_state.inventory.items.buc_status[0]) == 2, (
        "wielded item should now be UNCURSED"
    )


def test_restore_ability_restores_drained_stats():
    """RESTORE_ABILITY: every player stat → 18.

    Cite: vendor/nethack/src/potion.c::peffect_restore_ability.
    """
    state = _base_state(
        player_str=jnp.int16(6),
        player_dex=jnp.int8(4),
        player_con=jnp.int8(3),
        player_int=jnp.int8(5),
        player_wis=jnp.int8(7),
        player_cha=jnp.int8(8),
    )
    new_state = _run_effect(SpellId.RESTORE_ABILITY, state, seed=12)
    assert int(new_state.player_str) == 18
    assert int(new_state.player_dex) == 18
    assert int(new_state.player_con) == 18
    assert int(new_state.player_int) == 18
    assert int(new_state.player_wis) == 18
    assert int(new_state.player_cha) == 18


def test_create_familiar_spawns_aligned_monster():
    """CREATE_FAMILIAR: first free monster slot becomes alive + tame + peaceful.

    Cite: vendor/nethack/src/makemon.c::makemon, vendor/nethack/src/dog.c::makedog.
    """
    state = _base_state()
    # Ensure slot 0 is empty (alive=False).
    new_state = _run_effect(SpellId.CREATE_FAMILIAR, state, seed=13)
    assert bool(new_state.monster_ai.alive[0])
    assert bool(new_state.monster_ai.tame[0])
    assert bool(new_state.monster_ai.peaceful[0])


def test_create_monster_spawns_hostile():
    """CREATE_MONSTER: first free monster slot becomes alive + NOT tame/peaceful.

    Cite: vendor/nethack/src/makemon.c::makemon.
    """
    state = _base_state()
    new_state = _run_effect(SpellId.CREATE_MONSTER, state, seed=14)
    assert bool(new_state.monster_ai.alive[0])
    assert not bool(new_state.monster_ai.tame[0])
    assert not bool(new_state.monster_ai.peaceful[0])


def test_cancellation_clears_monster_intrinsics():
    """CANCELLATION: monster slot 0 attack dice are zeroed.

    File-ownership constraint: monster_ai has no intrinsics_mask array,
    so vendor cancel_monst is mirrored by stripping the monster's attack
    capability (the practical effect of "no more special powers").
    Cite: vendor/nethack/src/zap.c::cancel_monst.
    """
    state = _base_state()
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        hp=mai.hp.at[0].set(jnp.int32(20)),
        attack_dice_n=mai.attack_dice_n.at[0].set(jnp.int8(2)),
        attack_dice_sides=mai.attack_dice_sides.at[0].set(jnp.int8(6)),
    )
    state = state.replace(monster_ai=mai)
    assert int(state.monster_ai.attack_dice_n[0]) == 2

    new_state = _run_effect(SpellId.CANCELLATION, state, seed=15)
    assert int(new_state.monster_ai.attack_dice_n[0]) == 0
    assert int(new_state.monster_ai.attack_dice_sides[0]) == 0


def test_light_sets_lit_radius_flag():
    """LIGHT: dungeon.lit_radius_until_turn = timestep + 100.

    Cite: vendor/nethack/src/light.c::do_light_sources.
    """
    state = _base_state()
    state = state.replace(timestep=jnp.int32(7))
    new_state = _run_effect(SpellId.LIGHT, state, seed=16)
    assert int(new_state.dungeon.lit_radius_until_turn) == 107


def test_flame_sphere_spawns_fire_elemental():
    """FLAME_SPHERE: spawn a tame creature with entry_idx=3 (PM_FLAMING_SPHERE).

    Cite: vendor/nethack/src/makemon.c::makemon (PM_FLAMING_SPHERE).
    """
    state = _base_state()
    new_state = _run_effect(SpellId.FLAME_SPHERE, state, seed=17)
    assert bool(new_state.monster_ai.alive[0])
    assert int(new_state.monster_ai.entry_idx[0]) == 3
    assert bool(new_state.monster_ai.tame[0])


def test_freeze_sphere_spawns_freezing_sphere():
    """FREEZE_SPHERE: spawn a tame creature with entry_idx=4 (PM_FREEZING_SPHERE).

    Cite: vendor/nethack/src/makemon.c::makemon (PM_FREEZING_SPHERE).
    """
    state = _base_state()
    new_state = _run_effect(SpellId.FREEZE_SPHERE, state, seed=18)
    assert bool(new_state.monster_ai.alive[0])
    assert int(new_state.monster_ai.entry_idx[0]) == 4
    assert bool(new_state.monster_ai.tame[0])


def test_level_teleport_changes_current_level():
    """LEVEL_TELEPORT: dungeon.current_level lands in [1, MAX_LEVELS_PER_BRANCH].

    Cite: vendor/nethack/src/teleport.c::level_tele.
    """
    from Nethax.nethax.dungeon.branches import MAX_LEVELS_PER_BRANCH
    state = _base_state()
    state = state.replace(dungeon=state.dungeon.replace(current_level=jnp.int8(1)))

    # Run multiple seeds and ensure at least one differs from 1.
    saw_different = False
    for seed in range(20):
        new_state = _run_effect_obj(_effect_level_teleport, state, seed=seed)
        lv = int(new_state.dungeon.current_level)
        assert 1 <= lv <= MAX_LEVELS_PER_BRANCH
        if lv != 1:
            saw_different = True
    assert saw_different, "level_teleport should sometimes change current_level"
