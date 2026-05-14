"""Wave 6 save/load round-trip tests covering every subsystem field group.

Each test mutates a specific Wave 4/5/6 field on a fresh EnvState, then
round-trips through ``save_state`` / ``load_state`` and asserts the loaded
value matches exactly (shape, dtype, and contents).  The closing
``test_save_load_full_state_after_100_steps`` exercises an end-to-end
"play a bit, save, load" cycle and verifies *every* pytree leaf survives.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from Nethax.nethax import NethaxEnv
from Nethax.nethax.save_load import (
    IncompatibleSaveError,
    _NETHAX_SAVE_VERSION,
    load_state,
    save_state,
)
from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.conduct import N_CONDUCTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_state() -> EnvState:
    """Return a freshly-reset EnvState from NethaxEnv."""
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    return state


def _assert_pytree_equal(a, b) -> None:
    """Assert two pytrees have identical treedefs, shapes, dtypes, and values."""
    leaves_a, td_a = jax.tree_util.tree_flatten(a)
    leaves_b, td_b = jax.tree_util.tree_flatten(b)
    assert td_a == td_b, "treedefs differ"
    assert len(leaves_a) == len(leaves_b)
    for i, (la, lb) in enumerate(zip(leaves_a, leaves_b)):
        np_a = np.asarray(la)
        np_b = np.asarray(lb)
        assert np_a.shape == np_b.shape, f"leaf {i}: shape {np_a.shape} vs {np_b.shape}"
        assert np_a.dtype == np_b.dtype, f"leaf {i}: dtype {np_a.dtype} vs {np_b.dtype}"
        assert np.array_equal(np_a, np_b), f"leaf {i}: values differ"


def _round_trip(state: EnvState, tmp_path: Path, name: str) -> EnvState:
    """Save ``state`` to a tmp file under ``name`` and return the loaded copy."""
    path = tmp_path / f"{name}.npz"
    save_state(state, path)
    return load_state(path)


# ---------------------------------------------------------------------------
# Polymorph
# ---------------------------------------------------------------------------

def test_save_load_polymorph_state_fully_preserved(tmp_path: Path) -> None:
    """Every Wave 4/5/6 field on PolymorphState round-trips."""
    state = _fresh_state()

    # Populate every polymorph field with a distinctive non-default value.
    poly = state.polymorph.replace(
        is_polymorphed=jnp.bool_(True),
        current_form_idx=jnp.int16(42),
        poly_timer=jnp.int16(750),
        poly_controlled=jnp.bool_(True),
        controlled_poly_count=jnp.int8(3),
        lycanthropy_form=jnp.int8(7),
        lycanthropy_timer=jnp.int16(15),
        orig_role_idx=jnp.int8(2),
        orig_str=jnp.int16(18),
        orig_dex=jnp.int8(14),
        orig_con=jnp.int8(16),
        orig_hp_max=jnp.int32(55),
        orig_ac=jnp.int32(8),
        attack_types=jnp.array([1, 2, 3, 4, 5, 6], dtype=jnp.uint8),
        attack_damage_types=jnp.array([6, 5, 4, 3, 2, 1], dtype=jnp.uint8),
        attack_n_dice=jnp.array([2, 3, 1, 0, 0, 0], dtype=jnp.uint8),
        attack_n_sides=jnp.array([4, 6, 8, 0, 0, 0], dtype=jnp.uint8),
        orig_attack_types=jnp.array([7, 8, 9, 10, 11, 12], dtype=jnp.uint8),
        orig_attack_damage_types=jnp.array([12, 11, 10, 9, 8, 7], dtype=jnp.uint8),
        orig_attack_n_dice=jnp.array([1, 1, 1, 1, 1, 1], dtype=jnp.uint8),
        orig_attack_n_sides=jnp.array([2, 2, 2, 2, 2, 2], dtype=jnp.uint8),
        intrinsics_mask=jnp.int32(0xABCDEF),
    )
    state = state.replace(polymorph=poly)

    loaded = _round_trip(state, tmp_path, "polymorph")
    _assert_pytree_equal(state.polymorph, loaded.polymorph)


# ---------------------------------------------------------------------------
# Monster AI inventory (Wave 6 #80)
# ---------------------------------------------------------------------------

def test_save_load_monster_ai_inventory_preserved(tmp_path: Path) -> None:
    """All Wave 5/6 MonsterAIState extension fields round-trip."""
    state = _fresh_state()
    mai = state.monster_ai

    # Set a few slots with distinctive values for entry_idx / orig_entry_idx,
    # mtame, apport, and the full Wave 6 #80 minvent block.
    new_entry = mai.entry_idx.at[0].set(jnp.int16(99))
    new_entry = new_entry.at[5].set(jnp.int16(123))
    new_orig = mai.orig_entry_idx.at[0].set(jnp.int16(50))
    new_mtame = mai.mtame.at[0].set(jnp.int8(20))
    new_apport = mai.apport.at[0].set(jnp.int8(9))

    # Monster-inventory: slot 0 holds 2 items, slot 5 holds 1.
    new_inv_cat = mai.inv_category.at[0, 0].set(jnp.int8(3))
    new_inv_cat = new_inv_cat.at[0, 1].set(jnp.int8(2))
    new_inv_cat = new_inv_cat.at[5, 0].set(jnp.int8(1))
    new_inv_type = mai.inv_type_id.at[0, 0].set(jnp.int16(200))
    new_inv_type = new_inv_type.at[0, 1].set(jnp.int16(150))
    new_inv_buc = mai.inv_buc.at[0, 0].set(jnp.int8(1))
    new_inv_qty = mai.inv_quantity.at[0, 0].set(jnp.int16(7))
    new_inv_charges = mai.inv_charges.at[0, 0].set(jnp.int8(5))
    new_inv_ident = mai.inv_identified.at[0, 0].set(jnp.bool_(True))

    new_mai = mai.replace(
        entry_idx=new_entry,
        orig_entry_idx=new_orig,
        mtame=new_mtame,
        apport=new_apport,
        inv_category=new_inv_cat,
        inv_type_id=new_inv_type,
        inv_buc=new_inv_buc,
        inv_quantity=new_inv_qty,
        inv_charges=new_inv_charges,
        inv_identified=new_inv_ident,
    )
    state = state.replace(monster_ai=new_mai)

    loaded = _round_trip(state, tmp_path, "monster_ai")
    _assert_pytree_equal(state.monster_ai, loaded.monster_ai)


# ---------------------------------------------------------------------------
# Containers (Wave 5)
# ---------------------------------------------------------------------------

def test_save_load_containers_preserved(tmp_path: Path) -> None:
    """ContainerState (12 arrays of items, container_type, parent_slot, ...) round-trips."""
    state = _fresh_state()
    cs = state.containers

    new_cs = cs.replace(
        items_category=cs.items_category.at[0, 0].set(jnp.int8(5)),
        items_type_id=cs.items_type_id.at[0, 0].set(jnp.int16(321)),
        items_buc=cs.items_buc.at[0, 0].set(jnp.int8(3)),
        items_enchant=cs.items_enchant.at[0, 0].set(jnp.int8(2)),
        items_charges=cs.items_charges.at[0, 0].set(jnp.int8(7)),
        items_identified=cs.items_identified.at[0, 0].set(jnp.bool_(True)),
        items_quantity=cs.items_quantity.at[0, 0].set(jnp.int16(4)),
        items_weight=cs.items_weight.at[0, 0].set(jnp.int16(150)),
        container_type=cs.container_type.at[0].set(jnp.int8(3)),  # BAG_OF_HOLDING
        parent_slot=cs.parent_slot.at[0].set(jnp.int8(7)),
        is_open=cs.is_open.at[0].set(jnp.bool_(True)),
        container_buc=cs.container_buc.at[0].set(jnp.int8(3)),  # blessed
    )
    state = state.replace(containers=new_cs)

    loaded = _round_trip(state, tmp_path, "containers")
    _assert_pytree_equal(state.containers, loaded.containers)


# ---------------------------------------------------------------------------
# Engrave (Wave 5)
# ---------------------------------------------------------------------------

def test_save_load_engrave_preserved(tmp_path: Path) -> None:
    """EngraveState text/position arrays round-trip — including 'Elbereth' bytes."""
    state = _fresh_state()
    eng = state.engrave

    elbereth = jnp.array(list(b"Elbereth"), dtype=jnp.int8)
    new_text = eng.text.at[3, 4, :].set(elbereth)
    new_has = eng.has_engraving.at[3, 4].set(jnp.bool_(True))
    new_kind = eng.engraving_kind.at[3, 4].set(jnp.int8(1))  # ENGR_DUST

    new_eng = eng.replace(
        text=new_text,
        has_engraving=new_has,
        engraving_kind=new_kind,
    )
    state = state.replace(engrave=new_eng)

    loaded = _round_trip(state, tmp_path, "engrave")
    _assert_pytree_equal(state.engrave, loaded.engrave)

    # Spot-check the actual text bytes.
    assert np.array_equal(
        np.asarray(loaded.engrave.text[3, 4]),
        np.asarray(elbereth),
    )


# ---------------------------------------------------------------------------
# Prayer (Wave 6 #78)
# ---------------------------------------------------------------------------

def test_save_load_prayer_state_all_fields(tmp_path: Path) -> None:
    """Every PrayerState field — including the Wave 6 #78 trouble-state bools."""
    state = _fresh_state()
    pr = state.prayer.replace(
        alignment=jnp.int32(123),
        prayer_timeout=jnp.int32(40),
        luck=jnp.int32(-3),
        lucky_stones=jnp.int32(1),
        god_anger=jnp.int32(2),
        pray_timeout=jnp.int32(55),
        alignment_record=jnp.int16(77),
        last_pray_turn=jnp.int32(2000),
        god_name_idx=jnp.int8(2),
        punished=jnp.bool_(True),
        saddled_cursed=jnp.bool_(True),
        stuck_in_wall=jnp.bool_(True),
        in_region=jnp.bool_(True),
    )
    state = state.replace(prayer=pr)

    loaded = _round_trip(state, tmp_path, "prayer")
    _assert_pytree_equal(state.prayer, loaded.prayer)


# ---------------------------------------------------------------------------
# Identification detect-flags (Wave 6 #79)
# ---------------------------------------------------------------------------

def test_save_load_identification_detect_flags(tmp_path: Path) -> None:
    """Wave 6 detection-spell timer fields on IdentificationState round-trip."""
    state = _fresh_state()
    ident = state.identification.replace(
        detect_monsters_until_turn=jnp.int32(500),
        detect_food_until_turn=jnp.int32(1234),
        detect_treasure_until_turn=jnp.int32(9999),
    )
    state = state.replace(identification=ident)

    loaded = _round_trip(state, tmp_path, "identification")
    _assert_pytree_equal(state.identification, loaded.identification)

    assert int(loaded.identification.detect_monsters_until_turn) == 500
    assert int(loaded.identification.detect_food_until_turn) == 1234
    assert int(loaded.identification.detect_treasure_until_turn) == 9999


# ---------------------------------------------------------------------------
# Dungeon (Wave 6 vibrating-square / lit-radius)
# ---------------------------------------------------------------------------

def test_save_load_dungeon_vibrating_square(tmp_path: Path) -> None:
    """DungeonState.vibrating_square_revealed + lit_radius_until_turn round-trip."""
    state = _fresh_state()
    dg = state.dungeon.replace(
        vibrating_square_revealed=jnp.bool_(True),
        lit_radius_until_turn=jnp.int32(427),
    )
    state = state.replace(dungeon=dg)

    loaded = _round_trip(state, tmp_path, "dungeon")
    _assert_pytree_equal(state.dungeon, loaded.dungeon)

    assert bool(loaded.dungeon.vibrating_square_revealed) is True
    assert int(loaded.dungeon.lit_radius_until_turn) == 427


# ---------------------------------------------------------------------------
# Inventory user-names (Wave 6)
# ---------------------------------------------------------------------------

def test_save_load_inventory_user_names(tmp_path: Path) -> None:
    """InventoryState.user_names plus alternate_weapon_slot + worn_armor_ac_bonus."""
    state = _fresh_state()
    inv = state.inventory

    # Stamp a name like "Excalibur\0" into slot 0.
    name = b"Excalibur"
    name_padded = jnp.array(
        list(name) + [0] * (inv.user_names.shape[1] - len(name)),
        dtype=jnp.int8,
    )
    new_user_names = inv.user_names.at[0].set(name_padded)

    new_inv = inv.replace(
        alternate_weapon_slot=jnp.int8(4),
        worn_armor_ac_bonus=jnp.array(
            [1, 2, 3, 4, 5, 6, 7][: inv.worn_armor_ac_bonus.shape[0]],
            dtype=jnp.int8,
        ),
        user_names=new_user_names,
    )
    state = state.replace(inventory=new_inv)

    loaded = _round_trip(state, tmp_path, "inventory_names")
    _assert_pytree_equal(state.inventory, loaded.inventory)

    # Spot-check: name bytes survive intact.
    loaded_name = bytes(np.asarray(loaded.inventory.user_names[0]).tolist()[: len(name)])
    assert loaded_name == name


# ---------------------------------------------------------------------------
# Combat
# ---------------------------------------------------------------------------

def test_save_load_combat_two_weapon(tmp_path: Path) -> None:
    """CombatState fields including the Wave 5 two_weapon toggle round-trip."""
    state = _fresh_state()
    cb = state.combat.replace(
        weapon_skill=state.combat.weapon_skill.at[0].set(jnp.int8(3)),
        weapon_practice=state.combat.weapon_practice.at[0].set(jnp.int32(101)),
        last_attack_kind=jnp.int32(7),
        last_hit_landed=jnp.bool_(True),
        two_weapon=jnp.bool_(True),
    )
    state = state.replace(combat=cb)

    loaded = _round_trip(state, tmp_path, "combat")
    _assert_pytree_equal(state.combat, loaded.combat)

    assert bool(loaded.combat.two_weapon) is True
    assert bool(loaded.combat.last_hit_landed) is True


# ---------------------------------------------------------------------------
# Conduct — all 13 violations
# ---------------------------------------------------------------------------

def test_save_load_conduct_all_13_violations(tmp_path: Path) -> None:
    """Every ConductState violation flag (all 13) round-trips."""
    state = _fresh_state()
    # Flip all violations on.
    all_violated = jnp.ones((N_CONDUCTS,), dtype=jnp.bool_)
    state = state.replace(conduct=state.conduct.replace(violations=all_violated))

    loaded = _round_trip(state, tmp_path, "conduct_all_on")
    _assert_pytree_equal(state.conduct, loaded.conduct)
    assert bool(jnp.all(loaded.conduct.violations))

    # Flip alternating violations to mix pattern.
    pattern = jnp.array([i % 2 == 0 for i in range(N_CONDUCTS)], dtype=jnp.bool_)
    state = state.replace(conduct=state.conduct.replace(violations=pattern))

    loaded = _round_trip(state, tmp_path, "conduct_pattern")
    assert np.array_equal(
        np.asarray(loaded.conduct.violations),
        np.asarray(pattern),
    )


# ---------------------------------------------------------------------------
# Magic
# ---------------------------------------------------------------------------

def test_save_load_magic_pw_regen_and_spells(tmp_path: Path) -> None:
    """MagicState pw_regen_counter, spell_known, spell_letter, spell_memory."""
    state = _fresh_state()
    mg = state.magic
    new_mg = mg.replace(
        spell_memory=mg.spell_memory.at[0].set(jnp.int32(20000)),
        spell_known=mg.spell_known.at[0].set(jnp.bool_(True)),
        spell_letter=mg.spell_letter.at[0].set(jnp.int8(ord("a"))),
        pw_regen_counter=jnp.int32(42),
    )
    state = state.replace(magic=new_mg)

    loaded = _round_trip(state, tmp_path, "magic")
    _assert_pytree_equal(state.magic, loaded.magic)


# ---------------------------------------------------------------------------
# Full state after running steps + version mismatch
# ---------------------------------------------------------------------------

def test_save_load_full_state_after_100_steps(tmp_path: Path) -> None:
    """Run 100 steps with varied actions, save, load, and verify every leaf."""
    env = NethaxEnv()
    rng = jax.random.PRNGKey(2026)
    state, _ = env.reset(rng)

    # Mix of wait (.) and a directional step (north 'k').  Avoids non-
    # deterministic action paths that might involve I/O prompts.
    actions = [ord("."), ord("k"), ord("."), ord("l"), ord(".")]
    for i in range(100):
        rng, step_rng = jax.random.split(rng)
        action = jnp.int32(actions[i % len(actions)])
        state, _, _, _, _ = env.step(state, action, step_rng)

    loaded = _round_trip(state, tmp_path, "full_100_steps")
    _assert_pytree_equal(state, loaded)


def test_save_load_version_mismatch_raises(tmp_path: Path) -> None:
    """A save file with an unrecognized _version must raise IncompatibleSaveError."""
    state = _fresh_state()
    leaves, treedef = jax.tree_util.tree_flatten(state)
    np_leaves = {f"leaf_{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    treedef_bytes = pickle.dumps(treedef)

    bad_path = tmp_path / "bad_version.npz"
    np.savez_compressed(
        str(bad_path),
        _version=np.int32(_NETHAX_SAVE_VERSION + 100),
        treedef_str=np.frombuffer(treedef_bytes, dtype=np.uint8),
        **np_leaves,
    )

    with pytest.raises(IncompatibleSaveError):
        load_state(bad_path)
