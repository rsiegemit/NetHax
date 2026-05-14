"""Wave 6 Phase B+ — OBJECT material + MONSTER msound parity tests.

Verifies:
  - OBJECTS[type_id].material is populated from vendor's oc_material column.
  - food_material_for_type_id reads OBJECTS.material directly (no heuristics).
  - is_meat_material / is_animal_material match vendor src/eat.c VEGAN/VEGETARIAN
    classification (FLESH = meat+animal; WAX = dairy/animal product).
  - MonsterEntry.sound (msound) is populated and monster_ai._is_mage_entry uses
    MS_SPELL / MS_PRIEST instead of an entry_idx range heuristic.

Cite:
  vendor/nethack/include/objclass.h::obj_material_types
  vendor/nethack/src/eat.c::eatcorpse / eatfood (material-driven conduct)
  vendor/nethack/include/monflag.h::MS_SPELL / MS_PRIEST
  vendor/nethack/src/mcastu.c::castmu (sound-gate)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants.objects import OBJECTS, ObjectClass, Material
from Nethax.nethax.constants.monsters import MONSTERS, MS_SPELL, MS_PRIEST
from Nethax.nethax.subsystems.conduct import (
    Conduct,
    food_material_for_type_id,
    is_meat_material,
    is_animal_material,
)


_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_object_idx(name: str, class_: ObjectClass = None) -> int:
    """Return first OBJECTS index whose .name == name (and optional class)."""
    for i, o in enumerate(OBJECTS):
        if o.name == name and (class_ is None or int(o.class_) == int(class_)):
            return i
    raise ValueError(f"No OBJECTS entry named {name!r}")


def _state_with_food(type_id: int) -> EnvState:
    state = EnvState.default(_RNG)
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(ObjectClass.FOOD_CLASS)),
        type_id=items.type_id.at[0].set(jnp.int16(type_id)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=items.weight.at[0].set(jnp.int32(0)),
    )
    return state.replace(inventory=state.inventory.replace(items=new_items))


# ---------------------------------------------------------------------------
# OBJECTS material-field parity
# ---------------------------------------------------------------------------

def test_potion_material_liquid():
    """Potions store oc_material = LIQUID (Wave 6 Phase B spec deviation —
    canonical contents-material, not the GLASS-container material vendor
    objects.h uses; aligns with obj.h::MAT_LIQUID for conduct purposes)."""
    idx = _find_object_idx("healing", ObjectClass.POTION_CLASS)
    material = food_material_for_type_id(idx)
    assert int(material) == int(Material.LIQUID), (
        f"potion of healing material expected LIQUID(1), got {int(material)}"
    )


def test_food_corpse_material_flesh():
    """Vendor objects.h: FOOD("corpse", ..., FLESH, ...)."""
    idx = _find_object_idx("corpse", ObjectClass.FOOD_CLASS)
    assert int(OBJECTS[idx].material) == int(Material.FLESH)
    assert int(food_material_for_type_id(idx)) == int(Material.FLESH)


def test_food_apple_material_vegy():
    """Vendor objects.h: FOOD("apple", ..., VEGGY, ...)."""
    idx = _find_object_idx("apple", ObjectClass.FOOD_CLASS)
    assert int(OBJECTS[idx].material) == int(Material.VEGGY)
    assert int(food_material_for_type_id(idx)) == int(Material.VEGGY)


def test_armor_long_sword_material_iron():
    """Vendor objects.h: WEAPON("long sword", ..., IRON, ...)."""
    idx = _find_object_idx("long sword", ObjectClass.WEAPON_CLASS)
    assert int(OBJECTS[idx].material) == int(Material.IRON)


def test_armor_mithril_coat_material_mithril():
    """Vendor objects.h: ARMOR("dwarvish mithril-coat", ..., MITHRIL, ...)."""
    idx = _find_object_idx("dwarvish mithril-coat", ObjectClass.ARMOR_CLASS)
    assert int(OBJECTS[idx].material) == int(Material.MITHRIL)


def test_food_tripe_ration_material_flesh():
    """Vendor objects.h: FOOD("tripe ration", ..., FLESH, ...)."""
    idx = _find_object_idx("tripe ration", ObjectClass.FOOD_CLASS)
    assert int(OBJECTS[idx].material) == int(Material.FLESH)


def test_food_egg_material_flesh():
    """Vendor objects.h: FOOD("egg", ..., FLESH, ...).  Vendor flags egg as
    FLESH (not WAX); the egg-as-dairy distinction is handled by oc_tough not
    by material, so VEGAN/VEGETARIAN both flip on egg eats."""
    idx = _find_object_idx("egg", ObjectClass.FOOD_CLASS)
    assert int(OBJECTS[idx].material) == int(Material.FLESH)


def test_tallow_candle_material_wax():
    """Vendor objects.h: TOOL("tallow candle", ..., WAX, ...) — provides our
    dairy/animal-product material exemplar for is_animal_material."""
    idx = _find_object_idx("tallow candle", ObjectClass.TOOL_CLASS)
    assert int(OBJECTS[idx].material) == int(Material.WAX)


# ---------------------------------------------------------------------------
# Conduct predicates — material-driven VEGAN / VEGETARIAN
# ---------------------------------------------------------------------------

def test_eating_meat_violates_vegetarian_via_material():
    """Eating FLESH food flips VEGETARIAN (vendor src/eat.c)."""
    from Nethax.nethax.subsystems.action_dispatch import _handle_eat
    idx = _find_object_idx("meatball", ObjectClass.FOOD_CLASS)
    state = _state_with_food(type_id=idx)
    new_state = _handle_eat(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.VEGETARIAN)]) is True
    assert bool(new_state.conduct.violations[int(Conduct.VEGAN)]) is True


def test_eating_apple_does_not_violate_vegan():
    """Eating VEGGY food (apple) leaves VEGAN intact."""
    from Nethax.nethax.subsystems.action_dispatch import _handle_eat
    idx = _find_object_idx("apple", ObjectClass.FOOD_CLASS)
    state = _state_with_food(type_id=idx)
    new_state = _handle_eat(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.FOODLESS)]) is True
    assert bool(new_state.conduct.violations[int(Conduct.VEGAN)]) is False
    assert bool(new_state.conduct.violations[int(Conduct.VEGETARIAN)]) is False


def test_eating_cheese_violates_vegan_via_wax_material():
    """WAX material is dairy-equivalent: is_animal_material returns True on
    MAT_WAX, so a hypothetical cheese-like (WAX) food would flip VEGAN.

    We test the predicate directly since vendor has no native cheese FOOD."""
    wax_val = jnp.int32(int(Material.WAX))
    assert bool(is_animal_material(wax_val)) is True
    assert bool(is_meat_material(wax_val)) is False
    # And a real WAX object (tallow candle) routes through the same predicate.
    candle_idx = _find_object_idx("tallow candle", ObjectClass.TOOL_CLASS)
    mat = food_material_for_type_id(candle_idx)
    assert int(mat) == int(Material.WAX)
    assert bool(is_animal_material(mat)) is True


def test_is_meat_material_only_flesh():
    """is_meat_material is True only for FLESH; WAX / VEGGY / LIQUID are False."""
    assert bool(is_meat_material(jnp.int32(int(Material.FLESH)))) is True
    assert bool(is_meat_material(jnp.int32(int(Material.WAX)))) is False
    assert bool(is_meat_material(jnp.int32(int(Material.VEGGY)))) is False
    assert bool(is_meat_material(jnp.int32(int(Material.LIQUID)))) is False


def test_is_animal_material_flesh_and_wax():
    """is_animal_material returns True for FLESH and WAX (vendor VEGAN flag)."""
    assert bool(is_animal_material(jnp.int32(int(Material.FLESH)))) is True
    assert bool(is_animal_material(jnp.int32(int(Material.WAX)))) is True
    assert bool(is_animal_material(jnp.int32(int(Material.VEGGY)))) is False
    assert bool(is_animal_material(jnp.int32(int(Material.IRON)))) is False


def test_food_material_table_covers_all_objects():
    """Sanity: the OBJECTS material lookup table is the same length as OBJECTS,
    so type_id lookups don't silently clamp into an unrelated entry."""
    from Nethax.nethax.subsystems.conduct import _OBJECT_MATERIAL_TABLE
    assert int(_OBJECT_MATERIAL_TABLE.shape[0]) == len(OBJECTS)


# ---------------------------------------------------------------------------
# MonsterEntry.sound (msound) parity
# ---------------------------------------------------------------------------

def _find_monster_idx(name: str) -> int:
    for i, m in enumerate(MONSTERS):
        if m.name == name:
            return i
    raise ValueError(f"No monster named {name!r}")


def test_monster_mage_class_via_msound_spell():
    """A monster with sound == MS_SPELL is detected as mage-class."""
    from Nethax.nethax.subsystems.monster_ai import _is_mage_entry
    idx = _find_monster_idx("titan")
    assert int(MONSTERS[idx].sound) == MS_SPELL
    assert bool(_is_mage_entry(jnp.int32(idx))) is True


def test_monster_priest_class_via_msound_priest():
    """A monster with sound == MS_PRIEST is detected as mage-class."""
    from Nethax.nethax.subsystems.monster_ai import _is_mage_entry
    idx = _find_monster_idx("high priest")
    assert int(MONSTERS[idx].sound) == MS_PRIEST
    assert bool(_is_mage_entry(jnp.int32(idx))) is True


def test_non_mage_monster_does_not_cast_spell():
    """A non-mage monster (e.g. kitten, MS_MEW) returns False from _is_mage_entry."""
    from Nethax.nethax.subsystems.monster_ai import _is_mage_entry
    idx = _find_monster_idx("kitten")
    # kitten uses MS_MEW (2), not MS_SPELL/PRIEST
    assert int(MONSTERS[idx].sound) not in (MS_SPELL, MS_PRIEST)
    assert bool(_is_mage_entry(jnp.int32(idx))) is False


def test_non_mage_monster_does_not_cast_spell_dispatch():
    """monster_cast_spell on a non-mage monster is a no-op even when LoS &
    range conditions are satisfied (vendor castmu sound-gate)."""
    from Nethax.nethax.subsystems.monster_ai import monster_cast_spell

    idx = _find_monster_idx("kitten")
    state = EnvState.default(_RNG)
    state = state.replace(player_hp=jnp.int32(100), player_hp_max=jnp.int32(100))
    mai = state.monster_ai
    state = state.replace(monster_ai=mai.replace(
        alive=mai.alive.at[0].set(jnp.bool_(True)),
        peaceful=mai.peaceful.at[0].set(jnp.bool_(False)),
        asleep=mai.asleep.at[0].set(jnp.bool_(False)),
        hp=mai.hp.at[0].set(jnp.int32(20)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(20)),
        entry_idx=mai.entry_idx.at[0].set(jnp.int16(idx)),
        pos=mai.pos.at[0].set(jnp.array([5, 5], dtype=jnp.int16)),
    ))
    state = state.replace(player_pos=jnp.array([5, 6], dtype=jnp.int16))

    hp_before = int(state.player_hp)
    new_state = monster_cast_spell(state, _RNG, jnp.int32(0))
    hp_after = int(new_state.player_hp)
    # Non-mage: no damage applied.
    assert hp_after == hp_before, (
        f"kitten should not cast spells; hp changed {hp_before} -> {hp_after}"
    )


def test_monster_sound_field_populated_for_all_entries():
    """All MONSTERS entries have a valid sound (int8) field — Wave 6 parity."""
    for i, m in enumerate(MONSTERS):
        s = int(m.sound)
        # Sound is in [0, ~50) per vendor monflag.h.
        assert 0 <= s <= 64, f"MONSTERS[{i}].sound = {s} out of range"


def test_monster_flags1_flags2_flags3_populated():
    """All MONSTERS entries have flags1/2/3 fields populated."""
    for i, m in enumerate(MONSTERS):
        # Flags exist as ints (may be zero for some entries — that's fine).
        assert isinstance(m.flags1, int)
        assert isinstance(m.flags2, int)
        assert isinstance(m.flags3, int)
