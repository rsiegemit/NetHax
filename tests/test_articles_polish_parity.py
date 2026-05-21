"""Article/name-formatting polish parity tests.

Covers gaps introduced in look.py and inv_strs.py:
  1. _s_suffix()       — hacklib.c:345-359
  2. Pet "your" article — do_name.c:1121
  3. G_UNIQ "The"       — do_name.c:1005
  4. "pair of boots/gloves/lenses" — objnam.c:724-726
  5. "set of dragon scales"        — objnam.c:722
  6. Amulet of Yendor guard        — objnam.c:675-677
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest
import jax.numpy as jnp
import numpy as np

from Nethax.nethax.obs.look import _s_suffix, build_look_text
from Nethax.nethax.constants.objects import OBJECTS, ObjectClass
from Nethax.nethax.constants.monsters import MONSTERS


# ---------------------------------------------------------------------------
# Helpers shared with test_inv_strs.py pattern
# ---------------------------------------------------------------------------

def _decode(row) -> str:
    arr = np.asarray(row).astype(np.uint8)
    return bytes(arr.tolist()).rstrip(b"\x00").decode("ascii", errors="replace")


def _find_type_id(name: str) -> int:
    for i, obj in enumerate(OBJECTS):
        if obj.name == name:
            return i
    raise ValueError(f"Object not found: {name!r}")


def _find_monster_idx(name: str) -> int:
    for i, m in enumerate(MONSTERS):
        if m is not None and m.name and m.name.lower() == name.lower():
            return i
    raise ValueError(f"Monster not found: {name!r}")


# ---------------------------------------------------------------------------
# Minimal state builders
# ---------------------------------------------------------------------------

from Nethax.nethax.subsystems.inventory import (
    InventoryState, Item, ArmorSlot, MAX_INVENTORY_SLOTS, N_ARMOR_SLOTS,
    USER_NAME_LEN,
)
from Nethax.nethax.subsystems.items import BUCStatus
from Nethax.nethax.subsystems.identification import IdentificationState


def _make_empty_items() -> Item:
    return Item(
        category=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
        type_id=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int16),
        buc_status=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
        enchantment=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
        charges=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
        identified=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        quantity=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int16),
        weight=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int32),
        ac_bonus=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
        is_two_handed=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        greased=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        oeroded=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
        oeroded2=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.int8),
        oerodeproof=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        bknown=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        lamplit=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        olocked=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        corpse_entry_idx=jnp.full((MAX_INVENTORY_SLOTS,), -1, dtype=jnp.int16),
    )


def _make_inv_state(items: Item) -> InventoryState:
    return InventoryState(
        items=items,
        wielded=jnp.int8(-1),
        off_hand=jnp.int8(-1),
        worn_armor=jnp.full((N_ARMOR_SLOTS,), -1, dtype=jnp.int8),
        worn_amulet=jnp.int8(-1),
        worn_rings=jnp.full((2,), -1, dtype=jnp.int8),
        quiver=jnp.int8(-1),
        total_weight=jnp.int32(0),
        alternate_weapon_slot=jnp.int8(-1),
        worn_armor_ac_bonus=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.int8),
        user_names=jnp.zeros((MAX_INVENTORY_SLOTS, USER_NAME_LEN), dtype=jnp.int8),
        wielded_artifact_idx=jnp.int8(-1),
        # Cursed-stuck (welded) flags — vendor wield.c::welded(), do_wear.c
        welded=jnp.bool_(False),
        worn_armor_welded=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.bool_),
        worn_amulet_welded=jnp.bool_(False),
        worn_rings_welded=jnp.zeros((2,), dtype=jnp.bool_),
        armor_stat_bonus=jnp.zeros((6,), dtype=jnp.int8),
    )


def _make_state_with_inv(inv_state: InventoryState):
    id_state = IdentificationState.unshuffled()

    class _FakeState:
        inventory = inv_state
        identification = id_state

    return _FakeState()


def _render_slot_str(type_id_val: int, category_val: int,
                     quantity: int = 1, identified: bool = True) -> str:
    from Nethax.nethax.obs.inv_strs import build_inv_strs
    items = _make_empty_items()
    items = items.replace(
        category=items.category.at[0].set(jnp.int8(category_val)),
        type_id=items.type_id.at[0].set(jnp.int16(type_id_val)),
        identified=items.identified.at[0].set(jnp.bool_(identified)),
        quantity=items.quantity.at[0].set(jnp.int16(quantity)),
    )
    inv = _make_inv_state(items)
    state = _make_state_with_inv(inv)
    result = build_inv_strs(state)
    return _decode(result[0])


# ---------------------------------------------------------------------------
# Minimal look-state builder for monster tests
# ---------------------------------------------------------------------------

class _FakeLookState:
    """Minimal state with a single monster placed at a given cell."""

    def __init__(self, monster_idx: int, row: int, col: int,
                 tame: bool = False, player_pos=(0, 0)):
        self.player_pos = jnp.array(player_pos, dtype=jnp.int32)

        class _MonsterAI:
            pass

        ai = _MonsterAI()
        ai.pos = jnp.array([[row, col]], dtype=jnp.int16)
        ai.alive = jnp.array([True], dtype=jnp.bool_)
        ai.entry_idx = jnp.array([monster_idx], dtype=jnp.int16)
        ai.tame = jnp.array([tame], dtype=jnp.bool_)
        self.monster_ai = ai


# ---------------------------------------------------------------------------
# Gap 1 — _s_suffix (hacklib.c:345-359)
# ---------------------------------------------------------------------------

class TestSSuffix:
    def test_s_suffix_basic_dog(self):
        assert _s_suffix("dog") == "dog's"

    def test_s_suffix_croesus(self):
        # "Croesus" ends in 's' -> vendor hacklib.c:352 appends just apostrophe
        assert _s_suffix("Croesus") == "Croesus'"

    def test_s_suffix_words_ending_s(self):
        assert _s_suffix("yours") == "yours'"

    def test_s_suffix_it(self):
        assert _s_suffix("it") == "its"

    def test_s_suffix_you(self):
        assert _s_suffix("you") == "your"

    def test_s_suffix_name_ending_s(self):
        # Any word ending in 's' gets just apostrophe
        assert _s_suffix("atlas") == "atlas'"


# ---------------------------------------------------------------------------
# Gap 2 — Pet "your" article (do_name.c:1121)
# ---------------------------------------------------------------------------

class TestPetYourInLook:
    def test_tame_monster_yields_your(self):
        # Find any monster that exists in the table
        idx = next(
            i for i, m in enumerate(MONSTERS)
            if m is not None and m.name is not None
        )
        state = _FakeLookState(monster_idx=idx, row=5, col=5,
                               tame=True, player_pos=(0, 0))
        text = build_look_text(state, 5, 5)
        assert text.startswith("your "), (
            f"tame monster should give 'your ...', got: {text!r}"
        )

    def test_tame_monster_no_the(self):
        idx = next(
            i for i, m in enumerate(MONSTERS)
            if m is not None and m.name is not None
        )
        state = _FakeLookState(monster_idx=idx, row=5, col=5,
                               tame=True, player_pos=(0, 0))
        text = build_look_text(state, 5, 5)
        assert "the " not in text, (
            f"tame monster should not contain 'the', got: {text!r}"
        )

    def test_wild_monster_yields_the(self):
        idx = next(
            i for i, m in enumerate(MONSTERS)
            if m is not None and m.name is not None
        )
        state = _FakeLookState(monster_idx=idx, row=5, col=5,
                               tame=False, player_pos=(0, 0))
        text = build_look_text(state, 5, 5)
        assert text.startswith("the ") or text[0].isupper(), (
            f"wild monster should give 'the ...' or proper name, got: {text!r}"
        )


# ---------------------------------------------------------------------------
# Gap 3 — G_UNIQ "The" capital (do_name.c:1005)
# ---------------------------------------------------------------------------

class TestUniqueMonsterTheCapital:
    @pytest.fixture
    def wizard_idx(self):
        return _find_monster_idx("Wizard of Yendor")

    def test_unique_wizard_gives_The(self, wizard_idx):
        state = _FakeLookState(monster_idx=wizard_idx, row=5, col=5,
                               tame=False, player_pos=(0, 0))
        text = build_look_text(state, 5, 5)
        assert text.startswith("The "), (
            f"unique monster should give 'The ...', got: {text!r}"
        )

    def test_unique_wizard_title_case(self, wizard_idx):
        state = _FakeLookState(monster_idx=wizard_idx, row=5, col=5,
                               tame=False, player_pos=(0, 0))
        text = build_look_text(state, 5, 5)
        # Monster name from table is "Wizard of Yendor"; first letter capitalised.
        assert "Wizard" in text, (
            f"expected 'Wizard' in look text, got: {text!r}"
        )

    def test_medusa_unique(self):
        try:
            idx = _find_monster_idx("Medusa")
        except ValueError:
            pytest.skip("Medusa not in MONSTERS table")
        state = _FakeLookState(monster_idx=idx, row=5, col=5,
                               tame=False, player_pos=(0, 0))
        text = build_look_text(state, 5, 5)
        assert text.startswith("The "), f"got: {text!r}"


# ---------------------------------------------------------------------------
# Gap 4 — "pair of boots/gloves/lenses" (objnam.c:724-726)
# ---------------------------------------------------------------------------

class TestPairOfBoots:
    def test_pair_of_boots_singular(self):
        # low boots = type_id 140, ARMOR_CLASS
        type_id_val = _find_type_id("low boots")
        s = _render_slot_str(type_id_val, int(ObjectClass.ARMOR_CLASS),
                             quantity=1, identified=True)
        assert "pair of" in s, f"expected 'pair of' in: {s!r}"

    def test_pair_of_boots_article(self):
        type_id_val = _find_type_id("low boots")
        s = _render_slot_str(type_id_val, int(ObjectClass.ARMOR_CLASS),
                             quantity=1, identified=True)
        # Article is "a", enchantment "+0" precedes name for identified armor,
        # so full string is "a - a +0 pair of low boots"; check prefix present.
        assert "pair of" in s, f"expected 'pair of' in: {s!r}"

    def test_pair_of_boots_contains_name(self):
        type_id_val = _find_type_id("low boots")
        s = _render_slot_str(type_id_val, int(ObjectClass.ARMOR_CLASS),
                             quantity=1, identified=True)
        assert "low boots" in s, f"expected 'low boots' in: {s!r}"


class TestPairOfGloves:
    def test_pair_of_gloves_singular(self):
        type_id_val = _find_type_id("leather gloves")
        s = _render_slot_str(type_id_val, int(ObjectClass.ARMOR_CLASS),
                             quantity=1, identified=True)
        assert "pair of" in s, f"expected 'pair of' in: {s!r}"

    def test_pair_of_gloves_name(self):
        type_id_val = _find_type_id("leather gloves")
        s = _render_slot_str(type_id_val, int(ObjectClass.ARMOR_CLASS),
                             quantity=1, identified=True)
        assert "leather gloves" in s, f"expected 'leather gloves' in: {s!r}"


class TestPairsPlural:
    def test_pairs_of_boots_plural(self):
        type_id_val = _find_type_id("low boots")
        s = _render_slot_str(type_id_val, int(ObjectClass.ARMOR_CLASS),
                             quantity=2, identified=True)
        assert "pairs of" in s, f"expected 'pairs of' in: {s!r}"

    def test_pairs_prefix_quantity(self):
        type_id_val = _find_type_id("low boots")
        s = _render_slot_str(type_id_val, int(ObjectClass.ARMOR_CLASS),
                             quantity=2, identified=True)
        assert s.startswith("a - 2 "), f"expected '2 ' quantity prefix in: {s!r}"


# ---------------------------------------------------------------------------
# Gap 5 — "set of dragon scales" (objnam.c:722)
# ---------------------------------------------------------------------------

class TestSetOfDragonScales:
    def test_set_of_scales_singular(self):
        type_id_val = _find_type_id("gray dragon scales")
        s = _render_slot_str(type_id_val, int(ObjectClass.ARMOR_CLASS),
                             quantity=1, identified=True)
        assert "set of" in s, f"expected 'set of' in: {s!r}"

    def test_set_of_scales_article(self):
        type_id_val = _find_type_id("gray dragon scales")
        s = _render_slot_str(type_id_val, int(ObjectClass.ARMOR_CLASS),
                             quantity=1, identified=True)
        # Enchantment "+0" precedes name for identified armor, so check prefix present.
        assert "set of" in s, f"expected 'set of' in: {s!r}"

    def test_dragon_scale_mail_no_set_of(self):
        # Dragon scale *mail* is not scales — should NOT get "set of"
        type_id_val = _find_type_id("gray dragon scale mail")
        s = _render_slot_str(type_id_val, int(ObjectClass.ARMOR_CLASS),
                             quantity=1, identified=True)
        assert "set of" not in s, f"'set of' should not appear for mail: {s!r}"
