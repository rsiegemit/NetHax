"""Wave 6 #77 closing-audit — identification subsystem vendor parity.

Asserts that ``init_shuffled_appearances`` mirrors
vendor/nethack/src/o_init.c::shuffle_all (lines 322-347) which Fisher-Yates
shuffles description indices across each shufflable class.

Covered tests:
  - test_scroll_appearances_shuffled_deterministic_per_seed
  - test_potion_colors_shuffled
  - test_wand_materials_shuffled
  - test_ring_stones_shuffled
  - test_amulet_shapes_shuffled
  - test_spellbook_covers_shuffled
  - test_identical_seed_produces_identical_mapping
  - test_identify_marks_correct_object_per_appearance
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from Nethax.nethax.subsystems.identification import (
    IdentificationState,
    N_AMULET_TYPES,
    N_POTION_TYPES,
    N_RING_TYPES,
    N_SCROLL_TYPES,
    N_SPELLBOOK_TYPES,
    N_WAND_TYPES,
    POT_WATER_INDEX,
    check_known,
    full_identify,
    identified_name,
    init_shuffled_appearances,
    type_for_appearance,
    unidentified_appearance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_permutation(arr: jnp.ndarray, n: int) -> bool:
    """True iff arr is a permutation of [0..n)."""
    arr_np = np.asarray(arr).astype(np.int32)
    return arr_np.shape == (n,) and sorted(arr_np.tolist()) == list(range(n))


def _shuffled_from(arr: jnp.ndarray, n: int) -> bool:
    """True iff arr differs from identity permutation (vendor: shuffled)."""
    arr_np = np.asarray(arr).astype(np.int32)
    return not np.array_equal(arr_np, np.arange(n))


# ---------------------------------------------------------------------------
# Per-class shuffle is a valid permutation and not identity
# ---------------------------------------------------------------------------

class TestPerClassShuffle:
    """Each class array must be a permutation of [0..N_X) and differ from id."""

    @pytest.fixture
    def state(self) -> IdentificationState:
        rng = jax.random.PRNGKey(2024)
        return init_shuffled_appearances(rng)

    def test_scroll_appearances_shuffled_deterministic_per_seed(self, state):
        # Permutation property.
        assert _is_permutation(state.scroll_appearance, N_SCROLL_TYPES)
        # Probability of identity for 43 elements is 1/43! ≈ 0 — must differ.
        assert _shuffled_from(state.scroll_appearance, N_SCROLL_TYPES)

        # Deterministic for same key.
        rng = jax.random.PRNGKey(2024)
        state2 = init_shuffled_appearances(rng)
        assert np.array_equal(
            np.asarray(state.scroll_appearance),
            np.asarray(state2.scroll_appearance),
        )

    def test_potion_colors_shuffled(self, state):
        # POT_WATER (last index) has fixed description per
        # vendor obj_shuffle_range (o_init.c lines 289-293).
        assert int(state.potion_appearance[POT_WATER_INDEX]) == POT_WATER_INDEX
        # First N-1 slots must be a permutation of [0..N-1).
        front = np.asarray(state.potion_appearance[:POT_WATER_INDEX])
        assert sorted(front.tolist()) == list(range(POT_WATER_INDEX))
        # And actually shuffled (not identity).
        assert not np.array_equal(front, np.arange(POT_WATER_INDEX))

    def test_wand_materials_shuffled(self, state):
        assert _is_permutation(state.wand_appearance, N_WAND_TYPES)
        assert _shuffled_from(state.wand_appearance, N_WAND_TYPES)

    def test_ring_stones_shuffled(self, state):
        assert _is_permutation(state.ring_appearance, N_RING_TYPES)
        assert _shuffled_from(state.ring_appearance, N_RING_TYPES)

    def test_amulet_shapes_shuffled(self, state):
        assert _is_permutation(state.amulet_appearance, N_AMULET_TYPES)
        # 13 items — chance of identity is 1/13! ≈ 1.6e-10, safe to assert.
        assert _shuffled_from(state.amulet_appearance, N_AMULET_TYPES)

    def test_spellbook_covers_shuffled(self, state):
        assert _is_permutation(state.spellbook_appearance, N_SPELLBOOK_TYPES)
        assert _shuffled_from(state.spellbook_appearance, N_SPELLBOOK_TYPES)


# ---------------------------------------------------------------------------
# Seed-determinism: same seed → same mapping
# ---------------------------------------------------------------------------

class TestSeedDeterminism:
    def test_identical_seed_produces_identical_mapping(self):
        """Running shuffle twice with same key must yield identical perms."""
        rng = jax.random.PRNGKey(0xC0FFEE)
        a = init_shuffled_appearances(rng)
        b = init_shuffled_appearances(rng)
        for field in (
            "potion_appearance",
            "scroll_appearance",
            "wand_appearance",
            "ring_appearance",
            "amulet_appearance",
            "spellbook_appearance",
        ):
            assert np.array_equal(
                np.asarray(getattr(a, field)),
                np.asarray(getattr(b, field)),
            ), f"mismatch on field {field}"

    def test_different_seeds_produce_different_mapping(self):
        """Sanity: distinct seeds should (almost certainly) produce distinct
        permutations on the largest class."""
        a = init_shuffled_appearances(jax.random.PRNGKey(1))
        b = init_shuffled_appearances(jax.random.PRNGKey(2))
        assert not np.array_equal(
            np.asarray(a.scroll_appearance),
            np.asarray(b.scroll_appearance),
        )


# ---------------------------------------------------------------------------
# Identify marks the right object
# ---------------------------------------------------------------------------

class TestIdentifyCorrectness:
    def test_identify_marks_correct_object_per_appearance(self):
        """After full_identify(type_id), check_known(type_id) is True and
        type_for_appearance(state, class, app[type_id]) round-trips to type_id.
        """
        rng = jax.random.PRNGKey(7)
        state = init_shuffled_appearances(rng)
        # Pick an arbitrary scroll type to identify.
        target_type = 5
        appearance_idx = int(state.scroll_appearance[target_type])

        # Before ID: check_known is False.
        assert bool(check_known(state, target_type)) is False

        # full_identify flips the bit.
        state = full_identify(state, target_type)
        assert bool(check_known(state, target_type)) is True

        # Inverse lookup must round-trip.
        inv = type_for_appearance(state, "scroll", appearance_idx)
        assert inv == target_type

        # Other types untouched.
        assert bool(check_known(state, target_type + 1)) is False

    def test_unidentified_appearance_uses_perm(self):
        """unidentified_appearance(class, type_id) returns the pool entry at
        the *shuffled* index, not the canonical one."""
        rng = jax.random.PRNGKey(11)
        state = init_shuffled_appearances(rng)
        # Find a wand type whose appearance index differs from type id.
        wand_perm = np.asarray(state.wand_appearance)
        differing = [i for i in range(N_WAND_TYPES) if wand_perm[i] != i]
        assert differing, "wand permutation should not be identity"
        t = differing[0]
        seen = unidentified_appearance(state, "wand", t)
        true = identified_name("wand", t)
        assert seen != true

    def test_unidentified_appearance_invalid_class_raises(self):
        state = init_shuffled_appearances(jax.random.PRNGKey(0))
        with pytest.raises(ValueError):
            unidentified_appearance(state, "shield", 0)
