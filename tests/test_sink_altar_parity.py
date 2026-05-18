"""Parity tests for sink + altar effects.

Covers:
    sit_sink      — sit.c::dosit IS_SINK branch (6-outcome table)
    kick_sink     — dokick.c::kick_nondoor IS_SINK branch (4-outcome table)
    drop_at_altar — pray.c::doaltar BUC mutation on drop

Citations:
    vendor/nethack/src/sit.c::dosit lines 526-529
    vendor/nethack/src/dokick.c::kick_nondoor lines 1194-1240
    vendor/nethack/src/pray.c::doaltar (coaligned/cross-aligned bless/curse)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.features import sit_sink, kick_sink, drop_at_altar
from Nethax.nethax.subsystems.inventory import ItemCategory, MAX_INVENTORY_SLOTS

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(player_row: int = 5, player_col: int = 5) -> EnvState:
    state = EnvState.default(_RNG)
    return state.replace(
        player_pos=jnp.array([player_row, player_col], dtype=jnp.int16),
        player_hp=jnp.int32(50),
        player_hp_max=jnp.int32(50),
    )


def _with_item(state: EnvState, slot: int, category: int, buc: int = 2) -> EnvState:
    """Place a single uncursed item of *category* into *slot*."""
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[slot].set(jnp.int8(category)),
        buc_status=items.buc_status.at[slot].set(jnp.int8(buc)),
        quantity=items.quantity.at[slot].set(jnp.int16(1)),
    )
    return state.replace(inventory=state.inventory.replace(items=new_items))


def _set_altar(state: EnvState, alignment: int) -> EnvState:
    """Place an altar of *alignment* at the player's current tile."""
    max_lv = int(state.terrain.shape[1])
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    flat_lv = b * max_lv + lv
    row = int(state.player_pos[0])
    col = int(state.player_pos[1])
    new_altar = state.features.altar_alignment.at[flat_lv, row, col].set(jnp.int8(alignment))
    return state.replace(features=state.features.replace(altar_alignment=new_altar))


# ---------------------------------------------------------------------------
# kick_sink tests
# ---------------------------------------------------------------------------

class TestKickSink:
    def test_kick_sink_strange_shock(self):
        """Kicking a sink sometimes deals HP damage (shock/pudding outcomes).

        Cite: vendor/nethack/src/dokick.c::kick_nondoor lines 1194-1240.
        Over 80 rngs, at least one must reduce HP.
        """
        state = _make_state()
        any_damage = False
        for i in range(80):
            rng_i = jax.random.PRNGKey(100 + i)
            new_state, _ = kick_sink(state, rng_i)
            if int(new_state.player_hp) < int(state.player_hp):
                any_damage = True
                break
        assert any_damage, "Expected at least one HP-reducing outcome in 80 kick_sink trials"

    def test_kick_sink_jit_safe(self):
        """kick_sink must be jit-compilable."""
        state = _make_state()
        fn = jax.jit(kick_sink)
        new_state, outcome = fn(state, _RNG)
        assert new_state is not None
        assert outcome.shape == ()

    def test_kick_sink_returns_outcome_id(self):
        """kick_sink returns a scalar int32 outcome id in [0, 3]."""
        state = _make_state()
        _, outcome = kick_sink(state, _RNG)
        assert int(outcome) in range(4)

    def test_kick_sink_no_negative_hp(self):
        """HP must never go below 0 after kick_sink."""
        state = _make_state()
        state = state.replace(player_hp=jnp.int32(1))
        for i in range(20):
            rng_i = jax.random.PRNGKey(200 + i)
            new_state, _ = kick_sink(state, rng_i)
            assert int(new_state.player_hp) >= 0


# ---------------------------------------------------------------------------
# sit_sink tests
# ---------------------------------------------------------------------------

class TestSitSink:
    def test_sit_sink_random_outcome(self):
        """Over 60 rngs, sit_sink produces varied outcomes (not all identical).

        Cite: vendor/nethack/src/sit.c::dosit IS_SINK branch.
        """
        state = _make_state()
        hp_values = set()
        nutrition_values = set()
        for i in range(60):
            rng_i = jax.random.PRNGKey(300 + i)
            new_state = sit_sink(state, rng_i)
            hp_values.add(int(new_state.player_hp))
            nutrition_values.add(int(new_state.status.nutrition))
        # Expect multiple distinct outcomes across 60 seeds
        all_distinct = len(hp_values) + len(nutrition_values)
        assert all_distinct >= 3, (
            f"Expected varied outcomes; got hp_values={hp_values}, "
            f"nutrition_values={nutrition_values}"
        )

    def test_sit_sink_curse_worn_item(self):
        """sit_sink can curse the first worn item (buc → 1).

        Cite: sit.c rndcurse() proxy in sit_sink._curse_worn.
        """
        state = _make_state()
        state = _with_item(state, slot=0, category=int(ItemCategory.WEAPON), buc=2)
        any_cursed = False
        for i in range(60):
            rng_i = jax.random.PRNGKey(400 + i)
            new_state = sit_sink(state, rng_i)
            if int(new_state.inventory.items.buc_status[0]) == 1:
                any_cursed = True
                break
        assert any_cursed, "Expected at least one curse outcome in 60 sit_sink trials"

    def test_sit_sink_jit_safe(self):
        """sit_sink must be jit-compilable."""
        state = _make_state()
        fn = jax.jit(sit_sink)
        out = fn(state, _RNG)
        assert out is not None

    def test_sit_sink_no_negative_hp(self):
        """HP must never go below 0 after sit_sink."""
        state = _make_state()
        state = state.replace(player_hp=jnp.int32(1))
        for i in range(20):
            rng_i = jax.random.PRNGKey(500 + i)
            new_state = sit_sink(state, rng_i)
            assert int(new_state.player_hp) >= 0


# ---------------------------------------------------------------------------
# drop_at_altar tests
# ---------------------------------------------------------------------------

class TestDropAtAltar:
    def test_drop_on_coaligned_altar_blesses(self):
        """Drop uncursed item on coaligned altar → item becomes blessed.

        Cite: vendor/nethack/src/pray.c::doaltar — coaligned altar blesses.
        """
        state = _make_state()
        # player_align=2 (lawful), altar_alignment=2 (lawful) → coaligned
        state = state.replace(player_align=jnp.int8(2))
        state = _set_altar(state, alignment=2)
        state = _with_item(state, slot=0, category=int(ItemCategory.WEAPON), buc=2)

        new_state = drop_at_altar(state, jnp.int32(0))
        assert int(new_state.inventory.items.buc_status[0]) == 3, (
            "Uncursed item on coaligned altar should become blessed (buc=3)"
        )

    def test_drop_on_cross_altar_curses(self):
        """Drop uncursed item on cross-aligned altar → item becomes cursed.

        Cite: vendor/nethack/src/pray.c::doaltar — cross-aligned altar curses.
        """
        state = _make_state()
        # player_align=2 (lawful), altar_alignment=0 (chaotic) → cross-aligned
        state = state.replace(player_align=jnp.int8(2))
        state = _set_altar(state, alignment=0)
        state = _with_item(state, slot=0, category=int(ItemCategory.WEAPON), buc=2)

        new_state = drop_at_altar(state, jnp.int32(0))
        assert int(new_state.inventory.items.buc_status[0]) == 1, (
            "Uncursed item on cross-aligned altar should become cursed (buc=1)"
        )

    def test_drop_neutral_altar_no_change(self):
        """Drop uncursed item on neutral altar → BUC unchanged.

        Cite: pray.c::doaltar — neutral altar (align=1) leaves BUC unchanged.
        """
        state = _make_state()
        # player_align=2 (lawful), altar_alignment=1 (neutral) → no change
        state = state.replace(player_align=jnp.int8(2))
        state = _set_altar(state, alignment=1)
        state = _with_item(state, slot=0, category=int(ItemCategory.WEAPON), buc=2)

        new_state = drop_at_altar(state, jnp.int32(0))
        assert int(new_state.inventory.items.buc_status[0]) == 2, (
            "Uncursed item on neutral altar should stay uncursed (buc=2)"
        )

    def test_drop_no_altar_no_change(self):
        """Drop item when no altar present → BUC unchanged."""
        state = _make_state()
        # altar_alignment defaults to -1 (no altar) — no altar placed
        state = _with_item(state, slot=0, category=int(ItemCategory.WEAPON), buc=2)

        new_state = drop_at_altar(state, jnp.int32(0))
        assert int(new_state.inventory.items.buc_status[0]) == 2, (
            "Uncursed item with no altar present should stay uncursed (buc=2)"
        )

    def test_drop_at_altar_jit_safe(self):
        """drop_at_altar must be jit-compilable."""
        state = _make_state()
        state = state.replace(player_align=jnp.int8(2))
        state = _set_altar(state, alignment=2)
        state = _with_item(state, slot=0, category=int(ItemCategory.WEAPON), buc=2)
        fn = jax.jit(drop_at_altar)
        out = fn(state, jnp.int32(0))
        assert out is not None

    def test_blessed_item_stays_blessed_on_cross_altar(self):
        """Already-blessed item (buc=3) is not mutated by cross-aligned altar."""
        state = _make_state()
        state = state.replace(player_align=jnp.int8(2))
        state = _set_altar(state, alignment=0)
        state = _with_item(state, slot=0, category=int(ItemCategory.WEAPON), buc=3)

        new_state = drop_at_altar(state, jnp.int32(0))
        assert int(new_state.inventory.items.buc_status[0]) == 3, (
            "Already-blessed item should not be changed by cross-aligned altar"
        )
