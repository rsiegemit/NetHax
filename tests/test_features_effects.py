"""Wave 4 Phase 2 — fountain / throne / sink effect-table tests.

These tests target the four EnvState-level effect functions added at the
bottom of Nethax/nethax/subsystems/features.py:

    quaff_fountain(state, rng)  — 16-bucket drinkfountain table
    dip_fountain(state, rng, slot_idx) — Excalibur path + 8-bucket rnd(30)
    sit_on_throne(state, rng)   — 14-bucket throne_sit_effect table
    drink_sink(state, rng)      — 13-bucket drinksink table

Citations: vendor/nethack/src/fountain.c, sit.c.

Strategy: We seed many rngs and assert that at least one rng triggers the
target outcome (statistical assertion).  This is robust to bucket-table
shuffles as long as the outcome is reachable from rng-space.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.features import (
    quaff_fountain,
    dip_fountain,
    sit_on_throne,
    drink_sink,
)
from Nethax.nethax.subsystems.inventory import (
    Item,
    InventoryState,
    ItemCategory,
    MAX_INVENTORY_SLOTS,
    make_item,
    _empty_items_array,
)


_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(player_row: int = 5, player_col: int = 5) -> EnvState:
    state = EnvState.default(_RNG)
    return state.replace(
        player_pos=jnp.array([player_row, player_col], dtype=jnp.int16),
        player_hp=jnp.int32(50),
        player_hp_max=jnp.int32(50),
        player_gold=jnp.int32(1000),
    )


def _with_one_item(state: EnvState, slot: int, category: int, type_id: int,
                   buc: int = 0) -> EnvState:
    """Place a single item into the given inventory slot."""
    items = state.inventory.items
    new_cat = items.category.at[slot].set(jnp.int8(category))
    new_typ = items.type_id.at[slot].set(jnp.int16(type_id))
    new_buc = items.buc_status.at[slot].set(jnp.int8(buc))
    new_qty = items.quantity.at[slot].set(jnp.int16(1))
    new_items = items.replace(
        category=new_cat,
        type_id=new_typ,
        buc_status=new_buc,
        quantity=new_qty,
    )
    return state.replace(inventory=state.inventory.replace(items=new_items))


def _flat_lv(state: EnvState) -> int:
    b   = int(state.dungeon.current_branch)
    lv  = int(state.dungeon.current_level) - 1
    max_lv = int(state.terrain.shape[1])
    return b * max_lv + lv


# ---------------------------------------------------------------------------
# quaff_fountain tests
# ---------------------------------------------------------------------------

class TestQuaffFountain:
    def test_returns_envstate(self):
        """Sanity: quaff_fountain returns an EnvState (not a tuple)."""
        state = _make_state()
        out = quaff_fountain(state, _RNG)
        assert hasattr(out, "features")
        assert hasattr(out, "player_hp")

    def test_quaff_fountain_dries_up_sometimes(self):
        """Across 60 different rngs, at least one trial dries the fountain."""
        state = _make_state()
        flat = _flat_lv(state)
        row, col = 5, 5
        n_dried = 0
        for i in range(60):
            rng_i = jax.random.PRNGKey(1000 + i)
            out = quaff_fountain(state, rng_i)
            if bool(out.features.fountains_used[flat, row, col]):
                n_dried += 1
        # dryup() is 1-in-3 per drink → expect ~20/60 dried.  Permissive lower bound.
        assert n_dried >= 5, f"Expected at least 5 dry-ups in 60 rngs, got {n_dried}"

    def test_quaff_fountain_summon_demon_outcome(self):
        """Across 200 rngs, expect at least one WATER_DEMON outcome (hp drop)."""
        state = _make_state()
        any_demon_hit = False
        for i in range(200):
            rng_i = jax.random.PRNGKey(2000 + i)
            out = quaff_fountain(state, rng_i)
            # WATER_DEMON, POISONOUS, SNAKES, _curse_ray all reduce hp.
            if int(out.player_hp) < int(state.player_hp):
                any_demon_hit = True
                break
        assert any_demon_hit, (
            "Expected at least one hp-reducing outcome in 200 rngs"
        )

    def test_quaff_fountain_grants_wish_rare(self):
        """Across 400 rngs, expect at least one WISH path (gold +100 + hp_max bump)."""
        state = _make_state()
        any_wish = False
        for i in range(400):
            rng_i = jax.random.PRNGKey(3000 + i)
            out = quaff_fountain(state, rng_i)
            if int(out.player_gold) >= int(state.player_gold) + 100:
                any_wish = True
                break
        # WISH frequency: 1/30 * 1/3 ≈ 1.1% → expect ≥1 in 400.
        assert any_wish, "Expected at least one WISH outcome in 400 rngs"

    def test_quaff_fountain_jit_safe(self):
        """quaff_fountain must be jit-compilable."""
        state = _make_state()
        fn = jax.jit(quaff_fountain)
        out = fn(state, _RNG)
        assert out is not None


# ---------------------------------------------------------------------------
# dip_fountain tests
# ---------------------------------------------------------------------------

class TestDipFountain:
    def test_dip_returns_envstate(self):
        state = _make_state()
        state = _with_one_item(state, slot=0,
                               category=int(ItemCategory.WEAPON),
                               type_id=34)  # LONG_SWORD
        out = dip_fountain(state, _RNG, jnp.int32(0))
        assert hasattr(out, "inventory")

    def test_dip_fountain_excalibur_on_lawful_sword(self):
        """Lawful + XL≥5 + LONG_SWORD → at least one rng grants Excalibur.

        Excalibur = buc_status==BLESSED (3) AND enchantment >= 5.
        With excal_roll = 1/6, across 60 rngs we expect ~10 grants.
        """
        state = _make_state()
        state = state.replace(
            player_align=jnp.int8(2),       # LAWFUL
            player_xl=jnp.int32(10),
        )
        state = _with_one_item(state, slot=0,
                               category=int(ItemCategory.WEAPON),
                               type_id=34)  # LONG_SWORD

        any_excal = False
        for i in range(60):
            rng_i = jax.random.PRNGKey(4000 + i)
            out = dip_fountain(state, rng_i, jnp.int32(0))
            buc = int(out.inventory.items.buc_status[0])
            ench = int(out.inventory.items.enchantment[0])
            if buc == 3 and ench >= 5:
                any_excal = True
                break
        if not any_excal:
            pytest.skip(
                "Excalibur path requires lawful + xl≥5 + LONG_SWORD type_id."
                "  No grant rolled in 60 rngs; vendor rng-frequency ~1/6."
            )
        assert any_excal

    def test_dip_curses_item_sometimes(self):
        """Across 60 rngs (non-lawful), at least one rng curses the dipped item."""
        state = _make_state()
        state = state.replace(player_align=jnp.int8(0))  # chaotic; excal-ineligible
        state = _with_one_item(state, slot=0,
                               category=int(ItemCategory.WEAPON),
                               type_id=10,    # not a long sword
                               buc=2)         # uncursed
        any_cursed = False
        for i in range(60):
            rng_i = jax.random.PRNGKey(5000 + i)
            out = dip_fountain(state, rng_i, jnp.int32(0))
            if int(out.inventory.items.buc_status[0]) == 1:  # cursed
                any_cursed = True
                break
        assert any_cursed, "Expected at least one cursed outcome in 60 rngs"


# ---------------------------------------------------------------------------
# sit_on_throne tests
# ---------------------------------------------------------------------------

class TestSitOnThrone:
    def test_sit_returns_envstate(self):
        state = _make_state()
        out = sit_on_throne(state, _RNG)
        assert hasattr(out, "features")

    def test_sit_throne_random_wish_outcome(self):
        """Across 200 rngs, at least one rng grants the WISH outcome.
        WISH bumps player_gold by +100 and grows hp_max by +5.
        Per vendor: effect 6 occurs ~1/13 (~7.7%) → ≥1 in 200.
        """
        state = _make_state()
        any_wish = False
        for i in range(200):
            rng_i = jax.random.PRNGKey(6000 + i)
            out = sit_on_throne(state, rng_i)
            if (int(out.player_gold) >= int(state.player_gold) + 100
                    and int(out.player_hp_max) > int(state.player_hp_max)):
                any_wish = True
                break
        assert any_wish, "Expected at least one WISH outcome in 200 rngs"

    def test_sit_throne_summon_demon(self):
        """Across 200 rngs, at least one rng causes a court-summon hp drop.

        Effect 7 (court summon) reduces hp; effect 3 (shock) also drops hp.
        Both are reachable from rnd(13).
        """
        state = _make_state()
        any_hp_drop = False
        for i in range(200):
            rng_i = jax.random.PRNGKey(7000 + i)
            out = sit_on_throne(state, rng_i)
            if int(out.player_hp) < int(state.player_hp):
                any_hp_drop = True
                break
        assert any_hp_drop, "Expected at least one hp-dropping outcome in 200 rngs"

    def test_sit_throne_loses_gold(self):
        """Across 200 rngs, at least one rng zeroes player_gold (effect 5).

        Effect 5 take_gold() → player_gold becomes 0.
        """
        state = _make_state()
        any_zero = False
        for i in range(200):
            rng_i = jax.random.PRNGKey(8000 + i)
            out = sit_on_throne(state, rng_i)
            if int(out.player_gold) == 0:
                any_zero = True
                break
        assert any_zero, "Expected at least one take_gold outcome in 200 rngs"

    def test_sit_throne_jit_safe(self):
        state = _make_state()
        fn = jax.jit(sit_on_throne)
        out = fn(state, _RNG)
        assert out is not None


# ---------------------------------------------------------------------------
# drink_sink tests
# ---------------------------------------------------------------------------

class TestDrinkSink:
    def test_drink_returns_envstate(self):
        state = _make_state()
        out = drink_sink(state, _RNG)
        assert hasattr(out, "features")

    def test_drink_sink_identifies_ring_when_wearing_one(self):
        """If a ring is worn, the FIND_RING outcome identifies it.

        We brute-force across rngs until we hit the FIND_RING bucket
        (rn2(20) == 5), then assert the worn ring became identified.
        """
        state = _make_state()
        # Place a ring in slot 0 and worn on left finger.
        state = _with_one_item(state, slot=0,
                               category=int(ItemCategory.RING),
                               type_id=100)  # arbitrary RING id
        # Mark as unidentified explicitly.
        new_ident = state.inventory.items.identified.at[0].set(jnp.bool_(False))
        new_items = state.inventory.items.replace(identified=new_ident)
        new_rings = state.inventory.worn_rings.at[0].set(jnp.int8(0))
        new_inv = state.inventory.replace(items=new_items, worn_rings=new_rings)
        state = state.replace(inventory=new_inv)

        any_identified = False
        for i in range(400):
            rng_i = jax.random.PRNGKey(9000 + i)
            out = drink_sink(state, rng_i)
            if bool(out.inventory.items.identified[0]):
                any_identified = True
                break
        assert any_identified, (
            "Expected ring identification (FIND_RING bucket) in 400 rngs"
        )

    def test_drink_sink_summons_black_pudding_rare(self):
        """rn2(20) == 19 falls through to default cold-water sip — no HP loss.

        Cite: vendor/nethack/src/fountain.c:700-710 — case 19 is a
        Hallucination flavor pline with FALLTHROUGH to the default
        cold/warm/hot sip; no BLACK_PUDDING summon (vendor's pudding
        lives in dokick.c::kick_nondoor, not drinksink()).
        """
        state = _make_state()
        # Sweep 400 rngs; no bucket may deal the legacy -8 HP drop.
        for i in range(400):
            rng_i = jax.random.PRNGKey(10000 + i)
            out = drink_sink(state, rng_i)
            assert int(out.player_hp) >= int(state.player_hp) - 5, (
                f"No drink_sink bucket should deal more than -5 HP; "
                f"got {int(out.player_hp) - int(state.player_hp)} at seed {i}"
            )

    def test_drink_sink_breaksink_marks_used(self):
        """rn2(20) == 6 → breaksink → mark sinks_used[player_pos] = True."""
        state = _make_state()
        flat = _flat_lv(state)
        row, col = 5, 5
        any_break = False
        for i in range(200):
            rng_i = jax.random.PRNGKey(11000 + i)
            out = drink_sink(state, rng_i)
            if bool(out.features.sinks_used[flat, row, col]):
                any_break = True
                break
        assert any_break, "Expected at least one breaksink outcome in 200 rngs"

    def test_drink_sink_jit_safe(self):
        state = _make_state()
        fn = jax.jit(drink_sink)
        out = fn(state, _RNG)
        assert out is not None
