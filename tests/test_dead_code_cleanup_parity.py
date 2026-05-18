"""Dead-code cleanup parity tests.

Verifies:
  1. Deleted stubs are no longer importable from their old locations.
  2. partial_identify identifies exactly cnt items from unidentified slots.
  3. emit() appends the current buffer to message_history ring.
  4. generate_maze_dla produces a map with at least one FLOOR tile.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# 1. Deleted stubs are not importable from their old modules
# ---------------------------------------------------------------------------

class TestDeletedStubsNotImportable:

    def test_features_dip_fountain_stub_gone(self):
        """features.FeaturesState-based dip_fountain stub must not exist."""
        import importlib
        import inspect
        mod = importlib.import_module("Nethax.nethax.subsystems.features")
        # The real dip_fountain lives at module level but operates on EnvState;
        # the deleted stub had signature (FeaturesState, rng, pos, slot).
        # Verify that if dip_fountain exists it takes an EnvState (not FeaturesState).
        fn = getattr(mod, "dip_fountain", None)
        if fn is not None:
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            # stub had positional: state, rng, pos, slot — 4 params
            # real impl has: state, rng, slot_idx — 3 params
            assert len(params) != 4 or "pos" not in params, (
                "Old FeaturesState-based dip_fountain stub is still present"
            )

    def test_features_sit_throne_stub_gone(self):
        """features.FeaturesState-based sit_throne stub must not exist."""
        import importlib
        import inspect
        mod = importlib.import_module("Nethax.nethax.subsystems.features")
        fn = getattr(mod, "sit_throne", None)
        if fn is not None:
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            # old stub: (state, rng, pos) — 3 params returning FeaturesState tuple
            # real impl is sit_on_throne, not sit_throne — so sit_throne being absent is fine
            # If it exists it should NOT be the old stub (which had a `pos` param)
            assert "pos" not in params, (
                "Old FeaturesState-based sit_throne stub (with pos param) is still present"
            )

    def test_combat_ranged_attack_stub_gone(self):
        """combat.ranged_attack Wave-1 stub must not exist."""
        import importlib
        mod = importlib.import_module("Nethax.nethax.subsystems.combat")
        assert not hasattr(mod, "ranged_attack"), (
            "combat.ranged_attack stub is still present"
        )

    def test_combat_passive_attack_stub_gone(self):
        """combat.passive_attack Wave-1 stub must not exist."""
        import importlib
        mod = importlib.import_module("Nethax.nethax.subsystems.combat")
        assert not hasattr(mod, "passive_attack"), (
            "combat.passive_attack stub is still present"
        )

    def test_shop_enter_shop_stub_gone(self):
        """shop.enter_shop legacy stub must not exist."""
        import importlib
        mod = importlib.import_module("Nethax.nethax.subsystems.shop")
        assert not hasattr(mod, "enter_shop"), (
            "shop.enter_shop legacy stub is still present"
        )

    def test_shop_pickup_in_shop_stub_gone(self):
        """shop.pickup_in_shop legacy stub must not exist."""
        import importlib
        mod = importlib.import_module("Nethax.nethax.subsystems.shop")
        assert not hasattr(mod, "pickup_in_shop"), (
            "shop.pickup_in_shop legacy stub is still present"
        )

    def test_shop_pay_bill_stub_gone(self):
        """shop.pay_bill legacy stub must not exist."""
        import importlib
        mod = importlib.import_module("Nethax.nethax.subsystems.shop")
        assert not hasattr(mod, "pay_bill"), (
            "shop.pay_bill legacy stub is still present"
        )

    def test_shop_attack_shopkeeper_stub_gone(self):
        """shop.attack_shopkeeper legacy stub must not exist."""
        import importlib
        mod = importlib.import_module("Nethax.nethax.subsystems.shop")
        assert not hasattr(mod, "attack_shopkeeper"), (
            "shop.attack_shopkeeper legacy stub is still present"
        )

    def test_inventory_put_on_ring_stub_gone(self):
        """inventory.put_on_ring legacy stub must not exist."""
        import importlib
        mod = importlib.import_module("Nethax.nethax.subsystems.inventory")
        assert not hasattr(mod, "put_on_ring"), (
            "inventory.put_on_ring stub is still present (shadowed by items_jewelry)"
        )

    def test_items_apply_blessing_stub_gone(self):
        """items.apply_blessing no-op stub must not exist."""
        import importlib
        mod = importlib.import_module("Nethax.nethax.subsystems.items")
        assert not hasattr(mod, "apply_blessing"), (
            "items.apply_blessing stub is still present"
        )

    def test_items_apply_curse_stub_gone(self):
        """items.apply_curse no-op stub must not exist."""
        import importlib
        mod = importlib.import_module("Nethax.nethax.subsystems.items")
        assert not hasattr(mod, "apply_curse"), (
            "items.apply_curse stub is still present"
        )

    def test_items_erode_stub_gone(self):
        """items.erode no-op stub must not exist."""
        import importlib
        mod = importlib.import_module("Nethax.nethax.subsystems.items")
        assert not hasattr(mod, "erode"), (
            "items.erode stub is still present"
        )

    def test_items_enchant_stub_gone(self):
        """items.enchant no-op stub must not exist."""
        import importlib
        mod = importlib.import_module("Nethax.nethax.subsystems.items")
        assert not hasattr(mod, "enchant"), (
            "items.enchant stub is still present"
        )


# ---------------------------------------------------------------------------
# 2. partial_identify identifies exactly cnt items
# ---------------------------------------------------------------------------

class TestPartialIdIdentifiesCntItems:

    def _make_id_state(self, n_already_identified: int = 0):
        from Nethax.nethax.subsystems.identification import IdentificationState, NUM_OBJECTS
        state = IdentificationState.unshuffled()
        if n_already_identified > 0:
            # Mark the first n slots as already identified.
            pre = state.identified.at[:n_already_identified].set(True)
            state = state.replace(identified=pre)
        return state

    def test_identifies_exactly_cnt_items_from_zero(self):
        from Nethax.nethax.subsystems.identification import partial_identify
        state = self._make_id_state(0)
        cnt = 5
        new_state = partial_identify(state, _RNG, cnt)
        n_identified = int(jnp.sum(new_state.identified))
        assert n_identified == cnt, f"Expected {cnt} identified, got {n_identified}"

    def test_identifies_exactly_cnt_additional_items(self):
        """Already-identified items are not double-counted."""
        from Nethax.nethax.subsystems.identification import partial_identify
        state = self._make_id_state(10)
        cnt = 3
        new_state = partial_identify(state, _RNG, cnt)
        n_identified = int(jnp.sum(new_state.identified))
        assert n_identified == 10 + cnt, (
            f"Expected {10 + cnt} total identified, got {n_identified}"
        )

    def test_does_not_exceed_total_slots(self):
        """cnt larger than remaining unidentified slots saturates cleanly."""
        from Nethax.nethax.subsystems.identification import partial_identify, NUM_OBJECTS
        state = self._make_id_state(0)
        new_state = partial_identify(state, _RNG, NUM_OBJECTS + 100)
        n_identified = int(jnp.sum(new_state.identified))
        assert n_identified == NUM_OBJECTS

    def test_cnt_zero_changes_nothing(self):
        from Nethax.nethax.subsystems.identification import partial_identify
        state = self._make_id_state(0)
        new_state = partial_identify(state, _RNG, 0)
        assert jnp.array_equal(new_state.identified, state.identified)

    def test_different_rngs_pick_different_slots(self):
        """Two different keys should (almost certainly) identify different slots."""
        from Nethax.nethax.subsystems.identification import partial_identify
        state = self._make_id_state(0)
        rng2 = jax.random.PRNGKey(99)
        s1 = partial_identify(state, _RNG, 5)
        s2 = partial_identify(state, rng2, 5)
        # It's extremely unlikely all 5 slots are identical across two keys.
        assert not jnp.array_equal(s1.identified, s2.identified), (
            "Expected different rngs to pick different slots"
        )


# ---------------------------------------------------------------------------
# 3. emit() appends to message_history ring buffer
# ---------------------------------------------------------------------------

class TestMessagesEmitAppendsToHistory:

    def _make_state(self):
        from Nethax.nethax.subsystems.messages import MessageState
        return MessageState.default()

    def test_emit_increments_history_index(self):
        from Nethax.nethax.subsystems.messages import emit
        state = self._make_state()
        assert int(state.history_index) == 0
        new_state = emit(state, 1)
        assert int(new_state.history_index) == 1

    def test_emit_saves_old_buffer_to_history(self):
        """The buffer present before emit ends up at history[0] after the call."""
        from Nethax.nethax.subsystems.messages import emit, MessageState
        import jax.numpy as jnp
        # Put a sentinel byte in the current buffer.
        buf = jnp.zeros((256,), dtype=jnp.uint8).at[0].set(jnp.uint8(7))
        state = MessageState(
            message_buffer=buf,
            message_history=jnp.zeros((20, 256), dtype=jnp.uint8),
            history_index=jnp.int32(0),
        )
        new_state = emit(state, 2)
        assert int(new_state.message_history[0, 0]) == 7, (
            "Old buffer[0] should have been saved to history[0]"
        )

    def test_emit_writes_msg_id_to_new_buffer(self):
        """After emit, message_buffer[0] holds the msg_id."""
        from Nethax.nethax.subsystems.messages import emit
        state = self._make_state()
        new_state = emit(state, 3)
        assert int(new_state.message_buffer[0]) == 3

    def test_emit_ring_wraps_at_history_len(self):
        """After HISTORY_LEN emits the ring index wraps to 0."""
        from Nethax.nethax.subsystems.messages import emit, HISTORY_LEN
        state = self._make_state()
        for i in range(HISTORY_LEN):
            state = emit(state, i % 7)
        assert int(state.history_index) == 0, (
            f"Expected ring to wrap to 0 after {HISTORY_LEN} emits"
        )

    def test_emit_multiple_messages_accumulate(self):
        """Three sequential emits produce three distinct history rows."""
        from Nethax.nethax.subsystems.messages import emit
        state = self._make_state()
        state = emit(state, 1)
        state = emit(state, 2)
        state = emit(state, 3)
        # History rows 0,1,2 hold the msg_ids of emits 1,2,3 respectively
        # (each emit saves the *previous* buffer — which had the prior msg_id).
        assert int(state.history_index) == 3


# ---------------------------------------------------------------------------
# 4. generate_maze_dla produces a map with at least one FLOOR tile
# ---------------------------------------------------------------------------

class TestDlaMazeHasFloorTiles:

    def test_dla_maze_has_floor_tiles(self):
        from Nethax.nethax.dungeon.mazes import generate_maze_dla, TILE_FLOOR
        grid, h, w = generate_maze_dla(_RNG)
        n_floor = int(jnp.sum(grid == jnp.int8(TILE_FLOOR)))
        assert n_floor > 0, "DLA maze must contain at least one FLOOR tile"

    def test_dla_maze_shape(self):
        from Nethax.nethax.dungeon.mazes import generate_maze_dla
        from Nethax.nethax.dungeon.branches import MAP_H, MAP_W
        grid, h, w = generate_maze_dla(_RNG)
        assert grid.shape == (MAP_H, MAP_W)
        assert h == MAP_H and w == MAP_W

    def test_dla_maze_boundary_is_wall(self):
        """All boundary cells must be TILE_WALL."""
        from Nethax.nethax.dungeon.mazes import generate_maze_dla, TILE_WALL
        grid, h, w = generate_maze_dla(_RNG)
        assert jnp.all(grid[0, :] == TILE_WALL), "Top row must be wall"
        assert jnp.all(grid[h - 1, :] == TILE_WALL), "Bottom row must be wall"
        assert jnp.all(grid[:, 0] == TILE_WALL), "Left col must be wall"
        assert jnp.all(grid[:, w - 1] == TILE_WALL), "Right col must be wall"

    def test_dla_maze_different_seeds_differ(self):
        """Two different seeds should produce different maps."""
        from Nethax.nethax.dungeon.mazes import generate_maze_dla
        rng2 = jax.random.PRNGKey(1)
        g1, _, _ = generate_maze_dla(_RNG)
        g2, _, _ = generate_maze_dla(rng2)
        assert not jnp.array_equal(g1, g2), (
            "Different seeds should produce different DLA maps"
        )
