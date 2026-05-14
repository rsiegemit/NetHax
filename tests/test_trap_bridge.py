"""Wave 5 Phase 4 — wide-carrier ``trigger_trap_envstate`` dispatch tests.

The new ``trigger_trap_envstate(state, rng, row, col) -> EnvState`` operates
on the full EnvState pytree via ``jax.lax.switch`` over a fixed tuple of
branch functions, one per TrapType.  These tests exercise each major branch
and validate the "wide carrier" pytree-shape invariant + JIT compilation.

Vendor reference:  vendor/nethack/src/trap.c::dotrap (single switch(ttyp)).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.traps import (
    TrapType,
    trigger_trap_envstate,
    _TRAP_BRANCHES,
    place_trap,
)
from Nethax.nethax.subsystems.status_effects import TimedStatus
from Nethax.nethax.subsystems.inventory import (
    InventoryState,
    Item,
    ItemCategory,
    make_empty_item,
    MAX_INVENTORY_SLOTS,
    N_ARMOR_SLOTS,
)

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_lv(state: EnvState) -> int:
    b      = int(state.dungeon.current_branch)
    lv     = int(state.dungeon.current_level) - 1
    max_lv = int(state.terrain.shape[1])
    return b * max_lv + lv


def _make_state(player_row: int = 5, player_col: int = 5) -> EnvState:
    state = EnvState.default(_RNG)
    return state.replace(
        player_pos=jnp.array([player_row, player_col], dtype=jnp.int16),
        player_hp=jnp.int32(50),
        player_hp_max=jnp.int32(50),
        player_pw=jnp.int32(40),
        player_pw_max=jnp.int32(40),
    )


def _place_trap_at(state: EnvState, row: int, col: int,
                   kind: TrapType) -> EnvState:
    flat = _flat_lv(state)
    pos  = jnp.array([flat, row, col], dtype=jnp.int32)
    new_traps = place_trap(state.traps, pos, kind, _RNG)
    return state.replace(traps=new_traps)


def _add_floor_around(state: EnvState) -> EnvState:
    """Fill the current level with FLOOR tiles so teleport finds a target."""
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain.at[b, lv].set(
        jnp.full_like(state.terrain[b, lv], jnp.int8(TileType.FLOOR))
    )
    return state.replace(terrain=new_terrain)


# ---------------------------------------------------------------------------
# Per-branch unit tests
# ---------------------------------------------------------------------------

def test_arrow_trap_deals_d6_damage():
    """ARROW_TRAP: HP loss must be in [1, 6]."""
    state = _make_state()
    state = _place_trap_at(state, 5, 5, TrapType.ARROW_TRAP)
    out = trigger_trap_envstate(state, _RNG, 5, 5)
    loss = int(state.player_hp) - int(out.player_hp)
    assert 1 <= loss <= 6, f"Expected HP loss in [1,6], got {loss}"


def test_dart_trap_can_poison():
    """DART_TRAP: at least one seed in 32 must produce a SICK status."""
    state = _make_state()
    state = _place_trap_at(state, 5, 5, TrapType.DART_TRAP)
    any_poisoned = False
    for seed in range(32):
        rng = jax.random.PRNGKey(seed)
        out = trigger_trap_envstate(state, rng, 5, 5)
        if int(out.status.timed_statuses[int(TimedStatus.SICK)]) > 0:
            any_poisoned = True
            break
    assert any_poisoned, "Expected DART_TRAP to poison at least once in 32 trials"


def test_bear_trap_holds_player():
    """BEAR_TRAP: FROZEN status must be set (>=1 turn)."""
    state = _make_state()
    state = _place_trap_at(state, 5, 5, TrapType.BEAR_TRAP)
    out = trigger_trap_envstate(state, _RNG, 5, 5)
    frozen = int(out.status.timed_statuses[int(TimedStatus.FROZEN)])
    assert frozen >= 1, f"Expected FROZEN >= 1, got {frozen}"


def test_pit_trap_damages_and_holds():
    """SPIKED_PIT: HP must decrease and FROZEN must be set."""
    state = _make_state()
    state = _place_trap_at(state, 5, 5, TrapType.SPIKED_PIT)
    out = trigger_trap_envstate(state, _RNG, 5, 5)
    assert int(out.player_hp) < int(state.player_hp), "Expected HP loss"
    frozen = int(out.status.timed_statuses[int(TimedStatus.FROZEN)])
    assert frozen >= 1, f"Expected FROZEN >= 1, got {frozen}"


def test_magic_trap_random_outcome():
    """MAGIC_TRAP: across seeds, at least one outcome must change state."""
    state = _make_state()
    state = _place_trap_at(state, 5, 5, TrapType.MAGIC_TRAP)
    state = _add_floor_around(state)
    seen_change = False
    for seed in range(16):
        rng = jax.random.PRNGKey(seed)
        out = trigger_trap_envstate(state, rng, 5, 5)
        # Compare some observable scalars.
        if (int(out.player_hp)   != int(state.player_hp)
            or int(out.player_gold) != int(state.player_gold)
            or jnp.any(out.status.timed_statuses != state.status.timed_statuses)
            or not jnp.array_equal(out.player_pos, state.player_pos)
            or bool(out.polymorph.is_polymorphed)):
            seen_change = True
            break
    assert seen_change, "Expected MAGIC_TRAP to modify state in at least 1/16 seeds"


def test_poly_trap_polymorphs_player():
    """POLY_TRAP: player should end up polymorphed."""
    state = _make_state()
    state = _place_trap_at(state, 5, 5, TrapType.POLY_TRAP)
    out = trigger_trap_envstate(state, _RNG, 5, 5)
    assert bool(out.polymorph.is_polymorphed), \
        "Expected POLY_TRAP to set is_polymorphed=True"


def test_telep_trap_moves_player():
    """TELEP_TRAP: player_pos should differ from the starting position."""
    state = _make_state(player_row=5, player_col=5)
    state = _add_floor_around(state)
    state = _place_trap_at(state, 5, 5, TrapType.TELEP_TRAP)
    out = trigger_trap_envstate(state, _RNG, 5, 5)
    moved = not jnp.array_equal(out.player_pos, state.player_pos)
    assert moved, f"Expected player to move; pos unchanged at {out.player_pos}"


def test_hole_trap_descends_level():
    """HOLE: current_level should increase by 1."""
    state = _make_state()
    # Start at level 1 (default).  Need branch_levels to allow descent.
    state = _place_trap_at(state, 5, 5, TrapType.HOLE)
    out = trigger_trap_envstate(state, _RNG, 5, 5)
    assert int(out.dungeon.current_level) == int(state.dungeon.current_level) + 1, (
        f"Expected current_level += 1, got "
        f"{int(state.dungeon.current_level)} -> {int(out.dungeon.current_level)}"
    )


def test_sleeping_gas_sleeps_player():
    """SLP_GAS_TRAP: SLEEP timed status should be set (>= 5)."""
    state = _make_state()
    state = _place_trap_at(state, 5, 5, TrapType.SLP_GAS_TRAP)
    out = trigger_trap_envstate(state, _RNG, 5, 5)
    sleep = int(out.status.timed_statuses[int(TimedStatus.SLEEP)])
    assert sleep >= 5, f"Expected SLEEP >= 5, got {sleep}"


def test_fire_trap_damages_and_burns_scrolls():
    """FIRE_TRAP: HP decreases and any inventory scroll has quantity=0."""
    state = _make_state()
    # Place a scroll in slot 0.
    inv = state.inventory
    new_items = inv.items.replace(
        category=inv.items.category.at[0].set(jnp.int8(int(ItemCategory.SCROLL))),
        quantity=inv.items.quantity.at[0].set(jnp.int16(3)),
        type_id=inv.items.type_id.at[0].set(jnp.int16(1)),
    )
    state = state.replace(inventory=inv.replace(items=new_items))
    state = _place_trap_at(state, 5, 5, TrapType.FIRE_TRAP)

    out = trigger_trap_envstate(state, _RNG, 5, 5)
    assert int(out.player_hp) < int(state.player_hp), "Expected HP loss"
    assert int(out.inventory.items.quantity[0]) == 0, (
        f"Expected scroll quantity = 0, got {int(out.inventory.items.quantity[0])}"
    )


# ---------------------------------------------------------------------------
# Wide-carrier and JIT invariants
# ---------------------------------------------------------------------------

def test_all_branches_return_same_pytree_shape():
    """Every entry in _TRAP_BRANCHES must return an EnvState of identical
    pytree structure (the "wide carrier" invariant)."""
    state = _make_state()
    state = _add_floor_around(state)
    reference_struct = jax.tree_util.tree_structure(state)
    for i, branch in enumerate(_TRAP_BRANCHES):
        out = branch(state, _RNG)
        out_struct = jax.tree_util.tree_structure(out)
        assert out_struct == reference_struct, (
            f"Branch {i} ({branch.__name__}) returned different pytree "
            f"structure"
        )


def test_jit_compile_trigger_trap():
    """trigger_trap_envstate must compile cleanly under jax.jit."""
    state = _make_state()
    state = _add_floor_around(state)
    state = _place_trap_at(state, 5, 5, TrapType.ARROW_TRAP)

    @jax.jit
    def _jit_trigger(s, rng, row, col):
        return trigger_trap_envstate(s, rng, row, col)

    out = _jit_trigger(state, _RNG, jnp.int32(5), jnp.int32(5))
    # Sanity: arrow trap damage should be applied.
    assert int(out.player_hp) < int(state.player_hp), \
        f"Expected JIT'd arrow trap to reduce HP, got {int(out.player_hp)}"
