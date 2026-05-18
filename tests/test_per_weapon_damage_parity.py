"""Per-weapon damage parity tests.

Verifies that _single_melee_strike uses per-weapon dice from the objects table
rather than the old hardcoded 1d4.

Vendor reference: vendor/nethack/src/weapon.c::dmgval lines 225-295.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest
import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.combat import _single_melee_strike

_RNG = jax.random.PRNGKey(42)


def _fresh_state() -> EnvState:
    return EnvState.default(_RNG)


def _state_with_weapon(type_id: int, is_large_target: bool = False):
    """Return a state where the player wields item type_id and a live monster
    is at an adjacent tile.

    STR=18 gives dbon=2.  Unskilled weapon-skill gives skill_dmg=-2.
    Net bonus = 0, so raw dice == observed damage.

    Monster AC=50 ensures every strike hits.
    """
    state = _fresh_state().replace(
        player_pos=jnp.array([5, 5], dtype=jnp.int16),
        player_str=jnp.int16(18),    # dbon=2
        player_dex=jnp.int8(10),
        player_xl=jnp.int32(1),
        player_role=jnp.int8(0),     # non-Monk, non-Samurai
    )

    # Slot 0: the weapon to test; enchantment 0.
    items = state.inventory.items
    items = items.replace(
        type_id=items.type_id.at[0].set(jnp.int16(type_id)),
        enchantment=items.enchantment.at[0].set(jnp.int8(0)),
    )
    inventory = state.inventory.replace(
        items=items,
        wielded=jnp.int8(0),
    )

    # Monster 0: adjacent, alive, infinite HP, AC=50 (always hit), correct size.
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        hp=mai.hp.at[0].set(jnp.int32(99999)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(99999)),
        pos=mai.pos.at[0].set(jnp.array([5, 6], dtype=jnp.int16)),
        ac=mai.ac.at[0].set(jnp.int8(50)),      # guarantees hit
        is_large=mai.is_large.at[0].set(is_large_target),
    )

    return state.replace(inventory=inventory, monster_ai=mai)


def _state_bare_handed(is_large_target: bool = False):
    """State with no wielded weapon (wielded=-1)."""
    state = _fresh_state().replace(
        player_pos=jnp.array([5, 5], dtype=jnp.int16),
        player_str=jnp.int16(18),
        player_dex=jnp.int8(10),
        player_xl=jnp.int32(1),
        player_role=jnp.int8(0),
    )
    inventory = state.inventory.replace(wielded=jnp.int8(-1))
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        hp=mai.hp.at[0].set(jnp.int32(99999)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(99999)),
        pos=mai.pos.at[0].set(jnp.array([5, 6], dtype=jnp.int16)),
        ac=mai.ac.at[0].set(jnp.int8(50)),
        is_large=mai.is_large.at[0].set(is_large_target),
    )
    return state.replace(inventory=inventory, monster_ai=mai)


def _hit_damages(state, n: int = 200) -> list:
    """Collect damage values for hits over n trials."""
    rng = jax.random.PRNGKey(7)
    dmgs = []
    for _ in range(n):
        rng, sub = jax.random.split(rng)
        _, dmg, hit = _single_melee_strike(state, sub, jnp.int32(0))
        if bool(hit):
            dmgs.append(int(dmg))
    return dmgs


# ---------------------------------------------------------------------------
# Tests — with STR=18, dbon=2 and unskilled skill_dmg=-2, net bonus = 0.
# So mean(observed) ≈ mean(raw dice).
# ---------------------------------------------------------------------------

def test_dagger_does_1d4_vs_small():
    """Dagger (type_id=17) vs small target: sdam=(1,4) → mean ~2.5.

    Vendor: objects.c entry 17 oc_wsdam=4. weapon.c:264-265 rnd(4).
    """
    state = _state_with_weapon(type_id=17, is_large_target=False)
    dmgs = _hit_damages(state)
    assert len(dmgs) > 100, "too few hits"
    avg = sum(dmgs) / len(dmgs)
    assert all(1 <= d <= 4 for d in dmgs), f"dagger dmg out of [1,4]: {set(dmgs)}"
    assert 2.0 <= avg <= 3.2, f"dagger 1d4 mean expected ~2.5, got {avg:.3f}"


def test_long_sword_does_1d8_vs_small():
    """Long sword (type_id=37) vs small target: sdam=(1,8) → mean ~4.5.

    Vendor: objects.c entry 37 oc_wsdam=8. weapon.c:264-265 rnd(8).
    """
    state = _state_with_weapon(type_id=37, is_large_target=False)
    dmgs = _hit_damages(state)
    assert len(dmgs) > 100, "too few hits"
    avg = sum(dmgs) / len(dmgs)
    assert all(1 <= d <= 8 for d in dmgs), f"long sword small dmg out of [1,8]: {set(dmgs)}"
    assert 3.5 <= avg <= 5.8, f"long sword 1d8 mean expected ~4.5, got {avg:.3f}"


def test_long_sword_does_1d12_vs_large():
    """Long sword (type_id=37) vs large target: ldam=(1,12) → mean ~6.5.

    Vendor: objects.c entry 37 oc_wldam=12. weapon.c:226-227 rnd(12).
    """
    state = _state_with_weapon(type_id=37, is_large_target=True)
    dmgs = _hit_damages(state)
    assert len(dmgs) > 100, "too few hits"
    avg = sum(dmgs) / len(dmgs)
    assert all(1 <= d <= 12 for d in dmgs), f"long sword large dmg out of [1,12]: {set(dmgs)}"
    assert 5.0 <= avg <= 8.5, f"long sword 1d12 mean expected ~6.5, got {avg:.3f}"


@pytest.mark.timeout(600)
def test_two_handed_sword_does_3d6_vs_large():
    """Two-handed sword (type_id=38) vs large: 1d6+2d6=3d6 → mean ~10.5.

    Vendor: objects.c entry 38 oc_wldam=6 → rnd(6); weapon.c:259-261 += d(2,6).
    Uses extended timeout (600s) because JAX re-traces when ds2>0 for the first
    weapon with a large-target bonus, adding JIT compilation overhead.
    """
    state = _state_with_weapon(type_id=38, is_large_target=True)
    dmgs = _hit_damages(state)
    assert len(dmgs) > 100, "too few hits"
    avg = sum(dmgs) / len(dmgs)
    assert all(3 <= d <= 18 for d in dmgs), f"2H sword large dmg out of [3,18]: {set(dmgs)}"
    assert 9.0 <= avg <= 12.5, f"2H sword 3d6 mean expected ~10.5, got {avg:.3f}"


def test_fists_does_1d2():
    """Bare-handed (no weapon) vs small: 1d2 → mean ~1.5.

    Nethax sentinel: type_id clamped to 0 → sdam=(1,2).
    """
    state = _state_bare_handed(is_large_target=False)
    dmgs = _hit_damages(state)
    assert len(dmgs) > 100, "too few hits"
    avg = sum(dmgs) / len(dmgs)
    assert all(1 <= d <= 2 for d in dmgs), f"fists dmg out of [1,2]: {set(dmgs)}"
    assert 1.2 <= avg <= 1.8, f"fists 1d2 mean expected ~1.5, got {avg:.3f}"
