"""Endgame parity tests — per-plane damage, entry requirements, ascension.

Tests the subsystems/endgame.py mechanics against vendor/nethack/src/endgame.c
behaviour:

    test_plane_fire_damages_no_resist      — Fire plane does 1 HP/turn without RESIST_FIRE
    test_plane_fire_safe_with_resist       — Fire plane is harmless with RESIST_FIRE
    test_plane_water_drowns_no_breath      — Water plane does 1 HP/turn without MAGIC_BREATHING
    test_astral_drop_amulet_coaligned_ascends  — coaligned Astral altar + Amulet → ascended
    test_astral_drop_amulet_cross_aligned_no_ascend — cross-aligned altar → no ascension

Citations:
    vendor/nethack/src/endgame.c::endgame_env_damage
    vendor/nethack/src/end.c::done(ASCENDED)
    vendor/nethack/src/pray.c::dosacrifice / real_amulet
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.dungeon.branches import Branch
from Nethax.nethax.subsystems.status_effects import Intrinsic, add_intrinsic
from Nethax.nethax.subsystems.inventory import InventoryState, ItemCategory, make_item
from Nethax.nethax.subsystems.items_jewelry import AmuletEffect
from Nethax.nethax.subsystems.scoring import Achievement
from Nethax.nethax.subsystems.endgame import (
    tick_plane_damage,
    try_ascend,
    BRANCH_PLANE_FIRE,
    BRANCH_PLANE_WATER,
    N_PLANES,
)
from Nethax.nethax.dungeon.endgame import (
    ASTRAL_ALTAR_LAWFUL,
    ASTRAL_ALTAR_NEUTRAL,
    ASTRAL_ALTAR_CHAOTIC,
    ASTRAL_ALIGN_LAWFUL,
    ASTRAL_ALIGN_NEUTRAL,
    ASTRAL_ALIGN_CHAOTIC,
)
from Nethax.nethax.subsystems.endgame import BRANCH_PLANE_ASTRAL

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    *,
    branch: int = int(Branch.ENDGAME),
    level: int = BRANCH_PLANE_FIRE,
    pos: tuple[int, int] = (5, 10),
    align: int = ASTRAL_ALIGN_LAWFUL,
    with_amulet: bool = False,
) -> EnvState:
    state = EnvState.default(rng=_RNG, static=StaticParams())
    state = state.replace(
        player_pos=jnp.asarray(pos, dtype=jnp.int16),
        player_align=jnp.int8(align),
        dungeon=state.dungeon.replace(
            current_branch=jnp.int8(branch),
            current_level=jnp.int8(level),
        ),
    )
    if with_amulet:
        amulet = make_item(
            category=int(ItemCategory.AMULET),
            type_id=int(AmuletEffect.YENDOR),
            quantity=1,
            weight=20,
        )
        state = state.replace(inventory=InventoryState.from_items([amulet]))
    return state


def _give_intrinsic(state: EnvState, intrinsic: Intrinsic) -> EnvState:
    new_status = add_intrinsic(state.status, int(intrinsic))
    return state.replace(status=new_status)


# ---------------------------------------------------------------------------
# Fire-plane damage tests
# Citation: vendor/nethack/src/endgame.c::endgame_env_damage (fire branch)
# ---------------------------------------------------------------------------

class TestPlaneFire:
    def test_plane_fire_damages_no_resist(self):
        """Player without RESIST_FIRE on Fire plane loses 1 HP per tick."""
        state = _make_state(level=BRANCH_PLANE_FIRE)
        hp_before = int(state.player_hp)
        after = tick_plane_damage(state, _RNG)
        assert int(after.player_hp) == hp_before - 1, (
            "Fire plane should deal 1 HP/turn without RESIST_FIRE; "
            f"hp went {hp_before} -> {int(after.player_hp)}"
        )

    def test_plane_fire_safe_with_resist(self):
        """Player WITH RESIST_FIRE on Fire plane takes no damage."""
        state = _give_intrinsic(_make_state(level=BRANCH_PLANE_FIRE), Intrinsic.RESIST_FIRE)
        hp_before = int(state.player_hp)
        after = tick_plane_damage(state, _RNG)
        assert int(after.player_hp) == hp_before, (
            "Fire plane should deal 0 HP/turn with RESIST_FIRE; "
            f"hp changed from {hp_before} to {int(after.player_hp)}"
        )

    def test_plane_fire_no_damage_outside_endgame(self):
        """Fire-plane tick does nothing when not in the Endgame branch."""
        state = _make_state(branch=int(Branch.MAIN), level=BRANCH_PLANE_FIRE)
        hp_before = int(state.player_hp)
        after = tick_plane_damage(state, _RNG)
        assert int(after.player_hp) == hp_before


# ---------------------------------------------------------------------------
# Water-plane damage tests
# Citation: vendor/nethack/src/endgame.c::endgame_env_damage (water branch)
# ---------------------------------------------------------------------------

class TestPlaneWater:
    def test_plane_water_drowns_no_breath(self):
        """Player without MAGIC_BREATHING on Water plane loses 1 HP per tick."""
        state = _make_state(level=BRANCH_PLANE_WATER)
        hp_before = int(state.player_hp)
        after = tick_plane_damage(state, _RNG)
        assert int(after.player_hp) == hp_before - 1, (
            "Water plane should deal 1 HP/turn without MAGIC_BREATHING; "
            f"hp went {hp_before} -> {int(after.player_hp)}"
        )

    def test_plane_water_safe_with_magic_breathing(self):
        """Player WITH MAGIC_BREATHING on Water plane takes no damage."""
        state = _give_intrinsic(
            _make_state(level=BRANCH_PLANE_WATER), Intrinsic.MAGIC_BREATHING
        )
        hp_before = int(state.player_hp)
        after = tick_plane_damage(state, _RNG)
        assert int(after.player_hp) == hp_before

    def test_earth_plane_no_passive_damage(self):
        """Earth plane has no per-turn environmental damage."""
        state = _make_state(level=1)  # BRANCH_PLANE_EARTH
        hp_before = int(state.player_hp)
        after = tick_plane_damage(state, _RNG)
        assert int(after.player_hp) == hp_before

    def test_air_plane_no_passive_damage(self):
        """Air plane has no per-turn damage (hazard is movement, not ticking)."""
        state = _make_state(level=2)  # BRANCH_PLANE_AIR
        hp_before = int(state.player_hp)
        after = tick_plane_damage(state, _RNG)
        assert int(after.player_hp) == hp_before


# ---------------------------------------------------------------------------
# Astral-plane ascension tests
# Citation: vendor/nethack/src/end.c::done(ASCENDED),
#           vendor/nethack/src/pray.c::dosacrifice
# ---------------------------------------------------------------------------

class TestAstralAscension:
    def test_astral_drop_amulet_coaligned_ascends(self):
        """Dropping Amulet of Yendor on a coaligned Astral altar → ASCENDED.

        We simulate the 'drop on altar' trigger by constructing a state where
        the player stands on their coaligned altar with the Amulet in
        inventory, then calling try_ascend.  This mirrors the vendor path:
        dosacrifice → real_amulet → done(ASCENDED).
        """
        state = _make_state(
            level=BRANCH_PLANE_ASTRAL,
            pos=ASTRAL_ALTAR_LAWFUL,
            align=ASTRAL_ALIGN_LAWFUL,
            with_amulet=True,
        )
        result = try_ascend(state)
        assert bool(result.done) is True, "Coaligned Astral altar + Amulet must trigger ascension"
        achieved = result.scoring.achievements[int(Achievement.ASCENDED)]
        assert bool(achieved) is True, "ASCENDED achievement must be recorded"
        # Vendor end.c:1344-1351 doubles XP on coaligned ASCENDED, no flat
        # bonus (Audit G #4 removed the legacy +50000 hack); verify the
        # ``ascended`` flag is set so compute_final_score doubles urexp.
        assert bool(result.scoring.ascended) is True, "ascended flag must be set"

    def test_astral_drop_amulet_cross_aligned_no_ascend(self):
        """Dropping Amulet on a cross-aligned altar must NOT trigger ascension.

        Vendor: dosacrifice checks u.ualign.type against altar->ralign;
        mismatched alignment results in anger, not ascension (pray.c:1920-1940).
        """
        # Lawful player stands on Chaotic altar.
        state = _make_state(
            level=BRANCH_PLANE_ASTRAL,
            pos=ASTRAL_ALTAR_CHAOTIC,
            align=ASTRAL_ALIGN_LAWFUL,
            with_amulet=True,
        )
        result = try_ascend(state)
        assert bool(result.done) is False, (
            "Cross-aligned altar must not trigger ascension"
        )
        achieved = result.scoring.achievements[int(Achievement.ASCENDED)]
        assert bool(achieved) is False

    def test_astral_no_amulet_no_ascend(self):
        """Standing on a coaligned altar without the Amulet must not ascend."""
        state = _make_state(
            level=BRANCH_PLANE_ASTRAL,
            pos=ASTRAL_ALTAR_NEUTRAL,
            align=ASTRAL_ALIGN_NEUTRAL,
            with_amulet=False,
        )
        result = try_ascend(state)
        assert bool(result.done) is False

    def test_astral_neutral_altar_neutral_player_ascends(self):
        """Neutral player on Neutral altar with Amulet ascends."""
        state = _make_state(
            level=BRANCH_PLANE_ASTRAL,
            pos=ASTRAL_ALTAR_NEUTRAL,
            align=ASTRAL_ALIGN_NEUTRAL,
            with_amulet=True,
        )
        result = try_ascend(state)
        assert bool(result.done) is True

    def test_astral_chaotic_altar_chaotic_player_ascends(self):
        """Chaotic player on Chaotic altar with Amulet ascends."""
        state = _make_state(
            level=BRANCH_PLANE_ASTRAL,
            pos=ASTRAL_ALTAR_CHAOTIC,
            align=ASTRAL_ALIGN_CHAOTIC,
            with_amulet=True,
        )
        result = try_ascend(state)
        assert bool(result.done) is True


# ---------------------------------------------------------------------------
# Module-level constant sanity checks
# ---------------------------------------------------------------------------

class TestEndgameConstants:
    def test_n_planes(self):
        assert N_PLANES == 5

    def test_branch_plane_indices(self):
        from Nethax.nethax.subsystems.endgame import (
            BRANCH_PLANE_FIRE,
            BRANCH_PLANE_WATER,
            BRANCH_PLANE_ASTRAL,
        )
        assert BRANCH_PLANE_FIRE == 3
        assert BRANCH_PLANE_WATER == 4
        assert BRANCH_PLANE_ASTRAL == 5

    def test_endgame_levels_module_exports(self):
        from Nethax.nethax.dungeon.endgame_levels import (
            BRANCH_PLANE_EARTH,
            BRANCH_PLANE_AIR,
            BRANCH_PLANE_FIRE,
            BRANCH_PLANE_WATER,
            BRANCH_PLANE_ASTRAL,
            generate_plane_earth_level,
            generate_plane_air_level,
            generate_plane_fire_level,
            generate_plane_water_level,
            generate_plane_astral_level,
        )
        assert BRANCH_PLANE_EARTH == 1
        assert BRANCH_PLANE_AIR == 2
        assert BRANCH_PLANE_FIRE == 3
        assert BRANCH_PLANE_WATER == 4
        assert BRANCH_PLANE_ASTRAL == 5
        # Generators must be callable.
        import jax
        rng = jax.random.PRNGKey(0)
        for gen in (
            generate_plane_earth_level,
            generate_plane_air_level,
            generate_plane_fire_level,
            generate_plane_water_level,
            generate_plane_astral_level,
        ):
            terrain, monsters, items = gen(rng)
            assert terrain.shape[0] > 0
