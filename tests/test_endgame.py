"""Wave 5 Phase 4b — Endgame tests.

Covers:
  * 5 plane factories in Nethax/nethax/dungeon/endgame.py
  * Endgame branch wiring in dungeon/branches.py
  * Ascension condition + ascend in subsystems/ascension.py
  * Env-level wiring (step freezes after ascension).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.dungeon.branches import (
    MAP_H,
    MAP_W,
    Branch,
    BRANCH_TABLE,
    init_branch_graph,
)
from Nethax.nethax.dungeon.endgame import (
    generate_earth_plane,
    generate_air_plane,
    generate_fire_plane,
    generate_water_plane,
    generate_astral_plane,
    generate_endgame_level,
    ASTRAL_ALTAR_LAWFUL,
    ASTRAL_ALTAR_NEUTRAL,
    ASTRAL_ALTAR_CHAOTIC,
    ASTRAL_ALIGN_LAWFUL,
    ASTRAL_ALIGN_NEUTRAL,
    ASTRAL_ALIGN_CHAOTIC,
    _MON_EARTH_ELEMENTAL,
    _MON_FIRE_ELEMENTAL,
    _MON_WATER_ELEMENTAL,
    _MON_KRAKEN,
    _MON_ALEAX,
    _MON_HIGH_PRIEST,
)
from Nethax.nethax.subsystems.ascension import (
    check_ascension,
    ascend,
    maybe_ascend,
    player_holds_amulet,
    on_astral_plane,
    on_matching_altar,
    ASTRAL_LEVEL,
)
from Nethax.nethax.subsystems.scoring import Achievement
from Nethax.nethax.subsystems.inventory import (
    InventoryState,
    ItemCategory,
    make_item,
)
from Nethax.nethax.subsystems.items_jewelry import AmuletEffect
from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.env import NethaxEnv

# Tile constants used in assertions.
_T_VOID  = 0
_T_FLOOR = 1
_T_WALL  = 3
_T_LAVA  = 9
_T_ALTAR = 10
_T_POOL  = 19

_RNG = jax.random.PRNGKey(7)


# ---------------------------------------------------------------------------
# Plane factories
# ---------------------------------------------------------------------------


class TestEarthPlane:
    def test_earth_plane_factory_renders(self):
        terrain, monsters, items = generate_earth_plane(_RNG)
        assert terrain.shape == (MAP_H, MAP_W)
        assert monsters.shape[1] == 3
        assert items.shape[1] == 3
        # Earth plane has lots of solid rock (VOID) plus FLOOR caverns.
        n_void  = int(jnp.sum(terrain == _T_VOID))
        n_floor = int(jnp.sum(terrain == _T_FLOOR))
        assert n_void > n_floor, (
            "Earth plane should be mostly solid rock (VOID), with smaller floor caverns"
        )
        assert n_floor > 0

    def test_earth_plane_has_earth_elementals(self):
        _, monsters, _ = generate_earth_plane(_RNG)
        n_elementals = int(jnp.sum(monsters[:, 2] == _MON_EARTH_ELEMENTAL))
        assert n_elementals > 0, "Earth plane should host earth elementals"


class TestAirPlane:
    def test_air_plane_mostly_void(self):
        terrain, monsters, items = generate_air_plane(_RNG)
        assert terrain.shape == (MAP_H, MAP_W)
        n_void  = int(jnp.sum(terrain == _T_VOID))
        n_floor = int(jnp.sum(terrain == _T_FLOOR))
        # Air plane is almost entirely VOID (air); a few landing patches.
        assert n_void >= int(0.9 * MAP_H * MAP_W), (
            f"Air plane should be >=90% VOID; got {n_void} void cells out of "
            f"{MAP_H * MAP_W}"
        )
        assert n_floor > 0, "Air plane should have a small landing patch"


class TestFirePlane:
    def test_fire_plane_has_lava(self):
        terrain, monsters, items = generate_fire_plane(_RNG)
        assert terrain.shape == (MAP_H, MAP_W)
        n_lava = int(jnp.sum(terrain == _T_LAVA))
        assert n_lava > 0, "Fire plane must contain LAVA tiles"
        # Fire plane has lava AND floor islands.
        n_floor = int(jnp.sum(terrain == _T_FLOOR))
        assert n_floor > 0, "Fire plane should have floor islands"
        # Fire elementals must be present.
        _, monsters, _ = generate_fire_plane(_RNG)
        n_fire = int(jnp.sum(monsters[:, 2] == _MON_FIRE_ELEMENTAL))
        assert n_fire > 0


class TestWaterPlane:
    def test_water_plane_has_pools(self):
        terrain, monsters, items = generate_water_plane(_RNG)
        assert terrain.shape == (MAP_H, MAP_W)
        n_pool = int(jnp.sum(terrain == _T_POOL))
        assert n_pool > 0, "Water plane must contain POOL tiles"
        n_floor = int(jnp.sum(terrain == _T_FLOOR))
        # FLOOR bubbles for the air-pocket landings.
        assert n_floor > 0, "Water plane should have FLOOR bubble islands"
        n_kraken = int(jnp.sum(monsters[:, 2] == _MON_KRAKEN))
        assert n_kraken >= 1, "Water plane must have at least one kraken"


class TestAstralPlane:
    def test_astral_plane_has_three_altars(self):
        terrain, monsters, items = generate_astral_plane(_RNG)
        assert terrain.shape == (MAP_H, MAP_W)
        n_altars = int(jnp.sum(terrain == _T_ALTAR))
        assert n_altars == 3, f"Astral plane must have exactly 3 altars, got {n_altars}"
        # And the items[] array must encode their alignments.
        align_codes = set(int(items[i, 2]) for i in range(3))
        assert align_codes == {
            ASTRAL_ALIGN_LAWFUL,
            ASTRAL_ALIGN_NEUTRAL,
            ASTRAL_ALIGN_CHAOTIC,
        }

    def test_astral_plane_has_high_priests(self):
        _, monsters, _ = generate_astral_plane(_RNG)
        n_priests = int(jnp.sum(monsters[:, 2] == _MON_HIGH_PRIEST))
        assert n_priests >= 3, (
            f"Astral plane must have >=3 high priests (one per altar); got {n_priests}"
        )
        n_aleax = int(jnp.sum(monsters[:, 2] == _MON_ALEAX))
        assert n_aleax > 0, "Astral plane must have Aleax angel guards"


# ---------------------------------------------------------------------------
# Endgame branch wiring
# ---------------------------------------------------------------------------


class TestEndgameBranch:
    def test_endgame_branch_has_5_levels(self):
        info = BRANCH_TABLE[int(Branch.ENDGAME)]
        assert int(info.num_levels) == 5, (
            f"Endgame branch must have 5 levels; got {int(info.num_levels)}"
        )

    def test_endgame_branch_graph_wires_5_planes(self):
        graph = init_branch_graph(_RNG, static_params=None)
        # L1..L4 should link to the next plane in sequence.
        for lv in range(1, 5):
            dst_branch = int(graph.stair_links[Branch.ENDGAME, lv - 1, 0])
            dst_level  = int(graph.stair_links[Branch.ENDGAME, lv - 1, 1])
            assert dst_branch == int(Branch.ENDGAME), (
                f"Endgame L{lv} should link within Endgame; got branch {dst_branch}"
            )
            assert dst_level == lv + 1, (
                f"Endgame L{lv} should link to L{lv + 1}; got L{dst_level}"
            )
        # Parent branch is Gehennom; entry happens at Gehennom L16 (Sanctum).
        assert int(graph.parent_branch[Branch.ENDGAME]) == int(Branch.GEHENNOM)
        # Gehennom L16 (index 15) should link to Endgame L1.
        sanctum_link = graph.stair_links[Branch.GEHENNOM, 15]
        assert int(sanctum_link[0]) == int(Branch.ENDGAME)
        assert int(sanctum_link[1]) == 1


# ---------------------------------------------------------------------------
# Ascension condition
# ---------------------------------------------------------------------------


def _make_ascension_state(
    *,
    branch: int,
    level: int,
    pos,
    align: int,
    with_amulet: bool,
) -> EnvState:
    """Construct an EnvState ready for an ascension check.

    Sets player position / alignment, current branch+level, and optionally
    places an Amulet of Yendor in slot 0 of the inventory.
    """
    state = EnvState.default(rng=jax.random.PRNGKey(11), static=StaticParams())

    # Place the player at the requested altar / cell.
    state = state.replace(
        player_pos=jnp.asarray(pos, dtype=jnp.int16),
        player_align=jnp.int8(align),
    )

    # Update dungeon position to (branch, level).
    new_dungeon = state.dungeon.replace(
        current_branch=jnp.int8(branch),
        current_level=jnp.int8(level),
    )
    state = state.replace(dungeon=new_dungeon)

    if with_amulet:
        amulet = make_item(
            category=int(ItemCategory.AMULET),
            type_id=int(AmuletEffect.YENDOR),
            quantity=1,
            weight=20,
        )
        new_inv = InventoryState.from_items([amulet])
        state = state.replace(inventory=new_inv)
    return state


class TestAscensionCondition:
    def test_check_ascension_requires_amulet(self):
        # On the matching altar, with correct alignment, BUT no Amulet.
        state = _make_ascension_state(
            branch=int(Branch.ENDGAME),
            level=ASTRAL_LEVEL,
            pos=ASTRAL_ALTAR_LAWFUL,
            align=ASTRAL_ALIGN_LAWFUL,
            with_amulet=False,
        )
        assert bool(check_ascension(state)) is False
        # Now with the Amulet, ascension triggers.
        state2 = _make_ascension_state(
            branch=int(Branch.ENDGAME),
            level=ASTRAL_LEVEL,
            pos=ASTRAL_ALTAR_LAWFUL,
            align=ASTRAL_ALIGN_LAWFUL,
            with_amulet=True,
        )
        assert bool(check_ascension(state2)) is True

    def test_check_ascension_requires_matching_altar(self):
        # Player is Lawful but standing on the Chaotic altar; should fail.
        state = _make_ascension_state(
            branch=int(Branch.ENDGAME),
            level=ASTRAL_LEVEL,
            pos=ASTRAL_ALTAR_CHAOTIC,
            align=ASTRAL_ALIGN_LAWFUL,
            with_amulet=True,
        )
        assert bool(check_ascension(state)) is False
        # And not on any altar at all.
        state2 = _make_ascension_state(
            branch=int(Branch.ENDGAME),
            level=ASTRAL_LEVEL,
            pos=(1, 1),
            align=ASTRAL_ALIGN_LAWFUL,
            with_amulet=True,
        )
        assert bool(check_ascension(state2)) is False

    def test_check_ascension_requires_astral_plane(self):
        # Player has Amulet and is on a tile whose coords happen to be
        # the Lawful altar — but they're on Main Branch, not Endgame.
        state = _make_ascension_state(
            branch=int(Branch.MAIN),
            level=1,
            pos=ASTRAL_ALTAR_LAWFUL,
            align=ASTRAL_ALIGN_LAWFUL,
            with_amulet=True,
        )
        assert bool(on_astral_plane(state)) is False
        assert bool(check_ascension(state)) is False
        # On Endgame but the wrong level (Earth) — still fails.
        state2 = _make_ascension_state(
            branch=int(Branch.ENDGAME),
            level=1,
            pos=ASTRAL_ALTAR_LAWFUL,
            align=ASTRAL_ALIGN_LAWFUL,
            with_amulet=True,
        )
        assert bool(check_ascension(state2)) is False

    def test_ascend_sets_done_flag(self):
        state = _make_ascension_state(
            branch=int(Branch.ENDGAME),
            level=ASTRAL_LEVEL,
            pos=ASTRAL_ALTAR_NEUTRAL,
            align=ASTRAL_ALIGN_NEUTRAL,
            with_amulet=True,
        )
        # Sanity: ascension condition met.
        assert bool(check_ascension(state)) is True
        new_state = ascend(state)
        assert bool(new_state.done) is True
        # Ascension achievement recorded.
        achieved = new_state.scoring.achievements[int(Achievement.ASCENDED)]
        assert bool(achieved) is True
        # ``ascended`` flag is set so compute_final_score's ascension
        # XP-doubling fires (vendor end.c:1344-1351).  Audit G #4 removed
        # the legacy flat +50000 bonus that had no vendor analogue.
        assert bool(new_state.scoring.ascended) is True


class TestEnvAscensionWiring:
    def test_step_after_ascend_freezes_state(self):
        """Once state.done=True via ascend(), env.step is a no-op."""
        env = NethaxEnv(static=StaticParams())
        # Build an ascension-ready state.  Skip env.reset (it generates
        # Main L1 + monsters); we want a clean ascension scenario.
        state = _make_ascension_state(
            branch=int(Branch.ENDGAME),
            level=ASTRAL_LEVEL,
            pos=ASTRAL_ALTAR_CHAOTIC,
            align=ASTRAL_ALIGN_CHAOTIC,
            with_amulet=True,
        )
        # Trigger ascension explicitly.
        ascended = ascend(state)
        assert bool(ascended.done) is True
        # Now env.step should keep the state frozen.
        rng = jax.random.PRNGKey(13)
        action = jnp.int32(0)  # no-op / movement action; irrelevant.
        new_state, _obs, _reward, done, _info = env.step(ascended, action, rng)
        assert bool(done) is True
        # Timestep should not advance — proves _do_step branch was skipped.
        assert int(new_state.timestep) == int(ascended.timestep)


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------


class TestEndgameDispatcher:
    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
    def test_generate_endgame_level_dispatch(self, depth: int):
        terrain, monsters, items = generate_endgame_level(_RNG, depth)
        assert terrain.shape == (MAP_H, MAP_W)
        assert monsters.shape == (64, 3)
        assert items.shape == (64, 3)
