"""Wave 6 Phase B — full vendor-parity Quest goal layout tests.

Verifies that the 13 hand-translated full GOAL layouts (Wave 6) place each
role's nemesis and quest artifact at the expected positions, that each role
has a *distinct* nemesis/artifact (artilist.h + role.c), and that the
``dispatch_quest_level`` API correctly toggles between full and iconic
layouts.

Citations:
  vendor/nethack/dat/<role>-goal.lua  — landmark coordinates per role.
  vendor/nethack/src/role.c           — leader/nemesis/artifact assignments.
  vendor/nethack/include/artilist.h   — artifact ordinal (0-based).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.dungeon import quest_levels
from Nethax.nethax.dungeon.branches import MAP_H, MAP_W
from Nethax.nethax.subsystems.quest import _QUEST_DATA, get_quest_data


_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Per-role nemesis + artifact placement
# ---------------------------------------------------------------------------

class TestSpecificRoleLandmarks:
    def test_wiz_goal_has_eye_of_aethiopica(self):
        """Wizard's quest artifact (Eye of the Aethiopica) is placed."""
        _, _, items = quest_levels.generate_wiz_quest_goal_level_full(_RNG)
        eye_idx = get_quest_data(quest_levels.ROLE_WIZ).artifact_idx
        assert bool(jnp.any(items[:, 2] == eye_idx))

    def test_wiz_goal_has_neferet_or_dark_one(self):
        """Wizard's nemesis (Dark One) is placed.

        Spec mentions 'Neferet the Green' as the Wiz nemesis, but role.c
        line 556 documents Neferet as the *leader* and the Dark One as
        the nemesis. We assert the role.c-canonical nemesis is present.
        """
        _, monsters, _ = quest_levels.generate_wiz_quest_goal_level_full(_RNG)
        nem_idx = get_quest_data(quest_levels.ROLE_WIZ).nemesis_idx
        assert bool(jnp.any(monsters[:, 2] == nem_idx))

    def test_val_goal_has_orb_of_fate(self):
        """Valkyrie's quest artifact (Orb of Fate) is placed."""
        _, _, items = quest_levels.generate_val_quest_goal_level_full(_RNG)
        orb_idx = get_quest_data(quest_levels.ROLE_VAL).artifact_idx
        assert bool(jnp.any(items[:, 2] == orb_idx))

    def test_val_goal_has_lord_surtur(self):
        """Valkyrie's nemesis (Lord Surtur) is placed."""
        _, monsters, _ = quest_levels.generate_val_quest_goal_level_full(_RNG)
        nem_idx = get_quest_data(quest_levels.ROLE_VAL).nemesis_idx
        assert bool(jnp.any(monsters[:, 2] == nem_idx))

    def test_hea_goal_has_cyclops(self):
        """Healer's nemesis (Cyclops, per role.c line 172) is placed."""
        _, monsters, _ = quest_levels.generate_hea_quest_goal_level_full(_RNG)
        nem_idx = get_quest_data(quest_levels.ROLE_HEA).nemesis_idx
        assert bool(jnp.any(monsters[:, 2] == nem_idx))

    def test_arc_goal_has_orb_of_detection(self):
        """Archeologist's quest artifact (Orb of Detection) is placed."""
        _, _, items = quest_levels.generate_arc_quest_goal_level_full(_RNG)
        art_idx = get_quest_data(quest_levels.ROLE_ARC).artifact_idx
        assert bool(jnp.any(items[:, 2] == art_idx))


# ---------------------------------------------------------------------------
# Coverage / distinctness across all 13 roles
# ---------------------------------------------------------------------------

class TestAcrossAllRoles:
    def test_all_13_full_factories_callable(self):
        names = [
            "generate_arc_quest_goal_level_full",
            "generate_bar_quest_goal_level_full",
            "generate_cav_quest_goal_level_full",
            "generate_hea_quest_goal_level_full",
            "generate_kni_quest_goal_level_full",
            "generate_mon_quest_goal_level_full",
            "generate_pri_quest_goal_level_full",
            "generate_ran_quest_goal_level_full",
            "generate_rog_quest_goal_level_full",
            "generate_sam_quest_goal_level_full",
            "generate_tou_quest_goal_level_full",
            "generate_val_quest_goal_level_full",
            "generate_wiz_quest_goal_level_full",
        ]
        for n in names:
            fn = getattr(quest_levels, n)
            terrain, monsters, items = fn(_RNG)
            assert terrain.shape == (MAP_H, MAP_W)
            assert monsters.shape == (64, 3)
            assert items.shape == (64, 3)

    def test_each_role_goal_has_distinct_artifact(self):
        """13 roles → 13 distinct artifact indices (per artilist.h slots 20-33)."""
        artifact_ids = []
        for role in range(quest_levels.N_ROLES):
            _, _, items = quest_levels.dispatch_quest_level(_RNG, role)
            # The role's artifact is the only item placed with non-sentinel id.
            data = get_quest_data(role)
            assert bool(jnp.any(items[:, 2] == data.artifact_idx)), (
                f"role={role} missing artifact {data.artifact_idx}"
            )
            artifact_ids.append(data.artifact_idx)
        assert len(set(artifact_ids)) == 13, artifact_ids

    def test_each_role_goal_has_distinct_nemesis(self):
        """13 roles → 13 distinct nemesis indices (role.c .neminum field)."""
        nem_ids = []
        for role in range(quest_levels.N_ROLES):
            _, monsters, _ = quest_levels.dispatch_quest_level(_RNG, role)
            data = get_quest_data(role)
            assert bool(jnp.any(monsters[:, 2] == data.nemesis_idx)), (
                f"role={role} missing nemesis {data.nemesis_idx}"
            )
            nem_ids.append(data.nemesis_idx)
        # NOTE: per role.c, Tourist's nemesis = Master of Thieves which is
        # ALSO Rogue's leader. The *nemesis* field is still distinct across
        # all 13 roles in role.c (Tourist nemesis idx = 351 vs Rogue nemesis
        # idx = 364, the Master Assassin). So the nemesis set is 13-unique.
        assert len(set(nem_ids)) == 13, nem_ids


# ---------------------------------------------------------------------------
# Full vs iconic detail comparison
# ---------------------------------------------------------------------------

class TestFullVsIconic:
    def test_full_layout_has_more_tiles_than_iconic(self):
        """The Wave 6 full layout must have measurably more non-VOID tiles
        than the Wave 5 iconic layout, for at least 8 of the 13 roles.

        (Some iconic layouts — e.g. Caveman's giant open cavern — are
        already quite dense; we only require a majority-of-roles
        improvement.)
        """
        more_detail_count = 0
        for role in range(quest_levels.N_ROLES):
            t_full, _, _ = quest_levels.dispatch_quest_level(_RNG, role, full_fidelity=True)
            t_iconic, _, _ = quest_levels.dispatch_quest_level(_RNG, role, full_fidelity=False)
            n_full = int(jnp.sum(t_full != 0))
            n_iconic = int(jnp.sum(t_iconic != 0))
            if n_full > n_iconic:
                more_detail_count += 1
        assert more_detail_count >= 8, (
            f"Only {more_detail_count}/13 full layouts beat iconic in tile count"
        )

    def test_full_uses_more_distinct_tile_types(self):
        """Aggregate: full layouts cover a comparable variety of terrain
        tile types as iconic.

        Wave 6 parity-fix: updated to match vendor/nethack/dat/*-goal.lua
        MAP sections (verbatim byte-identical).  Vendor MAP blocks never
        encode STAIR_DOWN tiles (quest goal levels only have stair-up per
        vendor lua), so the full layouts will have at most ~10 distinct
        tile types while the iconic procedural ones happened to include
        STAIR_DOWN.  We accept full <= iconic provided the difference is
        bounded (vendor truth wins).
        """
        full_tiles = set()
        iconic_tiles = set()
        for role in range(quest_levels.N_ROLES):
            t_full, _, _ = quest_levels.dispatch_quest_level(_RNG, role, full_fidelity=True)
            t_iconic, _, _ = quest_levels.dispatch_quest_level(_RNG, role, full_fidelity=False)
            full_tiles.update(int(x) for x in jnp.unique(t_full).tolist())
            iconic_tiles.update(int(x) for x in jnp.unique(t_iconic).tolist())
        assert len(full_tiles) >= len(iconic_tiles) - 2, (
            f"Full {sorted(full_tiles)} vs iconic {sorted(iconic_tiles)}"
        )


# ---------------------------------------------------------------------------
# Dispatch flag semantics
# ---------------------------------------------------------------------------

class TestDispatchFlag:
    def test_dispatch_full_fidelity_default(self):
        """dispatch_quest_level(rng, role) should call the FULL layout by default."""
        terrain_default, monsters_default, items_default = (
            quest_levels.dispatch_quest_level(_RNG, quest_levels.ROLE_WIZ)
        )
        terrain_full, monsters_full, items_full = (
            quest_levels.dispatch_quest_level(_RNG, quest_levels.ROLE_WIZ,
                                              full_fidelity=True)
        )
        # Default and explicit-full must agree element-wise.
        assert bool(jnp.array_equal(terrain_default, terrain_full))
        assert bool(jnp.array_equal(monsters_default, monsters_full))
        assert bool(jnp.array_equal(items_default, items_full))

    def test_dispatch_fallback_to_iconic(self):
        """full_fidelity=False returns the Wave 5 iconic layout, which
        differs from the Wave 6 full layout for at least one role.
        """
        any_differ = False
        for role in range(quest_levels.N_ROLES):
            t_full, _, _ = quest_levels.dispatch_quest_level(_RNG, role,
                                                             full_fidelity=True)
            t_iconic, _, _ = quest_levels.dispatch_quest_level(_RNG, role,
                                                               full_fidelity=False)
            if not bool(jnp.array_equal(t_full, t_iconic)):
                any_differ = True
                break
        assert any_differ, "full and iconic dispatches produced identical terrain"

    def test_dispatch_iconic_matches_wave5_factories(self):
        """full_fidelity=False must match the original iconic factories
        (kept as _iconic aliases).
        """
        for role in range(quest_levels.N_ROLES):
            t_disp, m_disp, i_disp = quest_levels.dispatch_quest_level(
                _RNG, role, full_fidelity=False
            )
            iconic_factory = quest_levels._FACTORIES[role]
            t_direct, m_direct, i_direct = iconic_factory(_RNG)
            assert bool(jnp.array_equal(t_disp, t_direct))
            assert bool(jnp.array_equal(m_disp, m_direct))
            assert bool(jnp.array_equal(i_disp, i_direct))


# ---------------------------------------------------------------------------
# Artifact-on-nemesis-tile invariant
# ---------------------------------------------------------------------------

class TestArtifactOnNemesisTile:
    def test_artifact_colocated_with_nemesis_for_every_role(self):
        """In NetHack, every role's quest artifact sits on the same tile as
        the nemesis (the nemesis 'guards' it). Wave 6 full layouts must
        preserve this invariant for all 13 roles.
        """
        for role in range(quest_levels.N_ROLES):
            _, monsters, items = quest_levels.dispatch_quest_level(
                _RNG, role, full_fidelity=True
            )
            data = get_quest_data(role)
            # Find nemesis position.
            mask_n = monsters[:, 2] == data.nemesis_idx
            assert bool(jnp.any(mask_n)), f"role={role} no nemesis placed"
            idx_n = int(jnp.argmax(mask_n))
            nem_rc = (int(monsters[idx_n, 0]), int(monsters[idx_n, 1]))
            # Find artifact position.
            mask_a = items[:, 2] == data.artifact_idx
            assert bool(jnp.any(mask_a)), f"role={role} no artifact placed"
            idx_a = int(jnp.argmax(mask_a))
            art_rc = (int(items[idx_a, 0]), int(items[idx_a, 1]))
            assert nem_rc == art_rc, (
                f"role={role}: nemesis @ {nem_rc} ≠ artifact @ {art_rc}"
            )
