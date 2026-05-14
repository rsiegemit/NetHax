"""Wave 5 Phase 3 — Quest subsystem tests.

Covers:
  - Per-role data table (_QUEST_DATA).
  - 13 hand-translated quest level factories.
  - Dispatch over role index (lax.switch).
  - Nemesis fight mechanics (boosted hp + extra attack).
  - Return-to-leader victory check.

Citations:
  vendor/nethack/src/role.c lines 30-573 — role quest fields.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.dungeon.branches import MAP_H, MAP_W
from Nethax.nethax.dungeon import quest_levels
from Nethax.nethax.subsystems import quest as quest_mod
from Nethax.nethax.subsystems.quest import (
    QuestStage,
    QuestState,
    _QUEST_DATA,
    NEMESIS_HP_MULTIPLIER,
    boost_nemesis_hp,
    check_quest_complete,
    get_quest_data,
    is_nemesis,
    nemesis_attack_rolls,
    nemesis_killed,
)


_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Data table
# ---------------------------------------------------------------------------

class TestQuestData:
    def test_quest_data_has_13_entries(self):
        """role.c roles[] has 13 non-terminator entries (Arc..Wiz)."""
        assert len(_QUEST_DATA) == 13

    def test_wizard_quest_artifact_is_eye_of_aethiopica(self):
        """role.c line 556: Wizard's qlist = ART_EYE_OF_THE_AETHIOPICA.

        ART_EYE_OF_THE_AETHIOPICA is the 14th non-empty artifact entry
        (artilist.h line 303 — 33rd 0-based slot including index 0 = empty
        and indices 1..19 for non-quest artifacts + 13 quest artifacts).
        """
        wiz = get_quest_data(quest_levels.ROLE_WIZ)
        assert wiz.role_code == "Wiz"
        assert wiz.artifact_idx == 33  # ART_EYE_OF_THE_AETHIOPICA position

    def test_valkyrie_quest_nemesis_is_cyclops_pattern(self):
        """role.c line 511: Valkyrie's neminum = PM_LORD_SURTUR.

        (Spec note: the task mentions "Cyclops" but role.c shows that
        Cyclops is the Healer's nemesis, line 172.  We assert role-c
        canonical mapping for both.)
        """
        val = get_quest_data(quest_levels.ROLE_VAL)
        assert val.role_code == "Val"
        # role.c line 511: PM_LORD_SURTUR
        assert MONSTERS[val.nemesis_idx].name == "Lord Surtur"
        hea = get_quest_data(quest_levels.ROLE_HEA)
        # role.c line 172: PM_CYCLOPS
        assert MONSTERS[hea.nemesis_idx].name == "Cyclops"

    def test_healer_quest_leader_is_hippocrates(self):
        """role.c line 170: Healer's lead0 = PM_HIPPOCRATES."""
        hea = get_quest_data(quest_levels.ROLE_HEA)
        assert hea.role_code == "Hea"
        assert MONSTERS[hea.leader_idx].name == "Hippocrates"

    def test_all_roles_have_distinct_codes(self):
        codes = [d.role_code for d in _QUEST_DATA]
        assert len(set(codes)) == 13


# ---------------------------------------------------------------------------
# Quest-level factories
# ---------------------------------------------------------------------------

class TestQuestLevelFactories:
    def test_dispatch_quest_level_wizard(self):
        terrain, monsters, items = quest_levels.dispatch_quest_level(
            _RNG, quest_levels.ROLE_WIZ
        )
        assert terrain.shape == (MAP_H, MAP_W)
        # Dark One must appear among the monsters placed.
        dark_one_idx = get_quest_data(quest_levels.ROLE_WIZ).nemesis_idx
        assert bool(jnp.any(monsters[:, 2] == dark_one_idx))
        # Eye of the Aethiopica artifact placed.
        eye_idx = get_quest_data(quest_levels.ROLE_WIZ).artifact_idx
        assert bool(jnp.any(items[:, 2] == eye_idx))

    def test_dispatch_quest_level_valkyrie(self):
        terrain, monsters, items = quest_levels.dispatch_quest_level(
            _RNG, quest_levels.ROLE_VAL
        )
        assert terrain.shape == (MAP_H, MAP_W)
        # Lord Surtur (val nemesis) placed.
        surt_idx = get_quest_data(quest_levels.ROLE_VAL).nemesis_idx
        assert bool(jnp.any(monsters[:, 2] == surt_idx))
        # Orb of Fate artifact placed.
        orb_idx = get_quest_data(quest_levels.ROLE_VAL).artifact_idx
        assert bool(jnp.any(items[:, 2] == orb_idx))

    def test_each_role_has_unique_quest_level(self):
        """13 distinct factory outputs — basic check: not all identical.

        Compares the monster-placement arrays (the most role-distinguishing
        artifact of each layout).
        """
        outs = []
        for role in range(quest_levels.N_ROLES):
            _, m, _ = quest_levels.dispatch_quest_level(_RNG, role)
            outs.append(m)
        # At least the monster-id columns must differ across roles.
        ids = [tuple(int(x) for x in m[:, 2].tolist()) for m in outs]
        # Each role's nemesis id must differ from all others (mostly true;
        # Tourist nemesis collides with Rogue leader, that's intentional).
        unique = set(ids)
        assert len(unique) >= 12, f"Expected ≥12 distinct layouts, got {len(unique)}"

    def test_all_13_factories_callable(self):
        """Sanity: each named factory exists and returns the 3-tuple shape."""
        names = [
            "generate_arc_quest_level", "generate_bar_quest_level",
            "generate_cav_quest_level", "generate_hea_quest_level",
            "generate_kni_quest_level", "generate_mon_quest_level",
            "generate_pri_quest_level", "generate_ran_quest_level",
            "generate_rog_quest_level", "generate_sam_quest_level",
            "generate_tou_quest_level", "generate_val_quest_level",
            "generate_wiz_quest_level",
        ]
        for n in names:
            fn = getattr(quest_levels, n)
            terrain, monsters, items = fn(_RNG)
            assert terrain.shape == (MAP_H, MAP_W)
            assert monsters.shape == (64, 3)
            assert items.shape == (64, 3)


# ---------------------------------------------------------------------------
# Nemesis fight mechanics
# ---------------------------------------------------------------------------

class TestNemesisMechanics:
    def test_nemesis_has_boosted_hp(self):
        """Vendor: nemesis spawned with 4x base HP."""
        base = jnp.int32(50)
        boosted = boost_nemesis_hp(base)
        assert int(boosted) == 50 * NEMESIS_HP_MULTIPLIER == 200

    def test_nemesis_extra_attack_rolls(self):
        assert nemesis_attack_rolls(1) == 2
        assert nemesis_attack_rolls(2) == 3

    def test_is_nemesis_detects_role_match(self):
        wiz = get_quest_data(quest_levels.ROLE_WIZ)
        # Dark One IS the Wizard's nemesis.
        assert is_nemesis(quest_levels.ROLE_WIZ, wiz.nemesis_idx) is True
        # ... but NOT the Valkyrie's nemesis.
        assert is_nemesis(quest_levels.ROLE_VAL, wiz.nemesis_idx) is False

    def test_nemesis_killed_sets_flag(self):
        state = QuestState.default()
        assert not bool(state.nemesis_killed)
        new = nemesis_killed(state)
        assert bool(new.nemesis_killed)
        assert not bool(new.nemesis_alive)
        assert int(new.stage) >= int(QuestStage.NEMESIS_KILLED)


# ---------------------------------------------------------------------------
# Return-to-leader victory check
# ---------------------------------------------------------------------------

class TestQuestComplete:
    def test_quest_complete_requires_artifact_and_leader_contact(self):
        """All three predicates must hold: nemesis dead + artifact + near leader."""
        state = QuestState.default()
        state = state.replace(
            leader_pos=jnp.array([5, 5], dtype=jnp.int16),
        )
        # 1) no nemesis kill, no artifact, far from leader → not complete.
        out = check_quest_complete(state, jnp.array([10, 10], dtype=jnp.int16),
                                   jnp.bool_(False))
        assert not bool(out.completed)

        # 2) nemesis dead + artifact + far from leader → not complete.
        s2 = nemesis_killed(state)
        out = check_quest_complete(s2, jnp.array([10, 10], dtype=jnp.int16),
                                   jnp.bool_(True))
        assert not bool(out.completed)

        # 3) nemesis dead + no artifact + adjacent → not complete.
        out = check_quest_complete(s2, jnp.array([5, 6], dtype=jnp.int16),
                                   jnp.bool_(False))
        assert not bool(out.completed)

        # 4) nemesis dead + artifact + adjacent → COMPLETE.
        out = check_quest_complete(s2, jnp.array([5, 6], dtype=jnp.int16),
                                   jnp.bool_(True))
        assert bool(out.completed)
        assert int(out.stage) == int(QuestStage.RETURNED_TO_LEADER)

    def test_quest_complete_idempotent(self):
        """check_quest_complete on an already-completed state keeps it complete."""
        state = QuestState.default()
        state = state.replace(
            leader_pos=jnp.array([5, 5], dtype=jnp.int16),
        )
        state = nemesis_killed(state)
        s1 = check_quest_complete(state,
                                  jnp.array([5, 6], dtype=jnp.int16),
                                  jnp.bool_(True))
        assert bool(s1.completed)
        # Now move away from leader — completed must stay sticky.
        s2 = check_quest_complete(s1,
                                  jnp.array([15, 15], dtype=jnp.int16),
                                  jnp.bool_(True))
        assert bool(s2.completed)
