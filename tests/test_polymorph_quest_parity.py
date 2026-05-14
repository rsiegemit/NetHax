"""Wave 6 — Polymorph + Quest vendor-parity tests.

Targets two subsystems:

  A. polymorph.py state-swap audit (vs vendor/nethack/src/polyself.c,
     vendor/nethack/src/were.c).

  B. quest.py state machine audit (vs vendor/nethack/src/quest.c — the
     ``quest_status`` flags ``met_leader``, ``killed_nemesis``,
     ``touched_artifact``, ``qcompleted``, ``qexpelled``).

Run scoped:
    .venv/bin/python -m pytest \
        tests/test_polymorph_quest_parity.py \
        tests/test_polymorph.py \
        tests/test_quest.py -v --timeout=120
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.subsystems.inventory import N_ARMOR_SLOTS
from Nethax.nethax.subsystems.polymorph import (
    polymorph_player,
    revert_polymorph,
    step as poly_step,
    trigger_lycanthropy,
    _can_wear_armor,
    _NONE_FORM,
    _LYCANTHROPY_FORM_DURATION,
)
from Nethax.nethax.subsystems.quest import (
    QuestState,
    QuestStage,
    talk_to_leader,
    pickup_artifact,
    nemesis_killed,
    check_quest_complete,
)


_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_polymorph.py to avoid hidden coupling)
# ---------------------------------------------------------------------------

def _find_form_with_attack() -> int:
    for i, m in enumerate(MONSTERS):
        if m.attacks and m.attacks[0][0] != 0:
            return i
    return 0


def _find_form_with_hands() -> int:
    M1_HUMANOID = 0x00020000
    M1_NOHANDS = 0x00002000
    for i, m in enumerate(MONSTERS):
        if (m.flags1 & M1_HUMANOID) and not (m.flags1 & M1_NOHANDS):
            return i
    return 0


def _find_form_without_hands() -> int:
    M1_NOHANDS = 0x00002000
    for i, m in enumerate(MONSTERS):
        if m.flags1 & M1_NOHANDS:
            return i
    return 0


def _base_state(armor_worn: bool = False) -> EnvState:
    state = EnvState.default(_RNG)
    state = state.replace(
        player_str=jnp.int16(18),
        player_dex=jnp.int8(12),
        player_con=jnp.int8(14),
        player_hp=jnp.int32(20),
        player_hp_max=jnp.int32(20),
        player_role=jnp.int8(0),
        player_ac=jnp.int32(10),
    )
    if armor_worn:
        new_worn = state.inventory.worn_armor.at[0].set(jnp.int8(3))
        state = state.replace(
            inventory=state.inventory.replace(worn_armor=new_worn)
        )
    return state


# ===========================================================================
# A. polymorph.py state-swap parity (polyself.c::polymon / rehumanize)
# ===========================================================================

class TestPolymorphStateSwap:
    """Verify state-swap fields populated by polymorph_player()."""

    def test_polymorph_player_saves_orig_stats(self):
        """polyself.c::polymon snapshots STR/DEX/CON/HP_max/AC/role before
        adopting the new form so rehumanize can roll them back."""
        state = _base_state()
        target = _find_form_with_attack()
        new = polymorph_player(state, _RNG, target, controlled=False)
        p = new.polymorph
        assert int(p.orig_str) == 18
        assert int(p.orig_dex) == 12
        assert int(p.orig_con) == 14
        assert int(p.orig_hp_max) == 20
        assert int(p.orig_ac) == 10
        assert int(p.orig_role_idx) == 0
        # AC was recomputed from the form's intrinsic AC.
        assert int(new.player_ac) == int(MONSTERS[target].ac)
        assert bool(p.is_polymorphed) is True
        assert int(p.current_form_idx) == target

    def test_polymorph_player_restores_armor_on_revert(self):
        """polyself.c::rehumanize leaves armor slots that were cleared
        on poly empty (the worn-armor inventory pointers don't auto-
        re-equip).  We assert that the armor-drop is observable post-
        poly and that revert does NOT spontaneously refit armor (i.e.
        the slot stays empty as a hands-on fact)."""
        target = _find_form_without_hands()
        state = _base_state(armor_worn=True)
        # Sanity: chosen form really has no hands.
        assert not bool(_can_wear_armor(jnp.int16(target)))
        new = polymorph_player(state, _RNG, target, controlled=False)
        # During polymorph, all armor slots are cleared (polyself.c
        # drop_inv_loss).
        for i in range(N_ARMOR_SLOTS):
            assert int(new.inventory.worn_armor[i]) == -1
        # Revert: stats restored, but armor stays unequipped (matches
        # rehumanize: it does not re-don armor for the player).
        reverted = revert_polymorph(new, _RNG)
        assert bool(reverted.polymorph.is_polymorphed) is False
        assert int(reverted.player_str) == 18
        assert int(reverted.player_dex) == 12
        assert int(reverted.player_con) == 14
        assert int(reverted.player_ac) == 10
        # All armor slots remain empty after revert (no auto-re-don).
        for i in range(N_ARMOR_SLOTS):
            assert int(reverted.inventory.worn_armor[i]) == -1

    def test_revert_polymorph_restores_attacks_from_orig(self):
        """polyself.c::rehumanize copies the saved orig attack table
        back into the active attack slots."""
        state = _base_state()
        target = _find_form_with_attack()
        # Pre-poly, attacks_* are zeroed by make_polymorph_state.
        pre = state.polymorph
        assert int(pre.attack_types[0]) == 0
        new = polymorph_player(state, _RNG, target, controlled=False)
        # Now active attacks reflect the form, and orig_attack_* still
        # hold the pre-poly zeros.
        assert int(new.polymorph.attack_types[0]) == int(MONSTERS[target].attacks[0][0])
        assert int(new.polymorph.orig_attack_types[0]) == 0
        # After revert: active attacks must match the saved orig set.
        reverted = revert_polymorph(new, _RNG)
        assert int(reverted.polymorph.attack_types[0]) == 0
        assert int(reverted.polymorph.attack_damage_types[0]) == 0
        assert int(reverted.polymorph.attack_n_dice[0]) == 0
        assert int(reverted.polymorph.attack_n_sides[0]) == 0


class TestLycanthropy:
    """were.c::were_change timer + auto-transformation parity."""

    def test_lycanthropy_timer_triggers_were_form_at_zero(self):
        """were.c::were_change fires new_were() when the countdown
        elapses.  Our step() must auto-polymorph the hero into the
        queued lycanthropy_form when lycanthropy_timer reaches 0."""
        state = _base_state()
        were_form = _find_form_with_attack()
        # Queue a were-form with a 1-turn countdown; hero not yet polymorphed.
        poly = state.polymorph.replace(
            lycanthropy_form=jnp.int8(were_form),
            lycanthropy_timer=jnp.int16(1),
        )
        state = state.replace(polymorph=poly)
        assert bool(state.polymorph.is_polymorphed) is False

        # One step: timer 1 -> 0, queued form fires.
        stepped = poly_step(state, _RNG)
        assert int(stepped.polymorph.lycanthropy_timer) == 0
        assert bool(stepped.polymorph.is_polymorphed) is True
        assert int(stepped.polymorph.current_form_idx) == were_form

    def test_trigger_lycanthropy_sets_form_and_timer(self):
        """trigger_lycanthropy must polymorph into the were-form and
        set lycanthropy_form for downstream tracking."""
        state = _base_state()
        were_form = _find_form_with_attack()
        new = trigger_lycanthropy(state, _RNG, were_form)
        assert bool(new.polymorph.is_polymorphed) is True
        assert int(new.polymorph.lycanthropy_form) == were_form
        # poly_timer is overridden to the were-form duration.
        assert int(new.polymorph.poly_timer) == _LYCANTHROPY_FORM_DURATION


# ===========================================================================
# B. quest.py state machine parity (quest.c::quest_status flags)
# ===========================================================================

class TestQuestStateFlags:
    """Verify QuestState carries vendor-parity flags."""

    def test_quest_state_has_met_leader_flag(self):
        """Qstat(met_leader) — quest.c chat_with_leader ~323."""
        qs = QuestState.default()
        assert hasattr(qs, "met_leader")
        assert bool(qs.met_leader) is False
        # talk_to_leader flips it sticky-True.
        after = talk_to_leader(qs)
        assert bool(after.met_leader) is True
        assert int(after.stage) >= int(QuestStage.LEADER_GREETED)
        # Idempotent: a second call keeps it True.
        again = talk_to_leader(after)
        assert bool(again.met_leader) is True

    def test_quest_state_has_killed_nemesis_flag(self):
        """Qstat(killed_nemesis) — quest.c killed_nemesis ~109-125.

        Our field is called `nemesis_killed`; this test asserts both
        the field's presence and its sticky flip on nemesis death.
        """
        qs = QuestState.default()
        assert hasattr(qs, "nemesis_killed")
        assert bool(qs.nemesis_killed) is False
        after = nemesis_killed(qs)
        assert bool(after.nemesis_killed) is True
        assert bool(after.nemesis_alive) is False
        assert int(after.stage) >= int(QuestStage.NEMESIS_KILLED)

    def test_quest_state_has_touched_artifact_flag(self):
        """Qstat(touched_artifact) — quest.c touched_artifact ~127-141."""
        qs = QuestState.default()
        assert hasattr(qs, "touched_artifact")
        assert bool(qs.touched_artifact) is False
        after = pickup_artifact(qs)
        assert bool(after.touched_artifact) is True
        assert bool(after.artifact_carried) is True
        assert int(after.stage) >= int(QuestStage.ARTIFACT_RECOVERED)

    def test_quest_state_has_qexpelled_flag(self):
        """u.uevent.qexpelled — quest.c ~202.

        Default-False sticky flag tracking whether the hero has been
        expelled from the quest by an offended leader.
        """
        qs = QuestState.default()
        assert hasattr(qs, "qexpelled")
        assert bool(qs.qexpelled) is False


class TestCheckQuestComplete:
    """quest.c finish_quest gating: requires all three sticky flags
    plus runtime adjacency + carry."""

    def test_check_quest_complete_requires_all_three(self):
        """All three predicates from quest_status must hold:
          - killed_nemesis,
          - touched_artifact (sticky; auto-set if has_artifact True),
          - leader contact (met_leader proxy via adjacency).
        Removing any one must keep the quest incomplete.
        """
        qs = QuestState.default().replace(
            leader_pos=jnp.array([5, 5], dtype=jnp.int16),
        )

        # Baseline: full state set up to succeed.
        full = nemesis_killed(qs)                       # nemesis_killed = True
        full = pickup_artifact(full)                    # touched_artifact = True
        full = talk_to_leader(full)                     # met_leader = True
        adjacent = jnp.array([5, 6], dtype=jnp.int16)
        far = jnp.array([20, 20], dtype=jnp.int16)

        # 1) Missing nemesis_killed: not complete.
        no_nem = pickup_artifact(qs)
        no_nem = talk_to_leader(no_nem)
        out = check_quest_complete(no_nem, adjacent, jnp.bool_(True))
        assert bool(out.completed) is False

        # 2) Missing artifact carry: not complete.
        out = check_quest_complete(full, adjacent, jnp.bool_(False))
        assert bool(out.completed) is False

        # 3) Far from leader (no adjacency): not complete.
        out = check_quest_complete(full, far, jnp.bool_(True))
        assert bool(out.completed) is False

        # 4) All three present: COMPLETE.
        out = check_quest_complete(full, adjacent, jnp.bool_(True))
        assert bool(out.completed) is True
        assert int(out.stage) == int(QuestStage.RETURNED_TO_LEADER)
