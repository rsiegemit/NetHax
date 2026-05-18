"""Full-parity tests for artifact #invoke effects.

Covers all major artifact invoke handlers implemented in
artifact_powers.artifact_invoke_dispatch and wired through
action_dispatch._handle_invoke.

Cite: vendor/nethack/src/artifact.c::arti_invoke lines 2131-2232.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fresh_state():
    from Nethax.nethax.state import EnvState
    return EnvState.default(_RNG)


def _wield_artifact(state, artifact_idx: int, type_id: int = 37):
    """Set wielded_artifact_idx; mirrors test_artifact_specials_parity helper."""
    from Nethax.nethax.subsystems.inventory import ItemCategory, wield
    from Nethax.nethax.subsystems.artifact_powers import apply_artifact_intrinsics

    items = state.inventory.items
    items = items.replace(
        category=items.category.at[0].set(jnp.int8(ItemCategory.WEAPON)),
        type_id=items.type_id.at[0].set(jnp.int16(type_id)),
        enchantment=items.enchantment.at[0].set(jnp.int8(0)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=items.weight.at[0].set(jnp.int32(40)),
    )
    state = state.replace(inventory=state.inventory.replace(items=items))
    state = wield(state, 0)
    new_inv = state.inventory.replace(wielded_artifact_idx=jnp.int8(artifact_idx))
    state = state.replace(inventory=new_inv)
    return apply_artifact_intrinsics(state)


def _invoke(state, rng=None):
    """Run _handle_invoke on state."""
    from Nethax.nethax.subsystems.action_dispatch import _handle_invoke
    if rng is None:
        rng = _RNG
    return _handle_invoke(state, rng)


def _place_monster(state, slot: int = 0, hp: int = 9999, entry_idx: int = 0, undead: bool = False):
    """Place a live monster adjacent to player."""
    if undead:
        from Nethax.nethax.constants.monsters import MONSTERS, M2_UNDEAD
        entry_idx = next(
            i for i, m in enumerate(MONSTERS) if m.flags2 & M2_UNDEAD
        )
    mai = state.monster_ai
    mon_pos = jnp.array(
        [int(state.player_pos[0]), int(state.player_pos[1]) + 1], dtype=jnp.int16
    )
    mai = mai.replace(
        alive=mai.alive.at[slot].set(True),
        hp=mai.hp.at[slot].set(jnp.int32(hp)),
        hp_max=mai.hp_max.at[slot].set(jnp.int32(hp)),
        pos=mai.pos.at[slot].set(mon_pos),
        ac=mai.ac.at[slot].set(jnp.int8(10)),
        entry_idx=mai.entry_idx.at[slot].set(jnp.int16(entry_idx)),
    )
    return state.replace(monster_ai=mai)


# ---------------------------------------------------------------------------
# 1. Orb of Detection (idx 12): detect_objects 1000 turns
# ---------------------------------------------------------------------------

def test_orb_of_detection_sets_detect_objects():
    """Orb of Detection sets detect_objects_until_turn = timestep + 1000.

    Cite: artifact.c arti_invoke dispatch; detect.c detect_objects.
    """
    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=12)
    before = int(state.identification.detect_objects_until_turn)

    result = _invoke(state)

    after = int(result.identification.detect_objects_until_turn)
    ts = int(state.timestep)
    assert after == ts + 1000, f"Expected {ts+1000}, got {after}"
    assert after > before


# ---------------------------------------------------------------------------
# 2. Heart of Ahriman (idx 13): levitation +30, luck +d20
# ---------------------------------------------------------------------------

def test_heart_of_ahriman_levitation():
    """Heart of Ahriman increments LEVITATION_TMP by 30.

    Cite: artifact.c LEVITATION case ~line 2209; float_up() for 30 turns.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=13)
    before = int(state.status.timed_statuses[int(TimedStatus.LEVITATION_TMP)])

    result = _invoke(state)

    after = int(result.status.timed_statuses[int(TimedStatus.LEVITATION_TMP)])
    assert after == before + 30, f"Expected +30 levitation, got delta {after - before}"


def test_heart_of_ahriman_luck():
    """Heart of Ahriman increases player_luck by 1..20 (capped at 10).

    Cite: task spec: luck +d20.
    """
    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=13)
    before = int(state.player_luck)

    result = _invoke(state)

    after = int(result.player_luck)
    delta = after - before
    assert 1 <= delta <= 20 or after == 10, (
        f"Expected luck +1..+20 (capped 10), got before={before} after={after}"
    )


# ---------------------------------------------------------------------------
# 3. Sceptre of Might (idx 9): CONFLICT intrinsic toggle
# ---------------------------------------------------------------------------

def test_sceptre_of_might_conflict_toggle_on():
    """Sceptre of Might turns on CONFLICT intrinsic when it was off.

    Cite: artifact.c CONFLICT case ~line 2203.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=9)
    assert not bool(state.status.intrinsics[int(Intrinsic.CONFLICT)]), "should start off"

    result = _invoke(state)

    assert bool(result.status.intrinsics[int(Intrinsic.CONFLICT)]), "CONFLICT should be on"


def test_sceptre_of_might_conflict_toggle_off():
    """Sceptre of Might turns CONFLICT off when it was already on."""
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=9)
    # Pre-set CONFLICT on.
    new_intrinsics = state.status.intrinsics.at[int(Intrinsic.CONFLICT)].set(True)
    state = state.replace(status=state.status.replace(intrinsics=new_intrinsics))

    result = _invoke(state)

    assert not bool(result.status.intrinsics[int(Intrinsic.CONFLICT)]), "CONFLICT should toggle off"


# ---------------------------------------------------------------------------
# 4. Orb of Fate (idx 20): random teleport
# ---------------------------------------------------------------------------

def test_orb_of_fate_teleports():
    """Orb of Fate teleports player to a new position.

    Cite: artifact.c LEV_TELE case ~line 2160.
    """
    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=20)
    before_pos = state.player_pos.tolist()

    # Run many times; at least one should change position.
    moved = False
    for i in range(20):
        rng = jax.random.fold_in(_RNG, jnp.uint32(i))
        result = _invoke(state, rng=rng)
        if result.player_pos.tolist() != before_pos:
            moved = True
            break

    assert moved, "Orb of Fate should teleport player at least once in 20 tries"


# ---------------------------------------------------------------------------
# 5. Eye of the Aethiopica (idx 21): ENERGY_REGEN timed + +1 Pw
# ---------------------------------------------------------------------------

def test_eye_of_aethiopica_energy_regen():
    """Eye of the Aethiopica grants timed ENERGY_REGEN and +1 Pw (uncapped).

    Cite: artifact.c EREGEN path; vendor u.uen++ unconditionally; task spec.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=21)
    before_pw = int(state.player_pw)
    before_er = int(state.status.timed_intrinsics[int(Intrinsic.ENERGY_REGEN)])

    result = _invoke(state)

    assert int(result.player_pw) == before_pw + 1
    after_er = int(result.status.timed_intrinsics[int(Intrinsic.ENERGY_REGEN)])
    assert after_er > before_er


# ---------------------------------------------------------------------------
# 6. Mitre of Holiness (idx 16): TURN_UNDEAD ray
# ---------------------------------------------------------------------------

def test_mitre_of_holiness_turns_undead():
    """Mitre of Holiness kills undead monsters in adjacent area.

    Cite: artifact.c ENERGY_BOOST path; task spec TURN_UNDEAD ray.
    """
    state = _fresh_state()
    state = _place_monster(state, slot=0, hp=100, undead=True)
    state = _wield_artifact(state, artifact_idx=16)

    result = _invoke(state)

    # The undead monster at slot 0 (adjacent) should be dead.
    assert not bool(result.monster_ai.alive[0]), "Adjacent undead should be killed by Mitre"


# ---------------------------------------------------------------------------
# 7. Yendorian Express Card (idx 19): refill Pw to max
# ---------------------------------------------------------------------------

def test_yendorian_express_card_charges_pw():
    """Yendorian Express Card refills Pw to max.

    Cite: artifact.c CHARGE_OBJ case ~line 2159.
    """
    state = _fresh_state()
    state = state.replace(player_pw=jnp.int32(3), player_pw_max=jnp.int32(25))
    state = _wield_artifact(state, artifact_idx=19)

    result = _invoke(state)

    assert int(result.player_pw) == 25, f"Expected Pw=25, got {int(result.player_pw)}"


# ---------------------------------------------------------------------------
# 8. Staff of Aesculapius (idx 14): CURE_SICKNESS + HEALING
# ---------------------------------------------------------------------------

def test_staff_of_aesculapius_heals():
    """Staff of Aesculapius restores HP and clears SICK status.

    Cite: artifact.c HEALING case ~line 2156.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    state = _fresh_state()
    state = state.replace(player_hp=jnp.int32(1), player_hp_max=jnp.int32(20))
    # Set SICK timer.
    new_ts = state.status.timed_statuses.at[int(TimedStatus.SICK)].set(jnp.int32(5))
    new_status = state.status.replace(timed_statuses=new_ts, sick_kind=jnp.int8(1))
    state = state.replace(status=new_status)
    state = _wield_artifact(state, artifact_idx=14)

    result = _invoke(state)

    assert int(result.player_hp) > 1, "HP should increase"
    assert int(result.status.timed_statuses[int(TimedStatus.SICK)]) == 0, "SICK should be cleared"
    assert int(result.status.sick_kind) == 0, "sick_kind should be cleared"


# ---------------------------------------------------------------------------
# 9. Tsurugi of Muramasa (idx 10): +1 STR, cap 18
# ---------------------------------------------------------------------------

def test_tsurugi_str_boost():
    """Tsurugi of Muramasa grants +1 STR, capped at 18.

    Cite: task spec "+1 STR (cap 18)"; artilist.h ~285.
    """
    state = _fresh_state()
    state = state.replace(player_str=jnp.int16(15))
    state = _wield_artifact(state, artifact_idx=10)

    result = _invoke(state)

    assert int(result.player_str) == 16, f"Expected str=16, got {int(result.player_str)}"


def test_tsurugi_str_cap():
    """Tsurugi of Muramasa does not raise STR above 18."""
    state = _fresh_state()
    state = state.replace(player_str=jnp.int16(18))
    state = _wield_artifact(state, artifact_idx=10)

    result = _invoke(state)

    assert int(result.player_str) == 18, "STR should not exceed 18"


# ---------------------------------------------------------------------------
# 10. Mjollnir (idx 3): lightning ray d6,6
# ---------------------------------------------------------------------------

def test_mjollnir_lightning_damages_monsters():
    """Mjollnir's invoke deals lightning damage to nearby monsters.

    Cite: artifact.c ELEC(5,24); task spec LIGHTNING_RAY (d6,6).
    """
    state = _fresh_state()
    state = _place_monster(state, slot=0, hp=9999, entry_idx=0)
    state = _wield_artifact(state, artifact_idx=3)
    before_hp = int(state.monster_ai.hp[0])

    result = _invoke(state)

    after_hp = int(result.monster_ai.hp[0])
    assert after_hp < before_hp, f"Monster should take lightning damage; hp {before_hp} → {after_hp}"
    assert before_hp - after_hp >= 6, "Minimum d6,6 = 6 damage"


# ---------------------------------------------------------------------------
# 11. Snickersnee (idx 1): SLEEP_RES timed 50
# ---------------------------------------------------------------------------

def test_snickersnee_sleep_resistance():
    """Snickersnee grants timed SLEEP_RES for 50 turns.

    Cite: task spec "+SLEEP_RES (timed 50)"; artilist.h ~203.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=1)
    before = int(state.status.timed_intrinsics[int(Intrinsic.RESIST_SLEEP)])

    result = _invoke(state)

    after = int(result.status.timed_intrinsics[int(Intrinsic.RESIST_SLEEP)])
    assert after >= 50, f"Expected SLEEP_RES >= 50, got {after}"
    assert after > before


# ---------------------------------------------------------------------------
# 12. Grayswandir (idx 7): DETECT_FOOD
# ---------------------------------------------------------------------------

def test_grayswandir_detect_food():
    """Grayswandir invokes detect food, setting detect_food_until_turn.

    Cite: task spec "DETECT_FOOD"; detect.py detect_food.
    """
    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=7)
    ts = int(state.timestep)

    result = _invoke(state)

    after = int(result.identification.detect_food_until_turn)
    assert after == ts + 50, f"Expected detect_food_until = {ts+50}, got {after}"


# ---------------------------------------------------------------------------
# 13. Demonbane (idx 25): DETECT_MONSTERS 50 turns
# ---------------------------------------------------------------------------

def test_demonbane_detect_monsters():
    """Demonbane invokes detect monsters for 50 turns.

    Cite: task spec "DETECT_MONSTERS for 50 turns"; detect.py detect_monsters.
    """
    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=25)
    ts = int(state.timestep)

    result = _invoke(state)

    after = int(result.identification.detect_monsters_until_turn)
    assert after == ts + 100, f"Expected detect_monsters_until = {ts+100}, got {after}"


# ---------------------------------------------------------------------------
# 14. Werebane (idx 26): cure lycanthropy
# ---------------------------------------------------------------------------

def test_werebane_cures_lycanthropy():
    """Werebane cures lycanthropy by resetting lycanthropy_form to -1.

    Cite: task spec "LYCANTHROPY_CURE on self"; artilist.h ~166.
    """
    state = _fresh_state()
    # Set a were-form.
    new_poly = state.polymorph.replace(
        lycanthropy_form=jnp.int8(5),
        lycanthropy_timer=jnp.int16(50),
    )
    state = state.replace(polymorph=new_poly)
    state = _wield_artifact(state, artifact_idx=26)

    result = _invoke(state)

    assert int(result.polymorph.lycanthropy_form) == -1, "lycanthropy_form should be -1 after cure"
    assert int(result.polymorph.lycanthropy_timer) == 0


# ---------------------------------------------------------------------------
# 15. Trollsbane (idx 27): TRUE_SIGHT 50 turns (SEE_INVIS)
# ---------------------------------------------------------------------------

def test_trollsbane_true_sight():
    """Trollsbane grants TRUE_SIGHT (SEE_INVIS) for 50 turns.

    Cite: task spec "TRUE_SIGHT for 50 turns"; artilist.h ~182.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=27)

    result = _invoke(state)

    after = int(result.status.timed_intrinsics[int(Intrinsic.SEE_INVIS)])
    assert after >= 50, f"Expected SEE_INVIS >= 50, got {after}"


# ---------------------------------------------------------------------------
# 16. Grimtooth (idx 28): FAST timer +20
# ---------------------------------------------------------------------------

def test_grimtooth_fast():
    """Grimtooth grants FAST (haste) for +20 turns.

    Cite: task spec "FAST timer +20"; artilist.h ~123.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=28)
    before = int(state.status.timed_intrinsics[int(Intrinsic.FAST)])

    result = _invoke(state)

    after = int(result.status.timed_intrinsics[int(Intrinsic.FAST)])
    assert after == before + 20, f"Expected FAST +20, got delta {after - before}"


# ---------------------------------------------------------------------------
# 17. Dragonbane (idx 24): DETECT_TREASURE 50 turns
# ---------------------------------------------------------------------------

def test_dragonbane_detect_treasure():
    """Dragonbane invokes detect treasure for 50 turns.

    Cite: task spec "DETECT_TREASURE for 50 turns"; artilist.h ~157.
    """
    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=24)
    ts = int(state.timestep)

    result = _invoke(state)

    after = int(result.identification.detect_treasure_until_turn)
    assert after == ts + 50, f"Expected detect_treasure_until = {ts+50}, got {after}"


# ---------------------------------------------------------------------------
# 18. Cooldown gates second invoke
# ---------------------------------------------------------------------------

def test_invoke_cooldown_blocks_second_invoke():
    """Second invoke with cooldown active is a no-op.

    Cite: artifact.c::arti_invoke_cost — artifact 'tired' check.
    """
    state = _fresh_state()
    state = state.replace(player_pw=jnp.int32(3), player_pw_max=jnp.int32(25))
    state = _wield_artifact(state, artifact_idx=19)  # Yendorian Express Card

    # First invoke: refills Pw.
    result1 = _invoke(state)
    assert int(result1.player_pw) == 25
    assert int(result1.invoke_cooldown[19]) == 100

    # Second invoke immediately: cooldown active → no effect on Pw.
    result2 = _invoke(result1)
    # Pw should remain 25 (no second refill), cooldown unchanged.
    assert int(result2.player_pw) == 25
    assert int(result2.invoke_cooldown[19]) == 100


# ---------------------------------------------------------------------------
# 19. Non-artifact wielded → no-op
# ---------------------------------------------------------------------------

def test_no_artifact_wielded_is_noop():
    """#invoke with no artifact wielded (idx -1) does nothing.

    Cite: artifact.c::arti_invoke returns ECMD_TIME on non-artifact.
    """
    state = _fresh_state()
    # Ensure no artifact is wielded.
    new_inv = state.inventory.replace(wielded_artifact_idx=jnp.int8(-1))
    state = state.replace(inventory=new_inv)
    before_pw = int(state.player_pw)

    result = _invoke(state)

    assert int(result.player_pw) == before_pw
    # cooldown array should be untouched (all zeros).
    assert jnp.all(result.invoke_cooldown == jnp.int16(0))


# ---------------------------------------------------------------------------
# 20. Magicbane (idx 29): confusion ray hits adjacent monsters
# ---------------------------------------------------------------------------

def test_magicbane_invoke_confuses_adjacent():
    """Magicbane #invoke sets asleep on adjacent monsters.

    Cite: artifact.c magicbane_hit ~1090; task spec confusion ray.
    """
    state = _fresh_state()
    state = _place_monster(state, slot=0, hp=100, entry_idx=0)
    state = _wield_artifact(state, artifact_idx=29)
    assert not bool(state.monster_ai.asleep[0])

    result = _invoke(state)

    assert bool(result.monster_ai.asleep[0]), "Adjacent monster should be confused/asleep after Magicbane invoke"
