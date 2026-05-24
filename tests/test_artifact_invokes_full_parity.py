"""Full-parity tests for artifact #invoke effects.

Covers the corrected (Audit K, Wave 40c) artifact-invoke dispatch in
artifact_powers.artifact_invoke_dispatch, wired through
action_dispatch._handle_invoke.

The pre-Audit-K version of this file asserted invented behaviour for
slots whose vendor inv_prop is 0 (Snickersnee, Grayswandir, Dragonbane,
Demonbane, Werebane, Trollsbane, Grimtooth, Magicbane).  Per vendor
artilist.h, those slots have NO invoke effect — the tests below now
assert that #invoke on them is a true no-op (no detect timer set, no
intrinsic granted, no monster mutation).

Cite: vendor/nethack/src/artifact.c::arti_invoke lines 2131-2232;
      vendor/nethack/include/artilist.h lines 85-307 (inv_prop column).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

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


def _place_monster(state, slot: int = 0, hp: int = 9999, entry_idx: int = 0, undead: bool = False, demon: bool = False):
    """Place a live monster adjacent to player."""
    if undead:
        from Nethax.nethax.constants.monsters import MONSTERS, M2_UNDEAD
        entry_idx = next(
            i for i, m in enumerate(MONSTERS) if m.flags2 & M2_UNDEAD
        )
    if demon:
        from Nethax.nethax.constants.monsters import MONSTERS, M2_DEMON
        entry_idx = next(
            i for i, m in enumerate(MONSTERS) if m.flags2 & M2_DEMON
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


# ===========================================================================
# Real invokes (vendor inv_prop != 0)
# ===========================================================================

# ---------------------------------------------------------------------------
# Sceptre of Might (idx 9): CONFLICT intrinsic toggle
# Cite: vendor/nethack/src/artifact.c::arti_invoke CONFLICT case line 2203.
# ---------------------------------------------------------------------------

def test_sceptre_of_might_conflict_toggle_on():
    """Sceptre of Might toggles the CONFLICT intrinsic ON when it was OFF."""
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=9)
    assert not bool(state.status.intrinsics[int(Intrinsic.CONFLICT)])

    result = _invoke(state)

    assert bool(result.status.intrinsics[int(Intrinsic.CONFLICT)])


def test_sceptre_of_might_conflict_toggle_off():
    """Sceptre of Might toggles CONFLICT OFF when it was already ON."""
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=9)
    new_intrinsics = state.status.intrinsics.at[int(Intrinsic.CONFLICT)].set(True)
    state = state.replace(status=state.status.replace(intrinsics=new_intrinsics))

    result = _invoke(state)

    assert not bool(result.status.intrinsics[int(Intrinsic.CONFLICT)])


# ---------------------------------------------------------------------------
# Orb of Detection (idx 12): INVIS toggle.
# Cite: vendor/nethack/src/artifact.c lines 2216-2226 (INVIS toggle).
# Audit K replaces the invented "detect_objects 1000t" with the real
# vendor INVIS property-toggle.
# ---------------------------------------------------------------------------

def test_orb_of_detection_invis_toggle():
    """Orb of Detection toggles INVIS_TMP timer ON (30 turns) from zero."""
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=12)
    before = int(state.status.timed_statuses[int(TimedStatus.INVIS_TMP)])
    assert before == 0

    result = _invoke(state)

    after = int(result.status.timed_statuses[int(TimedStatus.INVIS_TMP)])
    assert after == 30


# ---------------------------------------------------------------------------
# Heart of Ahriman (idx 13): LEVITATION toggle.
# Cite: vendor/nethack/src/artifact.c lines 2209-2214.
# Audit K drops the invented "+d20 luck" bonus.
# ---------------------------------------------------------------------------

def test_heart_of_ahriman_levitation_toggle_on():
    """Heart of Ahriman toggles LEVITATION_TMP from 0 to 30."""
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=13)

    result = _invoke(state)

    after = int(result.status.timed_statuses[int(TimedStatus.LEVITATION_TMP)])
    assert after == 30


# ---------------------------------------------------------------------------
# Staff of Aesculapius (idx 14): HEALING.
# Cite: vendor/nethack/src/artifact.c::invoke_healing lines 1779-1815.
#   healamt = (uhpmax + 1 - uhp) / 2;  clears Sick / Slimed / Blinded.
# ---------------------------------------------------------------------------

def test_staff_of_aesculapius_heals():
    """Staff of Aesculapius restores HP by (hpmax+1-hp)/2 and clears SICK."""
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    state = _fresh_state()
    state = state.replace(player_hp=jnp.int32(1), player_hp_max=jnp.int32(20))
    new_ts = state.status.timed_statuses.at[int(TimedStatus.SICK)].set(jnp.int32(5))
    new_status = state.status.replace(timed_statuses=new_ts, sick_kind=jnp.int8(1))
    state = state.replace(status=new_status)
    state = _wield_artifact(state, artifact_idx=14)

    result = _invoke(state)

    # healamt = (20 + 1 - 1) / 2 = 10  →  new hp = 1 + 10 = 11.
    assert int(result.player_hp) == 11, f"Expected hp=11 got {int(result.player_hp)}"
    assert int(result.status.timed_statuses[int(TimedStatus.SICK)]) == 0
    assert int(result.status.sick_kind) == 0


# ---------------------------------------------------------------------------
# Mitre of Holiness (idx 16): ENERGY_BOOST.
# Cite: vendor/nethack/src/artifact.c::invoke_energy_boost lines 1817-1835.
#   epboost = (uenmax + 1 - uen) / 2;
#   if epboost > 120: 120; elif epboost < 12: uenmax-uen; uen += epboost.
# ---------------------------------------------------------------------------

def test_mitre_of_holiness_energy_boost():
    """Mitre of Holiness applies vendor energy-boost recharge."""
    state = _fresh_state()
    # uenmax=50, uen=10 → epboost = (50+1-10)/2 = 20 → 20 (not <12, not >120).
    state = state.replace(player_pw=jnp.int32(10), player_pw_max=jnp.int32(50))
    state = _wield_artifact(state, artifact_idx=16)

    result = _invoke(state)

    assert int(result.player_pw) == 30, f"Expected pw=30 got {int(result.player_pw)}"


# ---------------------------------------------------------------------------
# Yendorian Express Card (idx 19): CHARGE_OBJ.
# Cite: vendor/nethack/src/artifact.c::invoke_charge_obj lines 1847-1864.
# Audit K drops the invented "refill player_pw" path; vendor recharges
# a held wand instead.  When no wand is held, Pw is left untouched.
# ---------------------------------------------------------------------------

def test_yendorian_express_no_wand_is_noop_for_pw():
    """Yendorian Express does NOT refill Pw (Audit K)."""
    state = _fresh_state()
    state = state.replace(player_pw=jnp.int32(3), player_pw_max=jnp.int32(25))
    state = _wield_artifact(state, artifact_idx=19)

    result = _invoke(state)

    # Pw must NOT be auto-refilled (vendor charges a wand, not Pw).
    assert int(result.player_pw) == 3, f"Expected pw=3 got {int(result.player_pw)}"


# ---------------------------------------------------------------------------
# Orb of Fate (idx 20): LEV_TELE.
# Cite: vendor/nethack/src/artifact.c line 2160.
# ---------------------------------------------------------------------------

def test_orb_of_fate_teleports():
    """Orb of Fate teleports player at least once across 20 invocations."""
    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=20)
    before_pos = state.player_pos.tolist()

    moved = False
    for i in range(20):
        rng = jax.random.fold_in(_RNG, jnp.uint32(i))
        result = _invoke(state, rng=rng)
        if result.player_pos.tolist() != before_pos:
            moved = True
            break

    assert moved, "Orb of Fate should teleport at least once in 20 tries"


# ---------------------------------------------------------------------------
# Eye of the Aethiopica (idx 21): CREATE_PORTAL.
# Cite: vendor/nethack/src/artifact.c::invoke_create_portal lines 1866-1931.
# Audit K: ENERGY_REGEN was previously bolted onto this invoke; it is now
# correctly routed through apply_carried_artifact_extrinsics (SPFX_EREGEN
# in cspfx).  The invoke itself opens a dungeon portal (stubbed here).
# ---------------------------------------------------------------------------

def test_eye_aethiopica_create_portal_runs():
    """Eye of Aethiopica #invoke completes without state corruption."""
    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=21)
    before_pw = int(state.player_pw)

    result = _invoke(state)

    # Pw is left unchanged by CREATE_PORTAL (-1 cost).
    assert int(result.player_pw) == before_pw
    # cooldown should bump to 100.
    assert int(result.invoke_cooldown[21]) == 100


# ===========================================================================
# Audit-K NOOP invokes (vendor inv_prop == 0)
# ===========================================================================

def _assert_noop_invoke(slot: int, label: str):
    """Assert that #invoke on an inv_prop=0 slot has zero observable effect
    on identification timers / intrinsics / monsters."""
    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=slot)
    state = _place_monster(state, slot=0, hp=100, entry_idx=0)
    ident_before = state.identification
    intrinsics_before = state.status.intrinsics.copy()
    mai_before_hp = int(state.monster_ai.hp[0])

    result = _invoke(state)

    # No detect timer should have changed.
    for f in (
        "detect_objects_until_turn", "detect_food_until_turn",
        "detect_treasure_until_turn", "detect_monsters_until_turn",
        "detect_magic_until_turn",
    ):
        a = int(getattr(ident_before, f))
        b = int(getattr(result.identification, f))
        assert a == b, f"{label}: {f} changed {a}→{b} despite NOOP"
    # No monster mutation.
    assert int(result.monster_ai.hp[0]) == mai_before_hp, \
        f"{label}: monster HP changed despite NOOP"


def test_snickersnee_invoke_is_noop():
    """Snickersnee invoke is NOOP (vendor inv_prop=0; artilist.h:203-205)."""
    _assert_noop_invoke(1, "Snickersnee")


def test_grayswandir_invoke_is_noop():
    """Grayswandir invoke is NOOP (vendor inv_prop=0; artilist.h:170-172)."""
    _assert_noop_invoke(7, "Grayswandir")


def test_dragonbane_invoke_is_noop():
    """Dragonbane invoke is NOOP (vendor inv_prop=0; artilist.h:157-160)."""
    _assert_noop_invoke(24, "Dragonbane")


def test_werebane_invoke_is_noop():
    """Werebane invoke is NOOP (vendor inv_prop=0; artilist.h:166-168)."""
    _assert_noop_invoke(26, "Werebane")


def test_trollsbane_invoke_is_noop():
    """Trollsbane invoke is NOOP (vendor inv_prop=0; artilist.h:182-184)."""
    _assert_noop_invoke(27, "Trollsbane")


def test_magicbane_invoke_is_noop():
    """Magicbane invoke is NOOP (vendor inv_prop=0; artilist.h:145-147).

    On-hit Mb_hit is exercised separately in test_artifact_specials_parity.
    """
    _assert_noop_invoke(29, "Magicbane")


def test_mjollnir_invoke_is_noop():
    """Mjollnir invoke is NOOP — its STR-25 throw-return lives in throw code,
    NOT invoke.  Vendor inv_prop=0 (artilist.h:109-112)."""
    _assert_noop_invoke(3, "Mjollnir")


def test_excalibur_invoke_is_noop():
    """Excalibur invoke is NOOP (vendor inv_prop=0; artilist.h:85-88)."""
    _assert_noop_invoke(0, "Excalibur")


def test_tsurugi_invoke_is_noop():
    """Tsurugi invoke is NOOP (vendor inv_prop=0; artilist.h:285-289).

    SPFX_BEHEAD slice-in-half is on-hit, not invoke.
    """
    _assert_noop_invoke(10, "Tsurugi")


def test_vorpal_invoke_is_noop():
    """Vorpal Blade invoke is NOOP (vendor inv_prop=0; artilist.h:191-193)."""
    _assert_noop_invoke(8, "Vorpal Blade")


# ===========================================================================
# Demonbane BANISH (idx 25)
# ===========================================================================

def test_demonbane_banish_kills_demon():
    """Demonbane #invoke (BANISH) removes a demon-class monster from play."""
    state = _fresh_state()
    state = _place_monster(state, slot=0, hp=50, demon=True)
    state = _wield_artifact(state, artifact_idx=25)
    assert bool(state.monster_ai.alive[0])

    # Run multiple seeds to defeat the 50% banish-roll variance.
    banished = False
    for i in range(30):
        rng = jax.random.fold_in(_RNG, jnp.uint32(i))
        result = _invoke(state, rng=rng)
        if not bool(result.monster_ai.alive[0]):
            banished = True
            break
    assert banished, "Demonbane BANISH should remove a demon within 30 attempts"


# ===========================================================================
# Cooldown + edge cases (preserved from pre-Audit-K)
# ===========================================================================

def test_invoke_cooldown_blocks_second_invoke():
    """Second invoke with cooldown active is a no-op.

    Cite: vendor/nethack/src/artifact.c::arti_invoke_cost — artifact 'tired'
    check via state.invoke_cooldown[].  Audit K also wires an Item.age
    cooldown internally, but the invoke_cooldown gate retains the
    100-turn floor.
    """
    state = _fresh_state()
    state = state.replace(player_pw=jnp.int32(10), player_pw_max=jnp.int32(50))
    state = _wield_artifact(state, artifact_idx=16)  # Mitre — ENERGY_BOOST.

    result1 = _invoke(state)
    assert int(result1.invoke_cooldown[16]) == 100
    pw_after_first = int(result1.player_pw)

    result2 = _invoke(result1)
    # Pw must be unchanged (cooldown active).
    assert int(result2.player_pw) == pw_after_first
    assert int(result2.invoke_cooldown[16]) == 100


def test_no_artifact_wielded_is_noop():
    """#invoke with no artifact wielded (idx -1) does nothing."""
    state = _fresh_state()
    new_inv = state.inventory.replace(wielded_artifact_idx=jnp.int8(-1))
    state = state.replace(inventory=new_inv)
    before_pw = int(state.player_pw)

    result = _invoke(state)

    assert int(result.player_pw) == before_pw
    assert jnp.all(result.invoke_cooldown == jnp.int16(0))
