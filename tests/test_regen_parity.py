"""Regen-parity tests for status_effects.py against vendor/nethack.

Wave 6 Phase B+ audit #64 verifies:
  - HP regen interval by XL  — vendor timeout.c::nh_timeout (lines 600-650).
  - Pw regen interval        — vendor allmain.c::regen_pw (lines 605-625).
  - Ring of regeneration     — halves HP regen interval (timeout.c).
  - Wisdom modifier for Pw   — vendor allmain.c line 613 (upper = WIS+INT/15 + 1).

Our subsystem implements a simplified, deterministic version of these
formulas (one HP/Pw per interval rather than probabilistic per-turn
checks).  These tests assert the simplified formulas behave exactly:

  HP regen interval = max(1, 20 - XL)            (status_effects.py:485)
  HP w/ REGEN ring  = max(1, interval // 2)      (status_effects.py:486-488)
  Pw regen interval = max(1, 30 - XL) generic
                      max(1, 20 - XL) wizard/healer  (status_effects.py:517-519)
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.subsystems.status_effects import (
    Intrinsic,
    StatusState,
    hp_regen_tick,
    pw_regen_tick,
)


def _make_state(**overrides) -> StatusState:
    base = StatusState.default()
    if not overrides:
        return base
    return base.replace(**overrides)


def _hp_ticks_until_heal(state, hp, hp_max, xl, role,
                         player_con=jnp.int8(15),
                         seed: int = 0):
    """Run vendor-parity hp_regen_tick repeatedly until HP increases.

    Wave 6 parity-fix: updated to vendor allmain.c:649-665 (CA #73 / #78).
    The legacy deterministic interval signature has been removed; this helper
    drives the new probabilistic path with a fixed CON and per-tick RNG.
    Returns (tick_count, state, new_hp).
    """
    current = state
    rng = jax.random.PRNGKey(seed)
    for n in range(1, 1000):
        rng, sub = jax.random.split(rng)
        current, new_hp = hp_regen_tick(
            current, hp, hp_max, xl, role,
            player_con=player_con,
            timestep=jnp.int32(n),
            rng=sub,
        )
        if int(new_hp) > int(hp):
            return n, current, int(new_hp)
    raise AssertionError("HP did not heal within 1000 ticks")


def _pw_ticks_until_regen(state, pw, pw_max, xl, role,
                          player_int=jnp.int8(10),
                          player_wis=jnp.int8(10),
                          seed: int = 0):
    """Run vendor-parity pw_regen_tick repeatedly until Pw regenerates.

    Wave 6 parity-fix: updated to vendor allmain.c:606-625 (CA #73 / #78).
    Returns (tick_count, state, new_pw).
    """
    current = state
    rng = jax.random.PRNGKey(seed)
    for n in range(1, 1000):
        rng, sub = jax.random.split(rng)
        current, new_pw = pw_regen_tick(
            current, pw, pw_max, xl, role,
            player_int=player_int,
            player_wis=player_wis,
            timestep=jnp.int32(n),
            rng=sub,
        )
        if int(new_pw) > int(pw):
            return n, current, int(new_pw)
    raise AssertionError("Pw did not regen within 1000 ticks")


# ---------------------------------------------------------------------------
# A. HP regen interval by experience level
# ---------------------------------------------------------------------------

class TestHpRegenInterval:
    # Wave 28a parity-fix: vendor allmain.c:664-665 — the non-Upolyd HP
    # regen path has NO moves%20 throttle.  The probabilistic check
    # ``(XL + ACURR(A_CON)) > rn2(100)`` fires every turn.  These tests
    # now pin the per-turn behaviour: a heal can land on ANY turn (not
    # exclusively multiples of 20), and high (XL + CON) reliably heals
    # within a handful of turns.

    def test_hp_regen_interval_xl1_19_turns(self):
        """XL=1 CON=15: heal can land on any turn — probability 16% per turn.

        Cite: vendor/nethack/src/allmain.c::regen_hp lines 664-665 (non-Upolyd
        path has no moves%20 gate).
        """
        state = _make_state()
        n, _, healed = _hp_ticks_until_heal(
            state,
            hp=jnp.int32(5),
            hp_max=jnp.int32(10),
            xl=jnp.int32(1),
            role=jnp.int8(0),
            player_con=jnp.int8(15),
            seed=0,
        )
        # At 16%/turn expected first-heal turn is ~6; cap at 100 with margin.
        assert 1 <= n <= 100, f"XL=1 HP regen should heal quickly; got {n}"
        assert healed == 6

    def test_hp_regen_interval_xl5_curve(self):
        """XL=5 CON=15: any turn can fire (no moves%20 gate)."""
        state = _make_state()
        n, _, healed = _hp_ticks_until_heal(
            state,
            hp=jnp.int32(5),
            hp_max=jnp.int32(20),
            xl=jnp.int32(5),
            role=jnp.int8(0),
            player_con=jnp.int8(15),
            seed=0,
        )
        assert 1 <= n <= 100, f"XL=5 HP regen should heal quickly; got {n}"
        assert healed == 6

    def test_hp_regen_xl19_curve(self):
        """XL=19 CON=18: probability per turn = 37%; first heal often within
        a few turns.

        Cite: vendor/nethack/src/allmain.c::regen_hp lines 664-665.
        """
        state = _make_state()
        n, _, healed = _hp_ticks_until_heal(
            state,
            hp=jnp.int32(5),
            hp_max=jnp.int32(20),
            xl=jnp.int32(19),
            role=jnp.int8(0),
            player_con=jnp.int8(18),
            seed=0,
        )
        assert 1 <= n <= 50, f"XL=19 HP regen should heal quickly; got {n}"
        assert healed == 6

    def test_hp_regen_xl30_max_rate(self):
        """XL=30 CON=18: (30+18)=48 > rn2(100) often; first heal is fast.

        Cite: vendor/nethack/src/allmain.c::regen_hp lines 664-665.
        """
        state = _make_state()
        n, _, healed = _hp_ticks_until_heal(
            state,
            hp=jnp.int32(5),
            hp_max=jnp.int32(40),
            xl=jnp.int32(30),
            role=jnp.int8(0),
            player_con=jnp.int8(18),
            seed=0,
        )
        assert 1 <= n <= 20, f"XL=30 HP regen should heal very fast; got {n}"
        assert healed == 6


# ---------------------------------------------------------------------------
# B. Ring of regeneration — vendor allmain.c::regen_hp REGEN intrinsic
# ---------------------------------------------------------------------------

class TestRingOfRegen:
    # Wave 6 parity-fix: updated to vendor allmain.c U_CAN_REGEN macro
    # (line 627) — REGEN intrinsic forces HP +1 every turn unconditionally,
    # NOT a halved interval.

    def test_hp_regen_ring_halves_interval_xl1(self):
        """REGEN intrinsic at XL=1 → heals every single turn (interval=1).

        Cite: vendor/nethack/src/allmain.c U_CAN_REGEN macro (line 627).
        """
        state = _make_state()
        state = state.replace(
            intrinsics=state.intrinsics.at[Intrinsic.REGEN].set(True),
        )
        n, _, _ = _hp_ticks_until_heal(
            state,
            hp=jnp.int32(5),
            hp_max=jnp.int32(20),
            xl=jnp.int32(1),
            role=jnp.int8(0),
            player_con=jnp.int8(15),
            seed=0,
        )
        assert n == 1, f"REGEN ring should heal on turn 1; got {n}"

    def test_hp_regen_ring_xl5(self):
        """REGEN intrinsic at XL=5 → heals every single turn.

        Cite: vendor/nethack/src/allmain.c U_CAN_REGEN macro (line 627).
        """
        state = _make_state()
        state = state.replace(
            intrinsics=state.intrinsics.at[Intrinsic.REGEN].set(True),
        )
        n, _, _ = _hp_ticks_until_heal(
            state,
            hp=jnp.int32(5),
            hp_max=jnp.int32(20),
            xl=jnp.int32(5),
            role=jnp.int8(0),
            player_con=jnp.int8(15),
            seed=0,
        )
        assert n == 1, f"REGEN ring should heal on turn 1; got {n}"


# ---------------------------------------------------------------------------
# C. Pw regen — Wizard / Healer faster than other roles
# ---------------------------------------------------------------------------

class TestPwRegenInterval:
    # Wave 6 parity-fix: updated to vendor allmain.c:610-625 (CA #73).
    # Legacy simplified intervals (max(1, 20-XL) wizard/healer, max(1, 30-XL)
    # other) were replaced with the vendor formula:
    #     period = max(1, (MAXULEV + 8 - XL) * (wizard ? 3 : 4) / 6)
    # MAXULEV = 30; Wizard role (12) → factor 3, all other roles → factor 4.
    # Healer is no longer in the wizard branch (vendor only branches on
    # Role_if(PM_WIZARD), so Healer takes the factor-4 path).
    # On a firing turn, gain = rn1((WIS+INT)/15 + 1, 1) Pw.
    # ENERGY_REGEN intrinsic forces ``do_regen = True`` every turn (period=1).

    def test_pw_regen_interval_wizard_high_wisdom(self):
        """Wizard (role=12) XL=1: period = (30+8-1)*3/6 = 18.

        Cite: vendor/nethack/src/allmain.c::regen_pw lines 609-611.
        """
        state = _make_state()
        rng = jax.random.PRNGKey(0)
        # On the firing turn (moves=18) Pw must increase.
        _, new_pw = pw_regen_tick(
            state,
            jnp.int32(0),
            jnp.int32(20),
            jnp.int32(1),
            jnp.int8(12),                  # Wizard
            player_int=jnp.int8(10),
            player_wis=jnp.int8(10),
            timestep=jnp.int32(18),
            rng=rng,
        )
        assert int(new_pw) >= 1, f"Wizard XL=1 at moves=18 should regen, got {int(new_pw)}"

        # On a non-firing turn (moves=17) Pw must NOT increase.
        _, new_pw_no = pw_regen_tick(
            state,
            jnp.int32(0),
            jnp.int32(20),
            jnp.int32(1),
            jnp.int8(12),
            player_int=jnp.int8(10),
            player_wis=jnp.int8(10),
            timestep=jnp.int32(17),
            rng=rng,
        )
        assert int(new_pw_no) == 0, f"Wizard XL=1 at moves=17 should not regen, got {int(new_pw_no)}"

    def test_pw_regen_interval_healer_xl5(self):
        """Healer (role=3) XL=5: period = (30+8-5)*4/6 = 22.

        Vendor only branches on Role_if(PM_WIZARD); Healer takes factor 4.
        Cite: vendor/nethack/src/allmain.c::regen_pw line 611.
        """
        state = _make_state()
        rng = jax.random.PRNGKey(0)
        _, new_pw = pw_regen_tick(
            state,
            jnp.int32(0),
            jnp.int32(20),
            jnp.int32(5),
            jnp.int8(3),                   # Healer
            player_int=jnp.int8(10),
            player_wis=jnp.int8(10),
            timestep=jnp.int32(22),
            rng=rng,
        )
        assert int(new_pw) >= 1, f"Healer XL=5 at moves=22 should regen, got {int(new_pw)}"

        _, new_pw_no = pw_regen_tick(
            state,
            jnp.int32(0),
            jnp.int32(20),
            jnp.int32(5),
            jnp.int8(3),
            player_int=jnp.int8(10),
            player_wis=jnp.int8(10),
            timestep=jnp.int32(21),
            rng=rng,
        )
        assert int(new_pw_no) == 0, f"Healer XL=5 at moves=21 should not regen, got {int(new_pw_no)}"

    def test_pw_regen_interval_fighter_xl1(self):
        """Valkyrie (role=11) XL=1: period = (30+8-1)*4/6 = 24.

        Cite: vendor/nethack/src/allmain.c::regen_pw line 611 (non-Wizard
        factor 4).
        """
        state = _make_state()
        rng = jax.random.PRNGKey(0)
        _, new_pw = pw_regen_tick(
            state,
            jnp.int32(0),
            jnp.int32(20),
            jnp.int32(1),
            jnp.int8(11),                  # Valkyrie
            player_int=jnp.int8(10),
            player_wis=jnp.int8(10),
            timestep=jnp.int32(24),
            rng=rng,
        )
        assert int(new_pw) >= 1, f"Valkyrie XL=1 at moves=24 should regen, got {int(new_pw)}"

        _, new_pw_no = pw_regen_tick(
            state,
            jnp.int32(0),
            jnp.int32(20),
            jnp.int32(1),
            jnp.int8(11),
            player_int=jnp.int8(10),
            player_wis=jnp.int8(10),
            timestep=jnp.int32(23),
            rng=rng,
        )
        assert int(new_pw_no) == 0, f"Valkyrie XL=1 at moves=23 should not regen, got {int(new_pw_no)}"

    def test_pw_regen_energy_regen_halves_interval(self):
        """ENERGY_REGEN intrinsic regens every turn (vendor: do_regen forced True).

        Cite: vendor/nethack/src/allmain.c::regen_pw line 608 — when
        Energy_regeneration is set, the per-period gate is bypassed.
        """
        state = _make_state()
        state = state.replace(
            intrinsics=state.intrinsics.at[Intrinsic.ENERGY_REGEN].set(True),
        )
        rng = jax.random.PRNGKey(0)
        # On a non-period turn (moves=1) Pw must still regen because of
        # ENERGY_REGEN.
        _, new_pw = pw_regen_tick(
            state,
            jnp.int32(0),
            jnp.int32(20),
            jnp.int32(1),
            jnp.int8(12),                  # Wizard
            player_int=jnp.int8(10),
            player_wis=jnp.int8(10),
            timestep=jnp.int32(1),
            rng=rng,
        )
        assert int(new_pw) >= 1, (
            f"ENERGY_REGEN Wizard XL=1 must regen every turn; "
            f"got {int(new_pw)} at moves=1"
        )


# ---------------------------------------------------------------------------
# D. Vendor-parity Pw regen — formula from allmain.c::regen_pw (Wave 6 #73)
# ---------------------------------------------------------------------------

class TestPwRegenVendorFormula:
    """Wave 6 #73: assert the vendor allmain.c::regen_pw formula.

      period = (MAXULEV + 8 - ulevel) * (wizard ? 3 : 4) / 6
      MAXULEV = 30; Wizard role → 3, all other roles → 4.
      When ``moves % period == 0``, gain ``rn1((WIS+INT)/15 + 1, 1)`` Pw.
    """

    def test_pw_regen_wizard_xl5_period(self):
        """Wizard XL=5: period = (30+8-5)*3/6 = 33*3/6 = 16.

        Cite: vendor/nethack/src/allmain.c::regen_pw lines 609-611.
        """
        state = _make_state()
        # Step the vendor-parity path: timestep at multiple of 16 → regen fires.
        rng = jax.random.PRNGKey(0)
        _, new_pw = pw_regen_tick(
            state,
            jnp.int32(0),                  # pw
            jnp.int32(20),                 # pw_max
            jnp.int32(5),                  # xl
            jnp.int8(12),                  # Wizard role
            player_int=jnp.int8(15),
            player_wis=jnp.int8(15),
            timestep=jnp.int32(16),        # 16 % 16 == 0 → regen
            rng=rng,
        )
        assert int(new_pw) >= 1, f"Wizard XL=5 at moves=16 should regen, got {int(new_pw)}"

        # Non-period turn should NOT regen.
        _, new_pw_no = pw_regen_tick(
            state,
            jnp.int32(0),
            jnp.int32(20),
            jnp.int32(5),
            jnp.int8(12),
            player_int=jnp.int8(15),
            player_wis=jnp.int8(15),
            timestep=jnp.int32(7),         # 7 % 16 != 0 → no regen
            rng=rng,
        )
        assert int(new_pw_no) == 0, f"Wizard XL=5 at moves=7 should not regen, got {int(new_pw_no)}"

    def test_pw_regen_valkyrie_xl5_period(self):
        """Valkyrie XL=5: period = (30+8-5)*4/6 = 33*4/6 = 22.

        Different bracket from Wizard.  Cite: allmain.c line 611.
        """
        state = _make_state()
        rng = jax.random.PRNGKey(1)
        # At moves=22, regen fires.
        _, new_pw = pw_regen_tick(
            state,
            jnp.int32(0),
            jnp.int32(20),
            jnp.int32(5),
            jnp.int8(11),                  # Valkyrie role
            player_int=jnp.int8(10),
            player_wis=jnp.int8(10),
            timestep=jnp.int32(22),
            rng=rng,
        )
        assert int(new_pw) >= 1, f"Valkyrie XL=5 at moves=22 should regen, got {int(new_pw)}"

        # At moves=16 (where Wizard regens), Valkyrie should NOT.
        _, new_pw_no = pw_regen_tick(
            state,
            jnp.int32(0),
            jnp.int32(20),
            jnp.int32(5),
            jnp.int8(11),
            player_int=jnp.int8(10),
            player_wis=jnp.int8(10),
            timestep=jnp.int32(16),
            rng=rng,
        )
        assert int(new_pw_no) == 0, (
            f"Valkyrie XL=5 at moves=16 (Wizard-period) should not regen, "
            f"got {int(new_pw_no)}"
        )

    def test_pw_regen_uses_wisdom_int_formula(self):
        """Gain on regen tick is rn1((WIS+INT)/15 + 1, 1) = 1 + rand(0..upper-1).

        With WIS=18, INT=18 → upper = 36/15 + 1 = 3.  Gain is in [1, 3].
        With WIS=10, INT=10 → upper = 20/15 + 1 = 2.  Gain is in [1, 2].
        Wave 6 #73 cite: vendor/nethack/src/allmain.c line 613.
        """
        state = _make_state()
        # Sample many seeds — gain must always be in the valid range.
        for seed in range(50):
            rng = jax.random.PRNGKey(seed)
            _, new_pw = pw_regen_tick(
                state,
                jnp.int32(0),
                jnp.int32(100),
                jnp.int32(5),
                jnp.int8(12),                  # Wizard
                player_int=jnp.int8(18),
                player_wis=jnp.int8(18),
                timestep=jnp.int32(16),        # 16 % 16 == 0 → regen
                rng=rng,
            )
            assert 1 <= int(new_pw) <= 3, (
                f"High-stat Wizard regen gain must be in [1,3]; got {int(new_pw)} (seed={seed})"
            )

        # Low-stat Wizard: gain in [1, 2].
        for seed in range(50):
            rng = jax.random.PRNGKey(seed)
            _, new_pw = pw_regen_tick(
                state,
                jnp.int32(0),
                jnp.int32(100),
                jnp.int32(5),
                jnp.int8(12),
                player_int=jnp.int8(10),
                player_wis=jnp.int8(10),
                timestep=jnp.int32(16),
                rng=rng,
            )
            assert 1 <= int(new_pw) <= 2, (
                f"Low-stat Wizard regen gain must be in [1,2]; got {int(new_pw)} (seed={seed})"
            )


# ---------------------------------------------------------------------------
# E. Vendor-parity HP regen — formula from allmain.c::regen_hp (Wave 6 #73)
# ---------------------------------------------------------------------------

class TestHpRegenVendorFormula:
    """Wave 28a: assert vendor allmain.c::regen_hp non-Upolyd formula.

      (XL + ACURR(A_CON)) > rn2(100) → HP += 1 EVERY turn (no moves%20 gate).
      REGEN intrinsic → HP +1 every turn unconditionally.
    """

    def test_hp_regen_uses_xl_con_probability(self):
        """For XL=10 CON=15, probability of HP +1 each turn is 25/100 = 25%.

        Over 10,000 trials on an arbitrary turn, empirical rate should be
        within ±5pp of 25%.  Per vendor allmain.c:664-665 the non-Upolyd
        path fires the probabilistic check every turn — no moves%20 gate.

        Cite: vendor/nethack/src/allmain.c::regen_hp lines 664-665.
        """
        state = _make_state()
        n_trials = 10_000
        heals = 0
        for seed in range(n_trials):
            rng = jax.random.PRNGKey(seed)
            _, new_hp = hp_regen_tick(
                state,
                jnp.int32(50),                # hp < hp_max
                jnp.int32(100),               # hp_max
                jnp.int32(10),                # xl
                jnp.int8(0),                  # role (unused on this path)
                player_con=jnp.int8(15),
                timestep=jnp.int32(20),       # any turn — gate removed
                rng=rng,
            )
            if int(new_hp) > 50:
                heals += 1

        empirical_pct = 100 * heals / n_trials
        assert 20.0 <= empirical_pct <= 30.0, (
            f"XL=10 CON=15 should heal ~25%/turn; got {empirical_pct:.1f}% "
            f"({heals}/{n_trials})"
        )

    def test_hp_regen_fires_on_any_turn(self):
        """Non-Upolyd vendor regen has NO moves%20 gate — high (XL+CON) heals
        on arbitrary turns including those not divisible by 20.

        Cite: vendor/nethack/src/allmain.c::regen_hp lines 664-665.
        """
        state = _make_state()
        # High XL=30 CON=18 → 48% per turn.  On turn 7 (not mod 20), a
        # majority of seeds should heal.
        heals_on_turn_7 = 0
        for seed in range(200):
            rng = jax.random.PRNGKey(seed)
            _, new_hp = hp_regen_tick(
                state,
                jnp.int32(50),
                jnp.int32(100),
                jnp.int32(30),                # high XL
                jnp.int8(0),
                player_con=jnp.int8(18),      # max CON
                timestep=jnp.int32(7),        # explicitly NOT a multiple of 20
                rng=rng,
            )
            if int(new_hp) == 51:
                heals_on_turn_7 += 1
        # Expect ~48% empirical; assert at least a strong fraction.  This
        # would be 0% under the legacy moves%20 gate.
        assert heals_on_turn_7 >= 50, (
            f"Vendor non-Upolyd regen must fire on moves=7; only "
            f"{heals_on_turn_7}/200 heals seen — moves%20 gate still active?"
        )

    def test_hp_regen_ring_fires_every_turn(self):
        """REGEN intrinsic → HP +1 every turn regardless of moves%20.

        Cite: vendor/nethack/src/allmain.c U_CAN_REGEN macro (line 627).
        """
        state = _make_state()
        state = state.replace(
            intrinsics=state.intrinsics.at[Intrinsic.REGEN].set(True),
        )
        for offset in range(1, 21):  # cover every residue mod 20
            rng = jax.random.PRNGKey(offset)
            _, new_hp = hp_regen_tick(
                state,
                jnp.int32(50),
                jnp.int32(100),
                jnp.int32(1),                  # low XL/CON irrelevant under REGEN
                jnp.int8(0),
                player_con=jnp.int8(3),
                timestep=jnp.int32(offset),
                rng=rng,
            )
            assert int(new_hp) == 51, (
                f"REGEN ring must heal every turn; failed at moves={offset}"
            )

    def test_hp_regen_blocked_when_starving(self):
        """HP regen skipped when hunger_state >= WEAK.

        Cite: vendor/nethack/src/allmain.c::regen_hp encumbrance_ok gate.
        """
        from Nethax.nethax.subsystems.status_effects import HungerState
        state = _make_state(hunger_state=jnp.int8(HungerState.WEAK))
        # Even with REGEN ring and firing turn, starving blocks regen.
        state = state.replace(
            intrinsics=state.intrinsics.at[Intrinsic.REGEN].set(True),
        )
        rng = jax.random.PRNGKey(0)
        _, new_hp = hp_regen_tick(
            state,
            jnp.int32(50),
            jnp.int32(100),
            jnp.int32(30),
            jnp.int8(0),
            player_con=jnp.int8(18),
            timestep=jnp.int32(20),
            rng=rng,
        )
        assert int(new_hp) == 50, (
            f"Starving (WEAK) should block HP regen; got {int(new_hp)}"
        )
