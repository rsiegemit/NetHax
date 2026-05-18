"""Combat polish 3 parity tests.

Vendor references:
  vendor/nethack/src/uhitm.c:410       — trap: -3 to-hit when player in trap
  vendor/nethack/src/uhitm.c:2116-2138 — disarmed monster uses bare-hands
  vendor/nethack/src/uhitm.c:380       — AC_VALUE: negative AC softened via rnd(-ac)
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

import Nethax.nethax.subsystems.artifact_powers  # noqa: F401
import Nethax.nethax.subsystems.weapon_dice      # noqa: F401

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.combat import melee_attack, monster_attack_player

_RNG = jax.random.PRNGKey(99)
_N_TRIALS = 800


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state():
    state = EnvState.default(_RNG)
    state = state.replace(
        player_str=jnp.int16(18),
        player_dex=jnp.int8(10),
        player_xl=jnp.int32(5),
        player_luck=jnp.int8(0),
        player_uhitinc=jnp.int8(0),
        player_udaminc=jnp.int8(0),
        player_hp=jnp.int32(9999),
        player_hp_max=jnp.int32(9999),
    )
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(jnp.bool_(True)),
        hp=mai.hp.at[0].set(jnp.int32(9999)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(9999)),
        ac=mai.ac.at[0].set(jnp.int8(10)),
        asleep=mai.asleep.at[0].set(jnp.bool_(False)),
        mstrategy=mai.mstrategy.at[0].set(jnp.int8(0)),
        pos=mai.pos.at[0].set(jnp.array([3, 3], dtype=jnp.int16)),
        attack_dice_n=mai.attack_dice_n.at[0].set(jnp.int8(3)),
        attack_dice_sides=mai.attack_dice_sides.at[0].set(jnp.int8(8)),
    )
    return state.replace(monster_ai=mai)


def _hit_rate(state, n=_N_TRIALS, seed=11):
    """Fraction of melee_attack calls that land a hit on monster slot 0."""
    rng = jax.random.PRNGKey(seed)
    hits = 0
    fn = jax.jit(lambda s, k: melee_attack(s, k, jnp.int32(0)))
    for _ in range(n):
        rng, sub = jax.random.split(rng)
        _s, _dmg, hit = fn(state, sub)
        hits += int(hit)
    return hits / n


def _monster_hit_rate(state, n=_N_TRIALS, seed=13):
    """Fraction of monster_attack_player calls that deal >0 damage."""
    rng = jax.random.PRNGKey(seed)
    hits = 0
    fn = jax.jit(lambda s, k: monster_attack_player(s, k, jnp.int32(0)))
    for _ in range(n):
        rng, sub = jax.random.split(rng)
        _s, dmg = fn(state, sub)
        hits += int(dmg) > 0
    return hits / n


def _monster_avg_damage(state, n=_N_TRIALS, seed=17):
    """Average damage per turn from monster_attack_player (includes misses as 0)."""
    rng = jax.random.PRNGKey(seed)
    total = 0
    fn = jax.jit(lambda s, k: monster_attack_player(s, k, jnp.int32(0)))
    for _ in range(n):
        rng, sub = jax.random.split(rng)
        _s, dmg = fn(state, sub)
        total += int(dmg)
    return total / n


# ---------------------------------------------------------------------------
# Test 1: player_in_trap subtracts 3 from to-hit
# vendor/nethack/src/uhitm.c:410
# ---------------------------------------------------------------------------

def test_player_in_trap_subtracts_3():
    """Hit rate is lower when player_in_trap=True than when False."""
    state = _base_state()

    state_free = state.replace(player_in_trap=jnp.bool_(False))
    state_trap = state.replace(player_in_trap=jnp.bool_(True))

    rate_free = _hit_rate(state_free)
    rate_trap = _hit_rate(state_trap)

    # -3 penalty to tmp reduces hit rate; with ~N=800 samples the difference
    # should be clearly detectable (expected delta ~0.15 at typical AC).
    assert rate_trap < rate_free, (
        f"player_in_trap should reduce hit rate: free={rate_free:.3f} trap={rate_trap:.3f}"
    )
    # Sanity: penalty is not catastrophic (player can still hit sometimes).
    assert rate_trap > 0.0, "hit rate with trap should still be > 0"


# ---------------------------------------------------------------------------
# Test 2: disarmed monster uses bare-hands (1d2) instead of weapon dice
# vendor/nethack/src/uhitm.c:2116-2138 / weapon.c disarm
# ---------------------------------------------------------------------------

def test_disarmed_monster_uses_bare_hands():
    """Average damage is lower when is_unwielded=True (1d2 vs 3d8)."""
    state = _base_state()
    # Set player AC to -5 so monster hits almost always.
    state = state.replace(player_ac=jnp.int32(-5))

    # Armed: 3d8 -> expected avg damage on hit ~13.5; with near-certain hits, avg ~13.
    state_armed = state
    # Disarmed: 1d2 -> expected avg damage on hit ~1.5.
    mai = state.monster_ai
    mai_disarmed = mai.replace(
        is_unwielded=mai.is_unwielded.at[0].set(jnp.bool_(True)),
    )
    state_disarmed = state.replace(monster_ai=mai_disarmed)

    avg_armed    = _monster_avg_damage(state_armed)
    avg_disarmed = _monster_avg_damage(state_disarmed)

    assert avg_disarmed < avg_armed, (
        f"disarmed monster should deal less damage: armed={avg_armed:.2f} disarmed={avg_disarmed:.2f}"
    )
    # Bare-hands max is 2; armed max is 24. Gap should be large.
    assert avg_armed > avg_disarmed * 2, (
        f"armed damage {avg_armed:.2f} should be > 2x disarmed {avg_disarmed:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 3: negative AC softens hit rate (vendor/nethack/src/uhitm.c:380)
# ---------------------------------------------------------------------------

def test_negative_ac_softer():
    """Monster hit rate at player AC=-10 should be HIGHER than a naive
    symmetric reading of AC=+10 would predict — because AC_VALUE(negative)
    is randomised via rnd(-ac), making it sometimes closer to 0 and thus
    softening the AC bonus (player gets hit more than the raw -10 would imply).

    We verify by comparing:
      AC=+10  (normal, easy-to-hit player)  → high hit rate
      AC=-10  (well-armored player) → lower hit rate, but not zero
    and separately that AC=-10 gives a HIGHER hit rate than AC=-20
    (confirming the random softening scales with magnitude).
    """
    base = _base_state()
    # Weak monster (hp_max=4 → mlev~1) so the AC effect isn't masked by a
    # universal-hit ceiling. With tmp = AC_VALUE + 10 + mlev, mlev≈1 keeps the
    # roll inside the rnd(20) window so AC differences register.
    mai = base.monster_ai
    mai = mai.replace(hp_max=mai.hp_max.at[0].set(jnp.int32(4)))
    base = base.replace(monster_ai=mai)

    # compute_ac() reads state.player_ac directly only when polymorphed
    # (polyself.c::find_uac). Drive that path so player_ac is the AC used.
    poly = base.polymorph
    poly_on = poly.replace(is_polymorphed=jnp.bool_(True))
    base = base.replace(polymorph=poly_on)

    state_ac_pos10 = base.replace(player_ac=jnp.int32(10))
    state_ac_neg10 = base.replace(player_ac=jnp.int32(-10))
    state_ac_neg20 = base.replace(player_ac=jnp.int32(-20))

    rate_pos10 = _monster_hit_rate(state_ac_pos10)
    rate_neg10 = _monster_hit_rate(state_ac_neg10)
    rate_neg20 = _monster_hit_rate(state_ac_neg20)

    # AC=+10 (unarmored) → monster should hit more often than AC=-10 (armored).
    assert rate_pos10 > rate_neg10, (
        f"AC=+10 should yield higher monster hit rate than AC=-10: "
        f"{rate_pos10:.3f} vs {rate_neg10:.3f}"
    )
    # AC=-10 should hit more often than AC=-20 (softening is proportional).
    assert rate_neg10 > rate_neg20, (
        f"AC=-10 should yield higher monster hit rate than AC=-20 (softening): "
        f"{rate_neg10:.3f} vs {rate_neg20:.3f}"
    )
    # With softening, AC=-10 should still occasionally let hits through (> 0).
    assert rate_neg10 > 0.0, "AC=-10 with softening should still allow some hits"
