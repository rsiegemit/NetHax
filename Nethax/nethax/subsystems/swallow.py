"""Swallowed-player (engulf) mechanics.

Canonical source: vendor/nethack/src/mhitu.c::gulpmu  (lines 1287-1434)

Design notes
------------
* All public functions are JIT-pure (no Python control flow inside traces).
* ``SwallowState`` is a Flax ``struct.dataclass`` so it participates in the
  pytree used by ``EnvState``.
* ``_IS_ENGULFER`` is a module-level bool table built once at import time from
  the MONSTERS constant data — never traced.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from Nethax.nethax.rng import rn2, rnd, split_n


# ---------------------------------------------------------------------------
# _IS_ENGULFER lookup table
# Built from MONSTERS entries whose attack list contains AT_ENGL (value 11).
# vendor/nethack/include/attack.h / monst.c — M2_ENGULF flag is not present
# in the Nethax monster data; we derive engulf capability directly from the
# attack type table instead, which is equivalent.
# ---------------------------------------------------------------------------
def _build_is_engulfer_table() -> jnp.ndarray:
    from Nethax.nethax.constants.monsters import MONSTERS, AttackType
    at_engl = int(AttackType.AT_ENGL)
    flags = [any(a[0] == at_engl for a in m.attacks) for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


_IS_ENGULFER: jnp.ndarray = _build_is_engulfer_table()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@struct.dataclass
class SwallowState:
    """Swallow/engulf sub-pytree.

    Fields
    ------
    swallowed      : bool  — True while the player is inside a monster.
    engulfer_slot  : int32 — monster_ai slot of the engulfer; -1 if free.
    digest_timer   : int32 — turns until next digestion-damage tick; 0 if free.
    total_timer    : int32 — turns until auto-release; 0 if free.
                    Vendor: mhitu.c:1418 — engulfer digests for up to ~100 turns.
    """
    swallowed:     jnp.ndarray   # scalar bool
    engulfer_slot: jnp.ndarray   # scalar int32
    digest_timer:  jnp.ndarray   # scalar int32
    total_timer:   jnp.ndarray   # scalar int32

    @classmethod
    def default(cls) -> "SwallowState":
        return cls(
            swallowed=jnp.bool_(False),
            engulfer_slot=jnp.int32(-1),
            digest_timer=jnp.int32(0),
            total_timer=jnp.int32(0),
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def try_engulf(
    state,
    attacker_slot: jnp.ndarray,
    rng: jax.Array,
    attack_is_dgst=None,
    m_lev=None,
    con=None,
    uac=None,
):
    """Attempt to engulf the player.

    Called when an AT_ENGL attack lands.  No-ops if the player is already
    swallowed by another monster.

    Vendor: vendor/nethack/src/mhitu.c::gulpmu lines 1287, 1380-1395.

    Parameters
    ----------
    state         : EnvState
    attacker_slot : monster_ai slot index of the engulfer (int32 scalar)
    rng           : JAX PRNG key
    attack_is_dgst : optional bool/int32 — when provided (along with ``m_lev``,
                    ``con``, ``uac``), uswldtim is computed via the vendor
                    formula at mhitu.c:1380-1395:
                      if AD_DGST: tim_tmp = max(0, CON + 10 - uac + rn2(20))
                                  tim_tmp = tim_tmp / m_lev + 3
                      else      : tim_tmp = rnd(m_lev + 5)        # 10/2 == 5
                      uswldtim  = max(2, tim_tmp)
                    When ``attack_is_dgst`` is ``None``, falls back to the
                    legacy uniform [25, 100) approximation (3.6's rn1(25, 75)
                    shape for non-AD_DGST attacks).
    m_lev         : engulfer monster level (int32 scalar) — required when
                    ``attack_is_dgst`` is provided.
    con           : player CON (int32 scalar) — required for AD_DGST branch.
    uac           : player AC (int32 scalar) — required for AD_DGST branch.
    """
    already = state.swallow.swallowed

    if attack_is_dgst is None:
        # Legacy fallback: vendor 3.6 non-AD_DGST shape ``rn1(25, 75)`` ≈
        # uniform [25, 100).  Clamp to >= 2 mirrors mhitu.c:1395.
        total = jnp.maximum(jnp.int32(25) + rnd(rng, 75), jnp.int32(2))
    else:
        # Vendor mhitu.c:1380-1395 — split by attack adtyp == AD_DGST.
        rng_dgst, rng_other = jax.random.split(rng, 2)
        is_dgst = jnp.asarray(attack_is_dgst, dtype=jnp.bool_)
        m_lev_i = jnp.asarray(m_lev, dtype=jnp.int32)
        con_i   = jnp.asarray(con,   dtype=jnp.int32)
        uac_i   = jnp.asarray(uac,   dtype=jnp.int32)

        # AD_DGST branch (mhitu.c:1381-1389):
        #   tim_tmp = (int)ACURR(A_CON) + 10 - (int)u.uac + rn2(20);
        #   if (tim_tmp < 0) tim_tmp = 0;
        #   tim_tmp /= (int) mtmp->m_lev;
        #   tim_tmp += 3;
        dgst_raw = con_i + jnp.int32(10) - uac_i + rn2(rng_dgst, 20)
        dgst_clamped = jnp.maximum(dgst_raw, jnp.int32(0))
        # Avoid div-by-zero: m_lev guaranteed >= 1 in vendor; clamp defensively.
        safe_m_lev = jnp.maximum(m_lev_i, jnp.int32(1))
        dgst_tim = (dgst_clamped // safe_m_lev) + jnp.int32(3)

        # Non-AD_DGST branch (mhitu.c:1392): rnd((int) mtmp->m_lev + 10 / 2)
        # In C, 10/2 is evaluated as integer 5, so this is rnd(m_lev + 5),
        # i.e. uniform in [1, m_lev + 5].  We implement this inline because
        # rng.rnd takes a static Python upper bound but m_lev is traced.
        # Use uniform(0,1) * n + 1, floor — uniform across integer values.
        non_dgst_max = safe_m_lev + jnp.int32(5)
        u = jax.random.uniform(rng_other, (), dtype=jnp.float32)
        non_dgst_tim = jnp.int32(1) + jnp.floor(u * non_dgst_max.astype(jnp.float32)).astype(jnp.int32)
        # Guard against the (measure-zero) u == 1.0 corner.
        non_dgst_tim = jnp.minimum(non_dgst_tim, non_dgst_max)

        tim_tmp = jnp.where(is_dgst, dgst_tim, non_dgst_tim)
        # Vendor mhitu.c:1395: u.uswldtim = (tim_tmp < 2) ? 2 : tim_tmp;
        total = jnp.maximum(tim_tmp, jnp.int32(2))

    # Player's pos moves to the engulfer's pos (they are now inside).
    # vendor/nethack/src/mhitu.c:1301 — sets u.ux/u.uy to mtmp->mx/mtmp->my.
    engulfer_pos = state.monster_ai.pos[attacker_slot.astype(jnp.int32)]

    new_swallow = state.swallow.replace(
        swallowed=jnp.bool_(True),
        engulfer_slot=attacker_slot.astype(jnp.int32),
        digest_timer=jnp.int32(10),
        total_timer=total,
    )
    new_pos = engulfer_pos.astype(jnp.int16)

    swallowed_state = state.replace(
        swallow=new_swallow,
        player_pos=new_pos,
    )

    # No-op if already swallowed; apply engulf otherwise.
    return jax.lax.cond(
        already,
        lambda s: s,
        lambda s: swallowed_state,
        state,
    )


def release_from_engulf(state):
    """Release the player from the engulfer.

    Player retains current position (which was set to the engulfer's pos on
    engulf; they emerge at the same location as the now-dead or released
    engulfer).

    Vendor: vendor/nethack/src/mhitu.c::expels / fall-through at line 1418
    (after total_timer expires or engulfer dies).
    """
    new_swallow = state.swallow.replace(
        swallowed=jnp.bool_(False),
        engulfer_slot=jnp.int32(-1),
        digest_timer=jnp.int32(0),
        total_timer=jnp.int32(0),
    )
    return state.replace(swallow=new_swallow)


def digest_tick(state, rng: jax.Array):
    """Per-turn digestion tick while swallowed.

    Logic (vendor/nethack/src/mhitu.c:1418):
      * Decrement total_timer and digest_timer each by 1.
      * If digest_timer reaches 0: deal rnd(6)+1 HP damage, reset to 10.
      * If total_timer <= 0 OR engulfer is dead: release player.

    Parameters
    ----------
    state : EnvState (contains state.swallow and state.monster_ai)
    rng   : JAX PRNG key
    """
    # Fast path: if not swallowed, return unchanged.
    def _tick(s):
        sw = s.swallow
        key_dmg = rng

        new_total = sw.total_timer - jnp.int32(1)
        new_digest = sw.digest_timer - jnp.int32(1)

        # Digestion damage — vendor mhitu.c:1418: rnd(6)+1 per digest tick.
        dmg = rnd(key_dmg, 6) + jnp.int32(1)
        do_damage = new_digest <= jnp.int32(0)
        applied_dmg = jnp.where(do_damage, dmg, jnp.int32(0))
        reset_digest = jnp.where(do_damage, jnp.int32(10), new_digest)

        new_hp = jnp.maximum(s.player_hp - applied_dmg, jnp.int32(0))
        new_done = s.done | (new_hp <= jnp.int32(0))

        # Check whether engulfer died.
        slot = sw.engulfer_slot.astype(jnp.int32)
        engulfer_alive = s.monster_ai.alive[slot]

        should_release = (new_total <= jnp.int32(0)) | (~engulfer_alive)

        s2 = s.replace(
            player_hp=new_hp,
            done=new_done,
            swallow=sw.replace(
                total_timer=new_total,
                digest_timer=reset_digest,
            ),
        )

        return jax.lax.cond(
            should_release,
            lambda st: release_from_engulf(st),
            lambda st: st,
            s2,
        )

    return jax.lax.cond(
        state.swallow.swallowed,
        _tick,
        lambda s: s,
        state,
    )
