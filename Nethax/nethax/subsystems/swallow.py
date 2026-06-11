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
# Swallow-message variant table — chooses between SWALLOW_DIGESTS,
# SWALLOW_ENFOLDS, and SWALLOW_ENGULFS based on the engulfer's AT_ENGL attack
# damage type.  Mirrors mhitu.c:1335-1338:
#   digests(mtmp->data) ? "swallows you whole"      (AT_ENGL + AD_DGST)
#   enfolds(mtmp->data) ? "folds itself around you" (AT_ENGL + AD_WRAP)
#   else                  "engulfs you"             (any other AT_ENGL)
# Encoded as: 0 = ENGULFS (default), 1 = DIGESTS, 2 = ENFOLDS.
# ---------------------------------------------------------------------------
def _build_swallow_variant_table() -> jnp.ndarray:
    from Nethax.nethax.constants.monsters import MONSTERS, AttackType, DamageType
    at_engl = int(AttackType.AT_ENGL)
    ad_dgst = int(DamageType.AD_DGST)
    # AD_WRAP: mhitu/monattk uses AD_WRAP for enfolding (trapper/lurker).
    ad_wrap = getattr(DamageType, "AD_WRAP", None)
    ad_wrap_v = int(ad_wrap) if ad_wrap is not None else -1
    variants = []
    for m in MONSTERS:
        v = 0  # default ENGULFS
        for atk in m.attacks:
            if atk[0] == at_engl:
                if atk[1] == ad_dgst:
                    v = 1
                elif ad_wrap_v >= 0 and atk[1] == ad_wrap_v:
                    v = 2
                break
        variants.append(v)
    return jnp.array(variants, dtype=jnp.int8)


_SWALLOW_VARIANT: jnp.ndarray = _build_swallow_variant_table()


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
                    vendor 3.6 ``rn1(25, 75)`` = ``rn2(25) + 75`` ∈ [75, 99]
                    formula for non-AD_DGST attacks.
    m_lev         : engulfer monster level (int32 scalar) — required when
                    ``attack_is_dgst`` is provided.
    con           : player CON (int32 scalar) — required for AD_DGST branch.
    uac           : player AC (int32 scalar) — required for AD_DGST branch.
    """
    already = state.swallow.swallowed

    if attack_is_dgst is None:
        # Vendor 3.6 non-AD_DGST: ``rn1(25, 75)`` = ``rn2(25) + 75`` ∈ [75, 99].
        # Clamp to >= 2 mirrors mhitu.c:1395.
        total = jnp.maximum(
            jax.random.randint(rng, (), 0, 25, dtype=jnp.int32) + jnp.int32(75),
            jnp.int32(2),
        )
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
    slot_i32 = attacker_slot.astype(jnp.int32)
    engulfer_pos = state.monster_ai.pos[slot_i32]

    new_swallow = state.swallow.replace(
        swallowed=jnp.bool_(True),
        engulfer_slot=slot_i32,
        digest_timer=jnp.int32(10),
        total_timer=total,
    )
    new_pos = engulfer_pos.astype(jnp.int16)

    # Choose swallow-message variant by engulfer's AT_ENGL damage type.
    # Cite: vendor/nethack/src/mhitu.c:1335-1338 (digests/enfolds/engulfs).
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    entry_idx = state.monster_ai.entry_idx[slot_i32].astype(jnp.int32)
    safe_entry = jnp.clip(entry_idx, jnp.int32(0), jnp.int32(_SWALLOW_VARIANT.shape[0] - 1))
    variant = _SWALLOW_VARIANT[safe_entry].astype(jnp.int32)
    msg_id_engulfs = jnp.int32(int(_MsgId.SWALLOW_ENGULFS))
    msg_id_digests = jnp.int32(int(_MsgId.SWALLOW_DIGESTS))
    msg_id_enfolds = jnp.int32(int(_MsgId.SWALLOW_ENFOLDS))
    chosen_id = jnp.where(
        variant == jnp.int32(1), msg_id_digests,
        jnp.where(variant == jnp.int32(2), msg_id_enfolds, msg_id_engulfs),
    )
    new_messages = _msg_emit(state.messages, chosen_id)

    swallowed_state = state.replace(
        swallow=new_swallow,
        player_pos=new_pos,
        messages=new_messages,
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
        # ``engulfer_slot`` is -1 when idle (no engulfer); JAX silently
        # wraps a -1 scalar index into the last array element, so clip
        # first and treat the idle case as "not alive" → release.
        slot = sw.engulfer_slot.astype(jnp.int32)
        safe_slot = jnp.clip(slot, 0, s.monster_ai.alive.shape[0] - 1)
        engulfer_alive = jnp.where(
            slot >= jnp.int32(0),
            s.monster_ai.alive[safe_slot],
            jnp.bool_(False),
        )

        should_release = (new_total <= jnp.int32(0)) | (~engulfer_alive)

        s2 = s.replace(
            player_hp=new_hp,
            done=new_done,
            swallow=sw.replace(
                total_timer=new_total,
                digest_timer=reset_digest,
            ),
        )

        # Resolve via sys.modules to bypass LOAD_GLOBAL — PEP 562
        # __getattr__ fires for attribute access, not LOAD_GLOBAL inside a
        # function body. Under NETHAX_BRAX_ALL=1 `release_from_engulf` is
        # deleted from globals(); the lambda would NameError at trace time.
        import sys as _sys_re
        _release_fn = getattr(_sys_re.modules[__name__], "release_from_engulf")
        return jax.lax.cond(
            should_release,
            lambda st: _release_fn(st),
            lambda st: st,
            s2,
        )

    return jax.lax.cond(
        state.swallow.swallowed,
        _tick,
        lambda s: s,
        state,
    )

# Round 4 brax integration via PEP 562 lazy __getattr__ with cycle-break.
# When inventory_brax is loading and reads back our names, return the
# stored originals to break the cycle.  After inventory_brax is fully
# loaded, subsequent lookups return the brax versions.
import os as _os_brax
import sys as _sys_brax
if _os_brax.environ.get("NETHAX_BRAX_ALL", "0") == "1":
    _BRAX_ORIG = {
        "digest_tick": digest_tick,
        "release_from_engulf": release_from_engulf,
    }
    _BRAX_MAP = {
        "digest_tick": ("inventory_brax", "digest_tick_brax"),
        "release_from_engulf": ("inventory_brax", "release_from_engulf_brax"),
    }
    _BRAX_CACHE = {}

    def _make_brax_thunk(_name):
        def _thunk(*args, **kwargs):
            if _name not in _BRAX_CACHE:
                mn, bn = _BRAX_MAP[_name]
                _BRAX_CACHE[_name] = getattr(
                    __import__(f"Nethax.nethax.subsystems.{mn}", fromlist=[bn]), bn)
            return _BRAX_CACHE[_name](*args, **kwargs)
        _thunk.__name__ = _name
        _thunk.__qualname__ = _name
        return _thunk

    # Install thunks in globals so LOAD_GLOBAL inside function bodies
    # resolves to the brax target.  PEP 562 module __getattr__ alone
    # only fires for attribute access (mod.X), not LOAD_GLOBAL.
    for _name in list(_BRAX_MAP):
        if _name in globals():
            globals()[_name] = _make_brax_thunk(_name)

    def __getattr__(name):
        if name not in _BRAX_MAP:
            raise AttributeError(name)
        mod_name, brax_name = _BRAX_MAP[name]
        full = f"Nethax.nethax.subsystems.{mod_name}"
        if full in _sys_brax.modules:
            spec = getattr(_sys_brax.modules[full], "__spec__", None)
            if spec is not None and getattr(spec, "_initializing", False):
                return _BRAX_ORIG[name]
        if name not in _BRAX_CACHE:
            _BRAX_CACHE[name] = getattr(__import__(full, fromlist=[brax_name]), brax_name)
        return _BRAX_CACHE[name]
