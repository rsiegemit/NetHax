"""Experience (XP) subsystem — byte-equal port of vendor/nethack/src/exper.c.

Implements the seven core functions of the vendor file:

  * newuexp(lev)         — XP threshold for a given level (exper.c:13-23)
  * enermod(en)          — role-specific Pw multiplier (exper.c:25-41)
  * experience(...)      — XP awarded for a slain monster (exper.c:83-166)
  * more_experienced     — u.uexp / u.urexp accumulation (exper.c:168-203)
  * losexp(drainer)      — level-drain (exper.c:206-291)
  * pluslvl(incr)        — level-up (exper.c:306-372)
  * newexplevel()        — auto level-up check (exper.c:299-304)

All functions are pure with respect to ``EnvState`` and JIT-friendly: only
``jax.lax.cond / switch`` and ``jnp.*`` arithmetic are used on traced values.
Random rolls inside ``newhp()`` and ``newpw()`` are fed from an ``rng``
parameter (callers fold-in their own salt off ``state.rng`` or another key).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as _np

from Nethax.nethax.constants.roles import ROLES, Role
from Nethax.nethax.constants.races import RACES, Race


# ---------------------------------------------------------------------------
# Vendor constants  (vendor/nethack/include/global.h:413; permonst.h:48,80;
# include/monattk.h AT_/AD_; include/monflag.h M2_NASTY).
# ---------------------------------------------------------------------------

MAXULEV: int = 30
NATTK: int = 6
NORMAL_SPEED: int = 12

# Attack-type sentinels referenced by experience() (monattk.h).
AT_BUTT: int = 4
AT_WEAP: int = 254
AT_MAGC: int = 255

# Damage-type sentinels referenced by experience() (monattk.h).
AD_PHYS: int = 0
AD_BLND: int = 11
AD_DRLI: int = 15
AD_STON: int = 18
AD_WRAP: int = 28
AD_SLIM: int = 40

# Monster-symbol id for eels (S_EEL = 57; defsym.h).
S_EEL: int = 57

# M2_NASTY bit (monflag.h:410) — extra_nasty() per include/mondata.h:120.
M2_NASTY: int = 0x02000000

# Role enum-value constants used by enermod (you.h PM_*).
PM_WIZARD    = int(Role.WIZARD)
PM_CLERIC    = int(Role.PRIEST)     # vendor PM_PRIEST is the closest Nethax analog of PM_CLERIC
PM_HEALER    = int(Role.HEALER)
PM_KNIGHT    = int(Role.KNIGHT)
PM_BARBARIAN = int(Role.BARBARIAN)
PM_VALKYRIE  = int(Role.VALKYRIE)


# ---------------------------------------------------------------------------
# Per-monster lookup tables (built once at module import — host-side only).
# Mirror MONSTERS[i].* arrays used by experience() so the JIT-compiled body
# stays free of Python casts on tracers.
# ---------------------------------------------------------------------------

def _build_monster_tables():
    from Nethax.nethax.constants.monsters import MONSTERS
    n = len(MONSTERS)
    levels = [int(m.level) for m in MONSTERS]
    ac     = [int(m.ac)    for m in MONSTERS]
    speed  = [int(m.move_speed) for m in MONSTERS]
    nasty  = [(int(m.flags2) & M2_NASTY) != 0 for m in MONSTERS]
    symbol = [int(m.symbol) for m in MONSTERS]

    # Attack tables: NATTK slots per monster, each (aatyp, adtyp, damn, damd).
    aatyp = [[0]*NATTK for _ in range(n)]
    adtyp = [[0]*NATTK for _ in range(n)]
    damn  = [[0]*NATTK for _ in range(n)]
    damd  = [[0]*NATTK for _ in range(n)]
    for i, m in enumerate(MONSTERS):
        for k, atk in enumerate(m.attacks[:NATTK]):
            at, ad, dn, dd = atk
            aatyp[i][k] = int(at)
            adtyp[i][k] = int(ad)
            damn[i][k]  = int(dn)
            damd[i][k]  = int(dd)

    return (
        jnp.array(levels, dtype=jnp.int32),
        jnp.array(ac,     dtype=jnp.int32),
        jnp.array(speed,  dtype=jnp.int32),
        jnp.array(nasty,  dtype=jnp.bool_),
        jnp.array(symbol, dtype=jnp.int32),
        jnp.array(aatyp,  dtype=jnp.int32),
        jnp.array(adtyp,  dtype=jnp.int32),
        jnp.array(damn,   dtype=jnp.int32),
        jnp.array(damd,   dtype=jnp.int32),
    )


(
    _MON_LEVEL,
    _MON_AC,
    _MON_SPEED,
    _MON_NASTY,
    _MON_SYMBOL,
    _MON_AATYP,
    _MON_ADTYP,
    _MON_DAMN,
    _MON_DAMD,
) = _build_monster_tables()


# ---------------------------------------------------------------------------
# Per-role / per-race HP & Pw advance tables for newhp() / newpw().
# Built host-side; indexed by player_role/player_race at trace time.
# Each row = (infix, inrnd, lofix, lornd, hifix, hirnd).
# ---------------------------------------------------------------------------

def _build_adv_tables():
    role_hp = [[r.hpadv.infix, r.hpadv.inrnd,
                r.hpadv.lofix, r.hpadv.lornd,
                r.hpadv.hifix, r.hpadv.hirnd] for r in ROLES]
    role_en = [[r.enadv.infix, r.enadv.inrnd,
                r.enadv.lofix, r.enadv.lornd,
                r.enadv.hifix, r.enadv.hirnd] for r in ROLES]
    role_xlev = [int(r.xlev) for r in ROLES]

    race_hp = [[r.hpadv.infix, r.hpadv.inrnd,
                r.hpadv.lofix, r.hpadv.lornd,
                r.hpadv.hifix, r.hpadv.hirnd] for r in RACES]
    race_en = [[r.enadv.infix, r.enadv.inrnd,
                r.enadv.lofix, r.enadv.lornd,
                r.enadv.hifix, r.enadv.hirnd] for r in RACES]

    return (
        jnp.array(role_hp,  dtype=jnp.int32),
        jnp.array(role_en,  dtype=jnp.int32),
        jnp.array(role_xlev, dtype=jnp.int32),
        jnp.array(race_hp,  dtype=jnp.int32),
        jnp.array(race_en,  dtype=jnp.int32),
    )


_ROLE_HPADV, _ROLE_ENADV, _ROLE_XLEV, _RACE_HPADV, _RACE_ENADV = _build_adv_tables()


# ---------------------------------------------------------------------------
# newuexp(lev)  — exper.c:13-23
# ---------------------------------------------------------------------------

def newuexp(lev) -> jnp.ndarray:
    """XP threshold required to *reach* ``lev``.

    Vendor exper.c:13-23::

        if (lev < 1)  return 0L;
        if (lev < 10) return  10L      * (1L << lev);
        if (lev < 20) return  10000L   * (1L << (lev - 10));
        return                10000000L * (long)(lev - 19);
    """
    lev_i = jnp.int64(lev)
    band_a = jnp.int64(10) * (jnp.int64(1) << jnp.maximum(lev_i, jnp.int64(0)))
    # NB: jax shifts can't be negative; we clamp.  band_a is only used when 0 <= lev_i < 10.
    band_b = jnp.int64(10000) * (jnp.int64(1) << jnp.maximum(lev_i - jnp.int64(10), jnp.int64(0)))
    band_c = jnp.int64(10000000) * (lev_i - jnp.int64(19))

    is_neg   = lev_i < jnp.int64(1)
    is_band_a = (~is_neg) & (lev_i < jnp.int64(10))
    is_band_b = (~is_neg) & (~is_band_a) & (lev_i < jnp.int64(20))

    out = jnp.where(
        is_neg, jnp.int64(0),
        jnp.where(is_band_a, band_a,
                  jnp.where(is_band_b, band_b, band_c)),
    )
    return out


# ---------------------------------------------------------------------------
# enermod(en)  — exper.c:25-41
# ---------------------------------------------------------------------------

def enermod(en, role) -> jnp.ndarray:
    """Apply role's Pw multiplier to ``en``.

    Vendor exper.c:25-41::

        WIZARD, CLERIC  -> 2*en
        HEALER, KNIGHT  -> 3*en / 2
        BARBARIAN, VALKYRIE -> 3*en / 4
        default         -> en
    """
    en_i = jnp.int32(en)
    role_i = jnp.int32(role)

    is_x2  = (role_i == jnp.int32(PM_WIZARD))    | (role_i == jnp.int32(PM_CLERIC))
    is_x32 = (role_i == jnp.int32(PM_HEALER))    | (role_i == jnp.int32(PM_KNIGHT))
    is_x34 = (role_i == jnp.int32(PM_BARBARIAN)) | (role_i == jnp.int32(PM_VALKYRIE))

    out = jnp.where(
        is_x2,  jnp.int32(2) * en_i,
        jnp.where(is_x32, (jnp.int32(3) * en_i) // jnp.int32(2),
                  jnp.where(is_x34, (jnp.int32(3) * en_i) // jnp.int32(4), en_i)),
    )
    return out


# ---------------------------------------------------------------------------
# experience(mtmp, nk)  — exper.c:83-166
# ---------------------------------------------------------------------------

def experience(entry_idx, nk, amphibious=jnp.bool_(False), mrevived=jnp.bool_(False),
               mcloned=jnp.bool_(False)) -> jnp.ndarray:
    """Return XP value of a monster with vendor table-driven bonuses.

    Parameters
    ----------
    entry_idx  : scalar int — MONSTERS table row (mtmp->data offset)
    nk         : scalar int — kill_count for repeated-kill halving (mvitals.died)
    amphibious : bool       — hero is amphibious (skip AD_WRAP+S_EEL bonus)
    mrevived   : bool       — vendor mtmp->mrevived  (mondead.c)
    mcloned    : bool       — vendor mtmp->mcloned   (makemon.c)
    """
    idx = jnp.clip(jnp.int32(entry_idx), jnp.int32(0),
                   jnp.int32(_MON_LEVEL.shape[0] - 1))

    m_lev = _MON_LEVEL[idx]
    ac    = _MON_AC[idx]
    speed = _MON_SPEED[idx]
    nasty = _MON_NASTY[idx]
    sym   = _MON_SYMBOL[idx]

    tmp = jnp.int32(1) + m_lev * m_lev

    # AC bonus: (i = find_mac(mtmp)); if i < 3: tmp += (7 - i) * (i < 0 ? 2 : 1)
    i_ac = ac
    ac_lt3 = i_ac < jnp.int32(3)
    ac_mul = jnp.where(i_ac < jnp.int32(0), jnp.int32(2), jnp.int32(1))
    ac_bonus = jnp.where(ac_lt3, (jnp.int32(7) - i_ac) * ac_mul, jnp.int32(0))
    tmp = tmp + ac_bonus

    # Speed bonus: > NORMAL_SPEED ? (> 3*NORMAL_SPEED/2 ? 5 : 3) : 0
    fast = speed > jnp.int32(NORMAL_SPEED)
    very_fast = speed > jnp.int32(3 * NORMAL_SPEED // 2)
    speed_bonus = jnp.where(fast, jnp.where(very_fast, jnp.int32(5), jnp.int32(3)), jnp.int32(0))
    tmp = tmp + speed_bonus

    # Per-attack bonuses (loop NATTK; sum aatyp/adtyp contributions).
    aatyps = _MON_AATYP[idx]   # [NATTK]
    adtyps = _MON_ADTYP[idx]
    damn   = _MON_DAMN[idx]
    damd   = _MON_DAMD[idx]

    def _slot_atk_bonus(at_):
        is_special = at_ > jnp.int32(AT_BUTT)
        return jnp.where(
            is_special,
            jnp.where(at_ == jnp.int32(AT_WEAP), jnp.int32(5),
                      jnp.where(at_ == jnp.int32(AT_MAGC), jnp.int32(10),
                                jnp.int32(3))),
            jnp.int32(0),
        )

    atk_bonus = jnp.sum(jax.vmap(_slot_atk_bonus)(aatyps)).astype(jnp.int32)
    tmp = tmp + atk_bonus

    # Per-attack damage-type bonuses.
    def _slot_dmg_bonus(ad_, dn_, dd_):
        is_mid = (ad_ > jnp.int32(AD_PHYS)) & (ad_ < jnp.int32(AD_BLND))
        is_drli = (ad_ == jnp.int32(AD_DRLI)) | (ad_ == jnp.int32(AD_STON)) | (ad_ == jnp.int32(AD_SLIM))
        is_non_phys = ad_ != jnp.int32(AD_PHYS)
        # vendor uses if/else if chain — mid wins over drli wins over non_phys.
        type_bonus = jnp.where(
            is_mid, jnp.int32(2) * m_lev,
            jnp.where(is_drli, jnp.int32(50),
                      jnp.where(is_non_phys, m_lev, jnp.int32(0))),
        )
        # extra heavy-damage bonus (damd*damn > 23).
        heavy = (dn_ * dd_) > jnp.int32(23)
        type_bonus = type_bonus + jnp.where(heavy, m_lev, jnp.int32(0))
        # AD_WRAP + S_EEL + !Amphibious -> +1000
        eel = (ad_ == jnp.int32(AD_WRAP)) & (sym == jnp.int32(S_EEL)) & (~amphibious)
        type_bonus = type_bonus + jnp.where(eel, jnp.int32(1000), jnp.int32(0))
        return type_bonus

    dmg_bonus = jnp.sum(jax.vmap(_slot_dmg_bonus)(adtyps, damn, damd)).astype(jnp.int32)
    tmp = tmp + dmg_bonus

    # "extra nasty" monsters: +7 * m_lev
    tmp = tmp + jnp.where(nasty, jnp.int32(7) * m_lev, jnp.int32(0))

    # m_lev > 8 -> +50
    tmp = tmp + jnp.where(m_lev > jnp.int32(8), jnp.int32(50), jnp.int32(0))

    # Repeated-kill halving (gated by mrevived | mcloned).
    # Loop until nk <= tmp2 or tmp == 1; iteration:
    #   tmp = (tmp + 1) / 2;  nk -= tmp2;  if i&1: tmp2 += 20
    # Maximum useful iterations bounded by tmp halving — 32 iters is well over
    # enough (tmp shrinks to 1 within log2(starting)).
    do_halve = mrevived | mcloned
    init_carry = (jnp.int32(tmp), jnp.int32(nk), jnp.int32(20), jnp.int32(0))

    def _halve_step(carry, _):
        tmp_, nk_, tmp2_, i_ = carry
        active = do_halve & (nk_ > tmp2_) & (tmp_ > jnp.int32(1))
        new_tmp = jnp.where(active, (tmp_ + jnp.int32(1)) // jnp.int32(2), tmp_)
        new_nk  = jnp.where(active, nk_ - tmp2_, nk_)
        bump = jnp.int32((i_ & jnp.int32(1)) != jnp.int32(0))
        new_tmp2 = jnp.where(active, tmp2_ + jnp.int32(20) * bump, tmp2_)
        return (new_tmp, new_nk, new_tmp2, i_ + jnp.int32(1)), None

    (final_tmp, _nk_out, _tmp2_out, _i_out), _ = jax.lax.scan(
        _halve_step, init_carry, jnp.arange(32, dtype=jnp.int32),
    )
    return final_tmp


# ---------------------------------------------------------------------------
# more_experienced(exper, rexp)  — exper.c:168-203
# ---------------------------------------------------------------------------

_LONG_MAX_I64 = jnp.asarray(_np.int64((1 << 63) - 1))


def more_experienced(state, exper, rexp):
    """Add ``exper`` to ``u.uexp`` and ``4*exper + rexp`` to ``u.urexp``.

    Wraparound guard (exper.c:177-181): if the new value goes negative while
    the increment was positive, clamp to LONG_MAX (we use INT64_MAX as the
    Python/jax 64-bit analog).
    """
    exper_i = jnp.int64(exper)
    rexp_i  = jnp.int64(rexp)

    old_exp  = jnp.int64(state.player_xp)
    old_rexp = jnp.int64(state.player_urexp)

    new_exp  = old_exp + exper_i
    rexp_incr = jnp.int64(4) * exper_i + rexp_i
    new_rexp = old_rexp + rexp_incr

    # Wrap clamp: if new went negative and increment was positive -> LONG_MAX
    new_exp = jnp.where((new_exp < jnp.int64(0)) & (exper_i > jnp.int64(0)),
                       _LONG_MAX_I64, new_exp)
    new_rexp = jnp.where((new_rexp < jnp.int64(0)) & (rexp_incr > jnp.int64(0)),
                         _LONG_MAX_I64, new_rexp)

    # Also update scoring.experience_points (running int32 accumulator the
    # rest of the codebase uses for bl_score / reward); clamp to int32 max
    # to mirror NLE topten.c behavior in the running display.
    return state.replace(
        player_xp=new_exp.astype(state.player_xp.dtype),
        player_urexp=new_rexp.astype(state.player_urexp.dtype),
        scoring=state.scoring.replace(
            score=(state.scoring.score + exper_i.astype(state.scoring.score.dtype)).astype(state.scoring.score.dtype),
            experience_points=(state.scoring.experience_points
                               + exper_i.astype(state.scoring.experience_points.dtype)).astype(state.scoring.experience_points.dtype),
            monsters_killed=(state.scoring.monsters_killed + jnp.int32(1)).astype(state.scoring.monsters_killed.dtype),
        ),
    )


# ---------------------------------------------------------------------------
# newhp() / newpw() — used by pluslvl().  Vendor attrib.c:1079, exper.c:43-81.
# These are intentionally simpler than the C originals: we omit the
# polymorph-time HP-throttling at level == MAXULEV branch and the Pw 1200/600
# clamp, but the per-level increment that gets stored in uhpinc/ueninc is
# byte-equal for ulevel < MAXULEV.
# ---------------------------------------------------------------------------

def _rnd_pos(rng, n):
    """rnd(n): returns 1..n if n > 0, else 0."""
    n_i = jnp.int32(n)
    pos = n_i > jnp.int32(0)
    safe_n = jnp.maximum(n_i, jnp.int32(1))
    roll = jax.random.randint(rng, (), minval=1, maxval=safe_n + jnp.int32(1),
                              dtype=jnp.int32)
    return jnp.where(pos, roll, jnp.int32(0))


def newhp(state, rng):
    """Per-level HP increment (attrib.c::newhp lines 1079-1141, ulevel > 0 branch).

    Returns ``(hp_inc, state')`` where state' has player_uhpinc[ulevel] set
    when ulevel < MAXULEV (vendor lines 1129-1131).  Caller is responsible
    for adding hp_inc to player_hp / player_hp_max.
    """
    rng_a, rng_b = jax.random.split(rng)

    role_i = jnp.clip(jnp.int32(state.player_role), jnp.int32(0),
                      jnp.int32(_ROLE_HPADV.shape[0] - 1))
    race_i = jnp.clip(jnp.int32(state.player_race), jnp.int32(0),
                      jnp.int32(_RACE_HPADV.shape[0] - 1))

    r_hp = _ROLE_HPADV[role_i]
    rc_hp = _RACE_HPADV[race_i]
    xlev  = _ROLE_XLEV[role_i]
    ulev  = jnp.int32(state.player_xl)

    lo = ulev < xlev
    fix = jnp.where(lo, r_hp[2] + rc_hp[2], r_hp[4] + rc_hp[4])
    rnd_role = jnp.where(lo, r_hp[3], r_hp[5])
    rnd_race = jnp.where(lo, rc_hp[3], rc_hp[5])

    hp = fix + _rnd_pos(rng_a, rnd_role) + _rnd_pos(rng_b, rnd_race)

    # Con bonus (attrib.c:1111-1124).
    con = jnp.int32(state.player_con)
    conplus = jnp.where(
        con <= jnp.int32(3),  jnp.int32(-2),
        jnp.where(con <= jnp.int32(6),  jnp.int32(-1),
        jnp.where(con <= jnp.int32(14), jnp.int32(0),
        jnp.where(con <= jnp.int32(16), jnp.int32(1),
        jnp.where(con == jnp.int32(17), jnp.int32(2),
        jnp.where(con == jnp.int32(18), jnp.int32(3),
                  jnp.int32(4)))))))
    hp = hp + conplus
    hp = jnp.maximum(hp, jnp.int32(1))

    # Store increment in u.uhpinc[ulevel] (only valid when ulevel < MAXULEV).
    in_range = (ulev >= jnp.int32(0)) & (ulev < jnp.int32(MAXULEV))
    safe_ulev = jnp.clip(ulev, jnp.int32(0), jnp.int32(state.player_uhpinc.shape[0] - 1))
    new_uhpinc = state.player_uhpinc.at[safe_ulev].set(
        jnp.where(in_range, hp.astype(state.player_uhpinc.dtype),
                  state.player_uhpinc[safe_ulev])
    )
    new_state = state.replace(player_uhpinc=new_uhpinc)
    return hp, new_state


def newpw(state, rng):
    """Per-level Pw increment (exper.c:43-81, u.ulevel > 0 branch).

    Vendor::
        enrnd = ACURR(A_WIS)/2;
        if (ulevel < role.xlev): enrnd += role.lornd + race.lornd
                                  enfix  = role.lofix + race.lofix
        else:                    enrnd += role.hirnd + race.hirnd
                                  enfix  = role.hifix + race.hifix
        en = enermod( rn1(enrnd, enfix) )
        if (en <= 0) en = 1
        u.ueninc[ulevel] = en
    """
    rng_a, _ = jax.random.split(rng)

    role_i = jnp.clip(jnp.int32(state.player_role), jnp.int32(0),
                      jnp.int32(_ROLE_ENADV.shape[0] - 1))
    race_i = jnp.clip(jnp.int32(state.player_race), jnp.int32(0),
                      jnp.int32(_RACE_ENADV.shape[0] - 1))

    r_en  = _ROLE_ENADV[role_i]
    rc_en = _RACE_ENADV[race_i]
    xlev  = _ROLE_XLEV[role_i]
    ulev  = jnp.int32(state.player_xl)

    enrnd = jnp.int32(state.player_wis) // jnp.int32(2)

    lo = ulev < xlev
    enrnd = enrnd + jnp.where(lo, r_en[3] + rc_en[3], r_en[5] + rc_en[5])
    enfix = jnp.where(lo, r_en[2] + rc_en[2], r_en[4] + rc_en[4])

    # rn1(enrnd, enfix) = enfix + rn2(enrnd)  (rnd.h).  Guard enrnd>0.
    enrnd_safe = jnp.maximum(enrnd, jnp.int32(1))
    rn2_val = jax.random.randint(rng_a, (), minval=0, maxval=enrnd_safe,
                                  dtype=jnp.int32)
    base = enfix + jnp.where(enrnd > jnp.int32(0), rn2_val, jnp.int32(0))

    en = enermod(base, state.player_role)
    en = jnp.maximum(en, jnp.int32(1))

    in_range = (ulev >= jnp.int32(0)) & (ulev < jnp.int32(MAXULEV))
    safe_ulev = jnp.clip(ulev, jnp.int32(0), jnp.int32(state.player_ueninc.shape[0] - 1))
    new_ueninc = state.player_ueninc.at[safe_ulev].set(
        jnp.where(in_range, en.astype(state.player_ueninc.dtype),
                  state.player_ueninc[safe_ulev])
    )
    new_state = state.replace(player_ueninc=new_ueninc)
    return en, new_state


# ---------------------------------------------------------------------------
# losexp(drainer)  — exper.c:206-291
# ---------------------------------------------------------------------------

def losexp(state):
    """Drain one experience level.

    Lossy port: drli-resistance, livelog messages, polymorph rehumanize, and
    the drainer-killed branch (death at ulevel==1 from drain) are all skipped
    — those are out of scope of this XP module.  HP / Pw shaving and the
    ``uexp = newuexp(ulevel) - 1`` resync are byte-equal.
    """
    ulev = jnp.int32(state.player_xl)
    can_lose = ulev > jnp.int32(1)

    new_ulev = jnp.where(can_lose, ulev - jnp.int32(1), ulev).astype(state.player_xl.dtype)

    # Index uhpinc/ueninc at the *new* ulevel  (vendor: u.uhpinc[u.ulevel]
    # after the decrement, lines 251/269).  When can_lose is False we keep 0.
    safe_idx = jnp.clip(new_ulev.astype(jnp.int32), jnp.int32(0),
                        jnp.int32(state.player_uhpinc.shape[0] - 1))
    hp_dec = jnp.where(can_lose,
                       jnp.int32(state.player_uhpinc[safe_idx]),
                       jnp.int32(0))
    en_dec = jnp.where(can_lose,
                       jnp.int32(state.player_ueninc[safe_idx]),
                       jnp.int32(0))

    # HP_max shaving with minimum floor = max(ulevel, 10).
    uhpmin = jnp.maximum(new_ulev.astype(jnp.int32), jnp.int32(10))
    new_hp_max = jnp.int32(state.player_hp_max) - hp_dec
    new_hp_max = jnp.maximum(new_hp_max, uhpmin)
    # Vendor lines 260-261: don't allow uhpmax to rise above the original.
    new_hp_max = jnp.minimum(new_hp_max, jnp.int32(state.player_hp_max))

    new_hp = jnp.int32(state.player_hp) - hp_dec
    new_hp = jnp.where(new_hp < jnp.int32(1), jnp.int32(1), new_hp)
    new_hp = jnp.minimum(new_hp, new_hp_max)

    # Pw shaving.
    new_pw_max = jnp.int32(state.player_pw_max) - en_dec
    new_pw_max = jnp.maximum(new_pw_max, jnp.int32(0))
    new_pw = jnp.int32(state.player_pw) - en_dec
    new_pw = jnp.maximum(new_pw, jnp.int32(0))
    new_pw = jnp.minimum(new_pw, new_pw_max)

    # uexp resync: if u.uexp > 0 -> newuexp(ulevel) - 1
    cur_uexp = jnp.int64(state.player_xp)
    resync_target = newuexp(new_ulev) - jnp.int64(1)
    new_uexp = jnp.where(cur_uexp > jnp.int64(0), resync_target, cur_uexp)

    return state.replace(
        player_xl=new_ulev,
        player_hp_max=new_hp_max.astype(state.player_hp_max.dtype),
        player_hp=new_hp.astype(state.player_hp.dtype),
        player_pw_max=new_pw_max.astype(state.player_pw_max.dtype),
        player_pw=new_pw.astype(state.player_pw.dtype),
        player_xp=new_uexp.astype(state.player_xp.dtype),
    )


# ---------------------------------------------------------------------------
# pluslvl(incr)  — exper.c:306-372
# ---------------------------------------------------------------------------

def pluslvl(state, rng, incr: bool = True):
    """Gain one experience level.

    Vendor exper.c:306-372.  Polymorph branch is skipped (not modelled here).
    On level-up we:
        * HP_max += newhp();  HP += newhp()
        * Pw_max += newpw();  Pw += newpw()  (already enermod-scaled)
        * ulevel += 1 (clamped to MAXULEV)
        * if (incr): uexp = min(uexp, newuexp(ulevel+1) - 1)
          else:      uexp = newuexp(ulevel)
    """
    rng_hp, rng_pw = jax.random.split(rng)

    ulev = jnp.int32(state.player_xl)
    can_level = ulev < jnp.int32(MAXULEV)

    hp_inc, state = newhp(state, rng_hp)
    en_inc, state = newpw(state, rng_pw)

    # Only apply increments when can_level (vendor: at MAXULEV the throttle
    # branch of newhp/newpw clamps the increment, but we follow the simpler
    # rule of skipping the bump at the cap).
    hp_inc_eff = jnp.where(can_level, hp_inc, jnp.int32(0))
    en_inc_eff = jnp.where(can_level, en_inc, jnp.int32(0))

    new_hp_max = jnp.int32(state.player_hp_max) + hp_inc_eff
    new_hp = jnp.int32(state.player_hp) + hp_inc_eff
    new_hp = jnp.minimum(new_hp, new_hp_max)

    new_pw_max = jnp.int32(state.player_pw_max) + en_inc_eff
    new_pw = jnp.int32(state.player_pw) + en_inc_eff
    new_pw = jnp.minimum(new_pw, new_pw_max)

    new_ulev = jnp.where(can_level, ulev + jnp.int32(1), ulev)

    # uexp resync.
    cur_uexp = jnp.int64(state.player_xp)
    if incr:
        # vendor: if (u.uexp >= newuexp(u.ulevel+1)) u.uexp = newuexp(u.ulevel+1) - 1
        thresh = newuexp(new_ulev + jnp.int32(1)) - jnp.int64(1)
        new_uexp = jnp.where(can_level & (cur_uexp >= thresh + jnp.int64(1)), thresh, cur_uexp)
    else:
        new_uexp = jnp.where(can_level, newuexp(new_ulev), cur_uexp)

    return state.replace(
        player_xl=new_ulev.astype(state.player_xl.dtype),
        player_hp_max=new_hp_max.astype(state.player_hp_max.dtype),
        player_hp=new_hp.astype(state.player_hp.dtype),
        player_pw_max=new_pw_max.astype(state.player_pw_max.dtype),
        player_pw=new_pw.astype(state.player_pw.dtype),
        player_xp=new_uexp.astype(state.player_xp.dtype),
    )


# ---------------------------------------------------------------------------
# newexplevel()  — exper.c:299-304
# ---------------------------------------------------------------------------

def newexplevel(state, rng):
    """If ulevel<MAXULEV and uexp >= newuexp(ulevel), call pluslvl(TRUE).

    JIT-pure: ``lax.cond`` gates the level-up branch.
    """
    ulev = jnp.int32(state.player_xl)
    uexp = jnp.int64(state.player_xp)
    thresh = newuexp(ulev)
    should = (ulev < jnp.int32(MAXULEV)) & (uexp >= thresh)
    return jax.lax.cond(
        should,
        lambda s: pluslvl(s, rng, incr=True),
        lambda s: s,
        state,
    )
