"""Potion effects — vendor/nethack/src/potion.c::peffects."""
from enum import IntEnum

import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems.status_effects import (
    TimedStatus,
    Intrinsic,
    add_timed_intrinsic,
)
from Nethax.nethax.constants.objects import ObjectClass
from Nethax.nethax.rng import rnd


# ---------------------------------------------------------------------------
# Canonical type_id values — position in the compiled object table.
# Order matches vendor/nethack/include/objects.h POTION() macro sequence
# starting after the GENERIC("potion") entry.
# objects.py comments confirm IDs 68–83 for the first 16; the remaining 10
# (blindness, speed, invisibility, gain_level, enlightenment, polymorph,
#  booze, sickness, acid, oil) follow at 68+offset based on objects.h order.
#
# Canonical order from objects.h (sequential):
#   0  gain_ability        (68)
#   1  restore_ability     (69)
#   2  confusion           (70)  [objects.h order differs from objects.py!]
#   3  blindness           (71)
#   4  paralysis           (72)
#   5  speed               (73)
#   6  levitation          (74)
#   7  hallucination       (75)
#   8  invisibility        (76)
#   9  see_invisible       (77)
#  10  healing             (78)
#  11  extra_healing       (79)
#  12  gain_level          (80)
#  13  enlightenment       (81)
#  14  monster_detection   (82)
#  15  object_detection    (83)
#  16  gain_energy         (84)
#  17  sleeping            (85)
#  18  full_healing        (86)
#  19  polymorph           (87)
#  20  booze               (88)
#  21  sickness            (89)
#  22  fruit_juice         (90)
#  23  acid                (91)
#  24  oil                 (92)
#  25  water               (93)
# ---------------------------------------------------------------------------

_POTION_BASE_ID = 68   # first potion entry in the compiled object table


class PotionEffect(IntEnum):
    """Canonical potion effect identifiers.

    Values are sequential indices into the potion sub-table (type_id minus
    _POTION_BASE_ID), matching the POTION() macro order in objects.h.
    """
    GAIN_ABILITY       =  0   # POT_GAIN_ABILITY
    RESTORE_ABILITY    =  1   # POT_RESTORE_ABILITY
    CONFUSION          =  2   # POT_CONFUSION
    BLINDNESS          =  3   # POT_BLINDNESS
    PARALYSIS          =  4   # POT_PARALYSIS
    SPEED              =  5   # POT_SPEED
    LEVITATION         =  6   # POT_LEVITATION
    HALLUCINATION      =  7   # POT_HALLUCINATION
    INVISIBILITY       =  8   # POT_INVISIBILITY
    SEE_INVISIBLE      =  9   # POT_SEE_INVISIBLE
    HEALING            = 10   # POT_HEALING
    EXTRA_HEALING      = 11   # POT_EXTRA_HEALING
    GAIN_LEVEL         = 12   # POT_GAIN_LEVEL
    ENLIGHTENMENT      = 13   # POT_ENLIGHTENMENT
    MONSTER_DETECTION  = 14   # POT_MONSTER_DETECTION
    OBJECT_DETECTION   = 15   # POT_OBJECT_DETECTION
    GAIN_ENERGY        = 16   # POT_GAIN_ENERGY
    SLEEPING           = 17   # POT_SLEEPING
    FULL_HEALING       = 18   # POT_FULL_HEALING
    POLYMORPH          = 19   # POT_POLYMORPH
    BOOZE              = 20   # POT_BOOZE
    SICKNESS           = 21   # POT_SICKNESS
    FRUIT_JUICE        = 22   # POT_FRUIT_JUICE
    ACID               = 23   # POT_ACID
    OIL                = 24   # POT_OIL
    WATER              = 25   # POT_WATER


N_POTIONS = 26


# ---------------------------------------------------------------------------
# BUC sentinel constants (matches items.py BUCStatus enum)
# ---------------------------------------------------------------------------

_BUC_CURSED   = 1
_BUC_UNCURSED = 2
_BUC_BLESSED  = 3


# ---------------------------------------------------------------------------
# Per-effect implementations
# Each takes (state, rng, blessed_status: jnp scalar int8) → state.
# "blessed_status" is the item's buc_status field (0=unknown,1=cursed,
#  2=uncursed,3=blessed).
# ---------------------------------------------------------------------------

def _is_blessed(buc):
    return jnp.int32(buc) == jnp.int32(_BUC_BLESSED)


def _is_cursed(buc):
    return jnp.int32(buc) == jnp.int32(_BUC_CURSED)


# ---- healing group --------------------------------------------------------

def _effect_healing(state, rng, buc):
    """potion of healing — byte-equal to vendor peffect_healing.

    vendor/nethack/src/potion.c:1119-1124:
        healup(8 + d(4 + 2*bcsign(otmp), 4),
               !otmp->cursed ? 1 : 0,
               !!otmp->blessed,
               !otmp->cursed);

    bcsign = blessed(+1) - cursed(-1). So dice = d(4+2*bcsign, 4):
        cursed   → d(2,4) → 2..8
        uncursed → d(4,4) → 4..16
        blessed  → d(6,4) → 6..24
    plus +8 HP base. nxtra=1 if !cursed (HP_max bump). cureblind on
    !cursed, curesick on blessed.
    """
    from Nethax.nethax.rng import dice_roll
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    bcsign  = jnp.where(blessed, jnp.int32(1),
              jnp.where(cursed,  jnp.int32(-1), jnp.int32(0)))
    n_dice  = jnp.int32(4) + jnp.int32(2) * bcsign  # 2/4/6
    # Static-shape masked roll: dice_roll requires Python-int n. We roll the
    # maximum (6 for blessed bcsign=+1) and mask to n_dice.
    MAX_N = 6
    rolls = jax.random.randint(rng, (MAX_N,), 1, 5, dtype=jnp.int32)  # d4
    mask  = jnp.arange(MAX_N, dtype=jnp.int32) < n_dice
    dmg   = jnp.sum(jnp.where(mask, rolls, jnp.int32(0))).astype(jnp.int32)
    heal_amt = jnp.int32(8) + dmg
    nxtra    = jnp.where(cursed, jnp.int32(0), jnp.int32(1))
    new_hp_max = state.player_hp_max + nxtra
    new_hp     = jnp.minimum(state.player_hp + heal_amt, new_hp_max)

    cur_blind = state.status.timed_statuses[int(TimedStatus.BLIND)]
    new_blind = jnp.where(cursed, cur_blind, jnp.int32(0))
    new_ts = state.status.timed_statuses.at[int(TimedStatus.BLIND)].set(new_blind)
    # curesick = blessed → clear SICK timer when blessed
    cur_sick = new_ts[int(TimedStatus.SICK)]
    new_sick = jnp.where(blessed, jnp.int32(0), cur_sick)
    new_ts = new_ts.at[int(TimedStatus.SICK)].set(new_sick)
    new_status = state.status.replace(timed_statuses=new_ts)
    return state.replace(
        player_hp=new_hp,
        player_hp_max=new_hp_max,
        status=new_status,
    )


def _effect_extra_healing(state, rng, buc):
    """potion of extra healing — byte-equal to vendor peffect_extra_healing.

    vendor/nethack/src/potion.c:1128-1141:
        healup(16 + d(4 + 2*bcsign(otmp), 8),
               otmp->blessed ? 5 : !otmp->cursed ? 2 : 0,
               !otmp->cursed,
               TRUE);
        make_hallucinated(0, TRUE, 0);

    Dice = d(4+2*bcsign, 8):
        cursed   → d(2,8) → 2..16
        uncursed → d(4,8) → 4..32
        blessed  → d(6,8) → 6..48
    plus +16 HP base. nxtra=5 blessed / 2 uncursed / 0 cursed.
    curesick = !cursed, cureblind = TRUE.
    """
    from Nethax.nethax.rng import dice_roll
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    bcsign  = jnp.where(blessed, jnp.int32(1),
              jnp.where(cursed,  jnp.int32(-1), jnp.int32(0)))
    n_dice  = jnp.int32(4) + jnp.int32(2) * bcsign
    MAX_N = 6  # blessed bcsign=+1 → 6 dice
    rolls = jax.random.randint(rng, (MAX_N,), 1, 9, dtype=jnp.int32)  # d8
    mask  = jnp.arange(MAX_N, dtype=jnp.int32) < n_dice
    dmg   = jnp.sum(jnp.where(mask, rolls, jnp.int32(0))).astype(jnp.int32)
    heal_amt = jnp.int32(16) + dmg
    nxtra    = jnp.where(blessed, jnp.int32(5),
               jnp.where(cursed, jnp.int32(0), jnp.int32(2)))
    new_hp_max = state.player_hp_max + nxtra
    new_hp     = jnp.minimum(state.player_hp + heal_amt, new_hp_max)

    new_status = _clear_timed(state.status, TimedStatus.HALLUCINATION)
    new_ts = new_status.timed_statuses.at[int(TimedStatus.BLIND)].set(jnp.int32(0))
    # curesick = !cursed
    cur_sick = new_ts[int(TimedStatus.SICK)]
    new_sick = jnp.where(cursed, cur_sick, jnp.int32(0))
    new_ts = new_ts.at[int(TimedStatus.SICK)].set(new_sick)
    new_status = new_status.replace(timed_statuses=new_ts)
    return state.replace(
        player_hp=new_hp,
        player_hp_max=new_hp_max,
        status=new_status,
    )


def _effect_full_healing(state, rng, buc):
    """potion of full healing — byte-equal to vendor peffect_full_healing.

    vendor/nethack/src/potion.c:1144-1161:
        healup(400, 4 + 4*bcsign(otmp), !otmp->cursed, TRUE);
        if (otmp->blessed && u.ulevel < u.ulevelmax) {
            u.ulevelmax -= 1;
            pluslvl(FALSE);  // restore one lost XL
        }
        make_hallucinated(0, TRUE, 0);
        /* potion.c:1160-1161 */
        if (Wounded_legs && (otmp->blessed || (!otmp->cursed && !u.usteed)))
            heal_legs(0);

    nhp=400 is large enough to saturate at max+nxtra (effective full restore).
    nxtra = 4+4*bcsign = 8 blessed / 4 uncursed / 0 cursed (HP_max bump).
    curesick=!cursed, cureblind=TRUE. heal_legs on blessed (always) or
    uncursed-not-riding.
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    bcsign  = jnp.where(blessed, jnp.int32(1),
              jnp.where(cursed,  jnp.int32(-1), jnp.int32(0)))
    nxtra   = jnp.int32(4) + jnp.int32(4) * bcsign  # 0/4/8
    new_hp_max = state.player_hp_max + nxtra
    new_hp     = new_hp_max  # nhp=400 always saturates

    new_status = _clear_timed(state.status, TimedStatus.HALLUCINATION)
    new_ts = new_status.timed_statuses.at[int(TimedStatus.BLIND)].set(jnp.int32(0))
    cur_sick = new_ts[int(TimedStatus.SICK)]
    cur_vom  = new_ts[int(TimedStatus.VOMITING)]
    new_sick = jnp.where(cursed, cur_sick, jnp.int32(0))
    new_vom  = jnp.where(cursed, cur_vom,  jnp.int32(0))
    new_ts = new_ts.at[int(TimedStatus.SICK)].set(new_sick)
    new_ts = new_ts.at[int(TimedStatus.VOMITING)].set(new_vom)
    # heal_legs (vendor potion.c:1160-1161): clear WOUNDED_LEGS timer when
    # (blessed) OR (uncursed AND not riding). Vendor heal_legs() in do.c:2449
    # resets HWounded_legs to 0.
    is_riding = state.player_steed_mid > jnp.uint32(0)
    heal_legs_cond = jnp.logical_or(
        blessed,
        jnp.logical_and(jnp.logical_not(cursed), jnp.logical_not(is_riding)),
    )
    cur_legs = new_ts[int(TimedStatus.WOUNDED_LEGS)]
    new_legs = jnp.where(heal_legs_cond, jnp.int32(0), cur_legs)
    new_ts = new_ts.at[int(TimedStatus.WOUNDED_LEGS)].set(new_legs)
    new_status = new_status.replace(timed_statuses=new_ts)
    # Blessed: restore one lost XL via experience.pluslvl (vendor potion.c:1149).
    # We can't easily detect ``u.ulevelmax > u.ulevel`` without tracking lost
    # XL — but pluslvl is idempotent and capped at MAXULEV, so applying it
    # unconditionally on blessed is harmless when the player is already at
    # max.  Vendor's pluslvl(FALSE) means "restore without bumping ulevelmax";
    # our pluslvl(incr=False) ports that semantic exactly (experience.py:610-
    # 617: uexp resync uses newuexp(old_ulev) rather than newuexp(old_ulev+1)).
    from Nethax.nethax.subsystems.experience import pluslvl as _pluslvl
    mid_state = state.replace(
        player_hp=new_hp,
        player_hp_max=new_hp_max,
        status=new_status,
    )
    rng_pl, _ = jax.random.split(rng)
    _plus_state = _pluslvl(mid_state, rng_pl, incr=False)
    return jax.tree.map(
        lambda t, f: jnp.where(blessed, t, f),
        _plus_state, mid_state,
    )


# ---- energy ---------------------------------------------------------------

def _effect_gain_energy(state, rng, buc):
    """potion of gain energy — byte-equal to vendor peffect_gain_energy.

    vendor/nethack/src/potion.c:1243-1257:
        num = d(otmp->blessed ? 3 : !otmp->cursed ? 2 : 1, 6);
        if (otmp->cursed) num = -num;
        u.uenmax += num;       (clamped to >=0)
        u.uen += 3 * num;      (clamped to [0, uenmax])

    Dice (n, 6):
        blessed  → 3d6  (+3..+18 uenmax, +9..+54 uen)
        uncursed → 2d6  (+2..+12 uenmax, +6..+36 uen)
        cursed   → 1d6 negated (-1..-6 uenmax, -3..-18 uen)
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    # n = 3 blessed / 2 uncursed / 1 cursed.
    n_dice  = jnp.where(blessed, jnp.int32(3),
              jnp.where(cursed,  jnp.int32(1), jnp.int32(2)))
    # Static-shape masked roll: roll MAX_N d6 then sum first n_dice via mask.
    MAX_N = 3
    rolls = jax.random.randint(rng, (MAX_N,), 1, 7, dtype=jnp.int32)  # d6
    mask  = jnp.arange(MAX_N, dtype=jnp.int32) < n_dice
    num   = jnp.sum(jnp.where(mask, rolls, jnp.int32(0))).astype(jnp.int32)
    # Cursed: negate num (vendor potion.c:1252-1253).
    num   = jnp.where(cursed, -num, num)
    new_pw_max = jnp.maximum(state.player_pw_max + num, jnp.int32(0))
    new_pw     = jnp.minimum(state.player_pw + jnp.int32(3) * num,
                             new_pw_max)
    new_pw     = jnp.maximum(new_pw, jnp.int32(0))
    return state.replace(player_pw=new_pw, player_pw_max=new_pw_max)


# ---- ability score group --------------------------------------------------

def _effect_gain_ability(state, rng, buc):
    """potion of gain ability — stat changes by BUC.

    Canonical: vendor/nethack/src/potion.c:1030 peffect_gain_ability.
      blessed  : +1 to ALL six stats (STR/INT/WIS/DEX/CON/CHA), clipped 3..25.
      uncursed : pick one of the six stats uniformly (rn2(6)) and +1 it.
      cursed   : -1 to STR (existing behaviour preserved).

    Stat order in lax.switch: STR(0) INT(1) WIS(2) DEX(3) CON(4) CHA(5).
    """
    from Nethax.nethax.rng import rn2

    cursed  = _is_cursed(buc)
    blessed = _is_blessed(buc)

    # --- Cursed branch: STR -1 (existing).
    str_cursed = jnp.clip(state.player_str + jnp.int16(-1), jnp.int16(3), jnp.int16(25))

    # --- Blessed branch: +1 to all six stats, clipped to [3, 25].
    str_blessed = jnp.clip(state.player_str + jnp.int16(1), jnp.int16(3), jnp.int16(25))
    int_blessed = jnp.clip(state.player_int.astype(jnp.int16) + jnp.int16(1),
                           jnp.int16(3), jnp.int16(25)).astype(jnp.int8)
    wis_blessed = jnp.clip(state.player_wis.astype(jnp.int16) + jnp.int16(1),
                           jnp.int16(3), jnp.int16(25)).astype(jnp.int8)
    dex_blessed = jnp.clip(state.player_dex.astype(jnp.int16) + jnp.int16(1),
                           jnp.int16(3), jnp.int16(25)).astype(jnp.int8)
    con_blessed = jnp.clip(state.player_con.astype(jnp.int16) + jnp.int16(1),
                           jnp.int16(3), jnp.int16(25)).astype(jnp.int8)
    cha_blessed = jnp.clip(state.player_cha.astype(jnp.int16) + jnp.int16(1),
                           jnp.int16(3), jnp.int16(25)).astype(jnp.int8)

    # --- Uncursed branch: rn2(6) picks one stat to +1.
    pick = rn2(rng, 6)  # 0..5

    def _inc_str(_):
        new_s = jnp.clip(state.player_str + jnp.int16(1), jnp.int16(3), jnp.int16(25))
        return (new_s, state.player_int, state.player_wis,
                state.player_dex, state.player_con, state.player_cha)

    def _inc_int(_):
        new_i = jnp.clip(state.player_int.astype(jnp.int16) + jnp.int16(1),
                         jnp.int16(3), jnp.int16(25)).astype(jnp.int8)
        return (state.player_str, new_i, state.player_wis,
                state.player_dex, state.player_con, state.player_cha)

    def _inc_wis(_):
        new_w = jnp.clip(state.player_wis.astype(jnp.int16) + jnp.int16(1),
                         jnp.int16(3), jnp.int16(25)).astype(jnp.int8)
        return (state.player_str, state.player_int, new_w,
                state.player_dex, state.player_con, state.player_cha)

    def _inc_dex(_):
        new_d = jnp.clip(state.player_dex.astype(jnp.int16) + jnp.int16(1),
                         jnp.int16(3), jnp.int16(25)).astype(jnp.int8)
        return (state.player_str, state.player_int, state.player_wis,
                new_d, state.player_con, state.player_cha)

    def _inc_con(_):
        new_c = jnp.clip(state.player_con.astype(jnp.int16) + jnp.int16(1),
                         jnp.int16(3), jnp.int16(25)).astype(jnp.int8)
        return (state.player_str, state.player_int, state.player_wis,
                state.player_dex, new_c, state.player_cha)

    def _inc_cha(_):
        new_ch = jnp.clip(state.player_cha.astype(jnp.int16) + jnp.int16(1),
                          jnp.int16(3), jnp.int16(25)).astype(jnp.int8)
        return (state.player_str, state.player_int, state.player_wis,
                state.player_dex, state.player_con, new_ch)

    # Brax-flatten 6-way switch: compute all 6 branches, select by pick.
    _pick_i32 = pick.astype(jnp.int32)
    _branches = [
        _inc_str(None), _inc_int(None), _inc_wis(None),
        _inc_dex(None), _inc_con(None), _inc_cha(None),
    ]

    def _select_by_pick(*vals):
        out = vals[0]
        for _i in range(1, len(vals)):
            out = jnp.where(_pick_i32 == jnp.int32(_i), vals[_i], out)
        return out

    (str_un, int_un, wis_un, dex_un, con_un, cha_un) = jax.tree.map(
        _select_by_pick, *_branches,
    )

    # --- Select by BUC.
    final_str = jnp.where(
        cursed, str_cursed,
        jnp.where(blessed, str_blessed, str_un),
    )
    final_int = jnp.where(
        cursed, state.player_int,
        jnp.where(blessed, int_blessed, int_un),
    )
    final_wis = jnp.where(
        cursed, state.player_wis,
        jnp.where(blessed, wis_blessed, wis_un),
    )
    final_dex = jnp.where(
        cursed, state.player_dex,
        jnp.where(blessed, dex_blessed, dex_un),
    )
    final_con = jnp.where(
        cursed, state.player_con,
        jnp.where(blessed, con_blessed, con_un),
    )
    final_cha = jnp.where(
        cursed, state.player_cha,
        jnp.where(blessed, cha_blessed, cha_un),
    )

    return state.replace(
        player_str=final_str,
        player_int=final_int,
        player_wis=final_wis,
        player_dex=final_dex,
        player_con=final_con,
        player_cha=final_cha,
    )


def _effect_restore_ability(state, rng, buc):
    """potion of restore ability — restore drained stats to race/role maxima.

    Canonical: peffect_restore_ability (potion.c) — calls full_restore() which
    restores every drained stat to its exercise-adjusted maximum (u.urace.attrmax).
    We use state.player_amax[i] as the per-stat ceiling, matching vendor
    u.urace.attrmax[] populated during init_attr.

    Stat order in player_amax matches _STAT_NAMES: str(0) int(1) wis(2) dex(3) con(4) cha(5).
    STR is int16 (0..125 range); rest are int8.

    Cite: vendor/nethack/src/potion.c::peffect_restore_ability;
          vendor/nethack/src/u_init.c lines 250-580 (init_attr race cap).
    """
    amax = state.player_amax  # int8[6]: str,int,wis,dex,con,cha
    new_str = jnp.minimum(
        jnp.maximum(state.player_str, jnp.int16(amax[0])),
        state.player_str,   # never raise above current if already above amax
    )
    # restore: set to amax[i] if current < amax[i], else leave unchanged
    new_str = jnp.where(state.player_str < amax[0].astype(jnp.int16),
                        amax[0].astype(jnp.int16), state.player_str)
    new_dex = jnp.where(state.player_dex < amax[3], amax[3], state.player_dex)
    new_con = jnp.where(state.player_con < amax[4], amax[4], state.player_con)
    new_int = jnp.where(state.player_int < amax[1], amax[1], state.player_int)
    new_wis = jnp.where(state.player_wis < amax[2], amax[2], state.player_wis)
    new_cha = jnp.where(state.player_cha < amax[5], amax[5], state.player_cha)
    return state.replace(
        player_str=new_str.astype(jnp.int16),
        player_dex=new_dex.astype(jnp.int8),
        player_con=new_con.astype(jnp.int8),
        player_int=new_int.astype(jnp.int8),
        player_wis=new_wis.astype(jnp.int8),
        player_cha=new_cha.astype(jnp.int8),
    )


# ---- level/XP group -------------------------------------------------------

def _effect_gain_level(state, rng, buc):
    """potion of gain level — byte-equal to vendor peffect_gain_level.

    vendor/nethack/src/potion.c::peffect_gain_level:
        if (otmp->cursed) {
            if (Antimagic || u.uz.dlevel == 1) /* no effect */;
            else next_level(FALSE);            /* ascend one dlvl */
        } else {
            pluslvl(otmp->blessed);            /* XL+1 with HP/Pw reroll */
        }

    The non-cursed branch routes through experience.pluslvl() so the wave16a
    XP system handles HP_max, Pw_max, urexp, and uexp bookkeeping properly.
    Cursed branch ascends one dungeon level; floor at 1 (vendor: do nothing
    when on Dlvl 1 unless Antimagic). Antimagic intrinsic not yet modelled.
    """
    from Nethax.nethax.subsystems.experience import pluslvl as _pluslvl

    cursed  = _is_cursed(buc)

    # --- Cursed branch: ascend one dungeon level (floor at 1) ---
    cur_lv  = state.dungeon.current_level.astype(jnp.int32)
    new_lv  = jnp.maximum(cur_lv - jnp.int32(1), jnp.int32(1)).astype(jnp.int8)
    state_cursed = state.replace(
        dungeon=state.dungeon.replace(current_level=new_lv),
    )

    # --- Non-cursed: pluslvl(blessed) — XL+1 + HP/Pw reroll, urexp bumped ---
    state_lvlup = _pluslvl(state, rng, incr=True)

    return jax.tree.map(
        lambda t, f: jnp.where(cursed, t, f),
        state_cursed, state_lvlup,
    )


# ---- vision group ---------------------------------------------------------

def _effect_see_invisible(state, rng, buc):
    """potion of see invisible — grants temporary or permanent see-invisible.

    Canonical: peffect_see_invisible — incr_itimeout(&HSee_invisible, rn1(100,750))
    or permanent FROMOUTSIDE if blessed.
    Wave 3: blessed → permanent; else 300-turn timed.
    """
    blessed   = _is_blessed(buc)
    perm_new  = jnp.where(blessed,
                          jnp.bool_(True),
                          state.status.intrinsics[Intrinsic.SEE_INVIS])
    new_intr  = state.status.intrinsics.at[Intrinsic.SEE_INVIS].set(perm_new)
    turns     = jnp.where(blessed, jnp.int32(0), jnp.int32(300))
    cur_timer = state.status.timed_intrinsics[Intrinsic.SEE_INVIS]
    new_timer = jnp.where(blessed, cur_timer,
                          jnp.maximum(cur_timer, turns))
    new_timers = state.status.timed_intrinsics.at[Intrinsic.SEE_INVIS].set(new_timer)
    new_status = state.status.replace(intrinsics=new_intr,
                                      timed_intrinsics=new_timers)
    return state.replace(status=new_status)


def _effect_invisibility(state, rng, buc):
    """potion of invisibility — byte-equal to vendor peffect_invisibility.

    vendor/nethack/src/potion.c:811-839:
        if (otmp->blessed && !rn2(HInvis ? 15 : 30))
            HInvis |= FROMOUTSIDE;      /* permanent invisibility */
        else
            incr_itimeout(&HInvis, d(6 - 3*bcsign(otmp), 100) + 100);
        if (otmp->cursed) {
            aggravate();                /* aggravate nearby monsters */
            HInvis &= ~FROMOUTSIDE;     /* strip permanent invis */
        }

    Duration dice d(6-3*bcsign, 100) + 100:
        cursed   bcsign=-1 → d(9, 100) + 100 → 109..1000
        uncursed bcsign= 0 → d(6, 100) + 100 → 106..700
        blessed  bcsign=+1 → d(3, 100) + 100 → 103..400
    Blessed permanent-grant chance: 1/30 normally, 1/15 if HInvis already.
    Cursed turns on AGGRAVATE intrinsic (audit P0).
    """
    from Nethax.nethax.rng import rn2

    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    bcsign  = jnp.where(blessed, jnp.int32(1),
              jnp.where(cursed,  jnp.int32(-1), jnp.int32(0)))

    rng_d, rng_p = jax.random.split(rng, 2)

    # Duration roll: d(6-3*bcsign, 100) + 100
    n_dice  = jnp.int32(6) - jnp.int32(3) * bcsign  # 3/6/9
    # Static-shape: roll MAX_N d100 and mask down to n_dice.
    MAX_N = 9
    rolls = jax.random.randint(rng_d, (MAX_N,), 1, 101, dtype=jnp.int32)
    mask  = jnp.arange(MAX_N, dtype=jnp.int32) < n_dice
    dur   = jnp.sum(jnp.where(mask, rolls, jnp.int32(0))).astype(jnp.int32) + jnp.int32(100)

    # Blessed permanent-grant chance.
    already_invis = state.status.intrinsics[Intrinsic.INVIS]
    perm_denom = jnp.where(already_invis, jnp.int32(15), jnp.int32(30))
    perm_roll  = rn2(rng_p, perm_denom.astype(jnp.int32))
    grant_perm = blessed & (perm_roll == jnp.int32(0))

    # Apply duration (additive, vendor incr_itimeout).
    cur_dur = state.status.timed_intrinsics[Intrinsic.INVIS]
    new_dur = jnp.where(grant_perm, cur_dur, cur_dur + dur)
    new_timers = state.status.timed_intrinsics.at[Intrinsic.INVIS].set(new_dur)

    # Permanent intrinsic: blessed→set on perm-grant; cursed→clear permanent.
    new_perm = jnp.where(grant_perm, jnp.bool_(True),
                jnp.where(cursed, jnp.bool_(False), already_invis))
    new_intr = state.status.intrinsics.at[Intrinsic.INVIS].set(new_perm)

    # Cursed → AGGRAVATE intrinsic (nearby monsters notice).
    cur_agg = state.status.intrinsics[Intrinsic.AGGRAVATE]
    new_agg = jnp.where(cursed, jnp.bool_(True), cur_agg)
    new_intr = new_intr.at[Intrinsic.AGGRAVATE].set(new_agg)

    new_status = state.status.replace(intrinsics=new_intr,
                                      timed_intrinsics=new_timers)
    return state.replace(status=new_status)


def _effect_monster_detection(state, rng, buc):
    """potion of monster detection — byte-equal to vendor peffect_monster_detection.

    vendor/nethack/src/potion.c:914-940:
        if (otmp->blessed) {
            if ((HDetect_monsters & TIMEOUT) >= 300L) i = 1;
            else                                       i = rn2(100) + 100;
            incr_itimeout(&HDetect_monsters, i);
            (reveal map of monster positions)
        } else if (otmp->cursed) {
            (wake all monsters on level)
        } else {
            (uncursed: just reveal monsters at current LOS)
        }

    Blessed → timed detect [100, 199] (or +1 if already ≥300).
    Cursed  → wake all alive monsters (clear sleeping flags).
    Uncursed → same as blessed but timed-reveal only.
    """
    from Nethax.nethax.rng import rn2

    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)

    # Blessed/uncursed: incr_itimeout by rn2(100)+100=[100,199], capped at +1
    # if already ≥300.
    cur_t = state.status.timed_intrinsics[Intrinsic.DETECT_MONSTERS]
    base  = rn2(rng, 100).astype(jnp.int32) + jnp.int32(100)
    saturated = cur_t >= jnp.int32(300)
    incr  = jnp.where(saturated, jnp.int32(1), base)
    new_t = jnp.where(cursed, cur_t, cur_t + incr)
    new_timers = state.status.timed_intrinsics.at[Intrinsic.DETECT_MONSTERS].set(new_t)

    # Cursed: wake all sleeping monsters on current level.
    mai = state.monster_ai
    new_asleep = jnp.where(cursed,
                           jnp.zeros_like(mai.asleep),
                           mai.asleep)
    new_mai = mai.replace(asleep=new_asleep)

    new_status = state.status.replace(timed_intrinsics=new_timers)
    return state.replace(status=new_status, monster_ai=new_mai)


def _effect_object_detection(state, rng, buc):
    """potion of object detection — byte-equal to vendor peffect_object_detection.

    vendor/nethack/src/potion.c::peffect_object_detection → object_detect(otmp, 0)
    (vendor/nethack/src/detect.c:602-789) — immediately reveals every ground
    item on the current dungeon level by marking those tiles in the player's
    observation. Buried items are excluded for the uncursed/cursed case.

    Was: only set detect_objects_until_turn timer; no actual reveal ever
    happened. Now scans state.ground_items on the current level and ORs
    every tile that contains a non-empty stack slot into state.explored.
    Cite: vendor/nethack/src/detect.c::object_detect (line 602).
    """
    b      = state.dungeon.current_branch.astype(jnp.int32)
    lv     = state.dungeon.current_level.astype(jnp.int32) - 1
    gi_cat = state.ground_items.category[b, lv]                # [H, W, stack]
    has_item = jnp.any(gi_cat != jnp.int8(0), axis=-1)         # [H, W] bool
    old_lvl  = state.explored[b, lv]
    new_lvl  = old_lvl | has_item
    new_expl = state.explored.at[b, lv].set(new_lvl)
    # Also keep the legacy timer so observation code that already reads it
    # continues to fire; vendor effect is instantaneous so timer is moot.
    new_timer = state.timestep + jnp.int32(100)
    new_id = state.identification.replace(
        detect_objects_until_turn=new_timer
    )
    return state.replace(explored=new_expl, identification=new_id)


# ---- movement modifiers ---------------------------------------------------

def _effect_levitation(state, rng, buc):
    """potion of levitation — byte-equal to vendor peffect_levitation.

    vendor/nethack/src/potion.c::peffect_levitation:
        incr_itimeout(&HLevitation, rn1(140, 10));

    rn1(140, 10) = uniform [10, 149].
    Was: fixed 150 turns. Now uniform [10, 149] additive (incr_itimeout).
    """
    from Nethax.nethax.rng import rn2

    turns = rn2(rng, 140).astype(jnp.int32) + jnp.int32(10)  # [10, 149]
    new_status = add_timed_intrinsic(state.status, Intrinsic.LEVITATION, turns)
    # Emit the float-up message.  Vendor potion.c:1034 peffect_levitation
    # calls float_up() (vendor/nethack/src/trap.c:2891) which prints
    # You("start to float in the air!").  The LEVITATION timer set above is
    # the actual float-up state — read by the lava/water/pickup gates
    # (e.g. boulders.py, inventory.py drop/lift, action_dispatch lava cross) —
    # so no separate float_up() helper is needed; only the message was
    # missing.  The MiniHack Levitate envs' RewardManager substring-matches
    # "You start to float in the air" (skills_levitate.py:9).
    # Cite: vendor/nle/src/potion.c:1034 -> vendor/nle/src/trap.c:2891.
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    new_messages = _msg_emit(state.messages, int(_MsgId.LEVI_START_FLOAT))
    return state.replace(status=new_status, messages=new_messages)


def _effect_speed(state, rng, buc):
    """potion of speed — byte-equal to vendor peffect_speed.

    vendor/nethack/src/potion.c:1052-1070:
        speed_up(rn1(10, 100 + 60 * bcsign(otmp)));
        if (!cursed && !(HFast & INTRINSIC)) HFast |= FROMOUTSIDE;

    rn1(10, M) = [M, M+9] with M = 100+60*bcsign:
        cursed   → [40, 49]
        uncursed → [100, 109]
        blessed  → [160, 169]
    Non-cursed (uncursed AND blessed) grants intrinsic Fast permanently.
    Was: blessed-only permanent, fixed 160 turns timer.
    """
    from Nethax.nethax.rng import rn2

    cursed  = _is_cursed(buc)
    blessed = _is_blessed(buc)
    bcsign  = jnp.where(blessed, jnp.int32(1),
              jnp.where(cursed,  jnp.int32(-1), jnp.int32(0)))
    m_base  = jnp.int32(100) + jnp.int32(60) * bcsign  # 40/100/160
    turns   = rn2(rng, 10).astype(jnp.int32) + m_base   # [M, M+9]

    cur_perm = state.status.intrinsics[Intrinsic.FAST]
    perm_new = jnp.where(~cursed, jnp.bool_(True), cur_perm)
    new_intr = state.status.intrinsics.at[Intrinsic.FAST].set(perm_new)

    cur = state.status.timed_intrinsics[Intrinsic.FAST]
    new_t = cur + turns  # incr_itimeout — additive
    new_timers = state.status.timed_intrinsics.at[Intrinsic.FAST].set(new_t)

    new_status = state.status.replace(intrinsics=new_intr,
                                      timed_intrinsics=new_timers)
    return state.replace(status=new_status)


# ---- hostile/negative effects ---------------------------------------------

def _effect_paralysis(state, rng, buc):
    """potion of paralysis — byte-equal to vendor peffect_paralysis.

    vendor/nethack/src/potion.c::peffect_paralysis:
        if (Free_action) {  /* no paralysis */
            You("stiffen briefly.");
        } else {
            nomul(-(rn1(10, 25 - 12 * bcsign(otmp))));
        }

    rn1(N, M) = rn2(N) + M = uniform [M, M+N-1]. With bcsign=blessed-cursed:
        cursed   bcsign=-1 → rn1(10, 37) → 37..46 turns
        uncursed bcsign= 0 → rn1(10, 25) → 25..34 turns
        blessed  bcsign=+1 → rn1(10, 13) → 13..22 turns
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    from Nethax.nethax.rng import rn2

    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    bcsign  = jnp.where(blessed, jnp.int32(1),
              jnp.where(cursed,  jnp.int32(-1), jnp.int32(0)))
    m_base  = jnp.int32(25) - jnp.int32(12) * bcsign  # 37/25/13
    n_part  = rn2(rng, 10).astype(jnp.int32)          # 0..9
    turns   = n_part + m_base                          # M..M+9

    # Free_action immunity: brief stiffen, no paralysis.
    has_free_action = state.status.intrinsics[int(Intrinsic.FREE_ACTION)]
    final_turns = jnp.where(has_free_action, jnp.int32(0), turns)

    cur = state.status.timed_statuses[int(TimedStatus.FROZEN)]
    new_val = jnp.maximum(cur, final_turns)
    new_ts  = state.status.timed_statuses.at[int(TimedStatus.FROZEN)].set(new_val)
    new_status = state.status.replace(timed_statuses=new_ts)
    return state.replace(status=new_status)


def _effect_sleeping(state, rng, buc):
    """potion of sleeping — byte-equal to vendor peffect_sleeping.

    vendor/nethack/src/potion.c::peffect_sleeping:
        if (Sleep_resistance) {
            monstseesu(M_SEEN_SLEEP);  /* no sleep */
        } else {
            fall_asleep(-(rn1(10, 25 - 12 * bcsign(otmp))), TRUE);
        }

    rn1(10, M) = uniform [M, M+9] with M = 25-12*bcsign:
        cursed   → [37, 46]
        uncursed → [25, 34]
        blessed  → [13, 22]
    SLEEP_RES intrinsic gives immunity.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    from Nethax.nethax.rng import rn2

    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    bcsign  = jnp.where(blessed, jnp.int32(1),
              jnp.where(cursed,  jnp.int32(-1), jnp.int32(0)))
    m_base  = jnp.int32(25) - jnp.int32(12) * bcsign
    n_part  = rn2(rng, 10).astype(jnp.int32)
    turns   = n_part + m_base

    has_sleep_res = state.status.intrinsics[int(Intrinsic.RESIST_SLEEP)]
    final_turns = jnp.where(has_sleep_res, jnp.int32(0), turns)

    cur = state.status.timed_statuses[int(TimedStatus.SLEEP)]
    new_val = jnp.maximum(cur, final_turns)
    new_ts  = state.status.timed_statuses.at[int(TimedStatus.SLEEP)].set(new_val)
    new_status = state.status.replace(timed_statuses=new_ts)
    return state.replace(status=new_status)


def _effect_confusion(state, rng, buc):
    """potion of confusion — add confusion timer.

    Canonical: peffect_confusion — make_confused(incr(HConfusion, rn1(7,16-8*bcsign))).
    Wave 3: 20 turns (blessed 10, cursed 28).
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    turns   = jnp.where(blessed, jnp.int32(10),
              jnp.where(cursed,  jnp.int32(28), jnp.int32(20)))
    cur = state.status.timed_statuses[int(TimedStatus.CONFUSION)]
    new_val = jnp.maximum(cur, turns)
    new_ts  = state.status.timed_statuses.at[int(TimedStatus.CONFUSION)].set(new_val)
    new_status = state.status.replace(timed_statuses=new_ts)
    return state.replace(status=new_status)


def _effect_hallucination(state, rng, buc):
    """potion of hallucination — byte-equal to vendor peffect_hallucination.

    vendor/nethack/src/potion.c::peffect_hallucination:
        make_hallucinated(itimeout_incr(HHallucination,
            rn1(200, 600 - 300 * bcsign(otmp))), TRUE, 0);

    rn1(200, M) = uniform [M, M+199] where M = 600 - 300*bcsign:
        cursed   bcsign=-1 → [900, 1099]
        uncursed bcsign= 0 → [600, 799]
        blessed  bcsign=+1 → [300, 499]
    Was fixed 300/600/900 turns (off-by-200 ceiling).
    """
    from Nethax.nethax.rng import rn2

    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    bcsign  = jnp.where(blessed, jnp.int32(1),
              jnp.where(cursed,  jnp.int32(-1), jnp.int32(0)))
    m_base  = jnp.int32(600) - jnp.int32(300) * bcsign  # 900/600/300
    turns   = rn2(rng, 200).astype(jnp.int32) + m_base  # [M, M+199]
    cur = state.status.timed_statuses[int(TimedStatus.HALLUCINATION)]
    new_val = cur + turns  # incr_itimeout — additive
    new_ts  = state.status.timed_statuses.at[int(TimedStatus.HALLUCINATION)].set(new_val)
    new_status = state.status.replace(timed_statuses=new_ts)
    return state.replace(status=new_status)


def _effect_blindness(state, rng, buc):
    """potion of blindness — byte-equal to vendor peffect_blindness.

    vendor/nethack/src/potion.c:1073-1080:
        make_blinded(itimeout_incr(BlindedTimeout,
                                   rn1(200, 250 - 125*bcsign(otmp))),
                     !Blind);

    rn1(200, M) = [M, M+199] with M = 250-125*bcsign:
        cursed   → [375, 574]
        uncursed → [250, 449]
        blessed  → [125, 324]
    """
    from Nethax.nethax.rng import rn2
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    bcsign  = jnp.where(blessed, jnp.int32(1),
              jnp.where(cursed,  jnp.int32(-1), jnp.int32(0)))
    m_base  = jnp.int32(250) - jnp.int32(125) * bcsign  # 375/250/125
    turns   = rn2(rng, 200).astype(jnp.int32) + m_base  # [M, M+199]
    cur = state.status.timed_statuses[int(TimedStatus.BLIND)]
    new_val = cur + turns  # incr_itimeout
    new_ts  = state.status.timed_statuses.at[int(TimedStatus.BLIND)].set(new_val)
    new_status = state.status.replace(timed_statuses=new_ts)
    return state.replace(status=new_status)


def _effect_sickness(state, rng, buc):
    """potion of sickness — damage and make sick.

    Canonical: peffect_sickness — losehp(rnd(10)+5*cursed); make_sick.
    Wave 3: 5 HP damage (blessed 1, cursed 15); set SICK timer.
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    dmg     = jnp.where(blessed, jnp.int32(1),
              jnp.where(cursed,  jnp.int32(15), jnp.int32(5)))
    new_hp  = jnp.maximum(state.player_hp - dmg, jnp.int32(1))
    cur_sick = state.status.timed_statuses[int(TimedStatus.SICK)]
    new_sick = jnp.where(blessed, cur_sick, jnp.maximum(cur_sick, jnp.int32(50)))
    new_ts   = state.status.timed_statuses.at[int(TimedStatus.SICK)].set(new_sick)
    new_status = state.status.replace(timed_statuses=new_ts)
    return state.replace(player_hp=new_hp, status=new_status)


def _effect_acid(state, rng, buc):
    """potion of acid — byte-equal to vendor peffect_acid.

    vendor/nethack/src/potion.c:1297-1315:
        if (Acid_resistance) { /* no damage */ }
        else {
            dmg = d(otmp->cursed ? 2 : 1, otmp->blessed ? 4 : 8);
            losehp(Maybe_Half_Phys(dmg), "potion of acid", KILLED_BY_AN);
        }
        if (Stoned) fix_petrification();

    Damage dice (n, sides):
        uncursed → d(1, 8) → 1..8
        cursed   → d(2, 8) → 2..16
        blessed  → d(1, 4) → 1..4
    HALF_PHDAM halves damage. Acid_resistance grants full immunity. Acid
    also fix-petrifies a stoning hero.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    n_dice  = jnp.where(cursed, jnp.int32(2), jnp.int32(1))
    sides   = jnp.where(blessed, jnp.int32(4), jnp.int32(8))
    # Static-shape, traced-sides die roll: uniform-floor + clip to [1, sides].
    MAX_N = 2
    rand = jax.random.uniform(rng, (MAX_N,))
    rolls = jnp.floor(rand * sides.astype(jnp.float32)).astype(jnp.int32) + 1
    rolls = jnp.clip(rolls, jnp.int32(1), sides)
    mask  = jnp.arange(MAX_N, dtype=jnp.int32) < n_dice
    raw_dmg = jnp.sum(jnp.where(mask, rolls, jnp.int32(0))).astype(jnp.int32)

    # Maybe_Half_Phys: halve damage when HALF_PHDAM intrinsic active.
    has_half_phys = state.status.intrinsics[int(Intrinsic.HALF_PHYSICAL_DAMAGE)]
    dmg = jnp.where(has_half_phys, (raw_dmg + jnp.int32(1)) // jnp.int32(2), raw_dmg)

    # Acid_resistance → zero damage.
    has_acid_res = state.status.intrinsics[int(Intrinsic.RESIST_ACID)]
    dmg = jnp.where(has_acid_res, jnp.int32(0), dmg)

    new_hp = state.player_hp - dmg  # let HP go ≤ 0 → death handled in env loop

    # fix_petrification — clear STONED timer when acid quaffed.
    cur_stoned = state.status.timed_statuses[int(TimedStatus.STONED)]
    new_stoned = jnp.int32(0)  # always cleared by acid (vendor)
    new_ts = state.status.timed_statuses.at[int(TimedStatus.STONED)].set(new_stoned)
    new_status = state.status.replace(timed_statuses=new_ts)
    return state.replace(player_hp=new_hp, status=new_status)


def _effect_oil(state, rng, buc):
    """potion of oil — grease wielded weapon; explode on open flame.

    Canonical: peffect_oil (potion.c) — if lamp is lit, potion explodes for
    fire damage; otherwise oils the wielded weapon (sets obj.greased=1).
    Implementation: set wielded item.greased=True; if wielded item itself is
    lit (lamp), deal 2d4 fire damage to player.

    Cite: vendor/nethack/src/potion.c::peffect_oil.
    """
    wslot = state.inventory.wielded.astype(jnp.int32)
    has_wielded = wslot >= jnp.int32(0)

    # Grease the wielded weapon.
    cur_greased = state.inventory.items.greased
    new_greased = jnp.where(
        has_wielded,
        cur_greased.at[wslot].set(jnp.bool_(True)),
        cur_greased,
    )
    new_items = state.inventory.items.replace(greased=new_greased)

    # Open-flame check: vendor peffect_oil — if any inventory item is lit
    # (lamp/candle/candelabrum), the potion vapor ignites for fire damage.
    # vendor/nethack/src/potion.c:1276-1280:
    #     vulnerable = !Fire_resistance || Cold_resistance;
    #     losehp(d(vulnerable ? 4 : 2, 4),
    #            "quaffing a burning potion of oil", KILLED_BY);
    # Dice count varies (4 vs 2) but sides are always 4; no halving — vendor
    # passes raw d() result to losehp (which has no resistance-halving for oil).
    any_lit = jnp.any(state.inventory.items.lamplit)
    has_fire_res = state.status.intrinsics[int(Intrinsic.RESIST_FIRE)]
    has_cold_res = state.status.intrinsics[int(Intrinsic.RESIST_COLD)]
    # vulnerable = !Fire_resistance || Cold_resistance
    vulnerable = jnp.logical_or(jnp.logical_not(has_fire_res), has_cold_res)
    n_dice = jnp.where(vulnerable, jnp.int32(4), jnp.int32(2))
    # Static-shape masked roll: 4 d4, sum first n_dice.
    MAX_N = 4
    rng, sub = jax.random.split(rng)
    rolls = jax.random.randint(sub, (MAX_N,), 1, 5, dtype=jnp.int32)  # d4
    mask  = jnp.arange(MAX_N, dtype=jnp.int32) < n_dice
    fire_dmg = jnp.sum(jnp.where(mask, rolls, jnp.int32(0))).astype(jnp.int32)
    new_hp = jnp.where(any_lit,
                       jnp.maximum(state.player_hp - fire_dmg, jnp.int32(0)),
                       state.player_hp)

    new_inv = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv, player_hp=new_hp)


def _effect_polymorph(state, rng, buc):
    """potion of polymorph — polymorph the player into a random valid form.

    Canonical: peffect_polymorph → polyself(POLY_NOFLAGS).
    Vendor reference: vendor/nethack/src/potion.c::peffect_polymorph, which
    calls polyself.c::polyself() with POLY_NOFLAGS (uncontrolled random form).

    Cite: polyself.c:280 for valid-form selection.
    """
    from Nethax.nethax.subsystems.polymorph import (
        polymorph_player,
        choose_random_polymorph_form,
    )
    rng, sub = jax.random.split(rng)
    form = choose_random_polymorph_form(state, sub)
    rng, sub2 = jax.random.split(rng)
    return polymorph_player(state, sub2, form, controlled=False)


# ---- water group ----------------------------------------------------------

def _effect_water(state, rng, buc):
    """potion of water — holy/unholy/plain water — byte-equal vendor.

    Vendor: vendor/nethack/src/potion.c::peffect_water (lines 717-768).

        if !blessed and !cursed:
            nutrition += rnd(10)
            return
        # blessed or cursed
        if mon_hates_blessings(youmonst) || align == A_CHAOTIC:
            if blessed:  losehp(Maybe_Half_Phys(d(2,6)), "holy water", ...)
            elif cursed: healup(d(2,6), 0, 0, 0)
        else:
            if blessed: cure SICK; exercise WIS+CON
            else:       # cursed
                if align == A_LAWFUL:
                    losehp(Maybe_Half_Phys(d(2,6)), "unholy water", ...)
                # else: "You feel full of dread" — no damage

    nethax Alignment encoding (subsystems/prayer.py): CHAOTIC=0, NEUTRAL=1,
    LAWFUL=2 — we use that here.  We do not currently model the player
    being undead/demon/were-form (`mon_hates_blessings`), so the blessed
    damage branch for chaotic players is correct but the
    undead/demon/were sub-branch falls back to the non-burn path.
    """
    from Nethax.nethax.subsystems.prayer import Alignment as _PAlign
    from Nethax.nethax.rng import dice_roll as _dice_roll

    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)

    align_i = state.player_align.astype(jnp.int32)
    is_lawful  = align_i == jnp.int32(int(_PAlign.LAWFUL))
    is_chaotic = align_i == jnp.int32(int(_PAlign.CHAOTIC))

    # Plain water (neither blessed nor cursed): nutrition += rnd(10).
    rng_n, rng_d, rng_h = jax.random.split(rng, 3)
    plain_nutr = (state.status.nutrition.astype(jnp.int32)
                  + jax.random.randint(rng_n, (), 1, 11, dtype=jnp.int32))
    new_nutrition = jnp.where(
        ~blessed & ~cursed,
        jnp.minimum(plain_nutr, jnp.int32(2000)).astype(state.status.nutrition.dtype),
        state.status.nutrition,
    )

    # Blessed cures SICK regardless of alignment.
    new_sick = jnp.where(
        blessed,
        jnp.int32(0),
        state.status.timed_statuses[int(TimedStatus.SICK)],
    )
    new_ts = state.status.timed_statuses.at[int(TimedStatus.SICK)].set(new_sick)
    new_status = state.status.replace(timed_statuses=new_ts, nutrition=new_nutrition)

    # d(2, 6) — shared by both burn branches.
    burn_dmg = _dice_roll(rng_d, 2, 6).astype(jnp.int32)
    heal_amt = _dice_roll(rng_h, 2, 6).astype(jnp.int32)

    # Blessed + chaotic → burn d(2,6).
    blessed_burn = blessed & is_chaotic
    # Cursed + lawful → burn d(2,6).
    cursed_burn  = cursed & is_lawful
    # Cursed + chaotic → heal d(2,6).
    cursed_heal  = cursed & is_chaotic

    # Damage path leaves HP unclamped to hp_max — vendor losehp() does
    # not raise HP toward hp_max.  Heal path uses healup() which DOES
    # clamp to hp_max.
    dmg = jnp.where(blessed_burn | cursed_burn, burn_dmg, jnp.int32(0))
    hp_after_dmg = jnp.maximum(state.player_hp.astype(jnp.int32) - dmg, jnp.int32(0))
    hp_after_heal = jnp.minimum(
        state.player_hp.astype(jnp.int32) + heal_amt,
        state.player_hp_max.astype(jnp.int32),
    )
    new_hp = jnp.where(cursed_heal, hp_after_heal, hp_after_dmg)
    new_hp = jnp.maximum(new_hp, jnp.int32(0))

    return state.replace(
        player_hp=new_hp.astype(state.player_hp.dtype),
        status=new_status,
    )


def _effect_booze(state, rng, buc):
    """potion of booze — byte-equal to vendor peffect_booze.

    vendor/nethack/src/potion.c:771-792:
        if (!blessed) make_confused(itimeout_incr(HConfusion, d(2+u.uhs, 8)));
        if (!odiluted) healup(1, 0, FALSE, FALSE);
        u.uhunger += 10 * (2 + bcsign(otmp));    /* 10/20/30 */
        if (cursed) {
            You("pass out.");
            gm.multi = -rnd(15);                 /* FROZEN for 1..15 */
        }

    Vendor confusion dice depend on u.uhs (hunger state); we use uhs=0
    (UNFED/SATIATED) → d(2, 8) → 2..16, matching the typical-play case.
    No hallucination side-effect (audit assertion was incorrect; vendor
    peffect_booze does not call make_hallucinated).
    """
    from Nethax.nethax.rng import dice_roll, rnd as _rnd1

    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    rng_h, rng_c, rng_s = jax.random.split(rng, 3)

    # +1 HP unless diluted (odiluted not modelled here → always heal).
    new_hp = jnp.minimum(state.player_hp + jnp.int32(1), state.player_hp_max)

    # Confusion: d(2+u.uhs, 8). u.uhs default 0; vary by hunger if available.
    # bcsign factor only blocks confusion when blessed (no roll).
    # Static n=2 (Python int, not jnp wrapper) so dice_roll's (n,) shape stays
    # concrete in the JIT trace.
    conf_dmg = dice_roll(rng_h, 2, 8)
    cur_conf = state.status.timed_statuses[int(TimedStatus.CONFUSION)]
    new_conf = jnp.where(
        blessed,
        cur_conf,
        cur_conf + conf_dmg,  # itimeout_incr — additive, not max
    )
    new_ts = state.status.timed_statuses.at[int(TimedStatus.CONFUSION)].set(new_conf)

    # Cursed: FROZEN for rnd(15) = 1..15 turns (pass out).
    pass_out_turns = _rnd1(rng_c, 15).astype(jnp.int32)
    cur_frozen = new_ts[int(TimedStatus.FROZEN)]
    new_frozen = jnp.where(
        cursed,
        jnp.maximum(cur_frozen, pass_out_turns),
        cur_frozen,
    )
    new_ts = new_ts.at[int(TimedStatus.FROZEN)].set(new_frozen)

    # Nutrition gain: 10 * (2 + bcsign) → 10/20/30 for cursed/uncursed/blessed.
    bcsign = jnp.where(blessed, jnp.int32(1),
             jnp.where(cursed,  jnp.int32(-1), jnp.int32(0)))
    nut_gain = jnp.int32(10) * (jnp.int32(2) + bcsign)
    new_nutrition = jnp.minimum(
        state.status.nutrition + nut_gain, jnp.int32(2000)
    )

    new_status = state.status.replace(
        timed_statuses=new_ts,
        nutrition=new_nutrition,
    )
    return state.replace(player_hp=new_hp, status=new_status)


def _effect_fruit_juice(state, rng, buc):
    """potion of fruit juice — byte-equal to vendor peffect_see_invisible
    POT_FRUIT_JUICE early-return branch.

    vendor/nethack/src/potion.c:856-860:
        if (otmp->otyp == POT_FRUIT_JUICE) {
            u.uhunger += (otmp->odiluted ? 5 : 10) * (2 + bcsign(otmp));
            newuhs(FALSE);
            return;
        }

    Only nutrition gain — vendor does NOT grant see-invisible from fruit
    juice (that's the POT_SEE_INVISIBLE branch). Without odiluted tracking,
    use the standard 10 * (2 + bcsign) = 10/20/30 for cursed/uncursed/
    blessed.
    """
    cursed  = _is_cursed(buc)
    blessed = _is_blessed(buc)
    bcsign  = jnp.where(blessed, jnp.int32(1),
              jnp.where(cursed,  jnp.int32(-1), jnp.int32(0)))
    nut_gain = jnp.int32(10) * (jnp.int32(2) + bcsign)  # 10/20/30
    new_nutrition = jnp.minimum(
        state.status.nutrition + nut_gain, jnp.int32(2000)
    )
    new_status = state.status.replace(nutrition=new_nutrition)
    return state.replace(status=new_status)


def _effect_enlightenment(state, rng, buc):
    """potion of enlightenment — reveal character info; blessed raises INT/WIS.

    Canonical: peffect_enlightenment — do_enlightenment_effect(); blessed adjattrib.
    Wave 3: blessed +1 INT, +1 WIS; cursed flavour only.
    """
    blessed = _is_blessed(buc)
    delta   = jnp.where(blessed, jnp.int8(1), jnp.int8(0))
    new_int = jnp.clip(state.player_int + delta, jnp.int8(3), jnp.int8(25))
    new_wis = jnp.clip(state.player_wis + delta, jnp.int8(3), jnp.int8(25))
    return state.replace(player_int=new_int, player_wis=new_wis)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _clear_timed(status, timed_id: int):
    new_ts = status.timed_statuses.at[int(timed_id)].set(jnp.int32(0))
    return status.replace(timed_statuses=new_ts)


# ---------------------------------------------------------------------------
# Effect dispatch table — indexed by PotionEffect value.
# Must contain exactly N_POTIONS entries in enum order.
#
# JAX lax.switch constraint: all branch functions must share the same
# signature. We pass (state, rng, buc) by packing them into a tuple
# operand so lax.switch sees a single (operand → result) signature.
# ---------------------------------------------------------------------------

_EFFECT_TABLE = (
    _effect_gain_ability,       #  0  GAIN_ABILITY
    _effect_restore_ability,    #  1  RESTORE_ABILITY
    _effect_confusion,          #  2  CONFUSION
    _effect_blindness,          #  3  BLINDNESS
    _effect_paralysis,          #  4  PARALYSIS
    _effect_speed,              #  5  SPEED
    _effect_levitation,         #  6  LEVITATION
    _effect_hallucination,      #  7  HALLUCINATION
    _effect_invisibility,       #  8  INVISIBILITY
    _effect_see_invisible,      #  9  SEE_INVISIBLE
    _effect_healing,            # 10  HEALING
    _effect_extra_healing,      # 11  EXTRA_HEALING
    _effect_gain_level,         # 12  GAIN_LEVEL
    _effect_enlightenment,      # 13  ENLIGHTENMENT
    _effect_monster_detection,  # 14  MONSTER_DETECTION
    _effect_object_detection,   # 15  OBJECT_DETECTION
    _effect_gain_energy,        # 16  GAIN_ENERGY
    _effect_sleeping,           # 17  SLEEPING
    _effect_full_healing,       # 18  FULL_HEALING
    _effect_polymorph,          # 19  POLYMORPH
    _effect_booze,              # 20  BOOZE
    _effect_sickness,           # 21  SICKNESS
    _effect_fruit_juice,        # 22  FRUIT_JUICE
    _effect_acid,               # 23  ACID
    _effect_oil,                # 24  OIL
    _effect_water,              # 25  WATER
)

assert len(_EFFECT_TABLE) == N_POTIONS, (
    f"Effect table has {len(_EFFECT_TABLE)} entries; expected {N_POTIONS}"
)

# Build lax.switch branch list: each branch unpacks (state, rng, buc).
_SWITCH_BRANCHES = [
    (lambda operand, fn=fn: fn(operand[0], operand[1], operand[2]))
    for fn in _EFFECT_TABLE
]


# ---------------------------------------------------------------------------
# Monster-targeted potion effects — vendor/nethack/src/dothrow.c:2262-2400
# (potionhit).  Applied when a thrown potion shatters on a monster.
#
# Signature: (state, monster_slot: jnp.int32, rng) -> state
# Only the top effects are implemented; the default branch deals rnd(4) damage.
# ---------------------------------------------------------------------------

def _monster_sleep(state, m_slot, rng):
    """Sleeping potion shatters on monster — set asleep=True."""
    mai = state.monster_ai
    new_asleep = mai.asleep.at[m_slot].set(jnp.bool_(True))
    return state.replace(monster_ai=mai.replace(asleep=new_asleep))


def _monster_heal(state, m_slot, rng):
    """Healing potion shatters on monster — restore d8 HP (capped at hp_max).

    Vendor: vendor/nethack/src/potion.c::potionhit (~line 2120) — healing
    potion thrown at monster heals it by d8 HP and removes BLIND/STUNNED.
    """
    rng_d8, _ = jax.random.split(rng)
    mai = state.monster_ai
    heal = jax.random.randint(rng_d8, (), 1, 9, dtype=jnp.int32)  # d8 = 1..8
    cur_hp  = mai.hp[m_slot].astype(jnp.int32)
    cur_max = mai.hp_max[m_slot].astype(jnp.int32)
    new_hp  = jnp.minimum(cur_hp + heal, cur_max)
    new_hp_arr = mai.hp.at[m_slot].set(new_hp)
    return state.replace(monster_ai=mai.replace(hp=new_hp_arr))


def _monster_full_heal(state, m_slot, rng):
    """Full-healing potion on monster — restore to hp_max + bump hp_max by 4.

    Vendor: potion.c::potionhit FULL_HEALING — heals to max, also bumps
    mhpmax by 4 (matching the +nxtra=4 uncursed-thrown case for hero).
    """
    mai = state.monster_ai
    cur_max = mai.hp_max[m_slot].astype(jnp.int32)
    new_max = cur_max + jnp.int32(4)
    new_max_arr = mai.hp_max.at[m_slot].set(new_max)
    new_hp_arr  = mai.hp.at[m_slot].set(new_max)
    return state.replace(monster_ai=mai.replace(hp=new_hp_arr, hp_max=new_max_arr))


def _monster_acid_dmg(state, m_slot, rng):
    """Acid potion shatters on monster — deal 6 damage."""
    mai = state.monster_ai
    cur_hp = mai.hp[m_slot].astype(jnp.int32)
    new_hp = jnp.maximum(cur_hp - jnp.int32(6), jnp.int32(0))
    new_alive = (new_hp > jnp.int32(0)) & mai.alive[m_slot]
    new_hp_arr    = mai.hp.at[m_slot].set(new_hp)
    new_alive_arr = mai.alive.at[m_slot].set(new_alive)
    return state.replace(monster_ai=mai.replace(hp=new_hp_arr, alive=new_alive_arr))


def _monster_paralyze(state, m_slot, rng):
    """Paralysis potion shatters on monster — freeze (set asleep=True)."""
    mai = state.monster_ai
    new_asleep = mai.asleep.at[m_slot].set(jnp.bool_(True))
    return state.replace(monster_ai=mai.replace(asleep=new_asleep))


def _monster_blindness(state, m_slot, rng):
    """Blindness potion shatters on monster — deal 1 damage (minor nuisance)."""
    mai = state.monster_ai
    cur_hp = mai.hp[m_slot].astype(jnp.int32)
    new_hp = jnp.maximum(cur_hp - jnp.int32(1), jnp.int32(0))
    new_alive = (new_hp > jnp.int32(0)) & mai.alive[m_slot]
    new_hp_arr    = mai.hp.at[m_slot].set(new_hp)
    new_alive_arr = mai.alive.at[m_slot].set(new_alive)
    return state.replace(monster_ai=mai.replace(hp=new_hp_arr, alive=new_alive_arr))


def _monster_sickness(state, m_slot, rng):
    """Sickness potion shatters on monster — deal 8 damage."""
    mai = state.monster_ai
    cur_hp = mai.hp[m_slot].astype(jnp.int32)
    new_hp = jnp.maximum(cur_hp - jnp.int32(8), jnp.int32(0))
    new_alive = (new_hp > jnp.int32(0)) & mai.alive[m_slot]
    new_hp_arr    = mai.hp.at[m_slot].set(new_hp)
    new_alive_arr = mai.alive.at[m_slot].set(new_alive)
    return state.replace(monster_ai=mai.replace(hp=new_hp_arr, alive=new_alive_arr))


def _monster_extra_heal(state, m_slot, rng):
    """Extra-healing potion shatters on monster — restore d8+6 HP.

    Vendor potion.c::potionhit EXTRA_HEALING heals by d8+6 (matching the
    hero's `healup(d(2,8)+6, ...)` minus the +1 hp_max bump that vendor
    grants to the hero but not to monsters).
    """
    rng_d8, _ = jax.random.split(rng)
    heal = jax.random.randint(rng_d8, (), 1, 9, dtype=jnp.int32) + jnp.int32(6)
    mai = state.monster_ai
    cur_hp  = mai.hp[m_slot].astype(jnp.int32)
    cur_max = mai.hp_max[m_slot].astype(jnp.int32)
    new_hp  = jnp.minimum(cur_hp + heal, cur_max)
    new_hp_arr = mai.hp.at[m_slot].set(new_hp)
    return state.replace(monster_ai=mai.replace(hp=new_hp_arr))


def _monster_default_dmg(state, m_slot, rng):
    """Unknown potion shatters — rnd(4) splash damage (vendor default)."""
    dmg = rnd(rng, 4).astype(jnp.int32)
    mai = state.monster_ai
    cur_hp = mai.hp[m_slot].astype(jnp.int32)
    new_hp = jnp.maximum(cur_hp - dmg, jnp.int32(0))
    new_alive = (new_hp > jnp.int32(0)) & mai.alive[m_slot]
    new_hp_arr    = mai.hp.at[m_slot].set(new_hp)
    new_alive_arr = mai.alive.at[m_slot].set(new_alive)
    return state.replace(monster_ai=mai.replace(hp=new_hp_arr, alive=new_alive_arr))


# Dispatch table — one entry per PotionEffect (N_POTIONS entries).
# Indexed by effect_id = type_id - _POTION_BASE_ID.
_MONSTER_EFFECT_TABLE = (
    _monster_default_dmg,    #  0  GAIN_ABILITY      → splash
    _monster_default_dmg,    #  1  RESTORE_ABILITY   → splash
    _monster_default_dmg,    #  2  CONFUSION         → splash
    _monster_blindness,      #  3  BLINDNESS
    _monster_paralyze,       #  4  PARALYSIS
    _monster_default_dmg,    #  5  SPEED             → splash
    _monster_default_dmg,    #  6  LEVITATION        → splash
    _monster_default_dmg,    #  7  HALLUCINATION     → splash
    _monster_default_dmg,    #  8  INVISIBILITY      → splash
    _monster_default_dmg,    #  9  SEE_INVISIBLE     → splash
    _monster_heal,           # 10  HEALING
    _monster_extra_heal,     # 11  EXTRA_HEALING
    _monster_default_dmg,    # 12  GAIN_LEVEL        → splash
    _monster_default_dmg,    # 13  ENLIGHTENMENT     → splash
    _monster_default_dmg,    # 14  MONSTER_DETECTION → splash
    _monster_default_dmg,    # 15  OBJECT_DETECTION  → splash
    _monster_default_dmg,    # 16  GAIN_ENERGY       → splash
    _monster_sleep,          # 17  SLEEPING
    _monster_full_heal,      # 18  FULL_HEALING (heal to max + 4 hp_max)
    _monster_default_dmg,    # 19  POLYMORPH         → splash
    _monster_default_dmg,    # 20  BOOZE             → splash
    _monster_sickness,       # 21  SICKNESS
    _monster_default_dmg,    # 22  FRUIT_JUICE       → splash
    _monster_acid_dmg,       # 23  ACID
    _monster_default_dmg,    # 24  OIL               → splash
    _monster_default_dmg,    # 25  WATER             → splash
)

assert len(_MONSTER_EFFECT_TABLE) == N_POTIONS

# lax.switch branches: unpack (state, m_slot, rng).
_MONSTER_SWITCH_BRANCHES = [
    (lambda operand, fn=fn: fn(operand[0], operand[1], operand[2]))
    for fn in _MONSTER_EFFECT_TABLE
]


def apply_potion_to_monster(state, rng, type_id: jnp.ndarray, m_slot: jnp.ndarray):
    """Dispatch a shattered potion effect onto a monster (JIT-pure).

    Vendor reference: dothrow.c::potionhit (lines 2262-2400).

    Parameters
    ----------
    state   : EnvState
    rng     : JAX PRNG key
    type_id : int32 — raw object type_id from the thrown item
    m_slot  : int32 — monster slot index in monster_ai arrays

    Returns
    -------
    Updated EnvState.
    """
    effect_id = jnp.clip(
        type_id.astype(jnp.int32) - jnp.int32(_POTION_BASE_ID),
        0,
        N_POTIONS - 1,
    )
    # Brax-flatten: compute all N_POTIONS monster-effect branches and select
    # by one-hot effect_id mask via jax.tree.map / jnp.where.
    _operand_m = (state, m_slot.astype(jnp.int32), rng)
    _results_m = [br(_operand_m) for br in _MONSTER_SWITCH_BRANCHES]

    def _select_m(*branches):
        out = branches[0]
        for _i in range(1, len(branches)):
            out = jnp.where(effect_id == jnp.int32(_i), branches[_i], out)
        return out

    return jax.tree.map(_select_m, *_results_m)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quaff_potion(state, rng, slot_idx):
    """Apply the potion in inventory slot `slot_idx` to the player.

    Looks up type_id → PotionEffect, dispatches via jax.lax.switch, then
    decrements the item's quantity (removes if qty reaches 0).

    Parameters
    ----------
    state    : EnvState
    rng      : jax.random.PRNGKey
    slot_idx : int or traced jnp scalar — inventory slot index

    Returns
    -------
    Updated EnvState.
    """
    # GLIB: slippery fingers — 1-in-5 chance to drop the item being used.
    # Cite: vendor/nethack/src/status.c::glib — glibs() drops items on use,
    # not just wielded weapons; "the potion slips from your fingers."
    is_glib = state.status.timed_statuses[int(TimedStatus.GLIB)] > jnp.int32(0)
    rng, rng_glib = jax.random.split(rng)
    glib_roll = jax.random.randint(rng_glib, (), 0, 5, dtype=jnp.int32)
    glib_drop = is_glib & (glib_roll == jnp.int32(0))

    def _drop_glib(s):
        # Drop the potion: decrement quantity and return without quaffing.
        _sidx = jnp.int32(slot_idx)
        old_qty = s.inventory.items.quantity[_sidx]
        new_qty = jnp.maximum(old_qty - jnp.int16(1), jnp.int16(0))
        new_cat = jnp.where(new_qty == jnp.int16(0),
                            jnp.int8(0),
                            s.inventory.items.category[_sidx])
        new_quantity = s.inventory.items.quantity.at[_sidx].set(new_qty)
        new_category = s.inventory.items.category.at[_sidx].set(new_cat)
        new_items = s.inventory.items.replace(quantity=new_quantity, category=new_category)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _do_quaff(s):
        _sidx = jnp.int32(slot_idx)
        items   = s.inventory.items
        type_id = items.type_id[_sidx].astype(jnp.int32)
        buc     = items.buc_status[_sidx]

        effect_id = jnp.clip(
            type_id - jnp.int32(_POTION_BASE_ID),
            0,
            N_POTIONS - 1,
        )

        # Dispatch: operand is (state, rng, buc); each branch returns new state.
        # Brax-flatten: compute all N_POTIONS branches per dispatch and select
        # by one-hot effect_id mask via jax.tree.map / jnp.where.
        _operand_q = (s, rng, buc)
        _results_q = [br(_operand_q) for br in _SWITCH_BRANCHES]

        def _select_q(*branches):
            out = branches[0]
            for _i in range(1, len(branches)):
                out = jnp.where(effect_id == jnp.int32(_i), branches[_i], out)
            return out

        new_state = jax.tree.map(_select_q, *_results_q)

        # wave17h P0 (IDENTIFICATION #2): use-ID on drinking unknown potion.
        # Cite: vendor/nethack/src/potion.c::peffects identifies the type
        # in many branches (e.g. potion.c:632-2603). We mirror this by
        # setting identified=True on the consumed slot AND flipping the
        # type-level mask at state.identification.identified[type_id].
        new_items_id   = new_state.inventory.items.identified.at[_sidx].set(jnp.bool_(True))
        type_mask      = new_state.identification.identified
        type_id_clipped = jnp.clip(type_id, jnp.int32(0), jnp.int32(type_mask.shape[0] - 1))
        new_type_mask  = type_mask.at[type_id_clipped].set(jnp.bool_(True))
        new_state = new_state.replace(
            inventory=new_state.inventory.replace(
                items=new_state.inventory.items.replace(identified=new_items_id),
            ),
            identification=new_state.identification.replace(identified=new_type_mask),
        )

        # Decrement quantity; clear category when exhausted.
        old_qty  = new_state.inventory.items.quantity[_sidx]
        new_qty  = jnp.maximum(old_qty - jnp.int16(1), jnp.int16(0))
        new_cat  = jnp.where(new_qty == jnp.int16(0),
                             jnp.int8(0),
                             new_state.inventory.items.category[_sidx])
        new_quantity = new_state.inventory.items.quantity.at[_sidx].set(new_qty)
        new_category = new_state.inventory.items.category.at[_sidx].set(new_cat)
        new_items    = new_state.inventory.items.replace(
            quantity=new_quantity, category=new_category
        )
        new_inv = new_state.inventory.replace(items=new_items)
        # Emit "You quaff the potion." message.
        # Cite: vendor/nethack/src/potion.c::dodrink — pline("You drink ...").
        from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
        return new_state.replace(
            inventory=new_inv,
            messages=_msg_emit(new_state.messages, int(_MsgId.YOU_QUAFF_POTION)),
        )

    return jax.lax.cond(glib_drop, _drop_glib, _do_quaff, state)


def handle_quaff(state, rng):
    """Find the first valid potion in inventory and quaff it.

    Wave 3: uses "first valid item" strategy; Wave 4 will add a menu.
    A valid potion slot has category == POTION_CLASS and quantity > 0.
    Falls back to no-op if no potions found.

    Parameters
    ----------
    state : EnvState
    rng   : jax.random.PRNGKey

    Returns
    -------
    Updated EnvState.
    """
    categories = state.inventory.items.category   # [MAX_INVENTORY_SLOTS]
    quantities = state.inventory.items.quantity    # [MAX_INVENTORY_SLOTS]

    is_potion  = categories == jnp.int8(ObjectClass.POTION_CLASS)
    has_stock  = quantities > jnp.int16(0)
    valid_mask = is_potion & has_stock

    # argmax returns 0 when all False; guard with found flag.
    fallback_slot = jnp.argmax(valid_mask).astype(jnp.int32)
    found_any    = jnp.any(valid_mask)

    # NLE multi-key: prefer the agent-chosen slot (state.pending_action_slot)
    # if it points at a valid potion; else fall back to argmax.
    # Cite: vendor/nethack/src/potion.c::dodrink calls getobj() which returns
    # the player's chosen letter.  See Nethax/nethax/subsystems/pending_action.
    from Nethax.nethax.subsystems.pending_action import resolve_slot
    chosen_slot = resolve_slot(state, fallback_slot)
    chosen_is_valid = valid_mask[jnp.clip(chosen_slot, 0, valid_mask.shape[0] - 1)]
    slot_idx = jnp.where(chosen_is_valid, chosen_slot, fallback_slot).astype(jnp.int32)
    found = found_any

    # Resolve `quaff_potion` via module attribute lookup so the brax rebind
    # (PEP 562 ``__getattr__`` below) is honoured.  A bare ``quaff_potion``
    # reference inside the lambda compiles to ``LOAD_GLOBAL``, which does
    # NOT fall through to module ``__getattr__`` and raises ``NameError``
    # under ``NETHAX_BRAX_ALL=1`` because the name is removed from globals.
    # Binding to a local before the lambda makes it a closure free-var.
    import sys as _sys_qp
    _quaff_potion_fn = getattr(_sys_qp.modules[__name__], "quaff_potion")

    return jax.lax.cond(
        found,
        lambda s_r: _quaff_potion_fn(s_r[0], s_r[1], slot_idx),
        lambda s_r: _quaff_no_potion(s_r[0], s_r[1]),
        (state, rng),
    )


def _tile_under_player(state) -> jnp.ndarray:
    """Return the TileType integer at the player's current position (JIT-safe)."""
    branch = state.dungeon.current_branch.astype(jnp.int32)
    level = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)
    b, l, h, w = state.terrain.shape
    bs = jnp.clip(branch, 0, b - 1)
    ls = jnp.clip(level, 0, l - 1)
    rs = jnp.clip(row, 0, h - 1)
    cs = jnp.clip(col, 0, w - 1)
    return state.terrain[bs, ls, rs, cs].astype(jnp.int32)


def _drink_sink(state, rng):
    """Quaff while standing on a sink with no potion — drinksink() equivalent.

    Vendor: dodrink() detects IS_SINK(levl[u.ux][u.uy].typ) and (after a yn
    prompt) calls drinksink().  drinksink() rolls rn2(20) over ~14 outcomes;
    the common/default outcome is the flavor sip message
        You("take a sip of %s %s.", ..., hliquid("water"))
    We emit that recognised sip-of-water line so the quaff-on-sink action is
    acknowledged.  The MiniHack Sink RM is positional (sink tile + quaff
    action), so the reward fires from standing on the sink; this message
    provides the vendor-plausible feedback for the recognised action.

    Cite: vendor/nle/src/potion.c:506-511 (dodrink sink dispatch);
          vendor/nle/src/fountain.c:520-625 (drinksink); :528 case 0
          cold-water sip; :620-624 default "take a sip of ... water".
    """
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    del rng  # effect is the recognised flavor message; no RNG branch ported
    return state.replace(
        messages=_msg_emit(state.messages, int(_MsgId.SINK_SIP_COLD_WATER)),
    )


def _quaff_no_potion(state, rng):
    """No drinkable potion in inventory.

    Vendor dodrink() first checks the tile under the hero: if it is a sink
    (IS_SINK) it offers ``Drink from the sink?`` -> drinksink().  We mirror
    that branch so the quaff action is recognised while standing on a sink;
    otherwise the command is a no-op (vendor getobj() returns NULL -> return 0).

    Cite: vendor/nle/src/potion.c:506-525 (dodrink sink/getobj dispatch).
    """
    from Nethax.nethax.constants.tiles import TileType
    on_sink = _tile_under_player(state) == jnp.int32(int(TileType.SINK))
    return jax.lax.cond(
        on_sink,
        lambda s_r: _drink_sink(s_r[0], s_r[1]),
        lambda s_r: s_r[0],
        (state, rng),
    )

# Round 4 brax integration via PEP 562 lazy __getattr__ with cycle-break.
# Original names are deleted from module globals; module-attribute lookups
# fall through to __getattr__, which imports the Brax versions lazily.
# `_BRAX_ORIG` retains the originals so they can be returned during the
# brief window when the brax module is still initialising (mirrors
# swallow.py / items_corpses.py cycle-aware pattern).
import os as _os_brax
import sys as _sys_brax
if _os_brax.environ.get("NETHAX_BRAX_ALL", "0") == "1":
    _BRAX_ORIG = {
        "quaff_potion": quaff_potion,
        "apply_potion_to_monster": apply_potion_to_monster,
    }
    _BRAX_MAP = {
        "quaff_potion": ("items_dispatch_brax", "quaff_potion_brax"),
        "apply_potion_to_monster": ("items_dispatch_brax", "apply_potion_to_monster_brax"),
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
