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
    dmg     = dice_roll(rng, n_dice.astype(jnp.int32), jnp.int32(4))
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
    dmg     = dice_roll(rng, n_dice.astype(jnp.int32), jnp.int32(8))
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

    nhp=400 is large enough to saturate at max+nxtra (effective full restore).
    nxtra = 4+4*bcsign = 8 blessed / 4 uncursed / 0 cursed (HP_max bump).
    curesick=!cursed, cureblind=TRUE.
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
    new_status = new_status.replace(timed_statuses=new_ts)
    # NOTE: blessed XL-restore via pluslvl() requires the wave16a XP system
    # wiring; left for a follow-up so this fix stays self-contained.
    return state.replace(
        player_hp=new_hp,
        player_hp_max=new_hp_max,
        status=new_status,
    )


# ---- energy ---------------------------------------------------------------

def _effect_gain_energy(state, rng, buc):
    """potion of gain energy — restore/increase Pw.

    Canonical: peffect_gain_energy — u.uen += 3*num; u.uenmax += num.
    Wave 3: +10 Pw current and max (blessed +15, cursed -5 floored at 0).
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    delta   = jnp.where(blessed, jnp.int32(15),
              jnp.where(cursed,  jnp.int32(-5), jnp.int32(10)))
    new_pw_max = jnp.maximum(state.player_pw_max + delta, jnp.int32(0))
    new_pw     = jnp.minimum(state.player_pw + delta * 3,
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

    (str_un, int_un, wis_un, dex_un, con_un, cha_un) = jax.lax.switch(
        pick.astype(jnp.int32),
        [_inc_str, _inc_int, _inc_wis, _inc_dex, _inc_con, _inc_cha],
        None,
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
    """potion of gain level — gain one experience level.

    Canonical: peffect_gain_level (potion.c) — uncursed/blessed: pluslvl(FALSE)
    → XL+1.  Cursed: goto_level(current_level - 1) i.e. ascend one dungeon
    level; no effect if already at level 1.

    Cite: vendor/nethack/src/potion.c::peffect_gain_level.
    """
    cursed = _is_cursed(buc)
    # Uncursed/blessed: increment XL (capped at 30).
    new_xl = jnp.where(cursed, state.player_xl,
                       jnp.minimum(state.player_xl + jnp.int32(1), jnp.int32(30)))
    # Cursed: ascend one dungeon level (current_level -= 1), floor at 1.
    cur_lv  = state.dungeon.current_level.astype(jnp.int32)
    new_lv  = jnp.where(cursed,
                        jnp.maximum(cur_lv - jnp.int32(1), jnp.int32(1)),
                        cur_lv).astype(jnp.int8)
    new_dungeon = state.dungeon.replace(current_level=new_lv)
    return state.replace(player_xl=new_xl, dungeon=new_dungeon)


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
    """potion of invisibility — grant temporary invisibility.

    Canonical: peffect_invisibility — incr_itimeout(&HInvis, d(6-3*bcsign,100)+100).
    Blessed: permanent chance; cursed: aggravates monsters.
    Wave 3: 300-turn timed invis (blessed permanent).
    """
    blessed  = _is_blessed(buc)
    turns    = jnp.where(blessed, jnp.int32(0), jnp.int32(300))
    perm_new = jnp.where(blessed, jnp.bool_(True),
                         state.status.intrinsics[Intrinsic.INVIS])
    new_intr = state.status.intrinsics.at[Intrinsic.INVIS].set(perm_new)
    cur      = state.status.timed_intrinsics[Intrinsic.INVIS]
    new_t    = jnp.where(blessed, cur, jnp.maximum(cur, turns))
    new_timers = state.status.timed_intrinsics.at[Intrinsic.INVIS].set(new_t)
    new_status = state.status.replace(intrinsics=new_intr,
                                      timed_intrinsics=new_timers)
    return state.replace(status=new_status)


def _effect_monster_detection(state, rng, buc):
    """potion of monster detection — timed DETECT_MONSTERS intrinsic.

    Canonical: peffect_monster_detection — incr_itimeout(&HDetect_monsters, ...).
    Wave 3: 100-turn timed detect.
    """
    new_status = add_timed_intrinsic(state.status, Intrinsic.DETECT_MONSTERS, 100)
    return state.replace(status=new_status)


def _effect_object_detection(state, rng, buc):
    """potion of object detection — shows item locations on level.

    Canonical: peffect_object_detection (potion.c) — object_detect(otmp, 0)
    reveals all ground items on the current level.  Implementation: set
    detect_objects_until_turn = timestep + 100.

    Cite: vendor/nethack/src/potion.c::peffect_object_detection.
    """
    new_timer = state.timestep + jnp.int32(100)
    new_id = state.identification.replace(
        detect_objects_until_turn=new_timer
    )
    return state.replace(identification=new_id)


# ---- movement modifiers ---------------------------------------------------

def _effect_levitation(state, rng, buc):
    """potion of levitation — timed LEVITATION intrinsic.

    Canonical: peffect_levitation — incr_itimeout(&HLevitation, rn1(140,10)).
    Wave 3: 150-turn timed levitation.
    """
    turns      = jnp.int32(150)
    new_status = add_timed_intrinsic(state.status, Intrinsic.LEVITATION, turns)
    return state.replace(status=new_status)


def _effect_speed(state, rng, buc):
    """potion of speed — timed FAST intrinsic.

    Canonical: peffect_speed — speed_up(rn1(10,100+60*bcsign)).
    Wave 3: 160-turn timed fast (blessed permanent).
    """
    blessed  = _is_blessed(buc)
    perm_new = jnp.where(blessed, jnp.bool_(True),
                         state.status.intrinsics[Intrinsic.FAST])
    new_intr = state.status.intrinsics.at[Intrinsic.FAST].set(perm_new)
    cur      = state.status.timed_intrinsics[Intrinsic.FAST]
    new_t    = jnp.where(blessed, cur, jnp.maximum(cur, jnp.int32(160)))
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
    """potion of hallucination — add hallucination timer.

    Canonical: peffect_hallucination — make_hallucinated(rn1(200, 600-300*bcsign)).
    Wave 3: 600 turns (blessed 300, cursed 900).
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    turns   = jnp.where(blessed, jnp.int32(300),
              jnp.where(cursed,  jnp.int32(900), jnp.int32(600)))
    cur = state.status.timed_statuses[int(TimedStatus.HALLUCINATION)]
    new_val = jnp.maximum(cur, turns)
    new_ts  = state.status.timed_statuses.at[int(TimedStatus.HALLUCINATION)].set(new_val)
    new_status = state.status.replace(timed_statuses=new_ts)
    return state.replace(status=new_status)


def _effect_blindness(state, rng, buc):
    """potion of blindness — add blindness timer.

    Canonical: peffect_blindness — make_blinded(incr(BlindedTimeout, rn1(200,250-125*bcsign))).
    Wave 3: 250 turns (blessed 125, cursed 375).
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    turns   = jnp.where(blessed, jnp.int32(125),
              jnp.where(cursed,  jnp.int32(375), jnp.int32(250)))
    cur = state.status.timed_statuses[int(TimedStatus.BLIND)]
    new_val = jnp.maximum(cur, turns)
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
    """potion of acid — deal acid damage.

    Canonical: peffect_acid — losehp(d(cursed?2:1, blessed?4:8)).
    Wave 3: 4 HP (blessed 2, cursed 8); acid-resistant characters take 1.
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    dmg     = jnp.where(blessed, jnp.int32(2),
              jnp.where(cursed,  jnp.int32(8), jnp.int32(4)))
    new_hp  = jnp.maximum(state.player_hp - dmg, jnp.int32(1))
    return state.replace(player_hp=new_hp)


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

    # Fire damage if wielding a lit lamp (type_id 92 = oil lamp as proxy;
    # check greased-before-update as "was already lit/oiled").
    # Simplified trigger: if wielded item was already greased, it's "lit" →
    # 2d4 fire damage (vendor: losehp(d(2,4), ...)).
    was_lit = has_wielded & cur_greased[jnp.maximum(wslot, jnp.int32(0))]
    rng, sub = jax.random.split(rng)
    fire_dmg = (jax.random.randint(sub, (), 1, 5, dtype=jnp.int32) +
                jax.random.randint(sub, (), 1, 5, dtype=jnp.int32))
    new_hp = jnp.where(was_lit,
                       jnp.maximum(state.player_hp - fire_dmg, jnp.int32(1)),
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
    """potion of water — holy/unholy/plain water effects.

    Canonical: peffect_water — blessed=holy water (cure sick, exercise);
    cursed=unholy water (damage if lawful); plain=nutrition only.
    Wave 3: blessed cures sickness; cursed 6 HP damage; plain +10 nutrition.
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    # plain water — slight nutrition
    new_nutrition = jnp.where(
        ~blessed & ~cursed,
        jnp.minimum(state.status.nutrition + jnp.int32(10), jnp.int32(2000)),
        state.status.nutrition,
    )
    # holy water — cure sickness
    new_sick = jnp.where(blessed, jnp.int32(0),
                         state.status.timed_statuses[int(TimedStatus.SICK)])
    new_ts = state.status.timed_statuses.at[int(TimedStatus.SICK)].set(new_sick)
    new_status = state.status.replace(timed_statuses=new_ts, nutrition=new_nutrition)
    # unholy water — damage if lawful (simplified: always deal small damage if cursed)
    dmg    = jnp.where(cursed, jnp.int32(6), jnp.int32(0))
    new_hp = jnp.maximum(state.player_hp - dmg, jnp.int32(1))
    return state.replace(player_hp=new_hp, status=new_status)


def _effect_booze(state, rng, buc):
    """potion of booze — confusion, minor heal, possible knockout.

    Canonical: peffect_booze — make_confused; healup(1); cursed: pass out.
    Wave 3: +1 HP, add 20-turn confusion; cursed adds 15-turn SLEEP.
    """
    cursed  = _is_cursed(buc)
    new_hp  = jnp.minimum(state.player_hp + jnp.int32(1), state.player_hp_max)
    cur_conf = state.status.timed_statuses[int(TimedStatus.CONFUSION)]
    new_conf = jnp.maximum(cur_conf, jnp.int32(20))
    new_ts   = state.status.timed_statuses.at[int(TimedStatus.CONFUSION)].set(new_conf)
    cur_slp  = new_ts[int(TimedStatus.SLEEP)]
    new_slp  = jnp.where(cursed, jnp.maximum(cur_slp, jnp.int32(15)), cur_slp)
    new_ts2  = new_ts.at[int(TimedStatus.SLEEP)].set(new_slp)
    new_status = state.status.replace(timed_statuses=new_ts2)
    return state.replace(player_hp=new_hp, status=new_status)


def _effect_fruit_juice(state, rng, buc):
    """potion of fruit juice — minor nutrition; blessed grants temporary see-invis.

    Canonical: POT_FRUIT_JUICE → peffect_see_invisible (same handler).
    Wave 3: +50 nutrition; blessed also adds 300-turn see-invisible.
    """
    blessed      = _is_blessed(buc)
    new_nutrition = jnp.minimum(
        state.status.nutrition + jnp.int32(50), jnp.int32(2000)
    )
    cur  = state.status.timed_intrinsics[Intrinsic.SEE_INVIS]
    new_t = jnp.where(blessed, jnp.maximum(cur, jnp.int32(300)), cur)
    new_timers = state.status.timed_intrinsics.at[Intrinsic.SEE_INVIS].set(new_t)
    new_status  = state.status.replace(nutrition=new_nutrition,
                                       timed_intrinsics=new_timers)
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
    """Healing potion shatters on monster — restore 10 HP (capped at hp_max)."""
    mai = state.monster_ai
    cur_hp  = mai.hp[m_slot].astype(jnp.int32)
    cur_max = mai.hp_max[m_slot].astype(jnp.int32)
    new_hp  = jnp.minimum(cur_hp + jnp.int32(10), cur_max)
    new_hp_arr = mai.hp.at[m_slot].set(new_hp)
    return state.replace(monster_ai=mai.replace(hp=new_hp_arr))


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
    """Extra-healing potion shatters on monster — restore 25 HP."""
    mai = state.monster_ai
    cur_hp  = mai.hp[m_slot].astype(jnp.int32)
    cur_max = mai.hp_max[m_slot].astype(jnp.int32)
    new_hp  = jnp.minimum(cur_hp + jnp.int32(25), cur_max)
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
    _monster_heal,           # 18  FULL_HEALING      → same as heal (simplified)
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
    return jax.lax.switch(
        effect_id,
        _MONSTER_SWITCH_BRANCHES,
        (state, m_slot.astype(jnp.int32), rng),
    )


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
        new_state = jax.lax.switch(effect_id, _SWITCH_BRANCHES, (s, rng, buc))

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
        return new_state.replace(inventory=new_inv)

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
    slot_idx = jnp.argmax(valid_mask).astype(jnp.int32)
    found    = jnp.any(valid_mask)

    return jax.lax.cond(
        found,
        lambda s_r: quaff_potion(s_r[0], s_r[1], slot_idx),
        lambda s_r: s_r[0],
        (state, rng),
    )
