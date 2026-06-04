"""Depth-curve monster spawning.

Canonical sources:
    vendor/nethack/src/makemon.c::makemon  — monster creation logic
    vendor/nethack/include/permonst.h::monstr[]  — difficulty rating
    vendor/nethack/src/mondata.c::mstrength — difficulty formula
    vendor/nle/src/monst.c  — MON() macro's trailing ``d`` (difficulty)

Public surface:
    MONSTR_DIFFICULTIES: module-level JAX constant (one int per monster).
        Sourced from the vendor ``mons[i].difficulty`` field (preferred)
        and falls back to the full mstrength() speed/breath/petrify/...
        formula (``_compute_monstr_full``) when the vendor field is the
        uninitialised 0 sentinel.  See ``_compute_difficulties`` below.
    eligible_monsters_for_depth: depth-windowed mask excluding G_NOGEN /
        G_UNIQ / genocided species (vendor makemon.c::rndmonst lines
        1185-1244).
    pick_monster_for_level: weighted random selection by gen_freq
        (vendor pm_gen lines 1186-1213).
    spawn_initial_monsters: roll HP via newmonhp (``_roll_hp``) + pick
        valid placement tile.
    populate_level_with_monsters: write spawned monsters into EnvState.

Notes on coverage:
    - Group spawning (G_SGROUP / G_LGROUP) — flags are read for the
      ``mstrength`` difficulty bonus (see _compute_monstr_full lines
      125-126); spatially-clustered placement is delegated to the level
      generator that calls into this module (mklev.c::mkfount style
      bunching is handled at level-construction time).
    - Unique placement (G_UNIQ) — filtered out of rndmonst by
      ``_IS_UNIQ`` mask; unique monsters (Wizard of Yendor, Vlad, demon
      princes, named questleaders/nemeses) are placed explicitly by the
      special-level factories in special_levels.py / quest_levels.py per
      vendor src/dungeon.c::place_special.
    - HP roll = d(mlvl, 8), with rnd(4) for mlvl==0 — byte-equal vendor
      makemon.c::newmonhp (see ``_roll_hp``).
    - Terrain validity — caller supplies ``valid_tiles_mask`` derived
      from TileType; this mirrors vendor goodpos() (mondata.c lines
      1402-1470) at the granularity needed for spawn placement.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from Nethax.nethax.constants.monsters import (
    MONSTERS,
    NUMMONS,
    G_NOGEN,
    G_UNIQ,
    G_SGROUP,
    G_LGROUP,
    AttackType,
    DamageType,
    MonsterSymbol,
    MZ_LARGE,
    MZ_HUGE,
    MZ_GIGANTIC,
    M2_MAGIC,
    M2_NASTY,
    M2_GREEDY,
    M2_MERC,
    M2_STRONG,
    M2_NEUTER,
    M2_FEMALE,
    M2_MALE,
    M1_HUMANOID,
    M2_DEMON,
    M2_LORD,
    M2_PRINCE,
    M2_MINION,
    M2_HOSTILE,
    M2_PEACEFUL,
    M2_DOMESTIC,
    M2_SHAPESHIFTER,
    MS_SOLDIER,
    MS_PRIEST,
    MS_SPELL,
    MS_SELL,
    MS_LEADER,
    MS_GUARDIAN,
    MS_NEMESIS,
    MS_RIDER,
)
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.monster_ai import (
    MAX_MONSTERS_PER_LEVEL,
    MAX_MONSTER_INV,
    _MONSTER_MRESISTS,
    _MONSTER_UNDEAD,
    _MONSTER_NONLIVING,
)
from Nethax.nethax.vendor_rng import (
    Isaac64State, randint_jax, isaac_weighted_choice, isaac_rndmonst_choice,
    rnd_jax, next_uint64_jax,
)
from Nethax.nethax.subsystems.random_objects import (
    _armor_draws,
    _weapon_draws,
    _gem_draws,
    _potion_scroll_draws,
    _wand_draws,
    _food_draws,
)


# ---------------------------------------------------------------------------
# Module-level constants built at import time from the Python MONSTERS tuple
# ---------------------------------------------------------------------------

def _mstrength_ranged_attk(entry) -> bool:
    """Mirror of vendor mondata.c::mstrength_ranged_attk (lines 500-512).

    Returns True if any attack type is AT_WEAP or higher (AT_WEAP=254,
    AT_MAGC=255), OR is one of {AT_BREA, AT_SPIT, AT_GAZE}.

    Vendor (mondata.c:504-509):
        atk_mask = (1 << AT_BREA) | (1 << AT_SPIT) | (1 << AT_GAZE)
        for i in NATTK:
            j = mattk[i].aatyp
            if j >= AT_WEAP or (j < 32 && (atk_mask & (1 << j)) != 0):
                return TRUE
    """
    for atk in (entry.attacks or ()):
        j = int(atk[0])
        if j >= int(AttackType.AT_WEAP):
            return True
        if j < 32 and j in (
            int(AttackType.AT_BREA),
            int(AttackType.AT_SPIT),
            int(AttackType.AT_GAZE),
        ):
            return True
    return False


def _compute_monstr_full(entry) -> int:
    """Byte-equal vendor mstrength formula.

    Canonical source: vendor/nethack/src/mondata.c::mstrength
    (lines 428-498, with helper mstrength_ranged_attk at lines 500-512).

    Steps (vendor):
      tmp = ptr->mlevel; if (tmp > 49) tmp = 2 * (tmp - 6) / 4;
      n  = !!(geno & G_SGROUP) + (!!(geno & G_LGROUP) << 1)
      n += mstrength_ranged_attk(ptr)
      n += (ac < 4) + (ac < 0)
      n += (mmove >= 18)
      for each attack:
          n += (aatyp > 0)
          n += (aatyp == AT_MAGC)
          n += (aatyp == AT_WEAP && (mflags2 & M2_STRONG))
          if (aatyp == AT_EXPL):
              n += 3 if adtyp in (AD_COLD, AD_FIRE) else 5 if adtyp == AD_ELEC else 0
      for each attack:
          if adtyp in (AD_DRLI, AD_STON, AD_DRST, AD_DRDX, AD_DRCO, AD_WERE): n += 2
          elif name != "grid bug": n += (adtyp != AD_PHYS)
          n += (damd * damn > 23)
      if name == "leprechaun":  n -= 2
      if name in ("killer bee", "soldier ant"):  n += 2
      if n == 0:        tmp -= 1
      elif n < 6:       tmp += (n // 3 + 1)
      else:             tmp += (n // 2)
      return max(tmp, 0)
    """
    tmp = int(entry.level)
    if tmp > 49:
        tmp = 2 * (tmp - 6) // 4

    geno = int(entry.generation_mask)
    n = 1 if (geno & G_SGROUP) else 0
    n += (1 if (geno & G_LGROUP) else 0) << 1

    # Ranged attack bonus (mondata.c:440-441).
    if _mstrength_ranged_attk(entry):
        n += 1

    # AC threshold bumps (mondata.c:444-445).
    ac = int(entry.ac)
    if ac < 4:
        n += 1
    if ac < 0:
        n += 1

    # Speed bump (mondata.c:448).
    if int(entry.move_speed) >= 18:
        n += 1

    mflags2 = int(entry.flags2) & 0xFFFFFFFF
    name = entry.name

    AT_MAGC = int(AttackType.AT_MAGC)
    AT_WEAP = int(AttackType.AT_WEAP)
    AT_EXPL = int(AttackType.AT_EXPL)
    AD_COLD = int(DamageType.AD_COLD)
    AD_FIRE = int(DamageType.AD_FIRE)
    AD_ELEC = int(DamageType.AD_ELEC)
    AD_DRLI = int(DamageType.AD_DRLI)
    AD_STON = int(DamageType.AD_STON)
    AD_DRST = int(DamageType.AD_DRST)
    AD_DRDX = int(DamageType.AD_DRDX)
    AD_DRCO = int(DamageType.AD_DRCO)
    AD_WERE = int(DamageType.AD_WERE)
    AD_PHYS = int(DamageType.AD_PHYS)
    M2_STRONG_MASK = int(M2_STRONG) & 0xFFFFFFFF

    attacks = entry.attacks or ()

    # First attack loop (mondata.c:451-464).
    for atk in attacks:
        aatyp = int(atk[0])
        adtyp = int(atk[1])
        if aatyp > 0:
            n += 1
        if aatyp == AT_MAGC:
            n += 1
        if aatyp == AT_WEAP and (mflags2 & M2_STRONG_MASK):
            n += 1
        if aatyp == AT_EXPL:
            if adtyp == AD_COLD or adtyp == AD_FIRE:
                n += 3
            elif adtyp == AD_ELEC:
                n += 5

    # Second attack loop — special damage (mondata.c:467-475).
    for atk in attacks:
        adtyp = int(atk[1])
        damn = int(atk[2])
        damd = int(atk[3])
        if adtyp in (AD_DRLI, AD_STON, AD_DRST, AD_DRDX, AD_DRCO, AD_WERE):
            n += 2
        elif name != "grid bug":
            # Vendor uses strcmp() != 0 ⇒ adds (adtyp != AD_PHYS) for every
            # non-grid-bug.  Note: matches vendor even though it shadows the
            # AD_PHYS=0 default in NO_ATTK slots.
            n += 1 if (adtyp != AD_PHYS) else 0
        # damd * damn > 23 bump applies to every monster including grid bug.
        if (damd * damn) > 23:
            n += 1

    # Name-keyed bumps (mondata.c:479-486).
    if name == "leprechaun":
        n -= 2
    if name in ("killer bee", "soldier ant"):
        n += 2

    # Final scaling (mondata.c:488-494).
    if n == 0:
        tmp -= 1
    elif n < 6:
        tmp += (n // 3 + 1)
    else:
        tmp += (n // 2)

    return tmp if tmp >= 0 else 0


def _compute_difficulties() -> jnp.ndarray:
    """Build MONSTR_DIFFICULTIES array from MONSTERS at import time.

    Wave 6 closing audit: prefer the vendor-table ``difficulty`` field
    (mons[i].difficulty, from vendor/nle/src/monst.c MON() macro's trailing
    `d` arg — lines 47-50) when populated. Falls back to the
    speed/breath/petrify formula in ``_compute_monstr_full`` when
    ``entry.difficulty == 0`` (uninitialised sentinel).
    """
    diffs = []
    for m in MONSTERS:
        vendor_d = int(getattr(m, "difficulty", 0))
        if vendor_d > 0:
            diffs.append(vendor_d)
        else:
            diffs.append(_compute_monstr_full(m))
    return jnp.array(diffs, dtype=jnp.int32)


def _compute_mlevels() -> jnp.ndarray:
    """Build the base monster level array (vendor ``permonst.mlevel``).

    This is the raw ``mons[i].mlevel`` (the PermonstEntry ``level`` field),
    distinct from ``difficulty`` (the mstrength rating).  ``newmonhp`` uses
    ``mon->m_lev = adj_lev(ptr)`` which derives from ``mlevel`` — NOT the
    difficulty.  e.g. lichen has mlevel=0 (HP path = rnd(4)) but difficulty=1.
    Cite: vendor/nle/src/makemon.c:989 (m_lev = adj_lev(ptr)).
    """
    return jnp.array([int(m.level) for m in MONSTERS], dtype=jnp.int32)


def _adj_lev_jax(mlevel: jnp.ndarray, level_difficulty: int,
                 ulevel: int) -> jnp.ndarray:
    """JAX-traceable vendor ``adj_lev`` (makemon.c:1757-1790), common case.

    Excludes the Wizard-of-Yendor special case (mklev never spawns him) and
    the mlevel>49 special-demon clamp (no Dlvl-1 monster reaches that).  For
    ordinary monsters::

        tmp  = mlevel;
        d    = level_difficulty() - tmp;
        if (d < 0) tmp--; else tmp += d / 5;
        u    = u.ulevel - mlevel;
        if (u > 0) tmp += u / 4;
        cap  = min(3*mlevel/2, 49);
        return (tmp > cap) ? cap : (tmp > 0 ? tmp : 0);

    Cite: vendor/nle/src/makemon.c:1757-1790.
    """
    tmp = mlevel.astype(jnp.int32)
    ld = jnp.int32(level_difficulty)
    ul = jnp.int32(ulevel)
    d = ld - tmp
    tmp = jnp.where(d < 0, tmp - jnp.int32(1), tmp + d // jnp.int32(5))
    u = ul - mlevel.astype(jnp.int32)
    tmp = jnp.where(u > 0, tmp + u // jnp.int32(4), tmp)
    cap = jnp.minimum((jnp.int32(3) * mlevel.astype(jnp.int32)) // jnp.int32(2),
                      jnp.int32(49))
    return jnp.where(tmp > cap, cap, jnp.maximum(tmp, jnp.int32(0)))


def _compute_gen_freqs() -> jnp.ndarray:
    """Extract the generation frequency weight (vendor ``geno & G_FREQ``).

    G_FREQ is the low 3 bits (0x0007) of ``permonst.geno`` — values 0..7
    (vendor/nle/include/monflag.h:187).  The higher bits hold unrelated
    generation flags (G_GENO/G_SGROUP/G_LGROUP/...), so masking the whole
    low byte over-counts the frequency.  Vendor ``rndmonst`` weights each
    eligible monster by exactly ``(ptr->geno & G_FREQ)`` (makemon.c:1570).
    Cite: vendor/nle/include/monflag.h:187 (G_FREQ 0x0007);
          vendor/nle/src/makemon.c:1570.
    """
    freqs = [m.generation_mask & 0x0007 for m in MONSTERS]
    return jnp.array(freqs, dtype=jnp.int32)


def _compute_nogen_mask() -> jnp.ndarray:
    """True where monster has G_NOGEN flag (not spawnable via normal generation)."""
    flags = [(m.generation_mask & G_NOGEN) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_uniq_mask() -> jnp.ndarray:
    """True where monster has G_UNIQ flag."""
    flags = [(m.generation_mask & G_UNIQ) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_large() -> jnp.ndarray:
    """True where monster size >= MZ_LARGE."""
    flags = [m.size >= MZ_LARGE for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_base_ac() -> jnp.ndarray:
    """Base AC for each monster type."""
    acs = [m.ac for m in MONSTERS]
    return jnp.array(acs, dtype=jnp.int8)


def _compute_is_neuter() -> jnp.ndarray:
    """True where monster's flags2 has M2_NEUTER bit set.

    Cite: vendor/nle/include/mondata.h:121
        #define is_neuter(ptr) (((ptr)->mflags2 & M2_NEUTER) != 0L)
    Used to gate the post-newmonhp ``rn2(2)`` female draw in
    vendor/nle/src/makemon.c:1226.
    """
    mask = int(M2_NEUTER) & 0xFFFFFFFF
    flags = [(int(m.flags2) & 0xFFFFFFFF & mask) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_armed() -> jnp.ndarray:
    """True where monster has any attack of type AT_WEAP.

    Cite: vendor/nle/include/mondata.h:94
        #define is_armed(ptr) attacktype(ptr, AT_WEAP)
    Drives the m_initweap() branch in vendor/nle/src/makemon.c:1381-1382:
        if (is_armed(ptr))
            m_initweap(mtmp);
    """
    AT_WEAP = int(AttackType.AT_WEAP)
    flags = []
    for m in MONSTERS:
        armed = False
        for atk in (m.attacks or ()):
            if int(atk[0]) == AT_WEAP:
                armed = True
                break
        flags.append(armed)
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_primary_attack_dice() -> tuple[jnp.ndarray, jnp.ndarray]:
    """(n_dice, sides) for the first non-passive attack of each monster."""
    n_arr = []
    s_arr = []
    for m in MONSTERS:
        n, s = 1, 4  # default: 1d4
        for atk in m.attacks:
            if atk[0] != AttackType.AT_NONE and atk[2] > 0:
                n, s = atk[2], atk[3]
                break
        n_arr.append(n)
        s_arr.append(s)
    # Some vendor entries use values >127 (e.g. 255 sentinel); int16 is safe.
    return jnp.array(n_arr, dtype=jnp.int16), jnp.array(s_arr, dtype=jnp.int16)


# Build all constants once at import time.
MONSTR_DIFFICULTIES: jnp.ndarray = _compute_difficulties()   # [NUMMONS] int32
_MONSTER_MLEVEL: jnp.ndarray = _compute_mlevels()            # [NUMMONS] int32
_GEN_FREQS: jnp.ndarray = _compute_gen_freqs()               # [NUMMONS] int32
_IS_NOGEN: jnp.ndarray = _compute_nogen_mask()               # [NUMMONS] bool
_IS_UNIQ: jnp.ndarray = _compute_uniq_mask()                 # [NUMMONS] bool
_IS_LARGE: jnp.ndarray = _compute_is_large()                 # [NUMMONS] bool
_BASE_AC: jnp.ndarray = _compute_base_ac()                   # [NUMMONS] int8
_ATK_DICE_N, _ATK_DICE_S = _compute_primary_attack_dice()    # [NUMMONS] int8 each
_IS_NEUTER: jnp.ndarray = _compute_is_neuter()               # [NUMMONS] bool
_IS_ARMED: jnp.ndarray = _compute_is_armed()                 # [NUMMONS] bool


def _compute_has_fixed_gender() -> jnp.ndarray:
    """True where M2_FEMALE or M2_MALE flag2 is set.

    Vendor (vendor/nle/src/makemon.c:1214-1226):
        if (is_female(ptr))   mtmp->female = TRUE;
        else if (is_male(ptr)) mtmp->female = FALSE;
        ... (quest leader/nemesis preset) ...
        else mtmp->female = rn2(2);

    is_female/is_male check M2_FEMALE/M2_MALE (mondata.h:119-120).
    The rn2(2) draw fires when BOTH bits are absent — G_NEUTER monsters
    STILL draw rn2(2) (vendor comment "ignored for neuters" describes
    semantic ignoring; the RNG call still happens).
    """
    mask = (int(M2_FEMALE) | int(M2_MALE)) & 0xFFFFFFFF
    flags = [(int(m.flags2) & 0xFFFFFFFF & mask) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_likes_gold() -> jnp.ndarray:
    """True where M2_GREEDY flag2 is set.

    Vendor (vendor/nle/include/mondata.h:142):
        #define likes_gold(ptr) (((ptr)->mflags2 & M2_GREEDY) != 0L)
    Used by vendor/nle/src/makemon.c:798:
        if (likes_gold(ptr) && !findgold(mtmp->minvent) && !rn2(5)) ...
    C short-circuit: rn2(5) ONLY fires when likes_gold is true.
    """
    mask = int(M2_GREEDY) & 0xFFFFFFFF
    flags = [(int(m.flags2) & 0xFFFFFFFF & mask) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


_HAS_FIXED_GENDER: jnp.ndarray = _compute_has_fixed_gender()  # [NUMMONS] bool
_LIKES_GOLD:       jnp.ndarray = _compute_likes_gold()        # [NUMMONS] bool


def _compute_initweap_draw_count() -> jnp.ndarray:
    """Per-monster max rn2 draws inside vendor m_initweap's switch.

    Cite: vendor/nle/src/makemon.c:182-554 (switch(ptr->mlet) body).
    For each mlet branch the vendor code is straight-line conditional
    rn2() calls; this table records the WORST-CASE rn2 draw count
    (assuming every internal short-circuit branch is taken).  Used to
    bound a while_loop that replaces the previous cap-8 fori_loop —
    tighter per-monster upper bound than the static cap.

    Non-armed monsters get 0 (the `is_armed` gate skips the whole call).
    The trailing rn2(75) at vendor line 556 is NOT included (it's
    handled separately after the switch).
    """
    # Per-mlet worst-case draw counts inside m_initweap switch body.
    # Derived by counting straight-line rn2 calls in each case branch.
    # Conservative (worst-case branch taken).
    counts_by_mlet = {
        int(MonsterSymbol.S_GIANT):     1,   # rn2(2)            line 184
        int(MonsterSymbol.S_HUMAN):     7,   # elf branch peak; mercenary handled below
        # S_ANGEL handled explicitly outside the generic loop — gated on
        # humanoid(ptr) with conditional `!rn2(20) || is_lord` artiname,
        # rn2(4) spe, and `!rn2(4) || is_lord` shield pick.  See
        # _draw_angel_initweap.  Vendor: makemon.c:326-348.
        int(MonsterSymbol.S_ANGEL):     0,   # explicit handler              line 326-348
        int(MonsterSymbol.S_HUMANOID):  6,   # dwarf worst case             line 367-386
        # S_KOP handled explicitly outside the generic loop (two gates each
        # with conditional inner draws — m_initthrow rn1 and weapon-pick
        # rn2(2)).  See _draw_kop_initweap in _consume_makemon_post_hp_draws.
        # Vendor: makemon.c:389-396.
        int(MonsterSymbol.S_KOP):       0,   # explicit handler             line 389-396
        # S_ORC handled explicitly outside the generic loop — the rn2(2)
        # ORCISH_HELM gate plus per-PM subtype branches (PM_MORDOR_ORC,
        # PM_URUK_HAI, PM_ORC_CAPTAIN, PM_ORC_SHAMAN) each launch their
        # own mongets()/m_initthrow mksobj_init cascades.  See
        # _draw_orc_initweap.  Vendor: makemon.c:397-432.
        int(MonsterSymbol.S_ORC):       0,   # explicit handler              line 397-432
        # S_OGRE handled explicitly outside the generic loop — the rn2(N)
        # gate divisor varies by subtype (PM_OGRE=10 vs OGRE_LORD/KING=5)
        # and on gate fire mongets(WAR_HAMMER) runs the WEAPON_CLASS
        # mksobj_init cascade.  See _draw_ogre_initweap.  Vendor:
        # makemon.c:434-438.
        int(MonsterSymbol.S_OGRE):      0,   # explicit handler              line 434-438
        # S_TROLL handled explicitly outside the generic loop — gate rn2(2)
        # gates an inner rn2(4) gem-type pick + GEM_CLASS mksobj_init draw.
        # See _draw_troll_initweap.  Vendor: makemon.c:440-454.
        int(MonsterSymbol.S_TROLL):     0,   # explicit handler              line 440-454
        # S_KOBOLD handled explicitly outside the generic loop (gate + rn1
        # m_initthrow draw is gate-conditional).  See _draw_kobold_initweap
        # in _consume_makemon_post_hp_draws.  Vendor: makemon.c:456-459.
        int(MonsterSymbol.S_KOBOLD):    0,   # explicit handler              line 457
        # S_CENTAUR handled explicitly outside the generic loop (gate + rn1
        # m_initthrow draw is gate-conditional).  See _draw_centaur_initweap
        # in _consume_makemon_post_hp_draws.  Vendor: makemon.c:461-471.
        int(MonsterSymbol.S_CENTAUR):   0,   # explicit handler              line 462
        int(MonsterSymbol.S_WRAITH):    0,   # no rn2                         line 472-475
        # S_ZOMBIE handled explicitly outside the generic loop — each of the
        # three rn2 gates additionally launches a mongets()/mksobj_init
        # cascade for RING_MAIL / LONG_SWORD / mlet_zombie_weap[mlev] when it
        # fires.  Vendor: makemon.c:477-481.  See _draw_zombie_initweap.
        int(MonsterSymbol.S_ZOMBIE):    0,   # explicit handler              line 477-481
        # S_LIZARD handled explicitly outside the generic loop — gated on
        # PM_SALAMANDER (the only AT_WEAP S_LIZARD) with chained ternary
        # `rn2(7) ? SPEAR : rn2(3) ? TRIDENT : STILETTO`.  See
        # _draw_lizard_initweap.  Vendor: makemon.c:482-486.
        int(MonsterSymbol.S_LIZARD):    0,   # explicit handler              line 482-486
        # S_DEMON inner switch handled by _draw_demon_inner_switch outside
        # the loop; FALLTHRU to default is handled by _draw_default_case
        # which is wired on _DEFAULT_ELIGIBLE (true for is_demon S_DEMONs).
        int(MonsterSymbol.S_DEMON):     0,   # explicit (inner + default)   line 487-553
    }
    # Default case (vendor makemon.c:512-553) is handled by the explicit
    # _draw_default_case handler outside this loop, wired on _DEFAULT_ELIGIBLE.
    # Setting count=0 here means the placeholder loop draws nothing; the
    # explicit handler then draws rnd(14-2*bias) plus the per-case cascade.
    default_count = 0  # default-case → _draw_default_case (line 512-553)

    # Per-monster overrides for PMs whose explicit handler runs OUTSIDE the
    # generic loop and consumes the full byte budget itself; force their loop
    # count to 0 so the placeholder does not double-draw.  Currently:
    # PM_HOBBIT — S_HUMANOID's hobbit branch (vendor makemon.c:351-366).
    hobbit_name = "hobbit"
    pm_hobbit_idx = -1
    for i, m in enumerate(MONSTERS):
        if m.name == hobbit_name:
            pm_hobbit_idx = i
            break

    counts = []
    for i, m in enumerate(MONSTERS):
        # Non-armed monsters skip m_initweap entirely; assign 0.
        armed = False
        for atk in (m.attacks or ()):
            if int(atk[0]) == int(AttackType.AT_WEAP):
                armed = True
                break
        if not armed:
            counts.append(0)
            continue
        # PM-level overrides (explicit handlers run outside the loop).
        if i == pm_hobbit_idx:
            counts.append(0)
            continue
        c = counts_by_mlet.get(int(m.symbol), default_count)
        counts.append(c)
    return jnp.array(counts, dtype=jnp.int32)


_INITWEAP_DRAW_COUNT: jnp.ndarray = _compute_initweap_draw_count()  # [NUMMONS] int32


# ---------------------------------------------------------------------------
# HP-path classification arrays  (vendor makemon.c::newmonhp lines 1018-1044)
# ---------------------------------------------------------------------------
#
# Vendor newmonhp dispatch order (makemon.c:1018-1044):
#   1. is_golem(ptr)          → golemhp(mndx) fixed value
#   2. is_rider(ptr)          → d(10, 8)  (basehp=10)
#   3. ptr->mlevel > 49       → 2*(mlevel-6) fixed value
#   4. S_DRAGON && mndx>=GRAY → 4*m_lev + d(m_lev, 4)  (non-endgame)
#   5. !m_lev                 → rnd(4)
#   6. else                   → d(m_lev, 8)
#
# is_golem: mlet == S_GOLEM  (mondata.h:108)
# is_rider: ptr == Death/Famine/Pestilence  (mondata.h:161-163)
# adult dragon: mlet==S_DRAGON && mndx >= PM_GRAY_DRAGON

def _compute_is_shapeshifter() -> jnp.ndarray:
    """True where monster has M2_SHAPESHIFTER flag2.

    Cite: vendor/nethack/src/makemon.c:1356-1368 — shapechanger detection
    via pm_to_cham(); these monsters call newcham() which consumes 5-15 extra
    RNG draws and sets allow_minvent=FALSE (skipping m_initweap/m_initinv).
    """
    mask = int(M2_SHAPESHIFTER) & 0xFFFFFFFF
    flags = [((int(m.flags2) & 0xFFFFFFFF) & mask) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_golem() -> jnp.ndarray:
    """True where monster's mlet == S_GOLEM.

    Cite: vendor/nethack/include/mondata.h:108
        #define is_golem(ptr) ((ptr)->mlet == S_GOLEM)
    Used in vendor/nethack/src/makemon.c::newmonhp line 1018.
    """
    S_GOLEM_val = int(MonsterSymbol.S_GOLEM)
    flags = [int(m.symbol) == S_GOLEM_val for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_rider() -> jnp.ndarray:
    """True for Death, Famine, Pestilence (the Riders).

    Cite: vendor/nethack/include/mondata.h:161-163
        #define is_rider(ptr) ((ptr)==&mons[PM_DEATH] || ...)
    Used in vendor/nethack/src/makemon.c::newmonhp line 1021.
    """
    rider_names = {"Death", "Famine", "Pestilence"}
    flags = [m.name in rider_names for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _find_pm_index(name: str) -> int:
    """Locate a monster entry by name; returns -1 if absent.

    Defined here (rather than at later use sites) so the forward references
    in ``_compute_is_adult_dragon`` and ``_compute_is_demon_worm_eel`` see it.
    """
    for i, m in enumerate(MONSTERS):
        if m.name == name:
            return i
    return -1


def _compute_is_adult_dragon() -> jnp.ndarray:
    """True for adult dragons (mlet==S_DRAGON && mndx >= PM_GRAY_DRAGON).

    Cite: vendor/nethack/src/makemon.c::newmonhp line 1032
        else if (ptr->mlet == S_DRAGON && mndx >= PM_GRAY_DRAGON)
    Adult dragons use d(m_lev,4)+4*m_lev instead of d(m_lev,8).
    Baby dragons (mndx < PM_GRAY_DRAGON) fall through to the normal d(m_lev,8).
    """
    S_DRAGON_val = int(MonsterSymbol.S_DRAGON)
    gray_idx = _find_pm_index("gray dragon")  # PM_GRAY_DRAGON
    flags = [
        (int(m.symbol) == S_DRAGON_val and i >= gray_idx)
        for i, m in enumerate(MONSTERS)
    ]
    return jnp.array(flags, dtype=jnp.bool_)


# Golem fixed-HP lookup: golemhp() vendor/nethack/src/makemon.c:2233-2261.
# Keyed by monster name; monsters not in the table get 0 (unreachable if
# is_golem is True, since every S_GOLEM entry is listed here).
_GOLEM_HP_BY_NAME: dict[str, int] = {
    "straw golem":   20,
    "paper golem":   20,
    "rope golem":    30,
    "leather golem": 40,
    "gold golem":    60,
    "wood golem":    50,
    "flesh golem":   40,
    "clay golem":    70,
    "stone golem":   100,
    "glass golem":   80,
    "iron golem":    120,
}


def _compute_fixed_golem_hp() -> jnp.ndarray:
    """Fixed HP for each monster slot; non-zero only for golem entries.

    Cite: vendor/nethack/src/makemon.c::golemhp (lines 2233-2261).
    """
    vals = [_GOLEM_HP_BY_NAME.get(m.name, 0) for m in MONSTERS]
    return jnp.array(vals, dtype=jnp.int32)


_IS_SHAPESHIFTER: jnp.ndarray = _compute_is_shapeshifter()   # [NUMMONS] bool
_IS_GOLEM:        jnp.ndarray = _compute_is_golem()          # [NUMMONS] bool
_IS_RIDER:        jnp.ndarray = _compute_is_rider()          # [NUMMONS] bool
_IS_ADULT_DRAGON: jnp.ndarray = _compute_is_adult_dragon()   # [NUMMONS] bool
_FIXED_GOLEM_HP:  jnp.ndarray = _compute_fixed_golem_hp()    # [NUMMONS] int32


# ---------------------------------------------------------------------------
# Monster-flag masks for the post-newmonhp draw cascade.
# Cite: vendor/nle/src/makemon.c lines 1226-1386.
# ---------------------------------------------------------------------------

# Entry indices used by the in_mklev sleep gate (makemon.c:1319-1322:
#   is_ndemon(ptr) || mndx == PM_WUMPUS || mndx == PM_LONG_WORM
#                  || mndx == PM_GIANT_EEL).
# Resolved by name (chunk2.py:wumpus, long worm; chunk6.py:giant eel)
# to remain robust against future chunk reordering — the comment in
# populate_level_with_monsters claiming PM_LONG_WORM=118 is a stale
# index (true index is 113 in the current MONSTERS table).
# _find_pm_index defined earlier (above _compute_is_adult_dragon).
_PM_WUMPUS    = _find_pm_index("wumpus")
_PM_LONG_WORM = _find_pm_index("long worm")
_PM_GIANT_EEL = _find_pm_index("giant eel")


def _compute_is_demon_worm_eel() -> jnp.ndarray:
    """True where monster is is_ndemon() OR wumpus OR long worm OR giant eel.

    Cite: vendor/nle/src/makemon.c:1319-1322 (in_mklev msleeping gate).
        is_ndemon(ptr) = is_demon(ptr) && !(M2_LORD|M2_PRINCE)
        is_demon(ptr) = (mflags2 & M2_DEMON) != 0
    """
    M2_DEMON_MASK  = int(M2_DEMON)  & 0xFFFFFFFF
    M2_LORD_MASK   = int(M2_LORD)   & 0xFFFFFFFF
    M2_PRINCE_MASK = int(M2_PRINCE) & 0xFFFFFFFF
    flags = []
    for i, m in enumerate(MONSTERS):
        f2 = int(m.flags2) & 0xFFFFFFFF
        is_ndemon = (f2 & M2_DEMON_MASK) != 0 and (f2 & (M2_LORD_MASK | M2_PRINCE_MASK)) == 0
        is_special = (i == _PM_WUMPUS) or (i == _PM_LONG_WORM) or (i == _PM_GIANT_EEL)
        flags.append(bool(is_ndemon or is_special))
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_longworm() -> jnp.ndarray:
    """True only at the PM_LONG_WORM entry index.

    Cite: vendor/nle/src/makemon.c:1336-1348 (initworm(mtmp, rn2(5))).
    """
    flags = [(i == _PM_LONG_WORM) for i in range(len(MONSTERS))]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_group_s() -> jnp.ndarray:
    """True where monster has G_SGROUP geno flag.

    Cite: vendor/nle/src/makemon.c:1370 (if ((ptr->geno & G_SGROUP) && rn2(2))).
    """
    flags = [(int(m.generation_mask) & G_SGROUP) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_group_l() -> jnp.ndarray:
    """True where monster has G_LGROUP geno flag.

    Cite: vendor/nle/src/makemon.c:1372-1376 (if ptr->geno & G_LGROUP: rn2(3)).
    """
    flags = [(int(m.generation_mask) & G_LGROUP) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_humanoid() -> jnp.ndarray:
    """True where monster's mflags1 has M1_HUMANOID set.

    Cite: vendor/nle/include/mondata.h — humanoid(ptr) checks M1_HUMANOID.
    Used to gate S_ANGEL m_initweap body (vendor makemon.c:327).
    """
    mask = int(M1_HUMANOID) & 0xFFFFFFFF
    flags = [((int(m.flags1) & 0xFFFFFFFF) & mask) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_lord_mask() -> jnp.ndarray:
    """True where monster's mflags2 has M2_LORD set.

    Cite: vendor/nle/include/mondata.h — is_lord(ptr) checks M2_LORD.
    Used to short-circuit S_ANGEL artiname and shield-pick draws at
    vendor makemon.c:332 (`!rn2(20) || is_lord(ptr)`) and :340
    (`!rn2(4) || is_lord(ptr)`).
    """
    mask = int(M2_LORD) & 0xFFFFFFFF
    flags = [((int(m.flags2) & 0xFFFFFFFF) & mask) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_prince_mask() -> jnp.ndarray:
    """True where monster's mflags2 has M2_PRINCE set.

    Cite: vendor/nle/include/mondata.h:136 — is_prince(ptr) checks M2_PRINCE.
    Used in the default-case bias computation:
    bias = is_lord(ptr) + is_prince(ptr) * 2 + extra_nasty(ptr).
    """
    mask = int(M2_PRINCE) & 0xFFFFFFFF
    flags = [((int(m.flags2) & 0xFFFFFFFF) & mask) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_nasty_mask() -> jnp.ndarray:
    """True where monster's mflags2 has M2_NASTY set.

    Cite: vendor/nle/include/mondata.h:127 — extra_nasty(ptr) checks M2_NASTY.
    Used in the default-case bias computation.
    """
    mask = int(M2_NASTY) & 0xFFFFFFFF
    flags = [((int(m.flags2) & 0xFFFFFFFF) & mask) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_strong_mask() -> jnp.ndarray:
    """True where monster's mflags2 has M2_STRONG set.

    Cite: vendor/nle/include/mondata.h:128 — strongmonst(ptr) checks M2_STRONG.
    Used in the default-case to pick BATTLE_AXE / TWO_HANDED_SWORD /
    LONG_SWORD / LUCERN_HAMMER vs ammo m_initthrow + AKLYS.
    """
    mask = int(M2_STRONG) & 0xFFFFFFFF
    flags = [((int(m.flags2) & 0xFFFFFFFF) & mask) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_demon_mask() -> jnp.ndarray:
    """True where monster's mflags2 has M2_DEMON set.

    Cite: vendor/nle/include/mondata.h:117 — is_demon(ptr) checks M2_DEMON.
    Used to gate the S_DEMON FALLTHRU to default case at vendor
    makemon.c:509-511 (`if (!is_demon(ptr)) break; /*FALLTHRU*/`).
    """
    mask = int(M2_DEMON) & 0xFFFFFFFF
    flags = [((int(m.flags2) & 0xFFFFFFFF) & mask) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_bias_table() -> jnp.ndarray:
    """Per-monster int32 bias for default-case `rnd(14 - 2*bias)`.

    Cite: vendor/nle/src/makemon.c:518:
        bias = is_lord(ptr) + is_prince(ptr) * 2 + extra_nasty(ptr);

    Range: 0 .. 4 (0+0+0=0 for ordinary monsters; 1+2+1=4 for the
    rare M2_LORD|M2_PRINCE|M2_NASTY monsters — but is_lord and is_prince
    are mutually exclusive in practice, so max is 2+1=3).
    """
    M2_LORD_M   = int(M2_LORD)   & 0xFFFFFFFF
    M2_PRINCE_M = int(M2_PRINCE) & 0xFFFFFFFF
    M2_NASTY_M  = int(M2_NASTY)  & 0xFFFFFFFF
    vals = []
    for m in MONSTERS:
        f2 = int(m.flags2) & 0xFFFFFFFF
        lord = 1 if (f2 & M2_LORD_M) != 0 else 0
        prince = 1 if (f2 & M2_PRINCE_M) != 0 else 0
        nasty = 1 if (f2 & M2_NASTY_M) != 0 else 0
        vals.append(lord + prince * 2 + nasty)
    return jnp.array(vals, dtype=jnp.int32)


def _compute_default_case_eligible() -> jnp.ndarray:
    """True where m_initweap reaches the default case.

    Vendor makemon.c:512-553: the default case is hit when the explicit
    mlet switch hasn't `break`'d.  The explicit cases are:
        S_GIANT, S_HUMAN, S_ANGEL, S_HUMANOID, S_KOP, S_ORC, S_OGRE,
        S_TROLL, S_KOBOLD, S_CENTAUR, S_WRAITH, S_ZOMBIE, S_LIZARD,
        S_DEMON (with FALLTHRU to default when is_demon).
    All other armed monsters fall directly into default.  S_DEMON with
    is_demon also falls through after running its inner switch.
    """
    EXPLICIT_MLETS = {
        int(MonsterSymbol.S_GIANT),
        int(MonsterSymbol.S_HUMAN),
        int(MonsterSymbol.S_ANGEL),
        int(MonsterSymbol.S_HUMANOID),
        int(MonsterSymbol.S_KOP),
        int(MonsterSymbol.S_ORC),
        int(MonsterSymbol.S_OGRE),
        int(MonsterSymbol.S_TROLL),
        int(MonsterSymbol.S_KOBOLD),
        int(MonsterSymbol.S_CENTAUR),
        int(MonsterSymbol.S_WRAITH),
        int(MonsterSymbol.S_ZOMBIE),
        int(MonsterSymbol.S_LIZARD),
        int(MonsterSymbol.S_DEMON),  # FALLTHRU when is_demon; handled below
    }
    M2_DEMON_M = int(M2_DEMON) & 0xFFFFFFFF
    flags = []
    for m in MONSTERS:
        mlet = int(m.symbol)
        f2 = int(m.flags2) & 0xFFFFFFFF
        if mlet not in EXPLICIT_MLETS:
            flags.append(True)
            continue
        # S_DEMON falls through to default ONLY when is_demon(ptr).
        if mlet == int(MonsterSymbol.S_DEMON):
            flags.append((f2 & M2_DEMON_M) != 0)
            continue
        # Other explicit mlets break out before default — never fall through.
        flags.append(False)
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_domestic() -> jnp.ndarray:
    """True where monster has M2_DOMESTIC flag2.

    Cite: vendor/nle/include/mondata.h:116 (is_domestic).  Drives the
    saddle check at vendor/nle/src/makemon.c:1386 ``!rn2(100) && is_domestic``.
    """
    mask = int(M2_DOMESTIC) & 0xFFFFFFFF
    flags = [((int(m.flags2) & 0xFFFFFFFF) & mask) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_peace_minded_needs_rn2() -> jnp.ndarray:
    """True where peace_minded(ptr) reaches the trailing rn2 branches.

    Cite: vendor/nle/src/makemon.c:2003-2042 peace_minded.
    Vendor draws rn2(16 + record) and rn2(2 + abs(mal)) only when:
      * NOT always_peaceful (M2_PEACEFUL)
      * NOT always_hostile  (M2_HOSTILE)
      * msound NOT in (MS_LEADER, MS_GUARDIAN, MS_NEMESIS)
      * NOT race_peaceful / race_hostile (skipped here — race masks vary
        by chosen role; record=0 game-start co-aligned default is the
        common case which we model with co-aligned + non-minion)
      * sgn(mal) == sgn(ual)   — co-aligned check (call-site gated)
      * NOT (mal < 0 && have_amulet)   — game-start: no amulet
      * NOT is_minion(ptr)
    Per-monster contribution here covers the species-level filters; the
    co-aligned check is applied at consume time using the player's
    alignment + the precomputed maligntyp.
    """
    M2_PEACE   = int(M2_PEACEFUL) & 0xFFFFFFFF
    M2_HOST    = int(M2_HOSTILE)  & 0xFFFFFFFF
    M2_MIN     = int(M2_MINION)   & 0xFFFFFFFF
    flags = []
    for m in MONSTERS:
        f2  = int(m.flags2) & 0xFFFFFFFF
        snd = int(m.sound)
        if (f2 & M2_PEACE) != 0:
            flags.append(False); continue
        if (f2 & M2_HOST) != 0:
            flags.append(False); continue
        if snd in (int(MS_LEADER), int(MS_GUARDIAN), int(MS_NEMESIS)):
            flags.append(False); continue
        if (f2 & M2_MIN) != 0:
            flags.append(False); continue
        flags.append(True)
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_monster_maligntyp() -> jnp.ndarray:
    """Per-monster ``permonst.maligntyp`` (signed alignment magnitude).

    Cite: vendor/nle/src/makemon.c:2006 (mal = ptr->maligntyp).  Used by
    the peace_minded rn2(2 + abs(mal)) draw at makemon.c:2041.
    """
    vals = [int(getattr(m, "alignment", 0)) for m in MONSTERS]
    return jnp.array(vals, dtype=jnp.int32)


_IS_DEMON_WORM_EEL: jnp.ndarray = _compute_is_demon_worm_eel()   # [NUMMONS] bool
_IS_LONGWORM:       jnp.ndarray = _compute_is_longworm()         # [NUMMONS] bool
_IS_GROUP_S:        jnp.ndarray = _compute_is_group_s()          # [NUMMONS] bool
_IS_GROUP_L:        jnp.ndarray = _compute_is_group_l()          # [NUMMONS] bool
_IS_DOMESTIC:       jnp.ndarray = _compute_is_domestic()         # [NUMMONS] bool
_PEACE_MINDED_RN2_NEEDED: jnp.ndarray = _compute_peace_minded_needs_rn2()  # [NUMMONS] bool
_MONSTER_MALIGNTYP: jnp.ndarray = _compute_monster_maligntyp()   # [NUMMONS] int32


# ---------------------------------------------------------------------------
# m_initinv per-class body masks — vendor/nle/src/makemon.c:589-788
# ---------------------------------------------------------------------------
# Each array gates the RNG draws in the switch(ptr->mlet) body of m_initinv.
# Only classes that contain any rn2() call are listed; all others fall to the
# default: break branch (0 draws).

def _compute_mlet_mask(symbol: MonsterSymbol) -> jnp.ndarray:
    """Bool[NUMMONS]: True where MONSTERS[i].symbol == symbol."""
    flags = [m.symbol == symbol for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_mlet_human_merc() -> jnp.ndarray:
    """S_HUMAN monsters with M2_MERC flag (guard/soldier/watchman).

    Cite: vendor/nle/src/makemon.c:591 ``is_mercenary(ptr)``
          vendor/nle/include/mondata.h:118 ``M2_MERC``
    """
    mask = int(M2_MERC) & 0xFFFFFFFF
    flags = [
        m.symbol == MonsterSymbol.S_HUMAN and bool(int(m.flags2) & mask)
        for m in MONSTERS
    ]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_mlet_human_shopkeeper() -> jnp.ndarray:
    """S_HUMAN monsters with MS_SELL sound (shopkeeper).

    Cite: vendor/nle/src/makemon.c:673 ``ptr == &mons[PM_SHOPKEEPER]``
    Shopkeeper is identified by MS_SELL sound in our table.
    """
    flags = [
        m.symbol == MonsterSymbol.S_HUMAN and int(m.sound) == int(MS_SELL)
        for m in MONSTERS
    ]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_mlet_human_priest() -> jnp.ndarray:
    """S_HUMAN monsters with MS_PRIEST sound (aligned priest / high priest).

    Cite: vendor/nle/src/makemon.c:689 ``ptr->msound == MS_PRIEST``
    """
    flags = [
        m.symbol == MonsterSymbol.S_HUMAN and int(m.sound) == int(MS_PRIEST)
        for m in MONSTERS
    ]
    return jnp.array(flags, dtype=jnp.bool_)


# Per-class masks.  Computed once at import time.
_MLET_NYMPH:       jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_NYMPH)
_MLET_GNOME:       jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_GNOME)
_MLET_MUMMY:       jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_MUMMY)
_MLET_QUANTMECH:   jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_QUANTMECH)
_MLET_LEPRECHAUN:  jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_LEPRECHAUN)
_MLET_DEMON:       jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_DEMON)
_MLET_GIANT:       jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_GIANT)
_MLET_LICH:        jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_LICH)
_MLET_KOBOLD:      jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_KOBOLD)
_MLET_CENTAUR:     jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_CENTAUR)
_MLET_KOP:         jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_KOP)
_MLET_ZOMBIE:      jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_ZOMBIE)
_MLET_TROLL:       jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_TROLL)
_MLET_OGRE:        jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_OGRE)
_MLET_ORC:         jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_ORC)
_MLET_HUMAN_MERC:  jnp.ndarray = _compute_mlet_human_merc()
_MLET_HUMAN_SK:    jnp.ndarray = _compute_mlet_human_shopkeeper()
_MLET_HUMAN_PR:    jnp.ndarray = _compute_mlet_human_priest()
_MLET_ANGEL:       jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_ANGEL)
_MLET_HUMANOID:    jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_HUMANOID)
_IS_HUMANOID:      jnp.ndarray = _compute_is_humanoid()
_IS_LORD:          jnp.ndarray = _compute_is_lord_mask()


# Specific monster entry indices for short-circuit-gated m_initinv draws.
# Vendor uses ``ptr == &mons[PM_X]`` as a C short-circuit AND operand BEFORE
# the rn2 call, so rn2 only fires when the species matches.  Resolve by name
# to remain robust against future chunk reordering.
_PM_ICE_DEVIL   = _find_pm_index("ice devil")
_PM_MASTER_LICH = _find_pm_index("master lich")
_PM_ARCH_LICH   = _find_pm_index("arch-lich")
_PM_MINOTAUR    = _find_pm_index("minotaur")
# Mercenary subtypes — drive the per-subtype m_initinv tail at
# vendor/nle/src/makemon.c:653-672.  WATCHMAN draws rn2(3) whistle;
# soldier/officer types draw rn2(3) k-ration + rn2(2) c-ration + (for
# non-SOLDIER officers) rn2(3) bugle.  WATCH_CAPTAIN and GUARD draw nothing.
_PM_WATCHMAN       = _find_pm_index("watchman")
_PM_WATCH_CAPTAIN  = _find_pm_index("watch captain")
_PM_GUARD          = _find_pm_index("guard")
_PM_SOLDIER        = _find_pm_index("soldier")
# S_OGRE per-subtype rn2 gate divisor — vendor makemon.c:435:
#   if (!rn2(mm == PM_OGRE ? 10 : 5)) (void) mongets(mtmp, WAR_HAMMER);
# PM_OGRE divisor is 10; PM_OGRE_LORD / PM_OGRE_KING (and any other S_OGRE
# entries) use 5.  Build a length-NUMMONS int32 table; non-S_OGRE entries
# get 0 (unused — gated by _MLET_OGRE before lookup).
_PM_OGRE = _find_pm_index("ogre")
# S_ORC per-subtype branches — vendor makemon.c:404-431.  Each subtype
# triggers a distinct chain of mongets/m_initthrow calls (Mordor orc:
# ORCISH_SHORT_SWORD + ORCISH_BOW + ORCISH_ARROW; Uruk-hai: IRON_SKULL_CAP
# + ORCISH_SHORT_SWORD + ORCISH_BOW + ORCISH_ARROW; orc-captain / shaman:
# ORCISH_SHORT_SWORD or SCIMITAR pick).  Subtypes not listed (goblin /
# hobgoblin / orc / hill orc) get only the rn2(2) ORCISH_HELM check.
_PM_MORDOR_ORC   = _find_pm_index("Mordor orc")
_PM_URUK_HAI     = _find_pm_index("Uruk-hai")
_PM_ORC_CAPTAIN  = _find_pm_index("orc-captain")
_PM_ORC_SHAMAN   = _find_pm_index("orc shaman")
# S_LIZARD m_initweap is gated on PM_SALAMANDER specifically (vendor
# makemon.c:483 — `if (mm == PM_SALAMANDER)`); other S_LIZARD entries
# (newt/gecko/iguana/crocodile/lizard/chameleon) lack AT_WEAP and are
# already skipped by the outer `is_armed(ptr)` gate.  See
# _draw_lizard_initweap.
_PM_SALAMANDER   = _find_pm_index("salamander")
# S_HUMANOID m_initweap: PM_HOBBIT takes its own switch(rn2(3)) +
# 2× rn2(10) cloak gates branch (vendor makemon.c:351-366).  Dwarves
# and elven_mithril_coat path remain on the generic placeholder for
# now (separate fix).
_PM_HOBBIT       = _find_pm_index("hobbit")
# S_DEMON per-PM inner switch (vendor makemon.c:487-505) — each named
# PM has its own mongets cascade.  Other S_DEMONs (succubus, imp,
# lemure, manes, vrock, etc.) take the empty inner switch (0 bytes)
# and then FALLTHRU to the default case (vendor makemon.c:509-553);
# the default fallthrough is still on the generic placeholder for now
# (separate fix wired via `_DEFAULT_ELIGIBLE`).
_PM_BALROG       = _find_pm_index("balrog")
_PM_ORCUS        = _find_pm_index("Orcus")
_PM_HORNED_DEVIL = _find_pm_index("horned devil")
_PM_DISPATER     = _find_pm_index("Dispater")
_PM_YEENOGHU     = _find_pm_index("Yeenoghu")


def _compute_ogre_gate_divisor() -> jnp.ndarray:
    """Length-NUMMONS int32: rn2 divisor for the S_OGRE m_initweap gate.

    Cite: vendor/nle/src/makemon.c:435 (S_OGRE branch).
    PM_OGRE → 10; other S_OGRE entries → 5; non-S_OGRE entries → 0.
    """
    vals = []
    for i, m in enumerate(MONSTERS):
        if m.symbol != MonsterSymbol.S_OGRE:
            vals.append(0)
        elif i == _PM_OGRE:
            vals.append(10)
        else:
            vals.append(5)
    return jnp.array(vals, dtype=jnp.int32)


_OGRE_GATE_DIVISOR: jnp.ndarray = _compute_ogre_gate_divisor()


def _compute_is_pm(pm_index: int) -> jnp.ndarray:
    """Bool[NUMMONS]: True only at the given entry index."""
    flags = [(i == pm_index) for i in range(len(MONSTERS))]
    return jnp.array(flags, dtype=jnp.bool_)


_IS_PM_ICE_DEVIL:      jnp.ndarray = _compute_is_pm(_PM_ICE_DEVIL)
_IS_PM_MASTER_LICH:    jnp.ndarray = _compute_is_pm(_PM_MASTER_LICH)
_IS_PM_ARCH_LICH:      jnp.ndarray = _compute_is_pm(_PM_ARCH_LICH)
_IS_PM_MINOTAUR:       jnp.ndarray = _compute_is_pm(_PM_MINOTAUR)
_IS_PM_WATCHMAN:       jnp.ndarray = _compute_is_pm(_PM_WATCHMAN)
_IS_PM_WATCH_CAPTAIN:  jnp.ndarray = _compute_is_pm(_PM_WATCH_CAPTAIN)
_IS_PM_GUARD:          jnp.ndarray = _compute_is_pm(_PM_GUARD)
_IS_PM_SOLDIER:        jnp.ndarray = _compute_is_pm(_PM_SOLDIER)
_PM_GOBLIN       = _find_pm_index("goblin")
_IS_PM_GOBLIN:         jnp.ndarray = _compute_is_pm(_PM_GOBLIN)
_IS_PM_MORDOR_ORC:     jnp.ndarray = _compute_is_pm(_PM_MORDOR_ORC)
_IS_PM_URUK_HAI:       jnp.ndarray = _compute_is_pm(_PM_URUK_HAI)
_IS_PM_ORC_CAPTAIN:    jnp.ndarray = _compute_is_pm(_PM_ORC_CAPTAIN)
_IS_PM_ORC_SHAMAN:     jnp.ndarray = _compute_is_pm(_PM_ORC_SHAMAN)
_IS_PM_SALAMANDER:     jnp.ndarray = _compute_is_pm(_PM_SALAMANDER)
_IS_PM_HOBBIT:         jnp.ndarray = _compute_is_pm(_PM_HOBBIT)
_IS_PM_BALROG:         jnp.ndarray = _compute_is_pm(_PM_BALROG)
_IS_PM_ORCUS:          jnp.ndarray = _compute_is_pm(_PM_ORCUS)
_IS_PM_HORNED_DEVIL:   jnp.ndarray = _compute_is_pm(_PM_HORNED_DEVIL)
_IS_PM_DISPATER:       jnp.ndarray = _compute_is_pm(_PM_DISPATER)
_IS_PM_YEENOGHU:       jnp.ndarray = _compute_is_pm(_PM_YEENOGHU)

# Default-case m_initweap masks — vendor makemon.c:512-553 (default switch).
# bias = is_lord + is_prince*2 + extra_nasty.  strongmonst gates per-case
# weak-vs-strong weapon picks.  _DEFAULT_ELIGIBLE flags monsters that
# reach the default case (non-explicit-mlet OR is_demon S_DEMON fallthrough).
_IS_PRINCE_MASK:       jnp.ndarray = _compute_is_prince_mask()
_IS_NASTY_MASK:        jnp.ndarray = _compute_is_nasty_mask()
_IS_STRONG_MASK:       jnp.ndarray = _compute_is_strong_mask()
_IS_DEMON_FLAG:        jnp.ndarray = _compute_is_demon_mask()
_BIAS_TABLE:           jnp.ndarray = _compute_bias_table()
_DEFAULT_ELIGIBLE:     jnp.ndarray = _compute_default_case_eligible()


# ---------------------------------------------------------------------------
# Spawn-time inventory kits
# ---------------------------------------------------------------------------
# Vendor reference: src/makemon.c::mongets and src/makemon.c lines 180-260
# — per-class initial inventory drawn from monster's M2_* flags + class
# -keyed tables (e.g. weapon for soldiers, wand+scroll for mages, gold for
# shopkeepers).
#
# Implementation: five disjoint kits indexed by ``_MONSTER_KIT_BY_ENTRY``
# (computed at import time from each monster's MS_* sound code and M2_*
# flags2 bits per vendor priority order — see ``_compute_kit_per_entry``
# below).  Each kit fills up to MAX_MONSTER_INV slots with (category,
# type_id, quantity, charges) tuples.  The kit→item mapping below is the
# representative item per class from vendor mongets() (e.g. long sword +
# small shield for MS_SOLDIER, holy water + amulet of reflection for
# MS_PRIEST); vendor m_initweap/m_initinv adds further variance which is
# realised by the depth-keyed random draw in spawn_initial_monsters.

# ---- Item category / type IDs (mirror subsystems/inventory.ItemCategory
# and subsystems/items_{potions,scrolls,wands}.<Effect>) ------------------
_CAT_NONE   = 0
_CAT_WEAPON = 2
_CAT_ARMOR  = 3
_CAT_AMULET = 5
_CAT_POTION = 8
_CAT_SCROLL = 9
_CAT_SPBOOK = 10
_CAT_WAND   = 11
_CAT_COIN   = 12

_POT_HEALING      = 10
_SCR_TELEPORT     = 10
_WAN_FIRE         = 16
_SPBOOK_FORCEBOLT = 0      # placeholder type_id within SPBOOK category
_LONG_SWORD       = 37     # weapon type_id (matches objects.py "long sword")
_SMALL_SHIELD     = 129    # armor type_id
_AMULET_REFLECT   = 0      # amulet type within AMULET category (placeholder)
_HOLY_WATER       = 25     # PotionEffect.WATER — blessed-water variant

# Kit IDs.
_KIT_NONE    = 0
_KIT_MAGE    = 1   # MS_SPELL spellcaster or M2_MAGIC carrier
_KIT_PRIEST  = 2   # MS_PRIEST (aligned priest, high priest)
_KIT_SOLDIER = 3   # MS_SOLDIER (soldier, sergeant, captain, ...)
_KIT_GOLD    = 4   # MS_SELL (shopkeeper) or M2_GREEDY
_KIT_NASTY   = 5   # M2_NASTY (demons / nasty creatures)

# Per-kit inventory rows: each row is MAX_MONSTER_INV (category, type_id,
# quantity, charges) tuples.  Empty slot has category = 0.
def _build_kit_table() -> tuple:
    """Build per-kit fixed inventory.  Returns (cat, tid, qty, chg) tables
    of shape [N_KITS, MAX_MONSTER_INV].
    """
    n_kits = 6  # _KIT_NONE .. _KIT_NASTY
    cat = [[0] * MAX_MONSTER_INV for _ in range(n_kits)]
    tid = [[0] * MAX_MONSTER_INV for _ in range(n_kits)]
    qty = [[0] * MAX_MONSTER_INV for _ in range(n_kits)]
    chg = [[0] * MAX_MONSTER_INV for _ in range(n_kits)]

    def _set(k, slot, c, t, q, ch=0):
        cat[k][slot] = c
        tid[k][slot] = t
        qty[k][slot] = q
        chg[k][slot] = ch

    # Mage kit: wand of fire (5 charges) + scroll of teleport + potion of healing.
    _set(_KIT_MAGE, 0, _CAT_WAND,   _WAN_FIRE,        1, 5)
    _set(_KIT_MAGE, 1, _CAT_SCROLL, _SCR_TELEPORT,    2)
    _set(_KIT_MAGE, 2, _CAT_POTION, _POT_HEALING,     1)
    _set(_KIT_MAGE, 3, _CAT_SPBOOK, _SPBOOK_FORCEBOLT, 1)

    # Priest kit: holy water + spellbook (heal) + amulet of reflection.
    _set(_KIT_PRIEST, 0, _CAT_POTION, _HOLY_WATER,        2)
    _set(_KIT_PRIEST, 1, _CAT_SPBOOK, _SPBOOK_FORCEBOLT,  1)
    _set(_KIT_PRIEST, 2, _CAT_AMULET, _AMULET_REFLECT,    1)
    _set(_KIT_PRIEST, 3, _CAT_POTION, _POT_HEALING,       1)

    # Soldier kit: long sword + small shield.
    _set(_KIT_SOLDIER, 0, _CAT_WEAPON, _LONG_SWORD,   1)
    _set(_KIT_SOLDIER, 1, _CAT_ARMOR,  _SMALL_SHIELD, 1)

    # Shopkeeper / greedy kit: stack of gold.
    _set(_KIT_GOLD, 0, _CAT_COIN, 0, 100)

    # Nasty (demons): a healing potion + scroll teleport for muse use.
    _set(_KIT_NASTY, 0, _CAT_POTION, _POT_HEALING,  1)
    _set(_KIT_NASTY, 1, _CAT_SCROLL, _SCR_TELEPORT, 1)

    return (
        jnp.array(cat, dtype=jnp.int8),
        jnp.array(tid, dtype=jnp.int16),
        jnp.array(qty, dtype=jnp.int16),
        jnp.array(chg, dtype=jnp.int8),
    )


_KIT_CATS, _KIT_TIDS, _KIT_QTYS, _KIT_CHGS = _build_kit_table()


def _compute_kit_per_entry() -> jnp.ndarray:
    """Map each MONSTERS[i] → kit id.  Priority order:
        1. MS_SELL  or  M2_GREEDY      → _KIT_GOLD
        2. MS_PRIEST                   → _KIT_PRIEST
        3. MS_SPELL or  M2_MAGIC       → _KIT_MAGE
        4. MS_SOLDIER                  → _KIT_SOLDIER
        5. M2_NASTY                    → _KIT_NASTY
        6. else                        → _KIT_NONE
    """
    kits = []
    for m in MONSTERS:
        snd  = int(m.sound)
        f2   = int(m.flags2) & 0xFFFFFFFF
        if snd == int(MS_SELL) or (f2 & (int(M2_GREEDY) & 0xFFFFFFFF)):
            kits.append(_KIT_GOLD)
        elif snd == int(MS_PRIEST):
            kits.append(_KIT_PRIEST)
        elif snd == int(MS_SPELL) or (f2 & (int(M2_MAGIC) & 0xFFFFFFFF)):
            kits.append(_KIT_MAGE)
        elif snd == int(MS_SOLDIER):
            kits.append(_KIT_SOLDIER)
        elif f2 & (int(M2_NASTY) & 0xFFFFFFFF):
            kits.append(_KIT_NASTY)
        else:
            kits.append(_KIT_NONE)
    return jnp.array(kits, dtype=jnp.int8)


_MONSTER_KIT_BY_ENTRY: jnp.ndarray = _compute_kit_per_entry()   # [NUMMONS] int8


# ---------------------------------------------------------------------------
# Peace-minded table — precomputed [NUMMONS, 3] at module load.
# Cite: vendor/nethack/src/makemon.c::peace_minded.
#
# Vendor formula (makemon.c::peace_minded ~lines 2003-2042):
#   * always_peaceful(ptr) → True
#   * always_hostile(ptr)  → False
#   * sgn(mon.maligntyp) != sgn(player.alignment) → False
#   * else → True if player's align-record >= 0 (we collapse this to the
#           same-sign branch since at spawn the record is initialised to 0).
#
# Player alignment encoding (vendor align.h):
#   -1 = chaotic, 0 = neutral, +1 = lawful.
# We store the table as a [N_MONSTERS, 3] bool array indexed by
# (entry_idx, align_bucket) where align_bucket = align + 1 (0=chaotic,
# 1=neutral, 2=lawful).
# ---------------------------------------------------------------------------

def _compute_peace_minded_table() -> jnp.ndarray:
    """Build [NUMMONS, 3] bool table per vendor makemon.c::peace_minded."""
    from Nethax.nethax.constants.monsters import (
        MONSTERS, M2_PEACEFUL, M2_HOSTILE,
    )
    n = len(MONSTERS)
    out = [[False, False, False] for _ in range(n)]
    for i, m in enumerate(MONSTERS):
        f2 = int(m.flags2) & 0xFFFFFFFF
        always_peace = bool(f2 & (int(M2_PEACEFUL) & 0xFFFFFFFF))
        always_host  = bool(f2 & (int(M2_HOSTILE)  & 0xFFFFFFFF))
        mal = int(getattr(m, "alignment", 0))
        for align in (-1, 0, 1):
            bucket = align + 1
            if always_peace:
                out[i][bucket] = True
                continue
            if always_host:
                out[i][bucket] = False
                continue
            # Same-sign alignment ⇒ peaceful candidate; vendor uses the
            # alignment record at align==0 (record default ≥ 0 ⇒ True).
            if (mal > 0) != (align > 0):
                out[i][bucket] = False
                continue
            if (mal < 0) != (align < 0):
                out[i][bucket] = False
                continue
            out[i][bucket] = True
    return jnp.array(out, dtype=jnp.bool_)


_PEACE_MINDED_TABLE: jnp.ndarray = _compute_peace_minded_table()   # [NUMMONS, 3] bool


# ---------------------------------------------------------------------------
# Eligible-monster mask
# ---------------------------------------------------------------------------

def eligible_monsters_for_depth(depth: int, genocided=None, ulevel: int = 1) -> jnp.ndarray:
    """Return a bool mask [NUMMONS] of monsters that can spawn at ``depth``.

    Eligibility criteria (mirrors vendor makemon.c::pm_gen / rndmonst()):
        mon.gen_freq > 0
        AND mon.diff_lvl <= depth + 5     (vendor depth-cap; lower-bound is
                                            the dynamically-rolled "zlevel
                                            window", which on average opens
                                            at depth - 6)
        AND NOT G_NOGEN
        AND NOT G_UNIQ                    (unique placement is handled
                                            separately by m_initweap /
                                            place_special)
        AND NOT genocided_species[i]      (vendor/nethack/src/read.c:2826-3015
                                            do_genocide — genocided species
                                            never re-spawn)
    Citation: vendor/nle/src/makemon.c lines 1185-1244 (rndmonst -- where
    ``mons[i].difficulty > zlevel + 4`` rejects the entry); also
    vendor/nle/src/makemon.c::pm_gen (gen_freq weighting).

    The optional ``genocided`` argument is a bool[NUMMONS] mask
    (state.genocided_species) — entries True are filtered out.

    Note: G_HELL / G_NOHELL filtering (vendor makemon.c lines 1690, 1935,
    1998) is applied by the caller via an additional ``in_hell`` mask
    before invoking this helper — keeping it out here lets the function
    stay branch-only on the level-depth-derived window so it can be
    invoked from JITted call sites without an extra Inhell scalar.

    Level band (vendor makemon.c:1546-1560)::

        zlevel  = level_difficulty();            // == depth in the main dungeon
        minmlev = zlevel / 6;
        maxmlev = (zlevel + u.ulevel) / 2;
        if (tooweak(mndx, minmlev) || toostrong(mndx, maxmlev)) continue;

    where ``tooweak``/``toostrong`` reject ``difficulty < minmlev`` /
    ``difficulty > maxmlev``.  ``ulevel`` is the hero experience level (1 at
    game start, which is when mklev runs).  Cite: vendor/nle/src/makemon.c:
    1546-1560; vendor/nle/src/makemon.c:30-31 (toostrong/tooweak).
    """
    zlevel = depth
    minmlev = jnp.int32(zlevel // 6)
    maxmlev = jnp.int32((zlevel + ulevel) // 2)
    in_window = (MONSTR_DIFFICULTIES >= minmlev) & (MONSTR_DIFFICULTIES <= maxmlev)
    # mon.gen_freq > 0 -- entries with a zero generation frequency are
    # never produced by rndmonst (vendor makemon.c pm_gen weighting).
    has_freq = _GEN_FREQS > jnp.int32(0)
    eligible = in_window & has_freq & ~_IS_NOGEN & ~_IS_UNIQ
    if genocided is not None:
        eligible = eligible & ~genocided.astype(jnp.bool_)
    return eligible


# ---------------------------------------------------------------------------
# Pick one monster type for a given depth
# ---------------------------------------------------------------------------

def pick_monster_for_level(rng: jax.Array, depth: int,
                           genocided=None, vendor_rng=None):
    """Sample one monster type index (int32) for the given dungeon depth.

    Vendor reference: ``makemon.c::rndmonst()`` / ``pm_gen()``.  Weights are
    the monster's ``gen_freq`` (vendor: low byte of ``permonst.geno`` -- the
    set_mons_freq value populated by monst.c::G_FREQ).  Eligibility filters
    out G_NOGEN/G_UNIQ entries plus those whose ``diff_lvl > depth + 5``,
    and any species marked genocided in ``state.genocided_species``.

    Returns a scalar jnp.int32 in [0, NUMMONS).

    Byte-replay path: when ``vendor_rng`` (Isaac64State) is supplied, the
    monster is picked by vendor ``rndmonst`` exactly: weight each eligible
    monster by ``geno & G_FREQ`` (align_shift is 0 in the main dungeon,
    whose alignment is AM_NONE), sum to ``choice_count``, draw
    ``ct = rnd(choice_count)`` and walk the cumulative weights
    (:func:`vendor_rng.isaac_rndmonst_choice`).  In that case the return is
    ``(new_vendor_rng, monster_idx)``.  When ``vendor_rng is None`` the
    existing Threefry path is preserved and only ``monster_idx`` is
    returned.

    Cite: vendor/nle/src/makemon.c:1570 (ct = geno & G_FREQ + align_shift);
          vendor/nle/src/makemon.c:1591-1594 (rnd(choice_count) walk).
    """
    mask = eligible_monsters_for_depth(depth, genocided=genocided)
    weights = jnp.where(mask, _GEN_FREQS, jnp.int32(0)).astype(jnp.int32)
    # Guard: if all weights zero (very unusual depth), fall back to uniform over eligible.
    total = jnp.sum(weights)
    weights = jnp.where(total > 0, weights, mask.astype(jnp.int32))

    if vendor_rng is not None:
        new_vrng, idx = isaac_rndmonst_choice(vendor_rng, weights)
        return new_vrng, idx.astype(jnp.int32)

    probs = weights.astype(jnp.float32) / jnp.sum(weights).astype(jnp.float32)
    return jax.random.choice(rng, NUMMONS, p=probs).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Peaceful / hostile classification (vendor makemon.c::peace_minded)
# ---------------------------------------------------------------------------

def peace_minded(type_id: int, player_alignment: int, player_align_record: int) -> bool:
    """Return True if a freshly-spawned monster of ``type_id`` should be
    peaceful to the player at this alignment state.

    Vendor formula (vendor/nle/src/makemon.c::peace_minded lines 2003-2042):
      * always_peaceful(ptr) → True
      * always_hostile(ptr)  → False
      * sgn(monster.maligntyp) != sgn(player.alignment) → False
      * else: chance based on u.ualign.record (peaceful when record > 0).

    Implementation: reads ``MONSTERS[type_id].maligntyp`` for the monster
    side (vendor field permonst.maligntyp, mirrored on PermonstEntry as
    ``alignment``: negative = chaotic, 0 = neutral, positive = lawful)
    and compares to the supplied player alignment/record.  Same-sign
    alignment ⇒ peaceful candidate; differing-sign ⇒ hostile; ties broken
    by the player's u.ualign.record being non-negative — byte-equal to
    vendor makemon.c::peace_minded.  Returns a Python bool; used at
    level-construction time only (non-JIT path).
    """
    from Nethax.nethax.constants.monsters import MONSTERS

    if int(type_id) < 0 or int(type_id) >= len(MONSTERS):
        return False
    m = MONSTERS[int(type_id)]
    # Vendor field is permonst.maligntyp; our PermonstEntry mirrors it as
    # ``alignment`` (negative = chaotic, 0 = neutral, positive = lawful).
    mal = int(getattr(m, "alignment", 0))
    ual = int(player_alignment)
    # Same-sign alignment → peaceful candidate; differing → hostile.
    if (mal > 0) != (ual > 0):
        return False
    if (mal < 0) != (ual < 0):
        return False
    # If alignment record is non-negative, give the benefit of the doubt.
    return int(player_align_record) >= 0


# ---------------------------------------------------------------------------
# Monster HP roll  (vendor makemon.c::newmonhp -- d(hd, 8))
# ---------------------------------------------------------------------------

def roll_monster_hp(rng: jax.Array, hit_dice: int, vendor_rng=None):
    """Roll ``d(hd, 8)`` HP for a newly-created monster.

    Vendor reference: ``makemon.c::newmonhp`` -- "mon->mhp = mon->mhpmax =
    d((int) mon->m_lev, 8);" (1d8 per hit die, sum, min 1).

    This is a public scalar helper for tests; the in-graph spawning code
    uses ``_roll_hp`` which is the same formula but lax.scan-compiled.

    Byte-replay path: when ``vendor_rng`` (Isaac64State) is supplied,
    returns ``(new_vendor_rng, hp)`` so the caller can thread state.
    """
    return _roll_hp(rng, jnp.int32(hit_dice), vendor_rng=vendor_rng)


# ---------------------------------------------------------------------------
# Roll initial HP for a monster (makemon.c::newmonhp lines 1012-1054)
# ---------------------------------------------------------------------------

def _roll_hp(rng: jax.Array, level: jnp.ndarray, vendor_rng=None,
             type_id=None):
    """Roll initial HP for a freshly-generated monster — byte-equal vendor.

    Vendor: vendor/nethack/src/makemon.c::newmonhp (lines 1012-1054).

    Dispatch order (mirrors vendor):
      1. is_golem  → fixed HP from _FIXED_GOLEM_HP table; zero RNG draws.
         Cite: makemon.c:1018-1020  ``mon->mhpmax = golemhp(mndx)``
      2. is_rider  → d(10, 8); basehp=10, 10 dice of d8.
         Cite: makemon.c:1021-1024  ``basehp=10; d(basehp, 8)``
      3. adult dragon (S_DRAGON && mndx>=PM_GRAY_DRAGON, non-endgame):
         4*m_lev + d(m_lev, 4).
         Cite: makemon.c:1032-1036  ``4*basehp + d(basehp, 4)``
      4. m_lev==0 → rnd(4) (1..4).
         Cite: makemon.c:1037-1039
      5. else     → d(m_lev, 8).
         Cite: makemon.c:1040-1044

    ``type_id`` (optional int32 scalar): when supplied, paths 1-3 are
    active via the precomputed _IS_GOLEM/_IS_RIDER/_IS_ADULT_DRAGON masks
    and _FIXED_GOLEM_HP.  When None, only the d(m_lev,8)/rnd(4) path fires
    (backward-compatible for callers that don't pass type_id).

    Byte-replay path: when ``vendor_rng`` (Isaac64State) is supplied, all
    draws are routed through :func:`vendor_rng.randint_jax`.
    Returns ``(new_vendor_rng, hp)`` when vendor_rng is not None.
    """
    MAX_LEVEL = 20  # static cap; no monster generated by mklev exceeds this
    level_i32 = jnp.clip(level.astype(jnp.int32), 0, MAX_LEVEL)

    # ---- Golem / Rider / adult-dragon flags (all False when type_id absent) ---
    if type_id is not None:
        tid = type_id.astype(jnp.int32)
        is_golem_flag   = _IS_GOLEM[tid]
        is_rider_flag   = _IS_RIDER[tid]
        is_adragon_flag = _IS_ADULT_DRAGON[tid]
        golem_fixed_hp  = _FIXED_GOLEM_HP[tid]
    else:
        is_golem_flag   = jnp.bool_(False)
        is_rider_flag   = jnp.bool_(False)
        is_adragon_flag = jnp.bool_(False)
        golem_fixed_hp  = jnp.int32(0)

    if vendor_rng is not None:
        # ---- Byte-replay path: consume ISAAC64 ----
        #
        # CRITICAL: vendor ``d(n, x)`` (rnd.c:208-224) loops ``tmp += RND(x)``
        # n times, calling RND() directly — it does NOT route through rn2()/
        # rnd(), so the ``g_rnd_call_counter`` trace records NO line for each
        # die.  Only the level-0 ``rnd(4)`` path (makemon.c:1038) emits a
        # trace op.  We therefore roll the d(n,x) dice with
        # :func:`next_uint64_jax` (advances the ISAAC stream WITHOUT emitting
        # a trace op) and consume EXACTLY n draws via lax.fori_loop (dynamic
        # bound), so the stream stays byte-aligned while the trace stays
        # line-aligned with vendor.
        #
        # Golem: fixed HP, zero draws.
        # Rider: d(10, 8) — 10 untraced d8 draws.
        # Adult dragon: 4*m_lev + d(m_lev, 4) — m_lev untraced d4 draws.
        # Normal (m_lev>0): d(m_lev, 8) — m_lev untraced d8 draws.
        # Normal (m_lev==0): rnd(4) — a single traced "rnd" draw.
        # Cite: vendor/nle/src/rnd.c:208-224 (d); makemon.c:1037-1044.

        def _roll_dn(vrng, n, sides):
            """Vendor d(n, sides): sum of n untraced RND(sides) draws."""
            def _body(_i, carry):
                v, total = carry
                v, word = next_uint64_jax(v)        # RND(sides): no trace op
                roll = (word % jnp.uint64(sides)).astype(jnp.int32) + jnp.int32(1)
                return (v, total + roll)
            v_out, total = jax.lax.fori_loop(
                0, n, _body, (vrng, jnp.int32(0)),
            )
            return v_out, total

        def _run_rider(vrng):
            v2, total_r = _roll_dn(vrng, jnp.int32(10), jnp.int32(8))
            return v2, jnp.maximum(total_r, jnp.int32(1))

        def _run_dragon(vrng):
            v2, total_d = _roll_dn(vrng, level_i32, jnp.int32(4))
            hp_d = jnp.int32(4) * level_i32 + total_d
            return v2, jnp.maximum(hp_d, jnp.int32(1))

        def _run_normal(vrng):
            # m_lev==0 → single traced rnd(4); m_lev>0 → m_lev untraced d8.
            def _lev0(v):
                v, zero_roll = rnd_jax(v, 4)            # rnd(4): makemon.c:1038
                return v, zero_roll
            def _levN(v):
                v, total_n = _roll_dn(v, level_i32, jnp.int32(8))
                return v, total_n
            v3, hp_n = jax.lax.cond(
                level_i32 == jnp.int32(0), _lev0, _levN, vrng,
            )
            return v3, jnp.maximum(hp_n, jnp.int32(1))

        # Dispatch: golem → no draws; rider → rider scan; dragon → dragon
        # scan; else → normal scan.  jax.lax.cond is JIT-safe.
        vrng_final, rolled_hp = jax.lax.cond(
            is_golem_flag,
            lambda v: (v, golem_fixed_hp),
            lambda v: jax.lax.cond(
                is_rider_flag,
                _run_rider,
                lambda v2: jax.lax.cond(
                    is_adragon_flag,
                    _run_dragon,
                    _run_normal,
                    v2,
                ),
                v,
            ),
            vendor_rng,
        )
        return vrng_final, rolled_hp

    # ---- Threefry path (no vendor_rng) ----
    keys = jax.random.split(rng, MAX_LEVEL + 2)  # +2: zero-roll + dragon d4 tail

    # d(m_lev, 8) scan.
    def _d8_die(carry, args):
        die_idx, key = args
        roll = jax.random.randint(key, (), minval=1, maxval=9, dtype=jnp.int32)
        return carry + jnp.where(die_idx < level_i32, roll, jnp.int32(0)), None

    # d(m_lev, 4) scan for adult dragon.
    def _d4_die(carry, args):
        die_idx, key = args
        roll = jax.random.randint(key, (), minval=1, maxval=5, dtype=jnp.int32)
        return carry + jnp.where(die_idx < level_i32, roll, jnp.int32(0)), None

    d8_total, _ = jax.lax.scan(
        _d8_die, jnp.int32(0),
        (jnp.arange(MAX_LEVEL, dtype=jnp.int32), keys[:MAX_LEVEL]),
    )
    zero_roll = jax.random.randint(keys[MAX_LEVEL], (), minval=1, maxval=5,
                                   dtype=jnp.int32)
    normal_hp = jnp.where(level_i32 == jnp.int32(0), zero_roll, d8_total)
    normal_hp = jnp.maximum(normal_hp, jnp.int32(1))

    if type_id is not None:
        # Rider: d(10,8) — reuse d8 scan with cap=10.
        rider_level = jnp.int32(10)
        rider_cap   = jnp.int32(10)
        def _d8_rider(carry, args):
            die_idx, key = args
            roll = jax.random.randint(key, (), minval=1, maxval=9, dtype=jnp.int32)
            return carry + jnp.where(die_idx < rider_cap, roll, jnp.int32(0)), None
        rider_total, _ = jax.lax.scan(
            _d8_rider, jnp.int32(0),
            (jnp.arange(MAX_LEVEL, dtype=jnp.int32), keys[:MAX_LEVEL]),
        )
        rider_hp = jnp.maximum(rider_total, jnp.int32(1))

        # Dragon: 4*m_lev + d(m_lev, 4).
        d4_total, _ = jax.lax.scan(
            _d4_die, jnp.int32(0),
            (jnp.arange(MAX_LEVEL, dtype=jnp.int32), keys[:MAX_LEVEL]),
        )
        dragon_hp = jnp.maximum(jnp.int32(4) * level_i32 + d4_total, jnp.int32(1))

        hp = jnp.where(is_golem_flag, golem_fixed_hp,
             jnp.where(is_rider_flag, rider_hp,
             jnp.where(is_adragon_flag, dragon_hp,
             normal_hp)))
        return hp

    return normal_hp


# ---------------------------------------------------------------------------
# Pick a valid spawn tile (random FLOOR or CORRIDOR, not on stairs/player)
# ---------------------------------------------------------------------------

def _pick_valid_tile(
    rng: jax.Array,
    valid_tiles_mask: jnp.ndarray,
    map_h: int,
    map_w: int,
) -> jnp.ndarray:
    """Return a (row, col) int16 position sampled uniformly from valid_tiles_mask.

    valid_tiles_mask: bool[map_h, map_w] — True where spawning is allowed.

    Falls back to (0, 0) if the mask is entirely False (should not happen
    on a well-formed level, but guards against JIT shape issues).
    """
    flat_mask = valid_tiles_mask.reshape(-1).astype(jnp.float32)
    total = jnp.sum(flat_mask)
    probs = jnp.where(total > 0, flat_mask / total, jnp.ones(map_h * map_w) / (map_h * map_w))
    flat_idx = jax.random.choice(rng, map_h * map_w, p=probs).astype(jnp.int32)
    row = (flat_idx // map_w).astype(jnp.int16)
    col = (flat_idx % map_w).astype(jnp.int16)
    return jnp.stack([row, col])


# ---------------------------------------------------------------------------
# makemon post-HP RNG cascade — female + m_initinv + m_initweap draws
# ---------------------------------------------------------------------------
#
# Vendor reference: vendor/nle/src/makemon.c::makemon, post-newmonhp.
# After newmonhp() at line 1212, vendor draws (in order):
#
#   1. mtmp->female = rn2(2)            (line 1226, gated by !is_female &&
#                                        !is_male, i.e. effectively
#                                        non-neuter / non-gendered)
#   2. m_initweap(mtmp)                 (line 1382, gated by is_armed(ptr))
#         — variable cascade of rn2() draws per ``ptr->mlet`` branch
#           (lines 182-554); averages ~5-7 draws for armed monsters.
#           Tail: ``if ((int) mtmp->m_lev > rn2(75)) mongets(rnd_offensive_item)``
#           at line 556.
#   3. m_initinv(mtmp)                  (line 1383, unconditional)
#         — class-specific cascade (lines 589-788) followed by THREE
#           unconditional rn2 checks (lines 794, 796, 798):
#               if ((int) mtmp->m_lev > rn2(50))   defensive item
#               if ((int) mtmp->m_lev > rn2(100))  misc item
#               if (likes_gold(ptr) && !findgold && !rn2(5))  gold
#
# We do NOT model item creation (no mksobj()/mongets() effects on EnvState).
# We DO consume the matching ISAAC64 draws so the stream stays byte-aligned
# with vendor NLE for downstream subsystems (env.step, traps, etc.).
#
# JIT shape: lax.while_loop bounded by per-monster _INITWEAP_DRAW_COUNT
# (tighter mlet-specific upper bound vs the previous static cap-8).
# Loop body draws rn2(2) per iteration; the count matches the worst-case
# vendor branch within each mlet.
# ---------------------------------------------------------------------------


def _consume_makemon_post_hp_draws(vrng, type_id,
                                   player_align=0, player_align_record=0,
                                   in_mklev=True, level_difficulty=1,
                                   mm_nogrp=False):
    """Thin dispatcher — see :func:`_consume_makemon_post_hp_draws_body`.

    The real cascade lives in
    :func:`_consume_makemon_post_hp_draws_body`, which is wrapped in a
    module-level ``@jax.jit`` (:func:`_consume_makemon_post_hp_draws_jit`)
    so XLA compiles the 88-closure body ONCE per ``mm_nogrp`` value (two
    variants) and emits a CALL instruction at every site.  Prior to this
    hoist the cascade was inlined into ``fill_one_isaac``'s lax.scan HLO
    module — 137 MB / 411k-line graph, 60+ min cold compile.  Byte-
    equivalent to the prior inlined body.
    """
    # All non-static kwargs are normalised to JAX scalars by the body; the
    # JIT helper sees a stable shape on every call so the cached executable
    # is reused.  ``mm_nogrp`` controls a Python branch and is therefore a
    # static argument of the wrapper.
    return _consume_makemon_post_hp_draws_jit(
        vrng, type_id,
        player_align, player_align_record,
        in_mklev, level_difficulty,
        bool(mm_nogrp),
    )


@functools.partial(jax.jit, static_argnames=("mm_nogrp",))
def _consume_makemon_post_hp_draws_jit(vrng, type_id,
                                       player_align, player_align_record,
                                       in_mklev, level_difficulty,
                                       mm_nogrp):
    """Hoisted ``@jax.jit`` wrapper around the cascade body.  See
    :func:`_consume_makemon_post_hp_draws_body` for the implementation.
    """
    return _consume_makemon_post_hp_draws_body(
        vrng, type_id,
        player_align, player_align_record,
        in_mklev, level_difficulty,
        mm_nogrp,
    )


def _consume_makemon_post_hp_draws_body(vrng, type_id,
                                        player_align=0, player_align_record=0,
                                        in_mklev=True, level_difficulty=1,
                                        mm_nogrp=False):
    """Consume the post-newmonhp RNG cascade for one monster.

    Mirrors vendor/nle/src/makemon.c lines 1214-1386 in draw order:
      * line 1226: ``mtmp->female = rn2(2)`` (non-neuter monsters).
      * line 1236 → peace_minded() (vendor makemon.c:2003-2042): when the
        species reaches the tail (NOT always_peaceful / hostile, NOT
        LEADER/GUARDIAN/NEMESIS, NOT is_minion) AND is co-aligned with
        the player, vendor draws TWO rn2:
            rn2(16 + clamp(record, -15, ∞))
            rn2(2 + abs(maligntyp))
      * line 1321: in_mklev sleep gate for is_ndemon || wumpus ||
        long-worm || giant-eel — single ``rn2(5)``.
      * line 1345: long-worm ``initworm(mtmp, rn2(5))`` — single rn2(5).
      * lines 1370-1376: group rolls.  ``rn2(2)`` for G_SGROUP species;
        for G_LGROUP species ``rn2(3)``.  Vendor first checks SGROUP, and
        the LGROUP branch only fires when SGROUP is absent — so a monster
        consumes ONE of the two (never both).
      * lines 1382 + 182-558: ``m_initweap`` cascade if is_armed(ptr).
        Bounded by per-monster ``_INITWEAP_DRAW_COUNT`` table (per-mlet
        vendor worst-case).  Plus the trailing ``rn2(75)`` at line 556.
      * lines 1383 + 794/796/798: ``m_initinv`` unconditional tail —
        three rn2 draws (rn2(50), rn2(100), rn2(5)).
      * line 1386: saddle check ``!rn2(100) && is_domestic(...)``.  Due
        to C short-circuit, rn2(100) is evaluated for EVERY monster
        (is_domestic is checked only after the rn2(100) draw).

    Args:
        vrng:                Isaac64State.
        type_id:             scalar int32 monster entry index.
        player_align:        int — player's alignment sign (-1/0/+1).
        player_align_record: int — u.ualign.record (game-start: 0).
        in_mklev:            bool — True during level-gen (default).
                             Gates the line-1321 sleep draw.

    Returns:
        new Isaac64State with the appropriate draws consumed.
    """
    tid = type_id.astype(jnp.int32)
    has_fixed_gen  = _HAS_FIXED_GENDER[tid]
    is_armed       = _IS_ARMED[tid]
    is_dwe         = _IS_DEMON_WORM_EEL[tid]
    is_lworm       = _IS_LONGWORM[tid]
    is_sgrp        = _IS_GROUP_S[tid]
    is_lgrp        = _IS_GROUP_L[tid]
    is_dom         = _IS_DOMESTIC[tid]
    likes_gold     = _LIKES_GOLD[tid]
    peace_needs    = _PEACE_MINDED_RN2_NEEDED[tid]
    mal            = _MONSTER_MALIGNTYP[tid]
    # Shapeshifter: vendor makemon.c:1356-1368 — pm_to_cham() != NON_PM
    # → newcham() fires, allow_minvent=FALSE, initweap/initinv skipped.
    is_shapeshifter = _IS_SHAPESHIFTER[tid]

    ual_sgn = jnp.sign(jnp.int32(player_align))
    mal_sgn = jnp.sign(mal)
    co_aligned = ual_sgn == mal_sgn
    # vendor: !!rn2(16 + (record < -15 ? -15 : record))
    record = jnp.maximum(jnp.int32(player_align_record), jnp.int32(-15))
    rn2_arg_a = jnp.maximum(jnp.int32(16) + record, jnp.int32(1))
    # vendor: !!rn2(2 + abs(mal))
    rn2_arg_b = jnp.int32(2) + jnp.abs(mal)

    # --- 1. female draw — vendor makemon.c:1214-1226 ---
    # Vendor draws rn2(2) ONLY when is_female(ptr) AND is_male(ptr) are
    # both false (M2_FEMALE/M2_MALE flags absent).  Quest LEADER/NEMESIS
    # also skip (preset gender), but those never appear in initial-fill
    # spawning so we approximate with the flag2 gate.  Previously gated
    # on M2_NEUTER which suppressed draws for neuters (vendor still
    # draws for neuters — flag means "ignored", not "skipped").
    def _draw_female(v):
        new_v, _ = randint_jax(v, (), 0, 2)
        return new_v

    vrng = jax.lax.cond(has_fixed_gen, lambda v: v, _draw_female, vrng)

    # --- 2. peace_minded co-aligned tail — vendor makemon.c:2039-2041 ---
    #   return (boolean) (!!rn2(16 + ...) && !!rn2(2 + abs(mal)));
    # C short-circuit: the SECOND rn2 fires only when the FIRST is
    # non-zero.  The first returns 0 with probability 1/(16+record).
    peace_fires = peace_needs & co_aligned

    def _do_peace(v):
        v1, r1 = randint_jax(v, (), 0, rn2_arg_a)

        def _draw_peace_second(vv):
            new_v, _ = randint_jax(vv, (), 0, rn2_arg_b)
            return new_v

        return jax.lax.cond(
            r1 != jnp.int32(0), _draw_peace_second, lambda vv: vv, v1
        )

    vrng = jax.lax.cond(peace_fires, _do_peace, lambda v: v, vrng)

    # --- 3. in_mklev sleep gate — vendor makemon.c:1319-1322 ---
    #   if ((is_ndemon(ptr) || PM_WUMPUS || PM_LONG_WORM || PM_GIANT_EEL)
    #        && !u.uhave.amulet && rn2(5))
    # u.uhave.amulet is False during initial spawn; the rn2(5) is the
    # final operand and is evaluated for every species in the predicate.
    def _draw_sleep(v):
        new_v, _ = randint_jax(v, (), 0, 5)
        return new_v

    sleep_fires = jnp.bool_(in_mklev) & is_dwe
    vrng = jax.lax.cond(sleep_fires, _draw_sleep, lambda v: v, vrng)

    # --- 4. long-worm initworm — vendor makemon.c:1345 ---
    #   initworm(mtmp, rn2(5));
    def _draw_lworm(v):
        new_v, _ = randint_jax(v, (), 0, 5)
        return new_v

    vrng = jax.lax.cond(is_lworm, _draw_lworm, lambda v: v, vrng)

    # --- 5. group rolls — vendor makemon.c:1369-1378 ---
    #   if (anymon && !(mmflags & MM_NOGRP)) {
    #       if ((ptr->geno & G_SGROUP) && rn2(2))    → m_initsgrp
    #       else if (ptr->geno & G_LGROUP) { rn2(3) ... }
    #   }
    # The WHOLE block is gated by !(mmflags & MM_NOGRP).  Level-gen OROOM
    # sleeping monsters (mklev.c:816) pass MM_NOGRP, so the group rolls are
    # SKIPPED entirely — no rn2(2)/rn2(3) draw.  Mutually exclusive
    # species-level branches otherwise; consume the matching single draw.
    # Cite: vendor/nle/src/makemon.c:1369-1378; mklev.c:816 (MM_NOGRP).
    grp_allowed = not bool(mm_nogrp)

    def _draw_sgrp(v):
        new_v, _ = randint_jax(v, (), 0, 2)
        return new_v

    def _draw_lgrp(v):
        new_v, _ = randint_jax(v, (), 0, 3)
        return new_v

    if grp_allowed:
        vrng = jax.lax.cond(is_sgrp, _draw_sgrp, lambda v: v, vrng)
        # LGROUP branch only fires when SGROUP is absent (vendor "else if").
        vrng = jax.lax.cond(is_lgrp & ~is_sgrp, _draw_lgrp, lambda v: v, vrng)

    # --- 6/6b/7. initweap + initinv — gated by allow_minvent ---------------
    #
    # Vendor makemon.c:1367-1368: for shapechangers newcham() fires and sets
    # allow_minvent=FALSE, so m_initweap and m_initinv are both SKIPPED.
    # Instead the shapechanger consumes newcham draws (see step 6-newcham).
    #
    # Vendor makemon.c:1441-1444 (allow_minvent guard):
    #   if (allow_minvent) {
    #       if (is_armed(ptr)) m_initweap(mtmp);
    #       m_initinv(mtmp);
    #       ...
    #   }
    #
    # allow_minvent is True for all non-shapechangers.

    def _draw_normal_inv(v):
        """m_initweap + m_initinv + saddle for allow_minvent=TRUE monsters."""

        # --- 6. m_initweap cascade — vendor makemon.c:1382 + lines 182-558 ---
        # Vendor m_initweap is a switch(ptr->mlet) of straight-line
        # conditional rn2() calls.  There is no loop — the previous
        # cap-8 fori_loop drew rn2(2) eight times unconditionally
        # regardless of which mlet branch vendor would take.  Replace
        # with a while_loop bounded by per-monster vendor-tracked worst
        # case (_INITWEAP_DRAW_COUNT) so each species consumes at most
        # its mlet's actual maximum.  Tighter bound than cap-8 for the
        # vast majority of armed monsters (kobold=1, ogre=1, kop=3, …).
        # Cite: makemon.c:182-554 (switch body), 1441-1443 (call site).
        #
        # NOTE: this is still an upper bound — vendor may draw fewer
        # rn2 calls within a branch when inner short-circuits fire.
        # Per-mlet exact short-circuit modelling is a follow-up.
        initweap_n = _INITWEAP_DRAW_COUNT[tid]

        def _draw_initweap(vv):
            def _cond(carry):
                _, idx = carry
                return idx < initweap_n

            def _body(carry):
                vc, idx = carry
                # All m_initweap rn2 calls in vendor use the same shape;
                # mlet branches use varying N (rn2(2), rn2(3), rn2(4),
                # rn2(5), rn2(20), rn2(50)).  We use rn2(2) here as the
                # dominant call (most branches use it).  This is the
                # ISAAC64-byte-count approximation — bytes consumed are
                # the SAME for any rn2(N) since vendor RNG yields a
                # fixed 8-byte word and discards remainder.
                new_v, _r = randint_jax(vc, (), 0, 2)
                return (new_v, idx + jnp.int32(1))

            v_after, _ = jax.lax.while_loop(
                _cond, _body, (vv, jnp.int32(0))
            )

            # S_KOBOLD: explicit gate + m_initthrow draw — vendor makemon.c:456-459
            #   case S_KOBOLD:
            #       if (!rn2(4))
            #           m_initthrow(mtmp, DART, 12);   /* rn1(12, 3) */
            # Vendor draws rn2(4) unconditionally (the gate) and rn1(12, 3)
            # only when the gate result is 0 (1/4 of the time).  The generic
            # loop placeholder couldn't model the gate-conditional rn1 since
            # it draws rn2(2) regardless of the real gate value.  Bypass the
            # loop for kobolds (count=0 in _INITWEAP_DRAW_COUNT) and consume
            # the real rn2(4) here, then conditionally rn1.
            # Cite: vendor/nle/src/makemon.c:456-459 (S_KOBOLD case),
            #       vendor/nle/src/makemon.c:148-160 (m_initthrow rn1).
            def _draw_kobold_initweap(vc):
                vc, gate = randint_jax(vc, (), 0, 4)

                def _draw_initthrow(vd):
                    # m_initthrow(DART, 12) — vendor makemon.c:148-160:
                    #   1. mksobj(DART, TRUE, FALSE) — DART (otyp 7) is
                    #      multigen+poisonable so cascade draws rn1(6,6) +
                    #      rn2(100) = 2 ISAAC64 words.  artif=FALSE skips
                    #      the rn2(20) artifact roll.
                    #   2. otmp->quan = rn1(12, 3) — 1 ISAAC64 word.
                    vd = _weapon_draws(vd, jnp.int32(7), jnp.bool_(False))
                    nv, _ = randint_jax(vd, (), 3, 15)
                    return nv

                return jax.lax.cond(
                    gate == jnp.int32(0),
                    _draw_initthrow,
                    lambda vd: vd,
                    vc,
                )

            v_after = jax.lax.cond(
                _MLET_KOBOLD[tid], _draw_kobold_initweap, lambda vc: vc, v_after
            )

            # S_CENTAUR: explicit gate + m_initthrow draw — vendor makemon.c:461-471
            #   case S_CENTAUR:
            #       if (rn2(2)) {
            #           if (ptr == &mons[PM_FOREST_CENTAUR]) {
            #               (void) mongets(mtmp, BOW);
            #               m_initthrow(mtmp, ARROW, 12);          /* rn1(12, 3) */
            #           } else {
            #               (void) mongets(mtmp, CROSSBOW);
            #               m_initthrow(mtmp, CROSSBOW_BOLT, 12);  /* rn1(12, 3) */
            #           }
            #       }
            #       break;
            # Vendor draws rn2(2) unconditionally (the gate) and rn1(12, 3)
            # for the ARROW or CROSSBOW_BOLT stack only when the gate is
            # NON-zero (1/2 of the time — vendor: ``if (rn2(2))``).  The
            # generic loop placeholder drew a single rn2(2) for centaurs,
            # matching vendor only when the gate failed (1/2 case) and
            # under-drawing by 1 byte when the gate passed (1/2 case).
            # The inner ``ptr == FOREST_CENTAUR`` branch picks ARROW vs.
            # CROSSBOW_BOLT but both call m_initthrow with the same rn1(12, 3),
            # so the ISAAC64 byte cost is identical.
            # Cite: vendor/nle/src/makemon.c:461-471 (S_CENTAUR case),
            #       vendor/nle/src/makemon.c:148-160 (m_initthrow rn1).
            def _draw_centaur_initweap(vc):
                vc, gate = randint_jax(vc, (), 0, 2)

                def _draw_initthrow(vd):
                    # m_initthrow(ARROW/CROSSBOW_BOLT, 12) — vendor
                    # makemon.c:148-160:
                    #   1. mksobj(ARROW/CROSSBOW_BOLT, TRUE, FALSE) — both
                    #      multigen+poisonable so cascade draws rn1(6,6) +
                    #      rn2(100) = 2 ISAAC64 words.  ARROW=1 and
                    #      CROSSBOW_BOLT=6 both fall in vendor is_multigen
                    #      range [-P_SHURIKEN, -P_BOW]; bytes are identical.
                    #   2. otmp->quan = rn1(12, 3) — 1 ISAAC64 word.
                    vd = _weapon_draws(vd, jnp.int32(1), jnp.bool_(False))
                    nv, _ = randint_jax(vd, (), 3, 15)
                    return nv

                return jax.lax.cond(
                    gate != jnp.int32(0),
                    _draw_initthrow,
                    lambda vd: vd,
                    vc,
                )

            v_after = jax.lax.cond(
                _MLET_CENTAUR[tid], _draw_centaur_initweap, lambda vc: vc, v_after
            )

            # S_KOP: explicit two-gate cascade — vendor makemon.c:389-396
            #   case S_KOP:
            #       if (!rn2(4))
            #           m_initthrow(mtmp, CREAM_PIE, 2);   /* rn1(2, 3) */
            #       if (!rn2(3))
            #           (void) mongets(mtmp,
            #                          (rn2(2)) ? CLUB : RUBBER_HOSE);
            #       break;
            # Two independent gates, each with one conditional inner draw:
            #   gate1 = rn2(4): if 0, m_initthrow rn1(2, 3) (1 byte)
            #   gate2 = rn2(3): if 0, weapon-pick rn2(2)   (1 byte)
            # Worst case 4 bytes (both gates 0); typical 2-3 bytes.  The
            # generic loop placeholder drew 3 bytes always, which over-counts
            # by 1 in the 1/2 both-fail case and under-counts by 1 in the
            # 1/12 both-pass case.
            # Cite: vendor/nle/src/makemon.c:389-396 (S_KOP case),
            #       vendor/nle/src/makemon.c:148-160 (m_initthrow rn1).
            def _draw_kop_initweap(vc):
                # Gate 1: rn2(4)
                vc, gate1 = randint_jax(vc, (), 0, 4)

                def _draw_creampie(vd):
                    # m_initthrow(CREAM_PIE, 2) — vendor makemon.c:148-160:
                    #   1. mksobj(CREAM_PIE, TRUE, FALSE) — CREAM_PIE (otyp
                    #      262) is FOOD_CLASS; `_food_draws` draws rn2(6)
                    #      for non-CORPSE/MEAT_RING/KELP_FROND/pudding food
                    #      (1 byte).
                    #   2. otmp->quan = rn1(2, 3) — 1 ISAAC64 word.
                    vd = _food_draws(vd, jnp.int32(262))
                    nv, _ = randint_jax(vd, (), 3, 5)
                    return nv

                vc = jax.lax.cond(
                    gate1 == jnp.int32(0),
                    _draw_creampie,
                    lambda vd: vd,
                    vc,
                )

                # Gate 2: rn2(3)
                vc, gate2 = randint_jax(vc, (), 0, 3)

                def _draw_weapon_pick(vd):
                    # rn2(2) — CLUB vs RUBBER_HOSE pick.
                    nv, _ = randint_jax(vd, (), 0, 2)
                    return nv

                return jax.lax.cond(
                    gate2 == jnp.int32(0),
                    _draw_weapon_pick,
                    lambda vd: vd,
                    vc,
                )

            v_after = jax.lax.cond(
                _MLET_KOP[tid], _draw_kop_initweap, lambda vc: vc, v_after
            )

            # S_ZOMBIE: three independent gates each launching a
            # mongets()/mksobj_init cascade — vendor makemon.c:477-481.
            #   case S_ZOMBIE:
            #       if (!rn2(4)) (void) mongets(mtmp, RING_MAIL);
            #       if (!rn2(4)) (void) mongets(mtmp, LONG_SWORD);
            #       if (!rn2(3)) (void) mongets(mtmp, mlet_zombie_weap[mlev]);
            #       break;
            # Each mongets(otyp) calls mksobj(otyp, init=TRUE, artif=FALSE)
            # which runs the per-class mksobj_init body.
            #   RING_MAIL  (otyp 111) → ARMOR_CLASS cascade (mkobj.c:992-1005).
            #     Not in FUMBLE_BOOTS/LEVITATION_BOOTS/HELM_OF_OPPOSITE_ALIGNMENT/
            #     GAUNTLETS_OF_FUMBLING shortlist, so otyp=0 sentinel is byte-
            #     equivalent.  artif=FALSE skips the rn2(40) artifact check.
            #   LONG_SWORD (otyp 37)  → WEAPON_CLASS cascade (mkobj.c:803-818).
            #     Not is_multigen/is_poisonable so rn1(6,6) + rn2(100) are
            #     skipped; otyp=0 sentinel is byte-equivalent.  artif=FALSE
            #     skips the rn2(20) artifact check.
            #   mlet_zombie_weap[mlev] → WEAPON_CLASS, level-gated weapon
            #     table.  All entries are non-projectile melee weapons
            #     (KNIFE, MACE, SHORT_SWORD, ...), none multigen/poisonable,
            #     so otyp=0 sentinel reproduces the same byte cost.
            # Previously the generic loop drew rn2(2) three times (3 bytes
            # total) which approximated the GATE bytes but missed every
            # per-mongets mksobj_init cascade.  Add the cascades when each
            # gate fires.
            # Cite: vendor/nle/src/makemon.c:477-481 (S_ZOMBIE branch),
            #       vendor/nle/src/mkobj.c:992-1005 (ARMOR_CLASS mksobj_init),
            #       vendor/nle/src/mkobj.c:803-818  (WEAPON_CLASS mksobj_init).
            def _draw_zombie_initweap(vc):
                # Gate 1: rn2(4) — RING_MAIL armor.
                vc, gate1 = randint_jax(vc, (), 0, 4)
                vc = jax.lax.cond(
                    gate1 == jnp.int32(0),
                    lambda vd: _armor_draws(vd, jnp.int32(0), jnp.bool_(False)),
                    lambda vd: vd,
                    vc,
                )
                # Gate 2: rn2(4) — LONG_SWORD weapon.
                vc, gate2 = randint_jax(vc, (), 0, 4)
                vc = jax.lax.cond(
                    gate2 == jnp.int32(0),
                    lambda vd: _weapon_draws(vd, jnp.int32(0), jnp.bool_(False)),
                    lambda vd: vd,
                    vc,
                )
                # Gate 3: rn2(3) — mlet_zombie_weap[mlev] weapon.
                vc, gate3 = randint_jax(vc, (), 0, 3)
                vc = jax.lax.cond(
                    gate3 == jnp.int32(0),
                    lambda vd: _weapon_draws(vd, jnp.int32(0), jnp.bool_(False)),
                    lambda vd: vd,
                    vc,
                )
                return vc

            v_after = jax.lax.cond(
                _MLET_ZOMBIE[tid], _draw_zombie_initweap, lambda vc: vc, v_after
            )

            # S_TROLL: gate rn2(2) gates an inner rn2(4) gem-type pick which
            # picks one of RUBY/DIAMOND/SAPPHIRE/BLACK_OPAL.  Vendor
            # makemon.c:440-454:
            #   case S_TROLL:
            #       if (!rn2(2)) {
            #           switch (rn2(4)) {
            #               case 0: mongets(mtmp, RUBY); break;
            #               case 1: mongets(mtmp, DIAMOND); break;
            #               case 2: mongets(mtmp, SAPPHIRE); break;
            #               case 3: mongets(mtmp, BLACK_OPAL); break;
            #           }
            #       }
            #       break;
            # The outer rn2(2) always fires; the inner rn2(4) and the
            # mongets() mksobj_init cascade only fire when rn2(2) == 0
            # (1/2 probability).  RUBY (413), DIAMOND (412), SAPPHIRE (415),
            # and BLACK_OPAL (416) all take the GEM_CLASS "other gems" path
            # at mkobj.c:892 (rn2(6) — one draw), since none are LOADSTONE
            # (443) or LUCKSTONE (442).  otyp=0 sentinel reproduces the
            # same byte cost.
            # Previously the generic loop drew rn2(2) twice (2 bytes),
            # missing both the gem-type rn2(4) AND the GEM_CLASS mksobj_init
            # rn2(6) when the outer gate fired.
            # Cite: vendor/nle/src/makemon.c:440-454 (S_TROLL branch),
            #       vendor/nle/src/mkobj.c:886-895 (GEM_CLASS mksobj_init).
            def _draw_troll_initweap(vc):
                # Outer gate: rn2(2)
                vc, gate = randint_jax(vc, (), 0, 2)

                def _draw_gem_pick(vd):
                    # Inner rn2(4) picks the gem type (RUBY/DIAMOND/
                    # SAPPHIRE/BLACK_OPAL).  All four take the same
                    # GEM_CLASS draw path so the picked otyp doesn't
                    # affect byte cost.
                    vd, _r4 = randint_jax(vd, (), 0, 4)
                    # GEM_CLASS mksobj_init — rn2(6) one draw for non-
                    # LUCKSTONE/LOADSTONE gems.
                    vd = _gem_draws(vd, jnp.int32(0))
                    return vd

                return jax.lax.cond(
                    gate == jnp.int32(0),
                    _draw_gem_pick,
                    lambda vd: vd,
                    vc,
                )

            v_after = jax.lax.cond(
                _MLET_TROLL[tid], _draw_troll_initweap, lambda vc: vc, v_after
            )

            # S_OGRE: per-subtype rn2(N) gate that conditionally launches a
            # mongets(WAR_HAMMER) mksobj_init cascade.  Vendor makemon.c:
            # 434-438:
            #   case S_OGRE:
            #       if (!rn2(mm == PM_OGRE ? 10 : 5))
            #           (void) mongets(mtmp, WAR_HAMMER);
            #       break;
            # The gate divisor is 10 for PM_OGRE and 5 for PM_OGRE_LORD /
            # PM_OGRE_KING.  When the gate fires, mongets(WAR_HAMMER) runs
            # mksobj(WAR_HAMMER, init=TRUE, artif=FALSE) which dispatches
            # the WEAPON_CLASS mksobj_init cascade (mkobj.c:803-818).
            # WAR_HAMMER (otyp 58) is not is_multigen / is_poisonable so
            # rn1(6,6) + rn2(100) are skipped; otyp=0 sentinel is byte-
            # equivalent.  artif=FALSE skips the rn2(20) artifact check.
            # Previously the generic loop drew rn2(2) once (1 byte) which
            # approximated the gate but missed the WEAPON_CLASS cascade on
            # gate fire and used the wrong divisor (2 vs vendor's 10/5).
            # Cite: vendor/nle/src/makemon.c:434-438 (S_OGRE branch),
            #       vendor/nle/src/mkobj.c:803-818 (WEAPON_CLASS mksobj_init).
            def _draw_ogre_initweap(vc):
                divisor = _OGRE_GATE_DIVISOR[tid]
                vc, gate = randint_jax(vc, (), 0, divisor)
                return jax.lax.cond(
                    gate == jnp.int32(0),
                    lambda vd: _weapon_draws(vd, jnp.int32(0), jnp.bool_(False)),
                    lambda vd: vd,
                    vc,
                )

            v_after = jax.lax.cond(
                _MLET_OGRE[tid], _draw_ogre_initweap, lambda vc: vc, v_after
            )

            # S_ORC: rn2(2) ORCISH_HELM gate + per-PM subtype branches.
            # Vendor makemon.c:397-432:
            #   case S_ORC:
            #       if (rn2(2))
            #           (void) mongets(mtmp, ORCISH_HELM);
            #       if (mm == PM_MORDOR_ORC) {
            #           if (!rn2(3))
            #               (void) mongets(mtmp, ORCISH_SHORT_SWORD);
            #           if (!rn2(2)) {
            #               (void) mongets(mtmp, ORCISH_BOW);
            #               m_initthrow(mtmp, ORCISH_ARROW, 12);
            #           }
            #       } else if (mm == PM_URUK_HAI) {
            #           if (!rn2(3))
            #               (void) mongets(mtmp, IRON_SKULL_CAP);
            #           if (!rn2(3))
            #               (void) mongets(mtmp, ORCISH_SHORT_SWORD);
            #           if (!rn2(3)) {
            #               (void) mongets(mtmp, ORCISH_BOW);
            #               m_initthrow(mtmp, ORCISH_ARROW, 12);
            #           }
            #       } else if (orc_captain || mm == PM_ORC_SHAMAN) {
            #           (void) mongets(mtmp,
            #                          rn2(2) ? ORCISH_SHORT_SWORD : SCIMITAR);
            #       }
            #       break;
            #
            # Byte-cost notes:
            #   ORCISH_HELM (otyp 72) and IRON_SKULL_CAP (#define alias for
            #     ORCISH_HELM at otyp 72): ARMOR_CLASS, not in
            #     FUMBLE_BOOTS/LEVITATION_BOOTS/HELM_OF_OPPOSITE_ALIGNMENT/
            #     GAUNTLETS_OF_FUMBLING shortlist → otyp=0 sentinel byte-
            #     equivalent.  artif=FALSE.
            #   ORCISH_SHORT_SWORD (31), ORCISH_BOW (67), SCIMITAR (33):
            #     WEAPON_CLASS, none is_multigen/is_poisonable → otyp=0
            #     sentinel byte-equivalent.  artif=FALSE.
            #   ORCISH_ARROW (3): WEAPON_CLASS, IS multigen+poisonable so
            #     _weapon_draws MUST receive otyp=3 to draw rn1(6,6) +
            #     rn2(100) on top of the base cascade.  artif=FALSE.
            #
            # m_initthrow(mtmp, ORCISH_ARROW, 12) — vendor makemon.c:148-160:
            #     otmp = mksobj(otyp, TRUE, FALSE);     // mksobj_init draws
            #     otmp->quan = (long) rn1(oquan, 3);    // rn1(12, 3) = 1 draw
            #
            # Subtype gating uses _IS_PM_MORDOR_ORC / _URUK_HAI /
            # _ORC_CAPTAIN / _ORC_SHAMAN.  Subtypes NOT in those four
            # (goblin / hobgoblin / orc / hill orc) get only the rn2(2)
            # ORCISH_HELM check.
            #
            # Previously the generic m_initweap loop drew rn2(2) five times
            # (5 bytes) which over-counted the helm-skip case for orc
            # captain/shaman and under-counted Mordor orc / Uruk-hai (which
            # have up to 4 gate draws + 4 mksobj_init cascades each).
            #
            # Cite: vendor/nle/src/makemon.c:397-432 (S_ORC branch),
            #       vendor/nle/src/makemon.c:148-160 (m_initthrow),
            #       vendor/nle/src/mkobj.c:992-1005 (ARMOR_CLASS mksobj_init),
            #       vendor/nle/src/mkobj.c:803-818  (WEAPON_CLASS mksobj_init).
            def _draw_orc_initweap(vc):
                # rn2(2) ORCISH_HELM gate.  Vendor uses `if (rn2(2))` (NO `!`)
                # — gate fires when result is NON-zero (probability 1/2).
                vc, helm_gate = randint_jax(vc, (), 0, 2)
                vc = jax.lax.cond(
                    helm_gate != jnp.int32(0),
                    lambda vd: _armor_draws(vd, jnp.int32(0), jnp.bool_(False)),
                    lambda vd: vd,
                    vc,
                )

                # Subtype branches (mutually exclusive).
                is_mordor = _IS_PM_MORDOR_ORC[tid]
                is_uruk   = _IS_PM_URUK_HAI[tid]
                is_cap    = _IS_PM_ORC_CAPTAIN[tid]
                is_sham   = _IS_PM_ORC_SHAMAN[tid]

                def _draw_mordor(vd):
                    # rn2(3) → ORCISH_SHORT_SWORD on 0.
                    vd, g1 = randint_jax(vd, (), 0, 3)
                    vd = jax.lax.cond(
                        g1 == jnp.int32(0),
                        lambda ve: _weapon_draws(
                            ve, jnp.int32(0), jnp.bool_(False)),
                        lambda ve: ve,
                        vd,
                    )
                    # rn2(2) → ORCISH_BOW + m_initthrow(ORCISH_ARROW, 12) on 0.
                    vd, g2 = randint_jax(vd, (), 0, 2)

                    def _draw_bow_and_arrows(ve):
                        # ORCISH_BOW mksobj_init (non-multigen).
                        ve = _weapon_draws(ve, jnp.int32(0), jnp.bool_(False))
                        # m_initthrow body for ORCISH_ARROW:
                        #   1. mksobj_init for ORCISH_ARROW (multigen+poisonable
                        #      → otyp=3 to trigger rn1(6,6) + rn2(100)).
                        #   2. rn1(12, 3) = rn2(12) + 3 — 1 draw for quan.
                        ve = _weapon_draws(ve, jnp.int32(3), jnp.bool_(False))
                        ve, _q = randint_jax(ve, (), 3, 15)
                        return ve

                    vd = jax.lax.cond(
                        g2 == jnp.int32(0),
                        _draw_bow_and_arrows,
                        lambda ve: ve,
                        vd,
                    )
                    return vd

                def _draw_uruk(vd):
                    # rn2(3) → IRON_SKULL_CAP (= ORCISH_HELM, ARMOR) on 0.
                    vd, g1 = randint_jax(vd, (), 0, 3)
                    vd = jax.lax.cond(
                        g1 == jnp.int32(0),
                        lambda ve: _armor_draws(
                            ve, jnp.int32(0), jnp.bool_(False)),
                        lambda ve: ve,
                        vd,
                    )
                    # rn2(3) → ORCISH_SHORT_SWORD on 0.
                    vd, g2 = randint_jax(vd, (), 0, 3)
                    vd = jax.lax.cond(
                        g2 == jnp.int32(0),
                        lambda ve: _weapon_draws(
                            ve, jnp.int32(0), jnp.bool_(False)),
                        lambda ve: ve,
                        vd,
                    )
                    # rn2(3) → ORCISH_BOW + m_initthrow(ORCISH_ARROW, 12) on 0.
                    vd, g3 = randint_jax(vd, (), 0, 3)

                    def _draw_bow_and_arrows(ve):
                        ve = _weapon_draws(ve, jnp.int32(0), jnp.bool_(False))
                        ve = _weapon_draws(ve, jnp.int32(3), jnp.bool_(False))
                        ve, _q = randint_jax(ve, (), 3, 15)
                        return ve

                    vd = jax.lax.cond(
                        g3 == jnp.int32(0),
                        _draw_bow_and_arrows,
                        lambda ve: ve,
                        vd,
                    )
                    return vd

                def _draw_cap_or_sham(vd):
                    # rn2(2) weapon-pick (ORCISH_SHORT_SWORD vs SCIMITAR) +
                    # mongets(picked).  Both weapons are non-multigen so the
                    # mksobj_init cascade has identical byte cost regardless
                    # of the picked otyp; otyp=0 sentinel is byte-equivalent.
                    vd, _pick = randint_jax(vd, (), 0, 2)
                    vd = _weapon_draws(vd, jnp.int32(0), jnp.bool_(False))
                    return vd

                # Mutually-exclusive dispatch — vendor uses `if / else if /
                # else if`.  Build a single switch index over four cases:
                #   0 = no subtype branch (goblin / hobgoblin / orc / hill orc)
                #   1 = PM_MORDOR_ORC
                #   2 = PM_URUK_HAI
                #   3 = PM_ORC_CAPTAIN or PM_ORC_SHAMAN
                idx = jnp.where(
                    is_mordor, jnp.int32(1),
                    jnp.where(
                        is_uruk, jnp.int32(2),
                        jnp.where(is_cap | is_sham, jnp.int32(3), jnp.int32(0)),
                    ),
                )
                def _draw_default_orc(vd):
                    # Vendor makemon.c:426-430 — default subtype branch
                    # for PM_GOBLIN / PM_HOBGOBLIN / PM_ORC / PM_HILL_ORC
                    # (PM_ORC_SHAMAN routes via case 3 cap_or_sham above):
                    #   if (mm != PM_ORC_SHAMAN && rn2(2))
                    #     mongets(mtmp,
                    #             (mm == PM_GOBLIN || rn2(2) == 0)
                    #               ? ORCISH_DAGGER : SCIMITAR);
                    is_goblin = _IS_PM_GOBLIN[tid]
                    vd, gate = randint_jax(vd, (), 0, 2)

                    def _gate_hit(ve):
                        def _non_goblin(vf):
                            vf, _pick = randint_jax(vf, (), 0, 2)
                            return vf
                        ve = jax.lax.cond(is_goblin, lambda vf: vf, _non_goblin, ve)
                        # mongets(picked) — both ORCISH_DAGGER (multigen) and
                        # SCIMITAR (non-multigen) consume the same _weapon_draws
                        # cascade with otyp=0 sentinel; the otyp-dependent
                        # is_multigen rn1(6,6) gate is skipped (otyp=0 → False).
                        ve = _weapon_draws(ve, jnp.int32(0), jnp.bool_(False))
                        return ve

                    vd = jax.lax.cond(
                        gate != jnp.int32(0), _gate_hit, lambda ve: ve, vd,
                    )
                    return vd

                vc = jax.lax.switch(
                    idx,
                    [
                        _draw_default_orc,       # 0 — goblin/hobgoblin/orc/hill orc
                        _draw_mordor,            # 1
                        _draw_uruk,              # 2
                        _draw_cap_or_sham,       # 3 (TODO: split shaman→case 0)
                    ],
                    vc,
                )
                return vc

            v_after = jax.lax.cond(
                _MLET_ORC[tid], _draw_orc_initweap, lambda vc: vc, v_after
            )

            # S_LIZARD: PM_SALAMANDER chained-ternary weapon pick —
            # vendor makemon.c:482-486:
            #   case S_LIZARD:
            #       if (mm == PM_SALAMANDER)
            #           (void) mongets(mtmp,
            #                          (rn2(7) ? SPEAR : rn2(3) ? TRIDENT : STILETTO));
            #       break;
            # PM_SALAMANDER is the only AT_WEAP S_LIZARD entry (newt /
            # gecko / iguana / baby crocodile / lizard / chameleon /
            # crocodile all lack AT_WEAP in vendor monst.c), so non-
            # salamander S_LIZARDs are already skipped by the outer
            # is_armed gate.  Gating on _IS_PM_SALAMANDER matches vendor's
            # explicit `mm == PM_SALAMANDER` check.
            #
            # Draws: rn2(7) (1 byte) always; if rn2(7) == 0 then rn2(3)
            # (1 byte) picks TRIDENT vs STILETTO; plus WEAPON_CLASS
            # mongets cascade.  SPEAR (P_SPEAR), TRIDENT (P_TRIDENT),
            # and STILETTO (P_KNIFE) all have positive oc_skill so none
            # fall in vendor's is_multigen range [-P_SHURIKEN, -P_BOW]
            # (obj.h:197-204); otyp=0 sentinel reproduces the same byte
            # cost.  artif=FALSE skips the rn2(20) artifact check.
            # Previously the generic _INITWEAP_DRAW_COUNT[S_LIZARD]=2
            # placeholder loop drew rn2(2) twice for ALL salamander
            # spawns — matching only when the chained ternary fell into
            # the rn2(7)==0 branch (1/7 of the time) and over-drawing
            # by 1 byte in the rn2(7) non-zero branch (6/7 of the time).
            # Cite: vendor/nle/src/makemon.c:482-486 (S_LIZARD branch),
            #       vendor/nle/src/mkobj.c:803-818 (WEAPON_CLASS mksobj_init).
            def _draw_lizard_initweap(vc):
                vc, r7 = randint_jax(vc, (), 0, 7)

                def _trident_or_stiletto(vd):
                    vd, _r3 = randint_jax(vd, (), 0, 3)
                    return vd

                vc = jax.lax.cond(
                    r7 == jnp.int32(0),
                    _trident_or_stiletto,
                    lambda vd: vd,
                    vc,
                )
                return _weapon_draws(vc, jnp.int32(0), jnp.bool_(False))

            v_after = jax.lax.cond(
                _IS_PM_SALAMANDER[tid], _draw_lizard_initweap, lambda vc: vc, v_after
            )

            # S_ANGEL: humanoid-gated cascade — vendor makemon.c:326-348.
            #   case S_ANGEL:
            #       if (humanoid(ptr)) {
            #           otmp = mksobj(LONG_SWORD, FALSE, FALSE);  // 0 draws
            #           if (!rn2(20) || is_lord(ptr))             // 1 byte if !is_lord
            #               otmp = oname(otmp,
            #                            artiname(rn2(2) ? ART_DEMONBANE : ART_SUNSWORD));
            #           bless(otmp);
            #           otmp->oerodeproof = TRUE;
            #           otmp->spe = rn2(4);                       // 1 byte ALWAYS
            #           (void) mpickobj(mtmp, otmp);
            #           otmp = mksobj(!rn2(4) || is_lord(ptr)     // 1 byte if !is_lord
            #                         ? SHIELD_OF_REFLECTION : LARGE_SHIELD, FALSE, FALSE);
            #           ...
            #       }
            #       break;
            # Both mksobj() calls have init=TRUE, artif=FALSE:
            #   LONG_SWORD (otyp 37, WEAPON_CLASS, P_LONG_SWORD positive skill →
            #     NOT in is_multigen range [-P_SHURIKEN, -P_BOW]) → 0 cascade bytes.
            #   SHIELD_OF_REFLECTION / LARGE_SHIELD (ARMOR_CLASS, neither in the
            #     FUMBLE_BOOTS/LEVITATION_BOOTS/HELM_OF_OPPOSITE_ALIGNMENT/
            #     GAUNTLETS_OF_FUMBLING shortlist) → 0 cascade bytes.
            # `!rn2(N) || is_lord(ptr)` short-circuits: when is_lord, NO rn2(N)
            # is drawn (the LHS is never evaluated).  When !is_lord, rn2(N) is
            # drawn unconditionally.
            # Byte budget (per humanoid angel):
            #   non-lord, no artiname (19/20):  1 (rn20) + 1 (spe) + 1 (shield) = 3
            #   non-lord, with artiname (1/20): 1 (rn20) + 1 (art) + 1 (spe) + 1 (shield) = 4
            #   is_lord:                        1 (art)  + 1 (spe) = 2
            # Non-humanoid armed angels: 0 bytes (the whole body is gated).
            # Previously the generic placeholder loop drew rn2(2) thrice (3 bytes)
            # for ALL armed angels including non-humanoid ones (e.g., couatl),
            # and under-drew by 1 byte in the 1/20-of-non-lord with-artiname case.
            # Cite: vendor/nle/src/makemon.c:326-348 (S_ANGEL branch),
            #       vendor/nle/src/mkobj.c:803-818 (WEAPON_CLASS mksobj_init),
            #       vendor/nle/src/mkobj.c:993-1005 (ARMOR_CLASS mksobj_init).
            def _draw_angel_initweap(vc):
                is_lord = _IS_LORD[tid]

                def _humanoid_body(vd):
                    # !rn2(20) || is_lord artiname gate.
                    def _non_lord_artiname(ve):
                        ve, r20 = randint_jax(ve, (), 0, 20)

                        def _draw_artname(vf):
                            vf, _ = randint_jax(vf, (), 0, 2)
                            return vf

                        return jax.lax.cond(
                            r20 == jnp.int32(0),
                            _draw_artname,
                            lambda vf: vf,
                            ve,
                        )

                    def _lord_artiname(ve):
                        # rn2(20) short-circuited; rn2(2) ALWAYS drawn (LHS of || true).
                        ve, _ = randint_jax(ve, (), 0, 2)
                        return ve

                    vd = jax.lax.cond(is_lord, _lord_artiname, _non_lord_artiname, vd)
                    # spe = rn2(4) — ALWAYS
                    vd, _ = randint_jax(vd, (), 0, 4)

                    # !rn2(4) || is_lord shield-type gate.
                    def _shield_pick(ve):
                        ve, _ = randint_jax(ve, (), 0, 4)
                        return ve

                    vd = jax.lax.cond(is_lord, lambda ve: ve, _shield_pick, vd)
                    return vd

                return jax.lax.cond(
                    _IS_HUMANOID[tid], _humanoid_body, lambda vd: vd, vc,
                )

            v_after = jax.lax.cond(
                _MLET_ANGEL[tid], _draw_angel_initweap, lambda vc: vc, v_after
            )

            # S_HUMANOID PM_HOBBIT: switch(rn2(3)) DAGGER/ELVEN_DAGGER/SLING
            # weapon-pick + 2× rn2(10) armor gates — vendor makemon.c:351-366.
            #   case S_HUMANOID:
            #       if (mm == PM_HOBBIT) {
            #           switch (rn2(3)) {
            #               case 0: mongets(DAGGER); break;
            #               case 1: mongets(ELVEN_DAGGER); break;
            #               case 2: mongets(SLING); break;
            #           }
            #           if (!rn2(10)) (void) mongets(mtmp, ELVEN_MITHRIL_COAT);
            #           if (!rn2(10)) (void) mongets(mtmp, DWARVISH_CLOAK);
            #       } else if (is_dwarf(ptr)) { ... (deferred — see _INITWEAP_DRAW_COUNT) }
            # Hobbit byte budget: rn2(3) (1) + rn2(10) (1) + rn2(10) (1) = 3 bytes.
            # Mongets cascade per otyp (all init=TRUE, artif=FALSE):
            #   DAGGER (otyp 10, P_DAGGER skill — positive, not multigen): 0 bytes.
            #   ELVEN_DAGGER (otyp 11, P_DAGGER, not multigen): 0 bytes.
            #   SLING (otyp 73, P_SLING positive, not in is_multigen range): 0 bytes.
            #   ELVEN_MITHRIL_COAT (ARMOR, not in special shortlist): 0 bytes.
            #   DWARVISH_CLOAK (ARMOR, not in special shortlist): 0 bytes.
            # Previously the generic placeholder loop drew rn2(2) six times
            # (6 bytes) for hobbits using the S_HUMANOID dwarf-worst-case slot,
            # over-drawing by 3 bytes per hobbit spawn.  _INITWEAP_DRAW_COUNT
            # now overrides PM_HOBBIT to 0 so the placeholder is bypassed.
            # Cite: vendor/nle/src/makemon.c:351-366 (S_HUMANOID PM_HOBBIT branch),
            #       vendor/nle/include/obj.h:197-204 (is_multigen predicate).
            def _draw_hobbit_initweap(vc):
                # switch(rn2(3)) weapon pick — 1 byte
                vc, _ = randint_jax(vc, (), 0, 3)
                # !rn2(10) ELVEN_MITHRIL_COAT — 1 byte
                vc, _ = randint_jax(vc, (), 0, 10)
                # !rn2(10) DWARVISH_CLOAK — 1 byte
                vc, _ = randint_jax(vc, (), 0, 10)
                return vc

            v_after = jax.lax.cond(
                _IS_PM_HOBBIT[tid], _draw_hobbit_initweap, lambda vc: vc, v_after
            )

            # S_DEMON inner per-PM switch — vendor makemon.c:487-505:
            #   case S_DEMON:
            #       switch (mm) {
            #       case PM_BALROG:
            #           (void) mongets(mtmp, BULLWHIP);
            #           (void) mongets(mtmp, BROADSWORD);
            #           break;
            #       case PM_ORCUS:
            #           (void) mongets(mtmp, WAN_DEATH);
            #           break;
            #       case PM_HORNED_DEVIL:
            #           (void) mongets(mtmp, rn2(4) ? TRIDENT : BULLWHIP);
            #           break;
            #       case PM_DISPATER:
            #           (void) mongets(mtmp, WAN_STRIKING);
            #           break;
            #       case PM_YEENOGHU:
            #           (void) mongets(mtmp, FLAIL);
            #           break;
            #       }
            #       if (!is_demon(ptr)) break;
            #       /*FALLTHRU*/
            # The fallthrough to the default case (vendor makemon.c:512-553)
            # is still on the generic placeholder count=1 for the rnd(14-2*bias)
            # selector — full default-case modeling is a separate fix.
            #
            # Inner-switch byte budgets:
            #   PM_BALROG:       0 bytes (BULLWHIP non-multigen + BROADSWORD non-multigen)
            #   PM_ORCUS:        2-3 bytes (WAN_DEATH via _wand_draws: rn2(5) + blessorcurse(17))
            #   PM_HORNED_DEVIL: 1 byte (rn2(4) ternary) + 0 (TRIDENT/BULLWHIP non-multi)
            #   PM_DISPATER:     2-3 bytes (WAN_STRIKING via _wand_draws)
            #   PM_YEENOGHU:     0 bytes (FLAIL non-multigen)
            # Other S_DEMONs (succubus, imp, lemure, vrock, ice_devil, etc.):
            # 0 inner bytes — no matching case in the inner switch.
            #
            # Previously the placeholder drew rn2(2)=1 byte for ALL S_DEMONs,
            # under-counting ORCUS/DISPATER by 1-2 bytes (missing WAND cascade)
            # and HORNED_DEVIL by 0 bytes (matched), and over-counting other
            # S_DEMONs by 0-1 byte vs vendor's inner+default sum.
            #
            # Cite: vendor/nle/src/makemon.c:487-505 (S_DEMON inner switch),
            #       vendor/nle/src/mkobj.c:803-818 (WEAPON_CLASS mksobj_init),
            #       vendor/nle/src/mkobj.c:1019-1027 (WAND_CLASS mksobj_init).
            def _draw_demon_inner_switch(vc):
                # PM_ORCUS: WAN_DEATH mongets cascade.
                vc = jax.lax.cond(
                    _IS_PM_ORCUS[tid],
                    lambda vd: _wand_draws(vd),
                    lambda vd: vd,
                    vc,
                )
                # PM_HORNED_DEVIL: rn2(4) ternary (TRIDENT vs BULLWHIP).
                vc = jax.lax.cond(
                    _IS_PM_HORNED_DEVIL[tid],
                    lambda vd: randint_jax(vd, (), 0, 4)[0],
                    lambda vd: vd,
                    vc,
                )
                # PM_DISPATER: WAN_STRIKING mongets cascade.
                vc = jax.lax.cond(
                    _IS_PM_DISPATER[tid],
                    lambda vd: _wand_draws(vd),
                    lambda vd: vd,
                    vc,
                )
                # PM_BALROG and PM_YEENOGHU: only non-multigen WEAPONs, 0 bytes.
                # No explicit draws needed.
                return vc

            v_after = jax.lax.cond(
                _MLET_DEMON[tid], _draw_demon_inner_switch, lambda vc: vc, v_after
            )

            # Default-case — vendor makemon.c:512-553.  Fires for:
            #   (a) any armed monster whose mlet is NOT in the explicit case
            #       set {S_GIANT, S_HUMAN, S_ANGEL, S_HUMANOID, S_KOP, S_ORC,
            #       S_OGRE, S_TROLL, S_KOBOLD, S_CENTAUR, S_WRAITH, S_ZOMBIE,
            #       S_LIZARD, S_DEMON};
            #   (b) any S_DEMON monster with is_demon(ptr) == TRUE, via
            #       FALLTHRU from line 511 (`if (!is_demon(ptr)) break;`).
            # The _DEFAULT_ELIGIBLE mask precomputes membership.
            #
            # Vendor body:
            #     bias = is_lord(ptr) + is_prince(ptr)*2 + extra_nasty(ptr);
            #     switch (rnd(14 - 2*bias)) {
            #     case 1:
            #         if (strongmonst(ptr))
            #             (void) mongets(mtmp, BATTLE_AXE);
            #         else
            #             m_initthrow(mtmp, DART, 12);
            #         break;
            #     case 2:
            #         if (strongmonst(ptr))
            #             (void) mongets(mtmp, TWO_HANDED_SWORD);
            #         else {
            #             (void) mongets(mtmp, CROSSBOW);
            #             m_initthrow(mtmp, CROSSBOW_BOLT, 12);
            #         }
            #         break;
            #     case 3:
            #         (void) mongets(mtmp, BOW);
            #         m_initthrow(mtmp, ARROW, 12);
            #         break;
            #     case 4:
            #         if (strongmonst(ptr))
            #             (void) mongets(mtmp, LONG_SWORD);
            #         else
            #             m_initthrow(mtmp, DAGGER, 3);
            #         break;
            #     case 5:
            #         if (strongmonst(ptr))
            #             (void) mongets(mtmp, LUCERN_HAMMER);
            #         else
            #             (void) mongets(mtmp, AKLYS);
            #         break;
            #     default: break;
            #     }
            #
            # Byte costs per case (selector consumes 1 byte ALWAYS):
            #   case 1 strong: BATTLE_AXE non-multigen → 0 bytes
            #   case 1 weak:   DART mksobj (rn1(6,6)+rn2(100)=2) + rn1(12,3)=1 → 3
            #   case 2 strong: TWO_HANDED_SWORD non-multigen → 0 bytes
            #   case 2 weak:   CROSSBOW(0) + CROSSBOW_BOLT mksobj(2) + rn1(12,3)=1 → 3
            #   case 3 always: BOW(0) + ARROW mksobj(2) + rn1(12,3)=1 → 3
            #   case 4 strong: LONG_SWORD non-multigen → 0 bytes
            #   case 4 weak:   DAGGER non-multigen(0) + rn1(3,3)=1 → 1
            #   case 5 strong: LUCERN_HAMMER non-multigen → 0 bytes
            #   case 5 weak:   AKLYS non-multigen → 0 bytes
            #   default (6+): 0 bytes
            #
            # DART (otyp 7), CROSSBOW_BOLT (otyp 6), ARROW (otyp 1) — all in
            # vendor is_multigen range (negative oc_skill in [-P_SHURIKEN,
            # -P_BOW]); _weapon_draws with the right otyp triggers the
            # rn1(6,6) + rn2(100) multigen+poisonable cascade.
            #
            # Vendor m_initthrow order (vendor/nle/src/makemon.c:148-160):
            #   1. mksobj(otyp, TRUE, FALSE) — runs mksobj_init cascade first
            #   2. otmp->quan = rn1(oquan, 3) — 1 byte AFTER cascade
            # Must reproduce that order.
            #
            # Previously the placeholder loop drew rn2(2)=1 byte for ALL
            # default-eligible monsters (most armed Dlvl 1 monsters: rats,
            # dogs, wolves, cats, leopards, etc.).  Vendor average is
            # ~1.21-1.71 bytes depending on strongmonst, so the placeholder
            # under-counted by ~0.2-0.7 bytes per default-eligible spawn.
            #
            # Cite: vendor/nle/src/makemon.c:512-553 (default case),
            #       vendor/nle/src/makemon.c:148-160 (m_initthrow body),
            #       vendor/nle/src/mkobj.c:803-818 (WEAPON_CLASS mksobj_init),
            #       vendor/nle/include/obj.h:197-204 (is_multigen).
            _OTYP_DART          = jnp.int32(7)
            _OTYP_CROSSBOW_BOLT = jnp.int32(6)
            _OTYP_ARROW         = jnp.int32(1)

            def _draw_default_case(vc):
                bias = _BIAS_TABLE[tid]
                # rnd(14 - 2*bias) — vendor selector.  rnd(N) = 1 + rn2(N).
                # Cast to int32 explicitly so JAX picks a stable dtype.
                n = jnp.int32(14) - jnp.int32(2) * bias
                vc, r = rnd_jax(vc, n)
                is_strong = _IS_STRONG_MASK[tid]

                def _case1(vd):
                    def _strong(ve):
                        return ve  # BATTLE_AXE non-multigen → 0 bytes
                    def _weak(ve):
                        # m_initthrow(DART, 12): mksobj cascade FIRST, then rn1.
                        ve = _weapon_draws(ve, _OTYP_DART, jnp.bool_(False))
                        ve, _ = randint_jax(ve, (), 3, 15)  # rn1(12, 3)
                        return ve
                    return jax.lax.cond(is_strong, _strong, _weak, vd)

                def _case2(vd):
                    def _strong(ve):
                        return ve  # TWO_HANDED_SWORD non-multigen → 0 bytes
                    def _weak(ve):
                        # mongets(CROSSBOW) — non-multigen → 0 bytes
                        # m_initthrow(CROSSBOW_BOLT, 12)
                        ve = _weapon_draws(ve, _OTYP_CROSSBOW_BOLT, jnp.bool_(False))
                        ve, _ = randint_jax(ve, (), 3, 15)  # rn1(12, 3)
                        return ve
                    return jax.lax.cond(is_strong, _strong, _weak, vd)

                def _case3(vd):
                    # mongets(BOW) — non-multigen → 0 bytes
                    # m_initthrow(ARROW, 12) — always, no strongmonst branch
                    vd = _weapon_draws(vd, _OTYP_ARROW, jnp.bool_(False))
                    vd, _ = randint_jax(vd, (), 3, 15)  # rn1(12, 3)
                    return vd

                def _case4(vd):
                    def _strong(ve):
                        return ve  # LONG_SWORD non-multigen → 0 bytes
                    def _weak(ve):
                        # m_initthrow(DAGGER, 3) — DAGGER non-multigen (0 bytes)
                        ve, _ = randint_jax(ve, (), 3, 6)  # rn1(3, 3)
                        return ve
                    return jax.lax.cond(is_strong, _strong, _weak, vd)

                def _case5(vd):
                    # LUCERN_HAMMER or AKLYS — both non-multigen → 0 bytes
                    return vd

                def _default_case(vd):
                    return vd  # cases 6..14 → no draw

                # rnd returns 1..N.  Map to switch index [0..5] where
                # index 5 catches all "default" cases (r >= 6).
                idx = jnp.clip(r - jnp.int32(1), 0, 5)
                vc = jax.lax.switch(
                    idx,
                    [_case1, _case2, _case3, _case4, _case5, _default_case],
                    vc,
                )
                return vc

            v_after = jax.lax.cond(
                _DEFAULT_ELIGIBLE[tid], _draw_default_case, lambda vc: vc, v_after
            )

            # Trailing offensive-item check — vendor makemon.c:556
            #   if ((int) mtmp->m_lev > rn2(75)) (void) mongets(...)
            v_after, _ = randint_jax(v_after, (), 0, 75)
            return v_after

        v = jax.lax.cond(is_armed, _draw_initweap, lambda vv: vv, v)

        # --- 6b. m_initinv per-class body — vendor makemon.c:589-788 ---
        # Only classes with rn2() calls listed; others are 0 draws.
        # Cite: makemon.c:1444  m_initinv(mtmp)
        is_gnome     = _MLET_GNOME[tid]
        is_nymph     = _MLET_NYMPH[tid]
        is_mummy     = _MLET_MUMMY[tid]
        is_qmech     = _MLET_QUANTMECH[tid]
        is_lep       = _MLET_LEPRECHAUN[tid]
        is_ice_devil = _IS_PM_ICE_DEVIL[tid]
        is_giant_cls = _MLET_GIANT[tid]
        is_minotaur  = _IS_PM_MINOTAUR[tid]
        is_master_lich = _IS_PM_MASTER_LICH[tid]
        is_arch_lich   = _IS_PM_ARCH_LICH[tid]
        is_hmerc     = _MLET_HUMAN_MERC[tid]
        is_hsk       = _MLET_HUMAN_SK[tid]
        is_hpr       = _MLET_HUMAN_PR[tid]
        # Mercenary subtypes for the post-armor tail (vendor makemon.c:653-672).
        is_pm_watchman      = _IS_PM_WATCHMAN[tid]
        is_pm_watch_captain = _IS_PM_WATCH_CAPTAIN[tid]
        is_pm_guard         = _IS_PM_GUARD[tid]
        is_pm_soldier       = _IS_PM_SOLDIER[tid]

        # S_KOBOLD: vendor m_initinv (makemon.c:589-788) has NO case S_KOBOLD,
        # so kobolds consume ZERO draws inside m_initinv.  The previous
        # `rn2(4)` here mis-attributed vendor m_initweap's kobold gate
        # (vendor makemon.c:457 — `if (!rn2(4)) m_initthrow(mtmp, DART, 12);`)
        # which is ALREADY consumed inside ``_draw_initweap`` above via the
        # ``_INITWEAP_DRAW_COUNT[S_KOBOLD] == 1`` loop iteration.  Drawing
        # it again caused a phantom rn2(4) draw immediately after the
        # trailing rn2(75) — visible in seed=9 around L1405.
        # Vendor cite: vendor/nle/src/makemon.c:576-788 (m_initinv; no S_KOBOLD).

        # S_GNOME: rn2(60) gate + optional rn2(4) candle-type pick.
        # Vendor makemon.c:778-784:
        #   if (!rn2((In_mines(&u.uz) && in_mklev) ? 20 : 60)) {
        #       otmp = mksobj(rn2(4) ? TALLOW_CANDLE : WAX_CANDLE, ...);
        #       ...
        #   }
        # On Dlvl 1 we are NOT In_mines, so the outer divisor is 60.  When
        # the gate passes (rn2(60) == 0, probability 1/60), vendor draws
        # one additional rn2(4) to pick TALLOW vs WAX candle.  Previously
        # only the outer rn2(60) was modelled → under-consumed 1 ISAAC64
        # draw with probability 1/60 per S_GNOME spawn.
        # Cite: vendor/nle/src/makemon.c:778-779.
        def _draw_gnome(vv):
            nv, r_outer = randint_jax(vv, (), 0, 60)

            def _draw_candle_pick(vc):
                nvc, _ = randint_jax(vc, (), 0, 4)
                return nvc

            return jax.lax.cond(
                r_outer == jnp.int32(0), _draw_candle_pick, lambda vc: vc, nv,
            )
        v = jax.lax.cond(is_gnome, _draw_gnome, lambda vv: vv, v)

        # S_NYMPH: rn2(2) gate + optional mongets(MIRROR), then rn2(2) gate +
        # optional mongets(POT_OBJECT_DETECTION).  Vendor makemon.c:700-705:
        #   if (!rn2(2)) (void) mongets(mtmp, MIRROR);
        #   if (!rn2(2)) (void) mongets(mtmp, POT_OBJECT_DETECTION);
        # Each mongets() runs mksobj(otyp, init=TRUE, artif=FALSE) which
        # dispatches the mksobj_init body for the otyp's oclass:
        #   MIRROR (otyp 205, TOOL_CLASS): mkobj.c:897-966 TOOL switch has no
        #     case for MIRROR — falls through default → 0 RNG draws.
        #   POT_OBJECT_DETECTION (POTION_CLASS): mkobj.c:981-987 →
        #     blessorcurse(otmp, 4) → 1–2 ISAAC64 draws via _potion_scroll_draws.
        # Previously only the two outer rn2(2) gates were modelled, so the
        # POT_OBJECT_DETECTION mksobj_init blessorcurse was missing whenever
        # the second gate passed (probability 1/2 per nymph).
        # Cite: vendor/nle/src/makemon.c:700-705;
        #       vendor/nle/src/mkobj.c:981-987 (POTION blessorcurse cascade).
        def _draw_nymph(vv):
            # First gate (rn2(2)) — MIRROR mongets, TOOL default = 0 draws.
            v1, _r1 = randint_jax(vv, (), 0, 2)
            # Second gate (rn2(2)) — POT_OBJECT_DETECTION mongets.
            v2, r2 = randint_jax(v1, (), 0, 2)
            # On second gate fire (r2 == 0), consume POTION_CLASS mksobj_init
            # blessorcurse(4) draws via _potion_scroll_draws.
            v3 = jax.lax.cond(
                r2 == jnp.int32(0),
                _potion_scroll_draws,
                lambda vc: vc,
                v2,
            )
            return v3
        v = jax.lax.cond(is_nymph, _draw_nymph, lambda vv: vv, v)

        # S_MUMMY: rn2(7) — vendor makemon.c:741
        def _draw_mummy(vv):
            nv, _ = randint_jax(vv, (), 0, 7)
            return nv
        v = jax.lax.cond(is_mummy, _draw_mummy, lambda vv: vv, v)

        # S_QUANTMECH: rn2(20) gate — vendor makemon.c:744-762.
        #
        # Vendor body on gate fire:
        #   otmp = mksobj(LARGE_BOX, FALSE, FALSE);  // init=FALSE
        #   if ((catcorpse = mksobj(CORPSE, TRUE, FALSE)) != 0) { ... }
        #
        # The LARGE_BOX mksobj is init=FALSE (mkobj.c:801 ``if (init)`` gate
        # skips the entire mksobj_init switch body), so the TOOL_CLASS
        # LARGE_BOX cascade in mkobj.c:920-924 (rn2(5) olocked + rn2(10)
        # otrapped + mkbox_cnts) does NOT fire here — _tool_lbox_draws
        # would over-consume.
        #
        # The CORPSE mksobj is init=TRUE so the FOOD_CLASS CORPSE cascade
        # at mkobj.c:822-836 DOES fire — an undead_to_corpse do-loop
        # calling rndmonnum() up to 50 tries.  That cascade requires a
        # rndmonst monster-table port and is deferred (TODO: separate
        # commit once rndmonnum draws are modelled).
        #
        # The remaining cat-corpse manipulation (set_corpsenm, stop_timer,
        # add_to_container, weight, mpickobj) at lines 755-761 is RNG-free.
        #
        # Cite: vendor/nle/src/makemon.c:744-762 (S_QUANTMECH case);
        #       vendor/nle/src/mkobj.c:801 (init=FALSE skips mksobj_init);
        #       vendor/nle/src/mkobj.c:822-836 (FOOD_CLASS CORPSE cascade
        #         — deferred, needs rndmonnum port).
        def _draw_qmech(vv):
            nv, _ = randint_jax(vv, (), 0, 20)
            return nv
        v = jax.lax.cond(is_qmech, _draw_qmech, lambda vv: vv, v)

        # S_LEPRECHAUN: d(level_difficulty(), 30) gold — vendor makemon.c:765
        # Vendor:  mkmonmoney(mtmp, (long) d(level_difficulty(), 30));
        # d(n, x) rolls EXACTLY n dice of (rn2(x)+1) — n rn2(30) draws.
        # On Dlvl 1 with depth=1 this is 1 draw (was capped at 8 → over-consumed
        # 7 extra ISAAC64 draws per leprechaun).
        #
        # No mksobj_init cascade is needed: mkmonmoney() calls
        # ``mksobj(GOLD_PIECE, FALSE, FALSE)`` with init=FALSE, so the
        # mkobj.c:801 ``if (init)`` gate skips the entire mksobj_init switch
        # body.  COIN_CLASS's mksobj_init body (mkobj.c:1060-1061) is a noop
        # anyway, but init=FALSE skips it before that point.  S_LEPRECHAUN
        # is not AT_WEAP-armed so _INITWEAP_DRAW_COUNT[S_LEPRECHAUN] == 0
        # (no m_initweap draws either).  Verified: the N rn2(30) loop below
        # is the complete vendor draw set for the leprechaun cascade.
        # Cite: vendor/nle/src/rnd.c:208-224 (d(n,x) = N rn2(x) calls);
        #       vendor/nle/src/makemon.c:564-573 (mkmonmoney → mksobj
        #         init=FALSE);
        #       vendor/nle/src/mkobj.c:801 (init=FALSE skips mksobj_init).
        ld = jnp.maximum(jnp.int32(level_difficulty), jnp.int32(1))

        def _draw_lep(vv):
            def _cond(carry):
                _, idx = carry
                return idx < ld

            def _body(carry):
                vc, idx = carry
                nvc, _r = randint_jax(vc, (), 0, 30)
                return (nvc, idx + jnp.int32(1))

            v_after, _ = jax.lax.while_loop(_cond, _body, (vv, jnp.int32(0)))
            return v_after

        v = jax.lax.cond(is_lep, _draw_lep, lambda vv: vv, v)

        # S_DEMON: rn2(4) ice-devil gate — vendor makemon.c:770
        #   if (ptr == &mons[PM_ICE_DEVIL] && !rn2(4)) ...
        # C short-circuit: the species check is evaluated FIRST, so rn2(4)
        # only fires when ptr == PM_ICE_DEVIL.  Gating on the broader
        # _MLET_DEMON mask drew rn2(4) for every S_DEMON entry (succubus,
        # vrock, balrog, asmodeus, orcus, ...) when vendor draws 0 for them.
        # Cite: vendor/nle/src/makemon.c:770.
        def _draw_ice_devil(vv):
            nv, _ = randint_jax(vv, (), 0, 4)
            return nv
        v = jax.lax.cond(is_ice_devil, _draw_ice_devil, lambda vv: vv, v)

        # S_GIANT: split into minotaur vs other-giant branches — vendor
        # makemon.c:706-719:
        #   case S_GIANT:
        #       if (ptr == &mons[PM_MINOTAUR]) {
        #           if (!rn2(3) || (in_mklev && Is_earthlevel(&u.uz)))
        #               (void) mongets(mtmp, WAN_DIGGING);
        #       } else if (is_giant(ptr)) {
        #           for (cnt = rn2((int) (mtmp->m_lev / 2)); cnt; cnt--) {
        #               otmp = mksobj(rnd_class(DILITHIUM_CRYSTAL, LUCKSTONE - 1),
        #                             FALSE, FALSE);
        #               otmp->quan = (long) rn1(2, 3);
        #               ...
        #           }
        #       }
        #       break;
        #
        # The minotaur and is_giant branches are mutually exclusive (if/else if),
        # and PM_MINOTAUR itself does NOT carry M2_GIANT (mondata.h is_giant
        # checks the M2_GIANT flag, which PM_MINOTAUR lacks).  Previously this
        # single rn2(50) was drawn for ALL S_GIANT entries including
        # PM_MINOTAUR, which both over-consumed (rn2(50) for minotaur where
        # vendor draws rn2(3) + optional wand cascade) and missed the
        # WAN_DIGGING mongets cascade.
        #
        # Dlvl 1 main: Is_earthlevel(&u.uz) is FALSE, so the gate reduces to
        # `!rn2(3)`.  On gate fire, vendor mongets(WAN_DIGGING) →
        # mksobj(WAN_DIGGING, init=TRUE, artif=FALSE) → mkobj.c:1019-1027
        # WAND_CLASS cascade (rn1(5) charges + blessorcurse(17)) → _wand_draws.
        # Cite: vendor/nle/src/makemon.c:706-719;
        #       vendor/nle/src/mkobj.c:1019-1027 (WAND_CLASS cascade);
        #       vendor/nle/include/mondata.h:114 (is_giant = M2_GIANT flag).
        def _draw_minotaur(vv):
            v1, r1 = randint_jax(vv, (), 0, 3)
            # Gate fires when rn2(3) == 0 (Is_earthlevel FALSE on Dlvl 1).
            # On fire: mongets(WAN_DIGGING) consumes _wand_draws cascade.
            v2 = jax.lax.cond(
                r1 == jnp.int32(0),
                _wand_draws,
                lambda vc: vc,
                v1,
            )
            return v2

        # Non-minotaur S_GIANT entries (is_giant branch): rn2(m_lev/2) gem-loop
        # count, approximated as rn2(50).  Per-iteration mksobj(rnd_class(...))
        # is init=FALSE so the GEM_CLASS mksobj_init body is skipped; only the
        # rnd_class draw + rn1(2,3) quan draw fire — those remain unmodelled
        # here as a separate follow-up.
        def _draw_giant_cls(vv):
            nv, _ = randint_jax(vv, (), 0, 50)
            return nv

        # Dispatch: minotaur takes its own branch; other S_GIANT entries
        # fall through to the existing gem-loop approximation.
        v = jax.lax.cond(
            is_minotaur,
            _draw_minotaur,
            lambda vv: jax.lax.cond(
                is_giant_cls, _draw_giant_cls, lambda vc: vc, vv,
            ),
            v,
        )

        # S_LICH: per-species gated draws — vendor makemon.c:727-738.
        # Vendor structure:
        #   if (ptr == &mons[PM_MASTER_LICH] && !rn2(13))
        #       (void) mongets(mtmp, (rn2(7) ? ATHAME : WAN_NOTHING));
        #   else if (ptr == &mons[PM_ARCH_LICH] && !rn2(3)) {
        #       otmp = mksobj(rn2(3) ? ATHAME : QUARTERSTAFF, TRUE,
        #                     rn2(13) ? FALSE : TRUE);
        #       if (otmp->spe < 2) otmp->spe = rnd(3);
        #       if (!rn2(4)) otmp->oerodeproof = 1;
        #   }
        #
        # C short-circuit: the species check is evaluated FIRST in each
        # branch, so non-MASTER/non-ARCH liches (lich, demilich) consume
        # ZERO draws.  Previously _MLET_LICH gated rn2(13)+rn2(7) for ALL
        # S_LICH entries — over-consumed 2 ISAAC64 bytes per lich/demilich.
        #
        # PM_MASTER_LICH and PM_ARCH_LICH are disjoint entries; gate each
        # branch on the specific entry index so only the matching species
        # consumes draws.  The inner ``!rn2(N)`` is C-short-circuited by
        # the species check, then further short-circuits its inner draws
        # via rn2(N)==0.
        #
        # For ARCH_LICH we model the minimal byte-faithful path: rn2(3)
        # gate, then on fire we draw the rn2(3) weapon-pick + rn2(13)
        # blessed flag + rn2(4) erodeproof.  The conditional ``rnd(3)``
        # spe-boost is skipped (it depends on mksobj internals we don't
        # otherwise model).
        def _draw_master_lich(vv):
            v1, r1 = randint_jax(vv, (), 0, 13)

            def _draw_inner(vc):
                nv, _ = randint_jax(vc, (), 0, 7)
                return nv

            return jax.lax.cond(
                r1 == jnp.int32(0), _draw_inner, lambda vc: vc, v1
            )

        v = jax.lax.cond(is_master_lich, _draw_master_lich, lambda vv: vv, v)

        def _draw_arch_lich(vv):
            v1, r1 = randint_jax(vv, (), 0, 3)

            def _draw_inner(vc):
                vc2, _ = randint_jax(vc,  (), 0, 3)   # rn2(3) ATHAME vs QUARTERSTAFF
                vc3, _ = randint_jax(vc2, (), 0, 13)  # rn2(13) blessed flag
                vc4, _ = randint_jax(vc3, (), 0, 4)   # rn2(4) erodeproof
                return vc4

            return jax.lax.cond(
                r1 == jnp.int32(0), _draw_inner, lambda vc: vc, v1
            )

        v = jax.lax.cond(is_arch_lich, _draw_arch_lich, lambda vv: vv, v)

        # S_HUMAN / mercenary armor chain — vendor makemon.c:622-672.
        # Straight-line conditional draws (no loop): up to 9 rn2 calls in
        # worst case, but each block short-circuits on rn2 result.
        # Vendor structure (mac starts type-specific and grows via mongets):
        #   armor_block:  if (mac<-1 && rn2(5)) ...      [draw 1: rn2(5)]
        #                     +rn2(5) plate-mail pick   [draw 2: rn2(5)]
        #                 else if (mac<3 && rn2(5)) ... [draw if mac<3]
        #                     +rn2(3) splint-vs-banded  [draw if branch fired]
        #                 else if (rn2(5)) ...          [draw if both above 0]
        #                     +rn2(3) ring-vs-studded   [draw if branch fired]
        #                 else +leather                  [no draw]
        #   helmet_block: if (mac<10 && rn2(3)) ... else if (mac<10 && rn2(2)) ...
        #   shield_block: if (mac<10 && rn2(3)) ... else if (mac<10 && rn2(2)) ...
        #   boots_block:  if (mac<10 && rn2(3)) ... else if (mac<10 && rn2(2)) ...
        #   gloves_block: if (mac<10 && rn2(3)) ... else if (mac<10 && rn2(2)) ...
        # The C && short-circuits: 2nd rn2 in each "else if" only fires
        # when the first rn2 returned 0.  We honour every short-circuit
        # via lax.cond.  Since spawn-time minvent is empty and mongets()
        # success is tracked through mac monotonically, we model mac<10
        # conservatively as TRUE for the helmet/shield/boots/gloves blocks
        # (vendor reality: mac evolves but stays <10 for most types until
        # late in the chain).  Per-type mac initial values from vendor
        # 594-619 are abstracted to "mac < -1" / "mac < 3" gates picked
        # from the type table — for initial-fill, all mercenary types
        # start with mac in [-3, 3] so the first armor-block always fires
        # at least one rn2(5) draw.  After that, mac < 10 always holds
        # for the helmet/shield/boots/gloves blocks (each adds at most
        # 2-3 to mac, starting from mac in [-3, 3+7]=[-3, 10]).
        #
        # Trace (worst case for a SOLDIER with mac=3): 9 rn2 draws.
        # Trace (typical, e.g. LIEUTENANT mac=-2): 8 rn2 draws.
        # The cap-8 fori_loop approximated this but ignored short-circuits.
        # Replace with explicit vendor-faithful sequence.
        def _draw_hmerc(vv):
            # Block 1: armor chain (rn2(5) gated cascade).
            # mac<-1 first: a single rn2(5) draw fires unconditionally
            # for any mercenary entering this branch.  If it succeeds
            # (returned non-zero), inner rn2(5) for plate pick also fires.
            v1, r_a1 = randint_jax(vv, (), 0, 5)

            def _armor_branch_a(vc):
                # mac<-1 branch took: pick plate vs crystal plate
                nv, _ = randint_jax(vc, (), 0, 5)
                return nv

            def _armor_branch_skip_a(vc):
                # mac<-1 outer was 0 OR mac>=−1: fall to next else-if.
                # Always draw rn2(5) (mac<3 gate is type-specific; we
                # assume mac<3 holds, which is true for all non-WATCH
                # initial mercenaries except SOLDIER/WATCHMAN where the
                # earlier branch's rn2(5) was zero).
                nvc, r_a2 = randint_jax(vc, (), 0, 5)

                def _armor_branch_b(vd):
                    nv, _ = randint_jax(vd, (), 0, 3)
                    return nv

                def _armor_branch_skip_b(vd):
                    # mac<3 inner zero → fall to "else if (rn2(5))".
                    nv, r_a3 = randint_jax(vd, (), 0, 5)

                    def _armor_branch_c(ve):
                        new_v, _ = randint_jax(ve, (), 0, 3)
                        return new_v

                    return jax.lax.cond(
                        r_a3 != jnp.int32(0),
                        _armor_branch_c,
                        lambda ve: ve,
                        nv,
                    )

                return jax.lax.cond(
                    r_a2 != jnp.int32(0),
                    _armor_branch_b,
                    _armor_branch_skip_b,
                    nvc,
                )

            v_after_armor = jax.lax.cond(
                r_a1 != jnp.int32(0),
                _armor_branch_a,
                _armor_branch_skip_a,
                v1,
            )

            # Blocks 2-5: helmet / shield / boots / gloves-cloak.
            # Each is: if (mac<10 && rn2(N)) X else if (mac<10 && rn2(M)) Y
            # We model mac<10 as TRUE (vendor reality for initial spawn —
            # see comment above).  The 2nd rn2 in the else-if fires only
            # when the first rn2 returned 0 (C && short-circuit).
            def _pair_block(vc, n_outer, n_inner):
                nvc, r1 = randint_jax(vc, (), 0, n_outer)

                def _inner_skip(vd):
                    new_v, _ = randint_jax(vd, (), 0, n_inner)
                    return new_v

                return jax.lax.cond(
                    r1 != jnp.int32(0),
                    lambda vd: vd,        # first rn2 nonzero -> branch taken, no 2nd draw
                    _inner_skip,
                    nvc,
                )

            v_after_helm = _pair_block(v_after_armor, 3, 2)
            v_after_shld = _pair_block(v_after_helm,  3, 2)
            v_after_boot = _pair_block(v_after_shld,  3, 2)
            v_after_glov = _pair_block(v_after_boot,  3, 2)

            # Per-subtype tail — vendor makemon.c:653-672.
            #   if (ptr == WATCH_CAPTAIN) { /* no rn2 */ }
            #   else if (ptr == WATCHMAN)  { if (rn2(3)) mongets(TIN_WHISTLE); }
            #   else if (ptr == GUARD)     { /* unconditional curse(whistle) */ }
            #   else { /* soldiers + officers */
            #       if (!rn2(3)) mongets(K_RATION);
            #       if (!rn2(2)) mongets(C_RATION);
            #       if (ptr != SOLDIER && !rn2(3)) mongets(BUGLE);
            #   }
            # WATCH_CAPTAIN and GUARD draw zero in this tail.  WATCHMAN
            # draws exactly one rn2(3).  The else-branch (soldier and the
            # officer ranks SERGEANT/LIEUTENANT/CAPTAIN) draws rn2(3) +
            # rn2(2) unconditionally, plus rn2(3) when not PM_SOLDIER.
            # Cite: vendor/nle/src/makemon.c:653-672.
            def _watchman_tail(vc):
                nv, _ = randint_jax(vc, (), 0, 3)
                return nv

            def _soldier_officer_tail(vc):
                vc1, _ = randint_jax(vc,  (), 0, 3)   # K_RATION gate
                vc2, _ = randint_jax(vc1, (), 0, 2)   # C_RATION gate

                def _bugle(vd):
                    nvd, _ = randint_jax(vd, (), 0, 3)
                    return nvd

                # rn2(3) bugle only when ptr != PM_SOLDIER.
                return jax.lax.cond(
                    is_pm_soldier, lambda vd: vd, _bugle, vc2,
                )

            # Dispatch via nested cond — WATCH_CAPTAIN / GUARD: zero;
            # WATCHMAN: one rn2(3); else: soldier/officer tail.
            v_tail = jax.lax.cond(
                is_pm_watch_captain | is_pm_guard,
                lambda vc: vc,
                lambda vc: jax.lax.cond(
                    is_pm_watchman, _watchman_tail, _soldier_officer_tail, vc,
                ),
                v_after_glov,
            )
            return v_tail

        v = jax.lax.cond(is_hmerc, _draw_hmerc, lambda vv: vv, v)

        # S_HUMAN / shopkeeper: rn2(4) — vendor makemon.c:675
        def _draw_hsk(vv):
            nv, _ = randint_jax(vv, (), 0, 4)
            return nv
        v = jax.lax.cond(is_hsk, _draw_hsk, lambda vv: vv, v)

        # S_HUMAN / priest: rn2(7) + optional rn2(3) + rn2(10).
        # Vendor makemon.c:691-695:
        #   (void) mongets(mtmp, rn2(7) ? ROBE
        #                               : rn2(3) ? CLOAK_OF_PROTECTION
        #                                        : CLOAK_OF_MAGIC_RESISTANCE);
        #   (void) mongets(mtmp, SMALL_SHIELD);
        #   mkmonmoney(mtmp, (long) rn1(10, 20));
        # C ternary short-circuits: rn2(3) fires ONLY when rn2(7) == 0
        # (probability 1/7).  The rn1(10, 20) = rn2(10) + 20 in a separate
        # statement always fires.  Previously this drew rn2(3) every time
        # → over-consumed 1 ISAAC64 draw with probability 6/7 for priests.
        # Cite: vendor/nle/src/makemon.c:691-695.
        def _draw_hpr(vv):
            # Cloak choice (rn2(7) ternary, rn2(3) short-circuited per 634daf8).
            v1, r1 = randint_jax(vv, (), 0, 7)

            def _draw_inner_3(vc):
                nv, _ = randint_jax(vc, (), 0, 3)
                return nv

            v2 = jax.lax.cond(
                r1 == jnp.int32(0), _draw_inner_3, lambda vc: vc, v1
            )
            # Vendor mongets(cloak) then mongets(SMALL_SHIELD) each call
            # mksobj(otyp, init=TRUE, artif=FALSE) → mksobj_init ARMOR_CLASS
            # body (mkobj.c:992-1005).  ROBE/CLOAK_OF_PROTECTION/
            # CLOAK_OF_MAGIC_RESISTANCE/SMALL_SHIELD are none of FUMBLE_BOOTS/
            # LEVITATION_BOOTS/HELM_OF_OPPOSITE_ALIGNMENT/GAUNTLETS_OF_FUMBLING
            # so _armor_draws follows the generic ARMOR_CLASS draw cascade
            # (rn2(10) outer + conditional rn2(11)/rn2(10)/blessorcurse).
            # artif=FALSE so the rn2(40) artifact check is skipped.
            # otyp=0 sentinel avoids the FUMBLE_BOOTS / LEVITATION_BOOTS /
            # HELM_OF_OPPOSITE_ALIGNMENT / GAUNTLETS_OF_FUMBLING short-circuit
            # branch (which would skip the rn2(11) inner draw).  Priest cloaks
            # (ROBE / CLOAK_OF_PROTECTION / CLOAK_OF_MAGIC_RESISTANCE) and
            # SMALL_SHIELD are all in the generic cascade so otyp=0 is byte-
            # equivalent for them.
            v3 = _armor_draws(v2, jnp.int32(0), jnp.bool_(False))   # cloak
            v4 = _armor_draws(v3, jnp.int32(0), jnp.bool_(False))   # SMALL_SHIELD
            # mkmonmoney(mtmp, rn1(10, 20)) — rn2(10) + 20 (mkobj.c:1486-1504
            # mkgold: amount > 0 path skips the rnd(2)/rnd(3) gold cascade).
            v5, _ = randint_jax(v4, (), 0, 10)
            return v5
        v = jax.lax.cond(is_hpr, _draw_hpr, lambda vv: vv, v)

        # --- 7. m_initinv tail — vendor makemon.c:794,796,798 ---
        #   if ((int) mtmp->m_lev > rn2(50))   rnd_defensive_item
        #   if ((int) mtmp->m_lev > rn2(100))  rnd_misc_item
        #   if (likes_gold(ptr) && !findgold(mtmp->minvent) && !rn2(5)) ...
        # First two rn2 draws ALWAYS fire.  The third is C-short-circuit
        # gated by likes_gold (M2_GREEDY): rn2(5) is only evaluated when
        # likes_gold(ptr) is TRUE.  Initial-spawn minvent is empty so
        # !findgold is always true.  Previously drew rn2(5) for ALL monsters
        # → over-consumed 1 ISAAC64 draw per non-greedy monster (the
        # majority of species).
        v, _ = randint_jax(v, (), 0, 50)
        v, _ = randint_jax(v, (), 0, 100)

        def _draw_gold_tail(vv):
            nv, _ = randint_jax(vv, (), 0, 5)
            return nv

        v = jax.lax.cond(likes_gold, _draw_gold_tail, lambda vv: vv, v)

        return v

    def _draw_newcham(v):
        """Consume RNG draws for newcham() + select_newcham_form().

        Vendor: makemon.c:1367  newcham(mtmp, NULL, NO_NC_FLAGS)
                mon.c::newcham (lines 5278-5440) calls select_newcham_form
                then mgender_from_permonst, then newmonhp for the new form.

        Component breakdown (vendor mon.c:5157-5224):
          select_newcham_form switch per cham type:
            CHAMELEON    : rn2(3); if 0 -> pick_animal rn2(N); else fallback
            DOPPELGANGER : rn2(7), rn2(3), optional rn2(3) or do-loop ×5
            SANDESTIN    : rn2(7); if 0 -> fallback
            VAMPIRE*     : pickvampshape = 1–2 draws; no fallback
            NON_PM       : 0 draws; always fallback
          fallback (lines 5214-5223): tryct=50 do-loop, typically 1–3 draws.
          Outer newcham retry (line 5325): tryct=20; typically 1 iteration.
          Model: SELECT_CAP=3 covers 1 switch draw + 2 typical fallback draws.

        mgender_from_permonst (lines 5254-5272):
          rn2(10) fires when new form has ambiguous sex (most monsters).
          Always consume 1 draw.

        newmonhp for the new form (line 5373):
          d(m_lev_new, 8) draws; m_lev_new is unknown at JIT time.
          Modelled as fori_loop(0, MAX_NEW_FORM_LEVEL=20) matching the
          _roll_hp MAX_LEVEL=20 scan convention — same over-consumption
          tradeoff as the original-form HP scan.

        Total: SELECT_CAP(3) + mgender(1) + MAX_NEW_FORM_LEVEL(20) = 24 draws.
        Prior flat cap of 15 under-counted for target forms with m_lev > 11.

        Cite: vendor/nethack/src/mon.c:5157-5224 (select_newcham_form);
              vendor/nethack/src/mon.c:5254-5272 (mgender_from_permonst);
              vendor/nethack/src/mon.c:5278-5440 (newcham);
              vendor/nethack/src/makemon.c:1012-1054 (newmonhp).
        """
        _SELECT_CAP = 3          # 1 switch draw + ~2 typical fallback draws
        _MAX_NEW_FORM_LEVEL = 20  # matches _roll_hp MAX_LEVEL cap

        # --- select_newcham_form draws (switch + fallback) ---
        def _sel_body(_, vc):
            nv, _ = randint_jax(vc, (), 0, 50)
            return nv
        v = jax.lax.fori_loop(0, _SELECT_CAP, _sel_body, v)

        # --- mgender_from_permonst: always 1 draw (rn2(10)) ---
        v, _ = randint_jax(v, (), 0, 10)

        # --- newmonhp for the new form: d(m_lev_new, 8) draws ---
        # m_lev_new unknown; consume MAX_NEW_FORM_LEVEL draws (same convention
        # as _roll_hp which always consumes MAX_LEVEL=20 draws via scan).
        def _hp_body(_, vc):
            nv, _ = randint_jax(vc, (), 1, 9)
            return nv
        v = jax.lax.fori_loop(0, _MAX_NEW_FORM_LEVEL, _hp_body, v)

        return v

    # Dispatch: shapechangers get newcham draws; others get initweap/initinv.
    # Cite: vendor/nethack/src/makemon.c:1356-1368 + 1441-1444.
    vrng = jax.lax.cond(
        is_shapeshifter,
        _draw_newcham,
        _draw_normal_inv,
        vrng,
    )

    # --- 8. saddle check — vendor makemon.c:1447-1454 ---
    # ``!rn2(100) && is_domestic(ptr) && can_saddle(mtmp) && ...``
    # C short-circuit: rn2(100) evaluated for EVERY monster (is_domestic
    # checked only after).  Consume unconditionally for all monsters
    # including shapechangers (saddle check is outside the allow_minvent
    # block — it fires at line 1447, after the allow_minvent block ends at
    # line 1445).  Cite: vendor/nethack/src/makemon.c:1447.
    vrng, _ = randint_jax(vrng, (), 0, 100)
    _ = is_dom  # retained for documentation / future saddle modelling.

    return vrng


# ---------------------------------------------------------------------------
# Spawn initial monsters for a level
# ---------------------------------------------------------------------------

def spawn_initial_monsters(
    rng: jax.Array,
    depth: int,
    n_monsters: int,
    valid_tiles_mask: jnp.ndarray,
    map_h: int,
    map_w: int,
    genocided=None,
    vendor_rng=None,
    player_align: int = 0,
    player_align_record: int = 0,
    mm_nogrp: bool = False,
) -> tuple:
    """Spawn ``n_monsters`` monsters for dungeon level ``depth``.

    Returns
    -------
    positions  : int16[n_monsters, 2]
    type_ids   : int32[n_monsters]
    hps        : int32[n_monsters]
    max_hps    : int32[n_monsters]
    count      : int32 scalar  (== n_monsters; no early-exit pruning)

    When ``vendor_rng`` (Isaac64State) is supplied, the HP-roll draws are
    routed through :func:`vendor_rng.randint_jax` for byte-exact NLE
    replay, and the return tuple is prepended with ``new_vendor_rng`` so
    callers can thread the updated state.

    Uses jax.lax.fori_loop over n_monsters; JIT-compatible.
    """
    # Split rng into per-monster keys: (type_key, hp_key, pos_key) per slot.
    # We pre-split into n_monsters * 3 keys.
    all_keys = jax.random.split(rng, n_monsters * 3)
    type_keys = all_keys[0 * n_monsters : 1 * n_monsters]
    hp_keys   = all_keys[1 * n_monsters : 2 * n_monsters]
    pos_keys  = all_keys[2 * n_monsters : 3 * n_monsters]

    # Pre-sample all type_ids and positions (vectorized approach).
    # fori_loop carry: (positions, type_ids, hps, max_hps)
    init_positions = jnp.zeros((n_monsters, 2), dtype=jnp.int16)
    init_type_ids  = jnp.zeros((n_monsters,),   dtype=jnp.int32)
    init_hps       = jnp.ones((n_monsters,),    dtype=jnp.int32)
    init_max_hps   = jnp.ones((n_monsters,),    dtype=jnp.int32)

    if vendor_rng is not None:
        # Byte-replay path: thread Isaac64State through the per-monster HP roll.
        def _spawn_one_v(i, carry):
            positions, type_ids, hps, max_hps, vrng = carry

            # Byte-replay: thread ISAAC64 through the monster-type weighted draw.
            vrng, type_id = pick_monster_for_level(
                type_keys[i], depth, genocided=genocided, vendor_rng=vrng,
            )
            # newmonhp uses mon->m_lev = adj_lev(ptr), derived from the base
            # mlevel — NOT the mstrength difficulty.  Cite: makemon.c:989.
            m_lev = _adj_lev_jax(_MONSTER_MLEVEL[type_id],
                                 level_difficulty=depth, ulevel=1)
            vrng, hp = _roll_hp(hp_keys[i], m_lev, vendor_rng=vrng,
                                type_id=type_id)
            # Consume vendor makemon.c:1214-1386 post-newmonhp draws.
            # See _consume_makemon_post_hp_draws for the full cascade
            # (female / peace_minded / in_mklev sleep / long-worm /
            #  group / m_initweap / m_initinv / saddle).
            vrng = _consume_makemon_post_hp_draws(
                vrng, type_id,
                player_align=player_align,
                player_align_record=player_align_record,
                in_mklev=True,
                level_difficulty=depth,
                mm_nogrp=mm_nogrp,
            )
            pos = _pick_valid_tile(pos_keys[i], valid_tiles_mask, map_h, map_w)

            positions = positions.at[i].set(pos)
            type_ids  = type_ids.at[i].set(type_id)
            hps       = hps.at[i].set(hp)
            max_hps   = max_hps.at[i].set(hp)

            return positions, type_ids, hps, max_hps, vrng

        positions, type_ids, hps, max_hps, vrng_final = jax.lax.fori_loop(
            0, n_monsters, _spawn_one_v,
            (init_positions, init_type_ids, init_hps, init_max_hps, vendor_rng),
        )
        return vrng_final, positions, type_ids, hps, max_hps, jnp.int32(n_monsters)

    def _spawn_one(i, carry):
        positions, type_ids, hps, max_hps = carry

        type_id = pick_monster_for_level(type_keys[i], depth, genocided=genocided)
        level = MONSTR_DIFFICULTIES[type_id]
        hp = _roll_hp(hp_keys[i], level, type_id=type_id)
        pos = _pick_valid_tile(pos_keys[i], valid_tiles_mask, map_h, map_w)

        positions = positions.at[i].set(pos)
        type_ids  = type_ids.at[i].set(type_id)
        hps       = hps.at[i].set(hp)
        max_hps   = max_hps.at[i].set(hp)

        return positions, type_ids, hps, max_hps

    positions, type_ids, hps, max_hps = jax.lax.fori_loop(
        0, n_monsters, _spawn_one, (init_positions, init_type_ids, init_hps, init_max_hps)
    )

    return positions, type_ids, hps, max_hps, jnp.int32(n_monsters)


# ---------------------------------------------------------------------------
# Populate level in EnvState
# ---------------------------------------------------------------------------


def _populate_oroom_single(
    state,
    rng: jax.Array,
    rooms_arrays,
    i: int,
    next_slot: int,
    player_align_bucket_val: int,
    vendor_rng=None,
):
    """Per-OROOM sleeping-monster spawn for a SINGLE room — vendor mklev.c:813-817.

    Host-side helper extracted from :func:`_populate_per_oroom` so the
    per-OROOM monster spawn can be invoked as step 1 of vendor's
    ``fill_ordinary_room`` body (called from within
    :func:`Nethax.nethax.dungeon.rooms.fill_ordinary_rooms`'s per-room
    loop).  Interleaving the spawn with the per-room feature/trap fills
    matches vendor's draw order — see vendor/nle/src/mklev.c:813-885,
    where line 813 (sleeping monster) is the FIRST draw in each OROOM
    iteration, before traps/gold/fountain/etc.

    Vendor sequence for one OROOM:
      1. ``rn2(3)`` gate (mklev.c:813).
      2. ``somex(croom)`` / ``somey(croom)`` placement (mklev.c:814-815).
      3. ``pick_monster_for_level`` + ``_roll_hp`` (newmonhp) +
         ``_consume_makemon_post_hp_draws`` cascade.
      4. Write into ``state.monster_ai`` slot ``next_slot``, flagged
         asleep (MM_ASLEEP, mklev.c:817).

    Args:
        state:                     EnvState (read+write of monster_ai).
        rng:                       JAX PRNG key (threefry fallback path).
        rooms_arrays:              tuple of host-side numpy arrays
                                   ``(active_np, rtype_np, y1, x1, y2, x2)``.
        i:                         room index in [0, MAX_ROOMS_PER_LEVEL).
        next_slot:                 next free monster_ai slot.
        player_align_bucket_val:   bucket idx into _PEACE_MINDED_TABLE.
        vendor_rng:                optional Isaac64State (NLE_BYTEPARITY).

    Returns:
        ``(state_out, rng_out, vendor_rng_out, next_slot_out)``.  When the
        room is non-OROOM, inactive, or the gate misses, the state is
        returned unchanged and ``next_slot_out == next_slot``.  Under
        NLE_BYTEPARITY, ``rn2(3)`` is consumed regardless of OROOM gate
        only when the room is active (vendor mklev.c:805 short-circuits
        non-OROOM before reaching :813); inactive slots burn no draws.

    Vendor cite: vendor/nle/src/mklev.c:813-817.
    """
    from Nethax.nethax.dungeon.rooms import RoomType  # avoid circular

    active_np, rtype_np, y1_np, x1_np, y2_np, x2_np = rooms_arrays
    OROOM_VAL = int(RoomType.ORDINARY)

    if next_slot >= MAX_MONSTERS_PER_LEVEL:
        return state, rng, vendor_rng, next_slot
    if not bool(active_np[i]):
        return state, rng, vendor_rng, next_slot
    rt = int(rtype_np[i])
    if rt != OROOM_VAL:
        return state, rng, vendor_rng, next_slot

    terrain = state.terrain[0, 0]
    map_h, map_w = terrain.shape

    # Step 1: rn2(3) gate — vendor mklev.c:813.
    if vendor_rng is not None:
        vendor_rng, gate_arr = randint_jax(vendor_rng, (), 0, 3)
        gate_pass = int(gate_arr) == 0
    else:
        rng, k_gate = jax.random.split(rng, 2)
        gate_pass = int(jax.random.randint(k_gate, (), 0, 3)) == 0

    if not gate_pass:
        return state, rng, vendor_rng, next_slot

    # Step 2: somex/somey — vendor mklev.c:814-815.
    lx = int(x1_np[i])
    hx = int(x2_np[i])
    ly = int(y1_np[i])
    hy = int(y2_np[i])
    x_lo, x_hi = lx, hx + 1
    y_lo, y_hi = ly, hy + 1
    if x_hi <= x_lo or y_hi <= y_lo:
        return state, rng, vendor_rng, next_slot

    if vendor_rng is not None:
        vendor_rng, sx_arr = randint_jax(vendor_rng, (), x_lo, x_hi)
        vendor_rng, sy_arr = randint_jax(vendor_rng, (), y_lo, y_hi)
        sx = int(sx_arr)
        sy = int(sy_arr)
    else:
        rng, k_x, k_y = jax.random.split(rng, 3)
        sx = int(jax.random.randint(k_x, (), x_lo, x_hi))
        sy = int(jax.random.randint(k_y, (), y_lo, y_hi))
    sx = max(0, min(sx, map_w - 1))
    sy = max(0, min(sy, map_h - 1))

    # Step 3: spawn pipeline — pick_monster + _roll_hp + cascade.
    player_align_py = int(state.player_align)
    player_align_record_py = 0
    valid_tiles_mask = jnp.zeros((map_h, map_w), dtype=jnp.bool_).at[sy, sx].set(True)

    if vendor_rng is not None:
        rng, sub_rng = jax.random.split(rng, 2)
        (
            vendor_rng, positions, type_ids, hps, max_hps, _count,
        ) = spawn_initial_monsters(
            sub_rng, depth=1, n_monsters=1,
            valid_tiles_mask=valid_tiles_mask,
            map_h=map_h, map_w=map_w,
            genocided=state.genocided_species,
            vendor_rng=vendor_rng,
            player_align=player_align_py,
            player_align_record=player_align_record_py,
        )
    else:
        rng, sub_rng = jax.random.split(rng, 2)
        (
            positions, type_ids, hps, max_hps, _count,
        ) = spawn_initial_monsters(
            sub_rng, depth=1, n_monsters=1,
            valid_tiles_mask=valid_tiles_mask,
            map_h=map_h, map_w=map_w,
            genocided=state.genocided_species,
        )

    # Step 4: write spawn into monster_ai[next_slot].  Vendor mklev.c:817
    # passes MM_ASLEEP — level-gen monsters always start asleep.
    slot = next_slot
    mai = state.monster_ai
    tid = type_ids[0].astype(jnp.int32)
    pos_v = positions[0]
    hp_v = hps[0]
    max_hp_v = max_hps[0]

    kit_id = _MONSTER_KIT_BY_ENTRY[tid].astype(jnp.int32)
    kit_cats = _KIT_CATS[kit_id]
    kit_tids = _KIT_TIDS[kit_id]
    kit_qtys = _KIT_QTYS[kit_id]
    kit_chgs = _KIT_CHGS[kit_id]

    peace_bit = _PEACE_MINDED_TABLE[
        jnp.clip(tid, 0, _PEACE_MINDED_TABLE.shape[0] - 1),
        player_align_bucket_val,
    ]

    new_mai = mai.replace(
        pos=mai.pos.at[slot].set(pos_v),
        hp=mai.hp.at[slot].set(hp_v),
        hp_max=mai.hp_max.at[slot].set(max_hp_v),
        alive=mai.alive.at[slot].set(jnp.bool_(True)),
        ac=mai.ac.at[slot].set(_BASE_AC[tid]),
        is_large=mai.is_large.at[slot].set(_IS_LARGE[tid]),
        attack_dice_n=mai.attack_dice_n.at[slot].set(_ATK_DICE_N[tid].astype(jnp.int8)),
        attack_dice_sides=mai.attack_dice_sides.at[slot].set(_ATK_DICE_S[tid].astype(jnp.int8)),
        mstrategy=mai.mstrategy.at[slot].set(jnp.int8(0)),
        entry_idx=mai.entry_idx.at[slot].set(tid.astype(jnp.int16)),
        peaceful=mai.peaceful.at[slot].set(peace_bit),
        asleep=mai.asleep.at[slot].set(jnp.bool_(True)),
        sleep_timer=mai.sleep_timer.at[slot].set(jnp.int8(127)),
        # Vendor mtmp->movement starts at 0 (struct calloc default — see
        # vendor/nle/src/makemon.c: no explicit init).  Only youmonst gets
        # movement = NORMAL_SPEED at allmain.c:85.  Initializing to 12 was
        # making monsters act on step 1 (dochug fires when mp >= NS) which
        # spawned spurious distfleeck rn2(5) draws and misaligned the
        # ISAAC stream.  Cite: vendor/nle/src/allmain.c:85.
        movement_points=mai.movement_points.at[slot].set(jnp.int16(0)),
        inv_category=mai.inv_category.at[slot].set(kit_cats),
        inv_type_id=mai.inv_type_id.at[slot].set(kit_tids),
        inv_quantity=mai.inv_quantity.at[slot].set(kit_qtys),
        inv_charges=mai.inv_charges.at[slot].set(kit_chgs),
        resists=mai.resists.at[slot].set(
            jnp.take(_MONSTER_MRESISTS, tid, axis=0).astype(jnp.int32)),
        undead=mai.undead.at[slot].set(
            jnp.take(_MONSTER_UNDEAD, tid, axis=0).astype(jnp.bool_)),
        nonliving=mai.nonliving.at[slot].set(
            jnp.take(_MONSTER_NONLIVING, tid, axis=0).astype(jnp.bool_)),
    )
    state_out = state.replace(monster_ai=new_mai)
    return state_out, rng, vendor_rng, next_slot + 1


def spawn_oroom_monster_scanbody(
    monster_ai,
    vrng: "Isaac64State",
    next_slot: jnp.ndarray,
    rng: jax.Array,
    is_active_oroom: jnp.ndarray,
    y1: jnp.ndarray,
    x1: jnp.ndarray,
    y2: jnp.ndarray,
    x2: jnp.ndarray,
    genocided,
    player_align: jnp.ndarray,
    player_align_bucket: jnp.ndarray,
    map_h: int,
    map_w: int,
):
    """JIT-traceable per-OROOM sleeping-monster spawn — vendor mklev.c:813-817.

    Scan-body-callable counterpart of :func:`_populate_oroom_single`.  All
    arguments are traced JAX values so this can run as step 1 inside a
    single ``lax.scan`` over rooms (no Python unroll → one JIT trace).

    Threads ``(monster_ai, vrng, next_slot)`` and writes the spawned
    monster into ``monster_ai`` slot ``next_slot`` (a traced int index —
    ``.at[next_slot].set`` is fine in JAX).  Draw order matches
    :func:`_populate_oroom_single` exactly:

      1. ``rn2(3)`` gate (mklev.c:813) — drawn only when the room is an
         active OROOM (vendor mklev.c:805 short-circuits non-OROOM before
         reaching :813, so inactive/non-OROOM slots burn no ISAAC draws).
      2. ``somex`` (rn2 x_lo..x_hi) then ``somey`` (rn2 y_lo..y_hi)
         (mklev.c:814-815) — only on gate pass.
      3. ``pick_monster_for_level`` + ``_roll_hp`` (newmonhp) +
         ``_consume_makemon_post_hp_draws`` cascade — only on gate pass.
      4. Write into ``monster_ai`` slot ``next_slot`` flagged asleep
         (MM_ASLEEP, mklev.c:817); advance ``next_slot`` by 1.

    The whole spawn is wrapped in ``lax.cond`` on the active-OROOM gate so
    the ISAAC64 stream advances byte-exactly with vendor C (no phantom
    draws on inactive/non-OROOM rooms).

    Vendor cite: vendor/nle/src/mklev.c:813-817.
    """
    h_max = jnp.int32(map_h - 1)
    w_max = jnp.int32(map_w - 1)
    x_lo = x1.astype(jnp.int32)
    x_hi = (x2 + jnp.int32(1)).astype(jnp.int32)
    y_lo = y1.astype(jnp.int32)
    y_hi = (y2 + jnp.int32(1)).astype(jnp.int32)
    # Degenerate room bounds (x_hi<=x_lo etc.) would make rn2 illegal; the
    # active-OROOM gate already excludes inactive slots, but guard the
    # bounds so a malformed room cannot trigger a zero-width rn2.
    valid_bounds = (x_hi > x_lo) & (y_hi > y_lo)
    slot_free = next_slot < jnp.int32(MAX_MONSTERS_PER_LEVEL)
    spawn_gate = is_active_oroom & valid_bounds & slot_free

    def _do_spawn(carry):
        mai_in, v_in, slot_in = carry

        # Step 1: rn2(3) gate — vendor mklev.c:813.
        v_in, gate = randint_jax(v_in, (), 0, 3)
        gate_pass = gate == jnp.int32(0)

        def _gate_true(carry_g):
            mai_g, v_g, slot_g = carry_g
            # Step 2: somex/somey — vendor mklev.c:814-815 (X first, then Y).
            v_g, sx = randint_jax(v_g, (), x_lo, x_hi)
            v_g, sy = randint_jax(v_g, (), y_lo, y_hi)
            sx = jnp.clip(sx, 0, w_max)
            sy = jnp.clip(sy, 0, h_max)

            # Step 3: spawn pipeline.  valid_tiles_mask holds exactly the
            # single placement cell so _pick_valid_tile resolves to (sy, sx).
            valid_mask = jnp.zeros((map_h, map_w), dtype=jnp.bool_).at[sy, sx].set(True)
            v_g, _positions, type_ids, hps, _max_hps, _count = spawn_initial_monsters(
                rng, depth=1, n_monsters=1,
                valid_tiles_mask=valid_mask,
                map_h=map_h, map_w=map_w,
                genocided=genocided,
                vendor_rng=v_g,
                player_align=player_align,
                player_align_record=jnp.int32(0),
                # mklev.c:816 spawns with MM_NOGRP → group rolls skipped.
                mm_nogrp=True,
            )

            # Step 4: write spawn into monster_ai[slot].  Vendor mklev.c:817
            # passes MM_ASLEEP — level-gen monsters always start asleep.
            tid = type_ids[0].astype(jnp.int32)
            hp_v = hps[0]
            pos_v = jnp.stack([sy.astype(jnp.int16), sx.astype(jnp.int16)])

            kit_id = _MONSTER_KIT_BY_ENTRY[tid].astype(jnp.int32)
            peace_bit = _PEACE_MINDED_TABLE[
                jnp.clip(tid, 0, _PEACE_MINDED_TABLE.shape[0] - 1),
                player_align_bucket,
            ]

            mai_g = mai_g.replace(
                pos=mai_g.pos.at[slot_g].set(pos_v),
                hp=mai_g.hp.at[slot_g].set(hp_v),
                hp_max=mai_g.hp_max.at[slot_g].set(hp_v),
                alive=mai_g.alive.at[slot_g].set(jnp.bool_(True)),
                ac=mai_g.ac.at[slot_g].set(_BASE_AC[tid]),
                is_large=mai_g.is_large.at[slot_g].set(_IS_LARGE[tid]),
                attack_dice_n=mai_g.attack_dice_n.at[slot_g].set(_ATK_DICE_N[tid].astype(jnp.int8)),
                attack_dice_sides=mai_g.attack_dice_sides.at[slot_g].set(_ATK_DICE_S[tid].astype(jnp.int8)),
                mstrategy=mai_g.mstrategy.at[slot_g].set(jnp.int8(0)),
                entry_idx=mai_g.entry_idx.at[slot_g].set(tid.astype(jnp.int16)),
                peaceful=mai_g.peaceful.at[slot_g].set(peace_bit),
                asleep=mai_g.asleep.at[slot_g].set(jnp.bool_(True)),
                sleep_timer=mai_g.sleep_timer.at[slot_g].set(jnp.int8(127)),
                # Vendor: mtmp->movement starts at 0; see allmain.c:85 (only
                # youmonst gets NORMAL_SPEED).
                movement_points=mai_g.movement_points.at[slot_g].set(jnp.int16(0)),
                inv_category=mai_g.inv_category.at[slot_g].set(_KIT_CATS[kit_id]),
                inv_type_id=mai_g.inv_type_id.at[slot_g].set(_KIT_TIDS[kit_id]),
                inv_quantity=mai_g.inv_quantity.at[slot_g].set(_KIT_QTYS[kit_id]),
                inv_charges=mai_g.inv_charges.at[slot_g].set(_KIT_CHGS[kit_id]),
                resists=mai_g.resists.at[slot_g].set(
                    jnp.take(_MONSTER_MRESISTS, tid, axis=0).astype(jnp.int32)),
                undead=mai_g.undead.at[slot_g].set(
                    jnp.take(_MONSTER_UNDEAD, tid, axis=0).astype(jnp.bool_)),
                nonliving=mai_g.nonliving.at[slot_g].set(
                    jnp.take(_MONSTER_NONLIVING, tid, axis=0).astype(jnp.bool_)),
            )
            return mai_g, v_g, slot_g + jnp.int32(1)

        return jax.lax.cond(
            gate_pass, _gate_true, lambda c: c, (mai_in, v_in, slot_in),
        )

    return jax.lax.cond(
        spawn_gate, _do_spawn, lambda c: c, (monster_ai, vrng, next_slot),
    )


def _populate_per_oroom(state, rng: jax.Array, rooms, active, vendor_rng=None):
    """Per-OROOM sleeping-monster spawn matching vendor mklev.c:813-817.

    DEPRECATED post-refactor: the per-OROOM monster spawn is now invoked
    as step 1 of each room iteration inside
    :func:`Nethax.nethax.dungeon.rooms.fill_ordinary_rooms`, so the
    rn2(3) gate + somex/somey + makemon cascade interleave with each
    room's feature/trap fills (matching vendor's per-OROOM draw order).
    This function is retained for backward compatibility with callers
    that drive the spawn separately (e.g. legacy tests).  See
    :func:`_populate_oroom_single` for the per-room helper that the new
    in-loop call site uses.

    Vendor cite: vendor/nle/src/mklev.c:813-817; makemon spawn cascade
    at vendor/nle/src/makemon.c:1147-1400.
    """
    # Pull room arrays to host (small int arrays of length MAX_ROOMS_PER_LEVEL).
    active_np = jax.device_get(active)
    rtype_np  = jax.device_get(rooms.room_type)
    y1_np     = jax.device_get(rooms.y1)
    x1_np     = jax.device_get(rooms.x1)
    y2_np     = jax.device_get(rooms.y2)
    x2_np     = jax.device_get(rooms.x2)
    rooms_arrays = (active_np, rtype_np, y1_np, x1_np, y2_np, x2_np)

    player_align_bucket_val = int(jnp.clip(
        state.player_align.astype(jnp.int32) + jnp.int32(1),
        0, _PEACE_MINDED_TABLE.shape[1] - 1,
    ))

    next_slot = 0
    state_out = state
    for i in range(len(active_np)):
        state_out, rng, vendor_rng, next_slot = _populate_oroom_single(
            state_out, rng, rooms_arrays, i, next_slot,
            player_align_bucket_val, vendor_rng=vendor_rng,
        )

    if vendor_rng is not None:
        state_out = state_out.replace(vendor_rng=vendor_rng)
    return state_out


def populate_level_with_monsters(
    state,
    rng: jax.Array,
    n_monsters: int = 5,
    n_rooms: int | None = None,
    vendor_rng=None,
    rooms=None,
    active=None,
) -> object:
    """Spawn sleeping monsters into ordinary rooms.

    Vendor cite: vendor/nle/src/mklev.c:813-817 — the per-OROOM
    sleeping-monster loop body::

        if (u.uhave.amulet || !rn2(3)) {
            x = somex(croom);
            y = somey(croom);
            makemon(NULL, x, y, MM_NOGRP | MM_NOCOUNTBIRTH | MM_ASLEEP);
        }

    Vendor draws EXACTLY ONE potential spawn per OROOM, gated by
    ``!rn2(3)`` (~33% chance per room).  There is no level-wide monster
    count loop — the prior ``rnd((nroom>>1)+1)`` draw cited at this site
    actually drives ``make_niches`` (mklev.c:551), not monster spawning.

    Per-OROOM model (preferred path):
      When ``rooms`` and ``active`` are supplied, this function iterates
      every active OROOM/THEMEROOM, drawing the vendor rn2(3) gate +
      somex/somey per room.  Each room that passes the gate produces at
      most one monster (subject to the MAX_MONSTERS_PER_LEVEL slot cap).

    Legacy count model (fallback for the threefry-only path):
      When ``rooms``/``active`` are not supplied, falls back to the
      pre-existing fixed ``n_monsters`` count for backwards
      compatibility with tests that don't drive a real room table.

    Byte-replay path: when ``vendor_rng`` (Isaac64State) is supplied, the
    per-room rn2(3) gate, somex/somey, and the per-monster HP/inventory
    cascade are routed through :func:`vendor_rng.randint_jax`, and the
    returned state has its ``vendor_rng`` field replaced with the
    updated Isaac64State.  This function is host-side (called from
    env.reset).

    Args:
        state:      EnvState.
        rng:        JAX PRNG key.
        n_monsters: fallback count when neither ``rooms`` nor ``active``
                    are supplied (default 5).  Ignored in the per-OROOM
                    path.
        n_rooms:    deprecated — retained for the threefry legacy path
                    only.  Per-OROOM path supersedes this argument.
        vendor_rng: optional Isaac64State for byte-exact NLE replay.
        rooms:      optional Room pytree (length MAX_ROOMS_PER_LEVEL).
                    When supplied with ``active``, switches to the
                    vendor per-OROOM spawn model.
        active:     optional bool[MAX_ROOMS_PER_LEVEL] mask of which
                    slots are real rooms.

    Vendor cite: vendor/nle/src/mklev.c:813-817 (sleeping-monster spawn);
    vendor/nle/src/makemon.c:1147-1400 (makemon HP + cascade).
    """
    # ------------------------------------------------------------------
    # Per-OROOM model — vendor mklev.c:813-817.  Iterates active rooms;
    # for each, draws rn2(3) + somex/somey; on pass, runs the full
    # spawn pipeline (type pick, newmonhp, post-HP cascade).  At most
    # one monster per room.
    # ------------------------------------------------------------------
    if rooms is not None and active is not None:
        return _populate_per_oroom(
            state, rng, rooms, active, vendor_rng=vendor_rng,
        )

    terrain = state.terrain[0, 0]  # int8[MAP_H, MAP_W]
    map_h, map_w = terrain.shape

    # Valid tiles: FLOOR or CORRIDOR
    walkable = (
        (terrain == jnp.int8(TileType.FLOOR)) |
        (terrain == jnp.int8(TileType.CORRIDOR))
    )
    # Exclude player position
    pr, pc = state.player_pos[0].astype(jnp.int32), state.player_pos[1].astype(jnp.int32)
    player_tile_mask = jnp.ones((map_h, map_w), dtype=jnp.bool_).at[pr, pc].set(False)
    valid_tiles_mask = walkable & player_tile_mask

    # Player alignment fields drive the peace_minded co-aligned tail
    # (vendor makemon.c:2039-2041).  state.player_align is the alignment
    # sign (-1/0/+1); u.ualign.record is not yet tracked on EnvState and
    # defaults to 0 at game start (vendor role.c init_role).
    player_align_py = int(state.player_align)
    player_align_record_py = 0

    if vendor_rng is not None:
        vendor_rng, positions, type_ids, hps, max_hps, count = spawn_initial_monsters(
            rng, depth=1, n_monsters=n_monsters, valid_tiles_mask=valid_tiles_mask,
            map_h=map_h, map_w=map_w,
            genocided=state.genocided_species,
            vendor_rng=vendor_rng,
            player_align=player_align_py,
            player_align_record=player_align_record_py,
        )
        # Thread updated Isaac64State back into the EnvState so subsequent
        # consumers (env.step) read the post-spawn vendor RNG.
        state = state.replace(vendor_rng=vendor_rng)
    else:
        positions, type_ids, hps, max_hps, count = spawn_initial_monsters(
            rng, depth=1, n_monsters=n_monsters, valid_tiles_mask=valid_tiles_mask,
            map_h=map_h, map_w=map_w,
            genocided=state.genocided_species,
        )

    mai = state.monster_ai

    # Player alignment bucket for vendor peace_minded lookup.
    # Cite: vendor/nethack/src/makemon.c::peace_minded (line ~2003-2042).
    # Bucket = align + 1 (clamped to [0, 2]).
    player_align_bucket = jnp.clip(
        state.player_align.astype(jnp.int32) + jnp.int32(1),
        0, _PEACE_MINDED_TABLE.shape[1] - 1,
    )

    # Write slots [0, n_monsters) from spawn results.
    # Use fori_loop to stay JIT-compatible.
    def _write_slot(i, mai_carry):
        type_id = type_ids[i]
        new_pos       = mai_carry.pos.at[i].set(positions[i])
        new_hp        = mai_carry.hp.at[i].set(hps[i])
        new_hp_max    = mai_carry.hp_max.at[i].set(max_hps[i])
        new_alive     = mai_carry.alive.at[i].set(jnp.bool_(True))
        new_ac        = mai_carry.ac.at[i].set(_BASE_AC[type_id])
        new_is_large  = mai_carry.is_large.at[i].set(_IS_LARGE[type_id])
        new_atk_n     = mai_carry.attack_dice_n.at[i].set(_ATK_DICE_N[type_id].astype(jnp.int8))
        new_atk_s     = mai_carry.attack_dice_sides.at[i].set(_ATK_DICE_S[type_id].astype(jnp.int8))
        new_strategy  = mai_carry.mstrategy.at[i].set(jnp.int8(0))  # NONE until awakened
        new_entry     = mai_carry.entry_idx.at[i].set(type_id.astype(jnp.int16))
        # Vendor mtmp->movement starts at 0 (calloc default — makemon.c has
        # no explicit movement init; only youmonst gets NORMAL_SPEED at
        # allmain.c:85).  Starting at NORMAL_SPEED was making monsters dochug
        # on step 1, emitting spurious distfleeck rn2(5) draws and
        # misaligning the ISAAC stream.
        new_mp        = mai_carry.movement_points.at[i].set(jnp.int16(0))
        # vendor mklev.c::mkmon + makemon.c::makemon set mtmp->msleeping=1
        # on every level-gen monster (MM_ASLEEP flag).  Awakened via vendor
        # disturb() on LoS/sound checks.  Without this, level-1 monsters are
        # adjacent + active on turn 0 and kill low-HP roles in ~2 hits —
        # explains the 3.5x death-rate divergence (DEATH_RATE_INVESTIGATION.md).
        new_asleep = mai_carry.asleep.at[i].set(jnp.bool_(True))
        # sleep_timer: persistent vendor msleeping flag (no expiry on its own);
        # mirror with a large value so the timer doesn't run out before
        # disturb() wakes the monster on LoS.
        new_sleep_timer = mai_carry.sleep_timer.at[i].set(jnp.int8(127))
        # peace_minded lookup — vendor makemon.c::peace_minded.
        peace_bit = _PEACE_MINDED_TABLE[
            jnp.clip(type_id.astype(jnp.int32), 0, _PEACE_MINDED_TABLE.shape[0] - 1),
            player_align_bucket,
        ]
        new_peaceful  = mai_carry.peaceful.at[i].set(peace_bit)
        # Per-monster resist/undead/nonliving from MONSTERS table.
        # Cite: vendor/nethack/src/monst.c MON() mr1 field.
        tid           = type_id.astype(jnp.int32)
        new_resists   = mai_carry.resists.at[i].set(
            jnp.take(_MONSTER_MRESISTS, tid, axis=0).astype(jnp.int32))
        new_undead    = mai_carry.undead.at[i].set(
            jnp.take(_MONSTER_UNDEAD, tid, axis=0).astype(jnp.bool_))
        new_nonliving = mai_carry.nonliving.at[i].set(
            jnp.take(_MONSTER_NONLIVING, tid, axis=0).astype(jnp.bool_))

        # Vendor makemon.c::mongets: assign per-class inventory kit.
        kit_id = _MONSTER_KIT_BY_ENTRY[type_id.astype(jnp.int32)].astype(jnp.int32)
        kit_cats = _KIT_CATS[kit_id]   # [MAX_MONSTER_INV] int8
        kit_tids = _KIT_TIDS[kit_id]   # [MAX_MONSTER_INV] int16
        kit_qtys = _KIT_QTYS[kit_id]   # [MAX_MONSTER_INV] int16
        kit_chgs = _KIT_CHGS[kit_id]   # [MAX_MONSTER_INV] int8

        new_invc = mai_carry.inv_category.at[i].set(kit_cats)
        new_invt = mai_carry.inv_type_id.at[i].set(kit_tids)
        new_invq = mai_carry.inv_quantity.at[i].set(kit_qtys)
        new_invch = mai_carry.inv_charges.at[i].set(kit_chgs)

        return mai_carry.replace(
            pos=new_pos,
            hp=new_hp,
            hp_max=new_hp_max,
            alive=new_alive,
            ac=new_ac,
            is_large=new_is_large,
            attack_dice_n=new_atk_n,
            attack_dice_sides=new_atk_s,
            mstrategy=new_strategy,
            entry_idx=new_entry,
            peaceful=new_peaceful,
            movement_points=new_mp,
            inv_category=new_invc,
            inv_type_id=new_invt,
            inv_quantity=new_invq,
            inv_charges=new_invch,
            resists=new_resists,
            undead=new_undead,
            nonliving=new_nonliving,
            asleep=new_asleep,
            sleep_timer=new_sleep_timer,
        )

    new_mai = jax.lax.fori_loop(0, n_monsters, _write_slot, mai)
    state_after_mai = state.replace(monster_ai=new_mai)

    # Long-worm setup.
    # Cite: vendor/nethack/src/makemon.c lines 1405-1409:
    #     if (mndx == PM_LONG_WORM && (mtmp->wormno = get_wormno()) != 0) {
    #         initworm(mtmp, allowtail ? rn2(5) : 0);
    #         if (count_wsegs(mtmp))
    #             place_worm_tail_randomly(mtmp, x, y);
    #     }
    # PM_LONG_WORM is monster entry index 118 (constants/monster_entries/
    # chunk2.py).  For each newly-spawned long-worm slot we allocate a worm
    # slot, initialise its single head segment at the monster position, and
    # scatter random tail segments around it.
    from Nethax.nethax.subsystems.worm import (
        init_worm as _init_worm,
        place_worm_tail_randomly as _place_tail,
        get_wormno as _get_wormno,
    )
    PM_LONG_WORM = jnp.int32(113)

    def _worm_init_slot(i, carry):
        st, key = carry
        entry = type_ids[i].astype(jnp.int32)
        is_long_worm = entry == PM_LONG_WORM
        key, k_alloc, k_tail = jax.random.split(key, 3)
        found, wslot = _get_wormno(st)
        do_init = is_long_worm & found

        def _do(st_in):
            r_ = positions[i, 0]
            c_ = positions[i, 1]
            st1 = _init_worm(st_in, wslot, jnp.int32(i), r_, c_)
            st2 = _place_tail(st1, wslot, r_, c_, k_tail)
            return st2

        st_new = jax.lax.cond(do_init, _do, lambda s: s, st)
        return (st_new, key), None

    rng_worm, _ = jax.random.split(rng)
    (state_final, _), _ = jax.lax.scan(
        lambda c, i: _worm_init_slot(i, c),
        (state_after_mai, rng_worm),
        jnp.arange(n_monsters, dtype=jnp.int32),
    )
    return state_final


# ---------------------------------------------------------------------------
# makemon() — runtime monster spawn (vendor src/mon.c::makemon lines 1147+)
# ---------------------------------------------------------------------------
#
# Vendor reference:
#   vendor/nethack/src/makemon.c::makemon  (lines 1147-1400) — find a slot,
#     allocate a struct monst, set_mon_data(), newmonhp(), place_monster(),
#     mongets() per-class inventory.
#   vendor/nethack/src/makemon.c::newmonhp (lines 1012-1054) — HP roll;
#     d(m_lev, 8) with rnd(4) for m_lev==0 plus the "all-ones boost".
#   vendor/nethack/src/makemon.c::mongets  (lines 2181-2230) — per-class
#     starter inventory; here delegated to the per-entry kit table built at
#     module load (_MONSTER_KIT_BY_ENTRY + _KIT_*).
#
# JAX-required divergence:
#   * Slot lookup uses ``argmax(~alive)`` (first dead slot).  Vendor allocates
#     a fresh ``struct monst`` from a free pool; we work over the fixed
#     MAX_MONSTERS_PER_LEVEL array.  Slot 0 is the sentinel and is excluded.
#   * No goodpos()/enexto_core() fallback — caller supplies a position that
#     is already vetted (scroll/wand handlers clip to map bounds).  If the
#     tile is occupied or out of bounds the spawn is suppressed and the
#     returned ``success`` flag is False.
#   * MM_* flags (MM_ANGRY, MM_ASLEEP, MM_FEMALE, MM_NOCOUNTBIRTH, etc.) are
#     not modeled at this entry point — callers wanting non-default state
#     should patch the returned slot manually (mirrors vendor postprocessing
#     after the makemon() call returns).
# ---------------------------------------------------------------------------

def makemon(state, rng: jax.Array, entry_idx: jnp.ndarray,
            pos: jnp.ndarray) -> tuple:
    """Spawn one monster of species ``entry_idx`` at ``pos`` on the current level.

    Parameters
    ----------
    state:     EnvState.
    rng:       JAX PRNG key.
    entry_idx: int — species index into the MONSTERS table.  Clipped to
               [0, NUMMONS).  Caller is responsible for any G_NOGEN / G_UNIQ
               / genocide filtering (vendor makemon.c lines 1202-1232).
    pos:       int[2] (row, col) — where to place the spawn.

    Returns
    -------
    (new_state, slot_idx, success)
        new_state: EnvState with the chosen slot populated; identical to
                   ``state`` when no dead slot is available.
        slot_idx:  int32 — index of the populated slot (or 0 if !success).
        success:   bool — False if no dead slot was found.

    Vendor: mon.c::makemon lines 1147-1400; HP via newmonhp lines 1012-1054;
    inventory via mongets lines 2181-2230 (delegated to _MONSTER_KIT_BY_ENTRY).
    """
    mai = state.monster_ai
    eidx = jnp.clip(entry_idx.astype(jnp.int32), 0, NUMMONS - 1)

    # Slot 0 is reserved as a sentinel by the level-fill path; exclude it
    # so freshly-allocated runtime spawns don't clobber the empty-marker.
    dead_mask = (~mai.alive).at[0].set(False)
    has_dead = jnp.any(dead_mask)
    slot = jnp.argmax(dead_mask.astype(jnp.int32)).astype(jnp.int32)

    # ---- HP roll — vendor newmonhp (makemon.c:1012-1054) ----
    # Uses MONSTR_DIFFICULTIES as m_lev proxy (vendor mons[i].difficulty
    # tracks dungeon-curve hit dice; same field _roll_hp consumes).
    m_lev = MONSTR_DIFFICULTIES[eidx].astype(jnp.int32)
    new_hp = _roll_hp(rng, m_lev)

    # ---- Per-class inventory — vendor mongets (makemon.c:2181-2230) ----
    kit_id = _MONSTER_KIT_BY_ENTRY[eidx].astype(jnp.int32)
    kit_cats = _KIT_CATS[kit_id]
    kit_tids = _KIT_TIDS[kit_id]
    kit_qtys = _KIT_QTYS[kit_id]
    kit_chgs = _KIT_CHGS[kit_id]

    row16 = pos[0].astype(jnp.int16)
    col16 = pos[1].astype(jnp.int16)
    new_pos = jnp.stack([row16, col16])

    # Conditional writes gated on (has_dead AND in-bounds).  No goodpos()
    # check; caller-supplied positions are trusted (see header note).
    pr = pos[0].astype(jnp.int32)
    pc = pos[1].astype(jnp.int32)
    h = state.terrain.shape[2]
    w = state.terrain.shape[3]
    in_bounds = (pr >= 0) & (pr < h) & (pc >= 0) & (pc < w)
    do_spawn = has_dead & in_bounds

    def _apply(mai_in):
        new_alive  = mai_in.alive.at[slot].set(jnp.bool_(True))
        new_pos_a  = mai_in.pos.at[slot].set(new_pos)
        new_hp_a   = mai_in.hp.at[slot].set(new_hp.astype(jnp.int32))
        new_hpmx   = mai_in.hp_max.at[slot].set(new_hp.astype(jnp.int32))
        new_entry  = mai_in.entry_idx.at[slot].set(eidx.astype(jnp.int16))
        new_mlev   = mai_in.m_lev.at[slot].set(m_lev.astype(jnp.int16))
        new_ac     = mai_in.ac.at[slot].set(_BASE_AC[eidx])
        new_isl    = mai_in.is_large.at[slot].set(_IS_LARGE[eidx])
        new_atkn   = mai_in.attack_dice_n.at[slot].set(_ATK_DICE_N[eidx].astype(jnp.int8))
        new_atks   = mai_in.attack_dice_sides.at[slot].set(_ATK_DICE_S[eidx].astype(jnp.int8))
        # peaceful defaults to False (vendor MM_ANGRY-style default for the
        # runtime spawn path; scroll-of-create-monster + summons are hostile).
        new_peace  = mai_in.peaceful.at[slot].set(jnp.bool_(False))
        new_tame   = mai_in.tame.at[slot].set(jnp.bool_(False))
        new_resists = mai_in.resists.at[slot].set(
            jnp.take(_MONSTER_MRESISTS, eidx, axis=0).astype(jnp.int32))
        new_undead = mai_in.undead.at[slot].set(
            jnp.take(_MONSTER_UNDEAD, eidx, axis=0).astype(jnp.bool_))
        new_nonliv = mai_in.nonliving.at[slot].set(
            jnp.take(_MONSTER_NONLIVING, eidx, axis=0).astype(jnp.bool_))
        # Vendor mtmp->movement starts at 0 — see allmain.c:85.
        new_mp     = mai_in.movement_points.at[slot].set(jnp.int16(0))
        # Mongets inventory kit.
        new_ic = mai_in.inv_category.at[slot].set(kit_cats)
        new_it = mai_in.inv_type_id.at[slot].set(kit_tids)
        new_iq = mai_in.inv_quantity.at[slot].set(kit_qtys)
        new_ich = mai_in.inv_charges.at[slot].set(kit_chgs)
        # Strategy resets so the new spawn picks a hunt target next tick.
        new_strat = mai_in.mstrategy.at[slot].set(jnp.int8(0))
        return mai_in.replace(
            alive=new_alive,
            pos=new_pos_a,
            hp=new_hp_a,
            hp_max=new_hpmx,
            entry_idx=new_entry,
            m_lev=new_mlev,
            ac=new_ac,
            is_large=new_isl,
            attack_dice_n=new_atkn,
            attack_dice_sides=new_atks,
            peaceful=new_peace,
            tame=new_tame,
            resists=new_resists,
            undead=new_undead,
            nonliving=new_nonliv,
            movement_points=new_mp,
            inv_category=new_ic,
            inv_type_id=new_it,
            inv_quantity=new_iq,
            inv_charges=new_ich,
            mstrategy=new_strat,
        )

    new_mai = jax.lax.cond(do_spawn, _apply, lambda m: m, mai)
    new_state = state.replace(monster_ai=new_mai)
    return new_state, slot, do_spawn
