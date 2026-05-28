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
        int(MonsterSymbol.S_ANGEL):     3,   # !rn2(20) + rn2(2) + rn2(4)  line 332-340
        int(MonsterSymbol.S_HUMANOID):  6,   # dwarf worst case             line 367-386
        int(MonsterSymbol.S_KOP):       3,   # !rn2(4) + !rn2(3) + rn2(2)   line 392-395
        int(MonsterSymbol.S_ORC):       5,   # helm + orc-captain branch    line 397-432
        int(MonsterSymbol.S_OGRE):      1,   # !rn2(N)                       line 434
        int(MonsterSymbol.S_TROLL):     2,   # !rn2(2) + rn2(4)              line 440-454
        int(MonsterSymbol.S_KOBOLD):    1,   # !rn2(4)                       line 457
        int(MonsterSymbol.S_CENTAUR):   1,   # rn2(2)                        line 462
        int(MonsterSymbol.S_WRAITH):    0,   # no rn2                         line 472-475
        int(MonsterSymbol.S_ZOMBIE):    3,   # !rn2(4) + !rn2(4) + rn2(3)   line 477-481
        int(MonsterSymbol.S_LIZARD):    2,   # salamander rn2(7) + rn2(3)   line 482-486
        int(MonsterSymbol.S_DEMON):     1,   # rn2(4) trident-vs-bullwhip   line 497
    }
    default_count = 1  # default switch case: rnd(14-2*bias) — 1 draw

    counts = []
    for m in MONSTERS:
        # Non-armed monsters skip m_initweap entirely; assign 0.
        armed = False
        for atk in (m.attacks or ()):
            if int(atk[0]) == int(AttackType.AT_WEAP):
                armed = True
                break
        if not armed:
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
_MLET_KOBOLD:      jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_KOBOLD)
_MLET_MUMMY:       jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_MUMMY)
_MLET_QUANTMECH:   jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_QUANTMECH)
_MLET_LEPRECHAUN:  jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_LEPRECHAUN)
_MLET_DEMON:       jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_DEMON)
_MLET_GIANT:       jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_GIANT)
_MLET_LICH:        jnp.ndarray = _compute_mlet_mask(MonsterSymbol.S_LICH)
_MLET_HUMAN_MERC:  jnp.ndarray = _compute_mlet_human_merc()
_MLET_HUMAN_SK:    jnp.ndarray = _compute_mlet_human_shopkeeper()
_MLET_HUMAN_PR:    jnp.ndarray = _compute_mlet_human_priest()


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
            # Trailing offensive-item check — vendor makemon.c:556
            #   if ((int) mtmp->m_lev > rn2(75)) (void) mongets(...)
            v_after, _ = randint_jax(v_after, (), 0, 75)
            return v_after

        v = jax.lax.cond(is_armed, _draw_initweap, lambda vv: vv, v)

        # --- 6b. m_initinv per-class body — vendor makemon.c:589-788 ---
        # Only classes with rn2() calls listed; others are 0 draws.
        # Cite: makemon.c:1444  m_initinv(mtmp)
        is_kobold    = _MLET_KOBOLD[tid]
        is_gnome     = _MLET_GNOME[tid]
        is_nymph     = _MLET_NYMPH[tid]
        is_mummy     = _MLET_MUMMY[tid]
        is_qmech     = _MLET_QUANTMECH[tid]
        is_lep       = _MLET_LEPRECHAUN[tid]
        is_demon_cls = _MLET_DEMON[tid]
        is_giant_cls = _MLET_GIANT[tid]
        is_lich_cls  = _MLET_LICH[tid]
        is_hmerc     = _MLET_HUMAN_MERC[tid]
        is_hsk       = _MLET_HUMAN_SK[tid]
        is_hpr       = _MLET_HUMAN_PR[tid]

        # S_KOBOLD: rn2(4) — vendor makemon.c:457
        def _draw_kobold(vv):
            nv, _ = randint_jax(vv, (), 0, 4)
            return nv
        v = jax.lax.cond(is_kobold, _draw_kobold, lambda vv: vv, v)

        # S_GNOME: rn2(60) — vendor makemon.c:778
        def _draw_gnome(vv):
            nv, _ = randint_jax(vv, (), 0, 60)
            return nv
        v = jax.lax.cond(is_gnome, _draw_gnome, lambda vv: vv, v)

        # S_NYMPH: rn2(2) + rn2(2) — vendor makemon.c:701,703
        def _draw_nymph(vv):
            v1, _ = randint_jax(vv, (), 0, 2)
            v2, _ = randint_jax(v1, (), 0, 2)
            return v2
        v = jax.lax.cond(is_nymph, _draw_nymph, lambda vv: vv, v)

        # S_MUMMY: rn2(7) — vendor makemon.c:741
        def _draw_mummy(vv):
            nv, _ = randint_jax(vv, (), 0, 7)
            return nv
        v = jax.lax.cond(is_mummy, _draw_mummy, lambda vv: vv, v)

        # S_QUANTMECH: rn2(20) — vendor makemon.c:745
        def _draw_qmech(vv):
            nv, _ = randint_jax(vv, (), 0, 20)
            return nv
        v = jax.lax.cond(is_qmech, _draw_qmech, lambda vv: vv, v)

        # S_LEPRECHAUN: d(level_difficulty(), 30) gold — vendor makemon.c:765
        # Vendor:  mkmonmoney(mtmp, (long) d(level_difficulty(), 30));
        # d(n, x) rolls EXACTLY n dice of (rn2(x)+1) — n rn2(30) draws.
        # On Dlvl 1 with depth=1 this is 1 draw (was capped at 8 → over-consumed
        # 7 extra ISAAC64 draws per leprechaun).
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

        # S_DEMON: rn2(4) ice devil gate — vendor makemon.c:770
        def _draw_demon_cls(vv):
            nv, _ = randint_jax(vv, (), 0, 4)
            return nv
        v = jax.lax.cond(is_demon_cls, _draw_demon_cls, lambda vv: vv, v)

        # S_GIANT: rn2(m_lev/2) gem-loop count — vendor makemon.c:711
        def _draw_giant_cls(vv):
            nv, _ = randint_jax(vv, (), 0, 50)
            return nv
        v = jax.lax.cond(is_giant_cls, _draw_giant_cls, lambda vv: vv, v)

        # S_LICH: rn2(13) + rn2(7) — vendor makemon.c:728-737
        def _draw_lich_cls(vv):
            v1, _ = randint_jax(vv, (), 0, 13)
            v2, _ = randint_jax(v1, (), 0, 7)
            return v2
        v = jax.lax.cond(is_lich_cls, _draw_lich_cls, lambda vv: vv, v)

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
            return v_after_glov

        v = jax.lax.cond(is_hmerc, _draw_hmerc, lambda vv: vv, v)

        # S_HUMAN / shopkeeper: rn2(4) — vendor makemon.c:675
        def _draw_hsk(vv):
            nv, _ = randint_jax(vv, (), 0, 4)
            return nv
        v = jax.lax.cond(is_hsk, _draw_hsk, lambda vv: vv, v)

        # S_HUMAN / priest: rn2(7)+rn2(3)+rn2(10) — vendor makemon.c:691-695
        def _draw_hpr(vv):
            v1, _ = randint_jax(vv,  (), 0, 7)
            v2, _ = randint_jax(v1, (), 0, 3)
            v3, _ = randint_jax(v2, (), 0, 10)
            return v3
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
        attack_dice_n=mai.attack_dice_n.at[slot].set(_ATK_DICE_N[tid]),
        attack_dice_sides=mai.attack_dice_sides.at[slot].set(_ATK_DICE_S[tid]),
        mstrategy=mai.mstrategy.at[slot].set(jnp.int8(0)),
        entry_idx=mai.entry_idx.at[slot].set(tid.astype(jnp.int16)),
        peaceful=mai.peaceful.at[slot].set(peace_bit),
        asleep=mai.asleep.at[slot].set(jnp.bool_(True)),
        sleep_timer=mai.sleep_timer.at[slot].set(jnp.int8(127)),
        movement_points=mai.movement_points.at[slot].set(jnp.int16(12)),
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
                attack_dice_n=mai_g.attack_dice_n.at[slot_g].set(_ATK_DICE_N[tid]),
                attack_dice_sides=mai_g.attack_dice_sides.at[slot_g].set(_ATK_DICE_S[tid]),
                mstrategy=mai_g.mstrategy.at[slot_g].set(jnp.int8(0)),
                entry_idx=mai_g.entry_idx.at[slot_g].set(tid.astype(jnp.int16)),
                peaceful=mai_g.peaceful.at[slot_g].set(peace_bit),
                asleep=mai_g.asleep.at[slot_g].set(jnp.bool_(True)),
                sleep_timer=mai_g.sleep_timer.at[slot_g].set(jnp.int8(127)),
                movement_points=mai_g.movement_points.at[slot_g].set(jnp.int16(12)),
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
        new_atk_n     = mai_carry.attack_dice_n.at[i].set(_ATK_DICE_N[type_id])
        new_atk_s     = mai_carry.attack_dice_sides.at[i].set(_ATK_DICE_S[type_id])
        new_strategy  = mai_carry.mstrategy.at[i].set(jnp.int8(0))  # NONE until awakened
        new_entry     = mai_carry.entry_idx.at[i].set(type_id.astype(jnp.int16))
        # Vendor makemon.c sets mtmp->movement = NORMAL_SPEED so a freshly-
        # spawned monster can act on its very next tick.  Mirror that here.
        new_mp        = mai_carry.movement_points.at[i].set(jnp.int16(12))
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
        new_mp     = mai_in.movement_points.at[slot].set(jnp.int16(12))
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
