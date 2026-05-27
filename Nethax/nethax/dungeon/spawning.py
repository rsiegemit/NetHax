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
    MZ_LARGE,
    MZ_HUGE,
    MZ_GIGANTIC,
    M2_MAGIC,
    M2_NASTY,
    M2_GREEDY,
    M2_STRONG,
    M2_NEUTER,
    MS_SOLDIER,
    MS_PRIEST,
    MS_SPELL,
    MS_SELL,
)
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.monster_ai import (
    MAX_MONSTERS_PER_LEVEL,
    MAX_MONSTER_INV,
    _MONSTER_MRESISTS,
    _MONSTER_UNDEAD,
    _MONSTER_NONLIVING,
)
from Nethax.nethax.vendor_rng import Isaac64State, randint_jax, isaac_weighted_choice


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


def _compute_gen_freqs() -> jnp.ndarray:
    """Extract the low byte of generation_mask as generation frequency weight."""
    freqs = [m.generation_mask & 0xFF for m in MONSTERS]
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
_GEN_FREQS: jnp.ndarray = _compute_gen_freqs()               # [NUMMONS] int32
_IS_NOGEN: jnp.ndarray = _compute_nogen_mask()               # [NUMMONS] bool
_IS_UNIQ: jnp.ndarray = _compute_uniq_mask()                 # [NUMMONS] bool
_IS_LARGE: jnp.ndarray = _compute_is_large()                 # [NUMMONS] bool
_BASE_AC: jnp.ndarray = _compute_base_ac()                   # [NUMMONS] int8
_ATK_DICE_N, _ATK_DICE_S = _compute_primary_attack_dice()    # [NUMMONS] int8 each
_IS_NEUTER: jnp.ndarray = _compute_is_neuter()               # [NUMMONS] bool
_IS_ARMED: jnp.ndarray = _compute_is_armed()                 # [NUMMONS] bool


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

def eligible_monsters_for_depth(depth: int, genocided=None) -> jnp.ndarray:
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
    """
    lo = jnp.int32(depth - 6)
    hi = jnp.int32(depth + 5)
    in_window = (MONSTR_DIFFICULTIES >= lo) & (MONSTR_DIFFICULTIES <= hi)
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
    weighted draw is routed through :func:`vendor_rng.isaac_weighted_choice`
    so the chosen index is byte-exact with vendor C
    ``rndmonst_inner`` (rn2(total) over cumsum buckets).  In that case the
    return is ``(new_vendor_rng, monster_idx)``.  When ``vendor_rng is None``
    the existing Threefry path is preserved and only ``monster_idx`` is
    returned.
    """
    mask = eligible_monsters_for_depth(depth, genocided=genocided)
    weights = jnp.where(mask, _GEN_FREQS, jnp.int32(0)).astype(jnp.int32)
    # Guard: if all weights zero (very unusual depth), fall back to uniform over eligible.
    total = jnp.sum(weights)
    weights = jnp.where(total > 0, weights, mask.astype(jnp.int32))

    if vendor_rng is not None:
        new_vrng, idx = isaac_weighted_choice(vendor_rng, weights)
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
# Roll initial HP for a monster (makemon.c::newmonhp)
# ---------------------------------------------------------------------------

def _roll_hp(rng: jax.Array, level: jnp.ndarray, vendor_rng=None):
    """Roll initial HP for a freshly-generated monster — byte-equal vendor.

    Vendor: vendor/nethack/src/makemon.c::newmonhp (~line 1500)::

        if (mlvl == 0) hp = rnd(4);     // 1..4 for "zero-level" monsters
        else           hp = d(mlvl, 8); // sum of mlvl rolls of d8

    JIT-safe via masked scan over a static cap of 20 dice.

    Byte-replay path: when ``vendor_rng`` (Isaac64State) is supplied, the
    per-die ``rnd(8)`` and the zero-level ``rnd(4)`` draws are routed
    through :func:`vendor_rng.randint_jax`, which is byte-exact with vendor
    C ``minval + isaac64_next_uint64() % range``.  The scan carries
    ``vendor_rng`` through each iteration; returns ``(new_vendor_rng, hp)``
    so the caller can thread state.  ``vendor_rng=None`` retains the
    Threefry path unchanged.
    """
    MAX_LEVEL = 20  # static cap; no monster generated by mklev exceeds this
    level_i32 = jnp.clip(level.astype(jnp.int32), 0, MAX_LEVEL)

    if vendor_rng is not None:
        # Byte-replay path: consume ISAAC64 via randint_jax inside the scan.
        def _one_die_v(carry, die_idx):
            vrng, total = carry
            new_vrng, roll = randint_jax(vrng, (), 1, 9)
            roll_masked = jnp.where(die_idx < level_i32, roll, jnp.int32(0))
            return (new_vrng, total + roll_masked), None

        (vrng_after_dice, total), _ = jax.lax.scan(
            _one_die_v,
            (vendor_rng, jnp.int32(0)),
            jnp.arange(MAX_LEVEL, dtype=jnp.int32),
        )
        # rnd(4) = 1..4 when mlvl == 0.
        vrng_final, zero_roll = randint_jax(vrng_after_dice, (), 1, 5)
        hp = jnp.where(level_i32 == jnp.int32(0), zero_roll, total)
        return vrng_final, jnp.maximum(hp, jnp.int32(1))

    keys = jax.random.split(rng, MAX_LEVEL + 1)

    # d(mlvl, 8) for mlvl >= 1.
    def _one_die(carry, args):
        die_idx, key = args
        roll = jax.random.randint(key, (), minval=1, maxval=9, dtype=jnp.int32)
        roll_masked = jnp.where(die_idx < level_i32, roll, jnp.int32(0))
        return carry + roll_masked, None

    total, _ = jax.lax.scan(
        _one_die,
        jnp.int32(0),
        (jnp.arange(MAX_LEVEL, dtype=jnp.int32), keys[:MAX_LEVEL]),
    )

    # rnd(4) = 1..4 when mlvl == 0.
    zero_roll = jax.random.randint(keys[MAX_LEVEL], (), minval=1, maxval=5,
                                   dtype=jnp.int32)
    hp = jnp.where(level_i32 == jnp.int32(0), zero_roll, total)
    return jnp.maximum(hp, jnp.int32(1))


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
# JIT shape: fixed-cap fori_loop with N=8 weapon-cascade draws (covers the
# vendor median of 5-7 draws; over-consumes slightly for short branches but
# stays deterministic per-call which is what byte-replay requires).
# ---------------------------------------------------------------------------

_INITWEAP_CASCADE_CAP = 8  # static upper bound on per-monster weapon draws


def _consume_makemon_post_hp_draws(vrng, type_id):
    """Consume the post-newmonhp RNG cascade for one monster.

    Mirrors vendor/nle/src/makemon.c lines 1214-1383 in draw order:
      * line 1226: ``mtmp->female = rn2(2)`` — drawn iff non-neuter and not
        explicitly gendered (M2_MALE / M2_FEMALE).  Our flag table folds
        is_male/is_female into "gendered" — only neuter monsters skip the
        draw.  (Vendor draws for gendered monsters too if msound is not
        LEADER/NEMESIS; we approximate by always drawing when !neuter.)
      * lines 1382 + 182-558: ``m_initweap`` cascade if is_armed(ptr).
        Capped at _INITWEAP_CASCADE_CAP=8 sequential rn2() draws, matching
        the average 5-7 draws per armed branch (orcs/soldiers/giants).
        Plus the trailing ``rn2(75)`` at line 556.
      * lines 1383 + 794/796/798: ``m_initinv`` unconditional tail —
        three rn2() draws (rn2(50), rn2(100), rn2(5)) for every monster
        regardless of class.

    Args:
        vrng:   Isaac64State
        type_id: scalar int32 monster entry index.

    Returns:
        new Isaac64State with the appropriate draws consumed.
    """
    tid = type_id.astype(jnp.int32)
    is_neuter = _IS_NEUTER[tid]
    is_armed = _IS_ARMED[tid]

    # --- 1. female draw — vendor makemon.c:1226 ---
    def _draw_female(v):
        new_v, _ = randint_jax(v, (), 0, 2)
        return new_v

    vrng = jax.lax.cond(is_neuter, lambda v: v, _draw_female, vrng)

    # --- 2. m_initweap cascade — vendor makemon.c:1382 + lines 182-558 ---
    # Conservative fixed-cap consumer: N=8 rn2(2) draws covering the
    # average orc/soldier/elf branch depth.  The exact bound varies by
    # mlet but stays within [3, 8] for Dlvl-1-eligible armed species.
    def _draw_initweap(v):
        def _body(_, vc):
            new_v, _r = randint_jax(vc, (), 0, 2)
            return new_v
        v_after = jax.lax.fori_loop(0, _INITWEAP_CASCADE_CAP, _body, v)
        # Trailing offensive-item check — vendor makemon.c:556
        #   if ((int) mtmp->m_lev > rn2(75)) (void) mongets(...)
        v_after, _ = randint_jax(v_after, (), 0, 75)
        return v_after

    vrng = jax.lax.cond(is_armed, _draw_initweap, lambda v: v, vrng)

    # --- 3. m_initinv unconditional tail — vendor makemon.c:794, 796, 798 ---
    #   if ((int) mtmp->m_lev > rn2(50))  → rnd_defensive_item
    #   if ((int) mtmp->m_lev > rn2(100)) → rnd_misc_item
    #   if (likes_gold && !findgold && !rn2(5))  → mkmonmoney
    vrng, _ = randint_jax(vrng, (), 0, 50)
    vrng, _ = randint_jax(vrng, (), 0, 100)
    vrng, _ = randint_jax(vrng, (), 0, 5)

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
            level = MONSTR_DIFFICULTIES[type_id]
            vrng, hp = _roll_hp(hp_keys[i], level, vendor_rng=vrng)
            # Consume vendor makemon.c:1214-1383 post-newmonhp draws:
            #   * line 1226: rn2(2) female flag (non-neuter monsters)
            #   * line 1382 + 182-558: m_initweap cascade (is_armed monsters)
            #   * line 1383 + 794/796/798: m_initinv unconditional tail
            vrng = _consume_makemon_post_hp_draws(vrng, type_id)
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
        hp = _roll_hp(hp_keys[i], level)
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

def populate_level_with_monsters(
    state,
    rng: jax.Array,
    n_monsters: int = 5,
    n_rooms: int | None = None,
    vendor_rng=None,
) -> object:
    """Spawn monsters into state.monster_ai slots [0, n_monsters).

    Reads terrain from state.terrain[branch=0, level=0] (current level).
    Valid spawn tiles: FLOOR or CORRIDOR, not the player's starting position.

    Writes into the first n_monsters slots of state.monster_ai.

    Audit-N #6 (mklev.c:804): when ``n_rooms`` is provided the monster
    count is recomputed as ``rnd((nroom >> 1) + 1)`` instead of the fixed
    ``n_monsters`` default.  ``rnd(N) = randint(1, N+1)`` is sampled from
    a sub-key of ``rng``.

    Byte-replay path: when ``vendor_rng`` (Isaac64State) is supplied, the
    monster-count roll and the per-monster HP rolls are routed through
    :func:`vendor_rng.randint_jax`, and the returned state has its
    ``vendor_rng`` field replaced with the updated Isaac64State.  This
    function remains host-side (called from env.reset).

    Args:
        state:      EnvState.
        rng:        JAX PRNG key.
        n_monsters: fallback count when ``n_rooms`` is None (default 5).
        n_rooms:    optional int — current-level active room count.  When
                    set, ``ct = rnd((nroom >> 1) + 1)`` overrides
                    ``n_monsters``.
        vendor_rng: optional Isaac64State for byte-exact NLE replay.
    """
    if n_rooms is not None:
        upper = max(int(n_rooms) >> 1, 0) + 1
        if vendor_rng is not None:
            # Byte-replay: rnd((nroom>>1)+1) via ISAAC64.  rnd(N)=randint(1,N+1).
            vendor_rng, sampled_arr = randint_jax(vendor_rng, (), 1, upper + 1)
            sampled = int(sampled_arr)
        else:
            rng_ct, rng = jax.random.split(rng, 2)
            # rnd(N) = randint(1, N+1).  Per vendor mklev.c:804
            # ct = rnd((svn.nroom >> 1) + 1).
            sampled = int(jax.random.randint(rng_ct, (), 1, upper + 1))
        n_monsters = max(min(sampled, n_monsters), 1)
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

    if vendor_rng is not None:
        vendor_rng, positions, type_ids, hps, max_hps, count = spawn_initial_monsters(
            rng, depth=1, n_monsters=n_monsters, valid_tiles_mask=valid_tiles_mask,
            map_h=map_h, map_w=map_w,
            genocided=state.genocided_species,
            vendor_rng=vendor_rng,
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
    PM_LONG_WORM = jnp.int32(118)

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
