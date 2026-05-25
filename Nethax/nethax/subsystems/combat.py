"""Combat subsystem — melee/ranged/passive attacks, hit rolls, AC, weapon skills.

Canonical sources:
  vendor/nethack/src/uhitm.c   — hero attacks monster (hitum, backstab, cleave)
  vendor/nethack/src/mhitu.c   — monster attacks hero (hitmu, gulpmu, passiveum)
  vendor/nethack/src/mhitm.c   — monster attacks monster (hitmm, mdamagem, passivemm)
  vendor/nethack/src/mthrowu.c — monster ranged/thrown/breath attacks
  vendor/nethack/src/weapon.c  — to-hit/damage bonuses, weapon skill advancement
  vendor/nethack/src/dothrow.c — hero throw ('t'), auto-quiver, launcher selection
  vendor/nethack/src/do_wear.c — find_ac() (do_wear.c:2473)
  vendor/nethack/src/worn.c    — find_mac() (worn.c:717)
  vendor/nethack/include/skills.h — practice_needed_to_advance (line 106)

Status: vendor-parity — AC, to-hit d20, damage roll, melee/bump attack,
monster attack (per-instance ``m_lev``), skill practice advancement, ranged
/ throw / breath attacks, engulf / passive, polymorph combat, two-weapon.

JAX-required divergences (documented inline at each call site):
    - kill-count gating in XP award (vendor uses per-PM ``mvitals[pm].died``,
      a counter we do not maintain; total ``scoring.monsters_killed`` is used)
"""
import jax
import jax.numpy as jnp
from flax import struct

from Nethax.nethax.constants import Action  # noqa: F401 — reserved for action dispatch
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.rng import dice_roll, rnd, split_n
from Nethax.nethax.subsystems.inventory import (
    N_ARMOR_SLOTS,
    ItemCategory,
    compute_ac as _inv_compute_ac,
)
from Nethax.nethax.subsystems.items_potions import apply_potion_to_monster
from Nethax.nethax.subsystems.scoring import (
    record_kill as _scoring_record_kill,
    record_kill_pm as _scoring_record_kill_pm,
    died_for_pm as _scoring_died_for_pm,
)
from Nethax.nethax.subsystems.experience import (
    experience as _xp_experience,
    more_experienced as _xp_more_experienced,
)
# Module-level imports to avoid lazy-import tracer leaks (module-level jnp.array
# builds must happen outside any JIT trace).  Cite: tests show wave9 inline
# imports inside _single_melee_strike trigger jax.errors.UnexpectedTracerError.
from Nethax.nethax.subsystems.weapon_dice import weapon_damage_dice as _wdd
from Nethax.nethax.subsystems.artifact_powers import (
    artifact_bonus_damage as _arti_bonus,
    wielded_artifact_idx_from_state as _arti_idx,
    apply_artifact_hit_effects as _arti_hit_effects,
)
from Nethax.nethax.subsystems.throwing import (
    _HATES_SILVER,
    _OBJECT_MATERIAL,
    _IS_RETURNING_WEAPON,
    compute_throw_range,
    vendor_breaktest,
    _OTYP_MIRROR,
    _OTYP_EGG,
    _OTYP_EXPENSIVE_CAMERA,
    _PM_PYROLISK,
)
from Nethax.nethax.constants.objects import Material
from Nethax.nethax.rng import rn2 as _rn2
from Nethax.nethax.subsystems.skills import (
    use_skill as _skills_use_skill,
    _WEAPON_TYPE_TO_SKILL as _SKILL_WEAPON_TYPE_TO_SKILL,
)


# ---------------------------------------------------------------------------
# Monster XP lookup table — mirrors MONSTERS[i].level used as the XP award.
# Vendor reference: vendor/nethack/src/mon.c::experience() — the XP a monster
# is worth is proportional to its level (monst.c::permonst.mlevel).
# Built once at module load (same pattern as _MONSTER_SYMBOL_TABLE in this
# file and _MONSTER_LEVEL_TABLE in monster_ai.py).
# ---------------------------------------------------------------------------
def _build_monster_xp_table() -> jnp.ndarray:
    from Nethax.nethax.constants.monsters import MONSTERS
    return jnp.array([int(m.level) for m in MONSTERS], dtype=jnp.int32)

_MONSTER_XP_TABLE: jnp.ndarray = _build_monster_xp_table()


# ---------------------------------------------------------------------------
# Corpse-drops table — vendor/nethack/src/mondead.c::xkilled.
# Most monsters leave a corpse; exceptions: elementals (S_ELEMENTAL=31),
# ghosts (S_GHOST=54), vortices (S_VORTEX=22), and shades (S_SHADE=55).
# vendor/nethack/src/mondead.c (xkilled, corpse-generation block).
# ---------------------------------------------------------------------------
def _build_killed_drops_corpse_table() -> jnp.ndarray:
    from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol
    _NO_CORPSE_SYMS = {
        int(MonsterSymbol.S_ELEMENTAL),
        int(MonsterSymbol.S_GHOST),
        int(MonsterSymbol.S_VORTEX),
    }
    # S_SHADE may not exist in all builds; guard with getattr.
    shade_val = getattr(MonsterSymbol, "S_SHADE", None)
    if shade_val is not None:
        _NO_CORPSE_SYMS.add(int(shade_val))
    return jnp.array(
        [int(m.symbol) not in _NO_CORPSE_SYMS for m in MONSTERS],
        dtype=jnp.bool_,
    )

_KILLED_DROPS_CORPSE: jnp.ndarray = _build_killed_drops_corpse_table()

# Sentinel type_id for a corpse item (FOOD class, vendor objects.h index 240).
# Mirrors nle_obs.py::CORPSE_TYPE_ID = 260.
_CORPSE_TYPE_ID: int = 260

# ItemCategory.FOOD value = 7 (inventory.py ItemCategory enum).
_FOOD_CATEGORY: int = 7


# ---------------------------------------------------------------------------
# Monster primary-attack damage-type table — adtyp of attack[0] per entry.
# Vendor reference: vendor/nethack/src/uhitm.c::mhitm_ad_were (line 4265);
# src/were.c::set_ulycn (line 234).  Used to dispatch AD_WERE infection.
# Built once at module load; never traced inside a jit boundary.
# ---------------------------------------------------------------------------
def _build_monster_primary_adtyp_table() -> jnp.ndarray:
    from Nethax.nethax.constants.monsters import MONSTERS
    return jnp.array(
        [int(m.attacks[0][1]) if m.attacks else 0 for m in MONSTERS],
        dtype=jnp.int32,
    )

_MONSTER_PRIMARY_ADTYP_TABLE: jnp.ndarray = _build_monster_primary_adtyp_table()


# ---------------------------------------------------------------------------
# Per-monster MONSTERS-table fields needed by JIT-side spawn paths
# (release_camera_demon, summon scrolls).  Each cell holds the species-level
# default for one MONSTERS row; spawn paths copy them into monster_ai.
# Vendor refs: include/monst.h (struct permonst), src/makemon.c::newmonhp.
# ---------------------------------------------------------------------------
def _build_monster_spawn_tables():
    from Nethax.nethax.constants.monsters import MONSTERS
    levels = jnp.array([int(m.level) for m in MONSTERS], dtype=jnp.int16)
    acs    = jnp.array([int(m.ac)    for m in MONSTERS], dtype=jnp.int16)
    # int16 because some "dice" values overflow int8 (e.g. AT_BREA sentinel
    # rows where damd can be 255).  Camera demon ranks (homunculus / imp)
    # fit easily in int8 but we keep the broader type for table uniformity.
    atk_n  = jnp.array(
        [int(m.attacks[0][2]) if m.attacks else 1 for m in MONSTERS],
        dtype=jnp.int16,
    )
    atk_s  = jnp.array(
        [int(m.attacks[0][3]) if m.attacks else 1 for m in MONSTERS],
        dtype=jnp.int16,
    )
    return levels, acs, atk_n, atk_s


(_MONSTER_SPAWN_LEVEL,
 _MONSTER_SPAWN_AC,
 _MONSTER_SPAWN_ATK_N,
 _MONSTER_SPAWN_ATK_S) = _build_monster_spawn_tables()

# Vendor MONSTERS indices for release_camera_demon (dothrow.c:2461).
# Verified against Nethax.nethax.constants.monster_entries.chunk1 lines
# 934-967 (homunculus row 51, imp row 52).
_PM_HOMUNCULUS: int = 51
_PM_IMP:        int = 52

# AD_WERE value as a module-level int constant (mirrors DamageType.AD_WERE=29).
_AD_WERE: int = 29

# ---------------------------------------------------------------------------
# Engulfer table — True for monsters whose attack list includes AT_ENGL.
# Vendor: vendor/nethack/src/mhitu.c::gulpmu (line 1287).
# Imported from swallow.py; referenced here for the monster_attack_player hook.
# ---------------------------------------------------------------------------
from Nethax.nethax.subsystems.swallow import _IS_ENGULFER as _ENGULFER_TABLE, try_engulf as _try_engulf  # noqa: E402


# ---------------------------------------------------------------------------
# Immobile-monster mask — vendor/nethack/src/uhitm.c:393-394:
#   if (!mtmp->mcanmove) tmp += 4;
# Structural immobility: move_speed == 0 (e.g. brown mold, blue jelly).
# Runtime paralysis is handled at the use site via
# ``mai.paralyzed_timer[idx] > 0`` (see ``compute_to_hit`` below), which
# is byte-equal with vendor ``!mtmp->mcanmove`` (mcanmove flips false
# only while the FROZEN / paralysis timer is active).
# Built once at module load; never traced inside JIT.
# ---------------------------------------------------------------------------
def _build_is_immobile_table() -> jnp.ndarray:
    from Nethax.nethax.constants.monsters import MONSTERS
    return jnp.array([int(m.move_speed) == 0 for m in MONSTERS], dtype=jnp.bool_)

_IS_IMMOBILE: jnp.ndarray = _build_is_immobile_table()


# ---------------------------------------------------------------------------
# Role-specific weapon type ids (vendor/nethack/include/objects.h NAMING block;
# index into Nethax.nethax.constants.objects.WEAPONS).
# ---------------------------------------------------------------------------
# 14 — short sword (NetHack maps the samurai "wakizashi" onto SHORT_SWORD)
# 21 — katana       ("samurai sword")
# 40 — yumi         ("long bow")
WEAPON_TYPE_SHORT_SWORD: int = 14
WEAPON_TYPE_KATANA: int = 21
WEAPON_TYPE_YUMI: int = 40


# ---------------------------------------------------------------------------
# Static monster-symbol lookup (used for the Knight chivalric check).
# Built once at module import; never traces inside a jit boundary.
# Mirrors the table assembled in items_scrolls._build_monster_symbol_table.
# ---------------------------------------------------------------------------
def _build_monster_symbol_table() -> jnp.ndarray:
    from Nethax.nethax.constants.monsters import MONSTERS
    return jnp.array([int(m.symbol) for m in MONSTERS], dtype=jnp.int8)


_MONSTER_SYMBOL_TABLE: jnp.ndarray = _build_monster_symbol_table()

# vendor/nethack/include/defsym.h: S_HUMANOID=8, S_HUMAN=53.
_S_HUMANOID: int = 8
_S_HUMAN: int = 53


# ---------------------------------------------------------------------------
# Weapon skill tiers (vendor/nethack/include/skills.h)
# ---------------------------------------------------------------------------
SKILL_UNSKILLED = 0
SKILL_BASIC = 1
SKILL_SKILLED = 2
SKILL_EXPERT = 3
SKILL_MASTER = 4
SKILL_GRAND_MASTER = 5
N_SKILL_TIERS = 6

# Weapon-skill slots per character.
# ~30 weapon types + 8 spell schools + martial arts + two-weapon.
N_WEAPON_SKILLS = 40

# Practice required to advance to the next tier
# (vendor/nethack/include/skills.h:106 — practice_needed_to_advance(level) =
#  level * level * 20).  Vendor's macro is keyed by its 1-based skill level
# (P_UNSKILLED=1 … P_GRAND_MASTER=6) so the vendor-byte-equal output for our
# 0-based tier ``t`` is (t+1)^2 * 20.  Indexed by current tier.
_PRACTICE_TO_ADVANCE = jnp.array(
    [(tier + 1) * (tier + 1) * 20 for tier in range(N_SKILL_TIERS)],
    dtype=jnp.int32,
)

# Per-tier to-hit bonus.
# Bit-equal to vendor/nethack/src/weapon.c:1545-1577 (weapon_hit_bonus) for
# normal weapon skills:
#   P_ISRESTRICTED / P_UNSKILLED → -4
#   P_BASIC                       →  0
#   P_SKILLED                     → +2
#   P_EXPERT                      → +3
# Master / Grand Master are not reachable for ordinary weapons (only bare-
# handed combat); we extend the table with +3 sentinels so an out-of-range
# tier does not produce a spurious bonus.
# vendor/nethack/src/weapon.c:1566-1577 (weapon_hit_bonus table).
_SKILL_HIT_BONUS = jnp.array([-4, 0, 2, 3, 3, 3], dtype=jnp.int32)

# Per-tier damage bonus.
# Bit-equal to vendor/nethack/src/weapon.c:1656-1675 (weapon_dam_bonus):
#   P_ISRESTRICTED / P_UNSKILLED → -2
#   P_BASIC                       →  0
#   P_SKILLED                     → +1
#   P_EXPERT                      → +2
# Master / Grand Master sentinels at +2 (unreachable for ordinary weapons).
# vendor/nethack/src/weapon.c:1662-1675 (weapon_dam_bonus table).
_SKILL_DAM_BONUS = jnp.array([-2, 0, 1, 2, 2, 2], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# STR/DEX bonus tables (vendor/nethack/src/weapon.c::abon / dbon)
#
# NetHack encodes "exceptional" Strength on the 18/xx scale via the STR18
# macro (vendor/nethack/include/attrib.h:36 — STR18(x) = 18 + x).  So
#   18      = "18"          (raw 18)
#   18+50   = "18/50"       (STR18(50))
#   18+100  = "18/100"      (STR18(100), == 19 internally)
# In Nethax player_str is the same flat 0..125 integer.
# ---------------------------------------------------------------------------
def _adj_lev(state, m_lev: jnp.ndarray) -> jnp.ndarray:
    """Vendor-parity ``adj_lev`` for the player's current poly form.

    Mirror of vendor/nethack/src/makemon.c:2014-2046 — adjusts a monster's
    effective level based on dungeon depth and player XL.  Used by ``_abon``
    when ``Upolyd``.  We omit the PM_WIZARD_OF_YENDOR death-counter branch
    (the player cannot polymorph into the Wizard of Yendor).
    """
    tmp = m_lev.astype(jnp.int32)
    over_special = tmp > jnp.int32(49)
    # level_difficulty() proxy — vendor dungeon.c:2027.
    lvl_diff = state.dungeon.current_level.astype(jnp.int32)
    tmp2 = lvl_diff - tmp
    tmp_after_diff = jnp.where(
        tmp2 < jnp.int32(0),
        tmp - jnp.int32(1),
        tmp + (tmp2 // jnp.int32(5)),
    )
    xl_diff = state.player_xl.astype(jnp.int32) - m_lev.astype(jnp.int32)
    tmp_after_xl = tmp_after_diff + jnp.where(
        xl_diff > jnp.int32(0),
        xl_diff // jnp.int32(4),
        jnp.int32(0),
    )
    tmp2_cap = (jnp.int32(3) * m_lev.astype(jnp.int32)) // jnp.int32(2)
    tmp2_cap = jnp.minimum(tmp2_cap, jnp.int32(49))
    capped = jnp.minimum(tmp_after_xl, tmp2_cap)
    floored = jnp.maximum(capped, jnp.int32(0))
    return jnp.where(over_special, jnp.int32(50), floored).astype(jnp.int32)


def _abon(player_str: jnp.ndarray, player_dex: jnp.ndarray,
          player_xl: jnp.ndarray, state=None) -> jnp.ndarray:
    """Attack (to-hit) bonus for STR & DEX.

    Bit-equal mirror of vendor/nethack/src/weapon.c:950-988 ``abon``:

        if (Upolyd) return adj_lev(&mons[u.umonnum]) - 3;
        ... STR/DEX table ...

    When ``state`` is supplied and the player is polymorphed, we route
    through ``_adj_lev`` against the current poly form's ``m_lev``.
    """
    s = player_str.astype(jnp.int32)
    sbon = jnp.where(
        s < 6, -2,
        jnp.where(
            s < 8, -1,
            jnp.where(
                s < 17, 0,
                jnp.where(
                    s < 18 + 50, 1,
                    jnp.where(
                        s < 18 + 100, 2,
                        3,
                    ),
                ),
            ),
        ),
    )
    # Low-level fudge (weapon.c:977).
    sbon = sbon + jnp.where(player_xl.astype(jnp.int32) < 3, 1, 0)

    d = player_dex.astype(jnp.int32)
    dex_bonus = jnp.where(
        d < 4, -3,
        jnp.where(
            d < 6, -2,
            jnp.where(
                d < 8, -1,
                jnp.where(d < 14, 0, d - 14),
            ),
        ),
    )
    base = (sbon + dex_bonus).astype(jnp.int32)
    if state is None:
        return base
    poly = getattr(state, "polymorph", None)
    if poly is None or not hasattr(poly, "is_polymorphed"):
        return base
    # vendor weapon.c:955-956 — Upolyd path: adj_lev(&mons[u.umonnum]) - 3.
    form_idx = poly.current_form_idx.astype(jnp.int32)
    from Nethax.nethax.subsystems.monster_ai import _monster_level
    form_mlev = _monster_level(form_idx)
    poly_bonus = (_adj_lev(state, form_mlev) - jnp.int32(3)).astype(jnp.int32)
    return jnp.where(poly.is_polymorphed, poly_bonus, base).astype(jnp.int32)


def _dbon(player_str: jnp.ndarray) -> jnp.ndarray:
    """Damage bonus for STR.

    Mirror of vendor/nethack/src/weapon.c:992-1015.
    """
    s = player_str.astype(jnp.int32)
    return jnp.where(
        s < 6, -1,
        jnp.where(
            s < 16, 0,
            jnp.where(
                s < 18, 1,
                jnp.where(
                    s == 18, 2,
                    jnp.where(
                        s <= 18 + 75, 3,
                        jnp.where(
                            s <= 18 + 90, 4,
                            jnp.where(s < 18 + 100, 5, 6),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Pure-Python parity helpers (Wave 6 Phase B+).
#
# These functions are bit-equal to the corresponding vendor C formulas.  They
# accept plain integers (not JAX arrays) and return plain integers so the
# parity tests can compare against the vendor table without going through the
# JIT-trace path.  The runtime combat code keeps using the array helpers
# above (_abon / _dbon) which match these definitions exactly.
# ---------------------------------------------------------------------------
def strhitbon(str_value: int) -> int:
    """STR contribution to attack roll, matching vendor weapon.c:962-973.

    # vendor/nethack/src/weapon.c:962-973 (abon, STR portion).
    """
    s = int(str_value)
    if s < 6:
        return -2
    if s < 8:
        return -1
    if s < 17:
        return 0
    if s < 18 + 50:    # STR18(50)
        return 1
    if s < 18 + 100:   # STR18(100)
        return 2
    return 3


def strdambon(str_value: int) -> int:
    """STR contribution to damage roll, matching vendor weapon.c:1000-1015.

    # vendor/nethack/src/weapon.c:1000-1015 (dbon).
    """
    s = int(str_value)
    if s < 6:
        return -1
    if s < 16:
        return 0
    if s < 18:
        return 1
    if s == 18:
        return 2
    if s <= 18 + 75:   # STR18(75)
        return 3
    if s <= 18 + 90:   # STR18(90)
        return 4
    if s < 18 + 100:   # STR18(100)
        return 5
    return 6


def dexbon(dex_value: int) -> int:
    """DEX contribution to attack roll (additive on top of strhitbon).

    # vendor/nethack/src/weapon.c:979-988 (abon, DEX portion).
    """
    d = int(dex_value)
    if d < 4:
        return -3
    if d < 6:
        return -2
    if d < 8:
        return -1
    if d < 14:
        return 0
    return d - 14


def weapon_skill_hit_bonus(skill_tier: int) -> int:
    """Bit-equal to vendor weapon.c::weapon_hit_bonus for ordinary weapons.

    Maps skill tier (0..5 = unskilled..grand master) to the attack-roll
    bonus.  Only Basic..Expert are reachable for ordinary weapons; higher
    tiers are clamped to Expert.

    # vendor/nethack/src/weapon.c:1545-1577 (weapon_hit_bonus weapon table).
    """
    tier = int(skill_tier)
    table = (-4, 0, 2, 3, 3, 3)
    if tier < 0:
        return table[0]
    if tier >= len(table):
        return table[-1]
    return table[tier]


def weapon_skill_dam_bonus(skill_tier: int) -> int:
    """Bit-equal to vendor weapon.c::weapon_dam_bonus for ordinary weapons.

    Maps skill tier (0..5) to the damage bonus.  Only Basic..Expert are
    reachable for ordinary weapons; higher tiers clamp to Expert.

    # vendor/nethack/src/weapon.c:1644-1675 (weapon_dam_bonus weapon table).
    """
    tier = int(skill_tier)
    table = (-2, 0, 1, 2, 2, 2)
    if tier < 0:
        return table[0]
    if tier >= len(table):
        return table[-1]
    return table[tier]


def find_ac_formula(base_ac: int, worn_arm_bonuses) -> int:
    """Bit-equal to vendor do_wear.c::find_ac, ignoring rings/intrinsics.

    ``uac = base_ac - sum(ARM_BONUS(uarm[i]))``.  The vendor function then
    clamps |uac| to AC_MAX; we expose the raw formula (the caller can clamp
    if desired) so parity tests can verify the unclamped result.

    # vendor/nethack/src/do_wear.c:2473-2495 (find_ac armour loop).
    """
    total_bonus = 0
    for b in worn_arm_bonuses:
        total_bonus += int(b)
    return int(base_ac) - total_bonus


def monster_to_hit_tmp(player_uac: int, monster_m_lev: int) -> int:
    """Bit-equal to vendor mhitu.c::mattacku to-hit accumulator (no Luck /
    invisibility / sleep modifiers — the canonical positive-AC case).

    The hit check that follows is ``hit_iff tmp > rnd(20)`` (vendor uses the
    strict-greater comparison; the prior Nethax code used ``<=`` which was
    off by one).  Negative AC in vendor is randomised through ``AC_VALUE``
    (hack.h:1538) which we omit here — callers feed the deterministic raw
    AC.  ``tmp`` is clamped to >=1 (mhitu.c:717-718).

    # vendor/nethack/src/mhitu.c:709-718 (mattacku tmp accumulator).
    """
    uac = int(player_uac)
    # vendor/nethack/src/uhitm.c:380 — AC_VALUE macro (hack.h:1538):
    # AC>=0: use AC directly; AC<0: rnd(-AC) softens the bonus.
    # Pure-Python path: use deterministic midpoint for negative AC.
    if uac >= 0:
        ac_value = uac
    else:
        import random as _random
        ac_value = _random.randint(1, -uac)
    tmp = ac_value + 10 + int(monster_m_lev)
    return tmp if tmp > 0 else 1


def find_roll_to_hit_formula(
    str_value: int,
    dex_value: int,
    monster_ac: int,
    skill_tier: int = 1,
    weapon_enchant: int = 0,
    xl: int = 1,
) -> int:
    """Composite vendor ``find_roll_to_hit`` for a melee weapon attack
    (uhitm.c:365-427), restricted to the STR/DEX/AC/skill/enchant terms that
    Nethax models.  Hit iff ``tmp > rnd(20)``.

    # vendor/nethack/src/uhitm.c:376 (tmp accumulator)
    # vendor/nethack/src/uhitm.c:418-424 (weapon_hit_bonus add)
    # vendor/nethack/src/uhitm.c:709 (hit comparison: tmp > dieroll)
    """
    abon = strhitbon(str_value) + dexbon(dex_value)
    # Game-tuning kludge from weapon.c:977 (+1 to-hit while XL < 3).
    if int(xl) < 3:
        abon += 1
    return (
        1
        + abon
        + int(monster_ac)
        + weapon_skill_hit_bonus(skill_tier)
        + int(weapon_enchant)
    )


def dmgval_weapon(
    bigmonst: bool,
    sdam_roll: int,
    ldam_roll: int,
    spe: int = 0,
    is_weapon: bool = True,
) -> int:
    """Bit-equal to the core branch of vendor ``dmgval`` (weapon.c:215-302)
    for a vanilla weapon (no special otyp bonus, no material-vs-skin gating,
    no SHADE / HEAVY_IRON_BALL special cases).

    Parameters
    ----------
    bigmonst   : True if the target is a large monster (bigmonst()).
    sdam_roll  : the rolled value of ``rnd(oc_wsdam)`` (used when !bigmonst).
    ldam_roll  : the rolled value of ``rnd(oc_wldam)`` (used when bigmonst).
    spe        : the weapon's ``otmp->spe`` enchantment.
    is_weapon  : True for WEAPON_CLASS / is_weptool() items; controls the
                 enchantment add and the ``tmp < 0 → 0`` clamp.

    # vendor/nethack/src/weapon.c:215-302 (dmgval core).
    """
    if bigmonst:
        tmp = int(ldam_roll)
    else:
        tmp = int(sdam_roll)
    if is_weapon:
        tmp += int(spe)
        if tmp < 0:
            tmp = 0  # weapon.c:300-302
    return tmp


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@struct.dataclass
class CombatState:
    """Persistent combat-related player state.

    Fields
    ------
    weapon_skill     : current skill tier per weapon type (int8)
    weapon_practice  : practice counter toward next tier (int32)
    last_attack_kind : encoded attack type for combo/passive-trigger logic (int32)
    last_hit_landed  : whether the most recent attack connected (bool)
    two_weapon       : Wave 5 — two-weapon combat toggle (bool).
                       Mirrors u.twoweap (vendor/nethack/src/wield.c::dotwoweapon).
                       When True, melee_attack does TWO to-hit rolls.
    """

    weapon_skill: jnp.ndarray      # [N_WEAPON_SKILLS]  int8
    weapon_practice: jnp.ndarray   # [N_WEAPON_SKILLS]  int32
    last_attack_kind: jnp.ndarray  # scalar             int32
    last_hit_landed: jnp.ndarray   # scalar             bool
    two_weapon: jnp.ndarray        # scalar             bool

    @classmethod
    def default(cls) -> "CombatState":
        """Return a zeroed CombatState for a freshly created character."""
        return cls(
            weapon_skill=jnp.zeros((N_WEAPON_SKILLS,), dtype=jnp.int8),
            weapon_practice=jnp.zeros((N_WEAPON_SKILLS,), dtype=jnp.int32),
            last_attack_kind=jnp.int32(0),
            last_hit_landed=jnp.bool_(False),
            two_weapon=jnp.bool_(False),
        )


# ---------------------------------------------------------------------------
# AC computation (vendor/nethack/src/do_wear.c::find_ac, lines 2473-2525)
# ---------------------------------------------------------------------------
PLAYER_BASE_AC: int = 10  # human base AC; matches mons[PM_HUMAN].ac in monst.c


def compute_ac(state) -> jnp.ndarray:
    """Compute hero AC.

    Convenience wrapper around ``inventory.compute_ac`` that takes the full
    EnvState.  Sums ARM_BONUS over the 7 worn-armor slots (body, shield,
    helm, gloves, boots, cloak, shirt) and subtracts from base AC = 10.
    Mirrors vendor/nethack/src/do_wear.c:2473-2525 (find_ac).

    Wave 5: also reads ``state.inventory.worn_armor_ac_bonus`` (the cached
    per-slot AC bonus updated by wear_armor / take_off_armor), allowing
    bonuses to be applied even without populating Item records.

    Polymorph (Wave 5): if ``state.polymorph.is_polymorphed`` is True, the
    form's intrinsic AC replaces the armor-derived AC entirely (mirrors
    vendor/nethack/src/polyself.c::find_uac which uses mptr->ac directly
    when the player is polymorphed).

    Lower AC = better.  Stripped player → 10.  Internally uses ``lax.scan``.
    """
    # Optional cached per-slot AC bonus (Wave 5).  Older InventoryState
    # snapshots without the field fall back to the items-derived bonus.
    cache = getattr(state.inventory, "worn_armor_ac_bonus", None)
    armor_ac = _inv_compute_ac(
        state.inventory.items, state.inventory.worn_armor, cache,
    ).astype(jnp.int32)

    poly = getattr(state, "polymorph", None)
    if poly is None or not hasattr(poly, "is_polymorphed"):
        return armor_ac

    is_poly = poly.is_polymorphed
    form_ac = state.player_ac.astype(jnp.int32)
    return jnp.where(is_poly, form_ac, armor_ac)


# ---------------------------------------------------------------------------
# Wielded-weapon helpers
# ---------------------------------------------------------------------------
def _wielded_enchant(state) -> jnp.ndarray:
    """Return the enchantment of the wielded weapon (0 if bare-handed)."""
    wielded = state.inventory.wielded.astype(jnp.int32)
    safe = jnp.clip(wielded, 0, state.inventory.items.enchantment.shape[0] - 1)
    enchant = state.inventory.items.enchantment[safe].astype(jnp.int32)
    return jnp.where(wielded >= 0, enchant, jnp.int32(0))


def _wielded_skill_id(state) -> jnp.ndarray:
    """Return the weapon-skill id of the wielded weapon.

    Bit-equal mirror of vendor/nethack/src/weapon.c::weapon_type (lines
    1426-1450) — looks up ``objects[otmp->otyp].oc_skill`` and falls back
    to ``P_BARE_HANDED_COMBAT`` when no weapon is wielded.  Routes through
    the static ``_SKILL_WEAPON_TYPE_TO_SKILL`` table built from each
    object's ``oc_skill`` in OBJECTS (vendor include/objclass.h —
    ``oc_subtyp`` for weapons).
    """
    wielded = state.inventory.wielded.astype(jnp.int32)
    safe = jnp.clip(wielded, 0, state.inventory.items.type_id.shape[0] - 1)
    type_id = state.inventory.items.type_id[safe].astype(jnp.int32)
    safe_type = jnp.clip(type_id, 0, _SKILL_WEAPON_TYPE_TO_SKILL.shape[0] - 1)
    mapped = _SKILL_WEAPON_TYPE_TO_SKILL[safe_type].astype(jnp.int32)
    # Bare-handed slot — vendor uses P_BARE_HANDED_COMBAT (SkillId.MARTIAL_ARTS
    # in Nethax indexing).  Lazy-imported to avoid module-load cycles.
    from Nethax.nethax.subsystems.skills import SkillId
    bare = jnp.int32(int(SkillId.MARTIAL_ARTS))
    skill_id = jnp.where(wielded >= 0, mapped, bare)
    return skill_id.astype(jnp.int32)


def _skill_hit_bonus(state) -> jnp.ndarray:
    """Return the current to-hit bonus from the wielded weapon's skill tier.

    Reads from state.skills.level[skill_id] (SkillState) using the canonical
    weapon type_id → SkillId mapping (_SKILL_WEAPON_TYPE_TO_SKILL).
    Falls back to CombatState.weapon_skill when SkillState is absent.
    Cite: vendor/nethack/src/weapon.c:1566-1577 (weapon_hit_bonus table).
    """
    skills = getattr(state, "skills", None)
    if skills is not None:
        wep_type = _wielded_type_id(state)
        safe_type = jnp.clip(wep_type.astype(jnp.int32), 0, _SKILL_WEAPON_TYPE_TO_SKILL.shape[0] - 1)
        skill_id = _SKILL_WEAPON_TYPE_TO_SKILL[safe_type]
        tier = skills.level[skill_id].astype(jnp.int32)
    else:
        skill_id = _wielded_skill_id(state)
        tier = state.combat.weapon_skill[skill_id].astype(jnp.int32)
    safe_tier = jnp.clip(tier, 0, N_SKILL_TIERS - 1)
    return _SKILL_HIT_BONUS[safe_tier].astype(jnp.int32)


# ---------------------------------------------------------------------------
# To-hit roll (vendor/nethack/src/uhitm.c::find_roll_to_hit, lines 365-427)
# ---------------------------------------------------------------------------
def _compute_encumbrance(state) -> jnp.ndarray:
    """Compute encumbrance level (0=UNENCUMBERED .. 5=OVERLOADED).

    Bit-equal mirror of vendor/nethack/src/hack.c:
      * ``weight_cap`` (hack.c:4294-4346) —
            carrcap = WT_WEIGHTCAP_STRCON * (ACURRSTR + ACURR(A_CON))
                      + WT_WEIGHTCAP_SPARE
            min(carrcap, MAX_CARR_CAP), then max(carrcap, 1)
        Constants from vendor/nethack/include/weight.h:
          WT_WEIGHTCAP_STRCON = 25, WT_WEIGHTCAP_SPARE = 50,
          MAX_CARR_CAP = 1000.
      * ``calc_capacity`` (hack.c:4371-4382) —
            wt = inv_weight() = sum(item weight) - carrcap
            if wt <= 0:       return UNENCUMBERED (0)
            if carrcap <= 1:  return OVERLOADED   (5)
            return min((wt * 2 / carrcap) + 1, OVERLOADED)
      * ``near_capacity`` (hack.c:4385-4388) = calc_capacity(0)

    Vendor STR uses the ``ACURRSTR`` macro (attrib.c:1245-1262 acurrstr)
    which collapses 18/01..125 into 3..25 before the multiply.

    # JAX-required divergence: the Upolyd / Levitation / Wounded-legs /
    # Steed branches of vendor weight_cap are not modelled — the player
    # neither rides a steed nor has a wounded-leg state field today.
    """
    items = state.inventory.items
    weight_total = jnp.sum(
        items.weight.astype(jnp.int32) * items.quantity.astype(jnp.int32)
    ).astype(jnp.int32)

    # ACURRSTR mapping (vendor attrib.c:1245-1262).
    raw_str = state.player_str.astype(jnp.int32)
    s_le18 = jnp.maximum(raw_str, jnp.int32(3))
    s_le121 = jnp.int32(19) + raw_str // jnp.int32(50)
    s_high = jnp.minimum(raw_str, jnp.int32(125)) - jnp.int32(100)
    acurr_str = jnp.where(
        raw_str <= jnp.int32(18), s_le18,
        jnp.where(raw_str <= jnp.int32(121), s_le121, s_high),
    )

    con = state.player_con.astype(jnp.int32)
    carrcap = jnp.int32(25) * (acurr_str + con) + jnp.int32(50)
    carrcap = jnp.minimum(carrcap, jnp.int32(1000))
    carrcap = jnp.maximum(carrcap, jnp.int32(1))

    wt = weight_total - carrcap
    overloaded = carrcap <= jnp.int32(1)
    enc_calc = (wt * jnp.int32(2)) // carrcap + jnp.int32(1)
    enc = jnp.where(
        wt <= jnp.int32(0),
        jnp.int32(0),
        jnp.where(overloaded, jnp.int32(5), jnp.minimum(enc_calc, jnp.int32(5))),
    )
    return enc.astype(jnp.int32)


def to_hit_roll(rng: jax.Array, attacker_state, target_ac: jnp.ndarray) -> jnp.ndarray:
    """d20 hit check.

    Mirrors vendor/nethack/src/uhitm.c:376 —
        tmp = 1 + abon() + find_mac(mtmp) + bonuses
        hit iff tmp > rnd(20)   (uhitm.c:709-710 — strict greater-than)
    where ``find_mac`` returns the monster's AC directly (positive number;
    lower-is-better but kept positive — see worn.c:717).  Pre-fix Nethax
    used ``rnd(20) <= tmp`` which is off-by-one (would hit on tmp==dieroll).

    Returns True on hit.
    """
    roll = rnd(rng, 20).astype(jnp.int32)
    abon = _abon(
        attacker_state.player_str,
        attacker_state.player_dex,
        attacker_state.player_xl,
        state=attacker_state,
    )
    skill_bonus = _skill_hit_bonus(attacker_state)
    enchant = _wielded_enchant(attacker_state)
    # Vendor uhitm.c:376 — ``tmp = 1 + abon() + find_mac(mtmp) + ...``
    # No softening of negative AC; raw value used directly (heavily-armored
    # monsters CAN be unhittable by low-XL players).  find_mac (worn.c:717)
    # returns a signed AC that can be < 0.
    target_ac_i32 = target_ac.astype(jnp.int32)

    # vendor/nethack/src/uhitm.c:378 — XL contribution (player level).
    xl_bonus = attacker_state.player_xl.astype(jnp.int32)
    # vendor/nethack/src/uhitm.c:377 — Luck bonus: sgn(Luck)*((|Luck|+2)/3).
    luck = attacker_state.player_luck.astype(jnp.int32)
    luck_bonus = jnp.sign(luck) * ((jnp.abs(luck) + 2) // 3)
    # vendor/nethack/src/uhitm.c:376 — u.uhitinc (ring of increase accuracy).
    uhitinc = attacker_state.player_uhitinc.astype(jnp.int32)

    # vendor/nethack/src/uhitm.c:407-409 — encumbrance penalty: (2*enc)-1.
    enc = _compute_encumbrance(attacker_state)
    enc_penalty = jnp.where(enc > jnp.int32(0), -(jnp.int32(2) * enc - jnp.int32(1)), jnp.int32(0))

    # vendor/nethack/src/weapon.c:961 (abon) — confusion: -1 to-hit.
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    confused_timer = attacker_state.status.timed_statuses[int(TimedStatus.CONFUSION)].astype(jnp.int32)
    confusion_penalty = jnp.where(confused_timer > jnp.int32(0), jnp.int32(-1), jnp.int32(0))

    # vendor/nethack/src/uhitm.c:455 — stun: -1 to-hit.
    stunned_timer = attacker_state.status.timed_statuses[int(TimedStatus.STUNNED)].astype(jnp.int32)
    stun_penalty = jnp.where(stunned_timer > jnp.int32(0), jnp.int32(-1), jnp.int32(0))

    # vendor/nethack/src/uhitm.c:410 — trap: -3 to-hit when player is in a trap.
    trap_penalty = jnp.where(attacker_state.player_in_trap, jnp.int32(-3), jnp.int32(0))

    tmp = (jnp.int32(1) + abon + target_ac_i32 + skill_bonus + enchant
           + xl_bonus + luck_bonus + uhitinc
           + enc_penalty + confusion_penalty + stun_penalty + trap_penalty)
    # vendor/nethack/src/uhitm.c:709-710: mhit = (tmp > dieroll)
    return tmp > roll


# ---------------------------------------------------------------------------
# Damage roll (vendor/nethack/src/weapon.c::dmgval, lines 215-302)
# ---------------------------------------------------------------------------
def damage_roll(
    rng: jax.Array,
    weapon_entry,
    target_size_large: jnp.ndarray,
    sdam_n: int = 1,
    sdam_sides: int = 4,
    ldam_n: int = 1,
    ldam_sides: int = 4,
    str_bonus: jnp.ndarray = None,
) -> jnp.ndarray:
    """Roll weapon damage.

    Parameters
    ----------
    rng             : JAX PRNG key.
    weapon_entry    : Item-like pytree with .enchantment field (or None).
    target_size_large : bool — bigmonst flag (weapon.c:225).
    sdam_n, sdam_sides : small-target dice (objects[].oc_wsdam).
    ldam_n, ldam_sides : large-target dice (objects[].oc_wldam).
    str_bonus       : optional dbon() result; defaults to 0.

    Returns
    -------
    int32 — damage value, clamped at >= 0 (weapon.c:300-302).
    """
    key_small, key_large = split_n(rng, 2)
    small = dice_roll(key_small, sdam_n, sdam_sides)
    large = dice_roll(key_large, ldam_n, ldam_sides)
    base = jnp.where(target_size_large, large, small).astype(jnp.int32)

    enchant = jnp.int32(0) if weapon_entry is None else weapon_entry.enchantment.astype(jnp.int32)
    bonus = jnp.int32(0) if str_bonus is None else str_bonus.astype(jnp.int32)

    total = base + enchant + bonus
    return jnp.maximum(total, jnp.int32(0)).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Weapon skill practice & advancement (weapon.c:1167-1198)
# ---------------------------------------------------------------------------
def practice_skill(state, weapon_type: jnp.ndarray):
    """Increment practice for ``weapon_type``; advance tier if threshold hit.

    practice_needed_to_advance(tier) = (tier+1) * (tier+1) * 20  (skills.h:106
    keyed off vendor's 1-based level, see _PRACTICE_TO_ADVANCE above).
    Caps at SKILL_GRAND_MASTER.
    """
    skill_id = jnp.clip(weapon_type.astype(jnp.int32), 0, N_WEAPON_SKILLS - 1)
    combat = state.combat

    cur_tier = combat.weapon_skill[skill_id].astype(jnp.int32)
    cur_practice = combat.weapon_practice[skill_id].astype(jnp.int32)
    new_practice = cur_practice + jnp.int32(1)

    safe_tier = jnp.clip(cur_tier, 0, N_SKILL_TIERS - 1)
    threshold = _PRACTICE_TO_ADVANCE[safe_tier]
    can_advance = (new_practice >= threshold) & (cur_tier < SKILL_GRAND_MASTER)

    next_tier = jnp.where(can_advance, cur_tier + 1, cur_tier).astype(jnp.int8)
    next_practice = jnp.where(can_advance, jnp.int32(0), new_practice).astype(jnp.int32)

    new_weapon_skill = combat.weapon_skill.at[skill_id].set(next_tier)
    new_weapon_practice = combat.weapon_practice.at[skill_id].set(next_practice)
    new_combat = combat.replace(
        weapon_skill=new_weapon_skill,
        weapon_practice=new_weapon_practice,
    )
    return state.replace(combat=new_combat)


# ---------------------------------------------------------------------------
# Melee attack (uhitm.c::hitum / hmon paths)
# ---------------------------------------------------------------------------
def _polymorph_attack_dice(state):
    """Return (n_dice, n_sides) for the player's current polymorph attack.

    Reads ``state.polymorph.attack_n_dice[0]`` / ``attack_n_sides[0]``
    (the first attack slot of the current form — vendor/nethack/src/
    polyself.c uses ``mptr->mattk[0]`` as the primary).  Returns (1, 4)
    when not polymorphed or attack data is degenerate.
    """
    poly = getattr(state, "polymorph", None)
    if poly is None or not hasattr(poly, "is_polymorphed"):
        return jnp.int32(1), jnp.int32(4)

    is_poly = poly.is_polymorphed
    n_dice_raw = poly.attack_n_dice[0].astype(jnp.int32)
    n_sides_raw = poly.attack_n_sides[0].astype(jnp.int32)
    # Clamp to sane ranges so a zero attack slot doesn't produce 0 dmg.
    safe_dice = jnp.clip(n_dice_raw, 1, 16)
    safe_sides = jnp.clip(n_sides_raw, 1, 32)
    n_dice = jnp.where(is_poly & (n_dice_raw > 0), safe_dice, jnp.int32(1))
    n_sides = jnp.where(is_poly & (n_sides_raw > 0), safe_sides, jnp.int32(4))
    return n_dice, n_sides


def _roll_dice_sum(rng: jax.Array, n_dice: jnp.ndarray, n_sides: jnp.ndarray,
                   max_dice: int = 16) -> jnp.ndarray:
    """JIT-safe dice summation: roll up to ``max_dice`` d``n_sides`` and take
    the first ``n_dice`` rolls.  Mirrors monster_attack_player's pattern.
    """
    sides = jnp.maximum(n_sides.astype(jnp.int32), jnp.int32(1))

    def roll_one(carry, key):
        sub_roll = jax.random.randint(
            key, (), minval=1, maxval=sides + 1, dtype=jnp.int32
        )
        return carry, sub_roll

    keys = split_n(rng, max_dice)
    _, rolls = jax.lax.scan(roll_one, jnp.int32(0), keys)
    take_mask = jnp.arange(max_dice, dtype=jnp.int32) < n_dice
    return jnp.sum(jnp.where(take_mask, rolls, jnp.int32(0))).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Role-specific combat bonuses
# ---------------------------------------------------------------------------
def _wielded_type_id(state) -> jnp.ndarray:
    """Return the wielded weapon's type_id (or -1 when bare-handed)."""
    wielded = state.inventory.wielded.astype(jnp.int32)
    safe = jnp.clip(wielded, 0, state.inventory.items.type_id.shape[0] - 1)
    type_id = state.inventory.items.type_id[safe].astype(jnp.int32)
    return jnp.where(wielded >= 0, type_id, jnp.int32(-1))


def _monk_martial_arts_bonus(state, rng: jax.Array) -> jnp.ndarray:
    """Extra damage when a Monk fights bare-handed.

    Per vendor/nethack/src/uhitm.c::hmon_hitmon_barehands (line ~847,
    ``rnd(!martial_bonus() ? 2 : 4)``) and the Monk scaling described in
    uhitm.c::mon_arms_table: monks rolling bare-handed get an additional
    1d4 die per 4 experience levels (i.e. ``(XL - 1) // 4 + 1`` d4 dice
    when active).  Returns 0 when the monk is wielding a weapon or when
    the role is not Monk.

    JIT-safe: rolls a fixed cap of dice and masks unused ones.
    """
    is_monk = state.player_role == jnp.int8(int(Role.MONK))
    bare_handed = state.inventory.wielded == jnp.int8(-1)
    active = is_monk & bare_handed

    xl = state.player_xl.astype(jnp.int32)
    # 1 die at XL 1-4, 2 dice at XL 5-8, 3 dice at XL 9-12, ...
    n_dice = jnp.clip((xl - jnp.int32(1)) // jnp.int32(4) + jnp.int32(1),
                      jnp.int32(1), jnp.int32(8))
    raw = _roll_dice_sum(rng, n_dice, jnp.int32(4), max_dice=8)
    return jnp.where(active, raw, jnp.int32(0)).astype(jnp.int32)


def _samurai_bushido_bonus(state, rng: jax.Array) -> jnp.ndarray:
    """Extra damage when a Samurai wields a traditional samurai weapon.

    Per vendor/nethack/src/uhitm.c (Samurai weapon affinity, line ~969
    and ~1051) plus the dam_bonus structure in weapon.c::weapon_dam_bonus:
    we model the cultural bonus as +1d6 for katana and +1d4 for either
    the wakizashi (mapped to SHORT_SWORD in NetHack) or the yumi long bow.
    Returns 0 when the role is not Samurai or no qualifying weapon is
    wielded.
    """
    is_samurai = state.player_role == jnp.int8(int(Role.SAMURAI))
    type_id = _wielded_type_id(state)
    is_katana = type_id == jnp.int32(WEAPON_TYPE_KATANA)
    is_short_or_yumi = (
        (type_id == jnp.int32(WEAPON_TYPE_SHORT_SWORD))
        | (type_id == jnp.int32(WEAPON_TYPE_YUMI))
    )

    key_k, key_s = split_n(rng, 2)
    d6 = rnd(key_k, 6).astype(jnp.int32)
    d4 = rnd(key_s, 4).astype(jnp.int32)

    katana_bonus = jnp.where(is_samurai & is_katana, d6, jnp.int32(0))
    side_bonus = jnp.where(is_samurai & is_short_or_yumi, d4, jnp.int32(0))
    return (katana_bonus + side_bonus).astype(jnp.int32)


def _knight_chivalric_bonus(state, target_monster_idx: jnp.ndarray) -> jnp.ndarray:
    """+1 to-hit bonus for a Knight engaging a humanoid melee opponent.

    Mirrors the chivalric "single combat" attitude described by
    vendor/nethack/src/uhitm.c::check_caitiff (called from
    ``find_roll_to_hit``).  The vendor logic models knightly honour as
    alignment adjustments; here we expose it as a small mechanical bonus
    so Knights are sharper against humanoid foes (S_HUMANOID / S_HUMAN)
    and gain nothing against bestial targets such as dragons.

    Returns an int32 to-hit bonus (added into the ``tmp`` accumulator).
    """
    is_knight = state.player_role == jnp.int8(int(Role.KNIGHT))
    idx = target_monster_idx.astype(jnp.int32)
    entry = state.monster_ai.entry_idx[idx].astype(jnp.int32)
    safe_entry = jnp.clip(entry, 0, _MONSTER_SYMBOL_TABLE.shape[0] - 1)
    sym = _MONSTER_SYMBOL_TABLE[safe_entry].astype(jnp.int32)

    is_humanoid = (sym == jnp.int32(_S_HUMANOID)) | (sym == jnp.int32(_S_HUMAN))
    return jnp.where(is_knight & is_humanoid, jnp.int32(1), jnp.int32(0))


def _single_melee_strike(
    state,
    rng: jax.Array,
    target_monster_idx: jnp.ndarray,
    hit_penalty: jnp.ndarray = None,
):
    """Resolve a single melee strike against ``target_monster_idx``.

    Returns (new_state, dmg, hit).  Used by both single-weapon and
    two-weapon melee paths.  When the player is polymorphed, damage is
    rolled from the form's attack dice rather than the weapon.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    key_hit, key_dmg, key_monk, key_samurai, key_backstab, key_ac, key_silver = split_n(rng, 7)
    idx = target_monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    # Vendor uhitm.c:376 — tmp = 1 + abon() + find_mac(mtmp) + ...
    # find_mac (worn.c:717) returns the signed AC directly; negative
    # values contribute negatively to ``tmp`` without softening
    # (heavily-armored targets ARE genuinely harder for low-level
    # attackers).  No rnd(-ac) gate.
    target_ac = mai.ac[idx].astype(jnp.int32)

    target_large = mai.is_large[idx]
    target_alive = mai.alive[idx]

    # Standard to-hit; optional penalty (e.g. -1 per two-weapon strike,
    # mirroring vendor/nethack/src/uhitm.c::hitum's twohit penalty).  The
    # Knight chivalric bonus (uhitm.c::check_caitiff) adds +1 against
    # humanoid opponents and 0 otherwise.
    roll = rnd(key_hit, 20).astype(jnp.int32)
    abon = _abon(state.player_str, state.player_dex, state.player_xl, state=state)
    skill_bonus = _skill_hit_bonus(state)
    enchant = _wielded_enchant(state)
    pen = jnp.int32(0) if hit_penalty is None else hit_penalty.astype(jnp.int32)
    knight_bonus = _knight_chivalric_bonus(state, idx)
    # vendor/nethack/src/uhitm.c:378 — XL contribution.
    xl_bonus = state.player_xl.astype(jnp.int32)
    # vendor/nethack/src/uhitm.c:377 — Luck bonus: sgn(Luck)*((|Luck|+2)/3).
    luck = state.player_luck.astype(jnp.int32)
    luck_bonus = jnp.sign(luck) * ((jnp.abs(luck) + 2) // 3)
    # vendor/nethack/src/uhitm.c:376 — u.uhitinc (ring of increase accuracy).
    uhitinc = state.player_uhitinc.astype(jnp.int32)
    # vendor/nethack/src/uhitm.c:387-394 — target-state bonuses.
    # +2 sleeping (vendor msleeping).  Match either the int16 sleep_timer
    # countdown OR the asleep bool — they're kept in sync, but some
    # spawn paths set only one of them.
    sleeping_bonus = jnp.where(
        (mai.sleep_timer[idx].astype(jnp.int32) > jnp.int32(0)) | mai.asleep[idx],
        jnp.int32(2), jnp.int32(0),
    )
    # +2 stunned target (stun_timer > 0).  vendor/nethack/src/uhitm.c:388.
    target_stun_bonus = jnp.where(mai.stun_timer[idx].astype(jnp.int32) > jnp.int32(0), jnp.int32(2), jnp.int32(0))
    # +2 fleeing (flee_until_turn > timestep OR mstrategy == FLEE).
    # vendor/nethack/src/uhitm.c:389.
    timestep = getattr(state, "timestep", jnp.int32(0))
    fleeing_timer_bonus = jnp.where(
        mai.flee_until_turn[idx].astype(jnp.int32) > timestep.astype(jnp.int32),
        jnp.int32(2), jnp.int32(0),
    )
    fleeing_strat_bonus = jnp.where(
        mai.mstrategy[idx].astype(jnp.int32) == jnp.int32(4),
        jnp.int32(2), jnp.int32(0),
    )
    fleeing_bonus = jnp.maximum(fleeing_timer_bonus, fleeing_strat_bonus)
    # +4 paralyzed (paralyzed_timer > 0) or structurally immobile.
    # vendor/nethack/src/uhitm.c:393-394.
    paralyzed_bonus = jnp.where(mai.paralyzed_timer[idx].astype(jnp.int32) > jnp.int32(0), jnp.int32(4), jnp.int32(0))
    entry_i = jnp.clip(mai.entry_idx[idx].astype(jnp.int32), 0, _IS_IMMOBILE.shape[0] - 1)
    immobile_bonus = jnp.maximum(
        jnp.where(_IS_IMMOBILE[entry_i], jnp.int32(4), jnp.int32(0)),
        paralyzed_bonus,
    )

    # vendor/nethack/src/uhitm.c:455 — stun: -1 to-hit.
    stunned_timer = state.status.timed_statuses[int(TimedStatus.STUNNED)].astype(jnp.int32)
    stun_hit_penalty = jnp.where(stunned_timer > jnp.int32(0), jnp.int32(-1), jnp.int32(0))

    # vendor/nethack/src/uhitm.c:407-409 — encumbrance: -(2*enc-1) to-hit.
    enc = _compute_encumbrance(state)
    enc_penalty = jnp.where(enc > jnp.int32(0), -(jnp.int32(2) * enc - jnp.int32(1)), jnp.int32(0))

    # vendor/nethack/src/weapon.c:961 (abon) — confusion: -1 to-hit.
    confused_timer = state.status.timed_statuses[int(TimedStatus.CONFUSION)].astype(jnp.int32)
    confusion_penalty = jnp.where(confused_timer > jnp.int32(0), jnp.int32(-1), jnp.int32(0))

    # vendor/nethack/src/uhitm.c:410 — trap: -3 to-hit when player is in a trap.
    trap_penalty = jnp.where(state.player_in_trap, jnp.int32(-3), jnp.int32(0))

    tmp = (jnp.int32(1) + abon + target_ac + skill_bonus + enchant + pen + knight_bonus
           + xl_bonus + luck_bonus + uhitinc
           + sleeping_bonus + target_stun_bonus + fleeing_bonus + immobile_bonus
           + stun_hit_penalty + enc_penalty + confusion_penalty + trap_penalty)
    # vendor/nethack/src/uhitm.c:709-710 — strict ``tmp > dieroll``.
    hit = (tmp > roll) & target_alive

    # Polymorph: use form attack dice (polyself.c form attack table).
    poly_dice, poly_sides = _polymorph_attack_dice(state)
    poly = getattr(state, "polymorph", None)
    is_poly = (
        poly.is_polymorphed
        if (poly is not None and hasattr(poly, "is_polymorphed"))
        else jnp.bool_(False)
    )

    str_dmg = _dbon(state.player_str)
    weapon_enchant = _wielded_enchant(state)
    # Weapon-skill damage bonus per vendor/nethack/src/weapon.c:1644-1675
    # (weapon_dam_bonus).  Pre-fix Nethax did not apply this bonus to the
    # damage roll at all.
    skill_dmg_id = _wielded_skill_id(state)
    skill_dmg_tier = jnp.clip(
        state.combat.weapon_skill[skill_dmg_id].astype(jnp.int32),
        0, N_SKILL_TIERS - 1,
    )
    skill_dmg_bonus = _SKILL_DAM_BONUS[skill_dmg_tier].astype(jnp.int32)

    # Per-weapon damage dice: objects[].oc_wsdam/oc_wldam + vendor switch
    # bonuses (weapon.c::dmgval lines 225-295). type_id==-1 (bare-hands)
    # clamps to index 0 (fists sentinel: 1d2 small / 1d1 large).
    wep_type = _wielded_type_id(state)
    dn1, ds1, dn2, ds2 = _wdd(wep_type, target_large)

    # Weapon-path damage (per-weapon dice + STR + enchant + skill).
    key_dmg_w, key_dmg_p, key_dmg_w2 = split_n(key_dmg, 3)
    raw1 = _roll_dice_sum(key_dmg_w, dn1, ds1)
    raw2 = jnp.where(ds2 > 0, _roll_dice_sum(key_dmg_w2, dn2, ds2), jnp.int32(0))
    str_bonus_total = (str_dmg + weapon_enchant + skill_dmg_bonus).astype(jnp.int32)
    weapon_dmg = jnp.maximum(raw1 + raw2 + str_bonus_total, jnp.int32(0)).astype(jnp.int32)

    # Polymorph-path damage (form attack dice; no weapon enchant).
    poly_raw = _roll_dice_sum(key_dmg_p, poly_dice, poly_sides)
    poly_dmg = jnp.maximum(poly_raw + str_dmg, jnp.int32(0)).astype(jnp.int32)

    # Role-specific damage bonuses.  Each branch zeroes for non-matching
    # roles, so the sum is JIT-safe.  Monks lose their bonus when
    # polymorphed (no longer fighting bare-handed in the canonical sense).
    monk_bonus = _monk_martial_arts_bonus(state, key_monk)
    samurai_bonus = _samurai_bushido_bonus(state, key_samurai)
    role_bonus = (monk_bonus + samurai_bonus).astype(jnp.int32)

    # vendor/nethack/src/uhitm.c:1450 — u.udaminc (ring of increase damage).
    udaminc = state.player_udaminc.astype(jnp.int32)
    base_dmg = jnp.where(is_poly, poly_dmg, weapon_dmg + role_bonus).astype(jnp.int32)
    base_dmg = jnp.maximum(base_dmg + udaminc, jnp.int32(0)).astype(jnp.int32)

    # vendor/nethack/src/artifact.c::spec_dbon (lines 1091-1109) — artifact
    # bonus damage applied to weapon strikes against eligible targets.  Not
    # applied while polymorphed (no weapon strike).  JIT-pure: bonus is 0
    # when no artifact is wielded (artifact_bonus_damage handles arti==-1).
    arti_idx = _arti_idx(state)
    target_entry = mai.entry_idx[idx].astype(jnp.int32)
    key_dmg_arti = jax.random.fold_in(key_dmg, jnp.uint32(0xA47F))
    arti_bonus = _arti_bonus(arti_idx, target_entry, key_dmg_arti).astype(jnp.int32)
    # Skip artifact bonus while polymorphed (no weapon strike).
    arti_bonus = jnp.where(is_poly, jnp.int32(0), arti_bonus)
    base_dmg = (base_dmg + arti_bonus).astype(jnp.int32)

    # vendor/nethack/src/uhitm.c:387-394 — paralyzed target: +4 to-hit (already in
    # immobile_bonus above) and +4 damage.  Cite: uhitm.c:393-394.
    target_paralyzed = mai.paralyzed_timer[idx].astype(jnp.int32) > jnp.int32(0)
    paralyzed_dmg_bonus = jnp.where(target_paralyzed, jnp.int32(4), jnp.int32(0))
    base_dmg = (base_dmg + paralyzed_dmg_bonus).astype(jnp.int32)

    # vendor/nethack/src/uhitm.c:960-964 — Rogue backstab bonus.
    # If the player is a Rogue AND the target is sleeping, fleeing, or paralyzed,
    # deal +rnd(player_xl) extra damage.
    # vendor uhitm.c:960 checks !mtmp->mcanmove (paralyzed).
    is_rogue = state.player_role == jnp.int8(int(Role.ROGUE))
    target_fleeing_bs = mai.mstrategy[idx].astype(jnp.int32) == jnp.int32(4)
    target_vulnerable = mai.asleep[idx] | target_fleeing_bs | target_paralyzed
    xl_clamped = jnp.maximum(state.player_xl.astype(jnp.int32), jnp.int32(1))
    backstab_roll = rnd(key_backstab, xl_clamped).astype(jnp.int32)
    backstab_bonus = jnp.where(is_rogue & target_vulnerable, backstab_roll, jnp.int32(0))
    base_dmg = (base_dmg + backstab_bonus).astype(jnp.int32)

    # vendor/nethack/src/uhitm.c:455 — stun: -1 damage (in addition to to-hit penalty).
    stun_dmg_penalty = jnp.where(stunned_timer > jnp.int32(0), jnp.int32(-1), jnp.int32(0))
    base_dmg = jnp.maximum(base_dmg + stun_dmg_penalty, jnp.int32(0)).astype(jnp.int32)

    # vendor/nethack/src/weapon.c:331-332 — silver weapon vs hates_silver target:
    #   if (objects[otyp].oc_material == SILVER && hates_silver(mdat))
    #       tmp += rnd(20);
    # entry_idx maps monster slot -> MONSTERS table row.  type_id==-1 (bare
    # hands) clamps to index 0 (which is not SILVER), so is_silver is False.
    wep_type_silver = _wielded_type_id(state)
    safe_wep_for_mat = jnp.clip(wep_type_silver, 0, _OBJECT_MATERIAL.shape[0] - 1)
    wep_material = _OBJECT_MATERIAL[safe_wep_for_mat].astype(jnp.int32)
    is_silver_weapon = (wep_type_silver >= jnp.int32(0)) & (wep_material == jnp.int32(_MATERIAL_SILVER))
    silver_entry = jnp.clip(mai.entry_idx[idx].astype(jnp.int32), 0, _HATES_SILVER.shape[0] - 1)
    target_hates_silver_m = _HATES_SILVER[silver_entry]
    silver_d20 = rnd(key_silver, 20).astype(jnp.int32)
    silver_bonus_m = jnp.where(
        is_silver_weapon & target_hates_silver_m & ~is_poly,
        silver_d20,
        jnp.int32(0),
    )
    base_dmg = (base_dmg + silver_bonus_m).astype(jnp.int32)

    # vendor/nethack/src/uhitm.c:5424-5670 (hmonas) — when the hero is
    # polymorphed and the form has multiple natural attacks, slot 0 is the
    # primary strike (poly_dmg above) and slots 1..NATTK-1 add their own
    # dice rolls to the same target.  Slot 0 dice are already included in
    # poly_dmg.  We accumulate extra-slot damage with a static Python loop
    # using a simpler dice formula (slot_dice * (slot_sides+1)//2 expected
    # value with a random offset) to keep the JIT trace minimal.
    from Nethax.nethax.subsystems.polymorph import NATTK as _NATTK
    from Nethax.nethax.constants.monsters import AttackType as _AttackType
    AT_NONE_VAL = jnp.uint8(int(_AttackType.AT_NONE))
    poly_types = (
        poly.attack_types if (poly is not None and hasattr(poly, "attack_types"))
        else jnp.zeros((_NATTK,), dtype=jnp.uint8)
    )
    poly_ndice = (
        poly.attack_n_dice if (poly is not None and hasattr(poly, "attack_n_dice"))
        else jnp.zeros((_NATTK,), dtype=jnp.uint8)
    )
    poly_nsides = (
        poly.attack_n_sides if (poly is not None and hasattr(poly, "attack_n_sides"))
        else jnp.zeros((_NATTK,), dtype=jnp.uint8)
    )
    key_multi = jax.random.fold_in(key_dmg, jnp.uint32(0xA771))
    multi_keys = split_n(key_multi, _NATTK)
    extra_multi_total = jnp.int32(0)
    # Static-shape masked dice-sum: roll _MAX_DICE d-sides dice per attack
    # slot; only the first `slot_dice` rolls contribute.  Matches vendor
    # `d(slot_dice, slot_sides)` byte-equal (triangular distribution).
    # Cite: vendor/nethack/src/uhitm.c::hitum (polymorph multi-attack).
    _MAX_DICE = 8  # vendor dice counts on poly form attacks rarely exceed 6.
    for i in range(1, _NATTK):  # slot 0 is primary; slots 1..5 are extras
        slot_type = poly_types[i]
        slot_dice = poly_ndice[i].astype(jnp.int32)
        slot_sides = poly_nsides[i].astype(jnp.int32)
        slot_active = (
            (slot_type != AT_NONE_VAL)
            & (slot_dice > jnp.int32(0))
            & (slot_sides > jnp.int32(0))
        )
        # Roll _MAX_DICE dice on `slot_sides`; mask first `slot_dice`.
        safe_sides = jnp.maximum(slot_sides, jnp.int32(1))
        slot_rolls = jax.random.randint(
            multi_keys[i], (_MAX_DICE,), 0, safe_sides, dtype=jnp.int32,
        ) + jnp.int32(1)
        active_mask = jnp.arange(_MAX_DICE, dtype=jnp.int32) < slot_dice
        slot_sum = jnp.sum(jnp.where(active_mask, slot_rolls, jnp.int32(0))).astype(jnp.int32)
        extra_multi_total = extra_multi_total + jnp.where(
            slot_active, slot_sum, jnp.int32(0)
        ).astype(jnp.int32)
    base_dmg = (base_dmg + jnp.where(is_poly, extra_multi_total, jnp.int32(0))).astype(jnp.int32)

    dmg = jnp.where(hit, base_dmg, jnp.int32(0)).astype(jnp.int32)

    new_hp = jnp.maximum(mai.hp[idx] - dmg, jnp.int32(0)).astype(jnp.int32)
    new_alive = (new_hp > 0) & target_alive
    killed = target_alive & ~new_alive

    new_hp_arr = mai.hp.at[idx].set(new_hp)
    new_alive_arr = mai.alive.at[idx].set(new_alive)
    new_mai = mai.replace(hp=new_hp_arr, alive=new_alive_arr)

    new_combat = state.combat.replace(last_hit_landed=hit)
    new_state = state.replace(monster_ai=new_mai, combat=new_combat)

    # Skill practice on hit (legacy CombatState path).
    skill_id = _wielded_skill_id(new_state)
    new_state = jax.lax.cond(
        hit,
        lambda s: practice_skill(s, skill_id),
        lambda s: s,
        new_state,
    )

    # SkillState advancement — use_skill increments state.skills.advance.
    # Cite: vendor/nethack/src/weapon.c:1424 (use_skill).
    # Called on every hit regardless of success to mirror vendor behavior.
    wep_type_id = _wielded_type_id(new_state)
    safe_type_id = jnp.clip(wep_type_id.astype(jnp.int32), 0, _SKILL_WEAPON_TYPE_TO_SKILL.shape[0] - 1)
    wep_skill_id = _SKILL_WEAPON_TYPE_TO_SKILL[safe_type_id]
    new_state = jax.lax.cond(
        hit,
        lambda s: _skills_use_skill(s, wep_skill_id, 1),
        lambda s: s,
        new_state,
    )

    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated_if
    new_state = mark_violated_if(new_state, int(Conduct.PACIFIST), killed)

    # Award XP/score for kill.
    # Vendor reference: vendor/nethack/src/exper.c::experience (table-driven
    # XP value) followed by more_experienced (uexp / urexp accumulation;
    # exper.c:168-203).
    #
    # vendor exper.c:143-163 halves XP via ``nk`` only when the slain monster
    # is mrevived or mcloned; vendor's ``nk`` comes from
    # ``svm.mvitals[mtmp->data - mons].died`` (a per-PM kill counter).
    # The per-PM counter is tracked in ScoringState.monsters_died_per_pm
    # (vendor decl.h ``struct mvitals`` mirror).
    entry = jnp.clip(
        mai.entry_idx[idx].astype(jnp.int32),
        0, _MONSTER_XP_TABLE.shape[0] - 1,
    )
    kill_count = _scoring_died_for_pm(new_state.scoring, entry)
    mcloned = new_state.monster_ai.mcloned[idx]
    xp_award = _xp_experience(entry, kill_count, mcloned=mcloned)
    new_state = jax.lax.cond(
        killed,
        lambda s: _xp_more_experienced(s, xp_award, jnp.int32(0)),
        lambda s: s,
        new_state,
    )
    # Wave 30d: more_experienced is byte-equal vendor exper.c:168-203 and
    # only touches u.uexp / u.urexp.  Kill-counter and running-score side
    # effects (vendor end.c::done records per-genus/per-class kill counts;
    # u.urexp drives the score via topten.c:675) are bumped here via
    # scoring.record_kill, gated on the same ``killed`` flag.  The per-PM
    # mvitals.died counter is also bumped (vendor mondead.c::xkilled).
    new_state = jax.lax.cond(
        killed,
        lambda s: s.replace(scoring=_scoring_record_kill_pm(
            _scoring_record_kill(s.scoring, xp_award), entry
        )),
        lambda s: s,
        new_state,
    )

    # vendor/nethack/src/mondead.c::xkilled — death drops a corpse on the floor.
    # Most monsters leave a corpse; elementals, ghosts, vortices do not.
    # JIT-pure: lax.cond gates on killed; all writes are indexed array updates.
    safe_entry_corpse = jnp.clip(entry, 0, _KILLED_DROPS_CORPSE.shape[0] - 1)
    drops_corpse = _KILLED_DROPS_CORPSE[safe_entry_corpse]

    def _place_corpse(s):
        gi = s.ground_items
        death_pos = s.monster_ai.pos[idx].astype(jnp.int32)
        d_row = jnp.clip(death_pos[0], 0, s.terrain.shape[2] - 1)
        d_col = jnp.clip(death_pos[1], 0, s.terrain.shape[3] - 1)
        branch = s.dungeon.current_branch.astype(jnp.int32)
        level = s.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
        n_stack = gi.category.shape[-1]

        def _find_slot(carry, sidx):
            found, gs = carry
            is_empty = gi.category[branch, level, d_row, d_col, sidx] == jnp.int8(0)
            gs = jnp.where(~found & is_empty, sidx, gs)
            found = found | is_empty
            return (found, gs), None

        (gfound, gslot), _ = jax.lax.scan(
            _find_slot,
            (jnp.bool_(False), jnp.int32(0)),
            jnp.arange(n_stack, dtype=jnp.int32),
        )
        can_place = gfound & drops_corpse
        safe_gs = jnp.clip(gslot, 0, n_stack - 1)
        corpse_entry = s.monster_ai.entry_idx[idx].astype(jnp.int16)
        new_gi = gi.replace(
            category=gi.category.at[branch, level, d_row, d_col, safe_gs].set(
                jnp.where(can_place, jnp.int8(_FOOD_CATEGORY),
                          gi.category[branch, level, d_row, d_col, safe_gs])
            ),
            type_id=gi.type_id.at[branch, level, d_row, d_col, safe_gs].set(
                jnp.where(can_place, jnp.int16(_CORPSE_TYPE_ID),
                          gi.type_id[branch, level, d_row, d_col, safe_gs])
            ),
            quantity=gi.quantity.at[branch, level, d_row, d_col, safe_gs].set(
                jnp.where(can_place, jnp.int16(1),
                          gi.quantity[branch, level, d_row, d_col, safe_gs])
            ),
            corpse_entry_idx=gi.corpse_entry_idx.at[branch, level, d_row, d_col, safe_gs].set(
                jnp.where(can_place, corpse_entry,
                          gi.corpse_entry_idx[branch, level, d_row, d_col, safe_gs])
            ),
        )
        return s.replace(ground_items=new_gi)

    new_state = jax.lax.cond(
        killed,
        _place_corpse,
        lambda s: s,
        new_state,
    )

    # Quest nemesis kill hook (quest.c::nemdead ~109-113: Qstat(killed_nemesis)=TRUE).
    # JIT-pure: entry_idx comparison is scalar; lax.cond gates the state update.
    from Nethax.nethax.subsystems.quest import on_nemesis_killed, _NEMESIS_IDX_BY_ROLE
    role_idx_q = jnp.clip(new_state.player_role.astype(jnp.int32), 0, _NEMESIS_IDX_BY_ROLE.shape[0] - 1)
    nemesis_entry = _NEMESIS_IDX_BY_ROLE[role_idx_q].astype(jnp.int32)
    is_nemesis_kill = killed & (mai.entry_idx[idx].astype(jnp.int32) == nemesis_entry)
    new_state = jax.lax.cond(
        is_nemesis_kill,
        lambda s: on_nemesis_killed(s, mai.entry_idx[idx]),
        lambda s: s,
        new_state,
    )

    # vendor/nethack/src/artifact.c::artifact_hit lines 1220-1255 (Vorpal Blade)
    # vendor/nethack/src/artifact.c::magicbane_hit lines 1090-1170 (Magicbane)
    # Apply special on-hit artifact effects only when a hit landed and not poly.
    key_arti_hit = jax.random.fold_in(rng, jnp.uint32(0xB33F))
    new_state, arti_killed = jax.lax.cond(
        hit & ~is_poly,
        lambda s: _arti_hit_effects(s, idx, key_arti_hit),
        lambda s: (s, jnp.bool_(False)),
        new_state,
    )
    killed = killed | arti_killed

    return new_state, dmg, hit


def melee_attack(
    state,
    rng: jax.Array,
    target_monster_idx: jnp.ndarray,
):
    """Player melee-attacks the monster at ``target_monster_idx``.

    Wave 5 additions:
      * Two-weapon: when ``state.combat.two_weapon`` is True, perform TWO
        consecutive strikes (primary + alternate), each at -1 to-hit.
        Mirrors vendor/nethack/src/uhitm.c::hitum two-weapon branch.
      * Polymorph: when polymorphed, the form's intrinsic attack dice
        replace the weapon damage (polyself.c).

    Returns
    -------
    new_state    : EnvState
    damage_dealt : int32 (sum across strikes)
    hit_landed   : bool  (True if any strike connected)
    """
    two_weap = state.combat.two_weapon

    def _single(rngs):
        rng_a, _rng_b = rngs
        return _single_melee_strike(state, rng_a, target_monster_idx)

    def _double(rngs):
        rng_a, rng_b = rngs
        s1, dmg1, hit1 = _single_melee_strike(
            state, rng_a, target_monster_idx, hit_penalty=jnp.int32(-1),
        )
        s2, dmg2, hit2 = _single_melee_strike(
            s1, rng_b, target_monster_idx, hit_penalty=jnp.int32(-1),
        )
        return s2, dmg1 + dmg2, hit1 | hit2

    rng_a, rng_b = split_n(rng, 2)

    new_state, dmg, hit_landed = jax.lax.cond(
        two_weap, _double, _single, (rng_a, rng_b)
    )

    # Emit "You hit the monster." message on landing a melee strike.
    # Cite: vendor/nethack/src/uhitm.c::hmon — pline("You hit %s.", ...).
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    new_messages = jax.lax.cond(
        hit_landed,
        lambda m: _msg_emit(m, int(_MsgId.YOU_HIT_MONSTER)),
        lambda m: m,
        new_state.messages,
    )
    new_state = new_state.replace(messages=new_messages)
    return new_state, dmg, hit_landed


# ---------------------------------------------------------------------------
# Bump-attack — movement into a monster tile (hack.c::domove)
# ---------------------------------------------------------------------------
def bump_attack(state, rng: jax.Array, target_pos: jnp.ndarray):
    """Resolve a move-into-monster bump.

    Looks up the monster at ``target_pos``; if one is found, runs a melee
    attack.  If the monster died, the player moves onto the now-empty tile.

    Returns the updated EnvState.  No-ops cleanly when no monster occupies
    the tile (returns state unchanged apart from a stale rng burn).
    """
    target_pos_i32 = target_pos.astype(jnp.int32)
    mai = state.monster_ai

    # Find first alive monster matching the position; -1 if none.
    pos_i32 = mai.pos.astype(jnp.int32)            # [N, 2]
    matches = (
        (pos_i32[:, 0] == target_pos_i32[0])
        & (pos_i32[:, 1] == target_pos_i32[1])
        & mai.alive
    )
    # argmax returns 0 even if no True; gate with `any`.
    idx = jnp.argmax(matches).astype(jnp.int32)
    found = jnp.any(matches)

    # Safe-attack on a sentinel index when no monster present
    safe_idx = jnp.where(found, idx, jnp.int32(0))

    attacked_state, _dmg, _hit = melee_attack(state, rng, safe_idx)

    # If no monster was at the tile, keep the original state.  Use tree_map
    # over the pytree to merge — branches share structure by construction.
    def _pick(a, b):
        return jnp.where(found, a, b)

    new_state = jax.tree_util.tree_map(_pick, attacked_state, state)

    # If the monster died, move player onto its tile (hack.c::killed path).
    monster_died = found & ~new_state.monster_ai.alive[safe_idx]
    new_player_pos = jnp.where(
        monster_died,
        target_pos.astype(jnp.int16),
        new_state.player_pos,
    )
    new_state = new_state.replace(player_pos=new_player_pos)

    return new_state


# ---------------------------------------------------------------------------
# Monster-attacks-player (mhitu.c::mattacku/hitmu)
# ---------------------------------------------------------------------------
def monster_attack_player(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Monster ``monster_idx`` attacks the player with its natural attack.

    Returns (new_state, damage_dealt).

    Bit-equal to vendor/nethack/src/mhitu.c::mattacku — to-hit + per-attack
    dice rolls via ``d(damn, damd)`` (vendor mhitm.c::mdamagem analogue),
    with per-instance variance preserved through ``jax.random.randint`` on
    each strike.
    """
    key_hit, key_dmg = split_n(rng, 2)
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    # Use compute_ac for player AC.
    player_ac = compute_ac(state)
    alive = mai.alive[idx]

    # Monster to-hit, bit-equal to vendor/nethack/src/mhitu.c:709-718
    # (mattacku tmp accumulator) and uhitm.c:709-710 (strict ``tmp > dieroll``).
    #     tmp = AC_VALUE(u.uac) + 10 + mtmp->m_lev
    #     if (tmp <= 0) tmp = 1
    #     hit_iff tmp > rnd(20)
    # Per-instance ``m_lev`` is the per-monster level field from vendor
    # ``struct monst`` (include/monst.h).  Populated at spawn from
    # MONSTERS[entry_idx].level (wave 45a).  Falls back to the species-level
    # field for legacy slots seeded before wave 45a (zero == uninitialised).
    # vendor/nethack/src/mhitu.c:709-710 (tmp = AC_VALUE(u.uac) + 10 + m_lev)
    # vendor/nethack/src/mhitu.c:717-718 (clamp tmp >= 1)
    key_hit, key_ac = jax.random.split(key_hit)
    roll = rnd(key_hit, 20).astype(jnp.int32)
    from Nethax.nethax.subsystems.monster_ai import _monster_level
    species_lev = _monster_level(mai.entry_idx[idx])
    inst_mlev = mai.m_lev[idx].astype(jnp.int32)
    mlev = jnp.where(inst_mlev > jnp.int32(0), inst_mlev, species_lev)
    mlev = jnp.clip(mlev, jnp.int32(1), jnp.int32(49))
    # AC_VALUE: identity when AC>=0; rnd(-AC) softening when AC<0
    # vendor/nethack/src/hack.h:1538 (AC_VALUE macro); uhitm.c:380.
    ac_raw = player_ac.astype(jnp.int32)
    ac_neg_roll = jax.random.randint(
        key_ac, (), 1, jnp.maximum(-ac_raw + 1, 2), dtype=jnp.int32
    )
    ac_value = jnp.where(ac_raw >= 0, ac_raw, -ac_neg_roll)
    raw_tmp = ac_value + jnp.int32(10) + mlev
    tmp = jnp.maximum(raw_tmp, jnp.int32(1))
    # vendor/nethack/src/uhitm.c:709-710 — strict ``tmp > dieroll``.
    hit = (tmp > roll) & alive

    # vendor/nethack/src/weapon.c — disarmed monster uses bare-hands (1d2).
    # is_unwielded set by whip-pull or disarm artifact.
    raw_n_dice = jnp.clip(mai.attack_dice_n[idx].astype(jnp.int32), 1, 8)
    raw_sides  = jnp.clip(mai.attack_dice_sides[idx].astype(jnp.int32), 1, 12)
    n_dice = jnp.where(mai.is_unwielded[idx], jnp.int32(1), raw_n_dice)
    sides  = jnp.where(mai.is_unwielded[idx], jnp.int32(2), raw_sides)
    # Static unrolled dice draw using a small fixed cap (8) — JIT-safe.
    def roll_one(carry, key):
        sub_roll = jax.random.randint(
            key, (), minval=1, maxval=sides + 1, dtype=jnp.int32
        )
        return carry, sub_roll

    keys = split_n(key_dmg, 8)
    _, rolls = jax.lax.scan(roll_one, jnp.int32(0), keys)
    take_mask = jnp.arange(8, dtype=jnp.int32) < n_dice
    raw_dmg = jnp.sum(jnp.where(take_mask, rolls, jnp.int32(0))).astype(jnp.int32)

    # vendor/nethack/src/mhitu.c:1455-1530 — adtyp-based damage dispatch.
    # The monster's primary-attack damage-type controls how the rolled dmg is
    # applied: physical, elemental (with resistance halving), drain-life, or
    # sleep.  We dispatch via lax.switch on a small index mapped from adtyp.
    safe_entry = jnp.clip(
        mai.entry_idx[idx].astype(jnp.int32),
        0,
        _MONSTER_PRIMARY_ADTYP_TABLE.shape[0] - 1,
    )
    adtyp = _MONSTER_PRIMARY_ADTYP_TABLE[safe_entry]
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intr, TimedStatus as _TS
    intr = state.status.intrinsics
    fire_res  = intr[int(_Intr.RESIST_FIRE)]
    cold_res  = intr[int(_Intr.RESIST_COLD)]
    shock_res = intr[int(_Intr.RESIST_SHOCK)]
    acid_res  = intr[int(_Intr.RESIST_ACID)]
    sleep_res = intr[int(_Intr.RESIST_SLEEP)]

    # Damage-type sentinels (constants/monsters.py DamageType).
    _AD_PHYS_V = jnp.int32(0)
    _AD_FIRE_V = jnp.int32(2)
    _AD_COLD_V = jnp.int32(3)
    _AD_SLEE_V = jnp.int32(4)
    _AD_ELEC_V = jnp.int32(6)
    _AD_ACID_V = jnp.int32(8)
    _AD_DREN_V = jnp.int32(16)

    # Branch index: 0=PHYS 1=FIRE 2=COLD 3=SLEE 4=ELEC 5=ACID 6=DREN, default 0.
    idx_phys = jnp.int32(0)
    branch_idx = jnp.where(adtyp == _AD_FIRE_V, jnp.int32(1), idx_phys)
    branch_idx = jnp.where(adtyp == _AD_COLD_V, jnp.int32(2), branch_idx)
    branch_idx = jnp.where(adtyp == _AD_SLEE_V, jnp.int32(3), branch_idx)
    branch_idx = jnp.where(adtyp == _AD_ELEC_V, jnp.int32(4), branch_idx)
    branch_idx = jnp.where(adtyp == _AD_ACID_V, jnp.int32(5), branch_idx)
    branch_idx = jnp.where(adtyp == _AD_DREN_V, jnp.int32(6), branch_idx)

    def _b_phys(args):
        s, base = args
        return s, base

    def _b_fire(args):
        s, base = args
        halved = jnp.where(fire_res, base // jnp.int32(2), base)
        return s, halved

    def _b_cold(args):
        s, base = args
        halved = jnp.where(cold_res, base // jnp.int32(2), base)
        return s, halved

    def _b_slee(args):
        # Physical damage applied; if not RESIST_SLEEP, also add sleep timer.
        s, base = args
        sleep_dur = jnp.int32(10)
        cur_timer = s.status.timed_statuses[int(_TS.SLEEP)].astype(jnp.int32)
        new_sleep_timer = jnp.where(
            sleep_res, cur_timer, cur_timer + sleep_dur,
        )
        new_timed = s.status.timed_statuses.at[int(_TS.SLEEP)].set(new_sleep_timer)
        new_status = s.status.replace(timed_statuses=new_timed)
        # Gate: only apply on hit (we'll re-gate after the switch result is
        # multiplied through; this branch is reached even when hit=False
        # because lax.switch always evaluates; defer the hit gate below).
        return s.replace(status=new_status), base

    def _b_elec(args):
        s, base = args
        halved = jnp.where(shock_res, base // jnp.int32(2), base)
        return s, halved

    def _b_acid(args):
        s, base = args
        halved = jnp.where(acid_res, base // jnp.int32(2), base)
        return s, halved

    def _b_dren(args):
        # AD_DREN — drain XL.  Vendor: uhitm.c::mhitm_ad_drli calls
        # losexp(...) which shaves XL by 1 and HP_max by uhpinc[ulevel] and
        # Pw_max by ueninc[ulevel] (exper.c:206-291).  Drain resistance is
        # checked here so the secondary HP roll (base) still applies even if
        # the level-drain itself is resisted.
        from Nethax.nethax.subsystems.experience import losexp as _losexp
        from Nethax.nethax.subsystems.status_effects import Intrinsic as _DRIntr
        s, base = args
        # resists_drli: DRAIN_RES intrinsic.  Vendor exper.c:215.
        drli_res = (
            s.status.intrinsics[int(_DRIntr.RESIST_DRAIN)]
            | (s.status.timed_intrinsics[int(_DRIntr.RESIST_DRAIN)] > jnp.int32(0))
        )
        s2 = jax.lax.cond(drli_res, lambda st: st, _losexp, s)
        return s2, base

    # Apply switch.  When hit=False we still run the switch (purity), but the
    # final dmg is zeroed by the outer gate; side-effect branches (sleep/drain)
    # are gated below via lax.cond on `hit`.
    def _do_switch(s):
        return jax.lax.switch(
            branch_idx,
            [_b_phys, _b_fire, _b_cold, _b_slee, _b_elec, _b_acid, _b_dren],
            (s, raw_dmg),
        )

    def _skip_switch(s):
        return s, raw_dmg

    state_post_dispatch, eff_dmg = jax.lax.cond(hit, _do_switch, _skip_switch, state)

    dmg = jnp.where(hit, eff_dmg, jnp.int32(0)).astype(jnp.int32)

    new_hp = jnp.maximum(state_post_dispatch.player_hp - dmg, jnp.int32(0)).astype(jnp.int32)
    new_done = state_post_dispatch.done | (new_hp <= 0)
    new_state = state_post_dispatch.replace(player_hp=new_hp, done=new_done)

    # ------------------------------------------------------------------
    # AD_WERE lycanthropy infection — vendor/nethack/src/uhitm.c:mhitm_ad_were
    # (line 4265): when a were-creature lands a hit on the player, set
    # u.ulycn to the were's monster type (src/were.c:set_ulycn, line 234).
    # Gates: attack landed, player not already lycanthropic, no
    # Protection_from_shape_changers intrinsic.
    # ------------------------------------------------------------------

    poly = new_state.polymorph
    already_lycan = poly.lycanthropy_form >= jnp.int8(0)
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    prot_shape = new_state.status.intrinsics[int(Intrinsic.PROT_FROM_SHAPE_CHANGERS)]

    infect_cond = (
        hit
        & (adtyp == jnp.int32(_AD_WERE))
        & (~already_lycan)
        & (~prot_shape)
    )

    from Nethax.nethax.subsystems.polymorph import trigger_lycanthropy as _trigger_lycan
    were_form = mai.entry_idx[idx].astype(jnp.int32)

    new_state = jax.lax.cond(
        infect_cond,
        lambda s: _trigger_lycan(s, rng, were_form),
        lambda s: s,
        new_state,
    )

    # ------------------------------------------------------------------
    # AT_ENGL engulf hook — vendor/nethack/src/mhitu.c::gulpmu line 1287.
    # When the attacker is an engulfer and the hit landed, call try_engulf.
    # ------------------------------------------------------------------
    rng_engulf, _ = jax.random.split(rng)
    is_engulfer = _ENGULFER_TABLE[safe_entry]
    engulf_cond = hit & is_engulfer

    new_state = jax.lax.cond(
        engulf_cond,
        lambda s: _try_engulf(s, idx, rng_engulf),
        lambda s: s,
        new_state,
    )

    # Emit "The monster hits!" message when a hit landed.
    # Cite: vendor/nethack/src/mhitu.c::mattacku — pline("%s hits!", ...).
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    new_messages = jax.lax.cond(
        hit,
        lambda m: _msg_emit(m, int(_MsgId.MONSTER_HITS_YOU)),
        lambda m: m,
        new_state.messages,
    )
    new_state = new_state.replace(messages=new_messages)

    return new_state, dmg


# ---------------------------------------------------------------------------
# Thrown / ranged attack (vendor/nethack/src/dothrow.c::throwit)
# ---------------------------------------------------------------------------
THROW_MAX_RANGE: int = 8

# Impassable terrain values for obstacle check during flight.
# vendor/nethack/src/dothrow.c:1510-1580 — projectile stops at walls.
from Nethax.nethax.constants.tiles import TileType as _TileType
_IMPASSABLE_VOID: int = int(_TileType.VOID)
_IMPASSABLE_WALL: int = int(_TileType.WALL)

# Material ints referenced in throw logic.
_MATERIAL_SILVER: int = int(Material.SILVER)
_MATERIAL_GLASS: int  = int(Material.GLASS)
# POTTERY maps to MINERAL in our Material enum (closest analogue).
_MATERIAL_POTTERY: int = int(Material.MINERAL)


def thrown_attack(
    state,
    rng: jax.Array,
    slot_idx: jnp.ndarray,
    direction: jnp.ndarray,
):
    """Throw the item in ``slot_idx`` along ``direction``.

    Mirrors vendor/nethack/src/dothrow.c::throwit with these additions:

      Gap 1 — Obstacle check during flight (dothrow.c:1510-1580):
        lax.scan accumulates `still_flying`; stops at VOID or WALL.

      Gap 2 — Knockback on hit (dothrow.c:1130 mhurtle):
        weight > 100 knocks monster back 1 tile in throw direction.

      Gap 3 — Glass/POTTERY shatter on landing (dothrow.c:1825 + 2262):
        GLASS or MINERAL items have 50% chance to break; sets quantity=0.

      Gap 4 — Boomerang return (dothrow.c:1601-1611):
        BOOMERANG / AKLYS return to thrower on miss (Dex-based catch check).

      Gap 5 — Silver damage vs hates_silver (dothrow.c:1343):
        SILVER material weapon vs M2_UNDEAD|M2_WERE|M2_DEMON: +d20 damage.

      Gap 6 — Range formula (dothrow.c:1616-1625):
        range = max(1, str//2) - weight//40, clamped [1, 8].

    Parameters
    ----------
    state      : EnvState
    rng        : JAX PRNG key.
    slot_idx   : int32 -- inventory slot holding the projectile.
    direction  : int32[2] -- (dy, dx) step vector.

    Returns
    -------
    new_state with monster_ai, inventory, and ground_items updated.
    """
    slot = jnp.clip(slot_idx.astype(jnp.int32), 0, MAX_INVENTORY_SLOTS_C - 1)
    dy = direction[0].astype(jnp.int32)
    dx = direction[1].astype(jnp.int32)

    items = state.inventory.items
    has_item = items.category[slot] != jnp.int8(0)
    weight = items.weight[slot].astype(jnp.int32)
    enchant = items.enchantment[slot].astype(jnp.int32)
    type_id = items.type_id[slot].astype(jnp.int32)

    mai = state.monster_ai
    start_row = state.player_pos[0].astype(jnp.int32)
    start_col = state.player_pos[1].astype(jnp.int32)

    map_h = state.terrain.shape[2]
    map_w = state.terrain.shape[3]
    branch = state.dungeon.current_branch.astype(jnp.int32)
    level = (state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1))

    # --- Gap 6: dynamic range (dothrow.c:1616-1625) ---
    dyn_range = compute_throw_range(state.player_str, weight)

    # --- Gap 1 + monster scan: lax.scan over THROW_MAX_RANGE steps ---
    # Carry: (still_flying, hit, target_idx, last_r, last_c)
    # vendor/nethack/src/dothrow.c:1510-1580
    monster_pos = mai.pos.astype(jnp.int32)
    monster_alive = mai.alive

    init_scan = (
        jnp.bool_(True),   # still_flying
        jnp.bool_(False),  # hit
        jnp.int32(0),      # target_idx
        start_row,
        start_col,
    )

    def _flight_step(carry, step):
        flying, hit, tgt_idx, last_r, last_c = carry
        # Kill flying once step exceeds dynamic range (Gap 6).
        flying = flying & (step <= dyn_range)

        r = start_row + dy * step
        c = start_col + dx * step
        r_safe = jnp.clip(r, 0, map_h - 1)
        c_safe = jnp.clip(c, 0, map_w - 1)

        # Gap 1 -- stop at explicit WALL tiles (dothrow.c:1510-1580 IS_OBSTRUCTED).
        # VOID (0) = unexplored/unset terrain: treated as passable so that
        # states without full terrain maps (tests, default state) still work.
        tile = state.terrain[branch, level, r_safe, c_safe].astype(jnp.int32)
        is_blocked = (tile == jnp.int32(_IMPASSABLE_WALL))
        still_flying = flying & ~is_blocked

        # Monster detection: only while still flying and no hit yet.
        match = (monster_pos[:, 0] == r) & (monster_pos[:, 1] == c) & monster_alive
        any_match = jnp.any(match)
        m_idx = jnp.argmax(match).astype(jnp.int32)
        first_contact = any_match & ~hit & still_flying
        new_hit = hit | first_contact
        new_tgt = jnp.where(first_contact, m_idx, tgt_idx)

        # Advance position only while still flying.
        new_r = jnp.where(still_flying, r, last_r)
        new_c = jnp.where(still_flying, c, last_c)

        return (still_flying, new_hit, new_tgt, new_r, new_c), None

    steps = jnp.arange(1, THROW_MAX_RANGE + 1, dtype=jnp.int32)
    (_, found_hit, target_idx, end_row, end_col), _ = jax.lax.scan(
        _flight_step,
        init_scan,
        steps,
    )

    valid_throw = has_item & found_hit

    # --- To-hit roll ---
    (key_hit, key_dmg, key_silver, key_boom, key_break,
     key_mjol, key_brk_resist, key_brk_msg, key_brk_side) = split_n(rng, 9)
    target_ac = mai.ac[target_idx].astype(jnp.int32)
    target_alive = mai.alive[target_idx]
    roll = rnd(key_hit, 20).astype(jnp.int32)
    abon = _abon(state.player_str, state.player_dex, state.player_xl, state=state)
    tmp = jnp.int32(1) + abon + target_ac + enchant
    # vendor/nethack/src/uhitm.c:709-710 -- strict ``tmp > dieroll``.
    hit_landed = (tmp > roll) & valid_throw & target_alive

    # --- Damage: byte-equal vendor dmgval per-weapon dice (weapon.c:215-302).
    # ``weapon_damage_dice`` returns the small/large dice (oc_wsdam / oc_wldam
    # from include/objects.h) plus the per-otyp switch-statement bonus dice
    # from weapon.c:228-295 (TRIDENT +1, BATTLE_AXE +2d4 vs large, ...).
    # For items without a damage entry (oc_wsdam == 0) the roll is 0, leaving
    # only the spe (enchant) bonus — matching vendor dmgval behaviour for
    # projectiles that lack a weapon-class dice spec.
    safe_tgt_for_size = jnp.clip(target_idx, 0, mai.is_large.shape[0] - 1)
    target_large = mai.is_large[safe_tgt_for_size]
    dn1, ds1, dn2, ds2 = _wdd(type_id, target_large)
    key_dmg_a, key_dmg_b = split_n(key_dmg, 2)
    raw1 = jnp.where(ds1 > 0, _roll_dice_sum(key_dmg_a, dn1, ds1),
                     jnp.int32(0))
    raw2 = jnp.where(ds2 > 0, _roll_dice_sum(key_dmg_b, dn2, ds2),
                     jnp.int32(0))
    base_dmg = raw1 + raw2
    # vendor/nethack/src/weapon.c:298-302: tmp += spe; if tmp < 0, tmp = 0.
    raw_dmg = jnp.maximum(base_dmg + enchant, jnp.int32(0))

    # --- Gap 5: silver damage vs hates_silver (dothrow.c:1343) ---
    # if (obj->material == SILVER && hates_silver(mtmp->data)) dmg += d(20)
    # entry_idx maps monster slot → MONSTERS table index (species row).
    n_monsters = _HATES_SILVER.shape[0]
    safe_tgt_slot = jnp.clip(target_idx, 0, mai.entry_idx.shape[0] - 1)
    monster_entry = jnp.clip(
        mai.entry_idx[safe_tgt_slot].astype(jnp.int32), 0, n_monsters - 1
    )
    item_material = _OBJECT_MATERIAL[
        jnp.clip(type_id, 0, _OBJECT_MATERIAL.shape[0] - 1)
    ].astype(jnp.int32)
    is_silver = item_material == jnp.int32(_MATERIAL_SILVER)
    target_hates_silver = _HATES_SILVER[monster_entry]
    silver_bonus = rnd(key_silver, 20).astype(jnp.int32)
    silver_extra = jnp.where(
        is_silver & target_hates_silver & hit_landed,
        silver_bonus,
        jnp.int32(0),
    )

    dmg = jnp.where(hit_landed, jnp.maximum(raw_dmg + silver_extra, jnp.int32(1)), jnp.int32(0))

    new_hp = jnp.maximum(mai.hp[target_idx] - dmg, jnp.int32(0)).astype(jnp.int32)
    new_alive = (new_hp > 0) & target_alive
    new_hp_arr = mai.hp.at[target_idx].set(new_hp)
    new_alive_arr = mai.alive.at[target_idx].set(new_alive)
    new_mai = mai.replace(hp=new_hp_arr, alive=new_alive_arr)

    # --- Gap 2: knockback on hit (dothrow.c:1130 mhurtle) ---
    # Heavy weapon (weight > 100) knocks monster back 1 tile in throw direction.
    is_heavy = weight > jnp.int32(100)
    kb_r = mai.pos[target_idx, 0].astype(jnp.int32) + dy
    kb_c = mai.pos[target_idx, 1].astype(jnp.int32) + dx
    kb_r_safe = jnp.clip(kb_r, 0, map_h - 1)
    kb_c_safe = jnp.clip(kb_c, 0, map_w - 1)
    kb_tile = state.terrain[branch, level, kb_r_safe, kb_c_safe].astype(jnp.int32)
    kb_passable = (
        (kb_tile != jnp.int32(_IMPASSABLE_VOID)) &
        (kb_tile != jnp.int32(_IMPASSABLE_WALL))
    )
    do_knockback = hit_landed & is_heavy & kb_passable
    new_pos_r = jnp.where(do_knockback, kb_r.astype(jnp.int16), new_mai.pos[target_idx, 0])
    new_pos_c = jnp.where(do_knockback, kb_c.astype(jnp.int16), new_mai.pos[target_idx, 1])
    new_pos_arr = new_mai.pos.at[target_idx].set(
        jnp.stack([new_pos_r, new_pos_c]).astype(jnp.int16)
    )
    new_mai = new_mai.replace(pos=new_pos_arr)

    # --- Potion shatter -- vendor/nethack/src/dothrow.c:2262-2400 (potionhit) ---
    # When a thrown potion reaches its target (found_hit), it shatters and
    # applies its liquid effect instead of the weight-based damage above.
    # JIT-pure: jax.lax.cond gates on is_potion & found_hit; lax.switch
    # inside apply_potion_to_monster selects the per-effect branch.
    is_potion = items.category[slot] == jnp.int8(int(ItemCategory.POTION))
    potion_type_id = items.type_id[slot].astype(jnp.int32)
    key_pot, key_breathe = split_n(rng, 2)

    # --- D22: player potion-breathe side effect (dothrow.c::breakobj
    # POT_WATER branch lines 2498-2521).
    # Vendor: if next2u(x,y) && (!breathless || haseyes), call potionbreathe.
    # We model the most-impactful breathe effects as small status-timer bumps
    # on the player; this is a vendor-faithful approximation of
    # potionbreathe() (potion.c:1932-2107) for CONFUSION, BOOZE, SLEEPING,
    # BLINDNESS, HALLUCINATION.  Half_gas / breathless / wet-towel guards
    # are not modeled (Nethax lacks those player flags); the player is
    # assumed eye-bearing and breathing.
    # Cite: vendor/nethack/src/potion.c::potionbreathe lines 1932-2107.
    target_pos_r = mai.pos[target_idx, 0].astype(jnp.int32)
    target_pos_c = mai.pos[target_idx, 1].astype(jnp.int32)
    impact_r = jnp.where(found_hit, target_pos_r, end_row)
    impact_c = jnp.where(found_hit, target_pos_c, end_col)
    cheby = jnp.maximum(
        jnp.abs(impact_r - start_row),
        jnp.abs(impact_c - start_col),
    )
    player_adjacent = cheby <= jnp.int32(1)

    from Nethax.nethax.subsystems.items_potions import (
        _POTION_BASE_ID as _POT_BASE,
        N_POTIONS as _N_POT,
    )
    breathe_effect = jnp.clip(
        potion_type_id - jnp.int32(_POT_BASE), 0, _N_POT - 1
    )

    def _apply_breathe(s):
        # Per-effect player timer bumps.  Vendor potion.c:1932-2107 uses
        # ``itimeout_incr(rnd(5))`` for CONFUSION/BLIND/HALLU branches and
        # ``inc_timeout(SLEEPING, rnd(5))`` for the sleep arm.  Effects:
        #   CONFUSION (2) / BOOZE (20) → CONFUSION timer (rnd(5))
        #   SLEEPING  (17)             → SLEEP timer (rnd(5))
        #   BLINDNESS (16)             → BLIND timer (rnd(5))
        #   HALLU     (7)              → HALLUCINATION timer (rnd(5))
        # All others are no-ops (matches vendor's empty switch arms).
        from Nethax.nethax.subsystems.status_effects import TimedStatus
        ts = s.status.timed_statuses

        is_conf = (breathe_effect == jnp.int32(2)) | (breathe_effect == jnp.int32(20))
        is_sleep = breathe_effect == jnp.int32(17)
        is_blind = breathe_effect == jnp.int32(16)
        is_hallu = breathe_effect == jnp.int32(7)

        # rnd(5) ∈ [1, 5] — one shared draw, mirroring vendor's per-branch
        # itimeout_incr call (each arm fires at most one of the four, so a
        # single shared roll matches the per-arm rnd(5) outcome).
        bump = jax.random.randint(
            key_breathe, (), jnp.int32(1), jnp.int32(6), dtype=jnp.int32
        )
        ts = jnp.where(
            is_conf,
            ts.at[int(TimedStatus.CONFUSION)].add(bump),
            ts,
        )
        ts = jnp.where(
            is_sleep,
            ts.at[int(TimedStatus.SLEEP)].add(bump),
            ts,
        )
        ts = jnp.where(
            is_blind,
            ts.at[int(TimedStatus.BLIND)].add(bump),
            ts,
        )
        ts = jnp.where(
            is_hallu,
            ts.at[int(TimedStatus.HALLUCINATION)].add(bump),
            ts,
        )
        return s.replace(status=s.status.replace(timed_statuses=ts))

    def _shatter(s):
        # Commit new_mai (weight-based HP update cleared for potions) then
        # overlay the potion effect onto the target monster.
        s2 = s.replace(monster_ai=new_mai)
        s3 = apply_potion_to_monster(s2, key_pot, potion_type_id, target_idx)
        # D22: player-side breathe when adjacent to shatter point.
        return jax.lax.cond(player_adjacent, _apply_breathe, lambda x: x, s3)

    def _no_shatter(s):
        return s.replace(monster_ai=new_mai)

    state_after_hit = jax.lax.cond(
        is_potion & found_hit,
        _shatter,
        _no_shatter,
        state,
    )
    new_mai = state_after_hit.monster_ai

    # --- Remove projectile from inventory (1 unit) ---
    old_qty = items.quantity[slot].astype(jnp.int16)
    consume = has_item
    new_qty = jnp.where(
        consume,
        jnp.maximum(old_qty - jnp.int16(1), jnp.int16(0)),
        old_qty,
    )
    new_cat = jnp.where(
        consume & (new_qty == jnp.int16(0)),
        jnp.int8(0),
        items.category[slot],
    )
    new_items = items.replace(
        quantity=items.quantity.at[slot].set(new_qty),
        category=items.category.at[slot].set(new_cat),
    )

    # --- Drop on ground at terminal tile if no hit landed (projectile missed) ---
    drop_row = jnp.clip(end_row, 0, map_h - 1)
    drop_col = jnp.clip(end_col, 0, map_w - 1)

    should_drop = consume & ~hit_landed  # missed: projectile lands on floor

    # --- Gap 4: boomerang return (dothrow.c:1601-1611) ---
    # BOOMERANG / AKLYS return to thrower; rn2(100) > 2*Dex+50 => fumble => drop.
    n_objs = _IS_RETURNING_WEAPON.shape[0]
    is_returning = _IS_RETURNING_WEAPON[jnp.clip(type_id, 0, n_objs - 1)]
    dex_i = state.player_dex.astype(jnp.int32)
    catch_threshold = jnp.int32(2) * dex_i + jnp.int32(50)
    boom_roll = _rn2(key_boom, 100)
    catches = boom_roll <= catch_threshold
    # Returning and caught: undo the inventory decrement.
    boomerang_return = is_returning & catches & ~hit_landed & has_item

    # --- Mjollnir auto-return (dothrow.c:1710-1759, artilist.h:97-108) ---
    # Mjollnir thrown by Valkyrie with STR>=25 returns 99% of the time.
    # Implemented via mjollnir_throw_returns helper (artifact_powers.py).
    # The artifact-keyed gate (is_mjollnir & is_valk & has_str25) is checked
    # inside the helper; here we simply OR the result into the return path.
    from Nethax.nethax.subsystems.artifact_powers import (
        mjollnir_throw_returns as _mjollnir_returns,
    )
    mjol_return = _mjollnir_returns(state, key_mjol) & ~hit_landed & has_item

    any_return = boomerang_return | mjol_return
    return_qty = jnp.where(any_return, old_qty, new_items.quantity[slot])
    return_cat = jnp.where(any_return, items.category[slot], new_items.category[slot])
    new_items = new_items.replace(
        quantity=new_items.quantity.at[slot].set(return_qty),
        category=new_items.category.at[slot].set(return_cat),
    )
    # A caught returning weapon must not also land on the floor.
    should_drop = should_drop & ~any_return

    gi = state.ground_items
    # Find first empty slot in the ground stack at the terminal tile.
    n_stack = gi.category.shape[-1]

    def _find_gslot(carry, sidx):
        found, gs = carry
        is_empty = gi.category[branch, level, drop_row, drop_col, sidx] == 0
        gs = jnp.where(~found & is_empty, sidx, gs)
        found = found | is_empty
        return (found, gs), None

    (gfound, gslot), _ = jax.lax.scan(
        _find_gslot,
        (jnp.bool_(False), jnp.int32(0)),
        jnp.arange(n_stack, dtype=jnp.int32),
    )
    can_drop = should_drop & gfound
    safe_gs = jnp.clip(gslot, 0, n_stack - 1)

    # --- D25 / D28 prep: extract item info for breakobj side effects.
    item_corpsenm = items.corpse_entry_idx[slot].astype(jnp.int32)
    item_qty_pre = items.quantity[slot].astype(jnp.int32)
    item_spe = items.enchantment[slot].astype(jnp.int32)  # vendor obj->spe

    # --- D21 + D26: vendor-byte-equal breaktest (dothrow.c:2582-2609) ---
    # Replaces the legacy 50/50 GLASS|POTTERY break.  vendor_breaktest()
    # mirrors breaktest() exactly: GLASS+!artifact+!gem deterministic, plus
    # EXPENSIVE_CAMERA / POT_WATER / EGG / CREAM_PIE / MELON / venom switch
    # cases.  obj_resists() (rn2(100) < achance/ochance) gates artifacts
    # (99% resist) and crystal armor (90% non-break).  Triggered on:
    #   * floor landing (can_drop), AND
    #   * monster hits where the item is itself breakable
    #     (non-potion case — potion shatter is already handled above).
    # vendor cite: dothrow.c::breaks (line 2450) calls breaktest before
    # breakmsg+breakobj.
    item_cat_i = items.category[slot].astype(jnp.int32)
    item_otyp_i = items.type_id[slot].astype(jnp.int32)
    item_arti = items.artifact_idx[slot] >= jnp.int8(0)
    raw_does_break = vendor_breaktest(
        key_brk_resist,
        oclass=item_cat_i,
        otyp=item_otyp_i,
        material=item_material,
        is_artifact=item_arti,
    )
    # On a hit, potion shatter is already accounted for via apply_potion_*;
    # for non-potion hits we leave the projectile to land normally (vendor
    # breaks() is only called on landed/dropped objects, dothrow.c:1825).
    does_break = raw_does_break & can_drop
    drop_qty_base = jnp.where(
        can_drop,
        jnp.int16(1),
        gi.quantity[branch, level, drop_row, drop_col, safe_gs],
    )
    drop_qty = jnp.where(does_break, jnp.int16(0), drop_qty_base)
    # Also zero out inventory slot when item shatters.
    break_inv_qty = jnp.where(does_break, jnp.int16(0), new_items.quantity[slot])
    break_inv_cat = jnp.where(does_break, jnp.int8(0), new_items.category[slot])
    new_items = new_items.replace(
        quantity=new_items.quantity.at[slot].set(break_inv_qty),
        category=new_items.category.at[slot].set(break_inv_cat),
    )

    def _set_field_ground(field_g, field_inv):
        return field_g.at[branch, level, drop_row, drop_col, safe_gs].set(
            jnp.where(can_drop, field_inv[slot], field_g[branch, level, drop_row, drop_col, safe_gs])
        )

    new_ground = gi.replace(
        category=_set_field_ground(gi.category, items.category),
        type_id=_set_field_ground(gi.type_id, items.type_id),
        buc_status=_set_field_ground(gi.buc_status, items.buc_status),
        enchantment=_set_field_ground(gi.enchantment, items.enchantment),
        charges=_set_field_ground(gi.charges, items.charges),
        identified=_set_field_ground(gi.identified, items.identified),
        quantity=gi.quantity.at[branch, level, drop_row, drop_col, safe_gs].set(drop_qty),
        weight=_set_field_ground(gi.weight, items.weight),
        ac_bonus=_set_field_ground(gi.ac_bonus, items.ac_bonus),
        is_two_handed=_set_field_ground(gi.is_two_handed, items.is_two_handed),
    )

    new_inv = state.inventory.replace(items=new_items)

    # --- D25: mirror breakage → -2 luck (dothrow.c::breakobj MIRROR 2494-2497).
    is_mirror = item_otyp_i == jnp.int32(_OTYP_MIRROR)
    mirror_broke = does_break & is_mirror
    luck_after_mirror = jnp.where(
        mirror_broke,
        jnp.clip(state.player_luck.astype(jnp.int32) - jnp.int32(2),
                 jnp.int32(-13), jnp.int32(13)),
        state.player_luck.astype(jnp.int32),
    ).astype(jnp.int8)

    # --- D28: egg breakage side effects (dothrow.c::breakobj EGG 2525-2531).
    # vendor:
    #   if (hero_caused && obj->spe && ismnum(obj->corpsenm))
    #       change_luck(-min(obj->quan, 5));
    #   if (obj->corpsenm == PM_PYROLISK) explosion = TRUE;
    # ``obj->spe`` is 1 for fertile (laid by hero) eggs and triggers the
    # luck penalty (you destroyed your own future pet).  ``corpsenm`` maps
    # to the MONSTERS table index, mirrored as Item.corpse_entry_idx.
    is_egg = item_otyp_i == jnp.int32(_OTYP_EGG)
    egg_broke = does_break & is_egg
    is_fertile = item_spe > jnp.int32(0)
    has_corpsenm = item_corpsenm >= jnp.int32(0)
    egg_luck_active = egg_broke & is_fertile & has_corpsenm
    egg_penalty = jnp.minimum(item_qty_pre, jnp.int32(5))
    luck_after_egg = jnp.where(
        egg_luck_active,
        jnp.clip(luck_after_mirror.astype(jnp.int32) - egg_penalty,
                 jnp.int32(-13), jnp.int32(13)).astype(jnp.int8),
        luck_after_mirror,
    )

    # Pyrolisk egg → flame burst at landing tile.  Vendor explode() radius-1
    # fire damage; we model as 1d6+1 fire damage on the player if within
    # chebyshev 1 of the drop tile.
    is_pyrolisk = item_corpsenm == jnp.int32(_PM_PYROLISK)
    pyrolisk_explode = egg_broke & is_pyrolisk
    cheby_to_drop = jnp.maximum(
        jnp.abs(drop_row - start_row),
        jnp.abs(drop_col - start_col),
    )
    player_in_blast = pyrolisk_explode & (cheby_to_drop <= jnp.int32(1))
    fire_dmg = _rn2(key_brk_side, 6) + jnp.int32(1)
    fire_dmg_to_player = jnp.where(player_in_blast, fire_dmg, jnp.int32(0))
    new_player_hp = jnp.maximum(
        state.player_hp.astype(jnp.int32) - fire_dmg_to_player,
        jnp.int32(0),
    ).astype(jnp.int32)

    # --- D24: camera breakage releases a demon (dothrow.c::breakobj line
    # 2522 → release_camera_demon @ 2457-2470).  Vendor body:
    #     if (!rn2(3) && makemon(&mons[rn2(3) ? PM_HOMUNCULUS : PM_IMP], ...))
    # i.e. 1-in-3 chance to spawn, and within that 1-in-3 IMP / 2-in-3
    # HOMUNCULUS.  HP / m_lev / AC / attack dice are seeded from the MONSTERS
    # row exactly as ``newmonhp`` (makemon.c:1012-1054) does for non-special,
    # non-golem, non-Rider, non-dragon entries:
    #     m_lev = adj_lev(ptr) ≈ ptr->mlevel
    #     hp = hp_max = d(level, 8)   (basehp == level; floor at level+1)
    is_camera = item_otyp_i == jnp.int32(_OTYP_EXPENSIVE_CAMERA)
    camera_broke = does_break & is_camera
    cam_roll = _rn2(key_brk_msg, 3)
    cam_spawn = camera_broke & (cam_roll == jnp.int32(0))

    # Species pick: rn2(3) ? PM_HOMUNCULUS : PM_IMP  (so rn2==0 → IMP).
    key_cam_species = jax.random.fold_in(key_brk_msg, jnp.uint32(0xCAFE))
    species_roll = _rn2(key_cam_species, 3)
    cam_entry = jnp.where(
        species_roll == jnp.int32(0),
        jnp.int32(_PM_IMP),
        jnp.int32(_PM_HOMUNCULUS),
    )

    # newmonhp roll: d(level, 8); floor at level+1 (vendor:1050-1053).
    key_cam_hp = jax.random.fold_in(key_brk_msg, jnp.uint32(0xC0DE))
    cam_level = _MONSTER_SPAWN_LEVEL[cam_entry].astype(jnp.int32)
    cam_ac    = _MONSTER_SPAWN_AC[cam_entry].astype(jnp.int16)
    cam_atk_n = _MONSTER_SPAWN_ATK_N[cam_entry]
    cam_atk_s = _MONSTER_SPAWN_ATK_S[cam_entry]
    cam_hp = _roll_dice_sum(key_cam_hp, cam_level, jnp.int32(8))
    cam_hp = jnp.maximum(cam_hp, cam_level + jnp.int32(1))

    # Find a free monster slot.
    free_mask = ~new_mai.alive
    any_free = jnp.any(free_mask)
    spawn_slot = jnp.argmax(free_mask.astype(jnp.int32)).astype(jnp.int32)
    do_spawn = cam_spawn & any_free

    # Spawn at the drop tile.
    spawn_pos = jnp.stack([
        drop_row.astype(jnp.int16),
        drop_col.astype(jnp.int16),
    ])
    new_alive2 = new_mai.alive.at[spawn_slot].set(
        jnp.where(do_spawn, jnp.bool_(True), new_mai.alive[spawn_slot]))
    new_hp_max2 = new_mai.hp_max.at[spawn_slot].set(
        jnp.where(do_spawn, cam_hp, new_mai.hp_max[spawn_slot]))
    new_hp2 = new_mai.hp.at[spawn_slot].set(
        jnp.where(do_spawn, cam_hp, new_mai.hp[spawn_slot]))
    new_pos2 = new_mai.pos.at[spawn_slot].set(
        jnp.where(do_spawn, spawn_pos, new_mai.pos[spawn_slot]))
    new_entry2 = new_mai.entry_idx.at[spawn_slot].set(
        jnp.where(do_spawn, cam_entry.astype(jnp.int16),
                  new_mai.entry_idx[spawn_slot]))
    new_mlev2 = new_mai.m_lev.at[spawn_slot].set(
        jnp.where(do_spawn, cam_level.astype(new_mai.m_lev.dtype),
                  new_mai.m_lev[spawn_slot]))
    new_ac2 = new_mai.ac.at[spawn_slot].set(
        jnp.where(do_spawn, cam_ac.astype(new_mai.ac.dtype),
                  new_mai.ac[spawn_slot]))
    new_atk_n2 = new_mai.attack_dice_n.at[spawn_slot].set(
        jnp.where(do_spawn, cam_atk_n.astype(new_mai.attack_dice_n.dtype),
                  new_mai.attack_dice_n[spawn_slot]))
    new_atk_s2 = new_mai.attack_dice_sides.at[spawn_slot].set(
        jnp.where(do_spawn, cam_atk_s.astype(new_mai.attack_dice_sides.dtype),
                  new_mai.attack_dice_sides[spawn_slot]))
    # vendor dothrow.c:2467 sets ``mtmp->mpeaceful = !obj->cursed``; we do
    # not track the thrown camera's curse flag at this point, so default to
    # hostile (matches vendor's subsequent set_malign call for non-peaceful).
    final_mai = new_mai.replace(
        alive=new_alive2,
        hp=new_hp2,
        hp_max=new_hp_max2,
        pos=new_pos2,
        entry_idx=new_entry2,
        m_lev=new_mlev2,
        ac=new_ac2,
        attack_dice_n=new_atk_n2,
        attack_dice_sides=new_atk_s2,
    )

    return state_after_hit.replace(
        monster_ai=final_mai,
        inventory=new_inv,
        ground_items=new_ground,
        player_luck=luck_after_egg,
        player_hp=new_player_hp,
    )



# Constant alias to avoid an extra import inside thrown_attack.  Matches
# vendor/nethack/include/hack.h SZ_HEROINV (52).
MAX_INVENTORY_SLOTS_C: int = 52



# ---------------------------------------------------------------------------
# Two-weapon toggle (vendor/nethack/src/wield.c::dotwoweapon)
# ---------------------------------------------------------------------------
def can_twoweapon(state) -> jnp.ndarray:
    """Return True iff the player may toggle two-weapon combat ON.

    Mirrors vendor/nethack/src/wield.c::can_twoweapon (line 761).  The relevant
    state-side checks (we cannot easily reproduce role/polymorph guards in
    JIT, so those degrade to ``True``) are:

      * uwep must be set (a weapon is wielded);
      * the wielded weapon must not be two-handed (``bimanual(uwep)``,
        vendor wield.c line 786);
      * no shield equipped (``uarms``, vendor wield.c line 789).

    Returns a scalar bool JAX array suitable for ``jax.lax.cond``.

    Cite: vendor/nethack/src/wield.c::can_twoweapon lines 761-803.
    """
    wielded = state.inventory.wielded.astype(jnp.int32)
    has_wep = wielded >= jnp.int32(0)
    safe = jnp.clip(wielded, 0, state.inventory.items.is_two_handed.shape[0] - 1)
    is_2h = state.inventory.items.is_two_handed[safe] & has_wep

    # Shield slot occupied?
    from Nethax.nethax.subsystems.inventory import ArmorSlot as _ArmorSlot
    shield_slot = jnp.int32(int(_ArmorSlot.SHIELD))
    has_shield = state.inventory.worn_armor[shield_slot] >= jnp.int8(0)

    return has_wep & (~is_2h) & (~has_shield)


def handle_twoweapon(state, rng):
    """Toggle the two-weapon combat flag.

    Mirrors vendor/nethack/src/wield.c::dotwoweapon (line 845).  The vendor
    semantics are: turning two-weapon OFF always succeeds; turning it ON
    succeeds only when ``can_twoweapon()`` returns true (no shield, no 2H
    wielded weapon, both hands have a one-handed weapon).

    Cite: vendor/nethack/src/wield.c::dotwoweapon lines 845-864.
    """
    is_on = state.combat.two_weapon
    allow_turn_on = can_twoweapon(state)
    # OFF always toggles to False; ON only toggles to True when allowed.
    new_two_weapon = jnp.where(
        is_on,
        jnp.bool_(False),                       # always allowed: turn off
        jnp.where(allow_turn_on, jnp.bool_(True), jnp.bool_(False)),
    )
    new_combat = state.combat.replace(
        two_weapon=new_two_weapon,
    )
    return state.replace(combat=new_combat)


# ---------------------------------------------------------------------------
# Weapon-class predicate helpers
# ---------------------------------------------------------------------------
def _wielded_is_polearm(state) -> bool:
    """Return True when the wielded weapon maps to SkillId.POLEARMS.

    Cite: vendor/nethack/src/weapon.c::weapon_skill + uhitm.c::dolean.
    """
    from Nethax.nethax.subsystems.skills import _WEAPON_TYPE_TO_SKILL, SkillId
    type_id = _wielded_type_id(state)
    safe = jnp.clip(type_id, 0, _WEAPON_TYPE_TO_SKILL.shape[0] - 1)
    skill = _WEAPON_TYPE_TO_SKILL[safe]
    return bool(int(skill) == int(SkillId.POLEARMS))


def _wielded_is_axe(state) -> bool:
    """Return True when the wielded weapon maps to SkillId.AXE.

    Cite: vendor/nethack/src/weapon.c::weapon_skill + uhitm.c::cleave.
    """
    from Nethax.nethax.subsystems.skills import _WEAPON_TYPE_TO_SKILL, SkillId
    type_id = _wielded_type_id(state)
    safe = jnp.clip(type_id, 0, _WEAPON_TYPE_TO_SKILL.shape[0] - 1)
    skill = _WEAPON_TYPE_TO_SKILL[safe]
    return bool(int(skill) == int(SkillId.AXE))


def _handle_polearm_attack(state, rng, dir_idx: jnp.ndarray):
    """Attack the monster 2 tiles away in direction dir_idx using a polearm.

    Mirrors vendor/nethack/src/uhitm.c::dolean (reach weapon attack).
    dir_idx maps to (dy, dx) via the 8-direction table; 1 = East = (0, 1).

    Finds the first alive monster at distance exactly 2 in the given direction
    and applies a standard melee attack roll against it.  The adjacent tile
    (distance 1) is assumed empty (not checked for JIT safety).

    Cite: vendor/nethack/src/uhitm.c::dolean, circa line 3480.
    """
    # Direction table: 0=N, 1=E, 2=S, 3=W, 4=NE, 5=SE, 6=SW, 7=NW
    dy_table = jnp.array([[-1, 0, 1, 0, -1, 1, 1, -1]], dtype=jnp.int32)
    dx_table = jnp.array([[0, 1, 0, -1, 1, 1, -1, -1]], dtype=jnp.int32)
    safe_dir = jnp.clip(dir_idx.astype(jnp.int32), 0, 7)
    dy = dy_table[0, safe_dir]
    dx = dx_table[0, safe_dir]

    p_row = state.player_pos[0].astype(jnp.int32)
    p_col = state.player_pos[1].astype(jnp.int32)
    target_row = p_row + dy * jnp.int32(2)
    target_col = p_col + dx * jnp.int32(2)

    mai = state.monster_ai
    n = mai.alive.shape[0]
    indices = jnp.arange(n, dtype=jnp.int32)
    m_rows = mai.pos[:, 0].astype(jnp.int32)
    m_cols = mai.pos[:, 1].astype(jnp.int32)
    at_target = (m_rows == target_row) & (m_cols == target_col) & mai.alive
    idx = jnp.argmax(at_target).astype(jnp.int32)

    # Fall through to melee_attack — vendor apply.c::use_pole (line 3491)
    # ends in `gb.bhitpos = cc; mtmp = m_at(...); ... attack(mtmp)` which is
    # the standard hitmu dispatch.  Return only the new state, mirroring
    # vendor's ECMD_TIME side-effect-via-state pattern.
    # Cite: vendor/nethack/src/apply.c::use_pole lines 3489-3510.
    new_state, _dmg, _hit = melee_attack(state, rng, idx)
    return new_state


def _apply_cleave_splash(state, rng, primary_idx: jnp.ndarray, primary_dmg: jnp.ndarray):
    """Apply half of primary_dmg to monsters perpendicular to the attack direction.

    Mirrors vendor/nethack/src/uhitm.c::cleave splash behaviour: after an axe
    hit on a primary target, all alive monsters adjacent to the primary target
    (but not the primary target itself) take primary_dmg // 2 damage.

    Cite: vendor/nethack/src/uhitm.c::cleave, circa line 3620.
    """
    splash = (primary_dmg // jnp.int32(2)).astype(jnp.int32)
    mai = state.monster_ai

    primary_pos = mai.pos[primary_idx]
    p_row = primary_pos[0].astype(jnp.int32)
    p_col = primary_pos[1].astype(jnp.int32)

    n = mai.alive.shape[0]
    m_rows = mai.pos[:, 0].astype(jnp.int32)
    m_cols = mai.pos[:, 1].astype(jnp.int32)

    dr = jnp.abs(m_rows - p_row)
    dc = jnp.abs(m_cols - p_col)
    is_adjacent = (dr <= jnp.int32(1)) & (dc <= jnp.int32(1))
    not_primary = jnp.arange(n, dtype=jnp.int32) != primary_idx.astype(jnp.int32)
    gets_splash = is_adjacent & not_primary & mai.alive

    new_hp = jnp.where(
        gets_splash,
        jnp.maximum(mai.hp - splash, jnp.int32(0)),
        mai.hp,
    )
    new_alive = new_hp > jnp.int32(0)
    new_mai = mai.replace(hp=new_hp, alive=new_alive)
    return state.replace(monster_ai=new_mai)


def enforce_no_twohanded_while_riding(state):
    """Force-unwield a two-handed weapon when the player is riding a steed.

    Cite: vendor/nethack/src/do_wear.c (two-handed riding restriction,
    circa line 1820): you can't wield a two-handed weapon while mounted.

    When player_steed_mid > 0 (riding) and the wielded weapon has
    is_two_handed=True, sets inventory.wielded to -1 (bare-handed).
    The item stays in the inventory slot; it is merely unwielded.
    """
    wielded = state.inventory.wielded.astype(jnp.int32)
    is_riding = state.player_steed_mid > jnp.uint32(0)
    safe = jnp.clip(wielded, 0, state.inventory.items.is_two_handed.shape[0] - 1)
    wep_two_handed = state.inventory.items.is_two_handed[safe]
    should_unwield = is_riding & (wielded >= jnp.int32(0)) & wep_two_handed
    new_wielded = jnp.where(should_unwield, jnp.int8(-1), state.inventory.wielded)
    return state.replace(inventory=state.inventory.replace(wielded=new_wielded))


# ---------------------------------------------------------------------------
# Multi-shot thrown attack (vendor/nethack/src/dothrow.c::dofire)
# ---------------------------------------------------------------------------
def multishot_thrown_attack(state, rng, slot_idx, direction):
    """Fire N shots from slot_idx along direction where N = 1 + skill_tier.

    Mirrors vendor/nethack/src/dothrow.c::dofire multishot block (~line 386):
        n = 1 + P_SKILL(weapon_skill(otmp->otyp));
    We derive skill_tier from the wielded launcher's SkillId via the combat
    skills array (defaulting to SKILL_UNSKILLED=0 → N=1 when no launcher
    is wielded).

    Each shot is an independent ``thrown_attack`` call consuming one unit
    of ammo.  Calls are unrolled via lax.fori_loop for JIT safety.

    Cite: vendor/nethack/src/dothrow.c::dofire, multishot block.
    """
    from Nethax.nethax.subsystems.skills import SkillId

    # Determine launcher skill tier from the wielded weapon's skill entry.
    wielded = state.inventory.wielded.astype(jnp.int32)
    has_wielded = wielded >= jnp.int32(0)
    # Look up skill tier for BOW (covers most launcher types).
    bow_id = int(SkillId.BOW)
    skill_tier = jnp.where(
        has_wielded,
        state.skills.level[bow_id].astype(jnp.int32),
        jnp.int32(0),
    )
    n_shots = jnp.int32(1) + skill_tier  # N = 1 + skill_tier

    def _one_shot(i, s):
        rng_i = jax.random.fold_in(rng, i)
        return thrown_attack(s, rng_i, slot_idx, direction)

    return jax.lax.fori_loop(jnp.int32(0), n_shots, _one_shot, state)


# ---------------------------------------------------------------------------
# Throw action handler (vendor/nethack/src/dothrow.c::dothrow)
# ---------------------------------------------------------------------------
def handle_throw(state, rng):
    """Throw the first quivered / first thrown-capable inventory item east.

    JIT-safe wrapper around ``thrown_attack``.  Selects projectile via:
      1. inventory.quiver if set; else
      2. first WEAPON-category slot with quantity > 0.
    Direction defaults to east (0, 1) — full directional input requires
    a follow-up direction prompt which Wave 5 does not model.
    """
    items = state.inventory.items
    quiver = state.inventory.quiver.astype(jnp.int32)
    has_quiver = quiver >= jnp.int32(0)

    # Fallback: first weapon stock.
    is_weapon = items.category == jnp.int8(ItemCategory.WEAPON)
    in_stock = items.quantity > jnp.int16(0)
    valid_weap = is_weapon & in_stock
    first_weap = jnp.argmax(valid_weap).astype(jnp.int32)
    has_weap = jnp.any(valid_weap)

    slot = jnp.where(has_quiver, quiver, first_weap)
    can_throw = has_quiver | has_weap

    def _do_throw(_):
        return thrown_attack(
            state, rng, slot, jnp.array([0, 1], dtype=jnp.int32),
        )

    return jax.lax.cond(can_throw, _do_throw, lambda _: state, operand=None)


# ---------------------------------------------------------------------------
# Fight prefix handler (action_dispatch.py:handle_fight)
# ---------------------------------------------------------------------------
def handle_fight(state, rng):
    """Toggle the 'force-attack on next move' flag.

    NetHack's 'F' prefix (vendor/nethack/src/cmd.c::dofight) makes the next
    movement command attack into an empty tile rather than walking onto it.
    Wave 3 just stashes a single-bit flag in CombatState.last_attack_kind
    (bit 0 == force-fight); Wave 4 wires this into dispatch_action.
    """
    new_combat = state.combat.replace(
        last_attack_kind=(state.combat.last_attack_kind ^ jnp.int32(1)).astype(jnp.int32),
    )
    return state.replace(combat=new_combat)


# ---------------------------------------------------------------------------
# Per-turn upkeep
# ---------------------------------------------------------------------------
def step(state: CombatState, rng: jax.Array) -> CombatState:
    """Per-turn combat upkeep (no-op in Wave 3 — advancement is on-hit)."""
    return state
