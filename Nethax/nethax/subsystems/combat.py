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

Status: Wave 3 — core mechanics implemented (AC, to-hit d20, damage roll,
        melee/bump attack, monster attack, skill practice advancement).

Wave 3 simplifications (explicit):
    - Two-weapon: skip (Wave 4)
    - Two-handed: enforced at wield-time only (caller's responsibility)
    - Ranged / throw / breath: skip (Wave 4)
    - Engulf / passive: skip (Wave 4)
    - Polymorph combat: skip (Wave 4)
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

# ---------------------------------------------------------------------------
# Monster primary-attack damage-type table — adtyp of attack[0] per entry.
# Vendor reference: vendor/nethack/src/uhitm.c::mhitm_ad_were (line 4265);
# src/were.c::set_ulycn (line 234).  Used to dispatch AD_WERE infection.
# Built once at module load; never traced inside a jit boundary.
# ---------------------------------------------------------------------------
def _build_monster_primary_adtyp_table() -> jnp.ndarray:
    from Nethax.nethax.constants.monsters import MONSTERS  # noqa: PLC0415
    return jnp.array(
        [int(m.attacks[0][1]) if m.attacks else 0 for m in MONSTERS],
        dtype=jnp.int32,
    )


_MONSTER_PRIMARY_ADTYP_TABLE: jnp.ndarray = _build_monster_primary_adtyp_table()

# AD_WERE = 29  (mirrors DamageType.AD_WERE; plain int for use in lax.cond).
_AD_WERE: int = 29

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
#  level * level * 20).  Indexed by current tier.
_PRACTICE_TO_ADVANCE = jnp.array(
    [tier * tier * 20 for tier in range(N_SKILL_TIERS)],
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
def _abon(player_str: jnp.ndarray, player_dex: jnp.ndarray,
          player_xl: jnp.ndarray) -> jnp.ndarray:
    """Attack (to-hit) bonus for STR & DEX.

    Mirror of vendor/nethack/src/weapon.c:950-988 (Wave-3 simplification:
    drops the Upolyd branch and uses player_str/dex/xl directly).
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
    return (sbon + dex_bonus).astype(jnp.int32)


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
    # AC_VALUE for AC>=0 is the AC itself (hack.h:1538).
    ac_value = uac if uac >= 0 else uac  # placeholder; negative-path differs at runtime
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

    Wave 3 simplification: maps the wielded item's type_id into the
    [0, N_WEAPON_SKILLS) range via modulo.  Bare-hand → martial-arts slot 0.
    Wave 4 will install the canonical weapon → skill mapping from
    vendor/nethack/src/weapon.c:weapon_skill_index.
    """
    wielded = state.inventory.wielded.astype(jnp.int32)
    safe = jnp.clip(wielded, 0, state.inventory.items.type_id.shape[0] - 1)
    type_id = state.inventory.items.type_id[safe].astype(jnp.int32)
    skill_id = jnp.where(wielded >= 0, type_id % N_WEAPON_SKILLS, jnp.int32(0))
    return skill_id.astype(jnp.int32)


def _skill_hit_bonus(state) -> jnp.ndarray:
    """Return the current to-hit bonus from the wielded weapon's skill tier."""
    skill_id = _wielded_skill_id(state)
    tier = state.combat.weapon_skill[skill_id].astype(jnp.int32)
    safe_tier = jnp.clip(tier, 0, N_SKILL_TIERS - 1)
    return _SKILL_HIT_BONUS[safe_tier].astype(jnp.int32)


# ---------------------------------------------------------------------------
# To-hit roll (vendor/nethack/src/uhitm.c::find_roll_to_hit, lines 365-427)
# ---------------------------------------------------------------------------
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
    )
    skill_bonus = _skill_hit_bonus(attacker_state)
    enchant = _wielded_enchant(attacker_state)
    target_ac_i32 = target_ac.astype(jnp.int32)

    tmp = jnp.int32(1) + abon + target_ac_i32 + skill_bonus + enchant
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

    practice_needed_to_advance(tier) = tier * tier * 20  (skills.h:106).
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
    key_hit, key_dmg, key_monk, key_samurai = split_n(rng, 4)
    idx = target_monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    target_ac = mai.ac[idx].astype(jnp.int32)
    target_large = mai.is_large[idx]
    target_alive = mai.alive[idx]

    # Standard to-hit; optional penalty (e.g. -1 per two-weapon strike,
    # mirroring vendor/nethack/src/uhitm.c::hitum's twohit penalty).  The
    # Knight chivalric bonus (uhitm.c::check_caitiff) adds +1 against
    # humanoid opponents and 0 otherwise.
    roll = rnd(key_hit, 20).astype(jnp.int32)
    abon = _abon(state.player_str, state.player_dex, state.player_xl)
    skill_bonus = _skill_hit_bonus(state)
    enchant = _wielded_enchant(state)
    pen = jnp.int32(0) if hit_penalty is None else hit_penalty.astype(jnp.int32)
    knight_bonus = _knight_chivalric_bonus(state, idx)
    tmp = jnp.int32(1) + abon + target_ac + skill_bonus + enchant + pen + knight_bonus
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

    # Weapon-path damage (1d4 fallback + STR + enchant + skill).
    key_dmg_w, key_dmg_p = split_n(key_dmg, 2)
    weapon_dmg = damage_roll(
        key_dmg_w,
        None,
        target_large,
        sdam_n=1, sdam_sides=4,
        ldam_n=1, ldam_sides=4,
        str_bonus=str_dmg + weapon_enchant + skill_dmg_bonus,
    )

    # Polymorph-path damage (form attack dice; no weapon enchant).
    poly_raw = _roll_dice_sum(key_dmg_p, poly_dice, poly_sides)
    poly_dmg = jnp.maximum(poly_raw + str_dmg, jnp.int32(0)).astype(jnp.int32)

    # Role-specific damage bonuses.  Each branch zeroes for non-matching
    # roles, so the sum is JIT-safe.  Monks lose their bonus when
    # polymorphed (no longer fighting bare-handed in the canonical sense).
    monk_bonus = _monk_martial_arts_bonus(state, key_monk)
    samurai_bonus = _samurai_bushido_bonus(state, key_samurai)
    role_bonus = (monk_bonus + samurai_bonus).astype(jnp.int32)

    base_dmg = jnp.where(is_poly, poly_dmg, weapon_dmg + role_bonus).astype(jnp.int32)
    dmg = jnp.where(hit, base_dmg, jnp.int32(0)).astype(jnp.int32)

    new_hp = jnp.maximum(mai.hp[idx] - dmg, jnp.int32(0)).astype(jnp.int32)
    new_alive = (new_hp > 0) & target_alive
    killed = target_alive & ~new_alive

    new_hp_arr = mai.hp.at[idx].set(new_hp)
    new_alive_arr = mai.alive.at[idx].set(new_alive)
    new_mai = mai.replace(hp=new_hp_arr, alive=new_alive_arr)

    new_combat = state.combat.replace(last_hit_landed=hit)
    new_state = state.replace(monster_ai=new_mai, combat=new_combat)

    # Skill practice on hit.
    skill_id = _wielded_skill_id(new_state)
    new_state = jax.lax.cond(
        hit,
        lambda s: practice_skill(s, skill_id),
        lambda s: s,
        new_state,
    )

    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated_if
    new_state = mark_violated_if(new_state, int(Conduct.PACIFIST), killed)

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
    return jax.lax.cond(two_weap, _double, _single, (rng_a, rng_b))


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

    Wave 3 simplification: monsters always use their natural attack dice.
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
    # ``m_lev`` is not stored separately in the Nethax monster_ai state;
    # we approximate it from hp_max (clipped 1..30) which preserves the
    # vendor ordering relative to monster strength.
    # vendor/nethack/src/mhitu.c:709-710 (tmp = AC_VALUE(u.uac) + 10 + m_lev)
    # vendor/nethack/src/mhitu.c:717-718 (clamp tmp >= 1)
    roll = rnd(key_hit, 20).astype(jnp.int32)
    mlev = jnp.clip((mai.hp_max[idx] // 4).astype(jnp.int32), 1, 30)
    # AC_VALUE is the identity for AC>=0 (the deterministic Nethax case).
    ac_value = player_ac.astype(jnp.int32)
    raw_tmp = ac_value + jnp.int32(10) + mlev
    tmp = jnp.maximum(raw_tmp, jnp.int32(1))
    # vendor/nethack/src/uhitm.c:709-710 — strict ``tmp > dieroll``.
    hit = (tmp > roll) & alive

    n_dice = jnp.clip(mai.attack_dice_n[idx].astype(jnp.int32), 1, 8)
    sides = jnp.clip(mai.attack_dice_sides[idx].astype(jnp.int32), 1, 12)
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
    dmg = jnp.where(hit, raw_dmg, jnp.int32(0)).astype(jnp.int32)

    new_hp = jnp.maximum(state.player_hp - dmg, jnp.int32(0)).astype(jnp.int32)
    new_done = state.done | (new_hp <= 0)
    new_state = state.replace(player_hp=new_hp, done=new_done)
    # ------------------------------------------------------------------
    # AD_WERE lycanthropy infection
    # Vendor: vendor/nethack/src/uhitm.c::mhitm_ad_were (line 4265);
    # src/were.c::set_ulycn (line 234).
    # Gates: attack landed, player not already lycanthropic, no
    # Protection_from_shape_changers intrinsic.
    # ------------------------------------------------------------------
    safe_entry = jnp.clip(
        mai.entry_idx[idx].astype(jnp.int32),
        0,
        _MONSTER_PRIMARY_ADTYP_TABLE.shape[0] - 1,
    )
    adtyp = _MONSTER_PRIMARY_ADTYP_TABLE[safe_entry]

    poly = new_state.polymorph
    already_lycan = poly.lycanthropy_form >= jnp.int8(0)
    from Nethax.nethax.subsystems.status_effects import Intrinsic  # noqa: PLC0415
    prot_shape = new_state.status.intrinsics[
        int(Intrinsic.PROT_FROM_SHAPE_CHANGERS)
    ]

    infect_cond = (
        hit
        & (adtyp == jnp.int32(_AD_WERE))
        & (~already_lycan)
        & (~prot_shape)
    )

    from Nethax.nethax.subsystems.polymorph import trigger_lycanthropy as _trigger_lycan  # noqa: PLC0415
    were_form = mai.entry_idx[idx].astype(jnp.int32)

    new_state = jax.lax.cond(
        infect_cond,
        lambda s: _trigger_lycan(s, rng, were_form),
        lambda s: s,
        new_state,
    )

    return new_state, dmg


# ---------------------------------------------------------------------------
# Thrown / ranged attack (vendor/nethack/src/dothrow.c::throwit)
# ---------------------------------------------------------------------------
THROW_MAX_RANGE: int = 8


def thrown_attack(
    state,
    rng: jax.Array,
    slot_idx: jnp.ndarray,
    direction: jnp.ndarray,
):
    """Throw the item in ``slot_idx`` along ``direction``.

    Mirrors vendor/nethack/src/dothrow.c::throwit:
      * Walk up to THROW_MAX_RANGE tiles along (dy, dx).
      * On each tile, check for a live monster; on first match, roll
        to-hit; on hit, deal damage based on item weight (heuristic:
        max(1, weight // 30)) plus the item's enchantment.
      * If no monster is hit, drop the projectile on the floor at the
        terminal tile (top of the ground stack at the player's current
        branch/level).

    Parameters
    ----------
    state      : EnvState
    rng        : JAX PRNG key.
    slot_idx   : int32 — inventory slot holding the projectile.
    direction  : int32[2] — (dy, dx) step vector.

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

    mai = state.monster_ai
    start_row = state.player_pos[0].astype(jnp.int32)
    start_col = state.player_pos[1].astype(jnp.int32)

    # ---- Carry state for the trajectory loop ----
    # (hit, target_idx, last_row, last_col, key)
    init_carry = (
        jnp.bool_(False),
        jnp.int32(0),
        start_row,
        start_col,
        rng,
    )

    monster_pos = mai.pos.astype(jnp.int32)
    monster_alive = mai.alive

    def _body(i, carry):
        hit, tgt_idx, _last_r, _last_c, k = carry
        step = i + jnp.int32(1)
        r = start_row + dy * step
        c = start_col + dx * step

        # Find monster at (r, c)
        match = (monster_pos[:, 0] == r) & (monster_pos[:, 1] == c) & monster_alive
        any_match = jnp.any(match)
        m_idx = jnp.argmax(match).astype(jnp.int32)

        # First contact only.
        first_contact = any_match & ~hit
        new_hit = hit | first_contact
        new_tgt = jnp.where(first_contact, m_idx, tgt_idx)
        return (new_hit, new_tgt, r, c, k)

    (found_hit, target_idx, end_row, end_col, _k) = jax.lax.fori_loop(
        0, THROW_MAX_RANGE, _body, init_carry
    )

    valid_throw = has_item & found_hit

    # --- To-hit roll ---
    key_hit, key_dmg = split_n(rng, 2)
    target_ac = mai.ac[target_idx].astype(jnp.int32)
    target_alive = mai.alive[target_idx]
    roll = rnd(key_hit, 20).astype(jnp.int32)
    abon = _abon(state.player_str, state.player_dex, state.player_xl)
    tmp = jnp.int32(1) + abon + target_ac + enchant
    # vendor/nethack/src/uhitm.c:709-710 — strict ``tmp > dieroll``.
    hit_landed = (tmp > roll) & valid_throw & target_alive

    # --- Damage: weight-based heuristic + enchant + dex bonus ---
    base = jnp.maximum(weight // jnp.int32(30), jnp.int32(1))
    spread = rnd(key_dmg, 4).astype(jnp.int32)  # +1..4 variability
    raw_dmg = base + spread + enchant
    dmg = jnp.where(hit_landed, jnp.maximum(raw_dmg, jnp.int32(1)), jnp.int32(0))

    new_hp = jnp.maximum(mai.hp[target_idx] - dmg, jnp.int32(0)).astype(jnp.int32)
    new_alive = (new_hp > 0) & target_alive
    new_hp_arr = mai.hp.at[target_idx].set(new_hp)
    new_alive_arr = mai.alive.at[target_idx].set(new_alive)
    new_mai = mai.replace(hp=new_hp_arr, alive=new_alive_arr)

    # --- Potion shatter — vendor/nethack/src/dothrow.c:2262-2400 (potionhit) ---
    # When a thrown potion reaches its target it shatters and applies its liquid
    # effect.  JIT-pure: jax.lax.cond gates on is_potion & found_hit; lax.switch
    # inside apply_potion_to_monster selects the per-effect branch.
    is_potion = items.category[slot] == jnp.int8(int(ItemCategory.POTION))
    potion_type_id = items.type_id[slot].astype(jnp.int32)
    key_pot, _ = split_n(rng, 2)

    def _shatter(s):
        s2 = s.replace(monster_ai=new_mai)
        return apply_potion_to_monster(s2, key_pot, potion_type_id, target_idx)

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
    map_h = state.terrain.shape[2]
    map_w = state.terrain.shape[3]
    drop_row = jnp.clip(end_row, 0, map_h - 1)
    drop_col = jnp.clip(end_col, 0, map_w - 1)
    branch = state.dungeon.current_branch.astype(jnp.int32)
    level = (state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1))

    should_drop = consume & ~hit_landed  # missed: projectile lands on floor

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

    def _set_field_ground(field_g, field_inv):
        return field_g.at[branch, level, drop_row, drop_col, safe_gs].set(
            jnp.where(can_drop, field_inv[slot], field_g[branch, level, drop_row, drop_col, safe_gs])
        )

    # Use a quantity of 1 for the dropped projectile (the rest stays in inv).
    drop_qty = jnp.where(can_drop, jnp.int16(1),
                         gi.quantity[branch, level, drop_row, drop_col, safe_gs])

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
    return state.replace(monster_ai=new_mai, inventory=new_inv, ground_items=new_ground)


# Constant alias to avoid an extra import inside thrown_attack.  Matches
# vendor/nethack/include/hack.h SZ_HEROINV (52).
MAX_INVENTORY_SLOTS_C: int = 52


# ---------------------------------------------------------------------------
# Legacy Wave-1 API shims (kept so the rest of the engine keeps building)
# ---------------------------------------------------------------------------
def ranged_attack(state, rng, attacker_idx, target_pos):
    """No-op ranged stub (Wave 4)."""
    return state, jnp.int32(0)


def passive_attack(state, rng, defender_idx, attacker_idx):
    """No-op passive-attack stub (Wave 4)."""
    return state, jnp.int32(0)


# ---------------------------------------------------------------------------
# Two-weapon toggle (vendor/nethack/src/wield.c::dotwoweapon)
# ---------------------------------------------------------------------------
def handle_twoweapon(state, rng):
    """Toggle the two-weapon combat flag.

    Mirrors vendor/nethack/src/wield.c::dotwoweapon — when an alternate
    weapon is wielded the player can toggle two-weapon mode.  In Wave 5
    we simply flip the bit; the alternate-weapon slot is whatever the
    caller has stored in ``state.inventory.alternate_weapon_slot`` (set
    by future wield-related actions).
    """
    new_combat = state.combat.replace(
        two_weapon=~state.combat.two_weapon,
    )
    return state.replace(combat=new_combat)


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
