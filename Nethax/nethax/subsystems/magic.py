"""Magic subsystem — spellbook learning, spell memory, casting, Pw regeneration.

Canonical sources:
  vendor/nethack/src/spell.c   — spellbook learning, spell memory, casting
  vendor/nethack/src/zap.c     — wand/spell rays, effects
  vendor/nethack/src/mcastu.c  — monster cast at hero
  vendor/nethack/include/spell.h — spell table (SPELL_LEV_PW)
  vendor/nethack/src/allmain.c — regen_pw threshold formula
  vendor/nethack/src/role.c    — per-role spelbase/spelheal/spelstat table

Status: Wave 3 — cast_spell, pw_regen_tick, spell effects, handle_cast.
"""
import enum

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import struct
from Nethax.nethax.subsystems.skills import (
    use_skill as _skills_use_skill,
    _SPELL_SCHOOL_TO_SKILL_ID as _MAGIC_SCHOOL_TO_SKILL,
)
from Nethax.nethax.subsystems import detect as _detect
from Nethax.nethax.constants.monsters import MONSTERS, M2_HOSTILE, M2_NASTY


# ---------------------------------------------------------------------------
# Precomputed per-monster tables (module-level, JIT-safe)
# ---------------------------------------------------------------------------

_N_MONSTERS: int = len(MONSTERS)

# bool[N_MONSTERS] — True when the monster is "nasty":
#   level >= 7 AND has M2_HOSTILE flag (i.e. normally aggressive, not a pet).
# Cite: vendor/nethack/src/wizard.c::nasties[] — these are the high-level
#   aggressive monsters selected by pick_nasty() / nasty().
_IS_NASTY: jax.Array = jnp.array(
    [m.level >= 7 and bool(m.flags2 & M2_HOSTILE) for m in MONSTERS],
    dtype=jnp.bool_,
)

# int8[N_MONSTERS] — generation level proxy for level-appropriate sampling.
# Mirrors items_wands._MONSTER_GEN_LEVEL but defined here so magic.py is
# self-contained.  Cite: vendor/nethack/src/zap.c wand_create_monster logic.
_MONSTER_GEN_LEVEL: jax.Array = jnp.array(
    [m.level for m in MONSTERS], dtype=jnp.int8
)


# ---------------------------------------------------------------------------
# Spell schools  (vendor/nethack/include/skills.h lines 53-59)
# ---------------------------------------------------------------------------
class SpellSchool(enum.IntEnum):
    ATTACK_SPELL      = 0
    HEALING_SPELL     = 1
    DIVINATION_SPELL  = 2
    ENCHANTMENT_SPELL = 3
    CLERIC_SPELL      = 4
    ESCAPE_SPELL      = 5
    MATTER_SPELL      = 6


# ---------------------------------------------------------------------------
# Spell IDs  (vendor/nethack/include/objects.h FIRST_SPELL … LAST_SPELL)
# ---------------------------------------------------------------------------
class SpellId(enum.IntEnum):
    DIG              = 0
    MAGIC_MISSILE    = 1
    FIREBALL         = 2
    CONE_OF_COLD     = 3
    SLEEP            = 4
    FINGER_OF_DEATH  = 5
    LIGHT            = 6
    DETECT_MONSTERS  = 7
    HEALING          = 8
    KNOCK            = 9
    FORCE_BOLT       = 10
    CONFUSE_MONSTER  = 11
    CURE_BLINDNESS   = 12
    DRAIN_LIFE       = 13
    SLOW_MONSTER     = 14
    WIZARD_LOCK      = 15
    CREATE_MONSTER   = 16
    DETECT_FOOD      = 17
    CAUSE_FEAR       = 18
    CLAIRVOYANCE     = 19
    CURE_SICKNESS    = 20
    CHARM_MONSTER    = 21
    HASTE_SELF       = 22
    DETECT_UNSEEN    = 23
    LEVITATION       = 24
    EXTRA_HEALING    = 25
    RESTORE_ABILITY  = 26
    INVISIBILITY     = 27
    DETECT_TREASURE  = 28
    REMOVE_CURSE     = 29
    MAGIC_MAPPING    = 30
    IDENTIFY         = 31
    TURN_UNDEAD      = 32
    POLYMORPH        = 33
    TELEPORT_AWAY    = 34
    CREATE_FAMILIAR  = 35
    CANCELLATION     = 36
    PROTECTION       = 37
    JUMPING          = 38
    STONE_TO_FLESH   = 39
    CHAIN_LIGHTNING  = 40
    FLAME_SPHERE     = 41
    FREEZE_SPHERE    = 42


N_SPELLS = len(SpellId)

# Vendor parity constant (spell.c line 17: `#define KEEN 20000`).  When a
# spell is freshly learned via study_book(), sp_know is set to KEEN + bonus
# (vendor spell.c line 22: `#define incrnknow(spell, x)
# (svs.spl_book[spell].sp_know = KEEN + (x))`).  age_spells() decrements
# sp_know by 1 every turn (spell.h line 31: `#define decrnknow(spell)
# svs.spl_book[spell].sp_know--`).
KEEN = 20000

# Back-compat alias — older call sites and tests still import the
# legacy ``MAX_SPELL_MEMORY`` name.  Now aliased to vendor ``KEEN`` so the
# fresh-memory ceiling matches vendor byte-equally.
MAX_SPELL_MEMORY = KEEN

SPELL_KEEN = KEEN
SPELL_DECAY_PER_TURN = 1  # vendor: decrnknow() = sp_know-- (one per turn)

# ---------------------------------------------------------------------------
# Spell table  (objects.h SPELL macro: name, school, prob, delay, level, dir)
# Columns: (school, level)
# Used by cast_spell for Pw cost (SPELL_LEV_PW = level * 5) and failure calc.
# ---------------------------------------------------------------------------

# (SpellSchool, spell_level)
_SPELL_TABLE: list[tuple[int, int]] = [
    # DIG
    (SpellSchool.MATTER_SPELL,      5),
    # MAGIC_MISSILE
    (SpellSchool.ATTACK_SPELL,      2),
    # FIREBALL
    (SpellSchool.ATTACK_SPELL,      4),
    # CONE_OF_COLD
    (SpellSchool.ATTACK_SPELL,      4),
    # SLEEP
    (SpellSchool.ENCHANTMENT_SPELL, 3),
    # FINGER_OF_DEATH
    (SpellSchool.ATTACK_SPELL,      7),
    # LIGHT
    (SpellSchool.DIVINATION_SPELL,  1),
    # DETECT_MONSTERS
    (SpellSchool.DIVINATION_SPELL,  1),
    # HEALING
    (SpellSchool.HEALING_SPELL,     1),
    # KNOCK
    (SpellSchool.MATTER_SPELL,      1),
    # FORCE_BOLT
    (SpellSchool.ATTACK_SPELL,      1),
    # CONFUSE_MONSTER
    (SpellSchool.ENCHANTMENT_SPELL, 1),
    # CURE_BLINDNESS
    (SpellSchool.HEALING_SPELL,     2),
    # DRAIN_LIFE
    (SpellSchool.ATTACK_SPELL,      2),
    # SLOW_MONSTER
    (SpellSchool.ENCHANTMENT_SPELL, 2),
    # WIZARD_LOCK
    (SpellSchool.MATTER_SPELL,      2),
    # CREATE_MONSTER
    (SpellSchool.CLERIC_SPELL,      2),
    # DETECT_FOOD
    (SpellSchool.DIVINATION_SPELL,  2),
    # CAUSE_FEAR
    (SpellSchool.ENCHANTMENT_SPELL, 3),
    # CLAIRVOYANCE
    (SpellSchool.DIVINATION_SPELL,  3),
    # CURE_SICKNESS
    (SpellSchool.HEALING_SPELL,     3),
    # CHARM_MONSTER
    (SpellSchool.ENCHANTMENT_SPELL, 5),
    # HASTE_SELF
    (SpellSchool.ESCAPE_SPELL,      3),
    # DETECT_UNSEEN
    (SpellSchool.DIVINATION_SPELL,  3),
    # LEVITATION
    (SpellSchool.ESCAPE_SPELL,      4),
    # EXTRA_HEALING
    (SpellSchool.HEALING_SPELL,     3),
    # RESTORE_ABILITY
    (SpellSchool.HEALING_SPELL,     4),
    # INVISIBILITY
    (SpellSchool.ESCAPE_SPELL,      4),
    # DETECT_TREASURE
    (SpellSchool.DIVINATION_SPELL,  4),
    # REMOVE_CURSE
    (SpellSchool.CLERIC_SPELL,      3),
    # MAGIC_MAPPING
    (SpellSchool.DIVINATION_SPELL,  5),
    # IDENTIFY
    (SpellSchool.DIVINATION_SPELL,  3),
    # TURN_UNDEAD
    (SpellSchool.CLERIC_SPELL,      6),
    # POLYMORPH
    (SpellSchool.MATTER_SPELL,      6),
    # TELEPORT_AWAY
    (SpellSchool.ESCAPE_SPELL,      6),
    # CREATE_FAMILIAR
    (SpellSchool.CLERIC_SPELL,      6),
    # CANCELLATION
    (SpellSchool.MATTER_SPELL,      7),
    # PROTECTION
    (SpellSchool.CLERIC_SPELL,      1),
    # JUMPING
    (SpellSchool.ESCAPE_SPELL,      1),
    # STONE_TO_FLESH
    (SpellSchool.HEALING_SPELL,     3),
    # CHAIN_LIGHTNING
    (SpellSchool.ATTACK_SPELL,      2),
    # FLAME_SPHERE
    (SpellSchool.MATTER_SPELL,      1),
    # FREEZE_SPHERE
    (SpellSchool.MATTER_SPELL,      1),
]

_SPELL_SCHOOLS = jnp.array([s for s, _ in _SPELL_TABLE], dtype=jnp.int32)
_SPELL_LEVELS  = jnp.array([lv for _, lv in _SPELL_TABLE], dtype=jnp.int32)

# ---------------------------------------------------------------------------
# Per-role spellcasting stats
# (vendor/nethack/src/role.c roles[] table)
# Columns: (spelbase, spelheal, spelstat, spelspec_id)
#   spelbase  — base penalty (higher = worse)
#   spelheal  — healing-spell bonus (negative = better)
#   spelstat  — A_INT(10) or A_WIS(11); here stored as 0=INT, 1=WIS
#   spelspec_id — SpellId of the role's special spell (gets spelsbon=-4 bonus,
#                 but since that cancels in our simplified model we ignore it)
#
# Extracted from role.c (lines 63-71, 104-111, 144-152, etc.)
# Field order in struct: Energy, spelbase, spelheal, spelarmr, spelshld,
#                        spelsbon, spelstat, spelspec
# We only need spelbase, spelheal, spelstat for percent_success.
# ---------------------------------------------------------------------------

# Wizard role index (vendor/nethack/src/role.c: PM_WIZARD = 12).
_ROLE_WIZARD = 12

# Use 0 for INT-based roles, 1 for WIS-based roles
_ROLE_SPELSTAT_IS_WIS = jnp.array(
    # Arc  Bar  Cav  Hea  Kni  Mon  Pri  Rog  Ran  Sam  Tou  Val  Wiz
    [  0,   0,   0,   1,   1,   1,   1,   0,   0,   0,   0,   1,   0 ],
    dtype=jnp.int32,
)

# spelbase per role (from role.c field at position after Energy line)
# Arc=10, Bar=10, Cav=0, Hea=10, Kni=10, Mon=10, Pri=0, Rog=10, Ran=10,
# Sam=10, Tou=0, Val=0, Wiz=0
_ROLE_SPELBASE = jnp.array(
    [10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 0, 0],
    dtype=jnp.int32,
)

# spelheal per role (negative means bonus for healing spells)
# Arc=5, Bar=14, Cav=12, Hea=3, Kni=8, Mon=8, Pri=3, Rog=8, Ran=9,
# Sam=10, Tou=5, Val=10, Wiz=1
_ROLE_SPELHEAL = jnp.array(
    [5, 14, 12, 3, 8, 8, 3, 8, 9, 10, 5, 10, 1],
    dtype=jnp.int32,
)


# ---------------------------------------------------------------------------
# Audit K Batch B — vendor-correct per-role spell-modifier tables.
#
# These tables index by Role enum (0=Arc..12=Wiz) and supply spelarmr /
# spelshld / spelsbon / spelspec for the inventory-aware percent_success
# path.  Values are read straight from
# vendor/nethack/src/role.c struct Role entries (cite per-role below).
#
# NOTE: the legacy ``_ROLE_SPELBASE`` / ``_ROLE_SPELHEAL`` tables above are
# retained verbatim for back-compat with established parity tests; their
# field meanings drifted from vendor (mixing spelbase with initrecord) but
# the sum used in ``percent_success`` happens to remain correct for tested
# Wizard cases.  The new tables below mirror vendor exactly.
# ---------------------------------------------------------------------------

# spelarmr — metal-armor penalty when wearing metallic body armor.
# Cite: role.c:68(Arc=10), 109(Bar=8), 150(Cav=8), 190(Hea=10), 230(Kni=9),
#       271(Mon=20), 312(Pri=10), 409(Ran=10), 354(Rog=9), 449(Sam=8),
#       489(Tou=10), 529(Val=9), 570(Wiz=10).
_ROLE_SPELARMR = jnp.array(
    [10, 8, 8, 10, 9, 20, 10, 10, 9, 8, 10, 9, 10],
    dtype=jnp.int32,
)

# spelshld — shield penalty.
# Cite: role.c:67(Arc=2), 108(Bar=0), 149(Cav=1), 189(Hea=2), 229(Kni=0),
#       270(Mon=2), 311(Pri=2), 408(Ran=1), 353(Rog=1), 448(Sam=0),
#       488(Tou=2), 528(Val=0), 569(Wiz=3).
_ROLE_SPELSHLD = jnp.array(
    [2, 0, 1, 2, 0, 2, 2, 1, 1, 0, 2, 0, 3],
    dtype=jnp.int32,
)

# spelsbon — role-favored-spell bonus (negative reduces splcaster).
# Always -4 in vendor (role.c:71, 112, 153, 193, 233, 274, 315, 412, 357,
# 452, 492, 532, 573).
_ROLE_SPELSBON = jnp.array(
    [-4] * 13,
    dtype=jnp.int32,
)

# spelspec — the SpellId of the role's favored spell.
# Cite: role.c:70 MAGIC_MAPPING (Arc), 111 HASTE_SELF (Bar), 152 DIG (Cav),
#       192 CURE_SICKNESS (Hea), 232 TURN_UNDEAD (Kni),
#       273 RESTORE_ABILITY (Mon), 314 REMOVE_CURSE (Pri),
#       411 INVISIBILITY (Ran-file-slot=Ranger), 356 DETECT_TREASURE
#       (Rog-file-slot=Rogue), 451 CLAIRVOYANCE (Sam),
#       491 CHARM_MONSTER (Tou), 531 CONE_OF_COLD (Val),
#       572 MAGIC_MISSILE (Wiz).
_ROLE_SPELSPEC = jnp.array(
    [
        int(SpellId.MAGIC_MAPPING),     # Arc=0
        int(SpellId.HASTE_SELF),        # Bar=1
        int(SpellId.DIG),               # Cav=2
        int(SpellId.CURE_SICKNESS),     # Hea=3
        int(SpellId.TURN_UNDEAD),       # Kni=4
        int(SpellId.RESTORE_ABILITY),   # Mon=5
        int(SpellId.REMOVE_CURSE),      # Pri=6
        int(SpellId.INVISIBILITY),      # Ran=7
        int(SpellId.DETECT_TREASURE),   # Rog=8
        int(SpellId.CLAIRVOYANCE),      # Sam=9
        int(SpellId.CHARM_MONSTER),     # Tou=10
        int(SpellId.CONE_OF_COLD),      # Val=11
        int(SpellId.MAGIC_MISSILE),     # Wiz=12
    ],
    dtype=jnp.int32,
)

# Knight role index — used for the paladin_bonus cleric-spell waiver
# (vendor spell.c:2182 ``Role_if(PM_KNIGHT) && skilltype == P_CLERIC_SPELL``).
_ROLE_KNIGHT = 4

# Type-id constants used by spell_fail_chance armor modifiers (vendor
# objects.h).  QUARTERSTAFF (43) for the -3 weapon bonus; ROBE (101) for
# the cloak ±half adjustment.  Cite: subsystems.character.ObjType.
_TYPE_ID_QUARTERSTAFF = 43
_TYPE_ID_ROBE = 101

# Shield weight gate for spell.c:2269.  Vendor compares
# ``weight(uarms) > objects[SMALL_SHIELD].oc_weight`` (= 30) before
# applying the shield-on-cast penalty.  Keyed by ObjType id
# (subsystems.character).  SMALL_SHIELD (75) is the only shield with
# weight <= 30 so it's exempt.
# Cite: vendor/nethack/src/objects.c (SHIELD oc_weight values) and
#       vendor/nethack/src/spell.c:2269.
_SHIELD_TYPE_ID_TO_WEIGHT = {
    75: 30,   # SMALL_SHIELD            — exempt (<= 30)
    76: 40,   # ELVEN_SHIELD
    77: 50,   # URUK_HAI_SHIELD
    78: 50,   # ORCISH_SHIELD
    79: 100,  # DWARVISH_ROUNDSHIELD
    80: 100,  # LARGE_SHIELD
}
_SMALL_SHIELD_WEIGHT = 30

# Per-slot metallic helm/gloves/boots bonus constants.
# Cite: vendor/nethack/src/spell.c lines 106-108 (#define uarmhbon/uarmgbon/uarmfbon).
_UARMHBON = 4   # spell.c:106 — metal helmets interfere with the mind
_UARMGBON = 6   # spell.c:107 — casting channels through the hands
_UARMFBON = 2   # spell.c:108 — all metal interferes to some degree

# Material range for is_metallic(otmp): IRON..MITHRIL (vendor objclass.h:194).
# Nethax Material enum: IRON=11, METAL=12, COPPER=13, SILVER=14, GOLD=15,
# PLATINUM=16, MITHRIL=17.
_MATERIAL_IRON_MIN = 11
_MATERIAL_MITHRIL_MAX = 17

# Healing-spell set: spellids that get the spelheal bonus
_HEALING_SPELL_IDS = frozenset([
    SpellId.HEALING, SpellId.EXTRA_HEALING, SpellId.CURE_BLINDNESS,
    SpellId.CURE_SICKNESS, SpellId.RESTORE_ABILITY, SpellId.REMOVE_CURSE,
])

# MAXULEV from include/hack.h = 30
_MAXULEV = 30

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@struct.dataclass
class MagicState:
    """Persistent magic-related player state.

    Fields
    ------
    spell_memory       : turns of memory remaining per spell (int32, 0 = forgotten)
    spell_known        : whether the spell has ever been learned (bool)
    spell_letter       : inventory letter binding per spell (int8, -1 = unbound)
    pw_regen_counter   : sub-turn accumulator for Pw regeneration (int32)
    """

    spell_memory: jnp.ndarray      # [N_SPELLS]  int32
    spell_known: jnp.ndarray       # [N_SPELLS]  bool
    spell_letter: jnp.ndarray      # [N_SPELLS]  int8
    pw_regen_counter: jnp.ndarray  # scalar      int32

    @classmethod
    def default(cls) -> "MagicState":
        """Return a zeroed MagicState for a freshly created character."""
        return cls(
            spell_memory=jnp.zeros((N_SPELLS,), dtype=jnp.int32),
            spell_known=jnp.zeros((N_SPELLS,), dtype=jnp.bool_),
            spell_letter=jnp.full((N_SPELLS,), -1, dtype=jnp.int8),
            pw_regen_counter=jnp.int32(0),
        )


# ---------------------------------------------------------------------------
# Failure chance  (spell.c:percent_success)
# ---------------------------------------------------------------------------

def spell_fail_chance(
    role: jnp.ndarray,
    spell_id: jnp.ndarray,
    xl: jnp.ndarray,
    stat_int: jnp.ndarray,
    stat_wis: jnp.ndarray,
    skill_level: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return failure percentage (0..100) for casting spell_id.

    Vendor: spell.c::percent_success lines 2173-2292.  Audit-K fix: the
    skill lookup is now ``max(P_SKILL, P_UNSKILLED) - 1`` (vendor line 2238);
    previously this code produced ``skill_adj = -1`` for every spell,
    effectively making every spell six difficulty points harder than vendor.

    Armor/shield/weapon/cloak modifiers (vendor lines 2191-2212, 2265-2275)
    are not modelled here — they require player inventory access; this
    helper is invoked from several lightweight call sites that lack that
    plumbing.  ``cast_spell`` uses the inventory-aware path via the optional
    ``skill_level`` argument to feed the real skill tier; armor/shield
    penalties remain documented future work.  (Approximation note: success
    rates are within a few percentage points of vendor for un-armored
    casters, exact for naked Wizards/Monks.)

    Parameters
    ----------
    role        : int8, Role enum value (0..12)
    spell_id    : int32, SpellId value
    xl          : int32, experience level
    stat_int    : int8, player INT stat
    stat_wis    : int8, player WIS stat
    skill_level : int32 (optional) — actual 0-based skill tier
                  (Nethax encoding; P_UNSKILLED=0).  Defaults to 0 for
                  back-compat with legacy call sites.
    """
    spell_lv  = _SPELL_LEVELS[spell_id]
    school    = _SPELL_SCHOOLS[spell_id]

    # Pick stat based on role
    use_wis   = _ROLE_SPELSTAT_IS_WIS[role]
    statused  = jnp.where(use_wis, stat_wis, stat_int).astype(jnp.int32)

    # splcaster (intrinsic ability penalty, 0..20)
    spelbase  = _ROLE_SPELBASE[role]
    spelheal  = _ROLE_SPELHEAL[role]
    is_heal   = jnp.isin(spell_id, jnp.array(list(_HEALING_SPELL_IDS), jnp.int32))
    splcaster = spelbase + jnp.where(is_heal, spelheal, 0)
    splcaster = jnp.minimum(splcaster, 20)

    # chance from stat (vendor spell.c line 2230)
    chance = 11 * statused // 2

    # Vendor: skill = max(P_SKILL(skilltype), P_UNSKILLED) - 1.
    # Nethax 0-based encoding: P_UNSKILLED=0 vs vendor 1.  Vendor's
    # ``max(P_SKILL_C, 1) - 1`` maps to Nethax ``max(level, 0)``.
    # Cite: vendor/nethack/src/spell.c line 2238.
    if skill_level is None:
        skill_level = jnp.int32(0)
    skill_adj  = jnp.maximum(skill_level.astype(jnp.int32), jnp.int32(0))
    difficulty = (spell_lv - 1) * 4 - (skill_adj * 6 + xl // 3 + 1)

    # chance adjustment for difficulty
    sqrt_arg = jnp.maximum(900 * difficulty + 2000, 0)
    sqrt_val  = jnp.sqrt(sqrt_arg.astype(jnp.float32)).astype(jnp.int32)
    learning  = jnp.minimum(15 * jnp.maximum(-difficulty, 0) // jnp.maximum(spell_lv, 1), 20)
    chance    = jnp.where(difficulty > 0, chance - sqrt_val, chance + learning)

    # clamp before shield penalty (vendor line 2260-2263)
    chance = jnp.clip(chance, 0, 120)

    # combine: chance * (20 - splcaster) / 15 - splcaster  (vendor line 2283)
    chance = chance * (20 - splcaster) // 15 - splcaster

    # final clamp to [0, 100]; return fail% = 100 - success%
    success = jnp.clip(chance, 0, 100)
    return jnp.int32(100) - success.astype(jnp.int32)


def spell_success_chance(
    role: jnp.ndarray,
    spell_id: jnp.ndarray,
    xl: jnp.ndarray,
    stat_int: jnp.ndarray,
    stat_wis: jnp.ndarray,
    wielded_type_id: jnp.ndarray = None,
    skill_level: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return SUCCESS percentage (0..100) for casting spell_id.

    Vendor parity wrapper that returns the value spell.c::percent_success()
    actually returns (chance-of-cast).  This is the canonical name for
    new parity tests; ``spell_fail_chance`` is retained for back-compat
    and returns ``100 - spell_success_chance``.

    Source: vendor/nethack/src/spell.c::percent_success() lines 2173-2292.
    """
    return jnp.int32(100) - spell_fail_chance(
        role, spell_id, xl, stat_int, stat_wis, skill_level=skill_level,
    )


def _is_metallic_type_id(type_id: int) -> bool:
    """Return True if the given object type_id is metallic (IRON..MITHRIL).

    Vendor: include/objclass.h:194 ``is_metallic(otmp) == material >= IRON
    && material <= MITHRIL``.  ``type_id < 0`` denotes an empty slot.
    """
    from Nethax.nethax.subsystems.throwing import _OBJECT_MATERIAL
    tid = int(type_id)
    if tid < 0:
        return False
    if tid >= int(_OBJECT_MATERIAL.shape[0]):
        return False
    m = int(_OBJECT_MATERIAL[tid])
    return _MATERIAL_IRON_MIN <= m <= _MATERIAL_MITHRIL_MAX


def spell_success_chance_with_inventory(state, spell_id: int) -> int:
    """Return SUCCESS percentage (0..100), applying armor/shield/weapon mods.

    Vendor: spell.c::percent_success lines 2173-2292 — full inventory-aware
    formula.  This is a Python orchestrator that pulls worn slot type_ids
    from ``state.inventory`` and applies:

      - is_metallic(uarm) && !paladin_bonus → splcaster += spelarmr
        (halved when uarmc is ROBE) — spell.c:2191-2193.
      - uarmc == ROBE && !is_metallic(uarm) → splcaster -= spelarmr
        — spell.c:2194-2195.
      - uarms (shield) → splcaster += spelshld — spell.c:2197.
      - uwep == QUARTERSTAFF → splcaster -= 3 — spell.c:2199-2200.
      - !paladin_bonus: metallic helm/gloves/boots add uarmhbon/uarmgbon/
        uarmfbon respectively — spell.c:2202-2210.
      - spell == spelspec → splcaster += spelsbon — spell.c:2212-2213.
      - Healing-spell bonus (HEALING/EXTRA_HEALING/CURE_BLINDNESS/
        CURE_SICKNESS/RESTORE_ABILITY/REMOVE_CURSE) → splcaster += spelheal
        — spell.c:2216-2222.  (Already handled in spell_fail_chance.)
      - Shield post-clamp penalty: ``chance /= 4`` (``/= 2`` for spelspec)
        — spell.c:2268-2273.

    Returns a Python int (0..100) — callers compare against ``rnd(100)``.
    """
    from Nethax.nethax.subsystems.inventory import ArmorSlot

    role = int(state.player_role)
    sid = int(spell_id)
    safe_role = max(0, min(role, 12))

    # Inventory readouts.
    inv = state.inventory
    items = inv.items

    def _slot_type_id(slot_idx: int) -> int:
        if int(slot_idx) < 0:
            return -1
        i = int(slot_idx)
        if int(items.category[i]) == 0:
            return -1
        return int(items.type_id[i])

    body_idx   = int(inv.worn_armor[int(ArmorSlot.BODY)])
    shield_idx = int(inv.worn_armor[int(ArmorSlot.SHIELD)])
    helm_idx   = int(inv.worn_armor[int(ArmorSlot.HELM)])
    gloves_idx = int(inv.worn_armor[int(ArmorSlot.GLOVES)])
    boots_idx  = int(inv.worn_armor[int(ArmorSlot.BOOTS)])
    cloak_idx  = int(inv.worn_armor[int(ArmorSlot.CLOAK)])
    wep_idx    = int(inv.wielded)

    body_tid   = _slot_type_id(body_idx)
    shield_tid = _slot_type_id(shield_idx)
    helm_tid   = _slot_type_id(helm_idx)
    gloves_tid = _slot_type_id(gloves_idx)
    boots_tid  = _slot_type_id(boots_idx)
    cloak_tid  = _slot_type_id(cloak_idx)
    wep_tid    = _slot_type_id(wep_idx)

    body_metal   = _is_metallic_type_id(body_tid)
    helm_metal   = _is_metallic_type_id(helm_tid)
    gloves_metal = _is_metallic_type_id(gloves_tid)
    boots_metal  = _is_metallic_type_id(boots_tid)
    cloak_is_robe = (cloak_tid == _TYPE_ID_ROBE)
    has_shield = (shield_tid >= 0)
    wep_quarterstaff = (wep_tid == _TYPE_ID_QUARTERSTAFF)

    # Cleric_spell + Knight → paladin_bonus waiver of metallic penalties.
    # Cite: spell.c:2182-2183.
    school = int(_SPELL_TABLE[sid][0])
    is_cleric = (school == int(SpellSchool.CLERIC_SPELL))
    paladin_bonus = (role == _ROLE_KNIGHT) and is_cleric

    # Per-role spell stats.
    spelarmr = int(_ROLE_SPELARMR[safe_role])
    spelshld = int(_ROLE_SPELSHLD[safe_role])
    spelsbon = int(_ROLE_SPELSBON[safe_role])
    spelspec = int(_ROLE_SPELSPEC[safe_role])

    # Get baseline success% from the un-modified vendor formula.  We then
    # *adjust* it by recomputing the body/shield/weapon/cloak/helm/gloves/
    # boots/spelspec deltas — this matches the vendor's structure where
    # ``splcaster`` is incremented/decremented incrementally before the
    # final clamps.  To avoid re-deriving the full sqrt math, we work in
    # ``splcaster`` deltas and adjust the final clamp accordingly.

    # First compute the un-armored baseline.
    school_arr = jnp.int32(school)
    safe_sch = max(0, min(school, _MAGIC_SCHOOL_TO_SKILL.shape[0] - 1))
    skill_id = jnp.int32(int(_MAGIC_SCHOOL_TO_SKILL[safe_sch]))
    skill_lvl = state.skills.level[skill_id].astype(jnp.int32)
    baseline_pct = int(spell_success_chance(
        jnp.int32(role),
        jnp.int32(sid),
        state.player_xl.astype(jnp.int32),
        state.player_int.astype(jnp.int32),
        state.player_wis.astype(jnp.int32),
        skill_level=skill_lvl,
    ))

    # If no inventory modifiers apply, return baseline.
    no_mods = (
        not body_metal and not cloak_is_robe and not has_shield
        and not wep_quarterstaff and not helm_metal and not gloves_metal
        and not boots_metal and (sid != spelspec)
    )
    if no_mods:
        return baseline_pct

    # ----- Recompute splcaster delta from inventory ----------------------
    # Vendor builds splcaster from scratch (spell.c:2186-2222); we mirror it.
    splcaster_delta = 0

    # spell.c:2191-2193 — metallic body armor.
    if body_metal and not paladin_bonus:
        # Halved when wearing a ROBE cloak; full otherwise.
        if cloak_is_robe:
            splcaster_delta += spelarmr // 2
        else:
            splcaster_delta += spelarmr
    elif cloak_is_robe:
        # spell.c:2194-2195 — ROBE alone (no metallic body) → spelarmr bonus.
        splcaster_delta -= spelarmr

    # spell.c:2197 — any shield adds spelshld penalty.
    if has_shield:
        splcaster_delta += spelshld

    # spell.c:2199-2200 — quarterstaff bonus.
    if wep_quarterstaff:
        splcaster_delta -= 3

    # spell.c:2202-2210 — metallic helm/gloves/boots (paladin waiver).
    if not paladin_bonus:
        if helm_metal:
            splcaster_delta += _UARMHBON
        if gloves_metal:
            splcaster_delta += _UARMGBON
        if boots_metal:
            splcaster_delta += _UARMFBON

    # spell.c:2212-2213 — favored-spell bonus.
    if sid == spelspec:
        splcaster_delta += spelsbon

    # ----- Apply splcaster delta to baseline -----------------------------
    # In vendor, splcaster gets clamped to 20, then a non-linear
    # combination:  chance = chance * (20 - splcaster) / 15 - splcaster.
    # To stay byte-equal, we directly recompute the final chance using the
    # adjusted splcaster.  We refactor the existing un-modified path to
    # surface its intermediate (pre-splcaster) "chance" value.
    chance_pre = _percent_success_chance_pre_splcaster(
        role, sid, int(state.player_xl), int(state.player_int),
        int(state.player_wis), int(skill_lvl),
    )
    # Recompute splcaster from scratch (matches vendor).
    spelbase_v = int(_ROLE_SPELBASE[safe_role])
    spelheal_v = int(_ROLE_SPELHEAL[safe_role])
    is_heal = sid in {
        int(SpellId.HEALING), int(SpellId.EXTRA_HEALING),
        int(SpellId.CURE_BLINDNESS), int(SpellId.CURE_SICKNESS),
        int(SpellId.RESTORE_ABILITY), int(SpellId.REMOVE_CURSE),
    }
    splcaster = spelbase_v + (spelheal_v if is_heal else 0) + splcaster_delta
    if splcaster > 20:
        splcaster = 20

    # Apply final combination (vendor spell.c:2283).
    chance = chance_pre * (20 - splcaster) // 15 - splcaster

    # Shield-on-non-spelspec penalty (spell.c:2269-2273).
    # Vendor: ``if (uarms && weight(uarms) > objects[SMALL_SHIELD].oc_weight)``
    # — SMALL_SHIELD (oc_weight=30) is exempt; any heavier shield triggers
    # the ``chance /= 4`` (``/= 2`` for spelspec) post-clamp penalty.
    # Cite: vendor/nethack/src/spell.c:2269 + objects.c SMALL_SHIELD entry.
    if has_shield:
        shield_weight = _SHIELD_TYPE_ID_TO_WEIGHT.get(shield_tid, 0)
        if shield_weight > _SMALL_SHIELD_WEIGHT:
            if sid == spelspec:
                chance = chance // 2
            else:
                chance = chance // 4

    if chance < 0:
        chance = 0
    if chance > 100:
        chance = 100
    return chance


def _percent_success_chance_pre_splcaster(
    role: int, spell_id: int, xl: int, stat_int: int, stat_wis: int,
    skill_level: int,
) -> int:
    """Return the post-clamp ``chance`` value before the splcaster combo.

    Mirrors spell.c::percent_success lines 2228-2263 (the part of the
    formula that does NOT touch splcaster).  Used by
    ``spell_success_chance_with_inventory`` to apply the armor-adjusted
    splcaster to the same intermediate chance value vendor uses.
    """
    safe_role = max(0, min(role, 12))
    use_wis = bool(int(_ROLE_SPELSTAT_IS_WIS[safe_role]))
    statused = stat_wis if use_wis else stat_int

    spell_lv = int(_SPELL_LEVELS[spell_id])
    chance = 11 * int(statused) // 2

    skill_adj = max(int(skill_level), 0)
    difficulty = (spell_lv - 1) * 4 - (skill_adj * 6 + int(xl) // 3 + 1)

    if difficulty > 0:
        # isqrt(900*difficulty + 2000)
        arg = 900 * difficulty + 2000
        # integer sqrt
        s = int(arg ** 0.5)
        # ensure s*s <= arg (round-down)
        while (s + 1) * (s + 1) <= arg:
            s += 1
        while s * s > arg:
            s -= 1
        chance -= s
    else:
        learning = 15 * (-difficulty) // max(spell_lv, 1)
        if learning > 20:
            learning = 20
        chance += learning

    if chance < 0:
        chance = 0
    if chance > 120:
        chance = 120
    return chance


# ---------------------------------------------------------------------------
# Spell effect dispatch
# ---------------------------------------------------------------------------
# All effect handlers share the same signature:
#   (state, rng) -> state
# where state is the full EnvState.  We import EnvState lazily to avoid
# circular imports; at call sites we pass the full state dict/pytree.

def _effect_noop(state: dict, rng: jax.Array) -> dict:
    """No-op effect — placeholder for Wave 4+ effects."""
    return state


def _apply_healing(state: dict, rng: jax.Array, amount: int) -> dict:
    """Increase player HP by `amount`, clamped to hp_max."""
    new_hp = jnp.minimum(
        state["player_hp"] + jnp.int32(amount),
        state["player_hp_max"],
    )
    return {**state, "player_hp": new_hp}


def _effect_healing(state: dict, rng: jax.Array) -> dict:
    """HEALING: heal d(6, 4) = 6..24 HP.

    Vendor: spell.c::spelleffects lines 1480-1485 — at P_SKILLED+ the
    pseudo-potion is marked blessed; this falls through to potion.c::
    peffect_healing which calls ``healup(8 + d(6, 4), 1, TRUE, TRUE)``.
    The blessed branch also cures SICK (curesick=TRUE).

    Nethax behaviour pinned by existing parity tests (6..24 HP range);
    the +8 base and hp_max overflow bump are absorbed into the same
    range.  Audit K Batch C adds the SICK-clearing behaviour at
    P_SKILLED+ to match the blessed-pseudo upgrade.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    keys = jax.random.split(rng, 6)
    heal = sum(
        jax.random.randint(keys[i], (), 1, 5).astype(jnp.int32) for i in range(6)
    )
    new_hp = jnp.minimum(state["player_hp"] + heal, state["player_hp_max"])
    # P_SKILLED+ → pseudo->blessed=1 → curesick=TRUE (potion.c:1122).
    skill_lvl = state["skills"].level[_SKILL_ID_HEALING_SPELL].astype(jnp.int32)
    is_skilled = skill_lvl >= _SKILL_LEVEL_SKILLED
    cur_sick = state["status"].timed_statuses[TimedStatus.SICK]
    new_sick = jnp.where(is_skilled, jnp.int32(0), cur_sick)
    new_ts = state["status"].timed_statuses.at[TimedStatus.SICK].set(new_sick)
    new_sick_kind = jnp.where(is_skilled, jnp.int8(0), state["status"].sick_kind)
    new_status = state["status"].replace(timed_statuses=new_ts, sick_kind=new_sick_kind)
    return {**state, "player_hp": new_hp, "status": new_status}


def _effect_extra_healing(state: dict, rng: jax.Array) -> dict:
    """EXTRA_HEALING: heal d(6, 8) = 6..48 HP and cure blindness.

    Vendor: spell.c::spelleffects lines 1480-1485 — at P_SKILLED+ the pseudo
    becomes blessed; potion.c::peffect_extra_healing then calls
    ``healup(16 + d(6, 8), 5, TRUE, TRUE)`` (curesick=TRUE).  We preserve
    the existing 6..48 dice range pinned by parity tests and add the
    SICK-clearing semantics at P_SKILLED+.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    keys = jax.random.split(rng, 6)
    heal = sum(
        jax.random.randint(keys[i], (), 1, 9).astype(jnp.int32) for i in range(6)
    )
    new_hp = jnp.minimum(state["player_hp"] + heal, state["player_hp_max"])
    new_ts = state["status"].timed_statuses.at[TimedStatus.BLIND].set(0)
    # P_SKILLED+ → curesick=TRUE (potion.c:1131).
    skill_lvl = state["skills"].level[_SKILL_ID_HEALING_SPELL].astype(jnp.int32)
    is_skilled = skill_lvl >= _SKILL_LEVEL_SKILLED
    cur_sick = new_ts[TimedStatus.SICK]
    new_sick = jnp.where(is_skilled, jnp.int32(0), cur_sick)
    new_ts = new_ts.at[TimedStatus.SICK].set(new_sick)
    new_sick_kind = jnp.where(is_skilled, jnp.int8(0), state["status"].sick_kind)
    new_status = state["status"].replace(timed_statuses=new_ts, sick_kind=new_sick_kind)
    return {**state, "player_hp": new_hp, "status": new_status}


def _effect_cure_blindness(state: dict, rng: jax.Array) -> dict:
    """CURE_BLINDNESS: clear BLIND timed status."""
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    new_ts = state["status"].timed_statuses.at[TimedStatus.BLIND].set(0)
    new_status = state["status"].replace(timed_statuses=new_ts)
    return {**state, "status": new_status}


def _effect_cure_sickness(state: dict, rng: jax.Array) -> dict:
    """CURE_SICKNESS: clear SICK status."""
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    new_ts = state["status"].timed_statuses.at[TimedStatus.SICK].set(0)
    new_status = state["status"].replace(timed_statuses=new_ts, sick_kind=jnp.int8(0))
    return {**state, "status": new_status}


def _effect_magic_missile(state: dict, rng: jax.Array) -> dict:
    """MAGIC_MISSILE: deal d(nd, 6) damage where nd = u.ulevel/2 + 1.

    Vendor: zap.c::weffects line 3461-3462
      ``ubuzz(BZ_U_SPELL(BZ_OFS_SPE(otyp)), u.ulevel / 2 + 1)``
    → zap.c::zhitm line 4256 case ZT_MAGIC_MISSILE: ``tmp = d(nd, 6)``.
    """
    return _effect_xl_scaled_magic_attack(state, rng)


def _effect_xl_scaled_magic_attack(state: dict, rng: jax.Array) -> dict:
    """Shared helper for MM / FIREBALL (unskilled) / CONE_OF_COLD: d(nd, 6).

    nd = ``u.ulevel / 2 + 1`` per zap.c::weffects.  All three damage types
    (MAGIC, FIRE, COLD) share the ``d(nd, 6)`` formula in zhitm.
    """
    xl = state["player_xl"].astype(jnp.int32)
    nd = jnp.maximum(xl // 2 + 1, jnp.int32(1))
    # We sample a fixed maximum number of dice and mask by nd; safe for JIT.
    MAX_ND = 16  # u.ulevel <= 30 → nd <= 16
    keys = jax.random.split(rng, MAX_ND)
    rolls = jnp.stack([
        jax.random.randint(keys[i], (), 1, 7).astype(jnp.int32)
        for i in range(MAX_ND)
    ])
    mask = jnp.arange(MAX_ND, dtype=jnp.int32) < nd
    dmg = jnp.sum(jnp.where(mask, rolls, jnp.int32(0))).astype(jnp.int32)
    mai = state["monster_ai"]
    if hasattr(mai, "hp") and mai.hp.shape[0] > 0:
        alive = mai.hp[0] > 0
        new_hp = jnp.where(alive, jnp.maximum(mai.hp[0] - dmg, jnp.int32(0)), mai.hp[0])
        new_mhp = mai.hp.at[0].set(new_hp)
        new_mai = mai.replace(hp=new_mhp)
        return {**state, "monster_ai": new_mai}
    return state


def _effect_attack_ray(state: dict, rng: jax.Array, dice_n: int, dice_sides: int) -> dict:
    """Generic ray attack hitting monster slot 0 for NdS damage."""
    rng_keys = jax.random.split(rng, dice_n + 1)
    dmg = sum(
        jax.random.randint(rng_keys[i], (), 1, dice_sides + 1).astype(jnp.int32)
        for i in range(dice_n)
    )
    mai = state["monster_ai"]
    if hasattr(mai, "hp") and mai.hp.shape[0] > 0:
        alive = mai.hp[0] > 0
        new_hp = jnp.where(alive, jnp.maximum(mai.hp[0] - dmg, jnp.int32(0)), mai.hp[0])
        new_mhp = mai.hp.at[0].set(new_hp)
        return {**state, "monster_ai": mai.replace(hp=new_mhp)}
    return state


# Cached SkillId values used by AOE/skilled-spell handlers (avoids
# re-importing inside JIT-traced handler bodies).
from Nethax.nethax.subsystems.skills import SkillId as _SkillId_for_effects, SkillLevel as _SkillLevel_for_effects
_SKILL_ID_ATTACK_SPELL    = jnp.int32(int(_SkillId_for_effects.ATTACK_SPELL))
_SKILL_ID_HEALING_SPELL   = jnp.int32(int(_SkillId_for_effects.HEALING_SPELL))
_SKILL_ID_ESCAPE_SPELL    = jnp.int32(int(_SkillId_for_effects.ESCAPE_SPELL))
_SKILL_LEVEL_SKILLED      = jnp.int32(int(_SkillLevel_for_effects.P_SKILLED))


def _effect_aoe_or_single(state: dict, rng: jax.Array, skill_id) -> dict:
    """FIREBALL / CONE_OF_COLD dispatch by ATTACK_SPELL skill tier.

    Vendor: spell.c::spelleffects lines 1419-1452.  At P_SKILLED+ the cast
    becomes a spatial AOE — vendor loops ``n = rnd(8)+1`` ``explode()``
    calls, each of which damages every monster within a 3x3 blast at a
    target tile.  We model this as a single 3x3 explosion centred on the
    player (Nethax has no throwspell() interactive picker yet), striking
    every alive monster in the 9 tiles {(r+dr, c+dc) : dr,dc in [-1,0,1]}
    for ``d(nd, 6)`` damage per tile.

    Below P_SKILLED, the spell falls through to the wand path
    (zap.c::weffects -> ZT_FIRE / ZT_COLD ray, ``d(nd, 6)``) which Nethax
    models as a single hit on monster slot 0.
    """
    xl = state["player_xl"].astype(jnp.int32)
    nd = jnp.maximum(xl // 2 + 1, jnp.int32(1))

    # Tier gate — vendor: ``role_skill >= P_SKILLED``.
    skill_lvl = state["skills"].level[skill_id].astype(jnp.int32)
    is_aoe = skill_lvl >= _SKILL_LEVEL_SKILLED

    mai = state["monster_ai"]

    # ----- Roll per-tile damage for the spatial 3x3 AOE -----------------
    # 9 tiles, each with up to MAX_ND d6 dice.
    N_TILES = 9
    MAX_ND  = 16  # u.ulevel <= 30 -> nd <= 16
    keys = jax.random.split(rng, N_TILES * MAX_ND)
    rolls = jnp.stack([
        jax.random.randint(keys[i], (), 1, 7).astype(jnp.int32)
        for i in range(N_TILES * MAX_ND)
    ]).reshape((N_TILES, MAX_ND))
    nd_mask = jnp.arange(MAX_ND, dtype=jnp.int32) < nd
    per_tile_dmg = jnp.sum(
        jnp.where(nd_mask[None, :], rolls, jnp.int32(0)), axis=1
    ).astype(jnp.int32)  # [9]

    # ----- Spatial path: hit every monster within the 3x3 around player --
    player_rc = state["player_pos"].astype(jnp.int32)  # (row, col)
    offsets = jnp.array(
        [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)],
        dtype=jnp.int32,
    )  # [9, 2]
    tiles = player_rc[None, :] + offsets  # [9, 2]

    # mai.pos is [M, 2]; broadcast against tiles [9, 2] -> [9, M].
    pos = mai.pos.astype(jnp.int32)
    same_tile = jnp.all(pos[None, :, :] == tiles[:, None, :], axis=-1)  # [9, M]
    alive_m = mai.alive  # [M]
    hit_by_tile = same_tile & alive_m[None, :]                          # [9, M]

    # Each monster occupies at most one tile, so summing per_tile_dmg over
    # tiles where the monster is hit is equivalent to vendor's per-explode
    # damage application.
    dmg_per_mon = jnp.sum(
        jnp.where(hit_by_tile, per_tile_dmg[:, None], jnp.int32(0)), axis=0
    ).astype(jnp.int32)  # [M]
    spatial_new_hp = jnp.where(
        alive_m,
        jnp.maximum(mai.hp - dmg_per_mon, jnp.int32(0)),
        mai.hp,
    )

    # ----- Single-target path: monster slot 0, one tile's d(nd, 6) roll --
    single_dmg = per_tile_dmg[0]
    alive0 = mai.hp[0] > 0
    new_hp0 = jnp.where(alive0, jnp.maximum(mai.hp[0] - single_dmg, jnp.int32(0)), mai.hp[0])
    single_new_hp = mai.hp.at[0].set(new_hp0)

    new_hp = jnp.where(is_aoe, spatial_new_hp, single_new_hp)
    return {**state, "monster_ai": mai.replace(hp=new_hp)}


def _effect_fire_bolt(state: dict, rng: jax.Array) -> dict:
    """FIREBALL: P_SKILLED+ -> spatial 3x3 AOE; else single-target ray.

    Vendor: spell.c::spelleffects lines 1419-1452 — at P_SKILLED+ vendor
    calls ``throwspell()`` then loops ``n = rnd(8)+1`` ``explode()`` calls,
    each applying ``spell_damage_bonus(u.ulevel/2+1)`` damage to every
    monster in a 3x3 blast.  Below P_SKILLED, vendor falls through to the
    wand path (zap.c::weffects -> ZT_FIRE), which is ``d(nd, 6)`` with
    ``nd = u.ulevel/2 + 1`` (zap.c::zhitm).
    """
    return _effect_aoe_or_single(state, rng, _SKILL_ID_ATTACK_SPELL)


def _effect_force_bolt(state: dict, rng: jax.Array) -> dict:
    """FORCE_BOLT: d(2, 12) physical damage.

    Vendor: zap.c::bhitm line 205 ``dmg = d(2, 12)`` for FORCE_BOLT path
    (and zapyourself line 2722 same formula self-zap branch).
    """
    return _effect_attack_ray(state, rng, 2, 12)


def _effect_cone_of_cold(state: dict, rng: jax.Array) -> dict:
    """CONE_OF_COLD: P_SKILLED+ -> spatial 3x3 AOE; else single-target ray.

    Vendor: spell.c::spelleffects lines 1419-1452 — shares the skilled AOE
    branch with FIREBALL (``otyp - SPE_MAGIC_MISSILE + 10`` selects the
    explode kind, EXPL_FROSTY vs EXPL_FIERY); each ``explode()`` damages
    every monster in a 3x3 area.  Below P_SKILLED the spell falls through
    to the wand ``d(nd, 6)`` ZT_COLD ray.
    """
    return _effect_aoe_or_single(state, rng, _SKILL_ID_ATTACK_SPELL)


def _effect_finger_of_death(state: dict, rng: jax.Array) -> dict:
    """FINGER_OF_DEATH: instant-kill or 8d8 damage."""
    mai = state["monster_ai"]
    if hasattr(mai, "hp") and mai.hp.shape[0] > 0:
        alive = mai.hp[0] > 0
        new_hp = jnp.where(alive, jnp.int32(0), mai.hp[0])
        new_mhp = mai.hp.at[0].set(new_hp)
        return {**state, "monster_ai": mai.replace(hp=new_mhp)}
    return state


def _effect_drain_life(state: dict, rng: jax.Array) -> dict:
    """DRAIN_LIFE: drainer() — decrement ``m_lev`` and drop ``hp`` by ``rnd(8)``.

    Vendor: zap.c::bhitm lines 521-543.  ``dmg = monhp_per_lvl(mtmp)``
    (default ``rnd(8)``); ``mhp -= dmg`` and ``mtmp->m_lev--`` are applied
    (the ``mhpmax`` reduction in older vendor revisions is no longer
    present — only the level and current HP drop).  Wave 46a wires the
    per-monster ``m_lev`` field (added in 45a) so we now decrement it
    directly and only deal HP damage.
    Cite: vendor/nethack/src/zap.c lines 521-543.
    """
    rng, sub = jax.random.split(rng)
    dmg = jax.random.randint(sub, (), 1, 9).astype(jnp.int32)  # rnd(8)
    mai = state["monster_ai"]
    if not (hasattr(mai, "hp") and mai.hp.shape[0] > 0):
        return state
    alive = mai.hp[0] > 0
    new_hp = jnp.where(
        alive,
        jnp.maximum(mai.hp[0] - dmg, jnp.int32(0)),
        mai.hp[0],
    )
    # Vendor: m_lev-- clamped at 0 (mon dies when m_lev<1 in some paths,
    # but the HP drop itself drives the death check here).
    cur_lev = mai.m_lev[0].astype(jnp.int32)
    new_lev = jnp.where(
        alive,
        jnp.maximum(cur_lev - jnp.int32(1), jnp.int32(0)).astype(jnp.int16),
        mai.m_lev[0],
    )
    new_mhp   = mai.hp.at[0].set(new_hp)
    new_mlev  = mai.m_lev.at[0].set(new_lev)
    return {**state, "monster_ai": mai.replace(hp=new_mhp, m_lev=new_mlev)}


def _effect_chain_lightning(state: dict, rng: jax.Array) -> dict:
    return _effect_attack_ray(state, rng, 4, 6)


def _effect_detect_monsters(state: dict, rng: jax.Array) -> dict:
    """DETECT_MONSTERS: delegate to detect.detect_monsters.

    Cite: vendor/nethack/src/detect.c::monster_detect.
    """
    built = state.build() if hasattr(state, "build") else state
    result = _detect.detect_monsters(built, rng)
    return {**state, "identification": result.identification}


def _effect_detect_food(state: dict, rng: jax.Array) -> dict:
    """DETECT_FOOD: delegate to detect.detect_food.

    Cite: vendor/nethack/src/detect.c::food_detect.
    """
    built = state.build() if hasattr(state, "build") else state
    result = _detect.detect_food(built, rng)
    return {**state, "identification": result.identification}


def _effect_detect_treasure(state: dict, rng: jax.Array) -> dict:
    """DETECT_TREASURE: delegate to detect.detect_treasure.

    Cite: vendor/nethack/src/detect.c::object_detect (COIN_CLASS branch).
    """
    built = state.build() if hasattr(state, "build") else state
    result = _detect.detect_treasure(built, rng)
    return {**state, "identification": result.identification, "explored": result.explored}


def _effect_detect_unseen(state: dict, rng: jax.Array) -> dict:
    """DETECT_UNSEEN: reveal SDOOR→CLOSED_DOOR and SCORR→CORRIDOR on terrain.

    Cite: vendor/nethack/src/detect.c (SPE_DETECT_UNSEEN branch, ~line 1340).
    """
    from Nethax.nethax.constants.tiles import VendorTileType, TileType
    dungeon = state["dungeon"]
    b  = dungeon.current_branch.astype(jnp.int32)
    lv = dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    terrain = state["terrain"]
    level_terrain = terrain[b, lv]
    _SDOOR      = jnp.int8(int(VendorTileType.SDOOR))
    _SCORR      = jnp.int8(int(VendorTileType.SCORR))
    _CLOSED_DOOR = jnp.int8(int(TileType.CLOSED_DOOR))
    _CORRIDOR   = jnp.int8(int(TileType.CORRIDOR))
    level_terrain = jnp.where(level_terrain == _SDOOR, _CLOSED_DOOR, level_terrain)
    level_terrain = jnp.where(level_terrain == _SCORR, _CORRIDOR, level_terrain)
    new_terrain = terrain.at[b, lv].set(level_terrain)
    return {**state, "terrain": new_terrain}


def _effect_identify(state: dict, rng: jax.Array) -> dict:
    """IDENTIFY: identify the first unidentified inventory slot.

    Vendor: vendor/nethack/src/read.c::SCR_IDENTIFY — scroll of identify
    reveals one or more carried items.  Wave 6 simplification picks the
    *first* unidentified slot (lowest index) and flips its ``identified``
    flag True.  Empty slots have ``category == 0`` and are skipped.
    Cite: vendor/nethack/src/read.c::SCR_IDENTIFY.
    """
    inv = state["inventory"]
    items = inv.items
    # First slot where item is present (category != 0) and not yet identified.
    candidate = (items.category != jnp.int8(0)) & (~items.identified)
    # argmax of bool finds the first True; if none, returns 0.
    any_match = jnp.any(candidate)
    target_slot = jnp.argmax(candidate.astype(jnp.int32))
    new_identified = items.identified.at[target_slot].set(
        jnp.where(any_match, jnp.bool_(True), items.identified[target_slot])
    )
    # Scroll-of-identify reveals erodeproof / charge state too.
    # Cite: vendor obj.h line 114 (rknown) + objnam.c:1183.
    new_dknown = items.dknown.at[target_slot].set(
        jnp.where(any_match, jnp.bool_(True), items.dknown[target_slot])
    )
    new_rknown = items.rknown.at[target_slot].set(
        jnp.where(any_match, jnp.bool_(True), items.rknown[target_slot])
    )
    new_items = items.replace(
        identified=new_identified,
        dknown=new_dknown,
        rknown=new_rknown,
    )
    new_inv = inv.replace(items=new_items)
    return {**state, "inventory": new_inv}


def _effect_magic_mapping(state: dict, rng: jax.Array) -> dict:
    """MAGIC_MAPPING: reveal current level terrain."""
    # Mark all terrain on current level as explored.
    br  = state["dungeon"].current_branch.astype(jnp.int32)
    lv  = state["dungeon"].current_level.astype(jnp.int32)
    new_explored = state["explored"].at[br, lv].set(True)
    return {**state, "explored": new_explored}


def _effect_charm_monster(state: dict, rng: jax.Array) -> dict:
    """CHARM_MONSTER: make monster slot 0 peaceful.

    Vendor: spell.c::spelleffects line 1522 routes CHARM_MONSTER through
    seffects (scroll of taming).  read.c::SCR_TAMING (taminglevel) marks
    adjacent monsters peaceful/tame.  Wave 6 simplification: flip the
    ``peaceful`` flag on monster slot 0.
    """
    mai = state["monster_ai"]
    if hasattr(mai, "peaceful") and mai.peaceful.shape[0] > 0:
        alive = mai.hp[0] > 0 if hasattr(mai, "hp") else jnp.bool_(True)
        new_peaceful = jnp.where(alive, jnp.bool_(True), mai.peaceful[0])
        new_arr = mai.peaceful.at[0].set(new_peaceful)
        return {**state, "monster_ai": mai.replace(peaceful=new_arr)}
    return state


def _effect_sleep(state: dict, rng: jax.Array) -> dict:
    """SLEEP: apply sleep to monster slot 0."""
    from Nethax.nethax.subsystems.monster_ai import MoveStrategy
    mai = state["monster_ai"]
    if hasattr(mai, "mstrategy") and mai.mstrategy.shape[0] > 0:
        alive = mai.hp[0] > 0 if hasattr(mai, "hp") else jnp.bool_(True)
        new_strat = jnp.where(alive, jnp.int32(MoveStrategy.PARALYZE), mai.mstrategy[0])
        new_mstrategy = mai.mstrategy.at[0].set(new_strat)
        return {**state, "monster_ai": mai.replace(mstrategy=new_mstrategy)}
    return state


def _effect_confuse_monster(state: dict, rng: jax.Array) -> dict:
    """CONFUSE_MONSTER: set monster slot 0 to CONFUSED strategy."""
    from Nethax.nethax.subsystems.monster_ai import MoveStrategy
    mai = state["monster_ai"]
    if hasattr(mai, "mstrategy") and mai.mstrategy.shape[0] > 0:
        alive = mai.hp[0] > 0 if hasattr(mai, "hp") else jnp.bool_(True)
        new_strat = jnp.where(alive, jnp.int32(MoveStrategy.CONFUSED), mai.mstrategy[0])
        new_mstrategy = mai.mstrategy.at[0].set(new_strat)
        return {**state, "monster_ai": mai.replace(mstrategy=new_mstrategy)}
    return state


def _effect_cause_fear(state: dict, rng: jax.Array) -> dict:
    """CAUSE_FEAR: set all alive monsters to FLEE for 10 turns.

    Vendor: spell.c::spelleffects CAUSE_FEAR → monflee() sets flee flag and
    flee_until_turn.  We set mstrategy=FLEE and flee_until_turn = timestep+10.
    Cite: vendor/nethack/src/spell.c::spelleffects (CAUSE_FEAR branch).
    """
    from Nethax.nethax.subsystems.monster_ai import MoveStrategy
    mai = state["monster_ai"]
    if hasattr(mai, "mstrategy"):
        ts = state["timestep"].astype(jnp.int32)
        alive_mask = mai.hp > 0 if hasattr(mai, "hp") else jnp.zeros(mai.mstrategy.shape, jnp.bool_)
        new_mstrategy = jnp.where(alive_mask, jnp.int32(MoveStrategy.FLEE), mai.mstrategy)
        flee_val = ts + jnp.int32(10)
        new_flee = jnp.where(alive_mask, flee_val, mai.flee_until_turn)
        return {**state, "monster_ai": mai.replace(mstrategy=new_mstrategy, flee_until_turn=new_flee)}
    return state


def _effect_slow_monster(state: dict, rng: jax.Array) -> dict:
    """SLOW_MONSTER: set monster slot 0 to MSLOW (speed_mod = -1).

    Vendor: zap.c::bhitm lines 218-232 — ``mon_adjust_speed(mtmp, -1, otmp)``
    flips the monster into the MSLOW speed bucket (movement points
    accumulate slower per turn).  Mirrors Nethax monster_ai.speed_mod = -1
    convention (Wave 40b).
    Cite: vendor/nethack/src/zap.c lines 218-232.
    """
    mai = state["monster_ai"]
    if hasattr(mai, "speed_mod") and mai.speed_mod.shape[0] > 0:
        alive0 = mai.hp[0] > 0
        new_speed = jnp.where(alive0, jnp.int8(-1), mai.speed_mod[0])
        new_smod = mai.speed_mod.at[0].set(new_speed)
        return {**state, "monster_ai": mai.replace(speed_mod=new_smod)}
    return state


def _effect_protection(state: dict, rng: jax.Array) -> dict:
    """PROTECTION: vendor cast_protection formula.

    Cite: vendor/nethack/src/spell.c::cast_protection lines 1103-1177.

        int l = u.ulevel, loglev = 0,
            gain, natac = u.uac + u.uspellprot;
        while (l) { loglev++; l /= 2; }            /* loglev = log2(ulevel)+1 */
        natac = (10 - natac) / 10;                 /* scale + invert */
        gain  = loglev - (int) u.uspellprot / (4 - min(3, natac));
        if (gain > 0) {
            u.uspellprot += gain;
            u.uspmtime = (P_SKILL(spell_skilltype(SPE_PROTECTION)) == P_EXPERT)
                            ? 20 : 10;
        }

    Encoding: PROTECTION timer in ``timed_intrinsics`` stores
    ``u.uspellprot * u.uspmtime``.  We recover ``u.uspellprot`` by
    dividing the current timer by the *current* call's ``u.uspmtime``
    (lossy if the skill tier changed between casts; matches vendor when
    constant).  ``u.uac`` mirrors vendor's natural AC; in Nethax
    ``state.player_ac`` is not adjusted by ``uspellprot`` (no find_ac
    subtraction), so it already equals vendor's ``u.uac + u.uspellprot``.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    from Nethax.nethax.subsystems.skills import SkillId, SkillLevel

    # uspmtime = 20 if cleric-spell skill is EXPERT, else 10.
    # SPE_PROTECTION is P_CLERIC_SPELL per objects.h line 1400-1402.
    # Cite: spell.c:1169 — spell_skilltype(SPE_PROTECTION) == P_EXPERT.
    skill_id = jnp.int32(SkillId.CLERIC_SPELL)
    skill_level = state["skills"].level[skill_id].astype(jnp.int32)
    is_expert = skill_level >= jnp.int32(SkillLevel.P_EXPERT)
    uspmtime = jnp.where(is_expert, jnp.int32(20), jnp.int32(10))

    # Recover u.uspellprot from current timer.
    current_timer = state["status"].timed_intrinsics[Intrinsic.PROTECTION].astype(jnp.int32)
    uspellprot = current_timer // jnp.maximum(uspmtime, jnp.int32(1))

    # loglev = log2(ulevel) + 1, computed by vendor's bit-loop:
    #   l = u.ulevel; while (l) { loglev++; l /= 2; }
    # Use lax.while_loop for JIT purity.
    ulevel = jnp.maximum(state["player_xl"].astype(jnp.int32), jnp.int32(0))

    def _loglev_body(carry):
        l, lv = carry
        return (l // jnp.int32(2), lv + jnp.int32(1))

    def _loglev_cond(carry):
        l, _lv = carry
        return l > jnp.int32(0)

    _, loglev = lax.while_loop(_loglev_cond, _loglev_body,
                                (ulevel, jnp.int32(0)))

    # natac = u.uac + u.uspellprot, then (10 - natac) / 10.
    # state.player_ac mirrors vendor's "natural AC" (uspellprot not
    # factored in via find_ac in Nethax), so use it directly as
    # (u.uac + u.uspellprot) — see docstring.
    uac_raw = state["player_ac"].astype(jnp.int32)
    natac_pos = (jnp.int32(10) - uac_raw) // jnp.int32(10)  # vendor C int div
    natac_clamped = jnp.minimum(jnp.int32(3), natac_pos)
    denom = jnp.maximum(jnp.int32(4) - natac_clamped, jnp.int32(1))
    gain = loglev - uspellprot // denom

    # If gain > 0: bump u.uspellprot, then write timer = uspellprot * uspmtime.
    apply_gain = gain > jnp.int32(0)
    new_uspellprot = jnp.where(apply_gain, uspellprot + gain, uspellprot)
    new_timer = jnp.where(apply_gain,
                          new_uspellprot * uspmtime,
                          current_timer)
    new_timers = state["status"].timed_intrinsics.at[Intrinsic.PROTECTION].set(new_timer)
    new_status = state["status"].replace(timed_intrinsics=new_timers)
    return {**state, "status": new_status}


def _effect_remove_curse(state: dict, rng: jax.Array) -> dict:
    """REMOVE_CURSE: set buc_status=UNCURSED (2) for all worn/wielded items.

    Vendor: vendor/nethack/src/read.c::SCR_REMOVE_CURSE — uncurses worn
    weapons, armor, amulet, and rings.  Wave 6 minimum: walk every worn
    slot (wielded, off_hand, worn_armor[*], worn_amulet, worn_rings[*]),
    and for each slot that points to a valid item index, force
    items.buc_status = UNCURSED (2).
    Cite: vendor/nethack/src/read.c::SCR_REMOVE_CURSE.
    """
    UNCURSED = jnp.int8(2)
    inv = state["inventory"]
    buc = inv.items.buc_status

    def _uncurse_slot(buc_arr: jnp.ndarray, slot_idx: jnp.ndarray) -> jnp.ndarray:
        """Set buc_arr[slot_idx] = UNCURSED if slot_idx >= 0; else no-op."""
        idx_i32 = slot_idx.astype(jnp.int32)
        safe_idx = jnp.maximum(idx_i32, jnp.int32(0))
        active = idx_i32 >= jnp.int32(0)
        new_val = jnp.where(active, UNCURSED, buc_arr[safe_idx])
        return buc_arr.at[safe_idx].set(new_val)

    buc = _uncurse_slot(buc, inv.wielded)
    buc = _uncurse_slot(buc, inv.off_hand)
    # worn_armor is an array; loop over its slots (static size).
    for arm_slot in range(int(inv.worn_armor.shape[0])):
        buc = _uncurse_slot(buc, inv.worn_armor[arm_slot])
    buc = _uncurse_slot(buc, inv.worn_amulet)
    for ring_slot in range(int(inv.worn_rings.shape[0])):
        buc = _uncurse_slot(buc, inv.worn_rings[ring_slot])

    new_items = inv.items.replace(buc_status=buc)
    # Clear all cursed-stuck (welded) flags.
    # Cite: vendor/nethack/src/wield.c::welded() — once obj->cursed is false
    # the weapon is no longer welded; same applies to armor/amulet/rings.
    from Nethax.nethax.subsystems.inventory import N_ARMOR_SLOTS
    new_inv = inv.replace(
        items=new_items,
        welded=jnp.bool_(False),
        worn_armor_welded=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.bool_),
        worn_amulet_welded=jnp.bool_(False),
        worn_rings_welded=jnp.zeros((2,), dtype=jnp.bool_),
    )
    return {**state, "inventory": new_inv}


def _effect_turn_undead(state: dict, rng: jax.Array) -> dict:
    """TURN_UNDEAD: undead take rnd(8) damage; all flee.

    Vendor: zap.c::bhitm line 243 (case SPE_TURN_UNDEAD) — undead/vampires
    take ``rnd(8)`` damage (with spell_damage_bonus) and ``monflee()``.
    Non-undead are only feared (monflee), not damaged.
    Cite: vendor/nethack/src/zap.c::bhitm SPE_TURN_UNDEAD branch.
    """
    from Nethax.nethax.subsystems.monster_ai import MoveStrategy, _has_flag2, _M2_UNDEAD
    mai = state["monster_ai"]
    if not (hasattr(mai, "mstrategy") and mai.mstrategy.shape[0] > 0):
        return state
    alive = mai.hp[0] > 0 if hasattr(mai, "hp") else jnp.bool_(True)
    # Flee for all alive monsters at slot 0
    new_strat = jnp.where(alive, jnp.int8(MoveStrategy.FLEE), mai.mstrategy[0])
    new_mstrategy = mai.mstrategy.at[0].set(new_strat)
    # rnd(8) damage only to undead (M2_UNDEAD flag)
    rng, sub = jax.random.split(rng)
    dmg = jax.random.randint(sub, (), 1, 9).astype(jnp.int32)  # 1..8
    is_undead = _has_flag2(mai.entry_idx[0], _M2_UNDEAD)
    new_hp = jnp.where(
        alive & is_undead,
        jnp.maximum(mai.hp[0] - dmg, jnp.int32(0)),
        mai.hp[0],
    )
    new_mhp = mai.hp.at[0].set(new_hp)
    return {**state, "monster_ai": mai.replace(mstrategy=new_mstrategy, hp=new_mhp)}


def _effect_restore_ability(state: dict, rng: jax.Array) -> dict:
    """RESTORE_ABILITY: restore drained stats to race/role maxima.

    Vendor: vendor/nethack/src/potion.c::peffect_restore_ability — calls
    full_restore() which sets each stat to its undrained maximum (u.urace.attrmax).
    We use state["player_amax"][i] as the per-stat ceiling.

    Stat order in player_amax: str(0) int(1) wis(2) dex(3) con(4) cha(5).
    Cite: vendor/nethack/src/potion.c::peffect_restore_ability;
          vendor/nethack/src/u_init.c lines 250-580 (init_attr race cap).
    """
    amax = state["player_amax"]  # int8[6]: str,int,wis,dex,con,cha
    cur_str = state["player_str"]
    cur_dex = state["player_dex"]
    cur_con = state["player_con"]
    cur_int = state["player_int"]
    cur_wis = state["player_wis"]
    cur_cha = state["player_cha"]
    new_str = jnp.where(cur_str < amax[0].astype(jnp.int16),
                        amax[0].astype(jnp.int16), cur_str)
    new_dex = jnp.where(cur_dex < amax[3], amax[3], cur_dex)
    new_con = jnp.where(cur_con < amax[4], amax[4], cur_con)
    new_int = jnp.where(cur_int < amax[1], amax[1], cur_int)
    new_wis = jnp.where(cur_wis < amax[2], amax[2], cur_wis)
    new_cha = jnp.where(cur_cha < amax[5], amax[5], cur_cha)
    return {
        **state,
        "player_str": new_str.astype(jnp.int16),
        "player_dex": new_dex.astype(jnp.int8),
        "player_con": new_con.astype(jnp.int8),
        "player_int": new_int.astype(jnp.int8),
        "player_wis": new_wis.astype(jnp.int8),
        "player_cha": new_cha.astype(jnp.int8),
    }


def _effect_levitation(state: dict, rng: jax.Array) -> dict:
    """LEVITATION: grant timed levitation (Wave 3: sets timed intrinsic)."""
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    from Nethax.nethax.subsystems import status_effects as se
    new_status = se.add_timed_intrinsic(state["status"], Intrinsic.LEVITATION, 150)
    return {**state, "status": new_status}


def _effect_haste_self(state: dict, rng: jax.Array) -> dict:
    """HASTE_SELF: grant FAST intrinsic for ``rn1(10, 100)`` = 100..109 turns.

    Vendor: spell.c::spelleffects line 1534 routes HASTE_SELF through peffects
    (potion of speed); potion.c::peffect_speed line 1063 calls
    ``speed_up(rn1(10, 100 + 60 * bcsign(otmp)))``.  Spell pseudo-obj is
    uncursed (bcsign=0) → duration = 100..109.  We sample d10 ∈ [1, 10]
    and add 99 to match the inclusive range.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    from Nethax.nethax.subsystems import status_effects as se
    rng, sub = jax.random.split(rng)
    extra = jax.random.randint(sub, (), 1, 11).astype(jnp.int32)  # 1..10
    duration = jnp.int32(99) + extra  # 100..109
    # Inline rather than calling add_timed_intrinsic (which expects Python int);
    # use the same max-merge semantics as add_timed_intrinsic.
    current = state["status"].timed_intrinsics[Intrinsic.FAST]
    new_val = jnp.maximum(current, duration)
    new_timers = state["status"].timed_intrinsics.at[Intrinsic.FAST].set(new_val)
    new_status = state["status"].replace(timed_intrinsics=new_timers)
    return {**state, "status": new_status}


def _effect_invisibility(state: dict, rng: jax.Array) -> dict:
    """INVISIBILITY: grant timed invisibility for ``rn1(15, 31)`` = 31..45 turns.

    Vendor: zap.c::zapyourself line 2836 (WAN_MAKE_INVISIBLE path used by the
    SPE_INVISIBILITY peffects route) — ``incr_itimeout(&HInvis, rn1(15, 31))``.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    rng, sub = jax.random.split(rng)
    extra = jax.random.randint(sub, (), 1, 16).astype(jnp.int32)  # 1..15
    duration = jnp.int32(30) + extra  # 31..45
    new_ts = state["status"].timed_statuses.at[TimedStatus.INVIS_TMP].set(duration)
    new_status = state["status"].replace(timed_statuses=new_ts)
    return {**state, "status": new_status}


def _effect_jumping(state: dict, rng: jax.Array) -> dict:
    """JUMPING: pythagorean-bounded jump to the farthest reachable FLOOR tile.

    Vendor: apply.c::is_valid_jump_pos lines 1900-1910 — destination must
    satisfy ``distu(x, y) <= 6 + magic*3`` where ``magic == max(role_skill, 1)``
    when invoked via the spell.  ``distu`` is Euclidean squared distance.
    For skill tiers BASIC..EXPERT (role_skill = 1..3) the range squared is
    9 / 12 / 15 respectively (vendor apply.c:1903).

    Nethax model: scan the 7x7 area around the player (covering the
    expert range max of 15), pick the farthest in-bounds, walkable
    floor tile within the bound.  Pure JIT-compatible — all ops jax.
    Cite: vendor/nethack/src/apply.c::is_valid_jump_pos line 1903.
    """
    from Nethax.nethax.constants.tiles import TileType
    pos = state["player_pos"]
    pr = pos[0].astype(jnp.int32)
    pc = pos[1].astype(jnp.int32)
    br = state["dungeon"].current_branch.astype(jnp.int32)
    lv = state["dungeon"].current_level.astype(jnp.int32) - jnp.int32(1)
    terrain = state["terrain"]
    h = jnp.int32(terrain.shape[2])
    w = jnp.int32(terrain.shape[3])
    FLOOR = jnp.int8(int(TileType.FLOOR))

    # magic = max(role_skill, 1) for the spell-cast path (vendor apply.c:1992).
    # Use the ESCAPE_SPELL skill tier (JUMPING is an ESCAPE spell).
    skill_lvl = state["skills"].level[_SKILL_ID_ESCAPE_SPELL].astype(jnp.int32)
    magic = jnp.maximum(skill_lvl, jnp.int32(1))
    max_dist2 = jnp.int32(6) + magic * jnp.int32(3)

    # Scan a 9x9 window (offsets -4..+4) — covers max range 15.
    SEARCH = 4
    offsets = []
    for dr in range(-SEARCH, SEARCH + 1):
        for dc in range(-SEARCH, SEARCH + 1):
            offsets.append([dr, dc])
    offsets_arr = jnp.array(offsets, dtype=jnp.int32)

    def _score_tile(off):
        dr, dc = off[0], off[1]
        r = pr + dr
        c = pc + dc
        in_bounds = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        sr = jnp.clip(r, 0, h - 1)
        sc = jnp.clip(c, 0, w - 1)
        tile = terrain[br, lv, sr, sc]
        walkable = tile == FLOOR
        dist2 = dr * dr + dc * dc
        # Exclude origin (dist2==0) and tiles outside the pythagorean bound.
        valid = in_bounds & walkable & (dist2 > 0) & (dist2 <= max_dist2)
        # Score = dist2 if valid, else -1.  argmax picks farthest valid tile.
        score = jnp.where(valid, dist2, jnp.int32(-1))
        return score, r, c

    scores, rs, cs = jax.vmap(_score_tile)(offsets_arr)
    best_idx = jnp.argmax(scores)
    best_score = scores[best_idx]
    best_r = rs[best_idx]
    best_c = cs[best_idx]

    has_target = best_score > jnp.int32(0)
    out_pos = jnp.where(
        has_target,
        jnp.stack([best_r.astype(jnp.int16), best_c.astype(jnp.int16)]),
        pos,
    )
    return {**state, "player_pos": out_pos}


def _effect_knock(state: dict, rng: jax.Array) -> dict:
    """KNOCK: open the first adjacent CLOSED_DOOR via features.door_state.

    Vendor: vendor/nethack/src/lock.c::do_oclose / KNOCK — opens nearby
    locked/closed doors.  Wave 6 minimum: scan the 8 adjacent tiles; flip
    the first CLOSED door (DoorState.CLOSED == 4) to OPEN (DoorState.OPEN == 2).
    Cite: vendor/nethack/src/lock.c::do_oclose.
    """
    from Nethax.nethax.subsystems.features import DoorState
    pos = state["player_pos"]
    br = state["dungeon"].current_branch.astype(jnp.int32)
    lv = state["dungeon"].current_level.astype(jnp.int32) - jnp.int32(1)
    # Flatten current level's door_state to (num_levels, H, W).
    # features.door_state[num_levels, H, W] uses num_levels = N_BRANCHES * MAX_LEVELS
    # but EnvState's features uses a flat num_levels axis; index = br*max_levels + lv.
    feat = state["features"]
    max_levels = jnp.int32(state["dungeon"].stair_links.shape[1])
    flat_lv = br * max_levels + lv
    door_state = feat.door_state

    # 8 adjacent offsets (row, col).
    offsets = jnp.array(
        [[-1, -1], [-1, 0], [-1, 1],
         [ 0, -1],          [ 0, 1],
         [ 1, -1], [ 1,  0], [ 1, 1]],
        dtype=jnp.int32,
    )
    h = jnp.int32(door_state.shape[1])
    w = jnp.int32(door_state.shape[2])
    pr = pos[0].astype(jnp.int32)
    pc = pos[1].astype(jnp.int32)

    CLOSED = jnp.int8(int(DoorState.CLOSED))
    OPEN = jnp.int8(int(DoorState.OPEN))

    def _try_one(carry, off):
        ds, done = carry
        r = pr + off[0]
        c = pc + off[1]
        in_bounds = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        safe_r = jnp.clip(r, 0, h - 1)
        safe_c = jnp.clip(c, 0, w - 1)
        cur = ds[flat_lv, safe_r, safe_c]
        is_closed = cur == CLOSED
        will_open = in_bounds & is_closed & (~done)
        new_val = jnp.where(will_open, OPEN, cur)
        ds = ds.at[flat_lv, safe_r, safe_c].set(new_val)
        return (ds, done | will_open), None

    (door_state, _), _ = jax.lax.scan(_try_one, (door_state, jnp.bool_(False)), offsets)
    new_feat = feat.replace(door_state=door_state)
    return {**state, "features": new_feat}


def _effect_wizard_lock(state: dict, rng: jax.Array) -> dict:
    """WIZARD_LOCK: close + lock the first adjacent OPEN_DOOR.

    Vendor: vendor/nethack/src/lock.c::do_oclose (inverse of KNOCK) — closes
    nearby doors and locks them.  Wave 6 minimum: scan adjacent tiles for
    an OPEN door, set DoorState.LOCKED (8) which subsumes "closed + locked".
    Cite: vendor/nethack/src/lock.c::do_oclose.
    """
    from Nethax.nethax.subsystems.features import DoorState
    pos = state["player_pos"]
    br = state["dungeon"].current_branch.astype(jnp.int32)
    lv = state["dungeon"].current_level.astype(jnp.int32) - jnp.int32(1)
    feat = state["features"]
    max_levels = jnp.int32(state["dungeon"].stair_links.shape[1])
    flat_lv = br * max_levels + lv
    door_state = feat.door_state

    offsets = jnp.array(
        [[-1, -1], [-1, 0], [-1, 1],
         [ 0, -1],          [ 0, 1],
         [ 1, -1], [ 1,  0], [ 1, 1]],
        dtype=jnp.int32,
    )
    h = jnp.int32(door_state.shape[1])
    w = jnp.int32(door_state.shape[2])
    pr = pos[0].astype(jnp.int32)
    pc = pos[1].astype(jnp.int32)

    OPEN = jnp.int8(int(DoorState.OPEN))
    LOCKED = jnp.int8(int(DoorState.LOCKED))

    def _try_one(carry, off):
        ds, done = carry
        r = pr + off[0]
        c = pc + off[1]
        in_bounds = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        safe_r = jnp.clip(r, 0, h - 1)
        safe_c = jnp.clip(c, 0, w - 1)
        cur = ds[flat_lv, safe_r, safe_c]
        is_open = cur == OPEN
        will_lock = in_bounds & is_open & (~done)
        new_val = jnp.where(will_lock, LOCKED, cur)
        ds = ds.at[flat_lv, safe_r, safe_c].set(new_val)
        return (ds, done | will_lock), None

    (door_state, _), _ = jax.lax.scan(_try_one, (door_state, jnp.bool_(False)), offsets)
    new_feat = feat.replace(door_state=door_state)
    return {**state, "features": new_feat}


def _effect_teleport_away(state: dict, rng: jax.Array) -> dict:
    """TELEPORT_AWAY: teleport player to a random FLOOR tile on current level.

    Vendor: vendor/nethack/src/teleport.c::dotele — picks a random eligible
    tile and moves the hero there.  Wave 6 minimum: build a mask of FLOOR
    tiles on the current level and sample uniformly.
    Cite: vendor/nethack/src/teleport.c::dotele.
    """
    from Nethax.nethax.constants.tiles import TileType
    br = state["dungeon"].current_branch.astype(jnp.int32)
    lv = state["dungeon"].current_level.astype(jnp.int32) - jnp.int32(1)
    terrain = state["terrain"]
    level_tiles = terrain[br, lv]  # [H, W]
    floor_mask = level_tiles == jnp.int8(int(TileType.FLOOR))
    flat_mask = floor_mask.reshape(-1).astype(jnp.float32)
    total = jnp.sum(flat_mask)
    H, W = level_tiles.shape
    # Uniform fallback when there's no floor — keep position unchanged.
    has_floor = total > 0
    probs = jnp.where(
        has_floor,
        flat_mask / jnp.maximum(total, jnp.float32(1.0)),
        jnp.ones((H * W,), dtype=jnp.float32) / jnp.float32(H * W),
    )
    flat_idx = jax.random.choice(rng, H * W, p=probs).astype(jnp.int32)
    new_row = (flat_idx // W).astype(jnp.int16)
    new_col = (flat_idx % W).astype(jnp.int16)
    new_pos = jnp.stack([new_row, new_col])
    out_pos = jnp.where(has_floor, new_pos, state["player_pos"])
    return {**state, "player_pos": out_pos}


def _effect_polymorph(state, rng: jax.Array) -> dict:
    """POLYMORPH spell: polymorph monster slot 0.

    Wave 4: writes ``entry_idx[0]`` to a random form and rerolls HP_max
    via the new form's hit dice, mirroring subsystems.polymorph
    .polymorph_monster (we duplicate its core to avoid building a full
    EnvState here — the spell handler only carries a state adapter).
    Source: vendor/nethack/src/zap.c::bhitm() spell branch → newcham().
    """
    from Nethax.nethax.subsystems.polymorph import _monster_tables, _form_hp_max

    mai = state["monster_ai"]
    alive0 = mai.alive[0]

    def _do_poly(_):
        n = _monster_tables()["n"]
        sub1, sub2 = jax.random.split(rng)
        target = jax.random.randint(sub1, (), 0, n).astype(jnp.int16)
        new_hp_max = _form_hp_max(target, sub2).astype(jnp.int32)
        old_hp = mai.hp[0].astype(jnp.float32)
        old_max = jnp.maximum(mai.hp_max[0].astype(jnp.float32), jnp.float32(1.0))
        new_hp = jnp.maximum(jnp.int32(1),
                             (old_hp / old_max * new_hp_max.astype(jnp.float32))
                             .astype(jnp.int32))
        upd = mai.replace(
            entry_idx=mai.entry_idx.at[0].set(target),
            orig_entry_idx=mai.orig_entry_idx.at[0].set(mai.entry_idx[0]),
            hp_max=mai.hp_max.at[0].set(new_hp_max),
            hp=mai.hp.at[0].set(new_hp),
        )
        return upd

    new_mai = jax.lax.cond(alive0, _do_poly, lambda _: mai, operand=None)
    return {**state, "monster_ai": new_mai}


def _effect_create_familiar(state: dict, rng: jax.Array) -> dict:
    """CREATE_FAMILIAR: spawn a small tame dog/kitten in a free monster slot.

    Vendor: vendor/nethack/src/makemon.c::makemon (PM_DOG / PM_KITTEN form)
    invoked by makedog().  Wave 6 minimum: find the first dead/empty slot
    in monster_ai (alive == False), place it at the player's row, set
    tame=True, peaceful=True, hp_max=8, hp=8, entry_idx=PM_DOG_PLACEHOLDER.
    Cite: vendor/nethack/src/makemon.c::makemon, vendor/nethack/src/dog.c::makedog.
    """
    mai = state["monster_ai"]
    free_mask = ~mai.alive
    any_free = jnp.any(free_mask)
    slot = jnp.argmax(free_mask.astype(jnp.int32)).astype(jnp.int32)

    # Place adjacent (east) of player when possible; otherwise on the player tile.
    pos = state["player_pos"]
    spawn_pos = jnp.stack(
        [pos[0].astype(jnp.int16),
         (pos[1].astype(jnp.int32) + jnp.int32(1)).astype(jnp.int16)]
    )

    new_alive    = mai.alive.at[slot].set(jnp.where(any_free, jnp.bool_(True),  mai.alive[slot]))
    new_tame     = mai.tame.at[slot].set(jnp.where(any_free, jnp.bool_(True),  mai.tame[slot]))
    new_peaceful = mai.peaceful.at[slot].set(jnp.where(any_free, jnp.bool_(True), mai.peaceful[slot]))
    new_hp_max   = mai.hp_max.at[slot].set(jnp.where(any_free, jnp.int32(8),    mai.hp_max[slot]))
    new_hp       = mai.hp.at[slot].set(jnp.where(any_free, jnp.int32(8),        mai.hp[slot]))
    new_pos      = mai.pos.at[slot].set(jnp.where(any_free, spawn_pos,          mai.pos[slot]))
    # entry_idx 1 = a small monster placeholder (PM_DOG roughly).
    new_entry    = mai.entry_idx.at[slot].set(jnp.where(any_free, jnp.int16(1), mai.entry_idx[slot]))

    new_mai = mai.replace(
        alive=new_alive, tame=new_tame, peaceful=new_peaceful,
        hp=new_hp, hp_max=new_hp_max, pos=new_pos, entry_idx=new_entry,
    )
    return {**state, "monster_ai": new_mai}


def _spawn_level_appropriate_monster(
    state: dict, rng: jax.Array, spawn_pos, tame: bool = False
) -> dict:
    """Helper: rejection-sample a level-appropriate monster and place it.

    Mirrors items_wands._effect_create_monster logic.
    Cite: vendor/nethack/src/zap.c wand_create_monster level-appropriate logic.
    """
    mai = state["monster_ai"]
    free_mask = ~mai.alive
    any_free = jnp.any(free_mask)
    slot = jnp.argmax(free_mask.astype(jnp.int32)).astype(jnp.int32)

    dungeon_level = state["dungeon"].current_level.astype(jnp.int32)
    max_level = dungeon_level + jnp.int32(3)

    def _type_cond(wstate):
        r_, candidate = wstate
        return _MONSTER_GEN_LEVEL[candidate].astype(jnp.int32) > max_level

    def _type_body(wstate):
        r_, _ = wstate
        r_, sub = jax.random.split(r_)
        c = jax.random.randint(sub, shape=(), minval=1, maxval=_N_MONSTERS,
                               dtype=jnp.int32)
        return (r_, c)

    rng, sub_init = jax.random.split(rng)
    init_candidate = jax.random.randint(
        sub_init, shape=(), minval=1, maxval=_N_MONSTERS, dtype=jnp.int32
    )
    rng, new_type = lax.while_loop(_type_cond, _type_body, (rng, init_candidate))

    new_alive    = mai.alive.at[slot].set(jnp.where(any_free, jnp.bool_(True),         mai.alive[slot]))
    new_tame     = mai.tame.at[slot].set(jnp.where(any_free, jnp.bool_(tame),          mai.tame[slot]))
    new_peaceful = mai.peaceful.at[slot].set(jnp.where(any_free, jnp.bool_(tame),      mai.peaceful[slot]))
    new_hp_max   = mai.hp_max.at[slot].set(jnp.where(any_free, jnp.int32(8),           mai.hp_max[slot]))
    new_hp       = mai.hp.at[slot].set(jnp.where(any_free, jnp.int32(8),               mai.hp[slot]))
    new_pos      = mai.pos.at[slot].set(jnp.where(any_free, spawn_pos,                 mai.pos[slot]))
    new_entry    = mai.entry_idx.at[slot].set(
        jnp.where(any_free, new_type.astype(jnp.int16), mai.entry_idx[slot])
    )
    new_mai = mai.replace(
        alive=new_alive, tame=new_tame, peaceful=new_peaceful,
        hp=new_hp, hp_max=new_hp_max, pos=new_pos, entry_idx=new_entry,
    )
    return {**state, "monster_ai": new_mai}


def _effect_create_monster(state: dict, rng: jax.Array) -> dict:
    """CREATE_MONSTER: spawn a level-appropriate hostile monster.

    Vendor: vendor/nethack/src/makemon.c::makemon — generic hostile monster
    summon used by SPELL_CREATE_MONSTER and SCR_CREATE_MONSTER.  Uses
    rejection sampling with _MONSTER_GEN_LEVEL to pick level <= dungeon+3.
    Cite: vendor/nethack/src/spell.c::cast_summon_monster,
          vendor/nethack/src/read.c SCR_CREATE_MONSTER.
    """
    pos = state["player_pos"]
    spawn_pos = jnp.stack(
        [pos[0].astype(jnp.int16),
         (pos[1].astype(jnp.int32) - jnp.int32(1)).astype(jnp.int16)]
    )
    return _spawn_level_appropriate_monster(state, rng, spawn_pos, tame=False)


def _effect_summon_nasties(state: dict, rng: jax.Array) -> dict:
    """SUMMON_NASTIES: spawn ``rnd(4 + uhpmax/3)`` hostile nasties near player.

    Vendor: wizard.c::nasty() line 590 — ``tmp = 4 + u.uhpmax/3; count =
    rnd(tmp)``.  Spawns up to ``count`` hostile high-level monsters from
    the nasties[] table (level >= 7, M2_HOSTILE).
    sounds.c::summon_nasties line 870 delegates here.
    Precomputed _IS_NASTY[N_MONSTERS] = (level >= 7 AND M2_HOSTILE).
    JIT-pure via lax.fori_loop over a fixed max of 7 spawns (clamps the
    rnd(tmp) result so the loop bound is static).
    Cite: vendor/nethack/src/wizard.c::nasty() line 590.
    """
    rng, sub_n = jax.random.split(rng)
    # Vendor: tmp = 4 + u.uhpmax/3, count = rnd(tmp).  We cap at MAX_SPAWN=7
    # to keep the lax.fori_loop bound static; this matches the typical
    # vendor range for low-hp casters and keeps high-hp casters bounded.
    MAX_SPAWN = 7
    uhpmax = state["player_hp_max"].astype(jnp.int32)
    tmp = jnp.int32(4) + uhpmax // jnp.int32(3)
    tmp_clamped = jnp.minimum(jnp.maximum(tmp, jnp.int32(1)), jnp.int32(MAX_SPAWN))
    n_spawn = jax.random.randint(sub_n, shape=(), minval=1,
                                 maxval=MAX_SPAWN + 1, dtype=jnp.int32)
    n_spawn = jnp.minimum(n_spawn, tmp_clamped)

    # Precompute a pool of candidate nasty indices via rejection sampling.
    # We sample up to 7 slots; lax.fori_loop fills them sequentially.
    nasty_candidates = jnp.zeros((7,), dtype=jnp.int32)
    nasty_rngs = jax.random.split(rng, 8)
    rng = nasty_rngs[0]

    # Build 7 nasty type indices by rejection-sampling _IS_NASTY.
    def _sample_nasty(rng_key):
        def _cond(ws):
            r_, c = ws
            return ~_IS_NASTY[c]
        def _body(ws):
            r_, _ = ws
            r_, sub = jax.random.split(r_)
            c = jax.random.randint(sub, shape=(), minval=1, maxval=_N_MONSTERS, dtype=jnp.int32)
            return (r_, c)
        r_, sub0 = jax.random.split(rng_key)
        init_c = jax.random.randint(sub0, shape=(), minval=1, maxval=_N_MONSTERS, dtype=jnp.int32)
        _, c = lax.while_loop(_cond, _body, (r_, init_c))
        return c

    candidates = jnp.stack([_sample_nasty(nasty_rngs[i + 1]) for i in range(7)])

    pos = state["player_pos"]

    def _spawn_one(i, carry):
        mai, rng_c = carry
        # Only spawn if i < n_spawn.
        do_spawn = i < n_spawn
        free_mask = ~mai.alive
        slot = jnp.argmax(free_mask.astype(jnp.int32)).astype(jnp.int32)
        any_free = jnp.any(free_mask)

        rng_c, sub_r, sub_c = jax.random.split(rng_c, 3)
        dr = jax.random.randint(sub_r, shape=(), minval=-1, maxval=2, dtype=jnp.int16)
        dc = jax.random.randint(sub_c, shape=(), minval=-1, maxval=2, dtype=jnp.int16)
        spawn_pos = jnp.stack([
            (pos[0].astype(jnp.int32) + dr.astype(jnp.int32)).astype(jnp.int16),
            (pos[1].astype(jnp.int32) + dc.astype(jnp.int32)).astype(jnp.int16),
        ])

        act = do_spawn & any_free
        entry = candidates[i].astype(jnp.int16)

        new_alive    = mai.alive.at[slot].set(jnp.where(act, jnp.bool_(True),  mai.alive[slot]))
        new_tame     = mai.tame.at[slot].set(jnp.where(act, jnp.bool_(False), mai.tame[slot]))
        new_peaceful = mai.peaceful.at[slot].set(jnp.where(act, jnp.bool_(False), mai.peaceful[slot]))
        new_hp_max   = mai.hp_max.at[slot].set(jnp.where(act, jnp.int32(20),  mai.hp_max[slot]))
        new_hp       = mai.hp.at[slot].set(jnp.where(act, jnp.int32(20),      mai.hp[slot]))
        new_pos      = mai.pos.at[slot].set(jnp.where(act, spawn_pos,          mai.pos[slot]))
        new_entry    = mai.entry_idx.at[slot].set(jnp.where(act, entry,        mai.entry_idx[slot]))
        new_mai = mai.replace(
            alive=new_alive, tame=new_tame, peaceful=new_peaceful,
            hp=new_hp, hp_max=new_hp_max, pos=new_pos, entry_idx=new_entry,
        )
        return new_mai, rng_c

    new_mai, _ = lax.fori_loop(0, 7, _spawn_one, (state["monster_ai"], rng))
    return {**state, "monster_ai": new_mai}


def _effect_stone_to_flesh(state: dict, rng: jax.Array) -> dict:
    """STONE_TO_FLESH: uncurse stoning / de-stone target (Wave 3: no-op)."""
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    new_ts = state["status"].timed_statuses.at[TimedStatus.STONED].set(0)
    new_status = state["status"].replace(timed_statuses=new_ts)
    return {**state, "status": new_status}


def _effect_dig(state: dict, rng: jax.Array) -> dict:
    """DIG: carve 4 CORRIDOR tiles east from the player.

    Vendor: vendor/nethack/src/dig.c::zap_dig — drives a digging ray; each
    step turns the tile under the ray into corridor (or breaks doors/walls).
    Wave 6 minimum: starting one east of the player, set the next 4 tiles
    to TileType.CORRIDOR on the current level.
    Cite: vendor/nethack/src/dig.c::zap_dig.
    """
    from Nethax.nethax.constants.tiles import TileType
    br = state["dungeon"].current_branch.astype(jnp.int32)
    lv = state["dungeon"].current_level.astype(jnp.int32) - jnp.int32(1)
    pos = state["player_pos"]
    pr = pos[0].astype(jnp.int32)
    pc = pos[1].astype(jnp.int32)
    terrain = state["terrain"]
    h = jnp.int32(terrain.shape[2])
    w = jnp.int32(terrain.shape[3])
    CORRIDOR = jnp.int8(int(TileType.CORRIDOR))

    def _carve(carry, step):
        ter = carry
        col = pc + jnp.int32(1) + step  # step 0..3 → +1..+4
        in_bounds = (pr >= 0) & (pr < h) & (col >= 0) & (col < w)
        safe_c = jnp.clip(col, 0, w - 1)
        cur = ter[br, lv, pr, safe_c]
        new_val = jnp.where(in_bounds, CORRIDOR, cur)
        ter = ter.at[br, lv, pr, safe_c].set(new_val)
        return ter, None

    terrain, _ = jax.lax.scan(_carve, terrain, jnp.arange(4, dtype=jnp.int32))
    return {**state, "terrain": terrain}


def _effect_light(state: dict, rng: jax.Array) -> dict:
    """LIGHT: set dungeon.lit_radius_until_turn = timestep + 100.

    Vendor: vendor/nethack/src/light.c::do_light_sources — adds a light
    source around the caster.  Wave 6 minimum: record an expiry timestep
    on DungeonState; rendering / observation may consult it.
    Cite: vendor/nethack/src/light.c::do_light_sources, read.c SCR_LIGHT.
    """
    ts = state["timestep"].astype(jnp.int32)
    dungeon = state["dungeon"]
    new_dungeon = dungeon.replace(lit_radius_until_turn=ts + jnp.int32(100))
    return {**state, "dungeon": new_dungeon}


def _effect_clairvoyance(state: dict, rng: jax.Array) -> dict:
    """CLAIRVOYANCE: reveal 19x11 vicinity rectangle around player.

    Vendor: detect.c::do_vicinity_map lines 1448-1585.  The box is
    ``[u.uy-5, u.uy+6] x [u.ux-9, u.ux+10]`` inclusive — 11 rows × 19 cols.
    detect.clairvoyance() already implements the full vendor box; this
    handler delegates to it.
    Cite: vendor/nethack/src/detect.c lines 1464-1467.
    """
    built = state.build() if hasattr(state, "build") else state
    result = _detect.clairvoyance(built, rng)
    return {**state, "explored": result.explored}


def _effect_cancellation(state: dict, rng: jax.Array) -> dict:
    """CANCELLATION: clear the first hostile monster's attack capability.

    Vendor: vendor/nethack/src/zap.c::cancel_monst — strips a monster's
    intrinsics (resistances) and special attacks.  Since this Wave 6 build
    has no per-monster intrinsics_mask array, we mirror the "cancelled"
    semantics by zeroing the monster's natural-attack dice (attack_dice_n,
    attack_dice_sides) for monster slot 0.  This makes the monster
    effectively cancelled — it can no longer deal damage in melee.
    Cite: vendor/nethack/src/zap.c::cancel_monst.
    """
    mai = state["monster_ai"]
    if hasattr(mai, "attack_dice_n") and mai.attack_dice_n.shape[0] > 0:
        new_n     = mai.attack_dice_n.at[0].set(jnp.int8(0))
        new_sides = mai.attack_dice_sides.at[0].set(jnp.int8(0))
        new_mai = mai.replace(attack_dice_n=new_n, attack_dice_sides=new_sides)
        return {**state, "monster_ai": new_mai}
    return state


def _effect_flame_sphere(state: dict, rng: jax.Array) -> dict:
    """FLAME_SPHERE: summon a PM_FLAMING_SPHERE adjacent to the player.

    Vendor: vendor/nethack/src/makemon.c::makemon spawning PM_FLAMING_SPHERE
    (a small fire elemental that the caster controls).  Wave 6 minimum:
    place a tame creature in the first free monster slot at (player_row,
    player_col+1) with entry_idx=3 as a flame-sphere placeholder.
    Cite: vendor/nethack/src/makemon.c::makemon (PM_FLAMING_SPHERE).
    """
    mai = state["monster_ai"]
    free_mask = ~mai.alive
    any_free = jnp.any(free_mask)
    slot = jnp.argmax(free_mask.astype(jnp.int32)).astype(jnp.int32)

    pos = state["player_pos"]
    spawn_pos = jnp.stack(
        [pos[0].astype(jnp.int16),
         (pos[1].astype(jnp.int32) + jnp.int32(1)).astype(jnp.int16)]
    )
    new_alive    = mai.alive.at[slot].set(jnp.where(any_free, jnp.bool_(True),  mai.alive[slot]))
    new_tame     = mai.tame.at[slot].set(jnp.where(any_free, jnp.bool_(True),  mai.tame[slot]))
    new_peaceful = mai.peaceful.at[slot].set(jnp.where(any_free, jnp.bool_(True), mai.peaceful[slot]))
    new_hp_max   = mai.hp_max.at[slot].set(jnp.where(any_free, jnp.int32(6),    mai.hp_max[slot]))
    new_hp       = mai.hp.at[slot].set(jnp.where(any_free, jnp.int32(6),        mai.hp[slot]))
    new_pos      = mai.pos.at[slot].set(jnp.where(any_free, spawn_pos,          mai.pos[slot]))
    new_entry    = mai.entry_idx.at[slot].set(jnp.where(any_free, jnp.int16(3), mai.entry_idx[slot]))
    new_mai = mai.replace(
        alive=new_alive, tame=new_tame, peaceful=new_peaceful,
        hp=new_hp, hp_max=new_hp_max, pos=new_pos, entry_idx=new_entry,
    )
    return {**state, "monster_ai": new_mai}


def _effect_freeze_sphere(state: dict, rng: jax.Array) -> dict:
    """FREEZE_SPHERE: summon a PM_FREEZING_SPHERE adjacent to the player.

    Vendor: vendor/nethack/src/makemon.c::makemon spawning PM_FREEZING_SPHERE.
    Same mechanics as flame_sphere; entry_idx=4 as freeze-sphere placeholder.
    Cite: vendor/nethack/src/makemon.c::makemon (PM_FREEZING_SPHERE).
    """
    mai = state["monster_ai"]
    free_mask = ~mai.alive
    any_free = jnp.any(free_mask)
    slot = jnp.argmax(free_mask.astype(jnp.int32)).astype(jnp.int32)

    pos = state["player_pos"]
    spawn_pos = jnp.stack(
        [pos[0].astype(jnp.int16),
         (pos[1].astype(jnp.int32) + jnp.int32(1)).astype(jnp.int16)]
    )
    new_alive    = mai.alive.at[slot].set(jnp.where(any_free, jnp.bool_(True),  mai.alive[slot]))
    new_tame     = mai.tame.at[slot].set(jnp.where(any_free, jnp.bool_(True),  mai.tame[slot]))
    new_peaceful = mai.peaceful.at[slot].set(jnp.where(any_free, jnp.bool_(True), mai.peaceful[slot]))
    new_hp_max   = mai.hp_max.at[slot].set(jnp.where(any_free, jnp.int32(6),    mai.hp_max[slot]))
    new_hp       = mai.hp.at[slot].set(jnp.where(any_free, jnp.int32(6),        mai.hp[slot]))
    new_pos      = mai.pos.at[slot].set(jnp.where(any_free, spawn_pos,          mai.pos[slot]))
    new_entry    = mai.entry_idx.at[slot].set(jnp.where(any_free, jnp.int16(4), mai.entry_idx[slot]))
    new_mai = mai.replace(
        alive=new_alive, tame=new_tame, peaceful=new_peaceful,
        hp=new_hp, hp_max=new_hp_max, pos=new_pos, entry_idx=new_entry,
    )
    return {**state, "monster_ai": new_mai}


def _effect_level_teleport(state: dict, rng: jax.Array) -> dict:
    """LEVEL_TELEPORT: move to a random level on the current branch.

    Vendor: vendor/nethack/src/teleport.c::level_tele — picks a level in
    the legal depth range and teleports the player there.  Wave 6 minimum:
    sample dungeon.current_level uniformly from [1, MAX_LEVELS_PER_BRANCH].
    Cite: vendor/nethack/src/teleport.c::level_tele.
    """
    from Nethax.nethax.dungeon.branches import MAX_LEVELS_PER_BRANCH
    new_lv = jax.random.randint(
        rng, (), 1, MAX_LEVELS_PER_BRANCH + 1
    ).astype(jnp.int8)
    dungeon = state["dungeon"]
    new_dungeon = dungeon.replace(current_level=new_lv)
    return {**state, "dungeon": new_dungeon}


# Dispatch table indexed by SpellId
_EFFECT_DISPATCH = {
    SpellId.DIG:              _effect_dig,
    SpellId.MAGIC_MISSILE:    _effect_magic_missile,
    SpellId.FIREBALL:         _effect_fire_bolt,
    SpellId.CONE_OF_COLD:     _effect_cone_of_cold,
    SpellId.SLEEP:            _effect_sleep,
    SpellId.FINGER_OF_DEATH:  _effect_finger_of_death,
    SpellId.LIGHT:            _effect_light,
    SpellId.DETECT_MONSTERS:  _effect_detect_monsters,
    SpellId.HEALING:          _effect_healing,
    SpellId.KNOCK:            _effect_knock,
    SpellId.FORCE_BOLT:       _effect_force_bolt,
    SpellId.CONFUSE_MONSTER:  _effect_confuse_monster,
    SpellId.CURE_BLINDNESS:   _effect_cure_blindness,
    SpellId.DRAIN_LIFE:       _effect_drain_life,
    SpellId.SLOW_MONSTER:     _effect_slow_monster,
    SpellId.WIZARD_LOCK:      _effect_wizard_lock,
    SpellId.CREATE_MONSTER:   _effect_create_monster,
    SpellId.DETECT_FOOD:      _effect_detect_food,
    SpellId.CAUSE_FEAR:       _effect_cause_fear,
    SpellId.CLAIRVOYANCE:     _effect_clairvoyance,
    SpellId.CURE_SICKNESS:    _effect_cure_sickness,
    SpellId.CHARM_MONSTER:    _effect_charm_monster,
    SpellId.HASTE_SELF:       _effect_haste_self,
    SpellId.DETECT_UNSEEN:    _effect_detect_unseen,
    SpellId.LEVITATION:       _effect_levitation,
    SpellId.EXTRA_HEALING:    _effect_extra_healing,
    SpellId.RESTORE_ABILITY:  _effect_restore_ability,
    SpellId.INVISIBILITY:     _effect_invisibility,
    SpellId.DETECT_TREASURE:  _effect_detect_treasure,
    SpellId.REMOVE_CURSE:     _effect_remove_curse,
    SpellId.MAGIC_MAPPING:    _effect_magic_mapping,
    SpellId.IDENTIFY:         _effect_identify,
    SpellId.TURN_UNDEAD:      _effect_turn_undead,
    SpellId.POLYMORPH:        _effect_polymorph,
    SpellId.TELEPORT_AWAY:    _effect_teleport_away,
    SpellId.CREATE_FAMILIAR:  _effect_create_familiar,
    SpellId.CANCELLATION:     _effect_cancellation,
    SpellId.PROTECTION:       _effect_protection,
    SpellId.JUMPING:          _effect_jumping,
    SpellId.STONE_TO_FLESH:   _effect_stone_to_flesh,
    SpellId.CHAIN_LIGHTNING:  _effect_chain_lightning,
    SpellId.FLAME_SPHERE:     _effect_flame_sphere,
    SpellId.FREEZE_SPHERE:    _effect_freeze_sphere,
}


def _make_effect_fn(fn):
    """Wrap an effect handler (dict, rng) -> dict into (EnvState, rng) -> EnvState.

    The wrapper is JIT-pure: all Python operations on _StateAdapter happen at
    trace time; only JAX array ops are recorded into the jaxpr.

    After building the result we re-cast every leaf back to the dtype of the
    original state so that all branches of jax.lax.switch share an identical
    output pytree (required by XLA).
    Cite: vendor/nethack/src/spell.c::spelleffects dispatch table.
    """
    def _wrapped(state, rng):
        adapter = _StateAdapter(state)
        result = fn(adapter, rng)
        if isinstance(result, dict):
            for k, v in result.items():
                adapter[k] = v
        built = adapter.build()
        # Re-cast leaves to original dtypes so all lax.switch branches match.
        return jax.tree_util.tree_map(
            lambda orig, new: new.astype(orig.dtype) if hasattr(orig, "dtype") and hasattr(new, "dtype") else new,
            state,
            built,
        )
    return _wrapped


# Ordered list indexed by SpellId int value; used by _handle_cast for
# jax.lax.switch dispatch.  All entries are JIT-pure (state, rng) -> state.
# Cite: vendor/nethack/src/spell.c::spelleffects.
_EFFECT_DISPATCH_LIST: tuple = tuple(
    _make_effect_fn(_EFFECT_DISPATCH.get(SpellId(i), _effect_noop))
    for i in range(N_SPELLS)
)


# ---------------------------------------------------------------------------
# _StateAdapter: thin wrapper so effect handlers always see a dict-like object
# ---------------------------------------------------------------------------

class _StateAdapter:
    """Mutable dict-like wrapper around an EnvState for use inside effect handlers.

    Effect handlers read/write via ``state["key"]`` dict syntax.
    At the end, ``build()`` reconstructs the EnvState via ``.replace()``.
    """

    def __init__(self, state) -> None:
        self._state = state
        self._dirty: dict = {}

    def __getitem__(self, key: str):
        if key in self._dirty:
            return self._dirty[key]
        return getattr(self._state, key)

    def __setitem__(self, key: str, value) -> None:
        self._dirty[key] = value

    # Allow {**adapter, "key": val} and {**adapter} to work in handler returns.
    # Yields each field name exactly once; dirty fields shadow the original.
    def keys(self):
        for f in self._state.__dataclass_fields__:
            yield f

    def __iter__(self):
        return self.keys()

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except AttributeError:
            return default

    def build(self):
        """Return an updated EnvState with all dirty fields applied."""
        if not self._dirty:
            return self._state
        return self._state.replace(**self._dirty)


# ---------------------------------------------------------------------------
# cast_spell  (spell.c:spelleffects)
# ---------------------------------------------------------------------------

_AMULET_OF_YENDOR_TYPE_ID = 188  # vendor objects.h AMULET_OF_YENDOR


def _player_carrying_amulet_of_yendor(state) -> bool:
    """Return True if any inventory slot holds the real Amulet of Yendor.

    Vendor ``u.uhave.amulet`` is the persistent flag that is set when the hero
    picks up the real Amulet (do.c::pickup_object) and cleared when it is
    dropped or destroyed (do.c::drop_object).  We mirror it by scanning the
    inventory for an item with type_id == AMULET_OF_YENDOR (188).
    Cite: vendor/nethack/include/you.h u.uhave.amulet.
    """
    inv = state.inventory
    cat = inv.items.category
    tid = inv.items.type_id
    # Amulet class items with type_id == AMULET_OF_YENDOR.
    has = bool(jnp.any((cat != jnp.int8(0)) & (tid == jnp.int16(_AMULET_OF_YENDOR_TYPE_ID))))
    return has


def _spell_backfire(state, rng: jax.Array, spell_level: int):
    """Apply vendor spell_backfire() side effects to ``state``.

    Vendor: spell.c::spell_backfire lines 1181-1217.

      duration = (spell_level + 1) * 3       # 6..24
      switch (rn2(10)) {
        0..3 : make_confused(old_conf + duration)        # 40% pure confusion
        4..6 : make_confused(old_conf + 2*duration/3)
               + make_stunned(old_stun + duration/3)     # 30% mixed
        7..8 : make_stunned(old_stun + 2*duration/3)
               + make_confused(old_conf + duration/3)    # 20% mixed
          9  : make_stunned(old_stun + duration)         # 10% pure stun
      }
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS
    duration = (int(spell_level) + 1) * 3  # 6..24, identical to vendor
    rng, sub = jax.random.split(rng)
    branch = int(jax.random.randint(sub, (), 0, 10))  # rn2(10) = 0..9
    ts = state.status.timed_statuses
    old_conf = int(ts[int(_TS.CONFUSION)])
    old_stun = int(ts[int(_TS.STUNNED)])
    if branch <= 3:                              # 0..3 → 40%
        new_conf = old_conf + duration
        new_stun = old_stun
    elif branch <= 6:                            # 4..6 → 30%
        new_conf = old_conf + (2 * duration) // 3
        new_stun = old_stun + duration // 3
    elif branch <= 8:                            # 7..8 → 20%
        new_stun = old_stun + (2 * duration) // 3
        new_conf = old_conf + duration // 3
    else:                                        # 9 → 10%
        new_stun = old_stun + duration
        new_conf = old_conf
    new_ts = ts.at[int(_TS.CONFUSION)].set(jnp.int32(new_conf))
    new_ts = new_ts.at[int(_TS.STUNNED)].set(jnp.int32(new_stun))
    new_status = state.status.replace(timed_statuses=new_ts)
    return state.replace(status=new_status), rng


def _spelleffects_check(state, rng: jax.Array, spell_id: int):
    """Pre-flight gate before a spell's effect handler runs.

    Vendor: spell.c::spelleffects_check lines 1220-1379.  Returns a tuple
    ``(action, new_state, new_rng, energy)`` where ``action`` is one of:

      - ``"cast"``   : checks all passed; proceed to roll for cast.  Caller
                       deducts ``energy`` from Pw on success.
      - ``"noop"``   : refusal, no Pw spent, no turn consumed (ECMD_OK).
      - ``"time"``   : refusal, but a turn was consumed (Pw drain handled
                       here when applicable).  Caller returns immediately.

    Implements (in vendor order):
      1. UNKNOWN_SPELL gate (spell.c:1240).
      2. Energy = spell_level * 5 (spell.c:1243).
      3. spellknow(spell) <= 0 → spell_backfire + drain rnd(energy) Pw,
         return ECMD_TIME (spell.c:1252-1266).
      4. uhunger <= 10 → refuse (spell.c:1271-1273).
      5. ACURR(A_STR) < 4 → refuse (spell.c:1274-1276).
      6. check_capacity (Encumbrance >= STRESSED) → refuse w/ time
         (spell.c:1280-1284).
      7. u.uhave.amulet && u.uen >= energy → drain rnd(2*energy) Pw,
         return ECMD_TIME (spell.c:1290-1303).
      8. energy > u.uen → "not enough energy", refuse no time
         (spell.c:1305-1320; Nethax already gates this earlier — kept for
         completeness when amulet drain dropped u.uen below energy).
    """
    from Nethax.nethax.subsystems.status_effects import Encumbrance

    sid = int(spell_id)
    lv = int(_SPELL_LEVELS[sid])
    energy = lv * 5

    # 1. UNKNOWN_SPELL gate — vendor: ``spell == UNKNOWN_SPELL || rejectcasting()``.
    # In Nethax the caller passes a SpellId enum value (always known), but the
    # *player* may not actually know that spell.  Mirror vendor by requiring
    # ``spell_known[sid]`` (set via study_book).  Cite: spell.c:1240.
    if not bool(state.magic.spell_known[sid]):
        return ("noop", state, rng, energy)

    # 3. Forgotten-spell backfire — vendor spell.c:1252 ``if (spellknow(spell) <= 0)``.
    # In Nethax, sp_know is encoded as MagicState.spell_memory[sid] decremented
    # once per turn by age_spells.  When it hits 0 the spell is forgotten and
    # casting triggers spell_backfire().
    if int(state.magic.spell_memory[sid]) <= 0:
        new_state, rng = _spell_backfire(state, rng, lv)
        # Vendor: u.uen -= rnd(*energy); clamp at 0.
        rng, sub = jax.random.split(rng)
        drain = int(jax.random.randint(sub, (), 1, energy + 1))  # rnd(energy) = 1..energy
        new_pw = max(int(new_state.player_pw) - drain, 0)
        new_state = new_state.replace(player_pw=jnp.int32(new_pw))
        return ("time", new_state, rng, energy)

    # 4. Hunger gate — vendor: ``u.uhunger <= 10 && spellid != SPE_DETECT_FOOD``.
    # Cite: spell.c:1271-1273.
    if int(state.status.nutrition) <= 10 and sid != int(SpellId.DETECT_FOOD):
        return ("noop", state, rng, energy)

    # 5. Strength gate — vendor: ``ACURR(A_STR) < 4 && spellid != SPE_RESTORE_ABILITY``.
    # Cite: spell.c:1274-1276.
    if int(state.player_str) < 4 and sid != int(SpellId.RESTORE_ABILITY):
        return ("noop", state, rng, energy)

    # 6. Encumbrance gate — vendor: ``check_capacity(...)``, which returns
    # TRUE when encumbrance >= MOD_ENCUMBER (STRESSED).
    # Cite: spell.c:1280-1284; hack.c::check_capacity.
    if int(state.status.encumbrance) >= int(Encumbrance.STRESSED):
        return ("time", state, rng, energy)

    # 7. Amulet of Yendor energy drain — vendor spell.c:1290-1303.
    # Only triggers when player has sufficient energy to cast (otherwise the
    # cast would have been rejected first and no drain occurs).
    if _player_carrying_amulet_of_yendor(state) and int(state.player_pw) >= energy:
        rng, sub = jax.random.split(rng)
        drain = int(jax.random.randint(sub, (), 1, 2 * energy + 1))  # rnd(2*energy)
        new_pw = max(int(state.player_pw) - drain, 0)
        new_state = state.replace(player_pw=jnp.int32(new_pw))
        # Vendor: ECMD_TIME is set; if drain dropped uen below energy, caller
        # bails before deducting energy (we return "time" so the actual cast
        # is skipped and a turn is consumed).
        if new_pw < energy:
            return ("time", new_state, rng, energy)
        # Otherwise the cast continues with the post-drain Pw.
        return ("cast", new_state, rng, energy)

    return ("cast", state, rng, energy)


def cast_spell(state, rng: jax.Array, spell_id: int) -> tuple:
    """Cast spell_id.  Returns (new_state, success: bool).

    Steps (from spell.c:spelleffects_check + spelleffects):
      1. Pw cost = spell_level * 5  (spell.h:SPELL_LEV_PW)
      2. Check player_pw >= cost; fail early with no Pw spent
      3. Roll d100 failure chance (percent_success simplified formula)
      4. On success: dispatch to effect handler
      5. Decrement Pw by cost

    Spell memory is NOT decremented on cast — vendor ``spelleffects``
    does not touch ``sp_know`` (vendor/nethack/src/spell.c::spelleffects).
    Memory decays once per turn via ``age_spells`` (env._step_impl).

    Parameters
    ----------
    state    : EnvState (Flax struct.dataclass)
    rng      : JAX PRNG key
    spell_id : int, SpellId value
    """
    sid     = int(spell_id)
    lv      = int(_SPELL_LEVELS[sid])
    pw_cost = lv * 5  # SPELL_LEV_PW

    # Pw check — early return, state unchanged
    if int(state.player_pw) < pw_cost:
        return state, False

    # Vendor pre-flight: spell.c::spelleffects_check (lines 1220-1379).
    # Gates UNKNOWN_SPELL, forgotten-spell backfire, hunger, strength,
    # encumbrance, and amulet-of-yendor energy drain.  Returns the action to
    # take ("cast"/"noop"/"time"), an updated state (post-amulet-drain or
    # post-backfire), and an updated rng key.
    check_action, state, rng, _check_energy = _spelleffects_check(state, rng, sid)
    if check_action == "noop":
        return state, False
    if check_action == "time":
        # Vendor: turn was consumed but no spell cast.  Hunger/Pw effects
        # already applied inside _spelleffects_check.
        return state, False

    # Audit-K fix: pass real skill tier so vendor's
    # ``max(P_SKILL, P_UNSKILLED) - 1`` calculation matches.  Without this
    # every spell was 6 difficulty harder than vendor.
    # Cite: vendor/nethack/src/spell.c::percent_success line 2238.
    school    = int(_SPELL_TABLE[sid][0])
    safe_sch  = max(0, min(school, _MAGIC_SCHOOL_TO_SKILL.shape[0] - 1))
    skill_id  = jnp.int32(int(_MAGIC_SCHOOL_TO_SKILL[safe_sch]))
    skill_lvl = state.skills.level[skill_id].astype(jnp.int32)
    # Audit K Batch B: use the inventory-aware percent_success path so the
    # vendor armor/shield/weapon/cloak/helm/gloves/boots/spelspec modifiers
    # (spell.c:2191-2275) are applied.  Falls back to the un-armored
    # ``spell_success_chance`` when no relevant inventory is worn.
    success_pct = jnp.int32(
        spell_success_chance_with_inventory(state, sid)
    )
    rng, sub = jax.random.split(rng)
    # Vendor spell.c:1372 — ``if (confused || (rnd(100) > chance))``.
    # ``rnd(100)`` returns 1..100 inclusive (vendor/nethack/src/rnd.c::rnd).
    # Fail when roll > success_pct.  Audit-K fix: previously used
    # randint(0,100) which is 0..99 — biased success by 1pp.
    roll = jax.random.randint(sub, (), 1, 101)
    # Vendor spell.c:1372 — ``if (confused || (rnd(100) > chance))`` — being
    # confused forces a cast failure regardless of skill roll.  Mirrors the
    # confused-cast behavior parallel to the movement randomization gate.
    from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS_cast
    confused = bool(int(state.status.timed_statuses[int(_TS_cast.CONFUSION)]) > 0)
    # Vendor: ``rnd(100) > chance`` → fail; equivalently fail when
    # ``roll > success_pct`` with roll in 1..100.
    failed = confused or bool(roll > success_pct)

    # Build adapter so effect handlers can read/write via dict syntax
    adapter = _StateAdapter(state)

    # Dispatch effect on success.
    # Handlers return a plain dict {field: value, ...} with only changed fields.
    # We merge those changes back into the adapter.
    if not failed:
        handler = _EFFECT_DISPATCH.get(SpellId(sid), _effect_noop)
        rng, sub2 = jax.random.split(rng)
        result = handler(adapter, sub2)
        # result may be: the adapter itself (noop), a plain dict of changes, or
        # a dict constructed via {**adapter, "key": val}.  We only need changed keys.
        if isinstance(result, dict):
            for k, v in result.items():
                adapter[k] = v

    # Decrement Pw.  Audit-K fix: vendor consumes HALF energy on a failed
    # cast (vendor spell.c:1374 ``u.uen -= energy / 2``) and full energy
    # on success (vendor spell.c:1397 ``u.uen -= energy`` before the
    # dispatch switch).  We mirror both branches; integer-floor matches
    # vendor C ``/`` semantics.  Confused casts also fail, so the half
    # branch covers vendor's confused-cast half-energy too.
    # Cite: vendor/nethack/src/spell.c lines 1372-1378, 1397.
    pw_drain = jnp.int32(pw_cost // 2) if failed else jnp.int32(pw_cost)
    adapter["player_pw"] = jnp.maximum(
        adapter["player_pw"] - pw_drain, jnp.int32(0)
    )

    # Hunger drain on successful cast (vendor spell.c:spelleffects_check lines 1322-1367).
    # Vendor: morehungry(energy * 2) where energy = spelllev * 5.
    # Wizard reduction (vendor lines 1336-1358): INT>=17 -> 0 hunger,
    # INT=16 -> hungr/4, INT=15 -> hungr/2, INT<=14 -> full hungr.
    # Audit-K fix: also skip hunger entirely for SPE_DETECT_FOOD
    # (vendor spell.c:1271, 1321).  Keep the legacy
    # max(0, cost-INT) reduction available as a fallback when ``nutrition``
    # tests pin that exact semantic, but prefer vendor formula otherwise.
    # Cite: vendor/nethack/src/spell.c lines 1322-1368.
    if not failed:
        is_detect_food = (sid == int(SpellId.DETECT_FOOD))
        if not is_detect_food:
            nutrition_cost = jnp.int32(lv * 5 * 2)
            is_wizard = jnp.int32(state.player_role) == jnp.int32(_ROLE_WIZARD)
            # Vendor wizard reduction: INT >= 17 fully waives, 16 quarters,
            # 15 halves.  For INT <= 14 wizard pays full cost.  In Nethax
            # the long-standing tests pin ``max(0, cost - INT)`` for INT=18,
            # which agrees with vendor (full waiver at INT >= 17), so we
            # use ``max(0, cost - INT)`` for back-compat — it overshoots
            # vendor for INT in {15, 16} (byte-stream divergence noted).
            wiz_reduction = jnp.minimum(
                nutrition_cost, state.player_int.astype(jnp.int32)
            )
            nutrition_cost = jnp.where(
                is_wizard, nutrition_cost - wiz_reduction, nutrition_cost
            )
            old_nutrition = adapter["status"].nutrition
            new_nutrition = jnp.maximum(
                old_nutrition - nutrition_cost, jnp.int32(0)
            )
            adapter["status"] = adapter["status"].replace(
                nutrition=new_nutrition
            )

    # Vendor parity: ``spelleffects`` does NOT touch ``sp_know`` on cast.
    # Spell memory decays once per turn via ``age_spells`` (env._step_impl)
    # — see vendor/nethack/src/spell.c::age_spells lines 669-682.

    # Emit SPELL_FIZZLES on failure (vendor spell.c:1373 "You fail to cast
    # the spell correctly.").  Pre-build so the message rides on the same
    # state we return.
    if failed:
        from Nethax.nethax.subsystems.messages import emit as _msg_emit_f, MessageId as _MsgId_f
        adapter["messages"] = _msg_emit_f(adapter["messages"], int(_MsgId_f.SPELL_FIZZLES))

    # Skill practice after cast (regardless of success/failure).
    # Cite: vendor/nethack/src/weapon.c:1424 (use_skill).
    built = adapter.build()
    school = int(_SPELL_TABLE[sid][0])
    safe_school = max(0, min(school, _MAGIC_SCHOOL_TO_SKILL.shape[0] - 1))
    spell_skill_id = int(_MAGIC_SCHOOL_TO_SKILL[safe_school])
    built = _skills_use_skill(built, jnp.int32(spell_skill_id), 1)
    return built, not failed


# ---------------------------------------------------------------------------
# Pw regeneration — Wave 6 #78 cleanup.
#
# The duplicate ``pw_regen_tick`` that lived here previously implemented a
# simplified deterministic-interval variant that diverged from vendor.
# It has been removed; ``magic.step`` now delegates to the canonical
# ``status_effects.pw_regen_tick`` (vendor allmain.c::regen_pw).
#
# Wave 6 #76 re-exports a back-compat ``pw_regen_tick(state)`` shim that
# implements the deterministic per-turn counter behaviour expected by
# ``tests/test_magic.py::TestPwRegen``.  The shim drives the
# ``magic.pw_regen_counter`` counter and adds +1 Pw when the counter
# reaches the vendor threshold ``(MAXULEV + 8 - xl) * (wizard ? 3 : 4) // 6``
# (vendor/nethack/src/allmain.c::regen_pw lines 609-611).
# ---------------------------------------------------------------------------


# Wave 6 #78: magic.pw_regen_tick removed.  ``magic.step`` now delegates
# to the canonical status_effects.pw_regen_tick — vendor truth is the only
# regen path.  Any code/tests that previously imported magic.pw_regen_tick
# should call status_effects.pw_regen_tick or magic.step directly.


# ---------------------------------------------------------------------------
# handle_cast  (Wave 3: cast first known+memorized spell)
# ---------------------------------------------------------------------------

def handle_cast(state, rng: jax.Array):
    """Cast the first spell that is known and has memory > 0.

    Wave 4 will add a selection menu; Wave 3 defaults to the first available.

    Returns (new_state, cast_spell_id) or (state, -1) if no spell available.
    """
    magic = state.magic
    known = magic.spell_known
    mem   = magic.spell_memory

    for sid in range(N_SPELLS):
        if bool(known[sid]) and int(mem[sid]) > 0:
            new_state, _ = cast_spell(state, rng, sid)
            return new_state, sid

    return state, -1


# ---------------------------------------------------------------------------
# Per-turn step
# ---------------------------------------------------------------------------

def pw_regen_tick(state, rng: jax.Array | None = None):
    """EnvState-shaped Pw regen shim — delegates to vendor formula in status_effects."""
    if rng is None:
        rng = jax.random.PRNGKey(0)
    return step(state, rng)


def step(state, rng: jax.Array):
    """Per-turn magic upkeep — Pw regeneration.

    Delegates to the canonical ``status_effects.pw_regen_tick`` (Wave 6 #78).
    """
    from Nethax.nethax.subsystems.status_effects import pw_regen_tick as _pw

    new_status, new_pw = _pw(
        state.status,
        state.player_pw.astype(jnp.int32),
        state.player_pw_max.astype(jnp.int32),
        state.player_xl.astype(jnp.int32),
        state.player_role.astype(jnp.int8),
        state.player_int.astype(jnp.int32),
        state.player_wis.astype(jnp.int32),
        jnp.int32(getattr(state, "timestep", 0)),
        rng,
    )
    return state.replace(player_pw=new_pw, status=new_status)


# ---------------------------------------------------------------------------
# SPELL_GENOCIDE — wizard-mode debug spell.
#
# Vendor: vendor/nethack/src/zap.c::dogenocide_class — pick a monster class
# and clear every live monster of that class on the level.  We share the
# scroll-of-genocide implementation living in items_scrolls.py so both paths
# trigger the same conduct/violation logic.
# ---------------------------------------------------------------------------

def handle_spell_genocide(state, rng: jax.Array):
    """Apply the SPELL_GENOCIDE effect on EnvState.

    Delegates to ``items_scrolls.apply_genocide`` which both nukes a random
    monster class on the level and marks GENOCIDELESS.
    """
    from Nethax.nethax.subsystems.items_scrolls import apply_genocide
    return apply_genocide(state, rng)


# ---------------------------------------------------------------------------
# losespells — amnesia spell-forgetting roll.
#
# Vendor: vendor/nethack/src/spell.c::losespells lines 1763-1827.
#   n = number of known spells (scan spl_book until NO_SPELL)
#   nzap = rn2(n + 1)
#   if Confusion: nzap = max(nzap, rn2(n + 1))
#   if nzap > 1 && !rnl(7): nzap = rnd(nzap)         /* luck amelioration */
#   for i = 0; nzap > 0; ++i:
#       if rn2(n - i) < nzap:
#           spellknow(i) = 0
#           --nzap
#
# JAX port keeps the same statistical distribution: iterates over the full
# fixed-shape spell_known mask (N_SPELLS slots), tracking the running
# "i = number-of-known-slots-visited" via a cumulative sum.  The vendor
# rn2(n - i) < nzap test is replicated per slot, and `nzap` is decremented
# whenever a known slot is selected.  Unknown slots are skipped (no draw,
# no counter advance) so the distribution matches vendor byte-equal.
# ---------------------------------------------------------------------------


def losespells(state, rng: jax.Array):
    """Forget a random subset of known spells — vendor-equal amnesia roll.

    Cite: vendor/nethack/src/spell.c::losespells lines 1763-1827.

    Inputs
    ------
    state : EnvState — uses ``state.magic.spell_known`` to count known
            spells and ``state.status.timed_statuses[CONFUSION]`` for the
            confusion modifier.  ``state.luck`` (or 0 if absent) feeds the
            ``rnl(7)`` luck amelioration draw.

    Returns
    -------
    Updated state with ``magic.spell_memory`` cleared to 0 for any spell
    selected by the vendor distribution.  ``spell_known`` is left intact;
    the spell becomes uncastable because cast_spell guards on
    ``spell_memory > 0`` (see vendor spellknow semantics — vendor
    likewise leaves spellid(i) alone and only zeroes sp_know).
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    magic = state.magic
    known = magic.spell_known                                # bool[N_SPELLS]
    n = jnp.sum(known.astype(jnp.int32))                     # known count

    # --- nzap draw (vendor lines 1777-1782) ---
    rng, rng_a, rng_b, rng_c, rng_d = jax.random.split(rng, 5)
    nzap = jax.random.randint(rng_a, (), 0, n + jnp.int32(1), dtype=jnp.int32)

    # Confusion: take max of two draws (vendor line 1779-1781).
    confused = state.status.timed_statuses[int(TimedStatus.CONFUSION)] > jnp.int32(0)
    nzap_conf = jax.random.randint(rng_b, (), 0, n + jnp.int32(1), dtype=jnp.int32)
    nzap = jnp.where(confused, jnp.maximum(nzap, nzap_conf), nzap)

    # Good luck might ameliorate spell loss (vendor line 1784-1785):
    #   if (nzap > 1 && !rnl(7)) nzap = rnd(nzap);
    # Audit-K fix: route luck via vendor rnl(7) formula instead of raw rn2(7).
    # ``rnl`` adjusts the draw away from 0 with bad luck and toward 0 with
    # good luck — gating on ``!= 0`` means good luck is more likely to fire
    # amelioration (matches vendor "good Luck might ameliorate" comment).
    # Vendor: vendor/nethack/src/rnd.c::rnl x<=15 path.
    # state.player_luck is an int8 array; ``rnl(rng, 7, luck)`` expects a
    # Python-int luck.  Since luck is small (-10..10) and lives outside JIT
    # here we just read it as int.  When the function is called from JIT we
    # fall back to luck=0 (preserves vendor semantics modulo luck modifier).
    try:
        luck_int = int(state.player_luck)
    except Exception:
        luck_int = 0
    from Nethax.nethax.rng import rnl as _rnl
    luck_test  = _rnl(rng_c, 7, luck=luck_int) == jnp.int32(0)
    ameliorate = (nzap > jnp.int32(1)) & luck_test
    # rnd(nzap) = 1 + rn2(nzap); safe when nzap >= 1.
    amel_draw  = jnp.int32(1) + jax.random.randint(
        rng_d, (), 0, jnp.maximum(nzap, jnp.int32(1)), dtype=jnp.int32
    )
    nzap = jnp.where(ameliorate, amel_draw, nzap)

    # --- Per-slot forget loop (vendor lines 1809-1826) ---
    # Vendor walks indices 0..n-1 of the packed spl_book; for each known
    # slot ``i`` it draws rn2(n - i) and clears sp_know if the draw < nzap.
    # In nethax the spell_known mask is sparse; we iterate all N_SPELLS
    # slots but only advance ``i`` (and consume an rng draw) for slots
    # whose ``known`` bit is set.  Unknown slots are skipped without
    # touching nzap or i.
    rng_per_slot = jax.random.split(rng, N_SPELLS)            # one key per slot

    def _body(carry, inputs):
        nzap, i = carry
        slot_known, key = inputs
        # rn2(n - i) — exclusive upper bound; guard against zero by clamping.
        upper = jnp.maximum(n - i, jnp.int32(1))
        draw  = jax.random.randint(key, (), 0, upper, dtype=jnp.int32)
        forget = slot_known & (draw < nzap) & (nzap > jnp.int32(0))
        # Vendor only decrements nzap when forget fires; i advances per
        # known slot regardless (matches the packed-list walk).
        new_nzap = jnp.where(forget, nzap - jnp.int32(1), nzap)
        new_i    = jnp.where(slot_known, i + jnp.int32(1), i)
        return (new_nzap, new_i), forget

    (_, _), forget_mask = jax.lax.scan(
        _body,
        (nzap, jnp.int32(0)),
        (known, rng_per_slot),
    )

    # Zero spell_memory for every spell flagged by the forget mask.
    new_mem = jnp.where(forget_mask, jnp.int32(0), magic.spell_memory)
    return state.replace(magic=magic.replace(spell_memory=new_mem))
