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
from flax import struct


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

# vendor/nethack/src/spell.c: KEEN = 20000 — maximum freshly-learned memory
MAX_SPELL_MEMORY = 10000

# Vendor parity constant (spell.c line 17: `#define KEEN 20000`).  When a
# spell is freshly learned via study_book(), sp_know is set to KEEN, and
# age_spells() decrements sp_know by 1 every turn (spell.h line 31:
# `#define decrnknow(spell) svs.spl_book[spell].sp_know--`).
SPELL_KEEN = 20000
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
) -> jnp.ndarray:
    """Return failure percentage (0–100) for casting spell_id.

    Simplified from spell.c:percent_success — omits armor/shield penalties,
    uses P_UNSKILLED skill tier, and ignores role-specific spelspec bonus.

    Parameters
    ----------
    role     : int8, Role enum value (0..12)
    spell_id : int32, SpellId value
    xl       : int32, experience level
    stat_int : int8, player INT stat
    stat_wis : int8, player WIS stat
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

    # chance from stat (spell.c line 2230)
    chance = 11 * statused // 2

    # difficulty: P_UNSKILLED → skill=0, skill-1 = -1
    skill      = jnp.int32(0)  # P_UNSKILLED assumed; Wave 6 adds skill tracking
    skill_adj  = jnp.maximum(skill, 0) - 1  # unskilled → -1 => 0-1 = -1
    difficulty = (spell_lv - 1) * 4 - (skill_adj * 6 + xl // 3 + 1)

    # chance adjustment for difficulty
    # approx isqrt(900*d+2000) with integer math: clip to avoid negative sqrt arg
    sqrt_arg = jnp.maximum(900 * difficulty + 2000, 0)
    # JAX integer sqrt via float cast
    sqrt_val  = jnp.sqrt(sqrt_arg.astype(jnp.float32)).astype(jnp.int32)
    learning  = jnp.minimum(15 * jnp.maximum(-difficulty, 0) // jnp.maximum(spell_lv, 1), 20)
    chance    = jnp.where(difficulty > 0, chance - sqrt_val, chance + learning)

    # clamp before shield penalty (no shield modeled in Wave 3)
    chance = jnp.clip(chance, 0, 120)

    # combine: chance * (20 - splcaster) / 15 - splcaster  (spell.c line 2283)
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
) -> jnp.ndarray:
    """Return SUCCESS percentage (0–100) for casting spell_id.

    Vendor parity wrapper that returns the value spell.c::percent_success()
    actually returns (chance-of-cast).  This is the canonical name for
    new parity tests; ``spell_fail_chance`` is retained for back-compat
    and returns ``100 - spell_success_chance``.

    Source: vendor/nethack/src/spell.c::percent_success() lines 2173-2292.
    """
    return jnp.int32(100) - spell_fail_chance(
        role, spell_id, xl, stat_int, stat_wis
    )


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

    Vendor: zap.c::zapyourself line 2911 → ``healup(d(6, 4), 0, FALSE, FALSE)``.
    """
    keys = jax.random.split(rng, 6)
    heal = sum(
        jax.random.randint(keys[i], (), 1, 5).astype(jnp.int32) for i in range(6)
    )
    new_hp = jnp.minimum(state["player_hp"] + heal, state["player_hp_max"])
    return {**state, "player_hp": new_hp}


def _effect_extra_healing(state: dict, rng: jax.Array) -> dict:
    """EXTRA_HEALING: heal d(6, 8) = 6..48 HP and cure blindness.

    Vendor: zap.c::zapyourself line 2911 → ``healup(d(6, 8), 0, FALSE, TRUE)``.
    The fourth healup arg ``cureblind=TRUE`` clears the BLIND timer.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    keys = jax.random.split(rng, 6)
    heal = sum(
        jax.random.randint(keys[i], (), 1, 9).astype(jnp.int32) for i in range(6)
    )
    new_hp = jnp.minimum(state["player_hp"] + heal, state["player_hp_max"])
    new_ts = state["status"].timed_statuses.at[TimedStatus.BLIND].set(0)
    new_status = state["status"].replace(timed_statuses=new_ts)
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


def _effect_fire_bolt(state: dict, rng: jax.Array) -> dict:
    """FIREBALL (unskilled): d(nd, 6) where nd = u.ulevel/2 + 1.

    Vendor: zap.c::weffects line 3461 routes SPE_FIREBALL through ubuzz with
    nd=u.ulevel/2+1; zhitm ZT_FIRE case at line 4265 uses ``d(nd, 6)``.
    (The P_SKILLED ``explode()`` AOE path with ``d(6, 6)`` per blast is
    deferred — we model the unskilled single-target ray.)
    """
    return _effect_xl_scaled_magic_attack(state, rng)


def _effect_force_bolt(state: dict, rng: jax.Array) -> dict:
    """FORCE_BOLT: d(2, 12) physical damage.

    Vendor: zap.c::bhitm line 205 ``dmg = d(2, 12)`` for FORCE_BOLT path
    (and zapyourself line 2722 same formula self-zap branch).
    """
    return _effect_attack_ray(state, rng, 2, 12)


def _effect_cone_of_cold(state: dict, rng: jax.Array) -> dict:
    """CONE_OF_COLD (unskilled): d(nd, 6) where nd = u.ulevel/2 + 1.

    Vendor: zap.c::weffects line 3461 routes SPE_CONE_OF_COLD through ubuzz
    with nd=u.ulevel/2+1; zhitm ZT_COLD case at line 4283 uses ``d(nd, 6)``.
    """
    return _effect_xl_scaled_magic_attack(state, rng)


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
    """DRAIN_LIFE: deal ``monhp_per_lvl(mon)`` = 1d8 damage to monster.

    Vendor: zap.c::bhitm line 521-543 ``dmg = monhp_per_lvl(mtmp)``;
    makemon.c::monhp_per_lvl line 989 default is ``rnd(8)`` = 1..8.
    """
    return _effect_attack_ray(state, rng, 1, 8)


def _effect_chain_lightning(state: dict, rng: jax.Array) -> dict:
    return _effect_attack_ray(state, rng, 4, 6)


def _effect_detect_monsters(state: dict, rng: jax.Array) -> dict:
    """DETECT_MONSTERS: set identification.detect_monsters_until_turn.

    Vendor: vendor/nethack/src/detect.c::monster_detect — reveals every
    monster on the current level for a number of turns equal to the spell's
    base duration (typically 100 turns).  We track that window via a
    timer on IdentificationState; observation code may consult it later.
    Cite: vendor/nethack/src/detect.c::monster_detect.
    """
    ts = state["timestep"].astype(jnp.int32)
    ident = state["identification"]
    new_ident = ident.replace(
        detect_monsters_until_turn=ts + jnp.int32(100),
    )
    return {**state, "identification": new_ident}


def _effect_detect_food(state: dict, rng: jax.Array) -> dict:
    """DETECT_FOOD: set identification.detect_food_until_turn.

    Vendor: vendor/nethack/src/detect.c::food_detect — reveals every food
    item on the current level until the timer expires.
    Cite: vendor/nethack/src/detect.c::food_detect.
    """
    ts = state["timestep"].astype(jnp.int32)
    ident = state["identification"]
    new_ident = ident.replace(
        detect_food_until_turn=ts + jnp.int32(100),
    )
    return {**state, "identification": new_ident}


def _effect_detect_treasure(state: dict, rng: jax.Array) -> dict:
    """DETECT_TREASURE: set identification.detect_treasure_until_turn.

    Vendor: vendor/nethack/src/detect.c::trap_detect routes the treasure
    detection branch — we record an expiry timestep on IdentificationState.
    Cite: vendor/nethack/src/detect.c::trap_detect (treasure-detect branch).
    """
    ts = state["timestep"].astype(jnp.int32)
    ident = state["identification"]
    new_ident = ident.replace(
        detect_treasure_until_turn=ts + jnp.int32(100),
    )
    return {**state, "identification": new_ident}


def _effect_detect_unseen(state: dict, rng: jax.Array) -> dict:
    return state


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
    new_items = items.replace(identified=new_identified)
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
    """CAUSE_FEAR: set all monsters to FLEE."""
    from Nethax.nethax.subsystems.monster_ai import MoveStrategy
    mai = state["monster_ai"]
    if hasattr(mai, "mstrategy"):
        alive_mask = mai.hp > 0 if hasattr(mai, "hp") else jnp.zeros(mai.mstrategy.shape, jnp.bool_)
        new_mstrategy = jnp.where(alive_mask, jnp.int32(MoveStrategy.FLEE), mai.mstrategy)
        return {**state, "monster_ai": mai.replace(mstrategy=new_mstrategy)}
    return state


def _effect_slow_monster(state: dict, rng: jax.Array) -> dict:
    """SLOW_MONSTER: set monster slot 0 to PARALYZE for one turn."""
    return _effect_sleep(state, rng)


def _effect_protection(state: dict, rng: jax.Array) -> dict:
    """PROTECTION: grant timed PROTECTION intrinsic.

    Vendor: spell.c::cast_protection line 1169-1172 — sets
    ``u.uspmtime = P_EXPERT ? 20 : 10`` per stack of ``u.uspellprot``.
    Wave 6 simplified: grant 10-turn PROTECTION intrinsic on each cast.
    Cite: vendor/nethack/src/spell.c::cast_protection.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    current = state["status"].timed_intrinsics[Intrinsic.PROTECTION]
    new_val = jnp.maximum(current, jnp.int32(10))
    new_timers = state["status"].timed_intrinsics.at[Intrinsic.PROTECTION].set(new_val)
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
    new_inv = inv.replace(items=new_items)
    return {**state, "inventory": new_inv}


def _effect_turn_undead(state: dict, rng: jax.Array) -> dict:
    """TURN_UNDEAD: monsters flee; deal 1d8 vs undead.

    Vendor: zap.c::bhitm line 243 (case SPE_TURN_UNDEAD) — undead/vampires
    take ``rnd(8)`` damage (with spell_damage_bonus) and ``monflee()``.
    Wave 6 simplification: set monster slot 0 to FLEE strategy.
    """
    from Nethax.nethax.subsystems.monster_ai import MoveStrategy
    mai = state["monster_ai"]
    if hasattr(mai, "mstrategy") and mai.mstrategy.shape[0] > 0:
        alive = mai.hp[0] > 0 if hasattr(mai, "hp") else jnp.bool_(True)
        new_strat = jnp.where(alive, jnp.int32(MoveStrategy.FLEE), mai.mstrategy[0])
        new_mstrategy = mai.mstrategy.at[0].set(new_strat)
        return {**state, "monster_ai": mai.replace(mstrategy=new_mstrategy)}
    return state


def _effect_restore_ability(state: dict, rng: jax.Array) -> dict:
    """RESTORE_ABILITY: reset all six player abilities to 18.

    Vendor: vendor/nethack/src/potion.c::peffect_restore_ability — restores
    every drained ability score to its undrained value.  Wave 6 simplification
    sets STR/DEX/CON/INT/WIS/CHA = 18 (the standard cap for a fresh roll).
    Cite: vendor/nethack/src/potion.c::peffect_restore_ability.
    """
    return {
        **state,
        "player_str": jnp.int16(18),
        "player_dex": jnp.int8(18),
        "player_con": jnp.int8(18),
        "player_int": jnp.int8(18),
        "player_wis": jnp.int8(18),
        "player_cha": jnp.int8(18),
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
    """JUMPING: shift player_pos by (0, +2) when the destination is FLOOR.

    Vendor: vendor/nethack/src/cmd.c::dojump — picks a destination within
    a small range and verifies the target tile is walkable.  Wave 6 minimum:
    east-2-tile jump, only commits if the target is FLOOR.
    Cite: vendor/nethack/src/cmd.c::dojump.
    """
    from Nethax.nethax.constants.tiles import TileType
    pos = state["player_pos"]
    new_col = pos[1].astype(jnp.int32) + jnp.int32(2)
    new_row = pos[0].astype(jnp.int32)
    # Bounds check on map width.
    br = state["dungeon"].current_branch.astype(jnp.int32)
    lv = state["dungeon"].current_level.astype(jnp.int32) - jnp.int32(1)
    terrain = state["terrain"]
    h = jnp.int32(terrain.shape[2])
    w = jnp.int32(terrain.shape[3])
    in_bounds = (new_row >= 0) & (new_row < h) & (new_col >= 0) & (new_col < w)
    safe_row = jnp.clip(new_row, 0, h - 1)
    safe_col = jnp.clip(new_col, 0, w - 1)
    tile = terrain[br, lv, safe_row, safe_col]
    walkable = tile == jnp.int8(int(TileType.FLOOR))
    commit = in_bounds & walkable
    out_pos = jnp.where(
        commit,
        jnp.stack([new_row.astype(jnp.int16), new_col.astype(jnp.int16)]),
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


def _effect_create_monster(state: dict, rng: jax.Array) -> dict:
    """CREATE_MONSTER: spawn a hostile monster in the first empty slot.

    Vendor: vendor/nethack/src/makemon.c::makemon — generic hostile monster
    summon used by SPELL_CREATE_MONSTER and SCR_CREATE_MONSTER.
    Wave 6 minimum: pick a free slot, set alive=True, tame=False,
    peaceful=False, hp=hp_max=8, entry_idx=2 (placeholder hostile species),
    place adjacent (west) of the player.
    Cite: vendor/nethack/src/makemon.c::makemon, vendor/nethack/src/read.c SCR_CREATE_MONSTER.
    """
    mai = state["monster_ai"]
    free_mask = ~mai.alive
    any_free = jnp.any(free_mask)
    slot = jnp.argmax(free_mask.astype(jnp.int32)).astype(jnp.int32)

    pos = state["player_pos"]
    spawn_pos = jnp.stack(
        [pos[0].astype(jnp.int16),
         (pos[1].astype(jnp.int32) - jnp.int32(1)).astype(jnp.int16)]
    )

    new_alive    = mai.alive.at[slot].set(jnp.where(any_free, jnp.bool_(True),   mai.alive[slot]))
    new_tame     = mai.tame.at[slot].set(jnp.where(any_free, jnp.bool_(False),   mai.tame[slot]))
    new_peaceful = mai.peaceful.at[slot].set(jnp.where(any_free, jnp.bool_(False), mai.peaceful[slot]))
    new_hp_max   = mai.hp_max.at[slot].set(jnp.where(any_free, jnp.int32(8),     mai.hp_max[slot]))
    new_hp       = mai.hp.at[slot].set(jnp.where(any_free, jnp.int32(8),         mai.hp[slot]))
    new_pos      = mai.pos.at[slot].set(jnp.where(any_free, spawn_pos,           mai.pos[slot]))
    new_entry    = mai.entry_idx.at[slot].set(jnp.where(any_free, jnp.int16(2),  mai.entry_idx[slot]))

    new_mai = mai.replace(
        alive=new_alive, tame=new_tame, peaceful=new_peaceful,
        hp=new_hp, hp_max=new_hp_max, pos=new_pos, entry_idx=new_entry,
    )
    return {**state, "monster_ai": new_mai}


def _effect_summon_nasties(state: dict, rng: jax.Array) -> dict:
    """SUMMON_NASTIES: hostile monster spawn (Wave 3: no-op)."""
    return state


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
    """CLAIRVOYANCE: mark the entire current level as explored.

    Vendor: vendor/nethack/src/detect.c::do_vicinity_map — reveals tiles
    around the caster (extended through the spell).  Wave 6 minimum:
    set explored[branch, level, :, :] = True for the entire level.
    Cite: vendor/nethack/src/detect.c::do_vicinity_map (extended scope).
    """
    br = state["dungeon"].current_branch.astype(jnp.int32)
    lv = state["dungeon"].current_level.astype(jnp.int32) - jnp.int32(1)
    explored = state["explored"]
    H = explored.shape[2]
    W = explored.shape[3]
    true_layer = jnp.ones((H, W), dtype=jnp.bool_)
    new_explored = explored.at[br, lv].set(true_layer)
    return {**state, "explored": new_explored}


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

def cast_spell(state, rng: jax.Array, spell_id: int) -> tuple:
    """Cast spell_id.  Returns (new_state, success: bool).

    Steps (from spell.c:spelleffects_check + spelleffects):
      1. Pw cost = spell_level * 5  (spell.h:SPELL_LEV_PW)
      2. Check player_pw >= cost; fail early with no Pw spent
      3. Roll d100 failure chance (percent_success simplified formula)
      4. On success: dispatch to effect handler
      5. Decrement Pw by cost
      6. Decrement spell_memory by 1 (Wave 3 simplified decay)

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

    # Wave 6 #73 fix: vendor spell.c::percent_success returns the SUCCESS
    # percentage (chance-of-cast).  spell_fail_chance returns 100 - success
    # (the fail %).  Vendor logic (spell.c lines 2290-2310) compares
    # ``rnd(100) > chance`` → fail; equivalently ``roll < chance`` → success.
    # We use spell_success_chance for the cast-success probability directly.
    # Cite: vendor/nethack/src/spell.c::percent_success.
    success_pct = spell_success_chance(
        state.player_role.astype(jnp.int32),
        jnp.int32(sid),
        state.player_xl.astype(jnp.int32),
        state.player_int.astype(jnp.int32),
        state.player_wis.astype(jnp.int32),
    )
    rng, sub = jax.random.split(rng)
    roll = jax.random.randint(sub, (), 0, 100)
    failed = bool(roll >= success_pct)

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

    # Decrement Pw
    adapter["player_pw"] = jnp.maximum(
        adapter["player_pw"] - jnp.int32(pw_cost), jnp.int32(0)
    )

    # Hunger drain on successful cast (vendor spell.c:spelleffects_check lines 1322-1367).
    # Vendor: morehungry(energy * 2) where energy = spelllev * 5.
    # Wizard reduction: hunger cost reduced by ACURR(A_INT) for wizards (line ~1340).
    if not failed:
        nutrition_cost = jnp.int32(lv * 5 * 2)
        is_wizard = jnp.int32(state.player_role) == jnp.int32(_ROLE_WIZARD)
        wiz_reduction = jnp.minimum(
            nutrition_cost, state.player_int.astype(jnp.int32)
        )
        nutrition_cost = jnp.where(is_wizard, nutrition_cost - wiz_reduction, nutrition_cost)
        old_nutrition = adapter["status"].nutrition
        new_nutrition = jnp.maximum(old_nutrition - nutrition_cost, jnp.int32(0))
        adapter["status"] = adapter["status"].replace(nutrition=new_nutrition)

    # Decrement spell memory (Wave 3 simplified: -1 per cast, floor 0)
    magic = adapter["magic"]
    new_mem = magic.spell_memory.at[sid].set(
        jnp.maximum(magic.spell_memory[sid] - jnp.int32(1), jnp.int32(0))
    )
    adapter["magic"] = magic.replace(spell_memory=new_mem)

    return adapter.build(), not failed


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
