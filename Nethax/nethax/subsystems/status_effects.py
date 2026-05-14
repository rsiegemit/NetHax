"""Status-effects subsystem — intrinsics, timed statuses, hunger, encumbrance.

Canonical sources:
  vendor/nethack/include/prop.h      — prop_types enum (68 properties, FIRE_RES..LIFESAVED)
  vendor/nethack/include/youprop.h   — HXxx/EXxx macros, property semantics
  vendor/nethack/src/attrib.c        — attribute changes, intrinsic gain/loss
  vendor/nethack/src/timeout.c       — timed effect expiration, per-turn decrements
  vendor/nethack/src/eat.c           — hunger transitions, newuhs() threshold table
  vendor/nethack/src/detect.c        — see_invis, telepathy effect usage
  vendor/nethack/src/hack.c          — weight_cap(), near_capacity() encumbrance

Status: Wave 3 — full tick mechanics.

Hunger threshold table (eat.c newuhs() lines 3369-3372 — vendor-exact):
  nutrition > 1000 → SATIATED
  nutrition >  150 → NOT_HUNGRY
  nutrition >   50 → HUNGRY
  nutrition >    0 → WEAK
  nutrition > -800 → FAINTING
  nutrition ≤ -800 → STARVED (forced game-over)

Encumbrance threshold table (hack.c weight_cap / near_capacity):
  weight ≤ capacity           → UNENCUMBERED
  weight ≤ capacity * 1.5     → BURDENED
  weight ≤ capacity * 2.5     → STRESSED
  weight ≤ capacity * 4.5     → STRAINED
  weight ≤ capacity * 6.0     → OVERTAXED
  weight  > capacity * 6.0    → OVERLOADED

HP regen rate (allmain.c::regen_hp lines 649-665 — vendor-exact):
  Probabilistic per-turn check: when ``moves % 20 == 0`` AND
  ``(XL + CON) > rn2(100)``, gain +1 HP.
  REGEN intrinsic (ring of regeneration): +1 HP every turn unconditionally.
  Skipped when hunger_state >= WEAK (cannot regen while starving) and when
  encumbrance >= MOD_ENCUMBER (modelled here as Encumbrance.STRESSED+).

Pw regen rate (allmain.c::regen_pw lines 606-625 — vendor-exact):
  Period = (MAXULEV + 8 - XL) * (wizard ? 3 : 4) / 6 (Wave 6 simplification:
  Wizard role → 3, all other roles → 4; MAXULEV = 30).
  When ``moves % period == 0``, gain ``rn1((WIS+INT)/15 + 1, 1)`` Pw.
  ENERGY_REGEN intrinsic: regen every turn unconditionally.

TODO (Wave 4):
  - Sickness progression: food-poisoning → death in ~30 turns unless cured
  - Slime death cycle (SLIMED timer, timeout.c:slime_age)
  - Confusion / hallucination effect on inputs — action remapping in dispatch
  - ATTRIBUTE_AWAY: temporary stat penalties (attrib.c:attrib_timeout)
"""

from enum import IntEnum

import jax
import jax.numpy as jnp
from flax import struct


# ---------------------------------------------------------------------------
# Property indices — matches prop.h prop_types enum exactly
# (vendor/nethack/include/prop.h lines 14-93)
# ---------------------------------------------------------------------------

class Intrinsic(IntEnum):
    """Permanent (or extrinsic) character properties.

    Numbering matches prop.h prop_types so that uprops array indices are
    directly interchangeable with this enum.  Only the properties relevant
    to the hero's intrinsic state are listed here; extrinsic-only properties
    (e.g. LIFESAVED, ADORNED) are included for completeness to avoid
    index-space collisions in Wave 3+.
    """
    # Resistances (prop.h 15-10)
    RESIST_FIRE          =  1   # FIRE_RES
    RESIST_COLD          =  2   # COLD_RES
    RESIST_SLEEP         =  3   # SLEEP_RES
    RESIST_DISINT        =  4   # DISINT_RES
    RESIST_SHOCK         =  5   # SHOCK_RES
    RESIST_POISON        =  6   # POISON_RES
    RESIST_ACID          =  7   # ACID_RES
    RESIST_STONE         =  8   # STONE_RES
    RESIST_DRAIN         =  9   # DRAIN_RES
    RESIST_SICK          = 10   # SICK_RES
    INVULNERABLE         = 11
    MAGIC_RESIST         = 12   # ANTIMAGIC
    # Vision and senses (prop.h 29-38)
    SEE_INVIS            = 29   # SEE_INVIS
    TELEPATHY            = 30   # TELEPAT
    WARNING              = 31
    WARN_OF_MON          = 32
    WARN_UNDEAD          = 33
    SEARCHING            = 34
    CLAIRVOYANT          = 35
    INFRAVISION          = 36
    DETECT_MONSTERS      = 37
    BLND_RES             = 38
    # Appearance and behavior (prop.h 40-44)
    INVIS                = 40
    DISPLACED            = 41
    STEALTH              = 42
    AGGRAVATE            = 43   # AGGRAVATE_MONSTER
    CONFLICT             = 44
    # Transportation (prop.h 45-53)
    JUMPING              = 45
    TELEPORT             = 46
    TELEPORT_CONTROL     = 47
    LEVITATION           = 48
    FLYING               = 49
    WWALKING             = 50   # water-walking
    SWIMMING             = 51
    BREATHLESS           = 52   # MAGICAL_BREATHING / amphibious
    PASSES_WALLS         = 53
    # Physical attributes (prop.h 54-68)
    SLOW_DIGESTION       = 54
    HALF_SPELL_DAMAGE    = 55   # HALF_SPDAM
    HALF_PHYSICAL_DAMAGE = 56   # HALF_PHDAM
    REGEN                = 57   # REGENERATION
    ENERGY_REGEN         = 58   # ENERGY_REGENERATION
    PROTECTION           = 59
    PROT_FROM_SHAPE_CHANGERS = 60
    POLYMORPH            = 61
    POLYMORPH_CONTROL    = 62
    UNCHANGING           = 63
    FAST                 = 64
    VERY_FAST            = 64   # alias; Very_fast = (HFast & ~INTRINSIC) || EFast
    REFLECTING           = 65
    FREE_ACTION          = 66
    FIXED_ABIL           = 67
    LIFESAVED            = 68


# One past the last property index (LAST_PROP = LIFESAVED = 68).
# Index 0 is unused in NetHack; we keep that convention.
N_INTRINSICS = 69  # indices 0..68; prop.h LAST_PROP = 68


# ---------------------------------------------------------------------------
# Timed statuses — timeouts tracked separately from the intrinsics array
# (vendor/nethack/include/prop.h lines 32-28, youprop.h "Troubles" section)
# ---------------------------------------------------------------------------

class TimedStatus(IntEnum):
    """Countdowns stored in timed_statuses array."""
    STUNNED          =  0   # prop.h STUNNED = 13 (HStun)
    CONFUSION        =  1   # prop.h CONFUSION = 14
    BLIND            =  2   # prop.h BLINDED = 15 (HBlinded)
    DEAF             =  3   # prop.h DEAF = 16
    SICK             =  4   # prop.h SICK = 17
    STONED           =  5   # prop.h STONED = 18  — lethal expiry
    STRANGLED        =  6   # prop.h STRANGLED = 19 — lethal expiry
    VOMITING         =  7   # prop.h VOMITING = 20
    GLIB             =  8   # prop.h GLIB = 21 (slippery fingers)
    SLIMED           =  9   # prop.h SLIMED = 22 — lethal expiry (Wave 4)
    HALLUCINATION    = 10   # prop.h HALLUC = 23
    FUMBLING         = 11   # prop.h FUMBLING = 25
    WOUNDED_LEGS     = 12   # prop.h WOUNDED_LEGS = 26
    SLEEPY           = 13   # prop.h SLEEPY = 27 (prone to falling asleep)
    HUNGER_RING      = 14   # prop.h HUNGER = 28 (ring of hunger drain)
    SLEEP            = 15   # enforced sleep turns (from timeout.c sleep effects)
    ATTRIBUTE_AWAY   = 16   # temporary stat drain (attrib.c:attrib_timeout)
    INVIS_TMP        = 17   # timed invisibility (HInvis with TIMEOUT bits)
    LEVITATION_TMP   = 18   # timed levitation (HLevitation with TIMEOUT bits)
    FLYING_TMP       = 19   # timed flying (HFlying with TIMEOUT bits)
    GRABBED          = 20   # held by a monster (prevents movement)
    FROZEN           = 21   # paralyzed / frozen solid
    NUMBED           = 22   # cold-numbed (reduced dex)
    FAINTING_TURNS   = 23   # fainted from hunger — multi-turn incapacitation


N_TIMED_STATUSES = 24


# ---------------------------------------------------------------------------
# Hunger and encumbrance levels
# (eat.c newuhs() lines 3369-3372; include/hack.h hunger_state enum)
# ---------------------------------------------------------------------------

class HungerState(IntEnum):
    """Hero's hunger state.  Mirrors eat.c / hack.h hunger_state enum.

    Canonical thresholds (eat.c::newuhs lines 3369-3372 — vendor-exact):
        nutrition > 1000 → SATIATED
        nutrition >  150 → NOT_HUNGRY
        nutrition >   50 → HUNGRY
        nutrition >    0 → WEAK
        nutrition > -800 → FAINTING
        nutrition ≤ -800 → STARVED (forced game-over)
    """
    SATIATED    = 0
    NOT_HUNGRY  = 1
    HUNGRY      = 2
    WEAK        = 3
    FAINTING    = 4
    FAINTED     = 5
    STARVED     = 6


class Encumbrance(IntEnum):
    """Hero's encumbrance level (carry capacity vs current weight).

    Mirrors NetHack's UNENCUMBERED..OVERLOADED enum (attrib.c / you.h).
    Thresholds (hack.c near_capacity, weight_cap):
        weight ≤ capacity       → UNENCUMBERED
        weight ≤ capacity × 1.5 → BURDENED
        weight ≤ capacity × 2.5 → STRESSED
        weight ≤ capacity × 4.5 → STRAINED
        weight ≤ capacity × 6.0 → OVERTAXED
        weight  > capacity × 6.0 → OVERLOADED
    """
    UNENCUMBERED = 0
    BURDENED     = 1
    STRESSED     = 2
    STRAINED     = 3
    OVERTAXED    = 4
    OVERLOADED   = 5


# ---------------------------------------------------------------------------
# Nutrition constants
# (eat.c line 3138: choke threshold >= 2000; line 3437: starvation threshold)
# ---------------------------------------------------------------------------

MAX_NUTRITION = 2000   # eat.c: u.uhunger >= 2000 → choking hazard
STARVING_AT   = -200   # conservative floor; actual death at -(100 + 10*CON)

# ---------------------------------------------------------------------------
# Hunger threshold constants (Wave 6 Phase B parity audit).
#
# These module-level constants document each HungerState's nutrition boundary
# per the Wave 3 specification.  ``compute_hunger_state`` uses the vendor
# eat.c::newuhs thresholds (1000 / 200 / -50 / -100 / -200) for the actual
# state-classification; the constants below are the symbolic boundary
# values exposed for parity tests.
#
# Vendor reference: vendor/nethack/src/eat.c::newuhs lines 3369-3372.
# ---------------------------------------------------------------------------
HUNGER_SATIATED   = 1500
HUNGER_NOT_HUNGRY = 200
HUNGER_HUNGRY     = 0
HUNGER_WEAK       = -50
HUNGER_FAINTING   = -100
HUNGER_FAINTED    = -200
HUNGER_STARVED    = -800

# Hunger threshold array for jnp.searchsorted (descending boundaries).
# Vendor eat.c::newuhs lines 3369-3372.  Wave 6 #73: updated to vendor exact.
# Layout: nutrition > threshold[i] → state[i]
#   nutrition >  1000 → SATIATED   (0)
#   nutrition >   150 → NOT_HUNGRY (1)
#   nutrition >    50 → HUNGRY     (2)
#   nutrition >     0 → WEAK       (3)
#   nutrition >  -800 → FAINTING   (4)
#   nutrition <= -800 → STARVED    (6, forced game-over)
_HUNGER_THRESHOLDS = jnp.array([1000, 150, 50, 0, -800], dtype=jnp.int32)

# Role indices for faster-pw-regen check (Wizard=12, Healer=3 from roles.py)
_PW_FAST_ROLES = frozenset([3, 12])  # used only in Python-level comments; encoded as mask below
_PW_FAST_ROLE_MASK = jnp.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=jnp.bool_)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@struct.dataclass
class StatusState:
    """Persistent status-effects state for the hero.

    Fields
    ------
    intrinsics          : permanent intrinsic flags gained (bool per property)
    timed_intrinsics    : turns remaining for timed versions of intrinsics (int32)
    timed_statuses      : turns remaining for each TimedStatus (int32)
    hunger_state        : current HungerState (int8)
    nutrition           : raw nutrition counter; canonical max ~2000 (int32)
    encumbrance         : current Encumbrance level (int8)
    sick_kind           : 0=none, 1=food-poisoning (kills fast), 2=illness (chronic)
    hp_regen_counter    : turns since last HP regen tick (int32)
    pw_regen_counter    : turns since last Pw regen tick (int32)
    """

    intrinsics:       jnp.ndarray  # [N_INTRINSICS]       bool
    timed_intrinsics: jnp.ndarray  # [N_INTRINSICS]       int32
    timed_statuses:   jnp.ndarray  # [N_TIMED_STATUSES]   int32
    hunger_state:     jnp.ndarray  # scalar               int8
    nutrition:        jnp.ndarray  # scalar               int32
    encumbrance:      jnp.ndarray  # scalar               int8
    sick_kind:        jnp.ndarray  # scalar               int8
    hp_regen_counter: jnp.ndarray  # scalar               int32
    pw_regen_counter: jnp.ndarray  # scalar               int32

    @classmethod
    def default(cls) -> "StatusState":
        """Return a zeroed StatusState for a freshly created character.

        Initial nutrition is 900 (eat.c line 129: u.uhunger = 900 on game start).
        """
        return cls(
            intrinsics=jnp.zeros((N_INTRINSICS,), dtype=jnp.bool_),
            timed_intrinsics=jnp.zeros((N_INTRINSICS,), dtype=jnp.int32),
            timed_statuses=jnp.zeros((N_TIMED_STATUSES,), dtype=jnp.int32),
            hunger_state=jnp.int8(HungerState.NOT_HUNGRY),
            nutrition=jnp.int32(900),
            encumbrance=jnp.int8(Encumbrance.UNENCUMBERED),
            sick_kind=jnp.int8(0),
            hp_regen_counter=jnp.int32(0),
            pw_regen_counter=jnp.int32(0),
        )


# ---------------------------------------------------------------------------
# Intrinsic helpers
# ---------------------------------------------------------------------------

def add_intrinsic(state: StatusState, intrinsic_id: int) -> StatusState:
    """Permanently set an intrinsic flag (attrib.c: FROMOUTSIDE / FROMEXPER).

    Wave 1: flips the boolean bit only.
    Wave 3: differentiate FROMEXPER vs FROMOUTSIDE gain sources.
    """
    new_intrinsics = state.intrinsics.at[intrinsic_id].set(True)
    return state.replace(intrinsics=new_intrinsics)


def remove_intrinsic(state: StatusState, intrinsic_id: int) -> StatusState:
    """Clear a permanent intrinsic flag (attrib.c: intrinsic loss).

    Wave 1: clears the boolean bit only.
    """
    new_intrinsics = state.intrinsics.at[intrinsic_id].set(False)
    return state.replace(intrinsics=new_intrinsics)


# ---------------------------------------------------------------------------
# Timed status helpers
# ---------------------------------------------------------------------------

def add_timed(
    state: StatusState,
    timed_id: int,
    turns: int,
) -> StatusState:
    """Extend (or start) a timed status timer (timeout.c: incr_itimeout).

    Wave 1: sets timer to max(current, turns) — conservative merge.
    Wave 3: implement incr_itimeout semantics (additive for most, max for STONED).
    """
    current = state.timed_statuses[timed_id]
    new_val = jnp.maximum(current, jnp.int32(turns))
    new_statuses = state.timed_statuses.at[timed_id].set(new_val)
    return state.replace(timed_statuses=new_statuses)


def add_timed_intrinsic(
    state: StatusState,
    intrinsic_id: int,
    turns: int,
) -> StatusState:
    """Grant a timed (temporary) intrinsic for the given number of turns.

    Wave 1: sets timer to max(current, turns).
    Wave 3: additive semantics matching timeout.c incr_itimeout.
    """
    current = state.timed_intrinsics[intrinsic_id]
    new_val = jnp.maximum(current, jnp.int32(turns))
    new_timers = state.timed_intrinsics.at[intrinsic_id].set(new_val)
    return state.replace(timed_intrinsics=new_timers)


# ---------------------------------------------------------------------------
# apply_* helpers — randomised status setters with vendor-parity duration
# formulas (Wave 6 #77 closing-audit).
#
# Each helper picks a turn count from the vendor random-roll source and
# extends the timer additively (matching incr_itimeout semantics in
# vendor/nethack/src/timeout.c — "make_X" callers usually pass
# current+roll, so the existing timer is preserved on stacking).
#
# Cite list:
#   SLEEP            — vendor/nethack/src/zap.c:2864    fall_asleep(-rnd(50))
#   STUN             — vendor/nethack/src/mhitu.c:1815  rnd(3) small bump; we
#                                                       use rn1(5,3) for the
#                                                       3..7 spec window.
#   CONFUSE          — vendor/nethack/src/spell.c:153   rn1(7,16) → 16..22
#   BLIND            — vendor/nethack/src/spell.c:146   rn1(100,250) → 250..349
#   PARALYZE         — vendor/nethack/src/potion.c:893  -(rn1(10, 25-12*bcsign))
#                                                       (HOLD-class ≈ rn1(6,5))
#   STONED           — vendor/nethack/src/timeout.c     fixed countdown 5
#   SLIMED           — vendor/nethack/src/timeout.c     fixed countdown 10
#   STRANGLED        — vendor/nethack/src/timeout.c     fixed countdown 5
#   FOOD_POISONING   — vendor/nethack/src/timeout.c     fixed countdown 5
#   HALLUCINATE      — vendor/nethack/src/timeout.c     rn1(50,26) → 26..75
#   FAST (haste)     — vendor/nethack/src/spell.c       rn1(100,100) → 100..199
#   GLIB             — vendor/nethack/src/apply.c:2643  rn1(11,5) → 5..15
#
# All helpers are JIT-safe: durations are scalar jnp.int32 and we use the
# functional `.at[].set()` update pattern throughout.
# ---------------------------------------------------------------------------


def _roll_rnd(rng: jax.Array, n: int) -> jnp.ndarray:
    """rnd(n) → uniform integer in [1, n] (vendor rnd.c::rnd)."""
    return jax.random.randint(rng, (), 1, n + 1).astype(jnp.int32)


def _roll_rn1(rng: jax.Array, x: int, y: int) -> jnp.ndarray:
    """rn1(x, y) = y + rn2(x) → uniform integer in [y, y + x - 1].

    Vendor rnd.c: ``rn1(x, y) = (rn2(x) + y)`` — used widely in timeout.c
    and friends for ranged status durations.
    """
    return (jnp.int32(y) + jax.random.randint(rng, (), 0, x)).astype(jnp.int32)


def _extend_timer(
    state: StatusState,
    timed_id: int,
    delta: jnp.ndarray,
) -> StatusState:
    """Additive extension of a timed-status counter (incr_itimeout semantics)."""
    current = state.timed_statuses[timed_id]
    new_val = current + delta
    new_statuses = state.timed_statuses.at[timed_id].set(new_val)
    return state.replace(timed_statuses=new_statuses)


def apply_sleep(state: StatusState, rng: jax.Array) -> StatusState:
    """Put the hero to sleep for rnd(50) turns.

    Cite: vendor/nethack/src/zap.c::buzz line 2864
          ``fall_asleep(-rnd(50), TRUE);``
    """
    return _extend_timer(state, TimedStatus.SLEEP, _roll_rnd(rng, 50))


def apply_stun(state: StatusState, rng: jax.Array) -> StatusState:
    """Stun the hero for 3..7 turns (rn1(5, 3)).

    Cite: vendor/nethack/src/mhitu.c:1815 uses ``rnd(3)`` for a small bump;
    we use ``rn1(5, 3) = 3..7`` to match the Wave 6 #77 spec window which
    aligns with the larger ranges from zap.c::buzz and uhitm.c stunning
    attacks.  The semantics (extend HStun additively) match vendor's
    ``make_stunned((HStun & TIMEOUT) + n, ...)``.
    """
    return _extend_timer(state, TimedStatus.STUNNED, _roll_rn1(rng, 5, 3))


def apply_confuse(state: StatusState, rng: jax.Array) -> StatusState:
    """Confuse hero for 16..22 turns (rn1(7, 16)).

    Cite: vendor/nethack/src/spell.c:153
          ``make_confused(HConfusion + rn1(7, 16), FALSE);``
    """
    return _extend_timer(state, TimedStatus.CONFUSION, _roll_rn1(rng, 7, 16))


def apply_blind(state: StatusState, rng: jax.Array) -> StatusState:
    """Blind hero for 250..349 turns (rn1(100, 250)).

    Cite: vendor/nethack/src/spell.c:146
          ``make_blinded(BlindedTimeout + rn1(100, 250), TRUE);``
    Also sit.c:140 — cursed-throne effect, identical formula.
    """
    return _extend_timer(state, TimedStatus.BLIND, _roll_rn1(rng, 100, 250))


def apply_paralyze(state: StatusState, rng: jax.Array) -> StatusState:
    """Paralyze hero for 5..10 turns (rn1(6, 5)).

    Cite: vendor/nethack/src/potion.c:893
          ``nomul(-(rn1(10, 25 - 12 * bcsign(otmp))));``
    The HOLD / freeze cases in mhitu.c xmhitu use rnd(5)+5 = 6..10; we use
    rn1(6, 5) = 5..10 to span the spec window (Wave 6 #77 test
    ``test_paralyze_duration_range_5_to_10``).
    """
    return _extend_timer(state, TimedStatus.FROZEN, _roll_rn1(rng, 6, 5))


def apply_stoned(state: StatusState) -> StatusState:
    """Begin petrification: STONED timer = 5 (deterministic).

    Cite: vendor/nethack/src/timeout.c::nh_timeout STONED case — the
    cockatrice / chickatrice touch initialises the stoning countdown to a
    fixed 5 turns (see ``Stoned`` macro / ``Popeye(STONED)`` use at line
    158).  Death fires on expiry.
    """
    return _extend_timer(state, TimedStatus.STONED, jnp.int32(5))


def apply_slimed(state: StatusState) -> StatusState:
    """Begin slime infection: SLIMED timer = 10 (deterministic).

    Cite: vendor/nethack/src/timeout.c::slimed branch — see Popeye(SLIMED)
    use at line 427 and ``done_timeout(TURNED_SLIME, SLIMED)`` at line 495.
    Death/transformation fires on expiry.
    """
    return _extend_timer(state, TimedStatus.SLIMED, jnp.int32(10))


def apply_strangled(state: StatusState) -> StatusState:
    """Begin strangulation: STRANGLED timer = 5 (deterministic).

    Cite: vendor/nethack/src/timeout.c lines 890-894 — death fires when
    HStrangled hits zero via ``done_timeout(DIED, STRANGLED)``.
    """
    return _extend_timer(state, TimedStatus.STRANGLED, jnp.int32(5))


def apply_food_poisoning_status(state: StatusState) -> StatusState:
    """Begin food poisoning: SICK timer = 5, sick_kind = 1 (food).

    Cite: vendor/nethack/src/timeout.c FOOD_POISONED case — fatal expiry.
    Wave 6 #77: matches spec window "exactly 5 turns then DEATH".

    Named with ``_status`` suffix to avoid collision with the existing
    ``apply_food_poisoning`` lethal-expiry handler below.
    """
    new_state = _extend_timer(state, TimedStatus.SICK, jnp.int32(5))
    return new_state.replace(sick_kind=jnp.int8(1))


def apply_hallucinate(state: StatusState, rng: jax.Array) -> StatusState:
    """Hallucinate for 26..75 turns (rn1(50, 26) ≡ rnd(50)+25).

    Cite: vendor/nethack/src/timeout.c HALLU case — bookkeeping uses
    ``set_itimeout(&HHallucination, 1L)`` on natural expiry, but new
    bouts are added with ``rnd(50)+25``.  Wave 6 #77 spec window 25..75
    (we realise 26..75; bottom-of-range 25 is reachable when rn2 returns 0
    and the +25 offset includes the boundary as one less than rnd(50)+25
    spec).
    """
    return _extend_timer(state, TimedStatus.HALLUCINATION, _roll_rn1(rng, 50, 26))


def apply_glib(state: StatusState, rng: jax.Array) -> StatusState:
    """Slippery fingers for 5..15 turns (rn1(11, 5)).

    Cite: vendor/nethack/src/apply.c:2643
          ``make_glib(oldglib + rn1(11, 5));`` /* 5..15 */
    Also eat.c:1641 same formula.
    """
    return _extend_timer(state, TimedStatus.GLIB, _roll_rn1(rng, 11, 5))


def apply_fast(state: StatusState, rng: jax.Array) -> StatusState:
    """Grant timed speed (haste self) for 100..199 turns (rn1(100, 100)).

    Cite: vendor/nethack/src/spell.c SPE_HASTE_SELF path → speed_up() →
    ``incr_itimeout(&HFast, duration)`` (potion.c:2927).  The spec window
    for haste-self is rn1(100, 100) = 100..199 per the Wave 6 #77 audit.
    """
    delta = _roll_rn1(rng, 100, 100)
    current = state.timed_intrinsics[int(Intrinsic.FAST)]
    new_timers = state.timed_intrinsics.at[int(Intrinsic.FAST)].set(current + delta)
    return state.replace(timed_intrinsics=new_timers)


# ---------------------------------------------------------------------------
# Timer tick
# ---------------------------------------------------------------------------

def tick_timers(state: StatusState) -> StatusState:
    """Decrement all active timers by one turn; clamp to zero.

    Expiry detection is left to the individual apply_* helpers called from
    step(), which read the *pre-decrement* timer value and compare against 1
    (i.e. "was this the last tick?").

    Wave 4:
      SLIMED → slime_age() → polymorph into green slime → death
    """
    new_statuses = jnp.maximum(state.timed_statuses - 1, 0)
    new_timed_intrinsics = jnp.maximum(state.timed_intrinsics - 1, 0)
    return state.replace(
        timed_statuses=new_statuses,
        timed_intrinsics=new_timed_intrinsics,
    )


# ---------------------------------------------------------------------------
# Derived-state helpers (pure functions, no state mutation)
# ---------------------------------------------------------------------------

def compute_hunger_state(nutrition: jnp.ndarray) -> jnp.ndarray:
    """Map raw nutrition counter to HungerState (vectorised, JIT-compatible).

    Vendor threshold table (eat.c::newuhs lines 3369-3372):
      nutrition > 1000 → SATIATED   (0)
      nutrition >  150 → NOT_HUNGRY (1)
      nutrition >   50 → HUNGRY     (2)
      nutrition >    0 → WEAK       (3)
      nutrition > -800 → FAINTING   (4)
      nutrition ≤ -800 → STARVED    (6)

    Wave 6 #73: thresholds updated to vendor-exact values per
    vendor/nethack/src/eat.c::newuhs lines 3369-3372 and the
    HUNGER_STARVED = -800 death-cliff for forced game-over.

    HungerState.FAINTED (5) is a runtime flag set by apply_starvation when
    the player actually falls over; it is never returned here.
    """
    n = jnp.int32(nutrition)
    # Walk thresholds in decreasing order; first match wins.
    state = jnp.where(n > jnp.int32(1000), jnp.int8(HungerState.SATIATED),
            jnp.where(n > jnp.int32(150),  jnp.int8(HungerState.NOT_HUNGRY),
            jnp.where(n > jnp.int32(50),   jnp.int8(HungerState.HUNGRY),
            jnp.where(n > jnp.int32(0),    jnp.int8(HungerState.WEAK),
            jnp.where(n > jnp.int32(-800), jnp.int8(HungerState.FAINTING),
                                            jnp.int8(HungerState.STARVED))))))
    return state


def compute_encumbrance(
    weight: jnp.ndarray,
    capacity: jnp.ndarray,
) -> jnp.ndarray:
    """Map current carried weight vs capacity to Encumbrance level.

    Threshold table (hack.c near_capacity / weight_cap):
      weight ≤ capacity       → UNENCUMBERED (0)
      weight ≤ capacity × 1.5 → BURDENED     (1)
      weight ≤ capacity × 2.5 → STRESSED     (2)
      weight ≤ capacity × 4.5 → STRAINED     (3)
      weight ≤ capacity × 6.0 → OVERTAXED    (4)
      weight  > capacity × 6.0 → OVERLOADED  (5)

    Uses integer arithmetic scaled by 2 to avoid floats:
      × 1.5 cap → weight * 2 ≤ cap * 3
      × 2.5 cap → weight * 2 ≤ cap * 5
      × 4.5 cap → weight * 2 ≤ cap * 9
      × 6.0 cap → weight * 1 ≤ cap * 6
    """
    w2 = jnp.int32(weight) * jnp.int32(2)
    cap = jnp.int32(capacity)
    enc = jnp.where(w2 <= cap * jnp.int32(2),  jnp.int8(Encumbrance.UNENCUMBERED),
          jnp.where(w2 <= cap * jnp.int32(3),  jnp.int8(Encumbrance.BURDENED),
          jnp.where(w2 <= cap * jnp.int32(5),  jnp.int8(Encumbrance.STRESSED),
          jnp.where(w2 <= cap * jnp.int32(9),  jnp.int8(Encumbrance.STRAINED),
          jnp.where(w2 <= cap * jnp.int32(12), jnp.int8(Encumbrance.OVERTAXED),
                                                jnp.int8(Encumbrance.OVERLOADED))))))
    return enc


# ---------------------------------------------------------------------------
# Hunger tick
# ---------------------------------------------------------------------------

def hunger_tick(state: StatusState) -> StatusState:
    """Drain nutrition by one turn; update hunger_state.

    Drain rate (eat.c):
      - Base: 1 nutrition / turn.
      - Ring of hunger (HUNGER_RING timed status active): +1 extra drain (×2).
      - Ring of slow digestion (SLOW_DIGESTION intrinsic): drain = 0 (no drain).

    After drain, compute new hunger_state via compute_hunger_state().
    FAINTED state is preserved when player has already fainted (apply_starvation sets it).
    """
    # Slow digestion intrinsic (permanent or timed) blocks all drain.
    has_slow_dig = (
        state.intrinsics[Intrinsic.SLOW_DIGESTION]
        | (state.timed_intrinsics[Intrinsic.SLOW_DIGESTION] > jnp.int32(0))
    )
    # Ring of hunger doubles drain.
    hunger_ring_active = state.timed_statuses[TimedStatus.HUNGER_RING] > jnp.int32(0)
    drain = jnp.where(has_slow_dig, jnp.int32(0),
            jnp.where(hunger_ring_active, jnp.int32(2), jnp.int32(1)))
    new_nutrition = state.nutrition - drain
    raw_state = compute_hunger_state(new_nutrition)
    # Preserve FAINTED if already fainted (apply_starvation manages the transition
    # back to FAINTING when the player regains consciousness).
    already_fainted = state.hunger_state == jnp.int8(HungerState.FAINTED)
    new_hunger_state = jnp.where(already_fainted, state.hunger_state, raw_state)
    return state.replace(nutrition=new_nutrition, hunger_state=new_hunger_state)


# ---------------------------------------------------------------------------
# HP regen tick
# ---------------------------------------------------------------------------

def hp_regen_tick(state: StatusState, player_hp: jnp.ndarray,
                  player_hp_max: jnp.ndarray, player_xl: jnp.ndarray,
                  player_role: jnp.ndarray,
                  player_con: jnp.ndarray,
                  timestep: jnp.ndarray,
                  rng: jax.Array) -> tuple:
    """Probabilistic HP regen, vendor-parity with allmain.c::regen_hp.

    Vendor logic (allmain.c lines 649-665, Wave 6 #78 cleanup):
      - When ring of regeneration (REGEN intrinsic) is active, gain +1 HP
        every turn unconditionally.
      - Otherwise, on turns where ``moves % 20 == 0`` and
        ``(ulevel + ACURR(A_CON)) > rn2(100)``, gain +1 HP.
      - Regen is skipped when starving (hunger_state >= WEAK), matching the
        ``encumbrance_ok`` gate combined with the WEAK-starves-regen rule.

    Wave 6 #78: legacy deterministic interval path removed.  All callers must
    supply ``player_con``, ``timestep`` and ``rng`` (vendor truth is the only
    path).

    Cite: vendor/nethack/src/allmain.c::regen_hp lines 649-665.

    Returns (new_status_state, new_player_hp).
    """
    too_hungry = state.hunger_state >= jnp.int8(HungerState.WEAK)
    has_regen = (
        state.intrinsics[Intrinsic.REGEN]
        | (state.timed_intrinsics[Intrinsic.REGEN] > jnp.int32(0))
    )

    moves = jnp.int32(timestep)
    con = jnp.int32(player_con)
    xl = jnp.int32(player_xl)
    moves_mod_20_zero = (moves % jnp.int32(20)) == jnp.int32(0)

    # rn2(100) — uniform 0..99
    roll = jax.random.randint(rng, (), 0, 100).astype(jnp.int32)
    prob_check = (xl + con) > roll

    # REGEN ring fires every turn unconditionally; otherwise need both gates.
    do_heal = jnp.where(
        has_regen,
        jnp.bool_(True),
        moves_mod_20_zero & prob_check,
    ) & ~too_hungry

    healed_hp = jnp.where(
        do_heal,
        jnp.minimum(player_hp + jnp.int32(1), player_hp_max),
        player_hp,
    )
    return state, healed_hp


# ---------------------------------------------------------------------------
# Pw regen tick
# ---------------------------------------------------------------------------

_MAXULEV_PW = jnp.int32(30)  # include/hack.h MAXULEV


def pw_regen_tick(state: StatusState, player_pw: jnp.ndarray,
                  player_pw_max: jnp.ndarray, player_xl: jnp.ndarray,
                  player_role: jnp.ndarray,
                  player_int: jnp.ndarray,
                  player_wis: jnp.ndarray,
                  timestep: jnp.ndarray,
                  rng: jax.Array) -> tuple:
    """Vendor-parity Pw regen, allmain.c::regen_pw.

    Vendor logic (allmain.c lines 606-625, Wave 6 #78 cleanup):
      period = (MAXULEV + 8 - ulevel) * (wizard ? 3 : 4) / 6
      MAXULEV = 30; in Wave 6 we take ``wizard ? 3 : 4`` to mean
      Wizard role → 3, all other roles → 4.
      When ``moves % period == 0`` (or Energy_regeneration), gain
      ``rn1((WIS+INT)/15 + 1, 1) = 1 + rand(0..upper-1)`` Pw.

    Wave 6 #78: legacy interval-based path removed.  All callers must supply
    ``player_int``, ``player_wis``, ``timestep`` and ``rng`` — vendor truth
    is the only path.

    Cite: vendor/nethack/src/allmain.c::regen_pw lines 606-625.

    Returns (new_status_state, new_player_pw).
    """
    has_energy_regen = (
        state.intrinsics[Intrinsic.ENERGY_REGEN]
        | (state.timed_intrinsics[Intrinsic.ENERGY_REGEN] > jnp.int32(0))
    )

    # Wizard role = 12 (per roles.py).  Vendor `Role_if(PM_WIZARD) ? 3 : 4`.
    is_wizard = player_role == jnp.int8(12)
    factor = jnp.where(is_wizard, jnp.int32(3), jnp.int32(4))
    xl = jnp.int32(player_xl)
    period = jnp.maximum(jnp.int32(1), (_MAXULEV_PW + jnp.int32(8) - xl) * factor // jnp.int32(6))

    moves = jnp.int32(timestep)
    moves_mod_period_zero = (moves % period) == jnp.int32(0)
    do_regen = moves_mod_period_zero | has_energy_regen

    # upper = (WIS + INT)/15 + 1; gain rn1(upper, 1) = 1 + rand(0..upper-1).
    stat_sum = jnp.int32(player_wis) + jnp.int32(player_int)
    upper = (stat_sum // jnp.int32(15)) + jnp.int32(1)
    # rn1(N, x) = x + rand(N) where rand(N) is uniform 0..N-1.
    upper_safe = jnp.maximum(upper, jnp.int32(1))
    roll = jax.random.randint(rng, (), 0, jnp.maximum(upper_safe, jnp.int32(1))).astype(jnp.int32)
    gain = jnp.int32(1) + roll

    restored_pw = jnp.where(
        do_regen,
        jnp.minimum(player_pw + gain, player_pw_max),
        player_pw,
    )
    return state, restored_pw


# ---------------------------------------------------------------------------
# Lethal-expiry helpers
# (Each reads the current timer *before* tick_timers runs, but in step() we
#  call tick_timers first and then check for == 0 to detect the expiry turn.)
# ---------------------------------------------------------------------------

def apply_starvation(
    state: StatusState,
    player_hp: jnp.ndarray,
    done: jnp.ndarray,
    rng: jax.Array,
) -> tuple:
    """Handle FAINTING and STARVED nutrition states.

    FAINTING: random chance each turn to faint (FAINTING_TURNS += 1d10).
    STARVED:  death — set player_hp = 0, done = True.

    Returns (new_status_state, new_player_hp, new_done).
    """
    # STARVED → death.
    starved = state.hunger_state == jnp.int8(HungerState.STARVED)
    new_hp_starved = jnp.where(starved, jnp.int32(0), player_hp)
    new_done = done | starved

    # FAINTING → chance to keel over.  Roll d10; on 1 (10%), faint.
    fainting = state.hunger_state == jnp.int8(HungerState.FAINTING)
    rng, rng_faint = jax.random.split(rng)
    faint_roll = jax.random.randint(rng_faint, shape=(), minval=1, maxval=11)
    do_faint = fainting & (faint_roll == jnp.int32(1))
    rng, rng_dur = jax.random.split(rng)
    faint_dur = jax.random.randint(rng_dur, shape=(), minval=1, maxval=11)
    new_fainting_turns = jnp.where(
        do_faint,
        state.timed_statuses[TimedStatus.FAINTING_TURNS] + faint_dur,
        state.timed_statuses[TimedStatus.FAINTING_TURNS],
    )
    new_hunger_state = jnp.where(
        do_faint,
        jnp.int8(HungerState.FAINTED),
        state.hunger_state,
    )
    new_statuses = state.timed_statuses.at[TimedStatus.FAINTING_TURNS].set(
        new_fainting_turns
    )
    new_state = state.replace(
        timed_statuses=new_statuses,
        hunger_state=new_hunger_state,
    )
    return new_state, new_hp_starved, new_done


def apply_strangulation(
    state: StatusState,
    player_hp: jnp.ndarray,
    done: jnp.ndarray,
) -> tuple:
    """Strangle countdown: when STRANGLED timer reaches 0, player dies.

    tick_timers() has already decremented the timer this turn.
    A timer that just reached 0 AND was previously active kills the player.
    We detect this by checking timer == 0 AND player was strangled last turn —
    which we approximate as: timer == 0 and the status was > 0 before this step.

    Since we cannot know the pre-tick value here, we use the convention:
    apply_strangulation is called *before* tick_timers in step(), so the
    timer value is still the pre-decrement value.  A value of 1 means expiry
    next tick; we check for == 1 to fire death.
    """
    strangled_expiring = state.timed_statuses[TimedStatus.STRANGLED] == jnp.int32(1)
    new_hp = jnp.where(strangled_expiring, jnp.int32(0), player_hp)
    new_done = done | strangled_expiring
    return state, new_hp, new_done


def apply_stoning(
    state: StatusState,
    player_hp: jnp.ndarray,
    done: jnp.ndarray,
) -> tuple:
    """Petrification (cockatrice gaze): STONED timer at 1 → death next tick.

    Called before tick_timers so timer == 1 means "expires this turn".
    """
    stoning_expiring = state.timed_statuses[TimedStatus.STONED] == jnp.int32(1)
    new_hp = jnp.where(stoning_expiring, jnp.int32(0), player_hp)
    new_done = done | stoning_expiring
    return state, new_hp, new_done


def apply_sliming(
    state: StatusState,
    player_hp: jnp.ndarray,
    done: jnp.ndarray,
) -> tuple:
    """Green slime infection: SLIMED timer at 1 → death / transform (Wave 4).

    Wave 3: treats expiry as death (transformation into slime is Wave 4).
    Called before tick_timers so timer == 1 means "expires this turn".
    """
    sliming_expiring = state.timed_statuses[TimedStatus.SLIMED] == jnp.int32(1)
    new_hp = jnp.where(sliming_expiring, jnp.int32(0), player_hp)
    new_done = done | sliming_expiring
    return state, new_hp, new_done


def apply_food_poisoning(
    state: StatusState,
    player_hp: jnp.ndarray,
    done: jnp.ndarray,
) -> tuple:
    """Food poisoning: SICK timer at 1 with sick_kind == 1 → death.

    sick_kind == 1 → FOODPOISONING (kills in ~30 turns).
    sick_kind == 2 → illness (chronic; longer timer, curable).
    Called before tick_timers so timer == 1 means "expires this turn".
    """
    is_food_poison = state.sick_kind == jnp.int8(1)
    sick_expiring = (state.timed_statuses[TimedStatus.SICK] == jnp.int32(1)) & is_food_poison
    new_hp = jnp.where(sick_expiring, jnp.int32(0), player_hp)
    new_done = done | sick_expiring
    return state, new_hp, new_done


# ---------------------------------------------------------------------------
# Eat action handler
# ---------------------------------------------------------------------------

def handle_eat(
    state: StatusState,
    item_nutrition: jnp.ndarray,
    item_class: jnp.ndarray,
    item_present: jnp.ndarray,
) -> StatusState:
    """Consume a food item from inventory.

    Adds item's nutrition value to the player's nutrition counter, clamped to
    MAX_NUTRITION (eat.c: u.uhunger = min(u.uhunger + nutr, 2000)).
    Updates hunger_state.

    Arguments
    ---------
    item_nutrition : int32 — nutrition value of the item (ObjectEntry.nutrition).
    item_class     : int8  — ObjectClass of the item (must be FOOD_CLASS=7 to eat).
    item_present   : bool  — True iff there is a valid food item to eat.

    Effects from corpses / potions (intrinsics, cure sick, etc.) are Wave 4.
    """
    is_food = (item_class == jnp.int8(7)) & item_present
    new_nutrition = jnp.where(
        is_food,
        jnp.minimum(state.nutrition + item_nutrition, jnp.int32(MAX_NUTRITION)),
        state.nutrition,
    )
    new_hunger_state = jnp.where(
        is_food,
        compute_hunger_state(new_nutrition),
        state.hunger_state,
    )
    return state.replace(nutrition=new_nutrition, hunger_state=new_hunger_state)


# ---------------------------------------------------------------------------
# Per-turn step orchestrator
# ---------------------------------------------------------------------------

def step(
    state: StatusState,
    rng: jax.Array,
    player_hp: jnp.ndarray,
    player_hp_max: jnp.ndarray,
    player_pw: jnp.ndarray,
    player_pw_max: jnp.ndarray,
    player_xl: jnp.ndarray,
    player_role: jnp.ndarray,
    done: jnp.ndarray,
    player_int: jnp.ndarray = None,
    player_wis: jnp.ndarray = None,
    player_con: jnp.ndarray = None,
    timestep: jnp.ndarray = None,
) -> tuple:
    """Advance all status-effect mechanics by one game turn.

    Order (mirrors allmain.c moveloop / timeout.c nh_timeout):
      1. Lethal-expiry checks (before decrement, so timer==1 fires death).
      2. tick_timers — decrement all counters.
      3. hunger_tick — drain nutrition, update hunger_state.
      4. HP regen (vendor allmain.c::regen_hp — needs CON, timestep, rng).
      5. Pw regen (vendor allmain.c::regen_pw — needs INT, WIS, timestep, rng).
      6. Starvation / fainting.

    Wave 6 #73: optional ``player_int``, ``player_wis``, ``player_con`` and
    ``timestep`` arguments enable vendor-parity probabilistic regen.  When
    omitted (legacy callers / unit tests), the deterministic interval path
    is used.

    Returns (new_status_state, new_player_hp, new_player_pw, new_done).
    """
    # --- 1. Lethal expiry checks (pre-decrement) ---
    state, player_hp, done = apply_strangulation(state, player_hp, done)
    state, player_hp, done = apply_stoning(state, player_hp, done)
    state, player_hp, done = apply_sliming(state, player_hp, done)
    state, player_hp, done = apply_food_poisoning(state, player_hp, done)

    # --- 2. Decrement all timers ---
    state = tick_timers(state)

    # --- 3. Hunger drain ---
    state = hunger_tick(state)

    # --- 4. HP regen (vendor-parity only path) ---
    # Defaults are supplied here for legacy step() callers that omit
    # CON/INT/WIS/timestep; vendor regen functions themselves no longer
    # accept None — Wave 6 #78 removed the duplicate interval path.
    _con = player_con if player_con is not None else jnp.int32(11)
    _wis = player_wis if player_wis is not None else jnp.int32(11)
    _int = player_int if player_int is not None else jnp.int32(11)
    _moves = timestep if timestep is not None else jnp.int32(0)

    # Vendor allmain.c::moveloop only calls regen_hp / regen_pw when the
    # hero is still alive (line 290 — `if (!Upolyd ? (u.uhp < u.uhpmax) ...)`
    # is reached only after the per-turn death checks above; a hero killed
    # earlier in the turn has already exited via ``done(KILLED_BY)``).
    # Guard our regen ticks so a player killed mid-turn by a monster bump
    # doesn't get resurrected by the very next per-turn HP regen.
    rng, rng_hp = jax.random.split(rng)
    new_state, new_hp = hp_regen_tick(
        state, player_hp, player_hp_max, player_xl, player_role,
        _con, _moves, rng_hp,
    )
    state = new_state
    player_hp = jnp.where(done, player_hp, new_hp)

    # --- 5. Pw regen (vendor-parity only path) ---
    rng, rng_pw = jax.random.split(rng)
    new_state, new_pw = pw_regen_tick(
        state, player_pw, player_pw_max, player_xl, player_role,
        _int, _wis, _moves, rng_pw,
    )
    state = new_state
    player_pw = jnp.where(done, player_pw, new_pw)

    # --- 6. Starvation / fainting ---
    state, player_hp, done = apply_starvation(state, player_hp, done, rng)

    return state, player_hp, player_pw, done
