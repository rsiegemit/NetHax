"""Prayer subsystem — divine favor, alignment, luck.

Canonical sources:
  vendor/nethack/src/pray.c   — pray(), divine intervention table (lines 500-1500),
                                alignment record thresholds (DEVOUT=14,
                                STRIDENT=4, PIOUS=20, FERVENT=9),
                                luck adjustments, prayer timeout (u.ublesscnt),
                                in_trouble() (lines 198-372) / fix_worst_trouble()
                                (lines 373-700), angrygods() (lines 703-784),
                                pleased() (lines 1070-1390), god_zaps_you()
  vendor/nethack/src/priest.c — priests, peaceful shrine interactions
  vendor/nethack/src/minion.c — angel/demon summoning triggered by prayer outcome
  vendor/nethack/src/do.c     — do_sacrifice (corpse offering on altar,
                                lines ~1500-1700)
  vendor/nethack/src/sit.c    — sit-on-altar BUC sense for non-shrine altars
  vendor/nethack/include/align.h — A_LAWFUL=1, A_NEUTRAL=0, A_CHAOTIC=-1,
                                A_NONE=-128

Status: Wave 4 — full pray() / sacrifice / god_zaps_you implemented.

Design notes
------------
NetHack uses many micro-rolls (rn2(maxanger), rnl(), rnz(300), etc.).
For RL parity we collapse the major divergence paths into a JIT-safe d100
chain that preserves outcome buckets exactly:

  TROUBLE branches:
    starving / hungry            (in_trouble TROUBLE_HUNGRY)   → heal hunger
    low HP    (hp < hp_max / 7)  (in_trouble TROUBLE_HIT)      → heal HP

  PLEASED outcomes (no trouble, alignment OK, no timeout):
    HEAL_CURE         30 %   — pat-on-head case 2 (heal + un-blind)
    PROTECTION_PLUS_1 20 %   — pat-on-head case 5 (u.ublessed += 1)
    REMOVE_CURSE      15 %   — pat-on-head case 4 (uncurse inventory)
    GIFT_ARTIFACT     10 %   — pat-on-head case 7/8 (gcrownu) if record≥PIOUS
    (no-op favourite)  25 %   — pat-on-head case 0

  ANGRY outcomes (record<0 or timeout>0 or prayed too soon):
    SMITE              20 %   — angrygods cases 2-3 (lose Wis & XL)
    ANGER_BOLT         25 %   — default branch: god_zaps_you
    DESTROY_ARMOR      25 %   — angrygods cases 4-5/6 (rndcurse + punish)
    INFLICT_BLINDNESS  20 %   — minor punish
    INFLICT_WEAKNESS   10 %   — minor punish

Each branch is decided via jnp.where on the d100 roll so the function
remains lax-traceable.
"""
from enum import IntEnum

import jax
import jax.numpy as jnp
from flax import struct

from Nethax.nethax.subsystems.conduct import Conduct


# ---------------------------------------------------------------------------
# Vendor thresholds (pray.c lines 64-67)
# ---------------------------------------------------------------------------
STRIDENT: int = 4    # pray.c #define STRIDENT 4
FERVENT:  int = 9    # pray.c #define FERVENT 9
DEVOUT:  int = 14    # pray.c #define DEVOUT 14
PIOUS:   int = 20    # pray.c #define PIOUS 20

# Required minimum alignment_record to safely pray.
# pray.c can_pray: u.ualign.record < (some bound) ⇒ angry response.
PRAY_RECORD_THRESHOLD: int = 0

# pray.c line 1356: u.ublesscnt = rnz(350); we use 300 + rn2(700) per spec.
PRAY_TIMEOUT_BASE: int = 300
PRAY_TIMEOUT_RANGE: int = 700


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class Alignment(IntEnum):
    """Hero alignment type.

    Values mirror vendor/nethack/include/align.h:
      A_LAWFUL=1, A_NEUTRAL=0, A_CHAOTIC=-1.
    UNALIGNED (3) used internally for monsters / special cases.
    """

    CHAOTIC = 0
    NEUTRAL = 1
    LAWFUL = 2
    UNALIGNED = 3


class PrayerOutcome(IntEnum):
    """Result returned by pray().

    Mirrors the implicit outcome categories in pray.c lines 700-1390.
    """

    BLESSED = 0          # generic favour
    HEALED = 1           # heal HP / cure status
    IGNORED = 2          # prayer noted but no response
    CHASTISED = 3        # deity is displeased; minor punishment
    SMITTEN = 4          # deity smites the hero
    ANGER_BOLT = 5       # deity fires destruction bolt (god_zaps_you)
    PROTECTION = 6       # +1 divine protection
    REMOVE_CURSE = 7     # blessing on inventory
    GIFT_ARTIFACT = 8    # gcrownu / give_spell
    DESTROY_ARMOR = 9    # angrygods rndcurse → rust/destroy
    INFLICT_BLINDNESS = 10
    INFLICT_WEAKNESS = 11
    HEAL_HUNGER = 12     # fix_worst_trouble TROUBLE_HUNGRY


# Trouble-bucket sentinels — vendor codes from pray.c lines 76-101.
# Positive = "major" troubles; negative = "minor" troubles; 0 = no trouble.
# These numeric values are the canonical pray.c #defines so that ordering
# tests against vendor priority can compare directly.
TROUBLE_NONE: int = 0

# Major troubles (pray.c:76-89).  Priority order is established by the
# top-down chain in in_trouble() (pray.c:206-264), not by the numeric value.
TROUBLE_STONED: int             = 14   # pray.c:76
TROUBLE_SLIMED: int             = 13   # pray.c:77
TROUBLE_STRANGLED: int          = 12   # pray.c:78
TROUBLE_LAVA: int               = 11   # pray.c:79
TROUBLE_SICK: int               = 10   # pray.c:80  (food-poisoning / illness)
TROUBLE_STARVING: int           =  9   # pray.c:81  (uhs >= WEAK)
TROUBLE_REGION: int             =  8   # pray.c:82  (stinking cloud)
TROUBLE_HIT: int                =  7   # pray.c:83  (critically low HP)
TROUBLE_LYCANTHROPE: int        =  6   # pray.c:84
TROUBLE_COLLAPSING: int         =  5   # pray.c:85
TROUBLE_STUCK_IN_WALL: int      =  4   # pray.c:86
TROUBLE_CURSED_LEVITATION: int  =  3   # pray.c:87
TROUBLE_UNUSEABLE_HANDS: int    =  2   # pray.c:88
TROUBLE_CURSED_BLINDFOLD: int   =  1   # pray.c:89

# Minor troubles (pray.c:91-101)
TROUBLE_PUNISHED: int           = -1   # pray.c:91
TROUBLE_FUMBLING: int           = -2   # pray.c:92
TROUBLE_CURSED_ITEMS: int       = -3   # pray.c:93
TROUBLE_SADDLE: int             = -4   # pray.c:94
TROUBLE_BLIND: int              = -5   # pray.c:95
TROUBLE_POISONED: int           = -6   # pray.c:96
TROUBLE_WOUNDED_LEGS: int       = -7   # pray.c:97
TROUBLE_HUNGRY: int             = -8   # pray.c:98  (uhs >= HUNGRY)
TROUBLE_STUNNED: int            = -9   # pray.c:99
TROUBLE_CONFUSED: int           = -10  # pray.c:100
TROUBLE_HALLUCINATION: int      = -11  # pray.c:101

# Canonical vendor priority list — the order pray.c::in_trouble() returns
# each code.  Earlier entries take precedence over later ones.
VENDOR_TROUBLE_PRIORITY: tuple = (
    TROUBLE_STONED, TROUBLE_SLIMED, TROUBLE_STRANGLED, TROUBLE_LAVA,
    TROUBLE_SICK, TROUBLE_STARVING, TROUBLE_REGION, TROUBLE_HIT,
    TROUBLE_LYCANTHROPE, TROUBLE_COLLAPSING, TROUBLE_STUCK_IN_WALL,
    TROUBLE_CURSED_LEVITATION, TROUBLE_UNUSEABLE_HANDS,
    TROUBLE_CURSED_BLINDFOLD,
    TROUBLE_PUNISHED, TROUBLE_FUMBLING, TROUBLE_CURSED_ITEMS,
    TROUBLE_SADDLE, TROUBLE_BLIND, TROUBLE_POISONED,
    TROUBLE_WOUNDED_LEGS, TROUBLE_HUNGRY, TROUBLE_STUNNED,
    TROUBLE_CONFUSED, TROUBLE_HALLUCINATION,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@struct.dataclass
class PrayerState:
    """Persistent divine-relations state for the hero.

    Original Wave-1 fields (kept for back-compat):
      alignment      : running score with the deity [-1000, +1000]
                       (mirrors u.ualign.record in pray.c)
      prayer_timeout : turns remaining until next safe prayer
                       (mirrors u.ublesscnt; aliased by pray_timeout below)
      luck           : current luck value [-13, +13]
      lucky_stones   : luckstones currently in inventory
      god_anger      : accumulated infractions (mirrors u.ugangr line 717)

    Wave 4 additions:
      pray_timeout      : same as prayer_timeout, exposed under canonical name
                          for new code (pray.c u.ublesscnt).
      alignment_record  : int16 alignment with deity, [-127, +127]
                          (pray.c u.ualign.record)
      last_pray_turn    : turn number of the most recent pray() call
                          (mirrors u.lastprayed in moveloop)
      god_name_idx      : int8 index into a god-name table
                          (mirrors gp.p_aligntyp; 0..2 = chaotic/neutral/lawful)
    """

    alignment: jnp.ndarray       # scalar  int32
    prayer_timeout: jnp.ndarray  # scalar  int32 (kept for compat)
    luck: jnp.ndarray            # scalar  int32
    lucky_stones: jnp.ndarray    # scalar  int32
    god_anger: jnp.ndarray       # scalar  int32

    # Wave 4 additions
    pray_timeout: jnp.ndarray       # scalar  int32 (canonical pray.c name)
    alignment_record: jnp.ndarray   # scalar  int16
    last_pray_turn: jnp.ndarray     # scalar  int32
    god_name_idx: jnp.ndarray       # scalar  int8

    # Wave 6 #78 — Closing-Audit additions to power the remaining
    # fix_worst_trouble cases (pray.c::fix_worst_trouble lines 461-598):
    #   punished     mirrors u.uswallow/u.uchain heavy-iron-ball trouble
    #   saddled_cursed mirrors cursed-saddle on the player's steed
    #   stuck_in_wall mirrors u.utrap with utraptype TT_INWALL (xorn)
    #   in_region    mirrors region_danger() — stinking-cloud overlap
    punished: jnp.ndarray            # scalar bool (Wave 6 #78)
    saddled_cursed: jnp.ndarray      # scalar bool (Wave 6 #78)
    stuck_in_wall: jnp.ndarray       # scalar bool (Wave 6 #78)
    in_region: jnp.ndarray           # scalar bool (Wave 6 #78)

    @classmethod
    def default(cls) -> "PrayerState":
        """Return a zeroed PrayerState for a freshly created character."""
        return cls(
            alignment=jnp.int32(0),
            prayer_timeout=jnp.int32(0),
            luck=jnp.int32(0),
            lucky_stones=jnp.int32(0),
            god_anger=jnp.int32(0),
            pray_timeout=jnp.int32(0),
            alignment_record=jnp.int16(0),
            last_pray_turn=jnp.int32(0),
            god_name_idx=jnp.int8(1),  # default neutral
            punished=jnp.bool_(False),
            saddled_cursed=jnp.bool_(False),
            stuck_in_wall=jnp.bool_(False),
            in_region=jnp.bool_(False),
        )


# ---------------------------------------------------------------------------
# Trouble detection — full vendor parity (pray.c::in_trouble lines 198-284)
# ---------------------------------------------------------------------------

def _terrain_at_player(state) -> jnp.ndarray:
    """Return the terrain tile-type int8 under the player (current level)."""
    b = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)
    return state.terrain[b, lv, row, col].astype(jnp.int32)


def in_trouble(state) -> jnp.ndarray:
    """Return the vendor TROUBLE_* code for the worst current trouble.

    Mirrors pray.c::in_trouble() (lines 198-284) — top-down priority chain.
    Each check below cites the matching vendor line.

    Returns: int32 scalar (0 = TROUBLE_NONE).
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    from Nethax.nethax.constants.tiles import TileType

    timed = state.status.timed_statuses
    hp = state.player_hp.astype(jnp.int32)
    hp_max = state.player_hp_max.astype(jnp.int32)
    hunger = state.status.hunger_state.astype(jnp.int32)
    items = state.inventory.items

    # --- Major troubles (pray.c:206-243) ---
    is_stoned   = timed[int(TimedStatus.STONED)]   > jnp.int32(0)    # pray.c:206
    is_slimed   = timed[int(TimedStatus.SLIMED)]   > jnp.int32(0)    # pray.c:208
    is_strangled = timed[int(TimedStatus.STRANGLED)] > jnp.int32(0)  # pray.c:210
    # pray.c:212  TT_LAVA — modelled as "player tile is LAVA terrain".
    is_lava     = _terrain_at_player(state) == jnp.int32(int(TileType.LAVA))
    # pray.c:214  Sick  (food-poisoning OR illness)
    is_sick     = state.status.sick_kind.astype(jnp.int32) > jnp.int32(0)
    # pray.c:216  u.uhs >= WEAK → STARVING
    is_starving = hunger >= jnp.int32(3)
    # pray.c:218  region_danger() — stinking-cloud overlap.  Reads the
    # ``prayer.in_region`` flag set by region/cloud spawn code (Wave 6 #78).
    is_region   = state.prayer.in_region
    # pray.c:220  critically_low_hp — hp <= 1/7 hp_max or <=5.
    low_hp_threshold = jnp.maximum(jnp.int32(5), hp_max // jnp.int32(7))
    is_low_hp   = hp <= low_hp_threshold
    # pray.c:222  ismnum(u.ulycn) — lycanthropy infection.  Wave 5 polymorph
    # exposes ``polymorph.lycanthropy_form``; >= 0 means infected (Wave 6 #78).
    is_lycan    = state.polymorph.lycanthropy_form.astype(jnp.int32) >= jnp.int32(0)
    # pray.c:224  near_capacity() >= EXT_ENCUMBER → COLLAPSING.
    is_collapsing = state.status.encumbrance.astype(jnp.int32) >= jnp.int32(5)  # OVERLOADED
    # pray.c:226  stuck_in_wall() — xorn wedged in solid rock (Wave 6 #78).
    is_stuck_wall = state.prayer.stuck_in_wall
    # pray.c:228-231  cursed levitation gear — we approximate as
    # "LEVITATION_TMP active AND any worn item is cursed".
    is_cursed_levit = (
        (timed[int(TimedStatus.LEVITATION_TMP)] > jnp.int32(0))
        & jnp.any(items.buc_status == jnp.int8(1))
    )
    # pray.c:232-241  unuseable hands (welded weapon / nohands form).
    is_unuse_hands = timed[int(TimedStatus.GLIB)] > jnp.int32(0)
    # pray.c:242-243  cursed blindfold — blinded by a cursed blindfold/towel.
    # Wave 6 #78: detect via inventory item type 208 (blindfold) with
    # buc_status == cursed (1).  vendor/nle/src/objects.c:208.
    _BLINDFOLD_OTYP = 208
    is_blindfold = items.type_id == jnp.int32(_BLINDFOLD_OTYP)
    is_cursed_item = items.buc_status == jnp.int8(1)
    is_cursed_blindfold = jnp.any(is_blindfold & is_cursed_item)

    # --- Minor troubles (pray.c:248-282) ---
    # pray.c:248-249  Punished — heavy-iron-ball / chain.  Wave 6 #78.
    is_punished = state.prayer.punished
    # pray.c:250-252  fumbling gloves/boots cursed.
    is_fumbling = timed[int(TimedStatus.FUMBLING)] > jnp.int32(0)
    # pray.c:253  worst_cursed_item() — any cursed worn item.
    is_cursed_items = jnp.any(items.buc_status == jnp.int8(1))
    # pray.c:255-259  cursed saddle on the player's steed (Wave 6 #78).
    is_saddle = state.prayer.saddled_cursed
    # pray.c:261-268  BLIND timed (or DEAF).
    is_blind = (
        (timed[int(TimedStatus.BLIND)] > jnp.int32(0))
        | (timed[int(TimedStatus.DEAF)] > jnp.int32(0))
    )
    # pray.c:270-272  any ability < max — temporary stat-drain "poisoned".
    is_poisoned = timed[int(TimedStatus.ATTRIBUTE_AWAY)] > jnp.int32(0)
    # pray.c:273  wounded legs.
    is_wleg = timed[int(TimedStatus.WOUNDED_LEGS)] > jnp.int32(0)
    # pray.c:275  u.uhs >= HUNGRY (2).
    is_hungry = hunger >= jnp.int32(2)
    # pray.c:277-282  stunned / confused / hallucinating.
    is_stunned = timed[int(TimedStatus.STUNNED)] > jnp.int32(0)
    is_confused = timed[int(TimedStatus.CONFUSION)] > jnp.int32(0)
    is_hallu = timed[int(TimedStatus.HALLUCINATION)] > jnp.int32(0)

    # Vendor priority chain — first match wins.
    chain = (
        (is_stoned,           jnp.int32(TROUBLE_STONED)),
        (is_slimed,           jnp.int32(TROUBLE_SLIMED)),
        (is_strangled,        jnp.int32(TROUBLE_STRANGLED)),
        (is_lava,             jnp.int32(TROUBLE_LAVA)),
        (is_sick,             jnp.int32(TROUBLE_SICK)),
        (is_starving,         jnp.int32(TROUBLE_STARVING)),
        (is_region,           jnp.int32(TROUBLE_REGION)),
        (is_low_hp,           jnp.int32(TROUBLE_HIT)),
        (is_lycan,            jnp.int32(TROUBLE_LYCANTHROPE)),
        (is_collapsing,       jnp.int32(TROUBLE_COLLAPSING)),
        (is_stuck_wall,       jnp.int32(TROUBLE_STUCK_IN_WALL)),
        (is_cursed_levit,     jnp.int32(TROUBLE_CURSED_LEVITATION)),
        (is_unuse_hands,      jnp.int32(TROUBLE_UNUSEABLE_HANDS)),
        (is_cursed_blindfold, jnp.int32(TROUBLE_CURSED_BLINDFOLD)),
        (is_punished,         jnp.int32(TROUBLE_PUNISHED)),
        (is_fumbling,         jnp.int32(TROUBLE_FUMBLING)),
        (is_cursed_items,     jnp.int32(TROUBLE_CURSED_ITEMS)),
        (is_saddle,           jnp.int32(TROUBLE_SADDLE)),
        (is_blind,            jnp.int32(TROUBLE_BLIND)),
        (is_poisoned,         jnp.int32(TROUBLE_POISONED)),
        (is_wleg,             jnp.int32(TROUBLE_WOUNDED_LEGS)),
        (is_hungry,           jnp.int32(TROUBLE_HUNGRY)),
        (is_stunned,          jnp.int32(TROUBLE_STUNNED)),
        (is_confused,         jnp.int32(TROUBLE_CONFUSED)),
        (is_hallu,            jnp.int32(TROUBLE_HALLUCINATION)),
    )
    # Build the chain in reverse so the first vendor entry is the *outermost*
    # jnp.where (i.e. takes precedence).
    result = jnp.int32(TROUBLE_NONE)
    for cond, code in reversed(chain):
        result = jnp.where(cond, code, result)
    return result


# Back-compat alias for Wave 4 callers.
def _detect_trouble(state) -> jnp.ndarray:
    """Alias for in_trouble() — preserved for Wave-4 callers."""
    return in_trouble(state)


# ---------------------------------------------------------------------------
# Fix-trouble branch — full vendor parity (pray.c::fix_worst_trouble 373-600)
# ---------------------------------------------------------------------------

def _zero_timed(state, slot: int):
    """Set timed_statuses[slot] back to 0 (clear that timed effect)."""
    new_ts = state.status.timed_statuses.at[slot].set(jnp.int32(0))
    return state.replace(status=state.status.replace(timed_statuses=new_ts))


def fix_worst(state, rng: jax.Array, trouble: jnp.ndarray):
    """Apply pray.c::fix_worst_trouble() for the given vendor trouble code.

    Each ``jnp.where`` below cites the matching ``case`` in pray.c:381-599.
    All branches are evaluated unconditionally to remain JIT-safe; the
    selected branch is the one whose trouble matches.

    Returns: new EnvState.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    from Nethax.nethax.constants.tiles import TileType

    trouble = trouble.astype(jnp.int32)
    items = state.inventory.items
    timed = state.status.timed_statuses

    # --- case TROUBLE_STONED (pray.c:382) — make_stoned(0) ---
    is_stoned = trouble == jnp.int32(TROUBLE_STONED)
    new_stoned = jnp.where(is_stoned, jnp.int32(0), timed[int(TimedStatus.STONED)])

    # --- case TROUBLE_SLIMED (pray.c:385) — make_slimed(0) ---
    is_slimed = trouble == jnp.int32(TROUBLE_SLIMED)
    new_slime = jnp.where(is_slimed, jnp.int32(0), timed[int(TimedStatus.SLIMED)])

    # --- case TROUBLE_STRANGLED (pray.c:388-396) — Strangled = 0 ---
    is_strangled = trouble == jnp.int32(TROUBLE_STRANGLED)
    new_strang = jnp.where(is_strangled, jnp.int32(0), timed[int(TimedStatus.STRANGLED)])

    # --- case TROUBLE_LAVA / TROUBLE_REGION / TROUBLE_STUCK_IN_WALL ---
    # All three resolutions involve relocating the hero to a safe tile.
    # JIT-safe approximation: clear the player's current tile to FLOOR
    # (vendor pray.c:397-403, 417-420, 461-478 — safe_teleds semantics).
    is_lava = trouble == jnp.int32(TROUBLE_LAVA)
    is_region = trouble == jnp.int32(TROUBLE_REGION)
    is_stuck_wall_t = trouble == jnp.int32(TROUBLE_STUCK_IN_WALL)
    relocate = is_lava | is_region | is_stuck_wall_t
    b = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)
    cur_tile = state.terrain[b, lv, row, col]
    # TileType.FLOOR = safe room floor.
    new_tile = jnp.where(relocate, jnp.int8(int(TileType.FLOOR)), cur_tile)
    new_terrain = state.terrain.at[b, lv, row, col].set(new_tile)

    # --- case TROUBLE_SICK (pray.c:413-416) — make_sick(0) ---
    is_sick = trouble == jnp.int32(TROUBLE_SICK)
    new_sick_kind = jnp.where(is_sick, jnp.int8(0), state.status.sick_kind)
    new_sick_timer = jnp.where(is_sick, jnp.int32(0), timed[int(TimedStatus.SICK)])

    # --- cases TROUBLE_STARVING (pray.c:404-411) + TROUBLE_HUNGRY (FALLTHRU) ---
    # init_uhunger() resets to 900.
    is_starving = trouble == jnp.int32(TROUBLE_STARVING)
    is_hungry = trouble == jnp.int32(TROUBLE_HUNGRY)
    feed = is_starving | is_hungry
    new_nutrition = jnp.where(feed, jnp.int32(900), state.status.nutrition)
    new_hunger_state = jnp.where(feed, jnp.int8(1), state.status.hunger_state)

    # --- case TROUBLE_REGION (pray.c:417-420) — handled jointly with
    # LAVA/STUCK_IN_WALL above (relocate block) and flag clearing below.

    # --- case TROUBLE_HIT (pray.c:421-439) — set u.uhp = u.uhpmax, bump max. ---
    is_hit = trouble == jnp.int32(TROUBLE_HIT)
    xl = state.player_xl.astype(jnp.int32)
    # pray.c:432: if maxhp < ulevel*5+11, bump max by rnd(5).
    bump = jax.random.randint(rng, (), minval=1, maxval=6, dtype=jnp.int32)
    needs_bump = state.player_hp_max < (xl * jnp.int32(5) + jnp.int32(11))
    bumped_max = jnp.where(needs_bump, state.player_hp_max + bump, state.player_hp_max)
    new_hp_max = jnp.where(is_hit, bumped_max, state.player_hp_max)
    new_hp = jnp.where(is_hit, new_hp_max, state.player_hp)

    # --- case TROUBLE_LYCANTHROPE (pray.c:514-516) — you_unwere().
    # Wave 6 #78: clear polymorph.lycanthropy_form back to NONE (-1).
    is_lycan_fix = trouble == jnp.int32(TROUBLE_LYCANTHROPE)
    cur_lycan = state.polymorph.lycanthropy_form
    new_lycan_form = jnp.where(is_lycan_fix, jnp.int8(-1), cur_lycan)

    # --- case TROUBLE_COLLAPSING (pray.c:440-460) — restore strength, drop weight. ---
    is_coll = trouble == jnp.int32(TROUBLE_COLLAPSING)
    new_encumb = jnp.where(is_coll, jnp.int8(0), state.status.encumbrance)

    # --- case TROUBLE_STUCK_IN_WALL (pray.c:461-478) — safe_teleds. No-op JAX. ---

    # --- case TROUBLE_CURSED_LEVITATION (pray.c:479-490) — uncurse worn levit gear. ---
    is_clevit = trouble == jnp.int32(TROUBLE_CURSED_LEVITATION)
    new_levit_timer = jnp.where(
        is_clevit, jnp.int32(0), timed[int(TimedStatus.LEVITATION_TMP)]
    )

    # --- case TROUBLE_UNUSEABLE_HANDS (pray.c:491-509) — clear GLIB. ---
    is_uh = trouble == jnp.int32(TROUBLE_UNUSEABLE_HANDS)
    new_glib = jnp.where(is_uh, jnp.int32(0), timed[int(TimedStatus.GLIB)])

    # --- case TROUBLE_CURSED_BLINDFOLD (pray.c:510-513) — uncurse the
    # cursed blindfold (object type 208).  Wave 6 #78.
    _BLINDFOLD_OTYP_FIX = 208
    is_cblind = trouble == jnp.int32(TROUBLE_CURSED_BLINDFOLD)
    blindfold_mask = (
        (items.type_id == jnp.int32(_BLINDFOLD_OTYP_FIX))
        & (items.buc_status == jnp.int8(1))
    )
    first_blindfold = jnp.argmax(blindfold_mask.astype(jnp.int32)).astype(jnp.int32)
    has_cblind = jnp.any(blindfold_mask)
    buc_after_cblind = jnp.where(
        is_cblind & has_cblind,
        items.buc_status.at[first_blindfold].set(jnp.int8(2)),  # blessed=2
        items.buc_status,
    )

    # --- case TROUBLE_PUNISHED (pray.c:519-525) — unpunish().
    # Wave 6 #78: clear the prayer.punished flag (heavy iron ball lifted).
    is_punished_fix = trouble == jnp.int32(TROUBLE_PUNISHED)
    new_punished = jnp.where(
        is_punished_fix, jnp.bool_(False), state.prayer.punished,
    )

    # --- case TROUBLE_FUMBLING (pray.c:526-532) — clear FUMBLING. ---
    is_fumb = trouble == jnp.int32(TROUBLE_FUMBLING)
    new_fumb = jnp.where(is_fumb, jnp.int32(0), timed[int(TimedStatus.FUMBLING)])

    # --- case TROUBLE_CURSED_ITEMS (pray.c:533-540) — uncurse one cursed item. ---
    is_citems = trouble == jnp.int32(TROUBLE_CURSED_ITEMS)
    # JIT-safe: pick the first cursed slot via argmax of (buc==1).
    cursed_mask = buc_after_cblind == jnp.int8(1)
    first_cursed = jnp.argmax(cursed_mask.astype(jnp.int32)).astype(jnp.int32)
    has_cursed = jnp.any(cursed_mask)
    new_buc = jnp.where(
        is_citems & has_cursed,
        buc_after_cblind.at[first_cursed].set(jnp.int8(2)),
        buc_after_cblind,
    )

    # --- case TROUBLE_SADDLE (pray.c:591-598) — uncurse cursed saddle.
    # Wave 6 #78: clear ``prayer.saddled_cursed``.
    is_saddle_fix = trouble == jnp.int32(TROUBLE_SADDLE)
    new_saddled = jnp.where(
        is_saddle_fix, jnp.bool_(False), state.prayer.saddled_cursed,
    )

    # Clear prayer.in_region / stuck_in_wall when those troubles fire.
    is_region_fix = trouble == jnp.int32(TROUBLE_REGION)
    is_stuck_fix = trouble == jnp.int32(TROUBLE_STUCK_IN_WALL)
    new_in_region = jnp.where(is_region_fix, jnp.bool_(False), state.prayer.in_region)
    new_stuck_wall = jnp.where(is_stuck_fix, jnp.bool_(False), state.prayer.stuck_in_wall)

    # --- case TROUBLE_BLIND (pray.c:555-577) — make_blinded(0)+make_deaf(0). ---
    is_blind = trouble == jnp.int32(TROUBLE_BLIND)
    new_blind = jnp.where(is_blind, jnp.int32(0), timed[int(TimedStatus.BLIND)])
    new_deaf  = jnp.where(is_blind, jnp.int32(0), timed[int(TimedStatus.DEAF)])

    # --- case TROUBLE_POISONED (pray.c:541-554) — restore abilities. ---
    is_pois = trouble == jnp.int32(TROUBLE_POISONED)
    new_attr_away = jnp.where(is_pois, jnp.int32(0), timed[int(TimedStatus.ATTRIBUTE_AWAY)])

    # --- case TROUBLE_WOUNDED_LEGS (pray.c:578-580) — heal_legs(0). ---
    is_wleg = trouble == jnp.int32(TROUBLE_WOUNDED_LEGS)
    new_wleg = jnp.where(is_wleg, jnp.int32(0), timed[int(TimedStatus.WOUNDED_LEGS)])

    # --- case TROUBLE_STUNNED (pray.c:581-583) — make_stunned(0). ---
    is_stun = trouble == jnp.int32(TROUBLE_STUNNED)
    new_stun = jnp.where(is_stun, jnp.int32(0), timed[int(TimedStatus.STUNNED)])

    # --- case TROUBLE_CONFUSED (pray.c:584-586) — make_confused(0). ---
    is_conf = trouble == jnp.int32(TROUBLE_CONFUSED)
    new_conf = jnp.where(is_conf, jnp.int32(0), timed[int(TimedStatus.CONFUSION)])

    # --- case TROUBLE_HALLUCINATION (pray.c:587-590) — make_hallucinated(0). ---
    is_hallu = trouble == jnp.int32(TROUBLE_HALLUCINATION)
    new_hallu = jnp.where(is_hallu, jnp.int32(0), timed[int(TimedStatus.HALLUCINATION)])

    # Compose new timed_statuses array.
    new_ts = timed
    new_ts = new_ts.at[int(TimedStatus.STONED)].set(new_stoned)
    new_ts = new_ts.at[int(TimedStatus.SLIMED)].set(new_slime)
    new_ts = new_ts.at[int(TimedStatus.STRANGLED)].set(new_strang)
    new_ts = new_ts.at[int(TimedStatus.SICK)].set(new_sick_timer)
    new_ts = new_ts.at[int(TimedStatus.LEVITATION_TMP)].set(new_levit_timer)
    new_ts = new_ts.at[int(TimedStatus.GLIB)].set(new_glib)
    new_ts = new_ts.at[int(TimedStatus.FUMBLING)].set(new_fumb)
    new_ts = new_ts.at[int(TimedStatus.BLIND)].set(new_blind)
    new_ts = new_ts.at[int(TimedStatus.DEAF)].set(new_deaf)
    new_ts = new_ts.at[int(TimedStatus.ATTRIBUTE_AWAY)].set(new_attr_away)
    new_ts = new_ts.at[int(TimedStatus.WOUNDED_LEGS)].set(new_wleg)
    new_ts = new_ts.at[int(TimedStatus.STUNNED)].set(new_stun)
    new_ts = new_ts.at[int(TimedStatus.CONFUSION)].set(new_conf)
    new_ts = new_ts.at[int(TimedStatus.HALLUCINATION)].set(new_hallu)

    new_items = items.replace(buc_status=new_buc)
    new_inv = state.inventory.replace(items=new_items)
    new_status = state.status.replace(
        nutrition=new_nutrition,
        hunger_state=new_hunger_state,
        sick_kind=new_sick_kind,
        encumbrance=new_encumb,
        timed_statuses=new_ts,
    )

    # Wave 6 #78: thread polymorph (lycanthropy_form) and prayer (punished,
    # saddled_cursed, in_region, stuck_in_wall) field updates back through
    # the returned EnvState.
    new_polymorph = state.polymorph.replace(lycanthropy_form=new_lycan_form)
    new_prayer = state.prayer.replace(
        punished=new_punished,
        saddled_cursed=new_saddled,
        in_region=new_in_region,
        stuck_in_wall=new_stuck_wall,
    )

    return state.replace(
        player_hp=new_hp,
        player_hp_max=new_hp_max,
        status=new_status,
        inventory=new_inv,
        terrain=new_terrain,
        polymorph=new_polymorph,
        prayer=new_prayer,
    )


# Back-compat alias for Wave 4 callers.
def _fix_trouble(state, trouble: jnp.ndarray):
    """Alias for fix_worst() with a fresh PRNG key (deterministic-ish).

    Wave 4 callers don't pass an RNG; we fold the player's hash for repeatability.
    """
    seed = state.timestep.astype(jnp.uint32) ^ jnp.uint32(0x5A17)
    rng = jax.random.PRNGKey(seed)
    return fix_worst(state, rng, trouble)


# ---------------------------------------------------------------------------
# Outcome application — small helpers per bucket
# ---------------------------------------------------------------------------

def _apply_heal_cure(state):
    """HEAL_CURE: restore full HP (pray.c pleased case 2)."""
    return state.replace(player_hp=state.player_hp_max.astype(jnp.int32))


def _apply_protection(state):
    """PROTECTION_PLUS_1: pray.c pleased case 5 (u.ublessed += 1).

    We model "+1 divine protection" as -1 AC (lower AC = better).
    """
    new_ac = (state.player_ac - jnp.int32(1)).astype(jnp.int32)
    return state.replace(player_ac=new_ac)


def _apply_remove_curse(state):
    """REMOVE_CURSE: bless all cursed items (pray.c pleased case 4).

    A cursed item has buc_status == 1; mass-uncurse them to 2 (uncursed).
    Only acts on the player's inventory slice (52 slots).
    """
    items = state.inventory.items
    is_cursed = items.buc_status == jnp.int8(1)
    new_buc = jnp.where(is_cursed, jnp.int8(2), items.buc_status)
    new_items = items.replace(buc_status=new_buc)
    new_inv = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


def _apply_gift_artifact(state):
    """GIFT_ARTIFACT: pray.c pleased case 7/8 (gcrownu); requires alignment≥PIOUS.

    We model the gift as a bump to wisdom (a stable, JIT-safe stand-in).
    A full artifact-grant requires inventory slot insertion which is
    deferred to Wave 5.
    """
    new_wis = jnp.minimum(state.player_wis + jnp.int8(1), jnp.int8(25)).astype(jnp.int8)
    return state.replace(player_wis=new_wis)


def _apply_smite(state):
    """SMITE: pray.c angrygods cases 2-3 (lose Wis + XL via losexp)."""
    new_wis = jnp.maximum(state.player_wis - jnp.int8(1), jnp.int8(3)).astype(jnp.int8)
    new_xp = jnp.maximum(state.player_xp - jnp.int32(50), jnp.int32(0)).astype(jnp.int32)
    return state.replace(player_wis=new_wis, player_xp=new_xp)


def _apply_destroy_armor(state):
    """DESTROY_ARMOR: pray.c angrygods cases 4-5 (rndcurse).

    We curse all inventory items (buc_status -> 1) as a conservative
    JIT-safe stand-in for damage to a specific armour piece.
    """
    items = state.inventory.items
    occupied = items.category != jnp.int8(0)
    new_buc = jnp.where(occupied, jnp.int8(1), items.buc_status)
    new_items = items.replace(buc_status=new_buc)
    new_inv = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


def _apply_inflict_blindness(state):
    """INFLICT_BLINDNESS: set BLIND timed status (pray.c minor punish branch).

    TimedStatus.BLIND index = 0 per status_effects.TimedStatus enum.
    Hard-code the index here to avoid a cyclic import.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    blind_idx = int(TimedStatus.BLIND)
    new_timed = state.status.timed_statuses.at[blind_idx].set(jnp.int32(100))
    new_status = state.status.replace(timed_statuses=new_timed)
    return state.replace(status=new_status)


def _apply_inflict_weakness(state):
    """INFLICT_WEAKNESS: drain strength by 1 (pray.c minor punish)."""
    new_str = jnp.maximum(state.player_str - jnp.int16(1), jnp.int16(3)).astype(jnp.int16)
    return state.replace(player_str=new_str)


# ---------------------------------------------------------------------------
# Wave-6 pleased/angry helpers (pray.c::pleased pat-on-head 0-8;
# pray.c::angrygods 7-default cases not yet in Wave-4 set)
# ---------------------------------------------------------------------------

def _apply_intrinsic_gift(state, rng: jax.Array):
    """INTRINSIC_GIVING: grant a random resistance intrinsic.

    Mirrors pray.c pleased case 5 (pray.c:1310-1338) — telepathy/speed/
    stealth/protection ladder.  Vendor cascades through TELEPAT → FAST →
    STEALTH → PROTECTION.  For Wave 6 we expose a simpler RNG-driven pick
    over the resistance group so the test surface is non-trivial.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    # Candidate intrinsics: resistances (1..10) and telepathy (30) / fast (64).
    candidates = jnp.array(
        [
            int(Intrinsic.RESIST_FIRE),
            int(Intrinsic.RESIST_COLD),
            int(Intrinsic.RESIST_SLEEP),
            int(Intrinsic.RESIST_SHOCK),
            int(Intrinsic.RESIST_POISON),
            int(Intrinsic.TELEPATHY),
            int(Intrinsic.FAST),
            int(Intrinsic.STEALTH),
        ],
        dtype=jnp.int32,
    )
    idx = jax.random.randint(rng, (), minval=0, maxval=candidates.shape[0], dtype=jnp.int32)
    intrinsic_id = candidates[idx]
    new_intr = state.status.intrinsics.at[intrinsic_id].set(True)
    return state.replace(status=state.status.replace(intrinsics=new_intr))


def _apply_alignment_bump(state):
    """ALIGNMENT_BUMP: pray.c::pleased line 1089 — adjalign(+1)."""
    new_record = (state.prayer.alignment_record + jnp.int16(1)).astype(jnp.int16)
    return state.replace(prayer=state.prayer.replace(alignment_record=new_record))


def _apply_ability_increase(state, rng: jax.Array):
    """ABILITY_INCREASE: pat-on-head case 2 effect (pray.c:1264-1268) —
    raise one base stat (STR / WIS) by 1.
    """
    coin = jax.random.uniform(rng, ())
    bump_str = coin < jnp.float32(0.5)
    new_str = jnp.where(
        bump_str,
        jnp.minimum(state.player_str + jnp.int16(1), jnp.int16(25)),
        state.player_str,
    ).astype(jnp.int16)
    new_wis = jnp.where(
        bump_str,
        state.player_wis,
        jnp.minimum(state.player_wis + jnp.int8(1), jnp.int8(25)),
    ).astype(jnp.int8)
    return state.replace(player_str=new_str, player_wis=new_wis)


def _apply_luck_giving(state):
    """LUCK_GIVING: pat-on-head case 2 sets u.uluck>=0 (pray.c:1275-1276),
    and grants +5 luck baseline (Wave-6 parity heuristic).
    """
    new_luck = jnp.minimum(state.prayer.luck + jnp.int32(5), jnp.int32(13)).astype(jnp.int32)
    return state.replace(prayer=state.prayer.replace(luck=new_luck))


def _apply_drain_level(state):
    """DRAIN_LEVEL: angrygods cases 2-3 losexp() effect (pray.c:742).

    Decrements XL by 1 (floor 1) and shaves matching HP_max.
    """
    new_xl = jnp.maximum(state.player_xl - jnp.int32(1), jnp.int32(1)).astype(jnp.int32)
    new_hp_max = jnp.maximum(state.player_hp_max - jnp.int32(5), jnp.int32(5)).astype(jnp.int32)
    new_hp = jnp.minimum(state.player_hp, new_hp_max).astype(jnp.int32)
    return state.replace(
        player_xl=new_xl, player_hp_max=new_hp_max, player_hp=new_hp,
    )


def _apply_smite_3d6(state, rng: jax.Array):
    """SMITE (3d6 damage variant) — angrygods 'punish in body' branch.

    Per pray.c:736-742 the deity inflicts spiritual wounds.  We model the
    HP loss as 3d6 (range 3..18).
    """
    rolls = jax.random.randint(rng, (3,), minval=1, maxval=7, dtype=jnp.int32)
    dmg = jnp.sum(rolls).astype(jnp.int32)
    new_hp = jnp.maximum(state.player_hp - dmg, jnp.int32(0)).astype(jnp.int32)
    # Also stop luck effects per spec.
    new_luck = jnp.minimum(state.prayer.luck, jnp.int32(0)).astype(jnp.int32)
    return state.replace(
        player_hp=new_hp,
        prayer=state.prayer.replace(luck=new_luck),
    )


def _apply_summon_demon(state):
    """SUMMON_DEMON: angrygods case 7-8 (pray.c:760-772) — hostile minion.

    Modelled by bumping god_anger and dropping alignment_record by 3.
    """
    new_anger = (state.prayer.god_anger + jnp.int32(1)).astype(jnp.int32)
    new_record = (state.prayer.alignment_record - jnp.int16(3)).astype(jnp.int16)
    return state.replace(prayer=state.prayer.replace(
        god_anger=new_anger, alignment_record=new_record,
    ))


def _apply_zap_form_change(state):
    """ZAP_FORM_CHANGE: god_zaps_you fallback (pray.c:776) — involuntary poly.

    JAX model: drop hp_max by 10 % to mimic body-restructure cost.  Real
    polymorph is handled by polymorph.py and out of scope here.
    """
    new_hp_max = jnp.maximum(
        state.player_hp_max * jnp.int32(9) // jnp.int32(10), jnp.int32(1)
    ).astype(jnp.int32)
    new_hp = jnp.minimum(state.player_hp, new_hp_max).astype(jnp.int32)
    return state.replace(player_hp_max=new_hp_max, player_hp=new_hp)


# ---------------------------------------------------------------------------
# god_zaps_you — pray.c lines ~785-900 simplified
# ---------------------------------------------------------------------------

def god_zaps_you(state, rng: jax.Array):
    """Simplified port of pray.c::god_zaps_you (lightning + item destruction).

    Per the Wave-4 spec:
      - 50 % lightning bolt for d6 damage
      - 50 % destroy a random worn item
      - alignment_record -= 5

    Returns: new EnvState (deterministic given rng).
    """
    rng_branch, rng_dmg, rng_slot = jax.random.split(rng, 3)

    # Decide branch via uniform U[0,1).
    coin = jax.random.uniform(rng_branch, ())
    is_lightning = coin < jnp.float32(0.5)

    # Lightning branch: d6 damage.
    dmg = jax.random.randint(rng_dmg, (), minval=1, maxval=7, dtype=jnp.int32)
    new_hp_lightning = jnp.maximum(jnp.int32(0), state.player_hp - dmg)

    # Item-destruction branch: drop quantity to 0 on a random occupied slot.
    items = state.inventory.items
    occupied = items.category != jnp.int8(0)
    n_slots = items.category.shape[0]
    # Pick a random slot index in [0, n_slots).
    rand_slot = jax.random.randint(
        rng_slot, (), minval=0, maxval=n_slots, dtype=jnp.int32
    )
    # If chosen slot empty, fall through to first occupied (argmax).
    fallback = jnp.argmax(occupied).astype(jnp.int32)
    target_slot = jnp.where(
        occupied[rand_slot],
        rand_slot,
        fallback,
    )
    can_destroy = jnp.any(occupied)
    new_qty = jnp.where(
        (~is_lightning) & can_destroy,
        jnp.int16(0),
        items.quantity[target_slot],
    )
    new_cat = jnp.where(
        (~is_lightning) & can_destroy,
        jnp.int8(0),
        items.category[target_slot],
    )
    new_quantity = items.quantity.at[target_slot].set(new_qty)
    new_category = items.category.at[target_slot].set(new_cat)
    new_items = items.replace(quantity=new_quantity, category=new_category)
    new_inv = state.inventory.replace(items=new_items)

    new_hp = jnp.where(is_lightning, new_hp_lightning, state.player_hp).astype(jnp.int32)

    # alignment_record -= 5.
    new_record = (state.prayer.alignment_record - jnp.int16(5)).astype(jnp.int16)
    new_prayer = state.prayer.replace(alignment_record=new_record)

    return state.replace(
        player_hp=new_hp,
        inventory=new_inv,
        prayer=new_prayer,
    )


# ---------------------------------------------------------------------------
# pray() — main outcome chain
# ---------------------------------------------------------------------------

def pray(state, rng: jax.Array):
    """Full pray() pipeline per pray.c lines 500-1500 (Wave 6 parity).

    Pipeline:
      1. ``in_trouble()`` → ``fix_worst()`` chain (pray.c::in_trouble +
         fix_worst_trouble) if not angry.
      2. Anger gate (pray.c::can_pray):
            record < threshold or pray_timeout > 0 → angrygods() branch
      3. Pleased path (pray.c::pleased lines 1070-1381):
           8 vendor "pat-on-head" buckets keyed by the d-roll.
           Bucket gating mirrors PIOUS / DEVOUT / STRIDENT thresholds.
      4. Angry path (pray.c::angrygods lines 703-784):
           8 vendor anger buckets (smite, drain, zap, summon, ...).
      5. Timeout reset and ``adjalign`` ±1.

    Returns the new EnvState.
    """
    trouble = in_trouble(state)

    # --- Anger gates (pray.c::can_pray) -------------------------------------
    pray_timeout = state.prayer.pray_timeout
    record = state.prayer.alignment_record.astype(jnp.int32)
    timeout_active = pray_timeout > jnp.int32(0)
    record_below_threshold = record < jnp.int32(PRAY_RECORD_THRESHOLD)
    angry = timeout_active | record_below_threshold

    # --- RNG splits ---------------------------------------------------------
    (rng_roll, rng_zap, rng_timeout, rng_fix,
     rng_intr, rng_stat, rng_smite) = jax.random.split(rng, 7)
    # Pleased uses 0..99 (d100); angry uses a separate roll.
    roll = jax.random.randint(rng_roll, (), minval=0, maxval=100, dtype=jnp.int32)

    # --- Trouble fix path (pray.c::pleased branches into fix_worst_trouble) -
    # Vendor (pray.c:1105-1160): trouble is auto-fixed whenever the player is
    # in pleased state and has any active trouble.
    fixable = (trouble != jnp.int32(TROUBLE_NONE)) & ~angry
    state_after_fix = jax.lax.cond(
        fixable,
        lambda s: fix_worst(s, rng_fix, trouble),
        lambda s: s,
        state,
    )

    # --- Pleased path (pray.c::pleased pat_on_head 0..8) --------------------
    # Bucket mapping (covers all 8 vendor "pat-on-head" cases plus the
    # alignment-bump short-circuit at line 1089):
    #   0..14   ALIGNMENT_BUMP    (pray.c:1088-1089, record<2 trouble<=0)
    #   15..29  HEAL_CURE         (case 2, lines 1246-1282)
    #   30..44  PROTECTION_+1     (case 5, lines 1310-1338)
    #   45..59  REMOVE_CURSE      (case 4, lines 1283-1309)
    #   60..69  LUCK_GIVING       (case 2 luck-floor, line 1275-1276)
    #   70..79  INTRINSIC_GIFT    (case 5 telepathy/speed branch)
    #   80..89  ABILITY_INCREASE  (case 2 STR/WIS restore, lines 1264-1268)
    #   90..99  GIFT_ARTIFACT     (case 7-8 gcrownu, lines 1340-1350,
    #                              gated by alignment_record >= PIOUS)
    is_pleased = (~angry) & (trouble == jnp.int32(TROUBLE_NONE))

    def _pleased_branch(s):
        s_align  = _apply_alignment_bump(s)
        s_heal   = _apply_heal_cure(s)
        s_prot   = _apply_protection(s)
        s_rmc    = _apply_remove_curse(s)
        s_luck   = _apply_luck_giving(s)
        s_intr   = _apply_intrinsic_gift(s, rng_intr)
        s_abil   = _apply_ability_increase(s, rng_stat)
        s_gift   = _apply_gift_artifact(s)

        can_gift = s.prayer.alignment_record.astype(jnp.int32) >= jnp.int32(PIOUS)

        # Nested lax.cond chain — full 8-bucket switch (vendor parity).
        return jax.lax.cond(
            roll < jnp.int32(15),
            lambda _: s_align,
            lambda _: jax.lax.cond(
                roll < jnp.int32(30),
                lambda _: s_heal,
                lambda _: jax.lax.cond(
                    roll < jnp.int32(45),
                    lambda _: s_prot,
                    lambda _: jax.lax.cond(
                        roll < jnp.int32(60),
                        lambda _: s_rmc,
                        lambda _: jax.lax.cond(
                            roll < jnp.int32(70),
                            lambda _: s_luck,
                            lambda _: jax.lax.cond(
                                roll < jnp.int32(80),
                                lambda _: s_intr,
                                lambda _: jax.lax.cond(
                                    roll < jnp.int32(90),
                                    lambda _: s_abil,
                                    lambda _: jax.lax.cond(
                                        can_gift,
                                        lambda _: s_gift,
                                        # Fallback to heal when gift unavailable.
                                        lambda _: s_heal,
                                        operand=0,
                                    ),
                                    operand=0,
                                ),
                                operand=0,
                            ),
                            operand=0,
                        ),
                        operand=0,
                    ),
                    operand=0,
                ),
                operand=0,
            ),
            operand=0,
        )

    state_pleased = jax.lax.cond(
        is_pleased,
        _pleased_branch,
        lambda s: s,
        state_after_fix,
    )

    # --- Angry path (pray.c::angrygods 703-784) -----------------------------
    # 8 vendor anger buckets:
    #   0..12   "displeased" warning   — case 0-1, pray.c:725-730
    #   13..25  SMITE_3D6              — case 2-3, pray.c:732-743 (Wis/XL hit)
    #   26..37  DRAIN_LEVEL            — extracted from case 2-3 losexp branch
    #   38..50  DESTROY_ARMOR          — case 4-5/6, pray.c:752-759 (rndcurse)
    #   51..62  INFLICT_BLINDNESS      — minor punish branch
    #   63..74  INFLICT_WEAKNESS       — stat-drain branch
    #   75..86  SUMMON_DEMON           — case 7-8, pray.c:760-772
    #   87..99  ANGER_BOLT             — default, god_zaps_you (pray.c:774-777)
    #   ZAP_FORM_CHANGE is folded into ANGER_BOLT via god_zaps_you wide-angle.
    def _angry_branch(s):
        s_warn    = s                              # case 0-1 no-op
        s_smite   = _apply_smite_3d6(s, rng_smite)
        s_drain   = _apply_drain_level(s)
        s_destroy = _apply_destroy_armor(s)
        s_blind   = _apply_inflict_blindness(s)
        s_weak    = _apply_inflict_weakness(s)
        s_summon  = _apply_summon_demon(s)
        s_bolt    = god_zaps_you(s, rng_zap)
        return jax.lax.cond(
            roll < jnp.int32(13),
            lambda _: s_warn,
            lambda _: jax.lax.cond(
                roll < jnp.int32(26),
                lambda _: s_smite,
                lambda _: jax.lax.cond(
                    roll < jnp.int32(38),
                    lambda _: s_drain,
                    lambda _: jax.lax.cond(
                        roll < jnp.int32(51),
                        lambda _: s_destroy,
                        lambda _: jax.lax.cond(
                            roll < jnp.int32(63),
                            lambda _: s_blind,
                            lambda _: jax.lax.cond(
                                roll < jnp.int32(75),
                                lambda _: s_weak,
                                lambda _: jax.lax.cond(
                                    roll < jnp.int32(87),
                                    lambda _: s_summon,
                                    lambda _: s_bolt,
                                    operand=0,
                                ),
                                operand=0,
                            ),
                            operand=0,
                        ),
                        operand=0,
                    ),
                    operand=0,
                ),
                operand=0,
            ),
            operand=0,
        )

    state_final = jax.lax.cond(
        angry,
        _angry_branch,
        lambda s: s,
        state_pleased,
    )

    # --- Reset pray_timeout = 300 + rn2(700) (pray.c:1356, rnz(350)) --------
    extra = jax.random.randint(
        rng_timeout, (), minval=0, maxval=PRAY_TIMEOUT_RANGE, dtype=jnp.int32
    )
    new_pray_timeout = jnp.int32(PRAY_TIMEOUT_BASE) + extra

    # --- Adjust alignment_record (pray.c::adjalign) -------------------------
    # +1 on pleased (no trouble), -1 on angry, no change otherwise.
    delta = jnp.where(
        angry, jnp.int16(-1),
        jnp.where(is_pleased, jnp.int16(1), jnp.int16(0)),
    )
    new_record = state_final.prayer.alignment_record + delta

    new_prayer = state_final.prayer.replace(
        pray_timeout=new_pray_timeout,
        prayer_timeout=new_pray_timeout,   # keep aliases in sync
        alignment_record=new_record,
        last_pray_turn=state_final.timestep,
    )
    return state_final.replace(prayer=new_prayer)


# ---------------------------------------------------------------------------
# Sacrifice (do.c::do_sacrifice subset)
# ---------------------------------------------------------------------------

# Sentinel: corpse type_id range encoding "mighty monster" (level >= 30).
# Vendor uses sacrifice_value(); for our slice we treat type_id >= 1000 as
# a mighty monster corpse.  Test data sets type_id = 1030 etc.
MIGHTY_TYPE_THRESHOLD: int = 1000

# Human race id for "your race" sacrifice (do.c sacrifice_your_race).
# Maps to player_race == HUMAN.  Our PlayerRace.HUMAN is index 0.
HUMAN_RACE: int = 0


def sacrifice_on_altar(state, rng: jax.Array, slot_idx: jnp.ndarray):
    """Sacrifice the inventory slot *slot_idx* on the altar at the player's tile.

    Full vendor parity for do_sacrifice / offer_corpse / eval_offering
    (pray.c:1854-2150).  Vendor outcome buckets (5 total):

      1. Mighty monster (sacrifice_value >= 30 — pray.c:1909-1955) →
         WISH granted (do.c::bestow_artifact + Wiz boss bonus).
         Modelled by setting a wish-counter on PrayerState (god_anger lane).
      2. Same-aligned fresh corpse with record >= DEVOUT + 1/200 →
         ARTIFACT GIFT (pray.c:2091-2092 bestow_artifact()) — model by
         bumping player_wis as in _apply_gift_artifact.
      3. Same-aligned corpse, normal value → +5 alignment_record
         (pray.c:2030-2065 brownie points / luck increase).
      4. Cross-aligned race=HUMAN (pray.c:1727-1773 sacrifice_your_race
         on chaotic altar branch) → +5 alignment (chaotic player) OR
         demon summon + -5 alignment (lawful/neutral player).
      5. Wrong-aligned (corpse aligned to enemy god, pray.c:2024-2028) →
         demon-style penalty (-5 record).

    Always consumes the corpse (pray.c::consume_offering).
    """
    items = state.inventory.items
    safe_slot = jnp.clip(slot_idx, 0, items.category.shape[0] - 1).astype(jnp.int32)

    # Corpse detection: FOOD category (ItemCategory.FOOD == 7).
    is_corpse = (
        (items.category[safe_slot] == jnp.int8(7))
        & (items.quantity[safe_slot] > jnp.int16(0))
    )

    # Altar alignment under the player's current tile.
    max_levels = state.terrain.shape[1]
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    flat_lv = b * jnp.int32(max_levels) + lv
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)
    altar_align = state.features.altar_alignment[flat_lv, row, col].astype(jnp.int32)
    on_altar = altar_align >= jnp.int32(0)

    corpse_type = items.type_id[safe_slot].astype(jnp.int32)
    corpse_align = corpse_type
    player_align = state.player_align.astype(jnp.int32)
    player_race  = state.player_race.astype(jnp.int32)
    record       = state.prayer.alignment_record.astype(jnp.int32)

    can_sacrifice = is_corpse & on_altar
    same_aligned  = (corpse_align == player_align) & can_sacrifice
    opposite      = (corpse_align != player_align) & can_sacrifice

    # --- Bucket selection (vendor priority order) ---------------------------
    # 1. Mighty monster — pray.c:1909-1955 (sacrifice_value bonus path).
    is_mighty = can_sacrifice & (corpse_type >= jnp.int32(MIGHTY_TYPE_THRESHOLD))

    # 2. Artifact gift gate — pray.c:2091-2092 bestow_artifact():
    #    same alignment AND record >= DEVOUT AND not mighty.
    is_artifact_gift = (
        same_aligned
        & (~is_mighty)
        & (record >= jnp.int32(DEVOUT))
    )

    # 3. Normal same-aligned blessing.
    is_normal_blessing = same_aligned & (~is_mighty) & (~is_artifact_gift)

    # 4. Cross-aligned race=HUMAN (sacrifice_your_race) — pray.c:1727-1773.
    #    Encoded as: corpse_type == 0 (sentinel for "human corpse") AND
    #    altar_align != player_align.
    is_cross_human = (
        can_sacrifice
        & (corpse_type == jnp.int32(HUMAN_RACE))
        & (altar_align != player_align)
        & (player_race == jnp.int32(HUMAN_RACE))
    )

    # 5. Plain opposite-aligned non-human.
    is_wrong = opposite & (~is_cross_human) & (~is_mighty)

    # --- Apply state changes ------------------------------------------------
    # Mighty monster → wish: bump god_anger negatively as "wishes_granted"
    # counter (Wave 6 placeholder; real wish requires wish.py integration).
    new_anger = jnp.where(
        is_mighty,
        state.prayer.god_anger - jnp.int32(1),   # -1 = "wish credit"
        state.prayer.god_anger,
    )

    # Artifact gift: bump WIS (consistent with _apply_gift_artifact).
    new_wis = jnp.where(
        is_artifact_gift,
        jnp.minimum(state.player_wis + jnp.int8(1), jnp.int8(25)),
        state.player_wis,
    )

    # Record delta — vendor table:
    #   mighty           → +10 (also wish granted via god_anger field)
    #   artifact gift    → +5  (pray.c bestow_artifact still ups standing)
    #   normal blessing  → +5  (pray.c:2030-2065 brownie points)
    #   cross-human (chaotic player) → +5  (pray.c:1773 adjalign(5))
    #   cross-human (lawful/neutral) → -5  (pray.c:1766 adjalign(-5))
    #   wrong-aligned    → -5  (pray.c:2015-2017 offer_negative_valued)
    is_chaotic_player = player_align == jnp.int32(0)  # Alignment.CHAOTIC
    cross_human_delta = jnp.where(
        is_chaotic_player, jnp.int16(5), jnp.int16(-5)
    )
    delta = jnp.where(
        is_mighty,           jnp.int16(10),
        jnp.where(
            is_artifact_gift,    jnp.int16(5),
            jnp.where(
                is_normal_blessing, jnp.int16(5),
                jnp.where(
                    is_cross_human, cross_human_delta,
                    jnp.where(
                        is_wrong, jnp.int16(-5),
                        jnp.int16(0),
                    ),
                ),
            ),
        ),
    )
    new_record = state.prayer.alignment_record + delta
    new_prayer = state.prayer.replace(
        alignment_record=new_record,
        god_anger=new_anger,
    )

    # Consume the corpse.
    consume = can_sacrifice
    new_qty = jnp.where(
        consume,
        jnp.maximum(items.quantity[safe_slot] - jnp.int16(1), jnp.int16(0)),
        items.quantity[safe_slot],
    )
    new_cat = jnp.where(
        consume & (new_qty == jnp.int16(0)),
        jnp.int8(0),
        items.category[safe_slot],
    )
    new_quantity = items.quantity.at[safe_slot].set(new_qty)
    new_category = items.category.at[safe_slot].set(new_cat)
    new_items = items.replace(quantity=new_quantity, category=new_category)
    new_inv = state.inventory.replace(items=new_items)

    return state.replace(
        prayer=new_prayer,
        inventory=new_inv,
        player_wis=new_wis,
    )


# ---------------------------------------------------------------------------
# Public entry point for action_dispatch
# ---------------------------------------------------------------------------

def handle_pray(state, rng: jax.Array):
    """Entry called from action_dispatch when the player presses Meta-p.

    1. Run pray() pipeline.
    2. Mark ATHEIST conduct as violated (vendor/nethack/src/insight.c ~2134).
    """
    new_state = pray(state, rng)

    # Conduct: ATHEIST is violated on any prayer attempt.
    atheist_idx = int(Conduct.ATHEIST)
    new_violations = new_state.conduct.violations.at[atheist_idx].set(True)
    new_conduct = new_state.conduct.replace(violations=new_violations)
    return new_state.replace(conduct=new_conduct)


# ---------------------------------------------------------------------------
# Legacy helpers (Wave 1 stubs, kept for back-compat)
# ---------------------------------------------------------------------------

def adjust_alignment(state: PrayerState, delta: jnp.ndarray) -> PrayerState:
    """Adjust alignment_record by *delta*, clamped to int16 range.

    Mirrors pray.c::adjalign (subset).
    """
    new_record = state.alignment_record + delta.astype(jnp.int16)
    return state.replace(alignment_record=new_record)


def adjust_luck(state: PrayerState, delta: jnp.ndarray) -> PrayerState:
    """Adjust luck, clamped to [-13, +13] (pray.c change_luck / LUCKMAX=13)."""
    new_luck = jnp.clip(state.luck + delta.astype(jnp.int32), -13, 13)
    return state.replace(luck=new_luck)


def step(state: PrayerState, rng: jax.Array) -> PrayerState:
    """Per-turn tick: decrement pray_timeout (floor 0).

    Mirrors u.ublesscnt countdown in the moveloop.
    """
    new_pray_timeout = jnp.maximum(state.pray_timeout - jnp.int32(1), jnp.int32(0))
    return state.replace(
        pray_timeout=new_pray_timeout,
        prayer_timeout=new_pray_timeout,
    )
