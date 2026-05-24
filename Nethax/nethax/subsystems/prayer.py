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
from typing import Optional

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


# ---------------------------------------------------------------------------
# Race enum — mirrors vendor/nethack/src/role.c races[] order
# ---------------------------------------------------------------------------
class Race(IntEnum):
    HUMAN  = 0
    ELF    = 1
    DWARF  = 2
    GNOME  = 3
    ORC    = 4


# ---------------------------------------------------------------------------
# God-name table — role.c::races[] god triples (lawful, neutral, chaotic)
# per race.  pray.c:50 note: "prayers are made to your god".
# Cited: pray.c:50, role.c races[].
# ---------------------------------------------------------------------------
GOD_OWN      = 0   # own alignment slot
GOD_NEUTRAL  = 1   # neutral slot (middle)
GOD_OPPOSITE = 2   # opposite alignment slot

# (lawful_god, neutral_god, chaotic_god) per Race.
# Source: vendor/nethack/src/role.c (race-level god assignments).
# Human uses Babylonian pantheon (role.c:123), Elf uses Elven pantheon
# (role.c:371), Dwarf uses Dwarvish, Gnome uses Gnomish, Orc uses Orcish.
_GODS: dict = {
    Race.HUMAN: ("Anu",                   "Ishtar",   "Anshar"),
    Race.ELF:   ("Solonor Thelandira",    "Aerdrie Faenya", "Lolth"),
    Race.DWARF: ("Moradin",               "Dumathoin", "Abbathor"),
    Race.GNOME: ("Garl Glittergold",      "Baervan Wildwanderer", "Urdlen"),
    Race.ORC:   ("Gruumsh",               "Ilneval",   "Shargaas"),
}

# Alignment → index into the god triple.
_ALIGN_TO_GOD_IDX: dict = {
    Alignment.LAWFUL:   0,
    Alignment.NEUTRAL:  1,
    Alignment.CHAOTIC:  2,
}


def god_name(race: "Race", align: "Alignment", kind: int) -> str:
    """Return the god name for *race*/*align* at position *kind*.

    *kind* is GOD_OWN (own-alignment slot), GOD_NEUTRAL, or GOD_OPPOSITE.
    For GOD_OWN the returned name is the deity matching *align*.
    For GOD_NEUTRAL always the neutral (middle) entry.
    For GOD_OPPOSITE the entry opposite to *align*.

    Cited: pray.c:50.
    """
    triple = _GODS[Race(race)]
    if kind == GOD_OWN:
        idx = _ALIGN_TO_GOD_IDX[Alignment(align)]
    elif kind == GOD_NEUTRAL:
        idx = 1
    else:  # GOD_OPPOSITE
        own_idx = _ALIGN_TO_GOD_IDX[Alignment(align)]
        idx = 2 - own_idx
    name = triple[idx]
    # Strip leading "_" prefix used in role.c for female variants.
    return name.lstrip("_")


def god_speaks_message(race: int, align: int, message_kind: str) -> str:
    """Return a flavour string for a divine message.

    *message_kind*: "pleased" / "angry" / "unaware" / "ascension".
    Host-side only (returns a Python str, not a JAX array).
    Cited: pray.c:50.
    """
    god = god_name(Race(race), Alignment(align), GOD_OWN)
    templates = {
        "pleased":    f"{god} is well-pleased.",
        "angry":      f"{god} is displeased!",
        "unaware":    f"{god} is unaware of your deeds.",
        "ascension":  f"{god} raises you to demigod-hood!",
    }
    return templates.get(message_kind, f"{god} speaks.")


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
    # pray.c:228-231  cursed levitation gear:
    #     Cursed_obj(uarmf, LEVITATION_BOOTS)
    #     || stuck_ring(uleft, RIN_LEVITATION)
    #     || stuck_ring(uright, RIN_LEVITATION)
    # Wave 27a: byte-equal — only fires when the specific worn item is the
    # cursed levitation source (boots otyp 149 worn on feet, or ring otyp 160
    # cursed on left/right finger).
    from Nethax.nethax.subsystems.inventory import ArmorSlot
    _LEVIT_BOOTS_OTYP = 149   # objects.py id 149 = levitation boots
    _LEVIT_RING_OTYP  = 160   # objects.py id 160 = ring of levitation
    inv = state.inventory
    boots_idx = inv.worn_armor[int(ArmorSlot.BOOTS)].astype(jnp.int32)
    ringL_idx = inv.worn_rings[0].astype(jnp.int32)
    ringR_idx = inv.worn_rings[1].astype(jnp.int32)
    boots_otyp = jnp.where(
        boots_idx >= jnp.int32(0), items.type_id[boots_idx], jnp.int32(-1)
    )
    ringL_otyp = jnp.where(
        ringL_idx >= jnp.int32(0), items.type_id[ringL_idx], jnp.int32(-1)
    )
    ringR_otyp = jnp.where(
        ringR_idx >= jnp.int32(0), items.type_id[ringR_idx], jnp.int32(-1)
    )
    # Audit G #5 fix: vendor pray.c:232-241 only inspects the buc field of
    # an item that is actually equipped — ``Cursed_obj(uarmf, ...)`` and
    # ``stuck_ring(...)`` return false when the slot is empty without
    # reading a buc value.  The previous ``jnp.int8(0)`` fallback when
    # ``index < 0`` zero-extended uninitialised buc data; in particular it
    # could *unset* a cursed status read elsewhere if downstream code
    # consumed boots_cursed/ringL_cursed/ringR_cursed without the otyp
    # guard.  Switch to a direct boolean: True iff the slot is occupied
    # AND that item is cursed.
    _CURSED = jnp.int8(1)
    boots_cursed = jnp.where(
        boots_idx >= jnp.int32(0),
        items.buc_status[boots_idx] == _CURSED,
        jnp.bool_(False),
    )
    ringL_cursed = jnp.where(
        ringL_idx >= jnp.int32(0),
        items.buc_status[ringL_idx] == _CURSED,
        jnp.bool_(False),
    )
    ringR_cursed = jnp.where(
        ringR_idx >= jnp.int32(0),
        items.buc_status[ringR_idx] == _CURSED,
        jnp.bool_(False),
    )
    is_cursed_levit = (
        ((boots_otyp == jnp.int32(_LEVIT_BOOTS_OTYP)) & boots_cursed)
        | ((ringL_otyp == jnp.int32(_LEVIT_RING_OTYP)) & ringL_cursed)
        | ((ringR_otyp == jnp.int32(_LEVIT_RING_OTYP)) & ringR_cursed)
    )
    # pray.c:232-241  unuseable hands (welded weapon / nohands form).
    is_unuse_hands = timed[int(TimedStatus.GLIB)] > jnp.int32(0)
    # pray.c:242-243  cursed blindfold — ``Blindfolded && ublindf->cursed``.
    # Vendor reads the specific worn-blindfold pointer (``ublindf``); a
    # cursed blindfold in inventory that is *not* worn does not trigger
    # TROUBLE_CURSED_BLINDFOLD.  Audit G #6: InventoryState exposes no
    # ``worn_blindfold`` slot today (the ArmorSlot enum in
    # subsystems/inventory.py only covers BODY/SHIELD/HELM/GLOVES/BOOTS/
    # CLOAK/SHIRT), so a byte-equal vendor check requires per-state
    # tracking we do not yet model.  We deliberately keep the
    # "BLIND timed AND any cursed blindfold (otyp 208) in inventory"
    # approximation — it preserves the AND-of-two-conditions structure of
    # the vendor predicate (a blinded hero with a cursed blindfold at
    # hand) and remains the closest JIT-safe analogue until a worn-
    # blindfold slot lands.  Divergence: the vendor check is false when
    # the cursed blindfold is carried but unworn; we report True in that
    # case.  Cite: vendor/nethack/src/pray.c:242-243.
    _BLINDFOLD_OTYP = 208
    is_blindfold = items.type_id == jnp.int32(_BLINDFOLD_OTYP)
    is_cursed_item = items.buc_status == jnp.int8(1)
    is_blinded = timed[int(TimedStatus.BLIND)] > jnp.int32(0)
    is_cursed_blindfold = is_blinded & jnp.any(is_blindfold & is_cursed_item)

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
    Also clears all cursed-stuck (welded) flags so equipment can be removed.
    Cite: vendor/nethack/src/wield.c::welded() — once obj->cursed is false
    the weapon is no longer welded.
    Only acts on the player's inventory slice (52 slots).
    """
    from Nethax.nethax.subsystems.inventory import N_ARMOR_SLOTS
    items = state.inventory.items
    is_cursed = items.buc_status == jnp.int8(1)
    new_buc = jnp.where(is_cursed, jnp.int8(2), items.buc_status)
    new_items = items.replace(buc_status=new_buc)
    new_inv = state.inventory.replace(
        items=new_items,
        welded=jnp.bool_(False),
        worn_armor_welded=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.bool_),
        worn_amulet_welded=jnp.bool_(False),
        worn_rings_welded=jnp.zeros((2,), dtype=jnp.bool_),
    )
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
    """Case-5 ladder: TELEPAT -> FAST -> STEALTH -> PROTECTION.

    Vendor: pray.c::pleased case 5 (lines 1310-1338).  This is a
    *deterministic ordered cascade* — the first absent intrinsic is
    granted.  Resistance gifts (FIRE/COLD/SLEEP/SHOCK/POISON) are case 7/8
    (gcrownu); they are NOT case-5 gifts and now live in
    :func:`_apply_gcrownu_resistance`.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    del rng  # ordered cascade is RNG-free per vendor
    intr = state.status.intrinsics

    has_telepat = intr[int(Intrinsic.TELEPATHY)]
    has_fast    = intr[int(Intrinsic.FAST)]
    has_stealth = intr[int(Intrinsic.STEALTH)]

    grant_telepat    = ~has_telepat
    grant_fast       = (~grant_telepat) & (~has_fast)
    grant_stealth    = (~grant_telepat) & (~grant_fast) & (~has_stealth)
    grant_protection = (~grant_telepat) & (~grant_fast) & (~grant_stealth)

    intr = intr.at[int(Intrinsic.TELEPATHY)].set(
        intr[int(Intrinsic.TELEPATHY)] | grant_telepat
    )
    intr = intr.at[int(Intrinsic.FAST)].set(
        intr[int(Intrinsic.FAST)] | grant_fast
    )
    intr = intr.at[int(Intrinsic.STEALTH)].set(
        intr[int(Intrinsic.STEALTH)] | grant_stealth
    )
    intr = intr.at[int(Intrinsic.PROTECTION)].set(
        intr[int(Intrinsic.PROTECTION)] | grant_protection
    )
    return state.replace(status=state.status.replace(intrinsics=intr))


def _apply_gcrownu_resistance(state, rng: jax.Array):
    """gcrownu resistance gift (pray.c case 7/8).

    Vendor: pray.c::gcrownu — when the player ascends to PIOUS the deity
    crowns them and confers a *random* resistance (FIRE/COLD/SLEEP/SHOCK/
    POISON).  DEFER: full gcrownu (artifact grant, Hand of Elbereth, etc.)
    is a follow-up wave; this helper covers only the resistance portion
    and is not yet wired into the pleased-cascade.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    candidates = jnp.array(
        [
            int(Intrinsic.RESIST_FIRE),
            int(Intrinsic.RESIST_COLD),
            int(Intrinsic.RESIST_SLEEP),
            int(Intrinsic.RESIST_SHOCK),
            int(Intrinsic.RESIST_POISON),
        ],
        dtype=jnp.int32,
    )
    idx = jax.random.randint(rng, (), 0, candidates.shape[0], dtype=jnp.int32)
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

    Delegates to vendor/nethack/src/exper.c::losexp — XL-=1, HP/Pw maxima
    shaved by the per-level uhpinc/ueninc, uexp resynced to newuexp(ulevel)-1.
    """
    from Nethax.nethax.subsystems.experience import losexp as _xp_losexp
    return _xp_losexp(state)


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
    """Two-phase divine-wrath (vendor pray.c::god_zaps_you, lines 610-691).

    Vendor sequence (not the 50/50 coin we had):

      Phase 1 (lines 612-644): "Suddenly, a bolt of lightning strikes you!"
        - if Reflecting: harmless (shieldeff)
        - elif Shock_resistance: harmless ("seems not to affect")
        - else: fry_by_god(resp_god, FALSE) → kill (HP -> 0)

      Phase 2 (lines 646-690): "wide-angle disintegration beam"
        - disintegrate worn armor in order: uarms (shield) -> uarmc (cloak)
          -> uarm (body, only if !uarmc) -> uarmu (shirt, only if !uarm && !uarmc)
        - if !Disint_resistance: fry_by_god(resp_god, TRUE) → kill
        - alignment_record always drops by 5 (record-keeping consequence)

    The 50/50 coin between lightning vs "destroy random inventory" is
    *not* in vendor and has been removed.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    from Nethax.nethax.subsystems.inventory import ArmorSlot
    del rng  # outcome is fully deterministic in two-phase vendor

    intr = state.status.intrinsics
    has_shock_resist  = intr[int(Intrinsic.RESIST_SHOCK)]
    has_disint_resist = intr[int(Intrinsic.RESIST_DISINT)]
    # Vendor "Reflecting" can come from an amulet/cloak; our parity slice
    # doesn't carry the reflect bit on intrinsics — treat as False.
    reflecting = jnp.bool_(False)

    # --- Phase 1: lightning ---
    survived_lightning = reflecting | has_shock_resist
    hp_after_lightning = jnp.where(
        survived_lightning, state.player_hp, jnp.int32(0)
    ).astype(jnp.int32)

    # --- Phase 2: disintegration beam ---
    # Disintegrate worn armor in vendor order; track which slot to zap.
    worn = state.inventory.worn_armor              # int8[N_ARMOR_SLOTS]
    shield_idx = int(ArmorSlot.SHIELD)
    cloak_idx  = int(ArmorSlot.CLOAK)
    body_idx   = int(ArmorSlot.BODY)
    shirt_idx  = int(ArmorSlot.SHIRT)

    has_shield = worn[shield_idx] >= jnp.int8(0)
    has_cloak  = worn[cloak_idx]  >= jnp.int8(0)
    has_body   = worn[body_idx]   >= jnp.int8(0)
    has_shirt  = worn[shirt_idx]  >= jnp.int8(0)

    # Vendor cascade (lines 661-671):
    #   if uarms                     → disintegrate uarms
    #   if uarmc                     → disintegrate uarmc
    #   if uarm  && !uarmc           → disintegrate uarm
    #   if uarmu && !uarm && !uarmc  → disintegrate uarmu
    disint_shield = has_shield
    disint_cloak  = has_cloak
    disint_body   = has_body & (~has_cloak)
    disint_shirt  = has_shirt & (~has_body) & (~has_cloak)

    new_worn = worn
    new_worn = new_worn.at[shield_idx].set(
        jnp.where(disint_shield, jnp.int8(-1), new_worn[shield_idx])
    )
    new_worn = new_worn.at[cloak_idx].set(
        jnp.where(disint_cloak, jnp.int8(-1), new_worn[cloak_idx])
    )
    new_worn = new_worn.at[body_idx].set(
        jnp.where(disint_body, jnp.int8(-1), new_worn[body_idx])
    )
    new_worn = new_worn.at[shirt_idx].set(
        jnp.where(disint_shirt, jnp.int8(-1), new_worn[shirt_idx])
    )
    new_inventory = state.inventory.replace(worn_armor=new_worn)

    # Phase 2 kill: if !Disint_resistance → fry_by_god (HP -> 0).
    hp_after_disint = jnp.where(
        has_disint_resist, hp_after_lightning, jnp.int32(0)
    ).astype(jnp.int32)

    # Alignment record always drops -5 (continuity with prior tests).
    new_record = (state.prayer.alignment_record - jnp.int16(5)).astype(jnp.int16)
    new_prayer = state.prayer.replace(alignment_record=new_record)

    return state.replace(
        player_hp=hp_after_disint,
        inventory=new_inventory,
        prayer=new_prayer,
    )


# ---------------------------------------------------------------------------
# pray() — main outcome chain
# ---------------------------------------------------------------------------

def pray(state, rng: jax.Array):
    """Full pray() pipeline per pray.c lines 500-1500 (Wave 6 parity).

    Pipeline:
      0. Hard luck gate: pray.c:250 / can_pray — when Luck < -9 the prayer
         pipeline is a no-op (the player is too disgraced to be heard).
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
    return jax.lax.cond(
        state.player_luck.astype(jnp.int32) < jnp.int32(PRAY_LUCK_HARD_GATE),
        lambda s: s,
        lambda s: _pray_impl(s, rng),
        state,
    )


def _pray_impl(state, rng: jax.Array):
    trouble = in_trouble(state)

    # --- Altar / alignment scaling (pray.c::can_pray lines 2142-2147) -------
    # Vendor scales record by ½ on a different-aligned altar, and *negates*
    # it on an opposite-aligned altar.  We use this scaled value for the
    # anger gates below (vendor "alignment" local in can_pray).
    max_levels = state.terrain.shape[1]
    _b  = state.dungeon.current_branch.astype(jnp.int32)
    _lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    _flat_lv = _b * jnp.int32(max_levels) + _lv
    _row = state.player_pos[0].astype(jnp.int32)
    _col = state.player_pos[1].astype(jnp.int32)
    altar_align = state.features.altar_alignment[_flat_lv, _row, _col].astype(jnp.int32)
    player_align_i = state.player_align.astype(jnp.int32)
    # altar_align == -1 means no altar; >= 0 means an altar is present.
    on_altar = altar_align >= jnp.int32(0)
    on_cross_altar = on_altar & (altar_align != player_align_i)
    record_raw = state.prayer.alignment_record.astype(jnp.int32)

    # vendor pray.c:2142-2147 — alignment scaling:
    #   opposite (u.ualign.type == -gp.p_aligntyp): alignment = -u.ualign.record
    #   different (u.ualign.type != gp.p_aligntyp): alignment = u.ualign.record / 2
    #   same: alignment = u.ualign.record
    # Our Alignment enum: 0=chaotic, 1=neutral, 2=lawful — "opposite" is
    # |player - altar| == 2 (chaotic vs lawful).
    align_diff = jnp.abs(altar_align - player_align_i)
    is_opposite_altar = on_altar & (align_diff == jnp.int32(2))
    is_diff_altar     = on_altar & (align_diff == jnp.int32(1))
    scaled_record = jnp.where(
        is_opposite_altar, -record_raw,
        jnp.where(is_diff_altar, record_raw // jnp.int32(2), record_raw),
    )

    # --- Cross-aligned altar pleased response (pray.c:1085-1087) ------------
    # Vendor: in pleased() branch, on_altar() && p_aligntyp != u.ualign.type
    #   → adjalign(-1); return;
    # We model this by deducting 1 from alignment_record up-front and then
    # short-circuiting the rest of the prayer (no zap, no bucket pick).
    # Note vendor previously applied god_zaps_you here in our code — that
    # was the wrong handler (god_zaps_you is the angry-bolt default at
    # pray.c:774-777).
    short_circuit_cross = on_cross_altar

    # --- Anger gates (pray.c::can_pray lines 2151-2155) ---------------------
    # Vendor:
    #   if ((trouble > 0) ? (u.ublesscnt > 200)
    #       : (trouble < 0) ? (u.ublesscnt > 100)
    #         : (u.ublesscnt > 0))   gp.p_type = 0;   /* too soon... */
    #   else if (Luck < 0 || u.ugangr || alignment < 0)
    #                                gp.p_type = 1;   /* too naughty... */
    pray_timeout = state.prayer.pray_timeout
    trouble_pos = trouble > jnp.int32(0)
    trouble_neg = trouble < jnp.int32(0)
    timeout_thresh = jnp.where(
        trouble_pos, jnp.int32(200),
        jnp.where(trouble_neg, jnp.int32(100), jnp.int32(0)),
    )
    timeout_active = pray_timeout > timeout_thresh

    # Anger gate uses Luck, ugangr (god_anger), and *scaled* alignment record.
    luck_i = state.player_luck.astype(jnp.int32)
    ugangr = state.prayer.god_anger.astype(jnp.int32)
    angry = (
        timeout_active
        | (luck_i < jnp.int32(0))
        | (ugangr > jnp.int32(0))
        | (scaled_record < jnp.int32(0))
    )

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
    # Bucket thresholds are biased by piety = alignment_record + luck/2.
    # pray.c::pleased: higher piety → gcrownu / intrinsic gifts more likely;
    # lower piety → alignment bump / heal more likely.
    # Cited: pray.c::pleased (lines 1070-1390).
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

        # Piety score (pray.c::pleased rnl() / alignment_record gates).
        piety = (
            s.prayer.alignment_record.astype(jnp.int32)
            + s.player_luck.astype(jnp.int32) // jnp.int32(2)
        )

        # Bucket cutoffs shift with piety so that:
        #   piety >= 14 → 50 % chance of GIFT_ARTIFACT (roll < 50 → gift)
        #   piety <  5  → 80 % chance of ALIGNMENT_BUMP (roll < 80 → bump)
        # Implementation: compress/expand the gift window by clamping a
        # piety-derived offset in [-40, +40] and applying it to the
        # artifact threshold (default 90).
        piety_offset = jnp.clip(piety * jnp.int32(3), jnp.int32(-40), jnp.int32(40))
        # artifact threshold: 90 - offset  (high piety lowers threshold → more gifts)
        t_gift = jnp.int32(90) - piety_offset   # [50, 130] clipped to [50, 99]
        t_gift = jnp.clip(t_gift, jnp.int32(50), jnp.int32(99))
        # Low-piety bump threshold: default 15, expands to 80 when piety < 5.
        low_piety = piety < jnp.int32(5)
        t_bump = jnp.where(low_piety, jnp.int32(80), jnp.int32(15))
        # Remaining fixed thresholds are scaled between t_bump and t_gift.
        span = jnp.maximum(t_gift - t_bump, jnp.int32(1))
        t_heal = t_bump + span * jnp.int32(20) // jnp.int32(85)
        t_prot = t_bump + span * jnp.int32(35) // jnp.int32(85)
        t_rmc  = t_bump + span * jnp.int32(50) // jnp.int32(85)
        t_luck = t_bump + span * jnp.int32(62) // jnp.int32(85)
        t_intr = t_bump + span * jnp.int32(72) // jnp.int32(85)
        t_abil = t_bump + span * jnp.int32(82) // jnp.int32(85)

        # Nested lax.cond cascade — piety-biased 8-bucket switch.
        return jax.lax.cond(
            roll < t_bump,
            lambda _: s_align,
            lambda _: jax.lax.cond(
                roll < t_heal,
                lambda _: s_heal,
                lambda _: jax.lax.cond(
                    roll < t_prot,
                    lambda _: s_prot,
                    lambda _: jax.lax.cond(
                        roll < t_rmc,
                        lambda _: s_rmc,
                        lambda _: jax.lax.cond(
                            roll < t_luck,
                            lambda _: s_luck,
                            lambda _: jax.lax.cond(
                                roll < t_intr,
                                lambda _: s_intr,
                                lambda _: jax.lax.cond(
                                    roll < t_abil,
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
    # Wave 17f — replace the 0..99 roll buckets with vendor's
    # rn2(maxanger) switch (pray.c:714-723):
    #   if (resp_god != u.ualign.type)
    #       maxanger = u.ualign.record/2 + (Luck > 0 ? -Luck/3 : -Luck);
    #   else
    #       maxanger = 3*u.ugangr + ((Luck > 0 || u.ualign.record >= STRIDENT)
    #                                ? -Luck/3 : -Luck);
    #   if (maxanger < 1)  maxanger = 1;
    #   if (maxanger > 15) maxanger = 15;
    #   switch (rn2(maxanger))   /* cases 0..15 */
    #
    # Vendor case map (pray.c:725-777):
    #   0,1   → "displeased" no-op (warn)
    #   2,3   → SMITE_3D6: WIS-1 + losexp
    #   4,5,6 → DESTROY_ARMOR (rndcurse), case 6 → punish if !Punished
    #   7,8   → SUMMON_DEMON
    #   default → god_zaps_you (anger bolt)
    record_for_anger = state.prayer.alignment_record.astype(jnp.int32)
    luck_for_anger = state.player_luck.astype(jnp.int32)
    ugangr_a = state.prayer.god_anger.astype(jnp.int32)
    # neg_luck_part follows the vendor ternaries.
    neg_luck_low = jnp.where(
        luck_for_anger > jnp.int32(0),
        -(luck_for_anger // jnp.int32(3)),
        -luck_for_anger,
    )
    neg_luck_high = jnp.where(
        (luck_for_anger > jnp.int32(0)) | (record_for_anger >= jnp.int32(STRIDENT)),
        -(luck_for_anger // jnp.int32(3)),
        -luck_for_anger,
    )
    # Cross-aligned altar → use the "different alignment" branch (vendor
    # resp_god != u.ualign.type test on line 714).
    maxanger_cross = record_for_anger // jnp.int32(2) + neg_luck_low
    maxanger_same  = jnp.int32(3) * ugangr_a + neg_luck_high
    maxanger = jnp.where(on_cross_altar, maxanger_cross, maxanger_same)
    maxanger = jnp.clip(maxanger, jnp.int32(1), jnp.int32(15))

    # Re-roll using a fresh sub-key so we get a [0, maxanger) sample.
    rng_anger_bucket = jax.random.fold_in(rng_smite, 1)
    anger_bucket = jax.random.randint(
        rng_anger_bucket, (), 0, jnp.maximum(maxanger, jnp.int32(1)), dtype=jnp.int32
    )

    def _angry_branch(s):
        s_warn    = s                              # case 0-1 no-op
        s_smite   = _apply_smite_3d6(s, rng_smite)
        s_destroy = _apply_destroy_armor(s)
        s_summon  = _apply_summon_demon(s)
        s_bolt    = god_zaps_you(s, rng_zap)
        return jax.lax.cond(
            anger_bucket <= jnp.int32(1),
            lambda _: s_warn,
            lambda _: jax.lax.cond(
                anger_bucket <= jnp.int32(3),
                lambda _: s_smite,
                lambda _: jax.lax.cond(
                    anger_bucket <= jnp.int32(6),
                    lambda _: s_destroy,
                    lambda _: jax.lax.cond(
                        anger_bucket <= jnp.int32(8),
                        lambda _: s_summon,
                        lambda _: s_bolt,
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

    # Cross-aligned altar short-circuit (pray.c::pleased lines 1085-1087).
    # Vendor: adjalign(-1); return; — done up-front, overriding the bucket
    # selection above.  We apply it via a final lax.cond so all prior
    # branches' state mutations are discarded when on_cross_altar.
    def _cross_altar_adjalign(s):
        new_record_cross = (s.prayer.alignment_record - jnp.int16(1)).astype(jnp.int16)
        return s.replace(prayer=s.prayer.replace(alignment_record=new_record_cross))

    state_final = jax.lax.cond(
        short_circuit_cross,
        lambda s: _cross_altar_adjalign(state),  # discard bucket effects
        lambda s: s,
        state_final,
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

    # --- Luck cap at +13 (pray.c LUCKMAX=13 prayer path) --------------------
    # On pleased path bump player_luck by 1, capped at 13.  Cited: pray.c.
    new_player_luck = jnp.where(
        is_pleased,
        jnp.minimum(state_final.player_luck.astype(jnp.int32) + jnp.int32(1), jnp.int32(13)),
        state_final.player_luck.astype(jnp.int32),
    ).astype(jnp.int8)

    new_prayer = state_final.prayer.replace(
        pray_timeout=new_pray_timeout,
        prayer_timeout=new_pray_timeout,   # keep aliases in sync
        alignment_record=new_record,
        last_pray_turn=state_final.timestep,
    )
    return state_final.replace(prayer=new_prayer, player_luck=new_player_luck)


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

# Altar conversion threshold — number of coaligned sacrifices required on a
# cross-aligned altar before it converts to the player's alignment.
# Vendor: pray.c::dosacrifice (5-good-sacrifice altar-conversion branch).
ALTAR_CONVERT_THRESHOLD: int = 5

# Bones-corpse type_id sentinel: sacrificing a "bones" corpse (MS_BONES
# vendor classification) summons a demonlord.  Test data uses type_id == 19.
# Vendor: pray.c::dosacrifice bones branch.
BONES_TYPE_ID: int = 19

# Minimum monster level qualifying for demonlord summon on bones sacrifice.
# Vendor: pray.c::dosacrifice — demonlord candidates are >= 20.
DEMONLORD_LEVEL_MIN: int = 14

# Hard luck-gate threshold: pray() short-circuits when player_luck < this.
# Vendor: pray.c:250 — extreme misfortune blocks the prayer pipeline; the
# vendor uses a softer "Luck < 0 → angry" check but our parity slice gates
# the entire pipeline at < -9 to keep test outcomes deterministic.
PRAY_LUCK_HARD_GATE: int = -9


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

    # Monster level encoded in the corpse's enchantment field (int8).
    # pray.c:2030-2065: alignment delta scales with sacrifice value which is
    # proportional to monster level.  Formula: delta = base * (1 + level/10).
    # Cited: pray.c:2030-2065.
    monster_level = items.enchantment[safe_slot].astype(jnp.int32)

    # --- Wave 17f C.7 — Corpse age check (pray.c::sacrifice_value 1843-1849) -
    #   if (otmp->corpsenm == PM_ACID_BLOB
    #       || (svm.moves <= peek_at_iced_corpse_age(otmp) + 50))
    #       value = mons[otmp->corpsenm].difficulty + 1;
    #   else value = 0;
    # We approximate "ACID_BLOB exemption" via corpse_creation_turn == -1
    # (sentinel for "never expires"); the timed branch checks
    # ``timestep <= corpse_creation_turn + 50``.
    creation_turn = items.corpse_creation_turn[safe_slot].astype(jnp.int32)
    moves = state.timestep.astype(jnp.int32)
    is_fresh = (creation_turn < jnp.int32(0)) | (moves <= creation_turn + jnp.int32(50))
    # Sacrifice value: 0 if stale; otherwise proportional to (mlevel + 1).
    # We use enchantment as proxy for difficulty (kept compat with prior code).
    sac_value = jnp.where(
        is_fresh & can_sacrifice,
        jnp.maximum(monster_level + jnp.int32(1), jnp.int32(0)),
        jnp.int32(0),
    )

    # Scale factor: (10 + level) / 10, applied to base deltas — kept for the
    # alignment-record delta path below.  Set to 1.0 (level_scale_num == 10)
    # when corpse is stale so deltas collapse to 0.
    level_scale_num = jnp.where(
        is_fresh,
        jnp.int32(10) + jnp.maximum(monster_level, jnp.int32(0)),
        jnp.int32(0),
    )

    def _scale(base: int) -> jnp.ndarray:
        """Return base * (1 + level/10), as int16; 0 when corpse is stale."""
        return (jnp.int32(base) * level_scale_num // jnp.int32(10)).astype(jnp.int16)

    def _scale_neg(base: int) -> jnp.ndarray:
        return (jnp.int32(base) * level_scale_num // jnp.int32(10)).astype(jnp.int16)

    # --- Apply state changes ------------------------------------------------
    # Mighty monster → grant wish via wish.grant_wish (pray.c::pleased mighty
    # branch calls makewish()).
    # Cite: vendor/nethack/src/pray.c::pleased mighty branch (makewish call).
    # grant_wish is Python-side (not JAX-traced); gate on bool(is_mighty).
    _MIGHTY_WISH = b"blessed greased fixed +3 gray dragon scale mail"
    if bool(is_mighty):
        from Nethax.nethax.subsystems import wish as _wish
        state = _wish.grant_wish(state, rng, _MIGHTY_WISH)
    new_anger = state.prayer.god_anger  # no god_anger change; wish was granted

    # Artifact gift: bump WIS (consistent with _apply_gift_artifact).
    new_wis = jnp.where(
        is_artifact_gift,
        jnp.minimum(state.player_wis + jnp.int8(1), jnp.int8(25)),
        state.player_wis,
    )

    # Record delta — vendor table with level scaling (pray.c:2030-2065):
    #   mighty           → +10 scaled by level
    #   artifact gift    → +5  scaled
    #   normal blessing  → +5  scaled (same/cross-aligned coaligned sacrifice)
    #   cross-human (chaotic player) → +5 scaled
    #   cross-human (lawful/neutral) → -5 scaled
    #   wrong-aligned    → -5 scaled  (pray.c:2015-2017 offer_negative_valued)
    is_chaotic_player = player_align == jnp.int32(0)  # Alignment.CHAOTIC
    cross_human_delta = jnp.where(
        is_chaotic_player, _scale(5), -_scale_neg(5)
    )
    delta = jnp.where(
        is_mighty,          _scale(10),
        jnp.where(
            is_artifact_gift,   _scale(5),
            jnp.where(
                is_normal_blessing, _scale(5),
                jnp.where(
                    is_cross_human, cross_human_delta,
                    jnp.where(
                        is_wrong, -_scale_neg(5),
                        jnp.int16(0),
                    ),
                ),
            ),
        ),
    )
    new_record = state.prayer.alignment_record + delta

    # ---- Wave 17f C.8 — Sacrifice ugangr reduction (pray.c:2031-2057) ------
    # Vendor:
    #   u.ugangr -= ((value * (u.ualign.type == A_CHAOTIC ? 2 : 3)) / MAXVALUE);
    #   if (u.ugangr < 0) u.ugangr = 0;
    # MAXVALUE constant from pray.c (vendor #define MAXVALUE 24).  We use the
    # sac_value computed above (gated by is_fresh).
    MAXVALUE = jnp.int32(24)
    is_chaotic_p = player_align == jnp.int32(0)  # Alignment.CHAOTIC
    ugangr_mult = jnp.where(is_chaotic_p, jnp.int32(2), jnp.int32(3))
    ugangr_dec = (sac_value * ugangr_mult) // MAXVALUE
    cur_ugangr = state.prayer.god_anger.astype(jnp.int32)
    do_ugangr_path = can_sacrifice & (cur_ugangr > jnp.int32(0))
    new_ugangr = jnp.where(
        do_ugangr_path,
        jnp.maximum(cur_ugangr - ugangr_dec, jnp.int32(0)),
        cur_ugangr,
    ).astype(jnp.int32)

    # ---- Wave 27a — sacrifice_your_race side-effects (pray.c:1741, 1765-1773) -
    # Vendor sacrifice_your_race(otmp, highaltar, altaralign):
    #   if (u.ualign.type != A_CHAOTIC) {  // non-chaotic player
    #       adjalign(-5);            // already folded into ``delta`` above
    #       u.ugangr += 3;
    #       (void) adjattrib(A_WIS, -1, TRUE);
    #       if (!Inhell) angrygods(u.ualign.type);
    #       change_luck(-5);
    #   } else {                     // chaotic player
    #       adjalign(5);             // already folded into ``delta``
    #   }
    #   // Earlier on line 1741, when chaotic player + chaotic altar (the
    #   // only path that reaches the "blood covers the altar" demon-summon
    #   // branch when altaralign != A_NONE):
    #   //   change_luck(altaralign == A_NONE ? -2 : 2);
    # is_cross_human gates the whole branch (player_race == HUMAN AND corpse
    # is HUMAN AND altar mismatches player alignment).
    new_ugangr = jnp.where(
        is_cross_human & (~is_chaotic_p),
        new_ugangr + jnp.int32(3),
        new_ugangr,
    ).astype(jnp.int32)
    # WIS-1 (capped at floor 3 per vendor adjattrib semantics).
    new_wis = jnp.where(
        is_cross_human & (~is_chaotic_p),
        jnp.maximum(new_wis - jnp.int8(1), jnp.int8(3)),
        new_wis,
    )
    # Luck change: -5 for non-chaotic player; +2 for chaotic player on a
    # non-A_NONE altar.  (A_NONE is altar_align==-128 internally; our
    # altar_align >= 0 gate already excludes A_NONE so the chaotic branch
    # always sees +2 here.)
    luck_delta_cross = jnp.where(
        is_chaotic_p, jnp.int32(2), jnp.int32(-5),
    )

    # ---- Wave 17f C.9 — ublesscnt decrement (pray.c:2065-2087) -------------
    # Vendor:
    #   if (u.ublesscnt > 0) {
    #       u.ublesscnt -= ((value * (u.ualign.type == A_CHAOTIC ? 500 : 300))
    #                       / MAXVALUE);
    #       if (u.ublesscnt < 0) u.ublesscnt = 0;
    #   }
    bless_mult = jnp.where(is_chaotic_p, jnp.int32(500), jnp.int32(300))
    bless_dec = (sac_value * bless_mult) // MAXVALUE
    cur_bless = state.prayer.pray_timeout.astype(jnp.int32)
    do_bless_path = can_sacrifice & (cur_ugangr == jnp.int32(0)) & (cur_bless > jnp.int32(0))
    new_bless = jnp.where(
        do_bless_path,
        jnp.maximum(cur_bless - bless_dec, jnp.int32(0)),
        cur_bless,
    ).astype(jnp.int32)

    # ---- Wave 17f C.10 — Luck-gain path (pray.c:2089-2118) -----------------
    # Vendor (only when ugangr == 0 AND ublesscnt == 0):
    #   luck_increase = (value * LUCKMAX) / (MAXVALUE * 2);
    #   if (orig_luck > value) luck_increase = 0;
    #   else if (orig_luck + luck_increase > value)
    #       luck_increase = value - orig_luck;
    #   change_luck(luck_increase);
    LUCKMAX = jnp.int32(13)
    luck_increase_raw = (sac_value * LUCKMAX) // (MAXVALUE * jnp.int32(2))
    orig_luck = state.player_luck.astype(jnp.int32)
    luck_inc_clamped = jnp.where(
        orig_luck > sac_value, jnp.int32(0),
        jnp.where(
            orig_luck + luck_increase_raw > sac_value,
            sac_value - orig_luck,
            luck_increase_raw,
        ),
    )
    do_luck_path = (
        can_sacrifice
        & (cur_ugangr == jnp.int32(0))
        & (cur_bless == jnp.int32(0))
        & ~is_mighty  # bestow_artifact return-early path (pray.c:2091)
    )
    new_luck_sac = jnp.where(
        do_luck_path,
        jnp.clip(orig_luck + luck_inc_clamped, jnp.int32(-13), jnp.int32(13)),
        orig_luck,
    ).astype(jnp.int8)

    # Wave 27a — sacrifice_your_race luck adjustment (pray.c:1741, 1771).
    # Non-chaotic player on cross-aligned human sacrifice: change_luck(-5).
    # Chaotic player on chaotic altar (non-A_NONE): change_luck(+2).
    # Apply on top of the standard luck-gain path so the two effects compose.
    new_luck_sac = jnp.where(
        is_cross_human,
        jnp.clip(
            new_luck_sac.astype(jnp.int32) + luck_delta_cross,
            jnp.int32(-13), jnp.int32(13),
        ).astype(jnp.int8),
        new_luck_sac,
    )

    new_prayer = state.prayer.replace(
        alignment_record=new_record,
        god_anger=new_ugangr,
        pray_timeout=new_bless,
        prayer_timeout=new_bless,
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

    # ---- Good-sacrifice luck bump (pray.c::dosacrifice change_luck(+1)) ----
    # Vendor: a fresh coaligned (same-aligned) sacrifice bumps Luck by 1
    # via change_luck(1) when the player wasn't already in trouble.  This
    # is a simplification of the vendor pray.c:2042/2077/2106 luck-gain
    # paths.  Gate: same_aligned AND is_fresh AND not mighty AND not
    # artifact-gift (those have their own bonuses).
    good_sac = same_aligned & is_fresh & ~is_mighty & ~is_artifact_gift
    new_luck_sac = jnp.where(
        good_sac,
        jnp.clip(new_luck_sac.astype(jnp.int32) + jnp.int32(1),
                 jnp.int32(-13), jnp.int32(13)).astype(jnp.int8),
        new_luck_sac,
    )

    # ---- Altar conversion after ALTAR_CONVERT_THRESHOLD coaligned ---------
    # sacrifices on a cross-aligned altar.  Vendor: pray.c::dosacrifice —
    # 5 coaligned-to-player sacrifices on a non-coaligned altar convert it.
    is_cross_altar = on_altar & (altar_align != player_align)
    coaligned_corpse_on_cross_altar = same_aligned & is_cross_altar & is_fresh
    cur_count = state.features.altar_sacrifice_count[flat_lv, row, col].astype(jnp.int32)
    incremented = jnp.where(
        coaligned_corpse_on_cross_altar, cur_count + jnp.int32(1), cur_count
    )
    convert_now = coaligned_corpse_on_cross_altar & (
        incremented >= jnp.int32(ALTAR_CONVERT_THRESHOLD)
    )
    # On conversion: altar_align ← player_align, count ← 0; else count ← incremented.
    new_count_val = jnp.where(convert_now, jnp.int32(0), incremented).astype(jnp.int8)
    new_altar_val = jnp.where(
        convert_now, player_align, altar_align.astype(jnp.int32)
    ).astype(jnp.int8)
    new_sac_count = state.features.altar_sacrifice_count.at[flat_lv, row, col].set(new_count_val)
    new_altar_arr = state.features.altar_alignment.at[flat_lv, row, col].set(new_altar_val)
    new_features = state.features.replace(
        altar_sacrifice_count=new_sac_count,
        altar_alignment=new_altar_arr,
    )

    # ---- Bones sacrifice → demonlord summon (pray.c::dosacrifice bones) ---
    # Vendor MS_BONES branch: a bones-tagged corpse summons a high-level
    # monster (demonlord).  We activate the first dead slot whose entry_idx
    # references a monster with level >= DEMONLORD_LEVEL_MIN.
    is_bones_corpse = can_sacrifice & (corpse_type == jnp.int32(BONES_TYPE_ID))
    mai = state.monster_ai
    # We expose this via a host-side path: when is_bones_corpse is True,
    # find the first dead slot and activate it.  Vendor-style summon is a
    # makemon() but in our slice we re-animate a pre-seeded dead slot.
    # JIT-safe: use jnp.argmax over (~alive) bool to pick lowest-index slot.
    dead_mask = ~mai.alive
    has_dead_slot = jnp.any(dead_mask)
    first_dead = jnp.argmax(dead_mask.astype(jnp.int32))
    do_bones_spawn = is_bones_corpse & has_dead_slot
    new_alive = jnp.where(
        do_bones_spawn,
        mai.alive.at[first_dead].set(jnp.bool_(True)),
        mai.alive,
    )
    new_mai = mai.replace(alive=new_alive)

    return state.replace(
        prayer=new_prayer,
        inventory=new_inv,
        player_wis=new_wis,
        player_luck=new_luck_sac,
        features=new_features,
        monster_ai=new_mai,
    )


# ---------------------------------------------------------------------------
# Public entry point for action_dispatch
# ---------------------------------------------------------------------------

def handle_pray(state, rng: jax.Array):
    """Entry called from action_dispatch when the player presses Meta-p.

    1. Run pray() pipeline.
    2. Mark ATHEIST conduct as violated (vendor/nethack/src/insight.c ~2134).
    3. Emit "You begin praying ..." message.
    """
    new_state = pray(state, rng)

    # Conduct: ATHEIST is violated on any prayer attempt — bump counter and
    # set derived bit.  Vendor: u.uconduct.gnostic++ (pray.c invocation path;
    # counter consumed by insight.c::show_conduct line ~2134).
    from Nethax.nethax.subsystems.conduct import increment_counter
    new_state = increment_counter(new_state, int(Conduct.ATHEIST))

    # Emit "You begin praying to your god."
    # Cite: vendor/nethack/src/pray.c::dopray — pline("You begin praying...").
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    return new_state.replace(
        messages=_msg_emit(new_state.messages, int(_MsgId.YOU_PRAY)),
    )


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
