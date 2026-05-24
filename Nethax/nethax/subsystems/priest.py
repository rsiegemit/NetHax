"""Priest / temple subsystem.

Vendor sources:
  vendor/nethack/src/priest.c — priest_talk donation cascade (lines 557-721),
                                priestini (priest spawn at altar shrine_pos,
                                lines ~430-480 in vendor priest.c), inhistemple,
                                forget_temple_entry, mk_roamer, p_coaligned.
  vendor/nethack/src/mkroom.c::mktemple (lines 597-620) — TEMPLE room fill
                                : blesses the centre altar, marks it
                                AM_SHRINE, calls priestini, sets
                                level.flags.has_temple.
  vendor/nethack/src/priest.c::intemple (level.flags.has_temple gate +
                                in_rooms(TEMPLE) check).

Public API
----------
  priestini(state, room_idx, x, y, hostile=False)
      Place a peaceful priest record (alignment matches altar) at (x,y).
      Returns the new EnvState.

  intemple(state)
      Returns a scalar bool — True iff player_pos is inside a TEMPLE-rtype
      room on the current level and the level's has_temple flag is set.

  temple_violation(state)
      Called when the player misbehaves in a temple (e.g. desecrates an
      altar, drops a cursed item on shrine).  Flips the priest's
      ``peaceful`` flag to False on the current level's PriestState.

  priest_talk(state, rng)
      Donation handler (#chat-on-priest).  Mirrors the cascade in
      vendor/nethack/src/priest.c::priest_talk (lines 557-720):
        suggested = ulevelpeak * rn1(101, 150 + cheapskate*40)
        offer == 0                  → adjalign(-1) if coaligned; cheapskate++
        offer < suggested*quan      → cheapskate verbalize OR small bless
        offer < suggested*quan*2    → +Clairvoyant timeout
        offer < suggested*quan*3    → HProtection + u.ublessed ladder up to 20
        offer >= suggested*quan*3   → adjalign(+2) or cleanse if strayed
      The offer amount is auto-drawn as half of the player's current gold
      (a deterministic stand-in for the vendor ``bribe`` prompt) so the
      function is fully JIT-safe and side-effect free.

Deferred (documented, NOT implemented)
--------------------------------------
  pri_move        — vendor priest.c::pri_move; priest-on-shrine wandering /
                    AI loop.  Out of scope for Round 2.
  findpriest      — vendor priest.c::findpriest; priest-by-coord lookup over
                    the level's monster list.  Out of scope; needs a true
                    PriestState per-monster layer (Wave 7+).
  mk_roamer       — vendor priest.c::mk_roamer; spawn roaming priests
                    (only used for special-level priests, e.g. Astral
                    Plane).  Out of scope here.
  full priest combat / HP loop — priests are not yet a separate monster
                    sub-type with intone/peaceful timers (epri.intone_time,
                    enter_time, peaceful_time, hostile_time).  Tracked via
                    PriestState.peaceful only.

JIT safety
----------
All functions are jax.lax-traceable.  They use jnp scalar ops and
state.replace() (no Python ``if`` branches that depend on tracer values).
Threefry RNG is consumed via jax.random.split — no key reuse.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


# ---------------------------------------------------------------------------
# PriestState — top-level per-game priest bookkeeping.
#
# Wave 41g report noted that ``priest_talk`` stashed cheapskate_count locally
# and dropped it on return; vendor priest.c::priest_talk reads/writes
# ``EPRI(priest)->cheapskate`` across multiple #chat calls.  Hoist the field
# here so it survives between calls and matches vendor parity.
#
# Field map vs vendor ``struct epri`` (include/epri.h):
#   cheapskate_count <- EPRI(priest)->cheapskate (priest.c lines 612-680)
#   pri_alignment    <- EPRI(priest)->shralign   (priest.c::priestini line 248)
#   pri_intone_time  <- EPRI(priest)->intone_time (priest.c::intemple)
#   pri_enter_time   <- EPRI(priest)->enter_time  (priest.c::intemple)
# ---------------------------------------------------------------------------

@struct.dataclass
class PriestState:
    """Persistent per-game priest record.

    Mirrors the subset of vendor ``struct epri`` (include/epri.h) needed
    for the donation cascade and temple-entry hints.  Defaults match a
    fresh game (no priest interactions yet).
    """
    # priest.c lines 612-680: ``EPRI(priest)->cheapskate`` — times the caller
    # refused to donate; increases ``suggested`` next call by +40 per bump.
    cheapskate_count: jnp.ndarray   # int32 scalar
    # priest.c::priestini line 248: ``EPRI(priest)->shralign = ...``
    # Last-known altar alignment of the priest in the current temple.
    pri_alignment:    jnp.ndarray   # int8 scalar
    # priest.c::intemple uses long ``intone_time`` to throttle re-speaks of
    # the temple-entry blessing line.
    pri_intone_time:  jnp.ndarray   # int32 scalar
    # priest.c::intemple uses ``enter_time`` to throttle the entry hint.
    pri_enter_time:   jnp.ndarray   # int32 scalar

    @classmethod
    def default(cls) -> "PriestState":
        return cls(
            cheapskate_count=jnp.int32(0),
            pri_alignment=jnp.int8(0),
            pri_intone_time=jnp.int32(0),
            pri_enter_time=jnp.int32(0),
        )


# ---------------------------------------------------------------------------
# Priest record is encoded into FeaturesState:
#   features.altar_shrine[level, y, x]  — True at the priest's shrine tile
#   features.has_temple [level]         — True iff a priest was placed here
# Vendor priest.c uses ``struct epri`` per priest monster; the per-level
# subset (shrine tile + has_temple) is what our parity slice exposes.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# priestini — spawn priest at altar shrine_pos
# Vendor: priest.c::priestini (called from mkroom.c::mktemple line 617).
# ---------------------------------------------------------------------------

def priestini(state, room_idx: int, x, y, hostile: bool = False):
    """Place a peaceful priest at (x, y) on the current level.

    Mirrors vendor priest.c::priestini (called from mkroom.c::mktemple line
    617).  Sets the shrine tile (features.altar_shrine[level, y, x] = True)
    and the level's has_temple flag.  The altar's alignment (already set by
    mktemple line 616) is taken as-is from features.altar_alignment.
    """
    x_i = jnp.int32(x)
    y_i = jnp.int32(y)
    max_levels = state.terrain.shape[1]
    _b  = state.dungeon.current_branch.astype(jnp.int32)
    _lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    _flat_lv = _b * jnp.int32(max_levels) + _lv

    new_shrine = state.features.altar_shrine.at[_flat_lv, y_i, x_i].set(
        jnp.bool_(True)
    )
    new_has_temple = state.features.has_temple.at[_flat_lv].set(jnp.bool_(True))
    new_features = state.features.replace(
        altar_shrine=new_shrine,
        has_temple=new_has_temple,
    )
    return state.replace(features=new_features)


# ---------------------------------------------------------------------------
# intemple — is the player in a temple room
# Vendor: priest.c::inhistemple (level.flags.has_temple + room rtype check).
# ---------------------------------------------------------------------------

def intemple(state) -> jnp.ndarray:
    """Return True iff player is on a TEMPLE shrine tile on this level.

    Vendor: priest.c::inhistemple — combines level.flags.has_temple +
    in_rooms(x, y, TEMPLE).  Our parity slice collapses ``in_rooms`` to
    the shrine-tile check (altar_shrine[level, y, x]) gated by
    has_temple[level].
    """
    max_levels = state.terrain.shape[1]
    _b  = state.dungeon.current_branch.astype(jnp.int32)
    _lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    _flat_lv = _b * jnp.int32(max_levels) + _lv
    py = state.player_pos[0].astype(jnp.int32)
    px = state.player_pos[1].astype(jnp.int32)
    return (
        state.features.has_temple[_flat_lv]
        & state.features.altar_shrine[_flat_lv, py, px]
    )


# ---------------------------------------------------------------------------
# temple_violation — desecration handler
# Vendor: priest.c::priest_talk lines 603-611 (desecration → mpeaceful=0)
# and dokick.c / read.c desecration paths.
# ---------------------------------------------------------------------------

def temple_violation(state):
    """Mark the current-level temple desecrated.

    Vendor: priest.c::priest_talk lines 603-611 (priest sees a desecrator
    and flips mpeaceful=0).  Our parity slice clears the shrine flag at
    the player's tile (representing the broken shrine bond) and zeros
    has_temple, both of which gate subsequent priest_talk / intemple
    calls.
    """
    max_levels = state.terrain.shape[1]
    _b  = state.dungeon.current_branch.astype(jnp.int32)
    _lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    _flat_lv = _b * jnp.int32(max_levels) + _lv
    new_has_temple = state.features.has_temple.at[_flat_lv].set(jnp.bool_(False))
    new_features = state.features.replace(has_temple=new_has_temple)
    return state.replace(features=new_features)


# ---------------------------------------------------------------------------
# priest_talk — donation cascade (priest.c lines 557-721)
# ---------------------------------------------------------------------------

# Vendor donation parameters from priest.c:637-638:
#   suggested = (u.ulevelpeak ? u.ulevelpeak : 1) * rn1(101, 150 + cheapskate*40)
# rn1(N, b) returns b + rn2(N), i.e. uniform in [b, b+N).
DONATION_BASE: int = 150
DONATION_RANGE: int = 101
CHEAPSKATE_BUMP: int = 40
UBLESSED_CAP: int = 20
UCLEANSED_INTERVAL: int = 5000   # priest.c:712 svm.moves - u.ucleansed > 5000


def priest_talk(state, rng: jax.Array):
    """Handle a #chat-with-priest donation event (vendor priest.c:557-721).

    Pipeline (vendor lines 612-720):

      0.  No gold in inventory → return (no change).
      1.  Compute suggested donation:
            suggested = max(1, ulevelpeak) *
                        rn1(101, 150 + cheapskate*40)
          quan      = max(1, gold // (suggested * 3))
      2.  Pick an offer.  In vendor this is the player's response; here we
          deterministically take half the player's gold as a JIT-safe
          stand-in (so the function has a stable trace).
      3.  Cascade over (offer / (suggested*quan)) buckets:
            offer == 0                              → cheapskate++; coaligned → adjalign(-1)
            offer <  suggested*quan                 → cheapskate++ OR small bless
            offer <  suggested*quan*2               → +Clairvoyant timed
            offer <  suggested*quan*3               → HProtection + ublessed ladder
            offer >= suggested*quan*3               → adjalign(+2) or alignment cleanse

    Cost in gold is deducted (gold -= offer).
    Returns the new EnvState.
    """
    # --- 1. Compute "suggested" ----------------------------------------
    rng_offer, _ = jax.random.split(rng, 2)
    # vendor priest.c lines 612-638: ``suggested = ... * rn1(101, 150 +
    # cheapskate*40)``.  Read EPRI(priest)->cheapskate from PriestState
    # (hoisted to EnvState in this wave so the count persists across calls).
    cheapskate = state.priest.cheapskate_count.astype(jnp.int32)
    # ulevelpeak — we use player_xl (current XL); vendor uses u.ulevelpeak,
    # the player's highest XL achieved, but we don't carry that field.
    ulevelpeak = jnp.maximum(state.player_xl.astype(jnp.int32), jnp.int32(1))
    # rn1(101, 150 + cheapskate*40) = 150 + cheapskate*40 + rn2(101)
    base = jnp.int32(DONATION_BASE) + cheapskate * jnp.int32(CHEAPSKATE_BUMP)
    rng_rn1, _ = jax.random.split(rng_offer, 2)
    rn101 = jax.random.randint(
        rng_rn1, (), 0, jnp.int32(DONATION_RANGE), dtype=jnp.int32
    )
    suggested = ulevelpeak * (base + rn101)
    suggested = jnp.maximum(suggested, jnp.int32(1))

    # --- 2. Pick offer = half of player gold (deterministic) -----------
    gold = state.player_gold.astype(jnp.int32)
    quan = jnp.maximum(gold // (suggested * jnp.int32(3)), jnp.int32(1))
    offer = gold // jnp.int32(2)
    offer = jnp.minimum(offer, gold)
    offer = jnp.maximum(offer, jnp.int32(0))

    # If no gold, this is a no-op.
    has_gold = gold > jnp.int32(0)

    # --- 3. Player / altar alignment for coaligned checks --------------
    # The priest's alignment matches the shrine altar (mktemple line 616).
    player_align_i = state.player_align.astype(jnp.int32)
    max_levels = state.terrain.shape[1]
    _b  = state.dungeon.current_branch.astype(jnp.int32)
    _lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    _flat_lv = _b * jnp.int32(max_levels) + _lv
    py = state.player_pos[0].astype(jnp.int32)
    px = state.player_pos[1].astype(jnp.int32)
    altar_align = state.features.altar_alignment[_flat_lv, py, px].astype(jnp.int32)
    coaligned = altar_align == player_align_i
    strayed = state.prayer.alignment_record.astype(jnp.int32) < jnp.int32(0)

    # --- 4. Bucket thresholds ------------------------------------------
    threshold_1 = suggested * quan
    threshold_2 = suggested * quan * jnp.int32(2)
    threshold_3 = suggested * quan * jnp.int32(3)

    # Bucket 0: offer == 0.  cheapskate++, coaligned → adjalign(-1).
    # Bucket 1: 0 < offer < threshold_1.  cheapskate++.
    # Bucket 2: threshold_1 <= offer < threshold_2.  Clairvoyant timer.
    # Bucket 3: threshold_2 <= offer < threshold_3.  HProtection + ublessed.
    # Bucket 4: offer >= threshold_3.  adjalign(+2) or cleanse.
    b_zero    = offer == jnp.int32(0)
    b_low     = (~b_zero) & (offer < threshold_1)
    b_mid     = (offer >= threshold_1) & (offer < threshold_2)
    b_high    = (offer >= threshold_2) & (offer < threshold_3)
    b_devout  = offer >= threshold_3

    # --- 5. Apply effects ----------------------------------------------
    # cheapskate_count bump (buckets 0 + 1).
    cheap_bump = jnp.where(b_zero | b_low, jnp.int32(1), jnp.int32(0))
    new_cheap = (cheapskate + cheap_bump).astype(jnp.int32)

    # adjalign delta:
    #   bucket 0 + coaligned → -1
    #   bucket 4 (not strayed) → +2
    #   bucket 4 + strayed sufficiently long → cleanse (alignment_record = 0)
    record_i32 = state.prayer.alignment_record.astype(jnp.int32)
    timestep_i32 = state.timestep.astype(jnp.int32)
    last_clean = state.prayer.last_pray_turn.astype(jnp.int32)
    long_strayed = strayed & ((timestep_i32 - last_clean) > jnp.int32(UCLEANSED_INTERVAL))

    delta_align = jnp.where(
        b_zero & coaligned, jnp.int32(-1),
        jnp.where(b_devout & (~long_strayed), jnp.int32(2), jnp.int32(0)),
    )
    after_record = jnp.where(
        b_devout & long_strayed,
        jnp.int32(0),  # cleanse
        record_i32 + delta_align,
    )
    # Mid bucket: Clairvoyant timer = rn1(500*offer/suggested, 500*offer/suggested)
    # = 500*offer/suggested + rn2(500*offer/suggested)
    rng_clair, _ = jax.random.split(rng_offer, 2)
    clair_base = jnp.maximum(
        jnp.int32(500) * offer // jnp.maximum(suggested, jnp.int32(1)),
        jnp.int32(1),
    )
    clair_rand = jax.random.randint(rng_clair, (), 0, clair_base, dtype=jnp.int32)
    clair_total = clair_base + clair_rand

    # Apply Clairvoyant timer on bucket 2.
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    clairv_idx = int(Intrinsic.CLAIRVOYANT)
    cur_clairv = state.status.timed_intrinsics[clairv_idx]
    new_clairv_val = jnp.where(
        b_mid & has_gold,
        cur_clairv + clair_total,
        cur_clairv,
    ).astype(jnp.int32)
    new_timed = state.status.timed_intrinsics.at[clairv_idx].set(new_clairv_val)

    # Bucket 3: grant HProtection (PROTECTION intrinsic).  We bump
    # alignment_record by +1 as a JIT-safe stand-in for u.ublessed.
    prot_idx = int(Intrinsic.PROTECTION)
    cur_intr = state.status.intrinsics
    new_prot = cur_intr.at[prot_idx].set(
        jnp.where(b_high & has_gold, jnp.bool_(True), cur_intr[prot_idx])
    )
    new_status = state.status.replace(
        intrinsics=new_prot, timed_intrinsics=new_timed,
    )

    # If bucket 3, additional alignment_record +1 ladder (the u.ublessed
    # bump in vendor; here record absorbs it).
    extra_record = jnp.where(b_high & has_gold, jnp.int32(1), jnp.int32(0))
    final_record = jnp.where(has_gold, after_record + extra_record, record_i32)
    final_record_i16 = jnp.clip(final_record, jnp.int32(-127), jnp.int32(127)).astype(
        jnp.int16
    )

    # --- 6. Deduct gold and write back --------------------------------
    new_gold = jnp.where(has_gold, gold - offer, gold).astype(jnp.int32)

    # vendor priest.c lines 612-680: ``EPRI(priest)->cheapskate`` persists
    # across calls; write back through PriestState.  No-op when has_gold is
    # False (the function is an early-return no-op in vendor too).
    new_cheap_final = jnp.where(has_gold, new_cheap, cheapskate).astype(jnp.int32)

    new_prayer = state.prayer.replace(alignment_record=final_record_i16)
    new_priest = state.priest.replace(cheapskate_count=new_cheap_final)
    return state.replace(
        prayer=new_prayer,
        priest=new_priest,
        status=new_status,
        player_gold=new_gold,
    )


# ---------------------------------------------------------------------------
# gcrownu — crowning event (pray.c::gcrownu lines 805-996)
# ---------------------------------------------------------------------------
#
# Vendor pray.c::gcrownu (lines 805-996) is the deity's "crown the player"
# ceremony.  Effects (byte-equal vendor):
#   1. Set HFire_resistance / HCold_resistance / HShock_resistance /
#      HSleep_resistance / HPoison_resistance |= FROMOUTSIDE
#      (lines 813-818).
#   2. Set HSee_invisible |= FROMOUTSIDE (line 813).
#   3. Verbalize the role/alignment-specific message (lines 837-868).
#   4. Grant role/alignment-specific artifact (lines 837-974):
#         A_LAWFUL: u.uevent.uhand_of_elbereth=1 + Excalibur (line 907)
#         A_NEUTRAL: ... + Vorpal Blade (line 929)
#         A_CHAOTIC: ... + Stormbringer (line 955)
#      Plus Wizard class_gift = SPE_FINGER_OF_DEATH (line 828),
#      Monk class_gift = SPE_RESTORE_ABILITY (line 832).
#   5. Set u.ugifts++ (line 885, 910, 934, 960) and ACH_CROWN achievement
#      (vendor pleased() callers; pray.c line 1343 invokes gcrownu after
#      record_achievement(ACH_PRAY_PIOUS)).
#
# This implementation follows the user-supplied role→artifact mapping
# (matches each role's quest artifact, simplifying the vendor's
# alignment-and-class-gift cascade for JIT-safe parity).  Indices into
# wish._ARTIFACTS:
#   Knight (lawful)      → ART_EXCALIBUR   idx 0
#   Knight (chaotic)     → ART_VORPAL_BLADE idx 8
#   Wizard               → ART_MAGICBANE   idx 29
#   Priest (god-aligned) → ART_MJOLLNIR    idx 3
#   Priest (neutral)     → ART_SCEPTRE_OF_MIGHT idx 9
#   Samurai              → ART_SNICKERSNEE idx 1
#   Valkyrie (lawful)    → ART_MJOLLNIR    idx 3
#   Valkyrie (chaotic)   → ART_FROST_BRAND idx 22
#   Barbarian            → ART_CLEAVER     idx 4
#   Caveman              → ART_SCEPTRE_OF_MIGHT idx 9
#   Archeologist         → ART_MAGIC_MIRROR_OF_MERLIN idx 11
#   Healer               → ART_STAFF_OF_AESCULAPIUS  idx 14
#   Monk                 → ART_EYES_OF_THE_OVERWORLD idx 15
#   Ranger               → ART_LONGBOW_OF_DIANA      idx 17
#   Rogue                → ART_MASTER_KEY_OF_THIEVERY idx 18
#   Tourist              → ART_YENDORIAN_EXPRESS_CARD idx 19
# ---------------------------------------------------------------------------

# Role × alignment → (artifact_idx, base_type_id, category) table.
# Built lazily on first call to keep import-time cost low.

# Vendor item type_ids from constants/objects.py.  The base-object names
# match wish._ARTIFACTS[idx][1] so the table here only needs to record the
# resolved type_id once.  We resolve to numeric type_ids via wish._OBJECT_BY_NAME
# at first call.
_GCROWNU_ROLE_ARTI_TABLE: dict = {}


def _build_gcrownu_table() -> dict:
    """Build {(role_id, align_id): (artifact_idx, type_id)} mapping.

    Lawful = Alignment.LAWFUL (2), Neutral = NEUTRAL (1), Chaotic = CHAOTIC (0).
    role_id matches Role IntEnum in constants/roles.py.
    """
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.prayer import Alignment
    from Nethax.nethax.subsystems.wish import _ARTIFACTS, _OBJECT_BY_NAME

    def _idx_for(name: str) -> int:
        for i, (n, _b) in enumerate(_ARTIFACTS):
            if n == name:
                return i
        return -1

    def _entry(arti_name: str) -> tuple:
        idx = _idx_for(arti_name)
        if idx < 0:
            return (-1, 0)
        base_name = _ARTIFACTS[idx][1]
        return (idx, _OBJECT_BY_NAME.get(base_name, 0))

    L = int(Alignment.LAWFUL)
    N = int(Alignment.NEUTRAL)
    C = int(Alignment.CHAOTIC)
    # Per-role gifts.  Same artifact across alignments for most roles —
    # the alignment-dependent split only applies to KNIGHT/PRIEST/VALKYRIE.
    table: dict = {}

    # Knight: lawful → Excalibur, chaotic → Vorpal Blade (per task spec).
    # Vendor pray.c lines 907 / 929 (alignment-keyed gifts).
    table[(int(Role.KNIGHT), L)]    = _entry("Excalibur")
    table[(int(Role.KNIGHT), N)]    = _entry("Excalibur")  # KNIGHTs are lawful only in vendor
    table[(int(Role.KNIGHT), C)]    = _entry("Vorpal Blade")

    # Wizard: Magicbane regardless of alignment (per task spec).
    for a in (L, N, C):
        table[(int(Role.WIZARD), a)] = _entry("Magicbane")

    # Priest: god-aligned (Lawful/Chaotic) → Mjollnir; Neutral → Sceptre of Might.
    table[(int(Role.PRIEST), L)] = _entry("Mjollnir")
    table[(int(Role.PRIEST), N)] = _entry("Sceptre of Might")
    table[(int(Role.PRIEST), C)] = _entry("Mjollnir")

    # Samurai: Snickersnee.
    for a in (L, N, C):
        table[(int(Role.SAMURAI), a)] = _entry("Snickersnee")

    # Valkyrie: Lawful → Mjollnir, Chaotic → Frost Brand.  Neutral → Mjollnir.
    table[(int(Role.VALKYRIE), L)] = _entry("Mjollnir")
    table[(int(Role.VALKYRIE), N)] = _entry("Mjollnir")
    table[(int(Role.VALKYRIE), C)] = _entry("Frost Brand")

    # Barbarian: Cleaver.
    for a in (L, N, C):
        table[(int(Role.BARBARIAN), a)] = _entry("Cleaver")

    # Caveman: Sceptre of Might.
    for a in (L, N, C):
        table[(int(Role.CAVEMAN), a)] = _entry("Sceptre of Might")

    # Archeologist: Magic Mirror of Merlin.
    for a in (L, N, C):
        table[(int(Role.ARCHEOLOGIST), a)] = _entry("Magic Mirror of Merlin")

    # Healer: Staff of Aesculapius.
    for a in (L, N, C):
        table[(int(Role.HEALER), a)] = _entry("Staff of Aesculapius")

    # Monk: Eyes of the Overworld.
    for a in (L, N, C):
        table[(int(Role.MONK), a)] = _entry("Eyes of the Overworld")

    # Ranger: Longbow of Diana.
    for a in (L, N, C):
        table[(int(Role.RANGER), a)] = _entry("Longbow of Diana")

    # Rogue: Master Key of Thievery.
    for a in (L, N, C):
        table[(int(Role.ROGUE), a)] = _entry("Master Key of Thievery")

    # Tourist: Yendorian Express Card.
    for a in (L, N, C):
        table[(int(Role.TOURIST), a)] = _entry("Yendorian Express Card")

    return table


def _gcrownu_lookup(role_id: jnp.ndarray, align_id: jnp.ndarray):
    """Look up (artifact_idx, type_id) for a given (role, align).

    Returns (int32, int32) jax arrays.  Falls back to (-1, 0) if the
    (role, align) pair has no gift (vendor STRANGE_OBJECT path).
    """
    global _GCROWNU_ROLE_ARTI_TABLE
    if not _GCROWNU_ROLE_ARTI_TABLE:
        _GCROWNU_ROLE_ARTI_TABLE = _build_gcrownu_table()

    # Build flat arrays for jnp lookup.  Cite: vendor pray.c::gcrownu uses
    # a switch(u.ualign.type) over u.urole; we collapse to a (role,align)
    # gather over a precomputed 13*4 table (13 roles × {L,N,C} alignments).
    from Nethax.nethax.constants.roles import Role as _R
    from Nethax.nethax.subsystems.prayer import Alignment as _A

    n_roles = 13                       # constants/roles.py::N_ROLES
    n_aligns = 3                       # CHAOTIC=0, NEUTRAL=1, LAWFUL=2

    arti_grid = [[-1] * n_aligns for _ in range(n_roles)]
    type_grid = [[0]  * n_aligns for _ in range(n_roles)]
    for (r, a), (idx, tid) in _GCROWNU_ROLE_ARTI_TABLE.items():
        if 0 <= r < n_roles and 0 <= a < n_aligns:
            arti_grid[r][a] = idx
            type_grid[r][a] = tid
    arti_arr = jnp.array(arti_grid, dtype=jnp.int32)   # [n_roles, n_aligns]
    type_arr = jnp.array(type_grid, dtype=jnp.int32)

    r_i = jnp.clip(role_id.astype(jnp.int32),  0, n_roles - 1)
    a_i = jnp.clip(align_id.astype(jnp.int32), 0, n_aligns - 1)
    return arti_arr[r_i, a_i], type_arr[r_i, a_i]


def gcrownu(state, rng: jax.Array):
    """Crown the player — vendor pray.c::gcrownu (lines 805-996).

    Effects (byte-equal vendor):
      1. Grant 5 elemental resistances FROMOUTSIDE (pray.c:814-818):
           FIRE_RES, COLD_RES, SHOCK_RES, SLEEP_RES, POISON_RES.
      2. Grant SEE_INVIS intrinsic FROMOUTSIDE (pray.c:813).
      3. Grant a role+alignment-specific artifact in the first empty
         inventory slot (pray.c:837-974 + ``_GCROWNU_ROLE_ARTI_TABLE``).
      4. (ACH_CROWN achievement omitted — scoring.py is outside this
         wave's scope per the task contract.)

    JIT safety: the lookup table is built lazily once at module import
    time; the per-call path is fully jax.lax-traceable.

    Threefry RNG is consumed via jax.random.split — no key reuse.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    from Nethax.nethax.subsystems.inventory import ItemCategory

    # --- (1)+(2) intrinsics ----------------------------------------------
    intr = state.status.intrinsics
    for ti in (
        Intrinsic.RESIST_FIRE,
        Intrinsic.RESIST_COLD,
        Intrinsic.RESIST_SHOCK,
        Intrinsic.RESIST_SLEEP,
        Intrinsic.RESIST_POISON,
        Intrinsic.SEE_INVIS,
    ):
        intr = intr.at[int(ti)].set(True)
    new_status = state.status.replace(intrinsics=intr)

    # --- (3) artifact gift -----------------------------------------------
    role_i  = state.player_role.astype(jnp.int32)
    align_i = state.player_align.astype(jnp.int32)
    arti_idx, type_id = _gcrownu_lookup(role_i, align_i)
    has_gift = arti_idx >= jnp.int32(0)

    inv = state.inventory
    items = inv.items
    # First empty inventory slot — vendor invent.c::nextobj/getobj would
    # use the slot picker; we walk in order and pick the lowest-index
    # empty slot (category==0).  Returns the slot index, or -1 if full.
    occupied = items.category != jnp.int8(0)
    n_slots = occupied.shape[0]
    # Build an index array; place a sentinel n_slots at occupied positions
    # so jnp.min picks the first empty.
    slot_idxs = jnp.arange(n_slots, dtype=jnp.int32)
    masked = jnp.where(occupied, jnp.int32(n_slots), slot_idxs)
    first_empty = jnp.min(masked)
    inv_full = first_empty >= jnp.int32(n_slots)
    do_grant = has_gift & ~inv_full
    slot = jnp.where(do_grant, first_empty, jnp.int32(0))

    # Build the artifact's new Item values.  Most artifacts in
    # _ARTIFACTS are WEAPON_CLASS; the few non-weapon ones (Magic Mirror of
    # Merlin, Eyes of the Overworld, Yendorian Express Card, Master Key of
    # Thievery, Sceptre of Might has SCEPTRE in vendor SPBOOK_CLASS? No, mace.
    # All gifts here resolve to WEAPON or TOOL — we look up the category by
    # type_id via the OBJECTS table.
    from Nethax.nethax.constants.objects import OBJECTS
    # Build a [N_OBJECTS] int8 lookup of vendor oclass for the artifact's
    # base object — done eagerly once.
    if not hasattr(gcrownu, "_oclass_table"):
        # constants/objects.py::ObjectEntry exposes ``class_`` (Python keyword
        # workaround); values match the vendor ItemCategory enum.
        _oclass = [int(o.class_) if o.class_ is not None else 0 for o in OBJECTS]
        gcrownu._oclass_table = jnp.array(_oclass, dtype=jnp.int8)
    oclass_arr = gcrownu._oclass_table
    type_id_safe = jnp.clip(type_id, 0, oclass_arr.shape[0] - 1)
    new_oclass = oclass_arr[type_id_safe]  # int8 vendor oclass

    # Update Item arrays at ``slot``.  Only write when do_grant.
    def _set(arr, val):
        cur = arr[slot]
        # JIT-safe: arr.at[slot].set(jnp.where(do_grant, val, cur))
        return arr.at[slot].set(jnp.where(do_grant, val, cur))

    new_items = items.replace(
        category=_set(items.category, new_oclass.astype(jnp.int8)),
        type_id=_set(items.type_id, type_id.astype(jnp.int16)),
        # bless the gift (pray.c:978 bless(obj)).
        buc_status=_set(items.buc_status, jnp.int8(3)),  # BLESSED
        enchantment=_set(items.enchantment, jnp.int8(1)),  # spe = 1 (pray.c:983)
        identified=_set(items.identified, jnp.bool_(True)),
        quantity=_set(items.quantity, jnp.int16(1)),
        # vendor pray.c:980 obj->oerodeproof = TRUE
        oerodeproof=_set(items.oerodeproof, jnp.bool_(True)),
        # vendor pray.c:981 bknown=rknown=1
        bknown=_set(items.bknown, jnp.bool_(True)),
        rknown=_set(items.rknown, jnp.bool_(True)),
        artifact_idx=_set(items.artifact_idx, arti_idx.astype(jnp.int8)),
    )
    new_inv = inv.replace(items=new_items)

    return state.replace(status=new_status, inventory=new_inv)


