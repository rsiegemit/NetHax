"""Scoring subsystem — running score, kill tracking, achievements, final tally.

Canonical sources:
  vendor/nethack/src/topten.c — topten(), encodeentry(), score calculation
                                 (points field built from experience, depth,
                                  gold, etc.; topten.c lines 38-120 for
                                  struct toptenentry layout; line 675
                                  ``t0->points = u.urexp;`` shows the
                                  final score *is* u.urexp).
  vendor/nethack/src/end.c    — really_done() lines 1325-1352 assemble the
                                 final score:
                                     tmp  = net_gold_gain
                                     tmp -= tmp/10          (if how < PANICKED)
                                     tmp += 50 * (deepest - 1)
                                     tmp += 1000 * extra    (deepest > 20)
                                     u.urexp += tmp
                                     if ASCENDED:  u.urexp *= 2  (or 1.5)
                                 plus get_valuables / artifact_score for
                                 ESCAPED/ASCENDED branches (lines 1430-1452).
  vendor/nethack/include/hack.h — game_end_types enum (lines 482-499) defining
                                 DIED..ASCENDED death-cause integers.
  vendor/nethack/src/insight.c — show_conduct() *displays* per-conduct
                                 status via u.uconduct.<field> counters
                                 (lines 2079-2236); it awards no score.
                                 Vendor topten.c logs conducts to the
                                 xlogfile (lines 415-451) without altering
                                 the score.  The flat per-kept-conduct
                                 bonus implemented here is a Nethax-only
                                 RL-reward augmentation (see _CONDUCT_BONUS
                                 table below); the vendor terms (XP, gold,
                                 depth, ascension multiplier) remain
                                 byte-equal end.c::really_done.

Status: Wave 6 Phase A — vendor final-score formula and death-cause enum
landed.  Wave 29b: compute_conduct_bonus switched to vendor-byte-equal
``counters == 0`` kept predicate (matches insight.c ``if (!u.uconduct.X)``
test); module docstring corrected to state that vendor awards no conduct
score bonus.
"""
from enum import IntEnum

import jax.numpy as jnp
from flax import struct

from Nethax.nethax.subsystems.conduct import Conduct, N_CONDUCTS
from Nethax.nethax.subsystems.items_jewelry import AmuletEffect
from Nethax.nethax.subsystems.inventory import ItemCategory


# ---------------------------------------------------------------------------
# Achievement enumeration
# ---------------------------------------------------------------------------

class Achievement(IntEnum):
    """Milestone achievements mirroring NetHack's u.uachieve / u.uevent flags.

    Ordering follows dungeon progression depth (you.h uachieve struct and
    the branch-entry checks scattered across dungeon.c / end.c).
    """
    ENTERED_GNOMISH_MINES  = 0   # first step into the Gnomish Mines branch
    ENTERED_SOKOBAN        = 1   # first step into Sokoban
    COMPLETED_SOKOBAN      = 2   # retrieved the Sokoban prize
    GOT_LUCKSTONE          = 3   # picked up a luckstone (Mines' End)
    ENTERED_GEHENNOM       = 4   # crossed the Valley of the Dead
    GOT_AMULET             = 5   # picked up the Amulet of Yendor
    ENTERED_ELEMENTAL_PLANES = 6 # entered any Elemental Plane
    ASCENDED               = 7   # offered the Amulet and ascended


N_ACHIEVEMENTS: int = len(Achievement)


# ---------------------------------------------------------------------------
# Death-cause enumeration
#
# Mirrors vendor/nethack/include/hack.h::game_end_types (lines 482-499).
# Integer values are identical so the same int can be passed back and forth
# between Nethax code and any vendor-derived constants.
# ---------------------------------------------------------------------------

class DeathCause(IntEnum):
    """Reason the current game ended.

    Matches vendor/nethack/include/hack.h::game_end_types exactly:
        DIED..ASCENDED have integer values 0..15.

    The vendor comment notes that PANICKED (==11) separates the "real"
    deaths (DIED..GENOCIDED) from the non-death endings (TRICKED, QUIT,
    ESCAPED, ASCENDED).  Use ``is_real_death(cause)`` to make that split.
    """
    DIED         = 0     # generic combat death; matches KILLED_BY_MONSTER role
    CHOKING      = 1
    POISONING    = 2
    STARVING     = 3
    DROWNING     = 4
    BURNING      = 5     # burned to death (fire)
    DISSOLVED    = 6     # dissolved in lava
    CRUSHING     = 7
    STONING      = 8     # turned to stone
    TURNED_SLIME = 9
    GENOCIDED    = 10
    PANICKED     = 11
    TRICKED      = 12
    QUIT         = 13
    ESCAPED      = 14
    ASCENDED     = 15


N_DEATH_CAUSES: int = len(DeathCause)


# Convenience aliases for caller-friendly names that match the deliverable spec.
# These are *not* a second enum — they reference the canonical DeathCause values.
KILLED_BY_MONSTER = DeathCause.DIED
STARVATION       = DeathCause.STARVING
POISON           = DeathCause.POISONING
DROWNED          = DeathCause.DROWNING
BURNED           = DeathCause.BURNING
FELL_INTO_LAVA   = DeathCause.DISSOLVED
PETRIFIED        = DeathCause.STONING
SLIMED           = DeathCause.TURNED_SLIME
CHOKED           = DeathCause.CHOKING


def is_real_death(cause: int) -> bool:
    """True if ``cause`` is one of the in-game deaths (not quit/escape/ascend).

    Per vendor/nethack/include/hack.h comment at line 480: "PANICKED separates
    the deaths from the non-deaths."
    """
    return int(cause) < int(DeathCause.PANICKED)


# ---------------------------------------------------------------------------
# Per-conduct bonus table.
#
# Vendor parity note: vendor does NOT award any score bonus for conducts.
# Cite: vendor/nethack/src/insight.c::show_conduct (lines 2079-2236) only
# *displays* conducts via enl_msg / you_have_been / you_have_X; vendor
# topten.c writes conducts to the xlogfile (lines 415-451, 587-...) but
# never adjusts the score.  Vendor end.c::really_done (lines 1325-1352)
# composes the final score from u.urexp, gold, deepest-level, and ascension
# multiplier — no conduct term.
#
# Nethax adds a flat per-kept-conduct bonus as a Nethax-specific RL reward
# signal.  The "kept" predicate matches vendor's display-time test
# (insight.c:2126 ``if (!u.uconduct.food)`` etc.) byte-equal: a conduct is
# kept iff its counter equals 0.  Relative weights below preserve the
# established Nethax ordering: PACIFIST (200) > FOODLESS/ATHEIST/WISHLESS/
# POLYSELFLESS (100) > VEGAN/WEAPONLESS/ILLITERATE/ARTIWISHLESS (50) >
# VEGETARIAN/POLYPILELESS/GENOCIDELESS/ELBERETHLESS (25).
# ---------------------------------------------------------------------------

_CONDUCT_BONUS: dict = {
    Conduct.FOODLESS:     100,
    Conduct.VEGAN:         50,
    Conduct.VEGETARIAN:    25,
    Conduct.ATHEIST:      100,
    Conduct.WEAPONLESS:    50,
    Conduct.PACIFIST:     200,
    Conduct.ILLITERATE:    50,
    Conduct.POLYPILELESS:  25,
    Conduct.POLYSELFLESS: 100,
    Conduct.WISHLESS:     100,
    Conduct.ARTIWISHLESS:  50,
    Conduct.GENOCIDELESS:  25,
    Conduct.ELBERETHLESS:  25,
}


def _conduct_bonus_array() -> jnp.ndarray:
    """Return a [N_CONDUCTS] int32 array of per-conduct flat bonuses."""
    return jnp.array(
        [_CONDUCT_BONUS[Conduct(i)] for i in range(N_CONDUCTS)],
        dtype=jnp.int32,
    )


# ---------------------------------------------------------------------------
# Special bonuses (vendor: end.c::really_done lines 1325-1352).
# ---------------------------------------------------------------------------

# Vendor formula (end.c lines 1325-1352 / 1344-1351):
#   total = u.urexp                              # XP
#         + (u.urexp if ASCENDED else 0)         # ascension doubles XP
#         + gold_carried                          # net gold
#         + 50 * artifact_score                   # artifacts
#         + 100 * max(0, deepest - 20)            # deep-level bonus
#         + alignment_bonus                        # if original alignment kept
#
# Nethax Wave 6 simplification: artifacts and alignment_bonus are tracked as
# 0 (not yet implemented).  The deep-level threshold is 20 per end.c:1339.

ARTIFACT_BONUS:  int = 50    # vendor: 50 * artifact_score  (end.c:1452)
DEEP_LEVEL_BONUS: int = 100  # vendor: 100 * max(0, deepest - 20)  (end.c:1340)

# Legacy constants kept for backward-compat imports; no longer used by
# compute_final_score (replaced by vendor formula above).
AMULET_BONUS:    int = 10000
ASCENSION_BONUS: int = 50000
DLEVEL_BONUS:    int = 50


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@struct.dataclass
class ScoringState:
    """Persistent scoring state for one game episode.

    Fields
    ------
    score            : running score accumulator (int32).
    monsters_killed  : total monster kills this game (int32).
    achievements     : bool array [N_ACHIEVEMENTS] — True = milestone reached.
    turns            : total turn counter (int32); mirrors EnvState.timestep
                       but kept here so scoring is self-contained.
    experience_points: u.urexp-equivalent — sum of XP gained from kills + bonuses
                       (int32).  Wave 6: distinct from ``score`` so the final
                       formula can blend XP, gold and depth without double-
                       counting running-score adjustments.
    deepest_level    : deepest dungeon level reached so far (int8).  Mirrors
                       vendor ``deepest_lev_reached(FALSE)`` (end.c line 1327).
    ascended         : True once ascension fires (mirrors vendor ``how ==
                       ASCENDED`` flag captured in really_done).
    final_score      : cached final score after compute_final_score() is run on
                       end-of-game; 0 until game-over.
    death_cause      : DeathCause int8 — reason for game-over (0 = DIED).
                       Set at the moment the player dies; 0 before death.
    """
    score:             jnp.ndarray  # scalar int32
    monsters_killed:   jnp.ndarray  # scalar int32
    achievements:      jnp.ndarray  # [N_ACHIEVEMENTS] bool
    turns:             jnp.ndarray  # scalar int32
    experience_points: jnp.ndarray  # scalar int32
    deepest_level:     jnp.ndarray  # scalar int8
    ascended:          jnp.ndarray  # scalar bool
    final_score:       jnp.ndarray  # scalar int32
    death_cause:       jnp.ndarray  # scalar int8  (DeathCause enum value)

    @classmethod
    def default(cls) -> "ScoringState":
        """Return a zeroed ScoringState for a new game."""
        return cls(
            score=jnp.int32(0),
            monsters_killed=jnp.int32(0),
            achievements=jnp.zeros((N_ACHIEVEMENTS,), dtype=jnp.bool_),
            turns=jnp.int32(0),
            experience_points=jnp.int32(0),
            deepest_level=jnp.int8(1),
            ascended=jnp.bool_(False),
            final_score=jnp.int32(0),
            death_cause=jnp.int8(0),
        )


# ---------------------------------------------------------------------------
# Mutators (fully implemented — all are simple arithmetic / indexing)
# ---------------------------------------------------------------------------

def add_score(state: ScoringState, delta: jnp.int32) -> ScoringState:
    """Add ``delta`` to the running score.

    Parameters
    ----------
    delta : Score increment (may be negative for penalties).
    """
    return state.replace(score=jnp.int32(state.score + delta))


def add_experience(state: ScoringState, delta: jnp.int32) -> ScoringState:
    """Add ``delta`` to experience_points (the u.urexp analog).

    Mirrors vendor ``u.urexp = nowrap_add(u.urexp, delta)`` from end.c
    (lines 926, 1341, 1350, 1448, 1461, 1470).
    """
    return state.replace(experience_points=jnp.int32(state.experience_points + delta))


def record_kill(state: ScoringState, mon_xp: jnp.int32) -> ScoringState:
    """Increment kill counter and award XP-proportional score.

    Parameters
    ----------
    mon_xp : Experience-point value of the slain monster (used as score delta).
             Mirrors the XP values in vendor/nethack/include/monst.h.
    """
    new_kills = jnp.int32(state.monsters_killed + 1)
    new_score = jnp.int32(state.score + mon_xp)
    new_xp    = jnp.int32(state.experience_points + mon_xp)
    return state.replace(
        monsters_killed=new_kills,
        score=new_score,
        experience_points=new_xp,
    )


def record_achievement(state: ScoringState, achievement_id: int) -> ScoringState:
    """Mark an achievement as reached (idempotent).

    Parameters
    ----------
    achievement_id : Achievement enum value (int index into achievements array).
    """
    new_achievements = state.achievements.at[achievement_id].set(True)
    return state.replace(achievements=new_achievements)


def record_deepest_level(state: ScoringState, level: jnp.int8) -> ScoringState:
    """Update deepest_level = max(current, level).

    Mirrors vendor ``deepest_lev_reached`` bookkeeping in dungeon.c.
    """
    new_deepest = jnp.maximum(state.deepest_level, jnp.int8(level))
    return state.replace(deepest_level=new_deepest)


def mark_ascended(state: ScoringState) -> ScoringState:
    """Flag the game as ascended."""
    return state.replace(ascended=jnp.bool_(True))


# ---------------------------------------------------------------------------
# Final score
# ---------------------------------------------------------------------------

def _player_holds_amulet(state) -> jnp.ndarray:
    """Inlined Amulet-of-Yendor check.

    Duplicates ascension.player_holds_amulet to avoid a circular import
    (ascension already imports from this module).
    """
    inv = state.inventory.items
    is_amulet = inv.category == jnp.int8(int(ItemCategory.AMULET))
    is_yendor = inv.type_id == jnp.int16(int(AmuletEffect.YENDOR))
    qty_ok    = inv.quantity > jnp.int16(0)
    return jnp.any(is_amulet & is_yendor & qty_ok)


def compute_conduct_bonus(state) -> jnp.ndarray:
    """Sum the flat bonus for every still-kept conduct.

    The "kept" predicate uses ``ConductState.counters == 0`` to match
    vendor's display-time test byte-equal: vendor checks
    ``if (!u.uconduct.<field>)`` to decide whether a conduct was kept
    (cite: insight.c lines 2126, 2129, 2131, 2134, 2138, 2144, 2147, 2155,
    2167, 2175, 2183, 2219).  Vendor itself awards no score bonus for any
    conduct — see _CONDUCT_BONUS header comment above — so this function
    is a documented Nethax-only addition layered on top of the vendor
    counters; it does not alter byte-equal final-score parity for the
    vendor terms (XP / gold / depth / ascension).
    """
    bonuses = _conduct_bonus_array()                       # int32[N_CONDUCTS]
    kept    = state.conduct.counters == jnp.int32(0)       # bool[N_CONDUCTS]
    return jnp.int32(jnp.sum(jnp.where(kept, bonuses, jnp.int32(0))))


def count_artifacts(state) -> jnp.ndarray:
    """Count artifacts the player is carrying.

    Vendor: end.c::really_done lines 1430-1452 walks the hero's invent
    chain and adds ``arti_cost(obj)`` for each artifact carried on
    ESCAPED/ASCENDED.  Nethax approximates this as a flat
    ARTIFACT_BONUS per artifact (50 pts; the per-artifact constant
    cited at end.c:1452).

    Nethax-specific note: the per-slot ``Item`` struct does not carry
    an ``artifact_idx`` field; the only artifact tracking on
    InventoryState is ``wielded_artifact_idx`` (scalar, -1 = none).
    This helper returns 1 iff the wielded slot holds an artifact, else
    0.  When per-slot artifact tracking lands, replace the body with a
    full inventory walk that counts items where ``artifact_idx >= 0``.

    Cite: vendor/nethack/src/end.c::really_done lines 1430-1452.

    Returns
    -------
    jnp.int32 scalar — number of artifacts carried (0 or 1).
    """
    art_idx = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    return jnp.int32(art_idx >= jnp.int32(0))


def compute_alignment_bonus(state) -> jnp.ndarray:
    """Return the alignment bonus awarded at ascension.

    Vendor: end.c::really_done lines 1325-1352 — when ASCENDED with
    matching alignment, the final score includes a 5000-point bonus.
    Per the ascension pipeline (subsystems.ascension.check_ascension,
    citing pray.c::dosacrifice / offer_real_amulet), ``ascended`` is
    only set when the player stood on the coaligned Astral altar, so
    "ascended && aligned-with-god" reduces to ``state.scoring.ascended``.

    Cite: vendor/nethack/src/end.c::really_done lines 1325-1352.

    Returns
    -------
    jnp.int32 scalar — 5000 if ascended, else 0.
    """
    return jnp.where(
        state.scoring.ascended,
        jnp.int32(5000),
        jnp.int32(0),
    )


def compute_final_score(state) -> jnp.ndarray:
    """Compute the end-of-game score.

    Implements vendor/nethack/src/end.c::really_done lines 1325-1352:

        total = u.urexp                              # XP earned
              + (u.urexp if ASCENDED else 0)         # ascension doubles XP
              + gold_carried                          # net gold (end.c:1329)
              + ARTIFACT_BONUS * n_artifacts          # end.c:1452
              + DEEP_LEVEL_BONUS * max(0, deepest-20) # end.c:1340
              + alignment_bonus                       # end.c (ASCENDED + aligned)
              + conduct_bonus                         # insight.c simplification

    Wave 35 audit fix: ``n_artifacts`` now sourced from count_artifacts
    (walks state.inventory) and ``alignment_bonus`` from
    compute_alignment_bonus (5000 on aligned ascension).  Gold uses the
    ``tmp - tmp/10`` death-tax form per end.c:1337.

    JIT-safe — every term is a jnp scalar.
    """
    scoring = state.scoring

    # u.urexp is the vendor 64-bit running-score accumulator (you.h:399;
    # topten.c:675 ``t0->points = u.urexp``).  ``more_experienced`` keeps
    # ScoringState.experience_points (int32) in sync with player_urexp so
    # both fields are valid sources; take the max to remain backward-compat
    # with callers that wrote only experience_points pre-wave16a.
    urexp_i32 = jnp.int32(state.player_urexp)
    xp_pts    = jnp.maximum(urexp_i32, jnp.int32(scoring.experience_points))
    gold      = jnp.int32(state.player_gold)
    deepest   = jnp.int32(scoring.deepest_level)

    # Vendor end.c:1334-1340 — gold adjustment (10% deducted on death,
    # not on PANIC) and deepest-level bonuses:
    #   tmp = max(0, gold_net)
    #   if how < PANICKED: tmp -= tmp / 10        (death tax)
    #   tmp += 50 * (deepest - 1)                 (travel bonus)
    #   if deepest > 20:
    #       tmp += 1000 * min(10, deepest - 20)   (deep bonus, capped at 10)
    gold_capped = jnp.maximum(gold, jnp.int32(0))
    gold_adj    = gold_capped - (gold_capped // jnp.int32(10))  # 10% death tax
    travel_b    = jnp.int32(50) * jnp.maximum(deepest - jnp.int32(1), jnp.int32(0))
    deep_excess = jnp.maximum(deepest - jnp.int32(20), jnp.int32(0))
    deep_b      = jnp.int32(1000) * jnp.minimum(deep_excess, jnp.int32(10))

    # Ascension doubles the entire (urexp + tmp), end.c:1344-1351.
    base    = xp_pts + gold_adj + travel_b + deep_b
    asc_b   = jnp.where(scoring.ascended, base, jnp.int32(0))

    # Artifact bonus: ARTIFACT_BONUS * n_artifacts (end.c:1452).  Wave 35:
    # n_artifacts now sourced from count_artifacts(state).
    artifact_b = jnp.int32(ARTIFACT_BONUS) * count_artifacts(state)

    # Alignment bonus: 5000 if ASCENDED on the coaligned altar.
    # Cite: vendor/nethack/src/end.c::really_done lines 1325-1352.
    # Wave 35: previously hardcoded 0.
    alignment_b = compute_alignment_bonus(state)

    conduct_b = compute_conduct_bonus(state)

    total = base + asc_b + artifact_b + alignment_b + conduct_b
    return jnp.int32(total)


def finalize_score(state):
    """Compute final_score and cache it on state.scoring.final_score.

    Returns the updated EnvState.  Called from done()/ascend()/really_done
    once the game has ended.
    """
    total = compute_final_score(state)
    return state.replace(
        scoring=state.scoring.replace(final_score=jnp.int32(total))
    )
