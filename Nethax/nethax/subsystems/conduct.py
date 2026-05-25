"""Conduct subsystem — track voluntary self-restrictions throughout the game.

Canonical sources:
  vendor/nethack/src/insight.c  — show_conduct(), doconduct(), per-conduct
                                   display logic and violation counters
                                   (u.uconduct.food, .unvegan, .unvegetarian,
                                    .gnostic, .weaphit, .killer, .literate,
                                    .polypiles, .polyselfs, .wishes, .wisharti,
                                    .sokocheat — insight.c lines ~2079-2230)

Status: All 13 conducts are wired at the action-trigger sites listed
below.  Conduct enum + violations array + violation helpers all live in
this module; per-conduct triggers fire from their owning subsystems.

Wired (Wave 4):
    FOODLESS      — action_dispatch._handle_eat (any eat marks)
    VEGAN         — action_dispatch._handle_eat (animal-product material)
    VEGETARIAN    — action_dispatch._handle_eat (FLESH material)
    ATHEIST       — prayer.handle_pray (any prayer attempt)
    WEAPONLESS    — inventory.handle_wield (any wielded weapon)
    PACIFIST      — combat.melee_attack (monster killed branch)
    ILLITERATE    — items_scrolls.handle_read + items_spellbooks.handle_read_spellbook
    POLYSELFLESS  — polymorph.polymorph_player

Wired (Wave 5):
    POLYPILELESS  — traps.poly_pile_effect
    GENOCIDELESS  — items_scrolls.apply_genocide
    ELBERETHLESS  — engrave.handle_engrave

Wired (Wave 6 Phase B):
    WISHLESS      — wish.grant_wish (any wish)
    ARTIWISHLESS  — wish.grant_wish (artifact wish only)

All 13 conducts now have triggers; the enum is complete.
"""
from enum import IntEnum

import jax
import jax.numpy as jnp
from flax import struct


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
class Conduct(IntEnum):
    """Voluntary conduct restrictions, mirroring insight.c show_conduct order."""

    FOODLESS = 0       # never ate anything
    VEGAN = 1          # no animal products
    VEGETARIAN = 2     # no meat
    ATHEIST = 3        # no prayer/sacrifice/altar interaction
    WEAPONLESS = 4     # never hit a monster with a wielded weapon
    PACIFIST = 5       # never killed any monster
    ILLITERATE = 6     # never read a scroll or spellbook
    POLYPILELESS = 7   # never polymorphed an item
    POLYSELFLESS = 8   # never changed own form
    WISHLESS = 9       # never used a wish
    ARTIWISHLESS = 10  # never wished for an artifact
    GENOCIDELESS = 11  # never genocided a species
    ELBERETHLESS = 12  # never engraved Elbereth


N_CONDUCTS = len(Conduct)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@struct.dataclass
class ConductState:
    """Per-conduct violation counters for the current game.

    Mirrors vendor ``struct u_conduct`` (vendor/nethack/include/you.h:147-167)
    which stores one ``long`` counter per conduct ("number of times..." a
    challenge has been violated).  Counter values are used by vendor display
    code (insight.c::show_conduct, lines ~2141, 2150, 2170, 2178, 2186-2202)
    and by xlog output (topten.c:385-386 wish_cnt / arti_wish_cnt).

    Fields
    ------
    counters   : int32 array of shape [N_CONDUCTS]
                 Per-conduct increment counter (vendor: u.uconduct.<field>++).
                 0 = conduct still kept; >0 = number of violations.
    violations : bool array of shape [N_CONDUCTS]
                 Derived ``counters > 0`` mask kept explicit for back-compat
                 with existing callers (scoring, scoreboard).  Always equals
                 ``counters > 0``; maintained by ``increment_counter``.
    """

    counters: jnp.ndarray    # [N_CONDUCTS]  int32 — vendor long counter
    violations: jnp.ndarray  # [N_CONDUCTS]  bool  — derived counters>0 mask

    @classmethod
    def default(cls) -> "ConductState":
        """Return a clean ConductState (all conducts intact) for a new game."""
        return cls(
            counters=jnp.zeros((N_CONDUCTS,), dtype=jnp.int32),
            violations=jnp.zeros((N_CONDUCTS,), dtype=jnp.bool_),
        )


# ---------------------------------------------------------------------------
# Violation helpers
# ---------------------------------------------------------------------------
def violate(state: ConductState, conduct: int) -> ConductState:
    """Bump the counter and set the derived violations bit on a ConductState.

    Mirrors ``u.uconduct.<field>++`` from vendor/nethack/src/insight.c (the
    counter); also maintains the derived ``violations`` mask.
    Cite: vendor/nethack/include/you.h:147-167 ``struct u_conduct``.
    """
    new_counters = state.counters.at[conduct].add(jnp.int32(1))
    new_violations = state.violations.at[conduct].set(True)
    return state.replace(counters=new_counters, violations=new_violations)


def increment_counter(env_state, conduct_idx: int):
    """Bump counters[conduct_idx] by 1 and set violations[conduct_idx]=True.

    Mirrors vendor ``u.uconduct.<field>++`` increment-on-violation pattern
    (e.g. eat.c, pray.c, polyself.c, wish-grant paths).  JAX-safe via .at[].
    Cite: vendor/nethack/include/you.h:147-167 ``struct u_conduct``;
          vendor/nethack/src/insight.c::show_conduct lines ~2141, 2170, 2178,
          2186 (counter values consumed for display).
    """
    cur_counts = env_state.conduct.counters
    cur_vio    = env_state.conduct.violations
    new_counters   = cur_counts.at[conduct_idx].add(jnp.int32(1))
    new_violations = cur_vio.at[conduct_idx].set(True)
    return env_state.replace(
        conduct=env_state.conduct.replace(
            counters=new_counters, violations=new_violations
        )
    )


# Back-compat alias: existing callers say `mark_violated`.  Forward to
# `increment_counter` so every violation also bumps the vendor counter.
mark_violated = increment_counter


def increment_counter_if(env_state, conduct_idx: int, condition):
    """Conditionally bump counters[conduct_idx] (and set violations) JIT-safely.

    ``condition`` may be a traced bool.  When True: bump counter by 1 and set
    the derived violations bit.  When False: leave both unchanged.
    Cite: vendor/nethack/include/you.h:147-167 ``struct u_conduct``.
    """
    cur_counts = env_state.conduct.counters
    cur_vio    = env_state.conduct.violations
    inc        = jnp.where(condition, jnp.int32(1), jnp.int32(0))
    new_counters   = cur_counts.at[conduct_idx].add(inc)
    new_violations = cur_vio.at[conduct_idx].set(cur_vio[conduct_idx] | condition)
    return env_state.replace(
        conduct=env_state.conduct.replace(
            counters=new_counters, violations=new_violations
        )
    )


# Back-compat alias: existing callers say `mark_violated_if`.
mark_violated_if = increment_counter_if


# ---------------------------------------------------------------------------
# Food-material lookup for VEGAN / VEGETARIAN
#
# Wave 6 Phase B: pull oc_material straight from the canonical OBJECTS table
# (Nethax.nethax.constants.objects).  Earlier waves used a hand-rolled
# type_id → material table.  Predicates use vendor MAT_FLESH (meat),
# MAT_WAX (dairy/cheese), MAT_VEGGY (vegan-safe).
#
# Cite: vendor/nethack/include/objclass.h::obj_material_types
#       vendor/nethack/src/eat.c::eatcorpse  (VEGAN/VEGETARIAN tagging).
# ---------------------------------------------------------------------------
# Material codes (mirrors constants/objects.py::Material).  Pulled from the
# vendor MAT_* enum so callers don't need to import Material.
_MAT_NO_MATERIAL = 0
_MAT_LIQUID = 1
_MAT_WAX    = 2   # dairy proxy — vendor: tallow / wax / cheese-like
_MAT_VEGGY  = 3
_MAT_FLESH  = 4


def _build_object_material_table() -> jnp.ndarray:
    """Return int8[NUM_OBJECTS] of oc_material per OBJECTS entry.

    Wave 6 Phase B: replaces the legacy `_FOOD_MATERIAL_BY_TYPE_ID` hardcode.
    Entries beyond ``len(OBJECTS)`` are filled with NO_MATERIAL so out-of-
    range type_id lookups silently return "no material" (== vegan-safe).
    """
    from Nethax.nethax.constants.objects import OBJECTS, ObjectClass
    # Build a list of (type_id, material) for live OBJECTS entries.
    vals = []
    for o in OBJECTS:
        # Spec override: potions store the conceptual material (LIQUID) for
        # the conduct system even though vendor objects.h tags the container
        # as GLASS.  This keeps the type_id→material lookup aligned with the
        # `obj.h::MAT_*` enum used by the conduct subsystem.
        if int(o.class_) == int(ObjectClass.POTION_CLASS):
            vals.append(_MAT_LIQUID)
        else:
            vals.append(int(o.material))
    return jnp.array(vals, dtype=jnp.int8)


# Eager build at module load — avoids any tracer-in-init issues.
_OBJECT_MATERIAL_TABLE: jnp.ndarray = _build_object_material_table()


def food_material_for_type_id(type_id) -> jnp.ndarray:
    """Return oc_material for inventory ``Item.type_id`` (JIT-safe).

    Reads ``OBJECTS[type_id].material`` via a static int8 table built at
    import.  Out-of-range type_ids return ``NO_MATERIAL`` (treated as
    vegan-safe by the predicates below).
    Cite: vendor/nethack/include/objects.h ``oc_material`` column.
    """
    tab = _OBJECT_MATERIAL_TABLE
    tid = jnp.clip(jnp.int32(type_id), 0, tab.shape[0] - 1)
    return tab[tid].astype(jnp.int32)


def is_meat_material(material) -> jnp.ndarray:
    """True iff material is meat (FLESH); violates VEGETARIAN."""
    return jnp.equal(jnp.int32(material), jnp.int32(_MAT_FLESH))


def is_animal_material(material) -> jnp.ndarray:
    """True iff material is animal-derived (FLESH or WAX); violates VEGAN.

    Mirrors vendor src/eat.c VEGAN logic — meat AND animal-by-products
    (cheese / honey-wax / leather offal) all violate VEGAN.
    """
    m = jnp.int32(material)
    return (jnp.equal(m, jnp.int32(_MAT_FLESH))
            | jnp.equal(m, jnp.int32(_MAT_WAX)))


def step(state: ConductState, rng: jax.Array) -> ConductState:
    """Per-turn no-op — conducts only change on explicit violation events."""
    return state


# ---------------------------------------------------------------------------
# Wave 6 Phase A — end-of-game conduct scoreboard + scoring bonuses
#
# Mirrors vendor/nethack/src/end.c::list_conducts (end-of-game preserved
# conducts) and the per-conduct bonus contribution to compute_final_score.
# ---------------------------------------------------------------------------

# Display order for conduct_scoreboard, per end.c::list_conducts:
#   1. ATHEIST          2. WEAPONLESS       3. PACIFIST
#   4. ILLITERATE       5. POLYPILELESS     6. POLYSELFLESS
#   7. WISHLESS         8. ARTIWISHLESS     9. GENOCIDELESS
#  10. ELBERETHLESS    11. FOODLESS        12. VEGAN          13. VEGETARIAN
_SCOREBOARD_ORDER = (
    Conduct.ATHEIST,
    Conduct.WEAPONLESS,
    Conduct.PACIFIST,
    Conduct.ILLITERATE,
    Conduct.POLYPILELESS,
    Conduct.POLYSELFLESS,
    Conduct.WISHLESS,
    Conduct.ARTIWISHLESS,
    Conduct.GENOCIDELESS,
    Conduct.ELBERETHLESS,
    Conduct.FOODLESS,
    Conduct.VEGAN,
    Conduct.VEGETARIAN,
)

# Per-conduct line widths for the byte variant.  64 is plenty for the longest
# preserved-conduct string ("You preserved POLYSELFLESS.") plus NUL pad.
_SCOREBOARD_LINE_BYTES = 64


def conduct_scoreboard(state) -> list[str]:
    """Return list of strings describing which conducts the player preserved.

    For each Conduct value 0-12, check ``state.conduct.violations[i]``:
      - if False (preserved): emit ``"You preserved CONDUCT_NAME."``
      - if True (violated): skip.

    Returns lines in vendor end.c::list_conducts order.
    Cite: vendor/nethack/src/end.c::list_conducts.
    """
    violations = state.conduct.violations
    lines: list[str] = []
    for conduct in _SCOREBOARD_ORDER:
        if not bool(violations[int(conduct)]):
            lines.append(f"You preserved {conduct.name}.")
    return lines


def conduct_scoreboard_bytes(state) -> jnp.ndarray:
    """JIT-safe variant: int8[13, 64] — one fixed-width byte string per conduct.

    Each row corresponds to the same index in ``_SCOREBOARD_ORDER``.  Row
    contents are the ASCII bytes of ``"You preserved CONDUCT_NAME."`` padded
    with NUL (0x00) when the conduct is preserved, or all-NUL when violated.
    """
    violations = state.conduct.violations
    rows = []
    for conduct in _SCOREBOARD_ORDER:
        line = f"You preserved {conduct.name}."
        encoded = line.encode("ascii")
        if len(encoded) > _SCOREBOARD_LINE_BYTES:
            encoded = encoded[:_SCOREBOARD_LINE_BYTES]
        padded = encoded + b"\x00" * (_SCOREBOARD_LINE_BYTES - len(encoded))
        # mask out when violated
        line_arr = jnp.frombuffer(padded, dtype=jnp.uint8).astype(jnp.int8)
        preserved = jnp.logical_not(violations[int(conduct)])
        rows.append(jnp.where(preserved, line_arr, jnp.zeros_like(line_arr)))
    return jnp.stack(rows, axis=0)


# Per-conduct score bonuses.  Index-aligned with Conduct enum (0..12).
# Cite: vendor/nethack/src/end.c::compute_final_score (conduct contribution).
_CONDUCT_BONUSES = jnp.array(
    [
        100,   # FOODLESS
        50,    # VEGAN
        25,    # VEGETARIAN
        100,   # ATHEIST
        50,    # WEAPONLESS
        200,   # PACIFIST
        50,    # ILLITERATE
        25,    # POLYPILELESS
        100,   # POLYSELFLESS
        100,   # WISHLESS
        50,    # ARTIWISHLESS
        25,    # GENOCIDELESS
        25,    # ELBERETHLESS
    ],
    dtype=jnp.int32,
)


def conduct_score_bonus(state) -> jnp.ndarray:
    """Sum of bonuses for all preserved conducts.

    Returns int32 score contribution.  JIT-safe.
    Cite: vendor/nethack/src/end.c::compute_final_score.
    """
    not_violated = jnp.logical_not(state.conduct.violations)
    total = jnp.sum(_CONDUCT_BONUSES * not_violated.astype(jnp.int32))
    return total.astype(jnp.int32)
