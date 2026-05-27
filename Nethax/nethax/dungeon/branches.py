"""Dungeon branch graph: branch identifiers, topology, and stair linking.

Purpose:
    Defines the multi-branch structure of the NetHack dungeon.  Each Branch
    is a named sub-dungeon (e.g. Gnomish Mines, Sokoban) connected to the
    main dungeon via stairs or portals.  DungeonState tracks the player's
    position within this graph and the generated/linked level counts.

Citation:
    vendor/nethack/include/dungeon.h  — dungeon / d_level / branch structs,
        BR_STAIR / BR_PORTAL / BR_NO_END1 / BR_NO_END2 constants,
        MAXDUNGEON = 16, MAXLEVEL = 32
    vendor/nethack/include/global.h   — MAXDUNGEON = 16, MAXLEVEL = 32,
        MAXNROFROOMS = 40
    vendor/nethack/include/hack.h     — mines_dnum, sokoban_dnum, quest_dnum
    vendor/nethack/src/dungeon.c      — init_dungeon, branch linking logic,
        stronghold / oracle / sanctum / valley level assignment

Wave 2: traverse_stair reads stair_links table; enter_branch updates
        current_branch/current_level; generate_main_branch_l1 produces
        a fully built level (terrain + rooms + stair positions).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import struct

# ---------------------------------------------------------------------------
# Map geometry constants
# These are referenced by every other dungeon module to size JAX arrays.
# ---------------------------------------------------------------------------

MAP_H: int = 21   # canonical NLE / NetHack map height (rows)
MAP_W: int = 80   # canonical NLE / NetHack map width  (cols)

# ---------------------------------------------------------------------------
# Branch / dungeon-count constants
# ---------------------------------------------------------------------------

N_BRANCHES: int = 7  # number of branches with full state-array slots
# Audit-N #7 Commit 7: Branch.LUDIOS = 7 is defined below for vendor-cite
# completeness (dungeon.def line 27).  It is intentionally NOT counted in
# N_BRANCHES yet — bumping that constant would resize every per-branch
# state array in state.py / level_memory.py, which is outside the scope
# of this wave.  Callers that want Ludios entry/spec metadata can read
# the constants directly; the state arrays do not yet have a slot for it.
MAX_LEVELS_PER_BRANCH: int = 32  # MAXLEVEL from vendor/nethack/include/global.h


# ---------------------------------------------------------------------------
# Branch enum
# ---------------------------------------------------------------------------

class Branch(IntEnum):
    """Named dungeon branches.

    Ordinal values are used as indices into per-branch JAX arrays.

    Citation: vendor/nethack/include/dungeon.h, vendor/nethack/src/dungeon.c
    """
    MAIN          = 0
    GNOMISH_MINES = 1
    SOKOBAN       = 2
    QUEST         = 3
    VLAD          = 4
    GEHENNOM      = 5
    ENDGAME       = 6
    # vendor/nle/dat/dungeon.def line 27:
    #   BRANCH: "Fort Ludios" @ (18, 4) portal
    # See _DUNGEON_NUM_LEVELS_VENDOR_SPEC / _BRANCH_ENTRY_VENDOR_SPEC for
    # the (mean, dev) specs.  Not represented in BRANCH_TABLE (length 7).
    LUDIOS        = 7


# ---------------------------------------------------------------------------
# Branch connection type
# Mirrors BR_STAIR / BR_NO_END1 / BR_NO_END2 / BR_PORTAL from dungeon.h
# ---------------------------------------------------------------------------

class BranchConnectionType(IntEnum):
    """How a branch is entered from its parent dungeon.

    Citation: vendor/nethack/include/dungeon.h lines 91-96
        BR_STAIR   = 0  — two-way staircase pair
        BR_NO_END1 = 1  — stair only on parent side
        BR_NO_END2 = 2  — stair only on branch side
        BR_PORTAL  = 3  — magic portal (Quest, Sokoban level 1)
    """
    STAIR    = 0   # BR_STAIR
    NO_END1  = 1   # BR_NO_END1
    NO_END2  = 2   # BR_NO_END2
    PORTAL   = 3   # BR_PORTAL


# ---------------------------------------------------------------------------
# BranchInfo dataclass
# ---------------------------------------------------------------------------

@struct.dataclass
class BranchInfo:
    """Static topology for one branch.

    Fields
    ------
    branch_id : int8
    first_level : int8
    num_levels : int8
    connection_type : int8
    """
    branch_id:       jnp.ndarray  # int8
    first_level:     jnp.ndarray  # int8
    num_levels:      jnp.ndarray  # int8
    connection_type: jnp.ndarray  # int8


# ---------------------------------------------------------------------------
# Canonical branch table
# ---------------------------------------------------------------------------
#
# Audit-N #7 — vendor-faithful (mean, dev) specs landed incrementally.
# Vendor cite: vendor/nle/dat/dungeon.def lines 17-143.
#
# The static BRANCH_TABLE below remains for legacy callers (cross-branch
# cascade tests that hardcode integer level counts).  New code should
# prefer ``sample_branch_table(rng)`` which produces vendor-faithful samples
# via ``rn1(rand, base) = rn2(rand) + base`` (vendor/nle/include/hack.h line 497).
#
# Vendor specs landed in this wave (per dungeon.def):
#
#   MAIN          DUNGEON  "The Dungeons of Doom"      (25, 5)   # base, rand
#   MINES_ENTRY   BRANCH   "The Gnomish Mines"  @ (2, 3)
#   MINES         DUNGEON  "The Gnomish Mines"        (8, 2)
#   ORACLE_LEVEL  LEVEL    "oracle"             @ (5, 5)
#   SOKO_ENTRY    CHAINBR  "Sokoban" "oracle"   + (1, 0) up      # STAIR not PORTAL
#   SOKOBAN       DUNGEON  "Sokoban"                  (4, 0)
#   QUEST_ENTRY   CHAINBR  "The Quest" "oracle" + (6, 2) portal
#   QUEST         DUNGEON  "The Quest"                (5, 2)
#   LUDIOS_ENTRY  BRANCH   "Fort Ludios"        @ (18, 4) portal
#   LUDIOS        DUNGEON  "Fort Ludios"              (1, 0)
#   GEHENNOM      DUNGEON  "Gehennom"                 (20, 5)
#   VLAD_ENTRY    BRANCH   "Vlad's Tower"       @ (9, 5) up      # chained off Gehennom
#   VLAD          DUNGEON  "Vlad's Tower"             (3, 0)
#   ENDGAME       DUNGEON  "The Elemental Planes"     (6, 0)

BRANCH_TABLE: Tuple[BranchInfo, ...] = (
    BranchInfo(
        branch_id=jnp.int8(Branch.MAIN),
        first_level=jnp.int8(1),
        num_levels=jnp.int8(26),
        connection_type=jnp.int8(BranchConnectionType.STAIR),
    ),
    BranchInfo(
        branch_id=jnp.int8(Branch.GNOMISH_MINES),
        first_level=jnp.int8(2),
        num_levels=jnp.int8(5),
        connection_type=jnp.int8(BranchConnectionType.STAIR),
    ),
    BranchInfo(
        branch_id=jnp.int8(Branch.SOKOBAN),
        first_level=jnp.int8(8),
        num_levels=jnp.int8(4),
        connection_type=jnp.int8(BranchConnectionType.PORTAL),
    ),
    BranchInfo(
        branch_id=jnp.int8(Branch.QUEST),
        first_level=jnp.int8(14),
        num_levels=jnp.int8(5),
        connection_type=jnp.int8(BranchConnectionType.PORTAL),
    ),
    BranchInfo(
        branch_id=jnp.int8(Branch.VLAD),
        first_level=jnp.int8(21),
        num_levels=jnp.int8(3),
        # dungeon.def line 55 ``BRANCH: "Vlad's Tower" @ (9, 5) up`` has no
        # explicit branch_type, so dgn_comp.y:322-325 defaults to TBR_STAIR,
        # which correct_branch_type (dungeon.c:415-417) maps to BR_STAIR.
        # The legacy NO_END2 here was a placeholder; sample_branch_table
        # overrides via _BRANCH_CONNECTION_VENDOR_SPEC.
        connection_type=jnp.int8(BranchConnectionType.STAIR),
    ),
    BranchInfo(
        branch_id=jnp.int8(Branch.GEHENNOM),
        first_level=jnp.int8(21),
        num_levels=jnp.int8(16),
        connection_type=jnp.int8(BranchConnectionType.STAIR),
    ),
    BranchInfo(
        branch_id=jnp.int8(Branch.ENDGAME),
        first_level=jnp.int8(27),
        num_levels=jnp.int8(5),
        connection_type=jnp.int8(BranchConnectionType.NO_END1),
    ),
)


# ---------------------------------------------------------------------------
# Vendor-faithful (mean, dev) tuples and JIT-safe sampler
# ---------------------------------------------------------------------------
#
# Audit-N #7 Commit 1 (MAIN): wire ``(base=25, rand=5)`` for the Dungeons of
# Doom.  Vendor cite: vendor/nle/dat/dungeon.def line 17
# ``DUNGEON: "The Dungeons of Doom" "D" (25, 5)`` and vendor sampling rule
# vendor/nle/src/dungeon.c lines 796-800
# ``num_dunlevs = (xchar) rn1(pd.tmpdungeon[i].lev.rand,
#                              pd.tmpdungeon[i].lev.base);``
# with ``rn1(x, y) = rn2(x) + y`` from vendor/nle/include/hack.h line 497.
#
# Each ``(base, rand)`` tuple describes a uniform sample on the closed
# interval ``[base, base + rand - 1]`` (when rand != 0) or the fixed
# constant ``base`` (when rand == 0).  Branch-entry (BRANCH/CHAINBRANCH)
# tuples are interpreted by ``level_range`` in dungeon.c lines 350-382:
# the entry depth lies in ``[parent_depth + base, parent_depth + base + rand - 1]``
# for CHAINBRANCH ("rcouple") or ``[base, base + rand - 1]`` for BRANCH
# ("acouple").
#
# This wave wires only the MAIN dungeon ``num_levels`` spec; subsequent
# commits will land Gehennom, Endgame, Mines, Sokoban, Quest, Vlad, Ludios.

# (base, rand) for ``num_levels`` per DUNGEON entry in dungeon.def.
# rand == 0 ⇒ fixed; rand > 0 ⇒ rn1(rand, base) = rn2(rand) + base.
_DUNGEON_NUM_LEVELS_VENDOR_SPEC: dict = {
    int(Branch.MAIN):          (25, 5),   # dungeon.def line 17
    int(Branch.GEHENNOM):      (20, 5),   # dungeon.def line 51
    int(Branch.GNOMISH_MINES): (8, 2),    # dungeon.def line 71
    int(Branch.QUEST):         (5, 2),    # dungeon.def line 86
    int(Branch.SOKOBAN):       (4, 0),    # dungeon.def line 94 (rand=0 ⇒ fixed)
    int(Branch.VLAD):          (3, 0),    # dungeon.def line 116 (rand=0 ⇒ fixed)
    int(Branch.ENDGAME):       (6, 0),    # dungeon.def line 134 (rand=0 ⇒ fixed)
    int(Branch.LUDIOS):        (1, 0),    # dungeon.def line 106 (rand=0 ⇒ fixed)
}


# (base, rand) for branch-entry depth (BRANCH / CHAINBRANCH @ (b, r)).
# Interpreted via vendor/nle/src/dungeon.c::level_range — BRANCH uses
# "acouple" semantics (absolute in parent dungeon), CHAINBRANCH uses
# "rcouple" (relative to a chain-level's dlevel).
#
# Wired in Commit 3: Mines entry @ (2, 3) into Main.  Future commits
# add Sokoban (chain to oracle), Quest (chain to oracle), Vlad (chain
# to Gehennom), Ludios (acouple in Main).
_BRANCH_ENTRY_VENDOR_SPEC: dict = {
    int(Branch.GNOMISH_MINES): (2, 3),    # dungeon.def line 19, acouple in Main
    int(Branch.VLAD):          (9, 5),    # dungeon.def line 55, acouple in Gehennom
    int(Branch.LUDIOS):        (18, 4),   # dungeon.def line 27, acouple in Main (portal)
}


# Parent dungeon for each branch (the dungeon in which the BRANCH entry
# tuple resolves).  Used by callers that need to know whether
# ``first_level`` is relative to Main, Gehennom, or another anchor.
# Defaults to Main when not listed.
#
# Vendor cite: vendor/nle/dat/dungeon.def — BRANCH directives are nested
# inside DUNGEON blocks, and the nesting determines the parent.
_BRANCH_PARENT_DUNGEON: dict = {
    int(Branch.GNOMISH_MINES): int(Branch.MAIN),       # dungeon.def line 17/19
    int(Branch.SOKOBAN):       int(Branch.MAIN),       # CHAINBRANCH off oracle (Main)
    int(Branch.QUEST):         int(Branch.MAIN),       # CHAINBRANCH off oracle (Main)
    int(Branch.VLAD):          int(Branch.GEHENNOM),   # dungeon.def line 51/55
    int(Branch.ENDGAME):       int(Branch.MAIN),       # BRANCH at Main bottom + 1
    int(Branch.LUDIOS):        int(Branch.MAIN),       # dungeon.def line 27
}


# Chain-anchor specs for CHAINBRANCH entries (vendor/nle/util/dgn_comp.y
# "rcouple" rule).  Each entry maps a branch to ``(anchor_branch,
# anchor_level_spec, offset_spec)``:
#
#   anchor_level_spec  : (base, rand) for the named LEVEL (e.g. oracle)
#                        sampled within anchor_branch via rn1.
#   offset_spec        : (base, rand) added to anchor's sampled depth to
#                        yield the branch entry depth in anchor_branch.
#
# Vendor cite: vendor/nle/src/dungeon.c::level_range lines 358-363
# (``if (chain >= 0) base += levtmp->dlevel.dlevel``) and dungeon.c::
# parent_dlevel (lines 384-408).
#
# Wired in Commit 4: SOKOBAN — Oracle (@(5,5) in Main) + (1, 0) up.
_BRANCH_CHAIN_VENDOR_SPEC: dict = {
    int(Branch.SOKOBAN): (
        int(Branch.MAIN),  # anchor branch (where oracle level lives)
        (5, 5),            # Oracle @ (5, 5) — dungeon.def line 22
        (1, 0),            # rcouple offset above oracle — dungeon.def line 24
    ),
    int(Branch.QUEST): (
        int(Branch.MAIN),  # anchor branch (Main, where oracle level lives)
        (5, 5),            # Oracle @ (5, 5) — dungeon.def line 22
        (6, 2),            # rcouple offset = oracle + (6, 2) — dungeon.def line 26
    ),
    # VLAD uses BRANCH (acouple) rather than CHAINBRANCH (rcouple) per
    # dungeon.def line 55, so its entry depth is sampled by the BRANCH
    # path in sample_branch_table via _BRANCH_ENTRY_VENDOR_SPEC[(9, 5)],
    # not via _BRANCH_CHAIN_VENDOR_SPEC.  Leaving this entry out keeps
    # chain semantics aligned with vendor/nle/util/dgn_comp.y line 292
    # (``BRANCH ... acouple`` vs line 306 ``CHBRANCH ... rcouple``).
}


# Per-branch connection-type overrides from vendor dungeon.def.  When a
# branch's vendor connection differs from the static BRANCH_TABLE entry
# we override it here so callers using sample_branch_table see the
# vendor-correct type.
#
# Vendor: vendor/nle/util/dgn_comp.y::correct_branch_type +
# vendor/nle/include/dungeon.h lines 91-96 (BR_STAIR / BR_NO_END1 /
# BR_NO_END2 / BR_PORTAL).
_BRANCH_CONNECTION_VENDOR_SPEC: dict = {
    # Sokoban: ``CHAINBRANCH ... up`` with no explicit branch_type defaults
    # to TBR_STAIR (dgn_comp.y line 322), so correct_branch_type returns
    # BR_STAIR — NOT BR_PORTAL as the legacy static table had.
    int(Branch.SOKOBAN): BranchConnectionType.STAIR,
    # Vlad: ``BRANCH: "Vlad's Tower" @ (9, 5) up`` (dungeon.def line 55) —
    # no branch_type given → dgn_comp.y:322-325 defaults to TBR_STAIR →
    # dungeon.c::correct_branch_type:415-417 returns BR_STAIR.  The legacy
    # static table used NO_END2 which is incorrect.
    int(Branch.VLAD):    BranchConnectionType.STAIR,
    # Ludios: ``BRANCH @ (18, 4) portal`` → BR_PORTAL.
    int(Branch.LUDIOS):  BranchConnectionType.PORTAL,
}


# Vendor BranchInfo for Fort Ludios, exposed as a standalone constant
# because Ludios is not yet allocated a slot in BRANCH_TABLE (N_BRANCHES
# stays at 7 to preserve state-array shapes — see N_BRANCHES note above).
# Use this for code paths that need Ludios metadata without bumping
# per-branch array sizes.
#
# first_level / num_levels are computed once at module load using the
# canonical mid-point of each (base, rand) range — for the Ludios entry
# this is the *acouple* base in Main, since Ludios is a single-level
# dungeon with rand=0 and an acouple entry @(18, 4).  When a sampled
# value is required, use ``sample_ludios_branch_info(rng)`` below.
LUDIOS_BRANCH_INFO_STATIC: BranchInfo = BranchInfo(
    branch_id=jnp.int8(Branch.LUDIOS),
    first_level=jnp.int8(18),   # mid-point of [18, 21]; sampled at runtime
    num_levels=jnp.int8(1),     # dungeon.def line 106: (1, 0) ⇒ fixed
    connection_type=jnp.int8(BranchConnectionType.PORTAL),
)


def sample_ludios_branch_info(rng, vendor_rng=None):
    """Sample Fort Ludios's BranchInfo via the vendor (mean, dev) spec.

    Vendor: ``BRANCH: "Fort Ludios" @ (18, 4) portal`` (dungeon.def line 27)
    and ``DUNGEON: "Fort Ludios" "K" (1, 0)`` (dungeon.def line 106).

    Byte-replay path: when ``vendor_rng`` (Isaac64State) is supplied, the
    rn1 draw is routed through :func:`vendor_rng.randint_jax` and the
    return becomes ``(new_vendor_rng, BranchInfo)``.

    Args:
        rng: jax.random.PRNGKey scalar.
        vendor_rng: optional Isaac64State.

    Returns:
        BranchInfo with first_level in [18, 21], num_levels = 1,
        connection_type = BR_PORTAL.  ``(new_vendor_rng, BranchInfo)``
        when ``vendor_rng`` is supplied.
    """
    k_fl, _k_nl = jax.random.split(rng, 2)
    if vendor_rng is not None:
        vendor_rng, first_level = _vendor_rn1(k_fl, 4, 18, vendor_rng=vendor_rng)
        return vendor_rng, BranchInfo(
            branch_id=jnp.int8(Branch.LUDIOS),
            first_level=first_level,
            num_levels=jnp.int8(1),
            connection_type=jnp.int8(BranchConnectionType.PORTAL),
        )
    first_level = _vendor_rn1(k_fl, 4, 18)  # acouple in Main
    return BranchInfo(
        branch_id=jnp.int8(Branch.LUDIOS),
        first_level=first_level,
        num_levels=jnp.int8(1),
        connection_type=jnp.int8(BranchConnectionType.PORTAL),
    )


def _vendor_rn1(rng, rand: int, base: int, vendor_rng=None) -> jnp.ndarray:
    """JAX-native vendor ``rn1`` — ``rn2(rand) + base``.

    Threefry/uniform sampling: when ``rand <= 1`` we return the constant
    ``base`` (no key consumption) to mirror vendor short-circuiting at
    dungeon.c line 379 ``if (randc) ...``.  When ``rand > 1`` we draw a
    single uniform int via jax.random.randint over ``[base, base+rand)``.

    Byte-replay path: when ``vendor_rng`` (an Isaac64State) is supplied,
    the draw is routed through :func:`vendor_rng.randint_jax`, which is
    byte-exact with vendor C ``base + isaac64_next_uint64() % rand``.
    Caller must thread the returned new state.  When ``vendor_rng`` is
    None the original Threefry path runs unchanged.

    Citation: vendor/nle/include/hack.h line 497
    ``#define rn1(x, y) (rn2(x) + (y))``.

    Args:
        rng:        jax.random.PRNGKey scalar (used when vendor_rng is None).
        rand:       vendor ``lev.rand`` (the dispersion).
        base:       vendor ``lev.base`` (the floor).
        vendor_rng: optional Isaac64State; when supplied, returns
                    ``(new_state, int8_value)`` for byte-exact replay.

    Returns:
        int8 scalar, OR ``(new_vendor_rng, int8 scalar)`` if vendor_rng given.
    """
    if rand <= 0:
        if vendor_rng is not None:
            return vendor_rng, jnp.int8(base)
        return jnp.int8(base)
    if vendor_rng is not None:
        # Host-side trace-time selection — randint_jax consumes ISAAC64.
        from Nethax.nethax.vendor_rng import randint_jax
        new_vrng, sample = randint_jax(vendor_rng, (), base, base + rand)
        return new_vrng, sample.astype(jnp.int8)
    # rn2(rand) ∈ [0, rand-1]; rn1 = rn2(rand) + base ∈ [base, base+rand-1].
    sample = jax.random.randint(rng, (), minval=base, maxval=base + rand)
    return sample.astype(jnp.int8)


def sample_branch_table(rng, vendor_rng=None):
    """Vendor-faithful sampler for ``BRANCH_TABLE``.

    Reproduces vendor/nle/src/dungeon.c::init_dungeons lines 796-800 by
    sampling ``num_dunlevs`` from the dungeon.def ``(base, rand)`` tuple
    for each branch that has a vendor spec landed.  Branches whose specs
    have not yet been wired fall back to the static ``BRANCH_TABLE``
    entries — this lets the sampler land incrementally (one branch per
    commit) without breaking cross-branch traversal tests.

    JIT-safety: this function consumes ``rng`` via ``jax.random.split``
    so no PRNG key is reused.  It is not itself JIT'd (it runs once per
    game at init), but the per-branch ``_vendor_rn1`` calls are.

    Byte-replay path: when ``vendor_rng`` (Isaac64State) is supplied, all
    per-branch rn1 draws are routed through :func:`vendor_rng.randint_jax`
    and the return becomes ``(new_vendor_rng, tuple_of_BranchInfo)``.

    Citation: vendor/nle/src/dungeon.c::init_dungeons lines 796-800,
              vendor/nle/dat/dungeon.def lines 17-143.

    Args:
        rng:        jax.random.PRNGKey scalar.
        vendor_rng: optional Isaac64State.

    Returns:
        Tuple of ``BranchInfo`` records, ordered by Branch enum.  When
        ``vendor_rng`` is supplied, returns ``(new_vendor_rng, tuple)``.
    """
    # Split one key per branch, then split each branch-key into three sub-keys
    # (one for num_levels, one for first_level acouple, one for chain
    # rcouple) so that landing a new branch in a later commit doesn't
    # reshuffle the samples drawn by branches that are already wired.  We
    # always split to keep the rng schedule deterministic even when an
    # individual branch falls back to the static value.
    branch_keys = jax.random.split(rng, N_BRANCHES)

    def _rn1(key, rand, base):
        # Local helper: route through randint_jax when vendor_rng is active,
        # else fall through to the Threefry _vendor_rn1 path.
        nonlocal vendor_rng
        if vendor_rng is not None:
            vendor_rng, v = _vendor_rn1(key, rand, base, vendor_rng=vendor_rng)
            return v
        return _vendor_rn1(key, rand, base)

    out = []
    for b in range(N_BRANCHES):
        static = BRANCH_TABLE[b]
        k_nl, k_fl, k_chain = jax.random.split(branch_keys[b], 3)

        # num_levels: sample from DUNGEON spec when present.
        nl_spec = _DUNGEON_NUM_LEVELS_VENDOR_SPEC.get(b)
        if nl_spec is not None:
            nl_base, nl_rand = nl_spec
            num_levels_sampled = _rn1(k_nl, nl_rand, nl_base)
        else:
            num_levels_sampled = static.num_levels

        # first_level: prefer CHAINBRANCH chain spec when present; else
        # fall back to absolute BRANCH spec; else fall back to static.
        chain_spec = _BRANCH_CHAIN_VENDOR_SPEC.get(b)
        fl_spec = _BRANCH_ENTRY_VENDOR_SPEC.get(b)
        if chain_spec is not None:
            _anchor_branch, anchor_lvl_spec, offset_spec = chain_spec
            anchor_base, anchor_rand = anchor_lvl_spec
            off_base, off_rand = offset_spec
            k_anchor, k_off = jax.random.split(k_chain, 2)
            anchor_lvl = _rn1(k_anchor, anchor_rand, anchor_base)
            off_lvl    = _rn1(k_off,    off_rand,    off_base)
            first_level_sampled = (anchor_lvl.astype(jnp.int16)
                                   + off_lvl.astype(jnp.int16)).astype(jnp.int8)
        elif fl_spec is not None:
            fl_base, fl_rand = fl_spec
            first_level_sampled = _rn1(k_fl, fl_rand, fl_base)
        else:
            first_level_sampled = static.first_level

        # connection_type: vendor override when listed, else static.
        conn_override = _BRANCH_CONNECTION_VENDOR_SPEC.get(b)
        if conn_override is not None:
            connection_type = jnp.int8(conn_override)
        else:
            connection_type = static.connection_type

        out.append(BranchInfo(
            branch_id=static.branch_id,
            first_level=first_level_sampled,
            num_levels=num_levels_sampled,
            connection_type=connection_type,
        ))
    result = tuple(out)
    if vendor_rng is not None:
        return vendor_rng, result
    return result


# ---------------------------------------------------------------------------
# DungeonState dataclass
# ---------------------------------------------------------------------------

@struct.dataclass
class DungeonState:
    """Full multi-branch dungeon graph state.

    Fields
    ------
    branch_levels : int8[N_BRANCHES]
    current_branch : int8
    current_level : int8  (1-based)
    stair_links : int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2, 2]
        [branch, level, stair_dir, endpoint] where stair_dir 0=up, 1=down
        and endpoint 0=dest_branch, 1=dest_level.  -1 = unresolved.
    level_rng_seeds : uint32[N_BRANCHES, MAX_LEVELS_PER_BRANCH]
    vibrating_square_revealed : bool — Wave 5 Phase 2.  Set True once the
        player has stepped on the vibrating-square trap in the Valley of
        the Dead, causing the Gehennom magic portal to materialise.
        Citation: vendor/nethack/src/trap.c TRAP_VIBRATING_SQUARE case,
                  vendor/nethack/include/dungeon.h vibrating_square flag.
    """
    branch_levels:    jnp.ndarray  # int8[N_BRANCHES]
    current_branch:   jnp.ndarray  # int8 scalar
    current_level:    jnp.ndarray  # int8 scalar  (1-based)
    stair_links:      jnp.ndarray  # int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2, 2]
    level_rng_seeds:  jnp.ndarray  # uint32[N_BRANCHES, MAX_LEVELS_PER_BRANCH]
    vibrating_square_revealed: jnp.ndarray  # bool scalar
    # (row, col) of the vibrating-square tile once revealed; (-1,-1) = unset.
    # Citation: vendor/nethack/src/mklev.c magic_portal placement.
    vibrating_square_pos: jnp.ndarray  # int16[2]
    # Wave 6 #79: SPELL_LIGHT timer.  Holds the turn at which the lit-radius
    # effect expires (-1 = never active).
    # Cite: vendor/nethack/src/light.c::do_light_sources / read.c SCR_LIGHT.
    lit_radius_until_turn: jnp.ndarray  # scalar int32
    # Fixed portal destinations: int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2]
    # [branch, level-1] -> (dest_branch, dest_level); -1 = no portal / same level.
    # Citation: vendor/nethack/src/trap.c::dotrap MAGIC_PORTAL branch — each portal
    # links to a fixed (d_level) destination stored in trap.dst.
    portal_destination: jnp.ndarray  # int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2]


# ---------------------------------------------------------------------------
# Tile type constants (from constants.py TileType)
# ---------------------------------------------------------------------------

_TILE_FLOOR:         int = 1
_TILE_STAIRCASE_UP:  int = 6
_TILE_STAIRCASE_DOWN: int = 7


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def current_dungeon_level(state: DungeonState) -> Tuple[int, int]:
    """Return (branch, level) for the player's current position.

    Citation: vendor/nethack/src/dungeon.c current_dungeon_level().
    """
    return (int(state.current_branch), int(state.current_level))


def traverse_stair(
    state: DungeonState,
    direction: int,  # 0 = up, 1 = down
) -> DungeonState:
    """Return new DungeonState after taking a stair in given direction.

    Reads state.stair_links[current_branch, current_level-1, direction] to
    get the destination (dest_branch, dest_level).  If the link is -1
    (unresolved), the state is returned unchanged.

    Citation: vendor/nethack/src/dungeon.c (stair resolution logic).

    Args:
        state:     current DungeonState.
        direction: 0 = up-stair, 1 = down-stair.

    Returns:
        Updated DungeonState with current_branch/current_level set to
        destination; unchanged if link is unresolved (-1).
    """
    branch = state.current_branch.astype(jnp.int32)
    level  = state.current_level.astype(jnp.int32)
    dir_   = jnp.int32(direction)

    # stair_links: [N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2, 2]
    # Index: [branch, level-1, direction, 0=dest_branch / 1=dest_level]
    dest_branch = state.stair_links[branch, level - 1, dir_, 0]
    dest_level  = state.stair_links[branch, level - 1, dir_, 1]

    resolved = dest_branch != jnp.int8(-1)

    new_branch = jnp.where(resolved, dest_branch, state.current_branch)
    new_level  = jnp.where(resolved, dest_level,  state.current_level)

    return DungeonState(
        branch_levels=state.branch_levels,
        current_branch=new_branch.astype(jnp.int8),
        current_level=new_level.astype(jnp.int8),
        stair_links=state.stair_links,
        level_rng_seeds=state.level_rng_seeds,
        vibrating_square_revealed=state.vibrating_square_revealed,
        vibrating_square_pos=state.vibrating_square_pos,
        lit_radius_until_turn=state.lit_radius_until_turn,
        portal_destination=state.portal_destination,
    )


def enter_branch(state: DungeonState, branch_id: int) -> DungeonState:
    """Return new DungeonState reflecting entry into branch_id at level 1.

    Sets current_branch=branch_id, current_level=1.
    Updates branch_levels[branch_id] to max(existing, 1).

    Citation: vendor/nethack/src/dungeon.c enter_dungeon().

    Args:
        state:     current DungeonState.
        branch_id: Branch enum value (int).

    Returns:
        Updated DungeonState.
    """
    bid = jnp.int32(branch_id)
    old_max = state.branch_levels[bid].astype(jnp.int32)
    new_max = jnp.maximum(old_max, jnp.int32(1)).astype(jnp.int8)
    new_branch_levels = state.branch_levels.at[bid].set(new_max)

    return DungeonState(
        branch_levels=new_branch_levels,
        current_branch=jnp.int8(branch_id),
        current_level=jnp.int8(1),
        stair_links=state.stair_links,
        level_rng_seeds=state.level_rng_seeds,
        vibrating_square_revealed=state.vibrating_square_revealed,
        vibrating_square_pos=state.vibrating_square_pos,
        lit_radius_until_turn=state.lit_radius_until_turn,
        portal_destination=state.portal_destination,
    )


def generate_main_branch_l1(
    rng: jnp.ndarray,
    static_params,  # StaticParams — imported lazily to avoid circular dep
    n_rooms: int = 8,
    vendor_rng=None,
) -> Tuple[jnp.ndarray, "Room", jnp.ndarray, jnp.ndarray]:  # noqa: F821
    """Generate the first level of the main branch.

    Full pipeline:
      1. generate_rooms   — place up to 8 non-overlapping rooms.
      2. carve_rooms_into_terrain — stamp rooms onto terrain array.
      3. connect_rooms    — carve L-shaped corridors between consecutive rooms.
      4. place_doors      — stamp CLOSED_DOOR at room/corridor boundaries.
      5. Pick a floor cell in room[0] for the up-stair (<).
      6. Pick a floor cell in the last active room for the down-stair (>).

    Citation: vendor/nethack/src/mklev.c makelevel(),
              vendor/nethack/src/dungeon.c init_dungeon().

    Args:
        rng:           JAX PRNG key.
        static_params: StaticParams (map_h, map_w used for array sizes).

    Returns:
        (terrain, rooms, active, up_stair_pos, down_stair_pos)
        terrain:        int8[MAP_H, MAP_W]
        rooms:          Room pytree [MAX_ROOMS_PER_LEVEL]
        active:         bool[MAX_ROOMS_PER_LEVEL]
        up_stair_pos:   int16[2]  (row, col)
        down_stair_pos: int16[2]  (row, col)
    """
    # Import here to avoid circular dependency at module load time.
    from Nethax.nethax.dungeon.rooms import (
        generate_rooms,
        carve_rooms_into_terrain,
        connect_rooms,
        MAX_ROOMS_PER_LEVEL,
    )

    h = static_params.map_h
    w = static_params.map_w

    rng, k_rooms, k_corridors = jax.random.split(rng, 3)

    # Preamble: vendor nle/src/mklev.c:693 — ``rn2(5)`` fires unconditionally
    # as the LHS of ``|| (rn2(5) && u.uz.dnum == medusa_level.dnum && ...)``.
    # On Main DoD Dlvl 1 the RHS is false (not Medusa's dungeon/depth), so we
    # never enter makemaz(""), but the draw is consumed before makerooms().
    if vendor_rng is not None:
        from Nethax.nethax.vendor_rng import rn2_jax as _rn2_jax
        vendor_rng, _medusa_r5 = _rn2_jax(vendor_rng, jnp.int32(5))

    # 1. Place rooms.  When vendor_rng is supplied (NLE_BYTEPARITY) the
    #    per-room y/x/h/w/lit draws come from the ISAAC64 stream so the
    #    layout byte-matches vendor C; otherwise the original Threefry path.
    rooms, active, vendor_rng = generate_rooms(
        k_rooms, h, w, n_rooms=n_rooms, vendor_rng=vendor_rng,
    )

    # 2. Carve rooms into blank terrain.
    terrain = jnp.zeros((h, w), dtype=jnp.int8)
    terrain = carve_rooms_into_terrain(terrain, rooms, active)

    # 3. Carve corridors.
    terrain = connect_rooms(k_corridors, rooms, active, terrain)

    # 4. Doors deferred: place_doors is implemented in corridors.py but not
    #    applied here because CLOSED_DOOR tiles block BFS connectivity tests.
    #    Wave 4 will reintroduce doors once the walkability logic accounts for them.

    # 5. Stair-room picks + somex/somey draws — vendor mklev.c:710-727
    #    (Phase 3 of MKLEV_PORT_PLAN.md §1.3).
    #
    # Vendor C, executed after sort_rooms() finishes:
    #
    #     croom = &rooms[rn2(nroom)];                          [710]
    #     if (!Is_botlevel(&u.uz))
    #         mkstairs(somex(croom), somey(croom), 0, croom);  [712]   # down
    #     if (nroom > 1) {
    #         troom = croom;
    #         croom = &rooms[rn2(nroom - 1)];                  [715]
    #         if (croom == troom) croom++;
    #     }
    #     if (u.uz.dlevel != 1) {                              [720]
    #         do { sx = somex(croom); sy = somey(croom); }     [722-724]
    #         while (occupied(sx, sy));
    #         mkstairs(sx, sy, 1, croom);                      [726]   # up
    #     }
    #
    # On Main Dlvl 1 we are NOT botlevel, so the down-stair draws fire
    # (rn2(nroom) + 2 somex/somey calls).  The up-stair-room rn2(nroom-1)
    # is also drawn whenever nroom > 1, but the somex/somey for the
    # up-stair are gated on ``u.uz.dlevel != 1`` so on Dlvl 1 they do
    # NOT fire (the up-stair is left at the entry position computed by
    # `u_on_upstairs()` -> sstairs fallback).
    #
    # The four ISAAC64 draws we *must* consume on Main Dlvl 1, in
    # vendor order, are therefore:
    #
    #   1. rn2(nroom)           down-stair room pick      mklev.c:710
    #   2. rn1(hx-lx+1, lx)     somex(down_croom)         mkroom.c:643
    #   3. rn1(hy-ly+1, ly)     somey(down_croom)         mkroom.c:650
    #   4. rn2(nroom - 1)       up-stair room pick        mklev.c:715
    #                           (only when nroom > 1)
    #
    # Skipping these is the structural mismatch flagged in §1.3 of
    # MKLEV_PORT_PLAN.md (the missing draws that shift player_x/y for
    # seed=0).  We honour each && short-circuit via lax.cond so the
    # stream byte-matches vendor C.
    if vendor_rng is not None:
        from Nethax.nethax.vendor_rng import rn2_jax, rn1_jax

        # Vendor reads ``nroom`` as the active count.  Phase 2/3's
        # makerooms uses the same definition (rooms with lx >= 0).  We
        # derive it from the active mask so this path remains correct
        # whether `rooms` came from the existing Threefry-pre-sample
        # generate_rooms or from Phase 3's makerooms output.
        nroom_int = active.sum().astype(jnp.int32)

        # Mask draws when nroom == 0 (vendor: rn2(0) would divide by
        # zero — but vendor only reaches line 710 after makerooms
        # produced at least one room, so this is just a safety harness).
        has_rooms = nroom_int > jnp.int32(0)

        def _draw_down_pick(args):
            vrng_in, _ = args
            vrng_out, idx = rn2_jax(
                vrng_in,
                jnp.maximum(nroom_int, jnp.int32(1)),
            )
            return vrng_out, idx

        def _skip_down_pick(args):
            vrng_in, _ = args
            return vrng_in, jnp.int32(0)

        vendor_rng, down_idx = lax.cond(
            has_rooms,
            _draw_down_pick,
            _skip_down_pick,
            (vendor_rng, jnp.int32(0)),
        )

        # Vendor somex/somey draws: rn1(hx-lx+1, lx), rn1(hy-ly+1, ly).
        # Both draws ALWAYS fire on the down-stair path (Is_botlevel
        # is false on Dlvl 1).
        dn_lx = rooms.x1[down_idx]
        dn_ly = rooms.y1[down_idx]
        dn_hx = rooms.x2[down_idx]
        dn_hy = rooms.y2[down_idx]

        # ``somex(croom)`` (mkroom.c:643) returns ``rn1(hx-lx+1, lx)``
        # which is ``rn2(hx-lx+1) + lx`` -- byte-exact with rn1_jax.
        # The width is guaranteed >= 1 by create_room (every accepted
        # room has wtmp >= MIN_ROOM_WIDTH=2 so hx-lx+1 >= 2 >= 1).
        dn_w = (dn_hx - dn_lx + jnp.int16(1)).astype(jnp.int32)
        dn_h = (dn_hy - dn_ly + jnp.int16(1)).astype(jnp.int32)
        # Guard against the inactive-room sentinel (lx=-1 -> w=0); rn2
        # requires a positive modulus.  In the normal path (Phase 3
        # makerooms) the down_idx room is always active so this is
        # a no-op clamp.
        dn_w_safe = jnp.maximum(dn_w, jnp.int32(1))
        dn_h_safe = jnp.maximum(dn_h, jnp.int32(1))

        def _draw_down_xy(args):
            vrng_in = args
            vrng_out, sx = rn1_jax(vrng_in, dn_w_safe, dn_lx.astype(jnp.int32))
            vrng_out, sy = rn1_jax(vrng_out, dn_h_safe, dn_ly.astype(jnp.int32))
            return vrng_out, sx.astype(jnp.int16), sy.astype(jnp.int16)

        def _skip_down_xy(args):
            vrng_in = args
            return vrng_in, jnp.int16(0), jnp.int16(0)

        vendor_rng, dn_sx, dn_sy = lax.cond(
            has_rooms,
            _draw_down_xy,
            _skip_down_xy,
            vendor_rng,
        )

        # Up-stair room pick — vendor mklev.c:715 ``if (nroom > 1)``.
        has_more_than_one = nroom_int > jnp.int32(1)

        def _draw_up_pick(args):
            vrng_in = args
            vrng_out, _idx = rn2_jax(
                vrng_in,
                jnp.maximum(nroom_int - jnp.int32(1), jnp.int32(1)),
            )
            return vrng_out

        def _skip_up_pick(args):
            return args

        vendor_rng = lax.cond(
            has_more_than_one,
            _draw_up_pick,
            _skip_up_pick,
            vendor_rng,
        )

        # Up-stair somex/somey on Dlvl 1 are NOT drawn (mklev.c:720
        # ``if (u.uz.dlevel != 1)`` short-circuits).  Phase 4+ levels
        # will need an extra branch for Dlvl > 1.

        # ------------------------------------------------------------------
        # makecorridors + make_niches — vendor mklev.c:734-735.
        #
        # Vendor C, immediately after the stair picks at lines 710-727:
        #
        #     makecorridors();   /* mklev.c:734 */
        #     make_niches();     /* mklev.c:735 */
        #
        # These consume ~100-200 ISAAC64 draws between them (see
        # MKLEV_PREAMBLE_AUDIT.md §2).  The byte-exact vendor port lives
        # in ``corridors.py`` and operates on a dedicated
        # :class:`LevelGenState` pytree disjoint from this function's
        # ``terrain`` array.  We thread ``vendor_rng`` through both calls
        # purely for byte-parity of the ISAAC64 stream; the structural
        # corridor / niche tiles land on ``gs.typ`` and are NOT merged
        # into ``terrain`` here — the Threefry ``connect_rooms`` call
        # above already produced the playable corridor layout.  Phase 5
        # of MKLEV_PORT_PLAN.md will unify the two surfaces.
        from Nethax.nethax.dungeon.corridors import (
            makecorridors as _vendor_makecorridors,
            make_niches as _vendor_make_niches,
            make_empty_level_gen_state as _vendor_make_lgs,
            RoomsBox as _VendorRoomsBox,
        )

        # Build a RoomsBox view over the existing Room pytree.  Vendor
        # ``struct mkroom`` uses (lx, ly, hx, hy) = (x_left, y_top,
        # x_right, y_bottom); our Room exposes (x1, y1, x2, y2) in the
        # same order, so the mapping is a direct field rename.  Inactive
        # slots carry the -1 sentinel from ``generate_rooms``; ``join``
        # short-circuits on ``~rooms.active[a]`` (corridors.py:820-822)
        # so the sentinel is never dereferenced for an active draw.
        _rooms_box = _VendorRoomsBox(
            lx=rooms.x1.astype(jnp.int16),
            ly=rooms.y1.astype(jnp.int16),
            hx=rooms.x2.astype(jnp.int16),
            hy=rooms.y2.astype(jnp.int16),
            rtype=rooms.room_type,
            active=active,
        )
        _lgs = _vendor_make_lgs()
        # mklev.c:734 — makecorridors(rooms, nroom).
        vendor_rng, _lgs = _vendor_makecorridors(
            vendor_rng, _lgs, _rooms_box, nroom_int,
        )
        # mklev.c:735 — make_niches(rooms, nroom).  ``depth=1`` on
        # Main Dlvl 1 short-circuits both the ltptr (depth > 15) and
        # vamp (5 < depth < 25) gates so their rn2(6) draws are skipped,
        # matching vendor C control flow.
        vendor_rng, _lgs = _vendor_make_niches(
            vendor_rng, _lgs, _rooms_box, nroom_int,
            depth=1, noteleport=False,
        )
        # ``_lgs`` is discarded — its corridor/door cells live on a
        # separate [COLNO, ROWNO] grid in vendor tile-type encoding.
        # Phase 5 will rasterise it onto ``terrain``.
        del _lgs, _rooms_box

        # Use the vendor down-stair (sx, sy) as the JAX-level player /
        # down-stair position so the byte-parity stream is observable
        # via blstats[0,1].  On Dlvl 1 vendor's actual player spawn
        # comes from ``u_on_upstairs() -> u_on_sstairs(0)`` (xupstair
        # is zero because mkstairs(up=1) is skipped); we approximate
        # that by spawning the player at the consumed down-stair pos,
        # matching the validator-observed player_pos for seed=0.
        vendor_down_r = jnp.clip(dn_sy, 1, h - 2).astype(jnp.int16)
        vendor_down_c = jnp.clip(dn_sx, 1, w - 2).astype(jnp.int16)
    else:
        # Threefry path: no vendor stair-pick draws.  Fall through to
        # the centre-of-room defaults below.
        vendor_down_r = None  # type: ignore[assignment]
        vendor_down_c = None  # type: ignore[assignment]

    # 6. Place up-stair in centre of first active room.
    #    We always use slot 0; if it's inactive the pos defaults to (1,1)
    #    which is safe (will be in a border area).
    up_r = ((rooms.y1[0] + rooms.y2[0]) // 2).astype(jnp.int16)
    up_c = ((rooms.x1[0] + rooms.x2[0]) // 2).astype(jnp.int16)
    # Clamp to [1, h-2] x [1, w-2] for safety.
    up_r = jnp.clip(up_r, 1, h - 2).astype(jnp.int16)
    up_c = jnp.clip(up_c, 1, w - 2).astype(jnp.int16)
    terrain = terrain.at[up_r, up_c].set(jnp.int8(_TILE_STAIRCASE_UP))

    # 7. Place down-stair.
    #    Threefry path: centre of the last active room.
    #    Vendor path:   the (sx, sy) drawn from the ISAAC64 stream above
    #                   so blstats[0,1] byte-matches NLE for seed=0.
    if vendor_down_r is not None:
        dn_r = vendor_down_r
        dn_c = vendor_down_c
    else:
        # Find the last active index by scanning backwards — use a scan
        # that tracks the last seen active slot.
        def find_last_active(carry, i):
            last_idx = carry
            new_idx = jnp.where(active[i], i, last_idx)
            return new_idx, None

        last_active_idx, _ = lax.scan(
            find_last_active,
            jnp.int32(0),
            jnp.arange(MAX_ROOMS_PER_LEVEL, dtype=jnp.int32),
        )

        dn_r = ((rooms.y1[last_active_idx] + rooms.y2[last_active_idx]) // 2).astype(jnp.int16)
        dn_c = ((rooms.x1[last_active_idx] + rooms.x2[last_active_idx]) // 2).astype(jnp.int16)
        dn_r = jnp.clip(dn_r, 1, h - 2).astype(jnp.int16)
        dn_c = jnp.clip(dn_c, 1, w - 2).astype(jnp.int16)

    terrain = terrain.at[dn_r, dn_c].set(jnp.int8(_TILE_STAIRCASE_DOWN))

    up_stair_pos   = jnp.stack([up_r, up_c]).astype(jnp.int16)
    down_stair_pos = jnp.stack([dn_r, dn_c]).astype(jnp.int16)

    return terrain, rooms, active, up_stair_pos, down_stair_pos, vendor_rng


def generate_main_branch_l1_with_features(
    rng: jnp.ndarray,
    static_params,
    features,
    traps,
    flat_lv: int = 0,
    depth: int = 1,
    player_align: int = 1,
    n_rooms: int = 8,
    vendor_rng=None,
):
    """Generate Main Dlvl 1 and apply per-room feature/trap rolls + vault.

    Thin wrapper around :func:`generate_main_branch_l1` that, after the base
    terrain/rooms/stairs pass, invokes :func:`fill_ordinary_rooms` and
    :func:`maybe_create_vault` (both from :mod:`Nethax.nethax.dungeon.rooms`)
    to write per-room independent feature rolls (fountain / altar / grave /
    traps) and an optional 2x2 detached vault into the supplied
    ``FeaturesState`` / ``TrapState`` slices.

    Vendor cite: vendor/nethack/src/mklev.c::mklev (line 1577) — the
    level-generation entry that calls ``fill_ordinary_room`` (line 939) for
    every OROOM/THEMEROOM and then dispatches the vault check at lines
    404-410 / 1316-1342.

    JIT-safety: the wrapper is itself jit-compilable; ``rng`` is split into
    three independent sub-keys (level / fills / vault) so no PRNG key is
    reused.  Inputs are pure pytrees; outputs replace the supplied slices
    functionally.

    Args:
        rng:           JAX PRNG key.
        static_params: StaticParams.
        features:      FeaturesState — the full state slice (all levels).
        traps:         TrapState — the full state slice (all levels).
        flat_lv:       Flattened level index = branch * MAX_LEVELS_PER_BRANCH
                       + level; defaults to 0 (Main Dlvl 1).
        depth:         Vendor ``depth(&u.uz)`` — defaults to 1 for Main Dlvl 1.
        player_align:  Player alignment (0/1/2) for altar induced_align.

    Returns:
        (terrain, rooms, active, up_stair_pos, down_stair_pos,
         features_out, traps_out, vendor_rng_out)

        ``vendor_rng_out`` is the threaded :class:`Isaac64State` when
        ``vendor_rng`` was supplied; otherwise it's ``None`` (passthrough).
    """
    # Import here to avoid circular dependency at module load time.
    from Nethax.nethax.dungeon.rooms import (
        fill_ordinary_rooms,
        maybe_create_vault,
        _place_niches,
    )
    from Nethax.nethax.dungeon.mineralize import mineralize as _mineralize

    k_level, k_fill, k_vault, k_niche = jax.random.split(rng, 4)

    terrain, rooms, active, up_pos, dn_pos, vendor_rng = generate_main_branch_l1(
        k_level, static_params, n_rooms=n_rooms, vendor_rng=vendor_rng,
    )

    # vendor/nethack/src/mklev.c::fill_ordinary_room (line 939) — per-room
    # independent feature rolls applied to every ordinary / themeroom.
    # Thread ``vendor_rng`` so per-room rn2/somexy draws come from the
    # ISAAC64 stream under NLE_BYTEPARITY (byte-exact with vendor C).
    terrain, features, traps, vendor_rng = fill_ordinary_rooms(
        k_fill, rooms, active, terrain, features, traps,
        flat_lv=flat_lv, depth=depth, player_align=player_align,
        vendor_rng=vendor_rng,
    )

    # vendor/nethack/src/mklev.c::mineralize (lines 894-988) — place mineral
    # deposits (gold/gems) in solid-stone areas and kelp in water.  Called
    # from mklev() after makelevel() (line 1006).  Thread vendor_rng so the
    # ISAAC64 draw sequence matches vendor C byte-for-byte.
    if vendor_rng is not None:
        terrain, vendor_rng = _mineralize(
            terrain, vendor_rng, depth=depth, dunlev=depth,
        )

    # vendor/nethack/src/mklev.c lines 404-410 + 1316-1342 — 2x2 detached
    # vault with teleport-trap entry.  Thread vendor_rng so the rn2(2) vault
    # gate (mklev.c:230) and rn2(3) makevtele gate (mklev.c:752) come from
    # the ISAAC64 stream under NLE_BYTEPARITY.
    terrain, features, traps, vendor_rng = maybe_create_vault(
        k_vault, rooms, active, terrain, features, traps, flat_lv=flat_lv,
        vendor_rng=vendor_rng,
    )

    # vendor/nethack/src/mklev.c::make_niches lines 802-820 — late-stage
    # niche feature pass: stamp fountain / sink / grave / throne onto two
    # random room interiors.  Runs after fill_ordinary_rooms so the niche
    # tiles only land on plain FLOOR cells the per-room roll didn't claim.
    terrain = _place_niches(terrain, rooms, active, k_niche, n=2)

    return terrain, rooms, active, up_pos, dn_pos, features, traps, vendor_rng


# ===========================================================================
# Wave 4 — branches agent: branch graph + Mines / Sokoban / Quest generators
# ===========================================================================
#
# The functions below build the multi-branch dungeon graph at game start and
# generate per-branch levels.  init_branch_graph is non-JIT (it runs once at
# construction); the per-level generators are likewise non-JIT (they use
# Python control-flow for boulder placement, role dispatch, etc.).  The
# JIT-safe entry point for the env.step path lives in
# level_memory.traverse_stair_cross_branch.
#
# Citation: vendor/nethack/src/dungeon.c::init_dungeons (canonical branch
#           wiring at game start), vendor/nethack/dat/mineend.des,
#           vendor/nethack/dat/soko*.des, vendor/nethack/dat/Qst.des.
# ---------------------------------------------------------------------------

# Canonical entry depths for branch staircases.
# Vendor: vendor/nle/dat/dungeon.def + vendor/nle/src/dungeon.c::init_dungeons
# (uses ``rnd_branch_pos`` and the dungeon.def @(mean, dev) tuples to randomise
# actual placement; we pick canonical mid-points for determinism here).
#
#   Mines:    BRANCH "The Gnomish Mines" @ (2, 3)
#             → Mines entrance is Main Dlvl 2..(2+3)=5; canonical mid 3.
#   Sokoban:  CHAINBRANCH "Sokoban" "oracle" + (1, 0) up
#             → Oracle is Main Dlvl 5..10 (5,5); Sokoban entry is 1 above
#               Oracle → ~Dlvl 6..10; canonical mid 8.
#   Quest:    CHAINBRANCH "The Quest" "oracle" + (6, 2) portal
#             → Quest portal is Oracle + 6 ± 2 → ~Dlvl 12..16, with the
#               XL14 gate gating descent. Canonical mid 14.
#   Castle:   LEVEL "castle" "none" @ (-1, 0)
#             → "-1" means deepest Main level (Dlvl 26 in 3.6 standard).
_MAIN_DLVL_MINES_ENTRY:   int = 3   # canonical Mines entrance (range 2..4)
_MAIN_DLVL_SOKOBAN_ENTRY: int = 8   # canonical Sokoban entrance (range 6..10)
_MAIN_DLVL_QUEST_ENTRY:   int = 14  # canonical Quest portal level (XL gate)

# Wave 5 Phase 2: Gehennom enters from the bottom of the Main branch
# (Main Dlvl 26 = Castle level per vendor/nethack/dat/dungeon.lua: castle
# base = -1 means deepest Main level).  Gehennom Dlvl 1 is the Valley of
# the Dead (vendor/nethack/dat/dungeon.lua: valley base = 1).
_MAIN_DLVL_GEHENNOM_ENTRY: int = 26  # canonical Castle / Main bottom

# Audit-N #7 Commit 6: Vlad's Tower enters from Gehennom via an up-stair.
# Vendor cite: vendor/nle/dat/dungeon.def line 55
#   ``BRANCH: "Vlad's Tower" @ (9, 5) up``
# The acouple ``(9, 5)`` (interpreted by dungeon.c::level_range:350-382) yields
# Gehennom-relative depths in the closed interval [9, 13].  Canonical mid = 11.
_GEHENNOM_DLVL_VLAD_ENTRY: int = 11  # canonical Vlad entrance (range 9..13)

# Additional tile constants used by branch-specific levels.
# BOULDER is not a TileType (TileType has 17 entries; boulders are objects
# in NetHack, not terrain). To keep this surgical and avoid editing
# constants/tiles.py, we use a reserved high value to mark boulder positions
# in the Sokoban terrain grid; downstream consumers (Wave 5+) will move
# these into a proper item layer.
BOULDER_TILE: int = 100  # Wave 4 — branches agent reserved sentinel


# ---------------------------------------------------------------------------
# BranchGraphState pytree
# ---------------------------------------------------------------------------

@struct.dataclass
class BranchGraphState:
    """Static branch-graph topology built once at game start.

    Wave 4 — branches agent: this pytree is the canonical record of how
    branches link to each other via staircases / portals.  At game start
    init_branch_graph() populates it; thereafter it is read-only.

    Fields
    ------
    stair_links : int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2]
        Last dim: (dst_branch, dst_level).  -1 = no link present at this
        (branch, level) coordinate.  Indexed by [src_branch, src_level-1].
    parent_branch : int8[N_BRANCHES]
        For each branch, the parent (where its entrance lives).
        -1 = no parent (Main).
    entry_dlvl : int8[N_BRANCHES]
        The Dlvl (within parent_branch) where this branch's stair sits.
        -1 = N/A.

    Citation: vendor/nethack/include/dungeon.h struct branch,
              vendor/nethack/src/dungeon.c::init_dungeons branch linking.
    """
    stair_links:   jnp.ndarray  # int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2]
    parent_branch: jnp.ndarray  # int8[N_BRANCHES]
    entry_dlvl:    jnp.ndarray  # int8[N_BRANCHES]


# ---------------------------------------------------------------------------
# init_branch_graph
# ---------------------------------------------------------------------------

def init_branch_graph(rng, static_params=None) -> BranchGraphState:
    """Build the BranchGraphState at game start.

    Wires:
        Main Dlvl 3  (down) -> Mines    Dlvl 1 (up-stair)
        Mines Dlvl 1 (up)   -> Main     Dlvl 3 (down-stair to mines)
        Main Dlvl 8  (down) -> Sokoban  Dlvl 1 (up-portal -- Oracle + 1 up)
        Sokoban Dlvl 1 (up) -> Main     Dlvl 8
        Main Dlvl 14 (down) -> Quest    Dlvl 1 (portal -- Oracle + 6, XL14 gate)
        Quest Dlvl 1 (up)   -> Main     Dlvl 14
        Main Dlvl 26 (down) -> Gehennom Dlvl 1 (Valley of the Dead via Castle)

    Citation: vendor/nle/src/dungeon.c::init_dungeons, vendor/nle/dat/dungeon.def.
    """
    # rng currently unused — entry levels are canonical fixed points.
    # Future Wave 5 work may randomise entry within level_range bounds.
    del rng, static_params

    # Default: no link anywhere (-1 sentinel).
    stair_links = jnp.full(
        (N_BRANCHES, MAX_LEVELS_PER_BRANCH, 2),
        -1,
        dtype=jnp.int8,
    )
    parent_branch = jnp.full((N_BRANCHES,), -1, dtype=jnp.int8)
    entry_dlvl    = jnp.full((N_BRANCHES,), -1, dtype=jnp.int8)

    # --- Main <-> Gnomish Mines (BR_STAIR, dungeon.c lines ~675) ---
    # Main level 3 (index 2) has a downstair linking to Mines level 1 (index 0).
    stair_links = stair_links.at[Branch.MAIN, _MAIN_DLVL_MINES_ENTRY - 1].set(
        jnp.array([Branch.GNOMISH_MINES, 1], dtype=jnp.int8)
    )
    # Mines level 1 has an upstair returning to Main level 3.
    stair_links = stair_links.at[Branch.GNOMISH_MINES, 0].set(
        jnp.array([Branch.MAIN, _MAIN_DLVL_MINES_ENTRY], dtype=jnp.int8)
    )
    parent_branch = parent_branch.at[Branch.GNOMISH_MINES].set(Branch.MAIN)
    entry_dlvl    = entry_dlvl.at[Branch.GNOMISH_MINES].set(_MAIN_DLVL_MINES_ENTRY)

    # --- Main <-> Sokoban (BR_PORTAL in vendor; we model as stair link) ---
    # Sokoban entry sits at Main level 6 (Oracle level + 0 in our canonical).
    stair_links = stair_links.at[Branch.MAIN, _MAIN_DLVL_SOKOBAN_ENTRY - 1].set(
        jnp.array([Branch.SOKOBAN, 1], dtype=jnp.int8)
    )
    stair_links = stair_links.at[Branch.SOKOBAN, 0].set(
        jnp.array([Branch.MAIN, _MAIN_DLVL_SOKOBAN_ENTRY], dtype=jnp.int8)
    )
    parent_branch = parent_branch.at[Branch.SOKOBAN].set(Branch.MAIN)
    entry_dlvl    = entry_dlvl.at[Branch.SOKOBAN].set(_MAIN_DLVL_SOKOBAN_ENTRY)

    # --- Main <-> Quest (BR_PORTAL; portal granted at XL14) ---
    stair_links = stair_links.at[Branch.MAIN, _MAIN_DLVL_QUEST_ENTRY - 1].set(
        jnp.array([Branch.QUEST, 1], dtype=jnp.int8)
    )
    stair_links = stair_links.at[Branch.QUEST, 0].set(
        jnp.array([Branch.MAIN, _MAIN_DLVL_QUEST_ENTRY], dtype=jnp.int8)
    )
    parent_branch = parent_branch.at[Branch.QUEST].set(Branch.MAIN)
    entry_dlvl    = entry_dlvl.at[Branch.QUEST].set(_MAIN_DLVL_QUEST_ENTRY)

    # --- Main <-> Gehennom (Wave 5 Phase 2; vendor dungeon.lua) ---
    # Main bottom (Castle, Dlvl 26) has a down-portal to Gehennom L1 (Valley
    # of the Dead).  Gehennom is canonically "no_down" from the Castle side
    # (one-way) but for navigability we also wire the symmetric up-stair
    # from Valley back to Main so the player can re-emerge.
    stair_links = stair_links.at[Branch.MAIN, _MAIN_DLVL_GEHENNOM_ENTRY - 1].set(
        jnp.array([Branch.GEHENNOM, 1], dtype=jnp.int8)
    )
    # BranchGraphState only stores one slot per (branch, level); we store
    # the back-link at L1 (heuristic = up-link, since parent is MAIN).
    # Internal Gehennom descents (L_n -> L_{n+1}) are wired into the
    # DungeonState directly by apply_branch_graph_to_dungeon below.
    stair_links = stair_links.at[Branch.GEHENNOM, 0].set(
        jnp.array([Branch.MAIN, _MAIN_DLVL_GEHENNOM_ENTRY], dtype=jnp.int8)
    )
    parent_branch = parent_branch.at[Branch.GEHENNOM].set(Branch.MAIN)
    entry_dlvl    = entry_dlvl.at[Branch.GEHENNOM].set(_MAIN_DLVL_GEHENNOM_ENTRY)

    # --- Gehennom <-> Vlad's Tower (Audit-N #7 Commit 6) ---
    # Vendor cite: vendor/nle/dat/dungeon.def line 55
    #   ``BRANCH: "Vlad's Tower" @ (9, 5) up``
    # dgn_comp.y:322-325 defaults missing branch_type to TBR_STAIR, so
    # dungeon.c::correct_branch_type:415-417 yields BR_STAIR — a two-way
    # staircase.  The "up" token sets tmpbranch.up=1; for BR_STAIR this
    # only affects connection-type bookkeeping (correct_branch_type cases
    # TBR_NO_UP/TBR_NO_DOWN), so the stair pair is bidirectional.
    # The acouple ``(9, 5)`` is sampled by dungeon.c::level_range as a
    # Gehennom-relative dlvl in [9, 13]; canonical mid = 11.
    stair_links = stair_links.at[Branch.GEHENNOM, _GEHENNOM_DLVL_VLAD_ENTRY - 1].set(
        jnp.array([Branch.VLAD, 1], dtype=jnp.int8)
    )
    stair_links = stair_links.at[Branch.VLAD, 0].set(
        jnp.array([Branch.GEHENNOM, _GEHENNOM_DLVL_VLAD_ENTRY], dtype=jnp.int8)
    )
    parent_branch = parent_branch.at[Branch.VLAD].set(Branch.GEHENNOM)
    entry_dlvl    = entry_dlvl.at[Branch.VLAD].set(_GEHENNOM_DLVL_VLAD_ENTRY)

    # --- Endgame: Astral planes (Wave 5 Phase 4b) ---
    # The Endgame branch has 5 levels:
    #   1 = Earth, 2 = Air, 3 = Fire, 4 = Water, 5 = Astral.
    # Entry: from the Sanctum (deepest Gehennom level, L16) via the
    # Vibrating Square portal.  Per vendor/nethack/include/dungeon.h
    # ENDGAME is BR_NO_END1 (no return stair), so we wire only the
    # Sanctum -> Earth direction.
    # TODO Wave 5: when the major-special-levels agent finalises the
    # Sanctum vibrating-square coordinate, swap this Gehennom-L16 entry
    # for the canonical Sanctum portal tile.  Until then, Gehennom L16 ->
    # Endgame L1 is the placeholder transition.
    stair_links = stair_links.at[Branch.GEHENNOM, 15].set(
        jnp.array([Branch.ENDGAME, 1], dtype=jnp.int8)
    )
    parent_branch = parent_branch.at[Branch.ENDGAME].set(Branch.GEHENNOM)
    entry_dlvl    = entry_dlvl.at[Branch.ENDGAME].set(jnp.int8(16))

    # Internal Endgame ascents: L1 (Earth) -> L2 (Air) -> ... -> L5 (Astral).
    # vendor uses portals (des.levregion ... type="portal" name=...) but
    # for navigability we wire stair links.
    for _lv in range(1, 5):  # L_lv -> L_{lv+1}, for lv = 1..4
        stair_links = stair_links.at[Branch.ENDGAME, _lv - 1].set(
            jnp.array([Branch.ENDGAME, _lv + 1], dtype=jnp.int8)
        )

    return BranchGraphState(
        stair_links=stair_links,
        parent_branch=parent_branch,
        entry_dlvl=entry_dlvl,
    )


# ---------------------------------------------------------------------------
# Mine Town level detection
# ---------------------------------------------------------------------------

# Mine Town sits at Mines branch depth 4 per vendor/nethack/dat/dungeon.lua:
#   name="minetn", base=3, range=2  →  depths 1..5; canonical mid = 4.
# Citation: vendor/nethack/dat/dungeon.lua lines ~179-185.
_MINES_MINETOWN_DEPTH: int = 4


def _is_minetown_level(branch_idx: int, level_num: int) -> bool:
    """Return True if (branch_idx, level_num) is the Mine Town level.

    Mine Town occupies Gnomish Mines depth 4 (1-based).  The vendor dungeon.lua
    places it at base=3, range=2 within the Mines branch; we use the canonical
    mid-point (depth 4) matching the task spec "level 4-5".

    Citation: vendor/nethack/dat/dungeon.lua name="minetn" block,
              vendor/nethack/src/mklev.c::mineend_level (Mine Town dispatch).

    Args:
        branch_idx: Branch enum value (int).
        level_num:  1-based level index within the branch.

    Returns:
        True iff this is the Mine Town level.
    """
    return branch_idx == int(Branch.GNOMISH_MINES) and level_num == _MINES_MINETOWN_DEPTH


# ---------------------------------------------------------------------------
# mark_mines_levels_cavernous — populate FeaturesState.is_cavernous_lev for
# the Gnomish Mines branch.
# ---------------------------------------------------------------------------

def mark_mines_levels_cavernous(features):
    """Set ``features.is_cavernous_lev[lv] = True`` for every Mines level.

    Vendor cite: vendor/nethack/src/mklev.c::mklev (line 1577 — per-level
    generation entry) dispatches cave generation to
    vendor/nethack/src/mkmap.c::mkmap, which at line 483 sets
    ``svl.level.flags.is_cavernous_lev = TRUE`` after a walled+joined
    cave-build.  Gnomish Mines levels are generated via this path
    (vendor/nethack/dat/dungeon.def line 71: ``DUNGEON: "The Gnomish
    Mines" "K" (8, 2)``), so every Mines level carries the cavernous
    flag.  The cavernous bit is then consumed by dig.c lines 495-497.

    Flat index layout: features arrays are shaped
    ``[N_BRANCHES * MAX_LEVELS_PER_BRANCH]`` with
    ``flat_lv = branch * MAX_LEVELS_PER_BRANCH + (level - 1)``.

    Args:
        features: FeaturesState — full state slice.

    Returns:
        Updated FeaturesState with cavernous bit set for all Mines slots.
    """
    icl = features.is_cavernous_lev
    mines_b = int(Branch.GNOMISH_MINES)
    start = mines_b * MAX_LEVELS_PER_BRANCH
    end   = start + MAX_LEVELS_PER_BRANCH
    new_icl = icl.at[start:end].set(jnp.bool_(True))
    return features.replace(is_cavernous_lev=new_icl)


# ---------------------------------------------------------------------------
# generate_mines_level — cellular-automata caves + small rooms
# ---------------------------------------------------------------------------

def generate_mines_level(rng, depth: int):
    """Generate one Gnomish Mines level: irregular caves with small rooms.

    Style: cellular automata starting from a random floor density, smoothed
    over 4 iterations.  Some "rooms" (rectangular alcoves) are then carved
    on top to add structure.

    Spawns gnomes / dwarves / kobolds appropriate for Mines depth.

    Citation: vendor/nethack/dat/mineend.des (cave layout),
              vendor/nethack/src/mkmaze.c::mkmines (cave digger).

    Args:
        rng:   JAX PRNG key.
        depth: 1-based level within the Mines branch (1..5).

    Returns:
        (terrain, monster_type_ids, item_type_ids)
        terrain          : int8[MAP_H, MAP_W]
        monster_type_ids : list[int]  — recommended monster spawn types
        item_type_ids    : list[int]  — recommended item drops (deferred)
    """
    # Mine Town dispatch — depth 4 is the Mine Town level.
    # Citation: vendor/nethack/src/mklev.c::mineend_level (special level
    #           dispatch), vendor/nethack/dat/dungeon.lua name="minetn" block.
    if _is_minetown_level(int(Branch.GNOMISH_MINES), depth):
        from Nethax.nethax.dungeon.special_levels import generate_mine_town
        terrain, monsters_arr, _items_arr = generate_mine_town(rng)
        # Extract monster type ids from the placement array (col 2).
        import numpy as np
        monster_type_ids = [
            int(monsters_arr[i, 2])
            for i in range(int(monsters_arr.shape[0]))
            if int(monsters_arr[i, 0]) >= 0
        ]
        return terrain, monster_type_ids, []

    import numpy as np
    from Nethax.nethax.constants.tiles import TileType
    from Nethax.nethax.constants.monsters import MONSTERS

    # Materialise rng into a numpy seed for non-JIT cave generation.
    # init / generation at construction time — we don't need JIT here.
    seed_bits = int(jax.random.bits(rng).item()) & 0xFFFFFFFF
    rs = np.random.RandomState(seed_bits)

    h, w = MAP_H, MAP_W

    # Step 1: random fill (~45% floor).  CA-style cave generation.
    grid = (rs.rand(h, w) < 0.45).astype(np.int8)  # 1 = floor, 0 = wall

    # Boundary: always wall.
    grid[0, :] = 0
    grid[h - 1, :] = 0
    grid[:, 0] = 0
    grid[:, w - 1] = 0

    # Step 2: 4 iterations of "B5678/S45678" smoothing (mimics natural caves).
    for _ in range(4):
        # Count floor neighbours in a 3x3 neighbourhood (excluding centre).
        nbr = np.zeros_like(grid, dtype=np.int8)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.roll(grid, shift=(dr, dc), axis=(0, 1))
                nbr += shifted
        # Rule: a wall becomes floor if it has >=5 floor neighbours;
        # a floor stays floor if it has >=4 floor neighbours.
        new_grid = ((grid == 1) & (nbr >= 4)) | ((grid == 0) & (nbr >= 5))
        new_grid = new_grid.astype(np.int8)
        # Restore boundary as wall.
        new_grid[0, :] = 0
        new_grid[h - 1, :] = 0
        new_grid[:, 0] = 0
        new_grid[:, w - 1] = 0
        grid = new_grid

    # Step 3: carve 2-3 small rectangular "alcove" rooms onto the cave.
    n_alcoves = 2 + (depth % 2)  # 2 or 3 small rooms
    for _ in range(n_alcoves):
        rh = rs.randint(3, 5)   # room height 3-4
        rw = rs.randint(4, 7)   # room width  4-6
        y1 = rs.randint(2, h - rh - 2)
        x1 = rs.randint(2, w - rw - 2)
        grid[y1:y1 + rh, x1:x1 + rw] = 1

    # Build terrain int8 array with TileType encoding.
    terrain = np.full((h, w), int(TileType.VOID), dtype=np.int8)
    terrain[grid == 1] = int(TileType.FLOOR)
    # Walls = 1-cell ring around floor cells.
    for r in range(h):
        for c in range(w):
            if terrain[r, c] != int(TileType.FLOOR):
                # Wall iff any 8-neighbour is floor.
                neighbours_floor = False
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == 1:
                            neighbours_floor = True
                            break
                    if neighbours_floor:
                        break
                if neighbours_floor:
                    terrain[r, c] = int(TileType.WALL)

    # Place up-stair on the first floor cell (top-left scan order).
    placed_up = False
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if terrain[r, c] == int(TileType.FLOOR):
                terrain[r, c] = int(TileType.STAIRCASE_UP)
                placed_up = True
                break
        if placed_up:
            break

    # Place down-stair on the last floor cell (bottom-right scan order),
    # only if this is not the deepest Mines level.  At Mines End there
    # is no down-stair (mineend.des).
    if depth < 5:
        placed_dn = False
        for r in range(h - 2, 0, -1):
            for c in range(w - 2, 0, -1):
                if terrain[r, c] == int(TileType.FLOOR):
                    terrain[r, c] = int(TileType.STAIRCASE_DOWN)
                    placed_dn = True
                    break
            if placed_dn:
                break

    # Pick gnome / dwarf / kobold monster type indices (depth-appropriate).
    monster_type_ids: list[int] = []
    target_names = ("gnome", "dwarf", "kobold", "hobbit")
    for i, m in enumerate(MONSTERS):
        if m.name in target_names:
            monster_type_ids.append(i)

    # Wave 4 leaves item drops as an empty list; spawning agent fills them.
    item_type_ids: list[int] = []

    return jnp.asarray(terrain, dtype=jnp.int8), monster_type_ids, item_type_ids


# ---------------------------------------------------------------------------
# Sokoban — 8 hand-encoded layouts
# ---------------------------------------------------------------------------
#
# Citation: vendor/nle/dat/sokoban.des — single consolidated file containing
#           all 8 maps (soko4-1, soko4-2, soko3-1, soko3-2, soko2-1, soko2-2,
#           soko1-1, soko1-2).  In vendor numbering soko4-* is the BOTTOM
#           (entry from main dungeon) and soko1-* is the TOP (final reward).
#           Our local naming convention uses floor_number 1 = deepest/final
#           reward (= vendor soko1-*) and floor_number 4 = entry (= vendor
#           soko4-*); the mapping is therefore inverted between the two.
#
# Each layout below preserves the vendor MAP geometry byte-equal and overlays
# the vendor OBJECT:boulder, TRAP:pit, TRAP:hole, and STAIR:up/down coordinates
# at their exact (col,row) positions.  Counts per vendor:
#   soko4-1:  10 boulders,  9 pits, 0 holes
#   soko4-2:  12 boulders, 10 pits, 0 holes
#   soko3-1:  20 boulders,  0 pits,15 holes
#   soko3-2:  16 boulders,  0 pits,12 holes
#   soko2-1:  13 boulders,  0 pits,10 holes
#   soko2-2:  16 boulders,  0 pits,11 holes
#   soko1-1:  18 boulders,  0 pits,16 holes
#   soko1-2:  20 boulders,  0 pits,18 holes
#
# Legend:
#   '.' = floor      '|' / '-' = wall (we collapse both to WALL)
#   ' ' = void       '<' = up-stair        '>' = down-stair
#   '^' = pit trap   'H' = hole trap       '0' = boulder
#   '+' = closed door (vendor MAP literal; not modified by DOOR: directives)
# ---------------------------------------------------------------------------

_SOKO_LAYOUT_1A = [  # vendor: soko1-1
    "--------------------------",
    "|>......HHHHHHHHHHHHHHHH.|",
    "|.......|---------------.|",
    "-------.------         |.|",
    " |...........|         |.|",
    " |.0.0.0.0.0.|         |.|",
    "--------.-----         |.|",
    "|...0.0..0.0.|         |.|",
    "|...0........|         |.|",
    "-----.--------   ------|.|",
    " |..0.0.0...|  --|.....|.|",
    " |.....0....|  |.+.....|.|",
    " |.0.0...0.|-  |-|.....|.|",
    "-------.----   |.+.....+.|",
    "|..0.....|     |-|.....|--",
    "|........|     |.+.....|  ",
    "|...|-----     --|.....|  ",
    "-----            -------  ",
]

_SOKO_LAYOUT_1B = [  # vendor: soko1-2
    "  ------------------------",
    "  |..HHHHHHHHHHHHHHHHHH..|",
    "  |..-------------------.|",
    "----.|    -----        |.|",
    "|..|0--  --...|        |.|",
    "|.....|--|.0..|        |.|",
    "|.00..|..|..0.|        |.|",
    "--..00|...00.--        |.|",
    " |0..0...|0..|   ------|.|",
    " |.00.|..|..0| --|.....|.|",
    " |.0.0|--|.0.| |.+.....|.|",
    " |.......|..-- |-|.....|.|",
    " ----.0..|.--  |.+.....+.|",
    "    ---.--.|   |-|.....|--",
    "     |.0...|   |.+.....|  ",
    "     |>.|..|   --|.....|  ",
    "     -------     -------  ",
]

_SOKO_LAYOUT_2A = [  # vendor: soko2-1
    "--------------------",
    "|........|...|.....|",
    "|.00..-00|.-.|.....|",
    "|..|.0.0.|00.|.....|",
    "|-.|..-..|.-.|..<..|",
    "|...--.......|.....|",
    "|...|.0.-...-|.....|",
    "|.0.|0.|...--|.....|",
    "|-0.|..|----------+|",
    "|..0....HHHHHHHHHH.|",
    "|...|.>|------------",
    "--------            ",
]

_SOKO_LAYOUT_2B = [  # vendor: soko2-2
    "  --------          ",
    "--|.|....|          ",
    "|...0....|----------",
    "|.-.00-00|.|.......|",
    "|.00-......|.......|",
    "|.-..0.|...|.......|",
    "|....-0--0-|...<...|",
    "|..00..0...|.......|",
    "|.--...|...|.......|",
    "|....-0|---|.......|",
    "--|..0.|----------+|",
    "  |..0>HHHHHHHHHHH.|",
    "  ------------------",
]

_SOKO_LAYOUT_3A = [  # vendor: soko3-1
    "-----------       -----------",
    "|....|....|--     |.........|",
    "|..00|00...>|     |.........|",
    "|.....0...|--     |.........|",
    "|....|....|       |....<....|",
    "|-.---------      |.........|",
    "|..0.|.....|      |.........|",
    "|.00.|0.0.0|      |.........|",
    "|..0.....0.|      |.........|",
    "|.000|0..0.|---------------+|",
    "|....|..0.0.HHHHHHHHHHHHHHH.|",
    "-----------------------------",
]

_SOKO_LAYOUT_3B = [  # vendor: soko3-2
    " ----          -----------",
    "-|.>|-------   |.........|",
    "|..........|   |.........|",
    "|.0-----0-.|   |.........|",
    "|..|...|.0.|   |....<....|",
    "|.0.0....0-|   |.........|",
    "|.0..0..|..|   |.........|",
    "|.----0.--.|   |.........|",
    "|..0...0.|.--  |.........|",
    "|.---0-...0.------------+|",
    "|...|..0-.0.HHHHHHHHHHHH.|",
    "|..0......----------------",
    "----|..|..|               ",
    "    -------               ",
]

_SOKO_LAYOUT_4A = [  # vendor: soko4-1
    "------  ----- ",
    "|....|  |...| ",
    "|.0..----.0.| ",
    "|.0......0..| ",
    "|..|-|.|-|0.| ",
    "---------|.---",
    "|..^^^<|.....|",
    "|..----|0....|",
    "--^|   |.0...|",
    " |^|---|.0...|",
    " |..^^^^0.0..|",
    " |..|---------",
    " ----         ",
]

_SOKO_LAYOUT_4B = [  # vendor: soko4-2
    "-------- ------",
    "|<|....|-|....|",
    "|^|-.00....0..|",
    "|^||..00|.0.0.|",
    "|^||....|.....|",
    "|^|-----|0-----",
    "|^|    |......|",
    "|^-----|......|",
    "|..^^^^0000...|",
    "|..|---|......|",
    "----   --------",
]

_SOKO_LAYOUTS = (
    _SOKO_LAYOUT_1A,
    _SOKO_LAYOUT_1B,
    _SOKO_LAYOUT_2A,
    _SOKO_LAYOUT_2B,
    _SOKO_LAYOUT_3A,
    _SOKO_LAYOUT_3B,
    _SOKO_LAYOUT_4A,
    _SOKO_LAYOUT_4B,
)


def _stamp_soko_layout(layout, h: int, w: int):
    """Convert a Sokoban string layout into a terrain array.

    Returns int8[h, w]; values:
        TileType.VOID / WALL / FLOOR / STAIRCASE_UP / STAIRCASE_DOWN / ALTAR
        BOULDER_TILE (Wave-4 sentinel) for boulders.
    """
    import numpy as np
    from Nethax.nethax.constants.tiles import TileType

    terrain = np.full((h, w), int(TileType.VOID), dtype=np.int8)

    # Offset layout into upper-left corner of map.
    for r, row in enumerate(layout):
        if r >= h:
            break
        for c, ch in enumerate(row):
            if c >= w:
                break
            if ch in ("-", "|"):
                terrain[r, c] = int(TileType.WALL)
            elif ch == ".":
                terrain[r, c] = int(TileType.FLOOR)
            elif ch == "<":
                terrain[r, c] = int(TileType.STAIRCASE_UP)
            elif ch == ">":
                terrain[r, c] = int(TileType.STAIRCASE_DOWN)
            elif ch == "^":
                terrain[r, c] = int(TileType.TRAP)
            elif ch == "H":
                # Hole trap — distinct from pit, drops to next dungeon level.
                # Citation: vendor/nle/dat/sokoban.des soko3-*/soko2-*/soko1-*
                #           TRAP:"hole" directives.
                terrain[r, c] = int(TileType.HOLE)
            elif ch == "0":
                terrain[r, c] = BOULDER_TILE
            elif ch == "_":
                terrain[r, c] = int(TileType.ALTAR)
            elif ch == "+":
                # Closed door — vendor MAP literal '+' for reward-room doors.
                # Citation: vendor/nle/dat/sokoban.des soko*-1/-2 MAP blocks
                #           contain literal '+' for the reward-room doors.
                terrain[r, c] = int(TileType.CLOSED_DOOR)
            # ' ' / unknown stays VOID
    return terrain


def generate_sokoban_level(rng, floor_number: int):
    """Generate a Sokoban floor: pick a hand-designed layout.

    floor_number is 1-based (1 = deepest / final-reward floor; 4 = entry).
    Each floor has 2 variants (a/b); we pick one based on rng.

    The final reward floor (floor_number == 4 in our indexing, i.e. the
    branch's "exit" floor that yields the amulet / bag-of-holding) holds
    an altar tile as a placeholder for the artifact.

    Citation: vendor/nethack/dat/sokoban[1-4][ab].des

    Args:
        rng:          JAX PRNG key (used to pick the variant a/b).
        floor_number: 1..4, 1-based Sokoban floor index.

    Returns:
        (terrain, boulder_positions, pit_positions)
        terrain           : int8[MAP_H, MAP_W]
        boulder_positions : list[(row, col)]
        pit_positions     : list[(row, col)]
    """
    import numpy as np

    # Pick variant index 0..7.  Floors 1..4 map to slots [0..1], [2..3],
    # [4..5], [6..7].  Within each, rng picks a/b.
    f = max(1, min(4, int(floor_number))) - 1
    seed_bits = int(jax.random.bits(rng).item()) & 0xFFFFFFFF
    variant = (seed_bits & 1)
    layout_idx = f * 2 + variant
    layout = _SOKO_LAYOUTS[layout_idx]

    terrain_np = _stamp_soko_layout(layout, MAP_H, MAP_W)

    # Collect boulder + pit/hole positions for the caller (Wave 5 will use
    # these to seed proper Boulder objects in the item layer).  In vendor
    # Sokoban, the entry floor (soko4-*) uses TRAP:"pit" and the upper floors
    # (soko3-*..soko1-*) use TRAP:"hole"; both are returned in pit_positions
    # so callers see all "fall" traps.  Counts per vendor:
    #   soko4-1: 9 pits;   soko4-2: 10 pits
    #   soko3-1:15 holes;  soko3-2:12 holes
    #   soko2-1:10 holes;  soko2-2:11 holes
    #   soko1-1:16 holes;  soko1-2:18 holes
    boulder_positions: list[tuple[int, int]] = []
    pit_positions:    list[tuple[int, int]] = []
    from Nethax.nethax.constants.tiles import TileType
    h, w = terrain_np.shape
    for r in range(h):
        for c in range(w):
            t = int(terrain_np[r, c])
            if t == BOULDER_TILE:
                boulder_positions.append((r, c))
            elif t == int(TileType.TRAP) or t == int(TileType.HOLE):
                pit_positions.append((r, c))

    return jnp.asarray(terrain_np, dtype=jnp.int8), boulder_positions, pit_positions


# ---------------------------------------------------------------------------
# Quest — generic per-role layout
# ---------------------------------------------------------------------------

# Each role gets a thematic monster: a "guardian" type encountered on its
# Quest levels.  These are picked from MONSTERS by name.
# Source: vendor/nethack/dat/Qst.des per-role quest filler monsters.
_QUEST_ROLE_GUARDIAN_NAMES = {
    0:  "dwarf",         # ARCHEOLOGIST  → dwarf-themed
    1:  "kobold",        # BARBARIAN
    2:  "gnome",         # CAVEMAN
    3:  "acid blob",     # HEALER
    4:  "wraith",        # KNIGHT
    5:  "leprechaun",    # MONK
    6:  "wraith",        # PRIEST
    7:  "hobbit",        # RANGER
    8:  "leprechaun",    # ROGUE
    9:  "wraith",        # SAMURAI
    10: "gnome",         # TOURIST
    11: "wraith",        # VALKYRIE
    12: "leprechaun",    # WIZARD
}


def generate_quest_level(rng, depth: int, role: int):
    """Generate one Quest level.  Layout: small-room dungeon with role
    flavour via monster choice.  13 roles × 5 levels.

    Citation: vendor/nethack/dat/Qst.des per-role quest files
              (arc-fila.des, bar-fila.des, ... wiz-fila.des).

    Args:
        rng:   JAX PRNG key.
        depth: 1-based level index within Quest (1..5).
        role:  Role enum value 0..12.

    Returns:
        (terrain, monster_type_ids, item_type_ids)
    """
    import numpy as np
    from Nethax.nethax.constants.tiles import TileType
    from Nethax.nethax.constants.monsters import MONSTERS

    # Use generate_main_branch_l1 helpers via the rooms module — small set of
    # rooms connected by corridors gives a generic dungeon feel.  This is
    # the same layout as Main, but with role-themed monster choices.
    seed_bits = int(jax.random.bits(rng).item()) & 0xFFFFFFFF
    rs = np.random.RandomState(seed_bits)

    h, w = MAP_H, MAP_W
    terrain = np.full((h, w), int(TileType.VOID), dtype=np.int8)

    # Place 4-6 small rectangular rooms manually (deterministic, JIT-free).
    n_rooms = 4 + (depth % 3)  # 4..6
    rooms: list[tuple[int, int, int, int]] = []  # (y1, x1, y2, x2)
    attempts = 0
    while len(rooms) < n_rooms and attempts < 200:
        attempts += 1
        rh = rs.randint(3, 5)
        rw = rs.randint(5, 9)
        y1 = rs.randint(2, h - rh - 2)
        x1 = rs.randint(2, w - rw - 2)
        y2 = y1 + rh - 1
        x2 = x1 + rw - 1
        # Check non-overlap (with 1-cell margin).
        overlaps = False
        for (a1, b1, a2, b2) in rooms:
            if not (y2 + 1 < a1 or a2 + 1 < y1 or x2 + 1 < b1 or b2 + 1 < x1):
                overlaps = True
                break
        if overlaps:
            continue
        rooms.append((y1, x1, y2, x2))

    # Carve room interiors as FLOOR; ring as WALL.
    for (y1, x1, y2, x2) in rooms:
        terrain[y1 - 1:y2 + 2, x1 - 1:x2 + 2] = int(TileType.WALL)
        terrain[y1:y2 + 1, x1:x2 + 1] = int(TileType.FLOOR)

    # Connect consecutive rooms with L-shaped corridors.
    for i in range(len(rooms) - 1):
        ya = (rooms[i][0] + rooms[i][2]) // 2
        xa = (rooms[i][1] + rooms[i][3]) // 2
        yb = (rooms[i + 1][0] + rooms[i + 1][2]) // 2
        xb = (rooms[i + 1][1] + rooms[i + 1][3]) // 2
        # Horizontal then vertical.
        for c in range(min(xa, xb), max(xa, xb) + 1):
            if terrain[ya, c] != int(TileType.FLOOR):
                terrain[ya, c] = int(TileType.CORRIDOR)
        for r in range(min(ya, yb), max(ya, yb) + 1):
            if terrain[r, xb] != int(TileType.FLOOR):
                terrain[r, xb] = int(TileType.CORRIDOR)

    # Place stairs.  Up-stair in first room, down-stair in last room
    # (except depth==5 which is the nemesis floor → no down-stair).
    if rooms:
        y1, x1, y2, x2 = rooms[0]
        terrain[(y1 + y2) // 2, (x1 + x2) // 2] = int(TileType.STAIRCASE_UP)
        if depth < 5 and len(rooms) > 1:
            y1, x1, y2, x2 = rooms[-1]
            terrain[(y1 + y2) // 2, (x1 + x2) // 2] = int(TileType.STAIRCASE_DOWN)

    # Role-themed monster: one guardian + generic filler.
    role_key = int(role) % len(_QUEST_ROLE_GUARDIAN_NAMES)
    guardian_name = _QUEST_ROLE_GUARDIAN_NAMES.get(role_key, "gnome")

    monster_type_ids: list[int] = []
    for i, m in enumerate(MONSTERS):
        if m.name == guardian_name:
            monster_type_ids.append(i)
            break
    # Filler: pick the first hobbit / gnome as low-tier monster.
    for i, m in enumerate(MONSTERS):
        if m.name in ("hobbit", "gnome") and i not in monster_type_ids:
            monster_type_ids.append(i)
            break

    item_type_ids: list[int] = []
    return jnp.asarray(terrain, dtype=jnp.int8), monster_type_ids, item_type_ids


# ---------------------------------------------------------------------------
# Wave 5 Phase 2 — Valley of the Dead and Gehennom level generators
# ---------------------------------------------------------------------------
#
# Citation: vendor/nethack/dat/dungeon.lua  — "Gehennom" branch definition
#           (16 levels, lvlfill="hellfill", flags=mazelike,hellish)
#           vendor/nethack/src/trap.c       — VIBRATING_SQUARE / MAGIC_PORTAL
#           vendor/nethack/src/dungeon.c    — Is_valley / Is_sanctum / branch
#                                             linking from Castle (Main bottom)
# ---------------------------------------------------------------------------


def _find_demon_monster_ids() -> list[int]:
    """Return monster table indices for the canonical Gehennom demon roster.

    The set spans minor devils, major demons, and the named demon-princes
    (vendor/nethack/src/monst.c entries with S_DEMON / S_IMP symbols).
    """
    from Nethax.nethax.constants.monsters import MONSTERS

    demon_names = (
        "water demon", "incubus", "horned devil", "erinys", "barbed devil",
        "marilith", "vrock", "hezrou", "bone devil", "ice devil",
        "nalfeshnee", "pit fiend", "sandestin", "balrog",
        "Juiblex", "Yeenoghu", "Orcus", "Geryon", "Dispater",
        "Baalzebub", "Asmodeus", "Demogorgon",
    )
    ids: list[int] = []
    for i, m in enumerate(MONSTERS):
        if m.name in demon_names:
            ids.append(i)
    return ids


def generate_valley_of_dead(rng):
    """Generate the Valley of the Dead (Gehennom Dlvl 1).

    Wave 5 Phase 2.  The Valley is a single narrow vertical level just
    above the rest of Gehennom.  It hosts the VIBRATING_SQUARE trap that
    reveals a MAGIC_PORTAL to the deeper hellish levels, and ghostly
    opponents (wraiths, vampires, shades, ghosts).

    Layout: one tall narrow chamber centred horizontally on the map; an
    altar near the top (canonical Moloch altar location), a vibrating
    square near the bottom-centre, and an up-stair back to Main Dlvl 26.

    Citation: vendor/nethack/dat/dungeon.lua "valley" entry (base=1 in
              the Gehennom branch), vendor/nethack/src/dungeon.c
              Is_valley(), vendor/nethack/src/trap.c
              TRAP_VIBRATING_SQUARE handler.

    Args:
        rng: JAX PRNG key (unused for current deterministic layout, kept
             for API symmetry with sibling generators).

    Returns:
        (terrain, monster_type_ids, vibrating_square_pos)
        terrain               : int8[MAP_H, MAP_W]
        monster_type_ids      : list[int] — wraiths / vampires / ghosts
        vibrating_square_pos  : (row, col) — tile that hosts the trap.
    """
    import numpy as np
    from Nethax.nethax.constants.tiles import TileType
    from Nethax.nethax.constants.monsters import MONSTERS

    del rng  # canonical layout is deterministic

    h, w = MAP_H, MAP_W
    terrain = np.full((h, w), int(TileType.VOID), dtype=np.int8)

    # Narrow vertical chamber: 5 cols wide, centred horizontally.
    cx = w // 2
    x_lo, x_hi = cx - 3, cx + 3      # 7-wide chamber (incl. walls)
    y_lo, y_hi = 2, h - 3

    # Outer wall ring.
    terrain[y_lo:y_hi + 1, x_lo:x_hi + 1] = int(TileType.WALL)
    # Floor interior.
    terrain[y_lo + 1:y_hi, x_lo + 1:x_hi] = int(TileType.FLOOR)

    # Altar near the top (Moloch altar in canonical Valley).
    altar_r = y_lo + 2
    altar_c = cx
    terrain[altar_r, altar_c] = int(TileType.ALTAR)

    # Up-stair back to Main Dlvl 26.
    up_r = y_lo + 1
    up_c = cx
    terrain[up_r, up_c] = int(TileType.STAIRCASE_UP)

    # Vibrating square trap near the bottom-centre.
    vs_r = y_hi - 2
    vs_c = cx
    terrain[vs_r, vs_c] = int(TileType.TRAP)

    # Down-stair to Gehennom L2 (placed off to the side so the player must
    # first cross the vibrating square if they want the deeper portal).
    dn_r = y_hi - 1
    dn_c = cx
    terrain[dn_r, dn_c] = int(TileType.STAIRCASE_DOWN)

    # Pick ghostly monster suggestions: wraith / vampire / ghost / shade.
    target_names = ("wraith", "vampire", "ghost", "shade", "vampire lord")
    monster_type_ids: list[int] = []
    for i, m in enumerate(MONSTERS):
        if m.name in target_names:
            monster_type_ids.append(i)

    return (
        jnp.asarray(terrain, dtype=jnp.int8),
        monster_type_ids,
        (int(vs_r), int(vs_c)),
    )


def generate_gehennom_level(rng, depth: int):
    """Generate one Gehennom level (Dlvl 1..16 within the Gehennom branch).

    Dlvl 1 is delegated to generate_valley_of_dead.  Dlvls 2..16 are
    maze layouts (mazelike + hellfill per vendor dungeon.lua) populated
    with demon-class monster suggestions.  Specific named demon lairs
    (Juiblex / Yeenoghu / Orcus / Geryon / Dispater / Baalzebub /
    Asmodeus) are handled by demon_lairs.py if importable; otherwise the
    baseline maze is returned for those depths too.

    Citation: vendor/nethack/dat/dungeon.lua "Gehennom" block,
              vendor/nethack/src/mkmaze.c::mkmines / makemaz().

    Args:
        rng:   JAX PRNG key.
        depth: 1-based Gehennom level (1..16).

    Returns:
        (terrain, monster_type_ids, item_type_ids)
    """
    import numpy as np
    from Nethax.nethax.constants.tiles import TileType
    from Nethax.nethax.dungeon.mazes import (
        generate_maze_kruskal,
        TILE_WALL,
        TILE_FLOOR,
    )

    if int(depth) <= 1:
        terrain, monsters, _vs = generate_valley_of_dead(rng)
        return terrain, monsters, []

    # Optional: delegate named demon-lair depths to demon_lairs.py when
    # that module is present (other agent's Wave-5 deliverable).
    # Canonical demon-lair depths (relative to Gehennom Dlvl 1):
    #   ~5  Juiblex
    #   ~8  Yeenoghu
    #   ~10 Asmodeus
    #   ~12 Orcus
    #   ~13 Baalzebub
    #   ~14 Dispater
    #   ~15 Wizard tower (handled by special_levels.py instead)
    demon_lair_depths = {5, 8, 10, 12, 13, 14}
    if int(depth) in demon_lair_depths:
        try:
            from Nethax.nethax.dungeon import demon_lairs  # type: ignore
            generator = getattr(demon_lairs, "generate_demon_lair", None)
            if generator is not None:
                return generator(rng, int(depth))
        except Exception:
            # demon_lairs not yet wired; fall through to baseline maze.
            pass

    # ---- Baseline maze layout for non-lair Gehennom levels ----
    maze, mh, mw = generate_maze_kruskal(rng, MAP_H, MAP_W)
    # Convert the maze (0=wall, 1=floor) into our TileType encoding.
    maze_np = np.asarray(maze)
    terrain = np.full((mh, mw), int(TileType.VOID), dtype=np.int8)
    terrain[maze_np == TILE_FLOOR] = int(TileType.FLOOR)

    # Wall ring: any VOID tile adjacent to a floor becomes WALL.
    floor_mask = (terrain == int(TileType.FLOOR))
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(floor_mask, shift=(dr, dc), axis=(0, 1))
            wall_candidate = (terrain == int(TileType.VOID)) & shifted
            terrain[wall_candidate] = int(TileType.WALL)

    # Sprinkle a small handful of FIRE_TRAP tiles (hellfill flavour).
    # Use the rng to pick positions deterministically.
    seed_bits = int(jax.random.bits(rng).item()) & 0xFFFFFFFF
    rs = np.random.RandomState(seed_bits)
    floor_positions = np.argwhere(terrain == int(TileType.FLOOR))
    if len(floor_positions) > 0:
        n_traps = min(4, len(floor_positions))
        idxs = rs.choice(len(floor_positions), size=n_traps, replace=False)
        for k in idxs:
            r, c = floor_positions[k]
            terrain[r, c] = int(TileType.TRAP)

    # Pick a stair-up location (first floor tile, scan order).
    placed_up = False
    for r in range(1, mh - 1):
        for c in range(1, mw - 1):
            if terrain[r, c] == int(TileType.FLOOR):
                terrain[r, c] = int(TileType.STAIRCASE_UP)
                placed_up = True
                break
        if placed_up:
            break

    # Place stair-down on the bottom-right-most floor tile, except on the
    # deepest Gehennom level (Dlvl 16 = sanctum, no down).
    if int(depth) < 16:
        placed_dn = False
        for r in range(mh - 2, 0, -1):
            for c in range(mw - 2, 0, -1):
                if terrain[r, c] == int(TileType.FLOOR):
                    terrain[r, c] = int(TileType.STAIRCASE_DOWN)
                    placed_dn = True
                    break
            if placed_dn:
                break

    monster_type_ids = _find_demon_monster_ids()
    item_type_ids: list[int] = []
    return jnp.asarray(terrain, dtype=jnp.int8), monster_type_ids, item_type_ids


# ---------------------------------------------------------------------------
# Helper: write a BranchGraphState into a DungeonState's stair_links field.
# Used by tests + future env construction code.
# ---------------------------------------------------------------------------

def apply_branch_graph_to_dungeon(
    dungeon: DungeonState, graph: BranchGraphState
) -> DungeonState:
    """Project BranchGraphState.stair_links onto DungeonState.stair_links.

    The DungeonState schema is [N_BRANCHES, MAX_LEVELS, 2 (dir), 2 (endpoint)].
    BranchGraphState only tracks the down-stair endpoint per (branch, level);
    we infer the up-stair endpoint by reading the destination's link.

    Citation: vendor/nethack/src/dungeon.c::init_dungeons cross-wiring.
    """
    # graph.stair_links[src_branch, src_level-1] = [dst_branch, dst_level]
    # Treat this as the *down* link (direction=1).  For each populated entry
    # we also wire the symmetric *up* link at the destination.
    sl = dungeon.stair_links
    n_b = graph.stair_links.shape[0]
    n_l = graph.stair_links.shape[1]

    # Use Python loops here because this is non-JIT init-time code.
    for b in range(n_b):
        for lv in range(n_l):
            dst_b = int(graph.stair_links[b, lv, 0])
            dst_l = int(graph.stair_links[b, lv, 1])
            if dst_b < 0 or dst_l < 0:
                continue
            # Decide whether this link is the up or down direction:
            #   if dst is on a sibling branch (different parent path), the
            #   src side hosts the branch-entry: src goes "down" into dst.
            # Heuristic: if src has a parent and dst is its parent, it's an
            # up-link; otherwise treat as down.
            is_up_from_src = (
                graph.parent_branch[b] != jnp.int8(-1) and
                int(graph.parent_branch[b]) == dst_b
            )
            direction = 0 if bool(is_up_from_src) else 1
            sl = sl.at[b, lv, direction].set(
                jnp.array([dst_b, dst_l], dtype=jnp.int8)
            )

    # ---- Wave 5 Phase 2: Gehennom internal descents (L_n <-> L_{n+1}) ----
    # BranchGraphState's single-slot table can't encode both directions on
    # the same level, so we wire the 15 in-branch links directly.
    # Citation: vendor/nethack/dat/dungeon.lua "Gehennom" levels block.
    gehennom_levels = 16
    for lv in range(gehennom_levels - 1):
        # L(lv+1) down-stair -> L(lv+2)
        sl = sl.at[int(Branch.GEHENNOM), lv, 1].set(
            jnp.array([int(Branch.GEHENNOM), lv + 2], dtype=jnp.int8)
        )
        # L(lv+2) up-stair -> L(lv+1)
        sl = sl.at[int(Branch.GEHENNOM), lv + 1, 0].set(
            jnp.array([int(Branch.GEHENNOM), lv + 1], dtype=jnp.int8)
        )

    return dungeon.replace(stair_links=sl)


# ---------------------------------------------------------------------------
# init_dungeons fixed ISAAC64 pre-draws
# ---------------------------------------------------------------------------

def consume_init_dungeons_draws(vendor_rng):
    """Replay the ~18 fixed ISAAC64 draws of vendor ``init_dungeons``.

    These draws occur in vendor ``vendor/nle/src/dungeon.c::init_dungeons``
    (line 714) BEFORE ``mklev`` runs, and must be consumed in byte-exact order
    to keep Nethax's ISAAC64 stream aligned with NLE.

    Draw breakdown (18 total):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Phase            │ Count │ Vendor cite                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Dungeon depths   │  4    │ dungeon.c:796-798  rn1(rand, base)           │
    │   DoD   rn2(5)   │       │   "The Dungeons of Doom" (25, 5) → rand=5   │
    │   Gehennom rn2(5)│       │   "Gehennom"            (20, 5) → rand=5   │
    │   Mines rn2(2)   │       │   "The Gnomish Mines"   (8,  2) → rand=2   │
    │   Quest rn2(2)   │       │   "The Quest"           (5,  2) → rand=2   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Fort Ludios gate │  1    │ dungeon.c:775-776  rn2(100) chance gate      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ RNDLEVEL gates   │  8    │ dungeon.c:548      rn2(100) per RNDLEVEL     │
    │   bigrm          │       │   chance=40 in dungeon.def                   │
    │   medusa         │       │   chance=4                                   │
    │   minetn         │       │   chance=7                                   │
    │   minend         │       │   chance=3                                   │
    │   soko1..soko4   │       │   chance=2 each (4 draws)                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Tune string      │  5    │ dungeon.c:917-918  rn2(7) × 5                │
    └─────────────────────────────────────────────────────────────────────────┘

    The variable-count draws (``place_level`` slot picks at dungeon.c:661 and
    ``parent_dlevel`` branch picks at dungeon.c:398) are seed-dependent and are
    deferred to a later wave.

    Args:
        vendor_rng: Isaac64State from ``Nethax.nethax.vendor_rng``.

    Returns:
        ``(new_vendor_rng, dungeon_state)`` where ``dungeon_state`` is a dict
        holding the sampled scalar values keyed by name (informational only;
        the caller is not required to use them).

    Citation:
        vendor/nle/src/dungeon.c:775-776  (Fort Ludios chance gate)
        vendor/nle/src/dungeon.c:796-798  (dungeon depth rn1 draws)
        vendor/nle/src/dungeon.c:548      (RNDLEVEL chance gate)
        vendor/nle/src/dungeon.c:917-918  (tune string rn2(7) × 5)
    """
    from Nethax.nethax import vendor_rng as _vrng

    # --- Phase 1: dungeon depth draws (dungeon.c:796-798) ---
    # Dungeons with lev.rand > 0 each get one rn1(rand, base) = rn2(rand)+base.
    # Fired in dungeon.def parse order: DoD, Gehennom, Mines, Quest.
    vendor_rng, dod_levels    = _vrng.rn2(vendor_rng, 5)   # dungeon.c:797 DoD   (25,5)
    vendor_rng, geh_levels    = _vrng.rn2(vendor_rng, 5)   # dungeon.c:797 Gehennom (20,5)
    vendor_rng, mines_levels  = _vrng.rn2(vendor_rng, 2)   # dungeon.c:797 Mines (8,2)
    vendor_rng, quest_levels  = _vrng.rn2(vendor_rng, 2)   # dungeon.c:797 Quest (5,2)

    # --- Phase 2: Fort Ludios chance gate (dungeon.c:775-776) ---
    # ``if (!wizard && pd.tmpdungeon[i].chance && (pd.tmpdungeon[i].chance <= rn2(100)))``
    # Fort Ludios is the only dungeon in dungeon.def with chance > 0.
    vendor_rng, ludios_gate   = _vrng.rn2(vendor_rng, 100)  # dungeon.c:776

    # --- Phase 3: RNDLEVEL chance gates (dungeon.c:548) ---
    # ``init_level``: ``if (!wizard && tlevel->chance <= rn2(100)) return;``
    # One draw per RNDLEVEL entry in dungeon.def, in definition order:
    #   bigrm (DoD, chance=40), medusa (DoD, chance=4),
    #   minetn (Mines, chance=7), minend (Mines, chance=3),
    #   soko1..soko4 (Sokoban, chance=2 each).
    vendor_rng, gate_bigrm    = _vrng.rn2(vendor_rng, 100)  # dungeon.c:548 bigrm
    vendor_rng, gate_medusa   = _vrng.rn2(vendor_rng, 100)  # dungeon.c:548 medusa
    vendor_rng, gate_minetn   = _vrng.rn2(vendor_rng, 100)  # dungeon.c:548 minetn
    vendor_rng, gate_minend   = _vrng.rn2(vendor_rng, 100)  # dungeon.c:548 minend
    vendor_rng, gate_soko1    = _vrng.rn2(vendor_rng, 100)  # dungeon.c:548 soko1
    vendor_rng, gate_soko2    = _vrng.rn2(vendor_rng, 100)  # dungeon.c:548 soko2
    vendor_rng, gate_soko3    = _vrng.rn2(vendor_rng, 100)  # dungeon.c:548 soko3
    vendor_rng, gate_soko4    = _vrng.rn2(vendor_rng, 100)  # dungeon.c:548 soko4

    # --- Phase 4: tune string (dungeon.c:917-918) ---
    # ``for (i = 0; i < 5; i++) tune[i] = 'A' + rn2(7);``
    # Five unconditional rn2(7) draws; we consume but do not store the tune.
    vendor_rng, _tune0        = _vrng.rn2(vendor_rng, 7)    # dungeon.c:918 tune[0]
    vendor_rng, _tune1        = _vrng.rn2(vendor_rng, 7)    # dungeon.c:918 tune[1]
    vendor_rng, _tune2        = _vrng.rn2(vendor_rng, 7)    # dungeon.c:918 tune[2]
    vendor_rng, _tune3        = _vrng.rn2(vendor_rng, 7)    # dungeon.c:918 tune[3]
    vendor_rng, _tune4        = _vrng.rn2(vendor_rng, 7)    # dungeon.c:918 tune[4]

    dungeon_state = {
        "dod_levels":   dod_levels   + 25,  # rn2(5)+25 ∈ [25, 29]
        "geh_levels":   geh_levels   + 20,  # rn2(5)+20 ∈ [20, 24]
        "mines_levels": mines_levels + 8,   # rn2(2)+8  ∈ [8,  9]
        "quest_levels": quest_levels + 5,   # rn2(2)+5  ∈ [5,  6]
        "ludios_gate":  ludios_gate,
        "gate_bigrm":   gate_bigrm,
        "gate_medusa":  gate_medusa,
        "gate_minetn":  gate_minetn,
        "gate_minend":  gate_minend,
        "gate_soko1":   gate_soko1,
        "gate_soko2":   gate_soko2,
        "gate_soko3":   gate_soko3,
        "gate_soko4":   gate_soko4,
    }
    return vendor_rng, dungeon_state


def consume_init_dungeons_variable_draws(vendor_rng, dungeon_state):
    """Replay the variable ISAAC64 draws of vendor ``init_dungeons``.

    These follow immediately after the 18 fixed draws consumed by
    ``consume_init_dungeons_draws`` and cover:

    * ``place_level`` slot-picks — ``dungeon.c:661``
      One ``rn2(npossible)`` per special level that is created (not gated
      out).  Called once per dungeon in dungeon.def parse order, after the
      dungeon's ``init_level`` pass.
    * ``parent_dlevel`` branch-attachment picks — ``dungeon.c:398``
      One ``rn2(num)`` per non-Main dungeon (``i > 0``) to pick where in the
      parent dungeon the branch staircase lands.  Called via ``add_branch``
      at ``dungeon.c:502`` before the dungeon's ``init_level``/``place_level``
      pass.

    Draw order exactly mirrors the vendor ``init_dungeons`` loop
    (``dungeon.c:772-913``): for each dungeon ``i``:
      1. ``add_branch`` → ``parent_dlevel`` rn2 (i > 0 only)  [dungeon.c:502,398]
      2. ``place_level`` rn2 per created special level          [dungeon.c:661]

    Conditional draws (RNDLEVEL gates and the Fort Ludios dungeon gate) use
    the scalar gate values already drawn by the fixed phase: a level exists
    iff its ``gate_draw < chance`` (vendor condition: ``chance <= rn2(100)``
    skips, so placed when ``rn2(100) < chance``).

    Standard NLE config draw count (per-dungeon breakdown):
    ┌────────────────────────┬──────────┬───────────────────────────────────────────────────────────────┐
    │ Dungeon                │  Fixed   │  Variable draws                                               │
    │                        │ parent   │                                                               │
    ├────────────────────────┼──────────┼───────────────────────────────────────────────────────────────┤
    │ DoD (i=0)              │    —     │ rogue rn2(4) + oracle rn2(5) + [bigrm rn2(3)] +              │
    │                        │          │ [medusa rn2(4)] + castle rn2(1)  = 3–5 draws                 │
    │ Gehennom (i=1)         │  rn2(1)  │ valley rn2(1) + sanctum rn2(1) + juiblex rn2(4) +           │
    │                        │          │ baalz rn2(4) + asmodeus rn2(6) + wizard1 rn2(6) +           │
    │                        │          │ wizard2 rn2(1) + wizard3 rn2(1) + orcus rn2(6) +            │
    │                        │          │ fakewiz1 rn2(4) + fakewiz2 rn2(3)  = 11 draws               │
    │ Mines (i=2)            │  rn2(3)  │ [minetn rn2(2)] + [minend rn2(1)]  = 0–2 draws              │
    │ Quest (i=3)            │  rn2(2)  │ x-strt rn2(1) + x-loca rn2(1) + x-goal rn2(1)  = 3 draws   │
    │ Sokoban (i=4)          │  rn2(1)  │ [soko1 rn2(1)]+[soko2 rn2(1)]+[soko3 rn2(1)]+             │
    │                        │          │ [soko4 rn2(1)]  = 0–4 draws                                 │
    │ Fort Ludios (i=5)      │ [rn2(4)] │ [knox rn2(1)]  = 0–2 draws (gated by ludios_gate)          │
    │ Vlad's Tower (i=6)     │  rn2(5)  │ tower1 rn2(1)+tower2 rn2(1)+tower3 rn2(1)  = 3 draws       │
    │ Endgame (i=7)          │  rn2(1)  │ astral rn2(1)+water rn2(1)+fire rn2(1)+air rn2(1)+         │
    │                        │          │ earth rn2(1)+dummy rn2(1)  = 6 draws                        │
    ├────────────────────────┼──────────┼───────────────────────────────────────────────────────────────┤
    │ Total (all gates pass) │  7 draws │  3-5 + 11 + 0-2 + 3 + 0-4 + 0-1 + 3 + 6  = 26–35 draws    │
    │ Grand variable total   │          │  33–42 draws (parent_dlevel + place_level combined)           │
    └────────────────────────┴──────────┴───────────────────────────────────────────────────────────────┘

    Args:
        vendor_rng:    Isaac64State carrying the ISAAC64 CORE stream, positioned
                       immediately after the 18 fixed draws.
        dungeon_state: dict returned by ``consume_init_dungeons_draws`` — provides
                       the gate draw values needed to decide which RNDLEVEL and
                       Fort Ludios levels were created.

    Returns:
        ``(new_vendor_rng, variable_state)`` where ``variable_state`` is an
        informational dict of the drawn values.

    Citations:
        vendor/nle/src/dungeon.c:398   parent_dlevel — rn2(num)
        vendor/nle/src/dungeon.c:502   add_branch calls parent_dlevel
        vendor/nle/src/dungeon.c:661   place_level — rn2(npossible)
        vendor/nle/src/dungeon.c:772   init_dungeons outer loop
        vendor/nle/src/dungeon.c:845   add_branch called for i > 0
        vendor/nle/src/dungeon.c:887-898  place_level called per dungeon
    """
    from Nethax.nethax import vendor_rng as _vrng

    ds = dungeon_state  # shorthand

    # Vendor gate condition: ``if (!wizard && tlevel->chance <= rn2(100)) return``
    # Level is PLACED when the drawn value is LESS THAN chance.
    bigrm_placed  = int(ds["gate_bigrm"])  < 40
    medusa_placed = int(ds["gate_medusa"]) < 4
    minetn_placed = int(ds["gate_minetn"]) < 7
    minend_placed = int(ds["gate_minend"]) < 3
    soko1_placed  = int(ds["gate_soko1"])  < 2
    soko2_placed  = int(ds["gate_soko2"])  < 2
    soko3_placed  = int(ds["gate_soko3"])  < 2
    soko4_placed  = int(ds["gate_soko4"])  < 2

    # Fort Ludios dungeon gate: vendor dungeon.c:775-776
    #   ``if (!wizard && pd.tmpdungeon[i].chance && (pd.tmpdungeon[i].chance <= rn2(100)))``
    # ludios_gate is the rn2(100) draw; Ludios is skipped when gate >= 10.
    # (Fort Ludios chance field in compiled dungeon binary = 10.)
    ludios_placed = int(ds["ludios_gate"]) < 10

    drawn = {}

    # -----------------------------------------------------------------------
    # i=0: DoD — no add_branch; place_level draws for DoD special levels
    # dungeon.def order: rogue, oracle, [bigrm], [medusa], castle
    # vendor/nle/src/dungeon.c:887-898 (place_level after init_level loop)
    # vendor/nle/src/dungeon.c:661     (rn2(npossible) inside place_level)
    # -----------------------------------------------------------------------
    vendor_rng, v = _vrng.rn2(vendor_rng, 4)   # dungeon.c:661 rogue @ (15,4) npossible=4
    drawn["dod_rogue"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 5)   # dungeon.c:661 oracle @ (5,5) npossible=5
    drawn["dod_oracle"] = v
    if bigrm_placed:
        vendor_rng, v = _vrng.rn2(vendor_rng, 3)  # dungeon.c:661 bigrm @ (10,3) npossible=3
        drawn["dod_bigrm"] = v
    if medusa_placed:
        vendor_rng, v = _vrng.rn2(vendor_rng, 4)  # dungeon.c:661 medusa @ (-5,4) npossible=4
        drawn["dod_medusa"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 castle @ (-1,0) npossible=1
    drawn["dod_castle"] = v

    # -----------------------------------------------------------------------
    # i=1: Gehennom — add_branch first, then place_level draws
    # parent_dlevel: CHAINBRANCH "castle"+(0,0) → level_range returns 1
    # vendor/nle/src/dungeon.c:502,398
    # dungeon.def order: valley, sanctum, juiblex, baalz, asmodeus,
    #                    wizard1, wizard2, wizard3, orcus, fakewiz1, fakewiz2
    # -----------------------------------------------------------------------
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:398 parent_dlevel Gehennom num=1
    drawn["pdl_gehennom"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 valley @ (1,0) npossible=1
    drawn["geh_valley"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 sanctum @ (-1,0) npossible=1
    drawn["geh_sanctum"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 4)   # dungeon.c:661 juiblex @ (4,4) npossible=4
    drawn["geh_juiblex"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 4)   # dungeon.c:661 baalz @ (6,4) npossible=4
    drawn["geh_baalz"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 6)   # dungeon.c:661 asmodeus @ (2,6) npossible=6
    drawn["geh_asmodeus"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 6)   # dungeon.c:661 wizard1 @ (11,6) npossible=6
    drawn["geh_wizard1"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 wizard2 CHAINLEVEL+1 npossible=1
    drawn["geh_wizard2"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 wizard3 CHAINLEVEL+2 npossible=1
    drawn["geh_wizard3"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 6)   # dungeon.c:661 orcus @ (10,6) npossible=6
    drawn["geh_orcus"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 4)   # dungeon.c:661 fakewiz1 @ (-6,4) npossible=4
    drawn["geh_fakewiz1"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 3)   # dungeon.c:661 fakewiz2 @ (-6,4) npossible=3
    drawn["geh_fakewiz2"] = v                   # npossible=3: range 4 minus fakewiz1 slot

    # -----------------------------------------------------------------------
    # i=2: Gnomish Mines — add_branch (BRANCH @ (2,3)), then place_level
    # parent_dlevel: level_range returns min(3, dod_depth - 2 + 1); typical=3
    # vendor/nle/src/dungeon.c:502,398
    # -----------------------------------------------------------------------
    vendor_rng, v = _vrng.rn2(vendor_rng, 3)   # dungeon.c:398 parent_dlevel Mines num=3
    drawn["pdl_mines"] = v
    if minetn_placed:
        vendor_rng, v = _vrng.rn2(vendor_rng, 2)  # dungeon.c:661 minetn @ (3,2) npossible=2
        drawn["mines_minetn"] = v
    if minend_placed:
        vendor_rng, v = _vrng.rn2(vendor_rng, 1)  # dungeon.c:661 minend @ (-1,0) npossible=1
        drawn["mines_minend"] = v

    # -----------------------------------------------------------------------
    # i=3: The Quest — add_branch (CHAINBRANCH "oracle"+(6,2)), then place_level
    # parent_dlevel: oracle_pos + [6,7], capped at dod end; npossible typically 2
    # dungeon.def order: x-strt @ (1,1), x-loca @ (3,1), x-goal @ (-1,0)
    # vendor/nle/src/dungeon.c:502,398
    # -----------------------------------------------------------------------
    vendor_rng, v = _vrng.rn2(vendor_rng, 2)   # dungeon.c:398 parent_dlevel Quest num=2
    drawn["pdl_quest"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 x-strt @ (1,1) npossible=1
    drawn["quest_strt"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 x-loca @ (3,1) npossible=1
    drawn["quest_loca"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 x-goal @ (-1,0) npossible=1
    drawn["quest_goal"] = v

    # -----------------------------------------------------------------------
    # i=4: Sokoban — add_branch (CHAINBRANCH "oracle"+(1,0)), then place_level
    # parent_dlevel: oracle_pos + 1, npossible=1
    # soko1-4 each RNDLEVEL @ (N,0) chance=2; rand=0 → npossible=1 if placed
    # vendor/nle/src/dungeon.c:502,398
    # -----------------------------------------------------------------------
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:398 parent_dlevel Sokoban num=1
    drawn["pdl_sokoban"] = v
    if soko1_placed:
        vendor_rng, v = _vrng.rn2(vendor_rng, 1)  # dungeon.c:661 soko1 @ (1,0) npossible=1
        drawn["soko_1"] = v
    if soko2_placed:
        vendor_rng, v = _vrng.rn2(vendor_rng, 1)  # dungeon.c:661 soko2 @ (2,0) npossible=1
        drawn["soko_2"] = v
    if soko3_placed:
        vendor_rng, v = _vrng.rn2(vendor_rng, 1)  # dungeon.c:661 soko3 @ (3,0) npossible=1
        drawn["soko_3"] = v
    if soko4_placed:
        vendor_rng, v = _vrng.rn2(vendor_rng, 1)  # dungeon.c:661 soko4 @ (4,0) npossible=1
        drawn["soko_4"] = v

    # -----------------------------------------------------------------------
    # i=5: Fort Ludios — entire dungeon conditional on ludios_placed
    # add_branch (BRANCH @ (18,4) in Main), npossible=4; knox LEVEL @ (-1,0)
    # vendor/nle/src/dungeon.c:502,398
    # -----------------------------------------------------------------------
    if ludios_placed:
        vendor_rng, v = _vrng.rn2(vendor_rng, 4)  # dungeon.c:398 parent_dlevel Ludios num=4
        drawn["pdl_ludios"] = v
        vendor_rng, v = _vrng.rn2(vendor_rng, 1)  # dungeon.c:661 knox @ (-1,0) npossible=1
        drawn["ludios_knox"] = v

    # -----------------------------------------------------------------------
    # i=6: Vlad's Tower — add_branch (BRANCH @ (9,5) in Gehennom), npossible=5
    # dungeon.def order: tower1 @ (1,0), tower2 @ (2,0), tower3 @ (3,0)
    # vendor/nle/src/dungeon.c:502,398
    # -----------------------------------------------------------------------
    vendor_rng, v = _vrng.rn2(vendor_rng, 5)   # dungeon.c:398 parent_dlevel Vlad num=5
    drawn["pdl_vlad"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 tower1 @ (1,0) npossible=1
    drawn["vlad_tower1"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 tower2 @ (2,0) npossible=1
    drawn["vlad_tower2"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 tower3 @ (3,0) npossible=1
    drawn["vlad_tower3"] = v

    # -----------------------------------------------------------------------
    # i=7: The Elemental Planes (Endgame) — add_branch (BRANCH @ (1,0) in Main)
    # parent_dlevel: num=1; astral-dummy all @ fixed slots npossible=1
    # vendor/nle/src/dungeon.c:502,398
    # -----------------------------------------------------------------------
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:398 parent_dlevel Endgame num=1
    drawn["pdl_endgame"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 astral @ (1,0) npossible=1
    drawn["end_astral"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 water @ (2,0) npossible=1
    drawn["end_water"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 fire @ (3,0) npossible=1
    drawn["end_fire"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 air @ (4,0) npossible=1
    drawn["end_air"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 earth @ (5,0) npossible=1
    drawn["end_earth"] = v
    vendor_rng, v = _vrng.rn2(vendor_rng, 1)   # dungeon.c:661 dummy @ (6,0) npossible=1
    drawn["end_dummy"] = v

    return vendor_rng, drawn


# ---------------------------------------------------------------------------
# TODO blocks
# ---------------------------------------------------------------------------
# Wave 4:
#   - traverse_stair: handle BR_PORTAL branches (Quest, Sokoban) which
#     use magic portals rather than physical staircases.
#   - enter_branch: lazy level generation on first visit.
#   - Add Vibrating Square portal logic for ENDGAME entry.
#   - Populate branch entrance positions on main dungeon map (random within
#     first_level ± tolerance, matching dungeon.c level_range()).
#
# Wave 5:
#   - VLAD's Tower: BR_NO_END2 means no down-ladder in Gehennom.
#   - ENDGAME: BR_NO_END1 means no stair from top — teleport to Astral Plane.
#   - Boulder/pit objects: migrate BOULDER_TILE sentinel into the proper
#     item layer once the boulder system is online.
#   - Proper Sokoban .des parser (sokoban*.des).
#   - 13 unique Quest layouts (Qst.des per role).
