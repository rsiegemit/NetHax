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

import ctypes
from enum import IntEnum
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from flax import struct


# ---------------------------------------------------------------------------
# macOS qsort tie-break emulation
#
# Vendor's ``sort_rooms()`` (mklev.c:707) calls ``qsort()`` from libSystem
# with a comparator that keys on ``lx`` only.  macOS's BSD qsort is NOT
# stable — when two rooms share an ``lx`` value, qsort partitioning can
# swap them relative to creation order.  ``jnp.argsort(stable=True)``
# preserves creation order on ties, producing a different room order than
# vendor; seed=5 has tied lx=68 rooms and this swap drives a
# whole-level layout divergence.
#
# We reproduce vendor's qsort behavior via a host-side ``pure_callback``
# that invokes libSystem's actual qsort.  The shape is static (bounded
# room count) so the callback is JIT-friendly.
# ---------------------------------------------------------------------------

_libc = ctypes.CDLL("libSystem.dylib")
_libc.qsort.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p),
]
_libc.qsort.restype = None

_CMP_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)


def _cmp_int32_key(a, b):
    # First int32 of each 8-byte (key, orig_idx) record is the key.
    x = ctypes.cast(a, ctypes.POINTER(ctypes.c_int32))[0]
    y = ctypes.cast(b, ctypes.POINTER(ctypes.c_int32))[0]
    if x < y:
        return -1
    if x > y:
        return 1
    return 0


_CMP_INT32_CB = _CMP_TYPE(_cmp_int32_key)


def _host_qsort_argsort_int32(keys: np.ndarray) -> np.ndarray:
    """Argsort ``keys`` using macOS libSystem qsort (tie behavior matches
    vendor's sort_rooms()).  Returns the int32 permutation."""
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    n = keys.shape[0]
    arr = np.empty(n, dtype=[("k", "<i4"), ("o", "<i4")])
    arr["k"] = keys
    arr["o"] = np.arange(n, dtype=np.int32)
    _libc.qsort(arr.ctypes.data, n, 8, _CMP_INT32_CB)
    return arr["o"].astype(np.int32)


def _argsort_macos_qsort(keys: jnp.ndarray) -> jnp.ndarray:
    """JAX-callable wrapper around libSystem qsort.  Replaces
    ``jnp.argsort(stable=True)`` for sort_rooms parity (see
    ``_sort_rooms_by_lx``)."""
    return jax.pure_callback(
        _host_qsort_argsort_int32,
        jax.ShapeDtypeStruct(keys.shape, jnp.int32),
        keys.astype(jnp.int32),
    )

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
# Vendor levl.typ -> internal TileType lookup (byte-parity render path).
#
# The byte-parity reset path builds ``vendor_levl_grid`` (a [ROWNO, COLNO]
# int8 array in vendor ``levl[][].typ`` encoding, via corridors.py's
# ``stamp_rooms_into_typ`` + ``makecorridors`` + ``make_niches``).  The codes
# that actually appear in that grid for Main Dlvl 1 are:
#   STONE=0, VWALL=1, HWALL=2, ROOM=11, CORR=13, SCORR=14, DOOR=15
# (corridors.py uses CUSTOM small codes here, NOT vendor rm.h values — see
# Nethax/nethax/dungeon/corridors.py:272-281).  We also map the corner codes
# (3..6, in case stamp_rooms ever emits them), SDOOR(16) and IRONBARS(21) for
# completeness.  Mapping rationale (target = TileType):
#   STONE   (0)  -> VOID(0)       unexplored / rock
#   VWALL   (1)  -> WALL(3)       \
#   HWALL   (2)  -> WALL(3)        > all wall codes block + render as wall;
#   TLCORN..BRCORN (3..6) -> WALL  / _apply_wall_angle derives the variant.
#   ROOM    (11) -> FLOOR(1)      room floor
#   CORR    (13) -> CORRIDOR(2)   open corridor
#   SCORR   (14) -> VOID(0)       SECRET corridor: vendor back_to_glyph renders
#                                 it as S_stone and it is not walkable until
#                                 found, so VOID (not CORRIDOR) is both render-
#                                 and movement-correct.  Cite: rm.h SCORR is a
#                                 secret type; vendor/nethack/src/display.c
#                                 back_to_glyph maps SCORR->S_stone.
#   DOOR    (15) -> by door MASK (see below).  When the per-cell ``doormask``
#                                 is threaded in, a DOOR cell maps as:
#                                   D_NODOOR / D_BROKEN / D_ISOPEN -> OPEN_DOOR
#                                       (a passable, wall-ish door tile; vendor
#                                       renders a doorless gap as S_ndoor '.',
#                                       which the obs cmap table cannot express
#                                       — see the S_ndoor caveat in the
#                                       door-mask block below)
#                                   D_CLOSED / D_LOCKED -> CLOSED_DOOR ('+')
#                                 With no doormask (legacy path) DOOR falls back
#                                 to CLOSED_DOOR, the prior safe default.
#   SDOOR   (16) -> WALL(3)       secret door looks like a wall until found.
#   IRONBARS(21) -> WALL(3)       no IRONBARS TileType; closest blocking tile.
# Corner cells in this grid are encoded as ROOM (corridors.py stamp_rooms emits
# ROOM, not a corner code) so a separate geometric corner-promotion pass below
# turns the room-coded corner cells into WALL so _apply_wall_angle can pick the
# correct S_tlcorn/S_trcorn/S_blcorn/S_brcorn variant.
# ---------------------------------------------------------------------------
_VTILE_STONE:    int = 0
_VTILE_VWALL:    int = 1
_VTILE_HWALL:    int = 2
_VTILE_ROOM:     int = 11
_VTILE_CORR:     int = 13
_VTILE_SCORR:    int = 14
_VTILE_DOOR:     int = 15
_VTILE_SDOOR:    int = 16

# Vendor door masks (vendor/nle/include/rm.h:323-327): a DOOR cell's
# ``doormask`` says whether it is a doorless opening, broken, open, closed, or
# locked.  Vendor display.c::back_to_glyph renders each mask distinctly:
#   D_NODOOR / D_BROKEN -> S_ndoor   (doorless gap, char '.')  -> DOORWAY
#   D_ISOPEN            -> S_vodoor  (open door, '|'/'-')       -> OPEN_DOOR
#   D_CLOSED / D_LOCKED -> S_vcdoor  (shut door, '+')           -> CLOSED_DOOR
_DMASK_BROKEN: int = 1   # D_BROKEN  — broken door, doorless opening ('.')
_DMASK_ISOPEN: int = 2   # D_ISOPEN  — open door ('|'/'-')
_DMASK_CLOSED: int = 4   # D_CLOSED  — shut door ('+')
_DMASK_LOCKED: int = 8   # D_LOCKED  — locked shut door ('+')

_TILE_VOID:        int = 0
_TILE_CORRIDOR:    int = 2
_TILE_WALL:        int = 3
_TILE_CLOSED_DOOR: int = 4
_TILE_OPEN_DOOR:   int = 5
_TILE_DOORWAY:     int = 23  # doorless doorway (D_NODOOR / D_BROKEN); S_ndoor
# _TILE_FLOOR (= 1) is defined above (module-level tile constants).

# Static index-by-grid lookup (length 32 covers STONE..IRONBARS=21, clamped).
_VTYP_TO_TILE: jnp.ndarray = jnp.array(
    [
        _TILE_VOID,         # 0  STONE
        _TILE_WALL,         # 1  VWALL
        _TILE_WALL,         # 2  HWALL
        _TILE_WALL,         # 3  TLCORNER
        _TILE_WALL,         # 4  TRCORNER
        _TILE_WALL,         # 5  BLCORNER
        _TILE_WALL,         # 6  BRCORNER
        _TILE_WALL,         # 7  (CROSSWALL-ish, unused here)
        _TILE_WALL,         # 8  (TUWALL-ish)
        _TILE_WALL,         # 9  (TDWALL-ish)
        _TILE_WALL,         # 10 (TLWALL-ish)
        _TILE_FLOOR,        # 11 ROOM
        _TILE_WALL,         # 12 (DBWALL-ish)
        _TILE_CORRIDOR,     # 13 CORR
        _TILE_VOID,         # 14 SCORR (secret -> renders as stone)
        _TILE_CLOSED_DOOR,  # 15 DOOR
        _TILE_WALL,         # 16 SDOOR (secret door -> looks like wall)
        _TILE_VOID,         # 17 (unused)
        _TILE_VOID,         # 18 (unused)
        _TILE_VOID,         # 19 (unused)
        _TILE_VOID,         # 20 (unused)
        _TILE_WALL,         # 21 IRONBARS -> closest blocking tile
        _TILE_VOID,         # 22 (clamp tail)
        _TILE_VOID, _TILE_VOID, _TILE_VOID, _TILE_VOID, _TILE_VOID,
        _TILE_VOID, _TILE_VOID, _TILE_VOID, _TILE_VOID,
    ],
    dtype=jnp.int8,
)


def _vendor_grid_to_terrain(
    vendor_levl_grid: jnp.ndarray,
    door_mask_grid: jnp.ndarray = None,
) -> jnp.ndarray:
    """Map a vendor ``levl.typ`` grid ([ROWNO, COLNO]) to internal TileType.

    Steps (all JIT-safe, no Python branches on traced values):
      1. Static lookup ``_VTYP_TO_TILE[clip(grid, 0, 31)]`` -> base TileType.
         Corners are already stamped with their proper vendor codes
         (TLCORNER=3 / TRCORNER=4 / BLCORNER=5 / BRCORNER=6) by
         ``stamp_rooms_into_typ`` and the vault stamp block in
         ``generate_main_branch_l1``, all of which map directly to WALL via
         ``_VTYP_TO_TILE``.  No geometric corner-promotion is required — an
         earlier promotion pass that derived corners from "ROOM cell flanked
         by an H-wall segment E/W and a V-wall segment N/S" misfired on
         interior ROOM cells with a door on the W wall and a door on the S
         wall (the test treated both as wall continuations), wrongly
         upgrading the cell to WALL.  Concrete regression: seed=4 cell
         (col=26, row=5) — interior of rooms[3] (lx=26, hx=28, ly=3, hy=5)
         with a SDOOR at (col=25, row=5) and a DOOR at (col=26, row=6) —
         got promoted to WALL, breaking the graffiti loop's
         ``terrain[ry,cx] == FLOOR`` proxy for ``levl[x][y].typ == ROOM``
         and forcing one extra ``rn2(40)`` draw at byte-parity stream
         position 1833.  Encoding corners directly removes the failure
         mode.

      2. Door-mask refinement (only when ``door_mask_grid`` is supplied):
         step 1 maps every DOOR cell to CLOSED_DOOR, but vendor ``dosdoor``
         assigns a per-cell door MASK (D_NODOOR / D_BROKEN / D_ISOPEN /
         D_CLOSED / D_LOCKED — vendor/nle/include/rm.h).  Only D_CLOSED /
         D_LOCKED render as a shut door ('+'); a D_NODOOR / D_BROKEN / D_ISOPEN
         doorway is passable and maps to OPEN_DOOR.  We map by mask so e.g.
         seed-0's hero-room left-wall doorway at (69,10) — which vendor leaves
         D_NODOOR — no longer renders as a spurious closed door.  See the
         door-mask block below for the S_ndoor caveat (the obs cmap table has
         no S_ndoor TileType; OPEN_DOOR is used because it is both passable and
         wall-ish so the neighbouring room corner still resolves to S_blcorn).

    Args:
        vendor_levl_grid: int8[ROWNO, COLNO] in corridors.py VTILE_* encoding.
        door_mask_grid:   optional int8[ROWNO, COLNO] vendor ``doormask`` grid
                          (same orientation as ``vendor_levl_grid``).  When
                          None, every DOOR keeps the step-1 CLOSED_DOOR default.

    Returns:
        int8[ROWNO, COLNO] internal TileType array.
    """
    grid = vendor_levl_grid.astype(jnp.int32)
    idx = jnp.clip(grid, 0, _VTYP_TO_TILE.shape[0] - 1)
    terrain = _VTYP_TO_TILE[idx].astype(jnp.int8)

    # --- Door-mask refinement (only when the doormask is threaded in). ---
    # Re-map DOOR cells from the CLOSED_DOOR default to the mask-appropriate
    # tile.  Pure geometry/lookup — no RNG, JIT-safe.
    #
    # RENDER-TABLE CAVEAT (S_ndoor): vendor renders a D_NODOOR / D_BROKEN
    # doorway as S_ndoor (cmap 12, '.').  The obs ``_TILE_TO_CMAP``
    # (nle_obs.py, owned elsewhere) has NO TileType that maps to S_ndoor, and
    # the obs ``_apply_wall_angle`` pass treats ONLY WALL / CLOSED_DOOR /
    # OPEN_DOOR as wall-continuations when deriving corner variants.  A
    # doorless doorway therefore cannot be both (a) rendered as S_ndoor and
    # (b) counted as the wall-segment that lets its neighbouring room corner
    # resolve to S_blcorn — those two facts both require an nle_obs change
    # (add a DOORWAY TileType -> S_ndoor AND include it in is_wallish).
    #
    # Within this file we keep the corner correct (the fully-achievable win):
    # a D_NODOOR / D_BROKEN / D_ISOPEN doorway maps to OPEN_DOOR — a *passable*
    # door tile that is wall-ish in _apply_wall_angle, so the adjacent corner
    # still resolves to S_blcorn.  Only D_CLOSED / D_LOCKED render as a shut
    # door ('+').  This is strictly closer to vendor than the old
    # unconditional CLOSED_DOOR (a doorless opening no longer shows '+'); the
    # exact S_ndoor glyph for the doorway cell awaits the nle_obs DOORWAY
    # TileType.  Vendor cite: vendor/nle/include/rm.h door masks; display.c
    # back_to_glyph (D_NODOOR -> S_ndoor).
    if door_mask_grid is not None:
        dmask = door_mask_grid.astype(jnp.int32)
        is_door_cell = grid == jnp.int32(_VTILE_DOOR)
        shut = (dmask & jnp.int32(_DMASK_CLOSED | _DMASK_LOCKED)) != jnp.int32(0)
        is_open = (dmask & jnp.int32(_DMASK_ISOPEN)) != jnp.int32(0)
        # Three-way (vendor display.c back_to_glyph door masks):
        #   D_CLOSED / D_LOCKED -> CLOSED_DOOR ('+')
        #   D_ISOPEN            -> OPEN_DOOR   ('|'/'-')
        #   D_NODOOR / D_BROKEN -> DOORWAY     (S_ndoor doorless gap, '.')
        # DOORWAY is passable + counts as wall-ish in _apply_wall_angle, so the
        # adjacent BL corner still resolves to S_blcorn.
        door_tile = jnp.where(
            shut,
            jnp.int8(_TILE_CLOSED_DOOR),
            jnp.where(is_open, jnp.int8(_TILE_OPEN_DOOR), jnp.int8(_TILE_DOORWAY)),
        )
        terrain = jnp.where(is_door_cell, door_tile, terrain)

    # Corner cells are pre-stamped with their proper vendor codes
    # (TLCORNER=3 / TRCORNER=4 / BLCORNER=5 / BRCORNER=6) in
    # ``stamp_rooms_into_typ`` (corridors.py) and the vault-stamp block in
    # ``generate_main_branch_l1`` below; ``_VTYP_TO_TILE`` maps each of
    # those vendor codes directly to ``_TILE_WALL``, so the step-1 lookup
    # already renders corners correctly.  No geometric corner-promotion
    # pass is needed (and the previous geometric pass mis-fired on interior
    # ROOM cells with perpendicular adjacent doors — see the docstring).
    return terrain


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


def _sort_rooms_by_lx(rooms, active, nroom=None):
    """Reorder rooms by ``lx`` ascending — vendor ``sort_rooms()``.

    Vendor mklev.c:707 qsorts ``rooms[0..nroom)`` using ``do_comp``
    (mklev.c:62-64), which compares the left edge ``lx`` only.  Every
    post-makerooms consumer (stair picks, makecorridors' join(a, a+1)
    walk, make_niches) then indexes the lx-sorted array, so the sort must
    run before any of those draws to keep the ISAAC64 stream byte-exact.

    CRITICAL: vendor sorts ONLY the first ``nroom`` entries (the OROOM
    range).  When a vault was created in makerooms, the vault slot sits
    at index ``nroom`` (just past the OROOM range) with a sentinel
    ``hx=-1``, and is NOT touched by the sort.  If we pass the vault
    slot through the sort key, the vault's real ``lx`` interleaves it
    among the OROOMs and every downstream room-index shifts — the
    seed=2 structural divergence cause.

    Inactive slots carry the lx == -1 sentinel from makerooms.  We key
    them — AND the vault slot when ``nroom`` is supplied — to a large
    value (COLNO+1 = 81 > any real lx) so they land after every active
    room, matching vendor's "sort only the first nroom entries; leave
    the tail untouched" behaviour.  Stable sort (argsort) handles equal-
    lx ties deterministically; distinct lx is the common case for non-
    overlapping rooms.

    Args:
        rooms: Room pytree.
        active: bool[MAX_ROOMS_PER_LEVEL] — slots with lx >= 0.
        nroom: int32 scalar or None — vendor's pre-vault nroom (the
            OROOM count from makerooms).  When supplied, only slots
            with index < nroom participate in the sort; slots at or
            beyond nroom (including the vault) sink to the tail.  When
            ``None`` (Threefry path or legacy callers), every active
            slot is sorted.

    Citation: vendor/nle/src/mklev.c:707 (sort_rooms),
              vendor/nle/src/mklev.c:46-66 (do_comp keys on lx),
              vendor/nle/src/mklev.c:98-107 (qsort by do_comp).
    """
    if nroom is not None:
        slot_idx = jnp.arange(rooms.x1.shape[0], dtype=jnp.int32)
        sort_eligible = active & (slot_idx < nroom.astype(jnp.int32))
    else:
        sort_eligible = active
    sort_key = jnp.where(sort_eligible, rooms.x1.astype(jnp.int32), jnp.int32(81))
    # Vendor sort_rooms() uses libSystem qsort, which is NOT stable on
    # tied keys.  Emulating with jnp.argsort(stable=True) preserves
    # creation order on ties — mismatching vendor and breaking seeds
    # whose room layout has tied lx values (e.g. seed=5 has two rooms
    # both at lx=68).  Use a host-side qsort callback so the tie order
    # matches vendor byte-for-byte.
    order = _argsort_macos_qsort(sort_key)
    sorted_rooms = jax.tree_util.tree_map(lambda f: f[order], rooms)
    sorted_active = active[order]
    return sorted_rooms, sorted_active


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

    # 1. Place rooms.
    #
    # Byte-parity path (vendor_rng supplied): vendor makelevel() calls
    # init_rect() then makerooms() (mklev.c:705-706).  The legacy
    # generate_rooms uses its own rejection-sampling draw order, which
    # diverges from vendor's rnd_rect()/create_room() stream at the very
    # first room draw (the makerooms while-loop rnd_rect = rn2(rect_cnt)).
    # Route through the vendor-exact Phase-2/3 makerooms() so the ISAAC64
    # byte stream matches vendor C from mklev's room generation onward.
    # Citation: vendor/nle/src/mklev.c:705-706 (init_rect + makerooms),
    #           vendor/nle/src/rect.c:28-35 (init_rect),
    #           vendor/nle/src/rect.c:88-92 (rnd_rect = rn2(rect_cnt)).
    if vendor_rng is not None:
        from Nethax.nethax.dungeon.rooms import makerooms
        from Nethax.nethax.dungeon.rect_pool import init_rect
        pool0 = init_rect()
        vendor_rng, _pool, rooms, active, _nroom, _tried_vault, _vault_success, _mk_vault_x, _mk_vault_y = makerooms(
            vendor_rng, pool0, depth=1,
        )

        # Save vault_success so we can emit do_vault block draws AFTER make_niches
        # but BEFORE place_branch (matching vendor mklev.c:738-762 → 800 order).
        # Vendor's do_vault block (mklev.c:738) is gated on ``vault_x != -1``
        # which is set inside ``create_vault()`` ONLY when create_room(vault)
        # returned success (mklev.c:233 ``if (create_vault())`` branch).  So
        # ``tried_vault`` alone is insufficient: a seed that *attempted* a
        # vault and FAILED create_room would burn ``tried_vault`` but NOT run
        # the do_vault block — vendor draws 0 do_vault bytes in that case.
        # Use ``vault_success`` (returned from makerooms, computed as
        # ``take_vault & cr_success`` in the body) to gate the 5 do_vault
        # draws + nroom bump.  Vendor's do_vault block calls
        # ``add_room(...VAULT)`` which BOTH fills a rooms[] slot AND
        # increments nroom (mklev.c:200 in add_room).  So when vault
        # succeeds, vendor's nroom is bumped by 1 at do_vault time; all
        # subsequent ``rn2(nroom)`` calls (place_branch, etc.) use the
        # bumped value.  Nethax's makerooms vault path does NOT increment
        # nroom (rooms.py:577 ``incremented = success & ~vault``), so we
        # manually bump it here before place_branch.  Vendor cite:
        # vendor/nle/src/mklev.c:200 (add_room nroom++);
        # vendor/nle/src/mklev.c:233-235 (create_vault success branch);
        # vendor/nle/src/mklev.c:738-762 (do_vault block calls add_room).
        _vault_created_in_makerooms = _vault_success
        _nroom_post_vault = jnp.where(
            _vault_created_in_makerooms, _nroom + jnp.int32(1), _nroom,
        )

        # sort_rooms() — vendor mklev.c:707.  Vendor qsorts the rooms[]
        # array by ``lx`` (do_comp at mklev.c:62-64 compares lx only) so
        # that all subsequent indexing — the stair-room picks
        # rooms[rn2(nroom)] (mklev.c:710,715), makecorridors' sequential
        # join(a, a+1) walk (mklev.c:325), and make_niches — operate on
        # the ascending-lx order.  makerooms produces rooms in placement
        # order, NOT lx order, so without this sort the down-stair somex
        # width (rn2(hx-lx+1)) is drawn from the wrong room and the
        # ISAAC64 stream diverges at the first post-makerooms draw.
        # Inactive slots carry the lx=-1 sentinel; we key them to a large
        # value so they sort AFTER every active room (vendor only sorts
        # the first ``nroom`` entries; the tail sentinels are untouched).
        # Citation: vendor/nle/src/mklev.c:707 (sort_rooms),
        #           vendor/nle/src/mklev.c:46-66 (do_comp, lx-only key),
        #           vendor/nle/src/mklev.c:98-107 (qsort by do_comp).
        # Pass _nroom (vendor's pre-vault OROOM count) so the vault slot
        # at index _nroom is excluded from the sort, matching vendor's
        # "sort only rooms[0..nroom)" semantics (mklev.c:707).  Without
        # this, the vault's real lx interleaves it among the OROOMs and
        # every downstream room-index shifts — the seed=2 layout
        # divergence cause.
        rooms, active = _sort_rooms_by_lx(rooms, active, nroom=_nroom)
    else:
        # Threefry layout path (non-parity): original rejection sampler.
        rooms, active, vendor_rng = generate_rooms(
            k_rooms, h, w, n_rooms=n_rooms, vendor_rng=vendor_rng,
        )

    # Vendor-faithful ``levl[][]`` grid (rooms+walls+corridors+doors) for
    # the post-fill ``mineralize`` STONE scan.  Populated only on the
    # byte-parity (vendor_rng) path; ``None`` in the Threefry layout path.
    vendor_levl_grid = None

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

        # Vendor's stair-pick / makecorridors / make_niches all use
        # ``nroom`` as it stood at the END of makerooms — which is the
        # OROOM count, EXCLUDING the vault.  Vendor's add_room(VAULT) in
        # the do_vault block (mklev.c:746) bumps nroom by +1, but that
        # runs AFTER the stair picks (mklev.c:710-727) and after
        # makecorridors+make_niches (mklev.c:734-736).  So pre-vault
        # nroom is what we want here.
        #
        # ``active.sum()`` (the old computation) INCLUDED the vault slot
        # when the sentinel hx=-1 survived the makerooms loop (which
        # happens when makerooms exits right after the vault attempt
        # with no subsequent OROOM overwriting the slot).  In that case
        # active.sum() = _nroom + 1, producing a +1 modulus mismatch on
        # every post-makerooms rn2(nroom) draw — the seed=2 structural
        # divergence cause.  Use the makerooms-returned ``_nroom``
        # directly: that's the OROOM-increment counter (rooms.py: vault
        # path explicitly does NOT increment nroom, see
        # ``incremented = success & ~vault``), so it always reflects the
        # vendor pre-vault nroom regardless of whether the vault slot
        # was overwritten or survived.
        # Vendor cite: vendor/nle/src/mklev.c:200 (add_room nroom++),
        #              vendor/nle/src/mklev.c:710 (stair-pick rn2(nroom),
        #              runs BEFORE do_vault at line 738).
        nroom_int = _nroom.astype(jnp.int32)

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
            stamp_rooms_into_typ as _vendor_stamp_rooms,
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
        # Stamp the (now lx-sorted) rooms' walls + floor into the level-gen
        # grid BEFORE makecorridors, exactly as vendor add_room
        # (mklev.c:160-182) populates levl[][] during makerooms.  Without
        # this, finddpos' okdoor check never finds a wall and falls through
        # to its (xl, yh) fallback, shifting every corridor door position
        # and diverging the dig_corridor rn2(dix-diy+1) bias stream.
        _lgs = _vendor_stamp_rooms(_lgs, _rooms_box)
        # Stamp the down-stair into _lgs.typ at (dn_sx, dn_sy).  Vendor
        # mkstairs (mklev.c:1566) writes ``levl[x][y].typ = STAIRS`` after
        # picking the stair coordinates and BEFORE makecorridors / make_niches
        # run.  Niche placement's !IS_FURNITURE check (vendor mklev.c:470)
        # uses this stamp to reject "back-onto stairs" placements; without
        # it, Nethax saw ROOM there and accepted niche attempts that vendor
        # rejected — diverging the ISAAC64 stream (e.g. seed=4 at draw 1691).
        # Vendor cite: vendor/nle/src/mklev.c:712,1566.
        from Nethax.nethax.dungeon.corridors import VTILE_STAIRS as _VTILE_STAIRS
        _lgs = _lgs.replace(
            typ=_lgs.typ.at[
                dn_sx.astype(jnp.int32),
                dn_sy.astype(jnp.int32),
            ].set(jnp.int8(_VTILE_STAIRS))
        )
        # mklev.c:734 — makecorridors(rooms, nroom).  ``depth=1`` on
        # Main Dlvl 1 gates maybe_sdoor: depth>2 is false so dodoor skips
        # the rn2(8) draw and forces DOOR on every corridor endpoint.
        # Vendor cite: vendor/nethack/src/mklev.c:1793-1795 maybe_sdoor.
        vendor_rng, _lgs = _vendor_makecorridors(
            vendor_rng, _lgs, _rooms_box, nroom_int, depth=1,
        )
        # mklev.c:735 — make_niches(rooms, nroom).  ``depth=1`` on
        # Main Dlvl 1 short-circuits both the ltptr (depth > 15) and
        # vamp (5 < depth < 25) gates so their rn2(6) draws are skipped,
        # matching vendor C control flow.
        vendor_rng, _lgs = _vendor_make_niches(
            vendor_rng, _lgs, _rooms_box, nroom_int,
            depth=1, noteleport=False,
        )
        # ``_lgs.typ`` is the vendor-faithful ``levl[][]`` grid (rooms +
        # walls + corridors + doors, in vendor tile-type encoding with
        # STONE==0).  Vendor's ``mineralize`` (mklev.c:948-987) scans THIS
        # grid for all-STONE 3x3 neighbourhoods, so we hand it forward to
        # the mineralize call instead of the sparse JAX ``terrain`` (whose
        # missing corridors/doors left ~4x too many all-STONE cells, over-
        # drawing the rn2(1000) gold/gem scan).  Transpose [COLNO, ROWNO]
        # → [ROWNO, COLNO] to match mineralize's row-major terrain layout.
        # Vendor cite: vendor/nle/src/mklev.c:948-961 (levl[][] STONE scan).
        vendor_levl_grid = jnp.transpose(_lgs.typ)  # [ROWNO, COLNO]
        # Per-cell vendor ``doormask`` (D_NODOOR / D_BROKEN / D_ISOPEN /
        # D_CLOSED / D_LOCKED) in the same [ROWNO, COLNO] orientation.  Threaded
        # into _vendor_grid_to_terrain so DOOR cells render by mask (a doorless
        # opening no longer shows as a spurious '+').  Vendor cite:
        # vendor/nle/src/mklev.c::dosdoor sets rm.doormask per door.
        vendor_door_mask_grid = jnp.transpose(_lgs.doormask)  # [ROWNO, COLNO]
        # NOTE: ``_lgs`` and ``_rooms_box`` are kept alive past this point so
        # the do_vault block's ``makevtele -> makeniche(TELEP_TRAP)`` cascade
        # (vendor mklev.c:752-753, 568-571) can re-enter ``_makeniche`` with
        # the current level-gen surface.  They are released after the do_vault
        # block below.

        # ------------------------------------------------------------------
        # Materialise the RENDERED terrain from the vendor-exact level grid.
        #
        # Until now ``terrain`` carried the LEGACY Threefry ``connect_rooms``
        # corridor layout (carved above at the ``connect_rooms`` call), which
        # diverges from vendor ``makecorridors``.  The vendor-faithful surface
        # is ``vendor_levl_grid`` (rooms + walls + corridors + doors, byte-exact
        # ISAAC64 RNG).  Replace the obs/gameplay terrain with the grid-derived
        # TileType array so the rendered dungeon (glyphs/chars/colors/
        # screen_descriptions) byte-matches NLE.  The down-stair and niche
        # features are stamped ON TOP of this terrain below / in the wrapper,
        # so this must run before those overlays.  Vendor cite: the rooms in
        # ``vendor_levl_grid`` are the makerooms output (byte-exact) and the
        # corridors are makecorridors (vendor/nle/src/mklev.c:734), so the
        # whole walkable surface is vendor-consistent.  Threefry (non-vendor)
        # path is untouched and keeps using ``connect_rooms``.
        terrain = _vendor_grid_to_terrain(
            vendor_levl_grid, door_mask_grid=vendor_door_mask_grid,
        )

        # ------------------------------------------------------------------
        # do_vault block — vendor mklev.c:738-762.
        #
        # Runs AFTER make_niches and BEFORE place_branch.  On Dlvl 1 when the
        # vault path was taken in makerooms (``tried_vault`` AND an active
        # slot carries the hx=-1 sentinel), vendor performs:
        #
        #   check_room(vault_x, w=1, vault_y, h=1, TRUE)  — sp_lev.c:1063
        #     Scans cells in the area [vault_x-xlim .. vault_x+1+xlim] x
        #     [vault_y-ylim .. vault_y+1+ylim] (xlim=ylim=XLIM/YLIM+1=5).
        #     For each non-stone cell encountered, draws rn2(3).  On Main
        #     Dlvl 1 with a typical vault placement chosen by makerooms'
        #     create_vault (which uses rnd_rect and avoids existing rooms),
        #     the vault area is fully stone and check_room consumes ZERO
        #     RNG draws, returning TRUE immediately.  (If makecorridors had
        #     cut through the vault footprint, check_room would draw and
        #     potentially fail, falling through to the retry path at line
        #     754 — which is a separate code path with its own draws.)
        #
        #   fill_room(&rooms[nroom-1], FALSE)        — sp_lev.c:2447-2452
        #     for each of 4 vault interior tiles:
        #         mkgold(rn1(abs(depth) * 100, 51), x, y)
        #         → rn1(100, 51) = 51 + rn2(100); amount>0 so mkgold's inner
        #            rnd(2)/rnd(3) block is SKIPPED.  Net: 4 rn2(100).
        #   mk_knox_portal(...)                       — mklev.c:1865-1899
        #     On Dlvl 1 source->dnum < n_dgns is TRUE (init_dungeons sets it
        #     to dnum during branch insertion), so the rn2(3) at mklev.c:1884
        #     is SHORT-CIRCUITED.  Net: 0 draws on Dlvl 1.
        #   if (!noteleport && !rn2(3)) makevtele();  — mklev.c:752
        #     noteleport=0 on Dlvl 1 (mklev.c:622); 1 rn2(3) is drawn.
        #
        # Net for Dlvl 1 vault path (check_room success on first try):
        # 4 rn2(100) + 1 rn2(3) = 5 draws.
        # ------------------------------------------------------------------
        from Nethax.nethax.vendor_rng import rn2_jax as _rn2_jax_do_vault

        # Stamp the vault footprint into vendor_levl_grid so that mineralize's
        # all-STONE 3x3 neighbourhood scan sees the placed vault cells as
        # non-stone (vendor cite: mklev.c:746-750 add_room+fill_room places
        # VAULT walls + ROOM interior into levl[][]).  Without this, the
        # mineralize STONE scan would draw extra rn2(1000) gold/gem rolls on
        # cells that vendor has already converted to non-stone, over-drawing
        # the ISAAC64 stream by ~76 bytes for a 2x2 vault footprint (vault
        # walls block ~38 candidate cells × 2 rn2(1000) per cell).
        #
        # Vendor's add_room writes:
        #   HWALL on (lx-1..hx+1) × (ly-1, hy+1)
        #   VWALL on (lx-1, hx+1) × (ly..hy)
        #   ROOM  on interior (lx..hx) × (ly..hy)
        # where for vault: w=h=1 so interior is 2x2 at (vault_x..vault_x+1)
        # × (vault_y..vault_y+1), and walls extend one cell outward.
        #
        # Vault placement coords come from makerooms' static-variable capture
        # ``_mk_vault_x/_mk_vault_y``.  Vendor's create_vault() writes the
        # placement coords into static ``vault_x/vault_y`` (mklev.c:233-234)
        # BEFORE the next OROOM iteration's create_room may overwrite the
        # ``rooms[nroom]`` slot (mklev.c:235 sets ``hx=-1`` but the lx/ly
        # remain readable).  Nethax's makerooms captures the same static-
        # equivalent values via the ``capture_vault`` carry in
        # rooms.py:873-875, surviving any subsequent OROOM overwrite at the
        # same slot index.  Gate stamping on ``_vault_created_in_makerooms``
        # (the actual success flag from create_room) rather than a sentinel
        # presence check, since the sentinel is lost on overwrite — seed=1
        # and seed=2 both hit this case and previously skipped vault
        # stamping, leaving mineralize to draw rn2(1000) on the 16 vault
        # footprint cells that vendor had already converted to non-stone.
        # Vendor cite: vendor/nle/src/mklev.c:232-235, :738-762.
        _vault_present = _vault_created_in_makerooms
        _vault_x = _mk_vault_x.astype(jnp.int32)
        _vault_y = _mk_vault_y.astype(jnp.int32)
        # Vault interior bounds (w=h=1 → interior is 2 cells wide/tall):
        #   lx = vault_x, hx = vault_x + 1
        #   ly = vault_y, hy = vault_y + 1
        _v_lx = _vault_x
        _v_hx = _vault_x + jnp.int32(1)
        _v_ly = _vault_y
        _v_hy = _vault_y + jnp.int32(1)

        # Stamp vault walls + interior into vendor_levl_grid (which is in
        # [ROWNO, COLNO] orientation, i.e. [y, x] with VTILE_STONE=0,
        # VTILE_HWALL=2, VTILE_VWALL=1, VTILE_ROOM=11).  Only stamp when
        # the vault was actually created.
        from Nethax.nethax.dungeon.corridors import (
            VTILE_HWALL as _V_H, VTILE_VWALL as _V_V, VTILE_ROOM as _V_R,
            VTILE_TLCORN as _V_TL, VTILE_TRCORN as _V_TR,
            VTILE_BLCORN as _V_BL, VTILE_BRCORN as _V_BR,
        )
        _vh, _vw = vendor_levl_grid.shape  # [ROWNO, COLNO]
        _y_axis = jnp.arange(_vh, dtype=jnp.int32)[:, None]  # rows
        _x_axis = jnp.arange(_vw, dtype=jnp.int32)[None, :]  # cols
        # Wall band: x ∈ [lx-1, hx+1], y ∈ [ly-1, hy+1].
        _wall_x = (_x_axis >= _v_lx - 1) & (_x_axis <= _v_hx + 1)
        _wall_y = (_y_axis >= _v_ly - 1) & (_y_axis <= _v_hy + 1)
        _interior = (
            (_x_axis >= _v_lx) & (_x_axis <= _v_hx)
            & (_y_axis >= _v_ly) & (_y_axis <= _v_hy)
        )
        # Top/bottom HWALL rows (within wall-x band).
        _hwall = _wall_x & ((_y_axis == _v_ly - 1) | (_y_axis == _v_hy + 1))
        # Left/right VWALL cols (only for y ∈ [ly, hy], NOT the corners).
        _vwall = (
            ((_x_axis == _v_lx - 1) | (_x_axis == _v_hx + 1))
            & (_y_axis >= _v_ly) & (_y_axis <= _v_hy)
        )
        # Per-corner masks so each of the 4 vault corners gets its proper
        # vendor TLCORNER/TRCORNER/BLCORNER/BRCORNER code, matching
        # do_room_or_subroom (mklev.c:175-179).  ``_VTYP_TO_TILE`` maps these
        # directly to WALL via the static lookup in ``_vendor_grid_to_terrain``.
        _tl = (_x_axis == _v_lx - 1) & (_y_axis == _v_ly - 1)
        _tr = (_x_axis == _v_hx + 1) & (_y_axis == _v_ly - 1)
        _bl = (_x_axis == _v_lx - 1) & (_y_axis == _v_hy + 1)
        _br = (_x_axis == _v_hx + 1) & (_y_axis == _v_hy + 1)
        _corner = _tl | _tr | _bl | _br
        _gate = _vault_present
        _new_grid = vendor_levl_grid
        _new_grid = jnp.where(_gate & _hwall, jnp.int8(_V_H), _new_grid)
        _new_grid = jnp.where(_gate & _vwall, jnp.int8(_V_V), _new_grid)
        _new_grid = jnp.where(_gate & _interior, jnp.int8(_V_R), _new_grid)
        _new_grid = jnp.where(_gate & _tl, jnp.int8(_V_TL), _new_grid)
        _new_grid = jnp.where(_gate & _tr, jnp.int8(_V_TR), _new_grid)
        _new_grid = jnp.where(_gate & _bl, jnp.int8(_V_BL), _new_grid)
        _new_grid = jnp.where(_gate & _br, jnp.int8(_V_BR), _new_grid)
        vendor_levl_grid = _new_grid

        # Also stamp the vault FLOOR into the rendered ``terrain`` array so
        # the obs/gameplay surface shows the 2x2 vault interior (vendor's
        # fill_room emits gold there; we don't materialise the gold but the
        # FLOOR tiles need to be present for any future hero teleport into
        # the vault to be playable).  ``terrain`` is in [ROWNO, COLNO] using
        # TileType.FLOOR=1.
        from Nethax.nethax.constants.tiles import TileType as _TT_v
        _FLOOR = jnp.int8(int(_TT_v.FLOOR))
        _WALL  = jnp.int8(int(_TT_v.WALL))
        terrain = jnp.where(_gate & _interior, _FLOOR, terrain)
        # Stamp walls so the rendered terrain shows the vault perimeter.
        # NOTE: do NOT overwrite if cell is already FLOOR/CORRIDOR — but
        # since the vault placement is gated on check_room (no overlap),
        # the wall band is on STONE, so a simple where(_gate & wall, WALL)
        # is safe.
        _wall_cells = (_hwall | _vwall | _corner)
        terrain = jnp.where(_gate & _wall_cells, _WALL, terrain)

        # Build a post-vault RoomsBox view that includes the vault slot at
        # index ``_nroom_post_vault - 1`` with rtype = VAULT.  Vendor's
        # add_room call at mklev.c:746-747 sets the slot's rtype to VAULT,
        # which makes makeniche's ``rtype != OROOM`` filter (mklev.c:495)
        # reject the vault slot during room picks — without this override,
        # the slot would still read rtype=ORDINARY (the default from
        # _invoke_create_room) and the SCORR branch would mis-target it.
        # Vault interior bounds were computed above (_v_lx/_v_ly/_v_hx/_v_hy).
        # The slot's stored hx carries the -1 sentinel; we replace it here
        # with the real vault hx so place_niche's strip span is well-formed.
        from Nethax.nethax.dungeon.rooms import RoomType as _RoomType_v
        _ROOM_TYPE_VAULT = jnp.int8(int(_RoomType_v.VAULT))
        # Compute the vault slot index (used to patch the RoomsBox view with
        # the real vault bounds + rtype=VAULT).  When the vault wasn't created
        # we target the last slot harmlessly (gated below by
        # ``_vault_created_in_makerooms``).
        _slot_idx = jnp.where(
            _vault_created_in_makerooms,
            _nroom_post_vault - jnp.int32(1),
            jnp.int32(MAX_ROOMS_PER_LEVEL - 1),
        )
        _rooms_box_postvault = _rooms_box.replace(
            lx=_rooms_box.lx.at[_slot_idx].set(_v_lx.astype(jnp.int16)),
            ly=_rooms_box.ly.at[_slot_idx].set(_v_ly.astype(jnp.int16)),
            hx=_rooms_box.hx.at[_slot_idx].set(_v_hx.astype(jnp.int16)),
            hy=_rooms_box.hy.at[_slot_idx].set(_v_hy.astype(jnp.int16)),
            rtype=lax.cond(
                _vault_created_in_makerooms,
                lambda a: a.at[_slot_idx].set(_ROOM_TYPE_VAULT),
                lambda a: a,
                _rooms_box.rtype,
            ),
            active=lax.cond(
                _vault_created_in_makerooms,
                lambda a: a.at[_slot_idx].set(jnp.bool_(True)),
                lambda a: a,
                _rooms_box.active,
            ),
        )

        # Stamp the vault footprint into _lgs.typ so place_niche / finddpos
        # see the vault walls as HWALL/VWALL (vendor add_room writes these in
        # levl[][] before makevtele runs — mklev.c:746,160-182).  Without
        # this, niche candidates around the vault would be evaluated against
        # stale STONE cells.  The grid is [COLNO, ROWNO] (gs.typ orientation).
        from Nethax.nethax.dungeon.corridors import (
            VTILE_HWALL as _VH_v, VTILE_VWALL as _VV_v,
            VTILE_ROOM as _VR_v,
            VTILE_TLCORN as _VTL_v, VTILE_TRCORN as _VTR_v,
            VTILE_BLCORN as _VBL_v, VTILE_BRCORN as _VBR_v,
        )
        _cols_axis = jnp.arange(_lgs.typ.shape[0], dtype=jnp.int32)[:, None]
        _rows_axis = jnp.arange(_lgs.typ.shape[1], dtype=jnp.int32)[None, :]
        _v_wall_x = (_cols_axis >= _v_lx - 1) & (_cols_axis <= _v_hx + 1)
        _v_wall_y = (_rows_axis >= _v_ly - 1) & (_rows_axis <= _v_hy + 1)
        _v_interior = (
            (_cols_axis >= _v_lx) & (_cols_axis <= _v_hx)
            & (_rows_axis >= _v_ly) & (_rows_axis <= _v_hy)
        )
        _v_hwall_lgs = _v_wall_x & ((_rows_axis == _v_ly - 1) | (_rows_axis == _v_hy + 1))
        _v_vwall_lgs = (
            ((_cols_axis == _v_lx - 1) | (_cols_axis == _v_hx + 1))
            & (_rows_axis >= _v_ly) & (_rows_axis <= _v_hy)
        )
        # Per-corner masks so each of the 4 vault corners gets its proper
        # vendor TLCORNER/TRCORNER/BLCORNER/BRCORNER code — matching the
        # add_room corner-stamp pattern (mklev.c:175-179) and the room
        # stamping convention in ``stamp_rooms_into_typ``.  This lets
        # ``_vendor_grid_to_terrain`` map vault corners directly to WALL
        # via the static ``_VTYP_TO_TILE`` lookup (no geometric promotion).
        _v_tl = (_cols_axis == _v_lx - 1) & (_rows_axis == _v_ly - 1)
        _v_tr = (_cols_axis == _v_hx + 1) & (_rows_axis == _v_ly - 1)
        _v_bl = (_cols_axis == _v_lx - 1) & (_rows_axis == _v_hy + 1)
        _v_br = (_cols_axis == _v_hx + 1) & (_rows_axis == _v_hy + 1)
        _stamp_gate = _vault_created_in_makerooms
        _new_lgs_typ = _lgs.typ
        _new_lgs_typ = jnp.where(_stamp_gate & _v_hwall_lgs, jnp.int8(_VH_v), _new_lgs_typ)
        _new_lgs_typ = jnp.where(_stamp_gate & _v_vwall_lgs, jnp.int8(_VV_v), _new_lgs_typ)
        _new_lgs_typ = jnp.where(_stamp_gate & _v_interior,  jnp.int8(_VR_v), _new_lgs_typ)
        _new_lgs_typ = jnp.where(_stamp_gate & _v_tl, jnp.int8(_VTL_v), _new_lgs_typ)
        _new_lgs_typ = jnp.where(_stamp_gate & _v_tr, jnp.int8(_VTR_v), _new_lgs_typ)
        _new_lgs_typ = jnp.where(_stamp_gate & _v_bl, jnp.int8(_VBL_v), _new_lgs_typ)
        _new_lgs_typ = jnp.where(_stamp_gate & _v_br, jnp.int8(_VBR_v), _new_lgs_typ)
        _lgs = _lgs.replace(typ=_new_lgs_typ)

        from Nethax.nethax.dungeon.corridors import _makeniche as _vendor_makeniche

        # TELEP_TRAP=15 (vendor/nle/include/trap.h:73).  Non-zero, so
        # _makeniche short-circuits past the rn2(4) SCORR/CORR gate and
        # always takes the SCORR branch (matching vendor mklev.c:504).
        _TELEP_TRAP = jnp.int32(15)

        def _do_vault_draws(carry):
            v, gs_ = carry
            v, _ = _rn2_jax_do_vault(v, jnp.int32(100))   # vault tile 0
            v, _ = _rn2_jax_do_vault(v, jnp.int32(100))   # vault tile 1
            v, _ = _rn2_jax_do_vault(v, jnp.int32(100))   # vault tile 2
            v, _ = _rn2_jax_do_vault(v, jnp.int32(100))   # vault tile 3
            # makevtele gate — vendor mklev.c:752 ``if (!level.flags.noteleport
            # && !rn2(3)) makevtele();``.  On Main Dlvl 1 noteleport=False is
            # static, so the gate reduces to !rn2(3) (gate fires when r==0).
            v, _r3 = _rn2_jax_do_vault(v, jnp.int32(3))
            gate_fires = _r3 == jnp.int32(0)

            # makevtele() -> makeniche(TELEP_TRAP).  Single makeniche call.
            # Vendor cite: vendor/nle/src/mklev.c:567-571.  ``_makeniche``
            # already loops up to vct=8 internally and short-circuits on the
            # first successful placement, so calling it once matches vendor.
            def _do_makevtele(rg):
                r_, g_ = rg
                r_, g_ = _vendor_makeniche(
                    r_, g_, _rooms_box_postvault, _nroom_post_vault,
                    _TELEP_TRAP, depth=1,
                )
                return r_, g_

            v, gs_ = lax.cond(
                gate_fires, _do_makevtele, lambda rg: rg, (v, gs_),
            )
            return v, gs_

        def _skip_do_vault_draws(carry):
            return carry

        vendor_rng, _lgs = lax.cond(
            _vault_created_in_makerooms,
            _do_vault_draws, _skip_do_vault_draws, (vendor_rng, _lgs),
        )
        del _lgs, _rooms_box, _rooms_box_postvault

        # ------------------------------------------------------------------
        # place_branch(branchp, 0, 0) — vendor mklev.c:800.
        #
        # makelevel() places the multi-dungeon branch stairway AFTER
        # make_niches() and the (Dlvl-1-noop) do_vault()/SHOPBASE blocks,
        # immediately before the fill_ordinary_rooms loop.  On Main Dlvl 1
        # ``branchp`` is the Mines branch (BR_STAIR) so place_branch falls
        # into ``find_branch_room`` (mklev.c:1169-1172) which consumes:
        #
        #   1. rn2(nroom)         branch-room pick      mklev.c:1118
        #                         (do/while retry excluding dnstairs_room /
        #                          upstairs_room / non-OROOM, tryct < 100)
        #   2. somex(croom) = rn1(hx-lx+1, lx)          mkroom.c:643
        #   3. somey(croom) = rn1(hy-ly+1, ly)          mkroom.c:650
        #                         (do/while retry while occupied or the cell
        #                          is not CORR/ROOM — on a freshly generated
        #                          level the first interior cell is unoccupied
        #                          ROOM so a single somexy fires)
        #
        # The do_vault()/SHOPBASE blocks (mklev.c:738-796) draw NOTHING on
        # seed-0 Main Dlvl 1 (vault_x == -1 so do_vault() is false, and
        # every ``u_depth > N`` shop/court gate is false at depth 1), so
        # emitting these draws here — before the wrapper's maybe_create_vault
        # / fill_ordinary_rooms — is byte-identical to vendor's post-vault
        # placement.  Confirmed against the instrumented NLE caller trace:
        # vendor draw 1187 = rn2(5) @ find_branch_room+ (mklev.c:1118),
        # draws 1188/1189 = rn2(3)/rn2(3) @ somex/somey (mkroom.c:643,650).
        #
        # ``upstairs_room`` is NULL on Dlvl 1 (mkstairs(up=1) is skipped by
        # the dlevel!=1 gate at mklev.c:720), and ``dnstairs_room`` is the
        # down-stair room ``down_idx``.  All Dlvl 1 rooms are OROOM, so the
        # only exclusion that fires is ``croom == dnstairs_room``.
        # Vendor cite: vendor/nle/src/mklev.c:800,1105-1132,1537-1563;
        #              vendor/nle/src/mkroom.c:640-651.
        from Nethax.nethax.vendor_rng import rn1_jax as _rn1_jax

        # Vendor's nroom at place_branch time includes the vault slot (added
        # in do_vault's add_room call, mklev.c:746).  Use _nroom_post_vault
        # so rn2(nroom) matches vendor when vault was created.  The vault
        # slot itself is non-OROOM, so the do/while at mklev.c:1117-1120
        # ``rtype != OROOM`` filter rejects it — model this by treating
        # idx == vault_slot_idx (== _nroom_post_vault - 1 when vault present)
        # as a collision that triggers redraw.
        _nroom_branch_int = _nroom_post_vault
        _vault_slot_idx = jnp.where(
            _vault_created_in_makerooms,
            _nroom_post_vault - jnp.int32(1),
            jnp.int32(-1),
        )

        # nroom > 2 path (mklev.c:1114): do/while redraw rn2(nroom) while the
        # pick collides with the down-stair room or vault slot.  tryct < 100
        # caps the loop.
        def _branch_pick_cond(carry):
            _vr, idx, tryct = carry
            collide = (
                ((idx == down_idx) & has_rooms)
                | (idx == _vault_slot_idx)
            )
            return collide & (tryct < jnp.int32(100))

        def _branch_pick_body(carry):
            vr, _idx, tryct = carry
            vr, idx = rn2_jax(vr, jnp.maximum(_nroom_branch_int, jnp.int32(1)))
            return vr, idx, tryct + jnp.int32(1)

        # First (unconditional) draw of the do/while, then redraw-on-collide.
        vendor_rng, _br_idx0 = rn2_jax(
            vendor_rng, jnp.maximum(_nroom_branch_int, jnp.int32(1)),
        )
        vendor_rng, branch_idx, _br_tryct = lax.while_loop(
            _branch_pick_cond,
            _branch_pick_body,
            (vendor_rng, _br_idx0, jnp.int32(1)),
        )

        # somex(branch_croom) / somey(branch_croom) — single iteration on a
        # fresh level (interior cell is unoccupied ROOM, mklev.c:1127-1129).
        br_lx = rooms.x1[branch_idx]
        br_ly = rooms.y1[branch_idx]
        br_hx = rooms.x2[branch_idx]
        br_hy = rooms.y2[branch_idx]
        br_w = jnp.maximum(
            (br_hx - br_lx + jnp.int16(1)).astype(jnp.int32), jnp.int32(1)
        )
        br_h = jnp.maximum(
            (br_hy - br_ly + jnp.int16(1)).astype(jnp.int32), jnp.int32(1)
        )
        vendor_rng, br_sx = _rn1_jax(vendor_rng, br_w, br_lx.astype(jnp.int32))
        vendor_rng, br_sy = _rn1_jax(vendor_rng, br_h, br_ly.astype(jnp.int32))

        # The down-stair (>) glyph stays at the down-stair (sx, sy) drawn
        # above (vendor mklev.c:710-712).
        vendor_down_r = jnp.clip(dn_sy, 1, h - 2).astype(jnp.int16)
        vendor_down_c = jnp.clip(dn_sx, 1, w - 2).astype(jnp.int16)

        # Hero spawn — vendor places the hero on Dlvl 1 via
        # ``u_on_upstairs() -> u_on_sstairs(0)`` (dungeon.c:1260-1265).
        # ``xupstair`` is 0 (mkstairs(up=1) skipped by the dlevel!=1 gate),
        # so it falls to ``u_on_sstairs(0)`` which spawns the hero on the
        # branch staircase ``(sstairs.sx, sstairs.sy)`` placed by
        # ``place_branch() -> find_branch_room()`` (mklev.c:800,1105-1131).
        # That is the (br_sx, br_sy) cell drawn just above — NOT the
        # down-stair.  Placing the hero here makes blstats[0,1] (player
        # x,y) byte-match NLE for seed=0.  Cite: vendor/nle/src/dungeon.c:
        # 1249-1265 (u_on_sstairs/u_on_upstairs), mklev.c:1190-1191
        # (sstairs.sx/sy = branch cell).
        vendor_hero_r = jnp.clip(br_sy, 1, h - 2).astype(jnp.int16)
        vendor_hero_c = jnp.clip(br_sx, 1, w - 2).astype(jnp.int16)
    else:
        # Threefry path: no vendor stair-pick draws.  Fall through to
        # the centre-of-room defaults below.
        vendor_hero_r = None  # type: ignore[assignment]
        vendor_hero_c = None  # type: ignore[assignment]
        vendor_down_r = None  # type: ignore[assignment]
        vendor_down_c = None  # type: ignore[assignment]

    # 6. Hero spawn / up-stair.
    #    Vendor path (Dlvl 1): the hero spawns on the branch staircase.
    #    `mkstairs(up=1)` is correctly skipped on Dlvl 1 (mklev.c:720), but
    #    vendor's `place_branch()` (mklev.c:1190-1198) DOES stamp a STAIRS
    #    tile at the branch cell: `levl[x][y].typ = STAIRS; ladder = LA_UP`.
    #    `back_to_glyph` (display.c:1753-1755) maps STAIRS+LA_UP → S_upstair
    #    (cmap 23, glyph 2382, char '<').  We must mirror that here so the
    #    obs reveals '<' under the hero after they step off the cell.
    #    Threefry path: place an up-stair in the centre of room[0] for
    #    playability (no vendor byte-parity constraint).
    if vendor_hero_r is not None:
        up_r = vendor_hero_r
        up_c = vendor_hero_c
        terrain = terrain.at[up_r, up_c].set(jnp.int8(_TILE_STAIRCASE_UP))
    else:
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

    # Vendor nroom at fill_ordinary_rooms time includes the vault slot (added
    # in do_vault's add_room call).  Expose it so the wrapper can pass it to
    # fill_ordinary_rooms's ``nroom`` argument (used by the box-gate modulus
    # rn2(nroom*5/2) at mklev.c:853).  Defaults to ``active.sum()`` when no
    # vault was created — semantically equivalent to vendor nroom.
    if vendor_rng is not None:
        try:
            _nroom_for_fill = _nroom_post_vault
        except NameError:
            _nroom_for_fill = active.sum().astype(jnp.int32)
    else:
        _nroom_for_fill = active.sum().astype(jnp.int32)

    return (
        terrain, rooms, active, up_stair_pos, down_stair_pos, vendor_rng,
        vendor_levl_grid, _nroom_for_fill,
    )


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
    state=None,
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

        When ``state`` is supplied alongside ``vendor_rng``, the return
        is extended with a trailing ``state_out`` slot carrying the
        monster_ai writes produced by the in-loop per-OROOM sleeping-
        monster spawn (vendor/nle/src/mklev.c:813-817).
    """
    # Import here to avoid circular dependency at module load time.
    from Nethax.nethax.dungeon.rooms import (
        fill_ordinary_rooms,
        maybe_create_vault,
        _place_niches,
    )
    from Nethax.nethax.dungeon.mineralize import mineralize as _mineralize

    k_level, k_fill, k_vault, k_niche, k_monster = jax.random.split(rng, 5)

    # ------------------------------------------------------------------
    # Vendor call order — vendor/nle/src/mklev.c::makelevel (652-886) and
    # ::mklev (990-1036), traced for seed=0 Main Dlvl 1:
    #
    #   makelevel():
    #     clear_level_structures()                 [665]   no RNG
    #     rn2(5) Medusa gate                       [693]   *
    #     makerooms()                              [706]   *
    #     sort_rooms()                             [707]   no RNG
    #     rn2(nroom) + somex/y down-stair          [710-712] *
    #     rn2(nroom-1) up-stair-room               [715]   *  (somex/y
    #                                                          gated by
    #                                                          dlevel!=1)
    #     makecorridors()                          [734]   *
    #     make_niches()                            [735]   *
    #     do_vault() + vault fill + makevtele      [738-762] *
    #     SHOPBASE block                           [764-796] no RNG @ Dlvl 1
    #     place_branch(NULL, 0, 0)                 [800]   no RNG @ Dlvl 1
    #     fill_ordinary_rooms loop                 [803-885] *
    #   (makelevel returns)
    #     bound_digging()                          [1005]  no RNG
    #     mineralize(-1, -1, -1, -1, FALSE)        [1006]  *
    #
    # Items marked * consume ISAAC64 draws.  Steps already folded into
    # generate_main_branch_l1: Medusa gate, makerooms, stair picks,
    # makecorridors, make_niches.  This wrapper layers on, in vendor
    # order: do_vault -> fill_ordinary_rooms -> mineralize.
    # ------------------------------------------------------------------

    (
        terrain, rooms, active, up_pos, dn_pos, vendor_rng, vendor_levl_grid,
        _nroom_for_fill,
    ) = generate_main_branch_l1(
        k_level, static_params, n_rooms=n_rooms, vendor_rng=vendor_rng,
    )

    # vendor/nle/src/mklev.c:738-762 — do_vault() block runs BEFORE
    # fill_ordinary_rooms.  Thread vendor_rng so the rn2(2) vault gate
    # (mklev.c:230) and rn2(3) makevtele gate (mklev.c:752) come from
    # the ISAAC64 stream under NLE_BYTEPARITY in the correct slot.
    terrain, features, traps, vendor_rng = maybe_create_vault(
        k_vault, rooms, active, terrain, features, traps, flat_lv=flat_lv,
        vendor_rng=vendor_rng,
    )

    # vendor/nle/src/mklev.c:803-885 — fill_ordinary_rooms loop (per-OROOM
    # feature/trap rolls).  Runs AFTER the vault block in vendor C.
    # Thread vendor_rng so per-room rn2/somexy draws come from the
    # ISAAC64 stream under NLE_BYTEPARITY (byte-exact with vendor C).
    #
    # Pass ``nroom=_nroom_for_fill`` so the box-gate ``rn2(nroom*5/2)`` at
    # mklev.c:853 uses the vault-bumped count — vendor's nroom INCLUDES
    # the vault slot at fill_ordinary_rooms time.
    #
    # When ``state`` is supplied (NLE_BYTEPARITY reset path), the
    # per-OROOM sleeping-monster spawn (vendor mklev.c:813-817) runs as
    # step 1 of each room iteration INSIDE :func:`fill_ordinary_rooms`,
    # interleaved with that room's feature/trap fills.  The returned
    # ``state_out`` carries the monster_ai writes; the separate
    # post-fill ``populate_level_with_monsters`` call is no longer
    # needed.  When ``state`` is ``None``, the legacy 4-tuple is
    # returned and monster spawning is deferred to the caller.
    if state is not None and vendor_rng is not None:
        terrain, features, traps, vendor_rng, state = fill_ordinary_rooms(
            k_fill, rooms, active, terrain, features, traps,
            flat_lv=flat_lv, depth=depth, player_align=player_align,
            vendor_rng=vendor_rng, nroom=_nroom_for_fill,
            state=state, monster_rng=k_monster,
            vendor_levl_grid=vendor_levl_grid,
        )
    else:
        terrain, features, traps, vendor_rng = fill_ordinary_rooms(
            k_fill, rooms, active, terrain, features, traps,
            flat_lv=flat_lv, depth=depth, player_align=player_align,
            vendor_rng=vendor_rng, nroom=_nroom_for_fill,
            vendor_levl_grid=vendor_levl_grid,
        )

    # vendor/nle/src/mklev.c::mineralize line 1006 — called from mklev()
    # AFTER makelevel() returns, i.e. after fill_ordinary_rooms.  Thread
    # vendor_rng so the ISAAC64 draw sequence matches vendor C byte-for-byte.
    if vendor_rng is not None:
        # Vendor's mineralize scans the real ``levl[][]`` (rooms + walls +
        # corridors + doors) for all-STONE 3x3 neighbourhoods.  Use the
        # vendor-faithful ``vendor_levl_grid`` built in
        # generate_main_branch_l1 (via stamp_rooms_into_typ + makecorridors)
        # rather than the sparse JAX ``terrain`` — the latter omits the
        # vendor corridor/door network, leaving ~4x too many all-STONE
        # cells and over-drawing the rn2(1000) scan.  ``mineralize`` treats
        # value 0 as STONE, which is vendor STONE==0 in this grid.
        # Vendor cite: vendor/nle/src/mklev.c:948-987.
        _mineralize_grid = (
            vendor_levl_grid if vendor_levl_grid is not None else terrain
        )
        # Phase 2 of the mineralize port: emit gold + gem placements into
        # ``state.ground_items`` so downstream consumers (e.g. dog_goal's
        # SQSRCHRADIUS fobj scan — pet_dog_move._emit_dog_goal_fobj_scan,
        # vendor/nle/src/dogmove.c:502-553) see the same on-floor object
        # set as vendor.  Without this the orange-glass gem at (col 67,
        # row 12) on seed-0 Dlvl 1 is missing from ground_items, the dog
        # scan draws one fewer rn2(100), and the ISAAC stream misaligns
        # from step 2.  Cite: .test_runs/room_12_67_audit.md.
        _br = int(flat_lv) // MAX_LEVELS_PER_BRANCH
        _lv = int(flat_lv) %  MAX_LEVELS_PER_BRANCH
        if state is not None:
            _gcat = state.ground_items.category
            _gtyp = state.ground_items.type_id
            _gqty = state.ground_items.quantity
            (
                _mineralize_grid, vendor_rng, _gcat, _gtyp, _gqty,
            ) = _mineralize(
                _mineralize_grid, vendor_rng, depth=depth, dunlev=depth,
                gi_category=_gcat,
                gi_type_id=_gtyp,
                gi_quantity=_gqty,
                branch_idx=_br,
                level_idx=_lv,
            )
            state = state.replace(
                ground_items=state.ground_items.replace(
                    category=_gcat, type_id=_gtyp, quantity=_gqty,
                ),
            )
        else:
            _mineralize_grid, vendor_rng = _mineralize(
                _mineralize_grid, vendor_rng, depth=depth, dunlev=depth,
            )

    # Threefry-only post-pass (NOT a vendor call): stamp fountain / sink /
    # grave / throne tiles onto random FLOOR cells.  Vendor's make_niches
    # already ran inside generate_main_branch_l1 against the corridors.py
    # LevelGenState surface; this is the JAX-side terrain materialisation
    # so niche features are observable.  Runs last so it does not disturb
    # the vendor_rng (ISAAC64) byte-parity stream.
    #
    # In NLE_BYTEPARITY mode (vendor_rng is not None), vendor's make_niches +
    # do_vault + fill_ordinary_rooms have ALREADY stamped every niche feature
    # vendor itself would place onto vendor_levl_grid, and that grid is the
    # source of truth for ``terrain`` (via ``_vendor_grid_to_terrain``).
    # Running this extra Threefry-driven stamp on top of that adds a feature
    # vendor does not have — observed on seed=4 as a spurious throne at
    # (row 8, col 11).  Skip the post-pass when threading the vendor stream
    # so the rendered terrain matches NLE exactly.
    if vendor_rng is None:
        terrain = _place_niches(terrain, rooms, active, k_niche, n=2)

    # Hero placement — vendor/nle/src/allmain.c:628 ``u_on_upstairs()`` runs
    # immediately after ``mklev()`` returns (after mineralize) and **consumes
    # ZERO ISAAC64 draws**: the NLE reset stream for seed=0 rog-hum-cha goes
    # directly from the last mineralize ``rn2(1000)`` (draw 1780) to
    # ``makedog()->pet_type()``'s ``rn2(2)`` (draw 1781) with no intervening
    # placement draw.  On Main Dlvl 1 there is no up-stair (mkstairs(up=1) is
    # skipped by the ``dlevel!=1`` gate, mklev.c:720), so ``u_on_upstairs() ->
    # u_on_sstairs(0)`` (dungeon.c:1260-1265) spawns the hero on the branch
    # staircase ``(sstairs.sx, sstairs.sy)`` that ``place_branch() ->
    # find_branch_room()`` (mklev.c:800,1105-1131) placed earlier in the
    # stream (draws 1187-1189).  ``up_pos`` already carries that branch cell
    # (generate_main_branch_l1 set up_r/up_c = vendor_hero_r/c), so no
    # override is needed.  The previous code overrode ``up_pos = dn_pos``,
    # spawning the hero on the down-stair (col 40) instead of the branch
    # staircase (col 71) — a divergence in blstats player_x.
    # Cite: vendor/nle/src/dungeon.c:1249-1265, mklev.c:1190-1191.

    if state is not None:
        return terrain, rooms, active, up_pos, dn_pos, features, traps, vendor_rng, state
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
# place_level recursion (faithful port of vendor dungeon.c::place_level)
# ---------------------------------------------------------------------------

def _place_dungeon_levels(vendor_rng, num_dunlevs, protos):
    """Faithful Python port of ``vendor/nle/src/dungeon.c::place_level``.

    Replays ``rn2(npossible)`` draws with backtracking semantics identical to
    the vendor C code (lines 637-679).  Backtracking matters because for some
    seeds the natural pick collides with a CHAINLEVEL constraint that requires
    a sibling proto's slot — vendor falls back and re-picks until placement
    succeeds.

    Args:
        vendor_rng: Isaac64State.
        num_dunlevs: int — ``dungeons[dgn].num_dunlevs`` for this dungeon
            (sets the legal range [1, num_dunlevs] for slots).
        protos: list of dicts with keys::
            name (str, debug only)
            base (int)          — vendor lev.base (negative = from end)
            rand (int)          — vendor lev.rand (-1 = to end; 0 = single)
            chain_idx (int|None)— index into ``protos`` of a CHAINLEVEL parent
            created (bool)      — False if init_level gate dropped this proto
                                  (then no place_level draw fires; slot=None)

    Returns:
        ``(new_vendor_rng, slots)`` where ``slots[i]`` is the picked slot for
        ``protos[i]`` or ``None`` if not created.
    """
    from Nethax.nethax.vendor_rng import rn2 as _rn2

    n = len(protos)
    slots: list[int | None] = [None] * n

    def _level_range(idx):
        """Port of vendor ``level_range`` (dungeon.c:350-382).

        Returns ``(start, count)`` for proto ``idx`` based on currently
        committed slots.
        """
        p = protos[idx]
        base = p["base"]
        randc = p["rand"]
        chain = p["chain_idx"]
        if chain is not None:
            base = base + slots[chain]
        else:
            if base < 0:
                base = num_dunlevs + base + 1
        # base must be in [1, num_dunlevs]
        if randc == -1:
            count = num_dunlevs - base + 1
        elif randc:
            count = num_dunlevs - base + 1 if (base + randc - 1) > num_dunlevs else randc
        else:
            count = 1
        return base, count

    def _possible_places(idx):
        """Port of vendor ``possible_places`` (dungeon.c:573-602).

        Returns boolean map array (length num_dunlevs+1, indices 1..num_dunlevs
        valid) of slots available for proto ``idx``.
        """
        m = [False] * (num_dunlevs + 2)
        start, count = _level_range(idx)
        for s in range(start, start + count):
            if 1 <= s <= num_dunlevs:
                m[s] = True
        # Mark off slots taken by prior placements in this dungeon
        # (vendor: ``for (i = pd->start; i < idx; i++)``).
        for j in range(idx):
            if slots[j] is not None and m[slots[j]]:
                m[slots[j]] = False
        return m

    def _pick_nth(map_arr, nth):
        """Port of vendor ``pick_level`` (dungeon.c:606-616).

        Return the ``nth`` (0-based) TRUE entry in ``map_arr``.
        """
        for i in range(1, num_dunlevs + 1):
            if map_arr[i]:
                if nth == 0:
                    return i
                nth -= 1
        # vendor panics here; we return 0 to match (should never hit in practice).
        return 0

    def _place(idx, vrng):
        """Recursive port of ``place_level`` (dungeon.c:637-679).

        Returns ``(new_vrng, success)``.
        """
        if idx == n:
            return vrng, True
        if not protos[idx]["created"]:
            return _place(idx + 1, vrng)
        m = _possible_places(idx)
        npossible = sum(1 for s in range(1, num_dunlevs + 1) if m[s])
        while npossible > 0:
            vrng, pick = _rn2(vrng, npossible)
            chosen = _pick_nth(m, int(pick))
            slots[idx] = chosen
            vrng, ok = _place(idx + 1, vrng)
            if ok:
                return vrng, True
            # this choice didn't work — drop it and retry
            m[chosen] = False
            slots[idx] = None
            npossible -= 1
        return vrng, False

    vendor_rng, ok = _place(0, vendor_rng)
    # Vendor panics on failure; if we hit it the data is malformed.
    return vendor_rng, slots


# ---------------------------------------------------------------------------
# JAX-traceable place_level (lax.while_loop port of the recursive backtrack)
# ---------------------------------------------------------------------------

# Static upper bounds.  Largest dungeon in dungeon.def is Gehennom (11 protos);
# largest num_dunlevs is DoD lev.rand=5 base=25 -> max 30.  Sized with headroom.
_PL_MAX_PROTOS = 11
_PL_MAX_SLOTS = 32        # slot indices 0..31; valid range 1..num_dunlevs (<=30)


def _place_dungeon_levels_jax(vendor_rng, num_dunlevs, protos, created_dyn=None):
    """JAX-traceable port of ``_place_dungeon_levels`` / vendor ``place_level``.

    Bit-exact draw-stream equivalent to the host recursive version: emits the
    SAME sequence of ``rn2(npossible)`` draws including backtracking re-picks.
    Uses ``lax.while_loop`` over an explicit ``(idx, slots, mask, phase)``
    state machine so it is JIT- and vmap-safe.

    Args:
        vendor_rng: Isaac64State (traced).
        num_dunlevs: int32 scalar (traced or static) — the dungeon depth cap.
        protos: list of dicts with keys ``base``, ``rand``, ``chain_idx``,
            ``created`` (all Python-static).  ``len(protos) <= _PL_MAX_PROTOS``.
        created_dyn: optional traced bool array of shape ``(n_protos,)``
            overriding the per-proto ``created`` field.  Used when ``created``
            depends on a traced RNG gate (e.g. ``bigrm_placed = rn2(100) < 40``).
            If ``None``, the static ``proto["created"]`` value is used.

    Returns:
        ``(new_vendor_rng, slots)`` where ``slots`` is an int32 array of
        shape ``(_PL_MAX_PROTOS,)`` (valid entries at indices ``[0, len(protos))``;
        unused entries are 0).
    """
    n_protos = len(protos)
    assert n_protos <= _PL_MAX_PROTOS, f"need _PL_MAX_PROTOS >= {n_protos}"

    # ---- Static proto tables (Python ints; baked into the trace) --------
    bases = jnp.asarray(
        [p["base"] for p in protos] + [0] * (_PL_MAX_PROTOS - n_protos),
        dtype=jnp.int32,
    )
    rands = jnp.asarray(
        [p["rand"] for p in protos] + [0] * (_PL_MAX_PROTOS - n_protos),
        dtype=jnp.int32,
    )
    # chain_idx: use -1 sentinel for "no chain"
    chain_idx = jnp.asarray(
        [(-1 if p["chain_idx"] is None else p["chain_idx"]) for p in protos]
        + [-1] * (_PL_MAX_PROTOS - n_protos),
        dtype=jnp.int32,
    )
    if created_dyn is None:
        created = jnp.asarray(
            [bool(p["created"]) for p in protos] + [False] * (_PL_MAX_PROTOS - n_protos),
            dtype=jnp.bool_,
        )
    else:
        # Pad created_dyn (shape n_protos) to _PL_MAX_PROTOS with False.
        pad = _PL_MAX_PROTOS - n_protos
        created = jnp.concatenate(
            [created_dyn.astype(jnp.bool_),
             jnp.zeros((pad,), dtype=jnp.bool_)]
        )

    nd = jnp.asarray(num_dunlevs, dtype=jnp.int32)
    slot_ids = jnp.arange(_PL_MAX_SLOTS, dtype=jnp.int32)  # 0..31

    def _level_range_jax(idx, slots):
        """Port of vendor ``level_range`` (dungeon.c:350-382) — JAX form."""
        base = bases[idx]
        randc = rands[idx]
        ch = chain_idx[idx]
        # base adjustment: chain → base += slots[chain]; else negatives count from end
        base_chain = base + jnp.where(ch >= 0, slots[jnp.maximum(ch, 0)], 0)
        base_nochain = jnp.where(base < 0, nd + base + jnp.int32(1), base)
        base_eff = jnp.where(ch >= 0, base_chain, base_nochain)
        # count:
        #   randc == -1 → count = nd - base + 1
        #   randc  >  0 → if base+randc-1 > nd: nd-base+1 else randc
        #   randc == 0 → count = 1
        count_neg = nd - base_eff + jnp.int32(1)
        count_pos = jnp.where(
            (base_eff + randc - jnp.int32(1)) > nd,
            nd - base_eff + jnp.int32(1),
            randc,
        )
        count = jnp.where(
            randc == jnp.int32(-1),
            count_neg,
            jnp.where(randc > jnp.int32(0), count_pos, jnp.int32(1)),
        )
        return base_eff, count

    def _possible_places_jax(idx, slots):
        """Port of vendor ``possible_places`` (dungeon.c:573-602) — JAX form.

        Returns bool array of shape (_PL_MAX_SLOTS,) where ``mask[s]`` is True
        iff slot ``s`` is in [base, base+count) clipped to [1, num_dunlevs]
        AND not already taken by a prior placement ``slots[j]`` for ``j < idx``.
        """
        base_eff, count = _level_range_jax(idx, slots)
        in_range = (slot_ids >= base_eff) & (slot_ids < (base_eff + count))
        in_dunlev = (slot_ids >= jnp.int32(1)) & (slot_ids <= nd)
        m = in_range & in_dunlev

        # Subtract slots[j] for j < idx (vendor: pd->start..idx-1).
        # Sentinel: slots[j] == 0 means "not yet placed" — slot 0 isn't valid
        # anyway, so blanking it is a no-op.
        proto_idx = jnp.arange(_PL_MAX_PROTOS, dtype=jnp.int32)
        # Compare slots[j] (broadcast over slot_ids) to slot s, and mask only
        # j < idx.  Result: per-slot, is it claimed by some earlier proto?
        # slots[j] is shape (P,); slot_ids is shape (S,).
        # taken[s] = any_j ( j<idx AND slots[j]==s )
        active = (proto_idx < idx)[:, None]                       # (P, 1)
        eq = (slots[:, None] == slot_ids[None, :])                # (P, S)
        taken = jnp.any(active & eq, axis=0)                       # (S,)
        return m & ~taken

    def _pick_nth_jax(m, nth):
        """Vendor ``pick_level`` (dungeon.c:606-616): nth-True (0-based) in m.

        Returns the slot index s (1..num_dunlevs) corresponding to the nth
        True entry.  Uses cumsum: smallest s with cumsum(m)[s] == nth+1.
        """
        cum = jnp.cumsum(m.astype(jnp.int32))                       # (_PL_MAX_SLOTS,)
        # argmax over (cum > nth) returns the first index where this is True
        return jnp.argmax(cum > nth).astype(jnp.int32)

    # ---- Iterative backtracking state machine ----------------------------
    # Carry: (rng, idx, slots, mask, fresh)
    #   idx  : current proto being placed (int32, -1 = backtrack-failed)
    #   slots: int32[_PL_MAX_PROTOS] — placed slot (0 = not placed)
    #   mask : bool[_PL_MAX_PROTOS, _PL_MAX_SLOTS] — per-frame remaining
    #          eligible slots; on fresh entry filled from possible_places,
    #          on backtrack the failed pick is already cleared.
    #   fresh: bool[_PL_MAX_PROTOS] — True if frame needs possible_places init
    DONE_IDX = jnp.int32(n_protos)

    init_slots = jnp.zeros((_PL_MAX_PROTOS,), dtype=jnp.int32)
    init_mask = jnp.zeros((_PL_MAX_PROTOS, _PL_MAX_SLOTS), dtype=jnp.bool_)
    init_fresh = jnp.ones((_PL_MAX_PROTOS,), dtype=jnp.bool_)
    init_carry = (vendor_rng, jnp.int32(0), init_slots, init_mask, init_fresh)

    def cond(carry):
        _rng, idx, _slots, _mask, _fresh = carry
        # Loop while we haven't finished all protos AND haven't backtracked
        # past the root (idx >= 0 since vendor panics on root failure).
        return (idx < DONE_IDX) & (idx >= jnp.int32(0))

    def body(carry):
        rng, idx, slots, mask, fresh = carry

        # ---- Branch 1: proto not created → skip frame, advance ---------
        # Equivalent to ``if not protos[idx]["created"]: return _place(idx+1, vrng)``
        def skip_uncreated(_args):
            r, i, s, mk, fr = _args
            return r, i + jnp.int32(1), s, mk, fr

        # ---- Branch 2: proto created → maybe init mask, then try pick --
        def place_created(_args):
            r, i, s, mk, fr = _args
            # If this frame is fresh, recompute possible_places(i) from current slots.
            need_init = fr[i]
            new_row = _possible_places_jax(i, s)
            row = jnp.where(need_init, new_row, mk[i])
            mk_init = mk.at[i].set(row)
            fr_init = fr.at[i].set(jnp.bool_(False))

            npossible = jnp.sum(row.astype(jnp.int32))

            def do_backtrack(_a):
                rr, ii, ss, mmk, ffr = _a
                # No options remain at this frame → drop choice, return to parent.
                # Reset frame i so re-entry recomputes possible_places.
                ffr_new = ffr.at[ii].set(jnp.bool_(True))
                # Find prior CREATED frame to backtrack into.  Skip uncreated
                # frames (they didn't consume any draw).
                # Iterate from ii-1 downwards, decrement until either we hit
                # a created frame or fall below 0.
                def bt_cond(b):
                    j, _ = b
                    # stop when j<0 OR created[j] is True
                    return (j >= jnp.int32(0)) & (~created[jnp.maximum(j, 0)])

                def bt_body(b):
                    j, _flag = b
                    return j - jnp.int32(1), _flag

                # Start at ii-1
                j_final, _ = lax.while_loop(bt_cond, bt_body,
                                             (ii - jnp.int32(1), jnp.bool_(False)))
                # At j_final: either <0 (root failure) or points to a CREATED frame.
                # We need to remove the failed pick from that frame's mask AND
                # clear its slot.  If <0 the loop terminates next iteration.
                safe_j = jnp.maximum(j_final, jnp.int32(0))
                chosen_at_j = ss[safe_j]
                # mmk[safe_j, chosen_at_j] = False (only if j_final >= 0)
                cur_row = mmk[safe_j]
                new_row2 = jnp.where(
                    j_final >= jnp.int32(0),
                    cur_row.at[chosen_at_j].set(jnp.bool_(False)),
                    cur_row,
                )
                mmk2 = mmk.at[safe_j].set(new_row2)
                ss2 = jnp.where(
                    j_final >= jnp.int32(0),
                    ss.at[safe_j].set(jnp.int32(0)),
                    ss,
                )
                return rr, j_final, ss2, mmk2, ffr_new

            def do_pick(_a):
                rr, ii, ss, mmk, ffr = _a
                # rn2(npossible) — JAX-traceable draw on traced npossible.
                from Nethax.nethax.vendor_rng import rn2_jax as _rn2_jax
                new_rng, pick = _rn2_jax(rr, npossible)
                chosen = _pick_nth_jax(mmk[ii], pick)
                ss_new = ss.at[ii].set(chosen)
                # Clear chosen from this frame's mask so a later backtrack
                # into this frame retries a different slot (vendor semantics:
                # ``m_list[i] = m_list[--m_count]`` permanently removes).
                row_new = mmk[ii].at[chosen].set(jnp.bool_(False))
                mmk_new = mmk.at[ii].set(row_new)
                # Mark next frame as fresh so it recomputes possible_places.
                next_i = ii + jnp.int32(1)
                # Only mark fresh if next_i is within bounds (no-op otherwise).
                safe_next = jnp.minimum(next_i, jnp.int32(_PL_MAX_PROTOS - 1))
                ffr_new = ffr.at[safe_next].set(jnp.bool_(True))
                return new_rng, next_i, ss_new, mmk_new, ffr_new

            return lax.cond(
                npossible == jnp.int32(0),
                do_backtrack,
                do_pick,
                (r, i, s, mk_init, fr_init),
            )

        # Dispatch on protos[idx].created.  ``idx`` is traced; ``created`` is a
        # static array, so we index it with idx.
        is_created = created[jnp.minimum(jnp.maximum(idx, 0), _PL_MAX_PROTOS - 1)]
        return lax.cond(is_created, place_created, skip_uncreated,
                        (rng, idx, slots, mask, fresh))

    final = lax.while_loop(cond, body, init_carry)
    final_rng, _final_idx, final_slots, _final_mask, _final_fresh = final
    return final_rng, final_slots

def consume_init_dungeons_draws(vendor_rng):
    """Replay the ISAAC64 draws of vendor ``init_dungeons`` byte-exactly.

    Ports ``vendor/nle/src/dungeon.c::init_dungeons`` (line 714) line-for-line.
    The vendor function walks the 8 dungeons defined in ``vendor/nle/dat/dungeon.def``
    in order, and for each dungeon ``i`` interleaves the following draws (in
    this order) before moving to dungeon ``i+1``:

    1. Dungeon-skip gate (only if ``tmpdungeon[i].chance > 0``)::
           if (!wizard && tmpdungeon[i].chance && (tmpdungeon[i].chance <= rn2(100)))
       Cite: dungeon.c:775-776.  Only Fort Ludios (chance=10) triggers this.

    2. Dungeon depth (only if ``tmpdungeon[i].lev.rand > 0``)::
           dungeons[i].num_dunlevs = rn1(lev.rand, lev.base);  // == rn2(rand)+base
       Cite: dungeon.c:797-798.

    3. ``add_branch`` → ``parent_dlevel`` (only if ``i > 0``)::
           i = j = rn2(num);   // num = level_range(...) of attach window
       Cite: dungeon.c:845 (add_branch call), 502 (call to parent_dlevel),
             398 (the rn2 itself).

    4. ``init_level`` per prototype level for this dungeon, in dungeon.def
       order::
           if (!wizard && tlevel->chance <= rn2(100)) return;
       Cite: dungeon.c:548.  ``tlevel->chance`` defaults to 100 for ``LEVEL`` /
       ``CHAINLEVEL`` (dgn_comp.y:466), so the draw fires for EVERY level —
       not just for RNDLEVEL entries — but only RNDLEVELs can be gated out.

    5. ``place_level`` recursion → one ``rn2(npossible)`` per CREATED level
       (placed-not-gated), in dungeon.def order.  Cite: dungeon.c:661.  In
       failure-free placement (the common case) each created level draws
       exactly once; backtracking would add extra draws, which we accept as
       seed-dependent and trust the trace if/when divergence reappears.

    After all 8 dungeons::

    6. tune string — 5 unconditional ``rn2(7)`` draws.  Cite: dungeon.c:917-918.

    Args:
        vendor_rng: Isaac64State from ``Nethax.nethax.vendor_rng``.

    Returns:
        ``(new_vendor_rng, dungeon_state)`` where ``dungeon_state`` is a
        small dict of the values caller may want for downstream wiring.
    """
    from Nethax.nethax.vendor_rng import rn2_jax as _rn2

    drawn = {}

    # =======================================================================
    # i = 0 : "The Dungeons of Doom" (25, 5)
    #         tmpdungeon[0].chance = 0 → no skip-gate
    #         lev.rand = 5            → depth draw fires
    #         i == 0                  → no add_branch
    #         5 prototype levels: rogue, oracle, bigrm, medusa, castle
    # =======================================================================

    # depth draw — dungeon.c:797-798
    vendor_rng, dod_levels = _rn2(vendor_rng, 5)   # rn1(5, 25)
    drawn["dod_levels"] = dod_levels + jnp.int32(25)

    # init_level rn2(100) gates — dungeon.c:548 — in dungeon.def order.
    # PARITY FIX: dgn_comp.y's RNDLEVEL 1-INT production (lines 188-198) sets
    # only ``rndlevs`` and leaves ``chance`` at the init_level() default of 100.
    # Only ``bigrm`` (which has 2 INTs in dungeon.def) gets chance=40; medusa,
    # minetn, minend, and the four soko levels all get chance=100 (always
    # placed).  Verified by parsing the compiled
    # vendor/nle/build/.../dat/dungeon binary: medusa.chance=100.
    # Cite: vendor/nle/util/dgn_comp.y lines 188-209;
    # vendor/nle/dat/dungeon.def medusa/minetn/minend/soko entries.
    vendor_rng, gate_rogue  = _rn2(vendor_rng, 100)  # rogue   chance=100 (LEVEL)
    vendor_rng, gate_oracle = _rn2(vendor_rng, 100)  # oracle  chance=100 (LEVEL)
    vendor_rng, gate_bigrm  = _rn2(vendor_rng, 100)  # bigrm   chance=40  (RNDLEVEL 2-int)
    vendor_rng, gate_medusa = _rn2(vendor_rng, 100)  # medusa  chance=100 (RNDLEVEL 1-int → default)
    vendor_rng, gate_castle = _rn2(vendor_rng, 100)  # castle  chance=100 (LEVEL)

    bigrm_placed  = gate_bigrm < jnp.int32(40)          # traced bool
    medusa_placed = jnp.bool_(True)  # chance=100 → always placed (1-INT RNDLEVEL default)

    drawn["gate_bigrm"]  = gate_bigrm
    drawn["gate_medusa"] = gate_medusa

    # place_level recursion — dungeon.c:637-679.  See ``_place_dungeon_levels_jax``.
    dod_num_dunlevs = dod_levels + jnp.int32(25)
    dod_protos = [
        {"name": "rogue",  "base": 15, "rand": 4, "chain_idx": None, "created": True},
        {"name": "oracle", "base": 5,  "rand": 5, "chain_idx": None, "created": True},
        {"name": "bigrm",  "base": 10, "rand": 3, "chain_idx": None, "created": True},   # dyn override below
        {"name": "medusa", "base": -5, "rand": 4, "chain_idx": None, "created": True},
        {"name": "castle", "base": -1, "rand": 0, "chain_idx": None, "created": True},
    ]
    dod_created_dyn = jnp.stack([
        jnp.bool_(True), jnp.bool_(True), bigrm_placed, medusa_placed, jnp.bool_(True),
    ])
    vendor_rng, _ = _place_dungeon_levels_jax(
        vendor_rng, dod_num_dunlevs, dod_protos, created_dyn=dod_created_dyn
    )

    # =======================================================================
    # i = 1 : "Gehennom" (20, 5)
    #         tmpdungeon.chance = 0 → no skip-gate
    #         lev.rand = 5          → depth draw
    #         i > 0                 → parent_dlevel rn2; CHAINBRANCH "castle"+(0,0) → num=1
    #         11 levels: valley, sanctum, juiblex, baalz, asmodeus,
    #                    wizard1, wizard2, wizard3, orcus, fakewiz1, fakewiz2
    # =======================================================================

    vendor_rng, geh_levels = _rn2(vendor_rng, 5)   # dungeon.c:797-798 rn1(5, 20)
    drawn["geh_levels"] = geh_levels + jnp.int32(20)

    vendor_rng, _ = _rn2(vendor_rng, 1)            # dungeon.c:398 parent_dlevel num=1

    # init_level rn2(100) gates — all LEVEL/CHAINLEVEL chance=100
    vendor_rng, _ = _rn2(vendor_rng, 100)          # dungeon.c:548 valley
    vendor_rng, _ = _rn2(vendor_rng, 100)          # dungeon.c:548 sanctum
    vendor_rng, _ = _rn2(vendor_rng, 100)          # dungeon.c:548 juiblex
    vendor_rng, _ = _rn2(vendor_rng, 100)          # dungeon.c:548 baalz
    vendor_rng, _ = _rn2(vendor_rng, 100)          # dungeon.c:548 asmodeus
    vendor_rng, _ = _rn2(vendor_rng, 100)          # dungeon.c:548 wizard1
    vendor_rng, _ = _rn2(vendor_rng, 100)          # dungeon.c:548 wizard2
    vendor_rng, _ = _rn2(vendor_rng, 100)          # dungeon.c:548 wizard3
    vendor_rng, _ = _rn2(vendor_rng, 100)          # dungeon.c:548 orcus
    vendor_rng, _ = _rn2(vendor_rng, 100)          # dungeon.c:548 fakewiz1
    vendor_rng, _ = _rn2(vendor_rng, 100)          # dungeon.c:548 fakewiz2

    # place_level recursion — dungeon.c:637-679.
    # Gehennom has CHAINLEVEL entries (wizard2/wizard3 chain to wizard1), which
    # can force backtracking when wizard1's chosen slot leaves no room for the
    # chained slots.  The JAX placer handles this correctly.
    geh_num_dunlevs = geh_levels + jnp.int32(20)
    # local proto indices: 0=valley, 1=sanctum, 2=juiblex, 3=baalz, 4=asmodeus,
    # 5=wizard1, 6=wizard2 (CHAIN→5), 7=wizard3 (CHAIN→5), 8=orcus,
    # 9=fakewiz1, 10=fakewiz2.
    geh_protos = [
        {"name": "valley",   "base": 1,   "rand": 0, "chain_idx": None, "created": True},
        {"name": "sanctum",  "base": -1,  "rand": 0, "chain_idx": None, "created": True},
        {"name": "juiblex",  "base": 4,   "rand": 4, "chain_idx": None, "created": True},
        {"name": "baalz",    "base": 6,   "rand": 4, "chain_idx": None, "created": True},
        {"name": "asmodeus", "base": 2,   "rand": 6, "chain_idx": None, "created": True},
        {"name": "wizard1",  "base": 11,  "rand": 6, "chain_idx": None, "created": True},
        {"name": "wizard2",  "base": 1,   "rand": 0, "chain_idx": 5,    "created": True},
        {"name": "wizard3",  "base": 2,   "rand": 0, "chain_idx": 5,    "created": True},
        {"name": "orcus",    "base": 10,  "rand": 6, "chain_idx": None, "created": True},
        {"name": "fakewiz1", "base": -6,  "rand": 4, "chain_idx": None, "created": True},
        {"name": "fakewiz2", "base": -6,  "rand": 4, "chain_idx": None, "created": True},
    ]
    vendor_rng, _ = _place_dungeon_levels_jax(vendor_rng, geh_num_dunlevs, geh_protos)

    # =======================================================================
    # i = 2 : "The Gnomish Mines" (8, 2)
    #         chance=0, lev.rand=2, i>0 (BRANCH @ (2,3) → num=3)
    #         2 RNDLEVEL levels: minetn (ch=7), minend (ch=3)
    # =======================================================================

    vendor_rng, mines_levels = _rn2(vendor_rng, 2)  # dungeon.c:797-798 rn1(2, 8)
    drawn["mines_levels"] = mines_levels + jnp.int32(8)

    vendor_rng, _ = _rn2(vendor_rng, 3)             # dungeon.c:398 parent_dlevel num=3

    vendor_rng, gate_minetn = _rn2(vendor_rng, 100)  # dungeon.c:548 minetn chance=100 (1-int RNDLEVEL)
    vendor_rng, gate_minend = _rn2(vendor_rng, 100)  # dungeon.c:548 minend chance=100 (1-int RNDLEVEL)
    # chance=100 → always placed (static True)
    drawn["gate_minetn"] = gate_minetn
    drawn["gate_minend"] = gate_minend

    mines_num_dunlevs = mines_levels + jnp.int32(8)
    mines_protos = [
        {"name": "minetn", "base": 3,  "rand": 2, "chain_idx": None, "created": True},
        {"name": "minend", "base": -1, "rand": 0, "chain_idx": None, "created": True},
    ]
    vendor_rng, _ = _place_dungeon_levels_jax(vendor_rng, mines_num_dunlevs, mines_protos)

    # =======================================================================
    # i = 3 : "The Quest" (5, 2)
    #         chance=0, lev.rand=2, i>0 (CHAINBRANCH "oracle"+(6,2) portal → num=2)
    #         3 LEVEL levels: x-strt, x-loca, x-goal
    # =======================================================================

    vendor_rng, quest_levels = _rn2(vendor_rng, 2)  # dungeon.c:797-798 rn1(2, 5)
    drawn["quest_levels"] = quest_levels + jnp.int32(5)

    vendor_rng, _ = _rn2(vendor_rng, 2)             # dungeon.c:398 parent_dlevel num=2

    vendor_rng, _ = _rn2(vendor_rng, 100)           # dungeon.c:548 x-strt
    vendor_rng, _ = _rn2(vendor_rng, 100)           # dungeon.c:548 x-loca
    vendor_rng, _ = _rn2(vendor_rng, 100)           # dungeon.c:548 x-goal

    quest_num_dunlevs = quest_levels + jnp.int32(5)
    quest_protos = [
        {"name": "x-strt", "base": 1,  "rand": 1, "chain_idx": None, "created": True},
        {"name": "x-loca", "base": 3,  "rand": 1, "chain_idx": None, "created": True},
        {"name": "x-goal", "base": -1, "rand": 0, "chain_idx": None, "created": True},
    ]
    vendor_rng, _ = _place_dungeon_levels_jax(vendor_rng, quest_num_dunlevs, quest_protos)

    # =======================================================================
    # i = 4 : "Sokoban" (4, 0)
    #         chance=0, lev.rand=0 → NO depth draw
    #         i>0 (CHAINBRANCH "oracle"+(1,0) up → num=1)
    #         4 RNDLEVEL levels: soko1..soko4 (ch=2 each)
    # =======================================================================

    vendor_rng, _ = _rn2(vendor_rng, 1)             # dungeon.c:398 parent_dlevel num=1

    vendor_rng, gate_soko1 = _rn2(vendor_rng, 100)  # dungeon.c:548 soko1 chance=100 (1-int RNDLEVEL)
    vendor_rng, gate_soko2 = _rn2(vendor_rng, 100)  # dungeon.c:548 soko2 chance=100 (1-int RNDLEVEL)
    vendor_rng, gate_soko3 = _rn2(vendor_rng, 100)  # dungeon.c:548 soko3 chance=100 (1-int RNDLEVEL)
    vendor_rng, gate_soko4 = _rn2(vendor_rng, 100)  # dungeon.c:548 soko4 chance=100 (1-int RNDLEVEL)
    drawn["gate_soko1"] = gate_soko1
    drawn["gate_soko2"] = gate_soko2
    drawn["gate_soko3"] = gate_soko3
    drawn["gate_soko4"] = gate_soko4

    soko_num_dunlevs = jnp.int32(4)  # Sokoban (4, 0) — lev.rand=0 → fixed
    soko_protos = [
        {"name": "soko1", "base": 1, "rand": 0, "chain_idx": None, "created": True},
        {"name": "soko2", "base": 2, "rand": 0, "chain_idx": None, "created": True},
        {"name": "soko3", "base": 3, "rand": 0, "chain_idx": None, "created": True},
        {"name": "soko4", "base": 4, "rand": 0, "chain_idx": None, "created": True},
    ]
    vendor_rng, _ = _place_dungeon_levels_jax(vendor_rng, soko_num_dunlevs, soko_protos)

    # =======================================================================
    # i = 5 : "Fort Ludios" (1, 0)
    #         chance=0 (NOT 10!) per binary parse of
    #         vendor/nle/build/.../dat/dungeon — DUNGEON line in
    #         dungeon.def has no INT, so dgn_comp.y's optional_int rule
    #         (line ~158) defaults chance to 0.  init_dungeons short-circuits
    #         at ``pd.tmpdungeon[i].chance && ...`` (dungeon.c:775) so NO
    #         skip-gate rn2 fires.  Always placed.
    #         lev.rand=0 → no depth draw; 1 LEVEL: knox (chance=100 LEVEL).
    #         BRANCH @ (18,4) portal in Main → parent_dlevel num=4.
    # =======================================================================

    vendor_rng, _ = _rn2(vendor_rng, 4)         # dungeon.c:398 parent_dlevel num=4
    vendor_rng, _ = _rn2(vendor_rng, 100)       # dungeon.c:548 knox
    knox_protos = [
        {"name": "knox", "base": -1, "rand": 0, "chain_idx": None, "created": True},
    ]
    vendor_rng, _ = _place_dungeon_levels_jax(vendor_rng, jnp.int32(1), knox_protos)

    # =======================================================================
    # i = 6 : "Vlad's Tower" (3, 0)
    #         chance=0, lev.rand=0 (no depth), i>0 (BRANCH @ (9,5) up in Gehennom → num=5)
    #         3 LEVEL: tower1, tower2, tower3
    # =======================================================================

    vendor_rng, _ = _rn2(vendor_rng, 5)             # dungeon.c:398 parent_dlevel num=5

    vendor_rng, _ = _rn2(vendor_rng, 100)           # dungeon.c:548 tower1
    vendor_rng, _ = _rn2(vendor_rng, 100)           # dungeon.c:548 tower2
    vendor_rng, _ = _rn2(vendor_rng, 100)           # dungeon.c:548 tower3

    vlad_protos = [
        {"name": "tower1", "base": 1, "rand": 0, "chain_idx": None, "created": True},
        {"name": "tower2", "base": 2, "rand": 0, "chain_idx": None, "created": True},
        {"name": "tower3", "base": 3, "rand": 0, "chain_idx": None, "created": True},
    ]
    vendor_rng, _ = _place_dungeon_levels_jax(vendor_rng, jnp.int32(3), vlad_protos)

    # =======================================================================
    # i = 7 : "The Elemental Planes" (6, 0)
    #         chance=0, lev.rand=0 (no depth), i>0 (BRANCH @ (1,0) no_down up → num=1)
    #         6 LEVEL: astral, water, fire, air, earth, dummy
    # =======================================================================

    vendor_rng, _ = _rn2(vendor_rng, 1)             # dungeon.c:398 parent_dlevel num=1

    vendor_rng, _ = _rn2(vendor_rng, 100)           # dungeon.c:548 astral
    vendor_rng, _ = _rn2(vendor_rng, 100)           # dungeon.c:548 water
    vendor_rng, _ = _rn2(vendor_rng, 100)           # dungeon.c:548 fire
    vendor_rng, _ = _rn2(vendor_rng, 100)           # dungeon.c:548 air
    vendor_rng, _ = _rn2(vendor_rng, 100)           # dungeon.c:548 earth
    vendor_rng, _ = _rn2(vendor_rng, 100)           # dungeon.c:548 dummy

    planes_protos = [
        {"name": "astral", "base": 1, "rand": 0, "chain_idx": None, "created": True},
        {"name": "water",  "base": 2, "rand": 0, "chain_idx": None, "created": True},
        {"name": "fire",   "base": 3, "rand": 0, "chain_idx": None, "created": True},
        {"name": "air",    "base": 4, "rand": 0, "chain_idx": None, "created": True},
        {"name": "earth",  "base": 5, "rand": 0, "chain_idx": None, "created": True},
        {"name": "dummy",  "base": 6, "rand": 0, "chain_idx": None, "created": True},
    ]
    vendor_rng, _ = _place_dungeon_levels_jax(vendor_rng, jnp.int32(6), planes_protos)

    # =======================================================================
    # After the per-dungeon loop: tune string — dungeon.c:917-918
    #     for (i = 0; i < 5; i++) tune[i] = 'A' + rn2(7);
    # =======================================================================
    vendor_rng, _ = _rn2(vendor_rng, 7)              # dungeon.c:918 tune[0]
    vendor_rng, _ = _rn2(vendor_rng, 7)              # dungeon.c:918 tune[1]
    vendor_rng, _ = _rn2(vendor_rng, 7)              # dungeon.c:918 tune[2]
    vendor_rng, _ = _rn2(vendor_rng, 7)              # dungeon.c:918 tune[3]
    vendor_rng, _ = _rn2(vendor_rng, 7)              # dungeon.c:918 tune[4]

    return vendor_rng, drawn


def consume_init_dungeons_variable_draws(vendor_rng, dungeon_state):
    """Pass-through: all init_dungeons draws are now consumed by
    ``consume_init_dungeons_draws`` in a single byte-exact per-dungeon walk.

    The previous two-phase split (fixed pre-draws + variable post-draws) was
    factored out so the per-dungeon order — skip-gate, depth, parent_dlevel,
    init_level, place_level — could be honored exactly as vendor
    ``init_dungeons`` interleaves them.  This function is retained for ABI
    compatibility with existing callers in ``env.py``.

    Citation: vendor/nle/src/dungeon.c:714-918 (entire init_dungeons body).
    """
    return vendor_rng, dungeon_state


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
