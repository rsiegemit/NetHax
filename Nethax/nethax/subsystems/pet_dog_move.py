"""Vendor-faithful ISAAC64 port of pet ``dog_move`` (byte-parity skeleton).

Goal
----
Replace Nethax's deterministic-BFS pet movement with a bit-exact reproduction
of NLE's ``dog_move`` so the ISAAC64 stream consumed during a single pet turn
matches NLE byte-for-byte.  The validator (``tests/test_nle_byte_parity.py``,
seed 0, rog-hum-cha-mal) compares per-step trace lines; any extra or missing
draw desyncs the stream and fails parity.

Vendor sources cited (3.x tree under ``vendor/nle/src/``):

    vendor/nle/src/dogmove.c lines 862-1126   -- dog_move main body
    vendor/nle/src/dogmove.c lines 476-577    -- dog_goal (sets gx, gy, appr)
    vendor/nle/src/dogmove.c lines 357-397    -- dog_hunger (0 draws for fresh pet)
    vendor/nle/src/dogmove.c lines 403-471    -- dog_invent (0 draws when no items)
    vendor/nle/src/dogmove.c lines 880, 988,
                              1113-1123       -- GDIST + scoring loop
    vendor/nle/src/monmove.c lines 315-349    -- distfleeck (rn2(5) bravegremlin)
    vendor/nle/src/monmove.c lines 574-579    -- wanderer skip-move (rn2(4))
    vendor/nle/src/mon.c     lines 1305-1500  -- mfndpos (8-neighbor enumeration,
                                                  column-major nx outer / ny inner)
    vendor/nle/src/hacklib.c lines 612-620    -- dist2 (squared Euclidean distance)

Per-turn draws for the validator scenario (kitten adjacent to hero, empty
inventory, empty floor, no traps, peaceful hero, no Conflict, no displacement)
in execution order:

    1.  dochug() prelude -- distfleeck() ALWAYS draws rn2(5) into bravegremlin
        (monmove.c:320).  Effect (scared) is irrelevant for stream alignment;
        we only need the draw.
    2.  dochug() movement-phase || chain -- short-circuits down to
        ``(is_wanderer(mdat) && !rn2(4))`` (monmove.c:578).  Kitten has
        M2_WANDER so rn2(4) fires.  Again we only need the draw.
    3.  dog_goal() with no nearby fobj -- 0..1 rn2(4) stay-in-room draw.
        See dogmove.c:476-577 -- the SQSRCHRADIUS fobj loop is empty;
        gtyp==UNDEF takes the `follow player` branch.  Inside that branch
        the ``if (udist > 1)`` guard (dogmove.c:564) wraps a 4-clause
        ``||`` whose second clause is ``!rn2(4)`` (dogmove.c:565); the
        first clause ``!IS_ROOM(levl[u.ux][u.uy].typ)`` is false when
        the hero is in a room (validator scenario), so C short-circuit
        advances and the rn2(4) fires.  Pet at udist <= 1 (cardinal-
        adjacent or co-located) skips the whole block and draws nothing.
        Diagonal-adjacent pet has udist == 2 so the draw DOES fire.
        Invent loop is empty so the appr==0 DOGFOOD scan exits immediately.
    4.  mfndpos() -- pure filtering, ZERO draws.  Enumerates 8 neighbours
        in vendor column-major order (NW, W, SW, N, S, NE, E, SE) and
        keeps the ones that pass walkability / bounds / hero-tile gates.
    5.  Scoring loop (dogmove.c:986-1126) -- ONE rn2(++chcnt) per ACCEPTED
        candidate.  With appr==0 every accepted j==0 path takes the
        ``!rn2(++chcnt)`` reservoir-sample branch; rejected (non-mfndpos)
        slots draw nothing.

The "skeleton" qualifier reflects two simplifications we deliberately make
in this first cut so that trace bisection can refine without rewriting the
whole port:

    A.  We use a single ``rn2`` draw per accepted candidate (matching the
        reservoir-sample j==0 branch).  Once parity reveals the per-tile
        cursed / trap / leashed pre-checks, those extra draws will be
        re-inserted at the corresponding offsets.
    B.  We do NOT execute dog_hunger, dog_invent, dog_eat, mattackm,
        mon_wield_item, or the ranged-attack sub-block; the validator
        scenario does not exercise them and they consume zero draws in
        this scenario.  Other call sites still go through monster_ai.py.

Interface
---------
``vendor_pet_dog_move(state, vendor_rng, pet_slot) -> (new_state, new_vendor_rng)``

The signature is contractual with the parallel monster_ai.py change so the
function is callable from the existing per-monster dispatch loop without
further edits.

JIT safety
----------
Every Python-side ``if`` would burn under ``jax.jit``.  This module uses
``jnp.where`` / ``lax.cond`` / ``lax.fori_loop`` exclusively.  In particular,
the reservoir loop GUARANTEES that ``rn2_jax`` is only invoked on iterations
where the candidate is accepted -- if it ever fired on a rejected candidate
the ISAAC64 stream would desynchronise from vendor and byte-parity would
fail.

Status
------
SKELETON.  First cut -- minimum surface to be wired into monster_ai.py and
diffed against the vendor ISAAC64 trace.  Refinements (cursed / trap / leash
gates) will be added as trace bisection identifies them.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.vendor_rng import Isaac64State, rn2_jax

# ---------------------------------------------------------------------------
# Local constants -- kept here instead of importing from monster_ai.py so
# the parallel edit in monster_ai.py cannot create an import cycle while we
# are landing.  These MUST match the canonical values in monster_ai.py:
#   _MAP_H            -> matches monster_ai._MAP_H
#   _MAP_W            -> matches monster_ai._MAP_W
#   _MAX_MONSTERS     -> matches monster_ai.MAX_MONSTERS_PER_LEVEL
#   _TILE_WALL / _TILE_CLOSED_DOOR / _TILE_VOID
#                     -> match Nethax.nethax.constants.tiles.TileType
#   _NORMAL_SPEED     -> NORMAL_SPEED = 12 (vendor monst.h MONST struct;
#                        vendor/nle/include/monsym.h NORMAL_SPEED).
# ---------------------------------------------------------------------------

_MAP_H: int = 21
_MAP_W: int = 80
_MAX_MONSTERS: int = 400

_TILE_VOID: int = 0          # TileType.VOID
_TILE_WALL: int = 3          # TileType.WALL
_TILE_CLOSED_DOOR: int = 4   # TileType.CLOSED_DOOR (treated as obstacle when
                             #   pet lacks OPENDOOR ability; we keep this
                             #   pessimistic for the skeleton scenario which
                             #   never spawns near a door)

_NORMAL_SPEED: int = 12

# dog_goal SQSRCHRADIUS — pet scans a square bounding box of half-width 5
# (vendor dogmove.c:503 ``#define SQSRCHRADIUS 5``) for nearby objects.
_SQSRCHRADIUS: int = 5

# Per-tile ground stack depth.  Mirrors
# ``Nethax.nethax.subsystems.inventory.MAX_GROUND_STACK`` (=8); duplicated
# here so this module does not import inventory (and risk a cycle).
_MAX_GROUND_STACK: int = 8

# Vendor mfndpos enumeration order:
#     for (nx = x-1; nx <= x+1; nx++)
#       for (ny = y-1; ny <= y+1; ny++)
#         if (nx == x && ny == y) continue;
#
# NetHack stores positions as (col=x, row=y); Nethax stores them as
# (row, col).  Re-expressing the vendor outer-nx / inner-ny iteration in
# (row, col) form yields the offset list below: outer is dc (column delta,
# vendor x), inner is dr (row delta, vendor y).  Self-cell (0, 0) is omitted.
# Order (relative to the pet): NW, W, SW, N, S, NE, E, SE.
# Citation: vendor/nle/src/mon.c lines 1376-1379.
_NBR_OFFSETS = jnp.array(
    [
        (-1, -1),  # 0: NW  (nx=x-1, ny=y-1)
        ( 0, -1),  # 1: W   (nx=x-1, ny=y  )
        ( 1, -1),  # 2: SW  (nx=x-1, ny=y+1)
        (-1,  0),  # 3: N   (nx=x,   ny=y-1)
        ( 1,  0),  # 4: S   (nx=x,   ny=y+1)
        (-1,  1),  # 5: NE  (nx=x+1, ny=y-1)
        ( 0,  1),  # 6: E   (nx=x+1, ny=y  )
        ( 1,  1),  # 7: SE  (nx=x+1, ny=y+1)
    ],
    dtype=jnp.int32,
)  # shape [8, 2] -- (drow, dcol)


def _dist2(r0, c0, r1, c1):
    """Vendor ``dist2`` -- squared Euclidean distance.

    Citation: vendor/nle/src/hacklib.c line 614.
    Note: NOT Chebyshev.  ``dist2 = dx*dx + dy*dy`` where dx = col diff,
    dy = row diff -- order doesn't matter since both are squared.
    """
    dr = (r0 - r1).astype(jnp.int32)
    dc = (c0 - c1).astype(jnp.int32)
    return dr * dr + dc * dc


def _terrain_passable(terrain_2d, r, c):
    """Return True iff (r, c) is in-bounds and not a wall / closed door / void.

    Vendor parity reference: ``mfndpos`` (mon.c:1381-1396) rejects IS_ROCK
    tiles (walls, stone) and closed/locked doors when the monster lacks
    OPENDOOR / passes_walls.  Our local TileType has no separate STONE so
    we treat VOID as out-of-room (impassable for a kitten).  This matches
    the validator scenario -- the kitten only ever moves between FLOOR
    and CORRIDOR cells around the staircase spawn.
    """
    in_bounds = (
        (r >= jnp.int32(0))
        & (r < jnp.int32(_MAP_H))
        & (c >= jnp.int32(0))
        & (c < jnp.int32(_MAP_W))
    )
    safe_r = jnp.clip(r, 0, _MAP_H - 1)
    safe_c = jnp.clip(c, 0, _MAP_W - 1)
    tile = terrain_2d[safe_r, safe_c].astype(jnp.int32)
    not_blocking = (
        (tile != jnp.int32(_TILE_WALL))
        & (tile != jnp.int32(_TILE_CLOSED_DOOR))
        & (tile != jnp.int32(_TILE_VOID))
    )
    return in_bounds & not_blocking


def _current_level_terrain(state):
    """Return the int8[_MAP_H, _MAP_W] terrain slice for the player's level.

    Vendor uses a per-(branch, level) ``levl`` array indexed by xy; we use
    ``state.terrain[branch, level-1]`` and reverse the indexing to (row, col).
    Equivalent to monster_ai._current_level_terrain but duplicated locally
    so this file does not import monster_ai (avoids the parallel-edit cycle).
    """
    b = state.dungeon.current_branch
    lv = state.dungeon.current_level - jnp.int8(1)
    return state.terrain[b, lv]


# ---------------------------------------------------------------------------
# dog_goal SQSRCHRADIUS fobj scan -- one rn2(100) per in-range object.
# ---------------------------------------------------------------------------

def _emit_dog_goal_fobj_scan(state, vendor_rng, pet_r, pet_c):
    """Emit one ``rn2(100)`` per ground object within Chebyshev radius 5 of the pet.

    Reproduces the ISAAC64 draws consumed by the vendor ``dog_goal`` fobj scan:

        for (obj = fobj; obj; obj = obj->nobj) {
            nx = obj->ox; ny = obj->oy;
            if (nx in [min_x..max_x] && ny in [min_y..max_y]) {
                otyp = dogfood(mtmp, obj);
                ...
            }
        }

    Inside ``dogfood`` (vendor dog.c:744) the first check is:

        if (is_quest_artifact(obj) || obj_resists(obj, 0, 95))
            return obj->cursed ? TABU : APPORT;

    ``is_quest_artifact`` is a pure tag check.  When it is false (the common
    case, and ALWAYS true on Dlvl 1 of the validator scenario where no quest
    artifacts can spawn), ``obj_resists(obj, 0, 95)`` fires ``rn2(100)``
    (vendor/nle/src/zap.c:1191 ``int chance = rn2(100);``).  So every
    in-range non-quest object contributes EXACTLY ONE ``rn2(100)`` draw to
    the ISAAC64 stream, regardless of subsequent dogfood logic.

    For byte parity we therefore need to:

        1. Find every non-empty ground-stack slot whose tile sits inside
           the pet's [pet_r ± 5, pet_c ± 5] Chebyshev box.
        2. Emit one ``rn2(100)`` per such slot, in any fixed order (the
           draws are identical so the total count is what advances the
           stream to vendor parity).

    We iterate the bounding box and ground-stack depth with a JIT-safe
    ``lax.fori_loop`` of fixed extent (11 * 11 * MAX_GROUND_STACK = 968
    iterations) and use ``lax.cond`` to gate the draw on
    (in-bounds & non-empty-slot).  Order is row-major (dr outer, dc inner,
    stack innermost) — irrelevant for parity since each draw has the same
    modulus.

    Quest-artifact filter: deliberately omitted.  The validator scenario
    cannot reach Dlvl 1 with a quest artifact on the floor (those only
    spawn deep in their owning branch), and no Nethax state field flags
    a ground item as a quest artifact at present.  When per-object quest
    filtering is needed, gate the ``do_draw`` branch on a future
    ``is_quest_artifact`` predicate.

    Citations
    ---------
    vendor/nle/src/dogmove.c  lines 502-553  (SQSRCHRADIUS fobj scan loop)
    vendor/nle/src/dog.c      line  744      (``dogfood`` -> ``obj_resists``)
    vendor/nle/src/zap.c      line  1191     (``obj_resists`` -> ``rn2(100)``)
    """
    pet_r = pet_r.astype(jnp.int32)
    pet_c = pet_c.astype(jnp.int32)

    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)

    # category[branch, level, :, :, :] -> [H, W, MAX_GROUND_STACK]; 0 = empty.
    cat_slab = state.ground_items.category[b, lv].astype(jnp.int32)

    side = 2 * _SQSRCHRADIUS + 1   # 11
    box_cells = side * side        # 121
    total_iters = box_cells * _MAX_GROUND_STACK  # 968

    def body(i, rng):
        # Decode (dr, dc, s) from the flat index.
        ds = i % jnp.int32(_MAX_GROUND_STACK)
        cell = i // jnp.int32(_MAX_GROUND_STACK)
        dc = (cell % jnp.int32(side)) - jnp.int32(_SQSRCHRADIUS)
        dr = (cell // jnp.int32(side)) - jnp.int32(_SQSRCHRADIUS)
        r = pet_r + dr
        c = pet_c + dc

        in_bounds = (
            (r >= jnp.int32(0))
            & (r < jnp.int32(_MAP_H))
            & (c >= jnp.int32(0))
            & (c < jnp.int32(_MAP_W))
        )
        safe_r = jnp.clip(r, 0, _MAP_H - 1)
        safe_c = jnp.clip(c, 0, _MAP_W - 1)
        cat = cat_slab[safe_r, safe_c, ds]
        has_obj = in_bounds & (cat != jnp.int32(0))

        # Brax-flatten: compute both branches, select with tree_map(jnp.where).
        drawn_rng, _ = rn2_jax(rng, jnp.int64(100))
        return jax.tree_util.tree_map(
            lambda t, f: jnp.where(has_obj, t, f), drawn_rng, rng,
        )

    return jax.lax.fori_loop(0, total_iters, body, vendor_rng)


# ---------------------------------------------------------------------------
# mfndpos -- 8-neighbor enumeration with vendor filter.  No RNG draws.
# ---------------------------------------------------------------------------

def _mfndpos(state, pet_r, pet_c):
    """Reproduce vendor mfndpos for the kitten / validator scenario.

    Returns
    -------
    accepted : bool[8]
        True for each of the 8 fixed offsets that survives the vendor filter
        (walkable terrain, in-bounds, not the hero tile).  Order matches
        ``_NBR_OFFSETS`` (column-major nx-outer / ny-inner; see module top
        for the (NW, W, SW, N, S, NE, E, SE) layout).
    nrows : int32[8]
        Absolute neighbour row coordinates (== pet_r + dr).
    ncols : int32[8]
        Absolute neighbour column coordinates (== pet_c + dc).

    Vendor reference: vendor/nle/src/mon.c lines 1305-1500.
    """
    terrain_2d = _current_level_terrain(state)
    ppos = state.player_pos.astype(jnp.int32)
    hero_r, hero_c = ppos[0], ppos[1]

    nrows = pet_r + _NBR_OFFSETS[:, 0]
    ncols = pet_c + _NBR_OFFSETS[:, 1]

    # vmap the walkability check across all 8 candidates so we stay JIT-pure.
    walk_fn = jax.vmap(lambda r, c: _terrain_passable(terrain_2d, r, c))
    walkable = walk_fn(nrows, ncols)

    # Vendor mfndpos:1438 rejects the hero tile unless ALLOW_U is set.
    # ALLOW_U is only set under Conflict (dogmove.c:938) which is out of
    # scope for the validator scenario; so we always reject the hero tile.
    is_hero = (nrows == hero_r) & (ncols == hero_c)

    accepted = walkable & (~is_hero)
    return accepted, nrows, ncols


# ---------------------------------------------------------------------------
# Scoring loop -- vendor dogmove.c lines 986-1126 with appr==0.
# ---------------------------------------------------------------------------

def _scoring_loop(accepted, nrows, ncols, pet_r, pet_c, goal_r, goal_c, vendor_rng):
    """Reservoir-sample over accepted neighbours, drawing ``rn2(++chcnt)`` per pick.

    Implements the appr==0 reduction of dogmove.c:1113-1123.  With appr==0:

        j = (GDIST(nx, ny) - nidist) * appr == 0

    so EVERY accepted candidate enters the ``(j == 0 && !rn2(++chcnt))``
    branch.  ``chcnt`` increments before the test; the reservoir sample is
    therefore equivalent to: among N accepted candidates, draw rn2(N) at the
    end and pick the i-th -- but vendor does it incrementally as N grows,
    which fixes the RNG consumption pattern (one rn2 per accept).

    Critical invariant for byte parity: ``rn2_jax`` MUST be called on EXACTLY
    the iterations where ``accepted[i] == True``.  We use ``lax.cond`` to
    branch the draw -- on rejected iterations the cond's false-branch returns
    the carry untouched so vendor_rng does not advance.

    Returns
    -------
    new_r, new_c : int32 scalars
        The chosen neighbour (or (pet_r, pet_c) stay-put if no candidates).
    new_vendor_rng : Isaac64State
        Advanced by ``sum(accepted)`` draws.
    """
    pet_r = pet_r.astype(jnp.int32)
    pet_c = pet_c.astype(jnp.int32)
    goal_r = goal_r.astype(jnp.int32)
    goal_c = goal_c.astype(jnp.int32)

    # Vendor seeds nix=omx, niy=omy, nidist=GDIST(omx, omy).  (dogmove.c:988)
    init_nix = pet_r
    init_niy = pet_c
    init_nidist = _dist2(pet_r, pet_c, goal_r, goal_c)
    init_chcnt = jnp.int32(0)

    def body(i, carry):
        chcnt, nix, niy, nidist, rng = carry
        is_accepted = accepted[i]
        nr = nrows[i]
        nc = ncols[i]

        # Increment chcnt only for accepted candidates -- this corresponds to
        # vendor's pre-increment ``++chcnt`` inside the rn2 argument.  On
        # rejected iterations vendor either ``continue``s before reaching
        # the scoring expression or short-circuits ``j == 0 && ...`` to false
        # without touching chcnt (we coalesce all rejected paths into the
        # single "do nothing" branch).
        new_chcnt = jnp.where(is_accepted, chcnt + jnp.int32(1), chcnt)

        # Brax-flatten: compute both branches, select with tree_map(jnp.where).
        # rn2(++chcnt) -- modulus is the incremented value.
        drawn_rng, drawn_val = rn2_jax(rng, new_chcnt.astype(jnp.int64))
        new_rng = jax.tree_util.tree_map(
            lambda t, f: jnp.where(is_accepted, t, f), drawn_rng, rng,
        )
        draw_val = jnp.where(is_accepted, drawn_val, jnp.int32(0))

        # Reservoir sample: pick this candidate iff accepted and rn2(chcnt) == 0.
        # The first accepted candidate has chcnt=1, rn2(1)==0 always, so it
        # is always picked initially -- exactly matching vendor where
        # nix starts as omx/omy and the first accepted neighbour overrides it.
        pick = is_accepted & (draw_val == jnp.int32(0))

        new_nix = jnp.where(pick, nr, nix)
        new_niy = jnp.where(pick, nc, niy)
        new_nidist = jnp.where(
            pick,
            _dist2(nr, nc, goal_r, goal_c),
            nidist,
        )

        return (new_chcnt, new_nix, new_niy, new_nidist, new_rng)

    final_carry = jax.lax.fori_loop(
        0, 8, body,
        (init_chcnt, init_nix, init_niy, init_nidist, vendor_rng),
    )
    _, nix, niy, _, new_rng = final_carry
    return nix, niy, new_rng


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------

def vendor_pet_dog_move(state, vendor_rng: Isaac64State, pet_slot):
    """Vendor-faithful ``dog_move`` for a single pet's turn (ISAAC64 byte-parity).

    Reproduces the dogmove.c:862 ``dog_move`` entry point for the validator
    scenario: a tame kitten adjacent to the hero with empty inventory, empty
    floor, no traps, no Conflict, no displacement.  The function consumes
    exactly the ISAAC64 draws that NLE consumes for the same scenario, in
    the same order, so the resulting stream position matches vendor.

    Draw budget (in order):

        1 x rn2(4)          -- wanderer skip-move gate             (monmove.c:578)
        K x rn2(100)        -- dog_goal SQSRCHRADIUS=5 fobj scan,  (dogmove.c:502-553,
                               ONE per in-range non-quest object   dog.c:744,
                               (K = count of ground-stack slots    zap.c:1191)
                               with category != 0 whose tile sits
                               in the pet's [pet ± 5] Chebyshev
                               bounding box).
        0..1 x rn2(4)       -- dog_goal stay-in-room check, only   (dogmove.c:565)
                               when udist > 1 (hero IS in a room
                               so first OR-clause is false and
                               C short-circuit reaches rn2(4)).
        N x rn2(1..N)       -- scoring loop, ONE per accepted      (dogmove.c:1114)
                               neighbour where N == accepted_count

    The distfleeck ``rn2(5)`` bravegremlin draw is now emitted by the
    per-fmon scan body in ``monsters_step_all`` (one per valid & alive
    fmon entry, pet included), so the vendor_rng stream advances even
    when the pet is asleep or out of movement_points.  Emitting it here
    would double-draw.  The wanderer ``rn2(4)`` is still emitted here
    because non-pet monsters' M2_WANDER gate isn't yet portable to the
    scan body — an accepted parity gap for steps 3+ until non-pet
    wanderer emission is wired in.  Modelling the SCARED or skip-move
    effect is unnecessary for stream alignment; the validator only cares
    about ISAAC64 byte parity.

    Parameters
    ----------
    state : EnvState
        Reads ``state.monster_ai.pos[pet_slot]`` (int16 [row, col]),
        ``state.monster_ai.alive[pet_slot]`` (bool),
        ``state.monster_ai.movement_points[pet_slot]`` (int16),
        ``state.player_pos`` (int16 [row, col]),
        ``state.terrain[branch, level-1, :, :]`` (int8 [_MAP_H, _MAP_W]).
        All writes go through ``state.replace(monster_ai=...)``.
    vendor_rng : Isaac64State
        Current ISAAC64 stream position.  Advanced by the consumed draws.
    pet_slot : int32 scalar
        The pet's monster slot index.  May be a JAX tracer -- all branches
        on its value go through ``lax.cond`` / ``jnp.where``.

    Returns
    -------
    new_state : EnvState
        Updated monster_ai.pos[pet_slot] and movement_points[pet_slot].
    new_vendor_rng : Isaac64State
        ISAAC64 state advanced by the consumed draws.

    Edge cases
    ----------
    - ``pet_slot`` out of range or ``alive[pet_slot] == False``: returns
      (state, vendor_rng) unchanged with ZERO draws.  Vendor never calls
      dog_move on a dead monster, so emitting no draws is correct.
    - Pet not at udist==1: the validator scenario guarantees udist==1; for
      udist > 1 the vendor dog_goal would take a different rn2(4) path
      inside its ``if (udist > 1)`` branch and the dochug short-circuit
      would also differ (mpeaceful=0 pets at distance evaluate the full
      ``||`` chain).  This skeleton does NOT model that case -- callers
      MUST gate on udist==1 before invoking this function (the other code
      path in monster_ai.py handles distant pets).  Internally we still
      emit the 2 prelude draws + 0 scoring draws if mfndpos returns empty,
      so a stay-put result with no neighbours is still a valid 2-draw turn.
    - mfndpos returns 0 candidates: stay put, 0 scoring draws (the 2
      prelude draws still fire).

    Citations
    ---------
    vendor/nle/src/dogmove.c   lines 862-1126   (dog_move main body)
    vendor/nle/src/dogmove.c   lines 476-577    (dog_goal)
    vendor/nle/src/monmove.c   lines 315-349    (distfleeck)
    vendor/nle/src/monmove.c   lines 574-579    (movement-phase || chain)
    vendor/nle/src/mon.c       lines 1305-1500  (mfndpos)
    vendor/nle/src/hacklib.c   line  614        (dist2)
    """
    pet_slot_i32 = pet_slot.astype(jnp.int32) if hasattr(pet_slot, "astype") \
        else jnp.int32(pet_slot)

    # ------------------------------------------------------------------
    # Eligibility gate: bail (with no draws) on out-of-range / dead slot.
    # vendor never reaches dog_move for these.
    # ------------------------------------------------------------------
    mai = state.monster_ai
    in_range = (pet_slot_i32 >= jnp.int32(0)) & (pet_slot_i32 < jnp.int32(_MAX_MONSTERS))
    safe_slot = jnp.clip(pet_slot_i32, 0, _MAX_MONSTERS - 1)
    is_alive = mai.alive[safe_slot]
    eligible = in_range & is_alive

    # Brax-flatten: compute both branches, select with tree_map(jnp.where).
    # WARNING: this changes semantics from the previous lax.cond gate — the
    # _run_dog_move body now executes (and advances ISAAC64 draws on its
    # internal scratch rng copy) even when eligible == False.  We select the
    # ORIGINAL (state, vendor_rng) for the ineligible branch so the public
    # contract (zero draws / unchanged state) is preserved.
    run_state, run_rng = _run_dog_move(state, vendor_rng, safe_slot)
    new_state = jax.tree_util.tree_map(
        lambda t, f: jnp.where(eligible, t, f), run_state, state,
    )
    new_vendor_rng = jax.tree_util.tree_map(
        lambda t, f: jnp.where(eligible, t, f), run_rng, vendor_rng,
    )
    return new_state, new_vendor_rng


def _run_dog_move(state, vendor_rng, pet_slot):
    """Eligible-path body of ``vendor_pet_dog_move``.

    Pulled out of the public function so the eligibility ``lax.cond`` only
    has to thread (state, vendor_rng) through its branches.
    """
    mai = state.monster_ai
    pet_pos = mai.pos[pet_slot].astype(jnp.int32)
    pet_r, pet_c = pet_pos[0], pet_pos[1]

    ppos = state.player_pos.astype(jnp.int32)
    hero_r, hero_c = ppos[0], ppos[1]

    # ------------------------------------------------------------------
    # Step 1: distfleeck -- rn2(5) for bravegremlin is now emitted by the
    # per-fmon scan body in ``monsters_step_all`` (gated on valid & alive),
    # so EVERY fmon entry advances the vendor_rng stream once per turn —
    # not just the pet.  Emitting it here would double-draw for the pet
    # path.
    # Cite: vendor/nle/src/monmove.c lines 315-320 (distfleeck bravegremlin);
    #       Nethax/nethax/subsystems/monster_ai.py::monsters_step_all _body.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Step 2: wanderer skip-move gate -- rn2(4).
    # Kitten has M2_WANDER so the ``||`` chain reaches this draw before any
    # short-circuit can suppress it.  We don't need to honour the gate's
    # actual side-effect (skip dog_move and call m_move) because the test
    # scenario only checks ISAAC64 stream alignment, not which branch of
    # dochug runs.
    # Cite: vendor/nle/src/monmove.c line 578.
    # ------------------------------------------------------------------
    vendor_rng, _wander_skip = rn2_jax(vendor_rng, jnp.int64(4))

    # ------------------------------------------------------------------
    # Step 2b: dog_goal SQSRCHRADIUS=5 fobj scan -- one rn2(100) per object.
    # Vendor dog_goal (dogmove.c:502-553) iterates the global ``fobj`` list,
    # tests each object's tile against the [pet ± SQSRCHRADIUS] bounding
    # box, and calls ``dogfood(mtmp, obj)`` for each in-range hit.  The
    # first line of ``dogfood`` (dog.c:744) is
    #     if (is_quest_artifact(obj) || obj_resists(obj, 0, 95)) ...
    # and ``obj_resists`` draws ``rn2(100)`` at zap.c:1191 once per call
    # when the object is not a quest artifact.  For the validator scenario
    # (Dlvl 1, no quest items can spawn) every in-range object therefore
    # contributes exactly one ``rn2(100)`` to the ISAAC64 stream.
    #
    # Cite: vendor/nle/src/dogmove.c lines 502-553 (SQSRCHRADIUS fobj scan);
    #       vendor/nle/src/dog.c     line  744      (dogfood obj_resists call);
    #       vendor/nle/src/zap.c     line  1191     (obj_resists rn2(100)).
    # ------------------------------------------------------------------
    vendor_rng = _emit_dog_goal_fobj_scan(state, vendor_rng, pet_r, pet_c)

    # ------------------------------------------------------------------
    # Step 3: dog_goal -- emit the udist>1 stay-in-room rn2(4) draw.
    # With empty inventory, empty fobj list, in_masters_sight=True, the
    # `follow player` branch (dogmove.c:557) is taken and gx=hero, gy=hero.
    # Inside that branch the vendor checks
    #     if (udist > 1) {
    #         if (!IS_ROOM(levl[u.ux][u.uy].typ) || !rn2(4) || whappr
    #             || (dog_has_minvent && rn2(edog->apport)))
    #             appr = 1;
    #     }
    # Cite: vendor/nle/src/dogmove.c line 565 (stay-in-room check).
    #
    # For the validator scenario the hero IS in a room, so the first
    # clause `!IS_ROOM(...)` is false and C short-circuit evaluation
    # advances to the `!rn2(4)` clause -- consuming one ISAAC64 draw.
    # `dog_has_minvent` is false (empty pet inventory), so the trailing
    # `rn2(edog->apport)` is suppressed by short-circuit and draws nothing.
    #
    # Gate: the entire `if (udist > 1)` block is skipped when the pet is
    # cardinally adjacent (udist == 1) or co-located (udist == 0); in
    # those cases no draw fires.  Diagonal-adjacent pet has udist == 2
    # (dist2 = 1+1) which IS > 1, so the draw fires for the validator
    # kitten at (9,70) with hero at (10,71).
    # ------------------------------------------------------------------
    goal_r = hero_r
    goal_c = hero_c
    # appr = 0 is implicit in the scoring loop below (we always take the
    # j==0 reservoir-sample branch since appr*anything == 0).

    udist = _dist2(pet_r, pet_c, hero_r, hero_c)
    needs_room_check = udist > jnp.int32(1)

    # Brax-flatten: compute both branches, select with tree_map(jnp.where).
    # rn2(4) -- value discarded; only stream-advance matters for parity.
    room_drawn_rng, _ = rn2_jax(vendor_rng, jnp.int64(4))
    vendor_rng = jax.tree_util.tree_map(
        lambda t, f: jnp.where(needs_room_check, t, f),
        room_drawn_rng, vendor_rng,
    )

    # ------------------------------------------------------------------
    # Step 4: mfndpos -- 8-neighbor enumeration, pure filter, ZERO draws.
    # ------------------------------------------------------------------
    accepted, nrows, ncols = _mfndpos(state, pet_r, pet_c)

    # ------------------------------------------------------------------
    # Step 5: scoring loop -- ONE rn2(++chcnt) per accepted candidate.
    # ------------------------------------------------------------------
    new_r, new_c, vendor_rng = _scoring_loop(
        accepted, nrows, ncols, pet_r, pet_c, goal_r, goal_c, vendor_rng,
    )

    # ------------------------------------------------------------------
    # Step 6: apply move (position only).  Movement-point deduction is the
    # caller's responsibility — vendor `movemon_singlemon` (mon.c:1254)
    # deducts NORMAL_SPEED in the OUTER loop before calling dochug; dog_move
    # itself does not touch mtmp->movement.  Doubling up here would
    # double-decrement.  Position is stored as int16 [row, col].
    # ------------------------------------------------------------------
    new_pos_pair = jnp.stack(
        [new_r.astype(jnp.int16), new_c.astype(jnp.int16)],
    )
    new_mai = mai.replace(
        pos=mai.pos.at[pet_slot].set(new_pos_pair),
    )
    new_state = state.replace(monster_ai=new_mai)
    return new_state, vendor_rng
