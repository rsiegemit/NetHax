"""Vectorized (simultaneous-move) monster turn loop — the fast GPU training path.

Replaces the serial 400-iteration ``lax.scan`` monster turn loop.  That scan runs
a ~68k-equation body 400 times = ~27.3M serial op-executions per step, which on a
single-thread CPU is ~162 s/step and on GPU is a >8 h serial micro-kernel chain
(see .test_runs/BENCH_RESULTS.md / project_gpu_exec_wall_2026_06_22).  Vectorizing
runs the per-monster body ONCE as wide ``[400, ...]`` batched ops (the nested
BFS/LoS/attack loops become wide instead of 400-deep-serial), which is the shape a
GPU actually parallelizes.

SEMANTIC CHANGE: monsters act SIMULTANEOUSLY — every monster reads the FROZEN
start-of-turn state — instead of vendor-sequential (where monster B observes
monster A's just-completed move).  This is standard for batched JAX RL envs and is
acceptable for training, but it is NOT byte-parity, so this path is used ONLY in
the non-vendor-RNG (Threefry) branch.  The vendor/byte-parity loop stays serial and
untouched, so the 48/48 multi-seed parity is unaffected by construction.

Merge rules — every leaf is merged by an explicit rule (full-fidelity):
  * monster_ai per-slot ``[N, ...]`` leaves : DIAGONAL — slot i takes the value
    monster i wrote for its OWN slot.  Gathered inside the vmap (``leaf[idx]``) so
    the result is ``[N, per-slot]`` with no ``[N, N]`` materialization.  Exact for
    any dtype.
  * big shared arrays (> _BIG_LEAF_ELEMS elements) : KEPT FROZEN from the
    start-of-turn state.  Monster turns do not mutate multi-level terrain /
    explored / ground_items (verified: a move writes only monster_ai).  Excluded
    from the vmap output (placeholder) so no ``[N, terrain]`` blow-up / OOM.
    LIMITATION: a rare monster-induced big-array change (spell digging, corpse
    drop) is dropped in vec mode.
  * other shared bool leaves   : OR across monsters (tile reveals — monsters only
    set True).
  * other shared numeric leaves : additive sum-of-diffs (player hp, score, kill
    counters; a single changer is exact, multiple additive changers compose).
  * non-numeric/non-bool leaves (e.g. PRNGKey) : KEPT FROZEN.

After merging, same-tile move collisions are resolved by slot priority
(:func:`_resolve_collisions`) since simultaneous movers can pick the same tile.
"""
import jax
import jax.numpy as jnp
from jax import tree_util as jtu

# Leaves with more than this many elements are "big" shared arrays (multi-level
# terrain/explored/last_seen_terrain/ground_items) — kept frozen, never replicated
# across the N-way vmap.  Small per-level arrays (visible [H,W] ~1.6k) stay merged.
_BIG_LEAF_ELEMS = 100_000


def _under_monster_ai(path):
    return any(getattr(p, "name", None) == "monster_ai" for p in path)


def _is_per_slot(path, leaf, n):
    return _under_monster_ai(path) and leaf.ndim >= 1 and leaf.shape[0] == n


def vectorized_monster_turns(state, monster_turn, indices, turn_keys, can_act):
    """Simultaneous-move replacement for the serial monster turn scan.

    Args mirror the scan inputs: ``indices`` [N] slot ids, ``turn_keys`` [N] RNG
    keys, ``can_act`` [N] bool gate.  ``monster_turn(state, key, idx)`` is the
    existing per-slot turn function (reused verbatim).
    """
    n = indices.shape[0]
    s0 = state

    def _one(idx, key, may):
        out = jax.lax.cond(
            may, lambda: monster_turn(s0, key, idx), lambda: s0)

        def _compact(path, leaf):
            if _is_per_slot(path, leaf, n):
                return leaf[idx]                    # this monster's own slot
            if leaf.size > _BIG_LEAF_ELEMS:
                return jnp.zeros((), leaf.dtype)    # drop big -> placeholder
            return leaf                             # small shared -> keep
        return jtu.tree_map_with_path(_compact, out)

    batched = jax.vmap(_one)(indices, turn_keys, can_act)

    def _merge(path, s0_leaf, b_leaf):
        if _is_per_slot(path, s0_leaf, n):
            # b_leaf is [N, per-slot] == original full array; row i = monster i's
            # own-slot write. This IS the diagonal merge.
            return b_leaf.astype(s0_leaf.dtype)
        if s0_leaf.size > _BIG_LEAF_ELEMS:
            return s0_leaf                          # frozen big array
        if s0_leaf.dtype == jnp.bool_:
            return jnp.any(b_leaf, axis=0)          # reveal union
        if jnp.issubdtype(s0_leaf.dtype, jnp.number):
            delta = jnp.sum(b_leaf - s0_leaf[None], axis=0)
            return (s0_leaf + delta).astype(s0_leaf.dtype)
        return s0_leaf                              # PRNGKey / other -> frozen

    merged = jtu.tree_map_with_path(_merge, s0, batched)
    return _resolve_collisions(s0, merged, can_act)


def _resolve_collisions(s0, merged, can_act):
    """Revert simultaneous movers that landed on an already-claimed tile.

    Under simultaneous moves two monsters can pick the same destination, which
    vendor's sequential occupancy check forbids.  Resolution (one pass): a monster
    yields (reverts to its start-of-turn tile) if a LOWER-index alive monster
    claims the same final tile.  Lowest slot wins; vendor walks fmon in order so
    slot-priority is a faithful-enough simultaneous analogue.
    """
    mai0 = s0.monster_ai
    mai = merged.monster_ai
    alive = mai.alive
    moved = jnp.any(mai.pos != mai0.pos, axis=1) & can_act & alive
    pos = mai.pos.astype(jnp.int32)
    # pairwise same-tile (only meaningful between alive monsters)
    same = jnp.all(pos[:, None, :] == pos[None, :, :], axis=2)  # [N,N]
    alive_pair = alive[None, :] & alive[:, None]
    lower = jnp.tril(jnp.ones_like(same), k=-1).astype(bool)    # j < i
    conflict = jnp.any(same & alive_pair & lower, axis=1) & moved
    new_pos = jnp.where(conflict[:, None], mai0.pos, mai.pos)
    return merged.replace(monster_ai=mai.replace(pos=new_pos))
