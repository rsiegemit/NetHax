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

# Shared (non-monster_ai) leaves with more than this many elements are kept FROZEN
# from the start-of-turn state — never replicated across the N-way vmap.  This is
# both a correctness fact and the OOM fix: the verified monster_turn write surface
# (.test_runs/_monster_attack_surface.py) is monster_ai per-slot + player_hp +
# messages ONLY — every larger shared leaf (visible [H,W], status arrays,
# multi-level terrain, ground_items, ...) is untouched by a turn, so replicating
# it [B, N, ...] under the batch+slot vmaps was pure waste (e.g. visible became
# [B, 400, 21, 79] -> A100 OOM at B>=64).  Threshold 64 keeps scalars/tiny vectors
# (player_hp, gold, score) merged and freezes everything bigger.
_SHARED_MERGE_MAX_ELEMS = 64

# Monsters processed per vmap chunk (jax.lax.map batch_size).  Caps peak per-
# monster activation memory under the env x monster double-vmap.  Override via
# NETHAX_VEC_CHUNK; smaller = less memory, more (but short, GPU-fast) scan steps.
import os as _os
_VEC_CHUNK = int(_os.environ.get("NETHAX_VEC_CHUNK", "32"))


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
            if leaf.size > _SHARED_MERGE_MAX_ELEMS:
                return jnp.zeros((), leaf.dtype)    # frozen shared -> placeholder
            return leaf                             # tiny shared (hp,...) -> keep
        return jtu.tree_map_with_path(_compact, out)

    # Chunked vmap (jax.lax.map with batch_size) instead of a flat vmap over all
    # N monsters: vmap K monsters per chunk, scan across ceil(N/K) chunks.  The
    # OUTPUT is identical [N, compact], but peak ACTIVATION memory inside the
    # heavy monster_turn body (per-monster BFS dist-field / LoS -> [B,N,grid]
    # under the env x monster vmaps) is bounded to K, not N.  Flat vmap-over-400
    # OOM'd the A100 at B>=64 (job 24263295); chunking is the fix.  B=1 unaffected.
    batched = jax.lax.map(
        lambda a: _one(a[0], a[1], a[2]),
        (indices, turn_keys, can_act),
        batch_size=_VEC_CHUNK,
    )

    def _merge(path, s0_leaf, b_leaf):
        if _is_per_slot(path, s0_leaf, n):
            # b_leaf is [N, per-slot] == original full array; row i = monster i's
            # own-slot write. This IS the diagonal merge.
            return b_leaf.astype(s0_leaf.dtype)
        if s0_leaf.size > _SHARED_MERGE_MAX_ELEMS:
            return s0_leaf                          # frozen shared leaf
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


def vectorized_mattackm_strikes(state, mattackm_one, indices, mhit_keys,
                                conflict_active):
    """Simultaneous-move replacement for the serial monster-vs-monster strike
    scan (the 2744-eqn-body x400 loop = the next-biggest serial chain after the
    turn loop).

    ``mattackm_one(state, atk_slot, key, conflict_active)`` is the existing
    per-attacker strike body (reused verbatim).  Each attacker picks its adjacent
    different-faction defender and applies HP damage to that DEFENDER's slot.

    Write surface (verified, .test_runs/_mattackm_write_surface.py): a strike
    writes ONLY ``monster_ai.hp`` at the defender slot (death/cleanup is derived
    from HP downstream).  So the merge is a clean SCATTER-SUM of damage: every
    attacker runs against the FROZEN start state, and total damage to each
    defender is summed.  Multiple attackers hitting one defender compose
    additively (vs vendor-sequential, where the second sees reduced HP — the
    accepted simultaneous-move approximation, training path only).
    """
    s0 = state
    hp0 = s0.monster_ai.hp.astype(jnp.int32)
    n = indices.shape[0]

    def _one(atk, key):
        out = mattackm_one(s0, atk, key, conflict_active)
        delta = hp0 - out.monster_ai.hp.astype(jnp.int32)   # [N], nonzero @defender
        j = jnp.argmax(jnp.abs(delta)).astype(jnp.int32)    # the struck defender
        return j, delta[j]                                  # two scalars — compact

    js, dmgs = jax.lax.map(                                  # chunked vmap (memory)
        lambda a: _one(a[0], a[1]), (indices, mhit_keys),
        batch_size=_VEC_CHUNK)                               # [N], [N] (not [N,N])
    total_dmg = jnp.zeros((n,), jnp.int32).at[js].add(dmgs)  # scatter-sum damage
    new_hp = (hp0 - total_dmg).astype(s0.monster_ai.hp.dtype)
    return s0.replace(monster_ai=s0.monster_ai.replace(hp=new_hp))
