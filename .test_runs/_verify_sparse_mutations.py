"""PHASE 2 verification: sparse ground_items MUTATION primitives == dense.

For every mutation type we apply BOTH:
  (a) the DENSE write on a dense ground_items  -> dense_mutated
  (b) sparse = dense_to_sparse(orig, K); sparse_mutated = <sparse primitive>;
      recon = sparse_to_dense(sparse_mutated)
and assert recon == dense_mutated field-for-field on every category!=0 cell of
the FULL [B,L,H,W,S] grid (union mask catches missing AND extra cells).  Each
mutation is exercised under jax.jit AND under a B=4 jax.vmap.

Ground truth per type:
  * clear_slot / pickup : dense ``category.at[b,lv,r,c,slot].set(0)`` — the
    exact line used by timer_queue._clear_ground_slot, monster_ai clears, and
    inventory.pickup (line 952).
  * drop                : the REAL ``inventory.drop`` engine function.
  * set_cell            : direct dense stack overwrite ``.at[b,lv,r,c,:].set``.
  * overflow            : survivor-set equality vs canonical dense_to_sparse.

Run:
  JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python \
      .test_runs/_verify_sparse_mutations.py
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

import gym, gymnasium, gymnasium.spaces.dict as _g  # noqa: E402
_g.Space = (gymnasium.spaces.Space, gym.spaces.Space)

from Nethax.minihax.minihax_env import MinihaxEnv
from Nethax.nethax.subsystems.inventory import (
    Item, MAX_GROUND_STACK, make_item, make_empty_item, _stack_items,
    drop as dense_drop,
)
from Nethax.nethax.subsystems.ground_items_sparse import (
    dense_to_sparse, sparse_to_dense,
    sparse_clear_slot, sparse_pickup, sparse_set_cell, sparse_drop,
    sparse_read_tile,
)
from Nethax.nethax.dungeon.branches import (
    N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W,
)

S = MAX_GROUND_STACK
FIELDS = [f for f in make_empty_item().__dict__.keys()]


def _tile_stack(items):
    """Build an Item whose fields are [S] from a list of <=S Items."""
    padded = list(items) + [make_empty_item()] * (S - len(items))
    return _stack_items(padded[:S])


def cmp_dense(recon, dense_mut):
    """True iff recon == dense_mut on every cell occupied in EITHER (all fields)."""
    occ = (np.asarray(recon.category) != 0) | (np.asarray(dense_mut.category) != 0)
    for f in FIELDS:
        rv, dv = np.asarray(getattr(recon, f)), np.asarray(getattr(dense_mut, f))
        if not np.array_equal(rv[occ], dv[occ]):
            bad = np.argwhere((rv != dv) & occ)
            return False, f, (bad[0].tolist() if len(bad) else None)
    return True, None, None


def set_tile(gi, b, lv, r, c, stack):
    """Dense reference: overwrite the whole S-deep stack at (r,c)."""
    d = {}
    for f in FIELDS:
        d[f] = getattr(gi, f).at[b, lv, r, c, :].set(getattr(stack, f))
    return gi.replace(**d)


def clear_slot_dense(gi, b, lv, r, c, slot):
    return gi.replace(category=gi.category.at[b, lv, r, c, slot].set(jnp.int8(0)))


# ---------------------------------------------------------------------------
# Build a base dense ground_items from a real reset, then inject edge tiles.
# ---------------------------------------------------------------------------
def base_state():
    e = MinihaxEnv("MiniHack-Room-5x5-v0")
    s0, _ = e.reset(jax.random.key(0))
    s = s0
    for t in range(2):
        s, _, _, _ = e.step(s, 0, jax.random.key(100 + t))
    return e, s


def np_gi(gi):
    return {f: np.asarray(getattr(gi, f)).copy() for f in FIELDS}


def gi_from(d):
    return Item(**{f: jnp.asarray(d[f]) for f in FIELDS})


def put(d, b, lv, r, c, slot, it):
    for f in FIELDS:
        d[f][b, lv, r, c, slot] = np.asarray(getattr(it, f))


def main():
    print("=== PHASE 2 sparse mutation verification ===", flush=True)
    print(f"dims B={N_BRANCHES} L={MAX_LEVELS_PER_BRANCH} H={MAP_H} W={MAP_W} "
          f"S={S}", flush=True)
    e, st = base_state()
    b = int(st.dungeon.current_branch)
    lv = int(st.dungeon.current_level) - 1
    K = 64
    results = {}

    # Distinct sample items (identity differs so no accidental merges).
    A = make_item(category=2, type_id=50, quantity=3, weight=30, enchantment=1,
                  buc_status=2)
    Bt = make_item(category=3, type_id=77, quantity=1, weight=15, buc_status=3)
    Cd = make_item(category=4, type_id=120, quantity=5, weight=5, buc_status=1)
    Dd = make_item(category=5, type_id=200, quantity=2, weight=40)

    # tiles far apart on the level
    T_empty = (3, 5)
    T_one = (4, 10)
    T_multi = (6, 20)
    T_full = (8, 30)

    d0 = np_gi(st.ground_items)
    # T_one: single item at slot0
    put(d0, b, lv, *T_one, 0, A)
    # T_multi: 3 items slots 0,1,2
    put(d0, b, lv, *T_multi, 0, A)
    put(d0, b, lv, *T_multi, 1, Bt)
    put(d0, b, lv, *T_multi, 2, Cd)
    # T_full: full 8-deep stack (all distinct)
    for s in range(S):
        put(d0, b, lv, *T_full, s,
            make_item(category=(s % 6) + 1, type_id=300 + s, quantity=s + 1,
                      weight=(s + 1) * 2, buc_status=s % 4))
    gi0 = gi_from(d0)

    # ================= 1) clear_slot =================
    def clr_pipeline(gi, r, c, slot):
        sp = dense_to_sparse(gi, K)
        sp = sparse_clear_slot(sp, b, lv, r, c, slot)
        return sparse_to_dense(sp)
    clr_jit = jax.jit(clr_pipeline, static_argnums=())
    cases = [(*T_empty, 0), (*T_one, 0), (*T_multi, 1), (*T_full, 0), (*T_full, 7)]
    ok = True
    for (r, c, sl) in cases:
        dm = clear_slot_dense(gi0, b, lv, r, c, sl)
        recon = clr_jit(gi0, jnp.int32(r), jnp.int32(c), jnp.int32(sl))
        good, bad, at = cmp_dense(recon, dm)
        if not good:
            ok = False
            print(f"  [clear_slot JIT] FAIL tile=({r},{c},{sl}) field={bad} at={at}")
    results["clear_slot(jit)"] = ok

    # vmap B=4 over the 4 non-empty cases
    vcases = [(*T_one, 0), (*T_multi, 1), (*T_full, 0), (*T_full, 7)]
    rr = jnp.array([x[0] for x in vcases]); cc = jnp.array([x[1] for x in vcases])
    ss = jnp.array([x[2] for x in vcases])
    gib = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (4,) + a.shape), gi0)
    recon_b = jax.vmap(clr_pipeline, in_axes=(0, 0, 0, 0))(gib, rr, cc, ss)
    okv = True
    for i, (r, c, sl) in enumerate(vcases):
        dm = clear_slot_dense(gi0, b, lv, r, c, sl)
        recon_i = jax.tree_util.tree_map(lambda a: a[i], recon_b)
        good, bad, at = cmp_dense(recon_i, dm)
        if not good:
            okv = False
            print(f"  [clear_slot VMAP] FAIL tile=({r},{c},{sl}) field={bad} at={at}")
    results["clear_slot(vmap)"] = okv

    # ================= 2) pickup (= clear slot0) =================
    def pk_pipeline(gi, r, c):
        sp = dense_to_sparse(gi, K)
        sp = sparse_pickup(sp, b, lv, r, c)
        return sparse_to_dense(sp)
    pk_jit = jax.jit(pk_pipeline)
    ok = True
    for (r, c) in [T_one, T_multi, T_full, T_empty]:
        dm = clear_slot_dense(gi0, b, lv, r, c, 0)
        recon = pk_jit(gi0, jnp.int32(r), jnp.int32(c))
        good, bad, at = cmp_dense(recon, dm)
        if not good:
            ok = False
            print(f"  [pickup JIT] FAIL tile=({r},{c}) field={bad} at={at}")
    results["pickup(jit)"] = ok
    pcases = [T_one, T_multi, T_full, T_empty]
    rr = jnp.array([x[0] for x in pcases]); cc = jnp.array([x[1] for x in pcases])
    recon_b = jax.vmap(pk_pipeline, in_axes=(0, 0, 0))(gib, rr, cc)
    okv = True
    for i, (r, c) in enumerate(pcases):
        dm = clear_slot_dense(gi0, b, lv, r, c, 0)
        recon_i = jax.tree_util.tree_map(lambda a: a[i], recon_b)
        good, bad, at = cmp_dense(recon_i, dm)
        if not good:
            okv = False
            print(f"  [pickup VMAP] FAIL tile=({r},{c}) field={bad} at={at}")
    results["pickup(vmap)"] = okv

    # ================= 3) set_cell =================
    # payload stacks (fields [S])
    grow = _tile_stack([A, Bt, Cd, Dd, make_item(category=6, type_id=9, quantity=1)])
    shrink = _tile_stack([Dd, Cd])
    full8 = _tile_stack([make_item(category=(i % 6) + 1, type_id=400 + i,
                                   quantity=i + 1, weight=i, buc_status=i % 4)
                         for i in range(S)])
    empty_stk = _tile_stack([])
    set_cases = [
        (T_empty, grow, "empty->5"),
        (T_multi, shrink, "3->2 shrink"),
        (T_one, full8, "1->8 grow"),
        (T_full, empty_stk, "8->0 clear-all"),
    ]

    def sc_pipeline(gi, r, c, stack):
        sp = dense_to_sparse(gi, K)
        sp = sparse_set_cell(sp, b, lv, r, c, stack)
        return sparse_to_dense(sp)
    sc_jit = jax.jit(sc_pipeline)
    ok = True
    for (r, c), stack, name in set_cases:
        dm = set_tile(gi0, b, lv, r, c, stack)
        recon = sc_jit(gi0, jnp.int32(r), jnp.int32(c), stack)
        good, bad, at = cmp_dense(recon, dm)
        if not good:
            ok = False
            print(f"  [set_cell JIT] FAIL {name} tile=({r},{c}) field={bad} at={at}")
    results["set_cell(jit)"] = ok
    rr = jnp.array([x[0][0] for x in set_cases])
    cc = jnp.array([x[0][1] for x in set_cases])
    stk_b = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs),
                                   *[s for _, s, _ in set_cases])
    recon_b = jax.vmap(sc_pipeline, in_axes=(0, 0, 0, 0))(gib, rr, cc, stk_b)
    okv = True
    for i, (((r, c), stack, name)) in enumerate(set_cases):
        dm = set_tile(gi0, b, lv, r, c, stack)
        recon_i = jax.tree_util.tree_map(lambda a: a[i], recon_b)
        good, bad, at = cmp_dense(recon_i, dm)
        if not good:
            okv = False
            print(f"  [set_cell VMAP] FAIL {name} tile=({r},{c}) field={bad} at={at}")
    results["set_cell(vmap)"] = okv

    # ================= 4) drop (real engine ground-truth) =================
    # Build states where an inventory item is dropped at player_pos onto a
    # controlled ground tile; compare engine new_gi vs sparse_drop.
    def make_drop_state(base, r, c, ground_stack, inv_item):
        d = np_gi(base.ground_items)
        for s in range(S):
            put(d, b, lv, r, c, s, jax.tree_util.tree_map(
                lambda a: a[s], ground_stack))
        gi = gi_from(d)
        # inventory: put inv_item at slot 0, clear the rest
        inv = base.inventory
        items = inv.items
        newitems = {}
        for f in FIELDS:
            arr = getattr(items, f)
            newitems[f] = arr.at[0].set(getattr(inv_item, f))
        items = items.replace(**newitems)
        stt = base.replace(
            inventory=inv.replace(items=items, wielded=jnp.int8(-1),
                                  welded=jnp.bool_(False)),
            player_pos=jnp.array([r, c], dtype=base.player_pos.dtype),
        )
        return stt, gi

    # scenario a: empty tile -> fresh at slot0
    # scenario b: slot0 identical to inv item -> merge qty/wt
    # scenario c: slot0 occupied non-match, slot1 empty -> fresh slot1
    # scenario d: full 8-deep no match -> no-op
    inv_item = make_item(category=2, type_id=50, quantity=4, weight=8,
                         buc_status=2, enchantment=1)
    match_ground = make_item(category=2, type_id=50, quantity=3, weight=6,
                             buc_status=2, enchantment=1)  # same identity
    nonmatch = make_item(category=7, type_id=999, quantity=1, weight=1)

    drop_scen = {
        "empty->fresh": _tile_stack([]),
        "merge": _tile_stack([match_ground]),
        "nonmatch->fresh-slot1": _tile_stack([nonmatch]),
        "full-noop": _tile_stack([make_item(category=(i % 5) + 1, type_id=600 + i,
                                            quantity=i + 1, weight=i)
                                  for i in range(S)]),
    }

    def drop_pipeline(gi, r, c, item):
        sp = dense_to_sparse(gi, K)
        sp = sparse_drop(sp, b, lv, r, c, item)
        return sparse_to_dense(sp)
    drop_jit = jax.jit(drop_pipeline)

    R, C = 5, 12
    ok = True
    dstates = []
    for name, gstack in drop_scen.items():
        stt, gi = make_drop_state(st, R, C, gstack, inv_item)
        _, dm = dense_drop(stt, jax.random.key(3), gi, b, lv, 0)
        recon = drop_jit(gi, jnp.int32(R), jnp.int32(C), inv_item)
        good, bad, at = cmp_dense(recon, dm)
        if not good:
            ok = False
            print(f"  [drop JIT] FAIL {name} field={bad} at={at}")
        dstates.append((name, gi, dm))
    results["drop(jit)"] = ok

    # vmap over the 4 drop scenarios (same inv_item, same tile)
    gi_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs),
                                      *[gi for _, gi, _ in dstates])
    rr = jnp.full((4,), R, jnp.int32); cc = jnp.full((4,), C, jnp.int32)
    item_b = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (4,) + a.shape),
                                    inv_item)
    recon_b = jax.vmap(drop_pipeline, in_axes=(0, 0, 0, 0))(gi_stack, rr, cc, item_b)
    okv = True
    for i, (name, gi, dm) in enumerate(dstates):
        recon_i = jax.tree_util.tree_map(lambda a: a[i], recon_b)
        good, bad, at = cmp_dense(recon_i, dm)
        if not good:
            okv = False
            print(f"  [drop VMAP] FAIL {name} field={bad} at={at}")
    results["drop(vmap)"] = okv

    # ================= 5) K-overflow survivor-set parity =================
    # Small K; fill a level with >K occupied cells, then insert a LOWER-flat
    # item.  Correct survivor set == dense_to_sparse(dense_mutated, K).
    Ksmall = 8
    from Nethax.nethax.subsystems.inventory import _empty_ground_items_array
    gi_e = _empty_ground_items_array(N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W)
    de = np_gi(gi_e)
    # place K occupied cells at HIGH flat indices (large rows) so a new low-row
    # insert must evict the current max-flat survivor.
    occ_cells = [(15 + (i // MAP_W), (i % MAP_W)) for i in range(Ksmall)]
    for i, (r, c) in enumerate(occ_cells):
        put(de, b, lv, r, c, 0, make_item(category=(i % 6) + 1, type_id=700 + i,
                                          quantity=i + 1, weight=i))
    gi_ov = gi_from(de)
    # insert a NEW low-flat cell (row 1) via set_cell (one occupied slot)
    new_tile = (1, 2)
    new_stack = _tile_stack([make_item(category=3, type_id=42, quantity=9,
                                       weight=3)])
    dm_ov = set_tile(gi_ov, b, lv, *new_tile, new_stack)  # dense: all cells

    def ov_pipeline(gi):
        sp = dense_to_sparse(gi, Ksmall)
        sp = sparse_set_cell(sp, b, lv, new_tile[0], new_tile[1], new_stack)
        return sparse_to_dense(sp)
    recon_ov = jax.jit(ov_pipeline)(gi_ov)
    # canonical survivor image = sparse_to_dense(dense_to_sparse(dm_ov, K))
    canon = sparse_to_dense(dense_to_sparse(dm_ov, Ksmall))
    good, bad, at = cmp_dense(recon_ov, canon)
    results["overflow(jit)"] = good
    if not good:
        print(f"  [overflow JIT] FAIL field={bad} at={at}")
    # confirm eviction actually happened (survivors < total occupied)
    n_after = int((np.asarray(recon_ov.category) != 0).sum())
    n_dense = int((np.asarray(dm_ov.category) != 0).sum())
    print(f"  [overflow] K={Ksmall} dense_occupied={n_dense} "
          f"sparse_survivors={n_after} (eviction {'YES' if n_after < n_dense else 'NO'})")

    # ================= summary =================
    print("\n[RESULTS]", flush=True)
    allok = True
    for k, v in results.items():
        print(f"  {k:24s} {'PASS' if v else 'FAIL'}", flush=True)
        allok = allok and v
    print(f"\n[SUMMARY] {'ALL PASS' if allok else 'FAILURES PRESENT'}", flush=True)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
