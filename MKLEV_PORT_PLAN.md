# MKLEV_PORT_PLAN.md — Vendor-Exact Dungeon-Gen Port to JAX

## 0. Executive Summary

Player-position divergence (NLE `blstats[0]=15` vs Nethax `57`) is downstream of two structural mismatches in our ISAAC64 consume order vs vendor `mklev.c`:

1. Vendor room-placement loop is `while (nroom < MAXNROFROOMS && rnd_rect()) create_room(...)` (mklev.c:229). `rnd_rect()` itself draws `rn2(rect_cnt)` (rect.c:91) — room-count is implicit in RNG-driven rectangle selection, not pre-drawn. `rnd_rect()` is called again inside `create_room` (sp_lev.c:1175), and `split_rects` (rect.c:161) mutates the rect pool state-dependently.

2. Per-room draw order inside `create_room` is `lit_A → lit_B → dx → dy → x_pos → y_pos` (sp_lev.c:1154, 1188-1202), interleaved with rejection retries, not pre-batched.

Current code (a) pre-draws `rn2(5)` and `rn2(mc_upper)` in `env.py:163-167`, (b) batches all xywh draws via `lax.scan` before any lit draw, (c) skips two `rn2(nroom)` stair-room draws at mklev.c:710/715.

Fix: re-architect room placement as `lax.fori_loop` with `Isaac64State + NhRect[51]` carry, plus JAX port of `rect.c` (init/get/rnd/split/remove/add_rect) — state-pure, fixed-shape.

## 1. Per-Function Port Targets

### 1.1 `makedungeon()` — dungeon.c:714 `init_dungeons`

Already covered by `Nethax/nethax/dungeon/branches.py::sample_branch_table`. No new draws for seed-0 path. Vmappable.

### 1.2 `mklev()` — mklev.c:990-1036

| # | Vendor call | Cite | Meaning |
|---|---|---|---|
| 1 | `reseed_random(rn2)` | mklev.c:996 | Reseed CORE — no draw, replaces state |
| 2 | `reseed_random(rn2_on_display_rng)` | mklev.c:997 | Reseed display RNG (separate stream) |
| 3 | `getbones()` | mklev.c:1000 | No RNG (file IO) |
| — | `makelevel()` | mklev.c:1004 | All RNG inside |
| 4 | `reseed_random(rn2)` | mklev.c:1034 | Reseed at exit |

Sequential; trivial.

### 1.3 `makelevel()` — mklev.c:652-886

ISAAC64 consume order on Main Dlvl 1 path:

| # | Vendor call | Cite | Notes |
|---|---|---|---|
| 1 | `rn2(5)` gate | mklev.c:693 | `&&` LHS fires (verify) |
| 2..N | `makerooms()` — see §1.4 | mklev.c:706 | Variable count |
| N+1 | `rn2(nroom)` down-stair room | mklev.c:710 | **MISSING in Nethax** |
| N+2 | `rn2(nroom - 1)` up-stair room | mklev.c:715 | **MISSING in Nethax** |
| N+3 | `somex(croom)` down-stair x | mklev.c:712 | mkroom.c |
| N+4 | `somey(croom)` down-stair y | mklev.c:712 | |
| N+5,6 | `somex/somey` up-stair, `do{}while(occupied)` | mklev.c:723-725 | **MISSING** (deterministic centre) |
| N+7 | `makecorridors()` — see §1.5 | mklev.c:734 | |
| N+8 | `make_niches()` — see §1.6 | mklev.c:735 | |
| N+9 | `do_vault()` + `rn2(3)` for vtele | mklev.c:738-762 | If vault placed |
| N+10 | `rn2(u_depth)` SHOPBASE gate | mklev.c:770 | Compile-time false on Dlvl 1. **NO DRAW.** Verify. |
| N+11..N+M | `fill_ordinary_rooms` — see §1.7 | mklev.c:803-885 | |

All sequential. Inner room loops `lax.fori_loop` with Isaac64State carry.

### 1.4 `makerooms()` — mklev.c:222-241

Per-iteration ISAAC64:

| Step | Draw | Cite | Notes |
|---|---|---|---|
| A | `rn2(rect_cnt)` in `rnd_rect()` | rect.c:91 | Every loop test |
| B | `rn2(2)` vault attempt | mklev.c:230 | Only when `nroom >= 6 && !tried_vault` |
| C | `create_vault()` lit_A, lit_B | sp_lev.c:1185-1186 | Vault path |
| D | `create_room(...)` | sp_lev.c:1126 | See §1.4.1 |

#### 1.4.1 `create_room()` — sp_lev.c:1126-1292

| # | Draw | Cite | When |
|---|---|---|---|
| C1 | `rnd(1 + abs(depth))` (lit_A) | sp_lev.c:1154 | Always (rlit==-1) |
| C2 | `rn2(77)` (lit_B) | sp_lev.c:1154 | `&&` short-circuit; on Dlvl 1 always fires |
| D1 | `rnd_rect()` → `rn2(rect_cnt)` | sp_lev.c:1175 → rect.c:91 | Per do-while attempt (≤100) |
| D2 | `rn2((hx-lx > 28) ? 12 : 8)` (dx) | sp_lev.c:1188 | If not vault |
| D3 | `rn2(4)` (dy) | sp_lev.c:1189 | If not vault |
| D4 | `rn2(hx-(lx>0?lx:3)-dx-xborder+1)` (xabs) | sp_lev.c:1200 | If rect-fits |
| D5 | `rn2(hy-(ly>0?ly:2)-dy-yborder+1)` (yabs) | sp_lev.c:1202 | If rect-fits |
| D6 | `rn2(nroom)` (centre-yabs special) | sp_lev.c:1203 | Conditional on edge case |
| D7 | `rn1(3, 2)` (centre-yabs re-pick) | sp_lev.c:1205 | If D6 passes |
| — | `check_room(...)` — `rn2(3)` per non-stone cell | sp_lev.c:1103 | Only rooms 1+ if margin overlaps |
| — | `split_rects(r1, r2)` mutates rect pool | rect.c:161 | No draws but state-dependent |

Sequential per-attempt. No vmap across attempts.

### 1.5 `makecorridors()` — mklev.c:319-348

Many `join()`/`finddpos()`/`dodoor()` draws. Sequential.

### 1.6 `make_niches()` — mklev.c:548-566

`rnd((nroom>>1)+1)` for `ct` (mklev.c:551). Note: current env.py:148 calls this `rn2(mc_upper)` — it's NICHE count, not monster count.

### 1.7 `fill_ordinary_rooms` — mklev.c:803-885

Per OROOM, 17 draw sites (sleep gate, traps, gold, fountain, sink, altar, grave, statue, box, graffiti, mkobj outer). Sequential.

### 1.8 `makemon()` / `newmonhp()` — makemon.c:1106, 983

Already partially ported (`pick_monster_for_level`, `_roll_hp`). Full audit of `peace_minded`, `is_female`, attribute-flag rolls needed.

## 2. Architecture Recommendation

**Hybrid (c).**

- **Outer**: `lax.fori_loop(0, MAXNROFROOMS, body, carry)` where carry = `(Isaac64State, NhRect[51], rect_cnt, Rooms[40], nroom, tried_vault)`. Body unconditionally draws `rn2(rect_cnt)` and masks against `rect_cnt > 0`.
- **`create_room`**: `lax.fori_loop` with `trycnt < 100`, all 5-7 per-attempt rolls, masked acceptance.
- **`split_rects` recursion** → iterative stack with `MAX_SPLIT_DEPTH=64`, `lax.while_loop`.
- **Inner loops** (corridors, niches, fill): `lax.fori_loop` with `Isaac64State` carry. Variable inner loops bounded.

Pure pre-compute fails because `rnd_rect()` depends on `split_rects` state. Pure scan works but slow JIT.

## 3. State Pytree Extension

```python
@struct.dataclass
class DungeonGenState:
    rect_lx: jnp.ndarray       # int16[51]
    rect_ly: jnp.ndarray       # int16[51]
    rect_hx: jnp.ndarray       # int16[51]
    rect_hy: jnp.ndarray       # int16[51]
    rect_cnt: jnp.ndarray      # int32

    smeq: jnp.ndarray          # int32[40]

    door_x: jnp.ndarray        # int8[120]
    door_y: jnp.ndarray        # int8[120]
    door_count: jnp.ndarray    # int32

    tried_vault: jnp.ndarray   # bool

    split_stack_r1_lx: jnp.ndarray  # int16[64]
    split_stack_r1_ly: jnp.ndarray
    split_stack_r1_hx: jnp.ndarray
    split_stack_r1_hy: jnp.ndarray
    split_stack_r2_lx: jnp.ndarray
    split_stack_r2_ly: jnp.ndarray
    split_stack_r2_hx: jnp.ndarray
    split_stack_r2_hy: jnp.ndarray
    split_stack_top: jnp.ndarray    # int32
```

~1.5KB per env, ephemeral inside `mklev_jax()`.

## 4. Implementation Phases

### Phase 1 — `rect.c` port (1 commit)
**File:** new `Nethax/nethax/dungeon/rect_pool.py`. Implement `init_rect`, `get_rect`, `rnd_rect`, `remove_rect`, `add_rect`, `split_rects` as pure JAX. `split_rects` uses iterative stack.

### Phase 2 — `create_room` port (1 commit)
**File:** modify `rooms.py::generate_rooms`. Single `lax.fori_loop` over attempts, draws in vendor order `lit_A → lit_B → rnd_rect → dx → dy → xabs → yabs → (D6, D7) → check_room`.

### Phase 3 — `makerooms` + stair pick (1 commit) [unlocks player_x/y parity]
**File:** modify `branches.py::generate_main_branch_l1`. Replace fixed-N loop with vendor `while (rnd_rect() && nroom < MAXNROFROOMS)`. Add `rn2(nroom)` + `rn2(nroom-1)` stair picks. Add `somex/somey` for stair coords. **Drop env.py:163-167 pre-draws.**

### Phase 4 — `makecorridors` + `make_niches` (1 commit)
**File:** new `Nethax/nethax/dungeon/corridors.py`, modify `branches.py`. Port `join`, `finddpos`, `dodoor`, `dig_corridor`, `make_niches`.

### Phase 5 — `fill_ordinary_rooms` + `makemon` enrichment (1 commit)
**File:** modify `rooms.py::fill_one_isaac`, `spawning.py`. Re-order draws to vendor sequence; add missing box/graffiti/mkobj-outer gates.

## 5. Risk Assessment

### Top 3 risks

**R1 — `split_rects` recursion correctness.** Unbounded vendor recursion; JAX needs hard depth. Silent stack overflow on unusual seed cascades. Mitigation: depth=64 + fuzz across 1000 seeds.

**R2 — `&&` short-circuit semantics.** At least 4 vendor sites (sp_lev.c:1154, mklev.c:770, :858, :866) where rn2 RHS only fires when LHS true. Drawing unconditionally + masking shifts every downstream draw — exactly the bug class we shipped twice (`a783700`, `85ef963`). Each site must be `lax.cond`'d or proved LHS-static.

**R3 — Variable-length inner loops** (graffiti `do{}while(!rn2(40))`, mkobj cascade). Too-tight JAX cap → tail-divergence. Mitigation: cap=8, measure tail frequency.

## 6. Verification Strategy

- **Phase 1**: cross-check rect_cnt + rect[0..50] against C-instrumented vendor trace.
- **Phase 2**: per-room `(lit, dx, dy, xabs, yabs)` for seed=0 match.
- **Phase 3**: `blstats[0,1]` match NLE for seed=0,1,2,3,5,42.
- **Phase 4**: map-glyph diff drops from 99.2% to <50%.
- **Phase 5**: drops to <5%.

Global gate: `pytest tests/test_nle_byte_parity.py -k seed_0` must pass before merging Phase 5.

## Files

- `Nethax/nethax/dungeon/rooms.py`
- `Nethax/nethax/dungeon/branches.py`
- `Nethax/nethax/dungeon/spawning.py`
- `Nethax/nethax/env.py` (drop env.py:163-167 pre-draws)
- `Nethax/nethax/vendor_rng.py` (Isaac64State helpers)
- New: `Nethax/nethax/dungeon/rect_pool.py`

## Estimated Implementation Time

- Phase 1 (rect_pool): 1.5 days
- Phase 2 (create_room): 1 day
- Phase 3 (makerooms + stairs — critical-path for player_x/y): 1 day
- Phase 4 (corridors + niches): 2 days
- Phase 5 (fill + makemon): 1.5 days

**Total: ~7 engineering days.** Phase 3 alone resolves player_x/y for seed=0.
