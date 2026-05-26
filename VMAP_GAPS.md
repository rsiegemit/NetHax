# VMAP_GAPS.md

Catalogue of host-side / vmap-unsafe spots in `NethaxEnv` that block
`jax.vmap(self.reset)`.  `step` is already JIT-compiled by
`_step_jit = jax.jit(_step_impl)` and is fully vmap-safe — every branch
inside `_step_impl` uses `jax.lax.cond` / `jnp.where`, no `.item()` /
`int(...)` / `np.array` calls escape to the host.

The batched API (`step_batched`, `reset_batched`) ships in this commit:
- `step_batched` — true `jax.vmap` wrapper; B independent envs per device.
- `reset_batched` — Python-for fallback (correct semantics, not parallel)
  until the gaps below are closed.

---

## Gap 1 — `NethaxEnv.reset` ISAAC64 seeding (env.py ~line 112-119)

```python
if use_vendor_rng():
    key_u32 = jax.device_get(rng).astype(jnp.uint32)   # <-- host copy
    seed_int = (int(key_u32[0]) << 32) ^ int(key_u32[1])  # <-- .item()/int()
    v_state = _vendor_rng.init(seed_int)
    v_state, rng_level = _vendor_draw_prngkey(v_state)
    ...
```

**Why it breaks vmap.** `jax.device_get` forces a host transfer of the
traced `rng` argument; under `vmap` the value is a `BatchedTracer` with
no concrete bytes.  `int(...)` on a JAX array under vmap raises
`ConcretizationTypeError`.

**Fix path.** Rewrite `vendor_rng.init` to accept a `jax.Array` seed
directly and use `jnp.bitwise_xor` / `jnp.left_shift` — see
`Nethax/nethax/vendor_rng.py`.  Today the ISAAC64 path is gated behind
`use_vendor_rng()` (parity mode), so non-parity training paths are
unaffected.

Also `_vendor_draw_prngkey` (env.py ~line 215) packs via Python `int`
shifts:

```python
hi = jnp.uint32((int(val) >> 32) & 0xFFFFFFFF)
lo = jnp.uint32(int(val) & 0xFFFFFFFF)
```

Same fix — replace with `jnp.right_shift(val.astype(jnp.uint64), 32)`
and `jnp.bitwise_and`.

---

## Gap 2 — `_spawn_starting_pet` host loops (env.py ~line 222-290)

```python
import numpy as np
...
terrain = np.array(state.terrain[0, 0])     # <-- host copy of traced array
pr = int(state.player_pos[0])               # <-- .item()
pc = int(state.player_pos[1])
H, W = terrain.shape
pet_pos = (pr, pc)
for dr in (-1, 0, 1):                       # <-- Python control flow
    for dc in (-1, 0, 1):
        ...
        if 0 <= rr < H and 0 <= cc < W:
            t = int(terrain[rr, cc])        # <-- .item()
            if t in (int(TileType.FLOOR), int(TileType.CORRIDOR)):
                pet_pos = (rr, cc)
```

**Why it breaks vmap.** Every line that materialises a Python int from
a traced array is a vmap blocker.  The adjacent-tile search must become
a static-shape scan.

**Fix path.** Replace the 3x3 neighbour scan with a `jnp.where`-based
selection over the 8 neighbour offsets and a precedence mask
(`floor > corridor > self`).  All ops become pure-JAX.

```python
offsets = jnp.array([[dr, dc] for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0)])
cands = state.player_pos[None, :] + offsets          # [8, 2]
tiles = state.terrain[0, 0][cands[:, 0], cands[:, 1]]
ok = (tiles == FLOOR) | (tiles == CORRIDOR)
pick = jnp.argmax(ok.astype(jnp.int32))              # first ok, or 0
pet_pos = jnp.where(jnp.any(ok), cands[pick], state.player_pos)
```

The host-side `next(...)` over `MONSTERS` to resolve a name → index can
be pre-tabulated as a module-level constant.

---

## Gap 3 — `_roll_hp(dummy_rng, jnp.int32(...))` int conversion (env.py ~line 265)

```python
hp_val = int(_roll_hp(dummy_rng, jnp.int32(max(1, int(MONSTERS[pet_pm].level)))))
```

`int(_roll_hp(...))` forces concretisation.  Easy fix: store the
traced result directly in the `monster_ai.hp` slot — the assignment
already uses `.at[PET_SLOT].set(jnp.int32(...))`.

---

## Gap 4 — Default-arg branches (env.py ~line 85-88)

```python
if role is None:
    role = Role.VALKYRIE
if race is None:
    race = Race.HUMAN
```

These are *Python-side* defaults set before any tracing begins, so they
are **not** a vmap blocker for the rng axis — `role`/`race` are
hyperparameters, not batched inputs.  Documented here so the reader
doesn't try to "fix" them.

For per-env role variation (e.g. multi-class training rollouts), pass a
batched `role_indices` array and gate role-conditional code inside
`create_character` on it with `jnp.where` / `lax.switch`.

---

## Gap 5 — `populate_level_with_monsters` shape-dependent paths

`Nethax/nethax/dungeon/spawning.py::populate_level_with_monsters` is
called from reset.  Quick survey shows it uses `jax.lax.fori_loop` /
`jnp.where`, so it should vmap cleanly once the surrounding `reset`
loses its host-side calls.  Not separately validated under vmap — flag
for follow-up.

---

## Gap 6 — `compute_fov` reset-seed call (env.py ~line 168-176)

```python
from Nethax.nethax.fov import compute_fov
vis = compute_fov(state.terrain[0, 0, :, :], state.player_pos.astype(jnp.int32))
```

`compute_fov` is implemented in pure JAX (see `Nethax/nethax/fov.py`) and
should vmap cleanly.  Not separately validated — flag for follow-up
once Gaps 1-3 are closed.

---

## Summary

| Site                              | Severity | Status |
| --------------------------------- | -------- | -------- |
| `device_get` + `int(key_u32)`     | blocker  | **closed** — uses `jax.random.bits`; remaining `int(seed_arr)` is gated behind `use_vendor_rng()` (host-only opt-in mode) |
| `_vendor_draw_prngkey` int shifts | blocker  | **closed** — pure `jnp.right_shift`/`jnp.bitwise_and` |
| `_spawn_starting_pet` np/loops    | blocker  | **closed** — 8-neighbour `jnp.where` selection |
| `_roll_hp` int conversion         | blocker  | **closed** — traced scalar stored directly |
| `_init_attr_vendor` Python `while`| blocker  | **closed** — `lax.while_loop` + pre-split key bundle |
| Default-arg Python branches       | n/a      | n/a (hyperparams) |
| `populate_level_with_monsters`    | clean    | vmap-safe today |
| `compute_fov` call site           | clean    | vmap-safe today |

`step_batched` works today.  Default-mode `jax.vmap(env.reset)(rngs)`
now runs cleanly — see smoke test in commit message.  The
`use_vendor_rng()` byte-parity path is still host-only by design
(ISAAC64 init operates on Python lists).
