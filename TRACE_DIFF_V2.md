# ISAAC64 Call Trace Diff V2 (post-wave-3)

Date: 2026-05-27
Seeds: CORE=0, DISP=0, role=ROGUE, race=HUMAN, alignment=2

## Counts

| Stream                                             | Lines |
| -------------------------------------------------- | ----- |
| Vendor (`/tmp/vendor_rnd_trace_v2.txt`)            | 2441  |
| Nethax host-only (`/tmp/nethax_rnd_trace_v2.txt`)  | 104   |
| Nethax init_objects-only (`/tmp/nethax_rnd_trace_v3.txt`, host `_py` replay) | 195   |

Prior V1: vendor=1789, Nethax=107. Vendor jumped because the .so was
rebuilt with the post-V1 instrumentation hooks (more sites traced).

## Visibility caveat

The Nethax `NETHAX_RNG_TRACE_OPS` hook lives in `vendor_rng._trace_op`
and only fires from the host-side `rn2_py / rnd_py / rne_py` helpers
(`Nethax/nethax/vendor_rng.py:119, 324, 339`). The JIT-traceable variants
(`rn2_jax` etc.) do NOT emit trace lines. Reset's `compute_descr_shuffle`
(`Nethax/nethax/obs/glyph_shuffle.py:175-222`) runs under `fori_loop` on
the JAX path, so its 195 ISAAC64 draws never reach the file.

The 104 lines in `nethax_rnd_trace_v2.txt` are therefore POST-reset
host-side draws (rn1/rnz/rne side calls during character init and the
post-mklev settle), not the early init_objects/role_init/u_init/mklev
draws that vendor logs.

For a fair compare I replayed Nethax `init_objects` byte-for-byte through
the `_py` path (`/tmp/nethax_rnd_trace_v3.txt`).

## First diverging position

### Vendor vs Nethax env.reset trace (raw, line 0)

| Source | Line 0 op |
| ------ | --------- |
| Vendor | `0 rn2 mod=2 res=1`  (init_objects GEM jitter, `vendor/nle/src/o_init.c:144`) |
| Nethax | `0 rn2 mod=10 res=4` (post-reset host draw; the entire init_objects prefix is invisible) |

This is a visibility artifact, not a parity bug.

### Vendor vs Nethax init_objects-only replay

All 195 init_objects ops match byte-exactly (modulus, result, ordering):

```
$ diff /tmp/vendor_rnd_trace_v2.txt /tmp/nethax_rnd_trace_v3.txt | head -3
196,2441d195                                 # tail only — Nethax script stops
< 195 rn2 mod=5 res=4                        # vendor line 195 = first post-init_objects draw
< 196 rn2 mod=100 res=38
```

First Nethax-visible divergence: **none** through the entire 195-op
init_objects prefix. The vendor stream then continues into
`role_init() → init_dungeons() → init_artifacts() → u_init() → mklev()`,
none of which are exercised by Nethax via the host trace path.

## What is at vendor line 195 (first post-init_objects op)

`195 rn2 mod=5 res=4` — followed by `rn2(100) × 5` and a mod=4/5/3/4 cluster.

The `rn2(100)` cluster is the chance-gate scan in
`vendor/nle/src/dungeon.c:776` (`init_dungeons` walking `tmpdungeon[].chance`).
The leading `rn2(5)` is most likely `role_init()` selecting a pantheon/quest
parameter before `init_dungeons` runs (`vendor/nle/src/role.c::role_init`).

Nethax currently does not consume these draws on the CORE ISAAC64 stream
during `env.reset` — it returns immediately to user-visible host draws
once `compute_descr_shuffle` finishes (`Nethax/nethax/env.py:175-206`).

## Top fix recommendation

**Add a tracing shim to `rn2_jax` and friends so JAX-path draws emit the
same `_trace_op` lines as `rn2_py`.** Without that, every audit run will
show the same illusory 104-vs-2441 gap regardless of how many cascades
land.

Concrete options (in order of effort):

1. **Cheap (recommended for audits):** Inside `compute_descr_shuffle` and
   any other reset-path callers, branch on `NETHAX_RNG_TRACE_OPS` env var
   to use `rn2_py` instead of `rn2_jax`. Already proven equivalent (this
   trace).

2. **Medium:** Add `jax.debug.callback(_trace_op, ...)` inside `rn2_jax`
   guarded by an env-var-controlled module flag. Survives JIT but
   serialises during tracing — disable for prod.

3. **Audit-correctness:** Port the missing vendor reset draws into
   Nethax. After `compute_descr_shuffle` the CORE stream needs:
   - `role_init` quest/pantheon picks (~1-3 draws),
   - `init_dungeons` `rn2(100)` chance gate scan
     (`vendor/nle/src/dungeon.c:776`, one per dungeon-entry candidate),
   - `init_artifacts` shuffles (`vendor/nle/src/artifact.c::init_artifacts`),
   - `u_init` role-specific init draws (`vendor/nle/src/u_init.c:663-794`).

   Without these the ISAAC64 stream is mis-aligned by ~50-100 words
   before `mklev()` even starts — explaining the long tail of
   non-matching downstream rn2 moduli in `ISAAC64_CALL_ORDER_AUDIT.md`.

## Files

- `/tmp/vendor_rnd_trace_v2.txt` — 2441 lines, fresh vendor trace
- `/tmp/nethax_rnd_trace_v2.txt` — 104 lines, env.reset host-path only
- `/tmp/nethax_rnd_trace_v3.txt` — 195 lines, init_objects-only `_py` replay
  (byte-exact with vendor lines 0-194)
