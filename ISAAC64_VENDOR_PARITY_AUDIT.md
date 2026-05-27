# ISAAC64 Vendor Parity Audit

**Date:** 2026-05-27  
**Seed tested:** 0  
**Draws compared:** 256

## Result: All 256 match

The Python/JAX ISAAC64 implementation in `Nethax/nethax/vendor_rng.py` is
byte-identical with the vendor C for seed=0.

```
diff /tmp/isaac64_dump_c.txt /tmp/isaac64_python.txt
(empty — no differences)
```

First five output words (hex), confirming agreement:

```
9d39247e33776d41
2af7398005aaa5c7
44db015024623547
9c15f73e62a76ae2
75834465489c0c89
```

## Methodology

1. **Vendor C binary** — a self-contained C program (`/tmp/isaac64_dump.c`)
   was written that:
   - Inlines the full `isaac64_ctx` struct from
     `vendor/nle/include/isaac64.h:29-36`
   - Copies `isaac64_update`, `isaac64_mix`, `isaac64_init`,
     `isaac64_reseed`, `isaac64_next_uint64` verbatim from
     `vendor/nle/src/isaac64.c:46-160`
   - Implements `init_isaac64(ctx, 0UL)` exactly as
     `vendor/nle/src/rnd.c:39-57` (pack `unsigned long` LE into 8 bytes,
     call `isaac64_init`)
   - Calls `isaac64_next_uint64` 256 times and prints each as lowercase hex

2. **Python output** — ran:
   ```
   JAX_ENABLE_X64=1 PYTHONPATH=. python -c "
   from Nethax.nethax.vendor_rng import init, next_uint64
   rng = init(0)
   for i in range(256): rng, v = next_uint64(rng); print(f'{int(v):016x}')
   " > /tmp/isaac64_python.txt
   ```

3. **Comparison** — `diff` confirmed zero divergences across all 256 values.

## Implication

The ISAAC64 stream itself is correct. If NLE byte-parity diverges at the
dungeon-gen or combat level, the fault lies **downstream** of the RNG, not in
the RNG itself. Likely suspects:

- **Call order / call count** — which dungeon-gen or monster routines consume
  how many draws, and in what order. See `ISAAC64_CALL_ORDER_AUDIT.md` for
  prior analysis.
- **`rn2` vs `rnd` vs `rnl` routing** — a single extra or missing draw at any
  call site shifts every subsequent output.
- **Dual-RNG split** (CORE vs DISP) — `rnglist[CORE]` vs
  `rnglist[DISP]` in `vendor/nle/src/rnd.c:20-25`; if display draws are
  incorrectly routed through the core stream the sequences diverge.

No changes to `vendor_rng.py` are needed or warranted.
