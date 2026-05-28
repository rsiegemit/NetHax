# MKLEV ISAAC64 Trace Diff — first diverging draw

Date: 2026-05-27
Seed: `jax.random.PRNGKey(0)`, role=ROGUE, race=HUMAN, alignment=2 (chaotic),
ParityMode = `NLE_BYTEPARITY`.

## Method

`Nethax/nethax/vendor_rng.py` carries a JIT-traceable trace hook: `rn2_jax`,
`rnd_jax`, `rn1_jax`, `isaac_weighted_choice`, `randint_jax` each emit a
`jax.debug.callback(_emit_op_callback, ...)` line when JIT tracing is enabled.
The gate (`_jit_trace_enabled`) and the op-file opener (`_trace_init`) now
honour the env var `NETHAX_JIT_TRACE` (alias of the existing
`NETHAX_RNG_TRACE_OPS_JIT` / `NETHAX_RNG_TRACE_OPS`). Host-side draws
(`rn2_py`/`rnd_py` via `_trace_op`) and JIT draws write to the **same** file
in program order, so the capture is a single faithful sequential stream.

`jax.debug.callback` ordering was verified to fire in program order inside both
`lax.scan` and `lax.while_loop` (test: 0,1,2,3,4 emitted in order for both).
Because `env.reset` threads the `Isaac64State` pytree through each eager/JIT
call sequentially (and host helpers block via `int(np.asarray(...))`), host and
JIT draws interleave correctly.

Capture:

```bash
JAX_ENABLE_X64=1 NETHAX_JIT_TRACE=/tmp/nethax_jit_trace.txt PYTHONPATH=. \
  .venv/bin/python -c "
from Nethax.nethax.parity_mode import set_parity_mode, ParityMode
set_parity_mode(ParityMode.NLE_BYTEPARITY)
from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.races import Race
import jax
state, _ = NethaxEnv().reset(jax.random.PRNGKey(0), role=Role.ROGUE,
                             race=Race.HUMAN, alignment=2)
jax.block_until_ready(state)
"
```

Result: 8209 lines, matching `state.vendor_rng.draws` exactly (complete trace).
Vendor reference: `/tmp/vendor_rnd_trace_v2.txt` (2441 lines).

## Counts

| Stream                                  | Lines |
| --------------------------------------- | ----- |
| Vendor (`vendor_rnd_trace_v2.txt`)      | 2441  |
| Nethax JIT trace (`nethax_jit_trace.txt`) | 8209 |

The first **281 draws (index 0..280) are byte-identical** (op, mod, res, order):
`diff <(head -281 vendor) <(head -281 nethax)` returns nothing. These are
`init_objects` (195) + `role_init`/`init_dungeons` (86, ending with the 5×
`rn2(7)` tune string at `vendor/nle/src/dungeon.c:917-918`).

## First diverging draw

**Draw index 281.**

| idx | Vendor               | Nethax              |
| --- | -------------------- | ------------------- |
| 280 | `rn2 mod=7 res=0`    | `rn2 mod=7 res=0`   (match — tune string end) |
| 281 | `rnd mod=2 res=1`    | `rn2 mod=10 res=8`  **← first divergence** |
| 282 | `rn2 mod=90 res=51`  | `rn2 mod=11 res=5`  |
| 283 | `rn2 mod=10 res=5`   | `rn2 mod=10 res=5`  |
| 284 | `rn2 mod=11 res=7`   | `rn2 mod=10 res=5`  |

## Which vendor function

The divergence is **NOT yet in mklev** — it is in `u_init()`
(`vendor/nle/src/u_init.c:582`, called at `vendor/nle/src/allmain.c:615`),
the character-initialisation phase that runs immediately before `mklev`.

mklev's first room actually begins much later: the `create_room` lit roll
`rnd(1 + abs(depth)) < 11 && rn2(77)` (`vendor/nle/src/sp_lev.c:1154`) — the
`rnd(2)`+`rn2(77)` signature — first appears at **vendor draw 356**. Draws
281–355 are the tail of `u_init`:

- 281–~292: `ini_inv(Rogue)` per-item `mksobj` enchant/bless draws
  (`vendor/nle/src/u_init.c:749-756` → `vendor/nle/src/mkobj.c:771-1041`).
  The `rnd mod=1000` draws at vendor 293/296 are the `mksobj` random-type
  rolls (`vendor/nle/src/mkobj.c:251`).
- ~320–347: `init_attr(75)` `rn2(100)` cluster (`vendor/nle/src/attrib.c:627`,
  called at `vendor/nle/src/u_init.c:882`).
- ~348–354: attr-variation loop, 6× `rn2(20)` (`vendor/nle/src/u_init.c:887-894`).

Nethax reproduces these in `Nethax/nethax/subsystems/character.py`
(`_consume_ini_inv_rogue_draws` @ line 1145, `consume_init_attr_draws`,
`_consume_attr_variation_draws`), called from `create_character`
(`Nethax/nethax/subsystems/character.py:1379-1401`).

### Root cause

Vendor emits **two draws before the dagger-quantity roll** that Nethax omits:

- Vendor draw 281 = `rnd mod=2 res=1`
- Vendor draw 282 = `rn2 mod=90 res=51`

Nethax's `_consume_ini_inv_rogue_draws` starts directly with the dagger
quantity `rn1(10,6)` → `rn2(10)` (`character.py:1210`), which lines up with
vendor draw 283 (`rn2 mod=10`), not 281. So Nethax's `u_init` replay is
**short by exactly the two leading draws** `rnd(2)` + `rn2(90)`.

`rn2(90)` does not appear anywhere in `mksobj`/`ini_inv`; in the vendor sources
its only literal site is `vendor/nle/src/u_init.c:715` (Monk `M_spell[rn2(90)/30]`),
which is irrelevant for a Rogue. The pair `rnd(2)` then `rn2(90)` therefore
originates **before the role switch** — in the pre-switch block of `u_init`
(`vendor/nle/src/u_init.c:634-660`): `newhp()` / `newpw()` / `adjabil(0,1)` /
`init_uhunger()` and related setup. Nethax models `newhp`/`newpw` as 0-draw
(`_ini_hpwp_vendor`) for Rogue based on `urole.hpadv.inrnd == 0`
(`vendor/nle/src/role.c:352` `{10,0,0,8,1,0}`), which is correct for the HP
roll, but the live vendor stream still consumes `rnd(2)` + `rn2(90)` here.
These two draws are absent from Nethax's `create_character` reset path
(`character.py:1370-1401`), so every subsequent draw — all of `ini_inv`,
`init_attr`, the attr-variation loop, and the entire `mklev` cascade — is
shifted, producing the observed `player_x` offset and ~99% glyph divergence.

(A coincidental partial re-alignment occurs at draws 320–327 where both
streams sit in the `init_attr` `rn2(100)` cluster, but they re-diverge at 328
because the 2-draw upstream offset changed how many `rn2(100)` iterations
`init_attr` needs; by vendor draw 356 — mklev's first `create_room` — the
streams are fully desynced.)

## Recommended fix

1. **Identify the two missing pre-role-switch `u_init` draws.** Instrument
   vendor NLE (or read the live stream) at the boundary between `init_dungeons`
   tune-string end (draw 280) and the dagger `rn1(10,6)` (vendor draw 283) to
   confirm which `u_init` lines emit `rnd(2)` (281) and `rn2(90)` (282). Prime
   suspects, in `u_init.c:634-660` before `switch (Role_switch)`:
   - `newhp()` / `newpw()` (`attrib.c:981`, `exper.c:47`) — re-audit Rogue
     `hpadv`/`enadv` `inrnd` fields; if both are 0 these do not draw and the
     pair lives elsewhere.
   - `adjabil(0, 1)` (`attrib.c:909`) and its `postadjabil` (`attrib.c:688`).
   - `set_uasmon()` / `init_uhunger()` / `find_ac()` (`do_wear.c`).
   The `rn2(90)` modulus is unusual and should make the emitting site easy to
   pin (grep candidate ranges for an `rn2`/`rnd` whose argument evaluates to 90
   and 2 for a level-0 Rogue-Human-chaotic).

2. **Add a `_consume_uinit_preamble_draws(vendor_rng)` helper** in
   `Nethax/nethax/subsystems/character.py` that emits `rnd(2)` then `rn2(90)`
   (and any further pre-switch draws the audit surfaces), and call it in
   `create_character` **before** `_consume_ini_inv_rogue_draws`
   (`character.py:1382`). Cite the resolved `u_init.c` lines.

3. **Re-run the capture and diff.** Target: first 356+ draws byte-identical
   (through mklev's first `create_room`). Then proceed to the next divergence
   inside `makerooms`/`create_room` proper.

## Files / citations

- Capture hook: `Nethax/nethax/vendor_rng.py:96-153, 738-743` (`NETHAX_JIT_TRACE`).
- Nethax u_init replay: `Nethax/nethax/subsystems/character.py:1145-1319,
  1370-1401`.
- Vendor: `vendor/nle/src/u_init.c:582,634-660,749-756,882,887-894`;
  `vendor/nle/src/mkobj.c:251,771-818,981-1004`;
  `vendor/nle/src/attrib.c:627,909,981`; `vendor/nle/src/sp_lev.c:1154`;
  `vendor/nle/src/dungeon.c:917-918`; `vendor/nle/src/role.c:352`;
  `vendor/nle/src/allmain.c:615,627`.
- Traces: `/tmp/vendor_rnd_trace_v2.txt`, `/tmp/nethax_jit_trace.txt`.
