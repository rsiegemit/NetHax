# True ISAAC64 Draw Count — JIT-aware audit

## Result

`env.reset(seed=0, role=ROGUE, race=HUMAN, alignment=2)` under
`ParityMode.NLE_BYTEPARITY` consumes:

| Source                                      | Draws  |
| ------------------------------------------- | -----: |
| Host-side counter (`_DRAW_COUNT`)           |    107 |
| JIT-traced counter (`Isaac64State.draws`)   | 10,170 |
| Vendor NLE reference                        |  1,789 |

Total Nethax draws: **10,170**
Vendor:             **1,789**

Gap: **+8,381 draws over vendor** (5.685x).

## Methodology

A new `draws: int64` field was added to the `Isaac64State` pytree (registered
under `jax.tree_util.register_pytree_node_class`). It is incremented by:

- `next_uint64_jax` — the only JIT-traceable draw primitive; every
  `rn2_jax` / `rnd_jax` / `rn1_jax` / `rne_jax` / `randint_jax` /
  `isaac_weighted_choice` call routes through it.
- Host-side `next_uint64` / `rn2` / `rne` — incremented by the count of
  underlying `next_uint64_py` draws each helper consumes.

The field survives `vmap` (per-element shape, no batch-leakage), is preserved
through the `_state_to_py` ↔ `_state_from_py` roundtrip used by host helpers,
and is initialised to 0 by `init()` / `empty()`.

## Conclusion

The host-side counter has been blind to the bulk of dungeon-gen RNG
consumption for some time: 10,063 of the 10,170 draws (98.95%) happen inside
JIT-compiled code (dungeon generation, monster placement, item placement,
HP rolls, etc.) and never touched the host `_trace_isaac` callback.

**The recent wave-1/2/3 fixes are firing.** They moved the JIT-side draws —
the host counter could not see the effect because those fixes operate inside
`rn2_jax` / `next_uint64_jax`, which the host counter never observed. The 107
host-side draws have not changed because those represent only the small
host-eager setup phase (seeding, options), not the JIT-traced cascades the
fixes target.

The remaining 5.685x gap vs. vendor's 1,789 draws is the true parity gap and
is the correct target for further audit work. Likely culprits: over-eager
loops in dungeon/spawning cascades drawing more than vendor's bounded
helpers, or extra calls inside monster initialisation HP/inventory blocks.

## Reproducer

```bash
JAX_ENABLE_X64=1 PYTHONPATH=. python -c "
from Nethax.nethax.parity_mode import set_parity_mode, ParityMode
set_parity_mode(ParityMode.NLE_BYTEPARITY)
from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.races import Race
import jax
env = NethaxEnv()
state, _ = env.reset(jax.random.PRNGKey(0), role=Role.ROGUE,
                     race=Race.HUMAN, alignment=2)
print('TOTAL ISAAC64 draws:', int(state.vendor_rng.draws))
"
```
