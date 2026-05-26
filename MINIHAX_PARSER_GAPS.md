# MiniHax DES-parser gaps

Tracks `.des` files under `vendor/minihack/minihack/dat/` whose AST parses
but whose runtime build raises in `LevelGenerator`, causing
`_des_factory` (in `Nethax/minihax/envs/canonical.py`) to fall back to
the procedural LG builder.

Probe protocol: `_des_factory_from_source` invokes the compiled factory
once with `jax.random.PRNGKey(0)` and requires the returned object to
have a `.terrain` attribute (i.e. be a real `EnvState`).  Anything
else — exception or LG-instance return — triggers the fallback.

## Known gaps (as of 2026-05-26)

| .des file        | env(s) affected           | symptom                                                                 |
|------------------|---------------------------|-------------------------------------------------------------------------|
| `quest_hard.des` | `MiniHack-Quest-Hard-v0`  | `KeyError: unknown monster name 'Minotaur'` (line 63 of the .des).      |

All 35 other vendor `.des` files probe-build cleanly into an `EnvState`
with shape `(7, 32, 21, 80)`.

## Reproduction

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python -c "
from Nethax.minihax import des_parser as dp
import jax, os
with open('vendor/minihack/minihack/dat/quest_hard.des') as fh:
    src = fh.read()
factory = dp.des_to_factory(src, w=80, h=21)
factory(jax.random.PRNGKey(0))
"
```

## Triage notes

* `Minotaur` is a vanilla NetHack monster (`H` glyph, hostile, sleeping).
  Adding it to `Nethax/minihax/world_gen` (or the MONSTERS table the LG
  consults) would close this single env without further surgery.
* `quest_hard.des` is the only multi-floor quest .des that references
  a Minotaur; `quest_easy.des` and `quest_medium.des` are clean.
