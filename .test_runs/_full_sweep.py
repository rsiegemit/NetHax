"""FULL byte-parity sweep: every vendor-registered MiniHack env id that Minihax
also supports, across N seeds. Reset-time obs byte-equality (glyphs/chars/agent/
inventory) vs vendor MiniHack. Generalizes minihax_byteparity.py's Room-only
harness to the whole catalog by resolving the vendor class from the gymnasium
registry instead of a hardcoded 12-env map.

Run:  NETHAX_EAGER=1 JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python \
        .test_runs/_full_sweep.py --seeds 0,1,2,5
"""
import os, sys, argparse, random, importlib, collections
os.environ.setdefault("NETHAX_EAGER", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import gymnasium as _gym
sys.modules["gym"] = _gym
sys.modules["gym.spaces"] = _gym.spaces
sys.modules["gym.envs"] = _gym.envs
sys.modules["gym.envs.registration"] = _gym.envs.registration
import numpy as np
import jax
import minihack  # registers vendor ids
from Nethax.nethax.parity_mode import ParityMode, set_parity_mode
set_parity_mode(ParityMode.NLE_BYTEPARITY)  # REQUIRED: vendor-RNG level gen
from Nethax.minihax.minihax_env import MinihaxEnv
from Nethax.nethax.obs.nle_obs import build_nle_observation

_OBS_KEYS = ("glyphs", "chars", "blstats", "inv_glyphs", "inv_letters", "inv_strs")


def _agent_yx(bl):
    return (int(bl[1]), int(bl[0]))


def _vendor_cls(env_id):
    spec = _gym.registry[env_id]
    ep = spec.entry_point
    mod, cls = ep.split(":") if isinstance(ep, str) else (ep.__module__, ep.__name__)
    return getattr(importlib.import_module(mod), cls), dict(spec.kwargs or {})


def _dump(obs):
    return {k: np.asarray(obs[k]) for k in _OBS_KEYS}


def vendor_dump(env_id, seed):
    cls, kw = _vendor_cls(env_id)
    kw.pop("observation_keys", None); kw.pop("seeds", None)
    random.seed(seed)
    env = cls(observation_keys=_OBS_KEYS, character="arc-hum-law-mal", **kw)
    if env._level_seeds is not None:
        raise RuntimeError("level_seeds not None (would burn a draw)")
    env.seed(seed, seed, reseed=False)
    r = env.reset()
    obs = r[0] if isinstance(r, tuple) else r
    d = _dump(obs)
    try: env.close()
    except Exception: pass
    return d


def minihax_dump(env_id, seed):
    env = MinihaxEnv(env_id)
    state, _ = env.reset(jax.random.key(seed))
    return _dump(build_nle_observation(state))


def diff(v, m):
    for k in ("glyphs", "chars"):
        if v[k].shape != m[k].shape:
            return f"{k} shape {v[k].shape}!={m[k].shape}"
        ne = np.argwhere(v[k] != m[k])
        if ne.size:
            y, x = int(ne[0, 0]), int(ne[0, 1])
            return f"{k}@({y},{x}) v={int(v[k][y,x])} m={int(m[k][y,x])}"
    if _agent_yx(v["blstats"]) != _agent_yx(m["blstats"]):
        return f"agent {_agent_yx(v['blstats'])}!={_agent_yx(m['blstats'])}"
    for k in ("inv_glyphs", "inv_letters", "inv_strs"):
        if v[k].shape != m[k].shape:
            return f"{k} shape {v[k].shape}!={m[k].shape}"
        if not np.array_equal(v[k], m[k]):
            return f"{k} differs"
    return None


def family(env_id):
    # MiniHack-<Family>...-v0 -> <Family> (first token after MiniHack-)
    core = env_id[len("MiniHack-"):]
    for sep in ("-", "N", "S", "R"):
        pass
    import re
    m = re.match(r"([A-Za-z]+)", core)
    return m.group(1) if m else core


ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1,2,5")
ap.add_argument("--limit", type=int, default=0, help="cap ids (0=all)")
args = ap.parse_args()
seeds = [int(x) for x in args.seeds.split(",")]

vend_ids = sorted(k for k in _gym.registry.keys() if k.startswith("MiniHack"))
# keep only ids Minihax can build
mh_ids = []
for eid in vend_ids:
    try:
        MinihaxEnv(eid); mh_ids.append(eid)
    except Exception:
        pass
if args.limit:
    mh_ids = mh_ids[:args.limit]
print(f"vendor ids={len(vend_ids)}  minihax-supported={len(mh_ids)}  seeds={seeds}", flush=True)
print(f"sweeping {len(mh_ids)} envs x {len(seeds)} seeds = {len(mh_ids)*len(seeds)} checks\n", flush=True)

fam_pass = collections.Counter(); fam_tot = collections.Counter()
fails = []
for eid in mh_ids:
    fam = family(eid)
    row = []
    for s in seeds:
        fam_tot[fam] += 1
        try:
            d = diff(vendor_dump(eid, s), minihax_dump(eid, s))
            if d is None:
                fam_pass[fam] += 1; row.append(f"{s}:OK")
            else:
                row.append(f"{s}:X"); fails.append((eid, s, d))
        except Exception as e:
            row.append(f"{s}:ERR"); fails.append((eid, s, f"{type(e).__name__}: {str(e)[:70]}"))
    print(f"  {eid:40s} {' '.join(row)}", flush=True)

print("\n=== PER-FAMILY ===")
for fam in sorted(fam_tot):
    print(f"  {fam:16s} {fam_pass[fam]:4d}/{fam_tot[fam]:4d}")
tot_p = sum(fam_pass.values()); tot_t = sum(fam_tot.values())
print(f"\n=== TOTAL: {tot_p}/{tot_t} byte-exact ===")
if fails:
    print(f"\nfirst 25 failures/errors:")
    for eid, s, d in fails[:25]:
        print(f"  {eid} seed={s}: {d}")
print("SWEEP_DONE")
