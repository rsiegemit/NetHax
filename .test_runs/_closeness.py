"""Per-family 'closeness' to byte-parity: for one representative env per family,
count how many glyph cells differ vs vendor (seed 0). Few = cheap fix; many =
structural. Prioritizes which families to extend first."""
import os, sys, random, importlib, collections
os.environ.setdefault("NETHAX_EAGER", "1"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import gymnasium as _gym
sys.modules["gym"]=_gym; sys.modules["gym.spaces"]=_gym.spaces
sys.modules["gym.envs"]=_gym.envs; sys.modules["gym.envs.registration"]=_gym.envs.registration
import numpy as np, jax, minihack
from Nethax.nethax.parity_mode import ParityMode, set_parity_mode
set_parity_mode(ParityMode.NLE_BYTEPARITY)
from Nethax.minihax.minihax_env import MinihaxEnv
from Nethax.nethax.obs.nle_obs import build_nle_observation

def vend(eid, s):
    spec=_gym.registry[eid]; mod,cls=spec.entry_point.split(":")
    C=getattr(importlib.import_module(mod),cls)
    random.seed(s); e=C(observation_keys=("glyphs",),character="arc-hum-law-mal")
    if e._level_seeds is not None: raise RuntimeError("seeds")
    e.seed(s,s,reseed=False); o=e.reset(); o=o[0] if isinstance(o,tuple) else o
    return np.asarray(o["glyphs"])
def mh(eid, s):
    st,_=MinihaxEnv(eid).reset(jax.random.key(s)); return np.asarray(build_nle_observation(st)["glyphs"])

def fam(eid):
    import re; return re.match(r"([A-Za-z]+)", eid[len("MiniHack-"):]).group(1)

ids=sorted(k for k in _gym.registry if k.startswith("MiniHack"))
seen={}
for eid in ids:
    f=fam(eid)
    if f in seen or f=="Room" or f=="Boxoban": continue
    try: MinihaxEnv(eid)
    except Exception: continue
    seen[f]=eid
rows=[]
for f,eid in seen.items():
    try:
        v=vend(eid,0); m=mh(eid,0)
        if v.shape!=m.shape: rows.append((99999,f,eid,f"shape {v.shape}!={m.shape}")); continue
        nd=int(np.sum(v!=m)); tot=v.size
        # is minihax basically empty (player at 0,0 => level didn't gen)?
        m00=int(m[0,0]); note="player@origin(no-gen?)" if m00==327 else ""
        rows.append((nd,f,eid,f"{nd}/{tot} cells differ {note}"))
    except Exception as e:
        rows.append((88888,f,eid,f"ERR {type(e).__name__}: {str(e)[:40]}"))
rows.sort()
print("=== closeness (fewest differing cells first = cheapest to fix) ===")
for nd,f,eid,d in rows:
    print(f"  {f:16s} {d}")
print("CLOSE_DONE")
