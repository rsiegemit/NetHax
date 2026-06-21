"""Capture the COMPLETE CORE ISAAC64 draw stream (NETHAX_RND) for an env/seed.

Unlike NETHAX_RN2 (which traces only rn2()), NETHAX_RND traces every RND(x)
call — i.e. rn2, rnd, d, rnl, rn1 — so it captures the untraced rnd()/d()
draws that makemon/mktrap make.  Requires the NLE rebuild with the RND()
instrumentation (vendor/nle/src/rnd.c).

Usage:
    CAPTURE_ENV=MiniHack-Room-Trap-5x5-v0 CAPTURE_SEED=0 \
      JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python \
      .test_runs/_capture_full_rnd_stream.py
"""
import os, sys, subprocess

PY = sys.executable
ENV = os.environ.get("CAPTURE_ENV", "MiniHack-Room-Trap-5x5-v0")
SEED = os.environ.get("CAPTURE_SEED", "0")

# Map env_id -> vendor class.
CLS = {
    "MiniHack-Room-Trap-5x5-v0": "MiniHackRoom5x5Trap",
    "MiniHack-Room-Trap-15x15-v0": "MiniHackRoom15x15Trap",
    "MiniHack-Room-Monster-5x5-v0": "MiniHackRoom5x5Monster",
    "MiniHack-Room-Monster-15x15-v0": "MiniHackRoom15x15Monster",
    "MiniHack-Room-Ultimate-5x5-v0": "MiniHackRoom5x5Ultimate",
    "MiniHack-Room-Ultimate-15x15-v0": "MiniHackRoom15x15Ultimate",
}[ENV]

SCRIPT = f"""
import os, sys, random
sys.modules.setdefault('gym', __import__('gymnasium'))
import gymnasium as _g
sys.modules['gym.spaces'] = _g.spaces
sys.modules['gym.envs'] = _g.envs
sys.modules['gym.envs.registration'] = _g.envs.registration
import minihack
from minihack.envs.room import {CLS}
random.seed(int(os.environ['SEED']))
env = {CLS}(observation_keys=('chars',), character='arc-hum-law-mal')
env.seed(int(os.environ['SEED']), int(os.environ['SEED']), reseed=False)
env.reset()
env.close()
"""

OUT = f"/Users/rsiegelmann/Downloads/Projects/nethax/.test_runs/full_rnd_stream_{ENV.replace('-', '_')}_seed{SEED}.txt"

env = os.environ.copy()
env["NETHAX_RND_TRACE"] = "1"
env["NETHAX_RN2_TRACE"] = "1"  # keep rn2 markers too, for cross-reference
env["SEED"] = SEED
res = subprocess.run([PY, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=180)
lines = [ln for ln in res.stderr.splitlines()
         if ln.startswith("NETHAX_RND") or ln.startswith("NETHAX_RN2")
         or "BEGIN" in ln or "END" in ln]
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
n_rnd = sum(1 for ln in lines if ln.startswith("NETHAX_RND"))
n_rn2 = sum(1 for ln in lines if ln.startswith("NETHAX_RN2"))
print(f"Wrote {OUT}: {len(lines)} lines ({n_rnd} RND, {n_rn2} RN2)")
print(f"untraced (RND-only) draws = {n_rnd - n_rn2}")
