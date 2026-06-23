"""Transfer eval (Path A): run the PPO policy trained on REAL MiniHack on both
the real env and our Minihax (byte-parity + vec modes), compare return/success.

Agent action i (0..7) -> ASCII ord (N/E/S/W/NE/SE/SW/NW). Real env uses its own
action list; Minihax accepts the ASCII ord directly (Nethax: action>=86 = ord).
Obs: player-centered 9x9 glyph crop (identical preprocessing on both).

Usage:
  JAX_PLATFORMS=cpu NETHAX_EAGER=1 PYTHONPATH=. \
  .venv/bin/python .test_runs/eval_transfer.py --ckpt .test_runs/ppo_room5x5.pt \
     --real-env MiniHackRoom5x5 --minihax-env MiniHack-Room-5x5-v0 --episodes 20
"""
import argparse, numpy as np, torch

# gym/gymnasium shim before minihack
import gym, gymnasium
import gymnasium.spaces.dict as _gsd
_gsd.Space = (gymnasium.spaces.Space, gym.spaces.Space)

import importlib.util, os
spec = importlib.util.spec_from_file_location("ppo_minihack",
    os.path.join(os.path.dirname(__file__), "ppo_minihack.py"))
ppo = importlib.util.module_from_spec(spec); spec.loader.exec_module(ppo)
crop_glyphs, Agent = ppo.crop_glyphs, ppo.Agent

ORDS = [107, 108, 106, 104, 117, 110, 98, 121]  # N E S W NE SE SW NW (vi keys)
MAXSTEPS = 100


def load_agent(ckpt):
    d = torch.load(ckpt, map_location="cpu")
    ag = Agent(d["n_act"]); ag.load_state_dict(d["state_dict"]); ag.eval()
    return ag


def act(ag, glyph_crop):
    with torch.no_grad():
        logits, _ = ag(torch.tensor(glyph_crop[None], dtype=torch.long))
        return int(torch.argmax(logits, -1))   # greedy eval


def eval_real(ag, env_cls_name, episodes):
    mod = importlib.import_module("minihack.envs.room")
    rets, succ = [], 0
    for ep in range(episodes):
        e = getattr(mod, env_cls_name)()
        o = e.reset(); o = o[0] if isinstance(o, tuple) else o
        R = 0.0
        for t in range(MAXSTEPS):
            a = act(ag, crop_glyphs(o["glyphs"], o["blstats"]))
            step = e.step(a); o, r, done = step[0], step[1], step[2]
            o = o[0] if isinstance(o, tuple) else o
            R += r
            if done:
                if r > 0: succ += 1
                break
        rets.append(R)
    return float(np.mean(rets)), succ / episodes


def eval_minihax(ag, env_name, episodes):
    import jax
    from Nethax.minihax.minihax_env import MinihaxEnv
    from Nethax.nethax.obs.nle_obs import build_nle_observation
    env = MinihaxEnv(env_name)
    rets, succ = [], 0
    for ep in range(episodes):
        s, info = env.reset(jax.random.key(1000 + ep))
        fm = info["fired_mask"]; par = info["pits_at_reset"]
        R = 0.0
        for t in range(MAXSTEPS):
            ob = build_nle_observation(s)
            crop = crop_glyphs(np.asarray(ob["glyphs"]), np.asarray(ob["blstats"]))
            i = act(ag, crop)
            s, r, done, info = env.step(s, ORDS[i], jax.random.key(7000 + ep * 200 + t),
                                        fired_mask=fm, step_count=t, pits_at_reset=par)
            fm = info["fired_mask"]
            r = float(r); R += r
            if bool(done):
                if r > 0: succ += 1
                break
        rets.append(R)
    return float(np.mean(rets)), succ / episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=".test_runs/ppo_room5x5.pt")
    ap.add_argument("--real-env", default="MiniHackRoom5x5")
    ap.add_argument("--minihax-env", default="MiniHack-Room-5x5-v0")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()
    ag = load_agent(args.ckpt)

    print(f"[transfer] ckpt={args.ckpt} episodes={args.episodes}", flush=True)
    mr, sr = eval_real(ag, args.real_env, args.episodes)
    print(f"  REAL MiniHack   : mean_return {mr:+.3f}  success {sr:.0%}", flush=True)
    mr, sr = eval_minihax(ag, args.minihax_env, args.episodes)
    print(f"  Minihax (vec)   : mean_return {mr:+.3f}  success {sr:.0%}", flush=True)


if __name__ == "__main__":
    main()
