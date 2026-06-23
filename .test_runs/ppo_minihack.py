"""Minimal CNN-PPO for MiniHack-Room, trained on the REAL MiniHack C env.

Path A of the transfer eval: train here on real MiniHack, then evaluate the saved
policy on Minihax (eval_transfer.py) to measure sim-to-sim transfer.

Obs: player-centered 9x9 crop of `glyphs` (reproducible on any NLE-format env —
both real MiniHack and Minihax produce `glyphs` + `blstats`).  Glyph ids are
embedded, passed through a tiny conv, then actor/critic heads.

gym/gymnasium interop shim applied before importing minihack (minihack builds a
gymnasium Dict with gym Box children, which gymnasium rejects without this).

Usage:
  .venv/bin/python .test_runs/ppo_minihack.py --env MiniHackRoom5x5 \
      --total-steps 300000 --out .test_runs/ppo_room5x5.pt
"""
import argparse, time, numpy as np

# ---- gym/gymnasium interop shim (must precede minihack import) ----
import gym, gymnasium
import gymnasium.spaces.dict as _gsd
_gsd.Space = (gymnasium.spaces.Space, gym.spaces.Space)

import torch
import torch.nn as nn
import torch.nn.functional as F

CROP = 9
MAX_GLYPH = 6000  # NLE glyph ids fit under this


def make_env(name):
    import importlib
    mod = importlib.import_module("minihack.envs.room")
    return getattr(mod, name)()


def crop_glyphs(glyphs, blstats):
    """Player-centered CROPxCROP crop of the full glyph grid. blstats[0]=x col,
    blstats[1]=y row (NLE convention)."""
    g = np.asarray(glyphs)
    H, W = g.shape
    x = int(blstats[0]); y = int(blstats[1])
    r = CROP // 2
    pad = np.pad(g, r, mode="constant", constant_values=0)
    return pad[y:y + CROP, x:x + CROP]  # (CROP, CROP)


class Agent(nn.Module):
    def __init__(self, n_act):
        super().__init__()
        self.emb = nn.Embedding(MAX_GLYPH, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(32 * CROP * CROP, 128), nn.ReLU())
        self.actor = nn.Linear(128, n_act)
        self.critic = nn.Linear(128, 1)

    def forward(self, glyph_crop):  # [B, CROP, CROP] int
        x = self.emb(glyph_crop.long())            # [B,C,C,16]
        x = x.permute(0, 3, 1, 2)                  # [B,16,C,C]
        x = self.conv(x).flatten(1)
        h = self.fc(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniHackRoom5x5")
    ap.add_argument("--total-steps", type=int, default=300000)
    ap.add_argument("--rollout", type=int, default=1024)
    ap.add_argument("--out", default=".test_runs/ppo_room5x5.pt")
    args = ap.parse_args()

    env = make_env(args.env)
    n_act = env.action_space.n
    agent = Agent(n_act)
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4)

    def obs_to_crop(o):
        return crop_glyphs(o["glyphs"], o["blstats"])

    o = env.reset(); o = o[0] if isinstance(o, tuple) else o
    cur = obs_to_crop(o)
    gamma, lam, clip, epochs, mb = 0.99, 0.95, 0.2, 4, 256
    ep_ret, ep_len, rets = 0.0, 0, []
    t0 = time.time()
    steps_done = 0
    while steps_done < args.total_steps:
        S, A, R, D, V, LP = [], [], [], [], [], []
        for _ in range(args.rollout):
            st = torch.tensor(cur[None], dtype=torch.long)
            with torch.no_grad():
                logits, v = agent(st)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
            step_out = env.step(int(a))
            o2, r, done = step_out[0], step_out[1], step_out[2]
            o2 = o2[0] if isinstance(o2, tuple) else o2
            S.append(cur); A.append(int(a)); R.append(float(r)); D.append(bool(done))
            V.append(float(v)); LP.append(float(dist.log_prob(a)))
            ep_ret += r; ep_len += 1; steps_done += 1
            if done or ep_len >= 200:
                rets.append(ep_ret); ep_ret, ep_len = 0.0, 0
                o = env.reset(); o = o[0] if isinstance(o, tuple) else o
                cur = obs_to_crop(o)
            else:
                cur = obs_to_crop(o2)
        # GAE
        with torch.no_grad():
            last_v = agent(torch.tensor(cur[None], dtype=torch.long))[1].item()
        adv = np.zeros(len(R), dtype=np.float32); gae = 0.0
        for i in reversed(range(len(R))):
            nv = last_v if i == len(R) - 1 else V[i + 1]
            nonterm = 1.0 - D[i]
            delta = R[i] + gamma * nv * nonterm - V[i]
            gae = delta + gamma * lam * nonterm * gae
            adv[i] = gae
        ret = adv + np.array(V, dtype=np.float32)
        Sb = torch.tensor(np.array(S), dtype=torch.long)
        Ab = torch.tensor(A); LPb = torch.tensor(LP)
        advb = torch.tensor((adv - adv.mean()) / (adv.std() + 1e-8))
        retb = torch.tensor(ret)
        idx = np.arange(len(R))
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, len(R), mb):
                b = idx[s:s + mb]
                logits, v = agent(Sb[b])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(Ab[b])
                ratio = torch.exp(lp - LPb[b])
                s1 = ratio * advb[b]; s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advb[b]
                ploss = -torch.min(s1, s2).mean()
                vloss = F.mse_loss(v, retb[b])
                ent = dist.entropy().mean()
                loss = ploss + 0.5 * vloss - 0.01 * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5); opt.step()
        if rets:
            mr = float(np.mean(rets[-50:]))
            print(f"steps {steps_done:>7} | ep_ret(last50) {mr:+.3f} | eps {len(rets)} | {steps_done/(time.time()-t0):.0f} sps", flush=True)
    torch.save({"state_dict": agent.state_dict(), "n_act": n_act,
                "crop": CROP, "max_glyph": MAX_GLYPH}, args.out)
    print(f"SAVED {args.out} | final ep_ret(last50) {float(np.mean(rets[-50:])):+.3f}", flush=True)


if __name__ == "__main__":
    main()
