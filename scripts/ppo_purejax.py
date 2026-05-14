"""PureJaxRL-style PPO for NethaxEnv.

Validates that nethax is trainable end-to-end on a real GPU.  Reward is the
NLE-standard ``score_t - score_{t-1}`` delta (NetHack score lives at
``blstats[BL_SCORE]``).

Designed to be small (no external RL framework) and run start-to-finish in
under an hour on a single H100.  All control flow is jit-compatible —
``jax.lax.scan`` rollouts + ``jax.vmap`` envs, no Python loops in the hot path.

Run::

    cd ~/work/NetHax
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/ppo_purejax.py \
        --total-frames 5_000_000 --num-envs 256

Writes:
    bench/results/ppo_curve.json   step-vs-mean-reward
    bench/results/ppo_curve.png    matplotlib plot
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

# Ensure project root on sys.path when invoked as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from flax.training.train_state import TrainState
import optax

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants.blstats import BL_SCORE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="PureJaxRL PPO on NethaxEnv")
parser.add_argument("--total-frames", type=int, default=2_000_000)
parser.add_argument("--num-envs",     type=int, default=256)
parser.add_argument("--rollout-steps", type=int, default=64)
parser.add_argument("--num-epochs",   type=int, default=4)
parser.add_argument("--num-minibatches", type=int, default=4)
parser.add_argument("--lr",           type=float, default=2.5e-4)
parser.add_argument("--gamma",        type=float, default=0.99)
parser.add_argument("--gae-lambda",   type=float, default=0.95)
parser.add_argument("--clip-eps",     type=float, default=0.2)
parser.add_argument("--ent-coef",     type=float, default=0.01)
parser.add_argument("--vf-coef",      type=float, default=0.5)
parser.add_argument("--max-grad-norm", type=float, default=0.5)
parser.add_argument("--seed",         type=int, default=0)
parser.add_argument("--out-json", default="bench/results/ppo_curve.json")
parser.add_argument("--out-png",  default="bench/results/ppo_curve.png")
args = parser.parse_args()

NUM_ACTIONS = 121
GLYPH_VOCAB = 5977          # MAX_GLYPH + 1 to be safe for the no-glyph sentinel
EMBED_DIM   = 16
GLYPHS_H, GLYPHS_W = 21, 79
BLSTATS_LEN = 27
HIDDEN = 256


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, glyphs, blstats):
        emb = nn.Embed(num_embeddings=GLYPH_VOCAB, features=EMBED_DIM)(glyphs)
        # mean-pool the glyph embedding map → fixed-size vector
        emb_flat = emb.reshape(emb.shape[0], -1)
        bl = blstats.astype(jnp.float32) / 100.0
        x = jnp.concatenate([emb_flat, bl], axis=-1)
        x = nn.relu(nn.Dense(HIDDEN)(x))
        x = nn.relu(nn.Dense(HIDDEN)(x))
        logits = nn.Dense(self.num_actions)(x)
        value = nn.Dense(1)(x)[..., 0]
        return logits, value


# ---------------------------------------------------------------------------
# Env wrappers
# ---------------------------------------------------------------------------

def _obs_to_tensors(obs):
    """Pluck the (glyphs, blstats) slice the agent actually sees."""
    return obs["glyphs"].astype(jnp.int32), obs["blstats"].astype(jnp.int32)


def _reset_one(env, rng):
    state, obs = env.reset(rng)
    g, b = _obs_to_tensors(obs)
    return state, g, b, jnp.int32(obs["blstats"][BL_SCORE])


def _step_one(env, state, action, rng, prev_score):
    """One vmap-friendly env.step.

    NOTE: env.reset is *not* vmap-safe (character creation has Python-level
    int() casts).  So we don't auto-reset on done inside the rollout — the
    env's done flag is sticky, GAE handles the discount masking, and the
    outer loop re-seeds with a fresh sequential reset between training
    sessions.  Per-episode resets in the middle of training would require
    making create_character jit-friendly first (Wave 7 work).
    """
    new_state, obs, _r0, done, _info = env.step(state, action, rng)
    g, b = _obs_to_tensors(obs)
    score = jnp.int32(obs["blstats"][BL_SCORE])
    reward = (score - prev_score).astype(jnp.float32)
    # Mask reward to 0 on terminal step (we don't reset, so score wouldn't
    # actually drop — but we do want to stop accumulating reward on dead envs).
    reward = jnp.where(done, jnp.float32(0.0), reward)
    return new_state, g, b, score, reward, done


# ---------------------------------------------------------------------------
# Rollout + train step
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    glyphs:   jax.Array
    blstats:  jax.Array
    action:   jax.Array
    log_prob: jax.Array
    value:    jax.Array
    reward:   jax.Array
    done:     jax.Array


def make_train_fn(env):
    # env.step is fully jit-safe → safe under vmap.
    # env.reset is NOT vmap-safe (Python int() casts in create_character),
    # so the caller does N sequential resets and stacks the result manually.
    vstep = jax.vmap(_step_one, in_axes=(None, 0, 0, 0, 0))

    def rollout(carry, _):
        train_state, state, glyphs, blstats, score, rng = carry
        rng, sub = jax.random.split(rng)
        logits, value = train_state.apply_fn(train_state.params, glyphs, blstats)
        action = jax.random.categorical(sub, logits)
        log_prob = jax.nn.log_softmax(logits) \
                    [jnp.arange(args.num_envs), action]
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, args.num_envs)
        new_state, new_g, new_b, new_score, reward, done = vstep(
            env, state, action, step_rngs, score
        )
        tr = Transition(glyphs, blstats, action, log_prob, value, reward, done)
        return (train_state, new_state, new_g, new_b, new_score, rng), tr

    def gae(rewards, values, dones, last_value):
        def _step(carry, x):
            adv, _ = carry
            r, v_next, d, v = x
            delta = r + args.gamma * v_next * (1.0 - d) - v
            adv = delta + args.gamma * args.gae_lambda * (1.0 - d) * adv
            return (adv, v), adv
        # Build "next value" array: shift values up, last entry = bootstrap
        v_next = jnp.concatenate([values[1:], last_value[None]], axis=0)
        _, advs = jax.lax.scan(_step, (jnp.zeros_like(values[0]), last_value),
                               (rewards, v_next, dones, values),
                               reverse=True)
        returns = advs + values
        return advs, returns

    def ppo_loss(params, batch):
        glyphs, blstats, actions, old_logp, advs, returns = batch
        logits, values = jax.vmap(train_state_apply_for_loss, in_axes=(None, 0, 0))(
            params, glyphs, blstats
        )
        new_logp = jax.nn.log_softmax(logits)[jnp.arange(len(actions)), actions]
        ratio = jnp.exp(new_logp - old_logp)
        adv = (advs - advs.mean()) / (advs.std() + 1e-8)
        pg1 = ratio * adv
        pg2 = jnp.clip(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv
        pg_loss = -jnp.minimum(pg1, pg2).mean()
        v_loss = 0.5 * jnp.square(returns - values).mean()
        entropy = -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1).mean()
        loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy
        return loss, (pg_loss, v_loss, entropy)

    # vmap-friendly apply for the loss path
    def train_state_apply_for_loss(params, g, b):
        # g shape (H, W), b shape (BLSTATS,) — apply on a single sample
        logits, value = network.apply(params, g[None, ...], b[None, ...])
        return logits[0], value[0]

    @jax.jit
    def train_iter(train_state, env_carry):
        # 1. Rollout
        carry = (train_state,) + env_carry
        carry, traj = jax.lax.scan(rollout, carry, None, length=args.rollout_steps)
        train_state, state, g, b, score, rng = carry
        # bootstrap value for GAE
        _, last_value = train_state.apply_fn(train_state.params, g, b)

        # 2. GAE over (T, N) trajectory — reduce over T axis
        advs, returns = gae(traj.reward, traj.value, traj.done, last_value)

        # 3. Flatten (T, N, …) → (T*N, …) for minibatch SGD
        def _flat(x):
            return x.reshape((args.rollout_steps * args.num_envs,) + x.shape[2:])
        batch = (
            _flat(traj.glyphs), _flat(traj.blstats),
            _flat(traj.action), _flat(traj.log_prob),
            _flat(advs), _flat(returns),
        )

        # 4. Multi-epoch minibatch SGD
        def epoch_body(carry, _):
            ts, rng = carry
            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, args.rollout_steps * args.num_envs)
            mb_size = (args.rollout_steps * args.num_envs) // args.num_minibatches
            def mb_body(ts, mb_idx):
                idx = perm[mb_idx * mb_size:(mb_idx + 1) * mb_size]
                mb = tuple(b[idx] for b in batch)
                grads, aux = jax.grad(ppo_loss, has_aux=True)(ts.params, mb)
                ts = ts.apply_gradients(grads=grads)
                return ts, aux
            ts, _ = jax.lax.scan(mb_body, ts, jnp.arange(args.num_minibatches))
            return (ts, rng), None
        rng, epoch_rng = jax.random.split(rng)
        (train_state, _), _ = jax.lax.scan(
            epoch_body, (train_state, epoch_rng), None, length=args.num_epochs
        )

        # 5. Episode-return stats over rollout
        mean_step_reward = traj.reward.mean()
        return train_state, (state, g, b, score, rng), mean_step_reward

    return train_iter


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("PureJaxRL PPO on NethaxEnv")
    print("=" * 78)
    print(f"  device       : {jax.default_backend()} -> {jax.devices()}")
    print(f"  num_envs     : {args.num_envs}")
    print(f"  rollout_steps: {args.rollout_steps}")
    print(f"  total_frames : {args.total_frames:_}")
    print(f"  lr           : {args.lr}")
    print()

    env = NethaxEnv()

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    global network
    network = ActorCritic(num_actions=NUM_ACTIONS)
    dummy_g = jnp.zeros((1, GLYPHS_H, GLYPHS_W), dtype=jnp.int32)
    dummy_b = jnp.zeros((1, BLSTATS_LEN), dtype=jnp.int32)
    params = network.init(init_rng, dummy_g, dummy_b)

    n_params = sum(int(x.size) for x in jax.tree.leaves(params))
    print(f"  network params: {n_params:_}")

    tx = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.lr),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # Init parallel envs
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, args.num_envs)
    train_iter = make_train_fn(env)
    # Sequential reset (env.reset is not vmap-safe — see _step_one docstring).
    # One-shot cost: ~num_envs * (env JIT compile / num_envs) seconds.
    print(f"Resetting {args.num_envs} envs sequentially …")
    t_reset = time.perf_counter()
    states_l, g_l, b_l, sc_l = [], [], [], []
    for i in range(args.num_envs):
        s, gi, bi, sci = _reset_one(env, reset_rngs[i])
        states_l.append(s); g_l.append(gi); b_l.append(bi); sc_l.append(sci)
    state = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states_l)
    g = jnp.stack(g_l); b = jnp.stack(b_l); score = jnp.stack(sc_l)
    print(f"  done in {time.perf_counter() - t_reset:.1f}s")
    env_carry = (state, g, b, score, rng)

    # Warm-up compile
    t0 = time.perf_counter()
    print("Compiling train_iter (first call, may take 1-3 min) …")
    train_state, env_carry, mean_r = train_iter(train_state, env_carry)
    jax.block_until_ready(mean_r)
    print(f"  compile + first iter: {time.perf_counter() - t0:.1f}s, "
          f"mean step reward: {float(mean_r):.4f}")

    frames_per_iter = args.rollout_steps * args.num_envs
    n_iters = max(1, args.total_frames // frames_per_iter)
    print(f"  iters: {n_iters}, frames/iter: {frames_per_iter}")
    print()

    history = []
    t_start = time.perf_counter()
    for i in range(n_iters):
        train_state, env_carry, mean_r = train_iter(train_state, env_carry)
        mean_r_v = float(mean_r)
        frames = (i + 1) * frames_per_iter
        elapsed = time.perf_counter() - t_start
        sps = frames / elapsed if elapsed > 0 else 0.0
        history.append({
            "iter": i + 1,
            "frames": frames,
            "mean_step_reward": mean_r_v,
            "sps": sps,
        })
        if i % 5 == 0 or i == n_iters - 1:
            print(f"  iter {i+1:>4}/{n_iters}  "
                  f"frames {frames:>10,}  "
                  f"mean_r {mean_r_v:+.4f}  "
                  f"sps {sps:>10,.0f}")

    # Persist results
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": vars(args),
        "n_params": n_params,
        "history": history,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults: {out_json}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        frames = [h["frames"] for h in history]
        rewards = [h["mean_step_reward"] for h in history]
        plt.figure(figsize=(8, 4))
        plt.plot(frames, rewards, lw=1)
        plt.xlabel("frames")
        plt.ylabel("mean step reward (NLE score delta)")
        plt.title("Nethax PPO — single H100")
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=120)
        print(f"Plot:    {args.out_png}")
    except Exception as exc:
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
