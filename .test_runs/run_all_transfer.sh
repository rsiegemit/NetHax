#!/bin/bash
# Train a PPO on each MiniHack-Room variant (real C env), then transfer-eval the
# policy on real MiniHack vs our Minihax (vec mode). Prints a RESULT row per env.
# Transfer holds if real_succ ~= mhx_succ (envs behave the same under the policy).
set -uo pipefail
cd /Users/rsiegelmann/Downloads/Projects/nethax

# real_class  minihax_name  train_steps
PAIRS=(
  "MiniHackRoom5x5         MiniHack-Room-5x5-v0          150000"
  "MiniHackRoom15x15       MiniHack-Room-15x15-v0        300000"
  "MiniHackRoom5x5Dark     MiniHack-Room-Dark-5x5-v0     250000"
  "MiniHackRoom15x15Dark   MiniHack-Room-Dark-15x15-v0   400000"
  "MiniHackRoom5x5Monster  MiniHack-Room-Monster-5x5-v0  250000"
  "MiniHackRoom15x15Monster MiniHack-Room-Monster-15x15-v0 400000"
  "MiniHackRoom5x5Random   MiniHack-Room-Random-5x5-v0   250000"
  "MiniHackRoom15x15Random MiniHack-Room-Random-15x15-v0 400000"
)

for row in "${PAIRS[@]}"; do
  set -- $row; REAL=$1; MHX=$2; STEPS=$3
  CKPT=".test_runs/ppo_${REAL}.pt"
  echo "######## $REAL -> $MHX (train $STEPS) ########"
  if [ ! -f "$CKPT" ]; then
    JAX_PLATFORMS=cpu .venv/bin/python .test_runs/ppo_minihack.py \
      --env "$REAL" --total-steps "$STEPS" --out "$CKPT" 2>/dev/null | tail -1
  else
    echo "  (ckpt exists, skip train)"
  fi
  JAX_PLATFORMS=cpu NETHAX_EAGER=1 PYTHONPATH=. .venv/bin/python \
    .test_runs/eval_transfer.py --ckpt "$CKPT" --real-env "$REAL" \
    --minihax-env "$MHX" --episodes 5 2>/dev/null | grep -E "REAL|Minihax|RESULT"
done
echo "######## ALL DONE ########"
