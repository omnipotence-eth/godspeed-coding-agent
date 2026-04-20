#!/usr/bin/env bash
# Run the winning dev-split configuration against SWE-Bench Lite TEST split,
# N=3 seeds, for the headline number. Uses 3 of 10 sb-cli test-split quota.
#
# Fill WINNING_MODEL + AGENT_STACK_FLAGS after dev-split experiments complete.
#
# Usage:
#     bash experiments/swebench_lite/run_test_headline.sh

set -euo pipefail

# Load env (NVIDIA_NIM_API_KEY, SWEBENCH_API_KEY)
set -a
source ~/.godspeed/.env.local
set +a
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

# ---- CONFIGURED AFTER DEV EXPERIMENTS (2026-04-20) ----
# Dev-split sb-cli results:
#   kimi-k2.5                   -> 8/23 = 34.8% (winner)
#   qwen3.5-397b iter1          -> 6/23 = 26.1%
#   gpt-oss-120b                -> 6/23 = 26.1%
#   qwen3.5-397b seed3          -> 5/23 = 21.7%
#   qwen3.5-397b seed2          -> 4/23 = 17.4%
#   best-of-3 (Qwen3.5)         -> 4/23 = 17.4%
#   qwen3-next-thinking + stack -> 3/23 = 13.0% (agent stack hurt)
WINNING_MODEL="nvidia_nim/moonshotai/kimi-k2.5"
AGENT_FLAGS=""  # agent stack underperformed baseline on dev — keeping plain
SBCLI="C:/Users/ttimm/miniconda3/envs/mlenv/Scripts/sb-cli.exe"
RUN_NAME_PREFIX="godspeed-v2_11_0-kimi-k2_5"
# ------------------------------------------

cd "$(dirname "$0")/../.."

for seed in 1 2 3; do
  OUT="experiments/swebench_lite/test_split/predictions_${RUN_NAME_PREFIX}_seed${seed}.jsonl"
  METRICS="experiments/swebench_lite/test_split/run_metrics_${RUN_NAME_PREFIX}_seed${seed}.jsonl"
  RUN_ID="${RUN_NAME_PREFIX}-seed${seed}"

  echo "=== Seed ${seed}: producing predictions ==="
  mkdir -p experiments/swebench_lite/test_split
  python experiments/swebench_lite/run.py \
      --model "$WINNING_MODEL" \
      --split test \
      $AGENT_FLAGS \
      --out "$OUT" \
      --metrics "$METRICS" \
      --resume

  echo "=== Seed ${seed}: submitting to sb-cli (${RUN_ID}) ==="
  "$SBCLI" submit swe-bench_lite test \
      --predictions_path "$OUT" \
      --run_id "$RUN_ID" \
      --output_dir experiments/swebench_lite/reports/

  echo "=== Seed ${seed}: fetching report ==="
  "$SBCLI" get-report swe-bench_lite test "$RUN_ID" \
      -o experiments/swebench_lite/reports/ --overwrite 1
done

echo "Done. Compute mean ± std across the three seed reports for the headline."
