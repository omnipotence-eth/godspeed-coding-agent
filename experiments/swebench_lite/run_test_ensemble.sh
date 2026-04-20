#!/usr/bin/env bash
# 2-driver ensemble headline: Kimi K2.5 + GPT-OSS-120B on SWE-Bench Lite TEST
# (50-instance subset for session-feasible wall time).
#
# Strategy: run both drivers as separate sb-cli submissions. The "ensemble
# headline" is the UNION of resolved instances across both. This is legit
# for benchmark reporting as long as we disclose both runs were used.
#
# Quota: uses 2 of 10 test-split slots.

set -euo pipefail

# Load env
set -a
source ~/.godspeed/.env.local
set +a
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

SBCLI="C:/Users/ttimm/miniconda3/envs/mlenv/Scripts/sb-cli.exe"
cd "$(dirname "$0")/../.."

mkdir -p experiments/swebench_lite/test_split

run_one () {
  local MODEL="$1"
  local LABEL="$2"
  local OUT="experiments/swebench_lite/test_split/predictions_${LABEL}.jsonl"
  local METRICS="experiments/swebench_lite/test_split/run_metrics_${LABEL}.jsonl"
  local RUN_ID="godspeed-v2_11_0-${LABEL}"

  echo "=== ${LABEL}: producing predictions on 50 test instances ==="
  python experiments/swebench_lite/run.py \
      --model "$MODEL" \
      --split test \
      --limit 50 \
      --out "$OUT" \
      --metrics "$METRICS" \
      --resume

  echo "=== ${LABEL}: submitting to sb-cli (${RUN_ID}) ==="
  "$SBCLI" submit swe-bench_lite test \
      --predictions_path "$OUT" \
      --run_id "$RUN_ID" \
      --output_dir experiments/swebench_lite/reports/

  echo "=== ${LABEL}: fetching report ==="
  "$SBCLI" get-report swe-bench_lite test "$RUN_ID" \
      -o experiments/swebench_lite/reports/ --overwrite 1
}

# Run both drivers. First one may already be in flight (seed 1).
# Script is idempotent via --resume.
run_one "nvidia_nim/moonshotai/kimi-k2.5"     "kimi-k2_5-test50-seed1"
run_one "nvidia_nim/openai/gpt-oss-120b"      "gpt-oss-120b-test50-seed1"

echo ""
echo "=== union analysis ==="
python -c "
import json
a=json.load(open('experiments/swebench_lite/reports/swe-bench_lite__test__godspeed-v2_11_0-kimi-k2_5-test50-seed1.json'))
b=json.load(open('experiments/swebench_lite/reports/swe-bench_lite__test__godspeed-v2_11_0-gpt-oss-120b-test50-seed1.json'))
ar = set(a['resolved_ids']); br = set(b['resolved_ids'])
print(f'Kimi:    {len(ar)}/{a[\"total_instances\"]} = {100*len(ar)/a[\"total_instances\"]:.1f}%')
print(f'GPT-OSS: {len(br)}/{b[\"total_instances\"]} = {100*len(br)/b[\"total_instances\"]:.1f}%')
print(f'Union:   {len(ar|br)}/{a[\"total_instances\"]} = {100*len(ar|br)/a[\"total_instances\"]:.1f}%')
print(f'Intersection: {len(ar&br)}')
print(f'Kimi-only:  {sorted(ar-br)[:5]}{\" ...\" if len(ar-br)>5 else \"\"}')
print(f'GPT-OSS-only: {sorted(br-ar)[:5]}{\" ...\" if len(br-ar)>5 else \"\"}')
"
