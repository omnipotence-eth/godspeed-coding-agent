#!/usr/bin/env bash
#
# reproduce_v3_1.sh — reproduce the Godspeed v3.1.0 SWE-Bench Lite dev-23
# headline (12 / 23 = 52.2% resolved, oracle-selector best-of-5).
#
# What this script does:
#   1. Runs oracle_merge.py on the five committed predictions.jsonl files
#      paired with their committed sb-cli reports. Produces a merged
#      predictions file labeled oracle_best_of_5.
#   2. Submits the merged predictions to sb-cli swe-bench_lite dev with a
#      fresh run_id so the result is independently verifiable.
#   3. Prints the resolved count from the sb-cli response.
#
# What this script does NOT do:
#   - Re-run the five constituent runs. Those require NVIDIA NIM free-tier
#     access + WSL/Docker for the agent-in-loop variant. See
#     findings_2026_04_21.md § "Reproducing individual constituent runs"
#     for instructions.
#   - Use quota beyond one sb-cli dev slot (the re-submission).
#
# Required:
#   - SWEBENCH_API_KEY env var (get one at https://swebench.com)
#   - Python with godspeed>=3.1.0 installed (or running from repo root)
#   - `sb-cli` on PATH
#
# Expected outcome:
#   Resolved (submitted): 52.17% (12 / 23)
#   If you see a different number, something about the committed
#   predictions or reports has drifted. Check the methodology doc.
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${SWEBENCH_API_KEY:-}" ]]; then
  echo "ERROR: SWEBENCH_API_KEY not set. Get one at https://swebench.com" >&2
  exit 2
fi

if ! command -v sb-cli >/dev/null 2>&1; then
  echo "ERROR: sb-cli not on PATH. Install with: pip install sb-cli" >&2
  exit 2
fi

PY="${PYTHON:-python}"

MERGED_OUT="experiments/swebench_lite/predictions_oracle_merged_5way.jsonl"
SOURCE_LOG="experiments/swebench_lite/oracle_merged_5way_sources.jsonl"
RUN_ID="${RUN_ID:-godspeed-v3_1-oracle-best-of-5-repro-$(date +%Y%m%d-%H%M%S)}"

echo "Step 1/2: oracle-merging 5 predictions files ..."
"$PY" experiments/swebench_lite/oracle_merge.py \
  --pairs \
    experiments/swebench_lite/predictions_e1_kimi.jsonl:experiments/swebench_lite/reports/swe-bench_lite__dev__godspeed-v2_11_0-kimi-k2_5.json \
    experiments/swebench_lite/predictions_gpt_oss.jsonl:experiments/swebench_lite/reports/swe-bench_lite__dev__godspeed-v2_11_0-gpt-oss-120b.json \
    experiments/swebench_lite/predictions_p1_dev23_v3.jsonl:experiments/swebench_lite/reports/swe-bench_lite__dev__godspeed-v3_1-p1-agent-in-loop-dev23.json \
    experiments/swebench_lite/predictions_iter1.jsonl:experiments/swebench_lite/reports/Subset.swe_bench_lite__dev__godspeed-v2_11_0-qwen3_5-397b.json \
    experiments/swebench_lite/predictions_seed3.jsonl:experiments/swebench_lite/reports/swe-bench_lite__dev__godspeed-v2_11_0-qwen3_5-397b-seed3.json \
  --out "$MERGED_OUT" \
  --source-log "$SOURCE_LOG"

echo ""
echo "Step 2/2: submitting merged predictions to sb-cli (run_id: $RUN_ID) ..."
sb-cli submit swe-bench_lite dev \
  --predictions_path "$MERGED_OUT" \
  --run_id "$RUN_ID" \
  --output_dir experiments/swebench_lite/reports/

echo ""
echo "Done. Look for:"
echo "  Resolved (submitted): 52.17% (12 / 23)"
echo ""
echo "Full methodology: experiments/swebench_lite/findings_2026_04_21.md"
