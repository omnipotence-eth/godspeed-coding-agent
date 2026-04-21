#!/usr/bin/env bash
#
# run_local_swebench_eval.sh
#
# Run the official SWE-bench evaluation harness against the staged
# leaderboard submission's predictions, producing the per-instance
# logs/ directory required by SWE-bench/experiments.
#
# Context:
#   * The SWE-bench harness imports Python's `resource` module, which
#     is Unix-only. On Windows we invoke it through WSL Ubuntu.
#   * Docker must be running and accessible from WSL.
#   * First-run downloads Docker images for each affected repo / base
#     commit. Disk budget: ~20-50 GB for 23 Lite dev instances.
#   * Wall time: ~1-2 hours for all 23 instances at max_workers=4 on a
#     recent laptop; scales with worker count up to available CPU.
#
# Output layout (after success):
#   <SUBMISSION_DIR>/logs/<instance_id>/patch.diff
#   <SUBMISSION_DIR>/logs/<instance_id>/report.json
#   <SUBMISSION_DIR>/logs/<instance_id>/test_output.txt
#   <SUBMISSION_DIR>/logs/<instance_id>/run_instance.log    (optional)
#
# Once logs/ is populated, the submission directory satisfies every
# field on the SWE-bench/experiments checklist.md and can be PR'd
# upstream via the instructions in leaderboard_submission/README.md.
#
# Usage:
#   bash run_local_swebench_eval.sh [--max-workers N] [--dry-run]
#
# Defaults: --max-workers 4, not --dry-run. Edit paths below if you
# ever rename the submission directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_DIR="$SCRIPT_DIR/evaluation/lite/20260421_godspeed_v3_2_judge"
PRED_PATH="$SUBMISSION_DIR/all_preds.jsonl"
RUN_ID="godspeed_v3_2_judge_local_$(date +%Y%m%d-%H%M%S)"
MAX_WORKERS="${MAX_WORKERS:-4}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '1,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

echo "=== Godspeed v3.2 local SWE-bench eval ==="
echo "Submission dir: $SUBMISSION_DIR"
echo "Predictions:    $PRED_PATH"
echo "Run ID:         $RUN_ID"
echo "Max workers:    $MAX_WORKERS"

# Pre-flight checks
if [[ ! -f "$PRED_PATH" ]]; then
  echo "ERROR: predictions file not found at $PRED_PATH" >&2
  echo "Regenerate by copying predictions_judge_merged_5way.jsonl into the submission dir." >&2
  exit 2
fi

if ! command -v wsl >/dev/null 2>&1; then
  echo "ERROR: wsl not on PATH. This script requires WSL Ubuntu on Windows." >&2
  echo "On Linux, invoke the swebench harness directly instead:" >&2
  echo "  python3 -m swebench.harness.run_evaluation --predictions_path \"$PRED_PATH\" --run_id \"$RUN_ID\" --dataset_name SWE-bench/SWE-bench_Lite --split dev --max_workers $MAX_WORKERS" >&2
  exit 2
fi

# Verify WSL environment has docker + swebench
WSL_CHECK=$(wsl -d Ubuntu -e bash -c 'command -v docker >/dev/null && python3 -c "import swebench" 2>/dev/null && echo OK || echo MISSING' 2>&1 | tr -d '\r')
if [[ "$WSL_CHECK" != "OK" ]]; then
  echo "ERROR: WSL Ubuntu missing docker or swebench. Got: $WSL_CHECK" >&2
  echo "Install inside WSL:" >&2
  echo "  wsl -d Ubuntu -e bash -c 'sudo apt install -y docker.io && pip install swebench'" >&2
  exit 2
fi

# Verify Docker daemon is reachable from WSL
if ! wsl -d Ubuntu -e bash -c 'docker ps >/dev/null 2>&1'; then
  echo "ERROR: docker ps failed from WSL. Is Docker Desktop running with WSL integration enabled?" >&2
  exit 2
fi

# Convert a Windows or git-bash path to a WSL path.
#   C:\Users\foo     -> /mnt/c/Users/foo
#   c:/Users/foo     -> /mnt/c/Users/foo
#   /c/Users/foo     -> /mnt/c/Users/foo    (git-bash MSYS style)
#   /mnt/c/Users/foo -> /mnt/c/Users/foo    (already WSL)
#   /non/windows     -> /non/windows
win_to_wsl() {
  local p="$1"
  p="${p//\\/\/}"                     # backslashes -> forward slashes
  if [[ "$p" =~ ^/mnt/[a-z]/ ]]; then
    printf '%s' "$p"
    return
  fi
  if [[ "$p" =~ ^/([a-zA-Z])/(.*) ]]; then
    local drive="${BASH_REMATCH[1]}"
    local rest="${BASH_REMATCH[2]}"
    printf '/mnt/%s/%s' "$(echo "$drive" | tr '[:upper:]' '[:lower:]')" "$rest"
    return
  fi
  if [[ "$p" =~ ^([A-Za-z]):/(.*) ]]; then
    local drive="${BASH_REMATCH[1]}"
    local rest="${BASH_REMATCH[2]}"
    printf '/mnt/%s/%s' "$(echo "$drive" | tr '[:upper:]' '[:lower:]')" "$rest"
    return
  fi
  printf '%s' "$p"
}

WSL_SUBMISSION_DIR="$(win_to_wsl "$SUBMISSION_DIR")"
WSL_PRED_PATH="$(win_to_wsl "$PRED_PATH")"

echo ""
echo "WSL paths:"
echo "  Submission dir: $WSL_SUBMISSION_DIR"
echo "  Predictions:    $WSL_PRED_PATH"

if [[ $DRY_RUN -eq 1 ]]; then
  echo ""
  echo "=== Dry run complete — all prerequisites met ==="
  echo ""
  echo "To execute, re-run without --dry-run. Expected wall time ~1-2 hr."
  exit 0
fi

# Run the harness from inside WSL, writing logs into a workspace adjacent
# to the submission dir (swebench writes to ./logs/run_evaluation/<run_id>/).
WSL_WORKSPACE="$WSL_SUBMISSION_DIR/.harness_workspace"
wsl -d Ubuntu -e bash -c "
  set -euo pipefail
  mkdir -p '$WSL_WORKSPACE'
  cd '$WSL_WORKSPACE'
  python3 -m swebench.harness.run_evaluation \
    --predictions_path '$WSL_PRED_PATH' \
    --run_id '$RUN_ID' \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --split dev \
    --max_workers $MAX_WORKERS \
    --cache_level env
"

# Copy the per-instance artifacts into the leaderboard submission layout.
# swebench writes each instance to:
#   <WSL_WORKSPACE>/logs/run_evaluation/<run_id>/<model_name>/<instance_id>/
# We want:
#   <SUBMISSION_DIR>/logs/<instance_id>/
mkdir -p "$SUBMISSION_DIR/logs"

wsl -d Ubuntu -e bash -c "
  set -euo pipefail
  RUN_LOGS=\$(find '$WSL_WORKSPACE/logs/run_evaluation/$RUN_ID' -mindepth 2 -maxdepth 2 -type d | head -1)
  if [[ -z \"\$RUN_LOGS\" ]]; then
    echo 'ERROR: no per-instance log directory produced' >&2
    exit 1
  fi
  echo 'Copying artifacts from:'
  echo \"  \$RUN_LOGS\"
  echo 'to:'
  echo \"  $WSL_SUBMISSION_DIR/logs/\"
  cp -r \"\$RUN_LOGS\"/* '$WSL_SUBMISSION_DIR/logs/'
"

echo ""
echo "=== Done ==="
echo ""
echo "Inspect:"
echo "  ls $SUBMISSION_DIR/logs/ | head -5"
echo ""
echo "Once verified, the submission directory is checklist-complete."
echo "Proceed with the upstream PR workflow in leaderboard_submission/README.md."
