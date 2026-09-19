#!/usr/bin/env bash
# The two robustness sweeps. Steps 1 and 2 of finish.sh (MA2C training and the
# clean evaluation) are already done and must NOT be repeated - training has no
# resume, so re-running finish.sh would throw away a 109-minute run.
#
# Both sweeps here save after every episode and skip what is already recorded,
# so this script is safe to re-run after an interruption: it picks up where it
# stopped rather than starting over.
#
# Stop it by killing the TREE from this script's bash pid, never the python
# child alone:
#     taskkill //PID <pid of this script> //T //F

set -u
cd "$(dirname "$0")/../../../.." || exit 1

PY=./.venv/Scripts/python.exe
RUNS=results/corridor/logs/runs
STATUS="$RUNS/sweeps_status.txt"

mkdir -p "$RUNS"
note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >> "$STATUS"; }

step() {
    local number="$1" label="$2" log="$3"
    shift 3
    note "STEP $number  $label"
    if "$@" > "$RUNS/$log" 2>&1; then
        note "done $number"
    else
        local code=$?
        note "FAILED $number (exit $code) - stopping. Re-running this script"
        note "  resumes from the episodes already saved."
        note "  see $RUNS/$log"
        exit "$code"
    fi
}

step "1/2" "detector-failure sweep (resumable)" "robustness_detector.log" \
     $PY corridor/robustness.py --detector

step "2/2" "topology transfer (resumable)" "robustness_transfer.log" \
     $PY corridor/robustness.py --transfer

note "SWEEPS COMPLETE"
