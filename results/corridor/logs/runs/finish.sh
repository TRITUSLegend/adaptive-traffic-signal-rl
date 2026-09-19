#!/usr/bin/env bash
# Finish the corridor pipeline: MA2C training, then the full evaluation.
#
# STRICTLY SEQUENTIAL. Running two SUMO workloads at once halved throughput
# earlier in this project, and an overlapping chain once wrote the same model
# path twice. Each step waits for the previous one.
#
# To stop this, kill the TREE from this script's bash pid - killing the python
# child alone makes the script advance to the next step, which is how a false
# "CHAIN COMPLETE" got recorded once:
#     taskkill //PID <pid of this script> //T //F

set -u
cd "$(dirname "$0")/../../../.." || exit 1

PY=./.venv/Scripts/python.exe
RUNS=results/corridor/logs/runs
STATUS="$RUNS/finish_status.txt"

mkdir -p "$RUNS"
note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >> "$STATUS"; }

# Every later step depends on the model or the data an earlier one produces, so
# a failure must stop the run rather than cascade down the list. When training
# died of system memory at episode 11, this script went on to start the
# evaluation - which needed the model training never saved - and would have
# failed its way through all four steps, burning time and writing a misleading
# trail. Now the first failure stops everything and says which step it was.
step() {
    local number="$1" label="$2" log="$3"
    shift 3
    note "STEP $number  $label"
    if "$@" > "$RUNS/$log" 2>&1; then
        note "done $number"
    else
        local code=$?
        note "FAILED $number (exit $code) - stopping; later steps depend on this one"
        note "see $RUNS/$log"
        exit "$code"
    fi
}

step "1/4" "train ma2c (170 episodes)" "train_ma2c.log" \
     $PY corridor/train_ma2c.py --episodes 170

step "2/4" "evaluate all six controllers, four scenarios, five seeds" "evaluate.log" \
     $PY corridor/evaluate_corridor.py

step "3/4" "detector-failure sweep" "robustness_detector.log" \
     $PY corridor/robustness.py --detector

step "4/4" "topology transfer" "robustness_transfer.log" \
     $PY corridor/robustness.py --transfer

note "PIPELINE COMPLETE"
