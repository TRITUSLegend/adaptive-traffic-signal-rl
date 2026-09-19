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

note "STEP 1/4  train ma2c (170 episodes)"
$PY corridor/train_ma2c.py --episodes 170 > "$RUNS/train_ma2c.log" 2>&1
note "done 1/4 (exit $?)"

note "STEP 2/4  evaluate all six controllers, four scenarios, five seeds"
$PY corridor/evaluate_corridor.py > "$RUNS/evaluate.log" 2>&1
note "done 2/4 (exit $?)"

note "STEP 3/4  detector-failure sweep"
$PY corridor/robustness.py --detector > "$RUNS/robustness_detector.log" 2>&1
note "done 3/4 (exit $?)"

note "STEP 4/4  topology transfer"
$PY corridor/robustness.py --transfer > "$RUNS/robustness_transfer.log" 2>&1
note "done 4/4 (exit $?)"

note "PIPELINE COMPLETE"
