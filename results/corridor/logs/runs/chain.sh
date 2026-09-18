#!/usr/bin/env bash
# Corridor training chain: the three learned controllers, one seed each.
#
# Mirrors results/logs/runs/chain.sh from the single-intersection study: each
# step appends to chain_status.txt when it finishes, so an interrupted run can
# be resumed from the last completed step rather than from the top.
#
# A training run that is killed leaves no model (the save happens at the end,
# or on KeyboardInterrupt), so restart that step from the beginning. Its curve
# CSV is flushed every 10 episodes and is overwritten cleanly on restart.

set -u
cd "$(dirname "$0")/../../../.." || exit 1

PY=./.venv/Scripts/python.exe
RUNS=results/corridor/logs/runs
STATUS="$RUNS/chain_status.txt"
EPISODES=170

mkdir -p "$RUNS"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
note()  { echo "$(stamp)  $*" >> "$STATUS"; }

note "STEP 1/3  train idqn  ($EPISODES episodes)"
$PY corridor/train_independent.py --algo dqn --episodes $EPISODES \
    > "$RUNS/train_idqn.log" 2>&1
note "done 1/3 (exit $?)"

note "STEP 2/3  train ippo  ($EPISODES episodes)"
$PY corridor/train_independent.py --algo ppo --episodes $EPISODES \
    > "$RUNS/train_ippo.log" 2>&1
note "done 2/3 (exit $?)"

note "STEP 3/3  train ma2c  ($EPISODES episodes)"
$PY corridor/train_ma2c.py --episodes $EPISODES \
    > "$RUNS/train_ma2c.log" 2>&1
note "done 3/3 (exit $?)"

note "CHAIN COMPLETE"
