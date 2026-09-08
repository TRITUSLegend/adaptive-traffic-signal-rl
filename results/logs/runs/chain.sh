#!/bin/sh
# Phase 3 completion chain: the three outstanding training runs, then the two
# evaluation sweeps that turn them into numbers. Sequential on purpose - the
# simulator is the bottleneck and two SUMO processes would just contend.
set -e
cd "C:/Users/Aditya Raj Kar/Downloads/Design of Smart Cities"
PY=.venv/Scripts/python.exe
OUT=results/logs/runs

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "$(stamp)  STEP 1/5  train dqn_seed2" >> $OUT/chain_status.txt
$PY agent/train_dqn.py --seed 2 --tag seed2 > $OUT/train_dqn_seed2.log 2>&1
echo "$(stamp)  done 1/5" >> $OUT/chain_status.txt

echo "$(stamp)  STEP 2/5  train dqn_mixed" >> $OUT/chain_status.txt
$PY agent/train_mixed.py > $OUT/train_dqn_mixed.log 2>&1
echo "$(stamp)  done 2/5" >> $OUT/chain_status.txt

echo "$(stamp)  STEP 3/5  train ppo_long (400 episodes)" >> $OUT/chain_status.txt
$PY agent/train_ppo.py --episodes 400 --tag long > $OUT/train_ppo_long.log 2>&1
echo "$(stamp)  done 3/5" >> $OUT/chain_status.txt

echo "$(stamp)  STEP 4/5  evaluate dqn_seed2 dqn_mixed ppo_long" >> $OUT/chain_status.txt
$PY eval/evaluate.py --controllers dqn_seed2 dqn_mixed ppo_long --append \
    > $OUT/eval_new_models.log 2>&1
echo "$(stamp)  done 4/5" >> $OUT/chain_status.txt

echo "$(stamp)  STEP 5/5  evaluate the +safe wrappers" >> $OUT/chain_status.txt
$PY eval/evaluate.py --controllers dqn+safe ppo+safe greedy-queue+safe fixed-time-tuned+safe \
    --append > $OUT/eval_safe.log 2>&1
echo "$(stamp)  done 5/5" >> $OUT/chain_status.txt

echo "$(stamp)  CHAIN COMPLETE" >> $OUT/chain_status.txt
