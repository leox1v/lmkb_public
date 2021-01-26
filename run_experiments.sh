#!/bin/bash

# PARAMS
# $1 output directory
session='experiments'
tmux new-session -d -s ${session}
GPUS=(0 1 2 3 4 5 6 7)
SEEDS=(0 1 2 3 4 5 6 7)
output_dir=$1

for i in "${!GPUS[@]}"; do
  if [ "$i" -ne "0" ]
  then
    tmux new-window
  fi
  tmux send-keys -t ${session}:${i}.0 "bash experiment.sh ${GPUS[i]} ${SEEDS[i]} $1" ENTER
done

tmux attach-session -t "$session"
