#!/bin/bash

# PARAMS
# $1 GPU device number
# $2 Random seed
# $3 Output directory

output_dir=$3
common_vocab='--nouse_common_vocab'
save_cls_embedding='--save_cls_embedding'
additional_params='--num_predictions 10'

 for data in "Google_RE" "TREx" "ConceptNet" "BATS"
do
  for model in "albert-xxlarge-v2" "bert-base-cased" "bert-large-cased" "t5-large"
  do
    for num_priming_examples in 0 1 3 5 10 15 20
    do
      for priming in "nl" "{} -> {}" "{} => {}" "({}; {})"
      do
        for use_close_examples in "--use_close_examples" "--nouse_close_examples"
        do
          CMD="CUDA_VISIBLE_DEVICES=$1 python main.py --output_dir $output_dir --data $data --model $model --num_priming_examples $num_priming_examples --priming_type '$priming' --random_seed $2 $use_close_examples --batch_size 32 $common_vocab $save_cls_embedding $num_predictions"
          echo $CMD
          bash -c "$CMD"
        done
      done
    done
  done
done
