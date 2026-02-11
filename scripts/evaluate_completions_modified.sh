#!/bin/bash

CLS_PATH=$1
BEHAVIORS_PATH=$2
COMPLETIONS_PATH=$3
SAVE_PATH=$4
USE_ADVBENCH_FIRST=$5
INCLUDE_ADVBENCH_METRIC=$6
NUM_TOKENS=$7
SEED=$8
SAVE_FIRST_SUCCESS_ONLY=$9

ADVBENCH_FIRST_FLAG=""
if [ "$USE_ADVBENCH_FIRST" = "True" ]; then
    ADVBENCH_FIRST_FLAG="--use_advbench_first"
fi

ADVBENCH_METRIC_FLAG=""
if [ "$INCLUDE_ADVBENCH_METRIC" = "True" ]; then
    ADVBENCH_METRIC_FLAG="--include_advbench_metric"
fi

SAVE_FIRST_SUCCESS_FLAG=""
if [ "$SAVE_FIRST_SUCCESS_ONLY" = "True" ]; then
    SAVE_FIRST_SUCCESS_FLAG="--save_first_success_only"
fi

python evaluate_completions_modified_GPT.py \
    --cls_path="$CLS_PATH" \
    --behaviors_path="$BEHAVIORS_PATH" \
    --completions_path="$COMPLETIONS_PATH" \
    --save_path="$SAVE_PATH" \
    --num_tokens="$NUM_TOKENS" \
    --seed="$SEED" \
    $ADVBENCH_FIRST_FLAG \
    $ADVBENCH_METRIC_FLAG \
    $SAVE_FIRST_SUCCESS_FLAG
