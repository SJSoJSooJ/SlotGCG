#!/bin/bash

MODEL_NAME=$1
BEHAVIORS_PATH=$2
TEST_CASES_PATH=$3
SAVE_PATH=$4
MAX_NEW_TOKENS=$5
INCREMENTAL_UPDATE=$6
BATCH_SIZE=$7
GENERATE_WITH_VLLM=$8
ENABLE_MAX_REGEN=$9
MAX_REGENERATION_TOKENS=${10}

INCREMENTAL_FLAG=""
if [ "$INCREMENTAL_UPDATE" = "True" ]; then
    INCREMENTAL_FLAG="--incremental_update"
fi

VLLM_FLAG=""
if [ "$GENERATE_WITH_VLLM" = "True" ]; then
    VLLM_FLAG="--generate_with_vllm"
fi

MAX_REGEN_FLAG=""
if [ "$ENABLE_MAX_REGEN" = "True" ]; then
    MAX_REGEN_FLAG="--enable_max_length_regeneration"
fi

python generate_completions_modified.py \
    --model_name="$MODEL_NAME" \
    --behaviors_path="$BEHAVIORS_PATH" \
    --test_cases_path="$TEST_CASES_PATH" \
    --save_path="$SAVE_PATH" \
    --max_new_tokens="$MAX_NEW_TOKENS" \
    --batch_size="$BATCH_SIZE" \
    --max_regeneration_tokens="$MAX_REGENERATION_TOKENS" \
    $INCREMENTAL_FLAG \
    $VLLM_FLAG \
    $MAX_REGEN_FLAG
