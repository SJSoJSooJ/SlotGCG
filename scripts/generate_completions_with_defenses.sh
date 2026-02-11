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
DEFENSE_TYPE=${11}
SMOOTHLLM_PERT_TYPE=${12}
SMOOTHLLM_PERT_PCT=${13}
SMOOTHLLM_NUM_COPIES=${14}
PERPLEXITY_MODEL_PATH=${15}
PERPLEXITY_THRESHOLD=${16}
ERASE_AND_CHECK_MODE=${17}
ERASE_AND_CHECK_MAX_ERASE=${18}
ERASE_AND_CHECK_NUM_ADV=${19}
SAFE_DECODING_LORA_PATH=${20}
SAFE_DECODING_ALPHA=${21}
SAFE_DECODING_FIRST_M=${22}
SAFE_DECODING_TOP_K=${23}
SAFE_DECODING_NUM_COMMON_TOKENS=${24}

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

PERPLEXITY_THRESHOLD_FLAG=""
if [ "$PERPLEXITY_THRESHOLD" != "None" ]; then
    PERPLEXITY_THRESHOLD_FLAG="--perplexity_threshold $PERPLEXITY_THRESHOLD"
fi

python generate_completions_with_defenses.py \
    --model_name="$MODEL_NAME" \
    --behaviors_path="$BEHAVIORS_PATH" \
    --test_cases_path="$TEST_CASES_PATH" \
    --save_path="$SAVE_PATH" \
    --max_new_tokens="$MAX_NEW_TOKENS" \
    --batch_size="$BATCH_SIZE" \
    --max_regeneration_tokens="$MAX_REGENERATION_TOKENS" \
    --defense_type="$DEFENSE_TYPE" \
    --smoothllm_pert_type="$SMOOTHLLM_PERT_TYPE" \
    --smoothllm_pert_pct="$SMOOTHLLM_PERT_PCT" \
    --smoothllm_num_copies="$SMOOTHLLM_NUM_COPIES" \
    --perplexity_model_path="$PERPLEXITY_MODEL_PATH" \
    --erase_and_check_mode="$ERASE_AND_CHECK_MODE" \
    --erase_and_check_max_erase="$ERASE_AND_CHECK_MAX_ERASE" \
    --erase_and_check_num_adv="$ERASE_AND_CHECK_NUM_ADV" \
    --safe_decoding_lora_path="$SAFE_DECODING_LORA_PATH" \
    --safe_decoding_alpha="$SAFE_DECODING_ALPHA" \
    --safe_decoding_first_m="$SAFE_DECODING_FIRST_M" \
    --safe_decoding_top_k="$SAFE_DECODING_TOP_K" \
    --safe_decoding_num_common_tokens="$SAFE_DECODING_NUM_COMMON_TOKENS" \
    $INCREMENTAL_FLAG \
    $VLLM_FLAG \
    $MAX_REGEN_FLAG \
    $PERPLEXITY_THRESHOLD_FLAG
