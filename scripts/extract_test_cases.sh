#!/bin/bash

METHOD_NAME=$1
SAVE_DIR=$2
EXTRACT_MODE=$3

python extract_all_test_cases.py \
    --logs_file="$SAVE_DIR/logs.json" \
    --output_file="$SAVE_DIR/test_cases.json" \
    --extract_mode="$EXTRACT_MODE"
