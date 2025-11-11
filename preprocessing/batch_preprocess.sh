#!/bin/bash
#
# Batch Preprocessing Script for Multiple Datasets
#
# This script preprocesses multiple datasets in sequence.
# Edit the DATASETS array below to customize which datasets to process.
#
# Usage:
#   bash batch_preprocess.sh [device]
#
# Example:
#   bash batch_preprocess.sh cuda:0
#   bash batch_preprocess.sh cpu

set -e  # Exit on error

# Configuration
DEVICE="${1:-cuda:0}"
MODEL="text_sonar_basic_encoder"
BATCH_SIZE=32

# Define datasets to process
# Format: "input_path:output_path"
DATASETS=(
    "data/raw/gaia_train.json:data/processed/gaia_train_with_concepts.pkl"
    "data/raw/gaia_val.json:data/processed/gaia_val_with_concepts.pkl"
    # Add more datasets here:
    # "data/raw/xbench.json:data/processed/xbench_with_concepts.pkl"
    # "data/raw/hotpotqa_train.json:data/processed/hotpotqa_train_with_concepts.pkl"
)

echo "========================================"
echo "Batch Preprocessing with SONAR"
echo "========================================"
echo "Device: $DEVICE"
echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Datasets to process: ${#DATASETS[@]}"
echo ""

# Check if preprocess_concepts.py exists
if [ ! -f "preprocess_concepts.py" ]; then
    echo "ERROR: preprocess_concepts.py not found in current directory"
    echo "Please run this script from the preprocessing/ directory"
    exit 1
fi

# Process each dataset
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_DATASETS=()

for dataset in "${DATASETS[@]}"; do
    # Split input:output
    IFS=':' read -r INPUT OUTPUT <<< "$dataset"

    echo "========================================"
    echo "Processing: $INPUT"
    echo "Output: $OUTPUT"
    echo "========================================"

    # Check if input exists
    if [ ! -f "$INPUT" ]; then
        echo "⚠ WARNING: Input file not found: $INPUT"
        echo "Skipping..."
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_DATASETS+=("$INPUT (not found)")
        echo ""
        continue
    fi

    # Check if output already exists
    if [ -f "$OUTPUT" ]; then
        echo "⚠ WARNING: Output file already exists: $OUTPUT"
        read -p "Overwrite? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping..."
            echo ""
            continue
        fi
    fi

    # Run preprocessing
    if python preprocess_concepts.py \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --model "$MODEL" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE"; then

        echo "✓ Success!"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # Run validation
        echo ""
        echo "Running validation..."
        if python validate_concepts.py --dataset "$OUTPUT"; then
            echo "✓ Validation passed!"
        else
            echo "⚠ WARNING: Validation failed"
            FAILED_DATASETS+=("$OUTPUT (validation failed)")
        fi
    else
        echo "❌ Failed to preprocess $INPUT"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_DATASETS+=("$INPUT (preprocessing error)")
    fi

    echo ""
done

# Summary
echo "========================================"
echo "BATCH PROCESSING SUMMARY"
echo "========================================"
echo "Total datasets: ${#DATASETS[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "Failed datasets:"
    for failed in "${FAILED_DATASETS[@]}"; do
        echo "  - $failed"
    done
fi

echo ""
if [ $FAIL_COUNT -eq 0 ]; then
    echo "✓ All datasets processed successfully!"
    exit 0
else
    echo "❌ Some datasets failed to process"
    exit 1
fi
