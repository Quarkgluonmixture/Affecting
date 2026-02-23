#!/bin/bash
# Run all four FinQA evaluations and generate final report

set -e  # Exit on error

# Configuration
CACHE_DIR="/home/ec2-user/SageMaker/.cache/huggingface"
RESULTS_DIR="results"

# Create cache directory
mkdir -p "$CACHE_DIR"

echo "========================================"
echo "Starting FinQA Evaluation Pipeline"
echo "========================================"
echo ""

# Run 1: Qwen3-8B / oracle
echo "[1/4] Running Qwen/Qwen3-8B / oracle..."
python eval_finqa.py \
  --model_name Qwen/Qwen3-8B \
  --split test \
  --setting oracle \
  --cache_dir "$CACHE_DIR" \
  --max_new_tokens 32
echo "[1/4] Qwen/Qwen3-8B / oracle - DONE"
echo ""

# Run 2: Qwen3-8B / full
echo "[2/4] Running Qwen/Qwen3-8B / full..."
python eval_finqa.py \
  --model_name Qwen/Qwen3-8B \
  --split test \
  --setting full \
  --cache_dir "$CACHE_DIR" \
  --max_new_tokens 32
echo "[2/4] Qwen/Qwen3-8B / full - DONE"
echo ""

# Run 3: Qwen3-4B / oracle
echo "[3/4] Running Qwen/Qwen3-4B / oracle..."
python eval_finqa.py \
  --model_name Qwen/Qwen3-4B \
  --split test \
  --setting oracle \
  --cache_dir "$CACHE_DIR" \
  --max_new_tokens 32
echo "[3/4] Qwen/Qwen3-4B / oracle - DONE"
echo ""

# Run 4: Qwen3-4B / full
echo "[4/4] Running Qwen/Qwen3-4B / full..."
python eval_finqa.py \
  --model_name Qwen/Qwen3-4B \
  --split test \
  --setting full \
  --cache_dir "$CACHE_DIR" \
  --max_new_tokens 32
echo "[4/4] Qwen/Qwen3-4B / full - DONE"
echo ""

# Generate final report
echo "========================================"
echo "Generating final report..."
echo "========================================"
python generate_report.py --results_dir "$RESULTS_DIR" --output "$RESULTS_DIR/final_report.md"

echo ""
echo "========================================"
echo "All evaluations completed!"
echo "========================================"
echo "Report saved to: $RESULTS_DIR/final_report.md"
echo ""
