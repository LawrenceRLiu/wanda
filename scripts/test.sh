#!/bin/bash

# Set common variables
model="meta-llama/Llama-2-7b-hf"
sparsity_ratio=0.5
cuda_device=6

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    mkdir -p $3 && \
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --save $3 \
    > $3/log.log 2>&1
}

# llama-7b with wanda pruning method
echo "Running with wanda pruning method"
run_python_command "wanda" "unstructured" "out/llama-2-7b/unstructured/wanda/" 