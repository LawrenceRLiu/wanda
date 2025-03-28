#!/bin/bash

# Set common variables
model="meta-llama/Meta-Llama-3-70B"
sparsity_ratio=0.5
cuda_device=0,1,2,3,4,5,6,7,8,9

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    mkdir -p $3 && \
    python -u main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --save $3 \
    > $3/log.log 2>&1
}

# llama-7b with wanda pruning method
echo "Running with wanda pruning method"
run_python_command "wanda" "unstructured" "out/Meta-Llama-3-70B/unstructured/wanda/" 
run_python_command "wanda" "2:4" "out/Meta-Llama-3-70B/2-4/wanda/" 
run_python_command "wanda" "4:8" "out/Meta-Llama-3-70B/4-8/wanda/" 
echo "Finished wanda pruning method"

# llama-7b with sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "out/Meta-Llama-3-70B/unstructured/sparsegpt/" 
run_python_command "sparsegpt" "2:4" "out/Meta-Llama-3-70B/2-4/sparsegpt/" 
run_python_command "sparsegpt" "4:8" "out/Meta-Llama-3-70B/4-8/sparsegpt/" 
echo "Finished sparsegpt pruning method"

# # llama-7b with magnitude pruning method
# echo "Running with magnitude pruning method"
# run_python_command "magnitude" "unstructured" "out/Meta-Llama-3-70B/unstructured/magnitude/" 
# run_python_command "magnitude" "2:4" "out/Meta-Llama-3-70B/2-4/magnitude/" 
# run_python_command "magnitude" "4:8" "out/Meta-Llama-3-70B/4-8/magnitude/" 
# echo "Finished magnitude pruning method"