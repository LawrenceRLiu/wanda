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
    --eval_zero_shot_tasks None \
    > $3/log.log 2>&1
}

patterns=("256:512"
            "128:256"
            "64:128"
            "32:64"
            "16:32"
            "8:16"
)

for pattern in "${patterns[@]}"; do
    echo "Running with sparsity type: $pattern"
    run_python_command "sparsegpt" "$pattern" "out/llama-2-7b/$pattern/sparsegpt/"
    run_python_command "wanda" "$pattern" "out/llama-2-7b/$pattern/wanda/"
done

# # llama-7b with wanda pruning method
# echo "Running with wanda pruning method"
# run_python_command "wanda" "unstructured" "out/llama-2-7b/unstructured/wanda/" 
# run_python_command "wanda" "2:4" "out/llama-2-7b/2-4/wanda/" 
# run_python_command "wanda" "4:8" "out/llama-2-7b/4-8/wanda/" 
# echo "Finished wanda pruning method"

# # llama-7b with sparsegpt pruning method
# echo "Running with sparsegpt pruning method"
# run_python_command "sparsegpt" "unstructured" "out/llama-2-7b/unstructured/sparsegpt/" 
# run_python_command "sparsegpt" "2:4" "out/llama-2-7b/2-4/sparsegpt/" 
# run_python_command "sparsegpt" "4:8" "out/llama-2-7b/4-8/sparsegpt/" 
# echo "Finished sparsegpt pruning method"

# # llama-7b with magnitude pruning method
# echo "Running with magnitude pruning method"
# run_python_command "magnitude" "unstructured" "out/llama-2-7b/unstructured/magnitude/" 
# run_python_command "magnitude" "2:4" "out/llama-2-7b/2-4/magnitude/" 
# run_python_command "magnitude" "4:8" "out/llama-2-7b/4-8/magnitude/" 
# echo "Finished magnitude pruning method"