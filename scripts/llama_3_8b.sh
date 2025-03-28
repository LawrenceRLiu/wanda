!/bin/bash

# Set common variables
model="meta-llama/Meta-Llama-3-8B"
sparsity_ratio=0.5
cuda_device=7
export HF_DATASETS_TRUST_REMOTE_CODE=1

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
run_python_command "wanda" "unstructured" "out/llama-3-8b/unstructured/wanda/" 
run_python_command "wanda" "2:4" "out/llama-3-8b/2-4/wanda/" 
run_python_command "wanda" "4:8" "out/llama-3-8b/4-8/wanda/" 
echo "Finished wanda pruning method"

# llama-7b with sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "out/llama-3-8b/unstructured/sparsegpt/" 
run_python_command "sparsegpt" "2:4" "out/llama-3-8b/2-4/sparsegpt/" 
run_python_command "sparsegpt" "4:8" "out/llama-3-8b/4-8/sparsegpt/" 
echo "Finished sparsegpt pruning method"

# llama-7b with magnitude pruning method
# echo "Running with magnitude pruning method"
# run_python_command "magnitude" "unstructured" "out/llama-3-8b/unstructured/magnitude/" 
# run_python_command "magnitude" "2:4" "out/llama-3-8b/2-4/magnitude/" 
# run_python_command "magnitude" "4:8" "out/llama-3-8b/4-8/magnitude/" 
# echo "Finished magnitude pruning method"