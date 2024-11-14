#!/bin/bash

set -euo pipefail

# -----------------------
# Configuration
# -----------------------

MEGATRON_CKPTS_DIR=./local/runs/towermoe-440M-1.2B/checkpoints
HF_CKPTS_DIR=./local/runs/towermoe-440M-1.2B/hf-checkpoints
HF_TOKENIZER_PATH=./local/tokenizers/eurollm-tokenizer

TP_SIZE=1 # Replace with actual tensor parallel size
PP_SIZE=1 # Replace with actual pipeline parallel size
EP_SIZE=1 # Replace with actual expert parallel size

# -----------------------
# Setup conda environment
# -----------------------

if [ -z "${CONDA_HOME}" ]; then
    echo "CONDA_HOME is unset"
    exit 1
fi
source ${CONDA_HOME}/etc/profile.d/conda.sh

conda activate ${ENV_NAME:-megatron-lm}

# -----------------------
# Convert
# -----------------------

# Add Megatron-LM to the PYTHONPATH
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MEGATRON_DIR="$( cd "$( dirname "$( dirname "${SCRIPT_DIR}" )" )" && pwd )"

if [ -z "${PYTHONPATH+x}" ]; then
    export PYTHONPATH="${MEGATRON_DIR}"
else
    export PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH}"
fi

mkdir -p "$HF_CKPTS_DIR"

# Loop over all iter_{iter num} subdirectories in the base directory
for LOAD_DIR in "$MEGATRON_CKPTS_DIR"/iter_*; do
    if [ -d "$LOAD_DIR" ]; then
        # Extract the iteration number
        ITER_NUM=$(basename "$LOAD_DIR" | sed 's/^iter_//')

        # Define the save directory for the converted model
        SAVE_DIR="$HF_CKPTS_DIR/$ITER_NUM"

        # Call the convert.py script with the required arguments
        python examples/mixtral/convert.py mcore-to-hf \
            --mcore-load-dir "$LOAD_DIR" \
            --hf-save-dir "$SAVE_DIR" \
            --hf-tokenizer-path "$HF_TOKENIZER_PATH" \
            --source-tensor-parallel-size "$TP_SIZE" \
            --source-pipeline-parallel-size "$PP_SIZE" \
            --source-expert-parallel-size "$EP_SIZE" \
            --target-params-dtype bfloat16 \
            --moe-grouped-gemm
    fi
done
