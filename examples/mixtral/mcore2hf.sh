#!/bin/bash

set -euo pipefail

# -----------------------
# Configuration
# -----------------------

# Check if exactly two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_size> <config_name>"
    exit 1
fi

# Assign arguments to variables
model_size="$1"
config_name="$2"

supported=false
if [ "$model_size" == "8x7b" ]; then
    if [ "$config_name" == "inference" ]; then
        TP_SIZE=1
        EP_SIZE=1
        PP_SIZE=2
        supported=true
    # Training with nodes with 4gpus with nvlink
    elif [ "$config_name" == "training-4nvlink" ]; then
        TP_SIZE=1
        EP_SIZE=4
        PP_SIZE=8
        supported=true
    fi
fi

# Check if the model_size and config_name pair is supported
if [ "$supported" != true ]; then
    echo "Error: The pair model_size='$model_size' and config_name='$config_name' is not supported."
    exit 1
fi


# -----------------------
# Setup conda environment
# -----------------------

if [ -z "${CONDA_HOME}" ]; then
    echo "CONDA_HOME is unset"
    exit 1
fi
source ${CONDA_HOME}/etc/profile.d/conda.sh

conda activate megatron-core-env

# -----------------------
# Parallelism Arguments
# -----------------------

PARALLEL_ARGS=" \
    --source-tensor-parallel-size ${TP_SIZE} \
    --source-pipeline-parallel-size ${PP_SIZE} \
    --source-expert-parallel-size ${EP_SIZE}"

# -----------------------
# Load/Save Arguments
# -----------------------

MEGATRON_LOAD_DIR=/mnt/data/duarte/megatron-mixtral-tp${TP_SIZE}-pp${PP_SIZE}-ep${EP_SIZE}
HF_SAVE_DIR=/mnt/data/duarte/mixtral-converted
HF_TOKENIZER_PATH=/mnt/data/shared/mixtral/

LOAD_SAVE_ARGS=" \
    --mcore-load-dir ${MEGATRON_LOAD_DIR} \
    --hf-save-dir ${HF_SAVE_DIR} \
    --hf-tokenizer-path ${HF_TOKENIZER_PATH}"

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

python examples/mixtral/convert.py mcore-to-hf \
    ${LOAD_SAVE_ARGS} \
    ${PARALLEL_ARGS} \
    --target-params-dtype bfloat16 \
    --moe-grouped-gemm