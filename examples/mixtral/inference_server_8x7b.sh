#!/bin/bash

set -euo pipefail

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
# Architecture Arguments
# -----------------------
ARCH_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --normalization RMSNorm
    --max-position-embeddings 32768
    --position-embedding-type rope
    --no-position-embedding
    --rotary-base 1000000
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --swiglu
    --num-experts 8
    --moe-router-topk 2
    --tokenizer-type Llama2Tokenizer
)

# -----------------------
# Parallelism Arguments
# -----------------------

TP_SIZE=1
EP_SIZE=1
PP_SIZE=2

PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --expert-model-parallel-size ${EP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
)

# -----------------------
# Load Arguments
# -----------------------

MODEL_PATH=/mnt/data/duarte/megatron-mixtral-tp${TP_SIZE}-pp${PP_SIZE}-ep${EP_SIZE}
TOKENIZER_PATH=/mnt/data/shared/mixtral/tokenizer.model
LOAD_ARGS=(
    --load ${MODEL_PATH}
    --tokenizer-model ${TOKENIZER_PATH}
)

# -----------------------
# Inference Arguments
# -----------------------

INFERENCE_ARGS=(
    --use-mcore-models
    --bf16
    --use-flash-attn
    --no-masked-softmax-fusion
    --micro-batch-size 1
    --seq-length 1024
    --seed 42
    --moe-token-dispatcher-type alltoall
    --mock-data
    --moe-grouped-gemm
    --no-async-tensor-model-parallel-allreduce
)

# -----------------------
# Distributed Arguments
# -----------------------

DISTRIBUTED_ARGS=(
    --nproc_per_node 2
    --nnodes 1
    --master_addr localhost
    --master_port 24734
)

# -----------------------
# Run Inference
# -----------------------

if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "CUDA_VISIBLE_DEVICES is unset"
    exit 1
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun ${DISTRIBUTED_ARGS[@]} \
    tools/run_text_generation_server.py \
    ${ARCH_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${LOAD_ARGS[@]} \
    ${INFERENCE_ARGS[@]}