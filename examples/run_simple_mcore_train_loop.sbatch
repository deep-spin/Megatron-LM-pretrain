#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --job-name=simple-mcore-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

if [ -z "$CONDA_HOME" ]
then
    echo "Please set CONDA_HOME to the location of your conda installation"
    exit 1
fi

set -euo pipefail
set -x

source ${CONDA_HOME}/etc/profile.d/conda.sh

ENV_NAME=${ENV_NAME:-megatron-lm}

conda activate ${ENV_NAME}

cd ${SLURM_SUBMIT_DIR}

SRUN_ARGS=(
    --nodes=${SLURM_JOB_NUM_NODES}
    --ntasks-per-node=1
    --cpus-per-task=${SLURM_CPUS_ON_NODE}
    --gpus-per-node=${SLURM_GPUS_PER_TASK}
    --cpu-bind=none
    --mem-bind=none
    --label
)

RDZV_HOST=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

TORCHRUN_ARGS=(
    --nnodes=$SLURM_JOB_NUM_NODES
    --nproc_per_node=${SLURM_GPUS_PER_TASK}
    --rdzv-id=${SLURM_JOBID}
    --rdzv-backend=c10d
    --rdzv-endpoint="${RDZV_HOST}:${RDZV_PORT:-29400}"
)

export CUDA_DEVICE_MAX_CONNECTIONS=1
srun ${SRUN_ARGS[@]} -- \
    torchrun ${TORCHRUN_ARGS[@]} examples/run_simple_mcore_train_loop.py

set +x