#!/bin/zsh
# Script for setting up a conda environment with for launching servers
# It sidesteps system-wide installations by relying on conda for most packages
# and by building openssl from source
# TODO: only got it to work with a static build of OpenSSL, which is not ideal
set +x

ENV_NAME="${ENV_NAME:-megatron-core-env}"

# get the directory of this script, which should be the root directory of the project
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -eo pipefail

# check if CONDA_HOME is set
if [ -z "$CONDA_HOME" ]
then
    echo "Please set CONDA_HOME to the location of your conda installation"
    exit 1
fi

# Conda envs may not be in the conda home directory
CONDA_ENVS="${CONDA_ENVS:-$CONDA_HOME/envs}"

echo "Using environment name: $ENV_NAME"
echo "Using Conda home path: $CONDA_HOME"
echo "Using Conda environments path: $CONDA_ENVS"
echo "Megatron-LM dir: $DIR"

source ${CONDA_HOME}/etc/profile.d/conda.sh
# python can't handle this dependency madness, switch to C++
conda create -y -n ${ENV_NAME} python=3.10
conda activate ${ENV_NAME}


pip install ninja

# install our own copy of CUDA and set environment variables
conda install -y -c "nvidia/label/cuda-12.4.0" cuda-toolkit cuda-nvcc cudnn

export PATH=${CONDA_ENVS}/${ENV_NAME}/bin:$PATH
export LD_LIBRARY_PATH=${CONDA_ENVS}/${ENV_NAME}/lib:$LD_LIBRARY_PATH
export CUDA_HOME=${CONDA_ENVS}/${ENV_NAME}

# install tourch th
pip install torch torchvision torchaudio
# pip install "transformer_engine[pytorch]<1.11"
# it seems that 1.11 straight up doesn't work, but 1.10 has a problem with checkpointing
# so we install after that commit 
pip install -v --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@2215fa5c7557b66034068816020f9f611019e457

rm -rf .apex && git clone https://github.com/NVIDIA/apex .apex && 
cd .apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

conda env config vars set PATH=$PATH
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH
conda env config vars set CUDA_HOME=$CUDA_HOME
# Adding Megatron-LM to the PYTHONPATH as it is always needed
conda env config vars set PYTHONPATH=$DIR:$PYTHONPATH
