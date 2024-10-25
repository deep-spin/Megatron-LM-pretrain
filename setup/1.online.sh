#!/bin/zsh
# Sets up a conda environment with all necessary packages for Megatron-LM and clonining all necessary dependencies. Requires internet access.
set +x

ENV_NAME=megatron-core-env

# get the directory of this script, and go one up to get the root directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR="$(dirname "$DIR")"

set -eo pipefail

# check if CONDA_HOME is set and create environment
if [ -z "$CONDA_HOME" ]
then
    echo "Please set CONDA_HOME to the location of your conda installation"
    exit 1
fi
source ${CONDA_HOME}/etc/profile.d/conda.sh
# python can't handle this dependency madness, switch to C++
conda create -y -n ${ENV_NAME} python=3.10
conda activate ${ENV_NAME}

pip install ninja psutil einops importlib-metadata pydantic packaging pybind11

# Install a recent compiler
conda install -y "gxx=11" -c conda-forge
# Install a recent cmake
conda install -y "cmake>=3.21" -c conda-forge
# install our own copy of CUDA and set environment variables
conda install -y -c "nvidia/label/cuda-12.4.0" cuda-toolkit cuda-nvcc cudnn

export PATH=${CONDA_HOME}/envs/${ENV_NAME}/bin:$PATH
export LD_LIBRARY_PATH=${CONDA_HOME}/envs/${ENV_NAME}/lib:$LD_LIBRARY_PATH
export CUDA_HOME=${CONDA_HOME}/envs/${ENV_NAME}

# install tourch th
pip install torch torchvision torchaudio

rm -rf .flash-attn && git clone --branch v2.5.8 --recursive https://github.com/Dao-AILab/flash-attention .flash-attn

rm -rf .transformer-engine && git clone --recursive https://github.com/NVIDIA/TransformerEngine .transformer-engine
cd .transformer-engine && git checkout 2215fa5c7557b66034068816020f9f611019e457 && cd ..

rm -rf .apex && git clone https://github.com/NVIDIA/apex .apex

conda env config vars set PATH=$PATH
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH
conda env config vars set CUDA_HOME=$CUDA_HOME

set +e
set -x