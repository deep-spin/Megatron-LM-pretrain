#!/bin/zsh
# Compiles all dependencies of Megatron-LM. Does not require internet but requires CUDA and cuDNN to be installed.
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

conda activate ${ENV_NAME}

PIP_ARGS="-v --disable-pip-version-check --no-cache-dir --no-build-isolation --no-index"

cd .flash-attn
python setup.py install
cd ..

cd .transformer-engine
pip install $PIP_ARGS ./
cd ..

cd .apex
pip install $PIP_ARGS --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..

set +e
set -x