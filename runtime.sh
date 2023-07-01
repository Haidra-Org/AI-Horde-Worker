#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build the absolute path to the Conda environment
CONDA_ENV_PATH="$SCRIPT_DIR/conda/envs/linux/lib"

# Add the Conda environment to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_ENV_PATH:$LD_LIBRARY_PATH"
export MAMBA_ROOT_PREFIX="$SCRIPT_DIR/conda"

if [ ! -f "conda/envs/linux/bin/python" ]; then
./update-runtime.sh
fi
bin/micromamba run -r conda -n linux "$@"
