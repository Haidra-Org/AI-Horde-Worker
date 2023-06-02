#!/bin/bash

ignore_hordelib=false

# Parse command line arguments
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --hordelib)
    hordelib=true
    shift # past argument
    ;;
    --scribe)
    scribe=true
    shift
    ;;
    *)    # unknown option
    echo "Unknown option: $key"
    exit 1
    ;;
esac
shift # past argument or value
done

CONDA_ENVIRONMENT_FILE=environment.yaml
if [ "$scribe" = true ]; then
    CONDA_ENVIRONMENT_FILE=environment_scribe.yaml
fi

wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
if [ ! -f "conda/envs/linux/bin/python" ]; then
    bin/micromamba create --no-shortcuts -r conda -n linux -f ${CONDA_ENVIRONMENT_FILE} -y
fi
bin/micromamba create --no-shortcuts -r conda -n linux -f ${CONDA_ENVIRONMENT_FILE} -y

if [ "$hordelib" = true ]; then
 bin/micromamba run -r conda -n linux python -s -m pip uninstall -y hordelib horde_model_reference
 bin/micromamba run -r conda -n linux python -s -m pip install hordelib horde_model_reference
elif [ "$scribe" = true ]; then
 bin/micromamba run -r conda -n linux python -s -m pip install -r requirements-scribe.txt
else
 bin/micromamba run -r conda -n linux python -s -m pip uninstall -y nataili
 bin/micromamba run -r conda -n linux python -s -m pip install -r requirements.txt
fi
