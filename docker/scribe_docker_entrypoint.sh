#!/bin/bash

bin/micromamba run -r conda -n linux python docker/generate_docker_bridge_config.py 

./horde-scribe-bridge.sh "$@"
