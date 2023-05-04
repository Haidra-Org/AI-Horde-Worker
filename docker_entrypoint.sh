#!/bin/bash

bin/micromamba run -r conda -n linux python generate_docker_bridge_config.py

./horde-bridge.sh "$@"
