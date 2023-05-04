#!/bin/bash
if [ $# -eq 1 ]; then
  export AIWORKER_CACHE_HOME=$1
fi
./runtime.sh python -m hordelib.benchmark
