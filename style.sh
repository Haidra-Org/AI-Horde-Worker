#!/bin/bash

# Run this script with "--fix" to automatically fix the issues which can be fixed

# Set the working directory to where this script is located
cd "$(dirname ${BASH_SOURCE[0]})"

# exit script directly if any command fails
set -e

if [ "$1" == "--fix" ]
then
  echo "fix requested"
  BLACK_OPTS=""
  ISORT_OPTS=""
else
  echo "fix not requested"
  BLACK_OPTS="--check --diff"
  ISORT_OPTS="--check-only --diff"
fi

SRC="*.py nataili"

black --line-length=119 $BLACK_OPTS $SRC
flake8 $SRC
isort $ISORT_OPTS $SRC
