#!/bin/bash
Build=$(python ./src/optboolnet/version.py name)
cwd=$(pwd)

echo Building $Build for Anaconda.org

# default: run tests
TEST_ARGS=""
if [[ "$1" == "--skip-test" ]]; then
  echo "Skipping tests during conda-build"
  TEST_ARGS="--no-test"
fi

conda-build "./recipe" \
  --output-folder "./dist/conda" \
  -c colomoto -c gurobi -c conda-forge \
  $TEST_ARGS
