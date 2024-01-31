#!/bin/bash
Build=$(python ./src/optboolnet/version.py name)
cwd=$(pwd)
echo Building $Build for Anaconda.org
conda build "./recipe" --output-folder "./dist/conda" -c colomoto -c gurobi -c conda-forge