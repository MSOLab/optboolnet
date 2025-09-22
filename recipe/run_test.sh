#!/usr/bin/env bash
set -e  # exit immediately if a command fails

# install test-only dependencies
pip install colomoto-jupyter pyomo gurobipy pytest

# check for broken/missing dependencies
pip check

# run the test suite
pytest ./tests

# exit code from pytest is propagated automatically because of `set -e`
