#!/bin/bash
Build=$(python ./src/optboolnet/version.py name)
cwd=$(pwd)
echo Building $Build for PyPI
python -m build -s -w