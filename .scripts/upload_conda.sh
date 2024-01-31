#!/bin/bash
Build=$(python ./src/optboolnet/version.py name)
cwd=$(pwd)
echo Uploading $Build to Anaconda.org
anaconda upload $cwd/dist/conda/noarch/$Build-py_0.tar.bz2