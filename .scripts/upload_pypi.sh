#!/bin/bash
Build=$(python ./src/optboolnet/version.py name)
cwd=$(pwd)
echo Uploading $Build to PyPI
twine upload $cwd/dist/$Build.tar.gz
