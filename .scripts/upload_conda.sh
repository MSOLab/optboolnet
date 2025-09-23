#!/bin/bash
# Upload the conda package to Anaconda.org
# Usage: ./upload_conda.sh {personal|group}
Build=$(python ./src/optboolnet/version.py name)
cwd=$(pwd)
echo Uploading $Build to Anaconda.org

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 {personal|group}"
    exit 1
fi

CHANNEL=$1
case "$CHANNEL" in
    personal)
        echo "Uploading opboolnet to your personal Anaconda.org account"
        anaconda upload $cwd/dist/conda/noarch/$Build-py_0.conda
    ;;
    group)
        echo "Uploading opboolnet to Anaconda.org group 'msolab'"
        anaconda upload --user msolab $cwd/dist/conda/noarch/$Build-py_0.conda
    ;;
    *)
        echo "Unknown option: $CHANNEL (must be 'personal' or 'group')"
        exit 1
    ;;
esac