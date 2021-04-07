#!/usr/bin/env bash

cd "$(dirname "$0")"
cd ../..

export PYTHONPATH=$PYTHONPATH:$(pwd)/wavedata
echo $PYTHONPATH

echo "Running unit tests in $(pwd)/ammf"
coverage run --source ammf -m unittest discover -b --pattern "*_test.py"

#coverage report -m
