#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ---------------------------------------------
echo Downloading sports-1m-dataset labels...
wget \
  -N \
  https://raw.githubusercontent.com/gtoderici/sports-1m-dataset/master/labels.txt \
  --directory-prefix=${DIR}

echo ---------------------------------------------
echo Done!
