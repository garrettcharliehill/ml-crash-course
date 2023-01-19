#!/bin/bash

# download conda environment
bash ./setup/miniconda.sh -b -p miniconda
source ./miniconda/bin/activate
conda install -c apple tensorflow-deps

# install dependencies
pip install -r ./setup/requirements.txt