#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_97/x86_64-centos7-gcc8-opt/setup.sh

# Create directory for figures (if it doesn't exist)
mkdir -p figures 

# Add inlcude directory
export ROOT_INCLUDE_PATH=$ROOT_INCLUDE_PATH:$PWD/include