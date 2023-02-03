#!/bin/bash

inbox_paramsdir=$1

for params_path in $inbox_paramsdir/*;
do
    python $HOME/UQ_singleNN/evi/train/pipeline.py $params_path;
done
