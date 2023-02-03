#!/bin/bash

params_path=$1

# train model
python $HOME/UQ_singleNN/train/pipeline.py $params_path

# attack 
python $HOME/UQ_singleNN/inv/attack.py $params_path
