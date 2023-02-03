#!/bin/bash

total_gen=3
sampling_lr=3e-5

# create parameter file
python create_params.py --total_generations $total_gen --model__uncertainty_type 'ensemble' --loss__forces_loss 'mae' --sampling__lr $sampling_lr

python create_params.py --total_generations $total_gen --model__uncertainty_type 'mve' --loss__forces_loss 'nll' --sampling__lr $sampling_lr

python create_params.py --total_generations $total_gen --model__uncertainty_type 'evidential' --loss__forces_loss 'evidential' --sampling__lr $sampling_lr

python create_params.py --total_generations $total_gen --model__uncertainty_type 'gmm' --loss__forces_loss 'mae' --sampling__lr $sampling_lr
