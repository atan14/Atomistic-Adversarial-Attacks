#!/bin/bash 

molecules=("ethanol" "uracil" "malonaldehyde" "paracetamol" "aspirin" "azobenzene" "salicylic" "toluene" "benzene" "naphthalene")
split_types=(1 2 3 4 5)

for mol in ${molecules[@]};
do
    for split in ${split_types[@]};
    do
        ### ensemble model
        python create_params.py ${mol} --model__uncertainty_type ensemble --loss__forces_loss mae --dset__split_type ${split}

        ### mve model
        python create_params.py ${mol} --model__uncertainty_type mve --loss__forces_loss nll --dset__split_type ${split}

        ## evidential model
        python create_params.py ${mol} --model__uncertainty_type evidential --loss__forces_loss evidential --dset__split_type ${split}

        ## gmm model
        python create_params.py ${mol} --model__uncertainty_type gmm --loss__forces_loss mae --dset__split_type ${split} --model__n_atom_basis 32
    done
done
