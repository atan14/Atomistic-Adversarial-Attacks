# Single-model uncertainty quantification in neural network potentials does not consistently outperform model ensembles

[![DOI](https://arxiv.org/abs/2305.01754)](https://arxiv.org/abs/2305.01754)

Code for performing uncertainty quantification(UQ) for neural network(NN) interatomic potentials using single deterministic NNs and NN ensemble. The software was based on the paper ["Single-model uncertainty quantification in neural network potentials does not consistently outperform model ensembles"](https://arxiv.org/abs/2305.01754), and implemented by Aik Rui Tan. The code was adapted from the [NeuralForceField repo](https://github.com/learningmatter-mit/NeuralForceField.git) and [Atomistic-Adversarial-Attack repo](https://github.com/learningmatter-mit/Atomistic-Adversarial-Attacks.git).

The folder contains [`systems`](systems/) contains script to run training and adversarial attack on the rMD17, ammonia and silica data sets. 

The full atomistic data set for:
- rMD17 is available at [https://figshare.com/articles/dataset/Revised\_MD17\_dataset\_rMD17\_/12672038](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038).
- ammonia is available at [https://doi.org/10.24435/materialscloud:2w-6h](https://doi.org/10.24435/materialscloud:2w-6h).
- silica is available at [https://doi.org/10.24435/materialscloud:55-sd](https://doi.org/10.24435/materialscloud:55-sd).

## Citing

The reference for the paper is the following:
```
@misc{tan2023singlemodel,
      title={Single-model uncertainty quantification in neural network potentials does not consistently outperform model ensembles}, 
      author={Aik Rui Tan and Shingo Urata and Samuel Goldman and Johannes C. B. Dietschreit and Rafael Gómez-Bombarelli},
      year={2023},
      eprint={2305.01754},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
