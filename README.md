# Correlation Clustering with Same-Cluster Queries Bounded by Optimal Cost
This repository contains the code and data for reproducing the experiments in the paper ["Correlation Clustering with Same-Cluster Queries Bounded by Optimal Cost"](https://arxiv.org/abs/1908.04976) by Barna Saha and Sanjay Subramanian.
## Citation
If you use this work in your research, please cite our paper:
```
@inproceedings{SahaSubramanian19,
    author = {Barna Saha and Sanjay Subramanian},
    title = {{Correlation Clustering with Same-Cluster Queries Bounded by Optimal Cost}},
    booktitle = {European Symposium on Algorithms ({ESA})},
    year = {2019}
}
```
## Requirements
This code was tested on MacBook Pro and on Ubuntu 16.0 servers. The code was run using Python 2.7. The other dependencies are scipy and numpy. The Gurobi software and a Gurobi license is required for the Integer Linear Program (ILP) and Linear Program (LP) results.
## Running experiments
To run the experiments for Table 1, please type ```python experiments.py [N/D/S] [1/2/3]```. The first argument determines the way in which cliques are generated and each letter corresponds to the same letter in the paper's descriptions of the experiments. The second argument determines the way in which edge mistakes are generated; 1 corresponds to option I in the paper, 2 corresponds to option II in the paper, and 3 corresponds to option III in the paper.
To run the experiments for Table 2, please type ```python realdata_experiments.py [skew/sqrt] [1/2/3]```. The first argument determines which dataset to use to specify the cliques, and the second argument determines how edge mistakes are generated as explained above.
To run the experiments for Table 3, 4, 5, and 6, please type ```python realdata_experiments.py [cora/gym/landmarks/allsports] [0/1/2]```. (For cora, the second argument is not used.) The first argument determines the dataset to be used, and the second argument determines the type of experiment. In particular, choice 0 corresponds to Table 3 (and 4), choice 1 corresponds to Table 5, and choice 2 corresponds to Table 6.
Please note that in both experiments.py and realdata_experiments.py, there is a line for seeding the random number generator with a specific integer. These two integers were chosen arbitrarily, and we used seeding in order to ensure ease of reproducibility for the experiments.
