These python scripts requires python3, matplotlib, scikit and numpy to run.

Clustering
==========

The "clustering" folder contains 2 scripts: kmeans.py and em.py.
The first script applies the k-Means algorithm on the datasets. The following parameters are modifiable:
DATASET : The dataset to use "digits" or "phoneme"
N_CLUSTERS : The number of cluster to look for

The second script applies the EM algorithm and has the same parameters

Dimensionality Reduction
========================

The "dim" folder contains 4 scripts: PCA.py, ICA.py, RP.py and FA.py.
Each script applies the corresponding algorithm on the dataset.

The PCA.py file has the following modifiable parameters:
DATASET : The dataset to use "digits" or "phoneme"
N_COMPONENTS : The number of components to keep
N_CLUSTERS : The number of cluster to look for
MODE : The experiment to do using the script 'learning', 'compute_time' or 'reconstruction'
LEARNING_RATE : The neural network initial learning rate
TOLERANCE : The neural network solver tolerance
TOPOLOGY : The neural network nodes topology (hidden layers and node count)

The ICA.py file has the following modifiable parameters:
DATASET : The dataset to use "digits" or "phoneme"
N_COMPONENTS : The number of components to keep
N_CLUSTERS : The number of cluster to look for
MODE : The experiment to do using the script 'learning', 'compute_time' or 'reconstruction'
LEARNING_RATE : The neural network initial learning rate
TOLERANCE : The neural network solver tolerance
TOPOLOGY : The neural network nodes topology (hidden layers and node count)

The RP.py file has the following modifiable parameters:
DATASET : The dataset to use "digits" or "phoneme"
N_FEATURES : The number of different labels for the current dataset
N_COMPONENTS : The number of components to keep
N_CLUSTERS : The number of cluster to look for
MODE : The experiment to do using the script 'learning', 'compute_time' or 'compare'
LEARNING_RATE : The neural network initial learning rate
TOLERANCE : The neural network solver tolerance
TOPOLOGY : The neural network nodes topology (hidden layers and node count)

The FA.py file has the following modifiable parameters:
DATASET : The dataset to use "digits" or "phoneme"
N_FEATURES : The number of different labels for the current dataset
N_COMPONENTS : The number of components to keep
N_CLUSTERS : The number of cluster to look for
MODE : The experiment to do using the script 'learning', 'compute_time' or 'compare'
LEARNING_RATE : The neural network initial learning rate
TOLERANCE : The neural network solver tolerance
TOPOLOGY : The neural network nodes topology (hidden layers and node count)

Other
=====

The "plot" folder contains the script used to visualize one of the dataset projected on the 3 first features.