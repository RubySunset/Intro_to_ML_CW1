import numpy as np
from numpy.random import default_rng

# Shuffles indices up to instances_n, splits them and organises all the folds
def indices_split_folds(fold_n, instances_n):
    shuffled = default_rng().permutation(instances_n)
    splits   = np.array_split(shuffled, fold_n)

    folds = []
    for k in range(fold_n):
        test_indices = splits[k]
        train_indices = np.concatenate(splits[:k] + splits[k+1:])

        folds.append([train_indices, test_indices])
    
    return folds