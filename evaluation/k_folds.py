import numpy as np
from numpy.random import default_rng
from build_tree import decision_tree_learn
import evaluation.eval as eval

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

# Trains and tests k trees. Returns 2 arrays of dimension k with each accuracy and confusion matrix
def test_k_folds(dataset, k=10):
    confusions = []
    accuracies = []

    for (train_indices, test_indices) in indices_split_folds(k, len(dataset)):
        tree_top = decision_tree_learn(dataset[train_indices])[0] # Trains a tree on one fold iteration
        
        new_accuracy, new_confusion  = eval.evaluate(dataset[test_indices], tree_top)
        confusions.append(new_confusion)
        accuracies.append(new_accuracy)

    return accuracies, confusions