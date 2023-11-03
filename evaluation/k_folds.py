import numpy as np
from numpy.random import default_rng
from build_tree import decision_tree_learn
from evaluation.prune import prune_tree
from evaluation.eval import evaluate


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


# Splits the data into training and validation sets based on k folds
def validation_training_split(train_indices, k=10):
    validation_folds = []
    split_train = np.array_split(train_indices, k-1)
    for i in range(k-1):
        validation_indices = split_train[i]
        nested_train_indices = np.concatenate(split_train[:i] + split_train[i+1:])
        validation_folds.append([nested_train_indices, validation_indices])

    return validation_folds


# Trains and tests k trees. Returns 2 arrays of dimension k with each accuracy and confusion matrix
def test_k_folds(dataset, k=10):
    confusions = []
    accuracies = []

    for (train_indices, test_indices) in indices_split_folds(k, len(dataset)):
        tree_top = decision_tree_learn(dataset[train_indices])[0] # Trains a tree on one fold iteration
        
        new_accuracy, new_confusion  = evaluate(dataset[test_indices], tree_top)
        confusions.append(new_confusion)
        accuracies.append(new_accuracy)

    return accuracies, confusions


# Trains and tests k trees. Returns 2 arrays of dimension k with each accuracy and confusion matrix and the best tree
def test_k_folds_pruning(dataset, k=10):
    best_confusions = []
    best_accuracies = []
    best_test_tree = {}
    best_test_accuracy = 0

    for (train_indices, test_indices) in indices_split_folds(k, len(dataset)):
        test_dataset = dataset[test_indices]
        best_validation_accuracy = 0
        best_validation_tree = {}
        
        for (nest_train_indices, validation_indices) in validation_training_split(train_indices, k):
            training_dataset = dataset[nest_train_indices]
            validation_dataset = dataset[validation_indices]
            
            tree = decision_tree_learn(training_dataset)[0]
            tree, new_accuracy, new_confusion = prune_tree(tree, tree, training_dataset, validation_dataset)
        
            if new_accuracy > best_validation_accuracy:
                best_validation_accuracy = new_accuracy
                best_validation_tree = tree
            
            # print(f'----------------------------------------------------------\nNew Tree Validation Acc: {new_accuracy}, Best Validation Acc: {best_validation_accuracy}\n----------------------------------------------------------\n\n')

        test_accuracy, test_confusion = evaluate(test_dataset, best_validation_tree)
        best_accuracies.append(test_accuracy)
        best_confusions.append(test_confusion)       
        
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_test_tree = best_validation_tree
        
        # print(f'##########################################################\n      New Tree Test Acc: {test_accuracy}, Best Test Acc: {best_test_accuracy}\n##########################################################\n\n')

    return best_accuracies, best_confusions, best_test_tree