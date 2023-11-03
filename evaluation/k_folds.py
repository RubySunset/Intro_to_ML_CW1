import numpy as np
from numpy.random import default_rng
from training.build_tree import decision_tree_learn
from training.prune_tree import prune_tree
from evaluation.eval_tree import evaluate


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


# Splits the data into training and validation sets based on k-1 folds
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

    # Iterates over K test folds
    for (train_indices, test_indices) in indices_split_folds(k, len(dataset)):
        # Trains a tree on one fold iteration
        tree_top = decision_tree_learn(dataset[train_indices])[0] 

        # Evaluates each tree on the test dataset and records its accuracy and confusion matrix
        new_accuracy, new_confusion  = evaluate(dataset[test_indices], tree_top)
        confusions.append(new_confusion)
        accuracies.append(new_accuracy)
        
        print(f'''
            ################################################
                    Latest Test Fold Accuracy: {new_accuracy}
                    Best Test Fold Accuracy:   {max(accuracies)}   
            ################################################
        ''')

    # Returns an array of all the calculated accuracies and confusions
    return accuracies, confusions


# Trains, prunes and tests k trees. Returns 2 arrays of dimension k with each accuracy and confusion matrix and the best pruned tree
def test_k_folds_pruning(dataset, k=10):

    test_confusions = []
    test_accuracies = []
    best_test_tree = {}
    best_test_accuracy = 0

    # Iterates over K test folds
    for (train_indices, test_indices) in indices_split_folds(k, len(dataset)):
        test_dataset = dataset[test_indices]
        best_validation_accuracy = 0
        best_validation_tree = {}

        # Iterates over K-1 validation folds
        for (nest_train_indices, validation_indices) in validation_training_split(train_indices, k):
            training_dataset = dataset[nest_train_indices]
            validation_dataset = dataset[validation_indices]
            
            # Trains the tree and prunes it on the current test and validation fold iterations
            tree = decision_tree_learn(training_dataset)[0]
            tree, new_accuracy, new_confusion = prune_tree(tree, tree, training_dataset, validation_dataset)

            # Stores the best accuracy and corresponding tree over all validation folds in the current test fold
            if new_accuracy > best_validation_accuracy:
                best_validation_accuracy = new_accuracy
                best_validation_tree = tree

        # Evaluates the best tree on the test dataset and records its accuracy and confusion matrix
        test_accuracy, test_confusion = evaluate(test_dataset, best_validation_tree)
        test_accuracies.append(test_accuracy)
        test_confusions.append(test_confusion)       
        
        
        # Stores the best accuracy and corresponding tree over all test folds
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_test_tree = best_validation_tree

        print(f'''
            ################################################
                    Latest Test Fold Accuracy: {test_accuracy}
                    Best Test Fold Accuracy:   {best_test_accuracy}   
            ################################################
        ''')

    # Returns an array of all the calculated test accuracies and confusions as well as the most accurate tree
    return test_accuracies, test_confusions, best_test_tree