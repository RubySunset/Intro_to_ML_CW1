import copy
import numpy as np
from evaluation.eval_tree import evaluate
from training.build_tree import split_data


def count_labels(labels):

    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    return label_counts


# Recursive pruning function
def prune_tree(node, tree, training_dataset, validation_dataset):

    # Split the data if the current node is not a leaf
    if isinstance(node, dict):
        left_dataset, right_dataset = split_data(training_dataset, node["split value"], node["split feature"])
    
    # Check if child nodes are both leaf nodes 
    if isinstance(node, dict) and isinstance(node["left"], np.float64) and isinstance(node["right"], np.float64):
        
        # Counts the labels in right and left splits
        left_label_counts = count_labels(left_dataset[:, -1])
        right_label_counts = count_labels(right_dataset[:, -1])
        
        # Compare the counts of leaf children
        if left_label_counts[node["left"]] > right_label_counts[node["right"]]:
            pruned_node = node["left"]
        elif left_label_counts[node["left"]] < right_label_counts[node["right"]]:
            pruned_node = node["right"]
        else: # Labels have equal counts
            choice = np.random.choice([True,False])
            pruned_node = node["left"] if choice else node["right"]
        
        # Returns pruned version of node no matter what
        return pruned_node, 0, []
        
    # Make copy of original node
    original_node = copy.deepcopy(node)
    
    # Recursively prune the left and right subtrees if new accuracy is better or equal
    if isinstance(node["left"], dict):
        # Evaluate original and pruned accuracies
        best_accuracy, best_confusion = evaluate(validation_dataset, tree)
        node["left"] = prune_tree(node["left"], tree, left_dataset, validation_dataset)[0]
        pruned_accuracy, pruned_confusion = evaluate(validation_dataset, tree)

        # Compare accuracies and whether to prune or not
        if pruned_accuracy >= best_accuracy:
            best_accuracy = pruned_accuracy
            best_confusion = pruned_confusion
            del original_node["left"]
        else: 
            node["left"] = original_node["left"]

    if isinstance(node["right"], dict):
        # Evaluate original and pruned accuracies
        best_accuracy, best_confusion = evaluate(validation_dataset, tree)
        node["right"] = prune_tree(node["right"], tree, right_dataset, validation_dataset)[0]
        pruned_accuracy, pruned_confusion = evaluate(validation_dataset, tree)

        # Compare accuracies and whether to prune or not
        if pruned_accuracy >= best_accuracy:
            best_accuracy = pruned_accuracy
            best_confusion = pruned_confusion
            del original_node["right"]
        else: 
            node["right"] = original_node["right"]

    # Returns the current node and accuracy
    return node, best_accuracy, best_confusion