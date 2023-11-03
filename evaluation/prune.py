import numpy as np
import copy
from build_tree import split_data
from evaluation.eval import evaluate


def count_labels(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    return label_counts


# Recursive pruning function
def prune_tree(node, tree, training_dataset, validation_dataset):

    if isinstance(node, dict):
        # Split the data
        right_dataset, left_dataset = split_data(training_dataset, node["split value"], node["split feature"])
    
    # Check if child nodes are both leaf nodes 
    if isinstance(node, dict) and isinstance(node["left"], np.float64) and isinstance(node["right"], np.float64):
            
        right_label_counts = count_labels(right_dataset[:, -1])
        left_label_counts = count_labels(left_dataset[:, -1])
        
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
        # print(f'------------------------------------------\nOriginal Left Acc: {best_accuracy},\nPruned Acc: {pruned_accuracy},\n------------------------------------------\n\n')

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
        # print(f'------------------------------------------\nOriginal Right Acc: {best_accuracy},\nPruned Acc: {pruned_accuracy},\n------------------------------------------\n\n')

        # Compare accuracies and whether to prune or not
        if pruned_accuracy >= best_accuracy:
            best_accuracy = pruned_accuracy
            best_confusion = pruned_confusion
            del original_node["right"]
        else: 
            node["right"] = original_node["right"]
    
    # print(f'------------------------------------------\nBest Returned Acc: {best_accuracy},\n------------------------------------------\n\n')

    # Returns the current node and accuracy
    return node, best_accuracy, best_confusion