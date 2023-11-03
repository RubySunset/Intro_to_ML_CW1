import numpy as np
import math


# Creates a dictionary struct for decision tree nodes
def create_node(split_feature, split_value, node_depth=0, right_node={}, left_node={}):

    node = {
        "split feature": split_feature,
        "split value": split_value,
        "right": right_node,
        "left": left_node,
        "depth": node_depth,
    }
    
    return node


# Computes the Entropy of a subset S, where S is a array of labels
def entropy(S):

    total_samples = len(S)
    unique_labels = set(S)
    entropy_value = 0

    for label in unique_labels:
        p_k = np.count_nonzero(S == label) / total_samples
        entropy_value -= p_k * math.log2(p_k)

    return entropy_value


# Computes the remainder for subsets S_left and S_right
def remainder(S_left, S_right):

    total_samples = (S_left.shape[0]) + (S_right.shape[0])
    left_weighted_entropy = (S_left.shape[0] / total_samples) * entropy(S_left)
    right_weighted_entropy = (S_right.shape[0] / total_samples) * entropy(S_right)
    return  left_weighted_entropy + right_weighted_entropy 


# Computes the info gain for given subsets. S_all, S_left, S_right are the arrays of the labels
def information_gain(S_all, S_left, S_right):
    return entropy(S_all) - remainder(S_left, S_right)


# Maximises information gain and returns (split_feature, split_value)
def find_best_split(training_dataset, unique_labels, label_counts):

    best_split = (None, None)
    information_gain_max = 0

    # Iterates over columns/ features of dataset
    for i in range(training_dataset.shape[1]-1):
        # Extracts only ith and label columns and sorts them
        column_dataset = training_dataset[:, [i, -1]]
        sorted_indices = np.argsort(column_dataset[:, 0])
        sorted_dataset = column_dataset[sorted_indices]

        # Iterates over every value in the dataset
        for j in range(sorted_dataset.shape[0]-1):
            # Checks if current and next value are not equal
            if sorted_dataset[j, 0] != sorted_dataset[j+1, 0]:
                left_labels = sorted_dataset[:j+1, 1]
                right_labels = sorted_dataset[j+1:, 1]
                information_gain_new = information_gain(sorted_dataset[:, 1], left_labels, right_labels)

                if information_gain_new > information_gain_max:
                    information_gain_max = information_gain_new
                    # Split vallue takes midpoint between the current and next non equal values
                    split_value = (sorted_dataset[j][0] + sorted_dataset[j+1][0]) / 2
                    best_split = (i, split_value)

    return best_split


# Splits the dataset based on feature and value
def split_data(dataset, split_value, split_feature):

    left_dataset = dataset[dataset[:, split_feature] < split_value]
    right_dataset = dataset[dataset[:, split_feature] > split_value]
    return left_dataset, right_dataset


# Returns a numpy.array of the data and labels
def read_dataset(filepath):

    dataset = []
    for line in open(filepath):
        if line != "":
            row = line.strip().split()
            dataset.append(list(map(float, row)))

    dataset = np.array(dataset)
    return dataset


# Builds the decision tree and returns the root node
def decision_tree_learn(training_dataset, depth=0):

    dataset_labels = training_dataset[:, -1]
    unique_labels, label_counts = np.unique(dataset_labels, return_counts=True)

    # If there is only one label in the dataset return a leaf node
    if len(unique_labels) == 1:
        return unique_labels[0], depth

    # Finds best feature and value to split the dataset and splits it
    split_feature, split_value = find_best_split(training_dataset, unique_labels, label_counts)
    left_dataset, right_dataset = split_data(training_dataset, split_value, split_feature)

    # Creates the current node, and recurses over right and left children
    node = create_node(split_feature, split_value, node_depth=depth)
    node["left"], left_depth = decision_tree_learn(left_dataset, depth+1)
    node["right"], right_depth = decision_tree_learn(right_dataset, depth+1)

    return node, max(right_depth, left_depth)