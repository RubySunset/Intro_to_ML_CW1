import numpy as np
import math
from tree_vis import *
# from tree_vis_testing import basic_plot

CLEAN_DATA_PATH = "wifi_db/clean_dataset.txt"
NOISY_DATA_PATH = "wifi_db/noisy_dataset.txt"


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


# Pretty print function for tree
def tree_to_string(tree, depth=0):
    if depth == 0:
        string = "Tree\n"
    else:
        string = ""
    if type(tree) is dict:
        final_index = len(tree)-1
        current_index = 0
        for key, value in tree.items():
            key = key.capitalize()
            for indent in range(0, depth):
                string += "  |"
            if current_index == final_index:
                string += "  └-"
            else:
                string += "  ├-"
                current_index += 1
            if type(value) is dict:
                string += key + '\n' + tree_to_string(value, depth + 1)
            else:
                string += key + ": " + tree_to_string(value, depth+1) + '\n'
    else:
        string = str(tree)

    return string


# Computes the Entropy of a subset S, where S is a list of labels
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
    return (S_left.shape[0] / total_samples) * entropy(S_left) + (S_right.shape[0] / total_samples * entropy(S_right))


# Computes the info gain for given subsets. S_all, S_left, S_right are the lists of the labels
def information_gain(S_all, S_left, S_right):
    return entropy(S_all) - remainder(S_left, S_right)


# Maximises information gain and returns (split_feature, split_value)
# Try splitting on index and change in label???
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
                    # print(f'Info Gain: {information_gain_new}, Split Value: {split_value}, Index: {i, j}')

    return best_split


# Splits the dataset based on feature and value
def split_data(dataset, split_value, split_feature):
    right_dataset = dataset[dataset[:, split_feature] > split_value]
    left_dataset = dataset[dataset[:, split_feature] < split_value]
    return (right_dataset, left_dataset)


# Returns a numpy.array of the data and labels
def read_dataset(filepath):
    dataset = []
    for line in open(filepath):
        if line != "":
            row = line.strip().split()
            dataset.append(list(map(float, row)))

    dataset = np.array(dataset)
    return (dataset)


# Builds the decision tree and returns the root node
def decision_tree_learn(training_dataset, depth=0):
    dataset_labels = training_dataset[:, -1]
    unique_labels, label_counts = np.unique(dataset_labels, return_counts=True)

    if len(unique_labels) == 1:
        return (unique_labels[0], depth)

    # Finds best feature and value to split the dataset and splits it
    split_feature, split_value = find_best_split(training_dataset, unique_labels, label_counts)
    right_dataset, left_dataset = split_data(training_dataset, split_value, split_feature)

    # Creates the current node, and recurses over right and left children
    node = create_node(split_feature, split_value, node_depth=depth)
    node["right"], right_depth = decision_tree_learn(right_dataset, depth+1)
    node["left"], left_depth = decision_tree_learn(left_dataset, depth+1)

    # Alternative method
    # right_child, right_depth = decision_tree_learn(right_dataset, depth+1)
    # left_child, left_depth = decision_tree_learn(left_dataset, depth+1)
    # node = create_node(split_feature, split_value, right_child, left_child, depth)

    return (node, max(right_depth, left_depth))


# tree, depth = decision_tree_learn(read_dataset(CLEAN_DATA_PATH))
# print("Max Depth:", depth)
# print(tree_to_string(tree))

# # See basic plot first.
# # plt.figure()
# # plt.axis('off')
# # basic_plot(tree)

# # Compare to Reingold-Tilford algorithm.
# vis = TreeVis()
# vis.draw(tree)
