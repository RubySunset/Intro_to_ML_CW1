import numpy as np

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


# Calculates Enathalpy of a dataset
# def enthalpy(dataset):
#     labels = dataset[


# DUMMY FUNCTION
def find_best_split(training_dataset, label_counts):
    rng = np.random.default_rng()
    ints = rng.integers(low=-10, high=11, size=2)
    return ("Attr:"+str(ints[0]), ints[1])


# DUMMY FUNCTION
def split_data(training_dataset, split_value):
    half_size = int(len(training_dataset)/2)
    right_dataset = training_dataset[:half_size, :]
    left_dataset = training_dataset[half_size:, :]
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
    dataset_labels = training_dataset[:, 7]
    unique_labels, label_counts = np.unique(dataset_labels, return_counts=True)

    if len(unique_labels) == 1:
        return (unique_labels[0], depth)

    # Dummy Functions find_split and split_data - FIX LATER
    split_feature, split_value = find_best_split(training_dataset, label_counts)
    right_dataset, left_dataset = split_data(training_dataset, split_value)

    node = create_node(split_feature, split_value, node_depth=depth)
    node["right"], right_depth = decision_tree_learn(right_dataset, depth+1)
    node["left"], left_depth = decision_tree_learn(left_dataset, depth+1)

    # Alternative methods
    # right_child, right_depth = decision_tree_learn(right_dataset, depth+1)
    # left_child, left_depth = decision_tree_learn(left_dataset, depth+1)
    # node = create_node(split_feature, split_value, right_child, left_child, depth)
    # node = {
    #     "split feature": split_feature,
    #     "split value": split_value,
    #     "right": right_child,
    #     "left": left_child,
    #     "depth": depth,
    # }

    return (node, max(right_depth, left_depth))


print(tree_to_string(decision_tree_learn(read_dataset(CLEAN_DATA_PATH))[0]))
