import numpy as np
from evaluation.k_folds import indices_split_folds
import evaluation.eval as eval

### from ___ import decision_tree_learn

def main(dataset_path="wifi_db/clean_dataset.txt"):
    print(dataset_path)
    dataset = read_dataset(dataset_path)

    confusions, accuracies = test_k_folds(dataset)

    print(accuracies[0])
    print(confusions[0])

    return


# Returns a numpy.array of the data and labels
def read_dataset(filepath):
    dataset = []
    for line in open(filepath):
        if line != "":
            row = line.strip().split()
            dataset.append(list(map(float, row)))

    dataset = np.array(dataset)
    return (dataset)


# Returns 2 arrays of dimension k with each accuracy and confusion matrix
def test_k_folds(dataset, k=10):
    confusions = []
    accuracies = []

    for (train_indices, test_indices) in indices_split_folds(k, len(dataset)):
        tree_top = decision_tree_learn(dataset[train_indices])[0] # Trains a tree on one fold iteration

        
        new_confusion, new_accuracy = eval.evaluate(dataset[test_indices], tree_top)
        confusions.append(new_confusion)
        accuracies.append(new_accuracy)

    return confusions, accuracies

if __name__ == "__main__":    

    choice = input("""Which dataset do you want to test the algorithm on?
                        Options:
                        - Clean
                        - Noisy
                        - [insert own filepath]
                        """)
    
    if choice == "Clean" or choice == "1":
        dataset_path = "wifi_db/clean_dataset.txt"
    elif choice == "Noisy" or choice =="2":
        dataset_path = "wifi_db/clean_dataset.txt"
    else:
        dataset_path = choice

    main(dataset_path)