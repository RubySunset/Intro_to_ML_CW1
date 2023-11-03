import numpy as np
from evaluation.eval_tree import *
from visualisation.tree_vis import TreeVis
from training.build_tree import decision_tree_learn
from evaluation.k_folds import test_k_folds, test_k_folds_pruning


def train_and_test(dataset_path="wifi_db/clean_dataset.txt", prune=False):
    
    print("Decision trees trained on: " + dataset_path)
    try:
        dataset = read_dataset(dataset_path)
    except FileNotFoundError:
        print('File not found.')
        return

    # confusions is an array of confusion matrices. accuracies is also an array.
    # Both have a length equal to number of folds.
    if prune:
        accuracies, confusions, _ = test_k_folds_pruning(dataset)
    else:
        accuracies, confusions  = test_k_folds(dataset) 

    accuracy_avg = np.mean(accuracies)
    print("\nAccuracy average: " + str(accuracy_avg))

    # Combined confusion matrix is the matrix obtained from the sum of all the confusion matrices
    combined_confusion = np.zeros((4, 4), dtype=np.int32)
    for matrix in confusions:
        combined_confusion += matrix

    print("\nSum Confusion: ")
    print(combined_confusion)
    print("")

    # Precisions for each of the 4 classes
    precisions_matrix = np.zeros((4, len(confusions)))
    recalls_matrix = np.zeros((4, len(confusions)))
    f1_matrix = np.zeros((4, len(confusions)))

    # These will store the macro-averaged metrics for each fold.
    macro_precision = np.zeros(len(confusions))
    macro_recall = np.zeros(len(confusions))
    macro_f1 = np.zeros(len(confusions))

    for i, matrix in enumerate(confusions):
        new_precisions, macro_precision[i] = precision(matrix)
        new_recalls, macro_recall[i] = recall(matrix)
        for class_n in range(len(confusions[0])):
            # Precision of class class_n+1 becomes a part of precisions matrix.
            precisions_matrix[class_n, i] = new_precisions[class_n]
            recalls_matrix[class_n, i] = new_recalls[class_n]
            f1_matrix[class_n, i] = f1_score_class(new_precisions[class_n], new_recalls[class_n])
        macro_f1[i] = np.mean(f1_matrix[:, i])

    precision_matrix_avg = []
    recall_matrix_avg = []
    f1_matrix_avg = []
    for class_n_p, class_n_r, class_n_ef in zip(precisions_matrix, recalls_matrix, f1_matrix):
        precision_matrix_avg.append(np.mean(class_n_p))
        recall_matrix_avg.append(np.mean(class_n_r))
        f1_matrix_avg.append(np.mean(class_n_ef))

    print("Averaged precision per class: ")
    print(precision_matrix_avg)
    print("Averaged recall per class: ")
    print(recall_matrix_avg)
    print("Averaged F1 score per class: ")
    print(f1_matrix_avg)
    print("\n")
    print("Averaged macro-averaged metrics: ")
    print("Precision: ", np.mean(macro_precision))
    print("Recall: ", np.mean(macro_recall))
    print("F1-score: ", np.mean(macro_f1))


# Returns a numpy.array of the data and labels
def read_dataset(filepath):

    dataset = []
    for line in open(filepath):
        if line != "":
            row = line.strip().split()
            dataset.append(list(map(float, row)))

    dataset = np.array(dataset)
    return dataset


if __name__ == "__main__":
    vis = TreeVis()
    
    # Displays Options Menu
    while True:

        try:
            choice = int(input("""Options (enter number): 
                        1. Test base algortihm on clean dataset
                        2. Test base algorithm on noisy dataset
                        3. Test pruned version on clean dataset
                        4. Test pruned version on noisy dataset
                        5. Visualise base tree on clean dataset
                        6. Visualise base tree on noisy dataset
                        7. Visualise pruned tree on clean dataset
                        8. Visualise pruned tree on noisy dataset
                        9. Exit
                        """))
        except ValueError:
            print("Please enter a number.")
            continue
        
        if choice < 1 or choice > 9:
            print("Please enter a valid number.")
            continue
        
        # Exit
        if choice == 9:
            exit()
            
        # Choose dataset
        if choice % 2 == 1:
            dataset_path = "wifi_db/clean_dataset.txt"
        else:
            dataset_path = "wifi_db/noisy_dataset.txt"
            
        # Tests or Visualises either base or pruned tree
        if choice <= 4:
            train_and_test(dataset_path, prune=(choice>2))
        elif choice <= 6:
            dataset = read_dataset(dataset_path)
            root, depth = decision_tree_learn(dataset)
            vis.draw(root)
        elif choice <= 8:
            dataset = read_dataset(dataset_path)
            root = test_k_folds_pruning(dataset)[2]
            vis.draw(root)
        
        print('\n' * 2)