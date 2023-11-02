import numpy as np
from evaluation.k_folds import test_k_folds
import evaluation.eval as eval
from build_tree import decision_tree_learn
from tree_vis.tree_vis import TreeVis


def train_and_test(dataset_path="wifi_db/clean_dataset.txt"):
    print("Decision trees trained on: " + dataset_path)

    try:
        dataset = read_dataset(dataset_path)
    except FileNotFoundError:
        print('File not found.')
        return

    # confusions is an array of confusion matrices. accuracies is also an array.
    # Both have a length equal to number of folds.
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
        new_precisions, macro_precision[i] = eval.precision(matrix)
        new_recalls, macro_recall[i] = eval.recall(matrix)
        for class_n in range(len(confusions[0])):
            precisions_matrix[class_n, i] = new_precisions[class_n] # Precision of class class_n+1 becomes a part of precisions matrix.
            recalls_matrix[class_n, i] = new_recalls[class_n]
            f1_matrix[class_n, i] = eval.f1_score_class(new_precisions[class_n], new_recalls[class_n])
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

    ### These are calculations done by using the combined confusion matrix.
    ### The preferred averages are by averaging the individual metrics from each tree iteration.
    # confusion_accuracy = np.trace(combined_confusion)/np.sum(combined_confusion)
    # print("\nAccuracy from confusion: " + str(confusion_accuracy)) # This is the same as average accuracy

    # combined_precisions, precision_avg = eval.precision(combined_confusion)
    # combined_recalls, recall_avg = eval.recall(combined_confusion)

    # print("Combined averaged precisions: ")
    # print(combined_precisions)

    # print("Combined average recalls: ")
    # print(combined_recalls)

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


if __name__ == "__main__":

    vis = TreeVis()
    while True:
        choice = input("""Options (enter number or own filepath):
                    1. Test algorithm on clean dataset
                    2. Test algorithm on noisy dataset
                    3. Visualise tree on entire clean dataset
                    4. Exit
                    (otherwise test algorithm on own dataset)
                    """)
        if choice in ('3', 'Visualise'):
            dataset = read_dataset('wifi_db/clean_dataset.txt')
            root, depth = decision_tree_learn(dataset)
            vis.draw(root)
        elif choice == '4':
            exit()
        else:
            if choice in ('1', 'Clean'):
                dataset_path = "wifi_db/clean_dataset.txt"
            elif choice in ('2', 'Noisy'):
                dataset_path = "wifi_db/noisy_dataset.txt"
            else:
                dataset_path = choice
            train_and_test(dataset_path)
        print('\n' * 2)