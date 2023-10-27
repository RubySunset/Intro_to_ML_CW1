import numpy as np
from evaluation.k_folds import test_k_folds
import evaluation.eval as eval


def main(dataset_path="wifi_db/clean_dataset.txt"):
    print("Decision trees trained on: " + dataset_path)
    dataset = read_dataset(dataset_path)

    # confusions is an array of confusion matrices. accuracies is also an array.
    accuracies, confusions  = test_k_folds(dataset) 

    accuracy_avg = np.mean(accuracies)
    print("Accuracy average: " + str(accuracy_avg))
    
    # Combined confusion matrix is the matrix obtained from the sum of all the confusion matrices
    combined_confusion = np.zeros((4, 4), dtype=np.int32)
    for matrix in confusions:
        combined_confusion += matrix
    
    print("\nSum Confusion: ")
    print(combined_confusion)
    print("")
    ### These are calculations done by using the combined confusion matrix.
    ### The preferred averages are by averaging the individual metrics from each tree iteration.
    # confusion_accuracy = np.trace(combined_confusion)/np.sum(combined_confusion)
    # print("\nAccuracy from confusion: " + str(confusion_accuracy)) # This is the same as average accuracy

    # combined_precisions, precision_avg = eval.precision(combined_confusion)
    # combined_recalls, recall_avg = eval.recall(combined_confusion)

    # print("Micro-averaged precision: ")
    # print(combined_precisions) # micro-averaged

    # print("Micro_averaged recall: ")
    # print(combined_recalls)

    # Precisions for each of the 4 classes
    precisions_matrix = np.zeros((4, len(confusions)))
    recalls_matrix = np.zeros((4, len(confusions)))
    # f1_matrix = np.zeros((4, len(confusions)))
    f1_matrix = np.zeros((4, len(confusions)))

    for i, matrix in enumerate(confusions):
        new_precisions = eval.precision(matrix)[0]
        new_recalls = eval.recall(matrix)[0]
        # new_f1 = eval.f1_score_confusion(matrix)[0] ### MAKE this more efficient by not calculating precision and recall twice.
        for class_n in range(len(confusions[0])):
            precisions_matrix[class_n, i] = new_precisions[class_n] # Precision of class class_n+1 becomes a part of precisions matrix.
            recalls_matrix[class_n, i] = new_recalls[class_n]
            # f1_matrix[class_n, i] = new_f1[class_n]
            f1_matrix[class_n, i] = eval.f1_score_class(new_precisions[class_n], new_recalls[class_n])

    
    precision_matrix_avg = []
    recall_matrix_avg = []
    # f1_matrix_avg = []
    f1_matrix_avg = []
    for class_n_p, class_n_r, class_n_ef in zip(precisions_matrix, recalls_matrix, f1_matrix):
        precision_matrix_avg.append(np.mean(class_n_p))
        recall_matrix_avg.append(np.mean(class_n_r))
        # f1_matrix_avg.append(np.mean(class_n_f))
        f1_matrix_avg.append(np.mean(class_n_ef))
        

    print("Averaged precision per class: ")
    print(precision_matrix_avg)
    print("Averaged recall per class: ")
    print(recall_matrix_avg)
    print("Averaged F1 score per class: ")
    # print(f1_matrix_avg)
    print(f1_matrix_avg)

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

    choice = input("""Which dataset do you want to test the algorithm on?
                        Options:
                        - Clean
                        - Noisy
                        - [insert own filepath]
                        """)
    
    if choice == "Clean" or choice == "1":
        dataset_path = "wifi_db/clean_dataset.txt"
    elif choice == "Noisy" or choice =="2":
        dataset_path = "wifi_db/noisy_dataset.txt"
    else:
        dataset_path = choice

    main(dataset_path)