# Intro_to_ML_CW1

Overview of [test_all.py](test_all.py):
- `train_and_test(dataset_path="wifi_db/clean_dataset.txt")` Tests the algorithm on a dataset and prints the evaluation metrics: 
  - Total confusion matrix from summing all the matrices from each fold.
  - Average accuracy.
  - Average precision and recall per class (room).
  - Average F1 score per class. This is by calculating the F1 score per class for each fold, then finding the average across folds.
- Initialiser (`__name__==__main__`): Gives multiple options to the 

### [**Evaluation Folder**](evaluation/)
-----
This folder contains the functions related to evaluation metric and k_fold testing.

Functions in [eval.py](evaluation/eval.py):
- `classify(attributes, tree)` Finds the corresponding label of a list of attributes to a decision tree recursively. Knows when it's reached a leaf node by checking type.
- `evaluate(test_db, trained_tree)` Iterates through a dataset using `classify` for each row, while also updating the confusion matrix each iteration
- `precision` and `recall` Both use a confusion matrix and also return the macro precision and recall
- `f1_score_class(precision, recall)` Returns F1 score per class. Only this function for F1 score is used for efficiency so as to not calculate precision and recall metrics twice per class.

Functions in [k_folds.py](evaluation/k_folds.py)
- `indices_split_folds(fold_n, instances_n)` Shuffles indices and organises them into separate folds for each fold iteration.
- `test_k_folds(dataset, k=10)` Tests the algorithm on a dataset for k-fold cross validation. Builds a tree for each fold, tests, and returns list of accuracies and confusion matrices for each fold.